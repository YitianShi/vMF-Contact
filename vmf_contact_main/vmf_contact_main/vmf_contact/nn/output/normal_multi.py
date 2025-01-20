import math
import scipy.stats as st  # type: ignore
import torch
from torch.distributions import MultivariateNormal, Wishart
from ._base import ConjugatePrior, Likelihood, Posterior, PosteriorPredictive, PosteriorUpdate, Output
from abc import ABC, abstractmethod
from typing import List, Tuple

class MultivariateStudentT(PosteriorPredictive):
    """
    Multivariate Student's t-distribution as the posterior predictive of the Multivariate Normal likelihood.
    """

    def __init__(self, df: torch.Tensor, loc: torch.Tensor, scale_matrix: torch.Tensor):
        self.df = df
        self.loc = loc
        self.scale_matrix = scale_matrix

    def mean(self) -> torch.Tensor:
        return self.loc

    def uncertainty(self) -> torch.Tensor:
        d = self.scale_matrix.shape[-1]
        lbeta = torch.lgamma(0.5 * self.df) + 0.5 * d * (math.log(math.pi) - torch.log(self.df))
        lbeta -= torch.lgamma(0.5 * (self.df + d))
        scale_logdet = torch.logdet(self.scale_matrix)
        t1 = 0.5 * d * (self.df.log() + 1) + lbeta - 0.5 * scale_logdet
        t2 = (0.5 * (self.df + d) * torch.digamma(0.5 * (self.df + d)) - 0.5 * self.df * torch.digamma(0.5 * self.df))
        return t1 + t2

    def symmetric_confidence_level(self, data: torch.Tensor) -> torch.Tensor:
        assert (
            not torch.is_grad_enabled()
        ), "The confidence level cannot currently track the gradient."

        # Calculate Mahalanobis distance
        mahalanobis_dist = torch.sqrt((data - self.loc).unsqueeze(-2) @ torch.inverse(self.scale_matrix) @ (data - self.loc).unsqueeze(-1)).squeeze(-1)

        # Get probabilities from the cumulative distribution function
        probs_numpy = st.t.cdf(mahalanobis_dist.cpu().numpy(), df=self.df.cpu().numpy())
        probs = torch.from_numpy(probs_numpy).to(data.device)
        return 2 * (probs - 0.5).abs()


class NormalInverseWishartPrior(ConjugatePrior):
    """
    Normal-Inverse-Wishart distribution as the conjugate prior of the Multivariate Normal likelihood.
    """

    def __init__(self, mean: torch.Tensor, scale: torch.Tensor, df: float, evidence: float):
        """
        Args:
            mean: The expected mean of the outputs.
            scale: The scale matrix (covariance) of the outputs.
            df: Degrees of freedom for the scale matrix.
            evidence: The certainty for the expectation on mean and scale.
        """
        super().__init__(torch.stack([mean, scale]), torch.tensor(evidence))
        self.df = df

    def update(self, update: PosteriorUpdate) -> Posterior:
        prior_mean, prior_scale = self.sufficient_statistics
        mean, scale = update.sufficient_statistics

        prior_evidence = self.evidence
        evidence = update.log_evidence.exp()

        posterior_evidence = evidence + prior_evidence
        posterior_mean = (mean * evidence + prior_mean * prior_evidence) / posterior_evidence
        posterior_scale = prior_scale + scale + (evidence * prior_evidence / posterior_evidence) * (mean - prior_mean).unsqueeze(-1) @ (mean - prior_mean).unsqueeze(-2)

        return NormalInverseWishart(posterior_mean, posterior_scale, self.df + evidence)


class NormalInverseWishart(Posterior):
    """
    Normal-Inverse-Wishart distribution as the posterior of the Multivariate Normal likelihood.
    """

    def __init__(self, mean: torch.Tensor, scale: torch.Tensor, df: float):
        self.mean = mean
        self.scale = scale
        self.df = df

    def expected_log_likelihood(self, data: torch.Tensor) -> torch.Tensor:
        d = data.size(-1)
        diff = data - self.mean
        inv_scale = torch.inverse(self.scale)
        log_likelihood = -0.5 * (d * torch.log(2 * math.pi) + torch.logdet(self.scale) + (diff.unsqueeze(-2) @ inv_scale @ diff.unsqueeze(-1)).squeeze())
        return log_likelihood

    def entropy(self) -> torch.Tensor:
        d = self.scale.shape[-1]
        t1 = 0.5 * d * (1 + math.log(2 * math.pi)) + 0.5 * torch.logdet(self.scale)
        t2 = -0.5 * self.df * (torch.digamma(0.5 * (self.df - d + 1)) + d * math.log(2) - torch.digamma(0.5 * self.df))
        return t1 + t2

    def maximum_a_posteriori(self) -> Likelihood:
        return MultivariateNormal(self.mean, self.scale / (self.df - self.scale.size(-1) - 1))

    def posterior_predictive(self) -> PosteriorPredictive:
        scale_matrix = self.scale * (self.df + 1) / (self.df * (self.df - self.scale.size(-1) - 1))
        return MultivariateStudentT(self.df - self.scale.size(-1) + 1, self.mean, scale_matrix)


class MultivariateNormal(Likelihood):
    """
    Multivariate Normal distribution for modeling continuous data.
    """

    def __init__(self, loc: torch.Tensor, covariance_matrix: torch.Tensor):
        self.loc = loc
        self.covariance_matrix = covariance_matrix

    def mean(self) -> torch.Tensor:
        return self.loc

    def uncertainty(self) -> torch.Tensor:
        return 0.5 * torch.logdet(self.covariance_matrix)

    def expected_sufficient_statistics(self) -> torch.Tensor:
        return torch.stack([self.loc, self.covariance_matrix + torch.ger(self.loc, self.loc)], dim=-1)


class NormalMultiOutput(Output):
    """
    Multivariate Normal output with Normal-Inverse-Wishart prior. The prior yields a mean of 0 and a scale of 10.
    """

    def __init__(self, dim: int):
        """
        Args:
            dim: The dimension of the latent space.
        """
        super().__init__()
        self.linear = torch.nn.Linear(dim, 2 * dim)
        self.prior = NormalInverseWishartPrior(mean=torch.zeros(dim), scale=torch.eye(dim) * 10, df=dim, evidence=1)

    def forward(self, x: torch.Tensor) -> Likelihood:
        z = self.linear.forward(x)
        loc, log_scale_diag = chunk_squeeze_last(z)
        scale_matrix = torch.diag_embed(log_scale_diag.exp() + 1e-10)
        return MultivariateNormal(loc, scale_matrix)


def chunk_squeeze_last(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Splits the provided tensor into two chunks along the last dimension and returns them
    with the last dimension squeezed.

    Args:
        x: The tensor to chunk.

    Returns:
        The squeezed chunks.
    """
    half_size = x.size(-1) // 2
    loc = x[..., :half_size].squeeze(-1)
    log_scale_diag = x[..., half_size:].squeeze(-1)
    return loc, log_scale_diag
