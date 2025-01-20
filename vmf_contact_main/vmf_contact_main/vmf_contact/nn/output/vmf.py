import math
from typing import List

import torch

from ._base import (
    ConjugatePrior,
    Likelihood,
    Posterior,
    PosteriorPredictive,
    PosteriorUpdate,
)

class VMFPrior(ConjugatePrior):
    """
    von Mises-Fisher distribution as the conjugate prior for the vMF likelihood.
    """

    def __init__(self, mean_direction, concentration: torch.Tensor):
        """
        Args:
            mean_direction: The expected mean direction of the outputs.
            concentration: The expected concentration of the outputs.
        """
        mean_direction = torch.nn.functional.normalize(mean_direction, dim=-1)
        super().__init__(mean_direction, torch.tensor(concentration))

    def update(self, update: PosteriorUpdate) -> Posterior:
        update_device = update.sufficient_statistics.device  # device of the update
        mu_prior = self.sufficient_statistics.to(update_device)

        mu_likelihood = update.sufficient_statistics[..., :-1]  # mean direction
        mu_likelihood = torch.nn.functional.normalize(
            mu_likelihood, dim=-1
        )  # normalize mu_prior and mu_likelihood
        # normalize mu_likelihood
        kappa_likelihood = update.sufficient_statistics[..., -1].unsqueeze(
            -1
        )  # concentration
        evidence = update.log_evidence.exp().unsqueeze(-1)  # pseudo count
        evidence = torch.clamp(evidence, min=0, max=100)

        # Initialize Normal Gamma parameters
        kappa_post = self.evidence + evidence
        mu_post = (mu_likelihood * evidence + mu_prior * self.evidence) / kappa_post
        # mu post is already normalized because it is a weighted average of normalized mu_likelihood and mu_prior

        return VMFPosterior(mu_post, kappa_post, kappa_likelihood)


class VMFPosterior(Posterior):
    """
    von Mises-Fisher distribution as the posterior for the vMF likelihood.
    suppose kappa_likelihood is fixed for all samples
    """

    def __init__(
        self,
        mu_post: torch.Tensor,
        kappa_post: torch.Tensor,
        kappa_likelihood: torch.Tensor,
    ):
        self.mu_post = mu_post
        self.kappa_post = kappa_post
        self.kappa_likelihood = kappa_likelihood

    def expected_log_likelihood(self, data: torch.Tensor) -> torch.Tensor:
        # Index the posterior mean and concentration to match the ground truth data
        mu_post = self.mu_post
        kappa_post = self.kappa_post
        kappa_likelihood = self.kappa_likelihood

        # Compute the expected log-likelihood of the data under the posterior distribution
        # project data onto the posterior mean direction
        data_proj = torch.sum(mu_post * data, dim=-1, keepdim=True)

        expected_log_likelihood = log_normalizer_vmf(kappa_likelihood) \
            + kappa_likelihood * data_proj * \
                (1 / (torch.tanh(kappa_post) + 1e-7) - 1 / (kappa_post + 1e-7)
        )

        return expected_log_likelihood.squeeze(-1)

    def log_expected_likelihood(
        self, data: torch.Tensor, batch_ind=None, pred_ind=None
    ) -> torch.Tensor:
        # Index the posterior mean and concentration to match the ground truth data
        mu_post = self.mu_post
        kappa_post = self.kappa_post
        kappa_likelihood = self.kappa_likelihood

        if pred_ind is not None:
            mu_post = self.mu_post[pred_ind, batch_ind]
            kappa_post = self.kappa_post[pred_ind, batch_ind]
            kappa_likelihood = self.kappa_likelihood[pred_ind, batch_ind]
        else:
            mu_post = mu_post[:, batch_ind]
            kappa_post = self.kappa_post[:, batch_ind]
            kappa_likelihood = self.kappa_likelihood[:, batch_ind]

        theta = kappa_likelihood * data + kappa_post * mu_post
        theta_norm = torch.norm(theta, dim=-1, keepdim=True)
        # Calculate the log likelihood, here kappa_likelihood is fixed for all samples
        log_likelihoods = (
            log_normalizer_vmf(kappa_post)
            + log_normalizer_vmf(kappa_likelihood)
            - log_normalizer_vmf(theta_norm)
        )
        # Calculate the expected log likelihood over the MC samples
        return log_likelihoods

    def entropy(self) -> torch.Tensor:
        # Entropy of the posterior distribution
        kappa_post = self.kappa_post
        # Calculate the entropy of the posterior distribution
        return entropy_vmf(kappa_post)

    def maximum_a_posteriori(self) -> Likelihood:
        # Return the MAP estimate of the posterior distribution
        return VMFLikelihood(self.mu_post, self.kappa_likelihood)

    def posterior_predictive(self) -> PosteriorPredictive:
        raise NotImplementedError(
            "Posterior predictive not implemented for VMFPosterior."
        )


class VMFLikelihood(Likelihood):
    """
    von Mises-Fisher distribution for modeling directional data.
    """

    def __init__(self, mu: torch.Tensor, kappa: torch.Tensor):
        self.mu = mu
        self.kappa = kappa

    def mean(self) -> torch.Tensor:
        return self.mu

    def uncertainty(self) -> torch.Tensor:
        # aleatoric uncertainty as the entropy of the likelihood distribution
        return entropy_vmf(self.kappa)

    def expected_sufficient_statistics(self) -> torch.Tensor:
        # suppose kappa is fixed for all samples, so the expected sufficient statistics is the mean
        return torch.cat([self.mu, self.kappa], dim=-1)

    def negative_log_likelihood(
        self, data: torch.Tensor, reduction="mean"
    ) -> torch.Tensor:
        # Calculate the log likelihood of the data
        nll = -log_prob_vmf(data, self.mu, self.kappa)
        if reduction == "mean":
            return nll.mean()
        elif reduction == "sum":
            return nll.sum()
        else:
            return nll


class VMFOutput(torch.nn.Module):
    """
    von Mises-Fisher output with a VMF prior. The prior yields a mean direction of [1, 0] and a concentration of 1.
    """

    def __init__(self, device, prior=None):
        """
        Args:
            dim: The dimension of the latent space.
        """
        super().__init__()
        self.prior = VMFPrior(
            mean_direction=torch.tensor([0.0, 0.0, 1.0] if prior is None else prior, device=device), concentration=1.0
        )

    def forward(self, x: torch.Tensor, kappa_0=25) -> Likelihood:
        assert x.size(-1) == 4, "The last dimension of the input tensor must be 4."
        loc, kappa = x[..., :-1], x[..., -1]
        # normalize the mean direction
        loc = torch.nn.functional.normalize(loc, dim=-1)
        # scale the concentration
        kappa = torch.exp(kappa)
        kappa = torch.clamp(kappa, min=0, max=4 * kappa_0)[..., None]

        return VMFLikelihood(loc, kappa)


def chunk_squeeze_last(x: torch.Tensor) -> List[torch.Tensor]:
    """
    Splits the provided tensor into individual elements along the last dimension and returns the
    items with the last dimension squeezed.

    Args:
        x: The tensor to chunk.

    Returns:
        The squeezed chunks.
    """
    chunks = x.chunk(x.size(-1), dim=-1)
    return [c.squeeze(-1) for c in chunks]


def log_normalizer_vmf(kappa):
    # For large kappa, this is numerically stable
    return (
        torch.log(kappa)
        - math.log(2 * math.pi)
        - kappa
        - torch.log1p(-torch.exp(-2 * kappa) + 1e-7)
    )


def entropy_vmf(kappa):
    entropy = -log_normalizer_vmf(kappa) - kappa / (1e-7 + torch.tanh(kappa)) + 1
    return entropy.squeeze(-1)


def log_prob_vmf(x, mu, kappa):
    return kappa * torch.sum(mu * x, dim=-1, keepdim=True) + log_normalizer_vmf(kappa)
