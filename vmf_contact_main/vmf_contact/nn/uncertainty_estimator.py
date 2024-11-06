from typing import Tuple

import normflows as nf
import numpy as np
import torch
import torch.nn as nn
from normflows.distributions.base import BaseDistribution, GaussianMixture

from .output import PosteriorUpdate, VMFOutput
from .scaler import EvidenceScaler


class UncertaintyEstimator(nn.Module):

    def __init__(
        self,
        hidden_features,
        embedding_dim,
        flow_layers,
        certainty_budget,
        hidden_layers=6,
        gmm_components=50,
        logit=False,
        prob_baseline=None,
        flow_type="resflow",
    ):

        super().__init__()

        if flow_type == "resflow":
            flows = [nf.transforms.Logit()] if logit else []
            for _ in range(flow_layers):
                # Neural network with two hidden layers having 64 units each
                # Last layer is initialized by zeros making training more stable
                nnet = nf.nets.LipschitzMLP(
                    [embedding_dim]
                    + [hidden_features] * (hidden_layers - 1)
                    + [embedding_dim],
                    init_zeros=True,
                    lipschitz_const=0.98,
                    max_lipschitz_iter=None,
                    lipschitz_tolerance=1e-3,
                )
                flows.append(nf.flows.Residual(nnet, n_dist="poisson"))
                flows.append(nf.flows.ActNorm(embedding_dim))
            base_dist = GaussianMixture(gmm_components, dim=embedding_dim, trainable=True)
            self.flow = nf.NormalizingFlow(base_dist, flows)
        elif flow_type == "glow":
            L=3
            q0 = []
            merges = []
            flows = []
            for i in range(L):
                flows_ = []
                for j in range(flow_layers):
                    flows_ += [nf.flows.GlowBlock(embedding_dim, hidden_features,
                                                split_mode='channel', scale=True)]
                flows_ += [nf.flows.Squeeze()]
                flows += [flows_]
                if i > 0:
                    merges += [nf.flows.Merge()]
                latent_shape = embedding_dim
                q0 += [GaussianMixture(gmm_components, dim=latent_shape, trainable=True)]
            self.flow = nf.NormalizingFlow(base_dist, flows, merges)


        # If the target density is not given
        self.prob_baseline = prob_baseline
        #if prob_baseline == "post":
        self.flow = nf.NormalizingFlow(base_dist, flows)
        self.embedding_dim = embedding_dim
        self.scaler = EvidenceScaler(embedding_dim, certainty_budget)

    def posterior_update(
        self, features, baseline, prior = None
    ) -> Tuple[PosteriorUpdate, torch.Tensor]:
        """
        Computes the posterior update over the target distribution for each input independently.

        Args:
            x: The inputs that are first passed to the encoder.

        Returns:
            The posterior update for every input and the true log-probabilities.
        """

        # Compute log_prob, which is the epistemic uncertainty
        self.vmfoutput = VMFOutput(device=features.device, prior=prior)
        bs = features.shape[0] if len(features.shape) > 2 else 1
        log_prob = self.flow.log_prob(features.view(-1, self.embedding_dim))
        log_prob = log_prob.view(bs, -1) if bs > 1 else log_prob
        log_evidence = self.scaler.forward(log_prob)

        prediction_likelihood = self.vmfoutput.forward(baseline)
        sufficient_statistics = prediction_likelihood.expected_sufficient_statistics()

        update = PosteriorUpdate(sufficient_statistics, log_evidence)
        return self.vmfoutput.prior.update(update)

    def likelihood_update(self, baseline) -> Tuple[PosteriorUpdate, torch.Tensor]:
        self.vmfoutput = VMFOutput(device=baseline.device)
        return self.vmfoutput.forward(baseline)

    def log_prob(self, features) -> torch.Tensor:
        return self.flow.log_prob(features)

    def update_lipschitz(self):
        nf.utils.update_lipschitz(self.flow, 50)

    def forward_kld(self, feat):
        return self.flow.forward_kld(feat)


class GaussianMixtureFull(BaseDistribution):
    """
    Mixture of Gaussians with full covariance matrices using log of Cholesky factors of precision matrices
    """

    def __init__(
        self,
        n_modes,
        dim,
        loc=None,
        precisions_cholesky=None,
        weights=None,
        trainable=True,
    ):
        """Constructor

        Args:
          n_modes: Number of modes of the mixture model
          dim: Number of dimensions of each Gaussian
          loc: List of mean values
          precisions_cholesky: List of Cholesky factors of precision matrices
          weights: List of mode probabilities
          trainable: Flag, if true parameters will be optimized during training
        """
        super().__init__()

        self.n_modes = n_modes
        self.dim = dim

        if loc is None:
            loc = np.random.randn(self.n_modes, self.dim)
        if precisions_cholesky is None:
            precisions_cholesky = np.array(
                [np.linalg.cholesky(np.eye(self.dim)) for _ in range(self.n_modes)]
            )
        if weights is None:
            weights = np.ones(self.n_modes)
        weights /= weights.sum()  # Normalize the weights

        loc = np.array(loc)
        precisions_cholesky = np.array(precisions_cholesky)
        weights = np.array(weights)

        if trainable:
            self.loc = nn.Parameter(torch.tensor(loc, dtype=torch.float32))
            self.precisions_cholesky = nn.Parameter(
                torch.tensor(precisions_cholesky, dtype=torch.float32)
            )
            self.weights = nn.Parameter(torch.tensor(weights, dtype=torch.float32))
        else:
            self.register_buffer("loc", torch.tensor(loc, dtype=torch.float32))
            self.register_buffer(
                "precisions_cholesky",
                torch.tensor(precisions_cholesky, dtype=torch.float32),
            )
            self.register_buffer("weights", torch.tensor(weights, dtype=torch.float32))

    def forward(self, num_samples=1):

        # Sample mode indices
        mode = torch.multinomial(self.weights, num_samples, replacement=True)
        mode_1h = nn.functional.one_hot(mode, self.n_modes).float()

        # Get the selected means and precisions Cholesky factors
        mean = torch.sum(self.loc * mode_1h[..., None], dim=1)
        precisions_cholesky_selected = torch.sum(
            self.precisions_cholesky * mode_1h[..., None, None], dim=1
        )

        # Sample from the selected Gaussian modes
        eps = torch.randn(num_samples, self.dim, dtype=mean.dtype, device=mean.device)
        z = mean + torch.bmm(
            torch.inverse(precisions_cholesky_selected), eps.unsqueeze(-1)
        ).squeeze(-1)

        # Compute log probability
        log_p = self.log_prob(z)

        return z, log_p

    def log_prob(self, z):

        # Expand z for broadcasting
        z_exp = z.unsqueeze(1)  # Shape: (num_samples, 1, dim)

        # Calculate diff, exp_term, and log_det_cov for all modes simultaneously
        diff = z_exp - self.loc  # Shape: (num_samples, n_modes, dim)
        precisions_cholesky = self.precisions_cholesky  # Shape: (n_modes, dim, dim)

        # Compute the precision matrix from the Cholesky factor
        precisions = torch.bmm(
            precisions_cholesky, precisions_cholesky.transpose(-2, -1)
        )  # Shape: (n_modes, dim, dim)

        # Compute the quadratic term in the exponent
        exp_term = -0.5 * torch.sum(
            diff.unsqueeze(-2) * precisions.unsqueeze(0) @ diff.unsqueeze(-1),
            dim=(-2, -1),
        )  # Shape: (num_samples, n_modes)
        log_det_precision = 2.0 * torch.sum(
            torch.log(torch.diagonal(precisions_cholesky, dim1=-2, dim2=-1)), dim=-1
        )  # Shape: (n_modes,)

        # Calculate log probabilities for each mode
        log_p = (
            exp_term
            + log_det_precision
            - 0.5 * self.dim * np.log(2 * np.pi)
            + torch.log(self.weights)
        )

        # Combine log probabilities across modes
        log_p = torch.logsumexp(log_p, dim=1)

        return log_p


if __name__ == "__main__":
    device = "cuda:0"
    model = UncertaintyEstimator(384, 384, 3, 0.1).to(device)
    print(model)
    x = torch.randn(1000, 384).to(device)
    kl = model.forward_kld(x)
    kl.backward()
    print(kl)
