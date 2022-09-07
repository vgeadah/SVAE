"""Sparse variational auto-encoder model."""
import enum
import math
from typing import Callable, Tuple

import torch
from torch import distributions, nn
from torch.nn import functional as F

from svae import horseshoe


@distributions.kl.register_kl(distributions.Normal, distributions.Laplace)
def kl_normal_laplace(
    p: distributions.Normal, q: distributions.Laplace
) -> torch.Tensor:
    """KL divergence between Normal and Laplace distributions."""
    t1 = q.loc - p.loc
    t2 = (t1 / q.scale) * torch.erf(t1 / (math.sqrt(2) * p.scale))
    t3 = torch.exp(
        torch.log(p.scale)
        - torch.log(q.scale)
        + math.log(math.sqrt(2 / math.pi))
        - ((t1 ** 2) / (2 * p.variance))
    )
    t4 = torch.log(2 * q.scale) - 0.5 * torch.log(2 * math.pi * p.variance) - 0.5
    return t2 + t3 + t4


RECONSTRUCTION_FN_T = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor
]
KL_FN_T = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor],
]
"""VAE loss function type aliases."""


class Prior(enum.Enum):
    """Prior distributions for the SVAE."""

    GAUSSIAN = enum.auto()
    LAPLACE = enum.auto()
    CAUCHY = enum.auto()
    HORSESHOE = enum.auto()


class SVAE(nn.Module):
    """Sparse variational autoencoder.

    Args:
        prior: The prior to use.
        likelihood_logscale: The logit scale for the likelihood.
        collapse_delta: The KL threshold above which we consider a latent dimension
            uncollapsed

    Both arguments just set buffers which we can query later to construct the loss
    function.

    """

    def __init__(
        self,
        prior: Prior = Prior.LAPLACE,
        prior_scale: float = 1.0,
        likelihood_logscale: float = -1.0,
        collapse_delta: float = 0.1,
    ):
        super().__init__()
        self.register_buffer("prior", torch.tensor(prior.value))
        self.register_buffer("prior_scale", torch.tensor(prior_scale))
        self.register_buffer("collapse_epsilon", torch.tensor(collapse_delta))
        self.register_buffer("likelihood_scale", torch.exp(torch.tensor(likelihood_logscale)))
        self.fc1 = nn.Linear(144, 128)
        self.fc21 = nn.Linear(128, 256)
        self.fc22 = nn.Linear(128, 512)
        self.fc31 = nn.Linear(256, 256)
        self.fc32 = nn.Linear(512, 512)
        self.latent_dim = 169
        self.fc41 = nn.Linear(256, self.latent_dim)
        self.fc42 = nn.Linear(512, self.latent_dim)
        self.Phi = nn.Linear(self.latent_dim, 144)

        self._normalized_Phi = False
        if self._normalized_Phi:
            self._init_decoder_weights()

        self._encoder = nn.ModuleList([self.fc1, self.fc21, self.fc22, self.fc31, self.fc32, self.fc41, self.fc42])
        self._decoder = nn.ModuleList([self.Phi])

    def _init_decoder_weights(self) -> None:
        '''
        Regularization of the decoder weights Phi to emulate the post-hoc methodoly 
        used by Olshausen and Field to match the variance of the latent code to the data variance.

        Action:
            First normalizes the weights Phi along the feature dimension, the use weight normalization
            (Salimans & Kingma, 2016) to only train the weight direction and keep weight norm constant.
        '''
        # Normalize decoder weights
        nn.init.normal_(self.Phi.weight)
        with torch.no_grad():
            self.Phi.weight = nn.Parameter(torch.mul(
                self.Phi.weight, 
                # (1/torch.norm(self.Phi.weight, 2, dim=1)).repeat(169,1).T,
                (1/torch.norm(self.Phi.weight, 2, dim=0)).repeat(144,1)
                ))

        # Detach decoder weight direction from norm
        self.Phi = nn.utils.weight_norm(self.Phi, dim=0)
        self.Phi.weight_g.requires_grad = False # remove norm from optim
        self._normalized_Phi = True
        return

    def _encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h1 = F.relu(self.fc1(x))
        h21 = F.relu(self.fc21(h1))
        h22 = F.relu(self.fc22(h1))
        h31 = F.relu(self.fc31(h21))
        h32 = F.relu(self.fc32(h22))
        return self.fc41(h31), self.fc42(h32)

    def _reparameterize(
            self, loc: torch.Tensor, logscale: torch.Tensor, family: str = 'GAUSSIAN'
        ) -> torch.Tensor:
        if family=='LAPLACE': # use Laplace(0,1) noise in reparametrization trick
            scale = torch.sigmoid(torch.exp(logscale))
            eps_dist = distributions.Laplace(0, 1)
            eps = eps_dist.rsample(scale.shape)
        else:
            scale = torch.sigmoid(torch.exp(logscale))
            eps = torch.randn_like(scale)
        return eps.mul(scale).add_(loc)  # type: ignore

    def _decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.Phi(z)

    def output_weights(self) -> torch.Tensor:
        """Return the output weight matrix as a CPU tensor."""
        if self._normalized_Phi:
            weight = self.Phi.weight_v
        else:
            weight = self.Phi.weight
        return weight.detach().clone().cpu()  # type: ignore

    def forward(  # type: ignore
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Auto-encodes an image.

        Returns:
            A tuple containing the decode image, along with the location
            and log scale of the encoded prior (for computing the KL loss, later).

        """
        z_loc, z_logscale = self._encode(x)                                 # Encoder
        z = self._reparameterize(z_loc, z_logscale, family='GAUSSIAN')      # z ~ q(z|x)
        x_loc = self._decode(z)                                        # Decoder
        # x_new = self._reparameterize(                                   # x ~ p(x|z)
        #     ll_loc, 
        #     self.likelihood_scale*torch.ones_like(ll_loc)
        #     )
        return x_loc, z_loc, z_logscale

    def loss_fns(self) -> Tuple[RECONSTRUCTION_FN_T, KL_FN_T]:
        """Construct the sprase VAE loss functions.

        Returns:
            Two loss functions. The first is the reconstruction loss (expected
            negative log likelihood). The second is the KL divergence penalty function.

        """
        prior_type = Prior(self.prior.item())  # type: ignore
        prior_loc = torch.tensor(
            0.0, device=self.likelihood_scale.device  # type: ignore
        )  # type: ignore
        prior_scale = self.prior_scale
        if prior_type == Prior.GAUSSIAN:
            prior_dist: distributions.Distribution = distributions.Normal(
                prior_loc, prior_scale
            )
        elif prior_type == Prior.LAPLACE:
            prior_dist = distributions.Laplace(prior_loc, prior_scale)
        elif prior_type == Prior.CAUCHY:
            prior_dist = distributions.Cauchy(prior_loc, prior_scale)
        elif prior_type == Prior.HORSESHOE:
            prior_dist = horseshoe.Horseshoe(prior_loc, prior_scale)
        else:
            raise RuntimeError("Unknown prior type {0}".format(prior_type))

        def reconstruction_loss(
            X: torch.Tensor,
            X_pred: torch.Tensor,
            loc: torch.Tensor,
            logscale: torch.Tensor,
        ) -> torch.Tensor:
            """Calculate the mean reconstruction loss for the batch.

            Args:
                x: the input images, of shape `(b, d)`.
                x_pred: the reconstructed images, of shape `(b, d)`.
                loc: the mean of the latent code, of shape `(b, k)`. (not used).
                logscale: the logit standard deviation of the latent code,
                    of shape `(b, k)` (not used).

            Returns:
                The mean expected log probability under the variational posterior, a
                scalar, approximated with 1 Monte Carlo sample.

            """
            likelihood = distributions.Independent(
                distributions.Normal(X_pred, self.likelihood_scale),  # type: ignore
                reinterpreted_batch_ndims=1,
            )
            return -likelihood.log_prob(X).mean()

        def kl_penalty(
            X: torch.Tensor,
            X_pred: torch.Tensor,
            loc: torch.Tensor,
            logscale: torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """Calculate the KL divergence penalty for the batch.

            Args:
                x: the input images, of shape `(b, d)`.
                x_pred: the reconstructed images, of shape `(b, d)`.
                loc: the mean of the latent code, of shape `(b, k)`.
                logscale: the logit standard deviation of the latent code, of shape
                    `(b, k)`.

            Returns:
                The mean KL divergence penalty for the batch, and a `(k, )` tensor
                containing the mean fraction of non-white-noise latent codes in the
                batch.

            """
            variational_posterior = distributions.Normal(loc, torch.exp(logscale))
            prior = prior_dist.expand(loc.shape)

            analytic_kl = prior_type in (Prior.GAUSSIAN, Prior.LAPLACE)
            if analytic_kl:
                kl_divergences = distributions.kl_divergence(
                    variational_posterior, prior
                )
            elif prior_type in (Prior.CAUCHY, Prior.HORSESHOE):
                n_kl_samples = 100
                posterior_samples = variational_posterior.rsample((n_kl_samples,))
                kl_divergences = (
                    variational_posterior.log_prob(posterior_samples)
                    - prior.log_prob(posterior_samples)
                ).mean(0)
            else:
                raise RuntimeError(f"Can't compute KL for prior {prior_type}")
            penalty = kl_divergences.sum(axis=1).mean()
            sparsity_fraction = (
                (kl_divergences > self.collapse_epsilon).float().mean(dim=0).mean()
            )
            return penalty, sparsity_fraction

        return reconstruction_loss, kl_penalty 

class Sparsenet(nn.Module):
    '''
    Sparse coding model
        z ~ P(z),
        x = Φz + ϵ
    for image patches x, decoded from features Φ ( x ) with z coefficients 
    from sparse prior and noise ϵ ~ N(0, σ_ϵ^2). 
    '''
    def __init__(
        self,
        prior: Prior = Prior.LAPLACE,
        prior_scale: float = 1.0,
        likelihood_logscale: float = -1.0,
    ):
        super().__init__()
        self.register_buffer("prior", torch.tensor(prior.value))
        self.register_buffer("prior_scale", torch.tensor(prior_scale))
        self.register_buffer("likelihood_scale", torch.exp(torch.tensor(likelihood_logscale)))

        self.latent_dim = 169
        self.Phi = nn.Linear(self.latent_dim, 144, bias=False)
    
    def _decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.Phi(z)
    
    def _set_Phi(self, pretrained_Phi: torch.Tensor) -> None:
        self.Phi.weight = nn.Parameter(pretrained_Phi)
        return