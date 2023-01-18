"""Sparse variational auto-encoder model."""
import enum
import math
from typing import Callable, Tuple
from jax import grad
import time

import torch
from torch import distributions, nn
from torch.nn import functional as F

import sys
sys.path.append('/home/vg0233/PillowLab/SVAE')
from svae import horseshoe

from scipy.stats import cauchy
from sklearn.linear_model import Ridge, Lasso
from sklearn.base import BaseEstimator
import sklearn
import numpy as np
from scipy.optimize import minimize, root, root_scalar

class CauchyRegressor(BaseEstimator):
    def __init__(self, 
            prior_scale: float = 1.0, 
            likelihood_scale: float = np.exp(-2.0),
            tol: float = 0.0001,
            ) -> None:
        self.prior_scale = prior_scale
        self.likelihood_scale = likelihood_scale
        self.tol = tol
        self.prior = cauchy(loc=0.0, scale=prior_scale) # assumes standard Cauchy(0,1) prior

        # Relevant constants
        self.alpha = 2 * (likelihood_scale**2)


    def fit(self, Phi, x, method=None, optimization_step=0,):
        '''Fit a Linear Model with Cauchy prior.

        Phi: ndarray, of shape `(obs_dim, latent_dim)`
            Row matrix of features 
        x: ndarray, of shape `(obs_dim, )`
            Image
        '''
        _, M = Phi.shape

        def optimization_objective_f(z):
            '''MAP Optimization objective
            -|| x - Phi @ z ||_2^2 + alpha * log P(z)
            '''
            reconstruction_loss = (1/(2 * (self.likelihood_scale**2))) * np.sum(np.square(x - Phi @ z))
            regularization = np.sum(np.log(self.prior_scale**2 + np.square(z)))
            return reconstruction_loss + regularization

        def gradient_f(z):
            '''Gradient vector of `optimization_objective`'''
            N, = z.shape
            grad_ls = Phi.T @ (Phi @ z - x)
            grad_prior = 2 * np.divide(z, self.prior_scale**2 + np.square(z))
            grad = (1/self.likelihood_scale**2) * grad_ls + grad_prior
            assert grad.shape == (N,), grad.shape

            return grad
        
        def hessian_f(z):
            '''Hessian matrix  of `optimization_objective`'''
            N, = z.shape
            t1 = (1/self.likelihood_scale**2) * Phi.T @ Phi
            t2_diag = 2*np.divide(self.prior_scale**2 - np.square(z), np.square(self.prior_scale**2 + np.square(z)))
            t2 = np.diag(t2_diag)
            assert t2.shape == (N, N, )
            
            hess = t1 - t2
            assert hess.shape == (N, N, )
            return hess

        if method is None:
            coef0 = self.prior.rvs(size=M)

            # Use scipy's default
            res = minimize(optimization_objective_f, x0=coef0, tol=self.tol)
            self.coef_ = res.x
        
        elif method=='CG':
            coef0 = self.prior.rvs(size=M)
            # coef0 = np.zeros(shape=(M,))

            res = minimize(optimization_objective_f, x0=coef0, method='CG', jac=gradient_f, tol=self.tol)
            assert np.linalg.norm(res.x) != 0.
            self.coef_ = res.x

        elif method == 'manual_CGM':
            def roots_analytical(z, d):
                #! outdated, false
                ones = np.ones_like(d)
                A = Phi.T @ Phi
                sigma = self.prior_scale
                N = len(d)

                c_1 = d.T @ A @ d
                c_2 = z.T @ A @ d - sigma * d.T @ A @ ones - d.T @ A @ z - 2*x.T @ Phi @ d
                c_3 = sigma * z.T @ A @ ones + z.T @ A @ z - 2*sigma* x.T @ Phi @ z + 2*N
                return np.roots([c_1, c_2, c_3])

            def roots_empirical(z, d):
                func = lambda a: np.dot(gradient_f(z+a*d), d)
                # roots = root(func, x0=1., method='lm')
                try:
                    roots = root_scalar(func, bracket=[-1.0, 1.0])
                except ValueError:
                    '''bracket ends don't have different signs.'''
                    roots = root_scalar(func, x0=1e-02, x1=1e-03)
                return roots

            def roots_hessian(z, d):
                numerator = - np.dot(gradient_f(z), d)
                denominator = d.T @ hessian_f(z) @ d
                return numerator/denominator

            f = lambda z: optimization_objective_f(z) # shorten notation

            n_steps = 10
            # z_0 = self.prior.rvs(size=M)
            # z_0 = np.zeros(shape=(M,)) #+ np.random.randn()
            z_0 = Phi.T @ x
            z_i = z_0
            d_i = -gradient_f(z_i)
            r_i = d_i
            # print('')
            for i in range(n_steps):
                start = time.time()

                # alpha_i = roots_analytical(z_i, d_i)[0]

                result_empirical = roots_empirical(z_i, d_i) # approx 0.0006s # 0.00036
                # try:
                #     alpha_i = result_empirical.x[0]
                # except AttributeError:
                if not result_empirical.converged:
                    # alpha_i unchanged.
                    print('Warning: root finding did not converge.')
                else:
                    alpha_i = result_empirical.root

                # alpha_i = roots_hessian(z_i, d_i)  # approx 0.006s

                z_next = z_i + alpha_i * d_i
                r_next = -gradient_f(z_next)
                
                # beta_i = np.dot(r_next, r_next) / np.dot(r_i, r_i) # Fletcher–Reeves
                beta_PR = np.dot(r_next, r_next - r_i) / np.dot(r_i, r_i) # Polak–Ribière
                beta_i = max(beta_PR, 0)

                d_i = r_next + beta_i * d_i
                assert np.linalg.norm(d_i) > 0.

                # Update states
                z_previous = z_i
                z_i = z_next
                r_i = r_next

                # Check convergence and break early if possible
                if np.linalg.norm(r_i) < 0.001:
                    break
                elif np.abs((f(z_i)-f(z_previous))/f(z_previous)) < 0.01:
                    break

                # if i%50 == 0:
                # print('[{:}] f(z_i): {:4.4f}, r_i norm: {:4.4f}, alpha: {:1.2e}, step time: {:.5f}'.format(
                #      i, f(z_next), np.linalg.norm(r_next), alpha_i, time.time()-start
                #     ))
                # print(np.linalg.norm(z_next-z_i,2), np.linalg.norm(z_next-z_i,1))

            # print(np.linalg.norm(z_0 - z_i, 1))
            assert np.linalg.norm(z_i) != 0.
            self.coef_ = z_i

        elif method=='ODE':
            from scipy.integrate import solve_ivp

            def fun(z):
                N, = z.shape
                grad_ls = Phi.T @ (Phi @ z - x)
                grad_prior = 2 * np.divide(z, self.prior_scale**2 + np.square(z))
                out = grad_prior - (1/self.likelihood_scale**2) * grad_ls
                assert out.shape == (N,), out.shape

                return out

            # z_0 = self.prior.rvs(size=M)
            z_0 = np.zeros(shape=(M,)) #+ np.random.randn()
            # if optimization_step < 5: # early training, take longer trajectories
            #     T = 5.0
            # else:
            #     T = 0.5
            T = 100.0

            odesolver_result = solve_ivp(
                fun= lambda t,z : fun(z), 
                t_span=(0,T),
                y0=z_0,
                jac= lambda t,z : hessian_f(z),
            )
            # print(odesolver_result)

            assert odesolver_result.success
            # print(list(np.linalg.norm(odesolver_result.y, axis=0)[::100]))
            convergence_error = np.linalg.norm(odesolver_result.y[:,-1]-odesolver_result.y[:,-10])
            convergence_error_ratio = convergence_error/np.linalg.norm(odesolver_result.y[:,-1])
            if convergence_error_ratio > 0.01:
                '''we allow 1% error in convergence.'''
                print("WARNING: z coefficients trajectory did not converge. " + \
                        f"Norm difference: {convergence_error:2.4f}, {100*convergence_error_ratio:2.4f}%")

            self.coef_ = odesolver_result.y[:,-1]

        else:
            raise NotImplementedError


        
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

    def _set_regressor(self, tolerance) -> None:
        prior = Prior(self.prior.item())
        if prior== prior.GAUSSIAN:
            alpha = 2*(self.likelihood_scale**2) # Note: pi/4 = argmin_{scale}( KL(Normal(0, scale) || Laplace(0,1)) )
            self.reg = Ridge(alpha=alpha.item(), tol=tolerance)    # L2 penalty, or Ridge.
        
        elif prior==prior.CAUCHY:
            alpha = 2*(self.likelihood_scale**2) # Note: 0.1 = argmin_{scale}( KL(Cauchy(0, scale) || Laplace(0,1)) )
            self.reg = CauchyRegressor(alpha=alpha.item(), tol=tolerance)
        else:
            if prior != prior.LAPLACE:
                print("Unrecognized prior. Reverting to LAPLACE")
            
            # likelihood_scale = torch.sqrt(torch.tensor(dict_coefs_lambda / 2))
            lasso_lambda = 2*(self.likelihood_scale**2)
            alpha = lasso_lambda/(2 * 144)
            self.reg = Lasso(alpha=alpha.item(), tol=tolerance)    # L1 penalty, or LASSO. Fit model with coordinate descent

    def _encode(self, x):
        B, _ = x.shape
        S = torch.zeros(B, self.latent_dim).to(x.device)
        local_reg = sklearn.base.clone(self.reg)
        for i in range(B):
            local_reg.fit(self.Phi.weight.detach().cpu().numpy(), x[i, :].cpu().numpy())
            S[i] = torch.tensor(local_reg.coef_)
        assert torch.linalg.norm(S) != 0., 'Dictionary elements not updated. Optimization did not converge.'
        return S