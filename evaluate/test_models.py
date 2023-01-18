"""Test models.py."""
import torch
import torch.distributions as dist


def kl_normal_laplace_mc(
    p: dist.Distribution, q: dist.Distribution, num_samples: int = 1024
) -> torch.Tensor:
    """Compute MC estimates of KL divergence between two distributions.

    Args:
        p: The first distribution.
        q: The second distribution.
        num_samples: The number of samples to take.

    Returns:
        A tensor of shape (num_samples, ) containing the running history of the MC
        estimates.

    """
    samples = p.sample((num_samples,))
    estimates = p.log_prob(samples) - q.log_prob(samples)
    return torch.cumsum(estimates, dim=0) / torch.arange(  # type: ignore
        1, estimates.numel() + 1
    )


def test_analytic_laplace_normal_kl_matches_mcmc_estimate() -> None:
    from svae import models  # noqa: F401, register custom KL div function

    torch.manual_seed(42)
    p = dist.Normal(0, 1)
    q = dist.Laplace(2, 1)
    analytic = dist.kl_divergence(p, q)
    estimates = kl_normal_laplace_mc(p, q)
    assert torch.abs(analytic - estimates[-1]) < 0.1
