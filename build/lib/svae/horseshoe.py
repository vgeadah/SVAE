"""Horseshoe probability distribution."""
import math
import numbers

import torch
import torch.distributions.utils
from torch import distributions
from torch.distributions import constraints

DEFAULT_SAMPLE_SHAPE = torch.Size()


def log_e1(x: torch.Tensor) -> torch.Tensor:
    """Log space E1 exponential integral function.

    Barry, D. A; Parlange, J. -Y; Li, L (2000-01-31). "Approximation for the exponential
    integral (Theis well function)". Journal of Hydrology. 227 (1–4): 287–291.
    doi:10.1016/S0022-1694(99)00184-5.

    """
    # Constants from the reference
    G = 0.5614594835668852
    b = 1.042076493835121
    h_inf = 1.080135995250334

    # Helper constants
    G1m = 0.4385405164331148  # 1 - G
    q1 = 0.4255319148936170  # 20 / 47
    q2 = 1.091928428198338  # \sqrt{\frac{31}{26}}

    q = q1 * (x ** q2)
    h = (1 / (1 + x * torch.sqrt(x))) + (h_inf * q / (1 + q))
    return (
        -x
        - torch.log(G + G1m * torch.exp(-x / G1m))
        + torch.log(torch.log1p((G / x) - (G1m / (h + b * x) ** 2)))
    )


class Horseshoe(distributions.Distribution):
    """Horseshoe distribution."""

    arg_constraints = {"loc": constraints.real, "scale": constraints.positive}
    support = constraints.real
    has_rsample = True

    def __init__(self, loc, scale, validate_args=None) -> None:
        self.loc, self.scale = torch.distributions.utils.broadcast_all(loc, scale)
        if isinstance(loc, numbers.Number) and isinstance(scale, numbers.Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.loc.size()
        super(Horseshoe, self).__init__(batch_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None) -> "Horseshoe":
        """Expand to new batch shape."""
        new = self._get_checked_instance(Horseshoe, _instance)
        batch_shape = torch.Size(batch_shape)
        new.loc = self.loc.expand(batch_shape)
        new.scale = self.scale.expand(batch_shape)
        super(Horseshoe, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Log probabilities."""
        scale2 = self.scale ** 2
        r = x ** 2 / (2 * scale2)
        return -0.5 * torch.log(2 * math.pi ** 3 * scale2) + r + log_e1(r)

    def rsample(self, sample_shape=DEFAULT_SAMPLE_SHAPE) -> torch.Tensor:
        """Reparameterized sampling."""
        shape = self._extended_shape(sample_shape)
        local_shrinkage = self.loc.new_empty(shape).cauchy_().abs()
        eps = self.loc.new_empty(shape).normal_()
        return self.loc + local_shrinkage * self.scale * eps
