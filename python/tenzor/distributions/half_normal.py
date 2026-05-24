"""HalfNormal distribution."""
import math
import numpy as np
# V.39: scipy is lazy (see distribution.py).
from .distribution import Distribution, _to_numpy, _require_scipy_special


class HalfNormal(Distribution):
    """HalfNormal(scale) — half-normal distribution (non-negative side of N(0, scale)).

    Args:
        scale: Scale parameter (scalar or array, > 0).
    """

    has_rsample = True

    def __init__(self, scale):
        self.scale = _to_numpy(scale)
        super().__init__(self.scale.shape)

    @property
    def mean(self):
        # E[X] = scale * sqrt(2/pi)
        return self.scale * math.sqrt(2.0 / math.pi)

    @property
    def variance(self):
        # Var[X] = scale^2 * (1 - 2/pi)
        return self.scale ** 2 * (1.0 - 2.0 / math.pi)

    def sample(self, sample_shape=()):
        shape = tuple(sample_shape) + self._batch_shape
        return np.abs(np.random.normal(0.0, self.scale, size=shape or None))

    def rsample(self, sample_shape=()):
        return self.sample(sample_shape)

    def log_prob(self, value):
        value = _to_numpy(value)
        in_support = value >= 0
        # log p(x) = log(2) + Normal(0, scale).log_prob(x) for x >= 0
        z = value / self.scale
        lp = (math.log(2.0) - 0.5 * math.log(2 * math.pi)
              - np.log(self.scale) - 0.5 * z * z)
        return np.where(in_support, lp, -np.inf)

    def cdf(self, value):
        scipy_special = _require_scipy_special()
        value = _to_numpy(value)
        return np.where(
            value >= 0,
            scipy_special.erf(value / (self.scale * math.sqrt(2.0))),
            0.0
        )

    def icdf(self, p):
        scipy_special = _require_scipy_special()
        p = _to_numpy(p)
        return self.scale * math.sqrt(2.0) * scipy_special.erfinv(p)

    def entropy(self):
        # H = 0.5 * log(pi * scale^2 / 2) + 0.5
        return 0.5 * np.log(math.pi * self.scale ** 2 / 2.0) + 0.5

    def support(self):
        return "[0, inf)"
