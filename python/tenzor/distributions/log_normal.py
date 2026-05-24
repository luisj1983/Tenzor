"""LogNormal distribution."""
import math
import numpy as np
# V.39: scipy is lazy (see distribution.py).
from .distribution import Distribution, _to_numpy, _require_scipy_special


class LogNormal(Distribution):
    """LogNormal(loc, scale) distribution.

    If X ~ Normal(loc, scale), then exp(X) ~ LogNormal(loc, scale).

    Args:
        loc: Mean of the underlying normal (scalar or array).
        scale: Standard deviation of the underlying normal (scalar or array, > 0).
    """

    has_rsample = True

    def __init__(self, loc, scale):
        self.loc = _to_numpy(loc)
        self.scale = _to_numpy(scale)
        super().__init__(np.broadcast_shapes(self.loc.shape, self.scale.shape))

    @property
    def mean(self):
        return np.exp(self.loc + 0.5 * self.scale ** 2)

    @property
    def variance(self):
        s2 = self.scale ** 2
        return (np.exp(s2) - 1.0) * np.exp(2.0 * self.loc + s2)

    def sample(self, sample_shape=()):
        shape = tuple(sample_shape) + self._batch_shape
        return np.random.lognormal(self.loc, self.scale, size=shape or None)

    def rsample(self, sample_shape=()):
        return self.sample(sample_shape)

    def log_prob(self, value):
        value = _to_numpy(value)
        z = (np.log(value) - self.loc) / self.scale
        return (-0.5 * (math.log(2 * math.pi) + z * z)
                - np.log(self.scale) - np.log(value))

    def cdf(self, value):
        scipy_special = _require_scipy_special()
        value = _to_numpy(value)
        return 0.5 * (1.0 + scipy_special.erf(
            (np.log(value) - self.loc) / (self.scale * math.sqrt(2.0))
        ))

    def entropy(self):
        # H = log(scale * exp(loc + 0.5) * sqrt(2*pi))
        return self.loc + np.log(self.scale) + 0.5 * (1.0 + math.log(2 * math.pi))

    def icdf(self, q):
        """Inverse CDF of LogNormal (audit item E.5).

        Q(q) = exp(loc + scale * Phi^{-1}(q))   where Phi^{-1} is the
        inverse standard normal CDF (scipy.special.ndtri).
        """
        ndtri = _require_scipy_special().ndtri
        q = np.asarray(q, dtype=np.float64)
        # Clip away the 0/1 endpoints so ndtri does not return ±inf.
        eps = 1e-12
        q_clip = np.clip(q, eps, 1.0 - eps)
        return np.exp(self.loc + self.scale * ndtri(q_clip))

    def support(self):
        return "(0, inf)"
