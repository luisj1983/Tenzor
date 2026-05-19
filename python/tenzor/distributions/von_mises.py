"""VonMises distribution."""
import math
import numpy as np
from scipy.special import i0, i1
from .distribution import Distribution, _to_numpy


class VonMises(Distribution):
    """VonMises(loc, concentration) distribution on the circle.

    Args:
        loc: Location (mu) parameter in radians (scalar or array).
        concentration: Concentration (kappa) parameter (scalar or array, >= 0).
                       kappa=0 → uniform on circle; large kappa → concentrated near loc.
    """

    has_rsample = False  # Not straightforwardly reparameterizable

    def __init__(self, loc, concentration):
        self.loc = _to_numpy(loc)
        self.concentration = _to_numpy(concentration)
        super().__init__(np.broadcast_shapes(self.loc.shape, self.concentration.shape))

    @property
    def mean(self):
        # Circular mean = loc
        return self.loc.copy()

    @property
    def variance(self):
        # Circular variance = 1 - I1(kappa) / I0(kappa)
        kappa = self.concentration
        return 1.0 - i1(kappa) / i0(kappa)

    def sample(self, sample_shape=()):
        shape = tuple(sample_shape) + self._batch_shape
        return np.random.vonmises(self.loc, self.concentration, size=shape or None)

    def log_prob(self, value):
        value = _to_numpy(value)
        # log p(x) = kappa * cos(x - mu) - log(2*pi*I0(kappa))
        kappa = self.concentration
        return (kappa * np.cos(value - self.loc)
                - math.log(2 * math.pi) - np.log(i0(kappa)))

    def entropy(self):
        # H = log(2*pi*I0(kappa)) - kappa * I1(kappa) / I0(kappa)
        kappa = self.concentration
        i0k = i0(kappa)
        i1k = i1(kappa)
        return math.log(2 * math.pi) + np.log(i0k) - kappa * i1k / i0k

    def support(self):
        return "(-pi, pi]"
