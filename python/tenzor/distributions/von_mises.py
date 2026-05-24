"""VonMises distribution."""
import math
import numpy as np
# V.39: scipy is lazy (see distribution.py).
from .distribution import (
    Distribution,
    _to_numpy,
    _require_scipy_special,
    _require_scipy_stats,
)


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
        sp = _require_scipy_special()
        kappa = self.concentration
        return 1.0 - sp.i1(kappa) / sp.i0(kappa)

    def sample(self, sample_shape=()):
        shape = tuple(sample_shape) + self._batch_shape
        return np.random.vonmises(self.loc, self.concentration, size=shape or None)

    def log_prob(self, value):
        sp = _require_scipy_special()
        value = _to_numpy(value)
        # log p(x) = kappa * cos(x - mu) - log(2*pi*I0(kappa))
        kappa = self.concentration
        return (kappa * np.cos(value - self.loc)
                - math.log(2 * math.pi) - np.log(sp.i0(kappa)))

    def entropy(self):
        # H = log(2*pi*I0(kappa)) - kappa * I1(kappa) / I0(kappa)
        sp = _require_scipy_special()
        kappa = self.concentration
        i0k = sp.i0(kappa)
        i1k = sp.i1(kappa)
        return math.log(2 * math.pi) + np.log(i0k) - kappa * i1k / i0k

    def cdf(self, value):
        """CDF of VonMises (audit item E.5).

        Uses the Marsaglia series via SciPy's vonmises.cdf, which sums

            F(x; mu, kappa) = (x - mu) / (2 pi)
                            + (1 / (pi I_0(kappa)))
                              * Sum_{j>=1} (I_j(kappa) / j) * sin(j (x - mu))

        and returns CDF in [0, 1] over the principal branch (-pi, pi].
        """
        _scipy_vonmises = _require_scipy_stats().vonmises
        value = _to_numpy(value)
        # scipy parameterises VonMises as kappa with loc=mu; broadcast manually.
        return _scipy_vonmises.cdf(value, self.concentration, loc=self.loc)

    def icdf(self, q):
        """Inverse CDF (PPF) of VonMises (audit item E.5)."""
        _scipy_vonmises = _require_scipy_stats().vonmises
        q = _to_numpy(q)
        return _scipy_vonmises.ppf(q, self.concentration, loc=self.loc)

    def support(self):
        return "(-pi, pi]"
