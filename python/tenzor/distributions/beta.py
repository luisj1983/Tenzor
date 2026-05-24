"""Beta distribution."""
import numpy as np
# V.39: scipy is lazy (see distribution.py).
from .distribution import Distribution, _to_numpy, _require_scipy_special


class Beta(Distribution):
    """Beta(concentration1, concentration0) distribution on (0, 1).

    Args:
        concentration1: Alpha (scalar or array, > 0).
        concentration0: Beta (scalar or array, > 0).
    """

    has_rsample = True

    def __init__(self, concentration1, concentration0):
        self.concentration1 = _to_numpy(concentration1)  # alpha
        self.concentration0 = _to_numpy(concentration0)  # beta
        super().__init__(
            np.broadcast_shapes(self.concentration1.shape, self.concentration0.shape)
        )

    @property
    def mean(self):
        a = self.concentration1
        b = self.concentration0
        return a / (a + b)

    @property
    def variance(self):
        a = self.concentration1
        b = self.concentration0
        return a * b / ((a + b) ** 2 * (a + b + 1.0))

    def sample(self, sample_shape=()):
        shape = tuple(sample_shape) + self._batch_shape
        return np.random.beta(self.concentration1, self.concentration0,
                              size=shape or None)

    def rsample(self, sample_shape=()):
        return self.sample(sample_shape)

    def log_prob(self, value):
        scipy_special = _require_scipy_special()
        betaln = scipy_special.betaln
        value = _to_numpy(value)
        a = self.concentration1
        b = self.concentration0
        # log Beta(x; a, b) = (a-1)*log(x) + (b-1)*log(1-x) - log B(a, b)
        return ((a - 1.0) * np.log(value)
                + (b - 1.0) * np.log(1.0 - value)
                - betaln(a, b))

    def cdf(self, value):
        scipy_special = _require_scipy_special()
        value = _to_numpy(value)
        return scipy_special.betainc(self.concentration1, self.concentration0, value)

    def entropy(self):
        scipy_special = _require_scipy_special()
        betaln = scipy_special.betaln
        digamma = scipy_special.digamma
        a = self.concentration1
        b = self.concentration0
        # H = log B(a, b) - (a-1)*psi(a) - (b-1)*psi(b) + (a+b-2)*psi(a+b)
        return (betaln(a, b)
                - (a - 1.0) * digamma(a)
                - (b - 1.0) * digamma(b)
                + (a + b - 2.0) * digamma(a + b))

    def icdf(self, q):
        """Inverse CDF of Beta(a, b) (audit item E.5).

        Inverts the regularised incomplete beta function via
        scipy.special.betaincinv.
        """
        scipy_special = _require_scipy_special()
        q = np.asarray(q, dtype=np.float64)
        return scipy_special.betaincinv(self.concentration1, self.concentration0, q)

    def support(self):
        return "(0, 1)"
