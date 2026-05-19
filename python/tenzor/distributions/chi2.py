"""Chi-squared distribution."""
import numpy as np
from scipy.special import gammaln, digamma
from .distribution import Distribution, _to_numpy


class Chi2(Distribution):
    """Chi2(df) — chi-squared distribution with df degrees of freedom.

    Chi2(df) is a special case of Gamma(df/2, 1/2).

    Args:
        df: Degrees of freedom (scalar or array, > 0).
    """

    has_rsample = True

    def __init__(self, df):
        self.df = _to_numpy(df)
        super().__init__(self.df.shape)

    @property
    def mean(self):
        return self.df.copy()

    @property
    def variance(self):
        return 2.0 * self.df

    def sample(self, sample_shape=()):
        shape = tuple(sample_shape) + self._batch_shape
        return np.random.chisquare(df=self.df, size=shape or None)

    def rsample(self, sample_shape=()):
        return self.sample(sample_shape)

    def log_prob(self, value):
        value = _to_numpy(value)
        k = self.df
        # Chi2(df) = Gamma(df/2, 1/2):
        # log p(x) = (k/2 - 1)*log(x) - x/2 - (k/2)*log(2) - lgamma(k/2)
        return ((k / 2.0 - 1.0) * np.log(value)
                - value / 2.0
                - (k / 2.0) * math.log(2.0)
                - gammaln(k / 2.0))

    def entropy(self):
        k = self.df
        # H = k/2 + log(2) + lgamma(k/2) + (1 - k/2)*digamma(k/2)
        import math
        return (k / 2.0 + math.log(2.0) + gammaln(k / 2.0)
                + (1.0 - k / 2.0) * digamma(k / 2.0))

    def support(self):
        return "(0, inf)"


import math  # noqa: E402 — already at top, just ensuring availability
