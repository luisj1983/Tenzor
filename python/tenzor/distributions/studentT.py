"""Student's t distribution."""
import math
import numpy as np
from scipy.special import gammaln, hyp2f1
from scipy.stats import t as scipy_t
from .distribution import Distribution, _to_numpy


class StudentT(Distribution):
    """StudentT(df, loc=0, scale=1) — Student's t distribution.

    Args:
        df: Degrees of freedom (scalar or array, > 0).
        loc: Location parameter (scalar or array, default 0).
        scale: Scale parameter (scalar or array, > 0, default 1).
    """

    has_rsample = True

    def __init__(self, df, loc=0.0, scale=1.0):
        self.df = _to_numpy(df)
        self.loc = _to_numpy(loc)
        self.scale = _to_numpy(scale)
        super().__init__(
            np.broadcast_shapes(self.df.shape, self.loc.shape, self.scale.shape)
        )

    @property
    def mean(self):
        # Mean = loc for df > 1, undefined otherwise
        return np.where(self.df > 1.0, self.loc, np.nan)

    @property
    def variance(self):
        # Var = scale^2 * df/(df-2) for df > 2, inf for 1 < df <= 2, undefined for df <= 1
        df = self.df
        s2 = self.scale ** 2
        var = s2 * df / (df - 2.0)
        return np.where(df > 2.0, var, np.where(df > 1.0, np.inf, np.nan))

    def sample(self, sample_shape=()):
        shape = tuple(sample_shape) + self._batch_shape
        z = np.random.standard_t(self.df, size=shape or None)
        return self.loc + self.scale * z

    def rsample(self, sample_shape=()):
        return self.sample(sample_shape)

    def log_prob(self, value):
        value = _to_numpy(value)
        nu = self.df
        y = (value - self.loc) / self.scale
        # log p(y; nu) = lgamma((nu+1)/2) - 0.5*log(nu*pi) - lgamma(nu/2)
        #              - 0.5*(nu+1)*log(1 + y^2/nu)
        log_unnorm = gammaln((nu + 1.0) / 2.0) - 0.5 * np.log(nu * math.pi) - gammaln(nu / 2.0)
        log_kernel = -0.5 * (nu + 1.0) * np.log(1.0 + y * y / nu)
        return log_unnorm + log_kernel - np.log(self.scale)

    def cdf(self, value):
        value = _to_numpy(value)
        y = (value - self.loc) / self.scale
        return scipy_t.cdf(y, df=self.df)

    def icdf(self, p):
        p = _to_numpy(p)
        return self.loc + self.scale * scipy_t.ppf(p, df=self.df)

    def entropy(self):
        nu = self.df
        # H = 0.5*(nu+1)*(psi((nu+1)/2) - psi(nu/2))
        #   + log(sqrt(nu) * B(nu/2, 0.5))
        from scipy.special import digamma, betaln
        h = (0.5 * (nu + 1.0) * (digamma((nu + 1.0) / 2.0) - digamma(nu / 2.0))
             + 0.5 * np.log(nu) + betaln(nu / 2.0, 0.5))
        return h + np.log(self.scale)

    def support(self):
        return "(-inf, inf)"
