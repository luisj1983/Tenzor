"""Base Distribution class.

All distributions in this namespace use numpy for computation.
Parameters may be Python scalars, numpy scalars, or numpy arrays.
"""
import math
import numpy as np


def _to_numpy(x):
    """Convert x to a numpy float64 array."""
    if isinstance(x, np.ndarray):
        return x.astype(np.float64)
    return np.float64(x)


class Distribution:
    """Base class for all probability distributions.

    Subclasses must implement:
      - sample(sample_shape)   — draw samples, returns numpy array
      - log_prob(value)        — log probability/density, returns numpy array

    Optional: cdf, icdf, entropy, rsample, mean, variance.
    """

    has_rsample = False

    def __init__(self, batch_shape=()):
        self._batch_shape = tuple(batch_shape)

    @property
    def batch_shape(self):
        """Shape of batched distribution parameters."""
        return self._batch_shape

    @property
    def mean(self):
        raise NotImplementedError(f"{type(self).__name__} does not implement mean")

    @property
    def variance(self):
        raise NotImplementedError(f"{type(self).__name__} does not implement variance")

    @property
    def stddev(self):
        """Standard deviation — sqrt(variance)."""
        return np.sqrt(self.variance)

    def sample(self, sample_shape=()):
        """Draw samples from the distribution."""
        raise NotImplementedError(f"{type(self).__name__} does not implement sample")

    def rsample(self, sample_shape=()):
        """Draw a reparametrized sample (gradient flows through parameters).

        Raises NotImplementedError for non-reparameterizable distributions.
        No silent straight-through placeholder is inserted.
        """
        if not self.has_rsample:
            raise NotImplementedError(
                f"{type(self).__name__} is not reparameterizable. "
                "Use sample() instead. For gradient estimates, consider "
                "score-function (REINFORCE) estimators."
            )
        raise NotImplementedError(f"{type(self).__name__}.rsample not implemented")

    def log_prob(self, value):
        """Log probability (or log probability density) of value."""
        raise NotImplementedError(f"{type(self).__name__} does not implement log_prob")

    def cdf(self, value):
        """Cumulative distribution function."""
        raise NotImplementedError(f"{type(self).__name__} does not implement cdf")

    def icdf(self, p):
        """Inverse CDF (quantile function)."""
        raise NotImplementedError(f"{type(self).__name__} does not implement icdf")

    def entropy(self):
        """Differential/Shannon entropy of the distribution."""
        raise NotImplementedError(f"{type(self).__name__} does not implement entropy")

    def support(self):
        """String description of the support."""
        return None

    def __repr__(self):
        params = {k: v for k, v in vars(self).items() if not k.startswith("_")}
        param_str = ", ".join(f"{k}={v}" for k, v in params.items())
        return f"{type(self).__name__}({param_str})"
