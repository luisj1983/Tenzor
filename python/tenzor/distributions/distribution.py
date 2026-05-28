"""Base Distribution class for tenzor.distributions.

Tenzor's distribution layer is Tensor/Variable-native:

* Parameters passed in are coerced to ``tenzor.tenzor_core.Tensor`` (or
  ``Variable`` if they already carry ``requires_grad``).
* ``sample()`` and ``rsample()`` return Tensors / Variables — never
  ``numpy.ndarray``.
* ``rsample()`` is only implemented when a genuine reparameterised
  gradient path exists.  ``has_rsample`` is set to ``True`` exclusively
  on those classes.  A distribution that does *not* implement
  reparameterisation either has ``has_rsample = False`` (and inheriting
  ``rsample()`` raises ``NotImplementedError``) or overrides ``rsample``
  to raise explicitly — never to silently call ``sample`` and lie.

Numpy is still used inside ``sample()`` for discrete distributions where
Tenzor has no native discrete sampler (``Binomial``, ``Poisson``,
``Geometric``, ``Multinomial``, ``NegativeBinomial``); the result is
wrapped back into a Tensor before return.

Scipy is an *optional* runtime dependency.  Only a handful of
distributions (CDF / quantile via incomplete-gamma / incomplete-beta
functions etc.) need it.  The lazy-gate helpers below keep
``import tenzor.distributions`` scipy-free.
"""
from __future__ import annotations

import math
from typing import Sequence, Union

import numpy as np

# Tenzor core types & ops.  Importing the C++ extension is OK here —
# `tenzor.distributions` is only ever consumed from environments where
# tenzor itself has been initialised.
from tenzor.tenzor_core import Tensor, Variable, dtype as _dtype  # type: ignore

# ---------------------------------------------------------------------------
# Scipy gating (kept for distributions whose CDF/quantile have no closed
# form in elementary Tenzor ops — Beta, Gamma, StudentT, ...).
# ---------------------------------------------------------------------------

def _require_scipy_special():
    """Return scipy.special, raising a clear error if scipy is not installed."""
    try:
        from scipy import special as scipy_special  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            "scipy is required for this distribution's analytic CDF / quantile / "
            "log-prob.  Install with `pip install scipy`."
        ) from exc
    return scipy_special


def _require_scipy_stats():
    """Return scipy.stats, raising a clear error if scipy is not installed."""
    try:
        from scipy import stats as scipy_stats  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            "scipy is required for this distribution's sampling / quantile.  "
            "Install with `pip install scipy`."
        ) from exc
    return scipy_stats


# ---------------------------------------------------------------------------
# Coercion helpers
# ---------------------------------------------------------------------------

ScalarOrArray = Union[float, int, np.ndarray, Tensor, Variable]


def _to_tensor(x: ScalarOrArray, dtype=None) -> Tensor:
    """Coerce ``x`` to a CPU ``Tensor``.

    * Python scalars → 0-d tensor.
    * lists / tuples / numpy arrays → tensor via ``Tensor.from_numpy``.
    * ``Tensor`` passes through unchanged.
    * ``Variable`` → its underlying tensor (gradient is dropped — for
      cases where only the *value* is needed, e.g. inside numpy sample
      shims).
    """
    target = dtype if dtype is not None else _dtype.float32
    if isinstance(x, Variable):
        return x.tensor()
    if isinstance(x, Tensor):
        return x
    if isinstance(x, np.ndarray):
        # Coerce numpy → float32 by default to stay aligned with the
        # tenzor compute path.
        arr = x.astype(np.float32, copy=False) if target == _dtype.float32 else x
        return Tensor.from_numpy(arr)
    if isinstance(x, (list, tuple)):
        return Tensor.from_numpy(np.asarray(x, dtype=np.float32))
    # Scalar.
    import tenzor as _tz
    return _tz.full([], float(x), target)


def _to_variable(x: ScalarOrArray, requires_grad: bool = False, dtype=None) -> Variable:
    """Coerce ``x`` to a ``Variable``.

    Passes through Variables unchanged (preserving ``requires_grad``);
    Tensors are wrapped into a non-grad Variable unless
    ``requires_grad=True``.
    """
    if isinstance(x, Variable):
        return x
    t = _to_tensor(x, dtype=dtype)
    return Variable(t, requires_grad)


def _broadcast_shape(*shapes: Sequence[int]) -> tuple:
    """Compute the numpy-style broadcast of several integer shapes."""
    return tuple(np.broadcast_shapes(*[tuple(s) for s in shapes]))


def _shape_of(x: Union[Tensor, Variable]) -> tuple:
    """Return the shape of a Tensor / Variable as a tuple."""
    return tuple(x.shape)


def _to_numpy(x: ScalarOrArray) -> np.ndarray:
    """Convert ``x`` to a numpy float64 array.  Used by discrete
    distributions whose ``sample()`` goes through ``np.random.*``."""
    if isinstance(x, Variable):
        return np.asarray(x.tensor(), dtype=np.float64)
    if isinstance(x, Tensor):
        return np.asarray(x, dtype=np.float64)
    if isinstance(x, np.ndarray):
        return x.astype(np.float64, copy=False)
    return np.asarray(x, dtype=np.float64)


def _wrap_numpy(arr: np.ndarray) -> Tensor:
    """Wrap a numpy array as a tenzor Tensor (float32 by default)."""
    if arr.dtype == np.float64:
        arr = arr.astype(np.float32, copy=False)
    return Tensor.from_numpy(np.ascontiguousarray(arr))


def _wrap_numpy_int(arr: np.ndarray) -> Tensor:
    """Wrap an integer numpy array as a tenzor Int64 Tensor."""
    if arr.dtype != np.int64:
        arr = arr.astype(np.int64, copy=False)
    return Tensor.from_numpy(np.ascontiguousarray(arr))


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class Distribution:
    """Base class for all probability distributions.

    Subclasses must implement at least ``sample`` and ``log_prob``.
    Closed-form ``cdf`` / ``icdf`` / ``entropy`` / ``mean`` / ``variance``
    are optional.  ``rsample`` is opt-in via ``has_rsample = True`` plus
    a proper reparameterised implementation — never via a silent
    fallback to ``sample``.
    """

    has_rsample: bool = False

    def __init__(self, batch_shape: Sequence[int] = ()):  # noqa: D401
        self._batch_shape = tuple(int(s) for s in batch_shape)

    # -- shape API -----------------------------------------------------

    @property
    def batch_shape(self) -> tuple:
        """Shape of the batched distribution parameters."""
        return self._batch_shape

    @property
    def event_shape(self) -> tuple:
        """Shape of a single sample (excluding batch dimensions)."""
        return ()

    # -- statistics ----------------------------------------------------

    @property
    def mean(self):
        raise NotImplementedError(f"{type(self).__name__} does not implement mean")

    @property
    def variance(self):
        raise NotImplementedError(f"{type(self).__name__} does not implement variance")

    @property
    def stddev(self):
        """Standard deviation — ``sqrt(variance)``.

        Default uses ``tz.pow(.., 0.5)`` so the result is autograd-aware
        when ``variance`` is a Variable.
        """
        import tenzor as _tz
        v = self.variance
        if isinstance(v, (Variable, Tensor)):
            return _tz.pow(v, 0.5)
        return np.sqrt(v)

    # -- sampling ------------------------------------------------------

    def sample(self, sample_shape: Sequence[int] = ()) -> Union[Tensor, Variable]:
        """Draw samples from the distribution.

        Returns a Tenzor ``Tensor`` (or ``Variable`` for distributions
        that compose other Variables internally — most ``sample()``
        paths return plain Tensors).
        """
        raise NotImplementedError(f"{type(self).__name__} does not implement sample")

    def rsample(self, sample_shape: Sequence[int] = ()) -> Variable:
        """Draw a reparameterised sample — gradients flow through the
        distribution's parameters.

        ``Distribution`` raises ``NotImplementedError`` unconditionally:
        no silent ``sample()`` fallback.  Subclasses must override and
        set ``has_rsample = True``.
        """
        if not self.has_rsample:
            raise NotImplementedError(
                f"{type(self).__name__} is not reparameterisable. "
                "Use sample() instead.  For gradient estimates through "
                "non-reparameterisable distributions, consider the "
                "score-function (REINFORCE) estimator."
            )
        raise NotImplementedError(
            f"{type(self).__name__}.rsample must be overridden by the "
            "subclass when has_rsample is True."
        )

    # -- densities ------------------------------------------------------

    def log_prob(self, value):
        """Log probability (or log probability density) of ``value``.

        Should accept a ``Tensor`` / ``Variable`` / numpy / Python value
        and return a Tenzor object (autograd-aware when params have
        ``requires_grad``).
        """
        raise NotImplementedError(f"{type(self).__name__} does not implement log_prob")

    def cdf(self, value):
        """Cumulative distribution function."""
        raise NotImplementedError(f"{type(self).__name__} does not implement cdf")

    def icdf(self, q):
        """Inverse CDF (quantile function)."""
        raise NotImplementedError(f"{type(self).__name__} does not implement icdf")

    def entropy(self):
        """Differential / Shannon entropy of the distribution."""
        raise NotImplementedError(f"{type(self).__name__} does not implement entropy")

    def support(self):
        """String description of the support, or ``None``."""
        return None

    # -- repr ----------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        params = {k: v for k, v in vars(self).items() if not k.startswith("_")}
        param_str = ", ".join(f"{k}={v}" for k, v in params.items())
        return f"{type(self).__name__}({param_str})"
