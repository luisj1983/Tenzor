"""Normal (Gaussian) distribution — Tensor/Variable native."""
from __future__ import annotations

import math

import tenzor as _tz
from tenzor.tenzor_core import Tensor, Variable  # type: ignore

from .distribution import (
    Distribution,
    _broadcast_shape,
    _shape_of,
    _to_tensor,
    _to_variable,
    _require_scipy_special,
)
from ._reparam import standard_normal


class Normal(Distribution):
    """``Normal(loc, scale)`` — Gaussian distribution.

    Reparameterised: ``rsample = loc + scale * z`` with ``z ~ N(0, 1)``;
    gradients flow into both ``loc`` and ``scale``.

    Args:
        loc: Mean (scalar / array / Tensor / Variable).
        scale: Standard deviation (positive).
    """

    has_rsample = True

    def __init__(self, loc, scale):
        self.loc = _to_variable(loc)
        self.scale = _to_variable(scale)
        super().__init__(_broadcast_shape(_shape_of(self.loc), _shape_of(self.scale)))

    # -- statistics ---------------------------------------------------

    @property
    def mean(self):
        return self.loc

    @property
    def variance(self):
        return self.scale * self.scale

    @property
    def stddev(self):
        return self.scale

    # -- sampling -----------------------------------------------------

    def sample(self, sample_shape=()):
        # No grad through sample.
        out_shape = tuple(sample_shape) + self._batch_shape
        z = standard_normal(out_shape if out_shape else [])
        loc_t = self.loc.tensor() if isinstance(self.loc, Variable) else self.loc
        scale_t = self.scale.tensor() if isinstance(self.scale, Variable) else self.scale
        return loc_t + scale_t * z

    def rsample(self, sample_shape=()):
        out_shape = tuple(sample_shape) + self._batch_shape
        z = standard_normal(out_shape if out_shape else [])
        # Wrap z as a non-grad Variable so the Variable-arithmetic path
        # is taken (Variable * Tensor is not supported).
        z_v = Variable(z, False)
        return self.loc + self.scale * z_v

    # -- densities ----------------------------------------------------

    def log_prob(self, value):
        value = _to_variable(value)
        z = (value - self.loc) / self.scale
        # log_prob = -0.5*log(2π) - log(scale) - 0.5*z^2
        return (-0.5 * math.log(2.0 * math.pi)) - _tz.log(self.scale) - 0.5 * (z * z)

    def cdf(self, value):
        # Scalar-valued: route through scipy on numpy data.
        scipy_special = _require_scipy_special()
        import numpy as np
        v_np = np.asarray(_to_tensor(value), dtype=np.float64)
        loc_np = np.asarray(self.loc.tensor(), dtype=np.float64)
        scale_np = np.asarray(self.scale.tensor(), dtype=np.float64)
        out = 0.5 * (1.0 + scipy_special.erf((v_np - loc_np) /
                                             (scale_np * math.sqrt(2.0))))
        return Tensor.from_numpy(out.astype(np.float32, copy=False))

    def icdf(self, p):
        scipy_special = _require_scipy_special()
        import numpy as np
        p_np = np.asarray(_to_tensor(p), dtype=np.float64)
        loc_np = np.asarray(self.loc.tensor(), dtype=np.float64)
        scale_np = np.asarray(self.scale.tensor(), dtype=np.float64)
        out = loc_np + scale_np * math.sqrt(2.0) * scipy_special.erfinv(2.0 * p_np - 1.0)
        return Tensor.from_numpy(out.astype(np.float32, copy=False))

    def entropy(self):
        # H = 0.5 * log(2*pi*e) + log(scale)
        return 0.5 * math.log(2.0 * math.pi * math.e) + _tz.log(self.scale)

    def support(self):
        return "(-inf, inf)"
