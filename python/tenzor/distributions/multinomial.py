"""Multinomial distribution — Tensor-native (sample only)."""
from __future__ import annotations

import numpy as np

from tenzor.tenzor_core import Tensor, Variable  # type: ignore

from .distribution import (
    Distribution,
    _to_variable,
    _wrap_numpy,
    _wrap_numpy_int,
    _require_scipy_special,
)


class Multinomial(Distribution):
    """``Multinomial(total_count, probs)`` distribution.

    The last axis of ``probs`` is the event axis.  ``has_rsample =
    False`` (discrete distribution).
    """

    has_rsample = False

    def __init__(self, total_count, probs):
        self.total_count = int(total_count)
        probs = _to_variable(probs)
        probs_np = np.asarray(probs.tensor(), dtype=np.float64)
        total = probs_np.sum(axis=-1, keepdims=True)
        self._probs_np = (probs_np / total).astype(np.float32, copy=False)
        self.probs = Variable(Tensor.from_numpy(np.ascontiguousarray(self._probs_np)),
                              False)
        self._num_events = int(probs_np.shape[-1])
        super().__init__(tuple(int(s) for s in probs_np.shape[:-1]))

    @property
    def mean(self):
        return _wrap_numpy(self._probs_np * float(self.total_count))

    @property
    def variance(self):
        p = self._probs_np
        return _wrap_numpy(p * (1.0 - p) * float(self.total_count))

    def sample(self, sample_shape=()):
        sample_shape = tuple(int(s) for s in sample_shape)
        out_shape = sample_shape + self._batch_shape + (self._num_events,)
        # numpy.random.multinomial expects a 1-D probability vector.
        # Iterate over the leading batch dimensions.
        batch_size = int(np.prod(self._batch_shape)) if self._batch_shape else 1
        probs_2d = self._probs_np.reshape(batch_size, self._num_events)
        n_samples = int(np.prod(sample_shape)) if sample_shape else 1
        out = np.empty((batch_size, n_samples, self._num_events), dtype=np.int64)
        for b in range(batch_size):
            out[b] = np.random.multinomial(self.total_count,
                                           probs_2d[b],
                                           size=n_samples)
        # Reshape to out_shape (sample, batch, K).  out is (batch, sample, K).
        out_t = out.transpose(1, 0, 2).reshape(out_shape)
        return _wrap_numpy_int(out_t)

    def log_prob(self, value):
        gammaln = _require_scipy_special().gammaln
        v_np = np.asarray(_to_variable(value).tensor()
                          if isinstance(value, Variable)
                          else np.asarray(value, dtype=np.float64),
                          dtype=np.float64)
        eps = 1e-12
        p = np.clip(self._probs_np.astype(np.float64), eps, 1.0)
        log_norm = (gammaln(self.total_count + 1.0)
                    - gammaln(v_np + 1.0).sum(axis=-1))
        out = log_norm + (v_np * np.log(p)).sum(axis=-1)
        return _wrap_numpy(out)

    def support(self):
        return f"vectors of K non-negative integers summing to {self.total_count}"
