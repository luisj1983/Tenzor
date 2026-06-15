"""Multinomial distribution — Tensor-native (sample only)."""
from __future__ import annotations

import numpy as np

import tenzor as _tz
from tenzor.tenzor_core import Tensor, Variable  # type: ignore

from .distribution import (
    Distribution,
    _to_variable,
    _wrap_numpy,
    _wrap_numpy_int,
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
        # Normalise over the event (last) axis with autograd-aware ops so the
        # gradient flows back to the caller's `probs` through log_prob (the
        # previous numpy round-trip + detached Variable severed the graph).
        self.probs = probs / _tz.sum(probs, -1, True)
        # Detached numpy copy for the (inherently non-differentiable) sampler
        # and for mean/variance/shape bookkeeping.
        probs_np = np.asarray(self.probs.tensor(), dtype=np.float64)
        self._probs_np = probs_np.astype(np.float32, copy=False)
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
        batch_size = int(np.prod(self._batch_shape)) if self._batch_shape else 1
        probs_2d = self._probs_np.reshape(batch_size, self._num_events)
        n_samples = int(np.prod(sample_shape)) if sample_shape else 1
        K = self._num_events
        n = self.total_count
        # Vectorised draw: a Multinomial(n, p) count vector is the histogram of
        # n i.i.d. Categorical(p) draws. Build the per-row CDF once, then map
        # uniforms to event indices with np.searchsorted (O(n log K), no
        # K-sized comparison intermediate) and tally with a single bincount.
        # We chunk over the (batch*sample) rows so the largest live array is
        # bounded regardless of total_count / num_events / batch — the old
        # (batch, n_samples, n, K) boolean tensor could OOM at NLP scales.
        cdf = np.cumsum(probs_2d.astype(np.float64), axis=-1)
        cdf[:, -1] = 1.0  # guard against fp drift at the final bin
        rows = batch_size * n_samples
        out = np.empty((rows, K), dtype=np.int64)
        # Cap each chunk's index buffer (chunk_rows * n int64) at ~10M elements,
        # mirroring Binomial.entropy's memory guard; always at least one row.
        rows_per_chunk = max(1, 10_000_000 // max(1, n))
        # Map a flat row index r -> its batch index (probs_2d / cdf is indexed
        # by batch, replicated across the n_samples for that batch).
        # Row layout below is (batch, n_samples) flattened C-order.
        for start in range(0, rows, rows_per_chunk):
            stop = min(start + rows_per_chunk, rows)
            chunk_rows = stop - start
            batch_idx = (np.arange(start, stop, dtype=np.int64)
                         // max(1, n_samples))
            u = np.random.random_sample((chunk_rows, n))
            # Per-row inverse-CDF: searchsorted on each row's monotone CDF.
            idx = np.empty((chunk_rows, n), dtype=np.int64)
            for i in range(chunk_rows):
                idx[i] = np.searchsorted(cdf[batch_idx[i]], u[i], side="right")
            np.clip(idx, 0, K - 1, out=idx)
            # Tally counts per row into K bins via one offset bincount.
            row_offset = (np.arange(chunk_rows, dtype=np.int64) * K)[:, None]
            binned = np.bincount((idx + row_offset).ravel(),
                                 minlength=chunk_rows * K)
            out[start:stop] = binned.reshape(chunk_rows, K)
        out = out.reshape(batch_size, n_samples, K)
        # Reshape to out_shape (sample, batch, K).  out is (batch, sample, K).
        out_t = out.transpose(1, 0, 2).reshape(out_shape)
        return _wrap_numpy_int(out_t)

    def log_prob(self, value):
        # log p(x) = log(n! / prod_k x_k!) + sum_k x_k·log(p_k).  Built with
        # autograd-aware Variable ops so the gradient flows to probs (the
        # previous numpy round-trip detached it).  The multinomial-coefficient
        # normaliser log(n!) - sum_k lgamma(x_k+1) is computed on-device from
        # the (detached) value and contributes no gradient to probs.
        import math
        value_v = _to_variable(value)
        eps = 1e-12
        log_norm = (math.lgamma(self.total_count + 1.0)
                    - _tz.sum(_tz.lgamma(value_v + 1.0), -1))
        log_p = _tz.log(self.probs + eps)
        return log_norm + _tz.sum(value_v * log_p, -1)

    def support(self):
        return f"vectors of K non-negative integers summing to {self.total_count}"
