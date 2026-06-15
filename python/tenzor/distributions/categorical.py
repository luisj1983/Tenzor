"""Categorical distribution — Tensor/Variable native."""
from __future__ import annotations

import numpy as np

import tenzor as _tz
from tenzor.tenzor_core import Tensor, Variable  # type: ignore

from .distribution import (
    Distribution,
    _to_variable,
    _wrap_numpy_int,
)


class Categorical(Distribution):
    """``Categorical(probs)`` — categorical distribution over ``K`` classes.

    The trailing dimension of ``probs`` is the event axis.

    ``sample`` returns ``int64`` indices in a Tensor.  ``has_rsample`` is
    ``False``; for differentiable categorical-style sampling consider
    Gumbel-softmax (see ``tenzor.distributions.RelaxedCategorical`` —
    future work; currently absent).

    Args:
        probs: unnormalised probabilities (shape ``(..., K)``); the last
            axis is normalised to sum to 1.
    """

    has_rsample = False

    def __init__(self, probs):
        probs = _to_variable(probs)
        # Normalise over the event (last) axis with autograd-aware ops so the
        # gradient flows back to the caller's `probs` through log_prob/entropy
        # (the previous numpy round-trip + detached Variable severed the graph,
        # breaking REINFORCE / policy-gradient — the main use of Categorical).
        self.probs = probs / _tz.sum(probs, -1, True)
        # Detached numpy copy for the (inherently non-differentiable) sampler
        # and for shape bookkeeping.
        probs_np = np.asarray(
            self.probs.tensor() if isinstance(self.probs, Variable) else self.probs,
            dtype=np.float64)
        self._probs_np = probs_np.astype(np.float32, copy=False)
        self._num_events = int(probs_np.shape[-1])
        super().__init__(tuple(int(s) for s in probs_np.shape[:-1]))

    @property
    def num_events(self) -> int:
        return self._num_events

    @property
    def mean(self):
        raise NotImplementedError("Categorical has no scalar mean; use probs")

    @property
    def variance(self):
        raise NotImplementedError("Categorical has no scalar variance; use probs")

    def sample(self, sample_shape=()):
        sample_shape = tuple(int(s) for s in sample_shape)
        batch_size = int(np.prod(self._batch_shape)) if self._batch_shape else 1
        probs_2d = self._probs_np.reshape(batch_size, self._num_events)
        n_samples = int(np.prod(sample_shape)) if sample_shape else 1
        # Vectorised inverse-CDF draw: build the per-row cumulative
        # distribution once, then locate uniforms with searchsorted. This is a
        # constant number of numpy calls instead of one np.random.choice per
        # batch row.
        cdf = np.cumsum(probs_2d.astype(np.float64), axis=-1)
        # Guard against floating-point drift leaving the final bin < 1 so a
        # uniform very close to 1 always lands in a valid class.
        cdf[:, -1] = 1.0
        u = np.random.random_sample((batch_size, n_samples))
        # Inverse-CDF via a single broadcast compare-and-count over the event
        # axis: index = number of cumulative thresholds the uniform exceeds.
        # (batch, n_samples, 1) >= (batch, 1, K) -> sum over K.
        out = (u[:, :, None] >= cdf[:, None, :]).sum(axis=-1).astype(np.int64)
        # Clamp any index that rounded onto num_events (possible at u == 1.0).
        np.clip(out, 0, self._num_events - 1, out=out)
        out_shape = sample_shape + self._batch_shape
        if out_shape:
            return _wrap_numpy_int(out.T.reshape(out_shape))
        return _wrap_numpy_int(out.reshape(()))

    def log_prob(self, value):
        # log p(value) = log_probs[..., value].  Implemented as a one-hot dot
        # product over the event axis so the whole computation stays in
        # Variable space and the gradient flows to `probs` — gather/select via
        # numpy would detach it.  The one-hot selector is a constant (no grad).
        eps = 1e-12
        log_probs = _tz.log(self.probs + eps)            # Variable (..., K)

        value_np = np.asarray(
            value.tensor() if isinstance(value, Variable) else value,
            dtype=np.int64)
        onehot_np = np.zeros(value_np.shape + (self._num_events,), dtype=np.float32)
        np.put_along_axis(onehot_np, np.expand_dims(value_np, -1), 1.0, axis=-1)
        onehot = Variable(Tensor.from_numpy(np.ascontiguousarray(onehot_np)), False)

        return _tz.sum(log_probs * onehot, -1)

    def entropy(self):
        # H = -sum_k p_k * log p_k along the event axis, autograd-aware on probs.
        eps = 1e-12
        log_p = _tz.log(self.probs + eps)
        return _tz.sum(self.probs * log_p, -1) * (-1.0)

    def support(self):
        return f"{{0, 1, ..., {self._num_events - 1}}}"
