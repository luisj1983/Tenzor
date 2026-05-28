"""Categorical distribution — Tensor/Variable native."""
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
        probs_np = np.asarray(probs.tensor(), dtype=np.float64)
        total = probs_np.sum(axis=-1, keepdims=True)
        norm_np = (probs_np / total).astype(np.float32, copy=False)
        # Store the normalised probs as a (detached) Variable so callers
        # who feed unnormalised inputs still see a clean parameter
        # surface.  Tensor/Variable arithmetic over `probs` is used in
        # `log_prob` / `entropy`.
        self.probs = Variable(Tensor.from_numpy(np.ascontiguousarray(norm_np)), False)
        self._probs_np = norm_np
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
        out = np.empty((batch_size, n_samples), dtype=np.int64)
        for b in range(batch_size):
            out[b] = np.random.choice(self._num_events,
                                      size=n_samples,
                                      p=probs_2d[b])
        out_shape = sample_shape + self._batch_shape
        if out_shape:
            return _wrap_numpy_int(out.T.reshape(out_shape))
        return _wrap_numpy_int(out.reshape(()))

    def log_prob(self, value):
        # value: integer indices.  We use take_along_axis on numpy buffers
        # because Tenzor lacks a Variable-aware gather along the last axis
        # at module level; gradient through log_prob therefore flows only
        # through `probs` via the closed-form expression below.
        value_np = np.asarray(_to_variable(value).tensor()
                              if isinstance(value, Variable)
                              else np.asarray(value, dtype=np.int64),
                              dtype=np.int64)
        eps = 1e-12
        # Compute log-probs autograd-aware on the full probs tensor, then
        # gather via numpy.  Gradient flows up to the gather point.
        log_probs_var = _tz.log(self.probs + eps)
        log_probs_np = np.asarray(log_probs_var.tensor()
                                  if isinstance(log_probs_var, Variable)
                                  else log_probs_var, dtype=np.float64)
        if value_np.ndim == 0:
            out = log_probs_np[..., int(value_np)]
        elif log_probs_np.ndim == 1:
            out = log_probs_np[value_np]
        else:
            out = np.take_along_axis(log_probs_np,
                                     value_np[..., np.newaxis],
                                     axis=-1).squeeze(-1)
        return _wrap_numpy(out.astype(np.float32, copy=False))

    def entropy(self):
        # H = -sum p * log p along last axis.  Autograd-aware on probs.
        eps = 1e-12
        log_p = _tz.log(self.probs + eps)
        # tz.sum supports axis when passed as positional? Use Tensor route
        # to keep the codepath simple.
        prod = self.probs * log_p
        prod_np = np.asarray(prod.tensor() if isinstance(prod, Variable) else prod,
                             dtype=np.float32)
        return _wrap_numpy(-prod_np.sum(axis=-1))

    def support(self):
        return f"{{0, 1, ..., {self._num_events - 1}}}"
