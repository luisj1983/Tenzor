"""Categorical distribution."""
import numpy as np
from .distribution import Distribution, _to_numpy


class Categorical(Distribution):
    """Categorical(probs) — categorical distribution over K classes.

    Args:
        probs: Unnormalized probabilities (array of shape (..., K)).
               Normalized to sum to 1 along the last dimension.
    """

    has_rsample = False

    def __init__(self, probs):
        probs = _to_numpy(probs)
        total = probs.sum(axis=-1, keepdims=True)
        self.probs = probs / total
        self._num_events = probs.shape[-1]
        super().__init__(probs.shape[:-1])

    @property
    def num_events(self):
        return self._num_events

    @property
    def mean(self):
        raise NotImplementedError("Categorical has no scalar mean; use probs")

    @property
    def variance(self):
        raise NotImplementedError("Categorical has no scalar variance; use probs")

    def sample(self, sample_shape=()):
        batch_size = int(np.prod(self._batch_shape)) if self._batch_shape else 1
        probs_2d = self.probs.reshape(batch_size, self._num_events)
        n_samples = int(np.prod(sample_shape)) if sample_shape else 1
        out = np.array([
            np.random.choice(self._num_events, size=n_samples, p=probs_2d[b])
            for b in range(batch_size)
        ], dtype=np.int64)
        out_shape = tuple(sample_shape) + self._batch_shape
        return out.reshape(out_shape) if out_shape else out.reshape(-1)[0]

    def log_prob(self, value):
        # Use take_along_axis so batched probs (..., K) + matching-shape
        # value (...) return per-batch log-probability of the chosen
        # class — `lp[..., value]` advanced-indexes the whole `value`
        # array across the last axis, producing a Cartesian-product
        # shape (audit item A.9.a).
        value = np.asarray(value, dtype=np.int64)
        eps = 1e-7
        lp = np.log(np.clip(self.probs, eps, 1.0))
        if value.ndim == 0:
            # Scalar value broadcast across all (if any) batches.
            return lp[..., int(value)]
        # Audit H.3 — Categorical with unbatched probs(K,) needs to accept
        # any-shape value array (matches torch.distributions semantics:
        # log_prob output shape == value shape). The take_along_axis path
        # requires value.shape == lp.shape[:-1], which fails when the
        # batch is empty.  For the unbatched case fall back to direct
        # indexing along the single axis.
        if lp.ndim == 1:
            return lp[value]
        # Batched probs case: value.shape must match lp.shape[:-1].
        return np.take_along_axis(lp, value[..., np.newaxis], axis=-1).squeeze(-1)

    def entropy(self):
        eps = 1e-7
        p = np.clip(self.probs, eps, 1.0)
        return -(p * np.log(p)).sum(axis=-1)

    def support(self):
        return f"{{0, 1, ..., {self._num_events - 1}}}"
