"""
Tenzor Data Loading Utilities

Provides a PyTorch-compatible data loading pipeline with ``Dataset``,
``DataLoader``, ``Sampler``, and ``DistributedSampler``.

Usage:
    import tenzor as tz
    from tenzor.data import Dataset, DataLoader

    class MyDataset(Dataset):
        def __init__(self, data, labels):
            self.data = data
            self.labels = labels

        def __len__(self):
            return len(self.data)

        def __getitem__(self, index):
            return self.data[index], self.labels[index]

    dataset = MyDataset(data, labels)
    loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2)

    for batch_data, batch_labels in loader:
        output = model(batch_data)
        ...
"""

from __future__ import annotations

import dataclasses
import math
import multiprocessing as mp
import os
import random
import signal
import threading
from abc import ABC, abstractmethod
from collections.abc import Iterator, Sized
from queue import Empty, Full, Queue
from typing import Any, Callable, Generic, Optional, Sequence, TypeVar

# W.29: re-export MapDataset from the C++ binding so the .pyi declaration
# matches a real runtime symbol (the binding lives in tenzor_core.data and
# is the C++ base class of the concrete datasets).  Wrapped in try/except
# so a build that lacks tenzor_core (rare — e.g. doc-only environment) still
# imports cleanly.
try:
    from .tenzor_core.data import MapDataset  # type: ignore[attr-defined]
except (ImportError, AttributeError):
    MapDataset = None  # type: ignore[assignment]

T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class Dataset(ABC, Generic[T_co]):
    """Abstract base class for all datasets.

    All datasets that represent a map from keys to data samples should
    subclass this.  Subclasses must override :meth:`__getitem__` and
    should override :meth:`__len__`.

    Example
    -------
    >>> class NumbersDataset(Dataset):
    ...     def __init__(self, n):
    ...         self.n = n
    ...     def __len__(self):
    ...         return self.n
    ...     def __getitem__(self, index):
    ...         return index * 2
    """

    @abstractmethod
    def __getitem__(self, index: int) -> T_co:
        """Fetch the data sample at the given index.

        Parameters
        ----------
        index : int
            Index of the sample to retrieve.

        Returns
        -------
        T_co
            The data sample.
        """
        ...

    def __len__(self) -> int:
        """Return the number of samples in the dataset.

        Subclasses should override this for proper sampler support.

        Raises
        ------
        NotImplementedError
            If the subclass does not implement ``__len__``.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement __len__. "
            "Override __len__ to use default samplers."
        )


class IterableDataset(ABC, Generic[T_co]):
    """Abstract base class for iterable-style datasets.

    Unlike map-style ``Dataset``, an ``IterableDataset`` does not need
    ``__getitem__`` or ``__len__``.  Instead, subclasses implement
    ``__iter__`` to yield samples one at a time.  When used with
    multi-worker ``DataLoader``, each worker receives a copy and should
    use ``get_worker_info()`` to shard the data.

    Example
    -------
    >>> class MyStream(IterableDataset):
    ...     def __iter__(self):
    ...         info = get_worker_info()
    ...         # shard based on info.id / info.num_workers
    ...         for item in stream:
    ...             yield item
    """

    @abstractmethod
    def __iter__(self) -> Iterator[T_co]:
        ...


class WorkerInfo:
    """Metadata passed to each DataLoader worker.

    Attributes
    ----------
    id : int
        Index of this worker within the DataLoader (0-based).
    num_workers : int
        Total number of workers in this DataLoader.
    seed : int
        Per-worker random seed for reproducibility.
    dataset : Any
        Reference to the dataset replica in this worker.
    rank : int
        Distributed rank of this process (default 0).
    world_size : int
        Total number of distributed processes (default 1).
    """

    __slots__ = ("id", "num_workers", "seed", "dataset", "rank", "world_size")

    def __init__(
        self,
        id: int,
        num_workers: int,
        seed: int,
        dataset: Any,
        rank: int = 0,
        world_size: int = 1,
    ):
        self.id = id
        self.num_workers = num_workers
        self.seed = seed
        self.dataset = dataset
        self.rank = rank
        self.world_size = world_size

    def __repr__(self) -> str:
        return (
            f"WorkerInfo(id={self.id}, num_workers={self.num_workers}, "
            f"seed={self.seed}, rank={self.rank}, world_size={self.world_size})"
        )


# Audit item Z.15: _worker_info must be thread-local so the num_workers=1
# fast path (which uses a single prefetch thread inside the main process)
# can populate it without leaking the WorkerInfo into the main thread.
# Multi-process workers see this as their own process-local state.
class _WorkerInfoTLS(threading.local):
    info: Optional[WorkerInfo] = None


_worker_info_tls = _WorkerInfoTLS()


def get_worker_info() -> Optional[WorkerInfo]:
    """Return the ``WorkerInfo`` for the current DataLoader worker.

    Returns ``None`` if called outside a DataLoader worker process.

    When using multi-worker loading with ``IterableDataset``, each worker
    should call this function to determine its shard of the data.  The
    ``rank`` and ``world_size`` fields allow further sharding across
    distributed processes.

    Example
    -------
    >>> class MyStream(IterableDataset):
    ...     def __iter__(self):
    ...         info = get_worker_info()
    ...         if info is not None:
    ...             # Shard across workers and ranks
    ...             total = info.num_workers * info.world_size
    ...             shard_id = info.rank * info.num_workers + info.id
    ...             for i, item in enumerate(self._all_items()):
    ...                 if i % total == shard_id:
    ...                     yield item
    ...         else:
    ...             yield from self._all_items()
    """
    return _worker_info_tls.info


def set_worker_info(info: Optional[WorkerInfo]) -> None:
    """Set the ``WorkerInfo`` for the current process/thread.

    This is called internally by DataLoader worker processes and should
    not normally be called by user code.
    """
    _worker_info_tls.info = info


def clear_worker_info() -> None:
    """Clear the ``WorkerInfo`` for the current process/thread."""
    _worker_info_tls.info = None


class TensorDataset(Dataset):
    """Dataset wrapping one or more tensors.

    Each sample is retrieved by indexing the first dimension of all tensors.

    Parameters
    ----------
    *tensors : Tensor or Variable
        Tensors that all have the same size in the first dimension.

    Example
    -------
    >>> ds = TensorDataset(features, labels)
    >>> x, y = ds[0]
    """

    def __init__(self, *tensors):
        if not tensors:
            raise ValueError("At least one tensor is required")
        first_size = tensors[0].shape[0] if hasattr(tensors[0], 'shape') else len(tensors[0])
        for i, t in enumerate(tensors):
            t_size = t.shape[0] if hasattr(t, 'shape') else len(t)
            if t_size != first_size:
                raise ValueError(
                    f"Tensor at position {i} has size {t_size} in dimension 0, "
                    f"expected {first_size}"
                )
        self.tensors = tensors

    def __getitem__(self, index):
        return tuple(t[index] for t in self.tensors)

    def __len__(self):
        return self.tensors[0].shape[0] if hasattr(self.tensors[0], 'shape') else len(self.tensors[0])


class Subset(Dataset):
    """Subset of a dataset at specified indices.

    Parameters
    ----------
    dataset : Dataset
        The original dataset.
    indices : Sequence[int]
        Indices into the original dataset.

    Example
    -------
    >>> train_set = Subset(full_dataset, range(0, 800))
    >>> val_set = Subset(full_dataset, range(800, 1000))
    """

    def __init__(self, dataset: Dataset, indices: Sequence[int]):
        self.dataset = dataset
        self.indices = list(indices)

    def __getitem__(self, index):
        return self.dataset[self.indices[index]]

    def __len__(self):
        return len(self.indices)


class ConcatDataset(Dataset):
    """Concatenation of multiple datasets.

    Parameters
    ----------
    datasets : Sequence[Dataset]
        List of datasets to concatenate.

    Example
    -------
    >>> combined = ConcatDataset([dataset_a, dataset_b])
    """

    def __init__(self, datasets: Sequence[Dataset]):
        if not datasets:
            raise ValueError("At least one dataset is required")
        self.datasets = list(datasets)
        self.cumulative_sizes = []
        running = 0
        for ds in self.datasets:
            running += len(ds)
            self.cumulative_sizes.append(running)

    def __getitem__(self, index):
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(f"index {index} out of range for ConcatDataset of size {len(self)}")
        # Binary search for the dataset
        lo, hi = 0, len(self.cumulative_sizes) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if index < self.cumulative_sizes[mid]:
                hi = mid
            else:
                lo = mid + 1
        dataset_idx = lo
        if dataset_idx > 0:
            sample_idx = index - self.cumulative_sizes[dataset_idx - 1]
        else:
            sample_idx = index
        return self.datasets[dataset_idx][sample_idx]

    def __len__(self):
        return self.cumulative_sizes[-1] if self.cumulative_sizes else 0


# ---------------------------------------------------------------------------
# Samplers
# ---------------------------------------------------------------------------

class Sampler(ABC, Generic[T]):
    """Base class for all samplers.

    Every sampler subclass must implement :meth:`__iter__` which yields
    sample indices, and :meth:`__len__` returning the number of indices.
    """

    @abstractmethod
    def __iter__(self) -> Iterator[T]:
        ...

    @abstractmethod
    def __len__(self) -> int:
        ...


class SequentialSampler(Sampler[int]):
    """Samples elements sequentially, always in the same order.

    Parameters
    ----------
    data_source : Sized
        Dataset or any object with ``__len__``.
    """

    def __init__(self, data_source: Sized):
        self.data_source = data_source

    def __iter__(self) -> Iterator[int]:
        return iter(range(len(self.data_source)))

    def __len__(self) -> int:
        return len(self.data_source)


class RandomSampler(Sampler[int]):
    """Samples elements randomly without replacement.

    Parameters
    ----------
    data_source : Sized
        Dataset or any object with ``__len__``.
    generator : random.Random or None, optional
        Random number generator for reproducibility.
    """

    def __init__(self, data_source: Sized, generator: Optional[random.Random] = None):
        self.data_source = data_source
        self.generator = generator or random.Random()

    def __iter__(self) -> Iterator[int]:
        n = len(self.data_source)
        indices = list(range(n))
        self.generator.shuffle(indices)
        return iter(indices)

    def __len__(self) -> int:
        return len(self.data_source)


class BatchSampler(Sampler[list[int]]):
    """Wraps another sampler to yield mini-batches of indices.

    Parameters
    ----------
    sampler : Sampler[int]
        Base sampler.
    batch_size : int
        Size of each mini-batch.
    drop_last : bool
        If ``True``, drop the last incomplete batch.
    """

    def __init__(self, sampler: Sampler[int], batch_size: int, drop_last: bool = False):
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self) -> Iterator[list[int]]:
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if batch and not self.drop_last:
            yield batch

    def __len__(self) -> int:
        n = len(self.sampler)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size


class DistributedSampler(Sampler[int]):
    """Sampler that restricts data loading to a subset of the dataset for
    distributed (multi-GPU) training.

    Each process/rank gets a disjoint subset of approximately equal size.
    The dataset is optionally shuffled before splitting.

    Parameters
    ----------
    dataset : Sized
        Dataset to sample from.
    num_replicas : int
        Number of participating processes (world size).
    rank : int
        Rank of the current process within *num_replicas*.
    shuffle : bool, optional
        If ``True``, shuffle indices each epoch.  Default: ``True``.
    seed : int, optional
        Random seed used for shuffling.  Default: ``0``.
    drop_last : bool, optional
        If ``True``, drop tail samples to make the dataset evenly
        divisible across replicas.  Default: ``False``.

    Example
    -------
    >>> sampler = DistributedSampler(dataset, num_replicas=4, rank=0)
    >>> loader = DataLoader(dataset, sampler=sampler)
    >>> for epoch in range(num_epochs):
    ...     sampler.set_epoch(epoch)
    ...     for batch in loader:
    ...         ...
    """

    def __init__(
        self,
        dataset: Sized,
        num_replicas: int,
        rank: int,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ):
        if num_replicas <= 0:
            raise ValueError(f"num_replicas must be positive, got {num_replicas}")
        if rank < 0 or rank >= num_replicas:
            raise ValueError(f"rank must be in [0, {num_replicas}), got {rank}")

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0

        total_size = len(self.dataset)
        if self.drop_last and total_size % self.num_replicas != 0:
            # Truncate to evenly divisible size
            self.num_samples = total_size // self.num_replicas
            self.total_size = self.num_samples * self.num_replicas
        else:
            # Pad to evenly divisible size
            self.num_samples = math.ceil(total_size / self.num_replicas)
            self.total_size = self.num_samples * self.num_replicas

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch for deterministic shuffling.

        Call this at the beginning of each epoch to ensure different
        shuffling across epochs while keeping reproducibility.

        Parameters
        ----------
        epoch : int
            Current epoch number.
        """
        self.epoch = epoch

    def __iter__(self) -> Iterator[int]:
        g = random.Random(self.seed + self.epoch)
        total = len(self.dataset)
        indices = list(range(total))

        if self.shuffle:
            g.shuffle(indices)

        # Pad or truncate to total_size
        if self.total_size > total:
            # Repeat indices to fill
            padding = self.total_size - total
            indices += indices[:padding]
        else:
            indices = indices[: self.total_size]

        # Subsample for this rank
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples
        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples


class WeightedRandomSampler(Sampler[int]):
    """Samples elements according to given probabilities (weights).

    Elements are drawn from ``[0, len(weights))`` with probability
    proportional to *weights*.  Supports sampling with or without
    replacement.

    Parameters
    ----------
    weights : Sequence[float]
        Non-negative sampling weights (one per dataset element).
    num_samples : int
        Number of samples to draw.
    replacement : bool, optional
        If ``True``, draw with replacement.  Default: ``True``.
    seed : int, optional
        Random seed for reproducibility.  Default: ``0``.

    Example
    -------
    >>> sampler = WeightedRandomSampler([0.1, 0.9, 0.5], num_samples=5)
    >>> list(sampler)
    [1, 2, 1, 1, 2]
    """

    def __init__(
        self,
        weights: Sequence[float],
        num_samples: int,
        replacement: bool = True,
        seed: int = 0,
    ):
        if not weights:
            raise ValueError("weights must not be empty")
        if any(w < 0 for w in weights):
            raise ValueError("weights must be non-negative")
        if not replacement and num_samples > len(weights):
            raise ValueError(
                "num_samples must be <= len(weights) when sampling without replacement"
            )
        self.weights = list(weights)
        self.num_samples = num_samples
        self.replacement = replacement
        self.seed = seed

    def __iter__(self) -> Iterator[int]:
        g = random.Random(self.seed)
        if self.replacement:
            indices = g.choices(
                range(len(self.weights)), weights=self.weights, k=self.num_samples
            )
        else:
            # Weighted sampling without replacement — audit item I.10.
            # Previous implementation was O(N²) because each draw did a
            # linear pool.index(chosen) lookup followed by pool.pop(idx)
            # (also O(N)).  Use Efraimidis–Spirakis (2006) instead: assign
            # each item a key u^(1/w), sort descending by key, take the
            # top num_samples.  O(N log N), single pass.  Equivalent
            # distribution to repeated weighted draw-without-replacement.
            n = len(self.weights)
            # Generate keys u_i^(1/w_i).  Items with w=0 get key 0 so
            # they never compete with positive-weight items.
            keys = []
            for i in range(n):
                w_i = self.weights[i]
                if w_i <= 0.0:
                    keys.append((0.0, i))
                else:
                    u = g.random()
                    # u is in (0, 1); take key as u**(1.0/w_i)
                    keys.append((u ** (1.0 / w_i), i))
            keys.sort(key=lambda t: t[0], reverse=True)
            indices = [i for _, i in keys[: self.num_samples]]
        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples


class SubsetRandomSampler(Sampler[int]):
    """Samples elements randomly from a given list of indices.

    Useful for creating train/validation splits or restricting sampling
    to a subset of the dataset.

    Parameters
    ----------
    indices : Sequence[int]
        Indices to sample from.
    seed : int, optional
        Random seed for reproducibility.  Default: ``0``.

    Example
    -------
    >>> sampler = SubsetRandomSampler([10, 20, 30, 40])
    >>> list(sampler)
    [30, 10, 40, 20]
    """

    def __init__(self, indices: Sequence[int], seed: int = 0):
        self.indices = list(indices)
        self.seed = seed

    def __iter__(self) -> Iterator[int]:
        g = random.Random(self.seed)
        shuffled = list(self.indices)
        g.shuffle(shuffled)
        return iter(shuffled)

    def __len__(self) -> int:
        return len(self.indices)


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------

_MAX_COLLATE_DEPTH = 32


def default_collate(batch: list) -> Any:
    """Default collate function that stacks tensors and groups tuples.

    Matches the contract of ``torch.utils.data.default_collate``:

    - List/tuple samples are collated element-wise.
    - Dict samples are collated key-wise (audit item E.12 — was missing).
    - Tensor / Variable samples are stacked along a new leading dim via
      the C++ stack op.
    - Other types (scalars, strings, …) are returned as a Python list.

    Shape/dtype mismatches inside the C++ stack now propagate as a
    RuntimeError instead of being silently swallowed (audit item E.12).
    The previous "swallow RuntimeError → return un-batched list" path
    hid real bugs in user datasets — failures are now visible.

    Recursion is bounded by ``_MAX_COLLATE_DEPTH`` (audit item M.7); a
    pathologically nested dict/tuple sample raises ``RecursionError``
    with an actionable message at the boundary instead of overflowing
    the Python stack.
    """
    return _default_collate_impl(batch, 0)


def _default_collate_impl(batch: list, _current_depth: int) -> Any:
    if _current_depth > _MAX_COLLATE_DEPTH:
        raise RecursionError(
            f"default_collate exceeded max nested-dict depth of "
            f"{_MAX_COLLATE_DEPTH} — flatten via collate_fn="
        )

    if not batch:
        return batch

    elem = batch[0]

    # Dict: collate each key's values independently.
    if isinstance(elem, dict):
        # FF.21: validate that every sample carries the same key set
        # before iterating. Pre-fix, the inner ``[d[k] for d in batch]``
        # comprehension only inspected ``elem.keys()`` (sample 0); a
        # sample with an extra key was silently dropped and a sample
        # missing a key raised an opaque ``KeyError`` from deep inside
        # the comprehension. Surface the mismatch up-front with the
        # offending sample's index so the user can fix the dataset.
        expected_keys = set(elem.keys())
        for i, d in enumerate(batch):
            if set(d.keys()) != expected_keys:
                raise ValueError(
                    f"default_collate: sample {i} has keys "
                    f"{sorted(d.keys())} but sample 0 has "
                    f"{sorted(expected_keys)} — all dict samples must "
                    f"have the same keys"
                )
        keys = elem.keys()
        return {
            k: _default_collate_impl([d[k] for d in batch], _current_depth + 1)
            for k in keys
        }

    # R.24: dataclass instance — iterate declared fields and rebuild via the
    # constructor kwargs. Without this branch, `type(elem)(collated)` raises
    # TypeError on dataclass samples (the dataclass __init__ does not accept a
    # single list of values).
    if dataclasses.is_dataclass(elem) and not isinstance(elem, type):
        field_names = [f.name for f in dataclasses.fields(elem)]
        collated_fields = {
            name: _default_collate_impl(
                [getattr(d, name) for d in batch], _current_depth + 1)
            for name in field_names
        }
        return type(elem)(**collated_fields)

    # R.24: NamedTuple — detect via _fields attribute and unpack positionally.
    # `type(elem)(collated)` would pass a single list as the only field.
    if isinstance(elem, tuple) and hasattr(elem, '_fields'):
        collated = [
            _default_collate_impl([d[i] for d in batch], _current_depth + 1)
            for i in range(len(elem))
        ]
        return type(elem)(*collated)

    # Tuple / list: collate each position independently.
    if isinstance(elem, (tuple, list)):
        collated = [
            _default_collate_impl([d[i] for d in batch], _current_depth + 1)
            for i in range(len(elem))
        ]
        return type(elem)(collated)

    # Tensors / Variables: stack along new leading axis.  Propagate any
    # RuntimeError (e.g. shape mismatch) so the caller sees the actual
    # error message instead of a silently-unbatched list.
    try:
        from . import tenzor_core as _core
    except ImportError:
        # No C++ core available — fall through to the scalar return path
        # below; tensor stacking is impossible without it.
        return batch
    if isinstance(elem, (_core.Tensor, _core.Variable)):
        return _core.stack(batch, 0)

    # Numeric scalars: stack into a 1-D tensor matching PyTorch's default
    # collate (int/bool -> int64, float -> float32).  Without this, the common
    # ``Dataset.__getitem__ -> (tensor, int_label)`` pattern left the labels as
    # a raw Python list, breaking downstream loss/metric calls.  `bool` is a
    # subclass of `int`, so the int branch covers it (cast to int64).
    if isinstance(elem, int):
        import numpy as _np
        return _core.Tensor.from_numpy(_np.asarray(batch, dtype=_np.int64))
    if isinstance(elem, float):
        import numpy as _np
        return _core.Tensor.from_numpy(_np.asarray(batch, dtype=_np.float32))

    # Fallback for strings / other Python objects.
    return batch


# ---------------------------------------------------------------------------
# Prefetch workers
# ---------------------------------------------------------------------------

class _PrefetchWorker:
    """Background thread that prefetches batches into a queue."""

    def __init__(
        self,
        iterable,
        queue_size: int = 2,
        worker_init_fn: Optional[Callable] = None,
        dataset: Any = None,
    ):
        self._iterable = iterable
        self._queue: Queue = Queue(maxsize=queue_size)
        self._sentinel = object()
        # Audit item Z.15: with num_workers=1 the fast path used a plain
        # thread but never invoked worker_init_fn nor populated
        # get_worker_info().  Multi-worker semantics now match for
        # num_workers in {1, >=2}.
        self._worker_init_fn = worker_init_fn
        self._dataset = dataset
        # BB.22: cooperative-stop flag so a consumer that bails out
        # mid-iteration (raise, break, GC) can unblock the producer
        # thread which would otherwise sit forever on a full queue.
        self._alive = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        # Install a per-thread WorkerInfo so user code that calls
        # get_worker_info() inside the dataset iterator sees the same
        # contract as the multi-process path.
        _worker_info_tls.info = WorkerInfo(
            id=0,
            num_workers=1,
            seed=0,
            dataset=self._dataset,
        )
        try:
            if self._worker_init_fn is not None:
                self._worker_init_fn(0)
            for item in self._iterable:
                # BB.22: bail out early if the consumer has detached.
                if not self._alive:
                    break
                self._queue.put(item)
        except Exception as e:
            # Only try to surface the error if we haven't been closed; a
            # closed queue may already have been drained and pushing
            # again is harmless but pointless.
            if self._alive:
                try:
                    self._queue.put(e)
                except Exception:
                    pass
        finally:
            try:
                self._queue.put(self._sentinel)
            except Exception:
                pass

    def __iter__(self):
        return self

    def __next__(self):
        item = self._queue.get()
        if item is self._sentinel:
            raise StopIteration
        if isinstance(item, Exception):
            raise item
        return item

    def close(self):
        """BB.22: cooperatively stop the producer thread.

        Clears ``self._alive`` so ``_run`` exits its loop on the next
        iteration, and drains the queue so any ``put`` blocked on a
        full queue unblocks immediately.
        """
        self._alive = False
        try:
            while True:
                self._queue.get_nowait()
        except Empty:
            pass

    def reset_for_new_epoch(self, iterable, dataset=None):
        """FF.20: prepare a persistent num_workers=1 worker for a new epoch.

        Drains any leftover items from the previous epoch, swaps in the
        fresh batch generator for the new epoch, and re-arms the
        cooperative-stop flag. The worker thread is restarted so the
        new ``iterable`` is consumed under the same ``_PrefetchWorker``
        identity (the same WorkerInfo and worker_init_fn semantics
        apply across epochs).
        """
        # Stop the previous epoch's producer thread cooperatively and
        # wait for it to finish before swapping the iterable.
        self._alive = False
        try:
            while True:
                self._queue.get_nowait()
        except Empty:
            pass
        if self._thread.is_alive():
            self._thread.join(timeout=5.0)
        # Reset state for the new epoch.
        self._iterable = iterable
        if dataset is not None:
            self._dataset = dataset
        # Re-create the queue so any stale sentinel from the previous
        # epoch is discarded.
        self._queue = Queue(maxsize=self._queue.maxsize)
        self._alive = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def __del__(self):
        try:
            self.close()
        except Exception:
            # Destructors must not raise.
            pass


_WORKER_SENTINEL = "__DONE__"


def _worker_loop(
    dataset,
    index_queue: mp.Queue,
    output_queue: mp.Queue,
    collate_fn: Callable,
    worker_id: int,
    num_workers: int,
    seed: int,
    worker_init_fn: Optional[Callable],
    rank: int = 0,
    world_size: int = 1,
    batch_size: int = 1,
):
    """Target function for each DataLoader worker process."""
    _worker_info_tls.info = WorkerInfo(
        id=worker_id, num_workers=num_workers, seed=seed, dataset=dataset,
        rank=rank, world_size=world_size,
    )

    # Ignore SIGINT in workers — let the main process handle it
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    # OO.13: convert SIGTERM (sent by _stop_workers' terminate path) into a
    # KeyboardInterrupt so the worker exits via the poison-pill `finally`
    # block (which always emits a _WORKER_SENTINEL).  Without this the
    # parent can deadlock on output_queue.get() when a worker is killed
    # mid-batch before it had a chance to push its sentinel.
    def _sigterm_handler(_signum, _frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, _sigterm_handler)

    if worker_init_fn is not None:
        worker_init_fn(worker_id)

    # Seed the worker's RNG for reproducibility
    random.seed(seed + worker_id)

    is_iterable = isinstance(dataset, IterableDataset)

    try:
        if is_iterable:
            # For IterableDataset, iterate and send batches directly
            batch = []
            for item in dataset:
                batch.append(item)
                # Check for poison pill
                try:
                    msg = index_queue.get_nowait()
                    if msg is None:
                        return
                except Empty:
                    # Audit item H.1: narrow the catch from bare Exception
                    # to the specific "queue is empty" case so real bugs
                    # (e.g. a corrupted queue handle, ImportError on a
                    # late-binding inside the worker) propagate to the
                    # main thread instead of being silently swallowed.
                    pass
                # HH.16: honour the DataLoader's configured batch_size for
                # IterableDataset workers (was hardcoded to flush every 1
                # item, so multi-worker iterable loaders ignored batch_size
                # entirely and the collate_fn saw size-1 batches).
                if len(batch) >= batch_size:
                    # seq_id is None for IterableDataset (order is inherently
                    # per-worker; the consumer yields these as they arrive).
                    output_queue.put((worker_id, None, collate_fn(batch)))
                    batch = []
            # Trailing remainder flush (e.g. dataset length not divisible
            # by batch_size).
            if batch:
                # seq_id is None for IterableDataset (order is inherently
                # per-worker; the consumer yields these as they arrive).
                output_queue.put((worker_id, None, collate_fn(batch)))
        else:
            # For map-style datasets, receive index batches and produce results
            while True:
                msg = index_queue.get()
                if msg is None:  # Poison pill
                    break
                # Map-style index batches are tagged with a monotonic seq_id by
                # the dispatcher so the consumer can yield in submission order
                # regardless of which worker finishes first.
                seq_id, batch_indices = msg
                batch = [dataset[i] for i in batch_indices]
                output_queue.put((worker_id, seq_id, collate_fn(batch)))
    except KeyboardInterrupt:
        # OO.13: SIGTERM-triggered shutdown — fall through to the sentinel
        # emit in `finally` so the parent's output_queue.get() never blocks.
        pass
    except Exception as e:
        output_queue.put((worker_id, None, e))
    finally:
        try:
            output_queue.put((worker_id, None, _WORKER_SENTINEL))
        except (ValueError, OSError):
            # OO.13: parent may have already closed the queue (post _stop_workers
            # → cancel_join_thread/close).  Nothing to do — the parent's
            # sentinel-or-timeout drain logic handles missing sentinels.
            pass


class _MultiProcessLoader:
    """Manages a pool of worker processes for parallel data loading."""

    def __init__(
        self,
        dataset,
        batch_sampler,
        collate_fn: Callable,
        num_workers: int,
        prefetch_factor: int,
        worker_init_fn: Optional[Callable],
        persistent_workers: bool,
        timeout: float,
        seed: int,
        rank: int = 0,
        world_size: int = 1,
        batch_size: int = 1,
    ):
        self._dataset = dataset
        self._batch_sampler = batch_sampler
        self._collate_fn = collate_fn
        self._num_workers = num_workers
        self._prefetch_factor = prefetch_factor
        self._worker_init_fn = worker_init_fn
        self._persistent = persistent_workers
        self._timeout = timeout
        self._seed = seed
        self._rank = rank
        self._world_size = world_size
        # HH.16: forwarded to _worker_loop so IterableDataset workers flush
        # at the user's configured batch_size instead of every 1 item.
        self._batch_size = batch_size
        self._workers: list[mp.Process] = []
        self._index_queues: list[mp.Queue] = []
        self._output_queue: Optional[mp.Queue] = None

    def _start_workers(self):
        # V.33 / Audit-11 QQ.17: fork() corrupts any GPU driver context in
        # the child (silent garbage tensors, hangs). We must probe **every**
        # GPU backend, not just CUDA — ROCm, OneAPI/L0, and Vulkan all share
        # the same driver-context-after-fork hazard. Earlier code only
        # probed CUDA, so ROCm/OneAPI/Vulkan users silently forked into a
        # corrupt driver state.
        #
        # Respect a user-set start method: if the caller already pinned the
        # start method via `multiprocessing.set_start_method(...)`, honour
        # it and warn if it conflicts with what GPU safety would have
        # chosen.
        gpu_active = False
        if hasattr(os, "fork"):
            try:
                from . import tenzor_core as _core  # type: ignore[attr-defined]
                probes = (
                    ("cuda_is_initialized", "cuda_is_available"),
                    ("rocm_is_initialized", "rocm_is_available"),
                    ("oneapi_is_initialized", "oneapi_is_available"),
                    ("vulkan_is_initialized", "vulkan_is_available"),
                )
                for init_name, avail_name in probes:
                    init_fn = getattr(_core, init_name, None)
                    avail_fn = getattr(_core, avail_name, None)
                    try:
                        if (init_fn is not None and bool(init_fn())) or (
                            avail_fn is not None and bool(avail_fn())
                        ):
                            gpu_active = True
                            break
                    except Exception:
                        # A probe that throws is treated as "not active";
                        # the next probe still runs.
                        continue
            except ImportError:
                pass

        user_method = mp.get_start_method(allow_none=True)
        if user_method is not None:
            ctx_name = user_method
            if gpu_active and user_method == "fork":
                import warnings as _warnings
                _warnings.warn(
                    "DataLoader: a GPU backend (CUDA/ROCm/OneAPI/Vulkan) "
                    "is active but multiprocessing start method is "
                    "explicitly set to 'fork' — child processes will see "
                    "a corrupted driver context (silent garbage tensors "
                    "or hangs). Use 'spawn' or 'forkserver' for GPU "
                    "workloads.",
                    RuntimeWarning,
                    stacklevel=2,
                )
        elif gpu_active:
            # forkserver inherits parent's GPU state on first spawn,
            # so spawn is the unambiguous safe choice.
            ctx_name = "spawn"
        else:
            ctx_name = "fork" if hasattr(os, "fork") else "spawn"
        ctx = mp.get_context(ctx_name)
        self._output_queue = ctx.Queue(maxsize=self._prefetch_factor * self._num_workers)
        self._index_queues = [ctx.Queue(maxsize=self._prefetch_factor) for _ in range(self._num_workers)]
        self._workers = []

        for wid in range(self._num_workers):
            w = ctx.Process(
                target=_worker_loop,
                args=(
                    self._dataset,
                    self._index_queues[wid],
                    self._output_queue,
                    self._collate_fn,
                    wid,
                    self._num_workers,
                    self._seed,
                    self._worker_init_fn,
                    self._rank,
                    self._world_size,
                    self._batch_size,
                ),
                daemon=True,
            )
            w.start()
            self._workers.append(w)

    def _stop_workers(self):
        for q in self._index_queues:
            try:
                q.put_nowait(None)
            except (Full, ValueError, OSError):
                # OO.13: widen from bare `Full`.  After a previous
                # _stop_workers (e.g. on shutdown error path) the underlying
                # queue may already be closed → put_nowait raises ValueError
                # ("Queue is closed") or OSError ("handle is closed"). Treat
                # those identically to Full — the worker will see SIGTERM via
                # terminate() below if it hasn't already exited.
                pass
        for w in self._workers:
            w.join(timeout=5)
            if w.is_alive():
                w.terminate()
                # OO.13: terminate is asynchronous; wait for the worker to
                # actually exit so its multiprocessing.Queue cleanup runs
                # and our subsequent cancel_join_thread / close don't race
                # against a still-publishing producer.
                w.join(timeout=2.0)
            # audit-10 OO.11: reclaim the Process object's bookkeeping fds.
            # Python ≥3.7 exposes Process.close(); it raises ValueError if
            # the worker is still alive (defensive — should not happen after
            # the join above, but guard anyway so a stuck worker doesn't
            # take the whole loader down).
            if not w.is_alive():
                try:
                    w.close()
                except (AttributeError, ValueError):
                    pass
        self._workers.clear()
        # OO.13: release any leaked fds / semaphores from the output queue.
        # cancel_join_thread() lets us exit without blocking on a stuck
        # background feeder thread; close() releases the pipe handles.
        if self._output_queue is not None:
            try:
                self._output_queue.cancel_join_thread()
            except (AttributeError, OSError, ValueError):
                pass
            try:
                self._output_queue.close()
            except (AttributeError, OSError, ValueError):
                pass
            # Drop the reference so the next __iter__ falls into
            # _start_workers (which allocates a fresh queue) rather than
            # touching the closed handle.
            self._output_queue = None
        # audit-10 OO.11: also tear down the per-worker index queues so
        # their feeder threads exit and the underlying pipe / semaphore fds
        # are released. Without this, every epoch (or every loader created
        # and discarded) leaks num_workers Queue handles.
        for iq in self._index_queues:
            try:
                iq.cancel_join_thread()
            except (AttributeError, OSError, ValueError):
                pass
            try:
                iq.close()
            except (AttributeError, OSError, ValueError):
                pass
            try:
                iq.join_thread()
            except (AttributeError, OSError, ValueError, RuntimeError):
                pass
        self._index_queues = []

    def __iter__(self):
        if not self._workers:
            self._start_workers()
        else:
            # W.19: with persistent_workers=True, leftover items from a prior
            # epoch (early break / exception) can survive in the index / output
            # queues. Drain both before sending new indices, otherwise the next
            # epoch reissues those stale batches or sees the previous epoch's
            # tail as the new epoch's head.
            for q in self._index_queues:
                while True:
                    try:
                        q.get_nowait()
                    except Empty:
                        break
            while True:
                try:
                    self._output_queue.get_nowait()
                except Empty:
                    break

        is_iterable = isinstance(self._dataset, IterableDataset)

        if is_iterable:
            # IterableDataset: workers push results, we collect
            done_count = 0
            while done_count < self._num_workers:
                try:
                    wid, _seq, result = self._output_queue.get(timeout=self._timeout)
                except Empty:
                    raise RuntimeError("DataLoader worker timed out")
                if result is _WORKER_SENTINEL:
                    done_count += 1
                    continue
                if isinstance(result, Exception):
                    self._stop_workers()
                    raise result
                yield result
        else:
            # Map-style: distribute index batches round-robin
            batches_iter = iter(self._batch_sampler)
            total_batches = len(self._batch_sampler)
            sent = 0
            received = 0

            # Pre-fill queues. Each batch carries a monotonic seq_id (== the
            # order it was submitted) so the consumer can restore submission
            # order below even when workers finish out of order.
            for wid in range(self._num_workers):
                for _ in range(self._prefetch_factor):
                    try:
                        indices = next(batches_iter)
                        self._index_queues[wid].put((sent, indices))
                        sent += 1
                    except StopIteration:
                        break

            done_workers = 0
            # Reorder buffer: results may arrive out of submission order across
            # workers. Buffer by seq_id and yield strictly in increasing order
            # (PyTorch _rcvd_idx / _task_info semantics) so a shuffle=False
            # loader produces deterministic, submission-ordered batches.
            reorder_buffer = {}
            next_to_yield = 0
            while received < total_batches:
                try:
                    wid, seq_id, result = self._output_queue.get(timeout=self._timeout)
                except Empty:
                    raise RuntimeError("DataLoader worker timed out")

                if result is _WORKER_SENTINEL:
                    done_workers += 1
                    continue
                if isinstance(result, Exception):
                    self._stop_workers()
                    raise result

                received += 1

                reorder_buffer[seq_id] = result
                while next_to_yield in reorder_buffer:
                    yield reorder_buffer.pop(next_to_yield)
                    next_to_yield += 1

                # Send more work to keep workers busy
                if sent < total_batches:
                    try:
                        indices = next(batches_iter)
                        self._index_queues[wid].put((sent, indices))
                        sent += 1
                    except StopIteration:
                        pass

            # Signal workers to stop after this epoch
            if not self._persistent:
                self._stop_workers()

    def __del__(self):
        self._stop_workers()


# ---------------------------------------------------------------------------
# DataLoader
# ---------------------------------------------------------------------------

def _pin_memory_batch(batch):
    """Recursively copy tensors in a batch to pinned (page-locked) memory."""
    # Lazy import to avoid circular dependency at module level
    try:
        from . import tenzor_core as _core
        _has_core = True
    except ImportError:
        _has_core = False

    if not _has_core:
        return batch

    if isinstance(batch, _core.Tensor):
        # R.25: M.8 guarantees pin_memory exists when the DataLoader was constructed
        # with pin_memory=True. The previous hasattr ternary silently returned the
        # batch unpinned — bypassing the constructor-time hard error.
        return _core.pin_memory(batch)
    elif isinstance(batch, _core.Variable):
        # Tensor/Variable merge: batches are typically requires_grad=False
        # Variables now; pin the underlying tensor, preserve the flag.
        return _core.Variable(_core.pin_memory(batch.tensor()), batch.requires_grad)
    elif isinstance(batch, (tuple, list)):
        pinned = [_pin_memory_batch(item) for item in batch]
        return type(batch)(pinned)
    elif isinstance(batch, dict):
        return {k: _pin_memory_batch(v) for k, v in batch.items()}
    return batch


def _wrap_pin_memory(iterator, pin: bool):
    """Wrap an iterator to pin each yielded batch."""
    if not pin:
        yield from iterator
    else:
        for batch in iterator:
            yield _pin_memory_batch(batch)


class DataLoader(Generic[T_co]):
    """Combines a dataset and a sampler to provide an iterable over batches.

    Parameters
    ----------
    dataset : Dataset or IterableDataset
        Dataset from which to load data.
    batch_size : int, optional
        Number of samples per batch.  Default: ``1``.
    shuffle : bool, optional
        If ``True``, reshuffle data at every epoch.  Default: ``False``.
    sampler : Sampler or None, optional
        Custom sampler.  If provided, *shuffle* must be ``False``.
    batch_sampler : BatchSampler or None, optional
        Custom batch sampler.  Mutually exclusive with *batch_size*,
        *shuffle*, *sampler*, and *drop_last*.
    num_workers : int, optional
        Number of subprocesses for data loading.  ``0`` means data is
        loaded in the main process.  Default: ``0``.
    collate_fn : callable or None, optional
        Function to merge a list of samples into a batch.  Defaults to
        :func:`default_collate`.
    drop_last : bool, optional
        If ``True``, drop the last incomplete batch.  Default: ``False``.
    prefetch_factor : int, optional
        Number of batches to prefetch per worker.  Must be ``>= 1``.
        Default: ``2``.
    worker_init_fn : callable or None, optional
        Called with ``worker_id`` at the start of each worker process.
    persistent_workers : bool, optional
        If ``True``, worker processes are not shut down between epochs.
        Default: ``False``.
    timeout : float, optional
        Timeout in seconds for collecting a batch from workers.
        Default: ``60``.
    pin_memory : bool, optional
        If ``True``, copy tensors to pinned (page-locked) memory after
        collation for faster CPU-to-GPU transfer.  Default: ``False``.

    Example
    -------
    >>> loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)
    >>> for batch in loader:
    ...     output = model(batch)
    """

    def __init__(
        self,
        dataset,
        batch_size: int = 1,
        shuffle: bool = False,
        sampler: Optional[Sampler] = None,
        batch_sampler: Optional[BatchSampler] = None,
        num_workers: int = 0,
        collate_fn: Optional[Callable] = None,
        drop_last: bool = False,
        prefetch_factor: int = 2,
        worker_init_fn: Optional[Callable] = None,
        persistent_workers: bool = False,
        timeout: float = 60.0,
        pin_memory: bool = False,
        rank: int = 0,
        world_size: int = 1,
    ):
        self.dataset = dataset
        self.collate_fn = collate_fn or default_collate
        self.num_workers = num_workers
        # Audit item Z.16: prefetch_factor=0 caused unbounded queues (maxsize=0
        # in mp.Queue means *no limit*) and the prefill loop ran zero iterations,
        # producing a DataLoader that buffered the entire dataset in memory and
        # only managed work via worker-side blocking. Reject up-front instead of
        # silently degrading behaviour.
        if prefetch_factor < 1:
            raise ValueError(
                f"prefetch_factor must be >= 1, got {prefetch_factor}"
            )
        self.prefetch_factor = prefetch_factor
        self.worker_init_fn = worker_init_fn
        self.persistent_workers = persistent_workers
        self.timeout = timeout
        # Audit items E.13 / M.8: pin_memory=True silently no-op'd when
        # the C++ backend lacked a `pin_memory` symbol (non-CUDA build).
        # Users who asked for pinned memory lost the async-transfer
        # guarantees without any warning.  Validate up-front so the
        # constructor fails loudly instead of producing a silently
        # pageable loader.
        if pin_memory:
            try:
                from . import tenzor_core as _core_pm
            except ImportError as e:
                raise RuntimeError(
                    "pin_memory=True requires a CUDA-enabled Tenzor build "
                    "(TENZOR_USE_CUDA=ON); set pin_memory=False or rebuild"
                ) from e
            if not hasattr(_core_pm, "pin_memory"):
                raise RuntimeError(
                    "pin_memory=True requires a CUDA-enabled Tenzor build "
                    "(TENZOR_USE_CUDA=ON); set pin_memory=False or rebuild"
                )
        self.pin_memory = pin_memory
        self.rank = rank
        self.world_size = world_size
        self._multiprocess_loader: Optional[_MultiProcessLoader] = None
        # FF.20: persistent single-thread worker for num_workers=1 +
        # persistent_workers=True. Pre-fix, every __iter__ call built a
        # fresh _PrefetchWorker even when persistent_workers=True, so
        # the persistent_workers flag was silently ignored on the
        # single-worker fast path. We initialise lazily on first
        # __iter__ so the worker_init_fn / WorkerInfo state is only
        # created when actually iterating.
        self._persistent_worker: Optional[_PrefetchWorker] = None

        is_iterable = isinstance(dataset, IterableDataset)

        if batch_sampler is not None:
            if batch_size != 1 or shuffle or sampler is not None or drop_last:
                raise ValueError(
                    "batch_sampler is mutually exclusive with batch_size, "
                    "shuffle, sampler, and drop_last"
                )
            self.batch_sampler = batch_sampler
            self.batch_size = None
        elif is_iterable:
            self.batch_sampler = None
            self.batch_size = batch_size
        else:
            if sampler is not None and shuffle:
                raise ValueError("sampler and shuffle are mutually exclusive")

            if sampler is not None:
                self._sampler = sampler
            elif shuffle:
                self._sampler = RandomSampler(dataset)
            else:
                self._sampler = SequentialSampler(dataset)

            self.batch_size = batch_size
            self.batch_sampler = BatchSampler(self._sampler, batch_size, drop_last)

    def _generate_batches(self) -> Iterator:
        """Yield collated batches from the dataset."""
        if isinstance(self.dataset, IterableDataset):
            batch = []
            for item in self.dataset:
                batch.append(item)
                if len(batch) == self.batch_size:
                    yield self.collate_fn(batch)
                    batch = []
            if batch:
                yield self.collate_fn(batch)
        else:
            for batch_indices in self.batch_sampler:
                batch = [self.dataset[i] for i in batch_indices]
                yield self.collate_fn(batch)

    def __iter__(self) -> Iterator[T_co]:
        if self.num_workers <= 0:
            it = self._generate_batches()
            return _wrap_pin_memory(it, self.pin_memory)

        if self.num_workers == 1 and not isinstance(self.dataset, IterableDataset):
            # Single prefetch thread (fast path, avoids process overhead).
            # Audit item Z.15: forward worker_init_fn and dataset so the
            # thread populates WorkerInfo and runs user-supplied init code,
            # matching the multi-process path's contract.
            #
            # FF.20: when ``persistent_workers=True`` reuse a single
            # ``_PrefetchWorker`` across epochs. The legacy code always
            # built a fresh worker per __iter__ call, so the
            # ``persistent_workers`` flag was silently dropped on this
            # fast path (worker_init_fn would re-run every epoch). The
            # persistent worker is now created on first iteration and
            # reset (queue drained, batch sampler re-iterated) for each
            # subsequent epoch.
            if self.persistent_workers:
                if self._persistent_worker is None:
                    self._persistent_worker = _PrefetchWorker(
                        self._generate_batches(),
                        queue_size=self.prefetch_factor,
                        worker_init_fn=self.worker_init_fn,
                        dataset=self.dataset,
                    )
                else:
                    self._persistent_worker.reset_for_new_epoch(
                        self._generate_batches(),
                        dataset=self.dataset,
                    )
                return _wrap_pin_memory(self._persistent_worker, self.pin_memory)

            it = _PrefetchWorker(
                self._generate_batches(),
                queue_size=self.prefetch_factor,
                worker_init_fn=self.worker_init_fn,
                dataset=self.dataset,
            )
            return _wrap_pin_memory(it, self.pin_memory)

        # Multi-process loading
        if self._multiprocess_loader is None or not self.persistent_workers:
            self._multiprocess_loader = _MultiProcessLoader(
                dataset=self.dataset,
                batch_sampler=self.batch_sampler,
                collate_fn=self.collate_fn,
                num_workers=self.num_workers,
                prefetch_factor=self.prefetch_factor,
                worker_init_fn=self.worker_init_fn,
                persistent_workers=self.persistent_workers,
                timeout=self.timeout,
                seed=random.randint(0, 2**31),
                rank=self.rank,
                world_size=self.world_size,
                # HH.16: pass user-configured batch_size so IterableDataset
                # workers flush at the right granularity. Map-style loaders
                # use batch_sampler and ignore this.
                batch_size=(self.batch_size if self.batch_size is not None else 1),
            )
        return _wrap_pin_memory(iter(self._multiprocess_loader), self.pin_memory)

    def __len__(self) -> int:
        if self.batch_sampler is not None:
            return len(self.batch_sampler)
        raise TypeError("DataLoader with IterableDataset has no len()")

    def close(self) -> None:
        """FF.20: release the persistent single-thread worker (if any)
        and tear down the multi-process worker pool.

        Safe to call multiple times. Called automatically by ``__del__``
        but exposed so user code can release worker resources eagerly
        (e.g. between training and evaluation phases).
        """
        if self._persistent_worker is not None:
            try:
                self._persistent_worker.close()
            except Exception:
                pass
            self._persistent_worker = None
        if self._multiprocess_loader is not None:
            self._multiprocess_loader._stop_workers()
            self._multiprocess_loader = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            # Destructors must not raise.
            pass


# ---------------------------------------------------------------------------
# Convenience: random_split
# ---------------------------------------------------------------------------

def random_split(dataset: Dataset, lengths: Sequence[int],
                 generator: Optional[random.Random] = None) -> list[Subset]:
    """Randomly split a dataset into non-overlapping subsets.

    Parameters
    ----------
    dataset : Dataset
        Dataset to split.
    lengths : Sequence[int]
        Lengths of each split.  Must sum to ``len(dataset)``.
    generator : random.Random or None, optional
        RNG for reproducibility.

    Returns
    -------
    list[Subset]
        List of dataset subsets.

    Example
    -------
    >>> train, val = random_split(dataset, [800, 200])
    >>> train, val = random_split(dataset, [0.8, 0.2])  # fractions also work
    """
    # PyTorch-style fractional lengths: if every entry is a float and they sum
    # to ~1, convert to integer counts (floor each, then distribute the
    # remainder one item at a time, in order) so the splits cover the dataset.
    if len(lengths) > 0 and all(isinstance(length, float) for length in lengths) \
            and math.isclose(sum(lengths), 1.0, abs_tol=1e-9):
        n = len(dataset)
        int_lengths = [int(math.floor(n * frac)) for frac in lengths]
        remainder = n - sum(int_lengths)
        for i in range(remainder):
            int_lengths[i % len(int_lengths)] += 1
        lengths = int_lengths

    total = sum(lengths)
    if total != len(dataset):
        raise ValueError(
            f"Sum of lengths ({total}) does not equal dataset size ({len(dataset)})"
        )
    g = generator or random.Random()
    indices = list(range(len(dataset)))
    g.shuffle(indices)

    subsets = []
    offset = 0
    for length in lengths:
        subsets.append(Subset(dataset, indices[offset : offset + length]))
        offset += length
    return subsets


__all__ = [
    "Dataset",
    "IterableDataset",
    "WorkerInfo",
    "get_worker_info",
    "set_worker_info",
    "clear_worker_info",
    "TensorDataset",
    "Subset",
    "ConcatDataset",
    # X.9: MapDataset is re-exported from the C++ binding at the top of this
    # module; declare it here so check_pyi_drift sees the runtime symbol that
    # matches the .pyi stub.
    "MapDataset",
    "Sampler",
    "SequentialSampler",
    "RandomSampler",
    "BatchSampler",
    "DistributedSampler",
    "WeightedRandomSampler",
    "SubsetRandomSampler",
    "DataLoader",
    "default_collate",
    "random_split",
]
