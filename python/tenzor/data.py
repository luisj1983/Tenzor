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

    # Fallback for scalars / strings / other Python objects.
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
                self._queue.put(item)
        except Exception as e:
            self._queue.put(e)
        finally:
            self._queue.put(self._sentinel)

    def __iter__(self):
        return self

    def __next__(self):
        item = self._queue.get()
        if item is self._sentinel:
            raise StopIteration
        if isinstance(item, Exception):
            raise item
        return item


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
):
    """Target function for each DataLoader worker process."""
    _worker_info_tls.info = WorkerInfo(
        id=worker_id, num_workers=num_workers, seed=seed, dataset=dataset,
        rank=rank, world_size=world_size,
    )

    # Ignore SIGINT in workers — let the main process handle it
    signal.signal(signal.SIGINT, signal.SIG_IGN)

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
                if len(batch) >= 1:
                    output_queue.put((worker_id, collate_fn(batch)))
                    batch = []
            if batch:
                output_queue.put((worker_id, collate_fn(batch)))
        else:
            # For map-style datasets, receive index batches and produce results
            while True:
                msg = index_queue.get()
                if msg is None:  # Poison pill
                    break
                batch_indices = msg
                batch = [dataset[i] for i in batch_indices]
                output_queue.put((worker_id, collate_fn(batch)))
    except Exception as e:
        output_queue.put((worker_id, e))
    finally:
        output_queue.put((worker_id, _WORKER_SENTINEL))


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
        self._workers: list[mp.Process] = []
        self._index_queues: list[mp.Queue] = []
        self._output_queue: Optional[mp.Queue] = None

    def _start_workers(self):
        # V.33: fork() corrupts the CUDA driver context in the child
        # (silent garbage tensors, hangs).  When a non-CPU backend is live,
        # fall back to spawn (or forkserver where available — spawn is the
        # safest default).  We probe tenzor_core for an explicit query;
        # absent that, any CUDA-available system also forces spawn.
        ctx_name = "fork" if hasattr(os, "fork") else "spawn"
        if hasattr(os, "fork"):
            try:
                from . import tenzor_core as _core  # type: ignore[attr-defined]
                cuda_active = bool(
                    getattr(_core, "cuda_is_initialized", lambda: False)()
                    or getattr(_core, "cuda_is_available", lambda: False)()
                )
                if cuda_active:
                    # forkserver inherits parent's CUDA state on first spawn,
                    # so spawn is the unambiguous safe choice.
                    ctx_name = "spawn"
            except ImportError:
                pass
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
                ),
                daemon=True,
            )
            w.start()
            self._workers.append(w)

    def _stop_workers(self):
        for q in self._index_queues:
            try:
                q.put_nowait(None)
            except Full:
                # Queue is full and the worker is busy; the worker will
                # see the next poison-pill on the subsequent iteration.
                # (Audit item H.1 — was bare `except Exception: pass`.)
                pass
        for w in self._workers:
            w.join(timeout=5)
            if w.is_alive():
                w.terminate()
        self._workers.clear()

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
                    wid, result = self._output_queue.get(timeout=self._timeout)
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

            # Pre-fill queues
            for wid in range(self._num_workers):
                for _ in range(self._prefetch_factor):
                    try:
                        indices = next(batches_iter)
                        self._index_queues[wid].put(indices)
                        sent += 1
                    except StopIteration:
                        break

            done_workers = 0
            while received < total_batches:
                try:
                    wid, result = self._output_queue.get(timeout=self._timeout)
                except Empty:
                    raise RuntimeError("DataLoader worker timed out")

                if result is _WORKER_SENTINEL:
                    done_workers += 1
                    continue
                if isinstance(result, Exception):
                    self._stop_workers()
                    raise result

                received += 1
                yield result

                # Send more work to keep workers busy
                if sent < total_batches:
                    try:
                        indices = next(batches_iter)
                        self._index_queues[wid].put(indices)
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
            )
        return _wrap_pin_memory(iter(self._multiprocess_loader), self.pin_memory)

    def __len__(self) -> int:
        if self.batch_sampler is not None:
            return len(self.batch_sampler)
        raise TypeError("DataLoader with IterableDataset has no len()")

    def __del__(self):
        if self._multiprocess_loader is not None:
            self._multiprocess_loader._stop_workers()


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
    """
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
