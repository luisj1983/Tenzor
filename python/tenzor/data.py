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

import math
import random
import threading
from abc import ABC, abstractmethod
from collections.abc import Iterator, Sized
from queue import Queue
from typing import Any, Callable, Generic, Optional, Sequence, TypeVar

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


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------

def default_collate(batch: list) -> Any:
    """Default collate function that stacks tensors and groups tuples.

    Parameters
    ----------
    batch : list
        List of samples from the dataset.

    Returns
    -------
    Any
        Collated batch.  If samples are tuples, returns a tuple of
        collated elements.  Otherwise returns a list.
    """
    if not batch:
        return batch

    elem = batch[0]

    # If elements are tuples/lists, collate each position independently
    if isinstance(elem, (tuple, list)):
        collated = [default_collate([d[i] for d in batch]) for i in range(len(elem))]
        return type(elem)(collated)

    # Try to stack tensors using the C++ stack operation
    try:
        from . import tenzor_core as _core
        if isinstance(elem, (_core.Tensor, _core.Variable)):
            return _core.stack(batch, 0)
    except (ImportError, AttributeError, RuntimeError):
        pass

    # Fallback: return as-is (list of scalars, strings, etc.)
    return batch


# ---------------------------------------------------------------------------
# Prefetch worker
# ---------------------------------------------------------------------------

class _PrefetchWorker:
    """Background thread that prefetches batches into a queue."""

    def __init__(self, iterable, queue_size: int = 2):
        self._iterable = iterable
        self._queue: Queue = Queue(maxsize=queue_size)
        self._sentinel = object()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        try:
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


# ---------------------------------------------------------------------------
# DataLoader
# ---------------------------------------------------------------------------

class DataLoader(Generic[T_co]):
    """Combines a dataset and a sampler to provide an iterable over batches.

    Parameters
    ----------
    dataset : Dataset
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
        Number of prefetch threads.  ``0`` means data is loaded in the
        main thread.  Currently only ``0`` (synchronous) and ``1``
        (single prefetch thread) are supported.  Default: ``0``.
    collate_fn : callable or None, optional
        Function to merge a list of samples into a batch.  Defaults to
        :func:`default_collate`.
    drop_last : bool, optional
        If ``True``, drop the last incomplete batch.  Default: ``False``.
    prefetch_factor : int, optional
        Number of batches to prefetch per worker.  Only used when
        *num_workers* > 0.  Default: ``2``.

    Example
    -------
    >>> loader = DataLoader(dataset, batch_size=64, shuffle=True)
    >>> for batch in loader:
    ...     output = model(batch)
    """

    def __init__(
        self,
        dataset: Dataset[T_co],
        batch_size: int = 1,
        shuffle: bool = False,
        sampler: Optional[Sampler] = None,
        batch_sampler: Optional[BatchSampler] = None,
        num_workers: int = 0,
        collate_fn: Optional[Callable] = None,
        drop_last: bool = False,
        prefetch_factor: int = 2,
    ):
        self.dataset = dataset
        self.collate_fn = collate_fn or default_collate
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor

        if batch_sampler is not None:
            # batch_sampler is mutually exclusive with these options
            if batch_size != 1 or shuffle or sampler is not None or drop_last:
                raise ValueError(
                    "batch_sampler is mutually exclusive with batch_size, "
                    "shuffle, sampler, and drop_last"
                )
            self.batch_sampler = batch_sampler
            self.batch_size = None
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
        for batch_indices in self.batch_sampler:
            batch = [self.dataset[i] for i in batch_indices]
            yield self.collate_fn(batch)

    def __iter__(self) -> Iterator[T_co]:
        gen = self._generate_batches()
        if self.num_workers > 0:
            return _PrefetchWorker(gen, queue_size=self.prefetch_factor)
        return gen

    def __len__(self) -> int:
        return len(self.batch_sampler)


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
    "TensorDataset",
    "Subset",
    "ConcatDataset",
    "Sampler",
    "SequentialSampler",
    "RandomSampler",
    "BatchSampler",
    "DistributedSampler",
    "DataLoader",
    "default_collate",
    "random_split",
]
