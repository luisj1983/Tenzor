"""Type stubs for tenzor.data module (data loading utilities)."""

from __future__ import annotations
from typing import Optional, List, Iterator, Callable, Any, Sequence
from tenzor import Tensor

class Dataset:
    """Abstract base class for datasets."""

    def __len__(self) -> int: ...
    def __getitem__(self, index: int) -> Any: ...

class MapDataset(Dataset):
    """Map-style dataset: __getitem__ maps an index to a sample."""
    ...

class IterableDataset(Dataset):
    """Iterable-style dataset: __iter__ yields samples."""

    def __iter__(self) -> Iterator: ...

class WorkerInfo:
    """Metadata describing the current DataLoader worker context."""

    id: int
    num_workers: int
    seed: int
    dataset: Any
    rank: int
    world_size: int

    def __init__(
        self,
        id: int,
        num_workers: int,
        seed: int,
        dataset: Any,
        rank: int = 0,
        world_size: int = 1,
    ) -> None: ...

def get_worker_info() -> Optional[WorkerInfo]: ...
def set_worker_info(info: Optional[WorkerInfo]) -> None: ...
def clear_worker_info() -> None: ...

class TensorDataset(Dataset):
    """Dataset wrapping a list of tensors. Each sample is a tuple of tensors."""

    def __init__(self, *tensors: Tensor) -> None: ...
    def __len__(self) -> int: ...
    def __getitem__(self, index: int) -> tuple: ...

class ConcatDataset(Dataset):
    """Concatenation of multiple datasets."""

    def __init__(self, datasets: List[Dataset]) -> None: ...
    def __len__(self) -> int: ...

class Sampler:
    """Base class for all samplers."""

    def __iter__(self) -> Iterator[int]: ...
    def __len__(self) -> int: ...

class SequentialSampler(Sampler):
    """Samples elements sequentially, always in the same order."""

    def __init__(self, data_source: Dataset) -> None: ...

class RandomSampler(Sampler):
    """Samples elements randomly."""

    def __init__(self, data_source: Dataset, replacement: bool = False,
                 num_samples: Optional[int] = None, generator: Optional[Any] = None) -> None: ...

class BatchSampler(Sampler):
    """Wraps another sampler to yield a mini-batch of indices."""

    def __init__(self, sampler: Sampler, batch_size: int, drop_last: bool = False) -> None: ...

class DistributedSampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset for distributed training."""

    def __init__(self, dataset: Dataset, num_replicas: Optional[int] = None,
                 rank: Optional[int] = None, shuffle: bool = True,
                 seed: int = 0, drop_last: bool = False) -> None: ...
    def set_epoch(self, epoch: int) -> None: ...

class WeightedRandomSampler(Sampler):
    """Samples elements according to given probabilities (weights)."""

    def __init__(self, weights: Sequence[float], num_samples: int,
                 replacement: bool = True, seed: int = 0) -> None: ...
    def __iter__(self) -> Iterator[int]: ...
    def __len__(self) -> int: ...

class SubsetRandomSampler(Sampler):
    """Samples elements randomly from a given list of indices."""

    def __init__(self, indices: Sequence[int], seed: int = 0) -> None: ...
    def __iter__(self) -> Iterator[int]: ...
    def __len__(self) -> int: ...

class DataLoader:
    """Data loader that combines a dataset and a sampler, providing an iterable over the dataset."""

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 1,
        shuffle: bool = False,
        sampler: Optional[Sampler] = None,
        batch_sampler: Optional[BatchSampler] = None,
        num_workers: int = 0,
        collate_fn: Optional[Callable] = None,
        pin_memory: bool = False,
        drop_last: bool = False,
        timeout: float = 0,
        worker_init_fn: Optional[Callable[[int], None]] = None,
        prefetch_factor: int = 2,
        persistent_workers: bool = False,
        rank: int = 0,
        world_size: int = 1,
    ) -> None: ...

    def __iter__(self) -> Iterator: ...
    def __len__(self) -> int: ...
