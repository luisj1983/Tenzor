"""Type stubs for tenzor.distributed (CC.11).

`tenzor.distributed` is a pybind11 submodule exposed by `tenzor_core`
(see python/bindings/bindings_distributed.cpp). There is no Python-level
`distributed.py` to import — IDE/type-checker support depends on this stub.

The check_pyi_drift.py tool only diffs .pyi files against same-stem .py
modules; this stub has no .py sibling, so the drift tool reports 0 drift
on it by construction. Signatures here mirror the docstrings emitted by
pybind11 (verified via `help(tz.distributed.<name>)` against the running
binding).
"""

from __future__ import annotations
from enum import Enum
from typing import Any, Optional, Sequence
from tenzor import Tensor


# ---------------------------------------------------------------------------
# Reduce operations
# ---------------------------------------------------------------------------

class ReduceOp(Enum):
    SUM = 0
    PRODUCT = 1
    MIN = 2
    MAX = 3
    AVG = 4


SUM: ReduceOp
PRODUCT: ReduceOp
MIN: ReduceOp
MAX: ReduceOp
AVG: ReduceOp


# ---------------------------------------------------------------------------
# Process group lifecycle
# ---------------------------------------------------------------------------

class ProcessGroup:
    """Opaque handle to a distributed process group."""
    def rank(self) -> int: ...
    def world_size(self) -> int: ...


def init_process_group(
    backend: str = "nccl",
    rank: int = -1,
    world_size: int = -1,
    master_addr: str = "localhost",
    master_port: int = 29500,
) -> None:
    """Initialize the default distributed process group."""
    ...


def destroy_process_group() -> None:
    """Destroy the default process group and release resources."""
    ...


def get_process_group() -> ProcessGroup:
    """Return the process group initialized by init_process_group()."""
    ...


def is_initialized() -> bool:
    """Whether the default process group has been initialized."""
    ...


def get_rank() -> int:
    """Rank of the current process within the default process group."""
    ...


def get_world_size() -> int:
    """Total number of processes in the default process group."""
    ...


# ---------------------------------------------------------------------------
# Collective primitives
# ---------------------------------------------------------------------------

def all_reduce(tensor: Tensor, op: ReduceOp = ReduceOp.SUM) -> None:
    """In-place all-reduce across the default process group."""
    ...


def broadcast(tensor: Tensor, src_rank: int = 0) -> None:
    """In-place broadcast from `src_rank` to all other ranks."""
    ...


def barrier() -> None:
    """Synchronize all processes in the default group."""
    ...


# ---------------------------------------------------------------------------
# Parallel layers and high-level wrappers (pybind11 classes; opaque
# constructors documented only by the C++ binding).
# ---------------------------------------------------------------------------

class DistributedDataParallel:
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...


class ShardingStrategy(Enum):
    NO_SHARD = 0
    SHARD_GRAD_OP = 1
    FULL_SHARD = 2
    HYBRID_SHARD = 3


class FSDPConfig:
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...


class FullyShardedDataParallel:
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...


class ColumnParallelLinear:
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...


class RowParallelLinear:
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...


class ParallelAttention:
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...


class SequenceParallel:
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...


class PipelineStage:
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...


class CompressedGradient:
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...


class FP16Compressor:
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def compress(self, tensor: Tensor) -> CompressedGradient: ...
    def decompress(self, compressed: CompressedGradient) -> Tensor: ...


class TopKCompressor:
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def compress(self, tensor: Tensor) -> CompressedGradient: ...
    def decompress(self, compressed: CompressedGradient) -> Tensor: ...


# ---------------------------------------------------------------------------
# Nested rpc submodule (pybind11 sub-submodule). Detailed surface is
# documented in the rpc-specific binding; this stub just exposes the
# module attribute so static checkers don't flag `tz.distributed.rpc`.
# ---------------------------------------------------------------------------

class _RpcSubmodule:
    """Stub for the nested `tenzor.distributed.rpc` pybind11 submodule.

    The detailed binding surface lives in C++; this placeholder lets type
    checkers resolve `tz.distributed.rpc` without flagging it as missing.
    """
    def __getattr__(self, name: str) -> Any: ...


rpc: _RpcSubmodule
