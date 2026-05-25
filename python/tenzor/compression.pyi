"""Type stubs for tenzor.compression (CC.11).

`tenzor.compression` is a pybind11 submodule exposed by `tenzor_core`
(model pruning + sparsity analysis). There is no Python-level
`compression.py`; signatures below mirror the docstrings emitted by
pybind11 (verified via `help(tz.compression.<name>)`).
"""

from __future__ import annotations
from enum import Enum
from typing import Any, Callable, Dict, List, Mapping, Sequence

from tenzor import Tensor
from tenzor import nn as _nn


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ImportanceCriterion(Enum):
    L1 = 0
    L2 = 1
    L1Norm = 2
    L2Norm = 3


class PruningSchedule(Enum):
    OneShot = 0
    Iterative = 1
    Polynomial = 2


# ---------------------------------------------------------------------------
# Data carriers
# ---------------------------------------------------------------------------

class PruningMask:
    layer_name: str
    mask: Tensor
    current_sparsity: float
    def apply(self, tensor: Tensor) -> Tensor: ...
    def compute_sparsity(self) -> float: ...


class PruningConfig:
    criterion: ImportanceCriterion
    schedule: PruningSchedule
    target_sparsity: float
    current_sparsity: float
    current_iteration: int
    num_iterations: int
    masks: List[PruningMask]
    def get_current_sparsity(self) -> float: ...


# ---------------------------------------------------------------------------
# Free functions
# ---------------------------------------------------------------------------

def compute_importance(
    weights: Tensor, criterion: ImportanceCriterion
) -> Tensor: ...


def create_mask_from_importance(importance: Tensor, sparsity: float) -> Tensor: ...


def prune_unstructured(
    module: _nn.Module,
    sparsity: float,
    criterion: ImportanceCriterion = ImportanceCriterion.L1,
    global_pruning: bool = False,
) -> PruningConfig: ...


def prune_iterative(
    module: _nn.Module,
    target_sparsity: float,
    num_iterations: int,
    schedule: PruningSchedule = PruningSchedule.Iterative,
    criterion: ImportanceCriterion = ImportanceCriterion.L1,
) -> PruningConfig: ...


def prune_channels(
    module: _nn.Module,
    sparsity: float,
    criterion: ImportanceCriterion = ImportanceCriterion.L1,
) -> _nn.Module: ...


def prune_filters(
    module: _nn.Module,
    sparsity: float,
    criterion: ImportanceCriterion = ImportanceCriterion.L1,
) -> _nn.Module: ...


def prune_layers(
    module: _nn.Module,
    num_layers: int,
    criterion: ImportanceCriterion = ImportanceCriterion.L1,
) -> _nn.Module: ...


def apply_pruning_masks(module: _nn.Module, config: PruningConfig) -> None: ...


def remove_pruning(module: _nn.Module, config: PruningConfig) -> None: ...


def finalize_pruning(module: _nn.Module, config: PruningConfig) -> _nn.Module: ...


def analyze_layer_sparsity(module: _nn.Module) -> Dict[str, float]: ...


def compute_sparsity(module: _nn.Module) -> float: ...


def compute_compression_ratio(
    original_module: _nn.Module, pruned_module: _nn.Module
) -> float: ...


def estimate_flops_reduction(
    module: _nn.Module, input_shape: Sequence[int]
) -> float: ...


def find_lottery_ticket(
    module: _nn.Module,
    initial_weights: Mapping[str, Tensor],
    target_sparsity: float,
    num_rounds: int,
) -> PruningConfig: ...


def sensitivity_analysis(
    module: _nn.Module,
    validation_fn: Callable[[_nn.Module], float],
    sparsity_levels: Sequence[float] = (0.1, 0.3, 0.5, 0.7, 0.9),
) -> Dict[str, List[float]]: ...
