"""Tests for the Python pruning bindings (tz.compression).

C++ tests for pruning live in tests/test_pruning.cpp; this file validates the
Python-exposed surface (`prune_unstructured`, `prune_iterative`, `prune_channels`,
`apply_pruning_masks`, `compute_sparsity`) using a tiny Linear model.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../build/python'))
import tenzor as tz
tz.initialize()

import pytest


@pytest.fixture
def small_model():
    """A two-Linear MLP with deterministic random weights."""
    tz.manual_seed(42)
    model = tz.nn.Sequential(
        tz.nn.Linear(16, 32),
        tz.nn.ReLU(),
        tz.nn.Linear(32, 8),
    )
    return model


def test_prune_unstructured_l1_50pct(small_model):
    """50% unstructured L1 pruning should approximately halve the live weights."""
    config = tz.compression.prune_unstructured(
        small_model,
        sparsity=0.5,
        criterion=tz.compression.ImportanceCriterion.L1,
    )
    assert config is not None
    tz.compression.apply_pruning_masks(small_model, config)
    sparsity = tz.compression.compute_sparsity(small_model)
    # Allow some tolerance — compute_sparsity may include or exclude bias.
    assert 0.4 <= sparsity <= 0.6, f"expected ~50% sparsity, got {sparsity}"


def test_prune_unstructured_zero_sparsity(small_model):
    """sparsity=0 should leave the model essentially unchanged."""
    sparsity_before = tz.compression.compute_sparsity(small_model)
    config = tz.compression.prune_unstructured(small_model, sparsity=0.0)
    tz.compression.apply_pruning_masks(small_model, config)
    sparsity_after = tz.compression.compute_sparsity(small_model)
    # Sparsity shouldn't grow when target is 0.
    assert sparsity_after <= sparsity_before + 1e-6


def test_prune_iterative_polynomial(small_model):
    """Iterative pruning API call must succeed and return a config object.

    Note: apply_pruning_masks(model, config) may not directly hit target
    sparsity — `prune_iterative` is designed to be called N times in a
    training loop with the schedule producing per-iteration sparsities.
    Just verify the API binding works and produces a usable config.
    """
    config = tz.compression.prune_iterative(
        small_model,
        target_sparsity=0.7,
        num_iterations=5,
        schedule=tz.compression.PruningSchedule.Polynomial,
        criterion=tz.compression.ImportanceCriterion.L1,
    )
    assert config is not None
    # Must be applyable without crashing.
    tz.compression.apply_pruning_masks(small_model, config)


def test_pruned_model_still_forwards(small_model):
    """A pruned model must still produce output of the right shape."""
    config = tz.compression.prune_unstructured(small_model, sparsity=0.5)
    tz.compression.apply_pruning_masks(small_model, config)
    x = tz.Variable(tz.randn([4, 16], dtype=tz.dtype.float32), False)
    y = small_model(x)
    assert y.tensor().shape == [4, 8]


def test_importance_criterion_enum_values():
    """All four ImportanceCriterion values must be reachable."""
    for c in (
        tz.compression.ImportanceCriterion.L1,
        tz.compression.ImportanceCriterion.L2,
        tz.compression.ImportanceCriterion.L1Norm,
        tz.compression.ImportanceCriterion.L2Norm,
    ):
        assert c is not None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-xvs"]))
