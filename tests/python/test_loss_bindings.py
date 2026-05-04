#!/usr/bin/env python3
"""
Test Python bindings for loss functions and Sequential container.
Tests all loss functions according to NEW_TODO.md Phase 1, Task 4.
"""

import sys
import os

# Add build directory to path for importing tenzor_core
build_python_dir = os.path.join(os.path.dirname(__file__), '../../build/python')
sys.path.insert(0, build_python_dir)

import pytest
import tenzor.tenzor_core as tz
np = pytest.importorskip("numpy")


def test_reduction_enum():
    """Test Reduction enum is properly exposed."""
    print("Testing Reduction enum...")

    # Test enum values exist
    assert hasattr(tz.nn, 'Reduction'), "Reduction enum not found"
    assert hasattr(tz.nn.Reduction, 'NONE'), "Reduction.NONE not found"
    assert hasattr(tz.nn.Reduction, 'MEAN'), "Reduction.MEAN not found"
    assert hasattr(tz.nn.Reduction, 'SUM'), "Reduction.SUM not found"

    print("✓ Reduction enum test passed")


def test_mse_loss():
    """Test MSELoss bindings."""
    print("\nTesting MSELoss...")

    # Initialize Tenzor
    tz.initialize()

    # Create loss with default reduction (mean)
    criterion = tz.nn.MSELoss()

    # Create loss with explicit reduction
    criterion_sum = tz.nn.MSELoss(tz.nn.Reduction.SUM)
    criterion_none = tz.nn.MSELoss(tz.nn.Reduction.NONE)

    # Create test data
    input_data = tz.Variable(tz.Tensor([2, 3], tz.dtype.float32))
    target_data = tz.Variable(tz.Tensor([2, 3], tz.dtype.float32))

    # Test forward method
    loss = criterion.forward(input_data, target_data)
    assert loss is not None, "MSELoss forward failed"

    # Test __call__ method
    loss_call = criterion(input_data, target_data)
    assert loss_call is not None, "MSELoss __call__ failed"

    print("✓ MSELoss test passed")


def test_cross_entropy_loss():
    """Test CrossEntropyLoss bindings."""
    print("\nTesting CrossEntropyLoss...")

    # Create loss
    criterion = tz.nn.CrossEntropyLoss()
    criterion_sum = tz.nn.CrossEntropyLoss(tz.nn.Reduction.SUM)

    # Create test data - input is logits, target is class indices
    input_data = tz.Variable(tz.Tensor([2, 3], tz.dtype.float32))  # batch_size=2, num_classes=3
    target_data = tz.Tensor([2], tz.dtype.int64)  # batch_size=2 class indices

    # Test forward and __call__
    loss = criterion.forward(input_data, target_data)
    assert loss is not None, "CrossEntropyLoss forward failed"

    loss_call = criterion(input_data, target_data)
    assert loss_call is not None, "CrossEntropyLoss __call__ failed"

    print("✓ CrossEntropyLoss test passed")


def test_bce_loss():
    """Test BCELoss bindings."""
    print("\nTesting BCELoss...")

    # Create loss
    criterion = tz.nn.BCELoss()
    criterion_sum = tz.nn.BCELoss(tz.nn.Reduction.SUM)
    criterion_none = tz.nn.BCELoss(tz.nn.Reduction.NONE)

    # Create test data - probabilities in [0, 1]
    input_data = tz.Variable(tz.Tensor([2, 3], tz.dtype.float32))
    target_data = tz.Variable(tz.Tensor([2, 3], tz.dtype.float32))

    # Test forward and __call__
    loss = criterion.forward(input_data, target_data)
    assert loss is not None, "BCELoss forward failed"

    loss_call = criterion(input_data, target_data)
    assert loss_call is not None, "BCELoss __call__ failed"

    print("✓ BCELoss test passed")


def test_bce_with_logits_loss():
    """Test BCEWithLogitsLoss bindings."""
    print("\nTesting BCEWithLogitsLoss...")

    # Create loss
    criterion = tz.nn.BCEWithLogitsLoss()
    criterion_sum = tz.nn.BCEWithLogitsLoss(tz.nn.Reduction.SUM)

    # Create test data - raw logits
    input_data = tz.Variable(tz.Tensor([2, 3], tz.dtype.float32))
    target_data = tz.Variable(tz.Tensor([2, 3], tz.dtype.float32))

    # Test forward and __call__
    loss = criterion.forward(input_data, target_data)
    assert loss is not None, "BCEWithLogitsLoss forward failed"

    loss_call = criterion(input_data, target_data)
    assert loss_call is not None, "BCEWithLogitsLoss __call__ failed"

    print("✓ BCEWithLogitsLoss test passed")


def test_nll_loss():
    """Test NLLLoss bindings."""
    print("\nTesting NLLLoss...")

    # Create loss
    criterion = tz.nn.NLLLoss()
    criterion_sum = tz.nn.NLLLoss(tz.nn.Reduction.SUM)
    criterion_none = tz.nn.NLLLoss(tz.nn.Reduction.NONE)

    # Create test data - log probabilities and class indices
    input_data = tz.Variable(tz.Tensor([2, 3], tz.dtype.float32))
    target_data = tz.Tensor([2], tz.dtype.int64)

    # Test forward and __call__
    loss = criterion.forward(input_data, target_data)
    assert loss is not None, "NLLLoss forward failed"

    loss_call = criterion(input_data, target_data)
    assert loss_call is not None, "NLLLoss __call__ failed"

    print("✓ NLLLoss test passed")


def test_l1_loss():
    """Test L1Loss bindings."""
    print("\nTesting L1Loss...")

    # Create loss
    criterion = tz.nn.L1Loss()
    criterion_sum = tz.nn.L1Loss(tz.nn.Reduction.SUM)
    criterion_none = tz.nn.L1Loss(tz.nn.Reduction.NONE)

    # Create test data
    input_data = tz.Variable(tz.Tensor([2, 3], tz.dtype.float32))
    target_data = tz.Variable(tz.Tensor([2, 3], tz.dtype.float32))

    # Test forward and __call__
    loss = criterion.forward(input_data, target_data)
    assert loss is not None, "L1Loss forward failed"

    loss_call = criterion(input_data, target_data)
    assert loss_call is not None, "L1Loss __call__ failed"

    print("✓ L1Loss test passed")


def test_smooth_l1_loss():
    """Test SmoothL1Loss bindings."""
    print("\nTesting SmoothL1Loss...")

    # Create loss with default beta
    criterion = tz.nn.SmoothL1Loss()

    # Create loss with custom beta
    criterion_beta = tz.nn.SmoothL1Loss(tz.nn.Reduction.MEAN, 0.5)

    # Create loss with sum reduction
    criterion_sum = tz.nn.SmoothL1Loss(tz.nn.Reduction.SUM, 1.0)

    # Create test data
    input_data = tz.Variable(tz.Tensor([2, 3], tz.dtype.float32))
    target_data = tz.Variable(tz.Tensor([2, 3], tz.dtype.float32))

    # Test forward and __call__
    loss = criterion.forward(input_data, target_data)
    assert loss is not None, "SmoothL1Loss forward failed"

    loss_call = criterion_beta(input_data, target_data)
    assert loss_call is not None, "SmoothL1Loss __call__ failed"

    print("✓ SmoothL1Loss test passed")


def test_sequential_empty():
    """Test Sequential container with default constructor."""
    print("\nTesting Sequential (empty constructor)...")

    # Create empty Sequential
    model = tz.nn.Sequential()
    assert model is not None, "Sequential creation failed"

    # Add modules
    linear1 = tz.nn.Linear(10, 20)
    model.add_module(linear1)

    relu = tz.nn.ReLU()
    model.add_module(relu)

    linear2 = tz.nn.Linear(20, 5)
    model.add_module(linear2)

    print("✓ Sequential (empty) test passed")


def test_sequential_variadic():
    """Test Sequential container with variadic constructor."""
    print("\nTesting Sequential (variadic constructor)...")

    # Create modules
    linear1 = tz.nn.Linear(10, 20)
    relu = tz.nn.ReLU()
    linear2 = tz.nn.Linear(20, 5)

    # Create Sequential with variadic constructor
    model = tz.nn.Sequential(linear1, relu, linear2)
    assert model is not None, "Sequential variadic creation failed"

    # Test forward pass
    input_data = tz.Variable(tz.Tensor([1, 10], tz.dtype.float32))
    output = model.forward(input_data)
    assert output is not None, "Sequential forward failed"

    # Check output shape (Tensor.shape is a property, not a method)
    output_shape = output.data.shape
    assert output_shape[0] == 1, f"Wrong batch size: {output_shape[0]}"
    assert output_shape[1] == 5, f"Wrong output size: {output_shape[1]}"

    print("✓ Sequential (variadic) test passed")


def test_sequential_with_loss():
    """Test Sequential container with a loss function."""
    print("\nTesting Sequential + Loss integration...")

    # Create model
    model = tz.nn.Sequential(
        tz.nn.Linear(10, 20),
        tz.nn.ReLU(),
        tz.nn.Linear(20, 3)
    )

    # Create loss
    criterion = tz.nn.CrossEntropyLoss()

    # Create dummy data
    input_data = tz.Variable(tz.Tensor([2, 10], tz.dtype.float32))
    target_data = tz.Tensor([2], tz.dtype.int64)

    # Forward pass
    logits = model.forward(input_data)

    # Compute loss
    loss = criterion(logits, target_data)
    assert loss is not None, "Loss computation failed"

    print("✓ Sequential + Loss integration test passed")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing Loss Functions and Sequential Container Bindings")
    print("=" * 60)

    try:
        # Test Reduction enum
        test_reduction_enum()

        # Test all loss functions
        test_mse_loss()
        test_cross_entropy_loss()
        test_bce_loss()
        test_bce_with_logits_loss()
        test_nll_loss()
        test_l1_loss()
        test_smooth_l1_loss()

        # Test Sequential container
        test_sequential_empty()
        test_sequential_variadic()
        test_sequential_with_loss()

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED ✓")
        print("=" * 60)
        return 0

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
