#!/usr/bin/env python3
"""
Test script for DataLoader Python bindings.
Verifies that the DataLoader can be used from Python with all features working correctly.
"""

import sys
import numpy as np

# Add the build directory to Python path (relative to this file)
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'build', 'python'))

import tenzor.tenzor_core as tz

def test_tensor_dataset():
    """Test TensorDataset creation and access."""
    print("Testing TensorDataset...")

    # Create sample data
    inputs = tz.randn([100, 10])
    targets = tz.randn([100, 1])

    # Create dataset
    dataset = tz.data.TensorDataset(inputs, targets)

    # Test size
    assert len(dataset) == 100, f"Expected size 100, got {len(dataset)}"

    # Test get item
    input_sample, target_sample = dataset[0]
    assert input_sample.ndim == 1, f"Expected ndim 1, got {input_sample.ndim}"
    assert input_sample.shape[0] == 10, f"Expected shape[0] 10, got {input_sample.shape[0]}"

    print("  ✓ TensorDataset creation and access works")

def test_dataloader_basic():
    """Test basic DataLoader functionality."""
    print("Testing basic DataLoader...")

    # Create dataset
    inputs = tz.randn([100, 10])
    targets = tz.randn([100, 1])
    dataset = tz.data.TensorDataset(inputs, targets)

    # Create DataLoader
    loader = tz.data.DataLoader(dataset, batch_size=10, shuffle=False)

    # Test size
    assert len(loader) == 10, f"Expected 10 batches, got {len(loader)}"

    # Test iteration
    batch_count = 0
    for batch in loader:
        assert batch.inputs.shape[0] == 10, f"Expected batch size 10, got {batch.inputs.shape[0]}"
        assert batch.inputs.shape[1] == 10, f"Expected feature size 10, got {batch.inputs.shape[1]}"
        assert batch.targets.shape[0] == 10, f"Expected batch size 10, got {batch.targets.shape[0]}"
        batch_count += 1

    assert batch_count == 10, f"Expected 10 batches, got {batch_count}"

    print("  ✓ Basic DataLoader iteration works")

def test_dataloader_shuffle():
    """Test DataLoader with shuffling."""
    print("Testing DataLoader with shuffling...")

    # Create dataset with identifiable pattern
    inputs = tz.arange(0, 100).reshape([100, 1])
    targets = tz.arange(0, 100).reshape([100, 1])
    dataset = tz.data.TensorDataset(inputs, targets)

    # Create DataLoader with shuffle
    loader = tz.data.DataLoader(dataset, batch_size=10, shuffle=True)

    # Collect first samples from each batch
    first_epoch_samples = []
    for batch in loader:
        # Get first sample of batch
        first_sample = batch.inputs.slice(0, 0, 1)
        first_epoch_samples.append(float(first_sample.item()))

    # Reset and iterate again
    loader.reset()
    second_epoch_samples = []
    for batch in loader:
        first_sample = batch.inputs.slice(0, 0, 1)
        second_epoch_samples.append(float(first_sample.item()))

    # Check that orders are different (very unlikely to be the same with shuffle)
    different = any(a != b for a, b in zip(first_epoch_samples, second_epoch_samples))
    assert different, "Shuffle should produce different orders between epochs"

    print("  ✓ DataLoader shuffling works")

def test_dataloader_drop_last():
    """Test DataLoader with drop_last option."""
    print("Testing DataLoader with drop_last...")

    # Create dataset with 105 samples (not evenly divisible by 10)
    inputs = tz.randn([105, 10])
    targets = tz.randn([105, 1])
    dataset = tz.data.TensorDataset(inputs, targets)

    # Without drop_last
    loader1 = tz.data.DataLoader(dataset, batch_size=10, drop_last=False)
    assert len(loader1) == 11, f"Expected 11 batches, got {len(loader1)}"

    # With drop_last
    loader2 = tz.data.DataLoader(dataset, batch_size=10, drop_last=True)
    assert len(loader2) == 10, f"Expected 10 batches, got {len(loader2)}"

    # Verify all batches are full size with drop_last=True
    for batch in loader2:
        assert batch.inputs.shape[0] == 10, f"Expected full batch size 10, got {batch.inputs.shape[0]}"

    print("  ✓ DataLoader drop_last option works")

def test_dataloader_workers():
    """Test DataLoader with multiple workers."""
    print("Testing DataLoader with multiple workers...")

    inputs = tz.randn([100, 10])
    targets = tz.randn([100, 1])
    dataset = tz.data.TensorDataset(inputs, targets)

    # Create DataLoader with 4 workers
    loader = tz.data.DataLoader(dataset, batch_size=10, num_workers=4)

    batch_count = 0
    for batch in loader:
        assert batch.inputs.shape[0] == 10
        batch_count += 1

    assert batch_count == 10, f"Expected 10 batches, got {batch_count}"

    print("  ✓ DataLoader with multiple workers works")

def test_dataloader_config():
    """Test DataLoader with config object."""
    print("Testing DataLoader with config...")

    inputs = tz.randn([100, 10])
    targets = tz.randn([100, 1])
    dataset = tz.data.TensorDataset(inputs, targets)

    # Create config
    config = tz.data.DataLoaderConfig()
    config.batch_size = 20
    config.shuffle = True
    config.num_workers = 2
    config.drop_last = False

    # Create DataLoader with config
    loader = tz.data.DataLoader(dataset, config)

    assert len(loader) == 5, f"Expected 5 batches, got {len(loader)}"

    batch_count = 0
    for batch in loader:
        assert batch.inputs.shape[0] == 20
        batch_count += 1

    assert batch_count == 5, f"Expected 5 batches, got {batch_count}"

    print("  ✓ DataLoader with config object works")

def test_dataloader_reset():
    """Test DataLoader reset functionality."""
    print("Testing DataLoader reset...")

    inputs = tz.randn([50, 10])
    targets = tz.randn([50, 1])
    dataset = tz.data.TensorDataset(inputs, targets)

    loader = tz.data.DataLoader(dataset, batch_size=10, shuffle=False)

    # First iteration
    count1 = sum(1 for _ in loader)
    assert count1 == 5, f"Expected 5 batches, got {count1}"

    # Second iteration (automatic reset)
    count2 = sum(1 for _ in loader)
    assert count2 == 5, f"Expected 5 batches, got {count2}"

    # Manual reset
    loader.reset()
    count3 = sum(1 for _ in loader)
    assert count3 == 5, f"Expected 5 batches, got {count3}"

    print("  ✓ DataLoader reset functionality works")

def test_batch_struct():
    """Test Batch struct access."""
    print("Testing Batch struct...")

    inputs = tz.randn([20, 10])
    targets = tz.randn([20, 1])
    dataset = tz.data.TensorDataset(inputs, targets)

    loader = tz.data.DataLoader(dataset, batch_size=10)

    for batch in loader:
        # Access inputs and targets
        batch_inputs = batch.inputs
        batch_targets = batch.targets

        assert batch_inputs.shape[0] == 10
        assert batch_targets.shape[0] == 10

        # Verify we can use them for computation
        _ = batch_inputs.mean()
        _ = batch_targets.mean()
        break  # Just test first batch

    print("  ✓ Batch struct access works")

def main():
    """Run all tests."""
    print("=" * 60)
    print("DataLoader Python Bindings Test Suite")
    print("=" * 60)
    print()

    try:
        # Initialize Tenzor
        tz.initialize()

        # Run tests
        test_tensor_dataset()
        test_dataloader_basic()
        test_dataloader_shuffle()
        test_dataloader_drop_last()
        test_dataloader_workers()
        test_dataloader_config()
        test_dataloader_reset()
        test_batch_struct()

        print()
        print("=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)

        return 0

    except Exception as e:
        print()
        print("=" * 60)
        print(f"Test failed: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
