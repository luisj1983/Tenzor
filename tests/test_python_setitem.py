#!/usr/bin/env python3
"""
Comprehensive tests for Python tensor __setitem__ implementation.
Tests various indexing patterns: integer, slice, multi-dimensional, ellipsis.
"""

import sys
import os

# Add python module directory to path
python_dir = os.path.join(os.path.dirname(__file__), '..', 'build', 'python', 'tenzor')
sys.path.insert(0, python_dir)

import tenzor_core as tz
import numpy as np


def test_scalar_assignment():
    """Test basic scalar value assignment."""
    print("Testing scalar assignment...")

    # Single integer index
    t = tz.zeros([5], tz.dtype.float32)
    t[0] = 1.0
    t[2] = 2.5
    t[-1] = 3.0

    arr = t.numpy()
    assert arr[0] == 1.0, f"Expected 1.0, got {arr[0]}"
    assert arr[2] == 2.5, f"Expected 2.5, got {arr[2]}"
    assert arr[4] == 3.0, f"Expected 3.0, got {arr[4]}"

    print("  ✓ Single integer index with scalar")


def test_slice_scalar_assignment():
    """Test slice assignment with scalar value."""
    print("Testing slice with scalar assignment...")

    # 1D slice
    t = tz.zeros([10], tz.dtype.float32)
    t[2:7] = 5.0

    arr = t.numpy()
    assert np.all(arr[2:7] == 5.0), "Slice assignment failed"
    assert np.all(arr[:2] == 0.0), "Unchanged region was modified"
    assert np.all(arr[7:] == 0.0), "Unchanged region was modified"

    print("  ✓ Slice with scalar value")


def test_slice_tensor_assignment():
    """Test slice assignment with tensor value."""
    print("Testing slice with tensor assignment...")

    t = tz.zeros([10], tz.dtype.float32)
    value = tz.ones([5], tz.dtype.float32)
    value = value * 7.0  # Fill with 7.0

    t[2:7] = value

    arr = t.numpy()
    assert np.all(arr[2:7] == 7.0), f"Expected 7.0, got {arr[2:7]}"
    assert np.all(arr[:2] == 0.0), "Unchanged region was modified"

    print("  ✓ Slice with tensor value")


def test_2d_indexing():
    """Test 2D tensor indexing and assignment."""
    print("Testing 2D indexing...")

    # Row assignment
    t = tz.zeros([4, 5], tz.dtype.float32)
    t[1] = 3.0

    arr = t.numpy()
    assert np.all(arr[1, :] == 3.0), "Row assignment failed"
    assert np.all(arr[0, :] == 0.0), "Other rows should be unchanged"

    # Multi-dimensional slice
    t = tz.zeros([4, 5], tz.dtype.float32)
    t[1, 2] = 9.0

    arr = t.numpy()
    assert arr[1, 2] == 9.0, f"Expected 9.0 at [1,2], got {arr[1, 2]}"

    print("  ✓ 2D indexing")


def test_multidim_slice():
    """Test multi-dimensional slice assignment."""
    print("Testing multi-dimensional slice assignment...")

    t = tz.zeros([3, 4, 5], tz.dtype.float32)

    # Assign to sub-tensor
    t[1, :, 2:4] = 8.0

    arr = t.numpy()
    assert np.all(arr[1, :, 2:4] == 8.0), "Multi-dim slice assignment failed"
    assert np.all(arr[0, :, :] == 0.0), "Other slices should be unchanged"

    print("  ✓ Multi-dimensional slice")


def test_negative_indexing():
    """Test negative indexing support."""
    print("Testing negative indexing...")

    t = tz.zeros([10], tz.dtype.float32)
    t[-1] = 9.0
    t[-3] = 7.0

    arr = t.numpy()
    assert arr[-1] == 9.0, f"Expected 9.0 at [-1], got {arr[-1]}"
    assert arr[-3] == 7.0, f"Expected 7.0 at [-3], got {arr[-3]}"

    print("  ✓ Negative indexing")


def test_dtype_compatibility():
    """Test assignment across different dtypes."""
    print("Testing dtype compatibility...")

    # Float32
    t_f32 = tz.zeros([5], tz.dtype.float32)
    t_f32[0] = 1.5
    assert t_f32.numpy()[0] == 1.5

    # Int32
    t_i32 = tz.zeros([5], tz.dtype.int32)
    t_i32[0] = 42
    assert t_i32.numpy()[0] == 42

    # Int64
    t_i64 = tz.zeros([5], tz.dtype.int64)
    t_i64[0] = 100
    assert t_i64.numpy()[0] == 100

    print("  ✓ Multiple dtypes")


def test_broadcasting_scalar():
    """Test broadcasting with scalar values."""
    print("Testing scalar broadcasting...")

    t = tz.zeros([3, 4], tz.dtype.float32)
    t[:, 2] = 5.0

    arr = t.numpy()
    assert np.all(arr[:, 2] == 5.0), "Column assignment with broadcast failed"

    print("  ✓ Scalar broadcasting")


def test_error_handling():
    """Test error handling for invalid operations."""
    print("Testing error handling...")

    t = tz.zeros([5], tz.dtype.float32)

    # Out of bounds
    try:
        t[10] = 1.0
        assert False, "Should have raised index out of range"
    except IndexError:
        pass
    except RuntimeError as e:
        if "out of range" not in str(e).lower():
            raise

    # Negative out of bounds
    try:
        t[-10] = 1.0
        assert False, "Should have raised index out of range"
    except IndexError:
        pass
    except RuntimeError as e:
        if "out of range" not in str(e).lower():
            raise

    print("  ✓ Error handling")


def test_slice_edge_cases():
    """Test edge cases in slicing."""
    print("Testing slice edge cases...")

    t = tz.zeros([10], tz.dtype.float32)

    # Empty slice (no-op)
    t[5:5] = 1.0
    assert np.all(t.numpy() == 0.0), "Empty slice should not modify tensor"

    # Full slice
    t[:] = 3.0
    assert np.all(t.numpy() == 3.0), "Full slice assignment failed"

    print("  ✓ Slice edge cases")


def test_contiguous_vs_noncontiguous():
    """Test assignment to contiguous and non-contiguous tensors."""
    print("Testing contiguous tensors...")

    # Contiguous
    t = tz.zeros([4, 4], tz.dtype.float32)
    t[1:3, 1:3] = 2.0

    arr = t.numpy()
    assert np.all(arr[1:3, 1:3] == 2.0), "Contiguous assignment failed"

    print("  ✓ Contiguous assignment")


def run_all_tests():
    """Run all test functions."""
    print("\n" + "="*60)
    print("Python Tensor __setitem__ Tests")
    print("="*60 + "\n")

    # Initialize Tenzor
    tz.initialize()

    tests = [
        test_scalar_assignment,
        test_slice_scalar_assignment,
        test_slice_tensor_assignment,
        test_2d_indexing,
        test_multidim_slice,
        test_negative_indexing,
        test_dtype_compatibility,
        test_broadcasting_scalar,
        test_error_handling,
        test_slice_edge_cases,
        test_contiguous_vs_noncontiguous,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"\n  ✗ FAILED: {test.__name__}")
            print(f"    Error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "="*60)
    print(f"Results: {passed} passed, {failed} failed")
    print("="*60 + "\n")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
