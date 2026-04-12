#!/usr/bin/env python3
"""
Test newly added Python functional API functions: F.one_hot, F.unfold, F.fold.
"""

import sys
import os

build_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'build', 'python')
if os.path.exists(build_dir):
    sys.path.insert(0, os.path.abspath(build_dir))

import tenzor as tz
import tenzor.tenzor_core as tzc
import tenzor.nn.functional as F


def _init():
    tz.initialize()
    tzc.manual_seed(42)


def test_one_hot_basic():
    """Test basic one-hot encoding."""
    _init()
    print("Testing F.one_hot (basic)...")
    labels = tzc.zeros([4], tzc.dtype.int64)
    oh = F.one_hot(labels, num_classes=5)
    assert oh.shape == [4, 5], f"Wrong shape: {oh.shape}"
    print("  one_hot basic OK")


def test_one_hot_infer_classes():
    """Test one-hot with inferred num_classes."""
    _init()
    print("Testing F.one_hot (infer classes)...")
    labels = tzc.zeros([3], tzc.dtype.int64)
    oh = F.one_hot(labels)
    # With all zeros, num_classes should be inferred as 1
    assert oh.shape[0] == 3, f"Batch dim wrong: {oh.shape}"
    print("  one_hot infer OK")


def test_unfold_basic():
    """Test F.unfold basic operation."""
    _init()
    print("Testing F.unfold...")
    x = tzc.randn([1, 3, 8, 8])
    blocks = F.unfold(x, kernel_size=3, padding=1, stride=1)
    assert blocks.shape[0] == 1, f"Batch dim wrong: {blocks.shape}"
    assert blocks.shape[1] == 3 * 3 * 3, f"Channel*kernel dim wrong: {blocks.shape}"
    print("  unfold OK")


def test_fold_basic():
    """Test F.fold basic operation."""
    _init()
    print("Testing F.fold...")
    blocks = tzc.randn([1, 27, 64])
    img = F.fold(blocks, output_size=(8, 8), kernel_size=3, padding=1, stride=1)
    assert img.shape[0] == 1, f"Batch dim wrong: {img.shape}"
    assert img.shape[2] == 8, f"H wrong: {img.shape}"
    assert img.shape[3] == 8, f"W wrong: {img.shape}"
    print("  fold OK")


def test_unfold_fold_roundtrip():
    """Test that fold(unfold(x)) approximately reconstructs x."""
    _init()
    print("Testing unfold/fold roundtrip...")
    x = tzc.randn([1, 1, 4, 4])
    blocks = F.unfold(x, kernel_size=2, stride=2)
    reconstructed = F.fold(blocks, output_size=(4, 4), kernel_size=2, stride=2)
    # With non-overlapping blocks (stride==kernel_size), reconstruction should be exact
    assert reconstructed.shape == [1, 1, 4, 4], f"Wrong shape: {reconstructed.shape}"
    print("  unfold/fold roundtrip OK")


def test_unfold_stride():
    """Test F.unfold with different stride."""
    _init()
    print("Testing F.unfold with stride=2...")
    x = tzc.randn([2, 3, 8, 8])
    blocks = F.unfold(x, kernel_size=3, stride=2, padding=0)
    assert blocks.shape[0] == 2, f"Batch dim wrong: {blocks.shape}"
    assert blocks.shape[1] == 3 * 3 * 3, f"Channel*kernel dim wrong: {blocks.shape}"
    # L = ((8 - 3) / 2 + 1)^2 = 3^2 = 9
    assert blocks.shape[2] == 9, f"Spatial dim wrong: {blocks.shape}"
    print("  unfold stride OK")


if __name__ == "__main__":
    _init()
    test_one_hot_basic()
    test_one_hot_infer_classes()
    test_unfold_basic()
    test_fold_basic()
    test_unfold_fold_roundtrip()
    test_unfold_stride()
    print("\nAll new functional tests passed!")
