#!/usr/bin/env python3
"""
Test Python bindings for vision and detection operations.
"""

import sys
import os

build_python_dir = os.path.join(os.path.dirname(__file__), '../../build/python')
sys.path.insert(0, build_python_dir)

import tenzor.tenzor_core as tz


def test_unfold():
    """Test unfold (im2col) operation."""
    print("Testing vision.unfold...")
    x = tz.randn([1, 3, 8, 8])
    blocks = tz.vision.unfold(x, 3, stride=1, padding=1, dilation=1)
    # Output shape: (N, C*k*k, L) where L = H*W for stride=1, padding=1
    assert blocks.shape[0] == 1, f"Batch dim wrong: {blocks.shape}"
    assert blocks.shape[1] == 3 * 3 * 3, f"Channel*kernel dim wrong: {blocks.shape}"
    assert blocks.shape[2] == 8 * 8, f"Spatial dim wrong: {blocks.shape}"
    print("  unfold OK")


def test_fold():
    """Test fold (col2im) operation."""
    print("Testing vision.fold...")
    # Create input matching unfold output shape
    blocks = tz.randn([1, 27, 64])  # (N, C*k*k, L) with C=3,k=3,L=8*8
    img = tz.vision.fold(blocks, [8, 8], 3, stride=1, padding=1, dilation=1)
    assert img.shape[0] == 1, f"Batch dim wrong: {img.shape}"
    assert img.shape[2] == 8, f"H wrong: {img.shape}"
    assert img.shape[3] == 8, f"W wrong: {img.shape}"
    print("  fold OK")


def test_interpolate():
    """Test interpolation (resize) operation."""
    print("Testing vision.interpolate...")
    x = tz.randn([1, 3, 16, 16])
    y = tz.vision.interpolate(x, [32, 32], "bilinear", False)
    assert y.shape == [1, 3, 32, 32], f"Wrong shape: {y.shape}"

    # Downscale
    y2 = tz.vision.interpolate(x, [8, 8], "nearest", False)
    assert y2.shape == [1, 3, 8, 8], f"Wrong shape: {y2.shape}"
    print("  interpolate OK")


def test_grid_sample():
    """Test grid_sample (spatial transformer sampling)."""
    print("Testing vision.grid_sample...")
    x = tz.randn([1, 1, 4, 4])
    # Grid of shape (N, H_out, W_out, 2) with values in [-1, 1]
    grid = tz.randn([1, 2, 2, 2])  # Sample at 2x2 output
    y = tz.vision.grid_sample(x, grid, "bilinear", "zeros", False)
    assert y.shape == [1, 1, 2, 2], f"Wrong shape: {y.shape}"
    print("  grid_sample OK")


def test_affine_grid():
    """Test affine grid generation."""
    print("Testing vision.affine_grid...")
    # theta: (N, 2, 3) affine transformation matrix
    theta = tz.randn([1, 2, 3])
    grid = tz.vision.affine_grid(theta, [1, 1, 4, 4], False)
    assert grid.shape == [1, 4, 4, 2], f"Wrong shape: {grid.shape}"
    print("  affine_grid OK")


def test_box_iou():
    """Test bounding box IoU computation."""
    print("Testing vision.box_iou...")
    # Boxes in (x1, y1, x2, y2) format
    boxes1 = tz.full([2, 4], 0.0)  # 2 boxes
    boxes2 = tz.full([3, 4], 0.0)  # 3 boxes
    iou = tz.detection.box_iou(boxes1, boxes2)
    assert iou.shape == [2, 3], f"Wrong shape: {iou.shape}"
    print("  box_iou OK")


def test_nms():
    """Test Non-Maximum Suppression."""
    print("Testing vision.nms...")
    # 4 boxes with scores
    boxes = tz.randn([4, 4])
    scores = tz.randn([4])
    keep = tz.detection.nms(boxes, scores, 0.5)
    # Result is a 1D tensor of kept indices
    assert keep.ndim == 1, f"Expected 1D, got {keep.ndim}D"
    print("  nms OK")


if __name__ == "__main__":
    tz.initialize()
    tz.manual_seed(42)
    test_unfold()
    test_fold()
    test_interpolate()
    test_grid_sample()
    test_affine_grid()
    test_box_iou()
    test_nms()
    print("\nAll vision/detection tests passed!")
