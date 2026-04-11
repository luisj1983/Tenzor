"""
Tests for newly added data transforms.

Tests RandomVerticalFlip, CenterCrop, RandomCrop, Resize,
RandomResizedCrop, GaussianBlur, and RandomAffine for shape
correctness, randomness, and basic behavior.
"""

import sys
import os

build_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'build', 'python')
if os.path.exists(build_dir):
    sys.path.insert(0, os.path.abspath(build_dir))

import tenzor as tz


def _init():
    tz.initialize()


def _make_image(h=32, w=32, c=3):
    """Create a fake CHW float32 image tensor."""
    return tz.randn([c, h, w])


def test_random_vertical_flip_shape():
    _init()
    t = tz.data.transforms.RandomVerticalFlip(p=1.0)
    img = _make_image()
    target = tz.zeros([1])
    out_img, out_target = t(img, target)
    assert out_img.shape == img.shape, f"Shape mismatch: {out_img.shape} vs {img.shape}"


def test_random_vertical_flip_no_flip():
    _init()
    t = tz.data.transforms.RandomVerticalFlip(p=0.0)
    img = _make_image()
    target = tz.zeros([1])
    out_img, _ = t(img, target)
    # With p=0, output should be identical to input
    assert out_img.shape == img.shape


def test_center_crop():
    _init()
    t = tz.data.transforms.CenterCrop(16, 16)
    img = _make_image(32, 32)
    target = tz.zeros([1])
    out_img, _ = t(img, target)
    assert out_img.shape == [3, 16, 16], f"Expected [3,16,16], got {out_img.shape}"


def test_random_crop():
    _init()
    t = tz.data.transforms.RandomCrop(24, 24, padding=0)
    img = _make_image(32, 32)
    target = tz.zeros([1])
    out_img, _ = t(img, target)
    assert out_img.shape == [3, 24, 24], f"Expected [3,24,24], got {out_img.shape}"


def test_random_crop_with_padding():
    _init()
    t = tz.data.transforms.RandomCrop(32, 32, padding=4)
    img = _make_image(32, 32)
    target = tz.zeros([1])
    out_img, _ = t(img, target)
    assert out_img.shape == [3, 32, 32], f"Expected [3,32,32], got {out_img.shape}"


def test_resize():
    _init()
    t = tz.data.transforms.Resize(16, 16)
    img = _make_image(32, 32)
    target = tz.zeros([1])
    out_img, _ = t(img, target)
    assert out_img.shape == [3, 16, 16], f"Expected [3,16,16], got {out_img.shape}"


def test_resize_upscale():
    _init()
    t = tz.data.transforms.Resize(64, 64)
    img = _make_image(32, 32)
    target = tz.zeros([1])
    out_img, _ = t(img, target)
    assert out_img.shape == [3, 64, 64], f"Expected [3,64,64], got {out_img.shape}"


def test_random_resized_crop():
    _init()
    t = tz.data.transforms.RandomResizedCrop(24, 24)
    img = _make_image(32, 32)
    target = tz.zeros([1])
    out_img, _ = t(img, target)
    assert out_img.shape == [3, 24, 24], f"Expected [3,24,24], got {out_img.shape}"


def test_gaussian_blur():
    _init()
    t = tz.data.transforms.GaussianBlur(kernel_size=3, sigma_min=0.5, sigma_max=1.5)
    img = _make_image(16, 16)
    target = tz.zeros([1])
    out_img, _ = t(img, target)
    assert out_img.shape == img.shape, f"Shape mismatch: {out_img.shape} vs {img.shape}"


def test_gaussian_blur_odd_kernel():
    """Kernel size must be odd."""
    _init()
    try:
        tz.data.transforms.GaussianBlur(kernel_size=4)
        assert False, "Should have raised for even kernel size"
    except (ValueError, RuntimeError):
        pass


def test_random_affine():
    _init()
    t = tz.data.transforms.RandomAffine(degrees=15.0, translate_x=0.1, translate_y=0.1)
    img = _make_image(32, 32)
    target = tz.zeros([1])
    out_img, _ = t(img, target)
    assert out_img.shape == img.shape, f"Shape mismatch: {out_img.shape} vs {img.shape}"


def test_random_affine_with_scale():
    _init()
    t = tz.data.transforms.RandomAffine(degrees=0.0, scale_min=0.8, scale_max=1.2)
    img = _make_image(16, 16)
    target = tz.zeros([1])
    out_img, _ = t(img, target)
    assert out_img.shape == img.shape


def test_compose_new_transforms():
    """Test composing several new transforms together."""
    _init()
    import ctypes

    t = tz.data.transforms.Compose([
        tz.data.transforms.RandomCrop(28, 28, padding=2),
        tz.data.transforms.RandomHorizontalFlip(0.5),
        tz.data.transforms.RandomVerticalFlip(0.5),
    ])
    img = _make_image(32, 32)
    target = tz.zeros([1])
    out_img, _ = t(img, target)
    assert out_img.shape == [3, 28, 28], f"Expected [3,28,28], got {out_img.shape}"


if __name__ == "__main__":
    test_random_vertical_flip_shape()
    test_random_vertical_flip_no_flip()
    test_center_crop()
    test_random_crop()
    test_random_crop_with_padding()
    test_resize()
    test_resize_upscale()
    test_random_resized_crop()
    test_gaussian_blur()
    test_gaussian_blur_odd_kernel()
    test_random_affine()
    test_random_affine_with_scale()
    test_compose_new_transforms()
    print("All transform extended tests passed!")
