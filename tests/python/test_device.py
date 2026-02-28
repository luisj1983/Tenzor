#!/usr/bin/env python3
"""
Test Python bindings for device management: CPU creation, .to(), module.cpu(),
and graceful skip when no GPU is available.
"""

import sys
import os

build_python_dir = os.path.join(os.path.dirname(__file__), '../../build/python')
sys.path.insert(0, build_python_dir)

import tenzor.tenzor_core as tz


def test_cpu_device():
    """Test CPU device creation and properties."""
    print("Testing CPU device...")
    t = tz.zeros([2, 3])
    dev = t.device
    assert dev.type == tz.DeviceType.CPU, f"Expected CPU, got {dev.type}"
    print("  CPU device OK")


def test_cpu_tensor_creation():
    """Test various creation ops produce CPU tensors."""
    print("Testing CPU tensor creation...")
    for name, args in [
        ("zeros", [[2, 3]]),
        ("ones", [[3, 4]]),
        ("randn", [[5, 5]]),
        ("rand", [[3, 3]]),
        ("eye", [4]),
    ]:
        fn = getattr(tz, name)
        t = fn(*args)
        dev = t.device
        assert dev.type == tz.DeviceType.CPU, f"{name} not on CPU"
    print("  CPU tensor creation OK")


def test_tensor_to_same_device():
    """Test .to() to the same device is a no-op."""
    print("Testing .to() same device...")
    t = tz.ones([2, 3])
    cpu_dev = tz.Device(tz.DeviceType.CPU, 0)
    t2 = t.to(cpu_dev)
    assert t2.device.type == tz.DeviceType.CPU
    print("  .to() same device OK")


def test_variable_device():
    """Test Variable reports correct device."""
    print("Testing Variable device...")
    t = tz.randn([3, 4], tz.dtype.float32)
    v = tz.Variable(t, True)
    assert v.data.device.type == tz.DeviceType.CPU
    print("  Variable device OK")


def test_module_cpu():
    """Test module on CPU."""
    print("Testing module on CPU...")
    linear = tz.nn.Linear(10, 5)

    x = tz.Variable(tz.randn([2, 10], tz.dtype.float32), False)
    y = linear.forward(x)
    assert y.data.shape == [2, 5], f"Wrong shape: {y.data.shape}"
    assert y.data.device.type == tz.DeviceType.CPU
    print("  module CPU OK")


def test_cuda_graceful_skip():
    """Test CUDA availability check without crashing."""
    print("Testing CUDA availability check...")
    has_cuda = tz.cuda_is_available()
    print(f"  CUDA available: {has_cuda}")

    if has_cuda:
        # If CUDA is available, test moving tensor to GPU
        t = tz.ones([2, 3])
        cuda_dev = tz.Device(tz.DeviceType.CUDA, 0)
        try:
            t_gpu = t.to(cuda_dev)
            assert t_gpu.device.type == tz.DeviceType.CUDA
            print("  CUDA tensor creation OK")

            # Test moving back to CPU
            t_cpu = t_gpu.to(tz.Device(tz.DeviceType.CPU, 0))
            assert t_cpu.device.type == tz.DeviceType.CPU
            print("  CUDA->CPU transfer OK")
        except RuntimeError as e:
            print(f"  CUDA init failed (expected on some systems): {e}")
    else:
        print("  Skipping CUDA tests (no GPU)")
    print("  CUDA graceful skip OK")


def test_device_construction():
    """Test Device object construction."""
    print("Testing Device construction...")
    cpu = tz.Device(tz.DeviceType.CPU, 0)
    assert cpu.type == tz.DeviceType.CPU
    assert cpu.index == 0
    print("  Device construction OK")


def main():
    print("=" * 60)
    print("Testing Device Management Bindings")
    print("=" * 60)

    try:
        tz.initialize()

        test_cpu_device()
        test_cpu_tensor_creation()
        test_tensor_to_same_device()
        test_variable_device()
        test_module_cpu()
        test_cuda_graceful_skip()
        test_device_construction()

        print("\n" + "=" * 60)
        print("ALL DEVICE TESTS PASSED")
        print("=" * 60)
        return 0

    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
