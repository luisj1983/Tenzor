#!/usr/bin/env python3
"""
Test script for Python bindings of Tenzor neural network layers.
Tests all the newly added layer bindings to ensure they work correctly.

Each test body raises on failure (no swallowing try/except). ``main``
collects which test functions raised and exits non-zero if any did, so a
broken binding fails the suite instead of printing a checkmark.
"""

import sys
import os

# Add the build directory to Python path
build_dir = os.path.join(os.path.dirname(__file__), '..', 'build', 'python')
sys.path.insert(0, build_dir)

try:
    import tenzor_core as tz
    print("Successfully imported tenzor_core")
except ImportError as e:
    print(f"Failed to import tenzor_core: {e}")
    print("Make sure to build the Python bindings first:")
    print("  cd build && cmake .. -DBUILD_PYTHON=ON && make")
    sys.exit(1)

# Initialize the library
tz.initialize()
print("Initialized Tenzor library")

def test_conv_layers():
    """Test convolution layer bindings"""
    print("\n=== Testing Convolution Layers ===")

    # Test Conv2d
    conv2d = tz.nn.Conv2d(
        in_channels=3,
        out_channels=16,
        kernel_size=3,
        stride=1,
        padding=1,
        dilation=1,
        groups=1,
        bias=True
    )
    print("✓ Conv2d layer created successfully")

    # Create dummy input [batch=1, channels=3, height=32, width=32]
    input_tensor = tz.randn([1, 3, 32, 32])
    input_var = tz.Variable(input_tensor, requires_grad=True)
    output = conv2d.forward(input_var)
    print(f"  Conv2d output shape: {output.data.shape}")

    # Test Conv1d
    conv1d = tz.nn.Conv1d(
        in_channels=4,
        out_channels=8,
        kernel_size=3,
        stride=1,
        padding=1
    )
    print("✓ Conv1d layer created successfully")

    # Create dummy input [batch=1, channels=4, length=100]
    input_tensor = tz.randn([1, 4, 100])
    input_var = tz.Variable(input_tensor, requires_grad=True)
    output = conv1d.forward(input_var)
    print(f"  Conv1d output shape: {output.data.shape}")

    # Test ConvTranspose2d
    deconv = tz.nn.ConvTranspose2d(
        in_channels=16,
        out_channels=3,
        kernel_size=4,
        stride=2,
        padding=1
    )
    print("✓ ConvTranspose2d layer created successfully")

    # Create dummy input [batch=1, channels=16, height=16, width=16]
    input_tensor = tz.randn([1, 16, 16, 16])
    input_var = tz.Variable(input_tensor, requires_grad=True)
    output = deconv.forward(input_var)
    print(f"  ConvTranspose2d output shape: {output.data.shape}")

def test_normalization_layers():
    """Test normalization layer bindings"""
    print("\n=== Testing Normalization Layers ===")

    # Test BatchNorm2d
    bn2d = tz.nn.BatchNorm2d(
        num_features=16,
        eps=1e-5,
        momentum=0.1,
        affine=True,
        track_running_stats=True
    )
    print("✓ BatchNorm2d layer created successfully")

    # Create dummy input [batch=2, channels=16, height=32, width=32]
    input_tensor = tz.randn([2, 16, 32, 32])
    input_var = tz.Variable(input_tensor, requires_grad=True)
    output = bn2d.forward(input_var)
    print(f"  BatchNorm2d output shape: {output.data.shape}")

    # Test BatchNorm1d
    bn1d = tz.nn.BatchNorm1d(
        num_features=10,
        eps=1e-5,
        momentum=0.1
    )
    print("✓ BatchNorm1d layer created successfully")

    # Create dummy input [batch=4, features=10]
    input_tensor = tz.randn([4, 10])
    input_var = tz.Variable(input_tensor, requires_grad=True)
    output = bn1d.forward(input_var)
    print(f"  BatchNorm1d output shape: {output.data.shape}")

    # Test LayerNorm
    ln = tz.nn.LayerNorm(
        normalized_shape=[128],
        eps=1e-5,
        elementwise_affine=True
    )
    print("✓ LayerNorm layer created successfully")

    # Create dummy input [batch=4, features=128]
    input_tensor = tz.randn([4, 128])
    input_var = tz.Variable(input_tensor, requires_grad=True)
    output = ln.forward(input_var)
    print(f"  LayerNorm output shape: {output.data.shape}")

def test_regularization_layers():
    """Test regularization layer bindings"""
    print("\n=== Testing Regularization Layers ===")

    # Test Dropout
    dropout = tz.nn.Dropout(p=0.5)
    print("✓ Dropout layer created successfully")

    input_tensor = tz.randn([4, 100])
    input_var = tz.Variable(input_tensor, requires_grad=True)
    output = dropout.forward(input_var)
    print(f"  Dropout output shape: {output.data.shape}")

    # Test Dropout2d
    dropout2d = tz.nn.Dropout2d(p=0.3)
    print("✓ Dropout2d layer created successfully")

    input_tensor = tz.randn([2, 16, 32, 32])
    input_var = tz.Variable(input_tensor, requires_grad=True)
    output = dropout2d.forward(input_var)
    print(f"  Dropout2d output shape: {output.data.shape}")

    # Test AlphaDropout
    alpha_dropout = tz.nn.AlphaDropout(p=0.2)
    print("✓ AlphaDropout layer created successfully")

    input_tensor = tz.randn([4, 100])
    input_var = tz.Variable(input_tensor, requires_grad=True)
    output = alpha_dropout.forward(input_var)
    print(f"  AlphaDropout output shape: {output.data.shape}")

def test_pooling_layers():
    """Test pooling layer bindings"""
    print("\n=== Testing Pooling Layers ===")

    # Test MaxPool2d
    maxpool = tz.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    print("✓ MaxPool2d layer created successfully")

    input_tensor = tz.randn([1, 16, 32, 32])
    input_var = tz.Variable(input_tensor, requires_grad=True)
    output = maxpool.forward(input_var)
    print(f"  MaxPool2d output shape: {output.data.shape}")

    # Test AvgPool2d
    avgpool = tz.nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
    print("✓ AvgPool2d layer created successfully")

    input_tensor = tz.randn([1, 16, 32, 32])
    input_var = tz.Variable(input_tensor, requires_grad=True)
    output = avgpool.forward(input_var)
    print(f"  AvgPool2d output shape: {output.data.shape}")

    # Test AdaptiveAvgPool2d (two constructors)
    # Two-parameter constructor
    adaptive_pool = tz.nn.AdaptiveAvgPool2d(output_h=7, output_w=7)
    print("✓ AdaptiveAvgPool2d (h,w) layer created successfully")

    input_tensor = tz.randn([1, 16, 32, 32])
    input_var = tz.Variable(input_tensor, requires_grad=True)
    output = adaptive_pool.forward(input_var)
    print(f"  AdaptiveAvgPool2d output shape: {output.data.shape}")

    # Single-parameter constructor (square output)
    adaptive_pool_sq = tz.nn.AdaptiveAvgPool2d(output_size=1)
    output_sq = adaptive_pool_sq.forward(input_var)
    print(f"  AdaptiveAvgPool2d (square) output shape: {output_sq.data.shape}")

def test_utility_layers():
    """Test utility layer bindings"""
    print("\n=== Testing Utility Layers ===")

    # Test Flatten
    flatten = tz.nn.Flatten(start_dim=1, end_dim=-1)
    print("✓ Flatten layer created successfully")

    input_tensor = tz.randn([4, 3, 32, 32])
    input_var = tz.Variable(input_tensor, requires_grad=True)
    output = flatten.forward(input_var)
    print(f"  Flatten output shape: {output.data.shape}")

def test_sequential_container():
    """Test Sequential container"""
    print("\n=== Testing Sequential Container ===")

    # Create a sequential model
    model = tz.nn.Sequential()

    # Add layers
    conv = tz.nn.Conv2d(3, 16, kernel_size=3, padding=1)
    relu = tz.nn.ReLU()
    pool = tz.nn.MaxPool2d(kernel_size=2, stride=2)

    model.add_module(conv)
    model.add_module(relu)
    model.add_module(pool)

    print("✓ Sequential container created and layers added successfully")

    # Test forward pass
    input_tensor = tz.randn([1, 3, 32, 32])
    input_var = tz.Variable(input_tensor, requires_grad=True)
    output = model.forward(input_var)
    print(f"  Sequential output shape: {output.data.shape}")

def test_module_methods():
    """Test Module methods (train, eval, cuda, cpu)"""
    print("\n=== Testing Module Methods ===")

    conv = tz.nn.Conv2d(3, 16, kernel_size=3)

    # Test train/eval mode
    conv.train()
    print("✓ Module.train() called successfully")

    conv.eval()
    print("✓ Module.eval() called successfully")

    # Test cpu/cuda (cuda only when a CUDA device is present)
    conv.cpu()
    print("✓ Module.cpu() called successfully")

    if tz.cuda_is_available():
        conv.cuda()
        print("✓ Module.cuda() called successfully")
    else:
        print("  (CUDA not available, skipping cuda() test)")

    # Test parameters
    params = conv.parameters()
    print(f"✓ Module.parameters() returned {len(params)} parameters")

def main():
    """Run all tests. Exit non-zero if any test function raises."""
    print("=" * 60)
    print("Tenzor Python Bindings Test Suite")
    print("=" * 60)

    tests = [
        test_conv_layers,
        test_normalization_layers,
        test_regularization_layers,
        test_pooling_layers,
        test_utility_layers,
        test_sequential_container,
        test_module_methods,
    ]

    failures = []
    for test in tests:
        try:
            test()
        except Exception as e:
            import traceback
            print(f"✗ {test.__name__} FAILED: {e}")
            traceback.print_exc()
            failures.append(test.__name__)

    print("\n" + "=" * 60)
    if failures:
        print(f"Test suite FAILED: {len(failures)} test(s) failed: "
              f"{', '.join(failures)}")
        print("=" * 60)
        return 1
    print("Test suite completed: all tests passed!")
    print("=" * 60)
    return 0

if __name__ == "__main__":
    sys.exit(main())
