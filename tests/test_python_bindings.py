#!/usr/bin/env python3
"""
Test script for Python bindings of Tenzor neural network layers.
Tests all the newly added layer bindings to ensure they work correctly.
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
    try:
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
    except Exception as e:
        print(f"✗ Conv2d test failed: {e}")

    # Test Conv1d
    try:
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
    except Exception as e:
        print(f"✗ Conv1d test failed: {e}")

    # Test ConvTranspose2d
    try:
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
    except Exception as e:
        print(f"✗ ConvTranspose2d test failed: {e}")

def test_normalization_layers():
    """Test normalization layer bindings"""
    print("\n=== Testing Normalization Layers ===")

    # Test BatchNorm2d
    try:
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
    except Exception as e:
        print(f"✗ BatchNorm2d test failed: {e}")

    # Test BatchNorm1d
    try:
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
    except Exception as e:
        print(f"✗ BatchNorm1d test failed: {e}")

    # Test LayerNorm
    try:
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
    except Exception as e:
        print(f"✗ LayerNorm test failed: {e}")

def test_regularization_layers():
    """Test regularization layer bindings"""
    print("\n=== Testing Regularization Layers ===")

    # Test Dropout
    try:
        dropout = tz.nn.Dropout(p=0.5)
        print("✓ Dropout layer created successfully")

        input_tensor = tz.randn([4, 100])
        input_var = tz.Variable(input_tensor, requires_grad=True)
        output = dropout.forward(input_var)
        print(f"  Dropout output shape: {output.data.shape}")
    except Exception as e:
        print(f"✗ Dropout test failed: {e}")

    # Test Dropout2d
    try:
        dropout2d = tz.nn.Dropout2d(p=0.3)
        print("✓ Dropout2d layer created successfully")

        input_tensor = tz.randn([2, 16, 32, 32])
        input_var = tz.Variable(input_tensor, requires_grad=True)
        output = dropout2d.forward(input_var)
        print(f"  Dropout2d output shape: {output.data.shape}")
    except Exception as e:
        print(f"✗ Dropout2d test failed: {e}")

    # Test AlphaDropout
    try:
        alpha_dropout = tz.nn.AlphaDropout(p=0.2)
        print("✓ AlphaDropout layer created successfully")

        input_tensor = tz.randn([4, 100])
        input_var = tz.Variable(input_tensor, requires_grad=True)
        output = alpha_dropout.forward(input_var)
        print(f"  AlphaDropout output shape: {output.data.shape}")
    except Exception as e:
        print(f"✗ AlphaDropout test failed: {e}")

def test_pooling_layers():
    """Test pooling layer bindings"""
    print("\n=== Testing Pooling Layers ===")

    # Test MaxPool2d
    try:
        maxpool = tz.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        print("✓ MaxPool2d layer created successfully")

        input_tensor = tz.randn([1, 16, 32, 32])
        input_var = tz.Variable(input_tensor, requires_grad=True)
        output = maxpool.forward(input_var)
        print(f"  MaxPool2d output shape: {output.data.shape}")
    except Exception as e:
        print(f"✗ MaxPool2d test failed: {e}")

    # Test AvgPool2d
    try:
        avgpool = tz.nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        print("✓ AvgPool2d layer created successfully")

        input_tensor = tz.randn([1, 16, 32, 32])
        input_var = tz.Variable(input_tensor, requires_grad=True)
        output = avgpool.forward(input_var)
        print(f"  AvgPool2d output shape: {output.data.shape}")
    except Exception as e:
        print(f"✗ AvgPool2d test failed: {e}")

    # Test AdaptiveAvgPool2d (two constructors)
    try:
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
    except Exception as e:
        print(f"✗ AdaptiveAvgPool2d test failed: {e}")

def test_utility_layers():
    """Test utility layer bindings"""
    print("\n=== Testing Utility Layers ===")

    # Test Flatten
    try:
        flatten = tz.nn.Flatten(start_dim=1, end_dim=-1)
        print("✓ Flatten layer created successfully")

        input_tensor = tz.randn([4, 3, 32, 32])
        input_var = tz.Variable(input_tensor, requires_grad=True)
        output = flatten.forward(input_var)
        print(f"  Flatten output shape: {output.data.shape}")
    except Exception as e:
        print(f"✗ Flatten test failed: {e}")

def test_sequential_container():
    """Test Sequential container"""
    print("\n=== Testing Sequential Container ===")

    try:
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
    except Exception as e:
        print(f"✗ Sequential test failed: {e}")

def test_module_methods():
    """Test Module methods (train, eval, cuda, cpu)"""
    print("\n=== Testing Module Methods ===")

    try:
        conv = tz.nn.Conv2d(3, 16, kernel_size=3)

        # Test train/eval mode
        conv.train()
        print("✓ Module.train() called successfully")

        conv.eval()
        print("✓ Module.eval() called successfully")

        # Test cpu/cuda (may fail if CUDA not available)
        conv.cpu()
        print("✓ Module.cpu() called successfully")

        try:
            conv.cuda()
            print("✓ Module.cuda() called successfully")
        except:
            print("  (CUDA not available, skipping cuda() test)")

        # Test parameters
        params = conv.parameters()
        print(f"✓ Module.parameters() returned {len(params)} parameters")

    except Exception as e:
        print(f"✗ Module methods test failed: {e}")

def main():
    """Run all tests"""
    print("=" * 60)
    print("Tenzor Python Bindings Test Suite")
    print("=" * 60)

    test_conv_layers()
    test_normalization_layers()
    test_regularization_layers()
    test_pooling_layers()
    test_utility_layers()
    test_sequential_container()
    test_module_methods()

    print("\n" + "=" * 60)
    print("Test suite completed!")
    print("=" * 60)

if __name__ == "__main__":
    main()
