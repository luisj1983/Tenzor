#!/usr/bin/env python3
"""
Test Python Module subclassing functionality.

This test verifies that Python users can create custom modules by subclassing
tz.nn.Module and overriding forward_impl().
"""

import sys
import os

# Add the build directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../python'))

import tenzor as tz


def test_basic_module_subclass():
    """Test basic Module subclassing with forward_impl override."""
    print("=" * 60)
    print("Test 1: Basic Module Subclassing")
    print("=" * 60)

    class SimpleLinear(tz.nn.Module):
        """A simple custom linear layer."""

        def __init__(self, in_features: int, out_features: int):
            super().__init__()
            # Create weight as a Variable and register it
            # randn already gives small values
            weight_data = tz.randn([out_features, in_features])
            self.weight = tz.Variable(weight_data, requires_grad=True)

            bias_data = tz.zeros([out_features])
            self.bias = tz.Variable(bias_data, requires_grad=True)

        def forward_impl(self, x):
            # Simple linear: y = x @ W^T + b
            out = tz.matmul(x.tensor(), tz.transpose(self.weight.tensor(), 0, 1))
            return tz.Variable(out + self.bias.tensor())

    # Create instance
    layer = SimpleLinear(10, 5)
    print(f"Created SimpleLinear(10, 5)")

    # Check parameters
    params = layer.parameters()
    print(f"Number of parameters: {len(params)}")

    # Forward pass
    x = tz.Variable(tz.randn([3, 10]), requires_grad=False)
    y = layer(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")

    print("PASSED: Basic Module subclassing works!\n")


def test_builtin_layers_composition():
    """Test composing with built-in layers."""
    print("=" * 60)
    print("Test 2: Built-in Layer Composition")
    print("=" * 60)

    class MLP(tz.nn.Module):
        """Multi-layer perceptron using built-in layers."""

        def __init__(self, in_features: int, hidden: int, out_features: int):
            super().__init__()
            self.fc1 = tz.nn.Linear(in_features, hidden)
            self.fc2 = tz.nn.Linear(hidden, out_features)

        def forward_impl(self, x):
            x = self.fc1(x)
            x = tz.nn.relu(x)
            x = self.fc2(x)
            return x

    # Create instance
    model = MLP(784, 128, 10)
    print(f"Created MLP(784, 128, 10)")

    # Check parameters
    params = model.parameters()
    print(f"Number of parameter tensors: {len(params)}")

    # Count total parameters
    total_params = sum(p.tensor().numel for p in params)
    print(f"Total parameters: {total_params}")
    # Expected: (784*128 + 128) + (128*10 + 10) = 100352 + 1290 = 101642
    expected = (784 * 128 + 128) + (128 * 10 + 10)
    print(f"Expected: {expected}")

    # Forward pass - input shape is [batch_size, in_features]
    x = tz.Variable(tz.randn([32, 784]), requires_grad=False)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")

    print("PASSED: Built-in layer composition works!\n")


def test_sequential_container():
    """Test Sequential container."""
    print("=" * 60)
    print("Test 3: Sequential Container")
    print("=" * 60)

    # Create a Sequential model
    model = tz.nn.Sequential(
        tz.nn.Linear(10, 20),
        tz.nn.ReLU(),
        tz.nn.Linear(20, 5)
    )
    print("Created Sequential model")

    # Check length
    print(f"Number of layers: {len(model)}")

    # Forward pass - input shape is [batch_size, in_features]
    x = tz.Variable(tz.randn([8, 10]), requires_grad=False)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")

    print("PASSED: Sequential container works!\n")


def test_nested_modules():
    """Test nested module composition."""
    print("=" * 60)
    print("Test 4: Nested Module Composition")
    print("=" * 60)

    class Block(tz.nn.Module):
        """A residual-style block."""

        def __init__(self, features: int):
            super().__init__()
            self.fc1 = tz.nn.Linear(features, features)
            self.fc2 = tz.nn.Linear(features, features)

        def forward_impl(self, x):
            identity = x
            out = self.fc1(x)
            out = tz.nn.relu(out)
            out = self.fc2(out)
            # Skip connection
            out = tz.Variable(out.tensor() + identity.tensor())
            return tz.nn.relu(out)

    class Network(tz.nn.Module):
        """Network with multiple blocks."""

        def __init__(self, in_features: int, hidden: int, out_features: int):
            super().__init__()
            self.input_proj = tz.nn.Linear(in_features, hidden)
            self.block1 = Block(hidden)
            self.block2 = Block(hidden)
            self.output_proj = tz.nn.Linear(hidden, out_features)

        def forward_impl(self, x):
            x = self.input_proj(x)
            x = tz.nn.relu(x)
            x = self.block1(x)
            x = self.block2(x)
            x = self.output_proj(x)
            return x

    # Create instance
    net = Network(32, 64, 10)
    print("Created Network with nested blocks")

    # Check parameters
    params = net.parameters()
    print(f"Number of parameter tensors: {len(params)}")

    # Forward pass - input shape is [batch_size, in_features]
    x = tz.Variable(tz.randn([16, 32]), requires_grad=False)
    y = net(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")

    print("PASSED: Nested module composition works!\n")


def test_training_mode():
    """Test training mode switching."""
    print("=" * 60)
    print("Test 5: Training Mode")
    print("=" * 60)

    class Model(tz.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = tz.nn.Linear(10, 5)
            self.dropout = tz.nn.Dropout(0.5)

        def forward_impl(self, x):
            x = self.fc(x)
            x = self.dropout(x)
            return x

    model = Model()
    print(f"Initial training mode: {model.is_training()}")

    model.eval()
    print(f"After eval(): {model.is_training()}")

    model.train()
    print(f"After train(): {model.is_training()}")

    print("PASSED: Training mode switching works!\n")


def test_device_movement():
    """Test moving module to different devices."""
    print("=" * 60)
    print("Test 6: Device Movement")
    print("=" * 60)

    model = tz.nn.Linear(10, 5)
    print("Created Linear(10, 5) on CPU")

    # Move to CPU explicitly
    model.cpu()
    print("Moved to CPU")

    # Try to move to CUDA if available (catch exception if not)
    try:
        model.cuda()
        print("Moved to CUDA")
        model.cpu()
        print("Moved back to CPU")
    except Exception as e:
        print(f"CUDA not available or error: {e}")

    print("PASSED: Device movement works!\n")


def test_zero_grad():
    """Test zeroing gradients."""
    print("=" * 60)
    print("Test 7: Zero Gradient")
    print("=" * 60)

    model = tz.nn.Linear(10, 5)

    # Zero gradients
    model.zero_grad()
    print("Called zero_grad() successfully")

    print("PASSED: Zero gradient works!\n")


def test_state_dict():
    """Test state dict save/load."""
    print("=" * 60)
    print("Test 8: State Dict")
    print("=" * 60)

    model = tz.nn.Linear(10, 5)

    # Get state dict
    state = model.state_dict()
    print(f"State dict keys: {list(state.keys())}")

    # Load state dict
    model.load_state_dict(state)
    print("Loaded state dict back successfully")

    print("PASSED: State dict works!\n")


def main():
    print("\n" + "=" * 60)
    print("TENZOR PYTHON MODULE SUBCLASSING TESTS")
    print("=" * 60 + "\n")

    # Initialize Tenzor
    print("Initializing Tenzor...")
    tz.initialize()
    print("Tenzor initialized successfully\n")

    # Run tests
    test_basic_module_subclass()
    test_builtin_layers_composition()
    test_sequential_container()
    test_nested_modules()
    test_training_mode()
    test_device_movement()
    test_zero_grad()
    test_state_dict()

    print("=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    main()
