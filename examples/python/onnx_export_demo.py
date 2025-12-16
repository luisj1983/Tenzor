#!/usr/bin/env python3
"""
ONNX Export Demonstration for Tenzor

This example demonstrates how to export Tenzor models to ONNX format
using the high-level export API.
"""

import sys
import tenzor as tz
import numpy as np


def example_1_simple_linear():
    """Export a simple linear model to ONNX"""
    print("\n=== Example 1: Simple Linear Model ===")

    # Create a simple model
    model = tz.nn.Sequential(
        tz.nn.Linear(10, 20),
        tz.nn.ReLU(),
        tz.nn.Linear(20, 5)
    )

    # Create dummy input for shape inference
    dummy_input = tz.Tensor([1, 10], tz.dtype.float32, tz.Device.cpu())

    # Export to ONNX
    filepath = "simple_linear.onnx"
    tz.onnx.export(
        model,
        dummy_input,
        filepath,
        input_names=["input"],
        output_names=["output"],
        opset_version=13,
        verbose=True
    )
    print(f"Exported to {filepath}")


def example_2_cnn_model():
    """Export a CNN model to ONNX"""
    print("\n=== Example 2: CNN Model ===")

    # Create a simple CNN
    model = tz.nn.Sequential(
        tz.nn.Conv2d(1, 32, 3, 1, 1),  # 28x28 -> 28x28
        tz.nn.ReLU(),
        tz.nn.MaxPool2d(2, 2),  # 28x28 -> 14x14
        tz.nn.Conv2d(32, 64, 3, 1, 1),  # 14x14 -> 14x14
        tz.nn.ReLU(),
        tz.nn.MaxPool2d(2, 2),  # 14x14 -> 7x7
        tz.nn.Flatten(1),
        tz.nn.Linear(64 * 7 * 7, 128),
        tz.nn.ReLU(),
        tz.nn.Linear(128, 10)
    )

    # Create dummy input for shape inference (batch=1, channels=1, H=28, W=28)
    dummy_input = tz.Tensor([1, 1, 28, 28], tz.dtype.float32, tz.Device.cpu())

    # Export to ONNX
    filepath = "cnn_model.onnx"
    tz.onnx.export(
        model,
        dummy_input,
        filepath,
        input_names=["image"],
        output_names=["logits"],
        opset_version=13,
        verbose=True
    )
    print(f"Exported to {filepath}")


def example_3_custom_model():
    """Export a custom Python-defined model to ONNX"""
    print("\n=== Example 3: Custom Model ===")

    class ResidualBlock(tz.nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.conv1 = tz.nn.Conv2d(channels, channels, 3, 1, 1)
            self.bn1 = tz.nn.BatchNorm2d(channels)
            self.conv2 = tz.nn.Conv2d(channels, channels, 3, 1, 1)
            self.bn2 = tz.nn.BatchNorm2d(channels)

        def forward(self, x):
            residual = x
            out = tz.nn.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out = out + residual
            return tz.nn.relu(out)

    class CustomNet(tz.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = tz.nn.Conv2d(3, 64, 3, 1, 1)
            self.block1 = ResidualBlock(64)
            self.block2 = ResidualBlock(64)
            self.avgpool = tz.nn.AdaptiveAvgPool2d(1, 1)
            self.flatten = tz.nn.Flatten(1)
            self.fc = tz.nn.Linear(64, 10)

        def forward(self, x):
            x = tz.nn.relu(self.conv1(x))
            x = self.block1(x)
            x = self.block2(x)
            x = self.avgpool(x)
            x = self.flatten(x)
            return self.fc(x)

    model = CustomNet()

    # Create dummy input
    dummy_input = tz.Tensor([1, 3, 32, 32], tz.dtype.float32, tz.Device.cpu())

    # Export to ONNX
    filepath = "custom_model.onnx"
    tz.onnx.export(
        model,
        dummy_input,
        filepath,
        input_names=["input"],
        output_names=["output"],
        opset_version=13,
        verbose=True
    )
    print(f"Exported to {filepath}")


def main():
    """Run all examples"""
    print("=" * 60)
    print("TENZOR ONNX EXPORT EXAMPLES")
    print("=" * 60)

    # Initialize Tenzor
    tz.initialize()

    try:
        example_1_simple_linear()
        example_2_cnn_model()
        example_3_custom_model()

        print("\n" + "=" * 60)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print("\nExported ONNX models can be loaded in:")
        print("  - ONNX Runtime (Python, C++, C#)")
        print("  - TensorRT (NVIDIA)")
        print("  - OpenVINO (Intel)")
        print("  - CoreML (Apple)")
        print("  - TensorFlow/Keras (via onnx-tf)")
        print("  - PyTorch (via torch.onnx.load)")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
