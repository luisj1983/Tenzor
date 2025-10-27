#!/usr/bin/env python3
"""
ONNX Export Demonstration for Tenzor

This example demonstrates how to export Tenzor models to ONNX format,
including simple models, complex architectures, and dynamic shapes.
"""

import sys
import os

# Add build directory to path (adjust as needed)
build_dir = os.path.join(os.path.dirname(__file__), '..', 'build')
if os.path.exists(build_dir):
    sys.path.insert(0, build_dir)

try:
    import tenzor_core as tz
except ImportError:
    print("Error: tenzor_core module not found.")
    print(f"Please build Tenzor first, or adjust the build_dir path.")
    print(f"Looking in: {build_dir}")
    sys.exit(1)

import numpy as np


def example_1_simple_linear():
    """Export a simple linear layer to ONNX"""
    print("\n=== Example 1: Simple Linear Layer ===")

    # Initialize Tenzor
    tz.initialize()

    # Create ONNX exporter
    exporter = tz.onnx.Exporter(opset_version=13)
    exporter.set_model_name("simple_linear")
    exporter.set_description("Single linear layer example")

    # Create tensors
    input_t = tz.Tensor([1, 10], tz.DType.Float32, tz.Device.cpu())
    weight = tz.Tensor([20, 10], tz.DType.Float32, tz.Device.cpu())
    bias = tz.Tensor([20], tz.DType.Float32, tz.Device.cpu())
    output_t = tz.Tensor([1, 20], tz.DType.Float32, tz.Device.cpu())

    # Build graph
    exporter.add_input(input_t, "input", {})
    exporter.export_linear(input_t, weight, bias, output_t, "output")
    exporter.add_output(output_t, "output")

    # Export to file
    filepath = "simple_linear.onnx"
    exporter.export_to_file(filepath)
    print(f"✓ Exported to {filepath}")
    print(f"  Model: {exporter.get_graph().name}")
    print(f"  Nodes: {len(exporter.get_graph().nodes)}")
    print(f"  Inputs: {len(exporter.get_graph().inputs)}")
    print(f"  Outputs: {len(exporter.get_graph().outputs)}")


def example_2_cnn_architecture():
    """Export a CNN architecture to ONNX"""
    print("\n=== Example 2: CNN Architecture ===")

    tz.initialize()

    exporter = tz.onnx.Exporter(opset_version=13)
    exporter.set_model_name("simple_cnn")
    exporter.set_description("Simple CNN for MNIST-like data")

    # Input: 1x28x28 (grayscale image)
    input_t = tz.Tensor([1, 1, 28, 28], tz.DType.Float32, tz.Device.cpu())
    exporter.add_input(input_t, "input", {})

    # Conv1: 1 -> 32 channels
    conv1_w = tz.Tensor([32, 1, 3, 3], tz.DType.Float32, tz.Device.cpu())
    conv1_b = tz.Tensor([32], tz.DType.Float32, tz.Device.cpu())
    conv1_out = tz.Tensor([1, 32, 26, 26], tz.DType.Float32, tz.Device.cpu())
    exporter.export_conv2d(input_t, conv1_w, conv1_b, [3, 3], [1, 1], [0, 0], [1, 1], 1, conv1_out, "conv1")

    # ReLU1
    relu1_out = tz.Tensor([1, 32, 26, 26], tz.DType.Float32, tz.Device.cpu())
    exporter.export_relu(conv1_out, relu1_out, "relu1")

    # MaxPool1
    pool1_out = tz.Tensor([1, 32, 13, 13], tz.DType.Float32, tz.Device.cpu())
    exporter.export_maxpool2d(relu1_out, 2, 2, 0, pool1_out, "pool1")

    # Conv2: 32 -> 64 channels
    conv2_w = tz.Tensor([64, 32, 3, 3], tz.DType.Float32, tz.Device.cpu())
    conv2_b = tz.Tensor([64], tz.DType.Float32, tz.Device.cpu())
    conv2_out = tz.Tensor([1, 64, 11, 11], tz.DType.Float32, tz.Device.cpu())
    exporter.export_conv2d(pool1_out, conv2_w, conv2_b, [3, 3], [1, 1], [0, 0], [1, 1], 1, conv2_out, "conv2")

    # ReLU2
    relu2_out = tz.Tensor([1, 64, 11, 11], tz.DType.Float32, tz.Device.cpu())
    exporter.export_relu(conv2_out, relu2_out, "relu2")

    # MaxPool2
    pool2_out = tz.Tensor([1, 64, 5, 5], tz.DType.Float32, tz.Device.cpu())
    exporter.export_maxpool2d(relu2_out, 2, 2, 0, pool2_out, "pool2")

    # Flatten
    flatten_out = tz.Tensor([1, 1600], tz.DType.Float32, tz.Device.cpu())
    exporter.export_reshape(pool2_out, [1, 1600], flatten_out, "flatten")

    # FC1: 1600 -> 128
    fc1_w = tz.Tensor([128, 1600], tz.DType.Float32, tz.Device.cpu())
    fc1_b = tz.Tensor([128], tz.DType.Float32, tz.Device.cpu())
    fc1_out = tz.Tensor([1, 128], tz.DType.Float32, tz.Device.cpu())
    exporter.export_linear(flatten_out, fc1_w, fc1_b, fc1_out, "fc1")

    # ReLU3
    relu3_out = tz.Tensor([1, 128], tz.DType.Float32, tz.Device.cpu())
    exporter.export_relu(fc1_out, relu3_out, "relu3")

    # FC2: 128 -> 10 (output classes)
    fc2_w = tz.Tensor([10, 128], tz.DType.Float32, tz.Device.cpu())
    fc2_b = tz.Tensor([10], tz.DType.Float32, tz.Device.cpu())
    output_t = tz.Tensor([1, 10], tz.DType.Float32, tz.Device.cpu())
    exporter.export_linear(relu3_out, fc2_w, fc2_b, output_t, "output")

    exporter.add_output(output_t, "output")

    # Export
    filepath = "simple_cnn.onnx"
    exporter.export_to_file(filepath)
    print(f"✓ Exported to {filepath}")
    print(f"  Model: {exporter.get_graph().name}")
    print(f"  Nodes: {len(exporter.get_graph().nodes)}")
    print(f"  Layers: Conv(2) + Pool(2) + Linear(2) + ReLU(3) + Reshape(1)")


def example_3_resnet_block():
    """Export a ResNet-like block with skip connection"""
    print("\n=== Example 3: ResNet Block with Skip Connection ===")

    tz.initialize()

    exporter = tz.onnx.Exporter(opset_version=13)
    exporter.set_model_name("resnet_block")
    exporter.set_description("ResNet block with skip connection")

    # Input
    input_t = tz.Tensor([1, 64, 32, 32], tz.DType.Float32, tz.Device.cpu())
    exporter.add_input(input_t, "input", {})

    # Main path: Conv -> BN -> ReLU
    conv1_w = tz.Tensor([64, 64, 3, 3], tz.DType.Float32, tz.Device.cpu())
    conv1_b = tz.Tensor([64], tz.DType.Float32, tz.Device.cpu())
    conv1_out = tz.Tensor([1, 64, 32, 32], tz.DType.Float32, tz.Device.cpu())
    exporter.export_conv2d(input_t, conv1_w, conv1_b, [3, 3], [1, 1], [1, 1], [1, 1], 1, conv1_out, "conv1")

    # BatchNorm
    scale = tz.Tensor([64], tz.DType.Float32, tz.Device.cpu())
    bn_bias = tz.Tensor([64], tz.DType.Float32, tz.Device.cpu())
    mean = tz.Tensor([64], tz.DType.Float32, tz.Device.cpu())
    var = tz.Tensor([64], tz.DType.Float32, tz.Device.cpu())
    bn_out = tz.Tensor([1, 64, 32, 32], tz.DType.Float32, tz.Device.cpu())
    exporter.export_batchnorm2d(conv1_out, scale, bn_bias, mean, var, 1e-5, bn_out, "bn1")

    # ReLU
    relu_out = tz.Tensor([1, 64, 32, 32], tz.DType.Float32, tz.Device.cpu())
    exporter.export_relu(bn_out, relu_out, "relu1")

    # Skip connection: Add input + relu_out
    output_t = tz.Tensor([1, 64, 32, 32], tz.DType.Float32, tz.Device.cpu())
    exporter.export_add(input_t, relu_out, output_t, "output")

    exporter.add_output(output_t, "output")

    # Export
    filepath = "resnet_block.onnx"
    exporter.export_to_file(filepath)
    print(f"✓ Exported to {filepath}")
    print(f"  Model: {exporter.get_graph().name}")
    print(f"  Nodes: {len(exporter.get_graph().nodes)}")
    print(f"  Skip connection: input -> Add <- Conv->BN->ReLU")


def example_4_dynamic_shapes():
    """Export a model with dynamic batch size"""
    print("\n=== Example 4: Dynamic Batch Size ===")

    tz.initialize()

    exporter = tz.onnx.Exporter(opset_version=13)
    exporter.set_model_name("dynamic_batch")
    exporter.set_description("Model with dynamic batch dimension")

    # Create model with dynamic batch
    input_t = tz.Tensor([1, 10], tz.DType.Float32, tz.Device.cpu())
    weight = tz.Tensor([20, 10], tz.DType.Float32, tz.Device.cpu())
    bias = tz.Tensor([20], tz.DType.Float32, tz.Device.cpu())
    output_t = tz.Tensor([1, 20], tz.DType.Float32, tz.Device.cpu())

    # Mark dimension 0 (batch) as dynamic
    dynamic_axes = {0: "batch"}
    exporter.add_input(input_t, "input", dynamic_axes)

    exporter.export_linear(input_t, weight, bias, output_t, "output")
    exporter.add_output(output_t, "output")

    # Export
    filepath = "dynamic_batch.onnx"
    exporter.export_to_file(filepath)
    print(f"✓ Exported to {filepath}")
    print(f"  Dynamic axes: batch dimension (dim 0)")
    print(f"  Input shape: [batch, 10]")
    print(f"  Output shape: [batch, 20]")


def example_5_all_activations():
    """Demonstrate all supported activation functions"""
    print("\n=== Example 5: All Activation Functions ===")

    tz.initialize()

    activations = [
        ("ReLU", lambda exp, inp, out: exp.export_relu(inp, out, "relu")),
        ("LeakyReLU", lambda exp, inp, out: exp.export_leaky_relu(inp, 0.01, out, "leaky_relu")),
        ("Sigmoid", lambda exp, inp, out: exp.export_sigmoid(inp, out, "sigmoid")),
        ("Tanh", lambda exp, inp, out: exp.export_tanh(inp, out, "tanh")),
        ("GELU", lambda exp, inp, out: exp.export_gelu(inp, out, "gelu")),
        ("ELU", lambda exp, inp, out: exp.export_elu(inp, 1.0, out, "elu")),
        ("SELU", lambda exp, inp, out: exp.export_selu(inp, out, "selu")),
        ("Swish", lambda exp, inp, out: exp.export_swish(inp, out, "swish")),
    ]

    print(f"Supported activation functions: {len(activations)}")
    for name, export_fn in activations:
        exporter = tz.onnx.Exporter(opset_version=13)
        exporter.set_model_name(f"{name.lower()}_activation")

        input_t = tz.Tensor([1, 64], tz.DType.Float32, tz.Device.cpu())
        output_t = tz.Tensor([1, 64], tz.DType.Float32, tz.Device.cpu())

        exporter.add_input(input_t, "input", {})
        export_fn(exporter, input_t, output_t)
        exporter.add_output(output_t, "output")

        filepath = f"{name.lower()}_activation.onnx"
        exporter.export_to_file(filepath)
        print(f"  ✓ {name}: {filepath}")


def example_6_export_to_bytes():
    """Export model to bytes instead of file"""
    print("\n=== Example 6: Export to Bytes ===")

    tz.initialize()

    exporter = tz.onnx.Exporter(opset_version=13)
    exporter.set_model_name("bytes_export")

    input_t = tz.Tensor([1, 10], tz.DType.Float32, tz.Device.cpu())
    output_t = tz.Tensor([1, 10], tz.DType.Float32, tz.Device.cpu())

    exporter.add_input(input_t, "input", {})
    exporter.export_relu(input_t, output_t, "output")
    exporter.add_output(output_t, "output")

    # Export to bytes
    onnx_bytes = exporter.export_to_bytes()

    print(f"✓ Exported to bytes")
    print(f"  Size: {len(onnx_bytes)} bytes")
    print(f"  Can be sent over network or stored in memory")

    # Optionally write to file manually
    with open("from_bytes.onnx", "wb") as f:
        f.write(bytes(onnx_bytes))
    print(f"  ✓ Written to from_bytes.onnx")


def main():
    """Run all examples"""
    print("=" * 60)
    print("TENZOR ONNX EXPORT EXAMPLES")
    print("=" * 60)

    try:
        example_1_simple_linear()
        example_2_cnn_architecture()
        example_3_resnet_block()
        example_4_dynamic_shapes()
        example_5_all_activations()
        example_6_export_to_bytes()

        print("\n" + "=" * 60)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print("\nExported ONNX models can be loaded in:")
        print("  • ONNX Runtime (Python, C++, C#)")
        print("  • TensorRT (NVIDIA)")
        print("  • OpenVINO (Intel)")
        print("  • CoreML (Apple)")
        print("  • TensorFlow/Keras (via onnx-tf)")
        print("  • PyTorch (via torch.onnx.load)")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
