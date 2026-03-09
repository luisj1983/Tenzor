#!/usr/bin/env python3
"""
Test ONNX export Python bindings.
"""

import sys
import os
import tempfile

build_python_dir = os.path.join(os.path.dirname(__file__), '../../build/python')
sys.path.insert(0, build_python_dir)

import tenzor.tenzor_core as tz


def test_onnx_datatype_enum():
    """Verify ONNX DataType enum values are accessible."""
    print("Testing ONNX DataType enum...")
    assert tz.onnx.DataType.FLOAT is not None
    assert tz.onnx.DataType.DOUBLE is not None
    assert tz.onnx.DataType.FLOAT16 is not None
    assert tz.onnx.DataType.INT32 is not None
    assert tz.onnx.DataType.INT64 is not None
    assert tz.onnx.DataType.BFLOAT16 is not None
    print("  ONNX DataType enum OK")


def test_onnx_dtype_conversion():
    """Test dtype_to_onnx conversion utility."""
    print("Testing dtype conversion...")
    onnx_f32 = tz.onnx.dtype_to_onnx(tz.dtype.float32)
    assert onnx_f32 == tz.onnx.DataType.FLOAT
    onnx_f64 = tz.onnx.dtype_to_onnx(tz.dtype.float64)
    assert onnx_f64 == tz.onnx.DataType.DOUBLE
    print("  dtype conversion OK")


def test_onnx_graph_construction():
    """Test manual ONNX graph construction."""
    print("Testing ONNX graph construction...")
    graph = tz.onnx.Graph("test_graph")
    assert graph.name == "test_graph"

    # Add input
    inp = tz.onnx.ValueInfo("input", tz.onnx.DataType.FLOAT, [1, 10])
    graph.add_input(inp)

    # Add output
    out = tz.onnx.ValueInfo("output", tz.onnx.DataType.FLOAT, [1, 5])
    graph.add_output(out)

    # Add node
    node = tz.onnx.Node("MatMul", "matmul_0")
    node.add_input("input")
    node.add_output("output")
    graph.add_node(node)

    assert len(graph.nodes) == 1
    assert len(graph.inputs) == 1
    assert len(graph.outputs) == 1
    print("  ONNX graph construction OK")


def test_export_linear_model():
    """Export a Linear model to ONNX file."""
    print("Testing export linear model...")
    model = tz.nn.Linear(10, 5)
    dummy = tz.randn([1, 10])

    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        path = f.name

    try:
        tz.onnx.export(model, dummy, path,
                       input_names=["input"],
                       output_names=["output"],
                       opset_version=13)

        assert os.path.exists(path), f"ONNX file not created: {path}"
        file_size = os.path.getsize(path)
        assert file_size > 0, f"ONNX file is empty"
        print(f"  exported file size: {file_size} bytes")
    finally:
        if os.path.exists(path):
            os.unlink(path)
    print("  export linear model OK")


def test_export_conv_model():
    """Export a Conv2d model to ONNX file."""
    print("Testing export conv model...")
    model = tz.nn.Conv2d(3, 16, 3, stride=1, padding=1)
    dummy = tz.randn([1, 3, 32, 32])

    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        path = f.name

    try:
        tz.onnx.export(model, dummy, path,
                       input_names=["image"],
                       output_names=["features"],
                       opset_version=13)

        assert os.path.exists(path), f"ONNX file not created"
        assert os.path.getsize(path) > 0, "ONNX file is empty"
    finally:
        if os.path.exists(path):
            os.unlink(path)
    print("  export conv model OK")


def test_exporter_api():
    """Test low-level ONNXExporter API."""
    print("Testing ONNXExporter API...")
    exporter = tz.onnx.Exporter(opset_version=13)
    exporter.set_model_name("test_model")
    exporter.set_producer_name("tenzor")
    exporter.set_description("Test model")

    inp = tz.randn([1, 10])
    exporter.add_input(inp, "input")

    graph = exporter.get_graph()
    assert graph is not None
    exporter.clear()
    print("  ONNXExporter API OK")


def main():
    print("=" * 60)
    print("Testing ONNX Export Bindings")
    print("=" * 60)

    try:
        tz.initialize()

        test_onnx_datatype_enum()
        test_onnx_dtype_conversion()
        test_onnx_graph_construction()
        test_export_linear_model()
        test_export_conv_model()
        test_exporter_api()

        print("\n" + "=" * 60)
        print("All ONNX export tests PASSED!")
        print("=" * 60)
        return 0

    except Exception as e:
        print(f"\nFAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
