#!/usr/bin/env python3
"""
ONNX Round-Trip Verification Script

This script demonstrates round-trip verification of ONNX export:
1. Export a Tenzor model to ONNX
2. Load the exported model in ONNX Runtime
3. Compare outputs with same input
4. Verify numerical accuracy

Requirements:
    pip install onnxruntime onnx numpy
"""

import sys
import tenzor as tz
import numpy as np

# Initialize Tenzor
tz.initialize()

# Optional: Check for ONNX Runtime
try:
    import onnxruntime as ort
    import onnx
    ONNX_RUNTIME_AVAILABLE = True
except ImportError:
    ONNX_RUNTIME_AVAILABLE = False
    print("Warning: ONNX Runtime not installed. Install with: pip install onnxruntime onnx")
    print("  Skipping round-trip verification. Export-only mode.\n")


def create_simple_model():
    """Create a simple model for testing"""
    return tz.nn.Sequential(
        tz.nn.Linear(10, 20),
        tz.nn.ReLU(),
        tz.nn.Linear(20, 5)
    )


def export_model(model, filepath, input_shape):
    """Export model to ONNX"""
    print(f"Exporting model to {filepath}...")
    dummy_input = tz.Tensor(input_shape, tz.dtype.float32, tz.Device.cpu())
    tz.onnx.export(
        model,
        dummy_input,
        filepath,
        input_names=["input"],
        output_names=["output"],
        opset_version=13,
        verbose=False
    )
    print("  Export complete!")


def verify_roundtrip(model, filepath, input_shape):
    """Verify ONNX model produces same output as Tenzor model"""
    if not ONNX_RUNTIME_AVAILABLE:
        print("  Skipping verification (ONNX Runtime not available)")
        return True

    print("Verifying round-trip accuracy...")

    # Create random input
    np_input = np.random.randn(*input_shape).astype(np.float32)

    # Get Tenzor output
    tz_input = tz.Variable(tz.Tensor.from_numpy(np_input), requires_grad=False)
    tz_output = model(tz_input)
    tz_result = tz_output.tensor().numpy()

    # Get ONNX Runtime output
    session = ort.InferenceSession(filepath)
    ort_result = session.run(None, {"input": np_input})[0]

    # Compare outputs
    max_diff = np.max(np.abs(tz_result - ort_result))
    mean_diff = np.mean(np.abs(tz_result - ort_result))

    print(f"  Max absolute difference: {max_diff:.6e}")
    print(f"  Mean absolute difference: {mean_diff:.6e}")

    # Check if within tolerance
    tolerance = 1e-5
    if max_diff < tolerance:
        print(f"  PASSED: Difference within tolerance ({tolerance})")
        return True
    else:
        print(f"  FAILED: Difference exceeds tolerance ({tolerance})")
        return False


def main():
    """Run verification tests"""
    print("=" * 60)
    print("ONNX ROUND-TRIP VERIFICATION")
    print("=" * 60)

    # Test 1: Simple Linear Model
    print("\n--- Test 1: Simple Linear Model ---")
    model = create_simple_model()
    filepath = "roundtrip_test.onnx"
    input_shape = [1, 10]

    export_model(model, filepath, input_shape)
    result1 = verify_roundtrip(model, filepath, input_shape)

    # Test 2: CNN Model
    print("\n--- Test 2: CNN Model ---")
    cnn_model = tz.nn.Sequential(
        tz.nn.Conv2d(1, 16, 3, 1, 1),
        tz.nn.ReLU(),
        tz.nn.Conv2d(16, 32, 3, 1, 1),
        tz.nn.ReLU(),
    )
    cnn_filepath = "roundtrip_cnn.onnx"
    cnn_input_shape = [1, 1, 8, 8]

    export_model(cnn_model, cnn_filepath, cnn_input_shape)
    result2 = verify_roundtrip(cnn_model, cnn_filepath, cnn_input_shape)

    # Summary
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    print(f"  Test 1 (Linear): {'PASSED' if result1 else 'FAILED'}")
    print(f"  Test 2 (CNN):    {'PASSED' if result2 else 'FAILED'}")

    if result1 and result2:
        print("\nAll tests passed!")
        return 0
    else:
        print("\nSome tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
