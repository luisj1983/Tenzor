"""
Tenzor Tutorial 01: Tensor Basics
==================================
Learn the fundamentals of tensor creation and manipulation in Tenzor.
"""

import tenzor as tz
import numpy as np

def main():
    print("=" * 60)
    print("TENZOR TUTORIAL 01: TENSOR BASICS")
    print("=" * 60)

    # Initialize the Tenzor library (registers backends)
    print("\n[1] Initializing Tenzor library...")
    tz.initialize()
    print("✓ Tenzor initialized successfully")

    # ========================================================================
    # SECTION 1: Tensor Creation
    # ========================================================================
    print("\n" + "=" * 60)
    print("SECTION 1: Creating Tensors")
    print("=" * 60)

    # Create zeros tensor
    print("\n[1.1] Creating a tensor of zeros:")
    zeros = tz.zeros([2, 3], dtype=tz.dtype.float32)
    print(f"  Shape: {zeros.shape}")
    print(f"  Dtype: {zeros.dtype}")
    print(f"  Device: {zeros.device}")

    # Create ones tensor
    print("\n[1.2] Creating a tensor of ones:")
    ones = tz.ones([3, 4], dtype=tz.dtype.float32)
    print(f"  Shape: {ones.shape}")
    print(f"  Dtype: {ones.dtype}")

    # Create random tensor with normal distribution
    print("\n[1.3] Creating a random tensor (normal distribution):")
    random = tz.randn([2, 2], dtype=tz.dtype.float32)
    print(f"  Shape: {random.shape}")

    # ========================================================================
    # SECTION 2: Device Management
    # ========================================================================
    print("\n" + "=" * 60)
    print("SECTION 2: Device Management")
    print("=" * 60)

    # CPU device
    print("\n[2.1] Creating tensor on CPU:")
    cpu_tensor = tz.zeros([2, 2], device=tz.Device.cpu())
    print(f"  Device: {cpu_tensor.device}")

    # Try CUDA if available (will fall back to CPU if not)
    print("\n[2.2] Attempting to create tensor on CUDA:")
    try:
        cuda_tensor = tz.zeros([2, 2], device=tz.Device.cuda(0))
        print(f"  Device: {cuda_tensor.device}")
        print("  ✓ CUDA is available!")
    except Exception as e:
        print(f"  ⚠ CUDA not available (using CPU): {e}")

    # Transfer tensor between devices
    print("\n[2.3] Transferring tensor to different device:")
    cpu_data = tz.ones([3, 3])
    print(f"  Original device: {cpu_data.device}")

    # ========================================================================
    # SECTION 3: Basic Tensor Operations
    # ========================================================================
    print("\n" + "=" * 60)
    print("SECTION 3: Basic Tensor Operations")
    print("=" * 60)

    # Element-wise addition
    print("\n[3.1] Element-wise addition:")
    a = tz.ones([2, 3])
    b = tz.ones([2, 3])
    c = a + b
    print(f"  Result shape: {c.shape}")

    # Element-wise subtraction
    print("\n[3.2] Element-wise subtraction:")
    d = a - b
    print(f"  Result shape: {d.shape}")

    # Element-wise multiplication
    print("\n[3.3] Element-wise multiplication:")
    e = a * b
    print(f"  Result shape: {e.shape}")

    # ========================================================================
    # SECTION 4: Shape Manipulation
    # ========================================================================
    print("\n" + "=" * 60)
    print("SECTION 4: Shape Manipulation")
    print("=" * 60)

    # Reshape tensor
    print("\n[4.1] Reshaping tensor:")
    original = tz.ones([2, 3, 4])
    print(f"  Original shape: {original.shape}")

    reshaped = original.reshape([6, 4])
    print(f"  Reshaped: {reshaped.shape}")

    reshaped2 = original.reshape([8, 3])
    print(f"  Reshaped again: {reshaped2.shape}")

    # ========================================================================
    # SECTION 5: Matrix Multiplication
    # ========================================================================
    print("\n" + "=" * 60)
    print("SECTION 5: Matrix Multiplication")
    print("=" * 60)

    print("\n[5.1] Matrix multiplication:")
    mat1 = tz.ones([3, 4])
    mat2 = tz.ones([4, 5])
    result = tz.matmul(mat1, mat2)
    print(f"  Matrix 1 shape: {mat1.shape}")
    print(f"  Matrix 2 shape: {mat2.shape}")
    print(f"  Result shape: {result.shape}")

    # ========================================================================
    # SECTION 6: Different Data Types
    # ========================================================================
    print("\n" + "=" * 60)
    print("SECTION 6: Different Data Types")
    print("=" * 60)

    print("\n[6.1] Creating tensors with different dtypes:")
    float32_tensor = tz.zeros([2, 2], dtype=tz.dtype.float32)
    print(f"  float32: {float32_tensor.dtype}")

    float64_tensor = tz.zeros([2, 2], dtype=tz.dtype.float64)
    print(f"  float64: {float64_tensor.dtype}")

    int32_tensor = tz.zeros([2, 2], dtype=tz.dtype.int32)
    print(f"  int32: {int32_tensor.dtype}")

    print("\n" + "=" * 60)
    print("TUTORIAL COMPLETE!")
    print("=" * 60)
    print("\nKey Takeaways:")
    print("1. Always initialize Tenzor with tz.initialize()")
    print("2. Create tensors with zeros(), ones(), randn()")
    print("3. Manage device placement with Device.cpu() and Device.cuda()")
    print("4. Perform operations: +, -, *, matmul()")
    print("5. Reshape tensors with .reshape()")
    print("6. Support for multiple dtypes: float32, float64, int32, etc.")

if __name__ == "__main__":
    main()
