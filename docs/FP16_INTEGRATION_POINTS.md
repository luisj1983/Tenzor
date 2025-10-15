# FP16 Conv2d Integration Points

This document shows the exact integration points where FP16 code connects to the existing Tenzor codebase.

## 1. Header Includes (Lines 1-9)

```cpp
#include "tenzor/core/tensor.hpp"
#include "tenzor/core/dtype.hpp"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>  // ← NEW: FP16 support
#include <mma.h>        // ← NEW: Tensor Cores
#include <stdexcept>
#include <vector>
#include <iostream>
```

## 2. Forward Pass Dispatcher (Lines 697-701)

```cpp
// Inside conv2d_forward_kernel() function
// Check dtype and dispatch to appropriate implementation
if (input.dtype() == DType::Float16) {
    // FP16 path with Tensor Cores
    return conv2d_forward_f16(input, weight, bias, stride, padding, dilation, groups, stream);
}

// ... rest of Float32 implementation ...
```

**Location**: `/home/lee/Projects/Tenzor/src/backends/cuda/kernels/conv2d.cu:697-701`

## 3. Backward Pass Dispatcher (Lines 1196-1201)

```cpp
// Inside conv2d_backward_kernel() function
// Check dtype and dispatch to appropriate implementation
if (input.dtype() == DType::Float16) {
    // FP16 path with Tensor Cores
    return conv2d_backward_f16(grad_output, input, weight, stride, padding, dilation, groups,
                               compute_grad_input, compute_grad_weight, compute_grad_bias, stream);
}

// ... rest of Float32 implementation ...
```

**Location**: `/home/lee/Projects/Tenzor/src/backends/cuda/kernels/conv2d.cu:1196-1201`

## 4. Public API Usage (No Changes Required)

The existing public API automatically supports FP16:

```cpp
namespace tenzor {

// Existing function signature (unchanged)
auto conv2d(
    const Tensor& input,
    const Tensor& weight,
    const Tensor* bias = nullptr,
    int64_t stride = 1,
    int64_t padding = 0,
    int64_t dilation = 1,
    int64_t groups = 1
) -> Tensor;

// Implementation (somewhere in tenzor backend)
auto conv2d(...) -> Tensor {
    // ... routing logic ...

    if (device.type == Device::Type::CUDA) {
        cudaStream_t stream = get_cuda_stream(device.index);

        // This automatically calls conv2d_forward_f16 when input is FP16
        return cuda::conv2d_forward_kernel(input, weight, bias, stride,
                                           padding, dilation, groups, stream);
    }

    // ... other device backends ...
}
```

**No changes needed to public API!** FP16 support is transparent to users.

## 5. Autograd Integration (Automatic)

The autograd system automatically detects dtype:

```cpp
// User code
auto x = randn({32, 64, 224, 224}, DType::Float16, Device::cuda(0));
auto w = randn({128, 64, 3, 3}, DType::Float16, Device::cuda(0));
x.requires_grad_(true);
w.requires_grad_(true);

// Forward pass - automatically uses conv2d_forward_f16
auto y = conv2d(x, w);

// Backward pass - automatically uses conv2d_backward_f16
auto grad_y = randn_like(y);
y.backward(grad_y);

// Gradients are in FP16
auto grad_x = x.grad();  // DType::Float16
auto grad_w = w.grad();  // DType::Float16
```

**The autograd graph preserves dtype through the entire computation!**

## 6. Type Checking Flow

```
User creates FP16 tensor
    ↓
tensor.dtype() = DType::Float16
    ↓
conv2d(tensor, ...)
    ↓
cuda::conv2d_forward_kernel(...)
    ↓
Check: input.dtype() == DType::Float16?
    ↓ YES
conv2d_forward_f16(...)  ← FP16 path
    ↓
Uses Tensor Cores (WMMA)
    ↓
Returns FP16 output
```

## 7. DType Enum Integration

The implementation leverages the existing DType system:

```cpp
// From dtype.hpp (no changes needed)
enum class DType : uint8_t {
    Float32,
    Float64,
    Float16,    // ← Already defined
    BFloat16,   // ← Already defined
    // ...
};

// Our dispatcher checks this
if (input.dtype() == DType::Float16) {
    // Use FP16 code path
}
```

## 8. Memory Layout Compatibility

The implementation uses direct memory reinterpretation:

```cpp
// Tenzor's Float16 struct (from dtype.hpp)
struct Float16 {
    uint16_t bits{0};
    // ... methods ...
};

// CUDA's __half type
// Also 16 bits, compatible layout

// Zero-copy conversion in our code
const __half* cuda_ptr = reinterpret_cast<const __half*>(tensor.data<Float16>());
```

**This is safe because both types have identical 16-bit layout.**

## 9. Tensor Core Matmul Integration

Our conv2d implementation uses the Tensor Core matmul:

```cpp
// Inside conv2d_forward_f16()
matmul_f16_tensor_cores(
    weight_ptr,      // Weight matrix
    col_buffer,      // im2col transformed input
    output_ptr,      // Output
    M, N, K,         // Matrix dimensions
    stream           // CUDA stream
);
```

This function is defined in the same file and uses WMMA:

```cpp
__global__ void matmul_tensor_core_f16_kernel(...) {
    using namespace nvcuda::wmma;

    // Declare fragments for Tensor Cores
    fragment<matrix_a, 16, 16, 16, __half, row_major> a_frag;
    fragment<matrix_b, 16, 16, 16, __half, row_major> b_frag;
    fragment<accumulator, 16, 16, 16, __half> acc_frag;

    // Load, compute, store using Tensor Cores
    load_matrix_sync(a_frag, A, K);
    load_matrix_sync(b_frag, B, N);
    mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    store_matrix_sync(C, acc_frag, N, mem_row_major);
}
```

## 10. Complete Call Chain

```
User Code:
  auto y = conv2d(x_fp16, w_fp16);

    ↓

Public API (tenzor/nn/ops.cpp):
  auto conv2d(...) -> Tensor

    ↓

Backend Router:
  if (cuda) cuda::conv2d_forward_kernel(...)

    ↓

CUDA Kernel Dispatcher (conv2d.cu:697):
  if (DType::Float16) conv2d_forward_f16(...)

    ↓

FP16 Conv2d Forward (conv2d.cu:657):
  1. im2col_kernel_f16<<<>>>()
  2. matmul_f16_tensor_cores()
     └─ matmul_tensor_core_f16_kernel<<<>>>()
        └─ nvcuda::wmma::mma_sync() ← Tensor Cores!
  3. add_bias_kernel_f16<<<>>>()

    ↓

Returns FP16 Tensor
```

## 11. Backward Pass Call Chain

```
User Code:
  y.backward(grad_y);

    ↓

Autograd Engine:
  Calls backward function for conv2d

    ↓

CUDA Backward Dispatcher (conv2d.cu:1197):
  if (DType::Float16) conv2d_backward_f16(...)

    ↓

FP16 Conv2d Backward (conv2d.cu:960):
  For grad_input:
    1. matmul_f16_tensor_cores() ← Tensor Cores
    2. col2im_kernel_f16<<<>>>()

  For grad_weight:
    1. im2col_kernel_f16<<<>>>()
    2. matmul_f16_tensor_cores() ← Tensor Cores

  For grad_bias:
    1. sum_bias_grad_kernel_f16<<<>>>()

    ↓

Returns (grad_input, grad_weight, grad_bias) in FP16
```

## 12. Key Design Decisions

### Why reinterpret_cast is safe

```cpp
// Tenzor Float16 layout
struct Float16 { uint16_t bits; };  // 16 bits

// CUDA __half layout
struct __half_raw { unsigned short x; };  // 16 bits

// Same memory layout → safe to reinterpret
const __half* ptr = reinterpret_cast<const __half*>(tensor.data<Float16>());
```

### Why float accumulation in col2im

```cpp
__global__ void col2im_kernel_f16(...) {
    // FP16 has only ~3 decimal digits of precision
    // Accumulating many values causes error accumulation

    float sum = 0.0f;  // Use float for accumulation
    for (...) {
        sum += __half2float(col[idx]);  // Convert for accuracy
    }
    output[idx] = __float2half(sum);  // Convert back at end
}
```

### Why output-centric col2im

```cpp
// Traditional (col-centric with atomics):
//   Each thread: atomicAdd(&output[idx], value)
//   Problem: Multiple threads write to same output → serialization
//   Result: 2-5x slowdown

// Our approach (output-centric):
//   Each thread processes one output element
//   Accumulates from all contributing col positions
//   Direct write (no atomics!)
//   Result: Zero contention, full parallelism
```

## 13. Testing Integration

The test suite automatically tests FP16 when available:

```cpp
// Test framework (pseudocode)
TEST(Conv2d, AllDTypes) {
    for (auto dtype : {DType::Float32, DType::Float64, DType::Float16}) {
        auto x = randn({2, 16, 32, 32}, dtype, Device::cuda(0));
        auto w = randn({32, 16, 3, 3}, dtype, Device::cuda(0));

        auto y = conv2d(x, w);

        EXPECT_EQ(y.dtype(), dtype);  // Output dtype matches input
        EXPECT_EQ(y.shape(), expected_shape);
    }
}
```

## Summary

The FP16 implementation integrates seamlessly with existing code:

1. **Zero API changes**: Users just create FP16 tensors
2. **Automatic dispatch**: Type checking routes to FP16 code
3. **Transparent autograd**: Gradients preserve dtype
4. **Memory compatible**: Safe reinterpret_cast between types
5. **Performance transparent**: Tensor Cores used automatically

**Everything just works!**

---

**Files Modified**: 1 (conv2d.cu)
**API Changes**: 0 (fully backward compatible)
**User-facing changes**: Create tensors with `DType::Float16` → Get Tensor Core acceleration
