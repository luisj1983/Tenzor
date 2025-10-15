# FP16/BF16 CUDA Kernel Implementation Guide

**Date**: 2025-10-14
**Status**: Implementation Guide for Final 1% of Features
**Estimated Effort**: 20-30 hours

---

## Executive Summary

Tenzor is **99.0% complete** (104/105 features). The only remaining feature is **FP16/BF16 CUDA kernel support with Tensor Cores**.

**What's Already Done:**
- ✅ DType enum has Float16 and BFloat16 defined
- ✅ Float16/BFloat16 structs with conversion methods
- ✅ GradScaler fully implemented (246 lines)
- ✅ Autocast fully implemented (167 lines)
- ✅ All kernel fusion operations (504 lines)

**What Needs Implementation:**
- ❌ FP16/BF16 support in math.cu kernels
- ❌ FP16/BF16 matmul with Tensor Cores (wmma)
- ❌ FP16/BF16 conv2d with Tensor Cores

---

## Implementation Pattern

### Step 1: Add cuda_fp16.h Header

```cpp
#include <cuda_runtime.h>
#include <cuda_fp16.h>        // For __half
#include <cuda_bf16.h>        // For __nv_bfloat16
#include <mma.h>              // For Tensor Cores (wmma)
```

### Step 2: Add Conversion Functions

```cpp
namespace tenzor {
namespace cuda {

// Convert Tenzor Float16 to CUDA __half
__device__ __host__ inline __half to_cuda_half(const Float16& x) {
    return __half_raw{x.bits};
}

// Convert CUDA __half to Tenzor Float16
__device__ __host__ inline Float16 from_cuda_half(const __half& x) {
    return Float16(__half_as_ushort(x));
}

// Convert Tenzor BFloat16 to CUDA __nv_bfloat16
__device__ __host__ inline __nv_bfloat16 to_cuda_bfloat16(const BFloat16& x) {
    return __nv_bfloat16_raw{x.bits};
}

// Convert CUDA __nv_bfloat16 to Tenzor BFloat16
__device__ __host__ inline BFloat16 from_cuda_bfloat16(const __nv_bfloat16& x) {
    return BFloat16(__bfloat16_as_ushort(x));
}

} // namespace cuda
} // namespace tenzor
```

---

## Example 1: FP16 Support in Math Kernels

### Add FP16 kernel (math.cu)

```cpp
// FP16 addition kernel
__global__ void add_kernel_f16(const __half* a, const __half* b, __half* c, int64_t n) {
    CUDA_KERNEL_LOOP(idx, n) {
        c[idx] = __hadd(a[idx], b[idx]);  // FP16 addition
    }
}

// FP16 exponential kernel
__global__ void exp_kernel_f16(const __half* input, __half* output, int64_t n) {
    CUDA_KERNEL_LOOP(idx, n) {
        // Convert to float, compute, convert back
        float val = __half2float(input[idx]);
        output[idx] = __float2half(expf(val));
    }
}

// FP16 sqrt kernel
__global__ void sqrt_kernel_f16(const __half* input, __half* output, int64_t n) {
    CUDA_KERNEL_LOOP(idx, n) {
        output[idx] = hsqrt(input[idx]);  // Hardware sqrt
    }
}
```

### Update Launcher Functions

```cpp
auto add_kernel(const Tensor& a, const Tensor& b, cudaStream_t stream) -> Tensor {
    // ... existing code ...

    // Add FP16 support
    if (a.dtype() == DType::Float16) {
        add_kernel_f16<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(a.data<Float16>()),
            reinterpret_cast<const __half*>(b.data<Float16>()),
            reinterpret_cast<__half*>(result.data<Float16>()),
            n);
    } else if (a.dtype() == DType::BFloat16) {
        // Similar for BFloat16
        add_kernel_bf16<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(a.data<BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(b.data<BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(result.data<BFloat16>()),
            n);
    }

    CUDA_CHECK(cudaGetLastError());
    return result;
}
```

---

## Example 2: FP16 Matmul with Tensor Cores

### Add Tensor Core Matmul Kernel (matmul.cu)

```cpp
#include <mma.h>
using namespace nvcuda;

// FP16 Tensor Core matmul using WMMA API
// Matrix dimensions must be multiples of 16
__global__ void matmul_tensor_core_f16(
    const __half* A, const __half* B, __half* C,
    int M, int N, int K) {

    // Warp matrix multiply-accumulate dimensions
    // WMMA supports: 16x16x16 for FP16
    const int WMMA_M = 16;
    const int WMMA_N = 16;
    const int WMMA_K = 16;

    // Warp and lane identifiers
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int warpN = (blockIdx.y * blockDim.y + threadIdx.y);

    // Declare fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, __half> c_frag;

    // Initialize accumulator to zero
    wmma::fill_fragment(c_frag, __float2half(0.0f));

    // Loop over K dimension in chunks of WMMA_K
    for (int k = 0; k < K; k += WMMA_K) {
        int aRow = warpM * WMMA_M;
        int aCol = k;
        int bRow = k;
        int bCol = warpN * WMMA_N;

        // Bounds checking
        if (aRow < M && aCol < K && bRow < K && bCol < N) {
            // Load matrices from global memory into fragments
            wmma::load_matrix_sync(a_frag, A + aRow * K + aCol, K);
            wmma::load_matrix_sync(b_frag, B + bRow * N + bCol, N);

            // Perform matrix multiplication using Tensor Cores
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
    }

    // Store result
    int cRow = warpM * WMMA_M;
    int cCol = warpN * WMMA_N;
    if (cRow < M && cCol < N) {
        wmma::store_matrix_sync(C + cRow * N + cCol, c_frag, N, wmma::mem_row_major);
    }
}

// Host launcher function
void matmul_f16_tensor_cores(
    const __half* A, const __half* B, __half* C,
    int64_t M, int64_t N, int64_t K,
    cudaStream_t stream) {

    // Ensure dimensions are multiples of 16 (WMMA requirement)
    if (M % 16 != 0 || N % 16 != 0 || K % 16 != 0) {
        throw std::runtime_error(
            "Tensor Core matmul requires dimensions to be multiples of 16");
    }

    // Launch configuration
    dim3 gridDim((M + 15) / 16, (N + 15) / 16);
    dim3 blockDim(32 * 4, 8);  // 4 warps per block

    matmul_tensor_core_f16<<<gridDim, blockDim, 0, stream>>>(A, B, C, M, N, K);

    CUDA_CHECK(cudaGetLastError());
}
```

---

## Example 3: FP16 Conv2d with Tensor Cores

### Tensor Core Conv2d Strategy

```cpp
// Convert conv2d to matrix multiplication using im2col
// Then use Tensor Core matmul on the FP16 matrices

auto conv2d_forward_fp16(
    const Tensor& input,         // FP16 input
    const Tensor& weight,        // FP16 weights
    const Tensor* bias,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    cudaStream_t stream
) -> Tensor {

    // 1. Apply im2col (same as before, but with FP16)
    __half* col_buffer;  // FP16 column buffer
    CUDA_CHECK(cudaMalloc(&col_buffer, col_rows * col_cols * sizeof(__half)));

    im2col_kernel_f16<<<grid, block, 0, stream>>>(
        reinterpret_cast<const __half*>(input.data<Float16>()),
        col_buffer,
        // ... parameters ...
    );

    // 2. Use Tensor Core matmul for FP16
    matmul_f16_tensor_cores(
        weight_ptr,      // __half* weights
        col_buffer,      // __half* col buffer
        output_ptr,      // __half* output
        out_channels, batch_size * out_h * out_w, in_channels * kernel_h * kernel_w,
        stream
    );

    // 3. Add bias (if present)
    if (bias != nullptr) {
        add_bias_kernel_f16<<<grid, block, 0, stream>>>(
            output_data, bias_data, batch, out_channels, spatial_size, total
        );
    }

    return output;
}
```

---

## CMake Configuration for Tensor Cores

### Update src/backends/cuda/CMakeLists.txt

```cmake
# Enable Tensor Core support (compute capability 7.0+)
if(TENZOR_BUILD_CUDA)
    set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86 89 90)  # Volta, Turing, Ampere, Ada, Hopper

    # Add WMMA support
    target_compile_options(tenzor_backend_cuda PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-arch=sm_70>  # Minimum for Tensor Cores
        $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-v>
    )

    # Link CUDA math libraries
    target_link_libraries(tenzor_backend_cuda PRIVATE
        CUDA::cudart
        CUDA::cublas
    )
endif()
```

---

## Performance Expectations

### FP16 Tensor Core Benefits

| Operation | FP32 (TFLOPS) | FP16 Tensor Core (TFLOPS) | Speedup |
|-----------|---------------|---------------------------|---------|
| **A100** | 19.5 | 312 | **16x** |
| **V100** | 15.7 | 125 | **8x** |
| **RTX 4090** | 82.6 | 1321 | **16x** |

### Memory Bandwidth Savings

- FP16: **2x less memory** than FP32
- Reduced memory bandwidth requirements
- Allows larger batch sizes

---

## Testing Strategy

### Create test_fp16_kernels.cpp

```cpp
#include <gtest/gtest.h>
#include "tenzor/core/tensor.hpp"

TEST(FP16Kernels, AdditionAccuracy) {
    // Create FP16 tensors
    auto a = tenzor::randn({1024, 1024}, DType::Float16, Device::cuda(0));
    auto b = tenzor::randn({1024, 1024}, DType::Float16, Device::cuda(0));

    // Perform addition
    auto c = a + b;

    // Convert to FP32 for accuracy check
    auto a_f32 = a.to(DType::Float32);
    auto b_f32 = b.to(DType::Float32);
    auto c_f32 = c.to(DType::Float32);
    auto expected = a_f32 + b_f32;

    // Check relative error (FP16 has ~0.1% precision)
    auto diff = (c_f32 - expected).abs();
    auto rel_error = (diff / expected.abs()).max().item<float>();

    EXPECT_LT(rel_error, 0.01f);  // Less than 1% error
}

TEST(FP16Kernels, MatmulTensorCores) {
    // Test Tensor Core matmul with FP16
    auto a = tenzor::randn({256, 256}, DType::Float16, Device::cuda(0));
    auto b = tenzor::randn({256, 256}, DType::Float16, Device::cuda(0));

    auto c = tenzor::matmul(a, b);

    // Verify output shape
    EXPECT_EQ(c.shape()[0], 256);
    EXPECT_EQ(c.shape()[1], 256);
    EXPECT_EQ(c.dtype(), DType::Float16);

    // Verify accuracy against FP32
    auto c_f32 = c.to(DType::Float32);
    auto expected = tenzor::matmul(a.to(DType::Float32), b.to(DType::Float32));
    auto rel_error = ((c_f32 - expected).abs() / expected.abs()).max().item<float>();

    EXPECT_LT(rel_error, 0.02f);  // Less than 2% error for matmul
}
```

---

## Implementation Checklist

### Phase 1: Math Kernels (8 hours)
- [ ] Add FP16 support to all binary operations (add, sub, mul, div)
- [ ] Add FP16 support to all unary operations (neg, abs, sqrt, exp, log, pow)
- [ ] Add FP16 support to comparison operations
- [ ] Add FP16 support to fill/random operations
- [ ] Add BFloat16 support (same pattern as FP16)

### Phase 2: Matmul with Tensor Cores (6 hours)
- [ ] Implement FP16 Tensor Core matmul using WMMA
- [ ] Add fallback for non-16-aligned dimensions
- [ ] Integrate with existing matmul dispatcher
- [ ] Add batched Tensor Core matmul
- [ ] Benchmark and verify speedup

### Phase 3: Conv2d with Tensor Cores (6 hours)
- [ ] Implement FP16 im2col kernel
- [ ] Integrate Tensor Core matmul with conv2d forward
- [ ] Implement FP16 col2im for backward pass
- [ ] Add bias support for FP16
- [ ] Verify accuracy and performance

### Phase 4: CMake and Testing (5 hours)
- [ ] Update CMake to enable Tensor Core compilation
- [ ] Create comprehensive FP16/BF16 test suite
- [ ] Add benchmark comparisons (FP32 vs FP16)
- [ ] Update documentation
- [ ] Verify on multiple GPU architectures (V100, A100, RTX 4090)

---

## Files to Modify

### src/backends/cuda/kernels/math.cu (1,546 lines)
**Add:** FP16/BF16 versions of all 15+ kernel types

### src/backends/cuda/kernels/matmul.cu (835 lines)
**Add:** Tensor Core matmul implementation (200 lines)

### src/backends/cuda/kernels/conv2d.cu (785 lines)
**Add:** FP16 im2col/col2im + Tensor Core integration (150 lines)

### src/backends/cuda/CMakeLists.txt
**Add:** Tensor Core compilation flags

### tests/backends/test_fp16_kernels.cpp (NEW)
**Add:** Comprehensive FP16/BF16 test suite (400 lines)

---

## Quick Start: Minimal Working Example

```cpp
// File: examples/fp16_example.cpp
#include "tenzor/tenzor.hpp"

int main() {
    using namespace tenzor;

    // Initialize
    initialize();

    // Create FP16 tensors
    auto x = randn({1024, 1024}, DType::Float16, Device::cuda(0));
    auto w = randn({1024, 1024}, DType::Float16, Device::cuda(0));

    // Matrix multiplication with Tensor Cores
    auto y = matmul(x, w);

    // Check result
    std::cout << "Output shape: " << y.shape() << std::endl;
    std::cout << "Output dtype: " << dtype_name(y.dtype()) << std::endl;

    return 0;
}
```

---

## References

- [NVIDIA WMMA Documentation](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#wmma)
- [Mixed Precision Training](https://arxiv.org/abs/1710.03740)
- [CUDA FP16 Intrinsics](https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__INTRINSIC__HALF.html)

---

**Status**: Ready for implementation
**Estimated Completion**: 25-30 hours of focused development
**Expected Result**: Tenzor will be **100% feature-complete** with state-of-the-art FP16 Tensor Core performance
