# ROCm/HIP Porting Guide for Tenzor CUDA Backend

This document provides a comprehensive guide for porting the Tenzor CUDA backend to ROCm/HIP, enabling AMD GPU support.

## Table of Contents
1. [Overview](#overview)
2. [CUDA to HIP API Mapping](#cuda-to-hip-api-mapping)
3. [Kernel Files Analysis](#kernel-files-analysis)
4. [Backend Implementation](#backend-implementation)
5. [Build System Changes](#build-system-changes)
6. [Step-by-Step Porting Process](#step-by-step-porting-process)
7. [Testing Strategy](#testing-strategy)

---

## Overview

### Project Structure
```
src/backends/cuda/
├── cuda_backend.cpp              # Main backend interface
├── kernels/
│   ├── activations.cu            # Activation functions (ReLU, Sigmoid, Tanh, Softmax)
│   ├── transform.cu              # Tensor transformations (reshape, transpose, etc.)
│   ├── batchnorm.cu              # Batch normalization operations
│   ├── conv2d.cu                 # 2D convolution operations
│   ├── lstm.cu                   # LSTM cell operations
│   ├── gru.cu                    # GRU cell operations
│   ├── math.cu                   # Element-wise math operations
│   ├── matmul.cu                 # Matrix multiplication
│   ├── reduction.cu              # Reduction operations (sum, mean, max, min)
│   └── fused_ops.cu              # Fused operations for performance
└── caching_allocator.hpp/cpp     # Memory caching allocator
```

### Statistics
- **Total Kernel Files**: 10 CUDA kernel files
- **Total Functions**: ~60+ kernel functions
- **Lines of Code**: ~10,000+ lines
- **External Dependencies**:
  - CUDA Runtime API
  - cuBLAS (for matrix multiplication)
  - cuRAND (for random number generation)
  - CUB (for fused operations)

---

## CUDA to HIP API Mapping

### 1. Runtime API

| CUDA API | HIP API | Notes |
|----------|---------|-------|
| `cudaError_t` | `hipError_t` | Error type |
| `cudaSuccess` | `hipSuccess` | Success code |
| `cudaGetErrorString()` | `hipGetErrorString()` | Error string conversion |
| `cudaGetLastError()` | `hipGetLastError()` | Get last error |
| `cudaMalloc()` | `hipMalloc()` | Device memory allocation |
| `cudaFree()` | `hipFree()` | Device memory deallocation |
| `cudaMemcpy()` | `hipMemcpy()` | Memory copy |
| `cudaMemcpyAsync()` | `hipMemcpyAsync()` | Async memory copy |
| `cudaMemcpyKind` | `hipMemcpyKind` | Copy direction enum |
| `cudaMemcpyHostToHost` | `hipMemcpyHostToHost` | H2H copy |
| `cudaMemcpyHostToDevice` | `hipMemcpyHostToDevice` | H2D copy |
| `cudaMemcpyDeviceToHost` | `hipMemcpyDeviceToHost` | D2H copy |
| `cudaMemcpyDeviceToDevice` | `hipMemcpyDeviceToDevice` | D2D copy |
| `cudaMemset()` | `hipMemset()` | Memory set |
| `cudaMemsetAsync()` | `hipMemsetAsync()` | Async memory set |
| `cudaDeviceSynchronize()` | `hipDeviceSynchronize()` | Device synchronization |
| `cudaStreamSynchronize()` | `hipStreamSynchronize()` | Stream synchronization |
| `cudaGetDeviceCount()` | `hipGetDeviceCount()` | Get device count |
| `cudaSetDevice()` | `hipSetDevice()` | Set active device |
| `cudaStream_t` | `hipStream_t` | Stream handle type |
| `cudaStreamCreate()` | `hipStreamCreate()` | Create stream |
| `cudaStreamDestroy()` | `hipStreamDestroy()` | Destroy stream |
| `cudaPointerGetAttributes()` | `hipPointerGetAttributes()` | Query pointer attributes |
| `cudaPointerAttributes` | `hipPointerAttribute_t` | Pointer attributes struct |

### 2. Device Code Qualifiers

| CUDA | HIP | Notes |
|------|-----|-------|
| `__global__` | `__global__` | Kernel function |
| `__device__` | `__device__` | Device function |
| `__host__` | `__host__` | Host function |
| `__forceinline__` | `__forceinline__` | Force inline |
| `__shared__` | `__shared__` | Shared memory |
| `__restrict__` | `__restrict__` | Restrict pointer |
| `__syncthreads()` | `__syncthreads()` | Block synchronization |
| `__align__(N)` | `__align__(N)` | Memory alignment |

### 3. Built-in Variables

| CUDA | HIP | Notes |
|------|-----|-------|
| `threadIdx.x/y/z` | `threadIdx.x/y/z` | Thread index |
| `blockIdx.x/y/z` | `blockIdx.x/y/z` | Block index |
| `blockDim.x/y/z` | `blockDim.x/y/z` | Block dimension |
| `gridDim.x/y/z` | `gridDim.x/y/z` | Grid dimension |

### 4. Warp/Wave Intrinsics

| CUDA | HIP | Notes |
|------|-----|-------|
| `__shfl_down_sync()` | `__shfl_down()` | Shuffle down (note: no _sync suffix) |
| `__shfl_up_sync()` | `__shfl_up()` | Shuffle up |
| `__shfl_sync()` | `__shfl()` | Shuffle |
| `__shfl_xor_sync()` | `__shfl_xor()` | Shuffle XOR |
| Warp size: 32 | Wave size: 64 (RDNA/CDNA) or 32 (older) | **CRITICAL DIFFERENCE** |

### 5. Math Functions

| CUDA | HIP | Notes |
|------|-----|-------|
| `expf()` | `expf()` | Same |
| `logf()` | `logf()` | Same |
| `sqrtf()` | `sqrtf()` | Same |
| `rsqrtf()` | `rsqrtf()` | Same (reciprocal sqrt) |
| `fabsf()` | `fabsf()` | Same |
| `fminf()` | `fminf()` | Same |
| `fmaxf()` | `fmaxf()` | Same |
| `powf()` | `powf()` | Same |
| `tanhf()` | `tanhf()` | Same |
| `exp()` | `exp()` | Double precision |
| `log()` | `log()` | Double precision |
| `sqrt()` | `sqrt()` | Double precision |
| `tanh()` | `tanh()` | Double precision |

### 6. cuBLAS → rocBLAS

| cuBLAS | rocBLAS | Notes |
|--------|---------|-------|
| `cublasHandle_t` | `rocblas_handle` | Handle type |
| `cublasCreate()` | `rocblas_create_handle()` | Create handle |
| `cublasDestroy()` | `rocblas_destroy_handle()` | Destroy handle |
| `cublasSetStream()` | `rocblas_set_stream()` | Set stream |
| `cublasSgemm()` | `rocblas_sgemm()` | Single precision GEMM |
| `cublasDgemm()` | `rocblas_dgemm()` | Double precision GEMM |
| `cublasSgemmStridedBatched()` | `rocblas_sgemm_strided_batched()` | Batched SGEMM |
| `cublasDgemmStridedBatched()` | `rocblas_dgemm_strided_batched()` | Batched DGEMM |
| `cublasStatus_t` | `rocblas_status` | Status type |
| `CUBLAS_STATUS_SUCCESS` | `rocblas_status_success` | Success code |
| `CUBLAS_OP_N` | `rocblas_operation_none` | No transpose |
| `CUBLAS_OP_T` | `rocblas_operation_transpose` | Transpose |

### 7. cuRAND → rocRAND

| cuRAND | rocRAND | Notes |
|--------|---------|-------|
| `curandState` | `rocrand_state_philox4x32_10` | RNG state (Philox recommended) |
| `curand_init()` | `rocrand_init()` | Initialize state |
| `curand_uniform()` | `rocrand_uniform()` | Uniform distribution |
| `curand_normal()` | `rocrand_normal()` | Normal distribution |

### 8. Header Files

| CUDA | HIP | Notes |
|------|-----|-------|
| `<cuda_runtime.h>` | `<hip/hip_runtime.h>` | Runtime API |
| `<cuda_fp16.h>` | `<hip/hip_fp16.h>` | FP16 support |
| `<device_launch_parameters.h>` | `<hip/hip_runtime.h>` | Included in runtime |
| `<cublas_v2.h>` | `<rocblas/rocblas.h>` | BLAS library |
| `<curand_kernel.h>` | `<rocrand/rocrand_kernel.h>` | Random number generation |
| `<cub/cub.cuh>` | `<hipcub/hipcub.hpp>` | CUB primitives |

---

## Kernel Files Analysis

### 1. activations.cu (1,018 lines)

**Purpose**: Activation functions for neural networks

**Kernels**:
- `relu_forward_kernel<T>` - ReLU forward pass
- `relu_backward_kernel<T>` - ReLU backward pass
- `sigmoid_forward_kernel<T>` - Sigmoid forward
- `sigmoid_backward_kernel<T>` - Sigmoid backward
- `tanh_forward_kernel<T>` - Tanh forward
- `tanh_backward_kernel<T>` - Tanh backward
- `leaky_relu_forward_kernel<T>` - Leaky ReLU forward
- `leaky_relu_backward_kernel<T>` - Leaky ReLU backward
- `softmax_forward_kernel<T>` - Softmax forward (uses shared memory + warp reductions)
- `softmax_backward_kernel<T>` - Softmax backward
- `log_softmax_forward_kernel<T>` - Log-softmax forward
- `log_softmax_backward_kernel<T>` - Log-softmax backward

**CUDA-Specific Features**:
- Warp shuffle intrinsics (`__shfl_down_sync`) for reductions
- Shared memory for block-level reductions
- Grid-stride loops for scalability

**Host Functions** (extern "C"):
- `relu_forward_float/double`
- `relu_backward_float/double`
- `sigmoid_forward_float/double`
- `sigmoid_backward_float/double`
- `tanh_forward_float/double`
- `tanh_backward_float/double`
- `leaky_relu_forward_float/double`
- `leaky_relu_backward_float/double`
- `softmax_forward_float/double`
- `softmax_backward_float/double`
- `log_softmax_forward_float/double`
- `log_softmax_backward_float/double`

**Porting Notes**:
- Replace `__shfl_down_sync(0xffffffff, val, offset)` with `__shfl_down(val, offset)` for HIP
- **CRITICAL**: AMD GPUs have wave size 64 (vs warp size 32), adjust loop iterations:
  ```cpp
  // CUDA (warp size 32)
  for (int offset = 16; offset > 0; offset /= 2)

  // HIP (wave size 64 on RDNA/CDNA)
  for (int offset = warpSize / 2; offset > 0; offset /= 2)
  // where warpSize = __AMDGCN_WAVEFRONT_SIZE
  ```
- Softmax kernels use `FLT_MAX` - ensure it's available or use `std::numeric_limits<T>::max()`

### 2. transform.cu (245 lines)

**Purpose**: Tensor layout transformations

**Kernels**:
- `contiguous_kernel_impl<T>` - Make tensor contiguous
- No actual kernel for `clone`, `reshape`, `transpose`, `permute`, `squeeze`, `unsqueeze` - these are metadata operations

**Host Functions**:
- `contiguous_kernel()`
- `clone_kernel()` - uses `cudaMemcpyAsync`
- `reshape_kernel()` - metadata only
- `transpose_kernel()` - metadata only
- `permute_kernel()` - metadata only
- `squeeze_kernel()` - metadata only
- `unsqueeze_kernel()` - metadata only

**Porting Notes**:
- `contiguous_kernel_impl` uses grid-stride loops - straightforward port
- Replace `cudaMemcpyAsync` with `hipMemcpyAsync`
- Replace `cudaMalloc/cudaFree` with `hipMalloc/hipFree`

### 3. batchnorm.cu (679 lines)

**Purpose**: Batch normalization for CNNs

**Kernels**:
- `batchnorm_mean_kernel<T>` - Compute per-channel mean
- `batchnorm_variance_kernel<T>` - Compute per-channel variance
- `batchnorm_normalize_kernel<T>` - Normalize: (x - mean) / sqrt(var + eps)
- `batchnorm_forward_affine_kernel<T>` - Combined normalize + affine transform
- `batchnorm_update_running_stats_kernel<T>` - Update running statistics
- `batchnorm_backward_gamma_beta_kernel<T>` - Compute gradients w.r.t. gamma/beta
- `batchnorm_backward_input_kernel<T>` - Compute gradient w.r.t. input

**CUDA-Specific Features**:
- Welford's online algorithm for stable mean/variance
- Block-level reductions with warp shuffle
- Shared memory for intermediate results
- `rsqrtf()` for efficient 1/sqrt computation

**Porting Notes**:
- Warp reduction functions need wave size adjustment
- `rsqrtf()` available in HIP - no change needed
- Shared memory usage is similar in HIP
- `__syncthreads()` - same in HIP

### 4. conv2d.cu (785 lines)

**Purpose**: 2D convolution operations

**Kernels**:
- `im2col_kernel<T>` - Image-to-column transformation
- `col2im_kernel<T>` - Column-to-image (inverse of im2col)
  - Optimized version eliminates atomics by processing output-centric
- `add_bias_kernel` - Add bias to convolution output
- `sum_bias_grad_kernel` - Compute bias gradient

**CUDA-Specific Features**:
- cuBLAS `cublasSgemm/cublasDgemm` for matrix multiplication
- Strided batched GEMM for efficiency
- Atomic operations (`atomicAdd`) - **alternative version eliminates this**

**Host Functions**:
- `conv2d_forward_kernel()` - Uses im2col + cuBLAS GEMM
- `conv2d_backward_kernel()` - Computes gradients w.r.t. input, weight, bias

**Porting Notes**:
- Replace cuBLAS with rocBLAS:
  ```cpp
  // CUDA
  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasSgemm(handle, ...);

  // HIP
  rocblas_handle handle;
  rocblas_create_handle(&handle);
  rocblas_sgemm(handle, ...);
  ```
- col2im kernel uses output-centric approach (no atomics) - direct port
- Handle cuBLAS column-major vs row-major carefully

### 5. lstm.cu (404 lines)

**Purpose**: LSTM cell operations

**Kernels**:
- `lstm_cell_forward_fused<T>` - Fused LSTM cell forward (all 4 gates)
- `lstm_cell_backward_fused<T>` - Fused LSTM cell backward

**LSTM Equations**:
```
i_t = σ(W_ii @ x_t + W_hi @ h_{t-1} + b_i)  # Input gate
f_t = σ(W_if @ x_t + W_hf @ h_{t-1} + b_f)  # Forget gate
g_t = tanh(W_ig @ x_t + W_hg @ h_{t-1} + b_g)  # Cell gate
o_t = σ(W_io @ x_t + W_ho @ h_{t-1} + b_o)  # Output gate
c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t              # Cell state
h_t = o_t ⊙ tanh(c_t)                         # Hidden state
```

**Host Functions**:
- `lstm_cell_forward_float/double`
- `lstm_cell_backward_float/double`

**Porting Notes**:
- Element-wise operations - straightforward port
- No warp intrinsics or shared memory
- Uses `exp()` and `tanh()` math functions - available in HIP

### 6. gru.cu (430 lines)

**Purpose**: GRU cell operations

**Kernels**:
- `gru_cell_forward_fused<T>` - Fused GRU cell forward (3 gates)
- `gru_cell_backward_fused<T>` - Fused GRU cell backward

**GRU Equations**:
```
r_t = σ(W_ir @ x_t + W_hr @ h_{t-1} + b_r)  # Reset gate
z_t = σ(W_iz @ x_t + W_hz @ h_{t-1} + b_z)  # Update gate
n_t = tanh(W_in @ x_t + r_t ⊙ W_hn @ h_{t-1})  # New gate
h_t = (1 - z_t) ⊙ n_t + z_t ⊙ h_{t-1}      # Hidden state
```

**Porting Notes**:
- Similar to LSTM - element-wise operations only
- Direct port with header/API changes

### 7. math.cu (1,346 lines)

**Purpose**: Element-wise mathematical operations

**Kernels**:
- **Binary operations** (with broadcasting support):
  - `add_kernel_device<T>` - Addition
  - `sub_kernel_device<T>` - Subtraction
  - `mul_kernel_device<T>` - Multiplication
  - `div_kernel_device<T>` - Division
  - `broadcast_kernel<T, Op>` - Generic broadcasting kernel
- **Unary operations**:
  - `neg_kernel_device<T>` - Negation
  - `abs_kernel_device<T>` - Absolute value
  - `sqrt_kernel_f32/f64` - Square root
  - `exp_kernel_f32/f64` - Exponential
  - `log_kernel_f32/f64` - Natural logarithm
  - `pow_kernel_f32/f64` - Power
  - `clamp_kernel_f32/f64` - Clamp
  - `sign_kernel_f32/f64` - Sign function
- **Expansion/Fill**:
  - `expand_kernel_device<T>` - Expand tensor dimensions
  - `fill_kernel_device<T>` - Fill with constant value
- **Random generation**:
  - `init_curand_states` - Initialize cuRAND states
  - `rand_kernel_device` - Uniform random [0,1)
  - `randn_kernel_device` - Normal distribution N(0,1)

**CUDA-Specific Features**:
- cuRAND for random number generation
- Broadcasting logic with stride calculations
- Grid-stride loops for scalability

**Host Functions**:
- `add_kernel()`, `sub_kernel()`, `mul_kernel()`, `div_kernel()`
- `neg_kernel()`, `abs_kernel()`, `sqrt_kernel()`, `exp_kernel()`, `log_kernel()`
- `pow_kernel()`, `clamp_kernel()`, `sign_kernel()`
- `expand_kernel()`, `fill_kernel()`, `zeros_kernel()`, `ones_kernel()`, `full_kernel()`
- `rand_kernel()`, `randn_kernel()`

**Porting Notes**:
- Replace cuRAND with rocRAND:
  ```cpp
  // CUDA
  curandState* states;
  curand_init(seed, idx, 0, &states[idx]);
  float val = curand_uniform(&states[idx]);

  // HIP
  rocrand_state_philox4x32_10* states;
  rocrand_init(seed, idx, 0, &states[idx]);
  float val = rocrand_uniform(&states[idx]);
  ```
- Broadcasting logic - direct port
- Math functions (`fabsf`, `sqrtf`, etc.) - same in HIP

### 8. matmul.cu (835 lines)

**Purpose**: Matrix multiplication (tiled implementation + cuBLAS)

**Kernels**:
- `matmul_tiled_f32_kernel<TILE_M, TILE_N, TILE_K>` - Tiled matmul (float)
- `matmul_tiled_f64_kernel<TILE_M, TILE_N, TILE_K>` - Tiled matmul (double)
- `matmul_tiled_i32_kernel<TILE_M, TILE_N, TILE_K>` - Tiled matmul (int32)
- `batched_matmul_tiled_f32_kernel<TILE_M, TILE_N, TILE_K>` - Batched matmul (float)
- `batched_matmul_tiled_f64_kernel<TILE_M, TILE_N, TILE_K>` - Batched matmul (double)

**Tile Configuration**:
```cpp
constexpr int TILE_SIZE = 32;
constexpr int TILE_SIZE_K = 16;
```

**CUDA-Specific Features**:
- Shared memory tiling for cache optimization
- cuBLAS for large matrices (M, N, K >= 512)
- Template parameters for tile sizes
- `#pragma unroll` for loop unrolling

**Host Functions**:
- `matmul_f32()`, `matmul_f64()`, `matmul_i32()`
- `batched_matmul_f32()`, `batched_matmul_f64()`
- `matmul_cublas_f32()`, `matmul_cublas_f64()`
- `batched_matmul_cublas_f32()`, `batched_matmul_cublas_f64()`

**Porting Notes**:
- Shared memory - same in HIP
- Replace cuBLAS with rocBLAS (see table above)
- `#pragma unroll` - same in HIP
- Tiled implementation - direct port
- **IMPORTANT**: Verify rocBLAS column-major vs row-major behavior

### 9. reduction.cu (911 lines)

**Purpose**: Reduction operations (sum, mean, max, min)

**Kernels**:
- `sum_reduce_kernel<T>` - Full reduction (sum)
- `max_reduce_kernel<T>` - Full reduction (max)
- `min_reduce_kernel<T>` - Full reduction (min)
- `sum_along_dim_kernel<T>` - Reduction along specific dimension
- `max_along_dim_kernel<T>` - Max along dimension
- `min_along_dim_kernel<T>` - Min along dimension

**CUDA-Specific Features**:
- Warp shuffle reductions (`warp_reduce_sum`, `warp_reduce_max`, `warp_reduce_min`)
- Two-phase reduction for large arrays
- Shared memory for block-level reductions
- Grid-stride loops

**Host Functions**:
- `sum_kernel()` - Supports full or dimensional reduction
- `mean_kernel()` - Computes mean (sum / count)
- `max_kernel()`, `min_kernel()`

**Helper Functions**:
- `launch_full_reduction_sum<T>`
- `launch_full_reduction_max<T>`
- `launch_full_reduction_min<T>`
- `launch_dim_reduction_sum<T>`
- `launch_dim_reduction_max<T>`
- `launch_dim_reduction_min<T>`

**Porting Notes**:
- **CRITICAL**: Warp/wave size difference (32 vs 64)
  ```cpp
  // CUDA
  template<typename T>
  __device__ __forceinline__ T warp_reduce_sum(T val) {
      for (int offset = 16; offset > 0; offset /= 2) {  // 32/2 = 16
          val += __shfl_down_sync(0xffffffff, val, offset);
      }
      return val;
  }

  // HIP (RDNA/CDNA with wave size 64)
  template<typename T>
  __device__ __forceinline__ T warp_reduce_sum(T val) {
      for (int offset = __AMDGCN_WAVEFRONT_SIZE / 2; offset > 0; offset /= 2) {
          val += __shfl_down(val, offset);  // No _sync suffix
      }
      return val;
  }
  ```
- Replace `FLT_MAX` with `std::numeric_limits<float>::max()` or equivalent
- Shared memory block reductions - same approach

### 10. fused_ops.cu (504 lines)

**Purpose**: Fused operations for improved performance

**Kernels**:
- `fused_linear_relu_kernel<T>` - Linear layer + ReLU
- `fused_batchnorm_relu_kernel<T>` - BatchNorm + ReLU
- `fused_softmax_cross_entropy_kernel<T, BLOCK_SIZE>` - Softmax + Cross Entropy loss
- `fused_add_relu_kernel<T>` - Element-wise add + ReLU
- `fused_gelu_kernel<T>` - GELU activation
- `fused_layer_norm_kernel<T, BLOCK_SIZE>` - Layer normalization

**CUDA-Specific Features**:
- CUB library (`<cub/cub.cuh>`) - only included, not actively used in shown code
- Shared memory reductions for softmax/layer norm
- Template parameters for block size

**Host Functions**:
- `fused_linear_relu_cuda()`
- `fused_batchnorm_relu_cuda()`
- `fused_softmax_cross_entropy_cuda()`
- `fused_add_relu_cuda()`
- `fused_gelu_cuda()`
- `fused_layer_norm_cuda()`

**Porting Notes**:
- Replace `<cub/cub.cuh>` with `<hipcub/hipcub.hpp>` if using CUB primitives
- `rsqrtf()` available in HIP
- Shared memory reductions - adjust for wave size if using warp intrinsics
- GELU uses `tanhf()` - available in HIP

---

## Backend Implementation

### File: cuda_backend.cpp (826 lines)

**Key Classes**:
- `CUDABackend` - Main backend implementation
- Inherits from `Backend` interface

**Memory Management**:
- `allocate(size_t bytes, int32_t device_id)` - Uses `cudaMalloc`
- `deallocate(void* ptr)` - Uses `cudaFree`
- Optional caching allocator (controlled by `TENZOR_ENABLE_CACHING_ALLOCATOR` env var)
- Handles zero-byte allocations (empty tensors)

**Stream Management**:
- `create_stream(int32_t device_id)` - `cudaStreamCreate`
- `destroy_stream(StreamHandle stream)` - `cudaStreamDestroy`
- `synchronize_stream(StreamHandle stream)` - `cudaStreamSynchronize`

**Device Management**:
- `device_count()` - `cudaGetDeviceCount`
- `is_available()` - Checks if CUDA devices exist
- `synchronize(int32_t device_id)` - `cudaDeviceSynchronize`

**Operation Dispatch**:
- `dispatch()` - Routes operations to appropriate CUDA kernels
- ~50+ operation types supported
- Parses attributes (dim, keepdim, alpha, epsilon, etc.)
- Validates input tensors are on CUDA device

**Porting Notes for cuda_backend.cpp**:
1. Rename file to `rocm_backend.cpp`
2. Replace all CUDA API calls:
   ```cpp
   // CUDA
   cudaMalloc(&ptr, bytes);
   cudaFree(ptr);
   cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice);
   cudaSetDevice(device_id);
   cudaDeviceSynchronize();

   // HIP
   hipMalloc(&ptr, bytes);
   hipFree(ptr);
   hipMemcpy(dst, src, bytes, hipMemcpyHostToDevice);
   hipSetDevice(device_id);
   hipDeviceSynchronize();
   ```
3. Update error handling:
   ```cpp
   // CUDA
   cudaError_t err;
   if (err != cudaSuccess) {
       cudaGetErrorString(err);
   }

   // HIP
   hipError_t err;
   if (err != hipSuccess) {
       hipGetErrorString(err);
   }
   ```
4. Update stream types:
   ```cpp
   // CUDA
   cudaStream_t stream;

   // HIP
   hipStream_t stream;
   ```
5. Update pointer query:
   ```cpp
   // CUDA
   cudaPointerAttributes attrs;
   cudaPointerGetAttributes(&attrs, ptr);

   // HIP
   hipPointerAttribute_t attrs;
   hipPointerGetAttributes(&attrs, ptr);
   ```

---

## Build System Changes

### CMakeLists.txt Modifications

Create a new CMake option for ROCm:

```cmake
# Options
option(TENZOR_ENABLE_CUDA "Enable CUDA backend" ON)
option(TENZOR_ENABLE_ROCM "Enable ROCm backend" OFF)

if(TENZOR_ENABLE_ROCM)
    # Find HIP
    find_package(HIP REQUIRED)

    # Find rocBLAS
    find_package(rocblas REQUIRED)

    # Find rocRAND
    find_package(rocrand REQUIRED)

    # Find hipCUB
    find_package(hipcub REQUIRED)

    # Set HIP flags
    set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -O3 -fPIC")
    set(CMAKE_HIP_ARCHITECTURES "gfx900;gfx906;gfx908;gfx90a;gfx940;gfx1030;gfx1100")

    # ROCm backend source files
    set(ROCM_SOURCES
        src/backends/rocm/rocm_backend.cpp
        src/backends/rocm/kernels/activations.hip
        src/backends/rocm/kernels/transform.hip
        src/backends/rocm/kernels/batchnorm.hip
        src/backends/rocm/kernels/conv2d.hip
        src/backends/rocm/kernels/lstm.hip
        src/backends/rocm/kernels/gru.hip
        src/backends/rocm/kernels/math.hip
        src/backends/rocm/kernels/matmul.hip
        src/backends/rocm/kernels/reduction.hip
        src/backends/rocm/kernels/fused_ops.hip
    )

    # Create ROCm backend library
    hip_add_library(tenzor_rocm SHARED ${ROCM_SOURCES})

    target_link_libraries(tenzor_rocm
        PUBLIC
            tenzor_core
            hip::device
            roc::rocblas
            roc::rocrand
            hip::hipcub
    )

    target_include_directories(tenzor_rocm
        PUBLIC
            ${HIP_INCLUDE_DIRS}
            ${ROCBLAS_INCLUDE_DIRS}
            ${ROCRAND_INCLUDE_DIRS}
            ${HIPCUB_INCLUDE_DIRS}
    )

    target_compile_definitions(tenzor_rocm
        PUBLIC
            TENZOR_ROCM_AVAILABLE
            __HIP_PLATFORM_AMD__
    )

    install(TARGETS tenzor_rocm
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
    )
endif()
```

### File Extensions
- Rename `.cu` files to `.hip`
- HIP compiler accepts both `.cu` and `.hip`, but `.hip` is conventional

### Compiler Selection
```bash
# Set HIP compiler
export HIP_PATH=/opt/rocm
export HIP_PLATFORM=amd
export PATH=$PATH:$HIP_PATH/bin

# Build
mkdir build && cd build
cmake -DTENZOR_ENABLE_ROCM=ON -DCMAKE_HIP_COMPILER=hipcc ..
make -j$(nproc)
```

---

## Step-by-Step Porting Process

### Phase 1: Setup and Infrastructure (Week 1)

1. **Create ROCm directory structure**:
   ```bash
   mkdir -p src/backends/rocm/kernels
   ```

2. **Copy CUDA files to ROCm directory**:
   ```bash
   cp -r src/backends/cuda/* src/backends/rocm/
   cd src/backends/rocm/kernels
   rename 's/\.cu$/.hip/' *.cu
   ```

3. **Update CMake build system**:
   - Add ROCm option (see above)
   - Add HIP language support
   - Link rocBLAS, rocRAND, hipCUB

4. **Create header conversion script** (`convert_headers.sh`):
   ```bash
   #!/bin/bash
   # Automatically convert CUDA headers to HIP

   for file in src/backends/rocm/**/*.{cpp,hip,h,hpp}; do
       sed -i 's/<cuda_runtime\.h>/<hip\/hip_runtime.h>/g' "$file"
       sed -i 's/<cuda_fp16\.h>/<hip\/hip_fp16.h>/g' "$file"
       sed -i 's/<device_launch_parameters\.h>//g' "$file"
       sed -i 's/<cublas_v2\.h>/<rocblas\/rocblas.h>/g' "$file"
       sed -i 's/<curand_kernel\.h>/<rocrand\/rocrand_kernel.h>/g' "$file"
       sed -i 's/<cub\/cub\.cuh>/<hipcub\/hipcub.hpp>/g' "$file"
   done
   ```

5. **Run hipify-perl** (automated conversion tool):
   ```bash
   hipify-perl src/backends/rocm/rocm_backend.cpp -o src/backends/rocm/rocm_backend.cpp -inplace

   for file in src/backends/rocm/kernels/*.hip; do
       hipify-perl "$file" -o "$file" -inplace
   done
   ```

### Phase 2: Core Kernel Porting (Weeks 2-3)

Port kernels in order of complexity (simplest first):

1. **math.cu → math.hip** (Simple element-wise ops)
   - Replace headers
   - Update cuRAND → rocRAND
   - Test: `add`, `sub`, `mul`, `div`, `neg`, `abs`

2. **transform.cu → transform.hip** (Metadata ops + one kernel)
   - Replace `cudaMemcpy` → `hipMemcpy`
   - Test: `contiguous`, `clone`, `reshape`

3. **lstm.cu → lstm.hip** and **gru.cu → gru.hip** (No complex features)
   - Straightforward port
   - Test: Forward and backward passes

4. **activations.cu → activations.hip** (Warp reductions)
   - **CRITICAL**: Update warp size constants
   - Replace `__shfl_down_sync` → `__shfl_down`
   - Adjust loop iterations for wave size 64
   - Test: `relu`, `sigmoid`, `tanh`, `softmax`

5. **reduction.cu → reduction.hip** (Complex warp reductions)
   - Same warp size adjustments as activations
   - Test: `sum`, `mean`, `max`, `min`

6. **batchnorm.cu → batchnorm.hip** (Shared memory + reductions)
   - Update warp reductions
   - Test: Forward and backward passes

7. **conv2d.cu → conv2d.hip** (cuBLAS dependency)
   - Replace cuBLAS with rocBLAS (see mapping table)
   - Update column-major handling if needed
   - Test: Forward and backward convolution

8. **matmul.cu → matmul.hip** (Tiled SHMEM + cuBLAS)
   - Replace cuBLAS with rocBLAS
   - Verify shared memory tiling works
   - Test: 2D and 3D (batched) matrix multiplication

9. **fused_ops.cu → fused_ops.hip** (Optional optimizations)
   - Replace hipCUB headers
   - Test: Fused operations

### Phase 3: Backend Integration (Week 4)

1. **Port rocm_backend.cpp**:
   - Replace all CUDA API calls with HIP equivalents
   - Update error handling
   - Update stream management
   - Test: Basic tensor creation and operations

2. **Update device management**:
   - `hipGetDeviceCount()`
   - `hipSetDevice()`
   - `hipDeviceSynchronize()`

3. **Test caching allocator** (if used):
   - Ensure `hipPointerGetAttributes` works correctly

### Phase 4: Testing and Validation (Week 5)

1. **Unit tests** (run existing CUDA tests on ROCm):
   ```bash
   # Build tests
   cmake -DTENZOR_ENABLE_ROCM=ON -DTENZOR_BUILD_TESTS=ON ..
   make -j$(nproc)

   # Run tests
   ctest --output-on-failure
   ```

2. **Numerical accuracy verification**:
   - Compare ROCm results with CUDA results
   - Tolerance: `abs(rocm_result - cuda_result) < 1e-5` for FP32
   - Use double precision for reference if needed

3. **Performance profiling**:
   ```bash
   # Use rocprof for profiling
   rocprof --stats ./test_rocm_backend

   # Use roctracer for detailed tracing
   roctracer --hsa-trace ./test_rocm_backend
   ```

4. **Benchmark critical operations**:
   - Matrix multiplication (compare with rocBLAS benchmarks)
   - Convolution (should match cudnn performance roughly)
   - Reductions
   - Activation functions

### Phase 5: Optimization and Tuning (Week 6)

1. **Wave size optimization**:
   - Benchmark kernels with wave size 32 vs 64
   - Some algorithms may perform better with wavefront size 32 on RDNA

2. **Shared memory optimization**:
   - AMD GPUs have different cache hierarchies
   - May need to adjust tile sizes for matmul

3. **Occupancy tuning**:
   - Use `rocprof --occupancy` to check kernel occupancy
   - Adjust block sizes if needed

4. **Memory coalescing**:
   - Use `rocprof --mem-trace` to check memory access patterns
   - Optimize stride patterns if needed

---

## Testing Strategy

### Unit Test Coverage

Create test files for each kernel module:

```cpp
// tests/test_rocm_activations.cpp
#include <gtest/gtest.h>
#include "tenzor/core/tensor.hpp"

TEST(ROCmActivations, ReLUForward) {
    auto device = Device::rocm(0);
    auto input = rand({10, 20}, DType::Float32, device);
    auto output = relu(input);

    // Verify on CPU
    auto cpu_input = input.to(Device::cpu());
    auto cpu_output = relu(cpu_input);
    auto rocm_output = output.to(Device::cpu());

    EXPECT_TRUE(allclose(cpu_output, rocm_output, 1e-5, 1e-5));
}

TEST(ROCmActivations, SoftmaxForward) {
    auto device = Device::rocm(0);
    auto input = randn({64, 1000}, DType::Float32, device);
    auto output = softmax(input, /*dim=*/1);

    // Check sum along dim=1 equals 1.0
    auto sum = output.sum(/*dim=*/1);
    auto expected = ones({64, 1}, DType::Float32, device);

    EXPECT_TRUE(allclose(sum, expected, 1e-5, 1e-5));
}
```

### Integration Tests

Test complete neural network layers:

```cpp
TEST(ROCmIntegration, Conv2DLayer) {
    auto device = Device::rocm(0);

    // Create Conv2D layer
    int64_t batch = 4, in_channels = 3, out_channels = 64;
    int64_t H = 224, W = 224, kernel_size = 3;

    auto input = randn({batch, in_channels, H, W}, DType::Float32, device);
    auto weight = randn({out_channels, in_channels, kernel_size, kernel_size}, DType::Float32, device);
    auto bias = randn({out_channels}, DType::Float32, device);

    auto output = conv2d(input, weight, bias, /*stride=*/1, /*padding=*/1);

    // Verify output shape
    EXPECT_EQ(output.shape()[0], batch);
    EXPECT_EQ(output.shape()[1], out_channels);
    EXPECT_EQ(output.shape()[2], H);
    EXPECT_EQ(output.shape()[3], W);

    // Verify against CPU
    auto cpu_output = conv2d(input.to(Device::cpu()),
                              weight.to(Device::cpu()),
                              bias.to(Device::cpu()), 1, 1);
    EXPECT_TRUE(allclose(output.to(Device::cpu()), cpu_output, 1e-4, 1e-4));
}
```

### Performance Benchmarks

```cpp
// benchmarks/bench_rocm_matmul.cpp
#include <benchmark/benchmark.h>
#include "tenzor/core/tensor.hpp"

static void BM_ROCmMatMul_512x512(benchmark::State& state) {
    auto device = Device::rocm(0);
    auto A = randn({512, 512}, DType::Float32, device);
    auto B = randn({512, 512}, DType::Float32, device);

    // Warm-up
    auto C = matmul(A, B);
    device.synchronize();

    for (auto _ : state) {
        C = matmul(A, B);
        device.synchronize();
    }

    // Report GFLOPS
    int64_t flops = 2LL * 512 * 512 * 512;
    state.counters["GFLOPS"] = benchmark::Counter(
        flops, benchmark::Counter::kIsIterationInvariantRate,
        benchmark::Counter::OneK::kIs1000
    );
}
BENCHMARK(BM_ROCmMatMul_512x512);

static void BM_ROCmMatMul_4096x4096(benchmark::State& state) {
    auto device = Device::rocm(0);
    auto A = randn({4096, 4096}, DType::Float32, device);
    auto B = randn({4096, 4096}, DType::Float32, device);

    auto C = matmul(A, B);
    device.synchronize();

    for (auto _ : state) {
        C = matmul(A, B);
        device.synchronize();
    }

    int64_t flops = 2LL * 4096 * 4096 * 4096;
    state.counters["GFLOPS"] = benchmark::Counter(
        flops, benchmark::Counter::kIsIterationInvariantRate,
        benchmark::Counter::OneK::kIs1000
    );
}
BENCHMARK(BM_ROCmMatMul_4096x4096);

BENCHMARK_MAIN();
```

### Continuous Integration

Add ROCm testing to CI pipeline:

```yaml
# .github/workflows/rocm-ci.yml
name: ROCm CI

on: [push, pull_request]

jobs:
  rocm-build-test:
    runs-on: [self-hosted, rocm]

    steps:
      - uses: actions/checkout@v3

      - name: Setup ROCm Environment
        run: |
          export ROCM_PATH=/opt/rocm
          export PATH=$PATH:$ROCM_PATH/bin

      - name: Build ROCm Backend
        run: |
          mkdir build && cd build
          cmake -DTENZOR_ENABLE_ROCM=ON -DTENZOR_BUILD_TESTS=ON ..
          make -j$(nproc)

      - name: Run Unit Tests
        run: |
          cd build
          ctest --output-on-failure --timeout 300

      - name: Run Benchmarks
        run: |
          cd build
          ./benchmarks/bench_rocm_matmul --benchmark_format=json > bench_results.json

      - name: Upload Results
        uses: actions/upload-artifact@v3
        with:
          name: rocm-test-results
          path: build/bench_results.json
```

---

## Critical Porting Issues and Solutions

### Issue 1: Warp/Wave Size Difference

**Problem**: NVIDIA GPUs have warp size 32, AMD GPUs have wave size 64 (RDNA/CDNA) or 32 (older architectures).

**Solution**:
```cpp
// Define wave size portably
#if defined(__HIP_PLATFORM_AMD__)
    #define WAVE_SIZE __AMDGCN_WAVEFRONT_SIZE  // 64 or 32
#else
    #define WAVE_SIZE 32  // NVIDIA
#endif

// Use in warp reduction
template<typename T>
__device__ __forceinline__ T warp_reduce_sum(T val) {
    #pragma unroll
    for (int offset = WAVE_SIZE / 2; offset > 0; offset /= 2) {
        #if defined(__HIP_PLATFORM_AMD__)
            val += __shfl_down(val, offset);
        #else
            val += __shfl_down_sync(0xffffffff, val, offset);
        #endif
    }
    return val;
}
```

### Issue 2: cuBLAS vs rocBLAS Differences

**Problem**: Column-major vs row-major, different operation enums

**Solution**:
```cpp
// Wrapper function for GEMM
template<typename T>
void gemm_wrapper(
    bool transA, bool transB,
    int64_t M, int64_t N, int64_t K,
    T alpha,
    const T* A, int64_t lda,
    const T* B, int64_t ldb,
    T beta,
    T* C, int64_t ldc
) {
    #if defined(TENZOR_ROCM_AVAILABLE)
        rocblas_handle handle;
        rocblas_create_handle(&handle);

        rocblas_operation opA = transA ? rocblas_operation_transpose : rocblas_operation_none;
        rocblas_operation opB = transB ? rocblas_operation_transpose : rocblas_operation_none;

        if constexpr (std::is_same_v<T, float>) {
            rocblas_sgemm(handle, opB, opA, N, M, K, &alpha, B, ldb, A, lda, &beta, C, ldc);
        } else if constexpr (std::is_same_v<T, double>) {
            rocblas_dgemm(handle, opB, opA, N, M, K, &alpha, B, ldb, A, lda, &beta, C, ldc);
        }

        rocblas_destroy_handle(handle);
    #else
        // CUDA cuBLAS code
    #endif
}
```

### Issue 3: cuRAND vs rocRAND API Differences

**Problem**: Different state types and initialization

**Solution**:
```cpp
#if defined(TENZOR_ROCM_AVAILABLE)
    using RandState = rocrand_state_philox4x32_10;
    #define RAND_INIT rocrand_init
    #define RAND_UNIFORM rocrand_uniform
    #define RAND_NORMAL rocrand_normal
#else
    using RandState = curandState;
    #define RAND_INIT curand_init
    #define RAND_UNIFORM curand_uniform
    #define RAND_NORMAL curand_normal
#endif

__global__ void init_rand_states(RandState* states, uint64_t seed, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        RAND_INIT(seed, idx, 0, &states[idx]);
    }
}

__global__ void rand_uniform_kernel(float* output, RandState* states, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = RAND_UNIFORM(&states[idx]);
    }
}
```

### Issue 4: Shared Memory Bank Conflicts

**Problem**: AMD GPUs have different shared memory bank configuration

**Solution**: Pad shared memory arrays to avoid bank conflicts
```cpp
// CUDA: 32 banks, 4-byte wide
__shared__ float shmem[TILE_SIZE][TILE_SIZE];

// HIP: May need padding for AMD GPUs
__shared__ float shmem[TILE_SIZE][TILE_SIZE + 1];  // +1 padding
```

### Issue 5: FP16 Support

**Problem**: Different FP16 implementations

**Solution**:
```cpp
#if defined(TENZOR_ROCM_AVAILABLE)
    #include <hip/hip_fp16.h>
    using half = __half;
#else
    #include <cuda_fp16.h>
    using half = __half;
#endif

// Use same API for both
__device__ half half_add(half a, half b) {
    return __hadd(a, b);
}
```

---

## Expected Performance

### Target Metrics (compared to CUDA)

| Operation | Expected Performance | Notes |
|-----------|----------------------|-------|
| MatMul (large) | 90-100% | rocBLAS is highly optimized |
| MatMul (small) | 85-95% | Custom tiled kernels |
| Convolution | 85-95% | With rocBLAS GEMM backend |
| Reductions | 80-90% | Wave size 64 can help |
| Element-wise | 95-100% | Memory-bound, similar BW |
| Activations | 90-100% | Compute-bound, similar |

### Known Bottlenecks

1. **Warp divergence**: Wave size 64 can increase divergence in conditional code
2. **Shared memory**: Different L1 cache behavior may affect shared memory kernels
3. **Atomic operations**: Similar performance, but check for regression
4. **Random number generation**: rocRAND may be slightly slower than cuRAND

---

## Summary Checklist

### Pre-Porting
- [ ] Install ROCm 5.4+ with rocBLAS, rocRAND, hipCUB
- [ ] Set up CMake build system for HIP
- [ ] Understand warp/wave size differences
- [ ] Review cuBLAS → rocBLAS API mapping

### During Porting
- [ ] Run hipify-perl on all source files
- [ ] Manually review and fix generated code
- [ ] Update all header includes
- [ ] Replace CUDA API calls with HIP equivalents
- [ ] Adjust warp reduction code for wave size 64
- [ ] Update cuBLAS calls to rocBLAS
- [ ] Update cuRAND calls to rocRAND
- [ ] Test each kernel module independently

### Post-Porting
- [ ] Run complete unit test suite
- [ ] Compare numerical accuracy with CUDA
- [ ] Profile and benchmark critical operations
- [ ] Optimize for AMD GPU architecture
- [ ] Document any remaining issues or limitations
- [ ] Set up continuous integration for ROCm
- [ ] Update documentation and user guides

---

## Conclusion

This guide provides a comprehensive roadmap for porting the Tenzor CUDA backend to ROCm/HIP. The estimated timeline is **6 weeks** for a complete port with testing and optimization. The main challenges are:

1. Warp/wave size differences (32 vs 64)
2. cuBLAS → rocBLAS conversion
3. Numerical accuracy verification
4. Performance optimization for AMD architecture

With careful attention to these details and systematic testing, the ROCm backend should achieve 85-95% of CUDA performance for most operations.

---

## Appendix: Quick Reference Commands

### Build Commands
```bash
# Configure
cmake -DTENZOR_ENABLE_ROCM=ON -DCMAKE_HIP_COMPILER=hipcc ..

# Build
make -j$(nproc)

# Test
ctest --output-on-failure

# Benchmark
./benchmarks/bench_rocm_matmul
```

### Profiling Commands
```bash
# Basic statistics
rocprof --stats ./app

# Memory trace
rocprof --hsa-trace --hsa-trace-opts mem ./app

# Detailed profiling
rocprof --timestamp on --hsa-trace ./app

# Occupancy
rocprof --stats --occupancy ./app
```

### Debugging Commands
```bash
# Enable HIP debugging
export HIP_VISIBLE_DEVICES=0
export HIP_LAUNCH_BLOCKING=1
export AMD_LOG_LEVEL=3

# Use rocgdb
rocgdb --args ./app

# Check for errors
rocm-smi
rocminfo
```

---

**Document Version**: 1.0
**Last Updated**: 2025-10-14
**Author**: Claude (Anthropic AI)
**Review Status**: Comprehensive - Ready for Implementation
