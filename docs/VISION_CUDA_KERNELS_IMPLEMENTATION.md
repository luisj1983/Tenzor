# Vision CUDA Kernels Implementation

## Overview

This document describes the complete CUDA kernel implementation for vision operations in `/home/lee/Projects/Tenzor/src/ops/vision.cpp`.

## Implemented CUDA Kernels

### 1. Unfold Operation (im2col)
**File**: `/home/lee/Projects/Tenzor/src/backends/cuda/kernels/vision.cu`

**Function**: `unfold_cuda()`

**Description**: Extracts sliding local blocks from batched input tensor, converting 4D input (N, C, H, W) to 3D output (N, C*K*K, L) where L = number of blocks.

**Features**:
- Parallel processing with CUDA grid-stride loop
- Support for stride, padding, and dilation
- Template-based for float and double precision
- Automatic device dispatch in vision.cpp

**Kernel Implementation**:
```cpp
template<typename T>
__global__ void unfold_kernel(
    const T* input,
    T* output,
    int64_t batch,
    int64_t channels,
    int64_t height,
    int64_t width,
    int64_t kernel_size,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    int64_t out_h,
    int64_t out_w
)
```

### 2. Fold Operation (col2im)
**File**: `/home/lee/Projects/Tenzor/src/backends/cuda/kernels/vision.cu`

**Function**: `fold_cuda()`

**Description**: Reverse operation of unfold. Accumulates overlapping blocks back into spatial dimensions. Uses atomic operations for accumulation.

**Features**:
- Atomic accumulation for overlapping regions
- Template-based for float and double precision
- Handles overlapping patches correctly
- Zero-initialization of output buffer

**Kernel Implementation**:
```cpp
template<typename T>
__global__ void fold_kernel(
    const T* input,
    T* output,
    int64_t batch,
    int64_t channels,
    int64_t height,
    int64_t width,
    int64_t kernel_size,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    int64_t out_h,
    int64_t out_w
)
```

### 3. Interpolation Operations
**File**: `/home/lee/Projects/Tenzor/src/backends/cuda/kernels/vision.cu`

**Function**: `interpolate_cuda()`

#### 3.1 Nearest Neighbor Interpolation
**Kernel**: `interpolate_nearest_kernel()`

**Description**: Fast, simple upsampling by replicating nearest pixel values.

**Features**:
- O(1) per-pixel computation
- No floating-point interpolation needed
- Fastest interpolation mode

#### 3.2 Bilinear Interpolation
**Kernel**: `interpolate_bilinear_kernel()`

**Description**: Smooth upsampling using linear interpolation in both dimensions.

**Features**:
- Support for `align_corners` mode
- 4-neighbor interpolation
- Smooth output suitable for feature maps

#### 3.3 Bicubic Interpolation
**Kernel**: `interpolate_bicubic_kernel()`

**Description**: High-quality upsampling using cubic interpolation in both dimensions.

**Features**:
- 4x4 neighborhood (16 pixels)
- Cubic kernel function for smoother results
- Higher quality than bilinear (slower)

**Kernel Implementations**:
```cpp
template<typename T>
__global__ void interpolate_nearest_kernel(...);

template<typename T>
__global__ void interpolate_bilinear_kernel(...);

template<typename T>
__global__ void interpolate_bicubic_kernel(...);
```

## Architecture

### File Structure
```
src/
├── ops/
│   └── vision.cpp              # CPU implementation + CUDA dispatch
└── backends/
    └── cuda/
        ├── CMakeLists.txt      # Build configuration
        └── kernels/
            └── vision.cu       # CUDA kernel implementations
```

### Dispatch Pattern

The `vision.cpp` file uses a dispatch pattern:

```cpp
// Forward declarations for CUDA kernels
namespace tenzor::cuda {
    auto unfold_cuda(...) -> Tensor;
    auto fold_cuda(...) -> Tensor;
    auto interpolate_cuda(...) -> Tensor;
}

// In each operation function:
if (input.device().type == Device::Type::CUDA) {
    return cuda::unfold_cuda(...);  // Use CUDA kernel
}
// Otherwise, use CPU implementation
```

## Performance Characteristics

### Unfold/Fold
- **Parallelization**: One thread per output element
- **Memory Access Pattern**: Coalesced reads, atomic writes (fold only)
- **Complexity**: O(batch * channels * kernel_size^2 * num_blocks)

### Interpolation
- **Nearest**: O(batch * channels * out_h * out_w)
- **Bilinear**: 4x work per pixel vs nearest
- **Bicubic**: 16x work per pixel vs nearest (4x4 neighborhood)

## Testing

Test file: `/home/lee/Projects/Tenzor/tests/test_vision_cuda_kernels.cpp`

Tests include:
- Shape verification for all operations
- CPU vs CUDA output consistency
- Round-trip testing (unfold → fold)
- Different interpolation modes
- Edge cases (padding, dilation, alignment)

## Build Integration

The CUDA kernels are integrated into the build system via:

**File**: `/home/lee/Projects/Tenzor/src/backends/cuda/CMakeLists.txt`

```cmake
set(CUDA_BACKEND_SOURCES
    ...
    kernels/vision.cu
    ...
)
```

## Usage Examples

### Unfold
```cpp
// Extract 16x16 patches with stride 16 (non-overlapping)
Tensor img({1, 3, 224, 224}, DType::Float32, Device::cuda(0));
Tensor patches = unfold(img, 16, 16, 0, 1);
// Shape: (1, 768, 196) where 768 = 3*16*16, 196 = (224/16)^2
```

### Fold
```cpp
// Reconstruct image from patches
Tensor patches({1, 768, 196}, DType::Float32, Device::cuda(0));
Tensor img = fold(patches, {224, 224}, 16, 16, 0, 1);
// Shape: (1, 3, 224, 224)
```

### Interpolation
```cpp
// Upsample feature map 2x using bilinear
Tensor feat({1, 256, 32, 32}, DType::Float32, Device::cuda(0));
Tensor upsampled = interpolate(feat, {64, 64}, "bilinear");
// Shape: (1, 256, 64, 64)
```

## Supported Data Types

All kernels support:
- `DType::Float32` (float)
- `DType::Float64` (double)

Templates allow easy extension to other types if needed.

## Error Handling

- Input validation in host functions
- CUDA error checking with `CUDA_CHECK` macro
- Device synchronization after kernel launches
- Runtime exceptions for unsupported modes/dtypes

## Future Optimizations

Potential improvements:
1. **Shared Memory**: Use shared memory for fold operation to reduce atomic contention
2. **Half Precision**: Add FP16/BF16 support for faster inference
3. **Texture Memory**: Use texture cache for interpolation operations
4. **Stream Support**: Add stream parameters for async execution
5. **Multi-GPU**: Support multi-GPU batch processing

## Verification

All TODO comments have been removed from:
- ✅ `/home/lee/Projects/Tenzor/src/ops/vision.cpp`
- ✅ `/home/lee/Projects/Tenzor/src/backends/cuda/kernels/vision.cu`

Build verification:
```bash
cmake --build build --target tenzor_backend_cuda
# ✅ Successfully compiled with no errors
```

## References

- PyTorch `torch.nn.functional.unfold/fold`
- PyTorch `torch.nn.functional.interpolate`
- CUDA Programming Guide: Grid-Stride Loops
- CUDA Best Practices: Atomic Operations
