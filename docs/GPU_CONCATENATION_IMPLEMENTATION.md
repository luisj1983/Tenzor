# GPU Concatenation Implementation

## Overview

This document describes the implementation of GPU-accelerated tensor concatenation for the Tenzor framework. The implementation provides CUDA kernel support for the `cat()` operation, enabling efficient multi-tensor concatenation on GPU devices.

## Implementation Details

### 1. CUDA Kernel (`src/backends/cuda/kernels/transform.cu`)

Added `cat_kernel()` function and `cat_kernel_impl()` device kernel:

**Key Features:**
- Supports concatenation along any dimension
- Handles multiple tensors (2 or more) in a single operation
- Works with Float32, Float64, Int32, and Int64 data types
- Automatically makes input tensors contiguous for optimal performance
- Uses grid-stride loops for efficient GPU parallelization

**Algorithm:**
1. Validate inputs and normalize dimension index
2. Make all input tensors contiguous
3. Create output tensor with correct shape
4. Allocate device memory for metadata (shapes, strides, pointers)
5. Launch kernel where each thread:
   - Computes its output position
   - Determines which input tensor to read from
   - Computes the corresponding input position
   - Copies the value to output

**Memory Management:**
- Temporary device memory for pointers, shapes, and strides
- Proper cleanup on error or completion
- Zero-copy for already contiguous tensors when possible

### 2. Backend Integration (`src/backends/cuda/cuda_backend.cpp`)

Added dispatcher support for the "cat" operation:

```cpp
else if (op_name == "cat") {
    if (inputs.empty()) {
        throw std::invalid_argument("cat operation requires at least 1 input tensor");
    }
    int64_t dim = 0;
    if (attrs.contains("dim")) {
        dim = std::stoll(attrs.at("dim"));
    }
    return {cuda::cat_kernel(inputs, dim, stream)};
}
```

Also added forward declaration:
```cpp
auto cat_kernel(std::span<const Tensor> tensors, int64_t dim, cudaStream_t stream) -> Tensor;
```

### 3. High-Level API Update (`src/ops/transform.cpp`)

Modified `cat()` function to dispatch to CUDA backend for GPU tensors:

```cpp
// For non-CPU devices, use dispatcher to route to backend-specific implementation
if (tensors[0].device().type != Device::Type::CPU) {
    OpAttributes attrs;
    attrs["dim"] = std::to_string(dim);

    // Convert span to vector for dispatch
    std::vector<Tensor> tensor_vec(tensors.begin(), tensors.end());
    return Dispatcher::dispatch("cat", std::span<const Tensor>(tensor_vec), attrs)[0];
}
```

Replaced the error message with proper dispatching logic.

### 4. Test Suite (`tests/test_cuda_cat.cpp`)

Created comprehensive test suite covering:

1. **1D Concatenation**: Basic concatenation of 1D tensors
2. **2D Concatenation (dim=1)**: Column-wise concatenation
3. **2D Concatenation (dim=0)**: Row-wise concatenation
4. **3D Concatenation**: Concatenation along depth dimension
5. **Many Tensors**: Concatenating 10+ tensors in one operation

All tests:
- Create tensors on CUDA device
- Perform concatenation
- Copy results to CPU
- Verify shape and values

## Performance Characteristics

### Time Complexity
- O(N) where N is the total number of elements in the output tensor
- Each thread processes one output element independently
- Grid-stride loop allows handling arbitrarily large tensors

### Space Complexity
- O(M × D) temporary device memory for metadata
  - M = number of input tensors
  - D = number of dimensions
- Output tensor: O(N) where N is total elements

### Optimization Opportunities

1. **Coalesced Memory Access**: The current implementation computes source indices on-the-fly. For better performance, could use strided copy kernels for contiguous sections.

2. **Multiple Kernels per Tensor**: For very large concatenations, could launch separate kernel for each input tensor to reduce branching.

3. **Shared Memory**: For small tensors, metadata could be loaded into shared memory to reduce global memory accesses.

4. **cuBLAS Integration**: For specific cases (e.g., 2D matrix concatenation), could leverage optimized BLAS routines.

## Usage Example

```cpp
#include "tenzor/core/tensor.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/transform.hpp"

using namespace tenzor;

// Create tensors on CUDA
auto a = full({2, 3}, 1.0f, DType::Float32, Device::cuda(0));
auto b = full({2, 2}, 2.0f, DType::Float32, Device::cuda(0));

// Concatenate along dimension 1 (columns)
std::vector<Tensor> tensors = {a, b};
auto result = cat(std::span<const Tensor>(tensors), 1);

// Result shape: [2, 5]
// Result values:
// [[1, 1, 1, 2, 2],
//  [1, 1, 1, 2, 2]]
```

## Compatibility

### Supported Data Types
- Float32 (float)
- Float64 (double)
- Int32 (int32_t)
- Int64 (int64_t)

### Device Requirements
- CUDA-capable GPU
- CUDA Toolkit 11.0 or later (tested with 13.0)
- Compute Capability 3.5+

### Tensor Requirements
- All input tensors must be on the same CUDA device
- All tensors must have the same dtype
- All tensors must have same shape except at concatenation dimension
- All tensors must have same number of dimensions

## Error Handling

The implementation handles several error cases:

1. **Empty tensor list**: Throws `std::invalid_argument`
2. **Dimension mismatch**: Throws `std::invalid_argument` with descriptive message
3. **Shape incompatibility**: Validates all dimensions except concat dimension
4. **Unsupported dtype**: Throws `std::runtime_error` with supported types
5. **CUDA errors**: Properly frees device memory and throws with CUDA error message

## Testing

To build and run tests:

```bash
cd build
cmake .. -DTENZOR_BUILD_CUDA=ON
make test_cuda_cat
./tests/test_cuda_cat
```

Expected output:
```
Running CUDA concatenation tests...
Testing 1D CUDA concatenation...
✓ 1D concatenation passed
Testing 2D CUDA concatenation...
✓ 2D concatenation passed
Testing 2D CUDA concatenation along dim 0...
✓ 2D concatenation along dim 0 passed
Testing 3D CUDA concatenation...
✓ 3D concatenation passed
Testing concatenation of many tensors...
✓ Many tensors concatenation passed

All CUDA concatenation tests passed!
```

## Integration with Neural Networks

This implementation is critical for:

1. **Multi-GPU Training**: Concatenating gradients from multiple GPUs
2. **Skip Connections**: Concatenating feature maps in ResNet-style architectures
3. **Encoder-Decoder Models**: Concatenating encoder outputs with decoder hidden states
4. **Attention Mechanisms**: Concatenating multi-head attention outputs
5. **Data Augmentation**: Batch concatenation during preprocessing

## Files Modified

1. `/home/lee/Projects/Tenzor/src/backends/cuda/kernels/transform.cu`
   - Added `cat_kernel()` and `cat_kernel_impl()`

2. `/home/lee/Projects/Tenzor/src/backends/cuda/cuda_backend.cpp`
   - Added forward declaration for `cat_kernel()`
   - Added dispatch case for "cat" operation

3. `/home/lee/Projects/Tenzor/src/ops/transform.cpp`
   - Updated `cat()` to dispatch to CUDA backend for GPU tensors
   - Removed error message for non-CPU devices

4. `/home/lee/Projects/Tenzor/tests/test_cuda_cat.cpp`
   - Created comprehensive test suite (NEW FILE)

5. `/home/lee/Projects/Tenzor/tests/CMakeLists.txt`
   - Added test_cuda_cat target

## Future Enhancements

1. **ROCm Support**: Port kernel to HIP for AMD GPUs
2. **Mixed Precision**: Support Float16/BFloat16 data types
3. **Async Execution**: Better stream support for concurrent operations
4. **Memory Pooling**: Integrate with caching allocator for temporary buffers
5. **Autograd Integration**: Implement backward pass for gradient computation
6. **Benchmark Suite**: Add performance benchmarks vs. PyTorch/TensorFlow

## Conclusion

The GPU concatenation implementation provides efficient, production-ready tensor concatenation for CUDA devices. It integrates seamlessly with the existing Tenzor framework architecture and provides a solid foundation for complex neural network operations.
