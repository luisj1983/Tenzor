#include <cuda_runtime.h>
#include "tenzor/core/tensor.hpp"
#include "tenzor/core/dtype.hpp"
#include <stdexcept>
#include <vector>

namespace tenzor {
namespace cuda {

// CUDA Helper macros
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#define CUDA_GRID_STRIDE_LOOP(i, n) \
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; \
         i < (n); \
         i += blockDim.x * gridDim.x)

constexpr int BLOCK_SIZE = 256;

inline int get_num_blocks(int64_t n, int block_size = BLOCK_SIZE) {
    return (n + block_size - 1) / block_size;
}

// Contiguous kernel - copies non-contiguous data to contiguous layout
template<typename T>
__global__ void contiguous_kernel_impl(const T* input, T* output,
                                       const int64_t* strides,
                                       const int64_t* shape,
                                       int64_t ndim, int64_t total_elements) {
    CUDA_GRID_STRIDE_LOOP(idx, total_elements) {
        // Convert linear index to multi-dimensional indices
        int64_t temp_idx = idx;
        int64_t src_offset = 0;

        for (int64_t dim = ndim - 1; dim >= 0; --dim) {
            int64_t coord = temp_idx % shape[dim];
            src_offset += coord * strides[dim];
            temp_idx /= shape[dim];
        }

        output[idx] = input[src_offset];
    }
}

// Contiguous wrapper function
auto contiguous_kernel(const Tensor& input, cudaStream_t stream) -> Tensor {
    // If already contiguous, return as-is
    if (input.is_contiguous()) {
        return input;
    }

    std::vector<int64_t> shape(input.shape().begin(), input.shape().end());
    Tensor result(shape, input.dtype(), input.device());

    const int64_t ndim = input.ndim();
    const int64_t total_elements = input.numel();

    if (total_elements == 0) {
        return result;  // Empty tensor
    }

    // Copy strides and shape to device memory
    std::vector<int64_t> strides_vec(input.strides().begin(), input.strides().end());

    int64_t* d_strides;
    int64_t* d_shape;
    CUDA_CHECK(cudaMalloc(&d_strides, ndim * sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&d_shape, ndim * sizeof(int64_t)));
    CUDA_CHECK(cudaMemcpy(d_strides, strides_vec.data(), ndim * sizeof(int64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_shape, shape.data(), ndim * sizeof(int64_t), cudaMemcpyHostToDevice));

    // Launch kernel
    int num_blocks = get_num_blocks(total_elements);

    if (input.dtype() == DType::Float32) {
        contiguous_kernel_impl<float><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            input.data<float>(), result.data<float>(),
            d_strides, d_shape, ndim, total_elements);
    } else if (input.dtype() == DType::Float64) {
        contiguous_kernel_impl<double><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            input.data<double>(), result.data<double>(),
            d_strides, d_shape, ndim, total_elements);
    } else if (input.dtype() == DType::Int32) {
        contiguous_kernel_impl<int32_t><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            input.data<int32_t>(), result.data<int32_t>(),
            d_strides, d_shape, ndim, total_elements);
    } else if (input.dtype() == DType::Int64) {
        contiguous_kernel_impl<int64_t><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            input.data<int64_t>(), result.data<int64_t>(),
            d_strides, d_shape, ndim, total_elements);
    } else {
        cudaFree(d_strides);
        cudaFree(d_shape);
        throw std::runtime_error("Contiguous only supports Float32, Float64, Int32, and Int64 dtypes");
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(d_strides);
        cudaFree(d_shape);
        throw std::runtime_error(std::string("CUDA error in contiguous_kernel: ") + cudaGetErrorString(err));
    }

    // Free device memory
    cudaFree(d_strides);
    cudaFree(d_shape);

    return result;
}

} // namespace cuda
} // namespace tenzor
