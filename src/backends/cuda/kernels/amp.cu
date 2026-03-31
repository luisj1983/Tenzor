/**
 * @file amp.cu
 * @brief CUDA kernels for automatic mixed precision support
 */

#include "tenzor/core/tensor.hpp"
#include "cuda_common.cuh"
#include "cuda_launch_utils.cuh"
#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace tenzor {
namespace cuda {

namespace {

// Helper to create a Bool scalar tensor on the target device
inline Tensor make_bool_scalar(bool value, Device device) {
    Tensor result({}, DType::Bool, Device::cpu());
    result.data<bool>()[0] = value;
    return result.to(device);
}

} // anonymous namespace

template<typename T>
__global__ void check_inf_nan_kernel(const T* data, int64_t n, int* result) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        T val = data[idx];
        if (isinf(static_cast<float>(val)) || isnan(static_cast<float>(val))) {
            atomicExch(result, 1);
        }
    }
}

// Float64 specialization — use double-precision isinf/isnan directly
template<>
__global__ void check_inf_nan_kernel<double>(const double* data, int64_t n, int* result) {
    TENZOR_CUDA_KERNEL_LOOP(idx, n) {
        double val = data[idx];
        if (isinf(val) || isnan(val)) {
            atomicExch(result, 1);
        }
    }
}

auto has_inf_nan_kernel(const Tensor& input, int64_t /*dim*/, bool /*keepdim*/, cudaStream_t stream) -> Tensor {
    const int64_t numel = input.numel();
    if (numel == 0) {
        return make_bool_scalar(false, input.device());
    }

    // Allocate device flag (single int) via RAII
    CudaBuffer flag_buf(sizeof(int));
    int* d_flag = static_cast<int*>(flag_buf.ptr);
    TENZOR_CUDA_CHECK(cudaMemsetAsync(d_flag, 0, sizeof(int), stream));

    // Handle BFloat16/Float16 by casting to Float32 first
    Tensor scan = input;
    if (scan.dtype() == DType::BFloat16 || scan.dtype() == DType::Float16) {
        scan = scan.to(DType::Float32);
    }

    switch (scan.dtype()) {
        case DType::Float32: {
            auto [grid, block] = optimal_launch_config(
                check_inf_nan_kernel<float>, numel);
            check_inf_nan_kernel<float><<<grid, block, 0, stream>>>(
                scan.data<float>(), numel, d_flag);
            break;
        }
        case DType::Float64: {
            auto [grid, block] = optimal_launch_config(
                check_inf_nan_kernel<double>, numel);
            check_inf_nan_kernel<double><<<grid, block, 0, stream>>>(
                scan.data<double>(), numel, d_flag);
            break;
        }
        default:
            // Integer types can't have inf/nan
            return make_bool_scalar(false, input.device());
    }
    TENZOR_CUDA_CHECK(cudaGetLastError());

    // Copy result back
    int h_flag = 0;
    TENZOR_CUDA_CHECK(cudaMemcpyAsync(&h_flag, d_flag, sizeof(int),
                                        cudaMemcpyDeviceToHost, stream));
    TENZOR_CUDA_CHECK(cudaStreamSynchronize(stream));

    return make_bool_scalar(h_flag != 0, input.device());
}

} // namespace cuda
} // namespace tenzor
