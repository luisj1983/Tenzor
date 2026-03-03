#pragma once

/**
 * @file cublas_handle_pool.hpp
 * @brief Centralized cuBLAS handle management for the CUDA backend.
 *
 * Provides a single cuBLAS handle shared across all CUDA kernel files
 * (matmul, conv2d, fused_ops, cublas_ops) to avoid redundant handle
 * creation and inconsistent lifetime management.
 */

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace tenzor {
namespace cuda {

#ifndef CUBLAS_CHECK
#define CUBLAS_CHECK(call)                                                     \
    do {                                                                        \
        cublasStatus_t status = (call);                                        \
        if (status != CUBLAS_STATUS_SUCCESS) {                                 \
            throw std::runtime_error(                                          \
                std::string("cuBLAS error at ") + __FILE__ + ":" +            \
                std::to_string(__LINE__) + " - status code " +                \
                std::to_string(static_cast<int>(status)));                     \
        }                                                                      \
    } while (0)
#endif

class CuBLASHandlePool {
public:
    /// Returns a per-thread cuBLAS handle, optionally bound to the given stream.
    /// Each thread owns its own handle, so no data race on cublasSetStream.
    static cublasHandle_t get(cudaStream_t stream = nullptr) {
        static thread_local CuBLASHandlePool instance;
        if (stream != instance.last_stream_) {
            CUBLAS_CHECK(cublasSetStream(instance.handle_, stream));
            instance.last_stream_ = stream;
        }
        return instance.handle_;
    }

    static void shutdown() {
        // No-op: thread_local instances destroyed when each thread exits.
    }

private:
    CuBLASHandlePool() {
        CUBLAS_CHECK(cublasCreate(&handle_));
#if CUDA_VERSION >= 11000
        // TF32 Tensor Core math requires CUDA 11.0+ (Ampere and later GPUs).
        // The previous guard (>= 9000) was incorrect: CUBLAS_TF32_TENSOR_OP_MATH
        // is only defined starting with CUDA 11.0.
        CUBLAS_CHECK(cublasSetMathMode(handle_, CUBLAS_TF32_TENSOR_OP_MATH));
#endif
    }

    ~CuBLASHandlePool() {
        if (handle_) {
            cublasDestroy(handle_);
            handle_ = nullptr;
        }
    }

    CuBLASHandlePool(const CuBLASHandlePool&) = delete;
    CuBLASHandlePool& operator=(const CuBLASHandlePool&) = delete;

    cublasHandle_t handle_ = nullptr;
    cudaStream_t last_stream_ = nullptr;
};

} // namespace cuda
} // namespace tenzor
