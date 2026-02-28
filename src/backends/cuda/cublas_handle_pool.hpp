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
    static cublasHandle_t get(cudaStream_t stream = nullptr) {
        static CuBLASHandlePool instance;
        if (stream && stream != instance.last_stream_) {
            CUBLAS_CHECK(cublasSetStream(instance.handle_, stream));
            instance.last_stream_ = stream;
        }
        return instance.handle_;
    }

    static void shutdown() {
        // No-op: static singleton destroyed at program exit.
        // Backend shutdown relies on proper atexit ordering.
    }

private:
    CuBLASHandlePool() {
        CUBLAS_CHECK(cublasCreate(&handle_));
#if CUDA_VERSION >= 9000
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
