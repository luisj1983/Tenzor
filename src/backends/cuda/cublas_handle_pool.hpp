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
        ensure_initialized();
        if (stream != last_stream()) {
            CUBLAS_CHECK(cublasSetStream(handle(), stream));
            last_stream() = stream;
        }
        return handle();
    }

    static void shutdown() {
        // No-op: thread_local instances destroyed when each thread exits.
    }

    /// W.8: destroy the thread-local cuBLAS handle so the next get() lazily
    /// rebuilds. Frees the workspace memory cuBLAS retains internally.
    /// Intended for backend reset_state() / long-running training loops
    /// that rotate streams and accumulate idle handles.
    

private:
    // RAII guard owned per-thread; destroys the handle when the thread exits.
    struct HandleGuard {
        cublasHandle_t handle = nullptr;
        cudaStream_t last_stream = nullptr;
        ~HandleGuard() {
            if (handle) {
                cublasDestroy(handle);
                handle = nullptr;
            }
        }
    };
    static HandleGuard& guard() {
        static thread_local HandleGuard g;
        return g;
    }
    static cublasHandle_t& handle() { return guard().handle; }
    static cudaStream_t& last_stream() { return guard().last_stream; }
    static void ensure_initialized() {
        if (handle() == nullptr) {
            CUBLAS_CHECK(cublasCreate(&handle()));
#if CUDA_VERSION >= 11000
            int dev = 0;
            cudaGetDevice(&dev);
            int major = 0;
            cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
            if (major >= 8) {
                CUBLAS_CHECK(cublasSetMathMode(handle(), CUBLAS_TF32_TENSOR_OP_MATH));
            }
#endif
        }
    }

    CuBLASHandlePool() = delete;
    CuBLASHandlePool(const CuBLASHandlePool&) = delete;
    CuBLASHandlePool& operator=(const CuBLASHandlePool&) = delete;
};

} // namespace cuda
} // namespace tenzor
