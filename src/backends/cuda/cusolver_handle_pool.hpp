#pragma once

/**
 * @file cusolver_handle_pool.hpp
 * @brief Centralized cuSOLVER handle management for the CUDA backend.
 *
 * Provides a per-thread cuSOLVER handle shared across all CUDA linalg kernel
 * calls to avoid redundant handle creation and inconsistent lifetime management.
 * Follows the same pattern as cublas_handle_pool.hpp.
 */

#include <cusolverDn.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace tenzor {
namespace cuda {

#ifndef CUSOLVER_CHECK
#define CUSOLVER_CHECK(call)                                                    \
    do {                                                                         \
        cusolverStatus_t status = (call);                                       \
        if (status != CUSOLVER_STATUS_SUCCESS) {                                \
            throw std::runtime_error(                                           \
                std::string("cuSOLVER error at ") + __FILE__ + ":" +           \
                std::to_string(__LINE__) + " - status code " +                 \
                std::to_string(static_cast<int>(status)));                      \
        }                                                                       \
    } while (0)
#endif

class CuSOLVERHandlePool {
public:
    /// Returns a per-thread cuSOLVER handle, optionally bound to the given stream.
    static cusolverDnHandle_t get(cudaStream_t stream = nullptr) {
        static thread_local CuSOLVERHandlePool instance;
        if (stream && stream != instance.last_stream_) {
            CUSOLVER_CHECK(cusolverDnSetStream(instance.handle_, stream));
            instance.last_stream_ = stream;
        }
        return instance.handle_;
    }

    static void shutdown() {
        // No-op: thread_local instances destroyed when each thread exits.
    }

private:
    CuSOLVERHandlePool() {
        CUSOLVER_CHECK(cusolverDnCreate(&handle_));
    }

    ~CuSOLVERHandlePool() {
        if (handle_) {
            cusolverDnDestroy(handle_);
            handle_ = nullptr;
        }
    }

    CuSOLVERHandlePool(const CuSOLVERHandlePool&) = delete;
    CuSOLVERHandlePool& operator=(const CuSOLVERHandlePool&) = delete;

    cusolverDnHandle_t handle_ = nullptr;
    cudaStream_t last_stream_ = nullptr;
};

} // namespace cuda
} // namespace tenzor
