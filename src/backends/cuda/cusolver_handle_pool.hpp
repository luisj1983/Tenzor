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
        ensure_initialized();
        if (stream && stream != last_stream()) {
            CUSOLVER_CHECK(cusolverDnSetStream(handle(), stream));
            last_stream() = stream;
        }
        return handle();
    }

    static void shutdown() {
        // No-op: thread_local instances destroyed when each thread exits.
    }

    /// W.8: destroy the thread-local cuSOLVER handle so the next get() lazily
    /// rebuilds.  Frees the workspace memory cuSOLVER retains internally.
    static void clear_idle() {
        if (handle() != nullptr) {
            cusolverDnDestroy(handle());
            handle() = nullptr;
        }
        last_stream() = nullptr;
    }

private:
    struct HandleGuard {
        cusolverDnHandle_t handle = nullptr;
        cudaStream_t last_stream = nullptr;
        ~HandleGuard() {
            if (handle) {
                cusolverDnDestroy(handle);
                handle = nullptr;
            }
        }
    };
    static HandleGuard& guard() {
        static thread_local HandleGuard g;
        return g;
    }
    static cusolverDnHandle_t& handle() { return guard().handle; }
    static cudaStream_t& last_stream() { return guard().last_stream; }
    static void ensure_initialized() {
        if (handle() == nullptr) {
            CUSOLVER_CHECK(cusolverDnCreate(&handle()));
        }
    }

    CuSOLVERHandlePool() = delete;
    CuSOLVERHandlePool(const CuSOLVERHandlePool&) = delete;
    CuSOLVERHandlePool& operator=(const CuSOLVERHandlePool&) = delete;
};

} // namespace cuda
} // namespace tenzor
