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
#include <cstdint>

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
        // Rebind whenever the requested stream differs from the last one,
        // including stream==nullptr (the default stream 0). last_stream starts
        // at an impossible sentinel so the first call always binds. Guarding on
        // `stream &&` would leave the handle on a stale non-default stream when
        // a later call requests the default stream.
        if (stream != last_stream()) {
            CUSOLVER_CHECK(cusolverDnSetStream(handle(), stream));
            last_stream() = stream;
        }
        return handle();
    }

    static void shutdown() {
        // No-op: thread_local instances destroyed when each thread exits.
    }


private:
    struct HandleGuard {
        cusolverDnHandle_t handle = nullptr;
        // Sentinel that no real stream can equal, so get(nullptr) on first use
        // still issues the initial cusolverDnSetStream to the default stream.
        cudaStream_t last_stream = reinterpret_cast<cudaStream_t>(~uintptr_t(0));
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
