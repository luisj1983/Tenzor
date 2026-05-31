#pragma once

/**
 * @file cusparse_handle_pool.hpp
 * @brief Centralized cuSPARSE handle management for the CUDA backend.
 *
 * Provides a per-thread cuSPARSE handle shared across all CUDA sparse kernel
 * calls to avoid redundant handle creation and inconsistent lifetime management.
 * Follows the same pattern as cublas_handle_pool.hpp and cusolver_handle_pool.hpp.
 */

#include <cusparse.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace tenzor {
namespace cuda {

#ifndef CUSPARSE_CHECK
#define CUSPARSE_CHECK(call)                                                    \
    do {                                                                         \
        cusparseStatus_t status = (call);                                       \
        if (status != CUSPARSE_STATUS_SUCCESS) {                                \
            throw std::runtime_error(                                           \
                std::string("cuSPARSE error at ") + __FILE__ + ":" +           \
                std::to_string(__LINE__) + " - status code " +                 \
                std::to_string(static_cast<int>(status)));                      \
        }                                                                       \
    } while (0)
#endif

class CuSPARSEHandlePool {
public:
    /// Returns a per-thread cuSPARSE handle, optionally bound to the given stream.
    static cusparseHandle_t get(cudaStream_t stream = nullptr) {
        ensure_initialized();
        if (stream != last_stream()) {
            CUSPARSE_CHECK(cusparseSetStream(handle(), stream));
            last_stream() = stream;
        }
        return handle();
    }

    static void shutdown() {
        // No-op: thread_local instances destroyed when each thread exits.
    }

    /// W.8: destroy the thread-local cuSPARSE handle so the next get()
    /// lazily rebuilds.
    

private:
    struct HandleGuard {
        cusparseHandle_t handle = nullptr;
        cudaStream_t last_stream = reinterpret_cast<cudaStream_t>(~uintptr_t(0));
        ~HandleGuard() {
            if (handle) {
                cusparseDestroy(handle);
                handle = nullptr;
            }
        }
    };
    static HandleGuard& guard() {
        static thread_local HandleGuard g;
        return g;
    }
    static cusparseHandle_t& handle() { return guard().handle; }
    static cudaStream_t& last_stream() { return guard().last_stream; }
    static void ensure_initialized() {
        if (handle() == nullptr) {
            CUSPARSE_CHECK(cusparseCreate(&handle()));
        }
    }

    CuSPARSEHandlePool() = delete;
    CuSPARSEHandlePool(const CuSPARSEHandlePool&) = delete;
    CuSPARSEHandlePool& operator=(const CuSPARSEHandlePool&) = delete;
};

} // namespace cuda
} // namespace tenzor
