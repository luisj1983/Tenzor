#pragma once

/**
 * @file rocsolver_handle_pool.hpp
 * @brief Centralized rocSOLVER handle management for the ROCm backend.
 *
 * rocSOLVER uses rocBLAS handles, so this provides a per-thread rocBLAS handle
 * shared across all ROCm linalg kernel calls. Follows the same pattern as
 * the CUDA cusolver_handle_pool.hpp.
 */

#include <rocblas/rocblas.h>
#include <hip/hip_runtime.h>
#include <stdexcept>
#include <string>

#include "tenzor/backend/loader_fwd.hpp"

namespace tenzor {
namespace rocm {

#ifndef ROCBLAS_CHECK_LINALG
#define ROCBLAS_CHECK_LINALG(call)                                              \
    do {                                                                         \
        rocblas_status status = (call);                                          \
        if (status != rocblas_status_success) {                                  \
            throw std::runtime_error(                                            \
                std::string("rocBLAS/rocSOLVER error at ") + __FILE__ + ":" +   \
                std::to_string(__LINE__) + " - status code " +                  \
                std::to_string(static_cast<int>(status)));                       \
        }                                                                       \
    } while (0)
#endif

class RocSOLVERHandlePool {
public:
    /// Returns a per-thread rocBLAS handle (used by rocSOLVER), optionally bound to the given stream.
    static rocblas_handle get(hipStream_t stream = nullptr) {
        ensure_initialized();
        if (stream && stream != last_stream()) {
            ROCBLAS_CHECK_LINALG(rocblas_set_stream(handle(), stream));
            last_stream() = stream;
        }
        return handle();
    }

private:
    struct HandleGuard {
        rocblas_handle handle = nullptr;
        hipStream_t last_stream = nullptr;
        // Guard teardown with the backend-alive check (mirrors RocSPARSEHandlePool):
        // destroying a rocBLAS handle after the backend library has unloaded calls
        // into freed code. noexcept + try/catch because destructors must not throw.
        ~HandleGuard() noexcept {
            if (handle && is_backend_registry_alive()) {
                try { rocblas_destroy_handle(handle); }
                catch (...) { /* destructor must not throw */ }
                handle = nullptr;
            }
        }
    };
    static HandleGuard& guard() {
        static thread_local HandleGuard g;
        return g;
    }
    static rocblas_handle& handle() { return guard().handle; }
    static hipStream_t& last_stream() { return guard().last_stream; }
    static void ensure_initialized() {
        if (handle() == nullptr) {
            ROCBLAS_CHECK_LINALG(rocblas_create_handle(&handle()));
        }
    }

    RocSOLVERHandlePool() = delete;
    RocSOLVERHandlePool(const RocSOLVERHandlePool&) = delete;
    RocSOLVERHandlePool& operator=(const RocSOLVERHandlePool&) = delete;
};

} // namespace rocm
} // namespace tenzor
