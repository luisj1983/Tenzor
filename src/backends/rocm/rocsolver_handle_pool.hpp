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
        static thread_local RocSOLVERHandlePool instance;
        if (stream && stream != instance.last_stream_) {
            ROCBLAS_CHECK_LINALG(rocblas_set_stream(instance.handle_, stream));
            instance.last_stream_ = stream;
        }
        return instance.handle_;
    }

    static void shutdown() {
        // No-op: thread_local instances destroyed when each thread exits.
    }

private:
    RocSOLVERHandlePool() {
        ROCBLAS_CHECK_LINALG(rocblas_create_handle(&handle_));
    }

    ~RocSOLVERHandlePool() {
        if (handle_) {
            rocblas_destroy_handle(handle_);
            handle_ = nullptr;
        }
    }

    RocSOLVERHandlePool(const RocSOLVERHandlePool&) = delete;
    RocSOLVERHandlePool& operator=(const RocSOLVERHandlePool&) = delete;

    rocblas_handle handle_ = nullptr;
    hipStream_t last_stream_ = nullptr;
};

} // namespace rocm
} // namespace tenzor
