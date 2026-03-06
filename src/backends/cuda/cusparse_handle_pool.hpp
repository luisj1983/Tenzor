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
        static thread_local CuSPARSEHandlePool instance;
        if (stream != instance.last_stream_) {
            CUSPARSE_CHECK(cusparseSetStream(instance.handle_, stream));
            instance.last_stream_ = stream;
        }
        return instance.handle_;
    }

    static void shutdown() {
        // No-op: thread_local instances destroyed when each thread exits.
    }

private:
    CuSPARSEHandlePool() {
        CUSPARSE_CHECK(cusparseCreate(&handle_));
    }

    ~CuSPARSEHandlePool() {
        if (handle_) {
            cusparseDestroy(handle_);
            handle_ = nullptr;
        }
    }

    CuSPARSEHandlePool(const CuSPARSEHandlePool&) = delete;
    CuSPARSEHandlePool& operator=(const CuSPARSEHandlePool&) = delete;

    cusparseHandle_t handle_ = nullptr;
    cudaStream_t last_stream_ = nullptr;
};

} // namespace cuda
} // namespace tenzor
