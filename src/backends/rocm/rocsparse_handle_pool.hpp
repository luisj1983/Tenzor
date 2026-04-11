#pragma once

/**
 * @file rocsparse_handle_pool.hpp
 * @brief Centralized rocSPARSE handle management for the ROCm backend.
 *
 * Provides a per-thread rocSPARSE handle shared across all ROCm sparse kernel
 * calls to avoid redundant handle creation and inconsistent lifetime management.
 * Follows the same pattern as rocsolver_handle_pool.hpp and the CUDA
 * cusparse_handle_pool.hpp.
 */

#ifdef TENZOR_HAS_ROCSPARSE

#include <rocsparse/rocsparse.h>
#include <hip/hip_runtime.h>
#include <stdexcept>
#include <string>

#include "tenzor/backend/loader_fwd.hpp"

namespace tenzor {
namespace rocm {

#ifndef ROCSPARSE_CHECK
#define ROCSPARSE_CHECK(call)                                                   \
    do {                                                                         \
        rocsparse_status status = (call);                                       \
        if (status != rocsparse_status_success) {                               \
            throw std::runtime_error(                                           \
                std::string("rocSPARSE error at ") + __FILE__ + ":" +          \
                std::to_string(__LINE__) + " - status " +                      \
                std::to_string(static_cast<int>(status)));                     \
        }                                                                       \
    } while (0)
#endif

class RocSPARSEHandlePool {
public:
    /// Returns a per-thread rocSPARSE handle, optionally bound to the given stream.
    static rocsparse_handle get(hipStream_t stream = nullptr) {
        static thread_local RocSPARSEHandlePool instance;
        if (stream && stream != instance.last_stream_) {
            ROCSPARSE_CHECK(rocsparse_set_stream(instance.handle_, stream));
            instance.last_stream_ = stream;
        }
        return instance.handle_;
    }

    static void shutdown() {
        // No-op: thread_local instances destroyed when each thread exits.
    }

private:
    RocSPARSEHandlePool() {
        ROCSPARSE_CHECK(rocsparse_create_handle(&handle_));
    }

    ~RocSPARSEHandlePool() noexcept {
        if (handle_ && is_backend_registry_alive()) {
            try { rocsparse_destroy_handle(handle_); }
            catch (...) { /* destructor must not throw */ }
            handle_ = nullptr;
        }
    }

    RocSPARSEHandlePool(const RocSPARSEHandlePool&) = delete;
    RocSPARSEHandlePool& operator=(const RocSPARSEHandlePool&) = delete;

    rocsparse_handle handle_ = nullptr;
    hipStream_t last_stream_ = nullptr;
};

} // namespace rocm
} // namespace tenzor

#endif // TENZOR_HAS_ROCSPARSE
