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
        ensure_initialized();
        if (stream && stream != last_stream()) {
            ROCSPARSE_CHECK(rocsparse_set_stream(handle(), stream));
            last_stream() = stream;
        }
        return handle();
    }

private:
    struct HandleGuard {
        rocsparse_handle handle = nullptr;
        hipStream_t last_stream = nullptr;
        ~HandleGuard() noexcept {
            if (handle && is_backend_registry_alive()) {
                try { rocsparse_destroy_handle(handle); }
                catch (...) { /* destructor must not throw */ }
                handle = nullptr;
            }
        }
    };
    static HandleGuard& guard() {
        static thread_local HandleGuard g;
        return g;
    }
    static rocsparse_handle& handle() { return guard().handle; }
    static hipStream_t& last_stream() { return guard().last_stream; }
    static void ensure_initialized() {
        if (handle() == nullptr) {
            ROCSPARSE_CHECK(rocsparse_create_handle(&handle()));
        }
    }

    RocSPARSEHandlePool() = delete;
    RocSPARSEHandlePool(const RocSPARSEHandlePool&) = delete;
    RocSPARSEHandlePool& operator=(const RocSPARSEHandlePool&) = delete;
};

} // namespace rocm
} // namespace tenzor

#endif // TENZOR_HAS_ROCSPARSE
