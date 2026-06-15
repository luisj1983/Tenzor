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
#include <cstdint>
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
        // Rebind whenever the requested stream differs from the last bound one.
        // last_stream is seeded with an impossible sentinel (not nullptr) so that
        // the very first request — including a request for the default stream
        // (nullptr) — actually binds, and a later switch back to the default
        // stream is not silently skipped (which would leave work on a stale,
        // non-default stream — a stream-ordering race).
        if (stream != last_stream()) {
            ROCSPARSE_CHECK(rocsparse_set_stream(handle(), stream));
            last_stream() = stream;
        }
        return handle();
    }

private:
    struct HandleGuard {
        rocsparse_handle handle = nullptr;
        // Impossible sentinel so the first get() — even for the default (nullptr)
        // stream — performs the bind, and switching back to the default stream is
        // never skipped.
        hipStream_t last_stream = reinterpret_cast<hipStream_t>(~uintptr_t(0));
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
