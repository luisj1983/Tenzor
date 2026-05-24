/// @file onednn_cache.cpp
/// @brief Implementation of W.6's clear_dnnl_cache() multiplexer.
///
/// The per-op caches live as `thread_local` statics inside their respective
/// .cpp files. To clear them from a single entry point we maintain a global
/// list of clear callbacks; each callback is the per-file static
/// `clear_local_cache()` (registered via a static initializer). clear_dnnl_cache()
/// invokes each in turn on the calling thread.

#include "onednn_cache.hpp"

#include <mutex>
#include <vector>

namespace tenzor {
namespace cpu {

namespace {

std::mutex& callbacks_mutex() {
    static std::mutex m;
    return m;
}

std::vector<void (*)()>& callbacks() {
    static std::vector<void (*)()> cbs;
    return cbs;
}

} // anonymous namespace

void register_dnnl_cache_clear_callback(void (*cb)()) {
    if (cb == nullptr) return;
    std::lock_guard<std::mutex> g(callbacks_mutex());
    callbacks().push_back(cb);
}

void clear_dnnl_cache() {
    std::vector<void (*)()> snapshot;
    {
        std::lock_guard<std::mutex> g(callbacks_mutex());
        snapshot = callbacks();
    }
    for (auto* cb : snapshot) {
        cb();
    }
}

} // namespace cpu
} // namespace tenzor
