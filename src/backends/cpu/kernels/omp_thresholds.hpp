#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdlib>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace tenzor::cpu {

// Adaptive OpenMP thresholds based on operation complexity and thread count.
// For simple ops, the per-element work is tiny, so we need large tensors
// to amortize thread creation/join overhead (~50-100us per parallel region).
//
// Override at runtime via environment variables:
//   TENZOR_OMP_THRESHOLD_SIMPLE   (default: max(65536, 16384 * threads))
//   TENZOR_OMP_THRESHOLD_MEDIUM   (default: max(32768, 8192 * threads))
//   TENZOR_OMP_THRESHOLD_COMPLEX  (default: max(8192, 2048 * threads))
//   TENZOR_OMP_THRESHOLD_MATMUL   (default: max(1024, 256 * threads))
struct OmpThresholds {
    size_t simple;
    size_t medium;
    size_t complex;
    size_t matmul;
};

namespace detail {

inline auto env_or(const char* name, size_t default_val) -> size_t {
    const char* val = std::getenv(name);
    if (val) {
        char* end = nullptr;
        unsigned long long parsed = std::strtoull(val, &end, 10);
        if (end != val && parsed > 0) {
            return static_cast<size_t>(parsed);
        }
    }
    return default_val;
}

} // namespace detail

inline auto get_omp_thresholds() -> const OmpThresholds& {
    static const OmpThresholds t = [] {
        int n = 1;
#ifdef _OPENMP
        n = omp_get_max_threads();
#endif
        return OmpThresholds{
            detail::env_or("TENZOR_OMP_THRESHOLD_SIMPLE",  std::max(size_t(65536), size_t(16384) * n)),
            detail::env_or("TENZOR_OMP_THRESHOLD_MEDIUM",  std::max(size_t(32768), size_t(8192) * n)),
            detail::env_or("TENZOR_OMP_THRESHOLD_COMPLEX", std::max(size_t(8192), size_t(2048) * n)),
            detail::env_or("TENZOR_OMP_THRESHOLD_MATMUL",  std::max(size_t(1024), size_t(256) * n))
        };
    }();
    return t;
}

} // namespace tenzor::cpu
