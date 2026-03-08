#pragma once

#include <algorithm>
#include <cstddef>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace tenzor::cpu {

// Adaptive OpenMP thresholds based on operation complexity and thread count.
// For simple ops, the per-element work is tiny, so we need large tensors
// to amortize thread creation/join overhead (~50-100us per parallel region).
struct OmpThresholds {
    size_t simple;
    size_t medium;
    size_t complex;
    size_t matmul;
};

inline auto get_omp_thresholds() -> const OmpThresholds& {
    static const OmpThresholds t = [] {
        int n = 1;
#ifdef _OPENMP
        n = omp_get_max_threads();
#endif
        return OmpThresholds{
            std::max(size_t(65536), size_t(16384) * n),
            std::max(size_t(32768), size_t(8192) * n),
            std::max(size_t(8192), size_t(2048) * n),
            std::max(size_t(1024), size_t(256) * n)
        };
    }();
    return t;
}

} // namespace tenzor::cpu
