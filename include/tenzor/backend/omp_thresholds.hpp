#pragma once
#include <cstdint>
#include <thread>
#include <mutex>

namespace tenzor {

/// Centralized OpenMP parallelism thresholds based on hardware concurrency.
/// Lazy-initialized, thread-safe via std::call_once.
struct OmpThresholds {
    /// Element-wise ops (add, mul, etc.): 16K * cores
    static int64_t simple() { init(); return simple_; }
    /// Moderate ops (activations, broadcast): 8K * cores
    static int64_t medium() { init(); return medium_; }
    /// Heavy ops (reductions, scans): 2K * cores
    static int64_t complex() { init(); return complex_; }
    /// Matrix ops: always parallelize above this
    static int64_t matmul() { return 1024; }

private:
    static inline int64_t simple_ = 65536;
    static inline int64_t medium_ = 32768;
    static inline int64_t complex_ = 8192;
    static inline std::once_flag init_flag_;

    static void init() {
        std::call_once(init_flag_, [] {
            int cores = static_cast<int>(std::thread::hardware_concurrency());
            if (cores < 1) cores = 1;
            simple_ = 16384LL * cores;
            medium_ = 8192LL * cores;
            complex_ = 2048LL * cores;
        });
    }
};

} // namespace tenzor
