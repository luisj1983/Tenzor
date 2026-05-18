#pragma once
#include <cstdint>
#include <cstdlib>
#include <thread>
#include <mutex>

namespace tenzor {

/// Centralized OpenMP parallelism thresholds based on hardware concurrency.
/// Lazy-initialized, thread-safe via std::call_once.
///
/// All four thresholds can be overridden at runtime via environment variables:
///   TENZOR_OMP_THRESHOLD_SIMPLE   (default: 16384 * logical_cores)
///   TENZOR_OMP_THRESHOLD_MEDIUM   (default:  8192 * logical_cores)
///   TENZOR_OMP_THRESHOLD_COMPLEX  (default:  2048 * logical_cores)
///   TENZOR_OMP_THRESHOLD_MATMUL   (default: 1024)
struct OmpThresholds {
    /// Element-wise ops (add, mul, etc.): 16K * cores
    static int64_t simple()  { init(); return simple_; }
    /// Moderate ops (activations, broadcast): 8K * cores
    static int64_t medium()  { init(); return medium_; }
    /// Heavy ops (reductions, scans): 2K * cores
    static int64_t complex() { init(); return complex_; }
    /// Matrix ops: always parallelize above this
    static int64_t matmul()  { init(); return matmul_; }

private:
    static inline int64_t simple_  = 65536;
    static inline int64_t medium_  = 32768;
    static inline int64_t complex_ = 8192;
    static inline int64_t matmul_  = 1024;
    static inline std::once_flag init_flag_;

    static int64_t env_or(const char* name, int64_t default_val) {
        const char* val = std::getenv(name);
        if (val) {
            char* end = nullptr;
            long long parsed = std::strtoll(val, &end, 10);
            if (end != val && parsed > 0)
                return static_cast<int64_t>(parsed);
        }
        return default_val;
    }

    static void init() {
        std::call_once(init_flag_, [] {
            int cores = static_cast<int>(std::thread::hardware_concurrency());
            if (cores < 1) cores = 1;
            simple_  = env_or("TENZOR_OMP_THRESHOLD_SIMPLE",  16384LL * cores);
            medium_  = env_or("TENZOR_OMP_THRESHOLD_MEDIUM",   8192LL * cores);
            complex_ = env_or("TENZOR_OMP_THRESHOLD_COMPLEX",  2048LL * cores);
            matmul_  = env_or("TENZOR_OMP_THRESHOLD_MATMUL",   1024LL);
        });
    }
};

} // namespace tenzor
