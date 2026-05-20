#pragma once
#include <cstdint>

namespace tenzor {

/// Centralized OpenMP parallelism thresholds based on hardware concurrency.
/// Lazy-initialized, thread-safe — `init()` lives in a .cpp file so this
/// header keeps a minimal include footprint (audit item F.5: avoid
/// dragging <thread> + <mutex> into every kernel translation unit).
///
/// All four thresholds can be overridden at runtime via environment
/// variables:
///   TENZOR_OMP_THRESHOLD_SIMPLE   (default: 16384 * logical_cores)
///   TENZOR_OMP_THRESHOLD_MEDIUM   (default:  8192 * logical_cores)
///   TENZOR_OMP_THRESHOLD_COMPLEX  (default:  2048 * logical_cores)
///   TENZOR_OMP_THRESHOLD_MATMUL   (default: 1024)
struct OmpThresholds {
    /// Element-wise ops (add, mul, etc.): 16K * cores
    static int64_t simple();
    /// Moderate ops (activations, broadcast): 8K * cores
    static int64_t medium();
    /// Heavy ops (reductions, scans): 2K * cores
    static int64_t complex();
    /// Matrix ops: always parallelize above this
    static int64_t matmul();
};

} // namespace tenzor
