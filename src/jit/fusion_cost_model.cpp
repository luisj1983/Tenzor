/**
 * @file fusion_cost_model.cpp
 * @brief Implementation of the fusion cost model
 */

#include "../../include/tenzor/jit/fusion_cost_model.hpp"

namespace tenzor {
namespace jit {

auto FusionCostModel::should_fuse(const FusionCandidate& candidate) const -> bool {
    // Trivial case: nothing to fuse
    if (candidate.num_ops <= 1) {
        return false;
    }

    // Estimate the cost savings from fusion:
    //
    // 1. Launch overhead saved: by fusing N ops into 1 kernel, we eliminate
    //    (N - 1) kernel launches.
    //    saved_launch_us = (num_ops - 1) * launch_overhead_us
    //
    // 2. Memory traffic saved: fusion eliminates intermediate stores and
    //    reloads between fused ops. Each eliminated intermediate access
    //    saves one read + one write of `total_elements` worth of data.
    //    We estimate the number of eliminated intermediates as
    //    (num_memory_accesses - 2) -- the fused kernel still needs one
    //    global read and one global write.
    //
    // Fusion is profitable when the total time saved exceeds a minimum
    // threshold, ensuring we don't fuse tiny operations where the overhead
    // of a fused kernel (register pressure, compilation time) outweighs
    // the benefit.

    double saved_launch_us = static_cast<double>(candidate.num_ops - 1) * launch_overhead_us_;

    // Estimate memory traffic savings (in bytes, assuming float32 = 4 bytes)
    constexpr double bytes_per_element = 4.0;
    size_t eliminated_accesses = 0;
    if (candidate.num_memory_accesses > 2) {
        eliminated_accesses = candidate.num_memory_accesses - 2;
    }

    double bytes_saved = static_cast<double>(eliminated_accesses)
                       * static_cast<double>(candidate.total_elements)
                       * bytes_per_element;

    // Convert bandwidth from GB/s to bytes/us: GB/s * 1e9 / 1e6 = GB/s * 1e3
    double bandwidth_bytes_per_us = bandwidth_gbps_ * 1e3;

    // Time saved from reduced memory traffic (in microseconds)
    double memory_time_saved_us = 0.0;
    if (bandwidth_bytes_per_us > 0.0) {
        memory_time_saved_us = bytes_saved / bandwidth_bytes_per_us;
    }

    double total_time_saved_us = saved_launch_us + memory_time_saved_us;

    // Fusion is profitable if we save at least launch_overhead_us worth of time.
    // This ensures the fusion benefit meaningfully exceeds the cost of a single
    // kernel launch (accounts for increased register pressure, etc.).
    return total_time_saved_us > launch_overhead_us_;
}

} // namespace jit
} // namespace tenzor
