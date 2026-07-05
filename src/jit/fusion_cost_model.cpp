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

    // Estimate memory traffic savings (in bytes). Use the candidate's actual
    // element size so F16/BF16 (2B) and F64/Complex (8/16B) are not modelled as
    // Float32 (which over/under-estimates the traffic saved by fusion).
    const double bytes_per_element =
        static_cast<double>(candidate.bytes_per_element == 0 ? 4 : candidate.bytes_per_element);
    size_t eliminated_accesses = 0;
    if (candidate.num_memory_accesses > 2) {
        eliminated_accesses = candidate.num_memory_accesses - 2;
    }

    // total_elements < 0 is the "unknown element count" sentinel (a dynamic
    // or invalid input dim): we cannot estimate memory-traffic savings, so be
    // conservative and credit only the launch-overhead savings below.
    double bytes_saved = candidate.total_elements < 0
        ? 0.0
        : static_cast<double>(eliminated_accesses)
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

auto FusionCostModel::set_device_type(Device::Type type) -> void {
    device_type_ = type;
}

auto FusionCostModel::estimate_speedup(const FusionCandidate& candidate) const -> double {
    // Trivial case: nothing to fuse
    if (candidate.num_ops <= 1) {
        return 0.0;
    }

    // Check for known-beneficial fusion patterns first.
    // These patterns have well-understood performance characteristics
    // and get fixed speedup estimates regardless of element count.
    switch (candidate.kind) {
        case FusionKind::Softmax:
            return 2.0;
        case FusionKind::LayerNorm:
            return 1.8;
        // SwiGLU removed from the cost model: the pattern matcher no longer emits
        // FusionKind::SwiGLU (the extended codegen cannot generate it), so a 1.5x
        // speedup estimate here would only ever mis-score a kind that can never be
        // formed. Its constituent ops run via normal dispatch.
        case FusionKind::GemmEpilogue:
            return 1.3;
        default:
            break;
    }

    // Unknown element count (dynamic/invalid input dim): the element-count
    // heuristics below would multiply through a garbage value. Be conservative
    // and report a neutral speedup so an unknown-size fusion is not greedily
    // selected on a fabricated benefit.
    if (candidate.total_elements < 0) {
        return 1.0;
    }

    bool is_gpu = (device_type_ == Device::Type::CUDA ||
                   device_type_ == Device::Type::ROCm);

    if (is_gpu) {
        // GPU heuristic: kernel launch overhead (~5us) vs compute savings.
        // Small workloads don't amortize launch overhead of a fused kernel.
        if (candidate.total_elements < 1024) {
            return 0.8;
        }

        // Scale speedup based on element count: more elements -> more
        // memory traffic saved -> higher speedup. Range [1.2, 2.0].
        constexpr int64_t low_threshold = 1024;
        constexpr int64_t high_threshold = 1024 * 1024;
        double t = static_cast<double>(candidate.total_elements - low_threshold)
                 / static_cast<double>(high_threshold - low_threshold);
        if (t > 1.0) t = 1.0;
        return 1.2 + t * 0.8;  // [1.2, 2.0]

    } else {
        // CPU heuristic: memory bandwidth savings from keeping data in cache.
        // Working set = total_elements * 4 bytes (float32).
        // ~1MB L2 divided by the actual element size (not a hardcoded 4 bytes).
        const int64_t l2_elements =
            1024 * 1024 / static_cast<int64_t>(candidate.bytes_per_element == 0 ? 4 : candidate.bytes_per_element);

        if (candidate.total_elements <= l2_elements) {
            // Fits in L2: fusion keeps intermediates in cache -> good savings.
            // Scale in [1.1, 1.5] based on number of eliminated intermediates.
            double mem_factor = static_cast<double>(candidate.num_memory_accesses) / 10.0;
            if (mem_factor > 1.0) mem_factor = 1.0;
            return 1.1 + mem_factor * 0.4;  // [1.1, 1.5]
        } else {
            // Exceeds L2: still some benefit from reduced memory traffic,
            // but intermediates spill to main memory anyway.
            double mem_factor = static_cast<double>(candidate.num_memory_accesses) / 10.0;
            if (mem_factor > 1.0) mem_factor = 1.0;
            return 1.0 + mem_factor * 0.2;  // [1.0, 1.2]
        }
    }
}

} // namespace jit
} // namespace tenzor
