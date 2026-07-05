/**
 * @file fusion_cost_model.hpp
 * @brief Cost model for deciding when operator fusion is profitable
 *
 * Provides heuristics for determining whether fusing a set of operations
 * into a single kernel will improve performance, based on device bandwidth,
 * kernel launch overhead, and memory access patterns.
 */

#pragma once

#include <cstddef>
#include <cstdint>

#include "codegen.hpp"
#include "../core/device.hpp"

namespace tenzor {
namespace jit {

/**
 * @brief Describes a candidate set of operations for fusion.
 *
 * Contains the metrics needed by the cost model to evaluate
 * whether fusion is worthwhile.
 */
struct FusionCandidate {
    size_t num_ops{0};               ///< Number of operations in the candidate
    int64_t total_elements{0};       ///< Total number of elements processed
    size_t num_memory_accesses{0};   ///< Number of separate memory read/write ops
    FusionKind kind{FusionKind::ElementWise};  ///< Kind of fusion pattern
    size_t bytes_per_element{4};     ///< Element size in bytes of the fused
                                     ///< tensors' dtype (default 4 = Float32).
                                     ///< Used for memory-traffic estimates so
                                     ///< F16/BF16 (2B) and F64/Complex (8/16B)
                                     ///< are not mis-modelled as Float32.
};

/**
 * @brief Cost model for evaluating fusion profitability.
 *
 * Uses a simple heuristic: fusion is profitable when the kernel launch
 * overhead saved by reducing the number of kernels exceeds the benefit
 * of keeping operations separate.
 *
 * The key insight is that fusion eliminates intermediate memory traffic
 * and reduces kernel launch overhead. The model compares:
 * - Cost saved: (num_ops - 1) * launch_overhead
 * - Memory savings: reduced intermediate reads/writes at device bandwidth
 *
 * @code
 * FusionCostModel model;
 * model.set_device_bandwidth_gbps(900.0);  // e.g., H100
 * model.set_kernel_launch_overhead_us(5.0);
 *
 * FusionCandidate candidate{3, 1000000, 6};
 * if (model.should_fuse(candidate)) {
 *     // Proceed with fusion
 * }
 * @endcode
 */
class FusionCostModel {
public:
    /**
     * @brief Evaluate whether the candidate should be fused.
     *
     * Fuses when: num_ops * launch_overhead > memory_savings_from_fusion.
     * Memory savings are estimated from eliminated intermediate accesses.
     *
     * @param candidate Fusion candidate to evaluate
     * @return true if fusion is expected to improve performance
     */
    auto should_fuse(const FusionCandidate& candidate) const -> bool;

    /**
     * @brief Configure cost model for a specific device type.
     *
     * Adjusts internal heuristics based on device characteristics
     * (kernel launch overhead, memory hierarchy, bandwidth).
     *
     * @param type Target device type
     */
    auto set_device_type(Device::Type type) -> void;

    /**
     * @brief Estimate speedup ratio for a fusion candidate.
     *
     * Returns a value indicating how much faster the fused kernel is
     * expected to be compared to unfused execution. A value > 1.0
     * means fusion is beneficial; < 1.0 means fusion would hurt.
     *
     * The estimate accounts for device-specific characteristics:
     * - GPU: kernel launch overhead vs compute savings
     * - CPU: memory bandwidth savings and cache effects
     * - Known-beneficial patterns receive higher speedup estimates
     *
     * @param candidate Fusion candidate to evaluate
     * @return Estimated speedup ratio (>1.0 means beneficial)
     */
    auto estimate_speedup(const FusionCandidate& candidate) const -> double;

    /**
     * @brief Set device memory bandwidth in GB/s.
     *
     * @param bw Bandwidth in GB/s (default: 900.0 for H100-class GPU)
     */
    auto set_device_bandwidth_gbps(double bw) -> void { bandwidth_gbps_ = bw; }

    /**
     * @brief Set kernel launch overhead in microseconds.
     *
     * @param us Launch overhead in microseconds (default: 5.0)
     */
    auto set_kernel_launch_overhead_us(double us) -> void { launch_overhead_us_ = us; }

    /**
     * @brief Get current device bandwidth setting.
     *
     * @return Bandwidth in GB/s
     */
    auto device_bandwidth_gbps() const -> double { return bandwidth_gbps_; }

    /**
     * @brief Get current kernel launch overhead setting.
     *
     * @return Launch overhead in microseconds
     */
    auto kernel_launch_overhead_us() const -> double { return launch_overhead_us_; }

private:
    double bandwidth_gbps_{900.0};      ///< Device memory bandwidth (GB/s)
    double launch_overhead_us_{5.0};    ///< Kernel launch overhead (microseconds)
    Device::Type device_type_{Device::Type::CPU};  ///< Target device type
};

} // namespace jit
} // namespace tenzor
