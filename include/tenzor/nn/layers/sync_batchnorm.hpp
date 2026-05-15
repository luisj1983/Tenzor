/**
 * @file sync_batchnorm.hpp
 * @brief Synchronized Batch Normalization for distributed training
 *
 * SyncBatchNorm synchronizes mean/variance computation across all processes,
 * ensuring consistent normalization statistics when training with DDP.
 *
 * The all-reduce operation is injected via a callback to avoid a hard
 * dependency on the distributed library from tenzor_core.
 */

#pragma once

#include "../module.hpp"
#include "../../core/device.hpp"
#include <functional>
#include <memory>

namespace tenzor {

namespace distributed {
class ProcessGroupBase;
}  // namespace distributed

namespace nn {

/**
 * @brief Callback type for all-reduce operations.
 *
 * Takes a mutable tensor reference and reduces it (SUM) in-place across all
 * processes.
 *
 * @note Prefer the new `ProcessGroupBase`-based constructor for new code;
 *       it enables autograd-aware all-reduce (audit C1) so that the
 *       distributed `world_size > 1` backward path supports higher-order
 *       gradients (`create_graph=true`) by composing the all-reduce into
 *       the autograd graph via `tenzor::distributed_all_reduce` (A5).
 */
using AllReduceFn = std::function<void(Tensor& tensor)>;

/**
 * @brief Synchronized Batch Normalization across distributed processes.
 *
 * Computes batch normalization where the mean and variance are synchronized
 * across all processes via an injected all-reduce callback.
 *
 * Shape:
 * - Input: (N, C, H, W)
 * - Output: (N, C, H, W)
 *
 * @code
 * // Create with DDP process group
 * auto all_reduce = [&pg](Tensor& t) { pg.all_reduce(t, ncclSum, 0); };
 * SyncBatchNorm sbn(64, all_reduce, world_size);
 * @endcode
 */
class SyncBatchNorm : public Module {
public:
    /**
     * @brief Construct synchronized batch normalization layer.
     *
     * @param num_features Number of feature channels (C dimension)
     * @param all_reduce_fn Callback to all-reduce a tensor (SUM) across ranks
     * @param world_size Number of processes in the group
     * @param eps Small constant for numerical stability (default: 1e-5)
     * @param momentum Momentum for running statistics update (default: 0.1)
     * @param affine If true, learn scale and shift parameters (default: true)
     * @param track_running_stats If true, track running mean/var (default: true)
     */
    SyncBatchNorm(int64_t num_features,
                  AllReduceFn all_reduce_fn,
                  int world_size = 1,
                  double eps = 1e-5,
                  double momentum = 0.1,
                  bool affine = true,
                  bool track_running_stats = true,
                  std::shared_ptr<distributed::ProcessGroupBase> process_group = nullptr);

    /**
     * @brief Construct synchronized batch normalization with an autograd-aware
     *        process group (audit C1).
     *
     * When this constructor is used, `SyncBatchNorm2dBackward::backward_with_variables`
     * composes the gradient-side all-reduce as a Variable-level
     * `distributed_all_reduce` call. That keeps the all-reduce inside the
     * autograd graph, so `create_graph=true` produces a real second-order
     * graph even for `world_size > 1`.
     *
     * The `AllReduceFn` callback is auto-synthesized from `process_group` so
     * the existing first-order forward/backward paths remain unchanged.
     *
     * @param num_features Number of feature channels (C dimension)
     * @param process_group Process group spanning the participating ranks
     * @param world_size Number of processes in the group (defaults to
     *        `process_group->world_size()`, which is the typical setup)
     */
    SyncBatchNorm(int64_t num_features,
                  std::shared_ptr<distributed::ProcessGroupBase> process_group,
                  int world_size = 0,
                  double eps = 1e-5,
                  double momentum = 0.1,
                  bool affine = true,
                  bool track_running_stats = true);

    auto forward_impl(const Variable& input) -> Variable override;

    [[nodiscard]] auto eps() const -> double { return eps_; }

    auto extra_repr() const -> std::string override {
        return "num_features=" + std::to_string(num_features_) +
               ", eps=" + std::to_string(eps_) +
               ", momentum=" + std::to_string(momentum_) +
               ", affine=" + std::string(affine_ ? "True" : "False") +
               ", world_size=" + std::to_string(world_size_);
    }

private:
    int64_t num_features_;
    double eps_;
    double momentum_;
    bool affine_;
    bool track_running_stats_;
    int world_size_;
    AllReduceFn all_reduce_fn_;
    // Autograd-aware process group (audit C1). When non-null,
    // SyncBatchNormBackward routes higher-order gradients through
    // `distributed_all_reduce` so the all-reduce stays in the graph.
    std::shared_ptr<distributed::ProcessGroupBase> pg_;

    Variable weight_;
    Variable bias_;
    Variable running_mean_;
    Variable running_var_;
    Variable num_batches_tracked_;
};

} // namespace nn
} // namespace tenzor
