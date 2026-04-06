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

namespace tenzor {
namespace nn {

/**
 * @brief Callback type for all-reduce operations.
 *
 * Takes a mutable tensor reference and reduces it (SUM) in-place across all
 * processes. Also returns the world size.
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

    Variable weight_;
    Variable bias_;
    Variable running_mean_;
    Variable running_var_;
    Variable num_batches_tracked_;
};

} // namespace nn
} // namespace tenzor
