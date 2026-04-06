/**
 * @file sync_batchnorm.cpp
 * @brief Synchronized Batch Normalization implementation
 *
 * Synchronizes mean/variance across distributed processes via an injected
 * all-reduce callback. Falls back to regular BatchNorm when world_size == 1.
 */

#include "tenzor/nn/layers/sync_batchnorm.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/reduction.hpp"
#include "tenzor/ops/transform.hpp"
#include "tenzor/autograd/variable.hpp"
#include <cmath>
#include <stdexcept>

namespace tenzor::nn {

SyncBatchNorm::SyncBatchNorm(
    int64_t num_features,
    AllReduceFn all_reduce_fn,
    int world_size,
    double eps,
    double momentum,
    bool affine,
    bool track_running_stats)
    : num_features_(num_features),
      eps_(eps),
      momentum_(momentum),
      affine_(affine),
      track_running_stats_(track_running_stats),
      world_size_(world_size),
      all_reduce_fn_(std::move(all_reduce_fn))
{
    if (num_features <= 0) {
        throw std::invalid_argument("SyncBatchNorm: num_features must be positive");
    }

    if (affine_) {
        weight_ = Variable(ones({num_features}, DType::Float32), true);
        bias_ = Variable(zeros({num_features}, DType::Float32), true);
        register_parameter("weight", weight_);
        register_parameter("bias", bias_);
    }

    if (track_running_stats_) {
        running_mean_ = Variable(zeros({num_features}, DType::Float32), false);
        running_var_ = Variable(ones({num_features}, DType::Float32), false);
        num_batches_tracked_ = Variable(zeros({1}, DType::Int64), false);
        register_buffer("running_mean", running_mean_);
        register_buffer("running_var", running_var_);
        register_buffer("num_batches_tracked", num_batches_tracked_);
    }
}

auto SyncBatchNorm::forward_impl(const Variable& input) -> Variable {
    auto x = input.tensor();
    auto shape = x.shape();

    if (shape.size() != 4) {
        throw std::invalid_argument(
            "SyncBatchNorm: expected 4D input (N,C,H,W), got " +
            std::to_string(shape.size()) + "D");
    }

    int64_t N = shape[0];
    int64_t C = shape[1];
    int64_t H = shape[2];
    int64_t W = shape[3];

    if (C != num_features_) {
        throw std::invalid_argument(
            "SyncBatchNorm: expected " + std::to_string(num_features_) +
            " channels, got " + std::to_string(C));
    }

    Tensor batch_mean, batch_var;

    if (is_training()) {
        int64_t local_count = N * H * W;

        // Compute in FP32 for numerical stability
        auto x_fp32 = (x.dtype() != DType::Float32) ? x.to(DType::Float32) : x;

        // Per-channel local sum: reduce over N, H, W -> [C]
        auto local_sum = sum(sum(sum(x_fp32, 3, false), 2, false), 0, false);

        // Per-channel local sum of squares
        auto x_sq = x_fp32 * x_fp32;
        auto local_sum_sq = sum(sum(sum(x_sq, 3, false), 2, false), 0, false);

        // Pack [local_sum, local_sum_sq, count] for a single all-reduce
        auto count_tensor = full({1}, static_cast<float>(local_count), DType::Float32);
        auto packed = cat({local_sum, local_sum_sq, count_tensor}, 0);

        // All-reduce across ranks (SUM)
        if (world_size_ > 1 && all_reduce_fn_) {
            all_reduce_fn_(packed);
        }

        // Unpack global statistics
        auto global_sum = packed.slice(0, 0, C);
        auto global_sum_sq = packed.slice(0, C, 2 * C);
        auto global_count_t = packed.slice(0, 2 * C, 2 * C + 1);
        float global_count = global_count_t.data<float>()[0];
        float inv_count = 1.0f / global_count;

        batch_mean = global_sum * inv_count;
        // Var = E[X^2] - E[X]^2
        batch_var = global_sum_sq * inv_count - batch_mean * batch_mean;

        // Update running statistics
        if (track_running_stats_) {
            float decay = static_cast<float>(1.0 - momentum_);
            float mom = static_cast<float>(momentum_);
            auto rm = running_mean_.tensor() * decay + batch_mean * mom;
            auto rv = running_var_.tensor() * decay + batch_var * mom;
            running_mean_ = Variable(rm, false);
            running_var_ = Variable(rv, false);
        }
    } else {
        if (!track_running_stats_) {
            throw std::runtime_error(
                "SyncBatchNorm: cannot use eval mode without track_running_stats");
        }
        batch_mean = running_mean_.tensor();
        batch_var = running_var_.tensor();
    }

    // Normalize: y = (x - mean) / sqrt(var + eps) * weight + bias
    auto mean_4d = batch_mean.reshape({1, C, 1, 1});
    auto var_4d = batch_var.reshape({1, C, 1, 1});

    auto inv_std = reciprocal(sqrt(var_4d + static_cast<float>(eps_)));
    auto normalized = (x - mean_4d) * inv_std;

    if (affine_) {
        auto w = weight_.tensor().reshape({1, C, 1, 1});
        auto b = bias_.tensor().reshape({1, C, 1, 1});
        normalized = normalized * w + b;
    }

    return Variable(normalized, input.requires_grad());
}

} // namespace tenzor::nn
