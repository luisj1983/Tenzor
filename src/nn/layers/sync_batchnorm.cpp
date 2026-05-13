/**
 * @file sync_batchnorm.cpp
 * @brief Synchronized Batch Normalization implementation
 *
 * Synchronizes mean/variance across distributed processes via an injected
 * all-reduce callback. Falls back to regular BatchNorm when world_size == 1.
 */

#include "tenzor/nn/layers/sync_batchnorm.hpp"
#include "tenzor/autograd/function.hpp"
#include "tenzor/autograd/ops.hpp"
#include "tenzor/autograd/variable.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/reduction.hpp"
#include "tenzor/ops/transform.hpp"
#include <cmath>
#include <stdexcept>
#include <utility>

namespace tenzor::nn {

namespace {

// Custom autograd Function for SyncBatchNorm backward pass.
//
// Saved tensors (in order): input, mean, invstd, weight
//
// For world_size == 1 the math is identical to standard BatchNorm2d backward.
// For world_size > 1 the gradient sums would also need to be all-reduced
// (∂L/∂γ, ∂L/∂β are sums, and the ∂L/∂x term contains sum(∂L/∂y) and
//  sum(∂L/∂y * x_norm) which must be global). The all-reduce callback is
// captured here so that backward can use it on the reduction terms.
class SyncBatchNormBackward : public Function {
public:
    SyncBatchNormBackward(bool affine,
                          int world_size,
                          AllReduceFn all_reduce_fn,
                          int64_t global_count,
                          std::vector<Tensor> tensors_to_save)
        : affine_(affine),
          world_size_(world_size),
          all_reduce_fn_(std::move(all_reduce_fn)),
          global_count_(global_count) {
        save_for_backward(std::move(tensors_to_save));
    }

    auto forward(std::vector<Variable> /*inputs*/) -> std::vector<Variable> override {
        throw std::runtime_error("SyncBatchNormBackward::forward should not be called");
    }

    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override {
        auto grad_output = grad_outputs[0].contiguous();
        auto saved = saved_tensors();
        auto input  = saved[0].contiguous();
        auto mean   = saved[1].contiguous();
        auto invstd = saved[2].contiguous();
        auto weight = saved[3].contiguous();

        const auto shape = input.shape();
        const int64_t N = shape[0];
        const int64_t C = shape[1];
        const int64_t H = shape[2];
        const int64_t W = shape[3];
        const int64_t spatial_size = H * W;

        // For Float16, upcast to Float32 to avoid overflow/precision loss
        // (matches the BatchNorm2d backward path).
        DType orig_dtype = input.dtype();
        bool needs_upcast = (orig_dtype == DType::Float16 || orig_dtype == DType::BFloat16);
        if (needs_upcast) {
            grad_output = grad_output.to(DType::Float32);
            input       = input.to(DType::Float32);
            mean        = mean.to(DType::Float32);
            invstd      = invstd.to(DType::Float32);
            weight      = weight.to(DType::Float32);
        }

        auto mean_4d   = mean.unsqueeze(0).unsqueeze(2).unsqueeze(3).contiguous();
        auto invstd_4d = invstd.unsqueeze(0).unsqueeze(2).unsqueeze(3).contiguous();
        auto normalized = ((input - mean_4d) * invstd_4d).contiguous();

        // ∂L/∂γ = sum(∂L/∂y * x_norm) over [N, H, W] -> [C]
        auto grad_weight = sum(sum((grad_output * normalized)
            .reshape({N, C, spatial_size}).contiguous(), 0, false), 1, false);

        // ∂L/∂β = sum(∂L/∂y) over [N, H, W] -> [C]
        auto grad_bias = sum(sum(grad_output
            .reshape({N, C, spatial_size}).contiguous(), 0, false), 1, false);

        // For SyncBatchNorm with world_size > 1, the gradient reductions must
        // also be all-reduced so every rank computes ∂L/∂x using global sums.
        if (world_size_ > 1 && all_reduce_fn_) {
            all_reduce_fn_(grad_weight);
            all_reduce_fn_(grad_bias);
        }

        // Build ∂L/∂x using the standard fused BN backward formula.
        auto weight_4d = weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).contiguous();
        auto grad_normalized = (grad_output * weight_4d).contiguous();

        auto grad_in_resh = grad_normalized.reshape({N, C, spatial_size}).contiguous();
        auto norm_resh    = normalized.reshape({N, C, spatial_size}).contiguous();

        // sums per channel: shape [1, C, 1]
        auto sum_grad       = sum(sum(grad_in_resh, 0, true), 2, true).contiguous();
        auto sum_grad_xnorm = sum(sum((grad_in_resh * norm_resh), 0, true), 2, true).contiguous();

        // For sync BN, both per-channel sums must be the global ones too.
        if (world_size_ > 1 && all_reduce_fn_) {
            all_reduce_fn_(sum_grad);
            all_reduce_fn_(sum_grad_xnorm);
        }

        const float inv_global_count = 1.0f / static_cast<float>(global_count_);
        auto invstd_3d = invstd.unsqueeze(0).unsqueeze(-1).contiguous();

        auto term1 = (sum_grad * inv_global_count).contiguous();
        auto term2 = (norm_resh * sum_grad_xnorm * inv_global_count).contiguous();
        auto grad_input = ((grad_in_resh - term1 - term2) * invstd_3d).contiguous();

        grad_input = grad_input.reshape({N, C, H, W}).contiguous();

        if (needs_upcast) {
            grad_input  = grad_input.to(orig_dtype);
            grad_weight = grad_weight.to(orig_dtype);
            grad_bias   = grad_bias.to(orig_dtype);
        }

        if (!affine_) {
            // Bias/weight aren't tracked in this case but the engine still
            // expects per-input-variable gradients in order. We only registered
            // input as the variable, so return just grad_input.
            return {grad_input};
        }
        return {grad_input, grad_weight, grad_bias};
    }

    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override {
        // Variable-level BN backward so create_graph=true produces a real
        // 2nd-order graph. For world_size == 1 this is identical math to
        // BatchNorm2d::backward_with_variables. For world_size > 1 the
        // grad-side reductions would need to be all-reduced across workers;
        // expressing an all-reduce inside the autograd graph is not yet
        // supported, so we fall through to the tensor-level backward in
        // that case (graph disconnects, flagged via is_higher_order_stub()
        // below for the distributed path).
        if (world_size_ > 1 || !all_reduce_fn_) {
            std::vector<Tensor> tensor_grads;
            tensor_grads.reserve(grad_outputs.size());
            for (auto& v : grad_outputs) tensor_grads.push_back(v.tensor());
            auto results = backward(std::move(tensor_grads));
            std::vector<Variable> var_results;
            var_results.reserve(results.size());
            const bool rg = !grad_outputs.empty() && grad_outputs[0].requires_grad();
            for (auto& t : results) var_results.emplace_back(t, rg);
            return var_results;
        }

        // --- Single-process path: same shape handling as BatchNorm2d bwv ---
        auto& grad_out = grad_outputs[0];
        auto saved = saved_tensors();
        Variable input_var(saved[0], false);
        Variable mean_var(saved[1], false);
        Variable invstd_var(saved[2], false);
        Variable weight_var(saved[3], false);

        auto shape = input_var.shape();
        int64_t N = shape[0];
        int64_t C = shape[1];
        int64_t H = shape[2];
        int64_t W = shape[3];
        int64_t spatial = H * W;
        int64_t batch = N * spatial;

        auto mean_bc = unsqueeze(unsqueeze(unsqueeze(mean_var, 0), 2), 3);
        auto invstd_bc = unsqueeze(unsqueeze(unsqueeze(invstd_var, 0), 2), 3);
        auto x_hat = (input_var - mean_bc) * invstd_bc;

        Variable weight_bc = affine_
            ? unsqueeze(unsqueeze(unsqueeze(weight_var, 0), 2), 3)
            : Variable(ones({1, C, 1, 1}, input_var.dtype(), input_var.device()), false);

        auto grad_x_hat = grad_out * weight_bc;
        auto grad_x_hat_r = reshape(grad_x_hat, {N, C, spatial});
        auto x_hat_r = reshape(x_hat, {N, C, spatial});

        auto sum_g = sum(sum(grad_x_hat_r, 0, true), 2, true);
        auto mean_gxh = sum_g / static_cast<float>(batch);
        auto sum_gxh = sum(sum(grad_x_hat_r * x_hat_r, 0, true), 2, true);
        auto mean_gxh_xh = sum_gxh / static_cast<float>(batch);

        auto invstd_r = unsqueeze(unsqueeze(invstd_var, 0), -1);
        auto grad_input_r = (grad_x_hat_r - mean_gxh - x_hat_r * mean_gxh_xh) * invstd_r;
        auto grad_input = reshape(grad_input_r, {N, C, H, W});

        if (!affine_) return {grad_input};

        auto go_xhat = reshape(grad_out * x_hat, {N, C, spatial});
        auto grad_gamma = sum(sum(go_xhat, 0, false), 1, false);
        auto go_r = reshape(grad_out, {N, C, spatial});
        auto grad_beta = sum(sum(go_r, 0, false), 1, false);
        return {grad_input, grad_gamma, grad_beta};
    }

    auto supports_higher_order() const -> bool override { return true; }
    // Single-process path has a proper Variable-level 2nd-derivative graph;
    // the distributed (world_size > 1) path still disconnects.
    auto is_higher_order_stub() const -> bool override { return world_size_ > 1; }

private:
    bool affine_;
    int world_size_;
    AllReduceFn all_reduce_fn_;
    int64_t global_count_;
};

} // namespace

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
    int64_t global_count = 0;

    if (is_training()) {
        int64_t local_count = N * H * W;

        // audit-2026-05-03 Phase 10 — only cast UP, never DOWN. Casting
        // Float64 to Float32 for "numerical stability" actually destroyed
        // ~30 mantissa bits and broke Float64 SyncBatchNorm gradcheck on
        // every backend. Float16/BFloat16 still upcast to Float32.
        DType compute_dtype = (x.dtype() == DType::Float64) ? DType::Float64 : DType::Float32;
        auto x_compute = (x.dtype() != compute_dtype) ? x.to(compute_dtype) : x;

        // 5th-audit A2: two-pass variance, NOT `E[X^2] - E[X]^2`.
        //
        // Pre-fix used `var = global_sum_sq*inv_count - batch_mean*batch_mean`,
        // which is mathematically equivalent but numerically unstable
        // (catastrophic cancellation) when `mean^2` is close to `E[X^2]`.
        // The CPU LayerNorm-forward path was already fixed
        // (commit 2ee72b5b); SyncBatchNorm has the same bug across the
        // collective.
        //
        // The cost: ONE extra all-reduce. We accept this for correctness —
        // ZeRO-Stage-3 / FSDP runs at large world sizes already burn most of
        // their time in collectives, so the marginal cost is bounded, while
        // silent variance corruption surfaces as 4-7% gradcheck drift that
        // gets "fixed" by relaxing tolerance.

        // ---- Pass 1: per-channel mean ----------------------------------
        auto local_sum = sum(sum(sum(x_compute, 3, false), 2, false), 0, false);
        auto count_tensor = full({1}, static_cast<double>(local_count), compute_dtype).to(x.device());
        auto packed1 = cat({local_sum, count_tensor}, 0);

        if (world_size_ > 1 && all_reduce_fn_) {
            all_reduce_fn_(packed1);
        }

        auto global_sum = packed1.slice(0, 0, C);
        auto global_count_t = packed1.slice(0, C, C + 1);
        double global_count_d;
        if (compute_dtype == DType::Float64) {
            global_count_d = global_count_t.to(Device::cpu()).data<double>()[0];
        } else {
            global_count_d = static_cast<double>(global_count_t.to(Device::cpu()).data<float>()[0]);
        }
        global_count = static_cast<int64_t>(global_count_d);
        auto inv_count_t = full({1}, 1.0 / global_count_d, compute_dtype).to(x.device());
        batch_mean = global_sum * inv_count_t;

        // ---- Pass 2: per-channel sum of squared deviations -------------
        // Compute (x - global_mean)^2 against the freshly-reduced mean.
        auto mean_b = batch_mean.reshape({1, C, 1, 1});
        auto centred = x_compute - mean_b;
        auto dev_sq = centred * centred;
        auto local_sum_sq_dev = sum(sum(sum(dev_sq, 3, false), 2, false), 0, false);

        if (world_size_ > 1 && all_reduce_fn_) {
            all_reduce_fn_(local_sum_sq_dev);
        }

        batch_var = local_sum_sq_dev * inv_count_t;

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
        global_count = N * H * W;  // not used for backward in eval, but set for completeness
    }

    // Normalize: y = (x - mean) / sqrt(var + eps) * weight + bias
    auto mean_4d = batch_mean.reshape({1, C, 1, 1});
    auto var_4d = batch_var.reshape({1, C, 1, 1});

    auto inv_std_4d = reciprocal(sqrt(var_4d + static_cast<float>(eps_)));
    auto normalized = (x - mean_4d) * inv_std_4d;

    if (affine_) {
        auto w = weight_.tensor().reshape({1, C, 1, 1});
        auto b = bias_.tensor().reshape({1, C, 1, 1});
        normalized = normalized * w + b;
    }

    // Set up autograd if needed.
    bool requires_grad = input.requires_grad();
    if (affine_) {
        requires_grad = requires_grad || weight_.requires_grad() || bias_.requires_grad();
    }

    if (is_grad_enabled() && requires_grad && is_training()) {
        // 1D invstd per channel for backward
        Tensor invstd_1d = reciprocal(sqrt(batch_var + static_cast<float>(eps_)));

        // Weight to use in backward (ones if not affine)
        Tensor weight_for_bwd = affine_
            ? weight_.tensor()
            : ones({C}, x.dtype(), x.device());

        std::vector<Tensor> tensors_to_save = {
            input.tensor().contiguous(),
            batch_mean.contiguous(),
            invstd_1d.contiguous(),
            weight_for_bwd.contiguous(),
        };

        auto grad_fn = std::make_shared<SyncBatchNormBackward>(
            affine_, world_size_, all_reduce_fn_, global_count,
            std::move(tensors_to_save));

        Variable result(normalized, true);
        result.set_grad_fn(grad_fn);

        std::vector<Variable> input_vars = {input};
        if (affine_) {
            input_vars.push_back(weight_);
            input_vars.push_back(bias_);
        }
        grad_fn->set_input_variables(input_vars);

        // Continue backward chain through input's grad_fn if any.
        std::vector<std::shared_ptr<Function>> next_funcs;
        if (input.grad_fn()) {
            next_funcs.push_back(input.grad_fn());
        }
        grad_fn->set_next_functions(next_funcs);

        return result;
    }

    return Variable(normalized, false);
}

} // namespace tenzor::nn
