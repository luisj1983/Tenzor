#include "tenzor/nn/layers/batchnorm.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/reduction.hpp"
#include "tenzor/autograd/function.hpp"
#include "tenzor/autograd/ops.hpp"
#include "tenzor/backend/fast_dispatch.hpp"
#include "tenzor/ops/op_id.hpp"
#include "tenzor/backend/op_attributes.hpp"
#include "tenzor/utils/autograd_wrap.hpp"
#include <cmath>

// SIMD headers for optimized BatchNorm
#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

// OpenMP for parallel execution
#ifdef _OPENMP
#include <omp.h>
#include <thread>
#endif

namespace tenzor::nn {

namespace {

// audit-2026-06 Fix #7: guard multi-output backend dispatch results before
// indexing them. A backend kernel that returns fewer tensors than the
// {output, mean, var, ...} contract would otherwise cause an out-of-bounds
// std::vector::operator[] read (UB) instead of a clear error. Mirrors the
// check_norm_outputs helper in normalization.cpp.
inline auto check_norm_outputs(const std::vector<::tenzor::Tensor>& results,
                               const char* op_name,
                               std::size_t min_required) -> void {
    if (results.size() < min_required) {
        throw std::runtime_error(
            std::string(op_name) +
            ": backend kernel returned " +
            std::to_string(results.size()) +
            " tensors; the contract requires at least " +
            std::to_string(min_required) +
            ". Fix the backend kernel registration.");
    }
}

}  // namespace

// BatchNorm2d autograd function
class BatchNorm2dBackward : public Function {
public:
    BatchNorm2dBackward(bool affine, double eps, bool training,
                        std::vector<Tensor> tensors_to_save)
        : affine_(affine), eps_(eps), training_(training) {
        // Save tensors in constructor (protected member access is allowed here)
        save_for_backward(std::move(tensors_to_save));
    }

    auto forward([[maybe_unused]] std::vector<Variable> inputs) -> std::vector<Variable> override {
        // Not used - forward is handled by BatchNorm2d::forward
        throw std::runtime_error("BatchNorm2dBackward::forward should not be called");
    }

    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override;

    auto supports_higher_order() const -> bool override { return true; }
    auto is_higher_order_stub() const -> bool override { return false; }

    // A.4 multi-op JVP walker hooks. The multi-output JVP rule for
    // BatchNorm2dForwardAffine expects 5 primals (x, mean, var, gamma,
    // beta). BatchNorm2dBackward currently saves {x, mean, invstd, gamma}
    // and only carries (input, weight, bias) in input_variables, so the
    // walker cannot rebuild `var` (only invstd is saved) without
    // re-squaring. Repack here so the walker can dispatch the multi-output
    // rule: synthesise var = 1/invstd^2 - eps, and a zero-beta with zero
    // tangents for the non-x slots (gamma/beta are treated as constants
    // w.r.t. the JVP seed, so dgamma=dbeta=dvar=dmean=0).
    auto op_id() const -> OpId override { return OpId::BatchNorm2dForwardAffine; }

    auto saved_attributes() const -> OpAttributes override {
        OpAttributes attrs;
        attrs.set(AttrKey::Eps, eps_);
        return attrs;
    }

    auto jvp_pack_inputs_for_walker(
            const std::vector<Tensor>& walker_primals,
            const std::vector<Tensor>& walker_tangents) const
        -> std::optional<std::pair<std::vector<Tensor>, std::vector<Tensor>>> override {
        if (walker_primals.empty()) return std::nullopt;
        // Saved layout (per BatchNorm2d::forward): {input, mean, invstd, weight}.
        if (num_saved_tensors() < 4) return std::nullopt;
        const auto& sav = const_cast<BatchNorm2dBackward*>(this)->saved_tensors();
        const Tensor& mean   = sav[1];
        const Tensor& invstd = sav[2];
        const Tensor& gamma  = sav[3];

        // var = 1/invstd^2 - eps  (matches `invstd = 1/sqrt(var + eps)`)
        auto invstd_sq = tenzor::mul(invstd, invstd);
        auto one = tenzor::ones_like(invstd_sq);
        auto var = tenzor::sub(tenzor::div(one, invstd_sq), eps_);

        std::vector<int64_t> g_shape(gamma.shape().begin(), gamma.shape().end());
        Tensor beta = tenzor::zeros(g_shape, gamma.dtype(), gamma.device());

        auto zero_like = [](const Tensor& t) {
            std::vector<int64_t> sh(t.shape().begin(), t.shape().end());
            return tenzor::zeros(sh, t.dtype(), t.device());
        };

        std::vector<Tensor> primals  = { walker_primals[0],  mean,            var,            gamma,            beta };
        std::vector<Tensor> tangents = { walker_tangents[0], zero_like(mean), zero_like(var), zero_like(gamma), zero_like(beta) };
        return std::make_pair(std::move(primals), std::move(tangents));
    }

    // BatchNorm2dBackward saves {x, mean, invstd, gamma}. The forward
    // BatchNorm2dForwardAffine produces {y, mean, rstd}; saved[1] (mean) →
    // out 1, saved[2] (invstd ≡ rstd) → out 2. saved[0] (x) and saved[3]
    // (gamma) are inputs — leave at default 0.
    auto jvp_saved_tensor_to_output_idx(std::size_t saved_idx) const
        -> std::size_t override {
        switch (saved_idx) {
            case 1: return 1;  // mean
            case 2: return 2;  // rstd (saved as invstd)
            default: return 0;
        }
    }

private:
    bool affine_;
    double eps_;
    bool training_;
};

auto BatchNorm2dBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    // Ensure grad_output is contiguous
    auto grad_output = grad_outputs[0].contiguous();
    auto saved = saved_tensors();
    // Ensure all saved tensors are contiguous
    auto input = saved[0].contiguous();
    auto mean = saved[1].contiguous();
    auto invstd = saved[2].contiguous();  // saved_inv_var from cuDNN or computed invstd
    auto weight = saved[3].contiguous();

    // grad_output: [N, C, H, W]
    auto shape = input.shape();
    int64_t N = shape[0];
    int64_t C = shape[1];
    int64_t H = shape[2];
    int64_t W = shape[3];
    int64_t spatial_size = H * W;
    int64_t batch_size = N * spatial_size;

    // audit-2026-05-03 bug #3 — eval-mode backward.
    if (!training_) {
        // Reshape per-channel constants for broadcasting: (C,) → (1, C, 1, 1)
        auto reshape_chan = [&](const Tensor& t) {
            return t.reshape({1, C, 1, 1});
        };
        auto inv_b = reshape_chan(invstd);
        auto mean_b = reshape_chan(mean);
        auto weight_b = reshape_chan(weight);

        auto normalized = (input - mean_b) * inv_b;
        auto grad_input = grad_output * weight_b * inv_b;

        // Per-channel reductions across N, H, W (dims 0, 2, 3).
        auto grad_weight_full = grad_output * normalized;
        auto grad_weight = ::tenzor::sum(::tenzor::sum(::tenzor::sum(
            grad_weight_full, /*dim=*/3, /*keepdim=*/false),
            /*dim=*/2, /*keepdim=*/false),
            /*dim=*/0, /*keepdim=*/false);
        auto grad_bias = ::tenzor::sum(::tenzor::sum(::tenzor::sum(
            grad_output, /*dim=*/3, /*keepdim=*/false),
            /*dim=*/2, /*keepdim=*/false),
            /*dim=*/0, /*keepdim=*/false);

        return {grad_input, grad_weight, grad_bias};
    }

    // FAST GPU PATH
    if (input.device().type != Device::Type::CPU &&
        is_op_supported(OpId::BatchNorm2dBackward, input.device().type) &&
        (input.dtype() == DType::Float32 || input.dtype() == DType::Float16 ||
         input.dtype() == DType::BFloat16 || input.dtype() == DType::Float64)) {
        // Half-precision batch-norm backward must run with F32 statistics. CUDA
        // (cuDNN) accepts half I/O with F32 stats directly, but the ROCm / Vulkan
        // / OneAPI kernels require dtype-consistent buffers (verified: dispatching
        // half here throws a dtype-mismatch on ROCm). Promote to F32 on-device so
        // the accumulation is correct on every backend, then narrow the result
        // back — this entire sequence executes on the GPU (no CPU involvement).
        DType fast_orig_dtype = input.dtype();
        bool fast_upcast = (fast_orig_dtype == DType::Float16 || fast_orig_dtype == DType::BFloat16);
        if (fast_upcast) {
            grad_output = grad_output.to(DType::Float32);
            input = input.to(DType::Float32);
            weight = weight.to(DType::Float32);
            mean = mean.to(DType::Float32);
            invstd = invstd.to(DType::Float32);
        }

        OpAttributes backward_attrs;
        backward_attrs.set(AttrKey::Eps, static_cast<float>(eps_));

        std::vector<Tensor> backward_inputs = {grad_output, input, weight, mean, invstd};
        std::vector<Tensor> backward_results = dispatch(OpId::BatchNorm2dBackward, backward_inputs, backward_attrs);

        if (fast_upcast) {
            return {backward_results[0].to(fast_orig_dtype),
                    backward_results[1].to(fast_orig_dtype),
                    backward_results[2].to(fast_orig_dtype)};
        }
        return {backward_results[0], backward_results[1], backward_results[2]};
    }

    // FALLBACK: tensor ops
    DType orig_dtype = input.dtype();
    bool needs_upcast = (orig_dtype == DType::Float16 || orig_dtype == DType::BFloat16);
    if (needs_upcast) {
        grad_output = grad_output.to(DType::Float32);
        input = input.to(DType::Float32);
        mean = mean.to(DType::Float32);
        invstd = invstd.to(DType::Float32);
        weight = weight.to(DType::Float32);
    }

    auto mean_broadcast = mean.unsqueeze(0).unsqueeze(2).unsqueeze(3).contiguous();
    auto invstd_broadcast = invstd.unsqueeze(0).unsqueeze(2).unsqueeze(3).contiguous();
    auto normalized = ((input - mean_broadcast) * invstd_broadcast).contiguous();

    auto grad_weight = sum(sum((grad_output * normalized)
        .reshape({N, C, spatial_size}).contiguous(), 0, false), 1, false);

    auto grad_bias = sum(sum(grad_output
        .reshape({N, C, spatial_size}).contiguous(), 0, false), 1, false);

    auto weight_broadcast = weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).contiguous();
    auto grad_normalized = (grad_output * weight_broadcast).contiguous();

    auto grad_input_normalized = grad_normalized.reshape({N, C, spatial_size}).contiguous();
    auto normalized_reshaped = normalized.reshape({N, C, spatial_size}).contiguous();

    auto sum_grad = sum(sum(grad_input_normalized, 0, true), 2, true).contiguous();
    auto sum_grad_x_norm = sum(sum((grad_input_normalized * normalized_reshaped),
                            0, true), 2, true).contiguous();

    auto invstd_expanded = invstd.unsqueeze(0).unsqueeze(-1).contiguous();

    auto term1 = (sum_grad / static_cast<float>(batch_size)).contiguous();
    auto term2 = (normalized_reshaped * sum_grad_x_norm / static_cast<float>(batch_size)).contiguous();
    auto grad_input = ((grad_input_normalized - term1 - term2) * invstd_expanded).contiguous();

    grad_input = grad_input.reshape({N, C, H, W}).contiguous();

    if (needs_upcast) {
        grad_input = grad_input.to(orig_dtype);
        grad_weight = grad_weight.to(orig_dtype);
        grad_bias = grad_bias.to(orig_dtype);
    }

    return {grad_input, grad_weight, grad_bias};
}

auto BatchNorm2dBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    auto& grad_out = grad_outputs[0];

    Variable input_var, mean_var, invstd_var, weight_var;
    if (has_saved_variables()) {
        const auto& sv = saved_variables();
        input_var = sv[0];
        mean_var = sv[1];
        invstd_var = sv[2];
        weight_var = sv[3];
    } else {
        auto saved = saved_tensors();
        input_var = Variable(saved[0], false);
        mean_var = Variable(saved[1], false);
        invstd_var = Variable(saved[2], false);
        weight_var = Variable(saved[3], false);
    }

    auto shape = input_var.shape();
    int64_t N = shape[0];
    int64_t C = shape[1];
    int64_t H = shape[2];
    int64_t W = shape[3];
    int64_t spatial_size = H * W;
    int64_t batch_size = N * spatial_size;

    auto mean_bc = unsqueeze(unsqueeze(unsqueeze(mean_var, 0), 2), 3);
    auto invstd_bc = unsqueeze(unsqueeze(unsqueeze(invstd_var, 0), 2), 3);

    auto x_hat = (input_var - mean_bc) * invstd_bc;

    auto weight_bc = unsqueeze(unsqueeze(unsqueeze(weight_var, 0), 2), 3);

    auto grad_x_hat = grad_out * weight_bc;

    // Eval mode: running stats are constants, so the input gradient flows
    // straight through the affine normalization with NO batch-mean/variance
    // correction terms. The training-mode formula (below) would wrongly subtract
    // them. This mirrors the first-order backward()'s training_ branch.
    auto grad_input = [&]() -> Variable {
        if (training_) {
            auto grad_x_hat_r = reshape(grad_x_hat, {N, C, spatial_size});
            auto x_hat_r = reshape(x_hat, {N, C, spatial_size});

            auto sum_grad = sum(sum(grad_x_hat_r, 0, true), 2, true);
            auto mean_gxh = sum_grad / static_cast<float>(batch_size);

            auto sum_grad_xhat = sum(sum(grad_x_hat_r * x_hat_r, 0, true), 2, true);
            auto mean_gxh_xh = sum_grad_xhat / static_cast<float>(batch_size);

            auto invstd_r = unsqueeze(unsqueeze(invstd_var, 0), -1);
            auto grad_input_r =
                (grad_x_hat_r - mean_gxh - x_hat_r * mean_gxh_xh) * invstd_r;
            return reshape(grad_input_r, {N, C, H, W});
        }
        return grad_x_hat * invstd_bc;
    }();

    auto go_xhat = reshape(grad_out * x_hat, {N, C, spatial_size});
    auto grad_gamma = sum(sum(go_xhat, 0, false), 1, false);

    auto go_r = reshape(grad_out, {N, C, spatial_size});
    auto grad_beta = sum(sum(go_r, 0, false), 1, false);

    return {grad_input, grad_gamma, grad_beta};
}

namespace internal {
auto make_batch_norm2d_backward(bool affine, double eps, bool training,
                                std::vector<Tensor> tensors_to_save)
    -> std::shared_ptr<::tenzor::Function> {
    return std::make_shared<BatchNorm2dBackward>(affine, eps, training,
                                                 std::move(tensors_to_save));
}
}  // namespace internal

BatchNorm2d::BatchNorm2d(int64_t num_features, double eps, double momentum,
                        bool affine, bool track_running_stats)
    : num_features_(num_features), eps_(eps), momentum_(momentum),
      affine_(affine), track_running_stats_(track_running_stats) {

    // P.1: Reject the incoherent training=true / track_running_stats=false
    // combination at construction. Modules start in train mode (training_=true),
    // so the only way the user *intends* training-without-running-stats is by
    // explicitly disabling track_running_stats — which BN cannot honour because
    // the eval path falls back to running stats. PyTorch raises here.
    if (training_ && !track_running_stats) {
        throw std::invalid_argument(
            "BatchNorm: training=true with track_running_stats=false is "
            "incompatible — running stats are required during training");
    }

    if (affine) {
        weight_ = Variable(ones({num_features}), true);
        bias_ = Variable(zeros({num_features}), true);
        register_parameter("weight", weight_);
        register_parameter("bias", bias_);
        // Cache pointers to avoid hash map lookups in forward pass (~2-3ms savings)
        cached_weight_ = parameters_["weight"];
        cached_bias_ = parameters_["bias"];
    } else {
        weight_ = Variable(ones({num_features}), false);
        bias_ = Variable(zeros({num_features}), false);
    }

    if (track_running_stats) {
        running_mean_ = Variable(zeros({num_features}), false);
        running_var_ = Variable(ones({num_features}), false);
        num_batches_tracked_ = Variable(zeros({}, DType::Int64), false);
        register_buffer("running_mean", running_mean_);
        register_buffer("running_var", running_var_);
        register_buffer("num_batches_tracked", num_batches_tracked_);
    }

    reset_parameters();
}

auto BatchNorm2d::forward_impl(const Variable& input) -> Variable {
    // Input shape: [N, C, H, W]
    auto shape = input.shape();
    if (shape.size() != 4) {
        throw std::runtime_error("BatchNorm2d expects 4D input (got " +
                               std::to_string(shape.size()) + "D)");
    }

    int64_t N = shape[0];
    int64_t C = shape[1];
    int64_t H = shape[2];
    int64_t W = shape[3];
    int64_t spatial_size = H * W;

    if (C != num_features_) {
        throw std::runtime_error("Expected " + std::to_string(num_features_) +
                               " channels, got " + std::to_string(C));
    }

    // ============================================================================
    // FAST INFERENCE PATH: CPU Float32 eval mode with running stats
    // ============================================================================
    const bool needs_grad = is_grad_enabled() &&
        (input.requires_grad() || (affine_ && cached_weight_ && cached_weight_->requires_grad()));

    if (!needs_grad && !training_ && track_running_stats_ &&
        input.tensor().device().type == Device::Type::CPU && input.tensor().dtype() == DType::Float32) {

        const Tensor& input_tensor = input.tensor();
        const auto* input_data = input_tensor.data<float>();

        // Get running stats and affine parameters directly
        const Tensor& running_mean = buffers_["running_mean"]->tensor();
        const Tensor& running_var = buffers_["running_var"]->tensor();
        const auto* mean_data = running_mean.data<float>();
        const auto* var_data = running_var.data<float>();

        const float* gamma_data = nullptr;
        const float* beta_data = nullptr;
        if (affine_ && cached_weight_ && cached_bias_) {
            gamma_data = cached_weight_->tensor().data<float>();
            beta_data = cached_bias_->tensor().data<float>();
        }

        // Allocate output
        auto output = Tensor::empty_uninitialized(
            {N, C, H, W}, DType::Float32, Device::cpu());
        auto* output_data = output.data<float>();

        const float eps = static_cast<float>(eps_);

        // Use thread-local storage to avoid allocation per forward call
        // This gives significant speedup for repeated inference
        thread_local std::vector<float> tls_scale, tls_shift;
        if (static_cast<int64_t>(tls_scale.size()) < C) {
            tls_scale.resize(C);
            tls_shift.resize(C);
        }
        float* scale = tls_scale.data();
        float* shift = tls_shift.data();

        // Precompute scale and shift for each channel
        for (int64_t c = 0; c < C; c++) {
            float inv_std = 1.0f / std::sqrt(var_data[c] + eps);
            if (affine_) {
                scale[c] = gamma_data[c] * inv_std;
                shift[c] = beta_data[c] - mean_data[c] * scale[c];
            } else {
                scale[c] = inv_std;
                shift[c] = -mean_data[c] * inv_std;
            }
        }

        // Apply normalization with SIMD
        for (int64_t n = 0; n < N; n++) {
            for (int64_t c = 0; c < C; c++) {
                const float s = scale[c];
                const float b = shift[c];
                const int64_t idx = (n * C + c) * spatial_size;

                #if defined(__AVX512F__)
                __m512 v_scale = _mm512_set1_ps(s);
                __m512 v_shift = _mm512_set1_ps(b);
                int64_t i = 0;
                for (; i + 16 <= spatial_size; i += 16) {
                    __m512 v_in = _mm512_loadu_ps(input_data + idx + i);
                    __m512 v_out = _mm512_fmadd_ps(v_in, v_scale, v_shift);
                    _mm512_storeu_ps(output_data + idx + i, v_out);
                }
                for (; i < spatial_size; i++) {
                    output_data[idx + i] = input_data[idx + i] * s + b;
                }
                #else
                for (int64_t i = 0; i < spatial_size; i++) {
                    output_data[idx + i] = input_data[idx + i] * s + b;
                }
                #endif
            }
        }

        return Variable(output, false);
    }

    // ============================================================================
    // STANDARD PATH: Full dispatch with autograd support
    // ============================================================================

    // Validate to prevent division by zero
    int64_t batch_size = N * spatial_size;
    if (training_ && batch_size == 0) {
        throw std::runtime_error("BatchNorm2d: Cannot compute statistics for empty batch (N * H * W = 0)");
    }

    // Track original device for final output
    Device original_device = input.tensor().device();
    bool use_gpu = (original_device.type == Device::Type::CUDA);

    // Keep data on original device throughout (no CPU fallbacks)
    Tensor input_work = input.tensor();

    // FP16/BF16 forward upcast: compute in FP32 to prevent overflow/underflow
    // in mean/variance reductions. Matches cuDNN behavior and the backward pass pattern.
    DType orig_dtype = input_work.dtype();
    bool needs_upcast = (orig_dtype == DType::Float16 || orig_dtype == DType::BFloat16);
    if (needs_upcast) {
        input_work = input_work.to(DType::Float32);
    }

    Tensor batch_mean, batch_var;

    if (training_) {
        // cuDNN's batch-norm primitives clamp epsilon up to CUDNN_BN_MIN_EPSILON
        // (1e-5); a smaller value would either be rejected or silently raised, so
        // the fused cuDNN training path cannot honor a user eps < 1e-5. ROCm/CPU
        // and the hand-written CUDA kernels accept any eps > 0. To keep CUDA
        // matching every other backend for small eps, skip the fused cuDNN path
        // when eps < 1e-5 and take the standard training path below, which
        // normalizes via the native CUDA BatchNorm2dForwardAffine kernel and
        // computes invstd = 1/sqrt(var + eps) with the exact user eps.
        constexpr double kCudnnBnMinEpsilon = 1e-5;
        // Check if we can use fused cuDNN training path (CUDA + affine + track_running_stats)
        bool use_fused_training = use_gpu && affine_ && track_running_stats_ &&
                                  cached_weight_ && cached_bias_ &&
                                  eps_ >= kCudnnBnMinEpsilon;

        if (use_fused_training) {
            // ================================================================
            // FUSED CUDNN PATH: Single kernel for mean/var + normalize + update
            // ================================================================
            auto& rm_var_ptr = buffers_["running_mean"];
            auto& rv_var_ptr = buffers_["running_var"];

            Tensor running_mean = rm_var_ptr->tensor();
            Tensor running_var = rv_var_ptr->tensor();
            if (running_mean.device() != original_device) {
                running_mean = running_mean.to(original_device);
            }
            if (running_var.device() != original_device) {
                running_var = running_var.to(original_device);
            }
            // Convert running stats to input dtype for kernel compatibility
            DType input_dtype = input_work.dtype();
            if (running_mean.dtype() != input_dtype) {
                running_mean = running_mean.to(input_dtype);
            }
            if (running_var.dtype() != input_dtype) {
                running_var = running_var.to(input_dtype);
            }

            Tensor weight = cached_weight_->tensor();
            Tensor bias = cached_bias_->tensor();
            if (weight.device() != original_device) {
                weight = weight.to(original_device);
            }
            if (bias.device() != original_device) {
                bias = bias.to(original_device);
            }
            // Convert weight/bias to input dtype for kernel compatibility
            if (weight.dtype() != input_dtype) {
                weight = weight.to(input_dtype);
            }
            if (bias.dtype() != input_dtype) {
                bias = bias.to(input_dtype);
            }

            OpAttributes fused_attrs;
            fused_attrs.set(AttrKey::Eps, static_cast<float>(eps_));
            fused_attrs.set(AttrKey::Momentum, static_cast<float>(momentum_));

            std::vector<Tensor> fused_inputs = {input_work, running_mean, running_var, weight, bias};
            std::vector<Tensor> fused_results = dispatch(OpId::BatchNorm2dFusedTraining, fused_inputs, fused_attrs);
            // Reads indices [0..4] below ({output, running_mean, running_var,
            // saved_mean, saved_inv_var}); guard before indexing.
            check_norm_outputs(fused_results, "BatchNorm2dFusedTraining", /*min_required=*/5);

            batch_mean = fused_results[3];  // saved_mean for backward
            batch_var = fused_results[4];   // saved_inv_var for backward

            // The cuDNN fused kernel computes the running-stats update with the
            // unbiased (Bessel-corrected) variance internally, dividing by
            // (m - 1). For a single-element batch (N*H*W < 2) that divisor is 0,
            // so the returned running_var is +inf/NaN and would permanently
            // poison the buffer. The standard (non-fused) path skips the update
            // in that degenerate case; mirror it here so all backends agree.
            int64_t running_stats_batch_size = N * spatial_size;
            if (running_stats_batch_size >= 2) {
                // Update running stats from cuDNN (store back as Float32 for storage efficiency)
                rm_var_ptr->tensor() = fused_results[1].to(DType::Float32);
                rv_var_ptr->tensor() = fused_results[2].to(DType::Float32);
                // Increment num_batches_tracked on the buffer's native device —
                // host-staging a single int64 each forward used to round-trip
                // through CPU. Adding via tenzor::add stays on-device.
                auto& nbt = buffers_["num_batches_tracked"]->tensor();
                if (nbt.device().type == Device::Type::CPU) {
                    nbt.data<int64_t>()[0]++;
                } else {
                    Tensor one = ::tenzor::full({1}, 1.0, nbt.dtype(), nbt.device());
                    nbt = ::tenzor::add(nbt, one);
                }
            }

            // Return output directly (autograd handled below if needed)
            Tensor output = fused_results[0];

            // Downcast output back to original dtype if we upcasted
            if (needs_upcast) {
                output = output.to(orig_dtype);
            }

            // Handle autograd for fused path
            bool requires_grad = input.requires_grad();
            if (affine_ && cached_weight_) {
                requires_grad = requires_grad || cached_weight_->requires_grad();
            }
            if (is_grad_enabled() && requires_grad) {
                auto result = Variable(output, true);
                // Note: For fused path, batch_var is actually saved_inv_var from cuDNN
                std::vector<Tensor> tensors_to_save = {
                    input.tensor().contiguous(),
                    batch_mean.contiguous(),
                    batch_var.contiguous(),  // This is inv_std from cuDNN
                    weight.contiguous()
                };
                auto grad_fn = std::make_shared<BatchNorm2dBackward>(
                    affine_, eps_, training_, std::move(tensors_to_save)
                );

                // Save Variables for higher-order gradients when create_graph is active
                if (is_creating_graph()) {
                    grad_fn->save_variables_for_backward({
                        input,
                        Variable(batch_mean.contiguous(), false),
                        Variable(batch_var.contiguous(), false),  // inv_std from cuDNN
                        Variable(weight.contiguous(), true)
                    });
                }

                result.set_grad_fn(grad_fn);

                // Track input variables for gradient accumulation
                // MUST include all inputs to maintain 1:1 index correspondence with gradients
                // The engine correctly skips variables that don't require grad
                std::vector<Variable> input_vars = {input};
                if (affine_ && cached_weight_ && cached_bias_) {
                    input_vars.push_back(*cached_weight_);
                    input_vars.push_back(*cached_bias_);
                }
                grad_fn->set_input_variables(input_vars);

                // Connect to input's grad_fn to continue the backward chain
                std::vector<std::shared_ptr<Function>> next_funcs;
                if (input.grad_fn()) {
                    next_funcs.push_back(input.grad_fn());
                }
                grad_fn->set_next_functions(next_funcs);

                return result;
            }
            return Variable(output, false);
        }

        // ================================================================
        // STANDARD TRAINING PATH: Separate kernels for backward compat
        // ================================================================
        // Training mode: compute batch statistics using backend dispatch
        OpAttributes mean_var_attrs;
        std::vector<Tensor> mean_var_inputs = {input_work};
        std::vector<Tensor> mean_var_results = dispatch(OpId::BatchNorm2dMeanVar, mean_var_inputs, mean_var_attrs);
        // Reads indices [0] and [1] ({mean, var}); guard before indexing.
        check_norm_outputs(mean_var_results, "BatchNorm2dMeanVar", /*min_required=*/2);
        batch_mean = mean_var_results[0];
        batch_var = mean_var_results[1];

        // Update running statistics using backend kernel.
        // The Bessel correction below divides by (batch_size - 1); when
        // batch_size == 1 this is 0 and 1/0 = +inf would poison running_var
        // permanently. The unbiased variance is undefined for a single
        // sample, so skip the running-stats update in that degenerate case
        // (running stats stay unchanged), matching the guard PyTorch applies.
        // The current forward output is unaffected because it uses the biased
        // batch_var.
        int64_t running_stats_batch_size = N * spatial_size;
        if (track_running_stats_ && running_stats_batch_size >= 2) {
            // Use unbiased variance estimate for running statistics
            int64_t batch_size = running_stats_batch_size;
            auto unbiased_var = batch_var * (static_cast<float>(batch_size) /
                                            static_cast<float>(batch_size - 1));

            // Get running stats and transfer to input device if needed
            auto& rm_var_ptr = buffers_["running_mean"];
            auto& rv_var_ptr = buffers_["running_var"];

            Tensor running_mean_on_device = rm_var_ptr->tensor();
            Tensor running_var_on_device = rv_var_ptr->tensor();
            if (running_mean_on_device.device() != original_device) {
                running_mean_on_device = running_mean_on_device.to(original_device);
            }
            if (running_var_on_device.device() != original_device) {
                running_var_on_device = running_var_on_device.to(original_device);
            }
            // Convert running stats to input dtype for kernel compatibility
            DType input_dtype = input_work.dtype();
            if (running_mean_on_device.dtype() != input_dtype) {
                running_mean_on_device = running_mean_on_device.to(input_dtype);
            }
            if (running_var_on_device.dtype() != input_dtype) {
                running_var_on_device = running_var_on_device.to(input_dtype);
            }

            // Use backend kernel for running stats update
            NewOpAttributes update_attrs;
            update_attrs.set(AttrKey::Momentum, static_cast<double>(momentum_));
            std::vector<Tensor> update_inputs = {running_mean_on_device, running_var_on_device, batch_mean, unbiased_var};
            std::vector<Tensor> updated_stats = dispatch(OpId::BatchNorm2dUpdateRunningStats, update_inputs, update_attrs);

            // Store updated stats back in Float32 for storage efficiency
            rm_var_ptr->tensor() = updated_stats[0].to(DType::Float32);
            rv_var_ptr->tensor() = updated_stats[1].to(DType::Float32);

            // Increment on the buffer's native device (avoids host roundtrip).
            auto& nbt = buffers_["num_batches_tracked"]->tensor();
            if (nbt.device().type == Device::Type::CPU) {
                nbt.data<int64_t>()[0]++;
            } else {
                Tensor one = ::tenzor::full({1}, 1.0, nbt.dtype(), nbt.device());
                nbt = ::tenzor::add(nbt, one);
            }
        }
    } else {
        // Inference mode: use running statistics (transfer to input device if needed)
        if (track_running_stats_) {
            batch_mean = buffers_["running_mean"]->tensor();
            batch_var = buffers_["running_var"]->tensor();
            // Transfer to input device if they're on different devices
            if (batch_mean.device() != original_device) {
                batch_mean = batch_mean.to(original_device);
            }
            if (batch_var.device() != original_device) {
                batch_var = batch_var.to(original_device);
            }
            // Convert to input dtype for kernel compatibility
            DType input_dtype = input_work.dtype();
            if (batch_mean.dtype() != input_dtype) {
                batch_mean = batch_mean.to(input_dtype);
            }
            if (batch_var.dtype() != input_dtype) {
                batch_var = batch_var.to(input_dtype);
            }
        } else {
            throw std::runtime_error("BatchNorm2d in eval mode requires track_running_stats=true");
        }
    }

    // Normalize using backend kernel
    Tensor output;
    OpAttributes forward_attrs;
    forward_attrs.set(AttrKey::Eps, static_cast<float>(eps_));

    if (affine_ && cached_weight_ && cached_bias_) {
        // Use affine forward kernel: output = gamma * (x - mean) / sqrt(var + eps) + beta
        // Transfer weight and bias to input device if needed
        Tensor weight_on_device = cached_weight_->tensor();
        Tensor bias_on_device = cached_bias_->tensor();
        if (weight_on_device.device() != original_device) {
            weight_on_device = weight_on_device.to(original_device);
        }
        if (bias_on_device.device() != original_device) {
            bias_on_device = bias_on_device.to(original_device);
        }
        // Convert weight and bias to input dtype for kernel compatibility
        DType input_dtype = input_work.dtype();
        if (weight_on_device.dtype() != input_dtype) {
            weight_on_device = weight_on_device.to(input_dtype);
        }
        if (bias_on_device.dtype() != input_dtype) {
            bias_on_device = bias_on_device.to(input_dtype);
        }
        std::vector<Tensor> forward_inputs = {input_work, batch_mean, batch_var, weight_on_device, bias_on_device};
        std::vector<Tensor> forward_results = dispatch(OpId::BatchNorm2dForwardAffine, forward_inputs, forward_attrs);
        output = forward_results[0];
    } else {
        // Use non-affine forward kernel: output = (x - mean) / sqrt(var + eps)
        std::vector<Tensor> forward_inputs = {input_work, batch_mean, batch_var};
        std::vector<Tensor> forward_results = dispatch(OpId::BatchNorm2dForward, forward_inputs, forward_attrs);
        output = forward_results[0];
    }

    // Downcast output back to original dtype if we upcasted
    if (needs_upcast) {
        output = output.to(orig_dtype);
    }

    // Set up autograd if needed - check is_grad_enabled() first for fast inference path
    bool requires_grad = input.requires_grad();
    if (affine_ && cached_weight_) {
        requires_grad = requires_grad || cached_weight_->requires_grad();
    }
    if (is_grad_enabled() && requires_grad) {
        // Create result variable from output
        auto result = Variable(output, true);

        // Prepare tensors to save for backward (already on correct device, no transfers needed)
        // Compute invstd from variance for backward pass
        // Create epsilon tensor with same dtype as batch_var to avoid dtype mismatch
        auto eps_tensor = full({}, eps_, batch_var.dtype(), batch_var.device());
        auto invstd = pow(batch_var + eps_tensor, -0.5f);

        // Ensure contiguous for backward
        Tensor batch_mean_final = batch_mean.contiguous();
        Tensor invstd_final = invstd.contiguous();

        // Use cached weight pointer for efficiency, transfer to input device and dtype if needed
        Tensor weight_tensor;
        DType input_dtype = input.tensor().dtype();
        if (affine_ && cached_weight_) {
            weight_tensor = cached_weight_->tensor();
            if (weight_tensor.device() != original_device) {
                weight_tensor = weight_tensor.to(original_device);
            }
            if (weight_tensor.dtype() != input_dtype) {
                weight_tensor = weight_tensor.to(input_dtype);
            }
        } else {
            weight_tensor = ones({C}, input_dtype, original_device);
        }
        // Ensure all tensors are contiguous before saving
        std::vector<Tensor> tensors_to_save = {
            input.tensor().contiguous(),  // input (original device, made contiguous)
            batch_mean_final,             // mean (transferred to original device, already contiguous)
            invstd_final,                 // invstd (transferred to original device, already contiguous)
            weight_tensor.contiguous()    // weight (or ones on original device, made contiguous)
        };

        // Create backward function with saved tensors
        auto grad_fn = std::make_shared<BatchNorm2dBackward>(
            affine_, eps_, training_, std::move(tensors_to_save)
        );

        // Save Variables for higher-order gradients when create_graph is active
        if (is_creating_graph()) {
            Variable weight_variable;
            if (affine_ && cached_weight_) {
                weight_variable = *cached_weight_;
                // Ensure device/dtype match
                if (weight_variable.tensor().device() != original_device) {
                    tenzor::utils::wrap_preserving_grad(weight_variable, weight_variable.tensor().to(original_device));
                }
                if (weight_variable.tensor().dtype() != input_dtype) {
                    tenzor::utils::wrap_preserving_grad(weight_variable, weight_variable.tensor().to(input_dtype));
                }
            } else {
                weight_variable = Variable(ones({C}, input_dtype, original_device), false);
            }
            grad_fn->save_variables_for_backward({
                input,
                Variable(batch_mean_final, false),
                Variable(invstd_final, false),
                weight_variable
            });
        }

        result.set_grad_fn(grad_fn);

        // Track input variables for gradient accumulation
        // MUST include all inputs to maintain 1:1 index correspondence with gradients
        // The engine correctly skips variables that don't require grad
        std::vector<Variable> input_vars = {input};
        if (affine_ && cached_weight_ && cached_bias_) {
            input_vars.push_back(*cached_weight_);
            input_vars.push_back(*cached_bias_);
        }
        grad_fn->set_input_variables(input_vars);

        // CRITICAL FIX: Connect to input's grad_fn to continue the backward chain
        std::vector<std::shared_ptr<Function>> next_funcs;
        if (input.grad_fn()) {
            next_funcs.push_back(input.grad_fn());
        }
        grad_fn->set_next_functions(next_funcs);

        return result;
    } else {
        return Variable(output, false);
    }
}

auto BatchNorm2d::reset_parameters() -> void {
    // Weight initialized to 1, bias to 0 (already done in constructor)
    if (track_running_stats_) {
        // CRITICAL: Access from buffers_ map
        buffers_["running_mean"]->tensor().zero_();
        buffers_["running_var"]->tensor().fill_(1.0f);
        buffers_["num_batches_tracked"]->tensor().zero_();
    }
    // Refresh cached pointers in case parameters were re-registered
    if (affine_) {
        cached_weight_ = parameters_["weight"];
        cached_bias_ = parameters_["bias"];
    }
}

// ============================================================================
// BatchNorm1d Implementation
// ============================================================================

// BatchNorm1d autograd function
class BatchNorm1dBackward : public Function {
public:
    BatchNorm1dBackward(bool affine, double eps, bool training,
                        std::vector<Tensor> tensors_to_save)
        : affine_(affine), eps_(eps), training_(training) {
        save_for_backward(std::move(tensors_to_save));
    }

    auto forward([[maybe_unused]] std::vector<Variable> inputs) -> std::vector<Variable> override {
        throw std::runtime_error("BatchNorm1dBackward::forward should not be called");
    }

    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override {
        auto grad_output = grad_outputs[0].contiguous();
        auto saved = saved_tensors();
        auto input = saved[0].contiguous();
        auto mean = saved[1].contiguous();
        auto invstd = saved[2].contiguous();
        auto weight = saved[3].contiguous();

        auto shape = input.shape();
        int64_t N = shape[0];
        int64_t C = shape[1];
        int64_t L = shape.size() == 3 ? shape[2] : 1;
        int64_t batch_size = N * L;

        // Compute normalized input
        Tensor mean_broadcast, invstd_broadcast;
        if (shape.size() == 3) {
            mean_broadcast = mean.unsqueeze(0).unsqueeze(2).contiguous();
            invstd_broadcast = invstd.unsqueeze(0).unsqueeze(2).contiguous();
        } else {
            mean_broadcast = mean.unsqueeze(0).contiguous();
            invstd_broadcast = invstd.unsqueeze(0).contiguous();
        }

        auto normalized = ((input - mean_broadcast) * invstd_broadcast).contiguous();

        // Gradient with respect to weight: sum(grad_output * normalized, dim=[0,2])
        Tensor grad_weight;
        if (shape.size() == 3) {
            grad_weight = sum(sum((grad_output * normalized)
                .reshape({N, C, L}).contiguous(), 0, false), 1, false);
        } else {
            grad_weight = sum((grad_output * normalized).contiguous(), 0, false);
        }

        // Gradient with respect to bias: sum(grad_output, dim=[0,2])
        Tensor grad_bias;
        if (shape.size() == 3) {
            grad_bias = sum(sum(grad_output
                .reshape({N, C, L}).contiguous(), 0, false), 1, false);
        } else {
            grad_bias = sum(grad_output.contiguous(), 0, false);
        }

        // Gradient with respect to normalized input
        Tensor weight_broadcast;
        if (shape.size() == 3) {
            weight_broadcast = weight.unsqueeze(0).unsqueeze(2).contiguous();
        } else {
            weight_broadcast = weight.unsqueeze(0).contiguous();
        }
        auto grad_normalized = (grad_output * weight_broadcast).contiguous();

        // Gradient with respect to input
        Tensor grad_input_normalized;
        Tensor normalized_reshaped;
        if (shape.size() == 3) {
            grad_input_normalized = grad_normalized.reshape({N, C, L}).contiguous();
            normalized_reshaped = normalized.reshape({N, C, L}).contiguous();
        } else {
            grad_input_normalized = grad_normalized.contiguous();
            normalized_reshaped = normalized.contiguous();
        }

        Tensor sum_grad, sum_grad_x_norm;
        if (shape.size() == 3) {
            sum_grad = sum(sum(grad_input_normalized, 0, true), 2, true).contiguous();
            sum_grad_x_norm = sum(sum((grad_input_normalized * normalized_reshaped),
                                0, true), 2, true).contiguous();
        } else {
            sum_grad = sum(grad_input_normalized, 0, true).contiguous();
            sum_grad_x_norm = sum((grad_input_normalized * normalized_reshaped),
                                0, true).contiguous();
        }

        Tensor invstd_expanded;
        if (shape.size() == 3) {
            invstd_expanded = invstd.unsqueeze(0).unsqueeze(-1).contiguous();
        } else {
            invstd_expanded = invstd.unsqueeze(0).contiguous();
        }

        // audit-2026-05-03 bug #3 — eval-mode drops the term1/term2 corrections.
        Tensor grad_input;
        if (training_) {
            auto term1 = (sum_grad / static_cast<float>(batch_size)).contiguous();
            auto term2 = (normalized_reshaped * sum_grad_x_norm / static_cast<float>(batch_size)).contiguous();
            grad_input = ((grad_input_normalized - term1 - term2) * invstd_expanded).contiguous();
        } else {
            grad_input = (grad_input_normalized * invstd_expanded).contiguous();
        }

        if (shape.size() == 3) {
            grad_input = grad_input.reshape({N, C, L}).contiguous();
        }

        return {grad_input, grad_weight, grad_bias};
    }

    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override {
        // Higher-order gradient support for BatchNorm1d (adapted from BatchNorm2dBackward).
        auto& grad_out = grad_outputs[0];

        Variable input_var, mean_var, invstd_var, weight_var;
        if (has_saved_variables()) {
            const auto& sv = saved_variables();
            input_var = sv[0];
            mean_var = sv[1];
            invstd_var = sv[2];
            weight_var = sv[3];
        } else {
            auto saved = saved_tensors();
            input_var = Variable(saved[0], false);
            mean_var = Variable(saved[1], false);
            invstd_var = Variable(saved[2], false);
            weight_var = Variable(saved[3], false);
        }

        auto shape = input_var.shape();
        int64_t N = shape[0];
        int64_t C = shape[1];
        bool is_3d = shape.size() == 3;
        int64_t L = is_3d ? shape[2] : 1;
        int64_t batch_size = N * L;

        // Broadcast mean and invstd: [C] -> [1,C] or [1,C,1]
        Variable mean_bc, invstd_bc, weight_bc;
        if (is_3d) {
            mean_bc = unsqueeze(unsqueeze(mean_var, 0), 2);
            invstd_bc = unsqueeze(unsqueeze(invstd_var, 0), 2);
            weight_bc = unsqueeze(unsqueeze(weight_var, 0), 2);
        } else {
            mean_bc = unsqueeze(mean_var, 0);
            invstd_bc = unsqueeze(invstd_var, 0);
            weight_bc = unsqueeze(weight_var, 0);
        }

        auto x_hat = (input_var - mean_bc) * invstd_bc;
        auto grad_x_hat = grad_out * weight_bc;

        // Reduce over batch (and spatial if 3D)
        Variable grad_x_hat_r, x_hat_r;
        if (is_3d) {
            grad_x_hat_r = reshape(grad_x_hat, {N, C, L});
            x_hat_r = reshape(x_hat, {N, C, L});
        } else {
            grad_x_hat_r = grad_x_hat;
            x_hat_r = x_hat;
        }

        // Eval mode: running stats are constants, so the input gradient passes
        // straight through the affine normalization with NO batch-mean/variance
        // correction terms (mirrors the first-order backward()'s training_ branch).
        Variable grad_input;
        if (training_) {
            Variable sum_grad, sum_grad_xhat;
            if (is_3d) {
                sum_grad = sum(sum(grad_x_hat_r, 0, true), 2, true);
                sum_grad_xhat = sum(sum(grad_x_hat_r * x_hat_r, 0, true), 2, true);
            } else {
                sum_grad = sum(grad_x_hat_r, 0, true);
                sum_grad_xhat = sum(grad_x_hat_r * x_hat_r, 0, true);
            }
            auto mean_gxh = sum_grad / static_cast<float>(batch_size);
            auto mean_gxh_xh = sum_grad_xhat / static_cast<float>(batch_size);

            Variable invstd_r;
            if (is_3d) {
                invstd_r = unsqueeze(unsqueeze(invstd_var, 0), -1);
            } else {
                invstd_r = unsqueeze(invstd_var, 0);
            }

            auto grad_input_r = (grad_x_hat_r - mean_gxh - x_hat_r * mean_gxh_xh) * invstd_r;
            grad_input = is_3d ? reshape(grad_input_r, {N, C, L}) : grad_input_r;
        } else {
            grad_input = grad_x_hat * invstd_bc;
        }

        // grad_gamma and grad_beta
        Variable grad_gamma, grad_beta;
        if (is_3d) {
            auto go_xhat = reshape(grad_out * x_hat, {N, C, L});
            grad_gamma = sum(sum(go_xhat, 0, false), 1, false);
            auto go_r = reshape(grad_out, {N, C, L});
            grad_beta = sum(sum(go_r, 0, false), 1, false);
        } else {
            grad_gamma = sum(grad_out * x_hat, 0, false);
            grad_beta = sum(grad_out, 0, false);
        }

        return {grad_input, grad_gamma, grad_beta};
    }

    auto supports_higher_order() const -> bool override { return true; }
    auto is_higher_order_stub() const -> bool override { return false; }

private:
    bool affine_;
    double eps_;
    bool training_;
};

BatchNorm1d::BatchNorm1d(int64_t num_features, double eps, double momentum,
                        bool affine, bool track_running_stats)
    : num_features_(num_features), eps_(eps), momentum_(momentum),
      affine_(affine), track_running_stats_(track_running_stats) {

    // P.1: see BatchNorm2d ctor for rationale.
    if (training_ && !track_running_stats) {
        throw std::invalid_argument(
            "BatchNorm: training=true with track_running_stats=false is "
            "incompatible — running stats are required during training");
    }

    if (affine) {
        weight_ = Variable(ones({num_features}), true);
        bias_ = Variable(zeros({num_features}), true);
        register_parameter("weight", weight_);
        register_parameter("bias", bias_);
        // Cache pointers to avoid hash map lookups in forward pass (~2-3ms savings)
        cached_weight_ = parameters_["weight"];
        cached_bias_ = parameters_["bias"];
    } else {
        weight_ = Variable(ones({num_features}), false);
        bias_ = Variable(zeros({num_features}), false);
    }

    if (track_running_stats) {
        running_mean_ = Variable(zeros({num_features}), false);
        running_var_ = Variable(ones({num_features}), false);
        num_batches_tracked_ = Variable(zeros({}, DType::Int64), false);
        register_buffer("running_mean", running_mean_);
        register_buffer("running_var", running_var_);
        register_buffer("num_batches_tracked", num_batches_tracked_);
    }

    reset_parameters();
}

auto BatchNorm1d::forward_impl(const Variable& input) -> Variable {
    // Input shape: [N, C] or [N, C, L]
    auto shape = input.shape();
    if (shape.size() != 2 && shape.size() != 3) {
        throw std::runtime_error("BatchNorm1d expects 2D or 3D input (got " +
                               std::to_string(shape.size()) + "D)");
    }

    int64_t N = shape[0];
    int64_t C = shape[1];
    int64_t L = shape.size() == 3 ? shape[2] : 1;

    if (C != num_features_) {
        throw std::runtime_error("Expected " + std::to_string(num_features_) +
                               " features, got " + std::to_string(C));
    }

    int64_t batch_size = N * L;
    if (training_ && batch_size == 0) {
        throw std::runtime_error("BatchNorm1d: Cannot compute statistics for empty batch");
    }

    Device original_device = input.tensor().device();
    Tensor input_work = input.tensor();

    // FP16/BF16 forward upcast: compute in Float32 to prevent overflow /
    // underflow in mean / variance reductions. Without this, squaring many
    // Float16 values and summing quickly saturates Float16's ~6.5e4 range,
    // producing garbage stats on backends (notably Vulkan) whose reductions
    // don't internally widen. Mirrors the BN2d forward upcast path above.
    DType orig_dtype = input_work.dtype();
    const bool needs_upcast =
        (orig_dtype == DType::Float16 || orig_dtype == DType::BFloat16);
    if (needs_upcast) {
        input_work = input_work.to(DType::Float32);
    }

    Tensor batch_mean, batch_var;

    if (training_) {
        // Compute mean and variance over N and L dimensions.
        //
        // Input (N, C, L) is stored as data[n*C*L + c*L + l], so reshaping
        // directly to (N*L, C) would scramble channels. Permute to
        // (N, L, C) first so that channels end up as the trailing
        // dimension — the subsequent contiguous+reshape then gives the
        // expected (N*L, C) layout where row j is one per-channel sample.
        Tensor reshaped_input = shape.size() == 3 ?
            input_work.permute({0, 2, 1}).contiguous().reshape({N * L, C}) :
            input_work.contiguous();

        // Compute mean: average over batch dimension (N*L)
        batch_mean = mean(reshaped_input, 0, false);

        // Compute variance
        auto mean_broadcast = batch_mean.unsqueeze(0).contiguous();
        auto centered = (reshaped_input - mean_broadcast).contiguous();
        batch_var = mean(centered * centered, 0, false);

        // Update running statistics. The Bessel correction divides by
        // (batch_size - 1); when batch_size == 1 this is 0 and 1/0 = +inf
        // would poison running_var permanently. The unbiased variance is
        // undefined for a single sample, so skip the running-stats update in
        // that degenerate case (running stats stay unchanged), matching the
        // guard PyTorch applies. The current forward output is unaffected
        // because it uses the biased batch_var below.
        if (track_running_stats_ && batch_size >= 2) {
            auto unbiased_var = batch_var * (static_cast<float>(batch_size) /
                                            static_cast<float>(batch_size - 1));

            auto& rm_var_ptr = buffers_["running_mean"];
            auto& rv_var_ptr = buffers_["running_var"];

            // Exponential moving average update using tensor operations
            // This properly handles both CPU and CUDA/GPU tensors
            auto& rm_tensor = rm_var_ptr->tensor();
            auto& rv_tensor = rv_var_ptr->tensor();
            // Running stats are stored as Float32, but a Float64 input is not
            // upcast, so batch_mean/unbiased_var may be Float64. Reconcile to the
            // buffer dtype so the EMA update stays a same-dtype op and the buffers
            // remain Float32 (mirrors the BatchNorm2d running-stats path).
            Tensor bm = batch_mean.dtype() == rm_tensor.dtype()
                            ? batch_mean : batch_mean.to(rm_tensor.dtype());
            Tensor uv = unbiased_var.dtype() == rv_tensor.dtype()
                            ? unbiased_var : unbiased_var.to(rv_tensor.dtype());
            rm_tensor = rm_tensor * (1.0f - momentum_) + bm * momentum_;
            rv_tensor = rv_tensor * (1.0f - momentum_) + uv * momentum_;

            // Increment on the buffer's native device (avoids host roundtrip).
            auto& nbt = buffers_["num_batches_tracked"]->tensor();
            if (nbt.device().type == Device::Type::CPU) {
                nbt.data<int64_t>()[0]++;
            } else {
                Tensor one = ::tenzor::full({1}, 1.0, nbt.dtype(), nbt.device());
                nbt = ::tenzor::add(nbt, one);
            }
        }
    } else {
        if (track_running_stats_) {
            batch_mean = buffers_["running_mean"]->tensor();
            batch_var = buffers_["running_var"]->tensor();
            // Reconcile device/dtype with the input (mirrors BN2d eval branch).
            // Running stats are stored as Float32 on the buffer's home device;
            // normalization below mixes them directly with input_work, so they
            // must match input_work's device and dtype. Without this, a CUDA
            // input with CPU-stored stats yields a mixed-device subtraction,
            // and a Float64 input (which is not upcast) mismatches the Float32
            // stats.
            if (batch_mean.device() != original_device) {
                batch_mean = batch_mean.to(original_device);
            }
            if (batch_var.device() != original_device) {
                batch_var = batch_var.to(original_device);
            }
            const DType input_dtype = input_work.dtype();
            if (batch_mean.dtype() != input_dtype) {
                batch_mean = batch_mean.to(input_dtype);
            }
            if (batch_var.dtype() != input_dtype) {
                batch_var = batch_var.to(input_dtype);
            }
        } else {
            throw std::runtime_error("BatchNorm1d in eval mode requires track_running_stats=true");
        }
    }

    // Normalize. All intermediates must match input_work.dtype() — when we
    // upcasted above, affine weight/bias (stored at original dtype) need
    // to be cast to Float32 before mixing with upcasted tensors.
    const DType compute_dtype = input_work.dtype();
    auto to_compute = [&](const Tensor& t) -> Tensor {
        return t.dtype() == compute_dtype ? t : t.to(compute_dtype);
    };

    Tensor output;
    if (shape.size() == 3) {
        auto mean_broadcast = batch_mean.unsqueeze(0).unsqueeze(2).contiguous();
        auto var_broadcast = batch_var.unsqueeze(0).unsqueeze(2).contiguous();
        auto eps_tensor = full({}, eps_, var_broadcast.dtype(), var_broadcast.device());
        auto invstd = pow(var_broadcast + eps_tensor, -0.5f).contiguous();
        auto normalized = ((input_work - mean_broadcast) * invstd).contiguous();

        if (affine_ && cached_weight_ && cached_bias_) {
            auto weight_broadcast =
                to_compute(cached_weight_->tensor()).unsqueeze(0).unsqueeze(2).contiguous();
            auto bias_broadcast =
                to_compute(cached_bias_->tensor()).unsqueeze(0).unsqueeze(2).contiguous();
            output = (normalized * weight_broadcast + bias_broadcast).contiguous();
        } else {
            output = normalized;
        }
    } else {
        auto mean_broadcast = batch_mean.unsqueeze(0).contiguous();
        auto var_broadcast = batch_var.unsqueeze(0).contiguous();
        auto eps_tensor = full({}, eps_, var_broadcast.dtype(), var_broadcast.device());
        auto invstd = pow(var_broadcast + eps_tensor, -0.5f).contiguous();
        auto normalized = ((input_work - mean_broadcast) * invstd).contiguous();

        if (affine_ && cached_weight_ && cached_bias_) {
            auto weight_broadcast =
                to_compute(cached_weight_->tensor()).unsqueeze(0).contiguous();
            auto bias_broadcast =
                to_compute(cached_bias_->tensor()).unsqueeze(0).contiguous();
            output = (normalized * weight_broadcast + bias_broadcast).contiguous();
        } else {
            output = normalized;
        }
    }

    // Restore the caller's dtype (Float16/BFloat16) if we upcasted for stability.
    if (needs_upcast) {
        output = output.to(orig_dtype);
    }

    // Set up autograd if needed
    bool requires_grad = input.requires_grad();
    if (affine_ && cached_weight_) {
        requires_grad = requires_grad || cached_weight_->requires_grad();
    }

    if (is_grad_enabled() && requires_grad) {
        auto result = Variable(output, true);

        // Compute invstd for backward
        auto eps_tensor = full({}, eps_, batch_var.dtype(), batch_var.device());
        auto invstd = pow(batch_var + eps_tensor, -0.5f);

        Tensor batch_mean_final = batch_mean.contiguous();
        Tensor invstd_final = invstd.contiguous();

        Tensor weight_tensor = (affine_ && cached_weight_) ? cached_weight_->tensor() : ones({C}, input.tensor().dtype(), original_device);

        std::vector<Tensor> tensors_to_save = {
            input.tensor().contiguous(),
            batch_mean_final,
            invstd_final,
            weight_tensor.contiguous()
        };

        auto grad_fn = std::make_shared<BatchNorm1dBackward>(
            affine_, eps_, training_, std::move(tensors_to_save)
        );

        result.set_grad_fn(grad_fn);

        // Track input variables for gradient accumulation
        // MUST include all inputs to maintain 1:1 index correspondence with gradients
        // The engine correctly skips variables that don't require grad
        std::vector<Variable> input_vars = {input};
        if (affine_ && cached_weight_ && cached_bias_) {
            input_vars.push_back(*cached_weight_);
            input_vars.push_back(*cached_bias_);
        }
        grad_fn->set_input_variables(input_vars);

        std::vector<std::shared_ptr<Function>> next_funcs;
        if (input.grad_fn()) {
            next_funcs.push_back(input.grad_fn());
        }
        grad_fn->set_next_functions(next_funcs);

        return result;
    } else {
        return Variable(output, false);
    }
}

auto BatchNorm1d::reset_parameters() -> void {
    if (track_running_stats_) {
        buffers_["running_mean"]->tensor().zero_();
        buffers_["running_var"]->tensor().fill_(1.0f);
        buffers_["num_batches_tracked"]->tensor().zero_();
    }
}

// ============================================================================
// BatchNorm3d — reshape 5D to 4D, delegate to BatchNorm2d
// ============================================================================

// P.1: validation helper for BatchNorm3d's init-list (cannot run statements
// before bn2d_ is constructed otherwise). Returns num_features unchanged so
// the call expression remains usable in the init list; throws on bad combo.
static int64_t bn3d_check_training_and_track(int64_t num_features,
                                              bool training,
                                              bool track_running_stats) {
    if (training && !track_running_stats) {
        throw std::invalid_argument(
            "BatchNorm: training=true with track_running_stats=false is "
            "incompatible — running stats are required during training");
    }
    return num_features;
}

BatchNorm3d::BatchNorm3d(int64_t num_features, double eps, double momentum,
                         bool affine, bool track_running_stats)
    : num_features_(bn3d_check_training_and_track(num_features, /*training=*/true, track_running_stats)),
      // KK.17: heap-allocate bn2d_ via make_shared so register_module owns a
      // real shared_ptr (not a stack-member wrapped in a no-op deleter).
      // Mirrors FF.13's InstanceNorm3d fix.
      bn2d_(std::make_shared<BatchNorm2d>(num_features, eps, momentum, affine, track_running_stats)) {
    register_module("bn2d", bn2d_);
}

auto BatchNorm3d::forward_impl(const Variable& input) -> Variable {
    auto shape = input.shape();
    if (shape.size() != 5) {
        throw std::runtime_error("BatchNorm3d expects 5D input (N,C,D,H,W), got " +
            std::to_string(shape.size()) + "D");
    }

    int64_t N = shape[0], C = shape[1], D = shape[2], H = shape[3], W = shape[4];

    // Reshape (N, C, D, H, W) -> (N, C, D*H, W) to use BatchNorm2d.
    // Use ::tenzor::reshape (the autograd Variable overload declared in
    // tenzor/autograd/ops.hpp) so the grad_fn chain is preserved —
    // constructing a raw Variable(tensor, requires_grad) would drop the
    // upstream chain and silently break .backward() on the final output.
    Variable reshaped = ::tenzor::reshape(input, {N, C, D * H, W});
    Variable result = bn2d_->forward(reshaped);

    // Reshape back to (N, C, D, H, W), again autograd-aware.
    return ::tenzor::reshape(result, {N, C, D, H, W});
}

} // namespace tenzor::nn
