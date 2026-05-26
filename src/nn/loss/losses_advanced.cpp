/**
 * @file losses_advanced.cpp
 * @brief Implementation of advanced loss functions
 */

#include "tenzor/nn/loss/losses.hpp"
#include "tenzor/nn/activations/activations.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/reduction.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/indexing.hpp"
#include "tenzor/ops/transform.hpp"
#include "tenzor/autograd/ops.hpp"
#include "tenzor/autograd/function.hpp"
#include "tenzor/backend/fast_dispatch.hpp"
#include "tenzor/backend/op_attributes.hpp"
#include "tenzor/nn/utils/variable_cast.hpp"
#include <stdexcept>
#include <string>
#include <cmath>
#include <cstring>
#include <limits>
#include <algorithm>
#include <vector>

namespace tenzor::nn {

// Helper functions
namespace {
    // Create a scalar Variable that broadcasts with arithmetic ops
    auto scalar_var(float value, const Variable& ref) -> Variable {
        return Variable(full({1}, value, ref.dtype(), ref.device()), false);
    }

    auto apply_reduction(const Variable& loss, const std::string& reduction, int64_t batch_size = 0) -> Variable {
        if (reduction == "none") {
            return loss;
        } else if (reduction == "mean") {
            return mean(loss);
        } else if (reduction == "sum") {
            return sum(loss);
        } else if (reduction == "batchmean") {
            auto summed = sum(loss);
            if (batch_size > 0) {
                // Division by batch_size using scalar Variable
                auto scale = 1.0f / static_cast<float>(batch_size);
                auto scale_var = scalar_var(scale, summed);
                return summed * scale_var;
            }
            return mean(loss);
        }
        return loss;
    }
}

//==============================================================================
// KLDivLoss Implementation
//==============================================================================

KLDivLoss::KLDivLoss(const std::string& reduction, bool log_target)
    : reduction_(reduction), log_target_(log_target) {
    if (reduction_ != "none" && reduction_ != "mean" &&
        reduction_ != "sum" && reduction_ != "batchmean") {
        throw std::invalid_argument("KLDivLoss: reduction must be 'none', 'mean', 'sum', or 'batchmean'");
    }
}

auto KLDivLoss::forward(const Variable& input, const Variable& target) -> Variable {
    // KL(P||Q) = sum(P * (log(P) - log(Q)))
    // input = log(Q), target = P (or log(P) if log_target=true)
    //
    // AA.4: clamp(target, 1e-7, 1.0) and log() on Float16 collapse —
    // 1e-7f < 6.1e-5 (F16 smallest normal), so the clamp lower bound
    // rounds to 0 and log(0) blows up. exp(target) for log_target=true
    // is also unstable in the narrow F16 dynamic range. Widen the whole
    // computation to Float32 and cast the loss back to the input dtype
    // at the end.
    const DType orig_dtype = input.tensor().dtype();
    const bool needs_upcast = (orig_dtype == DType::Float16 ||
                               orig_dtype == DType::BFloat16);
    Variable input_f32 = needs_upcast
        ? tenzor::nn::variable_cast(input, DType::Float32)
        : input;
    Variable target_f32 = needs_upcast
        ? tenzor::nn::variable_cast(target, DType::Float32)
        : target;

    Variable result;
    // Simplified implementation matching manual test
    if (!log_target_) {
        // Standard case: target contains probabilities, input contains log probabilities
        auto target_clamped = clamp(target_f32, 1e-7f, 1.0f);
        auto log_target = log(target_clamped);
        auto diff = log_target - input_f32;
        auto loss_unreduced = target_f32 * diff;

        // Apply reduction inline
        if (reduction_ == "none") {
            result = loss_unreduced;
        } else if (reduction_ == "mean") {
            result = mean(loss_unreduced);
        } else if (reduction_ == "sum") {
            result = sum(loss_unreduced);
        } else { // batchmean
            auto summed = sum(loss_unreduced);
            int64_t batch_size = input_f32.shape()[0];
            if (batch_size > 0) {
                auto scale = 1.0f / static_cast<float>(batch_size);
                auto scale_var = scalar_var(scale, summed);
                result = summed * scale_var;
            } else {
                result = mean(loss_unreduced);
            }
        }
    } else {
        // log_target case: both inputs are log probabilities
        auto exp_target = exp(target_f32);
        auto diff = target_f32 - input_f32;
        auto loss_unreduced = exp_target * diff;

        // Apply reduction inline
        if (reduction_ == "none") {
            result = loss_unreduced;
        } else if (reduction_ == "mean") {
            result = mean(loss_unreduced);
        } else if (reduction_ == "sum") {
            result = sum(loss_unreduced);
        } else { // batchmean
            auto summed = sum(loss_unreduced);
            int64_t batch_size = input_f32.shape()[0];
            if (batch_size > 0) {
                auto scale = 1.0f / static_cast<float>(batch_size);
                auto scale_var = scalar_var(scale, summed);
                result = summed * scale_var;
            } else {
                result = mean(loss_unreduced);
            }
        }
    }

    if (needs_upcast) {
        result = tenzor::nn::variable_cast(result, orig_dtype);
    }
    return result;
}

//==============================================================================
// FocalLoss Implementation
//==============================================================================

FocalLoss::FocalLoss(double alpha, double gamma, const std::string& reduction)
    : alpha_(alpha), gamma_(gamma), reduction_(reduction) {
    if (reduction_ != "none" && reduction_ != "mean" && reduction_ != "sum") {
        throw std::invalid_argument("FocalLoss: reduction must be 'none', 'mean', or 'sum'");
    }
    if (gamma_ < 0.0) {
        throw std::invalid_argument("FocalLoss: gamma must be non-negative");
    }
}

auto FocalLoss::forward(const Variable& input, const Variable& target) -> Variable {
    // Focal Loss: FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    //
    // AA.4: clamp(probs, 1e-7, 1 - 1e-7) and log/exp on Float16 collapse —
    // 1e-7f < 6.1e-5 (F16 smallest normal), so the clamp lower bound rounds
    // to 0, log(0) blows up, and gamma * log(1 - p) at small 1-p saturates
    // the F16 dynamic range. Widen the entire computation to Float32 and
    // cast the loss back to the input dtype at the end. `variable_cast`
    // wires a TypeCastBackward node so gradients flow through the cast.
    const DType orig_dtype = input.tensor().dtype();
    const bool needs_upcast = (orig_dtype == DType::Float16 ||
                               orig_dtype == DType::BFloat16);
    Variable input_f32 = needs_upcast
        ? tenzor::nn::variable_cast(input, DType::Float32)
        : input;
    Variable target_f32 = needs_upcast
        ? tenzor::nn::variable_cast(target, DType::Float32)
        : target;

    // Use log_softmax + exp for autograd-aware softmax computation
    auto log_probs = tenzor::log_softmax(input_f32, 1);  // This has autograd support
    auto probs = exp(log_probs);  // This also has autograd support

    // Clamp to avoid numerical issues
    auto probs_clamped = clamp(probs, 1e-7f, 1.0f - 1e-7f);

    // Compute (1 - p_t)^gamma term
    // Use scalar Variables: 1 - p = p * (-1) + 1
    auto neg_one_var = scalar_var(-1.0f, probs_clamped);
    auto one_var = scalar_var(1.0f, probs_clamped);
    auto one_minus_p = (probs_clamped * neg_one_var) + one_var;

    // Power term - compute (1-p)^gamma using exp(gamma * log(1-p))
    // This correctly handles fractional gamma values
    auto gamma_var = scalar_var(static_cast<float>(gamma_), one_minus_p);
    auto one_minus_p_safe = clamp(one_minus_p, 1e-7f, 1.0f);
    auto exponent = gamma_var * log(one_minus_p_safe);
    auto exponent_safe = clamp(exponent, -50.0f, 50.0f);
    Variable modulating_factor = exp(exponent_safe);

    // Recompute log_probs from clamped probs for consistency
    auto log_probs_clamped = log(probs_clamped);

    // Compute focal loss: -alpha * (1-p)^gamma * log(p) * target
    auto alpha_var = scalar_var(static_cast<float>(alpha_), modulating_factor);
    auto loss_unreduced = neg(modulating_factor * alpha_var * log_probs_clamped * target_f32);

    // Sum over class dimension
    auto loss_per_sample = sum(loss_unreduced, 1, false);

    Variable reduced = apply_reduction(loss_per_sample, reduction_);
    if (needs_upcast) {
        reduced = tenzor::nn::variable_cast(reduced, orig_dtype);
    }
    return reduced;
}

//==============================================================================
// DiceLoss Implementation
//==============================================================================

DiceLoss::DiceLoss(double smooth, const std::string& reduction)
    : smooth_(smooth), reduction_(reduction) {
    if (reduction_ != "none" && reduction_ != "mean" && reduction_ != "sum") {
        throw std::invalid_argument("DiceLoss: reduction must be 'none', 'mean', or 'sum'");
    }
    if (smooth_ < 0.0) {
        throw std::invalid_argument("DiceLoss: smooth must be non-negative");
    }
}

auto DiceLoss::forward(const Variable& input, const Variable& target) -> Variable {
    // Validate: input must be at least 2D (batch + channel dimensions)
    if (input.shape().size() < 2) {
        throw std::invalid_argument(
            "DiceLoss: input must be at least 2D (batch + channels), got " +
            std::to_string(input.shape().size()) + "D");
    }

    // Dice = 1 - (2 * |X ∩ Y| + smooth) / (|X| + |Y| + smooth)
    // For differentiable version: intersection = sum(input * target)

    // Flatten spatial dimensions but keep batch and channel dims
    // input shape: (N, C, H, W) -> compute dice per channel per batch

    // Compute intersection, input_sum, and target_sum
    auto intersection = input * target;
    // Sum over all dims except batch (dim 0) for per-sample Dice computation
    int64_t ndim = static_cast<int64_t>(input.shape().size());
    Variable intersection_sum = intersection;
    Variable input_sum_v = input;
    Variable target_sum_v = target;
    // Reduce dims from last to first (skipping batch dim 0)
    for (int64_t d = ndim - 1; d >= 1; --d) {
        intersection_sum = sum(intersection_sum, d, false);
        input_sum_v = sum(input_sum_v, d, false);
        target_sum_v = sum(target_sum_v, d, false);
    }
    // If input is 1D or scalar, fall back to global sum
    if (ndim <= 1) {
        intersection_sum = sum(intersection);
        input_sum_v = sum(input);
        target_sum_v = sum(target);
    }

    // Numerator: 2 * intersection + smooth
    auto two_var = scalar_var(2.0f, intersection_sum);
    auto smooth_var = scalar_var(static_cast<float>(smooth_), intersection_sum);
    auto numerator = (intersection_sum * two_var) + smooth_var;

    // Denominator: sum(input) + sum(target) + smooth
    auto smooth_var2 = scalar_var(static_cast<float>(smooth_), input_sum_v);
    auto denominator = input_sum_v + target_sum_v + smooth_var2;

    // Dice coefficient
    auto dice_coeff = numerator / denominator;

    // Dice loss = 1 - dice_coefficient
    auto neg_one = scalar_var(-1.0f, dice_coeff);
    auto one = scalar_var(1.0f, dice_coeff);
    auto loss = (dice_coeff * neg_one) + one;

    return apply_reduction(loss, reduction_);
}

//==============================================================================
// HuberLoss Implementation
//==============================================================================

HuberLoss::HuberLoss(double delta, const std::string& reduction)
    : delta_(delta), reduction_(reduction) {
    if (reduction_ != "none" && reduction_ != "mean" && reduction_ != "sum") {
        throw std::invalid_argument("HuberLoss: reduction must be 'none', 'mean', or 'sum'");
    }
    if (delta_ <= 0.0) {
        throw std::invalid_argument("HuberLoss: delta must be positive");
    }
}

auto HuberLoss::forward(const Variable& input, const Variable& target) -> Variable {
    // Huber loss:
    // L(x, y) = 0.5 * (x - y)^2                    if |x - y| <= delta
    //         = delta * (|x - y| - 0.5 * delta)     otherwise
    //
    // Reformulation using clamp on |diff|:
    //   clipped = clamp(|diff|, 0, delta)
    //   L = 0.5 * clipped^2 + delta * (|diff| - clipped)
    // This is exact: small errors give 0.5*diff^2, large give delta*|diff| - 0.5*delta^2

    // audit-7 FF.14: small delta (e.g. 1e-3 or below) makes 0.5 * delta^2 and
    // delta * |diff| underflow / round to zero in F16/BF16, collapsing the
    // quadratic-vs-linear distinction.  Mirror the AA.4 / EE.9 upcast
    // pattern: widen input + target to Float32, compute clamp/quadratic at
    // F32, then narrow the loss back to the original dtype.
    const DType orig_dtype = input.tensor().dtype();
    const bool needs_upcast = (orig_dtype == DType::Float16 ||
                               orig_dtype == DType::BFloat16);
    Variable input_f32 = needs_upcast
        ? tenzor::nn::variable_cast(input, DType::Float32)
        : input;
    Variable target_f32 = needs_upcast
        ? tenzor::nn::variable_cast(target, DType::Float32)
        : target;

    auto diff = input_f32 - target_f32;
    auto abs_diff = abs(diff);

    auto clipped = clamp(abs_diff, 0.0f, static_cast<float>(delta_));

    auto half_var = scalar_var(0.5f, clipped);
    auto delta_var = scalar_var(static_cast<float>(delta_), clipped);

    // L = 0.5 * clipped^2 + delta * (|diff| - clipped)
    auto loss_unreduced = half_var * clipped * clipped + delta_var * (abs_diff - clipped);

    Variable reduced = apply_reduction(loss_unreduced, reduction_);
    if (needs_upcast) {
        reduced = tenzor::nn::variable_cast(reduced, orig_dtype);
    }
    return reduced;
}

//==============================================================================
// Functional Implementations
//==============================================================================

auto kl_div_loss(const Variable& input, const Variable& target,
                const std::string& reduction,
                bool log_target) -> Variable {
    KLDivLoss loss(reduction, log_target);
    return loss.forward(input, target);
}

auto focal_loss(const Variable& input, const Variable& target,
               double alpha, double gamma,
               const std::string& reduction) -> Variable {
    FocalLoss loss(alpha, gamma, reduction);
    return loss.forward(input, target);
}

auto dice_loss(const Variable& input, const Variable& target,
              double smooth,
              const std::string& reduction) -> Variable {
    DiceLoss loss(smooth, reduction);
    return loss.forward(input, target);
}

auto huber_loss(const Variable& input, const Variable& target,
               double delta,
               const std::string& reduction) -> Variable {
    HuberLoss loss(delta, reduction);
    return loss.forward(input, target);
}

// ============================================================================
// CTCLoss Implementation
// ============================================================================

namespace {

// Log-sum-exp for two values (numerically stable addition in log-space)
inline float log_sum_exp(float a, float b) {
    if (a == -std::numeric_limits<float>::infinity()) return b;
    if (b == -std::numeric_limits<float>::infinity()) return a;
    float max_val = std::max(a, b);
    return max_val + std::log(std::exp(a - max_val) + std::exp(b - max_val));
}

constexpr float NEG_INF = -std::numeric_limits<float>::infinity();

// Compute CTC loss for a single batch element using the forward-backward algorithm
// Returns: (loss, grad_log_probs)
// log_probs: (T, C) - log probabilities at each timestep
// target: pointer to S target labels
// T: input length, S: target length, C: number of classes
auto ctc_loss_single(
    const float* log_probs,  // (T, C)
    const int32_t* target,   // (S,)
    int64_t T, int64_t S, int64_t C,
    int64_t blank
) -> std::pair<float, std::vector<float>> {

    // Extended label: insert blanks between and around target labels
    // e.g., target = [1, 2, 3] -> extended = [blank, 1, blank, 2, blank, 3, blank]
    int64_t L = 2 * S + 1;  // extended label length
    std::vector<int32_t> ext_label(L);
    for (int64_t i = 0; i < L; ++i) {
        ext_label[i] = (i % 2 == 0) ? static_cast<int32_t>(blank) : target[i / 2];
    }

    // Forward pass: alpha[t][s] = log-probability of emitting ext_label[0..s] in time 0..t
    std::vector<float> alpha(T * L, NEG_INF);

    // Initialize t=0
    alpha[0 * L + 0] = log_probs[0 * C + ext_label[0]];
    if (L > 1) {
        alpha[0 * L + 1] = log_probs[0 * C + ext_label[1]];
    }

    // Fill forward
    for (int64_t t = 1; t < T; ++t) {
        for (int64_t s = 0; s < L; ++s) {
            float prev = alpha[(t - 1) * L + s];
            if (s > 0) {
                prev = log_sum_exp(prev, alpha[(t - 1) * L + (s - 1)]);
            }
            // Can skip a blank if current and 2-back are different
            if (s > 1 && ext_label[s] != blank && ext_label[s] != ext_label[s - 2]) {
                prev = log_sum_exp(prev, alpha[(t - 1) * L + (s - 2)]);
            }
            alpha[t * L + s] = prev + log_probs[t * C + ext_label[s]];
        }
    }

    // Total log-probability: log_sum_exp of the last two valid positions
    float log_prob = log_sum_exp(alpha[(T - 1) * L + (L - 1)],
                                  alpha[(T - 1) * L + (L - 2)]);

    // Backward pass: beta[t][s]
    std::vector<float> beta(T * L, NEG_INF);

    beta[(T - 1) * L + (L - 1)] = 0.0f;  // log(1)
    if (L > 1) {
        beta[(T - 1) * L + (L - 2)] = 0.0f;
    }

    for (int64_t t = T - 2; t >= 0; --t) {
        for (int64_t s = 0; s < L; ++s) {
            float next = beta[(t + 1) * L + s] + log_probs[(t + 1) * C + ext_label[s]];
            if (s < L - 1) {
                next = log_sum_exp(next,
                    beta[(t + 1) * L + (s + 1)] + log_probs[(t + 1) * C + ext_label[s + 1]]);
            }
            if (s < L - 2 && ext_label[s] != blank && ext_label[s] != ext_label[s + 2]) {
                next = log_sum_exp(next,
                    beta[(t + 1) * L + (s + 2)] + log_probs[(t + 1) * C + ext_label[s + 2]]);
            }
            beta[t * L + s] = next;
        }
    }

    // Compute gradients: grad[t][c] = -exp(alpha[t][s] + beta[t][s] - log_prob)
    // where s maps to label c
    std::vector<float> grad(T * C, 0.0f);

    for (int64_t t = 0; t < T; ++t) {
        // For each position in extended label, accumulate posterior
        for (int64_t s = 0; s < L; ++s) {
            float posterior = alpha[t * L + s] + beta[t * L + s];
            if (posterior > NEG_INF + 1.0f) {
                int32_t c = ext_label[s];
                // Accumulate in log-space, then convert
                float old_val = grad[t * C + c];
                if (old_val == 0.0f) {
                    grad[t * C + c] = posterior;
                } else {
                    grad[t * C + c] = log_sum_exp(old_val, posterior);
                }
            }
        }

        // Convert from log-posterior to gradient
        for (int64_t c = 0; c < C; ++c) {
            if (grad[t * C + c] != 0.0f) {
                grad[t * C + c] = std::exp(log_probs[t * C + c]) -
                                  std::exp(grad[t * C + c] - log_prob);
            } else {
                grad[t * C + c] = std::exp(log_probs[t * C + c]);
            }
        }
    }

    float loss = -log_prob;
    return {loss, grad};
}

// ---------------------------------------------------------------------------
// CTCLossBackward — attaches a pre-computed gradient tensor produced by the
// forward's CPU dynamic-programming pass. The reference implementation in
// CTCLoss::forward already computes d(loss)/d(log_probs) for each batch
// element; we just need to scale it by the scalar (or per-sample) upstream
// gradient and return it so autograd.backward() stops dropping it on the
// floor.
//
// The "scale" stored here is the same scale that the forward applied to the
// aggregated loss (1/total_target_len for "mean", 1.0 for "sum"), so that
// d(reduced_loss)/d(log_probs) = scale * raw_grad. For reduction="none",
// each batch sample's grad row must be multiplied by the corresponding
// grad_output[n] — handled in backward() below.
// ---------------------------------------------------------------------------
class CTCLossBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override {
        (void)inputs;
        return {};  // Never invoked through Function::forward path.
    }

    // grad_outputs[0] is d(L_total)/d(reduced_loss).
    //   - reduction="mean"/"sum": scalar tensor.
    //   - reduction="none"      : shape [N].
    //
    // raw_grad_ lives on the *original* device of log_probs — CPU when the
    // forward took the inline CPU path, GPU when forward dispatched through
    // OpId::CTCLossForward. The scaling here uses device-native tensor ops
    // (tenzor::mul) with no .to(CPU) round-trip.
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override {
        const Tensor& upstream_raw = grad_outputs[0];

        // Move/cast the upstream gradient to raw_grad_'s device and Float32
        // (raw_grad_ is always Float32, see the forward). These ops are
        // no-ops when already in the right device/dtype.
        Tensor g = upstream_raw.dtype() == DType::Float32
                   ? upstream_raw
                   : upstream_raw.to(DType::Float32);
        if (g.device() != raw_grad_.device()) {
            g = g.to(raw_grad_.device());
        }

        Tensor grad_input;
        if (is_per_sample_) {
            // reduction="none": broadcast g (N,) against raw_grad (T, N, C)
            // by reshaping to (1, N, 1).
            const int64_t N = raw_grad_.shape()[1];
            Tensor g_bcast = tenzor::reshape(g.contiguous(), {1, N, 1});
            grad_input = tenzor::mul(raw_grad_, g_bcast);
        } else {
            // reduction="mean"/"sum": multiply by scalar; g is shape {} or
            // {1}, both broadcast cleanly against (T, N, C). scale_ baked in
            // by the forward (1/total_target_len for "mean", 1.0 for "sum").
            grad_input = tenzor::mul(
                tenzor::mul(raw_grad_, g.contiguous()),
                static_cast<double>(scale_));
        }

        // Restore caller's dtype (raw_grad_ already lives on orig_device_,
        // so the device cast is usually a no-op).
        if (grad_input.dtype() != orig_dtype_) {
            grad_input = grad_input.to(orig_dtype_);
        }
        if (grad_input.device() != orig_device_) {
            grad_input = grad_input.to(orig_device_);
        }
        return {grad_input};
    }

    auto name() const -> std::string override { return "CTCLossBackward"; }

    // CTCLoss backward captures the full grad tensor (raw_grad_) at forward
    // time, so from this Function's perspective the grad-output-to-grad-input
    // mapping is a constant-weight linear transform. Second derivative *through
    // this node* is structurally zero (the true second derivative w.r.t.
    // log_probs lives inside the DP pass that filled raw_grad_ and is
    // intentionally not exposed — PyTorch's CTCLoss takes the same stance).
    TENZOR_HIGHER_ORDER_STRUCTURAL_ZERO_STUB()

    Tensor raw_grad_;             // (T, N, C) Float32 on the original device
                                  // (CPU or GPU) — populated by the forward DP
    bool is_per_sample_ = false;  // reduction="none" → true
    float scale_ = 1.0f;          // 1/total_target_len for "mean", 1.0 for "sum"
    DType orig_dtype_ = DType::Float32;
    Device orig_device_ = Device::cpu();
};

} // anonymous namespace

CTCLoss::CTCLoss(const std::string& reduction, int64_t blank, bool zero_infinity)
    : reduction_(reduction), blank_(blank), zero_infinity_(zero_infinity) {
    if (reduction != "none" && reduction != "mean" && reduction != "sum") {
        throw std::invalid_argument("CTCLoss: reduction must be 'none', 'mean', or 'sum'");
    }
}

auto CTCLoss::forward(const Variable& log_probs, const Tensor& targets,
                      const Tensor& input_lengths, const Tensor& target_lengths) -> Variable {
    // log_probs: (T, N, C) - log probabilities
    auto lp_shape = log_probs.shape();
    if (lp_shape.size() != 3) {
        throw std::invalid_argument("CTCLoss: log_probs must be 3D (T, N, C)");
    }

    int64_t T_max = lp_shape[0];
    int64_t N = lp_shape[1];
    int64_t C = lp_shape[2];

    const Device original_device = log_probs.tensor().device();
    const DType original_dtype = log_probs.tensor().dtype();
    const bool needs_grad = log_probs.requires_grad() && is_grad_enabled();

    // -----------------------------------------------------------------------
    // GPU device path: dispatch to the backend's native CTC kernel
    // (OpId::CTCLossForward). The kernel returns {loss_per_sample (N,),
    // raw_grad (T, N, C)} on the same device. Reductions and any necessary
    // scaling are done with device-native tensor ops — no host round-trip
    // for the heavy DP arithmetic.
    // -----------------------------------------------------------------------
    if (original_device.type != Device::Type::CPU) {
        // Stage inputs onto the device with the expected dtypes.
        Tensor lp_dev = log_probs.tensor();
        if (lp_dev.dtype() != DType::Float32) lp_dev = lp_dev.to(DType::Float32);
        lp_dev = lp_dev.contiguous();

        auto to_int32_on_dev = [&](const Tensor& t) {
            Tensor x = t;
            if (x.device() != original_device) x = x.to(original_device);
            if (x.dtype() != DType::Int32) x = x.to(DType::Int32);
            return x.contiguous();
        };
        Tensor tgt_dev = to_int32_on_dev(targets);
        Tensor il_dev  = to_int32_on_dev(input_lengths);
        Tensor tl_dev  = to_int32_on_dev(target_lengths);

        OpAttributes ctc_attrs;
        ctc_attrs.set(AttrKey::Blank, blank_);
        ctc_attrs.set(AttrKey::ZeroInfinity, zero_infinity_);

        std::vector<Tensor> ctc_inputs = {lp_dev, tgt_dev, il_dev, tl_dev};
        auto outputs = dispatch<OpId::CTCLossForward>(ctc_inputs, ctc_attrs);
        if (outputs.size() != 2) {
            throw std::runtime_error(
                "CTCLoss: backend kernel did not return the expected "
                "{loss_per_sample, raw_grad} pair");
        }
        Tensor losses_dev   = outputs[0];   // (N,) Float32
        Tensor raw_grad_dev = outputs[1];   // (T_max, N, C) Float32

        auto attach_grad_fn_dev = [&](Variable& out, bool per_sample, float scale) {
            if (!needs_grad) return;
            auto grad_fn = std::make_shared<CTCLossBackward>();
            grad_fn->raw_grad_ = raw_grad_dev;   // already on original_device
            grad_fn->is_per_sample_ = per_sample;
            grad_fn->scale_ = scale;
            grad_fn->orig_dtype_ = original_dtype;
            grad_fn->orig_device_ = original_device;

            std::vector<std::shared_ptr<Function>> next_funcs;
            if (log_probs.grad_fn()) next_funcs.push_back(log_probs.grad_fn());
            grad_fn->set_next_functions(next_funcs);
            grad_fn->set_input_variables({log_probs});

            out.set_grad_fn(grad_fn);
        };

        if (reduction_ == "none") {
            Tensor loss_out = losses_dev;
            if (loss_out.dtype() != original_dtype) {
                loss_out = loss_out.to(original_dtype);
            }
            Variable out(loss_out, needs_grad);
            attach_grad_fn_dev(out, /*per_sample=*/true, /*scale=*/1.0f);
            return out;
        }

        // For "mean" we need total_target_len (a scalar) — derive it on
        // host by reading the small (N,) target_lengths Int32 tensor. This
        // is a 4*N-byte transfer at most, not the algorithmic DP working
        // set, and is functionally identical to PyTorch's behaviour.
        float scale = 1.0f;
        if (reduction_ == "mean") {
            Tensor tl_cpu_small = (target_lengths.device() == Device::cpu())
                                  ? target_lengths.to(DType::Int32).contiguous()
                                  : target_lengths.to(Device::cpu()).to(DType::Int32).contiguous();
            const int32_t* tl_ptr = tl_cpu_small.data<int32_t>();
            float total_target_len = 0.0f;
            for (int64_t n = 0; n < N; ++n) total_target_len += static_cast<float>(tl_ptr[n]);
            if (total_target_len > 0.0f) scale = 1.0f / total_target_len;
        }
        // Reduction on the device: sum, then scalar-multiply by `scale`
        // (which is 1.0 for "sum" and 1/total_target_len for "mean").
        Tensor total = tenzor::sum(losses_dev);
        if (scale != 1.0f) {
            total = tenzor::mul(total, static_cast<double>(scale));
        }
        if (total.dtype() != original_dtype) total = total.to(original_dtype);
        Variable out(total, needs_grad);
        attach_grad_fn_dev(out, /*per_sample=*/false, scale);
        return out;
    }

    // -----------------------------------------------------------------------
    // CPU device path: inline implementation (unchanged). The kernel
    // version in src/backends/cpu/kernels/ctc.cpp is also registered for
    // dispatch parity, but invoking it here would require an extra
    // attribute set + dispatch indirection without any benefit over the
    // direct loop already written below.
    // -----------------------------------------------------------------------
    Tensor lp_cpu = log_probs.tensor().to(Device::cpu()).to(DType::Float32).contiguous();
    Tensor tgt_cpu = targets.to(Device::cpu()).to(DType::Int32).contiguous();
    Tensor il_cpu = input_lengths.to(Device::cpu()).to(DType::Int32).contiguous();
    Tensor tl_cpu = target_lengths.to(Device::cpu()).to(DType::Int32).contiguous();

    const float* lp_data = lp_cpu.data<float>();
    const int32_t* tgt_data = tgt_cpu.data<int32_t>();
    const int32_t* il_data = il_cpu.data<int32_t>();
    const int32_t* tl_data = tl_cpu.data<int32_t>();

    // targets shape: (N, S_max)
    auto tgt_shape = tgt_cpu.shape();
    int64_t S_max = tgt_shape.size() > 1 ? tgt_shape[1] : tgt_shape[0];

    // Compute per-element losses and gradients
    std::vector<float> losses(N);
    std::vector<float> all_grads(T_max * N * C, 0.0f);

    #pragma omp parallel for if(N > 4)
    for (int64_t n = 0; n < N; ++n) {
        int64_t T_n = il_data[n];
        int64_t S_n = tl_data[n];

        if (T_n <= 0 || S_n <= 0) {
            losses[n] = 0.0f;
            continue;
        }

        // Extract per-batch log_probs: (T_n, C) from (T, N, C)
        std::vector<float> lp_n(T_n * C);
        for (int64_t t = 0; t < T_n; ++t) {
            for (int64_t c = 0; c < C; ++c) {
                lp_n[t * C + c] = lp_data[t * N * C + n * C + c];
            }
        }

        // Extract per-batch targets: (S_n,)
        const int32_t* tgt_n = tgt_data + n * S_max;

        auto [loss, grad] = ctc_loss_single(lp_n.data(), tgt_n, T_n, S_n, C, blank_);

        if (zero_infinity_ && std::isinf(loss)) {
            loss = 0.0f;
            std::fill(grad.begin(), grad.end(), 0.0f);
        }

        losses[n] = loss;

        // Write gradients back to (T, N, C) layout
        for (int64_t t = 0; t < T_n; ++t) {
            for (int64_t c = 0; c < C; ++c) {
                all_grads[t * N * C + n * C + c] = grad[t * C + c];
            }
        }
    }

    // (CPU path) Pack the gradient tensor once; used below for both
    // reductions. original_device, original_dtype, needs_grad were
    // captured at function entry.
    Tensor grad_tensor({T_max, N, C}, DType::Float32, Device::cpu());
    std::memcpy(grad_tensor.data<float>(), all_grads.data(),
                all_grads.size() * sizeof(float));

    auto attach_grad_fn = [&](Variable& out, bool per_sample, float scale) {
        if (!needs_grad) return;
        auto grad_fn = std::make_shared<CTCLossBackward>();
        grad_fn->raw_grad_ = grad_tensor;
        grad_fn->is_per_sample_ = per_sample;
        grad_fn->scale_ = scale;
        grad_fn->orig_dtype_ = original_dtype;
        grad_fn->orig_device_ = original_device;

        std::vector<std::shared_ptr<Function>> next_funcs;
        if (log_probs.grad_fn()) next_funcs.push_back(log_probs.grad_fn());
        grad_fn->set_next_functions(next_funcs);
        grad_fn->set_input_variables({log_probs});

        out.set_grad_fn(grad_fn);
    };

    if (reduction_ == "none") {
        auto loss_tensor = Tensor({N}, DType::Float32, Device::cpu());
        std::memcpy(loss_tensor.data<float>(), losses.data(), N * sizeof(float));
        Variable out(loss_tensor.to(original_dtype).to(original_device), needs_grad);
        attach_grad_fn(out, /*per_sample=*/true, /*scale=*/1.0f);
        return out;
    }

    float total_loss = 0.0f;
    float scale = 1.0f;  // d(reduced_loss)/d(per-sample-grad-sum)
    if (reduction_ == "sum") {
        for (int64_t n = 0; n < N; ++n) {
            total_loss += losses[n];
        }
    } else {  // "mean"
        float total_target_len = 0.0f;
        for (int64_t n = 0; n < N; ++n) {
            total_loss += losses[n];
            total_target_len += static_cast<float>(tl_data[n]);
        }
        if (total_target_len > 0.0f) {
            total_loss /= total_target_len;
            scale = 1.0f / total_target_len;
        }
    }

    auto loss_tensor = Tensor({1}, DType::Float32, Device::cpu());
    loss_tensor.data<float>()[0] = total_loss;
    Variable out(loss_tensor.to(original_dtype).to(original_device), needs_grad);
    attach_grad_fn(out, /*per_sample=*/false, scale);
    return out;
}

//==============================================================================
// SoftMarginLoss Implementation
//==============================================================================

SoftMarginLoss::SoftMarginLoss(Reduction reduction) : reduction_(reduction) {}

auto SoftMarginLoss::forward(const Variable& input, const Variable& target) -> Variable {
    // KK.18: log(1 + exp(-y*x)) computed directly in F16/BF16 overflows for
    // |x| > 11 (exp(11) ≈ 6e4, F16 max ≈ 6.55e4).  Widen to Float32, compute,
    // narrow the loss back to the input dtype.  Mirrors AA.4 / EE.10
    // (KLDivLoss / FocalLoss / HuberLoss / PoissonNLLLoss).
    const DType orig_dtype = input.tensor().dtype();
    const bool needs_upcast = (orig_dtype == DType::Float16 ||
                               orig_dtype == DType::BFloat16);
    Variable input_c = needs_upcast
        ? tenzor::nn::variable_cast(input, DType::Float32)
        : input;
    Variable target_c = needs_upcast
        ? tenzor::nn::variable_cast(target, DType::Float32)
        : target;

    // loss = log(1 + exp(-y * x))
    // Use log1p(exp(-y*x)) for numerical stability when -y*x is large negative
    auto neg_yx = neg(input_c * target_c);
    // softplus: log(1 + exp(x)) = max(x,0) + log(1 + exp(-|x|))
    // But since we need autograd, use: log(1 + exp(x))
    auto one = scalar_var(1.0f, neg_yx);
    auto loss = log(one + exp(neg_yx));

    auto red_str = reduction_to_string(reduction_);
    Variable reduced = apply_reduction(loss, red_str);
    if (needs_upcast) {
        reduced = tenzor::nn::variable_cast(reduced, orig_dtype);
    }
    return reduced;
}

auto soft_margin_loss(const Variable& input, const Variable& target,
                     Reduction reduction) -> Variable {
    SoftMarginLoss loss_fn(reduction);
    return loss_fn.forward(input, target);
}

//==============================================================================
// HingeEmbeddingLoss Implementation
//==============================================================================

HingeEmbeddingLoss::HingeEmbeddingLoss(double margin, Reduction reduction)
    : margin_(margin), reduction_(reduction) {}

auto HingeEmbeddingLoss::forward(const Variable& input, const Variable& target) -> Variable {
    // loss = x  if y = 1
    // loss = max(0, margin - x)  if y = -1
    // Combined: loss = (y == 1) * x + (y == -1) * max(0, margin - x)

    auto one = scalar_var(1.0f, input);
    auto neg_one = scalar_var(-1.0f, input);
    auto margin_var = scalar_var(static_cast<float>(margin_), input);

    // Masks: (target + 1) / 2 gives 1 for y=1, 0 for y=-1
    auto pos_mask = (target + one) * scalar_var(0.5f, input);
    auto neg_mask = (one - pos_mask);

    auto hinge_part = relu(margin_var - input);
    auto loss = pos_mask * input + neg_mask * hinge_part;

    auto red_str = reduction_to_string(reduction_);
    return apply_reduction(loss, red_str);
}

auto hinge_embedding_loss(const Variable& input, const Variable& target,
                          double margin, Reduction reduction) -> Variable {
    HingeEmbeddingLoss loss_fn(margin, reduction);
    return loss_fn.forward(input, target);
}

//==============================================================================
// PoissonNLLLoss Implementation
//==============================================================================

PoissonNLLLoss::PoissonNLLLoss(bool log_input, bool full, double eps,
                               Reduction reduction)
    : log_input_(log_input), full_(full), eps_(eps), reduction_(reduction) {}

auto PoissonNLLLoss::forward(const Variable& input, const Variable& target) -> Variable {
    // AA.4 / EE.10: default eps_ = 1e-8 is below F16's smallest normal
    // (~6.1e-5), so `input + eps` and clamp(target, eps, FLT_MAX) both
    // collapse the eps term to 0 in Float16, then log(0) blows up. Widen
    // the entire computation to Float32 and cast the loss back to the
    // input dtype at the end. `variable_cast` wires a TypeCastBackward
    // node so gradients flow through the cast.
    const DType orig_dtype = input.tensor().dtype();
    const bool needs_upcast = (orig_dtype == DType::Float16 ||
                               orig_dtype == DType::BFloat16);
    Variable input_c = needs_upcast
        ? tenzor::nn::variable_cast(input, DType::Float32)
        : input;
    Variable target_c = needs_upcast
        ? tenzor::nn::variable_cast(target, DType::Float32)
        : target;

    Variable loss;
    if (log_input_) {
        // loss = exp(input) - target * input
        loss = exp(input_c) - target_c * input_c;
    } else {
        // loss = input - target * log(input + eps)
        auto eps_var = scalar_var(static_cast<float>(eps_), input_c);
        loss = input_c - target_c * log(input_c + eps_var);
    }

    if (full_) {
        // Add Stirling approximation: target * log(target) - target + 0.5 * log(2 * pi * target)
        // Simplified: use only the dominant term target * log(target) - target for target > 1
        auto target_safe = clamp(target_c, static_cast<float>(eps_), std::numeric_limits<float>::max());
        auto stirling = target_safe * log(target_safe) - target_safe;
        // Add 0.5 * log(2*pi*target)
        auto half = scalar_var(0.5f, target_c);
        auto two_pi = scalar_var(static_cast<float>(2.0 * M_PI), target_c);
        stirling = stirling + half * log(two_pi * target_safe);
        loss = loss + stirling;
    }

    auto red_str = reduction_to_string(reduction_);
    Variable reduced = apply_reduction(loss, red_str);
    if (needs_upcast) {
        reduced = tenzor::nn::variable_cast(reduced, orig_dtype);
    }
    return reduced;
}

auto poisson_nll_loss(const Variable& input, const Variable& target,
                      bool log_input, bool full, double eps,
                      Reduction reduction) -> Variable {
    PoissonNLLLoss loss_fn(log_input, full, eps, reduction);
    return loss_fn.forward(input, target);
}

//==============================================================================
// CosineEmbeddingLoss Implementation
//==============================================================================

CosineEmbeddingLoss::CosineEmbeddingLoss(double margin, Reduction reduction)
    : margin_(margin), reduction_(reduction) {}

auto CosineEmbeddingLoss::forward(const Variable& input1, const Variable& input2,
                                  const Variable& target) -> Variable {
    // AA.4 / EE.10: eps = 1e-8 used to stabilise the norm denominator is
    // below F16's smallest normal (~6.1e-5), so `norm_sq + eps` collapses
    // to norm_sq in Float16; when both vectors are near zero, sqrt(0) * sqrt(0)
    // produces a div-by-zero. Widen the entire computation to Float32 and
    // cast the loss back to the input dtype at the end. `variable_cast`
    // wires a TypeCastBackward node so gradients flow through the cast.
    const DType orig_dtype = input1.tensor().dtype();
    const bool needs_upcast = (orig_dtype == DType::Float16 ||
                               orig_dtype == DType::BFloat16);
    Variable input1_c = needs_upcast
        ? tenzor::nn::variable_cast(input1, DType::Float32)
        : input1;
    Variable input2_c = needs_upcast
        ? tenzor::nn::variable_cast(input2, DType::Float32)
        : input2;
    Variable target_c = needs_upcast
        ? tenzor::nn::variable_cast(target, DType::Float32)
        : target;

    // Compute cosine similarity along last dimension
    // cos_sim = sum(x1 * x2, dim=-1) / (norm(x1) * norm(x2))
    auto product = input1_c * input2_c;
    int64_t last_dim = static_cast<int64_t>(input1_c.shape().size()) - 1;
    auto dot = sum(product, last_dim, false);

    auto norm1_sq = sum(input1_c * input1_c, last_dim, false);
    auto norm2_sq = sum(input2_c * input2_c, last_dim, false);
    auto eps_var = scalar_var(1e-8f, norm1_sq);
    auto norms = sqrt(norm1_sq + eps_var) * sqrt(norm2_sq + eps_var);
    auto cos_sim = dot / norms;

    // loss = (1 - cos_sim) if y = 1
    // loss = max(0, cos_sim - margin) if y = -1
    auto one = scalar_var(1.0f, cos_sim);
    auto pos_mask = (target_c + one) * scalar_var(0.5f, cos_sim);
    auto neg_mask = (one - pos_mask);

    auto margin_var = scalar_var(static_cast<float>(margin_), cos_sim);
    auto pos_loss = one - cos_sim;
    auto neg_loss = relu(cos_sim - margin_var);
    auto loss = pos_mask * pos_loss + neg_mask * neg_loss;

    auto red_str = reduction_to_string(reduction_);
    Variable reduced = apply_reduction(loss, red_str);
    if (needs_upcast) {
        reduced = tenzor::nn::variable_cast(reduced, orig_dtype);
    }
    return reduced;
}

auto cosine_embedding_loss(const Variable& input1, const Variable& input2,
                           const Variable& target,
                           double margin, Reduction reduction) -> Variable {
    CosineEmbeddingLoss loss_fn(margin, reduction);
    return loss_fn.forward(input1, input2, target);
}

//==============================================================================
// TripletMarginLoss Implementation (delegates to TripletLoss logic)
//==============================================================================

TripletMarginLoss::TripletMarginLoss(double margin, double p, bool swap,
                                     Reduction reduction)
    : margin_(margin), p_(p), swap_(swap), reduction_(reduction) {}

auto TripletMarginLoss::forward(const Variable& anchor, const Variable& positive,
                                const Variable& negative) -> Variable {
    // Compute pairwise distances using p-norm
    // d(a, p) = ||a - p||_p, d(a, n) = ||a - n||_p
    //
    // CC.8: the `1e-8f` eps for the L2 sqrt is well below the F16 smallest
    // normal (~6.1e-5), so on F16/BF16 inputs the additive eps rounds to 0
    // and sqrt(0) leaves the backward `1/(2*sqrt(...))` term blowing up to
    // inf wherever anchor==positive or anchor==negative. Widen the diff
    // tensors to Float32 for the distance computation (mirrors AA.4 FocalLoss
    // / KLDiv upcast). `variable_cast` wires TypeCastBackward so gradients
    // flow through the cast, and we downcast d_pos / d_neg back to the
    // input dtype before the margin/relu/reduction so the loss dtype is
    // preserved.
    const DType orig_dtype = anchor.tensor().dtype();
    const bool needs_upcast = (orig_dtype == DType::Float16 ||
                               orig_dtype == DType::BFloat16);
    Variable anchor_c = needs_upcast
        ? tenzor::nn::variable_cast(anchor, DType::Float32)
        : anchor;
    Variable positive_c = needs_upcast
        ? tenzor::nn::variable_cast(positive, DType::Float32)
        : positive;
    Variable negative_c = needs_upcast
        ? tenzor::nn::variable_cast(negative, DType::Float32)
        : negative;

    auto diff_pos = anchor_c - positive_c;
    auto diff_neg = anchor_c - negative_c;

    // For p=2 (L2 norm): sqrt(sum(diff^2, dim=-1))
    // For p=1 (L1 norm): sum(|diff|, dim=-1)
    int64_t last_dim = static_cast<int64_t>(anchor.shape().size()) - 1;
    Variable d_pos, d_neg;
    auto eps_var = scalar_var(1e-8f, diff_pos);

    if (p_ == 2.0) {
        d_pos = sqrt(sum(diff_pos * diff_pos, last_dim, false) + eps_var);
        d_neg = sqrt(sum(diff_neg * diff_neg, last_dim, false) + eps_var);
    } else {
        d_pos = sum(abs(diff_pos), last_dim, false);
        d_neg = sum(abs(diff_neg), last_dim, false);
    }

    if (swap_) {
        // Distance swap: use min(d(a,n), d(p,n)) for negative distance
        auto diff_pn = positive_c - negative_c;
        Variable d_pn;
        if (p_ == 2.0) {
            d_pn = sqrt(sum(diff_pn * diff_pn, last_dim, false) + eps_var);
        } else {
            d_pn = sum(abs(diff_pn), last_dim, false);
        }
        // d_neg = min(d_neg, d_pn) via: d_neg - relu(d_neg - d_pn)
        d_neg = d_neg - relu(d_neg - d_pn);
    }

    // Downcast distances back to the input dtype before margin/relu so the
    // resulting loss has the user-visible input dtype.
    if (needs_upcast) {
        d_pos = tenzor::nn::variable_cast(d_pos, orig_dtype);
        d_neg = tenzor::nn::variable_cast(d_neg, orig_dtype);
    }

    auto margin_var = scalar_var(static_cast<float>(margin_), d_pos);
    auto loss = relu(d_pos - d_neg + margin_var);

    auto red_str = reduction_to_string(reduction_);
    return apply_reduction(loss, red_str);
}

auto triplet_margin_loss(const Variable& anchor, const Variable& positive,
                         const Variable& negative,
                         double margin, double p, bool swap,
                         Reduction reduction) -> Variable {
    TripletMarginLoss loss_fn(margin, p, swap, reduction);
    return loss_fn.forward(anchor, positive, negative);
}

//==============================================================================
// TripletMarginWithDistanceLoss Implementation
//==============================================================================

TripletMarginWithDistanceLoss::TripletMarginWithDistanceLoss(
    DistanceFunction distance_fn, double margin, bool swap, Reduction reduction)
    : distance_fn_(std::move(distance_fn))
    , margin_(margin)
    , swap_(swap)
    , reduction_(reduction) {}

auto TripletMarginWithDistanceLoss::forward(const Variable& anchor,
                                            const Variable& positive,
                                            const Variable& negative) -> Variable {
    auto d_pos = distance_fn_(anchor, positive);
    auto d_neg = distance_fn_(anchor, negative);

    if (swap_) {
        auto d_pn = distance_fn_(positive, negative);
        // d_neg = min(d_neg, d_pn) via: d_neg - relu(d_neg - d_pn)
        d_neg = d_neg - relu(d_neg - d_pn);
    }

    auto margin_var = scalar_var(static_cast<float>(margin_), d_pos);
    auto loss = relu(d_pos - d_neg + margin_var);

    auto red_str = reduction_to_string(reduction_);
    return apply_reduction(loss, red_str);
}

//==============================================================================
// MultiLabelSoftMarginLoss Implementation
//==============================================================================

MultiLabelSoftMarginLoss::MultiLabelSoftMarginLoss(Reduction reduction)
    : reduction_(reduction) {}

auto MultiLabelSoftMarginLoss::forward(const Variable& input,
                                        const Variable& target) -> Variable {
    // loss = -1/C * sum_c [y_c * log(sigma(x_c)) + (1-y_c) * log(1-sigma(x_c))]
    // Numerically stable: use log-sum-exp trick
    // log(sigma(x)) = x - log(1 + exp(x)) = -softplus(-x)
    // log(1-sigma(x)) = -log(1 + exp(x)) = -softplus(x)

    // KK.18: log(1 + exp(x)) computed directly in F16/BF16 overflows for
    // |x| > 11 (exp(11) ≈ 6e4, F16 max ≈ 6.55e4).  Widen to Float32, compute,
    // narrow the loss back.  Mirrors SoftMarginLoss above.
    const DType orig_dtype = input.tensor().dtype();
    const bool needs_upcast = (orig_dtype == DType::Float16 ||
                               orig_dtype == DType::BFloat16);
    Variable input_c = needs_upcast
        ? tenzor::nn::variable_cast(input, DType::Float32)
        : input;
    Variable target_c = needs_upcast
        ? tenzor::nn::variable_cast(target, DType::Float32)
        : target;

    auto one = scalar_var(1.0f, input_c);
    auto neg_input = neg(input_c);

    // softplus(x) = log(1 + exp(x))
    auto sp_pos = log(one + exp(input_c));     // log(1 + exp(x)) = softplus(x)
    auto sp_neg = log(one + exp(neg_input));   // log(1 + exp(-x)) = softplus(-x)

    // log(sigma(x)) = -softplus(-x), log(1-sigma(x)) = -softplus(x)
    auto loss_per_element = target_c * sp_neg + (one - target_c) * sp_pos;

    // Average over class dimension (last dim)
    int64_t class_dim = static_cast<int64_t>(input_c.shape().size()) - 1;
    auto loss_per_sample = mean(loss_per_element, class_dim, false);

    auto red_str = reduction_to_string(reduction_);
    Variable reduced = apply_reduction(loss_per_sample, red_str);
    if (needs_upcast) {
        reduced = tenzor::nn::variable_cast(reduced, orig_dtype);
    }
    return reduced;
}

auto multi_label_soft_margin_loss(const Variable& input, const Variable& target,
                                   Reduction reduction) -> Variable {
    MultiLabelSoftMarginLoss loss_fn(reduction);
    return loss_fn.forward(input, target);
}

//==============================================================================
// MultiMarginLoss Implementation
//==============================================================================

MultiMarginLoss::MultiMarginLoss(int p, double margin, Reduction reduction)
    : p_(p), margin_(margin), reduction_(reduction) {
    if (p_ != 1 && p_ != 2) {
        throw std::invalid_argument("MultiMarginLoss: p must be 1 or 2");
    }
}

auto MultiMarginLoss::forward(const Variable& input, const Tensor& target) -> Variable {
    // loss = 1/C * sum_{j != y} max(0, margin - x[y] + x[j])^p
    // input: (N, C), target: (N,) with class indices

    auto shape = input.shape();
    int64_t N = shape[0];
    int64_t C = shape[1];

    // Use Variable-aware gather to extract x[y] for each sample. This preserves
    // the autograd chain through the correct-class score path (the previous
    // implementation extracted via raw `input.tensor().to(cpu).data<float>()`,
    // which severs grad_fn — see MEMORY.md feedback_raw_tensor_op_bug.md).
    Tensor target_idx = target.to(DType::Int64).to(input.tensor().device()).contiguous().reshape({N, 1});
    auto correct_scores = ::tenzor::gather(input, /*dim=*/1, target_idx);  // (N, 1) Variable

    // margin - x[y] + x[j] for all j (broadcast (N,1) - (N,1) + (N,C) → (N,C))
    auto margin_var = scalar_var(static_cast<float>(margin_), input);
    auto diff = margin_var - correct_scores + input;
    auto hinge = relu(diff);

    // Build the one-hot mask via the device-aware one_hot op rather than
    // host-staging the targets and writing the mask in a CPU loop. The mask
    // is constant w.r.t. input, so wrap the result in a no-grad Variable.
    Tensor target_i64 = (target.dtype() == DType::Int64)
        ? target : target.to(DType::Int64);
    Tensor one_hot_i = ::tenzor::one_hot(target_i64, C);
    Tensor one_hot_t = one_hot_i.to(input.tensor().dtype());
    if (one_hot_t.device() != input.tensor().device()) {
        one_hot_t = one_hot_t.to(input.tensor().device());
    }
    auto one_hot = Variable(one_hot_t, false);

    auto one = scalar_var(1.0f, one_hot);
    auto mask = one - one_hot;  // 1 for incorrect classes, 0 for correct
    auto masked_hinge = hinge * mask;

    Variable loss_per_sample;
    if (p_ == 2) {
        loss_per_sample = mean(masked_hinge * masked_hinge, 1, false);
    } else {
        loss_per_sample = mean(masked_hinge, 1, false);
    }

    auto red_str = reduction_to_string(reduction_);
    return apply_reduction(loss_per_sample, red_str);
}

auto multi_margin_loss(const Variable& input, const Tensor& target,
                       int p, double margin, Reduction reduction) -> Variable {
    MultiMarginLoss loss_fn(p, margin, reduction);
    return loss_fn.forward(input, target);
}

//==============================================================================
// GaussianNLLLoss Implementation
//==============================================================================

GaussianNLLLoss::GaussianNLLLoss(bool full, double eps, Reduction reduction)
    : full_(full), eps_(eps), reduction_(reduction) {}

auto GaussianNLLLoss::forward(const Variable& input, const Variable& target,
                               const Variable& var) -> Variable {
    // loss = 0.5 * (log(var) + (input - target)^2 / var)
    // Optionally + 0.5 * log(2*pi) when full=true
    //
    // AA.4 / EE.10: default eps_ = 1e-6 is below F16's smallest normal
    // (~6.1e-5), so clamp(var, eps, FLT_MAX) collapses small variances to
    // 0 in Float16; log(0) blows up and diff*diff/0 produces inf. Widen
    // the entire computation to Float32 and cast the loss back to the
    // input dtype at the end. `variable_cast` wires a TypeCastBackward
    // node so gradients flow through the cast.
    const DType orig_dtype = input.tensor().dtype();
    const bool needs_upcast = (orig_dtype == DType::Float16 ||
                               orig_dtype == DType::BFloat16);
    Variable input_c = needs_upcast
        ? tenzor::nn::variable_cast(input, DType::Float32)
        : input;
    Variable target_c = needs_upcast
        ? tenzor::nn::variable_cast(target, DType::Float32)
        : target;
    Variable var_c = needs_upcast
        ? tenzor::nn::variable_cast(var, DType::Float32)
        : var;

    auto var_clamped = clamp(var_c, static_cast<float>(eps_), std::numeric_limits<float>::max());

    auto diff = input_c - target_c;
    auto half = scalar_var(0.5f, diff);
    auto loss = half * (log(var_clamped) + diff * diff / var_clamped);

    if (full_) {
        auto log_2pi = scalar_var(static_cast<float>(0.5 * std::log(2.0 * M_PI)), loss);
        loss = loss + log_2pi;
    }

    auto red_str = reduction_to_string(reduction_);
    Variable reduced = apply_reduction(loss, red_str);
    if (needs_upcast) {
        reduced = tenzor::nn::variable_cast(reduced, orig_dtype);
    }
    return reduced;
}

auto gaussian_nll_loss(const Variable& input, const Variable& target,
                       const Variable& var,
                       bool full, double eps, Reduction reduction) -> Variable {
    GaussianNLLLoss loss_fn(full, eps, reduction);
    return loss_fn.forward(input, target, var);
}

// =========================================================================
// MultiLabelMarginLoss
// =========================================================================

MultiLabelMarginLoss::MultiLabelMarginLoss(Reduction reduction)
    : reduction_(reduction) {}

// ---------------------------------------------------------------------------
// MultiLabelMarginLossBackward — stores the raw per-element gradient
// d(sum_of_per_sample_losses)/d(input), shape (N, C). The forward computes
// it alongside the loss. Reduction scaling is applied by attaching the
// built-in mean/sum autograd ops after the per-sample loss Variable.
// ---------------------------------------------------------------------------
class MultiLabelMarginLossBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override {
        (void)inputs;
        return {};
    }

    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override {
        // grad_outputs[0]: d(loss_reduced)/d(per_sample_loss), shape (N,).
        // We owe d(loss_reduced)/d(input), shape (N, C):
        //     grad_input[b, c] = grad_outputs[0][b] * raw_grad_[b, c]
        auto g_cpu = grad_outputs[0].to(Device::cpu()).to(DType::Float32).contiguous();
        const float* g = g_cpu.data<float>();
        const int64_t N = raw_grad_.shape()[0];
        const int64_t C = raw_grad_.shape()[1];

        Tensor out({N, C}, DType::Float32, Device::cpu());
        const float* r = raw_grad_.data<float>();
        float* o = out.data<float>();
        for (int64_t b = 0; b < N; ++b) {
            const float gb = g[b];
            for (int64_t c = 0; c < C; ++c) {
                o[b * C + c] = gb * r[b * C + c];
            }
        }
        return {out.to(orig_dtype_).to(orig_device_)};
    }

    auto name() const -> std::string override { return "MultiLabelMarginLossBackward"; }

    // MultiLabelMarginLoss is a sum of max(0, 1 - x_y + x_i) terms — piecewise-
    // linear in the input. Its second derivative is structurally zero at every
    // differentiable point, so the stub correctly represents the op for
    // create_graph=true instead of throwing.
    TENZOR_HIGHER_ORDER_STRUCTURAL_ZERO_STUB()

    Tensor raw_grad_;              // (N, C) float32 on CPU
    DType orig_dtype_ = DType::Float32;
    Device orig_device_ = Device::cpu();
};

auto MultiLabelMarginLoss::forward(const Variable& input, const Tensor& target) -> Variable {
    // Multi-label margin loss. target is (N, C) with class indices and -1 sentinels.
    //   per_sample_loss[b] = (1/C) * sum_{j: y_j >= 0} sum_{k: k not in targets}
    //                         max(0, 1 - (x[b, y_j] - x[b, k]))
    //
    // For active margin terms (margin > 0):
    //   d(sample_loss)/d(x[b, y_j]) += -1/C
    //   d(sample_loss)/d(x[b, k])   += +1/C
    auto input_t = input.tensor();
    int64_t batch_size = input_t.shape()[0];
    int64_t n_classes = input_t.shape()[1];

    // Compute on CPU for correctness (rarely in the hot path).
    auto cpu_input = input_t.to(Device::cpu()).to(DType::Float32);
    auto cpu_target = target.to(Device::cpu());
    auto cpu_loss = tenzor::zeros({batch_size}, DType::Float32, Device::cpu());
    auto cpu_grad = tenzor::zeros({batch_size, n_classes}, DType::Float32, Device::cpu());

    const float* in_data = cpu_input.data<float>();
    const int64_t* tgt_data = cpu_target.data<int64_t>();
    float* loss_data = cpu_loss.data<float>();
    float* grad_data = cpu_grad.data<float>();

    const float inv_C = 1.0f / static_cast<float>(n_classes);

    for (int64_t b = 0; b < batch_size; b++) {
        // Build a bitmap of which classes are targets for this sample.
        std::vector<bool> is_target(n_classes, false);
        for (int64_t t = 0; t < n_classes; t++) {
            int64_t yt = tgt_data[b * n_classes + t];
            if (yt < 0) break;
            if (yt >= 0 && yt < n_classes) is_target[yt] = true;
        }

        float sample_loss = 0.0f;
        for (int64_t j = 0; j < n_classes; j++) {
            int64_t y_j = tgt_data[b * n_classes + j];
            if (y_j < 0) break;
            float x_yj = in_data[b * n_classes + y_j];
            for (int64_t k = 0; k < n_classes; k++) {
                if (is_target[k]) continue;
                float margin = 1.0f - (x_yj - in_data[b * n_classes + k]);
                if (margin > 0.0f) {
                    sample_loss += margin;
                    grad_data[b * n_classes + y_j] -= inv_C;
                    grad_data[b * n_classes + k]   += inv_C;
                }
            }
        }
        loss_data[b] = sample_loss * inv_C;
    }

    const Device original_device = input_t.device();
    const DType  original_dtype  = input_t.dtype();
    auto per_sample_tensor = cpu_loss.to(original_dtype).to(original_device);

    const bool needs_grad = input.requires_grad() && is_grad_enabled();
    Variable loss_var(per_sample_tensor, needs_grad);

    if (needs_grad) {
        auto grad_fn = std::make_shared<MultiLabelMarginLossBackward>();
        grad_fn->raw_grad_ = cpu_grad;
        grad_fn->orig_dtype_ = original_dtype;
        grad_fn->orig_device_ = original_device;

        std::vector<std::shared_ptr<Function>> next_funcs;
        if (input.grad_fn()) next_funcs.push_back(input.grad_fn());
        grad_fn->set_next_functions(next_funcs);
        grad_fn->set_input_variables({input});

        loss_var.set_grad_fn(grad_fn);
    }

    // Reduction over the per-sample Variable — uses the standard autograd mean/sum
    // which will multiply the upstream scalar gradient back into (N,), meeting
    // our custom grad_fn at the right shape.
    switch (reduction_) {
        case Reduction::Mean: return mean(loss_var);
        case Reduction::Sum:  return sum(loss_var);
        default:              return loss_var;
    }
}

auto multilabel_margin_loss(const Variable& input, const Tensor& target,
                            Reduction reduction) -> Variable {
    MultiLabelMarginLoss loss_fn(reduction);
    return loss_fn.forward(input, target);
}

} // namespace tenzor::nn
