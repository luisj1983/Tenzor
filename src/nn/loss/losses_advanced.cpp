/**
 * @file losses_advanced.cpp
 * @brief Implementation of advanced loss functions
 */

#include "tenzor/nn/loss/losses.hpp"
#include "tenzor/nn/loss/contrastive.hpp"
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
            // When the caller does not pass an explicit batch size, infer it
            // from the unreduced loss's leading dimension. Previously this fell
            // through to mean(loss), so "batchmean" silently behaved as full
            // "mean" (a different scalar) at every call site that omitted the
            // batch_size argument.
            int64_t bs = batch_size;
            if (bs <= 0) {
                const auto& shp = loss.shape();
                bs = (!shp.empty()) ? shp[0] : 0;
            }
            if (bs > 0) {
                // Division by batch_size using scalar Variable
                auto scale = 1.0f / static_cast<float>(bs);
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
            const auto& shp = input_f32.shape();
            int64_t batch_size = (!shp.empty()) ? shp[0] : 0;
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
            const auto& shp = input_f32.shape();
            int64_t batch_size = (!shp.empty()) ? shp[0] : 0;
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

    // Use the original numerically-stable log_softmax output for the log(p_t)
    // term. Recomputing log(exp(log_softmax) clamped) round-trips through exp
    // and a re-log, introducing a small systematic bias (and an avoidable
    // double-log). probs_clamped is only needed for the (1-p_t)^gamma
    // modulating factor, never for log(p_t).

    // Compute focal loss: -alpha * (1-p)^gamma * log(p) * target
    auto alpha_var = scalar_var(static_cast<float>(alpha_), modulating_factor);
    auto loss_unreduced = neg(modulating_factor * alpha_var * log_probs * target_f32);

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

auto DiceLoss::forward(const Variable& input_in, const Variable& target_in) -> Variable {
    // Validate: input must be at least 2D (batch + channel dimensions)
    if (input_in.shape().size() < 2) {
        throw std::invalid_argument(
            "DiceLoss: input must be at least 2D (batch + channels), got " +
            std::to_string(input_in.shape().size()) + "D");
    }

    // Widen Float16/BFloat16 to Float32: the spatial sums below accumulate over
    // H*W elements and overflow half precision (max 65504) for large maps,
    // giving inf/inf = NaN. Every sibling advanced loss widens the same way.
    const DType orig_dtype = input_in.tensor().dtype();
    const bool widen = (orig_dtype == DType::Float16 || orig_dtype == DType::BFloat16);
    Variable input = widen ? tenzor::nn::variable_cast(input_in, DType::Float32) : input_in;
    Variable target = widen ? tenzor::nn::variable_cast(target_in, DType::Float32) : target_in;

    // Dice = 1 - (2 * |X ∩ Y| + smooth) / (|X| + |Y| + smooth)
    // For differentiable version: intersection = sum(input * target)

    // Flatten spatial dimensions but keep batch and channel dims
    // input shape: (N, C, H, W) -> compute dice per channel per batch

    // Compute intersection, input_sum, and target_sum
    auto intersection = input * target;
    // Sum over the SPATIAL dims only, keeping batch (dim 0) AND channel (dim 1)
    // so Dice is computed per channel per sample (as documented above and as
    // standard multi-class Dice does). Previously this reduced down to dim 1,
    // collapsing the channel dimension into a single pooled foreground/background
    // Dice — losing per-class weighting and merging per-class gradients.
    int64_t ndim = static_cast<int64_t>(input.shape().size());
    Variable intersection_sum = intersection;
    Variable input_sum_v = input;
    Variable target_sum_v = target;
    // Reduce spatial dims from last down to 2 (skip batch dim 0 and channel dim 1).
    // For a 2-D (N, C) input the loop is a no-op, giving per-(N,C) Dice directly.
    for (int64_t d = ndim - 1; d >= 2; --d) {
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

    auto reduced = apply_reduction(loss, reduction_);
    return widen ? tenzor::nn::variable_cast(reduced, orig_dtype) : reduced;
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

    int64_t N = lp_shape[1];

    const Device original_device = log_probs.tensor().device();
    const DType original_dtype = log_probs.tensor().dtype();
    const bool needs_grad = log_probs.requires_grad() && is_grad_enabled();

    // -----------------------------------------------------------------------
    // JIT-R092: dispatch to the backend's native CTC kernel (OpId::
    // CTCLossForward) unconditionally, on every device including CPU. The
    // kernel returns {loss_per_sample (N,), raw_grad (T, N, C)} on the same
    // device. Reductions and any necessary scaling are done with device-
    // native tensor ops — no host round-trip for the heavy DP arithmetic.
    // Previously this branch was gated to non-CPU devices only, and CPU fell
    // through to a ~150-line raw-pointer-loop reimplementation below (zero
    // dispatch() calls, 100% invisible to the JIT tracer — silently froze
    // the trace-dummy's loss/gradient on any traced CPU call). The CPU
    // kernel (src/backends/cpu/kernels/ctc.cpp) is already registered for
    // dispatch parity and computes the identical algorithm; keeping the
    // Float32-only staging below (matching this branch's prior GPU-only
    // behavior exactly) rather than also passing through its Float64
    // support, since that widening hasn't been verified against every
    // GPU backend's own kernel in this CPU-only environment.
    {
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

        // PyTorch reduction="mean" is mean_n( loss_n / max(target_len_n, 1) ),
        // NOT sum(loss)/sum(target_len). Divide each sample's loss by its OWN
        // target length, then average over N. The per-sample 1/len_n factor is
        // baked into the gradient rows (a single scalar scale_ can't express a
        // per-sample divisor); scale_ then carries only the uniform 1/N factor.
        float scale = 1.0f;
        Tensor scaled_losses = losses_dev;
        if (reduction_ == "mean") {
            Tensor len_clamped = tenzor::clamp_min(tl_dev.to(DType::Float32), 1.0);  // (N,)
            Tensor inv_len = tenzor::reciprocal(len_clamped);                        // (N,)
            scaled_losses = tenzor::mul(losses_dev, inv_len);                        // (N,)
            // Bake the per-sample factor into the captured gradient: (T,N,C)*(1,N,1).
            Tensor inv_len_bcast = tenzor::reshape(inv_len, {1, N, 1});
            raw_grad_dev = tenzor::mul(raw_grad_dev, inv_len_bcast);
            if (N > 0) scale = 1.0f / static_cast<float>(N);
        }
        // Reduction on the device: sum, then scalar-multiply by `scale`
        // (1.0 for "sum"; 1/N for "mean" — the per-sample 1/len_n is already
        // folded into scaled_losses and raw_grad_dev above).
        Tensor total = tenzor::sum(scaled_losses);
        if (scale != 1.0f) {
            total = tenzor::mul(total, static_cast<double>(scale));
        }
        if (total.dtype() != original_dtype) total = total.to(original_dtype);
        Variable out(total, needs_grad);
        attach_grad_fn_dev(out, /*per_sample=*/false, scale);
        return out;
    }

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
    // Numerically stable softplus: log(1 + exp(z)) = relu(z) + log(1 + exp(-|z|))
    // The direct log(1 + exp(z)) overflows to +inf once z > ~88 even in Float32.
    // Mirrors BCEWithLogitsLoss.
    auto one = scalar_var(1.0f, neg_yx);
    auto loss = relu(neg_yx) + log(one + exp(neg(abs(neg_yx))));

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

auto HingeEmbeddingLoss::forward(const Variable& input_in, const Variable& target_in) -> Variable {
    // loss = x  if y = 1
    // loss = max(0, margin - x)  if y = -1
    // Combined: loss = (y == 1) * x + (y == -1) * max(0, margin - x)

    // Widen Float16/BFloat16 to Float32 for consistency with the other advanced
    // losses (the sum reduction stays well-conditioned, but this keeps the whole
    // family on the same precision contract); narrow the result back.
    const DType orig_dtype = input_in.tensor().dtype();
    const bool widen = (orig_dtype == DType::Float16 || orig_dtype == DType::BFloat16);
    Variable input = widen ? tenzor::nn::variable_cast(input_in, DType::Float32) : input_in;
    Variable target = widen ? tenzor::nn::variable_cast(target_in, DType::Float32) : target_in;

    auto one = scalar_var(1.0f, input);
    auto neg_one = scalar_var(-1.0f, input);
    auto margin_var = scalar_var(static_cast<float>(margin_), input);

    // Masks: (target + 1) / 2 gives 1 for y=1, 0 for y=-1
    auto pos_mask = (target + one) * scalar_var(0.5f, input);
    auto neg_mask = (one - pos_mask);

    auto hinge_part = relu(margin_var - input);
    auto loss = pos_mask * input + neg_mask * hinge_part;

    auto red_str = reduction_to_string(reduction_);
    auto reduced = apply_reduction(loss, red_str);
    return widen ? tenzor::nn::variable_cast(reduced, orig_dtype) : reduced;
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
    //
    // audit-10 OO.4: previously `needs_upcast` was driven solely by the
    // input dtype.  When `log_input=true` the loss computes
    // `exp(input) - target * input + …`; if `target` is F16/BF16 but
    // `input` is F32, the product `target * input` was binding the result
    // to the lower precision (broadcast follows the smaller operand's
    // dtype in the math ops here), losing precision for large counts.
    // Widen if EITHER input OR target is reduced precision, and pick the
    // narrower dtype as the cast-back target so the user-visible dtype
    // matches the dominant input.  Mirrors KK.18 / KK.19 / KLDivLoss.
    const DType input_dtype = input.tensor().dtype();
    const DType target_dtype = target.tensor().dtype();
    auto is_reduced = [](DType d) {
        return d == DType::Float16 || d == DType::BFloat16;
    };
    const bool needs_upcast = is_reduced(input_dtype) || is_reduced(target_dtype);
    // Cast the loss back to the WIDER of input/target dtype, preferring input's
    // dtype on a tie (the user supplied it for the prediction). Previously this
    // picked target_dtype whenever input was already high precision, so an F32
    // input with an F16 target narrowed the F32 loss to F16 — a precision loss.
    auto dtype_width = [](DType d) -> int {
        switch (d) {
            case DType::Float64: return 64;
            case DType::Float32: return 32;
            default: return 16;  // Float16 / BFloat16
        }
    };
    const DType orig_dtype = (dtype_width(target_dtype) > dtype_width(input_dtype))
                                 ? target_dtype : input_dtype;
    Variable input_c = (needs_upcast && is_reduced(input_dtype))
        ? tenzor::nn::variable_cast(input, DType::Float32)
        : input;
    Variable target_c = (needs_upcast && is_reduced(target_dtype))
        ? tenzor::nn::variable_cast(target, DType::Float32)
        : target;
    // If only one side was reduced, the other may already be at F32 (or
    // even F64).  If they differ now, lift the lower-precision side up so
    // the loss math runs at uniform F32 (matching the comment above).
    if (input_c.tensor().dtype() != target_c.tensor().dtype()) {
        if (input_c.tensor().dtype() != DType::Float32) {
            input_c = tenzor::nn::variable_cast(input_c, DType::Float32);
        }
        if (target_c.tensor().dtype() != DType::Float32) {
            target_c = tenzor::nn::variable_cast(target_c, DType::Float32);
        }
    }

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
        // Stirling correction: target*log(target) - target + 0.5*log(2*pi*target).
        // PyTorch adds this ONLY where target > 1; for target <= 1 it contributes
        // 0. The previous code applied it to every element (clamped to eps),
        // which adds a spurious strongly-negative 0.5*log(2*pi*target) term for
        // fractional/small targets, diverging from the reference.
        auto target_safe = clamp(target_c, static_cast<float>(eps_), std::numeric_limits<float>::max());
        auto half = scalar_var(0.5f, target_c);
        auto two_pi = scalar_var(static_cast<float>(2.0 * M_PI), target_c);
        auto stirling = target_safe * log(target_safe) - target_safe
                        + half * log(two_pi * target_safe);
        // Detached 0/1 mask = (target > 1); gradient still flows through the
        // stirling term where the mask selects it (torch.where semantics).
        auto one_var = scalar_var(1.0f, target_c);
        Tensor mask_t = tenzor::gt(target_c.tensor(), one_var.tensor())
                            .to(target_c.tensor().dtype());
        Variable mask(mask_t, /*requires_grad=*/false);
        loss = loss + stirling * mask;
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
    // TripletMarginLoss is documented as an alias for TripletLoss. Previously
    // it was a second, hand-maintained implementation that had drifted from
    // TripletLoss: general-p handling (it treated any p != 2 as L1 instead of a
    // true Lp norm), and the two could return different numbers for the same
    // inputs. Delegate to the single TripletLoss implementation (contrastive.cpp)
    // so the "alias" is real — that path implements true Lp distance via
    // local_pairwise_distance and already widens F16/BF16 to Float32 for the
    // distance computation (and casts the loss back).
    TripletLoss impl(margin_, p_, swap_, reduction_);
    return impl.forward(anchor, positive, negative);
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
    auto abs_input = abs(input_c);

    // Numerically stable softplus: log(1 + exp(z)) = relu(z) + log(1 + exp(-|z|)).
    // The direct log(1 + exp(z)) overflows to +inf once z > ~88 even in Float32.
    auto sp_pos = relu(input_c) + log(one + exp(neg(abs_input)));     // softplus(x)
    auto sp_neg = relu(neg_input) + log(one + exp(neg(abs_input)));   // softplus(-x)

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

auto MultiMarginLoss::forward(const Variable& input_in, const Tensor& target) -> Variable {
    // loss = 1/C * sum_{j != y} max(0, margin - x[y] + x[j])^p
    // input: (N, C), target: (N,) with class indices

    if (input_in.tensor().ndim() < 2) {
        throw std::invalid_argument(
            "MultiMarginLoss: expects >=2D logits [N, C, ...]");
    }
    // Widen Float16/BFloat16 to Float32 for consistency with the other advanced
    // losses (target is class indices, so it is not widened); narrow back below.
    const DType orig_dtype = input_in.tensor().dtype();
    const bool widen = (orig_dtype == DType::Float16 || orig_dtype == DType::BFloat16);
    Variable input = widen ? tenzor::nn::variable_cast(input_in, DType::Float32) : input_in;
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
    auto reduced = apply_reduction(loss_per_sample, red_str);
    return widen ? tenzor::nn::variable_cast(reduced, orig_dtype) : reduced;
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
auto MultiLabelMarginLoss::forward(const Variable& input, const Tensor& target) -> Variable {
    // Multi-label margin loss, computed entirely on the input's device (no host
    // round-trip — the previous implementation ran a CPU loop and copied back,
    // violating the backend-agnostic contract). For the target-membership mask mt
    // (mt[b,c] = 1 iff class c is a target of sample b):
    //   per_sample[b] = (1/C) * sum_{t: mt[b,t]=1} sum_{k: mt[b,k]=0}
    //                     relu(1 - x[b,t] + x[b,k])
    // Built from differentiable Variable ops, so autograd derives the backward.
    if (input.tensor().ndim() < 2) {
        throw std::invalid_argument(
            "MultiLabelMarginLoss: expects >=2D logits [N, C, ...]");
    }
    const DType orig_dtype = input.tensor().dtype();
    const Device dev = input.tensor().device();
    const bool needs_upcast = (orig_dtype == DType::Float16 || orig_dtype == DType::BFloat16);
    Variable x = needs_upcast ? tenzor::nn::variable_cast(input, DType::Float32) : input;

    const int64_t N = x.tensor().shape()[0];
    const int64_t C = x.tensor().shape()[1];

    // Target-membership mask mt (N, C) on device — constant w.r.t. input.
    Tensor tgt_f = target.to(DType::Float32).to(dev).contiguous();
    Tensor zeros_nc = tenzor::full({N, C}, 0.0, DType::Float32, dev);
    Tensor valid = tenzor::ge(tgt_f, zeros_nc).to(DType::Float32);   // 1.0 where target >= 0
    Tensor safe = tenzor::clamp_min(tgt_f, 0.0).to(DType::Int64);    // -1 sentinels -> 0
    // one_hot of an (N,C) index tensor yields N*C rows of width C; some backends
    // return it as (N*C, C) and others as (N, C, C). Reshape to (N, C, C) so the
    // downstream broadcasting is backend-independent.
    Tensor oh = tenzor::one_hot(safe, C).to(DType::Float32).reshape({N, C, C});
    Tensor oh_valid = tenzor::mul(oh, valid.reshape({N, C, 1}));     // drop padded positions
    Tensor counts = tenzor::sum(oh_valid, 1, false);                // (N, C)
    Tensor mt = tenzor::clamp_max(counts, 1.0);                      // (N, C) in {0, 1}

    // Pairwise hinge P[b,t,k] = relu(1 - x[b,t] + x[b,k]).
    auto xt = unsqueeze(x, 2);            // (N, C, 1)
    auto xk = unsqueeze(x, 1);            // (N, 1, C)
    auto one = scalar_var(1.0f, x);
    auto P = relu(one - xt + xk);         // (N, C, C)

    // Weight w[b,t,k] = mt[b,t] * (1 - mt[b,k]) (constant, no grad).
    Tensor ones_1c = tenzor::full({N, 1, C}, 1.0, DType::Float32, dev);
    Tensor w = tenzor::mul(mt.reshape({N, C, 1}),
                           tenzor::sub(ones_1c, mt.reshape({N, 1, C})));  // (N, C, C)
    auto weighted = P * Variable(w, false);

    auto per_sample = sum(sum(weighted, 2, false), 1, false);  // (N,)
    per_sample = per_sample * scalar_var(1.0f / static_cast<float>(C), per_sample);

    Variable reduced;
    switch (reduction_) {
        case Reduction::Mean: reduced = mean(per_sample); break;
        case Reduction::Sum:  reduced = sum(per_sample); break;
        case Reduction::BatchMean: {
            // sum(per_sample) / batch_size. per_sample is shape (N,), so the
            // batch size is N. Handled explicitly so BatchMean does not fall
            // through to the unreduced per-sample output (silent wrong shape).
            auto summed = sum(per_sample);
            reduced = (N > 0)
                        ? summed * scalar_var(1.0f / static_cast<float>(N), per_sample)
                        : summed;
            break;
        }
        default:              reduced = per_sample; break;
    }
    if (needs_upcast) reduced = tenzor::nn::variable_cast(reduced, orig_dtype);
    return reduced;
}

auto multilabel_margin_loss(const Variable& input, const Tensor& target,
                            Reduction reduction) -> Variable {
    MultiLabelMarginLoss loss_fn(reduction);
    return loss_fn.forward(input, target);
}

} // namespace tenzor::nn
