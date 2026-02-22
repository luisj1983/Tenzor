/**
 * @file losses_advanced.cpp
 * @brief Implementation of advanced loss functions
 */

#include "tenzor/nn/loss/losses.hpp"
#include "tenzor/nn/activations/activations.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/reduction.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/autograd/ops.hpp"
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

    // Simplified implementation matching manual test
    if (!log_target_) {
        // Standard case: target contains probabilities, input contains log probabilities
        auto target_clamped = clamp(target, 1e-7f, 1.0f);
        auto log_target = log(target_clamped);
        auto diff = log_target - input;
        auto loss_unreduced = target * diff;

        // Apply reduction inline
        if (reduction_ == "none") {
            return loss_unreduced;
        } else if (reduction_ == "mean") {
            return mean(loss_unreduced);
        } else if (reduction_ == "sum") {
            return sum(loss_unreduced);
        } else { // batchmean
            auto summed = sum(loss_unreduced);
            int64_t batch_size = input.shape()[0];
            if (batch_size > 0) {
                auto scale = 1.0f / static_cast<float>(batch_size);
                auto scale_var = scalar_var(scale, summed);
                return summed * scale_var;
            }
            return mean(loss_unreduced);
        }
    } else {
        // log_target case: both inputs are log probabilities
        auto exp_target = exp(target);
        auto diff = target - input;
        auto loss_unreduced = exp_target * diff;

        // Apply reduction inline
        if (reduction_ == "none") {
            return loss_unreduced;
        } else if (reduction_ == "mean") {
            return mean(loss_unreduced);
        } else if (reduction_ == "sum") {
            return sum(loss_unreduced);
        } else { // batchmean
            auto summed = sum(loss_unreduced);
            int64_t batch_size = input.shape()[0];
            if (batch_size > 0) {
                auto scale = 1.0f / static_cast<float>(batch_size);
                auto scale_var = scalar_var(scale, summed);
                return summed * scale_var;
            }
            return mean(loss_unreduced);
        }
    }
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
    // Use log_softmax + exp for autograd-aware softmax computation
    auto log_probs = tenzor::log_softmax(input, 1);  // This has autograd support
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
    Variable modulating_factor = exp(gamma_var * log(one_minus_p));

    // Recompute log_probs from clamped probs for consistency
    auto log_probs_clamped = log(probs_clamped);

    // Compute focal loss: -alpha * (1-p)^gamma * log(p) * target
    auto alpha_var = scalar_var(static_cast<float>(alpha_), modulating_factor);
    auto loss_unreduced = neg(modulating_factor * alpha_var * log_probs_clamped * target);

    // Sum over class dimension
    auto loss_per_sample = sum(loss_unreduced, 1, false);

    return apply_reduction(loss_per_sample, reduction_);
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

    auto diff = input - target;
    auto abs_diff = abs(diff);

    auto clipped = clamp(abs_diff, 0.0f, static_cast<float>(delta_));

    auto half_var = scalar_var(0.5f, clipped);
    auto delta_var = scalar_var(static_cast<float>(delta_), clipped);

    // L = 0.5 * clipped^2 + delta * (|diff| - clipped)
    auto loss_unreduced = half_var * clipped * clipped + delta_var * (abs_diff - clipped);

    return apply_reduction(loss_unreduced, reduction_);
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

    // Ensure CPU computation (CTC is inherently sequential per batch element)
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

    // Create output tensor based on reduction
    Device original_device = log_probs.tensor().device();
    DType original_dtype = log_probs.tensor().dtype();

    if (reduction_ == "none") {
        auto loss_tensor = Tensor({N}, DType::Float32, Device::cpu());
        std::memcpy(loss_tensor.data<float>(), losses.data(), N * sizeof(float));
        return Variable(loss_tensor.to(original_dtype).to(original_device), log_probs.requires_grad());
    }

    float total_loss = 0.0f;
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
        }
    }

    auto loss_tensor = Tensor({1}, DType::Float32, Device::cpu());
    loss_tensor.data<float>()[0] = total_loss;
    return Variable(loss_tensor.to(original_dtype).to(original_device), log_probs.requires_grad());
}

} // namespace tenzor::nn
