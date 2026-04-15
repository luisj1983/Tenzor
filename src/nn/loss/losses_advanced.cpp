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
    auto one_minus_p_safe = clamp(one_minus_p, 1e-7f, 1.0f);
    auto exponent = gamma_var * log(one_minus_p_safe);
    auto exponent_safe = clamp(exponent, -50.0f, 50.0f);
    Variable modulating_factor = exp(exponent_safe);

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

//==============================================================================
// SoftMarginLoss Implementation
//==============================================================================

SoftMarginLoss::SoftMarginLoss(Reduction reduction) : reduction_(reduction) {}

auto SoftMarginLoss::forward(const Variable& input, const Variable& target) -> Variable {
    // loss = log(1 + exp(-y * x))
    // Use log1p(exp(-y*x)) for numerical stability when -y*x is large negative
    auto neg_yx = neg(input * target);
    // softplus: log(1 + exp(x)) = max(x,0) + log(1 + exp(-|x|))
    // But since we need autograd, use: log(1 + exp(x))
    auto one = scalar_var(1.0f, neg_yx);
    auto loss = log(one + exp(neg_yx));

    auto red_str = reduction_to_string(reduction_);
    return apply_reduction(loss, red_str);
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
    Variable loss;
    if (log_input_) {
        // loss = exp(input) - target * input
        loss = exp(input) - target * input;
    } else {
        // loss = input - target * log(input + eps)
        auto eps_var = scalar_var(static_cast<float>(eps_), input);
        loss = input - target * log(input + eps_var);
    }

    if (full_) {
        // Add Stirling approximation: target * log(target) - target + 0.5 * log(2 * pi * target)
        // Simplified: use only the dominant term target * log(target) - target for target > 1
        auto eps_var = scalar_var(static_cast<float>(eps_), target);
        auto target_safe = clamp(target, static_cast<float>(eps_), std::numeric_limits<float>::max());
        auto stirling = target_safe * log(target_safe) - target_safe;
        // Add 0.5 * log(2*pi*target)
        auto half = scalar_var(0.5f, target);
        auto two_pi = scalar_var(static_cast<float>(2.0 * M_PI), target);
        stirling = stirling + half * log(two_pi * target_safe);
        loss = loss + stirling;
    }

    auto red_str = reduction_to_string(reduction_);
    return apply_reduction(loss, red_str);
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
    // Compute cosine similarity along last dimension
    // cos_sim = sum(x1 * x2, dim=-1) / (norm(x1) * norm(x2))
    auto product = input1 * input2;
    int64_t last_dim = static_cast<int64_t>(input1.shape().size()) - 1;
    auto dot = sum(product, last_dim, false);

    auto norm1_sq = sum(input1 * input1, last_dim, false);
    auto norm2_sq = sum(input2 * input2, last_dim, false);
    auto eps_var = scalar_var(1e-8f, norm1_sq);
    auto norms = sqrt(norm1_sq + eps_var) * sqrt(norm2_sq + eps_var);
    auto cos_sim = dot / norms;

    // loss = (1 - cos_sim) if y = 1
    // loss = max(0, cos_sim - margin) if y = -1
    auto one = scalar_var(1.0f, cos_sim);
    auto pos_mask = (target + one) * scalar_var(0.5f, cos_sim);
    auto neg_mask = (one - pos_mask);

    auto margin_var = scalar_var(static_cast<float>(margin_), cos_sim);
    auto pos_loss = one - cos_sim;
    auto neg_loss = relu(cos_sim - margin_var);
    auto loss = pos_mask * pos_loss + neg_mask * neg_loss;

    auto red_str = reduction_to_string(reduction_);
    return apply_reduction(loss, red_str);
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
    auto diff_pos = anchor - positive;
    auto diff_neg = anchor - negative;

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
        auto diff_pn = positive - negative;
        Variable d_pn;
        if (p_ == 2.0) {
            d_pn = sqrt(sum(diff_pn * diff_pn, last_dim, false) + eps_var);
        } else {
            d_pn = sum(abs(diff_pn), last_dim, false);
        }
        // d_neg = min(d_neg, d_pn) via: d_neg - relu(d_neg - d_pn)
        d_neg = d_neg - relu(d_neg - d_pn);
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

    auto one = scalar_var(1.0f, input);
    auto neg_input = neg(input);

    // softplus(x) = log(1 + exp(x))
    auto sp_pos = log(one + exp(input));     // log(1 + exp(x)) = softplus(x)
    auto sp_neg = log(one + exp(neg_input)); // log(1 + exp(-x)) = softplus(-x)

    // log(sigma(x)) = -softplus(-x), log(1-sigma(x)) = -softplus(x)
    auto loss_per_element = target * sp_neg + (one - target) * sp_pos;

    // Average over class dimension (last dim)
    int64_t class_dim = static_cast<int64_t>(input.shape().size()) - 1;
    auto loss_per_sample = mean(loss_per_element, class_dim, false);

    auto red_str = reduction_to_string(reduction_);
    return apply_reduction(loss_per_sample, red_str);
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

    // Extract the correct-class scores: x[y] for each sample
    // Build one-hot from target, then use it to extract correct scores
    Tensor target_cpu = target.to(Device::cpu()).to(DType::Int64);
    const int64_t* target_data = target_cpu.data<int64_t>();

    // Create a tensor to hold x[y] for each sample, broadcast to (N, C)
    // We'll build this using indexing: for each sample i, correct_score[i] = input[i, target[i]]
    auto input_t = input.tensor().to(Device::cpu()).to(DType::Float32).contiguous();
    const float* input_data = input_t.data<float>();

    auto correct_scores_t = Tensor({N, 1}, DType::Float32, Device::cpu());
    float* cs_data = correct_scores_t.data<float>();
    for (int64_t i = 0; i < N; ++i) {
        int64_t y = target_data[i];
        cs_data[i] = input_data[i * C + y];
    }

    // Transfer to original device
    auto correct_scores = Variable(
        correct_scores_t.to(input.tensor().dtype()).to(input.tensor().device()),
        false);

    // margin - x[y] + x[j] for all j
    auto margin_var = scalar_var(static_cast<float>(margin_), input);
    auto diff = margin_var - correct_scores + input;
    auto hinge = relu(diff);

    // Zero out the correct class contribution
    // Create one-hot mask
    auto one_hot_t = zeros({N, C}, input.tensor().dtype(), input.tensor().device());
    auto one_hot_cpu = one_hot_t.to(Device::cpu()).to(DType::Float32);
    float* oh_data = one_hot_cpu.data<float>();
    for (int64_t i = 0; i < N; ++i) {
        oh_data[i * C + target_data[i]] = 1.0f;
    }
    auto one_hot = Variable(
        one_hot_cpu.to(input.tensor().dtype()).to(input.tensor().device()),
        false);

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

    auto eps_var = scalar_var(static_cast<float>(eps_), var);
    auto var_clamped = clamp(var, static_cast<float>(eps_), std::numeric_limits<float>::max());

    auto diff = input - target;
    auto half = scalar_var(0.5f, diff);
    auto loss = half * (log(var_clamped) + diff * diff / var_clamped);

    if (full_) {
        auto log_2pi = scalar_var(static_cast<float>(0.5 * std::log(2.0 * M_PI)), loss);
        loss = loss + log_2pi;
    }

    auto red_str = reduction_to_string(reduction_);
    return apply_reduction(loss, red_str);
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

auto MultiLabelMarginLoss::forward(const Variable& input, const Tensor& target) -> Variable {
    // Multi-label margin loss implementation using tensor ops
    // target is (N, C) with class indices and -1 sentinels
    // For each sample: sum over targets y_j, sum over non-targets k:
    //   max(0, 1 - (x[y_j] - x[k])) / C

    auto input_t = input.tensor();
    int64_t batch_size = input_t.shape()[0];
    int64_t n_classes = input_t.shape()[1];

    // Compute on CPU for correctness (this op is rarely in the hot path)
    auto cpu_input = input_t.to(Device::cpu());
    auto cpu_target = target.to(Device::cpu());
    auto cpu_loss = tenzor::zeros({batch_size}, DType::Float32, Device::cpu());

    const float* in_data = cpu_input.data<float>();
    const int64_t* tgt_data = cpu_target.data<int64_t>();
    float* loss_data = cpu_loss.data<float>();

    for (int64_t b = 0; b < batch_size; b++) {
        float sample_loss = 0.0f;
        for (int64_t j = 0; j < n_classes; j++) {
            int64_t y_j = tgt_data[b * n_classes + j];
            if (y_j < 0) break;
            float x_yj = in_data[b * n_classes + y_j];
            for (int64_t k = 0; k < n_classes; k++) {
                bool is_target = false;
                for (int64_t t = 0; t < n_classes; t++) {
                    int64_t yt = tgt_data[b * n_classes + t];
                    if (yt < 0) break;
                    if (yt == k) { is_target = true; break; }
                }
                if (is_target) continue;
                float margin = 1.0f - (x_yj - in_data[b * n_classes + k]);
                if (margin > 0.0f) sample_loss += margin;
            }
        }
        loss_data[b] = sample_loss / static_cast<float>(n_classes);
    }

    auto result = cpu_loss.to(input_t.device());
    auto loss_var = Variable(result, input.requires_grad());

    switch (reduction_) {
        case Reduction::Mean: return mean(loss_var);
        case Reduction::Sum: return sum(loss_var);
        default: return loss_var;
    }
}

auto multilabel_margin_loss(const Variable& input, const Tensor& target,
                            Reduction reduction) -> Variable {
    MultiLabelMarginLoss loss_fn(reduction);
    return loss_fn.forward(input, target);
}

} // namespace tenzor::nn
