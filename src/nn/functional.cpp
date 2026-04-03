/**
 * @file functional.cpp
 * @brief Implementations for non-trivial nn::functional operations
 *
 * Conv, pooling, normalization, dropout, and padding functional wrappers
 * that dispatch directly to backend kernels.
 */

#include "tenzor/nn/functional.hpp"
#include "tenzor/backend/fast_dispatch.hpp"
#include "tenzor/backend/op_attributes.hpp"
#include "tenzor/ops/op_id.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/transform.hpp"
#include "tenzor/ops/reduction.hpp"
#include "tenzor/ops/indexing.hpp"
#include "tenzor/autograd/function.hpp"
#include <stdexcept>
#include <cmath>

namespace tenzor::nn::functional {

// ============================================================================
// Convolution
// ============================================================================

auto conv2d(const Variable& input, const Variable& weight,
            const std::optional<Variable>& bias,
            std::pair<int64_t, int64_t> stride,
            std::pair<int64_t, int64_t> padding,
            std::pair<int64_t, int64_t> dilation,
            int64_t groups) -> Variable {
    auto input_shape = input.shape();
    if (input_shape.size() != 4) {
        throw std::invalid_argument(
            "F::conv2d expects 4D input [N, C_in, H, W], got " +
            std::to_string(input_shape.size()) + "D");
    }

    std::vector<Tensor> inputs_vec = {input.tensor(), weight.tensor()};
    if (bias.has_value()) {
        inputs_vec.push_back(bias->tensor());
    }

    NewOpAttributes attrs;
    attrs.set(AttrKey::Stride, stride.first);
    attrs.set(AttrKey::Padding, padding.first);
    attrs.set(AttrKey::Dilation, dilation.first);
    attrs.set(AttrKey::StrideH, stride.first);
    attrs.set(AttrKey::StrideW, stride.second);
    attrs.set(AttrKey::PaddingH, padding.first);
    attrs.set(AttrKey::PaddingW, padding.second);
    attrs.set(AttrKey::DilationH, dilation.first);
    attrs.set(AttrKey::DilationW, dilation.second);
    attrs.set(AttrKey::Groups, groups);

    auto result = dispatch_to_device(OpId::Conv2dForward,
        input.tensor().device().type, inputs_vec, attrs);

    bool requires_grad = input.requires_grad() || weight.requires_grad() ||
                         (bias.has_value() && bias->requires_grad());
    return Variable(result[0], requires_grad);
}

// ============================================================================
// Pooling
// ============================================================================

auto max_pool2d(const Variable& input,
                std::pair<int64_t, int64_t> kernel_size,
                std::pair<int64_t, int64_t> stride,
                std::pair<int64_t, int64_t> padding) -> Variable {
    if (input.shape().size() != 4) {
        throw std::invalid_argument(
            "F::max_pool2d expects 4D input [N, C, H, W]");
    }

    // Default stride = kernel_size
    if (stride.first < 0) stride.first = kernel_size.first;
    if (stride.second < 0) stride.second = kernel_size.second;

    NewOpAttributes attrs;
    attrs.set(AttrKey::KernelSize, kernel_size.first);
    attrs.set(AttrKey::Stride, stride.first);
    attrs.set(AttrKey::Padding, padding.first);

    std::vector<Tensor> inputs_vec = {input.tensor()};
    auto result = dispatch_to_device(OpId::MaxPool2dForward,
        input.tensor().device().type, inputs_vec, attrs);

    return Variable(result[0], input.requires_grad());
}

auto avg_pool2d(const Variable& input,
                std::pair<int64_t, int64_t> kernel_size,
                std::pair<int64_t, int64_t> stride,
                std::pair<int64_t, int64_t> padding) -> Variable {
    if (input.shape().size() != 4) {
        throw std::invalid_argument(
            "F::avg_pool2d expects 4D input [N, C, H, W]");
    }

    // Default stride = kernel_size
    if (stride.first < 0) stride.first = kernel_size.first;
    if (stride.second < 0) stride.second = kernel_size.second;

    NewOpAttributes attrs;
    attrs.set(AttrKey::KernelSize, kernel_size.first);
    attrs.set(AttrKey::Stride, stride.first);
    attrs.set(AttrKey::Padding, padding.first);

    std::vector<Tensor> inputs_vec = {input.tensor()};
    auto result = dispatch_to_device(OpId::AvgPool2dForward,
        input.tensor().device().type, inputs_vec, attrs);

    return Variable(result[0], input.requires_grad());
}

auto adaptive_avg_pool2d(const Variable& input,
                         std::pair<int64_t, int64_t> output_size) -> Variable {
    auto shape = input.shape();
    if (shape.size() != 4) {
        throw std::invalid_argument(
            "F::adaptive_avg_pool2d expects 4D input [N, C, H, W]");
    }

    int64_t in_h = shape[2];
    int64_t in_w = shape[3];
    int64_t out_h = output_size.first;
    int64_t out_w = output_size.second;

    // Compute adaptive kernel_size and stride
    int64_t kernel_h = in_h - (out_h - 1) * (in_h / out_h);
    int64_t kernel_w = in_w - (out_w - 1) * (in_w / out_w);
    int64_t stride_h = in_h / out_h;
    int64_t stride_w = in_w / out_w;

    // Use kernel_size = ceil(in/out), stride = floor(in/out) for exact output size
    if (out_h > 0) {
        stride_h = in_h / out_h;
        kernel_h = in_h - (out_h - 1) * stride_h;
    }
    if (out_w > 0) {
        stride_w = in_w / out_w;
        kernel_w = in_w - (out_w - 1) * stride_w;
    }

    return avg_pool2d(input, {kernel_h, kernel_w}, {stride_h, stride_w}, {0, 0});
}

// ============================================================================
// Normalization
// ============================================================================

auto batch_norm(const Variable& input,
                const Tensor& running_mean,
                const Tensor& running_var,
                const std::optional<Variable>& weight,
                const std::optional<Variable>& bias,
                bool training,
                double momentum,
                double eps) -> Variable {
    if (training) {
        // Training mode: compute batch statistics using dispatch
        auto input_t = input.tensor();

        std::vector<Tensor> norm_inputs = {input_t};
        NewOpAttributes mean_var_attrs;
        auto stats = dispatch(OpId::BatchNorm2dMeanVar, norm_inputs, mean_var_attrs);

        Tensor batch_mean = stats[0];
        Tensor batch_var = stats[1];

        // Normalize
        std::vector<Tensor> fwd_inputs = {input_t, batch_mean, batch_var};
        if (weight.has_value()) fwd_inputs.push_back(weight->tensor());
        if (bias.has_value()) fwd_inputs.push_back(bias->tensor());

        NewOpAttributes fwd_attrs;
        fwd_attrs.set(AttrKey::Eps, eps);
        auto output = dispatch(OpId::BatchNorm2dForwardAffine, fwd_inputs, fwd_attrs);

        return Variable(output[0], input.requires_grad());
    } else {
        // Eval mode: use running statistics
        std::vector<Tensor> inputs_vec = {input.tensor(), running_mean, running_var};
        if (weight.has_value()) inputs_vec.push_back(weight->tensor());
        if (bias.has_value()) inputs_vec.push_back(bias->tensor());

        NewOpAttributes attrs;
        attrs.set(AttrKey::Eps, eps);
        auto output = dispatch(OpId::BatchNorm2dForwardAffine, inputs_vec, attrs);

        return Variable(output[0], input.requires_grad());
    }
}

auto layer_norm(const Variable& input,
                std::vector<int64_t> normalized_shape,
                const std::optional<Variable>& weight,
                const std::optional<Variable>& bias,
                double eps) -> Variable {
    std::vector<Tensor> inputs_vec = {input.tensor()};
    if (weight.has_value()) inputs_vec.push_back(weight->tensor());
    if (bias.has_value()) inputs_vec.push_back(bias->tensor());

    // Build normalized_shape as comma-separated string attribute
    std::string shape_str;
    for (size_t i = 0; i < normalized_shape.size(); ++i) {
        if (i > 0) shape_str += ",";
        shape_str += std::to_string(normalized_shape[i]);
    }

    NewOpAttributes attrs;
    attrs.set(AttrKey::NormalizedShape, shape_str);
    attrs.set(AttrKey::Eps, eps);

    auto result = dispatch(OpId::FusedLayerNorm, inputs_vec, attrs);
    return Variable(result[0], input.requires_grad());
}

// ============================================================================
// Dropout
// ============================================================================

auto dropout(const Variable& input, double p, bool training) -> Variable {
    if (!training || p == 0.0) {
        return input;
    }
    if (p == 1.0) {
        auto s = input.tensor().shape();
        std::vector<int64_t> sv(s.begin(), s.end());
        return Variable(zeros(sv, input.dtype(), input.tensor().device()),
                        input.requires_grad());
    }

    // Generate Bernoulli mask via rand > p, then apply inverted dropout
    auto shape_span = input.tensor().shape();
    std::vector<int64_t> shape_vec(shape_span.begin(), shape_span.end());
    auto rand_t = tenzor::rand(shape_vec, input.dtype(), input.tensor().device());
    auto threshold = tenzor::full(shape_vec, static_cast<float>(p),
                                  input.dtype(), input.tensor().device());
    auto mask_bool = tenzor::gt(rand_t, threshold);
    auto ones_t = tenzor::ones(shape_vec, input.dtype(), input.tensor().device());
    auto zeros_t = tenzor::zeros(shape_vec, input.dtype(), input.tensor().device());
    auto mask = tenzor::where(mask_bool, ones_t, zeros_t);

    // Inverted dropout: scale by 1/(1-p)
    float scale = static_cast<float>(1.0 / (1.0 - p));
    auto scale_t = tenzor::full(shape_vec, scale, input.dtype(), input.tensor().device());
    auto output = tenzor::mul(tenzor::mul(input.tensor(), mask), scale_t);

    return Variable(output, input.requires_grad());
}

// ============================================================================
// Group Normalization
// ============================================================================

auto group_norm(const Variable& input, int64_t num_groups,
                const std::optional<Variable>& weight,
                const std::optional<Variable>& bias,
                double eps) -> Variable {
    std::vector<Tensor> inputs_vec = {input.tensor()};
    if (weight.has_value()) inputs_vec.push_back(weight->tensor());
    if (bias.has_value()) inputs_vec.push_back(bias->tensor());

    NewOpAttributes attrs;
    attrs.set(AttrKey::Groups, num_groups);
    attrs.set(AttrKey::Eps, eps);

    auto result = dispatch(OpId::GroupNorm, inputs_vec, attrs);
    return Variable(result[0], input.requires_grad());
}

// ============================================================================
// Instance Normalization
// ============================================================================

auto instance_norm(const Variable& input,
                   const std::optional<Tensor>& running_mean,
                   const std::optional<Tensor>& running_var,
                   const std::optional<Variable>& weight,
                   const std::optional<Variable>& bias,
                   bool training,
                   double momentum,
                   double eps) -> Variable {
    std::vector<Tensor> inputs_vec = {input.tensor()};
    if (weight.has_value()) inputs_vec.push_back(weight->tensor());
    if (bias.has_value()) inputs_vec.push_back(bias->tensor());

    NewOpAttributes attrs;
    attrs.set(AttrKey::Eps, eps);

    auto result = dispatch(OpId::InstanceNorm, inputs_vec, attrs);
    return Variable(result[0], input.requires_grad());
}

// ============================================================================
// Embedding
// ============================================================================

auto embedding(const Tensor& input, const Variable& weight) -> Variable {
    std::vector<Tensor> inputs_vec = {weight.tensor(), input};
    auto result = dispatch(OpId::Embedding, inputs_vec, {});
    return Variable(result[0], weight.requires_grad());
}

// ============================================================================
// Interpolation
// ============================================================================

auto interpolate(const Variable& input,
                 std::pair<int64_t, int64_t> size,
                 const std::string& mode,
                 bool align_corners) -> Variable {
    std::vector<Tensor> inputs_vec = {input.tensor()};

    std::string size_str = std::to_string(size.first) + "," + std::to_string(size.second);
    NewOpAttributes attrs;
    attrs.set(AttrKey::OutputSize, size_str);
    attrs.set(AttrKey::Mode, mode);
    attrs.set(AttrKey::AlignCorners, align_corners);

    auto result = dispatch(OpId::Interpolate, inputs_vec, attrs);
    return Variable(result[0], input.requires_grad());
}

// ============================================================================
// NLL Loss
// ============================================================================

auto nll_loss(const Variable& input, const Tensor& target,
              Reduction reduction) -> Variable {
    // NLL loss: -sum(input[i, target[i]]) / N
    auto input_t = input.tensor();
    auto shape = input_t.shape();
    int64_t N = shape[0];

    // Gather the log-probabilities at target indices
    auto gathered = tenzor::gather(input_t, 1,
        tenzor::reshape(target, {N, 1}));
    auto loss = tenzor::neg(tenzor::reshape(gathered, {N}));

    if (reduction == Reduction::Mean) return Variable(tenzor::mean(loss), input.requires_grad());
    if (reduction == Reduction::Sum) return Variable(tenzor::sum(loss), input.requires_grad());
    return Variable(loss, input.requires_grad());
}

// ============================================================================
// Smooth L1 Loss
// ============================================================================

auto smooth_l1_loss(const Variable& input, const Variable& target,
                    Reduction reduction, double beta) -> Variable {
    // smooth_l1: 0.5 * x^2 / beta if |x| < beta, else |x| - 0.5 * beta
    auto diff = input - target;
    auto abs_diff = tenzor::abs(diff);

    auto beta_t = tenzor::full(
        std::vector<int64_t>(abs_diff.shape().begin(), abs_diff.shape().end()),
        static_cast<float>(beta), abs_diff.dtype(), abs_diff.tensor().device());
    auto half_beta = tenzor::full(
        std::vector<int64_t>(abs_diff.shape().begin(), abs_diff.shape().end()),
        static_cast<float>(0.5 * beta), abs_diff.dtype(), abs_diff.tensor().device());

    // condition: |x| < beta
    auto cond = tenzor::lt(abs_diff.tensor(), beta_t);

    // quadratic branch: 0.5 * x^2 / beta
    auto quad = diff * diff * static_cast<float>(0.5 / beta);

    // linear branch: |x| - 0.5 * beta
    auto lin = abs_diff - Variable(half_beta, false);

    auto loss = Variable(
        tenzor::where(cond, quad.tensor(), lin.tensor()),
        input.requires_grad());

    if (reduction == Reduction::Mean) return tenzor::mean(loss);
    if (reduction == Reduction::Sum) return tenzor::sum(loss);
    return loss;
}

// ============================================================================
// Cosine Similarity
// ============================================================================

auto cosine_similarity(const Variable& x1, const Variable& x2,
                       int64_t dim, double eps) -> Variable {
    // cos_sim = (x1 . x2) / (||x1|| * ||x2|| + eps)
    auto dot = tenzor::sum(x1 * x2, dim);
    auto norm1 = tenzor::sqrt(tenzor::sum(x1 * x1, dim));
    auto norm2 = tenzor::sqrt(tenzor::sum(x2 * x2, dim));
    auto denom = norm1 * norm2 + static_cast<float>(eps);
    return dot / denom;
}

} // namespace tenzor::nn::functional
