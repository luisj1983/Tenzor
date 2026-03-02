#include "tenzor/ops/fused_ops.hpp"
#include "tenzor/backend/fast_dispatch.hpp"
#include "tenzor/backend/op_attributes.hpp"
#include "tenzor/ops/op_id.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/reduction.hpp"
#include <stdexcept>
#include <cmath>

namespace tenzor {
namespace ops {

auto fused_linear_relu(
    const Tensor& input,
    const Tensor& weight,
    const Tensor* bias
) -> Tensor {
    // Validate inputs
    if (input.ndim() < 2) {
        throw std::runtime_error(
            "fused_linear_relu: input must be at least 2D, got " +
            std::to_string(input.ndim()) + "D"
        );
    }

    if (weight.ndim() != 2) {
        throw std::runtime_error(
            "fused_linear_relu: weight must be 2D, got " +
            std::to_string(weight.ndim()) + "D"
        );
    }

    int64_t in_features = input.shape()[input.ndim() - 1];
    int64_t out_features = weight.shape()[0];

    if (weight.shape()[1] != in_features) {
        throw std::runtime_error(
            "fused_linear_relu: input features mismatch: input has " +
            std::to_string(in_features) + ", weight expects " +
            std::to_string(weight.shape()[1])
        );
    }

    if (bias != nullptr && bias->numel() != out_features) {
        throw std::runtime_error(
            "fused_linear_relu: bias size mismatch: expected " +
            std::to_string(out_features) + ", got " +
            std::to_string(bias->numel())
        );
    }

    // Prepare inputs for dispatcher
    std::vector<Tensor> inputs = {input, weight};
    if (bias != nullptr) {
        inputs.push_back(*bias);
    }

    OpAttributes attrs;
    attrs.set(AttrKey::HasBias, bias != nullptr);

    return dispatch(OpId::FusedLinearReLU, inputs, attrs)[0];
}

auto fused_conv2d_relu(
    const Tensor& input,
    const Tensor& weight,
    const Tensor* bias,
    int64_t stride,
    int64_t padding
) -> Tensor {
    // Validate inputs
    if (input.ndim() != 4) {
        throw std::runtime_error(
            "fused_conv2d_relu: input must be 4D (N, C, H, W), got " +
            std::to_string(input.ndim()) + "D"
        );
    }

    if (weight.ndim() != 4) {
        throw std::runtime_error(
            "fused_conv2d_relu: weight must be 4D (C_out, C_in, KH, KW), got " +
            std::to_string(weight.ndim()) + "D"
        );
    }

    int64_t in_channels = input.shape()[1];
    int64_t weight_in_channels = weight.shape()[1];
    int64_t out_channels = weight.shape()[0];

    if (in_channels != weight_in_channels) {
        throw std::runtime_error(
            "fused_conv2d_relu: input channels mismatch: input has " +
            std::to_string(in_channels) + ", weight expects " +
            std::to_string(weight_in_channels)
        );
    }

    if (bias != nullptr && bias->numel() != out_channels) {
        throw std::runtime_error(
            "fused_conv2d_relu: bias size mismatch: expected " +
            std::to_string(out_channels) + ", got " +
            std::to_string(bias->numel())
        );
    }

    // Prepare inputs for dispatcher
    std::vector<Tensor> inputs = {input, weight};
    if (bias != nullptr) {
        inputs.push_back(*bias);
    }

    OpAttributes attrs;
    attrs.set(AttrKey::HasBias, bias != nullptr);
    attrs.set(AttrKey::Stride, stride);
    attrs.set(AttrKey::Padding, padding);

    return dispatch(OpId::FusedConv2dReLU, inputs, attrs)[0];
}

static auto fused_conv2d_dispatch(
    const char* name,
    OpId op_id,
    const Tensor& input,
    const Tensor& weight,
    const Tensor* bias,
    int64_t stride,
    int64_t padding
) -> Tensor {
    if (input.ndim() != 4) {
        throw std::runtime_error(
            std::string(name) + ": input must be 4D (N, C, H, W), got " +
            std::to_string(input.ndim()) + "D"
        );
    }
    if (weight.ndim() != 4) {
        throw std::runtime_error(
            std::string(name) + ": weight must be 4D (C_out, C_in, KH, KW), got " +
            std::to_string(weight.ndim()) + "D"
        );
    }
    int64_t in_channels = input.shape()[1];
    int64_t weight_in_channels = weight.shape()[1];
    int64_t out_channels = weight.shape()[0];
    if (in_channels != weight_in_channels) {
        throw std::runtime_error(
            std::string(name) + ": input channels mismatch: input has " +
            std::to_string(in_channels) + ", weight expects " +
            std::to_string(weight_in_channels)
        );
    }
    if (bias != nullptr && bias->numel() != out_channels) {
        throw std::runtime_error(
            std::string(name) + ": bias size mismatch: expected " +
            std::to_string(out_channels) + ", got " +
            std::to_string(bias->numel())
        );
    }
    std::vector<Tensor> inputs = {input, weight};
    if (bias != nullptr) {
        inputs.push_back(*bias);
    }
    OpAttributes attrs;
    attrs.set(AttrKey::HasBias, bias != nullptr);
    attrs.set(AttrKey::Stride, stride);
    attrs.set(AttrKey::Padding, padding);
    return dispatch(op_id, inputs, attrs)[0];
}

auto fused_conv2d_sigmoid(
    const Tensor& input,
    const Tensor& weight,
    const Tensor* bias,
    int64_t stride,
    int64_t padding
) -> Tensor {
    return fused_conv2d_dispatch("fused_conv2d_sigmoid", OpId::FusedConv2dSigmoid,
                                 input, weight, bias, stride, padding);
}

auto fused_conv2d_tanh(
    const Tensor& input,
    const Tensor& weight,
    const Tensor* bias,
    int64_t stride,
    int64_t padding
) -> Tensor {
    return fused_conv2d_dispatch("fused_conv2d_tanh", OpId::FusedConv2dTanh,
                                 input, weight, bias, stride, padding);
}

auto fused_conv2d_swish(
    const Tensor& input,
    const Tensor& weight,
    const Tensor* bias,
    int64_t stride,
    int64_t padding
) -> Tensor {
    return fused_conv2d_dispatch("fused_conv2d_swish", OpId::FusedConv2dSwish,
                                 input, weight, bias, stride, padding);
}

auto fused_batchnorm_relu(
    const Tensor& input,
    const Tensor& running_mean,
    const Tensor& running_var,
    const Tensor& weight,
    const Tensor& bias,
    float eps
) -> Tensor {
    // Validate inputs
    if (input.ndim() < 2) {
        throw std::runtime_error(
            "fused_batchnorm_relu: input must be at least 2D, got " +
            std::to_string(input.ndim()) + "D"
        );
    }

    int64_t num_features = input.shape()[1];

    if (running_mean.numel() != num_features) {
        throw std::runtime_error(
            "fused_batchnorm_relu: running_mean size mismatch"
        );
    }

    if (running_var.numel() != num_features) {
        throw std::runtime_error(
            "fused_batchnorm_relu: running_var size mismatch"
        );
    }

    if (weight.numel() != num_features) {
        throw std::runtime_error(
            "fused_batchnorm_relu: weight size mismatch"
        );
    }

    if (bias.numel() != num_features) {
        throw std::runtime_error(
            "fused_batchnorm_relu: bias size mismatch"
        );
    }

    // Prepare inputs for dispatcher
    std::vector<Tensor> inputs = {input, running_mean, running_var, weight, bias};

    NewOpAttributes attrs;
    attrs.set(AttrKey::Eps, static_cast<double>(eps));

    return dispatch(OpId::FusedBatchNormReLU, inputs, attrs)[0];
}

auto fused_softmax_cross_entropy(
    const Tensor& logits,
    const Tensor& targets,
    const std::string& reduction
) -> Tensor {
    // Validate inputs
    if (logits.ndim() != 2) {
        throw std::runtime_error(
            "fused_softmax_cross_entropy: logits must be 2D (N, num_classes), got " +
            std::to_string(logits.ndim()) + "D"
        );
    }

    if (targets.ndim() != 1) {
        throw std::runtime_error(
            "fused_softmax_cross_entropy: targets must be 1D (N,), got " +
            std::to_string(targets.ndim()) + "D"
        );
    }

    int64_t batch_size = logits.shape()[0];
    if (targets.shape()[0] != batch_size) {
        throw std::runtime_error(
            "fused_softmax_cross_entropy: batch size mismatch: logits has " +
            std::to_string(batch_size) + ", targets has " +
            std::to_string(targets.shape()[0])
        );
    }

    if (reduction != "mean" && reduction != "sum" && reduction != "none") {
        throw std::runtime_error(
            "fused_softmax_cross_entropy: reduction must be 'mean', 'sum', or 'none', got '" +
            reduction + "'"
        );
    }

    // Prepare inputs for dispatcher
    std::vector<Tensor> inputs = {logits, targets};

    OpAttributes attrs;
    attrs.set(AttrKey::Reduction, reduction);

    return dispatch(OpId::FusedSoftmaxCrossEntropy, inputs, attrs)[0];
}

auto fused_add_relu(const Tensor& a, const Tensor& b) -> Tensor {
    // Simple fusion: add + relu
    // In-place computation on GPU can save memory bandwidth
    std::vector<Tensor> inputs = {a, b};
    return dispatch(OpId::FusedAddReLU, inputs)[0];
}

auto fused_gelu(const Tensor& input) -> Tensor {
    // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    std::vector<Tensor> inputs = {input};
    return dispatch(OpId::FusedGelu, inputs)[0];
}

auto fused_layer_norm(
    const Tensor& input,
    const std::vector<int64_t>& normalized_shape,
    const Tensor& weight,
    const Tensor& bias,
    float eps
) -> Tensor {
    // Validate inputs
    if (input.ndim() < normalized_shape.size()) {
        throw std::runtime_error(
            "fused_layer_norm: input ndim must be >= normalized_shape size"
        );
    }

    // Verify normalized_shape matches input dimensions
    int64_t offset = input.ndim() - normalized_shape.size();
    for (size_t i = 0; i < normalized_shape.size(); ++i) {
        if (input.shape()[offset + i] != normalized_shape[i]) {
            throw std::runtime_error(
                "fused_layer_norm: normalized_shape mismatch with input shape"
            );
        }
    }

    int64_t norm_size = 1;
    for (auto dim : normalized_shape) {
        norm_size *= dim;
    }

    if (weight.numel() != norm_size) {
        throw std::runtime_error(
            "fused_layer_norm: weight size mismatch"
        );
    }

    if (bias.numel() != norm_size) {
        throw std::runtime_error(
            "fused_layer_norm: bias size mismatch"
        );
    }

    // Prepare inputs for dispatcher
    std::vector<Tensor> inputs = {input, weight, bias};

    NewOpAttributes attrs;
    attrs.set(AttrKey::Eps, static_cast<double>(eps));

    // Encode normalized_shape as comma-separated string
    std::string shape_str;
    for (size_t i = 0; i < normalized_shape.size(); ++i) {
        if (i > 0) shape_str += ",";
        shape_str += std::to_string(normalized_shape[i]);
    }
    attrs.set(AttrKey::NormalizedShape, shape_str);

    return dispatch(OpId::FusedLayerNorm, inputs, attrs)[0];
}

} // namespace ops
} // namespace tenzor
