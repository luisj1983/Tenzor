/**
 * @file vulkan_kernel_registry.cpp
 * @brief Vulkan kernel registration for O(1) dispatch
 *
 * Registers all Vulkan kernel implementations with the dispatch table.
 * Each kernel wrapper calls the corresponding VulkanBackend method.
 */

#include "tenzor/backend/dispatch_table.hpp"
#include "tenzor/backend/kernel_registry.hpp"
#include "tenzor/backend/fast_dispatch.hpp"
#include "tenzor/ops/op_id.hpp"
#include "vulkan_backend.hpp"
#include <cstdlib>
#include <charconv>
#include <limits>

namespace tenzor {

// Helper to get VulkanBackend from the dispatch table registry
inline VulkanBackend* get_vulkan_backend() {
    auto* backend = DispatchTableRegistry::get_backend(Device::Type::Vulkan);
    return static_cast<VulkanBackend*>(backend);
}

// Helper to parse int64_t from attributes
inline int64_t parse_int64(const OpAttributes& attrs, const std::string& key, int64_t default_val = 0) {
    if (attrs.contains(key)) {
        return std::stoll(attrs.at(key));
    }
    return default_val;
}

// Helper to parse float from attributes
inline float parse_float(const OpAttributes& attrs, const std::string& key, float default_val = 0.0f) {
    if (attrs.contains(key)) {
        return std::stof(attrs.at(key));
    }
    return default_val;
}

// Helper to parse bool from attributes
inline bool parse_bool(const OpAttributes& attrs, const std::string& key, bool default_val = false) {
    if (attrs.contains(key)) {
        return attrs.at(key) == "1" || attrs.at(key) == "true";
    }
    return default_val;
}

// Helper to parse vector of int64_t from comma-separated string
inline std::vector<int64_t> parse_shape(const OpAttributes& attrs, const std::string& key) {
    std::vector<int64_t> result;
    if (!attrs.contains(key)) return result;

    std::string str = attrs.at(key);
    size_t pos = 0;
    while (pos < str.size()) {
        size_t comma = str.find(',', pos);
        if (comma == std::string::npos) comma = str.size();
        result.push_back(std::stoll(str.substr(pos, comma - pos)));
        pos = comma + 1;
    }
    return result;
}

/**
 * @brief Register all Vulkan kernels with the dispatch table.
 *
 * Each registration wraps a VulkanBackend method call.
 * The backend is retrieved from the dispatch table registry.
 */
void register_vulkan_kernels(BackendDispatchTable& table) {
    // ========================================================================
    // Binary Operations
    // ========================================================================
    table.register_kernel(OpId::Add, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return get_vulkan_backend()->dispatch("add", inputs, attrs);
    });

    table.register_kernel(OpId::Sub, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return get_vulkan_backend()->dispatch("sub", inputs, attrs);
    });

    table.register_kernel(OpId::Mul, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return get_vulkan_backend()->dispatch("mul", inputs, attrs);
    });

    table.register_kernel(OpId::Div, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return get_vulkan_backend()->dispatch("div", inputs, attrs);
    });

    table.register_kernel(OpId::MatMul, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return get_vulkan_backend()->dispatch("matmul", inputs, attrs);
    });

    table.register_kernel(OpId::Bmm, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return get_vulkan_backend()->dispatch("bmm", inputs, attrs);
    });

    table.register_kernel(OpId::Dot, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return get_vulkan_backend()->dispatch("dot", inputs, attrs);
    });

    // ========================================================================
    // In-place Operations (using InplaceKernelFn - no tensor copy)
    // ========================================================================
    table.register_inplace_kernel(OpId::AddInplace, [](Tensor& target, std::span<const Tensor> others, const OpAttributes& attrs) -> Tensor& {
        std::vector<Tensor> inputs = {target};
        inputs.insert(inputs.end(), others.begin(), others.end());
        get_vulkan_backend()->dispatch("add_inplace", inputs, attrs);
        return target;
    });

    table.register_inplace_kernel(OpId::SubInplace, [](Tensor& target, std::span<const Tensor> others, const OpAttributes& attrs) -> Tensor& {
        std::vector<Tensor> inputs = {target};
        inputs.insert(inputs.end(), others.begin(), others.end());
        get_vulkan_backend()->dispatch("sub_inplace", inputs, attrs);
        return target;
    });

    table.register_inplace_kernel(OpId::MulInplace, [](Tensor& target, std::span<const Tensor> others, const OpAttributes& attrs) -> Tensor& {
        std::vector<Tensor> inputs = {target};
        inputs.insert(inputs.end(), others.begin(), others.end());
        get_vulkan_backend()->dispatch("mul_inplace", inputs, attrs);
        return target;
    });

    table.register_inplace_kernel(OpId::DivInplace, [](Tensor& target, std::span<const Tensor> others, const OpAttributes& attrs) -> Tensor& {
        std::vector<Tensor> inputs = {target};
        inputs.insert(inputs.end(), others.begin(), others.end());
        get_vulkan_backend()->dispatch("div_inplace", inputs, attrs);
        return target;
    });

    // ========================================================================
    // Comparison Operations
    // ========================================================================
    table.register_kernel(OpId::Eq, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return get_vulkan_backend()->dispatch("eq", inputs, attrs);
    });

    table.register_kernel(OpId::Ne, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return get_vulkan_backend()->dispatch("ne", inputs, attrs);
    });

    table.register_kernel(OpId::Lt, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return get_vulkan_backend()->dispatch("lt", inputs, attrs);
    });

    table.register_kernel(OpId::Le, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return get_vulkan_backend()->dispatch("le", inputs, attrs);
    });

    table.register_kernel(OpId::Gt, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return get_vulkan_backend()->dispatch("gt", inputs, attrs);
    });

    table.register_kernel(OpId::Ge, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return get_vulkan_backend()->dispatch("ge", inputs, attrs);
    });

    // ========================================================================
    // Unary Math Operations
    // ========================================================================
    table.register_kernel(OpId::Sqrt, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return get_vulkan_backend()->dispatch("sqrt", inputs, attrs);
    });

    table.register_kernel(OpId::Neg, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return get_vulkan_backend()->dispatch("neg", inputs, attrs);
    });

    table.register_kernel(OpId::Abs, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return get_vulkan_backend()->dispatch("abs", inputs, attrs);
    });

    table.register_kernel(OpId::Sign, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return get_vulkan_backend()->dispatch("sign", inputs, attrs);
    });

    table.register_kernel(OpId::Log, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return get_vulkan_backend()->dispatch("log", inputs, attrs);
    });

    table.register_kernel(OpId::Exp, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return get_vulkan_backend()->dispatch("exp", inputs, attrs);
    });

    table.register_kernel(OpId::Pow, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return get_vulkan_backend()->dispatch("pow", inputs, attrs);
    });

    table.register_kernel(OpId::Reciprocal, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return get_vulkan_backend()->dispatch("reciprocal", inputs, attrs);
    });

    table.register_kernel(OpId::Floor, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return get_vulkan_backend()->dispatch("floor", inputs, attrs);
    });

    table.register_kernel(OpId::Ceil, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return get_vulkan_backend()->dispatch("ceil", inputs, attrs);
    });

    table.register_kernel(OpId::Round, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return get_vulkan_backend()->dispatch("round", inputs, attrs);
    });

    table.register_kernel(OpId::Clamp, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return get_vulkan_backend()->dispatch("clamp", inputs, attrs);
    });

    table.register_kernel(OpId::ClampMin, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return get_vulkan_backend()->dispatch("clamp_min", inputs, attrs);
    });

    table.register_kernel(OpId::ClampMax, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return get_vulkan_backend()->dispatch("clamp_max", inputs, attrs);
    });

    // ========================================================================
    // Trigonometric Operations
    // ========================================================================
    table.register_kernel(OpId::Sin, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return get_vulkan_backend()->dispatch("sin", inputs, attrs);
    });

    table.register_kernel(OpId::Cos, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return get_vulkan_backend()->dispatch("cos", inputs, attrs);
    });

    table.register_kernel(OpId::Tan, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return get_vulkan_backend()->dispatch("tan", inputs, attrs);
    });

    table.register_kernel(OpId::Asin, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return get_vulkan_backend()->dispatch("asin", inputs, attrs);
    });

    table.register_kernel(OpId::Acos, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return get_vulkan_backend()->dispatch("acos", inputs, attrs);
    });

    table.register_kernel(OpId::Atan, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return get_vulkan_backend()->dispatch("atan", inputs, attrs);
    });

    table.register_kernel(OpId::Sinh, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return get_vulkan_backend()->dispatch("sinh", inputs, attrs);
    });

    table.register_kernel(OpId::Cosh, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return get_vulkan_backend()->dispatch("cosh", inputs, attrs);
    });

    table.register_kernel(OpId::Tanh, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return get_vulkan_backend()->dispatch("tanh", inputs, attrs);
    });

    // ========================================================================
    // Reduction Operations
    // ========================================================================
    table.register_kernel(OpId::Sum, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return get_vulkan_backend()->dispatch("sum", inputs, attrs);
    });

    table.register_kernel(OpId::Mean, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return get_vulkan_backend()->dispatch("mean", inputs, attrs);
    });

    table.register_kernel(OpId::Max, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return get_vulkan_backend()->dispatch("max", inputs, attrs);
    });

    table.register_kernel(OpId::Min, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return get_vulkan_backend()->dispatch("min", inputs, attrs);
    });

    table.register_kernel(OpId::ArgMax, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return get_vulkan_backend()->dispatch("argmax", inputs, attrs);
    });

    table.register_kernel(OpId::ArgMin, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return get_vulkan_backend()->dispatch("argmin", inputs, attrs);
    });

    table.register_kernel(OpId::Prod, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return get_vulkan_backend()->dispatch("prod", inputs, attrs);
    });

    table.register_kernel(OpId::Var, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return get_vulkan_backend()->dispatch("var", inputs, attrs);
    });

    table.register_kernel(OpId::Std, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return get_vulkan_backend()->dispatch("std", inputs, attrs);
    });

    table.register_kernel(OpId::Norm, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return get_vulkan_backend()->dispatch("norm", inputs, attrs);
    });

    // ========================================================================
    // Activation Functions
    // ========================================================================
    table.register_kernel(OpId::ReLU, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return get_vulkan_backend()->dispatch("relu", inputs, attrs);
    });

    table.register_kernel(OpId::ReLUBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return get_vulkan_backend()->dispatch("relu_backward", inputs, attrs);
    });

    table.register_kernel(OpId::Sigmoid, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return get_vulkan_backend()->dispatch("sigmoid", inputs, attrs);
    });

    table.register_kernel(OpId::SigmoidBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return get_vulkan_backend()->dispatch("sigmoid_backward", inputs, attrs);
    });

    table.register_kernel(OpId::TanhActivation, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return get_vulkan_backend()->dispatch("tanh", inputs, attrs);
    });

    table.register_kernel(OpId::TanhBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return get_vulkan_backend()->dispatch("tanh_backward", inputs, attrs);
    });

    table.register_kernel(OpId::Gelu, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return get_vulkan_backend()->dispatch("gelu", inputs, attrs);
    });

    table.register_kernel(OpId::GeluBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return get_vulkan_backend()->dispatch("gelu_backward", inputs, attrs);
    });

    table.register_kernel(OpId::Swish, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return get_vulkan_backend()->dispatch("swish", inputs, attrs);
    });

    table.register_kernel(OpId::SwishBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return get_vulkan_backend()->dispatch("swish_backward", inputs, attrs);
    });

    table.register_kernel(OpId::LeakyReLU, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return get_vulkan_backend()->dispatch("leaky_relu", inputs, attrs);
    });

    table.register_kernel(OpId::LeakyReLUBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return get_vulkan_backend()->dispatch("leaky_relu_backward", inputs, attrs);
    });

    table.register_kernel(OpId::Elu, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return get_vulkan_backend()->dispatch("elu", inputs, attrs);
    });

    table.register_kernel(OpId::EluBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return get_vulkan_backend()->dispatch("elu_backward", inputs, attrs);
    });

    table.register_kernel(OpId::Selu, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return get_vulkan_backend()->dispatch("selu", inputs, attrs);
    });

    table.register_kernel(OpId::SeluBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return get_vulkan_backend()->dispatch("selu_backward", inputs, attrs);
    });

    table.register_kernel(OpId::Mish, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return get_vulkan_backend()->dispatch("mish", inputs, attrs);
    });

    table.register_kernel(OpId::MishBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return get_vulkan_backend()->dispatch("mish_backward", inputs, attrs);
    });

    table.register_kernel(OpId::Softplus, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return get_vulkan_backend()->dispatch("softplus", inputs, attrs);
    });

    table.register_kernel(OpId::SoftplusBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return get_vulkan_backend()->dispatch("softplus_backward", inputs, attrs);
    });

    // ========================================================================
    // Softmax Operations
    // ========================================================================
    table.register_kernel(OpId::Softmax, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return get_vulkan_backend()->dispatch("softmax", inputs, attrs);
    });

    table.register_kernel(OpId::SoftmaxBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return get_vulkan_backend()->dispatch("softmax_backward", inputs, attrs);
    });

    table.register_kernel(OpId::LogSoftmax, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return get_vulkan_backend()->dispatch("log_softmax", inputs, attrs);
    });

    table.register_kernel(OpId::LogSoftmaxBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return get_vulkan_backend()->dispatch("log_softmax_backward", inputs, attrs);
    });

    // ========================================================================
    // Transform Operations
    // ========================================================================
    table.register_kernel(OpId::Reshape, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return get_vulkan_backend()->dispatch("reshape", inputs, attrs);
    });

    table.register_kernel(OpId::Transpose, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return get_vulkan_backend()->dispatch("transpose", inputs, attrs);
    });

    table.register_kernel(OpId::Permute, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return get_vulkan_backend()->dispatch("permute", inputs, attrs);
    });

    table.register_kernel(OpId::Squeeze, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return get_vulkan_backend()->dispatch("squeeze", inputs, attrs);
    });

    table.register_kernel(OpId::Unsqueeze, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return get_vulkan_backend()->dispatch("unsqueeze", inputs, attrs);
    });

    table.register_kernel(OpId::Contiguous, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return get_vulkan_backend()->dispatch("contiguous", inputs, attrs);
    });

    table.register_kernel(OpId::Expand, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return get_vulkan_backend()->dispatch("expand", inputs, attrs);
    });

    table.register_kernel(OpId::Cat, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return get_vulkan_backend()->dispatch("cat", inputs, attrs);
    });

    table.register_kernel(OpId::Repeat, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return get_vulkan_backend()->dispatch("repeat", inputs, attrs);
    });

    // ========================================================================
    // Indexing Operations
    // ========================================================================
    table.register_kernel(OpId::IndexSelect, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return get_vulkan_backend()->dispatch("index_select", inputs, attrs);
    });

    table.register_kernel(OpId::Gather, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return get_vulkan_backend()->dispatch("gather", inputs, attrs);
    });

    table.register_kernel(OpId::Scatter, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return get_vulkan_backend()->dispatch("scatter", inputs, attrs);
    });

    table.register_kernel(OpId::Embedding, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return get_vulkan_backend()->dispatch("embedding", inputs, attrs);
    });

    table.register_kernel(OpId::MaskedSelect, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return get_vulkan_backend()->dispatch("masked_select", inputs, attrs);
    });

    table.register_kernel(OpId::MaskedFill, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return get_vulkan_backend()->dispatch("masked_fill", inputs, attrs);
    });

    table.register_kernel(OpId::Where, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return get_vulkan_backend()->dispatch("where", inputs, attrs);
    });

    // ========================================================================
    // Tensor Creation Operations
    // ========================================================================
    table.register_kernel(OpId::Zeros, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return get_vulkan_backend()->dispatch("zeros", inputs, attrs);
    });

    table.register_kernel(OpId::Ones, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return get_vulkan_backend()->dispatch("ones", inputs, attrs);
    });

    table.register_kernel(OpId::Full, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return get_vulkan_backend()->dispatch("full", inputs, attrs);
    });

    table.register_kernel(OpId::Fill, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return get_vulkan_backend()->dispatch("fill", inputs, attrs);
    });

    table.register_kernel(OpId::Clone, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return get_vulkan_backend()->dispatch("clone", inputs, attrs);
    });

    table.register_kernel(OpId::Rand, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return get_vulkan_backend()->dispatch("rand", inputs, attrs);
    });

    table.register_kernel(OpId::Randn, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return get_vulkan_backend()->dispatch("randn", inputs, attrs);
    });

    table.register_kernel(OpId::Arange, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return get_vulkan_backend()->dispatch("arange", inputs, attrs);
    });

    table.register_kernel(OpId::Linspace, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return get_vulkan_backend()->dispatch("linspace", inputs, attrs);
    });

    table.register_kernel(OpId::Eye, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return get_vulkan_backend()->dispatch("eye", inputs, attrs);
    });

    // ========================================================================
    // Pooling Operations
    // ========================================================================
    // Note: MaxPool2dForward is the canonical OpId for max pooling
    table.register_kernel(OpId::MaxPool2dForward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return get_vulkan_backend()->dispatch("max_pool2d_forward", inputs, attrs);
    });

    table.register_kernel(OpId::MaxPool2dBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return get_vulkan_backend()->dispatch("max_pool2d_backward", inputs, attrs);
    });

    // Note: AvgPool2dForward is the canonical OpId for avg pooling
    table.register_kernel(OpId::AvgPool2dForward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return get_vulkan_backend()->dispatch("avg_pool2d_forward", inputs, attrs);
    });

    table.register_kernel(OpId::AvgPool2dBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return get_vulkan_backend()->dispatch("avg_pool2d_backward", inputs, attrs);
    });

    table.register_kernel(OpId::AdaptiveMaxPool2d, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return get_vulkan_backend()->dispatch("adaptive_max_pool2d", inputs, attrs);
    });

    table.register_kernel(OpId::AdaptiveAvgPool2d, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return get_vulkan_backend()->dispatch("adaptive_avg_pool2d", inputs, attrs);
    });

    table.register_kernel(OpId::AdaptiveAvgPool2dBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return get_vulkan_backend()->dispatch("adaptive_avg_pool2d_backward", inputs, attrs);
    });

    // ========================================================================
    // Convolution Operations
    // ========================================================================
    table.register_kernel(OpId::Conv2dForward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return get_vulkan_backend()->dispatch("conv2d_forward", inputs, attrs);
    });

    table.register_kernel(OpId::Conv2dBackwardInput, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return get_vulkan_backend()->dispatch("conv2d_backward_input", inputs, attrs);
    });

    table.register_kernel(OpId::Conv2dBackwardWeight, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return get_vulkan_backend()->dispatch("conv2d_backward_weight", inputs, attrs);
    });

    table.register_kernel(OpId::Conv2dBackwardBias, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return get_vulkan_backend()->dispatch("conv2d_backward_bias", inputs, attrs);
    });

    table.register_kernel(OpId::ConvTranspose2dForward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return get_vulkan_backend()->dispatch("conv_transpose2d_forward", inputs, attrs);
    });

    // ========================================================================
    // BatchNorm Operations
    // ========================================================================
    table.register_kernel(OpId::BatchNorm2dForward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return get_vulkan_backend()->dispatch("batchnorm2d_forward", inputs, attrs);
    });

    table.register_kernel(OpId::BatchNorm2dForwardAffine, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return get_vulkan_backend()->dispatch("batchnorm2d_forward_affine", inputs, attrs);
    });

    table.register_kernel(OpId::BatchNorm2dBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return get_vulkan_backend()->dispatch("batchnorm2d_backward", inputs, attrs);
    });

    table.register_kernel(OpId::BatchNorm2dMeanVar, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return get_vulkan_backend()->dispatch("batchnorm2d_mean_var", inputs, attrs);
    });

    table.register_kernel(OpId::BatchNorm2dUpdateRunningStats, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return get_vulkan_backend()->dispatch("batchnorm2d_update_running_stats", inputs, attrs);
    });

    // ========================================================================
    // Vision Operations
    // ========================================================================
    table.register_kernel(OpId::Unfold, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return get_vulkan_backend()->dispatch("unfold", inputs, attrs);
    });

    table.register_kernel(OpId::Fold, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return get_vulkan_backend()->dispatch("fold", inputs, attrs);
    });

    table.register_kernel(OpId::ROIAlignForward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return get_vulkan_backend()->dispatch("roi_align_forward", inputs, attrs);
    });

    table.register_kernel(OpId::ROIAlignBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return get_vulkan_backend()->dispatch("roi_align_backward", inputs, attrs);
    });

    // Note: GatherRelativePositionBias not in OpId enum - registered via string dispatch only

    // ========================================================================
    // Normalization Backward Operations
    // ========================================================================
    table.register_kernel(OpId::LayerNormBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return get_vulkan_backend()->dispatch("layer_norm_backward", inputs, attrs);
    });

    table.register_kernel(OpId::GroupNormBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return get_vulkan_backend()->dispatch("group_norm_backward", inputs, attrs);
    });

    table.register_kernel(OpId::LayerNorm, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return get_vulkan_backend()->dispatch("layer_norm", inputs, attrs);
    });

    table.register_kernel(OpId::GroupNorm, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return get_vulkan_backend()->dispatch("group_norm", inputs, attrs);
    });

    table.register_kernel(OpId::FusedRMSNorm, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return get_vulkan_backend()->dispatch("fused_rms_norm", inputs, attrs);
    });

    table.register_kernel(OpId::RMSNormBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return get_vulkan_backend()->dispatch("rms_norm_backward", inputs, attrs);
    });

    // ========================================================================
    // Embedding Backward
    // ========================================================================
    table.register_kernel(OpId::EmbeddingBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return get_vulkan_backend()->dispatch("embedding_backward", inputs, attrs);
    });

    // ========================================================================
    // Phase 4: Fused Optimizer Steps
    // ========================================================================
    table.register_kernel(OpId::FusedRMSPropStep, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return get_vulkan_backend()->dispatch("fused_rmsprop_step", inputs, attrs);
    });

    table.register_kernel(OpId::FusedAdadeltaStep, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return get_vulkan_backend()->dispatch("fused_adadelta_step", inputs, attrs);
    });

    table.register_kernel(OpId::FusedAdagradStep, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return get_vulkan_backend()->dispatch("fused_adagrad_step", inputs, attrs);
    });

    // ========================================================================
    // Phase 3: Nonzero, OneHot, BoxIoU
    // ========================================================================
    table.register_kernel(OpId::Nonzero, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return get_vulkan_backend()->dispatch("nonzero", inputs, attrs);
    });

    table.register_kernel(OpId::OneHot, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return get_vulkan_backend()->dispatch("one_hot", inputs, attrs);
    });

    table.register_kernel(OpId::BoxIoU, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return get_vulkan_backend()->dispatch("box_iou", inputs, attrs);
    });

    // ========================================================================
    // In-place Activation Operations (using InplaceKernelFn - no tensor copy)
    // ========================================================================
    table.register_inplace_kernel(OpId::ReLUInplace, [](Tensor& target, std::span<const Tensor> others, const OpAttributes& attrs) -> Tensor& {
        std::vector<Tensor> inputs = {target};
        get_vulkan_backend()->dispatch("relu_inplace", inputs, attrs);
        return target;
    });

    table.register_inplace_kernel(OpId::SigmoidInplace, [](Tensor& target, std::span<const Tensor> others, const OpAttributes& attrs) -> Tensor& {
        std::vector<Tensor> inputs = {target};
        get_vulkan_backend()->dispatch("sigmoid_inplace", inputs, attrs);
        return target;
    });

    table.register_inplace_kernel(OpId::TanhInplace, [](Tensor& target, std::span<const Tensor> others, const OpAttributes& attrs) -> Tensor& {
        std::vector<Tensor> inputs = {target};
        get_vulkan_backend()->dispatch("tanh_inplace", inputs, attrs);
        return target;
    });

    table.register_inplace_kernel(OpId::LeakyReLUInplace, [](Tensor& target, std::span<const Tensor> others, const OpAttributes& attrs) -> Tensor& {
        std::vector<Tensor> inputs = {target};
        get_vulkan_backend()->dispatch("leaky_relu_inplace", inputs, attrs);
        return target;
    });

    table.register_inplace_kernel(OpId::GeluInplace, [](Tensor& target, std::span<const Tensor> others, const OpAttributes& attrs) -> Tensor& {
        std::vector<Tensor> inputs = {target};
        get_vulkan_backend()->dispatch("gelu_inplace", inputs, attrs);
        return target;
    });

    // ========================================================================
    // Fused Operations
    // ========================================================================
    table.register_kernel(OpId::FusedLinearReLU, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return get_vulkan_backend()->dispatch("fused_linear_relu", inputs, attrs);
    });

    table.register_kernel(OpId::FusedBatchNormReLU, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return get_vulkan_backend()->dispatch("fused_batchnorm_relu", inputs, attrs);
    });

    table.register_kernel(OpId::FusedAddReLU, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return get_vulkan_backend()->dispatch("fused_add_relu", inputs, attrs);
    });

    table.register_kernel(OpId::FusedGelu, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return get_vulkan_backend()->dispatch("fused_gelu", inputs, attrs);
    });

    table.register_kernel(OpId::FusedLayerNorm, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return get_vulkan_backend()->dispatch("fused_layer_norm", inputs, attrs);
    });

    // ========================================================================
    // Interpolation Operations
    // ========================================================================
    table.register_kernel(OpId::Interpolate, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return get_vulkan_backend()->dispatch("interpolate", inputs, attrs);
    });

    // ========================================================================
    // ArgSort (GPU bitonic sort for Float32 last-dim, CPU fallback otherwise)
    // ========================================================================
    table.register_kernel(OpId::ArgSort, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return get_vulkan_backend()->dispatch("argsort", inputs, attrs);
    });

    std::cout << "Vulkan dispatch table initialized with O(1) lookup" << std::endl;
}

} // namespace tenzor

// Export for dynamic loading via dlsym
extern "C" {
    void register_kernels(tenzor::BackendDispatchTable* table) {
        if (table) {
            tenzor::register_vulkan_kernels(*table);
        }
    }
}
