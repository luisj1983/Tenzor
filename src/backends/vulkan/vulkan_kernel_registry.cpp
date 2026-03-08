/**
 * @file vulkan_kernel_registry.cpp
 * @brief Vulkan kernel registration for O(1) dispatch
 *
 * Registers all Vulkan kernel implementations with the dispatch table.
 * Each kernel wrapper calls the corresponding VulkanBackend typed dispatch
 * method directly, bypassing string-based dispatch for O(1) lookup.
 */

#include "tenzor/backend/dispatch_table.hpp"
#include "tenzor/backend/kernel_registry.hpp"
#include "tenzor/backend/fast_dispatch.hpp"
#include "tenzor/backend/op_attributes.hpp"
#include "tenzor/ops/op_id.hpp"
#include "vulkan_backend.hpp"
#include <cstdlib>
#include <limits>

// Undefine Xlib Bool macro that conflicts with DType::Bool
#ifdef Bool
#undef Bool
#endif

namespace tenzor {

// Helper to get VulkanBackend from the dispatch table registry
inline VulkanBackend* get_vulkan_backend() {
    auto* backend = DispatchTableRegistry::get_backend(Device::Type::Vulkan);
    return static_cast<VulkanBackend*>(backend);
}

// Helper to convert dtype string to DType enum
inline DType dtype_from_string(std::string_view s, DType default_val = DType::Float32) {
    if (s == "float32") return DType::Float32;
    if (s == "float64") return DType::Float64;
    if (s == "float16") return DType::Float16;
    if (s == "bfloat16") return DType::BFloat16;
    if (s == "int32") return DType::Int32;
    if (s == "int64") return DType::Int64;
    if (s == "int8") return DType::Int8;
    if (s == "uint8") return DType::UInt8;
    if (s == "bool") return DType::Bool;
    if (s.empty()) return default_val;
    return default_val;
}

/**
 * @brief Register all Vulkan kernels with the dispatch table.
 *
 * Each registration wraps a VulkanBackend typed dispatch method call,
 * bypassing the string-based dispatch chain for O(1) lookup performance.
 */
void register_vulkan_kernels(BackendDispatchTable& table) {
    // ========================================================================
    // Binary Operations
    // ========================================================================
    table.register_kernel(OpId::Add, [](std::span<const Tensor> inputs, const OpAttributes&) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchBinaryOp("add", inputs[0], inputs[1])};
    });

    table.register_kernel(OpId::Sub, [](std::span<const Tensor> inputs, const OpAttributes&) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchBinaryOp("sub", inputs[0], inputs[1])};
    });

    table.register_kernel(OpId::Mul, [](std::span<const Tensor> inputs, const OpAttributes&) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchBinaryOp("mul", inputs[0], inputs[1])};
    });

    table.register_kernel(OpId::Div, [](std::span<const Tensor> inputs, const OpAttributes&) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchBinaryOp("div", inputs[0], inputs[1])};
    });

    table.register_kernel(OpId::MatMul, [](std::span<const Tensor> inputs, const OpAttributes&) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchMatmul(inputs[0], inputs[1])};
    });

    table.register_kernel(OpId::Bmm, [](std::span<const Tensor> inputs, const OpAttributes&) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchBmm(inputs[0], inputs[1])};
    });

    table.register_kernel(OpId::Dot, [](std::span<const Tensor> inputs, const OpAttributes&) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchDot(inputs[0], inputs[1])};
    });

    // ========================================================================
    // In-place Binary Operations
    // ========================================================================
    table.register_inplace_kernel(OpId::AddInplace, [](Tensor& target, std::span<const Tensor> others, const OpAttributes&) -> Tensor& {
        get_vulkan_backend()->dispatchBinaryOp("add_inplace", target, others[0]);
        return target;
    });

    table.register_inplace_kernel(OpId::SubInplace, [](Tensor& target, std::span<const Tensor> others, const OpAttributes&) -> Tensor& {
        get_vulkan_backend()->dispatchBinaryOp("sub_inplace", target, others[0]);
        return target;
    });

    table.register_inplace_kernel(OpId::MulInplace, [](Tensor& target, std::span<const Tensor> others, const OpAttributes&) -> Tensor& {
        get_vulkan_backend()->dispatchBinaryOp("mul_inplace", target, others[0]);
        return target;
    });

    table.register_inplace_kernel(OpId::DivInplace, [](Tensor& target, std::span<const Tensor> others, const OpAttributes&) -> Tensor& {
        get_vulkan_backend()->dispatchBinaryOp("div_inplace", target, others[0]);
        return target;
    });

    // ========================================================================
    // Comparison Operations
    // ========================================================================
    table.register_kernel(OpId::Eq, [](std::span<const Tensor> inputs, const OpAttributes&) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchComparisonOp("eq", inputs[0], inputs[1])};
    });

    table.register_kernel(OpId::Ne, [](std::span<const Tensor> inputs, const OpAttributes&) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchComparisonOp("ne", inputs[0], inputs[1])};
    });

    table.register_kernel(OpId::Lt, [](std::span<const Tensor> inputs, const OpAttributes&) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchComparisonOp("lt", inputs[0], inputs[1])};
    });

    table.register_kernel(OpId::Le, [](std::span<const Tensor> inputs, const OpAttributes&) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchComparisonOp("le", inputs[0], inputs[1])};
    });

    table.register_kernel(OpId::Gt, [](std::span<const Tensor> inputs, const OpAttributes&) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchComparisonOp("gt", inputs[0], inputs[1])};
    });

    table.register_kernel(OpId::Ge, [](std::span<const Tensor> inputs, const OpAttributes&) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchComparisonOp("ge", inputs[0], inputs[1])};
    });

    // ========================================================================
    // Unary Math Operations
    // ========================================================================
    table.register_kernel(OpId::Sqrt, [](std::span<const Tensor> inputs, const OpAttributes&) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchUnaryOp("sqrt", inputs[0])};
    });

    table.register_kernel(OpId::Neg, [](std::span<const Tensor> inputs, const OpAttributes&) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchUnaryOp("neg", inputs[0])};
    });

    table.register_kernel(OpId::Abs, [](std::span<const Tensor> inputs, const OpAttributes&) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchUnaryOp("abs", inputs[0])};
    });

    table.register_kernel(OpId::Sign, [](std::span<const Tensor> inputs, const OpAttributes&) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchUnaryOp("sign", inputs[0])};
    });

    table.register_kernel(OpId::Log, [](std::span<const Tensor> inputs, const OpAttributes&) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchUnaryOp("log", inputs[0])};
    });

    table.register_kernel(OpId::Exp, [](std::span<const Tensor> inputs, const OpAttributes&) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchUnaryOp("exp", inputs[0])};
    });

    table.register_kernel(OpId::Pow, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchUnaryOpWithParam("pow", inputs[0], static_cast<float>(attrs.get_float(AttrKey::Exponent, 2.0)))};
    });

    table.register_kernel(OpId::Reciprocal, [](std::span<const Tensor> inputs, const OpAttributes&) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchUnaryOp("reciprocal", inputs[0])};
    });

    table.register_kernel(OpId::Floor, [](std::span<const Tensor> inputs, const OpAttributes&) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchUnaryOp("floor", inputs[0])};
    });

    table.register_kernel(OpId::Ceil, [](std::span<const Tensor> inputs, const OpAttributes&) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchUnaryOp("ceil", inputs[0])};
    });

    table.register_kernel(OpId::Round, [](std::span<const Tensor> inputs, const OpAttributes&) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchUnaryOp("round", inputs[0])};
    });

    table.register_kernel(OpId::Trunc, [](std::span<const Tensor> inputs, const OpAttributes&) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchUnaryOp("trunc", inputs[0])};
    });

    table.register_kernel(OpId::Clamp, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchClamp(inputs[0],
            static_cast<float>(attrs.get_float(AttrKey::Min, -std::numeric_limits<float>::infinity())),
            static_cast<float>(attrs.get_float(AttrKey::Max, std::numeric_limits<float>::infinity())))};
    });

    table.register_kernel(OpId::ClampMin, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchClamp(inputs[0],
            static_cast<float>(attrs.get_float(AttrKey::Min, 0.0)), std::numeric_limits<float>::infinity())};
    });

    table.register_kernel(OpId::ClampMax, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchClamp(inputs[0],
            -std::numeric_limits<float>::infinity(), static_cast<float>(attrs.get_float(AttrKey::Max, 0.0)))};
    });

    // ========================================================================
    // Trigonometric Operations
    // ========================================================================
    table.register_kernel(OpId::Sin, [](std::span<const Tensor> inputs, const OpAttributes&) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchTrigonometricOp("sin", inputs[0])};
    });

    table.register_kernel(OpId::Cos, [](std::span<const Tensor> inputs, const OpAttributes&) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchTrigonometricOp("cos", inputs[0])};
    });

    table.register_kernel(OpId::Tan, [](std::span<const Tensor> inputs, const OpAttributes&) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchTrigonometricOp("tan", inputs[0])};
    });

    table.register_kernel(OpId::Asin, [](std::span<const Tensor> inputs, const OpAttributes&) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchTrigonometricOp("asin", inputs[0])};
    });

    table.register_kernel(OpId::Acos, [](std::span<const Tensor> inputs, const OpAttributes&) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchTrigonometricOp("acos", inputs[0])};
    });

    table.register_kernel(OpId::Atan, [](std::span<const Tensor> inputs, const OpAttributes&) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchTrigonometricOp("atan", inputs[0])};
    });

    table.register_kernel(OpId::Sinh, [](std::span<const Tensor> inputs, const OpAttributes&) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchHyperbolicOp("sinh", inputs[0])};
    });

    table.register_kernel(OpId::Cosh, [](std::span<const Tensor> inputs, const OpAttributes&) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchHyperbolicOp("cosh", inputs[0])};
    });

    table.register_kernel(OpId::Tanh, [](std::span<const Tensor> inputs, const OpAttributes&) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchHyperbolicOp("tanh", inputs[0])};
    });

    // ========================================================================
    // Reduction Operations
    // ========================================================================
    table.register_kernel(OpId::Sum, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchReduction("sum", inputs[0],
            attrs.get_int(AttrKey::Dim, -1), attrs.get_bool(AttrKey::Keepdim, false))};
    });

    table.register_kernel(OpId::Mean, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchReduction("mean", inputs[0],
            attrs.get_int(AttrKey::Dim, -1), attrs.get_bool(AttrKey::Keepdim, false))};
    });

    table.register_kernel(OpId::Max, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchReduction("max", inputs[0],
            attrs.get_int(AttrKey::Dim, -1), attrs.get_bool(AttrKey::Keepdim, false))};
    });

    table.register_kernel(OpId::Min, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchReduction("min", inputs[0],
            attrs.get_int(AttrKey::Dim, -1), attrs.get_bool(AttrKey::Keepdim, false))};
    });

    table.register_kernel(OpId::ArgMax, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchArgmax(inputs[0],
            attrs.get_int(AttrKey::Dim, -1), attrs.get_bool(AttrKey::Keepdim, false))};
    });

    table.register_kernel(OpId::ArgMin, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchArgmin(inputs[0],
            attrs.get_int(AttrKey::Dim, -1), attrs.get_bool(AttrKey::Keepdim, false))};
    });

    table.register_kernel(OpId::Prod, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchProd(inputs[0],
            attrs.get_int(AttrKey::Dim, -1), attrs.get_bool(AttrKey::Keepdim, false))};
    });

    table.register_kernel(OpId::Var, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchVariance(inputs[0],
            attrs.get_int(AttrKey::Dim, -1), attrs.get_bool(AttrKey::Unbiased, true),
            attrs.get_bool(AttrKey::Keepdim, false))};
    });

    table.register_kernel(OpId::Std, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchStd(inputs[0],
            attrs.get_int(AttrKey::Dim, -1), attrs.get_bool(AttrKey::Unbiased, true),
            attrs.get_bool(AttrKey::Keepdim, false))};
    });

    table.register_kernel(OpId::Norm, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchNorm(inputs[0],
            static_cast<float>(attrs.get_float(AttrKey::P, 2.0)), attrs.get_int(AttrKey::Dim, -1),
            attrs.get_bool(AttrKey::Keepdim, false))};
    });

    // ========================================================================
    // Activation Functions
    // ========================================================================
    table.register_kernel(OpId::ReLU, [](std::span<const Tensor> inputs, const OpAttributes&) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchActivation("relu", inputs[0], 0, 0.0f)};
    });

    table.register_kernel(OpId::ReLUBackward, [](std::span<const Tensor> inputs, const OpAttributes&) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchActivationBackward("relu_backward", inputs[0], inputs[1], 0, 0.0f)};
    });

    table.register_kernel(OpId::Sigmoid, [](std::span<const Tensor> inputs, const OpAttributes&) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchActivation("sigmoid", inputs[0], 1, 0.0f)};
    });

    table.register_kernel(OpId::SigmoidBackward, [](std::span<const Tensor> inputs, const OpAttributes&) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchActivationBackward("sigmoid_backward", inputs[0], inputs[1], 1, 0.0f)};
    });

    table.register_kernel(OpId::TanhActivation, [](std::span<const Tensor> inputs, const OpAttributes&) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchActivation("tanh", inputs[0], 2, 0.0f)};
    });

    table.register_kernel(OpId::TanhBackward, [](std::span<const Tensor> inputs, const OpAttributes&) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchActivationBackward("tanh_backward", inputs[0], inputs[1], 2, 0.0f)};
    });

    table.register_kernel(OpId::Gelu, [](std::span<const Tensor> inputs, const OpAttributes&) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchActivation("gelu", inputs[0], 3, 0.0f)};
    });

    table.register_kernel(OpId::GeluBackward, [](std::span<const Tensor> inputs, const OpAttributes&) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchActivationBackward("gelu_backward", inputs[0], inputs[1], 4, 0.0f)};
    });

    table.register_kernel(OpId::LeakyReLU, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchActivation("leaky_relu", inputs[0], 4, static_cast<float>(attrs.get_float(AttrKey::Alpha, 0.01)))};
    });

    table.register_kernel(OpId::LeakyReLUBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchActivationBackward("leaky_relu_backward", inputs[0], inputs[1], 3, static_cast<float>(attrs.get_float(AttrKey::Alpha, 0.01)))};
    });

    table.register_kernel(OpId::Swish, [](std::span<const Tensor> inputs, const OpAttributes&) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchActivation("swish", inputs[0], 5, 0.0f)};
    });

    table.register_kernel(OpId::SwishBackward, [](std::span<const Tensor> inputs, const OpAttributes&) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchSwishBackward(inputs[0], inputs[1])};
    });

    table.register_kernel(OpId::Elu, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchActivation("elu", inputs[0], 6, static_cast<float>(attrs.get_float(AttrKey::Alpha, 1.0)))};
    });

    table.register_kernel(OpId::EluBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchActivationBackward("elu_backward", inputs[0], inputs[1], 5, static_cast<float>(attrs.get_float(AttrKey::Alpha, 1.0)))};
    });

    table.register_kernel(OpId::Selu, [](std::span<const Tensor> inputs, const OpAttributes&) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchActivation("selu", inputs[0], 7, 0.0f)};
    });

    table.register_kernel(OpId::SeluBackward, [](std::span<const Tensor> inputs, const OpAttributes&) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchActivationBackward("selu_backward", inputs[0], inputs[1], 6, 0.0f)};
    });

    table.register_kernel(OpId::Mish, [](std::span<const Tensor> inputs, const OpAttributes&) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchActivation("mish", inputs[0], 8, 0.0f)};
    });

    table.register_kernel(OpId::MishBackward, [](std::span<const Tensor> inputs, const OpAttributes&) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchActivationBackward("mish_backward", inputs[0], inputs[1], 7, 0.0f)};
    });

    table.register_kernel(OpId::Softplus, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchActivation("softplus", inputs[0], 9, static_cast<float>(attrs.get_float(AttrKey::Beta, 1.0)))};
    });

    table.register_kernel(OpId::SoftplusBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchActivationBackward("softplus_backward", inputs[0], inputs[1], 8, static_cast<float>(attrs.get_float(AttrKey::Beta, 1.0)))};
    });

    // ========================================================================
    // Softmax Operations
    // ========================================================================
    table.register_kernel(OpId::Softmax, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchSoftmax(inputs[0], attrs.get_int(AttrKey::Dim, -1))};
    });

    table.register_kernel(OpId::SoftmaxBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchSoftmaxBackward(inputs[0], inputs[1], attrs.get_int(AttrKey::Dim, -1))};
    });

    table.register_kernel(OpId::LogSoftmax, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchLogSoftmax(inputs[0], attrs.get_int(AttrKey::Dim, -1))};
    });

    table.register_kernel(OpId::LogSoftmaxBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchLogSoftmaxBackward(inputs[0], inputs[1], attrs.get_int(AttrKey::Dim, -1))};
    });

    // ========================================================================
    // Transform Operations
    // ========================================================================
    table.register_kernel(OpId::Reshape, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchReshape(inputs[0], attrs.get_int_list(AttrKey::Shape))};
    });

    table.register_kernel(OpId::Transpose, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchTranspose(inputs[0],
            attrs.get_int(AttrKey::Dim0, 0), attrs.get_int(AttrKey::Dim1, 1))};
    });

    table.register_kernel(OpId::Permute, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        auto dims = attrs.get_int_list(AttrKey::Dims);
        return std::vector<Tensor>{get_vulkan_backend()->dispatchPermute(inputs[0], dims)};
    });

    table.register_kernel(OpId::Squeeze, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchSqueeze(inputs[0],
            attrs.get_int(AttrKey::Dim, -1))};
    });

    table.register_kernel(OpId::Unsqueeze, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchUnsqueeze(inputs[0],
            attrs.get_int(AttrKey::Dim, 0))};
    });

    table.register_kernel(OpId::Contiguous, [](std::span<const Tensor> inputs, const OpAttributes&) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchContiguous(inputs[0])};
    });

    table.register_kernel(OpId::Expand, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchExpand(inputs[0], attrs.get_int_list(AttrKey::Shape))};
    });

    table.register_kernel(OpId::Cat, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        std::vector<Tensor> tensor_list(inputs.begin(), inputs.end());
        return std::vector<Tensor>{get_vulkan_backend()->dispatchCat(tensor_list, attrs.get_int(AttrKey::Dim, 0))};
    });

    table.register_kernel(OpId::Repeat, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchRepeat(inputs[0], attrs.get_int_list(AttrKey::Repeats))};
    });

    // ========================================================================
    // Indexing Operations
    // ========================================================================
    table.register_kernel(OpId::IndexSelect, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchIndexSelect(inputs[0],
            attrs.get_int(AttrKey::Dim, 0), inputs[1])};
    });

    table.register_kernel(OpId::Gather, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchGather(inputs[0],
            attrs.get_int(AttrKey::Dim, 0), inputs[1])};
    });

    table.register_kernel(OpId::Scatter, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchScatter(inputs[0],
            attrs.get_int(AttrKey::Dim, 0), inputs[1], inputs[2],
            attrs.get_int(AttrKey::Reduction, 0))};
    });

    table.register_kernel(OpId::Embedding, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchEmbedding(inputs[0], inputs[1],
            attrs.get_int(AttrKey::PaddingIdx, -1))};
    });

    table.register_kernel(OpId::MaskedSelect, [](std::span<const Tensor> inputs, const OpAttributes&) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchMaskedSelect(inputs[0], inputs[1])};
    });

    table.register_kernel(OpId::MaskedFill, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchMaskedFill(inputs[0], inputs[1],
            static_cast<float>(attrs.get_float(AttrKey::Value, 0.0)))};
    });

    table.register_kernel(OpId::Where, [](std::span<const Tensor> inputs, const OpAttributes&) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchWhere(inputs[0], inputs[1], inputs[2])};
    });

    // ========================================================================
    // Tensor Creation Operations
    // ========================================================================
    table.register_kernel(OpId::Zeros, [](std::span<const Tensor>, const OpAttributes& attrs) {
        auto shape = attrs.get_int_list(AttrKey::Shape);
        auto dtype = dtype_from_string(attrs.get_string(AttrKey::Dtype));
        auto device_id = static_cast<int32_t>(attrs.get_int(AttrKey::Device, 0));
        return std::vector<Tensor>{get_vulkan_backend()->dispatchZeros(shape, dtype, Device(Device::Type::Vulkan, device_id))};
    });

    table.register_kernel(OpId::Ones, [](std::span<const Tensor>, const OpAttributes& attrs) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchOnes(attrs.get_int_list(AttrKey::Shape), dtype_from_string(attrs.get_string(AttrKey::Dtype)))};
    });

    table.register_kernel(OpId::Full, [](std::span<const Tensor>, const OpAttributes& attrs) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchFull(attrs.get_int_list(AttrKey::Shape),
            static_cast<float>(attrs.get_float(AttrKey::Value, 0.0)), dtype_from_string(attrs.get_string(AttrKey::Dtype)))};
    });

    table.register_kernel(OpId::Fill, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchFill(inputs[0], static_cast<float>(attrs.get_float(AttrKey::Value, 0.0)))};
    });

    table.register_kernel(OpId::Clone, [](std::span<const Tensor> inputs, const OpAttributes&) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchClone(inputs[0])};
    });

    table.register_kernel(OpId::Rand, [](std::span<const Tensor>, const OpAttributes& attrs) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchRand(attrs.get_int_list(AttrKey::Shape), dtype_from_string(attrs.get_string(AttrKey::Dtype)))};
    });

    table.register_kernel(OpId::Randn, [](std::span<const Tensor>, const OpAttributes& attrs) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchRandn(attrs.get_int_list(AttrKey::Shape), dtype_from_string(attrs.get_string(AttrKey::Dtype)))};
    });

    table.register_kernel(OpId::Arange, [](std::span<const Tensor>, const OpAttributes& attrs) {
        auto device_id = static_cast<int32_t>(attrs.get_int(AttrKey::Device, 0));
        return std::vector<Tensor>{get_vulkan_backend()->dispatchArange(
            static_cast<float>(attrs.get_float(AttrKey::Start, 0.0)), static_cast<float>(attrs.get_float(AttrKey::End, 0.0)),
            static_cast<float>(attrs.get_float(AttrKey::Step, 1.0)), dtype_from_string(attrs.get_string(AttrKey::Dtype)),
            Device(Device::Type::Vulkan, device_id))};
    });

    table.register_kernel(OpId::Linspace, [](std::span<const Tensor>, const OpAttributes& attrs) {
        auto device_id = static_cast<int32_t>(attrs.get_int(AttrKey::Device, 0));
        return std::vector<Tensor>{get_vulkan_backend()->dispatchLinspace(
            static_cast<float>(attrs.get_float(AttrKey::Start, 0.0)), static_cast<float>(attrs.get_float(AttrKey::End, 1.0)),
            attrs.get_int(AttrKey::Steps, 100), dtype_from_string(attrs.get_string(AttrKey::Dtype)),
            Device(Device::Type::Vulkan, device_id))};
    });

    table.register_kernel(OpId::Eye, [](std::span<const Tensor>, const OpAttributes& attrs) {
        auto device_id = static_cast<int32_t>(attrs.get_int(AttrKey::Device, 0));
        return std::vector<Tensor>{get_vulkan_backend()->dispatchEye(
            attrs.get_int(AttrKey::N, 0), attrs.get_int(AttrKey::M, 0),
            dtype_from_string(attrs.get_string(AttrKey::Dtype)), Device(Device::Type::Vulkan, device_id))};
    });

    // ========================================================================
    // Pooling Operations
    // ========================================================================
    table.register_kernel(OpId::MaxPool2dForward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchMaxPool2dForward(inputs[0], attrs)};
    });

    table.register_kernel(OpId::MaxPool2dBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchMaxPool2dBackwardWithIndices(
            inputs[0], inputs[1],
            attrs.get_int(AttrKey::InputH, 0), attrs.get_int(AttrKey::InputW, 0))};
    });

    table.register_kernel(OpId::AvgPool2dForward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchAvgPool2dForward(inputs[0], attrs)};
    });

    table.register_kernel(OpId::AvgPool2dBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchAvgPool2dBackward(inputs[0], inputs[1], attrs)};
    });

    table.register_kernel(OpId::AdaptiveMaxPool2d, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t out_h = attrs.get_int(AttrKey::OutputSizeH, 0);
        int64_t out_w = attrs.get_int(AttrKey::OutputSizeW, 0);
        auto [output, indices] = get_vulkan_backend()->dispatchAdaptiveMaxPool2d(inputs[0], out_h, out_w);
        return std::vector<Tensor>{output, indices};
    });

    table.register_kernel(OpId::AdaptiveAvgPool2d, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t out_h = attrs.get_int(AttrKey::OutputSizeH, 0);
        int64_t out_w = attrs.get_int(AttrKey::OutputSizeW, 0);
        return std::vector<Tensor>{get_vulkan_backend()->dispatchAdaptiveAvgPool2d(inputs[0], out_h, out_w)};
    });

    table.register_kernel(OpId::AdaptiveAvgPool2dBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchAdaptiveAvgPool2dBackward(
            inputs[0], attrs.get_int(AttrKey::InputH, 0), attrs.get_int(AttrKey::InputW, 0))};
    });

    // ========================================================================
    // Convolution Operations
    // ========================================================================
    table.register_kernel(OpId::Conv2dForward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        const Tensor* bias_ptr = inputs.size() >= 3 ? &inputs[2] : nullptr;
        return std::vector<Tensor>{get_vulkan_backend()->dispatchConv2dForward(inputs[0], inputs[1], bias_ptr, attrs)};
    });

    table.register_kernel(OpId::Conv2dBackwardInput, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchConv2dBackwardInput(
            inputs[0], inputs[1],
            attrs.get_int(AttrKey::Stride, 1), attrs.get_int(AttrKey::Padding, 0),
            attrs.get_int(AttrKey::Dilation, 1), attrs.get_int_list(AttrKey::InputShape),
            attrs.get_int(AttrKey::Groups, 1))};
    });

    table.register_kernel(OpId::Conv2dBackwardWeight, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchConv2dBackwardWeight(
            inputs[0], inputs[1],
            attrs.get_int(AttrKey::Stride, 1), attrs.get_int(AttrKey::Padding, 0),
            attrs.get_int(AttrKey::Dilation, 1), attrs.get_int_list(AttrKey::WeightShape),
            attrs.get_int(AttrKey::Groups, 1))};
    });

    table.register_kernel(OpId::Conv2dBackwardBias, [](std::span<const Tensor> inputs, const OpAttributes&) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchConv2dBackwardBias(inputs[0])};
    });

    table.register_kernel(OpId::ConvTranspose2dForward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        const Tensor* bias_ptr = inputs.size() >= 3 ? &inputs[2] : nullptr;
        return std::vector<Tensor>{get_vulkan_backend()->dispatchConvTranspose2dForward(
            inputs[0], inputs[1], bias_ptr, attrs)};
    });

    // ========================================================================
    // BatchNorm Operations
    // ========================================================================
    table.register_kernel(OpId::BatchNorm2dForward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        float eps = static_cast<float>(attrs.get_float(AttrKey::Eps, 1e-5));
        const Tensor* gamma = inputs.size() > 3 ? &inputs[3] : nullptr;
        const Tensor* beta = inputs.size() > 4 ? &inputs[4] : nullptr;
        return std::vector<Tensor>{get_vulkan_backend()->dispatchBatchNorm2dForward(
            inputs[0], inputs[1], inputs[2], gamma, beta, eps)};
    });

    table.register_kernel(OpId::BatchNorm2dForwardAffine, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        float eps = static_cast<float>(attrs.get_float(AttrKey::Eps, 1e-5));
        return std::vector<Tensor>{get_vulkan_backend()->dispatchBatchNorm2dForward(
            inputs[0], inputs[1], inputs[2], &inputs[3], &inputs[4], eps)};
    });

    table.register_kernel(OpId::BatchNorm2dBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        float eps = static_cast<float>(attrs.get_float(AttrKey::Eps, 1e-5));
        auto [grad_input, grad_gamma, grad_beta] = get_vulkan_backend()->dispatchBatchNorm2dBackward(
            inputs[0], inputs[1], inputs[3], inputs[4], &inputs[2], eps);
        return std::vector<Tensor>{grad_input, grad_gamma, grad_beta};
    });

    table.register_kernel(OpId::BatchNorm2dMeanVar, [](std::span<const Tensor> inputs, const OpAttributes&) {
        auto [mean, variance] = get_vulkan_backend()->dispatchBatchNorm2dMeanVar(inputs[0]);
        return std::vector<Tensor>{mean, variance};
    });

    table.register_kernel(OpId::BatchNorm2dUpdateRunningStats, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return get_vulkan_backend()->dispatchBatchNorm2dUpdateRunningStats(inputs, attrs);
    });

    // ========================================================================
    // Vision Operations
    // ========================================================================
    table.register_kernel(OpId::Unfold, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchIm2Col(inputs[0], attrs)};
    });

    table.register_kernel(OpId::Fold, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchCol2Im(inputs[0], attrs)};
    });

    table.register_kernel(OpId::ROIAlignForward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchROIAlignForward(inputs[0], inputs[1], attrs)};
    });

    table.register_kernel(OpId::ROIAlignBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchROIAlignBackward(inputs[0], inputs[1], attrs)};
    });

    table.register_kernel(OpId::GatherRelativePositionBias, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchGatherRelativePositionBias(
            inputs[0], inputs[1],
            attrs.get_int(AttrKey::NumPositions, 0), attrs.get_int(AttrKey::NumHeads, 0))};
    });

    // ========================================================================
    // Normalization Operations
    // ========================================================================
    table.register_kernel(OpId::LayerNorm, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        auto normalized_shape = attrs.get_int_list(AttrKey::NormalizedShape);
        int64_t normalized_size = 1;
        for (auto s : normalized_shape) normalized_size *= s;
        if (normalized_size <= 0) normalized_size = inputs[0].shape().back();
        float eps = static_cast<float>(attrs.get_float(AttrKey::Eps, 1e-5));
        const Tensor* gamma = inputs.size() > 1 ? &inputs[1] : nullptr;
        const Tensor* beta = inputs.size() > 2 ? &inputs[2] : nullptr;
        return std::vector<Tensor>{get_vulkan_backend()->dispatchLayerNorm(
            inputs[0], normalized_size, gamma, beta, eps)};
    });

    table.register_kernel(OpId::LayerNormBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t normalized_shape = attrs.get_int(AttrKey::NormalizedShape, inputs[0].shape().back());
        auto [grad_input, grad_weight, grad_bias] = get_vulkan_backend()->dispatchLayerNormBackward(
            inputs[0], inputs[1], inputs[2], inputs[3], &inputs[4], normalized_shape);
        return std::vector<Tensor>{grad_input, grad_weight, grad_bias};
    });

    table.register_kernel(OpId::GroupNorm, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t num_groups = attrs.get_int(AttrKey::Groups, 1);
        float eps = static_cast<float>(attrs.get_float(AttrKey::Eps, 1e-5));
        const Tensor* gamma = inputs.size() > 1 ? &inputs[1] : nullptr;
        const Tensor* beta = inputs.size() > 2 ? &inputs[2] : nullptr;
        return get_vulkan_backend()->dispatchGroupNorm(inputs[0], num_groups, gamma, beta, eps);
    });

    table.register_kernel(OpId::GroupNormBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t num_groups = attrs.get_int(AttrKey::Groups, 1);
        auto [grad_input, grad_weight, grad_bias] = get_vulkan_backend()->dispatchGroupNormBackward(
            inputs[0], inputs[1], inputs[2], inputs[3], &inputs[4], num_groups);
        return std::vector<Tensor>{grad_input, grad_weight, grad_bias};
    });

    table.register_kernel(OpId::FusedRMSNorm, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        float eps = static_cast<float>(attrs.get_float(AttrKey::Eps, 1e-5));
        int64_t normalized_shape = inputs[0].shape().back();
        auto [output, rrms] = get_vulkan_backend()->dispatchRMSNorm(
            inputs[0], inputs[1], normalized_shape, eps);
        return std::vector<Tensor>{output, rrms};
    });

    table.register_kernel(OpId::RMSNormBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t normalized_shape = attrs.get_int(AttrKey::NormalizedShape, inputs[0].shape().back());
        auto [grad_input, grad_weight] = get_vulkan_backend()->dispatchRMSNormBackward(
            inputs[0], inputs[1], inputs[2], inputs[3], normalized_shape);
        return std::vector<Tensor>{grad_input, grad_weight};
    });

    // ========================================================================
    // Embedding Backward
    // ========================================================================
    table.register_kernel(OpId::EmbeddingBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t num_embeddings = attrs.get_int(AttrKey::NumEmbeddings, 0);
        int64_t embedding_dim = inputs[0].shape().back();
        return std::vector<Tensor>{get_vulkan_backend()->dispatchEmbeddingBackward(
            inputs[0], inputs[1], num_embeddings, embedding_dim)};
    });

    // ========================================================================
    // Fused Optimizer Steps
    // ========================================================================
    table.register_kernel(OpId::FusedRMSPropStep, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return get_vulkan_backend()->dispatchFusedRMSPropStep(inputs, attrs);
    });

    table.register_kernel(OpId::FusedAdadeltaStep, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return get_vulkan_backend()->dispatchFusedAdadeltaStep(inputs, attrs);
    });

    table.register_kernel(OpId::FusedAdagradStep, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return get_vulkan_backend()->dispatchFusedAdagradStep(inputs, attrs);
    });

    // ========================================================================
    // Nonzero, OneHot, BoxIoU
    // ========================================================================
    table.register_kernel(OpId::Nonzero, [](std::span<const Tensor> inputs, const OpAttributes&) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchNonzero(inputs[0])};
    });

    table.register_kernel(OpId::OneHot, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchOneHot(inputs[0],
            attrs.get_int(AttrKey::NumClasses, 10))};
    });

    table.register_kernel(OpId::BoxIoU, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchBoxIoU(inputs[0], inputs[1],
            attrs.get_int(AttrKey::IouType, 0))};
    });

    // ========================================================================
    // In-place Activation Operations
    // ========================================================================
    table.register_inplace_kernel(OpId::ReLUInplace, [](Tensor& target, std::span<const Tensor>, const OpAttributes&) -> Tensor& {
        auto vk = get_vulkan_backend();
        auto result = vk->dispatchActivation("relu", target, 0, 0.0f);
        auto bytes = target.numel() * dtype_size(target.dtype());
        vk->copy(target.data_ptr(), result.data_ptr(), bytes, CopyKind::DeviceToDevice);
        return target;
    });

    table.register_inplace_kernel(OpId::SigmoidInplace, [](Tensor& target, std::span<const Tensor>, const OpAttributes&) -> Tensor& {
        auto vk = get_vulkan_backend();
        auto result = vk->dispatchActivation("sigmoid", target, 1, 0.0f);
        auto bytes = target.numel() * dtype_size(target.dtype());
        vk->copy(target.data_ptr(), result.data_ptr(), bytes, CopyKind::DeviceToDevice);
        return target;
    });

    table.register_inplace_kernel(OpId::TanhInplace, [](Tensor& target, std::span<const Tensor>, const OpAttributes&) -> Tensor& {
        auto vk = get_vulkan_backend();
        auto result = vk->dispatchActivation("tanh", target, 2, 0.0f);
        auto bytes = target.numel() * dtype_size(target.dtype());
        vk->copy(target.data_ptr(), result.data_ptr(), bytes, CopyKind::DeviceToDevice);
        return target;
    });

    table.register_inplace_kernel(OpId::LeakyReLUInplace, [](Tensor& target, std::span<const Tensor>, const OpAttributes& attrs) -> Tensor& {
        auto vk = get_vulkan_backend();
        auto result = vk->dispatchActivation("leaky_relu", target, 4, static_cast<float>(attrs.get_float(AttrKey::Alpha, 0.01)));
        auto bytes = target.numel() * dtype_size(target.dtype());
        vk->copy(target.data_ptr(), result.data_ptr(), bytes, CopyKind::DeviceToDevice);
        return target;
    });

    table.register_inplace_kernel(OpId::GeluInplace, [](Tensor& target, std::span<const Tensor>, const OpAttributes&) -> Tensor& {
        auto vk = get_vulkan_backend();
        auto result = vk->dispatchActivation("gelu", target, 3, 0.0f);
        auto bytes = target.numel() * dtype_size(target.dtype());
        vk->copy(target.data_ptr(), result.data_ptr(), bytes, CopyKind::DeviceToDevice);
        return target;
    });

    // ========================================================================
    // Fused Operations
    // ========================================================================
    table.register_kernel(OpId::FusedLinearReLU, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        auto vk = get_vulkan_backend();
        bool has_bias = attrs.get_string(AttrKey::Mode) == "true";
        auto input_shape = inputs[0].shape();
        int64_t in_features = input_shape.back();
        int64_t batch_size = 1;
        for (size_t i = 0; i < input_shape.size() - 1; ++i) batch_size *= input_shape[i];
        auto input_2d = vk->dispatchReshape(inputs[0], {batch_size, in_features});
        auto weight_t = vk->dispatchTranspose(inputs[1], 0, 1);
        auto mm_result = vk->dispatchMatmul(input_2d, weight_t);
        if (has_bias && inputs.size() > 2) {
            mm_result = vk->dispatchBinaryOp("add", mm_result, inputs[2]);
        }
        mm_result = vk->dispatchActivation("relu", mm_result, 0, 0.0f);
        auto out_features = inputs[1].shape()[0];
        std::vector<int64_t> output_shape(input_shape.begin(), input_shape.end() - 1);
        output_shape.push_back(out_features);
        return std::vector<Tensor>{vk->dispatchReshape(mm_result, output_shape)};
    });

    table.register_kernel(OpId::FusedBatchNormReLU, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        auto vk = get_vulkan_backend();
        float eps = static_cast<float>(attrs.get_float(AttrKey::Eps, 1e-5));
        auto input = inputs[0];
        auto shape = input.shape();
        bool needs_reshape = (shape.size() != 4);
        std::vector<int64_t> original_shape(shape.begin(), shape.end());
        if (needs_reshape) {
            int64_t N = shape[0], C = shape[1];
            int64_t spatial = 1;
            for (size_t i = 2; i < shape.size(); ++i) spatial *= shape[i];
            input = vk->dispatchReshape(input, {N, C, spatial, 1});
        }
        auto bn_result = vk->dispatchBatchNorm2dForward(input, inputs[1], inputs[2], &inputs[3], &inputs[4], eps);
        if (needs_reshape) {
            bn_result = vk->dispatchReshape(bn_result, original_shape);
        }
        return std::vector<Tensor>{vk->dispatchActivation("relu", bn_result, 0, 0.0f)};
    });

    table.register_kernel(OpId::FusedAddReLU, [](std::span<const Tensor> inputs, const OpAttributes&) {
        auto vk = get_vulkan_backend();
        auto sum = vk->dispatchBinaryOp("add", inputs[0], inputs[1]);
        return std::vector<Tensor>{vk->dispatchActivation("relu", sum, 0, 0.0f)};
    });

    table.register_kernel(OpId::FusedGelu, [](std::span<const Tensor> inputs, const OpAttributes&) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchActivation("gelu", inputs[0], 3, 0.0f)};
    });

    table.register_kernel(OpId::FusedLayerNorm, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        auto normalized_shape = attrs.get_int_list(AttrKey::NormalizedShape);
        int64_t normalized_size = 1;
        for (auto s : normalized_shape) normalized_size *= s;
        if (normalized_size <= 0) normalized_size = inputs[0].shape().back();
        float eps = static_cast<float>(attrs.get_float(AttrKey::Eps, 1e-5));
        return std::vector<Tensor>{get_vulkan_backend()->dispatchLayerNorm(
            inputs[0], normalized_size, &inputs[1], &inputs[2], eps)};
    });

    // ========================================================================
    // Interpolation
    // ========================================================================
    table.register_kernel(OpId::Interpolate, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchInterpolate(inputs[0], attrs)};
    });

    // ========================================================================
    // ArgSort
    // ========================================================================
    table.register_kernel(OpId::ArgSort, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchArgSort(inputs[0],
            attrs.get_int(AttrKey::Dim, -1), attrs.get_bool(AttrKey::Descending, false))};
    });

    // ========================================================================
    // Linear/FC Operations
    // ========================================================================
    table.register_single_output_kernel(OpId::Linear, [](std::span<const Tensor> inputs, const OpAttributes&) -> Tensor {
        // inputs: input, weight, [bias]
        const Tensor* bias = inputs.size() > 2 ? &inputs[2] : nullptr;
        return get_vulkan_backend()->dispatchLinear(inputs[0], inputs[1], bias);
    });

    table.register_kernel(OpId::LinearBackward, [](std::span<const Tensor> inputs, const OpAttributes&) {
        // inputs: grad_output, input, weight
        return get_vulkan_backend()->dispatchLinearBackward(inputs[0], inputs[1], inputs[2]);
    });

    // ========================================================================
    // Dropout Operations
    // ========================================================================
    table.register_kernel(OpId::Dropout, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        float p = static_cast<float>(attrs.get_float(AttrKey::P, 0.5));
        bool training = attrs.get_bool(AttrKey::Training, true);
        auto [output, mask] = get_vulkan_backend()->dispatchDropout(inputs[0], p, training);
        return std::vector<Tensor>{output, mask};
    });

    table.register_single_output_kernel(OpId::DropoutBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        float p = static_cast<float>(attrs.get_float(AttrKey::P, 0.5));
        return get_vulkan_backend()->dispatchDropoutBackward(inputs[0], inputs[1], p);
    });

    // ========================================================================
    // Slice/Split/Chunk/Flatten Operations
    // ========================================================================
    table.register_single_output_kernel(OpId::Slice, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        auto starts = attrs.get_int_list(AttrKey::Starts);
        auto ends = attrs.get_int_list(AttrKey::Ends);
        auto steps = attrs.get_int_list(AttrKey::Steps);
        return get_vulkan_backend()->dispatchSlice(inputs[0], starts, ends, steps);
    });

    table.register_kernel(OpId::Split, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t split_size = attrs.get_int(AttrKey::SplitSize, 1);
        int64_t dim = attrs.get_int(AttrKey::Dim, 0);
        return get_vulkan_backend()->dispatchSplit(inputs[0], split_size, dim);
    });

    table.register_kernel(OpId::Chunk, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t chunks = attrs.get_int(AttrKey::Chunks, 1);
        int64_t dim = attrs.get_int(AttrKey::Dim, 0);
        return get_vulkan_backend()->dispatchChunk(inputs[0], chunks, dim);
    });

    table.register_single_output_kernel(OpId::Flatten, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t start_dim = attrs.get_int(AttrKey::StartDim, 0);
        int64_t end_dim = attrs.get_int(AttrKey::EndDim, -1);
        return get_vulkan_backend()->dispatchFlatten(inputs[0], start_dim, end_dim);
    });

    // ========================================================================
    // Type Conversion (Cast)
    // ========================================================================
    table.register_single_output_kernel(OpId::Cast, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        if (!attrs.has(AttrKey::TargetDtype)) {
            throw std::runtime_error("vulkan cast: missing 'target_dtype' attribute");
        }
        DType target_dtype = static_cast<DType>(attrs.get_int(AttrKey::TargetDtype));
        return get_vulkan_backend()->dispatchCast(inputs[0], target_dtype);
    });

    // ========================================================================
    // Phase 11.3: RNN Operations
    // ========================================================================
    table.register_kernel(OpId::LSTMForward, [](std::span<const Tensor> inputs, const OpAttributes&) {
        return get_vulkan_backend()->dispatchLSTMForward(
            inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5], inputs[6]);
    });

    table.register_kernel(OpId::GRUForward, [](std::span<const Tensor> inputs, const OpAttributes&) {
        return get_vulkan_backend()->dispatchGRUForward(
            inputs[0], inputs[1], inputs[2], inputs[3], inputs[4]);
    });

    table.register_kernel(OpId::LSTMMultiLayerForward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t num_layers = attrs.get_int(AttrKey::NumLayers, 1);
        const Tensor& input = inputs[0];
        const Tensor& h0 = inputs[1];
        const Tensor& c0 = inputs[2];

        std::vector<Tensor> W_ih_list, W_hh_list, bias_list;
        for (int64_t l = 0; l < num_layers; ++l) {
            size_t base_idx = 3 + l * 3;
            W_ih_list.push_back(inputs[base_idx]);
            W_hh_list.push_back(inputs[base_idx + 1]);
            bias_list.push_back(inputs[base_idx + 2]);
        }

        return get_vulkan_backend()->dispatchLSTMMultiLayerForward(
            input, W_ih_list, W_hh_list, bias_list, h0, c0);
    });

    table.register_kernel(OpId::GRUMultiLayerForward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t num_layers = attrs.get_int(AttrKey::NumLayers, 1);
        const Tensor& input = inputs[0];
        const Tensor& h0 = inputs[1];

        std::vector<Tensor> W_ih_list, W_hh_list, bias_list;
        for (int64_t l = 0; l < num_layers; ++l) {
            size_t base_idx = 2 + l * 3;
            W_ih_list.push_back(inputs[base_idx]);
            W_hh_list.push_back(inputs[base_idx + 1]);
            bias_list.push_back(inputs[base_idx + 2]);
        }

        return get_vulkan_backend()->dispatchGRUMultiLayerForward(
            input, W_ih_list, W_hh_list, bias_list, h0);
    });

    table.register_kernel(OpId::BiLSTMForward, [](std::span<const Tensor> inputs, const OpAttributes&) {
        return get_vulkan_backend()->dispatchBiLSTMForward(
            inputs[0], inputs[1], inputs[2],
            inputs[3], inputs[4], inputs[5], inputs[6],
            inputs[7], inputs[8], inputs[9], inputs[10]);
    });

    // ========================================================================
    // Phase 11.4: Sorting Operations
    // ========================================================================
    table.register_kernel(OpId::Sort, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t dim = attrs.get_int(AttrKey::Dim, -1);
        bool descending = attrs.get_bool(AttrKey::Descending, false);
        auto [values, indices] = get_vulkan_backend()->dispatchSort(inputs[0], dim, descending);
        return std::vector<Tensor>{values, indices};
    });

    table.register_kernel(OpId::TopK, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t k = attrs.get_int(AttrKey::K, 1);
        int64_t dim = attrs.get_int(AttrKey::Dim, -1);
        bool largest = attrs.get_bool(AttrKey::Largest, true);
        bool sorted = attrs.get_bool(AttrKey::Sorted, true);
        auto [values, indices] = get_vulkan_backend()->dispatchTopK(inputs[0], k, dim, largest, sorted);
        return std::vector<Tensor>{values, indices};
    });

    table.register_kernel(OpId::Unique, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        bool sorted = attrs.get_bool(AttrKey::Sorted, true);
        bool return_inverse = attrs.get_bool(AttrKey::ReturnInverse, false);
        bool return_counts = attrs.get_bool(AttrKey::ReturnCounts, false);
        return get_vulkan_backend()->dispatchUnique(inputs[0], sorted, return_inverse, return_counts);
    });

    // ========================================================================
    // Phase 11.5: Misc Operations
    // ========================================================================
    table.register_inplace_kernel(OpId::StridedFill, [](Tensor& self, std::span<const Tensor>, const OpAttributes& attrs) -> Tensor& {
        float value = static_cast<float>(attrs.get_float(AttrKey::Value, 0.0));
        get_vulkan_backend()->dispatchStridedFill(self, value);
        return self;
    });

    table.register_single_output_kernel(OpId::ToMemoryFormat, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int format_int = static_cast<int>(attrs.get_int(AttrKey::MemoryFormat, 0));
        return get_vulkan_backend()->dispatchToMemoryFormat(inputs[0], format_int);
    });

    table.register_kernel(OpId::HasInfNan, [](std::span<const Tensor> inputs, const OpAttributes&) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchHasInfNan(inputs[0])};
    });

    table.register_single_output_kernel(OpId::DepthwiseConv2d, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t stride = attrs.get_int(AttrKey::Stride, 1);
        int64_t padding = attrs.get_int(AttrKey::Padding, 0);
        int64_t dilation = attrs.get_int(AttrKey::Dilation, 1);
        const Tensor* bias = inputs.size() > 2 ? &inputs[2] : nullptr;
        return get_vulkan_backend()->dispatchDepthwiseConv2d(inputs[0], inputs[1], bias, stride, padding, dilation);
    });

    table.register_single_output_kernel(OpId::AdaptiveMaxPool2dBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        auto input_shape = attrs.get_int_list(AttrKey::InputShape);
        return get_vulkan_backend()->dispatchAdaptiveMaxPool2dBackward(inputs[0], inputs[1], input_shape);
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
