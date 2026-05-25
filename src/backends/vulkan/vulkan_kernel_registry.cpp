/**
 * @file vulkan_kernel_registry.cpp
 * @brief Vulkan kernel registration for O(1) dispatch
 *
 * Registers all Vulkan kernel implementations with the dispatch table.
 * Each kernel wrapper calls the corresponding VulkanBackend typed dispatch
 * method directly, bypassing string-based dispatch for O(1) lookup.
 */

#include "tenzor/backend/dispatch_table.hpp"
#include "tenzor/nn/layers/flex_attention.hpp"  // Wave C: process-wide score_mod registry
#include <sstream>
#include "tenzor/backend/kernel_registry.hpp"
#include "tenzor/backend/fast_dispatch.hpp"
#include "tenzor/backend/op_attributes.hpp"
#include "tenzor/backend/attr_macros.hpp"
#include "tenzor/ops/op_id.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/vision.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/linalg.hpp"
#include "tenzor/ops/reduction.hpp"
#include "tenzor/ops/philox_dropout.hpp"   // F13/F22-followup: Philox-keyed dropout
#include "tenzor/ops/fft.hpp"
#include "tenzor/ops/advanced.hpp"
#include "tenzor/ops/indexing.hpp"
#include "tenzor/ops/transform.hpp"
#include "tenzor/sparse/sparse_ops.hpp"
#include "vulkan_backend.hpp"
#include <cstdlib>
#include <cstring>
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
    if (s == "complex64") return DType::Complex64;
    if (s == "complex128") return DType::Complex128;
    if (s.empty()) return default_val;
    return default_val;
}

// Helper: extract normalized_shape from attrs (may be string or int list)
static int64_t vulkan_extract_normalized_size(const OpAttributes& attrs, const Tensor& fallback) {
    auto ns_str = attrs.get_string(AttrKey::NormalizedShape);
    if (!ns_str.empty()) {
        return std::stoll(std::string(ns_str));
    }
    auto ns_list = attrs.get_int_list(AttrKey::NormalizedShape);
    if (!ns_list.empty()) {
        int64_t sz = 1;
        for (auto s : ns_list) sz *= s;
        if (sz > 0) return sz;
    }
    return fallback.shape().back();
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
        auto vk = get_vulkan_backend();
        auto result = vk->dispatchBinaryOp("add", target, others[0]);
        auto bytes = target.numel() * dtype_size(target.dtype());
        if (bytes > 0) vk->copy(target.data_ptr(), result.data_ptr(), bytes, CopyKind::DeviceToDevice);
        return target;
    });

    table.register_inplace_kernel(OpId::SubInplace, [](Tensor& target, std::span<const Tensor> others, const OpAttributes&) -> Tensor& {
        auto vk = get_vulkan_backend();
        auto result = vk->dispatchBinaryOp("sub", target, others[0]);
        auto bytes = target.numel() * dtype_size(target.dtype());
        if (bytes > 0) vk->copy(target.data_ptr(), result.data_ptr(), bytes, CopyKind::DeviceToDevice);
        return target;
    });

    table.register_inplace_kernel(OpId::MulInplace, [](Tensor& target, std::span<const Tensor> others, const OpAttributes&) -> Tensor& {
        auto vk = get_vulkan_backend();
        auto result = vk->dispatchBinaryOp("mul", target, others[0]);
        auto bytes = target.numel() * dtype_size(target.dtype());
        if (bytes > 0) vk->copy(target.data_ptr(), result.data_ptr(), bytes, CopyKind::DeviceToDevice);
        return target;
    });

    table.register_inplace_kernel(OpId::DivInplace, [](Tensor& target, std::span<const Tensor> others, const OpAttributes&) -> Tensor& {
        auto vk = get_vulkan_backend();
        auto result = vk->dispatchBinaryOp("div", target, others[0]);
        auto bytes = target.numel() * dtype_size(target.dtype());
        if (bytes > 0) vk->copy(target.data_ptr(), result.data_ptr(), bytes, CopyKind::DeviceToDevice);
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
            static_cast<float>(attrs.get_float(AttrKey::Min, 0.0)),
            std::numeric_limits<float>::infinity())};
    });

    table.register_kernel(OpId::ClampMax, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchClamp(inputs[0],
            -std::numeric_limits<float>::infinity(),
            static_cast<float>(attrs.get_float(AttrKey::Max, std::numeric_limits<float>::infinity())))};
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
            attrs.get_int(AttrKey::Dim, LLONG_MIN), attrs.get_bool(AttrKey::Keepdim, false))};
    });

    table.register_kernel(OpId::Mean, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchReduction("mean", inputs[0],
            attrs.get_int(AttrKey::Dim, LLONG_MIN), attrs.get_bool(AttrKey::Keepdim, false))};
    });

    table.register_kernel(OpId::Max, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchReduction("max", inputs[0],
            attrs.get_int(AttrKey::Dim, LLONG_MIN), attrs.get_bool(AttrKey::Keepdim, false))};
    });

    table.register_kernel(OpId::Min, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchReduction("min", inputs[0],
            attrs.get_int(AttrKey::Dim, LLONG_MIN), attrs.get_bool(AttrKey::Keepdim, false))};
    });

    table.register_kernel(OpId::ArgMax, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchArgmax(inputs[0],
            attrs.get_int(AttrKey::Dim, -1), attrs.get_bool(AttrKey::Keepdim, false))};
    });

    table.register_kernel(OpId::ArgMin, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchArgmin(inputs[0],
            attrs.get_int(AttrKey::Dim, -1), attrs.get_bool(AttrKey::Keepdim, false))};
    });

    // Vulkan reduction kernels use `dim < 0` as the "full reduction"
    // sentinel — that conflicts with valid negative dims (-1, -2…) which
    // refer to axes from the back. Default to INT64_MIN (the project-wide
    // "all dims" sentinel) and normalize user-specified negative dims to
    // positive. Pass -1 to dispatchProd/Variance/Std for full reduction.
    table.register_kernel(OpId::Prod, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t dim = attrs.get_int(AttrKey::Dim, INT64_MIN);
        if (dim != INT64_MIN && dim < 0) dim += static_cast<int64_t>(inputs[0].ndim());
        if (dim == INT64_MIN) dim = -1;
        return std::vector<Tensor>{get_vulkan_backend()->dispatchProd(inputs[0],
            dim, attrs.get_bool(AttrKey::Keepdim, false))};
    });

    table.register_kernel(OpId::Var, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t dim = attrs.get_int(AttrKey::Dim, INT64_MIN);
        if (dim != INT64_MIN && dim < 0) dim += static_cast<int64_t>(inputs[0].ndim());
        if (dim == INT64_MIN) dim = -1;
        return std::vector<Tensor>{get_vulkan_backend()->dispatchVariance(inputs[0],
            dim, attrs.get_bool(AttrKey::Unbiased, true),
            attrs.get_bool(AttrKey::Keepdim, false))};
    });

    table.register_kernel(OpId::Std, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t dim = attrs.get_int(AttrKey::Dim, INT64_MIN);
        if (dim != INT64_MIN && dim < 0) dim += static_cast<int64_t>(inputs[0].ndim());
        if (dim == INT64_MIN) dim = -1;
        return std::vector<Tensor>{get_vulkan_backend()->dispatchStd(inputs[0],
            dim, attrs.get_bool(AttrKey::Unbiased, true),
            attrs.get_bool(AttrKey::Keepdim, false))};
    });

    table.register_kernel(OpId::Norm, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchNorm(inputs[0],
            static_cast<float>(attrs.get_float(AttrKey::P, 2.0)), attrs.get_int(AttrKey::Dim, INT64_MIN),
            attrs.get_bool(AttrKey::Keepdim, false))};
    });

    table.register_kernel(OpId::LogSumExp, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t dim = attrs.get_int(AttrKey::Dim, INT64_MIN);
        bool keepdim = attrs.get_bool(AttrKey::Keepdim, false);
        return std::vector<Tensor>{get_vulkan_backend()->dispatchLogSumExp(inputs[0], dim, keepdim)};
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
        return std::vector<Tensor>{get_vulkan_backend()->dispatchActivation("leaky_relu", inputs[0], 4, attrs.get_float(AttrKey::Alpha, 0.01))};
    });

    table.register_kernel(OpId::LeakyReLUBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchActivationBackward("leaky_relu_backward", inputs[0], inputs[1], 3, attrs.get_float(AttrKey::Alpha, 0.01))};
    });

    table.register_kernel(OpId::Swish, [](std::span<const Tensor> inputs, const OpAttributes&) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchActivation("swish", inputs[0], 5, 0.0f)};
    });

    table.register_kernel(OpId::SwishBackward, [](std::span<const Tensor> inputs, const OpAttributes&) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchSwishBackward(inputs[0], inputs[1])};
    });

    table.register_kernel(OpId::Elu, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchActivation("elu", inputs[0], 6, attrs.get_float(AttrKey::Alpha, 1.0))};
    });

    table.register_kernel(OpId::EluBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchActivationBackward("elu_backward", inputs[0], inputs[1], 5, attrs.get_float(AttrKey::Alpha, 1.0))};
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
        return std::vector<Tensor>{get_vulkan_backend()->dispatchActivation("softplus", inputs[0], 9, attrs.get_float(AttrKey::Beta, 1.0))};
    });

    table.register_kernel(OpId::SoftplusBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchActivationBackward("softplus_backward", inputs[0], inputs[1], 8, attrs.get_float(AttrKey::Beta, 1.0))};
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

    table.register_kernel(OpId::EmbeddingWithBoundsCheck, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
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
        // Pass value as double so Float64 subnormals are preserved.
        return std::vector<Tensor>{get_vulkan_backend()->dispatchFull(attrs.get_int_list(AttrKey::Shape),
            attrs.get_float(AttrKey::Value, 0.0), dtype_from_string(attrs.get_string(AttrKey::Dtype)))};
    });

    table.register_kernel(OpId::Fill, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchFill(inputs[0], attrs.get_float(AttrKey::Value, 0.0))};
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

    table.register_kernel(OpId::Randint, [](std::span<const Tensor>, const OpAttributes& attrs) {
        int64_t low = attrs.get_int(AttrKey::Start, 0);
        int64_t high = attrs.get_int(AttrKey::End, 0);
        auto shape = attrs.get_int_list(AttrKey::Shape);
        DType dtype = dtype_from_string(attrs.get_string(AttrKey::Dtype, "int32"));
        auto device_id = static_cast<int32_t>(attrs.get_int(AttrKey::Device, 0));
        Device device(Device::Type::Vulkan, device_id);
        return std::vector<Tensor>{get_vulkan_backend()->dispatchRandint(low, high, shape, dtype, device)};
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
        // F.11: per-axis kernel/stride/padding via attr_macros helpers.
        // Note: dispatchMaxPool2d (with-indices overload) does not take dilation;
        // nn::MaxPool2d layer does not expose dilation, so this is intentional.
        const auto kernel_size = ::tenzor::backend::attrs::kernel_size_2d(attrs);
        const auto stride      = ::tenzor::backend::attrs::stride_2d(attrs);
        const auto padding     = ::tenzor::backend::attrs::padding_2d(attrs);
        auto [output, indices] = get_vulkan_backend()->dispatchMaxPool2d(
            inputs[0], kernel_size[0], kernel_size[1], stride[0], stride[1], padding[0], padding[1]);
        return std::vector<Tensor>{output, indices};
    });

    table.register_kernel(OpId::MaxPool2dBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        // inputs: [grad_output, indices, input] from autograd
        // Vulkan backward shader re-computes max positions from original input (inputs[2]),
        // rather than using pre-computed indices (inputs[1])
        return std::vector<Tensor>{get_vulkan_backend()->dispatchMaxPool2dBackward(
            inputs[0], inputs[2], attrs)};
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
    // 1D Pooling Operations
    // ========================================================================
    table.register_kernel(OpId::MaxPool1dForward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return get_vulkan_backend()->dispatchMaxPool1dForward(inputs[0], attrs);
    });

    table.register_kernel(OpId::MaxPool1dBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        auto input_shape = attrs.get_int_list(AttrKey::InputShape);
        int64_t L_in = (input_shape.size() == 3) ? input_shape[2] : 0;
        return std::vector<Tensor>{get_vulkan_backend()->dispatchMaxPool1dBackward(
            inputs[0], inputs[1], L_in)};
    });

    table.register_kernel(OpId::AvgPool1dForward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchAvgPool1dForward(inputs[0], attrs)};
    });

    table.register_kernel(OpId::AvgPool1dBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        // The Vulkan kernel needs an input tensor for shape/dtype/device. Construct
        // a placeholder from the InputShape attribute set by the autograd Function wrapper.
        auto input_shape = attrs.get_int_list(AttrKey::InputShape);
        Tensor placeholder(input_shape, inputs[0].dtype(), inputs[0].device());
        return std::vector<Tensor>{get_vulkan_backend()->dispatchAvgPool1dBackward(inputs[0], placeholder, attrs)};
    });

    table.register_kernel(OpId::AdaptiveMaxPool1d, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t out_size = attrs.get_int(AttrKey::OutputSize, 0);
        auto [output, indices] = get_vulkan_backend()->dispatchAdaptiveMaxPool1d(inputs[0], out_size);
        return std::vector<Tensor>{output, indices};
    });

    table.register_kernel(OpId::AdaptiveAvgPool1d, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t out_size = attrs.get_int(AttrKey::OutputSize, 0);
        return std::vector<Tensor>{get_vulkan_backend()->dispatchAdaptiveAvgPool1d(inputs[0], out_size)};
    });

    table.register_kernel(OpId::AdaptiveAvgPool1dBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        auto input_shape = attrs.get_int_list(AttrKey::InputShape);
        int64_t L_in = (input_shape.size() == 3) ? input_shape[2] : 0;
        return std::vector<Tensor>{get_vulkan_backend()->dispatchAdaptiveAvgPool1dBackward(
            inputs[0], L_in)};
    });

    table.register_single_output_kernel(OpId::AdaptiveMaxPool1dBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        auto input_shape = attrs.get_int_list(AttrKey::InputShape);
        return get_vulkan_backend()->dispatchAdaptiveMaxPool1dBackward(inputs[0], inputs[1], input_shape);
    });

    // ========================================================================
    // 3D Pooling Operations
    // ========================================================================
    table.register_kernel(OpId::MaxPool3dForward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return get_vulkan_backend()->dispatchMaxPool3dForward(inputs[0], attrs);
    });

    table.register_kernel(OpId::MaxPool3dBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        // Read DHW from InputShape attribute (set by autograd Function wrapper).
        auto input_shape = attrs.get_int_list(AttrKey::InputShape);
        int64_t D_in = (input_shape.size() == 5) ? input_shape[2] : 0;
        int64_t H_in = (input_shape.size() == 5) ? input_shape[3] : 0;
        int64_t W_in = (input_shape.size() == 5) ? input_shape[4] : 0;
        return std::vector<Tensor>{get_vulkan_backend()->dispatchMaxPool3dBackward(
            inputs[0], inputs[1], D_in, H_in, W_in)};
    });

    table.register_kernel(OpId::AvgPool3dForward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchAvgPool3dForward(inputs[0], attrs)};
    });

    table.register_kernel(OpId::AvgPool3dBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        // Construct placeholder input tensor from InputShape if caller only passed grad_output.
        auto input_shape = attrs.get_int_list(AttrKey::InputShape);
        if (input_shape.empty() || inputs.size() >= 2) {
            return std::vector<Tensor>{get_vulkan_backend()->dispatchAvgPool3dBackward(inputs[0], inputs[1], attrs)};
        }
        Tensor placeholder(input_shape, inputs[0].dtype(), inputs[0].device());
        return std::vector<Tensor>{get_vulkan_backend()->dispatchAvgPool3dBackward(inputs[0], placeholder, attrs)};
    });

    table.register_kernel(OpId::AdaptiveMaxPool3d, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t out_d = attrs.get_int(AttrKey::OutputSizeD, 0);
        int64_t out_h = attrs.get_int(AttrKey::OutputSizeH, 0);
        int64_t out_w = attrs.get_int(AttrKey::OutputSizeW, 0);
        auto [output, indices] = get_vulkan_backend()->dispatchAdaptiveMaxPool3d(inputs[0], out_d, out_h, out_w);
        return std::vector<Tensor>{output, indices};
    });

    table.register_kernel(OpId::AdaptiveAvgPool3d, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t out_d = attrs.get_int(AttrKey::OutputSizeD, 0);
        int64_t out_h = attrs.get_int(AttrKey::OutputSizeH, 0);
        int64_t out_w = attrs.get_int(AttrKey::OutputSizeW, 0);
        return std::vector<Tensor>{get_vulkan_backend()->dispatchAdaptiveAvgPool3d(inputs[0], out_d, out_h, out_w)};
    });

    table.register_kernel(OpId::AdaptiveAvgPool3dBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        auto input_shape = attrs.get_int_list(AttrKey::InputShape);
        int64_t D_in = (input_shape.size() == 5) ? input_shape[2] : 0;
        int64_t H_in = (input_shape.size() == 5) ? input_shape[3] : 0;
        int64_t W_in = (input_shape.size() == 5) ? input_shape[4] : 0;
        return std::vector<Tensor>{get_vulkan_backend()->dispatchAdaptiveAvgPool3dBackward(
            inputs[0], D_in, H_in, W_in)};
    });

    table.register_single_output_kernel(OpId::AdaptiveMaxPool3dBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        auto input_shape = attrs.get_int_list(AttrKey::InputShape);
        return get_vulkan_backend()->dispatchAdaptiveMaxPool3dBackward(inputs[0], inputs[1], input_shape);
    });

    // ========================================================================
    // Convolution Operations
    // ========================================================================
    // Wave B1/B2: Vulkan Conv2dForward natively supports per-axis stride/
    // padding/dilation via per-axis push-constants in conv2d_forward.comp
    // (and its f16/bf16/f64 variants). dispatchConv2dForward reads per-axis
    // attrs internally with scalar fallback.
    table.register_kernel(OpId::Conv2dForward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        const Tensor* bias_ptr = inputs.size() >= 3 ? &inputs[2] : nullptr;
        return std::vector<Tensor>{get_vulkan_backend()->dispatchConv2dForward(inputs[0], inputs[1], bias_ptr, attrs)};
    });

    // Conv2dBackwardInput: inputs = {grad_output, input, weight}
    // F.11: per-axis stride/padding/dilation via attr_macros helpers.
    table.register_kernel(OpId::Conv2dBackwardInput, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        TENZOR_READ_CONV2D_ATTRS();
        return std::vector<Tensor>{get_vulkan_backend()->dispatchConv2dBackwardInput(
            inputs[0], inputs[2],
            stride[0], stride[1], padding[0], padding[1], dilation[0], dilation[1],
            attrs.get_int_list(AttrKey::InputShape),
            groups)};
    });

    // Conv2dBackwardWeight: inputs = {grad_output, input, weight} — per-axis (Wave B1/B2).
    // F.11: per-axis stride/padding/dilation via attr_macros helpers.
    table.register_kernel(OpId::Conv2dBackwardWeight, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        TENZOR_READ_CONV2D_ATTRS();
        return std::vector<Tensor>{get_vulkan_backend()->dispatchConv2dBackwardWeight(
            inputs[0], inputs[1],
            stride[0], stride[1], padding[0], padding[1], dilation[0], dilation[1],
            attrs.get_int_list(AttrKey::WeightShape),
            groups)};
    });

    table.register_kernel(OpId::Conv2dBackwardBias, [](std::span<const Tensor> inputs, const OpAttributes&) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchConv2dBackwardBias(inputs[0])};
    });

    // ========================================================================
    // Conv1d Operations (wrap Conv2d by unsqueezing height dimension)
    // ========================================================================
    // Audit U.4: project scalar Stride/Padding/Dilation onto the W axis
    // only; pin H to neutral (stride=1, padding=0, dilation=1). See the
    // CUDA registry for the full rationale.
    table.register_kernel(OpId::Conv1dForward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
        auto input_4d = inputs[0].unsqueeze(2);
        auto weight_4d = inputs[1].unsqueeze(2);
        std::vector<Tensor> conv2d_inputs = inputs.size() > 2
            ? std::vector<Tensor>{input_4d, weight_4d, inputs[2]}
            : std::vector<Tensor>{input_4d, weight_4d};
        const auto conv2d_attrs = ::tenzor::backend::attrs::conv1d_to_conv2d_attrs(attrs);
        auto result = tenzor::dispatch(OpId::Conv2dForward, conv2d_inputs, conv2d_attrs);
        return {result[0].squeeze(2)};
    });

    table.register_kernel(OpId::Conv1dBackwardInput, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
        auto grad_4d = inputs[0].unsqueeze(2);
        auto input_4d = inputs[1].unsqueeze(2);
        auto weight_4d = inputs[2].unsqueeze(2);
        std::vector<Tensor> conv2d_inputs = {grad_4d, input_4d, weight_4d};
        const auto conv2d_attrs = ::tenzor::backend::attrs::conv1d_to_conv2d_attrs(attrs);
        auto result = tenzor::dispatch(OpId::Conv2dBackwardInput, conv2d_inputs, conv2d_attrs);
        return {result[0].squeeze(2)};
    });

    table.register_kernel(OpId::Conv1dBackwardWeight, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
        auto grad_4d = inputs[0].unsqueeze(2);
        auto input_4d = inputs[1].unsqueeze(2);
        auto weight_4d = inputs[2].unsqueeze(2);
        std::vector<Tensor> conv2d_inputs = {grad_4d, input_4d, weight_4d};
        const auto conv2d_attrs = ::tenzor::backend::attrs::conv1d_to_conv2d_attrs(attrs);
        auto result = tenzor::dispatch(OpId::Conv2dBackwardWeight, conv2d_inputs, conv2d_attrs);
        return {result[0].squeeze(2)};
    });

    table.register_kernel(OpId::Conv1dBackwardBias, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
        auto grad_4d = inputs[0].unsqueeze(2);
        std::vector<Tensor> conv2d_inputs = {grad_4d};
        // U.4: project to per-axis; preserves Stream / other keys.
        const auto conv2d_attrs = ::tenzor::backend::attrs::conv1d_to_conv2d_attrs(attrs);
        auto result = tenzor::dispatch(OpId::Conv2dBackwardBias, conv2d_inputs, conv2d_attrs);
        return {result[0]};
    });

    // ========================================================================
    // Conv3d Operations
    // ========================================================================
    // Wave B1/B2: Vulkan Conv3d natively supports per-axis stride/padding/dilation.
    // The conv3d_forward.comp shader already has per-axis push-constants;
    // dispatchConv3dForward now reads per-axis attrs (with scalar fallback).
    table.register_kernel(OpId::Conv3dForward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        const Tensor* bias_ptr = inputs.size() >= 3 ? &inputs[2] : nullptr;
        return std::vector<Tensor>{get_vulkan_backend()->dispatchConv3dForward(inputs[0], inputs[1], bias_ptr, attrs)};
    });

    // Conv3dBackwardInput: inputs = {grad_output, input, weight} — per-axis (Wave B1/B2).
    // F.11: per-axis stride/padding/dilation via attr_macros helpers.
    table.register_kernel(OpId::Conv3dBackwardInput, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        TENZOR_READ_CONV3D_ATTRS();
        return std::vector<Tensor>{get_vulkan_backend()->dispatchConv3dBackwardInput(
            inputs[0], inputs[2],
            stride[0], stride[1], stride[2],
            padding[0], padding[1], padding[2],
            dilation[0], dilation[1], dilation[2],
            attrs.get_int_list(AttrKey::InputShape),
            groups)};
    });

    // Conv3dBackwardWeight: inputs = {grad_output, input, weight} — per-axis (Wave B1/B2).
    // F.11: per-axis stride/padding/dilation via attr_macros helpers.
    table.register_kernel(OpId::Conv3dBackwardWeight, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        TENZOR_READ_CONV3D_ATTRS();
        return std::vector<Tensor>{get_vulkan_backend()->dispatchConv3dBackwardWeight(
            inputs[0], inputs[1],
            stride[0], stride[1], stride[2],
            padding[0], padding[1], padding[2],
            dilation[0], dilation[1], dilation[2],
            attrs.get_int_list(AttrKey::WeightShape),
            groups)};
    });

    table.register_kernel(OpId::Conv3dBackwardBias, [](std::span<const Tensor> inputs, const OpAttributes&) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchConv3dBackwardBias(inputs[0])};
    });

    // Wave B1/B2: Vulkan ConvTranspose3d natively supports per-axis stride/
    // padding/output_padding/dilation. dispatchConvTranspose3dForward reads
    // per-axis attrs internally (with scalar fallback).
    table.register_kernel(OpId::ConvTranspose3dForward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        const Tensor* bias_ptr = inputs.size() >= 3 ? &inputs[2] : nullptr;
        return std::vector<Tensor>{get_vulkan_backend()->dispatchConvTranspose3dForward(
            inputs[0], inputs[1], bias_ptr, attrs)};
    });

    table.register_kernel(OpId::ConvTranspose3dBackwardInput, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchConvTranspose3dBackwardInput(
            inputs[0], inputs[2], attrs)};
    });

    table.register_kernel(OpId::ConvTranspose3dBackwardWeight, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchConvTranspose3dBackwardWeight(
            inputs[0], inputs[1], attrs.get_int_list(AttrKey::WeightShape), attrs)};
    });

    table.register_kernel(OpId::ConvTranspose3dBackwardBias, [](std::span<const Tensor> inputs, const OpAttributes&) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchConvTranspose3dBackwardBias(inputs[0])};
    });

    // Wave B1/B2: Vulkan ConvTranspose2d natively supports per-axis stride/
    // padding/output_padding/dilation via per-axis push-constants in
    // conv_transpose2d_forward.comp (and f16/bf16/f64 variants).
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

    // GridSample / AffineGrid — native Vulkan compute shaders
    table.register_single_output_kernel(OpId::GridSample, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        std::string mode = std::string(attrs.get_string(AttrKey::Mode, "bilinear"));
        std::string padding_mode = std::string(attrs.get_string(AttrKey::PaddingMode, "zeros"));
        bool align_corners = attrs.get_bool(AttrKey::AlignCorners, false);
        return get_vulkan_backend()->dispatchGridSample(inputs[0], inputs[1], mode, padding_mode, align_corners);
    });
    table.register_single_output_kernel(OpId::AffineGrid, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        auto size_span = attrs.get_int_list(AttrKey::OutputSize);
        std::vector<int64_t> size(size_span.begin(), size_span.end());
        bool align_corners = attrs.get_bool(AttrKey::AlignCorners, false);
        return get_vulkan_backend()->dispatchAffineGrid(inputs[0], size, align_corners);
    });

    // audit Q.4: grid_sample / affine_grid backward (Vulkan compute shaders).
    table.register_kernel(OpId::GridSampleBackward,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            std::string mode = std::string(attrs.get_string(AttrKey::Mode, "bilinear"));
            std::string padding_mode = std::string(attrs.get_string(AttrKey::PaddingMode, "zeros"));
            bool align_corners = attrs.get_bool(AttrKey::AlignCorners, false);
            auto [gi, gg] = get_vulkan_backend()->dispatchGridSampleBackward(
                inputs[2], inputs[0], inputs[1], mode, padding_mode, align_corners);
            return {gi, gg};
        });
    table.register_single_output_kernel(OpId::AffineGridBackward,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            auto size_span = attrs.get_int_list(AttrKey::OutputSize);
            std::vector<int64_t> size(size_span.begin(), size_span.end());
            bool align_corners = attrs.get_bool(AttrKey::AlignCorners, false);
            return get_vulkan_backend()->dispatchAffineGridBackward(
                inputs[0], size, align_corners);
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
        int64_t normalized_size = vulkan_extract_normalized_size(attrs, inputs[0]);
        float eps = static_cast<float>(attrs.get_float(AttrKey::Eps, 1e-5));
        const Tensor* gamma = inputs.size() > 1 ? &inputs[1] : nullptr;
        const Tensor* beta = inputs.size() > 2 ? &inputs[2] : nullptr;
        auto [out, mean, rstd] = get_vulkan_backend()->dispatchLayerNorm(
            inputs[0], normalized_size, gamma, beta, eps);
        return std::vector<Tensor>{out, mean, rstd};
    });

    table.register_kernel(OpId::LayerNormBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t normalized_shape = vulkan_extract_normalized_size(attrs, inputs[0]);
        auto [grad_input, grad_weight, grad_bias] = get_vulkan_backend()->dispatchLayerNormBackward(
            inputs[0], inputs[1], inputs[2], inputs[3], &inputs[4], normalized_shape);
        return std::vector<Tensor>{grad_input, grad_weight, grad_bias};
    });

    table.register_kernel(OpId::GroupNorm, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t num_groups = attrs.get_int(AttrKey::NumGroups, attrs.get_int(AttrKey::Groups, 1));
        float eps = static_cast<float>(attrs.get_float(AttrKey::Eps, 1e-5));
        const Tensor* gamma = inputs.size() > 1 ? &inputs[1] : nullptr;
        const Tensor* beta = inputs.size() > 2 ? &inputs[2] : nullptr;
        return get_vulkan_backend()->dispatchGroupNorm(inputs[0], num_groups, gamma, beta, eps);
    });

    table.register_kernel(OpId::GroupNormBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t num_groups = attrs.get_int(AttrKey::NumGroups, attrs.get_int(AttrKey::Groups, 1));
        auto [grad_input, grad_weight, grad_bias] = get_vulkan_backend()->dispatchGroupNormBackward(
            inputs[0], inputs[1], inputs[2], inputs[3], &inputs[4], num_groups);
        return std::vector<Tensor>{grad_input, grad_weight, grad_bias};
    });

    table.register_kernel(OpId::RMSNorm, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        float eps = static_cast<float>(attrs.get_float(AttrKey::Eps, 1e-5));
        int64_t normalized_shape = inputs[0].shape().back();
        auto [output, rrms] = get_vulkan_backend()->dispatchRMSNorm(
            inputs[0], inputs[1], normalized_shape, eps);
        return std::vector<Tensor>{output, rrms};
    });

    table.register_kernel(OpId::FusedRMSNorm, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        // Per docs/internals/attention-contract.md, RMSNorm reads
        // AttrKey::NormalizedShape (multi-dim trailing-axes spec). Previously
        // this used .back() which silently truncated multi-dim normalisation
        // to the last dim (audit C20 Vulkan — singular-vs-plural drift in the
        // same registry, since LayerNorm at line 1225 reads the attr correctly).
        float eps = static_cast<float>(attrs.get_float(AttrKey::Eps, 1e-5));
        auto normalized_shape_list = attrs.get_int_list(AttrKey::NormalizedShape);
        int64_t normalized_size = 1;
        if (!normalized_shape_list.empty()) {
            for (auto d : normalized_shape_list) normalized_size *= d;
        } else {
            normalized_size = inputs[0].shape().back();
        }
        auto [output, rrms] = get_vulkan_backend()->dispatchRMSNorm(
            inputs[0], inputs[1], normalized_size, eps);
        return std::vector<Tensor>{output, rrms};
    });

    table.register_kernel(OpId::RMSNormBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        // The dispatcher (src/nn/layers/normalization.cpp:RMSNormBackward)
        // stores AttrKey::NormalizedShape as a string via std::to_string.
        // get_int would silently return 0 for a string-typed value, and
        // Vulkan's dispatchRMSNormBackward then computes
        // batch_size = input.numel() / 0 → SIGFPE. Parse via int_list and
        // fall back to the input's last dim if unset. (#55)
        auto ns_list = attrs.get_int_list(AttrKey::NormalizedShape);
        int64_t normalized_shape = ns_list.empty() ? inputs[0].shape().back()
                                                    : ns_list.front();
        if (normalized_shape <= 0) normalized_shape = inputs[0].shape().back();
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

    table.register_kernel(OpId::FusedSGDStep, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return get_vulkan_backend()->dispatchFusedSGDStep(inputs, attrs);
    });

    table.register_kernel(OpId::FusedAdamStep, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return get_vulkan_backend()->dispatchFusedAdamStep(inputs, attrs);
    });

    // ========================================================================
    // ScatterAdd
    // ========================================================================
    table.register_single_output_kernel(OpId::ScatterAdd, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t dim = attrs.get_int(AttrKey::Dim, 0);
        return get_vulkan_backend()->dispatchScatterAdd(inputs[0], dim, inputs[1], inputs[2]);
    });

    // ========================================================================
    // Instance Normalization
    // ========================================================================
    table.register_kernel(OpId::InstanceNorm, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        float eps = static_cast<float>(attrs.get_float(AttrKey::Eps, 1e-5));
        Tensor weight = inputs.size() > 1 ? inputs[1] : Tensor();
        Tensor bias = inputs.size() > 2 ? inputs[2] : Tensor();
        return get_vulkan_backend()->dispatchInstanceNorm(inputs[0], weight, bias, eps);
    });

    table.register_kernel(OpId::InstanceNormBackward, [](std::span<const Tensor> inputs, [[maybe_unused]] const OpAttributes& attrs) {
        // inputs: [grad_output, input, weight, mean, rstd]
        Tensor weight = inputs.size() > 2 ? inputs[2] : Tensor();
        auto [gi, gw, gb] = get_vulkan_backend()->dispatchInstanceNormBackward(
            inputs[0], inputs[1], inputs[3], inputs[4], weight);
        return std::vector<Tensor>{gi, gw, gb};
    });

    // ========================================================================
    // EmbeddingBag
    // ========================================================================
    table.register_single_output_kernel(OpId::EmbeddingBagForward,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            int64_t embedding_dim = attrs.get_int(AttrKey::EmbeddingDim, 0);
            std::string mode{attrs.get_string(AttrKey::Mode, "sum")};
            bool include_last_offset = attrs.get_bool(AttrKey::IncludeLastOffset, false);
            return get_vulkan_backend()->dispatchEmbeddingBag(
                inputs[0], inputs[1], embedding_dim, mode, include_last_offset);
        });

    table.register_kernel(OpId::EmbeddingBagBackward,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            // inputs: [grad_output, indices, offsets]
            int64_t num_embeddings = attrs.get_int(AttrKey::NumEmbeddings, 0);
            int64_t embedding_dim = attrs.get_int(AttrKey::EmbeddingDim, 0);
            std::string mode{attrs.get_string(AttrKey::Mode, "sum")};
            bool include_last_offset = attrs.get_bool(AttrKey::IncludeLastOffset, false);
            return {get_vulkan_backend()->dispatchEmbeddingBagBackward(
                inputs[0], inputs[1], inputs[2], num_embeddings, embedding_dim,
                mode, include_last_offset)};
        });

    // CTC Loss (audit Phase 3.7 + AA.11)
    // Vulkan SPIR-V compute shader port of src/backends/cuda/kernels/ctc.cu;
    // see src/backends/vulkan/kernels/ctc_forward.comp and
    // VulkanBackend::dispatchCTCLossForward in vulkan_ops_misc.cpp.
    table.register_kernel(OpId::CTCLossForward,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            if (inputs.size() != 4) {
                throw std::invalid_argument(
                    "CTCLossForward (Vulkan): expected 4 inputs "
                    "[log_probs, targets, input_lengths, target_lengths]");
            }
            int64_t blank = attrs.get_int(AttrKey::Blank, 0);
            bool zero_infinity = attrs.get_bool(AttrKey::ZeroInfinity, false);
            return get_vulkan_backend()->dispatchCTCLossForward(
                inputs[0], inputs[1], inputs[2], inputs[3], blank, zero_infinity);
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

    table.register_kernel(OpId::NMS, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        float iou_threshold = static_cast<float>(attrs.get_float(AttrKey::IouThreshold, 0.5));
        return std::vector<Tensor>{get_vulkan_backend()->dispatchNMS(inputs[0], inputs[1], iou_threshold)};
    });

    // ========================================================================
    // In-place Activation Operations
    // ========================================================================
    table.register_inplace_kernel(OpId::ReLUInplace, [](Tensor& target, std::span<const Tensor>, const OpAttributes&) -> Tensor& {
        auto vk = get_vulkan_backend();
        auto result = vk->dispatchActivation("relu", target, 0, 0.0f);
        auto bytes = target.numel() * dtype_size(target.dtype());
        if (bytes > 0) vk->copy(target.data_ptr(), result.data_ptr(), bytes, CopyKind::DeviceToDevice);
        return target;
    });

    table.register_inplace_kernel(OpId::SigmoidInplace, [](Tensor& target, std::span<const Tensor>, const OpAttributes&) -> Tensor& {
        auto vk = get_vulkan_backend();
        auto result = vk->dispatchActivation("sigmoid", target, 1, 0.0f);
        auto bytes = target.numel() * dtype_size(target.dtype());
        if (bytes > 0) vk->copy(target.data_ptr(), result.data_ptr(), bytes, CopyKind::DeviceToDevice);
        return target;
    });

    table.register_inplace_kernel(OpId::TanhInplace, [](Tensor& target, std::span<const Tensor>, const OpAttributes&) -> Tensor& {
        auto vk = get_vulkan_backend();
        auto result = vk->dispatchActivation("tanh", target, 2, 0.0f);
        auto bytes = target.numel() * dtype_size(target.dtype());
        if (bytes > 0) vk->copy(target.data_ptr(), result.data_ptr(), bytes, CopyKind::DeviceToDevice);
        return target;
    });

    table.register_inplace_kernel(OpId::LeakyReLUInplace, [](Tensor& target, std::span<const Tensor>, const OpAttributes& attrs) -> Tensor& {
        auto vk = get_vulkan_backend();
        auto result = vk->dispatchActivation("leaky_relu", target, 4, attrs.get_float(AttrKey::Alpha, 0.01));
        auto bytes = target.numel() * dtype_size(target.dtype());
        if (bytes > 0) vk->copy(target.data_ptr(), result.data_ptr(), bytes, CopyKind::DeviceToDevice);
        return target;
    });

    table.register_inplace_kernel(OpId::GeluInplace, [](Tensor& target, std::span<const Tensor>, const OpAttributes&) -> Tensor& {
        auto vk = get_vulkan_backend();
        auto result = vk->dispatchActivation("gelu", target, 3, 0.0f);
        auto bytes = target.numel() * dtype_size(target.dtype());
        if (bytes > 0) vk->copy(target.data_ptr(), result.data_ptr(), bytes, CopyKind::DeviceToDevice);
        return target;
    });

    // ========================================================================
    // Fused Operations
    // ========================================================================
    table.register_kernel(OpId::FusedLinearReLU, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        auto vk = get_vulkan_backend();
        bool has_bias = attrs.get_bool(AttrKey::HasBias, false);
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
        // Per docs/internals/attention-contract.md: FusedLayerNorm forward must
        // return (output, mean, rstd). Phase P0 / Fix 4 made
        // dispatchLayerNorm return that triple directly from the shader, so
        // we no longer need the host-level recomputation that used to live
        // here as a workaround.
        auto normalized_shape = attrs.get_int_list(AttrKey::NormalizedShape);
        int64_t normalized_size = 1;
        for (auto s : normalized_shape) normalized_size *= s;
        if (normalized_size <= 0) normalized_size = inputs[0].shape().back();
        float eps = static_cast<float>(attrs.get_float(AttrKey::Eps, 1e-5));

        Tensor output, mean, rstd;
        if (inputs[0].dtype() == DType::BFloat16) {
            Tensor input_f32 = inputs[0].to(DType::Float32);
            Tensor gamma_f32 = inputs[1].to(DType::Float32);
            Tensor beta_f32 = inputs[2].to(DType::Float32);
            auto [out_f32, m, r] = get_vulkan_backend()->dispatchLayerNorm(
                input_f32, normalized_size, &gamma_f32, &beta_f32, eps);
            output = out_f32.to(DType::BFloat16);
            mean = m;
            rstd = r;
        } else {
            auto [out, m, r] = get_vulkan_backend()->dispatchLayerNorm(
                inputs[0], normalized_size, &inputs[1], &inputs[2], eps);
            output = out;
            mean = m;
            rstd = r;
        }
        return std::vector<Tensor>{output, mean, rstd};
    });
    // The host-level mean/rstd recomputation below is no longer reachable —
    // the kernel returned above. Leaving the rest of the original block
    // commented out for one release in case any user code depended on a
    // specific intermediate tensor; safe to delete in a follow-up.
    /*
    {
        // (former host-level workaround follows)
        const Tensor& X = inputs[0];
        int64_t batch_size = X.numel() / normalized_size;
        Tensor X_f32 = (X.dtype() == DType::Float32) ? X : X.to(DType::Float32);
        Tensor X_2d = tenzor::reshape(X_f32, std::vector<int64_t>{batch_size, normalized_size});
        NewOpAttributes mean_attrs;
        mean_attrs.set(AttrKey::Dim, static_cast<int64_t>(-1));
        mean_attrs.set(AttrKey::Keepdim, false);
        std::vector<Tensor> mean_in = {X_2d};
        Tensor mean = tenzor::dispatch(OpId::Mean, mean_in, mean_attrs)[0];
        Tensor mean_unsq = tenzor::reshape(mean, std::vector<int64_t>{batch_size, 1});
        Tensor diff = tenzor::sub(X_2d, mean_unsq);
        Tensor diff_sq = tenzor::mul(diff, diff);
        std::vector<Tensor> var_in = {diff_sq};
        Tensor var = tenzor::dispatch(OpId::Mean, var_in, mean_attrs)[0];
        Tensor eps_t = tenzor::full(
            std::vector<int64_t>(var.shape().begin(), var.shape().end()),
            static_cast<double>(eps), var.dtype(), var.device());
        Tensor var_eps = tenzor::add(var, eps_t);
        Tensor rstd = tenzor::div(
            tenzor::full(std::vector<int64_t>(var_eps.shape().begin(), var_eps.shape().end()),
                         1.0, var_eps.dtype(), var_eps.device()),
            tenzor::sqrt(var_eps));
        return std::vector<Tensor>{output, mean, rstd};
    }
    */  // end of legacy host-level recompute block

    // ========================================================================
    // Interpolation
    // ========================================================================
    table.register_kernel(OpId::Interpolate, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchInterpolate(inputs[0], attrs)};
    });
    // D3-followup Vulkan: bilinear backward via the new
    // `interpolate_bilinear_backward.comp` shader + dispatchInterpolateBackward
    // host method (both added this audit pass). Float32 only — F64 atomicAdd
    // requires the separate `GL_EXT_shader_atomic_float2` extension.
    table.register_kernel(OpId::InterpolateBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        auto input_size = attrs.get_int_list(AttrKey::InputShape);
        std::string mode = std::string(attrs.get_string(AttrKey::Mode, "bilinear"));
        bool align_corners = attrs.get_bool(AttrKey::AlignCorners, false);
        return std::vector<Tensor>{
            get_vulkan_backend()->dispatchInterpolateBackward(
                inputs[0], input_size, mode, align_corners)};
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
    // Vulkan LSTMForward goes through the fused per-timestep dispatcher in
    // vulkan_ops_rnn.cpp + the lstm_cell.comp shader. The shader now uses
    // an exp()-based sigmoid/tanh formulation (precise float diff ≈ 2 ULP
    // vs std::tanh on tested NVIDIA / RADV stacks) instead of GLSL's
    // driver-supplied tanh approximation. Combined with the matmul shader's
    // FP32 accumulation, this keeps multi-step LSTM drift under ~1e-3 for
    // typical normal-distributed inputs and roughly an order of magnitude
    // larger for inputs that drive the sigmoid/tanh midrange. See
    // NNRNNParity.LSTM_Dropout_Eval for the realistic tolerance.
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

    table.register_kernel(OpId::Median, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t dim = attrs.get_int(AttrKey::Dim, -1);
        bool keepdim = attrs.get_bool(AttrKey::Keepdim, false);
        return get_vulkan_backend()->dispatchMedian(inputs[0], dim, keepdim);
    });

    table.register_kernel(OpId::Mode, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t dim = attrs.get_int(AttrKey::Dim, -1);
        bool keepdim = attrs.get_bool(AttrKey::Keepdim, false);
        return get_vulkan_backend()->dispatchMode(inputs[0], dim, keepdim);
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
        // F.11: per-axis read with scalar fallback; Vulkan depthwise shader
        // takes scalar stride/padding/dilation — fail loudly on asymmetric
        // rather than silently collapse StrideW/PaddingW/DilationW.
        TENZOR_READ_CONV2D_ATTRS();
        (void)groups;
        if (stride[0] != stride[1] || padding[0] != padding[1] || dilation[0] != dilation[1]) {
            throw std::invalid_argument(
                "DepthwiseConv2d (Vulkan): backend shader only supports symmetric "
                "stride/padding/dilation; got stride=" + std::to_string(stride[0]) + "x" + std::to_string(stride[1]) +
                ", padding=" + std::to_string(padding[0]) + "x" + std::to_string(padding[1]) +
                ", dilation=" + std::to_string(dilation[0]) + "x" + std::to_string(dilation[1]) + ".");
        }
        const Tensor* bias = inputs.size() > 2 ? &inputs[2] : nullptr;
        return get_vulkan_backend()->dispatchDepthwiseConv2d(inputs[0], inputs[1], bias, stride[0], padding[0], dilation[0]);
    });

    // DeformableConv2d (DCNv2) — inputs: {input, offset, weight, bias, mask}
    // F.11: per-axis read with scalar fallback via attr_macros helpers.
    // Previously read raw StrideH/W etc. without scalar fallback.
    table.register_kernel(OpId::DeformableConv2dForward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        TENZOR_READ_CONV2D_ATTRS();
        int64_t offset_groups = attrs.get_int(AttrKey::OffsetGroups, 1);
        bool use_mask = attrs.get_int(AttrKey::UseMask, 0) != 0;
        return std::vector<Tensor>{get_vulkan_backend()->dispatchDeformableConv2dForward(
            inputs[0], inputs[1], inputs[2], inputs[3], inputs[4],
            stride[0], stride[1], padding[0], padding[1], dilation[0], dilation[1],
            groups, offset_groups, use_mask)};
    });

    // DeformableConv2dBackwardInput — inputs: {grad_output, input, offset, weight, mask}
    // Returns: {grad_input, grad_offset[, grad_mask]}
    table.register_kernel(OpId::DeformableConv2dBackwardInput, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        TENZOR_READ_CONV2D_ATTRS();
        int64_t offset_groups = attrs.get_int(AttrKey::OffsetGroups, 1);
        bool use_mask = attrs.get_int(AttrKey::UseMask, 0) != 0;
        return get_vulkan_backend()->dispatchDeformableConv2dBackwardInput(
            inputs[0], inputs[1], inputs[2], inputs[3], inputs[4],
            stride[0], stride[1], padding[0], padding[1], dilation[0], dilation[1],
            groups, offset_groups, use_mask);
    });

    // DeformableConv2dBackwardWeight — inputs: {grad_output, input, offset, mask}
    table.register_kernel(OpId::DeformableConv2dBackwardWeight, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        TENZOR_READ_CONV2D_ATTRS();
        int64_t offset_groups = attrs.get_int(AttrKey::OffsetGroups, 1);
        bool use_mask = attrs.get_int(AttrKey::UseMask, 0) != 0;
        auto weight_shape = attrs.get_int_list(AttrKey::WeightShape);
        return std::vector<Tensor>{get_vulkan_backend()->dispatchDeformableConv2dBackwardWeight(
            inputs[0], inputs[1], inputs[2], inputs[3],
            stride[0], stride[1], padding[0], padding[1], dilation[0], dilation[1],
            groups, offset_groups, use_mask, weight_shape)};
    });

    // DeformableConv2dBackwardBias — reuse regular conv2d bias backward (channel-wise sum)
    table.register_kernel(OpId::DeformableConv2dBackwardBias, [](std::span<const Tensor> inputs, const OpAttributes&) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchConv2dBackwardBias(inputs[0])};
    });

    table.register_single_output_kernel(OpId::AdaptiveMaxPool2dBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        auto input_shape = attrs.get_int_list(AttrKey::InputShape);
        return get_vulkan_backend()->dispatchAdaptiveMaxPool2dBackward(inputs[0], inputs[1], input_shape);
    });

    table.register_single_output_kernel(OpId::CumSum, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t dim = attrs.get_int(AttrKey::Dim, 0);
        return get_vulkan_backend()->dispatchCumSum(inputs[0], dim);
    });

    table.register_single_output_kernel(OpId::CumProd, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t dim = attrs.get_int(AttrKey::Dim, 0);
        return get_vulkan_backend()->dispatchCumProd(inputs[0], dim);
    });

    // ========================================================================
    // Extended Math Operations (unary)
    // ========================================================================
    table.register_kernel(OpId::Log2, [](std::span<const Tensor> inputs, const OpAttributes&) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchUnaryOp("log2", inputs[0])};
    });

    table.register_kernel(OpId::Log10, [](std::span<const Tensor> inputs, const OpAttributes&) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchUnaryOp("log10", inputs[0])};
    });

    table.register_kernel(OpId::Log1p, [](std::span<const Tensor> inputs, const OpAttributes&) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchUnaryOp("log1p", inputs[0])};
    });

    table.register_kernel(OpId::Exp2, [](std::span<const Tensor> inputs, const OpAttributes&) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchUnaryOp("exp2", inputs[0])};
    });

    table.register_kernel(OpId::Expm1, [](std::span<const Tensor> inputs, const OpAttributes&) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchUnaryOp("expm1", inputs[0])};
    });

    table.register_kernel(OpId::Erf, [](std::span<const Tensor> inputs, const OpAttributes&) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchUnaryOp("erf", inputs[0])};
    });

    table.register_kernel(OpId::Erfc, [](std::span<const Tensor> inputs, const OpAttributes&) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchUnaryOp("erfc", inputs[0])};
    });

    // ========================================================================
    // New Unary Math Operations
    // ========================================================================
    table.register_kernel(OpId::Rsqrt, [](std::span<const Tensor> inputs, const OpAttributes&) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchUnaryOp("rsqrt", inputs[0])};
    });

    table.register_kernel(OpId::Square, [](std::span<const Tensor> inputs, const OpAttributes&) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchUnaryOp("square", inputs[0])};
    });

    table.register_kernel(OpId::Asinh, [](std::span<const Tensor> inputs, const OpAttributes&) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchUnaryOp("asinh", inputs[0])};
    });

    table.register_kernel(OpId::Acosh, [](std::span<const Tensor> inputs, const OpAttributes&) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchUnaryOp("acosh", inputs[0])};
    });

    table.register_kernel(OpId::Atanh, [](std::span<const Tensor> inputs, const OpAttributes&) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchUnaryOp("atanh", inputs[0])};
    });

    // ========================================================================
    // Special Math Functions — native Vulkan compute shaders
    // ========================================================================
    // Opcodes mirror special_math_unary.comp:
    //   0=gamma, 1=lgamma, 2=digamma, 3=bessel_j0, 4=bessel_j1,
    //   5=bessel_y0, 6=bessel_y1, 7=bessel_i0, 8=bessel_i1, 9=erfinv,
    //   10=sinc, 11=polygamma
    // KernelFn is a plain function pointer (no captures), so each op gets its
    // own non-capturing lambda hard-coding the opcode.
#define VK_REGISTER_UNARY_SPECIAL(OP_ID, OPCODE) \
    table.register_kernel(OpId::OP_ID, [](std::span<const Tensor> inputs, const OpAttributes&) { \
        return std::vector<Tensor>{ \
            get_vulkan_backend()->dispatchSpecialMathUnary(inputs[0], (OPCODE))}; \
    })

    VK_REGISTER_UNARY_SPECIAL(Gamma,    0);
    VK_REGISTER_UNARY_SPECIAL(Lgamma,   1);
    VK_REGISTER_UNARY_SPECIAL(Digamma,  2);
    VK_REGISTER_UNARY_SPECIAL(BesselJ0, 3);
    VK_REGISTER_UNARY_SPECIAL(BesselJ1, 4);
    VK_REGISTER_UNARY_SPECIAL(BesselY0, 5);
    VK_REGISTER_UNARY_SPECIAL(BesselY1, 6);
    VK_REGISTER_UNARY_SPECIAL(BesselI0, 7);
    VK_REGISTER_UNARY_SPECIAL(BesselI1, 8);
    VK_REGISTER_UNARY_SPECIAL(ErfInv,   9);
    VK_REGISTER_UNARY_SPECIAL(Sinc,     10);
    VK_REGISTER_UNARY_SPECIAL(I0e,      12);
    VK_REGISTER_UNARY_SPECIAL(I1e,      13);
    VK_REGISTER_UNARY_SPECIAL(Entr,     14);
    VK_REGISTER_UNARY_SPECIAL(SphericalBesselJ0, 15);
#undef VK_REGISTER_UNARY_SPECIAL

    table.register_kernel(OpId::Polygamma, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t n = static_cast<int64_t>(attrs.get_float(AttrKey::Order, 0.0));
        return std::vector<Tensor>{
            get_vulkan_backend()->dispatchSpecialMathUnary(inputs[0], 11, static_cast<int32_t>(n))};
    });

    // Binary special-math: 0=beta, 1=zeta, 2=logaddexp, 3=logaddexp2, 4=xlogy
    table.register_kernel(OpId::Beta, [](std::span<const Tensor> inputs, const OpAttributes&) {
        return std::vector<Tensor>{
            get_vulkan_backend()->dispatchSpecialMathBinary(inputs[0], inputs[1], 0)};
    });
    table.register_kernel(OpId::Zeta, [](std::span<const Tensor> inputs, const OpAttributes&) {
        return std::vector<Tensor>{
            get_vulkan_backend()->dispatchSpecialMathBinary(inputs[0], inputs[1], 1)};
    });
    table.register_kernel(OpId::LogAddExp, [](std::span<const Tensor> inputs, const OpAttributes&) {
        return std::vector<Tensor>{
            get_vulkan_backend()->dispatchSpecialMathBinary(inputs[0], inputs[1], 2)};
    });
    table.register_kernel(OpId::LogAddExp2, [](std::span<const Tensor> inputs, const OpAttributes&) {
        return std::vector<Tensor>{
            get_vulkan_backend()->dispatchSpecialMathBinary(inputs[0], inputs[1], 3)};
    });
    table.register_kernel(OpId::XLogY, [](std::span<const Tensor> inputs, const OpAttributes&) {
        return std::vector<Tensor>{
            get_vulkan_backend()->dispatchSpecialMathBinary(inputs[0], inputs[1], 4)};
    });

    // Ternary special-math: betainc
    table.register_kernel(OpId::BetaInc, [](std::span<const Tensor> inputs, const OpAttributes&) {
        return std::vector<Tensor>{
            get_vulkan_backend()->dispatchSpecialMathTernary(inputs[0], inputs[1], inputs[2])};
    });

    // ========================================================================
    // Ndtr / LogNdtr / Multigammaln / LinalgVectorNorm / LinalgMatrixNorm / LinalgVecdot
    // ========================================================================

    // Ndtr: Phi(x) = 0.5 * erfc(-x * M_SQRT1_2) -- composite from Erfc
    table.register_kernel(OpId::Ndtr, [](std::span<const Tensor> inputs, const OpAttributes&) {
        auto* vk = get_vulkan_backend();
        Tensor neg_x = vk->dispatchUnaryOp("neg", inputs[0]);
        Tensor sqrt1_2 = vk->dispatchFull({1}, 0.7071067811865476f, inputs[0].dtype());
        Tensor scaled = vk->dispatchBinaryOp("mul", neg_x, sqrt1_2);
        Tensor erfc_val = vk->dispatchUnaryOp("erfc", scaled);
        Tensor half_val = vk->dispatchFull({1}, 0.5f, inputs[0].dtype());
        return std::vector<Tensor>{vk->dispatchBinaryOp("mul", erfc_val, half_val)};
    });

    // LogNdtr: log(Phi(x)) -- composite from Ndtr + log
    table.register_kernel(OpId::LogNdtr, [](std::span<const Tensor> inputs, const OpAttributes&) {
        auto* vk = get_vulkan_backend();
        Tensor neg_x = vk->dispatchUnaryOp("neg", inputs[0]);
        Tensor sqrt1_2 = vk->dispatchFull({1}, 0.7071067811865476f, inputs[0].dtype());
        Tensor scaled = vk->dispatchBinaryOp("mul", neg_x, sqrt1_2);
        Tensor erfc_val = vk->dispatchUnaryOp("erfc", scaled);
        Tensor half_val = vk->dispatchFull({1}, 0.5f, inputs[0].dtype());
        Tensor ndtr_val = vk->dispatchBinaryOp("mul", erfc_val, half_val);
        return std::vector<Tensor>{vk->dispatchUnaryOp("log", ndtr_val)};
    });

    // Multigammaln: sum_{j=0}^{d-1} lgamma(x - j/2) + d*(d-1)/4 * log(pi) -- composite
    table.register_kernel(OpId::Multigammaln, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        auto* vk = get_vulkan_backend();
        int64_t d = attrs.get_int(AttrKey::Dim, 1);
        float log_pi_coeff = static_cast<float>(d) * static_cast<float>(d - 1) / 4.0f
                           * 1.1447298858494002f;
        Tensor result = vk->dispatchSpecialMathUnary(inputs[0], 1);
        for (int64_t j = 1; j < d; ++j) {
            Tensor offset = vk->dispatchFull({1}, static_cast<float>(j) * 0.5f, inputs[0].dtype());
            Tensor shifted = vk->dispatchBinaryOp("sub", inputs[0], offset);
            Tensor lg = vk->dispatchSpecialMathUnary(shifted, 1);
            result = vk->dispatchBinaryOp("add", result, lg);
        }
        Tensor coeff = vk->dispatchFull({1}, log_pi_coeff, inputs[0].dtype());
        result = vk->dispatchBinaryOp("add", result, coeff);
        return std::vector<Tensor>{result};
    });

    // LinalgVectorNorm: delegates to existing Norm dispatch
    table.register_kernel(OpId::LinalgVectorNorm, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        float p = static_cast<float>(attrs.get_float(AttrKey::P, 2.0));
        int64_t dim = attrs.get_int(AttrKey::Dim, INT64_MIN);
        bool keepdim = attrs.get_bool(AttrKey::Keepdim, false);
        return std::vector<Tensor>{get_vulkan_backend()->dispatchNorm(inputs[0], p, dim, keepdim)};
    });

    // LinalgMatrixNorm: Frobenius (ord=0), nuclear (ord=1), spectral (ord=2)
    table.register_kernel(OpId::LinalgMatrixNorm, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t ord = static_cast<int64_t>(attrs.get_float(AttrKey::Order, 0.0));
        auto* vk = get_vulkan_backend();
        if (ord == 0) {
            return std::vector<Tensor>{vk->dispatchNorm(inputs[0], 2.0f, INT64_MIN, false)};
        }
        auto svd_result = vk->dispatchLinalgSVD(inputs[0], false);
        Tensor S = svd_result[1];
        if (ord == 1) {
            return std::vector<Tensor>{vk->dispatchReduction("sum", S, INT64_MIN, false)};
        }
        return std::vector<Tensor>{vk->dispatchReduction("max", S, INT64_MIN, false)};
    });

    // LinalgVecdot: sum(a * b, dim)
    table.register_kernel(OpId::LinalgVecdot, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t dim = attrs.get_int(AttrKey::Dim, -1);
        auto* vk = get_vulkan_backend();
        Tensor product = vk->dispatchBinaryOp("mul", inputs[0], inputs[1]);
        return std::vector<Tensor>{vk->dispatchReduction("sum", product, dim, false)};
    });

    // CosineSimilarity: sum(a*b, dim) / (norm(a, dim) * norm(b, dim) + eps)
    table.register_single_output_kernel(OpId::CosineSimilarity, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t dim = attrs.get_int(AttrKey::Dim, 1);
        double eps = attrs.get_float(AttrKey::Eps, 1e-8);
        auto* vk = get_vulkan_backend();
        // dot product along dim
        Tensor ab = vk->dispatchBinaryOp("mul", inputs[0], inputs[1]);
        Tensor dot = vk->dispatchReduction("sum", ab, dim, false);
        // norms along dim
        Tensor norm_a = vk->dispatchNorm(inputs[0], 2.0f, dim, false);
        Tensor norm_b = vk->dispatchNorm(inputs[1], 2.0f, dim, false);
        // norm_a * norm_b + eps
        Tensor norms = vk->dispatchBinaryOp("mul", norm_a, norm_b);
        Tensor eps_tensor = vk->dispatchFull({1}, static_cast<float>(eps), norms.dtype());
        Tensor denom = vk->dispatchBinaryOp("add", norms, eps_tensor);
        return vk->dispatchBinaryOp("div", dot, denom);
    });

    // Renorm: scale slices along dim so p-norm <= maxnorm
    table.register_single_output_kernel(OpId::Renorm, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        double p = attrs.get_float(AttrKey::P, 2.0);
        int64_t dim = attrs.get_int(AttrKey::Dim, 0);
        double maxnorm = attrs.get_float(AttrKey::MaxNorm, 1.0);
        auto* vk = get_vulkan_backend();

        // Renorm's "dim" names the axis whose sub-tensors are renormalized —
        // so the p-norm is computed across every OTHER axis. The previous
        // code delegated to dispatchNorm(..., dim, keepdim) which reduces
        // the given dim instead (PyTorch-style norm semantics), giving
        // garbage norms and NaN gradients through the backward pass.
        //
        // Compute ||x||_p per slice with the right reduction axes:
        //   pow_p = |x|^p
        //   norm  = (sum over all axes != dim)^(1/p)
        // Keep all reduced axes as size-1 so the final scale broadcasts back.
        const Tensor& x = inputs[0];
        int64_t ndim = static_cast<int64_t>(x.shape().size());
        int64_t norm_dim = (dim < 0) ? ndim + dim : dim;

        Tensor abs_x = vk->dispatchUnaryOp("abs", x);
        Tensor pow_p = (p == 2.0)
            ? vk->dispatchBinaryOp("mul", abs_x, abs_x)
            : vk->dispatchUnaryOpWithParam("pow", abs_x, static_cast<float>(p));
        Tensor sum_p = pow_p;
        for (int64_t d = 0; d < ndim; ++d) {
            if (d == norm_dim) continue;
            sum_p = vk->dispatchReduction("sum", sum_p, d, /*keepdim=*/true);
        }
        // (1/p)-root. For p=2.0 use sqrt (lossless fast path).
        Tensor norm;
        if (p == 2.0) {
            norm = vk->dispatchUnaryOp("sqrt", sum_p);
        } else {
            norm = vk->dispatchUnaryOpWithParam("pow", sum_p, static_cast<float>(1.0 / p));
        }

        // scale = maxnorm / max(norm, maxnorm). Binary-op name is "maximum"
        // ("max" is a unary reduction and throws "Unknown binary operation").
        Tensor maxnorm_tensor = vk->dispatchFull({1}, static_cast<float>(maxnorm), norm.dtype());
        Tensor clamped_norm = vk->dispatchBinaryOp("maximum", norm, maxnorm_tensor);
        Tensor scale = vk->dispatchBinaryOp("div", maxnorm_tensor, clamped_norm);
        return vk->dispatchBinaryOp("mul", x, scale);
    });

    // ========================================================================
    // Binary Math Operations
    // ========================================================================
    table.register_kernel(OpId::Atan2, [](std::span<const Tensor> inputs, const OpAttributes&) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchBinaryOp("atan2", inputs[0], inputs[1])};
    });

    table.register_kernel(OpId::Fmod, [](std::span<const Tensor> inputs, const OpAttributes&) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchBinaryOp("fmod", inputs[0], inputs[1])};
    });

    table.register_kernel(OpId::Remainder, [](std::span<const Tensor> inputs, const OpAttributes&) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchBinaryOp("remainder", inputs[0], inputs[1])};
    });

    table.register_kernel(OpId::Minimum, [](std::span<const Tensor> inputs, const OpAttributes&) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchBinaryOp("minimum", inputs[0], inputs[1])};
    });

    table.register_kernel(OpId::Maximum, [](std::span<const Tensor> inputs, const OpAttributes&) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchBinaryOp("maximum", inputs[0], inputs[1])};
    });

    // ========================================================================
    // New Binary Math Operations
    // ========================================================================
    table.register_kernel(OpId::Hypot, [](std::span<const Tensor> inputs, const OpAttributes&) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchBinaryOp("hypot", inputs[0], inputs[1])};
    });

    table.register_kernel(OpId::Copysign, [](std::span<const Tensor> inputs, const OpAttributes&) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchBinaryOp("copysign", inputs[0], inputs[1])};
    });

    table.register_kernel(OpId::Nextafter, [](std::span<const Tensor> inputs, const OpAttributes&) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchBinaryOp("nextafter", inputs[0], inputs[1])};
    });

    table.register_kernel(OpId::Gcd, [](std::span<const Tensor> inputs, const OpAttributes&) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchBinaryOp("gcd", inputs[0], inputs[1])};
    });

    table.register_kernel(OpId::Lcm, [](std::span<const Tensor> inputs, const OpAttributes&) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchBinaryOp("lcm", inputs[0], inputs[1])};
    });

    table.register_kernel(OpId::Igamma, [](std::span<const Tensor> inputs, const OpAttributes&) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchBinaryOp("igamma", inputs[0], inputs[1])};
    });

    table.register_kernel(OpId::Igammac, [](std::span<const Tensor> inputs, const OpAttributes&) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchBinaryOp("igammac", inputs[0], inputs[1])};
    });

    // ========================================================================
    // New Ternary Math Operations
    // ========================================================================
    table.register_kernel(OpId::Addcmul, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        float value = attrs.get_float(AttrKey::Alpha, 1.0);
        return std::vector<Tensor>{get_vulkan_backend()->dispatchAddcmul(inputs[0], inputs[1], inputs[2], value)};
    });

    table.register_kernel(OpId::Addcdiv, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        float value = attrs.get_float(AttrKey::Alpha, 1.0);
        return std::vector<Tensor>{get_vulkan_backend()->dispatchAddcdiv(inputs[0], inputs[1], inputs[2], value)};
    });

    // ========================================================================
    // Bool Predicate Operations
    // ========================================================================
    table.register_kernel(OpId::IsNan, [](std::span<const Tensor> inputs, const OpAttributes&) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchBoolPredicateOp("isnan", inputs[0])};
    });

    table.register_kernel(OpId::IsInf, [](std::span<const Tensor> inputs, const OpAttributes&) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchBoolPredicateOp("isinf", inputs[0])};
    });

    table.register_kernel(OpId::IsFinite, [](std::span<const Tensor> inputs, const OpAttributes&) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchBoolPredicateOp("isfinite", inputs[0])};
    });

    // ========================================================================
    // Logical Operations
    // ========================================================================
    table.register_kernel(OpId::LogicalAnd, [](std::span<const Tensor> inputs, const OpAttributes&) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchLogicalOp("logical_and", inputs[0], inputs[1])};
    });

    table.register_kernel(OpId::LogicalOr, [](std::span<const Tensor> inputs, const OpAttributes&) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchLogicalOp("logical_or", inputs[0], inputs[1])};
    });

    table.register_kernel(OpId::LogicalNot, [](std::span<const Tensor> inputs, const OpAttributes&) {
        // LogicalNot is unary - pass same tensor as both a and b (b is ignored by shader)
        return std::vector<Tensor>{get_vulkan_backend()->dispatchLogicalOp("logical_not", inputs[0], inputs[0])};
    });

    table.register_kernel(OpId::LogicalXor, [](std::span<const Tensor> inputs, const OpAttributes&) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchLogicalOp("logical_xor", inputs[0], inputs[1])};
    });

    // ========================================================================
    // Lerp (Linear Interpolation)
    // ========================================================================
    table.register_kernel(OpId::Lerp, [](std::span<const Tensor> inputs, const OpAttributes&) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchLerp(inputs[0], inputs[1], inputs[2])};
    });

    // ========================================================================
    // Cross Product
    // ========================================================================
    table.register_single_output_kernel(OpId::Cross, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t dim = attrs.get_int(AttrKey::Dim, -1);
        return get_vulkan_backend()->dispatchCross(inputs[0], inputs[1], dim);
    });

    // ========================================================================
    // Boolean Reduction Operations
    // ========================================================================
    table.register_kernel(OpId::Any, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t dim = attrs.get_int(AttrKey::Dim, -1);
        bool keepdim = attrs.get_bool(AttrKey::Keepdim, false);
        return std::vector<Tensor>{get_vulkan_backend()->dispatchBooleanReduction("any", inputs[0], dim, keepdim)};
    });

    table.register_kernel(OpId::All, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t dim = attrs.get_int(AttrKey::Dim, -1);
        bool keepdim = attrs.get_bool(AttrKey::Keepdim, false);
        return std::vector<Tensor>{get_vulkan_backend()->dispatchBooleanReduction("all", inputs[0], dim, keepdim)};
    });

    // ========================================================================
    // Triangular Matrix Operations
    // ========================================================================
    table.register_single_output_kernel(OpId::Triu, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t diagonal = attrs.get_int(AttrKey::Diagonal, 0);
        return get_vulkan_backend()->dispatchTriuTril("triu", inputs[0], diagonal);
    });

    table.register_single_output_kernel(OpId::Tril, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t diagonal = attrs.get_int(AttrKey::Diagonal, 0);
        return get_vulkan_backend()->dispatchTriuTril("tril", inputs[0], diagonal);
    });

    // ========================================================================
    // Diagonal Operations
    // ========================================================================
    table.register_single_output_kernel(OpId::Diag, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t diagonal = attrs.get_int(AttrKey::Diagonal, 0);
        return get_vulkan_backend()->dispatchDiag(inputs[0], diagonal);
    });

    // ========================================================================
    // Trace Operation
    // ========================================================================
    table.register_single_output_kernel(OpId::Trace, [](std::span<const Tensor> inputs, const OpAttributes&) -> Tensor {
        return get_vulkan_backend()->dispatchTrace(inputs[0]);
    });

    // ========================================================================
    // Flip Operation
    // ========================================================================
    table.register_single_output_kernel(OpId::Flip, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        // Phase 27-followup #40 fix: tenzor::flip sets AttrKey::Dims
        // (plural comma-separated string), not AttrKey::Dim (int). Reading
        // the wrong key defaulted to 0 and flipped axis 0 regardless of the
        // requested dim. Parse the dim list and flip each in turn.
        auto dims_sv = attrs.get_string(AttrKey::Dims, "0");
        std::string dims_str(dims_sv);
        Tensor result = inputs[0];
        std::istringstream ss(dims_str);
        std::string token;
        while (std::getline(ss, token, ',')) {
            if (!token.empty()) {
                int64_t dim = std::stoll(token);
                result = get_vulkan_backend()->dispatchFlip(result, dim);
            }
        }
        return result;
    });

    // ========================================================================
    // Fused Conv2d + Activation Operations (composition pattern)
    // ========================================================================
    table.register_single_output_kernel(OpId::FusedConv2dReLU, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        auto vk = get_vulkan_backend();
        const Tensor* bias = inputs.size() > 2 ? &inputs[2] : nullptr;
        auto conv_result = vk->dispatchConv2dForward(inputs[0], inputs[1], bias, attrs);
        return vk->dispatchActivation("relu", conv_result, 0, 0.0f);
    });

    table.register_single_output_kernel(OpId::FusedConv2dSigmoid, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        auto vk = get_vulkan_backend();
        const Tensor* bias = inputs.size() > 2 ? &inputs[2] : nullptr;
        auto conv_result = vk->dispatchConv2dForward(inputs[0], inputs[1], bias, attrs);
        return vk->dispatchActivation("sigmoid", conv_result, 1, 0.0f);
    });

    table.register_single_output_kernel(OpId::FusedConv2dTanh, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        auto vk = get_vulkan_backend();
        const Tensor* bias = inputs.size() > 2 ? &inputs[2] : nullptr;
        auto conv_result = vk->dispatchConv2dForward(inputs[0], inputs[1], bias, attrs);
        return vk->dispatchActivation("tanh", conv_result, 2, 0.0f);
    });

    table.register_single_output_kernel(OpId::FusedConv2dSwish, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        auto vk = get_vulkan_backend();
        const Tensor* bias = inputs.size() > 2 ? &inputs[2] : nullptr;
        auto conv_result = vk->dispatchConv2dForward(inputs[0], inputs[1], bias, attrs);
        // Swish = x * sigmoid(x)
        auto sigmoid_result = vk->dispatchActivation("sigmoid", conv_result, 1, 0.0f);
        return vk->dispatchBinaryOp("mul", conv_result, sigmoid_result);
    });

    table.register_single_output_kernel(OpId::FusedConv2dBnReLU, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        // inputs: [input, weight, conv_bias, bn_gamma, bn_beta, bn_running_mean, bn_running_var]
        auto vk = get_vulkan_backend();
        const Tensor* conv_bias = inputs.size() > 2 ? &inputs[2] : nullptr;
        auto conv_result = vk->dispatchConv2dForward(inputs[0], inputs[1], conv_bias, attrs);
        float bn_eps = static_cast<float>(attrs.get_float(AttrKey::Eps, 1e-5));
        auto bn_result = vk->dispatchBatchNorm2dForward(conv_result, inputs[5], inputs[6],
                                                         &inputs[3], &inputs[4], bn_eps);
        return vk->dispatchActivation("relu", bn_result, 0, 0.0f);
    });

    // ========================================================================
    // Fused Softmax + Cross Entropy
    // ========================================================================
    table.register_kernel(OpId::FusedSoftmaxCrossEntropy, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        // Compose: softmax -> cross_entropy
        auto vk = get_vulkan_backend();
        int64_t dim = -1; // Softmax over last dim (class dim)
        auto log_probs = vk->dispatchLogSoftmax(inputs[0], dim);
        // Map reduction string to integer: 0=none, 1=mean, 2=sum
        std::string reduction_str = std::string(attrs.get_string(AttrKey::Reduction, "mean"));
        int64_t reduction = 1; // mean
        if (reduction_str == "none") reduction = 0;
        else if (reduction_str == "sum") reduction = 2;
        auto loss = vk->dispatchCrossEntropy(log_probs, inputs[1], reduction);
        return std::vector<Tensor>{loss};
    });

    // ========================================================================
    // Fused Adam-Atan2 Optimizer Step (compose from Adam step with atan2 update)
    // ========================================================================
    table.register_kernel(OpId::FusedAdamAtan2Step, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return get_vulkan_backend()->dispatchFusedAdamAtan2Step(inputs, attrs);
    });

    // ========================================================================
    // Fused Attention (Q, K, V -> output)
    // ========================================================================
    table.register_kernel(OpId::FusedAttention, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        // Compose: scaled dot-product attention via dispatch methods
        // inputs: [query, key, value]
        auto vk = get_vulkan_backend();
        float scale = static_cast<float>(attrs.get_float(AttrKey::Scale, 1.0));

        // scores = Q @ K^T
        auto key_t = vk->dispatchTranspose(inputs[1], -2, -1);
        auto scores = vk->dispatchBmm(inputs[0], key_t);

        // scores *= scale
        if (scale != 1.0f) {
            auto scale_tensor = vk->dispatchFull({1}, scale, scores.dtype());
            scores = vk->dispatchBinaryOp("mul", scores, scale_tensor);
        }

        // attn_weights = softmax(scores, dim=-1)
        auto attn_weights = vk->dispatchSoftmax(scores, -1);

        // output = attn_weights @ V
        auto output = vk->dispatchBmm(attn_weights, inputs[2]);
        return std::vector<Tensor>{output};
    });

    // ========================================================================
    // BatchNorm2d Fused Training
    // ========================================================================
    table.register_kernel(OpId::BatchNorm2dFusedTraining, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        // inputs: [input, running_mean, running_var, gamma, beta]
        auto vk = get_vulkan_backend();
        float epsilon = static_cast<float>(attrs.get_float(AttrKey::Eps, 1e-5));
        float momentum = static_cast<float>(attrs.get_float(AttrKey::Momentum, 0.1));

        // Compute batch mean and variance
        auto [batch_mean, batch_var] = vk->dispatchBatchNorm2dMeanVar(inputs[0]);

        // Normalize using batch statistics
        auto output = vk->dispatchBatchNorm2dForward(inputs[0], batch_mean, batch_var,
                                                      &inputs[3], &inputs[4], epsilon);

        // Update running stats: running = (1 - momentum) * running + momentum * batch
        OpAttributes update_attrs;
        update_attrs.set(AttrKey::Momentum, static_cast<double>(momentum));
        std::vector<Tensor> update_inputs = {inputs[1], inputs[2], batch_mean, batch_var};
        auto updated = vk->dispatchBatchNorm2dUpdateRunningStats(update_inputs, update_attrs);

        // Return: [output, updated_running_mean, updated_running_var, batch_mean, batch_var]
        return std::vector<Tensor>{output, updated[0], updated[1], batch_mean, batch_var};
    });

    // ========================================================================
    // Fused LayerNorm Backward
    // ========================================================================
    table.register_kernel(OpId::FusedLayerNormBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        // inputs: [grad_output, input, weight, mean, inv_std]
        auto vk = get_vulkan_backend();
        auto normalized_shape = attrs.get_int_list(AttrKey::NormalizedShape);
        int64_t normalized_size = 1;
        for (auto s : normalized_shape) normalized_size *= s;
        if (normalized_size <= 0) normalized_size = inputs[0].shape().back();
        auto [grad_input, grad_weight, grad_bias] = vk->dispatchLayerNormBackward(
            inputs[0], inputs[1], inputs[3], inputs[4], &inputs[2], normalized_size);
        return std::vector<Tensor>{grad_input, grad_weight, grad_bias};
    });

    // ========================================================================
    // Roll Operation (native Vulkan shader)
    // ========================================================================
    table.register_single_output_kernel(OpId::Roll, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t shift = attrs.get_int(AttrKey::Shift, 0);
        int64_t dim = attrs.get_int(AttrKey::Dim, 0);
        return get_vulkan_backend()->dispatchRoll(inputs[0], shift, dim);
    });

    // ========================================================================
    // RepeatInterleave (native Vulkan shader)
    // ========================================================================
    table.register_single_output_kernel(OpId::RepeatInterleave, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t dim = attrs.get_int(AttrKey::Dim, 0);
        int64_t num_repeats = attrs.get_int(AttrKey::NumRepeats, 1);
        if (num_repeats >= 0) {
            return get_vulkan_backend()->dispatchRepeatInterleave(inputs[0], num_repeats, dim);
        } else {
            return get_vulkan_backend()->dispatchRepeatInterleaveTensor(inputs[0], inputs[1], dim);
        }
    });

    // ========================================================================
    // Stack/Take/Tile/Put (native Vulkan shaders)
    // ========================================================================
    table.register_single_output_kernel(OpId::Stack, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t dim = attrs.get_int(AttrKey::Dim, 0);
        return get_vulkan_backend()->dispatchStack(inputs, dim);
    });

    table.register_single_output_kernel(OpId::Tile, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        auto reps = attrs.get_int_list(AttrKey::Reps);
        return get_vulkan_backend()->dispatchTile(inputs[0], reps);
    });

    table.register_single_output_kernel(OpId::Take, [](std::span<const Tensor> inputs, const OpAttributes&) -> Tensor {
        return get_vulkan_backend()->dispatchTake(inputs[0], inputs[1]);
    });

    table.register_single_output_kernel(OpId::Put, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        bool accumulate = attrs.get_bool(AttrKey::Accumulate, false);
        return get_vulkan_backend()->dispatchPut(inputs[0], inputs[1], inputs[2], accumulate);
    });

    // ========================================================================
    // Quantized ops (native Int8 GPU shaders)
    // ========================================================================
    table.register_single_output_kernel(OpId::QuantizedLinear,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            const auto& input = inputs[0];
            const auto& weight = inputs[1];

            // Create a zero bias if not provided
            auto weight_shape = weight.shape();
            Tensor bias = (inputs.size() > 2 && inputs[2].numel() > 0)
                ? inputs[2]
                : Tensor({weight_shape[0]}, DType::Float32, input.device());

            float input_scale = static_cast<float>(attrs.get_float(AttrKey::InputScale, 1.0));
            float weight_scale = static_cast<float>(attrs.get_float(AttrKey::WeightScaleQ, 1.0));
            int32_t input_zp = static_cast<int32_t>(attrs.get_int(AttrKey::InputZeroPoint, 0));
            int32_t weight_zp = static_cast<int32_t>(attrs.get_int(AttrKey::WeightZeroPoint, 0));

            return get_vulkan_backend()->dispatchQuantizedLinear(
                input, weight, bias, input_scale, weight_scale, input_zp, weight_zp);
        });

    table.register_single_output_kernel(OpId::QuantizedConv2d,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            const auto& input = inputs[0];
            const auto& weight = inputs[1];

            auto weight_shape = weight.shape();
            Tensor bias = (inputs.size() > 2 && inputs[2].numel() > 0)
                ? inputs[2]
                : Tensor({weight_shape[0]}, DType::Float32, input.device());

            // F.11: per-axis read with scalar fallback; Vulkan quantized conv
            // shader takes scalar stride/padding — fail loudly on asymmetric
            // rather than silently squash StrideW/PaddingW.
            const auto stride_arr  = ::tenzor::backend::attrs::stride_2d(attrs);
            const auto padding_arr = ::tenzor::backend::attrs::padding_2d(attrs);
            if (stride_arr[0] != stride_arr[1] || padding_arr[0] != padding_arr[1]) {
                throw std::invalid_argument(
                    "QuantizedConv2d (Vulkan): backend shader only supports symmetric "
                    "stride/padding; got stride=" + std::to_string(stride_arr[0]) + "x" + std::to_string(stride_arr[1]) +
                    ", padding=" + std::to_string(padding_arr[0]) + "x" + std::to_string(padding_arr[1]) + ".");
            }
            float input_scale = static_cast<float>(attrs.get_float(AttrKey::InputScale, 1.0));
            float weight_scale = static_cast<float>(attrs.get_float(AttrKey::WeightScaleQ, 1.0));
            int32_t input_zp = static_cast<int32_t>(attrs.get_int(AttrKey::InputZeroPoint, 0));
            int32_t weight_zp = static_cast<int32_t>(attrs.get_int(AttrKey::WeightZeroPoint, 0));

            return get_vulkan_backend()->dispatchQuantizedConv2d(
                input, weight, bias, stride_arr[0], padding_arr[0],
                input_scale, weight_scale, input_zp, weight_zp);
        });
    table.register_kernel(OpId::LSTMCellForward, [](std::span<const Tensor> inputs, [[maybe_unused]] const OpAttributes& attrs)
        -> std::vector<Tensor> {
        // inputs: [input, hx, cx, weight_ih, weight_hh, bias_ih, bias_hh]
        return get_vulkan_backend()->dispatchLSTMCellForward(
            inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5], inputs[6]);
    });
    table.register_kernel(OpId::LSTMCellBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        // inputs: [grad_h, grad_c_next, gates, c_prev, c_out]
        int64_t batch_size = attrs.get_int(AttrKey::BatchSize, 0);
        int64_t hidden_size = attrs.get_int(AttrKey::HiddenSize, 0);
        return get_vulkan_backend()->dispatchLSTMCellBackward(
            inputs[0], inputs[1], inputs[2], inputs[3], inputs[4],
            batch_size, hidden_size);
    });
    table.register_kernel(OpId::GRUCellForward, [](std::span<const Tensor> inputs, [[maybe_unused]] const OpAttributes& attrs)
        -> std::vector<Tensor> {
        // inputs: [input, hx, weight_ih, weight_hh, bias_ih, bias_hh]
        return std::vector<Tensor>{get_vulkan_backend()->dispatchGRUCellForward(
            inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5])};
    });
    table.register_kernel(OpId::GRUCellBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        // inputs: [grad_h, gates_x, gates_h, h_prev]
        int64_t batch_size = attrs.get_int(AttrKey::BatchSize, 0);
        int64_t hidden_size = attrs.get_int(AttrKey::HiddenSize, 0);
        return get_vulkan_backend()->dispatchGRUCellBackward(
            inputs[0], inputs[1], inputs[2], inputs[3],
            batch_size, hidden_size);
    });

    // ========================================================================
    // Complex number ops (native Vulkan shaders)
    // ========================================================================
    table.register_single_output_kernel(OpId::Conj, [](std::span<const Tensor> inputs, const OpAttributes&) -> Tensor {
        return get_vulkan_backend()->dispatchConj(inputs[0]);
    });

    table.register_single_output_kernel(OpId::Real, [](std::span<const Tensor> inputs, const OpAttributes&) -> Tensor {
        return get_vulkan_backend()->dispatchReal(inputs[0]);
    });

    table.register_single_output_kernel(OpId::Imag, [](std::span<const Tensor> inputs, const OpAttributes&) -> Tensor {
        return get_vulkan_backend()->dispatchImag(inputs[0]);
    });

    table.register_single_output_kernel(OpId::Angle, [](std::span<const Tensor> inputs, const OpAttributes&) -> Tensor {
        return get_vulkan_backend()->dispatchAngle(inputs[0]);
    });

    table.register_single_output_kernel(OpId::Polar, [](std::span<const Tensor> inputs, const OpAttributes&) -> Tensor {
        return get_vulkan_backend()->dispatchPolar(inputs[0], inputs[1]);
    });

    table.register_single_output_kernel(OpId::ComplexTensor, [](std::span<const Tensor> inputs, const OpAttributes&) -> Tensor {
        return get_vulkan_backend()->dispatchComplexTensor(inputs[0], inputs[1]);
    });

    // Linear algebra ops — single-workgroup Vulkan shaders for small matrices,
    // tiled GPU algorithms for larger matrices (inside dispatch methods)
    table.register_kernel(OpId::LinalgDet, [](std::span<const Tensor> inputs, const OpAttributes&)
        -> std::vector<Tensor> {
        return {get_vulkan_backend()->dispatchLinalgDet(inputs[0])};
    });
    table.register_kernel(OpId::LinalgInv, [](std::span<const Tensor> inputs, const OpAttributes&)
        -> std::vector<Tensor> {
        return {get_vulkan_backend()->dispatchLinalgInv(inputs[0])};
    });
    table.register_kernel(OpId::LinalgSolve, [](std::span<const Tensor> inputs, const OpAttributes&)
        -> std::vector<Tensor> {
        return {get_vulkan_backend()->dispatchLinalgSolve(inputs[0], inputs[1])};
    });
    // Cholesky — single-workgroup for small, tiled GPU for larger
    table.register_kernel(OpId::LinalgCholesky, [](std::span<const Tensor> inputs, const OpAttributes& attrs)
        -> std::vector<Tensor> {
        bool upper = attrs.get_bool(AttrKey::Upper, false);
        return {get_vulkan_backend()->dispatchLinalgCholesky(inputs[0], upper)};
    });
    // QR — single-workgroup for small, tiled GPU for larger
    table.register_kernel(OpId::LinalgQR, [](std::span<const Tensor> inputs, const OpAttributes&)
        -> std::vector<Tensor> {
        return get_vulkan_backend()->dispatchLinalgQR(inputs[0]);
    });
    // SVD — single-workgroup Jacobi for small, tiled GPU for larger
    table.register_kernel(OpId::LinalgSVD, [](std::span<const Tensor> inputs, const OpAttributes& attrs)
        -> std::vector<Tensor> {
        bool full = attrs.get_bool(AttrKey::FullMatrices, true);
        return get_vulkan_backend()->dispatchLinalgSVD(inputs[0], full);
    });
    // Eigh — single-workgroup Jacobi for small, tiled GPU for larger
    table.register_kernel(OpId::LinalgEigh, [](std::span<const Tensor> inputs, const OpAttributes&)
        -> std::vector<Tensor> {
        return get_vulkan_backend()->dispatchLinalgEigh(inputs[0]);
    });
    // Eig — general eigenvalue decomposition (single-workgroup for small, tiled GPU for larger)
    table.register_kernel(OpId::LinalgEig, [](std::span<const Tensor> inputs, const OpAttributes&)
        -> std::vector<Tensor> {
        return get_vulkan_backend()->dispatchLinalgEig(inputs[0]);
    });
    // Triangular solve — native linalg_trsm compute shader
    table.register_kernel(OpId::SolveTriangular, [](std::span<const Tensor> inputs, const OpAttributes& attrs)
        -> std::vector<Tensor> {
        bool upper = attrs.get_bool(AttrKey::Upper, true);
        bool unitriangular = attrs.get_bool(AttrKey::UnitTriangular, false);
        return {get_vulkan_backend()->dispatchLinalgSolveTriangular(inputs[0], inputs[1], upper, unitriangular)};
    });

    table.register_kernel(OpId::LinalgCholeskySolve, [](std::span<const Tensor> inputs, const OpAttributes& attrs)
        -> std::vector<Tensor> {
        bool upper = attrs.get_bool(AttrKey::Upper, false);
        auto* vk = get_vulkan_backend();
        if (!upper) {
            auto Y = vk->dispatchLinalgSolveTriangular(inputs[1], inputs[0], false, false);
            int64_t ndim = inputs[1].ndim();
            auto Lt = tenzor::transpose(inputs[1], ndim - 2, ndim - 1).contiguous();
            return {vk->dispatchLinalgSolveTriangular(Lt, Y, true, false)};
        } else {
            int64_t ndim = inputs[1].ndim();
            auto Ut = tenzor::transpose(inputs[1], ndim - 2, ndim - 1).contiguous();
            auto Y = vk->dispatchLinalgSolveTriangular(Ut, inputs[0], false, false);
            return {vk->dispatchLinalgSolveTriangular(inputs[1], Y, true, false)};
        }
    });

    // Flash Attention — composed from existing matmul + softmax shaders.
    // Both forward and backward are fully GPU-based using composed Vulkan dispatches.
    //
    // FUTURE OPTIMIZATION: Replace with a single fused SPIR-V compute shader that
    // performs Q*K^T → scale → causal mask → online softmax → probs*V entirely in
    // workgroup shared memory, eliminating intermediate global memory traffic.
    // This would require:
    //   1. A tiled algorithm processing blocks of Q rows against K/V columns
    //   2. Shared memory for the attention row (seq_len x tile_size)
    //   3. Online softmax (numerically stable, single-pass) within each tile
    //   4. Barrier synchronization between softmax and V multiplication
    // Expected speedup: 2-4x for long sequences (memory-bandwidth-bound).
    table.register_kernel(OpId::FlashAttention,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            // audit L.5: hoist the Float64 device-capability pre-check up to
            // the registry boundary so every dispatcher entry point that
            // reaches OpId::FlashAttention (autograd Function, IREE custom
            // call, nn::Attention layer, etc.) gets the same typed
            // user-facing error. Previously this guard lived only inside
            // `dispatchFlashAttention`, so callers that took alternate code
            // paths (e.g. the FP64 composed-ops slow path that still
            // compiles an FP64 SPIR-V shader downstream) would crash with a
            // cryptic SPIR-V compile failure instead.
            const Tensor& Q_pre = inputs[0];
            const Tensor& K_pre = inputs[1];
            const Tensor& V_pre = inputs[2];
            if (Q_pre.dtype() == DType::Float64
                || K_pre.dtype() == DType::Float64
                || V_pre.dtype() == DType::Float64)
            {
                int32_t device_id = Q_pre.device().index;
                DeviceInfo dev_info = get_vulkan_backend()->get_device_info(device_id);
                if (!dev_info.supports_fp64) {
                    throw std::runtime_error(
                        "Vulkan FlashAttention: Float64 requested but the active "
                        "Vulkan device does not advertise VkPhysicalDeviceFeatures::"
                        "shaderFloat64. FP64 compute shaders are not supported on "
                        "this GPU (common on mobile/integrated parts). The project "
                        "rule forbids CPU fallback or Float32 upcast — either run "
                        "on a discrete GPU with FP64 support or use a different "
                        "backend (CPU / CUDA / ROCm / OneAPI all have native FP64 "
                        "FlashAttention kernels).");
                }
            }

            // Phase 1.5: LSE is now emitted INSIDE the fused shader (binding
            // 4) in the same pass as the output. The previous composed-ops
            // `logsumexp(Q @ Kᵀ * scale)` recompute was deleted — that
            // materialised the (B, H, S_q, S_k) attention matrix and
            // defeated FlashAttention's whole reason for existing.
            //
            // Dropout: shader-level Philox is NOT supported. When
            // `dropout_p > 0` in training mode we fall back to a composed
            // softmax → Philox-mask → matmul path, which still keeps every
            // operation on-device via the `philox_dropout_mask.comp`
            // shader. The fused fast path is dropout-free.
            float scale = static_cast<float>(attrs.get_float(AttrKey::Scale, 1.0));
            bool causal = attrs.get_bool(AttrKey::Causal, false);
            float dropout_p = static_cast<float>(attrs.get_float(AttrKey::DropoutP, 0.0));
            bool is_training = attrs.get_bool(AttrKey::IsTraining, attrs.get_bool(AttrKey::Training, false));

            // Collapse 3D inputs (B, S, D) to 4D (B, 1, S, D) at the
            // dispatch boundary so downstream code only has to think about
            // a single rank. The dispatch helper itself still handles 3D
            // for the composed-ops slow path, but the registry boundary is
            // normalised here.
            const Tensor& Q_in = inputs[0];
            const Tensor& K_in = inputs[1];
            const Tensor& V_in = inputs[2];
            auto promote_to_4d = [](const Tensor& t) -> Tensor {
                if (t.ndim() == 4) return t;
                if (t.ndim() == 3) {
                    auto sh = t.shape();
                    return tenzor::reshape(t, std::vector<int64_t>{sh[0], 1, sh[1], sh[2]});
                }
                return t;
            };
            const bool was_3d = (Q_in.ndim() == 3);
            Tensor Q = promote_to_4d(Q_in);
            Tensor K = promote_to_4d(K_in);
            Tensor V = promote_to_4d(V_in);

            auto collapse_back = [&](const Tensor& t, bool drop_head_for_lse) -> Tensor {
                if (!was_3d) return t;
                auto sh = t.shape();
                if (drop_head_for_lse) {
                    // LSE: (B, 1, S_q) -> (B, S_q)
                    return tenzor::reshape(t, std::vector<int64_t>{sh[0], sh[2]});
                }
                // Output: (B, 1, S_q, Dv) -> (B, S_q, Dv)
                return tenzor::reshape(t, std::vector<int64_t>{sh[0], sh[2], sh[3]});
            };

            // Dropout path: stay composed-ops (Philox-keyed Bernoulli mask
            // built via `philox_dropout_mask.comp`, fully on-device). LSE
            // still comes from `logsumexp` of the masked pre-softmax
            // scores. Backward replays with the same (seed, offset).
            if (dropout_p > 0.0f && is_training) {
                Tensor Kt = tenzor::transpose(K, -1, -2);
                Tensor scores = tenzor::matmul(Q, Kt);
                Tensor scaled = tenzor::mul(scores, static_cast<double>(scale));
                if (causal) {
                    auto ss = scaled.shape();
                    int64_t S_q = ss[ss.size() - 2];
                    int64_t S_k = ss[ss.size() - 1];
                    Tensor row_idx = tenzor::arange(0, S_q, 1, DType::Int64, Q.device());
                    Tensor col_idx = tenzor::arange(0, S_k, 1, DType::Int64, Q.device());
                    Tensor rows_2d = tenzor::reshape(row_idx, std::vector<int64_t>{S_q, 1});
                    Tensor cols_2d = tenzor::reshape(col_idx, std::vector<int64_t>{1, S_k});
                    Tensor future = tenzor::gt(cols_2d, rows_2d);
                    std::vector<int64_t> bshape(ss.size(), 1);
                    bshape[ss.size() - 2] = S_q;
                    bshape[ss.size() - 1] = S_k;
                    Tensor future_b = tenzor::reshape(future, bshape);
                    Tensor neg_inf = tenzor::full(
                        std::vector<int64_t>(ss.begin(), ss.end()),
                        -std::numeric_limits<double>::infinity(),
                        scaled.dtype(), scaled.device());
                    scaled = tenzor::where(future_b, neg_inf, scaled);
                }
                NewOpAttributes sm_attrs;
                sm_attrs.set(AttrKey::Dim, static_cast<int64_t>(-1));
                std::vector<Tensor> sm_in = {scaled};
                Tensor attn = tenzor::dispatch(OpId::Softmax, sm_in, sm_attrs)[0];

                auto philox = tenzor::new_philox_stream();
                uint64_t seed_v = static_cast<uint64_t>(philox.seed.data<int64_t>()[0]);
                uint64_t offset_v = static_cast<uint64_t>(philox.offset.data<int64_t>()[0]);
                std::vector<int64_t> attn_shape(attn.shape().begin(), attn.shape().end());
                Tensor mask_dev = get_vulkan_backend()->dispatchPhiloxDropoutMask(
                    attn_shape, dropout_p, seed_v, offset_v);
                if (mask_dev.dtype() != attn.dtype()) {
                    mask_dev = mask_dev.to(attn.dtype());
                }
                Tensor attn_dropped = tenzor::mul(attn, mask_dev);

                Tensor output_comp = tenzor::matmul(attn_dropped, V);
                Tensor lse_comp = tenzor::logsumexp(scaled, -1, /*keepdim=*/false);
                if (lse_comp.dtype() != DType::Float32) {
                    lse_comp = lse_comp.to(DType::Float32);
                }
                return {collapse_back(output_comp, /*drop_head_for_lse=*/false),
                        collapse_back(lse_comp,   /*drop_head_for_lse=*/true),
                        philox.seed, philox.offset};
            }

            // Fused fast path (or composed slow path on dtype/shape miss):
            // `dispatchFlashAttention` now returns {output, lse} directly.
            // Both come from on-device dispatches; the registry no longer
            // touches host memory for LSE.
            auto fa_result = get_vulkan_backend()->dispatchFlashAttention(
                Q, K, V, scale, causal);
            // Philox seed/offset are empty Tensors when dropout disabled,
            // matching `attention_contract.hpp` (the contract requires
            // empty Tensors, not absent slots, so downstream code can
            // index into the return vector unconditionally).
            return {collapse_back(fa_result.first, /*drop_head_for_lse=*/false),
                    collapse_back(fa_result.second, /*drop_head_for_lse=*/true),
                    Tensor{}, Tensor{}};
        });
    // FlashAttentionBackward — composed from Vulkan matmul + softmax backward.
    // When the forward saved LSE (inputs[5]), use it to reconstruct
    // attn_weights = exp(scaled_scores - LSE) directly. That replaces the
    // separate max-reduction inside Softmax with a single elementwise
    // subtract — what makes FlashAttention backward numerically stable in
    // FP16 (audit C5 Vulkan FlashAttention: backward previously ignored the
    // LSE the forward shader emits to binding 4).
    table.register_kernel(OpId::FlashAttentionBackward,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            // inputs: [dO, Q, K, V, O] or [dO, Q, K, V, O, LSE]
            float scale = static_cast<float>(attrs.get_float(AttrKey::Scale, 1.0));
            bool causal = attrs.get_bool(AttrKey::Causal, false);

            const Tensor& dO = inputs[0];  // [B, H, S, D]
            const Tensor& Q = inputs[1];
            const Tensor& K = inputs[2];
            const Tensor& V = inputs[3];
            const bool have_lse = (inputs.size() > 5 &&
                                   inputs[5].is_valid() &&
                                   inputs[5].numel() > 0);

            auto* vk = get_vulkan_backend();

            // scores = Q @ K^T  (still needed even with LSE for the gradient
            // computation below).
            Tensor Kt = vk->dispatchTranspose(K, -1, -2);
            Tensor scores = vk->dispatchBmm(Q, Kt);  // [B, H, S, S]

            auto scores_shape = std::vector<int64_t>(scores.shape().begin(), scores.shape().end());
            Tensor scale_t = tenzor::full(scores_shape, static_cast<double>(scale),
                                           scores.dtype(), scores.device());
            scores = tenzor::mul(scores, scale_t);

            // Causal mask: -INFINITY (not -1e9 — FP16 saturates to -65504 and
            // leaks gradient mass through softmax, audit C15).
            if (causal) {
                int64_t seq_len = scores_shape[scores_shape.size() - 1];
                Tensor rows = tenzor::arange(0, seq_len, 1, DType::Int64, scores.device());
                Tensor cols = tenzor::arange(0, seq_len, 1, DType::Int64, scores.device());
                rows = vk->dispatchReshape(rows, {seq_len, 1});
                cols = vk->dispatchReshape(cols, {1, seq_len});
                Tensor causal_mask = tenzor::gt(cols.to(DType::Float32), rows.to(DType::Float32));
                Tensor neg_inf = tenzor::full(scores_shape,
                    -std::numeric_limits<float>::infinity(),
                    scores.dtype(), scores.device());
                scores = tenzor::add(scores, tenzor::mul(causal_mask.to(scores.dtype()), neg_inf));
            }

            Tensor attn_weights;
            if (have_lse) {
                // LSE has shape (..., S_q). Reshape to (..., S_q, 1) so it
                // broadcasts along the trailing kv-dim of scores.
                const Tensor& lse_raw = inputs[5];
                std::vector<int64_t> lse_shape(lse_raw.shape().begin(), lse_raw.shape().end());
                lse_shape.push_back(1);
                Tensor lse = vk->dispatchReshape(lse_raw, lse_shape);
                Tensor lse_cast = (lse.dtype() == scores.dtype())
                                    ? lse : lse.to(scores.dtype());
                attn_weights = tenzor::exp(tenzor::sub(scores, lse_cast));
            } else {
                attn_weights = vk->dispatchSoftmax(scores, -1);
            }

            // Backward through attention:
            // dV = attn_weights^T @ dO
            Tensor attn_t = vk->dispatchTranspose(attn_weights, -1, -2);
            Tensor dV = vk->dispatchBmm(attn_t, dO);

            // dAttn = dO @ V^T
            Tensor Vt = vk->dispatchTranspose(V, -1, -2);
            Tensor dAttn = vk->dispatchBmm(dO, Vt);

            // dScores = softmax_backward(dAttn, attn_weights)
            // softmax_backward: ds_i = attn_i * (dAttn_i - sum(attn * dAttn))
            Tensor attn_dAttn = tenzor::mul(attn_weights, dAttn);

            // Sum along last dim
            NewOpAttributes sum_attrs;
            sum_attrs.set(AttrKey::Dim, static_cast<int64_t>(-1));
            sum_attrs.set(AttrKey::Keepdim, true);
            std::vector<Tensor> sum_inputs = {attn_dAttn};
            auto sum_result = tenzor::dispatch(OpId::Sum, sum_inputs, sum_attrs);
            Tensor sum_ad = sum_result[0];

            Tensor dScores = tenzor::mul(attn_weights, tenzor::sub(dAttn, sum_ad));

            // Apply scale
            Tensor scale_t2 = tenzor::full(
                std::vector<int64_t>(dScores.shape().begin(), dScores.shape().end()),
                static_cast<double>(scale), dScores.dtype(), dScores.device());
            dScores = tenzor::mul(dScores, scale_t2);

            // dQ = dScores @ K
            Tensor dQ = vk->dispatchBmm(dScores, K);

            // dK = dScores^T @ Q
            Tensor dScores_t = vk->dispatchTranspose(dScores, -1, -2);
            Tensor dK = vk->dispatchBmm(dScores_t, Q);

            return {dQ, dK, dV};
        });

    // =========================================================================
    // FlexAttention (built-in score_mod registry; native programmable in M8)
    // =========================================================================
    // Per docs/internals/attention-contract.md, ScoreModId 0=identity, 1=causal.
    // Both reduce to FusedAttention; other IDs throw until M8 lands the native
    // block-sparse path (audit C21 Vulkan — was entirely unregistered).
    table.register_kernel(OpId::FlexAttention,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            float scale = static_cast<float>(attrs.get_float(AttrKey::Scale, 1.0));
            int64_t score_mod_id = attrs.get_int(AttrKey::ScoreModId, 0);

            if (score_mod_id == 0 || score_mod_id == 1) {
                bool causal = (score_mod_id == 1);
                // Phase 1.5 made dispatchFlashAttention return {output, lse}
                // — the fused shader emits per-row logsumexp directly
                // (binding 4) in the same pass as the output.
                auto [output, lse] = get_vulkan_backend()->dispatchFlashAttention(
                    inputs[0], inputs[1], inputs[2], scale, causal);

                // Phase 2.10: per the attention contract
                // (include/tenzor/ops/attention_contract.hpp,
                // docs/internals/attention-contract.md), FlexAttention
                // forward returns (output, lse) — LSE must always be a
                // valid Float32 tensor, never empty. The Phase 1.5 shader
                // emits LSE directly from the fused kernel; we just
                // narrow-cast to Float32 if it came back in a wider dtype
                // (-INFINITY sentinel for fully-masked rows is preserved).
                if (lse.dtype() != DType::Float32) {
                    lse = lse.to(DType::Float32);
                }
                return {output, lse};
            }

            if (score_mod_id == 2) {
                int64_t window_size = attrs.get_int(AttrKey::WindowSize, 0);
                if (window_size <= 0) {
                    throw std::invalid_argument(
                        "FlexAttention Vulkan: ScoreModId=2 requires AttrKey::WindowSize > 0.");
                }
                const Tensor& Q = inputs[0]; const Tensor& K = inputs[1]; const Tensor& V = inputs[2];
                int64_t S_q = Q.shape()[Q.shape().size() - 2];
                int64_t S_k = K.shape()[K.shape().size() - 2];
                Tensor Kt = tenzor::transpose(K, -1, -2);
                Tensor scores = tenzor::bmm(Q, Kt);
                auto scores_shape = std::vector<int64_t>(scores.shape().begin(), scores.shape().end());
                Tensor scale_t = tenzor::full(scores_shape, static_cast<double>(scale),
                                               scores.dtype(), scores.device());
                scores = scores * scale_t;
                int64_t half = window_size / 2;
                Tensor rows = tenzor::arange(0, S_q, 1, DType::Int64, Q.device());
                Tensor cols = tenzor::arange(0, S_k, 1, DType::Int64, Q.device());
                Tensor rows_2d = tenzor::reshape(rows.to(DType::Float32), std::vector<int64_t>{S_q, 1});
                Tensor cols_2d = tenzor::reshape(cols.to(DType::Float32), std::vector<int64_t>{1, S_k});
                Tensor abs_diff = tenzor::abs(tenzor::sub(rows_2d, cols_2d));
                Tensor half_t = tenzor::full({1}, static_cast<double>(half),
                                              abs_diff.dtype(), abs_diff.device());
                Tensor outside = tenzor::gt(abs_diff, half_t);
                Tensor neg_inf = tenzor::full(scores_shape,
                    -std::numeric_limits<float>::infinity(),
                    scores.dtype(), scores.device());
                scores = scores + (outside.to(scores.dtype()) * neg_inf);
                // Phase 2.10: emit a real Float32 LSE per the attention
                // contract — computed on the masked, pre-softmax scores so
                // that backward can recover P = exp(S - L).
                Tensor lse = tenzor::logsumexp(scores, -1, /*keepdim=*/false);
                if (lse.dtype() != DType::Float32) {
                    lse = lse.to(DType::Float32);
                }
                NewOpAttributes sm_attrs;
                sm_attrs.set(AttrKey::Dim, static_cast<int64_t>(-1));
                std::vector<Tensor> sm_in = {scores};
                Tensor probs = tenzor::dispatch(OpId::Softmax, sm_in, sm_attrs)[0];
                Tensor output = tenzor::bmm(probs, V);
                return {output, lse};
            }

            // Wave C: ScoreModId >= 3 routes through the process-wide score_mod
            // registry populated by `tenzor::nn::register_score_mod` (same
            // pattern as CPU/CUDA/OneAPI). Forward composes Q@K^T → user
            // functor → softmax → @V via tenzor:: ops (which dispatch to
            // Vulkan automatically since Q/K/V live on Vulkan).
            if (score_mod_id >= 3) {
                auto fn = tenzor::nn::find_registered_score_mod(score_mod_id);
                if (!fn) {
                    throw std::runtime_error(
                        "FlexAttention Vulkan: no user score_mod registered for ScoreModId=" +
                        std::to_string(score_mod_id) +
                        ". Register via tenzor::nn::register_score_mod(id, fn) before dispatch.");
                }
                const Tensor& Q = inputs[0]; const Tensor& K = inputs[1]; const Tensor& V = inputs[2];
                Tensor Kt = tenzor::transpose(K, -1, -2);
                Tensor scores = tenzor::bmm(Q, Kt);
                auto scores_shape = std::vector<int64_t>(scores.shape().begin(), scores.shape().end());
                Tensor scale_t = tenzor::full(scores_shape, static_cast<double>(scale),
                                               scores.dtype(), scores.device());
                scores = scores * scale_t;
                Tensor modified = fn(scores, /*b=*/0, /*h=*/0, /*q_start=*/0, /*kv_start=*/0);
                // Phase 2.10: emit a real Float32 LSE per the attention
                // contract — computed on the post-functor, pre-softmax
                // scores so that backward can recover P = exp(S - L).
                Tensor lse = tenzor::logsumexp(modified, -1, /*keepdim=*/false);
                if (lse.dtype() != DType::Float32) {
                    lse = lse.to(DType::Float32);
                }
                NewOpAttributes sm_attrs;
                sm_attrs.set(AttrKey::Dim, static_cast<int64_t>(-1));
                std::vector<Tensor> sm_in = {modified};
                Tensor probs = tenzor::dispatch(OpId::Softmax, sm_in, sm_attrs)[0];
                Tensor output = tenzor::bmm(probs, V);
                return {output, lse};
            }

            throw std::runtime_error(
                "FlexAttention Vulkan: ScoreModId=" + std::to_string(score_mod_id) +
                " not recognised (built-ins: 0=identity, 1=causal, 2=sliding_window; "
                "register user IDs >= 3 via tenzor::nn::register_score_mod).");
        });

    table.register_kernel(OpId::FlexAttentionBackward,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            float scale = static_cast<float>(attrs.get_float(AttrKey::Scale, 1.0));
            int64_t score_mod_id = attrs.get_int(AttrKey::ScoreModId, 0);

            // ScoreModId 0/1: route to fused FlashAttention backward.
            if (score_mod_id == 0 || score_mod_id == 1) {
                bool causal = (score_mod_id == 1);
                OpAttributes bwd_attrs;
                bwd_attrs.set(AttrKey::Scale, static_cast<double>(scale));
                bwd_attrs.set(AttrKey::Causal, causal);
                std::vector<Tensor> bwd_inputs(inputs.begin(), inputs.end());
                return tenzor::dispatch(OpId::FlashAttentionBackward, bwd_inputs, bwd_attrs);
            }

            // Wave C: ScoreModId == 2 (sliding window) or >= 3 (user functor) — composed backward.
            if (score_mod_id == 2 || score_mod_id >= 3) {
                const Tensor& dO = inputs[0];
                const Tensor& Q  = inputs[1];
                const Tensor& K  = inputs[2];
                const Tensor& V  = inputs[3];
                Tensor Kt = tenzor::transpose(K, -1, -2);
                Tensor scores = tenzor::bmm(Q, Kt);
                auto scores_shape = std::vector<int64_t>(scores.shape().begin(), scores.shape().end());
                Tensor scale_t = tenzor::full(scores_shape, static_cast<double>(scale),
                                               scores.dtype(), scores.device());
                scores = scores * scale_t;

                if (score_mod_id == 2) {
                    int64_t window_size = attrs.get_int(AttrKey::WindowSize, 0);
                    if (window_size <= 0) {
                        throw std::invalid_argument(
                            "FlexAttentionBackward Vulkan: ScoreModId=2 requires AttrKey::WindowSize > 0.");
                    }
                    int64_t S_q = Q.shape()[Q.shape().size() - 2];
                    int64_t S_k = K.shape()[K.shape().size() - 2];
                    int64_t half = window_size / 2;
                    Tensor rows = tenzor::arange(0, S_q, 1, DType::Int64, Q.device());
                    Tensor cols = tenzor::arange(0, S_k, 1, DType::Int64, Q.device());
                    Tensor rows_2d = tenzor::reshape(rows.to(DType::Float32), std::vector<int64_t>{S_q, 1});
                    Tensor cols_2d = tenzor::reshape(cols.to(DType::Float32), std::vector<int64_t>{1, S_k});
                    Tensor abs_diff = tenzor::abs(tenzor::sub(rows_2d, cols_2d));
                    Tensor half_t = tenzor::full({1}, static_cast<double>(half),
                                                  abs_diff.dtype(), abs_diff.device());
                    Tensor outside = tenzor::gt(abs_diff, half_t);
                    // Use large finite negative; softmax(-1e30) underflows to 0
                    // (same effect as -inf) without producing NaN at in-window
                    // positions via 0 * -inf.
                    Tensor large_neg = tenzor::full(scores_shape, -1.0e30,
                                                     scores.dtype(), scores.device());
                    scores = scores + (outside.to(scores.dtype()) * large_neg);
                } else {
                    auto fn = tenzor::nn::find_registered_score_mod(score_mod_id);
                    if (!fn) {
                        throw std::runtime_error(
                            "FlexAttentionBackward Vulkan: no user score_mod registered for ScoreModId=" +
                            std::to_string(score_mod_id));
                    }
                    scores = fn(scores, 0, 0, 0, 0);
                }

                NewOpAttributes sm_attrs;
                sm_attrs.set(AttrKey::Dim, static_cast<int64_t>(-1));
                std::vector<Tensor> sm_in = {scores};
                Tensor attn = tenzor::dispatch(OpId::Softmax, sm_in, sm_attrs)[0];

                Tensor attn_t = tenzor::transpose(attn, -1, -2);
                Tensor dV = tenzor::bmm(attn_t, dO);

                Tensor Vt = tenzor::transpose(V, -1, -2);
                Tensor dAttn = tenzor::bmm(dO, Vt);

                Tensor ad = tenzor::mul(attn, dAttn);
                NewOpAttributes sum_attrs;
                sum_attrs.set(AttrKey::Dim, static_cast<int64_t>(-1));
                sum_attrs.set(AttrKey::Keepdim, true);
                std::vector<Tensor> sum_inputs = {ad};
                Tensor sum_ad = tenzor::dispatch(OpId::Sum, sum_inputs, sum_attrs)[0];
                Tensor dScores = tenzor::mul(attn, tenzor::sub(dAttn, sum_ad));

                Tensor scale_t2 = tenzor::full(
                    std::vector<int64_t>(dScores.shape().begin(), dScores.shape().end()),
                    static_cast<double>(scale), dScores.dtype(), dScores.device());
                dScores = tenzor::mul(dScores, scale_t2);

                Tensor dQ = tenzor::bmm(dScores, K);
                Tensor dScores_t = tenzor::transpose(dScores, -1, -2);
                Tensor dK = tenzor::bmm(dScores_t, Q);

                return {dQ, dK, dV};
            }

            throw std::runtime_error(
                "FlexAttentionBackward Vulkan: ScoreModId=" + std::to_string(score_mod_id) +
                " not recognised.");
        });

    // =========================================================================
    // Einsum (composed — delegates to einsum_composed to avoid dispatch loop)
    // =========================================================================
    table.register_kernel(OpId::Einsum, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
        auto equation = std::string(attrs.get_string(AttrKey::EinsumEquation, ""));
        std::vector<Tensor> tensors(inputs.begin(), inputs.end());
        return {einsum_composed(equation, tensors)};
    });

    // SearchSorted — native GPU binary search shader
    table.register_single_output_kernel(OpId::SearchSorted,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> Tensor {
            return get_vulkan_backend()->dispatchSearchSorted(inputs[0], inputs[1]);
        });

    // GumbelSoftmax — composed from existing Vulkan ops (no dedicated shader needed)
    table.register_single_output_kernel(OpId::GumbelSoftmax,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            const Tensor& logits = inputs[0];
            double tau = attrs.get_float(AttrKey::Tau, 1.0);
            bool hard = attrs.get_bool(AttrKey::Hard, false);
            int64_t dim = attrs.get_int(AttrKey::Dim, -1);

            auto shape_vec = std::vector<int64_t>(logits.shape().begin(), logits.shape().end());

            // Gumbel noise: -log(-log(U)) where U ~ Uniform(0, 1)
            Tensor u = tenzor::rand(shape_vec, logits.dtype(), logits.device());
            Tensor eps_tensor = tenzor::full(shape_vec, 1e-20, logits.dtype(), logits.device());
            u = tenzor::add(u, eps_tensor);

            Tensor gumbels = tenzor::neg(tenzor::log(tenzor::neg(tenzor::log(u))));
            Tensor scaled = tenzor::div(tenzor::add(logits, gumbels),
                                tenzor::full(shape_vec, tau, logits.dtype(), logits.device()));

            NewOpAttributes sm_attrs;
            sm_attrs.set(AttrKey::Dim, dim);
            std::array<Tensor, 1> sm_inputs = {scaled};
            Tensor y_soft = dispatch<OpId::Softmax>(sm_inputs, sm_attrs)[0];

            if (!hard) return y_soft;

            // Straight-through estimator
            int64_t actual_dim = dim < 0 ? dim + logits.ndim() : dim;
            Tensor indices = argmax(y_soft, std::make_optional(actual_dim), /*keepdim=*/true);
            Tensor y_hard = tenzor::zeros(shape_vec, logits.dtype(), logits.device());
            std::array<Tensor, 3> scatter_inputs = {y_hard, indices,
                tenzor::full(std::vector<int64_t>(indices.shape().begin(), indices.shape().end()),
                     1.0, logits.dtype(), logits.device())};
            NewOpAttributes scatter_attrs;
            scatter_attrs.set(AttrKey::Dim, actual_dim);
            y_hard = dispatch<OpId::Scatter>(scatter_inputs, scatter_attrs)[0];

            return tenzor::add(tenzor::sub(y_hard, y_soft.detach()), y_soft);
        });

    // ========================================================================
    // FFT Operations — Native Vulkan compute shaders (Cooley-Tukey radix-2,
    // mixed-radix Stockham for factorable sizes, Bluestein for others — all on GPU).
    // Non-last-dim transforms use GPU transpose.
    // ========================================================================

    table.register_single_output_kernel(OpId::FFT, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t dim = attrs.get_int(AttrKey::Dim, -1);
        int64_t actual_dim = dim < 0 ? dim + inputs[0].ndim() : dim;
        int64_t n = attrs.get_int(AttrKey::N, inputs[0].shape()[actual_dim]);
        std::string norm(attrs.get_string(AttrKey::Norm, "backward"));
        return get_vulkan_backend()->dispatchFFT(inputs[0], dim, n, norm);
    });

    table.register_single_output_kernel(OpId::IFFT, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t dim = attrs.get_int(AttrKey::Dim, -1);
        int64_t actual_dim = dim < 0 ? dim + inputs[0].ndim() : dim;
        int64_t n = attrs.get_int(AttrKey::N, inputs[0].shape()[actual_dim]);
        std::string norm(attrs.get_string(AttrKey::Norm, "backward"));
        return get_vulkan_backend()->dispatchIFFT(inputs[0], dim, n, norm);
    });

    table.register_single_output_kernel(OpId::RFFT, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t dim = attrs.get_int(AttrKey::Dim, -1);
        int64_t actual_dim = dim < 0 ? dim + inputs[0].ndim() : dim;
        int64_t n = attrs.get_int(AttrKey::N, inputs[0].shape()[actual_dim]);
        std::string norm(attrs.get_string(AttrKey::Norm, "backward"));
        return get_vulkan_backend()->dispatchRFFT(inputs[0], dim, n, norm);
    });

    table.register_single_output_kernel(OpId::IRFFT, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t dim = attrs.get_int(AttrKey::Dim, -1);
        int64_t actual_dim = dim < 0 ? dim + inputs[0].ndim() : dim;
        int64_t n = attrs.get_int(AttrKey::N, inputs[0].shape()[actual_dim]);
        std::string norm(attrs.get_string(AttrKey::Norm, "backward"));
        return get_vulkan_backend()->dispatchIRFFT(inputs[0], dim, n, norm);
    });

    table.register_single_output_kernel(OpId::FFT2, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        auto dims = attrs.get_int_list(AttrKey::Dims);
        if (dims.empty()) {
            int64_t ndim = inputs[0].ndim();
            dims = {ndim - 2, ndim - 1};
        }
        std::string norm(attrs.get_string(AttrKey::Norm, "backward"));
        return get_vulkan_backend()->dispatchFFT2(inputs[0], dims, norm);
    });

    table.register_single_output_kernel(OpId::IFFT2, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        auto dims = attrs.get_int_list(AttrKey::Dims);
        if (dims.empty()) {
            int64_t ndim = inputs[0].ndim();
            dims = {ndim - 2, ndim - 1};
        }
        std::string norm(attrs.get_string(AttrKey::Norm, "backward"));
        return get_vulkan_backend()->dispatchIFFT2(inputs[0], dims, norm);
    });

    table.register_single_output_kernel(OpId::FFTN, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        auto dims = attrs.get_int_list(AttrKey::Dims);
        if (dims.empty()) {
            dims.resize(inputs[0].ndim());
            for (int64_t i = 0; i < inputs[0].ndim(); ++i) dims[i] = i;
        }
        std::string norm(attrs.get_string(AttrKey::Norm, "backward"));
        return get_vulkan_backend()->dispatchFFTN(inputs[0], dims, norm);
    });

    table.register_single_output_kernel(OpId::IFFTN, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        auto dims = attrs.get_int_list(AttrKey::Dims);
        if (dims.empty()) {
            dims.resize(inputs[0].ndim());
            for (int64_t i = 0; i < inputs[0].ndim(); ++i) dims[i] = i;
        }
        std::string norm(attrs.get_string(AttrKey::Norm, "backward"));
        return get_vulkan_backend()->dispatchIFFTN(inputs[0], dims, norm);
    });

    // Sparse tensor operations (OpIds 460-464)
    table.register_kernel(OpId::SparseSpMM, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
        int64_t M = attrs.get_int(AttrKey::M, 0);
        int64_t K = attrs.get_int(AttrKey::K, 0);
        // Infer N from the dense operand's trailing dim if the caller did not
        // supply it — matches the CPU dispatch where N is read off the dense
        // shape rather than from the attribute set. Older callers that pass
        // N explicitly still win because the attribute takes precedence.
        int64_t default_N = (inputs.size() >= 4 && inputs[3].ndim() == 2)
                                ? inputs[3].shape()[1] : 0;
        int64_t N = attrs.get_int(AttrKey::N, default_N);
        return {get_vulkan_backend()->dispatchSparseSpMM(inputs[0], inputs[1], inputs[2], inputs[3], M, K, N)};
    });

    table.register_kernel(OpId::SparseSpMV, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
        int64_t M = attrs.get_int(AttrKey::M, 0);
        int64_t K = attrs.get_int(AttrKey::K, 0);
        return {get_vulkan_backend()->dispatchSparseSpMV(inputs[0], inputs[1], inputs[2], inputs[3], M, K)};
    });

    table.register_kernel(OpId::SparseToDense, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
        int64_t M = attrs.get_int(AttrKey::M, 0);
        int64_t K = attrs.get_int(AttrKey::K, 0);
        DType dtype = inputs[2].dtype();  // values tensor dtype
        return {get_vulkan_backend()->dispatchSparseToDense(inputs[0], inputs[1], inputs[2], M, K, dtype)};
    });

    table.register_kernel(OpId::SparseAdd, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
        int64_t M = attrs.get_int(AttrKey::M, 0);
        int64_t K = attrs.get_int(AttrKey::K, 0);
        return {get_vulkan_backend()->dispatchSparseAdd(inputs[0], inputs[1], inputs[2], inputs[3], M, K)};
    });

    table.register_kernel(OpId::DenseToSparse, [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
        return get_vulkan_backend()->dispatchDenseToSparse(inputs[0]);
    });

    // Sparse SpGEMM, Trsv, Trsm (OpIds 465-467)
    table.register_kernel(OpId::SparseSpGEMM,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            int64_t M = attrs.get_int(AttrKey::M);
            int64_t K = attrs.get_int(AttrKey::K);
            int64_t N = attrs.get_int(AttrKey::N);
            return get_vulkan_backend()->dispatchSparseSpGEMM(
                inputs[0], inputs[1], inputs[2],
                inputs[3], inputs[4], inputs[5],
                M, K, N);
        });

    table.register_single_output_kernel(OpId::SparseTrsv,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            int64_t N = attrs.get_int(AttrKey::N);
            bool upper = attrs.get_bool(AttrKey::Upper, false);
            return get_vulkan_backend()->dispatchSparseTrsv(
                inputs[0], inputs[1], inputs[2], inputs[3], N, upper);
        });

    table.register_single_output_kernel(OpId::SparseTrsm,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            int64_t N = attrs.get_int(AttrKey::N);
            bool upper = attrs.get_bool(AttrKey::Upper, false);
            int64_t K_rhs = inputs[3].shape().size() > 1 ? inputs[3].shape()[1] : 1;
            return get_vulkan_backend()->dispatchSparseTrsm(
                inputs[0], inputs[1], inputs[2], inputs[3], N, K_rhs, upper);
        });

    // SparseSoftmax: row-wise softmax on CSR sparse tensor values
    table.register_single_output_kernel(OpId::SparseSoftmax,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            auto shape = attrs.get_int_list(AttrKey::Shape);
            auto sp = SparseTensor::sparse_csr(inputs[0], inputs[1], inputs[2], shape);
            auto result = sparse::sparse_softmax(sp);
            return result.values();
        });

    // SparseLogSoftmax: row-wise log-softmax on CSR sparse tensor values
    table.register_single_output_kernel(OpId::SparseLogSoftmax,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            auto shape = attrs.get_int_list(AttrKey::Shape);
            auto sp = SparseTensor::sparse_csr(inputs[0], inputs[1], inputs[2], shape);
            auto result = sparse::sparse_log_softmax(sp);
            return result.values();
        });

    // ========================================================================
    // Sampling / Statistics — native Vulkan compute shaders
    // ========================================================================

    table.register_single_output_kernel(OpId::Bernoulli,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> Tensor {
            return get_vulkan_backend()->dispatchBernoulli(inputs[0]);
        });

    table.register_single_output_kernel(OpId::PoissonSample,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> Tensor {
            return get_vulkan_backend()->dispatchPoissonSample(inputs[0]);
        });

    table.register_single_output_kernel(OpId::NormalSample,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> Tensor {
            return get_vulkan_backend()->dispatchNormalSample(inputs[0], inputs[1]);
        });

    table.register_single_output_kernel(OpId::ExponentialSample,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> Tensor {
            return get_vulkan_backend()->dispatchExponentialSample(inputs[0]);
        });

    table.register_single_output_kernel(OpId::Multinomial,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            int64_t num_samples = attrs.get_int(AttrKey::NumSamples, 1);
            bool replacement = attrs.get_bool(AttrKey::Replacement, false);
            return get_vulkan_backend()->dispatchMultinomial(inputs[0], num_samples, replacement);
        });

    table.register_single_output_kernel(OpId::Bucketize,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            bool right = attrs.get_bool(AttrKey::Right, false);
            return get_vulkan_backend()->dispatchBucketize(inputs[0], inputs[1], right);
        });

    table.register_kernel(OpId::Histogram,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            int64_t bins = attrs.get_int(AttrKey::NumBins, 10);
            double min_val = attrs.get_float(AttrKey::Min, 0.0);
            double max_val = attrs.get_float(AttrKey::Max, 0.0);
            auto [counts, edges] = get_vulkan_backend()->dispatchHistogram(inputs[0], bins, min_val, max_val);
            return {counts, edges};
        });

    table.register_kernel(OpId::Histogramdd,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            // Parse bins list from comma-separated string
            auto bins_list = attrs.get_int_list(AttrKey::BinsList);
            bool density = attrs.get_bool(AttrKey::Density, false);

            // Parse ranges from comma-separated string: min0,max0,min1,max1,...
            std::vector<std::pair<double,double>> ranges;
            auto ranges_str = attrs.get_string(AttrKey::RangesList, "");
            if (!ranges_str.empty()) {
                std::vector<double> vals;
                std::string s(ranges_str);
                size_t pos = 0;
                while (pos < s.size()) {
                    size_t next = s.find(',', pos);
                    if (next == std::string::npos) next = s.size();
                    vals.push_back(std::stod(s.substr(pos, next - pos)));
                    pos = next + 1;
                }
                for (size_t i = 0; i + 1 < vals.size(); i += 2) {
                    ranges.emplace_back(vals[i], vals[i + 1]);
                }
            }

            auto [counts, edges] = get_vulkan_backend()->dispatchHistogramdd(inputs[0], bins_list, ranges, density);
            std::vector<Tensor> results;
            results.push_back(counts);
            for (auto& e : edges) results.push_back(std::move(e));
            return results;
        });

    table.register_single_output_kernel(OpId::CDist,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            double p = attrs.get_float(AttrKey::DistP, 2.0);
            return get_vulkan_backend()->dispatchCDist(inputs[0], inputs[1], p);
        });

    // =========================================================================
    // Trapezoid / Cumulative Trapezoid / Gradient / PairwiseDistance / Pdist
    // =========================================================================
    table.register_single_output_kernel(OpId::Trapezoid, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t dim = attrs.get_int(AttrKey::Dim, -1);
        double dx = attrs.get_float(AttrKey::Dx, 1.0);
        const Tensor* x_ptr = (inputs.size() > 1) ? &inputs[1] : nullptr;
        return get_vulkan_backend()->dispatchTrapezoid(inputs[0], dim, dx, x_ptr);
    });

    table.register_single_output_kernel(OpId::CumulativeTrapezoid, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t dim = attrs.get_int(AttrKey::Dim, -1);
        double dx = attrs.get_float(AttrKey::Dx, 1.0);
        const Tensor* x_ptr = (inputs.size() > 1) ? &inputs[1] : nullptr;
        return get_vulkan_backend()->dispatchCumulativeTrapezoid(inputs[0], dim, dx, x_ptr);
    });

    table.register_single_output_kernel(OpId::NumericalGradient, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t dim = attrs.get_int(AttrKey::Dim, -1);
        double spacing = attrs.get_float(AttrKey::Spacing, 1.0);
        return get_vulkan_backend()->dispatchGradient(inputs[0], dim, spacing);
    });

    table.register_single_output_kernel(OpId::PairwiseDistance, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        double p = attrs.get_float(AttrKey::DistP, 2.0);
        return get_vulkan_backend()->dispatchPairwiseDistance(inputs[0], inputs[1], p);
    });

    table.register_single_output_kernel(OpId::Pdist, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        double p = attrs.get_float(AttrKey::DistP, 2.0);
        return get_vulkan_backend()->dispatchPdist(inputs[0], p);
    });

    // STFT / ISTFT — Phase 2.3: native Vulkan implementations.
    //
    // Root cause of the prior reconstruction divergence was in
    // dispatchRFFT's descriptor binding size for the unpack step:
    // `output.numel() * 4` was used as the byte size for a Complex64
    // buffer, which is half the actual required byte count. On
    // permissive drivers this silently produced wrong-valued spectra
    // (the "forward alone produces wrong-valued spectra" symptom noted
    // in the old TODO comment), which cascaded into the STFT round
    // trip. Fixed in vulkan_ops_fft.cpp by using complex_elem_size.
    table.register_single_output_kernel(OpId::STFT,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            int64_t n_fft = attrs.get_int(AttrKey::NFft);
            int64_t hop_length = attrs.get_int(AttrKey::HopLength, -1);
            int64_t win_length = attrs.get_int(AttrKey::WinLength, -1);
            bool center = attrs.get_bool(AttrKey::Centered, true);
            bool normalized = attrs.get_bool(AttrKey::Normalized, false);
            bool onesided = attrs.get_bool(AttrKey::OnesidedAttr, true);
            Tensor window = (inputs.size() > 1) ? inputs[1] : Tensor();
            return get_vulkan_backend()->dispatchSTFT(
                inputs[0], n_fft, hop_length, win_length, window,
                center, normalized, onesided);
        });

    table.register_single_output_kernel(OpId::ISTFT,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            int64_t n_fft = attrs.get_int(AttrKey::NFft);
            int64_t hop_length = attrs.get_int(AttrKey::HopLength, -1);
            int64_t win_length = attrs.get_int(AttrKey::WinLength, -1);
            bool center = attrs.get_bool(AttrKey::Centered, true);
            bool normalized = attrs.get_bool(AttrKey::Normalized, false);
            bool onesided = attrs.get_bool(AttrKey::OnesidedAttr, true);
            int64_t length_val = attrs.get_int(AttrKey::N, -1);
            Tensor window = (inputs.size() > 1) ? inputs[1] : Tensor();
            return get_vulkan_backend()->dispatchISTFT(
                inputs[0], n_fft, hop_length, win_length, window,
                center, normalized, onesided, length_val);
        });

    // =========================================================================
    // DCT / IDCT
    // =========================================================================
    table.register_single_output_kernel(OpId::DCT,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            int type = static_cast<int>(attrs.get_int(AttrKey::DCTType, 2));
            int64_t dim = attrs.get_int(AttrKey::Dim, -1);
            std::string norm{attrs.get_string(AttrKey::Norm, "backward")};
            std::optional<int64_t> n = std::nullopt;
            int64_t n_val = attrs.get_int(AttrKey::N, -1);
            if (n_val > 0) n = n_val;
            return fft::dct(inputs[0], type, n, dim, norm);
        });

    table.register_single_output_kernel(OpId::IDCT,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            int type = static_cast<int>(attrs.get_int(AttrKey::DCTType, 2));
            int64_t dim = attrs.get_int(AttrKey::Dim, -1);
            std::string norm{attrs.get_string(AttrKey::Norm, "backward")};
            std::optional<int64_t> n = std::nullopt;
            int64_t n_val = attrs.get_int(AttrKey::N, -1);
            if (n_val > 0) n = n_val;
            return fft::idct(inputs[0], type, n, dim, norm);
        });

    table.register_single_output_kernel(OpId::MelScale,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            int64_t n_mels = attrs.get_int(AttrKey::NumMels, 128);
            double f_min = attrs.get_float(AttrKey::FMin, 0.0);
            double f_max = attrs.get_float(AttrKey::FMax, 0.0);
            int64_t sample_rate = attrs.get_int(AttrKey::SampleRate, 16000);
            return fft::mel_scale(inputs[0], n_mels, f_min, f_max, sample_rate);
        });

    table.register_single_output_kernel(OpId::MFCC,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            int64_t sample_rate = attrs.get_int(AttrKey::SampleRate, 16000);
            int64_t n_mfcc = attrs.get_int(AttrKey::NumMFCC, 40);
            int64_t n_mels = attrs.get_int(AttrKey::NumMels, 128);
            int64_t n_fft = attrs.get_int(AttrKey::NFft, 400);
            int64_t hop_length = attrs.get_int(AttrKey::HopLength, 160);
            double f_min = attrs.get_float(AttrKey::FMin, 0.0);
            double f_max = attrs.get_float(AttrKey::FMax, 0.0);
            return fft::mfcc(inputs[0], sample_rate, n_mfcc, n_mels, n_fft, hop_length, f_min, f_max);
        });

    // AdvancedIndex / AdvancedIndexPut — native Vulkan compute shaders
    table.register_single_output_kernel(OpId::AdvancedIndex,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            int64_t num_indices = attrs.get_int(AttrKey::NumIndices, 0);
            std::vector<Tensor> indices(inputs.begin() + 1, inputs.end());
            return get_vulkan_backend()->dispatchAdvancedIndex(inputs[0], indices, num_indices);
        });

    table.register_single_output_kernel(OpId::AdvancedIndexPut,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            int64_t num_indices = attrs.get_int(AttrKey::NumIndices, 0);
            // inputs[0] = destination, inputs[1] = values, inputs[2..2+N) = indices
            const auto& values = inputs[1];
            std::vector<Tensor> indices(inputs.begin() + 2, inputs.begin() + 2 + num_indices);
            return get_vulkan_backend()->dispatchAdvancedIndexPut(inputs[0], indices, values, num_indices);
        });

    // Geqrf — raw QR factorization returning packed reflectors + tau
    table.register_kernel(OpId::Geqrf, [](std::span<const Tensor> inputs, const OpAttributes&)
        -> std::vector<Tensor> {
        return get_vulkan_backend()->dispatchGeqrf(inputs[0]);
    });
    // Ormqr — multiply matrix by Q from QR factorization
    table.register_single_output_kernel(OpId::Ormqr, [](std::span<const Tensor> inputs, const OpAttributes& attrs)
        -> Tensor {
        bool left = attrs.get_bool(AttrKey::Left, true);
        bool transpose_q = attrs.get_bool(AttrKey::TransposeQ, false);
        return get_vulkan_backend()->dispatchOrmqr(inputs[0], inputs[1], inputs[2], left, transpose_q);
    });

    // LinalgLU / LinalgLUSolve — native blocked LU + TRSM backsolve
    table.register_kernel(OpId::LinalgLU,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
            return get_vulkan_backend()->dispatchLinalgLU(inputs[0]);
        });

    table.register_single_output_kernel(OpId::LinalgLUSolve,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> Tensor {
            return get_vulkan_backend()->dispatchLinalgLUSolve(inputs[0], inputs[1], inputs[2]);
        });

    // ========================================================================
    // LinalgHouseholder, LinalgLDLFactor, LinalgLDLSolve,
    // CholeskyInverse, TensorInv, TensorSolve
    // ========================================================================
    table.register_single_output_kernel(OpId::LinalgHouseholder,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> Tensor {
            return get_vulkan_backend()->dispatchLinalgHouseholder(inputs[0], inputs[1]);
        });

    table.register_kernel(OpId::LinalgLDLFactor,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
            return get_vulkan_backend()->dispatchLinalgLDLFactor(inputs[0]);
        });

    table.register_single_output_kernel(OpId::LinalgLDLSolve,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> Tensor {
            return get_vulkan_backend()->dispatchLinalgLDLSolve(inputs[0], inputs[1], inputs[2]);
        });

    table.register_single_output_kernel(OpId::CholeskyInverse,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            bool upper = attrs.get_bool(AttrKey::Upper, false);
            return linalg::cholesky_inverse(inputs[0], upper);
        });

    table.register_single_output_kernel(OpId::TensorInv,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            int64_t ind = attrs.get_int(AttrKey::Ind, 2);
            return linalg::tensorinv(inputs[0], ind);
        });

    table.register_single_output_kernel(OpId::TensorSolve,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> Tensor {
            return linalg::tensorsolve(inputs[0], inputs[1]);
        });

    // ========================================================================
    // New Phase 4 ops
    // ========================================================================
    table.register_kernel(OpId::Frac, [](std::span<const Tensor> inputs, const OpAttributes&) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchUnaryOp("frac", inputs[0])};
    });
    table.register_kernel(OpId::Heaviside, [](std::span<const Tensor> inputs, const OpAttributes&) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchHeaviside(inputs[0], inputs[1])};
    });
    table.register_kernel(OpId::LogSigmoid, [](std::span<const Tensor> inputs, const OpAttributes&) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchUnaryOp("log_sigmoid", inputs[0])};
    });
    // NanToNum: native Vulkan dispatch via nan_to_num.comp shader
    table.register_kernel(OpId::NanToNum, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
        float nan_val    = static_cast<float>(attrs.get_float(AttrKey::NanValue, 0.0));
        float posinf_val = static_cast<float>(attrs.get_float(AttrKey::PosInfValue, static_cast<double>(std::numeric_limits<float>::max())));
        float neginf_val = static_cast<float>(attrs.get_float(AttrKey::NegInfValue, static_cast<double>(std::numeric_limits<float>::lowest())));
        return std::vector<Tensor>{get_vulkan_backend()->dispatchNanToNum(inputs[0], nan_val, posinf_val, neginf_val)};
    });

    // Bitwise ops on Vulkan: SPIR-V shaders are int32-only (AND/OR/XOR/NOT/
    // shifts). For Int8/Int16/Int64 inputs we promote to Int32 around the
    // shader call. Bit-preservation guarantees:
    //   - AND/OR/XOR/NOT: bit patterns are preserved through sign-extension
    //     and narrowing for any width whose values fit in Int32.
    //   - Shifts: C++ promotes narrow integer operands to int before shifting
    //     anyway, so the cast-promote-cast pattern matches native semantics
    //     when the shift count is < 32 and the value fits in Int32.
    //   - Int64 values outside [-2^31, 2^31) would lose information through
    //     this path; until a dedicated int64 shader is added, callers passing
    //     such values will get truncated bits in the high word.
    //
    // The narrow-back step (Int32 → original dtype) goes through CPU because
    // Vulkan's cast_f32_i8 / cast_f32_i16 shaders saturate to the target
    // range while PyTorch/numpy/CPU semantics is modular truncation. The
    // narrow-back goes through `dispatchCastTruncateInt32`, which uses
    // dedicated truncating shaders (cast_i32_i8_truncate / cast_i32_i16 /
    // cast_i32_bool / cast_i32_i64) — entirely on-device, no CPU roundtrip.
    table.register_kernel(OpId::BitwiseAnd, [](std::span<const Tensor> inputs, const OpAttributes&) {
        DType d = inputs[0].dtype();
        if (d == DType::Int32) {
            return std::vector<Tensor>{get_vulkan_backend()->dispatchBitwiseBinaryOp("bitwise_and", inputs[0], inputs[1])};
        }
        Tensor a32 = inputs[0].to(DType::Int32), b32 = inputs[1].to(DType::Int32);
        Tensor out32 = get_vulkan_backend()->dispatchBitwiseBinaryOp("bitwise_and", a32, b32);
        return std::vector<Tensor>{get_vulkan_backend()->dispatchCastTruncateInt32(out32, d)};
    });
    table.register_kernel(OpId::BitwiseOr, [](std::span<const Tensor> inputs, const OpAttributes&) {
        DType d = inputs[0].dtype();
        if (d == DType::Int32) {
            return std::vector<Tensor>{get_vulkan_backend()->dispatchBitwiseBinaryOp("bitwise_or", inputs[0], inputs[1])};
        }
        Tensor a32 = inputs[0].to(DType::Int32), b32 = inputs[1].to(DType::Int32);
        Tensor out32 = get_vulkan_backend()->dispatchBitwiseBinaryOp("bitwise_or", a32, b32);
        return std::vector<Tensor>{get_vulkan_backend()->dispatchCastTruncateInt32(out32, d)};
    });
    table.register_kernel(OpId::BitwiseXor, [](std::span<const Tensor> inputs, const OpAttributes&) {
        DType d = inputs[0].dtype();
        if (d == DType::Int32) {
            return std::vector<Tensor>{get_vulkan_backend()->dispatchBitwiseBinaryOp("bitwise_xor", inputs[0], inputs[1])};
        }
        Tensor a32 = inputs[0].to(DType::Int32), b32 = inputs[1].to(DType::Int32);
        Tensor out32 = get_vulkan_backend()->dispatchBitwiseBinaryOp("bitwise_xor", a32, b32);
        return std::vector<Tensor>{get_vulkan_backend()->dispatchCastTruncateInt32(out32, d)};
    });
    table.register_kernel(OpId::BitwiseNot, [](std::span<const Tensor> inputs, const OpAttributes&) {
        const Tensor& in = inputs[0];
        if (in.dtype() == DType::Int32) {
            return std::vector<Tensor>{get_vulkan_backend()->dispatchUnaryOp("bitwise_not", in)};
        }
        Tensor in32 = in.to(DType::Int32);
        Tensor out32 = get_vulkan_backend()->dispatchUnaryOp("bitwise_not", in32);
        return std::vector<Tensor>{get_vulkan_backend()->dispatchCastTruncateInt32(out32, in.dtype())};
    });

    // Native Vulkan RReLU forward
    table.register_kernel(OpId::RReLU, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        float lower = static_cast<float>(attrs.get_float(AttrKey::Lower, 0.125));
        float upper = static_cast<float>(attrs.get_float(AttrKey::High, 0.333));
        bool training = attrs.get_bool(AttrKey::Training, false);
        return std::vector<Tensor>{get_vulkan_backend()->dispatchRReLU(inputs[0], lower, upper, training)};
    });

    // Native Vulkan RReLU backward
    table.register_kernel(OpId::RReLUBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        float lower = static_cast<float>(attrs.get_float(AttrKey::Lower, 0.125));
        float upper = static_cast<float>(attrs.get_float(AttrKey::High, 0.333));
        float slope = (lower + upper) / 2.0f;
        return std::vector<Tensor>{get_vulkan_backend()->dispatchRReLUBackward(inputs[0], inputs[1], slope)};
    });

    // Native Vulkan LogSigmoid backward
    table.register_kernel(OpId::LogSigmoidBackward, [](std::span<const Tensor> inputs, const OpAttributes&) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchLogSigmoidBackward(inputs[0], inputs[1])};
    });

    // Native Vulkan CountNonzero
    table.register_single_output_kernel(OpId::CountNonzero, [](std::span<const Tensor> inputs, const OpAttributes&) -> Tensor {
        return get_vulkan_backend()->dispatchCountNonzero(inputs[0]);
    });

    // Native Vulkan Nansum / Nanmean.
    //
    // The native dispatchNansum/dispatchNanmean shaders always reduce to a
    // scalar — they neither read AttrKey::Dim nor AttrKey::Keepdim. When
    // the user calls nansum(x, dim=k), the composed ops in
    // src/ops/reduction.cpp expect a dim-aware reduction; routing that
    // through the scalar shader was returning the same value broadcast
    // across all rows, which silently broke nanvar's per-row variance
    // (caught by tests/backend_parity/test_nanstats_parity).
    //
    // Compose dim-aware nansum/nanmean from existing primitives so the
    // result has the correct shape and per-slice values:
    //     nansum(x, dim, keepdim)
    //         = sum(where(isnan(x), 0, x), dim, keepdim)
    //     nanmean(x, dim, keepdim)
    //         = nansum / count_non_nan
    table.register_single_output_kernel(OpId::Nansum, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        const Tensor& x = inputs[0];
        if (!attrs.has(AttrKey::Dim)) {
            return get_vulkan_backend()->dispatchNansum(x);
        }
        int64_t dim = attrs.get_int(AttrKey::Dim, 0);
        bool keepdim = attrs.get_bool(AttrKey::Keepdim, false);
        Tensor mask = isnan(x);
        Tensor zero = zeros_like(x);
        Tensor cleaned = where(mask, zero, x);
        return tenzor::sum(cleaned, dim, keepdim);
    });

    table.register_single_output_kernel(OpId::Nanmean, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        const Tensor& x = inputs[0];
        if (!attrs.has(AttrKey::Dim)) {
            return get_vulkan_backend()->dispatchNanmean(x);
        }
        int64_t dim = attrs.get_int(AttrKey::Dim, 0);
        bool keepdim = attrs.get_bool(AttrKey::Keepdim, false);
        Tensor mask = isnan(x);
        Tensor zero = zeros_like(x);
        Tensor cleaned = where(mask, zero, x);
        Tensor numer = tenzor::sum(cleaned, dim, keepdim);
        // count_non_nan = sum(!mask) cast to input dtype
        Tensor count = tenzor::sum(logical_not(mask).to(x.dtype()), dim, keepdim);
        return tenzor::div(numer, count);
    });

    // Native Vulkan Aminmax (returns 2 tensors: min, max)
    table.register_kernel(OpId::Aminmax, [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
        auto [min_t, max_t] = get_vulkan_backend()->dispatchAminmax(inputs[0]);
        return {min_t, max_t};
    });
    // Native Vulkan ScatterReduce
    table.register_single_output_kernel(OpId::ScatterReduce, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t dim = attrs.get_int(AttrKey::Dim, 0);
        std::string reduce = std::string(attrs.get_string(AttrKey::Reduction, "sum"));
        bool include_self = attrs.get_bool(AttrKey::IncludeSelf, true);
        return get_vulkan_backend()->dispatchScatterReduce(inputs[0], dim, inputs[1], inputs[2], reduce, include_self);
    });
    // Native Vulkan IndexAdd
    table.register_single_output_kernel(OpId::IndexAdd, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t dim = attrs.get_int(AttrKey::Dim, 0);
        return get_vulkan_backend()->dispatchIndexAdd(inputs[0], dim, inputs[1], inputs[2]);
    });

    // Native Vulkan IndexCopy
    table.register_single_output_kernel(OpId::IndexCopy, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t dim = attrs.get_int(AttrKey::Dim, 0);
        return get_vulkan_backend()->dispatchIndexCopy(inputs[0], dim, inputs[1], inputs[2]);
    });

    // Native Vulkan IndexFill
    table.register_single_output_kernel(OpId::IndexFill, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t dim = attrs.get_int(AttrKey::Dim, 0);
        float value = static_cast<float>(attrs.get_float(AttrKey::Value, 0.0));
        return get_vulkan_backend()->dispatchIndexFill(inputs[0], dim, inputs[1], value);
    });

    // SelectScatter: clone input, write src into the single slice [index] along `dim`
    // Vulkan tensors live in device-local memory, so a host memcpy through
    // slice.data_ptr() targets a synthetic address and silently no-ops.
    // Express the scatter as IndexCopy with a 1-element index: that kernel
    // runs as a real compute dispatch.
    table.register_single_output_kernel(OpId::SelectScatter,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            const auto& input = inputs[0];
            const auto& src = inputs[1];
            int64_t dim = attrs.get_int(AttrKey::Dim, 0);
            int64_t index = attrs.get_int(AttrKey::Index, 0);

            int64_t ndim = static_cast<int64_t>(input.shape().size());
            if (dim < 0) dim += ndim;

            // Build a length-1 index tensor on the input's device
            auto idx_cpu = Tensor({int64_t(1)}, DType::Int64, Device::cpu());
            idx_cpu.data<int64_t>()[0] = index;
            auto idx = idx_cpu.to(input.device());

            // IndexCopy expects src's shape to match input's shape, except along
            // `dim` where it has length index.numel() (here, 1). Unsqueeze src
            // so the reduced axis exists.
            auto src_unsq = src.unsqueeze(dim).contiguous();
            return get_vulkan_backend()->dispatchIndexCopy(input, dim, idx, src_unsq);
        });

    // SliceScatter: clone input, write src into the strided range [start:end:step]
    // along `dim`. Same device-memory reasoning as SelectScatter — implemented
    // as IndexCopy with a computed range of indices so the copy runs on GPU.
    table.register_single_output_kernel(OpId::SliceScatter,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            const auto& input = inputs[0];
            const auto& src = inputs[1];
            int64_t dim = attrs.get_int(AttrKey::Dim, 0);
            int64_t start = attrs.get_int(AttrKey::Start, 0);
            int64_t end = attrs.get_int(AttrKey::End, -1);
            int64_t step = attrs.get_int(AttrKey::Step, 1);

            int64_t ndim = static_cast<int64_t>(input.shape().size());
            if (dim < 0) dim += ndim;
            int64_t dim_size = input.shape()[dim];

            if (start < 0) start += dim_size;
            if (end < 0) end += dim_size + 1;
            if (start < 0) start = 0;
            if (end > dim_size) end = dim_size;
            if (step <= 0) step = 1;

            int64_t n_idx = (end > start) ? ((end - start + step - 1) / step) : 0;
            if (n_idx <= 0) return input.clone();

            auto idx_cpu = Tensor({n_idx}, DType::Int64, Device::cpu());
            auto* idx_data = idx_cpu.data<int64_t>();
            for (int64_t i = 0; i < n_idx; ++i) idx_data[i] = start + i * step;
            auto idx = idx_cpu.to(input.device());

            // Ensure src has shape matching [*, n_idx, *] along `dim`
            auto src_contig = src.contiguous();
            return get_vulkan_backend()->dispatchIndexCopy(input, dim, idx, src_contig);
        });

    // DiagonalScatter: clone input, write src values along the (dim1, dim2)
    // diagonal with the given offset. Encoded as AdvancedIndexPut using
    // generated row/col indices — the underlying compute shader handles the
    // scatter. Batch dims are flattened, and indices broadcast across them.
    table.register_single_output_kernel(OpId::DiagonalScatter,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            const auto& input = inputs[0];
            const auto& src_in = inputs[1];
            int64_t offset = attrs.get_int(AttrKey::Diagonal, 0);
            int64_t dim1 = attrs.get_int(AttrKey::Dim1, 0);
            int64_t dim2 = attrs.get_int(AttrKey::Dim2, 1);

            int64_t ndim = static_cast<int64_t>(input.shape().size());
            if (dim1 < 0) dim1 += ndim;
            if (dim2 < 0) dim2 += ndim;

            auto shape = input.shape();
            int64_t size1 = shape[dim1];
            int64_t size2 = shape[dim2];

            int64_t diag_len;
            if (offset >= 0) diag_len = std::min(size1, size2 - offset);
            else              diag_len = std::min(size1 + offset, size2);
            if (diag_len <= 0) return input.clone();

            int64_t r0 = (offset >= 0) ? 0 : -offset;
            int64_t c0 = (offset >= 0) ? offset : 0;

            // Build one index tensor per input dim. Batch axes get an arange,
            // (dim1, dim2) get a 1-D vector of [r0..r0+diag_len) / [c0..c0+diag_len).
            // Indices broadcast: each is shaped {batch_0, ..., batch_{m-1}, diag_len}.
            std::vector<int64_t> batch_sizes;
            std::vector<int64_t> batch_dims;
            int64_t batch_rank = 0;
            for (int64_t d = 0; d < ndim; ++d) {
                if (d == dim1 || d == dim2) continue;
                batch_dims.push_back(d);
                batch_sizes.push_back(shape[d]);
                ++batch_rank;
            }

            // Shape of every index tensor: [batch..., diag_len]
            std::vector<int64_t> idx_shape = batch_sizes;
            idx_shape.push_back(diag_len);

            // Compute on CPU (these are small) and move to device.
            int64_t total_idx = 1;
            for (auto d : idx_shape) total_idx *= d;

            // Build batch index tensors with proper broadcasting
            auto* vk = get_vulkan_backend();
            std::vector<Tensor> indices(ndim);
            for (int64_t d = 0; d < ndim; ++d) {
                auto t_cpu = Tensor(idx_shape, DType::Int64, Device::cpu());
                auto* data = t_cpu.data<int64_t>();
                // Enumerate every position (b0,...,b_{m-1}, k)
                std::vector<int64_t> coord(idx_shape.size(), 0);
                for (int64_t i = 0; i < total_idx; ++i) {
                    int64_t k = coord.back();
                    int64_t value = 0;
                    if (d == dim1)      value = r0 + k;
                    else if (d == dim2) value = c0 + k;
                    else {
                        // Find batch position for axis d
                        for (size_t bd = 0; bd < batch_dims.size(); ++bd) {
                            if (batch_dims[bd] == d) { value = coord[bd]; break; }
                        }
                    }
                    data[i] = value;
                    // Increment coord in row-major order
                    for (int64_t ax = static_cast<int64_t>(idx_shape.size()) - 1; ax >= 0; --ax) {
                        coord[ax]++;
                        if (coord[ax] < idx_shape[ax]) break;
                        coord[ax] = 0;
                    }
                }
                indices[d] = t_cpu.to(input.device());
            }

            // src has shape [batch..., diag_len] — same as idx_shape already.
            auto src_broadcast = src_in;
            auto src_shape = src_broadcast.shape();
            bool shape_matches = (src_shape.size() == idx_shape.size());
            if (shape_matches) {
                for (size_t i = 0; i < idx_shape.size(); ++i) {
                    if (src_shape[i] != idx_shape[i]) { shape_matches = false; break; }
                }
            }
            if (!shape_matches) {
                src_broadcast = src_broadcast.reshape(idx_shape);
            }
            src_broadcast = src_broadcast.contiguous();

            // AdvancedIndexPut: input[indices[0], indices[1], ..., indices[ndim-1]] = src
            return vk->dispatchAdvancedIndexPut(input, indices, src_broadcast,
                                                 static_cast<int64_t>(indices.size()));
        });

    // Bitwise shift ops: native Vulkan dispatch via standalone int32 shaders.
    // Same Int32-promote pattern as the AND/OR/XOR registrations above; the
    // narrow-back uses dispatchCastTruncateInt32 to preserve the bit pattern
    // (cast_f32_iX would saturate). All compute stays on-device.
    table.register_kernel(OpId::BitwiseLeftShift, [](std::span<const Tensor> inputs, const OpAttributes&) {
        const Tensor& a = inputs[0];
        const Tensor& b = inputs[1];
        if (a.dtype() == DType::Int32) {
            return std::vector<Tensor>{get_vulkan_backend()->dispatchBitwiseBinaryOp("bitwise_left_shift", a, b)};
        }
        Tensor a32 = a.to(DType::Int32), b32 = b.to(DType::Int32);
        Tensor out32 = get_vulkan_backend()->dispatchBitwiseBinaryOp("bitwise_left_shift", a32, b32);
        return std::vector<Tensor>{get_vulkan_backend()->dispatchCastTruncateInt32(out32, a.dtype())};
    });
    table.register_kernel(OpId::BitwiseRightShift, [](std::span<const Tensor> inputs, const OpAttributes&) {
        const Tensor& a = inputs[0];
        const Tensor& b = inputs[1];
        if (a.dtype() == DType::Int32) {
            return std::vector<Tensor>{get_vulkan_backend()->dispatchBitwiseBinaryOp("bitwise_right_shift", a, b)};
        }
        Tensor a32 = a.to(DType::Int32), b32 = b.to(DType::Int32);
        Tensor out32 = get_vulkan_backend()->dispatchBitwiseBinaryOp("bitwise_right_shift", a32, b32);
        return std::vector<Tensor>{get_vulkan_backend()->dispatchCastTruncateInt32(out32, a.dtype())};
    });

    // =========================================================================
    // Fused GEMM Operations (composed from existing Vulkan ops)
    // =========================================================================
    table.register_single_output_kernel(OpId::Addmm,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            double alpha = attrs.get_float(AttrKey::Alpha, 1.0);
            double beta = attrs.get_float(AttrKey::Beta, 1.0);
            auto* vk = get_vulkan_backend();
            // Compose: beta * input + alpha * (mat1 @ mat2)
            auto mm = vk->dispatchMatmul(inputs[1], inputs[2]);
            if (alpha != 1.0) {
                auto alpha_t = tenzor::full({1}, alpha, mm.dtype(), mm.device());
                mm = vk->dispatchBinaryOp("mul", mm, alpha_t);
            }
            if (beta == 0.0) return mm;
            Tensor inp = inputs[0];
            if (beta != 1.0) {
                auto beta_t = tenzor::full({1}, beta, inp.dtype(), inp.device());
                inp = vk->dispatchBinaryOp("mul", inp, beta_t);
            }
            return vk->dispatchBinaryOp("add", inp, mm);
        });

    table.register_single_output_kernel(OpId::Addmv,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            double alpha = attrs.get_float(AttrKey::Alpha, 1.0);
            double beta = attrs.get_float(AttrKey::Beta, 1.0);
            auto* vk = get_vulkan_backend();
            // mat @ vec via matmul with reshape
            auto vec_col = inputs[2].reshape({inputs[2].shape()[0], 1});
            auto mv = vk->dispatchMatmul(inputs[1], vec_col);
            mv = mv.reshape({inputs[1].shape()[0]});
            if (alpha != 1.0) {
                auto alpha_t = tenzor::full({1}, alpha, mv.dtype(), mv.device());
                mv = vk->dispatchBinaryOp("mul", mv, alpha_t);
            }
            if (beta == 0.0) return mv;
            Tensor inp = inputs[0];
            if (beta != 1.0) {
                auto beta_t = tenzor::full({1}, beta, inp.dtype(), inp.device());
                inp = vk->dispatchBinaryOp("mul", inp, beta_t);
            }
            return vk->dispatchBinaryOp("add", inp, mv);
        });

    table.register_single_output_kernel(OpId::Baddbmm,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            double alpha = attrs.get_float(AttrKey::Alpha, 1.0);
            double beta = attrs.get_float(AttrKey::Beta, 1.0);
            auto* vk = get_vulkan_backend();
            auto bmm = vk->dispatchBmm(inputs[1], inputs[2]);
            if (alpha != 1.0) {
                auto alpha_t = tenzor::full({1}, alpha, bmm.dtype(), bmm.device());
                bmm = vk->dispatchBinaryOp("mul", bmm, alpha_t);
            }
            if (beta == 0.0) return bmm;
            Tensor inp = inputs[0];
            if (beta != 1.0) {
                auto beta_t = tenzor::full({1}, beta, inp.dtype(), inp.device());
                inp = vk->dispatchBinaryOp("mul", inp, beta_t);
            }
            return vk->dispatchBinaryOp("add", inp, bmm);
        });

    // =========================================================================
    // Log-Cumulative-Sum-Exp (native Vulkan shader)
    // =========================================================================
    table.register_single_output_kernel(OpId::Logcumsumexp, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t dim = attrs.get_int(AttrKey::Dim, 0);
        return get_vulkan_backend()->dispatchLogcumsumexp(inputs[0], dim);
    });

    // =========================================================================
    // Bincount (native Vulkan shader)
    // =========================================================================
    table.register_single_output_kernel(OpId::Bincount, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t minlength = attrs.get_int(AttrKey::Minlength, 0);
        std::optional<Tensor> weights;
        if (inputs.size() > 1) {
            weights = inputs[1];
        }
        return get_vulkan_backend()->dispatchBincount(inputs[0], weights, minlength);
    });

    // =========================================================================
    // New Reduction Operations (CumMax, CumMin, Fmax, Fmin, Isin, Kthvalue, etc.)
    // =========================================================================

    table.register_kernel(OpId::CumMax, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t dim = attrs.get_int(AttrKey::Dim, 0);
        auto [values, indices] = get_vulkan_backend()->dispatchCumMax(inputs[0], dim);
        return std::vector<Tensor>{values, indices};
    });

    table.register_kernel(OpId::CumMin, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t dim = attrs.get_int(AttrKey::Dim, 0);
        auto [values, indices] = get_vulkan_backend()->dispatchCumMin(inputs[0], dim);
        return std::vector<Tensor>{values, indices};
    });

    table.register_single_output_kernel(OpId::Fmax,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> Tensor {
            return get_vulkan_backend()->dispatchFmax(inputs[0], inputs[1]);
        });

    table.register_single_output_kernel(OpId::Fmin,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> Tensor {
            return get_vulkan_backend()->dispatchFmin(inputs[0], inputs[1]);
        });

    table.register_single_output_kernel(OpId::Isin,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> Tensor {
            return get_vulkan_backend()->dispatchIsin(inputs[0], inputs[1]);
        });

    table.register_kernel(OpId::Kthvalue, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t k = attrs.get_int(AttrKey::K, 1);
        int64_t dim = attrs.get_int(AttrKey::Dim, -1);
        bool keepdim = attrs.get_bool(AttrKey::Keepdim, false);
        auto [values, indices] = get_vulkan_backend()->dispatchKthvalue(inputs[0], k, dim, keepdim);
        return std::vector<Tensor>{values, indices};
    });

    table.register_single_output_kernel(OpId::Quantile,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            double q = attrs.get_float(AttrKey::Alpha, 0.5);
            int64_t dim = attrs.get_int(AttrKey::Dim, -1);
            bool keepdim = attrs.get_bool(AttrKey::Keepdim, false);
            return get_vulkan_backend()->dispatchQuantile(inputs[0], q, dim, keepdim);
        });

    table.register_single_output_kernel(OpId::Nanquantile,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            double q = attrs.get_float(AttrKey::Alpha, 0.5);
            int64_t dim = attrs.get_int(AttrKey::Dim, -1);
            bool keepdim = attrs.get_bool(AttrKey::Keepdim, false);
            return get_vulkan_backend()->dispatchNanquantile(inputs[0], q, dim, keepdim);
        });

    table.register_single_output_kernel(OpId::Nanmedian,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            int64_t dim = attrs.get_int(AttrKey::Dim, -1);
            return get_vulkan_backend()->dispatchNanmedian(inputs[0], dim);
        });

    table.register_single_output_kernel(OpId::Histc,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            int64_t bins = attrs.get_int(AttrKey::N, 100);
            double min_val = attrs.get_float(AttrKey::Alpha, 0.0);
            double max_val = attrs.get_float(AttrKey::Beta, 0.0);
            return get_vulkan_backend()->dispatchHistc(inputs[0], bins, min_val, max_val);
        });

    table.register_kernel(OpId::UniqueConsecutive, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        bool return_inverse = attrs.get_bool(AttrKey::Keepdim, false);
        auto [unique_vals, inverse, counts] = get_vulkan_backend()->dispatchUniqueConsecutive(
            inputs[0], return_inverse);
        return std::vector<Tensor>{unique_vals, inverse, counts};
    });

    table.register_single_output_kernel(OpId::SegmentReduce, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t axis = attrs.get_int(AttrKey::Dim, 0);
        std::string reduce = std::string(attrs.get_string(AttrKey::Reduction, "sum"));
        return get_vulkan_backend()->dispatchSegmentReduce(inputs[0], inputs[1], reduce, axis);
    });

    // =========================================================================
    // TakeAlongDim — dispatch as Gather (semantically equivalent for contiguous tensors)
    // =========================================================================
    table.register_single_output_kernel(OpId::TakeAlongDim, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t dim = attrs.get_int(AttrKey::Dim, 0);
        return get_vulkan_backend()->dispatchGather(inputs[0], dim, inputs[1]);
    });

    // =========================================================================
    // MaskedScatter — native Vulkan via CumSum prefix on mask + scatter shader
    // =========================================================================
    table.register_single_output_kernel(OpId::MaskedScatter, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        auto input = inputs[0];
        auto mask = inputs[1];
        auto source = inputs[2];

        // The masked_scatter.comp shader indexes prefix_data by the flat
        // global invocation id, so the prefix sum has to be FLAT — i.e. a
        // cumulative count of mask trues across the entire flattened mask.
        // Earlier this called cumsum(mask, 0) on the 2-D mask directly,
        // which produced per-column cumulative sums and yielded incorrect
        // source indices for any mask whose first axis had more than one
        // element. Flatten before cumsum, then the shader-facing prefix
        // is correct regardless of the original mask's rank.
        Tensor mask_f32_flat = mask.to(DType::Float32).reshape({-1}).contiguous();
        Tensor prefix_flat = tenzor::cumsum(mask_f32_flat, 0);
        Tensor prefix_i32 = prefix_flat.to(DType::Int32);

        auto* backend = get_vulkan_backend();
        return backend->dispatchMaskedScatterWithPrefix(input, mask, source, prefix_i32);
    });

    // =========================================================================
    // TrilIndices — native Vulkan compute shader (tril_indices.comp)
    // =========================================================================
    table.register_single_output_kernel(OpId::TrilIndices, [](std::span<const Tensor> /*inputs*/, const OpAttributes& attrs) -> Tensor {
        int64_t row = attrs.get_int(AttrKey::M, 0);
        int64_t col = attrs.get_int(AttrKey::N, 0);
        int64_t offset = attrs.get_int(AttrKey::Diagonal, 0);
        return get_vulkan_backend()->dispatchTrilIndices(row, col, offset);
    });

    // =========================================================================
    // TriuIndices — native Vulkan compute shader (triu_indices.comp)
    // =========================================================================
    table.register_single_output_kernel(OpId::TriuIndices, [](std::span<const Tensor> /*inputs*/, const OpAttributes& attrs) -> Tensor {
        int64_t row = attrs.get_int(AttrKey::M, 0);
        int64_t col = attrs.get_int(AttrKey::N, 0);
        int64_t offset = attrs.get_int(AttrKey::Diagonal, 0);
        return get_vulkan_backend()->dispatchTriuIndices(row, col, offset);
    });

    // =========================================================================
    // Phase 9: Fractional Max Pool + Max Unpool — native Vulkan compute shaders
    // Shaders: fractional_maxpool2d.comp, fractional_maxpool2d_backward.comp,
    //          fractional_maxpool3d.comp, fractional_maxpool3d_backward.comp,
    //          max_unpool2d.comp, max_unpool2d_backward.comp,
    //          max_unpool3d.comp, max_unpool3d_backward.comp
    // Dispatch is delegated to VulkanBackend dispatch methods.
    // =========================================================================

    table.register_kernel(OpId::FractionalMaxPool2dForward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t out_h = attrs.get_int(AttrKey::OutputSizeH, 1);
        int64_t out_w = attrs.get_int(AttrKey::OutputSizeW, 1);
        const Tensor* samples = (inputs.size() > 1) ? &inputs[1] : nullptr;
        auto [output, indices] = get_vulkan_backend()->dispatchFractionalMaxPool2dForward(
            inputs[0], out_h, out_w, samples);
        return std::vector<Tensor>{output, indices};
    });

    table.register_single_output_kernel(OpId::FractionalMaxPool2dBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        auto input_shape = attrs.get_int_list(AttrKey::InputShape);
        return get_vulkan_backend()->dispatchFractionalMaxPool2dBackward(inputs[0], inputs[1], input_shape);
    });

    table.register_kernel(OpId::FractionalMaxPool3dForward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t out_d = attrs.get_int(AttrKey::OutputSizeD, 1);
        int64_t out_h = attrs.get_int(AttrKey::OutputSizeH, 1);
        int64_t out_w = attrs.get_int(AttrKey::OutputSizeW, 1);
        const Tensor* samples = (inputs.size() > 1) ? &inputs[1] : nullptr;
        auto [output, indices] = get_vulkan_backend()->dispatchFractionalMaxPool3dForward(
            inputs[0], out_d, out_h, out_w, samples);
        return std::vector<Tensor>{output, indices};
    });

    table.register_single_output_kernel(OpId::FractionalMaxPool3dBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        auto input_shape = attrs.get_int_list(AttrKey::InputShape);
        return get_vulkan_backend()->dispatchFractionalMaxPool3dBackward(inputs[0], inputs[1], input_shape);
    });

    table.register_single_output_kernel(OpId::MaxUnpool2dForward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t out_h = attrs.get_int(AttrKey::OutputSizeH, 1);
        int64_t out_w = attrs.get_int(AttrKey::OutputSizeW, 1);
        return get_vulkan_backend()->dispatchMaxUnpool2dForward(inputs[0], inputs[1], out_h, out_w);
    });

    table.register_single_output_kernel(OpId::MaxUnpool2dBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        auto input_shape = attrs.get_int_list(AttrKey::InputShape);
        return get_vulkan_backend()->dispatchMaxUnpool2dBackward(inputs[0], inputs[1], input_shape);
    });

    table.register_single_output_kernel(OpId::MaxUnpool3dForward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t out_d = attrs.get_int(AttrKey::OutputSizeD, 1);
        int64_t out_h = attrs.get_int(AttrKey::OutputSizeH, 1);
        int64_t out_w = attrs.get_int(AttrKey::OutputSizeW, 1);
        return get_vulkan_backend()->dispatchMaxUnpool3dForward(inputs[0], inputs[1], out_d, out_h, out_w);
    });

    table.register_single_output_kernel(OpId::MaxUnpool3dBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        auto input_shape = attrs.get_int_list(AttrKey::InputShape);
        return get_vulkan_backend()->dispatchMaxUnpool3dBackward(inputs[0], inputs[1], input_shape);
    });

    // =========================================================================
    // Phase A.1: Max Unpool 1D (Vulkan — wraps the 2D dispatcher via reshape).
    // =========================================================================
    table.register_single_output_kernel(OpId::MaxUnpool1dForward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t out_l = attrs.get_int(AttrKey::OutputSizeW, 1);
        auto in_shape = inputs[0].shape();
        int64_t N = in_shape[0], C = in_shape[1], in_l = in_shape[2];
        auto input_4d = inputs[0].contiguous().reshape({N, C, in_l, 1});
        auto indices_4d = inputs[1].contiguous().reshape({N, C, in_l, 1});
        auto out_4d = get_vulkan_backend()->dispatchMaxUnpool2dForward(input_4d, indices_4d, out_l, /*out_w=*/1);
        return out_4d.reshape({N, C, out_l});
    });

    table.register_single_output_kernel(OpId::MaxUnpool1dBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        auto input_shape = attrs.get_int_list(AttrKey::InputShape);
        int64_t N = input_shape[0], C = input_shape[1], in_l = input_shape[2];
        int64_t out_l = inputs[0].shape()[2];
        auto grad_4d = inputs[0].contiguous().reshape({N, C, out_l, 1});
        auto indices_4d = inputs[1].contiguous().reshape({N, C, in_l, 1});
        std::vector<int64_t> input_shape_4d = {N, C, in_l, 1};
        auto grad_in_4d = get_vulkan_backend()->dispatchMaxUnpool2dBackward(grad_4d, indices_4d, input_shape_4d);
        return grad_in_4d.reshape({N, C, in_l});
    });

    // ========================================================================
    // Phase: New Unary Math Operations — Deg2Rad, Rad2Deg, Logit
    // ========================================================================
    table.register_kernel(OpId::Deg2Rad, [](std::span<const Tensor> inputs, const OpAttributes&) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchUnaryOp("deg2rad", inputs[0])};
    });

    table.register_kernel(OpId::Rad2Deg, [](std::span<const Tensor> inputs, const OpAttributes&) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchUnaryOp("rad2deg", inputs[0])};
    });

    table.register_kernel(OpId::Logit, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        double eps = attrs.get_float(AttrKey::Eps, -1.0);
        auto* vk = get_vulkan_backend();
        if (eps > 0.0) {
            return std::vector<Tensor>{vk->dispatchUnaryOpWithParam("logit", inputs[0], static_cast<float>(eps))};
        }
        return std::vector<Tensor>{vk->dispatchUnaryOp("logit", inputs[0])};
    });

    // ========================================================================
    // Phase: Bool Predicate Operations — Signbit, IsPosInf, IsNegInf, IsReal
    // ========================================================================
    table.register_kernel(OpId::Signbit, [](std::span<const Tensor> inputs, const OpAttributes&) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchBoolPredicateOp("signbit", inputs[0])};
    });

    table.register_kernel(OpId::IsPosInf, [](std::span<const Tensor> inputs, const OpAttributes&) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchBoolPredicateOp("isposinf", inputs[0])};
    });

    table.register_kernel(OpId::IsNegInf, [](std::span<const Tensor> inputs, const OpAttributes&) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchBoolPredicateOp("isneginf", inputs[0])};
    });

    table.register_single_output_kernel(OpId::IsReal, [](std::span<const Tensor> inputs, const OpAttributes&) -> Tensor {
        // IsReal is a pure metadata check: real (non-complex) dtypes return all-true
        const auto& input = inputs[0];
        auto input_shape = input.shape();
        std::vector<int64_t> shape(input_shape.begin(), input_shape.end());
        bool is_real = (input.dtype() != DType::Complex64 && input.dtype() != DType::Complex128);
        // Return a Bool tensor filled with the result
        auto* vk = get_vulkan_backend();
        Tensor result = vk->dispatchFull(shape, is_real ? 1.0f : 0.0f, DType::Bool);
        return result;
    });

    // ========================================================================
    // Phase: Binary Math Operations — FloatPower, Xlog1py, Ldexp
    // ========================================================================
    table.register_kernel(OpId::FloatPower, [](std::span<const Tensor> inputs, const OpAttributes&) {
        // FloatPower promotes to Float64 for precision; Vulkan f64 shader handles it
        auto* vk = get_vulkan_backend();
        Tensor a = inputs[0];
        Tensor b = inputs[1];
        // Promote to Float64 if available, otherwise use Float32
        if (a.dtype() != DType::Float64) a = a.to(DType::Float64);
        if (b.dtype() != DType::Float64) b = b.to(DType::Float64);
        return std::vector<Tensor>{vk->dispatchBinaryOp("float_power", a, b)};
    });

    table.register_kernel(OpId::Xlog1py, [](std::span<const Tensor> inputs, const OpAttributes&) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchBinaryOp("xlog1py", inputs[0], inputs[1])};
    });

    table.register_kernel(OpId::Ldexp, [](std::span<const Tensor> inputs, const OpAttributes&) {
        return std::vector<Tensor>{get_vulkan_backend()->dispatchBinaryOp("ldexp", inputs[0], inputs[1])};
    });

    // ========================================================================
    // Phase: Two-output — Frexp
    // ========================================================================
    table.register_kernel(OpId::Frexp, [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
        auto [mantissa, exponent] = get_vulkan_backend()->dispatchFrexp(inputs[0]);
        return {mantissa, exponent};
    });

    // ========================================================================
    // Phase: Tensor manipulation — DiagEmbed, Diagflat
    // ========================================================================
    table.register_single_output_kernel(OpId::DiagEmbed, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t offset = attrs.get_int(AttrKey::Diagonal, 0);
        int64_t dim1 = attrs.get_int(AttrKey::Dim0, -2);
        int64_t dim2 = attrs.get_int(AttrKey::Dim1, -1);
        return get_vulkan_backend()->dispatchDiagEmbed(inputs[0], offset, dim1, dim2);
    });

    table.register_single_output_kernel(OpId::Diagflat, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t offset = attrs.get_int(AttrKey::Diagonal, 0);
        return get_vulkan_backend()->dispatchDiagflat(inputs[0], offset);
    });

    // ========================================================================
    // Phase: NaN-aware reductions — NanVar, NanStd
    // ========================================================================
    table.register_single_output_kernel(OpId::NanVar, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t correction = attrs.get_int(AttrKey::Correction, 1);
        return get_vulkan_backend()->dispatchNanVar(inputs[0], correction);
    });

    table.register_single_output_kernel(OpId::NanStd, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t correction = attrs.get_int(AttrKey::Correction, 1);
        return get_vulkan_backend()->dispatchNanStd(inputs[0], correction);
    });

    // =========================================================================
    // Nested Tensor Operations (fallback: unbind segments, apply regular ops)
    // =========================================================================
    table.register_single_output_kernel(OpId::NestedSoftmax, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t dim = attrs.get_int(AttrKey::Dim, -1);
        return get_vulkan_backend()->dispatchNestedSoftmax(inputs[0], inputs[1], dim);
    });

    table.register_single_output_kernel(OpId::NestedLogSoftmax, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t dim = attrs.get_int(AttrKey::Dim, -1);
        return get_vulkan_backend()->dispatchNestedLogSoftmax(inputs[0], inputs[1], dim);
    });

    table.register_single_output_kernel(OpId::NestedSum, [](std::span<const Tensor> inputs, const OpAttributes&) -> Tensor {
        return get_vulkan_backend()->dispatchNestedSum(inputs[0], inputs[1]);
    });

    table.register_single_output_kernel(OpId::NestedMean, [](std::span<const Tensor> inputs, const OpAttributes&) -> Tensor {
        return get_vulkan_backend()->dispatchNestedMean(inputs[0], inputs[1]);
    });

    table.register_single_output_kernel(OpId::NestedLayerNorm, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        // Fallback: per-row layer norm is the same as regular layer norm on values
        // since LN operates on the last dimension. Just dispatch to the regular LN.
        auto& values = inputs[0];
        const Tensor* weight_ptr = (inputs.size() > 2) ? &inputs[2] : nullptr;
        const Tensor* bias_ptr = (inputs.size() > 3) ? &inputs[3] : nullptr;
        float eps = static_cast<float>(attrs.get_float(AttrKey::Eps, 1e-5));
        int64_t D = values.shape().back();
        // Apply regular layer norm (operates on last dim, same for all rows)
        auto vk = get_vulkan_backend();
        auto [out, /*mean*/_m, /*rstd*/_r] =
            vk->dispatchLayerNorm(values, D, weight_ptr, bias_ptr, eps);
        (void)_m; (void)_r;
        return out;
    });

    table.register_single_output_kernel(OpId::NestedLinear, [](std::span<const Tensor> inputs, const OpAttributes&) -> Tensor {
        auto result = tenzor::matmul(inputs[0], inputs[2].transpose(0, 1));
        if (inputs.size() > 3) {
            result = tenzor::add(result, inputs[3]);
        }
        return result;
    });

    // NestedAttention — fused Vulkan compute shader reading offsets on device
    table.register_single_output_kernel(OpId::NestedAttention,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            float scale = static_cast<float>(attrs.get_float(AttrKey::Scale, 1.0));
            bool causal = attrs.get_bool(AttrKey::Causal, false);
            return get_vulkan_backend()->dispatchNestedAttention(
                inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], scale, causal);
        });

    table.register_single_output_kernel(OpId::NestedToPadded, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t max_len = attrs.get_int(AttrKey::MaxLen, 0);
        float padding_value = attrs.get_float(AttrKey::PaddingValue, 0.0f);
        return get_vulkan_backend()->dispatchNestedToPadded(inputs[0], inputs[1], max_len, padding_value);
    });

    table.register_single_output_kernel(OpId::NestedFromPadded, [](std::span<const Tensor> inputs, const OpAttributes&) -> Tensor {
        return get_vulkan_backend()->dispatchNestedFromPadded(inputs[0], inputs[1]);
    });

    // =========================================================================
    // AsStrided — metadata-only view with custom shape/strides
    // =========================================================================
    table.register_single_output_kernel(OpId::AsStrided,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            auto shape = attrs.get_int_list(AttrKey::Shape);
            auto strides = attrs.get_int_list(AttrKey::Strides);
            int64_t offset = attrs.get_int(AttrKey::StorageOffset, -1);
            std::optional<int64_t> storage_offset = (offset >= 0) ? std::optional(offset) : std::nullopt;
            return tenzor::as_strided(inputs[0], shape, strides, storage_offset);
        });

    // =========================================================================
    // NestedAttentionBackward — backward for segmented attention
    // Vulkan: dispatch to GPU compute shader (nested_attention_backward.comp)
    // Three-pass algorithm: max-score, softmax+dot, gradient accumulation.
    // inputs[4] (attn_out) is unused — weights are recomputed in the shader.
    // =========================================================================
    table.register_kernel(OpId::NestedAttentionBackward,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            float scale = static_cast<float>(attrs.get_float(AttrKey::Scale, 1.0));
            bool causal = attrs.get_bool(AttrKey::Causal, false);
            return get_vulkan_backend()->dispatchNestedAttentionBackward(
                inputs[0], inputs[1], inputs[2], inputs[3],
                inputs[5], inputs[6], scale, causal);
        });

    // =========================================================================
    // Statistical operations (Cov, Corrcoef) — composed from existing ops
    // =========================================================================
    table.register_single_output_kernel(OpId::Cov, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t correction = attrs.get_int(AttrKey::Correction, 1);
        return cov(inputs[0], correction);
    });

    table.register_single_output_kernel(OpId::Corrcoef, [](std::span<const Tensor> inputs, const OpAttributes&) -> Tensor {
        return corrcoef(inputs[0]);
    });

    // =========================================================================
    // LOBPCG — Locally Optimal Block Preconditioned Conjugate Gradient
    // =========================================================================
    table.register_kernel(OpId::LOBPCG,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            int64_t k = attrs.get_int(AttrKey::K, 1);
            int64_t max_iter = attrs.get_int(AttrKey::MaxIter, 100);
            double tol = attrs.get_float(AttrKey::Tolerance, 1e-6);
            Tensor B = inputs.size() > 2 ? inputs[2] : Tensor();
            auto [evals, evecs] = linalg::lobpcg(inputs[0], inputs[1], k, B, max_iter, tol);
            return {evals, evecs};
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
