/**
 * @file webgpu_kernel_registry.cpp
 * @brief WebGPU kernel registration for O(1) dispatch
 *
 * Registers WebGPU kernel implementations with the dispatch table.
 * Each kernel is a stub that delegates to the WebGPU compute pipeline
 * infrastructure, dispatching the appropriate WGSL shader entry point.
 *
 * The WebGPU backend is experimental; these stubs validate dispatch
 * integration and provide the framework for full compute shader execution
 * once the backend initialization is complete (see webgpu_backend.cpp).
 */

#include "tenzor/backend/dispatch_table.hpp"
#include "tenzor/backend/kernel_registry.hpp"
#include "tenzor/backend/op_attributes.hpp"
#include "tenzor/ops/op_id.hpp"
#include "tenzor/core/tensor.hpp"

#include <stdexcept>
#include <string>
#include <cstdint>
#include <climits>

namespace tenzor {

namespace {

/**
 * @brief Throw a "not yet implemented" error with the operation name.
 *
 * WebGPU kernels are registered so that dispatch routing works correctly,
 * but the actual compute shader execution requires an initialized WebGPU
 * device. Until the backend's initialize() no longer throws, all kernels
 * produce a descriptive error.
 */
[[noreturn]] void throw_webgpu_not_ready(const char* op_name) {
    throw std::runtime_error(
        std::string("WebGPU backend: operation '") + op_name +
        "' is registered but the WebGPU device is not initialized. "
        "The WebGPU backend is experimental.");
}

// ---------------------------------------------------------------------------
// Arithmetic stubs — WGSL entry points: math.wgsl (add, sub, mul, div)
//                                        matmul.wgsl (main, batchMatmul)
// ---------------------------------------------------------------------------

Tensor add_kernel(std::span<const Tensor> inputs, const OpAttributes&) {
    (void)inputs;
    throw_webgpu_not_ready("Add");
}

Tensor sub_kernel(std::span<const Tensor> inputs, const OpAttributes&) {
    (void)inputs;
    throw_webgpu_not_ready("Sub");
}

Tensor mul_kernel(std::span<const Tensor> inputs, const OpAttributes&) {
    (void)inputs;
    throw_webgpu_not_ready("Mul");
}

Tensor div_kernel(std::span<const Tensor> inputs, const OpAttributes&) {
    (void)inputs;
    throw_webgpu_not_ready("Div");
}

Tensor matmul_kernel(std::span<const Tensor> inputs, const OpAttributes&) {
    (void)inputs;
    throw_webgpu_not_ready("MatMul");
}

Tensor bmm_kernel(std::span<const Tensor> inputs, const OpAttributes&) {
    (void)inputs;
    throw_webgpu_not_ready("Bmm");
}

Tensor dot_kernel(std::span<const Tensor> inputs, const OpAttributes&) {
    (void)inputs;
    throw_webgpu_not_ready("Dot");
}

// ---------------------------------------------------------------------------
// Activation stubs — WGSL entry points: activations.wgsl
//   relu, sigmoid, tanhActivation, gelu, softmaxMax/ExpSum/Normalize,
//   leakyRelu, elu, swish, mish, softplus, logSoftmaxNormalize
// ---------------------------------------------------------------------------

Tensor relu_kernel(std::span<const Tensor> inputs, const OpAttributes&) {
    (void)inputs;
    throw_webgpu_not_ready("ReLU");
}

Tensor sigmoid_kernel(std::span<const Tensor> inputs, const OpAttributes&) {
    (void)inputs;
    throw_webgpu_not_ready("Sigmoid");
}

Tensor tanh_activation_kernel(std::span<const Tensor> inputs, const OpAttributes&) {
    (void)inputs;
    throw_webgpu_not_ready("TanhActivation");
}

Tensor gelu_kernel(std::span<const Tensor> inputs, const OpAttributes&) {
    (void)inputs;
    throw_webgpu_not_ready("Gelu");
}

Tensor swish_kernel(std::span<const Tensor> inputs, const OpAttributes&) {
    (void)inputs;
    throw_webgpu_not_ready("Swish");
}

Tensor leaky_relu_kernel(std::span<const Tensor> inputs, const OpAttributes&) {
    (void)inputs;
    throw_webgpu_not_ready("LeakyReLU");
}

Tensor elu_kernel(std::span<const Tensor> inputs, const OpAttributes&) {
    (void)inputs;
    throw_webgpu_not_ready("Elu");
}

Tensor selu_kernel(std::span<const Tensor> inputs, const OpAttributes&) {
    (void)inputs;
    throw_webgpu_not_ready("Selu");
}

Tensor mish_kernel(std::span<const Tensor> inputs, const OpAttributes&) {
    (void)inputs;
    throw_webgpu_not_ready("Mish");
}

Tensor softplus_kernel(std::span<const Tensor> inputs, const OpAttributes&) {
    (void)inputs;
    throw_webgpu_not_ready("Softplus");
}

std::vector<Tensor> softmax_kernel(std::span<const Tensor> inputs, const OpAttributes&) {
    (void)inputs;
    throw_webgpu_not_ready("Softmax");
}

std::vector<Tensor> log_softmax_kernel(std::span<const Tensor> inputs, const OpAttributes&) {
    (void)inputs;
    throw_webgpu_not_ready("LogSoftmax");
}

// ---------------------------------------------------------------------------
// Reduction stubs — WGSL entry points: reduction.wgsl
//   reduceSum, reduceMean, reduceMax, reduceMin, reduceProd,
//   argMax, argMin, reduceVariance, reduceStd, reduceNorm
// ---------------------------------------------------------------------------

std::vector<Tensor> sum_kernel(std::span<const Tensor> inputs, const OpAttributes& attrs) {
    (void)inputs; (void)attrs;
    throw_webgpu_not_ready("Sum");
}

std::vector<Tensor> mean_kernel(std::span<const Tensor> inputs, const OpAttributes& attrs) {
    (void)inputs; (void)attrs;
    throw_webgpu_not_ready("Mean");
}

std::vector<Tensor> max_kernel(std::span<const Tensor> inputs, const OpAttributes& attrs) {
    (void)inputs; (void)attrs;
    throw_webgpu_not_ready("Max");
}

std::vector<Tensor> min_kernel(std::span<const Tensor> inputs, const OpAttributes& attrs) {
    (void)inputs; (void)attrs;
    throw_webgpu_not_ready("Min");
}

std::vector<Tensor> argmax_kernel(std::span<const Tensor> inputs, const OpAttributes& attrs) {
    (void)inputs; (void)attrs;
    throw_webgpu_not_ready("ArgMax");
}

std::vector<Tensor> argmin_kernel(std::span<const Tensor> inputs, const OpAttributes& attrs) {
    (void)inputs; (void)attrs;
    throw_webgpu_not_ready("ArgMin");
}

std::vector<Tensor> prod_kernel(std::span<const Tensor> inputs, const OpAttributes& attrs) {
    (void)inputs; (void)attrs;
    throw_webgpu_not_ready("Prod");
}

std::vector<Tensor> var_kernel(std::span<const Tensor> inputs, const OpAttributes& attrs) {
    (void)inputs; (void)attrs;
    throw_webgpu_not_ready("Var");
}

std::vector<Tensor> std_kernel(std::span<const Tensor> inputs, const OpAttributes& attrs) {
    (void)inputs; (void)attrs;
    throw_webgpu_not_ready("Std");
}

std::vector<Tensor> norm_kernel(std::span<const Tensor> inputs, const OpAttributes& attrs) {
    (void)inputs; (void)attrs;
    throw_webgpu_not_ready("Norm");
}

// ---------------------------------------------------------------------------
// Element-wise math stubs — WGSL entry points: math.wgsl
//   sqrt, neg, abs, sign, exp, log, pow, ceil, floor, round, reciprocal
// ---------------------------------------------------------------------------

Tensor sqrt_kernel(std::span<const Tensor> inputs, const OpAttributes&) {
    (void)inputs;
    throw_webgpu_not_ready("Sqrt");
}

Tensor neg_kernel(std::span<const Tensor> inputs, const OpAttributes&) {
    (void)inputs;
    throw_webgpu_not_ready("Neg");
}

Tensor abs_kernel(std::span<const Tensor> inputs, const OpAttributes&) {
    (void)inputs;
    throw_webgpu_not_ready("Abs");
}

Tensor sign_kernel(std::span<const Tensor> inputs, const OpAttributes&) {
    (void)inputs;
    throw_webgpu_not_ready("Sign");
}

Tensor log_kernel(std::span<const Tensor> inputs, const OpAttributes&) {
    (void)inputs;
    throw_webgpu_not_ready("Log");
}

Tensor exp_kernel(std::span<const Tensor> inputs, const OpAttributes&) {
    (void)inputs;
    throw_webgpu_not_ready("Exp");
}

Tensor pow_kernel(std::span<const Tensor> inputs, const OpAttributes&) {
    (void)inputs;
    throw_webgpu_not_ready("Pow");
}

Tensor clamp_kernel(std::span<const Tensor> inputs, const OpAttributes&) {
    (void)inputs;
    throw_webgpu_not_ready("Clamp");
}

Tensor reciprocal_kernel(std::span<const Tensor> inputs, const OpAttributes&) {
    (void)inputs;
    throw_webgpu_not_ready("Reciprocal");
}

Tensor floor_kernel(std::span<const Tensor> inputs, const OpAttributes&) {
    (void)inputs;
    throw_webgpu_not_ready("Floor");
}

Tensor ceil_kernel(std::span<const Tensor> inputs, const OpAttributes&) {
    (void)inputs;
    throw_webgpu_not_ready("Ceil");
}

Tensor round_kernel(std::span<const Tensor> inputs, const OpAttributes&) {
    (void)inputs;
    throw_webgpu_not_ready("Round");
}

// ---------------------------------------------------------------------------
// Trigonometric stubs — WGSL entry points: math.wgsl
//   sin, cos, tan, asin, acos, atan
// ---------------------------------------------------------------------------

Tensor sin_kernel(std::span<const Tensor> inputs, const OpAttributes&) {
    (void)inputs;
    throw_webgpu_not_ready("Sin");
}

Tensor cos_kernel(std::span<const Tensor> inputs, const OpAttributes&) {
    (void)inputs;
    throw_webgpu_not_ready("Cos");
}

Tensor tan_kernel(std::span<const Tensor> inputs, const OpAttributes&) {
    (void)inputs;
    throw_webgpu_not_ready("Tan");
}

Tensor asin_kernel(std::span<const Tensor> inputs, const OpAttributes&) {
    (void)inputs;
    throw_webgpu_not_ready("Asin");
}

Tensor acos_kernel(std::span<const Tensor> inputs, const OpAttributes&) {
    (void)inputs;
    throw_webgpu_not_ready("Acos");
}

Tensor atan_kernel(std::span<const Tensor> inputs, const OpAttributes&) {
    (void)inputs;
    throw_webgpu_not_ready("Atan");
}

// ---------------------------------------------------------------------------
// Comparison stubs — WGSL entry points: math.wgsl
//   eq, ne, lt, le, gt, ge
// ---------------------------------------------------------------------------

Tensor eq_kernel(std::span<const Tensor> inputs, const OpAttributes&) {
    (void)inputs;
    throw_webgpu_not_ready("Eq");
}

Tensor ne_kernel(std::span<const Tensor> inputs, const OpAttributes&) {
    (void)inputs;
    throw_webgpu_not_ready("Ne");
}

Tensor lt_kernel(std::span<const Tensor> inputs, const OpAttributes&) {
    (void)inputs;
    throw_webgpu_not_ready("Lt");
}

Tensor le_kernel(std::span<const Tensor> inputs, const OpAttributes&) {
    (void)inputs;
    throw_webgpu_not_ready("Le");
}

Tensor gt_kernel(std::span<const Tensor> inputs, const OpAttributes&) {
    (void)inputs;
    throw_webgpu_not_ready("Gt");
}

Tensor ge_kernel(std::span<const Tensor> inputs, const OpAttributes&) {
    (void)inputs;
    throw_webgpu_not_ready("Ge");
}

// ---------------------------------------------------------------------------
// Shape / View stubs — WGSL entry points: transform.wgsl
//   reshape, transpose, transpose2d, flatten, squeeze, unsqueeze, slice,
//   concat (Cat), stack, split, tile, flip
// Also: math.wgsl: fill, copy (Clone/Contiguous)
// ---------------------------------------------------------------------------

Tensor reshape_kernel(std::span<const Tensor> inputs, const OpAttributes&) {
    (void)inputs;
    throw_webgpu_not_ready("Reshape");
}

Tensor transpose_kernel(std::span<const Tensor> inputs, const OpAttributes&) {
    (void)inputs;
    throw_webgpu_not_ready("Transpose");
}

Tensor permute_kernel(std::span<const Tensor> inputs, const OpAttributes&) {
    (void)inputs;
    throw_webgpu_not_ready("Permute");
}

Tensor squeeze_kernel(std::span<const Tensor> inputs, const OpAttributes&) {
    (void)inputs;
    throw_webgpu_not_ready("Squeeze");
}

Tensor unsqueeze_kernel(std::span<const Tensor> inputs, const OpAttributes&) {
    (void)inputs;
    throw_webgpu_not_ready("Unsqueeze");
}

Tensor flatten_kernel(std::span<const Tensor> inputs, const OpAttributes&) {
    (void)inputs;
    throw_webgpu_not_ready("Flatten");
}

Tensor contiguous_kernel(std::span<const Tensor> inputs, const OpAttributes&) {
    (void)inputs;
    throw_webgpu_not_ready("Contiguous");
}

Tensor clone_kernel(std::span<const Tensor> inputs, const OpAttributes&) {
    (void)inputs;
    throw_webgpu_not_ready("Clone");
}

Tensor fill_kernel(std::span<const Tensor> inputs, const OpAttributes&) {
    (void)inputs;
    throw_webgpu_not_ready("Fill");
}

Tensor repeat_kernel(std::span<const Tensor> inputs, const OpAttributes&) {
    (void)inputs;
    throw_webgpu_not_ready("Repeat");
}

Tensor tile_kernel(std::span<const Tensor> inputs, const OpAttributes&) {
    (void)inputs;
    throw_webgpu_not_ready("Tile");
}

Tensor expand_kernel(std::span<const Tensor> inputs, const OpAttributes&) {
    (void)inputs;
    throw_webgpu_not_ready("Expand");
}

std::vector<Tensor> stack_kernel(std::span<const Tensor> inputs, const OpAttributes&) {
    (void)inputs;
    throw_webgpu_not_ready("Stack");
}

std::vector<Tensor> split_kernel(std::span<const Tensor> inputs, const OpAttributes&) {
    (void)inputs;
    throw_webgpu_not_ready("Split");
}

// ---------------------------------------------------------------------------
// Indexing stubs — WGSL entry points: indexing.wgsl
//   indexSelect, gather, scatter, maskedFill, maskedSelect, take, put,
//   embedding, oneHot, booleanMask
// Also: math.wgsl: where
// Also: transform.wgsl: slice, concat
// ---------------------------------------------------------------------------

Tensor index_select_kernel(std::span<const Tensor> inputs, const OpAttributes&) {
    (void)inputs;
    throw_webgpu_not_ready("IndexSelect");
}

Tensor gather_kernel(std::span<const Tensor> inputs, const OpAttributes&) {
    (void)inputs;
    throw_webgpu_not_ready("Gather");
}

Tensor scatter_kernel(std::span<const Tensor> inputs, const OpAttributes&) {
    (void)inputs;
    throw_webgpu_not_ready("Scatter");
}

Tensor masked_select_kernel(std::span<const Tensor> inputs, const OpAttributes&) {
    (void)inputs;
    throw_webgpu_not_ready("MaskedSelect");
}

Tensor masked_fill_kernel(std::span<const Tensor> inputs, const OpAttributes&) {
    (void)inputs;
    throw_webgpu_not_ready("MaskedFill");
}

Tensor where_kernel(std::span<const Tensor> inputs, const OpAttributes&) {
    (void)inputs;
    throw_webgpu_not_ready("Where");
}

Tensor slice_kernel(std::span<const Tensor> inputs, const OpAttributes&) {
    (void)inputs;
    throw_webgpu_not_ready("Slice");
}

std::vector<Tensor> cat_kernel(std::span<const Tensor> inputs, const OpAttributes&) {
    (void)inputs;
    throw_webgpu_not_ready("Cat");
}

Tensor take_kernel(std::span<const Tensor> inputs, const OpAttributes&) {
    (void)inputs;
    throw_webgpu_not_ready("Take");
}

Tensor put_kernel(std::span<const Tensor> inputs, const OpAttributes&) {
    (void)inputs;
    throw_webgpu_not_ready("Put");
}

Tensor one_hot_kernel(std::span<const Tensor> inputs, const OpAttributes&) {
    (void)inputs;
    throw_webgpu_not_ready("OneHot");
}

// ---------------------------------------------------------------------------
// Creation stubs — No WGSL shaders needed (host-side allocation + fill)
//   Uses math.wgsl: fill for Zeros/Ones/Full
//   Uses host random generators for Rand/Randn
// ---------------------------------------------------------------------------

Tensor zeros_kernel(std::span<const Tensor> inputs, const OpAttributes&) {
    (void)inputs;
    throw_webgpu_not_ready("Zeros");
}

Tensor ones_kernel(std::span<const Tensor> inputs, const OpAttributes&) {
    (void)inputs;
    throw_webgpu_not_ready("Ones");
}

Tensor full_kernel(std::span<const Tensor> inputs, const OpAttributes&) {
    (void)inputs;
    throw_webgpu_not_ready("Full");
}

Tensor rand_kernel(std::span<const Tensor> inputs, const OpAttributes&) {
    (void)inputs;
    throw_webgpu_not_ready("Rand");
}

Tensor randn_kernel(std::span<const Tensor> inputs, const OpAttributes&) {
    (void)inputs;
    throw_webgpu_not_ready("Randn");
}

Tensor arange_kernel(std::span<const Tensor> inputs, const OpAttributes&) {
    (void)inputs;
    throw_webgpu_not_ready("Arange");
}

Tensor linspace_kernel(std::span<const Tensor> inputs, const OpAttributes&) {
    (void)inputs;
    throw_webgpu_not_ready("Linspace");
}

Tensor eye_kernel(std::span<const Tensor> inputs, const OpAttributes&) {
    (void)inputs;
    throw_webgpu_not_ready("Eye");
}

// ---------------------------------------------------------------------------
// Embedding stubs — WGSL entry points: indexing.wgsl (embedding)
// ---------------------------------------------------------------------------

Tensor embedding_kernel(std::span<const Tensor> inputs, const OpAttributes&) {
    (void)inputs;
    throw_webgpu_not_ready("Embedding");
}

// ---------------------------------------------------------------------------
// Normalization stubs — WGSL entry points: batchnorm.wgsl
//   batchnormMean, batchnormVariance, batchnormNormalize, batchnormForward,
//   layernorm, groupnorm
// ---------------------------------------------------------------------------

std::vector<Tensor> batchnorm_mean_var_kernel(std::span<const Tensor> inputs, const OpAttributes&) {
    (void)inputs;
    throw_webgpu_not_ready("BatchNorm2dMeanVar");
}

Tensor batchnorm_forward_kernel(std::span<const Tensor> inputs, const OpAttributes&) {
    (void)inputs;
    throw_webgpu_not_ready("BatchNorm2dForward");
}

Tensor layer_norm_kernel(std::span<const Tensor> inputs, const OpAttributes&) {
    (void)inputs;
    throw_webgpu_not_ready("LayerNorm");
}

Tensor group_norm_kernel(std::span<const Tensor> inputs, const OpAttributes&) {
    (void)inputs;
    throw_webgpu_not_ready("GroupNorm");
}

// ---------------------------------------------------------------------------
// Convolution stubs — WGSL entry points: conv2d.wgsl
//   conv2d_direct, conv2d_tiled, conv2d_depthwise
// ---------------------------------------------------------------------------

std::vector<Tensor> conv2d_forward_kernel(std::span<const Tensor> inputs, const OpAttributes&) {
    (void)inputs;
    throw_webgpu_not_ready("Conv2dForward");
}

Tensor depthwise_conv2d_kernel(std::span<const Tensor> inputs, const OpAttributes&) {
    (void)inputs;
    throw_webgpu_not_ready("DepthwiseConv2d");
}

// ---------------------------------------------------------------------------
// Pooling stubs — WGSL entry points: pooling.wgsl
//   maxPool2d, avgPool2d, globalAvgPool, globalMaxPool,
//   adaptiveAvgPool2d, adaptiveMaxPool2d
// ---------------------------------------------------------------------------

std::vector<Tensor> maxpool2d_forward_kernel(std::span<const Tensor> inputs, const OpAttributes&) {
    (void)inputs;
    throw_webgpu_not_ready("MaxPool2dForward");
}

Tensor avgpool2d_forward_kernel(std::span<const Tensor> inputs, const OpAttributes&) {
    (void)inputs;
    throw_webgpu_not_ready("AvgPool2dForward");
}

Tensor adaptive_avgpool2d_kernel(std::span<const Tensor> inputs, const OpAttributes&) {
    (void)inputs;
    throw_webgpu_not_ready("AdaptiveAvgPool2d");
}

std::vector<Tensor> adaptive_maxpool2d_kernel(std::span<const Tensor> inputs, const OpAttributes&) {
    (void)inputs;
    throw_webgpu_not_ready("AdaptiveMaxPool2d");
}

// ---------------------------------------------------------------------------
// Tensor manipulation stubs — WGSL entry points: transform.wgsl (flip)
// ---------------------------------------------------------------------------

Tensor flip_kernel(std::span<const Tensor> inputs, const OpAttributes&) {
    (void)inputs;
    throw_webgpu_not_ready("Flip");
}

// ---------------------------------------------------------------------------
// Linear stubs — Uses matmul.wgsl (main) + math.wgsl (add)
// ---------------------------------------------------------------------------

Tensor linear_kernel(std::span<const Tensor> inputs, const OpAttributes&) {
    (void)inputs;
    throw_webgpu_not_ready("Linear");
}

// ---------------------------------------------------------------------------
// Type cast stub
// ---------------------------------------------------------------------------

Tensor cast_kernel(std::span<const Tensor> inputs, const OpAttributes&) {
    (void)inputs;
    throw_webgpu_not_ready("Cast");
}

// ---------------------------------------------------------------------------
// Dropout stub (uses host random + math.wgsl: mul)
// ---------------------------------------------------------------------------

std::vector<Tensor> dropout_kernel(std::span<const Tensor> inputs, const OpAttributes&) {
    (void)inputs;
    throw_webgpu_not_ready("Dropout");
}

} // anonymous namespace

// ===========================================================================
// Public registration function
// ===========================================================================

/**
 * @brief Register all WebGPU kernels with the dispatch table.
 *
 * Maps ~80 core OpIds to WebGPU stub kernels. Each stub will eventually
 * dispatch the corresponding WGSL compute shader via the WebGPUBackend
 * pipeline infrastructure. The stubs are organized by WGSL shader file:
 *
 *   math.wgsl       — arithmetic, element-wise math, comparisons, fill/copy
 *   matmul.wgsl     — matmul, bmm, matvec
 *   activations.wgsl — relu, sigmoid, tanh, gelu, softmax, etc.
 *   reduction.wgsl  — sum, mean, max, min, prod, argmax, argmin, var, std, norm
 *   transform.wgsl  — reshape, transpose, permute, slice, cat, stack, etc.
 *   indexing.wgsl   — gather, scatter, index_select, masked_fill, embedding, etc.
 *   batchnorm.wgsl  — batchnorm, layernorm, groupnorm
 *   conv2d.wgsl     — conv2d_direct, conv2d_tiled, conv2d_depthwise
 *   pooling.wgsl    — maxpool2d, avgpool2d, adaptive pools
 */
void register_webgpu_kernels(BackendDispatchTable& table) {
    // =========================================================================
    // Arithmetic Operations — math.wgsl (add, sub, mul, div), matmul.wgsl
    // =========================================================================
    table.register_single_output_kernel(OpId::Add, add_kernel);
    table.register_single_output_kernel(OpId::Sub, sub_kernel);
    table.register_single_output_kernel(OpId::Mul, mul_kernel);
    table.register_single_output_kernel(OpId::Div, div_kernel);
    table.register_single_output_kernel(OpId::MatMul, matmul_kernel);
    table.register_single_output_kernel(OpId::Bmm, bmm_kernel);
    table.register_single_output_kernel(OpId::Dot, dot_kernel);

    // =========================================================================
    // Activation Operations — activations.wgsl
    // =========================================================================
    table.register_single_output_kernel(OpId::ReLU, relu_kernel);
    table.register_single_output_kernel(OpId::Sigmoid, sigmoid_kernel);
    table.register_single_output_kernel(OpId::TanhActivation, tanh_activation_kernel);
    table.register_single_output_kernel(OpId::Gelu, gelu_kernel);
    table.register_single_output_kernel(OpId::Swish, swish_kernel);
    table.register_single_output_kernel(OpId::LeakyReLU, leaky_relu_kernel);
    table.register_single_output_kernel(OpId::Elu, elu_kernel);
    table.register_single_output_kernel(OpId::Selu, selu_kernel);
    table.register_single_output_kernel(OpId::Mish, mish_kernel);
    table.register_single_output_kernel(OpId::Softplus, softplus_kernel);
    table.register_kernel(OpId::Softmax, softmax_kernel);
    table.register_kernel(OpId::LogSoftmax, log_softmax_kernel);

    // =========================================================================
    // Reduction Operations — reduction.wgsl
    // =========================================================================
    table.register_kernel(OpId::Sum, sum_kernel);
    table.register_kernel(OpId::Mean, mean_kernel);
    table.register_kernel(OpId::Max, max_kernel);
    table.register_kernel(OpId::Min, min_kernel);
    table.register_kernel(OpId::ArgMax, argmax_kernel);
    table.register_kernel(OpId::ArgMin, argmin_kernel);
    table.register_kernel(OpId::Prod, prod_kernel);
    table.register_kernel(OpId::Var, var_kernel);
    table.register_kernel(OpId::Std, std_kernel);
    table.register_kernel(OpId::Norm, norm_kernel);

    // =========================================================================
    // Element-wise Math — math.wgsl
    // =========================================================================
    table.register_single_output_kernel(OpId::Sqrt, sqrt_kernel);
    table.register_single_output_kernel(OpId::Neg, neg_kernel);
    table.register_single_output_kernel(OpId::Abs, abs_kernel);
    table.register_single_output_kernel(OpId::Sign, sign_kernel);
    table.register_single_output_kernel(OpId::Log, log_kernel);
    table.register_single_output_kernel(OpId::Exp, exp_kernel);
    table.register_single_output_kernel(OpId::Pow, pow_kernel);
    table.register_single_output_kernel(OpId::Clamp, clamp_kernel);
    table.register_single_output_kernel(OpId::Reciprocal, reciprocal_kernel);
    table.register_single_output_kernel(OpId::Floor, floor_kernel);
    table.register_single_output_kernel(OpId::Ceil, ceil_kernel);
    table.register_single_output_kernel(OpId::Round, round_kernel);

    // =========================================================================
    // Trigonometric Operations — math.wgsl
    // =========================================================================
    table.register_single_output_kernel(OpId::Sin, sin_kernel);
    table.register_single_output_kernel(OpId::Cos, cos_kernel);
    table.register_single_output_kernel(OpId::Tan, tan_kernel);
    table.register_single_output_kernel(OpId::Asin, asin_kernel);
    table.register_single_output_kernel(OpId::Acos, acos_kernel);
    table.register_single_output_kernel(OpId::Atan, atan_kernel);

    // =========================================================================
    // Comparison Operations — math.wgsl
    // =========================================================================
    table.register_single_output_kernel(OpId::Eq, eq_kernel);
    table.register_single_output_kernel(OpId::Ne, ne_kernel);
    table.register_single_output_kernel(OpId::Lt, lt_kernel);
    table.register_single_output_kernel(OpId::Le, le_kernel);
    table.register_single_output_kernel(OpId::Gt, gt_kernel);
    table.register_single_output_kernel(OpId::Ge, ge_kernel);

    // =========================================================================
    // Shape / View Operations — transform.wgsl, math.wgsl (fill, copy)
    // =========================================================================
    table.register_single_output_kernel(OpId::Reshape, reshape_kernel);
    table.register_single_output_kernel(OpId::Transpose, transpose_kernel);
    table.register_single_output_kernel(OpId::Permute, permute_kernel);
    table.register_single_output_kernel(OpId::Squeeze, squeeze_kernel);
    table.register_single_output_kernel(OpId::Unsqueeze, unsqueeze_kernel);
    table.register_single_output_kernel(OpId::Flatten, flatten_kernel);
    table.register_single_output_kernel(OpId::Contiguous, contiguous_kernel);
    table.register_single_output_kernel(OpId::Clone, clone_kernel);
    table.register_single_output_kernel(OpId::Fill, fill_kernel);
    table.register_single_output_kernel(OpId::Repeat, repeat_kernel);
    table.register_single_output_kernel(OpId::Tile, tile_kernel);
    table.register_single_output_kernel(OpId::Expand, expand_kernel);
    table.register_kernel(OpId::Stack, stack_kernel);
    table.register_kernel(OpId::Split, split_kernel);

    // =========================================================================
    // Indexing Operations — indexing.wgsl, math.wgsl (where), transform.wgsl (slice, concat)
    // =========================================================================
    table.register_single_output_kernel(OpId::IndexSelect, index_select_kernel);
    table.register_single_output_kernel(OpId::Gather, gather_kernel);
    table.register_single_output_kernel(OpId::Scatter, scatter_kernel);
    table.register_single_output_kernel(OpId::MaskedSelect, masked_select_kernel);
    table.register_single_output_kernel(OpId::MaskedFill, masked_fill_kernel);
    table.register_single_output_kernel(OpId::Where, where_kernel);
    table.register_single_output_kernel(OpId::Slice, slice_kernel);
    table.register_kernel(OpId::Cat, cat_kernel);
    table.register_single_output_kernel(OpId::Take, take_kernel);
    table.register_single_output_kernel(OpId::Put, put_kernel);
    table.register_single_output_kernel(OpId::OneHot, one_hot_kernel);

    // =========================================================================
    // Creation Operations — host-side + math.wgsl (fill)
    // =========================================================================
    table.register_single_output_kernel(OpId::Zeros, zeros_kernel);
    table.register_single_output_kernel(OpId::Ones, ones_kernel);
    table.register_single_output_kernel(OpId::Full, full_kernel);
    table.register_single_output_kernel(OpId::Rand, rand_kernel);
    table.register_single_output_kernel(OpId::Randn, randn_kernel);
    table.register_single_output_kernel(OpId::Arange, arange_kernel);
    table.register_single_output_kernel(OpId::Linspace, linspace_kernel);
    table.register_single_output_kernel(OpId::Eye, eye_kernel);

    // =========================================================================
    // Embedding — indexing.wgsl (embedding)
    // =========================================================================
    table.register_single_output_kernel(OpId::Embedding, embedding_kernel);

    // =========================================================================
    // Normalization — batchnorm.wgsl
    // =========================================================================
    table.register_kernel(OpId::BatchNorm2dMeanVar, batchnorm_mean_var_kernel);
    table.register_single_output_kernel(OpId::BatchNorm2dForward, batchnorm_forward_kernel);
    table.register_single_output_kernel(OpId::LayerNorm, layer_norm_kernel);
    table.register_single_output_kernel(OpId::GroupNorm, group_norm_kernel);

    // =========================================================================
    // Convolution — conv2d.wgsl
    // =========================================================================
    table.register_kernel(OpId::Conv2dForward, conv2d_forward_kernel);
    table.register_single_output_kernel(OpId::DepthwiseConv2d, depthwise_conv2d_kernel);

    // =========================================================================
    // Pooling — pooling.wgsl
    // =========================================================================
    table.register_kernel(OpId::MaxPool2dForward, maxpool2d_forward_kernel);
    table.register_single_output_kernel(OpId::AvgPool2dForward, avgpool2d_forward_kernel);
    table.register_single_output_kernel(OpId::AdaptiveAvgPool2d, adaptive_avgpool2d_kernel);
    table.register_kernel(OpId::AdaptiveMaxPool2d, adaptive_maxpool2d_kernel);

    // =========================================================================
    // Tensor Manipulation — transform.wgsl (flip)
    // =========================================================================
    table.register_single_output_kernel(OpId::Flip, flip_kernel);

    // =========================================================================
    // Linear — matmul.wgsl + math.wgsl (add)
    // =========================================================================
    table.register_single_output_kernel(OpId::Linear, linear_kernel);

    // =========================================================================
    // Type Conversion
    // =========================================================================
    table.register_single_output_kernel(OpId::Cast, cast_kernel);

    // =========================================================================
    // Dropout — host random + math.wgsl (mul)
    // =========================================================================
    table.register_kernel(OpId::Dropout, dropout_kernel);
}

} // namespace tenzor

// ===========================================================================
// C ABI export — called by init.cpp via dlsym("register_kernels")
// ===========================================================================
extern "C" {
    void register_kernels(tenzor::BackendDispatchTable* table) {
        if (table) {
            tenzor::register_webgpu_kernels(*table);
        }
    }
}
