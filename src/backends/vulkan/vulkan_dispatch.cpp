/**
 * @file vulkan_dispatch.cpp
 * @brief Vulkan backend top-level dispatch() method - routes op_name strings to dispatchXXX methods
 */

#include "vulkan_helpers.hpp"
#include "tenzor/backend/fast_dispatch.hpp"
#include "tenzor/backend/op_attributes.hpp"
#include "tenzor/utils/logging.hpp"

#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace tenzor {

auto VulkanBackend::dispatch(const std::string& op_name,
                            std::span<const Tensor> inputs,
                            const OpAttributes& attrs) -> std::vector<Tensor> {
    // Determine device from inputs or attributes for per-device locking.
    int32_t dispatch_device_id = 0;
    for (const auto& t : inputs) {
        if (t.device().type == Device::Type::Vulkan) {
            dispatch_device_id = t.device().index;
            break;
        }
    }
    if (attrs.has(AttrKey::DeviceId)) {
        dispatch_device_id = static_cast<int32_t>(attrs.get_int(AttrKey::DeviceId));
    }
    // Lock per-device mutex for independent multi-GPU operation.
    std::lock_guard<std::recursive_mutex> lock(devices_[dispatch_device_id].mutex);

    // =========================================================================
    // Float16 handling: operations with native F16 shaders use them directly;
    // operations without F16 support fall back to CPU computation.
    // Accumulation-heavy operations (softmax, layer_norm) upcast to F32 in
    // their individual dispatch functions for numerical stability.
    // =========================================================================
    {
        bool has_float16 = false;
        Device original_device{};
        for (const auto& t : inputs) {
            if (t.dtype() == DType::Float16) {
                has_float16 = true;
                original_device = t.device();
                break;
            }
        }

        if (has_float16) {
            // Operations with native Float16 shader support
            static const std::unordered_set<std::string> f16_native_ops = {
                // Binary ops (math_broadcast_f16 shader)
                "add", "sub", "mul", "div",
                "add_inplace", "sub_inplace", "mul_inplace", "div_inplace",
                // Activation ops (activations_f16 shader)
                "relu", "sigmoid", "tanh", "gelu", "leaky_relu", "swish",
                "relu_inplace", "sigmoid_inplace", "tanh_inplace",
                "gelu_inplace", "leaky_relu_inplace",
                // Activation backward (activations_backward_f16 shader)
                "relu_backward", "sigmoid_backward", "tanh_backward",
                "gelu_backward", "leaky_relu_backward",
                // Softmax (upcasts to F32 in dispatch function)
                "softmax", "softmax_backward",
                // Layer norm (upcasts to F32 in dispatch function)
                "layer_norm", "layer_norm_backward",
                // Group norm
                "group_norm", "group_norm_backward",
                // RMSNorm
                "fused_rms_norm", "rms_norm_backward",
                // Embedding backward
                "embedding_backward",
                // Conv2d forward (conv2d_forward_f16 shader with F32 accumulation)
                "conv2d", "conv2d_forward",
                // ConvTranspose2d (conv_transpose2d_forward_f16 shader)
                "conv_transpose2d_forward",
                // Adaptive pooling (adaptive_pooling_f16 shader)
                "adaptive_avg_pool2d", "adaptive_max_pool2d",
                "adaptive_avg_pool2d_backward",
                // Max pool backward with indices
                "max_pool2d_backward_with_indices",
                // Expand/repeat (expand_f16 shader)
                "expand", "repeat",
                // Pooling (max_pool2d_f16, avg_pool2d_f16 shaders)
                "max_pool2d", "max_pool2d_forward", "max_pool2d_backward",
                "avg_pool2d", "avg_pool2d_forward", "avg_pool2d_backward",
                // Strided copy (strided_copy_f16 shader)
                "strided_copy",
                // Random (random_f16 shader)
                "uniform_random", "normal_random",
                // Unary math (math_f16 shader)
                "sqrt", "exp", "log", "neg", "abs", "sign", "pow",
                "floor", "ceil", "round", "trunc", "reciprocal",
                // Trigonometric (trigonometric_f16 shader)
                "sin", "cos", "tan", "asin", "acos", "atan",
                // Hyperbolic (hyperbolic_f16 shader)
                "sinh", "cosh",
                // Comparison (comparison_f16 shader)
                "eq", "ne", "lt", "le", "gt", "ge",
                // Matrix ops (matmul_f16 shader with F32 accumulation)
                "matmul", "bmm", "dot",
                // Fused operations
                "fused_linear_relu", "fused_batchnorm_relu", "fused_add_relu",
                "fused_conv2d_relu", "fused_gelu", "fused_layer_norm",
                "fused_softmax_cross_entropy",
                // BatchNorm (native F16 shaders)
                "argmax", "argmin", "argsort",
                "batchnorm2d_forward", "batchnorm2d_forward_affine",
                "batchnorm2d_backward",
                "batchnorm2d_mean_var", "batchnorm2d_update_running_stats",
                "index_select",
                // Reduction (reduction_f16 shader)
                "sum", "mean", "max", "min", "prod",
                // Type-agnostic operations
                "reshape", "view", "contiguous", "to", "to_dtype",
                "zeros", "ones", "full", "empty",
                // Creation ops (native F16 shaders)
                "arange", "linspace", "eye", "one_hot", "diag",
                // Clamp (dispatch handles F16 via upcast)
                "clamp", "clamp_min", "clamp_max",
                // Conv2d backward (native F16 shaders with F32 accumulation)
                "conv2d_backward_input", "conv2d_backward_weight", "conv2d_backward_bias",
                // Shape ops (type-agnostic / metadata-only)
                "transpose", "permute", "cat", "squeeze", "unsqueeze",
                // Indexing ops (dispatch handles F16)
                "gather", "scatter", "embedding", "masked_fill",
                "masked_select", "where",
                // Reduction ops (upcast to F32 internally)
                "var", "std", "norm",
                // Log softmax (dispatch upcasts to F32)
                "log_softmax", "log_softmax_backward",
                // Interpolation
                "interpolate",
                // Memory ops (type-agnostic)
                "clone", "fill", "unfold", "fold",
                // Activation forward/backward (activations_f16 / activations_backward_f16 shaders)
                "elu", "elu_backward", "selu", "selu_backward",
                "mish", "mish_backward", "softplus", "softplus_backward",
                "swish_backward",
                // Pooling (pooling_f16, pooling_backward_f16, pooling_forward_with_indices_f16 shaders)
                "pooling", "pooling_backward", "pooling_forward_with_indices",
                // Max pool backward recompute (max_pool2d_backward_f16 shader)
                "max_pool2d_backward_recompute",
                // Adaptive max pool backward (adaptive_max_pool2d_backward_f16 shader)
                "adaptive_max_pool2d_backward",
                // LSTM cell (lstm_cell_f16, lstm_cell_backward_f16 shaders)
                "lstm_cell", "lstm_cell_backward",
                // GRU cell (gru_cell_f16, gru_cell_backward_f16 shaders)
                "gru_cell", "gru_cell_backward",
            };

            if (!f16_native_ops.contains(op_name)) {
                // CPU fallback for any remaining ops not in the native set
                // (This map should be empty — all known F16 ops are now handled natively)
                static const std::unordered_map<std::string, OpId> op_name_to_id = {
                };

                auto it = op_name_to_id.find(op_name);
                if (it != op_name_to_id.end()) {
                    TENZOR_LOG_WARNING(std::format("Vulkan: No shader for {} with dtype Float16; falling back to CPU", op_name));
                    std::vector<Tensor> cpu_inputs;
                    cpu_inputs.reserve(inputs.size());
                    for (const auto& t : inputs) {
                        cpu_inputs.push_back(t.to(Device::cpu()));
                    }
                    auto cpu_results = tenzor::dispatch(it->second, cpu_inputs, attrs);
                    std::vector<Tensor> vulkan_results;
                    vulkan_results.reserve(cpu_results.size());
                    for (auto& r : cpu_results) {
                        vulkan_results.push_back(r.to(original_device));
                    }
                    return vulkan_results;
                }
            }
        }
    }

    try {
    // Binary operations
    if (op_name == "add" || op_name == "sub" || op_name == "mul" || op_name == "div") {
        if (inputs.size() != 2) {
            throw std::invalid_argument(op_name + " requires 2 inputs");
        }
        return {dispatchBinaryOp(op_name, inputs[0], inputs[1])};
    }

    // In-place operations (modify first tensor in-place)
    if (op_name == "add_inplace" || op_name == "sub_inplace" ||
        op_name == "mul_inplace" || op_name == "div_inplace") {
        if (inputs.size() != 2) {
            throw std::invalid_argument(op_name + " requires 2 inputs");
        }
        // Extract the base operation name (remove "_inplace" suffix)
        std::string base_op = op_name.substr(0, op_name.find("_inplace"));

        // Perform the operation and copy result back to first tensor's buffer
        Tensor result = dispatchBinaryOp(base_op, inputs[0], inputs[1]);

        // Copy result data back to input tensor's buffer (in-place modification)
        // Note: const_cast is safe here as we're explicitly modifying the tensor in-place
        size_t bytes = result.numel() * result.dtype_size();
        if (bytes > 0) {
            void* dst = const_cast<void*>(inputs[0].data_ptr());
            copy(dst, result.data_ptr(), bytes, CopyKind::DeviceToDevice);
        }

        return {inputs[0]};
    }

    // In-place activation operations
    if (op_name == "relu_inplace" || op_name == "sigmoid_inplace" || op_name == "tanh_inplace" ||
        op_name == "leaky_relu_inplace" || op_name == "gelu_inplace") {
        if (inputs.size() < 1) {
            throw std::invalid_argument(op_name + " requires 1 input");
        }
        // Determine the base activation and its opcode/param
        std::string base_op = op_name.substr(0, op_name.find("_inplace"));
        uint32_t opcode = 0;
        float param = 0.0f;
        if (base_op == "relu") opcode = 0;
        else if (base_op == "sigmoid") opcode = 1;
        else if (base_op == "tanh") opcode = 2;
        else if (base_op == "gelu") opcode = 3;
        else if (base_op == "leaky_relu") {
            opcode = 4;
            param = static_cast<float>(attrs.get_float(AttrKey::Alpha, 0.01));
        }

        // Dispatch out-of-place activation, then copy result back
        Tensor result = dispatchActivation(base_op, inputs[0], opcode, param);
        size_t bytes = result.numel() * result.dtype_size();
        if (bytes > 0) {
            void* dst = const_cast<void*>(inputs[0].data_ptr());
            copy(dst, result.data_ptr(), bytes, CopyKind::DeviceToDevice);
        }
        return {inputs[0]};
    }

    // Comparison operations
    if (op_name == "eq" || op_name == "ne" || op_name == "lt" ||
        op_name == "le" || op_name == "gt" || op_name == "ge") {
        if (inputs.size() != 2) {
            throw std::invalid_argument(op_name + " requires 2 inputs");
        }
        return {dispatchComparisonOp(op_name, inputs[0], inputs[1])};
    }

    // Activation functions (use activations shader)
    if (op_name == "relu" || op_name == "sigmoid" || op_name == "tanh") {
        if (inputs.size() != 1) {
            throw std::invalid_argument(op_name + " requires 1 input");
        }
        uint32_t opcode = 0;
        if (op_name == "relu") opcode = 0;
        else if (op_name == "sigmoid") opcode = 1;
        else if (op_name == "tanh") opcode = 2;
        return {dispatchActivation(op_name, inputs[0], opcode, 0.0f)};
    }

    // GELU activation
    if (op_name == "gelu") {
        if (inputs.size() != 1) throw std::invalid_argument("gelu requires 1 input");
        return {dispatchActivation("gelu", inputs[0], 3, 0.0f)};
    }

    // LeakyReLU activation
    if (op_name == "leaky_relu") {
        if (inputs.size() != 1) throw std::invalid_argument("leaky_relu requires 1 input");
        float alpha = static_cast<float>(attrs.get_float(AttrKey::Alpha, 0.01));
        return {dispatchActivation("leaky_relu", inputs[0], 4, alpha)};
    }

    // Swish activation
    if (op_name == "swish") {
        if (inputs.size() != 1) throw std::invalid_argument("swish requires 1 input");
        return {dispatchActivation("swish", inputs[0], 5, 0.0f)};
    }

    // ELU activation
    if (op_name == "elu") {
        if (inputs.size() != 1) throw std::invalid_argument("elu requires 1 input");
        float alpha = static_cast<float>(attrs.get_float(AttrKey::Alpha, 1.0));
        return {dispatchActivation("elu", inputs[0], 6, alpha)};
    }

    // SELU activation
    if (op_name == "selu") {
        if (inputs.size() != 1) throw std::invalid_argument("selu requires 1 input");
        return {dispatchActivation("selu", inputs[0], 7, 0.0f)};
    }

    // Mish activation
    if (op_name == "mish") {
        if (inputs.size() != 1) throw std::invalid_argument("mish requires 1 input");
        return {dispatchActivation("mish", inputs[0], 8, 0.0f)};
    }

    // Softplus activation
    if (op_name == "softplus") {
        if (inputs.size() != 1) throw std::invalid_argument("softplus requires 1 input");
        float beta = static_cast<float>(attrs.get_float(AttrKey::Beta, 1.0));
        return {dispatchActivation("softplus", inputs[0], 9, beta)};
    }

    // Unary math operations (use math shader)
    if (op_name == "sqrt" || op_name == "exp" || op_name == "log" ||
        op_name == "neg" || op_name == "abs" || op_name == "sign" ||
        op_name == "floor" || op_name == "ceil" || op_name == "round" ||
        op_name == "trunc" || op_name == "reciprocal") {
        if (inputs.size() != 1) {
            throw std::invalid_argument(op_name + " requires 1 input");
        }
        return {dispatchUnaryOp(op_name, inputs[0])};
    }

    // Trigonometric operations
    if (op_name == "sin" || op_name == "cos" || op_name == "tan" ||
        op_name == "asin" || op_name == "acos" || op_name == "atan") {
        if (inputs.size() != 1) {
            throw std::invalid_argument(op_name + " requires 1 input");
        }
        return {dispatchTrigonometricOp(op_name, inputs[0])};
    }

    // Hyperbolic operations
    if (op_name == "sinh" || op_name == "cosh" || op_name == "tanh") {
        if (inputs.size() != 1) {
            throw std::invalid_argument(op_name + " requires 1 input");
        }
        return {dispatchHyperbolicOp(op_name, inputs[0])};
    }

    // Pow operation (unary with parameter)
    if (op_name == "pow") {
        if (inputs.size() != 1) {
            throw std::invalid_argument("pow requires 1 input");
        }
        float exponent = static_cast<float>(attrs.get_float(AttrKey::Exponent, 2.0));
        return {dispatchUnaryOpWithParam(op_name, inputs[0], exponent)};
    }

    // Backward activation operations
    if (op_name == "relu_backward") {
        if (inputs.size() != 2) throw std::invalid_argument("relu_backward requires 2 inputs");
        return {dispatchActivationBackward("relu_backward", inputs[0], inputs[1], 0, 0.0f)};
    }

    if (op_name == "sigmoid_backward") {
        if (inputs.size() != 2) throw std::invalid_argument("sigmoid_backward requires 2 inputs");
        return {dispatchActivationBackward("sigmoid_backward", inputs[0], inputs[1], 1, 0.0f)};
    }

    if (op_name == "tanh_backward") {
        if (inputs.size() != 2) throw std::invalid_argument("tanh_backward requires 2 inputs");
        return {dispatchActivationBackward("tanh_backward", inputs[0], inputs[1], 2, 0.0f)};
    }

    if (op_name == "leaky_relu_backward") {
        if (inputs.size() != 2) throw std::invalid_argument("leaky_relu_backward requires 2 inputs");
        float alpha = static_cast<float>(attrs.get_float(AttrKey::Alpha, 0.01));
        return {dispatchActivationBackward("leaky_relu_backward", inputs[0], inputs[1], 3, alpha)};
    }

    if (op_name == "gelu_backward") {
        if (inputs.size() != 2) throw std::invalid_argument("gelu_backward requires 2 inputs");
        return {dispatchActivationBackward("gelu_backward", inputs[0], inputs[1], 4, 0.0f)};
    }

    if (op_name == "elu_backward") {
        if (inputs.size() != 2) throw std::invalid_argument("elu_backward requires 2 inputs");
        float alpha = static_cast<float>(attrs.get_float(AttrKey::Alpha, 1.0));
        return {dispatchActivationBackward("elu_backward", inputs[0], inputs[1], 5, alpha)};
    }

    if (op_name == "selu_backward") {
        if (inputs.size() != 2) throw std::invalid_argument("selu_backward requires 2 inputs");
        return {dispatchActivationBackward("selu_backward", inputs[0], inputs[1], 6, 0.0f)};
    }

    if (op_name == "mish_backward") {
        if (inputs.size() != 2) throw std::invalid_argument("mish_backward requires 2 inputs");
        return {dispatchActivationBackward("mish_backward", inputs[0], inputs[1], 7, 0.0f)};
    }

    if (op_name == "softplus_backward") {
        if (inputs.size() != 2) throw std::invalid_argument("softplus_backward requires 2 inputs");
        float beta = static_cast<float>(attrs.get_float(AttrKey::Beta, 1.0));
        return {dispatchActivationBackward("softplus_backward", inputs[0], inputs[1], 8, beta)};
    }

    if (op_name == "swish_backward") {
        if (inputs.size() != 2) throw std::invalid_argument("swish_backward requires 2 inputs");
        return {dispatchSwishBackward(inputs[0], inputs[1])};
    }

    if (op_name == "softmax_backward") {
        if (inputs.size() != 2) throw std::invalid_argument("softmax_backward requires 2 inputs");
        int64_t dim = attrs.get_int(AttrKey::Dim, -1);
        return {dispatchSoftmaxBackward(inputs[0], inputs[1], dim)};
    }

    if (op_name == "log_softmax_backward") {
        if (inputs.size() != 2) throw std::invalid_argument("log_softmax_backward requires 2 inputs");
        int64_t dim = attrs.get_int(AttrKey::Dim, -1);
        return {dispatchLogSoftmaxBackward(inputs[0], inputs[1], dim)};
    }

    // Conv2d backward operations
    if (op_name == "conv2d_backward_input") {
        if (inputs.size() != 2) throw std::invalid_argument("conv2d_backward_input requires 2 inputs (grad_output, weight)");
        int64_t stride = attrs.get_int(AttrKey::Stride, 1);
        int64_t padding = attrs.get_int(AttrKey::Padding, 0);
        int64_t dilation = attrs.get_int(AttrKey::Dilation, 1);

        // Parse input shape from comma-separated string
        std::vector<int64_t> input_shape = attrs.get_int_list(AttrKey::InputShape);

        return {dispatchConv2dBackwardInput(inputs[0], inputs[1], stride, padding, dilation, input_shape)};
    }

    if (op_name == "conv2d_backward_weight") {
        if (inputs.size() != 2) throw std::invalid_argument("conv2d_backward_weight requires 2 inputs (grad_output, input)");
        int64_t stride = attrs.get_int(AttrKey::Stride, 1);
        int64_t padding = attrs.get_int(AttrKey::Padding, 0);
        int64_t dilation = attrs.get_int(AttrKey::Dilation, 1);

        // Parse weight shape from comma-separated string
        std::vector<int64_t> weight_shape = attrs.get_int_list(AttrKey::WeightShape);

        return {dispatchConv2dBackwardWeight(inputs[0], inputs[1], stride, padding, dilation, weight_shape)};
    }

    if (op_name == "conv2d_backward_bias") {
        if (inputs.size() != 1) throw std::invalid_argument("conv2d_backward_bias requires 1 input (grad_output)");
        return {dispatchConv2dBackwardBias(inputs[0])};
    }

    // Unified conv2d backward: computes grad_input, grad_weight, and optionally grad_bias
    // inputs: [grad_output, input, weight]
    if (op_name == "conv2d_backward") {
        if (inputs.size() != 3) {
            throw std::invalid_argument("conv2d_backward operation requires exactly 3 inputs (grad_output, input, weight)");
        }
        const Tensor& grad_output = inputs[0];
        const Tensor& input = inputs[1];
        const Tensor& weight = inputs[2];

        int64_t stride = attrs.get_int(AttrKey::Stride, 1);
        int64_t padding = attrs.get_int(AttrKey::Padding, 0);
        int64_t dilation = attrs.get_int(AttrKey::Dilation, 1);
        int64_t groups = attrs.get_int(AttrKey::Groups, 1);
        bool compute_grad_input = attrs.get_bool(AttrKey::ComputeGradInput, true);
        bool compute_grad_weight = attrs.get_bool(AttrKey::ComputeGradWeight, true);
        bool compute_grad_bias = attrs.get_bool(AttrKey::ComputeGradBias, false);

        std::vector<int64_t> input_shape(input.shape().begin(), input.shape().end());
        std::vector<int64_t> weight_shape(weight.shape().begin(), weight.shape().end());

        std::vector<Tensor> results;

        // Compute grad_input
        if (compute_grad_input) {
            results.push_back(dispatchConv2dBackwardInput(grad_output, weight, stride, padding, dilation, input_shape, groups));
        }

        // Compute grad_weight
        if (compute_grad_weight) {
            results.push_back(dispatchConv2dBackwardWeight(grad_output, input, stride, padding, dilation, weight_shape, groups));
        }

        // Compute grad_bias
        if (compute_grad_bias) {
            results.push_back(dispatchConv2dBackwardBias(grad_output));
        }

        return results;
    }

    // Reduction operations
    if (op_name == "sum" || op_name == "mean" || op_name == "max" || op_name == "min") {
        if (inputs.size() != 1) {
            throw std::invalid_argument(op_name + " requires 1 input");
        }
        // Use INT64_MIN as sentinel for "reduce all elements" (full reduction)
        // dim=-1, -2, etc. mean negative indexing from the end
        int64_t dim = attrs.has(AttrKey::Dim) ? attrs.get_int(AttrKey::Dim) : INT64_MIN;
        bool keepdim = attrs.get_bool(AttrKey::Keepdim, false);
        return {dispatchReduction(op_name, inputs[0], dim, keepdim)};
    }

    // Matrix multiplication
    if (op_name == "matmul") {
        if (inputs.size() != 2) {
            throw std::invalid_argument("matmul requires 2 inputs");
        }
        return {dispatchMatmul(inputs[0], inputs[1])};
    }

    // Batched matrix multiplication
    if (op_name == "bmm") {
        if (inputs.size() != 2) {
            throw std::invalid_argument("bmm requires 2 inputs");
        }
        return {dispatchBmm(inputs[0], inputs[1])};
    }

    // Pooling operations
    if (op_name == "max_pool2d") {
        // Support both kernel_h/kernel_w and kernel_size attributes
        int64_t kernel_h = 2, kernel_w = 2;
        if (attrs.has(AttrKey::KernelSizeH)) {
            kernel_h = attrs.get_int(AttrKey::KernelSizeH);
            kernel_w = attrs.has(AttrKey::KernelSizeW) ? attrs.get_int(AttrKey::KernelSizeW) : kernel_h;
        } else if (attrs.has(AttrKey::KernelSize)) {
            kernel_h = kernel_w = attrs.get_int(AttrKey::KernelSize);
        }

        // Support both stride_h/stride_w and stride attributes
        int64_t stride_h = kernel_h, stride_w = kernel_w;
        if (attrs.has(AttrKey::StrideH)) {
            stride_h = attrs.get_int(AttrKey::StrideH);
            stride_w = attrs.has(AttrKey::StrideW) ? attrs.get_int(AttrKey::StrideW) : stride_h;
        } else if (attrs.has(AttrKey::Stride)) {
            stride_h = stride_w = attrs.get_int(AttrKey::Stride);
        }

        // Support both padding_h/padding_w and padding attributes
        int64_t padding_h = 0, padding_w = 0;
        if (attrs.has(AttrKey::PaddingH)) {
            padding_h = attrs.get_int(AttrKey::PaddingH);
            padding_w = attrs.has(AttrKey::PaddingW) ? attrs.get_int(AttrKey::PaddingW) : padding_h;
        } else if (attrs.has(AttrKey::Padding)) {
            padding_h = padding_w = attrs.get_int(AttrKey::Padding);
        }

        // Float16: use max_pool2d_f16 shader via dispatch path (same as Float64)
        if (inputs[0].dtype() == DType::Float16) {
            OpAttributes pool_attrs;
            pool_attrs.set(AttrKey::KernelSizeH, kernel_h);
            pool_attrs.set(AttrKey::KernelSizeW, kernel_w);
            pool_attrs.set(AttrKey::StrideH, stride_h);
            pool_attrs.set(AttrKey::StrideW, stride_w);
            pool_attrs.set(AttrKey::PaddingH, padding_h);
            pool_attrs.set(AttrKey::PaddingW, padding_w);
            Tensor output = dispatchMaxPool2dForward(inputs[0], pool_attrs);
            std::vector<int64_t> out_shape_vec(output.shape().begin(), output.shape().end());
            Tensor pool_indices(out_shape_vec, DType::Int32, inputs[0].device());
            pool_indices = dispatchFill(pool_indices, 0.0f);
            return {output, pool_indices};
        }

        // Float64: use max_pool2d_f64 shader via new dispatch path
        if (inputs[0].dtype() == DType::Float64) {
            OpAttributes pool_attrs;
            pool_attrs.set(AttrKey::KernelSizeH, kernel_h);
            pool_attrs.set(AttrKey::KernelSizeW, kernel_w);
            pool_attrs.set(AttrKey::StrideH, stride_h);
            pool_attrs.set(AttrKey::StrideW, stride_w);
            pool_attrs.set(AttrKey::PaddingH, padding_h);
            pool_attrs.set(AttrKey::PaddingW, padding_w);
            Tensor output = dispatchMaxPool2dForward(inputs[0], pool_attrs);
            // Create indices tensor (Float64 max_pool2d doesn't produce indices)
            std::vector<int64_t> out_shape_vec(output.shape().begin(), output.shape().end());
            Tensor pool_indices(out_shape_vec, DType::Int32, inputs[0].device());
            pool_indices = dispatchFill(pool_indices, 0.0f);
            return {output, pool_indices};
        }

        auto [output, indices] = dispatchMaxPool2d(inputs[0], kernel_h, kernel_w,
                                                    stride_h, stride_w, padding_h, padding_w);
        return {output, indices};
    }

    if (op_name == "avg_pool2d") {
        // Support both kernel_h/kernel_w and kernel_size attributes
        int64_t kernel_h = 2, kernel_w = 2;
        if (attrs.has(AttrKey::KernelSizeH)) {
            kernel_h = attrs.get_int(AttrKey::KernelSizeH);
            kernel_w = attrs.has(AttrKey::KernelSizeW) ? attrs.get_int(AttrKey::KernelSizeW) : kernel_h;
        } else if (attrs.has(AttrKey::KernelSize)) {
            kernel_h = kernel_w = attrs.get_int(AttrKey::KernelSize);
        }

        // Support both stride_h/stride_w and stride attributes
        int64_t stride_h = kernel_h, stride_w = kernel_w;
        if (attrs.has(AttrKey::StrideH)) {
            stride_h = attrs.get_int(AttrKey::StrideH);
            stride_w = attrs.has(AttrKey::StrideW) ? attrs.get_int(AttrKey::StrideW) : stride_h;
        } else if (attrs.has(AttrKey::Stride)) {
            stride_h = stride_w = attrs.get_int(AttrKey::Stride);
        }
        int64_t padding_h = attrs.get_int(AttrKey::PaddingH, 0);
        int64_t padding_w = attrs.get_int(AttrKey::PaddingW, 0);

        // Float16: use avg_pool2d_f16 shader on GPU
        // (Falls through to dispatchAvgPool2d which will select the correct shader)

        return {dispatchAvgPool2d(inputs[0], kernel_h, kernel_w,
                                  stride_h, stride_w, padding_h, padding_w)};
    }

    if (op_name == "adaptive_max_pool2d") {
        // Support both naming conventions: output_height/output_width and output_h/output_w
        int64_t out_h = attrs.has(AttrKey::OutputHeight) ? attrs.get_int(AttrKey::OutputHeight)
                      : attrs.get_int(AttrKey::OutputSizeH);
        int64_t out_w = attrs.has(AttrKey::OutputWidth) ? attrs.get_int(AttrKey::OutputWidth)
                      : attrs.get_int(AttrKey::OutputSizeW);
        auto [output, indices] = dispatchAdaptiveMaxPool2d(inputs[0], out_h, out_w);
        return {output, indices};
    }

    if (op_name == "adaptive_avg_pool2d") {
        // Support both naming conventions: output_height/output_width and output_h/output_w
        int64_t out_h = attrs.has(AttrKey::OutputHeight) ? attrs.get_int(AttrKey::OutputHeight)
                      : attrs.get_int(AttrKey::OutputSizeH);
        int64_t out_w = attrs.has(AttrKey::OutputWidth) ? attrs.get_int(AttrKey::OutputWidth)
                      : attrs.get_int(AttrKey::OutputSizeW);
        return {dispatchAdaptiveAvgPool2d(inputs[0], out_h, out_w)};
    }

    if (op_name == "adaptive_avg_pool2d_backward") {
        if (inputs.size() != 1) {
            throw std::invalid_argument("adaptive_avg_pool2d_backward requires exactly 1 input (grad_output)");
        }
        int64_t H_in = attrs.get_int(AttrKey::InputH, 0);
        int64_t W_in = attrs.get_int(AttrKey::InputW, 0);
        return {dispatchAdaptiveAvgPool2dBackward(inputs[0], H_in, W_in)};
    }

    // Normalization
    if (op_name == "softmax") {
        int64_t dim = attrs.get_int(AttrKey::Dim, -1);
        return {dispatchSoftmax(inputs[0], dim)};
    }

    if (op_name == "log_softmax") {
        int64_t dim = attrs.get_int(AttrKey::Dim, -1);
        return {dispatchLogSoftmax(inputs[0], dim)};
    }

    // Advanced reductions
    if (op_name == "argmax") {
        int64_t dim = attrs.get_int(AttrKey::Dim, -1);
        bool keepdim = attrs.get_bool(AttrKey::Keepdim, false);
        return {dispatchArgmax(inputs[0], dim, keepdim)};
    }

    if (op_name == "argmin") {
        int64_t dim = attrs.get_int(AttrKey::Dim, -1);
        bool keepdim = attrs.get_bool(AttrKey::Keepdim, false);
        return {dispatchArgmin(inputs[0], dim, keepdim)};
    }

    if (op_name == "argsort") {
        int64_t dim = attrs.get_int(AttrKey::Dim, -1);
        bool descending = attrs.get_bool(AttrKey::Descending, false);
        return {dispatchArgSort(inputs[0], dim, descending)};
    }

    if (op_name == "var" || op_name == "variance") {
        int64_t dim = attrs.get_int(AttrKey::Dim, -1);
        bool unbiased = attrs.get_bool(AttrKey::Unbiased, true);
        bool keepdim = attrs.get_bool(AttrKey::Keepdim, false);
        return {dispatchVariance(inputs[0], dim, unbiased, keepdim)};
    }

    if (op_name == "std") {
        int64_t dim = attrs.get_int(AttrKey::Dim, -1);
        bool unbiased = attrs.get_bool(AttrKey::Unbiased, true);
        bool keepdim = attrs.get_bool(AttrKey::Keepdim, false);
        return {dispatchStd(inputs[0], dim, unbiased, keepdim)};
    }

    if (op_name == "prod") {
        int64_t dim = attrs.get_int(AttrKey::Dim, -1);
        bool keepdim = attrs.get_bool(AttrKey::Keepdim, false);
        return {dispatchProd(inputs[0], dim, keepdim)};
    }

    if (op_name == "norm") {
        float p = static_cast<float>(attrs.get_float(AttrKey::P, 2.0));
        // Use INT64_MIN to signal full reduction when dim is not specified
        int64_t dim = attrs.has(AttrKey::Dim) ? attrs.get_int(AttrKey::Dim) : INT64_MIN;
        bool keepdim = attrs.get_bool(AttrKey::Keepdim, false);
        return {dispatchNorm(inputs[0], p, dim, keepdim)};
    }

    // Indexing operations
    if (op_name == "embedding") {
        int64_t padding_idx = attrs.get_int(AttrKey::PaddingIdx, -1);
        return {dispatchEmbedding(inputs[0], inputs[1], padding_idx)};
    }

    if (op_name == "gather") {
        int64_t dim = attrs.get_int(AttrKey::Dim);
        return {dispatchGather(inputs[0], dim, inputs[1])};
    }

    if (op_name == "scatter") {
        int64_t dim = attrs.get_int(AttrKey::Dim);
        int64_t reduction = attrs.get_int(AttrKey::Reduction, 0);
        return {dispatchScatter(inputs[0], dim, inputs[1], inputs[2], reduction)};
    }

    if (op_name == "index_select") {
        int64_t dim = attrs.get_int(AttrKey::Dim);
        return {dispatchIndexSelect(inputs[0], dim, inputs[1])};
    }

    // Vision operations
    if (op_name == "gather_relative_position_bias") {
        if (inputs.size() != 2) {
            throw std::invalid_argument("gather_relative_position_bias operation requires exactly 2 inputs");
        }
        int64_t num_positions = attrs.get_int(AttrKey::NumPositions, 0);
        int64_t num_heads = attrs.get_int(AttrKey::NumHeads, 0);
        return {dispatchGatherRelativePositionBias(inputs[0], inputs[1], num_positions, num_heads)};
    }

    // Shape operations
    if (op_name == "reshape") {
        if (inputs.size() != 1) {
            throw std::invalid_argument("reshape requires 1 input");
        }
        if (!attrs.has(AttrKey::Shape)) {
            throw std::invalid_argument("reshape requires 'shape' attribute");
        }
        std::vector<int64_t> new_shape = attrs.get_int_list(AttrKey::Shape);
        return {dispatchReshape(inputs[0], new_shape)};
    }

    if (op_name == "transpose") {
        if (inputs.size() != 1) {
            throw std::invalid_argument("transpose requires 1 input");
        }
        int64_t dim0 = attrs.get_int(AttrKey::Dim0);
        int64_t dim1 = attrs.get_int(AttrKey::Dim1);
        return {dispatchTranspose(inputs[0], dim0, dim1)};
    }

    if (op_name == "permute") {
        if (inputs.size() != 1) {
            throw std::invalid_argument("permute requires 1 input");
        }
        std::vector<int64_t> dims = attrs.get_int_list(AttrKey::Dims);
        return {dispatchPermute(inputs[0], dims)};
    }

    if (op_name == "squeeze") {
        if (inputs.size() != 1) {
            throw std::invalid_argument("squeeze requires 1 input");
        }
        int64_t dim = attrs.get_int(AttrKey::Dim, -1);
        return {dispatchSqueeze(inputs[0], dim)};
    }

    if (op_name == "unsqueeze") {
        if (inputs.size() != 1) {
            throw std::invalid_argument("unsqueeze requires 1 input");
        }
        int64_t dim = attrs.get_int(AttrKey::Dim);
        return {dispatchUnsqueeze(inputs[0], dim)};
    }

    if (op_name == "contiguous") {
        if (inputs.size() != 1) {
            throw std::invalid_argument("contiguous requires 1 input");
        }
        return {dispatchContiguous(inputs[0])};
    }

    // Memory operations
    if (op_name == "zeros") {
        std::vector<int64_t> shape = attrs.get_int_list(AttrKey::Shape);
        // Get dtype and device from attributes or use defaults
        DType dtype = DType::Float32;
        if (attrs.has(AttrKey::Dtype)) {
            auto dtype_str = attrs.get_string(AttrKey::Dtype);
            if (dtype_str == "float32") dtype = DType::Float32;
            else if (dtype_str == "float64") dtype = DType::Float64;
            else if (dtype_str == "float16") dtype = DType::Float16;
            else if (dtype_str == "bfloat16") dtype = DType::BFloat16;
            else if (dtype_str == "int8") dtype = DType::Int8;
            else if (dtype_str == "int16") dtype = DType::Int16;
            else if (dtype_str == "int32") dtype = DType::Int32;
            else if (dtype_str == "int64") dtype = DType::Int64;
            else if (dtype_str == "uint8") dtype = DType::UInt8;
            else if (dtype_str == "uint16") dtype = DType::UInt16;
            else if (dtype_str == "uint32") dtype = DType::UInt32;
            else if (dtype_str == "uint64") dtype = DType::UInt64;
            else if (dtype_str == "bool") dtype = DType::Bool;
            else if (dtype_str == "complex64") dtype = DType::Complex64;
            else if (dtype_str == "complex128") dtype = DType::Complex128;
        }
        int32_t device_id = static_cast<int32_t>(attrs.get_int(AttrKey::DeviceId, 0));
        Device device = Device::vulkan(device_id);
        return {dispatchZeros(shape, dtype, device)};
    }

    if (op_name == "fill") {
        if (inputs.size() != 1) {
            throw std::invalid_argument("fill requires 1 input");
        }
        float value = static_cast<float>(attrs.get_float(AttrKey::Value));
        return {dispatchFill(inputs[0], value)};
    }

    if (op_name == "clone") {
        if (inputs.size() != 1) {
            throw std::invalid_argument("clone requires 1 input");
        }
        return {dispatchClone(inputs[0])};
    }

    // Vision operations
    if (op_name == "im2col" || op_name == "unfold") {
        if (inputs.size() != 1) {
            throw std::invalid_argument("im2col requires 1 input");
        }
        return {dispatchIm2Col(inputs[0], attrs)};
    }

    if (op_name == "col2im" || op_name == "fold") {
        if (inputs.size() != 1) {
            throw std::invalid_argument("col2im requires 1 input");
        }
        return {dispatchCol2Im(inputs[0], attrs)};
    }

    // Tensor manipulation operations
    if (op_name == "expand") {
        if (inputs.size() != 1) {
            throw std::invalid_argument("expand requires 1 input");
        }
        std::vector<int64_t> shape = attrs.get_int_list(AttrKey::Shape);
        return {dispatchExpand(inputs[0], shape)};
    }

    if (op_name == "cat" || op_name == "concatenate") {
        if (inputs.size() < 2) {
            throw std::invalid_argument("cat requires at least 2 inputs");
        }
        int64_t dim = attrs.get_int(AttrKey::Dim, 0);
        std::vector<Tensor> input_tensors(inputs.begin(), inputs.end());
        return {dispatchCat(input_tensors, dim)};
    }

    if (op_name == "clamp") {
        if (inputs.size() != 1) {
            throw std::invalid_argument("clamp requires 1 input");
        }
        float min_value = static_cast<float>(attrs.get_float(AttrKey::Min, -std::numeric_limits<double>::infinity()));
        float max_value = static_cast<float>(attrs.get_float(AttrKey::Max, std::numeric_limits<double>::infinity()));
        return {dispatchClamp(inputs[0], min_value, max_value)};
    }

    if (op_name == "clamp_min") {
        if (inputs.size() != 1) {
            throw std::invalid_argument("clamp_min requires 1 input");
        }
        float min_value = static_cast<float>(attrs.get_float(AttrKey::Min, -std::numeric_limits<double>::infinity()));
        return {dispatchClamp(inputs[0], min_value, std::numeric_limits<float>::infinity())};
    }

    if (op_name == "clamp_max") {
        if (inputs.size() != 1) {
            throw std::invalid_argument("clamp_max requires 1 input");
        }
        float max_value = static_cast<float>(attrs.get_float(AttrKey::Max, std::numeric_limits<double>::infinity()));
        return {dispatchClamp(inputs[0], -std::numeric_limits<float>::infinity(), max_value)};
    }

    if (op_name == "dot") {
        if (inputs.size() != 2) {
            throw std::invalid_argument("dot requires 2 inputs");
        }
        return {dispatchDot(inputs[0], inputs[1])};
    }

    // BatchNorm2d operations
    if (op_name == "batchnorm2d_forward") {
        if (inputs.size() < 3) {
            throw std::invalid_argument("batchnorm2d_forward requires at least 3 inputs (input, mean, var)");
        }
        const Tensor* gamma = (inputs.size() > 3) ? &inputs[3] : nullptr;
        const Tensor* beta = (inputs.size() > 4) ? &inputs[4] : nullptr;
        float epsilon = static_cast<float>(attrs.get_float(AttrKey::Eps, 1e-5));
        return {dispatchBatchNorm2dForward(inputs[0], inputs[1], inputs[2], gamma, beta, epsilon)};
    }

    if (op_name == "batchnorm2d_forward_affine") {
        // batchnorm2d_forward_affine is the same as batchnorm2d_forward with weight and bias
        // Inputs: input, mean, var, weight, bias
        if (inputs.size() < 5) {
            throw std::invalid_argument("batchnorm2d_forward_affine requires 5 inputs (input, mean, var, weight, bias)");
        }
        float epsilon = static_cast<float>(attrs.get_float(AttrKey::Eps, 1e-5));
        return {dispatchBatchNorm2dForward(inputs[0], inputs[1], inputs[2], &inputs[3], &inputs[4], epsilon)};
    }

    if (op_name == "batchnorm2d_backward") {
        // Autograd passes: [grad_output, input, weight(gamma), mean, invstd]
        if (inputs.size() < 5) {
            throw std::invalid_argument("batchnorm2d_backward requires 5 inputs (grad_output, input, weight, mean, invstd)");
        }
        const Tensor* gamma = &inputs[2];  // weight = gamma
        float epsilon = static_cast<float>(attrs.get_float(AttrKey::Eps, 1e-5));
        // Pass mean=inputs[3], invstd=inputs[4] (shader uses invstd directly)
        auto [grad_input, grad_gamma, grad_beta] = dispatchBatchNorm2dBackward(
            inputs[0], inputs[1], inputs[3], inputs[4], gamma, epsilon);
        return {grad_input, grad_gamma, grad_beta};
    }

    if (op_name == "batchnorm2d_mean_var") {
        if (inputs.size() != 1) {
            throw std::invalid_argument("batchnorm2d_mean_var requires 1 input");
        }
        auto [mean, variance] = dispatchBatchNorm2dMeanVar(inputs[0]);
        return {mean, variance};
    }

    if (op_name == "batchnorm2d_update_running_stats") {
        // Use GPU kernel for updating running statistics with exponential moving average
        if (inputs.size() != 4) {
            throw std::invalid_argument("batchnorm2d_update_running_stats requires 4 inputs (running_mean, running_var, batch_mean, batch_var)");
        }

        float momentum = static_cast<float>(attrs.get_float(AttrKey::Momentum, 0.1));

        const Tensor& running_mean = inputs[0];
        const Tensor& running_var = inputs[1];
        const Tensor& batch_mean = inputs[2];
        const Tensor& batch_var = inputs[3];

        int64_t n_channels = batch_mean.numel();
        if (n_channels == 0) {
            return {running_mean, running_var};
        }

        int32_t device_id = running_mean.device().index;

        // Select shader based on dtype
        std::string shader_name = "batchnorm_update_stats";
        if (running_mean.dtype() == DType::Float64) {
            shader_name = "batchnorm_update_stats_f64";
        } else if (running_mean.dtype() == DType::Float16) {
            shader_name = "batchnorm_update_stats_f16";
        } else if (running_mean.dtype() != DType::Float32) {
            throw std::runtime_error("Unsupported dtype for batchnorm2d_update_running_stats");
        }

        auto* pipeline = getPipeline(shader_name, device_id);

        const void* buffer_rm = running_mean.data_ptr();
        const void* buffer_rv = running_var.data_ptr();
        const void* buffer_bm = const_cast<void*>(batch_mean.data_ptr());
        const void* buffer_bv = const_cast<void*>(batch_var.data_ptr());

        size_t buffer_size = n_channels * running_mean.dtype_size();

        std::vector<std::pair<uint32_t, const void*>> bindings = {
            {0, buffer_rm},
            {1, buffer_rv},
            {2, buffer_bm},
            {3, buffer_bv}
        };
        std::vector<size_t> sizes = {buffer_size, buffer_size, buffer_size, buffer_size};

        VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
            device_id, pipeline, bindings, sizes);

        struct PushConstants {
            uint32_t n_channels;
            float momentum;
        } push_constants;

        push_constants.n_channels = static_cast<uint32_t>(n_channels);
        push_constants.momentum = momentum;

        VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
        vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                               pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
        vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                          VK_SHADER_STAGE_COMPUTE_BIT,
                          0, sizeof(PushConstants), &push_constants);

        uint32_t workgroups = div_wg(n_channels, devices_[device_id].workgroupSize);
        vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

        // Add memory barrier
        insertComputeBarrier(cmdBuffer);

        endSingleTimeCommands(cmdBuffer, device_id);

        // Return the updated tensors (modified in-place)
        return {running_mean, running_var};
    }

    // Pooling operations (new OpAttributes versions)
    if (op_name == "avg_pool2d_forward") {
        if (inputs.size() != 1) {
            throw std::invalid_argument("avg_pool2d_forward requires 1 input");
        }
        return {dispatchAvgPool2dForward(inputs[0], attrs)};
    }

    if (op_name == "max_pool2d_forward") {
        if (inputs.size() != 1) {
            throw std::invalid_argument("max_pool2d_forward requires 1 input");
        }
        return {dispatchMaxPool2dForward(inputs[0], attrs)};
    }

    if (op_name == "avg_pool2d_backward") {
        if (inputs.size() >= 2) {
            return {dispatchAvgPool2dBackward(inputs[0], inputs[1], attrs)};
        }
        if (inputs.size() == 1) {
            // Autograd path: 1 input (grad_output) + shape/pooling params in attrs
            // Build compatible attrs for dispatchAvgPool2dBackward
            OpAttributes bwd_attrs;

            // Parse input shape from "input_shape" attribute (comma-separated "N,C,H,W")
            int64_t in_n = 0, in_c = 0, in_h = 0, in_w = 0;
            if (attrs.has(AttrKey::InputShape)) {
                auto dims = attrs.get_int_list(AttrKey::InputShape);
                if (dims.size() >= 4) {
                    in_n = dims[0]; in_c = dims[1]; in_h = dims[2]; in_w = dims[3];
                }
            }

            // Parse pooling parameters (support both single-value and h/w variants)
            int64_t kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w;
            if (attrs.has(AttrKey::KernelSizeH)) {
                kernel_h = attrs.get_int(AttrKey::KernelSizeH);
                kernel_w = attrs.has(AttrKey::KernelSizeW) ? attrs.get_int(AttrKey::KernelSizeW) : kernel_h;
            } else {
                kernel_h = kernel_w = attrs.get_int(AttrKey::KernelSize);
            }
            if (attrs.has(AttrKey::StrideH)) {
                stride_h = attrs.get_int(AttrKey::StrideH);
                stride_w = attrs.has(AttrKey::StrideW) ? attrs.get_int(AttrKey::StrideW) : stride_h;
            } else if (attrs.has(AttrKey::Stride)) {
                stride_h = stride_w = attrs.get_int(AttrKey::Stride);
            } else {
                stride_h = kernel_h; stride_w = kernel_w;
            }
            if (attrs.has(AttrKey::PaddingH)) {
                padding_h = attrs.get_int(AttrKey::PaddingH);
                padding_w = attrs.has(AttrKey::PaddingW) ? attrs.get_int(AttrKey::PaddingW) : padding_h;
            } else if (attrs.has(AttrKey::Padding)) {
                padding_h = padding_w = attrs.get_int(AttrKey::Padding);
            } else {
                padding_h = padding_w = 0;
            }

            bwd_attrs.set(AttrKey::KernelSizeH, kernel_h);
            bwd_attrs.set(AttrKey::KernelSizeW, kernel_w);
            bwd_attrs.set(AttrKey::StrideH, stride_h);
            bwd_attrs.set(AttrKey::StrideW, stride_w);
            bwd_attrs.set(AttrKey::PaddingH, padding_h);
            bwd_attrs.set(AttrKey::PaddingW, padding_w);
            bwd_attrs.set(AttrKey::CountIncludePad, true);

            // Create a dummy input tensor with the right shape for dispatchAvgPool2dBackward
            Tensor dummy_input({in_n, in_c, in_h, in_w}, inputs[0].dtype(), inputs[0].device());

            return {dispatchAvgPool2dBackward(inputs[0], dummy_input, bwd_attrs)};
        }
        throw std::invalid_argument("avg_pool2d_backward requires at least 1 input");
    }

    if (op_name == "max_pool2d_backward") {
        if (inputs.size() != 2) {
            throw std::invalid_argument("max_pool2d_backward requires 2 inputs (grad_output, indices)");
        }
        // Extract H_in and W_in from attributes (required for output shape)
        int64_t H_in = attrs.get_int(AttrKey::InputH);
        int64_t W_in = attrs.get_int(AttrKey::InputW);
        return {dispatchMaxPool2dBackwardWithIndices(inputs[0], inputs[1], H_in, W_in)};
    }

    // Conv2d forward operation
    if (op_name == "conv2d_forward") {
        if (inputs.size() < 2) {
            throw std::invalid_argument("conv2d_forward requires at least 2 inputs (input, weight)");
        }
        const Tensor* bias_ptr = inputs.size() >= 3 ? &inputs[2] : nullptr;
        return {dispatchConv2dForward(inputs[0], inputs[1], bias_ptr, attrs)};
    }

    // ConvTranspose2d forward operation
    if (op_name == "conv_transpose2d_forward") {
        if (inputs.size() < 2) {
            throw std::invalid_argument("conv_transpose2d_forward requires at least 2 inputs (input, weight)");
        }
        // CPU fallback for unsupported dtypes (Int32, etc.)
        if (inputs[0].dtype() != DType::Float32 && inputs[0].dtype() != DType::Float64 && inputs[0].dtype() != DType::Float16) {
            TENZOR_LOG_WARNING(std::format("Vulkan: No shader for conv_transpose2d_forward with dtype {}; falling back to CPU",
                                           dtype_name(inputs[0].dtype())));
            Device original_device = inputs[0].device();
            std::vector<Tensor> cpu_inputs;
            cpu_inputs.reserve(inputs.size());
            for (const auto& t : inputs) {
                cpu_inputs.push_back(t.to(Device::cpu()));
            }
            auto cpu_results = tenzor::dispatch(OpId::ConvTranspose2dForward, cpu_inputs, attrs);
            return {cpu_results[0].to(original_device)};
        }
        const Tensor* bias_ptr = (inputs.size() >= 3) ? &inputs[2] : nullptr;
        return {dispatchConvTranspose2dForward(inputs[0], inputs[1], bias_ptr, attrs)};
    }

    // Full operation - create tensor filled with specific value
    if (op_name == "full") {
        // Extract shape and value from attributes
        std::vector<int64_t> shape = attrs.get_int_list(AttrKey::Shape);
        float value = static_cast<float>(attrs.get_float(AttrKey::Value, 0.0));
        DType dtype = DType::Float32;  // Default dtype
        if (attrs.has(AttrKey::Dtype)) {
            auto dtype_str = attrs.get_string(AttrKey::Dtype);
            if (dtype_str == "int32" || dtype_str == "Int32") {
                dtype = DType::Int32;
            } else if (dtype_str == "float32" || dtype_str == "Float32") {
                dtype = DType::Float32;
            } else if (dtype_str == "float16" || dtype_str == "Float16") {
                dtype = DType::Float16;
            } else if (dtype_str == "bfloat16" || dtype_str == "BFloat16") {
                dtype = DType::BFloat16;
            } else if (dtype_str == "float64" || dtype_str == "Float64") {
                dtype = DType::Float64;
            } else if (dtype_str == "int64" || dtype_str == "Int64") {
                dtype = DType::Int64;
            } else if (dtype_str == "int8" || dtype_str == "Int8") {
                dtype = DType::Int8;
            } else if (dtype_str == "uint8" || dtype_str == "UInt8") {
                dtype = DType::UInt8;
            } else if (dtype_str == "bool" || dtype_str == "Bool") {
                dtype = DType::Bool;
            }
        }
        return {dispatchFull(shape, value, dtype)};
    }

    // Ones operation - create tensor filled with 1.0
    if (op_name == "ones") {
        std::vector<int64_t> shape = attrs.get_int_list(AttrKey::Shape);
        DType dtype = DType::Float32;
        if (attrs.has(AttrKey::Dtype)) {
            auto dtype_str = attrs.get_string(AttrKey::Dtype);
            if (dtype_str == "float32" || dtype_str == "Float32") dtype = DType::Float32;
            else if (dtype_str == "float16" || dtype_str == "Float16") dtype = DType::Float16;
            else if (dtype_str == "bfloat16" || dtype_str == "BFloat16") dtype = DType::BFloat16;
            else if (dtype_str == "int32" || dtype_str == "Int32") dtype = DType::Int32;
            else if (dtype_str == "int64" || dtype_str == "Int64") dtype = DType::Int64;
            else if (dtype_str == "float64" || dtype_str == "Float64") dtype = DType::Float64;
            else if (dtype_str == "int8" || dtype_str == "Int8") dtype = DType::Int8;
            else if (dtype_str == "uint8" || dtype_str == "UInt8") dtype = DType::UInt8;
            else if (dtype_str == "bool" || dtype_str == "Bool") dtype = DType::Bool;
        }
        return {dispatchOnes(shape, dtype)};
    }

    // Rand operation - create tensor filled with uniform random values [0, 1)
    if (op_name == "rand") {
        std::vector<int64_t> shape = attrs.get_int_list(AttrKey::Shape);
        DType dtype = DType::Float32;
        if (attrs.has(AttrKey::Dtype)) {
            auto dtype_str = attrs.get_string(AttrKey::Dtype);
            if (dtype_str == "float32" || dtype_str == "Float32") dtype = DType::Float32;
            else if (dtype_str == "float64" || dtype_str == "Float64") dtype = DType::Float64;
            else if (dtype_str == "float16" || dtype_str == "Float16") dtype = DType::Float16;
            else if (dtype_str == "bfloat16" || dtype_str == "BFloat16") dtype = DType::BFloat16;
        }
        return {dispatchRand(shape, dtype)};
    }

    // Randn operation - create tensor filled with normal random values
    if (op_name == "randn") {
        std::vector<int64_t> shape = attrs.get_int_list(AttrKey::Shape);
        DType dtype = DType::Float32;
        if (attrs.has(AttrKey::Dtype)) {
            auto dtype_str = attrs.get_string(AttrKey::Dtype);
            if (dtype_str == "float32" || dtype_str == "Float32") dtype = DType::Float32;
            else if (dtype_str == "float64" || dtype_str == "Float64") dtype = DType::Float64;
            else if (dtype_str == "float16" || dtype_str == "Float16") dtype = DType::Float16;
            else if (dtype_str == "bfloat16" || dtype_str == "BFloat16") dtype = DType::BFloat16;
        }
        return {dispatchRandn(shape, dtype)};
    }

    // Arange operation
    if (op_name == "arange") {
        float start = static_cast<float>(attrs.get_float(AttrKey::Start, 0.0));
        float end_val = static_cast<float>(attrs.get_float(AttrKey::End, 0.0));
        float step = static_cast<float>(attrs.get_float(AttrKey::Step, 1.0));
        DType dtype = DType::Float32;
        if (attrs.has(AttrKey::Dtype)) {
            auto dtype_str = attrs.get_string(AttrKey::Dtype);
            if (dtype_str == "float64" || dtype_str == "Float64") dtype = DType::Float64;
            else if (dtype_str == "float16" || dtype_str == "Float16") dtype = DType::Float16;
            else if (dtype_str == "bfloat16" || dtype_str == "BFloat16") dtype = DType::BFloat16;
            else if (dtype_str == "int32" || dtype_str == "Int32") dtype = DType::Int32;
            else if (dtype_str == "int64" || dtype_str == "Int64") dtype = DType::Int64;
            else if (dtype_str == "int8" || dtype_str == "Int8") dtype = DType::Int8;
            else if (dtype_str == "uint8" || dtype_str == "UInt8") dtype = DType::UInt8;
            else if (dtype_str == "bool" || dtype_str == "Bool") dtype = DType::Bool;
        }
        int32_t device_id = static_cast<int32_t>(attrs.get_int(AttrKey::DeviceId, 0));
        Device device = Device::vulkan(device_id);
        return {dispatchArange(start, end_val, step, dtype, device)};
    }

    // Linspace operation
    if (op_name == "linspace") {
        float start = static_cast<float>(attrs.get_float(AttrKey::Start, 0.0));
        float end_val = static_cast<float>(attrs.get_float(AttrKey::End, 1.0));
        int64_t steps = attrs.get_int(AttrKey::Steps, 100);
        DType dtype = DType::Float32;
        if (attrs.has(AttrKey::Dtype)) {
            auto dtype_str = attrs.get_string(AttrKey::Dtype);
            if (dtype_str == "float64" || dtype_str == "Float64") dtype = DType::Float64;
            else if (dtype_str == "float16" || dtype_str == "Float16") dtype = DType::Float16;
            else if (dtype_str == "bfloat16" || dtype_str == "BFloat16") dtype = DType::BFloat16;
            else if (dtype_str == "int32" || dtype_str == "Int32") dtype = DType::Int32;
            else if (dtype_str == "int64" || dtype_str == "Int64") dtype = DType::Int64;
            else if (dtype_str == "int8" || dtype_str == "Int8") dtype = DType::Int8;
            else if (dtype_str == "uint8" || dtype_str == "UInt8") dtype = DType::UInt8;
            else if (dtype_str == "bool" || dtype_str == "Bool") dtype = DType::Bool;
        }
        int32_t device_id = static_cast<int32_t>(attrs.get_int(AttrKey::DeviceId, 0));
        Device device = Device::vulkan(device_id);
        return {dispatchLinspace(start, end_val, steps, dtype, device)};
    }

    // Eye operation
    if (op_name == "eye") {
        int64_t n = attrs.get_int(AttrKey::N, 0);
        int64_t m = attrs.get_int(AttrKey::M, -1);
        DType dtype = DType::Float32;
        if (attrs.has(AttrKey::Dtype)) {
            auto dtype_str = attrs.get_string(AttrKey::Dtype);
            if (dtype_str == "float64" || dtype_str == "Float64") dtype = DType::Float64;
            else if (dtype_str == "float16" || dtype_str == "Float16") dtype = DType::Float16;
            else if (dtype_str == "bfloat16" || dtype_str == "BFloat16") dtype = DType::BFloat16;
            else if (dtype_str == "int32" || dtype_str == "Int32") dtype = DType::Int32;
            else if (dtype_str == "int64" || dtype_str == "Int64") dtype = DType::Int64;
            else if (dtype_str == "int8" || dtype_str == "Int8") dtype = DType::Int8;
            else if (dtype_str == "uint8" || dtype_str == "UInt8") dtype = DType::UInt8;
            else if (dtype_str == "bool" || dtype_str == "Bool") dtype = DType::Bool;
        }
        int32_t device_id = static_cast<int32_t>(attrs.get_int(AttrKey::DeviceId, 0));
        Device device = Device::vulkan(device_id);
        return {dispatchEye(n, m, dtype, device)};
    }

    // Repeat operation
    if (op_name == "repeat") {
        if (inputs.size() != 1) {
            throw std::invalid_argument("repeat operation requires exactly 1 input");
        }
        if (!attrs.has(AttrKey::Repeats)) {
            throw std::invalid_argument("repeat operation requires 'repeats' attribute");
        }
        std::vector<int64_t> repeats = attrs.get_int_list(AttrKey::Repeats);
        return {dispatchRepeat(inputs[0], repeats)};
    }

    // Masked select operation
    if (op_name == "masked_select") {
        if (inputs.size() != 2) {
            throw std::invalid_argument("masked_select operation requires exactly 2 inputs");
        }
        return {dispatchMaskedSelect(inputs[0], inputs[1])};
    }

    // Masked fill operation
    if (op_name == "masked_fill") {
        if (inputs.size() != 2) {
            throw std::invalid_argument("masked_fill operation requires exactly 2 inputs");
        }
        if (!attrs.has(AttrKey::Value)) {
            throw std::invalid_argument("masked_fill operation requires 'value' attribute");
        }
        float value = static_cast<float>(attrs.get_float(AttrKey::Value));
        return {dispatchMaskedFill(inputs[0], inputs[1], value)};
    }

    // Where operation
    if (op_name == "where") {
        if (inputs.size() != 3) {
            throw std::invalid_argument("where operation requires exactly 3 inputs");
        }
        return {dispatchWhere(inputs[0], inputs[1], inputs[2])};
    }

    // ========================================================================
    // Fused Operations (composed from existing operations)
    // ========================================================================

    // Fused Linear + ReLU: matmul(input, weight^T) + bias + relu
    if (op_name == "fused_linear_relu") {
        if (inputs.size() < 2) {
            throw std::invalid_argument("fused_linear_relu requires at least 2 inputs (input, weight)");
        }
        bool has_bias = attrs.get_bool(AttrKey::HasBias, false);

        auto input_shape = inputs[0].shape();
        int64_t in_features = input_shape[input_shape.size() - 1];
        int64_t out_features = inputs[1].shape()[0];

        // Flatten input to 2D: (..., in_features) -> (batch_size, in_features)
        int64_t batch_size = 1;
        for (size_t i = 0; i < input_shape.size() - 1; ++i) {
            batch_size *= input_shape[i];
        }
        Tensor input_2d = inputs[0].reshape({batch_size, in_features});

        // Transpose weight as a view (swap shape/strides, same data)
        // dispatchMatmul will call dispatchContiguous if needed
        Tensor weight_t = inputs[1].transpose(0, 1);

        // MatMul: input_2d @ weight^T -> (batch_size, out_features)
        Tensor mm_result = dispatchMatmul(input_2d, weight_t);

        // Add bias if present
        if (has_bias && inputs.size() > 2) {
            mm_result = dispatchBinaryOp("add", mm_result, inputs[2]);
        }

        // ReLU activation
        Tensor result = dispatchActivation("relu", mm_result, 0, 0.0f);

        // Reshape back to original batch dimensions
        std::vector<int64_t> out_shape(input_shape.begin(), input_shape.end() - 1);
        out_shape.push_back(out_features);
        result = result.reshape(out_shape);

        return {result};
    }

    // Fused BatchNorm + ReLU
    if (op_name == "fused_batchnorm_relu") {
        if (inputs.size() < 5) {
            throw std::invalid_argument("fused_batchnorm_relu requires 5 inputs (input, mean, var, weight, bias)");
        }

        float eps = static_cast<float>(attrs.get_float(AttrKey::Eps, 1e-5));
        auto orig_shape = inputs[0].shape();

        // Reshape to 4D if needed: (N, C, ...) -> (N, C, H, W) where H*W = spatial_size
        Tensor input_4d = inputs[0];
        bool needs_reshape = (orig_shape.size() != 4);
        if (needs_reshape) {
            int64_t N = orig_shape[0];
            int64_t C = orig_shape[1];
            int64_t spatial = 1;
            for (size_t i = 2; i < orig_shape.size(); ++i) {
                spatial *= orig_shape[i];
            }
            input_4d = inputs[0].reshape({N, C, spatial, 1});
        }

        // BatchNorm forward
        Tensor bn_result = dispatchBatchNorm2dForward(input_4d, inputs[1], inputs[2],
                                                       &inputs[3], &inputs[4], eps);

        // Reshape back if needed
        if (needs_reshape) {
            std::vector<int64_t> shape_vec(orig_shape.begin(), orig_shape.end());
            bn_result = bn_result.reshape(shape_vec);
        }

        // ReLU activation
        Tensor result = dispatchActivation("relu", bn_result, 0, 0.0f);
        return {result};
    }

    // Fused Add + ReLU
    if (op_name == "fused_add_relu") {
        if (inputs.size() != 2) {
            throw std::invalid_argument("fused_add_relu requires 2 inputs");
        }
        Tensor add_result = dispatchBinaryOp("add", inputs[0], inputs[1]);
        return {dispatchActivation("relu", add_result, 0, 0.0f)};
    }

    // Fused GELU
    if (op_name == "fused_gelu") {
        if (inputs.size() != 1) {
            throw std::invalid_argument("fused_gelu requires 1 input");
        }
        return {dispatchActivation("gelu", inputs[0], 3, 0.0f)};
    }

    // Fused Layer Norm
    if (op_name == "fused_layer_norm") {
        if (inputs.size() < 3) {
            throw std::invalid_argument("fused_layer_norm requires 3 inputs (input, weight, bias)");
        }
        // Parse normalized_shape from comma-separated string to compute total size
        auto ns_str = attrs.get_string(AttrKey::NormalizedShape);
        int64_t normalized_size = 1;
        std::string ns_s{ns_str};
        std::stringstream ss(ns_s);
        std::string token;
        while (std::getline(ss, token, ',')) {
            normalized_size *= std::stoll(token);
        }
        float eps = static_cast<float>(attrs.get_float(AttrKey::Eps, 1e-5));
        Tensor result = dispatchLayerNorm(inputs[0], normalized_size,
                                           &inputs[1], &inputs[2], eps);
        return {result};
    }

    // ========================================================================
    // Interpolation Operation
    // ========================================================================
    if (op_name == "interpolate") {
        if (inputs.size() != 1) {
            throw std::invalid_argument("interpolate requires 1 input");
        }
        return {dispatchInterpolate(inputs[0], attrs)};
    }

    // ========================================================================
    // ROI Align Operations
    // ========================================================================
    if (op_name == "roi_align_forward") {
        if (inputs.size() < 2) {
            throw std::invalid_argument("roi_align_forward requires 2 inputs (features, rois)");
        }
        return {dispatchROIAlignForward(inputs[0], inputs[1], attrs)};
    }

    if (op_name == "roi_align_backward") {
        if (inputs.size() < 2) {
            throw std::invalid_argument("roi_align_backward requires 2 inputs (grad_output, rois)");
        }
        return {dispatchROIAlignBackward(inputs[0], inputs[1], attrs)};
    }

    // ========================================================================
    // LayerNorm / GroupNorm string dispatch
    // ========================================================================
    if (op_name == "layer_norm") {
        if (inputs.size() < 1) {
            throw std::invalid_argument("layer_norm requires at least 1 input");
        }
        int64_t normalized_size = 1;
        if (attrs.has(AttrKey::NormalizedShape)) {
            auto ns_str = attrs.get_string(AttrKey::NormalizedShape);
            std::string ns_s{ns_str};
            std::stringstream ss(ns_s);
            std::string token;
            while (std::getline(ss, token, ',')) {
                normalized_size *= std::stoll(token);
            }
        } else {
            normalized_size = inputs[0].shape().back();
        }
        float eps = static_cast<float>(attrs.get_float(AttrKey::Eps, 1e-5));
        const Tensor* gamma = (inputs.size() > 1) ? &inputs[1] : nullptr;
        const Tensor* beta = (inputs.size() > 2) ? &inputs[2] : nullptr;
        return {dispatchLayerNorm(inputs[0], normalized_size, gamma, beta, eps)};
    }

    if (op_name == "layer_norm_backward") {
        // inputs: [grad_output, input, mean, rstd, weight]
        if (inputs.size() < 5) {
            throw std::invalid_argument("layer_norm_backward requires 5 inputs (grad_output, input, mean, rstd, weight)");
        }
        int64_t normalized_shape = inputs[0].shape().back();
        if (attrs.has(AttrKey::NormalizedShape)) {
            normalized_shape = attrs.get_int(AttrKey::NormalizedShape);
        }
        auto [grad_input, grad_weight, grad_bias] = dispatchLayerNormBackward(
            inputs[0], inputs[1], inputs[2], inputs[3], &inputs[4], normalized_shape);
        return {grad_input, grad_weight, grad_bias};
    }

    if (op_name == "group_norm") {
        if (inputs.size() < 1) {
            throw std::invalid_argument("group_norm requires at least 1 input");
        }
        int64_t num_groups = attrs.get_int(AttrKey::NumGroups, 1);
        float eps = static_cast<float>(attrs.get_float(AttrKey::Eps, 1e-5));
        const Tensor* gamma = (inputs.size() > 1) ? &inputs[1] : nullptr;
        const Tensor* beta = (inputs.size() > 2) ? &inputs[2] : nullptr;
        return dispatchGroupNorm(inputs[0], num_groups, gamma, beta, eps);
    }

    if (op_name == "group_norm_backward") {
        // inputs: [grad_output, input, mean, rstd, weight]
        if (inputs.size() < 5) {
            throw std::invalid_argument("group_norm_backward requires 5 inputs (grad_output, input, mean, rstd, weight)");
        }
        int64_t num_groups = attrs.get_int(AttrKey::NumGroups, 1);
        auto [grad_input, grad_weight, grad_bias] = dispatchGroupNormBackward(
            inputs[0], inputs[1], inputs[2], inputs[3], &inputs[4], num_groups);
        return {grad_input, grad_weight, grad_bias};
    }

    // ========================================================================
    // Embedding Backward
    // ========================================================================
    if (op_name == "embedding_backward") {
        // inputs: [grad_output, indices]
        if (inputs.size() < 2) {
            throw std::invalid_argument("embedding_backward requires 2 inputs (grad_output, indices)");
        }
        int64_t num_embeddings = attrs.get_int(AttrKey::NumEmbeddings, 0);
        int64_t embedding_dim = inputs[0].shape().back();
        return {dispatchEmbeddingBackward(inputs[0], inputs[1], num_embeddings, embedding_dim)};
    }

    // ========================================================================
    // RMSNorm Forward and Backward
    // ========================================================================
    if (op_name == "fused_rms_norm") {
        if (inputs.size() < 2) {
            throw std::invalid_argument("fused_rms_norm requires 2 inputs (input, weight)");
        }
        float eps = static_cast<float>(attrs.get_float(AttrKey::Eps, 1e-5));
        int64_t normalized_shape = inputs[0].shape().back();
        auto [output, rrms] = dispatchRMSNorm(inputs[0], inputs[1], normalized_shape, eps);
        return {output, rrms};
    }

    if (op_name == "rms_norm_backward") {
        // inputs: [grad_output, input, rrms, weight]
        if (inputs.size() < 4) {
            throw std::invalid_argument("rms_norm_backward requires 4 inputs (grad_output, input, rrms, weight)");
        }
        int64_t normalized_shape = inputs[0].shape().back();
        if (attrs.has(AttrKey::NormalizedShape)) {
            normalized_shape = attrs.get_int(AttrKey::NormalizedShape);
        }
        auto [grad_input, grad_weight] = dispatchRMSNormBackward(
            inputs[0], inputs[1], inputs[2], inputs[3], normalized_shape);
        return {grad_input, grad_weight};
    }

    // ========================================================================
    // Phase 3: Nonzero, OneHot, BoxIoU
    // ========================================================================
    if (op_name == "nonzero") {
        if (inputs.size() < 1) {
            throw std::invalid_argument("nonzero requires 1 input");
        }
        return {dispatchNonzero(inputs[0])};
    }

    if (op_name == "one_hot") {
        if (inputs.size() < 1) {
            throw std::invalid_argument("one_hot requires 1 input (indices)");
        }
        int64_t num_classes = attrs.get_int(AttrKey::NumClasses, 10);
        return {dispatchOneHot(inputs[0], num_classes)};
    }

    if (op_name == "box_iou") {
        if (inputs.size() < 2) {
            throw std::invalid_argument("box_iou requires 2 inputs (boxes1, boxes2)");
        }
        int64_t iou_type = attrs.get_int(AttrKey::IouType, 0);
        return {dispatchBoxIoU(inputs[0], inputs[1], iou_type)};
    }

    // ========================================================================
    // Phase 4: Fused Optimizer Steps
    // ========================================================================
    if (op_name == "fused_rmsprop_step") {
        // inputs: [grad, param, square_avg, momentum_buf, grad_avg]
        if (inputs.size() < 3) {
            throw std::invalid_argument("fused_rmsprop_step requires at least 3 inputs");
        }
        int64_t numel = inputs[0].numel();
        float lr = static_cast<float>(attrs.get_float(AttrKey::Lr, 0.01));
        float alpha = static_cast<float>(attrs.get_float(AttrKey::Alpha, 0.99));
        float eps = static_cast<float>(attrs.get_float(AttrKey::Eps, 1e-8));
        float weight_decay = static_cast<float>(attrs.get_float(AttrKey::WeightDecay, 0.0));
        float momentum = static_cast<float>(attrs.get_float(AttrKey::Momentum, 0.0));
        bool centered = attrs.get_bool(AttrKey::Centered, false);

        int32_t device_id = inputs[0].device().index;
        auto* pipeline = getPipeline("fused_rmsprop_step", device_id);

        bool has_momentum = (inputs.size() > 3 && momentum > 0.0f);
        bool has_grad_avg = (inputs.size() > 4 && centered);

        const void* buf_grad = inputs[0].data_ptr();
        const void* buf_param = inputs[1].data_ptr();
        const void* buf_sq_avg = inputs[2].data_ptr();

        size_t buf_size = numel * inputs[0].dtype_size();
        std::vector<std::pair<uint32_t, const void*>> bindings = {
            {0, buf_grad}, {1, buf_param}, {2, buf_sq_avg},
        };
        std::vector<size_t> sizes = {buf_size, buf_size, buf_size};

        if (has_momentum) {
            bindings.push_back({3, inputs[3].data_ptr()});
            sizes.push_back(buf_size);
        } else {
            bindings.push_back({3, buf_param}); // dummy
            sizes.push_back(buf_size);
        }
        if (has_grad_avg) {
            bindings.push_back({4, inputs[4].data_ptr()});
            sizes.push_back(buf_size);
        } else {
            bindings.push_back({4, buf_param}); // dummy
            sizes.push_back(buf_size);
        }

        VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

        struct PushConstants {
            uint32_t numel;
            float lr;
            float alpha;
            float eps;
            float weight_decay;
            float momentum;
            uint32_t centered;
            uint32_t has_momentum;
        } pc;
        pc.numel = static_cast<uint32_t>(numel);
        pc.lr = lr;
        pc.alpha = alpha;
        pc.eps = eps;
        pc.weight_decay = weight_decay;
        pc.momentum = momentum;
        pc.centered = centered ? 1u : 0u;
        pc.has_momentum = has_momentum ? 1u : 0u;

        uint32_t workgroups = div_wg(numel, devices_[device_id].workgroupSize);
        VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
        vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                               pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
        vkCmdPushConstants(cmdBuffer, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                          0, sizeof(PushConstants), &pc);
        vkCmdDispatch(cmdBuffer, workgroups, 1, 1);
        insertComputeBarrier(cmdBuffer);
        endSingleTimeCommands(cmdBuffer, device_id);

        return {};  // In-place update, no outputs
    }

    if (op_name == "fused_adadelta_step") {
        // inputs: [grad, param, square_avg, acc_delta]
        if (inputs.size() < 4) {
            throw std::invalid_argument("fused_adadelta_step requires 4 inputs");
        }
        int64_t numel = inputs[0].numel();
        float lr = static_cast<float>(attrs.get_float(AttrKey::Lr, 1.0));
        float rho = static_cast<float>(attrs.get_float(AttrKey::Rho, 0.9));
        float eps = static_cast<float>(attrs.get_float(AttrKey::Eps, 1e-6));
        float weight_decay = static_cast<float>(attrs.get_float(AttrKey::WeightDecay, 0.0));

        int32_t device_id = inputs[0].device().index;
        auto* pipeline = getPipeline("fused_adadelta_step", device_id);

        size_t buf_size = numel * inputs[0].dtype_size();
        std::vector<std::pair<uint32_t, const void*>> bindings = {
            {0, inputs[0].data_ptr()},
            {1, inputs[1].data_ptr()},
            {2, inputs[2].data_ptr()},
            {3, inputs[3].data_ptr()},
        };
        std::vector<size_t> sizes = {buf_size, buf_size, buf_size, buf_size};

        VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

        struct PushConstants {
            uint32_t numel;
            float lr;
            float rho;
            float eps;
            float weight_decay;
            uint32_t padding0;
            uint32_t padding1;
            uint32_t padding2;
        } pc;
        pc.numel = static_cast<uint32_t>(numel);
        pc.lr = lr;
        pc.rho = rho;
        pc.eps = eps;
        pc.weight_decay = weight_decay;
        pc.padding0 = 0;
        pc.padding1 = 0;
        pc.padding2 = 0;

        uint32_t workgroups = div_wg(numel, devices_[device_id].workgroupSize);
        VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
        vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                               pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
        vkCmdPushConstants(cmdBuffer, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                          0, sizeof(PushConstants), &pc);
        vkCmdDispatch(cmdBuffer, workgroups, 1, 1);
        insertComputeBarrier(cmdBuffer);
        endSingleTimeCommands(cmdBuffer, device_id);

        return {};
    }

    if (op_name == "fused_adagrad_step") {
        // inputs: [grad, param, sum_sq]
        if (inputs.size() < 3) {
            throw std::invalid_argument("fused_adagrad_step requires 3 inputs");
        }
        int64_t numel = inputs[0].numel();
        float lr = static_cast<float>(attrs.get_float(AttrKey::Lr, 0.01));
        float eps = static_cast<float>(attrs.get_float(AttrKey::Eps, 1e-10));
        float weight_decay = static_cast<float>(attrs.get_float(AttrKey::WeightDecay, 0.0));

        int32_t device_id = inputs[0].device().index;
        auto* pipeline = getPipeline("fused_adagrad_step", device_id);

        size_t buf_size = numel * inputs[0].dtype_size();
        std::vector<std::pair<uint32_t, const void*>> bindings = {
            {0, inputs[0].data_ptr()},
            {1, inputs[1].data_ptr()},
            {2, inputs[2].data_ptr()},
        };
        std::vector<size_t> sizes = {buf_size, buf_size, buf_size};

        VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

        struct PushConstants {
            uint32_t numel;
            float lr;
            float eps;
            float weight_decay;
        } pc;
        pc.numel = static_cast<uint32_t>(numel);
        pc.lr = lr;
        pc.eps = eps;
        pc.weight_decay = weight_decay;

        uint32_t workgroups = div_wg(numel, devices_[device_id].workgroupSize);
        VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
        vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                               pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
        vkCmdPushConstants(cmdBuffer, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                          0, sizeof(PushConstants), &pc);
        vkCmdDispatch(cmdBuffer, workgroups, 1, 1);
        insertComputeBarrier(cmdBuffer);
        endSingleTimeCommands(cmdBuffer, device_id);

        return {};
    }

    throw std::runtime_error("VulkanBackend: Operation '" + op_name + "' not implemented");
    } catch (const std::out_of_range& e) {
        throw std::runtime_error("VulkanBackend::dispatch '" + op_name +
                               "' failed with out_of_range: " + e.what() +
                               " (attrs count: " + std::to_string(attrs.size()) + ")");
    }
}

} // namespace tenzor
