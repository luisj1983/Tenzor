/**
 * @file vulkan_dispatch.cpp
 * @brief Vulkan backend top-level dispatch() method - routes op_name strings to dispatchXXX methods
 */

#include "vulkan_helpers.hpp"
#include "tenzor/backend/fast_dispatch.hpp"
#include "tenzor/backend/op_attributes.hpp"
#include "tenzor/utils/logging.hpp"

#include <functional>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace tenzor {

// Handler signature: (backend, op_name, inputs, attrs) -> vector<Tensor>
// op_name is passed for error messages and for ops that forward it (e.g. dispatchBinaryOp).
using VkDispatchHandler = std::function<std::vector<Tensor>(
    VulkanBackend*, const std::string&, std::span<const Tensor>, const OpAttributes&)>;

static const std::unordered_map<std::string, VkDispatchHandler>& get_dispatch_table() {
    static const std::unordered_map<std::string, VkDispatchHandler> table = {
        // ==================================================================
        // Binary operations
        // ==================================================================
        {"add", [](VulkanBackend* b, const std::string& op, std::span<const Tensor> inputs, const OpAttributes&) {
            if (inputs.size() != 2) throw std::invalid_argument(op + " requires 2 inputs");
            return std::vector<Tensor>{b->dispatchBinaryOp(op, inputs[0], inputs[1])};
        }},
        {"sub", [](VulkanBackend* b, const std::string& op, std::span<const Tensor> inputs, const OpAttributes&) {
            if (inputs.size() != 2) throw std::invalid_argument(op + " requires 2 inputs");
            return std::vector<Tensor>{b->dispatchBinaryOp(op, inputs[0], inputs[1])};
        }},
        {"mul", [](VulkanBackend* b, const std::string& op, std::span<const Tensor> inputs, const OpAttributes&) {
            if (inputs.size() != 2) throw std::invalid_argument(op + " requires 2 inputs");
            return std::vector<Tensor>{b->dispatchBinaryOp(op, inputs[0], inputs[1])};
        }},
        {"div", [](VulkanBackend* b, const std::string& op, std::span<const Tensor> inputs, const OpAttributes&) {
            if (inputs.size() != 2) throw std::invalid_argument(op + " requires 2 inputs");
            return std::vector<Tensor>{b->dispatchBinaryOp(op, inputs[0], inputs[1])};
        }},

        // ==================================================================
        // In-place binary operations
        // ==================================================================
        {"add_inplace", [](VulkanBackend* b, const std::string& op, std::span<const Tensor> inputs, const OpAttributes&) {
            if (inputs.size() != 2) throw std::invalid_argument(op + " requires 2 inputs");
            std::string base_op = op.substr(0, op.find("_inplace"));
            Tensor result = b->dispatchBinaryOp(base_op, inputs[0], inputs[1]);
            size_t bytes = result.numel() * result.dtype_size();
            if (bytes > 0) {
                void* dst = const_cast<void*>(inputs[0].data_ptr());
                b->copy(dst, result.data_ptr(), bytes, CopyKind::DeviceToDevice);
            }
            return std::vector<Tensor>{inputs[0]};
        }},
        {"sub_inplace", [](VulkanBackend* b, const std::string& op, std::span<const Tensor> inputs, const OpAttributes&) {
            if (inputs.size() != 2) throw std::invalid_argument(op + " requires 2 inputs");
            std::string base_op = op.substr(0, op.find("_inplace"));
            Tensor result = b->dispatchBinaryOp(base_op, inputs[0], inputs[1]);
            size_t bytes = result.numel() * result.dtype_size();
            if (bytes > 0) {
                void* dst = const_cast<void*>(inputs[0].data_ptr());
                b->copy(dst, result.data_ptr(), bytes, CopyKind::DeviceToDevice);
            }
            return std::vector<Tensor>{inputs[0]};
        }},
        {"mul_inplace", [](VulkanBackend* b, const std::string& op, std::span<const Tensor> inputs, const OpAttributes&) {
            if (inputs.size() != 2) throw std::invalid_argument(op + " requires 2 inputs");
            std::string base_op = op.substr(0, op.find("_inplace"));
            Tensor result = b->dispatchBinaryOp(base_op, inputs[0], inputs[1]);
            size_t bytes = result.numel() * result.dtype_size();
            if (bytes > 0) {
                void* dst = const_cast<void*>(inputs[0].data_ptr());
                b->copy(dst, result.data_ptr(), bytes, CopyKind::DeviceToDevice);
            }
            return std::vector<Tensor>{inputs[0]};
        }},
        {"div_inplace", [](VulkanBackend* b, const std::string& op, std::span<const Tensor> inputs, const OpAttributes&) {
            if (inputs.size() != 2) throw std::invalid_argument(op + " requires 2 inputs");
            std::string base_op = op.substr(0, op.find("_inplace"));
            Tensor result = b->dispatchBinaryOp(base_op, inputs[0], inputs[1]);
            size_t bytes = result.numel() * result.dtype_size();
            if (bytes > 0) {
                void* dst = const_cast<void*>(inputs[0].data_ptr());
                b->copy(dst, result.data_ptr(), bytes, CopyKind::DeviceToDevice);
            }
            return std::vector<Tensor>{inputs[0]};
        }},

        // ==================================================================
        // In-place activation operations
        // ==================================================================
        {"relu_inplace", [](VulkanBackend* b, const std::string& op, std::span<const Tensor> inputs, const OpAttributes&) {
            if (inputs.size() < 1) throw std::invalid_argument(op + " requires 1 input");
            Tensor result = b->dispatchActivation("relu", inputs[0], 0, 0.0f);
            size_t bytes = result.numel() * result.dtype_size();
            if (bytes > 0) { void* dst = const_cast<void*>(inputs[0].data_ptr()); b->copy(dst, result.data_ptr(), bytes, CopyKind::DeviceToDevice); }
            return std::vector<Tensor>{inputs[0]};
        }},
        {"sigmoid_inplace", [](VulkanBackend* b, const std::string& op, std::span<const Tensor> inputs, const OpAttributes&) {
            if (inputs.size() < 1) throw std::invalid_argument(op + " requires 1 input");
            Tensor result = b->dispatchActivation("sigmoid", inputs[0], 1, 0.0f);
            size_t bytes = result.numel() * result.dtype_size();
            if (bytes > 0) { void* dst = const_cast<void*>(inputs[0].data_ptr()); b->copy(dst, result.data_ptr(), bytes, CopyKind::DeviceToDevice); }
            return std::vector<Tensor>{inputs[0]};
        }},
        {"tanh_inplace", [](VulkanBackend* b, const std::string& op, std::span<const Tensor> inputs, const OpAttributes&) {
            if (inputs.size() < 1) throw std::invalid_argument(op + " requires 1 input");
            Tensor result = b->dispatchActivation("tanh", inputs[0], 2, 0.0f);
            size_t bytes = result.numel() * result.dtype_size();
            if (bytes > 0) { void* dst = const_cast<void*>(inputs[0].data_ptr()); b->copy(dst, result.data_ptr(), bytes, CopyKind::DeviceToDevice); }
            return std::vector<Tensor>{inputs[0]};
        }},
        {"gelu_inplace", [](VulkanBackend* b, const std::string& op, std::span<const Tensor> inputs, const OpAttributes&) {
            if (inputs.size() < 1) throw std::invalid_argument(op + " requires 1 input");
            Tensor result = b->dispatchActivation("gelu", inputs[0], 3, 0.0f);
            size_t bytes = result.numel() * result.dtype_size();
            if (bytes > 0) { void* dst = const_cast<void*>(inputs[0].data_ptr()); b->copy(dst, result.data_ptr(), bytes, CopyKind::DeviceToDevice); }
            return std::vector<Tensor>{inputs[0]};
        }},
        {"leaky_relu_inplace", [](VulkanBackend* b, const std::string& op, std::span<const Tensor> inputs, const OpAttributes& attrs) {
            if (inputs.size() < 1) throw std::invalid_argument(op + " requires 1 input");
            float alpha = static_cast<float>(attrs.get_float(AttrKey::Alpha, 0.01));
            Tensor result = b->dispatchActivation("leaky_relu", inputs[0], 4, alpha);
            size_t bytes = result.numel() * result.dtype_size();
            if (bytes > 0) { void* dst = const_cast<void*>(inputs[0].data_ptr()); b->copy(dst, result.data_ptr(), bytes, CopyKind::DeviceToDevice); }
            return std::vector<Tensor>{inputs[0]};
        }},

        // ==================================================================
        // Comparison operations
        // ==================================================================
        {"eq", [](VulkanBackend* b, const std::string& op, std::span<const Tensor> inputs, const OpAttributes&) {
            if (inputs.size() != 2) throw std::invalid_argument(op + " requires 2 inputs");
            return std::vector<Tensor>{b->dispatchComparisonOp(op, inputs[0], inputs[1])};
        }},
        {"ne", [](VulkanBackend* b, const std::string& op, std::span<const Tensor> inputs, const OpAttributes&) {
            if (inputs.size() != 2) throw std::invalid_argument(op + " requires 2 inputs");
            return std::vector<Tensor>{b->dispatchComparisonOp(op, inputs[0], inputs[1])};
        }},
        {"lt", [](VulkanBackend* b, const std::string& op, std::span<const Tensor> inputs, const OpAttributes&) {
            if (inputs.size() != 2) throw std::invalid_argument(op + " requires 2 inputs");
            return std::vector<Tensor>{b->dispatchComparisonOp(op, inputs[0], inputs[1])};
        }},
        {"le", [](VulkanBackend* b, const std::string& op, std::span<const Tensor> inputs, const OpAttributes&) {
            if (inputs.size() != 2) throw std::invalid_argument(op + " requires 2 inputs");
            return std::vector<Tensor>{b->dispatchComparisonOp(op, inputs[0], inputs[1])};
        }},
        {"gt", [](VulkanBackend* b, const std::string& op, std::span<const Tensor> inputs, const OpAttributes&) {
            if (inputs.size() != 2) throw std::invalid_argument(op + " requires 2 inputs");
            return std::vector<Tensor>{b->dispatchComparisonOp(op, inputs[0], inputs[1])};
        }},
        {"ge", [](VulkanBackend* b, const std::string& op, std::span<const Tensor> inputs, const OpAttributes&) {
            if (inputs.size() != 2) throw std::invalid_argument(op + " requires 2 inputs");
            return std::vector<Tensor>{b->dispatchComparisonOp(op, inputs[0], inputs[1])};
        }},

        // ==================================================================
        // Activation functions
        // ==================================================================
        {"relu", [](VulkanBackend* b, const std::string& op, std::span<const Tensor> inputs, const OpAttributes&) {
            if (inputs.size() != 1) throw std::invalid_argument(op + " requires 1 input");
            return std::vector<Tensor>{b->dispatchActivation("relu", inputs[0], 0, 0.0f)};
        }},
        {"sigmoid", [](VulkanBackend* b, const std::string& op, std::span<const Tensor> inputs, const OpAttributes&) {
            if (inputs.size() != 1) throw std::invalid_argument(op + " requires 1 input");
            return std::vector<Tensor>{b->dispatchActivation("sigmoid", inputs[0], 1, 0.0f)};
        }},
        {"tanh", [](VulkanBackend* b, const std::string& op, std::span<const Tensor> inputs, const OpAttributes&) {
            if (inputs.size() != 1) throw std::invalid_argument(op + " requires 1 input");
            return std::vector<Tensor>{b->dispatchActivation("tanh", inputs[0], 2, 0.0f)};
        }},
        {"gelu", [](VulkanBackend* b, const std::string&, std::span<const Tensor> inputs, const OpAttributes&) {
            if (inputs.size() != 1) throw std::invalid_argument("gelu requires 1 input");
            return std::vector<Tensor>{b->dispatchActivation("gelu", inputs[0], 3, 0.0f)};
        }},
        {"leaky_relu", [](VulkanBackend* b, const std::string&, std::span<const Tensor> inputs, const OpAttributes& attrs) {
            if (inputs.size() != 1) throw std::invalid_argument("leaky_relu requires 1 input");
            float alpha = static_cast<float>(attrs.get_float(AttrKey::Alpha, 0.01));
            return std::vector<Tensor>{b->dispatchActivation("leaky_relu", inputs[0], 4, alpha)};
        }},
        {"swish", [](VulkanBackend* b, const std::string&, std::span<const Tensor> inputs, const OpAttributes&) {
            if (inputs.size() != 1) throw std::invalid_argument("swish requires 1 input");
            return std::vector<Tensor>{b->dispatchActivation("swish", inputs[0], 5, 0.0f)};
        }},
        {"elu", [](VulkanBackend* b, const std::string&, std::span<const Tensor> inputs, const OpAttributes& attrs) {
            if (inputs.size() != 1) throw std::invalid_argument("elu requires 1 input");
            float alpha = static_cast<float>(attrs.get_float(AttrKey::Alpha, 1.0));
            return std::vector<Tensor>{b->dispatchActivation("elu", inputs[0], 6, alpha)};
        }},
        {"selu", [](VulkanBackend* b, const std::string&, std::span<const Tensor> inputs, const OpAttributes&) {
            if (inputs.size() != 1) throw std::invalid_argument("selu requires 1 input");
            return std::vector<Tensor>{b->dispatchActivation("selu", inputs[0], 7, 0.0f)};
        }},
        {"mish", [](VulkanBackend* b, const std::string&, std::span<const Tensor> inputs, const OpAttributes&) {
            if (inputs.size() != 1) throw std::invalid_argument("mish requires 1 input");
            return std::vector<Tensor>{b->dispatchActivation("mish", inputs[0], 8, 0.0f)};
        }},
        {"softplus", [](VulkanBackend* b, const std::string&, std::span<const Tensor> inputs, const OpAttributes& attrs) {
            if (inputs.size() != 1) throw std::invalid_argument("softplus requires 1 input");
            float beta = static_cast<float>(attrs.get_float(AttrKey::Beta, 1.0));
            return std::vector<Tensor>{b->dispatchActivation("softplus", inputs[0], 9, beta)};
        }},

        // ==================================================================
        // Unary math operations
        // ==================================================================
        {"sqrt", [](VulkanBackend* b, const std::string& op, std::span<const Tensor> inputs, const OpAttributes&) {
            if (inputs.size() != 1) throw std::invalid_argument(op + " requires 1 input");
            return std::vector<Tensor>{b->dispatchUnaryOp(op, inputs[0])};
        }},
        {"exp", [](VulkanBackend* b, const std::string& op, std::span<const Tensor> inputs, const OpAttributes&) {
            if (inputs.size() != 1) throw std::invalid_argument(op + " requires 1 input");
            return std::vector<Tensor>{b->dispatchUnaryOp(op, inputs[0])};
        }},
        {"log", [](VulkanBackend* b, const std::string& op, std::span<const Tensor> inputs, const OpAttributes&) {
            if (inputs.size() != 1) throw std::invalid_argument(op + " requires 1 input");
            return std::vector<Tensor>{b->dispatchUnaryOp(op, inputs[0])};
        }},
        {"neg", [](VulkanBackend* b, const std::string& op, std::span<const Tensor> inputs, const OpAttributes&) {
            if (inputs.size() != 1) throw std::invalid_argument(op + " requires 1 input");
            return std::vector<Tensor>{b->dispatchUnaryOp(op, inputs[0])};
        }},
        {"abs", [](VulkanBackend* b, const std::string& op, std::span<const Tensor> inputs, const OpAttributes&) {
            if (inputs.size() != 1) throw std::invalid_argument(op + " requires 1 input");
            return std::vector<Tensor>{b->dispatchUnaryOp(op, inputs[0])};
        }},
        {"sign", [](VulkanBackend* b, const std::string& op, std::span<const Tensor> inputs, const OpAttributes&) {
            if (inputs.size() != 1) throw std::invalid_argument(op + " requires 1 input");
            return std::vector<Tensor>{b->dispatchUnaryOp(op, inputs[0])};
        }},
        {"floor", [](VulkanBackend* b, const std::string& op, std::span<const Tensor> inputs, const OpAttributes&) {
            if (inputs.size() != 1) throw std::invalid_argument(op + " requires 1 input");
            return std::vector<Tensor>{b->dispatchUnaryOp(op, inputs[0])};
        }},
        {"ceil", [](VulkanBackend* b, const std::string& op, std::span<const Tensor> inputs, const OpAttributes&) {
            if (inputs.size() != 1) throw std::invalid_argument(op + " requires 1 input");
            return std::vector<Tensor>{b->dispatchUnaryOp(op, inputs[0])};
        }},
        {"round", [](VulkanBackend* b, const std::string& op, std::span<const Tensor> inputs, const OpAttributes&) {
            if (inputs.size() != 1) throw std::invalid_argument(op + " requires 1 input");
            return std::vector<Tensor>{b->dispatchUnaryOp(op, inputs[0])};
        }},
        {"trunc", [](VulkanBackend* b, const std::string& op, std::span<const Tensor> inputs, const OpAttributes&) {
            if (inputs.size() != 1) throw std::invalid_argument(op + " requires 1 input");
            return std::vector<Tensor>{b->dispatchUnaryOp(op, inputs[0])};
        }},
        {"reciprocal", [](VulkanBackend* b, const std::string& op, std::span<const Tensor> inputs, const OpAttributes&) {
            if (inputs.size() != 1) throw std::invalid_argument(op + " requires 1 input");
            return std::vector<Tensor>{b->dispatchUnaryOp(op, inputs[0])};
        }},

        // ==================================================================
        // Trigonometric operations
        // ==================================================================
        {"sin", [](VulkanBackend* b, const std::string& op, std::span<const Tensor> inputs, const OpAttributes&) {
            if (inputs.size() != 1) throw std::invalid_argument(op + " requires 1 input");
            return std::vector<Tensor>{b->dispatchTrigonometricOp(op, inputs[0])};
        }},
        {"cos", [](VulkanBackend* b, const std::string& op, std::span<const Tensor> inputs, const OpAttributes&) {
            if (inputs.size() != 1) throw std::invalid_argument(op + " requires 1 input");
            return std::vector<Tensor>{b->dispatchTrigonometricOp(op, inputs[0])};
        }},
        {"tan", [](VulkanBackend* b, const std::string& op, std::span<const Tensor> inputs, const OpAttributes&) {
            if (inputs.size() != 1) throw std::invalid_argument(op + " requires 1 input");
            return std::vector<Tensor>{b->dispatchTrigonometricOp(op, inputs[0])};
        }},
        {"asin", [](VulkanBackend* b, const std::string& op, std::span<const Tensor> inputs, const OpAttributes&) {
            if (inputs.size() != 1) throw std::invalid_argument(op + " requires 1 input");
            return std::vector<Tensor>{b->dispatchTrigonometricOp(op, inputs[0])};
        }},
        {"acos", [](VulkanBackend* b, const std::string& op, std::span<const Tensor> inputs, const OpAttributes&) {
            if (inputs.size() != 1) throw std::invalid_argument(op + " requires 1 input");
            return std::vector<Tensor>{b->dispatchTrigonometricOp(op, inputs[0])};
        }},
        {"atan", [](VulkanBackend* b, const std::string& op, std::span<const Tensor> inputs, const OpAttributes&) {
            if (inputs.size() != 1) throw std::invalid_argument(op + " requires 1 input");
            return std::vector<Tensor>{b->dispatchTrigonometricOp(op, inputs[0])};
        }},

        // ==================================================================
        // Hyperbolic operations
        // ==================================================================
        {"sinh", [](VulkanBackend* b, const std::string& op, std::span<const Tensor> inputs, const OpAttributes&) {
            if (inputs.size() != 1) throw std::invalid_argument(op + " requires 1 input");
            return std::vector<Tensor>{b->dispatchHyperbolicOp(op, inputs[0])};
        }},
        {"cosh", [](VulkanBackend* b, const std::string& op, std::span<const Tensor> inputs, const OpAttributes&) {
            if (inputs.size() != 1) throw std::invalid_argument(op + " requires 1 input");
            return std::vector<Tensor>{b->dispatchHyperbolicOp(op, inputs[0])};
        }},

        // ==================================================================
        // Pow (unary with parameter)
        // ==================================================================
        {"pow", [](VulkanBackend* b, const std::string& op, std::span<const Tensor> inputs, const OpAttributes& attrs) {
            if (inputs.size() != 1) throw std::invalid_argument("pow requires 1 input");
            float exponent = static_cast<float>(attrs.get_float(AttrKey::Exponent, 2.0));
            return std::vector<Tensor>{b->dispatchUnaryOpWithParam(op, inputs[0], exponent)};
        }},

        // ==================================================================
        // Backward activation operations
        // ==================================================================
        {"relu_backward", [](VulkanBackend* b, const std::string&, std::span<const Tensor> inputs, const OpAttributes&) {
            if (inputs.size() != 2) throw std::invalid_argument("relu_backward requires 2 inputs");
            return std::vector<Tensor>{b->dispatchActivationBackward("relu_backward", inputs[0], inputs[1], 0, 0.0f)};
        }},
        {"sigmoid_backward", [](VulkanBackend* b, const std::string&, std::span<const Tensor> inputs, const OpAttributes&) {
            if (inputs.size() != 2) throw std::invalid_argument("sigmoid_backward requires 2 inputs");
            return std::vector<Tensor>{b->dispatchActivationBackward("sigmoid_backward", inputs[0], inputs[1], 1, 0.0f)};
        }},
        {"tanh_backward", [](VulkanBackend* b, const std::string&, std::span<const Tensor> inputs, const OpAttributes&) {
            if (inputs.size() != 2) throw std::invalid_argument("tanh_backward requires 2 inputs");
            return std::vector<Tensor>{b->dispatchActivationBackward("tanh_backward", inputs[0], inputs[1], 2, 0.0f)};
        }},
        {"leaky_relu_backward", [](VulkanBackend* b, const std::string&, std::span<const Tensor> inputs, const OpAttributes& attrs) {
            if (inputs.size() != 2) throw std::invalid_argument("leaky_relu_backward requires 2 inputs");
            float alpha = static_cast<float>(attrs.get_float(AttrKey::Alpha, 0.01));
            return std::vector<Tensor>{b->dispatchActivationBackward("leaky_relu_backward", inputs[0], inputs[1], 3, alpha)};
        }},
        {"gelu_backward", [](VulkanBackend* b, const std::string&, std::span<const Tensor> inputs, const OpAttributes&) {
            if (inputs.size() != 2) throw std::invalid_argument("gelu_backward requires 2 inputs");
            return std::vector<Tensor>{b->dispatchActivationBackward("gelu_backward", inputs[0], inputs[1], 4, 0.0f)};
        }},
        {"elu_backward", [](VulkanBackend* b, const std::string&, std::span<const Tensor> inputs, const OpAttributes& attrs) {
            if (inputs.size() != 2) throw std::invalid_argument("elu_backward requires 2 inputs");
            float alpha = static_cast<float>(attrs.get_float(AttrKey::Alpha, 1.0));
            return std::vector<Tensor>{b->dispatchActivationBackward("elu_backward", inputs[0], inputs[1], 5, alpha)};
        }},
        {"selu_backward", [](VulkanBackend* b, const std::string&, std::span<const Tensor> inputs, const OpAttributes&) {
            if (inputs.size() != 2) throw std::invalid_argument("selu_backward requires 2 inputs");
            return std::vector<Tensor>{b->dispatchActivationBackward("selu_backward", inputs[0], inputs[1], 6, 0.0f)};
        }},
        {"mish_backward", [](VulkanBackend* b, const std::string&, std::span<const Tensor> inputs, const OpAttributes&) {
            if (inputs.size() != 2) throw std::invalid_argument("mish_backward requires 2 inputs");
            return std::vector<Tensor>{b->dispatchActivationBackward("mish_backward", inputs[0], inputs[1], 7, 0.0f)};
        }},
        {"softplus_backward", [](VulkanBackend* b, const std::string&, std::span<const Tensor> inputs, const OpAttributes& attrs) {
            if (inputs.size() != 2) throw std::invalid_argument("softplus_backward requires 2 inputs");
            float beta = static_cast<float>(attrs.get_float(AttrKey::Beta, 1.0));
            return std::vector<Tensor>{b->dispatchActivationBackward("softplus_backward", inputs[0], inputs[1], 8, beta)};
        }},
        {"swish_backward", [](VulkanBackend* b, const std::string&, std::span<const Tensor> inputs, const OpAttributes&) {
            if (inputs.size() != 2) throw std::invalid_argument("swish_backward requires 2 inputs");
            return std::vector<Tensor>{b->dispatchSwishBackward(inputs[0], inputs[1])};
        }},
        {"softmax_backward", [](VulkanBackend* b, const std::string&, std::span<const Tensor> inputs, const OpAttributes& attrs) {
            if (inputs.size() != 2) throw std::invalid_argument("softmax_backward requires 2 inputs");
            int64_t dim = attrs.get_int(AttrKey::Dim, -1);
            return std::vector<Tensor>{b->dispatchSoftmaxBackward(inputs[0], inputs[1], dim)};
        }},
        {"log_softmax_backward", [](VulkanBackend* b, const std::string&, std::span<const Tensor> inputs, const OpAttributes& attrs) {
            if (inputs.size() != 2) throw std::invalid_argument("log_softmax_backward requires 2 inputs");
            int64_t dim = attrs.get_int(AttrKey::Dim, -1);
            return std::vector<Tensor>{b->dispatchLogSoftmaxBackward(inputs[0], inputs[1], dim)};
        }},

        // ==================================================================
        // Conv2d backward operations
        // ==================================================================
        {"conv2d_backward_input", [](VulkanBackend* b, const std::string&, std::span<const Tensor> inputs, const OpAttributes& attrs) {
            if (inputs.size() != 2) throw std::invalid_argument("conv2d_backward_input requires 2 inputs (grad_output, weight)");
            int64_t stride = attrs.get_int(AttrKey::Stride, 1);
            int64_t padding = attrs.get_int(AttrKey::Padding, 0);
            int64_t dilation = attrs.get_int(AttrKey::Dilation, 1);
            std::vector<int64_t> input_shape = attrs.get_int_list(AttrKey::InputShape);
            return std::vector<Tensor>{b->dispatchConv2dBackwardInput(inputs[0], inputs[1], stride, padding, dilation, input_shape)};
        }},
        {"conv2d_backward_weight", [](VulkanBackend* b, const std::string&, std::span<const Tensor> inputs, const OpAttributes& attrs) {
            if (inputs.size() != 2) throw std::invalid_argument("conv2d_backward_weight requires 2 inputs (grad_output, input)");
            int64_t stride = attrs.get_int(AttrKey::Stride, 1);
            int64_t padding = attrs.get_int(AttrKey::Padding, 0);
            int64_t dilation = attrs.get_int(AttrKey::Dilation, 1);
            std::vector<int64_t> weight_shape = attrs.get_int_list(AttrKey::WeightShape);
            return std::vector<Tensor>{b->dispatchConv2dBackwardWeight(inputs[0], inputs[1], stride, padding, dilation, weight_shape)};
        }},
        {"conv2d_backward_bias", [](VulkanBackend* b, const std::string&, std::span<const Tensor> inputs, const OpAttributes&) {
            if (inputs.size() != 1) throw std::invalid_argument("conv2d_backward_bias requires 1 input (grad_output)");
            return std::vector<Tensor>{b->dispatchConv2dBackwardBias(inputs[0])};
        }},
        {"conv2d_backward", [](VulkanBackend* b, const std::string&, std::span<const Tensor> inputs, const OpAttributes& attrs) {
            if (inputs.size() != 3) throw std::invalid_argument("conv2d_backward operation requires exactly 3 inputs (grad_output, input, weight)");
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
            if (compute_grad_input) results.push_back(b->dispatchConv2dBackwardInput(grad_output, weight, stride, padding, dilation, input_shape, groups));
            if (compute_grad_weight) results.push_back(b->dispatchConv2dBackwardWeight(grad_output, input, stride, padding, dilation, weight_shape, groups));
            if (compute_grad_bias) results.push_back(b->dispatchConv2dBackwardBias(grad_output));
            return results;
        }},

        // ==================================================================
        // Reduction operations
        // ==================================================================
        {"sum", [](VulkanBackend* b, const std::string& op, std::span<const Tensor> inputs, const OpAttributes& attrs) {
            if (inputs.size() != 1) throw std::invalid_argument(op + " requires 1 input");
            int64_t dim = attrs.has(AttrKey::Dim) ? attrs.get_int(AttrKey::Dim) : INT64_MIN;
            bool keepdim = attrs.get_bool(AttrKey::Keepdim, false);
            return std::vector<Tensor>{b->dispatchReduction(op, inputs[0], dim, keepdim)};
        }},
        {"mean", [](VulkanBackend* b, const std::string& op, std::span<const Tensor> inputs, const OpAttributes& attrs) {
            if (inputs.size() != 1) throw std::invalid_argument(op + " requires 1 input");
            int64_t dim = attrs.has(AttrKey::Dim) ? attrs.get_int(AttrKey::Dim) : INT64_MIN;
            bool keepdim = attrs.get_bool(AttrKey::Keepdim, false);
            return std::vector<Tensor>{b->dispatchReduction(op, inputs[0], dim, keepdim)};
        }},
        {"max", [](VulkanBackend* b, const std::string& op, std::span<const Tensor> inputs, const OpAttributes& attrs) {
            if (inputs.size() != 1) throw std::invalid_argument(op + " requires 1 input");
            int64_t dim = attrs.has(AttrKey::Dim) ? attrs.get_int(AttrKey::Dim) : INT64_MIN;
            bool keepdim = attrs.get_bool(AttrKey::Keepdim, false);
            return std::vector<Tensor>{b->dispatchReduction(op, inputs[0], dim, keepdim)};
        }},
        {"min", [](VulkanBackend* b, const std::string& op, std::span<const Tensor> inputs, const OpAttributes& attrs) {
            if (inputs.size() != 1) throw std::invalid_argument(op + " requires 1 input");
            int64_t dim = attrs.has(AttrKey::Dim) ? attrs.get_int(AttrKey::Dim) : INT64_MIN;
            bool keepdim = attrs.get_bool(AttrKey::Keepdim, false);
            return std::vector<Tensor>{b->dispatchReduction(op, inputs[0], dim, keepdim)};
        }},

        // ==================================================================
        // Matrix operations
        // ==================================================================
        {"matmul", [](VulkanBackend* b, const std::string&, std::span<const Tensor> inputs, const OpAttributes&) {
            if (inputs.size() != 2) throw std::invalid_argument("matmul requires 2 inputs");
            return std::vector<Tensor>{b->dispatchMatmul(inputs[0], inputs[1])};
        }},
        {"bmm", [](VulkanBackend* b, const std::string&, std::span<const Tensor> inputs, const OpAttributes&) {
            if (inputs.size() != 2) throw std::invalid_argument("bmm requires 2 inputs");
            return std::vector<Tensor>{b->dispatchBmm(inputs[0], inputs[1])};
        }},
        {"dot", [](VulkanBackend* b, const std::string&, std::span<const Tensor> inputs, const OpAttributes&) {
            if (inputs.size() != 2) throw std::invalid_argument("dot requires 2 inputs");
            return std::vector<Tensor>{b->dispatchDot(inputs[0], inputs[1])};
        }},
    };
    return table;
}

// Helper for pooling attr parsing
static void parse_pool_attrs(const OpAttributes& attrs,
    int64_t& kernel_h, int64_t& kernel_w,
    int64_t& stride_h, int64_t& stride_w,
    int64_t& padding_h, int64_t& padding_w) {
    kernel_h = 2; kernel_w = 2;
    if (attrs.has(AttrKey::KernelSizeH)) {
        kernel_h = attrs.get_int(AttrKey::KernelSizeH);
        kernel_w = attrs.has(AttrKey::KernelSizeW) ? attrs.get_int(AttrKey::KernelSizeW) : kernel_h;
    } else if (attrs.has(AttrKey::KernelSize)) {
        kernel_h = kernel_w = attrs.get_int(AttrKey::KernelSize);
    }
    stride_h = kernel_h; stride_w = kernel_w;
    if (attrs.has(AttrKey::StrideH)) {
        stride_h = attrs.get_int(AttrKey::StrideH);
        stride_w = attrs.has(AttrKey::StrideW) ? attrs.get_int(AttrKey::StrideW) : stride_h;
    } else if (attrs.has(AttrKey::Stride)) {
        stride_h = stride_w = attrs.get_int(AttrKey::Stride);
    }
    padding_h = 0; padding_w = 0;
    if (attrs.has(AttrKey::PaddingH)) {
        padding_h = attrs.get_int(AttrKey::PaddingH);
        padding_w = attrs.has(AttrKey::PaddingW) ? attrs.get_int(AttrKey::PaddingW) : padding_h;
    } else if (attrs.has(AttrKey::Padding)) {
        padding_h = padding_w = attrs.get_int(AttrKey::Padding);
    }
}

// Helper to parse dtype from string attribute
static DType parse_dtype_attr(const OpAttributes& attrs, DType default_val = DType::Float32) {
    if (!attrs.has(AttrKey::Dtype)) return default_val;
    auto s = attrs.get_string(AttrKey::Dtype);
    if (s == "float32" || s == "Float32") return DType::Float32;
    if (s == "float64" || s == "Float64") return DType::Float64;
    if (s == "float16" || s == "Float16") return DType::Float16;
    if (s == "bfloat16" || s == "BFloat16") return DType::BFloat16;
    if (s == "int32" || s == "Int32") return DType::Int32;
    if (s == "int64" || s == "Int64") return DType::Int64;
    if (s == "int8" || s == "Int8") return DType::Int8;
    if (s == "int16" || s == "Int16") return DType::Int16;
    if (s == "uint8" || s == "UInt8") return DType::UInt8;
    if (s == "uint16" || s == "Uint16") return DType::UInt16;
    if (s == "uint32" || s == "UInt32") return DType::UInt32;
    if (s == "uint64" || s == "UInt64") return DType::UInt64;
    if (s == "bool" || s == "Bool") return DType::Bool;
    if (s == "complex64" || s == "Complex64") return DType::Complex64;
    if (s == "complex128" || s == "Complex128") return DType::Complex128;
    return default_val;
}

// Second dispatch table for remaining operations
static const std::unordered_map<std::string, VkDispatchHandler>& get_dispatch_table_2() {
    static const std::unordered_map<std::string, VkDispatchHandler> table = {
        // ==================================================================
        // Pooling operations
        // ==================================================================
        {"max_pool2d", [](VulkanBackend* b, const std::string&, std::span<const Tensor> inputs, const OpAttributes& attrs) {
            int64_t kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w;
            parse_pool_attrs(attrs, kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w);
            // Float16/Float64: use typed shader via dispatchMaxPool2dForward
            if (inputs[0].dtype() == DType::Float16 || inputs[0].dtype() == DType::Float64) {
                OpAttributes pool_attrs;
                pool_attrs.set(AttrKey::KernelSizeH, kernel_h);
                pool_attrs.set(AttrKey::KernelSizeW, kernel_w);
                pool_attrs.set(AttrKey::StrideH, stride_h);
                pool_attrs.set(AttrKey::StrideW, stride_w);
                pool_attrs.set(AttrKey::PaddingH, padding_h);
                pool_attrs.set(AttrKey::PaddingW, padding_w);
                Tensor output = b->dispatchMaxPool2dForward(inputs[0], pool_attrs);
                std::vector<int64_t> out_shape_vec(output.shape().begin(), output.shape().end());
                Tensor pool_indices(out_shape_vec, DType::Int32, inputs[0].device());
                pool_indices = b->dispatchFill(pool_indices, 0.0f);
                return std::vector<Tensor>{output, pool_indices};
            }
            auto [output, indices] = b->dispatchMaxPool2d(inputs[0], kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w);
            return std::vector<Tensor>{output, indices};
        }},
        {"avg_pool2d", [](VulkanBackend* b, const std::string&, std::span<const Tensor> inputs, const OpAttributes& attrs) {
            int64_t kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w;
            parse_pool_attrs(attrs, kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w);
            return std::vector<Tensor>{b->dispatchAvgPool2d(inputs[0], kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w)};
        }},
        {"adaptive_max_pool2d", [](VulkanBackend* b, const std::string&, std::span<const Tensor> inputs, const OpAttributes& attrs) {
            int64_t out_h = attrs.has(AttrKey::OutputHeight) ? attrs.get_int(AttrKey::OutputHeight) : attrs.get_int(AttrKey::OutputSizeH);
            int64_t out_w = attrs.has(AttrKey::OutputWidth) ? attrs.get_int(AttrKey::OutputWidth) : attrs.get_int(AttrKey::OutputSizeW);
            auto [output, indices] = b->dispatchAdaptiveMaxPool2d(inputs[0], out_h, out_w);
            return std::vector<Tensor>{output, indices};
        }},
        {"adaptive_avg_pool2d", [](VulkanBackend* b, const std::string&, std::span<const Tensor> inputs, const OpAttributes& attrs) {
            int64_t out_h = attrs.has(AttrKey::OutputHeight) ? attrs.get_int(AttrKey::OutputHeight) : attrs.get_int(AttrKey::OutputSizeH);
            int64_t out_w = attrs.has(AttrKey::OutputWidth) ? attrs.get_int(AttrKey::OutputWidth) : attrs.get_int(AttrKey::OutputSizeW);
            return std::vector<Tensor>{b->dispatchAdaptiveAvgPool2d(inputs[0], out_h, out_w)};
        }},
        {"adaptive_avg_pool2d_backward", [](VulkanBackend* b, const std::string&, std::span<const Tensor> inputs, const OpAttributes& attrs) {
            if (inputs.size() != 1) throw std::invalid_argument("adaptive_avg_pool2d_backward requires exactly 1 input (grad_output)");
            int64_t H_in = attrs.get_int(AttrKey::InputH, 0);
            int64_t W_in = attrs.get_int(AttrKey::InputW, 0);
            return std::vector<Tensor>{b->dispatchAdaptiveAvgPool2dBackward(inputs[0], H_in, W_in)};
        }},

        // ==================================================================
        // Normalization
        // ==================================================================
        {"softmax", [](VulkanBackend* b, const std::string&, std::span<const Tensor> inputs, const OpAttributes& attrs) {
            int64_t dim = attrs.get_int(AttrKey::Dim, -1);
            return std::vector<Tensor>{b->dispatchSoftmax(inputs[0], dim)};
        }},
        {"log_softmax", [](VulkanBackend* b, const std::string&, std::span<const Tensor> inputs, const OpAttributes& attrs) {
            int64_t dim = attrs.get_int(AttrKey::Dim, -1);
            return std::vector<Tensor>{b->dispatchLogSoftmax(inputs[0], dim)};
        }},

        // ==================================================================
        // Advanced reductions
        // ==================================================================
        {"argmax", [](VulkanBackend* b, const std::string&, std::span<const Tensor> inputs, const OpAttributes& attrs) {
            int64_t dim = attrs.get_int(AttrKey::Dim, -1);
            bool keepdim = attrs.get_bool(AttrKey::Keepdim, false);
            return std::vector<Tensor>{b->dispatchArgmax(inputs[0], dim, keepdim)};
        }},
        {"argmin", [](VulkanBackend* b, const std::string&, std::span<const Tensor> inputs, const OpAttributes& attrs) {
            int64_t dim = attrs.get_int(AttrKey::Dim, -1);
            bool keepdim = attrs.get_bool(AttrKey::Keepdim, false);
            return std::vector<Tensor>{b->dispatchArgmin(inputs[0], dim, keepdim)};
        }},
        {"argsort", [](VulkanBackend* b, const std::string&, std::span<const Tensor> inputs, const OpAttributes& attrs) {
            int64_t dim = attrs.get_int(AttrKey::Dim, -1);
            bool descending = attrs.get_bool(AttrKey::Descending, false);
            return std::vector<Tensor>{b->dispatchArgSort(inputs[0], dim, descending)};
        }},
        {"var", [](VulkanBackend* b, const std::string&, std::span<const Tensor> inputs, const OpAttributes& attrs) {
            int64_t dim = attrs.get_int(AttrKey::Dim, -1);
            bool unbiased = attrs.get_bool(AttrKey::Unbiased, true);
            bool keepdim = attrs.get_bool(AttrKey::Keepdim, false);
            return std::vector<Tensor>{b->dispatchVariance(inputs[0], dim, unbiased, keepdim)};
        }},
        {"variance", [](VulkanBackend* b, const std::string&, std::span<const Tensor> inputs, const OpAttributes& attrs) {
            int64_t dim = attrs.get_int(AttrKey::Dim, -1);
            bool unbiased = attrs.get_bool(AttrKey::Unbiased, true);
            bool keepdim = attrs.get_bool(AttrKey::Keepdim, false);
            return std::vector<Tensor>{b->dispatchVariance(inputs[0], dim, unbiased, keepdim)};
        }},
        {"std", [](VulkanBackend* b, const std::string&, std::span<const Tensor> inputs, const OpAttributes& attrs) {
            int64_t dim = attrs.get_int(AttrKey::Dim, -1);
            bool unbiased = attrs.get_bool(AttrKey::Unbiased, true);
            bool keepdim = attrs.get_bool(AttrKey::Keepdim, false);
            return std::vector<Tensor>{b->dispatchStd(inputs[0], dim, unbiased, keepdim)};
        }},
        {"prod", [](VulkanBackend* b, const std::string&, std::span<const Tensor> inputs, const OpAttributes& attrs) {
            int64_t dim = attrs.get_int(AttrKey::Dim, -1);
            bool keepdim = attrs.get_bool(AttrKey::Keepdim, false);
            return std::vector<Tensor>{b->dispatchProd(inputs[0], dim, keepdim)};
        }},
        {"norm", [](VulkanBackend* b, const std::string&, std::span<const Tensor> inputs, const OpAttributes& attrs) {
            float p = static_cast<float>(attrs.get_float(AttrKey::P, 2.0));
            int64_t dim = attrs.has(AttrKey::Dim) ? attrs.get_int(AttrKey::Dim) : INT64_MIN;
            bool keepdim = attrs.get_bool(AttrKey::Keepdim, false);
            return std::vector<Tensor>{b->dispatchNorm(inputs[0], p, dim, keepdim)};
        }},

        // ==================================================================
        // Indexing operations
        // ==================================================================
        {"embedding", [](VulkanBackend* b, const std::string&, std::span<const Tensor> inputs, const OpAttributes& attrs) {
            int64_t padding_idx = attrs.get_int(AttrKey::PaddingIdx, -1);
            return std::vector<Tensor>{b->dispatchEmbedding(inputs[0], inputs[1], padding_idx)};
        }},
        {"gather", [](VulkanBackend* b, const std::string&, std::span<const Tensor> inputs, const OpAttributes& attrs) {
            int64_t dim = attrs.get_int(AttrKey::Dim);
            return std::vector<Tensor>{b->dispatchGather(inputs[0], dim, inputs[1])};
        }},
        {"scatter", [](VulkanBackend* b, const std::string&, std::span<const Tensor> inputs, const OpAttributes& attrs) {
            int64_t dim = attrs.get_int(AttrKey::Dim);
            int64_t reduction = attrs.get_int(AttrKey::Reduction, 0);
            return std::vector<Tensor>{b->dispatchScatter(inputs[0], dim, inputs[1], inputs[2], reduction)};
        }},
        {"index_select", [](VulkanBackend* b, const std::string&, std::span<const Tensor> inputs, const OpAttributes& attrs) {
            int64_t dim = attrs.get_int(AttrKey::Dim);
            return std::vector<Tensor>{b->dispatchIndexSelect(inputs[0], dim, inputs[1])};
        }},
        {"gather_relative_position_bias", [](VulkanBackend* b, const std::string&, std::span<const Tensor> inputs, const OpAttributes& attrs) {
            if (inputs.size() != 2) throw std::invalid_argument("gather_relative_position_bias operation requires exactly 2 inputs");
            int64_t num_positions = attrs.get_int(AttrKey::NumPositions, 0);
            int64_t num_heads = attrs.get_int(AttrKey::NumHeads, 0);
            return std::vector<Tensor>{b->dispatchGatherRelativePositionBias(inputs[0], inputs[1], num_positions, num_heads)};
        }},

        // ==================================================================
        // Shape operations
        // ==================================================================
        {"reshape", [](VulkanBackend* b, const std::string&, std::span<const Tensor> inputs, const OpAttributes& attrs) {
            if (inputs.size() != 1) throw std::invalid_argument("reshape requires 1 input");
            if (!attrs.has(AttrKey::Shape)) throw std::invalid_argument("reshape requires 'shape' attribute");
            std::vector<int64_t> new_shape = attrs.get_int_list(AttrKey::Shape);
            return std::vector<Tensor>{b->dispatchReshape(inputs[0], new_shape)};
        }},
        {"transpose", [](VulkanBackend* b, const std::string&, std::span<const Tensor> inputs, const OpAttributes& attrs) {
            if (inputs.size() != 1) throw std::invalid_argument("transpose requires 1 input");
            int64_t dim0 = attrs.get_int(AttrKey::Dim0);
            int64_t dim1 = attrs.get_int(AttrKey::Dim1);
            return std::vector<Tensor>{b->dispatchTranspose(inputs[0], dim0, dim1)};
        }},
        {"permute", [](VulkanBackend* b, const std::string&, std::span<const Tensor> inputs, const OpAttributes& attrs) {
            if (inputs.size() != 1) throw std::invalid_argument("permute requires 1 input");
            std::vector<int64_t> dims = attrs.get_int_list(AttrKey::Dims);
            return std::vector<Tensor>{b->dispatchPermute(inputs[0], dims)};
        }},
        {"squeeze", [](VulkanBackend* b, const std::string&, std::span<const Tensor> inputs, const OpAttributes& attrs) {
            if (inputs.size() != 1) throw std::invalid_argument("squeeze requires 1 input");
            int64_t dim = attrs.get_int(AttrKey::Dim, -1);
            return std::vector<Tensor>{b->dispatchSqueeze(inputs[0], dim)};
        }},
        {"unsqueeze", [](VulkanBackend* b, const std::string&, std::span<const Tensor> inputs, const OpAttributes& attrs) {
            if (inputs.size() != 1) throw std::invalid_argument("unsqueeze requires 1 input");
            int64_t dim = attrs.get_int(AttrKey::Dim);
            return std::vector<Tensor>{b->dispatchUnsqueeze(inputs[0], dim)};
        }},
        {"contiguous", [](VulkanBackend* b, const std::string&, std::span<const Tensor> inputs, const OpAttributes&) {
            if (inputs.size() != 1) throw std::invalid_argument("contiguous requires 1 input");
            return std::vector<Tensor>{b->dispatchContiguous(inputs[0])};
        }},

        // ==================================================================
        // Memory / creation operations
        // ==================================================================
        {"zeros", [](VulkanBackend* b, const std::string&, std::span<const Tensor>, const OpAttributes& attrs) {
            std::vector<int64_t> shape = attrs.get_int_list(AttrKey::Shape);
            DType dtype = parse_dtype_attr(attrs);
            int32_t device_id = static_cast<int32_t>(attrs.get_int(AttrKey::DeviceId, 0));
            Device device = Device::vulkan(device_id);
            return std::vector<Tensor>{b->dispatchZeros(shape, dtype, device)};
        }},
        {"fill", [](VulkanBackend* b, const std::string&, std::span<const Tensor> inputs, const OpAttributes& attrs) {
            if (inputs.size() != 1) throw std::invalid_argument("fill requires 1 input");
            float value = static_cast<float>(attrs.get_float(AttrKey::Value));
            return std::vector<Tensor>{b->dispatchFill(inputs[0], value)};
        }},
        {"clone", [](VulkanBackend* b, const std::string&, std::span<const Tensor> inputs, const OpAttributes&) {
            if (inputs.size() != 1) throw std::invalid_argument("clone requires 1 input");
            return std::vector<Tensor>{b->dispatchClone(inputs[0])};
        }},
        {"im2col", [](VulkanBackend* b, const std::string&, std::span<const Tensor> inputs, const OpAttributes& attrs) {
            if (inputs.size() != 1) throw std::invalid_argument("im2col requires 1 input");
            return std::vector<Tensor>{b->dispatchIm2Col(inputs[0], attrs)};
        }},
        {"unfold", [](VulkanBackend* b, const std::string&, std::span<const Tensor> inputs, const OpAttributes& attrs) {
            if (inputs.size() != 1) throw std::invalid_argument("im2col requires 1 input");
            return std::vector<Tensor>{b->dispatchIm2Col(inputs[0], attrs)};
        }},
        {"col2im", [](VulkanBackend* b, const std::string&, std::span<const Tensor> inputs, const OpAttributes& attrs) {
            if (inputs.size() != 1) throw std::invalid_argument("col2im requires 1 input");
            return std::vector<Tensor>{b->dispatchCol2Im(inputs[0], attrs)};
        }},
        {"fold", [](VulkanBackend* b, const std::string&, std::span<const Tensor> inputs, const OpAttributes& attrs) {
            if (inputs.size() != 1) throw std::invalid_argument("col2im requires 1 input");
            return std::vector<Tensor>{b->dispatchCol2Im(inputs[0], attrs)};
        }},
        {"expand", [](VulkanBackend* b, const std::string&, std::span<const Tensor> inputs, const OpAttributes& attrs) {
            if (inputs.size() != 1) throw std::invalid_argument("expand requires 1 input");
            std::vector<int64_t> shape = attrs.get_int_list(AttrKey::Shape);
            return std::vector<Tensor>{b->dispatchExpand(inputs[0], shape)};
        }},
        {"cat", [](VulkanBackend* b, const std::string&, std::span<const Tensor> inputs, const OpAttributes& attrs) {
            if (inputs.size() < 2) throw std::invalid_argument("cat requires at least 2 inputs");
            int64_t dim = attrs.get_int(AttrKey::Dim, 0);
            std::vector<Tensor> input_tensors(inputs.begin(), inputs.end());
            return std::vector<Tensor>{b->dispatchCat(input_tensors, dim)};
        }},
        {"concatenate", [](VulkanBackend* b, const std::string&, std::span<const Tensor> inputs, const OpAttributes& attrs) {
            if (inputs.size() < 2) throw std::invalid_argument("cat requires at least 2 inputs");
            int64_t dim = attrs.get_int(AttrKey::Dim, 0);
            std::vector<Tensor> input_tensors(inputs.begin(), inputs.end());
            return std::vector<Tensor>{b->dispatchCat(input_tensors, dim)};
        }},
        {"clamp", [](VulkanBackend* b, const std::string&, std::span<const Tensor> inputs, const OpAttributes& attrs) {
            if (inputs.size() != 1) throw std::invalid_argument("clamp requires 1 input");
            float min_value = static_cast<float>(attrs.get_float(AttrKey::Min, -std::numeric_limits<double>::infinity()));
            float max_value = static_cast<float>(attrs.get_float(AttrKey::Max, std::numeric_limits<double>::infinity()));
            return std::vector<Tensor>{b->dispatchClamp(inputs[0], min_value, max_value)};
        }},
        {"clamp_min", [](VulkanBackend* b, const std::string&, std::span<const Tensor> inputs, const OpAttributes& attrs) {
            if (inputs.size() != 1) throw std::invalid_argument("clamp_min requires 1 input");
            float min_value = static_cast<float>(attrs.get_float(AttrKey::Min, -std::numeric_limits<double>::infinity()));
            return std::vector<Tensor>{b->dispatchClamp(inputs[0], min_value, std::numeric_limits<float>::infinity())};
        }},
        {"clamp_max", [](VulkanBackend* b, const std::string&, std::span<const Tensor> inputs, const OpAttributes& attrs) {
            if (inputs.size() != 1) throw std::invalid_argument("clamp_max requires 1 input");
            float max_value = static_cast<float>(attrs.get_float(AttrKey::Max, std::numeric_limits<double>::infinity()));
            return std::vector<Tensor>{b->dispatchClamp(inputs[0], -std::numeric_limits<float>::infinity(), max_value)};
        }},
        {"masked_select", [](VulkanBackend* b, const std::string&, std::span<const Tensor> inputs, const OpAttributes&) {
            if (inputs.size() != 2) throw std::invalid_argument("masked_select operation requires exactly 2 inputs");
            return std::vector<Tensor>{b->dispatchMaskedSelect(inputs[0], inputs[1])};
        }},
        {"masked_fill", [](VulkanBackend* b, const std::string&, std::span<const Tensor> inputs, const OpAttributes& attrs) {
            if (inputs.size() != 2) throw std::invalid_argument("masked_fill operation requires exactly 2 inputs");
            if (!attrs.has(AttrKey::Value)) throw std::invalid_argument("masked_fill operation requires 'value' attribute");
            float value = static_cast<float>(attrs.get_float(AttrKey::Value));
            return std::vector<Tensor>{b->dispatchMaskedFill(inputs[0], inputs[1], value)};
        }},
        {"where", [](VulkanBackend* b, const std::string&, std::span<const Tensor> inputs, const OpAttributes&) {
            if (inputs.size() != 3) throw std::invalid_argument("where operation requires exactly 3 inputs");
            return std::vector<Tensor>{b->dispatchWhere(inputs[0], inputs[1], inputs[2])};
        }},
        {"repeat", [](VulkanBackend* b, const std::string&, std::span<const Tensor> inputs, const OpAttributes& attrs) {
            if (inputs.size() != 1) throw std::invalid_argument("repeat operation requires exactly 1 input");
            if (!attrs.has(AttrKey::Repeats)) throw std::invalid_argument("repeat operation requires 'repeats' attribute");
            std::vector<int64_t> repeats = attrs.get_int_list(AttrKey::Repeats);
            return std::vector<Tensor>{b->dispatchRepeat(inputs[0], repeats)};
        }},
        {"interpolate", [](VulkanBackend* b, const std::string&, std::span<const Tensor> inputs, const OpAttributes& attrs) {
            if (inputs.size() != 1) throw std::invalid_argument("interpolate requires 1 input");
            return std::vector<Tensor>{b->dispatchInterpolate(inputs[0], attrs)};
        }},
        {"roi_align_forward", [](VulkanBackend* b, const std::string&, std::span<const Tensor> inputs, const OpAttributes& attrs) {
            if (inputs.size() < 2) throw std::invalid_argument("roi_align_forward requires 2 inputs (features, rois)");
            return std::vector<Tensor>{b->dispatchROIAlignForward(inputs[0], inputs[1], attrs)};
        }},
        {"roi_align_backward", [](VulkanBackend* b, const std::string&, std::span<const Tensor> inputs, const OpAttributes& attrs) {
            if (inputs.size() < 2) throw std::invalid_argument("roi_align_backward requires 2 inputs (grad_output, rois)");
            return std::vector<Tensor>{b->dispatchROIAlignBackward(inputs[0], inputs[1], attrs)};
        }},
        {"nonzero", [](VulkanBackend* b, const std::string&, std::span<const Tensor> inputs, const OpAttributes&) {
            if (inputs.size() < 1) throw std::invalid_argument("nonzero requires 1 input");
            return std::vector<Tensor>{b->dispatchNonzero(inputs[0])};
        }},
        {"one_hot", [](VulkanBackend* b, const std::string&, std::span<const Tensor> inputs, const OpAttributes& attrs) {
            if (inputs.size() < 1) throw std::invalid_argument("one_hot requires 1 input (indices)");
            int64_t num_classes = attrs.get_int(AttrKey::NumClasses, 10);
            return std::vector<Tensor>{b->dispatchOneHot(inputs[0], num_classes)};
        }},
        {"box_iou", [](VulkanBackend* b, const std::string&, std::span<const Tensor> inputs, const OpAttributes& attrs) {
            if (inputs.size() < 2) throw std::invalid_argument("box_iou requires 2 inputs (boxes1, boxes2)");
            int64_t iou_type = attrs.get_int(AttrKey::IouType, 0);
            return std::vector<Tensor>{b->dispatchBoxIoU(inputs[0], inputs[1], iou_type)};
        }},
    };
    return table;
}

// Third dispatch table for BatchNorm, LayerNorm, GroupNorm, Embedding backward,
// RMSNorm, Conv2d forward, creation ops, fused ops, and fused optimizers
static const std::unordered_map<std::string, VkDispatchHandler>& get_dispatch_table_3() {
    static const std::unordered_map<std::string, VkDispatchHandler> table = {
        // ==================================================================
        // BatchNorm2d operations
        // ==================================================================
        {"batchnorm2d_forward", [](VulkanBackend* b, const std::string&, std::span<const Tensor> inputs, const OpAttributes& attrs) {
            if (inputs.size() < 3) throw std::invalid_argument("batchnorm2d_forward requires at least 3 inputs (input, mean, var)");
            const Tensor* gamma = (inputs.size() > 3) ? &inputs[3] : nullptr;
            const Tensor* beta = (inputs.size() > 4) ? &inputs[4] : nullptr;
            float epsilon = static_cast<float>(attrs.get_float(AttrKey::Eps, 1e-5));
            return std::vector<Tensor>{b->dispatchBatchNorm2dForward(inputs[0], inputs[1], inputs[2], gamma, beta, epsilon)};
        }},
        {"batchnorm2d_forward_affine", [](VulkanBackend* b, const std::string&, std::span<const Tensor> inputs, const OpAttributes& attrs) {
            if (inputs.size() < 5) throw std::invalid_argument("batchnorm2d_forward_affine requires 5 inputs (input, mean, var, weight, bias)");
            float epsilon = static_cast<float>(attrs.get_float(AttrKey::Eps, 1e-5));
            return std::vector<Tensor>{b->dispatchBatchNorm2dForward(inputs[0], inputs[1], inputs[2], &inputs[3], &inputs[4], epsilon)};
        }},
        {"batchnorm2d_backward", [](VulkanBackend* b, const std::string&, std::span<const Tensor> inputs, const OpAttributes& attrs) {
            if (inputs.size() < 5) throw std::invalid_argument("batchnorm2d_backward requires 5 inputs (grad_output, input, weight, mean, invstd)");
            const Tensor* gamma = &inputs[2];
            float epsilon = static_cast<float>(attrs.get_float(AttrKey::Eps, 1e-5));
            auto [grad_input, grad_gamma, grad_beta] = b->dispatchBatchNorm2dBackward(inputs[0], inputs[1], inputs[3], inputs[4], gamma, epsilon);
            return std::vector<Tensor>{grad_input, grad_gamma, grad_beta};
        }},
        {"batchnorm2d_mean_var", [](VulkanBackend* b, const std::string&, std::span<const Tensor> inputs, const OpAttributes&) {
            if (inputs.size() != 1) throw std::invalid_argument("batchnorm2d_mean_var requires 1 input");
            auto [mean, variance] = b->dispatchBatchNorm2dMeanVar(inputs[0]);
            return std::vector<Tensor>{mean, variance};
        }},

        // ==================================================================
        // Pooling operations (new OpAttributes versions)
        // ==================================================================
        {"avg_pool2d_forward", [](VulkanBackend* b, const std::string&, std::span<const Tensor> inputs, const OpAttributes& attrs) {
            if (inputs.size() != 1) throw std::invalid_argument("avg_pool2d_forward requires 1 input");
            return std::vector<Tensor>{b->dispatchAvgPool2dForward(inputs[0], attrs)};
        }},
        {"max_pool2d_forward", [](VulkanBackend* b, const std::string&, std::span<const Tensor> inputs, const OpAttributes& attrs) {
            if (inputs.size() != 1) throw std::invalid_argument("max_pool2d_forward requires 1 input");
            return std::vector<Tensor>{b->dispatchMaxPool2dForward(inputs[0], attrs)};
        }},
        {"avg_pool2d_backward", [](VulkanBackend* b, const std::string&, std::span<const Tensor> inputs, const OpAttributes& attrs) {
            if (inputs.size() >= 2) {
                return std::vector<Tensor>{b->dispatchAvgPool2dBackward(inputs[0], inputs[1], attrs)};
            }
            if (inputs.size() == 1) {
                OpAttributes bwd_attrs;
                int64_t in_n = 0, in_c = 0, in_h = 0, in_w = 0;
                if (attrs.has(AttrKey::InputShape)) {
                    auto dims = attrs.get_int_list(AttrKey::InputShape);
                    if (dims.size() >= 4) { in_n = dims[0]; in_c = dims[1]; in_h = dims[2]; in_w = dims[3]; }
                }
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
                Tensor dummy_input({in_n, in_c, in_h, in_w}, inputs[0].dtype(), inputs[0].device());
                return std::vector<Tensor>{b->dispatchAvgPool2dBackward(inputs[0], dummy_input, bwd_attrs)};
            }
            throw std::invalid_argument("avg_pool2d_backward requires at least 1 input");
        }},
        {"max_pool2d_backward", [](VulkanBackend* b, const std::string&, std::span<const Tensor> inputs, const OpAttributes& attrs) {
            if (inputs.size() != 2) throw std::invalid_argument("max_pool2d_backward requires 2 inputs (grad_output, indices)");
            int64_t H_in = attrs.get_int(AttrKey::InputH);
            int64_t W_in = attrs.get_int(AttrKey::InputW);
            return std::vector<Tensor>{b->dispatchMaxPool2dBackwardWithIndices(inputs[0], inputs[1], H_in, W_in)};
        }},

        // ==================================================================
        // Conv2d forward / ConvTranspose2d forward
        // ==================================================================
        {"conv2d_forward", [](VulkanBackend* b, const std::string&, std::span<const Tensor> inputs, const OpAttributes& attrs) {
            if (inputs.size() < 2) throw std::invalid_argument("conv2d_forward requires at least 2 inputs (input, weight)");
            const Tensor* bias_ptr = inputs.size() >= 3 ? &inputs[2] : nullptr;
            return std::vector<Tensor>{b->dispatchConv2dForward(inputs[0], inputs[1], bias_ptr, attrs)};
        }},
        {"conv_transpose2d_forward", [](VulkanBackend* b, const std::string&, std::span<const Tensor> inputs, const OpAttributes& attrs) {
            if (inputs.size() < 2) throw std::invalid_argument("conv_transpose2d_forward requires at least 2 inputs (input, weight)");
            if (inputs[0].dtype() != DType::Float32 && inputs[0].dtype() != DType::Float64 &&
                inputs[0].dtype() != DType::Float16 && inputs[0].dtype() != DType::BFloat16) {
                TENZOR_LOG_WARNING(std::format("Vulkan: No shader for conv_transpose2d_forward with dtype {}; falling back to CPU",
                                               dtype_name(inputs[0].dtype())));
                Device original_device = inputs[0].device();
                std::vector<Tensor> cpu_inputs;
                cpu_inputs.reserve(inputs.size());
                for (const auto& t : inputs) cpu_inputs.push_back(t.to(Device::cpu()));
                auto cpu_results = tenzor::dispatch(OpId::ConvTranspose2dForward, cpu_inputs, attrs);
                return std::vector<Tensor>{cpu_results[0].to(original_device)};
            }
            const Tensor* bias_ptr = (inputs.size() >= 3) ? &inputs[2] : nullptr;
            return std::vector<Tensor>{b->dispatchConvTranspose2dForward(inputs[0], inputs[1], bias_ptr, attrs)};
        }},

        // ==================================================================
        // Creation operations
        // ==================================================================
        {"full", [](VulkanBackend* b, const std::string&, std::span<const Tensor>, const OpAttributes& attrs) {
            std::vector<int64_t> shape = attrs.get_int_list(AttrKey::Shape);
            float value = static_cast<float>(attrs.get_float(AttrKey::Value, 0.0));
            DType dtype = parse_dtype_attr(attrs);
            return std::vector<Tensor>{b->dispatchFull(shape, value, dtype)};
        }},
        {"ones", [](VulkanBackend* b, const std::string&, std::span<const Tensor>, const OpAttributes& attrs) {
            std::vector<int64_t> shape = attrs.get_int_list(AttrKey::Shape);
            DType dtype = parse_dtype_attr(attrs);
            return std::vector<Tensor>{b->dispatchOnes(shape, dtype)};
        }},
        {"rand", [](VulkanBackend* b, const std::string&, std::span<const Tensor>, const OpAttributes& attrs) {
            std::vector<int64_t> shape = attrs.get_int_list(AttrKey::Shape);
            DType dtype = parse_dtype_attr(attrs);
            return std::vector<Tensor>{b->dispatchRand(shape, dtype)};
        }},
        {"randn", [](VulkanBackend* b, const std::string&, std::span<const Tensor>, const OpAttributes& attrs) {
            std::vector<int64_t> shape = attrs.get_int_list(AttrKey::Shape);
            DType dtype = parse_dtype_attr(attrs);
            return std::vector<Tensor>{b->dispatchRandn(shape, dtype)};
        }},
        {"arange", [](VulkanBackend* b, const std::string&, std::span<const Tensor>, const OpAttributes& attrs) {
            float start = static_cast<float>(attrs.get_float(AttrKey::Start, 0.0));
            float end_val = static_cast<float>(attrs.get_float(AttrKey::End, 0.0));
            float step = static_cast<float>(attrs.get_float(AttrKey::Step, 1.0));
            DType dtype = parse_dtype_attr(attrs);
            int32_t device_id = static_cast<int32_t>(attrs.get_int(AttrKey::DeviceId, 0));
            Device device = Device::vulkan(device_id);
            return std::vector<Tensor>{b->dispatchArange(start, end_val, step, dtype, device)};
        }},
        {"linspace", [](VulkanBackend* b, const std::string&, std::span<const Tensor>, const OpAttributes& attrs) {
            float start = static_cast<float>(attrs.get_float(AttrKey::Start, 0.0));
            float end_val = static_cast<float>(attrs.get_float(AttrKey::End, 1.0));
            int64_t steps = attrs.get_int(AttrKey::Steps, 100);
            DType dtype = parse_dtype_attr(attrs);
            int32_t device_id = static_cast<int32_t>(attrs.get_int(AttrKey::DeviceId, 0));
            Device device = Device::vulkan(device_id);
            return std::vector<Tensor>{b->dispatchLinspace(start, end_val, steps, dtype, device)};
        }},
        {"eye", [](VulkanBackend* b, const std::string&, std::span<const Tensor>, const OpAttributes& attrs) {
            int64_t n = attrs.get_int(AttrKey::N, 0);
            int64_t m = attrs.get_int(AttrKey::M, -1);
            DType dtype = parse_dtype_attr(attrs);
            int32_t device_id = static_cast<int32_t>(attrs.get_int(AttrKey::DeviceId, 0));
            Device device = Device::vulkan(device_id);
            return std::vector<Tensor>{b->dispatchEye(n, m, dtype, device)};
        }},

        // ==================================================================
        // LayerNorm / GroupNorm / RMSNorm / Embedding backward
        // ==================================================================
        {"layer_norm", [](VulkanBackend* b, const std::string&, std::span<const Tensor> inputs, const OpAttributes& attrs) {
            if (inputs.size() < 1) throw std::invalid_argument("layer_norm requires at least 1 input");
            int64_t normalized_size = 1;
            if (attrs.has(AttrKey::NormalizedShape)) {
                auto ns_str = attrs.get_string(AttrKey::NormalizedShape);
                std::string ns_s{ns_str};
                std::stringstream ss(ns_s);
                std::string token;
                while (std::getline(ss, token, ',')) normalized_size *= std::stoll(token);
            } else {
                normalized_size = inputs[0].shape().back();
            }
            float eps = static_cast<float>(attrs.get_float(AttrKey::Eps, 1e-5));
            const Tensor* gamma = (inputs.size() > 1) ? &inputs[1] : nullptr;
            const Tensor* beta = (inputs.size() > 2) ? &inputs[2] : nullptr;
            return std::vector<Tensor>{b->dispatchLayerNorm(inputs[0], normalized_size, gamma, beta, eps)};
        }},
        {"layer_norm_backward", [](VulkanBackend* b, const std::string&, std::span<const Tensor> inputs, const OpAttributes& attrs) {
            if (inputs.size() < 5) throw std::invalid_argument("layer_norm_backward requires 5 inputs (grad_output, input, mean, rstd, weight)");
            int64_t normalized_shape = inputs[0].shape().back();
            if (attrs.has(AttrKey::NormalizedShape)) normalized_shape = attrs.get_int(AttrKey::NormalizedShape);
            auto [grad_input, grad_weight, grad_bias] = b->dispatchLayerNormBackward(inputs[0], inputs[1], inputs[2], inputs[3], &inputs[4], normalized_shape);
            return std::vector<Tensor>{grad_input, grad_weight, grad_bias};
        }},
        {"group_norm", [](VulkanBackend* b, const std::string&, std::span<const Tensor> inputs, const OpAttributes& attrs) {
            if (inputs.size() < 1) throw std::invalid_argument("group_norm requires at least 1 input");
            int64_t num_groups = attrs.get_int(AttrKey::NumGroups, 1);
            float eps = static_cast<float>(attrs.get_float(AttrKey::Eps, 1e-5));
            const Tensor* gamma = (inputs.size() > 1) ? &inputs[1] : nullptr;
            const Tensor* beta = (inputs.size() > 2) ? &inputs[2] : nullptr;
            return b->dispatchGroupNorm(inputs[0], num_groups, gamma, beta, eps);
        }},
        {"group_norm_backward", [](VulkanBackend* b, const std::string&, std::span<const Tensor> inputs, const OpAttributes& attrs) {
            if (inputs.size() < 5) throw std::invalid_argument("group_norm_backward requires 5 inputs (grad_output, input, mean, rstd, weight)");
            int64_t num_groups = attrs.get_int(AttrKey::NumGroups, 1);
            auto [grad_input, grad_weight, grad_bias] = b->dispatchGroupNormBackward(inputs[0], inputs[1], inputs[2], inputs[3], &inputs[4], num_groups);
            return std::vector<Tensor>{grad_input, grad_weight, grad_bias};
        }},
        {"embedding_backward", [](VulkanBackend* b, const std::string&, std::span<const Tensor> inputs, const OpAttributes& attrs) {
            if (inputs.size() < 2) throw std::invalid_argument("embedding_backward requires 2 inputs (grad_output, indices)");
            int64_t num_embeddings = attrs.get_int(AttrKey::NumEmbeddings, 0);
            int64_t embedding_dim = inputs[0].shape().back();
            return std::vector<Tensor>{b->dispatchEmbeddingBackward(inputs[0], inputs[1], num_embeddings, embedding_dim)};
        }},
        {"fused_rms_norm", [](VulkanBackend* b, const std::string&, std::span<const Tensor> inputs, const OpAttributes& attrs) {
            if (inputs.size() < 2) throw std::invalid_argument("fused_rms_norm requires 2 inputs (input, weight)");
            float eps = static_cast<float>(attrs.get_float(AttrKey::Eps, 1e-5));
            int64_t normalized_shape = inputs[0].shape().back();
            auto [output, rrms] = b->dispatchRMSNorm(inputs[0], inputs[1], normalized_shape, eps);
            return std::vector<Tensor>{output, rrms};
        }},
        {"rms_norm_backward", [](VulkanBackend* b, const std::string&, std::span<const Tensor> inputs, const OpAttributes& attrs) {
            if (inputs.size() < 4) throw std::invalid_argument("rms_norm_backward requires 4 inputs (grad_output, input, rrms, weight)");
            int64_t normalized_shape = inputs[0].shape().back();
            if (attrs.has(AttrKey::NormalizedShape)) normalized_shape = attrs.get_int(AttrKey::NormalizedShape);
            auto [grad_input, grad_weight] = b->dispatchRMSNormBackward(inputs[0], inputs[1], inputs[2], inputs[3], normalized_shape);
            return std::vector<Tensor>{grad_input, grad_weight};
        }},

        // ==================================================================
        // Fused operations
        // ==================================================================
        {"fused_linear_relu", [](VulkanBackend* b, const std::string&, std::span<const Tensor> inputs, const OpAttributes& attrs) {
            if (inputs.size() < 2) throw std::invalid_argument("fused_linear_relu requires at least 2 inputs (input, weight)");
            bool has_bias = attrs.get_bool(AttrKey::HasBias, false);
            auto input_shape = inputs[0].shape();
            int64_t in_features = input_shape[input_shape.size() - 1];
            int64_t out_features = inputs[1].shape()[0];
            int64_t batch_size = 1;
            for (size_t i = 0; i < input_shape.size() - 1; ++i) batch_size *= input_shape[i];
            Tensor input_2d = inputs[0].reshape({batch_size, in_features});
            Tensor weight_t = inputs[1].transpose(0, 1);
            Tensor mm_result = b->dispatchMatmul(input_2d, weight_t);
            if (has_bias && inputs.size() > 2) mm_result = b->dispatchBinaryOp("add", mm_result, inputs[2]);
            Tensor result = b->dispatchActivation("relu", mm_result, 0, 0.0f);
            std::vector<int64_t> out_shape(input_shape.begin(), input_shape.end() - 1);
            out_shape.push_back(out_features);
            result = result.reshape(out_shape);
            return std::vector<Tensor>{result};
        }},
        {"fused_batchnorm_relu", [](VulkanBackend* b, const std::string&, std::span<const Tensor> inputs, const OpAttributes& attrs) {
            if (inputs.size() < 5) throw std::invalid_argument("fused_batchnorm_relu requires 5 inputs (input, mean, var, weight, bias)");
            float eps = static_cast<float>(attrs.get_float(AttrKey::Eps, 1e-5));
            auto orig_shape = inputs[0].shape();
            Tensor input_4d = inputs[0];
            bool needs_reshape = (orig_shape.size() != 4);
            if (needs_reshape) {
                int64_t N = orig_shape[0]; int64_t C = orig_shape[1]; int64_t spatial = 1;
                for (size_t i = 2; i < orig_shape.size(); ++i) spatial *= orig_shape[i];
                input_4d = inputs[0].reshape({N, C, spatial, 1});
            }
            Tensor bn_result = b->dispatchBatchNorm2dForward(input_4d, inputs[1], inputs[2], &inputs[3], &inputs[4], eps);
            if (needs_reshape) {
                std::vector<int64_t> shape_vec(orig_shape.begin(), orig_shape.end());
                bn_result = bn_result.reshape(shape_vec);
            }
            Tensor result = b->dispatchActivation("relu", bn_result, 0, 0.0f);
            return std::vector<Tensor>{result};
        }},
        {"fused_add_relu", [](VulkanBackend* b, const std::string&, std::span<const Tensor> inputs, const OpAttributes&) {
            if (inputs.size() != 2) throw std::invalid_argument("fused_add_relu requires 2 inputs");
            Tensor add_result = b->dispatchBinaryOp("add", inputs[0], inputs[1]);
            return std::vector<Tensor>{b->dispatchActivation("relu", add_result, 0, 0.0f)};
        }},
        {"fused_gelu", [](VulkanBackend* b, const std::string&, std::span<const Tensor> inputs, const OpAttributes&) {
            if (inputs.size() != 1) throw std::invalid_argument("fused_gelu requires 1 input");
            return std::vector<Tensor>{b->dispatchActivation("gelu", inputs[0], 3, 0.0f)};
        }},
        {"fused_layer_norm", [](VulkanBackend* b, const std::string&, std::span<const Tensor> inputs, const OpAttributes& attrs) {
            if (inputs.size() < 3) throw std::invalid_argument("fused_layer_norm requires 3 inputs (input, weight, bias)");
            auto ns_str = attrs.get_string(AttrKey::NormalizedShape);
            int64_t normalized_size = 1;
            std::string ns_s{ns_str};
            std::stringstream ss(ns_s);
            std::string token;
            while (std::getline(ss, token, ',')) normalized_size *= std::stoll(token);
            float eps = static_cast<float>(attrs.get_float(AttrKey::Eps, 1e-5));
            Tensor result = b->dispatchLayerNorm(inputs[0], normalized_size, &inputs[1], &inputs[2], eps);
            return std::vector<Tensor>{result};
        }},
    };
    return table;
}

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
                // Indexing ops (put_f16, take_f16, tile_f16, stack_f16, searchsorted_f16 shaders)
                "put", "take", "tile", "stack", "searchsorted",
                // Linalg (native F16 shaders)
                "linalg_det", "linalg_inv", "linalg_solve", "linalg_eigh",
                "linalg_svd", "linalg_cholesky", "linalg_qr",
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

    // =========================================================================
    // Float64 handling: operations with native F64 shaders use them directly;
    // operations without F64 support throw immediately to prevent silent
    // corruption from dispatching F64 data to F32 shaders.
    // =========================================================================
    {
        bool has_float64 = false;
        for (const auto& t : inputs) {
            if (t.dtype() == DType::Float64) {
                has_float64 = true;
                break;
            }
        }

        if (has_float64) {
            static const std::unordered_set<std::string> f64_native_ops = {
                // Binary ops (math_f64 shader)
                "add", "sub", "mul", "div",
                "add_inplace", "sub_inplace", "mul_inplace", "div_inplace",
                // Activation ops (activations_f64 shader)
                "relu", "sigmoid", "tanh", "gelu", "leaky_relu", "swish",
                "relu_inplace", "sigmoid_inplace", "tanh_inplace",
                "gelu_inplace", "leaky_relu_inplace",
                // Activation backward (activations_backward_f64 shader)
                "relu_backward", "sigmoid_backward", "tanh_backward",
                "gelu_backward", "leaky_relu_backward",
                // Softmax (softmax_f64 shader)
                "softmax", "softmax_backward",
                // Layer norm (layer_norm_f64 shader)
                "layer_norm", "layer_norm_backward",
                // Group norm
                "group_norm", "group_norm_backward",
                // RMSNorm
                "fused_rms_norm", "rms_norm_backward",
                // Embedding
                "embedding", "embedding_backward",
                // Conv2d
                "conv2d", "conv2d_forward",
                // Adaptive pooling
                "adaptive_avg_pool2d", "adaptive_max_pool2d",
                "adaptive_avg_pool2d_backward", "adaptive_max_pool2d_backward",
                // Pooling (pooling_f64, pooling_forward_with_indices_f64 shaders)
                "pooling", "pooling_backward", "pooling_forward_with_indices",
                "max_pool2d", "max_pool2d_forward", "max_pool2d_backward",
                "avg_pool2d", "avg_pool2d_forward", "avg_pool2d_backward",
                "max_pool2d_backward_with_indices", "max_pool2d_backward_recompute",
                // Unary math (math_f64 shader)
                "sqrt", "exp", "log", "neg", "abs", "sign", "pow",
                "floor", "ceil", "round", "trunc", "reciprocal",
                // Trigonometric (trigonometric_f64 shader)
                "sin", "cos", "tan", "asin", "acos", "atan",
                // Hyperbolic (hyperbolic_f64 shader)
                "sinh", "cosh",
                // Comparison (comparison_f64 shader)
                "eq", "ne", "lt", "le", "gt", "ge",
                // Matrix ops (matmul_f64 shader)
                "matmul", "bmm", "dot",
                // Fused operations
                "fused_linear_relu", "fused_batchnorm_relu", "fused_add_relu",
                "fused_conv2d_relu", "fused_gelu", "fused_layer_norm",
                "fused_softmax_cross_entropy",
                // BatchNorm
                "batchnorm2d_forward", "batchnorm2d_forward_affine",
                "batchnorm2d_backward",
                "batchnorm2d_mean_var", "batchnorm2d_update_running_stats",
                // Reduction (reduction_f64 shader)
                "sum", "mean", "max", "min", "prod",
                "argmax", "argmin", "argsort",
                // Type-agnostic operations
                "reshape", "view", "contiguous", "to", "to_dtype",
                "zeros", "ones", "full", "empty",
                // Creation ops
                "arange", "linspace", "eye", "one_hot", "diag",
                // Clamp
                "clamp", "clamp_min", "clamp_max",
                // Shape ops (type-agnostic / metadata-only)
                "transpose", "permute", "cat", "squeeze", "unsqueeze",
                // Indexing ops
                "gather", "scatter", "masked_fill",
                "masked_select", "where", "index_select",
                // Reduction ops
                "var", "std", "norm",
                // Log softmax
                "log_softmax", "log_softmax_backward",
                // Interpolation
                "interpolate",
                // Memory ops
                "clone", "fill", "unfold", "fold",
                // Activation extras
                "elu", "elu_backward", "selu", "selu_backward",
                "mish", "mish_backward", "softplus", "softplus_backward",
                "swish_backward",
                // Sort/unique (bitonic_sort_f64, unique_mark_f64, unique_compact_f64)
                "sort", "unique",
                // Cast
                "cast",
                // Random
                "uniform_random", "normal_random",
                // FFT
                "fft", "ifft",
                // Linalg
                "linalg_det", "linalg_inv", "linalg_solve", "linalg_eigh",
                "linalg_svd", "linalg_cholesky", "linalg_qr",
                // Complex
                "conj", "real", "imag", "angle", "polar",
                // Strided copy
                "strided_copy",
                // Expand/repeat
                "expand", "repeat",
                // Conv backward
                "conv2d_backward_input", "conv2d_backward_weight", "conv2d_backward_bias",
                // ConvTranspose2d
                "conv_transpose2d_forward",
                // Indexing ops
                "put", "take", "tile", "stack", "searchsorted",
                // Cross entropy
                "cross_entropy_forward", "cross_entropy_backward",
                // NLLLoss
                "nll_loss_forward", "nll_loss_backward",
                // Embedding
                "embedding_bag",
                // Optimizer steps
                "fused_sgd_step", "fused_adagrad_step",
                "fused_adadelta_step", "fused_rmsprop_step",
            };

            if (!f64_native_ops.contains(op_name)) {
                throw std::runtime_error(
                    "Vulkan: operation '" + op_name + "' does not support Float64 dtype. "
                    "No Float64 shader available — refusing to dispatch to prevent silent data corruption.");
            }
        }
    }

    try {
    // O(1) dispatch via hash table lookup (replaces ~117 if-else string comparisons)
    // Three tables are used to keep static initialization manageable.
    const auto& t1 = get_dispatch_table();
    auto it = t1.find(op_name);
    if (it != t1.end()) {
        return it->second(this, op_name, inputs, attrs);
    }

    const auto& t2 = get_dispatch_table_2();
    it = t2.find(op_name);
    if (it != t2.end()) {
        return it->second(this, op_name, inputs, attrs);
    }

    const auto& t3 = get_dispatch_table_3();
    it = t3.find(op_name);
    if (it != t3.end()) {
        return it->second(this, op_name, inputs, attrs);
    }

    // ========================================================================
    // Operations with inline Vulkan API calls (batchnorm update stats,
    // fused optimizer steps) — kept out of the static map because they
    // reference VulkanBackend member state (devices_, pipelines, etc.)
    // ========================================================================

    if (op_name == "batchnorm2d_update_running_stats") {
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

        insertComputeBarrier(cmdBuffer);

        endSingleTimeCommands(cmdBuffer, device_id);

        return {running_mean, running_var};
    }

    if (op_name == "fused_rmsprop_step") {
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
        bool is_float64 = (inputs[0].dtype() == DType::Float64);
        std::string rmsprop_shader = is_float64 ? "fused_rmsprop_step_f64" : "fused_rmsprop_step";
        auto* pipeline = getPipeline(rmsprop_shader, device_id);

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
        if (numel > static_cast<int64_t>(UINT32_MAX)) {
            throw std::runtime_error("Vulkan: tensor numel exceeds uint32_t push constant limit");
        }
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
        if (inputs.size() < 4) {
            throw std::invalid_argument("fused_adadelta_step requires 4 inputs");
        }
        int64_t numel = inputs[0].numel();
        float lr = static_cast<float>(attrs.get_float(AttrKey::Lr, 1.0));
        float rho = static_cast<float>(attrs.get_float(AttrKey::Rho, 0.9));
        float eps = static_cast<float>(attrs.get_float(AttrKey::Eps, 1e-6));
        float weight_decay = static_cast<float>(attrs.get_float(AttrKey::WeightDecay, 0.0));

        int32_t device_id = inputs[0].device().index;
        bool is_float64 = (inputs[0].dtype() == DType::Float64);
        std::string adadelta_shader = is_float64 ? "fused_adadelta_step_f64" : "fused_adadelta_step";
        auto* pipeline = getPipeline(adadelta_shader, device_id);

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
        if (numel > static_cast<int64_t>(UINT32_MAX)) {
            throw std::runtime_error("Vulkan: tensor numel exceeds uint32_t push constant limit");
        }
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
        if (inputs.size() < 3) {
            throw std::invalid_argument("fused_adagrad_step requires 3 inputs");
        }
        int64_t numel = inputs[0].numel();
        float lr = static_cast<float>(attrs.get_float(AttrKey::Lr, 0.01));
        float eps = static_cast<float>(attrs.get_float(AttrKey::Eps, 1e-10));
        float weight_decay = static_cast<float>(attrs.get_float(AttrKey::WeightDecay, 0.0));

        int32_t device_id = inputs[0].device().index;
        bool is_float64 = (inputs[0].dtype() == DType::Float64);
        std::string adagrad_shader = is_float64 ? "fused_adagrad_step_f64" : "fused_adagrad_step";
        auto* pipeline = getPipeline(adagrad_shader, device_id);

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
        if (numel > static_cast<int64_t>(UINT32_MAX)) {
            throw std::runtime_error("Vulkan: tensor numel exceeds uint32_t push constant limit");
        }
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
