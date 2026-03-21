#include "tenzor/nn/layers/linear.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/reduction.hpp"
#include "tenzor/ops/transform.hpp"
#include "tenzor/autograd/ops.hpp"
#include "tenzor/autograd/function.hpp"
#include <cmath>
#include <iostream>

namespace tenzor::nn {

// Namespace alias for autograd operations
namespace autograd = tenzor;

// TypeCast autograd function for dtype conversion with gradient flow
// Gradients are cast back to the original (input) dtype so that parameter
// gradients match the parameter dtype (e.g., Float16 weight gets Float16 grad).
class TypeCastBackward : public Function {
public:
    DType original_dtype_ = DType::Float32;

    TypeCastBackward() = default;

    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override {
        throw std::runtime_error("TypeCastBackward::forward should not be called");
    }

    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override {
        auto& grad = grad_outputs[0];
        if (grad.dtype() != original_dtype_) {
            return {grad.to(original_dtype_)};
        }
        return {grad};
    }

    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override {
        auto& grad = grad_outputs[0];
        if (grad.dtype() != original_dtype_) {
            return {Variable(grad.tensor().to(original_dtype_), grad.requires_grad())};
        }
        return {grad};
    }
};

// Helper function to cast a Variable to a new dtype with autograd support
static auto variable_cast(const Variable& input, DType target_dtype) -> Variable {
    if (input.dtype() == target_dtype) {
        return input;
    }

    auto converted_tensor = input.tensor().to(target_dtype);
    Variable result(converted_tensor, input.requires_grad());

    if (input.requires_grad() && is_grad_enabled()) {
        auto grad_fn = std::make_shared<TypeCastBackward>();
        grad_fn->original_dtype_ = input.dtype();

        // Track input variable for gradient accumulation
        std::vector<Variable> input_vars = {input};
        grad_fn->set_input_variables(input_vars);

        // Connect to input's grad_fn
        std::vector<std::shared_ptr<Function>> next_funcs;
        if (input.grad_fn()) {
            next_funcs.push_back(input.grad_fn());
        }
        grad_fn->set_next_functions(next_funcs);

        result.set_grad_fn(grad_fn);
    }

    return result;
}

Linear::Linear(int64_t in_features, int64_t out_features, bool bias)
    : in_features_(in_features), out_features_(out_features), has_bias_(bias) {

    if (in_features <= 0) {
        throw std::runtime_error("Linear: in_features must be positive, got " +
            std::to_string(in_features));
    }
    if (out_features <= 0) {
        throw std::runtime_error("Linear: out_features must be positive, got " +
            std::to_string(out_features));
    }

    // Initialize weight with Kaiming uniform (matches PyTorch default)
    // PyTorch uses U(-bound, bound) where bound = sqrt(1 / fan_in) for linear layers
    float bound = std::sqrt(1.0f / static_cast<float>(in_features));
    Variable weight(rand({out_features, in_features}) * (2.0f * bound) - bound, true);
    register_parameter("weight", std::move(weight));

    // Initialize bias with uniform(-1/sqrt(in_features), 1/sqrt(in_features))
    if (bias) {
        float bound = 1.0f / std::sqrt(static_cast<float>(in_features));
        Variable bias_var(rand({out_features}) * (2.0f * bound) - bound, true);
        register_parameter("bias", std::move(bias_var));
    }
}

// Helper function to compute linear using matmul (fallback for backends without fused linear)
static auto linear_via_matmul(const Variable& input, const Variable& weight,
                               const Variable* bias) -> Variable {
    // Transpose weight: [out_features, in_features] -> [in_features, out_features]
    auto weight_t = autograd::permute(weight, {1, 0});

    // Matrix multiplication: input @ weight.T
    auto output = autograd::matmul(input, weight_t);

    // Add bias if present
    if (bias) {
        output = output + *bias;
    }

    return output;
}

// Check if fused linear kernel is available for this backend
static bool has_fused_linear_kernel(Device device) {
    // CPU and CUDA backends have fused linear kernels
    return device.type == Device::Type::CPU || device.type == Device::Type::CUDA;
}

auto Linear::forward_impl(const Variable& input) -> Variable {
    // input: [*, in_features] where * can be any number of dimensions
    // weight: [out_features, in_features]
    // output: [*, out_features]

    auto input_shape = input.shape();
    if (input_shape.empty()) {
        throw std::invalid_argument("Linear: input tensor must have at least 1 dimension");
    }
    if (input_shape.back() != in_features_) {
        throw std::invalid_argument("Linear: expected input last dim=" +
            std::to_string(in_features_) + ", got " + std::to_string(input_shape.back()));
    }
    const bool is_2d = (input_shape.size() == 2);

    auto& weight = *parameters_.at("weight");

    // Fast path: 2D input - skip reshape operations entirely
    // This eliminates 2 ReshapeBackward allocations per forward pass
    if (is_2d) {
        DType compute_dtype = input.dtype();

        // Handle device mismatch - transfer weight/bias to input's device via autograd
        // This keeps computation on the input's device (e.g., GPU), avoiding
        // cross-device data access issues with backend-specific runtimes
        Variable weight_device = weight;
        if (input.tensor().device() != weight.tensor().device()) {
            weight_device = tenzor::to_device(weight, input.tensor().device());
        }

        // Handle dtype mismatch - convert weight/bias to input's dtype using gradient-aware cast
        // This ensures gradients flow back to weight with proper dtype conversion
        Variable weight_matched = variable_cast(weight_device, compute_dtype);

        // Get bias and convert if needed
        Variable bias_matched;
        Variable* bias_ptr = nullptr;
        if (has_bias_) {
            auto bias_it = parameters_.find("bias");
            if (bias_it != parameters_.end()) {
                Variable bias_device = *bias_it->second;
                if (input.tensor().device() != bias_device.tensor().device()) {
                    bias_device = tenzor::to_device(bias_device, input.tensor().device());
                }
                bias_matched = variable_cast(bias_device, compute_dtype);
                bias_ptr = &bias_matched;
            }
        }

        // Use fused linear kernel if available (CPU with MKL, CUDA with cuBLAS)
        // Falls back to matmul + add for other backends
        if (has_fused_linear_kernel(input.tensor().device())) {
            if (!bias_ptr) {
                // No bias: use matmul path to avoid unnecessary zero tensor allocation
                return linear_via_matmul(input, weight_matched, nullptr);
            } else {
                return autograd::linear(input, weight_matched, *bias_ptr);
            }
        } else {
            return linear_via_matmul(input, weight_matched, bias_ptr);
        }
    }

    // General path: N-D input requires reshape
    std::vector<int64_t> original_shape(input_shape.begin(), input_shape.end());
    DType compute_dtype = input.dtype();

    // Calculate total batch size
    int64_t batch_total = 1;
    for (size_t i = 0; i < original_shape.size() - 1; ++i) {
        batch_total *= original_shape[i];
    }

    // Flatten input to 2D
    std::vector<int64_t> flat_shape = {batch_total, in_features_};
    auto input_2d = autograd::reshape(input, flat_shape);

    // Handle device mismatch - transfer weight/bias to input's device via autograd
    Variable weight_device = weight;
    if (input_2d.tensor().device() != weight.tensor().device()) {
        weight_device = tenzor::to_device(weight, input_2d.tensor().device());
    }

    // Handle dtype mismatch - convert weight/bias to input's dtype using gradient-aware cast
    Variable weight_matched = variable_cast(weight_device, compute_dtype);

    // Get bias and convert if needed
    Variable bias_matched;
    Variable* bias_ptr = nullptr;
    if (has_bias_) {
        auto bias_it = parameters_.find("bias");
        if (bias_it != parameters_.end()) {
            Variable bias_device = *bias_it->second;
            if (input_2d.tensor().device() != bias_device.tensor().device()) {
                bias_device = tenzor::to_device(bias_device, input_2d.tensor().device());
            }
            bias_matched = variable_cast(bias_device, compute_dtype);
            bias_ptr = &bias_matched;
        }
    }

    // Compute linear operation
    Variable output_2d;
    if (has_fused_linear_kernel(input_2d.tensor().device()) && bias_ptr) {
        // Use fused linear kernel (CPU with MKL, CUDA with cuBLAS) — only when bias present
        output_2d = autograd::linear(input_2d, weight_matched, *bias_ptr);
    } else {
        // Fallback: use matmul + add for other backends
        output_2d = linear_via_matmul(input_2d, weight_matched, bias_ptr);
    }

    // Reshape output back
    std::vector<int64_t> output_shape = original_shape;
    output_shape.back() = out_features_;
    return autograd::reshape(output_2d, output_shape);
}

auto Linear::reset_parameters() -> void {
    float bound = std::sqrt(1.0f / static_cast<float>(in_features_));
    Variable weight(rand({out_features_, in_features_}) * (2.0f * bound) - bound, true);
    register_parameter("weight", std::move(weight));

    if (has_bias_) {
        Variable bias_var(rand({out_features_}) * (2.0f * bound) - bound, true);
        register_parameter("bias", std::move(bias_var));
    }
}

} // namespace tenzor::nn
