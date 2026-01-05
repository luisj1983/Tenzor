#include "tenzor/nn/layers/linear.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/reduction.hpp"
#include "tenzor/ops/transform.hpp"
#include "tenzor/autograd/ops.hpp"
#include <cmath>
#include <iostream>

namespace tenzor::nn {

// Namespace alias for autograd operations
namespace autograd = tenzor;

Linear::Linear(int64_t in_features, int64_t out_features, bool bias)
    : in_features_(in_features), out_features_(out_features), has_bias_(bias) {

    // Initialize weight with Xavier/Glorot initialization
    float std = std::sqrt(2.0f / (in_features + out_features));
    Variable weight(randn({out_features, in_features}) * std, true);
    register_parameter("weight", std::move(weight));

    // Initialize bias
    if (bias) {
        Variable bias_var(zeros({out_features}), true);
        register_parameter("bias", std::move(bias_var));
    }
}

// Helper function to compute linear using matmul (works on all backends)
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

auto Linear::forward_impl(const Variable& input) -> Variable {
    // input: [*, in_features] where * can be any number of dimensions
    // weight: [out_features, in_features]
    // output: [*, out_features]

    auto input_shape = input.shape();
    const bool is_2d = (input_shape.size() == 2);

    // Get weight and bias from parameters
    auto& weight_ptr = parameters_["weight"];
    auto& weight = *weight_ptr;

    // Check if we're on CPU - use fused linear kernel for better performance
    const bool is_cpu = (weight.tensor().device().type == Device::Type::CPU);

    // Fast path: 2D input - skip reshape operations entirely
    // This eliminates 2 ReshapeBackward allocations per forward pass
    if (is_2d) {
        // Handle device mismatch
        Variable input_device = input;
        if (input.tensor().device() != weight.tensor().device()) {
            auto input_transferred = input.tensor().to(weight.tensor().device());
            input_device = Variable(input_transferred, input.requires_grad());
            input_device.set_grad_fn(input.grad_fn());
        }

        // Handle dtype mismatch
        Variable weight_matched = weight;
        if (input_device.dtype() != weight.dtype()) {
            auto weight_converted = weight.tensor().to(input_device.dtype());
            weight_matched = Variable(weight_converted, weight.requires_grad());
            weight_matched.set_grad_fn(weight.grad_fn());
        }

        // Get bias
        Variable bias_matched;
        Variable* bias_ptr = nullptr;
        auto bias_it = parameters_.find("bias");
        if (bias_it != parameters_.end()) {
            auto& bias = *bias_it->second;
            bias_matched = bias;
            if (input_device.dtype() != bias.dtype()) {
                auto bias_converted = bias.tensor().to(input_device.dtype());
                bias_matched = Variable(bias_converted, bias.requires_grad());
                bias_matched.set_grad_fn(bias.grad_fn());
            }
            bias_ptr = &bias_matched;
        }

        // CPU: use fused linear kernel with MKL for optimal performance
        // GPU: use matmul + add which works on all backends
        if (is_cpu) {
            // Create zero bias if needed for fused kernel
            if (!bias_ptr) {
                auto zero_bias = zeros({out_features_}, input_device.dtype(), input_device.tensor().device());
                bias_matched = Variable(zero_bias, false);
            }
            return autograd::linear(input_device, weight_matched, bias_matched);
        } else {
            return linear_via_matmul(input_device, weight_matched, bias_ptr);
        }
    }

    // General path: N-D input requires reshape
    std::vector<int64_t> original_shape(input_shape.begin(), input_shape.end());

    // Calculate total batch size
    int64_t batch_total = 1;
    for (size_t i = 0; i < original_shape.size() - 1; ++i) {
        batch_total *= original_shape[i];
    }

    // Flatten input to 2D
    std::vector<int64_t> flat_shape = {batch_total, in_features_};
    auto input_2d = autograd::reshape(input, flat_shape);

    // Handle device mismatch
    Variable input_2d_device = input_2d;
    if (input_2d.tensor().device() != weight.tensor().device()) {
        auto input_transferred = input_2d.tensor().to(weight.tensor().device());
        input_2d_device = Variable(input_transferred, input_2d.requires_grad());
        input_2d_device.set_grad_fn(input_2d.grad_fn());
    }

    // Handle dtype mismatch
    Variable weight_matched = weight;
    if (input_2d_device.dtype() != weight.dtype()) {
        auto weight_converted = weight.tensor().to(input_2d_device.dtype());
        weight_matched = Variable(weight_converted, weight.requires_grad());
        weight_matched.set_grad_fn(weight.grad_fn());
    }

    // Get bias
    Variable bias_matched;
    Variable* bias_ptr = nullptr;
    auto bias_it = parameters_.find("bias");
    if (bias_it != parameters_.end()) {
        auto& bias = *bias_it->second;
        bias_matched = bias;
        if (input_2d_device.dtype() != bias.dtype()) {
            auto bias_converted = bias.tensor().to(input_2d_device.dtype());
            bias_matched = Variable(bias_converted, bias.requires_grad());
            bias_matched.set_grad_fn(bias.grad_fn());
        }
        bias_ptr = &bias_matched;
    }

    // Compute linear operation
    Variable output_2d;
    if (is_cpu) {
        // CPU: use fused linear kernel with MKL
        if (!bias_ptr) {
            auto zero_bias = zeros({out_features_}, input_2d_device.dtype(), input_2d_device.tensor().device());
            bias_matched = Variable(zero_bias, false);
        }
        output_2d = autograd::linear(input_2d_device, weight_matched, bias_matched);
    } else {
        // GPU: use matmul + add
        output_2d = linear_via_matmul(input_2d_device, weight_matched, bias_ptr);
    }

    // Reshape output back
    std::vector<int64_t> output_shape = original_shape;
    output_shape.back() = out_features_;
    return autograd::reshape(output_2d, output_shape);
}

auto Linear::reset_parameters() -> void {
    // Xavier/Glorot initialization already done in constructor
}

} // namespace tenzor::nn
