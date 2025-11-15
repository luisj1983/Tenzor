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

auto Linear::forward(const Variable& input) -> Variable {
    // input: [*, in_features] where * can be any number of dimensions
    // weight: [out_features, in_features]
    // output: [*, out_features]

    if (input.dtype() == DType::Float16) {
        std::cerr << "[LINEAR_F16] Starting Float16 forward, in_features=" << in_features_
                  << ", out_features=" << out_features_ << std::endl;
    }

    // Save original input shape (convert span to vector)
    if (input.dtype() == DType::Float16) {
        std::cerr << "[LINEAR_F16] About to save input shape" << std::endl;
    }
    std::vector<int64_t> original_shape(input.shape().begin(), input.shape().end());
    if (input.dtype() == DType::Float16) {
        std::cerr << "[LINEAR_F16] Saved input shape" << std::endl;
    }

    // Calculate total batch size (product of all dimensions except last)
    int64_t batch_total = 1;
    for (size_t i = 0; i < original_shape.size() - 1; ++i) {
        batch_total *= original_shape[i];
    }

    // Flatten input to 2D: (batch_total, in_features) using autograd reshape
    std::vector<int64_t> flat_shape = {batch_total, in_features_};
    try {
        if (input.dtype() == DType::Float16) {
            std::cerr << "[LINEAR_F16] About to reshape input" << std::endl;
        }
        auto input_2d = autograd::reshape(input, flat_shape);
        if (input.dtype() == DType::Float16) {
            std::cerr << "[LINEAR_F16] Reshaped input" << std::endl;
        }

        // Get weight from parameters (ensures correct device)
        auto& weight_ptr = parameters_["weight"];
        auto& weight = *weight_ptr;

        // Handle device mismatch: transfer input to weight's device if needed
        Variable input_2d_device = input_2d;
        if (input_2d.tensor().device() != weight.tensor().device()) {
            // Transfer input to weight's device
            auto input_transferred = input_2d.tensor().to(weight.tensor().device());
            input_2d_device = Variable(input_transferred, input_2d.requires_grad());
            input_2d_device.set_grad_fn(input_2d.grad_fn());
        }

        // Handle dtype mismatch: convert weight to input's dtype if needed
        Variable weight_dtype_matched = weight;
        if (input_2d_device.dtype() != weight.dtype()) {
            auto weight_converted = weight.tensor().to(input_2d_device.dtype());
            weight_dtype_matched = Variable(weight_converted, weight.requires_grad());
            weight_dtype_matched.set_grad_fn(weight.grad_fn());
        }

        // Compute output = input_2d @ weight.T
        // input_2d: (batch_total, in_features)
        // weight: (out_features, in_features)
        // weight.T: (in_features, out_features)
        // output: (batch_total, out_features)

        // Transpose weight
        auto weight_t = autograd::permute(weight_dtype_matched, {1, 0});  // (in_features, out_features)

        // Matrix multiplication (use device-matched input)
        if (input.dtype() == DType::Float16) {
            std::cerr << "[LINEAR_F16] About to call matmul" << std::endl;
        }
        auto output_2d = autograd::matmul(input_2d_device, weight_t);  // (batch_total, out_features)
        if (input.dtype() == DType::Float16) {
            std::cerr << "[LINEAR_F16] Matmul completed" << std::endl;
        }

        // Reshape output back to original dimensions: [*, out_features]
        std::vector<int64_t> output_shape = original_shape;
        output_shape.back() = out_features_;
        auto output = autograd::reshape(output_2d, output_shape);
        if (input.dtype() == DType::Float16) {
            std::cerr << "[LINEAR_F16] Reshape completed" << std::endl;
        }

        // Add bias if present (Variable operators already use autograd)
        auto bias_it = parameters_.find("bias");
        if (bias_it != parameters_.end()) {
            auto& bias_ptr = bias_it->second;
            auto& bias = *bias_ptr;

            // Convert bias to match input dtype if needed
            Variable bias_dtype_matched = bias;
            if (input_2d_device.dtype() != bias.dtype()) {
                auto bias_converted = bias.tensor().to(input_2d_device.dtype());
                bias_dtype_matched = Variable(bias_converted, bias.requires_grad());
                bias_dtype_matched.set_grad_fn(bias.grad_fn());
            }

            // Native broadcasting: bias [out_features] + output [*, out_features]
            if (input.dtype() == DType::Float16) {
                std::cerr << "[LINEAR_F16] Adding bias" << std::endl;
            }
            output = output + bias_dtype_matched;
            if (input.dtype() == DType::Float16) {
                std::cerr << "[LINEAR_F16] Bias addition completed" << std::endl;
            }
        }

        if (input.dtype() == DType::Float16) {
            std::cerr << "[LINEAR_F16] Completed Float16 forward" << std::endl;
        }

        return output;
    } catch (const std::exception& e) {
        throw;
    }
}

auto Linear::reset_parameters() -> void {
    // Xavier/Glorot initialization already done in constructor
}

} // namespace tenzor::nn
