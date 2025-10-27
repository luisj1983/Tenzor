#include "tenzor/nn/layers/linear.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/reduction.hpp"
#include "tenzor/ops/transform.hpp"
#include "tenzor/autograd/ops.hpp"
#include <cmath>

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

    // Save original input shape (convert span to vector)
    std::vector<int64_t> original_shape(input.shape().begin(), input.shape().end());

    // Calculate total batch size (product of all dimensions except last)
    int64_t batch_total = 1;
    for (size_t i = 0; i < original_shape.size() - 1; ++i) {
        batch_total *= original_shape[i];
    }

    // Flatten input to 2D: (batch_total, in_features) using autograd reshape
    std::vector<int64_t> flat_shape = {batch_total, in_features_};
    try {
        auto input_2d = autograd::reshape(input, flat_shape);

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

        // Compute output = input_2d @ weight.T
        // input_2d: (batch_total, in_features)
        // weight: (out_features, in_features)
        // weight.T: (in_features, out_features)
        // output: (batch_total, out_features)

        // Transpose weight
        auto weight_t = autograd::permute(weight, {1, 0});  // (in_features, out_features)

        // Matrix multiplication (use device-matched input)
        auto output_2d = autograd::matmul(input_2d_device, weight_t);  // (batch_total, out_features)

        // Reshape output back to original dimensions: [*, out_features]
        std::vector<int64_t> output_shape = original_shape;
        output_shape.back() = out_features_;
        auto output = autograd::reshape(output_2d, output_shape);

        // Add bias if present (Variable operators already use autograd)
        auto bias_it = parameters_.find("bias");
        if (bias_it != parameters_.end()) {
            auto& bias_ptr = bias_it->second;
            auto& bias = *bias_ptr;
            // Native broadcasting: bias [out_features] + output [*, out_features]
            output = output + bias;
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
