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

    std::cout << "    Linear::forward - input shape: [";
    for (size_t i = 0; i < input.shape().size(); ++i) {
        std::cout << input.shape()[i];
        if (i < input.shape().size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    // Save original input shape (convert span to vector)
    std::vector<int64_t> original_shape(input.shape().begin(), input.shape().end());

    // Calculate total batch size (product of all dimensions except last)
    int64_t batch_total = 1;
    for (size_t i = 0; i < original_shape.size() - 1; ++i) {
        batch_total *= original_shape[i];
    }

    std::cout << "    Linear::forward - batch_total: " << batch_total << ", in_features: " << in_features_ << std::endl;

    // Flatten input to 2D: (batch_total, in_features) using autograd reshape
    std::vector<int64_t> flat_shape = {batch_total, in_features_};
    std::cout << "    Linear::forward - reshaping to [" << flat_shape[0] << ", " << flat_shape[1] << "]..." << std::endl;
    try {
        auto input_2d = autograd::reshape(input, flat_shape);
        std::cout << "    Linear::forward - reshape OK, shape: [" << input_2d.shape()[0] << ", " << input_2d.shape()[1] << "]" << std::endl;

        // Get weight from parameters (ensures correct device)
        auto& weight_ptr = parameters_["weight"];
        auto& weight = *weight_ptr;
        std::cout << "    Linear::forward - weight shape: [" << weight.shape()[0] << ", " << weight.shape()[1] << "]" << std::endl;

        // TODO: Handle device mismatch properly with device transfer op
        // For now, assume weight and input are on same device

        // Compute output = input_2d @ weight.T
        // input_2d: (batch_total, in_features)
        // weight: (out_features, in_features)
        // weight.T: (in_features, out_features)
        // output: (batch_total, out_features)

        // Transpose weight
        std::cout << "    Linear::forward - permuting weight..." << std::endl;
        auto weight_t = autograd::permute(weight, {1, 0});  // (in_features, out_features)
        std::cout << "    Linear::forward - permute OK, weight_t shape: [" << weight_t.shape()[0] << ", " << weight_t.shape()[1] << "]" << std::endl;

        // Matrix multiplication
        std::cout << "    Linear::forward - matmul [" << input_2d.shape()[0] << ", " << input_2d.shape()[1]
                  << "] @ [" << weight_t.shape()[0] << ", " << weight_t.shape()[1] << "]..." << std::endl;
        auto output_2d = autograd::matmul(input_2d, weight_t);  // (batch_total, out_features)
        std::cout << "    Linear::forward - matmul OK, output_2d shape: [" << output_2d.shape()[0] << ", " << output_2d.shape()[1] << "]" << std::endl;

        // Reshape output back to original dimensions: [*, out_features]
        std::vector<int64_t> output_shape = original_shape;
        output_shape.back() = out_features_;
        std::cout << "    Linear::forward - reshaping output back to [";
        for (size_t i = 0; i < output_shape.size(); ++i) {
            std::cout << output_shape[i];
            if (i < output_shape.size() - 1) std::cout << ", ";
        }
        std::cout << "]..." << std::endl;
        auto output = autograd::reshape(output_2d, output_shape);
        std::cout << "    Linear::forward - reshape OK" << std::endl;

        // Add bias if present (Variable operators already use autograd)
        auto bias_it = parameters_.find("bias");
        if (bias_it != parameters_.end()) {
            auto& bias_ptr = bias_it->second;
            auto& bias = *bias_ptr;
            std::cout << "    Linear::forward - adding bias [" << bias.shape()[0] << "] to output [";
            for (size_t i = 0; i < output.shape().size(); ++i) {
                std::cout << output.shape()[i];
                if (i < output.shape().size() - 1) std::cout << ", ";
            }
            std::cout << "]..." << std::endl;
            // Native broadcasting: bias [out_features] + output [*, out_features]
            output = output + bias;
            std::cout << "    Linear::forward - bias add OK" << std::endl;
        }

        return output;
    } catch (const std::exception& e) {
        std::cout << "    Linear::forward - ERROR: " << e.what() << std::endl;
        throw;
    }
}

auto Linear::reset_parameters() -> void {
    // Xavier/Glorot initialization already done in constructor
}

} // namespace tenzor::nn
