#include <iostream>
#include "tenzor/tenzor.hpp"
#include "tenzor/nn/layers/linear.hpp"
#include "tenzor/autograd/variable.hpp"
#include "tenzor/ops/creation.hpp"

using namespace tenzor;
using namespace tenzor::nn;

int main() {
    tenzor::initialize();

    std::cout << "Testing Linear layer backward pass with Float16..." << std::endl;

    // Create a simple Linear layer (small for easier debugging)
    auto linear = Linear(768, 768);

    // Convert to Float16
    linear.to(DType::Float16);
    linear.train();

    std::cout << "Created Linear layer with Float16 parameters" << std::endl;

    // Create input (batch_size=1, seq_len=64, features=768)
    auto input_tensor = randn({1, 64, 768}, DType::Float16);
    Variable input(input_tensor, true);

    std::cout << "Created input tensor: shape [1, 64, 768]" << std::endl;

    // Run forward and backward pass multiple times
    for (int i = 0; i < 5; ++i) {
        std::cout << "\n=== Iteration " << i << " ===" << std::endl;

        // Forward pass
        std::cout << "Running forward pass..." << std::endl;
        auto output = linear.forward(input);
        std::cout << "Forward pass completed, output.shape = ["
                  << output.shape()[0] << ", "
                  << output.shape()[1] << ", "
                  << output.shape()[2] << "]" << std::endl;

        // Sum to scalar for backward
        std::cout << "Computing loss (sum)..." << std::endl;
        Variable loss = tenzor::sum(output);
        std::cout << "Loss computed, dtype="
                  << (loss.dtype() == DType::Float16 ? "Float16" : "other") << std::endl;

        // Backward pass - THIS IS WHERE THE CRASH SHOULD HAPPEN IF THERE'S AN ISSUE
        std::cout << "Running backward pass..." << std::endl;
        loss.backward();
        std::cout << "Backward pass completed!" << std::endl;

        // Check gradient
        auto params = linear.named_parameters();
        bool found_weight = false;
        for (const auto& [name, param] : params) {
            if (name == "weight") {
                found_weight = true;
                auto weight_grad = param->grad();
                if (weight_grad.has_value()) {
                    std::cout << "Weight gradient exists, dtype="
                              << (weight_grad->dtype() == DType::Float16 ? "Float16" : "other")
                              << ", shape=[" << weight_grad->shape()[0] << ", " << weight_grad->shape()[1] << "]"
                              << std::endl;
                } else {
                    std::cout << "ERROR: Weight has no gradient!" << std::endl;
                    return 1;
                }
                break;
            }
        }
        if (!found_weight) {
            std::cout << "ERROR: Weight parameter not found!" << std::endl;
            return 1;
        }

        // Clear gradients for next iteration
        linear.zero_grad();
        input.zero_grad();
        std::cout << "Gradients cleared" << std::endl;
    }

    std::cout << "\nLinear backward test completed successfully!" << std::endl;
    return 0;
}
