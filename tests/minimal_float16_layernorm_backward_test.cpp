#include <iostream>
#include "tenzor/tenzor.hpp"
#include "tenzor/nn/layers/normalization.hpp"
#include "tenzor/autograd/variable.hpp"
#include "tenzor/ops/creation.hpp"

using namespace tenzor;
using namespace tenzor::nn;

int main() {
    tenzor::initialize();

    std::cout << "Testing LayerNorm layer backward pass with Float16..." << std::endl;

    // Create LayerNorm layer (normalized_shape=768)
    auto layernorm = LayerNorm({768});

    // Convert to Float16
    layernorm.to(DType::Float16);
    layernorm.train();

    std::cout << "Created LayerNorm layer with Float16 parameters" << std::endl;

    // Create input (batch_size=8, seq_len=64, features=768)
    auto input_tensor = randn({8, 64, 768}, DType::Float16);
    Variable input(input_tensor, true);

    std::cout << "Created input tensor: shape [8, 64, 768]" << std::endl;

    // Run forward and backward pass multiple times
    for (int i = 0; i < 5; ++i) {
        std::cout << "\n=== Iteration " << i << " ===" << std::endl;

        // Forward pass
        std::cout << "Running forward pass..." << std::endl;
        auto output = layernorm.forward(input);
        std::cout << "Forward pass completed, output.shape = ["
                  << output.shape()[0] << ", "
                  << output.shape()[1] << ", "
                  << output.shape()[2] << "]" << std::endl;

        // Sum to scalar for backward
        std::cout << "Computing loss (sum)..." << std::endl;
        Variable loss = tenzor::sum(output);
        std::cout << "Loss computed, dtype="
                  << (loss.dtype() == DType::Float16 ? "Float16" : "other") << std::endl;

        // Backward pass
        std::cout << "Running backward pass..." << std::endl;
        loss.backward();
        std::cout << "Backward pass completed!" << std::endl;

        // Check gradient
        auto params = layernorm.named_parameters();
        if (!params.empty()) {
            bool found_weight = false;
            for (const auto& [name, param] : params) {
                if (name == "weight") {
                    found_weight = true;
                    auto weight_grad = param->grad();
                    if (weight_grad.has_value()) {
                        std::cout << "Weight gradient exists, dtype="
                                  << (weight_grad->dtype() == DType::Float16 ? "Float16" : "other")
                                  << ", numel=" << weight_grad->numel()
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
        } else {
            std::cout << "LayerNorm has no parameters (affine=false)" << std::endl;
        }

        // Clear gradients for next iteration
        layernorm.zero_grad();
        input.zero_grad();
        std::cout << "Gradients cleared" << std::endl;
    }

    std::cout << "\nLayerNorm backward test completed successfully!" << std::endl;
    return 0;
}
