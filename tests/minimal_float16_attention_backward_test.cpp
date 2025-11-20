#include <iostream>
#include "tenzor/tenzor.hpp"
#include "tenzor/nn/layers/attention.hpp"
#include "tenzor/autograd/variable.hpp"
#include "tenzor/ops/creation.hpp"

using namespace tenzor;
using namespace tenzor::nn;

int main() {
    tenzor::initialize();

    std::cout << "Testing MultiHeadAttention layer backward pass with Float16..." << std::endl;

    // Create attention layer (embed_dim=768, num_heads=12 - same as ALBERT)
    auto attention = MultiheadAttention(768, 12);

    // Convert to Float16
    attention.to(DType::Float16);
    attention.train();

    std::cout << "Created MultiheadAttention layer with Float16 parameters" << std::endl;

    // Create input (batch_size=8, seq_len=64, embed_dim=768)
    auto input_tensor = randn({8, 64, 768}, DType::Float16);
    Variable query(input_tensor, true);

    std::cout << "Created input tensor: shape [8, 64, 768]" << std::endl;

    // Run forward and backward pass multiple times
    for (int i = 0; i < 5; ++i) {
        std::cout << "\n=== Iteration " << i << " ===" << std::endl;

        // Forward pass - self-attention
        std::cout << "Running forward pass..." << std::endl;
        auto result = attention.forward(query, query, query);
        auto output = result.first;  // Extract output from pair
        std::cout << "Forward pass completed, output.shape = ["
                  << output.shape()[0] << ", "
                  << output.shape()[1] << ", "
                  << output.shape()[2] << "]" << std::endl;

        // Sum to scalar for backward
        std::cout << "Computing loss (sum)..." << std::endl;
        Variable loss = tenzor::sum(output);
        std::cout << "Loss computed, dtype="
                  << (loss.dtype() == DType::Float16 ? "Float16" : "other") << std::endl;

        // Backward pass - THIS IS WHERE THE CRASH MIGHT HAPPEN
        std::cout << "Running backward pass..." << std::endl;
        loss.backward();
        std::cout << "Backward pass completed!" << std::endl;

        // Check gradients for one of the attention parameters
        auto params = attention.named_parameters();
        bool found_param = false;
        for (const auto& [name, param] : params) {
            if (name.find("q_proj.weight") != std::string::npos) {
                found_param = true;
                auto param_grad = param->grad();
                if (param_grad.has_value()) {
                    std::cout << "Query projection gradient exists, dtype="
                              << (param_grad->dtype() == DType::Float16 ? "Float16" : "other")
                              << ", shape=[" << param_grad->shape()[0] << ", " << param_grad->shape()[1] << "]"
                              << std::endl;
                } else {
                    std::cout << "ERROR: Query projection has no gradient!" << std::endl;
                    return 1;
                }
                break;
            }
        }
        if (!found_param) {
            std::cout << "WARNING: q_proj.weight parameter not found, checking other params..." << std::endl;
            // Just verify we have some gradients
            int grad_count = 0;
            for (const auto& [name, param] : params) {
                if (param->grad().has_value()) {
                    grad_count++;
                }
            }
            std::cout << "Found " << grad_count << " parameters with gradients" << std::endl;
        }

        // Clear gradients for next iteration
        attention.zero_grad();
        query.zero_grad();
        std::cout << "Gradients cleared" << std::endl;
    }

    std::cout << "\nAttention backward test completed successfully!" << std::endl;
    return 0;
}
