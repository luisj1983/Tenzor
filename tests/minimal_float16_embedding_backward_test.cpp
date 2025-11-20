#include <iostream>
#include "tenzor/tenzor.hpp"
#include "tenzor/nn/layers/embedding.hpp"
#include "tenzor/autograd/variable.hpp"
#include "tenzor/ops/creation.hpp"

using namespace tenzor;
using namespace tenzor::nn;

int main() {
    tenzor::initialize();

    std::cout << "Testing Embedding layer backward pass with Float16..." << std::endl;

    // Create Embedding layer (vocab_size=30000, embedding_dim=128)
    auto embedding = Embedding(30000, 128);

    // Convert to Float16
    embedding.to(DType::Float16);
    embedding.train();

    std::cout << "Created Embedding layer with Float16 parameters" << std::endl;

    // Create input IDs (batch_size=8, seq_len=64)
    // Use full() to create constant token IDs
    auto input_ids_tensor = full({8, 64}, 100.0f, DType::Int64);
    Variable input_ids(input_ids_tensor, false);  // No gradient needed for indices

    std::cout << "Created input IDs: shape [8, 64]" << std::endl;

    // Run forward and backward pass multiple times
    for (int i = 0; i < 5; ++i) {
        std::cout << "\n=== Iteration " << i << " ===" << std::endl;

        // Forward pass
        std::cout << "Running forward pass..." << std::endl;
        auto output = embedding.forward(input_ids);
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
        auto params = embedding.named_parameters();
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
        embedding.zero_grad();
        std::cout << "Gradients cleared" << std::endl;
    }

    std::cout << "\nEmbedding backward test completed successfully!" << std::endl;
    return 0;
}
