#include <iostream>
#include "tenzor/tenzor.hpp"
#include "tenzor/nn/layers/linear.hpp"
#include "tenzor/autograd/variable.hpp"
#include "tenzor/autograd/ops.hpp"
#include "tenzor/ops/creation.hpp"

using namespace tenzor;
using namespace tenzor::nn;

int main() {
    tenzor::initialize();

    std::cout << "Testing Linear -> reshape -> permute backward with Float16..." << std::endl;

    // Simulate what happens in attention:
    // 1. Linear projection
    // 2. Reshape for multi-head
    // 3. Permute to (batch, num_heads, seq_len, head_dim)

    // Create Linear layer
    auto linear = Linear(768, 768);
    linear.to(DType::Float16);
    linear.train();

    // Input: (batch=8, seq_len=64, embed_dim=768)
    auto input_tensor = randn({8, 64, 768}, DType::Float16);
    Variable input(input_tensor, true);

    std::cout << "Created Linear layer and input tensor" << std::endl;

    int64_t num_heads = 12;
    int64_t head_dim = 768 / num_heads;  // 64

    // Test 3 iterations
    for (int i = 0; i < 3; ++i) {
        std::cout << "\n=== Iteration " << i << " ===" << std::endl;

        // Step 1: Linear projection
        std::cout << "Step 1: Linear projection..." << std::endl;
        auto projected = linear.forward(input);

        // Step 2: Reshape (like transpose_for_scores)
        // (batch, seq_len, embed_dim) -> (batch, seq_len, num_heads, head_dim)
        std::cout << "Step 2: Reshaping..." << std::endl;
        auto reshaped = reshape(projected, {8, 64, num_heads, head_dim});

        // Step 3: Permute
        // (batch, seq_len, num_heads, head_dim) -> (batch, num_heads, seq_len, head_dim)
        std::cout << "Step 3: Permuting..." << std::endl;
        auto permuted = permute(reshaped, {0, 2, 1, 3});

        // Sum to scalar
        std::cout << "Computing loss (sum)..." << std::endl;
        Variable loss = tenzor::sum(permuted);

        // Backward pass - THIS IS WHERE THE HANG MIGHT OCCUR
        std::cout << "Running backward pass..." << std::endl;
        loss.backward();
        std::cout << "Backward pass completed!" << std::endl;

        // Check gradients
        auto params = linear.named_parameters();
        for (const auto& [name, param] : params) {
            if (name == "weight") {
                auto weight_grad = param->grad();
                if (weight_grad.has_value()) {
                    std::cout << "Weight gradient exists, dtype="
                              << (weight_grad->dtype() == DType::Float16 ? "Float16" : "other") << std::endl;
                }
                break;
            }
        }

        // Clear gradients
        linear.zero_grad();
        input.zero_grad();
        std::cout << "Gradients cleared" << std::endl;
    }

    std::cout << "\nLinear + reshape + permute backward test completed successfully!" << std::endl;
    return 0;
}
