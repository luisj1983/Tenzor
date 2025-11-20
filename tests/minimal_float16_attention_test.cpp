#include <iostream>
#include "tenzor/tenzor.hpp"
#include "tenzor/nn/layers/attention.hpp"
#include "tenzor/autograd/variable.hpp"
#include "tenzor/ops/creation.hpp"

using namespace tenzor;
using namespace tenzor::nn;

int main() {
    tenzor::initialize();

    std::cout << "Testing MultiHeadAttention layer with Float16..." << std::endl;

    // Create attention layer similar to ALBERT
    // embed_dim=768, num_heads=12 (same as ALBERT)
    auto attention = MultiheadAttention(768, 12);

    // Convert attention parameters to Float16
    attention.to(DType::Float16);

    std::cout << "Created MultiheadAttention layer with Float16 parameters" << std::endl;

    // Create input (batch_size=8, seq_len=128, embed_dim=768)
    auto input_tensor = randn({8, 128, 768}, DType::Float16);
    Variable query(input_tensor, true);

    std::cout << "Created input tensor: shape [8, 128, 768]" << std::endl;

    // Run attention forward pass multiple times
    for (int i = 0; i < 50; ++i) {
        // Self-attention: query, key, value are all the same
        auto result = attention.forward(query, query, query);
        auto output = result.first;  // Extract output from pair

        if (i % 10 == 0) {
            std::cout << "Iteration " << i << ": output.shape = ["
                      << output.shape()[0] << ", "
                      << output.shape()[1] << ", "
                      << output.shape()[2] << "]"
                      << ", dtype=" << (output.dtype() == DType::Float16 ? "Float16" : "other")
                      << std::endl;
        }

        // Verify shape and dtype
        if (output.shape()[0] != 8 || output.shape()[1] != 128 || output.shape()[2] != 768) {
            std::cerr << "ERROR: Unexpected output shape!" << std::endl;
            return 1;
        }

        // Use output as input for next iteration (creates autograd chain)
        if (i < 49) {
            query = output;
        }
    }

    std::cout << "Attention test completed successfully!" << std::endl;
    return 0;
}
