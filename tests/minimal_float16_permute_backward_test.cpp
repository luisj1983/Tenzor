#include <iostream>
#include "tenzor/tenzor.hpp"
#include "tenzor/autograd/variable.hpp"
#include "tenzor/autograd/ops.hpp"
#include "tenzor/ops/creation.hpp"

using namespace tenzor;

int main() {
    tenzor::initialize();

    std::cout << "Testing permute backward pass with Float16..." << std::endl;

    // Test the permute pattern used in attention:
    // Transpose key: (batch*num_heads, seq_len_k, head_dim) -> (batch*num_heads, head_dim, seq_len_k)
    // This is what attention does at line 147 with permutation {0, 2, 1}

    auto input_tensor = randn({96, 16, 64}, DType::Float16);
    Variable input(input_tensor, true);  // requires_grad=true

    std::cout << "Created input tensor: shape [96, 16, 64]" << std::endl;

    // Test 3 iterations
    for (int i = 0; i < 3; ++i) {
        std::cout << "\n=== Iteration " << i << " ===" << std::endl;

        // Forward pass - permute like attention does for key transpose
        std::cout << "Running permute forward..." << std::endl;
        auto output = permute(input, {0, 2, 1});
        std::cout << "Forward completed, output.shape = ["
                  << output.shape()[0] << ", "
                  << output.shape()[1] << ", "
                  << output.shape()[2] << "]" << std::endl;

        // Sum to scalar
        std::cout << "Computing loss (sum)..." << std::endl;
        Variable loss = tenzor::sum(output);
        std::cout << "Loss computed" << std::endl;

        // Backward pass
        std::cout << "Running backward pass..." << std::endl;
        loss.backward();
        std::cout << "Backward pass completed!" << std::endl;

        // Check gradient
        if (input.grad().has_value()) {
            std::cout << "Gradient exists, dtype="
                      << (input.grad()->dtype() == DType::Float16 ? "Float16" : "other")
                      << ", shape=[" << input.grad()->shape()[0] << ", "
                      << input.grad()->shape()[1] << ", " << input.grad()->shape()[2] << "]"
                      << std::endl;
        } else {
            std::cout << "ERROR: Gradient is missing!" << std::endl;
            return 1;
        }

        // Clear gradient
        input.zero_grad();
        std::cout << "Gradient cleared" << std::endl;
    }

    std::cout << "\nPermute backward test completed successfully!" << std::endl;
    return 0;
}
