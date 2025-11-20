#include <iostream>
#include "tenzor/tenzor.hpp"
#include "tenzor/autograd/variable.hpp"
#include "tenzor/autograd/ops.hpp"
#include "tenzor/ops/creation.hpp"

using namespace tenzor;

int main() {
    tenzor::initialize();

    std::cout << "Testing reshape backward pass with Float16..." << std::endl;

    // Test the reshape pattern used in attention:
    // (batch, num_heads, seq_len, head_dim) -> (batch*num_heads, seq_len, head_dim)
    // This is what attention does at line 141

    auto input_tensor = randn({8, 12, 16, 64}, DType::Float16);
    Variable input(input_tensor, true);  // requires_grad=true

    std::cout << "Created input tensor: shape [8, 12, 16, 64]" << std::endl;

    // Test 3 iterations
    for (int i = 0; i < 3; ++i) {
        std::cout << "\n=== Iteration " << i << " ===" << std::endl;

        // Forward pass - reshape like attention does
        std::cout << "Running reshape forward..." << std::endl;
        auto output = reshape(input, {8 * 12, 16, 64});
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
                      << input.grad()->shape()[1] << ", " << input.grad()->shape()[2] << ", "
                      << input.grad()->shape()[3] << "]"
                      << std::endl;
        } else {
            std::cout << "ERROR: Gradient is missing!" << std::endl;
            return 1;
        }

        // Clear gradient
        input.zero_grad();
        std::cout << "Gradient cleared" << std::endl;
    }

    std::cout << "\nReshape backward test completed successfully!" << std::endl;
    return 0;
}
