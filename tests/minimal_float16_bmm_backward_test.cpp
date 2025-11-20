#include <iostream>
#include "tenzor/tenzor.hpp"
#include "tenzor/autograd/variable.hpp"
#include "tenzor/autograd/ops.hpp"
#include "tenzor/ops/creation.hpp"

using namespace tenzor;

int main() {
    tenzor::initialize();

    std::cout << "Testing bmm backward pass with Float16..." << std::endl;

    // Create simple bmm inputs matching attention use case
    // (batch*num_heads, seq_len, head_dim) @ (batch*num_heads, head_dim, seq_len)
    // = (batch*num_heads, seq_len, seq_len)

    // Simplified: batch_size=2, seq_len=4, head_dim=8
    auto a_tensor = randn({2, 4, 8}, DType::Float16);
    auto b_tensor = randn({2, 8, 4}, DType::Float16);

    Variable a(a_tensor, true);  // requires_grad=true
    Variable b(b_tensor, true);

    std::cout << "Created input tensors: a=[2,4,8], b=[2,8,4]" << std::endl;

    // Test 3 iterations
    for (int i = 0; i < 3; ++i) {
        std::cout << "\n=== Iteration " << i << " ===" << std::endl;

        // Forward pass
        std::cout << "Running bmm forward..." << std::endl;
        auto result = bmm(a, b);  // autograd bmm
        std::cout << "Forward completed, result.shape = ["
                  << result.shape()[0] << ", "
                  << result.shape()[1] << ", "
                  << result.shape()[2] << "]" << std::endl;

        // Sum to scalar
        std::cout << "Computing loss (sum)..." << std::endl;
        Variable loss = tenzor::sum(result);
        std::cout << "Loss computed" << std::endl;

        // Backward pass - THIS IS WHERE THE HANG MIGHT OCCUR
        std::cout << "Running backward pass..." << std::endl;
        loss.backward();
        std::cout << "Backward pass completed!" << std::endl;

        // Check gradients
        if (a.grad().has_value()) {
            std::cout << "Gradient a exists, dtype="
                      << (a.grad()->dtype() == DType::Float16 ? "Float16" : "other")
                      << ", shape=[" << a.grad()->shape()[0] << ", "
                      << a.grad()->shape()[1] << ", " << a.grad()->shape()[2] << "]"
                      << std::endl;
        } else {
            std::cout << "ERROR: Gradient a is missing!" << std::endl;
            return 1;
        }

        if (b.grad().has_value()) {
            std::cout << "Gradient b exists, dtype="
                      << (b.grad()->dtype() == DType::Float16 ? "Float16" : "other")
                      << ", shape=[" << b.grad()->shape()[0] << ", "
                      << b.grad()->shape()[1] << ", " << b.grad()->shape()[2] << "]"
                      << std::endl;
        } else {
            std::cout << "ERROR: Gradient b is missing!" << std::endl;
            return 1;
        }

        // Clear gradients
        a.zero_grad();
        b.zero_grad();
        std::cout << "Gradients cleared" << std::endl;
    }

    std::cout << "\nBMM backward test completed successfully!" << std::endl;
    return 0;
}
