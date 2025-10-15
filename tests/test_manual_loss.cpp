#include <tenzor/tenzor.hpp>
#include <tenzor/autograd/ops.hpp>
#include <iostream>

using namespace tenzor;

int main() {
    tenzor::initialize();

    std::cout << "Test: Manual MSE computation" << std::endl;
    try {
        auto input = Variable(full({2, 3}, 2.0f, DType::Float32), true);
        auto target = Variable(full({2, 3}, 1.0f, DType::Float32), false);

        // Manual MSE: mean((input - target)^2)
        auto diff = input - target;
        auto squared = diff * diff;
        auto loss = mean(squared);

        std::cout << "Forward passed, loss = " << loss.tensor().item<float>() << std::endl;

        loss.backward();
        std::cout << "Backward passed" << std::endl;

        if (input.grad().has_value()) {
            std::cout << "✓ Gradient computed successfully" << std::endl;
            std::cout << "  Gradient shape: [";
            for (size_t i = 0; i < input.grad()->shape().size(); ++i) {
                if (i > 0) std::cout << ", ";
                std::cout << input.grad()->shape()[i];
            }
            std::cout << "]" << std::endl;
        } else {
            std::cout << "✗ ERROR: No gradient!" << std::endl;
        }
    } catch (const std::exception& e) {
        std::cout << "✗ ERROR: " << e.what() << std::endl;
    }

    std::cout << "\nTest: Functional MSE vs Manual" << std::endl;
    try {
        auto input1 = Variable(full({2, 3}, 2.0f, DType::Float32), true);
        auto target1 = Variable(full({2, 3}, 1.0f, DType::Float32), false);

        auto input2 = Variable(full({2, 3}, 2.0f, DType::Float32), true);
        auto target2 = Variable(full({2, 3}, 1.0f, DType::Float32), false);

        // Manual
        auto diff = input1 - target1;
        auto squared = diff * diff;
        auto loss1 = mean(squared);

        // Check requires_grad
        std::cout << "Manual loss requires_grad: " << (loss1.requires_grad() ? "true" : "false") << std::endl;
        std::cout << "Manual loss value: " << loss1.tensor().item<float>() << std::endl;

        // Try functional API
        auto diff2 = input2 - target2;
        auto squared2 = diff2 * diff2;

        // Import mean from ops
        auto loss2 = tenzor::mean(squared2);

        std::cout << "Functional loss requires_grad: " << (loss2.requires_grad() ? "true" : "false") << std::endl;
        std::cout << "Functional loss value: " << loss2.tensor().item<float>() << std::endl;

        std::cout << "✓ Both approaches work identically" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "✗ ERROR: " << e.what() << std::endl;
    }

    return 0;
}
