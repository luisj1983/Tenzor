#include <tenzor/tenzor.hpp>
#include <tenzor/nn/loss/losses.hpp>
#include <iostream>

using namespace tenzor;
using namespace tenzor::nn;

int main() {
    tenzor::initialize();

    std::cout << "Test 1: MSELoss backward" << std::endl;
    try {
        auto input = Variable(full({2, 3}, 2.0f, DType::Float32), true);
        auto target = Variable(full({2, 3}, 1.0f, DType::Float32), false);

        auto loss = mse_loss(input, target);
        std::cout << "Forward passed, loss = " << loss.tensor().item<float>() << std::endl;

        loss.backward();
        std::cout << "Backward passed" << std::endl;

        if (input.grad().has_value()) {
            std::cout << "✓ Gradient computed successfully" << std::endl;
        } else {
            std::cout << "✗ ERROR: No gradient!" << std::endl;
        }
    } catch (const std::exception& e) {
        std::cout << "✗ ERROR: " << e.what() << std::endl;
    }

    std::cout << "\nTest 2: CrossEntropyLoss backward" << std::endl;
    try {
        auto input = Variable(full({2, 3}, 1.0f, DType::Float32), true);
        auto target_tensor = full({2, 3}, 0.333f, DType::Float32);

        auto loss = cross_entropy(input, target_tensor);
        std::cout << "Forward passed, loss = " << loss.tensor().item<float>() << std::endl;

        loss.backward();
        std::cout << "Backward passed" << std::endl;

        if (input.grad().has_value()) {
            std::cout << "✓ Gradient computed successfully" << std::endl;
        } else {
            std::cout << "✗ ERROR: No gradient!" << std::endl;
        }
    } catch (const std::exception& e) {
        std::cout << "✗ ERROR: " << e.what() << std::endl;
    }

    std::cout << "\nTest 3: BCELoss backward" << std::endl;
    try {
        auto input = Variable(full({2, 3}, 0.7f, DType::Float32), true);
        auto target = Variable(full({2, 3}, 1.0f, DType::Float32), false);

        auto loss = bce_loss(input, target);
        std::cout << "Forward passed, loss = " << loss.tensor().item<float>() << std::endl;

        loss.backward();
        std::cout << "Backward passed" << std::endl;

        if (input.grad().has_value()) {
            std::cout << "✓ Gradient computed successfully" << std::endl;
        } else {
            std::cout << "✗ ERROR: No gradient!" << std::endl;
        }
    } catch (const std::exception& e) {
        std::cout << "✗ ERROR: " << e.what() << std::endl;
    }

    return 0;
}
