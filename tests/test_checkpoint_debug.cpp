/**
 * @file test_checkpoint_debug.cpp
 * @brief Debug checkpoint gradient computation
 */

#include <iostream>
#include "tenzor/tenzor.hpp"
#include "tenzor/autograd/checkpoint.hpp"
#include "tenzor/autograd/variable.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"

using namespace tenzor;
using namespace tenzor::autograd;

int main() {
    tenzor::initialize();

    std::cout << "=== Debug: Checkpoint gradient computation ===" << std::endl;

    // Create leaf input
    auto x_tensor = ones({2, 2});
    Variable x(x_tensor, true);
    std::cout << "Created x (leaf): " << x.is_leaf() << ", requires_grad: " << x.requires_grad() << std::endl;

    // Checkpoint function: y = x^2
    auto checkpointed_fn = [](const Variable& input) -> Variable {
        std::cout << "  Inside checkpoint fn: computing x * x" << std::endl;
        auto result = input * input;
        std::cout << "  Result requires_grad: " << result.requires_grad() << std::endl;
        std::cout << "  Result has grad_fn: " << (result.grad_fn() != nullptr) << std::endl;
        return result;
    };

    std::cout << "\nForward pass:" << std::endl;
    auto y = checkpoint_with_original(checkpointed_fn, x, &x);
    std::cout << "y shape: " << y.tensor().shape()[0] << "x" << y.tensor().shape()[1] << std::endl;
    std::cout << "y requires_grad: " << y.requires_grad() << std::endl;
    std::cout << "y has grad_fn: " << (y.grad_fn() != nullptr) << std::endl;

    // Compute loss
    std::cout << "\nComputing loss = sum(y):" << std::endl;
    auto loss = sum(y);
    std::cout << "loss value: " << loss.tensor().data<float>()[0] << std::endl;

    // Backward pass
    std::cout << "\nBackward pass:" << std::endl;
    loss.backward();

    // Check gradient
    std::cout << "\nGradient check:" << std::endl;
    if (x.has_grad()) {
        const float* grad_data = x.grad()->data<float>();
        std::cout << "x.grad() values: [";
        for (int i = 0; i < 4; ++i) {
            std::cout << grad_data[i];
            if (i < 3) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        std::cout << "Expected: [2, 2, 2, 2] (dy/dx = 2*x = 2*1 = 2)" << std::endl;
    } else {
        std::cout << "ERROR: x.grad() is empty!" << std::endl;
    }

    tenzor::finalize();
    return 0;
}
