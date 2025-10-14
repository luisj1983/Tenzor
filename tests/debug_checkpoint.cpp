/**
 * Debug checkpoint backward propagation
 */

#include <iostream>
#include "tenzor/tenzor.hpp"
#include "tenzor/autograd/checkpoint.hpp"
#include "tenzor/autograd/ops.hpp"
#include "tenzor/ops/creation.hpp"

using namespace tenzor;
using namespace tenzor::autograd;

int main() {
    tenzor::initialize();

    std::cout << "Creating input variable..." << std::endl;
    auto x_tensor = ones({2, 2});
    Variable x(x_tensor, true);

    std::cout << "x is_leaf: " << x.is_leaf() << std::endl;
    std::cout << "x requires_grad: " << x.requires_grad() << std::endl;
    std::cout << "x has grad_fn: " << (x.grad_fn() != nullptr) << std::endl;

    std::cout << "\nDefining checkpoint function..." << std::endl;
    auto checkpointed_fn = [](const Variable& input) -> Variable {
        std::cout << "  Inside checkpoint function" << std::endl;
        auto shape = input.shape();
        std::vector<int64_t> shape_vec(shape.begin(), shape.end());
        auto two = Variable(full(shape_vec, 2.0f), false);
        std::cout << "  Computing input * two" << std::endl;
        auto result = input * two;
        std::cout << "  result requires_grad: " << result.requires_grad() << std::endl;
        std::cout << "  result has grad_fn: " << (result.grad_fn() != nullptr) << std::endl;
        return result;
    };

    std::cout << "\nCalling checkpoint..." << std::endl;
    auto y = checkpoint(checkpointed_fn, x);

    std::cout << "\ny is_leaf: " << y.is_leaf() << std::endl;
    std::cout << "y requires_grad: " << y.requires_grad() << std::endl;
    std::cout << "y has grad_fn: " << (y.grad_fn() != nullptr) << std::endl;

    std::cout << "\nComputing loss..." << std::endl;
    auto loss = sum(y);

    std::cout << "loss is_leaf: " << loss.is_leaf() << std::endl;
    std::cout << "loss requires_grad: " << loss.requires_grad() << std::endl;
    std::cout << "loss has grad_fn: " << (loss.grad_fn() != nullptr) << std::endl;

    std::cout << "\nCalling backward..." << std::endl;
    loss.backward();

    std::cout << "\nAfter backward:" << std::endl;
    std::cout << "x.has_grad(): " << x.has_grad() << std::endl;

    if (x.has_grad()) {
        std::cout << "SUCCESS: Gradient was computed!" << std::endl;
        const float* grad_data = x.grad()->data<float>();
        std::cout << "Gradient values: ";
        for (int i = 0; i < 4; ++i) {
            std::cout << grad_data[i] << " ";
        }
        std::cout << std::endl;
    } else {
        std::cout << "FAILURE: No gradient computed" << std::endl;
    }

    tenzor::finalize();
    return 0;
}
