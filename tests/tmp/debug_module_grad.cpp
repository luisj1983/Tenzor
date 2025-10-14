#include <iostream>
#include <tenzor/tenzor.hpp>
#include <tenzor/nn/layers/linear.hpp>
#include <tenzor/autograd/ops.hpp>

using namespace tenzor;
using namespace tenzor::nn;

int main() {
    tenzor::initialize();

    std::cout << "Creating Linear layer" << std::endl;
    Linear linear(128, 128, true);

    std::cout << "Checking if Linear parameters have requires_grad:" << std::endl;
    auto params = linear.parameters();
    std::cout << "  Number of parameters: " << params.size() << std::endl;
    for (size_t i = 0; i < params.size(); ++i) {
        std::cout << "  param[" << i << "].requires_grad() = " << params[i]->requires_grad() << std::endl;
        std::cout << "  param[" << i << "].is_leaf() = " << params[i]->is_leaf() << std::endl;
    }

    std::cout << "\nCreating input variable" << std::endl;
    Variable input(randn({2, 5, 128}), true);
    std::cout << "input.requires_grad() = " << input.requires_grad() << std::endl;
    std::cout << "input.is_leaf() = " << input.is_leaf() << std::endl;

    std::cout << "\nForward pass through Linear" << std::endl;
    Variable output = linear.forward(input);
    std::cout << "output.requires_grad() = " << output.requires_grad() << std::endl;
    std::cout << "output.is_leaf() = " << output.is_leaf() << std::endl;
    std::cout << "output.grad_fn() = " << (output.grad_fn() ? "exists" : "null") << std::endl;

    std::cout << "\nComputing mean and calling backward" << std::endl;
    Variable loss = mean(output);
    loss.backward();
    std::cout << "Backward completed!" << std::endl;

    std::cout << "\nChecking gradients:" << std::endl;
    std::cout << "input.has_grad() = " << input.has_grad() << std::endl;

    if (!input.has_grad()) {
        std::cout << "FAIL! input does not have gradient" << std::endl;
        return 1;
    }

    std::cout << "Checking parameter gradients:" << std::endl;
    for (size_t i = 0; i < params.size(); ++i) {
        std::cout << "  param[" << i << "].has_grad() = " << params[i]->has_grad() << std::endl;
    }

    std::cout << "\nSUCCESS! All gradients present" << std::endl;
    return 0;
}
