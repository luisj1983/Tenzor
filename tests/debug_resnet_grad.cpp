/**
 * @file debug_resnet_grad.cpp
 * @brief Debug gradient flow in ResNet BasicBlock
 */

#include <iostream>
#include <tenzor/tenzor.hpp>
#include <tenzor/models/resnet.hpp>

using namespace tenzor;
using namespace tenzor::models;

int main() {
    tenzor::initialize();

    Device device = Device::cpu();

    // Create BasicBlock with no downsampling
    auto block = std::make_shared<BasicBlock>(64, 64, 1, 1, 64, nullptr);
    block->train();

    // Create input with gradient tracking
    Variable input(Tensor({2, 64, 56, 56}, DType::Float32, device), true);

    std::cout << "Input requires_grad: " << input.requires_grad() << std::endl;
    std::cout << "Input is_leaf: " << input.is_leaf() << std::endl;
    std::cout << "Input grad_fn: " << (input.grad_fn() ? "yes" : "no") << std::endl;

    // Forward pass
    Variable output = block->forward(input);

    std::cout << "\nAfter forward:" << std::endl;
    std::cout << "Output requires_grad: " << output.requires_grad() << std::endl;
    std::cout << "Output is_leaf: " << output.is_leaf() << std::endl;
    std::cout << "Output grad_fn: " << (output.grad_fn() ? "yes" : "no") << std::endl;

    // Compute loss
    auto loss_tensor = tenzor::sum((output * output).tensor());
    Variable loss(loss_tensor, true);

    std::cout << "\nLoss created:" << std::endl;
    std::cout << "Loss requires_grad: " << loss.requires_grad() << std::endl;
    std::cout << "Loss is_leaf: " << loss.is_leaf() << std::endl;
    std::cout << "Loss grad_fn: " << (loss.grad_fn() ? "yes" : "no") << std::endl;

    // Backward pass
    std::cout << "\nCalling backward..." << std::endl;
    loss.backward();

    std::cout << "\nAfter backward:" << std::endl;
    std::cout << "Input has_grad: " << input.grad().has_value() << std::endl;
    std::cout << "Loss has_grad: " << loss.grad().has_value() << std::endl;

    // Check block parameters
    auto params = block->parameters();
    std::cout << "Block has " << params.size() << " parameters" << std::endl;
    int params_with_grad = 0;
    for (const auto& param : params) {
        if (param->grad().has_value()) {
            params_with_grad++;
        }
    }
    std::cout << "Parameters with gradients: " << params_with_grad << std::endl;

    return 0;
}
