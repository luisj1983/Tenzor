#include "tenzor/tenzor.hpp"
#include "tenzor/nn/layers/segmentation.hpp"
#include <iostream>

using namespace tenzor;

int main() {
    std::cout << "Testing upsample gradient flow..." << std::endl;

    // Create simple input
    Variable input(Tensor({1, 2, 4, 4}, DType::Float32, Device::cpu()), true);
    std::cout << "Input created: requires_grad=" << input.requires_grad()
              << ", is_leaf=" << input.is_leaf() << std::endl;

    // Apply upsampling
    auto output = nn::upsample_bilinear(input, 8, 8);
    std::cout << "Output created: requires_grad=" << output.requires_grad()
              << ", is_leaf=" << output.is_leaf()
              << ", has_grad_fn=" << (output.grad_fn() != nullptr) << std::endl;

    // Create loss
    auto loss = tenzor::sum(output);
    std::cout << "Loss created: requires_grad=" << loss.requires_grad()
              << ", is_leaf=" << loss.is_leaf()
              << ", has_grad_fn=" << (loss.grad_fn() != nullptr) << std::endl;

    // Backward
    std::cout << "Calling backward()..." << std::endl;
    loss.backward();

    // Check gradients
    std::cout << "After backward:" << std::endl;
    std::cout << "  input.grad().has_value() = " << input.grad().has_value() << std::endl;
    std::cout << "  output.grad().has_value() = " << output.grad().has_value() << std::endl;

    if (input.grad().has_value()) {
        std::cout << "✅ PASS: Input gradients computed successfully!" << std::endl;
        return 0;
    } else {
        std::cout << "❌ FAIL: Input gradients NOT computed!" << std::endl;
        return 1;
    }
}
