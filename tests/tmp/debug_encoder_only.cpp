#include <iostream>
#include <tenzor/tenzor.hpp>
#include <tenzor/nn/layers/transformer.hpp>

using namespace tenzor;
using namespace tenzor::nn;

int main() {
    tenzor::initialize();

    std::cout << "Testing TransformerEncoderLayer gradient flow\n\n";

    // Create single encoder layer
    TransformerEncoderLayer layer(128, 4, 512, 0.0, "relu", true);

    // Create input
    Variable src(randn({2, 5, 128}), true);
    std::cout << "src.requires_grad() = " << src.requires_grad() << std::endl;
    std::cout << "src.is_leaf() = " << src.is_leaf() << std::endl;

    // Forward
    std::cout << "\nCalling layer.forward(src)...\n";
    Variable output = layer.forward(src, Tensor{}, Tensor{});
    std::cout << "output.requires_grad() = " << output.requires_grad() << std::endl;
    std::cout << "output.grad_fn() = " << (output.grad_fn() ? "exists" : "null") << std::endl;

    // Loss
    Variable loss = mean(output);
    std::cout << "Loss value: " << loss.tensor().data<float>()[0] << std::endl;

    // Backward
    std::cout << "\nCalling loss.backward()...\n";
    loss.backward();
    std::cout << "Backward completed!\n";

    // Check gradients
    std::cout << "\nChecking gradients:\n";
    std::cout << "src.has_grad() = " << src.has_grad() << std::endl;

    if (!src.has_grad()) {
        std::cout << "\nFAIL! src does not have gradient\n";
        return 1;
    }

    std::cout << "\nSUCCESS! src has gradient\n";
    return 0;
}
