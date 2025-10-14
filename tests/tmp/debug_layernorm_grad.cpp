#include <iostream>
#include <tenzor/tenzor.hpp>
#include <tenzor/nn/layers/normalization.hpp>

using namespace tenzor;
using namespace tenzor::nn;

int main() {
    tenzor::initialize();

    std::cout << "Testing LayerNorm gradient flow\n\n";

    // Create layer norm
    LayerNorm norm(std::vector<int64_t>{128});

    // Create input
    Variable input(randn({2, 5, 128}), true);
    std::cout << "input.requires_grad() = " << input.requires_grad() << std::endl;
    std::cout << "input.is_leaf() = " << input.is_leaf() << std::endl;
    std::cout << "input address: " << &input << std::endl;

    // Forward
    std::cout << "\nCalling norm.forward(input)...\n";
    Variable output = norm.forward(input);
    std::cout << "output.requires_grad() = " << output.requires_grad() << std::endl;
    std::cout << "output.grad_fn() = " << (output.grad_fn() ? "exists" : "null") << std::endl;

    if (output.grad_fn()) {
        const auto& input_vars = output.grad_fn()->input_variables();
        std::cout << "Number of input_variables: " << input_vars.size() << std::endl;
        if (!input_vars.empty() && input_vars[0]) {
            std::cout << "input_variables[0] address: " << input_vars[0] << std::endl;
            std::cout << "Does it match input? " << (input_vars[0] == &input ? "YES" : "NO") << std::endl;
        }
    }

    // Loss
    Variable loss = mean(output);
    std::cout << "\nLoss value: " << loss.tensor().data<float>()[0] << std::endl;

    // Backward
    std::cout << "\nCalling loss.backward()...\n";
    loss.backward();
    std::cout << "Backward completed!\n";

    // Check gradients
    std::cout << "\nChecking gradients:\n";
    std::cout << "input.has_grad() = " << input.has_grad() << std::endl;

    if (!input.has_grad()) {
        std::cout << "\nFAIL! input does not have gradient\n";
        return 1;
    }

    std::cout << "\nSUCCESS! input has gradient through LayerNorm\n";
    return 0;
}
