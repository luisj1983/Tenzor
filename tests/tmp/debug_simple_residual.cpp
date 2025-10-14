#include <iostream>
#include <tenzor/tenzor.hpp>
#include <tenzor/nn/layers/normalization.hpp>
#include <tenzor/nn/layers/dropout.hpp>

using namespace tenzor;
using namespace tenzor::nn;

int main() {
    tenzor::initialize();

    std::cout << "Testing residual connection gradient flow\n\n";

    // Create layers
    LayerNorm norm(std::vector<int64_t>{128});
    Dropout dropout(0.0);

    // Create input
    Variable src(randn({2, 5, 128}), true);
    std::cout << "src.requires_grad() = " << src.requires_grad() << std::endl;

    // Simulate what encoder does: src + dropout(some_output)
    Tensor some_output_tensor = randn({2, 5, 128}) * 0.1f;
    Variable some_output(some_output_tensor, false);
    Variable dropout_out = dropout.forward(some_output);

    std::cout << "\nComputing: x = src + dropout_out\n";
    Variable x = src + dropout_out;
    std::cout << "x.grad_fn() = " << (x.grad_fn() ? "exists" : "null") << std::endl;

    std::cout << "\nApplying layer norm\n";
    x = norm.forward(x);
    std::cout << "x.grad_fn() after norm = " << (x.grad_fn() ? "exists" : "null") << std::endl;

    // Loss
    Variable loss = mean(x);
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

    std::cout << "\nSUCCESS! src has gradient through residual + norm\n";
    return 0;
}
