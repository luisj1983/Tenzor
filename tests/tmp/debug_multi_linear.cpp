#include <iostream>
#include <tenzor/tenzor.hpp>
#include <tenzor/nn/layers/linear.hpp>

using namespace tenzor;
using namespace tenzor::nn;

int main() {
    tenzor::initialize();

    std::cout << "Testing gradient accumulation with same input to multiple Linear layers\n\n";

    // Create three linear layers (like q_proj, k_proj, v_proj)
    Linear layer1(128, 128, true);
    Linear layer2(128, 128, true);
    Linear layer3(128, 128, true);

    // Create input variable (like query in attention)
    Variable input(randn({2, 5, 128}), true);
    std::cout << "Input created with requires_grad = " << input.requires_grad() << std::endl;
    std::cout << "Input is_leaf = " << input.is_leaf() << std::endl;

    // Project through all three layers (like Q, K, V projections)
    std::cout << "\nProjecting input through 3 Linear layers...\n";
    Variable out1 = layer1.forward(input);
    Variable out2 = layer2.forward(input);
    Variable out3 = layer3.forward(input);

    std::cout << "out1.grad_fn() = " << (out1.grad_fn() ? "exists" : "null") << std::endl;
    std::cout << "out2.grad_fn() = " << (out2.grad_fn() ? "exists" : "null") << std::endl;
    std::cout << "out3.grad_fn() = " << (out3.grad_fn() ? "exists" : "null") << std::endl;

    // Combine outputs (simplified attention-like operation)
    std::cout << "\nCombining outputs...\n";
    Variable combined = out1 + out2 + out3;
    Variable loss = mean(combined);

    std::cout << "Loss computed, value = " << loss.tensor().data<float>()[0] << std::endl;

    // Backward
    std::cout << "\nCalling backward...\n";
    loss.backward();
    std::cout << "Backward completed!\n";

    // Check gradients
    std::cout << "\nChecking gradients:\n";
    std::cout << "input.has_grad() = " << input.has_grad() << std::endl;

    if (!input.has_grad()) {
        std::cout << "\nFAIL! Input variable does not have gradient\n";
        std::cout << "This is the same issue as AttentionIntegrationTest.ForwardBackward\n";
        return 1;
    }

    std::cout << "\nSUCCESS! Input variable received accumulated gradients from 3 paths\n";
    return 0;
}
