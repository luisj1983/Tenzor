#include <iostream>
#include <tenzor/tenzor.hpp>
#include <tenzor/nn/layers/attention.hpp>

using namespace tenzor;
using namespace tenzor::nn;

int main() {
    tenzor::initialize();

    std::cout << "Testing encoder self-attention gradient flow\n\n";

    // Create self-attention like encoder does
    auto self_attn = std::make_shared<MultiheadAttention>(128, 4, 0.0, true, false, false, 0, 0, true);

    // Create input
    Variable src(randn({2, 5, 128}), true);
    std::cout << "src.requires_grad() = " << src.requires_grad() << std::endl;
    std::cout << "src address: " << &src << std::endl;

    // Self-attention block (like encoder does)
    std::cout << "\nCalling self_attn->forward(src, src, src)...\n";
    auto [attn_output, _] = self_attn->forward(src, src, src, Tensor{}, Tensor{}, false);

    std::cout << "attn_output.requires_grad() = " << attn_output.requires_grad() << std::endl;
    std::cout << "attn_output.grad_fn() = " << (attn_output.grad_fn() ? "exists" : "null") << std::endl;

    // Loss
    Variable loss = mean(attn_output);
    std::cout << "Loss value: " << loss.tensor().data<float>()[0] << std::endl;

    // Backward
    std::cout << "\nCalling loss.backward()...\n";
    loss.backward();
    std::cout << "Backward completed!\n";

    // Check gradients
    std::cout << "\nChecking gradients:\n";
    std::cout << "src.has_grad() = " << src.has_grad() << std::endl;

    if (!src.has_grad()) {
        std::cout << "\nFAIL! src does not have gradient through self-attention\n";
        return 1;
    }

    std::cout << "\nSUCCESS! src has gradient through self-attention\n";
    return 0;
}
