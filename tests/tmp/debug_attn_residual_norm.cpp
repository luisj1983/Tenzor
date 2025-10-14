#include <iostream>
#include <tenzor/tenzor.hpp>
#include <tenzor/nn/layers/attention.hpp>
#include <tenzor/nn/layers/normalization.hpp>
#include <tenzor/nn/layers/dropout.hpp>

using namespace tenzor;
using namespace tenzor::nn;

int main() {
    tenzor::initialize();

    std::cout << "Testing attention + residual + norm pattern\n\n";

    // Create layers like encoder does
    auto self_attn = std::make_shared<MultiheadAttention>(128, 4, 0.0, true, false, false, 0, 0, true);
    auto dropout1 = std::make_shared<Dropout>(0.0);
    auto norm1 = std::make_shared<LayerNorm>(std::vector<int64_t>{128});

    // Create input
    Variable src(randn({2, 5, 128}), true);
    std::cout << "src address: " << &src << std::endl;
    std::cout << "src.requires_grad() = " << src.requires_grad() << std::endl;

    // Self-attention block (EXACTLY like encoder)
    std::cout << "\nStep 1: Self-attention\n";
    auto [attn_output, _] = self_attn->forward(src, src, src, Tensor{}, Tensor{}, false);
    std::cout << "attn_output.grad_fn() = " << (attn_output.grad_fn() ? "exists" : "null") << std::endl;

    // Dropout + residual (EXACTLY like encoder)
    std::cout << "\nStep 2: Dropout + Residual\n";
    Variable dropout_out = dropout1->forward(attn_output);
    Variable residual1 = src + dropout_out;
    std::cout << "residual1 address: " << &residual1 << std::endl;
    std::cout << "residual1.grad_fn() = " << (residual1.grad_fn() ? "exists" : "null") << std::endl;

    // Norm (EXACTLY like encoder)
    std::cout << "\nStep 3: LayerNorm\n";
    Variable x_norm1 = norm1->forward(residual1);
    std::cout << "x_norm1.grad_fn() = " << (x_norm1.grad_fn() ? "exists" : "null") << std::endl;

    if (x_norm1.grad_fn()) {
        const auto& input_vars = x_norm1.grad_fn()->input_variables();
        std::cout << "LayerNorm input_variables count: " << input_vars.size() << std::endl;
        if (!input_vars.empty() && input_vars[0]) {
            std::cout << "input_variables[0] address: " << input_vars[0] << std::endl;
            std::cout << "Does it match residual1? " << (input_vars[0] == &residual1 ? "YES" : "NO") << std::endl;
        }
    }

    // Loss
    Variable loss = mean(x_norm1);
    std::cout << "\nLoss value: " << loss.tensor().data<float>()[0] << std::endl;

    // Backward
    std::cout << "\nCalling loss.backward()...\n";
    loss.backward();
    std::cout << "Backward completed!\n";

    // Check gradients
    std::cout << "\nChecking gradients:\n";
    std::cout << "src.has_grad() = " << src.has_grad() << std::endl;
    std::cout << "residual1.has_grad() = " << residual1.has_grad() << std::endl;

    if (!src.has_grad()) {
        std::cout << "\nFAIL! src does not have gradient\n";
        return 1;
    }

    std::cout << "\nSUCCESS! Gradients flow through attention + residual + norm\n";
    return 0;
}
