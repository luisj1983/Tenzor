#include <iostream>
#include <tenzor/tenzor.hpp>
#include <tenzor/nn/layers/normalization.hpp>

using namespace tenzor;
using namespace tenzor::nn;

int main() {
    tenzor::initialize();

    std::cout << "Testing addition + LayerNorm gradient flow\n\n";

    // Create layer norm
    LayerNorm norm(std::vector<int64_t>{128});

    // Create two inputs
    Variable a(randn({2, 5, 128}), true);
    Variable b(randn({2, 5, 128}), true);

    std::cout << "a.requires_grad() = " << a.requires_grad() << std::endl;
    std::cout << "b.requires_grad() = " << b.requires_grad() << std::endl;

    // Addition
    std::cout << "\nStep 1: a + b\n";
    Variable sum = a + b;
    std::cout << "sum.grad_fn() = " << (sum.grad_fn() ? "exists" : "null") << std::endl;

    // LayerNorm
    std::cout << "\nStep 2: LayerNorm(sum)\n";
    Variable output = norm.forward(sum);
    std::cout << "output.grad_fn() = " << (output.grad_fn() ? "exists" : "null") << std::endl;

    // Loss
    Variable loss = mean(output);
    std::cout << "\nLoss value: " << loss.tensor().data<float>()[0] << std::endl;

    // Backward
    std::cout << "\nCalling loss.backward()...\n";
    loss.backward();
    std::cout << "Backward completed!\n";

    // Check gradients
    std::cout << "\nChecking gradients:\n";
    std::cout << "a.has_grad() = " << a.has_grad() << std::endl;
    std::cout << "b.has_grad() = " << b.has_grad() << std::endl;
    std::cout << "sum.has_grad() = " << sum.has_grad() << std::endl;

    if (!a.has_grad() || !b.has_grad()) {
        std::cout << "\nFAIL! Gradients not flowing through addition + norm\n";
        return 1;
    }

    std::cout << "\nSUCCESS! Gradients flow through addition + norm\n";
    return 0;
}
