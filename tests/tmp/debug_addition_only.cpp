#include <iostream>
#include <tenzor/tenzor.hpp>

using namespace tenzor;

int main() {
    tenzor::initialize();

    std::cout << "Testing simple addition gradient flow\n\n";

    // Create two leaf variables
    Variable a(randn({2, 3}), true);
    Variable b(randn({2, 3}), true);

    std::cout << "a.requires_grad() = " << a.requires_grad() << std::endl;
    std::cout << "b.requires_grad() = " << b.requires_grad() << std::endl;
    std::cout << "a.is_leaf() = " << a.is_leaf() << std::endl;

    // Addition
    std::cout << "\nComputing: c = a + b\n";
    Variable c = a + b;
    std::cout << "c.grad_fn() = " << (c.grad_fn() ? "exists" : "null") << std::endl;

    // Loss
    Variable loss = mean(c);
    std::cout << "Loss value: " << loss.tensor().data<float>()[0] << std::endl;

    // Backward
    std::cout << "\nCalling loss.backward()...\n";
    loss.backward();
    std::cout << "Backward completed!\n";

    // Check gradients
    std::cout << "\nChecking gradients:\n";
    std::cout << "a.has_grad() = " << a.has_grad() << std::endl;
    std::cout << "b.has_grad() = " << b.has_grad() << std::endl;

    if (!a.has_grad() || !b.has_grad()) {
        std::cout << "\nFAIL! Gradients not flowing through addition\n";
        return 1;
    }

    std::cout << "\nSUCCESS! Gradients flow through addition\n";
    return 0;
}
