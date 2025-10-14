#include <iostream>
#include <tenzor/tenzor.hpp>
#include <tenzor/autograd/ops.hpp>

using namespace tenzor;

int main() {
    tenzor::initialize();

    std::cout << "Testing which Variable the grad_fn references\n\n";

    // Create original leaf variable
    Variable original(randn({2, 3}), true);
    std::cout << "original address: " << &original << std::endl;
    std::cout << "original.tensor() address: " << &original.tensor() << std::endl;

    // Make a copy (like attention does)
    Variable copy = original;
    std::cout << "\ncopy address: " << &copy << std::endl;
    std::cout << "copy.tensor() address: " << &copy.tensor() << std::endl;
    std::cout << "Same tensor? " << (&copy.tensor() == &original.tensor()) << std::endl;

    // Perform an operation on the copy
    Variable result = copy + copy;
    std::cout << "\nresult.grad_fn() = " << (result.grad_fn() ? "exists" : "null") << std::endl;

    if (result.grad_fn()) {
        const auto& inputs = result.grad_fn()->input_variables();
        std::cout << "Number of input_variables: " << inputs.size() << std::endl;
        if (!inputs.empty() && inputs[0]) {
            std::cout << "input_variables[0] address: " << inputs[0] << std::endl;
            std::cout << "Does it point to original? " << (inputs[0] == &original ? "YES" : "NO") << std::endl;
            std::cout << "Does it point to copy? " << (inputs[0] == &copy ? "YES" : "NO") << std::endl;
        }
    }

    // Backward
    Variable loss = mean(result);
    loss.backward();

    std::cout << "\nAfter backward:\n";
    std::cout << "original.has_grad() = " << original.has_grad() << std::endl;
    std::cout << "copy.has_grad() = " << copy.has_grad() << std::endl;

    return 0;
}
