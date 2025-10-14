#include <iostream>
#include <tenzor/tenzor.hpp>
#include <tenzor/nn/layers/linear.hpp>
#include <tenzor/autograd/ops.hpp>

using namespace tenzor;
using namespace tenzor::nn;

int main() {
    tenzor::initialize();

    std::cout << "Creating Linear layer" << std::endl;
    Linear linear(128, 128, true);

    auto params = linear.parameters();
    std::cout << "Weight parameter address: " << params[0] << std::endl;
    std::cout << "Weight requires_grad: " << params[0]->requires_grad() << std::endl;
    std::cout << "Weight is_leaf: " << params[0]->is_leaf() << std::endl;

    // Now simulate what Linear::forward does
    std::cout << "\nSimulating Linear::forward weight usage:" << std::endl;
    auto& weight_ref = *params[0];  // Reference like in Linear::forward
    std::cout << "weight_ref address: " << &weight_ref << std::endl;

    // Call permute like Linear does
    std::cout << "\nCalling permute on weight_ref" << std::endl;
    auto weight_t = permute(weight_ref, {1, 0});

    std::cout << "weight_t.grad_fn() = " << (weight_t.grad_fn() ? "exists" : "null") << std::endl;

    if (weight_t.grad_fn()) {
        const auto& input_vars = weight_t.grad_fn()->input_variables();
        std::cout << "Number of input_variables: " << input_vars.size() << std::endl;
        if (!input_vars.empty() && input_vars[0]) {
            std::cout << "input_variables[0] address: " << input_vars[0] << std::endl;
            std::cout << "Do they match? " << (input_vars[0] == &weight_ref ? "YES" : "NO") << std::endl;
            std::cout << "Do they match original param? " << (input_vars[0] == params[0] ? "YES" : "NO") << std::endl;
        }
    }

    return 0;
}
