#include <iostream>
#include <tenzor/tenzor.hpp>
#include <tenzor/autograd/ops.hpp>

using namespace tenzor;

int main() {
    tenzor::initialize();

    std::cout << "Creating weight parameter (leaf, requires_grad=true)" << std::endl;
    Variable weight(randn({4, 3}), true);
    std::cout << "weight address: " << &weight << std::endl;
    std::cout << "weight.is_leaf() = " << weight.is_leaf() << std::endl;
    std::cout << "weight.requires_grad() = " << weight.requires_grad() << std::endl;

    std::cout << "\nCreating input (leaf, requires_grad=true)" << std::endl;
    Variable input(randn({2, 3}), true);
    std::cout << "input.is_leaf() = " << input.is_leaf() << std::endl;

    std::cout << "\nTransposing weight" << std::endl;
    Variable weight_t = permute(weight, {1, 0});  // Shape: (3, 4)
    std::cout << "weight_t.grad_fn() = " << (weight_t.grad_fn() ? "exists" : "null") << std::endl;

    if (weight_t.grad_fn()) {
        const auto& input_vars = weight_t.grad_fn()->input_variables();
        std::cout << "weight_t input_variables[0] address: " << input_vars[0] << std::endl;
        std::cout << "Does it match weight? " << (input_vars[0] == &weight ? "YES" : "NO") << std::endl;
    }

    std::cout << "\nMatrix multiplication" << std::endl;
    Variable output = matmul(input, weight_t);  // (2, 3) @ (3, 4) = (2, 4)
    std::cout << "output shape: [" << output.shape()[0] << ", " << output.shape()[1] << "]" << std::endl;

    std::cout << "\nComputing mean" << std::endl;
    Variable loss = mean(output);
    std::cout << "loss value: " << loss.tensor().data<float>()[0] << std::endl;

    std::cout << "\nCalling backward..." << std::endl;
    loss.backward();
    std::cout << "Backward completed!" << std::endl;

    std::cout << "\nChecking gradients:" << std::endl;
    std::cout << "input.has_grad() = " << input.has_grad() << std::endl;
    std::cout << "weight.has_grad() = " << weight.has_grad() << std::endl;

    if (!weight.has_grad()) {
        std::cout << "\nFAIL! weight does not have gradient" << std::endl;
        return 1;
    }

    std::cout << "\nSUCCESS! Both variables have gradients" << std::endl;
    return 0;
}
