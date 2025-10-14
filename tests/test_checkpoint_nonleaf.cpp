#include <iostream>
#include "tenzor/tenzor.hpp"
#include "tenzor/autograd/checkpoint.hpp"
#include "tenzor/autograd/ops.hpp"
#include "tenzor/ops/creation.hpp"

using namespace tenzor;
using namespace tenzor::autograd;

int main() {
    tenzor::initialize();

    std::cout << "Test 1: Leaf input (should fail with current implementation)" << std::endl;
    {
        auto x_tensor = ones({2, 2});
        Variable x(x_tensor, true);

        auto checkpointed_fn = [](const Variable& input) -> Variable {
            auto shape = input.shape();
            std::vector<int64_t> shape_vec(shape.begin(), shape.end());
            auto two = Variable(full(shape_vec, 2.0f), false);
            return input * two;
        };

        auto y = checkpoint(checkpointed_fn, x);
        auto loss = sum(y);
        loss.backward();

        std::cout << "  x.has_grad(): " << x.has_grad() << std::endl;
        if (x.has_grad()) {
            std::cout << "  ✓ PASSED" << std::endl;
        } else {
            std::cout << "  ✗ FAILED" << std::endl;
        }
    }

    std::cout << "\nTest 2: Non-leaf input (might work)" << std::endl;
    {
        auto x_tensor = ones({2, 2});
        Variable x(x_tensor, true);

        // Make x_intermediate a non-leaf by doing an operation
        auto shape_vec = std::vector<int64_t>({2, 2});
        auto one_var = Variable(ones(shape_vec), false);
        Variable x_intermediate = x + one_var;  // x_intermediate is non-leaf

        auto checkpointed_fn = [](const Variable& input) -> Variable {
            auto shape = input.shape();
            std::vector<int64_t> shape_vec(shape.begin(), shape.end());
            auto two = Variable(full(shape_vec, 2.0f), false);
            return input * two;
        };

        auto y = checkpoint(checkpointed_fn, x_intermediate);
        auto loss = sum(y);
        loss.backward();

        std::cout << "  x.has_grad(): " << x.has_grad() << std::endl;
        std::cout << "  x_intermediate.has_grad(): " << x_intermediate.has_grad() << std::endl;
        if (x.has_grad()) {
            std::cout << "  ✓ PASSED - gradient flowed through non-leaf!" << std::endl;
        } else {
            std::cout << "  ✗ FAILED" << std::endl;
        }
    }

    tenzor::finalize();
    return 0;
}
