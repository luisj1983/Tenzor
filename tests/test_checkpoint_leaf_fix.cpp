/**
 * @file test_checkpoint_leaf_fix.cpp
 * @brief Minimal test to verify checkpoint() works with leaf variables
 */

#include <iostream>
#include "tenzor/tenzor.hpp"
#include "tenzor/autograd/checkpoint.hpp"
#include "tenzor/autograd/variable.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"

using namespace tenzor;
using namespace tenzor::autograd;

int main() {
    tenzor::initialize();

    std::cout << "=== Test 1: Leaf input (CRITICAL FIX TEST) ===" << std::endl;
    {
        // Create leaf input
        auto x_tensor = ones({2, 2});
        Variable x(x_tensor, true);
        std::cout << "Created leaf Variable x" << std::endl;

        // Checkpoint function: y = x * 2
        auto fn = [](const Variable& in) -> Variable {
            auto shape = in.shape();
            std::vector<int64_t> shape_vec(shape.begin(), shape.end());
            auto two = Variable(full(shape_vec, 2.0f), false);
            return in * two;
        };

        // CRITICAL: Use checkpoint_with_original to pass pointer to original x
        auto y = checkpoint_with_original(fn, x, &x);
        std::cout << "Checkpointed: y = x * 2" << std::endl;

        // Backward pass
        auto loss = sum(y);
        loss.backward();
        std::cout << "Backward complete" << std::endl;

        // Check gradient
        if (x.has_grad()) {
            const float* grad_data = x.grad()->data<float>();
            bool correct = true;
            for (int i = 0; i < 4; ++i) {
                if (std::abs(grad_data[i] - 2.0f) > 1e-5f) {
                    correct = false;
                    std::cout << "FAILED: grad[" << i << "] = " << grad_data[i] << ", expected 2.0" << std::endl;
                }
            }
            if (correct) {
                std::cout << "PASSED: Leaf input - gradient accumulated to original!" << std::endl;
            } else {
                std::cout << "FAILED: Leaf input - gradient incorrect" << std::endl;
            }
        } else {
            std::cout << "FAILED: Leaf input - no gradient accumulated" << std::endl;
        }
    }

    std::cout << "\n=== Test 2: Non-leaf input (should still work) ===" << std::endl;
    {
        // Create non-leaf input
        auto x_tensor = ones({2, 2});
        Variable x(x_tensor, true);
        auto three = Variable(full({2, 2}, 3.0f), false);
        auto x_non_leaf = x * three;  // x_non_leaf has grad_fn, not a leaf
        std::cout << "Created non-leaf Variable x_non_leaf = x * 3" << std::endl;

        // Checkpoint function: y = x_non_leaf * 2
        auto fn = [](const Variable& in) -> Variable {
            auto shape = in.shape();
            std::vector<int64_t> shape_vec(shape.begin(), shape.end());
            auto two = Variable(full(shape_vec, 2.0f), false);
            return in * two;
        };

        auto y = checkpoint(fn, x_non_leaf);
        std::cout << "Checkpointed: y = x_non_leaf * 2" << std::endl;

        // Backward pass
        auto loss = sum(y);
        loss.backward();
        std::cout << "Backward complete" << std::endl;

        // Check gradient
        if (x.has_grad()) {
            const float* grad_data = x.grad()->data<float>();
            bool correct = true;
            for (int i = 0; i < 4; ++i) {
                if (std::abs(grad_data[i] - 6.0f) > 1e-5f) {  // dy/dx = 3 * 2 = 6
                    correct = false;
                    std::cout << "FAILED: grad[" << i << "] = " << grad_data[i] << ", expected 6.0" << std::endl;
                }
            }
            if (correct) {
                std::cout << "PASSED: Non-leaf input - gradient flowed correctly!" << std::endl;
            } else {
                std::cout << "FAILED: Non-leaf input - gradient incorrect" << std::endl;
            }
        } else {
            std::cout << "FAILED: Non-leaf input - no gradient accumulated" << std::endl;
        }
    }

    tenzor::finalize();
    return 0;
}
