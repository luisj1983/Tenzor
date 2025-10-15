/**
 * Debug test to understand NestedCheckpoints gradient issue
 */

#include "tenzor/tenzor.hpp"
#include "tenzor/autograd/checkpoint.hpp"
#include "tenzor/autograd/variable.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include <iostream>
#include <iomanip>

using namespace tenzor;
using namespace tenzor::autograd;

void run_nested_checkpoint_test(const char* scenario) {
    std::cout << "\n========================================\n";
    std::cout << "Scenario: " << scenario << "\n";
    std::cout << "========================================\n";

    // Reset stats
    reset_checkpoint_stats();

    // Create input
    auto x_tensor = ones({2, 2});
    Variable x(x_tensor, true);

    std::cout << "Initial x: " << x.tensor().data<float>()[0] << "\n";

    // Outer checkpoint function
    auto outer_fn = [&x](const Variable& input) -> Variable {
        std::cout << "  [Outer] Forward pass, input value: "
                  << input.tensor().data<float>()[0] << "\n";

        // Inner checkpoint
        auto inner_fn = [](const Variable& in) -> Variable {
            auto shape = in.shape();
            std::vector<int64_t> shape_vec(shape.begin(), shape.end());
            auto three = Variable(full(shape_vec, 3.0f), false);
            std::cout << "    [Inner] Forward pass, multiplying by 3\n";
            return in * three;
        };

        auto intermediate = checkpoint(inner_fn, input);
        std::cout << "  [Outer] After inner checkpoint: "
                  << intermediate.tensor().data<float>()[0] << "\n";

        auto shape = input.shape();
        std::vector<int64_t> shape_vec(shape.begin(), shape.end());
        auto one = Variable(full(shape_vec, 1.0f), false);
        auto result = intermediate + one;
        std::cout << "  [Outer] After adding 1: "
                  << result.tensor().data<float>()[0] << "\n";
        return result;
    };

    // Use checkpoint_with_original
    std::cout << "Creating outer checkpoint...\n";
    auto y = checkpoint_with_original(outer_fn, x, &x);

    std::cout << "Forward complete. y value: " << y.tensor().data<float>()[0] << "\n";
    std::cout << "Expected: 4.0 (1*3 + 1)\n";

    // Backward
    std::cout << "\nStarting backward pass...\n";
    auto loss = sum(y);
    loss.backward();

    // Check gradient
    if (x.grad().has_value()) {
        float grad_value = x.grad()->data<float>()[0];
        std::cout << "\nGradient computed: " << grad_value << "\n";
        std::cout << "Expected gradient: 3.0\n";

        if (std::abs(grad_value - 3.0f) < 1e-5) {
            std::cout << "✅ TEST PASSED\n";
        } else {
            std::cout << "❌ TEST FAILED - gradient mismatch\n";
        }
    } else {
        std::cout << "❌ TEST FAILED - no gradient computed\n";
    }

    // Stats
    auto& stats = get_checkpoint_stats();
    std::cout << "\nCheckpoint Stats:\n";
    std::cout << "  Checkpoints: " << stats.num_checkpoints << "\n";
    std::cout << "  Recomputations: " << stats.num_recomputations << "\n";
}

void run_simple_checkpoint_first() {
    std::cout << "\n========================================\n";
    std::cout << "Running simple checkpoint (x*x) first\n";
    std::cout << "========================================\n";

    reset_checkpoint_stats();

    auto x_tensor = ones({3, 3});
    Variable x(x_tensor, true);

    auto checkpointed_fn = [](const Variable& input) -> Variable {
        std::cout << "  Simple checkpoint: x*x\n";
        return input * input;
    };

    auto y = checkpoint_with_original(checkpointed_fn, x, &x);
    auto loss = sum(y);
    loss.backward();

    if (x.grad().has_value()) {
        float grad_value = x.grad()->data<float>()[0];
        std::cout << "  Gradient: " << grad_value << " (expected 2.0)\n";
    }
}

int main() {
    tenzor::initialize();

    // Test 1: Run NestedCheckpoints alone
    run_nested_checkpoint_test("NestedCheckpoints alone");

    // Test 2: Run simple checkpoint first, then nested
    run_simple_checkpoint_first();
    run_nested_checkpoint_test("After running simple x*x checkpoint");

    tenzor::finalize();
    return 0;
}
