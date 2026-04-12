/**
 * @file test_autograd_thread_safety.cpp
 * @brief Thread safety tests for backward pass gradient accumulation
 *
 * Verifies that concurrent backward() calls with shared parameters
 * produce correct results without data races or deadlocks.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include "tenzor/autograd/variable.hpp"
#include "tenzor/autograd/engine.hpp"
#include "tenzor/autograd/ops.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/reduction.hpp"

#include <thread>
#include <vector>
#include <atomic>

using namespace tenzor;

// ============================================================================
// Concurrent backward with shared parameters
// ============================================================================

TEST(AutogradThreadSafety, ConcurrentBackwardSharedParam) {
    // Create a shared parameter used by multiple computation graphs
    auto w = Variable(randn({4, 4}, DType::Float32, Device::cpu()), true);
    w.make_thread_safe();

    constexpr int num_threads = 4;
    constexpr int iters_per_thread = 10;

    std::atomic<int> completed{0};
    std::vector<std::thread> threads;

    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&w, &completed]() {
            for (int i = 0; i < iters_per_thread; ++i) {
                // Each thread creates its own input and graph but shares w
                auto x = Variable(randn({4, 4}, DType::Float32, Device::cpu()), false);
                auto y = x * w;
                auto loss = tenzor::sum(y);
                loss.backward();

                // Clear grad for next iteration
                w.zero_grad();
            }
            ++completed;
        });
    }

    for (auto& th : threads) {
        th.join();
    }

    EXPECT_EQ(completed.load(), num_threads)
        << "All threads should complete without deadlock";
}

TEST(AutogradThreadSafety, ConcurrentBackwardIndependentParams) {
    // Each thread has its own parameters — no contention, but exercises
    // the thread-local BackwardEngine isolation
    constexpr int num_threads = 4;

    std::atomic<int> completed{0};
    std::vector<std::thread> threads;

    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&completed]() {
            auto a = Variable(randn({3, 3}, DType::Float32, Device::cpu()), true);
            auto b = Variable(randn({3, 3}, DType::Float32, Device::cpu()), true);
            auto c = a * b;
            auto loss = tenzor::sum(c);
            loss.backward();

            EXPECT_TRUE(a.has_grad());
            EXPECT_TRUE(b.has_grad());
            ++completed;
        });
    }

    for (auto& th : threads) {
        th.join();
    }

    EXPECT_EQ(completed.load(), num_threads);
}

TEST(AutogradThreadSafety, SharedParamGradientAccumulation) {
    // Verify that concurrent gradient accumulations to a shared parameter
    // produce a result consistent with sequential accumulation.
    auto w = Variable(ones({2, 2}, DType::Float32, Device::cpu()), true);
    w.make_thread_safe();

    constexpr int num_threads = 8;

    // Each thread computes: loss = sum(ones * w) = sum(w)
    // Gradient of sum(ones * w) w.r.t. w = ones (for element-wise mul)
    // Sum of 2x2 ones = 4. With num_threads, total grad_sum = num_threads * 4

    std::vector<std::thread> threads;
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&w]() {
            auto x = Variable(ones({2, 2}, DType::Float32, Device::cpu()), false);
            auto y = x * w;
            auto loss = tenzor::sum(y);
            // Don't clear grads — let them accumulate across threads
            loss.backward();
        });
    }

    for (auto& th : threads) {
        th.join();
    }

    // All threads accumulated to w.grad()
    ASSERT_TRUE(w.has_grad()) << "Shared parameter should have gradient";

    auto grad = w.grad().value();
    auto grad_sum = tenzor::sum(grad).data<float>()[0];

    // Each backward contributes grad = ones(2,2), sum = 4
    // Total accumulated = num_threads * 4 = 32
    float expected = static_cast<float>(num_threads) * 4.0f;
    EXPECT_NEAR(grad_sum, expected, 1e-4f)
        << "Accumulated gradient should equal sum of all thread contributions";
}

TEST(AutogradThreadSafety, MultiPathGraph) {
    // Single thread, but the parameter appears multiple times in the graph.
    // This tests that the engine correctly accumulates gradients from multiple
    // paths through the same shared parameter.
    auto w = Variable(ones({3, 3}, DType::Float32, Device::cpu()), true);

    // y = w * w (w used twice)
    auto y1 = w * w;
    auto loss = tenzor::sum(y1);
    loss.backward();

    ASSERT_TRUE(w.has_grad());
    // d/dW sum(W * W) = 2 * W = 2 * ones = [[2,2,2],[2,2,2],[2,2,2]]
    // Sum of gradient = 2 * 9 = 18
    auto grad_sum = tenzor::sum(w.grad().value()).data<float>()[0];
    EXPECT_NEAR(grad_sum, 18.0f, 1e-4f);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);

    try {
        if (!::testing::GTEST_FLAG(list_tests)) {
            tenzor::initialize();
        }
    } catch (const std::exception& e) {
        std::cerr << "Failed to initialize Tenzor: " << e.what() << std::endl;
        return 1;
    }

    int result = RUN_ALL_TESTS();

    try {
        tenzor::finalize();
    } catch (...) {}

    return result;
}
