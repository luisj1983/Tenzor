/**
 * @file test_autograd_thread_safety_multidtype.cpp
 * @brief Multi-backend, multi-dtype tests for concurrent backward pass safety
 *
 * Converted from test_autograd_thread_safety.cpp.  Uses MultiBackendDTypeTest
 * so that each test runs across all backend + dtype combinations.
 *
 * Result verification converts tensors to CPU Float32 before reading with
 * data<float>().
 */

#include <gtest/gtest.h>
#include "../multi_backend_dtype_fixture.hpp"
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
using namespace tenzor::testing;

class AutogradThreadSafetyMultiDTypeTest : public MultiBackendDTypeTest {};

// ============================================================================
// Concurrent backward with shared parameters
// ============================================================================

TEST_P(AutogradThreadSafetyMultiDTypeTest, ConcurrentBackwardSharedParam) {
    // Create a shared parameter used by multiple computation graphs
    auto w = Variable(createRandn({4, 4}), true);
    w.make_thread_safe();

    constexpr int num_threads = 4;
    constexpr int iters_per_thread = 10;

    std::atomic<int> completed{0};
    std::vector<std::thread> threads;

    // Capture dtype/device for use inside thread lambdas
    auto dev = device();
    auto dt = dtype();

    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&w, &completed, dev, dt]() {
            for (int i = 0; i < iters_per_thread; ++i) {
                // Each thread creates its own input and graph but shares w
                auto x_t = tenzor::randn({4, 4}, DType::Float32, dev);
                if (dt != DType::Float32) {
                    x_t = x_t.to(dt);
                }
                auto x = Variable(x_t, false);
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

TEST_P(AutogradThreadSafetyMultiDTypeTest, ConcurrentBackwardIndependentParams) {
    // Each thread has its own parameters -- no contention, but exercises
    // the thread-local BackwardEngine isolation
    constexpr int num_threads = 4;

    std::atomic<int> completed{0};
    std::vector<std::thread> threads;

    auto dev = device();
    auto dt = dtype();

    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&completed, dev, dt]() {
            auto make_randn = [&](std::vector<int64_t> shape) {
                auto t = tenzor::randn(shape, DType::Float32, dev);
                if (dt != DType::Float32) t = t.to(dt);
                return t;
            };
            auto a = Variable(make_randn({3, 3}), true);
            auto b = Variable(make_randn({3, 3}), true);
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

TEST_P(AutogradThreadSafetyMultiDTypeTest, SharedParamGradientAccumulation) {
    // Verify that concurrent gradient accumulations to a shared parameter
    // produce a result consistent with sequential accumulation.
    auto w = Variable(ones({2, 2}, dtype(), device()), true);
    w.make_thread_safe();

    constexpr int num_threads = 8;

    // Each thread computes: loss = sum(ones * w) = sum(w)
    // Gradient of sum(ones * w) w.r.t. w = ones (for element-wise mul)
    // Sum of 2x2 ones = 4. With num_threads, total grad_sum = num_threads * 4

    std::vector<std::thread> threads;
    auto dev = device();
    auto dt = dtype();

    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&w, dev, dt]() {
            auto x = Variable(ones({2, 2}, dt, dev), false);
            auto y = x * w;
            auto loss = tenzor::sum(y);
            // Don't clear grads -- let them accumulate across threads
            loss.backward();
        });
    }

    for (auto& th : threads) {
        th.join();
    }

    // All threads accumulated to w.grad()
    ASSERT_TRUE(w.has_grad()) << "Shared parameter should have gradient";

    auto grad = w.grad().value().to(Device::cpu()).to(DType::Float32);
    auto grad_sum = tenzor::sum(grad).data<float>()[0];

    // Each backward contributes grad = ones(2,2), sum = 4
    // Total accumulated = num_threads * 4 = 32
    float expected = static_cast<float>(num_threads) * 4.0f;
    EXPECT_NEAR(grad_sum, expected, atol() * num_threads)
        << "Accumulated gradient should equal sum of all thread contributions";
}

TEST_P(AutogradThreadSafetyMultiDTypeTest, MultiPathGraph) {
    // Single thread, but the parameter appears multiple times in the graph.
    // This tests that the engine correctly accumulates gradients from multiple
    // paths through the same shared parameter.
    auto w = Variable(ones({3, 3}, dtype(), device()), true);

    // y = w * w (w used twice)
    auto y1 = w * w;
    auto loss = tenzor::sum(y1);
    loss.backward();

    ASSERT_TRUE(w.has_grad());
    // d/dW sum(W * W) = 2 * W = 2 * ones = [[2,2,2],[2,2,2],[2,2,2]]
    // Sum of gradient = 2 * 9 = 18
    auto grad = w.grad().value().to(Device::cpu()).to(DType::Float32);
    auto grad_sum = tenzor::sum(grad).data<float>()[0];
    EXPECT_NEAR(grad_sum, 18.0f, atol() * 18.0f);
}

// ============================================================================
// Instantiate for all available backends and dtypes
// ============================================================================

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(AutogradThreadSafetyMultiDTypeTest);
