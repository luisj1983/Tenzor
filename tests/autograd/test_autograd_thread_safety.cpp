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
#include "../backend_test_fixture.hpp"

#include <thread>
#include <vector>
#include <atomic>

using namespace tenzor;

// Parameterized over all backends via BackendTest: every tensor is created on
// the fixture's `device`. The concurrent-backward logic itself is device-
// agnostic autograd engine code; tensors created inside worker threads must
// also live on `device`, so the device is captured by value into each lambda.
class AutogradThreadSafety : public ::tenzor::testing::BackendTest {};

// ============================================================================
// Concurrent backward with shared parameters
// ============================================================================

TEST_P(AutogradThreadSafety, ConcurrentBackwardSharedParam) {
    // Create a shared parameter used by multiple computation graphs
    auto w = Variable(randn({4, 4}, DType::Float32, device), true);
    w.make_thread_safe();

    constexpr int num_threads = 4;
    constexpr int iters_per_thread = 10;

    std::atomic<int> completed{0};
    std::vector<std::thread> threads;

    tenzor::Device dev = device;
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&w, &completed, dev]() {
            for (int i = 0; i < iters_per_thread; ++i) {
                // Each thread creates its own input and graph but shares w
                auto x = Variable(randn({4, 4}, DType::Float32, dev), false);
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

TEST_P(AutogradThreadSafety, ConcurrentBackwardIndependentParams) {
    // Each thread has its own parameters — no contention, but exercises
    // the thread-local BackwardEngine isolation
    constexpr int num_threads = 4;

    std::atomic<int> completed{0};
    std::vector<std::thread> threads;

    tenzor::Device dev = device;
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&completed, dev]() {
            auto a = Variable(randn({3, 3}, DType::Float32, dev), true);
            auto b = Variable(randn({3, 3}, DType::Float32, dev), true);
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

TEST_P(AutogradThreadSafety, SharedParamGradientAccumulation) {
    // Verify that concurrent gradient accumulations to a shared parameter
    // produce a result consistent with sequential accumulation.
    auto w = Variable(ones({2, 2}, DType::Float32, device), true);
    w.make_thread_safe();

    constexpr int num_threads = 8;

    // Each thread computes: loss = sum(ones * w) = sum(w)
    // Gradient of sum(ones * w) w.r.t. w = ones (for element-wise mul)
    // Sum of 2x2 ones = 4. With num_threads, total grad_sum = num_threads * 4

    tenzor::Device dev = device;
    std::vector<std::thread> threads;
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&w, dev]() {
            auto x = Variable(ones({2, 2}, DType::Float32, dev), false);
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
    auto grad_sum = tenzor::sum(grad).cpu().data<float>()[0];

    // Each backward contributes grad = ones(2,2), sum = 4
    // Total accumulated = num_threads * 4 = 32
    float expected = static_cast<float>(num_threads) * 4.0f;
    EXPECT_NEAR(grad_sum, expected, 1e-4f)
        << "Accumulated gradient should equal sum of all thread contributions";
}

TEST_P(AutogradThreadSafety, MultiPathGraph) {
    // Single thread, but the parameter appears multiple times in the graph.
    // This tests that the engine correctly accumulates gradients from multiple
    // paths through the same shared parameter.
    auto w = Variable(ones({3, 3}, DType::Float32, device), true);

    // y = w * w (w used twice)
    auto y1 = w * w;
    auto loss = tenzor::sum(y1);
    loss.backward();

    ASSERT_TRUE(w.has_grad());
    // d/dW sum(W * W) = 2 * W = 2 * ones = [[2,2,2],[2,2,2],[2,2,2]]
    // Sum of gradient = 2 * 9 = 18
    auto grad_sum = tenzor::sum(w.grad().value()).cpu().data<float>()[0];
    EXPECT_NEAR(grad_sum, 18.0f, 1e-4f);
}

TEST_P(AutogradThreadSafety, ConcurrentGradReadWithoutMakeThreadSafe) {
    // L1: Variable::grad()/has_grad()/accumulate_grad()/zero_grad() used to
    // only take grad_mutex_ when make_thread_safe() had been called, while
    // BackwardEngine::execute()'s leaf-accumulation lock took it
    // unconditionally on every leaf regardless of that flag — an easy-to-
    // miss asymmetry: a user thread reading .grad() without opting in raced
    // against a concurrent backward() writing it under lock. Locking is now
    // unconditional on both sides, so this must be race-free WITHOUT calling
    // make_thread_safe() at all — the opposite of the tests above, which
    // exercise the (now-legacy no-op) opt-in path.
    //
    // This is a best-effort stress test, not a guaranteed race detector: a
    // plain (non-TSAN) build can't deterministically prove the old
    // unconditional-vs-conditional lock asymmetry was unsafe, since the
    // race window is a few pointer-sized reads/writes. It does exercise the
    // exact interleaving the finding describes under real contention, and
    // will reliably crash/corrupt under ThreadSanitizer (-DTENZOR_ENABLE_TSAN)
    // if the accessor-side lock is ever made conditional again.
    auto w = Variable(randn({8, 8}, DType::Float32, device), true);
    // Deliberately NOT calling w.make_thread_safe().

    constexpr int num_writer_threads = 4;
    constexpr int num_reader_threads = 4;
    constexpr int iters = 200;

    std::atomic<bool> stop{false};
    std::atomic<int> writer_done{0};
    std::atomic<int> reader_iterations{0};
    std::vector<std::thread> threads;

    tenzor::Device dev = device;
    for (int t = 0; t < num_writer_threads; ++t) {
        threads.emplace_back([&w, &writer_done, dev]() {
            for (int i = 0; i < iters; ++i) {
                auto x = Variable(randn({8, 8}, DType::Float32, dev), false);
                auto y = x * w;
                auto loss = tenzor::sum(y);
                loss.backward();
                w.zero_grad();
            }
            ++writer_done;
        });
    }
    for (int t = 0; t < num_reader_threads; ++t) {
        threads.emplace_back([&w, &stop, &reader_iterations]() {
            // has_grad() and grad() are each individually synchronized, but
            // NOT as one atomic transaction across both calls — a writer's
            // zero_grad() can legitimately land between them, so checking
            // has_grad()==true implies a later grad().has_value()==true
            // would be a test bug, not a real race. Instead, call grad()
            // alone and check that WHATEVER it returns is internally
            // well-formed (a torn/half-written optional<Tensor> would show
            // up as a Tensor with a null/garbage data pointer or a shape
            // that doesn't match w's).
            int local_iters = 0;
            while (!stop.load(std::memory_order_relaxed)) {
                auto g = w.grad();
                if (g.has_value()) {
                    EXPECT_NE(g->data_ptr(), nullptr)
                        << "grad() returned a value with a null data "
                           "pointer — torn read of grad_ under concurrent "
                           "accumulation";
                    auto g_shape = std::vector<int64_t>(g->shape().begin(), g->shape().end());
                    auto w_shape = std::vector<int64_t>(w.tensor().shape().begin(), w.tensor().shape().end());
                    EXPECT_EQ(g_shape, w_shape)
                        << "grad() returned a value with the wrong shape — "
                           "torn read of grad_ under concurrent accumulation";
                }
                ++local_iters;
            }
            reader_iterations += local_iters;
        });
    }

    for (int t = 0; t < num_writer_threads; ++t) {
        threads[t].join();
    }
    stop.store(true, std::memory_order_relaxed);
    for (int t = num_writer_threads; t < num_writer_threads + num_reader_threads; ++t) {
        threads[t].join();
    }

    EXPECT_EQ(writer_done.load(), num_writer_threads);
    EXPECT_GT(reader_iterations.load(), 0)
        << "Reader threads should have observed at least one iteration";
}

INSTANTIATE_BACKEND_TESTS(AutogradThreadSafety);
