/**
 * @file test_vulkan_threading.cpp
 * @brief Vulkan backend thread-safety tests
 *
 * Validates that concurrent operations on the Vulkan backend are safe,
 * particularly the getPipeline() mutex fix and per-device locking.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <thread>
#include <vector>
#include <atomic>

using namespace tenzor;

class VulkanThreadingEnvironment : public ::testing::Environment {
public:
    void SetUp() override {
        tenzor::initialize();
    }
};

static ::testing::Environment* const vulkan_thread_env =
    ::testing::AddGlobalTestEnvironment(new VulkanThreadingEnvironment);

class VulkanThreadingTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Skip if Vulkan not available
        try {
            auto t = zeros({1}, DType::Float32, Device::vulkan(0));
            vulkan_available_ = true;
        } catch (...) {
            vulkan_available_ = false;
        }
    }

    bool vulkan_available_ = false;
};

TEST_F(VulkanThreadingTest, ConcurrentMatMul) {
    if (!vulkan_available_) {
        GTEST_SKIP() << "Vulkan backend not available";
    }

    constexpr int num_threads = 4;
    constexpr int ops_per_thread = 10;
    std::atomic<int> successes{0};
    std::atomic<int> failures{0};

    auto worker = [&](int thread_id) {
        try {
            for (int i = 0; i < ops_per_thread; ++i) {
                auto a = randn({8, 16}, DType::Float32, Device::vulkan(0));
                auto b = randn({16, 8}, DType::Float32, Device::vulkan(0));
                auto c = matmul(a, b);
                // Verify output shape
                EXPECT_EQ(c.shape()[0], 8);
                EXPECT_EQ(c.shape()[1], 8);
                successes++;
            }
        } catch (const std::exception& e) {
            failures++;
        }
    };

    std::vector<std::thread> threads;
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back(worker, i);
    }

    for (auto& t : threads) {
        t.join();
    }

    EXPECT_EQ(successes.load(), num_threads * ops_per_thread);
    EXPECT_EQ(failures.load(), 0);
}

TEST_F(VulkanThreadingTest, ConcurrentMixedOps) {
    if (!vulkan_available_) {
        GTEST_SKIP() << "Vulkan backend not available";
    }

    constexpr int num_threads = 4;
    constexpr int ops_per_thread = 5;
    std::atomic<int> successes{0};
    std::atomic<int> failures{0};

    auto worker = [&](int thread_id) {
        try {
            for (int i = 0; i < ops_per_thread; ++i) {
                auto a = randn({32, 32}, DType::Float32, Device::vulkan(0));
                auto b = randn({32, 32}, DType::Float32, Device::vulkan(0));

                // Mix different operations that trigger different pipelines
                switch ((thread_id + i) % 4) {
                    case 0: {
                        auto c = add(a, b);
                        (void)c;
                        break;
                    }
                    case 1: {
                        auto c = mul(a, b);
                        (void)c;
                        break;
                    }
                    case 2: {
                        auto c = matmul(a, b);
                        (void)c;
                        break;
                    }
                    case 3: {
                        auto c = sub(a, b);
                        (void)c;
                        break;
                    }
                }
                successes++;
            }
        } catch (const std::exception& e) {
            failures++;
        }
    };

    std::vector<std::thread> threads;
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back(worker, i);
    }

    for (auto& t : threads) {
        t.join();
    }

    EXPECT_EQ(successes.load(), num_threads * ops_per_thread);
    EXPECT_EQ(failures.load(), 0);
}

TEST_F(VulkanThreadingTest, ConcurrentPipelineCreation) {
    if (!vulkan_available_) {
        GTEST_SKIP() << "Vulkan backend not available";
    }

    // Stress test: many threads all triggering pipeline creation simultaneously
    // This specifically tests the getPipeline() mutex fix
    constexpr int num_threads = 8;
    std::atomic<int> successes{0};

    auto worker = [&](int thread_id) {
        try {
            // Each thread does a unique operation that requires its own pipeline
            auto a = randn({16, 16}, DType::Float32, Device::vulkan(0));
            auto b = randn({16, 16}, DType::Float32, Device::vulkan(0));

            // These all trigger different pipelines
            auto c1 = add(a, b);
            auto c2 = sub(a, b);
            auto c3 = mul(a, b);
            auto c4 = exp(a);
            auto c5 = log(abs(a) + full({16, 16}, 1.0f, DType::Float32, Device::vulkan(0)));
            (void)c1; (void)c2; (void)c3; (void)c4; (void)c5;

            successes++;
        } catch (const std::exception& e) {
            ADD_FAILURE() << "Thread " << thread_id << " failed: " << e.what();
        }
    };

    std::vector<std::thread> threads;
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back(worker, i);
    }

    for (auto& t : threads) {
        t.join();
    }

    EXPECT_EQ(successes.load(), num_threads);
}
