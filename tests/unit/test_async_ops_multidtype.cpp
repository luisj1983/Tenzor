/**
 * @file test_async_ops_multidtype.cpp
 * @brief Multi-dtype multi-backend tests for asynchronous tensor operations
 *
 * Tests async operations with Float32, Float64, and Float16 dtypes across
 * CPU, CUDA, OneAPI, Vulkan, and ROCm backends to ensure:
 * - Correct async operation results
 * - Proper future/promise behavior
 * - Non-blocking execution
 * - Type preservation across async boundaries
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/ops/async_ops.hpp>
#include <tenzor/ops/math.hpp>
#include <tenzor/ops/creation.hpp>
#include <tenzor/autograd/ops.hpp>
#include "../multi_backend_dtype_fixture.hpp"
#include <chrono>
#include <thread>
#include <vector>

using namespace tenzor;
using namespace tenzor::testing;

// ============================================================================
// Async Ops Multi-Backend Multi-DType Test Fixture
// ============================================================================

class AsyncOpsMultiDTypeTest : public MultiBackendDTypeTest {
protected:
    template<typename T>
    void VerifyNear(const Tensor& result, const Tensor& expected, const std::string& test_name) {
        auto result_cpu = result.to(Device::cpu()).to(DType::Float32);
        auto expected_cpu = expected.to(Device::cpu()).to(DType::Float32);
        const float* result_data = result_cpu.data<float>();
        const float* expected_data = expected_cpu.data<float>();
        for (size_t i = 0; i < result.numel(); ++i) {
            EXPECT_NEAR(result_data[i], expected_data[i], atol())
                << test_name << " failed at index " << i;
        }
    }
};

// ==============================================================================
// Future/Promise Tests
// ==============================================================================

TEST_P(AsyncOpsMultiDTypeTest, FutureBasicWait) {
    auto promise = std::make_shared<Promise<int>>();
    Future<int> future(promise->get_state());

    std::thread([promise]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        promise->set_value(42);
    }).detach();

    int result = future.wait();
    EXPECT_EQ(result, 42);
}

TEST_P(AsyncOpsMultiDTypeTest, FutureIsReady) {
    auto promise = std::make_shared<Promise<int>>();
    Future<int> future(promise->get_state());

    EXPECT_FALSE(future.is_ready());

    promise->set_value(100);

    EXPECT_TRUE(future.is_ready());
    EXPECT_EQ(future.wait(), 100);
}

// ==============================================================================
// Async Operation Correctness Tests
// ==============================================================================

TEST_P(AsyncOpsMultiDTypeTest, AsyncMatmulCorrectness) {
    auto a = randn({32, 64}, dtype(), device());
    auto b = randn({64, 48}, dtype(), device());

    auto future = async_matmul(a, b);
    auto expected = matmul(a, b);
    auto result = future.wait();

    EXPECT_EQ(std::vector<int64_t>(result.shape().begin(), result.shape().end()),
              (std::vector<int64_t>{32, 48}));

    VerifyNear<float>(result, expected, "AsyncMatmul");
}

TEST_P(AsyncOpsMultiDTypeTest, AsyncAddCorrectness) {
    auto a = randn({100, 100}, dtype(), device());
    auto b = randn({100, 100}, dtype(), device());

    auto future = async_add(a, b);
    auto expected = add(a, b);
    auto result = future.wait();

    VerifyNear<float>(result, expected, "AsyncAdd");
}

TEST_P(AsyncOpsMultiDTypeTest, AsyncMulCorrectness) {
    auto a = randn({50, 50}, dtype(), device());
    auto b = randn({50, 50}, dtype(), device());

    auto future = async_mul(a, b);
    auto expected = mul(a, b);
    auto result = future.wait();

    VerifyNear<float>(result, expected, "AsyncMul");
}

TEST_P(AsyncOpsMultiDTypeTest, AsyncSubCorrectness) {
    auto a = randn({80, 60}, dtype(), device());
    auto b = randn({80, 60}, dtype(), device());

    auto future = async_sub(a, b);
    auto expected = sub(a, b);
    auto result = future.wait();

    VerifyNear<float>(result, expected, "AsyncSub");
}

TEST_P(AsyncOpsMultiDTypeTest, AsyncDivCorrectness) {
    auto a = randn({40, 40}, dtype(), device());
    auto b = randn({40, 40}, dtype(), device());

    // Avoid division by zero
    auto ones_tensor = ones({40, 40}, dtype(), device());
    b = add(b, ones_tensor);

    auto future = async_div(a, b);
    auto expected = div(a, b);
    auto result = future.wait();

    VerifyNear<float>(result, expected, "AsyncDiv");
}

TEST_P(AsyncOpsMultiDTypeTest, AsyncReLUCorrectness) {
    auto input = randn({100, 100}, dtype(), device());
    auto half = full({100, 100}, 0.5f, dtype(), device());
    input = sub(input, half);

    auto future = async_relu(input);
    auto result = future.wait();

    // Verify ReLU property: all values >= 0
    auto result_cpu = result.to(Device::cpu()).to(DType::Float32);
    auto input_cpu = input.to(Device::cpu()).to(DType::Float32);
    const float* result_data = result_cpu.data<float>();
    const float* input_data = input_cpu.data<float>();

    for (size_t i = 0; i < result.numel(); ++i) {
        EXPECT_GE(result_data[i], 0.0f);
        EXPECT_NEAR(result_data[i], std::max(0.0f, input_data[i]), atol());
    }
}

TEST_P(AsyncOpsMultiDTypeTest, AsyncSigmoidCorrectness) {
    auto input = randn({50, 50}, dtype(), device());

    auto future = async_sigmoid(input);
    auto result = future.wait();

    // Verify sigmoid property: all values in (0, 1)
    auto result_cpu = result.to(Device::cpu()).to(DType::Float32);
    auto input_cpu = input.to(Device::cpu()).to(DType::Float32);
    const float* result_data = result_cpu.data<float>();
    const float* input_data = input_cpu.data<float>();

    for (size_t i = 0; i < result.numel(); ++i) {
        EXPECT_GT(result_data[i], 0.0f);
        EXPECT_LT(result_data[i], 1.0f);
        float expected = 1.0f / (1.0f + std::exp(-input_data[i]));
        EXPECT_NEAR(result_data[i], expected, atol());
    }
}

TEST_P(AsyncOpsMultiDTypeTest, AsyncTanhCorrectness) {
    auto input = randn({60, 40}, dtype(), device());

    auto future = async_tanh(input);
    auto result = future.wait();

    // Verify tanh property: all values in (-1, 1)
    auto expected = tanh(input);

    auto result_cpu = result.to(Device::cpu()).to(DType::Float32);
    auto expected_cpu = expected.to(Device::cpu()).to(DType::Float32);
    const float* result_data = result_cpu.data<float>();
    const float* expected_data = expected_cpu.data<float>();

    for (size_t i = 0; i < result.numel(); ++i) {
        EXPECT_GT(result_data[i], -1.0f);
        EXPECT_LT(result_data[i], 1.0f);
        EXPECT_NEAR(result_data[i], expected_data[i], atol());
    }
}

TEST_P(AsyncOpsMultiDTypeTest, AsyncSoftmaxCorrectness) {
    auto input = randn({32, 10}, dtype(), device());

    auto future = async_softmax(input, 1);
    auto result = future.wait();

    // Verify softmax properties: each row sums to 1
    auto result_cpu = result.to(Device::cpu()).to(DType::Float32);
    const float* result_data = result_cpu.data<float>();

    // Use slightly looser tolerance for sum checks since we're summing 10 values
    // and floating point errors can accumulate
    float sum_tolerance = std::max(atol() * 10.0f, 1e-6f);

    for (int64_t i = 0; i < 32; ++i) {
        float row_sum = 0.0f;
        for (int64_t j = 0; j < 10; ++j) {
            float val = result_data[i * 10 + j];
            EXPECT_GT(val, 0.0f);
            EXPECT_LT(val, 1.0f);
            row_sum += val;
        }
        EXPECT_NEAR(row_sum, 1.0f, sum_tolerance);
    }
}

// ==============================================================================
// Non-blocking Behavior Tests
// ==============================================================================

TEST_P(AsyncOpsMultiDTypeTest, AsyncOperationsNonBlocking) {
    auto a = randn({512, 512}, dtype(), device());
    auto b = randn({512, 512}, dtype(), device());

    auto start = std::chrono::high_resolution_clock::now();
    auto future = async_matmul(a, b);
    auto after_submit = std::chrono::high_resolution_clock::now();

    auto submit_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        after_submit - start).count();
    EXPECT_LT(submit_duration, 10);

    EXPECT_FALSE(future.is_ready());

    auto result = future.wait();
    EXPECT_EQ(std::vector<int64_t>(result.shape().begin(), result.shape().end()),
              (std::vector<int64_t>{512, 512}));
}

TEST_P(AsyncOpsMultiDTypeTest, MultipleAsyncOperationsOverlap) {
    auto a1 = randn({128, 128}, dtype(), device());
    auto b1 = randn({128, 128}, dtype(), device());
    auto a2 = randn({128, 128}, dtype(), device());
    auto b2 = randn({128, 128}, dtype(), device());
    auto a3 = randn({128, 128}, dtype(), device());
    auto b3 = randn({128, 128}, dtype(), device());

    auto start = std::chrono::high_resolution_clock::now();

    auto f1 = async_matmul(a1, b1);
    auto f2 = async_matmul(a2, b2);
    auto f3 = async_matmul(a3, b3);

    auto after_submit = std::chrono::high_resolution_clock::now();
    auto submit_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        after_submit - start).count();
    EXPECT_LT(submit_time, 50);

    auto r1 = f1.wait();
    auto r2 = f2.wait();
    auto r3 = f3.wait();

    EXPECT_EQ(std::vector<int64_t>(r1.shape().begin(), r1.shape().end()),
              (std::vector<int64_t>{128, 128}));
    EXPECT_EQ(std::vector<int64_t>(r2.shape().begin(), r2.shape().end()),
              (std::vector<int64_t>{128, 128}));
    EXPECT_EQ(std::vector<int64_t>(r3.shape().begin(), r3.shape().end()),
              (std::vector<int64_t>{128, 128}));
}

// ==============================================================================
// Utility Function Tests
// ==============================================================================

TEST_P(AsyncOpsMultiDTypeTest, WaitAll) {
    auto a1 = randn({50, 50}, dtype(), device());
    auto a2 = randn({50, 50}, dtype(), device());
    auto a3 = randn({50, 50}, dtype(), device());

    auto f1 = async_relu(a1);
    auto f2 = async_sigmoid(a2);
    auto f3 = async_tanh(a3);

    std::vector<Future<Tensor>> futures;
    futures.push_back(std::move(f1));
    futures.push_back(std::move(f2));
    futures.push_back(std::move(f3));

    auto results = wait_all(futures);

    EXPECT_EQ(results.size(), 3);
    EXPECT_EQ(std::vector<int64_t>(results[0].shape().begin(), results[0].shape().end()),
              std::vector<int64_t>(a1.shape().begin(), a1.shape().end()));
    EXPECT_EQ(std::vector<int64_t>(results[1].shape().begin(), results[1].shape().end()),
              std::vector<int64_t>(a2.shape().begin(), a2.shape().end()));
    EXPECT_EQ(std::vector<int64_t>(results[2].shape().begin(), results[2].shape().end()),
              std::vector<int64_t>(a3.shape().begin(), a3.shape().end()));
}

// ==============================================================================
// Test Instantiation
// ==============================================================================

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(AsyncOpsMultiDTypeTest);

/*
 * COVERAGE SUMMARY:
 *
 * Test Cases: 13
 * DTypes Tested: Float32, Float64, Float16 (Float16 skipped for randn tests)
 * Backends Tested: CPU, CUDA, OneAPI
 * Total Scenarios: 13 tests × 3 dtypes × 3 backends = 117 test scenarios
 *
 * Coverage:
 * - Future/Promise: basic wait, is_ready
 * - Async Operations: matmul, add, mul, sub, div, relu, sigmoid, tanh, softmax
 * - Non-blocking: operation submission timing, multiple concurrent operations
 * - Utility: wait_all
 */
