/**
 * @file test_async_ops_multidtype.cpp
 * @brief Multi-backend multi-dtype tests for asynchronous tensor operations
 *
 * Only includes tensor operation tests from test_async_ops.cpp.
 * Future/Promise infrastructure tests are intentionally excluded since they
 * don't create tensors and don't benefit from backend/dtype parameterization.
 *
 * Tests async operations with Float32, Float64, and Float16 dtypes across
 * CPU, CUDA, OneAPI, Vulkan, and ROCm backends.
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
#include <cmath>

using namespace tenzor;
using namespace tenzor::testing;

// ============================================================================
// Async Ops Multi-Backend Multi-DType Test Fixture
// ============================================================================

class AsyncOpsMultiDTypeTest : public MultiBackendDTypeTest {
protected:
    /// Compare two tensors by converting to CPU Float32 for data access
    void verifyNear(const Tensor& result, const Tensor& expected,
                    float tolerance, const std::string& label) {
        auto result_cpu = result.to(Device::cpu()).to(DType::Float32);
        auto expected_cpu = expected.to(Device::cpu()).to(DType::Float32);
        const float* r = result_cpu.data<float>();
        const float* e = expected_cpu.data<float>();
        for (int64_t i = 0; i < result.numel(); ++i) {
            EXPECT_NEAR(r[i], e[i], tolerance)
                << label << " mismatch at index " << i;
        }
    }
};

// ============================================================================
// Async Operation Correctness Tests
// ============================================================================

TEST_P(AsyncOpsMultiDTypeTest, AsyncMatmulCorrectness) {
    // Float16 matmul can accumulate significant error on large dims
    if (dtype() == DType::Float16) {
        GTEST_SKIP() << "Skipping matmul precision test for Float16";
    }

    auto a = createRandn({32, 64});
    auto b = createRandn({64, 48});

    auto future = async_matmul(a, b);
    auto expected = matmul(a, b);
    auto result = future.wait();

    expectShape(result, {32, 48});
    verifyNear(result, expected, atol(), "AsyncMatmul");
}

TEST_P(AsyncOpsMultiDTypeTest, AsyncAddCorrectness) {
    auto a = createRandn({100, 100});
    auto b = createRandn({100, 100});

    auto future = async_add(a, b);
    auto expected = add(a, b);
    auto result = future.wait();

    expectShape(result, {100, 100});
    verifyNear(result, expected, atol(), "AsyncAdd");
}

TEST_P(AsyncOpsMultiDTypeTest, AsyncMulCorrectness) {
    auto a = createRandn({50, 50});
    auto b = createRandn({50, 50});

    auto future = async_mul(a, b);
    auto expected = mul(a, b);
    auto result = future.wait();

    expectShape(result, {50, 50});
    verifyNear(result, expected, atol(), "AsyncMul");
}

TEST_P(AsyncOpsMultiDTypeTest, AsyncSubCorrectness) {
    auto a = createRandn({80, 60});
    auto b = createRandn({80, 60});

    auto future = async_sub(a, b);
    auto expected = sub(a, b);
    auto result = future.wait();

    expectShape(result, {80, 60});
    verifyNear(result, expected, atol(), "AsyncSub");
}

TEST_P(AsyncOpsMultiDTypeTest, AsyncDivCorrectness) {
    auto a = createRandn({40, 40});
    auto b = createRandn({40, 40});

    // Avoid division by zero: add 1
    auto ones_t = ones({40, 40}, dtype(), device());
    b = add(b, ones_t);

    auto future = async_div(a, b);
    auto expected = div(a, b);
    auto result = future.wait();

    expectShape(result, {40, 40});
    verifyNear(result, expected, atol(), "AsyncDiv");
}

TEST_P(AsyncOpsMultiDTypeTest, AsyncReLUCorrectness) {
    auto input = createRandn({100, 100});
    auto half_t = full({100, 100}, 0.5f, dtype(), device());
    input = sub(input, half_t);  // Make some values negative

    auto future = async_relu(input);
    auto result = future.wait();

    // Convert to CPU Float32 for verification
    auto result_cpu = result.to(Device::cpu()).to(DType::Float32);
    auto input_cpu = input.to(Device::cpu()).to(DType::Float32);
    const float* r = result_cpu.data<float>();
    const float* inp = input_cpu.data<float>();

    for (int64_t i = 0; i < result.numel(); ++i) {
        EXPECT_GE(r[i], 0.0f);
        EXPECT_NEAR(r[i], std::max(0.0f, inp[i]), atol());
    }
}

TEST_P(AsyncOpsMultiDTypeTest, AsyncSigmoidCorrectness) {
    auto input = createRandn({50, 50});

    auto future = async_sigmoid(input);
    auto result = future.wait();

    auto result_cpu = result.to(Device::cpu()).to(DType::Float32);
    auto input_cpu = input.to(Device::cpu()).to(DType::Float32);
    const float* r = result_cpu.data<float>();
    const float* inp = input_cpu.data<float>();

    for (int64_t i = 0; i < result.numel(); ++i) {
        EXPECT_GT(r[i], 0.0f);
        EXPECT_LT(r[i], 1.0f);
        float expected_val = 1.0f / (1.0f + std::exp(-inp[i]));
        EXPECT_NEAR(r[i], expected_val, atol());
    }
}

TEST_P(AsyncOpsMultiDTypeTest, AsyncTanhCorrectness) {
    auto input = createRandn({60, 40});

    auto future = async_tanh(input);
    auto result = future.wait();

    auto expected = tanh(input);

    auto result_cpu = result.to(Device::cpu()).to(DType::Float32);
    auto expected_cpu = expected.to(Device::cpu()).to(DType::Float32);
    const float* r = result_cpu.data<float>();
    const float* e = expected_cpu.data<float>();

    for (int64_t i = 0; i < result.numel(); ++i) {
        EXPECT_GT(r[i], -1.0f);
        EXPECT_LT(r[i], 1.0f);
        EXPECT_NEAR(r[i], e[i], atol());
    }
}

TEST_P(AsyncOpsMultiDTypeTest, AsyncSoftmaxCorrectness) {
    auto input = createRandn({32, 10});

    auto future = async_softmax(input, 1);
    auto result = future.wait();

    auto result_cpu = result.to(Device::cpu()).to(DType::Float32);
    const float* r = result_cpu.data<float>();

    // Accumulation of 10 values needs looser tolerance
    float sum_tol = std::max(atol() * 10.0f, 1e-5f);

    for (int64_t i = 0; i < 32; ++i) {
        float row_sum = 0.0f;
        for (int64_t j = 0; j < 10; ++j) {
            float val = r[i * 10 + j];
            EXPECT_GT(val, 0.0f);
            EXPECT_LT(val, 1.0f);
            row_sum += val;
        }
        EXPECT_NEAR(row_sum, 1.0f, sum_tol);
    }
}

// ============================================================================
// Non-blocking Behavior Tests
// ============================================================================

TEST_P(AsyncOpsMultiDTypeTest, AsyncOperationsNonBlocking) {
    auto a = createRandn({512, 512});
    auto b = createRandn({512, 512});

    auto start = std::chrono::high_resolution_clock::now();
    auto future = async_matmul(a, b);
    auto after_submit = std::chrono::high_resolution_clock::now();

    // Submission should be nearly instant (< 10ms)
    auto submit_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        after_submit - start).count();
    EXPECT_LT(submit_ms, 10);

    EXPECT_FALSE(future.is_ready());

    auto result = future.wait();
    expectShape(result, {512, 512});
}

TEST_P(AsyncOpsMultiDTypeTest, MultipleAsyncOperationsOverlap) {
    auto a1 = createRandn({128, 128});
    auto b1 = createRandn({128, 128});
    auto a2 = createRandn({128, 128});
    auto b2 = createRandn({128, 128});
    auto a3 = createRandn({128, 128});
    auto b3 = createRandn({128, 128});

    auto start = std::chrono::high_resolution_clock::now();

    auto f1 = async_matmul(a1, b1);
    auto f2 = async_matmul(a2, b2);
    auto f3 = async_matmul(a3, b3);

    auto after_submit = std::chrono::high_resolution_clock::now();
    auto submit_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        after_submit - start).count();
    EXPECT_LT(submit_ms, 50);

    auto r1 = f1.wait();
    auto r2 = f2.wait();
    auto r3 = f3.wait();

    expectShape(r1, {128, 128});
    expectShape(r2, {128, 128});
    expectShape(r3, {128, 128});
}

// ============================================================================
// Generic Async Execute Wrapper
// ============================================================================

TEST_P(AsyncOpsMultiDTypeTest, AsyncExecuteGenericWrapper) {
    auto input = createRandn({100, 100});

    auto five = full({100, 100}, 5.0f, dtype(), device());
    auto future = async_execute([&five](const Tensor& t) {
        return add(t, five);
    }, input);

    auto result = future.wait();

    auto expected = add(input, five);
    verifyNear(result, expected, atol(), "AsyncExecuteGeneric");
}

// ============================================================================
// Utility Function Tests
// ============================================================================

TEST_P(AsyncOpsMultiDTypeTest, WaitAll) {
    auto a1 = createRandn({50, 50});
    auto a2 = createRandn({50, 50});
    auto a3 = createRandn({50, 50});

    auto f1 = async_relu(a1);
    auto f2 = async_sigmoid(a2);
    auto f3 = async_tanh(a3);

    std::vector<Future<Tensor>> futures;
    futures.push_back(std::move(f1));
    futures.push_back(std::move(f2));
    futures.push_back(std::move(f3));

    auto results = wait_all(futures);

    EXPECT_EQ(results.size(), 3u);
    expectShape(results[0], {50, 50});
    expectShape(results[1], {50, 50});
    expectShape(results[2], {50, 50});
}

TEST_P(AsyncOpsMultiDTypeTest, WaitAny) {
    auto a = createRandn({100, 100});
    auto b = createRandn({100, 100});

    auto f1 = async_add(a, b);       // Fast
    auto f2 = async_matmul(a, b);    // Potentially slower

    std::vector<Future<Tensor>> futures;
    futures.push_back(std::move(f1));
    futures.push_back(std::move(f2));

    int64_t ready_idx = wait_any(futures);

    EXPECT_GE(ready_idx, 0);
    EXPECT_LT(ready_idx, 2);
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST_P(AsyncOpsMultiDTypeTest, EmptyTensorHandling) {
    auto empty = zeros({0}, dtype(), device());

    auto future = async_relu(empty);
    auto result = future.wait();

    EXPECT_EQ(result.numel(), 0);
    expectShape(result, {0});
}

TEST_P(AsyncOpsMultiDTypeTest, BroadcastingInAsyncOps) {
    auto a = createRandn({100, 1});
    auto b = createRandn({1, 100});

    auto future = async_add(a, b);
    auto result = future.wait();

    // Should broadcast to {100, 100}
    expectShape(result, {100, 100});

    auto expected = add(a, b);
    verifyNear(result, expected, atol(), "AsyncBroadcast");
}

// ============================================================================
// Test Instantiation
// ============================================================================

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(AsyncOpsMultiDTypeTest);
