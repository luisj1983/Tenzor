/**
 * @file test_async_ops_multidtype.cpp
 * @brief Multi-dtype tests for asynchronous tensor operations
 *
 * Coverage: DType testing for async operations
 * - Primary dtypes: Float32, Float64
 * - Backend: CPU (async ops are backend-agnostic)
 *
 * Note: Async operations primarily work with floating-point types for
 * neural network operations. Testing on CPU only to avoid backend complexity.
 */

#include <gtest/gtest.h>
#include "tenzor/tenzor.hpp"
#include "tenzor/ops/async_ops.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/autograd/ops.hpp"
#include <chrono>
#include <thread>
#include <vector>

using namespace tenzor;

// ============================================================================
// DType Parameterization
// ============================================================================

struct DTypeParam {
    DType dtype;
    std::string dtype_name;
    double tolerance;

    std::string ToString() const {
        return dtype_name;
    }
};

// Required for gtest_discover_tests to show human-readable test names
void PrintTo(const DTypeParam& param, std::ostream* os) {
    *os << param.ToString();
}

class AsyncOpsMultiDTypeTest : public ::testing::TestWithParam<DTypeParam> {
protected:
    DType dtype;
    double tolerance;
    Device device;

    void SetUp() override {
        auto param = GetParam();
        dtype = param.dtype;
        tolerance = param.tolerance;
        device = Device::cpu();
    }

    template<typename T>
    void VerifyNear(const Tensor& result, const Tensor& expected, const std::string& test_name) {
        const T* result_data = result.data<T>();
        const T* expected_data = expected.data<T>();
        for (size_t i = 0; i < result.numel(); ++i) {
            EXPECT_NEAR(static_cast<double>(result_data[i]),
                       static_cast<double>(expected_data[i]),
                       tolerance)
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
    auto a = randn({32, 64}, dtype, device);
    auto b = randn({64, 48}, dtype, device);

    auto future = async_matmul(a, b);
    auto expected = matmul(a, b);
    auto result = future.wait();

    EXPECT_EQ(std::vector<int64_t>(result.shape().begin(), result.shape().end()),
              (std::vector<int64_t>{32, 48}));

    if (dtype == DType::Float32) {
        VerifyNear<float>(result, expected, "AsyncMatmul");
    } else {
        VerifyNear<double>(result, expected, "AsyncMatmul");
    }
}

TEST_P(AsyncOpsMultiDTypeTest, AsyncAddCorrectness) {
    auto a = randn({100, 100}, dtype, device);
    auto b = randn({100, 100}, dtype, device);

    auto future = async_add(a, b);
    auto expected = add(a, b);
    auto result = future.wait();

    if (dtype == DType::Float32) {
        VerifyNear<float>(result, expected, "AsyncAdd");
    } else {
        VerifyNear<double>(result, expected, "AsyncAdd");
    }
}

TEST_P(AsyncOpsMultiDTypeTest, AsyncMulCorrectness) {
    auto a = randn({50, 50}, dtype, device);
    auto b = randn({50, 50}, dtype, device);

    auto future = async_mul(a, b);
    auto expected = mul(a, b);
    auto result = future.wait();

    if (dtype == DType::Float32) {
        VerifyNear<float>(result, expected, "AsyncMul");
    } else {
        VerifyNear<double>(result, expected, "AsyncMul");
    }
}

TEST_P(AsyncOpsMultiDTypeTest, AsyncSubCorrectness) {
    auto a = randn({80, 60}, dtype, device);
    auto b = randn({80, 60}, dtype, device);

    auto future = async_sub(a, b);
    auto expected = sub(a, b);
    auto result = future.wait();

    if (dtype == DType::Float32) {
        VerifyNear<float>(result, expected, "AsyncSub");
    } else {
        VerifyNear<double>(result, expected, "AsyncSub");
    }
}

TEST_P(AsyncOpsMultiDTypeTest, AsyncDivCorrectness) {
    auto a = randn({40, 40}, dtype, device);
    auto b = randn({40, 40}, dtype, device);

    // Avoid division by zero
    auto ones_tensor = ones({40, 40}, dtype, device);
    b = add(b, ones_tensor);

    auto future = async_div(a, b);
    auto expected = div(a, b);
    auto result = future.wait();

    if (dtype == DType::Float32) {
        VerifyNear<float>(result, expected, "AsyncDiv");
    } else {
        VerifyNear<double>(result, expected, "AsyncDiv");
    }
}

TEST_P(AsyncOpsMultiDTypeTest, AsyncReLUCorrectness) {
    auto input = randn({100, 100}, dtype, device);
    auto half = full({100, 100}, dtype == DType::Float32 ? 0.5f : 0.5, dtype, device);
    input = sub(input, half);

    auto future = async_relu(input);
    auto result = future.wait();

    // Verify ReLU property: all values >= 0
    if (dtype == DType::Float32) {
        const float* result_data = result.data<float>();
        const float* input_data = input.data<float>();
        for (size_t i = 0; i < result.numel(); ++i) {
            EXPECT_GE(result_data[i], 0.0f);
            EXPECT_FLOAT_EQ(result_data[i], std::max(0.0f, input_data[i]));
        }
    } else {
        const double* result_data = result.data<double>();
        const double* input_data = input.data<double>();
        for (size_t i = 0; i < result.numel(); ++i) {
            EXPECT_GE(result_data[i], 0.0);
            EXPECT_NEAR(result_data[i], std::max(0.0, input_data[i]), tolerance);
        }
    }
}

TEST_P(AsyncOpsMultiDTypeTest, AsyncSigmoidCorrectness) {
    auto input = randn({50, 50}, dtype, device);

    auto future = async_sigmoid(input);
    auto result = future.wait();

    // Verify sigmoid property: all values in (0, 1)
    if (dtype == DType::Float32) {
        const float* result_data = result.data<float>();
        const float* input_data = input.data<float>();
        for (size_t i = 0; i < result.numel(); ++i) {
            EXPECT_GT(result_data[i], 0.0f);
            EXPECT_LT(result_data[i], 1.0f);
            float expected = 1.0f / (1.0f + std::exp(-input_data[i]));
            EXPECT_NEAR(result_data[i], expected, tolerance);
        }
    } else {
        const double* result_data = result.data<double>();
        const double* input_data = input.data<double>();
        for (size_t i = 0; i < result.numel(); ++i) {
            EXPECT_GT(result_data[i], 0.0);
            EXPECT_LT(result_data[i], 1.0);
            double expected = 1.0 / (1.0 + std::exp(-input_data[i]));
            EXPECT_NEAR(result_data[i], expected, tolerance);
        }
    }
}

TEST_P(AsyncOpsMultiDTypeTest, AsyncTanhCorrectness) {
    auto input = randn({60, 40}, dtype, device);

    auto future = async_tanh(input);
    auto result = future.wait();

    // Verify tanh property: all values in (-1, 1)
    auto expected = tanh(input);

    if (dtype == DType::Float32) {
        const float* result_data = result.data<float>();
        const float* expected_data = expected.data<float>();
        for (size_t i = 0; i < result.numel(); ++i) {
            EXPECT_GT(result_data[i], -1.0f);
            EXPECT_LT(result_data[i], 1.0f);
            EXPECT_NEAR(result_data[i], expected_data[i], tolerance);
        }
    } else {
        const double* result_data = result.data<double>();
        const double* expected_data = expected.data<double>();
        for (size_t i = 0; i < result.numel(); ++i) {
            EXPECT_GT(result_data[i], -1.0);
            EXPECT_LT(result_data[i], 1.0);
            EXPECT_NEAR(result_data[i], expected_data[i], tolerance);
        }
    }
}

TEST_P(AsyncOpsMultiDTypeTest, AsyncSoftmaxCorrectness) {
    auto input = randn({32, 10}, dtype, device);

    auto future = async_softmax(input, 1);
    auto result = future.wait();

    // Verify softmax properties: each row sums to 1
    if (dtype == DType::Float32) {
        const float* result_data = result.data<float>();
        for (int64_t i = 0; i < 32; ++i) {
            float row_sum = 0.0f;
            for (int64_t j = 0; j < 10; ++j) {
                float val = result_data[i * 10 + j];
                EXPECT_GT(val, 0.0f);
                EXPECT_LT(val, 1.0f);
                row_sum += val;
            }
            EXPECT_NEAR(row_sum, 1.0f, tolerance);
        }
    } else {
        const double* result_data = result.data<double>();
        for (int64_t i = 0; i < 32; ++i) {
            double row_sum = 0.0;
            for (int64_t j = 0; j < 10; ++j) {
                double val = result_data[i * 10 + j];
                EXPECT_GT(val, 0.0);
                EXPECT_LT(val, 1.0);
                row_sum += val;
            }
            EXPECT_NEAR(row_sum, 1.0, tolerance);
        }
    }
}

// ==============================================================================
// Non-blocking Behavior Tests
// ==============================================================================

TEST_P(AsyncOpsMultiDTypeTest, AsyncOperationsNonBlocking) {
    auto a = randn({512, 512}, dtype, device);
    auto b = randn({512, 512}, dtype, device);

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
    auto a1 = randn({128, 128}, dtype, device);
    auto b1 = randn({128, 128}, dtype, device);
    auto a2 = randn({128, 128}, dtype, device);
    auto b2 = randn({128, 128}, dtype, device);
    auto a3 = randn({128, 128}, dtype, device);
    auto b3 = randn({128, 128}, dtype, device);

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
    auto a1 = randn({50, 50}, dtype, device);
    auto a2 = randn({50, 50}, dtype, device);
    auto a3 = randn({50, 50}, dtype, device);

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

std::vector<DTypeParam> GenerateAsyncOpsDTypes() {
    return {
        {DType::Float32, "float32", 1e-5},
        {DType::Float64, "float64", 1e-10},
    };
}

INSTANTIATE_TEST_SUITE_P(
    FloatDTypes,
    AsyncOpsMultiDTypeTest,
    ::testing::ValuesIn(GenerateAsyncOpsDTypes()),
    [](const ::testing::TestParamInfo<DTypeParam>& info) {
        return info.param.ToString();
    }
);

// ============================================================================
// Test Environment Setup
// ============================================================================

class AsyncOpsTestEnvironment : public ::testing::Environment {
public:
    void SetUp() override {
        initialize();
    }
};

static ::testing::Environment* const async_ops_env =
    ::testing::AddGlobalTestEnvironment(new AsyncOpsTestEnvironment);

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
