/**
 * @file test_new_ops.cpp
 * @brief Tests for newly added operations: sampling (NormalSample, ExponentialSample),
 *        math (ndtr, log_ndtr, multigammaln, trapezoid, gradient, pairwise_distance, pdist),
 *        linalg (cholesky_inverse, as_strided), and segment_reduce.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/ops/math.hpp>
#include <tenzor/ops/creation.hpp>
#include <tenzor/ops/reduction.hpp>
#include <tenzor/ops/transform.hpp>
#include <tenzor/ops/linalg.hpp>
#include "../multi_backend_dtype_fixture.hpp"
#include <cmath>
#include <numeric>

using namespace tenzor;
using namespace tenzor::testing;

// ============================================================================
// Fixture
// ============================================================================

class NewOpsTest : public MultiBackendDTypeTest {
protected:
    Tensor fromVec(const std::vector<float>& vals, std::vector<int64_t> shape = {}) {
        if (shape.empty()) shape = {static_cast<int64_t>(vals.size())};
        auto t = tenzor::full(shape, 0.0f, DType::Float32, Device::cpu());
        float* d = t.data<float>();
        for (size_t i = 0; i < vals.size(); ++i) d[i] = vals[i];
        return t.to(device());
    }
};

// ============================================================================
// NormalSample Tests
// ============================================================================

TEST_P(NewOpsTest, NormalSampleShape) {
    auto mean = tenzor::zeros({1000}, DType::Float32, device());
    auto stddev = tenzor::ones({1000}, DType::Float32, device());
    auto result = tenzor::normal(mean, stddev);

    EXPECT_EQ(result.shape()[0], 1000);
    EXPECT_EQ(result.dtype(), DType::Float32);
    expectDevice(result);
}

TEST_P(NewOpsTest, NormalSampleStatistics) {
    auto mean = tenzor::full({100000}, 5.0f, DType::Float32, device());
    auto stddev = tenzor::full({100000}, 2.0f, DType::Float32, device());
    auto result = tenzor::normal(mean, stddev);

    auto result_cpu = result.to(Device::cpu());
    auto mean_val = tenzor::mean(result_cpu);
    float m = mean_val.to(DType::Float32).to(Device::cpu()).data<float>()[0];
    EXPECT_NEAR(m, 5.0f, 0.1f) << "Mean should be approximately 5.0";
}

// ============================================================================
// ExponentialSample Tests
// ============================================================================

TEST_P(NewOpsTest, ExponentialSampleShape) {
    auto rate = tenzor::ones({1000}, DType::Float32, device());
    auto result = tenzor::exponential(rate);

    EXPECT_EQ(result.shape()[0], 1000);
    EXPECT_EQ(result.dtype(), DType::Float32);
    expectDevice(result);
}

TEST_P(NewOpsTest, ExponentialSamplePositive) {
    auto rate = tenzor::full({10000}, 1.0f, DType::Float32, device());
    auto result = tenzor::exponential(rate);

    // All samples should be positive
    auto result_cpu = result.to(Device::cpu()).to(DType::Float32);
    const float* d = result_cpu.data<float>();
    for (int64_t i = 0; i < result_cpu.numel(); ++i) {
        EXPECT_GT(d[i], 0.0f) << "Exponential samples must be positive";
    }
}

TEST_P(NewOpsTest, ExponentialSampleMean) {
    auto rate = tenzor::full({100000}, 2.0f, DType::Float32, device());
    auto result = tenzor::exponential(rate);

    auto mean_val = tenzor::mean(result.to(Device::cpu()));
    float m = mean_val.to(DType::Float32).to(Device::cpu()).data<float>()[0];
    // E[Exp(rate)] = 1/rate = 0.5
    EXPECT_NEAR(m, 0.5f, 0.05f) << "Mean should be approximately 1/rate = 0.5";
}

// ============================================================================
// Ndtr (Normal CDF) Tests
// ============================================================================

TEST_P(NewOpsTest, NdtrBasicValues) {
    auto input = fromVec({0.0f, -1e10f, 1e10f});
    auto result = tenzor::ndtr(input).to(Device::cpu()).to(DType::Float32);
    const float* d = result.data<float>();

    EXPECT_NEAR(d[0], 0.5f, 1e-5f) << "ndtr(0) = 0.5";
    EXPECT_NEAR(d[1], 0.0f, 1e-5f) << "ndtr(-inf) ~ 0";
    EXPECT_NEAR(d[2], 1.0f, 1e-5f) << "ndtr(+inf) ~ 1";
}

// ============================================================================
// Trapezoid Integration Tests
// ============================================================================

TEST_P(NewOpsTest, TrapezoidUniform) {
    // Integrate y = [1, 2, 3] with dx=1 => 0.5*(1+2)*1 + 0.5*(2+3)*1 = 4.0
    auto y = fromVec({1.0f, 2.0f, 3.0f});
    auto result = tenzor::trapezoid(y, 1.0, 0);
    auto r = result.to(Device::cpu()).to(DType::Float32);
    EXPECT_NEAR(r.data<float>()[0], 4.0f, 1e-4f);
}

// ============================================================================
// Gradient Tests
// ============================================================================

TEST_P(NewOpsTest, GradientLinear) {
    // y = [0, 1, 2, 3, 4] with spacing=1 => gradient should be all 1s
    auto y = fromVec({0.0f, 1.0f, 2.0f, 3.0f, 4.0f});
    auto result = tenzor::gradient(y, 0, 1.0);
    auto r = result.to(Device::cpu()).to(DType::Float32);
    const float* d = r.data<float>();

    for (int64_t i = 0; i < 5; ++i) {
        EXPECT_NEAR(d[i], 1.0f, 1e-4f) << "Linear gradient should be 1 everywhere";
    }
}

// ============================================================================
// PairwiseDistance Tests
// ============================================================================

TEST_P(NewOpsTest, PairwiseDistanceL2) {
    auto x1 = fromVec({1.0f, 0.0f, 0.0f, 3.0f}, {2, 2});
    auto x2 = fromVec({0.0f, 0.0f, 0.0f, 4.0f}, {2, 2});
    auto result = tenzor::pairwise_distance(x1, x2, 2.0);
    auto r = result.to(Device::cpu()).to(DType::Float32);
    const float* d = r.data<float>();

    EXPECT_NEAR(d[0], 1.0f, 1e-4f) << "||[1,0] - [0,0]||_2 = 1";
    EXPECT_NEAR(d[1], 1.0f, 1e-4f) << "||[0,3] - [0,4]||_2 = 1";
}

// ============================================================================
// SegmentReduce Tests
// ============================================================================

TEST_P(NewOpsTest, SegmentReduceSum) {
    // data = [1, 2, 3, 4, 5], offsets = [0, 2, 5]
    // segment 0: [1, 2] => sum=3, segment 1: [3, 4, 5] => sum=12
    auto data = fromVec({1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
    Tensor offsets = tenzor::full({3}, 0.0f, DType::Int64, Device::cpu());
    int64_t* off = offsets.data<int64_t>();
    off[0] = 0; off[1] = 2; off[2] = 5;
    offsets = offsets.to(device());

    auto result = tenzor::segment_reduce(data, offsets, "sum");
    Tensor r = result.to(Device::cpu()).to(DType::Float32);
    float* d = r.data<float>();

    EXPECT_NEAR(d[0], 3.0f, 1e-4f);
    EXPECT_NEAR(d[1], 12.0f, 1e-4f);
}

// ============================================================================
// AsStrided Tests
// ============================================================================

TEST_P(NewOpsTest, AsStridedBasic) {
    auto t = fromVec({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    std::vector<int64_t> size = {2, 3};
    std::vector<int64_t> stride = {3, 1};
    auto result = tenzor::as_strided(t, size, stride);

    EXPECT_EQ(result.shape()[0], 2);
    EXPECT_EQ(result.shape()[1], 3);

    auto r = result.to(DType::Float32).to(Device::cpu());
    const float* d = r.data<float>();
    EXPECT_NEAR(d[0], 1.0f, 1e-6f);
    EXPECT_NEAR(d[3], 4.0f, 1e-6f);
}

// ============================================================================
// Multigammaln Tests
// ============================================================================

TEST_P(NewOpsTest, MultigammalnP1) {
    // multigammaln(a, p=1) = lgamma(a)
    auto a = fromVec({2.0f, 3.0f, 4.0f});
    auto result = tenzor::multigammaln(a, 1);
    auto expected = tenzor::lgamma(a);

    auto r = result.to(Device::cpu()).to(DType::Float32);
    auto e = expected.to(Device::cpu()).to(DType::Float32);
    const float* rd = r.data<float>();
    const float* ed = e.data<float>();

    for (int64_t i = 0; i < 3; ++i) {
        EXPECT_NEAR(rd[i], ed[i], 1e-3f) << "multigammaln(a, 1) should equal lgamma(a)";
    }
}

// ============================================================================
// Instantiation
// ============================================================================

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(NewOpsTest);
