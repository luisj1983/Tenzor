/**
 * @file test_statistics_multidtype.cpp
 * @brief Multi-backend + multi-dtype tests for cov() and corrcoef() operations
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/ops/math.hpp>
#include <tenzor/ops/creation.hpp>
#include <tenzor/ops/reduction.hpp>
#include <tenzor/ops/transform.hpp>
#include "multi_backend_dtype_fixture.hpp"

using namespace tenzor;
using namespace tenzor::testing;

class StatisticsMultiDTypeTest : public MultiBackendDTypeTest {
protected:
    /**
     * @brief Create a tensor from a flat vector with given shape on the test device/dtype.
     */
    Tensor fromValues(const std::vector<float>& vals, const std::vector<int64_t>& shape) {
        auto t = tenzor::full(shape, 0.0f, DType::Float32, Device::cpu());
        auto* d = t.data<float>();
        for (size_t i = 0; i < vals.size(); ++i) d[i] = vals[i];
        return t.to(device()).to(dtype());
    }
};

// ---------------------------------------------------------------------------
// cov() tests
// ---------------------------------------------------------------------------

TEST_P(StatisticsMultiDTypeTest, CovBasicShape) {
    // 3 variables, 5 observations -> (3, 3) covariance matrix
    auto x = createRandn({3, 5});
    auto result = tenzor::cov(x);
    expectShape(result, {3, 3});
    expectDevice(result);
}

TEST_P(StatisticsMultiDTypeTest, Cov1DInput) {
    // 1D input of length 5 -> (1, 1) covariance matrix
    auto x = createRandn({5});
    auto result = tenzor::cov(x);
    expectShape(result, {1, 1});
    expectDevice(result);
}

TEST_P(StatisticsMultiDTypeTest, CovSymmetric) {
    // Covariance matrix must be symmetric: C[i,j] == C[j,i]
    auto x = createRandn({3, 10});
    auto C = tenzor::cov(x);
    auto Ct = tenzor::transpose(C, 0, 1);
    // Check that C and C^T are close
    auto close = tenzor::isclose(C.to(Device::cpu()).to(DType::Float32),
                                 Ct.to(Device::cpu()).to(DType::Float32),
                                 1e-4, 1e-5);
    auto cpu_close = close.to(Device::cpu());
    auto* data = cpu_close.data<bool>();
    int64_t n = cpu_close.numel();
    for (int64_t i = 0; i < n; ++i) {
        EXPECT_TRUE(data[i]) << "Covariance matrix not symmetric at element " << i;
    }
}

TEST_P(StatisticsMultiDTypeTest, CovPerfectCorrelation) {
    // Two identical variables should have equal variance and covariance
    // x = [[1, 2, 3, 4, 5],
    //      [1, 2, 3, 4, 5]]
    auto x = fromValues({1, 2, 3, 4, 5,
                         1, 2, 3, 4, 5}, {2, 5});
    auto C = tenzor::cov(x).to(Device::cpu()).to(DType::Float32);
    auto* d = C.data<float>();
    // All four entries should be equal (same variance)
    float expected = 2.5f;  // var([1,2,3,4,5]) = 10/4 = 2.5
    // reason: median/quantile/cov reduction noise — F16/BF16 widen-narrow
    // path accumulates relative error proportional to M
    EXPECT_NEAR(d[0], expected, 0.1f);
    EXPECT_NEAR(d[1], expected, 0.1f);
    EXPECT_NEAR(d[2], expected, 0.1f);
    EXPECT_NEAR(d[3], expected, 0.1f);
}

TEST_P(StatisticsMultiDTypeTest, CovZeroCorrection) {
    // With correction=0, divide by M instead of M-1
    auto x = fromValues({1, 2, 3, 4}, {1, 4});
    auto C_biased = tenzor::cov(x, /*correction=*/0).to(Device::cpu()).to(DType::Float32);
    auto C_unbiased = tenzor::cov(x, /*correction=*/1).to(Device::cpu()).to(DType::Float32);
    float biased = *C_biased.data<float>();
    float unbiased = *C_unbiased.data<float>();
    // unbiased = biased * M / (M - 1) => biased = unbiased * 3/4
    // reason: median/quantile/cov reduction noise (Bessel correction factor)
    EXPECT_NEAR(biased, unbiased * 3.0f / 4.0f, 0.1f);
}

// ---------------------------------------------------------------------------
// corrcoef() tests
// ---------------------------------------------------------------------------

TEST_P(StatisticsMultiDTypeTest, CorrcoefBasicShape) {
    // 3 variables, 10 observations -> (3, 3)
    auto x = createRandn({3, 10});
    auto result = tenzor::corrcoef(x);
    expectShape(result, {3, 3});
    expectDevice(result);
}

TEST_P(StatisticsMultiDTypeTest, CorrcoefDiagonalOnes) {
    // Diagonal of corrcoef should be 1.0 (each variable perfectly correlated with itself)
    auto x = createRandn({3, 20});
    auto R = tenzor::corrcoef(x).to(Device::cpu()).to(DType::Float32);
    auto d_tensor = tenzor::diag(R).to(Device::cpu()).to(DType::Float32);
    auto* d = d_tensor.data<float>();
    for (int64_t i = 0; i < 3; ++i) {
        EXPECT_NEAR(d[i], 1.0f, 1e-3f) << "Diagonal element " << i << " not 1.0";
    }
}

TEST_P(StatisticsMultiDTypeTest, CorrcoefSymmetric) {
    // Correlation matrix must be symmetric
    auto x = createRandn({4, 15});
    auto R = tenzor::corrcoef(x);
    auto Rt = tenzor::transpose(R, 0, 1);
    auto close = tenzor::isclose(R.to(Device::cpu()).to(DType::Float32),
                                 Rt.to(Device::cpu()).to(DType::Float32),
                                 1e-4, 1e-5);
    auto cpu_close = close.to(Device::cpu());
    auto* data = cpu_close.data<bool>();
    int64_t n = cpu_close.numel();
    for (int64_t i = 0; i < n; ++i) {
        EXPECT_TRUE(data[i]) << "Correlation matrix not symmetric at element " << i;
    }
}

TEST_P(StatisticsMultiDTypeTest, CorrcoefPerfectCorrelation) {
    // Two identical variables -> corrcoef = [[1, 1], [1, 1]]
    auto x = fromValues({1, 2, 3, 4, 5,
                         1, 2, 3, 4, 5}, {2, 5});
    auto R = tenzor::corrcoef(x).to(Device::cpu()).to(DType::Float32);
    auto* d = R.data<float>();
    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(d[i], 1.0f, 1e-3f) << "Element " << i << " not 1.0 for perfect correlation";
    }
}

TEST_P(StatisticsMultiDTypeTest, CorrcoefBoundedRange) {
    // All correlation coefficients should be in [-1, 1]
    auto x = createRandn({5, 20});
    auto R = tenzor::corrcoef(x).to(Device::cpu()).to(DType::Float32);
    auto* d = R.data<float>();
    int64_t n = R.numel();
    for (int64_t i = 0; i < n; ++i) {
        EXPECT_GE(d[i], -1.0f - 1e-5f) << "Element " << i << " below -1";
        EXPECT_LE(d[i],  1.0f + 1e-5f) << "Element " << i << " above 1";
    }
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(StatisticsMultiDTypeTest);
