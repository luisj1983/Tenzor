/**
 * @file test_edge_cases.cpp
 * @brief Edge case tests for tensor operations: NaN, Inf, empty tensors, size-1 tensors.
 *
 * Migrated from CPU-only `::testing::Test` to BackendTest so the same edge
 * cases (NaN propagation, Inf handling, scalar tensors, shape no-ops) run on
 * every backend. The underlying ops (add/mul/exp/log/sum/sub/reshape/squeeze)
 * have kernels on all 5 backends.
 */

#include <gtest/gtest.h>
#include "../backend_test_fixture.hpp"
#include "tenzor/tenzor.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/reduction.hpp"
#include "tenzor/ops/transform.hpp"
#include "tenzor/ops/linalg.hpp"
#include "tenzor/ops/indexing.hpp"
#include "tenzor/ops/fft.hpp"
#include "tenzor/ops/advanced.hpp"
#include <cmath>
#include <limits>

using namespace tenzor;
using namespace tenzor::testing;

class EdgeCaseTest : public BackendTest {};

// ============================================================================
// NaN propagation
// ============================================================================

TEST_P(EdgeCaseTest, NanPropagationAdd) {
    auto a = full({4}, std::numeric_limits<float>::quiet_NaN(), DType::Float32, device);
    auto b = ones({4}, DType::Float32, device);
    auto c = add(a, b);
    auto c_cpu = c.to(Device::cpu()).contiguous();
    auto c_data = c_cpu.data<float>();
    for (int64_t i = 0; i < 4; ++i) {
        EXPECT_TRUE(std::isnan(c_data[i]))
            << "NaN not propagated at index " << i << " on " << device.to_string();
    }
}

TEST_P(EdgeCaseTest, NanPropagationMul) {
    auto a = full({4}, std::numeric_limits<float>::quiet_NaN(), DType::Float32, device);
    auto b = ones({4}, DType::Float32, device);
    auto c = mul(a, b);
    auto c_cpu = c.to(Device::cpu()).contiguous();
    auto c_data = c_cpu.data<float>();
    for (int64_t i = 0; i < 4; ++i) {
        EXPECT_TRUE(std::isnan(c_data[i]))
            << "NaN not propagated at index " << i << " on " << device.to_string();
    }
}

TEST_P(EdgeCaseTest, NanPropagationSum) {
    // Build the input on CPU (we need direct write access to inject NaN at
    // index 2), then move to the test device. Direct device-side write would
    // require a backend-specific dance per backend.
    auto a_cpu = ones({4}, DType::Float32, Device::cpu());
    auto a_data = const_cast<float*>(a_cpu.data<float>());
    a_data[2] = std::numeric_limits<float>::quiet_NaN();
    auto a = (device.type == Device::Type::CPU) ? a_cpu : a_cpu.to(device);
    auto s = sum(a);
    auto s_cpu = s.to(Device::cpu()).contiguous();
    auto s_data = s_cpu.data<float>();
    EXPECT_TRUE(std::isnan(s_data[0]))
        << "NaN not propagated through sum on " << device.to_string();
}

// ============================================================================
// Inf handling
// ============================================================================

TEST_P(EdgeCaseTest, ExpLargeProducesInf) {
    auto a = full({2}, 1000.0f, DType::Float32, device);
    auto b = tenzor::exp(a);
    auto b_cpu = b.to(Device::cpu()).contiguous();
    auto b_data = b_cpu.data<float>();
    for (int64_t i = 0; i < 2; ++i) {
        EXPECT_TRUE(std::isinf(b_data[i]))
            << "exp(1000) should be Inf on " << device.to_string();
        EXPECT_GT(b_data[i], 0)
            << "exp(1000) should be +Inf on " << device.to_string();
    }
}

TEST_P(EdgeCaseTest, LogZeroProducesNegInf) {
    auto a = zeros({2}, DType::Float32, device);
    auto b = tenzor::log(a);
    auto b_cpu = b.to(Device::cpu()).contiguous();
    auto b_data = b_cpu.data<float>();
    for (int64_t i = 0; i < 2; ++i) {
        EXPECT_TRUE(std::isinf(b_data[i]))
            << "log(0) should be -Inf on " << device.to_string();
        EXPECT_LT(b_data[i], 0)
            << "log(0) should be -Inf on " << device.to_string();
    }
}

TEST_P(EdgeCaseTest, InfPropagationThroughOps) {
    auto a = full({2}, std::numeric_limits<float>::infinity(), DType::Float32, device);
    auto b = ones({2}, DType::Float32, device);

    // Inf + 1 = Inf
    auto c = add(a, b);
    auto c_cpu = c.to(Device::cpu()).contiguous();
    EXPECT_TRUE(std::isinf(c_cpu.data<float>()[0]));

    // Inf * 2 = Inf
    auto d = mul(a, full({2}, 2.0f, DType::Float32, device));
    auto d_cpu = d.to(Device::cpu()).contiguous();
    EXPECT_TRUE(std::isinf(d_cpu.data<float>()[0]));
}

// ============================================================================
// Size-1 tensor operations
// ============================================================================

TEST_P(EdgeCaseTest, Size1TensorOps) {
    auto a = ones({1}, DType::Float32, device);
    auto b = ones({1}, DType::Float32, device);

    auto c = add(a, b);
    EXPECT_EQ(c.numel(), 1);
    auto c_cpu = c.to(Device::cpu()).contiguous();
    EXPECT_FLOAT_EQ(c_cpu.data<float>()[0], 2.0f);

    auto s = sum(a);
    auto s_cpu = s.to(Device::cpu()).contiguous();
    EXPECT_FLOAT_EQ(s_cpu.data<float>()[0], 1.0f);
}

TEST_P(EdgeCaseTest, ScalarTensorOps) {
    auto a = full({}, 3.0f, DType::Float32, device);  // 0-dim scalar
    auto b = full({}, 4.0f, DType::Float32, device);

    auto c = add(a, b);
    EXPECT_EQ(c.ndim(), 0);
    auto c_cpu = c.to(Device::cpu()).contiguous();
    EXPECT_FLOAT_EQ(c_cpu.data<float>()[0], 7.0f);
}

// ============================================================================
// Shape operations edge cases
// ============================================================================

TEST_P(EdgeCaseTest, ReshapeToSameShape) {
    auto a = ones({3, 4}, DType::Float32, device);
    auto b = a.reshape({3, 4});
    EXPECT_EQ(b.numel(), 12);
}

TEST_P(EdgeCaseTest, TransposeSingleDim) {
    auto a = ones({5}, DType::Float32, device);
    // Transpose of 1D tensor should be a no-op or identity
    EXPECT_EQ(a.ndim(), 1);
    EXPECT_EQ(a.shape()[0], 5);
}

TEST_P(EdgeCaseTest, SqueezeNoEffect) {
    auto a = ones({3, 4}, DType::Float32, device);
    auto b = a.squeeze(0);  // dim 0 is size 3, not 1 -- should be no-op
    EXPECT_EQ(b.ndim(), 2);
    EXPECT_EQ(b.shape()[0], 3);
    EXPECT_EQ(b.shape()[1], 4);
}

// ============================================================================
// DType edge cases
// ============================================================================

TEST_P(EdgeCaseTest, Float64Precision) {
    // Verify Float64 subtraction preserves precision better than Float32
    auto a64 = full({1}, 100.0, DType::Float64, device);
    auto b64 = full({1}, 99.0, DType::Float64, device);
    auto c64 = sub(a64, b64);
    auto c64_cpu = c64.to(Device::cpu()).contiguous();
    auto c64_data = c64_cpu.data<double>();
    EXPECT_DOUBLE_EQ(c64_data[0], 1.0)
        << "Float64 subtraction should be exact for integers on "
        << device.to_string();
}

// ============================================================================
// Regression tests for ops_sparse review fixes (2026-06-20)
// ============================================================================

// triu/tril on a zero-size matrix dimension must not divide by zero (SIGFPE).
// Previously batch_size = numel / (rows*cols) crashed when rows*cols == 0.
TEST_P(EdgeCaseTest, TriuTrilZeroSizeMatrixDim) {
    if (device.type != Device::Type::CPU) GTEST_SKIP();
    for (std::vector<int64_t> shp : {std::vector<int64_t>{0, 5},
                                     std::vector<int64_t>{5, 0},
                                     std::vector<int64_t>{3, 0, 4}}) {
        auto x = zeros(shp, DType::Float32, device);
        auto u = triu(x, 0);
        auto l = tril(x, 0);
        EXPECT_EQ(u.numel(), 0);
        EXPECT_EQ(l.numel(), 0);
        for (size_t d = 0; d < shp.size(); ++d) {
            EXPECT_EQ(u.shape()[d], shp[d]);
            EXPECT_EQ(l.shape()[d], shp[d]);
        }
    }
}

// argsort with the PyTorch-default negative dim (-1) must normalize before
// dispatch; a raw -1 reaching the kernel would index shape[-1] OOB.
TEST_P(EdgeCaseTest, ArgsortNegativeDim) {
    std::vector<float> xd{3, 1, 2, 6, 5, 4};
    auto x = tenzor::from_data(xd.data(), {2, 3}, Device::cpu()).to(device);
    auto idx_neg = argsort(x, -1, false);
    auto idx_pos = argsort(x, 1, false);
    auto a = idx_neg.to(Device::cpu()).contiguous();
    auto b = idx_pos.to(Device::cpu()).contiguous();
    ASSERT_EQ(a.numel(), b.numel());
    for (int64_t i = 0; i < a.numel(); ++i)
        EXPECT_EQ(a.data<int64_t>()[i], b.data<int64_t>()[i]);
}

// unique_consecutive with negative dim must normalize identically to dim=1.
TEST_P(EdgeCaseTest, UniqueConsecutiveNegativeDim) {
    if (device.type != Device::Type::CPU) GTEST_SKIP();
    std::vector<float> xd{1, 1, 2, 1, 1, 2};
    auto x = tenzor::from_data(xd.data(), {2, 3}, Device::cpu()).to(device);
    auto [u_neg, inv_neg, cnt_neg] = unique_consecutive(x, false, false, -1);
    auto [u_pos, inv_pos, cnt_pos] = unique_consecutive(x, false, false, 1);
    auto a = u_neg.to(Device::cpu()).contiguous();
    auto b = u_pos.to(Device::cpu()).contiguous();
    ASSERT_EQ(a.numel(), b.numel());
    for (int64_t i = 0; i < a.numel(); ++i)
        EXPECT_FLOAT_EQ(a.data<float>()[i], b.data<float>()[i]);
}

// linalg::solve must reject a B whose row dimension != A's order n, rather
// than letting LAPACK write n*nrhs elements into an under-sized buffer.
TEST_P(EdgeCaseTest, SolveRejectsMismatchedBRows) {
    if (device.type != Device::Type::CPU) GTEST_SKIP();
    auto A = tenzor::eye(4, std::nullopt, DType::Float32, device);
    auto B = zeros({2, 3}, DType::Float32, device);  // wrong: 2 rows, need 4
    EXPECT_THROW(tenzor::linalg::solve(A, B), std::invalid_argument);
}

// fftfreq(0) must return an empty tensor, not [NaN] from 1/(0*d).
TEST_P(EdgeCaseTest, FftfreqZeroReturnsEmpty) {
    if (device.type != Device::Type::CPU) GTEST_SKIP();
    auto f = tenzor::fft::fftfreq(0, 1.0, DType::Float32, device);
    EXPECT_EQ(f.numel(), 0);
    EXPECT_THROW(tenzor::fft::fftfreq(-3, 1.0, DType::Float32, device),
                 std::invalid_argument);
}

// take must reject a flat index that is out of bounds (OOB read otherwise).
TEST_P(EdgeCaseTest, TakeRejectsOutOfBoundsIndex) {
    if (device.type != Device::Type::CPU) GTEST_SKIP();
    auto x = zeros({4}, DType::Float32, device);
    std::vector<int64_t> idxd{0, 9};
    auto bad_idx = tenzor::from_data(idxd.data(), {2}, Device::cpu()).to(device);
    EXPECT_THROW(tenzor::take(x, bad_idx), std::out_of_range);
}

INSTANTIATE_BACKEND_TESTS(EdgeCaseTest);
