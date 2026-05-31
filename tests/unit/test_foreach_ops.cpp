/**
 * @file test_foreach_ops.cpp
 * @brief Unit tests for foreach (multi-tensor optimizer) ops
 *
 * Covers each op family: binary out-of-place and in-place (add/sub/mul/div),
 * unary out-of-place and in-place (neg/abs/sqrt/zero), copy, ternary fused
 * (addcdiv/addcmul/lerp), reduction (norm), and a perf smoke test.
 */

#include <gtest/gtest.h>
#include <chrono>
#include <vector>
#include <cmath>

#include "tenzor/tenzor.hpp"
#include "tenzor/core/tensor.hpp"
#include "tenzor/core/dtype.hpp"
#include "tenzor/ops/foreach.hpp"
#include "tenzor/ops/creation.hpp"
#include "../backend_test_fixture.hpp"

namespace tz = ::tenzor;

class ForeachOps : public ::tenzor::testing::BackendTest {};

namespace {

// Make a list of n tensors of shape {4, 4} all filled with `value`, on `device`.
// Every tensor placed into a foreach list must live on the same device as the
// rest of the list, so the creation call is routed onto `device`.
std::vector<tz::Tensor> make_list(int n, double value, tz::Device device,
                                   tz::DType dt = tz::DType::Float32) {
    std::vector<tz::Tensor> out;
    out.reserve(n);
    for (int i = 0; i < n; ++i) {
        out.push_back(tz::full({4, 4}, value, dt, device));
    }
    return out;
}

// Verify every element of every tensor in the list equals expected.
void check_list_scalar(const std::vector<tz::Tensor>& list,
                        float expected, float tol = 1e-5f) {
    for (const auto& t : list) {
        const auto* p = t.cpu().data<float>();
        const int64_t numel = 4 * 4;
        for (int64_t i = 0; i < numel; ++i) {
            EXPECT_NEAR(p[i], expected, tol);
        }
    }
}

} // namespace

// =============================================================================
// Out-of-place binary ops
// =============================================================================

TEST_P(ForeachOps, Add) {
    auto a = make_list(8, 1.0, device);
    auto b = make_list(8, 2.0, device);
    auto c = tz::foreach_add(a, b);
    ASSERT_EQ(c.size(), 8u);
    check_list_scalar(c, 3.0f);
}

TEST_P(ForeachOps, Sub) {
    auto a = make_list(8, 5.0, device);
    auto b = make_list(8, 2.0, device);
    auto c = tz::foreach_sub(a, b);
    ASSERT_EQ(c.size(), 8u);
    check_list_scalar(c, 3.0f);
}

TEST_P(ForeachOps, Mul) {
    auto a = make_list(8, 3.0, device);
    auto b = make_list(8, 2.0, device);
    auto c = tz::foreach_mul(a, b);
    ASSERT_EQ(c.size(), 8u);
    check_list_scalar(c, 6.0f);
}

TEST_P(ForeachOps, Div) {
    auto a = make_list(8, 6.0, device);
    auto b = make_list(8, 3.0, device);
    auto c = tz::foreach_div(a, b);
    ASSERT_EQ(c.size(), 8u);
    check_list_scalar(c, 2.0f);
}

// =============================================================================
// Out-of-place unary ops
// =============================================================================

TEST_P(ForeachOps, Neg) {
    auto a = make_list(8, 3.0, device);
    auto b = tz::foreach_neg(a);
    ASSERT_EQ(b.size(), 8u);
    check_list_scalar(b, -3.0f);
    // a must be unmodified (out-of-place)
    check_list_scalar(a, 3.0f);
}

TEST_P(ForeachOps, Abs) {
    auto a = make_list(8, -4.0, device);
    auto b = tz::foreach_abs(a);
    ASSERT_EQ(b.size(), 8u);
    check_list_scalar(b, 4.0f);
    // a must be unmodified
    check_list_scalar(a, -4.0f);
}

TEST_P(ForeachOps, Sqrt) {
    auto a = make_list(8, 4.0, device);
    auto b = tz::foreach_sqrt(a);
    ASSERT_EQ(b.size(), 8u);
    check_list_scalar(b, 2.0f);
    // a must be unmodified
    check_list_scalar(a, 4.0f);
}

TEST_P(ForeachOps, Copy) {
    auto a = make_list(6, 7.0, device);
    auto b = tz::foreach_copy(a);
    ASSERT_EQ(b.size(), 6u);
    check_list_scalar(b, 7.0f);
    // Verify independence: modifying original list shouldn't affect copy.
    // (The tensors inside a are separate objects; copy must use clone.)
    check_list_scalar(a, 7.0f);
}

// =============================================================================
// In-place binary ops
// =============================================================================

TEST_P(ForeachOps, AddInplace) {
    auto a = make_list(8, 1.0, device);
    auto b = make_list(8, 2.0, device);
    tz::foreach_add_(a, b);
    check_list_scalar(a, 3.0f);
}

TEST_P(ForeachOps, SubInplace) {
    auto a = make_list(8, 5.0, device);
    auto b = make_list(8, 2.0, device);
    tz::foreach_sub_(a, b);
    check_list_scalar(a, 3.0f);
}

TEST_P(ForeachOps, MulInplace) {
    auto a = make_list(8, 3.0, device);
    auto b = make_list(8, 2.0, device);
    tz::foreach_mul_(a, b);
    check_list_scalar(a, 6.0f);
}

TEST_P(ForeachOps, DivInplace) {
    auto a = make_list(8, 6.0, device);
    auto b = make_list(8, 3.0, device);
    tz::foreach_div_(a, b);
    check_list_scalar(a, 2.0f);
}

// =============================================================================
// In-place unary ops
// =============================================================================

TEST_P(ForeachOps, NegInplace) {
    auto a = make_list(8, 3.0, device);
    tz::foreach_neg_(a);
    check_list_scalar(a, -3.0f);
}

TEST_P(ForeachOps, AbsInplace) {
    auto a = make_list(8, -4.0, device);
    tz::foreach_abs_(a);
    check_list_scalar(a, 4.0f);
}

TEST_P(ForeachOps, SqrtInplace) {
    auto a = make_list(8, 4.0, device);
    tz::foreach_sqrt_(a);
    check_list_scalar(a, 2.0f);
}

TEST_P(ForeachOps, Zero) {
    auto a = make_list(8, 7.0, device);
    tz::foreach_zero_(a);
    check_list_scalar(a, 0.0f);
}

// =============================================================================
// Ternary fused ops
// =============================================================================

TEST_P(ForeachOps, Addcdiv) {
    // self[i] += scalar * a[i] / b[i]
    // 1.0 + 0.5 * (6.0 / 3.0) = 1.0 + 1.0 = 2.0
    auto self_v = make_list(8, 1.0, device);
    auto a      = make_list(8, 6.0, device);
    auto b      = make_list(8, 3.0, device);
    tz::foreach_addcdiv_(self_v, a, b, 0.5);
    check_list_scalar(self_v, 2.0f, 1e-4f);
}

TEST_P(ForeachOps, Addcmul) {
    // self[i] += scalar * a[i] * b[i]
    // 1.0 + 0.5 * 2.0 * 3.0 = 1.0 + 3.0 = 4.0
    auto self_v = make_list(8, 1.0, device);
    auto a      = make_list(8, 2.0, device);
    auto b      = make_list(8, 3.0, device);
    tz::foreach_addcmul_(self_v, a, b, 0.5);
    check_list_scalar(self_v, 4.0f, 1e-4f);
}

TEST_P(ForeachOps, Lerp) {
    // self[i] = self[i] + scalar*(b[i] - self[i]) = (1-scalar)*self + scalar*b
    // = 0.5 * 0.0 + 0.5 * 1.0 = 0.5
    auto self_v = make_list(8, 0.0, device);
    auto b      = make_list(8, 1.0, device);
    tz::foreach_lerp_(self_v, b, 0.5);
    check_list_scalar(self_v, 0.5f, 1e-4f);
}

// =============================================================================
// Reduction
// =============================================================================

TEST_P(ForeachOps, Norm) {
    // 4x4 tensor of 3.0 → L2 norm = sqrt(16 * 9) = sqrt(144) = 12.0
    auto a = make_list(4, 3.0, device);
    auto norms = tz::foreach_norm(a, 2.0);
    ASSERT_EQ(norms.size(), 4u);
    for (const auto& n : norms) {
        EXPECT_NEAR(n.cpu().template item<float>(), 12.0f, 1e-3f);
    }
}

TEST_P(ForeachOps, NormL1) {
    // L1 norm of 4x4 tensor of 2.0 = 16 * 2.0 = 32.0
    auto a = make_list(4, 2.0, device);
    auto norms = tz::foreach_norm(a, 1.0);
    ASSERT_EQ(norms.size(), 4u);
    for (const auto& n : norms) {
        EXPECT_NEAR(n.cpu().template item<float>(), 32.0f, 1e-3f);
    }
}

// =============================================================================
// Error handling
// =============================================================================

TEST_P(ForeachOps, MismatchedSizesThrows) {
    auto a = make_list(4, 1.0, device);
    auto b = make_list(5, 2.0, device);
    EXPECT_THROW(tz::foreach_add(a, b), std::invalid_argument);
    EXPECT_THROW(tz::foreach_add_(a, b), std::invalid_argument);
}

// =============================================================================
// Single-element lists (edge case)
// =============================================================================

TEST_P(ForeachOps, SingleTensorList) {
    auto a = make_list(1, 3.0, device);
    auto b = make_list(1, 4.0, device);
    auto c = tz::foreach_add(a, b);
    ASSERT_EQ(c.size(), 1u);
    EXPECT_NEAR(c[0].cpu().data<float>()[0], 7.0f, 1e-5f);
}

// =============================================================================
// Float64 dtype
// =============================================================================

TEST_P(ForeachOps, Float64Add) {
    std::vector<tz::Tensor> a, b;
    for (int i = 0; i < 4; ++i) {
        a.push_back(tz::full({4, 4}, 1.0, tz::DType::Float64, device));
        b.push_back(tz::full({4, 4}, 2.0, tz::DType::Float64, device));
    }
    auto c = tz::foreach_add(a, b);
    ASSERT_EQ(c.size(), 4u);
    for (const auto& t : c) {
        const auto* p = t.cpu().data<double>();
        for (int i = 0; i < 16; ++i) EXPECT_NEAR(p[i], 3.0, 1e-12);
    }
}

// =============================================================================
// Perf smoke test: 1000 small tensors, assert < 200ms wall time
// =============================================================================

TEST_P(ForeachOps, PerfManyTensors) {
    auto a = make_list(1000, 1.0, device);
    auto b = make_list(1000, 2.0, device);
    auto t0 = std::chrono::high_resolution_clock::now();
    auto c = tz::foreach_add(a, b);
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    EXPECT_LT(ms, 200.0) << "foreach_add over 1000 tensors took " << ms << " ms";
    // Sanity check result
    ASSERT_EQ(c.size(), 1000u);
    EXPECT_NEAR(c[0].cpu().data<float>()[0], 3.0f, 1e-5f);
}

// Fan every TEST_P above over all five backends. BackendTest::SetUp skips a
// backend that is physically absent on the host; a present backend that does
// not implement a given foreach op throws → the corresponding cell FAILS.
INSTANTIATE_BACKEND_TESTS(ForeachOps);
