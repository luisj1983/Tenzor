/**
 * @file test_reduction_half_dtype.cpp
 * @brief Reduction half-dtype contract — audit items E.1 + E.2.
 *
 * Two reduction kernels previously violated their dtype contracts on
 * Float16 / BFloat16 inputs:
 *
 *  - `norm(dim=...)` hard-threw for half dtypes ("only Float32/Float64
 *    supported"), making `.norm(dim=...)` unusable on Float16 / BFloat16
 *    tensors.  Fixed to widen-narrow per the existing pattern.
 *
 *  - `prod` widened half-dtype inputs to Float32, computed the product,
 *    and stored the output as Float32 — silently changing the output
 *    dtype away from the input dtype.  Fixed to narrow back to the
 *    input dtype at the end.
 */

#include <gtest/gtest.h>

#include <cmath>

#include "../backend_test_fixture.hpp"
#include "tenzor/core/device.hpp"
#include "tenzor/core/dtype.hpp"
#include "tenzor/core/tensor.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/reduction.hpp"
#include "tenzor/tenzor.hpp"

using namespace tenzor;

namespace {

class ReductionHalfDtypeTest : public ::tenzor::testing::BackendTest {
protected:
    void SetUp() override {
        ::tenzor::testing::BackendTest::SetUp();
        if (::testing::Test::IsSkipped()) return;
    }
};

// ---------------------------------------------------------------------------
// E.1 — norm(dim=...) must accept Float16 and BFloat16 inputs.
// ---------------------------------------------------------------------------
TEST_P(ReductionHalfDtypeTest, NormDimFloat16) {
    Tensor x_f32({2, 3}, DType::Float32, Device::cpu());
    auto* p = x_f32.data<float>();
    for (int i = 0; i < 6; ++i) p[i] = static_cast<float>(i + 1);  // 1..6

    Tensor x_f16 = x_f32.to(DType::Float16).to(device);

    // Should NOT throw — half-dtype norm(dim=) is the audit fix.
    Tensor out = tenzor::norm(x_f16, 2.0f, /*dim=*/1, /*keepdim=*/false);
    EXPECT_EQ(out.dtype(), DType::Float16);
    EXPECT_EQ(out.shape().size(), 1u);
    EXPECT_EQ(out.shape()[0], 2);

    // Reference values (L2 norm of each row):
    //   row 0: sqrt(1+4+9) = sqrt(14) ≈ 3.742
    //   row 1: sqrt(16+25+36) = sqrt(77) ≈ 8.775
    Tensor out_f32 = out.cpu().to(DType::Float32);
    EXPECT_NEAR(out_f32.data<float>()[0], std::sqrt(14.0f), 0.05f);
    EXPECT_NEAR(out_f32.data<float>()[1], std::sqrt(77.0f), 0.1f);
}

TEST_P(ReductionHalfDtypeTest, NormDimBFloat16) {
    Tensor x_f32({2, 3}, DType::Float32, Device::cpu());
    auto* p = x_f32.data<float>();
    for (int i = 0; i < 6; ++i) p[i] = static_cast<float>(i + 1);

    Tensor x_bf16 = x_f32.to(DType::BFloat16).to(device);
    Tensor out = tenzor::norm(x_bf16, 2.0f, /*dim=*/1, /*keepdim=*/false);
    EXPECT_EQ(out.dtype(), DType::BFloat16);
    Tensor out_f32 = out.cpu().to(DType::Float32);
    EXPECT_NEAR(out_f32.data<float>()[0], std::sqrt(14.0f), 0.1f);
    EXPECT_NEAR(out_f32.data<float>()[1], std::sqrt(77.0f), 0.2f);
}

// ---------------------------------------------------------------------------
// E.2 — prod must preserve the input dtype on Float16 / BFloat16 (was
//       silently promoting to Float32).
// ---------------------------------------------------------------------------
TEST_P(ReductionHalfDtypeTest, ProdFloat16PreservesDtype) {
    Tensor x_f32({4}, DType::Float32, Device::cpu());
    auto* p = x_f32.data<float>();
    p[0] = 1.0f; p[1] = 2.0f; p[2] = 3.0f; p[3] = 4.0f;
    Tensor x_f16 = x_f32.to(DType::Float16).to(device);

    Tensor out = tenzor::prod(x_f16);
    EXPECT_EQ(out.dtype(), DType::Float16)
        << "prod must preserve the input dtype (Float16), not silently promote to Float32";
    Tensor out_f32 = out.cpu().to(DType::Float32);
    EXPECT_NEAR(out_f32.data<float>()[0], 24.0f, 0.1f);
}

TEST_P(ReductionHalfDtypeTest, ProdBFloat16PreservesDtype) {
    Tensor x_f32({3}, DType::Float32, Device::cpu());
    auto* p = x_f32.data<float>();
    p[0] = 2.0f; p[1] = 3.0f; p[2] = 5.0f;
    Tensor x_bf16 = x_f32.to(DType::BFloat16).to(device);

    Tensor out = tenzor::prod(x_bf16);
    EXPECT_EQ(out.dtype(), DType::BFloat16);
    Tensor out_f32 = out.cpu().to(DType::Float32);
    EXPECT_NEAR(out_f32.data<float>()[0], 30.0f, 0.2f);
}

TEST_P(ReductionHalfDtypeTest, ProdAlongDimFloat16PreservesDtype) {
    Tensor x_f32({2, 3}, DType::Float32, Device::cpu());
    auto* p = x_f32.data<float>();
    // [[1, 2, 3], [4, 5, 6]]
    for (int i = 0; i < 6; ++i) p[i] = static_cast<float>(i + 1);
    Tensor x_f16 = x_f32.to(DType::Float16).to(device);

    Tensor out = tenzor::prod(x_f16, /*dim=*/1, /*keepdim=*/false);
    EXPECT_EQ(out.dtype(), DType::Float16);
    EXPECT_EQ(out.shape().size(), 1u);
    EXPECT_EQ(out.shape()[0], 2);
    Tensor out_f32 = out.cpu().to(DType::Float32);
    EXPECT_NEAR(out_f32.data<float>()[0], 6.0f, 0.05f);   // 1*2*3
    EXPECT_NEAR(out_f32.data<float>()[1], 120.0f, 1.0f);  // 4*5*6
}

}  // namespace

INSTANTIATE_BACKEND_TESTS(ReductionHalfDtypeTest);
