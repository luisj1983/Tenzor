/**
 * @file test_scatter_variants_multidtype.cpp
 * @brief Multi-backend multi-dtype tests for select_scatter, slice_scatter, diagonal_scatter
 */

#include "../multi_backend_dtype_fixture.hpp"
#include "tenzor/core/tensor.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/indexing.hpp"

using namespace tenzor;
using namespace tenzor::testing;

class ScatterVariantsMultiDTypeTest : public MultiBackendDTypeTest {};

// ============================================================================
// select_scatter
// ============================================================================

TEST_P(ScatterVariantsMultiDTypeTest, SelectScatterBasic) {
    auto input = createZeros({3, 4});
    auto src = createOnes({4});

    auto result = select_scatter(input, src, /*dim=*/0, /*index=*/1);

    ASSERT_EQ(result.shape().size(), 2u);
    EXPECT_EQ(result.shape()[0], 3);
    EXPECT_EQ(result.shape()[1], 4);
    expectDType(result);

    auto cpu_result = result.to(Device::cpu()).to(DType::Float32);
    auto* data = cpu_result.data<float>();
    for (int j = 0; j < 4; ++j) {
        EXPECT_NEAR(data[0 * 4 + j], 0.0f, atol()) << "row 0, col " << j;
    }
    for (int j = 0; j < 4; ++j) {
        EXPECT_NEAR(data[1 * 4 + j], 1.0f, atol()) << "row 1, col " << j;
    }
    for (int j = 0; j < 4; ++j) {
        EXPECT_NEAR(data[2 * 4 + j], 0.0f, atol()) << "row 2, col " << j;
    }
}

TEST_P(ScatterVariantsMultiDTypeTest, SelectScatterDim1) {
    auto input = createZeros({3, 4});
    // Create source values on device with correct dtype
    auto src_cpu = zeros({3}, DType::Float32, Device::cpu());
    auto* sp = src_cpu.data<float>();
    sp[0] = 10.0f; sp[1] = 20.0f; sp[2] = 30.0f;
    auto src = src_cpu.to(dtype()).to(device());

    auto result = select_scatter(input, src, /*dim=*/1, /*index=*/2);

    auto cpu_result = result.to(Device::cpu()).to(DType::Float32);
    auto* data = cpu_result.data<float>();
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 4; ++j) {
            float expected = (j == 2) ? sp[i] : 0.0f;
            EXPECT_NEAR(data[i * 4 + j], expected, atol())
                << "at (" << i << ", " << j << ")";
        }
    }
}

TEST_P(ScatterVariantsMultiDTypeTest, SelectScatterNonMutating) {
    auto input = createOnes({3, 3});
    auto src = createZeros({3});

    auto result = select_scatter(input, src, 0, 0);

    // Original input should be unchanged
    auto orig = input.to(Device::cpu()).to(DType::Float32);
    auto* orig_data = orig.data<float>();
    for (int i = 0; i < 9; ++i) {
        EXPECT_NEAR(orig_data[i], 1.0f, atol()) << "input mutated at index " << i;
    }
}

// ============================================================================
// slice_scatter
// ============================================================================

TEST_P(ScatterVariantsMultiDTypeTest, SliceScatterBasic) {
    auto input = createZeros({4, 4});
    auto src = createOnes({2, 4});

    auto result = slice_scatter(input, src, /*dim=*/0, /*start=*/1, /*end=*/3);

    auto cpu_result = result.to(Device::cpu()).to(DType::Float32);
    auto* data = cpu_result.data<float>();
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            float expected = (i == 1 || i == 2) ? 1.0f : 0.0f;
            EXPECT_NEAR(data[i * 4 + j], expected, atol())
                << "at (" << i << ", " << j << ")";
        }
    }
}

TEST_P(ScatterVariantsMultiDTypeTest, SliceScatterWithStep) {
    auto input = createZeros({6});
    auto src_cpu = zeros({3}, DType::Float32, Device::cpu());
    auto* sp = src_cpu.data<float>();
    sp[0] = 10.0f; sp[1] = 20.0f; sp[2] = 30.0f;
    auto src = src_cpu.to(dtype()).to(device());

    auto result = slice_scatter(input, src, /*dim=*/0, /*start=*/0, /*end=*/6, /*step=*/2);

    auto cpu_result = result.to(Device::cpu()).to(DType::Float32);
    auto* data = cpu_result.data<float>();
    EXPECT_NEAR(data[0], 10.0f, atol());
    EXPECT_NEAR(data[1], 0.0f, atol());
    EXPECT_NEAR(data[2], 20.0f, atol());
    EXPECT_NEAR(data[3], 0.0f, atol());
    EXPECT_NEAR(data[4], 30.0f, atol());
    EXPECT_NEAR(data[5], 0.0f, atol());
}

// ============================================================================
// diagonal_scatter
// ============================================================================

TEST_P(ScatterVariantsMultiDTypeTest, DiagonalScatterBasic) {
    auto input = createZeros({3, 3});
    auto src_cpu = zeros({3}, DType::Float32, Device::cpu());
    auto* sp = src_cpu.data<float>();
    sp[0] = 1.0f; sp[1] = 2.0f; sp[2] = 3.0f;
    auto src = src_cpu.to(dtype()).to(device());

    auto result = diagonal_scatter(input, src, /*offset=*/0);

    auto cpu_result = result.to(Device::cpu()).to(DType::Float32);
    auto* data = cpu_result.data<float>();
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            float expected = (i == j) ? sp[i] : 0.0f;
            EXPECT_NEAR(data[i * 3 + j], expected, atol())
                << "at (" << i << ", " << j << ")";
        }
    }
}

TEST_P(ScatterVariantsMultiDTypeTest, DiagonalScatterNonMutating) {
    auto input = createOnes({3, 3});
    auto src = createZeros({3});

    auto result = diagonal_scatter(input, src, 0);

    auto orig = input.to(Device::cpu()).to(DType::Float32);
    auto* orig_data = orig.data<float>();
    for (int i = 0; i < 9; ++i) {
        EXPECT_NEAR(orig_data[i], 1.0f, atol());
    }
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(ScatterVariantsMultiDTypeTest);
