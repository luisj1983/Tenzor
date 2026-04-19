/**
 * @file test_dropout_variants_multidtype.cpp
 * @brief Multi-dtype tests for AlphaDropout and VariationalDropout layers
 *
 * Tests dropout variants with Float32, Float64, and Float16 dtypes across
 * CPU, CUDA, OneAPI, Vulkan, and ROCm backends to ensure:
 * - AlphaDropout modifies output in train mode, identity in eval mode
 * - VariationalDropout zeroes elements in train mode, identity in eval mode
 * - VariationalDropout mask reset behavior
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/nn/layers/dropout.hpp>
#include "../../multi_backend_dtype_fixture.hpp"
#include <cmath>

using namespace tenzor;
using namespace tenzor::nn;
using namespace tenzor::testing;

// ============================================================================
// Dropout Variants Multi-Backend Multi-DType Test Fixture
// ============================================================================

class DropoutVariantsMultiDTypeTest : public MultiBackendDTypeTest {};

// ============================================================================
// AlphaDropout Tests
// ============================================================================

TEST_P(DropoutVariantsMultiDTypeTest, AlphaDropoutTrainDrops) {
    AlphaDropout alpha_dropout(0.5);
    convert_model(alpha_dropout);
    alpha_dropout.train();

    auto input_tensor = createOnes({100, 100});
    Variable input(input_tensor, false);
    auto output = alpha_dropout.forward(input);

    expectDType(output.tensor());

    // In train mode, some elements should be modified (not all equal to 1.0)
    auto output_f32 = output.tensor().to(Device::cpu()).to(DType::Float32);
    auto input_f32 = input_tensor.to(Device::cpu()).to(DType::Float32);
    auto* out_data = output_f32.data<float>();
    auto* in_data = input_f32.data<float>();

    size_t modified_count = 0;
    for (int64_t i = 0; i < output_f32.numel(); ++i) {
        if (std::abs(out_data[i] - in_data[i]) > atol()) {
            modified_count++;
        }
    }

    EXPECT_GT(modified_count, 0)
        << "AlphaDropout in train mode should modify some elements";
}

TEST_P(DropoutVariantsMultiDTypeTest, AlphaDropoutEvalIdentity) {
    AlphaDropout alpha_dropout(0.5);
    convert_model(alpha_dropout);
    alpha_dropout.eval();

    auto input_tensor = createOnes({20, 20});
    Variable input(input_tensor, false);
    auto output = alpha_dropout.forward(input);

    expectDType(output.tensor());
    expectShape(output.tensor(), {20, 20});

    // In eval mode, output should equal input
    auto output_f32 = output.tensor().to(Device::cpu()).to(DType::Float32);
    auto* out_data = output_f32.data<float>();

    for (int64_t i = 0; i < output_f32.numel(); ++i) {
        EXPECT_NEAR(out_data[i], 1.0f, atol());
    }
}

// ============================================================================
// VariationalDropout Tests
// ============================================================================

TEST_P(DropoutVariantsMultiDTypeTest, VariationalDropoutTrainDrops) {
    VariationalDropout vdrop(0.5);
    convert_model(vdrop);
    vdrop.train();
    vdrop.reset_mask();

    auto input_tensor = createOnes({100, 100});
    Variable input(input_tensor, false);
    auto output = vdrop.forward(input);

    expectDType(output.tensor());

    // In train mode, some elements should be zeroed
    auto output_f32 = output.tensor().to(Device::cpu()).to(DType::Float32);
    auto* out_data = output_f32.data<float>();

    size_t zero_count = 0;
    size_t non_zero_count = 0;

    for (int64_t i = 0; i < output_f32.numel(); ++i) {
        if (std::abs(out_data[i]) < atol()) {
            zero_count++;
        } else {
            non_zero_count++;
        }
    }

    EXPECT_GT(zero_count, 0) << "VariationalDropout should zero some elements";
    EXPECT_GT(non_zero_count, 0) << "VariationalDropout should keep some elements";
}

TEST_P(DropoutVariantsMultiDTypeTest, VariationalDropoutEvalIdentity) {
    VariationalDropout vdrop(0.5);
    convert_model(vdrop);
    vdrop.eval();

    auto input_tensor = createOnes({20, 20});
    Variable input(input_tensor, false);
    auto output = vdrop.forward(input);

    expectDType(output.tensor());
    expectShape(output.tensor(), {20, 20});

    // In eval mode, output should equal input
    auto output_f32 = output.tensor().to(Device::cpu()).to(DType::Float32);
    auto* out_data = output_f32.data<float>();

    for (int64_t i = 0; i < output_f32.numel(); ++i) {
        EXPECT_NEAR(out_data[i], 1.0f, atol());
    }
}

TEST_P(DropoutVariantsMultiDTypeTest, VariationalDropoutResetMask) {
    VariationalDropout vdrop(0.5);
    convert_model(vdrop);
    vdrop.train();

    auto input_tensor = createOnes({50, 50});
    Variable input(input_tensor, false);

    // First forward pass with initial mask
    vdrop.reset_mask();
    auto output1 = vdrop.forward(input);
    auto out1_f32 = output1.tensor().to(Device::cpu()).to(DType::Float32);

    // Reset mask and run again -- mask should change
    vdrop.reset_mask();
    auto output2 = vdrop.forward(input);
    auto out2_f32 = output2.tensor().to(Device::cpu()).to(DType::Float32);

    auto* data1 = out1_f32.data<float>();
    auto* data2 = out2_f32.data<float>();

    // After reset_mask(), the outputs should differ (different random masks)
    size_t diff_count = 0;
    for (int64_t i = 0; i < out1_f32.numel(); ++i) {
        if (std::abs(data1[i] - data2[i]) > atol()) {
            diff_count++;
        }
    }

    EXPECT_GT(diff_count, 0)
        << "After reset_mask(), the dropout mask should change";
}

// ============================================================================
// Phase 3 additions: backward gradient population in eval mode (deterministic).
// In training mode dropout's stochastic mask makes gradient comparisons noisy;
// eval mode gives the deterministic identity path so we can verify the
// backward implementation forwards gradients through.
// ============================================================================

TEST_P(DropoutVariantsMultiDTypeTest, AlphaDropoutEvalBackwardGradPopulated) {
    AlphaDropout alpha_dropout(0.5);
    convert_model(alpha_dropout);
    alpha_dropout.eval();
    Variable input = createInput({4, 8}, true);
    auto output = alpha_dropout.forward(input);
    sum(output).backward();
    ASSERT_TRUE(input.has_grad()) << device().to_string();
    auto g_max = max(abs(input.grad()->to(Device::cpu()).to(DType::Float32)));
    EXPECT_GT(g_max.item<float>(), 0.0f);
}

TEST_P(DropoutVariantsMultiDTypeTest, VariationalDropoutEvalBackwardGradPopulated) {
    VariationalDropout vd(0.5);
    convert_model(vd);
    vd.eval();
    Variable input = createInput({4, 8}, true);
    auto output = vd.forward(input);
    sum(output).backward();
    ASSERT_TRUE(input.has_grad()) << device().to_string();
    auto g_max = max(abs(input.grad()->to(Device::cpu()).to(DType::Float32)));
    EXPECT_GT(g_max.item<float>(), 0.0f);
}

// ============================================================================
// Test Instantiation
// ============================================================================

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(DropoutVariantsMultiDTypeTest);
