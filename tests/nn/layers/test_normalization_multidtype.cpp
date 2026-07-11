/**
 * @file test_normalization_multidtype.cpp
 * @brief Multi-dtype tests for LayerNorm and GroupNorm layers
 *
 * Tests normalization layers with Float32, Float64, and Float16 dtypes across
 * CPU, CUDA, OneAPI, Vulkan, and ROCm backends to ensure:
 * - Correct normalization (zero mean, unit variance)
 * - Proper handling of affine parameters
 * - Gradient flow through normalization layers
 * - Batch and group handling
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include "../../multi_backend_dtype_fixture.hpp"
#include <cmath>

using namespace tenzor;
using namespace tenzor::nn;
using namespace tenzor::testing;

// ============================================================================
// Normalization Multi-Backend Multi-DType Test Fixture
// ============================================================================

class NormalizationMultiDTypeTest : public MultiBackendDTypeTest {
protected:
    // Additional tolerance for variance checks (normalization has inherent numerical error)
    float variance_tolerance() const {
        if (dtype() == DType::Float16) {
            return 0.1f;  // Float16 accumulates significant error in variance
        } else if (dtype() == DType::Float64) {
            return 1e-5f;
        } else if (dtype() == DType::BFloat16) {
            // BFloat16 has only 8 mantissa bits (~2-3 decimal digits). The
            // CPU kernel (src/backends/cpu/kernels/nn_kernels.cpp's
            // layer_norm_scalar<BFloat16>) already accumulates mean/variance
            // in double precision and narrows only the final stored output,
            // and every backend produces bit-identical results for this
            // test's small inputs — confirming the residual error is
            // expected BFloat16 output-rounding noise, not a computational
            // bug. Empirically observed ~4.8e-3 (2D case, the larger of the
            // two LayerNorm forward tests).
            return 1e-2f;
        }
        return 1e-4f;
    }

    Tensor create_tensor(const std::vector<float>& data, const std::vector<int64_t>& shape) {
        auto tensor = tenzor::from_data(data.data(), shape);
        if (dtype() != DType::Float32) {
            tensor = tensor.to(dtype());
        }
        if (device() != Device::cpu()) {
            tensor = tensor.to(device());
        }
        return tensor;
    }
};

// ============================================================================
// LayerNorm Tests
// ============================================================================

TEST_P(NormalizationMultiDTypeTest, LayerNormConstructorWithAffine) {
    LayerNorm ln({10}, 1e-5, true);
    auto params = ln.parameters();
    EXPECT_EQ(params.size(), 2);
}

TEST_P(NormalizationMultiDTypeTest, LayerNormForwardNormalization1D) {
    LayerNorm ln({4}, 1e-5, false);
    convert_model(ln);

    auto input_data = std::vector<float>{
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f
    };
    auto input_tensor = create_tensor(input_data, {2, 4});
    auto input = Variable(input_tensor, false);

    auto output = ln(input);
    expectDType(output.tensor());

    auto output_f32 = output.tensor().to(Device::cpu()).to(DType::Float32);
    auto output_data = output_f32.data<float>();

    float mean_row0 = 0.0f;
    for (int i = 0; i < 4; i++) {
        mean_row0 += output_data[i];
    }
    mean_row0 /= 4.0f;
    EXPECT_NEAR(mean_row0, 0.0f, atol() * 10);

    float var_row0 = 0.0f;
    for (int i = 0; i < 4; i++) {
        var_row0 += output_data[i] * output_data[i];
    }
    var_row0 /= 4.0f;
    EXPECT_NEAR(var_row0, 1.0f, variance_tolerance());
}

TEST_P(NormalizationMultiDTypeTest, LayerNormForwardNormalization2D) {
    LayerNorm ln({2, 2, 2}, 1e-5, false);
    convert_model(ln);

    auto input_data = std::vector<float>{
        1.0f, 2.0f,  3.0f, 4.0f,
        5.0f, 6.0f,  7.0f, 8.0f
    };
    auto input_tensor = create_tensor(input_data, {1, 2, 2, 2});
    auto input = Variable(input_tensor, false);

    auto output = ln(input);
    expectDType(output.tensor());

    auto output_f32 = output.tensor().to(Device::cpu()).to(DType::Float32);
    auto output_data = output_f32.data<float>();

    float mean = 0.0f;
    for (int i = 0; i < 8; i++) {
        mean += output_data[i];
    }
    mean /= 8.0f;
    EXPECT_NEAR(mean, 0.0f, atol() * 10);

    float var = 0.0f;
    for (int i = 0; i < 8; i++) {
        var += output_data[i] * output_data[i];
    }
    var /= 8.0f;
    EXPECT_NEAR(var, 1.0f, variance_tolerance());
}

TEST_P(NormalizationMultiDTypeTest, LayerNormBackwardGradientFlow) {
    LayerNorm ln({4}, 1e-5, true);
    convert_model(ln);
    ln.train();

    auto input_data = std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f};
    auto input_tensor = create_tensor(input_data, {1, 4});
    auto input = Variable(input_tensor, true);

    auto output = ln(input);

    auto grad_output = tenzor::ones({1, 4}, dtype(), device());
    output.backward(grad_output);

    EXPECT_TRUE(input.has_grad());
    EXPECT_EQ(input.grad()->dtype(), dtype());

    auto params = ln.parameters();
    for (auto& param : params) {
        EXPECT_TRUE(param->has_grad());
    }
}

TEST_P(NormalizationMultiDTypeTest, LayerNormMultipleBatches) {
    LayerNorm ln({3}, 1e-5, false);
    convert_model(ln);

    auto input_data = std::vector<float>{
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
        7.0f, 8.0f, 9.0f
    };
    auto input_tensor = create_tensor(input_data, {3, 3});
    auto input = Variable(input_tensor, false);

    auto output = ln(input);
    expectDType(output.tensor());

    auto output_f32 = output.tensor().to(Device::cpu()).to(DType::Float32);
    auto output_data = output_f32.data<float>();

    // Each row should be independently normalized
    for (int b = 0; b < 3; b++) {
        float mean = 0.0f;
        for (int i = 0; i < 3; i++) {
            mean += output_data[b * 3 + i];
        }
        mean /= 3.0f;
        EXPECT_NEAR(mean, 0.0f, atol() * 10);
    }
}

TEST_P(NormalizationMultiDTypeTest, LayerNormShapePreservation) {
    LayerNorm ln({64}, 1e-5, true);
    convert_model(ln);

    Variable input = createInput({4, 32, 64}, false);
    auto output = ln(input);

    expectShape(output.tensor(), {4, 32, 64});
    expectDType(output.tensor());
}

// ============================================================================
// GroupNorm Tests
// ============================================================================

TEST_P(NormalizationMultiDTypeTest, GroupNormConstructorWithAffine) {
    GroupNorm gn(2, 4, 1e-5, true);
    auto params = gn.parameters();
    EXPECT_EQ(params.size(), 2);
}

TEST_P(NormalizationMultiDTypeTest, GroupNormForwardNormalization) {
    GroupNorm gn(2, 4, 1e-5, false);
    convert_model(gn);

    auto input_data = std::vector<float>{
        // Channel 0 (Group 0)
        1.0f, 2.0f,
        3.0f, 4.0f,
        // Channel 1 (Group 0)
        5.0f, 6.0f,
        7.0f, 8.0f,
        // Channel 2 (Group 1)
        9.0f, 10.0f,
        11.0f, 12.0f,
        // Channel 3 (Group 1)
        13.0f, 14.0f,
        15.0f, 16.0f
    };
    auto input_tensor = create_tensor(input_data, {1, 4, 2, 2});
    auto input = Variable(input_tensor, false);

    auto output = gn(input);
    expectDType(output.tensor());

    auto output_f32 = output.tensor().to(Device::cpu()).to(DType::Float32);
    auto output_data = output_f32.data<float>();

    // Group 0: channels 0-1
    float mean_g0 = 0.0f;
    for (int i = 0; i < 8; i++) {
        mean_g0 += output_data[i];
    }
    mean_g0 /= 8.0f;
    EXPECT_NEAR(mean_g0, 0.0f, atol() * 10);

    // Group 1: channels 2-3
    float mean_g1 = 0.0f;
    for (int i = 8; i < 16; i++) {
        mean_g1 += output_data[i];
    }
    mean_g1 /= 8.0f;
    EXPECT_NEAR(mean_g1, 0.0f, atol() * 10);
}

TEST_P(NormalizationMultiDTypeTest, GroupNormSingleGroup) {
    GroupNorm gn(1, 4, 1e-5, false);
    convert_model(gn);

    auto input_data = std::vector<float>{
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f,
        9.0f, 10.0f, 11.0f, 12.0f,
        13.0f, 14.0f, 15.0f, 16.0f
    };
    auto input_tensor = create_tensor(input_data, {1, 4, 2, 2});
    auto input = Variable(input_tensor, false);

    auto output = gn(input);
    expectDType(output.tensor());

    auto output_f32 = output.tensor().to(Device::cpu()).to(DType::Float32);
    auto output_data = output_f32.data<float>();

    float mean = 0.0f;
    for (int i = 0; i < 16; i++) {
        mean += output_data[i];
    }
    mean /= 16.0f;
    EXPECT_NEAR(mean, 0.0f, atol() * 10);
}

TEST_P(NormalizationMultiDTypeTest, GroupNormGroupsEqualChannels) {
    GroupNorm gn(4, 4, 1e-5, false);
    convert_model(gn);

    auto input_data = std::vector<float>{
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f,
        9.0f, 10.0f, 11.0f, 12.0f,
        13.0f, 14.0f, 15.0f, 16.0f
    };
    auto input_tensor = create_tensor(input_data, {1, 4, 2, 2});
    auto input = Variable(input_tensor, false);

    auto output = gn(input);
    expectDType(output.tensor());

    auto output_f32 = output.tensor().to(Device::cpu()).to(DType::Float32);
    auto output_data = output_f32.data<float>();

    // Each channel normalized independently
    for (int c = 0; c < 4; c++) {
        float mean = 0.0f;
        for (int i = 0; i < 4; i++) {
            mean += output_data[c * 4 + i];
        }
        mean /= 4.0f;
        EXPECT_NEAR(mean, 0.0f, atol() * 10);
    }
}

TEST_P(NormalizationMultiDTypeTest, GroupNormBackwardGradientFlow) {
    GroupNorm gn(2, 4, 1e-5, true);
    convert_model(gn);
    gn.train();

    auto input_data = std::vector<float>(16, 1.0f);
    auto input_tensor = create_tensor(input_data, {1, 4, 2, 2});
    auto input = Variable(input_tensor, true);

    auto output = gn(input);

    auto grad_output = tenzor::ones({1, 4, 2, 2}, dtype(), device());
    output.backward(grad_output);

    EXPECT_TRUE(input.has_grad());
    EXPECT_EQ(input.grad()->dtype(), dtype());
}

TEST_P(NormalizationMultiDTypeTest, GroupNormMultipleBatches) {
    GroupNorm gn(2, 4, 1e-5, false);
    convert_model(gn);

    auto input_data = std::vector<float>(2 * 4 * 2 * 2);
    for (size_t i = 0; i < input_data.size(); i++) {
        input_data[i] = static_cast<float>(i + 1);
    }
    auto input_tensor = create_tensor(input_data, {2, 4, 2, 2});
    auto input = Variable(input_tensor, false);

    auto output = gn(input);
    EXPECT_EQ(output.shape()[0], 2);
    expectDType(output.tensor());
}

TEST_P(NormalizationMultiDTypeTest, GroupNormShapePreservation) {
    GroupNorm gn(8, 32, 1e-5, true);
    convert_model(gn);

    Variable input = createInput({4, 32, 16, 16}, false);
    auto output = gn(input);

    expectShape(output.tensor(), {4, 32, 16, 16});
    expectDType(output.tensor());
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST_P(NormalizationMultiDTypeTest, LayerNormLargeInput) {
    LayerNorm ln({64, 8, 8}, 1e-5, true);
    convert_model(ln);

    Variable input = createInput({2, 64, 8, 8}, true);
    auto output = ln(input);

    expectShape(output.tensor(), {2, 64, 8, 8});
    expectDType(output.tensor());
}

TEST_P(NormalizationMultiDTypeTest, GroupNormLargeInput) {
    GroupNorm gn(8, 64, 1e-5, true);
    convert_model(gn);

    Variable input = createInput({2, 64, 8, 8}, true);
    auto output = gn(input);

    expectShape(output.tensor(), {2, 64, 8, 8});
    expectDType(output.tensor());
}

TEST_P(NormalizationMultiDTypeTest, EpsilonEffect) {
    LayerNorm ln1({4}, 1e-5, false);
    LayerNorm ln2({4}, 1e-1, false);
    convert_model(ln1);
    convert_model(ln2);

    // Input with all same values (zero variance)
    auto input_data = std::vector<float>{2.0f, 2.0f, 2.0f, 2.0f};
    auto input1 = Variable(create_tensor(input_data, {1, 4}), false);
    auto input2 = Variable(create_tensor(input_data, {1, 4}), false);

    auto output1 = ln1(input1);
    auto output2 = ln2(input2);

    auto data1 = output1.tensor().to(Device::cpu()).to(DType::Float32).data<float>();
    auto data2 = output2.tensor().to(Device::cpu()).to(DType::Float32).data<float>();

    for (int i = 0; i < 4; i++) {
        EXPECT_TRUE(std::isfinite(data1[i]));
        EXPECT_TRUE(std::isfinite(data2[i]));
    }
}

TEST_P(NormalizationMultiDTypeTest, LayerNormDifferentNormalizedShapes) {
    std::vector<std::vector<int64_t>> normalized_shapes = {
        {8}, {8, 8}, {4, 8, 8}
    };

    for (const auto& norm_shape : normalized_shapes) {
        LayerNorm ln(norm_shape, 1e-5, true);
        convert_model(ln);

        std::vector<int64_t> input_shape = {2};
        for (auto dim : norm_shape) {
            input_shape.push_back(dim);
        }

        Variable input = createInput(input_shape, false);
        auto output = ln(input);

        expectShape(output.tensor(), input_shape);
        expectDType(output.tensor());
    }
}

TEST_P(NormalizationMultiDTypeTest, GroupNormDifferentGroupCounts) {
    std::vector<std::pair<int64_t, int64_t>> group_channel_pairs = {
        {1, 8}, {2, 8}, {4, 8}, {8, 8}
    };

    for (const auto& [num_groups, num_channels] : group_channel_pairs) {
        GroupNorm gn(num_groups, num_channels, 1e-5, true);
        convert_model(gn);

        Variable input = createInput({2, num_channels, 8, 8}, false);
        auto output = gn(input);

        expectShape(output.tensor(), {2, num_channels, 8, 8});
        expectDType(output.tensor());
    }
}

// ============================================================================
// Test Instantiation
// ============================================================================

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(NormalizationMultiDTypeTest);

/*
 * COVERAGE SUMMARY:
 *
 * Test Cases: 18
 * DTypes Tested: Float32, Float64, Float16
 * Backends Tested: CPU, CUDA, OneAPI
 * Total Scenarios: 18 tests × 3 dtypes × 3 backends = 162 test scenarios
 *
 * Coverage:
 * - LayerNorm: constructor, 1D/2D normalization, backward pass, batches, shapes
 * - GroupNorm: constructor, normalization, single group, groups=channels, backward, batches, shapes
 * - Edge cases: large inputs, epsilon effect, different normalized shapes, different group counts
 */
