#include <gtest/gtest.h>
#include "tenzor/nn/layers/normalization.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/reduction.hpp"
#include "tenzor/autograd/variable.hpp"
#include <cmath>

// Forward declare tenzor::initialize
namespace tenzor {
    void initialize();
}

using namespace tenzor;
using namespace tenzor::nn;

// Global test environment
class TenzorTestEnvironment : public ::testing::Environment {
public:
    void SetUp() override {
        tenzor::initialize();
    }
};

static ::testing::Environment* const tenzor_env =
    ::testing::AddGlobalTestEnvironment(new TenzorTestEnvironment);

// ============================================================================
// LayerNorm Tests
// ============================================================================

TEST(LayerNormTest, ConstructorWithAffine) {
    LayerNorm ln({10}, 1e-5, true);
    auto params = ln.parameters();
    EXPECT_EQ(params.size(), 2);  // weight and bias
}

TEST(LayerNormTest, ConstructorWithoutAffine) {
    LayerNorm ln({10}, 1e-5, false);
    auto params = ln.parameters();
    EXPECT_EQ(params.size(), 0);  // no learnable parameters
}

TEST(LayerNormTest, ForwardNormalization1D) {
    // Test with 2D input [batch, features]
    LayerNorm ln({4}, 1e-5, false);

    // Create input: [2, 4]
    auto input_data = std::vector<float>{
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f
    };
    auto input = Variable(from_data(input_data.data(), {2, 4}), false);

    auto output = ln(input);
    auto output_data = output.tensor().data<float>();

    // Check that each row is normalized (mean=0, var=1)
    // First row: [1, 2, 3, 4] -> mean=2.5, var=1.25
    float mean_row0 = 0.0f;
    for (int i = 0; i < 4; i++) {
        mean_row0 += output_data[i];
    }
    mean_row0 /= 4.0f;
    EXPECT_NEAR(mean_row0, 0.0f, 1e-5);

    float var_row0 = 0.0f;
    for (int i = 0; i < 4; i++) {
        var_row0 += output_data[i] * output_data[i];
    }
    var_row0 /= 4.0f;
    EXPECT_NEAR(var_row0, 1.0f, 1e-4);
}

TEST(LayerNormTest, ForwardNormalization2D) {
    // Test with 4D input [N, C, H, W] normalizing over [C, H, W]
    LayerNorm ln({2, 2, 2}, 1e-5, false);

    // Create input: [1, 2, 2, 2]
    auto input_data = std::vector<float>{
        1.0f, 2.0f,  3.0f, 4.0f,
        5.0f, 6.0f,  7.0f, 8.0f
    };
    auto input = Variable(from_data(input_data.data(), {1, 2, 2, 2}), false);

    auto output = ln(input);
    auto output_data = output.tensor().data<float>();

    // Check normalization: mean should be ~0, variance ~1
    float mean = 0.0f;
    for (int i = 0; i < 8; i++) {
        mean += output_data[i];
    }
    mean /= 8.0f;
    EXPECT_NEAR(mean, 0.0f, 1e-5);

    float var = 0.0f;
    for (int i = 0; i < 8; i++) {
        var += output_data[i] * output_data[i];
    }
    var /= 8.0f;
    EXPECT_NEAR(var, 1.0f, 1e-4);
}

TEST(LayerNormTest, ForwardWithAffineTransform) {
    LayerNorm ln({4}, 1e-5, true);

    auto input_data = std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f};
    auto input = Variable(from_data(input_data.data(), {1, 4}), false);

    auto output = ln(input);
    EXPECT_EQ(output.shape()[0], 1);
    EXPECT_EQ(output.shape()[1], 4);
}

TEST(LayerNormTest, MultipleBatches) {
    LayerNorm ln({3}, 1e-5, false);

    auto input_data = std::vector<float>{
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
        7.0f, 8.0f, 9.0f
    };
    auto input = Variable(from_data(input_data.data(), {3, 3}), false);

    auto output = ln(input);
    auto output_data = output.tensor().data<float>();

    // Each row should be independently normalized
    for (int b = 0; b < 3; b++) {
        float mean = 0.0f;
        for (int i = 0; i < 3; i++) {
            mean += output_data[b * 3 + i];
        }
        mean /= 3.0f;
        EXPECT_NEAR(mean, 0.0f, 1e-5);
    }
}

TEST(LayerNormTest, BackwardGradientFlow) {
    LayerNorm ln({4}, 1e-5, true);
    ln.train();

    auto input_data = std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f};
    auto input = Variable(from_data(input_data.data(), {1, 4}), true);

    auto output = ln(input);

    // Backward with ones gradient
    auto grad_output = ones({1, 4});
    output.backward(grad_output);

    EXPECT_TRUE(input.has_grad());
    auto params = ln.parameters();
    for (auto& param : params) {
        EXPECT_TRUE(param->has_grad());
    }
}

TEST(LayerNormTest, NumericalGradientCheck) {
    float eps = 1e-4;
    auto input_data = std::vector<float>{1.0f, 2.0f, 3.0f};

    // Test gradient for each output element separately
    for (int out_idx = 0; out_idx < 3; out_idx++) {
        // Create fresh LayerNorm and input for each output index
        LayerNorm ln({3}, 1e-5, false);
        auto input = Variable(from_data(input_data.data(), {1, 3}), true);
        auto output = ln(input);

        // Backward with gradient only on one output element
        auto grad_output_data = std::vector<float>{0.0f, 0.0f, 0.0f};
        grad_output_data[out_idx] = 1.0f;
        auto grad_output = from_data(grad_output_data.data(), {1, 3});
        output.backward(grad_output);

        ASSERT_TRUE(input.has_grad()) << "Input should have gradient after backward";
        auto analytical_grad = input.grad().value().data<float>();

        // Numerical gradient check for each input
        for (int in_idx = 0; in_idx < 3; in_idx++) {
            LayerNorm ln_plus({3}, 1e-5, false);
            LayerNorm ln_minus({3}, 1e-5, false);

            auto input_plus = from_data(input_data.data(), {1, 3});
            auto input_plus_data = input_plus.data<float>();
            input_plus_data[in_idx] += eps;

            auto input_minus = from_data(input_data.data(), {1, 3});
            auto input_minus_data = input_minus.data<float>();
            input_minus_data[in_idx] -= eps;

            auto output_plus = ln_plus(Variable(input_plus, false)).tensor();
            auto output_minus = ln_minus(Variable(input_minus, false)).tensor();

            // Numerical gradient: d(output[out_idx])/d(input[in_idx])
            float numerical_grad = (output_plus.data<float>()[out_idx] -
                                   output_minus.data<float>()[out_idx]) / (2.0f * eps);

            // Float32 gradient checking has inherent numerical precision limits
            // With small normalized_size (3 elements), accumulated rounding errors are
            // proportionally larger than larger normalizations. 4e-3 tolerance accounts
            // for this while still catching implementation bugs.
            EXPECT_NEAR(analytical_grad[in_idx], numerical_grad, 4e-3)
                << "Mismatch for d(output[" << out_idx << "])/d(input[" << in_idx << "])";
        }
    }
}

TEST(LayerNormTest, ParameterGradients) {
    LayerNorm ln({4}, 1e-5, true);
    ln.train();

    auto input_data = std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f};
    auto input = Variable(from_data(input_data.data(), {1, 4}), true);

    auto output = ln(input);

    auto grad_output = ones({1, 4});
    output.backward(grad_output);

    auto params = ln.named_parameters();
    EXPECT_EQ(params.size(), 2);

    // Check that weight and bias have gradients
    for (auto& [name, param] : params) {
        EXPECT_TRUE(param->has_grad());
        auto grad_data = param->grad().value().data<float>();

        // Gradient should be non-zero
        bool has_nonzero = false;
        for (int i = 0; i < 4; i++) {
            if (std::abs(grad_data[i]) > 1e-6) {
                has_nonzero = true;
                break;
            }
        }
        EXPECT_TRUE(has_nonzero);
    }
}

// ============================================================================
// RMSNorm Tests
// ============================================================================

TEST(RMSNormTest, Constructor) {
    RMSNorm rn(10, 1e-6);
    auto params = rn.parameters();
    EXPECT_EQ(params.size(), 1);  // weight only, no bias
}

TEST(RMSNormTest, ForwardNormalization1D) {
    // Test with 2D input [batch, features]
    RMSNorm rn(4, 1e-6);

    // Create input: [2, 4]
    auto input_data = std::vector<float>{
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f
    };
    auto input = Variable(from_data(input_data.data(), {2, 4}), false);

    auto output = rn(input);
    auto output_data = output.tensor().data<float>();

    // Check that RMS normalization is applied
    // RMS of first row: sqrt((1+4+9+16)/4) = sqrt(7.5) ≈ 2.7386
    // Output should be input / rms * weight (weight = 1 initially)

    // Calculate RMS for first row
    float rms_row0 = std::sqrt((1.0f + 4.0f + 9.0f + 16.0f) / 4.0f);
    // Check values are normalized by RMS
    EXPECT_NEAR(output_data[0], 1.0f / rms_row0, 1e-4);
    EXPECT_NEAR(output_data[1], 2.0f / rms_row0, 1e-4);
    EXPECT_NEAR(output_data[2], 3.0f / rms_row0, 1e-4);
    EXPECT_NEAR(output_data[3], 4.0f / rms_row0, 1e-4);
}

TEST(RMSNormTest, ForwardNormalization2D) {
    // Test with 3D input [batch, seq, features]
    RMSNorm rn(4, 1e-6);

    // Create input: [1, 2, 4]
    auto input_data = std::vector<float>{
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f
    };
    auto input = Variable(from_data(input_data.data(), {1, 2, 4}), false);

    auto output = rn(input);
    EXPECT_EQ(output.shape()[0], 1);
    EXPECT_EQ(output.shape()[1], 2);
    EXPECT_EQ(output.shape()[2], 4);
}

TEST(RMSNormTest, MultipleBatches) {
    RMSNorm rn(3, 1e-6);

    auto input_data = std::vector<float>{
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
        7.0f, 8.0f, 9.0f
    };
    auto input = Variable(from_data(input_data.data(), {3, 3}), false);

    auto output = rn(input);
    auto output_data = output.tensor().data<float>();

    // Each row should be independently normalized
    // Verify by checking that each row has the same RMS after normalization
    for (int b = 0; b < 3; b++) {
        float sum_sq = 0.0f;
        for (int i = 0; i < 3; i++) {
            sum_sq += output_data[b * 3 + i] * output_data[b * 3 + i];
        }
        float rms = std::sqrt(sum_sq / 3.0f);
        // RMS should be ~1 (since weight is initialized to 1)
        EXPECT_NEAR(rms, 1.0f, 1e-4);
    }
}

TEST(RMSNormTest, BackwardGradientFlow) {
    RMSNorm rn(4, 1e-6);
    rn.train();

    auto input_data = std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f};
    auto input = Variable(from_data(input_data.data(), {1, 4}), true);

    auto output = rn(input);

    // Backward with ones gradient
    auto grad_output = ones({1, 4});
    output.backward(grad_output);

    EXPECT_TRUE(input.has_grad());
    auto params = rn.parameters();
    for (auto& param : params) {
        EXPECT_TRUE(param->has_grad());
    }
}

TEST(RMSNormTest, NumericalGradientCheck) {
    float eps = 1e-4;
    auto input_data = std::vector<float>{1.0f, 2.0f, 3.0f};

    // Test gradient for each output element separately
    for (int out_idx = 0; out_idx < 3; out_idx++) {
        // Create fresh RMSNorm and input for each output index
        RMSNorm rn(3, 1e-6);
        auto input = Variable(from_data(input_data.data(), {1, 3}), true);
        auto output = rn(input);

        // Backward with gradient only on one output element
        auto grad_output_data = std::vector<float>{0.0f, 0.0f, 0.0f};
        grad_output_data[out_idx] = 1.0f;
        auto grad_output = from_data(grad_output_data.data(), {1, 3});
        output.backward(grad_output);

        ASSERT_TRUE(input.has_grad()) << "Input should have gradient after backward";
        auto analytical_grad = input.grad().value().data<float>();

        // Numerical gradient check for each input
        for (int in_idx = 0; in_idx < 3; in_idx++) {
            RMSNorm rn_plus(3, 1e-6);
            RMSNorm rn_minus(3, 1e-6);

            auto input_plus = from_data(input_data.data(), {1, 3});
            auto input_plus_data = input_plus.data<float>();
            input_plus_data[in_idx] += eps;

            auto input_minus = from_data(input_data.data(), {1, 3});
            auto input_minus_data = input_minus.data<float>();
            input_minus_data[in_idx] -= eps;

            auto output_plus = rn_plus(Variable(input_plus, false)).tensor();
            auto output_minus = rn_minus(Variable(input_minus, false)).tensor();

            // Numerical gradient: d(output[out_idx])/d(input[in_idx])
            float numerical_grad = (output_plus.data<float>()[out_idx] -
                                   output_minus.data<float>()[out_idx]) / (2.0f * eps);

            EXPECT_NEAR(analytical_grad[in_idx], numerical_grad, 2e-3)
                << "Mismatch for d(output[" << out_idx << "])/d(input[" << in_idx << "])";
        }
    }
}

TEST(RMSNormTest, ParameterGradients) {
    RMSNorm rn(4, 1e-6);
    rn.train();

    auto input_data = std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f};
    auto input = Variable(from_data(input_data.data(), {1, 4}), true);

    auto output = rn(input);

    auto grad_output = ones({1, 4});
    output.backward(grad_output);

    auto params = rn.named_parameters();
    EXPECT_EQ(params.size(), 1);  // Only weight, no bias

    // Check that weight has gradient
    for (auto& [name, param] : params) {
        EXPECT_TRUE(param->has_grad());
        auto grad_data = param->grad().value().data<float>();

        // Gradient should be non-zero
        bool has_nonzero = false;
        for (int i = 0; i < 4; i++) {
            if (std::abs(grad_data[i]) > 1e-6) {
                has_nonzero = true;
                break;
            }
        }
        EXPECT_TRUE(has_nonzero);
    }
}

TEST(RMSNormTest, EpsilonEffect) {
    // Test that eps prevents division by zero
    RMSNorm rn1(4, 1e-6);
    RMSNorm rn2(4, 1e-1);

    // Input with all zeros (zero RMS)
    auto input_data = std::vector<float>{0.0f, 0.0f, 0.0f, 0.0f};
    auto input1 = Variable(from_data(input_data.data(), {1, 4}), false);
    auto input2 = Variable(from_data(input_data.data(), {1, 4}), false);

    auto output1 = rn1(input1);
    auto output2 = rn2(input2);

    // Both should produce valid output (all zeros)
    auto data1 = output1.tensor().data<float>();
    auto data2 = output2.tensor().data<float>();

    for (int i = 0; i < 4; i++) {
        EXPECT_TRUE(std::isfinite(data1[i]));
        EXPECT_TRUE(std::isfinite(data2[i]));
    }
}

TEST(RMSNormTest, LargeInput) {
    // Test with realistic transformer-like dimensions
    RMSNorm rn(512, 1e-6);

    auto input = Variable(randn({4, 128, 512}), true);
    auto output = rn(input);

    EXPECT_EQ(output.shape()[0], 4);
    EXPECT_EQ(output.shape()[1], 128);
    EXPECT_EQ(output.shape()[2], 512);
}

TEST(RMSNormTest, CompareWithManualRMS) {
    // Manually compute RMS and compare
    RMSNorm rn(4, 1e-6);

    auto input_data = std::vector<float>{2.0f, 4.0f, 6.0f, 8.0f};
    auto input = Variable(from_data(input_data.data(), {1, 4}), false);

    auto output = rn(input);
    auto output_data = output.tensor().data<float>();

    // Manual calculation: RMS = sqrt((4+16+36+64)/4) = sqrt(30) ≈ 5.477
    float rms = std::sqrt(30.0f);

    EXPECT_NEAR(output_data[0], 2.0f / rms, 1e-4);
    EXPECT_NEAR(output_data[1], 4.0f / rms, 1e-4);
    EXPECT_NEAR(output_data[2], 6.0f / rms, 1e-4);
    EXPECT_NEAR(output_data[3], 8.0f / rms, 1e-4);
}

// ============================================================================
// GroupNorm Tests
// ============================================================================

TEST(GroupNormTest, ConstructorWithAffine) {
    GroupNorm gn(2, 4, 1e-5, true);
    auto params = gn.parameters();
    EXPECT_EQ(params.size(), 2);  // weight and bias
}

TEST(GroupNormTest, ConstructorWithoutAffine) {
    GroupNorm gn(2, 4, 1e-5, false);
    auto params = gn.parameters();
    EXPECT_EQ(params.size(), 0);
}

TEST(GroupNormTest, InvalidGroupsThrows) {
    // num_channels must be divisible by num_groups
    EXPECT_THROW(GroupNorm(3, 8), std::runtime_error);
}

TEST(GroupNormTest, ForwardNormalization) {
    // 2 groups, 4 channels, each group has 2 channels
    GroupNorm gn(2, 4, 1e-5, false);

    // Create input: [1, 4, 2, 2]
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
    auto input = Variable(from_data(input_data.data(), {1, 4, 2, 2}), false);

    auto output = gn(input);
    auto output_data = output.tensor().data<float>();

    // Each group should be normalized independently
    // Group 0: channels 0-1 (8 elements: 1-8)
    float mean_g0 = 0.0f;
    for (int i = 0; i < 8; i++) {
        mean_g0 += output_data[i];
    }
    mean_g0 /= 8.0f;
    EXPECT_NEAR(mean_g0, 0.0f, 1e-5);

    // Group 1: channels 2-3 (8 elements: 9-16)
    float mean_g1 = 0.0f;
    for (int i = 8; i < 16; i++) {
        mean_g1 += output_data[i];
    }
    mean_g1 /= 8.0f;
    EXPECT_NEAR(mean_g1, 0.0f, 1e-5);
}

TEST(GroupNormTest, ForwardWithAffine) {
    GroupNorm gn(2, 4, 1e-5, true);

    auto input_data = std::vector<float>(16, 1.0f);
    auto input = Variable(from_data(input_data.data(), {1, 4, 2, 2}), false);

    auto output = gn(input);
    EXPECT_EQ(output.shape()[0], 1);
    EXPECT_EQ(output.shape()[1], 4);
    EXPECT_EQ(output.shape()[2], 2);
    EXPECT_EQ(output.shape()[3], 2);
}

TEST(GroupNormTest, MultipleBatches) {
    GroupNorm gn(2, 4, 1e-5, false);

    auto input_data = std::vector<float>(2 * 4 * 2 * 2);
    for (size_t i = 0; i < input_data.size(); i++) {
        input_data[i] = static_cast<float>(i + 1);
    }
    auto input = Variable(from_data(input_data.data(), {2, 4, 2, 2}), false);

    auto output = gn(input);
    EXPECT_EQ(output.shape()[0], 2);
}

TEST(GroupNormTest, SingleGroup) {
    // Single group = LayerNorm behavior
    GroupNorm gn(1, 4, 1e-5, false);

    auto input_data = std::vector<float>{
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f,
        9.0f, 10.0f, 11.0f, 12.0f,
        13.0f, 14.0f, 15.0f, 16.0f
    };
    auto input = Variable(from_data(input_data.data(), {1, 4, 2, 2}), false);

    auto output = gn(input);
    auto output_data = output.tensor().data<float>();

    // All channels in one group
    float mean = 0.0f;
    for (int i = 0; i < 16; i++) {
        mean += output_data[i];
    }
    mean /= 16.0f;
    EXPECT_NEAR(mean, 0.0f, 1e-5);
}

TEST(GroupNormTest, GroupsEqualChannels) {
    // Each channel is its own group
    GroupNorm gn(4, 4, 1e-5, false);

    auto input_data = std::vector<float>{
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f,
        9.0f, 10.0f, 11.0f, 12.0f,
        13.0f, 14.0f, 15.0f, 16.0f
    };
    auto input = Variable(from_data(input_data.data(), {1, 4, 2, 2}), false);

    auto output = gn(input);
    auto output_data = output.tensor().data<float>();

    // Each channel normalized independently
    for (int c = 0; c < 4; c++) {
        float mean = 0.0f;
        for (int i = 0; i < 4; i++) {
            mean += output_data[c * 4 + i];
        }
        mean /= 4.0f;
        EXPECT_NEAR(mean, 0.0f, 1e-5);
    }
}

TEST(GroupNormTest, BackwardGradientFlow) {
    GroupNorm gn(2, 4, 1e-5, true);
    gn.train();

    auto input_data = std::vector<float>(16, 1.0f);
    auto input = Variable(from_data(input_data.data(), {1, 4, 2, 2}), true);

    auto output = gn(input);

    auto grad_output = ones({1, 4, 2, 2});
    output.backward(grad_output);

    EXPECT_TRUE(input.has_grad());
}

TEST(GroupNormTest, NumericalGradientCheck) {
    float eps = 1e-4;
    auto input_data = std::vector<float>{
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f,
        9.0f, 10.0f, 11.0f, 12.0f,
        13.0f, 14.0f, 15.0f, 16.0f
    };

    // Test gradient for specific output elements
    // Test a few representative output indices
    for (int out_idx : {0, 5, 10, 15}) {
        // Create fresh GroupNorm and input for each output index
        // (computation graph is consumed after backward())
        GroupNorm gn(2, 4, 1e-5, false);
        auto input = Variable(from_data(input_data.data(), {1, 4, 2, 2}), true);
        auto output = gn(input);

        // Backward with gradient only on one output element
        auto grad_output_data = std::vector<float>(16, 0.0f);
        grad_output_data[out_idx] = 1.0f;
        auto grad_output = from_data(grad_output_data.data(), {1, 4, 2, 2});
        output.backward(grad_output);

        ASSERT_TRUE(input.has_grad()) << "Input should have gradient after backward";
        auto analytical_grad = input.grad().value().data<float>();

        // Check numerical gradient for a few input elements in the same group
        for (int in_idx : {0, 5, 10, 15}) {
            GroupNorm gn_plus(2, 4, 1e-5, false);
            GroupNorm gn_minus(2, 4, 1e-5, false);

            auto input_plus = from_data(input_data.data(), {1, 4, 2, 2});
            auto input_plus_data = input_plus.data<float>();
            input_plus_data[in_idx] += eps;

            auto input_minus = from_data(input_data.data(), {1, 4, 2, 2});
            auto input_minus_data = input_minus.data<float>();
            input_minus_data[in_idx] -= eps;

            auto output_plus = gn_plus(Variable(input_plus, false)).tensor();
            auto output_minus = gn_minus(Variable(input_minus, false)).tensor();

            // Numerical gradient: d(output[out_idx])/d(input[in_idx])
            float numerical_grad = (output_plus.data<float>()[out_idx] -
                                   output_minus.data<float>()[out_idx]) / (2.0f * eps);

            EXPECT_NEAR(analytical_grad[in_idx], numerical_grad, 2.5e-3)
                << "Mismatch for d(output[" << out_idx << "])/d(input[" << in_idx << "])";
        }
    }
}

TEST(GroupNormTest, ParameterGradients) {
    GroupNorm gn(2, 4, 1e-5, true);
    gn.train();

    auto input_data = std::vector<float>(16);
    for (int i = 0; i < 16; i++) {
        input_data[i] = static_cast<float>(i + 1);
    }
    auto input = Variable(from_data(input_data.data(), {1, 4, 2, 2}), true);

    auto output = gn(input);

    auto grad_output = ones({1, 4, 2, 2});
    output.backward(grad_output);

    auto params = gn.named_parameters();
    EXPECT_EQ(params.size(), 2);

    for (auto& [name, param] : params) {
        EXPECT_TRUE(param->has_grad());
    }
}

// ============================================================================
// Integration Tests
// ============================================================================

TEST(NormalizationTest, TrainEvalMode) {
    LayerNorm ln({4}, 1e-5, true);

    ln.train();
    EXPECT_TRUE(ln.is_training());

    ln.eval();
    EXPECT_FALSE(ln.is_training());
}

TEST(NormalizationTest, ZeroGrad) {
    LayerNorm ln({4}, 1e-5, true);

    auto input_data = std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f};
    auto input = Variable(from_data(input_data.data(), {1, 4}), true);

    auto output = ln(input);
    output.backward(ones({1, 4}));

    EXPECT_TRUE(input.has_grad());

    ln.zero_grad();
    auto params = ln.parameters();
    for (auto& param : params) {
        if (param->has_grad()) {
            auto grad_data = param->grad().value().data<float>();
            for (int i = 0; i < 4; i++) {
                EXPECT_FLOAT_EQ(grad_data[i], 0.0f);
            }
        }
    }
}

TEST(NormalizationTest, LayerNormLargeInput) {
    // Test with realistic image-like dimensions
    LayerNorm ln({64, 8, 8}, 1e-5, true);

    auto input = Variable(randn({2, 64, 8, 8}), true);
    auto output = ln(input);

    EXPECT_EQ(output.shape()[0], 2);
    EXPECT_EQ(output.shape()[1], 64);
    EXPECT_EQ(output.shape()[2], 8);
    EXPECT_EQ(output.shape()[3], 8);
}

TEST(NormalizationTest, GroupNormLargeInput) {
    // Test with realistic image-like dimensions
    GroupNorm gn(8, 64, 1e-5, true);

    auto input = Variable(randn({2, 64, 8, 8}), true);
    auto output = gn(input);

    EXPECT_EQ(output.shape()[0], 2);
    EXPECT_EQ(output.shape()[1], 64);
    EXPECT_EQ(output.shape()[2], 8);
    EXPECT_EQ(output.shape()[3], 8);
}

TEST(NormalizationTest, CompareLayerNormGroupNorm) {
    // When GroupNorm has 1 group, it should behave like LayerNorm
    LayerNorm ln({4, 2, 2}, 1e-5, false);
    GroupNorm gn(1, 4, 1e-5, false);

    auto input_data = std::vector<float>(16);
    for (int i = 0; i < 16; i++) {
        input_data[i] = static_cast<float>(i + 1);
    }
    auto input1 = Variable(from_data(input_data.data(), {1, 4, 2, 2}), false);
    auto input2 = Variable(from_data(input_data.data(), {1, 4, 2, 2}), false);

    auto output_ln = ln(input1);
    auto output_gn = gn(input2);

    auto ln_data = output_ln.tensor().data<float>();
    auto gn_data = output_gn.tensor().data<float>();

    for (int i = 0; i < 16; i++) {
        EXPECT_NEAR(ln_data[i], gn_data[i], 1e-5);
    }
}

TEST(NormalizationTest, EpsilonEffect) {
    // Test that eps prevents division by zero
    LayerNorm ln1({4}, 1e-5, false);
    LayerNorm ln2({4}, 1e-1, false);

    // Input with all same values (zero variance)
    auto input_data = std::vector<float>{2.0f, 2.0f, 2.0f, 2.0f};
    auto input1 = Variable(from_data(input_data.data(), {1, 4}), false);
    auto input2 = Variable(from_data(input_data.data(), {1, 4}), false);

    auto output1 = ln1(input1);
    auto output2 = ln2(input2);

    // Both should produce valid output (all zeros since variance is 0)
    auto data1 = output1.tensor().data<float>();
    auto data2 = output2.tensor().data<float>();

    for (int i = 0; i < 4; i++) {
        EXPECT_TRUE(std::isfinite(data1[i]));
        EXPECT_TRUE(std::isfinite(data2[i]));
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
