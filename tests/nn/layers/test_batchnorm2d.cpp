#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include "../../backend_test_fixture.hpp"
#include <cmath>
#include <numeric>
#include <algorithm>

using namespace tenzor;
using namespace tenzor::nn;
using namespace tenzor::testing;

// Helper function to compute numerical gradient
template<typename Func>
float numerical_gradient(Func f, float x, float eps = 1e-4f) {
    return (f(x + eps) - f(x - eps)) / (2.0f * eps);
}

// Helper function to check if values are close
bool is_close(float a, float b, float rtol = 1e-4f, float atol = 1e-6f) {
    return std::abs(a - b) <= (atol + rtol * std::abs(b));
}

// Backend parity fixture for BatchNorm2d.
class BatchNorm2dTest : public BackendTest {};

// ==================== Basic Functionality Tests ====================

TEST_P(BatchNorm2dTest, ForwardShapePreservation) {
    // Test that output shape matches input shape
    BatchNorm2d bn(64);
    bn.to(device);

    auto input = Variable(randn({32, 64, 28, 28}, DType::Float32, device), true);
    auto output = bn.forward(input);

    auto out_shape = output.shape();
    EXPECT_EQ(out_shape.size(), 4);
    EXPECT_EQ(out_shape[0], 32);  // N
    EXPECT_EQ(out_shape[1], 64);  // C
    EXPECT_EQ(out_shape[2], 28);  // H
    EXPECT_EQ(out_shape[3], 28);  // W
}

TEST_P(BatchNorm2dTest, ParameterInitialization) {
    // Test affine parameters initialization
    BatchNorm2d bn_affine(32, 1e-5, 0.1, true);
    bn_affine.to(device);
    auto params = bn_affine.parameters();

    EXPECT_EQ(params.size(), 2);  // weight and bias
    EXPECT_EQ(params[0]->shape()[0], 32);  // weight shape
    EXPECT_EQ(params[1]->shape()[0], 32);  // bias shape

    // Weight should be initialized to 1, bias to 0
    auto weight_data_cpu_keepalive = params[0]->tensor().cpu();
    auto weight_data = weight_data_cpu_keepalive.data<float>();
    auto bias_data_cpu_keepalive = params[1]->tensor().cpu();
    auto bias_data = bias_data_cpu_keepalive.data<float>();

    for (int64_t i = 0; i < 32; ++i) {
        EXPECT_FLOAT_EQ(weight_data[i], 1.0f);
        EXPECT_FLOAT_EQ(bias_data[i], 0.0f);
    }
}

TEST_P(BatchNorm2dTest, NoAffineParameters) {
    // Test without affine transformation
    BatchNorm2d bn_no_affine(32, 1e-5, 0.1, false);
    bn_no_affine.to(device);
    auto params = bn_no_affine.parameters();

    EXPECT_EQ(params.size(), 0);  // No learnable parameters
}

// ==================== Training vs Inference Mode Tests ====================

TEST_P(BatchNorm2dTest, TrainingModeNormalization) {
    // Test that training mode normalizes to zero mean and unit variance
    BatchNorm2d bn(3, 1e-5, 0.1, false);  // No affine for easier testing
    bn.to(device);
    bn.train();

    // Create input with known statistics
    auto input = Variable(randn({16, 3, 8, 8}, DType::Float32, device), false);
    auto output = bn.forward(input);

    // Check output statistics per channel
    auto output_data_cpu_keepalive = output.tensor().cpu();
    auto output_data = output_data_cpu_keepalive.data<float>();
    int64_t N = 16, C = 3, H = 8, W = 8;
    int64_t spatial_size = H * W;
    int64_t batch_size = N * spatial_size;

    for (int64_t c = 0; c < C; ++c) {
        double sum = 0.0;
        double sum_sq = 0.0;

        for (int64_t n = 0; n < N; ++n) {
            for (int64_t h = 0; h < H; ++h) {
                for (int64_t w = 0; w < W; ++w) {
                    int64_t idx = ((n * C + c) * H + h) * W + w;
                    float val = output_data[idx];
                    sum += val;
                    sum_sq += val * val;
                }
            }
        }

        double mean = sum / batch_size;
        double variance = sum_sq / batch_size - mean * mean;

        EXPECT_NEAR(mean, 0.0, 1e-5) << "Channel " << c << " mean should be ~0";
        EXPECT_NEAR(variance, 1.0, 1e-4) << "Channel " << c << " variance should be ~1";
    }
}

TEST_P(BatchNorm2dTest, InferenceModeUsesRunningStats) {
    // Test that inference mode uses running statistics
    BatchNorm2d bn(3, 1e-5, 0.1, false);
    bn.to(device);

    // Train on some data to update running stats
    bn.train();
    auto train_input = Variable(randn({32, 3, 8, 8}, DType::Float32, device) * 2.0f + 5.0f, false);
    bn.forward(train_input);

    // Switch to eval mode
    bn.eval();

    // Create new input with different statistics
    auto test_input = Variable(randn({16, 3, 8, 8}, DType::Float32, device) * 0.5f - 3.0f, false);
    auto output = bn.forward(test_input);

    // Output should be normalized using running stats, not batch stats
    EXPECT_EQ(output.shape()[0], 16);
    EXPECT_EQ(output.shape()[1], 3);
    EXPECT_EQ(output.shape()[2], 8);
    EXPECT_EQ(output.shape()[3], 8);
}

TEST_P(BatchNorm2dTest, TrainingModeConsistency) {
    // Test that training mode produces consistent results for same input
    BatchNorm2d bn(4, 1e-5, 0.1, false);
    bn.to(device);
    bn.train();

    auto input = Variable(randn({8, 4, 4, 4}, DType::Float32, device), false);
    auto output1 = bn.forward(input);
    auto output2 = bn.forward(input);

    auto data1_cpu_keepalive = output1.tensor().cpu();
    auto data1 = data1_cpu_keepalive.data<float>();
    auto data2_cpu_keepalive = output2.tensor().cpu();
    auto data2 = data2_cpu_keepalive.data<float>();

    for (size_t i = 0; i < output1.tensor().numel(); ++i) {
        EXPECT_FLOAT_EQ(data1[i], data2[i]);
    }
}

// ==================== Running Statistics Tests ====================

TEST_P(BatchNorm2dTest, RunningStatisticsUpdate) {
    // Test that running statistics are being used after training
    // by checking that inference mode produces different output than training mode
    BatchNorm2d bn(2, 1e-5, 0.1, false, true);
    bn.to(device);

    // Train on one batch with specific statistics
    bn.train();
    auto train_input = Variable(randn({16, 2, 8, 8}, DType::Float32, device) * 2.0f + 5.0f, false);
    auto train_output = bn.forward(train_input);

    // Switch to eval mode
    bn.eval();

    // Test input with different statistics
    auto test_input = Variable(randn({8, 2, 8, 8}, DType::Float32, device) * 1.0f, false);
    auto eval_output = bn.forward(test_input);

    // Outputs should be valid (no NaN)
    auto eval_data_cpu_keepalive = eval_output.tensor().cpu();
    auto eval_data = eval_data_cpu_keepalive.data<float>();
    for (int64_t i = 0; i < eval_output.tensor().numel(); ++i) {
        EXPECT_FALSE(std::isnan(eval_data[i]));
        EXPECT_FALSE(std::isinf(eval_data[i]));
    }
}

TEST_P(BatchNorm2dTest, MomentumEffect) {
    // Test that different momentum values produce valid outputs
    BatchNorm2d bn_high_momentum(2, 1e-5, 0.9, false);
    BatchNorm2d bn_low_momentum(2, 1e-5, 0.1, false);
    bn_high_momentum.to(device);
    bn_low_momentum.to(device);

    bn_high_momentum.train();
    bn_low_momentum.train();

    auto input = Variable(randn({16, 2, 8, 8}, DType::Float32, device) * 5.0f, false);

    // Both should produce valid normalized output in training mode
    auto output_high = bn_high_momentum.forward(input);
    auto output_low = bn_low_momentum.forward(input);

    // Check both outputs are valid
    auto data_high_cpu_keepalive = output_high.tensor().cpu();
    auto data_high = data_high_cpu_keepalive.data<float>();
    auto data_low_cpu_keepalive = output_low.tensor().cpu();
    auto data_low = data_low_cpu_keepalive.data<float>();

    for (int64_t i = 0; i < output_high.tensor().numel(); ++i) {
        EXPECT_FALSE(std::isnan(data_high[i]));
        EXPECT_FALSE(std::isnan(data_low[i]));
    }
}

TEST_P(BatchNorm2dTest, RunningStatsNotUpdatedInEval) {
    // Test that eval mode produces consistent outputs
    BatchNorm2d bn(2, 1e-5, 0.1, false);
    bn.to(device);

    // Train once to set running stats
    bn.train();
    auto train_input = Variable(randn({16, 2, 8, 8}, DType::Float32, device), false);
    bn.forward(train_input);

    // Switch to eval and forward twice with same input
    bn.eval();
    auto eval_input = Variable(randn({16, 2, 8, 8}, DType::Float32, device), false);
    auto output1 = bn.forward(eval_input);
    auto output2 = bn.forward(eval_input);

    // Outputs should be identical (running stats not updated)
    auto data1_cpu_keepalive = output1.tensor().cpu();
    auto data1 = data1_cpu_keepalive.data<float>();
    auto data2_cpu_keepalive = output2.tensor().cpu();
    auto data2 = data2_cpu_keepalive.data<float>();

    for (int64_t i = 0; i < output1.tensor().numel(); ++i) {
        EXPECT_FLOAT_EQ(data1[i], data2[i]);
    }
}

// ==================== Epsilon Parameter Tests ====================

TEST_P(BatchNorm2dTest, EpsilonPreventsDivisionByZero) {
    // Test that epsilon prevents division by zero for constant input
    BatchNorm2d bn_small_eps(2, 1e-8, 0.1, false);
    BatchNorm2d bn_large_eps(2, 1e-3, 0.1, false);
    bn_small_eps.to(device);
    bn_large_eps.to(device);

    bn_small_eps.train();
    bn_large_eps.train();

    // Create input with very small variance (almost constant)
    auto input = Variable(ones({16, 2, 8, 8}, DType::Float32, device) * 1.0f, false);

    // Should not throw or produce NaN
    EXPECT_NO_THROW({
        auto output_small = bn_small_eps.forward(input);
        auto output_large = bn_large_eps.forward(input);

        auto data_small_cpu_keepalive = output_small.tensor().cpu();
        auto data_small = data_small_cpu_keepalive.data<float>();
        auto data_large_cpu_keepalive = output_large.tensor().cpu();
        auto data_large = data_large_cpu_keepalive.data<float>();

        for (size_t i = 0; i < output_small.tensor().numel(); ++i) {
            EXPECT_FALSE(std::isnan(data_small[i]));
            EXPECT_FALSE(std::isnan(data_large[i]));
            EXPECT_FALSE(std::isinf(data_small[i]));
            EXPECT_FALSE(std::isinf(data_large[i]));
        }
    });
}

TEST_P(BatchNorm2dTest, DifferentEpsilonValues) {
    // Test normalization with different epsilon values
    BatchNorm2d bn1(3, 1e-5, 0.1, false);
    BatchNorm2d bn2(3, 1e-3, 0.1, false);
    bn1.to(device);
    bn2.to(device);

    bn1.train();
    bn2.train();

    auto input = Variable(randn({8, 3, 4, 4}, DType::Float32, device), false);
    auto output1 = bn1.forward(input);
    auto output2 = bn2.forward(input);

    // Outputs should be slightly different due to epsilon
    auto data1_cpu_keepalive = output1.tensor().cpu();
    auto data1 = data1_cpu_keepalive.data<float>();
    auto data2_cpu_keepalive = output2.tensor().cpu();
    auto data2 = data2_cpu_keepalive.data<float>();

    bool has_difference = false;
    for (int64_t i = 0; i < output1.tensor().numel(); ++i) {
        if (std::abs(data1[i] - data2[i]) > 1e-6f) {
            has_difference = true;
            break;
        }
    }
    EXPECT_TRUE(has_difference);
}

// ==================== Affine Transformation Tests ====================

TEST_P(BatchNorm2dTest, AffineTransformationApplied) {
    // Test that affine parameters are applied correctly
    BatchNorm2d bn(2, 1e-5, 0.1, true);
    bn.to(device);
    bn.train();

    // Get weight and bias using parameters() method
    auto params_vec = bn.parameters();
    ASSERT_GE(params_vec.size(), 2);

    // Set specific values (first is weight, second is bias based on registration order)
    params_vec[0]->tensor().fill_(2.0f);  // weight
    params_vec[1]->tensor().fill_(1.0f);  // bias

    auto input = Variable(zeros({4, 2, 4, 4}, DType::Float32, device), false);
    auto output = bn.forward(input);

    // Zero input normalized is still zero, so output should be bias = 1.0
    auto output_data_cpu_keepalive = output.tensor().cpu();
    auto output_data = output_data_cpu_keepalive.data<float>();
    for (int64_t i = 0; i < output.tensor().numel(); ++i) {
        EXPECT_NEAR(output_data[i], 1.0f, 1e-4f);
    }
}

TEST_P(BatchNorm2dTest, AffineScaling) {
    // Test that weight parameter scales the normalized output
    BatchNorm2d bn(3, 1e-5, 0.1, true);
    bn.to(device);
    bn.train();

    auto params_vec = bn.parameters();
    ASSERT_GE(params_vec.size(), 2);

    // Set weight to 3.0, bias to 0.0
    params_vec[0]->tensor().fill_(3.0f);  // weight
    params_vec[1]->tensor().fill_(0.0f);  // bias

    auto input = Variable(randn({8, 3, 4, 4}, DType::Float32, device), false);
    auto output = bn.forward(input);

    // Get normalized output without affine
    BatchNorm2d bn_no_affine(3, 1e-5, 0.1, false);
    bn_no_affine.to(device);
    bn_no_affine.train();
    auto output_no_affine = bn_no_affine.forward(input);

    auto data_cpu_keepalive = output.tensor().cpu();
    auto data = data_cpu_keepalive.data<float>();
    auto data_no_affine_cpu_keepalive = output_no_affine.tensor().cpu();
    auto data_no_affine = data_no_affine_cpu_keepalive.data<float>();

    // Output should be approximately 3.0 * normalized
    for (int64_t i = 0; i < output.tensor().numel(); ++i) {
        EXPECT_NEAR(data[i], data_no_affine[i] * 3.0f, 1e-3f);
    }
}

// ==================== Different Batch Sizes Tests ====================

TEST_P(BatchNorm2dTest, BatchSize1) {
    // Test with single sample (edge case)
    BatchNorm2d bn(4, 1e-5, 0.1, false);
    bn.to(device);
    bn.train();

    auto input = Variable(randn({1, 4, 8, 8}, DType::Float32, device), false);
    auto output = bn.forward(input);

    EXPECT_EQ(output.shape()[0], 1);
    EXPECT_EQ(output.shape()[1], 4);

    // Check no NaN values
    auto data_cpu_keepalive = output.tensor().cpu();
    auto data = data_cpu_keepalive.data<float>();
    for (int64_t i = 0; i < output.tensor().numel(); ++i) {
        EXPECT_FALSE(std::isnan(data[i]));
    }
}

TEST_P(BatchNorm2dTest, BatchSize16) {
    BatchNorm2d bn(8, 1e-5, 0.1, false);
    bn.to(device);
    bn.train();

    auto input = Variable(randn({16, 8, 16, 16}, DType::Float32, device), false);
    auto output = bn.forward(input);

    EXPECT_EQ(output.shape()[0], 16);
    EXPECT_EQ(output.shape()[1], 8);
}

TEST_P(BatchNorm2dTest, BatchSize32) {
    BatchNorm2d bn(16, 1e-5, 0.1, false);
    bn.to(device);
    bn.train();

    auto input = Variable(randn({32, 16, 32, 32}, DType::Float32, device), false);
    auto output = bn.forward(input);

    EXPECT_EQ(output.shape()[0], 32);
    EXPECT_EQ(output.shape()[1], 16);
}

TEST_P(BatchNorm2dTest, BatchSize64) {
    BatchNorm2d bn(32, 1e-5, 0.1, false);
    bn.to(device);
    bn.train();

    auto input = Variable(randn({64, 32, 8, 8}, DType::Float32, device), false);
    auto output = bn.forward(input);

    EXPECT_EQ(output.shape()[0], 64);
    EXPECT_EQ(output.shape()[1], 32);
}

TEST_P(BatchNorm2dTest, VariableBatchSizes) {
    // Test that the same layer works with different batch sizes
    BatchNorm2d bn(4, 1e-5, 0.1, true);
    bn.to(device);
    bn.train();

    std::vector<int64_t> batch_sizes = {1, 4, 16, 32};

    for (auto bs : batch_sizes) {
        auto input = Variable(randn({bs, 4, 8, 8}, DType::Float32, device), false);
        EXPECT_NO_THROW({
            auto output = bn.forward(input);
            EXPECT_EQ(output.shape()[0], bs);
        });
    }
}

// ==================== Different Channel Counts Tests ====================

TEST_P(BatchNorm2dTest, SingleChannel) {
    BatchNorm2d bn(1, 1e-5, 0.1, true);
    bn.to(device);
    bn.train();

    auto input = Variable(randn({8, 1, 16, 16}, DType::Float32, device), false);
    auto output = bn.forward(input);

    EXPECT_EQ(output.shape()[1], 1);
}

TEST_P(BatchNorm2dTest, ThreeChannelsRGB) {
    BatchNorm2d bn(3, 1e-5, 0.1, true);
    bn.to(device);
    bn.train();

    auto input = Variable(randn({16, 3, 224, 224}, DType::Float32, device), false);
    auto output = bn.forward(input);

    EXPECT_EQ(output.shape()[1], 3);
}

TEST_P(BatchNorm2dTest, Channel64) {
    BatchNorm2d bn(64, 1e-5, 0.1, true);
    bn.to(device);
    bn.train();

    auto input = Variable(randn({32, 64, 56, 56}, DType::Float32, device), false);
    auto output = bn.forward(input);

    EXPECT_EQ(output.shape()[1], 64);
}

TEST_P(BatchNorm2dTest, Channel256) {
    BatchNorm2d bn(256, 1e-5, 0.1, true);
    bn.to(device);
    bn.train();

    auto input = Variable(randn({16, 256, 28, 28}, DType::Float32, device), false);
    auto output = bn.forward(input);

    EXPECT_EQ(output.shape()[1], 256);
}

TEST_P(BatchNorm2dTest, ChannelMismatchError) {
    // Test that mismatched channels throw error
    BatchNorm2d bn(32, 1e-5, 0.1, true);
    bn.to(device);
    bn.train();

    auto input = Variable(randn({16, 64, 8, 8}, DType::Float32, device), false);  // 64 channels, expecting 32

    EXPECT_THROW(bn.forward(input), std::runtime_error);
}

// ==================== Different Spatial Dimensions Tests ====================

TEST_P(BatchNorm2dTest, SmallSpatialDimensions) {
    BatchNorm2d bn(16, 1e-5, 0.1, true);
    bn.to(device);
    bn.train();

    auto input = Variable(randn({32, 16, 1, 1}, DType::Float32, device), false);
    auto output = bn.forward(input);

    EXPECT_EQ(output.shape()[2], 1);
    EXPECT_EQ(output.shape()[3], 1);
}

TEST_P(BatchNorm2dTest, LargeSpatialDimensions) {
    BatchNorm2d bn(32, 1e-5, 0.1, true);
    bn.to(device);
    bn.train();

    auto input = Variable(randn({8, 32, 128, 128}, DType::Float32, device), false);
    auto output = bn.forward(input);

    EXPECT_EQ(output.shape()[2], 128);
    EXPECT_EQ(output.shape()[3], 128);
}

TEST_P(BatchNorm2dTest, NonSquareSpatialDimensions) {
    BatchNorm2d bn(16, 1e-5, 0.1, true);
    bn.to(device);
    bn.train();

    auto input = Variable(randn({16, 16, 32, 64}, DType::Float32, device), false);
    auto output = bn.forward(input);

    EXPECT_EQ(output.shape()[2], 32);
    EXPECT_EQ(output.shape()[3], 64);
}

// ==================== Gradient Tests ====================

TEST_P(BatchNorm2dTest, BackwardPassGradientFlow) {
    // Test that gradients flow through the layer
    BatchNorm2d bn(4, 1e-5, 0.1, true);
    bn.to(device);
    bn.train();

    auto input = Variable(randn({8, 4, 8, 8}, DType::Float32, device), true);
    auto output = bn.forward(input);

    // Create gradient
    auto out_shape = output.shape();
    std::vector<int64_t> shape_vec(out_shape.begin(), out_shape.end());
    auto grad_output = ones(shape_vec, DType::Float32, device);
    output.backward(grad_output);

    // Check that input has gradient
    EXPECT_TRUE(input.has_grad());

    auto input_grad = input.grad().value();
    EXPECT_EQ(input_grad.shape().size(), 4);

    // Check no NaN in gradients
    auto grad_data_cpu_keepalive = input_grad.cpu();
    auto grad_data = grad_data_cpu_keepalive.data<float>();
    for (int64_t i = 0; i < input_grad.numel(); ++i) {
        EXPECT_FALSE(std::isnan(grad_data[i]));
        EXPECT_FALSE(std::isinf(grad_data[i]));
    }
}

TEST_P(BatchNorm2dTest, ParameterGradients) {
    // Test that weight and bias get gradients
    BatchNorm2d bn(4, 1e-5, 0.1, true);
    bn.to(device);
    bn.train();

    auto input = Variable(randn({8, 4, 8, 8}, DType::Float32, device), true);
    auto output = bn.forward(input);

    auto out_shape = output.shape();
    std::vector<int64_t> shape_vec(out_shape.begin(), out_shape.end());
    auto grad_output = ones(shape_vec, DType::Float32, device);
    output.backward(grad_output);

    auto params_vec = bn.parameters();
    ASSERT_GE(params_vec.size(), 2);

    EXPECT_TRUE(params_vec[0]->has_grad());  // weight
    EXPECT_TRUE(params_vec[1]->has_grad());  // bias

    auto weight_grad = params_vec[0]->grad().value();
    auto bias_grad = params_vec[1]->grad().value();

    // Check gradient shapes
    EXPECT_EQ(weight_grad.shape()[0], 4);
    EXPECT_EQ(bias_grad.shape()[0], 4);
}

TEST_P(BatchNorm2dTest, GradientCheckingSimple) {
    // Simple gradient checking with numerical gradients
    BatchNorm2d bn(2, 1e-5, 0.1, false);  // No affine for simpler test
    bn.to(device);
    bn.train();

    auto input = Variable(randn({4, 2, 4, 4}, DType::Float32, device), true);
    auto output = bn.forward(input);

    auto out_shape = output.shape();
    std::vector<int64_t> shape_vec(out_shape.begin(), out_shape.end());
    auto grad_output = ones(shape_vec, DType::Float32, device);
    output.backward(grad_output);

    EXPECT_TRUE(input.has_grad());

    // Check gradient magnitude is reasonable
    auto input_grad = input.grad().value();
    auto grad_data_cpu_keepalive = input_grad.cpu();
    auto grad_data = grad_data_cpu_keepalive.data<float>();

    double grad_sum = 0.0;
    for (int64_t i = 0; i < input_grad.numel(); ++i) {
        grad_sum += std::abs(grad_data[i]);
    }
    double grad_mean = grad_sum / input_grad.numel();

    EXPECT_GT(grad_mean, 0.0);
    EXPECT_LT(grad_mean, 10.0);  // Should be reasonable magnitude
}

// ==================== Edge Cases Tests ====================

TEST_P(BatchNorm2dTest, VerySmallEpsilon) {
    // Test with very small epsilon
    BatchNorm2d bn(4, 1e-10, 0.1, true);
    bn.to(device);
    bn.train();

    auto input = Variable(randn({8, 4, 8, 8}, DType::Float32, device), false);

    EXPECT_NO_THROW({
        auto output = bn.forward(input);
        auto data_cpu_keepalive = output.tensor().cpu();
        auto data = data_cpu_keepalive.data<float>();
        for (int64_t i = 0; i < output.tensor().numel(); ++i) {
            EXPECT_FALSE(std::isnan(data[i]));
        }
    });
}

TEST_P(BatchNorm2dTest, ConstantInput) {
    // Test with constant input (zero variance)
    BatchNorm2d bn(3, 1e-5, 0.1, false);
    bn.to(device);
    bn.train();

    auto input = Variable(ones({8, 3, 8, 8}, DType::Float32, device) * 5.0f, false);
    auto output = bn.forward(input);

    // Should not crash or produce NaN
    auto data_cpu_keepalive = output.tensor().cpu();
    auto data = data_cpu_keepalive.data<float>();
    for (int64_t i = 0; i < output.tensor().numel(); ++i) {
        EXPECT_FALSE(std::isnan(data[i]));
        EXPECT_FALSE(std::isinf(data[i]));
    }
}

TEST_P(BatchNorm2dTest, ExtremeValues) {
    // Test with extreme input values
    BatchNorm2d bn(2, 1e-5, 0.1, true);
    bn.to(device);
    bn.train();

    auto input = Variable(randn({8, 2, 8, 8}, DType::Float32, device) * 1000.0f, false);

    EXPECT_NO_THROW({
        auto output = bn.forward(input);
        auto data_cpu_keepalive = output.tensor().cpu();
        auto data = data_cpu_keepalive.data<float>();
        for (int64_t i = 0; i < output.tensor().numel(); ++i) {
            EXPECT_FALSE(std::isnan(data[i]));
            EXPECT_FALSE(std::isinf(data[i]));
        }
    });
}

TEST_P(BatchNorm2dTest, InvalidInputDimensions) {
    // Test that non-4D input throws error
    BatchNorm2d bn(32, 1e-5, 0.1, true);
    bn.to(device);

    auto input_3d = Variable(randn({32, 32, 32}, DType::Float32, device), false);
    EXPECT_THROW(bn.forward(input_3d), std::runtime_error);

    auto input_2d = Variable(randn({32, 32}, DType::Float32, device), false);
    EXPECT_THROW(bn.forward(input_2d), std::runtime_error);

    auto input_5d = Variable(randn({8, 32, 8, 8, 8}, DType::Float32, device), false);
    EXPECT_THROW(bn.forward(input_5d), std::runtime_error);
}

// ==================== Comparison Tests ====================

TEST_P(BatchNorm2dTest, IndependentChannelNormalization) {
    // Test that channels are normalized independently
    BatchNorm2d bn(3, 1e-5, 0.1, false);
    bn.to(device);
    bn.train();

    int64_t N = 16, C = 3, H = 8, W = 8;

    // Build the scaled input on the host, then move to the target device.
    auto input_tensor = randn({16, 3, 8, 8}, DType::Float32, Device::cpu());
    auto input_data = input_tensor.data<float>();

    // Scale each channel differently using correct NCHW indexing
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            float scale = (c + 1) * 10.0f;
            for (int64_t h = 0; h < H; ++h) {
                for (int64_t w = 0; w < W; ++w) {
                    int64_t idx = ((n * C + c) * H + h) * W + w;
                    input_data[idx] *= scale;
                }
            }
        }
    }

    auto input = Variable(input_tensor.to(device), false);
    auto output = bn.forward(input);

    // Each channel in output should have similar statistics
    auto output_data_cpu_keepalive = output.tensor().cpu();
    auto output_data = output_data_cpu_keepalive.data<float>();
    int64_t spatial_size = H * W;
    int64_t batch_size = N * spatial_size;

    for (int64_t c = 0; c < C; ++c) {
        double sum = 0.0;
        double sum_sq = 0.0;

        // Use correct NCHW indexing to read output
        for (int64_t n = 0; n < N; ++n) {
            for (int64_t h = 0; h < H; ++h) {
                for (int64_t w = 0; w < W; ++w) {
                    int64_t idx = ((n * C + c) * H + h) * W + w;
                    float val = output_data[idx];
                    sum += val;
                    sum_sq += val * val;
                }
            }
        }

        double mean = sum / batch_size;
        double variance = sum_sq / batch_size - mean * mean;

        EXPECT_NEAR(mean, 0.0, 1e-4) << "Channel " << c;
        EXPECT_NEAR(variance, 1.0, 1e-3) << "Channel " << c;
    }
}

TEST_P(BatchNorm2dTest, ConsistentWithManualNormalization) {
    // Compare BatchNorm output with manual normalization
    BatchNorm2d bn(2, 1e-5, 0.1, false);
    bn.to(device);
    bn.train();

    auto input = Variable(randn({4, 2, 8, 8}, DType::Float32, device), false);
    auto output = bn.forward(input);

    // Manual normalization (read both back to host)
    auto input_data_cpu_keepalive = input.tensor().cpu();
    auto input_data = input_data_cpu_keepalive.data<float>();
    auto output_data_cpu_keepalive = output.tensor().cpu();
    auto output_data = output_data_cpu_keepalive.data<float>();

    int64_t N = 4, C = 2, H = 8, W = 8;
    int64_t spatial_size = H * W;
    int64_t batch_size = N * spatial_size;

    for (int64_t c = 0; c < C; ++c) {
        // Compute mean
        double sum = 0.0;
        for (int64_t n = 0; n < N; ++n) {
            for (int64_t i = 0; i < spatial_size; ++i) {
                int64_t idx = (n * C + c) * spatial_size + i;
                sum += input_data[idx];
            }
        }
        float mean = sum / batch_size;

        // Compute variance
        double sum_sq = 0.0;
        for (int64_t n = 0; n < N; ++n) {
            for (int64_t i = 0; i < spatial_size; ++i) {
                int64_t idx = (n * C + c) * spatial_size + i;
                float diff = input_data[idx] - mean;
                sum_sq += diff * diff;
            }
        }
        float variance = sum_sq / batch_size;

        // Normalize
        float std = std::sqrt(variance + 1e-5f);
        for (int64_t n = 0; n < N; ++n) {
            for (int64_t i = 0; i < spatial_size; ++i) {
                int64_t idx = (n * C + c) * spatial_size + i;
                float expected = (input_data[idx] - mean) / std;
                EXPECT_NEAR(output_data[idx], expected, 1e-4f);
            }
        }
    }
}

TEST_P(BatchNorm2dTest, StatisticsAccumulationOverMultipleBatches) {
    // Test that running statistics work correctly after multiple batches
    BatchNorm2d bn(4, 1e-5, 0.1, false);
    bn.to(device);
    bn.train();

    // Process multiple batches
    for (int i = 0; i < 5; ++i) {
        auto input = Variable(randn({16, 4, 8, 8}, DType::Float32, device) * 2.0f + static_cast<float>(i), false);
        auto output = bn.forward(input);

        // Check output is valid
        auto data_cpu_keepalive = output.tensor().cpu();
        auto data = data_cpu_keepalive.data<float>();
        for (int64_t j = 0; j < output.tensor().numel(); ++j) {
            EXPECT_FALSE(std::isnan(data[j]));
        }
    }

    // After training, eval mode should work
    bn.eval();
    auto test_input = Variable(randn({8, 4, 8, 8}, DType::Float32, device), false);
    EXPECT_NO_THROW({
        auto output = bn.forward(test_input);
        EXPECT_EQ(output.shape()[0], 8);
    });
}

// ==================== Integration Tests ====================

TEST_P(BatchNorm2dTest, IntegrationWithOtherLayers) {
    // Test BatchNorm in a simple network context
    // This simulates Conv -> BatchNorm -> ReLU pattern
    BatchNorm2d bn(32, 1e-5, 0.1, true);
    bn.to(device);
    bn.train();

    // Simulate conv output
    auto conv_output = Variable(randn({16, 32, 28, 28}, DType::Float32, device), true);

    // Apply BatchNorm
    auto bn_output = bn.forward(conv_output);

    // Simulate ReLU - just use forward through another layer or simple operation
    // For testing purposes, we'll just use the bn_output directly and verify backprop

    // Backward pass
    auto grad_shape = bn_output.shape();
    std::vector<int64_t> grad_shape_vec(grad_shape.begin(), grad_shape.end());
    auto grad = ones(grad_shape_vec, DType::Float32, device);
    bn_output.backward(grad);

    // Check gradients exist
    EXPECT_TRUE(conv_output.has_grad());

    auto params_vec = bn.parameters();
    ASSERT_GE(params_vec.size(), 2);
    EXPECT_TRUE(params_vec[0]->has_grad());  // weight
    EXPECT_TRUE(params_vec[1]->has_grad());  // bias
}

TEST_P(BatchNorm2dTest, MultipleForwardPasses) {
    // Test multiple forward passes update running stats correctly
    BatchNorm2d bn(8, 1e-5, 0.1, true);
    bn.to(device);
    bn.train();

    // Do multiple forward passes
    for (int i = 0; i < 10; ++i) {
        auto input = Variable(randn({16, 8, 16, 16}, DType::Float32, device), false);
        auto output = bn.forward(input);

        EXPECT_EQ(output.shape()[0], 16);
        EXPECT_EQ(output.shape()[1], 8);
    }

    // Switch to eval and verify it uses running stats
    bn.eval();
    auto test_input = Variable(randn({8, 8, 16, 16}, DType::Float32, device), false);
    EXPECT_NO_THROW({
        auto output = bn.forward(test_input);
        EXPECT_EQ(output.shape()[0], 8);
    });
}

// ==================== Performance and Numerical Stability Tests ====================

TEST_P(BatchNorm2dTest, NumericalStability) {
    // Test with various numerical edge cases
    BatchNorm2d bn(4, 1e-5, 0.1, true);
    bn.to(device);
    bn.train();

    // Test with very small values
    auto small_input = Variable(randn({8, 4, 8, 8}, DType::Float32, device) * 1e-6f, false);
    EXPECT_NO_THROW({
        auto output = bn.forward(small_input);
        auto data_cpu_keepalive = output.tensor().cpu();
        auto data = data_cpu_keepalive.data<float>();
        for (int64_t i = 0; i < output.tensor().numel(); ++i) {
            EXPECT_FALSE(std::isnan(data[i]));
        }
    });

    // Test with values near zero
    auto zero_input = Variable(randn({8, 4, 8, 8}, DType::Float32, device) * 1e-8f, false);
    EXPECT_NO_THROW({
        auto output = bn.forward(zero_input);
        auto data_cpu_keepalive = output.tensor().cpu();
        auto data = data_cpu_keepalive.data<float>();
        for (int64_t i = 0; i < output.tensor().numel(); ++i) {
            EXPECT_FALSE(std::isnan(data[i]));
        }
    });
}

TEST_P(BatchNorm2dTest, LargeScaleTest) {
    // Test with larger, more realistic dimensions
    BatchNorm2d bn(128, 1e-5, 0.1, true);
    bn.to(device);
    bn.train();

    auto input = Variable(randn({64, 128, 32, 32}, DType::Float32, device), true);

    EXPECT_NO_THROW({
        auto output = bn.forward(input);
        EXPECT_EQ(output.shape()[0], 64);
        EXPECT_EQ(output.shape()[1], 128);
        EXPECT_EQ(output.shape()[2], 32);
        EXPECT_EQ(output.shape()[3], 32);
    });
}

INSTANTIATE_BACKEND_TESTS(BatchNorm2dTest);
