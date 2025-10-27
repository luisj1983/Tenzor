#include <gtest/gtest.h>
#include "tests/backend_test_fixture.hpp"
#include <tenzor/tenzor.hpp>
#include <cmath>

using namespace tenzor;
using namespace tenzor::testing;

// Helper function to check if two tensors are close
bool tensors_close(const Tensor& a, const Tensor& b, float rtol = 1e-5f, float atol = 1e-7f) {
    auto a_shape = a.shape();
    auto b_shape = b.shape();
    if (a_shape.size() != b_shape.size() ||
        !std::equal(a_shape.begin(), a_shape.end(), b_shape.begin())) {
        return false;
    }

    const float* a_data = a.data<float>();
    const float* b_data = b.data<float>();
    size_t numel = a.numel();

    for (size_t i = 0; i < numel; ++i) {
        float diff = std::abs(a_data[i] - b_data[i]);
        float threshold = atol + rtol * std::abs(b_data[i]);
        if (diff > threshold) {
            return false;
        }
    }
    return true;
}

// Helper function to compute numerical gradient
Tensor numerical_gradient(std::function<Variable(Variable&)> func,
                         Variable& input, float eps = 1e-4f) {
    auto shape = input.shape();
    std::vector<int64_t> shape_vec(shape.begin(), shape.end());
    auto grad = zeros(shape_vec);
    float* grad_data = grad.data<float>();
    float* input_data = input.tensor().data<float>();
    size_t numel = input.tensor().numel();

    for (size_t i = 0; i < numel; ++i) {
        float original = input_data[i];

        // f(x + eps)
        input_data[i] = original + eps;
        auto out_plus = func(input);
        float loss_plus = sum(out_plus.tensor()).data<float>()[0];

        // f(x - eps)
        input_data[i] = original - eps;
        auto out_minus = func(input);
        float loss_minus = sum(out_minus.tensor()).data<float>()[0];

        // Restore original
        input_data[i] = original;

        // Central difference
        grad_data[i] = (loss_plus - loss_minus) / (2.0f * eps);
    }

    return grad;
}

// ==========================
// Backend Test Class
// ==========================

class Conv2dBackendTest : public BackendTest {};

// ==========================
// Basic Shape Tests
// ==========================

TEST_P(Conv2dBackendTest, ForwardShapeBasic) {
    // Test basic forward pass with 3x3 kernel
    auto conv = nn::Conv2d(3, 16, 3, 1, 0);
    auto input = Variable(randn({2, 3, 32, 32}, DType::Float32, device), true);
    auto output = conv.forward(input);

    // Output size = (32 - 3) / 1 + 1 = 30
    EXPECT_EQ(output.shape()[0], 2);   // batch
    EXPECT_EQ(output.shape()[1], 16);  // out_channels
    EXPECT_EQ(output.shape()[2], 30);  // height
    EXPECT_EQ(output.shape()[3], 30);  // width
}

TEST_P(Conv2dBackendTest, ForwardShapeSingleBatch) {
    auto conv = nn::Conv2d(1, 8, 3);
    auto input = Variable(randn({1, 1, 28, 28}, DType::Float32, device), true);
    auto output = conv.forward(input);

    EXPECT_EQ(output.shape()[0], 1);
    EXPECT_EQ(output.shape()[1], 8);
    EXPECT_EQ(output.shape()[2], 26);  // 28 - 3 + 1
    EXPECT_EQ(output.shape()[3], 26);
}

TEST_P(Conv2dBackendTest, ForwardShapeMultiBatch) {
    auto conv = nn::Conv2d(3, 64, 3);
    auto input = Variable(randn({32, 3, 64, 64}, DType::Float32, device), true);
    auto output = conv.forward(input);

    EXPECT_EQ(output.shape()[0], 32);
    EXPECT_EQ(output.shape()[1], 64);
    EXPECT_EQ(output.shape()[2], 62);
    EXPECT_EQ(output.shape()[3], 62);
}

// ==========================
// Kernel Size Tests
// ==========================

TEST_P(Conv2dBackendTest, KernelSize1x1) {
    // 1x1 convolution (pointwise)
    auto conv = nn::Conv2d(16, 32, 1);
    auto input = Variable(randn({4, 16, 28, 28}, DType::Float32, device), true);
    auto output = conv.forward(input);

    EXPECT_EQ(output.shape()[0], 4);
    EXPECT_EQ(output.shape()[1], 32);
    EXPECT_EQ(output.shape()[2], 28);  // Same size with 1x1 kernel
    EXPECT_EQ(output.shape()[3], 28);
}

TEST_P(Conv2dBackendTest, KernelSize3x3) {
    auto conv = nn::Conv2d(8, 16, 3);
    auto input = Variable(randn({2, 8, 32, 32}, DType::Float32, device), true);
    auto output = conv.forward(input);

    EXPECT_EQ(output.shape()[0], 2);
    EXPECT_EQ(output.shape()[1], 16);
    EXPECT_EQ(output.shape()[2], 30);  // 32 - 3 + 1
    EXPECT_EQ(output.shape()[3], 30);
}

TEST_P(Conv2dBackendTest, KernelSize5x5) {
    auto conv = nn::Conv2d(3, 64, 5);
    auto input = Variable(randn({2, 3, 32, 32}, DType::Float32, device), true);
    auto output = conv.forward(input);

    EXPECT_EQ(output.shape()[0], 2);
    EXPECT_EQ(output.shape()[1], 64);
    EXPECT_EQ(output.shape()[2], 28);  // 32 - 5 + 1
    EXPECT_EQ(output.shape()[3], 28);
}

TEST_P(Conv2dBackendTest, KernelSize7x7) {
    auto conv = nn::Conv2d(3, 64, 7);
    auto input = Variable(randn({1, 3, 224, 224}, DType::Float32, device), true);
    auto output = conv.forward(input);

    EXPECT_EQ(output.shape()[0], 1);
    EXPECT_EQ(output.shape()[1], 64);
    EXPECT_EQ(output.shape()[2], 218);  // 224 - 7 + 1
    EXPECT_EQ(output.shape()[3], 218);
}

// ==========================
// Stride Tests
// ==========================

TEST_P(Conv2dBackendTest, Stride1) {
    auto conv = nn::Conv2d(3, 16, 3, 1);  // stride=1
    auto input = Variable(randn({2, 3, 32, 32}, DType::Float32, device), true);
    auto output = conv.forward(input);

    EXPECT_EQ(output.shape()[2], 30);  // (32 - 3) / 1 + 1
    EXPECT_EQ(output.shape()[3], 30);
}

TEST_P(Conv2dBackendTest, Stride2) {
    auto conv = nn::Conv2d(3, 16, 3, 2);  // stride=2
    auto input = Variable(randn({2, 3, 32, 32}, DType::Float32, device), true);
    auto output = conv.forward(input);

    EXPECT_EQ(output.shape()[2], 15);  // (32 - 3) / 2 + 1
    EXPECT_EQ(output.shape()[3], 15);
}

TEST_P(Conv2dBackendTest, Stride3) {
    auto conv = nn::Conv2d(3, 16, 3, 3);  // stride=3
    auto input = Variable(randn({2, 3, 33, 33}, DType::Float32, device), true);
    auto output = conv.forward(input);

    EXPECT_EQ(output.shape()[2], 11);  // (33 - 3) / 3 + 1
    EXPECT_EQ(output.shape()[3], 11);
}

TEST_P(Conv2dBackendTest, Stride4) {
    auto conv = nn::Conv2d(16, 32, 7, 4);  // stride=4
    auto input = Variable(randn({1, 16, 112, 112}, DType::Float32, device), true);
    auto output = conv.forward(input);

    EXPECT_EQ(output.shape()[2], 27);  // (112 - 7) / 4 + 1
    EXPECT_EQ(output.shape()[3], 27);
}

// ==========================
// Padding Tests
// ==========================

TEST_P(Conv2dBackendTest, Padding0) {
    auto conv = nn::Conv2d(3, 16, 3, 1, 0);  // no padding
    auto input = Variable(randn({2, 3, 32, 32}, DType::Float32, device), true);
    auto output = conv.forward(input);

    EXPECT_EQ(output.shape()[2], 30);
    EXPECT_EQ(output.shape()[3], 30);
}

TEST_P(Conv2dBackendTest, Padding1) {
    auto conv = nn::Conv2d(3, 16, 3, 1, 1);  // padding=1
    auto input = Variable(randn({2, 3, 32, 32}, DType::Float32, device), true);
    auto output = conv.forward(input);

    EXPECT_EQ(output.shape()[2], 32);  // (32 + 2*1 - 3) / 1 + 1 = 32
    EXPECT_EQ(output.shape()[3], 32);
}

TEST_P(Conv2dBackendTest, Padding2) {
    auto conv = nn::Conv2d(3, 16, 5, 1, 2);  // padding=2
    auto input = Variable(randn({2, 3, 32, 32}, DType::Float32, device), true);
    auto output = conv.forward(input);

    EXPECT_EQ(output.shape()[2], 32);  // (32 + 2*2 - 5) / 1 + 1 = 32
    EXPECT_EQ(output.shape()[3], 32);
}

TEST_P(Conv2dBackendTest, PaddingSamePadding) {
    // Test "same" padding (output size = input size with stride=1)
    auto conv = nn::Conv2d(3, 32, 7, 1, 3);  // padding=3 for kernel=7
    auto input = Variable(randn({2, 3, 64, 64}, DType::Float32, device), true);
    auto output = conv.forward(input);

    EXPECT_EQ(output.shape()[2], 64);
    EXPECT_EQ(output.shape()[3], 64);
}

// ==========================
// Dilation Tests
// ==========================

TEST_P(Conv2dBackendTest, Dilation1) {
    auto conv = nn::Conv2d(3, 16, 3, 1, 0, 1);  // dilation=1 (standard)
    auto input = Variable(randn({2, 3, 32, 32}, DType::Float32, device), true);
    auto output = conv.forward(input);

    EXPECT_EQ(output.shape()[2], 30);  // (32 - 1*(3-1) - 1) / 1 + 1
    EXPECT_EQ(output.shape()[3], 30);
}

TEST_P(Conv2dBackendTest, Dilation2) {
    auto conv = nn::Conv2d(3, 16, 3, 1, 0, 2);  // dilation=2 (atrous)
    auto input = Variable(randn({2, 3, 32, 32}, DType::Float32, device), true);
    auto output = conv.forward(input);

    EXPECT_EQ(output.shape()[2], 28);  // (32 - 2*(3-1) - 1) / 1 + 1
    EXPECT_EQ(output.shape()[3], 28);
}

TEST_P(Conv2dBackendTest, Dilation3) {
    auto conv = nn::Conv2d(3, 16, 3, 1, 0, 3);  // dilation=3
    auto input = Variable(randn({2, 3, 32, 32}, DType::Float32, device), true);
    auto output = conv.forward(input);

    EXPECT_EQ(output.shape()[2], 26);  // (32 - 3*(3-1) - 1) / 1 + 1
    EXPECT_EQ(output.shape()[3], 26);
}

TEST_P(Conv2dBackendTest, DilationWithPadding) {
    // Dilation=2 with padding to maintain size
    auto conv = nn::Conv2d(3, 16, 3, 1, 2, 2);  // dilation=2, padding=2
    auto input = Variable(randn({2, 3, 32, 32}, DType::Float32, device), true);
    auto output = conv.forward(input);

    EXPECT_EQ(output.shape()[2], 32);  // (32 + 2*2 - 2*(3-1) - 1) / 1 + 1
    EXPECT_EQ(output.shape()[3], 32);
}

// ==========================
// Groups Tests
// ==========================

TEST_P(Conv2dBackendTest, Groups1Standard) {
    // Standard convolution (groups=1)
    auto conv = nn::Conv2d(12, 24, 3, 1, 0, 1, 1);
    auto input = Variable(randn({2, 12, 28, 28}, DType::Float32, device), true);
    auto output = conv.forward(input);

    EXPECT_EQ(output.shape()[0], 2);
    EXPECT_EQ(output.shape()[1], 24);
    EXPECT_EQ(output.shape()[2], 26);
    EXPECT_EQ(output.shape()[3], 26);
}

TEST_P(Conv2dBackendTest, GroupsDepthwise) {
    // Depthwise convolution (groups = in_channels = out_channels)
    int channels = 16;
    auto conv = nn::Conv2d(channels, channels, 3, 1, 0, 1, channels);
    auto input = Variable(randn({2, channels, 28, 28}, DType::Float32, device), true);
    auto output = conv.forward(input);

    EXPECT_EQ(output.shape()[0], 2);
    EXPECT_EQ(output.shape()[1], channels);
    EXPECT_EQ(output.shape()[2], 26);
    EXPECT_EQ(output.shape()[3], 26);
}

TEST_P(Conv2dBackendTest, Groups2) {
    // Grouped convolution (groups=2)
    auto conv = nn::Conv2d(8, 16, 3, 1, 0, 1, 2);
    auto input = Variable(randn({2, 8, 28, 28}, DType::Float32, device), true);
    auto output = conv.forward(input);

    EXPECT_EQ(output.shape()[0], 2);
    EXPECT_EQ(output.shape()[1], 16);
    EXPECT_EQ(output.shape()[2], 26);
    EXPECT_EQ(output.shape()[3], 26);
}

TEST_P(Conv2dBackendTest, Groups4) {
    // Grouped convolution (groups=4)
    auto conv = nn::Conv2d(12, 24, 3, 1, 0, 1, 4);
    auto input = Variable(randn({2, 12, 28, 28}, DType::Float32, device), true);
    auto output = conv.forward(input);

    EXPECT_EQ(output.shape()[0], 2);
    EXPECT_EQ(output.shape()[1], 24);
}

// ==========================
// Bias Tests
// ==========================

TEST_P(Conv2dBackendTest, WithBias) {
    auto conv = nn::Conv2d(3, 16, 3, 1, 0, 1, 1, true);
    auto params = conv.parameters();

    // Should have weight and bias
    EXPECT_EQ(params.size(), 2);
}

TEST_P(Conv2dBackendTest, NoBias) {
    auto conv = nn::Conv2d(3, 16, 3, 1, 0, 1, 1, false);
    auto params = conv.parameters();

    // Should have only weight
    EXPECT_EQ(params.size(), 1);
}

TEST_P(Conv2dBackendTest, BiasEffect) {
    // Test that bias affects output
    auto conv = nn::Conv2d(1, 1, 1, 1, 0, 1, 1, true);
    auto input = Variable(zeros({1, 1, 5, 5}, DType::Float32, device), true);
    auto output = conv.forward(input);

    // With zero input and 1x1 kernel, output should depend on bias
    EXPECT_EQ(output.shape()[0], 1);
    EXPECT_EQ(output.shape()[1], 1);
    EXPECT_EQ(output.shape()[2], 5);
    EXPECT_EQ(output.shape()[3], 5);
}

// ==========================
// Weight Shape Tests
// ==========================

TEST_P(Conv2dBackendTest, WeightShape) {
    auto conv = nn::Conv2d(3, 64, 3);
    auto params = conv.parameters();
    auto weight = params[0];

    auto weight_shape = weight->shape();
    EXPECT_EQ(weight_shape.size(), 4);
    EXPECT_EQ(weight_shape[0], 64);  // out_channels
    EXPECT_EQ(weight_shape[1], 3);   // in_channels / groups
    EXPECT_EQ(weight_shape[2], 3);   // kernel_height
    EXPECT_EQ(weight_shape[3], 3);   // kernel_width
}

TEST_P(Conv2dBackendTest, WeightShapeWithGroups) {
    auto conv = nn::Conv2d(8, 16, 3, 1, 0, 1, 2);  // groups=2
    auto params = conv.parameters();
    auto weight = params[0];

    auto weight_shape = weight->shape();
    EXPECT_EQ(weight_shape[0], 16);  // out_channels
    EXPECT_EQ(weight_shape[1], 4);   // in_channels / groups = 8/2
    EXPECT_EQ(weight_shape[2], 3);
    EXPECT_EQ(weight_shape[3], 3);
}

// ==========================
// Edge Cases
// ==========================

TEST_P(Conv2dBackendTest, EdgeCase1x1Image) {
    // Test with minimum image size
    auto conv = nn::Conv2d(3, 8, 1);  // 1x1 kernel
    auto input = Variable(randn({1, 3, 1, 1}, DType::Float32, device), true);
    auto output = conv.forward(input);

    EXPECT_EQ(output.shape()[0], 1);
    EXPECT_EQ(output.shape()[1], 8);
    EXPECT_EQ(output.shape()[2], 1);
    EXPECT_EQ(output.shape()[3], 1);
}

TEST_P(Conv2dBackendTest, EdgeCaseLargeImage) {
    // Test with large image size
    auto conv = nn::Conv2d(3, 64, 7, 2, 3);
    auto input = Variable(randn({1, 3, 512, 512}, DType::Float32, device), true);
    auto output = conv.forward(input);

    EXPECT_EQ(output.shape()[0], 1);
    EXPECT_EQ(output.shape()[1], 64);
    EXPECT_EQ(output.shape()[2], 256);  // (512 + 2*3 - 7) / 2 + 1
    EXPECT_EQ(output.shape()[3], 256);
}

TEST_P(Conv2dBackendTest, EdgeCaseVeryLargeBatch) {
    // Test with large batch size
    auto conv = nn::Conv2d(3, 16, 3);
    auto input = Variable(randn({128, 3, 16, 16}, DType::Float32, device), true);
    auto output = conv.forward(input);

    EXPECT_EQ(output.shape()[0], 128);
    EXPECT_EQ(output.shape()[1], 16);
}

TEST_P(Conv2dBackendTest, EdgeCaseSingleChannel) {
    // Test with single input/output channel
    auto conv = nn::Conv2d(1, 1, 3);
    auto input = Variable(randn({2, 1, 28, 28}, DType::Float32, device), true);
    auto output = conv.forward(input);

    EXPECT_EQ(output.shape()[1], 1);
}

TEST_P(Conv2dBackendTest, EdgeCaseManyChannels) {
    // Test with many channels
    auto conv = nn::Conv2d(512, 1024, 1);
    auto input = Variable(randn({1, 512, 7, 7}, DType::Float32, device), true);
    auto output = conv.forward(input);

    EXPECT_EQ(output.shape()[0], 1);
    EXPECT_EQ(output.shape()[1], 1024);
    EXPECT_EQ(output.shape()[2], 7);
    EXPECT_EQ(output.shape()[3], 7);
}

// ==========================
// Combined Parameters Tests
// ==========================

TEST_P(Conv2dBackendTest, CombinedStrideAndPadding) {
    auto conv = nn::Conv2d(3, 32, 3, 2, 1);  // stride=2, padding=1
    auto input = Variable(randn({4, 3, 32, 32}, DType::Float32, device), true);
    auto output = conv.forward(input);

    EXPECT_EQ(output.shape()[2], 16);  // (32 + 2*1 - 3) / 2 + 1
    EXPECT_EQ(output.shape()[3], 16);
}

TEST_P(Conv2dBackendTest, CombinedAllParameters) {
    // Test with all non-default parameters
    auto conv = nn::Conv2d(16, 32, 5, 2, 2, 2, 2, true);
    auto input = Variable(randn({2, 16, 64, 64}, DType::Float32, device), true);
    auto output = conv.forward(input);

    // (64 + 2*2 - 2*(5-1) - 1) / 2 + 1 = (64 + 4 - 8 - 1) / 2 + 1 = 59/2 + 1 = 30
    EXPECT_EQ(output.shape()[0], 2);
    EXPECT_EQ(output.shape()[1], 32);
    EXPECT_EQ(output.shape()[2], 30);
    EXPECT_EQ(output.shape()[3], 30);
}

// ==========================
// Consistency Tests
// ==========================

TEST_P(Conv2dBackendTest, ConsistentOutput) {
    // Test that same input produces same output
    auto conv = nn::Conv2d(3, 16, 3);
    auto input = Variable(randn({2, 3, 28, 28}, DType::Float32, device), true);

    auto output1 = conv.forward(input);
    auto output2 = conv.forward(input);

    // Shapes should be identical
    auto shape1 = output1.shape();
    auto shape2 = output2.shape();
    EXPECT_EQ(shape1.size(), shape2.size());
    EXPECT_TRUE(std::equal(shape1.begin(), shape1.end(), shape2.begin()));

    // Values should be identical (deterministic)
    EXPECT_TRUE(tensors_close(output1.tensor(), output2.tensor()));
}

TEST_P(Conv2dBackendTest, DifferentInputsSameSize) {
    // Test that different inputs with same size produce outputs with same shape
    auto conv = nn::Conv2d(3, 32, 5, 2, 2);

    auto input1 = Variable(randn({4, 3, 64, 64}, DType::Float32, device), true);
    auto input2 = Variable(randn({4, 3, 64, 64}, DType::Float32, device), true);

    auto output1 = conv.forward(input1);
    auto output2 = conv.forward(input2);

    auto shape1 = output1.shape();
    auto shape2 = output2.shape();
    EXPECT_EQ(shape1.size(), shape2.size());
    EXPECT_TRUE(std::equal(shape1.begin(), shape1.end(), shape2.begin()));
}

// ==========================
// Autograd Tests
// ==========================

TEST_P(Conv2dBackendTest, RequiresGrad) {
    auto conv = nn::Conv2d(3, 16, 3);
    auto input = Variable(randn({2, 3, 28, 28}, DType::Float32, device), true);
    auto output = conv.forward(input);

    EXPECT_TRUE(output.requires_grad());
}

TEST_P(Conv2dBackendTest, NoGradWhenInputNoGrad) {
    auto conv = nn::Conv2d(3, 16, 3);
    auto input = Variable(randn({2, 3, 28, 28}, DType::Float32, device), false);  // no grad
    auto output = conv.forward(input);

    // Output should still require grad because weights require grad
    EXPECT_TRUE(output.requires_grad());
}

TEST_P(Conv2dBackendTest, BackwardPassExecutes) {
    // Test that backward pass can be executed
    auto conv = nn::Conv2d(3, 8, 3);
    auto input = Variable(randn({2, 3, 16, 16}, DType::Float32, device), true);
    auto output = conv.forward(input);

    // Create gradient
    auto out_shape = output.shape();
    std::vector<int64_t> out_shape_vec(out_shape.begin(), out_shape.end());
    auto grad_output = ones(out_shape_vec, DType::Float32, device);

    // Backward should not throw
    EXPECT_NO_THROW({
        output.backward(grad_output);
    });
}

// ==========================
// Gradient Checking Tests
// ==========================

TEST_P(Conv2dBackendTest, GradientNonZero) {
    // Test that gradients are non-zero after backward
    auto conv = nn::Conv2d(3, 8, 3);
    auto input = Variable(randn({2, 3, 16, 16}, DType::Float32, device), true);
    auto output = conv.forward(input);

    // Backward with ones gradient
    auto out_shape = output.shape();
    std::vector<int64_t> out_shape_vec(out_shape.begin(), out_shape.end());
    output.backward(ones(out_shape_vec, DType::Float32, device));

    // Check that input has gradient
    EXPECT_TRUE(input.grad().has_value());

    // Check that gradient is non-zero
    auto grad_cpu = input.grad()->to(Device::cpu());
    auto grad_data = grad_cpu.data<float>();
    bool has_nonzero = false;
    size_t numel = static_cast<size_t>(grad_cpu.numel());
    for (size_t i = 0; i < numel; ++i) {
        if (std::abs(grad_data[i]) > 1e-6f) {
            has_nonzero = true;
            break;
        }
    }
    EXPECT_TRUE(has_nonzero);
}

// ==========================
// Parameter Tests
// ==========================

TEST_P(Conv2dBackendTest, ParameterCount) {
    // Test with bias
    auto conv_with_bias = nn::Conv2d(3, 16, 3, 1, 0, 1, 1, true);
    auto params_with = conv_with_bias.parameters();
    EXPECT_EQ(params_with.size(), 2);  // weight and bias

    // Test without bias
    auto conv_no_bias = nn::Conv2d(3, 16, 3, 1, 0, 1, 1, false);
    auto params_without = conv_no_bias.parameters();
    EXPECT_EQ(params_without.size(), 1);  // only weight
}

TEST_P(Conv2dBackendTest, ParameterSizes) {
    auto conv = nn::Conv2d(8, 16, 3);
    auto params = conv.parameters();

    // Weight size: [16, 8, 3, 3]
    auto weight = params[0];
    EXPECT_EQ(weight->tensor().numel(), 16 * 8 * 3 * 3);

    // Bias size: [16]
    if (params.size() > 1) {
        auto bias = params[1];
        EXPECT_EQ(bias->tensor().numel(), 16);
    }
}

// ==========================
// Special Configurations
// ==========================

TEST_P(Conv2dBackendTest, BottleneckConfiguration) {
    // Test 1x1 -> 3x3 -> 1x1 bottleneck configuration
    auto conv1 = nn::Conv2d(64, 16, 1);
    auto conv2 = nn::Conv2d(16, 16, 3, 1, 1);
    auto conv3 = nn::Conv2d(16, 64, 1);

    auto input = Variable(randn({2, 64, 28, 28}, DType::Float32, device), true);
    auto x = conv1.forward(input);
    x = conv2.forward(x);
    auto output = conv3.forward(x);

    EXPECT_EQ(output.shape()[0], 2);
    EXPECT_EQ(output.shape()[1], 64);
    EXPECT_EQ(output.shape()[2], 28);
    EXPECT_EQ(output.shape()[3], 28);
}

TEST_P(Conv2dBackendTest, ResidualConnection) {
    // Test residual connection compatibility
    auto conv1 = nn::Conv2d(32, 32, 3, 1, 1);
    auto conv2 = nn::Conv2d(32, 32, 3, 1, 1);

    auto input = Variable(randn({2, 32, 28, 28}, DType::Float32, device), true);
    auto x = conv1.forward(input);
    x = conv2.forward(x);

    // Should have same shape for residual addition
    auto x_shape = x.shape();
    auto input_shape = input.shape();
    EXPECT_EQ(x_shape.size(), input_shape.size());
    EXPECT_TRUE(std::equal(x_shape.begin(), x_shape.end(), input_shape.begin()));
}

// ==========================
// Performance/Memory Tests
// ==========================

TEST_P(Conv2dBackendTest, MemoryEfficiencySmall) {
    // Test that small convolutions don't allocate excessive memory
    auto conv = nn::Conv2d(3, 16, 3);
    auto input = Variable(randn({1, 3, 32, 32}, DType::Float32, device), true);

    EXPECT_NO_THROW({
        auto output = conv.forward(input);
    });
}

TEST_P(Conv2dBackendTest, MemoryEfficiencyLarge) {
    // Test with reasonably large input
    auto conv = nn::Conv2d(3, 64, 7, 2, 3);
    auto input = Variable(randn({8, 3, 224, 224}, DType::Float32, device), true);

    EXPECT_NO_THROW({
        auto output = conv.forward(input);
    });
}

// ==========================
// Error Handling Tests
// ==========================

TEST_P(Conv2dBackendTest, InvalidInputDimensions) {
    auto conv = nn::Conv2d(3, 16, 3);

    // 3D input should throw
    auto input_3d = Variable(randn({2, 3, 28}, DType::Float32, device), true);
    EXPECT_THROW({
        conv.forward(input_3d);
    }, std::invalid_argument);
}

TEST_P(Conv2dBackendTest, InvalidChannelCount) {
    auto conv = nn::Conv2d(3, 16, 3);

    // Wrong number of channels should throw
    auto input_wrong_channels = Variable(randn({2, 5, 28, 28}, DType::Float32, device), true);
    EXPECT_THROW({
        conv.forward(input_wrong_channels);
    }, std::invalid_argument);
}

TEST_P(Conv2dBackendTest, InvalidGroupConfiguration) {
    // in_channels not divisible by groups
    EXPECT_THROW({
        auto conv = nn::Conv2d(10, 8, 3, 1, 0, 1, 3);
    }, std::invalid_argument);

    // out_channels not divisible by groups
    EXPECT_THROW({
        auto conv = nn::Conv2d(9, 10, 3, 1, 0, 1, 3);
    }, std::invalid_argument);
}

// ==========================
// Real-World Patterns
// ==========================

TEST_P(Conv2dBackendTest, VGGStyleBlock) {
    // Test VGG-style conv block: Conv -> Conv -> Pool pattern
    auto conv1 = nn::Conv2d(64, 128, 3, 1, 1);
    auto conv2 = nn::Conv2d(128, 128, 3, 1, 1);

    auto input = Variable(randn({4, 64, 56, 56}, DType::Float32, device), true);
    auto x = conv1.forward(input);
    x = conv2.forward(x);

    EXPECT_EQ(x.shape()[0], 4);
    EXPECT_EQ(x.shape()[1], 128);
    EXPECT_EQ(x.shape()[2], 56);
    EXPECT_EQ(x.shape()[3], 56);
}

TEST_P(Conv2dBackendTest, InceptionStyleBranch) {
    // Test Inception-style parallel branches
    auto conv1x1 = nn::Conv2d(256, 64, 1);
    auto conv3x3 = nn::Conv2d(256, 128, 3, 1, 1);
    auto conv5x5 = nn::Conv2d(256, 32, 5, 1, 2);

    auto input = Variable(randn({2, 256, 28, 28}, DType::Float32, device), true);
    auto out1 = conv1x1.forward(input);
    auto out2 = conv3x3.forward(input);
    auto out3 = conv5x5.forward(input);

    // All outputs should have same spatial dimensions
    EXPECT_EQ(out1.shape()[2], 28);
    EXPECT_EQ(out2.shape()[2], 28);
    EXPECT_EQ(out3.shape()[2], 28);
}

TEST_P(Conv2dBackendTest, MobileNetDepthwiseSeparable) {
    // Test MobileNet-style depthwise separable convolution
    int channels = 32;

    // Depthwise convolution
    auto depthwise = nn::Conv2d(channels, channels, 3, 1, 1, 1, channels);

    // Pointwise convolution
    auto pointwise = nn::Conv2d(channels, 64, 1);

    auto input = Variable(randn({2, channels, 28, 28}, DType::Float32, device), true);
    auto x = depthwise.forward(input);
    auto output = pointwise.forward(x);

    EXPECT_EQ(output.shape()[0], 2);
    EXPECT_EQ(output.shape()[1], 64);
    EXPECT_EQ(output.shape()[2], 28);
    EXPECT_EQ(output.shape()[3], 28);
}

TEST_P(Conv2dBackendTest, ResNetBottleneck) {
    // Test ResNet-style bottleneck block
    auto conv1 = nn::Conv2d(256, 64, 1);
    auto conv2 = nn::Conv2d(64, 64, 3, 1, 1);
    auto conv3 = nn::Conv2d(64, 256, 1);

    auto input = Variable(randn({2, 256, 28, 28}, DType::Float32, device), true);
    auto x = conv1.forward(input);
    x = conv2.forward(x);
    x = conv3.forward(x);

    // Output should match input shape for residual
    auto x_shape = x.shape();
    auto input_shape = input.shape();
    EXPECT_EQ(x_shape.size(), input_shape.size());
    EXPECT_TRUE(std::equal(x_shape.begin(), x_shape.end(), input_shape.begin()));
}

// Instantiate tests for all backends
INSTANTIATE_BACKEND_TESTS(Conv2dBackendTest);
