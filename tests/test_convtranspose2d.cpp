#include <gtest/gtest.h>
#include "tenzor/tenzor.hpp"
#include <cmath>

using namespace tenzor;
using namespace tenzor::nn;

// Test basic ConvTranspose2d construction
TEST(ConvTranspose2dTest, Construction) {
    EXPECT_NO_THROW({
        ConvTranspose2d layer(16, 32, 3);
    });

    EXPECT_NO_THROW({
        ConvTranspose2d layer(16, 32, 3, 2, 1, 1);
    });
}

// Test invalid parameters
TEST(ConvTranspose2dTest, InvalidParameters) {
    // in_channels not divisible by groups
    EXPECT_THROW({
        ConvTranspose2d layer(16, 32, 3, 1, 0, 0, 3);
    }, std::invalid_argument);

    // out_channels not divisible by groups
    EXPECT_THROW({
        ConvTranspose2d layer(16, 31, 3, 1, 0, 0, 2);
    }, std::invalid_argument);

    // output_padding >= stride
    EXPECT_THROW({
        ConvTranspose2d layer(16, 32, 3, 2, 0, 2);
    }, std::invalid_argument);
}

// Test output size calculation
TEST(ConvTranspose2dTest, OutputSize) {
    ConvTranspose2d layer(3, 16, 4, 2, 1, 0);

    // Input: [1, 3, 8, 8]
    auto input = randn({1, 3, 8, 8});
    auto output = layer.forward(Variable(input, false));

    // Expected output size:
    // H_out = (H_in - 1) * stride - 2*padding + kernel_size + output_padding
    // H_out = (8 - 1) * 2 - 2*1 + 4 + 0 = 14 - 2 + 4 = 16
    auto output_shape = output.shape();
    EXPECT_EQ(output_shape.size(), 4u);
    EXPECT_EQ(output_shape[0], 1);
    EXPECT_EQ(output_shape[1], 16);
    EXPECT_EQ(output_shape[2], 16);
    EXPECT_EQ(output_shape[3], 16);
}

// Test upsampling with stride=2 (2x upsampling)
TEST(ConvTranspose2dTest, Upsampling2x) {
    ConvTranspose2d layer(16, 3, 4, 2, 1, 0);

    // Input: [2, 16, 32, 32]
    auto input = randn({2, 16, 32, 32});
    auto output = layer.forward(Variable(input, false));

    // Expected output size:
    // H_out = (32 - 1) * 2 - 2*1 + 4 + 0 = 31*2 - 2 + 4 = 64
    auto output_shape = output.shape();
    EXPECT_EQ(output_shape[0], 2);
    EXPECT_EQ(output_shape[1], 3);
    EXPECT_EQ(output_shape[2], 64);
    EXPECT_EQ(output_shape[3], 64);
}

// Test with output_padding
TEST(ConvTranspose2dTest, OutputPadding) {
    ConvTranspose2d layer1(8, 16, 3, 2, 1, 0);
    ConvTranspose2d layer2(8, 16, 3, 2, 1, 1);

    auto input = randn({1, 8, 10, 10});

    auto output1 = layer1.forward(Variable(input, false));
    auto output2 = layer2.forward(Variable(input, false));

    // With output_padding=1, output should be 1 pixel larger
    auto shape1 = output1.shape();
    auto shape2 = output2.shape();

    EXPECT_EQ(shape2[2], shape1[2] + 1);
    EXPECT_EQ(shape2[3], shape1[3] + 1);
}

// Test gradient computation (backward pass)
TEST(ConvTranspose2dTest, BackwardPass) {
    ConvTranspose2d layer(4, 8, 3, 1, 1);

    auto input_tensor = randn({2, 4, 6, 6});
    auto input = Variable(input_tensor, true);

    auto output = layer.forward(input);
    EXPECT_TRUE(output.requires_grad());

    // Create gradient and perform backward
    auto out_shape = output.shape();
    std::vector<int64_t> grad_shape(out_shape.begin(), out_shape.end());
    auto grad_output = ones(grad_shape);
    output.backward(grad_output);

    // Check that gradients were computed
    EXPECT_TRUE(input.grad().has_value());

    auto grad_input = input.grad().value();
    auto gi_shape = grad_input.shape();
    auto in_shape = input.shape();
    EXPECT_EQ(gi_shape.size(), in_shape.size());
    for (size_t i = 0; i < gi_shape.size(); ++i) {
        EXPECT_EQ(gi_shape[i], in_shape[i]);
    }
}

// Test with groups
TEST(ConvTranspose2dTest, GroupedConvolution) {
    ConvTranspose2d layer(16, 32, 3, 1, 1, 0, 4);

    auto input = randn({1, 16, 8, 8});
    auto output = layer.forward(Variable(input, false));

    auto output_shape = output.shape();
    EXPECT_EQ(output_shape[0], 1);
    EXPECT_EQ(output_shape[1], 32);
    EXPECT_EQ(output_shape[2], 8);
    EXPECT_EQ(output_shape[3], 8);
}

// Test bias addition
TEST(ConvTranspose2dTest, BiasAddition) {
    // With bias
    ConvTranspose2d layer_with_bias(4, 8, 3, 1, 1, 0, 1, true);

    // Without bias
    ConvTranspose2d layer_no_bias(4, 8, 3, 1, 1, 0, 1, false);

    auto input = randn({1, 4, 8, 8});

    auto output1 = layer_with_bias.forward(Variable(input, false));
    auto output2 = layer_no_bias.forward(Variable(input, false));

    // Both should have same shape
    auto shape1 = output1.shape();
    auto shape2 = output2.shape();
    EXPECT_EQ(shape1.size(), shape2.size());
    for (size_t i = 0; i < shape1.size(); ++i) {
        EXPECT_EQ(shape1[i], shape2[i]);
    }
}

// Test numerical stability with different kernel sizes
TEST(ConvTranspose2dTest, DifferentKernelSizes) {
    std::vector<int64_t> kernel_sizes = {1, 2, 3, 5, 7};

    for (auto k : kernel_sizes) {
        EXPECT_NO_THROW({
            ConvTranspose2d layer(8, 16, k);
            auto input = randn({1, 8, 10, 10});
            auto output = layer.forward(Variable(input, false));
            EXPECT_GT(output.shape()[2], 0);
            EXPECT_GT(output.shape()[3], 0);
        });
    }
}

// Test that ConvTranspose2d is the approximate inverse of Conv2d
TEST(ConvTranspose2dTest, InverseOfConv2d) {
    // Create matching Conv2d and ConvTranspose2d layers
    Conv2d conv(3, 16, 4, 2, 1);
    ConvTranspose2d deconv(16, 3, 4, 2, 1);

    // Input: [1, 3, 64, 64]
    auto input = randn({1, 3, 64, 64});

    // Forward: Conv2d should downsample to [1, 16, 32, 32]
    auto conv_output = conv.forward(Variable(input, false));
    EXPECT_EQ(conv_output.shape()[2], 32);
    EXPECT_EQ(conv_output.shape()[3], 32);

    // Backward: ConvTranspose2d should upsample back to [1, 3, 64, 64]
    auto deconv_output = deconv.forward(conv_output);
    EXPECT_EQ(deconv_output.shape()[1], 3);
    EXPECT_EQ(deconv_output.shape()[2], 64);
    EXPECT_EQ(deconv_output.shape()[3], 64);
}

// Test gradient flow through the layer
TEST(ConvTranspose2dTest, GradientFlow) {
    ConvTranspose2d layer(8, 16, 3, 2, 1);

    auto input_tensor = randn({1, 8, 16, 16});
    auto input = Variable(input_tensor, true);

    auto output = layer.forward(input);

    // Backward pass
    auto out_shape = output.shape();
    std::vector<int64_t> grad_shape(out_shape.begin(), out_shape.end());
    auto grad = ones(grad_shape);
    output.backward(grad);

    // Verify gradients exist and are reasonable
    EXPECT_TRUE(input.grad().has_value());
    auto grad_input = input.grad().value();

    const float* grad_data = grad_input.data<float>();
    bool has_nonzero = false;
    for (int64_t i = 0; i < grad_input.numel(); ++i) {
        if (std::abs(grad_data[i]) > 1e-6) {
            has_nonzero = true;
            break;
        }
    }
    EXPECT_TRUE(has_nonzero);
}

// Test multiple strides
TEST(ConvTranspose2dTest, DifferentStrides) {
    auto input = randn({1, 8, 16, 16});

    // stride=1: output size ~ same
    ConvTranspose2d layer1(8, 16, 3, 1, 1);
    auto out1 = layer1.forward(Variable(input, false));
    EXPECT_EQ(out1.shape()[2], 16);

    // stride=2: output size ~ 2x
    ConvTranspose2d layer2(8, 16, 3, 2, 1);
    auto out2 = layer2.forward(Variable(input, false));
    EXPECT_GT(out2.shape()[2], out1.shape()[2]);

    // stride=4: output size ~ 4x
    ConvTranspose2d layer4(8, 16, 3, 4, 1);
    auto out4 = layer4.forward(Variable(input, false));
    EXPECT_GT(out4.shape()[2], out2.shape()[2]);
}

// Test batch processing
TEST(ConvTranspose2dTest, BatchProcessing) {
    ConvTranspose2d layer(4, 8, 3, 2, 1);

    // Test with different batch sizes
    for (int64_t batch : {1, 2, 4, 8}) {
        auto input = randn({batch, 4, 16, 16});
        auto output = layer.forward(Variable(input, false));

        EXPECT_EQ(output.shape()[0], batch);
        EXPECT_EQ(output.shape()[1], 8);
    }
}

// Test typical GAN generator architecture
TEST(ConvTranspose2dTest, GANGeneratorPattern) {
    // Typical GAN generator upsampling pattern
    ConvTranspose2d layer1(512, 256, 4, 2, 1);  // 4x4 -> 8x8
    ConvTranspose2d layer2(256, 128, 4, 2, 1);  // 8x8 -> 16x16
    ConvTranspose2d layer3(128, 64, 4, 2, 1);   // 16x16 -> 32x32
    ConvTranspose2d layer4(64, 3, 4, 2, 1);     // 32x32 -> 64x64

    auto z = randn({1, 512, 4, 4});

    auto h1 = layer1.forward(Variable(z, false));
    EXPECT_EQ(h1.shape()[2], 8);

    auto h2 = layer2.forward(h1);
    EXPECT_EQ(h2.shape()[2], 16);

    auto h3 = layer3.forward(h2);
    EXPECT_EQ(h3.shape()[2], 32);

    auto h4 = layer4.forward(h3);
    EXPECT_EQ(h4.shape()[1], 3);
    EXPECT_EQ(h4.shape()[2], 64);
    EXPECT_EQ(h4.shape()[3], 64);
}

int main(int argc, char** argv) {
    tenzor::initialize();
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
