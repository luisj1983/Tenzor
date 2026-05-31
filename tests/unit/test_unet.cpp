/**
 * @file test_unet.cpp
 * @brief Tests for U-Net segmentation model
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include "../../include/tenzor/models/unet.hpp"
#include "../backend_test_fixture.hpp"
#include "../grad_flow_helpers.hpp"

using namespace tenzor;
using namespace tenzor::models;

class UNetTest : public ::tenzor::testing::BackendTest {
protected:
    void SetUp() override {
        ::tenzor::testing::BackendTest::SetUp();
        if (::testing::Test::IsSkipped()) return;
    }
};

TEST_P(UNetTest, UNetForwardShape) {
    auto model = std::make_shared<UNet>(3, 21, false);  // 3 input channels, 21 output classes
    model->to(device);
    Variable images(randn({2, 3, 256, 256}, DType::Float32, device), true);
    Variable output = model->forward(images);

    auto shape = output.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape.begin(), shape.end()),
              (std::vector<int64_t>{2, 21, 256, 256}));
}

TEST_P(UNetTest, UNetGradientFlow) {
    auto model = std::make_shared<UNet>(3, 21, false);
    model->to(device);
    model->train();

    Variable images(randn({1, 3, 256, 256}, DType::Float32, device), true);
    Variable output = model->forward(images);
    Variable loss = tenzor::sum(output);
    loss.backward();

    EXPECT_GRAD_FLOWS(images);
    auto params = model->parameters();
    EXPECT_GT(params.size(), 0);
}

TEST_P(UNetTest, UNetParameterCount) {
    auto model = std::make_shared<UNet>(3, 21, false);
    model->to(device);
    auto params = model->parameters();

    size_t total_params = 0;
    for (const auto& p : params) {
        size_t param_size = 1;
        for (auto dim : p->tensor().shape()) {
            param_size *= dim;
        }
        total_params += param_size;
    }

    // U-Net should have around 31M parameters (allow 30% tolerance)
    EXPECT_GT(total_params, 20'000'000);
    EXPECT_LT(total_params, 45'000'000);
}

TEST_P(UNetTest, UNetDifferentInputSizes) {
    auto model = std::make_shared<UNet>(3, 21, false);
    model->to(device);

    // Test with 128x128
    Variable images_128(randn({1, 3, 128, 128}, DType::Float32, device), true);
    Variable output_128 = model->forward(images_128);
    auto shape_128 = output_128.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape_128.begin(), shape_128.end()),
              (std::vector<int64_t>{1, 21, 128, 128}));

    // Test with 512x512
    Variable images_512(randn({1, 3, 512, 512}, DType::Float32, device), true);
    Variable output_512 = model->forward(images_512);
    auto shape_512 = output_512.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape_512.begin(), shape_512.end()),
              (std::vector<int64_t>{1, 21, 512, 512}));
}

TEST_P(UNetTest, UNetBinarySegmentation) {
    auto model = std::make_shared<UNet>(3, 1, false);  // Binary segmentation
    model->to(device);
    Variable images(randn({2, 3, 256, 256}, DType::Float32, device), true);
    Variable output = model->forward(images);

    auto shape = output.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape.begin(), shape.end()),
              (std::vector<int64_t>{2, 1, 256, 256}));
}

TEST_P(UNetTest, UNetGrayscaleInput) {
    auto model = std::make_shared<UNet>(1, 21, false);  // 1 input channel (grayscale)
    model->to(device);
    Variable images(randn({1, 1, 256, 256}, DType::Float32, device), true);
    Variable output = model->forward(images);

    auto shape = output.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape.begin(), shape.end()),
              (std::vector<int64_t>{1, 21, 256, 256}));
}

// G10 regression: Up in bilinear mode must do real spatial upsampling
// (not just a channel-adjust 1x1 conv). Verify by feeding a small input and
// a larger skip and confirming the upsample target size is honored.
TEST_P(UNetTest, BilinearUpDoesRealSpatialUpsample_G10) {
    // 64 in, 32 out, bilinear=true. The up_ 1x1 conv maps 64→32 channels;
    // the bilinear upsample resizes to skip's spatial dims.
    auto up = std::make_shared<Up>(/*in_channels=*/64, /*out_channels=*/32, /*bilinear=*/true);
    up->to(device);

    // input : (1, 64, 8, 8)  — small "bottom" feature
    // skip  : (1, 32, 16, 16) — twice the spatial size (the encoder skip)
    Variable input(randn({1, 64, 8, 8}, DType::Float32, device), false);
    Variable skip(randn({1, 32, 16, 16}, DType::Float32, device), false);

    Variable output = up->forward(input, skip);
    auto shape = output.tensor().shape();

    // forward: bilinear-upsample input 8→16, 1×1 conv 64→32 → (1, 32, 16, 16),
    // cat with skip on dim 1 → (1, 64, 16, 16), DoubleConv → (1, 32, 16, 16).
    ASSERT_EQ(shape.size(), 4u);
    EXPECT_EQ(shape[0], 1);
    EXPECT_EQ(shape[1], 32);  // out_channels
    EXPECT_EQ(shape[2], 16);  // upsampled spatial — NOT 8 (input H)
    EXPECT_EQ(shape[3], 16);
}

// G10: bilinear-mode UNet must restore the full input spatial dims at the
// output — same as transposed-conv mode. A regression to "channel-adjust only"
// (no real upsample) would leave the output at bottleneck spatial size.
TEST_P(UNetTest, UNetBilinearMatchesOutputShape_G10) {
    auto model_bilinear  = std::make_shared<UNet>(3, 21, /*bilinear=*/true);
    auto model_transpose = std::make_shared<UNet>(3, 21, /*bilinear=*/false);
    model_bilinear->to(device);
    model_transpose->to(device);

    Variable images(randn({1, 3, 256, 256}, DType::Float32, device), false);
    Variable out_bi = model_bilinear->forward(images);
    Variable out_tr = model_transpose->forward(images);

    auto sb = out_bi.tensor().shape();
    auto st = out_tr.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(sb.begin(), sb.end()),
              std::vector<int64_t>(st.begin(), st.end()));
    // Both must restore the input spatial dims.
    EXPECT_EQ(sb[2], 256);
    EXPECT_EQ(sb[3], 256);
}

INSTANTIATE_BACKEND_TESTS(UNetTest);
