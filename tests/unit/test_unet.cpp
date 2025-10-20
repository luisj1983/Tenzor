/**
 * @file test_unet.cpp
 * @brief Tests for U-Net segmentation model
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include "../../include/tenzor/models/unet.hpp"

using namespace tenzor;
using namespace tenzor::models;

class UNetTest : public ::testing::Test {
protected:
    void SetUp() override { device_ = Device::cpu(); }
    Device device_;
};

TEST_F(UNetTest, UNetForwardShape) {
    auto model = std::make_shared<UNet>(3, 21, false);  // 3 input channels, 21 output classes
    Variable images(Tensor({2, 3, 256, 256}, DType::Float32, device_), true);
    Variable output = model->forward(images);

    auto shape = output.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape.begin(), shape.end()),
              (std::vector<int64_t>{2, 21, 256, 256}));
}

TEST_F(UNetTest, UNetGradientFlow) {
    auto model = std::make_shared<UNet>(3, 21, false);
    model->train();

    Variable images(Tensor({1, 3, 256, 256}, DType::Float32, device_), true);
    Variable output = model->forward(images);
    Variable loss = tenzor::sum(output);
    loss.backward();

    EXPECT_TRUE(images.grad().has_value());
    auto params = model->parameters();
    EXPECT_GT(params.size(), 0);
}

TEST_F(UNetTest, UNetParameterCount) {
    auto model = std::make_shared<UNet>(3, 21, false);
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

TEST_F(UNetTest, UNetDifferentInputSizes) {
    auto model = std::make_shared<UNet>(3, 21, false);

    // Test with 128x128
    Variable images_128(Tensor({1, 3, 128, 128}, DType::Float32, device_), true);
    Variable output_128 = model->forward(images_128);
    auto shape_128 = output_128.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape_128.begin(), shape_128.end()),
              (std::vector<int64_t>{1, 21, 128, 128}));

    // Test with 512x512
    Variable images_512(Tensor({1, 3, 512, 512}, DType::Float32, device_), true);
    Variable output_512 = model->forward(images_512);
    auto shape_512 = output_512.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape_512.begin(), shape_512.end()),
              (std::vector<int64_t>{1, 21, 512, 512}));
}

TEST_F(UNetTest, UNetBinarySegmentation) {
    auto model = std::make_shared<UNet>(3, 1, false);  // Binary segmentation
    Variable images(Tensor({2, 3, 256, 256}, DType::Float32, device_), true);
    Variable output = model->forward(images);

    auto shape = output.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape.begin(), shape.end()),
              (std::vector<int64_t>{2, 1, 256, 256}));
}

TEST_F(UNetTest, UNetGrayscaleInput) {
    auto model = std::make_shared<UNet>(1, 21, false);  // 1 input channel (grayscale)
    Variable images(Tensor({1, 1, 256, 256}, DType::Float32, device_), true);
    Variable output = model->forward(images);

    auto shape = output.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape.begin(), shape.end()),
              (std::vector<int64_t>{1, 21, 256, 256}));
}


// ============================================================================
// Main  
// ============================================================================

int main(int argc, char** argv) {
    tenzor::initialize();  // Initialize Tenzor library and backends
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
