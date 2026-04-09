/**
 * @file test_deeplabv3plus.cpp
 * @brief Tests for DeepLab v3+ segmentation model
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include "../../include/tenzor/models/deeplabv3plus.hpp"

using namespace tenzor;
using namespace tenzor::models;

class DeepLabV3PlusTest : public ::testing::Test {
protected:
    void SetUp() override { device_ = Device::cpu(); }
    Device device_;
};

TEST_F(DeepLabV3PlusTest, DeepLabV3PlusResNet50ForwardShape) {
    auto model = DeepLabV3Plus_ResNet50(21, 16, false);
    Variable images(Tensor({2, 3, 512, 512}, DType::Float32, device_), true);
    Variable output = model->forward(images);

    auto shape = output.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape.begin(), shape.end()),
              (std::vector<int64_t>{2, 21, 512, 512}));
}

TEST_F(DeepLabV3PlusTest, DeepLabV3PlusResNet50GradientFlow) {
    auto model = DeepLabV3Plus_ResNet50(21, 16, false);
    model->train();

    Variable images(Tensor({1, 3, 512, 512}, DType::Float32, device_), true);
    Variable output = model->forward(images);
    Variable loss = tenzor::sum(output);
    loss.backward();

    EXPECT_TRUE(images.grad().has_value());
    auto params = model->parameters();
    EXPECT_GT(params.size(), 0);
}

TEST_F(DeepLabV3PlusTest, DeepLabV3PlusResNet101ForwardShape) {
    auto model = DeepLabV3Plus_ResNet101(21, 16, false);
    Variable images(Tensor({1, 3, 512, 512}, DType::Float32, device_), true);
    Variable output = model->forward(images);

    auto shape = output.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape.begin(), shape.end()),
              (std::vector<int64_t>{1, 21, 512, 512}));
}

TEST_F(DeepLabV3PlusTest, DeepLabV3PlusResNet101GradientFlow) {
    auto model = DeepLabV3Plus_ResNet101(21, 16, false);
    model->train();

    Variable images(Tensor({1, 3, 512, 512}, DType::Float32, device_), true);
    Variable output = model->forward(images);
    Variable loss = tenzor::sum(output);
    loss.backward();

    EXPECT_TRUE(images.grad().has_value());
}

TEST_F(DeepLabV3PlusTest, DeepLabV3PlusMobileNetForwardShape) {
    auto model = DeepLabV3Plus_MobileNetV2(21, 16, false);
    Variable images(Tensor({2, 3, 512, 512}, DType::Float32, device_), true);
    Variable output = model->forward(images);

    auto shape = output.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape.begin(), shape.end()),
              (std::vector<int64_t>{2, 21, 512, 512}));
}

TEST_F(DeepLabV3PlusTest, DeepLabV3PlusDifferentSizes) {
    auto model = DeepLabV3Plus_ResNet50(21, 16, false);

    // Test with 256x256
    Variable images_256(Tensor({1, 3, 256, 256}, DType::Float32, device_), true);
    Variable output_256 = model->forward(images_256);
    auto shape_256 = output_256.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape_256.begin(), shape_256.end()),
              (std::vector<int64_t>{1, 21, 256, 256}));

    // Test with 1024x1024
    Variable images_1024(Tensor({1, 3, 1024, 1024}, DType::Float32, device_), true);
    Variable output_1024 = model->forward(images_1024);
    auto shape_1024 = output_1024.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape_1024.begin(), shape_1024.end()),
              (std::vector<int64_t>{1, 21, 1024, 1024}));
}

TEST_F(DeepLabV3PlusTest, DeepLabV3PlusParameterCount) {
    auto model = DeepLabV3Plus_ResNet50(21, 16, false);
    auto params = model->parameters();

    size_t total_params = 0;
    for (const auto& p : params) {
        size_t param_size = 1;
        for (auto dim : p->tensor().shape()) {
            param_size *= dim;
        }
        total_params += param_size;
    }

    // DeepLabV3+ ResNet50 should have ~40M parameters (allow 30% tolerance)
    EXPECT_GT(total_params, 30'000'000);
    EXPECT_LT(total_params, 55'000'000);
}

TEST_F(DeepLabV3PlusTest, DeepLabV3PlusBinarySegmentation) {
    auto model = DeepLabV3Plus_ResNet50(1, 16, false);
    Variable images(Tensor({2, 3, 512, 512}, DType::Float32, device_), true);
    Variable output = model->forward(images);

    auto shape = output.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape.begin(), shape.end()),
              (std::vector<int64_t>{2, 1, 512, 512}));
}


// ============================================================================
// Main  
// ============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    if (!::testing::GTEST_FLAG(list_tests)) {
        tenzor::initialize();
    }
    return RUN_ALL_TESTS();
}
