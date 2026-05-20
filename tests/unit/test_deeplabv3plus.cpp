/**
 * @file test_deeplabv3plus.cpp
 * @brief Tests for DeepLab v3+ segmentation model
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include "../../include/tenzor/models/deeplabv3plus.hpp"
#include "../grad_flow_helpers.hpp"

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

    EXPECT_GRAD_FLOWS(images);
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

    EXPECT_GRAD_FLOWS(images);
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

    // DeepLabV3+ ResNet50: torchvision reference is ~40M parameters. Tenzor
    // uses single-scale ASPP (no multi-scale FPN routing, G11-followup) so it
    // lands lower. Audit G11 also removed the fake feature_proj_ (525K params)
    // — real C2 features don't need a 2048→256 projection — pushing the model
    // from ~30.08M to ~29.56M. Widened the lower bound to reflect the audit's
    // structural cleanup.
    EXPECT_GT(total_params, 25'000'000);
    EXPECT_LT(total_params, 55'000'000);
}

// G11 regression: atrous ResNet variants must produce features at the
// requested output_stride, not stride 32. Specifically, ResNet50 with
// output_stride=16 should give a C5 spatial size of input/16 (not input/32).
TEST_F(DeepLabV3PlusTest, AtrousResNet50OutputStride16_G11) {
    using namespace tenzor::models;
    auto backbone = resnet50_atrous(/*num_classes=*/1000, /*output_stride=*/16, /*pretrained=*/false);
    Variable img(Tensor({1, 3, 256, 256}, DType::Float32, device_), false);
    auto c5 = backbone->forward_features(img);

    // With output_stride=16, the layer4 stride was replaced by dilation=2,
    // so C5 is at input/16 = 16x16, not the regular 8x8.
    auto shape = c5.tensor().shape();
    EXPECT_EQ(shape[2], 16);
    EXPECT_EQ(shape[3], 16);
    EXPECT_EQ(shape[1], 2048);  // channel count unchanged by atrous
}

TEST_F(DeepLabV3PlusTest, AtrousResNet50OutputStride8_G11) {
    using namespace tenzor::models;
    auto backbone = resnet50_atrous(1000, /*output_stride=*/8, false);
    Variable img(Tensor({1, 3, 256, 256}, DType::Float32, device_), false);
    auto c5 = backbone->forward_features(img);

    auto shape = c5.tensor().shape();
    EXPECT_EQ(shape[2], 32);  // input/8 = 32
    EXPECT_EQ(shape[3], 32);
    EXPECT_EQ(shape[1], 2048);
}

// G11: regular resnet50 must still produce stride 32 (no regression).
TEST_F(DeepLabV3PlusTest, RegularResNet50StillStride32_G11) {
    using namespace tenzor::models;
    auto backbone = resnet50(1000, false);
    Variable img(Tensor({1, 3, 256, 256}, DType::Float32, device_), false);
    auto c5 = backbone->forward_features(img);

    auto shape = c5.tensor().shape();
    EXPECT_EQ(shape[2], 8);  // input/32 = 8
    EXPECT_EQ(shape[3], 8);
}

// G11: ResNet18/34 (BasicBlock) must reject atrous construction with a
// clear error message. Documented in resnet.hpp / deeplabv3plus.cpp.
TEST_F(DeepLabV3PlusTest, BasicBlockRejectsAtrous_G11) {
    EXPECT_THROW(
        std::make_shared<tenzor::models::ResNet>(
            std::vector<int64_t>{2, 2, 2, 2}, 1000, /*use_basic_block=*/true,
            /*groups=*/1, /*width_per_group=*/64, /*output_stride=*/16),
        std::invalid_argument);
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
