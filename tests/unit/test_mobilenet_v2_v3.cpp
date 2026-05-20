/**
 * @file test_mobilenet_v2_v3.cpp
 * @brief Comprehensive tests for MobileNet V2 and V3 variants
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include "../../include/tenzor/models/mobilenet.hpp"
#include "../grad_flow_helpers.hpp"

using namespace tenzor;
using namespace tenzor::models;

class MobileNetTest : public ::testing::Test {
protected:
    void SetUp() override {
        device_ = Device::cpu();
    }
    Device device_;
};

// ============================================================================
// MobileNetV2 Tests
// ============================================================================

TEST_F(MobileNetTest, MobileNetV2ForwardShape) {
    auto model = mobilenet_v2(1000, false);
    Variable input(Tensor({2, 3, 224, 224}, DType::Float32, device_), true);
    Variable output = model->forward(input);

    auto shape = output.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape.begin(), shape.end()),
              (std::vector<int64_t>{2, 1000}));
}

TEST_F(MobileNetTest, MobileNetV2GradientFlow) {
    auto model = mobilenet_v2(10, false);
    model->train();

    Variable input(Tensor({1, 3, 224, 224}, DType::Float32, device_), true);
    Variable output = model->forward(input);
    Variable loss = tenzor::sum(output);
    loss.backward();

    EXPECT_GRAD_FLOWS(input);
    auto params = model->parameters();
    EXPECT_GT(params.size(), 0);
}

TEST_F(MobileNetTest, MobileNetV2WidthMultiplier) {
    // Test with width multiplier 0.5
    auto model_05 = mobilenet_v2_width(1000, 0.5, false);
    Variable input(Tensor({2, 3, 224, 224}, DType::Float32, device_), true);
    Variable output = model_05->forward(input);

    auto shape = output.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape.begin(), shape.end()),
              (std::vector<int64_t>{2, 1000}));
}

TEST_F(MobileNetTest, MobileNetV2ParameterCount) {
    auto model = mobilenet_v2(1000, false);
    auto params = model->parameters();

    size_t total_params = 0;
    for (const auto& p : params) {
        size_t param_size = 1;
        for (auto dim : p->tensor().shape()) {
            param_size *= dim;
        }
        total_params += param_size;
    }

    // MobileNetV2 should have ~3.5M parameters (allow 30% tolerance)
    EXPECT_GT(total_params, 2'500'000);
    EXPECT_LT(total_params, 5'000'000);
}

// ============================================================================
// MobileNetV3-Small Tests
// ============================================================================

TEST_F(MobileNetTest, MobileNetV3SmallForwardShape) {
    auto model = mobilenet_v3_small(1000, false);
    Variable input(Tensor({2, 3, 224, 224}, DType::Float32, device_), true);
    Variable output = model->forward(input);

    auto shape = output.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape.begin(), shape.end()),
              (std::vector<int64_t>{2, 1000}));
}

TEST_F(MobileNetTest, MobileNetV3SmallGradientFlow) {
    auto model = mobilenet_v3_small(10, false);
    model->train();

    Variable input(Tensor({1, 3, 224, 224}, DType::Float32, device_), true);
    Variable output = model->forward(input);
    Variable loss = tenzor::sum(output);
    loss.backward();

    EXPECT_GRAD_FLOWS(input);
}

TEST_F(MobileNetTest, MobileNetV3SmallParameterCount) {
    auto model = mobilenet_v3_small(1000, false);
    auto params = model->parameters();

    size_t total_params = 0;
    for (const auto& p : params) {
        size_t param_size = 1;
        for (auto dim : p->tensor().shape()) {
            param_size *= dim;
        }
        total_params += param_size;
    }

    // MobileNetV3-Small should have ~2.5M parameters (allow 30% tolerance)
    EXPECT_GT(total_params, 1'800'000);
    EXPECT_LT(total_params, 3'500'000);
}

// ============================================================================
// MobileNetV3-Large Tests
// ============================================================================

TEST_F(MobileNetTest, MobileNetV3LargeForwardShape) {
    auto model = mobilenet_v3_large(1000, false);
    Variable input(Tensor({2, 3, 224, 224}, DType::Float32, device_), true);
    Variable output = model->forward(input);

    auto shape = output.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape.begin(), shape.end()),
              (std::vector<int64_t>{2, 1000}));
}

TEST_F(MobileNetTest, MobileNetV3LargeGradientFlow) {
    auto model = mobilenet_v3_large(10, false);
    model->train();

    Variable input(Tensor({1, 3, 224, 224}, DType::Float32, device_), true);
    Variable output = model->forward(input);
    Variable loss = tenzor::sum(output);
    loss.backward();

    EXPECT_GRAD_FLOWS(input);
}

TEST_F(MobileNetTest, MobileNetV3LargeParameterCount) {
    auto model = mobilenet_v3_large(1000, false);
    auto params = model->parameters();

    size_t total_params = 0;
    for (const auto& p : params) {
        size_t param_size = 1;
        for (auto dim : p->tensor().shape()) {
            param_size *= dim;
        }
        total_params += param_size;
    }

    // MobileNetV3-Large should have ~5.4M parameters (allow 30% tolerance)
    EXPECT_GT(total_params, 4'000'000);
    EXPECT_LT(total_params, 7'000'000);
}

// ============================================================================
// Edge Case Tests
// ============================================================================

TEST_F(MobileNetTest, MobileNetV2BatchSizeOne) {
    auto model = mobilenet_v2(10, false);
    Variable input(Tensor({1, 3, 224, 224}, DType::Float32, device_), true);
    Variable output = model->forward(input);

    auto shape = output.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape.begin(), shape.end()),
              (std::vector<int64_t>{1, 10}));
}

TEST_F(MobileNetTest, MobileNetV3CustomClasses) {
    auto model = mobilenet_v3_large(100, false);
    Variable input(Tensor({2, 3, 224, 224}, DType::Float32, device_), true);
    Variable output = model->forward(input);

    auto shape = output.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape.begin(), shape.end()),
              (std::vector<int64_t>{2, 100}));
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
