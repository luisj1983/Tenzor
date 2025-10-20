/**
 * @file test_swin_transformer.cpp
 * @brief Comprehensive tests for Swin Transformer variants
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include "../../include/tenzor/models/swin_transformer.hpp"

using namespace tenzor;
using namespace tenzor::models;

class SwinTest : public ::testing::Test {
protected:
    void SetUp() override {
        device_ = Device::cpu();
    }
    Device device_;
};

// ============================================================================
// Swin-Tiny Tests
// ============================================================================

TEST_F(SwinTest, SwinTinyForwardShape) {
    auto model = swin_tiny(1000, 224, false);
    Variable input(Tensor({2, 3, 224, 224}, DType::Float32, device_), true);
    Variable output = model->forward(input);

    auto shape = output.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape.begin(), shape.end()),
              (std::vector<int64_t>{2, 1000}));
}

TEST_F(SwinTest, SwinTinyGradientFlow) {
    auto model = swin_tiny(10, 224, false);
    model->train();

    Variable input(Tensor({1, 3, 224, 224}, DType::Float32, device_), true);
    Variable output = model->forward(input);
    Variable loss = tenzor::sum(output);
    loss.backward();

    EXPECT_TRUE(input.grad().has_value());
    auto params = model->parameters();
    EXPECT_GT(params.size(), 0);
}

TEST_F(SwinTest, SwinTinyParameterCount) {
    auto model = swin_tiny(1000, 224, false);
    auto params = model->parameters();

    size_t total_params = 0;
    for (const auto& p : params) {
        size_t param_size = 1;
        for (auto dim : p->tensor().shape()) {
            param_size *= dim;
        }
        total_params += param_size;
    }

    // Swin-Tiny should have ~29M parameters (allow 20% tolerance)
    EXPECT_GT(total_params, 23'000'000);
    EXPECT_LT(total_params, 35'000'000);
}

// ============================================================================
// Swin-Small Tests
// ============================================================================

TEST_F(SwinTest, SwinSmallForwardShape) {
    auto model = swin_small(1000, 224, false);
    Variable input(Tensor({2, 3, 224, 224}, DType::Float32, device_), true);
    Variable output = model->forward(input);

    auto shape = output.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape.begin(), shape.end()),
              (std::vector<int64_t>{2, 1000}));
}

TEST_F(SwinTest, SwinSmallGradientFlow) {
    auto model = swin_small(10, 224, false);
    model->train();

    Variable input(Tensor({1, 3, 224, 224}, DType::Float32, device_), true);
    Variable output = model->forward(input);
    Variable loss = tenzor::sum(output);
    loss.backward();

    EXPECT_TRUE(input.grad().has_value());
}

// ============================================================================
// Swin-Base Tests
// ============================================================================

TEST_F(SwinTest, SwinBaseForwardShape) {
    auto model = swin_base(1000, 224, false);
    Variable input(Tensor({2, 3, 224, 224}, DType::Float32, device_), true);
    Variable output = model->forward(input);

    auto shape = output.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape.begin(), shape.end()),
              (std::vector<int64_t>{2, 1000}));
}

TEST_F(SwinTest, SwinBaseGradientFlow) {
    auto model = swin_base(10, 224, false);
    model->train();

    Variable input(Tensor({1, 3, 224, 224}, DType::Float32, device_), true);
    Variable output = model->forward(input);
    Variable loss = tenzor::sum(output);
    loss.backward();

    EXPECT_TRUE(input.grad().has_value());
}

TEST_F(SwinTest, SwinBaseParameterCount) {
    auto model = swin_base(1000, 224, false);
    auto params = model->parameters();

    size_t total_params = 0;
    for (const auto& p : params) {
        size_t param_size = 1;
        for (auto dim : p->tensor().shape()) {
            param_size *= dim;
        }
        total_params += param_size;
    }

    // Swin-Base should have ~88M parameters (allow 20% tolerance)
    EXPECT_GT(total_params, 70'000'000);
    EXPECT_LT(total_params, 105'000'000);
}

// ============================================================================
// Swin-Large Tests
// ============================================================================

TEST_F(SwinTest, SwinLargeForwardShape) {
    auto model = swin_large(1000, 224, false);
    Variable input(Tensor({1, 3, 224, 224}, DType::Float32, device_), true);
    Variable output = model->forward(input);

    auto shape = output.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape.begin(), shape.end()),
              (std::vector<int64_t>{1, 1000}));
}

TEST_F(SwinTest, SwinLargeGradientFlow) {
    auto model = swin_large(10, 224, false);
    model->train();

    Variable input(Tensor({1, 3, 224, 224}, DType::Float32, device_), true);
    Variable output = model->forward(input);
    Variable loss = tenzor::sum(output);
    loss.backward();

    EXPECT_TRUE(input.grad().has_value());
}

TEST_F(SwinTest, SwinLargeParameterCount) {
    auto model = swin_large(1000, 224, false);
    auto params = model->parameters();

    size_t total_params = 0;
    for (const auto& p : params) {
        size_t param_size = 1;
        for (auto dim : p->tensor().shape()) {
            param_size *= dim;
        }
        total_params += param_size;
    }

    // Swin-Large should have ~197M parameters (allow 20% tolerance)
    EXPECT_GT(total_params, 155'000'000);
    EXPECT_LT(total_params, 235'000'000);
}

// ============================================================================
// Edge Case Tests
// ============================================================================

TEST_F(SwinTest, SwinTinyBatchSizeOne) {
    auto model = swin_tiny(10, 224, false);
    Variable input(Tensor({1, 3, 224, 224}, DType::Float32, device_), true);
    Variable output = model->forward(input);

    auto shape = output.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape.begin(), shape.end()),
              (std::vector<int64_t>{1, 10}));
}

TEST_F(SwinTest, SwinTinyCustomClasses) {
    auto model = swin_tiny(100, 224, false);
    Variable input(Tensor({2, 3, 224, 224}, DType::Float32, device_), true);
    Variable output = model->forward(input);

    auto shape = output.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape.begin(), shape.end()),
              (std::vector<int64_t>{2, 100}));
}
