/**
 * @file test_convnext.cpp
 * @brief Comprehensive tests for ConvNeXt variants
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include "../../include/tenzor/models/convnext.hpp"

using namespace tenzor;
using namespace tenzor::models;

class ConvNeXtTest : public ::testing::Test {
protected:
    void SetUp() override {
        device_ = Device::cpu();
    }
    Device device_;
};

// ============================================================================
// ConvNeXt-Tiny Tests
// ============================================================================

TEST_F(ConvNeXtTest, ConvNeXtTinyForwardShape) {
    auto model = convnext_tiny(1000, false);
    Variable input(Tensor({2, 3, 224, 224}, DType::Float32, device_), true);
    Variable output = model->forward(input);

    auto shape = output.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape.begin(), shape.end()),
              (std::vector<int64_t>{2, 1000}));
}

TEST_F(ConvNeXtTest, ConvNeXtTinyGradientFlow) {
    auto model = convnext_tiny(10, false);
    model->train();

    Variable input(Tensor({1, 3, 224, 224}, DType::Float32, device_), true);
    Variable output = model->forward(input);
    Variable loss = tenzor::sum(output);
    loss.backward();

    EXPECT_TRUE(input.grad().has_value());
    auto params = model->parameters();
    EXPECT_GT(params.size(), 0);
}

TEST_F(ConvNeXtTest, ConvNeXtTinyParameterCount) {
    auto model = convnext_tiny(1000, false);
    auto params = model->parameters();

    size_t total_params = 0;
    for (const auto& p : params) {
        size_t param_size = 1;
        for (auto dim : p->tensor().shape()) {
            param_size *= dim;
        }
        total_params += param_size;
    }

    // ConvNeXt-Tiny should have ~28M parameters (allow 20% tolerance)
    EXPECT_GT(total_params, 22'000'000);
    EXPECT_LT(total_params, 34'000'000);
}

// ============================================================================
// ConvNeXt-Small Tests
// ============================================================================

TEST_F(ConvNeXtTest, ConvNeXtSmallForwardShape) {
    auto model = convnext_small(1000, false);
    Variable input(Tensor({2, 3, 224, 224}, DType::Float32, device_), true);
    Variable output = model->forward(input);

    auto shape = output.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape.begin(), shape.end()),
              (std::vector<int64_t>{2, 1000}));
}

TEST_F(ConvNeXtTest, ConvNeXtSmallGradientFlow) {
    auto model = convnext_small(10, false);
    model->train();

    Variable input(Tensor({1, 3, 224, 224}, DType::Float32, device_), true);
    Variable output = model->forward(input);
    Variable loss = tenzor::sum(output);
    loss.backward();

    EXPECT_TRUE(input.grad().has_value());
}

// ============================================================================
// ConvNeXt-Base Tests
// ============================================================================

TEST_F(ConvNeXtTest, ConvNeXtBaseForwardShape) {
    auto model = convnext_base(1000, false);
    Variable input(Tensor({2, 3, 224, 224}, DType::Float32, device_), true);
    Variable output = model->forward(input);

    auto shape = output.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape.begin(), shape.end()),
              (std::vector<int64_t>{2, 1000}));
}

TEST_F(ConvNeXtTest, ConvNeXtBaseGradientFlow) {
    auto model = convnext_base(10, false);
    model->train();

    Variable input(Tensor({1, 3, 224, 224}, DType::Float32, device_), true);
    Variable output = model->forward(input);
    Variable loss = tenzor::sum(output);
    loss.backward();

    EXPECT_TRUE(input.grad().has_value());
}

TEST_F(ConvNeXtTest, ConvNeXtBaseParameterCount) {
    auto model = convnext_base(1000, false);
    auto params = model->parameters();

    size_t total_params = 0;
    for (const auto& p : params) {
        size_t param_size = 1;
        for (auto dim : p->tensor().shape()) {
            param_size *= dim;
        }
        total_params += param_size;
    }

    // ConvNeXt-Base should have ~89M parameters (allow 20% tolerance)
    EXPECT_GT(total_params, 70'000'000);
    EXPECT_LT(total_params, 107'000'000);
}

// ============================================================================
// ConvNeXt-Large Tests
// ============================================================================

TEST_F(ConvNeXtTest, ConvNeXtLargeForwardShape) {
    auto model = convnext_large(1000, false);
    Variable input(Tensor({1, 3, 224, 224}, DType::Float32, device_), true);
    Variable output = model->forward(input);

    auto shape = output.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape.begin(), shape.end()),
              (std::vector<int64_t>{1, 1000}));
}

TEST_F(ConvNeXtTest, ConvNeXtLargeGradientFlow) {
    auto model = convnext_large(10, false);
    model->train();

    Variable input(Tensor({1, 3, 224, 224}, DType::Float32, device_), true);
    Variable output = model->forward(input);
    Variable loss = tenzor::sum(output);
    loss.backward();

    EXPECT_TRUE(input.grad().has_value());
}

TEST_F(ConvNeXtTest, ConvNeXtLargeParameterCount) {
    auto model = convnext_large(1000, false);
    auto params = model->parameters();

    size_t total_params = 0;
    for (const auto& p : params) {
        size_t param_size = 1;
        for (auto dim : p->tensor().shape()) {
            param_size *= dim;
        }
        total_params += param_size;
    }

    // ConvNeXt-Large should have ~198M parameters (allow 20% tolerance)
    EXPECT_GT(total_params, 160'000'000);
    EXPECT_LT(total_params, 240'000'000);
}

// ============================================================================
// ConvNeXt-XLarge Tests
// ============================================================================

TEST_F(ConvNeXtTest, ConvNeXtXLargeForwardShape) {
    auto model = convnext_xlarge(1000, false);
    Variable input(Tensor({1, 3, 224, 224}, DType::Float32, device_), true);
    Variable output = model->forward(input);

    auto shape = output.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape.begin(), shape.end()),
              (std::vector<int64_t>{1, 1000}));
}

TEST_F(ConvNeXtTest, ConvNeXtXLargeGradientFlow) {
    auto model = convnext_xlarge(10, false);
    model->train();

    Variable input(Tensor({1, 3, 224, 224}, DType::Float32, device_), true);
    Variable output = model->forward(input);
    Variable loss = tenzor::sum(output);
    loss.backward();

    EXPECT_TRUE(input.grad().has_value());
}

// ============================================================================
// Edge Case Tests
// ============================================================================

TEST_F(ConvNeXtTest, ConvNeXtTinyBatchSizeOne) {
    auto model = convnext_tiny(10, false);
    Variable input(Tensor({1, 3, 224, 224}, DType::Float32, device_), true);
    Variable output = model->forward(input);

    auto shape = output.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape.begin(), shape.end()),
              (std::vector<int64_t>{1, 10}));
}

TEST_F(ConvNeXtTest, ConvNeXtTinyCustomClasses) {
    auto model = convnext_tiny(100, false);
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
