/**
 * @file test_classic_models.cpp
 * @brief Comprehensive tests for VGG, AlexNet, and GoogLeNet
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include "../backend_test_fixture.hpp"
#include "../../include/tenzor/models/vgg.hpp"
#include "../../include/tenzor/models/alexnet.hpp"
#include "../../include/tenzor/models/googlenet.hpp"
#include "../../include/tenzor/core/tensor.hpp"
#include "../../include/tenzor/autograd/variable.hpp"
#include <memory>

using namespace tenzor;
using namespace tenzor::models;

class ClassicModelsTest : public ::tenzor::testing::BackendTest {
protected:
    void SetUp() override {
        ::tenzor::testing::BackendTest::SetUp();
        if (::testing::Test::IsSkipped()) return;
    }
};

// ============================================================================
// VGG Tests
// ============================================================================

TEST_P(ClassicModelsTest, VGG11Construction) {
    auto model = vgg11(1000, true, false);
    ASSERT_NE(model, nullptr);

    // Check that model has parameters
    auto params = model->parameters();
    EXPECT_GT(params.size(), 0);
}

TEST_P(ClassicModelsTest, VGG11Forward) {
    auto model = vgg11(10, true, false);  // CIFAR-10 classes
    model->eval();
    model->to(device);

    // Create input: (batch=2, channels=3, height=224, width=224)
    Tensor input({2, 3, 224, 224}, DType::Float32, device);
    input.fill_(0.5f);
    Variable x(input, true);

    // Forward pass
    auto output = model->forward(x);

    // Check output shape: (batch=2, num_classes=10)
    EXPECT_EQ(output.tensor().shape().size(), 2);
    EXPECT_EQ(output.tensor().shape()[0], 2);
    EXPECT_EQ(output.tensor().shape()[1], 10);
}

TEST_P(ClassicModelsTest, VGG13Construction) {
    auto model = vgg13(1000, true, false);
    ASSERT_NE(model, nullptr);

    auto params = model->parameters();
    EXPECT_GT(params.size(), 0);
}

TEST_P(ClassicModelsTest, VGG16Construction) {
    auto model = vgg16(1000, true, false);
    ASSERT_NE(model, nullptr);

    auto params = model->parameters();
    EXPECT_GT(params.size(), 0);
}

TEST_P(ClassicModelsTest, VGG16Forward) {
    auto model = vgg16(1000, true, false);
    model->eval();
    model->to(device);

    Tensor input({1, 3, 224, 224}, DType::Float32, device);
    input.fill_(0.5f);
    Variable x(input, true);

    auto output = model->forward(x);

    EXPECT_EQ(output.tensor().shape().size(), 2);
    EXPECT_EQ(output.tensor().shape()[0], 1);
    EXPECT_EQ(output.tensor().shape()[1], 1000);
}

TEST_P(ClassicModelsTest, VGG19Construction) {
    auto model = vgg19(1000, true, false);
    ASSERT_NE(model, nullptr);

    auto params = model->parameters();
    EXPECT_GT(params.size(), 0);
}

TEST_P(ClassicModelsTest, VGG19Forward) {
    auto model = vgg19(100, true, false);  // Custom num_classes
    model->eval();
    model->to(device);

    Tensor input({4, 3, 224, 224}, DType::Float32, device);
    input.fill_(0.3f);
    Variable x(input, true);

    auto output = model->forward(x);

    EXPECT_EQ(output.tensor().shape()[0], 4);
    EXPECT_EQ(output.tensor().shape()[1], 100);
}

TEST_P(ClassicModelsTest, VGGWithoutBatchNorm) {
    auto model = vgg11(10, false, false);  // No batch norm
    ASSERT_NE(model, nullptr);

    model->eval();
    model->to(device);
    Tensor input({1, 3, 224, 224}, DType::Float32, device);
    input.fill_(0.5f);
    Variable x(input, true);

    auto output = model->forward(x);
    EXPECT_EQ(output.tensor().shape()[1], 10);
}

// ============================================================================
// AlexNet Tests
// ============================================================================

TEST_P(ClassicModelsTest, AlexNetConstruction) {
    auto model = alexnet(1000, false);
    ASSERT_NE(model, nullptr);

    auto params = model->parameters();
    EXPECT_GT(params.size(), 0);
}

TEST_P(ClassicModelsTest, AlexNetForward) {
    auto model = alexnet(1000, false);
    model->eval();
    model->to(device);

    // Input: (batch=2, channels=3, height=224, width=224)
    Tensor input({2, 3, 224, 224}, DType::Float32, device);
    input.fill_(0.5f);
    Variable x(input, true);

    auto output = model->forward(x);

    // Output: (batch=2, num_classes=1000)
    EXPECT_EQ(output.tensor().shape().size(), 2);
    EXPECT_EQ(output.tensor().shape()[0], 2);
    EXPECT_EQ(output.tensor().shape()[1], 1000);
}

TEST_P(ClassicModelsTest, AlexNetCustomClasses) {
    auto model = alexnet(10, false);  // CIFAR-10
    model->eval();
    model->to(device);

    Tensor input({1, 3, 224, 224}, DType::Float32, device);
    input.fill_(0.5f);
    Variable x(input, true);

    auto output = model->forward(x);
    EXPECT_EQ(output.tensor().shape()[1], 10);
}

TEST_P(ClassicModelsTest, AlexNetBatchProcessing) {
    auto model = alexnet(100, false);
    model->eval();
    model->to(device);

    // Larger batch
    Tensor input({8, 3, 224, 224}, DType::Float32, device);
    input.fill_(0.5f);
    Variable x(input, true);

    auto output = model->forward(x);
    EXPECT_EQ(output.tensor().shape()[0], 8);
    EXPECT_EQ(output.tensor().shape()[1], 100);
}

// ============================================================================
// GoogLeNet Tests
// ============================================================================

TEST_P(ClassicModelsTest, GoogLeNetConstruction) {
    auto model = googlenet(1000, false, true);
    ASSERT_NE(model, nullptr);

    auto params = model->parameters();
    EXPECT_GT(params.size(), 0);
}

TEST_P(ClassicModelsTest, GoogLeNetForward) {
    auto model = googlenet(1000, false, false);  // No aux classifiers
    model->eval();
    model->to(device);

    Tensor input({1, 3, 224, 224}, DType::Float32, device);
    input.fill_(0.5f);
    Variable x(input, true);

    auto output = model->forward(x);

    EXPECT_EQ(output.tensor().shape().size(), 2);
    EXPECT_EQ(output.tensor().shape()[0], 1);
    EXPECT_EQ(output.tensor().shape()[1], 1000);
}

TEST_P(ClassicModelsTest, GoogLeNetWithAuxiliaryClassifiers) {
    auto model = googlenet(10, false, true);  // With aux classifiers
    model->train();  // Training mode
    model->to(device);

    Tensor input({2, 3, 224, 224}, DType::Float32, device);
    input.fill_(0.5f);
    Variable x(input, true);

    // Forward with auxiliary outputs
    auto [main_out, aux1_out, aux2_out] = model->forward_with_aux(x);

    // Check main output
    EXPECT_EQ(main_out.tensor().shape()[0], 2);
    EXPECT_EQ(main_out.tensor().shape()[1], 10);

    // Check auxiliary outputs (should be valid during training)
    EXPECT_EQ(aux1_out.tensor().shape()[0], 2);
    EXPECT_EQ(aux1_out.tensor().shape()[1], 10);
    EXPECT_EQ(aux2_out.tensor().shape()[0], 2);
    EXPECT_EQ(aux2_out.tensor().shape()[1], 10);
}

TEST_P(ClassicModelsTest, GoogLeNetInferenceMode) {
    auto model = googlenet(1000, false, true);
    model->eval();  // Evaluation mode
    model->to(device);

    Tensor input({1, 3, 224, 224}, DType::Float32, device);
    input.fill_(0.5f);
    Variable x(input, true);

    // Regular forward (no auxiliary outputs)
    auto output = model->forward(x);

    EXPECT_EQ(output.tensor().shape()[1], 1000);
}

TEST_P(ClassicModelsTest, InceptionModuleForward) {
    // Test individual Inception module
    auto inception = std::make_shared<InceptionModule>(
        192,        // in_channels
        64,         // out_1x1
        96, 128,    // reduce_3x3, out_3x3
        16, 32,     // reduce_5x5, out_5x5
        32          // out_pool_proj
    );
    inception->to(device);

    Tensor input({1, 192, 28, 28}, DType::Float32, device);
    input.fill_(0.5f);
    Variable x(input, true);

    auto output = inception->forward(x);

    // Output channels: 64 + 128 + 32 + 32 = 256
    EXPECT_EQ(output.tensor().shape()[0], 1);
    EXPECT_EQ(output.tensor().shape()[1], 256);
    EXPECT_EQ(output.tensor().shape()[2], 28);
    EXPECT_EQ(output.tensor().shape()[3], 28);
}

// ============================================================================
// Comparative Tests
// ============================================================================

TEST_P(ClassicModelsTest, ModelParameterCounts) {
    auto vgg16_model = vgg16(1000, true, false);
    auto alexnet_model = alexnet(1000, false);
    auto googlenet_model = googlenet(1000, false, false);

    auto vgg_params = vgg16_model->parameters();
    auto alex_params = alexnet_model->parameters();
    auto google_params = googlenet_model->parameters();

    // VGG-16 should have the most parameters
    EXPECT_GT(vgg_params.size(), alex_params.size());

    // All models should have parameters
    EXPECT_GT(vgg_params.size(), 0);
    EXPECT_GT(alex_params.size(), 0);
    EXPECT_GT(google_params.size(), 0);
}

TEST_P(ClassicModelsTest, TrainingModeSwitch) {
    auto model = vgg16(10, true, false);

    // Default should be training mode
    EXPECT_TRUE(model->is_training());

    // Switch to eval
    model->eval();
    EXPECT_FALSE(model->is_training());

    // Switch back to training
    model->train();
    EXPECT_TRUE(model->is_training());
}

TEST_P(ClassicModelsTest, GradientTracking) {
    auto model = vgg11(10, true, false);
    model->to(device);

    Tensor input({1, 3, 224, 224}, DType::Float32, device);
    input.fill_(0.5f);
    Variable x(input, true);  // requires_grad = true

    auto output = model->forward(x);

    // Output should track gradients when input does
    EXPECT_TRUE(output.requires_grad());
}

TEST_P(ClassicModelsTest, PretrainedWeightsNotImplemented) {
    // All models should throw when trying to load pretrained weights
    EXPECT_THROW(vgg16(1000, true, true), std::runtime_error);
    EXPECT_THROW(alexnet(1000, true), std::runtime_error);
    EXPECT_THROW(googlenet(1000, true, false), std::runtime_error);
}

// ============================================================================
// Stress Tests
// ============================================================================

TEST_P(ClassicModelsTest, LargeBatchVGG) {
    auto model = vgg11(10, true, false);
    model->eval();
    model->to(device);

    // Large batch
    Tensor input({16, 3, 224, 224}, DType::Float32, device);
    input.fill_(0.5f);
    Variable x(input, false);  // No gradient tracking for speed

    auto output = model->forward(x);
    EXPECT_EQ(output.tensor().shape()[0], 16);
}

TEST_P(ClassicModelsTest, SmallBatchGoogLeNet) {
    auto model = googlenet(1000, false, false);
    model->eval();
    model->to(device);

    // Single sample
    Tensor input({1, 3, 224, 224}, DType::Float32, device);
    input.fill_(0.5f);
    Variable x(input, false);

    auto output = model->forward(x);
    EXPECT_EQ(output.tensor().shape()[0], 1);
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST_P(ClassicModelsTest, VGGCustomDropout) {
    // Test with different dropout rates
    auto model1 = std::make_shared<VGG>(VGGConfig::vgg11(), 10, true, 0.3);
    auto model2 = std::make_shared<VGG>(VGGConfig::vgg11(), 10, true, 0.7);

    EXPECT_NE(model1, nullptr);
    EXPECT_NE(model2, nullptr);
}

TEST_P(ClassicModelsTest, AlexNetCustomDropout) {
    auto model = std::make_shared<AlexNet>(10, 0.3);
    EXPECT_NE(model, nullptr);

    model->eval();
    model->to(device);
    Tensor input({1, 3, 224, 224}, DType::Float32, device);
    input.fill_(0.5f);
    Variable x(input, true);

    auto output = model->forward(x);
    EXPECT_EQ(output.tensor().shape()[1], 10);
}

TEST_P(ClassicModelsTest, GoogLeNetCustomDropout) {
    auto model = std::make_shared<GoogLeNet>(10, false, 0.2);
    EXPECT_NE(model, nullptr);

    model->eval();
    model->to(device);
    Tensor input({1, 3, 224, 224}, DType::Float32, device);
    input.fill_(0.5f);
    Variable x(input, true);

    auto output = model->forward(x);
    EXPECT_EQ(output.tensor().shape()[1], 10);
}

INSTANTIATE_BACKEND_TESTS(ClassicModelsTest);
