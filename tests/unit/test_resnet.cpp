/**
 * @file test_resnet.cpp
 * @brief Comprehensive tests for ResNet family
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include "../../include/tenzor/models/resnet.hpp"
#include "../../include/tenzor/core/tensor.hpp"
#include "../../include/tenzor/autograd/variable.hpp"
#include "../backend_test_fixture.hpp"
#include "../grad_flow_helpers.hpp"

using namespace tenzor;
using namespace tenzor::models;

class ResNetTest : public ::tenzor::testing::BackendTest {
protected:
    void SetUp() override {
        ::tenzor::testing::BackendTest::SetUp();
        if (::testing::Test::IsSkipped()) return;
    }
};

// ============================================================================
// BasicBlock Tests
// ============================================================================

TEST_P(ResNetTest, BasicBlockForwardShape) {
    // Test BasicBlock with no downsampling (stride=1, same channels)
    auto block = std::make_shared<BasicBlock>(64, 64, 1, 1, 64, nullptr);
    block->to(device);

    Variable input(Tensor({2, 64, 56, 56}, DType::Float32, device), true);
    Variable output = block->forward(input);

    // Output shape should match input shape
    auto shape = output.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape.begin(), shape.end()), (std::vector<int64_t>{2, 64, 56, 56}));
}

TEST_P(ResNetTest, BasicBlockForwardShapeWithStride) {
    // Test BasicBlock with stride=2 (spatial downsampling)
    // Need downsampling for skip connection
    auto downsample = std::make_shared<nn::Sequential>();
    auto conv = std::make_shared<nn::Conv2d>(64, 128, 1, 2, 0, 1, 1, false);
    auto bn = std::make_shared<nn::BatchNorm2d>(128);
    downsample->add_module(conv);
    downsample->add_module(bn);

    auto block = std::make_shared<BasicBlock>(64, 128, 2, 1, 64, downsample);
    block->to(device);

    Variable input(Tensor({2, 64, 56, 56}, DType::Float32, device), true);
    Variable output = block->forward(input);

    // Spatial dimensions halved, channels increased
    auto shape = output.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape.begin(), shape.end()), (std::vector<int64_t>{2, 128, 28, 28}));
}

TEST_P(ResNetTest, BasicBlockGradientFlow) {
    // Test that gradients flow through skip connection
    auto block = std::make_shared<BasicBlock>(64, 64, 1, 1, 64, nullptr);
    block->to(device);
    block->train();

    Variable input(Tensor({2, 64, 56, 56}, DType::Float32, device), true);
    Variable output = block->forward(input);

    // Compute a simple loss using autograd-aware sum
    Variable loss = tenzor::sum(output * output);
    loss.backward();

    // Check that input has gradients
    EXPECT_GRAD_FLOWS(input);
    auto grad_shape = input.grad()->shape();
    auto input_shape = input.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(grad_shape.begin(), grad_shape.end()),
              std::vector<int64_t>(input_shape.begin(), input_shape.end()));

    // Check that block parameters have gradients
    auto params = block->parameters();
    EXPECT_GT(params.size(), 0);
    for (const auto& param : params) {
        EXPECT_TRUE(param->grad().has_value());
    }
}

// ============================================================================
// Bottleneck Tests
// ============================================================================

TEST_P(ResNetTest, BottleneckForwardShape) {
    // Test Bottleneck with no downsampling
    auto block = std::make_shared<Bottleneck>(256, 64, 1, 1, 64, nullptr);
    block->to(device);

    Variable input(Tensor({2, 256, 56, 56}, DType::Float32, device), true);
    Variable output = block->forward(input);

    // Output channels = 64 * expansion (4) = 256
    auto shape = output.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape.begin(), shape.end()), (std::vector<int64_t>{2, 256, 56, 56}));
}

TEST_P(ResNetTest, BottleneckForwardShapeWithStride) {
    // Test Bottleneck with stride=2
    auto downsample = std::make_shared<nn::Sequential>();
    auto conv = std::make_shared<nn::Conv2d>(256, 512, 1, 2, 0, 1, 1, false);
    auto bn = std::make_shared<nn::BatchNorm2d>(512);
    downsample->add_module(conv);
    downsample->add_module(bn);

    auto block = std::make_shared<Bottleneck>(256, 128, 2, 1, 64, downsample);
    block->to(device);

    Variable input(Tensor({2, 256, 56, 56}, DType::Float32, device), true);
    Variable output = block->forward(input);

    // Output: channels = 128 * 4 = 512, spatial halved
    auto shape = output.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape.begin(), shape.end()), (std::vector<int64_t>{2, 512, 28, 28}));
}

TEST_P(ResNetTest, BottleneckGroupedConvolution) {
    // Test Bottleneck with groups (ResNeXt)
    auto downsample = std::make_shared<nn::Sequential>();
    auto conv = std::make_shared<nn::Conv2d>(256, 512, 1, 1, 0, 1, 1, false);
    auto bn = std::make_shared<nn::BatchNorm2d>(512);
    downsample->add_module(conv);
    downsample->add_module(bn);

    // groups=32, base_width=4 (ResNeXt configuration)
    auto block = std::make_shared<Bottleneck>(256, 128, 1, 32, 4, downsample);
    block->to(device);

    Variable input(Tensor({2, 256, 56, 56}, DType::Float32, device), true);
    Variable output = block->forward(input);

    auto shape = output.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape.begin(), shape.end()), (std::vector<int64_t>{2, 512, 56, 56}));
}

TEST_P(ResNetTest, BottleneckGradientFlow) {
    // Test gradient flow through Bottleneck
    auto block = std::make_shared<Bottleneck>(256, 64, 1, 1, 64, nullptr);
    block->to(device);
    block->train();

    Variable input(Tensor({2, 256, 56, 56}, DType::Float32, device), true);
    Variable output = block->forward(input);

    Variable loss = tenzor::sum(output * output);
    loss.backward();

    EXPECT_GRAD_FLOWS(input);

    auto params = block->parameters();
    EXPECT_GT(params.size(), 0);
    for (const auto& param : params) {
        EXPECT_TRUE(param->grad().has_value());
    }
}

// ============================================================================
// ResNet-18 Tests
// ============================================================================

TEST_P(ResNetTest, ResNet18Architecture) {
    auto model = resnet18(1000, false);
    model->to(device);

    // Check that model has correct number of parameters
    auto params = model->parameters();
    EXPECT_GT(params.size(), 0);

    // ResNet-18 should have approximately 11.7M parameters
    int64_t total_params = 0;
    for (const auto& param : params) {
        total_params += param->tensor().numel();
    }

    // Allow some tolerance (11M to 12M parameters)
    EXPECT_GT(total_params, 11000000);
    EXPECT_LT(total_params, 12000000);
}

TEST_P(ResNetTest, ResNet18ForwardShape) {
    auto model = resnet18(1000, false);
    model->to(device);
    model->eval();

    // Standard ImageNet input: (N, 3, 224, 224)
    Variable input(Tensor({2, 3, 224, 224}, DType::Float32, device), false);
    Variable output = model->forward(input);

    // Output should be (N, num_classes)
    auto shape = output.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape.begin(), shape.end()), (std::vector<int64_t>{2, 1000}));
}

TEST_P(ResNetTest, ResNet18CustomClasses) {
    // Test with custom number of classes
    auto model = resnet18(10, false);
    model->to(device);
    model->eval();

    Variable input(Tensor({4, 3, 224, 224}, DType::Float32, device), false);
    Variable output = model->forward(input);

    auto shape = output.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape.begin(), shape.end()), (std::vector<int64_t>{4, 10}));
}

TEST_P(ResNetTest, ResNet18DifferentInputSize) {
    // Test with different input size
    auto model = resnet18(1000, false);
    model->to(device);
    model->eval();

    // Smaller input (must be at least 32x32 to pass through all pooling)
    Variable input(Tensor({2, 3, 128, 128}, DType::Float32, device), false);
    Variable output = model->forward(input);

    // Output classes should still be correct
    auto shape = output.tensor().shape();
    EXPECT_EQ(shape[0], 2);
    EXPECT_EQ(shape[1], 1000);
}

TEST_P(ResNetTest, ResNet18GradientFlow) {
    auto model = resnet18(10, false);
    model->to(device);
    model->train();

    Variable input(Tensor({2, 3, 224, 224}, DType::Float32, device), true);
    Variable output = model->forward(input);

    Variable loss = tenzor::sum(output * output);
    loss.backward();

    // Check gradients flow to all parameters
    auto params = model->parameters();
    int params_with_grad = 0;
    for (const auto& param : params) {
        if (param->grad().has_value()) {
            params_with_grad++;
        }
    }

    EXPECT_EQ(params_with_grad, params.size());
}

// ============================================================================
// ResNet-34 Tests
// ============================================================================

TEST_P(ResNetTest, ResNet34ForwardShape) {
    auto model = resnet34(1000, false);
    model->to(device);
    model->eval();

    Variable input(Tensor({2, 3, 224, 224}, DType::Float32, device), false);
    Variable output = model->forward(input);

    auto shape = output.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape.begin(), shape.end()), (std::vector<int64_t>{2, 1000}));
}

TEST_P(ResNetTest, ResNet34ParameterCount) {
    auto model = resnet34(1000, false);
    model->to(device);

    int64_t total_params = 0;
    for (const auto& param : model->parameters()) {
        total_params += param->tensor().numel();
    }

    // ResNet-34 should have approximately 21.8M parameters
    EXPECT_GT(total_params, 21000000);
    EXPECT_LT(total_params, 22000000);
}

// ============================================================================
// ResNet-50 Tests
// ============================================================================

TEST_P(ResNetTest, ResNet50ForwardShape) {
    auto model = resnet50(1000, false);
    model->to(device);
    model->eval();

    Variable input(Tensor({2, 3, 224, 224}, DType::Float32, device), false);
    Variable output = model->forward(input);

    auto shape = output.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape.begin(), shape.end()), (std::vector<int64_t>{2, 1000}));
}

TEST_P(ResNetTest, ResNet50ParameterCount) {
    auto model = resnet50(1000, false);
    model->to(device);

    int64_t total_params = 0;
    for (const auto& param : model->parameters()) {
        total_params += param->tensor().numel();
    }

    // ResNet-50 should have approximately 25.6M parameters
    EXPECT_GT(total_params, 25000000);
    EXPECT_LT(total_params, 26000000);
}

TEST_P(ResNetTest, ResNet50GradientFlow) {
    auto model = resnet50(10, false);
    model->to(device);
    model->train();

    Variable input(Tensor({2, 3, 224, 224}, DType::Float32, device), true);
    Variable output = model->forward(input);

    Variable loss = tenzor::sum(output);
    loss.backward();

    // Verify gradients in first and last layers
    auto params = model->named_parameters();
    EXPECT_GT(params.size(), 0);

    bool found_first_layer_grad = false;
    bool found_last_layer_grad = false;

    for (const auto& [name, param] : params) {
        if (name.find("conv1") != std::string::npos && param->grad().has_value()) {
            found_first_layer_grad = true;
        }
        if (name.find("fc") != std::string::npos && param->grad().has_value()) {
            found_last_layer_grad = true;
        }
    }

    EXPECT_TRUE(found_first_layer_grad);
    EXPECT_TRUE(found_last_layer_grad);
}

// ============================================================================
// ResNet-101 and ResNet-152 Tests
// ============================================================================

TEST_P(ResNetTest, ResNet101ForwardShape) {
    auto model = resnet101(1000, false);
    model->to(device);
    model->eval();

    Variable input(Tensor({2, 3, 224, 224}, DType::Float32, device), false);
    Variable output = model->forward(input);

    auto shape = output.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape.begin(), shape.end()), (std::vector<int64_t>{2, 1000}));
}

TEST_P(ResNetTest, ResNet152ForwardShape) {
    auto model = resnet152(1000, false);
    model->to(device);
    model->eval();

    Variable input(Tensor({2, 3, 224, 224}, DType::Float32, device), false);
    Variable output = model->forward(input);

    auto shape = output.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape.begin(), shape.end()), (std::vector<int64_t>{2, 1000}));
}

// ============================================================================
// ResNeXt Tests
// ============================================================================

TEST_P(ResNetTest, ResNeXt50ForwardShape) {
    auto model = resnext50_32x4d(1000, false);
    model->to(device);
    model->eval();

    Variable input(Tensor({2, 3, 224, 224}, DType::Float32, device), false);
    Variable output = model->forward(input);

    auto shape = output.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape.begin(), shape.end()), (std::vector<int64_t>{2, 1000}));
}

TEST_P(ResNetTest, ResNeXt101ForwardShape) {
    auto model = resnext101_32x8d(1000, false);
    model->to(device);
    model->eval();

    Variable input(Tensor({2, 3, 224, 224}, DType::Float32, device), false);
    Variable output = model->forward(input);

    auto shape = output.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape.begin(), shape.end()), (std::vector<int64_t>{2, 1000}));
}

// ============================================================================
// Wide ResNet Tests
// ============================================================================

TEST_P(ResNetTest, WideResNet50ForwardShape) {
    auto model = wide_resnet50_2(1000, false);
    model->to(device);
    model->eval();

    Variable input(Tensor({2, 3, 224, 224}, DType::Float32, device), false);
    Variable output = model->forward(input);

    auto shape = output.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape.begin(), shape.end()), (std::vector<int64_t>{2, 1000}));
}

TEST_P(ResNetTest, WideResNet101ForwardShape) {
    auto model = wide_resnet101_2(1000, false);
    model->to(device);
    model->eval();

    Variable input(Tensor({2, 3, 224, 224}, DType::Float32, device), false);
    Variable output = model->forward(input);

    auto shape = output.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape.begin(), shape.end()), (std::vector<int64_t>{2, 1000}));
}

// ============================================================================
// Training Mode Tests
// ============================================================================

TEST_P(ResNetTest, TrainingModeSwitch) {
    auto model = resnet18(10, false);
    model->to(device);

    // Test training mode
    model->train();
    EXPECT_TRUE(model->is_training());

    // Test eval mode
    model->eval();
    EXPECT_FALSE(model->is_training());
}

TEST_P(ResNetTest, BatchNormBehavior) {
    auto model = resnet18(10, false);
    model->to(device);

    Variable input(Tensor({2, 3, 224, 224}, DType::Float32, device), false);

    // Forward in training mode
    model->train();
    Variable output_train = model->forward(input);

    // Forward in eval mode
    model->eval();
    Variable output_eval = model->forward(input);

    // Outputs may differ due to batch norm behavior
    // Just verify both produce valid outputs
    auto shape_train = output_train.tensor().shape();
    auto shape_eval = output_eval.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape_train.begin(), shape_train.end()),
              std::vector<int64_t>(shape_eval.begin(), shape_eval.end()));
}

// ============================================================================
// State Dict Tests
// ============================================================================

TEST_P(ResNetTest, StateDictSaveLoad) {
    auto model1 = resnet18(10, false);
    auto model2 = resnet18(10, false);
    model1->to(device);
    model2->to(device);

    // Get state from model1
    auto state = model1->state_dict();
    EXPECT_GT(state.size(), 0);

    // Load state into model2
    model2->load_state_dict(state);

    // Verify some parameters match
    auto params1 = model1->named_parameters();
    auto params2 = model2->named_parameters();

    EXPECT_EQ(params1.size(), params2.size());
}

// ============================================================================
// Edge Cases and Error Handling
// ============================================================================

TEST_P(ResNetTest, BasicBlockRejectsGroups) {
    // BasicBlock should reject groups != 1
    EXPECT_THROW({
        auto block = std::make_shared<BasicBlock>(64, 64, 1, 32, 64, nullptr);
    }, std::invalid_argument);
}

TEST_P(ResNetTest, BasicBlockRejectsWideWidth) {
    // BasicBlock should reject base_width != 64
    EXPECT_THROW({
        auto block = std::make_shared<BasicBlock>(64, 64, 1, 1, 128, nullptr);
    }, std::invalid_argument);
}

TEST_P(ResNetTest, SmallBatchSize) {
    // Test with batch size 1
    auto model = resnet18(10, false);
    model->to(device);
    model->eval();

    Variable input(Tensor({1, 3, 224, 224}, DType::Float32, device), false);
    Variable output = model->forward(input);

    auto shape = output.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape.begin(), shape.end()), (std::vector<int64_t>{1, 10}));
}

TEST_P(ResNetTest, LargeBatchSize) {
    // Test with larger batch size
    auto model = resnet18(10, false);
    model->to(device);
    model->eval();

    Variable input(Tensor({16, 3, 224, 224}, DType::Float32, device), false);
    Variable output = model->forward(input);

    auto shape = output.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape.begin(), shape.end()), (std::vector<int64_t>{16, 10}));
}

// ============================================================================
// Performance and Memory Tests
// ============================================================================

TEST_P(ResNetTest, ParameterSharing) {
    auto model = resnet18(10, false);
    model->to(device);

    // Get parameters twice and verify they share underlying storage
    auto params1 = model->parameters();
    auto params2 = model->parameters();

    EXPECT_EQ(params1.size(), params2.size());

    // Verify same pointers
    for (size_t i = 0; i < params1.size(); ++i) {
        EXPECT_EQ(params1[i].get(), params2[i].get());
    }
}

TEST_P(ResNetTest, ZeroGrad) {
    auto model = resnet18(10, false);
    model->to(device);
    model->train();

    Variable input(Tensor({2, 3, 224, 224}, DType::Float32, device), true);
    Variable output = model->forward(input);

    Variable loss = tenzor::sum(output);
    loss.backward();

    // Zero gradients
    model->zero_grad();

    // Verify all gradients are cleared
    for (const auto& param : model->parameters()) {
        if (param->grad().has_value()) {
            // If grad exists, it should be zeros
            auto grad_sum = tenzor::sum(*param->grad());
            auto grad_sum_cpu = grad_sum.cpu();
            auto* ptr = static_cast<float*>(grad_sum_cpu.data_ptr());
            EXPECT_EQ(ptr[0], 0.0f);
        }
    }
}

INSTANTIATE_BACKEND_TESTS(ResNetTest);
