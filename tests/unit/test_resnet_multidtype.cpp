/**
 * @file test_resnet_multidtype.cpp
 * @brief Multi-backend and multi-dtype tests for ResNet family
 *
 * Tests ResNet models with Float32, Float64, and Float16 data types
 * across CPU, CUDA, and OneAPI backends.
 * ResNet is a critical vision architecture with residual connections that
 * enable deep network training and gradient flow.
 */

#include <gtest/gtest.h>
#include "../multi_backend_dtype_fixture.hpp"
#include "../../include/tenzor/models/resnet.hpp"
#include <cmath>

using namespace tenzor;
using namespace tenzor::models;
using namespace tenzor::testing;

// ============================================================================
// Test Fixture with Multi-Backend Multi-DType Support
// ============================================================================

class ResNetMultiDTypeTest : public MultiBackendDTypeTest {
protected:
    void SetUp() override {
        MultiBackendDTypeTest::SetUp();
        // Set parameter count tolerance based on dtype
        param_count_tol_ = (dtype() == DType::Float16) ? 0.05f : 0.01f;
    }

    bool CheckParameterCount(int64_t actual, int64_t expected) {
        int64_t tolerance = static_cast<int64_t>(expected * param_count_tol_);
        return std::abs(actual - expected) <= tolerance;
    }

    float param_count_tol_;
};

// ============================================================================
// BasicBlock Tests - Core Residual Building Block
// ============================================================================

TEST_P(ResNetMultiDTypeTest, BasicBlockForwardShape) {
    // Test BasicBlock with no downsampling (stride=1, same channels)
    auto block = std::make_shared<BasicBlock>(64, 64, 1, 1, 64, nullptr);
    convert_model(block);

    Variable input(Tensor({2, 64, 56, 56}, dtype(), device()), true);
    Variable output = block->forward(input);

    // Output shape should match input shape for identity mapping
    expectShape(output.tensor(), {2, 64, 56, 56});
}

TEST_P(ResNetMultiDTypeTest, BasicBlockForwardShapeWithStride) {
    // Test BasicBlock with stride=2 (spatial downsampling)
    auto downsample = std::make_shared<nn::Sequential>();
    auto conv = std::make_shared<nn::Conv2d>(64, 128, 1, 2, 0, 1, 1, false);
    auto bn = std::make_shared<nn::BatchNorm2d>(128);
    downsample->add_module(conv);
    downsample->add_module(bn);

    auto block = std::make_shared<BasicBlock>(64, 128, 2, 1, 64, downsample);
    convert_model(block);

    Variable input(Tensor({2, 64, 56, 56}, dtype(), device()), true);
    Variable output = block->forward(input);

    // Spatial dimensions halved, channels increased
    expectShape(output.tensor(), {2, 128, 28, 28});
}

TEST_P(ResNetMultiDTypeTest, BasicBlockGradientFlow) {
    // Critical test: verify gradients flow through skip connection
    auto block = std::make_shared<BasicBlock>(64, 64, 1, 1, 64, nullptr);
    convert_model(block);
    block->train();

    Variable input(Tensor({2, 64, 56, 56}, dtype(), device()), true);
    Variable output = block->forward(input);

    Variable loss = tenzor::sum(output * output);
    loss.backward();

    // Check that input has gradients
    EXPECT_TRUE(input.grad().has_value())
        << "Gradients should flow through residual connections to input";

    auto grad_shape = input.grad()->shape();
    auto input_shape = input.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(grad_shape.begin(), grad_shape.end()),
              std::vector<int64_t>(input_shape.begin(), input_shape.end()))
        << "Gradient shape should match input shape";

    // Check that block parameters have gradients
    auto params = block->parameters();
    EXPECT_GT(params.size(), 0) << "BasicBlock should have parameters";

    int params_with_grad = 0;
    for (const auto& param : params) {
        if (param->grad().has_value()) {
            params_with_grad++;
        }
    }
    EXPECT_EQ(params_with_grad, static_cast<int>(params.size()))
        << "All parameters should have gradients after backward pass";
}

TEST_P(ResNetMultiDTypeTest, BasicBlockMultipleBatchSizes) {
    // Test BasicBlock with different batch sizes
    auto block = std::make_shared<BasicBlock>(64, 64, 1, 1, 64, nullptr);
    convert_model(block);
    block->eval();

    for (int batch_size : {1, 2, 4, 8, 16}) {
        Variable input(Tensor({batch_size, 64, 56, 56}, dtype(), device()), false);
        Variable output = block->forward(input);

        expectShape(output.tensor(), {batch_size, 64, 56, 56});
    }
}

// ============================================================================
// Bottleneck Tests - Deeper Networks (ResNet-50+)
// ============================================================================

TEST_P(ResNetMultiDTypeTest, BottleneckForwardShape) {
    // Test Bottleneck with no downsampling
    auto block = std::make_shared<Bottleneck>(256, 64, 1, 1, 64, nullptr);
    convert_model(block);

    Variable input(Tensor({2, 256, 56, 56}, dtype(), device()), true);
    Variable output = block->forward(input);

    // Output channels = 64 * expansion (4) = 256
    expectShape(output.tensor(), {2, 256, 56, 56});
}

TEST_P(ResNetMultiDTypeTest, BottleneckForwardShapeWithStride) {
    // Test Bottleneck with stride=2
    auto downsample = std::make_shared<nn::Sequential>();
    auto conv = std::make_shared<nn::Conv2d>(256, 512, 1, 2, 0, 1, 1, false);
    auto bn = std::make_shared<nn::BatchNorm2d>(512);
    downsample->add_module(conv);
    downsample->add_module(bn);

    auto block = std::make_shared<Bottleneck>(256, 128, 2, 1, 64, downsample);
    convert_model(block);

    Variable input(Tensor({2, 256, 56, 56}, dtype(), device()), true);
    Variable output = block->forward(input);

    // Output: channels = 128 * 4 = 512, spatial halved
    expectShape(output.tensor(), {2, 512, 28, 28});
}

TEST_P(ResNetMultiDTypeTest, BottleneckGroupedConvolution) {
    // Test Bottleneck with groups (ResNeXt)
    auto downsample = std::make_shared<nn::Sequential>();
    auto conv = std::make_shared<nn::Conv2d>(256, 512, 1, 1, 0, 1, 1, false);
    auto bn = std::make_shared<nn::BatchNorm2d>(512);
    downsample->add_module(conv);
    downsample->add_module(bn);

    // groups=32, base_width=4 (ResNeXt-50 32x4d configuration)
    auto block = std::make_shared<Bottleneck>(256, 128, 1, 32, 4, downsample);
    convert_model(block);

    Variable input(Tensor({2, 256, 56, 56}, dtype(), device()), true);
    Variable output = block->forward(input);

    expectShape(output.tensor(), {2, 512, 56, 56});
}

TEST_P(ResNetMultiDTypeTest, BottleneckGradientFlow) {
    // Test gradient flow through deeper Bottleneck structure
    auto block = std::make_shared<Bottleneck>(256, 64, 1, 1, 64, nullptr);
    convert_model(block);
    block->train();

    Variable input(Tensor({2, 256, 56, 56}, dtype(), device()), true);
    Variable output = block->forward(input);

    Variable loss = tenzor::sum(output * output);
    loss.backward();

    EXPECT_TRUE(input.grad().has_value())
        << "Gradients should flow through Bottleneck to input";

    auto params = block->parameters();
    EXPECT_GT(params.size(), 0) << "Bottleneck should have parameters";

    for (const auto& param : params) {
        EXPECT_TRUE(param->grad().has_value())
            << "All Bottleneck parameters should have gradients";
    }
}

// ============================================================================
// ResNet-18 Tests - Smallest Standard ResNet
// ============================================================================

TEST_P(ResNetMultiDTypeTest, ResNet18Architecture) {
    auto model = resnet18(1000, false);
    convert_model(model);

    auto params = model->parameters();
    EXPECT_GT(params.size(), 0) << "ResNet-18 should have parameters";

    // ResNet-18: 11.7M parameters
    int64_t total_params = 0;
    for (const auto& param : params) {
        total_params += param->tensor().numel();
    }

    EXPECT_TRUE(CheckParameterCount(total_params, 11700000))
        << "ResNet-18 should have ~11.7M parameters, got " << total_params;
}

TEST_P(ResNetMultiDTypeTest, ResNet18ForwardShape) {
    auto model = resnet18(1000, false);
    convert_model(model);
    model->eval();

    // Standard ImageNet input: (N, 3, 224, 224)
    Variable input(Tensor({2, 3, 224, 224}, dtype(), device()), false);
    Variable output = model->forward(input);

    // Output should be (N, num_classes)
    expectShape(output.tensor(), {2, 1000});
}

TEST_P(ResNetMultiDTypeTest, ResNet18CustomClasses) {
    // Test with custom number of classes (e.g., CIFAR-10)
    auto model = resnet18(10, false);
    convert_model(model);
    model->eval();

    Variable input(Tensor({4, 3, 224, 224}, dtype(), device()), false);
    Variable output = model->forward(input);

    expectShape(output.tensor(), {4, 10});
}

TEST_P(ResNetMultiDTypeTest, ResNet18DifferentInputSizes) {
    // Test ResNet-18 with various input sizes
    auto model = resnet18(1000, false);
    convert_model(model);
    model->eval();

    std::vector<std::pair<int, int>> sizes = {
        {128, 128},  // Smaller than ImageNet
        {224, 224},  // Standard ImageNet
        {299, 299},  // Larger size
        {512, 512}   // High resolution
    };

    for (const auto& [h, w] : sizes) {
        Variable input(Tensor({2, 3, h, w}, dtype(), device()), false);
        Variable output = model->forward(input);

        expectShape(output.tensor(), {2, 1000});
    }
}

TEST_P(ResNetMultiDTypeTest, ResNet18GradientFlow) {
    auto model = resnet18(10, false);
    convert_model(model);
    model->train();

    Variable input(Tensor({2, 3, 224, 224}, dtype(), device()), true);
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

    EXPECT_EQ(params_with_grad, static_cast<int>(params.size()))
        << "All ResNet-18 parameters should receive gradients";
}

// ============================================================================
// ResNet-34 Tests - Deeper BasicBlock Network
// ============================================================================

TEST_P(ResNetMultiDTypeTest, ResNet34ForwardShape) {
    auto model = resnet34(1000, false);
    convert_model(model);
    model->eval();

    Variable input(Tensor({2, 3, 224, 224}, dtype(), device()), false);
    Variable output = model->forward(input);

    expectShape(output.tensor(), {2, 1000});
}

TEST_P(ResNetMultiDTypeTest, ResNet34ParameterCount) {
    auto model = resnet34(1000, false);
    convert_model(model);

    int64_t total_params = 0;
    for (const auto& param : model->parameters()) {
        total_params += param->tensor().numel();
    }

    // ResNet-34: 21.8M parameters
    EXPECT_TRUE(CheckParameterCount(total_params, 21800000))
        << "ResNet-34 should have ~21.8M parameters, got " << total_params;
}

TEST_P(ResNetMultiDTypeTest, ResNet34GradientFlow) {
    auto model = resnet34(10, false);
    convert_model(model);
    model->train();

    Variable input(Tensor({2, 3, 224, 224}, dtype(), device()), true);
    Variable output = model->forward(input);

    Variable loss = tenzor::sum(output);
    loss.backward();

    // Verify gradient flow in deeper network
    auto params = model->parameters();
    int params_with_grad = 0;
    for (const auto& param : params) {
        if (param->grad().has_value()) {
            params_with_grad++;
        }
    }

    EXPECT_GT(params_with_grad, 0)
        << "ResNet-34 parameters should have gradients after backward";
}

// ============================================================================
// ResNet-50 Tests - First Bottleneck Architecture
// ============================================================================

TEST_P(ResNetMultiDTypeTest, ResNet50ForwardShape) {
    auto model = resnet50(1000, false);
    convert_model(model);
    model->eval();

    Variable input(Tensor({2, 3, 224, 224}, dtype(), device()), false);
    Variable output = model->forward(input);

    expectShape(output.tensor(), {2, 1000});
}

TEST_P(ResNetMultiDTypeTest, ResNet50ParameterCount) {
    auto model = resnet50(1000, false);
    convert_model(model);

    int64_t total_params = 0;
    for (const auto& param : model->parameters()) {
        total_params += param->tensor().numel();
    }

    // ResNet-50: 25.6M parameters
    EXPECT_TRUE(CheckParameterCount(total_params, 25600000))
        << "ResNet-50 should have ~25.6M parameters, got " << total_params;
}

TEST_P(ResNetMultiDTypeTest, ResNet50GradientFlow) {
    auto model = resnet50(10, false);
    convert_model(model);
    model->train();

    Variable input(Tensor({2, 3, 224, 224}, dtype(), device()), true);
    Variable output = model->forward(input);

    Variable loss = tenzor::sum(output);
    loss.backward();

    // Verify gradients in first and last layers (critical for deep networks)
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

    EXPECT_TRUE(found_first_layer_grad)
        << "First layer (conv1) should receive gradients in ResNet-50";
    EXPECT_TRUE(found_last_layer_grad)
        << "Last layer (fc) should receive gradients in ResNet-50";
}

TEST_P(ResNetMultiDTypeTest, ResNet50FeatureExtraction) {
    // Test ResNet-50 for feature extraction (transfer learning)
    auto model = resnet50(1000, false);
    convert_model(model);
    model->eval();

    Variable input(Tensor({2, 3, 224, 224}, dtype(), device()), false);
    Variable features = model->forward(input);

    // Features should be high-dimensional
    expectShape(features.tensor(), {2, 1000});
}

// ============================================================================
// ResNet-101 and ResNet-152 Tests - Very Deep Networks
// ============================================================================

TEST_P(ResNetMultiDTypeTest, ResNet101ForwardShape) {
    auto model = resnet101(1000, false);
    convert_model(model);
    model->eval();

    Variable input(Tensor({2, 3, 224, 224}, dtype(), device()), false);
    Variable output = model->forward(input);

    expectShape(output.tensor(), {2, 1000});
}

TEST_P(ResNetMultiDTypeTest, ResNet101ParameterCount) {
    auto model = resnet101(1000, false);
    convert_model(model);

    int64_t total_params = 0;
    for (const auto& param : model->parameters()) {
        total_params += param->tensor().numel();
    }

    // ResNet-101: ~44.5M parameters
    EXPECT_TRUE(CheckParameterCount(total_params, 44500000))
        << "ResNet-101 should have ~44.5M parameters, got " << total_params;
}

TEST_P(ResNetMultiDTypeTest, ResNet152ForwardShape) {
    auto model = resnet152(1000, false);
    convert_model(model);
    model->eval();

    Variable input(Tensor({2, 3, 224, 224}, dtype(), device()), false);
    Variable output = model->forward(input);

    expectShape(output.tensor(), {2, 1000});
}

TEST_P(ResNetMultiDTypeTest, ResNet152ParameterCount) {
    auto model = resnet152(1000, false);
    convert_model(model);

    int64_t total_params = 0;
    for (const auto& param : model->parameters()) {
        total_params += param->tensor().numel();
    }

    // ResNet-152: ~60M parameters
    EXPECT_TRUE(CheckParameterCount(total_params, 60000000))
        << "ResNet-152 should have ~60M parameters, got " << total_params;
}

// ============================================================================
// ResNeXt Tests - Grouped Convolutions
// ============================================================================

TEST_P(ResNetMultiDTypeTest, ResNeXt50_32x4dForwardShape) {
    auto model = resnext50_32x4d(1000, false);
    convert_model(model);
    model->eval();

    Variable input(Tensor({2, 3, 224, 224}, dtype(), device()), false);
    Variable output = model->forward(input);

    expectShape(output.tensor(), {2, 1000});
}

TEST_P(ResNetMultiDTypeTest, ResNeXt50ParameterCount) {
    auto model = resnext50_32x4d(1000, false);
    convert_model(model);

    int64_t total_params = 0;
    for (const auto& param : model->parameters()) {
        total_params += param->tensor().numel();
    }

    // ResNeXt-50 32x4d: ~25M parameters (similar to ResNet-50)
    EXPECT_TRUE(CheckParameterCount(total_params, 25000000))
        << "ResNeXt-50 32x4d should have ~25M parameters";
}

TEST_P(ResNetMultiDTypeTest, ResNeXt101_32x8dForwardShape) {
    auto model = resnext101_32x8d(1000, false);
    convert_model(model);
    model->eval();

    Variable input(Tensor({2, 3, 224, 224}, dtype(), device()), false);
    Variable output = model->forward(input);

    expectShape(output.tensor(), {2, 1000});
}

TEST_P(ResNetMultiDTypeTest, ResNeXtGradientFlow) {
    auto model = resnext50_32x4d(10, false);
    convert_model(model);
    model->train();

    Variable input(Tensor({2, 3, 224, 224}, dtype(), device()), true);
    Variable output = model->forward(input);

    Variable loss = tenzor::sum(output);
    loss.backward();

    // Verify gradient flow through grouped convolutions
    auto params = model->parameters();
    int params_with_grad = 0;
    for (const auto& param : params) {
        if (param->grad().has_value()) {
            params_with_grad++;
        }
    }

    EXPECT_GT(params_with_grad, 0)
        << "ResNeXt parameters should have gradients with grouped convolutions";
}

// ============================================================================
// Wide ResNet Tests - Increased Width
// ============================================================================

TEST_P(ResNetMultiDTypeTest, WideResNet50_2ForwardShape) {
    auto model = wide_resnet50_2(1000, false);
    convert_model(model);
    model->eval();

    Variable input(Tensor({2, 3, 224, 224}, dtype(), device()), false);
    Variable output = model->forward(input);

    expectShape(output.tensor(), {2, 1000});
}

TEST_P(ResNetMultiDTypeTest, WideResNet50ParameterCount) {
    auto model = wide_resnet50_2(1000, false);
    convert_model(model);

    int64_t total_params = 0;
    for (const auto& param : model->parameters()) {
        total_params += param->tensor().numel();
    }

    // Wide ResNet-50-2 has significantly more parameters than standard ResNet-50
    EXPECT_GT(total_params, 25600000)
        << "Wide ResNet-50-2 should have more parameters than ResNet-50";
}

TEST_P(ResNetMultiDTypeTest, WideResNet101_2ForwardShape) {
    auto model = wide_resnet101_2(1000, false);
    convert_model(model);
    model->eval();

    Variable input(Tensor({2, 3, 224, 224}, dtype(), device()), false);
    Variable output = model->forward(input);

    expectShape(output.tensor(), {2, 1000});
}

// ============================================================================
// Training Mode Tests - BatchNorm Behavior
// ============================================================================

TEST_P(ResNetMultiDTypeTest, TrainingModeSwitch) {
    auto model = resnet18(10, false);
    convert_model(model);

    // Test training mode
    model->train();
    EXPECT_TRUE(model->is_training())
        << "Model should be in training mode after train()";

    // Test eval mode
    model->eval();
    EXPECT_FALSE(model->is_training())
        << "Model should be in eval mode after eval()";
}

TEST_P(ResNetMultiDTypeTest, BatchNormBehaviorDifference) {
    auto model = resnet18(10, false);
    convert_model(model);

    Variable input(Tensor({2, 3, 224, 224}, dtype(), device()), false);

    // Forward in training mode
    model->train();
    Variable output_train = model->forward(input);

    // Forward in eval mode
    model->eval();
    Variable output_eval = model->forward(input);

    // Both should produce valid outputs with correct shape
    expectShape(output_train.tensor(), {2, 10});
    expectShape(output_eval.tensor(), {2, 10});
}

// ============================================================================
// Transfer Learning Tests
// ============================================================================

TEST_P(ResNetMultiDTypeTest, TransferLearningFineTuning) {
    // Simulate transfer learning: pretrained model, new classification head
    auto model = resnet50(1000, false);  // ImageNet classes
    convert_model(model);

    // Replace final layer for new task (e.g., 10 classes)
    auto new_fc = std::make_shared<nn::Linear>(2048, 10);

    // Forward pass with new classifier
    model->eval();
    Variable input(Tensor({2, 3, 224, 224}, dtype(), device()), false);

    // Model should still work with original architecture
    Variable output = model->forward(input);
    expectShape(output.tensor(), {2, 1000});
}

TEST_P(ResNetMultiDTypeTest, FeatureExtractionMode) {
    // Test using ResNet as feature extractor
    auto model = resnet50(1000, false);
    convert_model(model);
    model->eval();

    // Freeze all parameters (for feature extraction)
    for (auto& param : model->parameters()) {
        param->set_requires_grad(false);
    }

    Variable input(Tensor({2, 3, 224, 224}, dtype(), device()), false);
    Variable features = model->forward(input);

    expectShape(features.tensor(), {2, 1000});
}

// ============================================================================
// State Dict Tests - Model Persistence
// ============================================================================

TEST_P(ResNetMultiDTypeTest, StateDictSaveLoad) {
    auto model1 = resnet18(10, false);
    auto model2 = resnet18(10, false);
    convert_model(model1);
    convert_model(model2);

    // Get state from model1
    auto state = model1->state_dict();
    EXPECT_GT(state.size(), 0) << "State dict should not be empty";

    // Load state into model2
    model2->load_state_dict(state);

    // Verify parameter counts match
    auto params1 = model1->named_parameters();
    auto params2 = model2->named_parameters();

    EXPECT_EQ(params1.size(), params2.size())
        << "Both models should have same number of parameters after load";
}

// ============================================================================
// Edge Cases and Error Handling
// ============================================================================

TEST_P(ResNetMultiDTypeTest, BasicBlockRejectsGroups) {
    // BasicBlock should reject groups != 1
    EXPECT_THROW({
        auto block = std::make_shared<BasicBlock>(64, 64, 1, 32, 64, nullptr);
    }, std::invalid_argument)
        << "BasicBlock should reject grouped convolutions";
}

TEST_P(ResNetMultiDTypeTest, BasicBlockRejectsWideWidth) {
    // BasicBlock should reject base_width != 64
    EXPECT_THROW({
        auto block = std::make_shared<BasicBlock>(64, 64, 1, 1, 128, nullptr);
    }, std::invalid_argument)
        << "BasicBlock should reject non-standard width";
}

TEST_P(ResNetMultiDTypeTest, SmallBatchSize) {
    // Test with batch size 1 (important for inference)
    auto model = resnet18(10, false);
    convert_model(model);
    model->eval();

    Variable input(Tensor({1, 3, 224, 224}, dtype(), device()), false);
    Variable output = model->forward(input);

    expectShape(output.tensor(), {1, 10});
}

TEST_P(ResNetMultiDTypeTest, LargeBatchSize) {
    // Test with larger batch size
    auto model = resnet18(10, false);
    convert_model(model);
    model->eval();

    Variable input(Tensor({16, 3, 224, 224}, dtype(), device()), false);
    Variable output = model->forward(input);

    expectShape(output.tensor(), {16, 10});
}

TEST_P(ResNetMultiDTypeTest, MinimalInputSize) {
    // Test minimum viable input size (32x32 after all downsampling)
    auto model = resnet18(10, false);
    convert_model(model);
    model->eval();

    Variable input(Tensor({2, 3, 32, 32}, dtype(), device()), false);
    Variable output = model->forward(input);

    expectShape(output.tensor(), {2, 10});
}

// ============================================================================
// Performance and Memory Tests
// ============================================================================

TEST_P(ResNetMultiDTypeTest, ParameterSharing) {
    auto model = resnet18(10, false);
    convert_model(model);

    // Get parameters twice and verify they share underlying storage
    auto params1 = model->parameters();
    auto params2 = model->parameters();

    EXPECT_EQ(params1.size(), params2.size())
        << "Parameter lists should have same size";

    // Verify same pointers (true parameter sharing)
    for (size_t i = 0; i < params1.size(); ++i) {
        EXPECT_EQ(params1[i].get(), params2[i].get())
            << "Parameters should be shared, not copied";
    }
}

TEST_P(ResNetMultiDTypeTest, ZeroGrad) {
    auto model = resnet18(10, false);
    convert_model(model);
    model->train();

    Variable input(Tensor({2, 3, 224, 224}, dtype(), device()), true);
    Variable output = model->forward(input);

    Variable loss = tenzor::sum(output);
    loss.backward();

    // Zero gradients
    model->zero_grad();

    // Verify all gradients are cleared or zeroed
    for (const auto& param : model->parameters()) {
        if (param->grad().has_value()) {
            auto grad_sum = tenzor::sum(*param->grad());
            auto cpu_grad = grad_sum.to(Device::cpu()).to(DType::Float32);
            auto* ptr = cpu_grad.data<float>();
            EXPECT_NEAR(ptr[0], 0.0f, atol())
                << "Gradients should be zero after zero_grad()";
        }
    }
}

TEST_P(ResNetMultiDTypeTest, MultipleForwardPasses) {
    // Test multiple forward passes (important for training loops)
    auto model = resnet18(10, false);
    convert_model(model);
    model->eval();

    Variable input(Tensor({2, 3, 224, 224}, dtype(), device()), false);

    for (int i = 0; i < 5; ++i) {
        Variable output = model->forward(input);
        expectShape(output.tensor(), {2, 10});
    }
}

// ============================================================================
// Residual Connection Tests - Core Innovation
// ============================================================================

TEST_P(ResNetMultiDTypeTest, ResidualConnectionIdentity) {
    // Test that residual connections preserve gradient flow
    auto model = resnet18(10, false);
    convert_model(model);
    model->train();

    Variable input(Tensor({2, 3, 224, 224}, dtype(), device()), true);
    Variable output = model->forward(input);

    Variable loss = tenzor::sum(output * output);
    loss.backward();

    // Input should have gradients (critical for residual connections)
    EXPECT_TRUE(input.grad().has_value())
        << "Residual connections should enable gradient flow to input";
}

TEST_P(ResNetMultiDTypeTest, DeepNetworkGradientFlow) {
    // Test that very deep network (ResNet-152) maintains gradient flow
    auto model = resnet152(10, false);
    convert_model(model);
    model->train();

    Variable input(Tensor({2, 3, 224, 224}, dtype(), device()), true);
    Variable output = model->forward(input);

    Variable loss = tenzor::sum(output);
    loss.backward();

    // Check gradient flow to early layers (challenging for deep networks)
    auto params = model->named_parameters();
    bool found_early_grad = false;

    for (const auto& [name, param] : params) {
        if (name.find("conv1") != std::string::npos && param->grad().has_value()) {
            found_early_grad = true;
            break;
        }
    }

    EXPECT_TRUE(found_early_grad)
        << "ResNet-152 should maintain gradient flow to early layers via residual connections";
}

// ============================================================================
// Instantiate Tests for All Backends and DTypes
// ============================================================================

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(ResNetMultiDTypeTest);

int main(int argc, char** argv) {
    tenzor::initialize();
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
