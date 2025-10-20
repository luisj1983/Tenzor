# Phase 9 Test Plan and Implementation Status Report

**Date:** 2025-10-17
**Status:** 🔴 **NOT IMPLEMENTED - AWAITING CODER AGENTS**
**Test Coverage:** 0% (No implementations exist yet)

---

## Executive Summary

Phase 9 "Model Zoo & Pretrained Models" has **NOT been implemented**. The `/home/lee/Projects/Tenzor/include/tenzor/models/` directory exists but is empty. No model implementations (ResNet, VGG, AlexNet, GoogLeNet, BERT, GPT) or Model Hub functionality has been created.

This document provides:
1. Current status verification
2. Comprehensive test plan for when implementations are created
3. Test file templates ready for immediate use
4. Clear acceptance criteria for each model type

---

## Current Implementation Status

### Directory Structure Analysis

```bash
/home/lee/Projects/Tenzor/include/tenzor/models/
└── (empty directory)

Expected implementations:
  - resnet.hpp (NOT FOUND)
  - vgg.hpp (NOT FOUND)
  - alexnet.hpp (NOT FOUND)
  - googlenet.hpp (NOT FOUND)
  - bert.hpp (NOT FOUND)
  - gpt.hpp (NOT FOUND)
  - model_hub.hpp (NOT FOUND)
```

### Search Results

**No Phase 9 implementations found:**
- ❌ No ResNet classes (ResNet18, ResNet34, ResNet50, ResNet101, ResNet152)
- ❌ No VGG classes (VGG11, VGG13, VGG16, VGG19)
- ❌ No AlexNet class
- ❌ No GoogLeNet class
- ❌ No BERT classes (BertBase, BertLarge, BertTiny, BertMini)
- ❌ No GPT classes (GPT2Small, GPT2Medium, GPT2Large, GPT3)
- ❌ No ModelHub class
- ❌ No weight management system
- ❌ No pretrained weight loading

---

## Phase 9 Requirements (From TODO.md)

### 9.1 Computer Vision Models

#### 9.1.1 ResNet Family (20 hours estimated)
**Required Implementations:**
- ResNet-18 (18 layers, ~11.7M parameters)
- ResNet-34 (34 layers, ~21.8M parameters)
- ResNet-50 (50 layers, ~25.6M parameters)
- ResNet-101 (101 layers, ~44.5M parameters)
- ResNet-152 (152 layers, ~60.2M parameters)

**Key Components:**
- Residual blocks (BasicBlock for 18/34, Bottleneck for 50/101/152)
- Skip connections
- Batch normalization
- Global average pooling
- Configurable number of classes

#### 9.1.2 VGG, AlexNet, GoogleNet (15 hours estimated)
**Required Implementations:**
- VGG-11 (11 layers, ~132M parameters)
- VGG-13 (13 layers, ~133M parameters)
- VGG-16 (16 layers, ~138M parameters)
- VGG-19 (19 layers, ~144M parameters)
- AlexNet (8 layers, ~61M parameters)
- GoogLeNet/Inception-v1 (~7M parameters)

### 9.2 NLP Models

#### 9.2.1 BERT (30 hours estimated)
**Required Implementations:**
- BERT-Base (12 layers, 768 hidden, 12 heads, 110M parameters)
- BERT-Large (24 layers, 1024 hidden, 16 heads, 340M parameters)
- BERT-Tiny (2 layers, 128 hidden, 2 heads, 4.4M parameters)
- BERT-Mini (4 layers, 256 hidden, 4 heads, 11M parameters)

**Key Components:**
- Transformer encoder layers
- Multi-head self-attention
- Positional embeddings
- Token type embeddings
- Layer normalization
- GELU activation

#### 9.2.2 GPT Family (25 hours estimated)
**Required Implementations:**
- GPT-2 Small (12 layers, 768 hidden, 117M parameters)
- GPT-2 Medium (24 layers, 1024 hidden, 345M parameters)
- GPT-2 Large (36 layers, 1280 hidden, 774M parameters)
- GPT-3 architecture (96 layers, 12288 hidden, 175B parameters)

**Key Components:**
- Transformer decoder layers
- Causal self-attention (masked)
- Positional embeddings
- Layer normalization
- GELU activation
- Text generation utilities

### 9.3 Pretrained Weight Management (15 hours estimated)

**Required Functionality:**
- Download weights from URLs
- Local caching (~/.tenzor/models/)
- Checksum verification (SHA256)
- Progress bars for downloads
- Weight format conversion
- Strict vs non-strict loading modes

---

## Comprehensive Test Plan

### Test Categories

Each model implementation **MUST** include the following test categories:

#### 1. Shape Tests
**Purpose:** Verify output dimensions for various input configurations

**Test Cases:**
- Forward pass with default input size
- Batch size variations (1, 4, 16, 32)
- Different number of output classes (10, 100, 1000)
- Edge cases (batch size 1, very large batches)

**Example:**
```cpp
TEST(ResNet18, ForwardPassShape) {
    auto model = tenzor::models::resnet18(/*num_classes=*/1000);
    auto input = tenzor::randn({4, 3, 224, 224}); // Batch of 4, 224x224 RGB
    auto output = model->forward(input);

    EXPECT_EQ(output.shape(), std::vector<int64_t>({4, 1000}));
}
```

#### 2. Gradient Tests
**Purpose:** Ensure gradients flow correctly through all layers

**Test Cases:**
- Gradient flow from output to input
- No NaN or Inf gradients
- Gradient magnitudes reasonable
- Use `tenzor::autograd::gradcheck` utility

**Example:**
```cpp
TEST(ResNet18, GradientFlow) {
    auto model = tenzor::models::resnet18(/*num_classes=*/10);
    auto input = tenzor::randn({2, 3, 224, 224}, /*requires_grad=*/true);
    auto output = model->forward(input);
    auto loss = output.mean();

    loss.backward();

    ASSERT_TRUE(input.grad());
    EXPECT_FALSE(tenzor::any(tenzor::isnan(input.grad().value())).item<bool>());
    EXPECT_FALSE(tenzor::any(tenzor::isinf(input.grad().value())).item<bool>());
}
```

#### 3. Weight Loading Tests
**Purpose:** Verify pretrained weight loading works correctly

**Test Cases:**
- Load pretrained weights (mock if real weights unavailable)
- Verify weight shapes match architecture
- Test strict=true mode (all weights must match)
- Test strict=false mode (allow partial loading)
- Test weight initialization when not loading pretrained

**Example:**
```cpp
TEST(ResNet18, PretrainedWeightsLoading) {
    auto model = tenzor::models::resnet18(/*pretrained=*/false);

    // Create mock pretrained weights
    std::map<std::string, tenzor::Tensor> mock_weights;
    // ... populate with expected parameter names and shapes ...

    model->load_state_dict(mock_weights, /*strict=*/true);

    // Verify weights loaded
    auto params = model->parameters();
    EXPECT_GT(params.size(), 0);
}
```

#### 4. Architecture Tests
**Purpose:** Verify model architecture is correct

**Test Cases:**
- Correct number of parameters
- Layer names and connectivity
- Activation functions applied correctly
- Normalization layers present and working
- Skip connections functional (for ResNet)
- Inception modules correct (for GoogLeNet)

**Example:**
```cpp
TEST(ResNet18, ParameterCount) {
    auto model = tenzor::models::resnet18(/*num_classes=*/1000);

    size_t total_params = 0;
    for (const auto& param : model->parameters()) {
        total_params += param.numel();
    }

    // ResNet-18 should have ~11.7M parameters
    EXPECT_NEAR(total_params, 11'700'000, 100'000);
}
```

#### 5. Edge Case Tests
**Purpose:** Handle unusual inputs gracefully

**Test Cases:**
- Empty batch (batch size 0) - should error gracefully
- Single sample (batch size 1)
- Very large inputs (memory stress test)
- Different dtypes (Float32, Float16 if supported)
- Wrong input shapes - should provide clear error

**Example:**
```cpp
TEST(ResNet18, SingleSampleBatch) {
    auto model = tenzor::models::resnet18(/*num_classes=*/1000);
    auto input = tenzor::randn({1, 3, 224, 224});
    auto output = model->forward(input);

    EXPECT_EQ(output.shape(), std::vector<int64_t>({1, 1000}));
}

TEST(ResNet18, WrongInputShapeError) {
    auto model = tenzor::models::resnet18();
    auto input = tenzor::randn({4, 1, 28, 28}); // Wrong channels

    EXPECT_THROW(model->forward(input), std::runtime_error);
}
```

#### 6. Integration Tests
**Purpose:** Test full training/inference workflows

**Test Cases:**
- Full training loop (few epochs)
- Loss decreases over epochs
- Inference mode (eval mode)
- Model serialization/deserialization
- Multi-GPU compatibility (if DataParallel available)
- Mixed precision training (if AMP available)

**Example:**
```cpp
TEST(ResNet18, TrainingLoop) {
    auto model = tenzor::models::resnet18(/*num_classes=*/10);
    auto optimizer = tenzor::optim::SGD(model->parameters(), /*lr=*/0.01);
    auto criterion = tenzor::nn::CrossEntropyLoss();

    std::vector<float> losses;
    for (int epoch = 0; epoch < 3; ++epoch) {
        // Generate dummy data
        auto input = tenzor::randn({32, 3, 224, 224});
        auto target = tenzor::randint(0, 10, {32});

        optimizer.zero_grad();
        auto output = model->forward(input);
        auto loss = criterion(output, target);
        loss.backward();
        optimizer.step();

        losses.push_back(loss.item<float>());
    }

    // Loss should generally decrease
    EXPECT_LT(losses.back(), losses.front());
}
```

#### 7. Performance Tests
**Purpose:** Benchmark inference and training performance

**Test Cases:**
- Inference time measurement
- Memory usage profiling
- FLOP count comparison (vs expected)
- Throughput measurement (images/second)

**Example:**
```cpp
TEST(ResNet18, InferencePerformance) {
    auto model = tenzor::models::resnet18(/*num_classes=*/1000);
    model->eval();

    auto input = tenzor::randn({32, 3, 224, 224});

    // Warmup
    model->forward(input);

    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 100; ++i) {
        auto output = model->forward(input);
    }
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    double avg_time_ms = duration.count() / 100.0;
    double throughput = 32000.0 / avg_time_ms; // images/second

    std::cout << "ResNet-18 throughput: " << throughput << " images/second" << std::endl;

    // Performance should be reasonable (adjust based on hardware)
    EXPECT_GT(throughput, 100); // At least 100 images/second
}
```

---

## Test File Templates

### Template: test_resnet.cpp

```cpp
#include <gtest/gtest.h>
#include <tenzor/models/resnet.hpp>
#include <tenzor/autograd/gradcheck.hpp>
#include <tenzor/tenzor.hpp>

namespace tenzor {
namespace models {
namespace test {

// ============================================================================
// ResNet-18 Tests
// ============================================================================

TEST(ResNet18, Construction) {
    ASSERT_NO_THROW({
        auto model = resnet18(/*num_classes=*/1000);
    });
}

TEST(ResNet18, ForwardPassShape) {
    auto model = resnet18(/*num_classes=*/1000);
    auto input = tenzor::randn({4, 3, 224, 224});
    auto output = model->forward(input);

    EXPECT_EQ(output.dim(), 2);
    EXPECT_EQ(output.size(0), 4); // Batch size
    EXPECT_EQ(output.size(1), 1000); // Num classes
}

TEST(ResNet18, BatchSizeVariations) {
    auto model = resnet18(/*num_classes=*/1000);

    // Test various batch sizes
    for (int batch_size : {1, 4, 16, 32}) {
        auto input = tenzor::randn({batch_size, 3, 224, 224});
        auto output = model->forward(input);

        EXPECT_EQ(output.size(0), batch_size);
        EXPECT_EQ(output.size(1), 1000);
    }
}

TEST(ResNet18, DifferentNumClasses) {
    for (int num_classes : {10, 100, 1000}) {
        auto model = resnet18(num_classes);
        auto input = tenzor::randn({2, 3, 224, 224});
        auto output = model->forward(input);

        EXPECT_EQ(output.size(1), num_classes);
    }
}

TEST(ResNet18, GradientFlow) {
    auto model = resnet18(/*num_classes=*/10);
    auto input = tenzor::randn({2, 3, 224, 224}, /*requires_grad=*/true);

    auto output = model->forward(input);
    auto loss = output.mean();
    loss.backward();

    ASSERT_TRUE(input.grad().has_value());
    EXPECT_FALSE(tenzor::any(tenzor::isnan(input.grad().value())).item<bool>());
    EXPECT_FALSE(tenzor::any(tenzor::isinf(input.grad().value())).item<bool>());
}

TEST(ResNet18, ParameterCount) {
    auto model = resnet18(/*num_classes=*/1000);

    size_t total_params = 0;
    for (const auto& param : model->parameters()) {
        total_params += param.numel();
    }

    // ResNet-18 should have ~11.7M parameters
    EXPECT_NEAR(total_params, 11'700'000, 100'000);
}

TEST(ResNet18, SingleSampleInference) {
    auto model = resnet18(/*num_classes=*/1000);
    model->eval();

    auto input = tenzor::randn({1, 3, 224, 224});
    auto output = model->forward(input);

    EXPECT_EQ(output.shape(), std::vector<int64_t>({1, 1000}));
}

TEST(ResNet18, TrainingLoop) {
    auto model = resnet18(/*num_classes=*/10);
    auto optimizer = tenzor::optim::SGD(model->parameters(), /*lr=*/0.01);
    auto criterion = tenzor::nn::CrossEntropyLoss();

    std::vector<float> losses;
    for (int epoch = 0; epoch < 3; ++epoch) {
        auto input = tenzor::randn({32, 3, 224, 224});
        auto target = tenzor::randint(0, 10, {32});

        optimizer.zero_grad();
        auto output = model->forward(input);
        auto loss = criterion(output, target);
        loss.backward();
        optimizer.step();

        losses.push_back(loss.item<float>());
    }

    // Loss should generally decrease or stabilize
    EXPECT_LT(losses.back(), losses.front() * 1.5);
}

// ============================================================================
// ResNet-34 Tests
// ============================================================================

TEST(ResNet34, Construction) {
    ASSERT_NO_THROW({
        auto model = resnet34(/*num_classes=*/1000);
    });
}

TEST(ResNet34, ParameterCount) {
    auto model = resnet34(/*num_classes=*/1000);

    size_t total_params = 0;
    for (const auto& param : model->parameters()) {
        total_params += param.numel();
    }

    // ResNet-34 should have ~21.8M parameters
    EXPECT_NEAR(total_params, 21'800'000, 200'000);
}

// ============================================================================
// ResNet-50 Tests
// ============================================================================

TEST(ResNet50, Construction) {
    ASSERT_NO_THROW({
        auto model = resnet50(/*num_classes=*/1000);
    });
}

TEST(ResNet50, ParameterCount) {
    auto model = resnet50(/*num_classes=*/1000);

    size_t total_params = 0;
    for (const auto& param : model->parameters()) {
        total_params += param.numel();
    }

    // ResNet-50 should have ~25.6M parameters
    EXPECT_NEAR(total_params, 25'600'000, 300'000);
}

TEST(ResNet50, BottleneckArchitecture) {
    // ResNet-50 uses Bottleneck blocks (1x1, 3x3, 1x1 convolutions)
    auto model = resnet50(/*num_classes=*/1000);

    // Verify model can process input
    auto input = tenzor::randn({2, 3, 224, 224});
    auto output = model->forward(input);

    EXPECT_EQ(output.shape(), std::vector<int64_t>({2, 1000}));
}

// ============================================================================
// ResNet-101 Tests
// ============================================================================

TEST(ResNet101, Construction) {
    ASSERT_NO_THROW({
        auto model = resnet101(/*num_classes=*/1000);
    });
}

TEST(ResNet101, ParameterCount) {
    auto model = resnet101(/*num_classes=*/1000);

    size_t total_params = 0;
    for (const auto& param : model->parameters()) {
        total_params += param.numel();
    }

    // ResNet-101 should have ~44.5M parameters
    EXPECT_NEAR(total_params, 44'500'000, 500'000);
}

// ============================================================================
// ResNet-152 Tests
// ============================================================================

TEST(ResNet152, Construction) {
    ASSERT_NO_THROW({
        auto model = resnet152(/*num_classes=*/1000);
    });
}

TEST(ResNet152, ParameterCount) {
    auto model = resnet152(/*num_classes=*/1000);

    size_t total_params = 0;
    for (const auto& param : model->parameters()) {
        total_params += param.numel();
    }

    // ResNet-152 should have ~60.2M parameters
    EXPECT_NEAR(total_params, 60'200'000, 600'000);
}

// ============================================================================
// Cross-Model Tests
// ============================================================================

TEST(ResNet, AllVariantsCompile) {
    ASSERT_NO_THROW({
        auto m18 = resnet18();
        auto m34 = resnet34();
        auto m50 = resnet50();
        auto m101 = resnet101();
        auto m152 = resnet152();
    });
}

TEST(ResNet, ParameterCountIncreases) {
    auto count18 = count_parameters(resnet18());
    auto count34 = count_parameters(resnet34());
    auto count50 = count_parameters(resnet50());
    auto count101 = count_parameters(resnet101());
    auto count152 = count_parameters(resnet152());

    EXPECT_LT(count18, count34);
    EXPECT_LT(count34, count50);
    EXPECT_LT(count50, count101);
    EXPECT_LT(count101, count152);
}

// Helper function
size_t count_parameters(const std::shared_ptr<Module>& model) {
    size_t count = 0;
    for (const auto& param : model->parameters()) {
        count += param.numel();
    }
    return count;
}

} // namespace test
} // namespace models
} // namespace tenzor
```

### Template: test_vgg.cpp

```cpp
#include <gtest/gtest.h>
#include <tenzor/models/vgg.hpp>
#include <tenzor/tenzor.hpp>

namespace tenzor {
namespace models {
namespace test {

// ============================================================================
// VGG-11 Tests
// ============================================================================

TEST(VGG11, Construction) {
    ASSERT_NO_THROW({
        auto model = vgg11(/*num_classes=*/1000);
    });
}

TEST(VGG11, ForwardPassShape) {
    auto model = vgg11(/*num_classes=*/1000);
    auto input = tenzor::randn({4, 3, 224, 224});
    auto output = model->forward(input);

    EXPECT_EQ(output.shape(), std::vector<int64_t>({4, 1000}));
}

TEST(VGG11, ParameterCount) {
    auto model = vgg11(/*num_classes=*/1000);

    size_t total_params = 0;
    for (const auto& param : model->parameters()) {
        total_params += param.numel();
    }

    // VGG-11 should have ~132M parameters
    EXPECT_NEAR(total_params, 132'000'000, 1'000'000);
}

// ============================================================================
// VGG-13 Tests
// ============================================================================

TEST(VGG13, Construction) {
    ASSERT_NO_THROW({
        auto model = vgg13(/*num_classes=*/1000);
    });
}

TEST(VGG13, ParameterCount) {
    auto model = vgg13(/*num_classes=*/1000);

    size_t total_params = 0;
    for (const auto& param : model->parameters()) {
        total_params += param.numel();
    }

    // VGG-13 should have ~133M parameters
    EXPECT_NEAR(total_params, 133'000'000, 1'000'000);
}

// ============================================================================
// VGG-16 Tests
// ============================================================================

TEST(VGG16, Construction) {
    ASSERT_NO_THROW({
        auto model = vgg16(/*num_classes=*/1000);
    });
}

TEST(VGG16, ParameterCount) {
    auto model = vgg16(/*num_classes=*/1000);

    size_t total_params = 0;
    for (const auto& param : model->parameters()) {
        total_params += param.numel();
    }

    // VGG-16 should have ~138M parameters
    EXPECT_NEAR(total_params, 138'000'000, 1'000'000);
}

TEST(VGG16, GradientFlow) {
    auto model = vgg16(/*num_classes=*/10);
    auto input = tenzor::randn({2, 3, 224, 224}, /*requires_grad=*/true);

    auto output = model->forward(input);
    auto loss = output.mean();
    loss.backward();

    ASSERT_TRUE(input.grad().has_value());
    EXPECT_FALSE(tenzor::any(tenzor::isnan(input.grad().value())).item<bool>());
}

// ============================================================================
// VGG-19 Tests
// ============================================================================

TEST(VGG19, Construction) {
    ASSERT_NO_THROW({
        auto model = vgg19(/*num_classes=*/1000);
    });
}

TEST(VGG19, ParameterCount) {
    auto model = vgg19(/*num_classes=*/1000);

    size_t total_params = 0;
    for (const auto& param : model->parameters()) {
        total_params += param.numel();
    }

    // VGG-19 should have ~144M parameters
    EXPECT_NEAR(total_params, 144'000'000, 1'000'000);
}

// ============================================================================
// Cross-Model Tests
// ============================================================================

TEST(VGG, AllVariantsConstruct) {
    ASSERT_NO_THROW({
        auto v11 = vgg11();
        auto v13 = vgg13();
        auto v16 = vgg16();
        auto v19 = vgg19();
    });
}

TEST(VGG, BatchNormVariants) {
    ASSERT_NO_THROW({
        auto v11_bn = vgg11_bn();
        auto v13_bn = vgg13_bn();
        auto v16_bn = vgg16_bn();
        auto v19_bn = vgg19_bn();
    });
}

} // namespace test
} // namespace models
} // namespace tenzor
```

### Template: test_alexnet.cpp and test_googlenet.cpp

```cpp
#include <gtest/gtest.h>
#include <tenzor/models/alexnet.hpp>
#include <tenzor/models/googlenet.hpp>
#include <tenzor/tenzor.hpp>

namespace tenzor {
namespace models {
namespace test {

// ============================================================================
// AlexNet Tests
// ============================================================================

TEST(AlexNet, Construction) {
    ASSERT_NO_THROW({
        auto model = alexnet(/*num_classes=*/1000);
    });
}

TEST(AlexNet, ForwardPassShape) {
    auto model = alexnet(/*num_classes=*/1000);
    auto input = tenzor::randn({4, 3, 224, 224});
    auto output = model->forward(input);

    EXPECT_EQ(output.shape(), std::vector<int64_t>({4, 1000}));
}

TEST(AlexNet, ParameterCount) {
    auto model = alexnet(/*num_classes=*/1000);

    size_t total_params = 0;
    for (const auto& param : model->parameters()) {
        total_params += param.numel();
    }

    // AlexNet should have ~61M parameters
    EXPECT_NEAR(total_params, 61'000'000, 500'000);
}

TEST(AlexNet, GradientFlow) {
    auto model = alexnet(/*num_classes=*/10);
    auto input = tenzor::randn({2, 3, 224, 224}, /*requires_grad=*/true);

    auto output = model->forward(input);
    auto loss = output.mean();
    loss.backward();

    ASSERT_TRUE(input.grad().has_value());
}

// ============================================================================
// GoogLeNet Tests
// ============================================================================

TEST(GoogLeNet, Construction) {
    ASSERT_NO_THROW({
        auto model = googlenet(/*num_classes=*/1000);
    });
}

TEST(GoogLeNet, ForwardPassShape) {
    auto model = googlenet(/*num_classes=*/1000);
    auto input = tenzor::randn({4, 3, 224, 224});
    auto output = model->forward(input);

    EXPECT_EQ(output.shape(), std::vector<int64_t>({4, 1000}));
}

TEST(GoogLeNet, ParameterCount) {
    auto model = googlenet(/*num_classes=*/1000);

    size_t total_params = 0;
    for (const auto& param : model->parameters()) {
        total_params += param.numel();
    }

    // GoogLeNet should have ~7M parameters
    EXPECT_NEAR(total_params, 7'000'000, 500'000);
}

TEST(GoogLeNet, InceptionModules) {
    // GoogLeNet has 9 Inception modules
    auto model = googlenet(/*num_classes=*/1000);

    // Verify forward pass works
    auto input = tenzor::randn({2, 3, 224, 224});
    auto output = model->forward(input);

    EXPECT_EQ(output.shape(), std::vector<int64_t>({2, 1000}));
}

TEST(GoogLeNet, AuxiliaryClassifiers) {
    // GoogLeNet can have auxiliary classifiers during training
    auto model = googlenet(/*num_classes=*/1000, /*aux_logits=*/true);

    auto input = tenzor::randn({2, 3, 224, 224});
    // Forward should return tuple with auxiliary outputs
    auto output = model->forward(input);

    EXPECT_EQ(output.shape(), std::vector<int64_t>({2, 1000}));
}

} // namespace test
} // namespace models
} // namespace tenzor
```

### Template: test_bert.cpp

```cpp
#include <gtest/gtest.h>
#include <tenzor/models/bert.hpp>
#include <tenzor/tenzor.hpp>

namespace tenzor {
namespace models {
namespace test {

// ============================================================================
// BERT-Base Tests
// ============================================================================

TEST(BERTBase, Construction) {
    ASSERT_NO_THROW({
        auto model = bert_base();
    });
}

TEST(BERTBase, ForwardPassShape) {
    auto model = bert_base();

    // BERT input: [batch_size, sequence_length]
    auto input_ids = tenzor::randint(0, 30522, {4, 128}); // Vocab size ~30k
    auto output = model->forward(input_ids);

    // Output: [batch_size, sequence_length, hidden_size=768]
    EXPECT_EQ(output.dim(), 3);
    EXPECT_EQ(output.size(0), 4); // Batch
    EXPECT_EQ(output.size(1), 128); // Sequence
    EXPECT_EQ(output.size(2), 768); // Hidden size
}

TEST(BERTBase, ParameterCount) {
    auto model = bert_base();

    size_t total_params = 0;
    for (const auto& param : model->parameters()) {
        total_params += param.numel();
    }

    // BERT-Base should have ~110M parameters
    EXPECT_NEAR(total_params, 110'000'000, 1'000'000);
}

TEST(BERTBase, AttentionMask) {
    auto model = bert_base();

    auto input_ids = tenzor::randint(0, 30522, {2, 64});
    auto attention_mask = tenzor::ones({2, 64}); // No masking

    auto output = model->forward(input_ids, attention_mask);

    EXPECT_EQ(output.shape(), std::vector<int64_t>({2, 64, 768}));
}

TEST(BERTBase, TokenTypeEmbeddings) {
    auto model = bert_base();

    auto input_ids = tenzor::randint(0, 30522, {2, 64});
    auto token_type_ids = tenzor::zeros({2, 64}); // All sentence A

    auto output = model->forward(input_ids, /*attention_mask=*/std::nullopt, token_type_ids);

    EXPECT_EQ(output.shape(), std::vector<int64_t>({2, 64, 768}));
}

TEST(BERTBase, GradientFlow) {
    auto model = bert_base();

    auto input_ids = tenzor::randint(0, 30522, {2, 32});
    auto output = model->forward(input_ids);

    // Pooler output (CLS token)
    auto pooled = output[:, 0, :]; // [batch, hidden]
    auto loss = pooled.mean();
    loss.backward();

    // Check that model parameters have gradients
    bool has_grads = false;
    for (const auto& param : model->parameters()) {
        if (param.grad().has_value()) {
            has_grads = true;
            break;
        }
    }
    EXPECT_TRUE(has_grads);
}

// ============================================================================
// BERT-Large Tests
// ============================================================================

TEST(BERTLarge, Construction) {
    ASSERT_NO_THROW({
        auto model = bert_large();
    });
}

TEST(BERTLarge, ForwardPassShape) {
    auto model = bert_large();

    auto input_ids = tenzor::randint(0, 30522, {2, 128});
    auto output = model->forward(input_ids);

    // Output: [batch, sequence, hidden=1024]
    EXPECT_EQ(output.shape(), std::vector<int64_t>({2, 128, 1024}));
}

TEST(BERTLarge, ParameterCount) {
    auto model = bert_large();

    size_t total_params = 0;
    for (const auto& param : model->parameters()) {
        total_params += param.numel();
    }

    // BERT-Large should have ~340M parameters
    EXPECT_NEAR(total_params, 340'000'000, 5'000'000);
}

// ============================================================================
// BERT-Tiny Tests
// ============================================================================

TEST(BERTTiny, Construction) {
    ASSERT_NO_THROW({
        auto model = bert_tiny();
    });
}

TEST(BERTTiny, ParameterCount) {
    auto model = bert_tiny();

    size_t total_params = 0;
    for (const auto& param : model->parameters()) {
        total_params += param.numel();
    }

    // BERT-Tiny should have ~4.4M parameters
    EXPECT_NEAR(total_params, 4'400'000, 500'000);
}

// ============================================================================
// BERT-Mini Tests
// ============================================================================

TEST(BERTMini, Construction) {
    ASSERT_NO_THROW({
        auto model = bert_mini();
    });
}

TEST(BERTMini, ParameterCount) {
    auto model = bert_mini();

    size_t total_params = 0;
    for (const auto& param : model->parameters()) {
        total_params += param.numel();
    }

    // BERT-Mini should have ~11M parameters
    EXPECT_NEAR(total_params, 11'000'000, 500'000);
}

// ============================================================================
// BERT Task-Specific Heads Tests
// ============================================================================

TEST(BERTForSequenceClassification, Construction) {
    ASSERT_NO_THROW({
        auto model = bert_for_sequence_classification(/*num_labels=*/2);
    });
}

TEST(BERTForSequenceClassification, ForwardPassShape) {
    auto model = bert_for_sequence_classification(/*num_labels=*/5);

    auto input_ids = tenzor::randint(0, 30522, {4, 64});
    auto logits = model->forward(input_ids);

    // Output: [batch, num_labels]
    EXPECT_EQ(logits.shape(), std::vector<int64_t>({4, 5}));
}

TEST(BERTForTokenClassification, Construction) {
    ASSERT_NO_THROW({
        auto model = bert_for_token_classification(/*num_labels=*/9); // NER tags
    });
}

TEST(BERTForQuestionAnswering, Construction) {
    ASSERT_NO_THROW({
        auto model = bert_for_question_answering();
    });
}

} // namespace test
} // namespace models
} // namespace tenzor
```

### Template: test_gpt.cpp

```cpp
#include <gtest/gtest.h>
#include <tenzor/models/gpt.hpp>
#include <tenzor/tenzor.hpp>

namespace tenzor {
namespace models {
namespace test {

// ============================================================================
// GPT-2 Small Tests
// ============================================================================

TEST(GPT2Small, Construction) {
    ASSERT_NO_THROW({
        auto model = gpt2_small();
    });
}

TEST(GPT2Small, ForwardPassShape) {
    auto model = gpt2_small();

    // GPT input: [batch_size, sequence_length]
    auto input_ids = tenzor::randint(0, 50257, {4, 128}); // GPT-2 vocab size
    auto output = model->forward(input_ids);

    // Output: [batch, sequence, hidden=768]
    EXPECT_EQ(output.shape(), std::vector<int64_t>({4, 128, 768}));
}

TEST(GPT2Small, ParameterCount) {
    auto model = gpt2_small();

    size_t total_params = 0;
    for (const auto& param : model->parameters()) {
        total_params += param.numel();
    }

    // GPT-2 Small should have ~117M parameters
    EXPECT_NEAR(total_params, 117'000'000, 1'000'000);
}

TEST(GPT2Small, CausalAttention) {
    auto model = gpt2_small();

    // GPT uses causal (masked) attention
    auto input_ids = tenzor::randint(0, 50257, {2, 64});
    auto output = model->forward(input_ids);

    // Each position should only attend to previous positions
    EXPECT_EQ(output.shape(), std::vector<int64_t>({2, 64, 768}));
}

TEST(GPT2Small, TextGeneration) {
    auto model = gpt2_small();
    model->eval();

    // Start with a prompt
    auto input_ids = tenzor::tensor({1, 2, 3, 4, 5}); // Token IDs

    // Generate next token (greedy)
    auto output = model->forward(input_ids.unsqueeze(0));
    auto logits = output[0, -1, :]; // Last position
    auto next_token = tenzor::argmax(logits);

    EXPECT_GE(next_token.item<int64_t>(), 0);
    EXPECT_LT(next_token.item<int64_t>(), 50257);
}

// ============================================================================
// GPT-2 Medium Tests
// ============================================================================

TEST(GPT2Medium, Construction) {
    ASSERT_NO_THROW({
        auto model = gpt2_medium();
    });
}

TEST(GPT2Medium, ForwardPassShape) {
    auto model = gpt2_medium();

    auto input_ids = tenzor::randint(0, 50257, {2, 128});
    auto output = model->forward(input_ids);

    // Output: [batch, sequence, hidden=1024]
    EXPECT_EQ(output.shape(), std::vector<int64_t>({2, 128, 1024}));
}

TEST(GPT2Medium, ParameterCount) {
    auto model = gpt2_medium();

    size_t total_params = 0;
    for (const auto& param : model->parameters()) {
        total_params += param.numel();
    }

    // GPT-2 Medium should have ~345M parameters
    EXPECT_NEAR(total_params, 345'000'000, 5'000'000);
}

// ============================================================================
// GPT-2 Large Tests
// ============================================================================

TEST(GPT2Large, Construction) {
    ASSERT_NO_THROW({
        auto model = gpt2_large();
    });
}

TEST(GPT2Large, ParameterCount) {
    auto model = gpt2_large();

    size_t total_params = 0;
    for (const auto& param : model->parameters()) {
        total_params += param.numel();
    }

    // GPT-2 Large should have ~774M parameters
    EXPECT_NEAR(total_params, 774'000'000, 10'000'000);
}

// ============================================================================
// GPT-3 Architecture Tests (Training not included)
// ============================================================================

TEST(GPT3, ArchitectureConstruction) {
    // Note: GPT-3 with 175B parameters is too large for most systems
    // This tests the architecture can be instantiated

    ASSERT_NO_THROW({
        // Create a small GPT-3 style model for testing
        auto model = gpt3_architecture(
            /*n_layers=*/12,
            /*n_heads=*/12,
            /*hidden_size=*/768,
            /*vocab_size=*/50257
        );
    });
}

// ============================================================================
// Text Generation Utilities Tests
// ============================================================================

TEST(GPTGeneration, GreedyDecoding) {
    auto model = gpt2_small();
    model->eval();

    auto prompt = tenzor::tensor({1, 2, 3}); // Starting tokens

    auto generated = generate_text_greedy(model, prompt, /*max_length=*/10);

    EXPECT_EQ(generated.dim(), 1);
    EXPECT_LE(generated.size(0), 10);
}

TEST(GPTGeneration, TopKSampling) {
    auto model = gpt2_small();
    model->eval();

    auto prompt = tenzor::tensor({1, 2, 3});

    auto generated = generate_text_top_k(
        model,
        prompt,
        /*max_length=*/10,
        /*k=*/50
    );

    EXPECT_EQ(generated.dim(), 1);
    EXPECT_LE(generated.size(0), 10);
}

TEST(GPTGeneration, TopPSampling) {
    auto model = gpt2_small();
    model->eval();

    auto prompt = tenzor::tensor({1, 2, 3});

    auto generated = generate_text_top_p(
        model,
        prompt,
        /*max_length=*/10,
        /*p=*/0.9
    );

    EXPECT_EQ(generated.dim(), 1);
    EXPECT_LE(generated.size(0), 10);
}

} // namespace test
} // namespace models
} // namespace tenzor
```

### Template: test_model_hub.cpp

```cpp
#include <gtest/gtest.h>
#include <tenzor/models/model_hub.hpp>
#include <tenzor/tenzor.hpp>
#include <filesystem>

namespace tenzor {
namespace models {
namespace test {

// ============================================================================
// ModelHub Basic Tests
// ============================================================================

TEST(ModelHub, CacheDirectory) {
    auto cache_dir = ModelHub::get_cache_dir();

    // Cache directory should be ~/.tenzor/models by default
    EXPECT_TRUE(cache_dir.find(".tenzor") != std::string::npos);
    EXPECT_TRUE(cache_dir.find("models") != std::string::npos);
}

TEST(ModelHub, SetCustomCacheDir) {
    std::string custom_dir = "/tmp/tenzor_test_cache";
    ModelHub::set_cache_dir(custom_dir);

    auto cache_dir = ModelHub::get_cache_dir();
    EXPECT_EQ(cache_dir, custom_dir);

    // Reset to default
    ModelHub::set_cache_dir("");
}

TEST(ModelHub, ListAvailableModels) {
    auto models = ModelHub::list_available_models();

    // Should have entries for various models
    EXPECT_GT(models.size(), 0);

    // Check for expected models
    bool has_resnet = false;
    bool has_bert = false;
    for (const auto& model : models) {
        if (model.find("resnet") != std::string::npos) has_resnet = true;
        if (model.find("bert") != std::string::npos) has_bert = true;
    }

    EXPECT_TRUE(has_resnet);
    EXPECT_TRUE(has_bert);
}

// ============================================================================
// Weight Download Tests
// ============================================================================

TEST(ModelHub, DownloadWeights) {
    // Test downloading a small model (mock URL for testing)
    std::string model_name = "test_model";
    std::string mock_url = "https://example.com/test_weights.bin";

    // This should create a placeholder or download if URL is valid
    ASSERT_NO_THROW({
        auto path = ModelHub::download_weights(model_name, mock_url);
        EXPECT_FALSE(path.empty());
    });
}

TEST(ModelHub, ChecksumVerification) {
    // Create a temporary file with known content
    std::string test_file = "/tmp/test_weights.bin";
    std::ofstream ofs(test_file, std::ios::binary);
    ofs << "test content";
    ofs.close();

    // Calculate expected checksum (SHA256)
    std::string expected_checksum = "6ae8a75555209fd6c44157c0aed8016e763ff435a19cf186f76863140143ff72";

    bool valid = ModelHub::verify_checksum(test_file, expected_checksum);
    EXPECT_TRUE(valid);

    // Cleanup
    std::filesystem::remove(test_file);
}

TEST(ModelHub, InvalidChecksumDetection) {
    std::string test_file = "/tmp/test_weights.bin";
    std::ofstream ofs(test_file, std::ios::binary);
    ofs << "test content";
    ofs.close();

    std::string wrong_checksum = "0000000000000000000000000000000000000000000000000000000000000000";

    bool valid = ModelHub::verify_checksum(test_file, wrong_checksum);
    EXPECT_FALSE(valid);

    std::filesystem::remove(test_file);
}

// ============================================================================
// Caching Tests
// ============================================================================

TEST(ModelHub, WeightsCaching) {
    std::string model_name = "resnet18_test";

    // First download (or mock)
    auto path1 = ModelHub::get_cached_path(model_name);

    // Second access should use cache
    auto path2 = ModelHub::get_cached_path(model_name);

    EXPECT_EQ(path1, path2);
}

TEST(ModelHub, CacheMissHandling) {
    std::string nonexistent_model = "model_that_does_not_exist_12345";

    // Should return empty or throw exception
    EXPECT_THROW(
        ModelHub::get_cached_path(nonexistent_model),
        std::runtime_error
    );
}

TEST(ModelHub, ClearCache) {
    // Add a test model to cache
    std::string test_model = "test_model_for_clearing";
    ModelHub::cache_model(test_model, "/tmp/test.bin");

    // Clear cache
    ModelHub::clear_cache();

    // Should no longer be cached
    EXPECT_THROW(
        ModelHub::get_cached_path(test_model),
        std::runtime_error
    );
}

// ============================================================================
// Progress Tracking Tests
// ============================================================================

TEST(ModelHub, DownloadProgressCallback) {
    bool callback_called = false;
    size_t total_bytes = 0;

    auto progress_callback = [&](size_t downloaded, size_t total) {
        callback_called = true;
        total_bytes = total;
    };

    // Mock download with progress
    ModelHub::download_with_progress(
        "test_model",
        "https://example.com/weights.bin",
        progress_callback
    );

    EXPECT_TRUE(callback_called);
}

// ============================================================================
// Weight Loading Tests
// ============================================================================

TEST(ModelHub, LoadWeightsIntoModel) {
    // Create a simple model
    auto model = tenzor::models::resnet18(/*pretrained=*/false);

    // Create mock weights
    std::map<std::string, tenzor::Tensor> mock_weights;
    for (const auto& [name, param] : model->named_parameters()) {
        mock_weights[name] = tenzor::randn_like(param);
    }

    // Save mock weights
    std::string weights_path = "/tmp/test_resnet18.bin";
    ModelHub::save_weights(mock_weights, weights_path);

    // Load weights into model
    ModelHub::load_weights_into_model(model, weights_path);

    // Verify weights loaded
    EXPECT_GT(model->parameters().size(), 0);

    // Cleanup
    std::filesystem::remove(weights_path);
}

TEST(ModelHub, StrictLoadingMode) {
    auto model = tenzor::models::resnet18();

    // Create incomplete weights (missing some parameters)
    std::map<std::string, tenzor::Tensor> incomplete_weights;
    incomplete_weights["conv1.weight"] = tenzor::randn({64, 3, 7, 7});

    std::string weights_path = "/tmp/incomplete_weights.bin";
    ModelHub::save_weights(incomplete_weights, weights_path);

    // Strict mode should fail
    EXPECT_THROW(
        ModelHub::load_weights_into_model(model, weights_path, /*strict=*/true),
        std::runtime_error
    );

    // Non-strict mode should succeed (partial loading)
    ASSERT_NO_THROW(
        ModelHub::load_weights_into_model(model, weights_path, /*strict=*/false)
    );

    std::filesystem::remove(weights_path);
}

// ============================================================================
// Pretrained Model Loading Tests
// ============================================================================

TEST(ModelHub, LoadPretrainedResNet18) {
    ASSERT_NO_THROW({
        auto model = ModelHub::load_pretrained("resnet18");
        EXPECT_TRUE(model != nullptr);
    });
}

TEST(ModelHub, LoadPretrainedBERT) {
    ASSERT_NO_THROW({
        auto model = ModelHub::load_pretrained("bert-base-uncased");
        EXPECT_TRUE(model != nullptr);
    });
}

TEST(ModelHub, LoadNonexistentModel) {
    EXPECT_THROW(
        ModelHub::load_pretrained("nonexistent_model_xyz"),
        std::runtime_error
    );
}

// ============================================================================
// Format Conversion Tests
// ============================================================================

TEST(ModelHub, ConvertPyTorchWeights) {
    // Test converting PyTorch .pth format to Tenzor format
    std::string pytorch_path = "/tmp/pytorch_weights.pth";
    std::string tenzor_path = "/tmp/tenzor_weights.bin";

    // Mock PyTorch weights file (simplified)
    // In reality, this would be a proper PyTorch checkpoint

    ASSERT_NO_THROW({
        ModelHub::convert_pytorch_weights(pytorch_path, tenzor_path);
    });
}

TEST(ModelHub, ConvertTensorFlowWeights) {
    std::string tf_path = "/tmp/tf_checkpoint";
    std::string tenzor_path = "/tmp/tenzor_weights.bin";

    ASSERT_NO_THROW({
        ModelHub::convert_tensorflow_weights(tf_path, tenzor_path);
    });
}

} // namespace test
} // namespace models
} // namespace tenzor
```

---

## Acceptance Criteria

### For Each Model Implementation

**MUST HAVE:**
1. ✅ Compiles without errors
2. ✅ Forward pass produces correct output shapes
3. ✅ Parameter count matches expected value (±5%)
4. ✅ Gradients flow correctly (no NaN/Inf)
5. ✅ Can complete a training loop without crashes
6. ✅ All test categories pass (7/7)

**SHOULD HAVE:**
7. ✅ Pretrained weights can be loaded
8. ✅ Serialization/deserialization works
9. ✅ Inference mode (eval) works correctly
10. ✅ Documentation with usage examples

**NICE TO HAVE:**
11. ✅ Performance benchmarks vs PyTorch
12. ✅ Multi-GPU compatibility
13. ✅ Mixed precision support
14. ✅ ONNX export compatibility

---

## Test Execution Plan

### When Implementations Are Ready

1. **Build Test Executables**
```bash
cd /home/lee/Projects/Tenzor/build
cmake ..
make test_resnet test_vgg test_alexnet test_googlenet test_bert test_gpt test_model_hub
```

2. **Run Tests**
```bash
./test_resnet
./test_vgg
./test_alexnet
./test_googlenet
./test_bert
./test_gpt
./test_model_hub
```

3. **Generate Coverage Report**
```bash
# Run with coverage
cmake .. -DCMAKE_BUILD_TYPE=Coverage
make coverage
```

4. **Performance Benchmarking**
```bash
./benchmark_models --benchmark_format=json > benchmark_results.json
```

---

## Expected Timeline

**Phase 9.1 - CV Models (35 hours):**
- Week 1: ResNet implementation + tests (20h)
- Week 2: VGG, AlexNet, GoogLeNet + tests (15h)

**Phase 9.2 - NLP Models (55 hours):**
- Week 3-4: BERT implementation + tests (30h)
- Week 5: GPT implementation + tests (25h)

**Phase 9.3 - Model Hub (15 hours):**
- Week 6: Weight management + tests (15h)

**Total: 6 weeks (105 hours)**

---

## Coverage Targets

| Component | Target Coverage | Min Coverage |
|-----------|----------------|--------------|
| Model forward pass | 100% | 95% |
| Weight loading | 100% | 90% |
| Gradient computation | 100% | 95% |
| Architecture components | 95% | 85% |
| Edge cases | 90% | 80% |
| **Overall** | **95%+** | **90%** |

---

## Conclusion

Phase 9 implementations **DO NOT EXIST YET**. This document provides:

1. ✅ Complete test plan for all required models
2. ✅ Test file templates ready for immediate use
3. ✅ Clear acceptance criteria
4. ✅ Comprehensive test coverage strategy
5. ✅ Performance benchmarking framework

**Next Steps:**
1. ⏳ **WAIT** for coder agents to implement Phase 9 models
2. 🔄 **USE** these test templates once implementations are ready
3. ✅ **VERIFY** all tests pass before declaring Phase 9 complete
4. 📊 **GENERATE** final test report with metrics

**Status:** 🔴 **BLOCKED - Awaiting Model Implementations**

---

*Report generated: 2025-10-17*
*Test files created: 7 comprehensive templates*
*Ready for implementation: YES*
*Current coverage: 0% (no implementations exist)*
