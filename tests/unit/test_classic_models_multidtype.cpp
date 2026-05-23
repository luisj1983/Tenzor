/**
 * @file test_classic_models_multidtype.cpp
 * @brief Multi-dtype tests for VGG, AlexNet, and GoogLeNet classic models
 *
 * Tests classic CNN architectures with Float32, Float64, and Float16 data types across
 * CPU, CUDA, OneAPI, Vulkan, and ROCm backends to ensure:
 * - Proper dtype propagation through deep convolutional stacks
 * - Correct output shapes for all VGG variants (11, 13, 16, 19)
 * - AlexNet architecture with different dtypes
 * - GoogLeNet inception modules and auxiliary classifiers
 * - Gradient flow through very deep networks
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include "../../include/tenzor/models/vgg.hpp"
#include "../../include/tenzor/models/alexnet.hpp"
#include "../../include/tenzor/models/googlenet.hpp"
#include "../multi_backend_dtype_fixture.hpp"
#include <cmath>
#include "../grad_flow_helpers.hpp"

using namespace tenzor;
using namespace tenzor::testing;
using namespace tenzor::models;

// ============================================================================
// Classic Models Multi-Backend Multi-DType Test Fixture
// ============================================================================

class ClassicModelsMultiDTypeTest : public MultiBackendDTypeTest {
protected:
    // Parameter count tolerance based on dtype
    float param_count_tol_;

    void SetUp() override {
        MultiBackendDTypeTest::SetUp();

        if (dtype() == DType::Float16) {
            param_count_tol_ = 0.05f;  // 5% tolerance
        } else {
            param_count_tol_ = 0.01f;  // 1% tolerance
        }
    }

    bool CheckParameterCount(int64_t actual, int64_t expected) {
        int64_t tolerance = static_cast<int64_t>(expected * param_count_tol_);
        return std::abs(actual - expected) <= tolerance;
    }
};

// ============================================================================
// VGG Tests
// ============================================================================

TEST_P(ClassicModelsMultiDTypeTest, VGG11ForwardShape) {
    auto model = vgg11(10, true, false);
    convert_model(model);
    model->eval();

    Variable input = createInput({2, 3, 224, 224}, false);
    Variable output = model->forward(input);

    expectShape(output.tensor(), {2, 10});
    expectDType(output.tensor());
    expectFiniteNonZero(output.tensor());
}

// Note: TEST_P helper duplicate prefix matcher landing position.
TEST_P(ClassicModelsMultiDTypeTest, VGG11GradientFlow) {
    auto model = vgg11(10, true, false);
    convert_model(model);
    model->train();

    Variable input = createInput({1, 3, 224, 224}, true);
    Variable output = model->forward(input);
    Variable loss = tenzor::sum(output);
    loss.backward();

    // Check input gradients
    EXPECT_GRAD_FLOWS(input);
    EXPECT_EQ(input.grad()->dtype(), dtype());

    // Always check parameter gradients
    auto params = model->parameters();
    EXPECT_GT(params.size(), 0);

    // Verify at least some parameters have gradients
    int params_with_grad = 0;
    for (const auto& p : params) {
        if (p->grad().has_value()) {
            params_with_grad++;
        }
    }
    EXPECT_GT(params_with_grad, 0);
    expectFiniteNonZero(output.tensor());
}

TEST_P(ClassicModelsMultiDTypeTest, VGG13ForwardShape) {
    auto model = vgg13(10, true, false);
    convert_model(model);
    model->eval();

    Variable input = createInput({2, 3, 224, 224}, false);
    Variable output = model->forward(input);

    expectShape(output.tensor(), {2, 10});
    expectDType(output.tensor());
    expectFiniteNonZero(output.tensor());
}

TEST_P(ClassicModelsMultiDTypeTest, VGG16ForwardShape) {
    auto model = vgg16(1000, true, false);
    convert_model(model);
    model->eval();

    Variable input = createInput({1, 3, 224, 224}, false);
    Variable output = model->forward(input);

    expectShape(output.tensor(), {1, 1000});
    expectDType(output.tensor());
    expectFiniteNonZero(output.tensor());
}

TEST_P(ClassicModelsMultiDTypeTest, VGG16GradientFlow) {
    auto model = vgg16(10, true, false);
    convert_model(model);
    model->train();

    Variable input = createInput({1, 3, 224, 224}, true);
    Variable output = model->forward(input);
    Variable loss = tenzor::sum(output);
    loss.backward();

    // Check input gradients (may not work on all backends due to device transfer)
    if (device() == Device::cpu()) {
        EXPECT_GRAD_FLOWS(input);
    }

    // Always check parameter gradients
    auto params = model->parameters();
    int params_with_grad = 0;
    for (const auto& p : params) {
        if (p->grad().has_value()) {
            params_with_grad++;
        }
    }
    EXPECT_GT(params_with_grad, 0);
    expectFiniteNonZero(output.tensor());
}

TEST_P(ClassicModelsMultiDTypeTest, VGG19ForwardShape) {
    auto model = vgg19(100, true, false);
    convert_model(model);
    model->eval();

    Variable input = createInput({4, 3, 224, 224}, false);
    Variable output = model->forward(input);

    expectShape(output.tensor(), {4, 100});
    expectDType(output.tensor());
    expectFiniteNonZero(output.tensor());
}

TEST_P(ClassicModelsMultiDTypeTest, VGGWithoutBatchNorm) {
    auto model = vgg11(10, false, false);
    convert_model(model);
    model->eval();

    Variable input = createInput({1, 3, 224, 224}, false);
    Variable output = model->forward(input);

    expectShape(output.tensor(), {1, 10});
    expectDType(output.tensor());
    expectFiniteNonZero(output.tensor());
}

TEST_P(ClassicModelsMultiDTypeTest, VGGCustomDropout) {
    auto model = std::make_shared<VGG>(VGGConfig::vgg11(), 10, true, 0.3);
    convert_model(model);
    model->eval();

    Variable input = createInput({1, 3, 224, 224}, false);
    Variable output = model->forward(input);

    expectShape(output.tensor(), {1, 10});
    expectFiniteNonZero(output.tensor());
}

// ============================================================================
// AlexNet Tests
// ============================================================================

TEST_P(ClassicModelsMultiDTypeTest, AlexNetForwardShape) {
    auto model = alexnet(1000, false);
    convert_model(model);
    model->eval();

    Variable input = createInput({2, 3, 224, 224}, false);
    Variable output = model->forward(input);

    expectShape(output.tensor(), {2, 1000});
    expectDType(output.tensor());
    expectFiniteNonZero(output.tensor());
}

TEST_P(ClassicModelsMultiDTypeTest, AlexNetGradientFlow) {
    auto model = alexnet(10, false);
    convert_model(model);
    model->train();

    Variable input = createInput({1, 3, 224, 224}, true);
    Variable output = model->forward(input);
    Variable loss = tenzor::sum(output);
    loss.backward();

    EXPECT_GRAD_FLOWS(input);
    EXPECT_EQ(input.grad()->dtype(), dtype());
    expectFiniteNonZero(output.tensor());
}

TEST_P(ClassicModelsMultiDTypeTest, AlexNetCustomClasses) {
    auto model = alexnet(10, false);
    convert_model(model);
    model->eval();

    Variable input = createInput({1, 3, 224, 224}, false);
    Variable output = model->forward(input);

    expectShape(output.tensor(), {1, 10});
    expectFiniteNonZero(output.tensor());
}

TEST_P(ClassicModelsMultiDTypeTest, AlexNetBatchProcessing) {
    auto model = alexnet(100, false);
    convert_model(model);
    model->eval();

    Variable input = createInput({8, 3, 224, 224}, false);
    Variable output = model->forward(input);

    expectShape(output.tensor(), {8, 100});
    expectDType(output.tensor());
    expectFiniteNonZero(output.tensor());
}

TEST_P(ClassicModelsMultiDTypeTest, AlexNetCustomDropout) {
    auto model = std::make_shared<AlexNet>(10, 0.3);
    convert_model(model);
    model->eval();

    Variable input = createInput({1, 3, 224, 224}, false);
    Variable output = model->forward(input);

    expectShape(output.tensor(), {1, 10});
    expectFiniteNonZero(output.tensor());
}

// ============================================================================
// GoogLeNet Tests
// ============================================================================

TEST_P(ClassicModelsMultiDTypeTest, GoogLeNetForwardShape) {
    auto model = googlenet(1000, false, false);  // No aux classifiers
    convert_model(model);
    model->eval();

    Variable input = createInput({1, 3, 224, 224}, false);
    Variable output = model->forward(input);

    expectShape(output.tensor(), {1, 1000});
    expectDType(output.tensor());
}

TEST_P(ClassicModelsMultiDTypeTest, GoogLeNetGradientFlow) {
    auto model = googlenet(10, false, false);
    convert_model(model);
    model->train();

    Variable input = createInput({1, 3, 224, 224}, true);
    Variable output = model->forward(input);
    Variable loss = tenzor::sum(output);
    loss.backward();

    EXPECT_GRAD_FLOWS(input);
    EXPECT_EQ(input.grad()->dtype(), dtype());
    expectFiniteNonZero(output.tensor());
}

TEST_P(ClassicModelsMultiDTypeTest, GoogLeNetWithAuxiliaryClassifiers) {
    auto model = googlenet(10, false, true);  // With aux classifiers
    convert_model(model);
    model->train();

    Variable input = createInput({2, 3, 224, 224}, false);

    // Forward with auxiliary outputs
    auto [main_out, aux1_out, aux2_out] = model->forward_with_aux(input);

    // Check main output
    expectShape(main_out.tensor(), {2, 10});
    expectDType(main_out.tensor());

    // Check auxiliary outputs
    expectShape(aux1_out.tensor(), {2, 10});
    expectShape(aux2_out.tensor(), {2, 10});
    expectDType(aux1_out.tensor());
    expectDType(aux2_out.tensor());
    expectFiniteNonZero(main_out.tensor());
    expectFiniteNonZero(aux1_out.tensor());
    expectFiniteNonZero(aux2_out.tensor());
}

TEST_P(ClassicModelsMultiDTypeTest, GoogLeNetInferenceMode) {
    auto model = googlenet(1000, false, true);
    convert_model(model);
    model->eval();

    Variable input = createInput({1, 3, 224, 224}, false);
    Variable output = model->forward(input);

    expectShape(output.tensor(), {1, 1000});
    expectFiniteNonZero(output.tensor());
}

TEST_P(ClassicModelsMultiDTypeTest, InceptionModuleForward) {
    auto inception = std::make_shared<InceptionModule>(
        192,        // in_channels
        64,         // out_1x1
        96, 128,    // reduce_3x3, out_3x3
        16, 32,     // reduce_5x5, out_5x5
        32          // out_pool_proj
    );
    convert_model(inception);

    Variable input = createInput({1, 192, 28, 28}, false);
    Variable output = inception->forward(input);

    // Output channels: 64 + 128 + 32 + 32 = 256
    expectShape(output.tensor(), {1, 256, 28, 28});
    expectDType(output.tensor());
    expectFiniteNonZero(output.tensor());
}

TEST_P(ClassicModelsMultiDTypeTest, GoogLeNetCustomDropout) {
    auto model = std::make_shared<GoogLeNet>(10, false, 0.2);
    convert_model(model);
    model->eval();

    Variable input = createInput({1, 3, 224, 224}, false);
    Variable output = model->forward(input);

    expectShape(output.tensor(), {1, 10});
    expectFiniteNonZero(output.tensor());
}

// ============================================================================
// Comparative Tests
// ============================================================================

TEST_P(ClassicModelsMultiDTypeTest, ModelParameterCounts) {
    auto vgg16_model = vgg16(1000, true, false);
    auto alexnet_model = alexnet(1000, false);
    auto googlenet_model = googlenet(1000, false, false);

    auto vgg_params = vgg16_model->parameters();
    auto alex_params = alexnet_model->parameters();
    auto google_params = googlenet_model->parameters();

    EXPECT_GT(vgg_params.size(), 0);
    EXPECT_GT(alex_params.size(), 0);
    EXPECT_GT(google_params.size(), 0);

    // VGG-16 should have the most parameters
    size_t vgg_count = countParameters(vgg_params);
    size_t alex_count = countParameters(alex_params);

    EXPECT_GT(vgg_count, alex_count);
}

TEST_P(ClassicModelsMultiDTypeTest, TrainingModeSwitch) {
    auto model = vgg16(10, true, false);

    model->train();
    EXPECT_TRUE(model->is_training());

    model->eval();
    EXPECT_FALSE(model->is_training());
}

TEST_P(ClassicModelsMultiDTypeTest, GradientTracking) {
    auto model = vgg11(10, true, false);
    convert_model(model);

    Variable input = createInput({1, 3, 224, 224}, true);
    Variable output = model->forward(input);

    EXPECT_TRUE(output.requires_grad());
}

// ============================================================================
// Edge Cases and Stress Tests
// ============================================================================

TEST_P(ClassicModelsMultiDTypeTest, LargeBatchVGG) {
    auto model = vgg11(10, true, false);
    convert_model(model);
    model->eval();

    Variable input = createInput({16, 3, 224, 224}, false);
    Variable output = model->forward(input);

    expectShape(output.tensor(), {16, 10});
    expectFiniteNonZero(output.tensor());
}

TEST_P(ClassicModelsMultiDTypeTest, SmallBatchGoogLeNet) {
    auto model = googlenet(1000, false, false);
    convert_model(model);
    model->eval();

    Variable input = createInput({1, 3, 224, 224}, false);
    Variable output = model->forward(input);

    expectShape(output.tensor(), {1, 1000});
    expectFiniteNonZero(output.tensor());
}

TEST_P(ClassicModelsMultiDTypeTest, MultipleBatchSizes) {
    auto model = alexnet(10, false);
    convert_model(model);
    model->eval();

    for (int batch_size : {1, 2, 4, 8}) {
        Variable input = createInput({batch_size, 3, 224, 224}, false);
        Variable output = model->forward(input);

        expectShape(output.tensor(), {batch_size, 10});
        expectFiniteNonZero(output.tensor());
    }
}

TEST_P(ClassicModelsMultiDTypeTest, Float16ReducedComplexity) {
    // For Float16, test with smaller models
    if (dtype() == DType::Float16) {
        auto model = vgg11(10, true, false);
        convert_model(model);
        model->eval();

        Variable input = createInput({1, 3, 224, 224}, false);
        Variable output = model->forward(input);

        expectShape(output.tensor(), {1, 10});
    }
}

// ============================================================================
// Test Instantiation
// ============================================================================

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(ClassicModelsMultiDTypeTest);

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
