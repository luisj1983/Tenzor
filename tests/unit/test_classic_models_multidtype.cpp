/**
 * @file test_classic_models_multidtype.cpp
 * @brief Multi-dtype tests for VGG, AlexNet, and GoogLeNet classic models
 *
 * Tests classic CNN architectures with Float32, Float64, and Float16 data types across
 * CPU, CUDA, Vulkan, and OneAPI backends to ensure:
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
#include "../../include/tenzor/core/tensor.hpp"
#include "../../include/tenzor/autograd/variable.hpp"
#include <cmath>
#include <memory>

using namespace tenzor;
using namespace tenzor::models;

// ============================================================================
// Multi-DType Test Parameter Structure
// ============================================================================

struct BackendDTypeParam {
    std::string backend_name;
    DType dtype;
    std::string dtype_name;

    std::string ToString() const {
        return backend_name + "_" + dtype_name;
    }
};

// ============================================================================
// Multi-DType Test Fixture
// ============================================================================

class ClassicModelsMultiDTypeTest : public ::testing::TestWithParam<BackendDTypeParam> {
protected:
    Device device_;
    DType dtype_;
    float abs_tol_;
    float rel_tol_;
    float param_count_tol_;

    void SetUp() override {
        tenzor::initialize();

        auto param = GetParam();
        dtype_ = param.dtype;

        // Set up device based on backend
        if (param.backend_name == "cpu") {
            device_ = Device::cpu();
        }
        else if (param.backend_name == "cuda") {
            if (!isBackendAvailable(Device::Type::CUDA)) {
                GTEST_SKIP() << "CUDA not available";
            }
            device_ = Device::cuda(0);
        }
        else if (param.backend_name == "vulkan") {
            if (!isBackendAvailable(Device::Type::Vulkan)) {
                GTEST_SKIP() << "Vulkan not available";
            }
            device_ = Device::vulkan(0);
        }
        else if (param.backend_name == "oneapi") {
            if (!isBackendAvailable(Device::Type::OneAPI)) {
                GTEST_SKIP() << "OneAPI not available";
            }
            device_ = Device::oneapi(0);
        }

        // Set tolerance based on data type
        if (dtype_ == DType::Float16) {
            abs_tol_ = 1e-2f;
            rel_tol_ = 1e-2f;
            param_count_tol_ = 0.05f;  // 5% tolerance
        } else if (dtype_ == DType::Float64) {
            abs_tol_ = 1e-10f;
            rel_tol_ = 1e-10f;
            param_count_tol_ = 0.01f;
        } else {  // Float32
            abs_tol_ = 1e-5f;
            rel_tol_ = 1e-5f;
            param_count_tol_ = 0.01f;
        }
    }

    static bool isBackendAvailable(Device::Type type) {
        try {
            Device test_device{type, 0};
            Tensor test_tensor({2, 2}, DType::Float32, test_device);
            return true;
        } catch (...) {
            return false;
        }
    }

    Variable createInput(const std::vector<int64_t>& shape, bool requires_grad = true) {
        Tensor tensor(shape, dtype_, device_);
        tensor.fill_(0.5f);  // Initialize with non-zero values
        return Variable(tensor, requires_grad);
    }

    bool CheckShape(const Variable& var, const std::vector<int64_t>& expected_shape) {
        auto shape = var.tensor().shape();
        return std::vector<int64_t>(shape.begin(), shape.end()) == expected_shape;
    }

    bool CheckParameterCount(int64_t actual, int64_t expected) {
        int64_t tolerance = static_cast<int64_t>(expected * param_count_tol_);
        return std::abs(actual - expected) <= tolerance;
    }

    size_t CountParameters(const std::vector<std::shared_ptr<Variable>>& params) {
        size_t total = 0;
        for (const auto& p : params) {
            size_t param_size = 1;
            for (auto dim : p->tensor().shape()) {
                param_size *= dim;
            }
            total += param_size;
        }
        return total;
    }
};

// ============================================================================
// VGG Tests
// ============================================================================

TEST_P(ClassicModelsMultiDTypeTest, VGG11ForwardShape) {
    auto model = vgg11(10, true, false);

    // Convert model to test dtype
    model->to(dtype_);

    model->eval();

    Variable input = createInput({2, 3, 224, 224}, false);
    Variable output = model->forward(input);

    EXPECT_TRUE(CheckShape(output, {2, 10}))
        << "VGG-11 should output (batch, 10)";
    EXPECT_EQ(output.tensor().dtype(), dtype_)
        << "Output dtype should match input";
}

TEST_P(ClassicModelsMultiDTypeTest, VGG11GradientFlow) {
    auto model = vgg11(10, true, false);

    // Convert model to test dtype
    model->to(dtype_);

    model->train();

    Variable input = createInput({1, 3, 224, 224}, true);
    Variable output = model->forward(input);
    Variable loss = tenzor::sum(output);
    loss.backward();

    EXPECT_TRUE(input.grad().has_value())
        << "Gradients should flow through VGG-11 to input";
    EXPECT_EQ(input.grad()->dtype(), dtype_);

    auto params = model->parameters();
    EXPECT_GT(params.size(), 0);
}

TEST_P(ClassicModelsMultiDTypeTest, VGG13ForwardShape) {
    auto model = vgg13(10, true, false);
    model->eval();

    Variable input = createInput({2, 3, 224, 224}, false);
    Variable output = model->forward(input);

    EXPECT_TRUE(CheckShape(output, {2, 10}))
        << "VGG-13 should output (batch, 10)";
    EXPECT_EQ(output.tensor().dtype(), dtype_);
}

TEST_P(ClassicModelsMultiDTypeTest, VGG16ForwardShape) {
    auto model = vgg16(1000, true, false);
    model->eval();

    Variable input = createInput({1, 3, 224, 224}, false);
    Variable output = model->forward(input);

    EXPECT_TRUE(CheckShape(output, {1, 1000}))
        << "VGG-16 should output (batch, 1000)";
    EXPECT_EQ(output.tensor().dtype(), dtype_);
}

TEST_P(ClassicModelsMultiDTypeTest, VGG16GradientFlow) {
    auto model = vgg16(10, true, false);
    model->train();

    Variable input = createInput({1, 3, 224, 224}, true);
    Variable output = model->forward(input);
    Variable loss = tenzor::sum(output);
    loss.backward();

    EXPECT_TRUE(input.grad().has_value())
        << "Gradients should flow through deep VGG-16";
}

TEST_P(ClassicModelsMultiDTypeTest, VGG19ForwardShape) {
    auto model = vgg19(100, true, false);
    model->eval();

    Variable input = createInput({4, 3, 224, 224}, false);
    Variable output = model->forward(input);

    EXPECT_TRUE(CheckShape(output, {4, 100}))
        << "VGG-19 should output (batch, 100)";
    EXPECT_EQ(output.tensor().dtype(), dtype_);
}

TEST_P(ClassicModelsMultiDTypeTest, VGGWithoutBatchNorm) {
    auto model = vgg11(10, false, false);
    model->eval();

    Variable input = createInput({1, 3, 224, 224}, false);
    Variable output = model->forward(input);

    EXPECT_TRUE(CheckShape(output, {1, 10}))
        << "VGG without BatchNorm should work";
    EXPECT_EQ(output.tensor().dtype(), dtype_);
}

TEST_P(ClassicModelsMultiDTypeTest, VGGCustomDropout) {
    auto model = std::make_shared<VGG>(VGGConfig::vgg11(), 10, true, 0.3);
    model->eval();

    Variable input = createInput({1, 3, 224, 224}, false);
    Variable output = model->forward(input);

    EXPECT_TRUE(CheckShape(output, {1, 10}))
        << "VGG with custom dropout should work";
}

// ============================================================================
// AlexNet Tests
// ============================================================================

TEST_P(ClassicModelsMultiDTypeTest, AlexNetForwardShape) {
    auto model = alexnet(1000, false);

    // Convert model to test dtype
    model->to(dtype_);

    model->eval();

    Variable input = createInput({2, 3, 224, 224}, false);
    Variable output = model->forward(input);

    EXPECT_TRUE(CheckShape(output, {2, 1000}))
        << "AlexNet should output (batch, 1000)";
    EXPECT_EQ(output.tensor().dtype(), dtype_);
}

TEST_P(ClassicModelsMultiDTypeTest, AlexNetGradientFlow) {
    auto model = alexnet(10, false);

    // Convert model to test dtype
    model->to(dtype_);

    model->train();

    Variable input = createInput({1, 3, 224, 224}, true);
    Variable output = model->forward(input);
    Variable loss = tenzor::sum(output);
    loss.backward();

    EXPECT_TRUE(input.grad().has_value())
        << "Gradients should flow through AlexNet";
    EXPECT_EQ(input.grad()->dtype(), dtype_);
}

TEST_P(ClassicModelsMultiDTypeTest, AlexNetCustomClasses) {
    auto model = alexnet(10, false);

    // Convert model to test dtype
    model->to(dtype_);

    model->eval();

    Variable input = createInput({1, 3, 224, 224}, false);
    Variable output = model->forward(input);

    EXPECT_TRUE(CheckShape(output, {1, 10}))
        << "AlexNet should support custom classes";
}

TEST_P(ClassicModelsMultiDTypeTest, AlexNetBatchProcessing) {
    auto model = alexnet(100, false);

    // Convert model to test dtype
    model->to(dtype_);

    model->eval();

    Variable input = createInput({8, 3, 224, 224}, false);
    Variable output = model->forward(input);

    EXPECT_TRUE(CheckShape(output, {8, 100}))
        << "AlexNet should handle larger batches";
    EXPECT_EQ(output.tensor().dtype(), dtype_);
}

TEST_P(ClassicModelsMultiDTypeTest, AlexNetCustomDropout) {
    auto model = std::make_shared<AlexNet>(10, 0.3);

    // Convert model to test dtype
    model->to(dtype_);

    model->eval();

    Variable input = createInput({1, 3, 224, 224}, false);
    Variable output = model->forward(input);

    EXPECT_TRUE(CheckShape(output, {1, 10}))
        << "AlexNet with custom dropout should work";
}

// ============================================================================
// GoogLeNet Tests
// ============================================================================

TEST_P(ClassicModelsMultiDTypeTest, GoogLeNetForwardShape) {
    auto model = googlenet(1000, false, false);  // No aux classifiers
    model->eval();

    Variable input = createInput({1, 3, 224, 224}, false);
    Variable output = model->forward(input);

    EXPECT_TRUE(CheckShape(output, {1, 1000}))
        << "GoogLeNet should output (batch, 1000)";
    EXPECT_EQ(output.tensor().dtype(), dtype_);
}

TEST_P(ClassicModelsMultiDTypeTest, GoogLeNetGradientFlow) {
    auto model = googlenet(10, false, false);
    model->train();

    Variable input = createInput({1, 3, 224, 224}, true);
    Variable output = model->forward(input);
    Variable loss = tenzor::sum(output);
    loss.backward();

    EXPECT_TRUE(input.grad().has_value())
        << "Gradients should flow through GoogLeNet";
    EXPECT_EQ(input.grad()->dtype(), dtype_);
}

TEST_P(ClassicModelsMultiDTypeTest, GoogLeNetWithAuxiliaryClassifiers) {
    auto model = googlenet(10, false, true);  // With aux classifiers
    model->train();

    Variable input = createInput({2, 3, 224, 224}, false);

    // Forward with auxiliary outputs
    auto [main_out, aux1_out, aux2_out] = model->forward_with_aux(input);

    // Check main output
    EXPECT_TRUE(CheckShape(main_out, {2, 10}))
        << "GoogLeNet main output should be (2, 10)";
    EXPECT_EQ(main_out.tensor().dtype(), dtype_);

    // Check auxiliary outputs
    EXPECT_TRUE(CheckShape(aux1_out, {2, 10}))
        << "Auxiliary classifier 1 should output (2, 10)";
    EXPECT_TRUE(CheckShape(aux2_out, {2, 10}))
        << "Auxiliary classifier 2 should output (2, 10)";
    EXPECT_EQ(aux1_out.tensor().dtype(), dtype_);
    EXPECT_EQ(aux2_out.tensor().dtype(), dtype_);
}

TEST_P(ClassicModelsMultiDTypeTest, GoogLeNetInferenceMode) {
    auto model = googlenet(1000, false, true);
    model->eval();

    Variable input = createInput({1, 3, 224, 224}, false);
    Variable output = model->forward(input);

    EXPECT_TRUE(CheckShape(output, {1, 1000}))
        << "GoogLeNet in eval mode should only output main prediction";
}

TEST_P(ClassicModelsMultiDTypeTest, InceptionModuleForward) {
    auto inception = std::make_shared<InceptionModule>(
        192,        // in_channels
        64,         // out_1x1
        96, 128,    // reduce_3x3, out_3x3
        16, 32,     // reduce_5x5, out_5x5
        32          // out_pool_proj
    );

    Variable input = createInput({1, 192, 28, 28}, false);
    Variable output = inception->forward(input);

    // Output channels: 64 + 128 + 32 + 32 = 256
    EXPECT_TRUE(CheckShape(output, {1, 256, 28, 28}))
        << "Inception module should produce correct output shape";
    EXPECT_EQ(output.tensor().dtype(), dtype_);
}

TEST_P(ClassicModelsMultiDTypeTest, GoogLeNetCustomDropout) {
    auto model = std::make_shared<GoogLeNet>(10, false, 0.2);
    model->eval();

    Variable input = createInput({1, 3, 224, 224}, false);
    Variable output = model->forward(input);

    EXPECT_TRUE(CheckShape(output, {1, 10}))
        << "GoogLeNet with custom dropout should work";
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
    size_t vgg_count = CountParameters(vgg_params);
    size_t alex_count = CountParameters(alex_params);

    EXPECT_GT(vgg_count, alex_count)
        << "VGG-16 should have more parameters than AlexNet";
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

    Variable input = createInput({1, 3, 224, 224}, true);
    Variable output = model->forward(input);

    EXPECT_TRUE(output.requires_grad())
        << "Output should track gradients when input does";
}

// ============================================================================
// Edge Cases and Stress Tests
// ============================================================================

TEST_P(ClassicModelsMultiDTypeTest, LargeBatchVGG) {
    auto model = vgg11(10, true, false);
    model->eval();

    Variable input = createInput({16, 3, 224, 224}, false);
    Variable output = model->forward(input);

    EXPECT_TRUE(CheckShape(output, {16, 10}))
        << "VGG should handle large batch sizes";
}

TEST_P(ClassicModelsMultiDTypeTest, SmallBatchGoogLeNet) {
    auto model = googlenet(1000, false, false);
    model->eval();

    Variable input = createInput({1, 3, 224, 224}, false);
    Variable output = model->forward(input);

    EXPECT_TRUE(CheckShape(output, {1, 1000}))
        << "GoogLeNet should handle single sample";
}

TEST_P(ClassicModelsMultiDTypeTest, MultipleBatchSizes) {
    auto model = alexnet(10, false);
    model->eval();

    for (int batch_size : {1, 2, 4, 8}) {
        Variable input = createInput({batch_size, 3, 224, 224}, false);
        Variable output = model->forward(input);

        EXPECT_TRUE(CheckShape(output, {batch_size, 10}))
            << "AlexNet should handle batch_size=" << batch_size;
    }
}

TEST_P(ClassicModelsMultiDTypeTest, Float16ReducedComplexity) {
    // For Float16, test with smaller models
    if (dtype_ == DType::Float16) {
        auto model = vgg11(10, true, false);
        model->eval();

        Variable input = createInput({1, 3, 224, 224}, false);
        Variable output = model->forward(input);

        EXPECT_TRUE(CheckShape(output, {1, 10}))
            << "Classic models should work with Float16";
    }
}

// ============================================================================
// Parameter Instantiation
// ============================================================================

std::vector<BackendDTypeParam> GenerateParams() {
    std::vector<BackendDTypeParam> params;

    std::vector<std::string> backends = {"cpu", "cuda", "vulkan", "oneapi"};
    std::vector<std::pair<DType, std::string>> dtypes = {
        {DType::Float32, "float32"},
        {DType::Float64, "float64"},
        {DType::Float16, "float16"}
    };

    for (const auto& backend : backends) {
        for (const auto& [dtype, dtype_name] : dtypes) {
            params.push_back({backend, dtype, dtype_name});
        }
    }

    return params;
}

INSTANTIATE_TEST_SUITE_P(
    ClassicModelsMultiDType,
    ClassicModelsMultiDTypeTest,
    ::testing::ValuesIn(GenerateParams()),
    [](const ::testing::TestParamInfo<BackendDTypeParam>& info) {
        return info.param.ToString();
    }
);

int main(int argc, char** argv) {
    tenzor::initialize();
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
