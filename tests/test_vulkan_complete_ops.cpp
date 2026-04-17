/**
 * @file test_vulkan_complete_ops.cpp
 * @brief Comprehensive test suite for all Vulkan backend operations
 */

#include <gtest/gtest.h>
#include "tenzor/core/tensor.hpp"
#include "tenzor/core/device.hpp"
#include "tenzor/backend/backend.hpp"
#include "tenzor/backend/loader.hpp"
#include "tenzor/tenzor.hpp"
#include "tenzor/backend/op_attributes.hpp"
#include <cmath>
#include <algorithm>

using namespace tenzor;

// Global test environment for initialization
class VulkanTestEnvironment : public ::testing::Environment {
public:
    void SetUp() override {
        tenzor::initialize();
    }
};

static ::testing::Environment* const vulkan_env =
    ::testing::AddGlobalTestEnvironment(new VulkanTestEnvironment);

class VulkanOpsTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Try to get Vulkan device
        try {
            vulkan_device = Device::vulkan(0);
        } catch (...) {
            GTEST_SKIP() << "Vulkan backend not available";
        }
    }

    Device vulkan_device;
    Device cpu_device = Device::cpu();

    // Helper to compare tensors with tolerance
    bool tensors_close(const Tensor& a, const Tensor& b, float rtol = 1e-4, float atol = 1e-5) {
        auto a_shape = a.shape();
        auto b_shape = b.shape();
        if (a_shape.size() != b_shape.size()) return false;
        for (size_t i = 0; i < a_shape.size(); i++) {
            if (a_shape[i] != b_shape[i]) return false;
        }

        auto a_cpu = a.to(cpu_device);
        auto b_cpu = b.to(cpu_device);

        auto a_data = a_cpu.data<float>();
        auto b_data = b_cpu.data<float>();

        for (size_t i = 0; i < a_cpu.numel(); i++) {
            float diff = std::abs(a_data[i] - b_data[i]);
            float threshold = atol + rtol * std::abs(b_data[i]);
            if (diff > threshold) {
                return false;
            }
        }
        return true;
    }
};

// ============================================================================
// Pooling Operations Tests
// ============================================================================

TEST_F(VulkanOpsTest, MaxPool2dForward) {
    // Create input tensor [batch=1, channels=1, height=4, width=4]
    Tensor input({1, 1, 4, 4}, DType::Float32, cpu_device);
    auto input_data = input.data<float>();
    for (int i = 0; i < 16; i++) {
        input_data[i] = static_cast<float>(i);
    }

    input = input.to(vulkan_device);

    // MaxPool2d with kernel=2, stride=2
    OpAttributes attrs;
    attrs.set(AttrKey::KernelSizeH, 2);
    attrs.set(AttrKey::KernelSizeW, 2);
    attrs.set(AttrKey::StrideH, 2);
    attrs.set(AttrKey::StrideW, 2);
    attrs.set(AttrKey::PaddingH, 0);
    attrs.set(AttrKey::PaddingW, 0);

    auto& registry = backend_registry();
    auto* backend = registry.get_backend("vulkan");
    std::vector<Tensor> inputs_vec = {input};
    auto outputs = backend->dispatch("max_pool2d", inputs_vec, attrs);

    ASSERT_EQ(outputs.size(), 2);  // output and indices
    auto output = outputs[0].to(cpu_device);

    auto output_shape = output.shape();
    std::vector<int64_t> expected_shape = {1, 1, 2, 2};
    ASSERT_EQ(output_shape.size(), expected_shape.size());
    for (size_t i = 0; i < expected_shape.size(); i++) {
        EXPECT_EQ(output_shape[i], expected_shape[i]);
    }

    // Expected: max of each 2x2 block
    auto out_data = output.data<float>();
    EXPECT_FLOAT_EQ(out_data[0], 5.0f);   // max(0,1,4,5)
    EXPECT_FLOAT_EQ(out_data[1], 7.0f);   // max(2,3,6,7)
    EXPECT_FLOAT_EQ(out_data[2], 13.0f);  // max(8,9,12,13)
    EXPECT_FLOAT_EQ(out_data[3], 15.0f);  // max(10,11,14,15)
}

TEST_F(VulkanOpsTest, AvgPool2dForward) {
    Tensor input({1, 1, 4, 4}, DType::Float32, cpu_device);
    auto input_data = input.data<float>();
    for (int i = 0; i < 16; i++) {
        input_data[i] = static_cast<float>(i);
    }

    input = input.to(vulkan_device);

    OpAttributes attrs;
    attrs.set(AttrKey::KernelSizeH, 2);
    attrs.set(AttrKey::KernelSizeW, 2);
    attrs.set(AttrKey::StrideH, 2);
    attrs.set(AttrKey::StrideW, 2);

    auto& registry = backend_registry();
    auto* backend = registry.get_backend("vulkan");
    std::vector<Tensor> inputs_vec = {input};
    auto outputs = backend->dispatch("avg_pool2d", inputs_vec, attrs);

    auto output = outputs[0].to(cpu_device);
    auto output_shape = output.shape();
    std::vector<int64_t> expected_shape = {1, 1, 2, 2};
    ASSERT_EQ(output_shape.size(), expected_shape.size());
    for (size_t i = 0; i < expected_shape.size(); i++) {
        EXPECT_EQ(output_shape[i], expected_shape[i]);
    }

    auto out_data = output.data<float>();
    EXPECT_NEAR(out_data[0], 2.5f, 1e-5);   // avg(0,1,4,5)
    EXPECT_NEAR(out_data[1], 4.5f, 1e-5);   // avg(2,3,6,7)
    EXPECT_NEAR(out_data[2], 10.5f, 1e-5);  // avg(8,9,12,13)
    EXPECT_NEAR(out_data[3], 12.5f, 1e-5);  // avg(10,11,14,15)
}

TEST_F(VulkanOpsTest, AdaptiveMaxPool2d) {
    Tensor input({1, 1, 4, 4}, DType::Float32, cpu_device);
    auto input_data = input.data<float>();
    for (int i = 0; i < 16; i++) {
        input_data[i] = static_cast<float>(i);
    }

    input = input.to(vulkan_device);

    OpAttributes attrs;
    attrs.set(AttrKey::OutputSizeH, 2);
    attrs.set(AttrKey::OutputSizeW, 2);

    auto& registry = backend_registry();
    auto* backend = registry.get_backend("vulkan");
    std::vector<Tensor> inputs_vec = {input};
    auto outputs = backend->dispatch("adaptive_max_pool2d", inputs_vec, attrs);

    auto output = outputs[0].to(cpu_device);
    auto output_shape = output.shape();
    std::vector<int64_t> expected_shape = {1, 1, 2, 2};
    ASSERT_EQ(output_shape.size(), expected_shape.size());
    for (size_t i = 0; i < expected_shape.size(); i++) {
        EXPECT_EQ(output_shape[i], expected_shape[i]);
    }
}

TEST_F(VulkanOpsTest, AdaptiveAvgPool2d) {
    Tensor input({1, 1, 4, 4}, DType::Float32, cpu_device);
    auto input_data = input.data<float>();
    for (int i = 0; i < 16; i++) {
        input_data[i] = static_cast<float>(i);
    }

    input = input.to(vulkan_device);

    OpAttributes attrs;
    attrs.set(AttrKey::OutputSizeH, 2);
    attrs.set(AttrKey::OutputSizeW, 2);

    auto& registry = backend_registry();
    auto* backend = registry.get_backend("vulkan");
    std::vector<Tensor> inputs_vec = {input};
    auto outputs = backend->dispatch("adaptive_avg_pool2d", inputs_vec, attrs);

    auto output = outputs[0].to(cpu_device);
    auto output_shape = output.shape();
    std::vector<int64_t> expected_shape = {1, 1, 2, 2};
    ASSERT_EQ(output_shape.size(), expected_shape.size());
    for (size_t i = 0; i < expected_shape.size(); i++) {
        EXPECT_EQ(output_shape[i], expected_shape[i]);
    }
}

// ============================================================================
// Normalization Tests
// ============================================================================

TEST_F(VulkanOpsTest, Softmax) {
    // Create input tensor [2, 4] (batch_size=2, num_classes=4)
    Tensor input({2, 4}, DType::Float32, cpu_device);
    auto input_data = input.data<float>();
    input_data[0] = 1.0f; input_data[1] = 2.0f; input_data[2] = 3.0f; input_data[3] = 4.0f;
    input_data[4] = 1.0f; input_data[5] = 2.0f; input_data[6] = 3.0f; input_data[7] = 4.0f;

    input = input.to(vulkan_device);

    OpAttributes attrs;
    attrs.set(AttrKey::Dim, -1);

    auto& registry = backend_registry();
    auto* backend = registry.get_backend("vulkan");
    std::vector<Tensor> inputs_vec = {input};
    auto outputs = backend->dispatch("softmax", inputs_vec, attrs);

    auto output = outputs[0].to(cpu_device);
    auto out_data = output.data<float>();

    // Check that each row sums to 1
    float row0_sum = out_data[0] + out_data[1] + out_data[2] + out_data[3];
    float row1_sum = out_data[4] + out_data[5] + out_data[6] + out_data[7];

    EXPECT_NEAR(row0_sum, 1.0f, 1e-5);
    EXPECT_NEAR(row1_sum, 1.0f, 1e-5);

    // Check that outputs are in [0, 1]
    for (int i = 0; i < 8; i++) {
        EXPECT_GE(out_data[i], 0.0f);
        EXPECT_LE(out_data[i], 1.0f);
    }
}

TEST_F(VulkanOpsTest, LogSoftmax) {
    Tensor input({2, 4}, DType::Float32, cpu_device);
    auto input_data = input.data<float>();
    for (int i = 0; i < 8; i++) {
        input_data[i] = static_cast<float>(i % 4);
    }

    input = input.to(vulkan_device);

    OpAttributes attrs;
    attrs.set(AttrKey::Dim, -1);

    auto& registry = backend_registry();
    auto* backend = registry.get_backend("vulkan");
    std::vector<Tensor> inputs_vec = {input};
    auto outputs = backend->dispatch("log_softmax", inputs_vec, attrs);

    auto output = outputs[0].to(cpu_device);
    auto out_data = output.data<float>();

    // Check that log_softmax outputs are negative (since probabilities are < 1)
    for (int i = 0; i < 8; i++) {
        EXPECT_LE(out_data[i], 0.0f);
    }
}

// ============================================================================
// Advanced Reduction Tests
// ============================================================================

TEST_F(VulkanOpsTest, Argmax) {
    Tensor input({2, 4}, DType::Float32, cpu_device);
    auto input_data = input.data<float>();
    input_data[0] = 1.0f; input_data[1] = 4.0f; input_data[2] = 2.0f; input_data[3] = 3.0f;
    input_data[4] = 3.0f; input_data[5] = 1.0f; input_data[6] = 4.0f; input_data[7] = 2.0f;

    input = input.to(vulkan_device);

    OpAttributes attrs;
    attrs.set(AttrKey::Dim, 1);
    attrs.set(AttrKey::Keepdim, false);

    auto& registry = backend_registry();
    auto* backend = registry.get_backend("vulkan");
    std::vector<Tensor> inputs_vec = {input};
    auto outputs = backend->dispatch("argmax", inputs_vec, attrs);

    auto output = outputs[0].to(cpu_device);
    auto out_data = output.data<int32_t>();

    EXPECT_EQ(out_data[0], 1);  // index of max in first row
    EXPECT_EQ(out_data[1], 2);  // index of max in second row
}

TEST_F(VulkanOpsTest, Argmin) {
    Tensor input({2, 4}, DType::Float32, cpu_device);
    auto input_data = input.data<float>();
    input_data[0] = 4.0f; input_data[1] = 1.0f; input_data[2] = 3.0f; input_data[3] = 2.0f;
    input_data[4] = 2.0f; input_data[5] = 4.0f; input_data[6] = 1.0f; input_data[7] = 3.0f;

    input = input.to(vulkan_device);

    OpAttributes attrs;
    attrs.set(AttrKey::Dim, 1);
    attrs.set(AttrKey::Keepdim, false);

    auto& registry = backend_registry();
    auto* backend = registry.get_backend("vulkan");
    std::vector<Tensor> inputs_vec = {input};
    auto outputs = backend->dispatch("argmin", inputs_vec, attrs);

    auto output = outputs[0].to(cpu_device);
    auto out_data = output.data<int32_t>();

    EXPECT_EQ(out_data[0], 1);  // index of min in first row
    EXPECT_EQ(out_data[1], 2);  // index of min in second row
}

TEST_F(VulkanOpsTest, Variance) {
    Tensor input({1, 4}, DType::Float32, cpu_device);
    auto input_data = input.data<float>();
    input_data[0] = 1.0f; input_data[1] = 2.0f; input_data[2] = 3.0f; input_data[3] = 4.0f;

    input = input.to(vulkan_device);

    OpAttributes attrs;
    attrs.set(AttrKey::Dim, 1);
    attrs.set(AttrKey::Unbiased, false);
    attrs.set(AttrKey::Keepdim, false);

    auto& registry = backend_registry();
    auto* backend = registry.get_backend("vulkan");
    std::vector<Tensor> inputs_vec = {input};
    auto outputs = backend->dispatch("var", inputs_vec, attrs);

    auto output = outputs[0].to(cpu_device);
    auto out_data = output.data<float>();

    // Variance of [1,2,3,4] with mean=2.5 is 1.25
    EXPECT_NEAR(out_data[0], 1.25f, 1e-4);
}

TEST_F(VulkanOpsTest, Std) {
    Tensor input({1, 4}, DType::Float32, cpu_device);
    auto input_data = input.data<float>();
    input_data[0] = 1.0f; input_data[1] = 2.0f; input_data[2] = 3.0f; input_data[3] = 4.0f;

    input = input.to(vulkan_device);

    OpAttributes attrs;
    attrs.set(AttrKey::Dim, 1);
    attrs.set(AttrKey::Unbiased, false);
    attrs.set(AttrKey::Keepdim, false);

    auto& registry = backend_registry();
    auto* backend = registry.get_backend("vulkan");
    std::vector<Tensor> inputs_vec = {input};
    auto outputs = backend->dispatch("std", inputs_vec, attrs);

    auto output = outputs[0].to(cpu_device);
    auto out_data = output.data<float>();

    // Std of [1,2,3,4] is sqrt(1.25) ≈ 1.118
    EXPECT_NEAR(out_data[0], std::sqrt(1.25f), 1e-4);
}

TEST_F(VulkanOpsTest, Prod) {
    Tensor input({1, 4}, DType::Float32, cpu_device);
    auto input_data = input.data<float>();
    input_data[0] = 1.0f; input_data[1] = 2.0f; input_data[2] = 3.0f; input_data[3] = 4.0f;

    input = input.to(vulkan_device);

    OpAttributes attrs;
    attrs.set(AttrKey::Dim, 1);
    attrs.set(AttrKey::Keepdim, false);

    auto& registry = backend_registry();
    auto* backend = registry.get_backend("vulkan");
    std::vector<Tensor> inputs_vec = {input};
    auto outputs = backend->dispatch("prod", inputs_vec, attrs);

    auto output = outputs[0].to(cpu_device);
    auto out_data = output.data<float>();

    // Product of [1,2,3,4] is 24
    EXPECT_NEAR(out_data[0], 24.0f, 1e-4);
}

// ============================================================================
// Indexing Operations Tests
// ============================================================================

TEST_F(VulkanOpsTest, Embedding) {
    // Create embedding table [vocab_size=10, embedding_dim=4]
    Tensor weight({10, 4}, DType::Float32, cpu_device);
    auto weight_data = weight.data<float>();
    for (int i = 0; i < 40; i++) {
        weight_data[i] = static_cast<float>(i);
    }

    // Create indices [2, 3] (batch=2, sequence=3)
    Tensor indices({2, 3}, DType::Int32, cpu_device);
    auto indices_data = indices.data<int32_t>();
    indices_data[0] = 0; indices_data[1] = 1; indices_data[2] = 2;
    indices_data[3] = 3; indices_data[4] = 4; indices_data[5] = 5;

    weight = weight.to(vulkan_device);
    indices = indices.to(vulkan_device);

    OpAttributes attrs;
    attrs.set(AttrKey::PaddingIdx, static_cast<int64_t>(-1));

    auto& registry = backend_registry();
    auto* backend = registry.get_backend("vulkan");
    std::vector<Tensor> inputs_vec = {weight, indices};
    auto outputs = backend->dispatch("embedding", inputs_vec, attrs);

    auto output = outputs[0].to(cpu_device);

    // Output shape should be [2, 3, 4]
    auto output_shape = output.shape();
    std::vector<int64_t> expected_shape = {2, 3, 4};
    ASSERT_EQ(output_shape.size(), expected_shape.size());
    for (size_t i = 0; i < expected_shape.size(); i++) {
        EXPECT_EQ(output_shape[i], expected_shape[i]);
    }
}

TEST_F(VulkanOpsTest, Gather) {
    // Create input [2, 4]
    Tensor input({2, 4}, DType::Float32, cpu_device);
    auto input_data = input.data<float>();
    for (int i = 0; i < 8; i++) {
        input_data[i] = static_cast<float>(i);
    }

    // Gather indices [2, 2]
    Tensor indices({2, 2}, DType::Int32, cpu_device);
    auto indices_data = indices.data<int32_t>();
    indices_data[0] = 0; indices_data[1] = 2;
    indices_data[2] = 1; indices_data[3] = 3;

    input = input.to(vulkan_device);
    indices = indices.to(vulkan_device);

    OpAttributes attrs;
    attrs.set(AttrKey::Dim, static_cast<int64_t>(1));

    auto& registry = backend_registry();
    auto* backend = registry.get_backend("vulkan");
    std::vector<Tensor> inputs_vec = {input, indices};
    auto outputs = backend->dispatch("gather", inputs_vec, attrs);

    auto output = outputs[0].to(cpu_device);
    auto output_shape = output.shape();
    std::vector<int64_t> expected_shape = {2, 2};
    ASSERT_EQ(output_shape.size(), expected_shape.size());
    for (size_t i = 0; i < expected_shape.size(); i++) {
        EXPECT_EQ(output_shape[i], expected_shape[i]);
    }
}

TEST_F(VulkanOpsTest, IndexSelect) {
    // Create input [3, 4]
    Tensor input({3, 4}, DType::Float32, cpu_device);
    auto input_data = input.data<float>();
    for (int i = 0; i < 12; i++) {
        input_data[i] = static_cast<float>(i);
    }

    // Select indices [2] (select rows 0 and 2)
    Tensor indices({2}, DType::Int32, cpu_device);
    auto indices_data = indices.data<int32_t>();
    indices_data[0] = 0;
    indices_data[1] = 2;

    input = input.to(vulkan_device);
    indices = indices.to(vulkan_device);

    OpAttributes attrs;
    attrs.set(AttrKey::Dim, 0);

    auto& registry = backend_registry();
    auto* backend = registry.get_backend("vulkan");
    std::vector<Tensor> inputs_vec = {input, indices};
    auto outputs = backend->dispatch("index_select", inputs_vec, attrs);

    auto output = outputs[0].to(cpu_device);

    // Output shape should be [2, 4]
    auto output_shape = output.shape();
    std::vector<int64_t> expected_shape = {2, 4};
    ASSERT_EQ(output_shape.size(), expected_shape.size());
    for (size_t i = 0; i < expected_shape.size(); i++) {
        EXPECT_EQ(output_shape[i], expected_shape[i]);
    }

    auto out_data = output.template data<float>();
    // First selected row (row 0)
    EXPECT_FLOAT_EQ(out_data[0], 0.0f);
    EXPECT_FLOAT_EQ(out_data[1], 1.0f);
    EXPECT_FLOAT_EQ(out_data[2], 2.0f);
    EXPECT_FLOAT_EQ(out_data[3], 3.0f);
    // Second selected row (row 2)
    EXPECT_FLOAT_EQ(out_data[4], 8.0f);
    EXPECT_FLOAT_EQ(out_data[5], 9.0f);
    EXPECT_FLOAT_EQ(out_data[6], 10.0f);
    EXPECT_FLOAT_EQ(out_data[7], 11.0f);
}

// ============================================================================
// Performance Benchmarks
// ============================================================================

TEST_F(VulkanOpsTest, BenchmarkLargeConv2d) {
    // Large convolution for performance testing
    Tensor input({32, 64, 128, 128}, DType::Float32, vulkan_device);
    Tensor weight({128, 64, 3, 3}, DType::Float32, vulkan_device);

    OpAttributes attrs;
    attrs.set(AttrKey::Stride, 1);
    attrs.set(AttrKey::Padding, 1);

    auto& registry = backend_registry();
    auto* backend = registry.get_backend("vulkan");

    auto start = std::chrono::high_resolution_clock::now();
    std::vector<Tensor> inputs_vec = {input, weight};
    auto outputs = backend->dispatch("conv2d", inputs_vec, attrs);
    backend->synchronize(vulkan_device.index);
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Conv2d time: " << duration.count() << "ms\n";
}

TEST_F(VulkanOpsTest, BenchmarkLargeMatmul) {
    // Large matrix multiplication
    Tensor a({1024, 1024}, DType::Float32, vulkan_device);
    Tensor b({1024, 1024}, DType::Float32, vulkan_device);

    auto& registry = backend_registry();
    auto* backend = registry.get_backend("vulkan");

    auto start = std::chrono::high_resolution_clock::now();
    std::vector<Tensor> inputs_vec = {a, b};
    OpAttributes empty_attrs;
    auto outputs = backend->dispatch("matmul", inputs_vec, empty_attrs);
    backend->synchronize(vulkan_device.index);
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Matmul time: " << duration.count() << "ms\n";
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
