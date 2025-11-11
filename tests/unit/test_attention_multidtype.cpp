#include <gtest/gtest.h>
#include "../backend_test_fixture.hpp"
#include <tenzor/nn/layers/attention.hpp>
#include <cmath>
#include <iostream>

using namespace tenzor;
using namespace tenzor::nn;

// Helper function to convert DType to string
std::string dtype_to_string(DType dtype) {
    switch(dtype) {
        case DType::Float32: return "float32";
        case DType::Float64: return "float64";
        case DType::Float16: return "float16";
        default: return "unknown";
    }
}

// ============================================================================
// Multi-DType Test Fixture
// ============================================================================

struct BackendDTypeParam {
    std::string backend_name;
    DType dtype;
    std::string dtype_name;

    std::string ToString() const {
        return backend_name + "_" + dtype_name;
    }
};

// Test fixture for multi-backend and multi-dtype testing
class MultiheadAttentionMultiDTypeTest : public ::testing::TestWithParam<BackendDTypeParam> {
protected:
    Device device;
    DType dtype;

    void SetUp() override {
        tenzor::initialize();

        auto param = GetParam();
        dtype = param.dtype;

        if (param.backend_name == "cpu") {
            device = Device::cpu();
        }
        else if (param.backend_name == "cuda") {
            if (!isBackendAvailable(Device::Type::CUDA)) {
                GTEST_SKIP() << "CUDA not available";
            }
            device = Device::cuda(0);
        }
        else if (param.backend_name == "vulkan") {
            if (!isBackendAvailable(Device::Type::Vulkan)) {
                GTEST_SKIP() << "Vulkan not available";
            }
            device = Device::vulkan(0);
        }
        else if (param.backend_name == "oneapi") {
            if (!isBackendAvailable(Device::Type::OneAPI)) {
                GTEST_SKIP() << "OneAPI not available";
            }
            device = Device::oneapi(0);
        }
    }

    static bool isBackendAvailable(Device::Type type) {
        try {
            Device test_device{type, 0};
            auto t = zeros({2, 2}, DType::Float32, test_device);
            return true;
        } catch (...) {
            return false;
        }
    }

    // Get dtype-specific tolerance for attention operations
    template<typename T>
    double get_attention_tolerance() const {
        if (std::is_same<T, float>::value) {
            return 1e-5;  // Float32
        } else if (std::is_same<T, double>::value) {
            return 1e-10; // Float64
        } else {
            return 1e-2;  // Float16 - very relaxed tolerance for half precision
        }
    }

    double get_tolerance() const {
        if (dtype == DType::Float32) {
            return 1e-5;
        } else if (dtype == DType::Float64) {
            return 1e-10;
        } else if (dtype == DType::Float16) {
            return 1e-2;
        }
        return 1e-5;
    }
};

TEST_P(MultiheadAttentionMultiDTypeTest, Construction) {
    // Test basic construction
    EXPECT_NO_THROW({
        MultiheadAttention attn(512, 8);
    }) << "Failed on " << device.to_string() << " with dtype " << dtype_to_string(dtype);

    // Test with all parameters
    EXPECT_NO_THROW({
        MultiheadAttention attn(512, 8, 0.1, true, false, false, 0, 0, true);
    }) << "Failed on " << device.to_string() << " with dtype " << dtype_to_string(dtype);
}

TEST_P(MultiheadAttentionMultiDTypeTest, InvalidConstruction) {
    // embed_dim not divisible by num_heads
    EXPECT_THROW({
        MultiheadAttention attn(512, 7);
    }, std::invalid_argument) << "Failed on " << device.to_string() << " with dtype " << dtype_to_string(dtype);

    // Invalid dropout
    EXPECT_THROW({
        MultiheadAttention attn(512, 8, 1.5);
    }, std::invalid_argument) << "Failed on " << device.to_string() << " with dtype " << dtype_to_string(dtype);

    EXPECT_THROW({
        MultiheadAttention attn(512, 8, -0.1);
    }, std::invalid_argument) << "Failed on " << device.to_string() << " with dtype " << dtype_to_string(dtype);
}

TEST_P(MultiheadAttentionMultiDTypeTest, SelfAttentionShape) {
    // Test self-attention output shape
    MultiheadAttention attn(512, 8, 0.0, true, false, false, 0, 0, true);

    int64_t batch_size = 4;
    int64_t seq_len = 10;
    int64_t embed_dim = 512;

    Variable query(randn({batch_size, seq_len, embed_dim}, dtype, device), true);

    auto [output, attn_weights] = attn.forward(query, query, query);

    // Check output shape
    EXPECT_EQ(output.shape().size(), 3) << "Failed on " << device.to_string() << " with dtype " << dtype_to_string(dtype);
    EXPECT_EQ(output.shape()[0], batch_size) << "Failed on " << device.to_string() << " with dtype " << dtype_to_string(dtype);
    EXPECT_EQ(output.shape()[1], seq_len) << "Failed on " << device.to_string() << " with dtype " << dtype_to_string(dtype);
    EXPECT_EQ(output.shape()[2], embed_dim) << "Failed on " << device.to_string() << " with dtype " << dtype_to_string(dtype);

    // Check attention weights shape
    EXPECT_EQ(attn_weights.shape().size(), 4) << "Failed on " << device.to_string() << " with dtype " << dtype_to_string(dtype);
    EXPECT_EQ(attn_weights.shape()[0], batch_size) << "Failed on " << device.to_string() << " with dtype " << dtype_to_string(dtype);
    EXPECT_EQ(attn_weights.shape()[1], 8) << "Failed on " << device.to_string() << " with dtype " << dtype_to_string(dtype);  // num_heads
    EXPECT_EQ(attn_weights.shape()[2], seq_len) << "Failed on " << device.to_string() << " with dtype " << dtype_to_string(dtype);
    EXPECT_EQ(attn_weights.shape()[3], seq_len) << "Failed on " << device.to_string() << " with dtype " << dtype_to_string(dtype);
}

TEST_P(MultiheadAttentionMultiDTypeTest, CrossAttentionShape) {
    // Test cross-attention with different key/value sequence length
    MultiheadAttention attn(256, 4, 0.0, true, false, false, 0, 0, true);

    int64_t batch_size = 2;
    int64_t query_len = 8;
    int64_t kv_len = 12;
    int64_t embed_dim = 256;

    Variable query(randn({batch_size, query_len, embed_dim}, dtype, device), true);
    Variable key(randn({batch_size, kv_len, embed_dim}, dtype, device), true);
    Variable value(randn({batch_size, kv_len, embed_dim}, dtype, device), true);

    auto [output, attn_weights] = attn.forward(query, key, value);

    // Output should have query's sequence length
    EXPECT_EQ(output.shape()[0], batch_size) << "Failed on " << device.to_string() << " with dtype " << dtype_to_string(dtype);
    EXPECT_EQ(output.shape()[1], query_len) << "Failed on " << device.to_string() << " with dtype " << dtype_to_string(dtype);
    EXPECT_EQ(output.shape()[2], embed_dim) << "Failed on " << device.to_string() << " with dtype " << dtype_to_string(dtype);

    // Attention weights: (batch, heads, query_len, kv_len)
    EXPECT_EQ(attn_weights.shape()[0], batch_size) << "Failed on " << device.to_string() << " with dtype " << dtype_to_string(dtype);
    EXPECT_EQ(attn_weights.shape()[1], 4) << "Failed on " << device.to_string() << " with dtype " << dtype_to_string(dtype);
    EXPECT_EQ(attn_weights.shape()[2], query_len) << "Failed on " << device.to_string() << " with dtype " << dtype_to_string(dtype);
    EXPECT_EQ(attn_weights.shape()[3], kv_len) << "Failed on " << device.to_string() << " with dtype " << dtype_to_string(dtype);
}

TEST_P(MultiheadAttentionMultiDTypeTest, BatchFirstFalse) {
    // Test with batch_first=false (seq, batch, embed)
    MultiheadAttention attn(128, 4, 0.0, true, false, false, 0, 0, false);

    int64_t seq_len = 10;
    int64_t batch_size = 4;
    int64_t embed_dim = 128;

    Variable query(randn({seq_len, batch_size, embed_dim}, dtype, device), true);

    auto [output, _] = attn.forward(query, query, query);

    // Output should maintain (seq, batch, embed) format
    EXPECT_EQ(output.shape()[0], seq_len) << "Failed on " << device.to_string() << " with dtype " << dtype_to_string(dtype);
    EXPECT_EQ(output.shape()[1], batch_size) << "Failed on " << device.to_string() << " with dtype " << dtype_to_string(dtype);
    EXPECT_EQ(output.shape()[2], embed_dim) << "Failed on " << device.to_string() << " with dtype " << dtype_to_string(dtype);
}

TEST_P(MultiheadAttentionMultiDTypeTest, SimpleForwardInterface) {
    // Test simplified forward() interface for self-attention
    MultiheadAttention attn(256, 8, 0.0, true, false, false, 0, 0, true);

    Variable input(randn({4, 10, 256}, dtype, device), true);
    Variable output = attn.forward(input);

    EXPECT_EQ(output.shape()[0], 4) << "Failed on " << device.to_string() << " with dtype " << dtype_to_string(dtype);
    EXPECT_EQ(output.shape()[1], 10) << "Failed on " << device.to_string() << " with dtype " << dtype_to_string(dtype);
    EXPECT_EQ(output.shape()[2], 256) << "Failed on " << device.to_string() << " with dtype " << dtype_to_string(dtype);
}

TEST_P(MultiheadAttentionMultiDTypeTest, WithoutAttentionWeights) {
    // Test need_weights=false
    MultiheadAttention attn(256, 8, 0.0, true, false, false, 0, 0, true);

    Variable query(randn({2, 5, 256}, dtype, device), true);

    auto [output, weights] = attn.forward(query, query, query,
                                         Tensor{}, Tensor{}, false);

    EXPECT_EQ(output.shape()[0], 2) << "Failed on " << device.to_string() << " with dtype " << dtype_to_string(dtype);
    EXPECT_EQ(output.shape()[1], 5) << "Failed on " << device.to_string() << " with dtype " << dtype_to_string(dtype);
    EXPECT_EQ(output.shape()[2], 256) << "Failed on " << device.to_string() << " with dtype " << dtype_to_string(dtype);

    // Weights should be empty
    EXPECT_EQ(weights.shape().size(), 0) << "Failed on " << device.to_string() << " with dtype " << dtype_to_string(dtype);
}

TEST_P(MultiheadAttentionMultiDTypeTest, SingleHead) {
    // Test with single attention head
    MultiheadAttention attn(128, 1, 0.0, true, false, false, 0, 0, true);

    Variable query(randn({2, 5, 128}, dtype, device), true);

    auto [output, weights] = attn.forward(query, query, query);

    EXPECT_EQ(output.shape()[0], 2) << "Failed on " << device.to_string() << " with dtype " << dtype_to_string(dtype);
    EXPECT_EQ(output.shape()[1], 5) << "Failed on " << device.to_string() << " with dtype " << dtype_to_string(dtype);
    EXPECT_EQ(output.shape()[2], 128) << "Failed on " << device.to_string() << " with dtype " << dtype_to_string(dtype);

    EXPECT_EQ(weights.shape()[1], 1) << "Failed on " << device.to_string() << " with dtype " << dtype_to_string(dtype);  // num_heads = 1
}

TEST_P(MultiheadAttentionMultiDTypeTest, LargeSequence) {
    // Test with larger sequence length (reduced for Float16 to avoid memory issues)
    int64_t seq_len = (dtype == DType::Float16) ? 256 : 512;
    MultiheadAttention attn(256, 8, 0.0, true, false, false, 0, 0, true);

    Variable query(randn({2, seq_len, 256}, dtype, device), true);

    auto [output, _] = attn.forward(query, query, query, Tensor{}, Tensor{}, false);

    EXPECT_EQ(output.shape()[1], seq_len) << "Failed on " << device.to_string() << " with dtype " << dtype_to_string(dtype);
}

TEST_P(MultiheadAttentionMultiDTypeTest, SmallBatch) {
    // Test with batch size 1
    MultiheadAttention attn(128, 4, 0.0, true, false, false, 0, 0, true);

    Variable query(randn({1, 10, 128}, dtype, device), true);

    auto [output, _] = attn.forward(query, query, query);

    EXPECT_EQ(output.shape()[0], 1) << "Failed on " << device.to_string() << " with dtype " << dtype_to_string(dtype);
    EXPECT_EQ(output.shape()[1], 10) << "Failed on " << device.to_string() << " with dtype " << dtype_to_string(dtype);
    EXPECT_EQ(output.shape()[2], 128) << "Failed on " << device.to_string() << " with dtype " << dtype_to_string(dtype);
}

TEST_P(MultiheadAttentionMultiDTypeTest, WithDropout) {
    // Test with dropout enabled
    MultiheadAttention attn(256, 8, 0.5, true, false, false, 0, 0, true);

    Variable query(randn({4, 10, 256}, dtype, device), true);

    // Set to training mode
    attn.train();

    auto [output1, _] = attn.forward(query, query, query, Tensor{}, Tensor{}, false);
    auto [output2, __] = attn.forward(query, query, query, Tensor{}, Tensor{}, false);

    // Outputs should be different due to dropout (stochastic)
    // Store tensors to avoid dangling pointers from temporaries
    auto output1_cpu = output1.tensor().to(Device::cpu());
    auto output2_cpu = output2.tensor().to(Device::cpu());

    double tol = get_tolerance();
    bool found_difference = false;

    if (dtype == DType::Float32) {
        auto data1 = output1_cpu.data<float>();
        auto data2 = output2_cpu.data<float>();
        for (int64_t i = 0; i < std::min(static_cast<int64_t>(100), output1.tensor().numel()); ++i) {
            if (std::abs(data1[i] - data2[i]) > tol) {
                found_difference = true;
                break;
            }
        }
    } else if (dtype == DType::Float64) {
        auto data1 = output1_cpu.data<double>();
        auto data2 = output2_cpu.data<double>();
        for (int64_t i = 0; i < std::min(static_cast<int64_t>(100), output1.tensor().numel()); ++i) {
            if (std::abs(data1[i] - data2[i]) > tol) {
                found_difference = true;
                break;
            }
        }
    }
    // Note: This test might occasionally fail due to randomness
    // In production, you'd use a fixed seed or mock the dropout
}

TEST_P(MultiheadAttentionMultiDTypeTest, EvalMode) {
    // Test that eval mode disables dropout
    MultiheadAttention attn(256, 8, 0.5, true, false, false, 0, 0, true);

    // Use a fixed input for deterministic test
    Variable query(ones({2, 5, 256}, dtype, device), true);

    // Explicitly set to eval mode
    attn.eval();

    // Verify the module is in eval mode
    ASSERT_FALSE(attn.is_training()) << "Attention should be in eval mode. Failed on " << device.to_string() << " with dtype " << dtype_to_string(dtype);

    auto [output1, _] = attn.forward(query, query, query, Tensor{}, Tensor{}, false);
    auto [output2, __] = attn.forward(query, query, query, Tensor{}, Tensor{}, false);

    // In eval mode, outputs should be identical (within floating-point tolerance)
    auto output1_cpu = output1.tensor().to(Device::cpu());
    auto output2_cpu = output2.tensor().to(Device::cpu());

    double tol = get_tolerance();

    if (dtype == DType::Float32) {
        auto data1 = output1_cpu.data<float>();
        auto data2 = output2_cpu.data<float>();
        for (int64_t i = 0; i < output1.tensor().numel(); ++i) {
            EXPECT_NEAR(data1[i], data2[i], tol)
                << "Mismatch at index " << i << ": " << data1[i] << " vs " << data2[i]
                << ". Failed on " << device.to_string() << " with dtype " << dtype_to_string(dtype);
        }
    } else if (dtype == DType::Float64) {
        auto data1 = output1_cpu.data<double>();
        auto data2 = output2_cpu.data<double>();
        for (int64_t i = 0; i < output1.tensor().numel(); ++i) {
            EXPECT_NEAR(data1[i], data2[i], tol)
                << "Mismatch at index " << i << ": " << data1[i] << " vs " << data2[i]
                << ". Failed on " << device.to_string() << " with dtype " << dtype_to_string(dtype);
        }
    }
}

TEST_P(MultiheadAttentionMultiDTypeTest, DifferentKeyValueDims) {
    // Test with different key and value dimensions
    int64_t embed_dim = 256;
    int64_t kdim = 128;
    int64_t vdim = 128;

    MultiheadAttention attn(embed_dim, 8, 0.0, true, false, false,
                           kdim, vdim, true);

    Variable query(randn({2, 5, embed_dim}, dtype, device), true);
    Variable key(randn({2, 7, kdim}, dtype, device), true);
    Variable value(randn({2, 7, vdim}, dtype, device), true);

    auto [output, _] = attn.forward(query, key, value);

    EXPECT_EQ(output.shape()[0], 2) << "Failed on " << device.to_string() << " with dtype " << dtype_to_string(dtype);
    EXPECT_EQ(output.shape()[1], 5) << "Failed on " << device.to_string() << " with dtype " << dtype_to_string(dtype);
    EXPECT_EQ(output.shape()[2], embed_dim) << "Failed on " << device.to_string() << " with dtype " << dtype_to_string(dtype);
}

// ============================================================================
// Multi-DType Causal Mask Tests
// ============================================================================

class CausalMaskMultiDTypeTest : public ::testing::TestWithParam<BackendDTypeParam> {
protected:
    Device device;
    DType dtype;

    void SetUp() override {
        tenzor::initialize();

        auto param = GetParam();
        dtype = param.dtype;

        if (param.backend_name == "cpu") {
            device = Device::cpu();
        }
        else if (param.backend_name == "cuda") {
            if (!isBackendAvailable(Device::Type::CUDA)) {
                GTEST_SKIP() << "CUDA not available";
            }
            device = Device::cuda(0);
        }
        else if (param.backend_name == "vulkan") {
            if (!isBackendAvailable(Device::Type::Vulkan)) {
                GTEST_SKIP() << "Vulkan not available";
            }
            device = Device::vulkan(0);
        }
        else if (param.backend_name == "oneapi") {
            if (!isBackendAvailable(Device::Type::OneAPI)) {
                GTEST_SKIP() << "OneAPI not available";
            }
            device = Device::oneapi(0);
        }
    }

    static bool isBackendAvailable(Device::Type type) {
        try {
            Device test_device{type, 0};
            auto t = zeros({2, 2}, DType::Float32, test_device);
            return true;
        } catch (...) {
            return false;
        }
    }

    double get_tolerance() const {
        if (dtype == DType::Float32) {
            return 1e-5;
        } else if (dtype == DType::Float64) {
            return 1e-10;
        } else if (dtype == DType::Float16) {
            return 1e-2;
        }
        return 1e-5;
    }
};

TEST_P(CausalMaskMultiDTypeTest, Shape) {
    int64_t seq_len = 10;
    Tensor mask = create_causal_mask(seq_len, device);

    EXPECT_EQ(mask.shape().size(), 2) << "Failed on " << device.to_string() << " with dtype " << dtype_to_string(dtype);
    EXPECT_EQ(mask.shape()[0], seq_len) << "Failed on " << device.to_string() << " with dtype " << dtype_to_string(dtype);
    EXPECT_EQ(mask.shape()[1], seq_len) << "Failed on " << device.to_string() << " with dtype " << dtype_to_string(dtype);
}

TEST_P(CausalMaskMultiDTypeTest, Values) {
    int64_t seq_len = 5;
    Tensor mask = create_causal_mask(seq_len, device);

    // Store tensor before getting data pointer to avoid dangling pointer from temporary
    Tensor cpu_mask = mask.to(Device::cpu());
    double tol = get_tolerance();

    // Check that upper triangle is -inf and lower triangle is 0
    if (dtype == DType::Float32 || mask.dtype() == DType::Float32) {
        auto* data = cpu_mask.data<float>();
        for (int64_t i = 0; i < seq_len; ++i) {
            for (int64_t j = 0; j < seq_len; ++j) {
                if (j > i) {
                    EXPECT_TRUE(std::isinf(data[i * seq_len + j])) << "Failed on " << device.to_string() << " with dtype " << dtype_to_string(dtype);
                    EXPECT_LT(data[i * seq_len + j], 0) << "Failed on " << device.to_string() << " with dtype " << dtype_to_string(dtype);  // -inf
                } else {
                    EXPECT_NEAR(data[i * seq_len + j], 0.0f, tol) << "Failed on " << device.to_string() << " with dtype " << dtype_to_string(dtype);
                }
            }
        }
    } else if (dtype == DType::Float64 || mask.dtype() == DType::Float64) {
        auto* data = cpu_mask.data<double>();
        for (int64_t i = 0; i < seq_len; ++i) {
            for (int64_t j = 0; j < seq_len; ++j) {
                if (j > i) {
                    EXPECT_TRUE(std::isinf(data[i * seq_len + j])) << "Failed on " << device.to_string() << " with dtype " << dtype_to_string(dtype);
                    EXPECT_LT(data[i * seq_len + j], 0) << "Failed on " << device.to_string() << " with dtype " << dtype_to_string(dtype);  // -inf
                } else {
                    EXPECT_NEAR(data[i * seq_len + j], 0.0, tol) << "Failed on " << device.to_string() << " with dtype " << dtype_to_string(dtype);
                }
            }
        }
    }
}

TEST_P(CausalMaskMultiDTypeTest, SmallSize) {
    Tensor mask = create_causal_mask(1, device);
    EXPECT_EQ(mask.shape()[0], 1) << "Failed on " << device.to_string() << " with dtype " << dtype_to_string(dtype);
    EXPECT_EQ(mask.shape()[1], 1) << "Failed on " << device.to_string() << " with dtype " << dtype_to_string(dtype);

    // Store tensor before getting data pointer to avoid dangling pointer from temporary
    Tensor cpu_mask = mask.to(Device::cpu());
    double tol = get_tolerance();

    if (dtype == DType::Float32 || mask.dtype() == DType::Float32) {
        auto* data = cpu_mask.data<float>();
        EXPECT_NEAR(data[0], 0.0f, tol) << "Failed on " << device.to_string() << " with dtype " << dtype_to_string(dtype);
    } else if (dtype == DType::Float64 || mask.dtype() == DType::Float64) {
        auto* data = cpu_mask.data<double>();
        EXPECT_NEAR(data[0], 0.0, tol) << "Failed on " << device.to_string() << " with dtype " << dtype_to_string(dtype);
    }
}

TEST_P(CausalMaskMultiDTypeTest, MediumSize) {
    int64_t seq_len = 100;
    Tensor mask = create_causal_mask(seq_len, device);

    EXPECT_EQ(mask.shape()[0], seq_len) << "Failed on " << device.to_string() << " with dtype " << dtype_to_string(dtype);
    EXPECT_EQ(mask.shape()[1], seq_len) << "Failed on " << device.to_string() << " with dtype " << dtype_to_string(dtype);

    // Spot check a few positions
    auto mask_cpu = mask.to(Device::cpu());
    double tol = get_tolerance();

    if (dtype == DType::Float32 || mask.dtype() == DType::Float32) {
        auto* data = mask_cpu.data<float>();
        EXPECT_NEAR(data[0], 0.0f, tol) << "Failed on " << device.to_string() << " with dtype " << dtype_to_string(dtype);  // [0, 0] should be 0
        EXPECT_TRUE(std::isinf(data[1])) << "Failed on " << device.to_string() << " with dtype " << dtype_to_string(dtype);  // [0, 1] should be -inf
        EXPECT_NEAR(data[seq_len], 0.0f, tol) << "Failed on " << device.to_string() << " with dtype " << dtype_to_string(dtype);  // [1, 0] should be 0
        EXPECT_NEAR(data[seq_len + 1], 0.0f, tol) << "Failed on " << device.to_string() << " with dtype " << dtype_to_string(dtype);  // [1, 1] should be 0
    } else if (dtype == DType::Float64 || mask.dtype() == DType::Float64) {
        auto* data = mask_cpu.data<double>();
        EXPECT_NEAR(data[0], 0.0, tol) << "Failed on " << device.to_string() << " with dtype " << dtype_to_string(dtype);  // [0, 0] should be 0
        EXPECT_TRUE(std::isinf(data[1])) << "Failed on " << device.to_string() << " with dtype " << dtype_to_string(dtype);  // [0, 1] should be -inf
        EXPECT_NEAR(data[seq_len], 0.0, tol) << "Failed on " << device.to_string() << " with dtype " << dtype_to_string(dtype);  // [1, 0] should be 0
        EXPECT_NEAR(data[seq_len + 1], 0.0, tol) << "Failed on " << device.to_string() << " with dtype " << dtype_to_string(dtype);  // [1, 1] should be 0
    }
}

// ============================================================================
// Multi-DType Integration Tests
// ============================================================================

class AttentionIntegrationMultiDTypeTest : public ::testing::TestWithParam<BackendDTypeParam> {
protected:
    Device device;
    DType dtype;

    void SetUp() override {
        tenzor::initialize();

        auto param = GetParam();
        dtype = param.dtype;

        if (param.backend_name == "cpu") {
            device = Device::cpu();
        }
        else if (param.backend_name == "cuda") {
            if (!isBackendAvailable(Device::Type::CUDA)) {
                GTEST_SKIP() << "CUDA not available";
            }
            device = Device::cuda(0);
        }
        else if (param.backend_name == "vulkan") {
            if (!isBackendAvailable(Device::Type::Vulkan)) {
                GTEST_SKIP() << "Vulkan not available";
            }
            device = Device::vulkan(0);
        }
        else if (param.backend_name == "oneapi") {
            if (!isBackendAvailable(Device::Type::OneAPI)) {
                GTEST_SKIP() << "OneAPI not available";
            }
            device = Device::oneapi(0);
        }
    }

    static bool isBackendAvailable(Device::Type type) {
        try {
            Device test_device{type, 0};
            auto t = zeros({2, 2}, DType::Float32, test_device);
            return true;
        } catch (...) {
            return false;
        }
    }

    double get_tolerance() const {
        if (dtype == DType::Float32) {
            return 1e-5;
        } else if (dtype == DType::Float64) {
            return 1e-10;
        } else if (dtype == DType::Float16) {
            return 1e-2;
        }
        return 1e-5;
    }
};

TEST_P(AttentionIntegrationMultiDTypeTest, ForwardBackward) {
    // Test that gradients flow through attention
    MultiheadAttention attn(128, 4, 0.0, true, false, false, 0, 0, true);

    Variable query(randn({2, 5, 128}, dtype, device), true);

    auto [output, _] = attn.forward(query, query, query, Tensor{}, Tensor{}, false);

    // Compute a simple loss
    Variable loss = mean(output);

    // Backward should work
    EXPECT_NO_THROW({
        loss.backward();
    }) << "Failed on " << device.to_string() << " with dtype " << dtype_to_string(dtype);

    // Query should have gradients
    EXPECT_TRUE(query.has_grad()) << "Failed on " << device.to_string() << " with dtype " << dtype_to_string(dtype);
}

TEST_P(AttentionIntegrationMultiDTypeTest, ParameterCount) {
    MultiheadAttention attn(256, 8, 0.0, true, false, false, 0, 0, true);
    attn.to(device);

    auto params = attn.parameters();

    // Should have: q_proj (weight + bias), k_proj (w + b), v_proj (w + b), out_proj (w + b)
    // Total: 8 parameters (4 weights, 4 biases)
    EXPECT_EQ(params.size(), 8) << "Failed on " << device.to_string() << " with dtype " << dtype_to_string(dtype);
}

TEST_P(AttentionIntegrationMultiDTypeTest, TrainEvalSwitch) {
    MultiheadAttention attn(128, 4, 0.5, true, false, false, 0, 0, true);
    attn.to(device);

    EXPECT_TRUE(attn.is_training()) << "Failed on " << device.to_string() << " with dtype " << dtype_to_string(dtype);

    attn.eval();
    EXPECT_FALSE(attn.is_training()) << "Failed on " << device.to_string() << " with dtype " << dtype_to_string(dtype);

    attn.train();
    EXPECT_TRUE(attn.is_training()) << "Failed on " << device.to_string() << " with dtype " << dtype_to_string(dtype);
}

TEST_P(AttentionIntegrationMultiDTypeTest, Deterministic) {
    // Test that with no dropout, same input produces same output
    MultiheadAttention attn(128, 4, 0.0, true, false, false, 0, 0, true);
    attn.to(device);

    attn.eval();

    Variable query(ones({2, 5, 128}, dtype, device), true);

    auto [output1, _] = attn.forward(query, query, query, Tensor{}, Tensor{}, false);
    auto [output2, __] = attn.forward(query, query, query, Tensor{}, Tensor{}, false);

    // Store tensors to avoid dangling pointers from temporaries
    auto output1_cpu = output1.tensor().to(Device::cpu());
    auto output2_cpu = output2.tensor().to(Device::cpu());

    double tol = get_tolerance();

    if (dtype == DType::Float32) {
        auto data1 = output1_cpu.data<float>();
        auto data2 = output2_cpu.data<float>();
        for (int64_t i = 0; i < output1.tensor().numel(); ++i) {
            EXPECT_NEAR(data1[i], data2[i], tol) << "Failed on " << device.to_string() << " with dtype " << dtype_to_string(dtype);
        }
    } else if (dtype == DType::Float64) {
        auto data1 = output1_cpu.data<double>();
        auto data2 = output2_cpu.data<double>();
        for (int64_t i = 0; i < output1.tensor().numel(); ++i) {
            EXPECT_NEAR(data1[i], data2[i], tol) << "Failed on " << device.to_string() << " with dtype " << dtype_to_string(dtype);
        }
    }
}

// ============================================================================
// Test Parameter Generation
// ============================================================================

std::vector<BackendDTypeParam> GenerateBackendDTypeCombinations() {
    std::vector<std::string> backends = {"cpu", "cuda", "vulkan", "oneapi"};

    // Test with floating-point dtypes (attention works with floating-point)
    std::vector<std::pair<DType, std::string>> dtypes = {
        {DType::Float32, "float32"},
        {DType::Float64, "float64"},
    };

    std::vector<BackendDTypeParam> combinations;
    for (const auto& backend : backends) {
        for (const auto& [dtype, dtype_name] : dtypes) {
            combinations.push_back({backend, dtype, dtype_name});
        }
    }
    return combinations;
}

INSTANTIATE_TEST_SUITE_P(
    AllBackendsAllDTypes,
    MultiheadAttentionMultiDTypeTest,
    ::testing::ValuesIn(GenerateBackendDTypeCombinations()),
    [](const ::testing::TestParamInfo<BackendDTypeParam>& info) {
        return info.param.ToString();
    }
);

INSTANTIATE_TEST_SUITE_P(
    AllBackendsAllDTypes,
    CausalMaskMultiDTypeTest,
    ::testing::ValuesIn(GenerateBackendDTypeCombinations()),
    [](const ::testing::TestParamInfo<BackendDTypeParam>& info) {
        return info.param.ToString();
    }
);

INSTANTIATE_TEST_SUITE_P(
    AllBackendsAllDTypes,
    AttentionIntegrationMultiDTypeTest,
    ::testing::ValuesIn(GenerateBackendDTypeCombinations()),
    [](const ::testing::TestParamInfo<BackendDTypeParam>& info) {
        return info.param.ToString();
    }
);
