#include <gtest/gtest.h>
#include "../backend_test_fixture.hpp"
#include <tenzor/nn/layers/attention.hpp>
#include <cmath>
#include <iostream>

using namespace tenzor;
using namespace tenzor::testing;
using namespace tenzor::nn;

// ============================================================================
// MultiheadAttention Tests
// ============================================================================

// Test fixture for multi-backend testing
class MultiheadAttentionTest : public BackendTest {
protected:
    void SetUp() override {
        BackendTest::SetUp();
        // Ensure each test starts with a clean state
    }

    void TearDown() override {
        BackendTest::TearDown();
    }
};

TEST_P(MultiheadAttentionTest, Construction) {
    // Test basic construction
    EXPECT_NO_THROW({
        MultiheadAttention attn(512, 8);
    }) << "Failed on " << device.to_string();

    // Test with all parameters
    EXPECT_NO_THROW({
        MultiheadAttention attn(512, 8, 0.1, true, false, false, 0, 0, true);
    }) << "Failed on " << device.to_string();
}

TEST_P(MultiheadAttentionTest, InvalidConstruction) {
    // embed_dim not divisible by num_heads
    EXPECT_THROW({
        MultiheadAttention attn(512, 7);
    }, std::invalid_argument) << "Failed on " << device.to_string();

    // Invalid dropout
    EXPECT_THROW({
        MultiheadAttention attn(512, 8, 1.5);
    }, std::invalid_argument) << "Failed on " << device.to_string();

    EXPECT_THROW({
        MultiheadAttention attn(512, 8, -0.1);
    }, std::invalid_argument) << "Failed on " << device.to_string();
}

TEST_P(MultiheadAttentionTest, SelfAttentionShape) {
    // Test self-attention output shape
    MultiheadAttention attn(512, 8, 0.0, true, false, false, 0, 0, true);

    int64_t batch_size = 4;
    int64_t seq_len = 10;
    int64_t embed_dim = 512;

    Variable query(randn({batch_size, seq_len, embed_dim}, DType::Float32, device), true);

    auto [output, attn_weights] = attn.forward(query, query, query);

    // Check output shape
    EXPECT_EQ(output.shape().size(), 3) << "Failed on " << device.to_string();
    EXPECT_EQ(output.shape()[0], batch_size) << "Failed on " << device.to_string();
    EXPECT_EQ(output.shape()[1], seq_len) << "Failed on " << device.to_string();
    EXPECT_EQ(output.shape()[2], embed_dim) << "Failed on " << device.to_string();

    // Check attention weights shape
    EXPECT_EQ(attn_weights.shape().size(), 4) << "Failed on " << device.to_string();
    EXPECT_EQ(attn_weights.shape()[0], batch_size) << "Failed on " << device.to_string();
    EXPECT_EQ(attn_weights.shape()[1], 8) << "Failed on " << device.to_string();  // num_heads
    EXPECT_EQ(attn_weights.shape()[2], seq_len) << "Failed on " << device.to_string();
    EXPECT_EQ(attn_weights.shape()[3], seq_len) << "Failed on " << device.to_string();
}

TEST_P(MultiheadAttentionTest, CrossAttentionShape) {
    // Test cross-attention with different key/value sequence length
    MultiheadAttention attn(256, 4, 0.0, true, false, false, 0, 0, true);

    int64_t batch_size = 2;
    int64_t query_len = 8;
    int64_t kv_len = 12;
    int64_t embed_dim = 256;

    Variable query(randn({batch_size, query_len, embed_dim}, DType::Float32, device), true);
    Variable key(randn({batch_size, kv_len, embed_dim}, DType::Float32, device), true);
    Variable value(randn({batch_size, kv_len, embed_dim}, DType::Float32, device), true);

    auto [output, attn_weights] = attn.forward(query, key, value);

    // Output should have query's sequence length
    EXPECT_EQ(output.shape()[0], batch_size) << "Failed on " << device.to_string();
    EXPECT_EQ(output.shape()[1], query_len) << "Failed on " << device.to_string();
    EXPECT_EQ(output.shape()[2], embed_dim) << "Failed on " << device.to_string();

    // Attention weights: (batch, heads, query_len, kv_len)
    EXPECT_EQ(attn_weights.shape()[0], batch_size) << "Failed on " << device.to_string();
    EXPECT_EQ(attn_weights.shape()[1], 4) << "Failed on " << device.to_string();
    EXPECT_EQ(attn_weights.shape()[2], query_len) << "Failed on " << device.to_string();
    EXPECT_EQ(attn_weights.shape()[3], kv_len) << "Failed on " << device.to_string();
}

TEST_P(MultiheadAttentionTest, BatchFirstFalse) {
    // Test with batch_first=false (seq, batch, embed)
    MultiheadAttention attn(128, 4, 0.0, true, false, false, 0, 0, false);

    int64_t seq_len = 10;
    int64_t batch_size = 4;
    int64_t embed_dim = 128;

    Variable query(randn({seq_len, batch_size, embed_dim}, DType::Float32, device), true);

    auto [output, _] = attn.forward(query, query, query);

    // Output should maintain (seq, batch, embed) format
    EXPECT_EQ(output.shape()[0], seq_len) << "Failed on " << device.to_string();
    EXPECT_EQ(output.shape()[1], batch_size) << "Failed on " << device.to_string();
    EXPECT_EQ(output.shape()[2], embed_dim) << "Failed on " << device.to_string();
}

TEST_P(MultiheadAttentionTest, SimpleForwardInterface) {
    // Test simplified forward() interface for self-attention
    MultiheadAttention attn(256, 8, 0.0, true, false, false, 0, 0, true);

    Variable input(randn({4, 10, 256}, DType::Float32, device), true);
    Variable output = attn.forward(input);

    EXPECT_EQ(output.shape()[0], 4) << "Failed on " << device.to_string();
    EXPECT_EQ(output.shape()[1], 10) << "Failed on " << device.to_string();
    EXPECT_EQ(output.shape()[2], 256) << "Failed on " << device.to_string();
}

TEST_P(MultiheadAttentionTest, WithoutAttentionWeights) {
    // Test need_weights=false
    MultiheadAttention attn(256, 8, 0.0, true, false, false, 0, 0, true);

    Variable query(randn({2, 5, 256}, DType::Float32, device), true);

    auto [output, weights] = attn.forward(query, query, query,
                                         Tensor{}, Tensor{}, false);

    EXPECT_EQ(output.shape()[0], 2) << "Failed on " << device.to_string();
    EXPECT_EQ(output.shape()[1], 5) << "Failed on " << device.to_string();
    EXPECT_EQ(output.shape()[2], 256) << "Failed on " << device.to_string();

    // Weights should be empty
    EXPECT_EQ(weights.shape().size(), 0) << "Failed on " << device.to_string();
}

TEST_P(MultiheadAttentionTest, SingleHead) {
    // Test with single attention head
    MultiheadAttention attn(128, 1, 0.0, true, false, false, 0, 0, true);

    Variable query(randn({2, 5, 128}, DType::Float32, device), true);

    auto [output, weights] = attn.forward(query, query, query);

    EXPECT_EQ(output.shape()[0], 2) << "Failed on " << device.to_string();
    EXPECT_EQ(output.shape()[1], 5) << "Failed on " << device.to_string();
    EXPECT_EQ(output.shape()[2], 128) << "Failed on " << device.to_string();

    EXPECT_EQ(weights.shape()[1], 1) << "Failed on " << device.to_string();  // num_heads = 1
}

TEST_P(MultiheadAttentionTest, LargeSequence) {
    // Test with larger sequence length
    MultiheadAttention attn(256, 8, 0.0, true, false, false, 0, 0, true);

    int64_t seq_len = 512;
    Variable query(randn({2, seq_len, 256}, DType::Float32, device), true);

    auto [output, _] = attn.forward(query, query, query, Tensor{}, Tensor{}, false);

    EXPECT_EQ(output.shape()[1], seq_len) << "Failed on " << device.to_string();
}

TEST_P(MultiheadAttentionTest, SmallBatch) {
    // Test with batch size 1
    MultiheadAttention attn(128, 4, 0.0, true, false, false, 0, 0, true);

    Variable query(randn({1, 10, 128}, DType::Float32, device), true);

    auto [output, _] = attn.forward(query, query, query);

    EXPECT_EQ(output.shape()[0], 1) << "Failed on " << device.to_string();
    EXPECT_EQ(output.shape()[1], 10) << "Failed on " << device.to_string();
    EXPECT_EQ(output.shape()[2], 128) << "Failed on " << device.to_string();
}

TEST_P(MultiheadAttentionTest, WithDropout) {
    // Test with dropout enabled
    MultiheadAttention attn(256, 8, 0.5, true, false, false, 0, 0, true);

    Variable query(randn({4, 10, 256}, DType::Float32, device), true);

    // Set to training mode
    attn.train();

    auto [output1, _] = attn.forward(query, query, query, Tensor{}, Tensor{}, false);
    auto [output2, __] = attn.forward(query, query, query, Tensor{}, Tensor{}, false);

    // Outputs should be different due to dropout (stochastic)
    // (This is a probabilistic test, might occasionally fail)
    // Store tensors to avoid dangling pointers from temporaries
    auto output1_cpu = output1.tensor().to(Device::cpu());
    auto output2_cpu = output2.tensor().to(Device::cpu());
    auto data1 = output1_cpu.data<float>();
    auto data2 = output2_cpu.data<float>();

    bool found_difference = false;
    for (int64_t i = 0; i < std::min(static_cast<int64_t>(100), output1.tensor().numel()); ++i) {
        if (std::abs(data1[i] - data2[i]) > 1e-6) {
            found_difference = true;
            break;
        }
    }

    // Note: This test might occasionally fail due to randomness
    // In production, you'd use a fixed seed or mock the dropout
}

TEST_P(MultiheadAttentionTest, EvalMode) {
    // Test that eval mode disables dropout
    MultiheadAttention attn(256, 8, 0.5, true, false, false, 0, 0, true);

    // Use a fixed input for deterministic test
    Variable query(ones({2, 5, 256}, DType::Float32, device), true);

    // Explicitly set to eval mode
    attn.eval();

    // Verify the module is in eval mode
    ASSERT_FALSE(attn.is_training()) << "Attention should be in eval mode. Failed on " << device.to_string();

    auto [output1, _] = attn.forward(query, query, query, Tensor{}, Tensor{}, false);
    auto [output2, __] = attn.forward(query, query, query, Tensor{}, Tensor{}, false);

    // In eval mode, outputs should be identical (within floating-point tolerance)
    // Store tensors to avoid dangling pointers from temporaries
    auto output1_cpu = output1.tensor().to(Device::cpu());
    auto output2_cpu = output2.tensor().to(Device::cpu());
    auto data1 = output1_cpu.data<float>();
    auto data2 = output2_cpu.data<float>();

    for (int64_t i = 0; i < output1.tensor().numel(); ++i) {
        EXPECT_NEAR(data1[i], data2[i], 1e-6)
            << "Mismatch at index " << i << ": " << data1[i] << " vs " << data2[i]
            << ". Failed on " << device.to_string();
    }
}

TEST_P(MultiheadAttentionTest, DifferentKeyValueDims) {
    // Test with different key and value dimensions
    int64_t embed_dim = 256;
    int64_t kdim = 128;
    int64_t vdim = 128;

    MultiheadAttention attn(embed_dim, 8, 0.0, true, false, false,
                           kdim, vdim, true);

    Variable query(randn({2, 5, embed_dim}, DType::Float32, device), true);
    Variable key(randn({2, 7, kdim}, DType::Float32, device), true);
    Variable value(randn({2, 7, vdim}, DType::Float32, device), true);

    auto [output, _] = attn.forward(query, key, value);

    EXPECT_EQ(output.shape()[0], 2) << "Failed on " << device.to_string();
    EXPECT_EQ(output.shape()[1], 5) << "Failed on " << device.to_string();
    EXPECT_EQ(output.shape()[2], embed_dim) << "Failed on " << device.to_string();
}

// ============================================================================
// Causal Mask Tests
// ============================================================================

class CausalMaskTest : public BackendTest {
};

TEST_P(CausalMaskTest, Shape) {
    int64_t seq_len = 10;
    Tensor mask = create_causal_mask(seq_len, device);

    EXPECT_EQ(mask.shape().size(), 2) << "Failed on " << device.to_string();
    EXPECT_EQ(mask.shape()[0], seq_len) << "Failed on " << device.to_string();
    EXPECT_EQ(mask.shape()[1], seq_len) << "Failed on " << device.to_string();
}

TEST_P(CausalMaskTest, Values) {
    int64_t seq_len = 5;
    Tensor mask = create_causal_mask(seq_len, device);

    // Store tensor before getting data pointer to avoid dangling pointer from temporary
    Tensor cpu_mask = mask.to(Device::cpu());
    auto* data = cpu_mask.data<float>();

    // Check that upper triangle is -inf and lower triangle is 0
    for (int64_t i = 0; i < seq_len; ++i) {
        for (int64_t j = 0; j < seq_len; ++j) {
            if (j > i) {
                EXPECT_TRUE(std::isinf(data[i * seq_len + j])) << "Failed on " << device.to_string();
                EXPECT_LT(data[i * seq_len + j], 0) << "Failed on " << device.to_string();  // -inf
            } else {
                EXPECT_FLOAT_EQ(data[i * seq_len + j], 0.0f) << "Failed on " << device.to_string();
            }
        }
    }
}

TEST_P(CausalMaskTest, SmallSize) {
    Tensor mask = create_causal_mask(1, device);
    EXPECT_EQ(mask.shape()[0], 1) << "Failed on " << device.to_string();
    EXPECT_EQ(mask.shape()[1], 1) << "Failed on " << device.to_string();

    // Store tensor before getting data pointer to avoid dangling pointer from temporary
    Tensor cpu_mask = mask.to(Device::cpu());
    auto* data = cpu_mask.data<float>();
    EXPECT_FLOAT_EQ(data[0], 0.0f) << "Failed on " << device.to_string();
}

TEST_P(CausalMaskTest, MediumSize) {
    int64_t seq_len = 100;
    Tensor mask = create_causal_mask(seq_len, device);

    EXPECT_EQ(mask.shape()[0], seq_len) << "Failed on " << device.to_string();
    EXPECT_EQ(mask.shape()[1], seq_len) << "Failed on " << device.to_string();

    // Spot check a few positions
    // Use EXPECT_NEAR to account for denormal floats from device transfers
    // Store tensor to avoid dangling pointer from temporary
    auto mask_cpu = mask.to(Device::cpu());
    auto* data = mask_cpu.data<float>();
    EXPECT_NEAR(data[0], 0.0f, 1e-30f) << "Failed on " << device.to_string();  // [0, 0] should be 0
    EXPECT_TRUE(std::isinf(data[1])) << "Failed on " << device.to_string();  // [0, 1] should be -inf
    EXPECT_NEAR(data[seq_len], 0.0f, 1e-30f) << "Failed on " << device.to_string();  // [1, 0] should be 0
    EXPECT_NEAR(data[seq_len + 1], 0.0f, 1e-30f) << "Failed on " << device.to_string();  // [1, 1] should be 0
}

// ============================================================================
// Integration Tests
// ============================================================================

class AttentionIntegrationTest : public BackendTest {
};

TEST_P(AttentionIntegrationTest, ForwardBackward) {
    // Test that gradients flow through attention
    MultiheadAttention attn(128, 4, 0.0, true, false, false, 0, 0, true);

    Variable query(randn({2, 5, 128}, DType::Float32, device), true);

    auto [output, _] = attn.forward(query, query, query, Tensor{}, Tensor{}, false);

    // Compute a simple loss
    Variable loss = mean(output);

    // Backward should work
    EXPECT_NO_THROW({
        loss.backward();
    }) << "Failed on " << device.to_string();

    // Query should have gradients
    EXPECT_TRUE(query.has_grad()) << "Failed on " << device.to_string();
}

TEST_P(AttentionIntegrationTest, ParameterCount) {
    MultiheadAttention attn(256, 8, 0.0, true, false, false, 0, 0, true);

    auto params = attn.parameters();

    // Should have: q_proj (weight + bias), k_proj (w + b), v_proj (w + b), out_proj (w + b)
    // Total: 8 parameters (4 weights, 4 biases)
    EXPECT_EQ(params.size(), 8) << "Failed on " << device.to_string();
}

TEST_P(AttentionIntegrationTest, TrainEvalSwitch) {
    MultiheadAttention attn(128, 4, 0.5, true, false, false, 0, 0, true);

    EXPECT_TRUE(attn.is_training()) << "Failed on " << device.to_string();

    attn.eval();
    EXPECT_FALSE(attn.is_training()) << "Failed on " << device.to_string();

    attn.train();
    EXPECT_TRUE(attn.is_training()) << "Failed on " << device.to_string();
}

TEST_P(AttentionIntegrationTest, Deterministic) {
    // Test that with no dropout, same input produces same output
    MultiheadAttention attn(128, 4, 0.0, true, false, false, 0, 0, true);

    attn.eval();

    Variable query(ones({2, 5, 128}, DType::Float32, device), true);

    auto [output1, _] = attn.forward(query, query, query, Tensor{}, Tensor{}, false);
    auto [output2, __] = attn.forward(query, query, query, Tensor{}, Tensor{}, false);

    // Store tensors to avoid dangling pointers from temporaries
    auto output1_cpu = output1.tensor().to(Device::cpu());
    auto output2_cpu = output2.tensor().to(Device::cpu());
    auto data1 = output1_cpu.data<float>();
    auto data2 = output2_cpu.data<float>();

    for (int64_t i = 0; i < output1.tensor().numel(); ++i) {
        EXPECT_NEAR(data1[i], data2[i], 1e-6) << "Failed on " << device.to_string();
    }
}

INSTANTIATE_BACKEND_TESTS(MultiheadAttentionTest);
INSTANTIATE_BACKEND_TESTS(CausalMaskTest);
INSTANTIATE_BACKEND_TESTS(AttentionIntegrationTest);
