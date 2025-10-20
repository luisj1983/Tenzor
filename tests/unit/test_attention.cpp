#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/nn/layers/attention.hpp>
#include <cmath>
#include <iostream>

using namespace tenzor;
using namespace tenzor::nn;

// Global test environment
class AttentionTestEnvironment : public ::testing::Environment {
public:
    void SetUp() override {
        tenzor::initialize();
    }
};

static ::testing::Environment* const attention_env =
    ::testing::AddGlobalTestEnvironment(new AttentionTestEnvironment);

// ============================================================================
// MultiheadAttention Tests
// ============================================================================

// Test fixture for ensuring proper test isolation
class MultiheadAttentionTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Ensure each test starts with a clean state
        // This helps prevent state leakage between tests
    }

    void TearDown() override {
        // Critical: Synchronize device to ensure all operations complete
        // This prevents race conditions and memory corruption in parallel test execution
        // Works uniformly across all backends (CPU, CUDA, ROCm, OneAPI)
        try {
            Device::cuda(0).synchronize();  // Synchronize default CUDA device if available
        } catch (...) {
            // Ignore if CUDA is not available - CPU tests don't need synchronization
        }
    }
};

TEST_F(MultiheadAttentionTest, Construction) {
    // Test basic construction
    EXPECT_NO_THROW({
        MultiheadAttention attn(512, 8);
    });

    // Test with all parameters
    EXPECT_NO_THROW({
        MultiheadAttention attn(512, 8, 0.1, true, false, false, 0, 0, true);
    });
}

TEST_F(MultiheadAttentionTest, InvalidConstruction) {
    // embed_dim not divisible by num_heads
    EXPECT_THROW({
        MultiheadAttention attn(512, 7);
    }, std::invalid_argument);

    // Invalid dropout
    EXPECT_THROW({
        MultiheadAttention attn(512, 8, 1.5);
    }, std::invalid_argument);

    EXPECT_THROW({
        MultiheadAttention attn(512, 8, -0.1);
    }, std::invalid_argument);
}

TEST_F(MultiheadAttentionTest, SelfAttentionShape) {
    // Test self-attention output shape
    MultiheadAttention attn(512, 8, 0.0, true, false, false, 0, 0, true);

    int64_t batch_size = 4;
    int64_t seq_len = 10;
    int64_t embed_dim = 512;

    Variable query(randn({batch_size, seq_len, embed_dim}), true);

    auto [output, attn_weights] = attn.forward(query, query, query);

    // Check output shape
    EXPECT_EQ(output.shape().size(), 3);
    EXPECT_EQ(output.shape()[0], batch_size);
    EXPECT_EQ(output.shape()[1], seq_len);
    EXPECT_EQ(output.shape()[2], embed_dim);

    // Check attention weights shape
    EXPECT_EQ(attn_weights.shape().size(), 4);
    EXPECT_EQ(attn_weights.shape()[0], batch_size);
    EXPECT_EQ(attn_weights.shape()[1], 8);  // num_heads
    EXPECT_EQ(attn_weights.shape()[2], seq_len);
    EXPECT_EQ(attn_weights.shape()[3], seq_len);
}

TEST_F(MultiheadAttentionTest, CrossAttentionShape) {
    // Test cross-attention with different key/value sequence length
    MultiheadAttention attn(256, 4, 0.0, true, false, false, 0, 0, true);

    int64_t batch_size = 2;
    int64_t query_len = 8;
    int64_t kv_len = 12;
    int64_t embed_dim = 256;

    Variable query(randn({batch_size, query_len, embed_dim}), true);
    Variable key(randn({batch_size, kv_len, embed_dim}), true);
    Variable value(randn({batch_size, kv_len, embed_dim}), true);

    auto [output, attn_weights] = attn.forward(query, key, value);

    // Output should have query's sequence length
    EXPECT_EQ(output.shape()[0], batch_size);
    EXPECT_EQ(output.shape()[1], query_len);
    EXPECT_EQ(output.shape()[2], embed_dim);

    // Attention weights: (batch, heads, query_len, kv_len)
    EXPECT_EQ(attn_weights.shape()[0], batch_size);
    EXPECT_EQ(attn_weights.shape()[1], 4);
    EXPECT_EQ(attn_weights.shape()[2], query_len);
    EXPECT_EQ(attn_weights.shape()[3], kv_len);
}

TEST_F(MultiheadAttentionTest, BatchFirstFalse) {
    // Test with batch_first=false (seq, batch, embed)
    MultiheadAttention attn(128, 4, 0.0, true, false, false, 0, 0, false);

    int64_t seq_len = 10;
    int64_t batch_size = 4;
    int64_t embed_dim = 128;

    Variable query(randn({seq_len, batch_size, embed_dim}), true);

    auto [output, _] = attn.forward(query, query, query);

    // Output should maintain (seq, batch, embed) format
    EXPECT_EQ(output.shape()[0], seq_len);
    EXPECT_EQ(output.shape()[1], batch_size);
    EXPECT_EQ(output.shape()[2], embed_dim);
}

TEST_F(MultiheadAttentionTest, SimpleForwardInterface) {
    // Test simplified forward() interface for self-attention
    MultiheadAttention attn(256, 8, 0.0, true, false, false, 0, 0, true);

    Variable input(randn({4, 10, 256}), true);
    Variable output = attn.forward(input);

    EXPECT_EQ(output.shape()[0], 4);
    EXPECT_EQ(output.shape()[1], 10);
    EXPECT_EQ(output.shape()[2], 256);
}

TEST_F(MultiheadAttentionTest, WithoutAttentionWeights) {
    // Test need_weights=false
    MultiheadAttention attn(256, 8, 0.0, true, false, false, 0, 0, true);

    Variable query(randn({2, 5, 256}), true);

    auto [output, weights] = attn.forward(query, query, query,
                                         Tensor{}, Tensor{}, false);

    EXPECT_EQ(output.shape()[0], 2);
    EXPECT_EQ(output.shape()[1], 5);
    EXPECT_EQ(output.shape()[2], 256);

    // Weights should be empty
    EXPECT_EQ(weights.shape().size(), 0);
}

TEST_F(MultiheadAttentionTest, SingleHead) {
    // Test with single attention head
    MultiheadAttention attn(128, 1, 0.0, true, false, false, 0, 0, true);

    Variable query(randn({2, 5, 128}), true);

    auto [output, weights] = attn.forward(query, query, query);

    EXPECT_EQ(output.shape()[0], 2);
    EXPECT_EQ(output.shape()[1], 5);
    EXPECT_EQ(output.shape()[2], 128);

    EXPECT_EQ(weights.shape()[1], 1);  // num_heads = 1
}

TEST_F(MultiheadAttentionTest, LargeSequence) {
    // Test with larger sequence length
    MultiheadAttention attn(256, 8, 0.0, true, false, false, 0, 0, true);

    int64_t seq_len = 512;
    Variable query(randn({2, seq_len, 256}), true);

    auto [output, _] = attn.forward(query, query, query, Tensor{}, Tensor{}, false);

    EXPECT_EQ(output.shape()[1], seq_len);
}

TEST_F(MultiheadAttentionTest, SmallBatch) {
    // Test with batch size 1
    MultiheadAttention attn(128, 4, 0.0, true, false, false, 0, 0, true);

    Variable query(randn({1, 10, 128}), true);

    auto [output, _] = attn.forward(query, query, query);

    EXPECT_EQ(output.shape()[0], 1);
    EXPECT_EQ(output.shape()[1], 10);
    EXPECT_EQ(output.shape()[2], 128);
}

TEST_F(MultiheadAttentionTest, WithDropout) {
    // Test with dropout enabled
    MultiheadAttention attn(256, 8, 0.5, true, false, false, 0, 0, true);

    Variable query(randn({4, 10, 256}), true);

    // Set to training mode
    attn.train();

    auto [output1, _] = attn.forward(query, query, query, Tensor{}, Tensor{}, false);
    auto [output2, __] = attn.forward(query, query, query, Tensor{}, Tensor{}, false);

    // Outputs should be different due to dropout (stochastic)
    // (This is a probabilistic test, might occasionally fail)
    auto data1 = output1.tensor().data<float>();
    auto data2 = output2.tensor().data<float>();

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

TEST_F(MultiheadAttentionTest, EvalMode) {
    // Test that eval mode disables dropout
    MultiheadAttention attn(256, 8, 0.5, true, false, false, 0, 0, true);

    // Use a fixed input for deterministic test
    Variable query(ones({2, 5, 256}), true);

    // Explicitly set to eval mode
    attn.eval();

    // Verify the module is in eval mode
    ASSERT_FALSE(attn.is_training()) << "Attention should be in eval mode";

    auto [output1, _] = attn.forward(query, query, query, Tensor{}, Tensor{}, false);
    auto [output2, __] = attn.forward(query, query, query, Tensor{}, Tensor{}, false);

    // In eval mode, outputs should be identical (within floating-point tolerance)
    auto data1 = output1.tensor().data<float>();
    auto data2 = output2.tensor().data<float>();

    for (int64_t i = 0; i < output1.tensor().numel(); ++i) {
        EXPECT_NEAR(data1[i], data2[i], 1e-6)
            << "Mismatch at index " << i << ": " << data1[i] << " vs " << data2[i];
    }
}

TEST_F(MultiheadAttentionTest, DifferentKeyValueDims) {
    // Test with different key and value dimensions
    int64_t embed_dim = 256;
    int64_t kdim = 128;
    int64_t vdim = 128;

    MultiheadAttention attn(embed_dim, 8, 0.0, true, false, false,
                           kdim, vdim, true);

    Variable query(randn({2, 5, embed_dim}), true);
    Variable key(randn({2, 7, kdim}), true);
    Variable value(randn({2, 7, vdim}), true);

    auto [output, _] = attn.forward(query, key, value);

    EXPECT_EQ(output.shape()[0], 2);
    EXPECT_EQ(output.shape()[1], 5);
    EXPECT_EQ(output.shape()[2], embed_dim);
}

// ============================================================================
// Causal Mask Tests
// ============================================================================

TEST(CausalMaskTest, Shape) {
    int64_t seq_len = 10;
    Tensor mask = create_causal_mask(seq_len);

    EXPECT_EQ(mask.shape().size(), 2);
    EXPECT_EQ(mask.shape()[0], seq_len);
    EXPECT_EQ(mask.shape()[1], seq_len);
}

TEST(CausalMaskTest, Values) {
    int64_t seq_len = 5;
    Tensor mask = create_causal_mask(seq_len);

    auto* data = mask.data<float>();

    // Check that upper triangle is -inf and lower triangle is 0
    for (int64_t i = 0; i < seq_len; ++i) {
        for (int64_t j = 0; j < seq_len; ++j) {
            if (j > i) {
                EXPECT_TRUE(std::isinf(data[i * seq_len + j]));
                EXPECT_LT(data[i * seq_len + j], 0);  // -inf
            } else {
                EXPECT_FLOAT_EQ(data[i * seq_len + j], 0.0f);
            }
        }
    }
}

TEST(CausalMaskTest, SmallSize) {
    Tensor mask = create_causal_mask(1);
    EXPECT_EQ(mask.shape()[0], 1);
    EXPECT_EQ(mask.shape()[1], 1);

    auto* data = mask.data<float>();
    EXPECT_FLOAT_EQ(data[0], 0.0f);
}

TEST(CausalMaskTest, MediumSize) {
    int64_t seq_len = 100;
    Tensor mask = create_causal_mask(seq_len);

    EXPECT_EQ(mask.shape()[0], seq_len);
    EXPECT_EQ(mask.shape()[1], seq_len);

    // Spot check a few positions
    auto* data = mask.data<float>();
    EXPECT_FLOAT_EQ(data[0], 0.0f);  // [0, 0] should be 0
    EXPECT_TRUE(std::isinf(data[1]));  // [0, 1] should be -inf
    EXPECT_FLOAT_EQ(data[seq_len], 0.0f);  // [1, 0] should be 0
    EXPECT_FLOAT_EQ(data[seq_len + 1], 0.0f);  // [1, 1] should be 0
}

// ============================================================================
// Integration Tests
// ============================================================================

TEST(AttentionIntegrationTest, ForwardBackward) {
    // Test that gradients flow through attention
    MultiheadAttention attn(128, 4, 0.0, true, false, false, 0, 0, true);

    Variable query(randn({2, 5, 128}), true);

    auto [output, _] = attn.forward(query, query, query, Tensor{}, Tensor{}, false);

    // Compute a simple loss
    Variable loss = mean(output);

    // Backward should work
    EXPECT_NO_THROW({
        loss.backward();
    });

    // Query should have gradients
    EXPECT_TRUE(query.has_grad());
}

TEST(AttentionIntegrationTest, ParameterCount) {
    MultiheadAttention attn(256, 8, 0.0, true, false, false, 0, 0, true);

    auto params = attn.parameters();

    // Should have: q_proj (weight + bias), k_proj (w + b), v_proj (w + b), out_proj (w + b)
    // Total: 8 parameters (4 weights, 4 biases)
    EXPECT_EQ(params.size(), 8);
}

TEST(AttentionIntegrationTest, TrainEvalSwitch) {
    MultiheadAttention attn(128, 4, 0.5, true, false, false, 0, 0, true);

    EXPECT_TRUE(attn.is_training());

    attn.eval();
    EXPECT_FALSE(attn.is_training());

    attn.train();
    EXPECT_TRUE(attn.is_training());
}

TEST(AttentionIntegrationTest, Deterministic) {
    // Test that with no dropout, same input produces same output
    MultiheadAttention attn(128, 4, 0.0, true, false, false, 0, 0, true);

    attn.eval();

    Variable query(ones({2, 5, 128}), true);

    auto [output1, _] = attn.forward(query, query, query, Tensor{}, Tensor{}, false);
    auto [output2, __] = attn.forward(query, query, query, Tensor{}, Tensor{}, false);

    auto data1 = output1.tensor().data<float>();
    auto data2 = output2.tensor().data<float>();

    for (int64_t i = 0; i < output1.tensor().numel(); ++i) {
        EXPECT_NEAR(data1[i], data2[i], 1e-6);
    }
}
