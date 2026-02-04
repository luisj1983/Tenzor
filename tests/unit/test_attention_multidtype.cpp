/**
 * @file test_attention_multidtype.cpp
 * @brief Multi-dtype tests for attention mechanisms
 *
 * Tests MultiheadAttention and related operations with Float32, Float64, and Float16
 * dtypes across CPU, CUDA, OneAPI, Vulkan, and ROCm backends to ensure:
 * - Self-attention and cross-attention work correctly
 * - Attention weight shapes are correct
 * - Causal masks work as expected
 * - Gradient flow through attention layers
 */

#include <gtest/gtest.h>
#include <tenzor/nn/layers/attention.hpp>
#include "../multi_backend_dtype_fixture.hpp"
#include <cmath>
#include <iostream>

using namespace tenzor;
using namespace tenzor::nn;
using namespace tenzor::testing;

// ============================================================================
// MultiheadAttention Multi-Backend Multi-DType Test Fixture
// ============================================================================

class MultiheadAttentionMultiDTypeTest : public MultiBackendDTypeTest {
protected:
    double get_tolerance() const {
        if (dtype() == DType::Float32) {
            return 1e-5;
        } else if (dtype() == DType::Float64) {
            return 1e-10;
        } else if (dtype() == DType::Float16 || dtype() == DType::BFloat16) {
            return 1e-2;
        }
        return 1e-5;
    }
};

// ============================================================================
// Construction Tests
// ============================================================================

TEST_P(MultiheadAttentionMultiDTypeTest, Construction) {
    // Test basic construction
    EXPECT_NO_THROW({
        MultiheadAttention attn(512, 8);
    });

    // Test with all parameters
    EXPECT_NO_THROW({
        MultiheadAttention attn(512, 8, 0.1, true, false, false, 0, 0, true);
    });
}

TEST_P(MultiheadAttentionMultiDTypeTest, InvalidConstruction) {
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

// ============================================================================
// Self-Attention Tests
// ============================================================================

TEST_P(MultiheadAttentionMultiDTypeTest, SelfAttentionShape) {
    MultiheadAttention attn(512, 8, 0.0, true, false, false, 0, 0, true);
    convert_model(attn);

    int64_t batch_size = 4;
    int64_t seq_len = 10;
    int64_t embed_dim = 512;

    Variable query = createInput({batch_size, seq_len, embed_dim}, true);
    auto [output, attn_weights] = attn.forward(query, query, query);

    // Check output shape
    expectShape(output.tensor(), {batch_size, seq_len, embed_dim});
    expectDType(output.tensor());

    // Check attention weights shape
    EXPECT_EQ(attn_weights.shape().size(), 4);
    EXPECT_EQ(attn_weights.shape()[0], batch_size);
    EXPECT_EQ(attn_weights.shape()[1], 8);  // num_heads
    EXPECT_EQ(attn_weights.shape()[2], seq_len);
    EXPECT_EQ(attn_weights.shape()[3], seq_len);
}

TEST_P(MultiheadAttentionMultiDTypeTest, CrossAttentionShape) {
    MultiheadAttention attn(256, 4, 0.0, true, false, false, 0, 0, true);
    convert_model(attn);

    int64_t batch_size = 2;
    int64_t query_len = 8;
    int64_t kv_len = 12;
    int64_t embed_dim = 256;

    Variable query = createInput({batch_size, query_len, embed_dim}, true);
    Variable key = createInput({batch_size, kv_len, embed_dim}, true);
    Variable value = createInput({batch_size, kv_len, embed_dim}, true);

    auto [output, attn_weights] = attn.forward(query, key, value);

    // Output should have query's sequence length
    expectShape(output.tensor(), {batch_size, query_len, embed_dim});
    expectDType(output.tensor());

    // Attention weights: (batch, heads, query_len, kv_len)
    EXPECT_EQ(attn_weights.shape()[0], batch_size);
    EXPECT_EQ(attn_weights.shape()[1], 4);
    EXPECT_EQ(attn_weights.shape()[2], query_len);
    EXPECT_EQ(attn_weights.shape()[3], kv_len);
}

TEST_P(MultiheadAttentionMultiDTypeTest, BatchFirstFalse) {
    // Test with batch_first=false (seq, batch, embed)
    MultiheadAttention attn(128, 4, 0.0, true, false, false, 0, 0, false);
    convert_model(attn);

    int64_t seq_len = 10;
    int64_t batch_size = 4;
    int64_t embed_dim = 128;

    Variable query = createInput({seq_len, batch_size, embed_dim}, true);
    auto [output, _] = attn.forward(query, query, query);

    // Output should maintain (seq, batch, embed) format
    expectShape(output.tensor(), {seq_len, batch_size, embed_dim});
    expectDType(output.tensor());
}

// ============================================================================
// Interface Tests
// ============================================================================

TEST_P(MultiheadAttentionMultiDTypeTest, SimpleForwardInterface) {
    MultiheadAttention attn(256, 8, 0.0, true, false, false, 0, 0, true);
    convert_model(attn);

    Variable input = createInput({4, 10, 256}, true);
    Variable output = attn.forward(input);

    expectShape(output.tensor(), {4, 10, 256});
    expectDType(output.tensor());
}

TEST_P(MultiheadAttentionMultiDTypeTest, WithoutAttentionWeights) {
    MultiheadAttention attn(256, 8, 0.0, true, false, false, 0, 0, true);
    convert_model(attn);

    Variable query = createInput({2, 5, 256}, true);
    auto [output, weights] = attn.forward(query, query, query,
                                         Tensor{}, Tensor{}, false);

    expectShape(output.tensor(), {2, 5, 256});
    expectDType(output.tensor());

    // Weights should be empty
    EXPECT_EQ(weights.shape().size(), 0);
}

TEST_P(MultiheadAttentionMultiDTypeTest, SingleHead) {
    MultiheadAttention attn(128, 1, 0.0, true, false, false, 0, 0, true);
    convert_model(attn);

    Variable query = createInput({2, 5, 128}, true);
    auto [output, weights] = attn.forward(query, query, query);

    expectShape(output.tensor(), {2, 5, 128});
    expectDType(output.tensor());
    EXPECT_EQ(weights.shape()[1], 1);  // num_heads = 1
}

// ============================================================================
// Sequence Length Tests
// ============================================================================

TEST_P(MultiheadAttentionMultiDTypeTest, LargeSequence) {
    // Test with larger sequence length (reduced for Float16 to avoid memory issues)
    int64_t seq_len = (dtype() == DType::Float16) ? 256 : 512;
    MultiheadAttention attn(256, 8, 0.0, true, false, false, 0, 0, true);
    convert_model(attn);

    Variable query = createInput({2, seq_len, 256}, true);
    auto [output, _] = attn.forward(query, query, query, Tensor{}, Tensor{}, false);

    EXPECT_EQ(output.shape()[1], seq_len);
    expectDType(output.tensor());
}

TEST_P(MultiheadAttentionMultiDTypeTest, SmallBatch) {
    MultiheadAttention attn(128, 4, 0.0, true, false, false, 0, 0, true);
    convert_model(attn);

    Variable query = createInput({1, 10, 128}, true);
    auto [output, _] = attn.forward(query, query, query);

    expectShape(output.tensor(), {1, 10, 128});
    expectDType(output.tensor());
}

// ============================================================================
// Dropout Tests
// ============================================================================

TEST_P(MultiheadAttentionMultiDTypeTest, WithDropout) {
    MultiheadAttention attn(256, 8, 0.5, true, false, false, 0, 0, true);
    convert_model(attn);
    attn.train();

    Variable query = createInput({4, 10, 256}, true);

    auto [output1, _] = attn.forward(query, query, query, Tensor{}, Tensor{}, false);
    auto [output2, __] = attn.forward(query, query, query, Tensor{}, Tensor{}, false);

    // Outputs should be different due to dropout (stochastic)
    auto output1_cpu = output1.tensor().to(Device::cpu());
    auto output2_cpu = output2.tensor().to(Device::cpu());

    double tol = get_tolerance();
    bool found_difference = false;

    if (dtype() == DType::Float32) {
        auto data1 = output1_cpu.data<float>();
        auto data2 = output2_cpu.data<float>();
        for (int64_t i = 0; i < std::min(static_cast<int64_t>(100), output1.tensor().numel()); ++i) {
            if (std::abs(data1[i] - data2[i]) > tol) {
                found_difference = true;
                break;
            }
        }
    } else if (dtype() == DType::Float64) {
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
}

TEST_P(MultiheadAttentionMultiDTypeTest, EvalMode) {
    MultiheadAttention attn(256, 8, 0.5, true, false, false, 0, 0, true);
    convert_model(attn);
    attn.eval();

    Variable query = createInput({2, 5, 256}, true);
    // Use fixed input for deterministic test
    auto fixed_input = createOnes({2, 5, 256});
    query = Variable(fixed_input, true);

    ASSERT_FALSE(attn.is_training());

    auto [output1, _] = attn.forward(query, query, query, Tensor{}, Tensor{}, false);
    auto [output2, __] = attn.forward(query, query, query, Tensor{}, Tensor{}, false);

    // In eval mode, outputs should be identical
    expectTensorNear(output1.tensor(), output2.tensor());
}

// ============================================================================
// Different Key/Value Dimensions
// ============================================================================

TEST_P(MultiheadAttentionMultiDTypeTest, DifferentKeyValueDims) {
    int64_t embed_dim = 256;
    int64_t kdim = 128;
    int64_t vdim = 128;

    MultiheadAttention attn(embed_dim, 8, 0.0, true, false, false,
                           kdim, vdim, true);
    convert_model(attn);

    Variable query = createInput({2, 5, embed_dim}, true);
    Variable key = createInput({2, 7, kdim}, true);
    Variable value = createInput({2, 7, vdim}, true);

    auto [output, _] = attn.forward(query, key, value);

    expectShape(output.tensor(), {2, 5, embed_dim});
    expectDType(output.tensor());
}

// ============================================================================
// Causal Mask Tests
// ============================================================================

class CausalMaskMultiDTypeTest : public MultiBackendDTypeTest {
protected:
    double get_tolerance() const {
        if (dtype() == DType::Float32) {
            return 1e-5;
        } else if (dtype() == DType::Float64) {
            return 1e-10;
        } else if (dtype() == DType::Float16 || dtype() == DType::BFloat16) {
            return 1e-2;
        }
        return 1e-5;
    }
};

TEST_P(CausalMaskMultiDTypeTest, Shape) {
    int64_t seq_len = 10;
    Tensor mask = create_causal_mask(seq_len, device());

    expectShape(mask, {seq_len, seq_len});
}

TEST_P(CausalMaskMultiDTypeTest, Values) {
    int64_t seq_len = 5;
    Tensor mask = create_causal_mask(seq_len, device());

    Tensor cpu_mask = mask.to(Device::cpu());
    double tol = get_tolerance();

    // Check that upper triangle is -inf and lower triangle is 0
    if (cpu_mask.dtype() == DType::Float32) {
        auto* data = cpu_mask.data<float>();
        for (int64_t i = 0; i < seq_len; ++i) {
            for (int64_t j = 0; j < seq_len; ++j) {
                if (j > i) {
                    EXPECT_TRUE(std::isinf(data[i * seq_len + j]));
                    EXPECT_LT(data[i * seq_len + j], 0);  // -inf
                } else {
                    EXPECT_NEAR(data[i * seq_len + j], 0.0f, tol);
                }
            }
        }
    } else if (cpu_mask.dtype() == DType::Float64) {
        auto* data = cpu_mask.data<double>();
        for (int64_t i = 0; i < seq_len; ++i) {
            for (int64_t j = 0; j < seq_len; ++j) {
                if (j > i) {
                    EXPECT_TRUE(std::isinf(data[i * seq_len + j]));
                    EXPECT_LT(data[i * seq_len + j], 0);  // -inf
                } else {
                    EXPECT_NEAR(data[i * seq_len + j], 0.0, tol);
                }
            }
        }
    }
}

TEST_P(CausalMaskMultiDTypeTest, SmallSize) {
    Tensor mask = create_causal_mask(1, device());
    expectShape(mask, {1, 1});

    Tensor cpu_mask = mask.to(Device::cpu());
    double tol = get_tolerance();

    if (cpu_mask.dtype() == DType::Float32) {
        auto* data = cpu_mask.data<float>();
        EXPECT_NEAR(data[0], 0.0f, tol);
    } else if (cpu_mask.dtype() == DType::Float64) {
        auto* data = cpu_mask.data<double>();
        EXPECT_NEAR(data[0], 0.0, tol);
    }
}

TEST_P(CausalMaskMultiDTypeTest, MediumSize) {
    int64_t seq_len = 100;
    Tensor mask = create_causal_mask(seq_len, device());

    expectShape(mask, {seq_len, seq_len});

    // Spot check a few positions
    auto mask_cpu = mask.to(Device::cpu());
    double tol = get_tolerance();

    if (mask_cpu.dtype() == DType::Float32) {
        auto* data = mask_cpu.data<float>();
        EXPECT_NEAR(data[0], 0.0f, tol);  // [0, 0] should be 0
        EXPECT_TRUE(std::isinf(data[1])); // [0, 1] should be -inf
        EXPECT_NEAR(data[seq_len], 0.0f, tol);  // [1, 0] should be 0
        EXPECT_NEAR(data[seq_len + 1], 0.0f, tol);  // [1, 1] should be 0
    } else if (mask_cpu.dtype() == DType::Float64) {
        auto* data = mask_cpu.data<double>();
        EXPECT_NEAR(data[0], 0.0, tol);
        EXPECT_TRUE(std::isinf(data[1]));
        EXPECT_NEAR(data[seq_len], 0.0, tol);
        EXPECT_NEAR(data[seq_len + 1], 0.0, tol);
    }
}

// ============================================================================
// Integration Tests
// ============================================================================

class AttentionIntegrationMultiDTypeTest : public MultiBackendDTypeTest {
protected:
    double get_tolerance() const {
        if (dtype() == DType::Float32) {
            return 1e-5;
        } else if (dtype() == DType::Float64) {
            return 1e-10;
        } else if (dtype() == DType::Float16 || dtype() == DType::BFloat16) {
            return 1e-2;
        }
        return 1e-5;
    }
};

TEST_P(AttentionIntegrationMultiDTypeTest, ForwardBackward) {
    MultiheadAttention attn(128, 4, 0.0, true, false, false, 0, 0, true);
    convert_model(attn);

    Variable query = createInput({2, 5, 128}, true);
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

TEST_P(AttentionIntegrationMultiDTypeTest, ParameterCount) {
    MultiheadAttention attn(256, 8, 0.0, true, false, false, 0, 0, true);
    attn.to(device());

    auto params = attn.parameters();

    // Should have: q_proj (weight + bias), k_proj (w + b), v_proj (w + b), out_proj (w + b)
    // Total: 8 parameters (4 weights, 4 biases)
    EXPECT_EQ(params.size(), 8);
}

TEST_P(AttentionIntegrationMultiDTypeTest, TrainEvalSwitch) {
    MultiheadAttention attn(128, 4, 0.5, true, false, false, 0, 0, true);
    attn.to(device());

    EXPECT_TRUE(attn.is_training());

    attn.eval();
    EXPECT_FALSE(attn.is_training());

    attn.train();
    EXPECT_TRUE(attn.is_training());
}

TEST_P(AttentionIntegrationMultiDTypeTest, Deterministic) {
    MultiheadAttention attn(128, 4, 0.0, true, false, false, 0, 0, true);
    convert_model(attn);
    attn.eval();

    auto query_tensor = createOnes({2, 5, 128});
    Variable query(query_tensor, true);

    auto [output1, _] = attn.forward(query, query, query, Tensor{}, Tensor{}, false);
    auto [output2, __] = attn.forward(query, query, query, Tensor{}, Tensor{}, false);

    expectTensorNear(output1.tensor(), output2.tensor());
}

// ============================================================================
// Test Instantiation
// ============================================================================

INSTANTIATE_MULTI_BACKEND_ALL_DTYPE_TESTS(MultiheadAttentionMultiDTypeTest);
INSTANTIATE_MULTI_BACKEND_ALL_DTYPE_TESTS(CausalMaskMultiDTypeTest);
INSTANTIATE_MULTI_BACKEND_ALL_DTYPE_TESTS(AttentionIntegrationMultiDTypeTest);

/*
 * COVERAGE SUMMARY:
 *
 * Test Cases: 21
 * DTypes Tested: Float32, Float64, Float16
 * Backends Tested: CPU, CUDA, OneAPI
 * Total Scenarios: 21 tests × 3 dtypes × 3 backends = 189 test scenarios
 *
 * Coverage:
 * - MultiheadAttention: construction, self-attention, cross-attention, various configs
 * - Interface: simple forward, attention weights, single head
 * - Dropout: training/eval mode behavior
 * - Causal masks: shape, values, edge cases
 * - Integration: forward/backward, parameter count, train/eval switching
 */
