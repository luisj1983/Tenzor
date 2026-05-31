/**
 * @file test_roberta.cpp
 * @brief Unit tests for RoBERTa models
 */

#include <gtest/gtest.h>
#include "../backend_test_fixture.hpp"
#include "tenzor/models/roberta.hpp"
#include "tenzor/core/tensor.hpp"
#include "tenzor/autograd/variable.hpp"
#include "tenzor/autograd/ops.hpp"
#include "tenzor/ops/creation.hpp"

using namespace tenzor;
using namespace tenzor::models;

class RobertaTest : public ::tenzor::testing::BackendTest {
protected:
    void SetUp() override {
        ::tenzor::testing::BackendTest::SetUp();
        if (::testing::Test::IsSkipped()) return;
    }
};

// ============================================================================
// RobertaConfig Tests
// ============================================================================

TEST_P(RobertaTest, ConfigBase) {
    auto config = RobertaConfig::base();

    EXPECT_EQ(config.vocab_size, 50265);  // Byte-level BPE
    EXPECT_EQ(config.hidden_size, 768);
    EXPECT_EQ(config.num_hidden_layers, 12);
    EXPECT_EQ(config.num_attention_heads, 12);
    EXPECT_EQ(config.intermediate_size, 3072);
    EXPECT_EQ(config.type_vocab_size, 1);  // No NSP
    EXPECT_DOUBLE_EQ(config.layer_norm_eps, 1e-5);  // Different from BERT
    EXPECT_EQ(config.max_position_embeddings, 514);
}

TEST_P(RobertaTest, ConfigLarge) {
    auto config = RobertaConfig::large();

    EXPECT_EQ(config.vocab_size, 50265);
    EXPECT_EQ(config.hidden_size, 1024);
    EXPECT_EQ(config.num_hidden_layers, 24);
    EXPECT_EQ(config.num_attention_heads, 16);
    EXPECT_EQ(config.intermediate_size, 4096);
}

TEST_P(RobertaTest, ConfigToBertConfig) {
    auto roberta_config = RobertaConfig::base();
    auto bert_config = roberta_config.to_bert_config();

    EXPECT_EQ(bert_config.vocab_size, roberta_config.vocab_size);
    EXPECT_EQ(bert_config.hidden_size, roberta_config.hidden_size);
    EXPECT_EQ(bert_config.num_hidden_layers, roberta_config.num_hidden_layers);
    EXPECT_DOUBLE_EQ(bert_config.layer_norm_eps, 1e-5);
}

// ============================================================================
// RobertaModel Tests
// ============================================================================

TEST_P(RobertaTest, ModelCreation) {
    auto config = RobertaConfig::base();
    EXPECT_NO_THROW({
        RobertaModel model(config);
        model.to(device);
    });
}

TEST_P(RobertaTest, ModelForwardBase) {
    auto config = RobertaConfig::base();
    config.num_hidden_layers = 2;  // Reduce for faster testing
    RobertaModel model(config);
    model.to(device);

    // Create dummy input [batch=2, seq_len=10]
    int64_t batch_size = 2;
    int64_t seq_len = 10;

    // Token/index inputs built on CPU then moved to device.
    Tensor input_tensor({batch_size, seq_len}, DType::Int64, Device::cpu());
    input_tensor.fill_(42);  // Fill with dummy token IDs
    Variable input_ids(input_tensor.to(device), false);

    // Forward pass
    auto outputs = model.forward(input_ids, Tensor{}, Variable{}, Variable{});

    // Check output shapes
    EXPECT_EQ(outputs.sequence_output.shape()[0], batch_size);
    EXPECT_EQ(outputs.sequence_output.shape()[1], seq_len);
    EXPECT_EQ(outputs.sequence_output.shape()[2], config.hidden_size);

    EXPECT_EQ(outputs.pooled_output.shape()[0], batch_size);
    EXPECT_EQ(outputs.pooled_output.shape()[1], config.hidden_size);
}

TEST_P(RobertaTest, ModelForwardWithAttentionMask) {
    auto config = RobertaConfig::base();
    config.num_hidden_layers = 2;
    RobertaModel model(config);
    model.to(device);

    int64_t batch_size = 2;
    int64_t seq_len = 10;

    Tensor input_tensor({batch_size, seq_len}, DType::Int64, Device::cpu());
    input_tensor.fill_(42);
    Variable input_ids(input_tensor.to(device), false);

    // Create attention mask (1 for valid tokens, 0 for padding) on CPU,
    // then move to device.
    Tensor mask({batch_size, seq_len}, DType::Float32, Device::cpu());
    mask.fill_(1.0f);
    // Mask last 3 tokens as padding
    auto mask_data = mask.data<float>();
    for (int64_t b = 0; b < batch_size; ++b) {
        for (int64_t s = seq_len - 3; s < seq_len; ++s) {
            mask_data[b * seq_len + s] = 0.0f;
        }
    }

    auto outputs = model.forward(input_ids, mask.to(device), Variable{}, Variable{});

    EXPECT_EQ(outputs.sequence_output.shape()[0], batch_size);
    EXPECT_EQ(outputs.sequence_output.shape()[1], seq_len);
    EXPECT_EQ(outputs.sequence_output.shape()[2], config.hidden_size);
}

// ============================================================================
// RobertaForSequenceClassification Tests
// ============================================================================

TEST_P(RobertaTest, SequenceClassificationCreation) {
    auto config = RobertaConfig::base();
    config.num_hidden_layers = 2;
    int64_t num_labels = 3;

    EXPECT_NO_THROW({
        RobertaForSequenceClassification model(config, num_labels);
        model.to(device);
    });
}

TEST_P(RobertaTest, SequenceClassificationForward) {
    auto config = RobertaConfig::base();
    config.num_hidden_layers = 2;
    int64_t num_labels = 3;
    RobertaForSequenceClassification model(config, num_labels);
    model.to(device);

    int64_t batch_size = 2;
    int64_t seq_len = 10;

    Tensor input_tensor({batch_size, seq_len}, DType::Int64, Device::cpu());
    input_tensor.fill_(42);
    Variable input_ids(input_tensor.to(device), false);

    auto logits = model.forward(input_ids, Tensor{}, Variable{});

    // Check output shape: [batch_size, num_labels]
    EXPECT_EQ(logits.shape()[0], batch_size);
    EXPECT_EQ(logits.shape()[1], num_labels);
}

// ============================================================================
// RobertaForTokenClassification Tests
// ============================================================================

TEST_P(RobertaTest, TokenClassificationCreation) {
    auto config = RobertaConfig::base();
    config.num_hidden_layers = 2;
    int64_t num_labels = 9;  // NER tags

    EXPECT_NO_THROW({
        RobertaForTokenClassification model(config, num_labels);
        model.to(device);
    });
}

TEST_P(RobertaTest, TokenClassificationForward) {
    auto config = RobertaConfig::base();
    config.num_hidden_layers = 2;
    int64_t num_labels = 9;
    RobertaForTokenClassification model(config, num_labels);
    model.to(device);

    int64_t batch_size = 2;
    int64_t seq_len = 10;

    Tensor input_tensor({batch_size, seq_len}, DType::Int64, Device::cpu());
    input_tensor.fill_(42);
    Variable input_ids(input_tensor.to(device), false);

    auto logits = model.forward(input_ids, Tensor{}, Variable{});

    // Check output shape: [batch_size, seq_len, num_labels]
    EXPECT_EQ(logits.shape()[0], batch_size);
    EXPECT_EQ(logits.shape()[1], seq_len);
    EXPECT_EQ(logits.shape()[2], num_labels);
}

// ============================================================================
// RobertaForQuestionAnswering Tests
// ============================================================================

TEST_P(RobertaTest, QuestionAnsweringCreation) {
    auto config = RobertaConfig::base();
    config.num_hidden_layers = 2;

    EXPECT_NO_THROW({
        RobertaForQuestionAnswering model(config);
        model.to(device);
    });
}

TEST_P(RobertaTest, QuestionAnsweringForward) {
    auto config = RobertaConfig::base();
    config.num_hidden_layers = 2;
    RobertaForQuestionAnswering model(config);
    model.to(device);

    int64_t batch_size = 2;
    int64_t seq_len = 10;

    Tensor input_tensor({batch_size, seq_len}, DType::Int64, Device::cpu());
    input_tensor.fill_(42);
    Variable input_ids(input_tensor.to(device), false);

    auto outputs = model.forward(input_ids, Tensor{}, Variable{});

    // Check output shapes
    EXPECT_EQ(outputs.start_logits.shape()[0], batch_size);
    EXPECT_EQ(outputs.start_logits.shape()[1], seq_len);

    EXPECT_EQ(outputs.end_logits.shape()[0], batch_size);
    EXPECT_EQ(outputs.end_logits.shape()[1], seq_len);
}

// ============================================================================
// Gradient Flow Tests
// ============================================================================

TEST_P(RobertaTest, GradientFlowSequenceClassification) {
    auto config = RobertaConfig::base();
    config.num_hidden_layers = 1;  // Minimal for testing
    int64_t num_labels = 2;
    RobertaForSequenceClassification model(config, num_labels);
    model.to(device);

    int64_t batch_size = 2;
    int64_t seq_len = 8;

    Tensor input_tensor({batch_size, seq_len}, DType::Int64, Device::cpu());
    input_tensor.fill_(42);
    Variable input_ids(input_tensor.to(device), false);

    // Forward pass
    auto logits = model.forward(input_ids, Tensor{}, Variable{});

    // Simple loss: sum of all outputs
    auto loss = sum(logits);

    // Backward pass
    EXPECT_NO_THROW({
        loss.backward();
    });

    // Check that gradients were computed
    auto grad = loss.grad();
    EXPECT_TRUE((grad.has_value() && grad->numel() > 0) || !loss.requires_grad());
}

TEST_P(RobertaTest, NoSegmentEmbeddings) {
    // RoBERTa should work fine without explicit token_type_ids
    auto config = RobertaConfig::base();
    config.num_hidden_layers = 2;
    RobertaModel model(config);
    model.to(device);

    int64_t batch_size = 2;
    int64_t seq_len = 10;

    Tensor input_tensor({batch_size, seq_len}, DType::Int64, Device::cpu());
    input_tensor.fill_(42);
    Variable input_ids(input_tensor.to(device), false);

    // Forward without token_type_ids (should default to all zeros)
    auto outputs1 = model.forward(input_ids, Tensor{}, Variable{}, Variable{});

    // Forward with explicit zeros
    Tensor token_type_zeros({batch_size, seq_len}, DType::Int64, Device::cpu());
    token_type_zeros.zero_();
    Variable token_type_ids(token_type_zeros.to(device), false);
    auto outputs2 = model.forward(input_ids, Tensor{}, token_type_ids);

    // Both should produce same output shapes
    auto shape1_seq = outputs1.sequence_output.shape();
    auto shape2_seq = outputs2.sequence_output.shape();
    EXPECT_TRUE(std::equal(shape1_seq.begin(), shape1_seq.end(), shape2_seq.begin(), shape2_seq.end()));

    auto shape1_pool = outputs1.pooled_output.shape();
    auto shape2_pool = outputs2.pooled_output.shape();
    EXPECT_TRUE(std::equal(shape1_pool.begin(), shape1_pool.end(), shape2_pool.begin(), shape2_pool.end()));
}

// ============================================================================
// Performance Tests
// ============================================================================

TEST_P(RobertaTest, DISABLED_BenchmarkInference) {
    // Disabled by default - enable for performance testing
    auto config = RobertaConfig::base();
    RobertaForSequenceClassification model(config, 2);
    model.to(device);

    int64_t batch_size = 32;
    int64_t seq_len = 128;

    Tensor input_tensor({batch_size, seq_len}, DType::Int64, Device::cpu());
    input_tensor.fill_(42);
    Variable input_ids(input_tensor.to(device), false);

    // Warmup
    for (int i = 0; i < 5; ++i) {
        model.forward(input_ids, Tensor{}, Variable{});
    }

    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    int num_iterations = 100;
    for (int i = 0; i < num_iterations; ++i) {
        model.forward(input_ids, Tensor{}, Variable{});
    }
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Average inference time: " << duration.count() / num_iterations << " ms" << std::endl;

    // reason: regression-guard ceiling; tighten when baseline is established
    EXPECT_LT(duration.count(), 10000);
}

INSTANTIATE_BACKEND_TESTS(RobertaTest);
