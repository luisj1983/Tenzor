/**
 * @file test_roberta_multidtype.cpp
 * @brief Multi-backend and multi-dtype tests for RoBERTa model family
 *
 * Tests RoBERTa architecture across CPU, CUDA, OneAPI backends
 * with Float32, Float64, and Float16 data types.
 *
 * RoBERTa (Robustly Optimized BERT Approach) is an optimized BERT variant with:
 * - Byte-level BPE vocabulary (50265 tokens)
 * - No NSP task (type_vocab_size = 1)
 * - Different layer normalization epsilon (1e-5)
 * - Dynamic masking during training
 */

#include <gtest/gtest.h>
#include "../multi_backend_dtype_fixture.hpp"
#include "tenzor/models/roberta.hpp"
#include "tenzor/ops/reduction.hpp"
#include "tenzor/ops/math.hpp"
#include <cmath>

using namespace tenzor;
using namespace tenzor::models;
using namespace tenzor::testing;

// ============================================================================
// RoBERTa Multi-Backend Multi-DType Test Fixture
// ============================================================================

class RobertaMultiDTypeTest : public MultiBackendDTypeTest {
protected:
    RobertaConfig config_;
    int64_t batch_size_;
    int64_t seq_len_;

    void SetUp() override {
        MultiBackendDTypeTest::SetUp();

        // Base config for most tests - smaller for faster tests
        config_ = RobertaConfig();
        config_.vocab_size = 1000;  // Reduced for testing
        config_.hidden_size = get_hidden_size();
        config_.num_hidden_layers = get_num_layers();
        config_.num_attention_heads = get_num_heads();
        config_.intermediate_size = get_intermediate_size();
        config_.max_position_embeddings = 64;
        config_.type_vocab_size = 1;  // RoBERTa doesn't use segment embeddings
        config_.hidden_dropout_prob = 0.0;  // Disable dropout for testing
        config_.attention_probs_dropout_prob = 0.0;
        config_.layer_norm_eps = 1e-5;  // Different from BERT

        batch_size_ = 2;
        seq_len_ = 16;
    }

    // Adjust model size based on dtype to manage memory
    int64_t get_hidden_size() const {
        if (dtype() == DType::Float16) return 64;  // Smaller for Float16
        return 128;
    }

    int64_t get_num_layers() const {
        if (dtype() == DType::Float16) return 1;   // Fewer layers for Float16
        return 2;
    }

    int64_t get_num_heads() const {
        if (dtype() == DType::Float16) return 2;
        return 4;
    }

    int64_t get_intermediate_size() const {
        if (dtype() == DType::Float16) return 256;
        return 512;
    }

    // Helper: Create input token IDs (always Int64)
    Variable create_input_ids() {
        // Create on CPU first, fill data, then transfer to device
        Tensor input_ids_cpu({batch_size_, seq_len_}, DType::Int64, Device::cpu());

        auto data = input_ids_cpu.data<int64_t>();
        for (int64_t i = 0; i < input_ids_cpu.numel(); ++i) {
            data[i] = i % config_.vocab_size;
        }

        Tensor input_ids = (device() == Device::cpu())
            ? input_ids_cpu
            : input_ids_cpu.to(device());

        return Variable(input_ids, true);
    }

    // Helper: Create attention mask
    Tensor create_attention_mask(bool with_padding = false) {
        // Create on CPU first, fill data, then transfer to device
        Tensor mask_cpu({batch_size_, seq_len_}, dtype(), Device::cpu());

        // Use template lambda to handle different dtypes
        auto fill_mask = [&]<typename T>(T* ptr) {
            for (int64_t b = 0; b < batch_size_; ++b) {
                for (int64_t i = 0; i < seq_len_; ++i) {
                    // If with_padding, mask last 3 tokens as padding (0)
                    if (with_padding && i >= seq_len_ - 3) {
                        ptr[b * seq_len_ + i] = static_cast<T>(0.0f);
                    } else {
                        ptr[b * seq_len_ + i] = static_cast<T>(1.0f);
                    }
                }
            }
        };

        if (dtype() == DType::Float32) {
            fill_mask(mask_cpu.data<float>());
        } else if (dtype() == DType::Float64) {
            fill_mask(mask_cpu.data<double>());
        } else if (dtype() == DType::Float16) {
            fill_mask(mask_cpu.data<Float16>());
        }

        return (device() == Device::cpu())
            ? mask_cpu
            : mask_cpu.to(device());
    }

    // Helper: Create token type IDs (always Int64, always zeros for RoBERTa)
    Variable create_token_type_ids() {
        Tensor type_ids({batch_size_, seq_len_}, DType::Int64, device());
        type_ids.zero_();  // RoBERTa uses all zeros
        return Variable(type_ids, false);
    }

    // Helper: Create hidden states tensor
    Variable create_hidden_states() {
        Tensor hidden_states({batch_size_, seq_len_, config_.hidden_size}, dtype(), device());
        hidden_states.fill_(0.1f);
        return Variable(hidden_states, true);
    }
};

// ============================================================================
// RobertaConfig Tests
// ============================================================================

TEST_P(RobertaMultiDTypeTest, ConfigBase) {
    auto config = RobertaConfig::base();

    EXPECT_EQ(config.vocab_size, 50265);  // Byte-level BPE
    EXPECT_EQ(config.hidden_size, 768);
    EXPECT_EQ(config.num_hidden_layers, 12);
    EXPECT_EQ(config.num_attention_heads, 12);
    EXPECT_EQ(config.intermediate_size, 3072);
    EXPECT_EQ(config.type_vocab_size, 1);  // No NSP task
    EXPECT_DOUBLE_EQ(config.layer_norm_eps, 1e-5);  // Different from BERT
    EXPECT_EQ(config.max_position_embeddings, 514);
}

TEST_P(RobertaMultiDTypeTest, ConfigLarge) {
    auto config = RobertaConfig::large();

    EXPECT_EQ(config.vocab_size, 50265);
    EXPECT_EQ(config.hidden_size, 1024);
    EXPECT_EQ(config.num_hidden_layers, 24);
    EXPECT_EQ(config.num_attention_heads, 16);
    EXPECT_EQ(config.intermediate_size, 4096);
    EXPECT_EQ(config.type_vocab_size, 1);
}

TEST_P(RobertaMultiDTypeTest, ConfigToBertConfig) {
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

TEST_P(RobertaMultiDTypeTest, ModelCreation) {
    EXPECT_NO_THROW({
        RobertaModel model(config_);
    });
}

TEST_P(RobertaMultiDTypeTest, ModelForwardBase) {
    RobertaModel model(config_);
    convert_model(model);

    auto input_ids = create_input_ids();

    // Forward pass
    auto outputs = model.forward(input_ids, Tensor{}, Variable{}, Variable{});

    // Check output shapes
    expectShape(outputs.sequence_output.tensor(), {batch_size_, seq_len_, config_.hidden_size});
    expectShape(outputs.pooled_output.tensor(), {batch_size_, config_.hidden_size});
}

TEST_P(RobertaMultiDTypeTest, ModelForwardShortSequence) {
    RobertaModel model(config_);
    convert_model(model);

    // Create shorter sequence (8 tokens) on CPU first, then transfer
    int64_t short_seq_len = 8;
    Tensor input_tensor_cpu({batch_size_, short_seq_len}, DType::Int64, Device::cpu());
    auto data = input_tensor_cpu.data<int64_t>();
    for (int64_t i = 0; i < input_tensor_cpu.numel(); ++i) {
        data[i] = i % config_.vocab_size;
    }
    Tensor input_tensor = (device() == Device::cpu())
        ? input_tensor_cpu
        : input_tensor_cpu.to(device());
    Variable input_ids(input_tensor, false);

    auto outputs = model.forward(input_ids, Tensor{}, Variable{}, Variable{});

    // Check output shapes match short sequence length
    expectShape(outputs.sequence_output.tensor(), {batch_size_, short_seq_len, config_.hidden_size});
}

TEST_P(RobertaMultiDTypeTest, ModelForwardLongSequence) {
    RobertaModel model(config_);
    convert_model(model);

    // Create longer sequence (32 tokens) on CPU first, then transfer
    int64_t long_seq_len = 32;
    Tensor input_tensor_cpu({batch_size_, long_seq_len}, DType::Int64, Device::cpu());
    auto data = input_tensor_cpu.data<int64_t>();
    for (int64_t i = 0; i < input_tensor_cpu.numel(); ++i) {
        data[i] = i % config_.vocab_size;
    }
    Tensor input_tensor = (device() == Device::cpu())
        ? input_tensor_cpu
        : input_tensor_cpu.to(device());
    Variable input_ids(input_tensor, false);

    auto outputs = model.forward(input_ids, Tensor{}, Variable{}, Variable{});

    // Check output shapes match long sequence length
    expectShape(outputs.sequence_output.tensor(), {batch_size_, long_seq_len, config_.hidden_size});
}

TEST_P(RobertaMultiDTypeTest, ModelForwardWithAttentionMask) {
    RobertaModel model(config_);
    convert_model(model);

    auto input_ids = create_input_ids();
    auto mask = create_attention_mask(true);  // With padding

    auto outputs = model.forward(input_ids, mask, Variable{}, Variable{});

    // Check output shapes
    expectShape(outputs.sequence_output.tensor(), {batch_size_, seq_len_, config_.hidden_size});
}

TEST_P(RobertaMultiDTypeTest, NoSegmentEmbeddings) {
    // RoBERTa should work fine without explicit token_type_ids
    RobertaModel model(config_);
    convert_model(model);

    auto input_ids = create_input_ids();

    // Forward without token_type_ids (should default to all zeros)
    auto outputs1 = model.forward(input_ids, Tensor{}, Variable{}, Variable{});

    // Forward with explicit zeros
    auto token_type_ids = create_token_type_ids();
    auto outputs2 = model.forward(input_ids, Tensor{}, token_type_ids);

    // Both should produce same output shapes
    auto shape1 = outputs1.sequence_output.tensor().shape();
    auto shape2 = outputs2.sequence_output.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape1.begin(), shape1.end()),
              std::vector<int64_t>(shape2.begin(), shape2.end()));
}

// ============================================================================
// RobertaForSequenceClassification Tests
// ============================================================================

TEST_P(RobertaMultiDTypeTest, SequenceClassificationCreation) {
    int64_t num_labels = 3;

    EXPECT_NO_THROW({
        RobertaForSequenceClassification model(config_, num_labels);
    });
}

TEST_P(RobertaMultiDTypeTest, SequenceClassificationForward) {
    int64_t num_labels = 3;
    RobertaForSequenceClassification model(config_, num_labels);
    convert_model(model);

    auto input_ids = create_input_ids();

    auto logits = model.forward(input_ids, Tensor{}, Variable{});

    // Check output shape: [batch_size, num_labels]
    expectShape(logits.tensor(), {batch_size_, num_labels});
}

TEST_P(RobertaMultiDTypeTest, SequenceClassificationBinaryTask) {
    int64_t num_labels = 2;  // Binary classification
    RobertaForSequenceClassification model(config_, num_labels);
    convert_model(model);

    auto input_ids = create_input_ids();

    auto logits = model.forward(input_ids, Tensor{}, Variable{});

    expectShape(logits.tensor(), {batch_size_, num_labels});
}

TEST_P(RobertaMultiDTypeTest, SequenceClassificationMultiClass) {
    int64_t num_labels = 10;  // Multi-class classification
    RobertaForSequenceClassification model(config_, num_labels);
    convert_model(model);

    auto input_ids = create_input_ids();

    auto logits = model.forward(input_ids, Tensor{}, Variable{});

    expectShape(logits.tensor(), {batch_size_, num_labels});
}

TEST_P(RobertaMultiDTypeTest, SequenceClassificationWithMask) {
    int64_t num_labels = 3;
    RobertaForSequenceClassification model(config_, num_labels);
    convert_model(model);

    auto input_ids = create_input_ids();
    auto mask = create_attention_mask(true);

    auto logits = model.forward(input_ids, mask, Variable{});

    expectShape(logits.tensor(), {batch_size_, num_labels});
}

// ============================================================================
// RobertaForTokenClassification Tests
// ============================================================================

TEST_P(RobertaMultiDTypeTest, TokenClassificationCreation) {
    int64_t num_labels = 9;  // NER tags

    EXPECT_NO_THROW({
        RobertaForTokenClassification model(config_, num_labels);
    });
}

TEST_P(RobertaMultiDTypeTest, TokenClassificationForward) {
    int64_t num_labels = 9;
    RobertaForTokenClassification model(config_, num_labels);
    convert_model(model);

    auto input_ids = create_input_ids();

    auto logits = model.forward(input_ids, Tensor{}, Variable{});

    // Check output shape: [batch_size, seq_len, num_labels]
    expectShape(logits.tensor(), {batch_size_, seq_len_, num_labels});
}

TEST_P(RobertaMultiDTypeTest, TokenClassificationNERTask) {
    int64_t num_labels = 9;  // B-PER, I-PER, B-ORG, I-ORG, B-LOC, I-LOC, B-MISC, I-MISC, O
    RobertaForTokenClassification model(config_, num_labels);
    convert_model(model);

    auto input_ids = create_input_ids();

    auto logits = model.forward(input_ids, Tensor{}, Variable{});

    // Verify shape for token-level predictions
    expectShape(logits.tensor(), {batch_size_, seq_len_, num_labels});
}

TEST_P(RobertaMultiDTypeTest, TokenClassificationPOSTagging) {
    int64_t num_labels = 17;  // Penn Treebank POS tags
    RobertaForTokenClassification model(config_, num_labels);
    convert_model(model);

    auto input_ids = create_input_ids();

    auto logits = model.forward(input_ids, Tensor{}, Variable{});

    expectShape(logits.tensor(), {batch_size_, seq_len_, num_labels});
}

TEST_P(RobertaMultiDTypeTest, TokenClassificationWithMask) {
    int64_t num_labels = 9;
    RobertaForTokenClassification model(config_, num_labels);
    convert_model(model);

    auto input_ids = create_input_ids();
    auto mask = create_attention_mask(true);

    auto logits = model.forward(input_ids, mask, Variable{});

    expectShape(logits.tensor(), {batch_size_, seq_len_, num_labels});
}

// ============================================================================
// RobertaForQuestionAnswering Tests
// ============================================================================

TEST_P(RobertaMultiDTypeTest, QuestionAnsweringCreation) {
    EXPECT_NO_THROW({
        RobertaForQuestionAnswering model(config_);
    });
}

TEST_P(RobertaMultiDTypeTest, QuestionAnsweringForward) {
    RobertaForQuestionAnswering model(config_);
    convert_model(model);

    auto input_ids = create_input_ids();

    auto outputs = model.forward(input_ids, Tensor{}, Variable{});

    // Check output shapes for span prediction
    expectShape(outputs.start_logits.tensor(), {batch_size_, seq_len_});
    expectShape(outputs.end_logits.tensor(), {batch_size_, seq_len_});
}

TEST_P(RobertaMultiDTypeTest, QuestionAnsweringSpanPrediction) {
    RobertaForQuestionAnswering model(config_);
    convert_model(model);

    auto input_ids = create_input_ids();

    auto outputs = model.forward(input_ids, Tensor{}, Variable{});

    // Both start and end logits should have same shape
    EXPECT_EQ(outputs.start_logits.tensor().shape()[0], outputs.end_logits.tensor().shape()[0]);
    EXPECT_EQ(outputs.start_logits.tensor().shape()[1], outputs.end_logits.tensor().shape()[1]);

    // Shape should be [batch_size, seq_len]
    EXPECT_EQ(outputs.start_logits.tensor().ndim(), 2);
    EXPECT_EQ(outputs.end_logits.tensor().ndim(), 2);
}

TEST_P(RobertaMultiDTypeTest, QuestionAnsweringWithMask) {
    RobertaForQuestionAnswering model(config_);
    convert_model(model);

    auto input_ids = create_input_ids();
    auto mask = create_attention_mask(true);

    auto outputs = model.forward(input_ids, mask, Variable{});

    expectShape(outputs.start_logits.tensor(), {batch_size_, seq_len_});
    expectShape(outputs.end_logits.tensor(), {batch_size_, seq_len_});
}

// ============================================================================
// Gradient Flow Tests
// ============================================================================

TEST_P(RobertaMultiDTypeTest, GradientFlowSequenceClassification) {
    int64_t num_labels = 2;
    RobertaForSequenceClassification model(config_, num_labels);
    convert_model(model);
    model.train();

    auto input_ids = create_input_ids();

    // Forward pass
    auto logits = model.forward(input_ids, Tensor{}, Variable{});

    EXPECT_TRUE(logits.requires_grad()) << "Logits should require gradients";

    // Simple loss: sum of all outputs
    auto loss = sum(logits);

    // Backward pass
    EXPECT_NO_THROW({
        loss.backward();
    }) << "Backward pass should complete without errors";

    // Check model parameters have gradients
    auto params = model.parameters();
    EXPECT_FALSE(params.empty()) << "Model should have parameters";

    int params_with_grad = 0;
    for (const auto& param : params) {
        if (param->has_grad()) {
            params_with_grad++;
        }
    }
    EXPECT_GT(params_with_grad, 0) << "At least some parameters should have gradients";
}

TEST_P(RobertaMultiDTypeTest, GradientFlowTokenClassification) {
    int64_t num_labels = 9;
    RobertaForTokenClassification model(config_, num_labels);
    convert_model(model);
    model.train();

    auto input_ids = create_input_ids();

    // Forward pass
    auto logits = model.forward(input_ids, Tensor{}, Variable{});

    EXPECT_TRUE(logits.requires_grad());

    // Loss: mean of all outputs
    auto loss = mean(logits);

    // Backward pass
    EXPECT_NO_THROW({
        loss.backward();
    });

    // Check gradients exist
    auto params = model.parameters();
    int params_with_grad = 0;
    for (const auto& param : params) {
        if (param->has_grad()) {
            params_with_grad++;
        }
    }
    EXPECT_GT(params_with_grad, 0);
}

TEST_P(RobertaMultiDTypeTest, GradientFlowQuestionAnswering) {
    RobertaForQuestionAnswering model(config_);
    convert_model(model);
    model.train();

    auto input_ids = create_input_ids();

    // Forward pass
    auto outputs = model.forward(input_ids, Tensor{}, Variable{});

    EXPECT_TRUE(outputs.start_logits.requires_grad());
    EXPECT_TRUE(outputs.end_logits.requires_grad());

    // Combined loss from both start and end logits
    auto loss = sum(outputs.start_logits) + sum(outputs.end_logits);

    // Backward pass
    EXPECT_NO_THROW({
        loss.backward();
    });

    // Check gradients
    auto params = model.parameters();
    int params_with_grad = 0;
    for (const auto& param : params) {
        if (param->has_grad()) {
            params_with_grad++;
        }
    }
    EXPECT_GT(params_with_grad, 0);
}

TEST_P(RobertaMultiDTypeTest, GradientFlowWithAttentionMask) {
    int64_t num_labels = 2;
    RobertaForSequenceClassification model(config_, num_labels);
    convert_model(model);
    model.train();

    auto input_ids = create_input_ids();
    auto mask = create_attention_mask(true);

    // Forward with mask
    auto logits = model.forward(input_ids, mask, Variable{});

    auto loss = sum(logits);

    // Backward should work with masked attention
    EXPECT_NO_THROW({
        loss.backward();
    });

    auto params = model.parameters();
    int params_with_grad = 0;
    for (const auto& param : params) {
        if (param->has_grad()) {
            params_with_grad++;
        }
    }
    EXPECT_GT(params_with_grad, 0);
}

// ============================================================================
// Task-Specific Head Tests
// ============================================================================

TEST_P(RobertaMultiDTypeTest, ClassificationHeadStructure) {
    int64_t num_labels = 5;
    RobertaForSequenceClassification model(config_, num_labels);
    convert_model(model);

    // Model should have base RoBERTa + classification head
    auto params = model.parameters();
    EXPECT_FALSE(params.empty()) << "Model should have parameters";

    // Forward to verify head produces correct output dimension
    auto input_ids = create_input_ids();
    auto logits = model.forward(input_ids, Tensor{}, Variable{});

    EXPECT_EQ(logits.tensor().shape()[1], num_labels) << "Classification head should output num_labels dimensions";
}

TEST_P(RobertaMultiDTypeTest, TokenClassificationHeadStructure) {
    int64_t num_labels = 7;
    RobertaForTokenClassification model(config_, num_labels);
    convert_model(model);

    auto params = model.parameters();
    EXPECT_FALSE(params.empty());

    // Forward to verify head produces correct output dimensions
    auto input_ids = create_input_ids();
    auto logits = model.forward(input_ids, Tensor{}, Variable{});

    EXPECT_EQ(logits.tensor().shape()[2], num_labels) << "Token classification head should output num_labels per token";
}

TEST_P(RobertaMultiDTypeTest, QuestionAnsweringHeadStructure) {
    RobertaForQuestionAnswering model(config_);
    convert_model(model);

    auto params = model.parameters();
    EXPECT_FALSE(params.empty());

    // Forward to verify head produces start and end logits
    auto input_ids = create_input_ids();
    auto outputs = model.forward(input_ids, Tensor{}, Variable{});

    // Both logits should span the sequence length
    EXPECT_EQ(outputs.start_logits.tensor().shape()[1], seq_len_);
    EXPECT_EQ(outputs.end_logits.tensor().shape()[1], seq_len_);
}

// ============================================================================
// Edge Cases and Robustness Tests
// ============================================================================

TEST_P(RobertaMultiDTypeTest, SingleTokenSequence) {
    RobertaModel model(config_);
    convert_model(model);

    // Create single token input
    Tensor input_tensor({batch_size_, 1}, DType::Int64, device());
    input_tensor.fill_(42);
    Variable input_ids(input_tensor, false);

    EXPECT_NO_THROW({
        auto outputs = model.forward(input_ids, Tensor{}, Variable{}, Variable{});
        EXPECT_EQ(outputs.sequence_output.tensor().shape()[1], 1);
    });
}

TEST_P(RobertaMultiDTypeTest, MaxSequenceLength) {
    // Test with maximum supported sequence length
    RobertaModel model(config_);
    convert_model(model);

    int64_t max_seq = config_.max_position_embeddings - 2;  // Account for special tokens
    Tensor input_tensor({1, max_seq}, DType::Int64, device());
    input_tensor.fill_(42);
    Variable input_ids(input_tensor, false);

    EXPECT_NO_THROW({
        auto outputs = model.forward(input_ids, Tensor{}, Variable{}, Variable{});
        EXPECT_EQ(outputs.sequence_output.tensor().shape()[1], max_seq);
    });
}

TEST_P(RobertaMultiDTypeTest, LargeBatchSize) {
    if (dtype() == DType::Float16) {
        GTEST_SKIP() << "Skipping large batch test for Float16 to avoid memory issues";
    }

    RobertaForSequenceClassification model(config_, 2);
    convert_model(model);

    // Large batch size
    int64_t large_batch = 16;
    Tensor input_tensor({large_batch, seq_len_}, DType::Int64, device());
    input_tensor.fill_(42);
    Variable input_ids(input_tensor, false);

    EXPECT_NO_THROW({
        auto logits = model.forward(input_ids, Tensor{}, Variable{});
        EXPECT_EQ(logits.tensor().shape()[0], large_batch);
    });
}

TEST_P(RobertaMultiDTypeTest, AllMaskedSequence) {
    RobertaModel model(config_);
    convert_model(model);
    model.eval();

    auto input_ids = create_input_ids();

    // Create mask with all zeros (all masked)
    Tensor mask({batch_size_, seq_len_}, dtype(), device());
    mask.zero_();

    // Should still run without errors
    EXPECT_NO_THROW({
        auto outputs = model.forward(input_ids, mask, Variable{}, Variable{});
        EXPECT_EQ(outputs.sequence_output.tensor().shape()[0], batch_size_);
    });
}

// ============================================================================
// Precision-Specific Tests
// ============================================================================

TEST_P(RobertaMultiDTypeTest, OutputNumericalStability) {
    RobertaModel model(config_);
    convert_model(model);
    model.eval();

    auto input_ids = create_input_ids();

    auto outputs = model.forward(input_ids, Tensor{}, Variable{}, Variable{});

    // Check outputs are not NaN or Inf
    float max_value = compute_max_abs(outputs.sequence_output.tensor());

    EXPECT_FALSE(std::isnan(max_value)) << "Output contains NaN values";
    EXPECT_FALSE(std::isinf(max_value)) << "Output contains Inf values";
    EXPECT_GT(max_value, 0.0f) << "Output should contain non-zero values";
}

TEST_P(RobertaMultiDTypeTest, DtypeConsistency) {
    RobertaForSequenceClassification model(config_, 3);
    convert_model(model);

    auto input_ids = create_input_ids();

    auto logits = model.forward(input_ids, Tensor{}, Variable{});

    // Output dtype should be preserved
    expectShape(logits.tensor(), {batch_size_, 3});
    expectDType(logits.tensor());
}

// ============================================================================
// Instantiate Tests for All Backends and DTypes
// ============================================================================

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(RobertaMultiDTypeTest);

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
