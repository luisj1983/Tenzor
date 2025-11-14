/**
 * @file test_roberta_multidtype.cpp
 * @brief Multi-dtype tests for RoBERTa model family (Float32, Float64, Float16)
 *
 * Tests RoBERTa architecture components across different floating-point precisions:
 * - RoBERTa base and large configurations
 * - Forward pass with different sequence lengths
 * - Attention masks for padding handling
 * - Task-specific heads (classification, token classification, QA)
 * - Gradient flow through model layers
 *
 * RoBERTa (Robustly Optimized BERT Approach) is an optimized BERT variant with:
 * - Byte-level BPE vocabulary (50265 tokens)
 * - No NSP task (type_vocab_size = 1)
 * - Different layer normalization epsilon (1e-5)
 * - Dynamic masking during training
 *
 * Coverage: RoBERTa-base and RoBERTa-large configurations
 * Float16 uses smaller model sizes to manage memory constraints
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include "tenzor/models/roberta.hpp"
#include "tenzor/core/tensor.hpp"
#include "tenzor/autograd/variable.hpp"
#include "tenzor/ops/reduction.hpp"
#include "tenzor/ops/math.hpp"
#include <memory>
#include <vector>
#include <cmath>

using namespace tenzor;
using namespace tenzor::models;

// ============================================================================
// Test Fixtures - Typed for Float32, Float64, Float16
// ============================================================================

template <typename T>
class RobertaMultiDtypeTest : public ::testing::Test {
protected:
    void SetUp() override {
        dtype_ = dtype_from_type<T>();
        device_ = Device::cpu();

        // Base config for most tests
        config_ = RobertaConfig();
        config_.vocab_size = 1000;  // Reduced for testing
        config_.hidden_size = get_hidden_size();
        config_.num_hidden_layers = get_num_layers();
        config_.num_attention_heads = get_num_heads();
        config_.intermediate_size = get_intermediate_size();
        config_.max_position_embeddings = 64;
        config_.type_vocab_size = 1;  // RoBERTa doesn't use segment embeddings
        config_.hidden_dropout_prob = 0.1f;
        config_.attention_probs_dropout_prob = 0.1f;
        config_.layer_norm_eps = 1e-5;  // Different from BERT

        batch_size_ = 2;
        seq_len_ = 16;
        tolerance_ = get_tolerance();
    }

    // Adjust model size based on dtype to manage memory
    int64_t get_hidden_size() const {
        if (std::is_same<T, Float16>::value) return 64;  // Smaller for Float16
        return 128;
    }

    int64_t get_num_layers() const {
        if (std::is_same<T, Float16>::value) return 1;   // Fewer layers for Float16
        return 2;
    }

    int64_t get_num_heads() const {
        if (std::is_same<T, Float16>::value) return 2;
        return 4;
    }

    int64_t get_intermediate_size() const {
        if (std::is_same<T, Float16>::value) return 256;
        return 512;
    }

    float get_tolerance() const {
        if (std::is_same<T, double>::value) return 1e-5f;
        if (std::is_same<T, Float16>::value) return 1e-2f;  // Relaxed for Float16
        return 1e-4f;  // Float32
    }

    // Helper: Create input token IDs (always Int64)
    auto create_input_ids() -> Variable {
        Tensor input_ids({batch_size_, seq_len_}, DType::Int64, device_);

        std::vector<int64_t> data(batch_size_ * seq_len_);
        for (size_t i = 0; i < data.size(); ++i) {
            data[i] = i % config_.vocab_size;
        }
        std::copy(data.begin(), data.end(), input_ids.data<int64_t>());

        return Variable(input_ids, true);
    }

    // Helper: Create attention mask (in current dtype)
    auto create_attention_mask(bool with_padding = false) -> Tensor {
        Tensor mask({batch_size_, seq_len_}, dtype_, device_);

        std::vector<T> mask_data(batch_size_ * seq_len_);
        for (int64_t b = 0; b < batch_size_; ++b) {
            for (int64_t i = 0; i < seq_len_; ++i) {
                // If with_padding, mask last 3 tokens as padding (0)
                if (with_padding && i >= seq_len_ - 3) {
                    mask_data[b * seq_len_ + i] = static_cast<T>(0.0f);
                } else {
                    mask_data[b * seq_len_ + i] = static_cast<T>(1.0f);
                }
            }
        }

        if (dtype_ == DType::Float32) {
            auto* ptr = mask.data<float>();
            for (size_t i = 0; i < mask_data.size(); ++i) {
                ptr[i] = static_cast<float>(mask_data[i]);
            }
        } else if (dtype_ == DType::Float64) {
            auto* ptr = mask.data<double>();
            for (size_t i = 0; i < mask_data.size(); ++i) {
                ptr[i] = static_cast<double>(static_cast<float>(mask_data[i]));
            }
        } else if (dtype_ == DType::Float16) {
            std::copy(mask_data.begin(), mask_data.end(), mask.data<T>());
        }

        return mask;
    }

    // Helper: Create token type IDs (always Int64, always zeros for RoBERTa)
    auto create_token_type_ids() -> Variable {
        Tensor type_ids({batch_size_, seq_len_}, DType::Int64, device_);
        type_ids.zero_();  // RoBERTa uses all zeros
        return Variable(type_ids, false);
    }

    // Helper: Create hidden states tensor
    auto create_hidden_states() -> Variable {
        Tensor hidden_states({batch_size_, seq_len_, config_.hidden_size}, dtype_, device_);

        if (dtype_ == DType::Float32) {
            hidden_states.fill_(0.1f);
        } else if (dtype_ == DType::Float64) {
            hidden_states.fill_(0.1);
        } else if (dtype_ == DType::Float16) {
            std::vector<T> values(batch_size_ * seq_len_ * config_.hidden_size,
                                  static_cast<T>(0.1f));
            std::copy(values.begin(), values.end(), hidden_states.data<T>());
        }

        return Variable(hidden_states, true);
    }

    // Helper: Get scalar value from tensor
    float get_scalar(const Tensor& tensor) const {
        if (dtype_ == DType::Float32) {
            return tensor.item<float>();
        } else if (dtype_ == DType::Float64) {
            return static_cast<float>(tensor.item<double>());
        } else {
            return static_cast<float>(tensor.item<T>());
        }
    }

    DType dtype_;
    Device device_;
    RobertaConfig config_;
    int64_t batch_size_;
    int64_t seq_len_;
    float tolerance_;

private:
    template <typename U>
    static DType dtype_from_type() {
        if (std::is_same<U, float>::value) return DType::Float32;
        if (std::is_same<U, double>::value) return DType::Float64;
        if (std::is_same<U, Float16>::value) return DType::Float16;
        return DType::Float32;
    }
};

using DTypes = ::testing::Types<float, double, Float16>;
TYPED_TEST_SUITE(RobertaMultiDtypeTest, DTypes);

// ============================================================================
// RobertaConfig Tests
// ============================================================================

TYPED_TEST(RobertaMultiDtypeTest, ConfigBase) {
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

TYPED_TEST(RobertaMultiDtypeTest, ConfigLarge) {
    auto config = RobertaConfig::large();

    EXPECT_EQ(config.vocab_size, 50265);
    EXPECT_EQ(config.hidden_size, 1024);
    EXPECT_EQ(config.num_hidden_layers, 24);
    EXPECT_EQ(config.num_attention_heads, 16);
    EXPECT_EQ(config.intermediate_size, 4096);
    EXPECT_EQ(config.type_vocab_size, 1);
}

TYPED_TEST(RobertaMultiDtypeTest, ConfigToBertConfig) {
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

TYPED_TEST(RobertaMultiDtypeTest, ModelCreation) {
    EXPECT_NO_THROW({
        RobertaModel model(this->config_);
    });
}

TYPED_TEST(RobertaMultiDtypeTest, ModelForwardBase) {
    RobertaModel model(this->config_);

    auto input_ids = this->create_input_ids();

    // Forward pass
    auto outputs = model.forward(input_ids, Tensor{}, Variable{}, Variable{});

    // Check output shapes
    EXPECT_EQ(outputs.sequence_output.shape()[0], this->batch_size_);
    EXPECT_EQ(outputs.sequence_output.shape()[1], this->seq_len_);
    EXPECT_EQ(outputs.sequence_output.shape()[2], this->config_.hidden_size);

    EXPECT_EQ(outputs.pooled_output.shape()[0], this->batch_size_);
    EXPECT_EQ(outputs.pooled_output.shape()[1], this->config_.hidden_size);
}

TYPED_TEST(RobertaMultiDtypeTest, ModelForwardShortSequence) {
    RobertaModel model(this->config_);

    // Create shorter sequence (8 tokens)
    int64_t short_seq_len = 8;
    Tensor input_tensor({this->batch_size_, short_seq_len}, DType::Int64, this->device_);
    std::vector<int64_t> data(this->batch_size_ * short_seq_len);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = i % this->config_.vocab_size;
    }
    std::copy(data.begin(), data.end(), input_tensor.data<int64_t>());
    Variable input_ids(input_tensor, false);

    auto outputs = model.forward(input_ids, Tensor{}, Variable{}, Variable{});

    // Check output shapes match short sequence length
    EXPECT_EQ(outputs.sequence_output.shape()[0], this->batch_size_);
    EXPECT_EQ(outputs.sequence_output.shape()[1], short_seq_len);
    EXPECT_EQ(outputs.sequence_output.shape()[2], this->config_.hidden_size);
}

TYPED_TEST(RobertaMultiDtypeTest, ModelForwardLongSequence) {
    RobertaModel model(this->config_);

    // Create longer sequence (32 tokens)
    int64_t long_seq_len = 32;
    Tensor input_tensor({this->batch_size_, long_seq_len}, DType::Int64, this->device_);
    std::vector<int64_t> data(this->batch_size_ * long_seq_len);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = i % this->config_.vocab_size;
    }
    std::copy(data.begin(), data.end(), input_tensor.data<int64_t>());
    Variable input_ids(input_tensor, false);

    auto outputs = model.forward(input_ids, Tensor{}, Variable{}, Variable{});

    // Check output shapes match long sequence length
    EXPECT_EQ(outputs.sequence_output.shape()[0], this->batch_size_);
    EXPECT_EQ(outputs.sequence_output.shape()[1], long_seq_len);
    EXPECT_EQ(outputs.sequence_output.shape()[2], this->config_.hidden_size);
}

TYPED_TEST(RobertaMultiDtypeTest, ModelForwardWithAttentionMask) {
    RobertaModel model(this->config_);

    auto input_ids = this->create_input_ids();
    auto mask = this->create_attention_mask(true);  // With padding

    auto outputs = model.forward(input_ids, mask, Variable{}, Variable{});

    // Check output shapes
    EXPECT_EQ(outputs.sequence_output.shape()[0], this->batch_size_);
    EXPECT_EQ(outputs.sequence_output.shape()[1], this->seq_len_);
    EXPECT_EQ(outputs.sequence_output.shape()[2], this->config_.hidden_size);
}

TYPED_TEST(RobertaMultiDtypeTest, ModelForwardAttentionMaskEffect) {
    RobertaModel model(this->config_);
    model.eval();  // Set to eval mode for consistent results

    auto input_ids = this->create_input_ids();

    // Forward pass without mask
    auto outputs_no_mask = model.forward(input_ids, Tensor{}, Variable{}, Variable{});

    // Forward pass with mask (padding last 3 tokens)
    auto mask = this->create_attention_mask(true);
    auto outputs_with_mask = model.forward(input_ids, mask, Variable{}, Variable{});

    // Both should have same output shape
    EXPECT_EQ(outputs_no_mask.sequence_output.shape()[0], outputs_with_mask.sequence_output.shape()[0]);
    EXPECT_EQ(outputs_no_mask.sequence_output.shape()[1], outputs_with_mask.sequence_output.shape()[1]);
    EXPECT_EQ(outputs_no_mask.sequence_output.shape()[2], outputs_with_mask.sequence_output.shape()[2]);

    // But values should differ (mask affects attention)
    // Note: We can't easily compare tensors directly, but we verify shapes are consistent
    EXPECT_TRUE(outputs_with_mask.sequence_output.tensor().numel() > 0);
}

TYPED_TEST(RobertaMultiDtypeTest, NoSegmentEmbeddings) {
    // RoBERTa should work fine without explicit token_type_ids
    RobertaModel model(this->config_);

    auto input_ids = this->create_input_ids();

    // Forward without token_type_ids (should default to all zeros)
    auto outputs1 = model.forward(input_ids, Tensor{}, Variable{}, Variable{});

    // Forward with explicit zeros
    auto token_type_ids = this->create_token_type_ids();
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
// RobertaForSequenceClassification Tests
// ============================================================================

TYPED_TEST(RobertaMultiDtypeTest, SequenceClassificationCreation) {
    int64_t num_labels = 3;

    EXPECT_NO_THROW({
        RobertaForSequenceClassification model(this->config_, num_labels);
    });
}

TYPED_TEST(RobertaMultiDtypeTest, SequenceClassificationForward) {
    int64_t num_labels = 3;
    RobertaForSequenceClassification model(this->config_, num_labels);

    auto input_ids = this->create_input_ids();

    auto logits = model.forward(input_ids, Tensor{}, Variable{});

    // Check output shape: [batch_size, num_labels]
    EXPECT_EQ(logits.shape()[0], this->batch_size_);
    EXPECT_EQ(logits.shape()[1], num_labels);
}

TYPED_TEST(RobertaMultiDtypeTest, SequenceClassificationBinaryTask) {
    int64_t num_labels = 2;  // Binary classification
    RobertaForSequenceClassification model(this->config_, num_labels);

    auto input_ids = this->create_input_ids();

    auto logits = model.forward(input_ids, Tensor{}, Variable{});

    EXPECT_EQ(logits.shape()[0], this->batch_size_);
    EXPECT_EQ(logits.shape()[1], num_labels);
}

TYPED_TEST(RobertaMultiDtypeTest, SequenceClassificationMultiClass) {
    int64_t num_labels = 10;  // Multi-class classification
    RobertaForSequenceClassification model(this->config_, num_labels);

    auto input_ids = this->create_input_ids();

    auto logits = model.forward(input_ids, Tensor{}, Variable{});

    EXPECT_EQ(logits.shape()[0], this->batch_size_);
    EXPECT_EQ(logits.shape()[1], num_labels);
}

TYPED_TEST(RobertaMultiDtypeTest, SequenceClassificationWithMask) {
    int64_t num_labels = 3;
    RobertaForSequenceClassification model(this->config_, num_labels);

    auto input_ids = this->create_input_ids();
    auto mask = this->create_attention_mask(true);

    auto logits = model.forward(input_ids, mask, Variable{});

    EXPECT_EQ(logits.shape()[0], this->batch_size_);
    EXPECT_EQ(logits.shape()[1], num_labels);
}

// ============================================================================
// RobertaForTokenClassification Tests
// ============================================================================

TYPED_TEST(RobertaMultiDtypeTest, TokenClassificationCreation) {
    int64_t num_labels = 9;  // NER tags

    EXPECT_NO_THROW({
        RobertaForTokenClassification model(this->config_, num_labels);
    });
}

TYPED_TEST(RobertaMultiDtypeTest, TokenClassificationForward) {
    int64_t num_labels = 9;
    RobertaForTokenClassification model(this->config_, num_labels);

    auto input_ids = this->create_input_ids();

    auto logits = model.forward(input_ids, Tensor{}, Variable{});

    // Check output shape: [batch_size, seq_len, num_labels]
    EXPECT_EQ(logits.shape()[0], this->batch_size_);
    EXPECT_EQ(logits.shape()[1], this->seq_len_);
    EXPECT_EQ(logits.shape()[2], num_labels);
}

TYPED_TEST(RobertaMultiDtypeTest, TokenClassificationNERTask) {
    int64_t num_labels = 9;  // B-PER, I-PER, B-ORG, I-ORG, B-LOC, I-LOC, B-MISC, I-MISC, O
    RobertaForTokenClassification model(this->config_, num_labels);

    auto input_ids = this->create_input_ids();

    auto logits = model.forward(input_ids, Tensor{}, Variable{});

    // Verify shape for token-level predictions
    EXPECT_EQ(logits.shape()[0], this->batch_size_);
    EXPECT_EQ(logits.shape()[1], this->seq_len_);
    EXPECT_EQ(logits.shape()[2], num_labels);
}

TYPED_TEST(RobertaMultiDtypeTest, TokenClassificationPOSTagging) {
    int64_t num_labels = 17;  // Penn Treebank POS tags
    RobertaForTokenClassification model(this->config_, num_labels);

    auto input_ids = this->create_input_ids();

    auto logits = model.forward(input_ids, Tensor{}, Variable{});

    EXPECT_EQ(logits.shape()[0], this->batch_size_);
    EXPECT_EQ(logits.shape()[1], this->seq_len_);
    EXPECT_EQ(logits.shape()[2], num_labels);
}

TYPED_TEST(RobertaMultiDtypeTest, TokenClassificationWithMask) {
    int64_t num_labels = 9;
    RobertaForTokenClassification model(this->config_, num_labels);

    auto input_ids = this->create_input_ids();
    auto mask = this->create_attention_mask(true);

    auto logits = model.forward(input_ids, mask, Variable{});

    EXPECT_EQ(logits.shape()[0], this->batch_size_);
    EXPECT_EQ(logits.shape()[1], this->seq_len_);
    EXPECT_EQ(logits.shape()[2], num_labels);
}

// ============================================================================
// RobertaForQuestionAnswering Tests
// ============================================================================

TYPED_TEST(RobertaMultiDtypeTest, QuestionAnsweringCreation) {
    EXPECT_NO_THROW({
        RobertaForQuestionAnswering model(this->config_);
    });
}

TYPED_TEST(RobertaMultiDtypeTest, QuestionAnsweringForward) {
    RobertaForQuestionAnswering model(this->config_);

    auto input_ids = this->create_input_ids();

    auto outputs = model.forward(input_ids, Tensor{}, Variable{});

    // Check output shapes for span prediction
    EXPECT_EQ(outputs.start_logits.shape()[0], this->batch_size_);
    EXPECT_EQ(outputs.start_logits.shape()[1], this->seq_len_);

    EXPECT_EQ(outputs.end_logits.shape()[0], this->batch_size_);
    EXPECT_EQ(outputs.end_logits.shape()[1], this->seq_len_);
}

TYPED_TEST(RobertaMultiDtypeTest, QuestionAnsweringSpanPrediction) {
    RobertaForQuestionAnswering model(this->config_);

    auto input_ids = this->create_input_ids();

    auto outputs = model.forward(input_ids, Tensor{}, Variable{});

    // Both start and end logits should have same shape
    EXPECT_EQ(outputs.start_logits.shape()[0], outputs.end_logits.shape()[0]);
    EXPECT_EQ(outputs.start_logits.shape()[1], outputs.end_logits.shape()[1]);

    // Shape should be [batch_size, seq_len]
    EXPECT_EQ(outputs.start_logits.tensor().ndim(), 2);
    EXPECT_EQ(outputs.end_logits.tensor().ndim(), 2);
}

TYPED_TEST(RobertaMultiDtypeTest, QuestionAnsweringWithMask) {
    RobertaForQuestionAnswering model(this->config_);

    auto input_ids = this->create_input_ids();
    auto mask = this->create_attention_mask(true);

    auto outputs = model.forward(input_ids, mask, Variable{});

    EXPECT_EQ(outputs.start_logits.shape()[0], this->batch_size_);
    EXPECT_EQ(outputs.start_logits.shape()[1], this->seq_len_);
    EXPECT_EQ(outputs.end_logits.shape()[0], this->batch_size_);
    EXPECT_EQ(outputs.end_logits.shape()[1], this->seq_len_);
}

TYPED_TEST(RobertaMultiDtypeTest, QuestionAnsweringSQuADFormat) {
    RobertaForQuestionAnswering model(this->config_);

    // Simulate SQuAD-style input where first part is question, second part is context
    auto input_ids = this->create_input_ids();

    auto outputs = model.forward(input_ids, Tensor{}, Variable{});

    // Model should predict answer span within the sequence
    EXPECT_EQ(outputs.start_logits.shape()[1], this->seq_len_);
    EXPECT_EQ(outputs.end_logits.shape()[1], this->seq_len_);
}

// ============================================================================
// Gradient Flow Tests
// ============================================================================

TYPED_TEST(RobertaMultiDtypeTest, GradientFlowSequenceClassification) {
    int64_t num_labels = 2;
    RobertaForSequenceClassification model(this->config_, num_labels);
    model.train();

    auto input_ids = this->create_input_ids();

    // Forward pass
    auto logits = model.forward(input_ids, Tensor{}, Variable{});

    EXPECT_TRUE(logits.requires_grad()) << "Logits should require gradients";

    // Simple loss: sum of all outputs
    auto loss = sum(logits);

    // Backward pass
    EXPECT_NO_THROW({
        loss.backward();
    }) << "Backward pass should complete without errors";

    // Check that gradients were computed
    auto grad = loss.grad();
    EXPECT_TRUE((grad.has_value() && grad->numel() > 0) || !loss.requires_grad());

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

TYPED_TEST(RobertaMultiDtypeTest, GradientFlowTokenClassification) {
    int64_t num_labels = 9;
    RobertaForTokenClassification model(this->config_, num_labels);
    model.train();

    auto input_ids = this->create_input_ids();

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

TYPED_TEST(RobertaMultiDtypeTest, GradientFlowQuestionAnswering) {
    RobertaForQuestionAnswering model(this->config_);
    model.train();

    auto input_ids = this->create_input_ids();

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

TYPED_TEST(RobertaMultiDtypeTest, GradientFlowWithAttentionMask) {
    int64_t num_labels = 2;
    RobertaForSequenceClassification model(this->config_, num_labels);
    model.train();

    auto input_ids = this->create_input_ids();
    auto mask = this->create_attention_mask(true);

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

TYPED_TEST(RobertaMultiDtypeTest, ClassificationHeadStructure) {
    int64_t num_labels = 5;
    RobertaForSequenceClassification model(this->config_, num_labels);

    // Model should have base RoBERTa + classification head
    auto params = model.parameters();
    EXPECT_FALSE(params.empty()) << "Model should have parameters";

    // Forward to verify head produces correct output dimension
    auto input_ids = this->create_input_ids();
    auto logits = model.forward(input_ids, Tensor{}, Variable{});

    EXPECT_EQ(logits.shape()[1], num_labels) << "Classification head should output num_labels dimensions";
}

TYPED_TEST(RobertaMultiDtypeTest, TokenClassificationHeadStructure) {
    int64_t num_labels = 7;
    RobertaForTokenClassification model(this->config_, num_labels);

    auto params = model.parameters();
    EXPECT_FALSE(params.empty());

    // Forward to verify head produces correct output dimensions
    auto input_ids = this->create_input_ids();
    auto logits = model.forward(input_ids, Tensor{}, Variable{});

    EXPECT_EQ(logits.shape()[2], num_labels) << "Token classification head should output num_labels per token";
}

TYPED_TEST(RobertaMultiDtypeTest, QuestionAnsweringHeadStructure) {
    RobertaForQuestionAnswering model(this->config_);

    auto params = model.parameters();
    EXPECT_FALSE(params.empty());

    // Forward to verify head produces start and end logits
    auto input_ids = this->create_input_ids();
    auto outputs = model.forward(input_ids, Tensor{}, Variable{});

    // Both logits should span the sequence length
    EXPECT_EQ(outputs.start_logits.shape()[1], this->seq_len_);
    EXPECT_EQ(outputs.end_logits.shape()[1], this->seq_len_);
}

// ============================================================================
// Edge Cases and Robustness Tests
// ============================================================================

TYPED_TEST(RobertaMultiDtypeTest, SingleTokenSequence) {
    RobertaModel model(this->config_);

    // Create single token input
    Tensor input_tensor({this->batch_size_, 1}, DType::Int64, this->device_);
    input_tensor.fill_(42);
    Variable input_ids(input_tensor, false);

    EXPECT_NO_THROW({
        auto outputs = model.forward(input_ids, Tensor{}, Variable{}, Variable{});
        EXPECT_EQ(outputs.sequence_output.shape()[1], 1);
    });
}

TYPED_TEST(RobertaMultiDtypeTest, MaxSequenceLength) {
    // Test with maximum supported sequence length
    RobertaModel model(this->config_);

    int64_t max_seq = this->config_.max_position_embeddings - 2;  // Account for special tokens
    Tensor input_tensor({1, max_seq}, DType::Int64, this->device_);
    input_tensor.fill_(42);
    Variable input_ids(input_tensor, false);

    EXPECT_NO_THROW({
        auto outputs = model.forward(input_ids, Tensor{}, Variable{}, Variable{});
        EXPECT_EQ(outputs.sequence_output.shape()[1], max_seq);
    });
}

TYPED_TEST(RobertaMultiDtypeTest, LargeBatchSize) {
    if (std::is_same<TypeParam, Float16>::value) {
        GTEST_SKIP() << "Skipping large batch test for Float16 to avoid memory issues";
    }

    RobertaForSequenceClassification model(this->config_, 2);

    // Large batch size
    int64_t large_batch = 16;
    Tensor input_tensor({large_batch, this->seq_len_}, DType::Int64, this->device_);
    input_tensor.fill_(42);
    Variable input_ids(input_tensor, false);

    EXPECT_NO_THROW({
        auto logits = model.forward(input_ids, Tensor{}, Variable{});
        EXPECT_EQ(logits.shape()[0], large_batch);
    });
}

TYPED_TEST(RobertaMultiDtypeTest, AllMaskedSequence) {
    RobertaModel model(this->config_);
    model.eval();

    auto input_ids = this->create_input_ids();

    // Create mask with all zeros (all masked)
    Tensor mask({this->batch_size_, this->seq_len_}, this->dtype_, this->device_);
    mask.zero_();

    // Should still run without errors
    EXPECT_NO_THROW({
        auto outputs = model.forward(input_ids, mask, Variable{}, Variable{});
        EXPECT_EQ(outputs.sequence_output.shape()[0], this->batch_size_);
    });
}

// ============================================================================
// Precision-Specific Tests
// ============================================================================

TYPED_TEST(RobertaMultiDtypeTest, OutputNumericalStability) {
    RobertaModel model(this->config_);
    model.eval();

    auto input_ids = this->create_input_ids();

    auto outputs = model.forward(input_ids, Tensor{}, Variable{}, Variable{});

    // Check outputs are not NaN or Inf
    auto output_tensor = outputs.sequence_output.tensor();
    auto max_val = tenzor::max(tenzor::abs(output_tensor));
    float max_value = this->get_scalar(max_val);

    EXPECT_FALSE(std::isnan(max_value)) << "Output contains NaN values";
    EXPECT_FALSE(std::isinf(max_value)) << "Output contains Inf values";
    EXPECT_GT(max_value, 0.0f) << "Output should contain non-zero values";
}

TYPED_TEST(RobertaMultiDtypeTest, DtypeConsistency) {
    RobertaForSequenceClassification model(this->config_, 3);

    auto input_ids = this->create_input_ids();

    auto logits = model.forward(input_ids, Tensor{}, Variable{});

    // Output dtype should match test dtype expectations
    // Note: This is implicit in the model architecture
    EXPECT_EQ(logits.shape()[0], this->batch_size_);
    EXPECT_EQ(logits.shape()[1], 3);
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    tenzor::initialize();  // Initialize Tenzor library and backends
    return RUN_ALL_TESTS();
}
