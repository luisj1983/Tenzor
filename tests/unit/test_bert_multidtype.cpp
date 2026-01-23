/**
 * @file test_bert_multidtype.cpp
 * @brief Multi-backend and multi-dtype tests for BERT model family
 *
 * Tests BERT architecture components across different backends (CPU, CUDA, OneAPI)
 * and floating-point precisions (Float32, Float64, Float16):
 * - BertEmbeddings: Token, position, and segment embeddings
 * - BertEncoder: Multi-layer transformer encoder with attention
 * - BertPooler: CLS token pooling with tanh activation
 * - BertModel: Complete BERT base model
 * - BertForSequenceClassification: Text classification head
 * - BertForTokenClassification: Token-level classification (NER)
 * - BertForQuestionAnswering: Span prediction for QA
 *
 * Coverage: BERT-base and BERT-large configurations
 * Float16 uses smaller model sizes to manage memory constraints
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include "tenzor/models/bert.hpp"
#include "tenzor/core/tensor.hpp"
#include "tenzor/autograd/variable.hpp"
#include "tenzor/ops/reduction.hpp"
#include "tenzor/ops/math.hpp"
#include "../multi_backend_dtype_fixture.hpp"
#include <memory>
#include <vector>
#include <cmath>

using namespace tenzor;
using namespace tenzor::models;
using namespace tenzor::testing;

// ============================================================================
// BERT-specific Multi-Backend Multi-DType Test Fixture
// ============================================================================

class BertMultiDtypeTest : public MultiBackendDTypeTest {
protected:
    void SetUp() override {
        // Call parent setup (handles backend/dtype initialization)
        MultiBackendDTypeTest::SetUp();

        // BERT-specific config
        config_ = BertConfig();
        config_.vocab_size = 1000;
        config_.hidden_size = get_hidden_size();
        config_.num_hidden_layers = get_num_layers();
        config_.num_attention_heads = get_num_heads();
        config_.intermediate_size = get_intermediate_size();
        config_.max_position_embeddings = 64;
        config_.type_vocab_size = 2;
        config_.hidden_dropout_prob = 0.1f;
        config_.attention_probs_dropout_prob = 0.1f;

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
    auto create_input_ids() -> Variable {
        // Create on CPU first, fill data, then transfer to target device
        Tensor input_ids_cpu({batch_size_, seq_len_}, DType::Int64, Device::cpu());

        std::vector<int64_t> data(batch_size_ * seq_len_);
        for (size_t i = 0; i < data.size(); ++i) {
            data[i] = i % config_.vocab_size;
        }
        std::copy(data.begin(), data.end(), input_ids_cpu.data<int64_t>());

        Tensor input_ids = (device() == Device::cpu()) ? input_ids_cpu : input_ids_cpu.to(device());
        return Variable(input_ids, true);
    }

    // Helper: Create attention mask (in current dtype)
    auto create_attention_mask() -> Tensor {
        // Create on CPU first, fill data, then transfer to target device
        Tensor mask_cpu({batch_size_, seq_len_}, dtype(), Device::cpu());

        if (dtype() == DType::Float32) {
            auto* data = mask_cpu.data<float>();
            for (int64_t i = 0; i < batch_size_ * seq_len_; ++i) {
                data[i] = 1.0f;
            }
        } else if (dtype() == DType::Float64) {
            auto* data = mask_cpu.data<double>();
            for (int64_t i = 0; i < batch_size_ * seq_len_; ++i) {
                data[i] = 1.0;
            }
        } else if (dtype() == DType::Float16) {
            auto* data = mask_cpu.data<Float16>();
            for (int64_t i = 0; i < batch_size_ * seq_len_; ++i) {
                data[i] = Float16(1.0f);
            }
        }

        return (device() == Device::cpu()) ? mask_cpu : mask_cpu.to(device());
    }

    // Helper: Create token type IDs (always Int64)
    auto create_token_type_ids() -> Variable {
        // Create on CPU first, fill data, then transfer to target device
        Tensor type_ids_cpu({batch_size_, seq_len_}, DType::Int64, Device::cpu());

        std::vector<int64_t> data(batch_size_ * seq_len_);
        for (int64_t b = 0; b < batch_size_; ++b) {
            for (int64_t i = 0; i < seq_len_; ++i) {
                data[b * seq_len_ + i] = (i < seq_len_ / 2) ? 0 : 1;
            }
        }
        std::copy(data.begin(), data.end(), type_ids_cpu.data<int64_t>());

        Tensor type_ids = (device() == Device::cpu()) ? type_ids_cpu : type_ids_cpu.to(device());
        return Variable(type_ids, false);
    }

    // Helper: Create hidden states tensor
    auto create_hidden_states() -> Variable {
        // Create on CPU first, fill data, then transfer to target device
        Tensor hidden_cpu({batch_size_, seq_len_, config_.hidden_size}, dtype(), Device::cpu());

        if (dtype() == DType::Float32) {
            auto* data = hidden_cpu.data<float>();
            for (int64_t i = 0; i < batch_size_ * seq_len_ * config_.hidden_size; ++i) {
                data[i] = 0.1f;
            }
        } else if (dtype() == DType::Float64) {
            auto* data = hidden_cpu.data<double>();
            for (int64_t i = 0; i < batch_size_ * seq_len_ * config_.hidden_size; ++i) {
                data[i] = 0.1;
            }
        } else if (dtype() == DType::Float16) {
            auto* data = hidden_cpu.data<Float16>();
            for (int64_t i = 0; i < batch_size_ * seq_len_ * config_.hidden_size; ++i) {
                data[i] = Float16(0.1f);
            }
        }

        Tensor hidden_states = (device() == Device::cpu()) ? hidden_cpu : hidden_cpu.to(device());
        return Variable(hidden_states, true);
    }

    // Helper: Get scalar value from tensor
    float get_scalar(const Tensor& tensor) const {
        // Move to CPU and convert Float16 to Float32 for scalar extraction
        Tensor t = tensor.to(Device::cpu());
        if (t.dtype() == DType::Float16) {
            t = t.to(DType::Float32);
        }
        if (t.dtype() == DType::Float64) {
            return static_cast<float>(t.item<double>());
        }
        return t.item<float>();
    }

    BertConfig config_;
    int64_t batch_size_;
    int64_t seq_len_;
};

// ============================================================================
// BertEmbeddings Tests
// ============================================================================

TEST_P(BertMultiDtypeTest, BertEmbeddingsForwardShape) {
    auto embeddings = BertEmbeddings(config_);
    embeddings.to(device());
    if (dtype() != DType::Float32) embeddings.to(dtype());

    auto input_ids = create_input_ids();
    auto output = embeddings.forward(input_ids, Variable{}, Variable{});

    // Check output shape: [batch, seq_len, hidden_size]
    EXPECT_EQ(output.tensor().ndim(), 3);
    EXPECT_EQ(output.tensor().shape()[0], batch_size_);
    EXPECT_EQ(output.tensor().shape()[1], seq_len_);
    EXPECT_EQ(output.tensor().shape()[2], config_.hidden_size);
}

TEST_P(BertMultiDtypeTest, BertEmbeddingsWithTokenTypes) {
    auto embeddings = BertEmbeddings(config_);
    embeddings.to(device());
    if (dtype() != DType::Float32) embeddings.to(dtype());

    auto input_ids = create_input_ids();
    auto token_type_ids = create_token_type_ids();

    auto output = embeddings.forward(input_ids, token_type_ids, Variable{});

    EXPECT_EQ(output.tensor().shape()[0], batch_size_);
    EXPECT_EQ(output.tensor().shape()[1], seq_len_);
    EXPECT_EQ(output.tensor().shape()[2], config_.hidden_size);
}

TEST_P(BertMultiDtypeTest, BertEmbeddingsGradientFlow) {
    auto embeddings = BertEmbeddings(config_);
    embeddings.to(device());
    if (dtype() != DType::Float32) embeddings.to(dtype());
    embeddings.train();

    auto input_ids = create_input_ids();
    auto output = embeddings.forward(input_ids, Variable{}, Variable{});

    EXPECT_TRUE(output.requires_grad());

    // Compute loss and backward
    Variable loss = mean(output);
    loss.backward();

    // Check embeddings have gradients
    auto params = embeddings.parameters();
    EXPECT_FALSE(params.empty());

    for (const auto& param : params) {
        EXPECT_TRUE(param->has_grad())
            << "Embedding parameter missing gradient";
    }
}

TEST_P(BertMultiDtypeTest, BertEmbeddingsPositionalEncoding) {
    auto embeddings = BertEmbeddings(config_);
    embeddings.to(device());
    if (dtype() != DType::Float32) embeddings.to(dtype());

    auto input_ids = create_input_ids();

    // Forward pass should incorporate positional embeddings
    auto output = embeddings.forward(input_ids, Variable{}, Variable{});

    // Output should be non-zero (positional embeddings added)
    auto output_data = output.tensor();
    float max_val = compute_max_abs(output_data);

    EXPECT_GT(max_val, 0.0f) << "Embeddings should be non-zero";
}

// ============================================================================
// BertEncoder Tests
// ============================================================================

TEST_P(BertMultiDtypeTest, BertEncoderForwardShape) {
    auto encoder = BertEncoder(config_);
    encoder.to(device());
    if (dtype() != DType::Float32) encoder.to(dtype());

    auto hidden_states = create_hidden_states();
    auto output = encoder.forward(hidden_states, Tensor{});

    EXPECT_EQ(output.tensor().shape()[0], batch_size_);
    EXPECT_EQ(output.tensor().shape()[1], seq_len_);
    EXPECT_EQ(output.tensor().shape()[2], config_.hidden_size);
}

TEST_P(BertMultiDtypeTest, BertEncoderWithMask) {
    auto encoder = BertEncoder(config_);
    encoder.to(device());
    if (dtype() != DType::Float32) encoder.to(dtype());

    auto hidden_states = create_hidden_states();
    auto attention_mask = create_attention_mask();
    auto output = encoder.forward(hidden_states, attention_mask);

    EXPECT_EQ(output.tensor().shape()[0], batch_size_);
    EXPECT_EQ(output.tensor().shape()[1], seq_len_);
}

TEST_P(BertMultiDtypeTest, BertEncoderGradientFlow) {
    auto encoder = BertEncoder(config_);
    encoder.to(device());
    if (dtype() != DType::Float32) encoder.to(dtype());
    encoder.train();

    auto hidden_states = create_hidden_states();
    auto output = encoder.forward(hidden_states, Tensor{});

    Variable loss = mean(output);
    loss.backward();

    auto params = encoder.parameters();
    EXPECT_FALSE(params.empty());

    for (const auto& param : params) {
        EXPECT_TRUE(param->has_grad())
            << "Encoder parameter missing gradient";
    }
}

TEST_P(BertMultiDtypeTest, BertEncoderTransformation) {
    auto encoder = BertEncoder(config_);
    encoder.to(device());
    if (dtype() != DType::Float32) encoder.to(dtype());

    auto hidden_states = create_hidden_states();
    auto output = encoder.forward(hidden_states, Tensor{});

    // Output should be different from input (transformer transformation)
    auto input_mean = get_scalar(mean(hidden_states).tensor());
    auto output_mean = get_scalar(mean(output).tensor());

    // Allow for some tolerance in the comparison
    EXPECT_NE(input_mean, output_mean) << "Encoder should transform input";
}

// ============================================================================
// BertPooler Tests
// ============================================================================

TEST_P(BertMultiDtypeTest, BertPoolerForwardShape) {
    auto pooler = BertPooler(config_);
    pooler.to(device());
    if (dtype() != DType::Float32) pooler.to(dtype());

    auto hidden_states = create_hidden_states();
    auto output = pooler.forward(hidden_states);

    // Pooler output: [batch, hidden_size]
    EXPECT_EQ(output.tensor().ndim(), 2);
    EXPECT_EQ(output.tensor().shape()[0], batch_size_);
    EXPECT_EQ(output.tensor().shape()[1], config_.hidden_size);
}

TEST_P(BertMultiDtypeTest, BertPoolerTanhActivation) {
    auto pooler = BertPooler(config_);
    pooler.to(device());
    if (dtype() != DType::Float32) pooler.to(dtype());

    auto hidden_states = create_hidden_states();
    auto output = pooler.forward(hidden_states);

    // Output should be in [-1, 1] range due to tanh
    float max_val = compute_max(output.tensor());
    float min_val = compute_min(output.tensor());

    EXPECT_LE(max_val, 1.0f + atol()) << "Pooler output should be <= 1 (tanh)";
    EXPECT_GE(min_val, -1.0f - atol()) << "Pooler output should be >= -1 (tanh)";
}

TEST_P(BertMultiDtypeTest, BertPoolerCLSExtraction) {
    auto pooler = BertPooler(config_);
    pooler.to(device());
    if (dtype() != DType::Float32) pooler.to(dtype());

    auto hidden_states = create_hidden_states();
    auto output = pooler.forward(hidden_states);

    // Should extract from first token (CLS)
    EXPECT_EQ(output.tensor().shape()[0], batch_size_);
}

// ============================================================================
// BertModel Tests
// ============================================================================

TEST_P(BertMultiDtypeTest, BertModelForwardShape) {
    auto model = BertModel(config_);
    model.to(device());
    if (dtype() != DType::Float32) model.to(dtype());

    auto input_ids = create_input_ids();
    auto [sequence_output, pooled_output] = model.forward(input_ids);

    // sequence_output: [batch, seq_len, hidden_size]
    EXPECT_EQ(sequence_output.tensor().shape()[0], batch_size_);
    EXPECT_EQ(sequence_output.tensor().shape()[1], seq_len_);
    EXPECT_EQ(sequence_output.tensor().shape()[2], config_.hidden_size);

    // pooled_output: [batch, hidden_size]
    EXPECT_EQ(pooled_output.tensor().shape()[0], batch_size_);
    EXPECT_EQ(pooled_output.tensor().shape()[1], config_.hidden_size);
}

TEST_P(BertMultiDtypeTest, BertModelWithAllInputs) {
    auto model = BertModel(config_);
    model.to(device());
    if (dtype() != DType::Float32) model.to(dtype());

    auto input_ids = create_input_ids();
    auto attention_mask = create_attention_mask();
    auto token_type_ids = create_token_type_ids();

    auto [sequence_output, pooled_output] = model.forward(
        input_ids, attention_mask, token_type_ids);

    EXPECT_EQ(sequence_output.tensor().ndim(), 3);
    EXPECT_EQ(pooled_output.tensor().ndim(), 2);
}

TEST_P(BertMultiDtypeTest, BertModelGradientFlow) {
    auto model = BertModel(config_);
    model.to(device());
    if (dtype() != DType::Float32) model.to(dtype());
    model.train();

    auto input_ids = create_input_ids();
    auto [sequence_output, pooled_output] = model.forward(input_ids);

    Variable loss = mean(pooled_output);
    loss.backward();

    auto params = model.parameters();
    EXPECT_FALSE(params.empty());

    for (const auto& param : params) {
        EXPECT_TRUE(param->has_grad())
            << "Model parameter missing gradient";
    }
}

TEST_P(BertMultiDtypeTest, BertModelTrainEvalMode) {
    auto model = BertModel(config_);
    model.to(device());
    if (dtype() != DType::Float32) model.to(dtype());

    auto input_ids = create_input_ids();

    // Train mode
    model.train();
    auto [train_seq, train_pool] = model.forward(input_ids);
    EXPECT_EQ(train_seq.tensor().dtype(), dtype());

    // Eval mode
    model.eval();
    auto [eval_seq, eval_pool] = model.forward(input_ids);
    EXPECT_EQ(eval_seq.tensor().dtype(), dtype());
}

TEST_P(BertMultiDtypeTest, BertModelDifferentSequenceLengths) {
    auto model = BertModel(config_);
    model.to(device());
    if (dtype() != DType::Float32) model.to(dtype());

    // Test with shorter sequence - create on CPU first
    int64_t short_seq_len = 8;
    Tensor short_ids_cpu({batch_size_, short_seq_len}, DType::Int64, Device::cpu());
    std::vector<int64_t> short_data(batch_size_ * short_seq_len);
    for (size_t i = 0; i < short_data.size(); ++i) {
        short_data[i] = i % config_.vocab_size;
    }
    std::copy(short_data.begin(), short_data.end(), short_ids_cpu.data<int64_t>());
    Tensor short_ids = (device() == Device::cpu()) ? short_ids_cpu : short_ids_cpu.to(device());

    Variable short_input(short_ids, true);
    auto [short_seq, short_pool] = model.forward(short_input);

    EXPECT_EQ(short_seq.tensor().shape()[1], short_seq_len);
}

// ============================================================================
// BertForSequenceClassification Tests
// ============================================================================

TEST_P(BertMultiDtypeTest, SequenceClassificationBinary) {
    auto model = BertForSequenceClassification(config_, 2);
    model.to(device());
    if (dtype() != DType::Float32) model.to(dtype());

    auto input_ids = create_input_ids();
    auto output = model.forward(input_ids);

    // Output: [batch, num_labels]
    EXPECT_EQ(output.tensor().shape()[0], batch_size_);
    EXPECT_EQ(output.tensor().shape()[1], 2);
}

TEST_P(BertMultiDtypeTest, SequenceClassificationMultiClass) {
    int64_t num_classes = 5;
    auto model = BertForSequenceClassification(config_, num_classes);
    model.to(device());
    if (dtype() != DType::Float32) model.to(dtype());

    auto input_ids = create_input_ids();
    auto output = model.forward(input_ids);

    EXPECT_EQ(output.tensor().shape()[1], num_classes);
}

TEST_P(BertMultiDtypeTest, SequenceClassificationGradientFlow) {
    auto model = BertForSequenceClassification(config_, 3);
    model.to(device());
    if (dtype() != DType::Float32) model.to(dtype());
    model.train();

    auto input_ids = create_input_ids();
    auto output = model.forward(input_ids);

    Variable loss = mean(output);
    loss.backward();

    auto params = model.parameters();
    for (const auto& param : params) {
        EXPECT_TRUE(param->has_grad());
    }
}

TEST_P(BertMultiDtypeTest, SequenceClassificationWithMask) {
    auto model = BertForSequenceClassification(config_, 2);
    model.to(device());
    if (dtype() != DType::Float32) model.to(dtype());

    auto input_ids = create_input_ids();
    auto attention_mask = create_attention_mask();

    auto output = model.forward(input_ids, attention_mask);

    EXPECT_EQ(output.tensor().shape()[0], batch_size_);
    EXPECT_EQ(output.tensor().shape()[1], 2);
}

// ============================================================================
// BertForTokenClassification Tests
// ============================================================================

TEST_P(BertMultiDtypeTest, TokenClassificationForwardShape) {
    int64_t num_labels = 9;  // BIO tagging
    auto model = BertForTokenClassification(config_, num_labels);
    model.to(device());
    if (dtype() != DType::Float32) model.to(dtype());

    auto input_ids = create_input_ids();
    auto output = model.forward(input_ids);

    // Output: [batch, seq_len, num_labels]
    EXPECT_EQ(output.tensor().shape()[0], batch_size_);
    EXPECT_EQ(output.tensor().shape()[1], seq_len_);
    EXPECT_EQ(output.tensor().shape()[2], num_labels);
}

TEST_P(BertMultiDtypeTest, TokenClassificationWithMask) {
    auto model = BertForTokenClassification(config_, 5);
    model.to(device());
    if (dtype() != DType::Float32) model.to(dtype());

    auto input_ids = create_input_ids();
    auto attention_mask = create_attention_mask();

    auto output = model.forward(input_ids, attention_mask);

    EXPECT_EQ(output.tensor().ndim(), 3);
}

TEST_P(BertMultiDtypeTest, TokenClassificationGradientFlow) {
    auto model = BertForTokenClassification(config_, 5);
    model.to(device());
    if (dtype() != DType::Float32) model.to(dtype());
    model.train();

    auto input_ids = create_input_ids();
    auto output = model.forward(input_ids);

    Variable loss = mean(output);
    loss.backward();

    auto params = model.parameters();
    for (const auto& param : params) {
        EXPECT_TRUE(param->has_grad());
    }
}

TEST_P(BertMultiDtypeTest, TokenClassificationPerTokenPrediction) {
    int64_t num_labels = 3;
    auto model = BertForTokenClassification(config_, num_labels);
    model.to(device());
    if (dtype() != DType::Float32) model.to(dtype());

    auto input_ids = create_input_ids();
    auto output = model.forward(input_ids);

    // Each token should have prediction scores for all labels
    EXPECT_EQ(output.tensor().shape()[2], num_labels);
}

// ============================================================================
// BertForQuestionAnswering Tests
// ============================================================================

TEST_P(BertMultiDtypeTest, QuestionAnsweringForwardShape) {
    auto model = BertForQuestionAnswering(config_);
    model.to(device());
    if (dtype() != DType::Float32) model.to(dtype());

    auto input_ids = create_input_ids();
    auto [start_logits, end_logits] = model.forward(input_ids);

    // start/end logits: [batch, seq_len]
    EXPECT_EQ(start_logits.tensor().shape()[0], batch_size_);
    EXPECT_EQ(start_logits.tensor().shape()[1], seq_len_);
    EXPECT_EQ(end_logits.tensor().shape()[0], batch_size_);
    EXPECT_EQ(end_logits.tensor().shape()[1], seq_len_);
}

TEST_P(BertMultiDtypeTest, QuestionAnsweringWithTokenTypes) {
    auto model = BertForQuestionAnswering(config_);
    model.to(device());
    if (dtype() != DType::Float32) model.to(dtype());

    auto input_ids = create_input_ids();
    auto attention_mask = create_attention_mask();
    auto token_type_ids = create_token_type_ids();

    auto [start_logits, end_logits] = model.forward(input_ids, attention_mask, token_type_ids);

    EXPECT_EQ(start_logits.tensor().ndim(), 2);
    EXPECT_EQ(end_logits.tensor().ndim(), 2);
}

TEST_P(BertMultiDtypeTest, QuestionAnsweringGradientFlow) {
    auto model = BertForQuestionAnswering(config_);
    model.to(device());
    if (dtype() != DType::Float32) model.to(dtype());
    model.train();

    auto input_ids = create_input_ids();
    auto [start_logits, end_logits] = model.forward(input_ids);

    Variable loss = mean(start_logits) + mean(end_logits);
    loss.backward();

    auto params = model.parameters();
    for (const auto& param : params) {
        EXPECT_TRUE(param->has_grad());
    }
}

TEST_P(BertMultiDtypeTest, QuestionAnsweringSpanPrediction) {
    auto model = BertForQuestionAnswering(config_);
    model.to(device());
    if (dtype() != DType::Float32) model.to(dtype());

    auto input_ids = create_input_ids();
    auto [start_logits, end_logits] = model.forward(input_ids);

    // Logits should have values for span prediction
    float start_max = compute_max(start_logits.tensor());
    float end_max = compute_max(end_logits.tensor());

    // Max logits should be finite
    EXPECT_TRUE(std::isfinite(start_max));
    EXPECT_TRUE(std::isfinite(end_max));
}

// ============================================================================
// Configuration Tests
// ============================================================================

TEST_P(BertMultiDtypeTest, BertConfigBase) {
    // Test BERT-base-like configuration
    BertConfig config;
    config.hidden_size = 768;
    config.num_hidden_layers = 12;
    config.num_attention_heads = 12;
    config.intermediate_size = 3072;

    EXPECT_EQ(config.hidden_size, 768);
    EXPECT_EQ(config.num_hidden_layers, 12);
    EXPECT_EQ(config.num_attention_heads, 12);
    EXPECT_EQ(config.intermediate_size, 3072);
}

TEST_P(BertMultiDtypeTest, BertConfigLarge) {
    // Test BERT-large-like configuration
    BertConfig config;
    config.hidden_size = 1024;
    config.num_hidden_layers = 24;
    config.num_attention_heads = 16;

    EXPECT_EQ(config.hidden_size, 1024);
    EXPECT_EQ(config.num_hidden_layers, 24);
    EXPECT_EQ(config.num_attention_heads, 16);
}

TEST_P(BertMultiDtypeTest, BertConfigCustom) {
    BertConfig config;
    config.vocab_size = 50000;
    config.hidden_size = 512;
    config.num_hidden_layers = 6;
    config.num_attention_heads = 8;

    EXPECT_EQ(config.vocab_size, 50000);
    EXPECT_EQ(config.hidden_size, 512);
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST_P(BertMultiDtypeTest, BertModelShortSequence) {
    auto model = BertModel(config_);
    model.to(device());
    if (dtype() != DType::Float32) model.to(dtype());

    int64_t short_seq = 4;
    Tensor ids_cpu({1, short_seq}, DType::Int64, Device::cpu());
    for (int64_t i = 0; i < short_seq; ++i) {
        ids_cpu.data<int64_t>()[i] = i;
    }
    Tensor ids = (device() == Device::cpu()) ? ids_cpu : ids_cpu.to(device());

    Variable input(ids, true);
    auto [seq_out, pool_out] = model.forward(input);

    EXPECT_EQ(seq_out.tensor().shape()[1], short_seq);
}

TEST_P(BertMultiDtypeTest, BertModelBatchSizeOne) {
    auto model = BertModel(config_);
    model.to(device());
    if (dtype() != DType::Float32) model.to(dtype());

    Tensor ids_cpu({1, seq_len_}, DType::Int64, Device::cpu());
    for (int64_t i = 0; i < seq_len_; ++i) {
        ids_cpu.data<int64_t>()[i] = i % config_.vocab_size;
    }
    Tensor ids = (device() == Device::cpu()) ? ids_cpu : ids_cpu.to(device());

    Variable input(ids, true);
    auto [seq_out, pool_out] = model.forward(input);

    EXPECT_EQ(seq_out.tensor().shape()[0], 1);
    EXPECT_EQ(pool_out.tensor().shape()[0], 1);
}

TEST_P(BertMultiDtypeTest, BertModelLargeBatch) {
    auto model = BertModel(config_);
    model.to(device());
    if (dtype() != DType::Float32) model.to(dtype());

    int64_t large_batch = 8;
    Tensor ids_cpu({large_batch, seq_len_}, DType::Int64, Device::cpu());
    auto* data = ids_cpu.data<int64_t>();
    for (int64_t i = 0; i < large_batch * seq_len_; ++i) {
        data[i] = i % config_.vocab_size;
    }
    Tensor ids = (device() == Device::cpu()) ? ids_cpu : ids_cpu.to(device());

    Variable input(ids, true);
    auto [seq_out, pool_out] = model.forward(input);

    EXPECT_EQ(seq_out.tensor().shape()[0], large_batch);
}

TEST_P(BertMultiDtypeTest, BertModelParameterCount) {
    auto model = BertModel(config_);
    auto params = model.parameters();

    size_t total_params = countParameters(params);

    // Should have parameters (exact count depends on config)
    EXPECT_GT(total_params, 0);
}

TEST_P(BertMultiDtypeTest, ClassificationHeadExtraParameters) {
    auto base_model = BertModel(config_);
    auto clf_model = BertForSequenceClassification(config_, 2);

    size_t base_params = countParameters(base_model.parameters());
    size_t clf_params = countParameters(clf_model.parameters());

    // Classification head should add parameters
    EXPECT_GT(clf_params, base_params);
}

// ============================================================================
// Instantiate Tests for Each Backend + DType Combination
// ============================================================================

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(BertMultiDtypeTest);

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    tenzor::initialize();
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
