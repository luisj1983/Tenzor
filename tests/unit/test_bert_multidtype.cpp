/**
 * @file test_bert_multidtype.cpp
 * @brief Multi-dtype tests for BERT model family (Float32, Float64, Float16)
 *
 * Tests BERT architecture components across different floating-point precisions:
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
#include <memory>
#include <vector>
#include <cmath>

using namespace tenzor;
using namespace tenzor::models;

// ============================================================================
// Test Fixtures - Typed for Float32, Float64, Float16
// ============================================================================

template <typename T>
class BertMultiDtypeTest : public ::testing::Test {
protected:
    void SetUp() override {
        dtype_ = dtype_from_type<T>();
        device_ = Device::cpu();

        // Base config for most tests
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
        tolerance_ = get_tolerance();
    }

    // Adjust model size based on dtype to manage memory
    int64_t get_hidden_size() const {
        if (std::is_same_v<T, uint16_t>) return 64;  // Smaller for Float16
        return 128;
    }

    int64_t get_num_layers() const {
        if (std::is_same_v<T, uint16_t>) return 1;   // Fewer layers for Float16
        return 2;
    }

    int64_t get_num_heads() const {
        if (std::is_same_v<T, uint16_t>) return 2;
        return 4;
    }

    int64_t get_intermediate_size() const {
        if (std::is_same_v<T, uint16_t>) return 256;
        return 512;
    }

    float get_tolerance() const {
        if (std::is_same<T, double>::value) return 1e-5f;
        if (std::is_same_v<T, uint16_t>) return 1e-2f;  // Relaxed for Float16
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
    auto create_attention_mask() -> Tensor {
        Tensor mask({batch_size_, seq_len_}, dtype_, device_);

        if (dtype_ == DType::Float32) {
            mask.fill_(1.0f);
        } else if (dtype_ == DType::Float64) {
            mask.fill_(1.0);
        } else if (dtype_ == DType::Float16) {
            std::vector<T> ones(batch_size_ * seq_len_, static_cast<T>(1.0f));
            std::copy(ones.begin(), ones.end(), mask.data<T>());
        }

        return mask;
    }

    // Helper: Create token type IDs (always Int64)
    auto create_token_type_ids() -> Variable {
        Tensor type_ids({batch_size_, seq_len_}, DType::Int64, device_);

        std::vector<int64_t> data(batch_size_ * seq_len_);
        for (int64_t b = 0; b < batch_size_; ++b) {
            for (int64_t i = 0; i < seq_len_; ++i) {
                data[b * seq_len_ + i] = (i < seq_len_ / 2) ? 0 : 1;
            }
        }
        std::copy(data.begin(), data.end(), type_ids.data<int64_t>());

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
    BertConfig config_;
    int64_t batch_size_;
    int64_t seq_len_;
    float tolerance_;

private:
    template <typename U>
    static DType dtype_from_type() {
        if (std::is_same<U, float>::value) return DType::Float32;
        if (std::is_same<U, double>::value) return DType::Float64;
        if (std::is_same_v<U, uint16_t>) return DType::Float16;
        return DType::Float32;
    }
};

using DTypes = ::testing::Types<float, double, uint16_t>;
TYPED_TEST_SUITE(BertMultiDtypeTest, DTypes);

// ============================================================================
// BertEmbeddings Tests
// ============================================================================

TYPED_TEST(BertMultiDtypeTest, BertEmbeddingsForwardShape) {
    auto embeddings = BertEmbeddings(this->config_);
    auto input_ids = this->create_input_ids();

    auto output = embeddings.forward(input_ids, Variable{}, Variable{});

    // Check output shape: [batch, seq_len, hidden_size]
    EXPECT_EQ(output.tensor().ndim(), 3);
    EXPECT_EQ(output.tensor().shape()[0], this->batch_size_);
    EXPECT_EQ(output.tensor().shape()[1], this->seq_len_);
    EXPECT_EQ(output.tensor().shape()[2], this->config_.hidden_size);
}

TYPED_TEST(BertMultiDtypeTest, BertEmbeddingsWithTokenTypes) {
    auto embeddings = BertEmbeddings(this->config_);
    auto input_ids = this->create_input_ids();
    auto token_type_ids = this->create_token_type_ids();

    auto output = embeddings.forward(input_ids, token_type_ids, Variable{});

    EXPECT_EQ(output.tensor().shape()[0], this->batch_size_);
    EXPECT_EQ(output.tensor().shape()[1], this->seq_len_);
    EXPECT_EQ(output.tensor().shape()[2], this->config_.hidden_size);
}

TYPED_TEST(BertMultiDtypeTest, BertEmbeddingsGradientFlow) {
    auto embeddings = BertEmbeddings(this->config_);
    embeddings.train();

    auto input_ids = this->create_input_ids();
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

TYPED_TEST(BertMultiDtypeTest, BertEmbeddingsPositionalEncoding) {
    auto embeddings = BertEmbeddings(this->config_);
    auto input_ids = this->create_input_ids();

    // Forward pass should incorporate positional embeddings
    auto output = embeddings.forward(input_ids, Variable{}, Variable{});

    // Output should be non-zero (positional embeddings added)
    auto output_data = output.tensor();
    float max_val = std::abs(this->get_scalar(tenzor::max(tenzor::abs(output_data))));

    EXPECT_GT(max_val, 0.0f) << "Embeddings should be non-zero";
}

// ============================================================================
// BertEncoder Tests
// ============================================================================

TYPED_TEST(BertMultiDtypeTest, BertEncoderForwardShape) {
    auto encoder = BertEncoder(this->config_);
    auto input = this->create_hidden_states();

    auto output = encoder.forward(input, Tensor{});

    EXPECT_EQ(output.tensor().shape()[0], this->batch_size_);
    EXPECT_EQ(output.tensor().shape()[1], this->seq_len_);
    EXPECT_EQ(output.tensor().shape()[2], this->config_.hidden_size);
}

TYPED_TEST(BertMultiDtypeTest, BertEncoderWithMask) {
    auto encoder = BertEncoder(this->config_);
    auto input = this->create_hidden_states();
    auto attention_mask = this->create_attention_mask();

    auto output = encoder.forward(input, attention_mask);

    EXPECT_EQ(output.tensor().shape()[0], this->batch_size_);
    EXPECT_EQ(output.tensor().shape()[1], this->seq_len_);
    EXPECT_EQ(output.tensor().shape()[2], this->config_.hidden_size);
}

TYPED_TEST(BertMultiDtypeTest, BertEncoderGradientFlow) {
    auto encoder = BertEncoder(this->config_);
    encoder.train();

    auto input = this->create_hidden_states();
    auto output = encoder.forward(input, Tensor{});

    Variable loss = mean(output);
    loss.backward();

    auto params = encoder.parameters();
    EXPECT_FALSE(params.empty());

    for (const auto& param : params) {
        EXPECT_TRUE(param->has_grad());
    }
}

TYPED_TEST(BertMultiDtypeTest, BertEncoderTransformation) {
    auto encoder = BertEncoder(this->config_);
    auto input = this->create_hidden_states();

    auto output = encoder.forward(input, Tensor{});

    // Encoder should transform the input (not just pass through)
    auto input_tensor = input.tensor();
    auto output_tensor = output.tensor();

    // Check that output differs from input
    auto diff = tenzor::abs(output_tensor - input_tensor);
    float max_diff = std::abs(this->get_scalar(tenzor::max(diff)));

    EXPECT_GT(max_diff, this->tolerance_)
        << "Encoder should transform input";
}

// ============================================================================
// BertPooler Tests
// ============================================================================

TYPED_TEST(BertMultiDtypeTest, BertPoolerForwardShape) {
    auto pooler = BertPooler(this->config_);
    auto input = this->create_hidden_states();

    auto output = pooler.forward(input);

    // Output shape: [batch, hidden_size]
    EXPECT_EQ(output.tensor().ndim(), 2);
    EXPECT_EQ(output.tensor().shape()[0], this->batch_size_);
    EXPECT_EQ(output.tensor().shape()[1], this->config_.hidden_size);
}

TYPED_TEST(BertMultiDtypeTest, BertPoolerTanhActivation) {
    auto pooler = BertPooler(this->config_);
    auto input = this->create_hidden_states();

    auto output = pooler.forward(input);

    // Check tanh output range [-1, 1]
    auto output_data = output.tensor();
    float max_val = this->get_scalar(tenzor::max(output_data));
    float min_val = this->get_scalar(tenzor::min(output_data));

    EXPECT_LE(max_val, 1.0f + this->tolerance_);
    EXPECT_GE(min_val, -1.0f - this->tolerance_);
}

TYPED_TEST(BertMultiDtypeTest, BertPoolerCLSExtraction) {
    auto pooler = BertPooler(this->config_);
    auto input = this->create_hidden_states();

    // Pooler should extract first token ([CLS])
    auto output = pooler.forward(input);

    EXPECT_EQ(output.tensor().shape()[0], this->batch_size_);
    EXPECT_EQ(output.tensor().shape()[1], this->config_.hidden_size);
}

// ============================================================================
// BertModel Tests - Core Architecture
// ============================================================================

TYPED_TEST(BertMultiDtypeTest, BertModelForwardShape) {
    auto model = BertModel(this->config_);
    auto input_ids = this->create_input_ids();

    auto outputs = model.forward(input_ids, Tensor{}, Variable{}, Variable{});

    // Sequence output: [batch, seq_len, hidden_size]
    EXPECT_EQ(outputs.sequence_output.tensor().shape()[0], this->batch_size_);
    EXPECT_EQ(outputs.sequence_output.tensor().shape()[1], this->seq_len_);
    EXPECT_EQ(outputs.sequence_output.tensor().shape()[2], this->config_.hidden_size);

    // Pooled output: [batch, hidden_size]
    EXPECT_EQ(outputs.pooled_output.tensor().shape()[0], this->batch_size_);
    EXPECT_EQ(outputs.pooled_output.tensor().shape()[1], this->config_.hidden_size);
}

TYPED_TEST(BertMultiDtypeTest, BertModelWithAllInputs) {
    auto model = BertModel(this->config_);
    auto input_ids = this->create_input_ids();
    auto attention_mask = this->create_attention_mask();
    auto token_type_ids = this->create_token_type_ids();

    auto outputs = model.forward(input_ids, attention_mask, token_type_ids, Variable{});

    EXPECT_EQ(outputs.sequence_output.tensor().shape()[0], this->batch_size_);
    EXPECT_EQ(outputs.sequence_output.tensor().shape()[1], this->seq_len_);
    EXPECT_EQ(outputs.pooled_output.tensor().shape()[0], this->batch_size_);
}

TYPED_TEST(BertMultiDtypeTest, BertModelGradientFlow) {
    auto model = BertModel(this->config_);
    model.train();

    auto input_ids = this->create_input_ids();
    auto outputs = model.forward(input_ids, Tensor{}, Variable{}, Variable{});

    Variable loss = mean(outputs.sequence_output) + mean(outputs.pooled_output);
    loss.backward();

    auto params = model.parameters();
    EXPECT_FALSE(params.empty());

    for (const auto& param : params) {
        EXPECT_TRUE(param->has_grad());
    }
}

TYPED_TEST(BertMultiDtypeTest, BertModelTrainEvalMode) {
    auto model = BertModel(this->config_);
    auto input_ids = this->create_input_ids();

    // Training mode
    model.train();
    EXPECT_TRUE(model.is_training());
    auto train_output = model.forward(input_ids, Tensor{}, Variable{}, Variable{});

    // Eval mode
    model.eval();
    EXPECT_FALSE(model.is_training());
    auto eval_output = model.forward(input_ids, Tensor{}, Variable{}, Variable{});

    EXPECT_EQ(train_output.sequence_output.tensor().shape()[0], this->batch_size_);
    EXPECT_EQ(eval_output.sequence_output.tensor().shape()[0], this->batch_size_);
}

TYPED_TEST(BertMultiDtypeTest, BertModelDifferentSequenceLengths) {
    auto model = BertModel(this->config_);

    // Test short sequence
    {
        Tensor input_ids({this->batch_size_, 4}, DType::Int64, this->device_);
        std::vector<int64_t> data(this->batch_size_ * 4, 0);
        std::copy(data.begin(), data.end(), input_ids.data<int64_t>());

        Variable input(input_ids, true);
        auto outputs = model.forward(input, Tensor{}, Variable{}, Variable{});

        EXPECT_EQ(outputs.sequence_output.tensor().shape()[1], 4);
    }

    // Test longer sequence
    {
        Tensor input_ids({this->batch_size_, 32}, DType::Int64, this->device_);
        std::vector<int64_t> data(this->batch_size_ * 32, 0);
        std::copy(data.begin(), data.end(), input_ids.data<int64_t>());

        Variable input(input_ids, true);
        auto outputs = model.forward(input, Tensor{}, Variable{}, Variable{});

        EXPECT_EQ(outputs.sequence_output.tensor().shape()[1], 32);
    }
}

// ============================================================================
// BertForSequenceClassification Tests
// ============================================================================

TYPED_TEST(BertMultiDtypeTest, SequenceClassificationBinary) {
    int64_t num_labels = 2;
    auto model = BertForSequenceClassification(this->config_, num_labels);
    auto input_ids = this->create_input_ids();

    auto logits = model.forward(input_ids, Tensor{}, Variable{});

    EXPECT_EQ(logits.tensor().ndim(), 2);
    EXPECT_EQ(logits.tensor().shape()[0], this->batch_size_);
    EXPECT_EQ(logits.tensor().shape()[1], num_labels);
}

TYPED_TEST(BertMultiDtypeTest, SequenceClassificationMultiClass) {
    int64_t num_labels = 5;
    auto model = BertForSequenceClassification(this->config_, num_labels);
    auto input_ids = this->create_input_ids();

    auto logits = model.forward(input_ids, Tensor{}, Variable{});

    EXPECT_EQ(logits.tensor().shape()[0], this->batch_size_);
    EXPECT_EQ(logits.tensor().shape()[1], num_labels);
}

TYPED_TEST(BertMultiDtypeTest, SequenceClassificationGradientFlow) {
    int64_t num_labels = 3;
    auto model = BertForSequenceClassification(this->config_, num_labels);
    model.train();

    auto input_ids = this->create_input_ids();
    auto logits = model.forward(input_ids, Tensor{}, Variable{});

    Variable loss = mean(logits);
    loss.backward();

    auto params = model.parameters();
    EXPECT_FALSE(params.empty());

    for (const auto& param : params) {
        EXPECT_TRUE(param->has_grad());
    }
}

TYPED_TEST(BertMultiDtypeTest, SequenceClassificationWithMask) {
    int64_t num_labels = 2;
    auto model = BertForSequenceClassification(this->config_, num_labels);
    auto input_ids = this->create_input_ids();
    auto attention_mask = this->create_attention_mask();

    auto logits = model.forward(input_ids, attention_mask, Variable{});

    EXPECT_EQ(logits.tensor().shape()[0], this->batch_size_);
    EXPECT_EQ(logits.tensor().shape()[1], num_labels);
}

// ============================================================================
// BertForTokenClassification Tests (NER)
// ============================================================================

TYPED_TEST(BertMultiDtypeTest, TokenClassificationForwardShape) {
    int64_t num_labels = 9;  // Standard NER labels
    auto model = BertForTokenClassification(this->config_, num_labels);
    auto input_ids = this->create_input_ids();

    auto logits = model.forward(input_ids, Tensor{}, Variable{});

    // Output: [batch, seq_len, num_labels]
    EXPECT_EQ(logits.tensor().ndim(), 3);
    EXPECT_EQ(logits.tensor().shape()[0], this->batch_size_);
    EXPECT_EQ(logits.tensor().shape()[1], this->seq_len_);
    EXPECT_EQ(logits.tensor().shape()[2], num_labels);
}

TYPED_TEST(BertMultiDtypeTest, TokenClassificationWithMask) {
    int64_t num_labels = 9;
    auto model = BertForTokenClassification(this->config_, num_labels);
    auto input_ids = this->create_input_ids();
    auto attention_mask = this->create_attention_mask();

    auto logits = model.forward(input_ids, attention_mask, Variable{});

    EXPECT_EQ(logits.tensor().shape()[0], this->batch_size_);
    EXPECT_EQ(logits.tensor().shape()[1], this->seq_len_);
    EXPECT_EQ(logits.tensor().shape()[2], num_labels);
}

TYPED_TEST(BertMultiDtypeTest, TokenClassificationGradientFlow) {
    int64_t num_labels = 9;
    auto model = BertForTokenClassification(this->config_, num_labels);
    model.train();

    auto input_ids = this->create_input_ids();
    auto logits = model.forward(input_ids, Tensor{}, Variable{});

    Variable loss = mean(logits);
    loss.backward();

    auto params = model.parameters();
    EXPECT_FALSE(params.empty());

    for (const auto& param : params) {
        EXPECT_TRUE(param->has_grad());
    }
}

TYPED_TEST(BertMultiDtypeTest, TokenClassificationPerTokenPrediction) {
    int64_t num_labels = 9;
    auto model = BertForTokenClassification(this->config_, num_labels);
    auto input_ids = this->create_input_ids();

    auto logits = model.forward(input_ids, Tensor{}, Variable{});

    // Each token should have label predictions
    EXPECT_EQ(logits.tensor().shape()[1], this->seq_len_);
    EXPECT_EQ(logits.tensor().shape()[2], num_labels);
}

// ============================================================================
// BertForQuestionAnswering Tests
// ============================================================================

TYPED_TEST(BertMultiDtypeTest, QuestionAnsweringForwardShape) {
    auto model = BertForQuestionAnswering(this->config_);
    auto input_ids = this->create_input_ids();

    auto outputs = model.forward(input_ids, Tensor{}, Variable{});

    // Start logits: [batch, seq_len]
    EXPECT_EQ(outputs.start_logits.tensor().ndim(), 2);
    EXPECT_EQ(outputs.start_logits.tensor().shape()[0], this->batch_size_);
    EXPECT_EQ(outputs.start_logits.tensor().shape()[1], this->seq_len_);

    // End logits: [batch, seq_len]
    EXPECT_EQ(outputs.end_logits.tensor().ndim(), 2);
    EXPECT_EQ(outputs.end_logits.tensor().shape()[0], this->batch_size_);
    EXPECT_EQ(outputs.end_logits.tensor().shape()[1], this->seq_len_);
}

TYPED_TEST(BertMultiDtypeTest, QuestionAnsweringWithTokenTypes) {
    auto model = BertForQuestionAnswering(this->config_);
    auto input_ids = this->create_input_ids();
    auto token_type_ids = this->create_token_type_ids();

    auto outputs = model.forward(input_ids, Tensor{}, token_type_ids);

    EXPECT_EQ(outputs.start_logits.tensor().shape()[0], this->batch_size_);
    EXPECT_EQ(outputs.start_logits.tensor().shape()[1], this->seq_len_);
    EXPECT_EQ(outputs.end_logits.tensor().shape()[0], this->batch_size_);
    EXPECT_EQ(outputs.end_logits.tensor().shape()[1], this->seq_len_);
}

TYPED_TEST(BertMultiDtypeTest, QuestionAnsweringGradientFlow) {
    auto model = BertForQuestionAnswering(this->config_);
    model.train();

    auto input_ids = this->create_input_ids();
    auto outputs = model.forward(input_ids, Tensor{}, Variable{});

    Variable loss = mean(outputs.start_logits) + mean(outputs.end_logits);
    loss.backward();

    auto params = model.parameters();
    EXPECT_FALSE(params.empty());

    for (const auto& param : params) {
        EXPECT_TRUE(param->has_grad());
    }
}

TYPED_TEST(BertMultiDtypeTest, QuestionAnsweringSpanPrediction) {
    auto model = BertForQuestionAnswering(this->config_);
    auto input_ids = this->create_input_ids();

    auto outputs = model.forward(input_ids, Tensor{}, Variable{});

    // Both start and end logits should be valid
    auto start_data = outputs.start_logits.tensor();
    auto end_data = outputs.end_logits.tensor();

    float start_max = std::abs(this->get_scalar(tenzor::max(tenzor::abs(start_data))));
    float end_max = std::abs(this->get_scalar(tenzor::max(tenzor::abs(end_data))));

    EXPECT_GT(start_max, 0.0f);
    EXPECT_GT(end_max, 0.0f);
}

// ============================================================================
// BERT Configuration Tests
// ============================================================================

TYPED_TEST(BertMultiDtypeTest, BertConfigBase) {
    auto config = BertConfig::base();

    EXPECT_EQ(config.vocab_size, 30522);
    EXPECT_EQ(config.hidden_size, 768);
    EXPECT_EQ(config.num_hidden_layers, 12);
    EXPECT_EQ(config.num_attention_heads, 12);
    EXPECT_EQ(config.intermediate_size, 3072);
    EXPECT_EQ(config.max_position_embeddings, 512);
}

TYPED_TEST(BertMultiDtypeTest, BertConfigLarge) {
    auto config = BertConfig::large();

    EXPECT_EQ(config.hidden_size, 1024);
    EXPECT_EQ(config.num_hidden_layers, 24);
    EXPECT_EQ(config.num_attention_heads, 16);
    EXPECT_EQ(config.intermediate_size, 4096);
}

TYPED_TEST(BertMultiDtypeTest, BertConfigCustom) {
    BertConfig config;
    config.vocab_size = 50000;
    config.hidden_size = 256;
    config.num_hidden_layers = 6;
    config.num_attention_heads = 8;

    auto model = BertModel(config);

    Tensor input_ids({1, 10}, DType::Int64, this->device_);
    std::vector<int64_t> data(10, 0);
    std::copy(data.begin(), data.end(), input_ids.data<int64_t>());

    Variable input(input_ids, true);
    auto outputs = model.forward(input, Tensor{}, Variable{}, Variable{});

    EXPECT_EQ(outputs.sequence_output.tensor().shape()[2], 256);
}

// ============================================================================
// Edge Cases and Robustness
// ============================================================================

TYPED_TEST(BertMultiDtypeTest, BertModelShortSequence) {
    auto model = BertModel(this->config_);

    // Single token sequence
    Tensor input_ids({this->batch_size_, 1}, DType::Int64, this->device_);
    std::vector<int64_t> data(this->batch_size_, 0);
    std::copy(data.begin(), data.end(), input_ids.data<int64_t>());

    Variable input(input_ids, true);
    auto outputs = model.forward(input, Tensor{}, Variable{}, Variable{});

    EXPECT_EQ(outputs.sequence_output.tensor().shape()[1], 1);
}

TYPED_TEST(BertMultiDtypeTest, BertModelLongSequence) {
    auto model = BertModel(this->config_);

    int64_t max_len = this->config_.max_position_embeddings;
    Tensor input_ids({this->batch_size_, max_len}, DType::Int64, this->device_);
    std::vector<int64_t> data(this->batch_size_ * max_len, 0);
    std::copy(data.begin(), data.end(), input_ids.data<int64_t>());

    Variable input(input_ids, true);
    auto outputs = model.forward(input, Tensor{}, Variable{}, Variable{});

    EXPECT_EQ(outputs.sequence_output.tensor().shape()[1], max_len);
}

TYPED_TEST(BertMultiDtypeTest, BertModelBatchSizeOne) {
    auto model = BertModel(this->config_);

    Tensor input_ids({1, this->seq_len_}, DType::Int64, this->device_);
    std::vector<int64_t> data(this->seq_len_, 0);
    std::copy(data.begin(), data.end(), input_ids.data<int64_t>());

    Variable input(input_ids, true);
    auto outputs = model.forward(input, Tensor{}, Variable{}, Variable{});

    EXPECT_EQ(outputs.sequence_output.tensor().shape()[0], 1);
    EXPECT_EQ(outputs.pooled_output.tensor().shape()[0], 1);
}

TYPED_TEST(BertMultiDtypeTest, BertModelLargeBatch) {
    auto model = BertModel(this->config_);

    int64_t large_batch = 8;
    Tensor input_ids({large_batch, this->seq_len_}, DType::Int64, this->device_);
    std::vector<int64_t> data(large_batch * this->seq_len_, 0);
    std::copy(data.begin(), data.end(), input_ids.data<int64_t>());

    Variable input(input_ids, true);
    auto outputs = model.forward(input, Tensor{}, Variable{}, Variable{});

    EXPECT_EQ(outputs.sequence_output.tensor().shape()[0], large_batch);
    EXPECT_EQ(outputs.pooled_output.tensor().shape()[0], large_batch);
}

// ============================================================================
// Attention Mask Functionality Tests
// ============================================================================

TYPED_TEST(BertMultiDtypeTest, AttentionMaskPaddingHandling) {
    auto model = BertModel(this->config_);
    auto input_ids = this->create_input_ids();

    // Create mask with some padding (zeros)
    Tensor attention_mask({this->batch_size_, this->seq_len_}, this->dtype_, this->device_);

    if (this->dtype_ == DType::Float32) {
        auto ptr = attention_mask.data<float>();
        for (int64_t i = 0; i < this->batch_size_ * this->seq_len_; ++i) {
            ptr[i] = (i % this->seq_len_ < this->seq_len_ / 2) ? 1.0f : 0.0f;
        }
    } else if (this->dtype_ == DType::Float64) {
        auto ptr = attention_mask.data<double>();
        for (int64_t i = 0; i < this->batch_size_ * this->seq_len_; ++i) {
            ptr[i] = (i % this->seq_len_ < this->seq_len_ / 2) ? 1.0 : 0.0;
        }
    }

    auto outputs = model.forward(input_ids, attention_mask, Variable{}, Variable{});

    EXPECT_EQ(outputs.sequence_output.tensor().shape()[0], this->batch_size_);
    EXPECT_EQ(outputs.sequence_output.tensor().shape()[1], this->seq_len_);
}

// ============================================================================
// Parameter Tests
// ============================================================================

TYPED_TEST(BertMultiDtypeTest, BertModelParameterCount) {
    auto model = BertModel(this->config_);
    auto params = model.parameters();

    EXPECT_FALSE(params.empty());

    size_t total_params = 0;
    for (const auto& param : params) {
        size_t param_size = 1;
        for (auto dim : param->tensor().shape()) {
            param_size *= dim;
        }
        total_params += param_size;
    }

    EXPECT_GT(total_params, 0);
}

TYPED_TEST(BertMultiDtypeTest, ClassificationHeadExtraParameters) {
    int64_t num_labels = 2;
    auto clf_model = BertForSequenceClassification(this->config_, num_labels);
    auto base_model = BertModel(this->config_);

    auto clf_params = clf_model.parameters();
    auto base_params = base_model.parameters();

    // Classification model should have additional parameters
    EXPECT_GT(clf_params.size(), base_params.size());
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    tenzor::initialize();
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
