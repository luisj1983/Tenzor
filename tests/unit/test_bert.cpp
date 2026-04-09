/**
 * @file test_bert.cpp
 * @brief Comprehensive unit tests for BERT model family
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

using namespace tenzor;
using namespace tenzor::models;

// ============================================================================
// Test Fixtures
// ============================================================================

class BertTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Use small config for fast testing
        config_ = BertConfig();
        config_.vocab_size = 1000;
        config_.hidden_size = 128;
        config_.num_hidden_layers = 2;
        config_.num_attention_heads = 4;
        config_.intermediate_size = 512;
        config_.max_position_embeddings = 64;
        config_.type_vocab_size = 2;

        batch_size_ = 2;
        seq_len_ = 16;
        device_ = Device::cpu();
    }

    BertConfig config_;
    int64_t batch_size_;
    int64_t seq_len_;
    Device device_;

    // Helper to create input token IDs
    auto create_input_ids() -> Variable {
        Tensor input_ids({batch_size_, seq_len_}, DType::Int64, device_);

        // Fill with random token IDs
        std::vector<int64_t> data(batch_size_ * seq_len_);
        for (size_t i = 0; i < data.size(); ++i) {
            data[i] = i % config_.vocab_size;
        }
        std::copy(data.begin(), data.end(), input_ids.data<int64_t>());

        return Variable(input_ids, true);
    }

    // Helper to create attention mask
    auto create_attention_mask() -> Tensor {
        Tensor mask({batch_size_, seq_len_}, DType::Float32, device_);
        mask.fill_(1.0f);  // All valid (no padding)
        return mask;
    }

    // Helper to create token type IDs
    auto create_token_type_ids() -> Variable {
        Tensor type_ids({batch_size_, seq_len_}, DType::Int64, device_);

        // First half is segment 0, second half is segment 1
        std::vector<int64_t> data(batch_size_ * seq_len_);
        for (int64_t b = 0; b < batch_size_; ++b) {
            for (int64_t i = 0; i < seq_len_; ++i) {
                data[b * seq_len_ + i] = (i < seq_len_ / 2) ? 0 : 1;
            }
        }
        std::copy(data.begin(), data.end(), type_ids.data<int64_t>());

        return Variable(type_ids, false);
    }
};

// ============================================================================
// BertEmbeddings Tests
// ============================================================================

TEST_F(BertTest, BertEmbeddingsForwardShape) {
    auto embeddings = BertEmbeddings(config_);
    auto input_ids = create_input_ids();

    auto output = embeddings.forward(input_ids, Variable{}, Variable{});

    // Check output shape: [batch, seq_len, hidden_size]
    EXPECT_EQ(output.tensor().ndim(), 3);
    EXPECT_EQ(output.tensor().shape()[0], batch_size_);
    EXPECT_EQ(output.tensor().shape()[1], seq_len_);
    EXPECT_EQ(output.tensor().shape()[2], config_.hidden_size);
}

TEST_F(BertTest, BertEmbeddingsWithTokenTypes) {
    auto embeddings = BertEmbeddings(config_);
    auto input_ids = create_input_ids();
    auto token_type_ids = create_token_type_ids();

    auto output = embeddings.forward(input_ids, token_type_ids, Variable{});

    // Check output shape
    EXPECT_EQ(output.tensor().shape()[0], batch_size_);
    EXPECT_EQ(output.tensor().shape()[1], seq_len_);
    EXPECT_EQ(output.tensor().shape()[2], config_.hidden_size);
}

TEST_F(BertTest, BertEmbeddingsGradientFlow) {
    auto embeddings = BertEmbeddings(config_);
    embeddings.train();

    auto input_ids = create_input_ids();
    auto output = embeddings.forward(input_ids, Variable{}, Variable{});

    // Check gradient computation works
    EXPECT_TRUE(output.requires_grad());

    // Create dummy loss and backward
    Variable loss = mean(output);  // Use autograd mean to preserve gradient graph
    loss.backward();

    // Check that embeddings have gradients
    auto params = embeddings.parameters();
    EXPECT_FALSE(params.empty());

    for (const auto& param : params) {
        EXPECT_TRUE(param->has_grad());
    }
}

// ============================================================================
// BertEncoder Tests
// ============================================================================

TEST_F(BertTest, BertEncoderForwardShape) {
    auto encoder = BertEncoder(config_);

    // Create input embeddings
    Tensor hidden_states({batch_size_, seq_len_, config_.hidden_size},
                        DType::Float32, device_);
    hidden_states.fill_(0.1f);
    Variable input(hidden_states, true);

    auto output = encoder.forward(input, Tensor{});

    // Check output shape: [batch, seq_len, hidden_size]
    EXPECT_EQ(output.tensor().shape()[0], batch_size_);
    EXPECT_EQ(output.tensor().shape()[1], seq_len_);
    EXPECT_EQ(output.tensor().shape()[2], config_.hidden_size);
}

TEST_F(BertTest, BertEncoderWithMask) {
    auto encoder = BertEncoder(config_);

    Tensor hidden_states({batch_size_, seq_len_, config_.hidden_size},
                        DType::Float32, device_);
    hidden_states.fill_(0.1f);
    Variable input(hidden_states, true);

    auto attention_mask = create_attention_mask();
    auto output = encoder.forward(input, attention_mask);

    // Check output shape matches input
    EXPECT_EQ(output.tensor().shape()[0], batch_size_);
    EXPECT_EQ(output.tensor().shape()[1], seq_len_);
    EXPECT_EQ(output.tensor().shape()[2], config_.hidden_size);
}

TEST_F(BertTest, BertEncoderGradientFlow) {
    auto encoder = BertEncoder(config_);
    encoder.train();

    Tensor hidden_states({batch_size_, seq_len_, config_.hidden_size},
                        DType::Float32, device_);
    hidden_states.fill_(0.1f);
    Variable input(hidden_states, true);

    auto output = encoder.forward(input, Tensor{});
    Variable loss = mean(output);  // Use autograd mean to preserve gradient graph
    loss.backward();

    // Check gradients exist
    auto params = encoder.parameters();
    EXPECT_FALSE(params.empty());

    for (const auto& param : params) {
        EXPECT_TRUE(param->has_grad());
    }
}

// ============================================================================
// BertPooler Tests
// ============================================================================

TEST_F(BertTest, BertPoolerForwardShape) {
    auto pooler = BertPooler(config_);

    Tensor hidden_states({batch_size_, seq_len_, config_.hidden_size},
                        DType::Float32, device_);
    hidden_states.fill_(0.1f);
    Variable input(hidden_states, true);

    auto output = pooler.forward(input);

    // Check output shape: [batch, hidden_size]
    EXPECT_EQ(output.tensor().ndim(), 2);
    EXPECT_EQ(output.tensor().shape()[0], batch_size_);
    EXPECT_EQ(output.tensor().shape()[1], config_.hidden_size);
}

TEST_F(BertTest, BertPoolerTanhActivation) {
    auto pooler = BertPooler(config_);

    Tensor hidden_states({batch_size_, seq_len_, config_.hidden_size},
                        DType::Float32, device_);
    hidden_states.fill_(0.1f);
    Variable input(hidden_states, true);

    auto output = pooler.forward(input);

    // Check that output is in tanh range [-1, 1]
    auto output_data = output.tensor();
    float max_val = tenzor::max(output_data).item<float>();
    float min_val = tenzor::min(output_data).item<float>();

    EXPECT_LE(max_val, 1.0f);
    EXPECT_GE(min_val, -1.0f);
}

// ============================================================================
// BertModel Tests
// ============================================================================

TEST_F(BertTest, BertModelForwardShape) {
    auto model = BertModel(config_);
    auto input_ids = create_input_ids();

    auto outputs = model.forward(input_ids, Tensor{}, Variable{}, Variable{});

    // Check sequence output shape: [batch, seq_len, hidden_size]
    EXPECT_EQ(outputs.sequence_output.tensor().shape()[0], batch_size_);
    EXPECT_EQ(outputs.sequence_output.tensor().shape()[1], seq_len_);
    EXPECT_EQ(outputs.sequence_output.tensor().shape()[2], config_.hidden_size);

    // Check pooled output shape: [batch, hidden_size]
    EXPECT_EQ(outputs.pooled_output.tensor().shape()[0], batch_size_);
    EXPECT_EQ(outputs.pooled_output.tensor().shape()[1], config_.hidden_size);
}

TEST_F(BertTest, BertModelWithAllInputs) {
    auto model = BertModel(config_);
    auto input_ids = create_input_ids();
    auto attention_mask = create_attention_mask();
    auto token_type_ids = create_token_type_ids();

    auto outputs = model.forward(input_ids, attention_mask, token_type_ids, Variable{});

    // Check output shapes
    EXPECT_EQ(outputs.sequence_output.tensor().shape()[0], batch_size_);
    EXPECT_EQ(outputs.sequence_output.tensor().shape()[1], seq_len_);
    EXPECT_EQ(outputs.sequence_output.tensor().shape()[2], config_.hidden_size);

    EXPECT_EQ(outputs.pooled_output.tensor().shape()[0], batch_size_);
    EXPECT_EQ(outputs.pooled_output.tensor().shape()[1], config_.hidden_size);
}

TEST_F(BertTest, BertModelGradientFlow) {
    auto model = BertModel(config_);
    model.train();

    auto input_ids = create_input_ids();
    auto outputs = model.forward(input_ids, Tensor{}, Variable{}, Variable{});

    // Compute loss from both outputs
    Variable loss = mean(outputs.sequence_output) + mean(outputs.pooled_output);  // Use autograd mean
    loss.backward();

    // Check gradients exist
    auto params = model.parameters();
    EXPECT_FALSE(params.empty());

    for (const auto& param : params) {
        EXPECT_TRUE(param->has_grad());
    }
}

TEST_F(BertTest, BertModelTrainEvalMode) {
    auto model = BertModel(config_);
    auto input_ids = create_input_ids();

    // Test training mode
    model.train();
    EXPECT_TRUE(model.is_training());
    auto train_output = model.forward(input_ids, Tensor{}, Variable{}, Variable{});

    // Test eval mode
    model.eval();
    EXPECT_FALSE(model.is_training());
    auto eval_output = model.forward(input_ids, Tensor{}, Variable{}, Variable{});

    // Both should produce valid outputs
    EXPECT_EQ(train_output.sequence_output.tensor().shape()[0], batch_size_);
    EXPECT_EQ(eval_output.sequence_output.tensor().shape()[0], batch_size_);
}

// ============================================================================
// BertForSequenceClassification Tests
// ============================================================================

TEST_F(BertTest, SequenceClassificationForwardShape) {
    int64_t num_labels = 2;  // Binary classification
    auto model = BertForSequenceClassification(config_, num_labels);
    auto input_ids = create_input_ids();

    auto logits = model.forward(input_ids, Tensor{}, Variable{});

    // Check output shape: [batch, num_labels]
    EXPECT_EQ(logits.tensor().ndim(), 2);
    EXPECT_EQ(logits.tensor().shape()[0], batch_size_);
    EXPECT_EQ(logits.tensor().shape()[1], num_labels);
}

TEST_F(BertTest, SequenceClassificationMultiClass) {
    int64_t num_labels = 5;  // 5-way classification
    auto model = BertForSequenceClassification(config_, num_labels);
    auto input_ids = create_input_ids();

    auto logits = model.forward(input_ids, Tensor{}, Variable{});

    EXPECT_EQ(logits.tensor().shape()[0], batch_size_);
    EXPECT_EQ(logits.tensor().shape()[1], num_labels);
}

TEST_F(BertTest, SequenceClassificationGradientFlow) {
    int64_t num_labels = 3;
    auto model = BertForSequenceClassification(config_, num_labels);
    model.train();

    auto input_ids = create_input_ids();
    auto logits = model.forward(input_ids, Tensor{}, Variable{});

    // Compute loss
    Variable loss = mean(logits);  // Use autograd mean to preserve gradient graph
    loss.backward();

    // Check gradients
    auto params = model.parameters();
    EXPECT_FALSE(params.empty());

    for (const auto& param : params) {
        EXPECT_TRUE(param->has_grad());
    }
}

// ============================================================================
// BertForTokenClassification Tests
// ============================================================================

TEST_F(BertTest, TokenClassificationForwardShape) {
    int64_t num_labels = 9;  // NER with 9 tags
    auto model = BertForTokenClassification(config_, num_labels);
    auto input_ids = create_input_ids();

    auto logits = model.forward(input_ids, Tensor{}, Variable{});

    // Check output shape: [batch, seq_len, num_labels]
    EXPECT_EQ(logits.tensor().ndim(), 3);
    EXPECT_EQ(logits.tensor().shape()[0], batch_size_);
    EXPECT_EQ(logits.tensor().shape()[1], seq_len_);
    EXPECT_EQ(logits.tensor().shape()[2], num_labels);
}

TEST_F(BertTest, TokenClassificationWithMask) {
    int64_t num_labels = 9;
    auto model = BertForTokenClassification(config_, num_labels);
    auto input_ids = create_input_ids();
    auto attention_mask = create_attention_mask();

    auto logits = model.forward(input_ids, attention_mask, Variable{});

    EXPECT_EQ(logits.tensor().shape()[0], batch_size_);
    EXPECT_EQ(logits.tensor().shape()[1], seq_len_);
    EXPECT_EQ(logits.tensor().shape()[2], num_labels);
}

TEST_F(BertTest, TokenClassificationGradientFlow) {
    int64_t num_labels = 9;
    auto model = BertForTokenClassification(config_, num_labels);
    model.train();

    auto input_ids = create_input_ids();
    auto logits = model.forward(input_ids, Tensor{}, Variable{});

    Variable loss = mean(logits);  // Use autograd mean to preserve gradient graph
    loss.backward();

    auto params = model.parameters();
    EXPECT_FALSE(params.empty());

    for (const auto& param : params) {
        EXPECT_TRUE(param->has_grad());
    }
}

// ============================================================================
// BertForQuestionAnswering Tests
// ============================================================================

TEST_F(BertTest, QuestionAnsweringForwardShape) {
    auto model = BertForQuestionAnswering(config_);
    auto input_ids = create_input_ids();

    auto outputs = model.forward(input_ids, Tensor{}, Variable{});

    // Check start logits shape: [batch, seq_len]
    EXPECT_EQ(outputs.start_logits.tensor().ndim(), 2);
    EXPECT_EQ(outputs.start_logits.tensor().shape()[0], batch_size_);
    EXPECT_EQ(outputs.start_logits.tensor().shape()[1], seq_len_);

    // Check end logits shape: [batch, seq_len]
    EXPECT_EQ(outputs.end_logits.tensor().ndim(), 2);
    EXPECT_EQ(outputs.end_logits.tensor().shape()[0], batch_size_);
    EXPECT_EQ(outputs.end_logits.tensor().shape()[1], seq_len_);
}

TEST_F(BertTest, QuestionAnsweringWithTokenTypes) {
    auto model = BertForQuestionAnswering(config_);
    auto input_ids = create_input_ids();
    auto token_type_ids = create_token_type_ids();

    // First segment (0) is question, second segment (1) is context
    auto outputs = model.forward(input_ids, Tensor{}, token_type_ids);

    EXPECT_EQ(outputs.start_logits.tensor().shape()[0], batch_size_);
    EXPECT_EQ(outputs.end_logits.tensor().shape()[0], batch_size_);
}

TEST_F(BertTest, QuestionAnsweringGradientFlow) {
    auto model = BertForQuestionAnswering(config_);
    model.train();

    auto input_ids = create_input_ids();
    auto outputs = model.forward(input_ids, Tensor{}, Variable{});

    // Compute loss from both start and end logits
    Variable loss = mean(outputs.start_logits) + mean(outputs.end_logits);  // Use autograd mean
    loss.backward();

    auto params = model.parameters();
    EXPECT_FALSE(params.empty());

    for (const auto& param : params) {
        EXPECT_TRUE(param->has_grad());
    }
}

// ============================================================================
// Configuration Tests
// ============================================================================

TEST_F(BertTest, BertConfigBase) {
    auto config = BertConfig::base();

    EXPECT_EQ(config.vocab_size, 30522);
    EXPECT_EQ(config.hidden_size, 768);
    EXPECT_EQ(config.num_hidden_layers, 12);
    EXPECT_EQ(config.num_attention_heads, 12);
    EXPECT_EQ(config.intermediate_size, 3072);
}

TEST_F(BertTest, BertConfigLarge) {
    auto config = BertConfig::large();

    EXPECT_EQ(config.hidden_size, 1024);
    EXPECT_EQ(config.num_hidden_layers, 24);
    EXPECT_EQ(config.num_attention_heads, 16);
    EXPECT_EQ(config.intermediate_size, 4096);
}

// ============================================================================
// Parameter Counting Tests
// ============================================================================

TEST_F(BertTest, BertModelParameterCount) {
    auto model = BertModel(config_);
    auto params = model.parameters();

    // Should have parameters from embeddings, encoder, and pooler
    EXPECT_FALSE(params.empty());

    // Count total parameters
    size_t total_params = 0;
    for (const auto& param : params) {
        size_t param_size = 1;
        for (auto dim : param->tensor().shape()) {
            param_size *= dim;
        }
        total_params += param_size;
    }

    // Should have a reasonable number of parameters
    EXPECT_GT(total_params, 0);
}

TEST_F(BertTest, SequenceClassificationExtraParameters) {
    int64_t num_labels = 2;
    auto model = BertForSequenceClassification(config_, num_labels);
    auto params = model.parameters();

    // Should have more parameters than base BERT (due to classifier head)
    auto base_model = BertModel(config_);
    auto base_params = base_model.parameters();

    EXPECT_GT(params.size(), base_params.size());
}

// ============================================================================
// Device Tests
// ============================================================================

TEST_F(BertTest, BertModelCPU) {
    auto model = BertModel(config_);
    model.cpu();

    auto input_ids = create_input_ids();
    auto outputs = model.forward(input_ids, Tensor{}, Variable{}, Variable{});

    // Check output device is CPU
    EXPECT_EQ(outputs.sequence_output.tensor().device().type, Device::Type::CPU);
    EXPECT_EQ(outputs.pooled_output.tensor().device().type, Device::Type::CPU);
}

#ifdef TENZOR_CUDA_AVAILABLE
TEST_F(BertTest, BertModelCUDA) {
    if (!Device::cuda(0).is_available()) {
        GTEST_SKIP() << "CUDA not available";
    }

    auto model = BertModel(config_);
    model.cuda(0);

    // Create input on CUDA
    Tensor input_ids({batch_size_, seq_len_}, DType::Int64, Device::cuda(0));
    // ... fill with data ...
    Variable input(input_ids, true);

    auto outputs = model.forward(input, Tensor{}, Variable{}, Variable{});

    // Check output device is CUDA
    EXPECT_EQ(outputs.sequence_output.tensor().device().type, Device::Type::CUDA);
    EXPECT_EQ(outputs.pooled_output.tensor().device().type, Device::Type::CUDA);
}
#endif

// ============================================================================
// Serialization Tests
// ============================================================================

TEST_F(BertTest, BertModelStateDictSaveLoad) {
    auto model1 = BertModel(config_);
    auto model2 = BertModel(config_);

    // Save state from model1
    auto state = model1.state_dict();
    EXPECT_FALSE(state.empty());

    // Load into model2
    model2.load_state_dict(state);

    // Verify that all parameters were correctly loaded by comparing parameter values directly
    auto params1 = model1.parameters();
    auto params2 = model2.parameters();

    EXPECT_EQ(params1.size(), params2.size());

    // Check that all parameters match exactly
    for (size_t i = 0; i < params1.size(); ++i) {
        auto& p1 = params1[i]->tensor();
        auto& p2 = params2[i]->tensor();

        EXPECT_EQ(p1.shape().size(), p2.shape().size());
        for (size_t j = 0; j < p1.shape().size(); ++j) {
            EXPECT_EQ(p1.shape()[j], p2.shape()[j]);
        }

        // Parameters should be exactly equal (not just similar)
        auto diff = tenzor::abs(p1 - p2);
        EXPECT_LT(tenzor::max(diff).item<float>(), 1e-7)
            << "Parameter " << i << " differs after load_state_dict";
    }
}

// ============================================================================
// ModelHub Tests (basic)
// ============================================================================

TEST_F(BertTest, ModelHubParameterNameMapping) {
    // Test parameter name mapping
    auto mapped = BertModelHub::map_parameter_name("bert.embeddings.word_embeddings.weight");
    EXPECT_EQ(mapped, "embeddings.word_embeddings.weight");

    mapped = BertModelHub::map_parameter_name("bert.encoder.layer.0.attention.self.query.weight");
    EXPECT_NE(mapped.find("encoder.layers.0"), std::string::npos);
}

TEST_F(BertTest, ModelHubDownloadNotImplemented) {
    // Should throw error since downloading is not yet implemented
    EXPECT_THROW(
        BertModelHub::download_model("bert-base-uncased"),
        std::runtime_error
    );
}

TEST_F(BertTest, ModelHubLoadCheckpointNotImplemented) {
    // Should throw error since checkpoint loading is not yet implemented
    EXPECT_THROW(
        BertModelHub::load_pytorch_checkpoint("dummy_path.bin"),
        std::runtime_error
    );
}

// ============================================================================
// Edge Case Tests
// ============================================================================

TEST_F(BertTest, BertModelShortSequence) {
    auto model = BertModel(config_);

    // Very short sequence (1 token)
    Tensor input_ids({batch_size_, 1}, DType::Int64, device_);
    std::vector<int64_t> data(batch_size_, 0);
    std::copy(data.begin(), data.end(), input_ids.data<int64_t>());

    Variable input(input_ids, true);
    auto outputs = model.forward(input, Tensor{}, Variable{}, Variable{});

    EXPECT_EQ(outputs.sequence_output.tensor().shape()[1], 1);
}

TEST_F(BertTest, BertModelLongSequence) {
    auto model = BertModel(config_);

    // Maximum length sequence
    int64_t max_len = config_.max_position_embeddings;
    Tensor input_ids({batch_size_, max_len}, DType::Int64, device_);
    std::vector<int64_t> data(batch_size_ * max_len, 0);
    std::copy(data.begin(), data.end(), input_ids.data<int64_t>());

    Variable input(input_ids, true);
    auto outputs = model.forward(input, Tensor{}, Variable{}, Variable{});

    EXPECT_EQ(outputs.sequence_output.tensor().shape()[1], max_len);
}

TEST_F(BertTest, BertModelBatchSizeOne) {
    auto model = BertModel(config_);

    Tensor input_ids({1, seq_len_}, DType::Int64, device_);
    std::vector<int64_t> data(seq_len_, 0);
    std::copy(data.begin(), data.end(), input_ids.data<int64_t>());

    Variable input(input_ids, true);
    auto outputs = model.forward(input, Tensor{}, Variable{}, Variable{});

    EXPECT_EQ(outputs.sequence_output.tensor().shape()[0], 1);
    EXPECT_EQ(outputs.pooled_output.tensor().shape()[0], 1);
}

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
