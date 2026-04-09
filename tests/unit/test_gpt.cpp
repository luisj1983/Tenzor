/**
 * @file test_gpt.cpp
 * @brief Comprehensive unit tests for GPT models
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include "../../include/tenzor/models/gpt.hpp"
#include "../../include/tenzor/autograd/variable.hpp"
#include "../../include/tenzor/core/tensor.hpp"
#include "../../include/tenzor/ops/reduction.hpp"

using namespace tenzor;
using namespace tenzor::models;

class GPTTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Use small configs for testing
        config_ = GPT2Config{};
        config_.vocab_size = 1000;
        config_.n_positions = 128;
        config_.n_embd = 64;
        config_.n_layer = 2;
        config_.n_head = 4;
        config_.n_inner = 256;
        config_.attn_pdrop = 0.0;  // Disable dropout for deterministic tests
        config_.embd_pdrop = 0.0;
        config_.resid_pdrop = 0.0;

        batch_size_ = 2;
        seq_len_ = 16;
        device_ = Device::cpu();
    }

    GPT2Config config_;
    int64_t batch_size_;
    int64_t seq_len_;
    Device device_;
};

// ============================================================================
// GPTEmbeddings Tests
// ============================================================================

TEST_F(GPTTest, EmbeddingsForwardShape) {
    GPTEmbeddings embeddings(config_);
    embeddings.train(false);

    // Create input token IDs
    Tensor input_ids({batch_size_, seq_len_}, DType::Int64, device_);
    auto data = input_ids.data<int64_t>();
    for (int64_t i = 0; i < input_ids.numel(); ++i) {
        data[i] = i % config_.vocab_size;
    }

    Variable input_var(input_ids, false);
    auto output = embeddings.forward(input_var, Variable{});

    // Check output shape
    EXPECT_EQ(output.tensor().ndim(), 3);
    EXPECT_EQ(output.tensor().shape()[0], batch_size_);
    EXPECT_EQ(output.tensor().shape()[1], seq_len_);
    EXPECT_EQ(output.tensor().shape()[2], config_.n_embd);
}

TEST_F(GPTTest, EmbeddingsWithPositionIds) {
    GPTEmbeddings embeddings(config_);
    embeddings.train(false);

    // Create input token IDs and position IDs
    Tensor input_ids({batch_size_, seq_len_}, DType::Int64, device_);
    Tensor position_ids({batch_size_, seq_len_}, DType::Int64, device_);

    auto input_data = input_ids.data<int64_t>();
    auto pos_data = position_ids.data<int64_t>();

    for (int64_t b = 0; b < batch_size_; ++b) {
        for (int64_t s = 0; s < seq_len_; ++s) {
            input_data[b * seq_len_ + s] = s % config_.vocab_size;
            pos_data[b * seq_len_ + s] = s;
        }
    }

    Variable input_var(input_ids, false);
    Variable pos_var(position_ids, false);

    auto output = embeddings.forward(input_var, pos_var);

    EXPECT_EQ(output.tensor().shape()[0], batch_size_);
    EXPECT_EQ(output.tensor().shape()[1], seq_len_);
    EXPECT_EQ(output.tensor().shape()[2], config_.n_embd);
}

TEST_F(GPTTest, EmbeddingsGradientFlow) {
    GPTEmbeddings embeddings(config_);
    embeddings.train();

    Tensor input_ids({batch_size_, seq_len_}, DType::Int64, device_);
    auto data = input_ids.data<int64_t>();
    for (int64_t i = 0; i < input_ids.numel(); ++i) {
        data[i] = i % config_.vocab_size;
    }

    Variable input_var(input_ids, false);
    auto output = embeddings.forward(input_var, Variable{});

    // Check that output requires grad (from embeddings)
    EXPECT_TRUE(output.requires_grad());

    // Create simple loss and backward
    auto loss = tenzor::sum(output);
    loss.backward();

    // Check that embeddings have gradients
    auto params = embeddings.parameters();
    EXPECT_GT(params.size(), 0);

    for (const auto& param : params) {
        if (param->requires_grad()) {
            auto grad = param->grad();
            EXPECT_TRUE(grad.has_value());
        }
    }
}

// ============================================================================
// GPTDecoderLayer Tests
// ============================================================================

TEST_F(GPTTest, DecoderLayerForwardShape) {
    GPTDecoderLayer layer(config_);
    layer.train(false);

    Tensor hidden_states({batch_size_, seq_len_, config_.n_embd}, DType::Float32, device_);
    hidden_states.fill_(0.1f);

    Variable input_var(hidden_states, false);
    auto output = layer.forward(input_var, Tensor{}, false);

    // Output should have same shape as input
    EXPECT_EQ(output.tensor().shape()[0], batch_size_);
    EXPECT_EQ(output.tensor().shape()[1], seq_len_);
    EXPECT_EQ(output.tensor().shape()[2], config_.n_embd);
}

TEST_F(GPTTest, DecoderLayerWithCausalMask) {
    GPTDecoderLayer layer(config_);
    layer.train(false);

    Tensor hidden_states({batch_size_, seq_len_, config_.n_embd}, DType::Float32, device_);
    hidden_states.fill_(0.1f);

    // Create causal mask
    Tensor causal_mask = nn::create_causal_mask(seq_len_, device_);

    Variable input_var(hidden_states, false);
    auto output = layer.forward(input_var, causal_mask);

    EXPECT_EQ(output.tensor().shape()[0], batch_size_);
    EXPECT_EQ(output.tensor().shape()[1], seq_len_);
    EXPECT_EQ(output.tensor().shape()[2], config_.n_embd);
}

TEST_F(GPTTest, DecoderLayerGradientFlow) {
    GPTDecoderLayer layer(config_);
    layer.train();

    Tensor hidden_states({batch_size_, seq_len_, config_.n_embd}, DType::Float32, device_);
    hidden_states.fill_(0.1f);

    Variable input_var(hidden_states, true);
    auto output = layer.forward(input_var, Tensor{}, false);

    auto loss = tenzor::sum(output);
    loss.backward();

    // Check gradients
    auto params = layer.parameters();
    EXPECT_GT(params.size(), 0);

    int grad_count = 0;
    for (const auto& param : params) {
        if (param->requires_grad()) {
            auto grad = param->grad();
            if (grad.has_value()) {
                grad_count++;
            }
        }
    }
    EXPECT_GT(grad_count, 0);
}

// ============================================================================
// GPT2Model Tests
// ============================================================================

TEST_F(GPTTest, GPT2ModelForwardShape) {
    GPT2Model model(config_);
    model.train(false);

    Tensor input_ids({batch_size_, seq_len_}, DType::Int64, device_);
    auto data = input_ids.data<int64_t>();
    for (int64_t i = 0; i < input_ids.numel(); ++i) {
        data[i] = i % config_.vocab_size;
    }

    Variable input_var(input_ids, false);
    auto output = model.forward(input_var, Variable{}, Tensor{});

    // Output should be [batch, seq_len, n_embd]
    EXPECT_EQ(output.tensor().shape()[0], batch_size_);
    EXPECT_EQ(output.tensor().shape()[1], seq_len_);
    EXPECT_EQ(output.tensor().shape()[2], config_.n_embd);
}

TEST_F(GPTTest, GPT2ModelParameterCount) {
    GPT2Model model(config_);

    auto params = model.parameters();
    EXPECT_GT(params.size(), 0);

    // Count total parameters
    int64_t total_params = 0;
    for (const auto& param : params) {
        total_params += param->tensor().numel();
    }

    EXPECT_GT(total_params, 0);

    // Approximate expected parameter count:
    // Token embeddings: vocab_size * n_embd
    // Position embeddings: n_positions * n_embd
    // Layers: n_layer * (attention + FFN parameters)
    // This is a rough check
    int64_t min_expected = config_.vocab_size * config_.n_embd +
                          config_.n_positions * config_.n_embd;
    EXPECT_GT(total_params, min_expected);
}

TEST_F(GPTTest, GPT2ModelGradientFlow) {
    GPT2Model model(config_);
    model.train();

    Tensor input_ids({batch_size_, seq_len_}, DType::Int64, device_);
    auto data = input_ids.data<int64_t>();
    for (int64_t i = 0; i < input_ids.numel(); ++i) {
        data[i] = i % config_.vocab_size;
    }

    Variable input_var(input_ids, false);
    auto output = model.forward(input_var, Variable{}, Tensor{});

    auto loss = tenzor::sum(output);
    loss.backward();

    // Verify gradients exist
    auto params = model.parameters();
    int grad_count = 0;
    for (const auto& param : params) {
        if (param->requires_grad()) {
            auto grad = param->grad();
            if (grad.has_value()) {
                grad_count++;
            }
        }
    }
    EXPECT_GT(grad_count, 0);
}

// ============================================================================
// GPT2LMHeadModel Tests
// ============================================================================

TEST_F(GPTTest, GPT2LMHeadForwardShape) {
    GPT2LMHeadModel model(config_);
    model.train(false);

    Tensor input_ids({batch_size_, seq_len_}, DType::Int64, device_);
    auto data = input_ids.data<int64_t>();
    for (int64_t i = 0; i < input_ids.numel(); ++i) {
        data[i] = i % config_.vocab_size;
    }

    Variable input_var(input_ids, false);
    auto logits = model.forward(input_var, Variable{}, Tensor{});

    // Output should be [batch, seq_len, vocab_size]
    EXPECT_EQ(logits.tensor().shape()[0], batch_size_);
    EXPECT_EQ(logits.tensor().shape()[1], seq_len_);
    EXPECT_EQ(logits.tensor().shape()[2], config_.vocab_size);
}

TEST_F(GPTTest, GPT2LMHeadLogitsRange) {
    GPT2LMHeadModel model(config_);
    model.train(false);

    Tensor input_ids({1, 4}, DType::Int64, device_);
    auto data = input_ids.data<int64_t>();
    data[0] = 10; data[1] = 20; data[2] = 30; data[3] = 40;

    Variable input_var(input_ids, false);
    auto logits = model.forward(input_var, Variable{}, Tensor{});

    // Check that logits are reasonable (not NaN or inf)
    auto logits_data = logits.tensor().data<float>();
    for (int64_t i = 0; i < logits.tensor().numel(); ++i) {
        EXPECT_FALSE(std::isnan(logits_data[i]));
        EXPECT_FALSE(std::isinf(logits_data[i]));
    }
}

TEST_F(GPTTest, GPT2LMHeadGradientFlow) {
    GPT2LMHeadModel model(config_);
    model.train();

    Tensor input_ids({batch_size_, seq_len_}, DType::Int64, device_);
    auto data = input_ids.data<int64_t>();
    for (int64_t i = 0; i < input_ids.numel(); ++i) {
        data[i] = i % config_.vocab_size;
    }

    Variable input_var(input_ids, false);
    auto logits = model.forward(input_var, Variable{}, Tensor{});

    auto loss = tenzor::sum(logits);
    loss.backward();

    // Verify gradients
    auto params = model.parameters();
    int grad_count = 0;
    for (const auto& param : params) {
        if (param->requires_grad()) {
            auto grad = param->grad();
            if (grad.has_value()) {
                grad_count++;
            }
        }
    }
    EXPECT_GT(grad_count, 0);
}

// ============================================================================
// GPT3Model Tests
// ============================================================================

TEST_F(GPTTest, GPT3ModelConstruction) {
    auto gpt3_config = GPT3Config::gpt3_small();
    gpt3_config.vocab_size = 1000;  // Reduce for test
    gpt3_config.n_positions = 64;
    gpt3_config.n_layer = 2;

    GPT3Model model(gpt3_config);

    auto params = model.parameters();
    EXPECT_GT(params.size(), 0);
}

TEST_F(GPTTest, GPT3LMHeadModelForward) {
    auto gpt3_config = GPT3Config::gpt3_small();
    gpt3_config.vocab_size = 1000;
    gpt3_config.n_positions = 64;
    gpt3_config.n_layer = 2;

    GPT3LMHeadModel model(gpt3_config);
    model.train(false);

    Tensor input_ids({1, 8}, DType::Int64, device_);
    auto data = input_ids.data<int64_t>();
    for (int64_t i = 0; i < 8; ++i) {
        data[i] = i * 10;
    }

    Variable input_var(input_ids, false);
    auto logits = model.forward(input_var, Variable{}, Tensor{});

    EXPECT_EQ(logits.tensor().shape()[0], 1);
    EXPECT_EQ(logits.tensor().shape()[1], 8);
    EXPECT_EQ(logits.tensor().shape()[2], gpt3_config.vocab_size);
}

// ============================================================================
// TextGenerator Tests
// ============================================================================

TEST_F(GPTTest, GeneratorGreedySearch) {
    GPT2LMHeadModel model(config_);
    model.train(false);

    GenerationConfig gen_config;
    gen_config.max_length = 20;
    gen_config.strategy = GenerationStrategy::Greedy;

    TextGenerator generator(model, gen_config);

    Tensor input_ids({1, 4}, DType::Int64, device_);
    auto data = input_ids.data<int64_t>();
    data[0] = 10; data[1] = 20; data[2] = 30; data[3] = 40;

    auto output = generator.greedy_search(input_ids);

    EXPECT_EQ(output.shape()[0], 1);
    EXPECT_EQ(output.shape()[1], gen_config.max_length);

    // Check that first tokens match input
    auto output_data = output.data<int64_t>();
    EXPECT_EQ(output_data[0], 10);
    EXPECT_EQ(output_data[1], 20);
    EXPECT_EQ(output_data[2], 30);
    EXPECT_EQ(output_data[3], 40);
}

TEST_F(GPTTest, GeneratorTopKSampling) {
    GPT2LMHeadModel model(config_);
    model.train(false);

    GenerationConfig gen_config;
    gen_config.max_length = 15;
    gen_config.strategy = GenerationStrategy::TopK;
    gen_config.top_k = 10;
    gen_config.temperature = 1.0;
    gen_config.seed = 42;

    TextGenerator generator(model, gen_config);

    Tensor input_ids({1, 3}, DType::Int64, device_);
    auto data = input_ids.data<int64_t>();
    data[0] = 5; data[1] = 15; data[2] = 25;

    auto output = generator.top_k_sampling(input_ids, gen_config.top_k, gen_config.temperature);

    EXPECT_EQ(output.shape()[0], 1);
    EXPECT_EQ(output.shape()[1], gen_config.max_length);

    // Verify input tokens are preserved
    auto output_data = output.data<int64_t>();
    EXPECT_EQ(output_data[0], 5);
    EXPECT_EQ(output_data[1], 15);
    EXPECT_EQ(output_data[2], 25);
}

TEST_F(GPTTest, GeneratorTopPSampling) {
    GPT2LMHeadModel model(config_);
    model.train(false);

    GenerationConfig gen_config;
    gen_config.max_length = 15;
    gen_config.strategy = GenerationStrategy::TopP;
    gen_config.top_p = 0.9;
    gen_config.temperature = 0.8;
    gen_config.seed = 123;

    TextGenerator generator(model, gen_config);

    Tensor input_ids({1, 3}, DType::Int64, device_);
    auto data = input_ids.data<int64_t>();
    data[0] = 7; data[1] = 17; data[2] = 27;

    auto output = generator.top_p_sampling(input_ids, gen_config.top_p, gen_config.temperature);

    EXPECT_EQ(output.shape()[0], 1);
    EXPECT_EQ(output.shape()[1], gen_config.max_length);
}

TEST_F(GPTTest, GeneratorBeamSearch) {
    GPT2LMHeadModel model(config_);
    model.train(false);

    GenerationConfig gen_config;
    gen_config.max_length = 12;
    gen_config.strategy = GenerationStrategy::BeamSearch;
    gen_config.num_beams = 3;

    TextGenerator generator(model, gen_config);

    Tensor input_ids({1, 2}, DType::Int64, device_);
    auto data = input_ids.data<int64_t>();
    data[0] = 100; data[1] = 200;

    auto output = generator.beam_search(input_ids, gen_config.num_beams);

    EXPECT_EQ(output.shape()[0], 1);
    EXPECT_EQ(output.shape()[1], gen_config.max_length);

    // Check input tokens preserved
    auto output_data = output.data<int64_t>();
    EXPECT_EQ(output_data[0], 100);
    EXPECT_EQ(output_data[1], 200);
}

TEST_F(GPTTest, GeneratorGenericGenerate) {
    GPT2LMHeadModel model(config_);
    model.train(false);

    GenerationConfig gen_config;
    gen_config.max_length = 10;
    gen_config.strategy = GenerationStrategy::Greedy;

    TextGenerator generator(model, gen_config);

    Tensor input_ids({1, 2}, DType::Int64, device_);
    auto data = input_ids.data<int64_t>();
    data[0] = 50; data[1] = 60;

    auto output = generator.generate(input_ids);

    EXPECT_EQ(output.shape()[0], 1);
    EXPECT_EQ(output.shape()[1], gen_config.max_length);
}

// ============================================================================
// Configuration Tests
// ============================================================================

TEST_F(GPTTest, GPT2ConfigPresets) {
    auto small = GPT2Config::gpt2_small();
    EXPECT_EQ(small.n_embd, 768);
    EXPECT_EQ(small.n_layer, 12);
    EXPECT_EQ(small.n_head, 12);

    auto medium = GPT2Config::gpt2_medium();
    EXPECT_EQ(medium.n_embd, 1024);
    EXPECT_EQ(medium.n_layer, 24);

    auto large = GPT2Config::gpt2_large();
    EXPECT_EQ(large.n_embd, 1280);
    EXPECT_EQ(large.n_layer, 36);

    auto xl = GPT2Config::gpt2_xl();
    EXPECT_EQ(xl.n_embd, 1600);
    EXPECT_EQ(xl.n_layer, 48);
}

TEST_F(GPTTest, GPT3ConfigPresets) {
    auto small = GPT3Config::gpt3_small();
    EXPECT_EQ(small.n_embd, 768);

    auto medium = GPT3Config::gpt3_medium();
    EXPECT_EQ(medium.n_embd, 1024);

    auto large = GPT3Config::gpt3_large();
    EXPECT_EQ(large.n_embd, 1536);

    auto xl = GPT3Config::gpt3_xl();
    EXPECT_EQ(xl.n_embd, 2048);

    auto b2_7 = GPT3Config::gpt3_2_7b();
    EXPECT_EQ(b2_7.n_embd, 2560);

    auto b6_7 = GPT3Config::gpt3_6_7b();
    EXPECT_EQ(b6_7.n_embd, 4096);

    auto b13 = GPT3Config::gpt3_13b();
    EXPECT_EQ(b13.n_embd, 5120);

    auto b175 = GPT3Config::gpt3_175b();
    EXPECT_EQ(b175.n_embd, 12288);
}

// ============================================================================
// Integration Tests
// ============================================================================

TEST_F(GPTTest, EndToEndTextGeneration) {
    // Small model for fast testing
    config_.n_layer = 1;

    GPT2LMHeadModel model(config_);
    model.train(false);

    GenerationConfig gen_config;
    gen_config.max_length = 10;
    gen_config.strategy = GenerationStrategy::Greedy;

    TextGenerator generator(model, gen_config);

    // Create input
    Tensor input_ids({1, 3}, DType::Int64, device_);
    auto data = input_ids.data<int64_t>();
    data[0] = 1; data[1] = 2; data[2] = 3;

    // Generate
    auto output = generator.generate(input_ids);

    // Verify output shape and content
    EXPECT_EQ(output.shape()[0], 1);
    EXPECT_EQ(output.shape()[1], gen_config.max_length);

    auto output_data = output.data<int64_t>();

    // Input tokens should be preserved
    EXPECT_EQ(output_data[0], 1);
    EXPECT_EQ(output_data[1], 2);
    EXPECT_EQ(output_data[2], 3);

    // Generated tokens should be valid (< vocab_size)
    for (int64_t i = 3; i < gen_config.max_length; ++i) {
        EXPECT_GE(output_data[i], 0);
        EXPECT_LT(output_data[i], config_.vocab_size);
    }
}

TEST_F(GPTTest, TrainingModeVsEvalMode) {
    GPT2LMHeadModel model(config_);

    Tensor input_ids({1, 4}, DType::Int64, device_);
    auto data = input_ids.data<int64_t>();
    data[0] = 10; data[1] = 20; data[2] = 30; data[3] = 40;
    Variable input_var(input_ids, false);

    // Training mode
    model.train(true);
    EXPECT_TRUE(model.is_training());
    auto train_output = model.forward(input_var, Variable{}, Tensor{});

    // Eval mode
    model.eval();
    EXPECT_FALSE(model.is_training());
    auto eval_output = model.forward(input_var, Variable{}, Tensor{});

    // Outputs should have same shape
    EXPECT_EQ(train_output.tensor().shape()[0], eval_output.tensor().shape()[0]);
    EXPECT_EQ(train_output.tensor().shape()[1], eval_output.tensor().shape()[1]);
    EXPECT_EQ(train_output.tensor().shape()[2], eval_output.tensor().shape()[2]);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    if (!::testing::GTEST_FLAG(list_tests)) {
        tenzor::initialize();
    }
    return RUN_ALL_TESTS();
}
