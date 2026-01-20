/**
 * @file test_gpt_multidtype.cpp
 * @brief Multi-backend and multi-dtype comprehensive unit tests for GPT models
 *
 * Tests GPT-2 and GPT-3 models with Float32, Float64, and Float16 support
 * across CPU, CUDA, and OneAPI backends.
 *
 * Coverage:
 * - GPT model construction (embeddings, decoder layers, LM head)
 * - Causal attention masking
 * - Forward pass with different sequence lengths
 * - Position embeddings
 * - Text generation strategies (greedy, top-k, top-p, beam search)
 * - GPT-2 and GPT-3 configurations
 * - Gradient flow and training
 */

#include <gtest/gtest.h>
#include "../multi_backend_dtype_fixture.hpp"
#include "../../include/tenzor/models/gpt.hpp"
#include "../../include/tenzor/ops/reduction.hpp"
#include <cmath>

using namespace tenzor;
using namespace tenzor::models;
using namespace tenzor::testing;

// ============================================================================
// GPT Multi-Backend Multi-DType Test Fixture
// ============================================================================

class GPTMultiDTypeTest : public MultiBackendDTypeTest {
protected:
    GPT2Config config_;
    int64_t batch_size_;
    int64_t seq_len_;

    void SetUp() override {
        MultiBackendDTypeTest::SetUp();

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
    }

    // Helper to check if values are NaN or Inf
    bool has_invalid_values(const Tensor& tensor) {
        auto cpu_tensor = tensor.to(Device::cpu()).to(DType::Float32);
        auto data = cpu_tensor.data<float>();
        for (int64_t i = 0; i < cpu_tensor.numel(); ++i) {
            if (std::isnan(data[i]) || std::isinf(data[i])) {
                return true;
            }
        }
        return false;
    }

    // Helper to create input IDs tensor
    Tensor create_input_ids(int64_t batch, int64_t seq_len) {
        Tensor input_ids({batch, seq_len}, DType::Int64, device());
        auto data = input_ids.data<int64_t>();
        for (int64_t i = 0; i < input_ids.numel(); ++i) {
            data[i] = i % config_.vocab_size;
        }
        return input_ids;
    }

    // Helper to create position IDs tensor
    Tensor create_position_ids(int64_t batch, int64_t seq_len) {
        Tensor position_ids({batch, seq_len}, DType::Int64, device());
        auto pos_data = position_ids.data<int64_t>();
        for (int64_t b = 0; b < batch; ++b) {
            for (int64_t s = 0; s < seq_len; ++s) {
                pos_data[b * seq_len + s] = s;
            }
        }
        return position_ids;
    }

    // Helper to convert tensor to target dtype
    Tensor to_dtype(Tensor tensor) {
        if (tensor.dtype() != dtype()) {
            return tensor.to(dtype());
        }
        return tensor;
    }
};

// ============================================================================
// GPTEmbeddings Tests
// ============================================================================

TEST_P(GPTMultiDTypeTest, EmbeddingsForwardShape) {
    GPTEmbeddings embeddings(config_);
    convert_model(embeddings);
    embeddings.train(false);

    Tensor input_ids = create_input_ids(batch_size_, seq_len_);
    Variable input_var(input_ids, false);
    auto output = embeddings.forward(input_var, Variable{});

    // Check output shape
    EXPECT_EQ(output.tensor().ndim(), 3);
    EXPECT_EQ(output.tensor().shape()[0], batch_size_);
    EXPECT_EQ(output.tensor().shape()[1], seq_len_);
    EXPECT_EQ(output.tensor().shape()[2], config_.n_embd);

    // Embeddings output should match dtype (converted internally if needed)
    EXPECT_FALSE(has_invalid_values(output.tensor()));
}

TEST_P(GPTMultiDTypeTest, EmbeddingsWithPositionIds) {
    GPTEmbeddings embeddings(config_);
    convert_model(embeddings);
    embeddings.train(false);

    Tensor input_ids = create_input_ids(batch_size_, seq_len_);
    Tensor position_ids = create_position_ids(batch_size_, seq_len_);

    Variable input_var(input_ids, false);
    Variable pos_var(position_ids, false);

    auto output = embeddings.forward(input_var, pos_var);

    EXPECT_EQ(output.tensor().shape()[0], batch_size_);
    EXPECT_EQ(output.tensor().shape()[1], seq_len_);
    EXPECT_EQ(output.tensor().shape()[2], config_.n_embd);
    EXPECT_FALSE(has_invalid_values(output.tensor()));
}

TEST_P(GPTMultiDTypeTest, EmbeddingsGradientFlow) {
    GPTEmbeddings embeddings(config_);
    convert_model(embeddings);
    embeddings.train();

    Tensor input_ids = create_input_ids(batch_size_, seq_len_);
    Variable input_var(input_ids, false);
    auto output = embeddings.forward(input_var, Variable{});

    // Check that output requires grad
    EXPECT_TRUE(output.requires_grad());

    // Create simple loss and backward
    auto loss = tenzor::sum(output);
    loss.backward();

    // Check that embeddings have gradients
    auto params = embeddings.parameters();
    EXPECT_GT(params.size(), 0);

    int grad_count = 0;
    for (const auto& param : params) {
        if (param->requires_grad()) {
            auto grad = param->grad();
            if (grad.has_value()) {
                EXPECT_FALSE(has_invalid_values(grad.value()));
                grad_count++;
            }
        }
    }
    EXPECT_GT(grad_count, 0);
}

TEST_P(GPTMultiDTypeTest, EmbeddingsDifferentSequenceLengths) {
    GPTEmbeddings embeddings(config_);
    convert_model(embeddings);
    embeddings.train(false);

    // Test with various sequence lengths
    std::vector<int64_t> seq_lengths = {4, 8, 16, 32, 64};

    for (auto seq_len : seq_lengths) {
        if (seq_len > config_.n_positions) continue;

        Tensor input_ids = create_input_ids(1, seq_len);
        Variable input_var(input_ids, false);
        auto output = embeddings.forward(input_var, Variable{});

        EXPECT_EQ(output.tensor().shape()[0], 1);
        EXPECT_EQ(output.tensor().shape()[1], seq_len);
        EXPECT_EQ(output.tensor().shape()[2], config_.n_embd);
        EXPECT_FALSE(has_invalid_values(output.tensor()));
    }
}

// ============================================================================
// GPTDecoderLayer Tests
// ============================================================================

TEST_P(GPTMultiDTypeTest, DecoderLayerForwardShape) {
    GPTDecoderLayer layer(config_);
    convert_model(layer);
    layer.train(false);

    Tensor hidden_states({batch_size_, seq_len_, config_.n_embd}, DType::Float32, device());
    hidden_states.fill_(0.1f);
    hidden_states = to_dtype(hidden_states);

    Variable input_var(hidden_states, false);
    auto output = layer.forward(input_var, Tensor{}, false);

    // Output should have same shape as input
    EXPECT_EQ(output.tensor().shape()[0], batch_size_);
    EXPECT_EQ(output.tensor().shape()[1], seq_len_);
    EXPECT_EQ(output.tensor().shape()[2], config_.n_embd);
    EXPECT_FALSE(has_invalid_values(output.tensor()));
}

TEST_P(GPTMultiDTypeTest, DecoderLayerWithCausalMask) {
    GPTDecoderLayer layer(config_);
    convert_model(layer);
    layer.train(false);

    Tensor hidden_states({batch_size_, seq_len_, config_.n_embd}, DType::Float32, device());
    hidden_states.fill_(0.1f);
    hidden_states = to_dtype(hidden_states);

    // Create causal mask
    Tensor causal_mask = nn::create_causal_mask(seq_len_, device());

    Variable input_var(hidden_states, false);
    auto output = layer.forward(input_var, causal_mask);

    EXPECT_EQ(output.tensor().shape()[0], batch_size_);
    EXPECT_EQ(output.tensor().shape()[1], seq_len_);
    EXPECT_EQ(output.tensor().shape()[2], config_.n_embd);
    EXPECT_FALSE(has_invalid_values(output.tensor()));
}

TEST_P(GPTMultiDTypeTest, DecoderLayerGradientFlow) {
    GPTDecoderLayer layer(config_);
    convert_model(layer);
    layer.train();

    Tensor hidden_states({batch_size_, seq_len_, config_.n_embd}, DType::Float32, device());
    hidden_states.fill_(0.1f);
    hidden_states = to_dtype(hidden_states);

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
                EXPECT_FALSE(has_invalid_values(grad.value()));
                grad_count++;
            }
        }
    }
    EXPECT_GT(grad_count, 0);
}

TEST_P(GPTMultiDTypeTest, DecoderLayerDifferentSequenceLengths) {
    GPTDecoderLayer layer(config_);
    convert_model(layer);
    layer.train(false);

    std::vector<int64_t> seq_lengths = {4, 8, 16, 32};

    for (auto seq_len : seq_lengths) {
        Tensor hidden_states({1, seq_len, config_.n_embd}, DType::Float32, device());
        hidden_states.fill_(0.1f);
        hidden_states = to_dtype(hidden_states);

        Variable input_var(hidden_states, false);
        auto output = layer.forward(input_var, Tensor{}, false);

        EXPECT_EQ(output.tensor().shape()[0], 1);
        EXPECT_EQ(output.tensor().shape()[1], seq_len);
        EXPECT_EQ(output.tensor().shape()[2], config_.n_embd);
        EXPECT_FALSE(has_invalid_values(output.tensor()));
    }
}

// ============================================================================
// GPT2Model Tests
// ============================================================================

TEST_P(GPTMultiDTypeTest, GPT2ModelForwardShape) {
    GPT2Model model(config_);
    convert_model(model);
    model.train(false);

    Tensor input_ids = create_input_ids(batch_size_, seq_len_);
    Variable input_var(input_ids, false);
    auto output = model.forward(input_var, Variable{}, Tensor{});

    // Output should be [batch, seq_len, n_embd]
    EXPECT_EQ(output.tensor().shape()[0], batch_size_);
    EXPECT_EQ(output.tensor().shape()[1], seq_len_);
    EXPECT_EQ(output.tensor().shape()[2], config_.n_embd);
    EXPECT_FALSE(has_invalid_values(output.tensor()));
}

TEST_P(GPTMultiDTypeTest, GPT2ModelParameterCount) {
    GPT2Model model(config_);
    convert_model(model);

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
    int64_t min_expected = config_.vocab_size * config_.n_embd +
                          config_.n_positions * config_.n_embd;
    EXPECT_GT(total_params, min_expected);
}

TEST_P(GPTMultiDTypeTest, GPT2ModelGradientFlow) {
    GPT2Model model(config_);
    convert_model(model);
    model.train();

    Tensor input_ids = create_input_ids(batch_size_, seq_len_);
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
                EXPECT_FALSE(has_invalid_values(grad.value()));
                grad_count++;
            }
        }
    }
    EXPECT_GT(grad_count, 0);
}

TEST_P(GPTMultiDTypeTest, GPT2ModelWithPositionIds) {
    GPT2Model model(config_);
    convert_model(model);
    model.train(false);

    Tensor input_ids = create_input_ids(batch_size_, seq_len_);
    Tensor position_ids = create_position_ids(batch_size_, seq_len_);

    Variable input_var(input_ids, false);
    Variable pos_var(position_ids, false);

    auto output = model.forward(input_var, pos_var, Tensor{});

    EXPECT_EQ(output.tensor().shape()[0], batch_size_);
    EXPECT_EQ(output.tensor().shape()[1], seq_len_);
    EXPECT_EQ(output.tensor().shape()[2], config_.n_embd);
    EXPECT_FALSE(has_invalid_values(output.tensor()));
}

TEST_P(GPTMultiDTypeTest, GPT2ModelWithCausalMask) {
    GPT2Model model(config_);
    convert_model(model);
    model.train(false);

    Tensor input_ids = create_input_ids(batch_size_, seq_len_);
    Tensor causal_mask = nn::create_causal_mask(seq_len_, device());

    Variable input_var(input_ids, false);
    auto output = model.forward(input_var, Variable{}, causal_mask);

    EXPECT_EQ(output.tensor().shape()[0], batch_size_);
    EXPECT_EQ(output.tensor().shape()[1], seq_len_);
    EXPECT_EQ(output.tensor().shape()[2], config_.n_embd);
    EXPECT_FALSE(has_invalid_values(output.tensor()));
}

// ============================================================================
// GPT2LMHeadModel Tests
// ============================================================================

TEST_P(GPTMultiDTypeTest, GPT2LMHeadForwardShape) {
    GPT2LMHeadModel model(config_);
    convert_model(model);
    model.train(false);

    Tensor input_ids = create_input_ids(batch_size_, seq_len_);
    Variable input_var(input_ids, false);
    auto logits = model.forward(input_var, Variable{}, Tensor{});

    // Output should be [batch, seq_len, vocab_size]
    EXPECT_EQ(logits.tensor().shape()[0], batch_size_);
    EXPECT_EQ(logits.tensor().shape()[1], seq_len_);
    EXPECT_EQ(logits.tensor().shape()[2], config_.vocab_size);
    EXPECT_FALSE(has_invalid_values(logits.tensor()));
}

TEST_P(GPTMultiDTypeTest, GPT2LMHeadLogitsRange) {
    GPT2LMHeadModel model(config_);
    convert_model(model);
    model.train(false);

    Tensor input_ids({1, 4}, DType::Int64, device());
    auto data = input_ids.data<int64_t>();
    data[0] = 10; data[1] = 20; data[2] = 30; data[3] = 40;

    Variable input_var(input_ids, false);
    auto logits = model.forward(input_var, Variable{}, Tensor{});

    // Check that logits are reasonable (not NaN or inf)
    EXPECT_FALSE(has_invalid_values(logits.tensor()));
}

TEST_P(GPTMultiDTypeTest, GPT2LMHeadGradientFlow) {
    GPT2LMHeadModel model(config_);
    convert_model(model);
    model.train();

    Tensor input_ids = create_input_ids(batch_size_, seq_len_);
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
                EXPECT_FALSE(has_invalid_values(grad.value()));
                grad_count++;
            }
        }
    }
    EXPECT_GT(grad_count, 0);
}

TEST_P(GPTMultiDTypeTest, GPT2LMHeadDifferentSequenceLengths) {
    GPT2LMHeadModel model(config_);
    convert_model(model);
    model.train(false);

    std::vector<int64_t> seq_lengths = {4, 8, 16, 32, 64};

    for (auto seq_len : seq_lengths) {
        if (seq_len > config_.n_positions) continue;

        Tensor input_ids = create_input_ids(1, seq_len);
        Variable input_var(input_ids, false);
        auto logits = model.forward(input_var, Variable{}, Tensor{});

        EXPECT_EQ(logits.tensor().shape()[0], 1);
        EXPECT_EQ(logits.tensor().shape()[1], seq_len);
        EXPECT_EQ(logits.tensor().shape()[2], config_.vocab_size);
        EXPECT_FALSE(has_invalid_values(logits.tensor()));
    }
}

// ============================================================================
// GPT3Model Tests
// ============================================================================

TEST_P(GPTMultiDTypeTest, GPT3ModelConstruction) {
    auto gpt3_config = GPT3Config::gpt3_small();
    gpt3_config.vocab_size = 1000;  // Reduce for test
    gpt3_config.n_positions = 64;
    gpt3_config.n_layer = 2;

    GPT3Model model(gpt3_config);
    convert_model(model);

    auto params = model.parameters();
    EXPECT_GT(params.size(), 0);
}

TEST_P(GPTMultiDTypeTest, GPT3LMHeadModelForward) {
    auto gpt3_config = GPT3Config::gpt3_small();
    gpt3_config.vocab_size = 1000;
    gpt3_config.n_positions = 64;
    gpt3_config.n_layer = 2;

    GPT3LMHeadModel model(gpt3_config);
    convert_model(model);
    model.train(false);

    Tensor input_ids({1, 8}, DType::Int64, device());
    auto data = input_ids.data<int64_t>();
    for (int64_t i = 0; i < 8; ++i) {
        data[i] = i * 10;
    }

    Variable input_var(input_ids, false);
    auto logits = model.forward(input_var, Variable{}, Tensor{});

    EXPECT_EQ(logits.tensor().shape()[0], 1);
    EXPECT_EQ(logits.tensor().shape()[1], 8);
    EXPECT_EQ(logits.tensor().shape()[2], gpt3_config.vocab_size);
    EXPECT_FALSE(has_invalid_values(logits.tensor()));
}

TEST_P(GPTMultiDTypeTest, GPT3LMHeadModelGradientFlow) {
    auto gpt3_config = GPT3Config::gpt3_small();
    gpt3_config.vocab_size = 1000;
    gpt3_config.n_positions = 64;
    gpt3_config.n_layer = 1;  // Single layer for faster test

    GPT3LMHeadModel model(gpt3_config);
    convert_model(model);
    model.train();

    Tensor input_ids({1, 8}, DType::Int64, device());
    auto data = input_ids.data<int64_t>();
    for (int64_t i = 0; i < 8; ++i) {
        data[i] = i * 10;
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
                EXPECT_FALSE(has_invalid_values(grad.value()));
                grad_count++;
            }
        }
    }
    EXPECT_GT(grad_count, 0);
}

// ============================================================================
// TextGenerator Tests
// ============================================================================

TEST_P(GPTMultiDTypeTest, GeneratorGreedySearch) {
    GPT2LMHeadModel model(config_);
    convert_model(model);
    model.train(false);

    GenerationConfig gen_config;
    gen_config.max_length = 20;
    gen_config.strategy = GenerationStrategy::Greedy;

    TextGenerator generator(model, gen_config);

    Tensor input_ids({1, 4}, DType::Int64, device());
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

    // Verify all generated tokens are valid
    for (int64_t i = 0; i < gen_config.max_length; ++i) {
        EXPECT_GE(output_data[i], 0);
        EXPECT_LT(output_data[i], config_.vocab_size);
    }
}

TEST_P(GPTMultiDTypeTest, GeneratorTopKSampling) {
    GPT2LMHeadModel model(config_);
    convert_model(model);
    model.train(false);

    GenerationConfig gen_config;
    gen_config.max_length = 15;
    gen_config.strategy = GenerationStrategy::TopK;
    gen_config.top_k = 10;
    gen_config.temperature = 1.0;
    gen_config.seed = 42;

    TextGenerator generator(model, gen_config);

    Tensor input_ids({1, 3}, DType::Int64, device());
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

    // Verify all tokens are valid
    for (int64_t i = 0; i < gen_config.max_length; ++i) {
        EXPECT_GE(output_data[i], 0);
        EXPECT_LT(output_data[i], config_.vocab_size);
    }
}

TEST_P(GPTMultiDTypeTest, GeneratorTopPSampling) {
    GPT2LMHeadModel model(config_);
    convert_model(model);
    model.train(false);

    GenerationConfig gen_config;
    gen_config.max_length = 15;
    gen_config.strategy = GenerationStrategy::TopP;
    gen_config.top_p = 0.9;
    gen_config.temperature = 0.8;
    gen_config.seed = 123;

    TextGenerator generator(model, gen_config);

    Tensor input_ids({1, 3}, DType::Int64, device());
    auto data = input_ids.data<int64_t>();
    data[0] = 7; data[1] = 17; data[2] = 27;

    auto output = generator.top_p_sampling(input_ids, gen_config.top_p, gen_config.temperature);

    EXPECT_EQ(output.shape()[0], 1);
    EXPECT_EQ(output.shape()[1], gen_config.max_length);

    // Verify all tokens are valid
    auto output_data = output.data<int64_t>();
    for (int64_t i = 0; i < gen_config.max_length; ++i) {
        EXPECT_GE(output_data[i], 0);
        EXPECT_LT(output_data[i], config_.vocab_size);
    }
}

TEST_P(GPTMultiDTypeTest, GeneratorBeamSearch) {
    GPT2LMHeadModel model(config_);
    convert_model(model);
    model.train(false);

    GenerationConfig gen_config;
    gen_config.max_length = 12;
    gen_config.strategy = GenerationStrategy::BeamSearch;
    gen_config.num_beams = 3;

    TextGenerator generator(model, gen_config);

    Tensor input_ids({1, 2}, DType::Int64, device());
    auto data = input_ids.data<int64_t>();
    data[0] = 100; data[1] = 200;

    auto output = generator.beam_search(input_ids, gen_config.num_beams);

    EXPECT_EQ(output.shape()[0], 1);
    EXPECT_EQ(output.shape()[1], gen_config.max_length);

    // Check input tokens preserved
    auto output_data = output.data<int64_t>();
    EXPECT_EQ(output_data[0], 100);
    EXPECT_EQ(output_data[1], 200);

    // Verify all tokens are valid
    for (int64_t i = 0; i < gen_config.max_length; ++i) {
        EXPECT_GE(output_data[i], 0);
        EXPECT_LT(output_data[i], config_.vocab_size);
    }
}

TEST_P(GPTMultiDTypeTest, GeneratorGenericGenerate) {
    GPT2LMHeadModel model(config_);
    convert_model(model);
    model.train(false);

    GenerationConfig gen_config;
    gen_config.max_length = 10;
    gen_config.strategy = GenerationStrategy::Greedy;

    TextGenerator generator(model, gen_config);

    Tensor input_ids({1, 2}, DType::Int64, device());
    auto data = input_ids.data<int64_t>();
    data[0] = 50; data[1] = 60;

    auto output = generator.generate(input_ids);

    EXPECT_EQ(output.shape()[0], 1);
    EXPECT_EQ(output.shape()[1], gen_config.max_length);

    // Verify all tokens are valid
    auto output_data = output.data<int64_t>();
    for (int64_t i = 0; i < gen_config.max_length; ++i) {
        EXPECT_GE(output_data[i], 0);
        EXPECT_LT(output_data[i], config_.vocab_size);
    }
}

// ============================================================================
// Configuration Tests
// ============================================================================

TEST_P(GPTMultiDTypeTest, GPT2ConfigPresets) {
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

TEST_P(GPTMultiDTypeTest, GPT3ConfigPresets) {
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

TEST_P(GPTMultiDTypeTest, EndToEndTextGeneration) {
    // Small model for fast testing
    config_.n_layer = 1;

    GPT2LMHeadModel model(config_);
    convert_model(model);
    model.train(false);

    GenerationConfig gen_config;
    gen_config.max_length = 10;
    gen_config.strategy = GenerationStrategy::Greedy;

    TextGenerator generator(model, gen_config);

    // Create input
    Tensor input_ids({1, 3}, DType::Int64, device());
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

TEST_P(GPTMultiDTypeTest, TrainingModeVsEvalMode) {
    GPT2LMHeadModel model(config_);
    convert_model(model);

    Tensor input_ids({1, 4}, DType::Int64, device());
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

TEST_P(GPTMultiDTypeTest, LongSequenceHandling) {
    // Test with longer sequences
    config_.n_positions = 256;
    config_.n_layer = 1;  // Single layer for speed

    GPT2LMHeadModel model(config_);
    convert_model(model);
    model.train(false);

    int64_t long_seq_len = 128;
    Tensor input_ids = create_input_ids(1, long_seq_len);
    Variable input_var(input_ids, false);

    auto logits = model.forward(input_var, Variable{}, Tensor{});

    EXPECT_EQ(logits.tensor().shape()[0], 1);
    EXPECT_EQ(logits.tensor().shape()[1], long_seq_len);
    EXPECT_EQ(logits.tensor().shape()[2], config_.vocab_size);
    EXPECT_FALSE(has_invalid_values(logits.tensor()));
}

TEST_P(GPTMultiDTypeTest, BatchedGeneration) {
    config_.n_layer = 1;
    GPT2LMHeadModel model(config_);
    convert_model(model);
    model.train(false);

    GenerationConfig gen_config;
    gen_config.max_length = 10;
    gen_config.strategy = GenerationStrategy::Greedy;

    TextGenerator generator(model, gen_config);

    // Test with batch size 3
    Tensor input_ids({3, 4}, DType::Int64, device());
    auto data = input_ids.data<int64_t>();
    for (int64_t i = 0; i < input_ids.numel(); ++i) {
        data[i] = (i * 10) % config_.vocab_size;
    }

    auto output = generator.greedy_search(input_ids);

    EXPECT_EQ(output.shape()[0], 3);
    EXPECT_EQ(output.shape()[1], gen_config.max_length);

    // Verify all tokens are valid
    auto output_data = output.data<int64_t>();
    for (int64_t i = 0; i < output.numel(); ++i) {
        EXPECT_GE(output_data[i], 0);
        EXPECT_LT(output_data[i], config_.vocab_size);
    }
}

// ============================================================================
// Test Instantiation
// ============================================================================

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(GPTMultiDTypeTest);

int main(int argc, char** argv) {
    tenzor::initialize();
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
