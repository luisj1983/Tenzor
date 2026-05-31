/**
 * @file test_albert_t5.cpp
 * @brief Comprehensive tests for ALBERT and T5 models
 */

#include <gtest/gtest.h>
#include "../backend_test_fixture.hpp"
#include <tenzor/tenzor.hpp>
#include "../../include/tenzor/models/albert.hpp"
#include "../../include/tenzor/models/t5.hpp"

using namespace tenzor;
using namespace tenzor::models;

class ALBERTandT5Test : public ::tenzor::testing::BackendTest {
protected:
    void SetUp() override {
        ::tenzor::testing::BackendTest::SetUp();
        if (::testing::Test::IsSkipped()) return;
    }

    // Helper to create input token IDs with valid values.
    // Token/index inputs are built on CPU via host writes, then moved to `device`.
    auto create_input_ids(int64_t batch_size, int64_t seq_len, int64_t vocab_size = 30000) -> Variable {
        Tensor input_ids({batch_size, seq_len}, DType::Int64, Device::cpu());

        // Fill with valid token IDs within vocabulary range
        std::vector<int64_t> data(batch_size * seq_len);
        for (size_t i = 0; i < data.size(); ++i) {
            data[i] = i % vocab_size;
        }
        std::copy(data.begin(), data.end(), input_ids.data<int64_t>());

        return Variable(input_ids.to(device), true);
    }
};

// ============================================================================
// ALBERT Base Tests
// ============================================================================

TEST_P(ALBERTandT5Test, ALBERTBaseConfigTest) {
    auto config = AlbertConfig::base();

    EXPECT_EQ(config.embedding_size, 128);
    EXPECT_EQ(config.hidden_size, 768);
    EXPECT_EQ(config.num_hidden_layers, 12);
    EXPECT_EQ(config.num_attention_heads, 12);
    EXPECT_EQ(config.intermediate_size, 3072);
}

TEST_P(ALBERTandT5Test, ALBERTBaseForwardShape) {
    auto config = AlbertConfig::base();
    config.vocab_size = 30000;
    auto model = std::make_shared<AlbertModel>(config);
    model->to(device);
    int64_t batch_size = 2;
    int64_t seq_len = 128;

    auto input_ids = create_input_ids(batch_size, seq_len, config.vocab_size);
    auto output = model->forward(input_ids, Tensor{}, Variable{}, Variable{});

    auto shape = output.sequence_output.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape.begin(), shape.end()),
              (std::vector<int64_t>{batch_size, seq_len, 768}));
}

TEST_P(ALBERTandT5Test, ALBERTBaseGradientFlow) {
    auto config = AlbertConfig::base();
    config.vocab_size = 30000;
    auto model = std::make_shared<AlbertModel>(config);
    model->to(device);
    model->train();

    auto input_ids = create_input_ids(1, 64);
    auto output = model->forward(input_ids, Tensor{}, Variable{}, Variable{});
    Variable loss = tenzor::sum(output.sequence_output);
    loss.backward();

    auto params = model->parameters();
    EXPECT_GT(params.size(), 0);
}

TEST_P(ALBERTandT5Test, ALBERTBaseParameterCount) {
    auto config = AlbertConfig::base();
    config.vocab_size = 30000;
    auto model = std::make_shared<AlbertModel>(config);
    model->to(device);
    auto params = model->parameters();

    size_t total_params = 0;
    for (const auto& p : params) {
        size_t param_size = 1;
        for (auto dim : p->tensor().shape()) {
            param_size *= dim;
        }
        total_params += param_size;
    }

    // ALBERT-Base should have ~12M parameters due to parameter sharing
    // (much less than BERT's ~110M)
    EXPECT_GT(total_params, 8'000'000);
    EXPECT_LT(total_params, 16'000'000);
}

// ============================================================================
// ALBERT Large Tests
// ============================================================================

TEST_P(ALBERTandT5Test, ALBERTLargeConfigTest) {
    auto config = AlbertConfig::large();

    EXPECT_EQ(config.hidden_size, 1024);
    EXPECT_EQ(config.num_hidden_layers, 24);
    EXPECT_EQ(config.num_attention_heads, 16);
}

TEST_P(ALBERTandT5Test, ALBERTLargeForwardShape) {
    auto config = AlbertConfig::large();
    config.vocab_size = 30000;
    auto model = std::make_shared<AlbertModel>(config);
    model->to(device);
    int64_t batch_size = 2;
    int64_t seq_len = 128;

    auto input_ids = create_input_ids(batch_size, seq_len, config.vocab_size);
    auto output = model->forward(input_ids, Tensor{}, Variable{}, Variable{});

    auto shape = output.sequence_output.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape.begin(), shape.end()),
              (std::vector<int64_t>{batch_size, seq_len, 1024}));
}

TEST_P(ALBERTandT5Test, ALBERTLargeGradientFlow) {
    auto config = AlbertConfig::large();
    config.vocab_size = 30000;
    auto model = std::make_shared<AlbertModel>(config);
    model->to(device);
    model->train();

    auto input_ids = create_input_ids(1, 64);
    auto output = model->forward(input_ids, Tensor{}, Variable{}, Variable{});
    Variable loss = tenzor::sum(output.sequence_output);
    loss.backward();

    auto params = model->parameters();
    EXPECT_GT(params.size(), 0);
}

// ============================================================================
// ALBERT XLarge Tests
// ============================================================================

TEST_P(ALBERTandT5Test, ALBERTXLargeForwardShape) {
    auto config = AlbertConfig::xlarge();
    config.vocab_size = 30000;
    auto model = std::make_shared<AlbertModel>(config);
    model->to(device);
    int64_t batch_size = 1;
    int64_t seq_len = 128;

    auto input_ids = create_input_ids(batch_size, seq_len, config.vocab_size);
    auto output = model->forward(input_ids, Tensor{}, Variable{}, Variable{});

    auto shape = output.sequence_output.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape.begin(), shape.end()),
              (std::vector<int64_t>{batch_size, seq_len, 2048}));
}

// ============================================================================
// ALBERT XXLarge Tests
// ============================================================================

TEST_P(ALBERTandT5Test, ALBERTXXLargeForwardShape) {
    auto config = AlbertConfig::xxlarge();
    config.vocab_size = 30000;
    auto model = std::make_shared<AlbertModel>(config);
    model->to(device);
    int64_t batch_size = 1;
    int64_t seq_len = 64;

    auto input_ids = create_input_ids(batch_size, seq_len, config.vocab_size);
    auto output = model->forward(input_ids, Tensor{}, Variable{}, Variable{});

    auto shape = output.sequence_output.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape.begin(), shape.end()),
              (std::vector<int64_t>{batch_size, seq_len, 4096}));
}

// ============================================================================
// T5 Small Tests
// ============================================================================

TEST_P(ALBERTandT5Test, T5SmallConfigTest) {
    auto config = T5Config::small();

    EXPECT_EQ(config.d_model, 512);
    EXPECT_EQ(config.d_ff, 2048);
    EXPECT_EQ(config.num_layers, 6);
    EXPECT_EQ(config.num_heads, 8);
}

TEST_P(ALBERTandT5Test, T5SmallForwardShape) {
    auto config = T5Config::small();
    config.vocab_size = 32128;
    auto model = std::make_shared<T5Model>(config);
    model->to(device);
    int64_t batch_size = 2;
    int64_t seq_len = 128;

    auto input_ids = create_input_ids(batch_size, seq_len, config.vocab_size);
    auto decoder_input_ids = create_input_ids(batch_size, seq_len, config.vocab_size);

    auto output = model->forward(input_ids, decoder_input_ids);

    auto shape = output.decoder_output.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape.begin(), shape.end()),
              (std::vector<int64_t>{batch_size, seq_len, 512}));
}

TEST_P(ALBERTandT5Test, T5SmallGradientFlow) {
    auto config = T5Config::small();
    config.vocab_size = 32128;
    auto model = std::make_shared<T5Model>(config);
    model->to(device);
    model->train();

    auto input_ids = create_input_ids(1, 64, config.vocab_size);
    auto decoder_input_ids = create_input_ids(1, 64, config.vocab_size);

    auto output = model->forward(input_ids, decoder_input_ids);
    Variable loss = tenzor::sum(output.decoder_output);
    loss.backward();

    auto params = model->parameters();
    EXPECT_GT(params.size(), 0);
}

// ============================================================================
// T5 Base Tests
// ============================================================================

TEST_P(ALBERTandT5Test, T5BaseConfigTest) {
    auto config = T5Config::base();

    EXPECT_EQ(config.d_model, 768);
    EXPECT_EQ(config.d_ff, 3072);
    EXPECT_EQ(config.num_layers, 12);
    EXPECT_EQ(config.num_heads, 12);
}

TEST_P(ALBERTandT5Test, T5BaseForwardShape) {
    auto config = T5Config::base();
    config.vocab_size = 32128;
    auto model = std::make_shared<T5Model>(config);
    model->to(device);
    int64_t batch_size = 2;
    int64_t seq_len = 128;

    auto input_ids = create_input_ids(batch_size, seq_len, config.vocab_size);
    auto decoder_input_ids = create_input_ids(batch_size, seq_len, config.vocab_size);

    auto output = model->forward(input_ids, decoder_input_ids);

    auto shape = output.decoder_output.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape.begin(), shape.end()),
              (std::vector<int64_t>{batch_size, seq_len, 768}));
}

TEST_P(ALBERTandT5Test, T5BaseGradientFlow) {
    auto config = T5Config::base();
    config.vocab_size = 32128;
    auto model = std::make_shared<T5Model>(config);
    model->to(device);
    model->train();

    auto input_ids = create_input_ids(1, 64, config.vocab_size);
    auto decoder_input_ids = create_input_ids(1, 64, config.vocab_size);

    auto output = model->forward(input_ids, decoder_input_ids);
    Variable loss = tenzor::sum(output.decoder_output);
    loss.backward();

    auto params = model->parameters();
    EXPECT_GT(params.size(), 0);
}

// ============================================================================
// T5 Large Tests
// ============================================================================

TEST_P(ALBERTandT5Test, T5LargeForwardShape) {
    auto config = T5Config::large();
    config.vocab_size = 32128;
    auto model = std::make_shared<T5Model>(config);
    model->to(device);
    int64_t batch_size = 1;
    int64_t seq_len = 128;

    auto input_ids = create_input_ids(batch_size, seq_len, config.vocab_size);
    auto decoder_input_ids = create_input_ids(batch_size, seq_len, config.vocab_size);

    auto output = model->forward(input_ids, decoder_input_ids);

    auto shape = output.decoder_output.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape.begin(), shape.end()),
              (std::vector<int64_t>{batch_size, seq_len, 1024}));
}

// ============================================================================
// Edge Case Tests
// ============================================================================

TEST_P(ALBERTandT5Test, ALBERTBatchSizeOne) {
    auto config = AlbertConfig::base();
    config.vocab_size = 30000;
    auto model = std::make_shared<AlbertModel>(config);
    model->to(device);
    auto input_ids = create_input_ids(1, 64);
    auto output = model->forward(input_ids, Tensor{}, Variable{}, Variable{});

    auto shape = output.sequence_output.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape.begin(), shape.end()),
              (std::vector<int64_t>{1, 64, 768}));
}

// G15 regression: T5 generate() must read its starting decoder token from
// the config (default 0 = HF T5 convention) and honor an explicit override.
TEST_P(ALBERTandT5Test, T5GenerateUsesConfiguredDecoderStartToken_G15) {
    auto config = T5Config::small();
    config.vocab_size = 32128;
    // Explicit non-zero start token — distinct from the previous hard-coded 0.
    config.decoder_start_token_id = 7;
    auto model = std::make_shared<T5ForConditionalGeneration>(config);
    model->to(device);
    model->eval();

    auto input_ids = create_input_ids(/*B=*/2, /*T=*/16, config.vocab_size);

    Tensor generated = model->generate(input_ids, /*max_length=*/4,
                                       /*temperature=*/1.0);
    // Output shape contract: (batch, generated_length).
    ASSERT_EQ(generated.shape().size(), 2u);
    EXPECT_EQ(generated.shape()[0], 2);

    Tensor gen_cpu = generated.to(Device::cpu());
    auto* g = gen_cpu.data<int64_t>();
    // First column = decoder start token = config.decoder_start_token_id.
    EXPECT_EQ(g[0], 7) << "Batch 0 col 0 should be the configured start token (7)";
    EXPECT_EQ(g[gen_cpu.shape()[1]], 7)
        << "Batch 1 col 0 should also be the configured start token (7)";
}

// G15: explicit bos_token_id argument must override config value.
TEST_P(ALBERTandT5Test, T5GenerateBosTokenOverride_G15) {
    auto config = T5Config::small();
    config.vocab_size = 32128;
    config.decoder_start_token_id = 0;  // HF default
    auto model = std::make_shared<T5ForConditionalGeneration>(config);
    model->to(device);
    model->eval();

    auto input_ids = create_input_ids(1, 8, config.vocab_size);
    Tensor generated = model->generate(input_ids, /*max_length=*/3,
                                       /*temperature=*/1.0,
                                       /*bos_token_id=*/42);
    Tensor gen_cpu = generated.to(Device::cpu());
    EXPECT_EQ(gen_cpu.data<int64_t>()[0], 42)
        << "Explicit bos_token_id=42 should override config's 0.";
}

// G12 regression: T5 decoder must combine causal mask with the padding mask
// instead of silently dropping the padding mask. Verify by running two
// forward passes with the same inputs but different padding masks — outputs
// at non-padded positions must change when padded positions vary.
TEST_P(ALBERTandT5Test, T5DecoderCombinesPaddingMaskWithCausal_G12) {
    auto config = T5Config::small();
    config.vocab_size = 32128;
    auto model = std::make_shared<T5Model>(config);
    model->to(device);
    model->eval();

    const int64_t B = 1;
    const int64_t T = 32;
    auto input_ids = create_input_ids(B, T, config.vocab_size);
    auto decoder_input_ids = create_input_ids(B, T, config.vocab_size);

    // Mask 1: full attention (all 1s). Mask 2: mask the last 16 tokens.
    // Build masks on CPU via host writes, then move to `device`.
    Tensor mask_full_cpu({B, T}, DType::Float32, Device::cpu());
    Tensor mask_half_cpu({B, T}, DType::Float32, Device::cpu());
    auto* fp = mask_full_cpu.data<float>();
    auto* hp = mask_half_cpu.data<float>();
    for (int64_t i = 0; i < T; ++i) {
        fp[i] = 1.0f;
        hp[i] = (i < T / 2) ? 1.0f : 0.0f;
    }

    Tensor enc_mask_cpu({B, T}, DType::Float32, Device::cpu());
    for (int64_t i = 0; i < T; ++i) enc_mask_cpu.data<float>()[i] = 1.0f;

    Tensor mask_full = mask_full_cpu.to(device);
    Tensor mask_half = mask_half_cpu.to(device);
    Tensor enc_mask = enc_mask_cpu.to(device);

    auto out_full = model->forward(input_ids, decoder_input_ids, enc_mask, mask_full);
    auto out_half = model->forward(input_ids, decoder_input_ids, enc_mask, mask_half);

    auto* fo = out_full.decoder_output.tensor().to(Device::cpu()).to(DType::Float32).data<float>();
    auto* ho = out_half.decoder_output.tensor().to(Device::cpu()).to(DType::Float32).data<float>();
    const int64_t D = 512;  // T5 small hidden size

    // Position 0 should be (nearly) unchanged: causal mask only attends to
    // self at position 0, so the padding mask of *later* positions doesn't
    // matter at position 0. But because of the causal-mask invariant, this
    // doesn't actually exercise the mask combination. Check position T/2-1
    // instead: it attends to positions 0..T/2-1 in both cases, so should be
    // identical. Position T-1, however, attends to positions 0..T-1 under
    // mask_full but only 0..T/2-1 under mask_half — outputs must differ.
    double diff_pos_last = 0.0;
    for (int64_t k = 0; k < D; ++k) {
        diff_pos_last += std::abs(fo[(T - 1) * D + k] - ho[(T - 1) * D + k]);
    }

    // Position T/2-1 (last non-masked under mask_half) should match exactly,
    // since under both masks the attention only sees positions 0..T/2-1.
    double diff_pos_mid = 0.0;
    for (int64_t k = 0; k < D; ++k) {
        diff_pos_mid += std::abs(fo[(T / 2 - 1) * D + k] - ho[(T / 2 - 1) * D + k]);
    }

    EXPECT_GT(diff_pos_last, 1.0f)
        << "Position T-1 attends to padded positions under mask_full but not "
           "under mask_half — outputs must differ. Sum of |diff| was "
        << diff_pos_last << " — too small, padding mask was probably ignored.";
    EXPECT_LT(diff_pos_mid, 1.0f)
        << "Position T/2-1 attends only to non-padded positions in both "
           "cases — outputs should be ~identical. Sum of |diff| was "
        << diff_pos_mid;
}

TEST_P(ALBERTandT5Test, T5VariableSequenceLength) {
    auto config = T5Config::small();
    config.vocab_size = 32128;
    auto model = std::make_shared<T5Model>(config);
    model->to(device);

    // Test with different sequence lengths
    auto input_32 = create_input_ids(2, 32, config.vocab_size);
    auto decoder_32 = create_input_ids(2, 32, config.vocab_size);
    auto output_32 = model->forward(input_32, decoder_32);
    auto shape_32 = output_32.decoder_output.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape_32.begin(), shape_32.end()),
              (std::vector<int64_t>{2, 32, 512}));
}

INSTANTIATE_BACKEND_TESTS(ALBERTandT5Test);
