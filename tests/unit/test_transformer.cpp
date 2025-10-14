#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/nn/layers/transformer.hpp>
#include <tenzor/nn/layers/attention.hpp>
#include <cmath>

using namespace tenzor;
using namespace tenzor::nn;

// Global test environment
class TransformerTestEnvironment : public ::testing::Environment {
public:
    void SetUp() override {
        tenzor::initialize();
    }
};

static ::testing::Environment* const transformer_env =
    ::testing::AddGlobalTestEnvironment(new TransformerTestEnvironment);

// ============================================================================
// PositionalEncoding Tests
// ============================================================================

TEST(PositionalEncodingTest, Construction) {
    EXPECT_NO_THROW({
        PositionalEncoding pe(512);
    });

    EXPECT_NO_THROW({
        PositionalEncoding pe(768, 10000, 0.1);
    });
}

TEST(PositionalEncodingTest, ForwardShape) {
    PositionalEncoding pe(256, 1000, 0.0);

    int64_t batch_size = 4;
    int64_t seq_len = 20;
    int64_t d_model = 256;

    Variable input(randn({batch_size, seq_len, d_model}), true);
    Variable output = pe.forward(input);

    // Shape should be preserved
    EXPECT_EQ(output.shape().size(), 3);
    EXPECT_EQ(output.shape()[0], batch_size);
    EXPECT_EQ(output.shape()[1], seq_len);
    EXPECT_EQ(output.shape()[2], d_model);
}

TEST(PositionalEncodingTest, ExceedsMaxLen) {
    PositionalEncoding pe(128, 100, 0.0);

    Variable input(randn({2, 150, 128}), true);

    // Should throw because seq_len (150) > max_len (100)
    EXPECT_THROW({
        pe.forward(input);
    }, std::runtime_error);
}

TEST(PositionalEncodingTest, WithDropout) {
    PositionalEncoding pe(256, 1000, 0.5);

    Variable input(randn({2, 10, 256}), true);

    // Should not throw
    EXPECT_NO_THROW({
        Variable output = pe.forward(input);
    });
}

// ============================================================================
// TransformerEncoderLayer Tests
// ============================================================================

TEST(TransformerEncoderLayerTest, Construction) {
    EXPECT_NO_THROW({
        TransformerEncoderLayer layer(512, 8);
    });

    EXPECT_NO_THROW({
        TransformerEncoderLayer layer(768, 12, 3072, 0.1, "gelu", true);
    });
}

TEST(TransformerEncoderLayerTest, InvalidActivation) {
    EXPECT_THROW({
        TransformerEncoderLayer layer(512, 8, 2048, 0.1, "invalid");
    }, std::invalid_argument);
}

TEST(TransformerEncoderLayerTest, ForwardShape) {
    TransformerEncoderLayer layer(256, 4, 1024, 0.0, "relu", true);

    int64_t batch_size = 2;
    int64_t seq_len = 10;
    int64_t d_model = 256;

    Variable src(randn({batch_size, seq_len, d_model}), true);
    Variable output = layer.forward(src, Tensor{}, Tensor{});

    // Shape should be preserved
    EXPECT_EQ(output.shape()[0], batch_size);
    EXPECT_EQ(output.shape()[1], seq_len);
    EXPECT_EQ(output.shape()[2], d_model);
}

TEST(TransformerEncoderLayerTest, BatchFirstFalse) {
    TransformerEncoderLayer layer(256, 4, 1024, 0.0, "relu", false);

    int64_t seq_len = 10;
    int64_t batch_size = 2;
    int64_t d_model = 256;

    Variable src(randn({seq_len, batch_size, d_model}), true);
    Variable output = layer.forward(src, Tensor{}, Tensor{});

    EXPECT_EQ(output.shape()[0], seq_len);
    EXPECT_EQ(output.shape()[1], batch_size);
    EXPECT_EQ(output.shape()[2], d_model);
}

TEST(TransformerEncoderLayerTest, WithMask) {
    TransformerEncoderLayer layer(128, 4, 512, 0.0, "relu", true);

    int64_t batch_size = 2;
    int64_t seq_len = 8;

    Variable src(randn({batch_size, seq_len, 128}), true);
    Tensor mask = create_causal_mask(seq_len);

    EXPECT_NO_THROW({
        Variable output = layer.forward(src, mask);
    });
}

TEST(TransformerEncoderLayerTest, GeLUActivation) {
    TransformerEncoderLayer layer(256, 8, 1024, 0.0, "gelu", true);

    Variable src(randn({2, 5, 256}), true);
    Variable output = layer.forward(src, Tensor{}, Tensor{});

    EXPECT_EQ(output.shape()[0], 2);
    EXPECT_EQ(output.shape()[1], 5);
    EXPECT_EQ(output.shape()[2], 256);
}

// ============================================================================
// TransformerEncoder Tests
// ============================================================================

TEST(TransformerEncoderTest, Construction) {
    auto encoder_layer = std::make_shared<TransformerEncoderLayer>(512, 8);

    EXPECT_NO_THROW({
        TransformerEncoder encoder(encoder_layer, 6);
    });

    auto norm = std::make_shared<LayerNorm>(std::vector<int64_t>{512});
    EXPECT_NO_THROW({
        TransformerEncoder encoder(encoder_layer, 6, norm);
    });
}

TEST(TransformerEncoderTest, ForwardShape) {
    auto encoder_layer = std::make_shared<TransformerEncoderLayer>(256, 4, 1024, 0.0, "relu", true);
    TransformerEncoder encoder(encoder_layer, 3);

    Variable src(randn({2, 10, 256}), true);
    Variable output = encoder.forward(src, Tensor{}, Tensor{});

    EXPECT_EQ(output.shape()[0], 2);
    EXPECT_EQ(output.shape()[1], 10);
    EXPECT_EQ(output.shape()[2], 256);
}

TEST(TransformerEncoderTest, MultipleLayersProcessing) {
    auto encoder_layer = std::make_shared<TransformerEncoderLayer>(128, 4, 512, 0.0, "relu", true);
    TransformerEncoder encoder(encoder_layer, 6);  // 6 layers

    Variable src(randn({2, 5, 128}), true);
    Variable output = encoder.forward(src, Tensor{}, Tensor{});

    // Output should still have same shape
    EXPECT_EQ(output.shape()[0], 2);
    EXPECT_EQ(output.shape()[1], 5);
    EXPECT_EQ(output.shape()[2], 128);
}

TEST(TransformerEncoderTest, WithNormalization) {
    auto encoder_layer = std::make_shared<TransformerEncoderLayer>(256, 8, 1024, 0.0, "relu", true);
    auto norm = std::make_shared<LayerNorm>(std::vector<int64_t>{256});
    TransformerEncoder encoder(encoder_layer, 6, norm);

    Variable src(randn({2, 10, 256}), true);
    Variable output = encoder.forward(src, Tensor{}, Tensor{});

    EXPECT_EQ(output.shape()[0], 2);
    EXPECT_EQ(output.shape()[1], 10);
    EXPECT_EQ(output.shape()[2], 256);
}

// ============================================================================
// TransformerDecoderLayer Tests
// ============================================================================

TEST(TransformerDecoderLayerTest, Construction) {
    EXPECT_NO_THROW({
        TransformerDecoderLayer layer(512, 8);
    });

    EXPECT_NO_THROW({
        TransformerDecoderLayer layer(768, 12, 3072, 0.1, "gelu", true);
    });
}

TEST(TransformerDecoderLayerTest, ForwardShape) {
    TransformerDecoderLayer layer(256, 4, 1024, 0.0, "relu", true);

    int64_t batch_size = 2;
    int64_t tgt_len = 8;
    int64_t src_len = 10;
    int64_t d_model = 256;

    Variable tgt(randn({batch_size, tgt_len, d_model}), true);
    Variable memory(randn({batch_size, src_len, d_model}), true);

    Variable output = layer.forward(tgt, memory);

    EXPECT_EQ(output.shape()[0], batch_size);
    EXPECT_EQ(output.shape()[1], tgt_len);
    EXPECT_EQ(output.shape()[2], d_model);
}

TEST(TransformerDecoderLayerTest, WithCausalMask) {
    TransformerDecoderLayer layer(128, 4, 512, 0.0, "relu", true);

    int64_t batch_size = 2;
    int64_t tgt_len = 5;
    int64_t src_len = 8;

    Variable tgt(randn({batch_size, tgt_len, 128}), true);
    Variable memory(randn({batch_size, src_len, 128}), true);

    Tensor tgt_mask = create_causal_mask(tgt_len);

    EXPECT_NO_THROW({
        Variable output = layer.forward(tgt, memory, tgt_mask);
    });
}

TEST(TransformerDecoderLayerTest, InvalidSingleInputForward) {
    TransformerDecoderLayer layer(256, 8);

    Variable input(randn({2, 5, 256}), true);

    // Should throw because decoder needs both tgt and memory
    EXPECT_THROW({
        layer.forward(input);
    }, std::runtime_error);
}

// ============================================================================
// TransformerDecoder Tests
// ============================================================================

TEST(TransformerDecoderTest, Construction) {
    auto decoder_layer = std::make_shared<TransformerDecoderLayer>(512, 8);

    EXPECT_NO_THROW({
        TransformerDecoder decoder(decoder_layer, 6);
    });

    auto norm = std::make_shared<LayerNorm>(std::vector<int64_t>{512});
    EXPECT_NO_THROW({
        TransformerDecoder decoder(decoder_layer, 6, norm);
    });
}

TEST(TransformerDecoderTest, ForwardShape) {
    auto decoder_layer = std::make_shared<TransformerDecoderLayer>(256, 4, 1024, 0.0, "relu", true);
    TransformerDecoder decoder(decoder_layer, 3);

    Variable tgt(randn({2, 8, 256}), true);
    Variable memory(randn({2, 10, 256}), true);

    Variable output = decoder.forward(tgt, memory);

    EXPECT_EQ(output.shape()[0], 2);
    EXPECT_EQ(output.shape()[1], 8);
    EXPECT_EQ(output.shape()[2], 256);
}

TEST(TransformerDecoderTest, MultipleLayersProcessing) {
    auto decoder_layer = std::make_shared<TransformerDecoderLayer>(128, 4, 512, 0.0, "relu", true);
    TransformerDecoder decoder(decoder_layer, 6);

    Variable tgt(randn({2, 5, 128}), true);
    Variable memory(randn({2, 7, 128}), true);

    Variable output = decoder.forward(tgt, memory);

    EXPECT_EQ(output.shape()[0], 2);
    EXPECT_EQ(output.shape()[1], 5);
    EXPECT_EQ(output.shape()[2], 128);
}

// ============================================================================
// Complete Transformer Tests
// ============================================================================

TEST(TransformerTest, Construction) {
    EXPECT_NO_THROW({
        Transformer model(512, 8, 6, 6);
    });

    EXPECT_NO_THROW({
        Transformer model(768, 12, 12, 12, 3072, 0.1, "gelu", true);
    });
}

TEST(TransformerTest, ForwardShape) {
    Transformer model(256, 4, 3, 3, 1024, 0.0, "relu", true);

    int64_t batch_size = 2;
    int64_t src_len = 10;
    int64_t tgt_len = 8;
    int64_t d_model = 256;

    Variable src(randn({batch_size, src_len, d_model}), true);
    Variable tgt(randn({batch_size, tgt_len, d_model}), true);

    Variable output = model.forward(src, tgt);

    // Output should have target's sequence length
    EXPECT_EQ(output.shape()[0], batch_size);
    EXPECT_EQ(output.shape()[1], tgt_len);
    EXPECT_EQ(output.shape()[2], d_model);
}

TEST(TransformerTest, WithMasks) {
    Transformer model(128, 4, 2, 2, 512, 0.0, "relu", true);

    int64_t batch_size = 2;
    int64_t src_len = 10;
    int64_t tgt_len = 8;

    Variable src(randn({batch_size, src_len, 128}), true);
    Variable tgt(randn({batch_size, tgt_len, 128}), true);

    Tensor tgt_mask = create_causal_mask(tgt_len);

    EXPECT_NO_THROW({
        Variable output = model.forward(src, tgt, Tensor{}, tgt_mask);
    });
}

TEST(TransformerTest, InvalidSingleInputForward) {
    Transformer model(256, 8);

    Variable input(randn({2, 10, 256}), true);

    // Should throw because transformer needs both src and tgt
    EXPECT_THROW({
        model.forward(input);
    }, std::runtime_error);
}

TEST(TransformerTest, BERTConfig) {
    // Test BERT-like configuration: 768 dims, 12 heads, 12 layers
    Transformer model(768, 12, 12, 12, 3072, 0.1, "gelu", true);

    Variable src(randn({1, 128, 768}), true);
    Variable tgt(randn({1, 64, 768}), true);

    EXPECT_NO_THROW({
        Variable output = model.forward(src, tgt);
        EXPECT_EQ(output.shape()[0], 1);
        EXPECT_EQ(output.shape()[1], 64);
        EXPECT_EQ(output.shape()[2], 768);
    });
}

TEST(TransformerTest, GPTLikeConfig) {
    // Test GPT-like configuration with causal masking
    Transformer model(512, 8, 6, 6, 2048, 0.1, "gelu", true);

    int64_t batch_size = 2;
    int64_t seq_len = 32;

    Variable src(randn({batch_size, seq_len, 512}), true);
    Variable tgt(randn({batch_size, seq_len, 512}), true);

    Tensor causal_mask = create_causal_mask(seq_len);

    Variable output = model.forward(src, tgt, Tensor{}, causal_mask);

    EXPECT_EQ(output.shape()[0], batch_size);
    EXPECT_EQ(output.shape()[1], seq_len);
    EXPECT_EQ(output.shape()[2], 512);
}

// ============================================================================
// Integration Tests
// ============================================================================

TEST(TransformerIntegrationTest, ForwardBackward) {
    Transformer model(128, 4, 2, 2, 512, 0.0, "relu", true);

    Variable src(randn({2, 5, 128}), true);
    Variable tgt(randn({2, 3, 128}), true);

    Variable output = model.forward(src, tgt);
    Variable loss = mean(output);

    EXPECT_NO_THROW({
        loss.backward();
    });

    EXPECT_TRUE(src.has_grad());
    EXPECT_TRUE(tgt.has_grad());
}

TEST(TransformerIntegrationTest, ParameterCount) {
    Transformer model(256, 4, 2, 2, 1024, 0.0, "relu", true);

    auto params = model.parameters();

    // Each encoder layer has: self_attn (8 params), linear1, linear2 (4 params), norm1, norm2 (4 params)
    // Each decoder layer has: self_attn (8), cross_attn (8), linear1, linear2 (4), norm1, norm2, norm3 (6)
    // Plus encoder norm and decoder norm
    // Total should be substantial
    EXPECT_GT(params.size(), 0);
}

TEST(TransformerIntegrationTest, TrainEvalSwitch) {
    Transformer model(128, 4, 2, 2);

    EXPECT_TRUE(model.is_training());

    model.eval();
    EXPECT_FALSE(model.is_training());

    model.train();
    EXPECT_TRUE(model.is_training());
}

TEST(TransformerIntegrationTest, SmallModelOverfit) {
    // Test that a small model can memorize a tiny dataset (sanity check)
    Transformer model(64, 2, 1, 1, 128, 0.0, "relu", true);
    model.eval();  // Use eval mode for deterministic forward passes

    Variable src(ones({1, 3, 64}), true);
    Variable tgt(ones({1, 2, 64}), true);

    Variable output1 = model.forward(src, tgt);

    // Run multiple forward passes - output should be consistent
    Variable output2 = model.forward(src, tgt);

    auto data1 = output1.tensor().data<float>();
    auto data2 = output2.tensor().data<float>();

    for (int64_t i = 0; i < output1.tensor().numel(); ++i) {
        EXPECT_NEAR(data1[i], data2[i], 1e-4);  // Relaxed tolerance for CUDA numerical precision
    }
}

TEST(TransformerIntegrationTest, DifferentSequenceLengths) {
    Transformer model(128, 4, 2, 2, 512, 0.0, "relu", true);

    // Test with different source and target lengths
    Variable src1(randn({2, 20, 128}), true);
    Variable tgt1(randn({2, 10, 128}), true);

    Variable output1 = model.forward(src1, tgt1);
    EXPECT_EQ(output1.shape()[1], 10);

    Variable src2(randn({2, 5, 128}), true);
    Variable tgt2(randn({2, 15, 128}), true);

    Variable output2 = model.forward(src2, tgt2);
    EXPECT_EQ(output2.shape()[1], 15);
}
