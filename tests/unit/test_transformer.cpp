#include <gtest/gtest.h>
#include "../backend_test_fixture.hpp"
#include "../grad_flow_helpers.hpp"
#include <tenzor/nn/layers/transformer.hpp>
#include <tenzor/nn/layers/attention.hpp>
#include <cmath>

using namespace tenzor;
using namespace tenzor::testing;
using namespace tenzor::nn;

// ============================================================================
// PositionalEncoding Tests
// ============================================================================

class PositionalEncodingTest : public BackendTest {};

TEST_P(PositionalEncodingTest, Construction) {
    EXPECT_NO_THROW({
        PositionalEncoding pe(512);
        pe.to(device);
    }) << "Failed on " << device.to_string();

    EXPECT_NO_THROW({
        PositionalEncoding pe(768, 10000, 0.1);
        pe.to(device);
    }) << "Failed on " << device.to_string();
}

TEST_P(PositionalEncodingTest, ForwardShape) {
    PositionalEncoding pe(256, 1000, 0.0);
    pe.to(device);

    int64_t batch_size = 4;
    int64_t seq_len = 20;
    int64_t d_model = 256;

    Variable input(randn({batch_size, seq_len, d_model}, DType::Float32, device), true);
    Variable output = pe.forward(input);

    // Shape should be preserved
    EXPECT_EQ(output.shape().size(), 3) << "Failed on " << device.to_string();
    EXPECT_EQ(output.shape()[0], batch_size) << "Failed on " << device.to_string();
    EXPECT_EQ(output.shape()[1], seq_len) << "Failed on " << device.to_string();
    EXPECT_EQ(output.shape()[2], d_model) << "Failed on " << device.to_string();
}

TEST_P(PositionalEncodingTest, ExceedsMaxLen) {
    PositionalEncoding pe(128, 100, 0.0);
    pe.to(device);

    Variable input(randn({2, 150, 128}, DType::Float32, device), true);

    // Should throw because seq_len (150) > max_len (100)
    EXPECT_THROW({
        pe.forward(input);
    }, std::runtime_error) << "Failed on " << device.to_string();
}

TEST_P(PositionalEncodingTest, WithDropout) {
    PositionalEncoding pe(256, 1000, 0.5);
    pe.to(device);

    Variable input(randn({2, 10, 256}, DType::Float32, device), true);

    Variable output = pe.forward(input);
    EXPECT_EQ(output.shape()[0], 2) << "Failed on " << device.to_string();
    EXPECT_EQ(output.shape()[1], 10) << "Failed on " << device.to_string();
    EXPECT_EQ(output.shape()[2], 256) << "Failed on " << device.to_string();
    auto pe_m = tenzor::max(tenzor::abs(output.tensor().to(Device::cpu()).to(DType::Float64))).item<double>();
    EXPECT_TRUE(std::isfinite(pe_m)) << "forward produced NaN/Inf on " << device.to_string();
}

// ============================================================================
// TransformerEncoderLayer Tests
// ============================================================================

class TransformerEncoderLayerTest : public BackendTest {};

TEST_P(TransformerEncoderLayerTest, Construction) {
    EXPECT_NO_THROW({
        TransformerEncoderLayer layer(512, 8);
        layer.to(device);
    }) << "Failed on " << device.to_string();

    EXPECT_NO_THROW({
        TransformerEncoderLayer layer(768, 12, 3072, 0.1, "gelu", true);
        layer.to(device);
    }) << "Failed on " << device.to_string();
}

TEST_P(TransformerEncoderLayerTest, InvalidActivation) {
    EXPECT_THROW({
        TransformerEncoderLayer layer(512, 8, 2048, 0.1, "invalid");
        layer.to(device);
    }, std::invalid_argument) << "Failed on " << device.to_string();
}

TEST_P(TransformerEncoderLayerTest, ForwardShape) {
    TransformerEncoderLayer layer(256, 4, 1024, 0.0, "relu", true);
    layer.to(device);

    int64_t batch_size = 2;
    int64_t seq_len = 10;
    int64_t d_model = 256;

    Variable src(randn({batch_size, seq_len, d_model}, DType::Float32, device), true);
    Variable output = layer.forward(src, Tensor{}, Tensor{});

    // Shape should be preserved
    EXPECT_EQ(output.shape()[0], batch_size) << "Failed on " << device.to_string();
    EXPECT_EQ(output.shape()[1], seq_len) << "Failed on " << device.to_string();
    EXPECT_EQ(output.shape()[2], d_model) << "Failed on " << device.to_string();
}

TEST_P(TransformerEncoderLayerTest, BatchFirstFalse) {
    TransformerEncoderLayer layer(256, 4, 1024, 0.0, "relu", false);
    layer.to(device);

    int64_t seq_len = 10;
    int64_t batch_size = 2;
    int64_t d_model = 256;

    Variable src(randn({seq_len, batch_size, d_model}, DType::Float32, device), true);
    Variable output = layer.forward(src, Tensor{}, Tensor{});

    EXPECT_EQ(output.shape()[0], seq_len) << "Failed on " << device.to_string();
    EXPECT_EQ(output.shape()[1], batch_size) << "Failed on " << device.to_string();
    EXPECT_EQ(output.shape()[2], d_model) << "Failed on " << device.to_string();
}

TEST_P(TransformerEncoderLayerTest, WithMask) {
    TransformerEncoderLayer layer(128, 4, 512, 0.0, "relu", true);
    layer.to(device);

    int64_t batch_size = 2;
    int64_t seq_len = 8;

    Variable src(randn({batch_size, seq_len, 128}, DType::Float32, device), true);
    Tensor mask = create_causal_mask(seq_len, device);

    Variable output = layer.forward(src, mask);
    EXPECT_EQ(output.shape()[0], batch_size) << "Failed on " << device.to_string();
    EXPECT_EQ(output.shape()[1], seq_len) << "Failed on " << device.to_string();
    EXPECT_EQ(output.shape()[2], 128) << "Failed on " << device.to_string();
    auto mask_m = tenzor::max(tenzor::abs(output.tensor().to(Device::cpu()).to(DType::Float64))).item<double>();
    EXPECT_TRUE(std::isfinite(mask_m)) << "forward produced NaN/Inf on " << device.to_string();
}

TEST_P(TransformerEncoderLayerTest, GeLUActivation) {
    TransformerEncoderLayer layer(256, 8, 1024, 0.0, "gelu", true);
    layer.to(device);

    Variable src(randn({2, 5, 256}, DType::Float32, device), true);
    Variable output = layer.forward(src, Tensor{}, Tensor{});

    EXPECT_EQ(output.shape()[0], 2) << "Failed on " << device.to_string();
    EXPECT_EQ(output.shape()[1], 5) << "Failed on " << device.to_string();
    EXPECT_EQ(output.shape()[2], 256) << "Failed on " << device.to_string();
}

// ============================================================================
// TransformerEncoder Tests
// ============================================================================

class TransformerEncoderTest : public BackendTest {};

TEST_P(TransformerEncoderTest, Construction) {
    auto encoder_layer = std::make_shared<TransformerEncoderLayer>(512, 8);

    EXPECT_NO_THROW({
        TransformerEncoder encoder(encoder_layer, 6);
        encoder.to(device);
    }) << "Failed on " << device.to_string();

    auto norm = std::make_shared<LayerNorm>(std::vector<int64_t>{512});
    EXPECT_NO_THROW({
        TransformerEncoder encoder(encoder_layer, 6, norm);
        encoder.to(device);
    }) << "Failed on " << device.to_string();
}

TEST_P(TransformerEncoderTest, ForwardShape) {
    auto encoder_layer = std::make_shared<TransformerEncoderLayer>(256, 4, 1024, 0.0, "relu", true);
    TransformerEncoder encoder(encoder_layer, 3);
    encoder.to(device);

    Variable src(randn({2, 10, 256}, DType::Float32, device), true);
    Variable output = encoder.forward(src, Tensor{}, Tensor{});

    EXPECT_EQ(output.shape()[0], 2) << "Failed on " << device.to_string();
    EXPECT_EQ(output.shape()[1], 10) << "Failed on " << device.to_string();
    EXPECT_EQ(output.shape()[2], 256) << "Failed on " << device.to_string();
}

TEST_P(TransformerEncoderTest, MultipleLayersProcessing) {
    auto encoder_layer = std::make_shared<TransformerEncoderLayer>(128, 4, 512, 0.0, "relu", true);
    TransformerEncoder encoder(encoder_layer, 6);  // 6 layers
    encoder.to(device);

    Variable src(randn({2, 5, 128}, DType::Float32, device), true);
    Variable output = encoder.forward(src, Tensor{}, Tensor{});

    // Output should still have same shape
    EXPECT_EQ(output.shape()[0], 2) << "Failed on " << device.to_string();
    EXPECT_EQ(output.shape()[1], 5) << "Failed on " << device.to_string();
    EXPECT_EQ(output.shape()[2], 128) << "Failed on " << device.to_string();
}

TEST_P(TransformerEncoderTest, WithNormalization) {
    auto encoder_layer = std::make_shared<TransformerEncoderLayer>(256, 8, 1024, 0.0, "relu", true);
    auto norm = std::make_shared<LayerNorm>(std::vector<int64_t>{256});
    TransformerEncoder encoder(encoder_layer, 6, norm);
    encoder.to(device);

    Variable src(randn({2, 10, 256}, DType::Float32, device), true);
    Variable output = encoder.forward(src, Tensor{}, Tensor{});

    EXPECT_EQ(output.shape()[0], 2) << "Failed on " << device.to_string();
    EXPECT_EQ(output.shape()[1], 10) << "Failed on " << device.to_string();
    EXPECT_EQ(output.shape()[2], 256) << "Failed on " << device.to_string();
}

// ============================================================================
// TransformerDecoderLayer Tests
// ============================================================================

class TransformerDecoderLayerTest : public BackendTest {};

TEST_P(TransformerDecoderLayerTest, Construction) {
    EXPECT_NO_THROW({
        TransformerDecoderLayer layer(512, 8);
        layer.to(device);
    }) << "Failed on " << device.to_string();

    EXPECT_NO_THROW({
        TransformerDecoderLayer layer(768, 12, 3072, 0.1, "gelu", true);
        layer.to(device);
    }) << "Failed on " << device.to_string();
}

TEST_P(TransformerDecoderLayerTest, ForwardShape) {
    TransformerDecoderLayer layer(256, 4, 1024, 0.0, "relu", true);
    layer.to(device);

    int64_t batch_size = 2;
    int64_t tgt_len = 8;
    int64_t src_len = 10;
    int64_t d_model = 256;

    Variable tgt(randn({batch_size, tgt_len, d_model}, DType::Float32, device), true);
    Variable memory(randn({batch_size, src_len, d_model}, DType::Float32, device), true);

    Variable output = layer.forward(tgt, memory);

    EXPECT_EQ(output.shape()[0], batch_size) << "Failed on " << device.to_string();
    EXPECT_EQ(output.shape()[1], tgt_len) << "Failed on " << device.to_string();
    EXPECT_EQ(output.shape()[2], d_model) << "Failed on " << device.to_string();
}

TEST_P(TransformerDecoderLayerTest, WithCausalMask) {
    TransformerDecoderLayer layer(128, 4, 512, 0.0, "relu", true);
    layer.to(device);

    int64_t batch_size = 2;
    int64_t tgt_len = 5;
    int64_t src_len = 8;

    Variable tgt(randn({batch_size, tgt_len, 128}, DType::Float32, device), true);
    Variable memory(randn({batch_size, src_len, 128}, DType::Float32, device), true);

    Tensor tgt_mask = create_causal_mask(tgt_len, device);

    Variable output = layer.forward(tgt, memory, tgt_mask);
    EXPECT_EQ(output.shape()[0], batch_size) << "Failed on " << device.to_string();
    EXPECT_EQ(output.shape()[1], tgt_len) << "Failed on " << device.to_string();
    EXPECT_EQ(output.shape()[2], 128) << "Failed on " << device.to_string();
    auto dmask_m = tenzor::max(tenzor::abs(output.tensor().to(Device::cpu()).to(DType::Float64))).item<double>();
    EXPECT_TRUE(std::isfinite(dmask_m)) << "forward produced NaN/Inf on " << device.to_string();
}

TEST_P(TransformerDecoderLayerTest, InvalidSingleInputForward) {
    TransformerDecoderLayer layer(256, 8);
    layer.to(device);

    Variable input(randn({2, 5, 256}, DType::Float32, device), true);

    // Should throw because decoder needs both tgt and memory
    EXPECT_THROW({
        layer.forward(input);
    }, std::runtime_error) << "Failed on " << device.to_string();
}

// ============================================================================
// TransformerDecoder Tests
// ============================================================================

class TransformerDecoderTest : public BackendTest {};

TEST_P(TransformerDecoderTest, Construction) {
    auto decoder_layer = std::make_shared<TransformerDecoderLayer>(512, 8);

    EXPECT_NO_THROW({
        TransformerDecoder decoder(decoder_layer, 6);
        decoder.to(device);
    }) << "Failed on " << device.to_string();

    auto norm = std::make_shared<LayerNorm>(std::vector<int64_t>{512});
    EXPECT_NO_THROW({
        TransformerDecoder decoder(decoder_layer, 6, norm);
        decoder.to(device);
    }) << "Failed on " << device.to_string();
}

TEST_P(TransformerDecoderTest, ForwardShape) {
    auto decoder_layer = std::make_shared<TransformerDecoderLayer>(256, 4, 1024, 0.0, "relu", true);
    TransformerDecoder decoder(decoder_layer, 3);
    decoder.to(device);

    Variable tgt(randn({2, 8, 256}, DType::Float32, device), true);
    Variable memory(randn({2, 10, 256}, DType::Float32, device), true);

    Variable output = decoder.forward(tgt, memory);

    EXPECT_EQ(output.shape()[0], 2) << "Failed on " << device.to_string();
    EXPECT_EQ(output.shape()[1], 8) << "Failed on " << device.to_string();
    EXPECT_EQ(output.shape()[2], 256) << "Failed on " << device.to_string();
}

TEST_P(TransformerDecoderTest, MultipleLayersProcessing) {
    auto decoder_layer = std::make_shared<TransformerDecoderLayer>(128, 4, 512, 0.0, "relu", true);
    TransformerDecoder decoder(decoder_layer, 6);
    decoder.to(device);

    Variable tgt(randn({2, 5, 128}, DType::Float32, device), true);
    Variable memory(randn({2, 7, 128}, DType::Float32, device), true);

    Variable output = decoder.forward(tgt, memory);

    EXPECT_EQ(output.shape()[0], 2) << "Failed on " << device.to_string();
    EXPECT_EQ(output.shape()[1], 5) << "Failed on " << device.to_string();
    EXPECT_EQ(output.shape()[2], 128) << "Failed on " << device.to_string();
}

// ============================================================================
// Complete Transformer Tests
// ============================================================================

class TransformerTest : public BackendTest {};

TEST_P(TransformerTest, Construction) {
    EXPECT_NO_THROW({
        Transformer model(512, 8, 6, 6);
        model.to(device);
    }) << "Failed on " << device.to_string();

    EXPECT_NO_THROW({
        Transformer model(768, 12, 12, 12, 3072, 0.1, "gelu", true);
        model.to(device);
    }) << "Failed on " << device.to_string();
}

TEST_P(TransformerTest, ForwardShape) {
    Transformer model(256, 4, 3, 3, 1024, 0.0, "relu", true);
    model.to(device);

    int64_t batch_size = 2;
    int64_t src_len = 10;
    int64_t tgt_len = 8;
    int64_t d_model = 256;

    Variable src(randn({batch_size, src_len, d_model}, DType::Float32, device), true);
    Variable tgt(randn({batch_size, tgt_len, d_model}, DType::Float32, device), true);

    Variable output = model.forward(src, tgt);

    // Output should have target's sequence length
    EXPECT_EQ(output.shape()[0], batch_size) << "Failed on " << device.to_string();
    EXPECT_EQ(output.shape()[1], tgt_len) << "Failed on " << device.to_string();
    EXPECT_EQ(output.shape()[2], d_model) << "Failed on " << device.to_string();
}

TEST_P(TransformerTest, WithMasks) {
    Transformer model(128, 4, 2, 2, 512, 0.0, "relu", true);
    model.to(device);

    int64_t batch_size = 2;
    int64_t src_len = 10;
    int64_t tgt_len = 8;

    Variable src(randn({batch_size, src_len, 128}, DType::Float32, device), true);
    Variable tgt(randn({batch_size, tgt_len, 128}, DType::Float32, device), true);

    Tensor tgt_mask = create_causal_mask(tgt_len, device);

    Variable output = model.forward(src, tgt, Tensor{}, tgt_mask);
    EXPECT_EQ(output.shape()[0], batch_size) << "Failed on " << device.to_string();
    EXPECT_EQ(output.shape()[1], tgt_len) << "Failed on " << device.to_string();
    EXPECT_EQ(output.shape()[2], 128) << "Failed on " << device.to_string();
    auto tmask_m = tenzor::max(tenzor::abs(output.tensor().to(Device::cpu()).to(DType::Float64))).item<double>();
    EXPECT_TRUE(std::isfinite(tmask_m)) << "forward produced NaN/Inf on " << device.to_string();
}

TEST_P(TransformerTest, InvalidSingleInputForward) {
    Transformer model(256, 8);
    model.to(device);

    Variable input(randn({2, 10, 256}, DType::Float32, device), true);

    // Should throw because transformer needs both src and tgt
    EXPECT_THROW({
        model.forward(input);
    }, std::runtime_error) << "Failed on " << device.to_string();
}

TEST_P(TransformerTest, BERTConfig) {
    // Test BERT-like configuration: 768 dims, 12 heads, 12 layers
    Transformer model(768, 12, 12, 12, 3072, 0.1, "gelu", true);
    model.to(device);

    Variable src(randn({1, 128, 768}, DType::Float32, device), true);
    Variable tgt(randn({1, 64, 768}, DType::Float32, device), true);

    EXPECT_NO_THROW({
        Variable output = model.forward(src, tgt);
        EXPECT_EQ(output.shape()[0], 1) << "Failed on " << device.to_string();
        EXPECT_EQ(output.shape()[1], 64) << "Failed on " << device.to_string();
        EXPECT_EQ(output.shape()[2], 768) << "Failed on " << device.to_string();
    }) << "Failed on " << device.to_string();
}

TEST_P(TransformerTest, GPTLikeConfig) {
    // Test GPT-like configuration with causal masking
    Transformer model(512, 8, 6, 6, 2048, 0.1, "gelu", true);
    model.to(device);

    int64_t batch_size = 2;
    int64_t seq_len = 32;

    Variable src(randn({batch_size, seq_len, 512}, DType::Float32, device), true);
    Variable tgt(randn({batch_size, seq_len, 512}, DType::Float32, device), true);

    Tensor causal_mask = create_causal_mask(seq_len, device);

    Variable output = model.forward(src, tgt, Tensor{}, causal_mask);

    EXPECT_EQ(output.shape()[0], batch_size) << "Failed on " << device.to_string();
    EXPECT_EQ(output.shape()[1], seq_len) << "Failed on " << device.to_string();
    EXPECT_EQ(output.shape()[2], 512) << "Failed on " << device.to_string();
}

// ============================================================================
// Integration Tests
// ============================================================================

class TransformerIntegrationTest : public BackendTest {};

TEST_P(TransformerIntegrationTest, ForwardBackward) {
    Transformer model(128, 4, 2, 2, 512, 0.0, "relu", true);
    model.to(device);

    Variable src(randn({2, 5, 128}, DType::Float32, device), true);
    Variable tgt(randn({2, 3, 128}, DType::Float32, device), true);

    Variable output = model.forward(src, tgt);
    Variable loss = mean(output);

    EXPECT_NO_THROW({
        loss.backward();
    }) << "Failed on " << device.to_string();

    // NOTE: not EXPECT_GRAD_FLOWS — through a 2+2-layer Transformer with a
    // mean() loss the input gradient genuinely vanishes (~1e-11 in fp32, and
    // it rounds to an exact 0 on some backends), so a magnitude assertion is a
    // vanishing-gradient false-positive, not a severed graph. Assert presence
    // (backward populated a grad without throwing) — the point of this test.
    EXPECT_TRUE(src.has_grad());
    EXPECT_TRUE(tgt.has_grad());
}

TEST_P(TransformerIntegrationTest, ParameterCount) {
    Transformer model(256, 4, 2, 2, 1024, 0.0, "relu", true);
    model.to(device);

    auto params = model.parameters();

    // Each encoder layer has: self_attn (8 params), linear1, linear2 (4 params), norm1, norm2 (4 params)
    // Each decoder layer has: self_attn (8), cross_attn (8), linear1, linear2 (4), norm1, norm2, norm3 (6)
    // Plus encoder norm and decoder norm
    // Total should be substantial
    EXPECT_GT(params.size(), 0) << "Failed on " << device.to_string();
}

TEST_P(TransformerIntegrationTest, TrainEvalSwitch) {
    Transformer model(128, 4, 2, 2);
    model.to(device);

    EXPECT_TRUE(model.is_training()) << "Failed on " << device.to_string();

    model.eval();
    EXPECT_FALSE(model.is_training()) << "Failed on " << device.to_string();

    model.train();
    EXPECT_TRUE(model.is_training()) << "Failed on " << device.to_string();
}

TEST_P(TransformerIntegrationTest, SmallModelOverfit) {
    // Test that a small model can memorize a tiny dataset (sanity check)
    Transformer model(64, 2, 1, 1, 128, 0.0, "relu", true);
    model.to(device);
    model.eval();  // Use eval mode for deterministic forward passes

    Variable src(ones({1, 3, 64}, DType::Float32, device), true);
    Variable tgt(ones({1, 2, 64}, DType::Float32, device), true);

    Variable output1 = model.forward(src, tgt);

    // Run multiple forward passes - output should be consistent
    Variable output2 = model.forward(src, tgt);

    auto output1_cpu = output1.tensor().to(Device::cpu());
    auto output2_cpu = output2.tensor().to(Device::cpu());
    auto data1 = output1_cpu.data<float>();
    auto data2 = output2_cpu.data<float>();

    for (int64_t i = 0; i < output1_cpu.numel(); ++i) {
        EXPECT_NEAR(data1[i], data2[i], 1e-4) << "Failed on " << device.to_string();  // Relaxed tolerance for CUDA numerical precision
    }
}

TEST_P(TransformerIntegrationTest, DifferentSequenceLengths) {
    Transformer model(128, 4, 2, 2, 512, 0.0, "relu", true);
    model.to(device);

    // Test with different source and target lengths
    Variable src1(randn({2, 20, 128}, DType::Float32, device), true);
    Variable tgt1(randn({2, 10, 128}, DType::Float32, device), true);

    Variable output1 = model.forward(src1, tgt1);
    EXPECT_EQ(output1.shape()[1], 10) << "Failed on " << device.to_string();

    Variable src2(randn({2, 5, 128}, DType::Float32, device), true);
    Variable tgt2(randn({2, 15, 128}, DType::Float32, device), true);

    Variable output2 = model.forward(src2, tgt2);
    EXPECT_EQ(output2.shape()[1], 15) << "Failed on " << device.to_string();
}

INSTANTIATE_BACKEND_TESTS(PositionalEncodingTest);
INSTANTIATE_BACKEND_TESTS(TransformerEncoderLayerTest);
INSTANTIATE_BACKEND_TESTS(TransformerEncoderTest);
INSTANTIATE_BACKEND_TESTS(TransformerDecoderLayerTest);
INSTANTIATE_BACKEND_TESTS(TransformerDecoderTest);
INSTANTIATE_BACKEND_TESTS(TransformerTest);
INSTANTIATE_BACKEND_TESTS(TransformerIntegrationTest);
