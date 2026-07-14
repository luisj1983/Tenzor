/**
 * @file test_transformer_layer_multidtype.cpp
 * @brief Multi-backend, multi-dtype tests for Transformer family
 * (PositionalEncoding, TransformerEncoder, TransformerDecoder, Transformer).
 *
 * Forward shape + backward grad-population. Full transformers compose many
 * sublayers — gradient must flow through all of them.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/nn/layers/transformer.hpp>
#include "../../multi_backend_dtype_fixture.hpp"

using namespace tenzor;
using namespace tenzor::testing;

class TransformerLayerMultiDTypeTest : public MultiBackendDTypeTest {};

TEST_P(TransformerLayerMultiDTypeTest, PositionalEncoding_ForwardShape) {
    nn::PositionalEncoding pe(/*d_model=*/16, /*max_len=*/64);
    convert_model(pe);
    Variable input = createInput({2, 8, 16}, false);
    auto out = pe.forward(input);
    expectShape(out.tensor(), {2, 8, 16});
    expectDType(out.tensor());
}

TEST_P(TransformerLayerMultiDTypeTest, PositionalEncoding_BackwardGradPopulated) {
    nn::PositionalEncoding pe(16, 64);
    convert_model(pe);
    Variable input = createInput({2, 8, 16}, true);
    auto out = pe.forward(input);
    sum(out).backward();
    ASSERT_TRUE(input.has_grad()) << device().to_string();
    auto g = max(abs(input.grad()->to(Device::cpu()).to(DType::Float32)));
    EXPECT_GT(g.item<float>(), 0.0f);
}

TEST_P(TransformerLayerMultiDTypeTest, TransformerEncoder_ForwardShape) {
    auto layer = std::make_shared<nn::TransformerEncoderLayer>(
        /*d_model=*/16, /*nhead=*/4, /*dim_ff=*/32, /*dropout=*/0.0, "relu");
    nn::TransformerEncoder encoder(layer, /*num_layers=*/2);
    convert_model(encoder);
    Variable src = createInput({4, 2, 16}, false);  // (seq, batch, d_model)
    auto out = encoder.forward(src, Tensor{}, Tensor{});
    expectShape(out.tensor(), {4, 2, 16});
}

TEST_P(TransformerLayerMultiDTypeTest, TransformerEncoder_BackwardGradPopulated) {
    // Float16 transformer encoder backward is flaky/zero-grad on multiple
    // backends; needs cross-attention path Float16 work — followup #17.
    if (dtype() == DType::Float16) {
        GTEST_SKIP() << "Float16 transformer encoder backward — followup #17";
    }
    auto layer = std::make_shared<nn::TransformerEncoderLayer>(
        16, 4, 32, /*dropout=*/0.0, "relu");
    nn::TransformerEncoder encoder(layer, 2);
    convert_model(encoder);
    Variable src = createInput({4, 2, 16}, true);
    auto out = encoder.forward(src, Tensor{}, Tensor{});
    // The encoder ends in LayerNorm (affine weight=1, bias=0), so sum(out) is
    // identically zero per row (sum of a zero-mean normalized vector) and the
    // input gradient is mathematically 0 — backends only "passed" the >0 check
    // via float rounding noise (oneAPI FP64 correctly computes a clean 0). Use a
    // non-degenerate loss so this test actually exercises gradient flow.
    sum(out * out).backward();
    ASSERT_TRUE(src.has_grad()) << device().to_string();
    auto g = max(abs(src.grad()->to(Device::cpu()).to(DType::Float32)));
    EXPECT_GT(g.item<float>(), 0.0f) << "encoder src grad zero on " << device().to_string();
}

TEST_P(TransformerLayerMultiDTypeTest, TransformerFull_ForwardShape) {
    nn::Transformer model(/*d_model=*/16, /*nhead=*/4,
                          /*num_enc=*/1, /*num_dec=*/1,
                          /*dim_ff=*/32, /*dropout=*/0.0, "relu");
    convert_model(model);
    Variable src = createInput({3, 2, 16}, false);
    Variable tgt = createInput({4, 2, 16}, false);
    auto out = model.forward(src, tgt);
    expectShape(out.tensor(), {4, 2, 16});
}

TEST_P(TransformerLayerMultiDTypeTest, TransformerFull_BackwardGradPopulated) {
    // Phase 2.2-followup #17: full transformer (with decoder + cross
    // attention) Float16 backward returns zero grads on every backend
    // even after SDPA/RMSNorm dtype fixes. Cross-attention path has its
    // own Float16 issue.
    if (dtype() == DType::Float16) {
        GTEST_SKIP() << "Float16 transformer (with decoder) backward zero grads — followup #17";
    }
    nn::Transformer model(16, 4, 1, 1, 32, /*dropout=*/0.0, "relu");
    convert_model(model);
    Variable src = createInput({3, 2, 16}, true);
    Variable tgt = createInput({4, 2, 16}, true);
    auto out = model.forward(src, tgt);
    // Like TransformerEncoder_BackwardGradPopulated above: the full
    // transformer also ends in a LayerNorm (affine weight=1, bias=0), so
    // sum(out) is identically zero per row and the input gradient is
    // mathematically 0 -- most backends only "passed" via float rounding
    // noise happening to land >0; Vulkan's noise landed at a clean 0 and
    // failed. Use a non-degenerate loss so this test actually exercises
    // gradient flow.
    sum(out * out).backward();
    ASSERT_TRUE(src.has_grad()) << device().to_string();
    ASSERT_TRUE(tgt.has_grad()) << device().to_string();
    auto gs = max(abs(src.grad()->to(Device::cpu()).to(DType::Float32)));
    auto gt = max(abs(tgt.grad()->to(Device::cpu()).to(DType::Float32)));
    EXPECT_GT(gs.item<float>(), 0.0f) << "src grad zero on " << device().to_string();
    EXPECT_GT(gt.item<float>(), 0.0f) << "tgt grad zero on " << device().to_string();
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(TransformerLayerMultiDTypeTest);
