/**
 * @file test_transformer_containers_multidtype.cpp
 * @brief Multi-backend × multi-dtype tests for the TransformerEncoder
 *        and TransformerDecoder *container* classes (audit-2026-05-03 N2).
 *        The per-layer building blocks (TransformerEncoderLayer /
 *        TransformerDecoderLayer) are already covered elsewhere; this file
 *        exercises layer stacking, optional final norm, and grad flow
 *        through the multi-layer chain.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/nn/layers/transformer.hpp>
#include <tenzor/nn/layers/normalization.hpp>
#include <tenzor/autograd/variable.hpp>
#include "../../multi_backend_dtype_fixture.hpp"
#include "../../grad_flow_helpers.hpp"

using namespace tenzor;
using namespace tenzor::nn;
using namespace tenzor::testing;

class TransformerContainerMultiDTypeTest : public MultiBackendDTypeTest {};

namespace {
constexpr int64_t kDModel = 16;
constexpr int64_t kNHead = 4;
constexpr int64_t kFFD = 32;
constexpr int64_t kNumLayers = 3;
constexpr int64_t kBatch = 2;
constexpr int64_t kSeqLen = 5;
} // anonymous

// ============================================================================
// TransformerEncoder
// ============================================================================

TEST_P(TransformerContainerMultiDTypeTest, TransformerEncoder_ForwardShape) {
    auto layer = std::make_shared<TransformerEncoderLayer>(
        kDModel, kNHead, kFFD, /*dropout=*/0.0,
        /*activation=*/"relu", /*batch_first=*/true,
        /*norm_first=*/false);
    TransformerEncoder enc(layer, kNumLayers);
    enc.to(device());
    enc.to(dtype());

    auto src = Variable(createRandn({kBatch, kSeqLen, kDModel}), false);
    auto out = enc.forward(src, Tensor{}, Tensor{});

    ASSERT_EQ(out.shape().size(), 3u);
    EXPECT_EQ(out.shape()[0], kBatch);
    EXPECT_EQ(out.shape()[1], kSeqLen);
    EXPECT_EQ(out.shape()[2], kDModel);
    expectDType(out.tensor());
}

TEST_P(TransformerContainerMultiDTypeTest, TransformerEncoder_WithFinalNorm) {
    auto layer = std::make_shared<TransformerEncoderLayer>(
        kDModel, kNHead, kFFD, /*dropout=*/0.0,
        /*activation=*/"gelu", /*batch_first=*/true,
        /*norm_first=*/true);
    auto norm = std::make_shared<LayerNorm>(std::vector<int64_t>{kDModel});
    TransformerEncoder enc(layer, kNumLayers, norm);
    enc.to(device());
    enc.to(dtype());

    auto src = Variable(createRandn({kBatch, kSeqLen, kDModel}), false);
    auto out = enc.forward(src, Tensor{}, Tensor{});
    EXPECT_EQ(out.shape()[2], kDModel);
}

TEST_P(TransformerContainerMultiDTypeTest, TransformerEncoder_BackwardGradFlow) {
    auto layer = std::make_shared<TransformerEncoderLayer>(
        kDModel, kNHead, kFFD, /*dropout=*/0.0);
    TransformerEncoder enc(layer, kNumLayers);
    enc.to(device());
    enc.to(dtype());

    auto src = Variable(createRandn({kBatch, kSeqLen, kDModel}), true);
    auto out = enc.forward(src, Tensor{}, Tensor{});
    // Scale loss to keep gradients above Float16 underflow threshold
    // (~5.96e-8). Backward through a 3-layer transformer produces ~1e-3
    // gradients for unit-loss input, which underflows Float16 to zero
    // and trips EXPECT_GRAD_FLOWS' "max-abs > 0" check despite the
    // computation being correct on Float32+.
    auto loss = tenzor::sum(out);
    if (dtype() == DType::Float16) loss = loss * 1024.0f;
    loss.backward();
    EXPECT_GRAD_FLOWS(src);
}

// ============================================================================
// TransformerDecoder
// ============================================================================

TEST_P(TransformerContainerMultiDTypeTest, TransformerDecoder_ForwardShape) {
    auto layer = std::make_shared<TransformerDecoderLayer>(
        kDModel, kNHead, kFFD, /*dropout=*/0.0,
        /*activation=*/"relu", /*batch_first=*/true,
        /*norm_first=*/false);
    TransformerDecoder dec(layer, kNumLayers);
    dec.to(device());
    dec.to(dtype());

    auto tgt = Variable(createRandn({kBatch, kSeqLen, kDModel}), false);
    auto memory = Variable(createRandn({kBatch, kSeqLen, kDModel}), false);
    auto out = dec.forward(tgt, memory, Tensor{}, Tensor{}, Tensor{}, Tensor{});

    ASSERT_EQ(out.shape().size(), 3u);
    EXPECT_EQ(out.shape()[0], kBatch);
    EXPECT_EQ(out.shape()[1], kSeqLen);
    EXPECT_EQ(out.shape()[2], kDModel);
    expectDType(out.tensor());
}

TEST_P(TransformerContainerMultiDTypeTest, TransformerDecoder_BackwardGradFlow) {
    auto layer = std::make_shared<TransformerDecoderLayer>(
        kDModel, kNHead, kFFD, /*dropout=*/0.0);
    TransformerDecoder dec(layer, kNumLayers);
    dec.to(device());
    dec.to(dtype());

    auto tgt = Variable(createRandn({kBatch, kSeqLen, kDModel}), true);
    auto memory = Variable(createRandn({kBatch, kSeqLen, kDModel}), true);
    auto out = dec.forward(tgt, memory, Tensor{}, Tensor{}, Tensor{}, Tensor{});
    // Scale loss for Float16 — see TransformerEncoder_BackwardGradFlow.
    auto loss = tenzor::sum(out);
    if (dtype() == DType::Float16) loss = loss * 1024.0f;
    loss.backward();
    EXPECT_GRAD_FLOWS(tgt);
    EXPECT_GRAD_FLOWS(memory);
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(TransformerContainerMultiDTypeTest);
