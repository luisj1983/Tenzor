/**
 * @file test_patch_embedding_multidtype.cpp
 * @brief Multi-backend, multi-dtype tests for nn::PatchEmbedding (ViT layer).
 *
 * PatchEmbedding wraps a Conv2d with stride=patch_size to produce
 * (N, num_patches, embed_dim). Verify output shape, dtype, and that
 * backward populates input + parameter gradients.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/nn/layers/vision.hpp>
#include "../../multi_backend_dtype_fixture.hpp"
#include "../../grad_flow_helpers.hpp"

using namespace tenzor;
using namespace tenzor::testing;

class PatchEmbeddingMultiDTypeTest : public MultiBackendDTypeTest {};

TEST_P(PatchEmbeddingMultiDTypeTest, ForwardShape) {
    nn::PatchEmbedding pe(/*in_channels=*/3, /*embed_dim=*/64,
                          /*patch_size=*/4, /*img_size=*/16);
    convert_model(pe);
    Variable input = createInput({2, 3, 16, 16}, /*requires_grad=*/false);
    auto output = pe.forward(input);
    // num_patches = (16/4)^2 = 16; output [N=2, num_patches=16, embed_dim=64].
    expectShape(output.tensor(), {2, 16, 64});
    expectDType(output.tensor());
    EXPECT_EQ(pe.num_patches(), 16);
}

TEST_P(PatchEmbeddingMultiDTypeTest, BackwardGradPopulated) {
    nn::PatchEmbedding pe(3, 32, 4, 16);
    convert_model(pe);
    Variable input = createInput({1, 3, 16, 16}, /*requires_grad=*/true);
    auto output = pe.forward(input);
    sum(output).backward();

    EXPECT_GRAD_FLOWS(input);
    EXPECT_EQ(input.grad()->numel(), input.tensor().numel());

    // Verify the wrapped Conv2d weights got gradient too.
    int populated = 0;
    for (auto& [name, param] : pe.named_parameters()) {
        if (param->has_grad()) {
            auto p_max = max(abs(param->grad()->to(Device::cpu()).to(DType::Float32)));
            if (p_max.item<float>() > 0.0f) ++populated;
        }
    }
    EXPECT_GT(populated, 0) << "no PatchEmbedding param grad populated on " << device().to_string();
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(PatchEmbeddingMultiDTypeTest);
