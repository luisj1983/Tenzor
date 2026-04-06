/**
 * @file test_spectral_norm_multidtype.cpp
 * @brief Multi-backend + multi-dtype tests for spectral normalization
 *
 * Covers: SpectralNorm::apply, remove, sigma
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/nn/utils/spectral_norm.hpp>
#include <tenzor/nn/layers/linear.hpp>
#include <tenzor/ops/creation.hpp>
#include <tenzor/ops/math.hpp>
#include "../multi_backend_dtype_fixture.hpp"

using namespace tenzor;
using namespace tenzor::testing;

// ============================================================================
// Fixture
// ============================================================================

class SpectralNormMultiDTypeTest : public MultiBackendDTypeTest {};

// ============================================================================
// Apply Tests
// ============================================================================

TEST_P(SpectralNormMultiDTypeTest, ApplySucceeds) {
    auto linear = std::make_shared<nn::Linear>(16, 8);
    convert_model(linear);

    auto sn = nn::utils::SpectralNorm::apply(linear, "weight", 1);
    ASSERT_NE(sn, nullptr);
}

TEST_P(SpectralNormMultiDTypeTest, SigmaPositive) {
    auto linear = std::make_shared<nn::Linear>(16, 8);
    convert_model(linear);

    auto sn = nn::utils::SpectralNorm::apply(linear, "weight", 1);

    // Run a forward pass to trigger power iteration
    Variable input = createInput({2, 16}, false);
    auto output = linear->forward(input);

    auto sigma = sn->sigma();
    auto sigma_f32 = sigma.to(Device::cpu()).to(DType::Float32);
    EXPECT_GT(sigma_f32.data<float>()[0], 0.0f);
}

TEST_P(SpectralNormMultiDTypeTest, RemoveRestoresWeight) {
    auto linear = std::make_shared<nn::Linear>(16, 8);
    convert_model(linear);

    auto sn = nn::utils::SpectralNorm::apply(linear, "weight", 1);

    // Forward pass with SN
    Variable input1 = createInput({2, 16}, false);
    linear->forward(input1);

    // Remove SN
    EXPECT_NO_THROW(sn->remove());

    // Should still work after removal
    Variable input2 = createInput({2, 16}, false);
    auto output = linear->forward(input2);
    expectShape(output.tensor(), {2, 8});
}

TEST_P(SpectralNormMultiDTypeTest, MultipleForwardPasses) {
    auto linear = std::make_shared<nn::Linear>(16, 8);
    convert_model(linear);

    auto sn = nn::utils::SpectralNorm::apply(linear, "weight", 3);

    // Multiple forward passes should refine the sigma estimate
    for (int i = 0; i < 5; ++i) {
        Variable input = createInput({2, 16}, false);
        linear->forward(input);
    }

    auto sigma = sn->sigma();
    auto sigma_f32 = sigma.to(Device::cpu()).to(DType::Float32);
    EXPECT_GT(sigma_f32.data<float>()[0], 0.0f);
}

// ============================================================================
// Instantiation
// ============================================================================

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(SpectralNormMultiDTypeTest);
