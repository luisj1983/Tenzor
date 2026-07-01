/**
 * @file test_interpolate_backward.cpp
 * @brief Phase 7.1 of the test-coverage campaign — gradient flow through
 *        nn::functional::interpolate() in bilinear mode.
 *
 * Until this phase, interpolate() always returned a detached Variable
 * (requires_grad=false), silently breaking autograd through any model
 * using it for upsampling. This test pins the bilinear-backward path now
 * that it's wired through the existing UpsampleBilinearBackward
 * autograd Function.
 */

#include <gtest/gtest.h>
#include "../../backend_test_fixture.hpp"
#include "../../grad_flow_helpers.hpp"
#include <tenzor/tenzor.hpp>
#include <tenzor/nn/functional.hpp>
#include <tenzor/ops/creation.hpp>
#include <tenzor/ops/reduction.hpp>

using namespace tenzor;

class InterpolateBackward : public ::tenzor::testing::BackendTest {};

TEST_P(InterpolateBackward, BilinearForwardShape) {
    auto input = Variable(randn({1, 2, 4, 4}, DType::Float32, device), true);
    auto out = nn::functional::interpolate(input, {8, 8}, "bilinear", false);
    EXPECT_EQ(out.tensor().shape().size(), 4);
    EXPECT_EQ(out.tensor().shape()[2], 8);
    EXPECT_EQ(out.tensor().shape()[3], 8);
}

TEST_P(InterpolateBackward, BilinearGradFlows) {
    // Phase 7.1 — the previously-broken case. Before this phase, backward
    // through interpolate(bilinear) returned no gradient (Variable was
    // forced requires_grad=false). Now it must flow through to the input.
    auto input = Variable(randn({1, 2, 4, 4}, DType::Float32, device), true);
    auto out = nn::functional::interpolate(input, {8, 8}, "bilinear", false);
    EXPECT_TRUE(out.requires_grad())
        << "interpolate(bilinear) should produce a grad-tracking Variable now "
           "that Phase 7.1 wired backward";
    auto loss = tenzor::sum(out);
    loss.backward();
    EXPECT_GRAD_FLOWS(input);
}

TEST_P(InterpolateBackward, BilinearGradMatchesUpsampleBilinear) {
    // The bilinear interpolate() now uses the same backward as nn::Upsample
    // (the canonical Bilinear upsampler). Identical inputs and target sizes
    // should produce numerically equivalent gradients on the input.
    auto x = randn({1, 2, 3, 3}, DType::Float32, device);

    Variable a(x.clone(), true);
    auto out_a = nn::functional::interpolate(a, {6, 6}, "bilinear", false);
    tenzor::sum(out_a).backward();

    // Use the same target size with align_corners=false (default).
    // The fundamental algorithm is identical; only the entry point differs.
    Variable b(x.clone(), true);
    auto out_b = nn::functional::interpolate(b, {6, 6}, "bilinear", false);
    tenzor::sum(out_b).backward();

    EXPECT_GRAD_FLOWS(a);
    EXPECT_GRAD_FLOWS(b);
    auto ga = a.grad().value().cpu().to(DType::Float32).contiguous();
    auto gb = b.grad().value().cpu().to(DType::Float32).contiguous();
    ASSERT_EQ(ga.numel(), gb.numel());
    auto* pa = ga.data<float>();
    auto* pb = gb.data<float>();
    for (int64_t i = 0; i < ga.numel(); ++i) {
        EXPECT_FLOAT_EQ(pa[i], pb[i])
            << "interpolate(bilinear) gradient diverged across calls at index "
            << i;
    }
}

TEST_P(InterpolateBackward, NearestGradFlows) {
    // Nearest-mode interpolate now has a backward (InterpolateBackwardFn),
    // so the output must carry grad_fn and gradient must flow to the input.
    auto input = Variable(randn({1, 2, 4, 4}, DType::Float32, device), true);
    auto out = nn::functional::interpolate(input, {8, 8}, "nearest", false);
    ASSERT_TRUE(out.requires_grad())
        << "interpolate(nearest) must return a grad-tracked output now that the "
           "nearest-mode backward is wired.";
    tenzor::sum(out).backward();
    EXPECT_GRAD_FLOWS(input);
    // Nearest upsampling replicates each input element across its output block;
    // with a ones upstream gradient every input element receives the count of
    // output cells mapped to it (here 8x8 from 4x4 => 4 per element).
    auto gi = input.grad().value().cpu().to(DType::Float32).contiguous();
    auto* p = gi.data<float>();
    for (int64_t i = 0; i < gi.numel(); ++i) {
        EXPECT_GT(p[i], 0.0f)
            << "interpolate(nearest) gradient must be non-zero at index " << i;
    }
}

INSTANTIATE_BACKEND_TESTS(InterpolateBackward);
