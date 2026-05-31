/**
 * @file test_color_jitter.cpp
 * @brief Regression tests for ColorJitter — audit item I.11.
 *
 * Two previous gaps:
 *  - contrast used a single global mean across all images/channels,
 *    not the per-image luminance mean that torchvision uses.
 *  - hue was documented as "not applied in this simplified version"
 *    and silently no-op'd.
 */

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "../backend_test_fixture.hpp"
#include "tenzor/data/transforms.hpp"
#include "tenzor/core/dtype.hpp"
#include "tenzor/core/device.hpp"
#include "tenzor/core/tensor.hpp"
#include "tenzor/tenzor.hpp"

using namespace tenzor;
using namespace tenzor::data;
using namespace tenzor::data::transforms;

namespace {

class ColorJitterTest : public ::tenzor::testing::BackendTest {
protected:
    void SetUp() override {
        ::tenzor::testing::BackendTest::SetUp();
        if (::testing::Test::IsSkipped()) return;
    }

    /// Build a [3, 2, 2] RGB image with distinct colour per channel, on `device`.
    Tensor make_rgb(float r, float g, float b) const {
        Tensor host({3, 2, 2}, DType::Float32, Device::cpu());
        auto* p = host.data<float>();
        // Channel-major: R-plane, G-plane, B-plane each 2x2 spatial.
        for (int i = 0; i < 4; ++i) p[i]     = r;
        for (int i = 0; i < 4; ++i) p[4 + i] = g;
        for (int i = 0; i < 4; ++i) p[8 + i] = b;
        return host.to(device);
    }
};

// ---------------------------------------------------------------------------
// I.11 — Hue shift must actually rotate pixel values in HSV space.
// ---------------------------------------------------------------------------
TEST_P(ColorJitterTest, HueRotationChangesPixels) {
    tenzor::manual_seed(42);
    // Pure red input — hue 0°, sat 1.0, val 1.0.
    auto input = make_rgb(1.0f, 0.0f, 0.0f);

    // hue=0.5 ⇒ uniform draw in [-0.5, 0.5]: with seed=42 a non-zero shift
    // is essentially guaranteed.  Force a non-trivial draw by sampling many
    // times until we see a meaningful rotation.
    bool any_changed = false;
    for (int attempt = 0; attempt < 4; ++attempt) {
        ColorJitter jitter(/*brightness=*/0.0f, /*contrast=*/0.0f,
                           /*saturation=*/0.0f, /*hue=*/0.5f);
        auto cloned = input;  // ColorJitter writes to a copy already
        auto [out, _] = jitter(cloned, Tensor{});
        auto out_cpu = out.cpu();
        const auto* o = out_cpu.data<float>();
        // Original is (R=1, G=0, B=0).  After ANY non-zero hue shift it
        // should differ in G or B by at least a small amount.
        const float g_max = std::max({o[4], o[5], o[6], o[7]});
        const float b_max = std::max({o[8], o[9], o[10], o[11]});
        if (g_max > 1e-3f || b_max > 1e-3f) {
            any_changed = true;
            break;
        }
    }
    EXPECT_TRUE(any_changed)
        << "hue shift did not change pixel values after 4 attempts — "
           "audit item I.11 fix may have regressed";
}

// ---------------------------------------------------------------------------
// I.11 — Contrast must use per-image luminance mean, not a global mean.
//
// Build a batch of two distinct uniform images; with the correct per-image
// behaviour, each image individually keeps the SAME pixel values after
// "contrast" with factor=1 around its OWN mean.  With the buggy global
// mean, the two images shift toward a single common reference, so the
// per-image identity invariant is broken.
// ---------------------------------------------------------------------------
TEST_P(ColorJitterTest, ContrastUsesPerImageMean) {
    tenzor::manual_seed(7);
    // Build a 2-image batch [N=2, C=3, H=2, W=2] where each image is
    // uniform: image 0 = (0.2, 0.2, 0.2); image 1 = (0.8, 0.8, 0.8).
    Tensor host({2, 3, 2, 2}, DType::Float32, Device::cpu());
    auto* p = host.data<float>();
    for (int i = 0; i < 12; ++i)  p[i]      = 0.2f;
    for (int i = 0; i < 12; ++i)  p[12 + i] = 0.8f;
    auto input = host.to(device);

    // contrast=0 ⇒ U[1, 1] = factor 1, so the math reduces to
    // mean + 1*(x - mean) = x.  But the loop only fires when contrast_ > 0
    // (the factor range U[1-c, 1+c] is degenerate at c=0).  Use a tiny
    // contrast so the factor is ~1, then assert the per-image identity
    // is preserved on a uniform image (factor*0 = 0).
    ColorJitter jitter(/*brightness=*/0.0f, /*contrast=*/0.05f,
                       /*saturation=*/0.0f, /*hue=*/0.0f);
    auto [out, _] = jitter(input, Tensor{});
    auto out_cpu = out.cpu();
    const auto* o = out_cpu.data<float>();
    // Image 0 was uniformly 0.2 ⇒ stays uniformly 0.2 regardless of factor.
    for (int i = 0; i < 12; ++i) {
        EXPECT_NEAR(o[i], 0.2f, 1e-5f) << "image 0 should stay uniform at i=" << i;
    }
    // Image 1 was uniformly 0.8 ⇒ stays uniformly 0.8.  With the previous
    // global-mean bug, image 1 would shift toward the global mean ((0.2 +
    // 0.8) / 2 = 0.5), failing this check.
    for (int i = 0; i < 12; ++i) {
        EXPECT_NEAR(o[12 + i], 0.8f, 1e-5f) << "image 1 should stay uniform at i=" << i;
    }
}

INSTANTIATE_BACKEND_TESTS(ColorJitterTest);

}  // namespace
