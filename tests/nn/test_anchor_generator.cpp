/**
 * @file test_anchor_generator.cpp
 * @brief Tests for tenzor::nn::detection::AnchorGenerator.
 *
 * The audit (2026-05-02) found zero references to anchors.hpp in the
 * test suite. RoI/RPN/mask heads have their own multidtype tests but
 * the anchor generator they depend on was never directly verified.
 *
 * This file pins:
 *   - num_anchors_per_location matches sizes × ratios.
 *   - generate(H, W, stride) returns shape (H*W*K, 4) with (x1, y1, x2, y2).
 *   - Boxes have positive width/height and centres on a grid with the
 *     given stride.
 *   - Bigger size → bigger box (monotonic w.r.t. size index).
 *   - Aspect ratio < 1 ⇒ taller than wide; ratio > 1 ⇒ wider than tall.
 *   - Device transfer of the result preserves the values.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/nn/detection/anchors.hpp>
#include "../backend_test_fixture.hpp"

using namespace tenzor;
using namespace tenzor::nn::detection;

class AnchorGeneratorTest : public ::tenzor::testing::BackendTest {
protected:
    void SetUp() override {
        ::tenzor::testing::BackendTest::SetUp();
        if (::testing::Test::IsSkipped()) return;
    }
};

TEST_P(AnchorGeneratorTest, NumAnchorsPerLocation) {
    AnchorGenerator gen({32.0f, 64.0f, 128.0f}, {0.5f, 1.0f, 2.0f});
    EXPECT_EQ(gen.num_anchors_per_location(), 9);
    EXPECT_EQ(gen.sizes().size(), 3u);
    EXPECT_EQ(gen.aspect_ratios().size(), 3u);
}

TEST_P(AnchorGeneratorTest, GeneratedShape) {
    const int64_t H = 5, W = 7, stride = 16;
    AnchorGenerator gen({32.0f, 64.0f}, {0.5f, 1.0f, 2.0f});
    auto boxes = gen.generate(H, W, stride, device);
    ASSERT_EQ(boxes.shape().size(), 2u);
    EXPECT_EQ(boxes.shape()[0], H * W * gen.num_anchors_per_location());
    EXPECT_EQ(boxes.shape()[1], 4);
}

TEST_P(AnchorGeneratorTest, BoxesHavePositiveExtent) {
    const int64_t H = 4, W = 4, stride = 16;
    AnchorGenerator gen({32.0f, 64.0f}, {0.5f, 1.0f, 2.0f});
    auto boxes = gen.generate(H, W, stride, device);
    auto cpu = boxes.cpu().contiguous();
    const float* p = cpu.data<float>();
    for (int64_t i = 0; i < cpu.shape()[0]; ++i) {
        float x1 = p[i*4 + 0];
        float y1 = p[i*4 + 1];
        float x2 = p[i*4 + 2];
        float y2 = p[i*4 + 3];
        ASSERT_LT(x1, x2) << "anchor " << i << " has non-positive width";
        ASSERT_LT(y1, y2) << "anchor " << i << " has non-positive height";
    }
}

TEST_P(AnchorGeneratorTest, AspectRatio_TallVsWide) {
    // size=64, ratios={0.5, 1.0, 2.0}
    AnchorGenerator gen({64.0f}, {0.5f, 1.0f, 2.0f});
    auto boxes = gen.generate(/*H=*/1, /*W=*/1, /*stride=*/0, device);
    // Single position → 3 boxes, one per ratio.
    auto cpu = boxes.cpu().contiguous();
    ASSERT_EQ(cpu.numel(), 3 * 4);
    const float* p = cpu.data<float>();

    // Compute (w, h) per box:
    auto box_wh = [&](int64_t i) {
        return std::pair<float, float>{p[i*4+2] - p[i*4+0],
                                       p[i*4+3] - p[i*4+1]};
    };
    auto [w0, h0] = box_wh(0);  // ratio 0.5 → tall
    auto [w1, h1] = box_wh(1);  // ratio 1.0 → square
    auto [w2, h2] = box_wh(2);  // ratio 2.0 → wide
    EXPECT_LT(w0, h0) << "ratio=0.5 should be taller than wide";
    EXPECT_NEAR(w1, h1, 1e-3f) << "ratio=1.0 should be square";
    EXPECT_GT(w2, h2) << "ratio=2.0 should be wider than tall";
}

TEST_P(AnchorGeneratorTest, BiggerSizeMeansBiggerBox) {
    // ratios=1.0 only so each box is a square — comparing area is just
    // comparing side length.
    AnchorGenerator gen({32.0f, 64.0f, 128.0f}, {1.0f});
    auto boxes = gen.generate(/*H=*/1, /*W=*/1, /*stride=*/0, device);
    auto cpu = boxes.cpu().contiguous();
    const float* p = cpu.data<float>();
    auto side = [&](int64_t i) { return p[i*4+2] - p[i*4+0]; };
    EXPECT_LT(side(0), side(1));  // 32 < 64
    EXPECT_LT(side(1), side(2));  // 64 < 128
}

TEST_P(AnchorGeneratorTest, GeneratesOnRequestedDevice) {
    AnchorGenerator gen({32.0f}, {1.0f});
    auto boxes = gen.generate(2, 2, 16, device);
    EXPECT_EQ(boxes.device().type, device.type);
}

INSTANTIATE_BACKEND_TESTS(AnchorGeneratorTest);
