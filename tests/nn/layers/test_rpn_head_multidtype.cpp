/**
 * @file test_rpn_head_multidtype.cpp
 * @brief Multi-dtype tests for RPNHead (Region Proposal Network Head)
 *
 * Tests RPNHead with Float32, Float64, and Float16 dtypes across
 * CPU, CUDA, OneAPI, Vulkan, and ROCm backends to ensure:
 * - Correct output shapes for objectness logits and box regression
 * - Dtype preservation through forward pass
 * - Gradient flow through the network
 * - Numerical stability (no NaN/Inf in outputs)
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/nn/detection/rpn.hpp>
#include "../../multi_backend_dtype_fixture.hpp"
#include <cmath>

using namespace tenzor;
using namespace tenzor::nn;
using namespace tenzor::nn::detection;
using namespace tenzor::testing;

// ============================================================================
// RPNHead Multi-Backend Multi-DType Test Fixture
// ============================================================================

class RPNHeadMultiDTypeTest : public MultiBackendDTypeTest {
protected:
    /**
     * @brief Move RPNHead sub-modules to the test device and dtype.
     *
     * RPNHead is not a Module, so convert_model() won't work.
     * We manually move each Conv2d sub-module instead.
     */
    void convertRPNHead(RPNHead& head) {
        head.conv_->to(device());
        head.cls_logits_->to(device());
        head.bbox_pred_->to(device());
        if (dtype() != DType::Float32) {
            head.conv_->to(dtype());
            head.cls_logits_->to(dtype());
            head.bbox_pred_->to(dtype());
        }
    }
};

// ============================================================================
// Construction Test
// ============================================================================

TEST_P(RPNHeadMultiDTypeTest, Construction) {
    EXPECT_NO_THROW({
        RPNHead head(256, 3);
        convertRPNHead(head);
    });
}

// ============================================================================
// Forward Output Shape Tests
// ============================================================================

TEST_P(RPNHeadMultiDTypeTest, ForwardOutputShapes) {
    RPNHead head(256, 3);
    convertRPNHead(head);

    // Input: (N=2, C=256, H=16, W=16), num_anchors=3
    auto input = createInput({2, 256, 16, 16}, false);
    auto [objectness, box_regression] = head.forward(input);

    // objectness: (N, num_anchors * H * W) = (2, 3*16*16) = (2, 768)
    expectShape(objectness.tensor(), {2, 768});

    // box_regression: (N, num_anchors * H * W, 4) = (2, 768, 4)
    expectShape(box_regression.tensor(), {2, 768, 4});
}

TEST_P(RPNHeadMultiDTypeTest, ForwardDifferentFeatureMapSizes) {
    RPNHead head(256, 9);
    convertRPNHead(head);

    // Input: (N=1, C=256, H=32, W=32), num_anchors=9
    auto input = createInput({1, 256, 32, 32}, false);
    auto [objectness, box_regression] = head.forward(input);

    // objectness: (1, 9*32*32) = (1, 9216)
    expectShape(objectness.tensor(), {1, 9216});

    // box_regression: (1, 9216, 4)
    expectShape(box_regression.tensor(), {1, 9216, 4});
}

TEST_P(RPNHeadMultiDTypeTest, ForwardSingleAnchor) {
    RPNHead head(128, 1);
    convertRPNHead(head);

    // Input: (N=4, C=128, H=8, W=8), num_anchors=1
    auto input = createInput({4, 128, 8, 8}, false);
    auto [objectness, box_regression] = head.forward(input);

    // objectness: (4, 1*8*8) = (4, 64)
    expectShape(objectness.tensor(), {4, 64});

    // box_regression: (4, 64, 4)
    expectShape(box_regression.tensor(), {4, 64, 4});
}

// ============================================================================
// DType Preservation Test
// ============================================================================

TEST_P(RPNHeadMultiDTypeTest, ForwardOutputDType) {
    RPNHead head(256, 3);
    convertRPNHead(head);

    auto input = createInput({2, 256, 16, 16}, false);
    auto [objectness, box_regression] = head.forward(input);

    expectDType(objectness.tensor());
    expectDType(box_regression.tensor());
}

// ============================================================================
// Gradient Flow Test
// ============================================================================

TEST_P(RPNHeadMultiDTypeTest, BackwardGradientFlow) {
    RPNHead head(256, 3);
    convertRPNHead(head);

    auto input = createInput({1, 256, 8, 8}, true);
    auto [objectness, box_regression] = head.forward(input);

    // Backward through objectness sum
    auto loss = tenzor::sum(objectness);
    loss.backward();

    ASSERT_TRUE(input.grad().has_value());
    expectShape(input.grad().value(), {1, 256, 8, 8});
}

// ============================================================================
// Numerical Stability Tests
// ============================================================================

TEST_P(RPNHeadMultiDTypeTest, ObjectnessScoresFinite) {
    RPNHead head(256, 3);
    convertRPNHead(head);

    auto input = createInput({2, 256, 8, 8}, false);
    auto [objectness, box_regression] = head.forward(input);

    // Move to CPU Float32 for inspection
    auto obj_cpu = objectness.tensor().to(Device::cpu());
    if (obj_cpu.dtype() != DType::Float32) {
        obj_cpu = obj_cpu.to(DType::Float32);
    }
    const float* data = obj_cpu.data<float>();
    for (int64_t i = 0; i < obj_cpu.numel(); ++i) {
        EXPECT_TRUE(std::isfinite(data[i]))
            << "Non-finite objectness score at index " << i
            << ": " << data[i];
    }
}

TEST_P(RPNHeadMultiDTypeTest, BoxDeltasFinite) {
    RPNHead head(256, 3);
    convertRPNHead(head);

    auto input = createInput({2, 256, 8, 8}, false);
    auto [objectness, box_regression] = head.forward(input);

    // Move to CPU Float32 for inspection
    auto box_cpu = box_regression.tensor().to(Device::cpu());
    if (box_cpu.dtype() != DType::Float32) {
        box_cpu = box_cpu.to(DType::Float32);
    }
    const float* data = box_cpu.data<float>();
    for (int64_t i = 0; i < box_cpu.numel(); ++i) {
        EXPECT_TRUE(std::isfinite(data[i]))
            << "Non-finite box delta at index " << i
            << ": " << data[i];
    }
}

// ============================================================================
// Test Instantiation
// ============================================================================

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(RPNHeadMultiDTypeTest);

/*
 * COVERAGE SUMMARY:
 *
 * Test Cases: 8
 * DTypes Tested: Float32, Float64, Float16
 * Backends Tested: CPU, CUDA, OneAPI, Vulkan, ROCm
 * Total Scenarios: 8 tests x 3 dtypes x 5 backends = 120 test scenarios
 *
 * Coverage:
 * - Construction: verify RPNHead(in_channels, num_anchors) does not crash
 * - Forward shapes: multiple (N, C, H, W) inputs with different num_anchors
 * - DType preservation: both outputs match the input dtype
 * - Gradient flow: backward through objectness sum reaches input
 * - Numerical stability: no NaN/Inf in objectness logits or box deltas
 */
