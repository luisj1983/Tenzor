/**
 * @file test_roi_head_multidtype.cpp
 * @brief Multi-dtype tests for RoIBoxHead and RoIHead detection modules
 *
 * Tests RoIBoxHead and RoIHead from the detection module with Float32, Float64,
 * and Float16 dtypes across CPU, CUDA, OneAPI, Vulkan, and ROCm backends to ensure:
 * - Correct construction and parameter existence
 * - Proper output shapes from forward_features()
 * - DType preservation through the forward pass
 * - Gradient flow through the box head
 * - Finite outputs (no NaN/Inf)
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/nn/detection/roi_head.hpp>
#include "../../multi_backend_dtype_fixture.hpp"
#include <cmath>

using namespace tenzor;
using namespace tenzor::nn;
using namespace tenzor::nn::detection;
using namespace tenzor::testing;

// ============================================================================
// RoIBoxHead Multi-Backend Multi-DType Tests
// ============================================================================

class RoIHeadMultiDTypeTest : public MultiBackendDTypeTest {};

TEST_P(RoIHeadMultiDTypeTest, BoxHead_Construction) {
    // RoIBoxHead with 256 input channels, 7x7 roi size, 20 classes
    auto box_head = RoIBoxHead(256, 7, 20);
    convert_model(box_head);

    // Verify parameters exist (should have FC layers + cls/bbox heads)
    auto params = box_head.parameters();
    EXPECT_FALSE(params.empty()) << "RoIBoxHead should have learnable parameters";
}

TEST_P(RoIHeadMultiDTypeTest, BoxHead_ForwardOutputShapes) {
    auto box_head = RoIBoxHead(256, 7, 20);
    convert_model(box_head);

    // 50 RoIs, 256 channels, 7x7 spatial
    Variable roi_features = createInput({50, 256, 7, 7}, false);
    auto [class_logits, box_deltas] = box_head.forward_features(roi_features);

    // class_logits: (num_rois, num_classes + 1) = (50, 21)
    expectShape(class_logits.tensor(), {50, 21});
    // box_deltas: (num_rois, num_classes * 4) = (50, 80)
    expectShape(box_deltas.tensor(), {50, 80});
}

TEST_P(RoIHeadMultiDTypeTest, BoxHead_ForwardOutputDType) {
    auto box_head = RoIBoxHead(256, 7, 20);
    convert_model(box_head);

    Variable roi_features = createInput({10, 256, 7, 7}, false);
    auto [class_logits, box_deltas] = box_head.forward_features(roi_features);

    // Both outputs should preserve the test dtype
    expectDType(class_logits.tensor());
    expectDType(box_deltas.tensor());
}

TEST_P(RoIHeadMultiDTypeTest, BoxHead_ForwardSingleROI) {
    auto box_head = RoIBoxHead(256, 7, 20);
    convert_model(box_head);

    // Single RoI input
    Variable roi_features = createInput({1, 256, 7, 7}, false);
    auto [class_logits, box_deltas] = box_head.forward_features(roi_features);

    // class_logits: (1, 21), box_deltas: (1, 80)
    expectShape(class_logits.tensor(), {1, 21});
    expectShape(box_deltas.tensor(), {1, 80});
}

TEST_P(RoIHeadMultiDTypeTest, BoxHead_BackwardGradientFlow) {
    auto box_head = RoIBoxHead(256, 7, 20);
    convert_model(box_head);

    Variable roi_features = createInput({4, 256, 7, 7}, true);
    auto [class_logits, box_deltas] = box_head.forward_features(roi_features);

    // Backward through class_logits
    auto cls_shape = class_logits.shape();
    std::vector<int64_t> cls_shape_vec(cls_shape.begin(), cls_shape.end());
    auto grad_output = tenzor::ones(cls_shape_vec, dtype(), device());
    class_logits.backward(grad_output);

    // Input should have gradients
    EXPECT_TRUE(roi_features.has_grad())
        << "Input should have gradients after backward pass";
}

TEST_P(RoIHeadMultiDTypeTest, BoxHead_OutputsFinite) {
    auto box_head = RoIBoxHead(256, 7, 20);
    convert_model(box_head);

    Variable roi_features = createInput({8, 256, 7, 7}, false);
    auto [class_logits, box_deltas] = box_head.forward_features(roi_features);

    // Check no NaN or Inf in class_logits
    float cls_max = compute_max_abs(class_logits.tensor());
    EXPECT_TRUE(std::isfinite(cls_max))
        << "class_logits contains NaN or Inf values";

    // Check no NaN or Inf in box_deltas
    float box_max = compute_max_abs(box_deltas.tensor());
    EXPECT_TRUE(std::isfinite(box_max))
        << "box_deltas contains NaN or Inf values";
}

// ============================================================================
// RoIHead Multi-Backend Multi-DType Tests
// ============================================================================

TEST_P(RoIHeadMultiDTypeTest, RoIHead_Construction) {
    // RoIHead with 256 input channels, 20 classes, default parameters
    auto roi_head = RoIHead(256, 20);
    convert_model(roi_head);

    // Verify parameters exist
    auto params = roi_head.parameters();
    EXPECT_FALSE(params.empty()) << "RoIHead should have learnable parameters";
}

TEST_P(RoIHeadMultiDTypeTest, RoIHead_ParameterCount) {
    auto roi_head = RoIHead(256, 20);
    convert_model(roi_head);

    auto params = roi_head.parameters();
    EXPECT_GT(params.size(), 0u)
        << "RoIHead should have multiple parameter tensors";

    // Verify total parameter count is reasonable (FC layers + heads)
    size_t total_params = countParameters(params);
    EXPECT_GT(total_params, 0u)
        << "RoIHead should have a non-zero number of learnable parameters";
}

// ============================================================================
// Test Instantiation
// ============================================================================

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(RoIHeadMultiDTypeTest);
