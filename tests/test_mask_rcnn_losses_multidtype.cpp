/**
 * @file test_mask_rcnn_losses_multidtype.cpp
 * @brief Multi-dtype / multi-backend companion to test_mask_rcnn_losses.cpp.
 *
 * The plain file (BackendTest, Float32-only) builds a full Mask R-CNN
 * (ResNet50-FPN) per test and asserts the five head losses (RPN cls/box, ROI
 * cls/box, mask) are finite / non-zero / in reasonable range, plus a
 * gradient-flow regression guard. This companion adds the dtype axis across
 * {Float32, Float64, Float16} x {cpu, cuda, vulkan, oneapi, rocm, mps} via
 * MultiBackendDTypeTest.
 *
 * Image sizes, box coordinates, label values, and mask regions are kept
 * IDENTICAL to the plain file — the plain file already proved those sizes
 * fit every backend at Float32 (forward-only), and GradientFlow already uses
 * 512x512 because 800x800 *training* exceeds the 8 GB cuda backend. Rescaling
 * the inputs would change which ROIs are sampled and weaken the range
 * assertions; the only axis added here is dtype.
 *
 * Two dtype/backend combos are skipped categorically rather than left to
 * crash the suite:
 *   - Float16 / BFloat16: Mask R-CNN end-to-end is not validated in half
 *     precision (cuDNN F16 gaps through the full detector + BCE/smooth-L1
 *     loss numerics) -> NumericalDivergence.
 *   - Float64 on a non-CPU backend: the detector + saved activations at 2x
 *     the Float32 footprint exceed GPU memory for the larger image sizes
 *     (800x800 / 800x1200) used by the forward-only tests, and the
 *     GradientFlow *training* path at 2x footprint would OOM cuda even at
 *     512x512. Float64 on CPU is retained and exercises the loss math at
 *     higher precision. -> NumericalDivergence (memory, not numerics; the
 *     closest categorized reason).
 *
 * Loss scalars are read back via a dtype-safe helper (cast to Float32 before
 * .item<float>()) so the Float64 path does not narrow a Float64 tensor with
 * .item<float>().
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/models/mask_rcnn.hpp>
#include <tenzor/ops/detection.hpp>
#include <tenzor/ops/creation.hpp>

#include "multi_backend_dtype_fixture.hpp"
#include "grad_flow_helpers.hpp"

using namespace tenzor;
using namespace tenzor::models;
using namespace tenzor::testing;

// F16/BF16: end-to-end Mask R-CNN is not validated in half precision.
#define skip_if_half_rcnn()                                              \
    do {                                                                 \
        if (dtype() == DType::Float16 || dtype() == DType::BFloat16) {   \
            SKIP_WITH_REASON(                                            \
                ::tenzor::testing::SkipReason::NumericalDivergence,      \
                "Mask R-CNN end-to-end not validated in Float16/BFloat16 "\
                "(cuDNN F16 gaps + loss numerics)");                      \
        }                                                                \
    } while (0)

// Float64 on GPU: 2x the Float32 footprint exceeds GPU memory for the
// 800x800 / 800x1200 forward-only tests and the 512x512 training path. CPU
// Float64 is retained.
#define skip_if_float64_gpu()                                            \
    do {                                                                 \
        if (dtype() == DType::Float64 &&                                 \
            device().type != Device::Type::CPU) {                        \
            SKIP_WITH_REASON(                                            \
                ::tenzor::testing::SkipReason::NumericalDivergence,      \
                "Float64 Mask R-CNN forward_train exceeds GPU memory "   \
                "(2x Float32 footprint)");                                \
        }                                                                \
    } while (0)

class MaskRCNNLossMultiDType : public MultiBackendDTypeTest {
protected:
    void SetUp() override {
        MultiBackendDTypeTest::SetUp();
        if (::testing::Test::IsSkipped()) return;
        // Per-instance reseed so every (backend, dtype) combo gets identical
        // model weights — parity / range assertions are then a same-weights
        // comparison, not order-dependent RNG draws (mirrors the plain file).
        tenzor::manual_seed(42);
    }

    // Read a scalar loss as float regardless of the test dtype. The plain file
    // uses .cpu().item<float>() directly, which is only valid for Float32; the
    // companion widens to Float32 first so Float64 loss values narrow safely.
    static float loss_value(const Variable& v) {
        return v.tensor().to(Device::cpu()).to(DType::Float32).item<float>();
    }

    // Build a constant-filled Float32 image tensor on host, then move to the
    // test device and dtype.
    Tensor make_images(int64_t n, int64_t h, int64_t w, float fill) {
        Tensor t({n, 3, h, w}, DType::Float32, Device::cpu());
        t.fill_(fill);
        return t.to(device()).to(dtype());
    }
    Tensor make_masks_zeros(int64_t n, int64_t objs, int64_t h, int64_t w) {
        Tensor t({n, objs, h, w}, DType::Float32, Device::cpu());
        t.fill_(0.0f);
        return t.to(device()).to(dtype());
    }
    Tensor to_dev_dtype(const Tensor& cpu_f32) {
        return cpu_f32.to(device()).to(dtype());
    }
};

// ============================================================================
// RPN Loss Tests
// ============================================================================

TEST_P(MaskRCNNLossMultiDType, RPNLossBasic) {
    skip_if_half_rcnn();
    skip_if_float64_gpu();
    auto model = mask_rcnn_resnet50_fpn(80, false);
    convert_model(model);
    model->train();

    Variable images(make_images(1, 800, 800, 0.5f), true);

    Tensor gt_boxes_cpu({1, 5, 4}, DType::Float32, Device::cpu());
    auto* b = gt_boxes_cpu.data<float>();
    b[0] = 10.0f;  b[1] = 10.0f;  b[2] = 100.0f; b[3] = 100.0f;
    b[4] = 150.0f; b[5] = 150.0f; b[6] = 250.0f; b[7] = 250.0f;
    b[8] = 300.0f; b[9] = 300.0f; b[10] = 400.0f; b[11] = 400.0f;
    b[12] = 450.0f; b[13] = 450.0f; b[14] = 550.0f; b[15] = 550.0f;
    b[16] = 600.0f; b[17] = 600.0f; b[18] = 700.0f; b[19] = 700.0f;
    Tensor gt_boxes = to_dev_dtype(gt_boxes_cpu);

    Tensor gt_labels_cpu({1, 5}, DType::Int64, Device::cpu());
    auto* l = gt_labels_cpu.data<int64_t>();
    l[0] = 1; l[1] = 2; l[2] = 3; l[3] = 4; l[4] = 5;
    Tensor gt_labels = gt_labels_cpu.to(device());

    Tensor gt_masks = make_masks_zeros(1, 5, 800, 800);

    auto [rpn_cls_loss, rpn_box_loss, roi_cls_loss, roi_box_loss, mask_loss] =
        model->forward_train(images, gt_boxes, gt_labels, gt_masks);

    float rpn_cls = loss_value(rpn_cls_loss);
    EXPECT_GT(rpn_cls, 0.0f);
    EXPECT_LT(rpn_cls, 100.0f);
    EXPECT_FALSE(std::isnan(rpn_cls));
    EXPECT_FALSE(std::isinf(rpn_cls));

    float rpn_box = loss_value(rpn_box_loss);
    EXPECT_GE(rpn_box, 0.0f);
    EXPECT_FALSE(std::isnan(rpn_box));
    EXPECT_FALSE(std::isinf(rpn_box));
}

TEST_P(MaskRCNNLossMultiDType, RPNLossMultipleImages) {
    skip_if_half_rcnn();
    skip_if_float64_gpu();
    auto model = mask_rcnn_resnet50_fpn(80, false);
    convert_model(model);
    model->train();

    Variable images(make_images(2, 800, 800, 0.5f), true);

    Tensor gt_boxes_cpu({2, 3, 4}, DType::Float32, Device::cpu());
    auto* b = gt_boxes_cpu.data<float>();
    // Image 1
    b[0] = 50.0f;  b[1] = 50.0f;  b[2] = 150.0f; b[3] = 150.0f;
    b[4] = 200.0f; b[5] = 200.0f; b[6] = 300.0f; b[7] = 300.0f;
    b[8] = 400.0f; b[9] = 400.0f; b[10] = 500.0f; b[11] = 500.0f;
    // Image 2
    b[12] = 100.0f; b[13] = 100.0f; b[14] = 200.0f; b[15] = 200.0f;
    b[16] = 300.0f; b[17] = 300.0f; b[18] = 400.0f; b[19] = 400.0f;
    b[20] = 500.0f; b[21] = 500.0f; b[22] = 600.0f; b[23] = 600.0f;
    Tensor gt_boxes = to_dev_dtype(gt_boxes_cpu);

    Tensor gt_labels_cpu({2, 3}, DType::Int64, Device::cpu());
    auto* l = gt_labels_cpu.data<int64_t>();
    l[0] = 1; l[1] = 2; l[2] = 3; l[3] = 1; l[4] = 2; l[5] = 3;
    Tensor gt_labels = gt_labels_cpu.to(device());

    Tensor gt_masks = make_masks_zeros(2, 3, 800, 800);

    auto [rpn_cls_loss, rpn_box_loss, roi_cls_loss, roi_box_loss, mask_loss] =
        model->forward_train(images, gt_boxes, gt_labels, gt_masks);

    float rpn_cls = loss_value(rpn_cls_loss);
    EXPECT_GT(rpn_cls, 0.0f);
    EXPECT_FALSE(std::isnan(rpn_cls));
}

// ============================================================================
// ROI Head Loss Tests
// ============================================================================

TEST_P(MaskRCNNLossMultiDType, ROIHeadLossBasic) {
    skip_if_half_rcnn();
    skip_if_float64_gpu();
    auto model = mask_rcnn_resnet50_fpn(80, false);
    convert_model(model);
    model->train();

    Variable images(make_images(1, 800, 800, 0.5f), true);

    Tensor gt_boxes_cpu({1, 3, 4}, DType::Float32, Device::cpu());
    auto* b = gt_boxes_cpu.data<float>();
    b[0] = 100.0f; b[1] = 100.0f; b[2] = 200.0f; b[3] = 200.0f;
    b[4] = 300.0f; b[5] = 300.0f; b[6] = 400.0f; b[7] = 400.0f;
    b[8] = 500.0f; b[9] = 500.0f; b[10] = 600.0f; b[11] = 600.0f;
    Tensor gt_boxes = to_dev_dtype(gt_boxes_cpu);

    Tensor gt_labels_cpu({1, 3}, DType::Int64, Device::cpu());
    auto* l = gt_labels_cpu.data<int64_t>();
    l[0] = 10; l[1] = 20; l[2] = 30;
    Tensor gt_labels = gt_labels_cpu.to(device());

    Tensor gt_masks = make_masks_zeros(1, 3, 800, 800);

    auto [rpn_cls_loss, rpn_box_loss, roi_cls_loss, roi_box_loss, mask_loss] =
        model->forward_train(images, gt_boxes, gt_labels, gt_masks);

    float roi_cls = loss_value(roi_cls_loss);
    EXPECT_GT(roi_cls, 0.0f);
    EXPECT_LT(roi_cls, 100.0f);
    EXPECT_FALSE(std::isnan(roi_cls));
    EXPECT_FALSE(std::isinf(roi_cls));

    float roi_box = loss_value(roi_box_loss);
    EXPECT_GE(roi_box, 0.0f);
    EXPECT_FALSE(std::isnan(roi_box));
}

TEST_P(MaskRCNNLossMultiDType, ROIHeadLossClassImbalance) {
    skip_if_half_rcnn();
    skip_if_float64_gpu();
    auto model = mask_rcnn_resnet50_fpn(80, false);
    convert_model(model);
    model->train();

    Variable images(make_images(1, 800, 800, 0.5f), true);

    Tensor gt_boxes_cpu({1, 10, 4}, DType::Float32, Device::cpu());
    auto* b = gt_boxes_cpu.data<float>();
    for (int i = 0; i < 10; ++i) {
        float offset = i * 70.0f;
        b[i * 4 + 0] = 10.0f + offset;
        b[i * 4 + 1] = 10.0f + offset;
        b[i * 4 + 2] = 60.0f + offset;
        b[i * 4 + 3] = 60.0f + offset;
    }
    Tensor gt_boxes = to_dev_dtype(gt_boxes_cpu);

    Tensor gt_labels_cpu({1, 10}, DType::Int64, Device::cpu());
    auto* l = gt_labels_cpu.data<int64_t>();
    for (int i = 0; i < 10; ++i) l[i] = 5;
    Tensor gt_labels = gt_labels_cpu.to(device());

    Tensor gt_masks = make_masks_zeros(1, 10, 800, 800);

    auto [rpn_cls_loss, rpn_box_loss, roi_cls_loss, roi_box_loss, mask_loss] =
        model->forward_train(images, gt_boxes, gt_labels, gt_masks);

    float roi_cls = loss_value(roi_cls_loss);
    EXPECT_GT(roi_cls, 0.0f);
    EXPECT_FALSE(std::isnan(roi_cls));
}

// ============================================================================
// Mask Loss Tests
// ============================================================================

TEST_P(MaskRCNNLossMultiDType, MaskLossBasic) {
    skip_if_half_rcnn();
    skip_if_float64_gpu();
    auto model = mask_rcnn_resnet50_fpn(80, false);
    convert_model(model);
    model->train();

    Variable images(make_images(1, 800, 800, 0.5f), true);

    Tensor gt_boxes_cpu({1, 2, 4}, DType::Float32, Device::cpu());
    auto* b = gt_boxes_cpu.data<float>();
    b[0] = 100.0f; b[1] = 100.0f; b[2] = 300.0f; b[3] = 300.0f;
    b[4] = 400.0f; b[5] = 400.0f; b[6] = 600.0f; b[7] = 600.0f;
    Tensor gt_boxes = to_dev_dtype(gt_boxes_cpu);

    Tensor gt_labels_cpu({1, 2}, DType::Int64, Device::cpu());
    auto* l = gt_labels_cpu.data<int64_t>();
    l[0] = 1; l[1] = 2;
    Tensor gt_labels = gt_labels_cpu.to(device());

    Tensor gt_masks_cpu({1, 2, 800, 800}, DType::Float32, Device::cpu());
    auto* m = gt_masks_cpu.data<float>();
    std::fill(m, m + gt_masks_cpu.numel(), 0.0f);
    for (int64_t h = 100; h < 300; ++h)
        for (int64_t w = 100; w < 300; ++w)
            m[0 * 800 * 800 + h * 800 + w] = 1.0f;
    for (int64_t h = 400; h < 600; ++h)
        for (int64_t w = 400; w < 600; ++w)
            m[1 * 800 * 800 + h * 800 + w] = 1.0f;
    Tensor gt_masks = to_dev_dtype(gt_masks_cpu);

    auto [rpn_cls_loss, rpn_box_loss, roi_cls_loss, roi_box_loss, mask_loss] =
        model->forward_train(images, gt_boxes, gt_labels, gt_masks);

    float mask = loss_value(mask_loss);
    EXPECT_GT(mask, 0.0f);
    EXPECT_LT(mask, 10.0f);
    EXPECT_FALSE(std::isnan(mask));
    EXPECT_FALSE(std::isinf(mask));
}

TEST_P(MaskRCNNLossMultiDType, MaskLossSmallObjects) {
    skip_if_half_rcnn();
    skip_if_float64_gpu();
    auto model = mask_rcnn_resnet50_fpn(80, false);
    convert_model(model);
    model->train();

    Variable images(make_images(1, 800, 800, 0.5f), true);

    Tensor gt_boxes_cpu({1, 3, 4}, DType::Float32, Device::cpu());
    auto* b = gt_boxes_cpu.data<float>();
    b[0] = 50.0f;  b[1] = 50.0f;  b[2] = 80.0f;  b[3] = 80.0f;
    b[4] = 200.0f; b[5] = 200.0f; b[6] = 230.0f; b[7] = 230.0f;
    b[8] = 400.0f; b[9] = 400.0f; b[10] = 430.0f; b[11] = 430.0f;
    Tensor gt_boxes = to_dev_dtype(gt_boxes_cpu);

    Tensor gt_labels_cpu({1, 3}, DType::Int64, Device::cpu());
    auto* l = gt_labels_cpu.data<int64_t>();
    l[0] = 1; l[1] = 2; l[2] = 3;
    Tensor gt_labels = gt_labels_cpu.to(device());

    Tensor gt_masks_cpu({1, 3, 800, 800}, DType::Float32, Device::cpu());
    auto* m = gt_masks_cpu.data<float>();
    std::fill(m, m + gt_masks_cpu.numel(), 0.0f);
    for (int i = 0; i < 3; ++i) {
        int64_t x1 = static_cast<int64_t>(b[i * 4 + 0]);
        int64_t y1 = static_cast<int64_t>(b[i * 4 + 1]);
        int64_t x2 = static_cast<int64_t>(b[i * 4 + 2]);
        int64_t y2 = static_cast<int64_t>(b[i * 4 + 3]);
        for (int64_t h = y1; h < y2; ++h)
            for (int64_t w = x1; w < x2; ++w)
                m[i * 800 * 800 + h * 800 + w] = 1.0f;
    }
    Tensor gt_masks = to_dev_dtype(gt_masks_cpu);

    auto [rpn_cls_loss, rpn_box_loss, roi_cls_loss, roi_box_loss, mask_loss] =
        model->forward_train(images, gt_boxes, gt_labels, gt_masks);

    float mask = loss_value(mask_loss);
    EXPECT_GE(mask, 0.0f);
    EXPECT_FALSE(std::isnan(mask));
}

// ============================================================================
// End-to-End Training Tests
// ============================================================================

TEST_P(MaskRCNNLossMultiDType, EndToEndTrainingLoop) {
    skip_if_half_rcnn();
    skip_if_float64_gpu();
    auto model = mask_rcnn_resnet50_fpn(80, false);
    convert_model(model);
    model->train();

    for (int iter = 0; iter < 3; ++iter) {
        Variable images(make_images(1, 800, 800, 0.5f + iter * 0.1f), true);

        Tensor gt_boxes_cpu({1, 2, 4}, DType::Float32, Device::cpu());
        auto* b = gt_boxes_cpu.data<float>();
        b[0] = 100.0f; b[1] = 100.0f; b[2] = 200.0f; b[3] = 200.0f;
        b[4] = 300.0f; b[5] = 300.0f; b[6] = 400.0f; b[7] = 400.0f;
        Tensor gt_boxes = to_dev_dtype(gt_boxes_cpu);

        Tensor gt_labels_cpu({1, 2}, DType::Int64, Device::cpu());
        auto* l = gt_labels_cpu.data<int64_t>();
        l[0] = 1; l[1] = 2;
        Tensor gt_labels = gt_labels_cpu.to(device());

        Tensor gt_masks = make_masks_zeros(1, 2, 800, 800);

        auto [rpn_cls_loss, rpn_box_loss, roi_cls_loss, roi_box_loss, mask_loss] =
            model->forward_train(images, gt_boxes, gt_labels, gt_masks);

        float rpn_cls = loss_value(rpn_cls_loss);
        float rpn_box = loss_value(rpn_box_loss);
        float roi_cls = loss_value(roi_cls_loss);
        float roi_box = loss_value(roi_box_loss);
        float mask = loss_value(mask_loss);
        EXPECT_FALSE(std::isnan(rpn_cls));
        EXPECT_FALSE(std::isnan(rpn_box));
        EXPECT_FALSE(std::isnan(roi_cls));
        EXPECT_FALSE(std::isnan(roi_box));
        EXPECT_FALSE(std::isnan(mask));

        // Sum the loss VALUES (no backward here, so no graph concern) for a
        // total-range check, matching the plain file's EndToEnd loop.
        float total = rpn_cls + rpn_box + roi_cls + roi_box + mask;
        EXPECT_GT(total, 0.0f);
        EXPECT_FALSE(std::isnan(total));
    }
}

TEST_P(MaskRCNNLossMultiDType, GradientFlow) {
    skip_if_half_rcnn();
    skip_if_float64_gpu();
    auto model = mask_rcnn_resnet50_fpn(80, false);
    convert_model(model);
    model->train();

    // 512x512, not the 800x800 used by the forward-shape tests: this is a
    // TRAINING test (forward_train + backward + grad-flow through every head),
    // and 800x800 training exceeds the 8 GB cuda backend. The plain file uses
    // 512x512 here for the same reason.
    Variable images(make_images(1, 512, 512, 0.5f), true);

    Tensor gt_boxes_cpu({1, 2, 4}, DType::Float32, Device::cpu());
    auto* b = gt_boxes_cpu.data<float>();
    b[0] = 64.0f;  b[1] = 64.0f;  b[2] = 192.0f; b[3] = 192.0f;
    b[4] = 256.0f; b[5] = 256.0f; b[6] = 384.0f; b[7] = 384.0f;
    Tensor gt_boxes = to_dev_dtype(gt_boxes_cpu);

    Tensor gt_labels_cpu({1, 2}, DType::Int64, Device::cpu());
    auto* l = gt_labels_cpu.data<int64_t>();
    l[0] = 1; l[1] = 2;
    Tensor gt_labels = gt_labels_cpu.to(device());

    Tensor gt_masks_cpu({1, 2, 512, 512}, DType::Float32, Device::cpu());
    {
        auto* m = gt_masks_cpu.data<float>();
        std::fill(m, m + gt_masks_cpu.numel(), 0.0f);
        for (int64_t h = 64; h < 192; ++h)
            for (int64_t w = 64; w < 192; ++w)
                m[0 * 512 * 512 + h * 512 + w] = 1.0f;
        for (int64_t h = 256; h < 384; ++h)
            for (int64_t w = 256; w < 384; ++w)
                m[1 * 512 * 512 + h * 512 + w] = 1.0f;
    }
    Tensor gt_masks = to_dev_dtype(gt_masks_cpu);

    auto [rpn_cls_loss, rpn_box_loss, roi_cls_loss, roi_box_loss, mask_loss] =
        model->forward_train(images, gt_boxes, gt_labels, gt_masks);

    // SUM THE VARIABLES (autograd-aware) so every head's grad_fn chain merges
    // back to every parameter — the regression guard the plain file enforces.
    Variable total_loss =
        rpn_cls_loss + rpn_box_loss + roi_cls_loss + roi_box_loss + mask_loss;

    EXPECT_EQ(total_loss.tensor().ndim(), 0);  // Scalar
    float total = loss_value(total_loss);
    EXPECT_GT(total, 0.0f);

    total_loss.backward();

    auto named = model->named_parameters();
    ASSERT_GT(named.size(), 0u);

    auto group_has_nonzero_grad = [&](const std::string& prefix) -> bool {
        for (auto& [name, param] : named) {
            if (name.rfind(prefix, 0) != 0) continue;
            const auto& grad_opt = param->grad();
            if (!grad_opt.has_value()) continue;
            if (grad_opt.value().numel() == 0) continue;
            auto g_max = ::tenzor::max(
                             ::tenzor::abs(grad_opt.value().to(Device::cpu())
                                               .to(::tenzor::DType::Float64)))
                             .item<double>();
            if (g_max > 0.0) return true;
        }
        return false;
    };

    EXPECT_TRUE(group_has_nonzero_grad("backbone."))
        << "no backbone parameter received a gradient — grad_fn severed "
           "before the backbone";
    EXPECT_TRUE(group_has_nonzero_grad("rpn."))
        << "no RPN-head parameter received a gradient — RPN loss severed its "
           "grad_fn chain";
    EXPECT_TRUE(group_has_nonzero_grad("roi_head."))
        << "no ROI-head parameter received a gradient — ROI loss severed its "
           "grad_fn chain";
    EXPECT_TRUE(group_has_nonzero_grad("mask_head."))
        << "no mask-head parameter received a gradient — mask loss severed its "
           "grad_fn chain";
}

TEST_P(MaskRCNNLossMultiDType, LossReasonableValues) {
    skip_if_half_rcnn();
    skip_if_float64_gpu();
    auto model = mask_rcnn_resnet50_fpn(80, false);
    convert_model(model);
    model->train();

    Variable images(make_images(1, 800, 800, 0.5f), true);

    Tensor gt_boxes_cpu({1, 3, 4}, DType::Float32, Device::cpu());
    auto* b = gt_boxes_cpu.data<float>();
    b[0] = 100.0f; b[1] = 100.0f; b[2] = 200.0f; b[3] = 200.0f;
    b[4] = 300.0f; b[5] = 300.0f; b[6] = 400.0f; b[7] = 400.0f;
    b[8] = 500.0f; b[9] = 500.0f; b[10] = 600.0f; b[11] = 600.0f;
    Tensor gt_boxes = to_dev_dtype(gt_boxes_cpu);

    Tensor gt_labels_cpu({1, 3}, DType::Int64, Device::cpu());
    auto* l = gt_labels_cpu.data<int64_t>();
    l[0] = 1; l[1] = 2; l[2] = 3;
    Tensor gt_labels = gt_labels_cpu.to(device());

    Tensor gt_masks = make_masks_zeros(1, 3, 800, 800);

    auto [rpn_cls_loss, rpn_box_loss, roi_cls_loss, roi_box_loss, mask_loss] =
        model->forward_train(images, gt_boxes, gt_labels, gt_masks);

    float rpn_cls = loss_value(rpn_cls_loss);
    float rpn_box = loss_value(rpn_box_loss);
    float roi_cls = loss_value(roi_cls_loss);
    float roi_box = loss_value(roi_box_loss);
    float mask = loss_value(mask_loss);

    EXPECT_GE(rpn_cls, 0.0f); EXPECT_LT(rpn_cls, 50.0f);
    EXPECT_GE(roi_cls, 0.0f); EXPECT_LT(roi_cls, 50.0f);
    EXPECT_GE(rpn_box, 0.0f); EXPECT_LT(rpn_box, 20.0f);
    EXPECT_GE(roi_box, 0.0f); EXPECT_LT(roi_box, 20.0f);
    EXPECT_GE(mask, 0.0f); EXPECT_LT(mask, 10.0f);
}

// ============================================================================
// COCO-style Annotations Test
// ============================================================================

TEST_P(MaskRCNNLossMultiDType, COCOStyleAnnotations) {
    skip_if_half_rcnn();
    skip_if_float64_gpu();
    auto model = mask_rcnn_resnet50_fpn(80, false);
    convert_model(model);
    model->train();

    Variable images(make_images(1, 800, 1200, 0.5f), true);

    Tensor gt_boxes_cpu({1, 5, 4}, DType::Float32, Device::cpu());
    auto* b = gt_boxes_cpu.data<float>();
    b[0] = 100.0f; b[1] = 150.0f; b[2] = 300.0f; b[3] = 600.0f;
    b[4] = 400.0f; b[5] = 200.0f; b[6] = 700.0f; b[7] = 500.0f;
    b[8] = 50.0f;  b[9] = 50.0f;  b[10] = 150.0f; b[11] = 150.0f;
    b[12] = 800.0f; b[13] = 300.0f; b[14] = 1000.0f; b[15] = 600.0f;
    b[16] = 200.0f; b[17] = 100.0f; b[18] = 250.0f; b[19] = 200.0f;
    Tensor gt_boxes = to_dev_dtype(gt_boxes_cpu);

    Tensor gt_labels_cpu({1, 5}, DType::Int64, Device::cpu());
    auto* l = gt_labels_cpu.data<int64_t>();
    l[0] = 1;  l[1] = 3;  l[2] = 18; l[3] = 62; l[4] = 44;
    Tensor gt_labels = gt_labels_cpu.to(device());

    Tensor gt_masks = make_masks_zeros(1, 5, 800, 1200);

    auto [rpn_cls_loss, rpn_box_loss, roi_cls_loss, roi_box_loss, mask_loss] =
        model->forward_train(images, gt_boxes, gt_labels, gt_masks);

    EXPECT_FALSE(std::isnan(loss_value(rpn_cls_loss)));
    EXPECT_FALSE(std::isnan(loss_value(rpn_box_loss)));
    EXPECT_FALSE(std::isnan(loss_value(roi_cls_loss)));
    EXPECT_FALSE(std::isnan(loss_value(roi_box_loss)));
    EXPECT_FALSE(std::isnan(loss_value(mask_loss)));
    EXPECT_GT(loss_value(rpn_cls_loss), 0.0f);
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(MaskRCNNLossMultiDType);