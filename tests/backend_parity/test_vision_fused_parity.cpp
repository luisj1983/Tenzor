/**
 * @file test_vision_fused_parity.cpp
 * @brief Backend parity for vision helpers and fused ops (Phase 3.1).
 *
 * Covers the free-function ops in include/tenzor/ops/vision.hpp
 * (unfold, fold, interpolate, grid_sample, affine_grid) and the fused ops
 * in include/tenzor/ops/fused_ops.hpp (fused_linear_relu, fused_conv2d_relu,
 * fused_conv2d_sigmoid, fused_conv2d_tanh).
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/ops/vision.hpp>
#include <tenzor/ops/fused_ops.hpp>
#include <tenzor/ops/detection.hpp>
#include <tenzor/nn/detection/roi_ops.hpp>
#include "parity_test_utils.hpp"

using namespace tenzor;
using namespace tenzor::testing;

// ============================================================================
// Vision helpers
// ============================================================================

TEST(VisionFusedParity, Unfold_4D) {
    auto input = randn({1, 3, 16, 16}, DType::Float32, Device::cpu());
    test_operation_parity(
        [](const std::vector<Tensor>& ins) {
            return ops::unfold(ins[0], /*kernel=*/3, /*stride=*/1,
                               /*padding=*/1, /*dilation=*/1);
        },
        {input}, 1e-5f, 1e-7f, "unfold");
}

// Fixed: Vulkan's dispatchCol2Im was reading AttrKey::Channels/Height/Width
// that the public ops::fold() wrapper never sets (wrapper stores H/W as a
// comma-separated OutputSize string and derives C from input shape).
// Previously Vulkan returned an empty 0-element tensor. Fix in
// src/backends/vulkan/vulkan_ops_math.cpp::dispatchCol2Im.
TEST(VisionFusedParity, Fold_4D) {
    // Unfold a reference (1, 3, 16, 16), then fold back — shape math:
    // unfold(stride=1, padding=1, k=3) -> (1, 27, 256); fold(out_hw={16,16})
    auto img = randn({1, 3, 16, 16}, DType::Float32, Device::cpu());
    auto unfolded_cpu = ops::unfold(img, 3, 1, 1, 1);

    test_operation_parity(
        [](const std::vector<Tensor>& ins) {
            return ops::fold(ins[0], /*output_size=*/{16, 16},
                             /*kernel=*/3, /*stride=*/1,
                             /*padding=*/1, /*dilation=*/1);
        },
        {unfolded_cpu}, 1e-4f, 1e-6f, "fold");
}

TEST(VisionFusedParity, Interpolate_Nearest) {
    auto input = randn({1, 3, 8, 8}, DType::Float32, Device::cpu());
    test_operation_parity(
        [](const std::vector<Tensor>& ins) {
            return ops::interpolate(ins[0], {16, 16}, "nearest", false);
        },
        {input}, 1e-5f, 1e-7f, "interpolate nearest");
}

TEST(VisionFusedParity, Interpolate_Bilinear) {
    auto input = randn({1, 3, 8, 8}, DType::Float32, Device::cpu());
    test_operation_parity(
        [](const std::vector<Tensor>& ins) {
            return ops::interpolate(ins[0], {16, 16}, "bilinear", false);
        },
        {input}, 1e-4f, 1e-6f, "interpolate bilinear");
}

TEST(VisionFusedParity, GridSample_Bilinear) {
    // Input (N=1, C=3, H=8, W=8), grid (N=1, H_out=8, W_out=8, 2)
    auto input = randn({1, 3, 8, 8}, DType::Float32, Device::cpu());
    // Identity-ish grid: sample a 8x8 output linearly from [-1, 1]
    auto theta = zeros({1, 2, 3}, DType::Float32, Device::cpu());
    auto* t = theta.data<float>();
    t[0] = 1.0f; t[4] = 1.0f;  // identity 2x3
    auto grid = ops::affine_grid(theta, {1, 3, 8, 8}, false);

    test_operation_parity(
        [&grid](const std::vector<Tensor>& ins) {
            return ops::grid_sample(ins[0], grid, "bilinear", "zeros", false);
        },
        {input}, 1e-4f, 1e-6f, "grid_sample");
}

TEST(VisionFusedParity, AffineGrid) {
    auto theta = randn({2, 2, 3}, DType::Float32, Device::cpu());
    test_operation_parity(
        [](const std::vector<Tensor>& ins) {
            return ops::affine_grid(ins[0], {2, 3, 8, 8}, false);
        },
        {theta}, 1e-4f, 1e-6f, "affine_grid");
}

// ============================================================================
// Fused ops
// ============================================================================

TEST(VisionFusedParity, FusedLinearReLU) {
    auto input = randn({8, 32}, DType::Float32, Device::cpu());
    auto weight = randn({16, 32}, DType::Float32, Device::cpu());
    auto bias = randn({16}, DType::Float32, Device::cpu());

    test_operation_parity(
        [](const std::vector<Tensor>& ins) {
            return ops::fused_linear_relu(ins[0], ins[1], &ins[2]);
        },
        {input, weight, bias}, 1e-4f, 1e-6f, "fused_linear_relu");
}

TEST(VisionFusedParity, FusedConv2dReLU) {
    auto input = randn({1, 3, 8, 8}, DType::Float32, Device::cpu());
    auto weight = randn({8, 3, 3, 3}, DType::Float32, Device::cpu());
    auto bias = randn({8}, DType::Float32, Device::cpu());

    test_operation_parity(
        [](const std::vector<Tensor>& ins) {
            return ops::fused_conv2d_relu(ins[0], ins[1], &ins[2],
                                          /*stride=*/1, /*padding=*/1);
        },
        {input, weight, bias}, 1e-3f, 1e-5f, "fused_conv2d_relu");
}

// Fixed: the OneAPI `FusedConv2dSigmoid` fused lambda was racing on the
// intermediate conv_out USM memory because conv2d_forward drives oneDNN on
// an interop stream, and `dnnl_stream.wait()` only drains the dnnl stream —
// the outer sycl::queue can still have pending task-graph state from prior
// dispatches. Added `queue.wait()` on both sides of `sigmoid_kernel` in
// `src/backends/oneapi/oneapi_kernel_registry.cpp`. Verified passing
// standalone and via `--gtest_filter=*FusedConv2dSigmoid*`. (Note: the
// non-fused `FusedConv2dReLU` test still hangs inside this binary — a
// pre-existing issue unrelated to the sigmoid fix; tracked separately.)
TEST(VisionFusedParity, FusedConv2dSigmoid) {
    auto input = randn({1, 3, 8, 8}, DType::Float32, Device::cpu());
    auto weight = randn({8, 3, 3, 3}, DType::Float32, Device::cpu());
    auto bias = randn({8}, DType::Float32, Device::cpu());

    test_operation_parity(
        [](const std::vector<Tensor>& ins) {
            return ops::fused_conv2d_sigmoid(ins[0], ins[1], &ins[2], 1, 1);
        },
        {input, weight, bias}, 1e-3f, 1e-5f, "fused_conv2d_sigmoid");
}

TEST(VisionFusedParity, FusedConv2dTanh) {
    auto input = randn({1, 3, 8, 8}, DType::Float32, Device::cpu());
    auto weight = randn({8, 3, 3, 3}, DType::Float32, Device::cpu());
    auto bias = randn({8}, DType::Float32, Device::cpu());

    test_operation_parity(
        [](const std::vector<Tensor>& ins) {
            return ops::fused_conv2d_tanh(ins[0], ins[1], &ins[2], 1, 1);
        },
        {input, weight, bias}, 1e-3f, 1e-5f, "fused_conv2d_tanh");
}

// ============================================================================
// Detection ops — ROIAlign, NMS, BoxIoU
// ============================================================================

TEST(VisionFusedParity, BoxIoU) {
    // Boxes in (x1, y1, x2, y2) format. Use deterministic values so no test
    // flake from RNG — parity must match exactly since box_iou is deterministic.
    auto boxes1 = zeros({3, 4}, DType::Float32, Device::cpu());
    auto* b1 = boxes1.data<float>();
    b1[0]=0; b1[1]=0; b1[2]=10; b1[3]=10;
    b1[4]=5; b1[5]=5; b1[6]=15; b1[7]=15;
    b1[8]=20; b1[9]=20; b1[10]=30; b1[11]=30;
    auto boxes2 = zeros({2, 4}, DType::Float32, Device::cpu());
    auto* b2 = boxes2.data<float>();
    b2[0]=0; b2[1]=0; b2[2]=8; b2[3]=8;
    b2[4]=12; b2[5]=12; b2[6]=22; b2[7]=22;

    test_operation_parity(
        [](const std::vector<Tensor>& ins) {
            return ops::box_iou(ins[0], ins[1], ops::IoUType::IoU);
        },
        {boxes1, boxes2}, 1e-5f, 1e-7f, "box_iou");
}

TEST(VisionFusedParity, NMS) {
    // 5 boxes, 3 should survive NMS with threshold 0.5.
    auto boxes = zeros({5, 4}, DType::Float32, Device::cpu());
    auto* b = boxes.data<float>();
    // Overlapping cluster centered at (10,10)
    b[0]=0;  b[1]=0;  b[2]=10; b[3]=10;
    b[4]=1;  b[5]=1;  b[6]=11; b[7]=11;
    b[8]=2;  b[9]=2;  b[10]=12; b[11]=12;
    // Isolated box far away
    b[12]=50; b[13]=50; b[14]=60; b[15]=60;
    // Another isolated box
    b[16]=100; b[17]=100; b[18]=110; b[19]=110;

    auto scores = zeros({5}, DType::Float32, Device::cpu());
    auto* s = scores.data<float>();
    s[0]=0.9f; s[1]=0.8f; s[2]=0.7f; s[3]=0.95f; s[4]=0.6f;

    // NMS returns Int64 indices — strict equality, not tolerance.
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    Tensor ref;
    try {
        ref = ops::nms(boxes, scores, 0.5);
    } catch (const std::exception& e) {
        GTEST_SKIP() << "nms CPU reference failed: " << e.what();
    }

    for (size_t i = 1; i < backends.size(); ++i) {
        try {
            auto out = ops::nms(boxes.to(backends[i]), scores.to(backends[i]),
                                 0.5);
            backends[i].synchronize();
            SCOPED_TRACE(std::string("nms on ") + backend_name(backends[i]));
            // Compare index sets — they must match exactly.
            auto out_cpu = out.to(Device::cpu());
            ASSERT_EQ(ref.numel(), out_cpu.numel());
            const auto* r = ref.data<int64_t>();
            const auto* o = out_cpu.data<int64_t>();
            for (int64_t k = 0; k < ref.numel(); ++k) {
                EXPECT_EQ(r[k], o[k]) << "nms index " << k << " differs";
            }
        } catch (const std::exception& e) {
            ADD_FAILURE() << "nms failed on " << backend_name(backends[i])
                      << ": " << e.what() << std::endl;
        }
    }
}

TEST(VisionFusedParity, ROIAlign) {
    // 1 image, 8 channels, 16x16 feature map; 3 ROIs of shape (5,) each.
    // ROI format: (batch_index, x1, y1, x2, y2)
    auto features = randn({1, 8, 16, 16}, DType::Float32, Device::cpu());
    auto rois = zeros({3, 5}, DType::Float32, Device::cpu());
    auto* r = rois.data<float>();
    r[0]=0; r[1]=1.0f; r[2]=1.0f; r[3]=5.0f; r[4]=5.0f;
    r[5]=0; r[6]=3.0f; r[7]=3.0f; r[8]=9.0f; r[9]=9.0f;
    r[10]=0; r[11]=0; r[12]=0; r[13]=14.0f; r[14]=14.0f;

    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    Tensor ref;
    try {
        nn::detection::ROIAlign roi(/*out_h=*/4, /*out_w=*/4,
                                    /*spatial_scale=*/1.0,
                                    /*sampling_ratio=*/2,
                                    /*aligned=*/true);
        ref = roi.forward(Variable(features, false), rois).tensor();
    } catch (const std::exception& e) {
        GTEST_SKIP() << "ROIAlign CPU reference failed: " << e.what();
    }

    for (size_t i = 1; i < backends.size(); ++i) {
        try {
            nn::detection::ROIAlign roi_dev(4, 4, 1.0, 2, true);
            roi_dev.to(backends[i]);
            auto features_dev = features.to(backends[i]);
            auto rois_dev = rois.to(backends[i]);
            auto out = roi_dev.forward(Variable(features_dev, false), rois_dev).tensor();
            backends[i].synchronize();
            SCOPED_TRACE(std::string("ROIAlign on ") + backend_name(backends[i]));
            EXPECT_TENSORS_CLOSE(ref, out.to(Device::cpu()), 1e-4f, 1e-5f);
        } catch (const std::exception& e) {
            ADD_FAILURE() << "ROIAlign failed on " << backend_name(backends[i])
                      << ": " << e.what() << std::endl;
        }
    }
}

// ============================================================================
// Additional fused ops — FusedSoftmaxCrossEntropy, FusedAddReLU
// ============================================================================

TEST(VisionFusedParity, FusedSoftmaxCrossEntropy) {
    auto logits = randn({8, 10}, DType::Float32, Device::cpu());
    auto targets = zeros({8}, DType::Int64, Device::cpu());
    auto* t = targets.data<int64_t>();
    for (int64_t i = 0; i < 8; ++i) t[i] = i % 10;

    test_operation_parity(
        [](const std::vector<Tensor>& ins) {
            return ops::fused_softmax_cross_entropy(ins[0], ins[1], "mean");
        },
        {logits, targets}, 1e-4f, 1e-6f, "fused_softmax_cross_entropy");
}

TEST(VisionFusedParity, FusedAddReLU) {
    auto a = randn({8, 16}, DType::Float32, Device::cpu());
    auto b = randn({8, 16}, DType::Float32, Device::cpu());
    test_operation_parity(
        [](const std::vector<Tensor>& ins) {
            return ops::fused_add_relu(ins[0], ins[1]);
        },
        {a, b}, 1e-5f, 1e-7f, "fused_add_relu");
}

// ============================================================================
// Fused ops — additional coverage (Phase 1.5)
// ============================================================================

TEST(VisionFusedParity, FusedConv2dSwish) {
    auto input = randn({1, 3, 8, 8}, DType::Float32, Device::cpu());
    auto weight = randn({8, 3, 3, 3}, DType::Float32, Device::cpu());
    auto bias = randn({8}, DType::Float32, Device::cpu());

    test_operation_parity(
        [](const std::vector<Tensor>& ins) {
            return ops::fused_conv2d_swish(ins[0], ins[1], &ins[2], 1, 1);
        },
        {input, weight, bias}, 1e-3f, 1e-5f, "fused_conv2d_swish");
}

TEST(VisionFusedParity, FusedBatchNormReLU) {
    int64_t C = 8;
    auto input = randn({2, C, 4, 4}, DType::Float32, Device::cpu());
    auto mean = randn({C}, DType::Float32, Device::cpu());
    auto var = tenzor::add(tenzor::abs(randn({C}, DType::Float32, Device::cpu())), 0.1f);
    auto weight = randn({C}, DType::Float32, Device::cpu());
    auto bn_bias = randn({C}, DType::Float32, Device::cpu());

    test_operation_parity(
        [](const std::vector<Tensor>& ins) {
            return ops::fused_batchnorm_relu(ins[0], ins[1], ins[2], ins[3], ins[4]);
        },
        {input, mean, var, weight, bn_bias}, 1e-4f, 1e-6f, "fused_batchnorm_relu");
}

TEST(VisionFusedParity, FusedGELU) {
    auto input = randn({8, 16}, DType::Float32, Device::cpu());
    test_operation_parity(
        [](const std::vector<Tensor>& ins) {
            return ops::fused_gelu(ins[0]);
        },
        {input}, 1e-5f, 1e-6f, "fused_gelu");
}

TEST(VisionFusedParity, FusedLayerNorm) {
    auto input = randn({4, 8, 32}, DType::Float32, Device::cpu());
    auto weight = randn({32}, DType::Float32, Device::cpu());
    auto ln_bias = randn({32}, DType::Float32, Device::cpu());

    test_operation_parity(
        [](const std::vector<Tensor>& ins) {
            return ops::fused_layer_norm(ins[0], {32}, ins[1], ins[2]);
        },
        {input, weight, ln_bias}, 1e-4f, 1e-6f, "fused_layer_norm");
}

// NOTE: FusedAttention has an OpId entry but no public free-function wrapper
// in include/tenzor/ops/fused_ops.hpp. Parity coverage would need to go
// through the dispatch layer directly — out of scope for this task.
// Similarly, fused_sgd_step / fused_adam_step are CUDA-only backend APIs
// (include/tenzor/backend/fused_ops.hpp) without general dispatch wrappers.

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    try {
        if (!::testing::GTEST_FLAG(list_tests)) {
            tenzor::initialize();
        }
    } catch (const std::exception& e) {
        std::cerr << "Failed to initialize Tenzor: " << e.what() << std::endl;
        return 1;
    }
    int result = RUN_ALL_TESTS();
    try { tenzor::finalize(); } catch (...) {}
    return result;
}
