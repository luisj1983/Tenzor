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
#include <tenzor/backend/fast_dispatch.hpp>
#include <tenzor/backend/op_attributes.hpp>
#include <tenzor/ops/op_id.hpp>
#include "../backend_test_fixture.hpp"
#include "parity_test_utils.hpp"

using namespace tenzor;
using namespace tenzor::testing;


class VisionFusedParity : public BackendTest {};
// ============================================================================
// Vision helpers
// ============================================================================

TEST_P(VisionFusedParity, Unfold_4D) {
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
TEST_P(VisionFusedParity, Fold_4D) {
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

TEST_P(VisionFusedParity, Interpolate_Nearest) {
    auto input = randn({1, 3, 8, 8}, DType::Float32, Device::cpu());
    test_operation_parity(
        [](const std::vector<Tensor>& ins) {
            return ops::interpolate(ins[0], {16, 16}, "nearest", false);
        },
        {input}, 1e-5f, 1e-7f, "interpolate nearest");
}

TEST_P(VisionFusedParity, Interpolate_Bilinear) {
    auto input = randn({1, 3, 8, 8}, DType::Float32, Device::cpu());
    test_operation_parity(
        [](const std::vector<Tensor>& ins) {
            return ops::interpolate(ins[0], {16, 16}, "bilinear", false);
        },
        {input}, 1e-4f, 1e-6f, "interpolate bilinear");
}

// M29: trilinear (5D) forward — Vulkan previously had no forward
// implementation at all (dispatchInterpolate unconditionally required 4D),
// even though a full native trilinear backward shader already existed as
// unreachable dead code. A genuinely non-trivial resize (upsample one axis,
// downsample another) exercises the new gather shader's coordinate
// transform on both branches of the align_corners=false half-pixel formula.
//
// Explicit per-device dispatch (not test_operation_parity): that helper
// silently EXCLUDES any backend whose call throws from the comparison
// (catches, cerr-logs, moves on) rather than failing — with 4 other
// backends already implementing trilinear, a Vulkan exception would still
// leave >=2 successful results and the test would pass without ever
// exercising Vulkan. This directly asserts THIS device's call succeeds and
// matches the CPU reference.
TEST_P(VisionFusedParity, Interpolate_Trilinear) {
    auto input = randn({1, 2, 3, 4, 5}, DType::Float32, Device::cpu());
    auto ref = ops::interpolate(input, {6, 7, 9}, "trilinear", false);

    auto out = ops::interpolate(input.to(device), {6, 7, 9}, "trilinear", false);
    device.synchronize();
    EXPECT_TENSORS_CLOSE(ref, out.to(Device::cpu()), 1e-4f, 1e-6f);
}

// interpolate_backward bicubic (4D) parity — newly added on all GPU backends (W4).
TEST_P(VisionFusedParity, InterpolateBackward_Bicubic) {
    auto grad_out = randn({1, 3, 16, 16}, DType::Float32, Device::cpu());
    test_operation_parity(
        [](const std::vector<Tensor>& ins) {
            OpAttributes a;
            a.set(AttrKey::InputShape, std::string("8,8"));
            a.set(AttrKey::Mode, std::string("bicubic"));
            a.set(AttrKey::AlignCorners, false);
            return dispatch(OpId::InterpolateBackward, ins, a)[0];
        },
        {grad_out}, 1e-3f, 1e-4f, "interpolate bicubic backward");
}

// interpolate_backward trilinear (5D) parity — newly added on all GPU backends (W4).
TEST_P(VisionFusedParity, InterpolateBackward_Trilinear) {
    auto grad_out = randn({1, 2, 6, 7, 9}, DType::Float32, Device::cpu());
    test_operation_parity(
        [](const std::vector<Tensor>& ins) {
            OpAttributes a;
            a.set(AttrKey::InputShape, std::string("3,4,5"));
            a.set(AttrKey::Mode, std::string("trilinear"));
            a.set(AttrKey::AlignCorners, false);
            return dispatch(OpId::InterpolateBackward, ins, a)[0];
        },
        {grad_out}, 1e-3f, 1e-4f, "interpolate trilinear backward");
}

TEST_P(VisionFusedParity, GridSample_Bilinear) {
    // Input (N=1, C=3, H=8, W=8), grid (N=1, H_out=8, W_out=8, 2)
    auto input = randn({1, 3, 8, 8}, DType::Float32, Device::cpu());
    // Identity-ish grid: sample a 8x8 output linearly from [-1, 1]
    auto theta = zeros({1, 2, 3}, DType::Float32, Device::cpu());
    auto* t = theta.data<float>();
    t[0] = 1.0f; t[4] = 1.0f;  // identity 2x3
    auto grid = ops::affine_grid(theta, {1, 3, 8, 8}, false);

    test_operation_parity(
        [&grid](const std::vector<Tensor>& ins) {
            // grid must live on the same device as the input — otherwise the
            // backend correctly rejects the mixed-device call and the parity
            // harness skips for lack of a second backend (this test previously
            // never actually ran on any GPU backend).
            return ops::grid_sample(ins[0], grid.to(ins[0].device()), "bilinear", "zeros", false);
        },
        {input}, 1e-4f, 1e-6f, "grid_sample");
}

// Reflection-padding grad_grid carries a +/-1 fold-sign from the reflection.
// CPU/ROCm/OneAPI previously dropped that sign while CUDA applied it (PyTorch
// behaviour), so grad_grid diverged across backends. Make BOTH input and grid
// differentiable and use grid coords that land in reflected regions so the
// fold-sign is exercised; the parity harness then compares grad_grid across all
// backends. Run for align_corners=true and false (the reflection span differs).
// L5: nearest-mode grid_sample's round-half tie-break must match across
// backends (round-half-to-even, matching CPU std::nearbyint / CUDA rint /
// PyTorch-ATen convention) — ROCm/OneAPI/Vulkan/MPS previously used
// round-half-away-from-zero (or, for Vulkan, GLSL's implementation-defined
// round()), silently picking a DIFFERENT pixel than CPU/CUDA whenever a
// grid coordinate lands exactly on a .5 boundary between two pixels.
// W_in=H_in=6, align_corners=true, grid_x=grid_y=0 gives
// ix=iy=(0+1)/2*(6-1)=2.5 exactly — round-to-even picks pixel index 2,
// round-away-from-zero picks index 3, so a divergence here is unambiguous,
// not rounding noise. Each input element is set to its own flat index, so
// the sampled VALUE directly reveals which pixel was picked.
TEST_P(VisionFusedParity, GridSample_NearestTieBreak) {
    auto input = zeros({1, 1, 6, 6}, DType::Float32, Device::cpu());
    {
        float* d = input.data<float>();
        for (int i = 0; i < 36; ++i) d[i] = static_cast<float>(i);
    }
    auto grid = zeros({1, 1, 1, 2}, DType::Float32, Device::cpu());  // grid_x=grid_y=0

    test_operation_parity(
        [&grid](const std::vector<Tensor>& ins) {
            return ops::grid_sample(ins[0], grid.to(ins[0].device()), "nearest", "zeros", true);
        },
        {input}, 1e-6f, 1e-7f, "grid_sample nearest tie-break");
}

TEST_P(VisionFusedParity, GridSample_Reflection) {
    auto input = randn({1, 2, 5, 5}, DType::Float32, Device::cpu());
    // DETERMINISTIC grid whose samples land in reflected regions but comfortably
    // mid-leg — away from the fold turning points (where the gradient sign is
    // discontinuous) and away from integer pixel boundaries (where bilinear is
    // non-smooth). A random grid would inevitably straddle those discontinuities
    // and make CPU/GPU disagree by an O(1) sign flip for FP-rounding reasons,
    // which is a property of reflection, not of the kernels.
    auto grid = zeros({1, 2, 2, 2}, DType::Float32, Device::cpu());
    {
        float* g = grid.data<float>();
        const float vals[8] = {1.40f, 1.30f, -1.40f, -1.30f,
                               1.35f, -1.25f, -1.35f, 1.20f};
        for (int i = 0; i < 8; ++i) g[i] = vals[i];
    }

    for (bool align : {true, false}) {
        test_operation_parity(
            [align](const std::vector<Tensor>& ins) {
                return ops::grid_sample(ins[0], ins[1], "bilinear", "reflection", align);
            },
            {input, grid}, 1e-4f, 1e-4f, "grid_sample_reflection");
    }
}

// Release audit: bicubic forward must match CPU on every backend. ROCm/OneAPI
// previously silently computed bilinear for mode='bicubic'.
TEST_P(VisionFusedParity, GridSample_Bicubic) {
    auto input = randn({1, 3, 8, 8}, DType::Float32, Device::cpu());
    // Use a SCALED (fractional) sampling grid, not identity: at integer sample
    // positions bilinear and bicubic coincide, which would mask a backend that
    // silently computes bilinear. Scale 0.5 + small shift -> sub-pixel samples
    // where bicubic genuinely differs from bilinear.
    auto theta = zeros({1, 2, 3}, DType::Float32, Device::cpu());
    auto* t = theta.data<float>();
    t[0] = 0.5f; t[2] = 0.1f; t[4] = 0.5f; t[5] = -0.1f;
    auto grid = ops::affine_grid(theta, {1, 3, 8, 8}, false);

    test_operation_parity(
        [&grid](const std::vector<Tensor>& ins) {
            return ops::grid_sample(ins[0], grid.to(ins[0].device()), "bicubic", "zeros", false);
        },
        {input}, 1e-4f, 1e-6f, "grid_sample_bicubic");
}

// Release audit: native Float64 grid_sample must match CPU on every backend.
// ROCm/OneAPI previously force-downcast Float64 inputs to Float32.
TEST_P(VisionFusedParity, GridSample_Bicubic_Float64) {
    // Native Float64 grid_sample is provided by CPU/CUDA/ROCm/OneAPI. The Vulkan
    // grid_sample kernel computes in Float32 by design (it casts other dtypes via
    // dispatchCast), so it cannot meet a tight f64 tolerance and is excluded from
    // this native-f64 parity check; its bicubic correctness is covered by the
    // Float32 test above. (The parity helper loops backends internally, so we
    // pass an explicit Vulkan-excluded list rather than skipping per-param.)
    std::vector<Device> backends;
    for (const auto& d : get_available_backends()) {
        if (d.type != Device::Type::Vulkan) backends.push_back(d);
    }
    if (backends.size() < 2) {
        GTEST_SKIP() << "need >=2 non-Vulkan backends for native-f64 grid_sample parity";
    }

    // Hand-built pure-Float64 grid (no affine_grid, which may downcast) so the
    // whole path is native f64 and sample positions are fractional (bicubic !=
    // bilinear). Values in (-1,1) map to sub-pixel input coordinates.
    const int64_t H = 8, W = 8;
    auto input = randn({1, 3, H, W}, DType::Float64, Device::cpu());
    auto grid = zeros({1, H, W, 2}, DType::Float64, Device::cpu());
    {
        double* g = grid.data<double>();
        for (int64_t h = 0; h < H; ++h)
            for (int64_t w = 0; w < W; ++w) {
                int64_t k = (h * W + w) * 2;
                g[k]     = -0.85 + 1.7 * (double(w) + 0.37) / double(W);  // x in (-0.85,0.85)
                g[k + 1] = -0.85 + 1.7 * (double(h) + 0.62) / double(H);  // y
            }
    }

    test_operation_parity_backends(
        [&grid](const std::vector<Tensor>& ins) {
            return ops::grid_sample(ins[0], grid.to(ins[0].device()), "bicubic", "zeros", false);
        },
        {input}, backends, 1e-5f, 1e-7f, "grid_sample_bicubic_f64");
}

TEST_P(VisionFusedParity, AffineGrid) {
    auto theta = randn({2, 2, 3}, DType::Float32, Device::cpu());
    test_operation_parity(
        [](const std::vector<Tensor>& ins) {
            return ops::affine_grid(ins[0], {2, 3, 8, 8}, false);
        },
        {theta}, 1e-4f, 1e-6f, "affine_grid");
}

// M10: OneAPI's affine_grid forward widened Float16/BFloat16 theta to
// Float32 for its compute kernel but never narrowed the output grid back,
// silently upcasting the whole affine_grid -> grid_sample pipeline's dtype.
TEST_P(VisionFusedParity, AffineGrid_Float16BFloat16PreservesDtype) {
    for (DType dt : {DType::Float16, DType::BFloat16}) {
        auto theta = randn({1, 2, 3}, DType::Float32, Device::cpu()).to(dt).to(device);
        auto grid = ops::affine_grid(theta, {1, 3, 4, 4}, false);
        EXPECT_EQ(grid.dtype(), dt)
            << "affine_grid dtype not preserved for " << dtype_name(dt)
            << " on device " << device.to_string();
    }
}

// M9: ROCm's AffineGridBackward kernel was float-only (no native double
// compute path) even when the caller passed a Float64 grad_grid, so a
// Float64 theta.grad() silently lost precision to Float32 epsilon. Forward
// tolerance stays at Float32 epsilon since affine_grid's forward compute
// on CUDA/ROCm is a widen-narrow by design (M8 only restores the dtype, it
// doesn't add a native double forward kernel — see affine_grid_cuda). The
// backward tolerance is tight (well below Float32 epsilon ~1.2e-7) so a
// silent F32 compute in AffineGridBackward fails this test against the CPU
// double reference, which does have a native double backward kernel.
// Vulkan is excluded: its affine_grid_backward.comp uses atomicAdd(float)
// (VK_EXT_shader_atomic_float) and has no double-atomic path, so it computes
// in Float32 by design — same documented exclusion as GridSample_Bicubic_Float64
// above. It still restores the caller's dtype on return (dispatchCast to
// gr_dt), so there is no dtype-drop bug there, only expected reduced precision.
TEST_P(VisionFusedParity, AffineGrid_Float64GradientParity) {
    std::vector<Device> backends;
    for (const auto& d : get_available_backends()) {
        if (d.type != Device::Type::Vulkan) backends.push_back(d);
    }
    if (backends.size() < 2) {
        GTEST_SKIP() << "need >=2 non-Vulkan backends for native-f64 affine_grid gradient parity";
    }
    // H=5, W=7 (not powers of 2): x_norm/y_norm are non-dyadic rationals, so
    // Float32 vs Float64 accumulation genuinely differs in the low bits —
    // power-of-2 sizes would make every intermediate exactly representable
    // in both precisions and mask a Float32-only backward kernel entirely.
    auto theta = randn({2, 2, 3}, DType::Float64, Device::cpu());
    test_gradient_parity(
        [](std::vector<Variable>& ins) {
            return tenzor::affine_grid(ins[0], {2, 3, 5, 7}, false);
        },
        {theta}, {}, 1e-5f, 1e-6f, 1e-9f, 1e-12f, backends, "affine_grid_f64_grad");
}

// ============================================================================
// Fused ops
// ============================================================================

TEST_P(VisionFusedParity, FusedLinearReLU) {
    auto input = randn({8, 32}, DType::Float32, Device::cpu());
    auto weight = randn({16, 32}, DType::Float32, Device::cpu());
    auto bias = randn({16}, DType::Float32, Device::cpu());

    test_operation_parity(
        [](const std::vector<Tensor>& ins) {
            return ops::fused_linear_relu(ins[0], ins[1], &ins[2]);
        },
        {input, weight, bias}, 1e-4f, 1e-6f, "fused_linear_relu");
}

TEST_P(VisionFusedParity, FusedConv2dReLU) {
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
TEST_P(VisionFusedParity, FusedConv2dSigmoid) {
    auto input = randn({1, 3, 8, 8}, DType::Float32, Device::cpu());
    auto weight = randn({8, 3, 3, 3}, DType::Float32, Device::cpu());
    auto bias = randn({8}, DType::Float32, Device::cpu());

    test_operation_parity(
        [](const std::vector<Tensor>& ins) {
            return ops::fused_conv2d_sigmoid(ins[0], ins[1], &ins[2], 1, 1);
        },
        {input, weight, bias}, 1e-3f, 1e-5f, "fused_conv2d_sigmoid");
}

TEST_P(VisionFusedParity, FusedConv2dTanh) {
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

TEST_P(VisionFusedParity, BoxIoU) {
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

TEST_P(VisionFusedParity, NMS) {
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
    REQUIRE_MULTI_BACKEND_OR_SKIP("vision fused parity");

    Tensor ref = ops::nms(boxes, scores, 0.5);

    for (size_t i = first_gpu_index(backends); i < backends.size(); ++i) {
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

TEST_P(VisionFusedParity, ROIAlign) {
    // 1 image, 8 channels, 16x16 feature map; 3 ROIs of shape (5,) each.
    // ROI format: (batch_index, x1, y1, x2, y2)
    auto features = randn({1, 8, 16, 16}, DType::Float32, Device::cpu());
    auto rois = zeros({3, 5}, DType::Float32, Device::cpu());
    auto* r = rois.data<float>();
    r[0]=0; r[1]=1.0f; r[2]=1.0f; r[3]=5.0f; r[4]=5.0f;
    r[5]=0; r[6]=3.0f; r[7]=3.0f; r[8]=9.0f; r[9]=9.0f;
    r[10]=0; r[11]=0; r[12]=0; r[13]=14.0f; r[14]=14.0f;

    auto backends = get_available_backends();
    REQUIRE_MULTI_BACKEND_OR_SKIP("vision fused parity");

    Tensor ref;
    {
        nn::detection::ROIAlign roi(/*out_h=*/4, /*out_w=*/4,
                                    /*spatial_scale=*/1.0,
                                    /*sampling_ratio=*/2,
                                    /*aligned=*/true);
        ref = roi.forward(Variable(features, false), rois).tensor();
    }

    for (size_t i = first_gpu_index(backends); i < backends.size(); ++i) {
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

// M15: spatial_scale (and, on CUDA/ROCm, the ROI box coordinates) used to be
// narrowed to Float32 on every GPU backend regardless of feature dtype, even
// though CUDA/ROCm/OneAPI/Vulkan all have genuine native-double roi_align
// kernels (matching CPU). A non-power-of-2 spatial_scale (1.0/3.0) forces the
// narrowing to actually lose precision (~7 vs ~15-16 accurate digits) instead
// of rounding exactly, and a tight tolerance well below Float32 epsilon
// catches it against the CPU double reference.
TEST_P(VisionFusedParity, ROIAlign_Float64_NonPow2SpatialScale) {
    REQUIRE_MULTI_BACKEND_OR_SKIP("vision fused parity Float64 roi_align");
    auto features = randn({1, 4, 16, 16}, DType::Float64, Device::cpu());
    auto rois = zeros({2, 5}, DType::Float64, Device::cpu());
    auto* r = rois.data<double>();
    r[0] = 0; r[1] = 1.0; r[2] = 1.0; r[3] = 11.0; r[4] = 13.0;
    r[5] = 0; r[6] = 2.0; r[7] = 0.0; r[8] = 15.0; r[9] = 9.0;

    auto backends = get_available_backends();

    Tensor ref;
    {
        nn::detection::ROIAlign roi(4, 4, /*spatial_scale=*/1.0 / 3.0,
                                    /*sampling_ratio=*/2, /*aligned=*/true);
        ref = roi.forward(Variable(features, false), rois).tensor();
    }

    for (size_t i = first_gpu_index(backends); i < backends.size(); ++i) {
        try {
            nn::detection::ROIAlign roi_dev(4, 4, 1.0 / 3.0, 2, true);
            roi_dev.to(backends[i]);
            auto features_dev = features.to(backends[i]);
            auto rois_dev = rois.to(backends[i]);
            auto out = roi_dev.forward(Variable(features_dev, false), rois_dev).tensor();
            backends[i].synchronize();
            SCOPED_TRACE(std::string("ROIAlign_Float64_NonPow2SpatialScale on ") + backend_name(backends[i]));
            EXPECT_TENSORS_CLOSE(ref, out.to(Device::cpu()), 1e-9f, 1e-12f);
        } catch (const std::exception& e) {
            ADD_FAILURE() << "ROIAlign_Float64_NonPow2SpatialScale failed on "
                      << backend_name(backends[i]) << ": " << e.what() << std::endl;
        }
    }
}

// L4: OneAPI/Vulkan's ROIAlignForward registry defaulted SpatialScale to 1.0
// (OneAPI) or 0.0 (Vulkan, via OpAttributes::get_float()'s generic fallback)
// and Aligned to false, diverging from CPU/CUDA/ROCm's SpatialScale=1/16,
// Aligned=true. Every real call site (nn::detection::ROIAlign) sets both
// explicitly, so this was dead — but a latent trap for any future
// direct-dispatch caller that omits them, silently producing a
// wrong-by-construction result (a 0.0 spatial_scale collapses every ROI to
// a single point) instead of an error. Dispatch OpId::ROIAlignForward
// directly with an OpAttributes that OMITS SpatialScale/Aligned entirely
// (only setting the otherwise-required OutputSizeH/W) to exercise the
// DEFAULT fallback path itself, and confirm every backend's default-driven
// result now matches CPU's.
TEST_P(VisionFusedParity, ROIAlign_DefaultAttrsMatchAcrossBackends) {
    REQUIRE_MULTI_BACKEND_OR_SKIP("vision fused parity roi_align defaults");
    auto features = randn({1, 4, 16, 16}, DType::Float32, Device::cpu());
    auto rois = zeros({2, 5}, DType::Float32, Device::cpu());
    auto* r = rois.data<float>();
    r[0] = 0; r[1] = 1.0f; r[2] = 1.0f; r[3] = 11.0f; r[4] = 13.0f;
    r[5] = 0; r[6] = 2.0f; r[7] = 0.0f; r[8] = 15.0f; r[9] = 9.0f;

    OpAttributes attrs;
    attrs.set(AttrKey::OutputSizeH, int64_t{4});
    attrs.set(AttrKey::OutputSizeW, int64_t{4});
    // Deliberately NOT setting AttrKey::SpatialScale or AttrKey::Aligned —
    // the whole point is to exercise each backend's own default.

    auto backends = get_available_backends();

    Tensor ref = dispatch(OpId::ROIAlignForward,
        std::vector<Tensor>{features, rois}, attrs)[0];

    for (size_t i = first_gpu_index(backends); i < backends.size(); ++i) {
        try {
            auto features_dev = features.to(backends[i]);
            auto rois_dev = rois.to(backends[i]);
            auto out = dispatch(OpId::ROIAlignForward,
                std::vector<Tensor>{features_dev, rois_dev}, attrs)[0];
            backends[i].synchronize();
            SCOPED_TRACE(std::string("ROIAlign_DefaultAttrsMatchAcrossBackends on ") + backend_name(backends[i]));
            EXPECT_TENSORS_CLOSE(ref, out.to(Device::cpu()), 1e-4f, 1e-5f);
        } catch (const std::exception& e) {
            ADD_FAILURE() << "ROIAlign_DefaultAttrsMatchAcrossBackends failed on "
                      << backend_name(backends[i]) << ": " << e.what() << std::endl;
        }
    }
}

// ============================================================================
// Additional fused ops — FusedSoftmaxCrossEntropy, FusedAddReLU
// ============================================================================

TEST_P(VisionFusedParity, FusedSoftmaxCrossEntropy) {
    auto logits = randn({8, 10}, DType::Float32, Device::cpu());
    auto targets = zeros({8}, DType::Int64, Device::cpu());
    auto* t = targets.data<int64_t>();
    for (int64_t i = 0; i < 8; ++i) t[i] = i % 10;

    // CPU returns shape [], GPU backends return shape [1]; reshape to [1]
    // so parity comparison doesn't bail on the leading shape check.
    // atol=1e-5 (not 1e-6) because the mean reduces log-sum-exp across 10
    // classes and backends pick slightly different summation paths.
    test_operation_parity(
        [](const std::vector<Tensor>& ins) {
            return ops::fused_softmax_cross_entropy(ins[0], ins[1], "mean")
                .reshape({1});
        },
        {logits, targets}, 1e-4f, 1e-5f, "fused_softmax_cross_entropy");
}

TEST_P(VisionFusedParity, FusedAddReLU) {
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

TEST_P(VisionFusedParity, FusedConv2dSwish) {
    auto input = randn({1, 3, 8, 8}, DType::Float32, Device::cpu());
    auto weight = randn({8, 3, 3, 3}, DType::Float32, Device::cpu());
    auto bias = randn({8}, DType::Float32, Device::cpu());

    test_operation_parity(
        [](const std::vector<Tensor>& ins) {
            return ops::fused_conv2d_swish(ins[0], ins[1], &ins[2], 1, 1);
        },
        {input, weight, bias}, 1e-3f, 1e-5f, "fused_conv2d_swish");
}

TEST_P(VisionFusedParity, FusedBatchNormReLU) {
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

TEST_P(VisionFusedParity, FusedGELU) {
    auto input = randn({8, 16}, DType::Float32, Device::cpu());
    test_operation_parity(
        [](const std::vector<Tensor>& ins) {
            return ops::fused_gelu(ins[0]);
        },
        {input}, 1e-5f, 1e-6f, "fused_gelu");
}

TEST_P(VisionFusedParity, FusedLayerNorm) {
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

// Release audit: dedicated depthwise Conv1d/Conv3d kernels on every GPU backend
// (previously throw-stubs, so depthwise-separable models hard-failed on GPU).
// Contract: Conv1d input [N,C,1,L], weight [C,1,1,kL]; Conv3d input [N,C,D,H,W],
// weight [C,1,kD,kH,kW]. Compared against the CPU reference kernel.
TEST_P(VisionFusedParity, DepthwiseConv1d) {
    const int64_t C = 4, L = 12, kL = 3;
    auto input  = randn({1, C, 1, L}, DType::Float32, Device::cpu());
    auto weight = randn({C, 1, 1, kL}, DType::Float32, Device::cpu());
    auto bias   = randn({C}, DType::Float32, Device::cpu());
    test_operation_parity(
        [](const std::vector<Tensor>& ins) {
            OpAttributes a;
            a.set(AttrKey::Stride, (int64_t)2);
            a.set(AttrKey::Padding, (int64_t)1);
            a.set(AttrKey::Dilation, (int64_t)1);
            return dispatch(OpId::DepthwiseConv1d, ins, a)[0];
        },
        {input, weight, bias}, 1e-4f, 1e-6f, "depthwise_conv1d");
}

TEST_P(VisionFusedParity, DepthwiseConv3d) {
    const int64_t C = 3, D = 6, H = 6, W = 6, kD = 3, kH = 3, kW = 3;
    auto input  = randn({1, C, D, H, W}, DType::Float32, Device::cpu());
    auto weight = randn({C, 1, kD, kH, kW}, DType::Float32, Device::cpu());
    auto bias   = randn({C}, DType::Float32, Device::cpu());
    test_operation_parity(
        [](const std::vector<Tensor>& ins) {
            OpAttributes a;
            a.set(AttrKey::Stride, (int64_t)1);
            a.set(AttrKey::Padding, (int64_t)1);
            a.set(AttrKey::Dilation, (int64_t)1);
            return dispatch(OpId::DepthwiseConv3d, ins, a)[0];
        },
        {input, weight, bias}, 1e-4f, 1e-6f, "depthwise_conv3d");
}

INSTANTIATE_BACKEND_TESTS(VisionFusedParity);


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
