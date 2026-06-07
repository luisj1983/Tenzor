/**
 * @file test_roi_ops_multidtype.cpp
 * @brief Multi-dtype tests for ROI operations (ROIAlign)
 *
 * Tests ROI Align with Float32, Float64, and Float16 dtypes across
 * CPU, CUDA, OneAPI, Vulkan, and ROCm backends to ensure:
 * - Correct output shape
 * - Gradient flow through ROIAlign
 * - Different spatial scale settings
 * - Single ROI handling
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/nn/detection/roi_ops.hpp>
#include "multi_backend_dtype_fixture.hpp"
#include <cmath>

using namespace tenzor;
using namespace tenzor::nn;
using namespace tenzor::nn::detection;
using namespace tenzor::testing;

// ============================================================================
// ROI Ops Multi-Backend Multi-DType Test Fixture
// ============================================================================

class ROIOpsMultiDTypeTest : public MultiBackendDTypeTest {
protected:
    /**
     * @brief Create a ROI tensor with format [batch_idx, x1, y1, x2, y2].
     *
     * ROIs are always Float32 regardless of the test dtype since they
     * represent coordinates, not features.
     */
    Tensor createROIs(int64_t num_rois, int64_t batch_idx = 0) {
        // Create ROIs in Float32 on the test device
        auto rois = tenzor::zeros({num_rois, 5}, DType::Float32, device());

        // Fill with reasonable ROI coordinates on CPU then transfer
        auto rois_cpu = tenzor::zeros({num_rois, 5}, DType::Float32, Device::cpu());
        auto* data = rois_cpu.data<float>();

        for (int64_t i = 0; i < num_rois; ++i) {
            data[i * 5 + 0] = static_cast<float>(batch_idx);  // batch index
            data[i * 5 + 1] = 2.0f + static_cast<float>(i);   // x1
            data[i * 5 + 2] = 2.0f + static_cast<float>(i);   // y1
            data[i * 5 + 3] = 10.0f + static_cast<float>(i);  // x2
            data[i * 5 + 4] = 10.0f + static_cast<float>(i);  // y2
        }

        if (device() != Device::cpu()) {
            return rois_cpu.to(device());
        }
        return rois_cpu;
    }
};

// ============================================================================
// Output Shape Test
// ============================================================================

TEST_P(ROIOpsMultiDTypeTest, OutputShape) {
    const int64_t num_rois = 5;
    const int64_t C = 3;
    const int64_t output_h = 7;
    const int64_t output_w = 7;

    ROIAlign roi_align(output_h, output_w, 1.0 / 16.0, 2);

    auto feats_cpu = tenzor::randn({1, C, 16, 16}, DType::Float32, Device::cpu());
    auto rois_cpu_full = createROIs(num_rois, 0).to(Device::cpu());
    auto ref = roi_align.forward(Variable(feats_cpu, false), rois_cpu_full);

    convert_model(roi_align);
    Variable features = Variable(feats_cpu.to(dtype_).to(device_), false);
    auto rois = createROIs(num_rois, 0);

    auto output = roi_align.forward(features, rois);

    expectShape(output.tensor(), {num_rois, C, output_h, output_w});
    expectDType(output.tensor());
    expectTensorNear(output.tensor(), ref.tensor(), std::max(atol_, 5e-2f));
}

// ============================================================================
// Gradient Flow Test
// ============================================================================

TEST_P(ROIOpsMultiDTypeTest, GradientFlow) {
    const int64_t output_h = 7;
    const int64_t output_w = 7;

    ROIAlign roi_ref(output_h, output_w, 1.0 / 16.0, 2);
    auto feats_cpu = tenzor::randn({1, 3, 16, 16}, DType::Float32, Device::cpu());
    auto rois_cpu_full = createROIs(3, 0).to(Device::cpu());
    auto feats_ref = Variable(feats_cpu, true);
    auto out_ref = roi_ref.forward(feats_ref, rois_cpu_full);
    auto out_ref_shape = out_ref.shape();
    std::vector<int64_t> out_ref_shape_vec(out_ref_shape.begin(), out_ref_shape.end());
    out_ref.backward(tenzor::ones(out_ref_shape_vec, DType::Float32, Device::cpu()));

    ROIAlign roi_align(output_h, output_w, 1.0 / 16.0, 2);
    convert_model(roi_align);
    Variable features = Variable(feats_cpu.to(dtype_).to(device_), true);
    auto rois = createROIs(3, 0);

    auto output = roi_align.forward(features, rois);
    auto out_shape = output.shape();
    std::vector<int64_t> shape_vec(out_shape.begin(), out_shape.end());
    auto grad_output = tenzor::ones(shape_vec, dtype(), device());
    output.backward(grad_output);

    EXPECT_TRUE(features.has_grad());
    EXPECT_EQ(features.grad()->dtype(), dtype());
    expectShape(*features.grad(), {1, 3, 16, 16});

    // Verify gradients are finite
    auto grad_f32 = features.grad()->to(Device::cpu()).to(DType::Float32);
    auto* grad_data = grad_f32.data<float>();
    for (int64_t i = 0; i < grad_f32.numel(); ++i) {
        EXPECT_FALSE(std::isnan(grad_data[i]));
        EXPECT_FALSE(std::isinf(grad_data[i]));
    }
    // Cross-check forward and input gradient against CPU reference.
    expectTensorNear(output.tensor(), out_ref.tensor(), std::max(atol_, 5e-2f));

    // The input gradient is validated with a RELATIVE tolerance rather than the
    // pure-absolute expectTensorNear. With grad_output = ones, ROIAlign's
    // backward overlap-adds many bilinear contributions onto the corner feature
    // pixels, so those gradients reach large magnitudes (~136 here). The kernel
    // computes in Float32 and narrows the result to the requested dtype, so a
    // Float16 gradient is the correctly-rounded Float16 value — but Float16's
    // ulp at magnitude 136 is ~0.125, larger than a 5e-2 absolute tolerance, so
    // an absolute comparison can never pass at that scale regardless of
    // correctness. |a-b| <= atol + rtol*|b| validates the actual numerical
    // property (relative accuracy) for every dtype.
    {
        auto ga = features.grad()->to(Device::cpu()).to(DType::Float32);
        auto gb = feats_ref.grad()->to(Device::cpu()).to(DType::Float32);
        ASSERT_EQ(ga.numel(), gb.numel());
        const auto* a = ga.data<float>();
        const auto* b = gb.data<float>();
        const float atol = std::max(atol_, 5e-2f);
        // Float16 carries ~10 mantissa bits (≈1e-3 relative); BFloat16 ~7
        // (≈8e-3). 2e-2 covers both with margin while still catching real
        // gradient bugs (which manifest as O(1) relative error).
        const float rtol = (dtype_ == DType::Float16 ||
                            dtype_ == DType::BFloat16) ? 2e-2f : 1e-3f;
        for (int64_t i = 0; i < ga.numel(); ++i) {
            EXPECT_LE(std::abs(a[i] - b[i]), atol + rtol * std::abs(b[i]))
                << "Gradient mismatch at index " << i << " (actual " << a[i]
                << " vs ref " << b[i] << ") with dtype "
                << static_cast<int>(dtype_);
        }
    }
}

// ============================================================================
// Spatial Scale Test
// ============================================================================

TEST_P(ROIOpsMultiDTypeTest, SpatialScale) {
    const int64_t output_h = 7;
    const int64_t output_w = 7;

    std::vector<double> scales = {1.0 / 4.0, 1.0 / 8.0, 1.0 / 16.0, 1.0 / 32.0};

    auto feats_cpu = tenzor::randn({1, 3, 16, 16}, DType::Float32, Device::cpu());
    auto rois_cpu_full = createROIs(2, 0).to(Device::cpu());
    Variable features = Variable(feats_cpu.to(dtype_).to(device_), false);
    auto rois = createROIs(2, 0);

    for (double scale : scales) {
        ROIAlign roi_ref(output_h, output_w, scale, 2);
        auto ref = roi_ref.forward(Variable(feats_cpu, false), rois_cpu_full);

        ROIAlign roi_align(output_h, output_w, scale, 2);
        convert_model(roi_align);
        auto output = roi_align.forward(features, rois);

        expectShape(output.tensor(), {2, 3, output_h, output_w});
        expectDType(output.tensor());
        expectTensorNear(output.tensor(), ref.tensor(), std::max(atol_, 5e-2f));

        // Verify output contains finite values
        auto output_f32 = output.tensor().to(Device::cpu()).to(DType::Float32);
        auto* data = output_f32.data<float>();
        for (int64_t i = 0; i < output_f32.numel(); ++i) {
            EXPECT_FALSE(std::isnan(data[i]));
            EXPECT_FALSE(std::isinf(data[i]));
        }
    }
}

// ============================================================================
// Single ROI Test
// ============================================================================

TEST_P(ROIOpsMultiDTypeTest, SingleROI) {
    const int64_t output_h = 7;
    const int64_t output_w = 7;

    ROIAlign roi_ref(output_h, output_w, 1.0 / 16.0, 2);
    auto feats_cpu = tenzor::randn({1, 3, 16, 16}, DType::Float32, Device::cpu());
    auto rois_cpu_full = createROIs(1, 0).to(Device::cpu());
    auto ref = roi_ref.forward(Variable(feats_cpu, false), rois_cpu_full);

    ROIAlign roi_align(output_h, output_w, 1.0 / 16.0, 2);
    convert_model(roi_align);

    Variable features = Variable(feats_cpu.to(dtype_).to(device_), false);
    auto rois = createROIs(1, 0);

    auto output = roi_align.forward(features, rois);

    expectShape(output.tensor(), {1, 3, output_h, output_w});
    expectDType(output.tensor());
    expectTensorNear(output.tensor(), ref.tensor(), std::max(atol_, 5e-2f));

    // Verify output contains finite values
    auto output_f32 = output.tensor().to(Device::cpu()).to(DType::Float32);
    auto* data = output_f32.data<float>();
    for (int64_t i = 0; i < output_f32.numel(); ++i) {
        EXPECT_FALSE(std::isnan(data[i]));
        EXPECT_FALSE(std::isinf(data[i]));
    }
}

// ============================================================================
// Test Instantiation
// ============================================================================

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(ROIOpsMultiDTypeTest);
