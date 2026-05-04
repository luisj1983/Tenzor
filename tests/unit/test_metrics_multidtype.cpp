/**
 * @file test_metrics_multidtype.cpp
 * @brief Multi-backend × multi-dtype tests for nn::metric classes.
 *
 * Closes audit-2026-05-03 N1 ("Metrics tests CPU-only"). The existing
 * `tests/unit/test_metrics.cpp` and `_extended.cpp` use TEST_F on CPU
 * Float32; this file lifts each metric to MultiBackendDTypeTest so the
 * GPU dispatch surface (which exercises argmax / comparisons / reductions
 * inside metric.update()) gets covered.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/nn/metrics.hpp>
#include <tenzor/ops/creation.hpp>
#include "../multi_backend_dtype_fixture.hpp"

using namespace tenzor;
using namespace tenzor::nn;
using namespace tenzor::testing;

class MetricsMultiBackendDTypeTest : public MultiBackendDTypeTest {};

namespace {

// Build a (B, C) prediction tensor with one-hot-ish "logits" on the
// requested device/dtype. preds[i, label_i] = high; everything else = low.
auto make_preds(const std::vector<int64_t>& labels, int64_t C,
                Device dev, DType dt) -> Tensor {
    int64_t B = static_cast<int64_t>(labels.size());
    auto cpu = zeros({B, C}, DType::Float32, Device::cpu());
    auto* p = cpu.data<float>();
    for (int64_t i = 0; i < B; ++i) {
        for (int64_t c = 0; c < C; ++c) {
            p[i * C + c] = (c == labels[i]) ? 0.9f : 0.1f;
        }
    }
    return cpu.to(dt).to(dev);
}

auto make_targets(const std::vector<int64_t>& labels, Device dev, DType dt)
    -> Tensor {
    int64_t B = static_cast<int64_t>(labels.size());
    auto cpu = zeros({B}, DType::Float32, Device::cpu());
    auto* p = cpu.data<float>();
    for (int64_t i = 0; i < B; ++i) p[i] = static_cast<float>(labels[i]);
    return cpu.to(dt).to(dev);
}

auto scalar_to_double(const Tensor& t) -> double {
    auto cpu = t.to(Device::cpu()).to(DType::Float64).contiguous();
    return cpu.data<double>()[0];
}

} // anonymous

// ============================================================================
// Accuracy
// ============================================================================

TEST_P(MetricsMultiBackendDTypeTest, Accuracy_Perfect) {
    Accuracy acc(/*num_classes=*/2);
    auto preds = make_preds({1, 0, 1}, 2, device(), dtype());
    auto targets = make_targets({1, 0, 1}, device(), dtype());
    acc.update(preds, targets);
    EXPECT_NEAR(scalar_to_double(acc.compute()), 1.0, 1e-3);
}

TEST_P(MetricsMultiBackendDTypeTest, Accuracy_Partial) {
    Accuracy acc(2);
    // pred = [1, 0, 1, 0]; target = [1, 1, 0, 0] → 2/4 = 0.5
    auto preds = make_preds({1, 0, 1, 0}, 2, device(), dtype());
    auto targets = make_targets({1, 1, 0, 0}, device(), dtype());
    acc.update(preds, targets);
    EXPECT_NEAR(scalar_to_double(acc.compute()), 0.5, 1e-3);
}

// ============================================================================
// Precision / Recall / F1Score
// ============================================================================

TEST_P(MetricsMultiBackendDTypeTest, Precision_Perfect) {
    Precision prec(2);
    auto preds = make_preds({1, 0, 1, 0}, 2, device(), dtype());
    auto targets = make_targets({1, 0, 1, 0}, device(), dtype());
    prec.update(preds, targets);
    EXPECT_NEAR(scalar_to_double(prec.compute()), 1.0, 1e-3);
}

TEST_P(MetricsMultiBackendDTypeTest, Recall_Perfect) {
    Recall rec(2);
    auto preds = make_preds({1, 0, 1, 0}, 2, device(), dtype());
    auto targets = make_targets({1, 0, 1, 0}, device(), dtype());
    rec.update(preds, targets);
    EXPECT_NEAR(scalar_to_double(rec.compute()), 1.0, 1e-3);
}

TEST_P(MetricsMultiBackendDTypeTest, F1Score_Perfect) {
    F1Score f1(2);
    auto preds = make_preds({1, 0, 1, 0}, 2, device(), dtype());
    auto targets = make_targets({1, 0, 1, 0}, device(), dtype());
    f1.update(preds, targets);
    EXPECT_NEAR(scalar_to_double(f1.compute()), 1.0, 1e-3);
}

// ============================================================================
// Regression metrics — MAE / MSE
// ============================================================================

TEST_P(MetricsMultiBackendDTypeTest, MAE_KnownValues) {
    MeanAbsoluteError mae;
    // |1-2| + |3-1| + |0-0| = 1 + 2 + 0 = 3, mean = 1.0
    auto cpu_preds = zeros({3}, DType::Float32, Device::cpu());
    auto* pp = cpu_preds.data<float>();
    pp[0] = 1.0f; pp[1] = 3.0f; pp[2] = 0.0f;
    auto cpu_t = zeros({3}, DType::Float32, Device::cpu());
    auto* pt = cpu_t.data<float>();
    pt[0] = 2.0f; pt[1] = 1.0f; pt[2] = 0.0f;
    auto preds = cpu_preds.to(dtype()).to(device());
    auto targets = cpu_t.to(dtype()).to(device());
    mae.update(preds, targets);
    // Half-precision can only resolve 1.0 to ~1e-3.
    const double tol = (dtype() == DType::Float16 ||
                        dtype() == DType::BFloat16) ? 5e-2 : 1e-3;
    EXPECT_NEAR(scalar_to_double(mae.compute()), 1.0, tol);
}

TEST_P(MetricsMultiBackendDTypeTest, MSE_KnownValues) {
    MeanSquaredError mse;
    // (1-2)^2 + (3-1)^2 + (0-0)^2 = 1 + 4 + 0 = 5, mean = 5/3
    auto cpu_preds = zeros({3}, DType::Float32, Device::cpu());
    auto* pp = cpu_preds.data<float>();
    pp[0] = 1.0f; pp[1] = 3.0f; pp[2] = 0.0f;
    auto cpu_t = zeros({3}, DType::Float32, Device::cpu());
    auto* pt = cpu_t.data<float>();
    pt[0] = 2.0f; pt[1] = 1.0f; pt[2] = 0.0f;
    auto preds = cpu_preds.to(dtype()).to(device());
    auto targets = cpu_t.to(dtype()).to(device());
    mse.update(preds, targets);
    const double tol = (dtype() == DType::Float16 ||
                        dtype() == DType::BFloat16) ? 5e-2 : 1e-3;
    EXPECT_NEAR(scalar_to_double(mse.compute()), 5.0 / 3.0, tol);
}

// ============================================================================
// AUROC and ConfusionMatrix
// ============================================================================

TEST_P(MetricsMultiBackendDTypeTest, AUROC_PerfectPredictions) {
    AUROC auroc;
    // Perfect: positives have higher score than negatives.
    auto cpu_preds = zeros({4}, DType::Float32, Device::cpu());
    auto* pp = cpu_preds.data<float>();
    pp[0] = 0.9f; pp[1] = 0.8f; pp[2] = 0.2f; pp[3] = 0.1f;
    auto cpu_t = zeros({4}, DType::Float32, Device::cpu());
    auto* pt = cpu_t.data<float>();
    pt[0] = 1.0f; pt[1] = 1.0f; pt[2] = 0.0f; pt[3] = 0.0f;
    auto preds = cpu_preds.to(dtype()).to(device());
    auto targets = cpu_t.to(dtype()).to(device());
    auroc.update(preds, targets);
    EXPECT_NEAR(scalar_to_double(auroc.compute()), 1.0, 1e-2);
}

TEST_P(MetricsMultiBackendDTypeTest, ConfusionMatrix_Construction) {
    ConfusionMatrix cm(/*num_classes=*/3);
    auto preds = make_preds({0, 1, 2, 0, 1}, 3, device(), dtype());
    auto targets = make_targets({0, 1, 2, 0, 1}, device(), dtype());
    cm.update(preds, targets);
    auto m = cm.compute();
    // Result is a 3x3 matrix.
    ASSERT_EQ(m.shape().size(), 2u);
    EXPECT_EQ(m.shape()[0], 3);
    EXPECT_EQ(m.shape()[1], 3);
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(MetricsMultiBackendDTypeTest);
