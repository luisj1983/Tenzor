/**
 * @file test_metrics_extended.cpp
 * @brief Unit tests for AUROC and ConfusionMatrix metrics.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/nn/metrics.hpp>
#include <tenzor/ops/creation.hpp>
#include <cmath>

#include "../backend_test_fixture.hpp"

using namespace tenzor;
using namespace tenzor::nn;

class MetricsExtendedTest : public ::tenzor::testing::BackendTest {
protected:
    void SetUp() override {
        ::tenzor::testing::BackendTest::SetUp();
        if (::testing::Test::IsSkipped()) return;
    }
};

// ============================================================================
// AUROC tests
// ============================================================================

TEST_P(MetricsExtendedTest, AUROC_Construction) {
    AUROC auroc;
    // Should not crash; name should be correct.
    EXPECT_EQ(auroc.name(), "auroc");
}

TEST_P(MetricsExtendedTest, AUROC_PerfectPredictions) {
    // Negative class gets low scores, positive class gets high scores -> AUC = 1.0
    AUROC auroc;

    float pred_data[] = {0.0f, 0.0f, 1.0f, 1.0f};
    float target_data[] = {0.0f, 0.0f, 1.0f, 1.0f};

    auto preds = from_data(pred_data, {4}, device);
    auto targets = from_data(target_data, {4}, device);

    auroc.update(preds, targets);
    auto result = auroc.compute();

    auto result_cpu = result.cpu();
    EXPECT_NEAR(result_cpu.data<float>()[0], 1.0f, 1e-5);
}

TEST_P(MetricsExtendedTest, AUROC_InvertedPredictions) {
    // Perfectly inverted: positive class gets low scores, negative gets high -> AUC = 0.0
    AUROC auroc;

    float pred_data[] = {1.0f, 1.0f, 0.0f, 0.0f};
    float target_data[] = {0.0f, 0.0f, 1.0f, 1.0f};

    auto preds = from_data(pred_data, {4}, device);
    auto targets = from_data(target_data, {4}, device);

    auroc.update(preds, targets);
    auto result = auroc.compute();

    auto result_cpu = result.cpu();
    EXPECT_NEAR(result_cpu.data<float>()[0], 0.0f, 1e-5);
}

TEST_P(MetricsExtendedTest, AUROC_RandomPredictions) {
    // Mixed predictions should give AUC between 0 and 1
    AUROC auroc;

    float pred_data[] = {0.3f, 0.7f, 0.4f, 0.6f, 0.2f, 0.8f};
    float target_data[] = {0.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f};

    auto preds = from_data(pred_data, {6}, device);
    auto targets = from_data(target_data, {6}, device);

    auroc.update(preds, targets);
    auto result = auroc.compute();

    auto result_cpu = result.cpu();
    float auc = result_cpu.data<float>()[0];
    EXPECT_GE(auc, 0.0f);
    EXPECT_LE(auc, 1.0f);
}

TEST_P(MetricsExtendedTest, AUROC_Reset) {
    AUROC auroc;

    float pred_data[] = {0.0f, 1.0f};
    float target_data[] = {0.0f, 1.0f};

    auto preds = from_data(pred_data, {2}, device);
    auto targets = from_data(target_data, {2}, device);

    auroc.update(preds, targets);
    auto result1 = auroc.compute();
    auto result1_cpu = result1.cpu();
    EXPECT_NEAR(result1_cpu.data<float>()[0], 1.0f, 1e-5);

    auroc.reset();

    // After reset, compute on new (inverted) data should differ.
    float pred_data2[] = {1.0f, 0.0f};
    float target_data2[] = {0.0f, 1.0f};

    auto preds2 = from_data(pred_data2, {2}, device);
    auto targets2 = from_data(target_data2, {2}, device);

    auroc.update(preds2, targets2);
    auto result2 = auroc.compute();
    auto result2_cpu = result2.cpu();
    EXPECT_NEAR(result2_cpu.data<float>()[0], 0.0f, 1e-5);
}

TEST_P(MetricsExtendedTest, AUROC_MultipleUpdates) {
    // Multiple update calls should accumulate data identically to a single call.
    AUROC auroc_single;
    AUROC auroc_multi;

    float pred_data1[] = {0.1f, 0.9f};
    float target_data1[] = {0.0f, 1.0f};
    float pred_data2[] = {0.2f, 0.8f};
    float target_data2[] = {0.0f, 1.0f};

    auto p1 = from_data(pred_data1, {2}, device);
    auto t1 = from_data(target_data1, {2}, device);
    auto p2 = from_data(pred_data2, {2}, device);
    auto t2 = from_data(target_data2, {2}, device);

    // Single update with all data
    float pred_all[] = {0.1f, 0.9f, 0.2f, 0.8f};
    float target_all[] = {0.0f, 1.0f, 0.0f, 1.0f};
    auroc_single.update(from_data(pred_all, {4}, device), from_data(target_all, {4}, device));

    // Multiple updates
    auroc_multi.update(p1, t1);
    auroc_multi.update(p2, t2);

    auto result_single = auroc_single.compute();
    auto result_multi = auroc_multi.compute();

    auto result_single_cpu = result_single.cpu();
    auto result_multi_cpu = result_multi.cpu();
    EXPECT_NEAR(result_single_cpu.data<float>()[0], result_multi_cpu.data<float>()[0], 1e-5);
}

// ============================================================================
// ConfusionMatrix tests
// ============================================================================

TEST_P(MetricsExtendedTest, ConfusionMatrix_Construction) {
    ConfusionMatrix cm(3);
    EXPECT_EQ(cm.name(), "confusion_matrix");
}

TEST_P(MetricsExtendedTest, ConfusionMatrix_PerfectPredictions) {
    // 3-class, all predictions correct -> diagonal nonzero, off-diagonal zero
    ConfusionMatrix cm(3);

    // Predictions as class logits: each row is [c0, c1, c2], argmax gives class
    // Sample 0: class 0 (logits: 1,0,0), Sample 1: class 1 (0,1,0), Sample 2: class 2 (0,0,1)
    float pred_data[] = {
        1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 1.0f
    };
    float target_data[] = {0.0f, 1.0f, 2.0f};

    auto preds = from_data(pred_data, {3, 3}, device);
    auto targets = from_data(target_data, {3}, device);

    cm.update(preds, targets);
    auto mat = cm.compute();

    // Check shape: 3x3
    ASSERT_EQ(mat.shape().size(), 2u);
    EXPECT_EQ(mat.shape()[0], 3);
    EXPECT_EQ(mat.shape()[1], 3);

    auto mat_cpu = mat.cpu();
    const float* d = mat_cpu.data<float>();
    // Diagonal should be 1 each
    EXPECT_NEAR(d[0 * 3 + 0], 1.0f, 1e-5);  // (0,0)
    EXPECT_NEAR(d[1 * 3 + 1], 1.0f, 1e-5);  // (1,1)
    EXPECT_NEAR(d[2 * 3 + 2], 1.0f, 1e-5);  // (2,2)

    // Off-diagonal should be 0
    EXPECT_NEAR(d[0 * 3 + 1], 0.0f, 1e-5);
    EXPECT_NEAR(d[0 * 3 + 2], 0.0f, 1e-5);
    EXPECT_NEAR(d[1 * 3 + 0], 0.0f, 1e-5);
    EXPECT_NEAR(d[1 * 3 + 2], 0.0f, 1e-5);
    EXPECT_NEAR(d[2 * 3 + 0], 0.0f, 1e-5);
    EXPECT_NEAR(d[2 * 3 + 1], 0.0f, 1e-5);
}

TEST_P(MetricsExtendedTest, ConfusionMatrix_KnownConfusion) {
    // 3-class problem:
    // predictions (argmax) = {0, 1, 0, 2}
    // targets               = {0, 0, 0, 2}
    //
    // Expected matrix (row=true, col=predicted):
    //          pred0  pred1  pred2
    // true 0:   2      1      0     (samples 0,2 correct; sample 1 misclassified as 1)
    // true 1:   0      0      0
    // true 2:   0      0      1     (sample 3 correct)
    ConfusionMatrix cm(3);

    float pred_data[] = {
        1.0f, 0.0f, 0.0f,   // argmax -> 0
        0.0f, 1.0f, 0.0f,   // argmax -> 1
        1.0f, 0.0f, 0.0f,   // argmax -> 0
        0.0f, 0.0f, 1.0f    // argmax -> 2
    };
    float target_data[] = {0.0f, 0.0f, 0.0f, 2.0f};

    auto preds = from_data(pred_data, {4, 3}, device);
    auto targets = from_data(target_data, {4}, device);

    cm.update(preds, targets);
    auto mat = cm.compute();

    auto mat_cpu = mat.cpu();
    const float* d = mat_cpu.data<float>();
    // matrix[0][0] = 2 (true=0, pred=0)
    EXPECT_NEAR(d[0 * 3 + 0], 2.0f, 1e-5);
    // matrix[0][1] = 1 (true=0, pred=1)
    EXPECT_NEAR(d[0 * 3 + 1], 1.0f, 1e-5);
    // matrix[0][2] = 0
    EXPECT_NEAR(d[0 * 3 + 2], 0.0f, 1e-5);
    // matrix[2][2] = 1 (true=2, pred=2)
    EXPECT_NEAR(d[2 * 3 + 2], 1.0f, 1e-5);
    // All of row 1 should be zero (no samples with true label 1)
    EXPECT_NEAR(d[1 * 3 + 0], 0.0f, 1e-5);
    EXPECT_NEAR(d[1 * 3 + 1], 0.0f, 1e-5);
    EXPECT_NEAR(d[1 * 3 + 2], 0.0f, 1e-5);
}

TEST_P(MetricsExtendedTest, ConfusionMatrix_Reset) {
    ConfusionMatrix cm(2);

    float pred_data[] = {0.0f, 1.0f, 1.0f, 0.0f};
    float target_data[] = {1.0f, 0.0f};

    auto preds = from_data(pred_data, {2, 2}, device);
    auto targets = from_data(target_data, {2}, device);

    cm.update(preds, targets);
    auto mat1 = cm.compute();

    cm.reset();
    auto mat2 = cm.compute();

    auto mat2_cpu = mat2.cpu();
    const float* d = mat2_cpu.data<float>();
    // After reset, matrix should be all zeros
    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(d[i], 0.0f, 1e-5) << "index " << i;
    }
}

TEST_P(MetricsExtendedTest, ConfusionMatrix_MatrixShape) {
    // Verify that compute() returns an (N, N) tensor for various N
    for (int64_t n : {2, 3, 5, 10}) {
        ConfusionMatrix cm(n);
        auto mat = cm.compute();

        ASSERT_EQ(mat.shape().size(), 2u) << "num_classes=" << n;
        EXPECT_EQ(mat.shape()[0], n) << "num_classes=" << n;
        EXPECT_EQ(mat.shape()[1], n) << "num_classes=" << n;
    }
}

INSTANTIATE_BACKEND_TESTS(MetricsExtendedTest);
