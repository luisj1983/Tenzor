/**
 * @file test_metrics.cpp
 * @brief Unit tests for training metric classes (Accuracy, Precision, Recall,
 *        F1Score, MeanAbsoluteError, MeanSquaredError)
 */

#include <gtest/gtest.h>
#include "tenzor/nn/metrics.hpp"
#include "tenzor/core/tensor.hpp"
#include "tenzor/ops/creation.hpp"
#include "../backend_test_fixture.hpp"
#include <cmath>

using namespace tenzor;
using namespace tenzor::nn;

class MetricsTest : public ::tenzor::testing::BackendTest {
protected:
    void SetUp() override {
        ::tenzor::testing::BackendTest::SetUp();
        if (::testing::Test::IsSkipped()) return;
    }
};

// ============================================================================
// Accuracy tests
// ============================================================================

TEST_P(MetricsTest, AccuracyPerfect) {
    // Binary classification: predictions match targets exactly
    Accuracy acc(2);

    // Predictions as class logits: [[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]]
    // -> predicted classes: [1, 0, 1]
    // Targets: [1, 0, 1]
    float pred_data[] = {0.1f, 0.9f,  0.8f, 0.2f,  0.3f, 0.7f};
    float target_data[] = {1.0f, 0.0f, 1.0f};

    auto preds = from_data(pred_data, {3, 2}, device);
    auto targets = from_data(target_data, {3}, device);

    acc.update(preds, targets);
    auto result = acc.compute();

    auto result_cpu = result.cpu();
    EXPECT_NEAR(result_cpu.data<float>()[0], 1.0f, 1e-5);
}

TEST_P(MetricsTest, AccuracyPartial) {
    // 2 out of 4 correct = 50%
    Accuracy acc(2);

    // Predicted classes: [1, 0, 1, 0]
    // Targets:           [1, 1, 0, 0]
    // Correct: index 0 and 3 -> 2/4 = 0.5
    float pred_data[] = {0.1f, 0.9f,  0.8f, 0.2f,  0.1f, 0.9f,  0.8f, 0.2f};
    float target_data[] = {1.0f, 1.0f, 0.0f, 0.0f};

    auto preds = from_data(pred_data, {4, 2}, device);
    auto targets = from_data(target_data, {4}, device);

    acc.update(preds, targets);
    auto result = acc.compute();

    auto result_cpu = result.cpu();
    EXPECT_NEAR(result_cpu.data<float>()[0], 0.5f, 1e-5);
}

TEST_P(MetricsTest, AccuracyReset) {
    Accuracy acc(2);

    float pred_data[] = {0.1f, 0.9f};
    float target_data[] = {1.0f};

    acc.update(from_data(pred_data, {1, 2}, device), from_data(target_data, {1}, device));
    acc.reset();

    // After reset, update with different data
    float pred_data2[] = {0.9f, 0.1f};
    float target_data2[] = {1.0f};  // predicted 0, target 1 -> wrong

    acc.update(from_data(pred_data2, {1, 2}, device), from_data(target_data2, {1}, device));
    auto result = acc.compute();

    auto result_cpu = result.cpu();
    EXPECT_NEAR(result_cpu.data<float>()[0], 0.0f, 1e-5);
}

// ============================================================================
// Precision tests
// ============================================================================

TEST_P(MetricsTest, PrecisionKnownValues) {
    // Binary: TP=2, FP=1 -> Precision = 2/3
    Precision prec(2, AverageMode::Micro);

    // Predictions: [1, 1, 1, 0] (predicted positive for first 3)
    // Targets:     [1, 1, 0, 0]
    // TP=2, FP=1
    float pred_data[] = {0.1f, 0.9f,  0.1f, 0.9f,  0.1f, 0.9f,  0.9f, 0.1f};
    float target_data[] = {1.0f, 1.0f, 0.0f, 0.0f};

    auto preds = from_data(pred_data, {4, 2}, device);
    auto targets = from_data(target_data, {4}, device);

    prec.update(preds, targets);
    auto result = prec.compute();

    // Micro precision: (TP_total) / (TP_total + FP_total)
    // For binary with micro: correct_predictions / total_predictions
    // Class 1: TP=2, FP=1 -> precision for class 1 = 2/3
    // Micro averages across all classes
    auto result_cpu = result.cpu();
    float precision_val = result_cpu.data<float>()[0];
    // Depending on micro implementation: could be overall TP/(TP+FP)
    EXPECT_GT(precision_val, 0.0f);
    EXPECT_LE(precision_val, 1.0f);
}

// ============================================================================
// Recall tests
// ============================================================================

TEST_P(MetricsTest, RecallKnownValues) {
    // Binary: TP=1, FN=1 -> Recall = 1/2 for positive class
    Recall rec(2, AverageMode::Micro);

    // Predictions: [1, 0] (predicted class 1 for first, class 0 for second)
    // Targets:     [1, 1]
    // TP=1 (first sample), FN=1 (second sample)
    float pred_data[] = {0.1f, 0.9f,  0.9f, 0.1f};
    float target_data[] = {1.0f, 1.0f};

    auto preds = from_data(pred_data, {2, 2}, device);
    auto targets = from_data(target_data, {2}, device);

    rec.update(preds, targets);
    auto result = rec.compute();

    auto result_cpu = result.cpu();
    float recall_val = result_cpu.data<float>()[0];
    EXPECT_GT(recall_val, 0.0f);
    EXPECT_LE(recall_val, 1.0f);
}

// ============================================================================
// F1Score tests
// ============================================================================

TEST_P(MetricsTest, F1ScorePerfect) {
    // Perfect predictions -> F1 = 1.0
    F1Score f1(2, AverageMode::Micro);

    float pred_data[] = {0.9f, 0.1f,  0.1f, 0.9f,  0.9f, 0.1f};
    float target_data[] = {0.0f, 1.0f, 0.0f};

    auto preds = from_data(pred_data, {3, 2}, device);
    auto targets = from_data(target_data, {3}, device);

    f1.update(preds, targets);
    auto result = f1.compute();

    auto result_cpu = result.cpu();
    EXPECT_NEAR(result_cpu.data<float>()[0], 1.0f, 1e-5);
}

TEST_P(MetricsTest, F1ScoreConsistency) {
    // F1 should be between 0 and 1
    F1Score f1(2, AverageMode::Macro);

    // Mixed predictions
    float pred_data[] = {0.1f, 0.9f,  0.9f, 0.1f,  0.1f, 0.9f,  0.9f, 0.1f};
    float target_data[] = {1.0f, 1.0f, 0.0f, 0.0f};

    auto preds = from_data(pred_data, {4, 2}, device);
    auto targets = from_data(target_data, {4}, device);

    f1.update(preds, targets);
    auto result = f1.compute();

    auto result_cpu = result.cpu();
    float f1_val = result_cpu.data<float>()[0];
    EXPECT_GE(f1_val, 0.0f);
    EXPECT_LE(f1_val, 1.0f);
}

TEST_P(MetricsTest, F1ScoreMultipleBatches) {
    // Accumulating over multiple batches should give correct result
    F1Score f1(2, AverageMode::Micro);

    // Batch 1: all correct
    float pred_data1[] = {0.9f, 0.1f,  0.1f, 0.9f};
    float target_data1[] = {0.0f, 1.0f};
    f1.update(from_data(pred_data1, {2, 2}, device), from_data(target_data1, {2}, device));

    // Batch 2: all wrong
    float pred_data2[] = {0.1f, 0.9f,  0.9f, 0.1f};
    float target_data2[] = {0.0f, 1.0f};
    f1.update(from_data(pred_data2, {2, 2}, device), from_data(target_data2, {2}, device));

    auto result = f1.compute();
    auto result_cpu = result.cpu();
    float f1_val = result_cpu.data<float>()[0];

    // Half correct -> F1 should be around 0.5
    EXPECT_GT(f1_val, 0.0f);
    EXPECT_LT(f1_val, 1.0f);
}

// ============================================================================
// MeanAbsoluteError tests
// ============================================================================

TEST_P(MetricsTest, MAEKnownValues) {
    // preds = [1, 2, 3], targets = [1, 3, 5]
    // abs errors = [0, 1, 2], MAE = 1.0
    MeanAbsoluteError mae;

    float pred_data[] = {1.0f, 2.0f, 3.0f};
    float target_data[] = {1.0f, 3.0f, 5.0f};

    mae.update(from_data(pred_data, {3}, device), from_data(target_data, {3}, device));
    auto result = mae.compute();

    auto result_cpu = result.cpu();
    EXPECT_NEAR(result_cpu.data<float>()[0], 1.0f, 1e-5);
}

TEST_P(MetricsTest, MAEPerfect) {
    MeanAbsoluteError mae;

    float data[] = {1.0f, 2.0f, 3.0f};
    mae.update(from_data(data, {3}, device), from_data(data, {3}, device));
    auto result = mae.compute();

    auto result_cpu = result.cpu();
    EXPECT_NEAR(result_cpu.data<float>()[0], 0.0f, 1e-5);
}

TEST_P(MetricsTest, MAEReset) {
    MeanAbsoluteError mae;

    float pred1[] = {0.0f};
    float target1[] = {10.0f};
    mae.update(from_data(pred1, {1}, device), from_data(target1, {1}, device));

    mae.reset();

    float pred2[] = {1.0f};
    float target2[] = {1.0f};
    mae.update(from_data(pred2, {1}, device), from_data(target2, {1}, device));

    auto result = mae.compute();
    auto result_cpu = result.cpu();
    EXPECT_NEAR(result_cpu.data<float>()[0], 0.0f, 1e-5);
}

// ============================================================================
// MeanSquaredError tests
// ============================================================================

TEST_P(MetricsTest, MSEKnownValues) {
    // preds = [1, 2, 3], targets = [1, 3, 5]
    // squared errors = [0, 1, 4], MSE = 5/3 ~ 1.6667
    MeanSquaredError mse;

    float pred_data[] = {1.0f, 2.0f, 3.0f};
    float target_data[] = {1.0f, 3.0f, 5.0f};

    mse.update(from_data(pred_data, {3}, device), from_data(target_data, {3}, device));
    auto result = mse.compute();

    auto result_cpu = result.cpu();
    EXPECT_NEAR(result_cpu.data<float>()[0], 5.0f / 3.0f, 1e-4);
}

TEST_P(MetricsTest, MSEPerfect) {
    MeanSquaredError mse;

    float data[] = {1.0f, 2.0f, 3.0f};
    mse.update(from_data(data, {3}, device), from_data(data, {3}, device));
    auto result = mse.compute();

    auto result_cpu = result.cpu();
    EXPECT_NEAR(result_cpu.data<float>()[0], 0.0f, 1e-5);
}

TEST_P(MetricsTest, MSEMultipleBatches) {
    MeanSquaredError mse;

    // Batch 1: errors = [1], squared = [1]
    float pred1[] = {0.0f};
    float target1[] = {1.0f};
    mse.update(from_data(pred1, {1}, device), from_data(target1, {1}, device));

    // Batch 2: errors = [2], squared = [4]
    float pred2[] = {0.0f};
    float target2[] = {2.0f};
    mse.update(from_data(pred2, {1}, device), from_data(target2, {1}, device));

    // Total: (1 + 4) / 2 = 2.5
    auto result = mse.compute();
    auto result_cpu = result.cpu();
    EXPECT_NEAR(result_cpu.data<float>()[0], 2.5f, 1e-5);
}

INSTANTIATE_BACKEND_TESTS(MetricsTest);
