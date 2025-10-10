#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <cmath>

using namespace tenzor;
using namespace tenzor::nn;

class LossTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Common test data setup
    }
};

// MSE Loss Tests
TEST_F(LossTest, MSELossBasic) {
    // Create simple predictions and targets
    auto pred = Variable(full({2, 3}, 1.0f, DType::Float32), true);
    auto target = Variable(full({2, 3}, 2.0f, DType::Float32), false);

    auto loss = mse_loss(pred, target, Reduction::Mean);

    // Expected: mean((1-2)^2) = mean(1) = 1.0
    EXPECT_EQ(loss.tensor().numel(), 1);
    const float* loss_data = loss.tensor().data<float>();
    EXPECT_FLOAT_EQ(loss_data[0], 1.0f);
}

TEST_F(LossTest, MSELossZero) {
    // Perfect predictions
    auto pred = Variable(full({3, 3}, 5.0f, DType::Float32), true);
    auto target = Variable(full({3, 3}, 5.0f, DType::Float32), false);

    auto loss = mse_loss(pred, target, Reduction::Mean);

    const float* loss_data = loss.tensor().data<float>();
    EXPECT_NEAR(loss_data[0], 0.0f, 1e-6);
}

TEST_F(LossTest, MSELossSum) {
    auto pred = Variable(full({2, 2}, 1.0f, DType::Float32), true);
    auto target = Variable(full({2, 2}, 2.0f, DType::Float32), false);

    auto loss = mse_loss(pred, target, Reduction::Sum);

    // Expected: sum((1-2)^2) = sum(1, 1, 1, 1) = 4.0
    const float* loss_data = loss.tensor().data<float>();
    EXPECT_FLOAT_EQ(loss_data[0], 4.0f);
}

TEST_F(LossTest, MSELossNone) {
    auto pred = Variable(full({2, 2}, 1.0f, DType::Float32), true);
    auto target = Variable(full({2, 2}, 2.0f, DType::Float32), false);

    auto loss = mse_loss(pred, target, Reduction::None);

    // Expected: element-wise (1-2)^2 = [1, 1, 1, 1]
    EXPECT_EQ(loss.tensor().numel(), 4);
    const float* loss_data = loss.tensor().data<float>();
    for (int i = 0; i < 4; i++) {
        EXPECT_FLOAT_EQ(loss_data[i], 1.0f);
    }
}

// Binary Cross Entropy Tests
TEST_F(LossTest, BCELossBasic) {
    auto pred = Variable(full({2, 3}, 0.5f, DType::Float32), true);
    auto target = Variable(ones({2, 3}, DType::Float32), false);

    auto loss = bce_loss(pred, target, Reduction::Mean);

    // Expected: -[1 * log(0.5) + 0 * log(0.5)] = -log(0.5) ≈ 0.693
    const float* loss_data = loss.tensor().data<float>();
    EXPECT_GT(loss_data[0], 0.0f);
    EXPECT_NEAR(loss_data[0], -std::log(0.5f), 0.01f);
}

TEST_F(LossTest, BCELossPerfectPrediction) {
    auto pred = Variable(ones({3, 3}, DType::Float32), true);
    auto target = Variable(ones({3, 3}, DType::Float32), false);

    auto loss = bce_loss(pred, target, Reduction::Mean);

    // With clamping, we expect a very small loss (not exactly 0)
    const float* loss_data = loss.tensor().data<float>();
    EXPECT_LT(loss_data[0], 0.001f);
}

TEST_F(LossTest, BCELossMixedTargets) {
    // Create predictions and targets
    auto pred = Variable(full({2}, 0.7f, DType::Float32), true);
    auto target = Variable(full({2}, 0.5f, DType::Float32), false);

    auto loss = bce_loss(pred, target, Reduction::Mean);

    // Loss should be positive
    const float* loss_data = loss.tensor().data<float>();
    EXPECT_GT(loss_data[0], 0.0f);
}

// L1 Loss Tests
TEST_F(LossTest, L1LossBasic) {
    auto pred = Variable(full({2, 3}, 3.0f, DType::Float32), true);
    auto target = Variable(full({2, 3}, 1.0f, DType::Float32), false);

    auto loss = l1_loss(pred, target, Reduction::Mean);

    // Expected: mean(|3-1|) = mean(2) = 2.0
    const float* loss_data = loss.tensor().data<float>();
    EXPECT_FLOAT_EQ(loss_data[0], 2.0f);
}

TEST_F(LossTest, L1LossNegativeDiff) {
    auto pred = Variable(full({2, 2}, 1.0f, DType::Float32), true);
    auto target = Variable(full({2, 2}, 3.0f, DType::Float32), false);

    auto loss = l1_loss(pred, target, Reduction::Mean);

    // Expected: mean(|1-3|) = mean(2) = 2.0
    const float* loss_data = loss.tensor().data<float>();
    EXPECT_FLOAT_EQ(loss_data[0], 2.0f);
}

TEST_F(LossTest, L1LossSum) {
    auto pred = Variable(full({2, 2}, 5.0f, DType::Float32), true);
    auto target = Variable(full({2, 2}, 3.0f, DType::Float32), false);

    auto loss = l1_loss(pred, target, Reduction::Sum);

    // Expected: sum(|5-3|) = sum(2, 2, 2, 2) = 8.0
    const float* loss_data = loss.tensor().data<float>();
    EXPECT_FLOAT_EQ(loss_data[0], 8.0f);
}

TEST_F(LossTest, L1LossZero) {
    auto pred = Variable(full({3, 3}, 7.0f, DType::Float32), true);
    auto target = Variable(full({3, 3}, 7.0f, DType::Float32), false);

    auto loss = l1_loss(pred, target, Reduction::Mean);

    const float* loss_data = loss.tensor().data<float>();
    EXPECT_FLOAT_EQ(loss_data[0], 0.0f);
}

// Cross Entropy Loss Tests
TEST_F(LossTest, CrossEntropyBasic) {
    // Create logits for 2 samples, 3 classes
    auto logits = Variable(full({2, 3}, 0.0f, DType::Float32), true);
    auto targets = zeros({2, 3}, DType::Float32);

    // Set first class to 1.0 for both samples
    float* target_data = const_cast<float*>(targets.data<float>());
    target_data[0] = 1.0f;  // First sample, first class
    target_data[3] = 1.0f;  // Second sample, first class

    auto loss = cross_entropy(logits, targets, Reduction::Mean);

    // Loss should be positive
    const float* loss_data = loss.tensor().data<float>();
    EXPECT_GT(loss_data[0], 0.0f);
}

TEST_F(LossTest, CrossEntropyUniformLogits) {
    // For uniform logits across classes, cross entropy should be approximately log(num_classes)
    auto logits = Variable(zeros({4, 3}, DType::Float32), true);
    auto targets = zeros({4, 3}, DType::Float32);

    // One-hot encode: each sample has class 0
    float* target_data = const_cast<float*>(targets.data<float>());
    for (int i = 0; i < 4; i++) {
        target_data[i * 3] = 1.0f;
    }

    auto loss = cross_entropy(logits, targets, Reduction::Mean);

    const float* loss_data = loss.tensor().data<float>();
    // For uniform probabilities over 3 classes: -log(1/3) ≈ 1.099
    EXPECT_NEAR(loss_data[0], std::log(3.0f), 0.1f);
}

// NLL Loss Tests
TEST_F(LossTest, NLLLossBasic) {
    // Create log probabilities (already log-transformed)
    auto log_probs = Variable(full({2, 3}, std::log(0.5f), DType::Float32), true);
    auto targets = zeros({2, 3}, DType::Float32);

    float* target_data = const_cast<float*>(targets.data<float>());
    target_data[0] = 1.0f;
    target_data[3] = 1.0f;

    auto loss = nll_loss(log_probs, targets, Reduction::Mean);

    const float* loss_data = loss.tensor().data<float>();
    EXPECT_GT(loss_data[0], 0.0f);
}

// Test loss function classes (not just functional API)
TEST_F(LossTest, MSELossClass) {
    MSELoss criterion(Reduction::Mean);

    auto pred = Variable(full({2, 2}, 2.0f, DType::Float32), true);
    auto target = Variable(full({2, 2}, 1.0f, DType::Float32), false);

    auto loss = criterion(pred, target);

    const float* loss_data = loss.tensor().data<float>();
    EXPECT_FLOAT_EQ(loss_data[0], 1.0f);
}

TEST_F(LossTest, BCELossClass) {
    BCELoss criterion(Reduction::Mean);

    auto pred = Variable(full({2, 2}, 0.8f, DType::Float32), true);
    auto target = Variable(ones({2, 2}, DType::Float32), false);

    auto loss = criterion(pred, target);

    const float* loss_data = loss.tensor().data<float>();
    EXPECT_GT(loss_data[0], 0.0f);
}

TEST_F(LossTest, L1LossClass) {
    L1Loss criterion(Reduction::Mean);

    auto pred = Variable(full({3, 3}, 5.0f, DType::Float32), true);
    auto target = Variable(full({3, 3}, 2.0f, DType::Float32), false);

    auto loss = criterion(pred, target);

    const float* loss_data = loss.tensor().data<float>();
    EXPECT_FLOAT_EQ(loss_data[0], 3.0f);
}

TEST_F(LossTest, CrossEntropyLossClass) {
    CrossEntropyLoss criterion(Reduction::Mean);

    auto logits = Variable(zeros({2, 3}, DType::Float32), true);
    auto targets = zeros({2, 3}, DType::Float32);

    float* target_data = const_cast<float*>(targets.data<float>());
    target_data[0] = 1.0f;
    target_data[3] = 1.0f;

    auto loss = criterion(logits, targets);

    const float* loss_data = loss.tensor().data<float>();
    EXPECT_GT(loss_data[0], 0.0f);
}

// Gradient requirement tests
TEST_F(LossTest, MSELossGradientRequired) {
    auto pred = Variable(full({2, 2}, 1.0f, DType::Float32), true);
    auto target = Variable(full({2, 2}, 2.0f, DType::Float32), false);

    auto loss = mse_loss(pred, target, Reduction::Mean);

    // Loss should require gradient since pred requires grad
    EXPECT_TRUE(loss.requires_grad());
}

TEST_F(LossTest, BCELossGradientRequired) {
    auto pred = Variable(full({2, 2}, 0.6f, DType::Float32), true);
    auto target = Variable(ones({2, 2}, DType::Float32), false);

    auto loss = bce_loss(pred, target, Reduction::Mean);

    EXPECT_TRUE(loss.requires_grad());
}

TEST_F(LossTest, L1LossGradientRequired) {
    auto pred = Variable(full({2, 2}, 3.0f, DType::Float32), true);
    auto target = Variable(full({2, 2}, 1.0f, DType::Float32), false);

    auto loss = l1_loss(pred, target, Reduction::Mean);

    EXPECT_TRUE(loss.requires_grad());
}
