#include <gtest/gtest.h>
#include "../backend_test_fixture.hpp"
#include <cmath>

using namespace tenzor;
using namespace tenzor::testing;
using namespace tenzor::nn;

class LossTest : public BackendTest {
protected:
    void SetUp() override {
        BackendTest::SetUp();
        // Common test data setup
    }
};

// MSE Loss Tests
TEST_P(LossTest, MSELossBasic) {
    // Create simple predictions and targets
    auto pred = Variable(full({2, 3}, 1.0f, DType::Float32, device), true);
    auto target = Variable(full({2, 3}, 2.0f, DType::Float32, device), false);

    auto loss = mse_loss(pred, target, Reduction::Mean);

    // Expected: mean((1-2)^2) = mean(1) = 1.0
    EXPECT_EQ(loss.tensor().numel(), 1) << "Failed on " << device.to_string();
    auto loss_cpu = loss.tensor().to(Device::cpu());  // Store tensor to keep it alive
    const float* loss_data = loss_cpu.data<float>();
    EXPECT_FLOAT_EQ(loss_data[0], 1.0f) << "Failed on " << device.to_string();
}

TEST_P(LossTest, MSELossZero) {
    // Perfect predictions
    auto pred = Variable(full({3, 3}, 5.0f, DType::Float32, device), true);
    auto target = Variable(full({3, 3}, 5.0f, DType::Float32, device), false);

    auto loss = mse_loss(pred, target, Reduction::Mean);

    auto loss_cpu = loss.tensor().to(Device::cpu());


    const float* loss_data = loss_cpu.data<float>();
    EXPECT_NEAR(loss_data[0], 0.0f, 1e-6) << "Failed on " << device.to_string();
}

TEST_P(LossTest, MSELossSum) {
    auto pred = Variable(full({2, 2}, 1.0f, DType::Float32, device), true);
    auto target = Variable(full({2, 2}, 2.0f, DType::Float32, device), false);

    auto loss = mse_loss(pred, target, Reduction::Sum);

    // Expected: sum((1-2)^2) = sum(1, 1, 1, 1) = 4.0
    auto loss_cpu = loss.tensor().to(Device::cpu());

    const float* loss_data = loss_cpu.data<float>();
    EXPECT_FLOAT_EQ(loss_data[0], 4.0f) << "Failed on " << device.to_string();
}

TEST_P(LossTest, MSELossNone) {
    auto pred = Variable(full({2, 2}, 1.0f, DType::Float32, device), true);
    auto target = Variable(full({2, 2}, 2.0f, DType::Float32, device), false);

    auto loss = mse_loss(pred, target, Reduction::None);

    // Expected: element-wise (1-2)^2 = [1, 1, 1, 1]
    EXPECT_EQ(loss.tensor().numel(), 4) << "Failed on " << device.to_string();
    auto loss_cpu = loss.tensor().to(Device::cpu());

    const float* loss_data = loss_cpu.data<float>();
    for (int i = 0; i < 4; i++) {
        EXPECT_FLOAT_EQ(loss_data[i], 1.0f) << "Failed on " << device.to_string();
    }
}

// Binary Cross Entropy Tests
TEST_P(LossTest, BCELossBasic) {
    auto pred = Variable(full({2, 3}, 0.5f, DType::Float32, device), true);
    auto target = Variable(ones({2, 3}, DType::Float32, device), false);

    auto loss = bce_loss(pred, target, Reduction::Mean);

    // Expected: -[1 * log(0.5) + 0 * log(0.5)] = -log(0.5) ≈ 0.693
    auto loss_cpu = loss.tensor().to(Device::cpu());

    const float* loss_data = loss_cpu.data<float>();
    EXPECT_GT(loss_data[0], 0.0f) << "Failed on " << device.to_string();
    EXPECT_NEAR(loss_data[0], -std::log(0.5f), 0.01f) << "Failed on " << device.to_string();
}

TEST_P(LossTest, BCELossPerfectPrediction) {
    auto pred = Variable(ones({3, 3}, DType::Float32, device), true);
    auto target = Variable(ones({3, 3}, DType::Float32, device), false);

    auto loss = bce_loss(pred, target, Reduction::Mean);

    // With clamping, we expect a very small loss (not exactly 0)
    auto loss_cpu = loss.tensor().to(Device::cpu());

    const float* loss_data = loss_cpu.data<float>();
    EXPECT_LT(loss_data[0], 0.001f) << "Failed on " << device.to_string();
}

TEST_P(LossTest, BCELossMixedTargets) {
    // Create predictions and targets
    auto pred = Variable(full({2}, 0.7f, DType::Float32, device), true);
    auto target = Variable(full({2}, 0.5f, DType::Float32, device), false);

    auto loss = bce_loss(pred, target, Reduction::Mean);

    // Loss should be positive
    auto loss_cpu = loss.tensor().to(Device::cpu());

    const float* loss_data = loss_cpu.data<float>();
    EXPECT_GT(loss_data[0], 0.0f) << "Failed on " << device.to_string();
}

// L1 Loss Tests
TEST_P(LossTest, L1LossBasic) {
    auto pred = Variable(full({2, 3}, 3.0f, DType::Float32, device), true);
    auto target = Variable(full({2, 3}, 1.0f, DType::Float32, device), false);

    auto loss = l1_loss(pred, target, Reduction::Mean);

    // Expected: mean(|3-1|) = mean(2) = 2.0
    auto loss_cpu = loss.tensor().to(Device::cpu());

    const float* loss_data = loss_cpu.data<float>();
    EXPECT_FLOAT_EQ(loss_data[0], 2.0f) << "Failed on " << device.to_string();
}

TEST_P(LossTest, L1LossNegativeDiff) {
    auto pred = Variable(full({2, 2}, 1.0f, DType::Float32, device), true);
    auto target = Variable(full({2, 2}, 3.0f, DType::Float32, device), false);

    auto loss = l1_loss(pred, target, Reduction::Mean);

    // Expected: mean(|1-3|) = mean(2) = 2.0
    auto loss_cpu = loss.tensor().to(Device::cpu());

    const float* loss_data = loss_cpu.data<float>();
    EXPECT_FLOAT_EQ(loss_data[0], 2.0f) << "Failed on " << device.to_string();
}

TEST_P(LossTest, L1LossSum) {
    auto pred = Variable(full({2, 2}, 5.0f, DType::Float32, device), true);
    auto target = Variable(full({2, 2}, 3.0f, DType::Float32, device), false);

    auto loss = l1_loss(pred, target, Reduction::Sum);

    // Expected: sum(|5-3|) = sum(2, 2, 2, 2) = 8.0
    auto loss_cpu = loss.tensor().to(Device::cpu());

    const float* loss_data = loss_cpu.data<float>();
    EXPECT_FLOAT_EQ(loss_data[0], 8.0f) << "Failed on " << device.to_string();
}

TEST_P(LossTest, L1LossZero) {
    auto pred = Variable(full({3, 3}, 7.0f, DType::Float32, device), true);
    auto target = Variable(full({3, 3}, 7.0f, DType::Float32, device), false);

    auto loss = l1_loss(pred, target, Reduction::Mean);

    auto loss_cpu = loss.tensor().to(Device::cpu());


    const float* loss_data = loss_cpu.data<float>();
    EXPECT_FLOAT_EQ(loss_data[0], 0.0f) << "Failed on " << device.to_string();
}

// Cross Entropy Loss Tests
TEST_P(LossTest, CrossEntropyBasic) {
    // Create logits for 2 samples, 3 classes
    auto logits = Variable(full({2, 3}, 0.0f, DType::Float32, device), true);
    auto targets = zeros({2, 3}, DType::Float32, device);

    // Set first class to 1.0 for both samples
    auto targets_cpu = targets.to(Device::cpu());
    float* target_data = const_cast<float*>(targets_cpu.data<float>());
    target_data[0] = 1.0f;  // First sample, first class
    target_data[3] = 1.0f;  // Second sample, first class
    targets = targets_cpu.to(device);

    auto loss = cross_entropy(logits, targets, Reduction::Mean);

    // Loss should be positive
    auto loss_cpu = loss.tensor().to(Device::cpu());

    const float* loss_data = loss_cpu.data<float>();
    EXPECT_GT(loss_data[0], 0.0f) << "Failed on " << device.to_string();
}

// Regression: CrossEntropy on a >2D [N,C,H,W] input with a class-index target
// [N,H,W] (semantic segmentation) must equal the same CE computed by flattening
// to [N*H*W, C] / [N*H*W]. The previous code appended the one-hot class axis at
// the end, misaligning it with the input, and threw a broadcast error.
TEST_P(LossTest, CrossEntropySegmentation2DTarget) {
    auto in_t = arange(0, 12, 1, DType::Float32, Device::cpu()).reshape({1, 3, 2, 2});
    in_t = in_t.to(device);
    auto tgt = zeros({1, 2, 2}, DType::Int64, Device::cpu());
    int64_t* td = tgt.data<int64_t>();
    td[0] = 0; td[1] = 1; td[2] = 2; td[3] = 0;
    tgt = tgt.to(device);

    auto seg = cross_entropy(Variable(in_t, true), tgt, Reduction::Mean);

    // Reference: move C to last, flatten pixels -> [N*H*W, C] and [N*H*W].
    auto in_flat = in_t.permute({0, 2, 3, 1}).contiguous().reshape({4, 3});
    auto tgt_flat = tgt.reshape({4});
    auto ref = cross_entropy(Variable(in_flat, true), tgt_flat, Reduction::Mean);

    auto s = seg.tensor().to(Device::cpu()).to(DType::Float64).data<double>()[0];
    auto r = ref.tensor().to(Device::cpu()).to(DType::Float64).data<double>()[0];
    EXPECT_NEAR(s, r, 1e-4) << "segmentation CE != flattened CE on " << device.to_string();

    // Gradient must flow back to the input (graph not severed).
    seg.backward();
}

TEST_P(LossTest, CrossEntropyUniformLogits) {
    // For uniform logits across classes, cross entropy should be approximately log(num_classes)
    auto logits = Variable(zeros({4, 3}, DType::Float32, device), true);
    auto targets = zeros({4, 3}, DType::Float32, device);

    // One-hot encode: each sample has class 0
    auto targets_cpu = targets.to(Device::cpu());
    float* target_data = const_cast<float*>(targets_cpu.data<float>());
    for (int i = 0; i < 4; i++) {
        target_data[i * 3] = 1.0f;
    }
    targets = targets_cpu.to(device);

    auto loss = cross_entropy(logits, targets, Reduction::Mean);

    auto loss_cpu = loss.tensor().to(Device::cpu());


    const float* loss_data = loss_cpu.data<float>();
    // For uniform probabilities over 3 classes: -log(1/3) ≈ 1.099
    EXPECT_NEAR(loss_data[0], std::log(3.0f), 0.1f) << "Failed on " << device.to_string();
}

// NLL Loss Tests
TEST_P(LossTest, NLLLossBasic) {
    // Create log probabilities (already log-transformed)
    auto log_probs = Variable(full({2, 3}, std::log(0.5f), DType::Float32, device), true);
    auto targets = zeros({2, 3}, DType::Float32, device);

    auto targets_cpu = targets.to(Device::cpu());
    float* target_data = const_cast<float*>(targets_cpu.data<float>());
    target_data[0] = 1.0f;
    target_data[3] = 1.0f;
    targets = targets_cpu.to(device);

    auto loss = nll_loss(log_probs, targets, Reduction::Mean);

    auto loss_cpu = loss.tensor().to(Device::cpu());


    const float* loss_data = loss_cpu.data<float>();
    EXPECT_GT(loss_data[0], 0.0f) << "Failed on " << device.to_string();
}

// Test loss function classes (not just functional API)
TEST_P(LossTest, MSELossClass) {
    MSELoss criterion(Reduction::Mean);

    auto pred = Variable(full({2, 2}, 2.0f, DType::Float32, device), true);
    auto target = Variable(full({2, 2}, 1.0f, DType::Float32, device), false);

    auto loss = criterion(pred, target);

    auto loss_cpu = loss.tensor().to(Device::cpu());


    const float* loss_data = loss_cpu.data<float>();
    EXPECT_FLOAT_EQ(loss_data[0], 1.0f) << "Failed on " << device.to_string();
}

TEST_P(LossTest, BCELossClass) {
    BCELoss criterion(Reduction::Mean);

    auto pred = Variable(full({2, 2}, 0.8f, DType::Float32, device), true);
    auto target = Variable(ones({2, 2}, DType::Float32, device), false);

    auto loss = criterion(pred, target);

    auto loss_cpu = loss.tensor().to(Device::cpu());


    const float* loss_data = loss_cpu.data<float>();
    EXPECT_GT(loss_data[0], 0.0f) << "Failed on " << device.to_string();
}

TEST_P(LossTest, L1LossClass) {
    L1Loss criterion(Reduction::Mean);

    auto pred = Variable(full({3, 3}, 5.0f, DType::Float32, device), true);
    auto target = Variable(full({3, 3}, 2.0f, DType::Float32, device), false);

    auto loss = criterion(pred, target);

    auto loss_cpu = loss.tensor().to(Device::cpu());


    const float* loss_data = loss_cpu.data<float>();
    EXPECT_FLOAT_EQ(loss_data[0], 3.0f) << "Failed on " << device.to_string();
}

TEST_P(LossTest, CrossEntropyLossClass) {
    CrossEntropyLoss criterion(Reduction::Mean);

    auto logits = Variable(zeros({2, 3}, DType::Float32, device), true);
    auto targets = zeros({2, 3}, DType::Float32, device);

    auto targets_cpu = targets.to(Device::cpu());
    float* target_data = const_cast<float*>(targets_cpu.data<float>());
    target_data[0] = 1.0f;
    target_data[3] = 1.0f;
    targets = targets_cpu.to(device);

    auto loss = criterion(logits, targets);

    auto loss_cpu = loss.tensor().to(Device::cpu());


    const float* loss_data = loss_cpu.data<float>();
    EXPECT_GT(loss_data[0], 0.0f) << "Failed on " << device.to_string();
}

// Gradient requirement tests
TEST_P(LossTest, MSELossGradientRequired) {
    auto pred = Variable(full({2, 2}, 1.0f, DType::Float32, device), true);
    auto target = Variable(full({2, 2}, 2.0f, DType::Float32, device), false);

    auto loss = mse_loss(pred, target, Reduction::Mean);

    // Loss should require gradient since pred requires grad
    EXPECT_TRUE(loss.requires_grad()) << "Failed on " << device.to_string();
}

TEST_P(LossTest, BCELossGradientRequired) {
    auto pred = Variable(full({2, 2}, 0.6f, DType::Float32, device), true);
    auto target = Variable(ones({2, 2}, DType::Float32, device), false);

    auto loss = bce_loss(pred, target, Reduction::Mean);

    EXPECT_TRUE(loss.requires_grad()) << "Failed on " << device.to_string();
}

TEST_P(LossTest, L1LossGradientRequired) {
    auto pred = Variable(full({2, 2}, 3.0f, DType::Float32, device), true);
    auto target = Variable(full({2, 2}, 1.0f, DType::Float32, device), false);

    auto loss = l1_loss(pred, target, Reduction::Mean);

    EXPECT_TRUE(loss.requires_grad()) << "Failed on " << device.to_string();
}

INSTANTIATE_BACKEND_TESTS(LossTest);
