/**
 * @file test_distillation.cpp
 * @brief Comprehensive tests for Knowledge Distillation
 *
 * Tests focus on numerical stability fixes by Agent 9, particularly:
 * - Temperature-scaled softmax stability with extreme values
 * - KL divergence computation accuracy
 * - Teacher-student training convergence
 * - Model compression effectiveness
 */

#include <gtest/gtest.h>
#include "tenzor/tenzor.hpp"
#include "tenzor/nn/compression/distillation.hpp"
#include <cmath>
#include <limits>

using namespace tenzor;
using namespace tenzor::nn;
using namespace tenzor::nn::compression;

// =============================================================================
// Helper Functions
// =============================================================================

/**
 * @brief Check if tensor contains NaN or Inf values
 */
bool has_nan_or_inf(const Tensor& t) {
    const float* data = t.data<const float>();
    int64_t numel = t.numel();

    for (int64_t i = 0; i < numel; ++i) {
        if (std::isnan(data[i]) || std::isinf(data[i])) {
            return true;
        }
    }
    return false;
}

/**
 * @brief Naive (unstable) temperature softmax for comparison
 */
Tensor naive_temperature_softmax(const Tensor& logits, float temperature) {
    const float* input_data = logits.data<const float>();
    auto shape = logits.shape();

    Tensor output = Tensor::zeros_like(logits);
    float* output_data = output.data<float>();

    // Compute per-row softmax for (batch, features) shape
    int64_t batch_size = shape[0];
    int64_t num_features = shape[1];

    for (int64_t b = 0; b < batch_size; ++b) {
        // Compute exp(x/T) for this row - potentially unstable
        float sum = 0.0f;
        for (int64_t f = 0; f < num_features; ++f) {
            int64_t idx = b * num_features + f;
            float val = std::exp(input_data[idx] / temperature);
            output_data[idx] = val;
            sum += val;
        }

        // Normalize this row
        for (int64_t f = 0; f < num_features; ++f) {
            int64_t idx = b * num_features + f;
            output_data[idx] /= sum;
        }
    }

    return output;
}

/**
 * @brief Simple student model for testing
 */
class SimpleStudent : public Module {
public:
    SimpleStudent(int64_t input_size, int64_t hidden_size, int64_t output_size) {
        fc1_ = std::make_shared<Linear>(input_size, hidden_size);
        fc2_ = std::make_shared<Linear>(hidden_size, output_size);
        relu_ = std::make_shared<ReLU>();
        register_module("fc1", fc1_);
        register_module("fc2", fc2_);
    }

    auto forward(const Variable& x) -> Variable override {
        auto h = relu_->forward(fc1_->forward(x));
        return fc2_->forward(h);
    }

private:
    std::shared_ptr<Linear> fc1_;
    std::shared_ptr<Linear> fc2_;
    std::shared_ptr<ReLU> relu_;
};

/**
 * @brief Simple teacher model for testing (larger capacity)
 */
class SimpleTeacher : public Module {
public:
    SimpleTeacher(int64_t input_size, int64_t hidden1_size, int64_t hidden2_size, int64_t output_size) {
        fc1_ = std::make_shared<Linear>(input_size, hidden1_size);
        fc2_ = std::make_shared<Linear>(hidden1_size, hidden2_size);
        fc3_ = std::make_shared<Linear>(hidden2_size, output_size);
        relu_ = std::make_shared<ReLU>();
        register_module("fc1", fc1_);
        register_module("fc2", fc2_);
        register_module("fc3", fc3_);
    }

    auto forward(const Variable& x) -> Variable override {
        auto h1 = relu_->forward(fc1_->forward(x));
        auto h2 = relu_->forward(fc2_->forward(h1));
        return fc3_->forward(h2);
    }

private:
    std::shared_ptr<Linear> fc1_;
    std::shared_ptr<Linear> fc2_;
    std::shared_ptr<Linear> fc3_;
    std::shared_ptr<ReLU> relu_;
};

// =============================================================================
// Test Fixture
// =============================================================================

class DistillationTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create simple test data
        logits_normal_ = Tensor({2, 3}, DType::Float32, Device::cpu());
        float* data = logits_normal_.data<float>();
        data[0] = 1.0f; data[1] = 2.0f; data[2] = 3.0f;
        data[3] = -1.0f; data[4] = 0.0f; data[5] = 1.0f;

        // Create extreme value test data
        logits_extreme_ = Tensor({2, 3}, DType::Float32, Device::cpu());
        float* extreme_data = logits_extreme_.data<float>();
        extreme_data[0] = 100.0f; extreme_data[1] = 200.0f; extreme_data[2] = 300.0f;
        extreme_data[3] = -100.0f; extreme_data[4] = -200.0f; extreme_data[5] = -300.0f;

        // Very large values
        logits_very_large_ = Tensor({1, 3}, DType::Float32, Device::cpu());
        float* large_data = logits_very_large_.data<float>();
        large_data[0] = 1000.0f; large_data[1] = 2000.0f; large_data[2] = 3000.0f;
    }

    Tensor logits_normal_;
    Tensor logits_extreme_;
    Tensor logits_very_large_;
};

// =============================================================================
// Temperature Softmax Stability Tests (Agent 9's Critical Fixes)
// =============================================================================

TEST_F(DistillationTest, TemperatureSoftmaxNormalValues) {
    Variable logits(logits_normal_, true);

    auto result = temperature_softmax(logits, 1.0f, -1);

    // No NaN or Inf
    EXPECT_FALSE(has_nan_or_inf(result.tensor()));

    // Check probabilities sum to 1 for each sample
    const float* data = result.tensor().data<const float>();
    float sum1 = data[0] + data[1] + data[2];
    float sum2 = data[3] + data[4] + data[5];

    EXPECT_NEAR(sum1, 1.0f, 1e-5);
    EXPECT_NEAR(sum2, 1.0f, 1e-5);
}

TEST_F(DistillationTest, TemperatureSoftmaxExtremeValues) {
    // CRITICAL TEST: Agent 9's fix for numerical stability
    Variable logits(logits_extreme_, true);

    for (float temp : {0.01f, 0.1f, 1.0f, 5.0f, 10.0f}) {
        auto result = temperature_softmax(logits, temp, -1);

        // Must not produce NaN or Inf
        EXPECT_FALSE(has_nan_or_inf(result.tensor()))
            << "Failed for temperature: " << temp;

        // Probabilities must sum to 1
        const float* data = result.tensor().data<const float>();
        float sum1 = data[0] + data[1] + data[2];
        float sum2 = data[3] + data[4] + data[5];

        EXPECT_NEAR(sum1, 1.0f, 1e-4)
            << "Sample 1 sum failed for temperature: " << temp;
        EXPECT_NEAR(sum2, 1.0f, 1e-4)
            << "Sample 2 sum failed for temperature: " << temp;

        // All probabilities should be in [0, 1]
        for (int64_t i = 0; i < 6; ++i) {
            EXPECT_GE(data[i], 0.0f) << "Negative probability at index " << i;
            EXPECT_LE(data[i], 1.0f) << "Probability > 1 at index " << i;
        }
    }
}

TEST_F(DistillationTest, TemperatureSoftmaxVeryLargeValues) {
    // Test with values that would overflow naive exp(x/T)
    Variable logits(logits_very_large_, true);

    for (float temp : {0.01f, 0.1f, 1.0f, 10.0f}) {
        auto result = temperature_softmax(logits, temp, -1);

        // Must not overflow
        EXPECT_FALSE(has_nan_or_inf(result.tensor()))
            << "Overflow with temperature: " << temp;

        // Must sum to 1
        const float* data = result.tensor().data<const float>();
        float sum = data[0] + data[1] + data[2];
        EXPECT_NEAR(sum, 1.0f, 1e-4);
    }
}

TEST_F(DistillationTest, TemperatureSoftmaxStabilityVsNaive) {
    // Compare stable implementation with naive for normal values
    Variable logits(logits_normal_, true);

    auto stable_result = temperature_softmax(logits, 2.0f, -1);
    auto naive_result = naive_temperature_softmax(logits_normal_, 2.0f);

    // Should match for normal values
    const float* stable_data = stable_result.tensor().data<const float>();
    const float* naive_data = naive_result.data<const float>();

    for (int64_t i = 0; i < 6; ++i) {
        EXPECT_NEAR(stable_data[i], naive_data[i], 1e-5)
            << "Mismatch at index " << i;
    }

    // Now test with extreme values - naive should fail
    auto stable_extreme = temperature_softmax(Variable(logits_extreme_, true), 0.1f, -1);
    auto naive_extreme = naive_temperature_softmax(logits_extreme_, 0.1f);

    // Stable version should work
    EXPECT_FALSE(has_nan_or_inf(stable_extreme.tensor()));

    // Naive version likely has Inf/NaN (we don't assert this, just document the difference)
}

TEST_F(DistillationTest, TemperatureSoftmaxSmallTemperature) {
    // Very small temperature should approach one-hot
    Variable logits(logits_normal_, true);
    auto result = temperature_softmax(logits, 0.01f, -1);

    EXPECT_FALSE(has_nan_or_inf(result.tensor()));

    const float* data = result.tensor().data<const float>();

    // First sample: largest logit is index 2 (value 3.0)
    EXPECT_NEAR(data[2], 1.0f, 0.01f);  // Should be close to 1
    EXPECT_LT(data[0], 0.01f);  // Others should be close to 0
    EXPECT_LT(data[1], 0.01f);

    // Second sample: largest logit is index 5 (value 1.0)
    EXPECT_NEAR(data[5], 1.0f, 0.01f);
    EXPECT_LT(data[3], 0.01f);
    EXPECT_LT(data[4], 0.01f);
}

TEST_F(DistillationTest, TemperatureSoftmaxLargeTemperature) {
    // Large temperature should approach uniform distribution
    Variable logits(logits_normal_, true);
    auto result = temperature_softmax(logits, 100.0f, -1);

    EXPECT_FALSE(has_nan_or_inf(result.tensor()));

    const float* data = result.tensor().data<const float>();

    // Each sample should have roughly uniform probabilities
    float expected = 1.0f / 3.0f;
    for (int i = 0; i < 3; ++i) {
        EXPECT_NEAR(data[i], expected, 0.05f);
        EXPECT_NEAR(data[i + 3], expected, 0.05f);
    }
}

// =============================================================================
// Temperature Log-Softmax Tests
// =============================================================================

TEST_F(DistillationTest, TemperatureLogSoftmaxStability) {
    Variable logits(logits_extreme_, true);

    for (float temp : {0.01f, 0.1f, 1.0f, 10.0f}) {
        auto result = temperature_log_softmax(logits, temp, -1);

        // Must not produce NaN or Inf
        EXPECT_FALSE(has_nan_or_inf(result.tensor()))
            << "Failed for temperature: " << temp;

        // Log probabilities should be negative
        const float* data = result.tensor().data<const float>();
        for (int64_t i = 0; i < 6; ++i) {
            EXPECT_LE(data[i], 0.0f) << "Log prob should be <= 0 at index " << i;
        }
    }
}

TEST_F(DistillationTest, TemperatureLogSoftmaxVsSoftmax) {
    // log_softmax should match log(softmax)
    Variable logits(logits_normal_, true);

    auto log_result = temperature_log_softmax(logits, 2.0f, -1);
    auto softmax_result = temperature_softmax(logits, 2.0f, -1);

    // Manually compute log of softmax
    const float* softmax_data = softmax_result.tensor().data<const float>();
    const float* log_data = log_result.tensor().data<const float>();

    for (int64_t i = 0; i < 6; ++i) {
        float expected_log = std::log(softmax_data[i]);
        EXPECT_NEAR(log_data[i], expected_log, 1e-5)
            << "Mismatch at index " << i;
    }
}

// =============================================================================
// KL Divergence Tests
// =============================================================================

TEST_F(DistillationTest, KLDivergenceBasic) {
    // Create simple probability distributions
    Tensor p_tensor({2, 3}, DType::Float32, Device::cpu());
    Tensor log_q_tensor({2, 3}, DType::Float32, Device::cpu());

    float* p_data = p_tensor.data<float>();
    float* log_q_data = log_q_tensor.data<float>();

    // P = [0.5, 0.3, 0.2]
    p_data[0] = 0.5f; p_data[1] = 0.3f; p_data[2] = 0.2f;
    // Q = [0.4, 0.4, 0.2], log Q = [log(0.4), log(0.4), log(0.2)]
    log_q_data[0] = std::log(0.4f);
    log_q_data[1] = std::log(0.4f);
    log_q_data[2] = std::log(0.2f);

    // Second sample (same for simplicity)
    p_data[3] = 0.5f; p_data[4] = 0.3f; p_data[5] = 0.2f;
    log_q_data[3] = std::log(0.4f);
    log_q_data[4] = std::log(0.4f);
    log_q_data[5] = std::log(0.2f);

    Variable p(p_tensor, false);
    Variable log_q(log_q_tensor, false);

    auto kl = kl_divergence(log_q, p, "sum");

    EXPECT_FALSE(has_nan_or_inf(kl.tensor()));

    // KL divergence sum should be non-negative (individual elements can be negative)
    float kl_sum = kl.tensor().data<const float>()[0];
    EXPECT_GE(kl_sum, 0.0f) << "Total KL divergence should be non-negative";

    // Manually verify: KL(P||Q) = sum(P * (log P - log Q))
    // P = [0.5, 0.3, 0.2], Q = [0.4, 0.4, 0.2]
    // KL = 0.5*log(0.5/0.4) + 0.3*log(0.3/0.4) + 0.2*log(0.2/0.2)
    //    = 0.5*log(1.25) + 0.3*log(0.75) + 0.2*0
    //    ≈ 0.1116 - 0.0863 + 0 = 0.0253 > 0
    EXPECT_GT(kl_sum, 0.0f) << "KL(P||Q) should be positive when P != Q";
}

TEST_F(DistillationTest, KLDivergenceIdenticalDistributions) {
    // KL(P||P) should be 0
    Tensor p_tensor({1, 3}, DType::Float32, Device::cpu());
    float* data = p_tensor.data<float>();
    data[0] = 0.5f; data[1] = 0.3f; data[2] = 0.2f;

    Tensor log_p_tensor({1, 3}, DType::Float32, Device::cpu());
    float* log_data = log_p_tensor.data<float>();
    log_data[0] = std::log(0.5f);
    log_data[1] = std::log(0.3f);
    log_data[2] = std::log(0.2f);

    Variable p(p_tensor, false);
    Variable log_p(log_p_tensor, false);

    auto kl = kl_divergence(log_p, p, "batchmean");

    // Should be very close to 0
    float kl_value = kl.tensor().data<const float>()[0];
    EXPECT_NEAR(kl_value, 0.0f, 1e-5);
}

// =============================================================================
// Distillation Loss Tests
// =============================================================================

TEST_F(DistillationTest, DistillationLossAlphaBlending) {
    // Test that alpha correctly balances soft and hard losses
    Tensor student_logits({2, 3}, DType::Float32, Device::cpu());
    Tensor teacher_logits({2, 3}, DType::Float32, Device::cpu());
    Tensor targets({2}, DType::Int64, Device::cpu());

    // Initialize with some values
    float* s_data = student_logits.data<float>();
    float* t_data = teacher_logits.data<float>();
    int64_t* target_data = targets.data<int64_t>();

    for (int i = 0; i < 6; ++i) {
        s_data[i] = static_cast<float>(i) * 0.5f;
        t_data[i] = static_cast<float>(i) * 0.7f;
    }
    target_data[0] = 1;
    target_data[1] = 2;

    Variable student(student_logits, true);
    Variable teacher(teacher_logits, false);

    // Test alpha = 0.0 (only hard targets)
    DistillationConfig config_hard;
    config_hard.alpha = 0.0f;
    config_hard.use_hard_targets = true;
    config_hard.temperature = 3.0f;

    auto loss_hard = distillation_loss(student, teacher, targets, config_hard);
    EXPECT_FALSE(has_nan_or_inf(loss_hard.tensor()));

    // Test alpha = 1.0 (only soft targets)
    DistillationConfig config_soft;
    config_soft.alpha = 1.0f;
    config_soft.use_hard_targets = false;
    config_soft.temperature = 3.0f;

    auto loss_soft = distillation_loss(student, teacher, std::nullopt, config_soft);
    EXPECT_FALSE(has_nan_or_inf(loss_soft.tensor()));

    // Test alpha = 0.5 (balanced)
    DistillationConfig config_balanced;
    config_balanced.alpha = 0.5f;
    config_balanced.use_hard_targets = true;
    config_balanced.temperature = 3.0f;

    auto loss_balanced = distillation_loss(student, teacher, targets, config_balanced);
    EXPECT_FALSE(has_nan_or_inf(loss_balanced.tensor()));
}

TEST_F(DistillationTest, DistillationLossTemperatureScaling) {
    Tensor student_logits({1, 3}, DType::Float32, Device::cpu());
    Tensor teacher_logits({1, 3}, DType::Float32, Device::cpu());

    float* s_data = student_logits.data<float>();
    float* t_data = teacher_logits.data<float>();

    s_data[0] = 1.0f; s_data[1] = 2.0f; s_data[2] = 3.0f;
    t_data[0] = 1.5f; t_data[1] = 2.5f; t_data[2] = 3.5f;

    Variable student(student_logits, true);
    Variable teacher(teacher_logits, false);

    DistillationConfig config;
    config.alpha = 1.0f;
    config.use_hard_targets = false;
    config.normalize_temperature = true;

    // Test different temperatures
    for (float temp : {1.0f, 3.0f, 5.0f, 10.0f}) {
        config.temperature = temp;
        auto loss = distillation_loss(student, teacher, std::nullopt, config);

        EXPECT_FALSE(has_nan_or_inf(loss.tensor()))
            << "Loss has NaN/Inf for temperature: " << temp;

        float loss_val = loss.tensor().data<const float>()[0];
        EXPECT_GE(loss_val, 0.0f)
            << "Loss should be non-negative for temperature: " << temp;
    }
}

TEST_F(DistillationTest, DistillationLossNormalization) {
    Tensor student_logits({1, 3}, DType::Float32, Device::cpu());
    Tensor teacher_logits({1, 3}, DType::Float32, Device::cpu());

    float* s_data = student_logits.data<float>();
    float* t_data = teacher_logits.data<float>();

    s_data[0] = 1.0f; s_data[1] = 2.0f; s_data[2] = 3.0f;
    t_data[0] = 1.0f; t_data[1] = 2.0f; t_data[2] = 3.0f;

    Variable student(student_logits, true);
    Variable teacher(teacher_logits, false);

    DistillationConfig config_normalized;
    config_normalized.temperature = 5.0f;
    config_normalized.alpha = 1.0f;
    config_normalized.use_hard_targets = false;
    config_normalized.normalize_temperature = true;

    auto loss_normalized = distillation_loss(student, teacher, std::nullopt, config_normalized);

    DistillationConfig config_unnormalized;
    config_unnormalized.temperature = 5.0f;
    config_unnormalized.alpha = 1.0f;
    config_unnormalized.use_hard_targets = false;
    config_unnormalized.normalize_temperature = false;

    auto loss_unnormalized = distillation_loss(student, teacher, std::nullopt, config_unnormalized);

    // Normalized loss should be scaled by T^2
    float norm_val = loss_normalized.tensor().data<const float>()[0];
    float unnorm_val = loss_unnormalized.tensor().data<const float>()[0];

    EXPECT_NEAR(norm_val, unnorm_val * 25.0f, 0.1f);  // 5^2 = 25
}

// =============================================================================
// KnowledgeDistillation Class Tests
// =============================================================================

TEST_F(DistillationTest, KnowledgeDistillationConstruction) {
    auto teacher = std::make_shared<SimpleTeacher>(10, 20, 15, 5);
    auto student = std::make_shared<SimpleStudent>(10, 8, 5);

    DistillationConfig config;
    config.temperature = 4.0f;
    config.alpha = 0.8f;

    auto distiller = KnowledgeDistillation(teacher, student, config);

    EXPECT_EQ(distiller.teacher(), teacher);
    EXPECT_EQ(distiller.student(), student);
    EXPECT_FLOAT_EQ(distiller.config().temperature, 4.0f);
    EXPECT_FLOAT_EQ(distiller.config().alpha, 0.8f);
}

TEST_F(DistillationTest, KnowledgeDistillationForwardPass) {
    auto teacher = std::make_shared<SimpleTeacher>(10, 20, 15, 5);
    auto student = std::make_shared<SimpleStudent>(10, 8, 5);

    auto distiller = KnowledgeDistillation(teacher, student);

    Tensor input({2, 10}, DType::Float32, Device::cpu());
    input.fill_(0.5f);
    Variable input_var(input, true);

    auto [student_out, teacher_out] = distiller.forward(input_var);

    EXPECT_EQ(student_out.tensor().shape()[0], 2);
    EXPECT_EQ(student_out.tensor().shape()[1], 5);
    EXPECT_EQ(teacher_out.tensor().shape()[0], 2);
    EXPECT_EQ(teacher_out.tensor().shape()[1], 5);

    EXPECT_FALSE(has_nan_or_inf(student_out.tensor()));
    EXPECT_FALSE(has_nan_or_inf(teacher_out.tensor()));
}

TEST_F(DistillationTest, KnowledgeDistillationComputeLoss) {
    auto teacher = std::make_shared<SimpleTeacher>(10, 20, 15, 5);
    auto student = std::make_shared<SimpleStudent>(10, 8, 5);

    DistillationConfig config;
    config.temperature = 3.0f;
    config.alpha = 0.7f;

    auto distiller = KnowledgeDistillation(teacher, student, config);

    Tensor input({2, 10}, DType::Float32, Device::cpu());
    input.fill_(0.5f);
    Variable input_var(input, true);

    Tensor targets({2}, DType::Int64, Device::cpu());
    targets.data<int64_t>()[0] = 1;
    targets.data<int64_t>()[1] = 3;

    auto loss = distiller.compute_loss(input_var, targets);

    EXPECT_FALSE(has_nan_or_inf(loss.tensor()));

    float loss_val = loss.tensor().data<const float>()[0];
    EXPECT_GE(loss_val, 0.0f);
}

TEST_F(DistillationTest, KnowledgeDistillationConfigUpdate) {
    auto teacher = std::make_shared<SimpleTeacher>(10, 20, 15, 5);
    auto student = std::make_shared<SimpleStudent>(10, 8, 5);

    auto distiller = KnowledgeDistillation(teacher, student);

    // Update temperature
    distiller.set_temperature(6.0f);
    EXPECT_FLOAT_EQ(distiller.config().temperature, 6.0f);

    // Update alpha
    distiller.set_alpha(0.9f);
    EXPECT_FLOAT_EQ(distiller.config().alpha, 0.9f);

    // Update entire config
    DistillationConfig new_config;
    new_config.temperature = 2.5f;
    new_config.alpha = 0.6f;
    new_config.use_hard_targets = false;

    distiller.set_config(new_config);
    EXPECT_FLOAT_EQ(distiller.config().temperature, 2.5f);
    EXPECT_FLOAT_EQ(distiller.config().alpha, 0.6f);
    EXPECT_FALSE(distiller.config().use_hard_targets);
}

// =============================================================================
// Temperature Schedule Tests
// =============================================================================

TEST_F(DistillationTest, TemperatureScheduleLinear) {
    float initial = 10.0f;
    float final = 2.0f;
    int total_epochs = 100;

    // At epoch 0, should be initial
    float temp0 = temperature_schedule(initial, final, 0, total_epochs, "linear");
    EXPECT_FLOAT_EQ(temp0, initial);

    // At epoch 50, should be halfway
    float temp50 = temperature_schedule(initial, final, 50, total_epochs, "linear");
    EXPECT_NEAR(temp50, 6.0f, 0.1f);

    // At final epoch, should be final
    float temp100 = temperature_schedule(initial, final, 100, total_epochs, "linear");
    EXPECT_FLOAT_EQ(temp100, final);
}

TEST_F(DistillationTest, TemperatureScheduleExponential) {
    float initial = 10.0f;
    float final = 1.0f;
    int total_epochs = 100;

    float temp0 = temperature_schedule(initial, final, 0, total_epochs, "exponential");
    EXPECT_FLOAT_EQ(temp0, initial);

    float temp100 = temperature_schedule(initial, final, 100, total_epochs, "exponential");
    EXPECT_NEAR(temp100, final, 0.1f);

    // Exponential should decay faster initially
    float temp25 = temperature_schedule(initial, final, 25, total_epochs, "exponential");
    float linear_temp25 = temperature_schedule(initial, final, 25, total_epochs, "linear");
    EXPECT_LT(temp25, linear_temp25);
}

TEST_F(DistillationTest, TemperatureScheduleCosine) {
    float initial = 10.0f;
    float final = 2.0f;
    int total_epochs = 100;

    float temp0 = temperature_schedule(initial, final, 0, total_epochs, "cosine");
    EXPECT_NEAR(temp0, initial, 0.1f);

    float temp100 = temperature_schedule(initial, final, 100, total_epochs, "cosine");
    EXPECT_NEAR(temp100, final, 0.1f);
}

// =============================================================================
// Configuration Presets Tests
// =============================================================================

TEST_F(DistillationTest, ClassificationConfig) {
    auto config = make_classification_distillation_config();

    EXPECT_FLOAT_EQ(config.temperature, 3.0f);
    EXPECT_FLOAT_EQ(config.alpha, 0.7f);
    EXPECT_TRUE(config.use_hard_targets);
    EXPECT_TRUE(config.normalize_temperature);
}

TEST_F(DistillationTest, DetectionConfig) {
    auto config = make_detection_distillation_config();

    EXPECT_FLOAT_EQ(config.temperature, 2.0f);
    EXPECT_FLOAT_EQ(config.alpha, 0.5f);
    EXPECT_TRUE(config.use_hard_targets);
    EXPECT_TRUE(config.normalize_temperature);
}

TEST_F(DistillationTest, SegmentationConfig) {
    auto config = make_segmentation_distillation_config();

    EXPECT_FLOAT_EQ(config.temperature, 1.5f);
    EXPECT_FLOAT_EQ(config.alpha, 0.8f);
    EXPECT_TRUE(config.use_hard_targets);
    EXPECT_FALSE(config.normalize_temperature);  // Different for dense predictions
}

// =============================================================================
// Compression Ratio Tests
// =============================================================================

TEST_F(DistillationTest, CompressionRatio) {
    auto teacher = std::make_shared<SimpleTeacher>(10, 20, 15, 5);
    auto student = std::make_shared<SimpleStudent>(10, 8, 5);

    float ratio = compute_distillation_compression_ratio(teacher, student);

    // Teacher has more parameters than student
    EXPECT_GT(ratio, 1.0f);

    // Ratio should be reasonable (not infinity or too small)
    EXPECT_LT(ratio, 100.0f);
}

// =============================================================================
// Edge Cases and Error Handling
// =============================================================================

TEST_F(DistillationTest, InvalidTemperature) {
    Variable logits(logits_normal_, true);

    // Temperature must be positive
    EXPECT_THROW(temperature_softmax(logits, 0.0f, -1), std::runtime_error);
    EXPECT_THROW(temperature_softmax(logits, -1.0f, -1), std::runtime_error);
    EXPECT_THROW(temperature_log_softmax(logits, 0.0f, -1), std::runtime_error);
    EXPECT_THROW(temperature_log_softmax(logits, -1.0f, -1), std::runtime_error);
}

TEST_F(DistillationTest, EmptyTensorHandling) {
    // Test with empty tensors (should not crash)
    Tensor empty({0, 3}, DType::Float32, Device::cpu());
    Variable empty_var(empty, false);

    // These should handle empty input gracefully
    // (actual behavior depends on implementation)
}

TEST_F(DistillationTest, SingleClassProblem) {
    // Test with single class (edge case)
    Tensor single_class({2, 1}, DType::Float32, Device::cpu());
    single_class.fill_(1.0f);

    Variable logits(single_class, true);
    auto result = temperature_softmax(logits, 1.0f, -1);

    EXPECT_FALSE(has_nan_or_inf(result.tensor()));

    // Single class should have probability 1.0
    const float* data = result.tensor().data<const float>();
    EXPECT_NEAR(data[0], 1.0f, 1e-5);
    EXPECT_NEAR(data[1], 1.0f, 1e-5);
}

// =============================================================================
// Numerical Stability Stress Tests
// =============================================================================

TEST_F(DistillationTest, NumericalStabilityExtremeTemperatures) {
    Variable logits(logits_extreme_, true);

    // Very small temperatures (approaching hard targets)
    for (float temp : {0.001f, 0.01f, 0.05f}) {
        auto result = temperature_softmax(logits, temp, -1);
        EXPECT_FALSE(has_nan_or_inf(result.tensor()))
            << "Failed with small temperature: " << temp;
    }

    // Very large temperatures (approaching uniform)
    for (float temp : {50.0f, 100.0f, 1000.0f}) {
        auto result = temperature_softmax(logits, temp, -1);
        EXPECT_FALSE(has_nan_or_inf(result.tensor()))
            << "Failed with large temperature: " << temp;
    }
}

TEST_F(DistillationTest, NumericalStabilityMixedRanges) {
    // Test with mixed positive and negative extreme values
    Tensor mixed({1, 4}, DType::Float32, Device::cpu());
    float* data = mixed.data<float>();
    data[0] = -500.0f;
    data[1] = 0.0f;
    data[2] = 500.0f;
    data[3] = 1000.0f;

    Variable logits(mixed, true);

    for (float temp : {0.1f, 1.0f, 10.0f}) {
        auto result = temperature_softmax(logits, temp, -1);

        EXPECT_FALSE(has_nan_or_inf(result.tensor()));

        const float* result_data = result.tensor().data<const float>();
        float sum = result_data[0] + result_data[1] + result_data[2] + result_data[3];
        EXPECT_NEAR(sum, 1.0f, 1e-4);
    }
}

TEST_F(DistillationTest, NumericalStabilityAllNegative) {
    // Test with all negative values
    Tensor all_negative({1, 3}, DType::Float32, Device::cpu());
    float* data = all_negative.data<float>();
    data[0] = -100.0f;
    data[1] = -200.0f;
    data[2] = -300.0f;

    Variable logits(all_negative, true);
    auto result = temperature_softmax(logits, 1.0f, -1);

    EXPECT_FALSE(has_nan_or_inf(result.tensor()));

    const float* result_data = result.tensor().data<const float>();
    float sum = result_data[0] + result_data[1] + result_data[2];
    EXPECT_NEAR(sum, 1.0f, 1e-5);
}

TEST_F(DistillationTest, NumericalStabilityAllZeros) {
    // Test with all zeros (should give uniform distribution)
    Tensor all_zeros({1, 3}, DType::Float32, Device::cpu());
    all_zeros.fill_(0.0f);

    Variable logits(all_zeros, true);
    auto result = temperature_softmax(logits, 1.0f, -1);

    EXPECT_FALSE(has_nan_or_inf(result.tensor()));

    const float* data = result.tensor().data<const float>();
    float expected = 1.0f / 3.0f;
    for (int i = 0; i < 3; ++i) {
        EXPECT_NEAR(data[i], expected, 1e-5);
    }
}

// =============================================================================
// Main
// =============================================================================

int main(int argc, char** argv) {
    // Initialize Tenzor backend
    tenzor::initialize();

    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
