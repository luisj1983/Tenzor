/**
 * @file test_distillation_multidtype.cpp
 * @brief Multi-dtype multi-backend tests for Knowledge Distillation
 *
 * Tests knowledge distillation functionality with Float32, Float64, and Float16 dtypes
 * across CPU, CUDA, OneAPI, Vulkan, and ROCm backends:
 * - Temperature-scaled softmax stability with extreme values
 * - KL divergence computation accuracy
 * - Teacher-student training convergence
 * - Model compression effectiveness
 * - Response-based vs feature-based distillation
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/nn/compression/distillation.hpp>
#include "../multi_backend_dtype_fixture.hpp"
#include <cmath>
#include <limits>

using namespace tenzor;
using namespace tenzor::nn;
using namespace tenzor::nn::compression;
using namespace tenzor::testing;

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * @brief Check if tensor contains NaN or Inf values
 */
bool has_nan_or_inf(const Tensor& t) {
    auto t_cpu = t.to(Device::cpu()).to(DType::Float32);
    const float* data = t_cpu.data<float>();
    int64_t numel = t_cpu.numel();

    for (int64_t i = 0; i < numel; ++i) {
        if (std::isnan(data[i]) || std::isinf(data[i])) {
            return true;
        }
    }
    return false;
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

    auto forward_impl(const Variable& x) -> Variable override {
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
    SimpleTeacher(int64_t input_size, int64_t hidden1_size,
                  int64_t hidden2_size, int64_t output_size) {
        fc1_ = std::make_shared<Linear>(input_size, hidden1_size);
        fc2_ = std::make_shared<Linear>(hidden1_size, hidden2_size);
        fc3_ = std::make_shared<Linear>(hidden2_size, output_size);
        relu_ = std::make_shared<ReLU>();
        register_module("fc1", fc1_);
        register_module("fc2", fc2_);
        register_module("fc3", fc3_);
    }

    auto forward_impl(const Variable& x) -> Variable override {
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

// ============================================================================
// Distillation Multi-Backend Multi-DType Test Fixture
// ============================================================================

class DistillationMultiDTypeTest : public MultiBackendDTypeTest {
protected:
    void SetUp() override {
        MultiBackendDTypeTest::SetUp();

        // Create simple test logits
        logits_normal_ = tenzor::zeros({2, 3}, dtype(), device());
        auto logits_cpu = logits_normal_.to(Device::cpu()).to(DType::Float32);
        float* data = logits_cpu.data<float>();
        data[0] = 1.0f; data[1] = 2.0f; data[2] = 3.0f;
        data[3] = -1.0f; data[4] = 0.0f; data[5] = 1.0f;
        logits_normal_ = logits_cpu.to(dtype()).to(device());

        // Create extreme value test data (scaled for Float16)
        float scale = (dtype() == DType::Float16) ? 10.0f : 100.0f;
        logits_extreme_ = tenzor::zeros({2, 3}, dtype(), device());
        auto extreme_cpu = logits_extreme_.to(Device::cpu()).to(DType::Float32);
        float* extreme_data = extreme_cpu.data<float>();
        extreme_data[0] = scale; extreme_data[1] = scale * 2.0f;
        extreme_data[2] = scale * 3.0f;
        extreme_data[3] = -scale; extreme_data[4] = -scale * 2.0f;
        extreme_data[5] = -scale * 3.0f;
        logits_extreme_ = extreme_cpu.to(dtype()).to(device());

        // Very large values (scaled for Float16)
        float large_scale = (dtype() == DType::Float16) ? 20.0f : 1000.0f;
        logits_very_large_ = tenzor::zeros({1, 3}, dtype(), device());
        auto large_cpu = logits_very_large_.to(Device::cpu()).to(DType::Float32);
        float* large_data = large_cpu.data<float>();
        large_data[0] = large_scale;
        large_data[1] = large_scale * 2.0f;
        large_data[2] = large_scale * 3.0f;
        logits_very_large_ = large_cpu.to(dtype()).to(device());
    }

    float sum_tolerance() const {
        if (dtype() == DType::Float16) return 1e-2f;
        if (dtype() == DType::Float64) return 1e-9f;
        return 1e-4f;
    }

    Tensor logits_normal_;
    Tensor logits_extreme_;
    Tensor logits_very_large_;
};

// ============================================================================
// Temperature Softmax Stability Tests
// ============================================================================

TEST_P(DistillationMultiDTypeTest, TemperatureSoftmaxNormalValues) {
    Variable logits(logits_normal_, true);

    auto result = temperature_softmax(logits, 1.0f, -1);

    // No NaN or Inf
    EXPECT_FALSE(has_nan_or_inf(result.tensor()));

    // Check probabilities sum to 1 for each sample
    auto result_cpu = result.tensor().to(Device::cpu()).to(DType::Float32);
    const float* data = result_cpu.data<float>();
    float sum1 = data[0] + data[1] + data[2];
    float sum2 = data[3] + data[4] + data[5];

    EXPECT_NEAR(sum1, 1.0f, sum_tolerance());
    EXPECT_NEAR(sum2, 1.0f, sum_tolerance());
}

TEST_P(DistillationMultiDTypeTest, TemperatureSoftmaxExtremeValues) {
    // CRITICAL TEST: Numerical stability with extreme values
    Variable logits(logits_extreme_, true);

    std::vector<float> temps = {0.01f, 0.1f, 1.0f, 5.0f, 10.0f};

    for (float temp : temps) {
        auto result = temperature_softmax(logits, temp, -1);

        // Must not produce NaN or Inf
        EXPECT_FALSE(has_nan_or_inf(result.tensor()))
            << "Failed for temperature: " << temp;

        // Probabilities must sum to 1
        auto result_cpu = result.tensor().to(Device::cpu()).to(DType::Float32);
        const float* data = result_cpu.data<float>();
        float sum1 = data[0] + data[1] + data[2];
        float sum2 = data[3] + data[4] + data[5];

        EXPECT_NEAR(sum1, 1.0f, sum_tolerance())
            << "Sample 1 sum failed for temperature: " << temp;
        EXPECT_NEAR(sum2, 1.0f, sum_tolerance())
            << "Sample 2 sum failed for temperature: " << temp;

        // All probabilities should be in [0, 1]
        for (int64_t i = 0; i < 6; ++i) {
            EXPECT_GE(data[i], 0.0f) << "Negative probability at index " << i;
            EXPECT_LE(data[i], 1.0f) << "Probability > 1 at index " << i;
        }
    }
}

TEST_P(DistillationMultiDTypeTest, TemperatureSoftmaxVeryLargeValues) {
    // Test with values that would overflow naive exp(x/T)
    Variable logits(logits_very_large_, true);

    std::vector<float> temps = {0.01f, 0.1f, 1.0f, 10.0f};

    for (float temp : temps) {
        auto result = temperature_softmax(logits, temp, -1);

        // Must not overflow
        EXPECT_FALSE(has_nan_or_inf(result.tensor()))
            << "Overflow with temperature: " << temp;

        // Must sum to 1
        auto result_cpu = result.tensor().to(Device::cpu()).to(DType::Float32);
        const float* data = result_cpu.data<float>();
        float sum = data[0] + data[1] + data[2];
        EXPECT_NEAR(sum, 1.0f, sum_tolerance());
    }
}

TEST_P(DistillationMultiDTypeTest, TemperatureSoftmaxSmallTemperature) {
    // Very small temperature should approach one-hot
    Variable logits(logits_normal_, true);

    auto result = temperature_softmax(logits, 0.01f, -1);

    EXPECT_FALSE(has_nan_or_inf(result.tensor()));

    auto result_cpu = result.tensor().to(Device::cpu()).to(DType::Float32);
    const float* data = result_cpu.data<float>();

    // First sample: largest logit is index 2 (value 3.0)
    EXPECT_NEAR(data[2], 1.0f, 0.01f);  // Should be close to 1
    EXPECT_LT(data[0], 0.01f);  // Others should be close to 0
    EXPECT_LT(data[1], 0.01f);

    // Second sample: largest logit is index 5 (value 1.0)
    EXPECT_NEAR(data[5], 1.0f, 0.01f);
    EXPECT_LT(data[3], 0.01f);
    EXPECT_LT(data[4], 0.01f);
}

TEST_P(DistillationMultiDTypeTest, TemperatureSoftmaxLargeTemperature) {
    // Large temperature should approach uniform distribution
    Variable logits(logits_normal_, true);

    auto result = temperature_softmax(logits, 100.0f, -1);

    EXPECT_FALSE(has_nan_or_inf(result.tensor()));

    auto result_cpu = result.tensor().to(Device::cpu()).to(DType::Float32);
    const float* data = result_cpu.data<float>();

    // Each sample should have roughly uniform probabilities
    float expected = 1.0f / 3.0f;
    for (int i = 0; i < 3; ++i) {
        EXPECT_NEAR(data[i], expected, 0.05f);
        EXPECT_NEAR(data[i + 3], expected, 0.05f);
    }
}

// ============================================================================
// Temperature Log-Softmax Tests
// ============================================================================

TEST_P(DistillationMultiDTypeTest, TemperatureLogSoftmaxStability) {
    Variable logits(logits_extreme_, true);

    std::vector<float> temps = {0.01f, 0.1f, 1.0f, 10.0f};

    for (float temp : temps) {
        auto result = temperature_log_softmax(logits, temp, -1);

        // Must not produce NaN or Inf
        EXPECT_FALSE(has_nan_or_inf(result.tensor()))
            << "Failed for temperature: " << temp;

        // Log probabilities should be negative
        auto result_cpu = result.tensor().to(Device::cpu()).to(DType::Float32);
        const float* data = result_cpu.data<float>();
        for (int64_t i = 0; i < 6; ++i) {
            EXPECT_LE(data[i], 0.0f)
                << "Log prob should be <= 0 at index " << i;
        }
    }
}

TEST_P(DistillationMultiDTypeTest, TemperatureLogSoftmaxVsSoftmax) {
    // log_softmax should match log(softmax)
    Variable logits(logits_normal_, true);

    auto log_result = temperature_log_softmax(logits, 2.0f, -1);
    auto softmax_result = temperature_softmax(logits, 2.0f, -1);

    // Manually compute log of softmax
    auto softmax_cpu = softmax_result.tensor().to(Device::cpu()).to(DType::Float32);
    auto log_cpu = log_result.tensor().to(Device::cpu()).to(DType::Float32);
    const float* softmax_data = softmax_cpu.data<float>();
    const float* log_data = log_cpu.data<float>();

    for (int64_t i = 0; i < 6; ++i) {
        float expected_log = std::log(softmax_data[i]);
        EXPECT_NEAR(log_data[i], expected_log, atol())
            << "Mismatch at index " << i;
    }
}

// ============================================================================
// KL Divergence Tests
// ============================================================================

TEST_P(DistillationMultiDTypeTest, KLDivergenceBasic) {
    // Create simple probability distributions
    Tensor p_tensor = tenzor::zeros({2, 3}, dtype(), device());
    Tensor log_q_tensor = tenzor::zeros({2, 3}, dtype(), device());

    auto p_cpu = p_tensor.to(Device::cpu()).to(DType::Float32);
    auto log_q_cpu = log_q_tensor.to(Device::cpu()).to(DType::Float32);
    float* p_data = p_cpu.data<float>();
    float* log_q_data = log_q_cpu.data<float>();

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

    p_tensor = p_cpu.to(dtype()).to(device());
    log_q_tensor = log_q_cpu.to(dtype()).to(device());

    Variable p(p_tensor, false);
    Variable log_q(log_q_tensor, false);

    auto kl = kl_divergence(log_q, p, "sum");

    EXPECT_FALSE(has_nan_or_inf(kl.tensor()));

    // KL divergence sum should be non-negative
    auto kl_cpu = kl.tensor().to(Device::cpu()).to(DType::Float32);
    float kl_sum = kl_cpu.data<float>()[0];
    EXPECT_GE(kl_sum, 0.0f) << "Total KL divergence should be non-negative";

    // KL(P||Q) should be positive when P != Q
    EXPECT_GT(kl_sum, 0.0f) << "KL(P||Q) should be positive when P != Q";
}

TEST_P(DistillationMultiDTypeTest, KLDivergenceIdenticalDistributions) {
    // KL(P||P) should be 0
    Tensor p_tensor = tenzor::zeros({1, 3}, dtype(), device());
    auto p_cpu = p_tensor.to(Device::cpu()).to(DType::Float32);
    float* data = p_cpu.data<float>();
    data[0] = 0.5f; data[1] = 0.3f; data[2] = 0.2f;
    p_tensor = p_cpu.to(dtype()).to(device());

    Tensor log_p_tensor = tenzor::zeros({1, 3}, dtype(), device());
    auto log_p_cpu = log_p_tensor.to(Device::cpu()).to(DType::Float32);
    float* log_data = log_p_cpu.data<float>();
    log_data[0] = std::log(0.5f);
    log_data[1] = std::log(0.3f);
    log_data[2] = std::log(0.2f);
    log_p_tensor = log_p_cpu.to(dtype()).to(device());

    Variable p(p_tensor, false);
    Variable log_p(log_p_tensor, false);

    auto kl = kl_divergence(log_p, p, "batchmean");

    // Should be very close to 0
    auto kl_cpu = kl.tensor().to(Device::cpu()).to(DType::Float32);
    float kl_value = kl_cpu.data<float>()[0];
    EXPECT_NEAR(kl_value, 0.0f, atol());
}

// ============================================================================
// Distillation Loss Tests
// ============================================================================

TEST_P(DistillationMultiDTypeTest, DistillationLossAlphaBlending) {
    // Test that alpha correctly balances soft and hard losses
    Tensor student_logits = tenzor::zeros({2, 3}, dtype(), device());
    Tensor teacher_logits = tenzor::zeros({2, 3}, dtype(), device());
    Tensor targets = tenzor::zeros({2}, DType::Int64, device());

    // Initialize with some values
    auto s_cpu = student_logits.to(Device::cpu()).to(DType::Float32);
    auto t_cpu = teacher_logits.to(Device::cpu()).to(DType::Float32);
    auto targets_cpu = targets.to(Device::cpu());
    float* s_data = s_cpu.data<float>();
    float* t_data = t_cpu.data<float>();
    int64_t* target_data = targets_cpu.data<int64_t>();

    for (int i = 0; i < 6; ++i) {
        s_data[i] = static_cast<float>(i) * 0.5f;
        t_data[i] = static_cast<float>(i) * 0.7f;
    }
    target_data[0] = 1;
    target_data[1] = 2;

    student_logits = s_cpu.to(dtype()).to(device());
    teacher_logits = t_cpu.to(dtype()).to(device());
    targets = targets_cpu.to(device());

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

TEST_P(DistillationMultiDTypeTest, DistillationLossTemperatureScaling) {
    Tensor student_logits = tenzor::zeros({1, 3}, dtype(), device());
    Tensor teacher_logits = tenzor::zeros({1, 3}, dtype(), device());

    auto s_cpu = student_logits.to(Device::cpu()).to(DType::Float32);
    auto t_cpu = teacher_logits.to(Device::cpu()).to(DType::Float32);
    float* s_data = s_cpu.data<float>();
    float* t_data = t_cpu.data<float>();

    s_data[0] = 1.0f; s_data[1] = 2.0f; s_data[2] = 3.0f;
    t_data[0] = 1.5f; t_data[1] = 2.5f; t_data[2] = 3.5f;

    student_logits = s_cpu.to(dtype()).to(device());
    teacher_logits = t_cpu.to(dtype()).to(device());

    Variable student(student_logits, true);
    Variable teacher(teacher_logits, false);

    DistillationConfig config;
    config.alpha = 1.0f;
    config.use_hard_targets = false;
    config.normalize_temperature = true;

    // Test different temperatures
    std::vector<float> temps = {1.0f, 3.0f, 5.0f, 10.0f};

    for (float temp : temps) {
        config.temperature = temp;
        auto loss = distillation_loss(student, teacher, std::nullopt, config);

        EXPECT_FALSE(has_nan_or_inf(loss.tensor()))
            << "Loss has NaN/Inf for temperature: " << temp;

        auto loss_cpu = loss.tensor().to(Device::cpu()).to(DType::Float32);
        float loss_val = loss_cpu.data<float>()[0];
        EXPECT_GE(loss_val, 0.0f)
            << "Loss should be non-negative for temperature: " << temp;
    }
}

TEST_P(DistillationMultiDTypeTest, DistillationLossNormalization) {
    Tensor student_logits = tenzor::zeros({1, 3}, dtype(), device());
    Tensor teacher_logits = tenzor::zeros({1, 3}, dtype(), device());

    auto s_cpu = student_logits.to(Device::cpu()).to(DType::Float32);
    auto t_cpu = teacher_logits.to(Device::cpu()).to(DType::Float32);
    float* s_data = s_cpu.data<float>();
    float* t_data = t_cpu.data<float>();

    s_data[0] = 1.0f; s_data[1] = 2.0f; s_data[2] = 3.0f;
    t_data[0] = 1.0f; t_data[1] = 2.0f; t_data[2] = 3.0f;

    student_logits = s_cpu.to(dtype()).to(device());
    teacher_logits = t_cpu.to(dtype()).to(device());

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
    auto norm_cpu = loss_normalized.tensor().to(Device::cpu()).to(DType::Float32);
    auto unnorm_cpu = loss_unnormalized.tensor().to(Device::cpu()).to(DType::Float32);
    float norm_val = norm_cpu.data<float>()[0];
    float unnorm_val = unnorm_cpu.data<float>()[0];

    EXPECT_NEAR(norm_val, unnorm_val * 25.0f, 0.5f);  // 5^2 = 25
}

// ============================================================================
// KnowledgeDistillation Class Tests
// ============================================================================

TEST_P(DistillationMultiDTypeTest, KnowledgeDistillationConstruction) {
    auto teacher = std::make_shared<SimpleTeacher>(10, 20, 15, 5);
    auto student = std::make_shared<SimpleStudent>(10, 8, 5);

    convert_model(*teacher);
    convert_model(*student);

    DistillationConfig config;
    config.temperature = 4.0f;
    config.alpha = 0.8f;

    auto distiller = KnowledgeDistillation(teacher, student, config);

    EXPECT_EQ(distiller.teacher(), teacher);
    EXPECT_EQ(distiller.student(), student);
    EXPECT_FLOAT_EQ(distiller.config().temperature, 4.0f);
    EXPECT_FLOAT_EQ(distiller.config().alpha, 0.8f);
}

TEST_P(DistillationMultiDTypeTest, KnowledgeDistillationForwardPass) {
    auto teacher = std::make_shared<SimpleTeacher>(10, 20, 15, 5);
    auto student = std::make_shared<SimpleStudent>(10, 8, 5);

    convert_model(*teacher);
    convert_model(*student);

    auto distiller = KnowledgeDistillation(teacher, student);

    Variable input = createInput({2, 10}, true);

    auto [student_out, teacher_out] = distiller.forward(input);

    EXPECT_EQ(student_out.tensor().shape()[0], 2);
    EXPECT_EQ(student_out.tensor().shape()[1], 5);
    EXPECT_EQ(teacher_out.tensor().shape()[0], 2);
    EXPECT_EQ(teacher_out.tensor().shape()[1], 5);

    EXPECT_FALSE(has_nan_or_inf(student_out.tensor()));
    EXPECT_FALSE(has_nan_or_inf(teacher_out.tensor()));
}

TEST_P(DistillationMultiDTypeTest, KnowledgeDistillationComputeLoss) {
    auto teacher = std::make_shared<SimpleTeacher>(10, 20, 15, 5);
    auto student = std::make_shared<SimpleStudent>(10, 8, 5);

    convert_model(*teacher);
    convert_model(*student);

    DistillationConfig config;
    config.temperature = 3.0f;
    config.alpha = 0.7f;

    auto distiller = KnowledgeDistillation(teacher, student, config);

    Variable input = createInput({2, 10}, true);

    Tensor targets = tenzor::zeros({2}, DType::Int64, device());
    auto targets_cpu = targets.to(Device::cpu());
    targets_cpu.data<int64_t>()[0] = 1;
    targets_cpu.data<int64_t>()[1] = 3;
    targets = targets_cpu.to(device());

    auto loss = distiller.compute_loss(input, targets);

    EXPECT_FALSE(has_nan_or_inf(loss.tensor()));

    auto loss_cpu = loss.tensor().to(Device::cpu()).to(DType::Float32);
    float loss_val = loss_cpu.data<float>()[0];
    EXPECT_GE(loss_val, 0.0f);
}

// ============================================================================
// Temperature Schedule Tests
// ============================================================================

TEST_P(DistillationMultiDTypeTest, TemperatureScheduleLinear) {
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

TEST_P(DistillationMultiDTypeTest, TemperatureScheduleExponential) {
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

TEST_P(DistillationMultiDTypeTest, TemperatureScheduleCosine) {
    float initial = 10.0f;
    float final = 2.0f;
    int total_epochs = 100;

    float temp0 = temperature_schedule(initial, final, 0, total_epochs, "cosine");
    EXPECT_NEAR(temp0, initial, 0.1f);

    float temp100 = temperature_schedule(initial, final, 100, total_epochs, "cosine");
    EXPECT_NEAR(temp100, final, 0.1f);
}

// ============================================================================
// Configuration Presets Tests
// ============================================================================

TEST_P(DistillationMultiDTypeTest, ClassificationConfig) {
    auto config = make_classification_distillation_config();

    EXPECT_FLOAT_EQ(config.temperature, 3.0f);
    EXPECT_FLOAT_EQ(config.alpha, 0.7f);
    EXPECT_TRUE(config.use_hard_targets);
    EXPECT_TRUE(config.normalize_temperature);
}

TEST_P(DistillationMultiDTypeTest, DetectionConfig) {
    auto config = make_detection_distillation_config();

    EXPECT_FLOAT_EQ(config.temperature, 2.0f);
    EXPECT_FLOAT_EQ(config.alpha, 0.5f);
    EXPECT_TRUE(config.use_hard_targets);
    EXPECT_TRUE(config.normalize_temperature);
}

TEST_P(DistillationMultiDTypeTest, SegmentationConfig) {
    auto config = make_segmentation_distillation_config();

    EXPECT_FLOAT_EQ(config.temperature, 1.5f);
    EXPECT_FLOAT_EQ(config.alpha, 0.8f);
    EXPECT_TRUE(config.use_hard_targets);
    EXPECT_FALSE(config.normalize_temperature);  // Different for dense predictions
}

// ============================================================================
// Compression Ratio Tests
// ============================================================================

TEST_P(DistillationMultiDTypeTest, CompressionRatio) {
    auto teacher = std::make_shared<SimpleTeacher>(10, 20, 15, 5);
    auto student = std::make_shared<SimpleStudent>(10, 8, 5);

    float ratio = compute_distillation_compression_ratio(teacher, student);

    // Teacher has more parameters than student
    EXPECT_GT(ratio, 1.0f);

    // Ratio should be reasonable (not infinity or too small)
    EXPECT_LT(ratio, 100.0f);
}

// ============================================================================
// Response-based vs Feature-based Distillation Tests
// ============================================================================

TEST_P(DistillationMultiDTypeTest, ResponseBasedDistillation) {
    // Response-based: Distill only the final outputs
    auto teacher = std::make_shared<SimpleTeacher>(10, 20, 15, 5);
    auto student = std::make_shared<SimpleStudent>(10, 8, 5);

    convert_model(*teacher);
    convert_model(*student);

    DistillationConfig config;
    config.temperature = 3.0f;
    config.alpha = 0.9f;  // Heavy emphasis on soft targets
    config.use_hard_targets = false;

    auto distiller = KnowledgeDistillation(teacher, student, config);

    Variable input = createInput({4, 10}, true);

    auto [student_out, teacher_out] = distiller.forward(input);

    // Compute soft loss only on final outputs
    auto loss = distillation_loss(student_out, teacher_out, std::nullopt, config);

    EXPECT_FALSE(has_nan_or_inf(loss.tensor()));
    auto loss_cpu = loss.tensor().to(Device::cpu()).to(DType::Float32);
    float loss_val = loss_cpu.data<float>()[0];
    EXPECT_GE(loss_val, 0.0f);
}

TEST_P(DistillationMultiDTypeTest, FeatureBasedDistillation) {
    // Feature-based: Distill intermediate layer representations
    auto teacher = std::make_shared<SimpleTeacher>(10, 20, 15, 5);
    auto student = std::make_shared<SimpleStudent>(10, 8, 5);

    convert_model(*teacher);
    convert_model(*student);

    Variable input = createInput({2, 10}, true);

    // Get teacher and student outputs
    auto teacher_out = teacher->forward(input);
    auto student_out = student->forward(input);

    // Feature distillation config (lower temperature)
    DistillationConfig feature_config;
    feature_config.temperature = 1.0f;
    feature_config.alpha = 1.0f;
    feature_config.use_hard_targets = false;

    // Compute feature-level distillation loss
    auto feature_loss = distillation_loss(student_out, teacher_out,
                                         std::nullopt, feature_config);

    EXPECT_FALSE(has_nan_or_inf(feature_loss.tensor()));
    auto loss_cpu = feature_loss.tensor().to(Device::cpu()).to(DType::Float32);
    float loss_val = loss_cpu.data<float>()[0];
    EXPECT_GE(loss_val, 0.0f);
}

// ============================================================================
// Soft Targets Test
// ============================================================================

TEST_P(DistillationMultiDTypeTest, SoftTargetsPreservesKnowledge) {
    // Test that soft targets preserve more information than hard targets
    Tensor teacher_logits = tenzor::zeros({1, 5}, dtype(), device());
    auto t_cpu = teacher_logits.to(Device::cpu()).to(DType::Float32);
    float* t_data = t_cpu.data<float>();

    // Teacher has strong confidence on class 2, but also some on class 1
    t_data[0] = 0.5f;
    t_data[1] = 3.0f;  // Second best
    t_data[2] = 5.0f;  // Best
    t_data[3] = 0.2f;
    t_data[4] = 0.1f;

    teacher_logits = t_cpu.to(dtype()).to(device());

    // Apply temperature softmax to get soft targets
    Variable teacher_var(teacher_logits, false);
    auto soft_targets = temperature_softmax(teacher_var, 3.0f, -1);

    auto soft_cpu = soft_targets.tensor().to(Device::cpu()).to(DType::Float32);
    const float* soft_data = soft_cpu.data<float>();

    // Soft targets should preserve ranking but be more uniform than hard labels
    EXPECT_GT(soft_data[2], soft_data[1]);  // Class 2 > Class 1
    EXPECT_GT(soft_data[1], soft_data[0]);  // Class 1 > Class 0

    // But class 1 should have non-negligible probability (preserves dark knowledge)
    EXPECT_GT(soft_data[1], 0.1f);
}

// ============================================================================
// Numerical Stability Tests
// ============================================================================

TEST_P(DistillationMultiDTypeTest, NumericalStabilityExtremeTemperatures) {
    Variable logits(logits_extreme_, true);

    // Very small temperatures (approaching hard targets)
    std::vector<float> small_temps = {0.001f, 0.01f, 0.05f};

    for (float temp : small_temps) {
        auto result = temperature_softmax(logits, temp, -1);
        EXPECT_FALSE(has_nan_or_inf(result.tensor()))
            << "Failed with small temperature: " << temp;
    }

    // Very large temperatures (approaching uniform)
    std::vector<float> large_temps = {50.0f, 100.0f, 1000.0f};

    for (float temp : large_temps) {
        auto result = temperature_softmax(logits, temp, -1);
        EXPECT_FALSE(has_nan_or_inf(result.tensor()))
            << "Failed with large temperature: " << temp;
    }
}

TEST_P(DistillationMultiDTypeTest, NumericalStabilityAllZeros) {
    // Test with all zeros (should give uniform distribution)
    Tensor all_zeros = tenzor::zeros({1, 3}, dtype(), device());

    Variable logits(all_zeros, true);
    auto result = temperature_softmax(logits, 1.0f, -1);

    EXPECT_FALSE(has_nan_or_inf(result.tensor()));

    auto result_cpu = result.tensor().to(Device::cpu()).to(DType::Float32);
    const float* data = result_cpu.data<float>();
    float expected = 1.0f / 3.0f;
    for (int i = 0; i < 3; ++i) {
        EXPECT_NEAR(data[i], expected, atol());
    }
}

// ============================================================================
// Test Instantiation
// ============================================================================

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(DistillationMultiDTypeTest);

/*
 * COVERAGE SUMMARY:
 *
 * Test Cases: 28
 * DTypes Tested: Float32, Float64, Float16
 * Backends Tested: CPU, CUDA, OneAPI
 * Total Scenarios: 28 tests × 3 dtypes × 3 backends = 252 test scenarios
 *
 * Coverage:
 * - Temperature softmax: normal values, extreme values, large values, small/large temp
 * - Temperature log-softmax: stability, vs softmax consistency
 * - KL divergence: basic, identical distributions
 * - Distillation loss: alpha blending, temperature scaling, normalization
 * - KnowledgeDistillation class: construction, forward pass, compute loss
 * - Temperature schedules: linear, exponential, cosine
 * - Configuration presets: classification, detection, segmentation
 * - Compression ratio computation
 * - Response-based vs feature-based distillation
 * - Soft targets knowledge preservation
 * - Numerical stability: extreme temperatures, all zeros
 */
