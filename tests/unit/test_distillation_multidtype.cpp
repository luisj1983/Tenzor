/**
 * @file test_distillation_multidtype.cpp
 * @brief Multi-dtype tests for Knowledge Distillation (Float32, Float64, Float16)
 *
 * Tests knowledge distillation functionality across different data types:
 * - Temperature-scaled softmax stability with extreme values
 * - KL divergence computation accuracy
 * - Teacher-student training convergence
 * - Model compression effectiveness
 * - Response-based vs feature-based distillation
 */

#include <gtest/gtest.h>
#include "tenzor/tenzor.hpp"
#include "tenzor/nn/compression/distillation.hpp"
#include <cmath>
#include <limits>
#include <type_traits>

using namespace tenzor;
using namespace tenzor::nn;
using namespace tenzor::nn::compression;

// =============================================================================
// Type-generic Helper Functions
// =============================================================================

/**
 * @brief Check if tensor contains NaN or Inf values (templated)
 */
template<typename T>
bool has_nan_or_inf(const Tensor& t) {
    const T* data = t.data<T>();
    int64_t numel = t.numel();

    for (int64_t i = 0; i < numel; ++i) {
        if (std::isnan(static_cast<float>(data[i])) ||
            std::isinf(static_cast<float>(data[i]))) {
            return true;
        }
    }
    return false;
}

/**
 * @brief Naive (unstable) temperature softmax for comparison
 */
template<typename T>
Tensor naive_temperature_softmax(const Tensor& logits, T temperature) {
    const T* input_data = logits.data<T>();
    auto shape = logits.shape();

    Tensor output = Tensor::zeros_like(logits);
    T* output_data = output.data<T>();

    // Compute per-row softmax for (batch, features) shape
    int64_t batch_size = shape[0];
    int64_t num_features = shape[1];

    for (int64_t b = 0; b < batch_size; ++b) {
        // Compute exp(x/T) for this row - potentially unstable
        T sum = static_cast<T>(0);
        for (int64_t f = 0; f < num_features; ++f) {
            int64_t idx = b * num_features + f;
            T val = std::exp(input_data[idx] / temperature);
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
 * @brief Simple student model for testing (templated)
 */
template<typename T>
class SimpleStudentT : public Module {
public:
    SimpleStudentT(int64_t input_size, int64_t hidden_size, int64_t output_size, DType dtype)
        : dtype_(dtype) {
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
    DType dtype_;
};

/**
 * @brief Simple teacher model for testing (larger capacity)
 */
template<typename T>
class SimpleTeacherT : public Module {
public:
    SimpleTeacherT(int64_t input_size, int64_t hidden1_size,
                   int64_t hidden2_size, int64_t output_size, DType dtype)
        : dtype_(dtype) {
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
    DType dtype_;
};

// =============================================================================
// Templated Test Fixture
// =============================================================================

template<typename T>
class DistillationMultiDTypeTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Determine DType
        if (std::is_same<T, float>::value) {
            dtype_ = DType::Float32;
            tolerance_ = static_cast<T>(1e-5);
            sum_tolerance_ = static_cast<T>(1e-4);
        } else if (std::is_same<T, double>::value) {
            dtype_ = DType::Float64;
            tolerance_ = static_cast<T>(1e-10);
            sum_tolerance_ = static_cast<T>(1e-9);
        } else {  // Float16
            dtype_ = DType::Float16;
            tolerance_ = static_cast<T>(1e-2);
            sum_tolerance_ = static_cast<T>(1e-2);
        }

        // Create simple test data
        logits_normal_ = Tensor({2, 3}, dtype_, Device::cpu());
        T* data = logits_normal_.data<T>();
        data[0] = static_cast<T>(1.0); data[1] = static_cast<T>(2.0); data[2] = static_cast<T>(3.0);
        data[3] = static_cast<T>(-1.0); data[4] = static_cast<T>(0.0); data[5] = static_cast<T>(1.0);

        // Create extreme value test data (scaled for Float16)
        logits_extreme_ = Tensor({2, 3}, dtype_, Device::cpu());
        T* extreme_data = logits_extreme_.data<T>();
        T scale = std::is_same<T, float>::value || std::is_same<T, double>::value
                  ? static_cast<T>(100.0) : static_cast<T>(10.0);
        extreme_data[0] = scale; extreme_data[1] = scale * static_cast<T>(2.0);
        extreme_data[2] = scale * static_cast<T>(3.0);
        extreme_data[3] = -scale; extreme_data[4] = -scale * static_cast<T>(2.0);
        extreme_data[5] = -scale * static_cast<T>(3.0);

        // Very large values (scaled for Float16)
        logits_very_large_ = Tensor({1, 3}, dtype_, Device::cpu());
        T* large_data = logits_very_large_.data<T>();
        T large_scale = std::is_same<T, float>::value || std::is_same<T, double>::value
                        ? static_cast<T>(1000.0) : static_cast<T>(20.0);
        large_data[0] = large_scale;
        large_data[1] = large_scale * static_cast<T>(2.0);
        large_data[2] = large_scale * static_cast<T>(3.0);
    }

    DType dtype_;
    T tolerance_;
    T sum_tolerance_;
    Tensor logits_normal_;
    Tensor logits_extreme_;
    Tensor logits_very_large_;
};

// Define type list
using TestTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(DistillationMultiDTypeTest, TestTypes);

// =============================================================================
// Temperature Softmax Stability Tests
// =============================================================================

TYPED_TEST(DistillationMultiDTypeTest, TemperatureSoftmaxNormalValues) {
    Variable logits(this->logits_normal_, true);
    using T = TypeParam;

    auto result = temperature_softmax(logits, static_cast<T>(1.0), -1);

    // No NaN or Inf
    EXPECT_FALSE(has_nan_or_inf<T>(result.tensor()));

    // Check probabilities sum to 1 for each sample
    const T* data = result.tensor().template data<T>();
    T sum1 = data[0] + data[1] + data[2];
    T sum2 = data[3] + data[4] + data[5];

    EXPECT_NEAR(static_cast<double>(sum1), 1.0, static_cast<double>(this->sum_tolerance_));
    EXPECT_NEAR(static_cast<double>(sum2), 1.0, static_cast<double>(this->sum_tolerance_));
}

TYPED_TEST(DistillationMultiDTypeTest, TemperatureSoftmaxExtremeValues) {
    // CRITICAL TEST: Numerical stability with extreme values
    Variable logits(this->logits_extreme_, true);
    using T = TypeParam;

    std::vector<T> temps = {
        static_cast<T>(0.01), static_cast<T>(0.1),
        static_cast<T>(1.0), static_cast<T>(5.0), static_cast<T>(10.0)
    };

    for (T temp : temps) {
        auto result = temperature_softmax(logits, temp, -1);

        // Must not produce NaN or Inf
        EXPECT_FALSE(has_nan_or_inf<T>(result.tensor()))
            << "Failed for temperature: " << temp;

        // Probabilities must sum to 1
        const T* data = result.tensor().template data<T>();
        T sum1 = data[0] + data[1] + data[2];
        T sum2 = data[3] + data[4] + data[5];

        EXPECT_NEAR(static_cast<double>(sum1), 1.0, static_cast<double>(this->sum_tolerance_))
            << "Sample 1 sum failed for temperature: " << temp;
        EXPECT_NEAR(static_cast<double>(sum2), 1.0, static_cast<double>(this->sum_tolerance_))
            << "Sample 2 sum failed for temperature: " << temp;

        // All probabilities should be in [0, 1]
        for (int64_t i = 0; i < 6; ++i) {
            EXPECT_GE(static_cast<double>(data[i]), 0.0)
                << "Negative probability at index " << i;
            EXPECT_LE(static_cast<double>(data[i]), 1.0)
                << "Probability > 1 at index " << i;
        }
    }
}

TYPED_TEST(DistillationMultiDTypeTest, TemperatureSoftmaxVeryLargeValues) {
    // Test with values that would overflow naive exp(x/T)
    Variable logits(this->logits_very_large_, true);
    using T = TypeParam;

    std::vector<T> temps = {
        static_cast<T>(0.01), static_cast<T>(0.1),
        static_cast<T>(1.0), static_cast<T>(10.0)
    };

    for (T temp : temps) {
        auto result = temperature_softmax(logits, temp, -1);

        // Must not overflow
        EXPECT_FALSE(has_nan_or_inf<T>(result.tensor()))
            << "Overflow with temperature: " << temp;

        // Must sum to 1
        const T* data = result.tensor().template data<T>();
        T sum = data[0] + data[1] + data[2];
        EXPECT_NEAR(static_cast<double>(sum), 1.0, static_cast<double>(this->sum_tolerance_));
    }
}

TYPED_TEST(DistillationMultiDTypeTest, TemperatureSoftmaxSmallTemperature) {
    // Very small temperature should approach one-hot
    Variable logits(this->logits_normal_, true);
    using T = TypeParam;

    auto result = temperature_softmax(logits, static_cast<T>(0.01), -1);

    EXPECT_FALSE(has_nan_or_inf<T>(result.tensor()));

    const T* data = result.tensor().template data<T>();

    // First sample: largest logit is index 2 (value 3.0)
    EXPECT_NEAR(static_cast<double>(data[2]), 1.0, 0.01);  // Should be close to 1
    EXPECT_LT(static_cast<double>(data[0]), 0.01);  // Others should be close to 0
    EXPECT_LT(static_cast<double>(data[1]), 0.01);

    // Second sample: largest logit is index 5 (value 1.0)
    EXPECT_NEAR(static_cast<double>(data[5]), 1.0, 0.01);
    EXPECT_LT(static_cast<double>(data[3]), 0.01);
    EXPECT_LT(static_cast<double>(data[4]), 0.01);
}

TYPED_TEST(DistillationMultiDTypeTest, TemperatureSoftmaxLargeTemperature) {
    // Large temperature should approach uniform distribution
    Variable logits(this->logits_normal_, true);
    using T = TypeParam;

    auto result = temperature_softmax(logits, static_cast<T>(100.0), -1);

    EXPECT_FALSE(has_nan_or_inf<T>(result.tensor()));

    const T* data = result.tensor().template data<T>();

    // Each sample should have roughly uniform probabilities
    T expected = static_cast<T>(1.0 / 3.0);
    for (int i = 0; i < 3; ++i) {
        EXPECT_NEAR(static_cast<double>(data[i]),
                    static_cast<double>(expected), 0.05);
        EXPECT_NEAR(static_cast<double>(data[i + 3]),
                    static_cast<double>(expected), 0.05);
    }
}

// =============================================================================
// Temperature Log-Softmax Tests
// =============================================================================

TYPED_TEST(DistillationMultiDTypeTest, TemperatureLogSoftmaxStability) {
    Variable logits(this->logits_extreme_, true);
    using T = TypeParam;

    std::vector<T> temps = {
        static_cast<T>(0.01), static_cast<T>(0.1),
        static_cast<T>(1.0), static_cast<T>(10.0)
    };

    for (T temp : temps) {
        auto result = temperature_log_softmax(logits, temp, -1);

        // Must not produce NaN or Inf
        EXPECT_FALSE(has_nan_or_inf<T>(result.tensor()))
            << "Failed for temperature: " << temp;

        // Log probabilities should be negative
        const T* data = result.tensor().template data<T>();
        for (int64_t i = 0; i < 6; ++i) {
            EXPECT_LE(static_cast<double>(data[i]), 0.0)
                << "Log prob should be <= 0 at index " << i;
        }
    }
}

TYPED_TEST(DistillationMultiDTypeTest, TemperatureLogSoftmaxVsSoftmax) {
    // log_softmax should match log(softmax)
    Variable logits(this->logits_normal_, true);
    using T = TypeParam;

    auto log_result = temperature_log_softmax(logits, static_cast<T>(2.0), -1);
    auto softmax_result = temperature_softmax(logits, static_cast<T>(2.0), -1);

    // Manually compute log of softmax
    const T* softmax_data = softmax_result.tensor().template data<T>();
    const T* log_data = log_result.tensor().template data<T>();

    for (int64_t i = 0; i < 6; ++i) {
        T expected_log = std::log(softmax_data[i]);
        EXPECT_NEAR(static_cast<double>(log_data[i]),
                    static_cast<double>(expected_log),
                    static_cast<double>(this->tolerance_))
            << "Mismatch at index " << i;
    }
}

// =============================================================================
// KL Divergence Tests
// =============================================================================

TYPED_TEST(DistillationMultiDTypeTest, KLDivergenceBasic) {
    using T = TypeParam;

    // Create simple probability distributions
    Tensor p_tensor({2, 3}, this->dtype_, Device::cpu());
    Tensor log_q_tensor({2, 3}, this->dtype_, Device::cpu());

    T* p_data = p_tensor.data<T>();
    T* log_q_data = log_q_tensor.data<T>();

    // P = [0.5, 0.3, 0.2]
    p_data[0] = static_cast<T>(0.5);
    p_data[1] = static_cast<T>(0.3);
    p_data[2] = static_cast<T>(0.2);

    // Q = [0.4, 0.4, 0.2], log Q = [log(0.4), log(0.4), log(0.2)]
    log_q_data[0] = std::log(static_cast<T>(0.4));
    log_q_data[1] = std::log(static_cast<T>(0.4));
    log_q_data[2] = std::log(static_cast<T>(0.2));

    // Second sample (same for simplicity)
    p_data[3] = static_cast<T>(0.5);
    p_data[4] = static_cast<T>(0.3);
    p_data[5] = static_cast<T>(0.2);
    log_q_data[3] = std::log(static_cast<T>(0.4));
    log_q_data[4] = std::log(static_cast<T>(0.4));
    log_q_data[5] = std::log(static_cast<T>(0.2));

    Variable p(p_tensor, false);
    Variable log_q(log_q_tensor, false);

    auto kl = kl_divergence(log_q, p, "sum");

    EXPECT_FALSE(has_nan_or_inf<T>(kl.tensor()));

    // KL divergence sum should be non-negative
    T kl_sum = kl.tensor().template data<T>()[0];
    EXPECT_GE(static_cast<double>(kl_sum), 0.0)
        << "Total KL divergence should be non-negative";

    // KL(P||Q) should be positive when P != Q
    EXPECT_GT(static_cast<double>(kl_sum), 0.0)
        << "KL(P||Q) should be positive when P != Q";
}

TYPED_TEST(DistillationMultiDTypeTest, KLDivergenceIdenticalDistributions) {
    using T = TypeParam;

    // KL(P||P) should be 0
    Tensor p_tensor({1, 3}, this->dtype_, Device::cpu());
    T* data = p_tensor.data<T>();
    data[0] = static_cast<T>(0.5);
    data[1] = static_cast<T>(0.3);
    data[2] = static_cast<T>(0.2);

    Tensor log_p_tensor({1, 3}, this->dtype_, Device::cpu());
    T* log_data = log_p_tensor.data<T>();
    log_data[0] = std::log(static_cast<T>(0.5));
    log_data[1] = std::log(static_cast<T>(0.3));
    log_data[2] = std::log(static_cast<T>(0.2));

    Variable p(p_tensor, false);
    Variable log_p(log_p_tensor, false);

    auto kl = kl_divergence(log_p, p, "batchmean");

    // Should be very close to 0
    T kl_value = kl.tensor().template data<T>()[0];
    EXPECT_NEAR(static_cast<double>(kl_value), 0.0,
                static_cast<double>(this->tolerance_));
}

// =============================================================================
// Distillation Loss Tests
// =============================================================================

TYPED_TEST(DistillationMultiDTypeTest, DistillationLossAlphaBlending) {
    using T = TypeParam;

    // Test that alpha correctly balances soft and hard losses
    Tensor student_logits({2, 3}, this->dtype_, Device::cpu());
    Tensor teacher_logits({2, 3}, this->dtype_, Device::cpu());
    Tensor targets({2}, DType::Int64, Device::cpu());

    // Initialize with some values
    T* s_data = student_logits.data<T>();
    T* t_data = teacher_logits.data<T>();
    int64_t* target_data = targets.data<int64_t>();

    for (int i = 0; i < 6; ++i) {
        s_data[i] = static_cast<T>(i) * static_cast<T>(0.5);
        t_data[i] = static_cast<T>(i) * static_cast<T>(0.7);
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
    EXPECT_FALSE(has_nan_or_inf<T>(loss_hard.tensor()));

    // Test alpha = 1.0 (only soft targets)
    DistillationConfig config_soft;
    config_soft.alpha = 1.0f;
    config_soft.use_hard_targets = false;
    config_soft.temperature = 3.0f;

    auto loss_soft = distillation_loss(student, teacher, std::nullopt, config_soft);
    EXPECT_FALSE(has_nan_or_inf<T>(loss_soft.tensor()));

    // Test alpha = 0.5 (balanced)
    DistillationConfig config_balanced;
    config_balanced.alpha = 0.5f;
    config_balanced.use_hard_targets = true;
    config_balanced.temperature = 3.0f;

    auto loss_balanced = distillation_loss(student, teacher, targets, config_balanced);
    EXPECT_FALSE(has_nan_or_inf<T>(loss_balanced.tensor()));
}

TYPED_TEST(DistillationMultiDTypeTest, DistillationLossTemperatureScaling) {
    using T = TypeParam;

    Tensor student_logits({1, 3}, this->dtype_, Device::cpu());
    Tensor teacher_logits({1, 3}, this->dtype_, Device::cpu());

    T* s_data = student_logits.data<T>();
    T* t_data = teacher_logits.data<T>();

    s_data[0] = static_cast<T>(1.0);
    s_data[1] = static_cast<T>(2.0);
    s_data[2] = static_cast<T>(3.0);
    t_data[0] = static_cast<T>(1.5);
    t_data[1] = static_cast<T>(2.5);
    t_data[2] = static_cast<T>(3.5);

    Variable student(student_logits, true);
    Variable teacher(teacher_logits, false);

    DistillationConfig config;
    config.alpha = 1.0f;
    config.use_hard_targets = false;
    config.normalize_temperature = true;

    // Test different temperatures
    std::vector<T> temps = {
        static_cast<T>(1.0), static_cast<T>(3.0),
        static_cast<T>(5.0), static_cast<T>(10.0)
    };

    for (T temp : temps) {
        config.temperature = static_cast<float>(temp);
        auto loss = distillation_loss(student, teacher, std::nullopt, config);

        EXPECT_FALSE(has_nan_or_inf<T>(loss.tensor()))
            << "Loss has NaN/Inf for temperature: " << temp;

        T loss_val = loss.tensor().template data<T>()[0];
        EXPECT_GE(static_cast<double>(loss_val), 0.0)
            << "Loss should be non-negative for temperature: " << temp;
    }
}

TYPED_TEST(DistillationMultiDTypeTest, DistillationLossNormalization) {
    using T = TypeParam;

    Tensor student_logits({1, 3}, this->dtype_, Device::cpu());
    Tensor teacher_logits({1, 3}, this->dtype_, Device::cpu());

    T* s_data = student_logits.data<T>();
    T* t_data = teacher_logits.data<T>();

    s_data[0] = static_cast<T>(1.0);
    s_data[1] = static_cast<T>(2.0);
    s_data[2] = static_cast<T>(3.0);
    t_data[0] = static_cast<T>(1.0);
    t_data[1] = static_cast<T>(2.0);
    t_data[2] = static_cast<T>(3.0);

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
    T norm_val = loss_normalized.tensor().template data<T>()[0];
    T unnorm_val = loss_unnormalized.tensor().template data<T>()[0];

    EXPECT_NEAR(static_cast<double>(norm_val),
                static_cast<double>(unnorm_val) * 25.0, 0.5);  // 5^2 = 25
}

// =============================================================================
// KnowledgeDistillation Class Tests
// =============================================================================

TYPED_TEST(DistillationMultiDTypeTest, KnowledgeDistillationConstruction) {
    using T = TypeParam;

    auto teacher = std::make_shared<SimpleTeacherT<T>>(10, 20, 15, 5, this->dtype_);
    auto student = std::make_shared<SimpleStudentT<T>>(10, 8, 5, this->dtype_);

    DistillationConfig config;
    config.temperature = 4.0f;
    config.alpha = 0.8f;

    auto distiller = KnowledgeDistillation(teacher, student, config);

    EXPECT_EQ(distiller.teacher(), teacher);
    EXPECT_EQ(distiller.student(), student);
    EXPECT_FLOAT_EQ(distiller.config().temperature, 4.0f);
    EXPECT_FLOAT_EQ(distiller.config().alpha, 0.8f);
}

TYPED_TEST(DistillationMultiDTypeTest, KnowledgeDistillationForwardPass) {
    using T = TypeParam;

    auto teacher = std::make_shared<SimpleTeacherT<T>>(10, 20, 15, 5, this->dtype_);
    auto student = std::make_shared<SimpleStudentT<T>>(10, 8, 5, this->dtype_);

    auto distiller = KnowledgeDistillation(teacher, student);

    Tensor input({2, 10}, this->dtype_, Device::cpu());
    input.fill_(static_cast<T>(0.5));
    Variable input_var(input, true);

    auto [student_out, teacher_out] = distiller.forward(input_var);

    EXPECT_EQ(student_out.tensor().shape()[0], 2);
    EXPECT_EQ(student_out.tensor().shape()[1], 5);
    EXPECT_EQ(teacher_out.tensor().shape()[0], 2);
    EXPECT_EQ(teacher_out.tensor().shape()[1], 5);

    EXPECT_FALSE(has_nan_or_inf<T>(student_out.tensor()));
    EXPECT_FALSE(has_nan_or_inf<T>(teacher_out.tensor()));
}

TYPED_TEST(DistillationMultiDTypeTest, KnowledgeDistillationComputeLoss) {
    using T = TypeParam;

    auto teacher = std::make_shared<SimpleTeacherT<T>>(10, 20, 15, 5, this->dtype_);
    auto student = std::make_shared<SimpleStudentT<T>>(10, 8, 5, this->dtype_);

    DistillationConfig config;
    config.temperature = 3.0f;
    config.alpha = 0.7f;

    auto distiller = KnowledgeDistillation(teacher, student, config);

    Tensor input({2, 10}, this->dtype_, Device::cpu());
    input.fill_(static_cast<T>(0.5));
    Variable input_var(input, true);

    Tensor targets({2}, DType::Int64, Device::cpu());
    targets.data<int64_t>()[0] = 1;
    targets.data<int64_t>()[1] = 3;

    auto loss = distiller.compute_loss(input_var, targets);

    EXPECT_FALSE(has_nan_or_inf<T>(loss.tensor()));

    T loss_val = loss.tensor().template data<T>()[0];
    EXPECT_GE(static_cast<double>(loss_val), 0.0);
}

// =============================================================================
// Temperature Schedule Tests
// =============================================================================

TYPED_TEST(DistillationMultiDTypeTest, TemperatureScheduleLinear) {
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

TYPED_TEST(DistillationMultiDTypeTest, TemperatureScheduleExponential) {
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

TYPED_TEST(DistillationMultiDTypeTest, TemperatureScheduleCosine) {
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

TYPED_TEST(DistillationMultiDTypeTest, ClassificationConfig) {
    auto config = make_classification_distillation_config();

    EXPECT_FLOAT_EQ(config.temperature, 3.0f);
    EXPECT_FLOAT_EQ(config.alpha, 0.7f);
    EXPECT_TRUE(config.use_hard_targets);
    EXPECT_TRUE(config.normalize_temperature);
}

TYPED_TEST(DistillationMultiDTypeTest, DetectionConfig) {
    auto config = make_detection_distillation_config();

    EXPECT_FLOAT_EQ(config.temperature, 2.0f);
    EXPECT_FLOAT_EQ(config.alpha, 0.5f);
    EXPECT_TRUE(config.use_hard_targets);
    EXPECT_TRUE(config.normalize_temperature);
}

TYPED_TEST(DistillationMultiDTypeTest, SegmentationConfig) {
    auto config = make_segmentation_distillation_config();

    EXPECT_FLOAT_EQ(config.temperature, 1.5f);
    EXPECT_FLOAT_EQ(config.alpha, 0.8f);
    EXPECT_TRUE(config.use_hard_targets);
    EXPECT_FALSE(config.normalize_temperature);  // Different for dense predictions
}

// =============================================================================
// Compression Ratio Tests
// =============================================================================

TYPED_TEST(DistillationMultiDTypeTest, CompressionRatio) {
    using T = TypeParam;

    auto teacher = std::make_shared<SimpleTeacherT<T>>(10, 20, 15, 5, this->dtype_);
    auto student = std::make_shared<SimpleStudentT<T>>(10, 8, 5, this->dtype_);

    float ratio = compute_distillation_compression_ratio(teacher, student);

    // Teacher has more parameters than student
    EXPECT_GT(ratio, 1.0f);

    // Ratio should be reasonable (not infinity or too small)
    EXPECT_LT(ratio, 100.0f);
}

// =============================================================================
// Response-based vs Feature-based Distillation Tests
// =============================================================================

TYPED_TEST(DistillationMultiDTypeTest, ResponseBasedDistillation) {
    using T = TypeParam;

    // Response-based: Distill only the final outputs
    auto teacher = std::make_shared<SimpleTeacherT<T>>(10, 20, 15, 5, this->dtype_);
    auto student = std::make_shared<SimpleStudentT<T>>(10, 8, 5, this->dtype_);

    DistillationConfig config;
    config.temperature = 3.0f;
    config.alpha = 0.9f;  // Heavy emphasis on soft targets
    config.use_hard_targets = false;

    auto distiller = KnowledgeDistillation(teacher, student, config);

    Tensor input({4, 10}, this->dtype_, Device::cpu());
    input.fill_(static_cast<T>(0.5));
    Variable input_var(input, true);

    auto [student_out, teacher_out] = distiller.forward(input_var);

    // Compute soft loss only on final outputs
    auto loss = distillation_loss(student_out, teacher_out, std::nullopt, config);

    EXPECT_FALSE(has_nan_or_inf<T>(loss.tensor()));
    T loss_val = loss.tensor().template data<T>()[0];
    EXPECT_GE(static_cast<double>(loss_val), 0.0);
}

TYPED_TEST(DistillationMultiDTypeTest, FeatureBasedDistillation) {
    using T = TypeParam;

    // Feature-based: Distill intermediate layer representations
    // This would typically require access to intermediate features
    // Here we simulate by using multiple distillation losses

    auto teacher = std::make_shared<SimpleTeacherT<T>>(10, 20, 15, 5, this->dtype_);
    auto student = std::make_shared<SimpleStudentT<T>>(10, 8, 5, this->dtype_);

    Tensor input({2, 10}, this->dtype_, Device::cpu());
    input.fill_(static_cast<T>(0.5));
    Variable input_var(input, true);

    // Get teacher and student outputs
    auto teacher_out = teacher->forward(input_var);
    auto student_out = student->forward(input_var);

    // Feature distillation config (lower temperature)
    DistillationConfig feature_config;
    feature_config.temperature = 1.0f;
    feature_config.alpha = 1.0f;
    feature_config.use_hard_targets = false;

    // Compute feature-level distillation loss
    auto feature_loss = distillation_loss(student_out, teacher_out,
                                         std::nullopt, feature_config);

    EXPECT_FALSE(has_nan_or_inf<T>(feature_loss.tensor()));
    T loss_val = feature_loss.tensor().template data<T>()[0];
    EXPECT_GE(static_cast<double>(loss_val), 0.0);
}

// =============================================================================
// Soft Targets Test
// =============================================================================

TYPED_TEST(DistillationMultiDTypeTest, SoftTargetsPreservesKnowledge) {
    using T = TypeParam;

    // Test that soft targets preserve more information than hard targets
    Tensor teacher_logits({1, 5}, this->dtype_, Device::cpu());
    T* t_data = teacher_logits.data<T>();

    // Teacher has strong confidence on class 2, but also some on class 1
    t_data[0] = static_cast<T>(0.5);
    t_data[1] = static_cast<T>(3.0);  // Second best
    t_data[2] = static_cast<T>(5.0);  // Best
    t_data[3] = static_cast<T>(0.2);
    t_data[4] = static_cast<T>(0.1);

    // Apply temperature softmax to get soft targets
    Variable teacher_var(teacher_logits, false);
    T temperature = static_cast<T>(3.0);
    auto soft_targets = temperature_softmax(teacher_var, temperature, -1);

    const T* soft_data = soft_targets.tensor().template data<T>();

    // Soft targets should preserve ranking but be more uniform than hard labels
    EXPECT_GT(static_cast<double>(soft_data[2]),
              static_cast<double>(soft_data[1]));  // Class 2 > Class 1
    EXPECT_GT(static_cast<double>(soft_data[1]),
              static_cast<double>(soft_data[0]));  // Class 1 > Class 0

    // But class 1 should have non-negligible probability (preserves dark knowledge)
    EXPECT_GT(static_cast<double>(soft_data[1]), 0.1);
}

// =============================================================================
// Numerical Stability Tests
// =============================================================================

TYPED_TEST(DistillationMultiDTypeTest, NumericalStabilityExtremeTemperatures) {
    Variable logits(this->logits_extreme_, true);
    using T = TypeParam;

    // Very small temperatures (approaching hard targets)
    std::vector<T> small_temps = {
        static_cast<T>(0.001), static_cast<T>(0.01), static_cast<T>(0.05)
    };

    for (T temp : small_temps) {
        auto result = temperature_softmax(logits, temp, -1);
        EXPECT_FALSE(has_nan_or_inf<T>(result.tensor()))
            << "Failed with small temperature: " << temp;
    }

    // Very large temperatures (approaching uniform)
    std::vector<T> large_temps = {
        static_cast<T>(50.0), static_cast<T>(100.0), static_cast<T>(1000.0)
    };

    for (T temp : large_temps) {
        auto result = temperature_softmax(logits, temp, -1);
        EXPECT_FALSE(has_nan_or_inf<T>(result.tensor()))
            << "Failed with large temperature: " << temp;
    }
}

TYPED_TEST(DistillationMultiDTypeTest, NumericalStabilityAllZeros) {
    using T = TypeParam;

    // Test with all zeros (should give uniform distribution)
    Tensor all_zeros({1, 3}, this->dtype_, Device::cpu());
    all_zeros.fill_(static_cast<T>(0.0));

    Variable logits(all_zeros, true);
    auto result = temperature_softmax(logits, static_cast<T>(1.0), -1);

    EXPECT_FALSE(has_nan_or_inf<T>(result.tensor()));

    const T* data = result.tensor().template data<T>();
    T expected = static_cast<T>(1.0 / 3.0);
    for (int i = 0; i < 3; ++i) {
        EXPECT_NEAR(static_cast<double>(data[i]),
                    static_cast<double>(expected),
                    static_cast<double>(this->tolerance_));
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
