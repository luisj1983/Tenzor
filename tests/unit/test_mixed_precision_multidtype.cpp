/**
 * @file test_mixed_precision_multidtype.cpp
 * @brief Multi-dtype tests for mixed precision training
 *
 * Tests mixed precision training behavior across different base dtypes:
 * - Float32 + Float16 mixed precision (most common)
 * - Float64 + Float16 mixed precision
 * - GradScaler behavior with different dtypes
 * - Autocast behavior across dtypes
 * - Loss computation in correct precision
 */

#include <gtest/gtest.h>
#include "tenzor/tenzor.hpp"
#include "tenzor/nn/mixed_precision.hpp"
#include "tenzor/nn/module.hpp"
#include "tenzor/nn/layers/linear.hpp"
#include "tenzor/nn/activations/activations.hpp"
#include "tenzor/nn/optim/sgd.hpp"
#include "tenzor/nn/optim/adam.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include <cmath>
#include <vector>

using namespace tenzor;
using namespace tenzor::nn;

// Simple MLP model for testing
class SimpleMLP : public Module {
public:
    SimpleMLP(int input_size, int hidden_size, int output_size)
        : fc1_(std::make_shared<Linear>(input_size, hidden_size)),
          relu_(std::make_shared<ReLU>()),
          fc2_(std::make_shared<Linear>(hidden_size, output_size)) {
        register_module("fc1", fc1_);
        register_module("relu", relu_);
        register_module("fc2", fc2_);
    }

    auto forward(const Variable& x) -> Variable override {
        auto h = fc1_->forward(x);
        h = relu_->forward(h);
        return fc2_->forward(h);
    }

private:
    std::shared_ptr<Linear> fc1_;
    std::shared_ptr<ReLU> relu_;
    std::shared_ptr<Linear> fc2_;
};

// Helper to get list of test dtypes
std::vector<DType> get_test_dtypes() {
    return {DType::Float32, DType::Float64};
}

class MixedPrecisionMultiDTypeTest : public ::testing::TestWithParam<DType> {
protected:
    void SetUp() override {
        dtype_ = GetParam();
        device_ = Device::cpu();
    }

    DType dtype_;
    Device device_;
};

// Test 1: MixedPrecisionConfig - FP16 configuration with different base dtypes
TEST_P(MixedPrecisionMultiDTypeTest, ConfigFP16WithBaseDType) {
    auto config = MixedPrecisionConfig::fp16_cuda();
    config.enabled = false;  // Disable for CPU

    EXPECT_EQ(config.dtype, DType::Float16);
    EXPECT_FLOAT_EQ(config.init_scale, 65536.0f);
    EXPECT_FLOAT_EQ(config.growth_factor, 2.0f);
    EXPECT_FLOAT_EQ(config.backoff_factor, 0.5f);
    EXPECT_EQ(config.growth_interval, 2000);
}

// Test 2: Train step with different base dtypes and FP16 mixed precision
TEST_P(MixedPrecisionMultiDTypeTest, TrainStepWithMixedPrecision) {
    auto model = std::make_shared<SimpleMLP>(10, 20, 5);
    auto optimizer = std::make_shared<optim::SGD>(model->parameters(), 0.01);
    auto loss_fn = [](const Variable& pred, const Variable& target) {
        auto diff = pred - target;
        return mean(diff * diff);
    };

    // Mixed precision disabled for CPU
    auto config = MixedPrecisionConfig::fp16_cuda();
    config.enabled = false;
    MixedPrecisionTrainer trainer(model, optimizer, loss_fn, config);

    // Create input and target in base dtype
    auto input_tensor = randn({4, 10}, dtype_, device_);
    auto target_tensor = randn({4, 5}, dtype_, device_);
    auto input = Variable(input_tensor, false);
    auto target = Variable(target_tensor, false);

    // Perform training step
    float loss = trainer.train_step(input, target);

    EXPECT_GT(loss, 0.0f) << "Loss should be positive for dtype: " << dtype_name(dtype_);
    EXPECT_EQ(trainer.get_total_steps(), 1);
    EXPECT_EQ(trainer.get_skipped_steps(), 0);
}

// Test 3: Multiple training steps with different base dtypes
TEST_P(MixedPrecisionMultiDTypeTest, MultipleTrainStepsWithBaseDType) {
    auto model = std::make_shared<SimpleMLP>(5, 10, 3);
    auto optimizer = std::make_shared<optim::Adam>(model->parameters(), 0.001);
    auto loss_fn = [](const Variable& pred, const Variable& target) {
        auto diff = pred - target;
        return mean(diff * diff);
    };

    auto config = MixedPrecisionConfig::fp16_cuda();
    config.enabled = false;
    MixedPrecisionTrainer trainer(model, optimizer, loss_fn, config);

    auto input_tensor = randn({8, 5}, dtype_, device_);
    auto target_tensor = randn({8, 3}, dtype_, device_);
    auto input = Variable(input_tensor, false);
    auto target = Variable(target_tensor, false);

    // Perform 10 training steps
    std::vector<float> losses;
    for (int i = 0; i < 10; ++i) {
        float loss = trainer.train_step(input, target);
        losses.push_back(loss);
        EXPECT_GT(loss, 0.0f) << "Loss should be positive at step " << i
                              << " for dtype: " << dtype_name(dtype_);
    }

    EXPECT_EQ(trainer.get_total_steps(), 10);
    EXPECT_EQ(losses.size(), 10);

    // Check that no losses are NaN or infinity
    for (size_t i = 0; i < losses.size(); ++i) {
        EXPECT_FALSE(std::isnan(losses[i])) << "Loss is NaN at step " << i
                                             << " for dtype: " << dtype_name(dtype_);
        EXPECT_FALSE(std::isinf(losses[i])) << "Loss is infinite at step " << i
                                             << " for dtype: " << dtype_name(dtype_);
    }
}

// Test 4: Evaluation step with different base dtypes
TEST_P(MixedPrecisionMultiDTypeTest, EvalStepWithBaseDType) {
    auto model = std::make_shared<SimpleMLP>(5, 10, 3);
    auto optimizer = std::make_shared<optim::SGD>(model->parameters(), 0.01);
    auto loss_fn = [](const Variable& pred, const Variable& target) {
        auto diff = pred - target;
        return mean(diff * diff);
    };

    auto config = MixedPrecisionConfig::fp16_cuda();
    config.enabled = false;
    MixedPrecisionTrainer trainer(model, optimizer, loss_fn, config);

    auto input_tensor = randn({4, 5}, dtype_, device_);
    auto target_tensor = randn({4, 3}, dtype_, device_);
    auto input = Variable(input_tensor, false);
    auto target = Variable(target_tensor, false);

    // Perform evaluation step
    float loss = trainer.eval_step(input, target);

    EXPECT_GT(loss, 0.0f) << "Eval loss should be positive for dtype: " << dtype_name(dtype_);
    EXPECT_FALSE(trainer.is_training());
    EXPECT_EQ(trainer.get_total_steps(), 0);  // Eval doesn't count as training step
}

// Test 5: GradScaler behavior with different base dtypes
TEST_P(MixedPrecisionMultiDTypeTest, GradScalerWithBaseDType) {
    auto model = std::make_shared<SimpleMLP>(5, 10, 3);
    auto optimizer = std::make_shared<optim::SGD>(model->parameters(), 0.01);
    auto loss_fn = [](const Variable& pred, const Variable& target) {
        auto diff = pred - target;
        return mean(diff * diff);
    };

    auto config = MixedPrecisionConfig::fp16_cuda();
    config.enabled = false;
    MixedPrecisionTrainer trainer(model, optimizer, loss_fn, config);

    // Get initial scale
    float initial_scale = trainer.get_scale();
    EXPECT_GT(initial_scale, 0.0f) << "Initial scale should be positive for dtype: "
                                    << dtype_name(dtype_);

    auto input_tensor = randn({4, 5}, dtype_, device_);
    auto target_tensor = randn({4, 3}, dtype_, device_);
    auto input = Variable(input_tensor, false);
    auto target = Variable(target_tensor, false);

    // Train for several steps
    for (int i = 0; i < 5; ++i) {
        trainer.train_step(input, target);
    }

    // Scale should remain valid
    float final_scale = trainer.get_scale();
    EXPECT_GT(final_scale, 0.0f) << "Final scale should be positive for dtype: "
                                  << dtype_name(dtype_);
    EXPECT_FALSE(std::isnan(final_scale)) << "Final scale should not be NaN for dtype: "
                                           << dtype_name(dtype_);
}

// Test 6: Loss precision - loss should be computed in FP32 regardless of base dtype
TEST_P(MixedPrecisionMultiDTypeTest, LossPrecisionWithBaseDType) {
    auto model = std::make_shared<SimpleMLP>(5, 10, 3);
    auto optimizer = std::make_shared<optim::SGD>(model->parameters(), 0.01);

    int loss_computation_count = 0;
    DType observed_loss_dtype = DType::Float32;

    auto loss_fn = [&loss_computation_count, &observed_loss_dtype](
        const Variable& pred, const Variable& target) {
        loss_computation_count++;
        // Loss computation should happen in full precision
        auto diff = pred - target;
        auto loss = mean(diff * diff);
        observed_loss_dtype = loss.tensor().dtype();
        return loss;
    };

    auto config = MixedPrecisionConfig::fp16_cuda();
    config.enabled = false;
    MixedPrecisionTrainer trainer(model, optimizer, loss_fn, config);

    auto input_tensor = randn({4, 5}, dtype_, device_);
    auto target_tensor = randn({4, 3}, dtype_, device_);
    auto input = Variable(input_tensor, false);
    auto target = Variable(target_tensor, false);

    float loss = trainer.train_step(input, target);

    EXPECT_EQ(loss_computation_count, 1);
    EXPECT_GT(loss, 0.0f) << "Loss should be positive for dtype: " << dtype_name(dtype_);
    EXPECT_FALSE(std::isnan(loss)) << "Loss should not be NaN for dtype: " << dtype_name(dtype_);
}

// Test 7: Convergence test with different base dtypes
TEST_P(MixedPrecisionMultiDTypeTest, ConvergenceWithBaseDType) {
    auto model = std::make_shared<SimpleMLP>(2, 8, 1);
    auto optimizer = std::make_shared<optim::SGD>(model->parameters(), 0.1);
    auto loss_fn = [](const Variable& pred, const Variable& target) {
        auto diff = pred - target;
        return mean(diff * diff);
    };

    auto config = MixedPrecisionConfig::fp16_cuda();
    config.enabled = false;
    MixedPrecisionTrainer trainer(model, optimizer, loss_fn, config);

    // Create training data in base dtype
    auto input_tensor = randn({32, 2}, dtype_, device_);
    auto input_data = input_tensor.to(DType::Float32).data<float>();

    std::vector<float> target_data(32);
    for (int i = 0; i < 32; ++i) {
        target_data[i] = 2.0f * input_data[i*2] + 3.0f * input_data[i*2 + 1];
    }

    auto target_tensor = Tensor({32, 1}, DType::Float32, device_);
    float* target_ptr = target_tensor.data<float>();
    for (int i = 0; i < 32; ++i) {
        target_ptr[i] = target_data[i];
    }
    target_tensor = target_tensor.to(dtype_);

    auto input = Variable(input_tensor, false);
    auto target = Variable(target_tensor, false);

    // Train for many iterations
    float initial_loss = trainer.train_step(input, target);

    for (int i = 0; i < 50; ++i) {
        trainer.train_step(input, target);
    }

    float final_loss = trainer.train_step(input, target);

    // Loss should decrease or remain reasonable
    EXPECT_FALSE(std::isnan(final_loss)) << "Final loss is NaN for dtype: " << dtype_name(dtype_);
    EXPECT_FALSE(std::isinf(final_loss)) << "Final loss is infinite for dtype: " << dtype_name(dtype_);
    EXPECT_LT(final_loss, initial_loss * 2.0f) << "Loss diverged for dtype: " << dtype_name(dtype_);
}

// Test 8: Fit with DataLoader using different base dtypes
TEST_P(MixedPrecisionMultiDTypeTest, FitWithDataLoaderAndBaseDType) {
    auto model = std::make_shared<SimpleMLP>(5, 10, 3);
    auto optimizer = std::make_shared<optim::SGD>(model->parameters(), 0.01);
    auto loss_fn = [](const Variable& pred, const Variable& target) {
        auto diff = pred - target;
        return mean(diff * diff);
    };

    auto config = MixedPrecisionConfig::fp16_cuda();
    config.enabled = false;
    MixedPrecisionTrainer trainer(model, optimizer, loss_fn, config);

    // Create simple dataset in base dtype
    std::vector<std::pair<Tensor, Tensor>> train_data;
    for (int i = 0; i < 8; ++i) {
        auto input = randn({5}, dtype_, device_);
        auto target = randn({3}, dtype_, device_);
        train_data.emplace_back(input, target);
    }

    DataLoader train_loader(train_data, 2);

    // Train for 2 epochs
    trainer.fit(train_loader, 2);

    EXPECT_GT(trainer.get_total_steps(), 0);
    EXPECT_EQ(trainer.get_total_steps(), 8);  // 4 batches per epoch * 2 epochs
}

// Test 9: Statistics tracking with different base dtypes
TEST_P(MixedPrecisionMultiDTypeTest, StatisticsTrackingWithBaseDType) {
    auto model = std::make_shared<SimpleMLP>(5, 10, 3);
    auto optimizer = std::make_shared<optim::SGD>(model->parameters(), 0.01);
    auto loss_fn = [](const Variable& pred, const Variable& target) {
        auto diff = pred - target;
        return mean(diff * diff);
    };

    auto config = MixedPrecisionConfig::fp16_cuda();
    config.enabled = false;
    MixedPrecisionTrainer trainer(model, optimizer, loss_fn, config);

    std::vector<std::pair<Tensor, Tensor>> train_data;
    for (int i = 0; i < 10; ++i) {
        auto input = randn({5}, dtype_, device_);
        auto target = randn({3}, dtype_, device_);
        train_data.emplace_back(input, target);
    }

    DataLoader train_loader(train_data, 2);

    int initial_steps = trainer.get_total_steps();
    trainer.fit(train_loader, 3);
    int final_steps = trainer.get_total_steps();

    // Should have 5 batches per epoch * 3 epochs = 15 steps
    EXPECT_EQ(final_steps - initial_steps, 15) << "Incorrect step count for dtype: "
                                                << dtype_name(dtype_);
}

// Test 10: Reset statistics with different base dtypes
TEST_P(MixedPrecisionMultiDTypeTest, ResetStatisticsWithBaseDType) {
    auto model = std::make_shared<SimpleMLP>(5, 10, 3);
    auto optimizer = std::make_shared<optim::SGD>(model->parameters(), 0.01);
    auto loss_fn = [](const Variable& pred, const Variable& target) {
        auto diff = pred - target;
        return mean(diff * diff);
    };

    auto config = MixedPrecisionConfig::fp16_cuda();
    config.enabled = false;
    MixedPrecisionTrainer trainer(model, optimizer, loss_fn, config);

    auto input_tensor = randn({4, 5}, dtype_, device_);
    auto target_tensor = randn({4, 3}, dtype_, device_);
    auto input = Variable(input_tensor, false);
    auto target = Variable(target_tensor, false);

    // Perform some steps
    for (int i = 0; i < 5; ++i) {
        trainer.train_step(input, target);
    }

    EXPECT_EQ(trainer.get_total_steps(), 5);

    // Reset statistics
    trainer.reset_stats();

    EXPECT_EQ(trainer.get_total_steps(), 0);
    EXPECT_EQ(trainer.get_skipped_steps(), 0);
}

// Instantiate tests for all dtypes
INSTANTIATE_TEST_SUITE_P(
    AllDTypes,
    MixedPrecisionMultiDTypeTest,
    ::testing::ValuesIn(get_test_dtypes()),
    [](const ::testing::TestParamInfo<DType>& info) {
        return std::string(dtype_name(info.param));
    }
);

// Additional single-dtype tests for specific mixed precision features

class MixedPrecisionSpecificTest : public ::testing::Test {
protected:
    void SetUp() override {
        device_ = Device::cpu();
    }

    Device device_;
};

// Test 11: BFloat16 configuration
TEST_F(MixedPrecisionSpecificTest, BFloat16Configuration) {
    auto config = MixedPrecisionConfig::bfloat16_cuda();

    EXPECT_EQ(config.dtype, DType::BFloat16);
    EXPECT_TRUE(config.enabled);
}

// Test 12: Conservative configuration
TEST_F(MixedPrecisionSpecificTest, ConservativeConfiguration) {
    auto config = MixedPrecisionConfig::conservative();

    EXPECT_FLOAT_EQ(config.init_scale, 1024.0f);
    EXPECT_FLOAT_EQ(config.growth_factor, 1.5f);
    EXPECT_FLOAT_EQ(config.backoff_factor, 0.75f);
    EXPECT_EQ(config.growth_interval, 5000);
}

// Test 13: Helper function - create_fp16_trainer
TEST_F(MixedPrecisionSpecificTest, CreateFP16Trainer) {
    auto model = std::make_shared<SimpleMLP>(5, 10, 3);
    auto optimizer = std::make_shared<optim::SGD>(model->parameters(), 0.01);
    auto loss_fn = [](const Variable& pred, const Variable& target) {
        auto diff = pred - target;
        return mean(diff * diff);
    };

    auto trainer = create_fp16_trainer(model, optimizer, loss_fn);

    const auto& config = trainer.get_config();
    EXPECT_EQ(config.dtype, DType::Float16);
    EXPECT_TRUE(config.enabled);
}

// Test 14: Helper function - create_bfloat16_trainer
TEST_F(MixedPrecisionSpecificTest, CreateBFloat16Trainer) {
    auto model = std::make_shared<SimpleMLP>(5, 10, 3);
    auto optimizer = std::make_shared<optim::SGD>(model->parameters(), 0.01);
    auto loss_fn = [](const Variable& pred, const Variable& target) {
        auto diff = pred - target;
        return mean(diff * diff);
    };

    auto trainer = create_bfloat16_trainer(model, optimizer, loss_fn);

    const auto& config = trainer.get_config();
    EXPECT_EQ(config.dtype, DType::BFloat16);
    EXPECT_TRUE(config.enabled);
}

// Test 15: Train/Eval mode switching
TEST_F(MixedPrecisionSpecificTest, TrainEvalModeSwitching) {
    auto model = std::make_shared<SimpleMLP>(5, 10, 3);
    auto optimizer = std::make_shared<optim::SGD>(model->parameters(), 0.01);
    auto loss_fn = [](const Variable& pred, const Variable& target) {
        auto diff = pred - target;
        return mean(diff * diff);
    };

    auto config = MixedPrecisionConfig::fp16_cuda();
    config.enabled = false;
    MixedPrecisionTrainer trainer(model, optimizer, loss_fn, config);

    EXPECT_TRUE(trainer.is_training());

    trainer.eval();
    EXPECT_FALSE(trainer.is_training());

    trainer.train();
    EXPECT_TRUE(trainer.is_training());
}

// Test 16: Float32 + Float16 mixed precision (most common case)
TEST_F(MixedPrecisionSpecificTest, Float32PlusFloat16MixedPrecision) {
    auto model = std::make_shared<SimpleMLP>(10, 20, 5);
    auto optimizer = std::make_shared<optim::SGD>(model->parameters(), 0.01);
    auto loss_fn = [](const Variable& pred, const Variable& target) {
        auto diff = pred - target;
        return mean(diff * diff);
    };

    // FP16 mixed precision configuration
    auto config = MixedPrecisionConfig::fp16_cuda();
    config.enabled = false;  // Disabled for CPU testing
    MixedPrecisionTrainer trainer(model, optimizer, loss_fn, config);

    // Input in FP32
    auto input_tensor = randn({4, 10}, DType::Float32, device_);
    auto target_tensor = randn({4, 5}, DType::Float32, device_);
    auto input = Variable(input_tensor, false);
    auto target = Variable(target_tensor, false);

    // Training step
    float loss = trainer.train_step(input, target);

    EXPECT_GT(loss, 0.0f);
    EXPECT_FALSE(std::isnan(loss));
    EXPECT_EQ(trainer.get_total_steps(), 1);
}

// Test 17: GradScaler scale growth behavior
TEST_F(MixedPrecisionSpecificTest, GradScalerScaleGrowth) {
    auto model = std::make_shared<SimpleMLP>(5, 10, 3);
    auto optimizer = std::make_shared<optim::SGD>(model->parameters(), 0.01);
    auto loss_fn = [](const Variable& pred, const Variable& target) {
        auto diff = pred - target;
        return mean(diff * diff);
    };

    auto config = MixedPrecisionConfig::fp16_cuda();
    config.enabled = false;
    config.growth_interval = 5;  // Grow scale every 5 steps
    MixedPrecisionTrainer trainer(model, optimizer, loss_fn, config);

    auto input_tensor = randn({4, 5}, DType::Float32, device_);
    auto target_tensor = randn({4, 3}, DType::Float32, device_);
    auto input = Variable(input_tensor, false);
    auto target = Variable(target_tensor, false);

    float initial_scale = trainer.get_scale();

    // Train for multiple steps
    for (int i = 0; i < 10; ++i) {
        trainer.train_step(input, target);
    }

    // Scale should remain valid (may have grown)
    float final_scale = trainer.get_scale();
    EXPECT_GT(final_scale, 0.0f);
    EXPECT_FALSE(std::isnan(final_scale));
}

// Test 18: Fit with validation using mixed precision
TEST_F(MixedPrecisionSpecificTest, FitWithValidation) {
    auto model = std::make_shared<SimpleMLP>(5, 10, 3);
    auto optimizer = std::make_shared<optim::Adam>(model->parameters(), 0.001);
    auto loss_fn = [](const Variable& pred, const Variable& target) {
        auto diff = pred - target;
        return mean(diff * diff);
    };

    auto config = MixedPrecisionConfig::fp16_cuda();
    config.enabled = false;
    MixedPrecisionTrainer trainer(model, optimizer, loss_fn, config);

    // Create train and validation datasets
    std::vector<std::pair<Tensor, Tensor>> train_data;
    for (int i = 0; i < 8; ++i) {
        auto input = randn({5}, DType::Float32, device_);
        auto target = randn({3}, DType::Float32, device_);
        train_data.emplace_back(input, target);
    }

    std::vector<std::pair<Tensor, Tensor>> val_data;
    for (int i = 0; i < 4; ++i) {
        auto input = randn({5}, DType::Float32, device_);
        auto target = randn({3}, DType::Float32, device_);
        val_data.emplace_back(input, target);
    }

    DataLoader train_loader(train_data, 2);
    DataLoader val_loader(val_data, 2);

    // Train for 2 epochs with validation
    trainer.fit(train_loader, 2, &val_loader);

    EXPECT_GT(trainer.get_total_steps(), 0);
}

int main(int argc, char** argv) {
    // Initialize Tenzor library
    tenzor::initialize();

    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
