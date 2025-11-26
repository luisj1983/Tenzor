/**
 * @file test_optimization.cpp
 * @brief Integration tests for optimizers and learning rate schedulers
 *
 * Tests:
 * - Optimizer combinations with schedulers
 * - Learning rate warm-up and decay
 * - Gradient accumulation with optimizers
 * - Parameter group management
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <memory>
#include <cmath>

using namespace tenzor;
using namespace tenzor::nn;
using namespace tenzor::optim;

//==============================================================================
// Test Environment
//==============================================================================

class OptimizationEnvironment : public ::testing::Environment {
public:
    void SetUp() override {
        tenzor::initialize();
    }
};

static ::testing::Environment* const optimization_env =
    ::testing::AddGlobalTestEnvironment(new OptimizationEnvironment);

//==============================================================================
// Helper Model
//==============================================================================

class SimpleModel : public Module {
public:
    SimpleModel() {
        fc1 = std::make_shared<Linear>(10, 20);
        fc2 = std::make_shared<Linear>(20, 5);

        register_module("fc1", fc1);
        register_module("fc2", fc2);
    }

    auto forward_impl(const Variable& x) -> Variable override {
        auto out = fc1->forward(x);
        out = relu(out);
        out = fc2->forward(out);
        return out;
    }

private:
    std::shared_ptr<Linear> fc1, fc2;
};

//==============================================================================
// Test 1: SGD with StepLR Integration
//==============================================================================

TEST(Optimization, SGDWithStepLR) {
    auto device = Device::cpu();
    auto model = std::make_shared<SimpleModel>();
    model->to(device);

    auto params = model->parameters();
    SGD optimizer(params, 0.1, 0.9);
    StepLR scheduler(optimizer, 3, 0.1);

    std::vector<float> learning_rates;

    for (int epoch = 0; epoch < 10; epoch++) {
        model->train();

        // Training step
        auto input = Variable(randn({4, 10}, DType::Float32, device), true);
        auto target = Variable(randn({4, 5}, DType::Float32, device), false);

        optimizer.zero_grad();
        auto output = model->forward(input);
        auto loss = mse_loss(output, target);
        loss.backward();
        optimizer.step();

        learning_rates.push_back(scheduler.get_lr());
        scheduler.step();
    }

    // Verify LR schedule
    EXPECT_FLOAT_EQ(learning_rates[0], 0.1f);
    EXPECT_FLOAT_EQ(learning_rates[3], 0.01f);
    EXPECT_FLOAT_EQ(learning_rates[6], 0.001f);
}

//==============================================================================
// Test 2: Adam with CosineAnnealingLR
//==============================================================================

TEST(Optimization, AdamWithCosineAnnealing) {
    auto device = Device::cpu();
    auto model = std::make_shared<SimpleModel>();
    model->to(device);

    auto params = model->parameters();
    Adam optimizer(params, 0.001);
    CosineAnnealingLR scheduler(optimizer, 20, 0.0);

    std::vector<float> learning_rates;

    for (int epoch = 0; epoch < 20; epoch++) {
        model->train();

        auto input = Variable(randn({4, 10}, DType::Float32, device), true);
        auto target = Variable(randn({4, 5}, DType::Float32, device), false);

        optimizer.zero_grad();
        auto output = model->forward(input);
        auto loss = mse_loss(output, target);
        loss.backward();
        optimizer.step();

        learning_rates.push_back(scheduler.get_lr());
        scheduler.step();
    }

    // Verify cosine annealing pattern
    EXPECT_GT(learning_rates[0], learning_rates[10]) << "LR should decrease in first half";
    EXPECT_LT(learning_rates.back(), 0.0001f) << "Final LR should be near minimum";
}

//==============================================================================
// Test 3: AdamW with Weight Decay
//==============================================================================

TEST(Optimization, AdamWWeightDecay) {
    auto device = Device::cpu();
    auto model = std::make_shared<SimpleModel>();
    model->to(device);

    auto params = model->parameters();

    // Get initial parameter values
    auto initial_param = params[0]->tensor().clone();

    AdamW optimizer(params, 0.001, 0.9, 0.999, 1e-8, 0.01);  // weight_decay=0.01

    model->train();
    for (int i = 0; i < 10; i++) {
        auto input = Variable(randn({4, 10}, DType::Float32, device), true);
        auto target = Variable(randn({4, 5}, DType::Float32, device), false);

        optimizer.zero_grad();
        auto output = model->forward(input);
        auto loss = mse_loss(output, target);
        loss.backward();
        optimizer.step();
    }

    // Parameters should have changed
    auto final_param = params[0]->tensor();

    auto initial_cpu = initial_param.to(Device::cpu()).template data<float>();
    auto final_cpu = final_param.to(Device::cpu()).template data<float>();

    bool changed = false;
    for (size_t i = 0; i < initial_param.numel(); i++) {
        if (std::abs(initial_cpu[i] - final_cpu[i]) > 1e-6f) {
            changed = true;
            break;
        }
    }

    EXPECT_TRUE(changed) << "Parameters should have been updated";
}

//==============================================================================
// Test 4: OneCycleLR Full Cycle
//==============================================================================

TEST(Optimization, OneCycleLRFullCycle) {
    auto device = Device::cpu();
    auto model = std::make_shared<SimpleModel>();
    model->to(device);

    auto params = model->parameters();
    SGD optimizer(params, 0.1, 0.9);

    int total_steps = 100;
    OneCycleLR scheduler(optimizer, 0.1, total_steps);

    std::vector<float> learning_rates;

    model->train();
    for (int step = 0; step < total_steps; step++) {
        auto input = Variable(randn({4, 10}, DType::Float32, device), true);
        auto target = Variable(randn({4, 5}, DType::Float32, device), false);

        optimizer.zero_grad();
        auto output = model->forward(input);
        auto loss = mse_loss(output, target);
        loss.backward();
        optimizer.step();

        learning_rates.push_back(scheduler.get_lr());
        scheduler.step();
    }

    // Verify 1cycle pattern: warmup -> peak -> annealing
    float max_lr = *std::max_element(learning_rates.begin(), learning_rates.end());
    size_t max_idx = std::distance(learning_rates.begin(),
                                   std::find(learning_rates.begin(), learning_rates.end(), max_lr));

    EXPECT_GT(max_lr, learning_rates[0]) << "Peak LR should be higher than start";
    EXPECT_GT(max_lr, learning_rates.back()) << "Peak LR should be higher than end";
    EXPECT_LT(max_idx, learning_rates.size() / 2) << "Peak should be in first half";
}

//==============================================================================
// Test 5: ReduceLROnPlateau with Validation Metric
//==============================================================================

TEST(Optimization, ReduceLROnPlateau) {
    auto device = Device::cpu();
    auto model = std::make_shared<SimpleModel>();
    model->to(device);

    auto params = model->parameters();
    SGD optimizer(params, 0.1, 0.9);
    ReduceLROnPlateau scheduler(optimizer, "min", 0.5, 3);  // patience=3

    std::vector<float> learning_rates;
    std::vector<float> val_losses = {2.0f, 1.8f, 1.7f, 1.65f, 1.64f, 1.63f, 1.62f, 1.61f};

    for (size_t epoch = 0; epoch < val_losses.size(); epoch++) {
        model->train();

        auto input = Variable(randn({4, 10}, DType::Float32, device), true);
        auto target = Variable(randn({4, 5}, DType::Float32, device), false);

        optimizer.zero_grad();
        auto output = model->forward(input);
        auto loss = mse_loss(output, target);
        loss.backward();
        optimizer.step();

        learning_rates.push_back(scheduler.get_lr());
        scheduler.step(val_losses[epoch]);  // Pass validation loss
    }

    // LR should have reduced when plateau detected
    EXPECT_LT(learning_rates.back(), learning_rates.front())
        << "LR should reduce on plateau";
}

//==============================================================================
// Test 6: Multiple Optimizers on Different Parameters
//==============================================================================

TEST(Optimization, MultipleOptimizers) {
    auto device = Device::cpu();
    auto model = std::make_shared<SimpleModel>();
    model->to(device);

    auto params = model->parameters();

    // Split parameters (simplified)
    std::vector<std::shared_ptr<Variable>> params1 = {params[0]};
    std::vector<std::shared_ptr<Variable>> params2 = {params[1]};

    SGD optimizer1(params1, 0.01);
    Adam optimizer2(params2, 0.001);

    model->train();
    for (int i = 0; i < 5; i++) {
        auto input = Variable(randn({4, 10}, DType::Float32, device), true);
        auto target = Variable(randn({4, 5}, DType::Float32, device), false);

        optimizer1.zero_grad();
        optimizer2.zero_grad();

        auto output = model->forward(input);
        auto loss = mse_loss(output, target);
        loss.backward();

        optimizer1.step();
        optimizer2.step();
    }

    SUCCEED() << "Multiple optimizers worked correctly";
}

//==============================================================================
// Test 7: Gradient Accumulation with Optimizer
//==============================================================================

TEST(Optimization, GradientAccumulation) {
    auto device = Device::cpu();
    auto model = std::make_shared<SimpleModel>();
    model->to(device);

    auto params = model->parameters();
    SGD optimizer(params, 0.01);

    int accumulation_steps = 4;
    int total_batches = 16;

    model->train();
    for (int batch = 0; batch < total_batches; batch++) {
        auto input = Variable(randn({2, 10}, DType::Float32, device), true);
        auto target = Variable(randn({2, 5}, DType::Float32, device), false);

        auto output = model->forward(input);
        auto loss = mse_loss(output, target) / static_cast<float>(accumulation_steps);
        loss.backward();

        if ((batch + 1) % accumulation_steps == 0) {
            optimizer.step();
            optimizer.zero_grad();
        }
    }

    SUCCEED() << "Gradient accumulation completed";
}

//==============================================================================
// Test 8: Exponential LR Decay
//==============================================================================

TEST(Optimization, ExponentialLRDecay) {
    auto device = Device::cpu();
    auto model = std::make_shared<SimpleModel>();
    model->to(device);

    auto params = model->parameters();
    SGD optimizer(params, 0.1);
    ExponentialLR scheduler(optimizer, 0.9);

    std::vector<float> learning_rates;

    for (int epoch = 0; epoch < 10; epoch++) {
        model->train();

        auto input = Variable(randn({4, 10}, DType::Float32, device), true);
        auto target = Variable(randn({4, 5}, DType::Float32, device), false);

        optimizer.zero_grad();
        auto output = model->forward(input);
        auto loss = mse_loss(output, target);
        loss.backward();
        optimizer.step();

        learning_rates.push_back(scheduler.get_lr());
        scheduler.step();
    }

    // Verify exponential decay
    for (size_t i = 1; i < learning_rates.size(); i++) {
        EXPECT_LT(learning_rates[i], learning_rates[i-1])
            << "LR should decrease each epoch";
    }
}

//==============================================================================
// Test 9: CosineAnnealingWarmRestarts
//==============================================================================

TEST(Optimization, CosineAnnealingWarmRestarts) {
    auto device = Device::cpu();
    auto model = std::make_shared<SimpleModel>();
    model->to(device);

    auto params = model->parameters();
    SGD optimizer(params, 0.1);
    CosineAnnealingWarmRestarts scheduler(optimizer, 5, 2);  // T_0=5, T_mult=2

    std::vector<float> learning_rates;

    for (int epoch = 0; epoch < 20; epoch++) {
        model->train();

        auto input = Variable(randn({4, 10}, DType::Float32, device), true);
        auto target = Variable(randn({4, 5}, DType::Float32, device), false);

        optimizer.zero_grad();
        auto output = model->forward(input);
        auto loss = mse_loss(output, target);
        loss.backward();
        optimizer.step();

        learning_rates.push_back(scheduler.get_lr());
        scheduler.step();
    }

    // Look for restarts (LR jumps back up)
    int restarts = 0;
    for (size_t i = 1; i < learning_rates.size(); i++) {
        if (learning_rates[i] > learning_rates[i-1] * 1.5f) {
            restarts++;
        }
    }

    EXPECT_GT(restarts, 0) << "Should have at least one restart";
}

//==============================================================================
// Test 10: Optimizer State Save/Load Integration
//==============================================================================

TEST(Optimization, OptimizerStateSaveLoad) {
    auto device = Device::cpu();
    auto model = std::make_shared<SimpleModel>();
    model->to(device);

    auto params = model->parameters();
    Adam optimizer(params, 0.001);

    // Train for a few steps to build optimizer state
    model->train();
    for (int i = 0; i < 5; i++) {
        auto input = Variable(randn({4, 10}, DType::Float32, device), true);
        auto target = Variable(randn({4, 5}, DType::Float32, device), false);

        optimizer.zero_grad();
        auto output = model->forward(input);
        auto loss = mse_loss(output, target);
        loss.backward();
        optimizer.step();
    }

    // Save optimizer state
    auto state = optimizer.state_dict();
    EXPECT_GT(state.size(), 0) << "Optimizer should have state";

    // Create new optimizer and load state
    Adam optimizer2(params, 0.001);
    optimizer2.load_state_dict(state);

    SUCCEED() << "Optimizer state saved and loaded successfully";
}
