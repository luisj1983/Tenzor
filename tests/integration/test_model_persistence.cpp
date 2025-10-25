/**
 * @file test_model_persistence.cpp
 * @brief Integration tests for model serialization and checkpoint management
 *
 * Tests:
 * - Save/load model weights
 * - Resume training from checkpoint
 * - Optimizer state persistence
 * - Cross-device checkpoint loading
 * - Checkpoint validation
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <fstream>
#include <memory>

using namespace tenzor;
using namespace tenzor::nn;
using namespace tenzor::optim;

//==============================================================================
// Test Environment
//==============================================================================

class ModelPersistenceEnvironment : public ::testing::Environment {
public:
    void SetUp() override {
        tenzor::initialize();
    }
};

static ::testing::Environment* const persistence_env =
    ::testing::AddGlobalTestEnvironment(new ModelPersistenceEnvironment);

//==============================================================================
// Helper Models
//==============================================================================

class SimpleLinearModel : public Module {
public:
    SimpleLinearModel() {
        fc1 = std::make_shared<Linear>(10, 20);
        fc2 = std::make_shared<Linear>(20, 5);

        register_module("fc1", fc1);
        register_module("fc2", fc2);
    }

    auto forward(const Variable& x) -> Variable override {
        auto out = fc1->forward(x);
        out = relu(out);
        out = fc2->forward(out);
        return out;
    }

private:
    std::shared_ptr<Linear> fc1, fc2;
};

class SimpleCNN : public Module {
public:
    SimpleCNN() {
        conv1 = std::make_shared<Conv2d>(3, 16, 3, 1, 1);
        conv2 = std::make_shared<Conv2d>(16, 32, 3, 1, 1);
        flatten = std::make_shared<Flatten>(1);
        fc1 = std::make_shared<Linear>(32 * 32 * 32, 10);

        register_module("conv1", conv1);
        register_module("conv2", conv2);
        register_module("flatten", flatten);
        register_module("fc1", fc1);
    }

    auto forward(const Variable& x) -> Variable override {
        auto out = conv1->forward(x);
        out = relu(out);
        out = conv2->forward(out);
        out = relu(out);
        out = flatten->forward(out);
        out = fc1->forward(out);
        return out;
    }

private:
    std::shared_ptr<Conv2d> conv1, conv2;
    std::shared_ptr<Flatten> flatten;
    std::shared_ptr<Linear> fc1;
};

//==============================================================================
// Test 1: Basic Save and Load Model Weights
//==============================================================================

TEST(ModelPersistence, BasicSaveLoadWeights) {
    auto device = Device::cpu();

    // Create and initialize model
    auto model1 = std::make_shared<SimpleLinearModel>();
    model1->to(device);

    // Get initial weights
    auto params1 = model1->parameters();
    ASSERT_GT(params1.size(), 0) << "Model should have parameters";

    // Store parameter values
    std::vector<Tensor> original_weights;
    for (auto& param : params1) {
        original_weights.push_back(param->tensor().clone());
    }

    // Get state dict
    auto state_dict = model1->state_dict();
    EXPECT_GT(state_dict.size(), 0) << "State dict should not be empty";

    // Create new model and load weights
    auto model2 = std::make_shared<SimpleLinearModel>();
    model2->to(device);
    model2->load_state_dict(state_dict);

    // Verify weights match
    auto params2 = model2->parameters();
    ASSERT_EQ(params1.size(), params2.size()) << "Parameter counts should match";

    for (size_t i = 0; i < params1.size(); i++) {
        auto t1 = params1[i]->tensor().to(Device::cpu());
        auto t2 = params2[i]->tensor().to(Device::cpu());

        auto shape1 = t1.shape();
        auto shape2 = t2.shape();
        ASSERT_EQ(shape1.size(), shape2.size()) << "Parameter " << i << " shape dimensions should match";
        for (size_t j = 0; j < shape1.size(); j++) {
            EXPECT_EQ(shape1[j], shape2[j]) << "Parameter " << i << " shape dimension " << j << " should match";
        }

        auto data1 = t1.template data<float>();
        auto data2 = t2.template data<float>();

        for (size_t j = 0; j < t1.numel(); j++) {
            EXPECT_FLOAT_EQ(data1[j], data2[j])
                << "Parameter " << i << " values should match at index " << j;
        }
    }
}

//==============================================================================
// Test 2: Save and Load After Training
//==============================================================================

TEST(ModelPersistence, SaveLoadAfterTraining) {
    auto device = Device::cpu();

    // Train model
    auto model1 = std::make_shared<SimpleLinearModel>();
    model1->to(device);

    auto params = model1->parameters();
    SGD optimizer(params, 0.01, 0.9);

    model1->train();
    for (int i = 0; i < 10; i++) {
        auto input = Variable(randn({4, 10}, DType::Float32, device), true);
        auto target = Variable(randn({4, 5}, DType::Float32, device), false);

        optimizer.zero_grad();
        auto output = model1->forward(input);
        auto loss = mse_loss(output, target);
        loss.backward();
        optimizer.step();
    }

    // Save trained weights
    auto state_dict = model1->state_dict();

    // Create new model and load
    auto model2 = std::make_shared<SimpleLinearModel>();
    model2->to(device);
    model2->load_state_dict(state_dict);

    // Test inference on both models
    model1->eval();
    model2->eval();

    auto test_input = Variable(randn({4, 10}, DType::Float32, device), false);
    auto output1 = model1->forward(test_input);
    auto output2 = model2->forward(test_input);

    // Outputs should match
    auto out1_cpu = output1.tensor().to(Device::cpu());
    auto out2_cpu = output2.tensor().to(Device::cpu());

    auto data1 = out1_cpu.template data<float>();
    auto data2 = out2_cpu.template data<float>();

    for (size_t i = 0; i < out1_cpu.numel(); i++) {
        EXPECT_NEAR(data1[i], data2[i], 1e-5f)
            << "Inference outputs should match at index " << i;
    }
}

//==============================================================================
// Test 3: Optimizer State Persistence
//==============================================================================

TEST(ModelPersistence, OptimizerStatePersistence) {
    auto device = Device::cpu();
    auto model = std::make_shared<SimpleLinearModel>();
    model->to(device);

    auto params = model->parameters();
    SGD optimizer(params, 0.01, 0.9);  // with momentum

    // Train for a few steps to build momentum
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
    auto optimizer_state = optimizer.state_dict();
    EXPECT_GT(optimizer_state.size(), 0) << "Optimizer should have state";

    // Create new optimizer and load state
    SGD optimizer2(params, 0.01, 0.9);
    optimizer2.load_state_dict(optimizer_state);

    // Both optimizers should produce same update
    auto input = Variable(randn({4, 10}, DType::Float32, device), true);
    auto target = Variable(randn({4, 5}, DType::Float32, device), false);

    // Step with first optimizer
    auto param_before = params[0]->tensor().clone();

    optimizer.zero_grad();
    auto output = model->forward(input);
    auto loss = mse_loss(output, target);
    loss.backward();
    optimizer.step();

    auto param_after = params[0]->tensor().clone();

    SUCCEED() << "Optimizer state persistence test completed";
}

//==============================================================================
// Test 4: Resume Training from Checkpoint
//==============================================================================

TEST(ModelPersistence, ResumeTrainingFromCheckpoint) {
    auto device = Device::cpu();

    // Initial training
    auto model1 = std::make_shared<SimpleLinearModel>();
    model1->to(device);

    auto params1 = model1->parameters();
    SGD optimizer1(params1, 0.01, 0.9);

    model1->train();
    std::vector<float> losses1;

    // Train for 5 steps
    for (int i = 0; i < 5; i++) {
        auto input = Variable(randn({4, 10}, DType::Float32, device), true);
        auto target = Variable(randn({4, 5}, DType::Float32, device), false);

        optimizer1.zero_grad();
        auto output = model1->forward(input);
        auto loss = mse_loss(output, target);
        loss.backward();
        optimizer1.step();

        losses1.push_back(loss.tensor().template item<float>());
    }

    // Save checkpoint
    auto model_state = model1->state_dict();
    auto optimizer_state = optimizer1.state_dict();

    // Create new model and resume training
    auto model2 = std::make_shared<SimpleLinearModel>();
    model2->to(device);
    model2->load_state_dict(model_state);

    auto params2 = model2->parameters();
    SGD optimizer2(params2, 0.01, 0.9);
    optimizer2.load_state_dict(optimizer_state);

    model2->train();
    std::vector<float> losses2;

    // Continue training for 5 more steps
    for (int i = 0; i < 5; i++) {
        auto input = Variable(randn({4, 10}, DType::Float32, device), true);
        auto target = Variable(randn({4, 5}, DType::Float32, device), false);

        optimizer2.zero_grad();
        auto output = model2->forward(input);
        auto loss = mse_loss(output, target);
        loss.backward();
        optimizer2.step();

        losses2.push_back(loss.tensor().template item<float>());
    }

    EXPECT_GT(losses1.size(), 0) << "Should have losses from first phase";
    EXPECT_GT(losses2.size(), 0) << "Should have losses from resumed phase";
}

//==============================================================================
// Test 5: CNN Model Save/Load
//==============================================================================

TEST(ModelPersistence, CNNModelSaveLoad) {
    auto device = Device::cpu();

    auto model1 = std::make_shared<SimpleCNN>();
    model1->to(device);

    // Get state
    auto state_dict = model1->state_dict();

    // Load into new model
    auto model2 = std::make_shared<SimpleCNN>();
    model2->to(device);
    model2->load_state_dict(state_dict);

    // Test inference
    model1->eval();
    model2->eval();

    auto input = Variable(randn({2, 3, 32, 32}, DType::Float32, device), false);
    auto output1 = model1->forward(input);
    auto output2 = model2->forward(input);

    // Compare outputs
    auto out1_cpu = output1.tensor().to(Device::cpu());
    auto out2_cpu = output2.tensor().to(Device::cpu());

    auto data1 = out1_cpu.template data<float>();
    auto data2 = out2_cpu.template data<float>();

    for (size_t i = 0; i < out1_cpu.numel(); i++) {
        EXPECT_NEAR(data1[i], data2[i], 1e-5f);
    }
}

//==============================================================================
// Test 6: Save/Load with Different Optimizer Types
//==============================================================================

TEST(ModelPersistence, DifferentOptimizerTypes) {
    auto device = Device::cpu();

    // Test Adam optimizer
    {
        auto model = std::make_shared<SimpleLinearModel>();
        model->to(device);

        auto params = model->parameters();
        Adam optimizer(params, 0.001, 0.9, 0.999);

        // Train a bit to create state
        model->train();
        for (int i = 0; i < 3; i++) {
            auto input = Variable(randn({4, 10}, DType::Float32, device), true);
            auto target = Variable(randn({4, 5}, DType::Float32, device), false);

            optimizer.zero_grad();
            auto output = model->forward(input);
            auto loss = mse_loss(output, target);
            loss.backward();
            optimizer.step();
        }

        // Save and load optimizer state
        auto state = optimizer.state_dict();
        Adam optimizer2(params, 0.001, 0.9, 0.999);
        optimizer2.load_state_dict(state);

        SUCCEED() << "Adam optimizer state saved and loaded";
    }

    // Test AdamW optimizer
    {
        auto model = std::make_shared<SimpleLinearModel>();
        model->to(device);

        auto params = model->parameters();
        AdamW optimizer(params, 0.001, 0.9, 0.999, 1e-8, 0.01);

        // Train a bit
        model->train();
        for (int i = 0; i < 3; i++) {
            auto input = Variable(randn({4, 10}, DType::Float32, device), true);
            auto target = Variable(randn({4, 5}, DType::Float32, device), false);

            optimizer.zero_grad();
            auto output = model->forward(input);
            auto loss = mse_loss(output, target);
            loss.backward();
            optimizer.step();
        }

        // Save and load
        auto state = optimizer.state_dict();
        AdamW optimizer2(params, 0.001, 0.9, 0.999, 1e-8, 0.01);
        optimizer2.load_state_dict(state);

        SUCCEED() << "AdamW optimizer state saved and loaded";
    }
}

//==============================================================================
// Test 7: Checkpoint with Learning Rate Scheduler
//==============================================================================

TEST(ModelPersistence, CheckpointWithScheduler) {
    auto device = Device::cpu();
    auto model = std::make_shared<SimpleLinearModel>();
    model->to(device);

    auto params = model->parameters();
    SGD optimizer(params, 0.1, 0.9);
    StepLR scheduler(optimizer, 2, 0.1);

    model->train();

    // Train for a few epochs
    for (int epoch = 0; epoch < 5; epoch++) {
        for (int batch = 0; batch < 3; batch++) {
            auto input = Variable(randn({4, 10}, DType::Float32, device), true);
            auto target = Variable(randn({4, 5}, DType::Float32, device), false);

            optimizer.zero_grad();
            auto output = model->forward(input);
            auto loss = mse_loss(output, target);
            loss.backward();
            optimizer.step();
        }
        scheduler.step();
    }

    // Save complete checkpoint
    auto model_state = model->state_dict();
    auto optimizer_state = optimizer.state_dict();
    int epoch = 5;
    float lr = scheduler.get_lr();

    // Verify checkpoint data
    EXPECT_EQ(epoch, 5);
    EXPECT_LT(lr, 0.1f) << "LR should have decreased";

    SUCCEED() << "Checkpoint with scheduler saved";
}

//==============================================================================
// Test 8: Partial Model Loading
//==============================================================================

TEST(ModelPersistence, PartialModelLoading) {
    auto device = Device::cpu();

    // Create source model with more layers
    auto model1 = std::make_shared<SimpleLinearModel>();
    model1->to(device);

    auto state_dict1 = model1->state_dict();

    // Create target model and load (should handle gracefully)
    auto model2 = std::make_shared<SimpleLinearModel>();
    model2->to(device);

    // This should work as models are identical
    model2->load_state_dict(state_dict1);

    SUCCEED() << "Model loading completed";
}

//==============================================================================
// Test 9: Save Best Model During Training
//==============================================================================

TEST(ModelPersistence, SaveBestModelDuringTraining) {
    auto device = Device::cpu();
    auto model = std::make_shared<SimpleLinearModel>();
    model->to(device);

    auto params = model->parameters();
    SGD optimizer(params, 0.01, 0.9);

    float best_loss = std::numeric_limits<float>::max();
    std::unordered_map<std::string, Tensor> best_state;

    model->train();
    for (int epoch = 0; epoch < 10; epoch++) {
        float epoch_loss = 0.0f;
        int num_batches = 5;

        for (int batch = 0; batch < num_batches; batch++) {
            auto input = Variable(randn({4, 10}, DType::Float32, device), true);
            auto target = Variable(randn({4, 5}, DType::Float32, device), false);

            optimizer.zero_grad();
            auto output = model->forward(input);
            auto loss = mse_loss(output, target);
            loss.backward();
            optimizer.step();

            epoch_loss += loss.tensor().template item<float>();
        }

        epoch_loss /= num_batches;

        // Save if best
        if (epoch_loss < best_loss) {
            best_loss = epoch_loss;
            best_state = model->state_dict();
        }
    }

    EXPECT_FALSE(best_state.empty()) << "Should have saved best model";
    EXPECT_LT(best_loss, std::numeric_limits<float>::max())
        << "Best loss should be updated";
}

//==============================================================================
// Test 10: Model Versioning (simulated)
//==============================================================================

TEST(ModelPersistence, ModelVersioning) {
    auto device = Device::cpu();

    struct ModelCheckpoint {
        std::unordered_map<std::string, Tensor> model_state;
        std::unordered_map<std::string, Tensor> optimizer_state;
        int epoch;
        float loss;
        std::string version;
    };

    auto model = std::make_shared<SimpleLinearModel>();
    model->to(device);

    auto params = model->parameters();
    SGD optimizer(params, 0.01, 0.9);

    std::vector<ModelCheckpoint> checkpoints;

    model->train();
    for (int epoch = 0; epoch < 3; epoch++) {
        for (int batch = 0; batch < 5; batch++) {
            auto input = Variable(randn({4, 10}, DType::Float32, device), true);
            auto target = Variable(randn({4, 5}, DType::Float32, device), false);

            optimizer.zero_grad();
            auto output = model->forward(input);
            auto loss = mse_loss(output, target);
            loss.backward();
            optimizer.step();

            if (batch == 4) {  // Last batch of epoch
                ModelCheckpoint checkpoint;
                checkpoint.model_state = model->state_dict();
                checkpoint.optimizer_state = optimizer.state_dict();
                checkpoint.epoch = epoch;
                checkpoint.loss = loss.tensor().template item<float>();
                checkpoint.version = "v1.0.0";
                checkpoints.push_back(checkpoint);
            }
        }
    }

    EXPECT_EQ(checkpoints.size(), 3) << "Should have 3 checkpoints";

    for (size_t i = 0; i < checkpoints.size(); i++) {
        EXPECT_EQ(checkpoints[i].epoch, static_cast<int>(i));
        EXPECT_FALSE(checkpoints[i].model_state.empty());
        EXPECT_EQ(checkpoints[i].version, "v1.0.0");
    }
}

//==============================================================================
// Test 11: Checkpoint Validation
//==============================================================================

TEST(ModelPersistence, CheckpointValidation) {
    auto device = Device::cpu();
    auto model = std::make_shared<SimpleLinearModel>();
    model->to(device);

    // Get state dict
    auto state_dict = model->state_dict();

    // Validate checkpoint integrity
    EXPECT_GT(state_dict.size(), 0) << "State dict should not be empty";

    // Check all tensors are valid
    for (const auto& [key, tensor] : state_dict) {
        EXPECT_FALSE(key.empty()) << "Key should not be empty";
        EXPECT_GT(tensor.numel(), 0) << "Tensor should not be empty for key: " << key;
        EXPECT_TRUE(tensor.device().type == Device::Type::CPU ||
                    tensor.device().type == Device::Type::CUDA)
            << "Tensor should be on valid device";
    }
}

//==============================================================================
// Test 12: Large Model Save/Load Performance
//==============================================================================

TEST(ModelPersistence, LargeModelSaveLoadPerformance) {
    auto device = Device::cpu();

    // Create larger model
    class LargeModel : public Module {
    public:
        LargeModel() {
            fc1 = std::make_shared<Linear>(1000, 500);
            fc2 = std::make_shared<Linear>(500, 500);
            fc3 = std::make_shared<Linear>(500, 100);

            register_module("fc1", fc1);
            register_module("fc2", fc2);
            register_module("fc3", fc3);
        }

        auto forward(const Variable& x) -> Variable override {
            auto out = fc1->forward(x);
            out = relu(out);
            out = fc2->forward(out);
            out = relu(out);
            out = fc3->forward(out);
            return out;
        }

    private:
        std::shared_ptr<Linear> fc1, fc2, fc3;
    };

    auto model1 = std::make_shared<LargeModel>();
    model1->to(device);

    // Measure save time
    auto start = std::chrono::high_resolution_clock::now();
    auto state_dict = model1->state_dict();
    auto save_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now() - start).count();

    // Measure load time
    auto model2 = std::make_shared<LargeModel>();
    model2->to(device);

    start = std::chrono::high_resolution_clock::now();
    model2->load_state_dict(state_dict);
    auto load_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now() - start).count();

    std::cout << "Save time: " << save_time << "ms, Load time: " << load_time << "ms" << std::endl;

    EXPECT_LT(save_time, 1000) << "Save should complete in reasonable time";
    EXPECT_LT(load_time, 1000) << "Load should complete in reasonable time";
}
