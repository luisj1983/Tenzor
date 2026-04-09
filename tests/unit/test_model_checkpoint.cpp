/**
 * @file test_model_checkpoint.cpp
 * @brief Comprehensive tests for model checkpointing
 */

#include <gtest/gtest.h>
#include "tenzor/tenzor.hpp"
#include "tenzor/nn/checkpoint.hpp"
#include "tenzor/nn/module.hpp"
#include "tenzor/nn/layers/linear.hpp"
#include "tenzor/nn/optim/sgd.hpp"
#include "tenzor/nn/optim/adam.hpp"
#include "tenzor/nn/optim/scheduler.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/autograd/ops.hpp"
#include <filesystem>
#include <fstream>
#include <sstream>
#include <thread>
#include <unistd.h>

using namespace tenzor;
using namespace tenzor::nn;

class ModelCheckpointTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create unique directory per test instance to avoid parallel test conflicts
        std::stringstream ss;
        ss << "./test_checkpoints_tmp_" << getpid() << "_" << std::this_thread::get_id();
        test_dir_ = ss.str();
        std::filesystem::create_directories(test_dir_);
    }

    void TearDown() override {
        // Clean up test directory
        if (std::filesystem::exists(test_dir_)) {
            std::filesystem::remove_all(test_dir_);
        }
    }

    std::string test_dir_;
};

// ==============================================================================
// ModelCheckpoint Tests
// ==============================================================================

TEST_F(ModelCheckpointTest, Construction) {
    ModelCheckpoint checkpoint;
    EXPECT_EQ(checkpoint.config().compression, CompressionType::None);
    EXPECT_TRUE(checkpoint.config().save_optimizer);
    EXPECT_TRUE(checkpoint.config().save_scheduler);
    EXPECT_TRUE(checkpoint.config().verify_checksum);
    EXPECT_TRUE(checkpoint.config().atomic_save);
}

TEST_F(ModelCheckpointTest, CustomConfig) {
    CheckpointConfig config;
    config.compression = CompressionType::LZ4;
    config.save_optimizer = false;
    config.atomic_save = false;

    ModelCheckpoint checkpoint(config);
    EXPECT_EQ(checkpoint.config().compression, CompressionType::LZ4);
    EXPECT_FALSE(checkpoint.config().save_optimizer);
}

TEST_F(ModelCheckpointTest, SaveLoadModelOnly) {
    // Create simple linear layer
    Linear model(10, 5);

    // Initialize with specific values
    auto params = model.parameters();
    auto weight_data = params[0]->tensor().data<float>();
    auto bias_data = params[1]->tensor().data<float>();

    // Note: We're reading const data for comparison later, not modifying

    // Save model
    std::string path = test_dir_ + "/model.pt";
    ModelCheckpoint checkpoint;
    checkpoint.save_model(path, model);

    // Verify file exists
    EXPECT_TRUE(std::filesystem::exists(path));

    // Load model state
    auto loaded_state = checkpoint.load_model(path);

    // Verify state contains expected keys
    EXPECT_TRUE(loaded_state.find("weight") != loaded_state.end());
    EXPECT_TRUE(loaded_state.find("bias") != loaded_state.end());

    // Verify values match
    const float* loaded_weight = loaded_state["weight"].data<float>();
    const float* loaded_bias = loaded_state["bias"].data<float>();

    for (int i = 0; i < 50; ++i) {
        EXPECT_FLOAT_EQ(loaded_weight[i], weight_data[i]);
    }

    for (int i = 0; i < 5; ++i) {
        EXPECT_FLOAT_EQ(loaded_bias[i], bias_data[i]);
    }
}

TEST_F(ModelCheckpointTest, SaveLoadWithOptimizer) {
    // Create model and optimizer
    Linear model(8, 4);
    auto params = model.parameters();
    optim::SGD optimizer(params, 0.01);

    // Take a few optimizer steps to initialize state
    for (int i = 0; i < 3; ++i) {
        auto input = randn({2, 8});
        auto output = model.forward(Variable(input, true));
        auto loss = tenzor::sum(output);
        optimizer.zero_grad();
        loss.backward();
        optimizer.step();
    }

    // Save checkpoint
    std::string path = test_dir_ + "/checkpoint_with_optim.pt";
    ModelCheckpoint checkpoint;
    checkpoint.save(path, model, &optimizer);

    // Load checkpoint
    auto loaded = checkpoint.load(path);

    // Verify model state
    EXPECT_FALSE(loaded.model_state.empty());

    // Verify optimizer state (SGD has momentum buffers)
    EXPECT_FALSE(loaded.optimizer_state.empty());

    // Verify version
    EXPECT_EQ(loaded.version, CHECKPOINT_VERSION);
}

TEST_F(ModelCheckpointTest, SaveLoadWithMetadata) {
    Linear model(4, 2);

    TrainingMetadata metadata;
    metadata.epoch = 10;
    metadata.global_step = 1000;
    metadata.learning_rate = 0.001;
    metadata.train_loss = 0.25;
    metadata.val_loss = 0.30;
    metadata.train_accuracy = 0.85;
    metadata.val_accuracy = 0.82;
    metadata.best_val_loss = 0.28;
    metadata.best_val_accuracy = 0.83;
    metadata.custom_metrics["f1_score"] = 0.81;
    metadata.custom_metrics["precision"] = 0.84;

    std::string path = test_dir_ + "/checkpoint_with_metadata.pt";
    ModelCheckpoint checkpoint;
    checkpoint.save_model(path, model, metadata);

    // Load and verify metadata
    auto loaded_metadata = checkpoint.get_metadata(path);

    EXPECT_EQ(loaded_metadata.epoch, 10);
    EXPECT_EQ(loaded_metadata.global_step, 1000);
    EXPECT_DOUBLE_EQ(loaded_metadata.learning_rate, 0.001);
    EXPECT_DOUBLE_EQ(loaded_metadata.train_loss, 0.25);
    EXPECT_DOUBLE_EQ(loaded_metadata.val_loss, 0.30);
    EXPECT_DOUBLE_EQ(loaded_metadata.train_accuracy, 0.85);
    EXPECT_DOUBLE_EQ(loaded_metadata.val_accuracy, 0.82);
    EXPECT_DOUBLE_EQ(loaded_metadata.best_val_loss, 0.28);
    EXPECT_DOUBLE_EQ(loaded_metadata.best_val_accuracy, 0.83);

    EXPECT_DOUBLE_EQ(loaded_metadata.custom_metrics["f1_score"], 0.81);
    EXPECT_DOUBLE_EQ(loaded_metadata.custom_metrics["precision"], 0.84);
}

TEST_F(ModelCheckpointTest, VerifyCheckpoint) {
    Linear model(3, 2);

    std::string path = test_dir_ + "/verify_test.pt";
    ModelCheckpoint checkpoint;
    checkpoint.save_model(path, model);

    // Verify valid checkpoint
    EXPECT_TRUE(checkpoint.verify_checkpoint(path));

    // Corrupt checkpoint
    std::ofstream corrupt(path, std::ios::binary | std::ios::app);
    corrupt << "CORRUPT_DATA";
    corrupt.close();

    // Verification should fail for corrupted checkpoint
    EXPECT_FALSE(checkpoint.verify_checkpoint(path));
}

TEST_F(ModelCheckpointTest, GetVersion) {
    Linear model(2, 1);

    std::string path = test_dir_ + "/version_test.pt";
    ModelCheckpoint checkpoint;
    checkpoint.save_model(path, model);

    uint32_t version = checkpoint.get_version(path);
    EXPECT_EQ(version, CHECKPOINT_VERSION);
}

TEST_F(ModelCheckpointTest, IsCompatible) {
    Linear model(2, 1);

    std::string path = test_dir_ + "/compat_test.pt";
    ModelCheckpoint checkpoint;
    checkpoint.save_model(path, model);

    EXPECT_TRUE(checkpoint.is_compatible(path));
}

TEST_F(ModelCheckpointTest, RoundtripPreservesValues) {
    // Create model with known values
    Linear model(5, 3);

    auto params = model.parameters();
    const float* weight = params[0]->tensor().data<float>();
    const float* bias = params[1]->tensor().data<float>();

    // Note: Reading const data for verification, model has its own initialized values

    // Save and load
    std::string path = test_dir_ + "/roundtrip.pt";
    ModelCheckpoint checkpoint;
    checkpoint.save_model(path, model);

    auto loaded_state = checkpoint.load_model(path);

    // Verify exact match
    const float* loaded_weight = loaded_state["weight"].data<float>();
    const float* loaded_bias = loaded_state["bias"].data<float>();

    for (int i = 0; i < 15; ++i) {
        EXPECT_FLOAT_EQ(loaded_weight[i], weight[i]);
    }

    for (int i = 0; i < 3; ++i) {
        EXPECT_FLOAT_EQ(loaded_bias[i], bias[i]);
    }
}

TEST_F(ModelCheckpointTest, StateDictSize) {
    Linear model(100, 50);

    Checkpoint chkpt;
    chkpt.model_state = model.state_dict();

    size_t size = chkpt.size_bytes();

    // Should be at least the size of weights + biases
    size_t expected_min = (100 * 50 + 50) * sizeof(float);
    EXPECT_GE(size, expected_min);
}

TEST_F(ModelCheckpointTest, CheckpointIsValid) {
    Checkpoint chkpt;
    chkpt.version = CHECKPOINT_VERSION;

    Linear model(3, 2);
    chkpt.model_state = model.state_dict();

    EXPECT_TRUE(chkpt.is_valid());

    // Empty checkpoint should be invalid
    Checkpoint empty;
    EXPECT_FALSE(empty.is_valid());
}

// ==============================================================================
// AutoCheckpoint Tests
// ==============================================================================

TEST_F(ModelCheckpointTest, AutoCheckpointConstruction) {
    AutoCheckpoint auto_checkpoint(test_dir_, 5, 2);

    EXPECT_EQ(auto_checkpoint.checkpoint_paths().size(), 0);
    EXPECT_EQ(auto_checkpoint.best_checkpoint_path(), "");
}

TEST_F(ModelCheckpointTest, AutoCheckpointStep) {
    AutoCheckpoint auto_checkpoint(test_dir_, 3, 1);
    auto_checkpoint.set_metric_mode("min");

    Linear model(4, 2);
    auto params = model.parameters();
    optim::SGD optimizer(params, 0.01);

    // Step 1: epoch 0, loss 1.0
    bool saved = auto_checkpoint.step(model, optimizer, 0, 1.0, "loss");
    EXPECT_TRUE(saved);
    EXPECT_EQ(auto_checkpoint.checkpoint_paths().size(), 1);
    EXPECT_DOUBLE_EQ(auto_checkpoint.best_metric_value(), 1.0);

    // Step 2: epoch 1, loss 0.8 (better)
    saved = auto_checkpoint.step(model, optimizer, 1, 0.8, "loss");
    EXPECT_TRUE(saved);
    EXPECT_EQ(auto_checkpoint.checkpoint_paths().size(), 2);
    EXPECT_DOUBLE_EQ(auto_checkpoint.best_metric_value(), 0.8);

    // Step 3: epoch 2, loss 0.9 (worse, but still save)
    saved = auto_checkpoint.step(model, optimizer, 2, 0.9, "loss");
    EXPECT_TRUE(saved);
    EXPECT_EQ(auto_checkpoint.checkpoint_paths().size(), 3);
    EXPECT_DOUBLE_EQ(auto_checkpoint.best_metric_value(), 0.8);
}

TEST_F(ModelCheckpointTest, AutoCheckpointMaxCheckpoints) {
    AutoCheckpoint auto_checkpoint(test_dir_, 2, 1);  // Keep only 2
    auto_checkpoint.set_metric_mode("min");

    Linear model(3, 2);
    auto params = model.parameters();
    optim::SGD optimizer(params, 0.01);

    // Save 4 checkpoints
    for (int epoch = 0; epoch < 4; ++epoch) {
        double loss = 1.0 - epoch * 0.1;  // Decreasing loss
        auto_checkpoint.step(model, optimizer, epoch, loss, "loss");
    }

    // Should keep only best 2
    EXPECT_LE(auto_checkpoint.checkpoint_paths().size(), 2);
}

TEST_F(ModelCheckpointTest, AutoCheckpointMetricMode) {
    // Test "max" mode (for accuracy)
    AutoCheckpoint auto_checkpoint(test_dir_, 3, 1);
    auto_checkpoint.set_metric_mode("max");

    Linear model(2, 1);
    auto params = model.parameters();
    optim::SGD optimizer(params, 0.01);

    auto_checkpoint.step(model, optimizer, 0, 0.7, "accuracy");
    EXPECT_DOUBLE_EQ(auto_checkpoint.best_metric_value(), 0.7);

    auto_checkpoint.step(model, optimizer, 1, 0.85, "accuracy");
    EXPECT_DOUBLE_EQ(auto_checkpoint.best_metric_value(), 0.85);

    auto_checkpoint.step(model, optimizer, 2, 0.8, "accuracy");
    EXPECT_DOUBLE_EQ(auto_checkpoint.best_metric_value(), 0.85);  // Best is still 0.85
}

TEST_F(ModelCheckpointTest, AutoCheckpointSaveFrequency) {
    AutoCheckpoint auto_checkpoint(test_dir_, 5, 3);  // Save every 3 epochs

    Linear model(2, 1);
    auto params = model.parameters();
    optim::SGD optimizer(params, 0.01);

    // Epoch 0: should save (first epoch)
    bool saved = auto_checkpoint.step(model, optimizer, 0, 1.0, "loss");
    EXPECT_TRUE(saved);

    // Epoch 1: should not save (not multiple of 3)
    saved = auto_checkpoint.step(model, optimizer, 1, 0.9, "loss");
    EXPECT_FALSE(saved);

    // Epoch 2: should not save
    saved = auto_checkpoint.step(model, optimizer, 2, 0.8, "loss");
    EXPECT_FALSE(saved);

    // Epoch 3: should save (multiple of 3)
    saved = auto_checkpoint.step(model, optimizer, 3, 0.7, "loss");
    EXPECT_TRUE(saved);
}

TEST_F(ModelCheckpointTest, AutoCheckpointBestPath) {
    AutoCheckpoint auto_checkpoint(test_dir_, 3, 1);
    auto_checkpoint.set_metric_mode("min");

    Linear model(2, 1);
    auto params = model.parameters();
    optim::SGD optimizer(params, 0.01);

    auto_checkpoint.step(model, optimizer, 0, 1.0, "loss");
    auto_checkpoint.step(model, optimizer, 1, 0.5, "loss");  // Best
    auto_checkpoint.step(model, optimizer, 2, 0.8, "loss");

    std::string best_path = auto_checkpoint.best_checkpoint_path();
    EXPECT_FALSE(best_path.empty());
    EXPECT_TRUE(std::filesystem::exists(best_path));
}

int main(int argc, char** argv) {
    // Initialize Tenzor library (loads backends and registers operations)

    ::testing::InitGoogleTest(&argc, argv);
    if (!::testing::GTEST_FLAG(list_tests)) {
        tenzor::initialize();
    }
    int result = RUN_ALL_TESTS();

    // Cleanup
    tenzor::finalize();

    return result;
}
