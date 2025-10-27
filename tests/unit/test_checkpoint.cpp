/**
 * @file test_checkpoint.cpp
 * @brief Comprehensive unit tests for model checkpoint functionality
 */

#include <gtest/gtest.h>
#include "../../include/tenzor/nn/checkpoint.hpp"
#include "../../include/tenzor/nn/layers/linear.hpp"
#include "../../include/tenzor/nn/optim/sgd.hpp"
#include "../../include/tenzor/nn/optim/adam.hpp"
#include "../../include/tenzor/nn/optim/scheduler.hpp"
#include "../../include/tenzor/ops/creation.hpp"
#include <filesystem>
#include <fstream>
#include <cmath>

using namespace tenzor;
using namespace tenzor::nn;

// ============================================================================
// Test Fixtures
// ============================================================================

class CheckpointTest : public ::testing::Test {
protected:
    void SetUp() override {
        test_dir_ = "test_checkpoints_temp";
        std::filesystem::create_directories(test_dir_);
    }

    void TearDown() override {
        // Cleanup test directory
        if (std::filesystem::exists(test_dir_)) {
            std::filesystem::remove_all(test_dir_);
        }
    }

    std::string test_dir_;
};

// ============================================================================
// Helper: Simple Test Model
// ============================================================================

class SimpleModel : public Module {
public:
    SimpleModel() {
        fc1_ = std::make_shared<Linear>(10, 20);
        fc2_ = std::make_shared<Linear>(20, 5);
        register_module("fc1", fc1_);
        register_module("fc2", fc2_);
    }

    auto forward(const Variable& x) -> Variable override {
        auto h = fc1_->forward(x);
        return fc2_->forward(h);
    }

private:
    std::shared_ptr<Linear> fc1_;
    std::shared_ptr<Linear> fc2_;
};

// ============================================================================
// TrainingMetadata Tests
// ============================================================================

TEST_F(CheckpointTest, MetadataSerializationRoundTrip) {
    TrainingMetadata original;
    original.epoch = 42;
    original.global_step = 1000;
    original.learning_rate = 0.001;
    original.train_loss = 0.25;
    original.val_loss = 0.30;
    original.train_accuracy = 0.95;
    original.val_accuracy = 0.92;
    original.best_val_loss = 0.28;
    original.best_val_accuracy = 0.93;
    original.timestamp = "2024-01-15_10-30-00";
    original.custom_metrics["f1_score"] = 0.91;
    original.custom_metrics["precision"] = 0.94;

    // Serialize to dict
    auto dict = original.to_dict();

    // Deserialize from dict
    TrainingMetadata restored;
    restored.from_dict(dict);

    // Verify all fields match
    EXPECT_EQ(restored.epoch, original.epoch);
    EXPECT_EQ(restored.global_step, original.global_step);
    EXPECT_DOUBLE_EQ(restored.learning_rate, original.learning_rate);
    EXPECT_DOUBLE_EQ(restored.train_loss, original.train_loss);
    EXPECT_DOUBLE_EQ(restored.val_loss, original.val_loss);
    EXPECT_DOUBLE_EQ(restored.train_accuracy, original.train_accuracy);
    EXPECT_DOUBLE_EQ(restored.val_accuracy, original.val_accuracy);
    EXPECT_DOUBLE_EQ(restored.best_val_loss, original.best_val_loss);
    EXPECT_DOUBLE_EQ(restored.best_val_accuracy, original.best_val_accuracy);
    EXPECT_EQ(restored.timestamp, original.timestamp);
    EXPECT_EQ(restored.custom_metrics.size(), original.custom_metrics.size());
    EXPECT_DOUBLE_EQ(restored.custom_metrics["f1_score"], 0.91);
    EXPECT_DOUBLE_EQ(restored.custom_metrics["precision"], 0.94);
}

TEST_F(CheckpointTest, MetadataWithMissingFields) {
    std::unordered_map<std::string, std::string> incomplete_dict;
    incomplete_dict["epoch"] = "10";
    incomplete_dict["train_loss"] = "0.5";

    TrainingMetadata metadata;
    metadata.from_dict(incomplete_dict);

    EXPECT_EQ(metadata.epoch, 10);
    EXPECT_DOUBLE_EQ(metadata.train_loss, 0.5);
    EXPECT_EQ(metadata.global_step, 0);  // Default value
    EXPECT_DOUBLE_EQ(metadata.learning_rate, 0.0);  // Default value
}

// ============================================================================
// Checkpoint Structure Tests
// ============================================================================

TEST_F(CheckpointTest, CheckpointSizeCalculation) {
    Checkpoint checkpoint;

    // Add model state tensors
    checkpoint.model_state["weight1"] = tenzor::randn({10, 20}, DType::Float32);
    checkpoint.model_state["bias1"] = tenzor::randn({20}, DType::Float32);
    checkpoint.model_state["weight2"] = tenzor::randn({20, 5}, DType::Float32);
    checkpoint.model_state["bias2"] = tenzor::randn({5}, DType::Float32);

    size_t expected_size =
        10 * 20 * sizeof(float) +  // weight1
        20 * sizeof(float) +       // bias1
        20 * 5 * sizeof(float) +   // weight2
        5 * sizeof(float) +        // bias2
        1024;                      // metadata overhead

    EXPECT_EQ(checkpoint.size_bytes(), expected_size);
}

TEST_F(CheckpointTest, CheckpointValidity) {
    Checkpoint checkpoint;
    EXPECT_FALSE(checkpoint.is_valid());  // No model state

    checkpoint.model_state["weight"] = tenzor::randn({5, 5}, DType::Float32);
    EXPECT_TRUE(checkpoint.is_valid());  // Has model state and correct version
}

// ============================================================================
// ModelCheckpoint Save/Load Tests
// ============================================================================

TEST_F(CheckpointTest, SaveLoadModelOnlyRoundTrip) {
    SimpleModel model;
    std::string checkpoint_path = test_dir_ + "/model_only.pt";

    // Create checkpoint manager
    ModelCheckpoint checkpoint_manager;

    // Save model
    TrainingMetadata metadata;
    metadata.epoch = 10;
    metadata.train_loss = 0.25;
    checkpoint_manager.save_model(checkpoint_path, model, metadata);

    // Verify file exists
    EXPECT_TRUE(std::filesystem::exists(checkpoint_path));

    // Load model state
    auto loaded_state = checkpoint_manager.load_model(checkpoint_path);

    // Verify state matches
    auto original_state = model.state_dict();
    EXPECT_EQ(loaded_state.size(), original_state.size());

    for (const auto& [name, tensor] : original_state) {
        ASSERT_TRUE(loaded_state.count(name) > 0) << "Missing parameter: " << name;

        const auto& loaded_tensor = loaded_state.at(name);
        auto loaded_shape = loaded_tensor.shape();
        auto original_shape = tensor.shape();
        EXPECT_EQ(std::vector<int64_t>(loaded_shape.begin(), loaded_shape.end()),
                  std::vector<int64_t>(original_shape.begin(), original_shape.end()));
        EXPECT_EQ(loaded_tensor.dtype(), tensor.dtype());

        // Verify data is identical
        const float* orig_data = static_cast<const float*>(tensor.data_ptr());
        const float* loaded_data = static_cast<const float*>(loaded_tensor.data_ptr());
        size_t numel = tensor.numel();

        for (size_t i = 0; i < numel; ++i) {
            EXPECT_FLOAT_EQ(loaded_data[i], orig_data[i])
                << "Data mismatch at index " << i << " in parameter " << name;
        }
    }
}

TEST_F(CheckpointTest, SaveLoadWithOptimizerState) {
    SimpleModel model;
    auto params = model.parameters();
    optim::SGD optimizer(params, 0.01, 0.9);  // lr=0.01, momentum=0.9

    // Perform one optimizer step to create momentum buffers
    for (auto& param : params) {
        auto shape = param->tensor().shape();
        Tensor grad = tenzor::ones(std::vector<int64_t>(shape.begin(), shape.end()), param->tensor().dtype());
        param->set_grad(grad);
    }
    optimizer.step();

    std::string checkpoint_path = test_dir_ + "/with_optimizer.pt";

    // Create checkpoint manager
    CheckpointConfig config;
    config.save_optimizer = true;
    ModelCheckpoint checkpoint_manager(config);

    // Save model and optimizer
    TrainingMetadata metadata;
    metadata.epoch = 5;
    checkpoint_manager.save(checkpoint_path, model, &optimizer, nullptr, metadata);

    // Load checkpoint
    auto loaded = checkpoint_manager.load(checkpoint_path);

    // Verify metadata
    EXPECT_EQ(loaded.metadata.epoch, 5);

    // Verify model state
    EXPECT_EQ(loaded.model_state.size(), model.state_dict().size());

    // Verify optimizer state exists
    EXPECT_GT(loaded.optimizer_state.size(), 0);
}

TEST_F(CheckpointTest, SaveLoadWithScheduler) {
    SimpleModel model;
    auto params = model.parameters();
    optim::Adam optimizer(params, 0.001);
    optim::StepLR scheduler(optimizer, 10, 0.1);

    // Step scheduler a few times
    for (int i = 0; i < 5; ++i) {
        scheduler.step();
    }

    std::string checkpoint_path = test_dir_ + "/with_scheduler.pt";

    // Create checkpoint manager
    CheckpointConfig config;
    config.save_optimizer = true;
    config.save_scheduler = true;
    ModelCheckpoint checkpoint_manager(config);

    // Save everything
    checkpoint_manager.save(checkpoint_path, model, &optimizer, &scheduler, TrainingMetadata{});

    // Load checkpoint
    auto loaded = checkpoint_manager.load(checkpoint_path);

    // Verify all components exist
    EXPECT_GT(loaded.model_state.size(), 0);
    EXPECT_GT(loaded.optimizer_state.size(), 0);
    // Scheduler state would be checked if implemented
}

// ============================================================================
// Checkpoint Verification Tests
// ============================================================================

TEST_F(CheckpointTest, VerifyValidCheckpoint) {
    SimpleModel model;
    std::string checkpoint_path = test_dir_ + "/valid.pt";

    ModelCheckpoint checkpoint_manager;
    checkpoint_manager.save_model(checkpoint_path, model);

    EXPECT_TRUE(checkpoint_manager.verify_checkpoint(checkpoint_path));
}

TEST_F(CheckpointTest, VerifyInvalidFile) {
    std::string invalid_path = test_dir_ + "/nonexistent.pt";

    ModelCheckpoint checkpoint_manager;
    EXPECT_FALSE(checkpoint_manager.verify_checkpoint(invalid_path));
}

TEST_F(CheckpointTest, VerifyCorruptedFile) {
    SimpleModel model;
    std::string checkpoint_path = test_dir_ + "/corrupted.pt";

    ModelCheckpoint checkpoint_manager;
    checkpoint_manager.save_model(checkpoint_path, model);

    // Corrupt the file by truncating it
    {
        std::ofstream file(checkpoint_path, std::ios::binary | std::ios::trunc);
        file << "corrupted data";
    }

    EXPECT_FALSE(checkpoint_manager.verify_checkpoint(checkpoint_path));
}

TEST_F(CheckpointTest, GetMetadataWithoutFullLoad) {
    SimpleModel model;
    std::string checkpoint_path = test_dir_ + "/metadata_test.pt";

    TrainingMetadata original_metadata;
    original_metadata.epoch = 42;
    original_metadata.train_loss = 0.123;
    original_metadata.val_loss = 0.456;

    ModelCheckpoint checkpoint_manager;
    checkpoint_manager.save_model(checkpoint_path, model, original_metadata);

    // Get metadata without loading full checkpoint
    auto loaded_metadata = checkpoint_manager.get_metadata(checkpoint_path);

    EXPECT_EQ(loaded_metadata.epoch, original_metadata.epoch);
    EXPECT_DOUBLE_EQ(loaded_metadata.train_loss, original_metadata.train_loss);
    EXPECT_DOUBLE_EQ(loaded_metadata.val_loss, original_metadata.val_loss);
}

// ============================================================================
// Version Compatibility Tests
// ============================================================================

TEST_F(CheckpointTest, GetCheckpointVersion) {
    SimpleModel model;
    std::string checkpoint_path = test_dir_ + "/version_test.pt";

    ModelCheckpoint checkpoint_manager;
    checkpoint_manager.save_model(checkpoint_path, model);

    uint32_t version = checkpoint_manager.get_version(checkpoint_path);
    EXPECT_EQ(version, CHECKPOINT_VERSION);
}

TEST_F(CheckpointTest, CheckCompatibility) {
    SimpleModel model;
    std::string checkpoint_path = test_dir_ + "/compat_test.pt";

    ModelCheckpoint checkpoint_manager;
    checkpoint_manager.save_model(checkpoint_path, model);

    EXPECT_TRUE(checkpoint_manager.is_compatible(checkpoint_path));
}

// ============================================================================
// Atomic Save Tests
// ============================================================================

TEST_F(CheckpointTest, AtomicSaveCreatesTemporaryFile) {
    SimpleModel model;
    std::string checkpoint_path = test_dir_ + "/atomic_save.pt";
    std::string temp_path = checkpoint_path + ".tmp";

    CheckpointConfig config;
    config.atomic_save = true;
    ModelCheckpoint checkpoint_manager(config);

    checkpoint_manager.save_model(checkpoint_path, model);

    // Final file should exist
    EXPECT_TRUE(std::filesystem::exists(checkpoint_path));
    // Temporary file should be removed after rename
    EXPECT_FALSE(std::filesystem::exists(temp_path));
}

// ============================================================================
// Checksum Verification Tests
// ============================================================================

TEST_F(CheckpointTest, SaveLoadWithChecksumVerification) {
    SimpleModel model;
    std::string checkpoint_path = test_dir_ + "/with_checksum.pt";

    CheckpointConfig config;
    config.verify_checksum = true;
    ModelCheckpoint checkpoint_manager(config);

    checkpoint_manager.save_model(checkpoint_path, model);

    // Verify checkpoint is valid with checksum
    EXPECT_TRUE(checkpoint_manager.verify_checkpoint(checkpoint_path));

    // Load should succeed
    auto loaded_state = checkpoint_manager.load_model(checkpoint_path);
    EXPECT_EQ(loaded_state.size(), model.state_dict().size());
}

TEST_F(CheckpointTest, ChecksumDetectsCorruption) {
    SimpleModel model;
    std::string checkpoint_path = test_dir_ + "/checksum_corrupt.pt";

    CheckpointConfig config;
    config.verify_checksum = true;
    ModelCheckpoint checkpoint_manager(config);

    checkpoint_manager.save_model(checkpoint_path, model);

    // Read file, corrupt one byte, write back
    std::vector<uint8_t> file_data;
    {
        std::ifstream file(checkpoint_path, std::ios::binary);
        file.seekg(0, std::ios::end);
        size_t size = file.tellg();
        file.seekg(0, std::ios::beg);
        file_data.resize(size);
        file.read(reinterpret_cast<char*>(file_data.data()), size);
    }

    // Corrupt a byte in the middle
    if (file_data.size() > 100) {
        file_data[50] ^= 0xFF;
    }

    {
        std::ofstream file(checkpoint_path, std::ios::binary);
        file.write(reinterpret_cast<const char*>(file_data.data()), file_data.size());
    }

    // Verification should fail due to checksum mismatch
    EXPECT_FALSE(checkpoint_manager.verify_checkpoint(checkpoint_path));
}

// ============================================================================
// AutoCheckpoint Tests
// ============================================================================

TEST_F(CheckpointTest, AutoCheckpointBasicFunctionality) {
    std::string checkpoint_dir = test_dir_ + "/auto_checkpoints";
    AutoCheckpoint auto_checkpoint(checkpoint_dir, 3, 1);  // Keep top 3, save every epoch

    SimpleModel model;
    auto params = model.parameters();
    optim::SGD optimizer(params, 0.01);

    // Simulate training for 5 epochs
    for (int epoch = 0; epoch < 5; ++epoch) {
        double val_loss = 1.0 / (epoch + 1);  // Decreasing loss
        auto_checkpoint.step(model, optimizer, epoch, val_loss, "val_loss");
    }

    // Should have 3 checkpoints (max_checkpoints = 3)
    auto paths = auto_checkpoint.checkpoint_paths();
    EXPECT_LE(paths.size(), 3);

    // Verify checkpoint directory exists
    EXPECT_TRUE(std::filesystem::exists(checkpoint_dir));
}

TEST_F(CheckpointTest, AutoCheckpointMinMode) {
    std::string checkpoint_dir = test_dir_ + "/auto_min";
    AutoCheckpoint auto_checkpoint(checkpoint_dir, 2, 1);
    auto_checkpoint.set_metric_mode("min");

    SimpleModel model;
    auto params = model.parameters();
    optim::SGD optimizer(params, 0.01);

    // Save checkpoints with different losses
    auto_checkpoint.step(model, optimizer, 0, 0.5, "val_loss");
    auto_checkpoint.step(model, optimizer, 1, 0.3, "val_loss");  // Better (lower)
    auto_checkpoint.step(model, optimizer, 2, 0.4, "val_loss");

    // Best should be epoch 1 with loss 0.3
    EXPECT_DOUBLE_EQ(auto_checkpoint.best_metric_value(), 0.3);
    EXPECT_TRUE(auto_checkpoint.best_checkpoint_path().find("epoch_0001") != std::string::npos);
}

TEST_F(CheckpointTest, AutoCheckpointMaxMode) {
    std::string checkpoint_dir = test_dir_ + "/auto_max";
    AutoCheckpoint auto_checkpoint(checkpoint_dir, 2, 1);
    auto_checkpoint.set_metric_mode("max");

    SimpleModel model;
    auto params = model.parameters();
    optim::SGD optimizer(params, 0.01);

    // Save checkpoints with different accuracies
    auto_checkpoint.step(model, optimizer, 0, 0.85, "val_accuracy");
    auto_checkpoint.step(model, optimizer, 1, 0.92, "val_accuracy");  // Better (higher)
    auto_checkpoint.step(model, optimizer, 2, 0.88, "val_accuracy");

    // Best should be epoch 1 with accuracy 0.92
    EXPECT_DOUBLE_EQ(auto_checkpoint.best_metric_value(), 0.92);
    EXPECT_TRUE(auto_checkpoint.best_checkpoint_path().find("epoch_0001") != std::string::npos);
}

TEST_F(CheckpointTest, AutoCheckpointSaveFrequency) {
    std::string checkpoint_dir = test_dir_ + "/auto_freq";
    AutoCheckpoint auto_checkpoint(checkpoint_dir, 5, 2);  // Save every 2 epochs

    SimpleModel model;
    auto params = model.parameters();
    optim::SGD optimizer(params, 0.01);

    // Simulate 6 epochs
    int saved_count = 0;
    for (int epoch = 0; epoch < 6; ++epoch) {
        bool saved = auto_checkpoint.step(model, optimizer, epoch, 0.5, "val_loss");
        if (saved) saved_count++;
    }

    // Should save at epochs 0, 2, 4 (every 2 epochs) = 3 saves
    EXPECT_EQ(saved_count, 3);
}

TEST_F(CheckpointTest, AutoCheckpointCleanup) {
    std::string checkpoint_dir = test_dir_ + "/auto_cleanup";
    AutoCheckpoint auto_checkpoint(checkpoint_dir, 2, 1);  // Keep only top 2
    auto_checkpoint.set_metric_mode("min");

    SimpleModel model;
    auto params = model.parameters();
    optim::SGD optimizer(params, 0.01);

    // Save 5 checkpoints
    auto_checkpoint.step(model, optimizer, 0, 0.9, "val_loss");
    auto_checkpoint.step(model, optimizer, 1, 0.7, "val_loss");
    auto_checkpoint.step(model, optimizer, 2, 0.5, "val_loss");  // Best
    auto_checkpoint.step(model, optimizer, 3, 0.6, "val_loss");  // Second best
    auto_checkpoint.step(model, optimizer, 4, 0.8, "val_loss");

    // Should keep only 2 best checkpoints (epochs 2 and 3)
    auto paths = auto_checkpoint.checkpoint_paths();
    EXPECT_EQ(paths.size(), 2);

    // Verify the kept checkpoints are the best ones
    std::string best_path = auto_checkpoint.best_checkpoint_path();
    EXPECT_TRUE(best_path.find("epoch_0002") != std::string::npos);
}

// ============================================================================
// Edge Cases and Error Handling
// ============================================================================

TEST_F(CheckpointTest, LoadNonexistentFile) {
    ModelCheckpoint checkpoint_manager;
    std::string nonexistent = test_dir_ + "/does_not_exist.pt";

    EXPECT_THROW(checkpoint_manager.load(nonexistent), std::runtime_error);
}

TEST_F(CheckpointTest, SaveToInvalidPath) {
    SimpleModel model;
    ModelCheckpoint checkpoint_manager;

    // Try to save to a path with non-existent parent directory
    std::string invalid_path = "/nonexistent_dir_12345/model.pt";

    EXPECT_THROW(checkpoint_manager.save_model(invalid_path, model), std::runtime_error);
}

TEST_F(CheckpointTest, EmptyModelState) {
    // Create a checkpoint with empty model state
    Checkpoint checkpoint;
    checkpoint.version = CHECKPOINT_VERSION;

    EXPECT_FALSE(checkpoint.is_valid());  // Should be invalid without model state
}

TEST_F(CheckpointTest, LargeCheckpoint) {
    // Test with a larger model to ensure scalability
    class LargeModel : public Module {
    public:
        LargeModel() {
            fc1_ = std::make_shared<Linear>(1000, 1000);
            fc2_ = std::make_shared<Linear>(1000, 1000);
            fc3_ = std::make_shared<Linear>(1000, 100);
            register_module("fc1", fc1_);
            register_module("fc2", fc2_);
            register_module("fc3", fc3_);
        }

        auto forward(const Variable& x) -> Variable override {
            return fc3_->forward(fc2_->forward(fc1_->forward(x)));
        }

    private:
        std::shared_ptr<Linear> fc1_, fc2_, fc3_;
    };

    LargeModel model;
    std::string checkpoint_path = test_dir_ + "/large_model.pt";

    ModelCheckpoint checkpoint_manager;
    checkpoint_manager.save_model(checkpoint_path, model);

    // Verify file was created
    EXPECT_TRUE(std::filesystem::exists(checkpoint_path));

    // Verify file size is reasonable (should be several MB)
    auto file_size = std::filesystem::file_size(checkpoint_path);
    EXPECT_GT(file_size, 1000000);  // At least 1 MB

    // Verify can load
    auto loaded_state = checkpoint_manager.load_model(checkpoint_path);
    EXPECT_EQ(loaded_state.size(), model.state_dict().size());
}

TEST_F(CheckpointTest, AutoCheckpointInvalidMetricMode) {
    std::string checkpoint_dir = test_dir_ + "/invalid_mode";
    AutoCheckpoint auto_checkpoint(checkpoint_dir);

    EXPECT_THROW(auto_checkpoint.set_metric_mode("invalid"), std::runtime_error);
}

// ============================================================================
// Performance Tests
// ============================================================================

TEST_F(CheckpointTest, SaveLoadPerformance) {
    SimpleModel model;
    std::string checkpoint_path = test_dir_ + "/perf_test.pt";

    ModelCheckpoint checkpoint_manager;

    // Measure save time
    auto start_save = std::chrono::high_resolution_clock::now();
    checkpoint_manager.save_model(checkpoint_path, model);
    auto end_save = std::chrono::high_resolution_clock::now();
    auto save_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_save - start_save);

    // Measure load time
    auto start_load = std::chrono::high_resolution_clock::now();
    auto loaded_state = checkpoint_manager.load_model(checkpoint_path);
    auto end_load = std::chrono::high_resolution_clock::now();
    auto load_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_load - start_load);

    // Save and load should be reasonably fast (under 100ms for small model)
    EXPECT_LT(save_duration.count(), 100);
    EXPECT_LT(load_duration.count(), 100);

    std::cout << "Save time: " << save_duration.count() << "ms, "
              << "Load time: " << load_duration.count() << "ms" << std::endl;
}

// ============================================================================
// Integration Tests
// ============================================================================

TEST_F(CheckpointTest, CompleteTrainingWorkflow) {
    // Simulate a complete training workflow with checkpointing
    std::string checkpoint_dir = test_dir_ + "/training_workflow";
    AutoCheckpoint auto_checkpoint(checkpoint_dir, 3, 1);
    auto_checkpoint.set_metric_mode("min");

    SimpleModel model;
    auto params = model.parameters();
    optim::Adam optimizer(params, 0.001);
    optim::StepLR scheduler(optimizer, 5, 0.1);

    // Simulate 10 epochs of training
    for (int epoch = 0; epoch < 10; ++epoch) {
        // Simulate decreasing validation loss
        double val_loss = 1.0 * std::exp(-0.2 * epoch) + 0.1;

        // Save checkpoint
        auto_checkpoint.step(model, optimizer, epoch, val_loss, "val_loss", &scheduler);

        // Step scheduler
        scheduler.step();
    }

    // Verify best checkpoint exists
    std::string best_path = auto_checkpoint.best_checkpoint_path();
    EXPECT_TRUE(std::filesystem::exists(best_path));

    // Load best checkpoint and verify
    ModelCheckpoint checkpoint_manager;
    auto best_checkpoint = checkpoint_manager.load(best_path);

    EXPECT_GT(best_checkpoint.model_state.size(), 0);
    EXPECT_GT(best_checkpoint.optimizer_state.size(), 0);
    EXPECT_GT(best_checkpoint.metadata.epoch, 0);
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
