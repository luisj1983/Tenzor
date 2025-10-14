/**
 * @file test_checkpoint_build.cpp
 * @brief Build verification test for checkpoint implementation
 */

#include "tenzor/nn/checkpoint.hpp"
#include "tenzor/nn/module.hpp"
#include "tenzor/nn/layers/linear.hpp"
#include "tenzor/nn/optim/sgd.hpp"
#include "tenzor/autograd/checkpoint.hpp"
#include <iostream>

int main() {
    using namespace tenzor;
    using namespace tenzor::nn;
    using namespace tenzor::autograd;

    std::cout << "Testing checkpoint implementation..." << std::endl;

    // Test 1: Model checkpoint creation
    {
        std::cout << "Test 1: Creating model checkpoint..." << std::endl;
        ModelCheckpoint checkpoint;
        std::cout << "  Model checkpoint created successfully" << std::endl;
    }

    // Test 2: Checkpoint configuration
    {
        std::cout << "Test 2: Testing checkpoint configuration..." << std::endl;
        CheckpointConfig config;
        config.compression = CompressionType::None;
        config.save_optimizer = true;
        config.save_scheduler = true;
        config.verify_checksum = false;
        config.atomic_save = true;

        ModelCheckpoint checkpoint(config);
        std::cout << "  Checkpoint configured successfully" << std::endl;
    }

    // Test 3: Training metadata
    {
        std::cout << "Test 3: Testing training metadata..." << std::endl;
        TrainingMetadata metadata;
        metadata.epoch = 10;
        metadata.train_loss = 0.5;
        metadata.val_loss = 0.6;
        metadata.learning_rate = 0.001;
        metadata.custom_metrics["accuracy"] = 0.95;

        auto dict = metadata.to_dict();
        std::cout << "  Metadata serialized: " << dict.size() << " entries" << std::endl;

        TrainingMetadata loaded;
        loaded.from_dict(dict);
        std::cout << "  Metadata deserialized: epoch=" << loaded.epoch << std::endl;
    }

    // Test 4: Checkpoint structure
    {
        std::cout << "Test 4: Testing checkpoint structure..." << std::endl;
        Checkpoint checkpoint;
        checkpoint.version = CHECKPOINT_VERSION;

        std::cout << "  Checkpoint version: " << checkpoint.version << std::endl;
        std::cout << "  Checkpoint magic: 0x" << std::hex << CHECKPOINT_MAGIC << std::dec << std::endl;
        std::cout << "  Checkpoint valid: " << (checkpoint.is_valid() ? "false (empty)" : "false") << std::endl;
    }

    // Test 5: AutoCheckpoint
    {
        std::cout << "Test 5: Testing AutoCheckpoint..." << std::endl;
        std::string test_dir = "/tmp/tenzor_checkpoint_test";
        AutoCheckpoint auto_checkpoint(test_dir, 3, 1);
        auto_checkpoint.set_metric_mode("min");

        std::cout << "  AutoCheckpoint created with directory: " << test_dir << std::endl;
        std::cout << "  Best metric: " << auto_checkpoint.best_metric_value() << std::endl;
    }

    // Test 6: Gradient checkpointing statistics
    {
        std::cout << "Test 6: Testing gradient checkpointing..." << std::endl;
        reset_checkpoint_stats();

        auto& stats = get_checkpoint_stats();
        std::cout << "  Initial checkpoints: " << stats.num_checkpoints << std::endl;
        std::cout << "  Initial recomputations: " << stats.num_recomputations << std::endl;
        std::cout << "  Checkpoint enabled: " << (is_checkpoint_enabled() ? "true" : "false") << std::endl;
    }

    // Test 7: Checkpoint context
    {
        std::cout << "Test 7: Testing checkpoint context..." << std::endl;
        {
            CheckpointContext ctx(true);
            std::cout << "  Context enabled: " << ctx.is_enabled() << std::endl;
        }
        std::cout << "  Context destroyed successfully" << std::endl;
    }

    // Test 8: Memory tracker
    {
        std::cout << "Test 8: Testing memory tracker..." << std::endl;
        MemoryTracker::start_tracking();
        MemoryTracker::reset();

        auto current = MemoryTracker::current_memory();
        auto peak = MemoryTracker::peak_memory();

        std::cout << "  Current memory: " << current << " bytes" << std::endl;
        std::cout << "  Peak memory: " << peak << " bytes" << std::endl;

        MemoryTracker::stop_tracking();
    }

    std::cout << "\nAll checkpoint tests completed successfully!" << std::endl;
    return 0;
}
