/**
 * @file elastic_trainer.cpp
 * @brief Implementation of elastic training orchestrator
 */

#include "tenzor/distributed/elastic/elastic_trainer.hpp"
#include <iostream>
#include <stdexcept>

namespace tenzor {
namespace distributed {
namespace elastic {

ElasticTrainer::ElasticTrainer(ElasticConfig config)
    : config_(std::move(config)) {
    rendezvous_ = std::make_unique<C10dRendezvous>(config_.rendezvous);
}

auto ElasticTrainer::run(TrainFunction train_fn) -> void {
    while (restart_count_ <= config_.max_restarts) {
        // Phase 1: Rendezvous to get rank assignment
        try {
            auto result = rendezvous_->join();
            current_rank_ = result.rank;
            current_world_size_ = result.world_size;
        } catch (const std::exception& e) {
            std::cerr << "[ElasticTrainer] Rendezvous failed: " << e.what() << std::endl;
            if (restart_count_ >= config_.max_restarts) {
                throw;
            }
            ++restart_count_;
            continue;
        }

        // Phase 2: Run training
        try {
            train_fn(current_rank_, current_world_size_);
            // Training completed successfully
            return;
        } catch (const std::exception& e) {
            std::cerr << "[ElasticTrainer] Training failed (attempt "
                      << restart_count_ + 1 << "/" << config_.max_restarts + 1
                      << "): " << e.what() << std::endl;

            // Phase 3: Checkpoint if configured
            if (config_.auto_checkpoint) {
                // In production: save model, optimizer, scheduler state
                // using nn::ModelCheckpoint
            }

            // Phase 4: Leave current rendezvous
            try {
                rendezvous_->leave();
            } catch (...) {}

            ++restart_count_;

            if (restart_count_ > config_.max_restarts) {
                throw std::runtime_error(
                    "[ElasticTrainer] Max restarts exceeded (" +
                    std::to_string(config_.max_restarts) + ")");
            }
        }
    }
}

auto ElasticTrainer::check_and_recover() -> bool {
    if (!health_monitor_) return false;

    auto dead = health_monitor_->dead_workers();
    if (dead.empty()) return false;

    std::cerr << "[ElasticTrainer] Detected " << dead.size() << " dead worker(s), recovering..."
              << std::endl;

    // Checkpoint
    if (config_.auto_checkpoint) {
        // Save state to config_.checkpoint_dir
    }

    // Re-rendezvous
    try {
        rendezvous_->leave();
        auto result = rendezvous_->join();
        current_rank_ = result.rank;
        current_world_size_ = result.world_size;
    } catch (const std::exception& e) {
        std::cerr << "[ElasticTrainer] Re-rendezvous failed: " << e.what() << std::endl;
        return false;
    }

    ++restart_count_;
    return true;
}

auto elastic_launch(ElasticConfig config, TrainFunction train_fn) -> void {
    ElasticTrainer trainer(std::move(config));
    trainer.run(std::move(train_fn));
}

} // namespace elastic
} // namespace distributed
} // namespace tenzor
