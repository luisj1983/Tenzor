/**
 * @file elastic_trainer.cpp
 * @brief Implementation of elastic training orchestrator
 */

#include "tenzor/distributed/elastic/elastic_trainer.hpp"
#include "tenzor/distributed/rpc/rpc.hpp"
#include "tenzor/distributed/rpc/rpc_agent.hpp"
#include "tenzor/utils/log.hpp"
#include <filesystem>  // Audit J15: checkpoint dir creation
#include <fstream>     // Audit J15: marker file write
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
            // Audit I.4: route to unified logger.
            TENZOR_LOG_ERROR("[ElasticTrainer] Rendezvous failed: {}", e.what());
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
            // Audit I.4: route to unified logger.
            TENZOR_LOG_ERROR("[ElasticTrainer] Training failed (attempt {}/{}): {}",
                             restart_count_ + 1,
                             config_.max_restarts + 1,
                             e.what());

            // Audit J15: real checkpoint hook.
            //
            // Phase 3: Checkpoint if configured. The trainer doesn't have
            // direct access to the user's model/optimizer (they live inside
            // train_fn's closure), so checkpointing goes through the user-
            // provided callback. If no callback was registered, we still
            // write a small recovery-marker file so failures are observable
            // in the checkpoint dir.
            if (config_.auto_checkpoint) {
                try {
                    std::filesystem::create_directories(config_.checkpoint_dir);
                    if (config_.checkpoint_fn) {
                        std::string path = config_.checkpoint_dir +
                            "/rank" + std::to_string(current_rank_) +
                            "_attempt" + std::to_string(restart_count_) + ".ckpt";
                        config_.checkpoint_fn(path, current_rank_);
                    } else {
                        std::string marker = config_.checkpoint_dir +
                            "/recovery_rank" + std::to_string(current_rank_) +
                            "_attempt" + std::to_string(restart_count_) + ".marker";
                        std::ofstream m(marker);
                        m << "elastic_trainer recovery: rank=" << current_rank_
                          << " attempt=" << restart_count_ << "\n"
                          << "exception=" << e.what() << "\n";
                    }
                } catch (const std::exception& ce) {
                    // Audit I.4: route to unified logger.
                    TENZOR_LOG_ERROR("[ElasticTrainer] auto_checkpoint hook failed: "
                                     "{} (continuing recovery)", ce.what());
                }
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
    // Lazily construct and start the health monitor on first use. The trainer
    // doesn't own the RPC agent (the user initializes it via init_rpc()), so we
    // pull the global agent here. Without an initialized RPC agent there is no
    // way to probe peers, so recovery is genuinely unavailable — report it once.
    if (!health_monitor_) {
        std::shared_ptr<rpc::TcpRpcAgent> agent;
        try {
            agent = rpc::get_agent();
        } catch (const std::exception& e) {
            TENZOR_LOG_WARN("[ElasticTrainer] check_and_recover() requires an "
                            "initialized RPC agent (call init_rpc() first): {}",
                            e.what());
            return false;
        }
        health_monitor_ = std::make_unique<HealthMonitor>(agent, config_.health);
        // Monitor every peer in the current world except ourselves.
        std::vector<int32_t> worker_ids;
        worker_ids.reserve(static_cast<size_t>(
            current_world_size_ > 0 ? current_world_size_ - 1 : 0));
        for (int32_t id = 0; id < current_world_size_; ++id) {
            if (id != current_rank_) worker_ids.push_back(id);
        }
        health_monitor_->start(worker_ids);
    }

    auto dead = health_monitor_->dead_workers();
    if (dead.empty()) return false;

    // Audit I.4: route to unified logger.
    TENZOR_LOG_WARN("[ElasticTrainer] Detected {} dead worker(s), recovering...",
                    dead.size());

    // Audit J15: real checkpoint hook (mirror of the failure-path branch
    // above). Invoked on health-monitor-detected dead workers.
    if (config_.auto_checkpoint) {
        try {
            std::filesystem::create_directories(config_.checkpoint_dir);
            if (config_.checkpoint_fn) {
                std::string path = config_.checkpoint_dir +
                    "/rank" + std::to_string(current_rank_) +
                    "_recover.ckpt";
                config_.checkpoint_fn(path, current_rank_);
            } else {
                std::string marker = config_.checkpoint_dir +
                    "/recovery_rank" + std::to_string(current_rank_) +
                    "_dead_worker.marker";
                std::ofstream m(marker);
                m << "elastic_trainer recovery on dead-worker detection: rank="
                  << current_rank_ << "\n";
            }
        } catch (const std::exception& ce) {
            // Audit I.4: route to unified logger.
            TENZOR_LOG_ERROR("[ElasticTrainer] auto_checkpoint hook failed: "
                             "{} (continuing recovery)", ce.what());
        }
    }

    // Re-rendezvous
    try {
        rendezvous_->leave();
        auto result = rendezvous_->join();
        current_rank_ = result.rank;
        current_world_size_ = result.world_size;
    } catch (const std::exception& e) {
        // Audit I.4: route to unified logger.
        TENZOR_LOG_ERROR("[ElasticTrainer] Re-rendezvous failed: {}", e.what());
        return false;
    }

    // World membership changed; restart the monitor on the new worker set so it
    // doesn't keep probing departed ranks or miss newly-joined ones.
    if (health_monitor_) {
        health_monitor_->stop();
        std::vector<int32_t> worker_ids;
        worker_ids.reserve(static_cast<size_t>(
            current_world_size_ > 0 ? current_world_size_ - 1 : 0));
        for (int32_t id = 0; id < current_world_size_; ++id) {
            if (id != current_rank_) worker_ids.push_back(id);
        }
        health_monitor_->start(worker_ids);
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
