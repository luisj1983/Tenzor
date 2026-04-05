/**
 * @file elastic_trainer.hpp
 * @brief Fault-tolerant elastic training orchestrator
 *
 * Enables distributed training jobs to survive worker failures.
 * Recovery sequence: detect failure -> checkpoint -> teardown old PG
 * -> re-rendezvous -> new PG -> re-shard -> resume.
 */

#pragma once

#include <functional>
#include <memory>
#include <string>
#include "rendezvous.hpp"
#include "health_monitor.hpp"
#include "../../nn/module.hpp"

namespace tenzor {

namespace nn { class Module; }

namespace distributed {
namespace elastic {

/**
 * @brief Configuration for elastic training.
 */
struct ElasticConfig {
    RendezvousConfig rendezvous;
    HealthMonitorConfig health;
    std::string checkpoint_dir{"/tmp/tenzor_elastic"};
    int32_t max_restarts{3};           ///< Maximum recovery attempts
    bool auto_checkpoint{true};        ///< Checkpoint on failure detection
};

/**
 * @brief Training function signature.
 *
 * Takes rank and world_size, performs training until completion
 * or exception.
 */
using TrainFunction = std::function<void(int32_t rank, int32_t world_size)>;

/**
 * @brief Elastic training orchestrator.
 *
 * Wraps a training function with fault tolerance. When a worker fails:
 * 1. Health monitor detects failure (within 15-30s)
 * 2. Surviving workers checkpoint model state
 * 3. Old process group is torn down
 * 4. Workers re-rendezvous with new membership
 * 5. New process group is created
 * 6. Model/optimizer states are re-sharded for new world_size
 * 7. Training resumes from checkpoint
 *
 * Usage:
 * @code
 * ElasticTrainer trainer(config);
 * trainer.run([](int32_t rank, int32_t world_size) {
 *     // Your training code here
 *     auto model = create_model();
 *     auto ddp = DistributedDataParallel(model, ...);
 *     for (auto& batch : dataloader) {
 *         ddp.forward(batch);
 *         // ...
 *     }
 * });
 * @endcode
 */
class ElasticTrainer {
public:
    explicit ElasticTrainer(ElasticConfig config);

    /**
     * @brief Run training with elastic fault tolerance.
     *
     * Manages the full lifecycle: rendezvous, training, failure recovery.
     * Returns when training completes or max_restarts is exceeded.
     *
     * @param train_fn Training function to execute
     */
    auto run(TrainFunction train_fn) -> void;

    /**
     * @brief Check for failures and recover if needed.
     *
     * Can be called periodically from an existing training loop
     * instead of using run().
     *
     * @return true if recovery was needed (training state may have changed)
     */
    auto check_and_recover() -> bool;

    /**
     * @brief Get current rank after most recent rendezvous.
     */
    auto rank() const -> int32_t { return current_rank_; }

    /**
     * @brief Get current world size.
     */
    auto world_size() const -> int32_t { return current_world_size_; }

    /**
     * @brief Get number of restarts so far.
     */
    auto restart_count() const -> int32_t { return restart_count_; }

private:
    ElasticConfig config_;
    int32_t current_rank_{-1};
    int32_t current_world_size_{0};
    int32_t restart_count_{0};
    std::unique_ptr<C10dRendezvous> rendezvous_;
    std::unique_ptr<HealthMonitor> health_monitor_;
};

/**
 * @brief Launch elastic training with process spawning.
 *
 * Combines spawn() from launch.hpp with elastic fault tolerance.
 *
 * @param config Elastic configuration
 * @param train_fn Training function
 */
auto elastic_launch(ElasticConfig config, TrainFunction train_fn) -> void;

} // namespace elastic
} // namespace distributed
} // namespace tenzor
