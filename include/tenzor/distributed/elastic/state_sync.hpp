/**
 * @file state_sync.hpp
 * @brief State synchronization for elastic training recovery
 *
 * Handles re-sharding of model and optimizer states when the training
 * group changes size due to worker join/leave events.
 */

#pragma once

#include <memory>
#include <string>
#include "../distributed.hpp"
#include "../../nn/module.hpp"

namespace tenzor {
namespace distributed {
namespace elastic {

/**
 * @brief Synchronizes model/optimizer state after membership changes.
 *
 * When world_size changes during elastic training:
 * - DDP: Broadcast full model from a surviving rank
 * - FSDP: All-gather shards from old group, re-partition for new world_size
 * - Falls back to disk checkpoint when memory is insufficient
 */
class StateSync {
public:
    /**
     * @brief Synchronize model state across a new process group.
     *
     * For DDP (all ranks have full model), broadcasts from source_rank.
     * For FSDP (sharded), all-gathers then re-shards.
     *
     * @param module Model to synchronize
     * @param pg New process group
     * @param source_rank Rank to broadcast from (for DDP)
     */
    static auto sync_model(nn::Module& module, ProcessGroup& pg,
                           int source_rank = 0) -> void;

    /**
     * @brief Save state to disk for recovery.
     *
     * @param module Model to save
     * @param checkpoint_dir Directory to save checkpoint
     * @param rank Current rank
     */
    static auto save_checkpoint(const nn::Module& module,
                                const std::string& checkpoint_dir,
                                int rank) -> void;

    /**
     * @brief Load state from disk after recovery.
     *
     * @param module Model to load into
     * @param checkpoint_dir Directory containing checkpoint
     * @param rank Current rank (may differ from saved rank)
     */
    static auto load_checkpoint(nn::Module& module,
                                const std::string& checkpoint_dir,
                                int rank) -> void;
};

} // namespace elastic
} // namespace distributed
} // namespace tenzor
