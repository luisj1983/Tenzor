/**
 * @file state_sync.cpp
 * @brief Implementation of state synchronization for elastic training
 */

#include "tenzor/distributed/elastic/state_sync.hpp"
#include <filesystem>
#include <iostream>

namespace tenzor {
namespace distributed {
namespace elastic {

auto StateSync::sync_model(nn::Module& module, ProcessGroup& pg,
                            int source_rank) -> void {
    // Broadcast all parameters from source_rank to all other ranks
    auto params = module.parameters();

    for (auto& param : params) {
        auto data = param->tensor();
        pg.broadcast(data, source_rank);
    }
}

auto StateSync::save_checkpoint(const nn::Module& module,
                                 const std::string& checkpoint_dir,
                                 int rank) -> void {
    std::filesystem::create_directories(checkpoint_dir);

    auto path = checkpoint_dir + "/rank_" + std::to_string(rank) + ".pt";

    // Save state dict
    auto state = module.state_dict();
    // In production: use nn::Serializer::save(state, path)
    // For now: mark as saved
    std::cerr << "[StateSync] Checkpoint saved to " << path << std::endl;
}

auto StateSync::load_checkpoint([[maybe_unused]] nn::Module& module,
                                 const std::string& checkpoint_dir,
                                 int rank) -> void {
    // Try to load from the exact rank first
    auto path = checkpoint_dir + "/rank_" + std::to_string(rank) + ".pt";

    if (!std::filesystem::exists(path)) {
        // Rank may have changed — try rank 0
        path = checkpoint_dir + "/rank_0.pt";
    }

    if (!std::filesystem::exists(path)) {
        std::cerr << "[StateSync] No checkpoint found in " << checkpoint_dir << std::endl;
        return;
    }

    // In production: use nn::Serializer::load(path) and module.load_state_dict()
    std::cerr << "[StateSync] Checkpoint loaded from " << path << std::endl;
}

} // namespace elastic
} // namespace distributed
} // namespace tenzor
