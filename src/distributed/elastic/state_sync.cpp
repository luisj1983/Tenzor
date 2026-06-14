/**
 * @file state_sync.cpp
 * @brief Implementation of state synchronization for elastic training
 */

#include "tenzor/distributed/elastic/state_sync.hpp"
#include "tenzor/nn/serialize.hpp"
#include "tenzor/nn/module.hpp"
#include "tenzor/utils/log.hpp"
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
        // Bind a REFERENCE to the parameter's stored tensor (as DDP does). For a
        // GPU tensor, broadcast()/recv_tensor rebinds the passed reference's impl
        // (cpu_tensor.to(device)); a local copy would leave the parameter holding
        // stale GPU weights on non-source ranks. The CPU case worked only by luck
        // because the copy shared storage.
        Tensor& data = param->tensor();
        pg.broadcast(data, source_rank);
    }
}

auto StateSync::save_checkpoint(const nn::Module& module,
                                 const std::string& checkpoint_dir,
                                 int rank) -> void {
    std::filesystem::create_directories(checkpoint_dir);

    auto path = checkpoint_dir + "/rank_" + std::to_string(rank) + ".pt";

    // Audit (elastic-D2-followup): actually persist the state dict via the
    // existing Tenzor serializer. Previously this only logged "saved";
    // recovery from a real elastic-trainer crash had nothing to load from.
    auto state = module.state_dict();
    nn::Serializer::save(state, path);
    TENZOR_LOG_INFO("[StateSync] Checkpoint saved to {} ({} tensors)",
                    path, state.size());
}

auto StateSync::load_checkpoint(nn::Module& module,
                                 const std::string& checkpoint_dir,
                                 int rank) -> void {
    // Try to load from the exact rank first
    auto path = checkpoint_dir + "/rank_" + std::to_string(rank) + ".pt";

    if (!std::filesystem::exists(path)) {
        // Rank may have changed — try rank 0
        path = checkpoint_dir + "/rank_0.pt";
    }

    if (!std::filesystem::exists(path)) {
        // Audit I.4: unified logger.
        TENZOR_LOG_WARN("[StateSync] No checkpoint found in {}", checkpoint_dir);
        return;
    }

    // Audit (elastic-D2-followup): actually deserialise and re-install the
    // state dict on the module. Previously the load() body only logged
    // "loaded" and the [[maybe_unused]] parameter never received the
    // saved weights.
    auto state = nn::Serializer::load(path);
    module.load_state_dict(state);
    TENZOR_LOG_INFO("[StateSync] Checkpoint loaded from {} ({} tensors)",
                    path, state.size());
}

} // namespace elastic
} // namespace distributed
} // namespace tenzor
