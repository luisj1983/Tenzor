/**
 * @file state_sync.cpp
 * @brief Implementation of state synchronization for elastic training
 */

#include "tenzor/distributed/elastic/state_sync.hpp"
#include "tenzor/nn/serialize.hpp"
#include "tenzor/nn/module.hpp"
#include "tenzor/utils/log.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <string>

namespace tenzor {
namespace distributed {
namespace elastic {

namespace {

// Sidecar manifest written next to every checkpoint shard. It records which
// rank the shard was actually saved for so that, after an elastic membership
// change reassigns ranks, load_checkpoint can detect when it is about to
// install a *different* rank's shard (or a stale rank_0 fallback) instead of
// silently loading mismatched weights.
//
// Format (one "key value" pair per line):
//   tenzor_state_sync_manifest 1   // magic + manifest version
//   rank <int>                     // rank this shard belongs to
constexpr const char* kManifestMagic = "tenzor_state_sync_manifest";
constexpr int kManifestVersion = 1;

auto manifest_path_for(const std::string& checkpoint_path) -> std::string {
    return checkpoint_path + ".meta";
}

auto write_manifest(const std::string& checkpoint_path, int rank) -> void {
    std::ofstream meta(manifest_path_for(checkpoint_path), std::ios::trunc);
    if (!meta) {
        // Non-fatal: the checkpoint itself is still valid, we just lose the
        // ability to verify rank provenance on load. Warn so the gap is
        // visible rather than silent.
        TENZOR_LOG_WARN("[StateSync] Could not write checkpoint manifest {}",
                        manifest_path_for(checkpoint_path));
        return;
    }
    meta << kManifestMagic << ' ' << kManifestVersion << '\n';
    meta << "rank " << rank << '\n';
}

// Returns the rank recorded in the shard's manifest, or std::nullopt if the
// manifest is absent/unreadable/malformed (e.g. an older checkpoint written
// before manifests existed).
auto read_manifest_rank(const std::string& checkpoint_path)
    -> std::optional<int> {
    std::ifstream meta(manifest_path_for(checkpoint_path));
    if (!meta) {
        return std::nullopt;
    }
    std::string magic;
    int version = 0;
    if (!(meta >> magic >> version) || magic != kManifestMagic ||
        version != kManifestVersion) {
        return std::nullopt;
    }
    std::string key;
    int saved_rank = 0;
    if (!(meta >> key >> saved_rank) || key != "rank") {
        return std::nullopt;
    }
    return saved_rank;
}

} // namespace

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

    // Also broadcast non-trainable buffers (e.g. BatchNorm running_mean/
    // running_var). parameters() never touches the buffers_ map, so without
    // this the in-memory recovery path would leave non-source ranks with stale/
    // divergent running stats — inconsistent with the disk path, which uses
    // state_dict() and DOES include buffers.
    auto buffers = module.buffers();
    for (auto& buffer : buffers) {
        Tensor& data = buffer->tensor();
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

    // Persist a sidecar manifest recording the owning rank so that a later
    // load (possibly after an elastic rank reassignment) can detect when it is
    // about to install a different rank's shard instead of silently accepting
    // a mismatched/stale checkpoint.
    write_manifest(path, rank);

    TENZOR_LOG_INFO("[StateSync] Checkpoint saved to {} ({} tensors)",
                    path, state.size());
}

auto StateSync::load_checkpoint(nn::Module& module,
                                 const std::string& checkpoint_dir,
                                 int rank) -> void {
    // Try to load from the exact rank first
    auto path = checkpoint_dir + "/rank_" + std::to_string(rank) + ".pt";
    bool used_fallback = false;

    if (!std::filesystem::exists(path)) {
        // Rank may have changed (elastic membership change) — fall back to
        // rank 0. This is a best-effort recovery: rank_0.pt may belong to a
        // different attempt/epoch or be only rank 0's shard, so the load below
        // is verified against the shard's manifest and surfaced as a WARN.
        path = checkpoint_dir + "/rank_0.pt";
        used_fallback = true;
    }

    if (!std::filesystem::exists(path)) {
        // Audit I.4: unified logger.
        TENZOR_LOG_WARN("[StateSync] No checkpoint found in {}", checkpoint_dir);
        return;
    }

    // Verify shard provenance against the sidecar manifest. The manifest may be
    // absent for checkpoints written before manifests existed, in which case we
    // cannot confirm the owning rank.
    auto manifest_rank = read_manifest_rank(path);

    // Audit (elastic-D2-followup): actually deserialise and re-install the
    // state dict on the module. Previously the load() body only logged
    // "loaded" and the [[maybe_unused]] parameter never received the
    // saved weights.
    auto state = nn::Serializer::load(path);
    module.load_state_dict(state);

    if (manifest_rank.has_value() && manifest_rank.value() != rank) {
        // The shard we loaded belongs to a different rank than requested. This
        // happens on the rank_0 fallback after a reassignment, but can also
        // occur if an exact-named file was copied/restored from another rank.
        // Surface it loudly: a mismatched shard may carry stale or foreign
        // weights, which used to be loaded with a misleading INFO "loaded".
        TENZOR_LOG_WARN(
            "[StateSync] Loaded checkpoint {} ({} tensors) saved for rank {} "
            "but requested rank {} — possible stale/mismatched shard after an "
            "elastic membership change; verify training state",
            path, state.size(), manifest_rank.value(), rank);
    } else if (used_fallback) {
        // Fell back to rank_0.pt and the manifest could not confirm the rank
        // (e.g. legacy checkpoint without a manifest). Still warn, since the
        // fallback bypasses the exact-rank match.
        TENZOR_LOG_WARN(
            "[StateSync] Loaded fallback checkpoint {} ({} tensors) for "
            "requested rank {} (rank-specific checkpoint missing, manifest "
            "absent/unverified) — possible stale/mismatched shard",
            path, state.size(), rank);
    } else {
        TENZOR_LOG_INFO("[StateSync] Checkpoint loaded from {} ({} tensors)",
                        path, state.size());
    }
}

} // namespace elastic
} // namespace distributed
} // namespace tenzor
