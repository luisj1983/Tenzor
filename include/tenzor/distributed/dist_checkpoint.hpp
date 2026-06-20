/**
 * @file dist_checkpoint.hpp
 * @brief Distributed checkpointing with async save and resharding support
 *
 * Saves each rank's local state independently, enabling:
 * - Async save: training continues during disk I/O
 * - Resharding on load: checkpoint saved with N ranks can be loaded onto M ranks
 *
 * Binary format per shard file ({path}/rank_{rank}.ckpt):
 *   [Header]
 *     magic:       4 bytes  ("TZCK")
 *     version:     4 bytes  (uint32_t, currently 2)
 *     world_size:  8 bytes  (int64_t, world size at save time)
 *     num_entries: 8 bytes  (int64_t)
 *   [Entry] (repeated num_entries times)
 *     name_len:    8 bytes  (int64_t)
 *     name:        name_len bytes  (UTF-8, no null terminator)
 *     ndim:        8 bytes  (int64_t)
 *     shape:       ndim * 8 bytes  (int64_t each)
 *     dtype:       4 bytes  (uint32_t, DType enum value)
 *     is_sharded:  1 byte   (uint8_t, v2+: 1 if this tensor is sharded across
 *                            ranks along shard_dim, 0 if replicated)
 *     shard_dim:   8 bytes  (int64_t, v2+: the dimension this tensor is
 *                            sharded along; meaningful only when is_sharded==1)
 *     data_bytes:  8 bytes  (int64_t)
 *     data:        data_bytes bytes  (raw tensor data, contiguous)
 *
 * Version history:
 *   v1: no is_sharded/shard_dim fields. On resharding load, whether a
 *       parameter is row-sharded was *inferred* from byte-identity across
 *       shards, which mis-classifies byte-identical row-sharded params as
 *       replicated. v1 files are still readable for backward compatibility
 *       (the legacy heuristic is applied) but cannot be saved.
 *   v2: explicit per-tensor is_sharded flag + shard_dim, supplied by the
 *       caller at save time. Resharding uses these directly — no guessing.
 */

#pragma once

#include "../core/tensor.hpp"
#include "../core/dtype.hpp"
#include "../nn/module.hpp"
#include <string>
#include <unordered_map>
#include <future>
#include <functional>
#include <cstdint>
#include <vector>
#include <filesystem>

namespace tenzor::distributed {

/**
 * @brief Per-tensor sharding metadata.
 *
 * Describes how a parameter in the state dict is distributed across ranks at
 * save time. Persisted into the checkpoint so resharding on load is exact
 * rather than inferred from byte content.
 *
 * - A *replicated* tensor (sharded == false): every rank holds the full,
 *   identical tensor (e.g. norm weights, biases, buffers). On load it is taken
 *   as-is from a single shard.
 * - A *sharded* tensor (sharded == true): the global tensor was split along
 *   `dim` and each rank holds one contiguous slice. On load all slices are
 *   concatenated along `dim` and re-split for the new world size.
 */
struct ShardSpec {
    /** @brief True if the tensor is split across ranks along `dim`. */
    bool sharded = false;

    /** @brief Dimension the tensor is sharded along (valid when sharded). */
    int64_t dim = 0;
};

/**
 * @brief Configuration for distributed checkpointing.
 */
struct CheckpointConfig {
    /** @brief Directory to save checkpoint shards */
    std::string storage_path;

    /** @brief Save in a background thread (training continues during I/O) */
    bool async_save = true;

    /** @brief Maximum bytes per shard file (default: 1 GB) */
    int64_t max_shard_size = int64_t{1} << 30;
};

/**
 * @brief Distributed checkpoint with async save and resharding support.
 *
 * Each rank saves its local state independently to a per-rank shard file.
 * On load, shards can be redistributed across a different number of ranks,
 * enabling elastic training (changing world size between save and load).
 *
 * Usage:
 * @code
 * CheckpointConfig config;
 * config.storage_path = "/checkpoints";
 * config.async_save = true;
 *
 * DistributedCheckpoint ckpt(config);
 *
 * // Save (async — returns immediately)
 * auto state = model->state_dict();
 * auto future = ckpt.save_async("step_1000", state, rank, world_size);
 *
 * // ... continue training ...
 *
 * // Wait for save to complete (optional, e.g., before exit)
 * future.get();
 *
 * // Load (potentially with different world size)
 * auto loaded = ckpt.load("step_1000", new_rank, new_world_size);
 * model->load_state_dict(loaded);
 * @endcode
 */
class DistributedCheckpoint {
public:
    /**
     * @brief Construct distributed checkpoint handler.
     *
     * @param config Checkpoint configuration
     */
    explicit DistributedCheckpoint(CheckpointConfig config);

    ~DistributedCheckpoint() = default;

    // Non-copyable (owns futures)
    DistributedCheckpoint(const DistributedCheckpoint&) = delete;
    DistributedCheckpoint& operator=(const DistributedCheckpoint&) = delete;

    // Movable
    DistributedCheckpoint(DistributedCheckpoint&&) noexcept = default;
    DistributedCheckpoint& operator=(DistributedCheckpoint&&) noexcept = default;

    /**
     * @brief Save state dict asynchronously.
     *
     * Copies tensor data into a serialization buffer, then writes to disk
     * in a background thread. The returned future completes when the write
     * finishes. Tensor data is snapshotted at call time — the caller can
     * modify tensors immediately after this returns.
     *
     * @param path Subdirectory name within storage_path (e.g., "step_1000")
     * @param state_dict Map of parameter name to tensor
     * @param rank This process's rank
     * @param world_size Total number of processes
     * @param shard_specs Optional per-tensor sharding metadata. Any tensor not
     *        present in this map (or absent map entirely) is treated as
     *        replicated. Supplying accurate specs is what makes resharding on a
     *        changed world size correct.
     * @return Future that completes when save is done
     */
    auto save_async(const std::string& path,
                    const std::unordered_map<std::string, Tensor>& state_dict,
                    int64_t rank = 0,
                    int64_t world_size = 1,
                    const std::unordered_map<std::string, ShardSpec>& shard_specs = {})
        -> std::future<void>;

    /**
     * @brief Save state dict synchronously (blocks until complete).
     *
     * @param path Subdirectory name within storage_path
     * @param state_dict Map of parameter name to tensor
     * @param rank This process's rank
     * @param world_size Total number of processes
     * @param shard_specs Optional per-tensor sharding metadata (see save_async).
     */
    auto save(const std::string& path,
              const std::unordered_map<std::string, Tensor>& state_dict,
              int64_t rank = 0,
              int64_t world_size = 1,
              const std::unordered_map<std::string, ShardSpec>& shard_specs = {})
        -> void;

    /**
     * @brief Load checkpoint, potentially resharding across different world size.
     *
     * If the checkpoint was saved with a different world_size, this reads all
     * shard files, concatenates tensors, and redistributes them for the new
     * world_size. Each rank loads only its portion.
     *
     * @param path Subdirectory name within storage_path
     * @param rank This process's rank
     * @param world_size Total number of processes
     * @return State dict for this rank
     */
    auto load(const std::string& path,
              int64_t rank = 0,
              int64_t world_size = 1)
        -> std::unordered_map<std::string, Tensor>;

private:
    CheckpointConfig config_;

    static constexpr uint32_t MAGIC = 0x5A43'5A54;  // "TZCK" in little-endian
    // Format version written by this build. v1 files remain readable.
    static constexpr uint32_t VERSION = 2;
    static constexpr uint32_t VERSION_LEGACY_NO_SHARD_META = 1;

    /**
     * @brief Result of deserializing one shard file.
     *
     * Carries the tensors, the saved world size, the format version that
     * produced the file, and the per-tensor sharding metadata (empty for
     * legacy v1 files, where sharding must be inferred).
     */
    struct ShardContents {
        std::unordered_map<std::string, Tensor> tensors;
        int64_t world_size = 0;
        uint32_t version = 0;
        std::unordered_map<std::string, ShardSpec> shard_specs;
    };

    /**
     * @brief Serialize a state dict to binary format (current VERSION).
     *
     * @param state State dict to serialize
     * @param world_size World size to write into the header
     * @param shard_specs Per-tensor sharding metadata (missing => replicated)
     * @return Serialized bytes
     */
    auto serialize_state(const std::unordered_map<std::string, Tensor>& state,
                         int64_t world_size,
                         const std::unordered_map<std::string, ShardSpec>& shard_specs) const
        -> std::vector<uint8_t>;

    /**
     * @brief Deserialize a shard file from binary format.
     *
     * Supports both the current VERSION and the legacy v1 layout.
     *
     * @param data Serialized bytes
     * @return Tensors, saved world_size, file version, and per-tensor specs
     */
    auto deserialize_state(const std::vector<uint8_t>& data) const -> ShardContents;

    /**
     * @brief Build the full filesystem path for a shard file.
     *
     * @param path Checkpoint subdirectory
     * @param rank Rank index
     * @return Full path: {storage_path}/{path}/rank_{rank}.ckpt
     */
    auto shard_path(const std::string& path, int64_t rank) const
        -> std::filesystem::path;
};

} // namespace tenzor::distributed
