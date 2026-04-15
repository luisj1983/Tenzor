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
 *     version:     4 bytes  (uint32_t, currently 1)
 *     world_size:  8 bytes  (int64_t, world size at save time)
 *     num_entries: 8 bytes  (int64_t)
 *   [Entry] (repeated num_entries times)
 *     name_len:    8 bytes  (int64_t)
 *     name:        name_len bytes  (UTF-8, no null terminator)
 *     ndim:        8 bytes  (int64_t)
 *     shape:       ndim * 8 bytes  (int64_t each)
 *     dtype:       4 bytes  (uint32_t, DType enum value)
 *     data_bytes:  8 bytes  (int64_t)
 *     data:        data_bytes bytes  (raw tensor data, contiguous)
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
     * @return Future that completes when save is done
     */
    auto save_async(const std::string& path,
                    const std::unordered_map<std::string, Tensor>& state_dict,
                    int64_t rank = 0,
                    int64_t world_size = 1) -> std::future<void>;

    /**
     * @brief Save state dict synchronously (blocks until complete).
     *
     * @param path Subdirectory name within storage_path
     * @param state_dict Map of parameter name to tensor
     * @param rank This process's rank
     * @param world_size Total number of processes
     */
    auto save(const std::string& path,
              const std::unordered_map<std::string, Tensor>& state_dict,
              int64_t rank = 0,
              int64_t world_size = 1) -> void;

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
    static constexpr uint32_t VERSION = 1;

    /**
     * @brief Serialize a state dict to binary format.
     *
     * @param state State dict to serialize
     * @param world_size World size to write into the header
     * @return Serialized bytes
     */
    auto serialize_state(const std::unordered_map<std::string, Tensor>& state,
                         int64_t world_size) const -> std::vector<uint8_t>;

    /**
     * @brief Deserialize a state dict from binary format.
     *
     * @param data Serialized bytes
     * @return Deserialized state dict and the world_size from the header
     */
    auto deserialize_state(const std::vector<uint8_t>& data) const
        -> std::pair<std::unordered_map<std::string, Tensor>, int64_t>;

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
