/**
 * @file checkpoint.hpp
 * @brief Model checkpointing for saving and loading trained models
 *
 * Provides utilities for serializing model state, optimizer state,
 * and training metadata to disk. Supports versioning, compression,
 * and backward compatibility.
 */

#pragma once

#include <string>
#include <unordered_map>
#include <memory>
#include <optional>
#include <vector>
#include <cstdint>
#include "../core/tensor.hpp"
#include "module.hpp"
#include "optim/optimizer.hpp"
#include "optim/scheduler.hpp"

namespace tenzor {
namespace nn {

/**
 * @brief Checkpoint file format version
 *
 * v2 (audit K.2): adds an `rng_state` section after scheduler_state
 * so save/load survives across the dropout / BN-noise / sampling
 * random sequence.  v1 files are read with rng_state left empty —
 * callers should treat that as "RNG was not captured".
 */
constexpr uint32_t CHECKPOINT_VERSION = 2;

/**
 * @brief Checkpoint file magic number for format identification
 */
constexpr uint32_t CHECKPOINT_MAGIC = 0x544E5A52;  // "TNZR" in hex

/**
 * @brief Compression type for checkpoint data
 */
enum class CompressionType : uint8_t {
    None = 0,      ///< No compression
    LZ4 = 1,       ///< LZ4 fast compression
    Zstd = 2,      ///< Zstandard compression (best ratio)
    Gzip = 3       ///< Gzip compression (widely supported)
};

/**
 * @brief Training metadata stored with checkpoint
 */
struct TrainingMetadata {
    int epoch{0};                           ///< Current training epoch
    int global_step{0};                     ///< Total training steps
    double learning_rate{0.0};              ///< Current learning rate
    double train_loss{0.0};                 ///< Last training loss
    double val_loss{0.0};                   ///< Last validation loss
    double train_accuracy{0.0};             ///< Last training accuracy
    double val_accuracy{0.0};               ///< Last validation accuracy
    double best_val_loss{std::numeric_limits<double>::infinity()};  ///< Best validation loss
    double best_val_accuracy{0.0};          ///< Best validation accuracy
    std::string timestamp;                  ///< Checkpoint creation time
    std::unordered_map<std::string, double> custom_metrics;  ///< User-defined metrics

    /**
     * @brief Serialize metadata to map
     *
     * @return String-to-string map of metadata
     */
    auto to_dict() const -> std::unordered_map<std::string, std::string>;

    /**
     * @brief Deserialize metadata from map
     *
     * @param dict String-to-string map of metadata
     */
    auto from_dict(const std::unordered_map<std::string, std::string>& dict) -> void;
};

/**
 * @brief Checkpoint configuration
 */
struct CheckpointConfig {
    CompressionType compression{CompressionType::None};  ///< Compression algorithm
    int compression_level{-1};                            ///< Compression level (-1 = default)
    bool save_optimizer{true};                            ///< Include optimizer state
    bool save_scheduler{true};                            ///< Include scheduler state
    bool verify_checksum{true};                           ///< Verify data integrity
    bool atomic_save{true};                               ///< Use atomic writes (temp + rename)
};

/**
 * @brief Complete checkpoint data structure
 */
struct Checkpoint {
    uint32_t version{CHECKPOINT_VERSION};                ///< Format version
    std::unordered_map<std::string, Tensor> model_state;  ///< Model parameters and buffers
    std::unordered_map<std::string, Tensor> optimizer_state;  ///< Optimizer state
    std::unordered_map<std::string, Tensor> scheduler_state;  ///< Scheduler state
    /// Per-device RNG snapshots (audit K.2).  Each entry is keyed
    /// "<device_type>:<index>" (e.g. "cpu:0", "cuda:0") and stores the
    /// concatenated (seed, initial_seed, engine_state[..]) as a 1-D
    /// Int64 tensor.  ModelCheckpoint::save() populates this from
    /// tenzor::default_generator(device) for every device the model
    /// touches; ModelCheckpoint::load() restores via Generator::set_state.
    std::unordered_map<std::string, Tensor> rng_state;
    TrainingMetadata metadata;                            ///< Training metadata
    CheckpointConfig config;                              ///< Checkpoint configuration

    /**
     * @brief Get total size of checkpoint in bytes
     *
     * @return Total memory footprint
     */
    auto size_bytes() const -> size_t;

    /**
     * @brief Check if checkpoint is valid
     *
     * @return true if checkpoint has required fields
     */
    auto is_valid() const -> bool;
};

/**
 * @brief Model checkpoint manager
 *
 * Handles saving and loading of model checkpoints with support for:
 * - Model state (parameters, buffers)
 * - Optimizer state (momentum, adaptive learning rates, etc.)
 * - Scheduler state (step counts, learning rate history)
 * - Training metadata (epoch, loss, metrics)
 * - Versioning and backward compatibility
 * - Optional compression
 * - Atomic writes for crash safety
 *
 * File Format:
 * [Header: Magic(4) + Version(4) + Config(N)]
 * [Model State: NameLen(4) + Name + TensorData]
 * [Optimizer State: NameLen(4) + Name + TensorData]
 * [Scheduler State: NameLen(4) + Name + TensorData]
 * [Metadata: JSON string]
 * [Footer: Checksum(8)]
 *
 * @par Thread Safety
 * Not thread-safe. Use mutex if saving/loading from multiple threads.
 *
 * @code
 * // Save checkpoint
 * ModelCheckpoint checkpoint;
 * checkpoint.save(
 *     "model_epoch_10.pt",
 *     model,
 *     optimizer,
 *     scheduler,
 *     {.epoch = 10, .train_loss = 0.25, .val_loss = 0.30}
 * );
 *
 * // Load checkpoint
 * auto loaded = checkpoint.load("model_epoch_10.pt");
 * model->load_state_dict(loaded.model_state);
 * optimizer->load_state_dict(loaded.optimizer_state);
 * std::cout << "Resuming from epoch " << loaded.metadata.epoch << std::endl;
 * @endcode
 */
class ModelCheckpoint {
public:
    /**
     * @brief Construct model checkpoint manager
     *
     * @param config Checkpoint configuration
     */
    explicit ModelCheckpoint(CheckpointConfig config = CheckpointConfig{});

    /**
     * @brief Save complete checkpoint to file
     *
     * Saves model, optimizer, scheduler state, and metadata atomically.
     *
     * @param path File path for checkpoint
     * @param module Model to save
     * @param optimizer Optimizer to save (optional)
     * @param scheduler Learning rate scheduler to save (optional)
     * @param metadata Training metadata (optional)
     * @throws std::runtime_error if save fails
     *
     * @code
     * checkpoint.save(
     *     "checkpoint.pt",
     *     model,
     *     &optimizer,
     *     &scheduler,
     *     {.epoch = 5, .train_loss = 0.3}
     * );
     * @endcode
     */
    auto save(
        const std::string& path,
        const Module& module,
        const optim::Optimizer* optimizer = nullptr,
        const optim::LRScheduler* scheduler = nullptr,
        const TrainingMetadata& metadata = TrainingMetadata{}
    ) -> void;

    /**
     * @brief Load complete checkpoint from file
     *
     * @param path File path to load checkpoint from
     * @return Loaded checkpoint data
     * @throws std::runtime_error if load fails or version incompatible
     *
     * @code
     * auto checkpoint = manager.load("checkpoint.pt");
     * model->load_state_dict(checkpoint.model_state);
     * @endcode
     */
    auto load(const std::string& path) -> Checkpoint;

    /**
     * @brief Save only model state (no optimizer/scheduler)
     *
     * Lightweight checkpoint for inference or model sharing.
     *
     * @param path File path for model
     * @param module Model to save
     * @param metadata Optional metadata
     */
    auto save_model(
        const std::string& path,
        const Module& module,
        const TrainingMetadata& metadata = TrainingMetadata{}
    ) -> void;

    /**
     * @brief Load only model state
     *
     * @param path File path to load from
     * @return Model state dictionary
     */
    auto load_model(const std::string& path) -> std::unordered_map<std::string, Tensor>;

    /**
     * @brief Verify checkpoint file integrity
     *
     * Checks magic number, version, and checksum without fully loading.
     *
     * @param path Checkpoint file path
     * @return true if checkpoint is valid
     */
    auto verify_checkpoint(const std::string& path) -> bool;

    /**
     * @brief Get checkpoint metadata without loading full checkpoint
     *
     * Reads only metadata for quick inspection.
     *
     * @param path Checkpoint file path
     * @return Training metadata
     */
    auto get_metadata(const std::string& path) -> TrainingMetadata;

    /**
     * @brief Get checkpoint version
     *
     * @param path Checkpoint file path
     * @return Version number
     */
    auto get_version(const std::string& path) -> uint32_t;

    /**
     * @brief Check if checkpoint is compatible with current version
     *
     * @param path Checkpoint file path
     * @return true if checkpoint can be loaded
     */
    auto is_compatible(const std::string& path) -> bool;

    /**
     * @brief Get current configuration
     *
     * @return Checkpoint configuration
     */
    auto config() const -> const CheckpointConfig& { return config_; }

    /**
     * @brief Set configuration
     *
     * @param config New configuration
     */
    auto set_config(CheckpointConfig config) -> void { config_ = std::move(config); }

private:
    CheckpointConfig config_;

    /**
     * @brief Write checkpoint to file
     *
     * @param path Output file path
     * @param checkpoint Checkpoint data to write
     */
    auto write_checkpoint(const std::string& path, const Checkpoint& checkpoint) -> void;

    /**
     * @brief Read checkpoint from file
     *
     * @param path Input file path
     * @return Loaded checkpoint
     */
    auto read_checkpoint(const std::string& path) -> Checkpoint;

    /**
     * @brief Compute checksum for data integrity
     *
     * @param data Data to checksum
     * @param size Data size in bytes
     * @return CRC64 checksum
     */
    auto compute_checksum(const void* data, size_t size) -> uint64_t;

    /**
     * @brief Compress data buffer
     *
     * @param input Input data
     * @param input_size Input size in bytes
     * @param output Output buffer (allocated by function)
     * @param output_size Output size after compression
     * @return true if compression successful
     */
    auto compress_data(
        const void* input,
        size_t input_size,
        std::vector<uint8_t>& output
    ) -> bool;

    /**
     * @brief Decompress data buffer
     *
     * @param input Compressed input data
     * @param input_size Input size in bytes
     * @param output Output buffer (allocated by function)
     * @param expected_size Expected output size
     * @return true if decompression successful
     */
    auto decompress_data(
        const void* input,
        size_t input_size,
        std::vector<uint8_t>& output,
        size_t expected_size
    ) -> bool;

    /**
     * @brief Get current timestamp string
     *
     * @return ISO 8601 formatted timestamp
     */
    auto get_timestamp() -> std::string;
};

/**
 * @brief Automatic checkpoint manager for training loops
 *
 * Automatically saves checkpoints at specified intervals and
 * keeps only the best N checkpoints based on a metric.
 *
 * Features:
 * - Save every N epochs
 * - Save every N steps
 * - Keep top K checkpoints by metric
 * - Early stopping integration
 * - Automatic cleanup of old checkpoints
 *
 * @code
 * AutoCheckpoint auto_checkpoint("./checkpoints", 5);  // Keep top 5
 * auto_checkpoint.set_metric_mode("min");  // Lower is better
 *
 * for (int epoch = 0; epoch < num_epochs; ++epoch) {
 *     // Training code...
 *     double val_loss = validate(model);
 *
 *     // Automatically saves and manages checkpoints
 *     auto_checkpoint.step(
 *         model,
 *         optimizer,
 *         epoch,
 *         val_loss,
 *         "val_loss"
 *     );
 * }
 *
 * // Get path to best checkpoint
 * std::string best_path = auto_checkpoint.best_checkpoint_path();
 * @endcode
 */
class AutoCheckpoint {
public:
    /**
     * @brief Construct auto checkpoint manager
     *
     * @param directory Directory to save checkpoints
     * @param max_checkpoints Maximum number of checkpoints to keep (default: 3)
     * @param save_frequency Save every N epochs (default: 1)
     */
    AutoCheckpoint(
        std::string directory,
        int max_checkpoints = 3,
        int save_frequency = 1
    );

    /**
     * @brief Step function to call after each epoch/step
     *
     * Decides whether to save a checkpoint based on configuration.
     *
     * @param module Model to save
     * @param optimizer Optimizer to save
     * @param epoch Current epoch number
     * @param metric_value Current metric value
     * @param metric_name Metric name for tracking
     * @param scheduler Optional scheduler to save
     * @return true if checkpoint was saved
     */
    auto step(
        const Module& module,
        const optim::Optimizer& optimizer,
        int epoch,
        double metric_value,
        const std::string& metric_name,
        const optim::LRScheduler* scheduler = nullptr
    ) -> bool;

    /**
     * @brief Set metric optimization mode
     *
     * @param mode "min" or "max" (default: "min")
     *
     * @code
     * auto_checkpoint.set_metric_mode("min");  // For loss
     * auto_checkpoint.set_metric_mode("max");  // For accuracy
     * @endcode
     */
    auto set_metric_mode(const std::string& mode) -> void;

    /**
     * @brief Get path to best checkpoint
     *
     * @return File path of best checkpoint
     */
    auto best_checkpoint_path() const -> std::string;

    /**
     * @brief Get best metric value
     *
     * @return Best metric value achieved
     */
    auto best_metric_value() const -> double { return best_metric_value_; }

    /**
     * @brief Get list of all checkpoint paths
     *
     * @return Vector of checkpoint file paths
     */
    auto checkpoint_paths() const -> std::vector<std::string>;

    /**
     * @brief Clean up old checkpoints (keep only top K)
     */
    auto cleanup() -> void;

private:
    std::string directory_;
    int max_checkpoints_;
    int save_frequency_;
    std::string metric_mode_{"min"};  // "min" or "max"

    double best_metric_value_;
    std::string best_checkpoint_path_;

    struct CheckpointInfo {
        std::string path;
        double metric_value;
        int epoch;
        std::string timestamp;
    };

    std::vector<CheckpointInfo> checkpoints_;

    /**
     * @brief Check if new metric is better than best
     *
     * @param new_value New metric value
     * @return true if new value is better
     */
    auto is_better(double new_value) const -> bool;

    /**
     * @brief Generate checkpoint filename
     *
     * @param epoch Epoch number
     * @param metric_value Metric value
     * @return Checkpoint filename
     */
    auto generate_filename(int epoch, double metric_value) const -> std::string;
};

} // namespace nn
} // namespace tenzor
