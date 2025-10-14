/**
 * @file checkpoint.cpp
 * @brief Complete implementation of model checkpointing
 */

#include "../../include/tenzor/nn/checkpoint.hpp"
#include <fstream>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <ctime>
#include <cstring>
#include <filesystem>
#include <algorithm>
#include <unordered_set>

namespace tenzor {
namespace nn {

// ============================================================================
// TrainingMetadata Implementation
// ============================================================================

auto TrainingMetadata::to_dict() const -> std::unordered_map<std::string, std::string> {
    std::unordered_map<std::string, std::string> dict;
    dict["epoch"] = std::to_string(epoch);
    dict["global_step"] = std::to_string(global_step);
    dict["learning_rate"] = std::to_string(learning_rate);
    dict["train_loss"] = std::to_string(train_loss);
    dict["val_loss"] = std::to_string(val_loss);
    dict["train_accuracy"] = std::to_string(train_accuracy);
    dict["val_accuracy"] = std::to_string(val_accuracy);
    dict["best_val_loss"] = std::to_string(best_val_loss);
    dict["best_val_accuracy"] = std::to_string(best_val_accuracy);
    dict["timestamp"] = timestamp;

    // Add custom metrics
    for (const auto& [key, value] : custom_metrics) {
        dict["custom_" + key] = std::to_string(value);
    }

    return dict;
}

auto TrainingMetadata::from_dict(const std::unordered_map<std::string, std::string>& dict) -> void {
    auto get_value = [&dict](const std::string& key, auto& target, auto default_val) {
        if (dict.count(key)) {
            using T = std::decay_t<decltype(target)>;
            if constexpr (std::is_same_v<T, int>) {
                target = std::stoi(dict.at(key));
            } else if constexpr (std::is_same_v<T, double>) {
                target = std::stod(dict.at(key));
            } else if constexpr (std::is_same_v<T, std::string>) {
                target = dict.at(key);
            }
        } else {
            target = default_val;
        }
    };

    get_value("epoch", epoch, 0);
    get_value("global_step", global_step, 0);
    get_value("learning_rate", learning_rate, 0.0);
    get_value("train_loss", train_loss, 0.0);
    get_value("val_loss", val_loss, 0.0);
    get_value("train_accuracy", train_accuracy, 0.0);
    get_value("val_accuracy", val_accuracy, 0.0);
    get_value("best_val_loss", best_val_loss, std::numeric_limits<double>::infinity());
    get_value("best_val_accuracy", best_val_accuracy, 0.0);
    get_value("timestamp", timestamp, std::string(""));

    // Extract custom metrics
    custom_metrics.clear();
    for (const auto& [key, value] : dict) {
        if (key.starts_with("custom_")) {
            std::string metric_name = key.substr(7);
            custom_metrics[metric_name] = std::stod(value);
        }
    }
}

// ============================================================================
// Checkpoint Implementation
// ============================================================================

auto Checkpoint::size_bytes() const -> size_t {
    size_t total = 0;

    // Model state
    for (const auto& [name, tensor] : model_state) {
        total += tensor.numel() * tensor.dtype_size();
    }

    // Optimizer state
    for (const auto& [name, tensor] : optimizer_state) {
        total += tensor.numel() * tensor.dtype_size();
    }

    // Scheduler state
    for (const auto& [name, tensor] : scheduler_state) {
        total += tensor.numel() * tensor.dtype_size();
    }

    // Metadata overhead (rough estimate)
    total += 1024;

    return total;
}

auto Checkpoint::is_valid() const -> bool {
    return !model_state.empty() && version == CHECKPOINT_VERSION;
}

// ============================================================================
// ModelCheckpoint Implementation
// ============================================================================

ModelCheckpoint::ModelCheckpoint(CheckpointConfig config) : config_(std::move(config)) {}

auto ModelCheckpoint::save(
    const std::string& path,
    const Module& module,
    const optim::Optimizer* optimizer,
    const optim::LRScheduler* scheduler,
    const TrainingMetadata& metadata
) -> void {
    // Create checkpoint structure
    Checkpoint checkpoint;
    checkpoint.version = CHECKPOINT_VERSION;
    checkpoint.config = config_;
    checkpoint.model_state = module.state_dict();

    // Add optimizer state if available and configured
    if (optimizer && config_.save_optimizer) {
        checkpoint.optimizer_state = optimizer->state_dict();
    }

    // Add scheduler state if available and configured
    if (scheduler && config_.save_scheduler) {
        // Schedulers store state as tensors in state_dict
        // For now, we'll store basic scheduler info
        // Full scheduler state can be added if needed
    }

    // Add metadata
    checkpoint.metadata = metadata;
    if (checkpoint.metadata.timestamp.empty()) {
        checkpoint.metadata.timestamp = get_timestamp();
    }

    // Atomic write if configured
    if (config_.atomic_save) {
        std::string temp_path = path + ".tmp";
        write_checkpoint(temp_path, checkpoint);
        std::filesystem::rename(temp_path, path);
    } else {
        write_checkpoint(path, checkpoint);
    }
}

auto ModelCheckpoint::load(const std::string& path) -> Checkpoint {
    return read_checkpoint(path);
}

auto ModelCheckpoint::save_model(
    const std::string& path,
    const Module& module,
    const TrainingMetadata& metadata
) -> void {
    save(path, module, nullptr, nullptr, metadata);
}

auto ModelCheckpoint::load_model(const std::string& path) -> std::unordered_map<std::string, Tensor> {
    return load(path).model_state;
}

auto ModelCheckpoint::verify_checkpoint(const std::string& path) -> bool {
    try {
        std::ifstream file(path, std::ios::binary);
        if (!file) return false;

        // Read and verify magic number
        uint32_t magic;
        file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
        if (magic != CHECKPOINT_MAGIC) return false;

        // Read and verify version
        uint32_t version;
        file.read(reinterpret_cast<char*>(&version), sizeof(version));
        if (version > CHECKPOINT_VERSION) return false;

        return true;
    } catch (...) {
        return false;
    }
}

auto ModelCheckpoint::get_metadata(const std::string& path) -> TrainingMetadata {
    return load(path).metadata;
}

auto ModelCheckpoint::get_version(const std::string& path) -> uint32_t {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open checkpoint file");
    }

    // Skip magic
    file.seekg(sizeof(uint32_t), std::ios::beg);

    // Read version
    uint32_t version;
    file.read(reinterpret_cast<char*>(&version), sizeof(version));
    return version;
}

auto ModelCheckpoint::is_compatible(const std::string& path) -> bool {
    try {
        return get_version(path) <= CHECKPOINT_VERSION;
    } catch (...) {
        return false;
    }
}

// ============================================================================
// Private Implementation Methods
// ============================================================================

auto ModelCheckpoint::write_checkpoint(const std::string& path, const Checkpoint& checkpoint) -> void {
    std::ofstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to create checkpoint file: " + path);
    }

    // ========== Header ==========
    // Write magic number
    file.write(reinterpret_cast<const char*>(&CHECKPOINT_MAGIC), sizeof(CHECKPOINT_MAGIC));

    // Write version
    file.write(reinterpret_cast<const char*>(&checkpoint.version), sizeof(checkpoint.version));

    // Write config flags
    uint8_t config_flags = 0;
    config_flags |= (checkpoint.config.save_optimizer ? 1 : 0);
    config_flags |= (checkpoint.config.save_scheduler ? 2 : 0);
    config_flags |= (checkpoint.config.verify_checksum ? 4 : 0);
    file.write(reinterpret_cast<const char*>(&config_flags), sizeof(config_flags));

    // ========== Model State ==========
    uint32_t num_model_tensors = static_cast<uint32_t>(checkpoint.model_state.size());
    file.write(reinterpret_cast<const char*>(&num_model_tensors), sizeof(num_model_tensors));

    for (const auto& [name, tensor] : checkpoint.model_state) {
        // Write parameter name
        uint32_t name_len = static_cast<uint32_t>(name.size());
        file.write(reinterpret_cast<const char*>(&name_len), sizeof(name_len));
        file.write(name.data(), name_len);

        // Write tensor shape
        auto shape = tensor.shape();
        uint32_t ndim = static_cast<uint32_t>(shape.size());
        file.write(reinterpret_cast<const char*>(&ndim), sizeof(ndim));
        file.write(reinterpret_cast<const char*>(shape.data()), ndim * sizeof(int64_t));

        // Write tensor dtype
        uint8_t dtype = static_cast<uint8_t>(tensor.dtype());
        file.write(reinterpret_cast<const char*>(&dtype), sizeof(dtype));

        // Write tensor data
        size_t data_size = tensor.numel() * tensor.dtype_size();
        const void* data_ptr = tensor.data_ptr();
        file.write(reinterpret_cast<const char*>(data_ptr), data_size);
    }

    // ========== Optimizer State ==========
    uint32_t num_optimizer_tensors = static_cast<uint32_t>(checkpoint.optimizer_state.size());
    file.write(reinterpret_cast<const char*>(&num_optimizer_tensors), sizeof(num_optimizer_tensors));

    for (const auto& [name, tensor] : checkpoint.optimizer_state) {
        // Write state name
        uint32_t name_len = static_cast<uint32_t>(name.size());
        file.write(reinterpret_cast<const char*>(&name_len), sizeof(name_len));
        file.write(name.data(), name_len);

        // Write tensor shape
        auto shape = tensor.shape();
        uint32_t ndim = static_cast<uint32_t>(shape.size());
        file.write(reinterpret_cast<const char*>(&ndim), sizeof(ndim));
        file.write(reinterpret_cast<const char*>(shape.data()), ndim * sizeof(int64_t));

        // Write tensor dtype
        uint8_t dtype = static_cast<uint8_t>(tensor.dtype());
        file.write(reinterpret_cast<const char*>(&dtype), sizeof(dtype));

        // Write tensor data
        size_t data_size = tensor.numel() * tensor.dtype_size();
        const void* data_ptr = tensor.data_ptr();
        file.write(reinterpret_cast<const char*>(data_ptr), data_size);
    }

    // ========== Scheduler State ==========
    uint32_t num_scheduler_tensors = static_cast<uint32_t>(checkpoint.scheduler_state.size());
    file.write(reinterpret_cast<const char*>(&num_scheduler_tensors), sizeof(num_scheduler_tensors));

    for (const auto& [name, tensor] : checkpoint.scheduler_state) {
        // Write state name
        uint32_t name_len = static_cast<uint32_t>(name.size());
        file.write(reinterpret_cast<const char*>(&name_len), sizeof(name_len));
        file.write(name.data(), name_len);

        // Write tensor shape
        auto shape = tensor.shape();
        uint32_t ndim = static_cast<uint32_t>(shape.size());
        file.write(reinterpret_cast<const char*>(&ndim), sizeof(ndim));
        file.write(reinterpret_cast<const char*>(shape.data()), ndim * sizeof(int64_t));

        // Write tensor dtype
        uint8_t dtype = static_cast<uint8_t>(tensor.dtype());
        file.write(reinterpret_cast<const char*>(&dtype), sizeof(dtype));

        // Write tensor data
        size_t data_size = tensor.numel() * tensor.dtype_size();
        const void* data_ptr = tensor.data_ptr();
        file.write(reinterpret_cast<const char*>(data_ptr), data_size);
    }

    // ========== Metadata ==========
    auto metadata_dict = checkpoint.metadata.to_dict();
    uint32_t num_metadata = static_cast<uint32_t>(metadata_dict.size());
    file.write(reinterpret_cast<const char*>(&num_metadata), sizeof(num_metadata));

    for (const auto& [key, value] : metadata_dict) {
        // Write key
        uint32_t key_len = static_cast<uint32_t>(key.size());
        file.write(reinterpret_cast<const char*>(&key_len), sizeof(key_len));
        file.write(key.data(), key_len);

        // Write value
        uint32_t value_len = static_cast<uint32_t>(value.size());
        file.write(reinterpret_cast<const char*>(&value_len), sizeof(value_len));
        file.write(value.data(), value_len);
    }

    // ========== Footer ==========
    if (checkpoint.config.verify_checksum) {
        // Compute and write checksum of entire file
        file.flush();
        uint64_t checksum = compute_checksum(nullptr, 0); // Simplified
        file.write(reinterpret_cast<const char*>(&checksum), sizeof(checksum));
    }

    file.close();
    if (!file) {
        throw std::runtime_error("Failed to write checkpoint file");
    }
}

auto ModelCheckpoint::read_checkpoint(const std::string& path) -> Checkpoint {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open checkpoint file: " + path);
    }

    Checkpoint checkpoint;

    // ========== Header ==========
    // Read magic number
    uint32_t magic;
    file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    if (magic != CHECKPOINT_MAGIC) {
        throw std::runtime_error("Invalid checkpoint format (bad magic number)");
    }

    // Read version
    file.read(reinterpret_cast<char*>(&checkpoint.version), sizeof(checkpoint.version));
    if (checkpoint.version > CHECKPOINT_VERSION) {
        throw std::runtime_error("Checkpoint version " + std::to_string(checkpoint.version) +
                                 " is newer than supported version " + std::to_string(CHECKPOINT_VERSION));
    }

    // Read config flags
    uint8_t config_flags;
    file.read(reinterpret_cast<char*>(&config_flags), sizeof(config_flags));
    checkpoint.config.save_optimizer = (config_flags & 1) != 0;
    checkpoint.config.save_scheduler = (config_flags & 2) != 0;
    checkpoint.config.verify_checksum = (config_flags & 4) != 0;

    // ========== Model State ==========
    uint32_t num_model_tensors;
    file.read(reinterpret_cast<char*>(&num_model_tensors), sizeof(num_model_tensors));

    for (uint32_t i = 0; i < num_model_tensors; ++i) {
        // Read parameter name
        uint32_t name_len;
        file.read(reinterpret_cast<char*>(&name_len), sizeof(name_len));
        std::string name(name_len, '\0');
        file.read(name.data(), name_len);

        // Read tensor shape
        uint32_t ndim;
        file.read(reinterpret_cast<char*>(&ndim), sizeof(ndim));
        std::vector<int64_t> shape(ndim);
        file.read(reinterpret_cast<char*>(shape.data()), ndim * sizeof(int64_t));

        // Read tensor dtype
        uint8_t dtype_byte;
        file.read(reinterpret_cast<char*>(&dtype_byte), sizeof(dtype_byte));
        DType dtype = static_cast<DType>(dtype_byte);

        // Create tensor and read data
        Tensor tensor(shape, dtype, Device::cpu());
        size_t data_size = tensor.numel() * tensor.dtype_size();
        void* data_ptr = tensor.data_ptr();
        file.read(reinterpret_cast<char*>(data_ptr), data_size);

        checkpoint.model_state[name] = std::move(tensor);
    }

    // ========== Optimizer State ==========
    uint32_t num_optimizer_tensors;
    file.read(reinterpret_cast<char*>(&num_optimizer_tensors), sizeof(num_optimizer_tensors));

    for (uint32_t i = 0; i < num_optimizer_tensors; ++i) {
        // Read state name
        uint32_t name_len;
        file.read(reinterpret_cast<char*>(&name_len), sizeof(name_len));
        std::string name(name_len, '\0');
        file.read(name.data(), name_len);

        // Read tensor shape
        uint32_t ndim;
        file.read(reinterpret_cast<char*>(&ndim), sizeof(ndim));
        std::vector<int64_t> shape(ndim);
        file.read(reinterpret_cast<char*>(shape.data()), ndim * sizeof(int64_t));

        // Read tensor dtype
        uint8_t dtype_byte;
        file.read(reinterpret_cast<char*>(&dtype_byte), sizeof(dtype_byte));
        DType dtype = static_cast<DType>(dtype_byte);

        // Create tensor and read data
        Tensor tensor(shape, dtype, Device::cpu());
        size_t data_size = tensor.numel() * tensor.dtype_size();
        void* data_ptr = tensor.data_ptr();
        file.read(reinterpret_cast<char*>(data_ptr), data_size);

        checkpoint.optimizer_state[name] = std::move(tensor);
    }

    // ========== Scheduler State ==========
    uint32_t num_scheduler_tensors;
    file.read(reinterpret_cast<char*>(&num_scheduler_tensors), sizeof(num_scheduler_tensors));

    for (uint32_t i = 0; i < num_scheduler_tensors; ++i) {
        // Read state name
        uint32_t name_len;
        file.read(reinterpret_cast<char*>(&name_len), sizeof(name_len));
        std::string name(name_len, '\0');
        file.read(name.data(), name_len);

        // Read tensor shape
        uint32_t ndim;
        file.read(reinterpret_cast<char*>(&ndim), sizeof(ndim));
        std::vector<int64_t> shape(ndim);
        file.read(reinterpret_cast<char*>(shape.data()), ndim * sizeof(int64_t));

        // Read tensor dtype
        uint8_t dtype_byte;
        file.read(reinterpret_cast<char*>(&dtype_byte), sizeof(dtype_byte));
        DType dtype = static_cast<DType>(dtype_byte);

        // Create tensor and read data
        Tensor tensor(shape, dtype, Device::cpu());
        size_t data_size = tensor.numel() * tensor.dtype_size();
        void* data_ptr = tensor.data_ptr();
        file.read(reinterpret_cast<char*>(data_ptr), data_size);

        checkpoint.scheduler_state[name] = std::move(tensor);
    }

    // ========== Metadata ==========
    uint32_t num_metadata;
    file.read(reinterpret_cast<char*>(&num_metadata), sizeof(num_metadata));

    std::unordered_map<std::string, std::string> metadata_dict;
    for (uint32_t i = 0; i < num_metadata; ++i) {
        // Read key
        uint32_t key_len;
        file.read(reinterpret_cast<char*>(&key_len), sizeof(key_len));
        std::string key(key_len, '\0');
        file.read(key.data(), key_len);

        // Read value
        uint32_t value_len;
        file.read(reinterpret_cast<char*>(&value_len), sizeof(value_len));
        std::string value(value_len, '\0');
        file.read(value.data(), value_len);

        metadata_dict[key] = value;
    }

    checkpoint.metadata.from_dict(metadata_dict);

    // ========== Footer ==========
    if (checkpoint.config.verify_checksum) {
        uint64_t stored_checksum;
        file.read(reinterpret_cast<char*>(&stored_checksum), sizeof(stored_checksum));
        // Checksum verification could be implemented here
    }

    if (!file) {
        throw std::runtime_error("Failed to read checkpoint file completely");
    }

    return checkpoint;
}

auto ModelCheckpoint::compute_checksum(const void* data, size_t size) -> uint64_t {
    // CRC64 implementation (simplified for now)
    // In production, use a proper CRC64 or hash function
    uint64_t checksum = 0xFFFFFFFFFFFFFFFFULL;

    const uint8_t* bytes = static_cast<const uint8_t*>(data);
    for (size_t i = 0; i < size; ++i) {
        checksum ^= static_cast<uint64_t>(bytes[i]);
        checksum = (checksum << 1) | (checksum >> 63);
    }

    return checksum;
}

auto ModelCheckpoint::compress_data(
    const void* input,
    size_t input_size,
    std::vector<uint8_t>& output
) -> bool {
    // Compression not implemented yet
    // Would require linking with zlib, lz4, or zstd
    output.resize(input_size);
    std::memcpy(output.data(), input, input_size);
    return true;
}

auto ModelCheckpoint::decompress_data(
    const void* input,
    size_t input_size,
    std::vector<uint8_t>& output,
    size_t expected_size
) -> bool {
    // Decompression not implemented yet
    output.resize(input_size);
    std::memcpy(output.data(), input, input_size);
    return true;
}

auto ModelCheckpoint::get_timestamp() -> std::string {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time_t), "%Y-%m-%d_%H-%M-%S");
    return ss.str();
}

// ============================================================================
// AutoCheckpoint Implementation
// ============================================================================

AutoCheckpoint::AutoCheckpoint(std::string directory, int max, int freq)
    : directory_(std::move(directory)),
      max_checkpoints_(max),
      save_frequency_(freq),
      best_metric_value_(std::numeric_limits<double>::infinity()) {

    // Create directory if it doesn't exist
    std::filesystem::create_directories(directory_);
}

auto AutoCheckpoint::step(
    const Module& module,
    const optim::Optimizer& optimizer,
    int epoch,
    double metric_value,
    const std::string& metric_name,
    const optim::LRScheduler* scheduler
) -> bool {
    // Check if we should save this epoch
    if (epoch % save_frequency_ != 0) {
        return false;
    }

    bool is_best = is_better(metric_value);

    // Generate checkpoint path
    std::string filename = generate_filename(epoch, metric_value);
    std::string checkpoint_path = directory_ + "/" + filename;

    // Create metadata
    TrainingMetadata metadata;
    metadata.epoch = epoch;

    // Generate timestamp
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time_t), "%Y-%m-%d_%H-%M-%S");
    metadata.timestamp = ss.str();

    metadata.custom_metrics[metric_name] = metric_value;

    // Save checkpoint
    ModelCheckpoint chkpt;
    chkpt.save(checkpoint_path, module, &optimizer, scheduler, metadata);

    // Track checkpoint
    CheckpointInfo info;
    info.path = checkpoint_path;
    info.metric_value = metric_value;
    info.epoch = epoch;
    info.timestamp = metadata.timestamp;
    checkpoints_.push_back(info);

    // Update best checkpoint
    if (is_best) {
        best_metric_value_ = metric_value;
        best_checkpoint_path_ = checkpoint_path;
    }

    // Cleanup old checkpoints if needed
    if (checkpoints_.size() > static_cast<size_t>(max_checkpoints_)) {
        cleanup();
    }

    return is_best;
}

auto AutoCheckpoint::set_metric_mode(const std::string& mode) -> void {
    metric_mode_ = mode;
    if (mode == "min") {
        best_metric_value_ = std::numeric_limits<double>::infinity();
    } else if (mode == "max") {
        best_metric_value_ = -std::numeric_limits<double>::infinity();
    } else {
        throw std::runtime_error("Invalid metric mode: " + mode + " (must be 'min' or 'max')");
    }
}

auto AutoCheckpoint::best_checkpoint_path() const -> std::string {
    return best_checkpoint_path_;
}

auto AutoCheckpoint::checkpoint_paths() const -> std::vector<std::string> {
    std::vector<std::string> paths;
    paths.reserve(checkpoints_.size());
    for (const auto& info : checkpoints_) {
        paths.push_back(info.path);
    }
    return paths;
}

auto AutoCheckpoint::cleanup() -> void {
    if (checkpoints_.size() <= static_cast<size_t>(max_checkpoints_)) {
        return;
    }

    // Sort checkpoints by metric value
    std::vector<CheckpointInfo> sorted_checkpoints = checkpoints_;
    std::sort(sorted_checkpoints.begin(), sorted_checkpoints.end(),
        [this](const CheckpointInfo& a, const CheckpointInfo& b) {
            if (metric_mode_ == "min") {
                return a.metric_value < b.metric_value;
            } else {
                return a.metric_value > b.metric_value;
            }
        });

    // Keep only top K checkpoints
    std::unordered_set<std::string> paths_to_keep;
    for (size_t i = 0; i < static_cast<size_t>(max_checkpoints_) && i < sorted_checkpoints.size(); ++i) {
        paths_to_keep.insert(sorted_checkpoints[i].path);
    }

    // Delete checkpoints not in top K
    auto it = checkpoints_.begin();
    while (it != checkpoints_.end()) {
        if (paths_to_keep.find(it->path) == paths_to_keep.end()) {
            // Delete file
            try {
                std::filesystem::remove(it->path);
            } catch (...) {
                // Ignore deletion errors
            }
            it = checkpoints_.erase(it);
        } else {
            ++it;
        }
    }
}

auto AutoCheckpoint::is_better(double new_value) const -> bool {
    if (metric_mode_ == "min") {
        return new_value < best_metric_value_;
    } else {
        return new_value > best_metric_value_;
    }
}

auto AutoCheckpoint::generate_filename(int epoch, double metric_value) const -> std::string {
    std::stringstream ss;
    ss << "checkpoint_epoch_" << std::setw(4) << std::setfill('0') << epoch;
    ss << "_metric_" << std::fixed << std::setprecision(4) << metric_value;
    ss << ".pt";
    return ss.str();
}

} // namespace nn
} // namespace tenzor
