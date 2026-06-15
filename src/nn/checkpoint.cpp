/**
 * @file checkpoint.cpp
 * @brief Complete implementation of model checkpointing
 */

#include "../../include/tenzor/nn/checkpoint.hpp"
#include "../../include/tenzor/nn/safetensors.hpp"
#include "../../include/tenzor/core/generator.hpp"  // K.2 per-device RNG snapshots
#include <fstream>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <ctime>
#include <cstring>
#include <filesystem>
#include <algorithm>
#include <set>
#include <unordered_set>
#include <cstdint>
#include <limits>

namespace tenzor {

namespace {

// Bytes remaining between the current get position and end of stream. Used to
// bound untrusted length/size fields so a corrupt or malicious checkpoint file
// cannot drive huge allocations or reads past the end of the file.
auto ckpt_stream_remaining(std::ifstream& file) -> std::uintmax_t {
    const auto cur = file.tellg();
    if (cur < 0) {
        return 0;
    }
    file.seekg(0, std::ios::end);
    const auto endpos = file.tellg();
    file.seekg(cur, std::ios::beg);
    if (endpos < 0 || endpos < cur) {
        return 0;
    }
    return static_cast<std::uintmax_t>(endpos - cur);
}

// Read a uint32 length, validating it against the bytes left in the file.
auto ckpt_read_length(std::ifstream& file) -> uint32_t {
    uint32_t len = 0;
    file.read(reinterpret_cast<char*>(&len), sizeof(len));
    if (!file) {
        throw std::runtime_error("Checkpoint: unexpected end of file reading length");
    }
    if (static_cast<std::uintmax_t>(len) > ckpt_stream_remaining(file)) {
        throw std::runtime_error("Checkpoint: length field exceeds remaining file size");
    }
    return len;
}

auto ckpt_read_string(std::ifstream& file) -> std::string {
    const uint32_t len = ckpt_read_length(file);
    std::string s(len, '\0');
    file.read(s.data(), static_cast<std::streamsize>(len));
    if (!file) {
        throw std::runtime_error("Checkpoint: unexpected end of file reading string");
    }
    return s;
}

// Read a tensor (shape, dtype, data) with full validation of shape rank,
// per-dimension non-negativity, element-count overflow, dtype validity, and
// that the data payload fits within the remaining file bytes.
auto ckpt_read_tensor(std::ifstream& file) -> Tensor {
    uint32_t ndim = 0;
    file.read(reinterpret_cast<char*>(&ndim), sizeof(ndim));
    if (!file) {
        throw std::runtime_error("Checkpoint: unexpected end of file reading rank");
    }
    if (static_cast<std::uintmax_t>(ndim) * sizeof(int64_t) > ckpt_stream_remaining(file)) {
        throw std::runtime_error("Checkpoint: shape rank exceeds remaining file size");
    }
    std::vector<int64_t> shape(ndim);
    file.read(reinterpret_cast<char*>(shape.data()),
              static_cast<std::streamsize>(ndim * sizeof(int64_t)));
    if (!file) {
        throw std::runtime_error("Checkpoint: unexpected end of file reading shape");
    }
    int64_t numel = 1;
    for (uint32_t d = 0; d < ndim; ++d) {
        if (shape[d] < 0) {
            throw std::runtime_error("Checkpoint: negative dimension in shape");
        }
        if (shape[d] != 0 && numel > std::numeric_limits<int64_t>::max() / shape[d]) {
            throw std::runtime_error("Checkpoint: tensor element count overflow");
        }
        numel *= shape[d];
    }

    uint8_t dtype_byte = 0;
    file.read(reinterpret_cast<char*>(&dtype_byte), sizeof(dtype_byte));
    if (!file) {
        throw std::runtime_error("Checkpoint: unexpected end of file reading dtype");
    }
    const DType dtype = static_cast<DType>(dtype_byte);
    switch (dtype) {
        case DType::Float32: case DType::Float64:
        case DType::Int32:   case DType::Int64:
        case DType::UInt8:   case DType::Bool:
        case DType::Float16: case DType::BFloat16:
        case DType::Int8:    case DType::Int16:
            break;
        default:
            throw std::runtime_error("Checkpoint: unknown dtype in file: " +
                std::to_string(static_cast<int>(dtype)));
    }

    Tensor tensor(shape, dtype, Device::cpu());
    const size_t data_size = tensor.numel() * tensor.dtype_size();
    if (static_cast<std::uintmax_t>(data_size) > ckpt_stream_remaining(file)) {
        throw std::runtime_error("Checkpoint: tensor data exceeds remaining file size");
    }
    file.read(reinterpret_cast<char*>(tensor.data_ptr()),
              static_cast<std::streamsize>(data_size));
    if (!file) {
        throw std::runtime_error("Checkpoint: unexpected end of file reading tensor data");
    }
    return tensor;
}

} // namespace
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

    // Add scheduler state if available and configured (audit Q.11).
    //
    // Prior to v3 this block wrote num_scheduler_tensors=0 even when
    // save_scheduler was set, so LR schedules silently reset on every
    // resume.  Now we serialise LRScheduler::state_dict() — base class
    // captures last_lr; derived classes overlay their own counters
    // (step_count_, epoch_, T_cur_, …).
    if (scheduler && config_.save_scheduler) {
        checkpoint.scheduler_state = scheduler->state_dict();
    }

    // Add metadata
    checkpoint.metadata = metadata;
    if (checkpoint.metadata.timestamp.empty()) {
        checkpoint.metadata.timestamp = get_timestamp();
    }

    // Audit K.2: capture per-device RNG snapshots so resuming reproduces
    // dropout / BN-noise / sampling exactly.  We always snapshot the CPU
    // default generator; we also snapshot any GPU device that holds at
    // least one model parameter so the model's stochastic forward path
    // resumes deterministically.
    {
        auto pack_state = [](const tenzor::GeneratorState& s) -> Tensor {
            // Layout: [seed, initial_seed, engine_state[0..N-1]]
            const size_t n = 2 + s.engine_state.size();
            Tensor t({static_cast<int64_t>(n)}, DType::Int64, Device::cpu());
            auto* p = t.data<int64_t>();
            p[0] = static_cast<int64_t>(s.seed);
            p[1] = static_cast<int64_t>(s.initial_seed);
            for (size_t i = 0; i < s.engine_state.size(); ++i) {
                p[2 + i] = static_cast<int64_t>(s.engine_state[i]);
            }
            return t;
        };
        auto dev_key = [](Device::Type dt, int32_t idx) -> std::string {
            const char* name = "cpu";
            switch (dt) {
                case Device::Type::CPU:    name = "cpu";    break;
                case Device::Type::CUDA:   name = "cuda";   break;
                case Device::Type::ROCm:   name = "rocm";   break;
                case Device::Type::OneAPI: name = "oneapi"; break;
                case Device::Type::Vulkan: name = "vulkan"; break;
                case Device::Type::MPS:    name = "mps";    break;
                default: name = "cpu"; break;  // sentinel / COUNT — should never hit
            }
            return std::string(name) + ":" + std::to_string(idx);
        };
        // CPU default generator: always snapshot.
        try {
            checkpoint.rng_state[dev_key(Device::Type::CPU, 0)] =
                pack_state(tenzor::default_generator(Device::cpu()).get_state());
        } catch (const std::exception&) {
            // CPU generator should always exist; ignore if not.
        }
        // Per-GPU snapshot for every device the model touches.
        std::set<std::pair<int, int32_t>> devices_seen;
        for (const auto& [name, t] : checkpoint.model_state) {
            const auto& d = t.device();
            if (d.type != Device::Type::CPU) {
                devices_seen.insert({static_cast<int>(d.type), d.index});
            }
        }
        for (const auto& [dt_int, idx] : devices_seen) {
            Device dev{static_cast<Device::Type>(dt_int), idx};
            try {
                const auto& gen = tenzor::default_generator(dev);
                checkpoint.rng_state[dev_key(dev.type, idx)] =
                    pack_state(gen.get_state());
            } catch (const std::exception&) {
                // Generator for this device may not be exposed yet
                // (e.g., Vulkan currently); skip rather than fail the
                // save — partial RNG coverage is better than none.
            }
        }
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
    Checkpoint c = read_checkpoint(path);

    // Audit K.2: restore per-device RNG snapshots.  This runs at load()
    // time (not inside the user's resume loop) so subsequent model
    // forward passes reproduce dropout / sampling masks.
    if (!c.rng_state.empty()) {
        auto unpack_state = [](const Tensor& t) -> tenzor::GeneratorState {
            tenzor::GeneratorState s;
            if (t.numel() < 2) return s;
            Tensor host = (t.device().type != Device::Type::CPU) ? t.cpu() : t;
            const auto* p = host.data<int64_t>();
            s.seed         = static_cast<uint64_t>(p[0]);
            s.initial_seed = static_cast<uint64_t>(p[1]);
            const int64_t engine_n = host.numel() - 2;
            s.engine_state.resize(static_cast<size_t>(engine_n));
            for (int64_t i = 0; i < engine_n; ++i) {
                s.engine_state[static_cast<size_t>(i)] =
                    static_cast<uint64_t>(p[2 + i]);
            }
            return s;
        };
        for (const auto& [key, t] : c.rng_state) {
            // Key format: "<device_type>:<index>".
            auto colon = key.rfind(':');
            if (colon == std::string::npos) continue;
            std::string dev_str = key.substr(0, colon);
            int32_t idx = 0;
            try {
                idx = std::stoi(key.substr(colon + 1));
            } catch (const std::exception&) {
                continue;
            }
            Device::Type dt = Device::Type::CPU;
            if      (dev_str == "cpu")    dt = Device::Type::CPU;
            else if (dev_str == "cuda")   dt = Device::Type::CUDA;
            else if (dev_str == "rocm")   dt = Device::Type::ROCm;
            else if (dev_str == "oneapi") dt = Device::Type::OneAPI;
            else if (dev_str == "vulkan") dt = Device::Type::Vulkan;
            else if (dev_str == "mps")    dt = Device::Type::MPS;
            else continue;
            Device dev{dt, idx};
            try {
                auto& gen = tenzor::default_generator(dev);
                gen.set_state(unpack_state(t));
            } catch (const std::exception&) {
                // Device generator unavailable — skip; matches save-side
                // partial-coverage semantics.
            }
        }
    }

    return c;
}

auto ModelCheckpoint::load_and_apply(
    const std::string& path,
    Module* module,
    optim::Optimizer* optimizer,
    optim::LRScheduler* scheduler
) -> Checkpoint {
    Checkpoint c = load(path);
    if (module && !c.model_state.empty()) {
        module->load_state_dict(c.model_state);
    }
    if (optimizer && !c.optimizer_state.empty()) {
        optimizer->load_state_dict(c.optimizer_state);
    }
    // audit Q.11: actually apply the deserialised scheduler state — prior
    // to v3 this map was always empty so load was a no-op.
    if (scheduler && !c.scheduler_state.empty()) {
        scheduler->load_state_dict(c.scheduler_state);
    }
    return c;
}

auto ModelCheckpoint::save_model(
    const std::string& path,
    const Module& module,
    const TrainingMetadata& metadata
) -> void {
    save(path, module, nullptr, nullptr, metadata);
}

auto ModelCheckpoint::load_model(const std::string& path) -> std::unordered_map<std::string, Tensor> {
    // Auto-detect format: SafeTensors files are loaded directly,
    // native checkpoint files go through the full checkpoint path.
    auto format = detect_format(path);
    if (format == SerializeFormat::SafeTensors) {
        return SafeTensorsSerializer::load(path);
    }
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

        // Read config flags
        uint8_t config_flags;
        file.read(reinterpret_cast<char*>(&config_flags), sizeof(config_flags));
        bool verify_checksum = (config_flags & 4) != 0;

        // If checksum verification is enabled, verify file integrity
        if (verify_checksum) {
            // Get file size
            file.seekg(0, std::ios::end);
            size_t file_size = file.tellg();

            // Check if file is large enough to contain checksum
            if (file_size < sizeof(uint64_t)) {
                return false;
            }

            // Read entire file content (excluding the checksum at the end)
            file.seekg(0, std::ios::beg);
            size_t content_size = file_size - sizeof(uint64_t);
            std::vector<uint8_t> file_content(content_size);
            file.read(reinterpret_cast<char*>(file_content.data()), content_size);

            // Read stored checksum
            uint64_t stored_checksum;
            file.read(reinterpret_cast<char*>(&stored_checksum), sizeof(stored_checksum));

            // Compute actual checksum
            uint64_t computed_checksum = compute_checksum(file_content.data(), file_content.size());

            // Verify checksums match
            if (computed_checksum != stored_checksum) {
                return false;  // Checksum mismatch = corrupted file
            }
        }

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

        // Write tensor data - transfer to CPU if on a GPU device
        Tensor host_tensor = (tensor.device().type != Device::Type::CPU) ? tensor.cpu() : tensor;
        size_t data_size = host_tensor.numel() * host_tensor.dtype_size();
        const void* data_ptr = host_tensor.data_ptr();
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

        // Write tensor data - transfer to CPU if on a GPU device
        Tensor host_tensor = (tensor.device().type != Device::Type::CPU) ? tensor.cpu() : tensor;
        size_t data_size = host_tensor.numel() * host_tensor.dtype_size();
        const void* data_ptr = host_tensor.data_ptr();
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

        // Write tensor data - transfer to CPU if on a GPU device
        Tensor host_tensor = (tensor.device().type != Device::Type::CPU) ? tensor.cpu() : tensor;
        size_t data_size = host_tensor.numel() * host_tensor.dtype_size();
        const void* data_ptr = host_tensor.data_ptr();
        file.write(reinterpret_cast<const char*>(data_ptr), data_size);
    }

    // ========== RNG State (v2+, audit K.2) ==========
    // V.23: the read side gates this section on `version >= 2`; the write
    // side must mirror the same gate or a v1 round-trip mis-aligns by the
    // RNG section's byte width (num_rng_tensors + per-tensor headers).
    // This affects downgrade workflows where a caller deliberately sets
    // `checkpoint.version` below CHECKPOINT_VERSION to produce a file an
    // older Tenzor build can ingest.
    if (checkpoint.version >= 2) {
        uint32_t num_rng_tensors = static_cast<uint32_t>(checkpoint.rng_state.size());
        file.write(reinterpret_cast<const char*>(&num_rng_tensors), sizeof(num_rng_tensors));

        for (const auto& [name, tensor] : checkpoint.rng_state) {
            // Write state name (device key like "cpu:0", "cuda:0").
            uint32_t name_len = static_cast<uint32_t>(name.size());
            file.write(reinterpret_cast<const char*>(&name_len), sizeof(name_len));
            file.write(name.data(), name_len);

            // Write tensor shape (always 1-D, but encode generically for future
            // device-specific RNG formats).
            auto shape = tensor.shape();
            uint32_t ndim = static_cast<uint32_t>(shape.size());
            file.write(reinterpret_cast<const char*>(&ndim), sizeof(ndim));
            file.write(reinterpret_cast<const char*>(shape.data()),
                       ndim * sizeof(int64_t));

            // Write tensor dtype (always Int64 for current packing).
            uint8_t dtype = static_cast<uint8_t>(tensor.dtype());
            file.write(reinterpret_cast<const char*>(&dtype), sizeof(dtype));

            // RNG snapshots are always CPU-resident already, but mirror the
            // pattern for forward-compat if a backend's generator state goes
            // through device-side packing in the future.
            Tensor host_tensor = (tensor.device().type != Device::Type::CPU) ? tensor.cpu() : tensor;
            size_t data_size = host_tensor.numel() * host_tensor.dtype_size();
            const void* data_ptr = host_tensor.data_ptr();
            file.write(reinterpret_cast<const char*>(data_ptr), data_size);
        }
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
        // Compute and write checksum of entire file content
        file.flush();

        // Get current position (end of content, before checksum)
        auto content_end = file.tellp();

        // Read all content written so far
        file.close();
        std::ifstream read_file(path, std::ios::binary);
        std::vector<uint8_t> content(content_end);
        read_file.read(reinterpret_cast<char*>(content.data()), content_end);
        read_file.close();

        // Compute checksum on content
        uint64_t checksum = compute_checksum(content.data(), content.size());

        // Reopen file in append mode and write checksum
        std::ofstream append_file(path, std::ios::binary | std::ios::app);
        append_file.write(reinterpret_cast<const char*>(&checksum), sizeof(checksum));
        append_file.close();

        if (!append_file) {
            throw std::runtime_error("Failed to write checkpoint checksum");
        }
    } else {
        file.close();
        if (!file) {
            throw std::runtime_error("Failed to write checkpoint file");
        }
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

    // When the checkpoint advertises a checksum, verify file integrity up front
    // (recomputing CRC64 over the content excluding the 8-byte trailer and
    // comparing against the stored value) before parsing any body bytes.
    // Previously load()/load_and_apply() went through this path and silently
    // accepted corrupted/tampered files despite the flag — only the separate
    // verify_checkpoint() helper actually checked.
    if (checkpoint.config.verify_checksum) {
        if (!verify_checkpoint(path)) {
            throw std::runtime_error(
                "Checkpoint checksum verification failed (corrupted or tampered "
                "file): " + path);
        }
    }

    // ========== Model State ==========
    uint32_t num_model_tensors;
    file.read(reinterpret_cast<char*>(&num_model_tensors), sizeof(num_model_tensors));

    for (uint32_t i = 0; i < num_model_tensors; ++i) {
        std::string name = ckpt_read_string(file);
        checkpoint.model_state[name] = ckpt_read_tensor(file);
    }

    // ========== Optimizer State ==========
    uint32_t num_optimizer_tensors;
    file.read(reinterpret_cast<char*>(&num_optimizer_tensors), sizeof(num_optimizer_tensors));

    for (uint32_t i = 0; i < num_optimizer_tensors; ++i) {
        std::string name = ckpt_read_string(file);
        checkpoint.optimizer_state[name] = ckpt_read_tensor(file);
    }

    // ========== Scheduler State ==========
    uint32_t num_scheduler_tensors;
    file.read(reinterpret_cast<char*>(&num_scheduler_tensors), sizeof(num_scheduler_tensors));

    for (uint32_t i = 0; i < num_scheduler_tensors; ++i) {
        std::string name = ckpt_read_string(file);
        checkpoint.scheduler_state[name] = ckpt_read_tensor(file);
    }

    // ========== RNG State (v2+, audit K.2) ==========
    // v1 files written before K.2 don't have this section; the version
    // check skips the read and leaves checkpoint.rng_state empty so
    // callers can detect "RNG wasn't captured".
    if (checkpoint.version >= 2) {
        uint32_t num_rng_tensors;
        file.read(reinterpret_cast<char*>(&num_rng_tensors), sizeof(num_rng_tensors));

        for (uint32_t i = 0; i < num_rng_tensors; ++i) {
            std::string name = ckpt_read_string(file);
            checkpoint.rng_state[name] = ckpt_read_tensor(file);
        }
    }

    // ========== Metadata ==========
    uint32_t num_metadata;
    file.read(reinterpret_cast<char*>(&num_metadata), sizeof(num_metadata));

    std::unordered_map<std::string, std::string> metadata_dict;
    for (uint32_t i = 0; i < num_metadata; ++i) {
        std::string key = ckpt_read_string(file);
        std::string value = ckpt_read_string(file);
        metadata_dict[key] = value;
    }

    checkpoint.metadata.from_dict(metadata_dict);

    // ========== Footer ==========
    // The checksum (if present) was already verified up front against the full
    // file content via verify_checkpoint(). Consume the trailer here so the
    // final completeness check below sees a clean stream.
    if (checkpoint.config.verify_checksum) {
        uint64_t stored_checksum;
        file.read(reinterpret_cast<char*>(&stored_checksum), sizeof(stored_checksum));
        (void)stored_checksum;
    }

    if (!file) {
        throw std::runtime_error("Failed to read checkpoint file completely");
    }

    return checkpoint;
}

auto ModelCheckpoint::compute_checksum(const void* data, size_t size) -> uint64_t {
    // Proper CRC64-ECMA implementation
    // Polynomial: 0x42F0E1EBA9EA3693
    static constexpr uint64_t POLY = 0x42F0E1EBA9EA3693ULL;

    uint64_t crc = 0xFFFFFFFFFFFFFFFFULL;
    const uint8_t* bytes = static_cast<const uint8_t*>(data);

    for (size_t i = 0; i < size; ++i) {
        crc ^= static_cast<uint64_t>(bytes[i]) << 56;
        for (int j = 0; j < 8; ++j) {
            if (crc & 0x8000000000000000ULL) {
                crc = (crc << 1) ^ POLY;
            } else {
                crc <<= 1;
            }
        }
    }

    return crc ^ 0xFFFFFFFFFFFFFFFFULL;
}

auto ModelCheckpoint::compress_data(
    const void* input,
    size_t input_size,
    std::vector<uint8_t>& output
) -> bool {
    // Implement simple RLE (Run-Length Encoding) compression for tensor data
    // This is a lightweight compression suitable for neural network weights
    // which often contain repeated patterns and zeros

    const uint8_t* input_bytes = static_cast<const uint8_t*>(input);
    output.clear();
    output.reserve(input_size);  // Reserve at least input size

    if (input_size == 0) {
        return true;
    }

    // RLE: Store byte followed by count if repeated, or raw bytes for non-repeated sequences
    size_t i = 0;
    while (i < input_size) {
        uint8_t current = input_bytes[i];
        size_t run_length = 1;

        // Count consecutive identical bytes (max 255 for uint8_t counter)
        while (i + run_length < input_size &&
               input_bytes[i + run_length] == current &&
               run_length < 255) {
            run_length++;
        }

        if (run_length >= 3) {
            // Use RLE encoding for runs of 3 or more
            output.push_back(0xFF);  // RLE marker
            output.push_back(static_cast<uint8_t>(run_length));
            output.push_back(current);
            i += run_length;
        } else {
            // Store raw bytes for short sequences
            // Count non-repeated bytes
            size_t raw_count = 0;
            size_t start = i;
            while (i < input_size && raw_count < 255) {
                // Look ahead to see if we have a long run coming
                size_t lookahead = 1;
                while (i + lookahead < input_size &&
                       input_bytes[i + lookahead] == input_bytes[i] &&
                       lookahead < 3) {
                    lookahead++;
                }

                if (lookahead >= 3) {
                    break;  // Stop before a compressible run
                }

                raw_count++;
                i++;
            }

            // Write raw sequence: count followed by bytes
            output.push_back(static_cast<uint8_t>(raw_count));
            for (size_t j = 0; j < raw_count; ++j) {
                output.push_back(input_bytes[start + j]);
            }
        }
    }

    return true;
}

auto ModelCheckpoint::decompress_data(
    const void* input,
    size_t input_size,
    std::vector<uint8_t>& output,
    size_t expected_size
) -> bool {
    // Decompress RLE-encoded data
    const uint8_t* input_bytes = static_cast<const uint8_t*>(input);
    output.clear();
    output.reserve(expected_size);

    size_t i = 0;
    while (i < input_size) {
        if (input_bytes[i] == 0xFF && i + 2 < input_size) {
            // RLE encoded run
            uint8_t run_length = input_bytes[i + 1];
            uint8_t value = input_bytes[i + 2];

            for (uint8_t j = 0; j < run_length; ++j) {
                output.push_back(value);
            }

            i += 3;
        } else {
            // Raw sequence
            uint8_t raw_count = input_bytes[i];
            i++;

            if (i + raw_count > input_size) {
                return false;  // Corrupted data
            }

            for (uint8_t j = 0; j < raw_count; ++j) {
                output.push_back(input_bytes[i + j]);
            }

            i += raw_count;
        }
    }

    // Verify expected size if provided
    if (expected_size > 0 && output.size() != expected_size) {
        return false;  // Size mismatch
    }

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

    return true;  // Fixed: Return "saved" status instead of "is_best"
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
