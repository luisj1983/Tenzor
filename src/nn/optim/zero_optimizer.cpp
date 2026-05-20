/**
 * @file zero_optimizer.cpp
 * @brief Implementation of ZeRO Stage 1 Optimizer
 */

#include "tenzor/nn/optim/zero_optimizer.hpp"
#include "tenzor/nn/optim/gradient_utils.hpp"
#include "tenzor/nn/optim/adam.hpp"
#include "tenzor/nn/optim/sgd.hpp"
#include "tenzor/nn/serialize.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/transform.hpp"
#include "tenzor/ops/reduction.hpp"
// Phase B (B1): FusedAdamStep dispatch path -- mirrors src/nn/optim/adam.cpp:20-82.
#include "tenzor/backend/fast_dispatch.hpp"
#include "tenzor/backend/op_attributes.hpp"
#include "tenzor/ops/op_id.hpp"
// Phase E (E2): activation-checkpoint integration -- recompute hook registry.
#include "tenzor/autograd/checkpoint.hpp"
#include <filesystem>
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>

// CUDA/HIP runtime for the dedicated comm stream. Same conditional pattern as ddp.cpp:
// when neither CUDA nor ROCm is compiled in, the async path is unavailable and the
// optimizer transparently falls back to fully-synchronous all_reduce.
#if defined(TENZOR_USE_CUDA)
    #include <cuda_runtime.h>
#elif defined(TENZOR_USE_ROCM)
    #include <hip/hip_runtime.h>
    #define cudaStream_t              hipStream_t
    #define cudaStreamCreateWithFlags hipStreamCreateWithFlags
    #define cudaStreamNonBlocking     hipStreamNonBlocking
    #define cudaStreamDestroy         hipStreamDestroy
    #define cudaStreamSynchronize     hipStreamSynchronize
    #define cudaSuccess               hipSuccess
    #define cudaGetErrorString        hipGetErrorString
    using cudaError_t = hipError_t;
#endif

namespace tenzor {
namespace optim {

// =============================================================================
// Helpers
// =============================================================================
namespace {

// Build a 1-element tensor in the same dtype/device as `ref` carrying `value`.
// Used by the in-place optimizer kernels to multiply/add a scalar without forcing a
// fp32 demotion (the previous implementation cast everything through `float`, which
// silently lost precision for Float64 params — see feedback_float32_accum_bug.md).
inline auto scalar_like(double value, const Tensor& ref) -> Tensor {
    return full({1}, value, ref.dtype(), ref.device());
}

// Per-tensor int8 quantization. Returns the int8 payload and a 1-element fp32 scale
// tensor; reconstruct via `q.to(orig_dtype) * scale`. The scalar reduction `max(|t|)`
// runs on whatever device `t` lives on, then we materialise the scale via the
// scalar-tensor builder so dequantization can broadcast it without a copy back to host.
struct QuantizedInt8 {
    Tensor data;   // shape == t.shape(), dtype == Int8
    Tensor scale;  // shape {1}, dtype == Float32, on t.device()
};

inline auto quantize_to_int8(const Tensor& t) -> QuantizedInt8 {
    // scale = max(|t|) / 127. If all-zero (max==0), pin scale to 1 so we don't
    // emit a NaN/Inf-laden int8 payload.
    Tensor abs_t = abs(t);
    Tensor max_t = tenzor::max(abs_t);
    float max_val = max_t.item<float>();
    float scale = (max_val > 0.0f) ? (max_val / 127.0f) : 1.0f;

    // q = clamp(round(t / scale), -128, 127) cast to Int8.
    Tensor scaled = t * (1.0 / scale);
    Tensor rounded = round(scaled);
    Tensor clamped = clamp(rounded, -128.0f, 127.0f);
    QuantizedInt8 out;
    out.data = clamped.to(DType::Int8);
    out.scale = full({1}, scale, DType::Float32, t.device());
    return out;
}

inline auto dequantize_from_int8(const Tensor& q, const Tensor& scale, DType target_dtype) -> Tensor {
    // First widen int8 → target dtype, then multiply by the broadcast scalar scale.
    // The order matters: doing q * scale in int8 would clip to int8 range.
    Tensor widened = q.to(target_dtype);
    Tensor scale_in_target = (scale.dtype() != target_dtype) ? scale.to(target_dtype) : scale;
    return widened * scale_in_target;
}

// ---------------------------------------------------------------------------
// NVMe offload helpers (synchronous std::ofstream / std::ifstream)
// ---------------------------------------------------------------------------
//
// Format: raw little-endian element bytes only. Shape and dtype are tracked in
// the optimizer's StatePartition::DiskSlot rather than in a file header — the
// optimizer is the only writer + reader, so a self-describing format would just
// duplicate state that we already have.

inline auto resolve_nvme_dir(const std::string& configured) -> std::filesystem::path {
    namespace fs = std::filesystem;
    fs::path dir = configured.empty()
        ? (fs::temp_directory_path() / "tenzor_zero_offload")
        : fs::path(configured);
    fs::create_directories(dir);
    return dir;
}

// Stage GPU→host (if needed), then write the raw bytes to `path`. Returns the
// number of bytes written so the caller can assert it later if desired. Throws
// std::runtime_error on IO failure rather than ignoring it — silent disk-full
// during training is way worse than a hard failure at the offload boundary.
inline auto write_tensor_blob(const Tensor& t, const std::filesystem::path& path) -> size_t {
    Tensor host_tensor = (t.device().type == Device::Type::CPU)
        ? t.contiguous()
        : t.to(Device::cpu()).contiguous();

    const size_t bytes = static_cast<size_t>(host_tensor.numel()) * dtype_size(host_tensor.dtype());

    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
        throw std::runtime_error("write_tensor_blob: failed to open '" + path.string() + "'");
    }
    if (bytes > 0) {
        out.write(static_cast<const char*>(host_tensor.data_ptr()), static_cast<std::streamsize>(bytes));
        if (!out) {
            throw std::runtime_error("write_tensor_blob: short write to '" + path.string() + "'");
        }
    }
    return bytes;
}

// Allocate a tensor with the recorded shape/dtype on `device` and read the file
// contents into it. Use a host-side staging tensor when device != CPU.
inline auto read_tensor_blob(
    const std::filesystem::path& path,
    const std::vector<int64_t>& shape,
    DType dtype,
    Device device
) -> Tensor {
    // First materialise into a CPU tensor so we can read into its data_ptr() directly.
    Tensor host_tensor = empty(shape, dtype, Device::cpu());
    const size_t bytes = static_cast<size_t>(host_tensor.numel()) * dtype_size(dtype);

    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("read_tensor_blob: failed to open '" + path.string() + "'");
    }
    if (bytes > 0) {
        in.read(static_cast<char*>(host_tensor.data_ptr()), static_cast<std::streamsize>(bytes));
        if (in.gcount() != static_cast<std::streamsize>(bytes)) {
            throw std::runtime_error("read_tensor_blob: short read from '" + path.string() + "'");
        }
    }
    return (device.type == Device::Type::CPU) ? host_tensor : host_tensor.to(device);
}

// Build a file path for a particular state slot. Deterministic so a restart with
// the same config can pick up where the previous run left off (best-effort —
// we don't currently checksum, so corrupted files would silently break training).
inline auto state_blob_path(
    const std::filesystem::path& dir,
    int rank,
    size_t param_idx,
    std::string_view state_name
) -> std::filesystem::path {
    std::string fname = "zero_r" + std::to_string(rank) +
                        "_p" + std::to_string(param_idx) +
                        "_" + std::string(state_name) + ".bin";
    return dir / fname;
}

}  // namespace

// =============================================================================
// Constructor & Destructor
// =============================================================================

ZeROStage1Optimizer::ZeROStage1Optimizer(
    std::shared_ptr<Optimizer> base_optimizer,
    const ZeROStage1Config& config
) : Optimizer(base_optimizer ? base_optimizer->parameters() : std::vector<std::shared_ptr<Variable>>{}),
    base_optimizer_(std::move(base_optimizer)),
    config_(config) {

    if (!base_optimizer_) {
        throw std::invalid_argument("base_optimizer cannot be null");
    }
    if (config_.rank < 0 || config_.rank >= config_.world_size) {
        throw std::invalid_argument("Invalid rank: must be in [0, world_size)");
    }
    if (config_.world_size <= 0) {
        throw std::invalid_argument("world_size must be > 0");
    }

    if (!config_.process_group && distributed::is_initialized()) {
        config_.process_group = distributed::DistributedContext::get_process_group();
    }

    // Detect whether the process group can run all_reduce on a caller-supplied stream
    // (NCCL/RCCL on GPU). When yes, allocate a dedicated non-blocking stream so the
    // all-reduce in step_impl can overlap with the host-side fetch_states_to_gpu work.
    if (config_.process_group && config_.process_group->supports_async_stream()) {
#if defined(TENZOR_USE_CUDA) || defined(TENZOR_USE_ROCM)
        cudaStream_t s = nullptr;
        cudaError_t err = cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
        if (err == cudaSuccess) {
            comm_stream_ = static_cast<void*>(s);
            use_gpu_comm_ = true;
        }
        // If stream creation fails, fall back to sync path; not fatal.
#endif
    }

    partition_parameters();

    if (config_.offload_to_cpu) {
        initialize_offload_engine();
    }

    initialize_optimizer_states();
}

ZeROStage1Optimizer::~ZeROStage1Optimizer() {
    // Best-effort cleanup of NVMe scratch files. We don't want a destructor exception under
    // any circumstance, so we swallow any IO error — leaving stale blobs behind is at worst
    // a disk-space leak that the user can wipe via the configured nvme_path.
    if (config_.offload_to_nvme) {
        try {
            for (auto& part : partitions_) {
                auto remove_slot = [](StatePartition::DiskSlot& s) {
                    if (s.on_disk()) {
                        std::error_code ec;
                        std::filesystem::remove(s.path, ec);
                        s.path.clear();
                    }
                };
                for (auto& s : part.momentum_disk)       remove_slot(s);
                for (auto& s : part.variance_disk)       remove_slot(s);
                for (auto& s : part.momentum_scale_disk) remove_slot(s);
                for (auto& s : part.variance_scale_disk) remove_slot(s);
            }
        } catch (...) {
            // swallow — destructors must not throw
        }
    }
    // Tear down the dedicated comm stream if we allocated one. cudaStreamDestroy waits for
    // any in-flight ops on the stream, so this is safe even if the user dropped the
    // optimizer mid-step (which they shouldn't but might in error paths).
#if defined(TENZOR_USE_CUDA) || defined(TENZOR_USE_ROCM)
    if (comm_stream_) {
        cudaStreamDestroy(static_cast<cudaStream_t>(comm_stream_));
        comm_stream_ = nullptr;
    }
#endif

    // Other cleanup is automatic via smart pointers.
}

// =============================================================================
// Optimizer Interface
// =============================================================================

auto ZeROStage1Optimizer::step_impl() -> void {
    std::lock_guard<std::mutex> lock(mutex_);

    // Start profiling
    auto step_start = std::chrono::steady_clock::now();
    if (profiling_enabled_) {
        step_start_time_ = step_start;
    }

    // Step 1a: Issue async all-reduce of gradients on the comm stream. When the process
    // group supports stream-based collectives, this returns immediately and lets us run
    // host/PCIe-side optimizer-state fetch concurrently with the all-reduce. The wait is
    // deferred to step 1b just before update_local_partition reads the grads.
    auto comm_start = std::chrono::steady_clock::now();
    if (config_.world_size > 1) {
        issue_async_all_reduce_gradients();
    }

    // Step 2: Fetch optimizer states from CPU if offloaded — this overlaps with the
    // in-flight all-reduce when use_gpu_comm_ is on.
    if ((config_.offload_to_cpu && offload_engine_) || config_.offload_to_nvme) {
        auto offload_start = std::chrono::steady_clock::now();
        fetch_states_to_gpu();
        if (profiling_enabled_) {
            auto offload_end = std::chrono::steady_clock::now();
            auto offload_duration = std::chrono::duration<double, std::milli>(offload_end - offload_start).count();
            std::lock_guard<std::mutex> prof_lock(profiling_mutex_);
            profiling_stats_.offload_time_ms += offload_duration;
        }
    }

    // Step 1b: Block on the comm stream before the update reads gradients. Profiling
    // accounts for the *wait* time only, so the comm time is reported as the slack
    // between step 1a and here — anything that overlapped with fetch_states is excluded
    // from the comm budget, which is the metric users actually care about.
    if (config_.world_size > 1) {
        wait_for_async_all_reduce();
        if (profiling_enabled_) {
            auto comm_end = std::chrono::steady_clock::now();
            auto comm_duration = std::chrono::duration<double, std::milli>(comm_end - comm_start).count();
            std::lock_guard<std::mutex> prof_lock(profiling_mutex_);
            profiling_stats_.all_reduce_time_ms += comm_duration;
            profiling_stats_.communication_time_ms += comm_duration;
            profiling_stats_.num_all_reduces++;
        }
    }

    // Step 3: Update local partition of parameters
    auto compute_start = std::chrono::steady_clock::now();
    if (config_.partitioning_mode == PartitioningMode::ElementLevel) {
        update_local_partition_element_mode();
    } else {
        update_local_partition();
    }
    if (profiling_enabled_) {
        auto compute_end = std::chrono::steady_clock::now();
        auto compute_duration = std::chrono::duration<double, std::milli>(compute_end - compute_start).count();
        std::lock_guard<std::mutex> prof_lock(profiling_mutex_);
        profiling_stats_.compute_time_ms += compute_duration;
    }

    // Step 4: Offload states back to CPU if enabled
    if ((config_.offload_to_cpu && offload_engine_) || config_.offload_to_nvme) {
        auto offload_start = std::chrono::steady_clock::now();
        offload_states_to_cpu();
        if (profiling_enabled_) {
            auto offload_end = std::chrono::steady_clock::now();
            auto offload_duration = std::chrono::duration<double, std::milli>(offload_end - offload_start).count();
            std::lock_guard<std::mutex> prof_lock(profiling_mutex_);
            profiling_stats_.offload_time_ms += offload_duration;
        }
    }

    // Step 5: All-gather updated parameters across ranks
    if (config_.world_size > 1) {
        auto gather_start = std::chrono::steady_clock::now();
        if (config_.partitioning_mode == PartitioningMode::ElementLevel) {
            all_gather_parameters_element_mode();
        } else {
            all_gather_parameters();
        }
        if (profiling_enabled_) {
            auto gather_end = std::chrono::steady_clock::now();
            auto gather_duration = std::chrono::duration<double, std::milli>(gather_end - gather_start).count();
            std::lock_guard<std::mutex> prof_lock(profiling_mutex_);
            profiling_stats_.all_gather_time_ms += gather_duration;
            profiling_stats_.gather_time_ms += gather_duration;
            profiling_stats_.communication_time_ms += gather_duration;
            profiling_stats_.num_all_gathers++;
            profiling_stats_.num_gathers++;
        }
    }

    // Complete profiling
    if (profiling_enabled_) {
        auto step_end = std::chrono::steady_clock::now();
        auto total_duration = std::chrono::duration<double, std::milli>(step_end - step_start).count();

        std::lock_guard<std::mutex> prof_lock(profiling_mutex_);
        profiling_stats_.total_step_time_ms += total_duration;
        profiling_stats_.num_steps++;

        // Update averages
        profiling_stats_.avg_step_time_ms =
            profiling_stats_.total_step_time_ms / profiling_stats_.num_steps;
        if (profiling_stats_.num_all_reduces > 0) {
            profiling_stats_.avg_all_reduce_time_ms =
                profiling_stats_.all_reduce_time_ms / profiling_stats_.num_all_reduces;
        }
        if (profiling_stats_.num_gathers > 0) {
            profiling_stats_.avg_gather_time_ms =
                profiling_stats_.gather_time_ms / profiling_stats_.num_gathers;
        }

        // Calculate communication/compute overlap ratio
        // Overlap = 1 - (comm_time / total_time) when compute and comm can overlap
        if (total_duration > 0) {
            double sequential_time = profiling_stats_.communication_time_ms + profiling_stats_.compute_time_ms;
            if (sequential_time > total_duration) {
                profiling_stats_.comm_compute_overlap_ratio =
                    1.0 - (total_duration / sequential_time);
            }
        }

        // Calculate effective bandwidth (MB/s)
        if (profiling_stats_.communication_time_ms > 0) {
            double seconds = profiling_stats_.communication_time_ms / 1000.0;
            double megabytes = profiling_stats_.transferred_bytes / (1024.0 * 1024.0);
            profiling_stats_.effective_bandwidth_mbps = megabytes / seconds;
        }
    }
}

auto ZeROStage1Optimizer::zero_grad() -> void {
    base_optimizer_->zero_grad();
}

auto ZeROStage1Optimizer::state_dict_unlocked() const -> std::unordered_map<std::string, Tensor> {
    // Return only local partition state (caller must hold mutex_)
    std::unordered_map<std::string, Tensor> state;

    const auto& partition = local_partition();

    // Add partition metadata
    state["rank"] = Tensor({1}, DType::Int32, Device::cpu());
    state["rank"].fill_(config_.rank);

    state["world_size"] = Tensor({1}, DType::Int32, Device::cpu());
    state["world_size"].fill_(config_.world_size);

    // Add optimizer states
    for (size_t i = 0; i < partition.momentum.size(); ++i) {
        std::string key = "momentum_" + std::to_string(i);
        state[key] = partition.momentum[i];
    }

    for (size_t i = 0; i < partition.variance.size(); ++i) {
        std::string key = "variance_" + std::to_string(i);
        state[key] = partition.variance[i];
    }

    return state;
}

auto ZeROStage1Optimizer::state_dict() const -> std::unordered_map<std::string, Tensor> {
    std::lock_guard<std::mutex> lock(mutex_);
    return state_dict_unlocked();
}

auto ZeROStage1Optimizer::load_state_dict_unlocked(
    const std::unordered_map<std::string, Tensor>& state
) -> void {
    // Caller must hold mutex_

    // Verify rank and world_size match
    if (state.count("rank")) {
        int saved_rank = state.at("rank").data<int32_t>()[0];
        if (saved_rank != config_.rank) {
            throw std::runtime_error(
                "Rank mismatch: saved=" + std::to_string(saved_rank) +
                ", current=" + std::to_string(config_.rank)
            );
        }
    }

    if (state.count("world_size")) {
        int saved_world_size = state.at("world_size").data<int32_t>()[0];
        if (saved_world_size != config_.world_size) {
            throw std::runtime_error(
                "World size mismatch: saved=" + std::to_string(saved_world_size) +
                ", current=" + std::to_string(config_.world_size)
            );
        }
    }

    // Load optimizer states
    auto& partition = local_partition();

    for (size_t i = 0; i < partition.momentum.size(); ++i) {
        std::string key = "momentum_" + std::to_string(i);
        if (state.count(key)) {
            partition.momentum[i] = state.at(key).to(partition.device);
        }
    }

    for (size_t i = 0; i < partition.variance.size(); ++i) {
        std::string key = "variance_" + std::to_string(i);
        if (state.count(key)) {
            partition.variance[i] = state.at(key).to(partition.device);
        }
    }
}

auto ZeROStage1Optimizer::load_state_dict(
    const std::unordered_map<std::string, Tensor>& state
) -> void {
    std::lock_guard<std::mutex> lock(mutex_);
    load_state_dict_unlocked(state);
}

auto ZeROStage1Optimizer::save_checkpoint(const std::string& path_prefix) const -> void {
    std::lock_guard<std::mutex> lock(mutex_);

    // Each rank saves its own partition
    std::string rank_path = path_prefix + "_rank_" + std::to_string(config_.rank) + ".pt";

    try {
        // Get state dictionary for this rank's partition (use unlocked version since we hold mutex_)
        auto state = state_dict_unlocked();

        // Use Serializer to save state tensors
        nn::Serializer::save(state, rank_path);

        // Master rank saves metadata file
        if (config_.rank == 0) {
            std::string metadata_path = path_prefix + "_metadata.txt";
            std::ofstream meta_file(metadata_path);
            if (!meta_file) {
                throw std::runtime_error("Failed to open metadata file: " + metadata_path);
            }

            // Write checkpoint metadata
            meta_file << "version=1\n";
            meta_file << "world_size=" << config_.world_size << "\n";
            meta_file << "num_partitions=" << partitions_.size() << "\n";
            meta_file << "offload_to_cpu=" << (config_.offload_to_cpu ? "true" : "false") << "\n";
            meta_file << "total_parameters=" << parameters_.size() << "\n";
            meta_file << "partitioning_mode=" <<
                (config_.partitioning_mode == PartitioningMode::ElementLevel
                    ? "element_level" : "param_level") << "\n";

            // Write partition sizes for verification
            for (int rank = 0; rank < config_.world_size; ++rank) {
                const auto& partition = partitions_[rank];
                meta_file << "partition_" << rank << "_size=" << partition.params.size() << "\n";
                meta_file << "partition_" << rank << "_memory=" << partition.memory_bytes << "\n";
            }

            meta_file.close();

            if (!meta_file) {
                throw std::runtime_error("Failed to write metadata file: " + metadata_path);
            }
        }

        // Barrier to ensure all ranks complete before returning
        if (config_.process_group && config_.world_size > 1) {
            config_.process_group->barrier();
        }

    } catch (const std::exception& e) {
        throw std::runtime_error(
            "Rank " + std::to_string(config_.rank) +
            " failed to save checkpoint: " + std::string(e.what())
        );
    }
}

auto ZeROStage1Optimizer::load_checkpoint(const std::string& path_prefix) -> void {
    std::lock_guard<std::mutex> lock(mutex_);

    std::string rank_path = path_prefix + "_rank_" + std::to_string(config_.rank) + ".pt";

    try {
        // Master rank validates metadata first
        if (config_.rank == 0) {
            std::string metadata_path = path_prefix + "_metadata.txt";
            std::ifstream meta_file(metadata_path);
            if (!meta_file) {
                throw std::runtime_error("Metadata file not found: " + metadata_path);
            }

            // Parse and validate metadata
            std::string line;
            std::unordered_map<std::string, std::string> metadata;
            while (std::getline(meta_file, line)) {
                size_t eq_pos = line.find('=');
                if (eq_pos != std::string::npos) {
                    std::string key = line.substr(0, eq_pos);
                    std::string value = line.substr(eq_pos + 1);
                    metadata[key] = value;
                }
            }
            meta_file.close();

            // Verify world_size matches
            if (metadata.count("world_size")) {
                int saved_world_size = std::stoi(metadata["world_size"]);
                if (saved_world_size != config_.world_size) {
                    throw std::runtime_error(
                        "World size mismatch: checkpoint=" + std::to_string(saved_world_size) +
                        ", current=" + std::to_string(config_.world_size) +
                        ". Cannot load checkpoint with different world size."
                    );
                }
            } else {
                throw std::runtime_error("Metadata missing world_size field");
            }

            // Verify partition counts match
            if (metadata.count("num_partitions")) {
                int saved_partitions = std::stoi(metadata["num_partitions"]);
                if (saved_partitions != static_cast<int>(partitions_.size())) {
                    throw std::runtime_error(
                        "Partition count mismatch: checkpoint=" + std::to_string(saved_partitions) +
                        ", current=" + std::to_string(partitions_.size())
                    );
                }
            }

            // Verify partitioning_mode matches (older checkpoints without the field default to param_level)
            {
                const std::string saved_mode = metadata.count("partitioning_mode")
                    ? metadata["partitioning_mode"]
                    : "param_level";  // older checkpoints had no mode → assumed ParamLevel
                const std::string current_mode =
                    config_.partitioning_mode == PartitioningMode::ElementLevel
                        ? "element_level" : "param_level";
                if (saved_mode != current_mode) {
                    throw std::runtime_error(
                        "Partitioning mode mismatch: checkpoint was saved with '" + saved_mode +
                        "' but optimizer is configured for '" + current_mode + "'. Cross-mode loads "
                        "are not supported (the on-disk layout differs). Recreate the optimizer with "
                        "the matching partitioning_mode, or re-save the checkpoint after switching.");
                }
            }

            // Verify partition sizes match
            std::string partition_key = "partition_" + std::to_string(config_.rank) + "_size";
            if (metadata.count(partition_key)) {
                size_t saved_size = std::stoull(metadata[partition_key]);
                if (saved_size != local_partition().params.size()) {
                    throw std::runtime_error(
                        "Rank " + std::to_string(config_.rank) +
                        " partition size mismatch: checkpoint=" + std::to_string(saved_size) +
                        ", current=" + std::to_string(local_partition().params.size())
                    );
                }
            }
        }

        // Synchronize ranks before loading
        if (config_.process_group && config_.world_size > 1) {
            config_.process_group->barrier();
        }

        // Verify checkpoint file exists
        if (!nn::Serializer::is_valid_file(rank_path)) {
            throw std::runtime_error("Invalid or missing checkpoint file: " + rank_path);
        }

        // Load state dictionary from checkpoint
        auto loaded_state = nn::Serializer::load(rank_path);

        // Verify loaded state contains expected keys
        if (!loaded_state.count("rank") || !loaded_state.count("world_size")) {
            throw std::runtime_error("Checkpoint missing rank or world_size metadata");
        }

        // Verify rank matches
        int saved_rank = loaded_state["rank"].data<int32_t>()[0];
        if (saved_rank != config_.rank) {
            throw std::runtime_error(
                "Rank mismatch in checkpoint: file is for rank " + std::to_string(saved_rank) +
                " but loading on rank " + std::to_string(config_.rank)
            );
        }

        // Load state into optimizer (use unlocked version since we hold mutex_)
        load_state_dict_unlocked(loaded_state);

        // Synchronize ranks after loading
        if (config_.process_group && config_.world_size > 1) {
            config_.process_group->barrier();
        }

    } catch (const std::exception& e) {
        throw std::runtime_error(
            "Rank " + std::to_string(config_.rank) +
            " failed to load checkpoint: " + std::string(e.what())
        );
    }
}

auto ZeROStage1Optimizer::local_param_count() const -> size_t {
    return local_partition().params.size();
}

auto ZeROStage1Optimizer::get_memory_stats() const -> MemoryStats {
    MemoryStats stats;
    
    for (const auto& partition : partitions_) {
        stats.num_parameters += partition.params.size();
        
        if (partition.rank == config_.rank) {
            stats.num_local_parameters = partition.params.size();
            
            if (partition.device.type == Device::Type::CPU) {
                stats.cpu_optimizer_memory = partition.memory_bytes;
            } else {
                stats.gpu_optimizer_memory = partition.memory_bytes;
            }
        }
    }
    
    // Calculate gradient memory
    for (const auto& param : parameters_) {
        if (param->has_grad()) {
            const auto& grad_opt = param->grad();
            if (grad_opt.has_value()) {
                const auto& grad = grad_opt.value();
                stats.gpu_gradient_memory += grad.numel() * dtype_size(grad.dtype());
            }
        }
    }
    
    return stats;
}

// =============================================================================
// Private: Initialization
// =============================================================================

auto ZeROStage1Optimizer::partition_parameters() -> void {
    // ElementLevel: compute the element-level partition layout, then initialise one
    // StatePartition per rank with its rank/device. params is left empty in this mode
    // — element slices are tracked via partition_layout_, not via per-rank Variable
    // lists. The ParamLevel code below is skipped.
    if (config_.partitioning_mode == PartitioningMode::ElementLevel) {
        compute_element_partition_layout();

        partitions_.resize(config_.world_size);
        Device dev = !parameters_.empty() ? parameters_[0]->tensor().device() : Device::cpu();
        for (int r = 0; r < config_.world_size; ++r) {
            partitions_[r].rank = r;
            partitions_[r].device = dev;
            // params is intentionally empty in ElementLevel mode — element slices are
            // tracked via partition_layout_, not via per-rank Variable lists. The
            // existing `params` field is reused only by the ParamLevel path.
        }
        return;
    }

    const auto& params = parameters_;
    size_t total_params = params.size();

    // Determine device from first parameter (all params should be on same device)
    Device param_device = !params.empty() ? params[0]->tensor().device() : Device::cpu();

    // Create partitions for all ranks
    partitions_.resize(config_.world_size);
    for (int rank = 0; rank < config_.world_size; ++rank) {
        partitions_[rank].rank = rank;
        // Always use parameter device for computation
        // States will be offloaded to CPU after initialization if offload_to_cpu is enabled
        partitions_[rank].device = param_device;
    }

    if (config_.balanced_partitioning) {
        // Size-aware greedy LPT: sort params by size descending, assign each to the
        // currently least-loaded rank. Provably within 4/3 of optimal makespan; in
        // practice the imbalance for typical transformer-shaped param distributions
        // collapses from "rank-with-embedding holds everything" to ≤2× across ranks.
        struct Item {
            size_t param_idx;
            size_t bytes;
        };
        std::vector<Item> items;
        items.reserve(total_params);
        for (size_t i = 0; i < total_params; ++i) {
            const auto& t = params[i]->tensor();
            items.push_back({i, static_cast<size_t>(t.numel()) * dtype_size(t.dtype())});
        }
        // Stable sort: when two params tie on size, the one with the lower index
        // wins, so partition assignment is deterministic and reproducible across
        // runs (matters for checkpoint compatibility within a single flag value).
        std::stable_sort(items.begin(), items.end(),
                         [](const Item& a, const Item& b) { return a.bytes > b.bytes; });

        for (const auto& it : items) {
            // Pick the rank with smallest current memory_bytes; tie-break on rank
            // id so the schedule is fully deterministic.
            int best_rank = 0;
            size_t best_load = partitions_[0].memory_bytes;
            for (int r = 1; r < config_.world_size; ++r) {
                if (partitions_[r].memory_bytes < best_load) {
                    best_load = partitions_[r].memory_bytes;
                    best_rank = r;
                }
            }
            auto& part = partitions_[best_rank];
            part.params.push_back(params[it.param_idx]);
            part.memory_bytes += it.bytes;
        }
        return;
    }

    // Legacy: contiguous index-slice. Preserved for checkpoint compatibility and
    // for the existing distributed tests that hard-code the assignment.
    size_t params_per_rank = (total_params + config_.world_size - 1) / config_.world_size;
    for (int rank = 0; rank < config_.world_size; ++rank) {
        auto& partition = partitions_[rank];
        size_t start_idx = rank * params_per_rank;
        size_t end_idx = std::min(start_idx + params_per_rank, total_params);

        for (size_t i = start_idx; i < end_idx; ++i) {
            partition.params.push_back(params[i]);
            const auto& tensor = params[i]->tensor();
            partition.memory_bytes += tensor.numel() * dtype_size(tensor.dtype());
        }
    }
}

auto ZeROStage1Optimizer::compute_element_partition_layout() -> void {
    partition_layout_ = PartitionLayout{};   // reset every field to defaults
    PartitionLayout& L = partition_layout_;

    // Walk every parameter, recording (global offset, numel, shape, dtype).
    int64_t total = 0;
    L.params.reserve(parameters_.size());
    for (const auto& p : parameters_) {
        const Tensor& t = p->tensor();
        PartitionLayout::ParamEntry entry;
        entry.global_offset = total;
        entry.numel = t.numel();
        auto sh = t.shape();
        entry.original_shape.assign(sh.begin(), sh.end());
        entry.dtype = t.dtype();
        total += entry.numel;
        L.params.push_back(std::move(entry));
    }

    // Round total up to a multiple of world_size so reduce_scatter / all_gather can
    // hand each rank an equal-sized slice. The padding bytes are zeros and contribute
    // nothing to the optimizer math (no parameter touches them).
    const int64_t W = static_cast<int64_t>(config_.world_size);
    int64_t padded = ((total + W - 1) / W) * W;
    L.total_elements_padded = padded;

    // Equal split. rank_starts has world_size + 1 entries so rank_size(R) is just a
    // subtraction.
    L.rank_starts.assign(config_.world_size + 1, 0);
    int64_t per_rank = padded / W;
    for (int r = 0; r <= config_.world_size; ++r) {
        L.rank_starts[r] = std::min<int64_t>(per_rank * r, padded);
    }
}

auto ZeROStage1Optimizer::initialize_optimizer_states() -> void {
    if (config_.partitioning_mode == PartitioningMode::ElementLevel) {
        auto& partition = local_partition();
        const int64_t slice_n = partition_layout_.rank_size(config_.rank);

        Device dev = partition.device;
        // For element-mode, the rank's optimizer state is a single flat slice covering
        // every parameter's contribution. This is exactly the global flat buffer slice
        // [rank_starts[rank], rank_starts[rank+1]). Use the dtype of the first param as
        // the state dtype unless overridden — same rule as ParamLevel.
        DType state_dtype = config_.state_dtype.value_or(
            !parameters_.empty() ? parameters_[0]->tensor().dtype() : DType::Float32);

        partition.momentum.assign(1, zeros({slice_n}, state_dtype, dev));
        partition.variance.assign(1, zeros({slice_n}, state_dtype, dev));

        if (config_.use_master_fp32) {
            // The master is a fp32 copy of the rank's parameter slice. We populate it
            // by gathering each param's portion of the rank's slice; for now zero-init
            // it (the all-gather of params at the end of the FIRST step copies real
            // values into the master via the same code path used for legacy Stage 1).
            //
            // NOTE: master is meaningful only when training in fp16/bf16; if all params
            // are already fp32 the master is wasteful and we skip allocation.
            bool any_low_precision = false;
            for (const auto& e : partition_layout_.params) {
                if (e.dtype != DType::Float32) { any_low_precision = true; break; }
            }
            if (any_low_precision) {
                partition.master_params.assign(1, zeros({slice_n}, DType::Float32, dev));
            } else {
                partition.master_params.assign(1, Tensor());  // no master needed
            }
        }

        partition.memory_bytes =
            slice_n * dtype_size(state_dtype) * 2 +  // momentum + variance
            ((partition.master_params.empty() || partition.master_params[0].numel() == 0)
                ? 0
                : partition.master_params[0].numel() * dtype_size(DType::Float32));

        // If we're going to int8-quantize fp32 optimizer states (on CPU or NVMe),
        // the offload path (offload_states_to_cpu / fetch_states_to_gpu) writes/reads
        // per-tensor scale tensors out of partition.momentum_scales / variance_scales.
        // In ElementLevel mode each of momentum and variance is a single flat slice,
        // so we allocate exactly one scale slot per state, default-constructed. They
        // are populated lazily on the first quantize-and-offload pass — same pattern
        // as ParamLevel (see line ~782).
        const bool any_offload = config_.offload_to_cpu || config_.offload_to_nvme;
        if (any_offload && config_.quantize_offloaded_states_int8) {
            partition.momentum_scales.assign(1, Tensor());
            partition.variance_scales.assign(1, Tensor());
        }
        // Audit G2: NVMe disk-slot metadata mirrors momentum/variance — one slot
        // each in ElementLevel mode. Sized here so subsequent offload_states_to_cpu()
        // / fetch_states_to_gpu() cycles can write/read without a re-alloc.
        if (config_.offload_to_nvme) {
            partition.momentum_disk.assign(1, StatePartition::DiskSlot{});
            partition.variance_disk.assign(1, StatePartition::DiskSlot{});
            if (config_.quantize_offloaded_states_int8) {
                partition.momentum_scale_disk.assign(1, StatePartition::DiskSlot{});
                partition.variance_scale_disk.assign(1, StatePartition::DiskSlot{});
            }
        }

        // CPU offload: pull each freshly-zeroed slice to CPU now to keep peak GPU memory
        // minimal during initialization. Subsequent offload cycles (step-time) go through
        // offload_states_to_cpu() which handles int8 quantization when configured — the
        // scale slots above were just allocated for that path.
        const bool should_cpu_offload = config_.offload_to_cpu && offload_engine_;
        if (should_cpu_offload) {
            partition.momentum[0] = offload_engine_->offload_to_cpu(partition.momentum[0]);
            partition.variance[0] = offload_engine_->offload_to_cpu(partition.variance[0]);
            if (!partition.master_params.empty() && partition.master_params[0].numel() > 0) {
                partition.master_params[0] =
                    offload_engine_->offload_to_cpu(partition.master_params[0]);
            }
            states_on_cpu_ = true;
        }

        // Audit G2: NVMe path for ElementLevel. Previously threw with a
        // "planned: Task 9.1" message — now writes the freshly-zeroed flat
        // slice to disk via the same `write_tensor_blob` helpers that ParamLevel
        // uses, clears the in-memory tensor, and records the slot metadata
        // (path + shape + dtype). The existing `offload_states_to_cpu` and
        // `fetch_states_to_gpu` already iterate `partition.momentum.size()` and
        // handle the NVMe branch generically, so no further changes are needed
        // for step-time. Master copies (when fp32 master is used) stay in
        // memory — same convention as ParamLevel: master is high-precision on
        // the hot path and serialising it defeats the purpose.
        if (config_.offload_to_nvme) {
            std::filesystem::path nvme_dir = resolve_nvme_dir(config_.nvme_path);

            auto shape = std::vector<int64_t>{slice_n};

            partition.momentum_disk[0] = StatePartition::DiskSlot{
                state_blob_path(nvme_dir, partition.rank, /*slot_idx=*/0, "momentum").string(),
                shape, state_dtype
            };
            write_tensor_blob(partition.momentum[0], partition.momentum_disk[0].path);
            partition.momentum[0] = Tensor();  // free the GPU/CPU buffer

            partition.variance_disk[0] = StatePartition::DiskSlot{
                state_blob_path(nvme_dir, partition.rank, /*slot_idx=*/0, "variance").string(),
                shape, state_dtype
            };
            write_tensor_blob(partition.variance[0], partition.variance_disk[0].path);
            partition.variance[0] = Tensor();

            states_on_cpu_ = true;  // "not on GPU device, needs fetch before step()"
        }
        return;
    }

    // Initialize states for local partition only
    auto& partition = local_partition();

    // Detect optimizer type and create appropriate states
    // For Adam/AdamW: need momentum and variance
    // For SGD with momentum: need momentum only

    partition.momentum.reserve(partition.params.size());
    partition.variance.reserve(partition.params.size());
    if (config_.use_master_fp32) {
        partition.master_params.reserve(partition.params.size());
    }
    const bool any_offload = config_.offload_to_cpu || config_.offload_to_nvme;
    if (any_offload && config_.quantize_offloaded_states_int8) {
        // One scale slot per state tensor, all initially empty (= "not currently quantized").
        // Populated when the state goes to CPU; cleared on fetch back to GPU.
        partition.momentum_scales.assign(partition.params.size(), Tensor());
        partition.variance_scales.assign(partition.params.size(), Tensor());
    }
    if (config_.offload_to_nvme) {
        partition.momentum_disk.resize(partition.params.size());
        partition.variance_disk.resize(partition.params.size());
        if (config_.quantize_offloaded_states_int8) {
            partition.momentum_scale_disk.resize(partition.params.size());
            partition.variance_scale_disk.resize(partition.params.size());
        }
    }

    // If CPU offload is enabled, create states on GPU and immediately offload each one
    // This minimizes peak GPU memory usage during initialization
    const bool should_cpu_offload = config_.offload_to_cpu && offload_engine_;
    const bool should_nvme_offload = config_.offload_to_nvme;
    std::filesystem::path nvme_dir;
    if (should_nvme_offload) {
        nvme_dir = resolve_nvme_dir(config_.nvme_path);
    }

    for (size_t i = 0; i < partition.params.size(); ++i) {
        const auto& param = partition.params[i];
        const Tensor& p = param->tensor();
        // Resolve effective state dtype: caller can override via config_.state_dtype, else
        // match the parameter dtype (legacy behaviour).
        DType state_dtype = config_.state_dtype.value_or(p.dtype());

        // Momentum and variance live in `state_dtype` so that fp16/bf16 training can keep
        // optimizer-state precision in fp32 without doubling the parameter storage. zeros() is
        // used (instead of zeros_like + to(state_dtype)) to avoid an extra allocation pass.
        std::vector<int64_t> shape(p.shape().begin(), p.shape().end());
        Tensor momentum = zeros(shape, state_dtype, partition.device);
        Tensor variance = zeros(shape, state_dtype, partition.device);

        partition.memory_bytes += momentum.numel() * dtype_size(momentum.dtype());
        partition.memory_bytes += variance.numel() * dtype_size(variance.dtype());

        // Optional fp32 master copy. Only allocate if the param itself isn't already fp32 —
        // otherwise the master would just be an alias and we'd double the memory for nothing.
        Tensor master;
        if (config_.use_master_fp32 && p.dtype() != DType::Float32) {
            master = p.to(DType::Float32).to(partition.device);
            partition.memory_bytes += master.numel() * dtype_size(master.dtype());
        }

        if (should_cpu_offload) {
            // Immediately offload to CPU if enabled (minimizes peak GPU memory)
            momentum = offload_engine_->offload_to_cpu(momentum);
            variance = offload_engine_->offload_to_cpu(variance);
            if (master.numel() > 0) {
                master = offload_engine_->offload_to_cpu(master);
            }
        } else if (should_nvme_offload) {
            // NVMe path: write each freshly-zeroed state to disk and clear the in-memory
            // tensor. Master copies stay in memory — the whole point of master is high
            // precision on the hot path; serialising it would defeat the purpose.
            partition.momentum_disk[i] = StatePartition::DiskSlot{
                state_blob_path(nvme_dir, partition.rank, i, "momentum").string(),
                shape, state_dtype
            };
            write_tensor_blob(momentum, partition.momentum_disk[i].path);
            momentum = Tensor();  // free the GPU/CPU buffer

            partition.variance_disk[i] = StatePartition::DiskSlot{
                state_blob_path(nvme_dir, partition.rank, i, "variance").string(),
                shape, state_dtype
            };
            write_tensor_blob(variance, partition.variance_disk[i].path);
            variance = Tensor();
        }

        partition.momentum.push_back(momentum);
        partition.variance.push_back(variance);
        if (config_.use_master_fp32) {
            // Always push (possibly empty Tensor) so the index aligns with `params`. An empty
            // master signals "param is already fp32, use it directly".
            partition.master_params.push_back(master);
        }
    }

    if (should_cpu_offload || should_nvme_offload) {
        states_on_cpu_ = true;  // means "not on GPU device, needs fetch before step()"
    }
}

auto ZeROStage1Optimizer::initialize_offload_engine() -> void {
    if (!config_.offload_to_cpu) {
        return;
    }

    // Prefer a caller-supplied engine so the pinned-host buffer pool is shared with
    // any other subsystem (typically activation offload) that already has one. The
    // legacy path constructed a private 1 GB pinned pool unconditionally — that's
    // still the fallback when the caller doesn't provide one.
    if (config_.shared_offload_engine) {
        offload_engine_ = config_.shared_offload_engine;
        return;
    }

    core::OffloadEngine::Config offload_config;
    offload_config.pinned_memory_size = 1024ULL * 1024 * 1024;  // 1GB default
    offload_config.num_transfer_streams = 4;
    offload_config.enable_prefetch = true;

    offload_engine_ = std::make_shared<core::OffloadEngine>(offload_config);
}

// =============================================================================
// Private: Communication
// =============================================================================

auto ZeROStage1Optimizer::all_reduce_gradients() -> void {
    // Synchronous wrapper that preserves the legacy contract: kick off async, wait
    // immediately. Callers that want to overlap the all-reduce with other work (notably
    // step_impl wrapping fetch_states_to_gpu in between) should call the issue/wait pair
    // directly rather than this wrapper.
    issue_async_all_reduce_gradients();
    wait_for_async_all_reduce();
}

auto ZeROStage1Optimizer::issue_async_all_reduce_gradients() -> void {
    if (!config_.process_group) {
        throw std::runtime_error("Process group not initialized");
    }
    if (async_all_reduce_in_flight_) {
        throw std::runtime_error("issue_async_all_reduce_gradients called with another async all-reduce already in flight");
    }

    in_flight_compressed_.clear();
    if (config_.grad_compressor) {
        // Reserve once so the vector address is stable across the per-param compress
        // calls (we hand off the .data tensor by value, not by reference, so addresses
        // matter only for our own bookkeeping below).
        in_flight_compressed_.reserve(parameters_.size());
    }

    // All-reduce gradients for all parameters. When use_gpu_comm_ is set we route the
    // collective through the dedicated comm_stream_ so it can overlap with the host /
    // PCIe-side fetch_states_to_gpu work that step_impl runs next.
    for (auto& param : parameters_) {
        if (!param->has_grad()) continue;
        const auto& grad_opt = param->grad();
        if (!grad_opt.has_value()) continue;

        Tensor grad = grad_opt.value();

        if (profiling_enabled_) {
            size_t bytes = grad.numel() * dtype_size(grad.dtype());
            std::lock_guard<std::mutex> prof_lock(profiling_mutex_);
            profiling_stats_.transferred_bytes += bytes * config_.world_size;
        }

        if (config_.grad_compressor) {
            auto compressed = config_.grad_compressor->compress(grad);
            if (use_gpu_comm_) {
                config_.process_group->all_reduce_async(compressed.data,
                                                        distributed::ReduceOp::AVG,
                                                        comm_stream_);
            } else {
                config_.process_group->all_reduce(compressed.data,
                                                  distributed::ReduceOp::AVG);
            }
            in_flight_compressed_.push_back(std::move(compressed));
        } else {
            // Linear all-reduce: AVG is native on NCCL ≥ 2.10, folded for other backends.
            if (use_gpu_comm_) {
                config_.process_group->all_reduce_async(grad,
                                                        distributed::ReduceOp::AVG,
                                                        comm_stream_);
            } else {
                config_.process_group->all_reduce(grad, distributed::ReduceOp::AVG);
            }
            // Without compression the grad tensor is mutated in-place by the collective —
            // write it back to the Variable now so the wait phase doesn't have to remember
            // which params took which path.
            param->set_grad(grad);
        }
    }

    async_all_reduce_in_flight_ = true;
}

auto ZeROStage1Optimizer::wait_for_async_all_reduce() -> void {
    if (!async_all_reduce_in_flight_) {
        return;
    }

    // Block until every async all-reduce we issued has completed on the comm stream. For
    // the sync fallback path (no GPU comm), this is a no-op — the all_reduce calls in
    // issue_* already finished synchronously.
#if defined(TENZOR_USE_CUDA) || defined(TENZOR_USE_ROCM)
    if (use_gpu_comm_ && comm_stream_) {
        cudaStreamSynchronize(static_cast<cudaStream_t>(comm_stream_));
    }
#endif

    // Decompress the held-over compressed payloads, if any, and write the resulting
    // full-precision grads back to their Variables. Order matches the order in which they
    // were pushed in issue_*; the parameters_ traversal here mirrors that.
    if (!in_flight_compressed_.empty() && config_.grad_compressor) {
        size_t comp_idx = 0;
        for (auto& param : parameters_) {
            if (!param->has_grad()) continue;
            if (comp_idx >= in_flight_compressed_.size()) break;
            Tensor grad = config_.grad_compressor->decompress(in_flight_compressed_[comp_idx]);
            param->set_grad(grad);
            ++comp_idx;
        }
        in_flight_compressed_.clear();
    }

    async_all_reduce_in_flight_ = false;
}

auto ZeROStage1Optimizer::all_gather_parameters() -> void {
    if (!config_.process_group) {
        throw std::runtime_error("Process group not initialized");
    }

    // Coalesce per-parameter broadcasts into a single broadcast per source rank. The legacy
    // pattern issued one collective per parameter — for a 1000-parameter model that's 1000
    // collective calls per step, dominated by per-call latency. With this change we issue
    // exactly `world_size` broadcasts regardless of parameter count, so the latency floor
    // is fixed instead of scaling with model size.
    //
    // Per-rank work:
    //   - Owner: flatten local params into a contiguous fp/int buffer and broadcast it.
    //   - Non-owner: allocate a same-shaped destination, receive the broadcast, then
    //     unflatten back into per-param tensors (rebinds them to slices of the new buffer).
    for (int rank = 0; rank < config_.world_size; ++rank) {
        auto& partition = partitions_[rank];
        if (partition.params.empty()) continue;

        // Snapshot each param's current tensor; for non-owner we'll rebind these in-place
        // after the broadcast.
        std::vector<Tensor> param_tensors;
        param_tensors.reserve(partition.params.size());
        size_t total_elements = 0;
        for (auto& p : partition.params) {
            param_tensors.push_back(p->tensor());
            total_elements += static_cast<size_t>(p->tensor().numel());
        }

        if (param_tensors.empty()) continue;

        const DType dt = param_tensors.front().dtype();
        const Device dev = param_tensors.front().device();

        Tensor flat;
        if (rank == config_.rank) {
            // Owner: contiguous flatten of local params. flatten_tensors returns a fresh
            // contiguous buffer containing each param's data back-to-back.
            flat = flatten_tensors(param_tensors);
        } else {
            // Phase B (B2): Non-owner — reuse a persistent receive buffer per source rank
            // instead of allocating fresh every step. param_buffers_ is declared on the
            // optimizer (hpp:597) and was never wired up; this is the same lazy-realloc
            // pattern as GradientBucket::flat_buffer (zero_optimizer.cpp:2430-2436).
            //
            // Saves world_size × partition_bytes of allocator churn per step; for an
            // 8-rank 1B-param fp16 setup that's ~2 GB freed+reallocated per step.
            if (param_buffers_.size() < static_cast<size_t>(config_.world_size)) {
                param_buffers_.resize(config_.world_size);
            }
            Tensor& slot = param_buffers_[rank];
            const int64_t needed = static_cast<int64_t>(total_elements);
            const bool need_realloc = slot.numel() != needed
                                   || slot.dtype() != dt
                                   || slot.device() != dev;
            if (need_realloc) {
                slot = empty({needed}, dt, dev);
            }
            flat = slot;
        }

        // Profiling accounting matches the legacy contract (bytes-per-broadcast, summed
        // across calls). Same total bytes as the per-param loop, just fewer calls.
        if (profiling_enabled_) {
            const size_t bytes = static_cast<size_t>(flat.numel()) * dtype_size(dt);
            std::lock_guard<std::mutex> prof_lock(profiling_mutex_);
            profiling_stats_.transferred_bytes += bytes;
        }

        // ONE broadcast for the whole partition.
        config_.process_group->broadcast(flat, rank);

        if (rank != config_.rank) {
            // Unflatten back: each entry in param_tensors gets rebound to a slice+reshape
            // view of `flat`. Then write each new tensor through to the Variable.
            unflatten_into(flat, param_tensors);
            for (size_t i = 0; i < partition.params.size(); ++i) {
                partition.params[i]->tensor() = param_tensors[i];
            }
        }
    }
}

auto ZeROStage1Optimizer::all_gather_parameters_element_mode() -> void {
    if (config_.world_size <= 1) return;  // nothing to gather
    if (!config_.process_group) {
        throw std::runtime_error("Process group not initialized");
    }

    const auto& L = partition_layout_;
    const int64_t per_rank = L.rank_starts[1] - L.rank_starts[0];
    const int64_t total = L.total_elements_padded;
    Device dev = local_partition().device;
    DType dt = !parameters_.empty() ? parameters_[0]->tensor().dtype() : DType::Float32;

    // Build the local flat slice from the just-updated parameter tensors.
    // Phase B (B3): reuse element_local_flat_buf_ across steps.
    if (element_local_flat_buf_.numel() != per_rank
        || element_local_flat_buf_.dtype() != dt
        || element_local_flat_buf_.device() != dev) {
        element_local_flat_buf_ = zeros({per_rank}, dt, dev);
    } else {
        element_local_flat_buf_.zero_();
    }
    Tensor local_flat = element_local_flat_buf_;
    const int64_t rs = L.rank_starts[config_.rank];
    const int64_t re = L.rank_starts[config_.rank + 1];
    for (size_t i = 0; i < L.params.size(); ++i) {
        const auto& e = L.params[i];
        int64_t p_start = e.global_offset;
        int64_t p_end = e.global_offset + e.numel;
        int64_t lap_start = std::max(p_start, rs);
        int64_t lap_end = std::min(p_end, re);
        if (lap_end <= lap_start) continue;
        Tensor pflat = parameters_[i]->tensor().contiguous().view({-1});
        Tensor src = pflat.slice(0, lap_start - p_start, lap_end - p_start);
        if (src.dtype() != dt) src = src.to(dt);
        Tensor dst = local_flat.slice(0, lap_start - rs, lap_end - rs);
        add_(dst, src);  // dst is zero
    }

    // Single all_gather: world_size × per_rank → global flat buffer.
    // distributed::ProcessGroup::all_gather(input, output): parts is sized world_size,
    // empty slots are populated by the collective with one tensor per rank.
    std::vector<Tensor> parts(config_.world_size);
    config_.process_group->all_gather(local_flat, parts);

    if (profiling_enabled_) {
        // Each rank sends per_rank elements and receives world_size * per_rank elements.
        // Account it the same way the legacy path does: total bytes-on-the-wire summed.
        const size_t bytes = static_cast<size_t>(per_rank) * static_cast<size_t>(config_.world_size)
                           * dtype_size(dt);
        std::lock_guard<std::mutex> prof_lock(profiling_mutex_);
        profiling_stats_.transferred_bytes += bytes;
    }

    // Distribute back: for each parameter, read its global range from the gathered parts.
    for (size_t i = 0; i < L.params.size(); ++i) {
        const auto& e = L.params[i];
        if (!parameters_[i]->tensor().is_contiguous()) {
            throw std::runtime_error(
                "ElementLevel all-gather requires contiguous parameter tensors; "
                "non-contiguous param at index " + std::to_string(i));
        }
        Tensor pflat = parameters_[i]->tensor().view({-1});
        for (int r = 0; r < config_.world_size; ++r) {
            int64_t r_start = L.rank_starts[r];
            int64_t r_end = L.rank_starts[r + 1];
            int64_t lap_start = std::max(e.global_offset, r_start);
            int64_t lap_end = std::min(e.global_offset + e.numel, r_end);
            if (lap_end <= lap_start) continue;
            Tensor src = parts[r].slice(0, lap_start - r_start, lap_end - r_start);
            if (src.dtype() != pflat.dtype()) src = src.to(pflat.dtype());
            Tensor dst = pflat.slice(0, lap_start - e.global_offset, lap_end - e.global_offset);
            dst.zero_();
            add_(dst, src);
        }
    }
    (void)total;  // padding bytes are inert
}

// =============================================================================
// Private: State Management
// =============================================================================

auto ZeROStage1Optimizer::update_local_partition() -> void {
    auto& partition = local_partition();

    // Detect base optimizer type and apply appropriate update algorithm
    // We need to manually apply the optimizer's update logic to only our partition

    // Try to cast to known optimizer types
    auto* adam_opt = dynamic_cast<Adam*>(base_optimizer_.get());
    auto* adamw_opt = dynamic_cast<AdamW*>(base_optimizer_.get());
    auto* sgd_opt = dynamic_cast<SGD*>(base_optimizer_.get());

    if (adam_opt) {
        // Apply Adam update to local partition
        update_partition_adam(partition, adam_opt->get_lr(), 0.9, 0.999, 1e-8, 0.0);
    } else if (adamw_opt) {
        // Apply AdamW update to local partition
        update_partition_adamw(partition, adamw_opt->get_lr(), 0.9, 0.999, 1e-8, 0.01);
    } else if (sgd_opt) {
        // Apply SGD update to local partition
        update_partition_sgd(partition, sgd_opt->get_lr(), 0.9, 0.0);
    } else {
        // Fallback: Try to use base optimizer's step() directly on local partition
        // This may not be optimal but maintains compatibility with unknown optimizer types

        // Store original parameters
        auto original_params = parameters_;

        // Temporarily set only local partition parameters
        parameters_ = partition.params;

        // Call base optimizer step
        try {
            base_optimizer_->step();
        } catch (const std::exception& e) {
            // Restore original parameters and rethrow
            parameters_ = original_params;
            throw std::runtime_error(
                std::string("Failed to update local partition with base optimizer: ") + e.what()
            );
        }

        // Restore original parameters
        parameters_ = original_params;
    }
}

auto ZeROStage1Optimizer::update_local_partition_element_mode() -> void {
    auto& partition = local_partition();
    const auto& L = partition_layout_;
    const int64_t rs = L.rank_starts[config_.rank];
    const int64_t re = L.rank_starts[config_.rank + 1];
    const int64_t slice_n = re - rs;
    if (slice_n == 0) return;

    Device dev = partition.device;
    DType state_dtype = partition.momentum[0].dtype();

    // ---- 1) Build a flat grad slice for this rank. ----
    // Delegates to build_rank_grad_slice() which Stage 2 overrides to read from
    // element_buckets_ (post-reduce_scatter slices) instead of param->grad().
    Tensor grad_slice = build_rank_grad_slice();

    // ---- 2) Run Adam/AdamW/SGD math on the rank's slice. ----
    auto* adam_opt  = dynamic_cast<Adam*>(base_optimizer_.get());
    auto* adamw_opt = dynamic_cast<AdamW*>(base_optimizer_.get());
    auto* sgd_opt   = dynamic_cast<SGD*>(base_optimizer_.get());

    Tensor& m = partition.momentum[0];
    Tensor& v = partition.variance[0];
    bool has_master = config_.use_master_fp32
                   && !partition.master_params.empty()
                   && partition.master_params[0].numel() > 0;

    // For the no-master path we need to gather the current parameter slice from each
    // param->tensor() into a temporary `param_slice`, do the math in place there, then
    // scatter back. The master path keeps a persistent fp32 slice and skips the gather.
    Tensor param_slice;
    if (!has_master) {
        // Phase B (B3): reuse element_param_slice_buf_ across steps.
        if (element_param_slice_buf_.numel() != slice_n
            || element_param_slice_buf_.dtype() != state_dtype
            || element_param_slice_buf_.device() != dev) {
            element_param_slice_buf_ = zeros({slice_n}, state_dtype, dev);
        } else {
            element_param_slice_buf_.zero_();
        }
        param_slice = element_param_slice_buf_;
        for (size_t i = 0; i < L.params.size(); ++i) {
            const auto& e = L.params[i];
            int64_t p_start = e.global_offset;
            int64_t p_end = e.global_offset + e.numel;
            int64_t lap_start = std::max(p_start, rs);
            int64_t lap_end = std::min(p_end, re);
            if (lap_end <= lap_start) continue;
            Tensor pflat = parameters_[i]->tensor().contiguous().view({-1});
            Tensor src = pflat.slice(0, lap_start - p_start, lap_end - p_start);
            Tensor dst = param_slice.slice(0, lap_start - rs, lap_end - rs);
            if (src.dtype() != state_dtype) src = src.to(state_dtype);
            add_(dst, src);
        }
    }

    Tensor& target = has_master ? partition.master_params[0] : param_slice;

    if (adam_opt || adamw_opt) {
        const double lr     = adam_opt ? adam_opt->get_lr()  : adamw_opt->get_lr();
        const double beta1  = 0.9;
        const double beta2  = 0.999;
        const double eps    = 1e-8;
        const double wd     = adamw_opt ? 0.01 : 0.0;
        step_count_++;

        // m = beta1*m + (1-beta1)*g
        mul_(m, scalar_like(beta1, m));
        Tensor scaled_g = grad_slice * (1.0 - beta1);
        add_(m, scaled_g);
        // v = beta2*v + (1-beta2)*g^2
        mul_(v, scalar_like(beta2, v));
        Tensor g_sq = grad_slice * grad_slice;
        mul_(g_sq, scalar_like(1.0 - beta2, g_sq));
        add_(v, g_sq);

        double bc1 = 1.0 - std::pow(beta1, step_count_);
        double bc2 = 1.0 - std::pow(beta2, step_count_);
        Tensor m_hat = m / bc1;
        Tensor v_hat = v / bc2;
        Tensor denom = sqrt(v_hat) + eps;

        if (adamw_opt) {
            // AdamW: decoupled weight decay
            target = target * (1.0 - lr * wd);
        }
        target = target - (m_hat / denom) * lr;
    } else if (sgd_opt) {
        const double lr = sgd_opt->get_lr();
        const double momentum_coef = 0.9;
        if (momentum_coef != 0.0) {
            mul_(m, scalar_like(momentum_coef, m));
            add_(m, grad_slice);
            target = target - m * lr;
        } else {
            target = target - grad_slice * lr;
        }
    } else {
        throw std::runtime_error(
            "ElementLevel mode currently supports Adam, AdamW, SGD base optimizers only");
    }

    // ---- 3) Scatter `target` back into per-parameter tensors. ----
    // For each param overlapping the rank's slice, copy the corresponding slice of
    // target into the param's flat view at the appropriate offset. Downcast on the
    // master path back to the param's dtype.
    //
    // Invariant: every parameter overlapping the rank's slice MUST be contiguous.
    // We write through `pflat = tensor().contiguous().view({-1})`, which only shares
    // storage with `tensor()` when `tensor()` is already contiguous; otherwise
    // contiguous() copies and our writes are silently lost. Parameters are always
    // contiguous at construction in typical optimizer use, but we assert here to
    // catch future regressions before they become silent data-corruption bugs.
    for (size_t i = 0; i < L.params.size(); ++i) {
        const auto& e = L.params[i];
        int64_t p_start = e.global_offset;
        int64_t p_end = e.global_offset + e.numel;
        int64_t lap_start = std::max(p_start, rs);
        int64_t lap_end = std::min(p_end, re);
        if (lap_end <= lap_start) continue;
        if (!parameters_[i]->tensor().is_contiguous()) {
            throw std::runtime_error(
                "ElementLevel scatter-back requires contiguous parameter tensors; "
                "non-contiguous param at index " + std::to_string(i));
        }
        Tensor pflat = parameters_[i]->tensor().contiguous().view({-1});
        Tensor src = target.slice(0, lap_start - rs, lap_end - rs);
        if (src.dtype() != pflat.dtype()) src = src.to(pflat.dtype());
        Tensor dst = pflat.slice(0, lap_start - p_start, lap_end - p_start);
        dst.zero_();
        add_(dst, src);
    }
}

auto ZeROStage1Optimizer::build_rank_grad_slice() -> Tensor {
    const auto& L = partition_layout_;
    const int64_t rs = L.rank_starts[config_.rank];
    const int64_t re = L.rank_starts[config_.rank + 1];
    const int64_t slice_n = re - rs;
    Device dev = local_partition().device;
    DType state_dtype = local_partition().momentum[0].dtype();

    // Phase B (B3): reuse element_grad_slice_buf_ across steps. Saves ~slice_n *
    // dtype_size bytes of allocator churn per step on the element-mode hot path.
    if (element_grad_slice_buf_.numel() != slice_n
        || element_grad_slice_buf_.dtype() != state_dtype
        || element_grad_slice_buf_.device() != dev) {
        element_grad_slice_buf_ = zeros({slice_n}, state_dtype, dev);
    } else {
        element_grad_slice_buf_.zero_();
    }
    Tensor grad_slice = element_grad_slice_buf_;
    for (size_t i = 0; i < L.params.size(); ++i) {
        const auto& e = L.params[i];
        int64_t p_start = e.global_offset;
        int64_t p_end = e.global_offset + e.numel;
        int64_t lap_start = std::max(p_start, rs);
        int64_t lap_end = std::min(p_end, re);
        if (lap_end <= lap_start) continue;

        const auto& var = parameters_[i];
        if (!var->has_grad()) continue;
        const auto& go = var->grad();
        if (!go.has_value()) continue;
        Tensor grad_flat = go.value().contiguous().view({-1});
        if (grad_flat.dtype() != state_dtype) grad_flat = grad_flat.to(state_dtype);

        Tensor src = grad_flat.slice(0, lap_start - p_start, lap_end - p_start);
        Tensor dst = grad_slice.slice(0, lap_start - rs, lap_end - rs);
        add_(dst, src);
    }
    return grad_slice;
}

auto ZeROStage1Optimizer::update_partition_adam(
    StatePartition& partition,
    double lr,
    double beta1,
    double beta2,
    double eps,
    double weight_decay
) -> void {
    // Increment step counter for bias correction
    step_count_++;

    for (size_t i = 0; i < partition.params.size(); ++i) {
        auto& param = partition.params[i];

        if (!param->has_grad()) {
            continue;
        }

        const auto& grad_opt = param->grad();
        if (!grad_opt.has_value()) {
            continue;
        }

        // Pick the tensor we'll actually do the optimizer math on. With use_master_fp32 the
        // master copy is fp32; we read+write it here, then downcast back into param->tensor()
        // at the end of the iteration. Without master we operate directly on param->tensor()
        // (legacy behaviour).
        bool has_master = config_.use_master_fp32
                       && i < partition.master_params.size()
                       && partition.master_params[i].numel() > 0;
        Tensor& target = has_master ? partition.master_params[i] : param->tensor();
        DType target_dtype = target.dtype();

        // Cast grad up to target dtype if necessary (fp16 grad → fp32 master math).
        const Tensor& raw_grad = grad_opt.value();
        Tensor grad = (raw_grad.dtype() != target_dtype) ? raw_grad.to(target_dtype) : raw_grad;

        // Phase B (B1): FusedAdamStep CUDA fast-path. Mirrors src/nn/optim/adam.cpp:30-55.
        // One kernel instead of ~6 OoP allocations + ops per param. Conditions:
        //   - target lives on a CUDA device (the fused kernel is CUDA-only today),
        //   - target dtype is fp32 or fp64 (fused kernel doesn't yet support fp16/bf16),
        //   - no decoupled weight decay (Adam uses L2 reg via grad mutation, AdamW
        //     dispatches with Decoupled=true in update_partition_adamw).
        // The master-FP32 + sync-back-to-param logic still wraps this: the fused
        // dispatch updates `target` (which IS partition.master_params[i] when has_master),
        // and the downcast-to-param.tensor() below copies it back to fp16/bf16.
        if (target.device().type == Device::Type::CUDA &&
            grad.device().type == Device::Type::CUDA &&
            (target_dtype == DType::Float32 || target_dtype == DType::Float64)) {

            std::vector<Tensor> inputs = {
                target, grad, partition.momentum[i], partition.variance[i]
            };

            NewOpAttributes attrs;
            attrs.set(AttrKey::Lr, lr);
            attrs.set(AttrKey::Beta1, beta1);
            attrs.set(AttrKey::Beta2, beta2);
            attrs.set(AttrKey::Eps, eps);
            attrs.set(AttrKey::WeightDecay, weight_decay);
            attrs.set(AttrKey::Step, static_cast<int64_t>(step_count_));
            attrs.set(AttrKey::Decoupled, false);  // Adam: L2 reg
            attrs.set(AttrKey::Amsgrad, false);

            dispatch(OpId::FusedAdamStep, inputs, attrs);

            // Sync master → fp16/bf16 user param so the next forward sees the updates.
            if (has_master) {
                param->tensor() = target.to(param->tensor().dtype());
            }
            continue;
        }

        // CPU / non-fused fallback path.
        // Apply weight decay (L2 regularization)
        Tensor grad_with_decay = grad;
        if (weight_decay != 0.0) {
            grad_with_decay = grad + target * weight_decay;
        }

        // Update biased first moment estimate IN PLACE: m = beta1*m + (1-beta1)*g_eff
        // Saves one full-partition allocation per step versus the rebind-via-OoP form.
        Tensor& momentum = partition.momentum[i];
        mul_(momentum, scalar_like(beta1, momentum));
        {
            Tensor scaled_g = grad_with_decay * (1.0 - beta1);
            add_(momentum, scaled_g);
        }

        // Update biased second moment estimate IN PLACE: v = beta2*v + (1-beta2)*g^2
        Tensor& variance = partition.variance[i];
        mul_(variance, scalar_like(beta2, variance));
        {
            Tensor g_sq_scaled = grad_with_decay * grad_with_decay;
            mul_(g_sq_scaled, scalar_like(1.0 - beta2, g_sq_scaled));
            add_(variance, g_sq_scaled);
        }

        // Compute bias-corrected moment estimates (same as legacy code)
        double bias_correction1 = 1.0 - std::pow(beta1, step_count_);
        double bias_correction2 = 1.0 - std::pow(beta2, step_count_);

        Tensor momentum_corrected = momentum * (1.0 / bias_correction1);
        Tensor variance_corrected = variance * (1.0 / bias_correction2);

        // Compute step: theta = theta - lr * m_hat / (sqrt(v_hat) + eps)
        Tensor denom = sqrt(variance_corrected) + eps;

        // Update target (master if present, else the user-visible param) directly.
        target = target - div(momentum_corrected, denom) * lr;

        // Sync master → fp16/bf16 user param so the next forward sees the updated weights.
        if (has_master) {
            param->tensor() = target.to(param->tensor().dtype());
        }
    }
}

auto ZeROStage1Optimizer::update_partition_adamw(
    StatePartition& partition,
    double lr,
    double beta1,
    double beta2,
    double eps,
    double weight_decay
) -> void {
    // Increment step counter for bias correction (shares same counter as Adam)
    step_count_++;

    for (size_t i = 0; i < partition.params.size(); ++i) {
        auto& param = partition.params[i];

        if (!param->has_grad()) {
            continue;
        }

        const auto& grad_opt = param->grad();
        if (!grad_opt.has_value()) {
            continue;
        }

        // Master-fp32 plumbing: same idea as in update_partition_adam — when enabled, do the
        // optimizer math against the fp32 master and downcast to the user's fp16/bf16 param at
        // the end. Without master, operate directly on the param tensor (legacy behaviour).
        bool has_master = config_.use_master_fp32
                       && i < partition.master_params.size()
                       && partition.master_params[i].numel() > 0;
        Tensor& target = has_master ? partition.master_params[i] : param->tensor();
        DType target_dtype = target.dtype();

        const Tensor& raw_grad = grad_opt.value();
        Tensor grad = (raw_grad.dtype() != target_dtype) ? raw_grad.to(target_dtype) : raw_grad;

        // Phase B (B1): FusedAdamStep CUDA fast-path with Decoupled=true (AdamW path).
        // See update_partition_adam for full rationale.
        if (target.device().type == Device::Type::CUDA &&
            grad.device().type == Device::Type::CUDA &&
            (target_dtype == DType::Float32 || target_dtype == DType::Float64)) {

            std::vector<Tensor> inputs = {
                target, grad, partition.momentum[i], partition.variance[i]
            };

            NewOpAttributes attrs;
            attrs.set(AttrKey::Lr, lr);
            attrs.set(AttrKey::Beta1, beta1);
            attrs.set(AttrKey::Beta2, beta2);
            attrs.set(AttrKey::Eps, eps);
            attrs.set(AttrKey::WeightDecay, weight_decay);
            attrs.set(AttrKey::Step, static_cast<int64_t>(step_count_));
            attrs.set(AttrKey::Decoupled, true);   // AdamW: decoupled weight decay
            attrs.set(AttrKey::Amsgrad, false);

            dispatch(OpId::FusedAdamStep, inputs, attrs);

            if (has_master) {
                param->tensor() = target.to(param->tensor().dtype());
            }
            continue;
        }

        Tensor& momentum = partition.momentum[i];
        Tensor& variance = partition.variance[i];

        // m = beta1*m + (1-beta1)*grad   (in-place; AdamW does *not* fold weight decay into grad)
        mul_(momentum, scalar_like(beta1, momentum));
        {
            Tensor scaled_g = grad * (1.0 - beta1);
            add_(momentum, scaled_g);
        }

        // v = beta2*v + (1-beta2)*grad^2  (in-place)
        mul_(variance, scalar_like(beta2, variance));
        {
            Tensor g_sq_scaled = grad * grad;
            mul_(g_sq_scaled, scalar_like(1.0 - beta2, g_sq_scaled));
            add_(variance, g_sq_scaled);
        }

        // Final update: keep the legacy numerical sequence so the loss trajectory matches the
        // pre-refactor implementation bit-for-bit.
        double bias_correction1 = 1.0 - std::pow(beta1, step_count_);
        double bias_correction2 = 1.0 - std::pow(beta2, step_count_);

        Tensor momentum_corrected = momentum * (1.0 / bias_correction1);
        Tensor variance_corrected = variance * (1.0 / bias_correction2);
        Tensor denom = sqrt(variance_corrected) + eps;

        if (weight_decay != 0.0) {
            // AdamW: decoupled weight decay
            target = target * (1.0 - lr * weight_decay) -
                     div(momentum_corrected, denom) * lr;
        } else {
            target = target - div(momentum_corrected, denom) * lr;
        }

        if (has_master) {
            param->tensor() = target.to(param->tensor().dtype());
        }
    }
}

auto ZeROStage1Optimizer::update_partition_sgd(
    StatePartition& partition,
    double lr,
    double momentum_coef,
    double weight_decay
) -> void {
    for (size_t i = 0; i < partition.params.size(); ++i) {
        auto& param = partition.params[i];

        if (!param->has_grad()) {
            continue;
        }

        const auto& grad_opt = param->grad();
        if (!grad_opt.has_value()) {
            continue;
        }

        // Master-fp32 plumbing — see update_partition_adam for rationale.
        bool has_master = config_.use_master_fp32
                       && i < partition.master_params.size()
                       && partition.master_params[i].numel() > 0;
        Tensor& target = has_master ? partition.master_params[i] : param->tensor();
        DType target_dtype = target.dtype();

        const Tensor& raw_grad = grad_opt.value();
        Tensor grad = (raw_grad.dtype() != target_dtype) ? raw_grad.to(target_dtype) : raw_grad;

        // Apply weight decay (L2 regularization)
        Tensor grad_with_decay = grad;
        if (weight_decay != 0.0) {
            grad_with_decay = grad + target * weight_decay;
        }

        if (momentum_coef != 0.0) {
            // SGD with momentum (in-place buffer update; param update via legacy OoP form
            // because in-place ops on the parameter tensor are blocked by requires_grad).
            Tensor& momentum = partition.momentum[i];
            mul_(momentum, scalar_like(momentum_coef, momentum));
            add_(momentum, grad_with_decay);
            target = target - momentum * lr;
        } else {
            // Vanilla SGD
            target = target - grad_with_decay * lr;
        }

        if (has_master) {
            param->tensor() = target.to(param->tensor().dtype());
        }
    }
}

auto ZeROStage1Optimizer::fetch_states_to_gpu() -> void {
    if (!states_on_cpu_) {
        return;  // Nothing to fetch if states are already on GPU
    }

    auto& partition = local_partition();
    Device param_device = !parameters_.empty() ? parameters_[0]->tensor().device() : Device::cpu();

    // NVMe path: read each state blob from disk, dequantize if applicable, place on the
    // parameter device. Master copies always stayed in memory (they were never written to
    // disk) so we don't touch them here.
    if (config_.offload_to_nvme) {
        size_t total_bytes_nvme = 0;
        const DType dequant_target = DType::Float32;

        auto load_slot = [&](Tensor& state,
                             StatePartition::DiskSlot& slot,
                             Tensor& scale_inmem,
                             StatePartition::DiskSlot& scale_slot) {
            if (!slot.on_disk()) return;
            // Read the int8 payload (or raw fp32 if not quantized) into the param device.
            Tensor data = read_tensor_blob(slot.path, slot.shape, slot.dtype, param_device);
            if (scale_slot.on_disk()) {
                Tensor scale = read_tensor_blob(scale_slot.path, scale_slot.shape, scale_slot.dtype, param_device);
                state = dequantize_from_int8(data, scale, dequant_target);
                scale_slot.path.clear();
                scale_inmem = Tensor();  // make sure no stale in-mem scale lingers
            } else {
                state = data;
            }
            slot.path.clear();
            total_bytes_nvme += static_cast<size_t>(state.numel()) * dtype_size(state.dtype());
        };

        Tensor dummy_inmem;
        StatePartition::DiskSlot dummy_slot;

        for (size_t i = 0; i < partition.momentum.size(); ++i) {
            Tensor& m_scale_inmem = (i < partition.momentum_scales.size())
                ? partition.momentum_scales[i] : dummy_inmem;
            StatePartition::DiskSlot& m_scale_slot = (i < partition.momentum_scale_disk.size())
                ? partition.momentum_scale_disk[i] : dummy_slot;
            load_slot(partition.momentum[i], partition.momentum_disk[i],
                      m_scale_inmem, m_scale_slot);

            Tensor& v_scale_inmem = (i < partition.variance_scales.size())
                ? partition.variance_scales[i] : dummy_inmem;
            StatePartition::DiskSlot& v_scale_slot = (i < partition.variance_scale_disk.size())
                ? partition.variance_scale_disk[i] : dummy_slot;
            load_slot(partition.variance[i], partition.variance_disk[i],
                      v_scale_inmem, v_scale_slot);
        }

        states_on_cpu_ = false;
        if (profiling_enabled_) {
            std::lock_guard<std::mutex> prof_lock(profiling_mutex_);
            profiling_stats_.offloaded_bytes += total_bytes_nvme;
        }
        return;
    }

    // CPU offload path: requires offload_engine and a non-CPU param device.
    if (!offload_engine_) {
        return;
    }
    if (param_device.type == Device::Type::CPU) {
        return;  // Skip for CPU-only training
    }

    // Effective dtype to dequantize back to. We only quantize fp32 states (see
    // offload_states_to_cpu), so dequantization always targets fp32.
    const DType dequant_target = DType::Float32;

    auto load_and_maybe_dequantize = [&](Tensor& state, Tensor& scale_slot) {
        if (scale_slot.numel() > 0) {
            // Quantized payload: pull both pieces to GPU, dequantize, drop the scale slot.
            Tensor gpu_int8 = offload_engine_->load_to_gpu(state, param_device);
            Tensor gpu_scale = offload_engine_->load_to_gpu(scale_slot, param_device);
            state = dequantize_from_int8(gpu_int8, gpu_scale, dequant_target);
            scale_slot = Tensor();  // back to "not quantized"
        } else {
            state = offload_engine_->load_to_gpu(state, param_device);
        }
    };

    // Synchronously load all states to GPU and update tensor references
    size_t total_bytes = 0;
    for (size_t i = 0; i < partition.momentum.size(); ++i) {
        Tensor* m_scale = (i < partition.momentum_scales.size()) ? &partition.momentum_scales[i] : nullptr;
        Tensor* v_scale = (i < partition.variance_scales.size()) ? &partition.variance_scales[i] : nullptr;
        Tensor dummy;

        load_and_maybe_dequantize(partition.momentum[i], m_scale ? *m_scale : dummy);
        load_and_maybe_dequantize(partition.variance[i], v_scale ? *v_scale : dummy);

        if (profiling_enabled_) {
            total_bytes += partition.momentum[i].numel() * dtype_size(partition.momentum[i].dtype());
            total_bytes += partition.variance[i].numel() * dtype_size(partition.variance[i].dtype());
        }
    }

    // Master fp32 copies live alongside momentum/variance — fetch them too so the optimizer
    // step has something to read+write. Skip slots where the master is empty (param was
    // already fp32 at init time, so we never allocated a master).
    for (size_t i = 0; i < partition.master_params.size(); ++i) {
        if (partition.master_params[i].numel() == 0) continue;
        Tensor gpu_master = offload_engine_->load_to_gpu(partition.master_params[i], param_device);
        partition.master_params[i] = gpu_master;
        if (profiling_enabled_) {
            total_bytes += gpu_master.numel() * dtype_size(gpu_master.dtype());
        }
    }

    states_on_cpu_ = false;

    // Track bytes transferred
    if (profiling_enabled_) {
        std::lock_guard<std::mutex> prof_lock(profiling_mutex_);
        profiling_stats_.offloaded_bytes += total_bytes;
    }
}

auto ZeROStage1Optimizer::offload_states_to_cpu() -> void {
    if (states_on_cpu_) {
        return;  // Nothing to offload if states are already off-GPU (CPU or NVMe)
    }

    auto& partition = local_partition();

    // NVMe path: write each state (optionally quantized) to disk, free the in-memory
    // tensor. The per-tensor file path was assigned during initialize_optimizer_states.
    if (config_.offload_to_nvme) {
        const bool quantize = config_.quantize_offloaded_states_int8;
        size_t total_bytes_nvme = 0;
        std::filesystem::path nvme_dir = resolve_nvme_dir(config_.nvme_path);

        auto write_slot = [&](Tensor& state,
                              StatePartition::DiskSlot& slot,
                              size_t slot_idx,
                              std::string_view state_name,
                              Tensor& scale_inmem,
                              StatePartition::DiskSlot* scale_slot,
                              std::string_view scale_name) {
            if (state.numel() == 0) return;  // already on disk; idempotent
            QuantizedInt8 qs;
            const Tensor* data_to_write = &state;
            std::vector<int64_t> shape(state.shape().begin(), state.shape().end());
            DType dt = state.dtype();
            if (quantize && state.dtype() == DType::Float32) {
                qs = quantize_to_int8(state);
                data_to_write = &qs.data;
                dt = qs.data.dtype();
            }
            slot.path  = state_blob_path(nvme_dir, partition.rank, slot_idx, state_name).string();
            slot.shape = std::move(shape);
            slot.dtype = dt;
            total_bytes_nvme += write_tensor_blob(*data_to_write, slot.path);

            if (qs.scale.numel() > 0 && scale_slot != nullptr) {
                scale_slot->path  = state_blob_path(nvme_dir, partition.rank, slot_idx, scale_name).string();
                scale_slot->shape = std::vector<int64_t>(qs.scale.shape().begin(), qs.scale.shape().end());
                scale_slot->dtype = qs.scale.dtype();
                write_tensor_blob(qs.scale, scale_slot->path);
                scale_inmem = Tensor();  // ensure no stale in-memory scale
            }

            // Free the GPU/CPU-resident state tensor.
            state = Tensor();
        };

        Tensor dummy_inmem;
        StatePartition::DiskSlot dummy_slot;

        for (size_t i = 0; i < partition.momentum.size(); ++i) {
            Tensor& m_scale_inmem = (i < partition.momentum_scales.size())
                ? partition.momentum_scales[i] : dummy_inmem;
            StatePartition::DiskSlot* m_scale_slot = (i < partition.momentum_scale_disk.size())
                ? &partition.momentum_scale_disk[i] : nullptr;
            write_slot(partition.momentum[i], partition.momentum_disk[i], i,
                       "momentum", m_scale_inmem, m_scale_slot, "momentum_scale");

            Tensor& v_scale_inmem = (i < partition.variance_scales.size())
                ? partition.variance_scales[i] : dummy_inmem;
            StatePartition::DiskSlot* v_scale_slot = (i < partition.variance_scale_disk.size())
                ? &partition.variance_scale_disk[i] : nullptr;
            write_slot(partition.variance[i], partition.variance_disk[i], i,
                       "variance", v_scale_inmem, v_scale_slot, "variance_scale");
        }

        states_on_cpu_ = true;
        if (profiling_enabled_) {
            std::lock_guard<std::mutex> prof_lock(profiling_mutex_);
            profiling_stats_.offloaded_bytes += total_bytes_nvme;
            profiling_stats_.num_offloads++;
        }
        return;
    }

    if (!offload_engine_) {
        return;
    }

    // Only offload if parameters are on GPU (CPU offload only makes sense for GPU training)
    Device param_device = !parameters_.empty() ? parameters_[0]->tensor().device() : Device::cpu();
    if (param_device.type == Device::Type::CPU) {
        return;  // Skip for CPU-only training
    }

    // Whether int8 quantization should be applied this round. Only quantize fp32 states —
    // fp16/bf16 starting points lose too much precision when squashed to int8, and fp64
    // states are rare enough that we'd rather leave them untouched.
    const bool quantize = config_.quantize_offloaded_states_int8;

    auto maybe_quantize_and_offload = [&](Tensor& state, Tensor& scale_slot) {
        if (quantize && state.dtype() == DType::Float32) {
            // Quantize while still on GPU (keeps the fp32→int8 conversion off the host),
            // then offload the smaller int8 payload + scalar scale.
            QuantizedInt8 qs = quantize_to_int8(state);
            state = offload_engine_->offload_to_cpu(qs.data);
            scale_slot = offload_engine_->offload_to_cpu(qs.scale);
        } else {
            state = offload_engine_->offload_to_cpu(state);
        }
    };

    // Synchronously offload all states to CPU and update tensor references
    size_t total_bytes = 0;
    for (size_t i = 0; i < partition.momentum.size(); ++i) {
        Tensor* m_scale = (i < partition.momentum_scales.size()) ? &partition.momentum_scales[i] : nullptr;
        Tensor* v_scale = (i < partition.variance_scales.size()) ? &partition.variance_scales[i] : nullptr;
        Tensor dummy;  // for paths where the partition didn't allocate scale slots

        maybe_quantize_and_offload(partition.momentum[i], m_scale ? *m_scale : dummy);
        maybe_quantize_and_offload(partition.variance[i], v_scale ? *v_scale : dummy);

        if (profiling_enabled_) {
            total_bytes += partition.momentum[i].numel() * dtype_size(partition.momentum[i].dtype());
            total_bytes += partition.variance[i].numel() * dtype_size(partition.variance[i].dtype());
            // Scales are tiny (1 element each); skip from the bandwidth accounting.
        }
    }

    // Master fp32 copies follow the same offload lifecycle as momentum/variance.
    for (size_t i = 0; i < partition.master_params.size(); ++i) {
        if (partition.master_params[i].numel() == 0) continue;
        Tensor cpu_master = offload_engine_->offload_to_cpu(partition.master_params[i]);
        partition.master_params[i] = cpu_master;
        if (profiling_enabled_) {
            total_bytes += cpu_master.numel() * dtype_size(cpu_master.dtype());
        }
    }

    states_on_cpu_ = true;

    // Track offloaded bytes and offload operations
    if (profiling_enabled_) {
        std::lock_guard<std::mutex> prof_lock(profiling_mutex_);
        profiling_stats_.offloaded_bytes += total_bytes;
        profiling_stats_.num_offloads++;
    }
}

// =============================================================================
// ZeRO Stage 2 Optimizer Implementation
// =============================================================================

ZeROStage2Optimizer::ZeROStage2Optimizer(
    std::shared_ptr<Optimizer> base_optimizer,
    const ZeROStage2Config& config
) : ZeROStage1Optimizer(std::move(base_optimizer), config),
    stage2_config_(config) {

    if (stage2_config_.gradient_bucketing) {
        if (config_.partitioning_mode == PartitioningMode::ElementLevel) {
            create_gradient_buckets_element_mode();
        } else {
            create_gradient_buckets();
        }
    }
}

ZeROStage2Optimizer::~ZeROStage2Optimizer() {
    // Detach autograd hooks before any of our state (this, buckets) goes away — otherwise
    // a backward() that runs after the optimizer is destroyed would dereference freed memory.
    try {
        unregister_backward_hooks();
    } catch (...) {
        // Destructors must not throw. Hook removal is best-effort.
    }
}

auto ZeROStage2Optimizer::step_impl() -> void {
    std::lock_guard<std::mutex> lock(mutex_);

    // Start profiling
    auto step_start = std::chrono::steady_clock::now();
    if (profiling_enabled_) {
        step_start_time_ = step_start;
    }

    // Step 1: Gradients are already reduced-scattered via backward hooks
    // No need for all-reduce like in Stage 1

    // Step 2: Fetch optimizer states from CPU if offloaded
    if ((config_.offload_to_cpu && offload_engine_) || config_.offload_to_nvme) {
        auto offload_start = std::chrono::steady_clock::now();
        fetch_states_to_gpu();
        if (profiling_enabled_) {
            auto offload_end = std::chrono::steady_clock::now();
            auto offload_duration = std::chrono::duration<double, std::milli>(offload_end - offload_start).count();
            std::lock_guard<std::mutex> prof_lock(profiling_mutex_);
            profiling_stats_.offload_time_ms += offload_duration;
        }
    }

    // Step 3: Update local partition of parameters
    // Uses the local (reduced-scattered) gradients
    auto compute_start = std::chrono::steady_clock::now();
    if (config_.partitioning_mode == PartitioningMode::ElementLevel) {
        update_local_partition_element_mode();
    } else {
        update_local_partition();
    }
    if (profiling_enabled_) {
        auto compute_end = std::chrono::steady_clock::now();
        auto compute_duration = std::chrono::duration<double, std::milli>(compute_end - compute_start).count();
        std::lock_guard<std::mutex> prof_lock(profiling_mutex_);
        profiling_stats_.compute_time_ms += compute_duration;
    }

    // Step 4: Offload states back to CPU if enabled
    if ((config_.offload_to_cpu && offload_engine_) || config_.offload_to_nvme) {
        auto offload_start = std::chrono::steady_clock::now();
        offload_states_to_cpu();
        if (profiling_enabled_) {
            auto offload_end = std::chrono::steady_clock::now();
            auto offload_duration = std::chrono::duration<double, std::milli>(offload_end - offload_start).count();
            std::lock_guard<std::mutex> prof_lock(profiling_mutex_);
            profiling_stats_.offload_time_ms += offload_duration;
        }
    }

    // Step 5: All-gather updated parameters across ranks
    if (config_.world_size > 1) {
        auto gather_start = std::chrono::steady_clock::now();
        if (config_.partitioning_mode == PartitioningMode::ElementLevel) {
            all_gather_parameters_element_mode();
        } else {
            all_gather_parameters();
        }
        if (profiling_enabled_) {
            auto gather_end = std::chrono::steady_clock::now();
            auto gather_duration = std::chrono::duration<double, std::milli>(gather_end - gather_start).count();
            std::lock_guard<std::mutex> prof_lock(profiling_mutex_);
            profiling_stats_.all_gather_time_ms += gather_duration;
            profiling_stats_.gather_time_ms += gather_duration;
            profiling_stats_.communication_time_ms += gather_duration;
            profiling_stats_.num_all_gathers++;
            profiling_stats_.num_gathers++;
        }
    }

    // Complete profiling
    if (profiling_enabled_) {
        auto step_end = std::chrono::steady_clock::now();
        auto total_duration = std::chrono::duration<double, std::milli>(step_end - step_start).count();

        std::lock_guard<std::mutex> prof_lock(profiling_mutex_);
        profiling_stats_.total_step_time_ms += total_duration;
        profiling_stats_.num_steps++;

        // Update averages
        profiling_stats_.avg_step_time_ms =
            profiling_stats_.total_step_time_ms / profiling_stats_.num_steps;
        if (profiling_stats_.num_gathers > 0) {
            profiling_stats_.avg_gather_time_ms =
                profiling_stats_.gather_time_ms / profiling_stats_.num_gathers;
        }
        if (profiling_stats_.num_scatters > 0) {
            profiling_stats_.avg_scatter_time_ms =
                profiling_stats_.scatter_time_ms / profiling_stats_.num_scatters;
        }

        // Calculate communication/compute overlap
        if (total_duration > 0) {
            double sequential_time = profiling_stats_.communication_time_ms + profiling_stats_.compute_time_ms;
            if (sequential_time > total_duration) {
                profiling_stats_.comm_compute_overlap_ratio =
                    1.0 - (total_duration / sequential_time);
            }
        }

        // Calculate bandwidth
        if (profiling_stats_.communication_time_ms > 0) {
            double seconds = profiling_stats_.communication_time_ms / 1000.0;
            double megabytes = profiling_stats_.transferred_bytes / (1024.0 * 1024.0);
            profiling_stats_.effective_bandwidth_mbps = megabytes / seconds;
        }
    }
}

auto ZeROStage2Optimizer::register_backward_hooks() -> void {
    std::lock_guard<std::mutex> lock(buckets_mutex_);

    if (hooks_registered_) {
        return;  // Already registered
    }

    if (!stage2_config_.reduce_scatter_in_backward) {
        hooks_registered_ = true;
        return;  // Hooks disabled in config
    }

    if (config_.partitioning_mode == PartitioningMode::ElementLevel) {
        // One hook per parameter; routes into element_gradient_hook.
        element_hook_ids_.assign(parameters_.size(), 0);
        for (size_t i = 0; i < parameters_.size(); ++i) {
            auto& p = parameters_[i];
            if (!p) continue;
            element_hook_ids_[i] = p->register_hook(
                [this, i](const Tensor& grad) -> Tensor {
                    this->element_gradient_hook(i, grad);
                    return grad;
                });
        }
        hooks_registered_ = true;
        return;
    }

    // ... existing ParamLevel code unchanged below ...

    // Register a hook on every parameter Variable. The autograd engine calls these
    // during backward, *before* gradient accumulation, with the freshly-computed grad.
    // We stash the grad into the bucket's per-param slot and, once the bucket is full,
    // fire reduce-scatter so it can overlap with the rest of backward.
    for (size_t bucket_idx = 0; bucket_idx < gradient_buckets_.size(); ++bucket_idx) {
        auto& bucket = gradient_buckets_[bucket_idx];

        bucket.gradient_buffers.assign(bucket.params.size(), Tensor());
        bucket.hook_ids.assign(bucket.params.size(), 0);
        bucket.gradients_received = 0;
        bucket.ready = false;

        for (size_t param_idx = 0; param_idx < bucket.params.size(); ++param_idx) {
            auto& param = bucket.params[param_idx];
            if (!param) {
                continue;
            }

            // Capture indices by value; capture `this` raw because the destructor
            // unregisters hooks before optimizer state is destroyed (see ~ZeROStage2Optimizer).
            size_t hook_id = param->register_hook(
                [this, bucket_idx, param_idx](const Tensor& grad) -> Tensor {
                    this->gradient_hook(bucket_idx, param_idx, grad);
                    return grad;  // Pass-through; reduce-scatter has stashed its own copy.
                }
            );

            bucket.hook_ids[param_idx] = hook_id;
        }
    }

    hooks_registered_ = true;
}

auto ZeROStage2Optimizer::unregister_backward_hooks() -> void {
    std::lock_guard<std::mutex> lock(buckets_mutex_);

    if (!hooks_registered_) {
        return;
    }

    if (config_.partitioning_mode == PartitioningMode::ElementLevel) {
        for (size_t i = 0; i < parameters_.size() && i < element_hook_ids_.size(); ++i) {
            if (parameters_[i]) parameters_[i]->unregister_hook(element_hook_ids_[i]);
        }
        element_hook_ids_.clear();
        for (auto& b : element_buckets_) {
            b.flat_buffer = Tensor();
            b.hooks_received = 0;
        }
        hooks_registered_ = false;
        return;
    }

    // ... existing ParamLevel code unchanged below ...

    for (auto& bucket : gradient_buckets_) {
        for (size_t i = 0; i < bucket.params.size() && i < bucket.hook_ids.size(); ++i) {
            auto& param = bucket.params[i];
            if (param) {
                param->unregister_hook(bucket.hook_ids[i]);
            }
        }
        bucket.hook_ids.clear();
        bucket.gradient_buffers.clear();
        bucket.gradients_received = 0;
        bucket.ready = false;
    }

    hooks_registered_ = false;
}

auto ZeROStage2Optimizer::get_bucket_stats() const -> BucketStats {
    std::lock_guard<std::mutex> lock(buckets_mutex_);

    BucketStats stats;
    stats.num_buckets = gradient_buckets_.size();

    size_t total_size = 0;
    size_t max_size = 0;

    for (const auto& bucket : gradient_buckets_) {
        total_size += bucket.total_size;
        max_size = std::max(max_size, bucket.total_size);
    }

    stats.total_gradient_memory = total_size;
    stats.max_bucket_size = max_size;

    if (stats.num_buckets > 0) {
        stats.avg_bucket_size = total_size / stats.num_buckets;
    }

    return stats;
}

// =============================================================================
// Private: Initialization
// =============================================================================

auto ZeROStage2Optimizer::create_gradient_buckets() -> void {
    std::lock_guard<std::mutex> lock(buckets_mutex_);

    // Group parameters into buckets based on target rank and size
    // Goal: Create buckets of approximately gradient_bucket_size bytes

    gradient_buckets_.clear();

    // Create one bucket per rank to start
    gradient_buckets_.resize(config_.world_size);

    for (int rank = 0; rank < config_.world_size; ++rank) {
        gradient_buckets_[rank].target_rank = rank;
    }

    // Assign parameters to buckets based on which rank owns them
    for (size_t param_idx = 0; param_idx < parameters_.size(); ++param_idx) {
        const auto& param = parameters_[param_idx];

        // Determine which rank owns this parameter (same as Stage 1 partitioning)
        size_t params_per_rank = (parameters_.size() + config_.world_size - 1) / config_.world_size;
        int owner_rank = static_cast<int>(param_idx / params_per_rank);

        if (owner_rank >= config_.world_size) {
            owner_rank = config_.world_size - 1;
        }

        // Add parameter to the bucket for its owner rank
        auto& bucket = gradient_buckets_[owner_rank];
        bucket.params.push_back(param);

        // Calculate gradient size
        const auto& tensor = param->tensor();
        size_t grad_size = tensor.numel() * dtype_size(tensor.dtype());
        bucket.total_size += grad_size;
    }

    // If bucketing is enabled, potentially split large buckets
    if (stage2_config_.gradient_bucketing && stage2_config_.gradient_bucket_size > 0) {
        std::vector<GradientBucket> new_buckets;

        for (auto& bucket : gradient_buckets_) {
            // If bucket is too large, split it
            if (bucket.total_size > stage2_config_.gradient_bucket_size * 2) {
                // Split into multiple sub-buckets
                size_t target_num_buckets =
                    (bucket.total_size + stage2_config_.gradient_bucket_size - 1) /
                    stage2_config_.gradient_bucket_size;

                size_t params_per_bucket =
                    (bucket.params.size() + target_num_buckets - 1) / target_num_buckets;

                for (size_t i = 0; i < bucket.params.size(); i += params_per_bucket) {
                    GradientBucket sub_bucket;
                    sub_bucket.target_rank = bucket.target_rank;

                    size_t end_idx = std::min(i + params_per_bucket, bucket.params.size());

                    for (size_t j = i; j < end_idx; ++j) {
                        sub_bucket.params.push_back(bucket.params[j]);
                        const auto& tensor = bucket.params[j]->tensor();
                        size_t grad_size = tensor.numel() * dtype_size(tensor.dtype());
                        sub_bucket.total_size += grad_size;
                    }

                    new_buckets.push_back(std::move(sub_bucket));
                }
            } else {
                new_buckets.push_back(std::move(bucket));
            }
        }

        gradient_buckets_ = std::move(new_buckets);
    }

    // Initialize gradient buffers for each bucket and pre-compute the per-param offsets
    // into the persistent flat_buffer. The flat_buffer itself is allocated lazily on the
    // first reduce_scatter_gradients call (we don't yet know dtype/device until we see a
    // real grad — and for tests that never fire the hook, we shouldn't allocate at all).
    for (auto& bucket : gradient_buckets_) {
        bucket.gradient_buffers.reserve(bucket.params.size());
        bucket.gradients_received = 0;
        bucket.ready = false;

        bucket.param_offsets_elem.clear();
        bucket.param_sizes_elem.clear();
        bucket.param_offsets_elem.reserve(bucket.params.size());
        bucket.param_sizes_elem.reserve(bucket.params.size());
        int64_t offset = 0;
        for (const auto& param : bucket.params) {
            int64_t n = param ? static_cast<int64_t>(param->tensor().numel()) : 0;
            bucket.param_offsets_elem.push_back(offset);
            bucket.param_sizes_elem.push_back(n);
            offset += n;
        }
    }
}

auto ZeROStage2Optimizer::create_gradient_buckets_element_mode() -> void {
    std::lock_guard<std::mutex> lock(buckets_mutex_);
    element_buckets_.clear();

    const auto& L = partition_layout_;
    if (L.params.empty()) return;

    DType dt = parameters_.empty() ? DType::Float32 : parameters_[0]->tensor().dtype();
    const size_t bytes_per_elem = dtype_size(dt);
    const size_t target_bucket_bytes = stage2_config_.gradient_bucket_size;
    const int64_t W = config_.world_size;

    // Bucket by GLOBAL element range. Round each bucket size DOWN to a world_size
    // multiple so reduce_scatter cleanly splits it.
    const int64_t bucket_elems = std::max<int64_t>(
        W,  // bucket must hold at least one element per rank
        ((static_cast<int64_t>(target_bucket_bytes / bytes_per_elem)) / W) * W);
    int64_t cursor = 0;
    int64_t total = L.total_elements_padded;
    while (cursor < total) {
        ElementBucket b;
        b.global_start = cursor;
        b.global_end = std::min(cursor + bucket_elems, total);

        // Find which param indices have any element in [global_start, global_end).
        for (size_t i = 0; i < L.params.size(); ++i) {
            const auto& e = L.params[i];
            int64_t p_end = e.global_offset + e.numel;
            if (p_end <= b.global_start) continue;
            if (e.global_offset >= b.global_end) break;
            b.param_indices.push_back(i);
        }
        element_buckets_.push_back(std::move(b));
        cursor = element_buckets_.back().global_end;
    }
}

// =============================================================================
// Private: Communication
// =============================================================================

auto ZeROStage2Optimizer::reduce_scatter_gradients(GradientBucket& bucket) -> void {
    if (!config_.process_group || config_.world_size <= 1) {
        return;  // No communication needed
    }

    auto scatter_start = std::chrono::steady_clock::now();

    // Phase B (B4): if gradient_hook pre-staged grads directly into bucket.flat_buffer
    // (the new path that releases param->grad() immediately to halve transient grad
    // memory), skip the gradient_buffers staging entirely -- flat_buffer is already
    // populated and ready for the collective.
    bool pre_staged = false;
    {
        std::lock_guard<std::mutex> stash_lock(*bucket.mutex);
        pre_staged = bucket.flat_pre_staged;
    }

    // Collect all gradients from the bucket. Prefer the per-bucket stash populated by
    // gradient_hook(...) (autograd-driven path), since param->grad() may not be populated yet
    // when the hook for the LAST param fires (engine.cpp accumulates *after* hooks run).
    // Fall back to param->grad() for callers who never installed hooks.
    std::vector<Tensor> gradients;
    gradients.reserve(bucket.params.size());

    size_t total_bytes = 0;
    bool buffers_have_data = false;
    if (!pre_staged) {
        std::lock_guard<std::mutex> stash_lock(*bucket.mutex);
        if (bucket.gradient_buffers.size() == bucket.params.size()) {
            buffers_have_data = std::all_of(
                bucket.gradient_buffers.begin(), bucket.gradient_buffers.end(),
                [](const Tensor& t) { return t.numel() > 0; }
            );
        }
        if (buffers_have_data) {
            for (const auto& g : bucket.gradient_buffers) {
                gradients.push_back(g);
                if (profiling_enabled_) {
                    total_bytes += g.numel() * dtype_size(g.dtype());
                }
            }
        }
    } else {
        // Pre-staged: bytes are already in flat_buffer. We still need to record total
        // bytes for profiling.
        if (profiling_enabled_) {
            total_bytes = static_cast<size_t>(bucket.flat_buffer.numel())
                        * dtype_size(bucket.flat_buffer.dtype());
        }
    }

    if (!pre_staged && !buffers_have_data) {
        for (const auto& param : bucket.params) {
            if (!param->has_grad()) {
                throw std::runtime_error("Parameter missing gradient in reduce-scatter");
            }

            const auto& grad_opt = param->grad();
            if (!grad_opt.has_value()) {
                throw std::runtime_error("Parameter gradient not computed in reduce-scatter");
            }

            Tensor grad = grad_opt.value();
            gradients.push_back(grad);

            if (profiling_enabled_) {
                total_bytes += grad.numel() * dtype_size(grad.dtype());
            }
        }
    }

    if (!pre_staged && gradients.empty()) {
        return;
    }

    // Stage gradients into the bucket's persistent flat buffer instead of allocating a fresh
    // tensor every step. The buffer is sized once (lazily, when we know dtype/device from the
    // first real grad) and reused for the lifetime of the bucket. Per training step this
    // eliminates one full-bucket-sized allocation (the legacy flatten_tensors result) and one
    // partition-sized allocation (the legacy zeros() output of reduce_scatter).
    // When pre_staged, flat_buffer is already populated by gradient_hook (no allocation
    // or staging needed). Otherwise, lazy-alloc + stage from the gradient_buffers stash
    // (legacy path for callers driving gradient_hook manually without bucket geometry).
    if (!pre_staged) {
        int64_t total_elements = 0;
        for (int64_t n : bucket.param_sizes_elem) total_elements += n;

        DType buf_dtype = gradients.front().dtype();
        Device buf_device = gradients.front().device();

        if (bucket.flat_buffer.numel() != total_elements ||
            bucket.flat_buffer.dtype() != buf_dtype ||
            bucket.flat_buffer.device() != buf_device) {
            bucket.flat_buffer = zeros({total_elements}, buf_dtype, buf_device);
        } else {
            // Reset stale residue from the previous step before re-staging.
            bucket.flat_buffer.zero_();
        }

        // Stage each grad into its slot. The buffer is currently zero everywhere, so add_ acts
        // as a copy (zero + grad = grad) without needing a public copy_ API.
        for (size_t i = 0; i < gradients.size(); ++i) {
            const Tensor& g = gradients[i];
            if (i >= bucket.param_offsets_elem.size() || g.numel() == 0) continue;
            int64_t off = bucket.param_offsets_elem[i];
            int64_t sz  = bucket.param_sizes_elem[i];
            if (g.numel() != sz) {
                throw std::runtime_error("reduce_scatter_gradients: gradient size mismatch");
            }
            Tensor slot = bucket.flat_buffer.slice(0, off, off + sz);
            Tensor flat_grad = g.contiguous().view({-1});
            add_(slot, flat_grad);
        }
    }

    Tensor& flat_grads = bucket.flat_buffer;

    // Run the collective. Replace the legacy chunked reduce_scatter with a single `reduce`
    // into the bucket's owner rank — that's what ZeRO-Stage-2 actually wants. The legacy
    // path split the bucket into world_size equal chunks and gave each rank one chunk back,
    // then asserted (incorrectly) that the owner's chunk equalled the bucket's full
    // gradient. With shapes mismatched, the subsequent `unflatten_into(local_grad_sum,
    // full_grads)` would have thrown for any world_size > 1 — only single-process tests
    // ever reached the post-collective code path, masking the bug. The reduce form is also
    // strictly less network traffic on the owner side: each non-owner sends its bucket
    // once, the owner receives + sums, instead of all-to-all chunk distribution.
    if (config_.world_size > 1 && config_.process_group) {
        // Phase E (E4): apply the optional gradient compressor before the
        // collective. Stage 1 already supports this for all-reduce
        // (zero_optimizer.cpp:1083); Stage 2's reduce path now mirrors the
        // shape: compress flat_grads, reduce the (smaller) compressed payload,
        // decompress on the owner. Saves bandwidth proportional to the
        // compression ratio.
        //
        // Caveat: this assumes the compressor is sum-additive
        // (sum(compress(x_i)) == compress(sum(x_i))). Lossy schemes that
        // don't satisfy this should leave grad_compressor unset for Stage 2.
        if (config_.grad_compressor) {
            auto compressed = config_.grad_compressor->compress(flat_grads);
            config_.process_group->reduce(compressed.data, bucket.target_rank,
                                          distributed::ReduceOp::SUM);
            if (bucket.target_rank == config_.rank) {
                Tensor decompressed = config_.grad_compressor->decompress(compressed);
                if (decompressed.numel() == flat_grads.numel()) {
                    // Write decompressed bytes into flat_grads via the
                    // zero+add trick (no public copy_).
                    flat_grads.zero_();
                    add_(flat_grads, decompressed.contiguous().view({-1}));
                }
            }
        } else {
            config_.process_group->reduce(flat_grads, bucket.target_rank,
                                          distributed::ReduceOp::SUM);
        }
        // After this call only `target_rank` has valid data in flat_grads. Non-owners read
        // nothing from it below — they zero their per-param grads instead.
        //
        // Phase E (E3): NOT switched to reduce_scatter here -- the ParamLevel
        // bucket layout assigns each bucket entirely to one owner, which is
        // semantically `reduce`. Switching to `reduce_scatter` would require
        // restructuring buckets so each contains striped slices across all
        // ranks (the element-mode path at reduce_scatter_element_bucket
        // already does this). That layout change is deferred -- the existing
        // ParamLevel callers depend on the per-bucket ownership model.
    }

    // Owner: flat_grads contains the bucket's full reduced sum. Slice + reshape back into
    // per-param grads. Non-owner: free per-param grads since this rank is no longer
    // responsible for them in Stage 2 semantics.
    if (bucket.target_rank == config_.rank) {
        // Build local_grads. For the legacy (non-pre-staged) path each param->grad() is
        // still alive and we just rebind. For the pre-staged path the hook already cleared
        // param->grad(), so we slice flat_buffer per-param and create fresh tensors with
        // the original parameter shape.
        std::vector<Tensor> local_grads;
        local_grads.reserve(bucket.params.size());

        if (pre_staged) {
            for (size_t i = 0; i < bucket.params.size(); ++i) {
                if (i >= bucket.param_offsets_elem.size()) continue;
                int64_t off = bucket.param_offsets_elem[i];
                int64_t sz  = bucket.param_sizes_elem[i];
                Tensor flat_slice = flat_grads.slice(0, off, off + sz);
                // Reshape to the parameter's shape so set_grad gets the right geometry.
                const auto& param = bucket.params[i];
                auto shape_span = param->tensor().shape();
                std::vector<int64_t> shape(shape_span.begin(), shape_span.end());
                local_grads.push_back(flat_slice.view(shape).clone());
            }
        } else {
            for (const auto& param : bucket.params) {
                if (param->has_grad()) {
                    auto& grad_opt = param->grad();
                    if (grad_opt.has_value()) {
                        local_grads.push_back(grad_opt.value());
                    }
                }
            }
            if (!local_grads.empty()) {
                // Sizes now line up: flat_grads.numel() == sum(local_grads numel). The legacy
                // shape-mismatch crash that #9 in the review flagged is fixed here.
                unflatten_into(flat_grads, local_grads);
            }
        }

        if (!local_grads.empty()) {
            for (size_t i = 0; i < bucket.params.size() && i < local_grads.size(); ++i) {
                bucket.params[i]->set_grad(local_grads[i]);
            }
        }
    } else {
        // This rank doesn't own these parameters — free their gradients.
        for (auto& param : bucket.params) {
            if (param->has_grad()) {
                param->zero_grad();
            }
        }
    }

    // Mark bucket as processed and release the stashed gradient references so we don't
    // pin GPU memory between steps.
    {
        std::lock_guard<std::mutex> stash_lock(*bucket.mutex);
        bucket.ready = false;
        bucket.gradients_received = 0;
        for (auto& slot : bucket.gradient_buffers) {
            slot = Tensor();
        }
        // Phase B (B4): reset for next step. flat_buffer itself stays alive (persistent
        // scratch); only the "ready to use" flag flips so the next first hook of the next
        // step zero_()s the buffer before re-staging.
        bucket.flat_pre_staged = false;
    }

    // Track profiling stats
    if (profiling_enabled_) {
        auto scatter_end = std::chrono::steady_clock::now();
        auto scatter_duration = std::chrono::duration<double, std::milli>(scatter_end - scatter_start).count();

        std::lock_guard<std::mutex> prof_lock(profiling_mutex_);
        profiling_stats_.scatter_time_ms += scatter_duration;
        profiling_stats_.communication_time_ms += scatter_duration;
        profiling_stats_.num_scatters++;
        profiling_stats_.transferred_bytes += total_bytes * config_.world_size;  // Each rank sends/receives
    }
}

auto ZeROStage2Optimizer::gradient_hook(size_t bucket_idx, size_t param_idx) -> void {
    // Manual-trigger path (used by tests that call gradient_hook explicitly after backward()).
    // Pull the grad out of param->grad() and forward to the buffer-based overload.
    if (bucket_idx >= gradient_buckets_.size()) {
        return;
    }
    auto& bucket = gradient_buckets_[bucket_idx];
    if (param_idx >= bucket.params.size()) {
        return;
    }
    auto& param = bucket.params[param_idx];
    if (!param || !param->has_grad()) {
        return;
    }
    const auto& grad_opt = param->grad();
    if (!grad_opt.has_value()) {
        return;
    }
    gradient_hook(bucket_idx, param_idx, grad_opt.value());
}

auto ZeROStage2Optimizer::gradient_hook(size_t bucket_idx, size_t param_idx, const Tensor& grad) -> void {
    if (bucket_idx >= gradient_buckets_.size()) {
        return;
    }

    auto& bucket = gradient_buckets_[bucket_idx];
    if (param_idx >= bucket.params.size()) {
        return;
    }

    bool fire_reduce_scatter = false;
    {
        std::lock_guard<std::mutex> lock(*bucket.mutex);

        // Phase B (B4): pre-stage the grad bytes directly into bucket.flat_buffer at this
        // param's offset, then release the original grad. Without this, gradient_buffers[i]
        // held a refcount to param->grad() until reduce-scatter -- so each in-flight bucket
        // had two live copies of every grad in it (Variable's grad slot + bucket stash).
        // For deep models with bucket size 25 MB, several buckets-worth of duplicate grads
        // were alive concurrently during backward. Pre-staging at hook time + immediate
        // zero_grad halves transient grad memory during backward.
        //
        // Lazy-alloc the persistent flat_buffer with the proper total size on first hook.
        if (param_idx < bucket.param_offsets_elem.size()
            && param_idx < bucket.param_sizes_elem.size()
            && grad.numel() > 0) {

            int64_t total_elements = 0;
            for (int64_t n : bucket.param_sizes_elem) total_elements += n;

            DType buf_dtype = grad.dtype();
            Device buf_device = grad.device();
            if (bucket.flat_buffer.numel() != total_elements
                || bucket.flat_buffer.dtype() != buf_dtype
                || bucket.flat_buffer.device() != buf_device) {
                bucket.flat_buffer = zeros({total_elements}, buf_dtype, buf_device);
            } else if (!bucket.flat_pre_staged) {
                // First param of a new step in a previously-staged buffer: zero residue.
                bucket.flat_buffer.zero_();
            }

            int64_t off = bucket.param_offsets_elem[param_idx];
            int64_t sz  = bucket.param_sizes_elem[param_idx];
            if (grad.numel() == sz) {
                Tensor slot = bucket.flat_buffer.slice(0, off, off + sz);
                Tensor flat_grad = grad.contiguous().view({-1});
                // The slot is zero (either freshly-allocated or zero_()'d above on the
                // first param), so add_ behaves as a copy.
                add_(slot, flat_grad);
                bucket.flat_pre_staged = true;
            }

            // Release the param's grad now that we've copied its bytes -- this drops the
            // duplicate live copy. The hook's `grad` arg keeps the storage alive until
            // this function returns, after which it goes to refcount 0.
            auto& param = bucket.params[param_idx];
            if (param) {
                param->zero_grad();
            }
        } else {
            // Fallback: still stash a Tensor ref for paths where bucket geometry isn't set
            // (manual callers that haven't called create_gradient_buckets yet).
            if (bucket.gradient_buffers.size() < bucket.params.size()) {
                bucket.gradient_buffers.resize(bucket.params.size());
            }
            bucket.gradient_buffers[param_idx] = grad;
        }

        bucket.gradients_received++;

        if (bucket.gradients_received >= bucket.params.size()) {
            bucket.ready = true;
            fire_reduce_scatter = true;
        }
    }

    if (fire_reduce_scatter) {
        reduce_scatter_gradients(bucket);
    }
}

auto ZeROStage2Optimizer::is_bucket_ready(const GradientBucket& bucket) const -> bool {
    return bucket.ready && bucket.gradients_received >= bucket.params.size();
}

auto ZeROStage2Optimizer::flatten_tensors(const std::vector<Tensor>& tensors) -> Tensor {
    // Use the gradient_utils implementation
    return tenzor::optim::flatten_tensors(tensors);
}

auto ZeROStage2Optimizer::unflatten_into(
    const Tensor& flattened,
    std::vector<Tensor>& targets
) -> void {
    // Use the gradient_utils implementation
    tenzor::optim::unflatten_into(flattened, targets);
}

// =============================================================================
// Stage 2: ElementLevel gradient hook and reduce-scatter
// =============================================================================

auto ZeROStage2Optimizer::element_gradient_hook(size_t param_idx, const Tensor& grad) -> void {
    const auto& L = partition_layout_;
    const auto& e = L.params[param_idx];
    int64_t p_start = e.global_offset;
    int64_t p_end = e.global_offset + e.numel;
    DType dt = e.dtype;
    Device dev = grad.device();

    Tensor grad_flat = grad.contiguous().view({-1});
    if (grad_flat.dtype() != dt) grad_flat = grad_flat.to(dt);

    for (auto& b : element_buckets_) {
        int64_t lap_start = std::max(p_start, b.global_start);
        int64_t lap_end = std::min(p_end, b.global_end);
        if (lap_end <= lap_start) continue;

        std::lock_guard<std::mutex> bl(*b.mutex);
        if (b.flat_buffer.numel() != b.global_end - b.global_start
            || b.flat_buffer.dtype() != dt
            || b.flat_buffer.device() != dev) {
            b.flat_buffer = zeros({b.global_end - b.global_start}, dt, dev);
        }
        Tensor src = grad_flat.slice(0, lap_start - p_start, lap_end - p_start);
        Tensor dst = b.flat_buffer.slice(0, lap_start - b.global_start,
                                            lap_end - b.global_start);
        add_(dst, src);
    }

    // Mark this param's contribution to each overlapping bucket. Fire reduce_scatter
    // when a bucket has all its params' grads in.
    for (auto& b : element_buckets_) {
        bool in_bucket = std::find(b.param_indices.begin(), b.param_indices.end(),
                                    param_idx) != b.param_indices.end();
        if (!in_bucket) continue;
        std::lock_guard<std::mutex> bl(*b.mutex);
        b.hooks_received++;
        if (b.hooks_received == b.param_indices.size()) {
            reduce_scatter_element_bucket(b);
        }
    }
}

auto ZeROStage2Optimizer::reduce_scatter_element_bucket(ElementBucket& bucket) -> void {
    // Caller holds bucket.mutex.
    if (config_.world_size <= 1 || !config_.process_group) {
        // Single-rank path: nothing to scatter; the optimizer step picks up the
        // bucket's flat_buffer directly during update_local_partition_element_mode.
        return;
    }

    const int64_t bucket_n = bucket.global_end - bucket.global_start;
    const int64_t per_rank = bucket_n / config_.world_size;
    DType dt = bucket.flat_buffer.dtype();
    Device dev = bucket.flat_buffer.device();

    // Split bucket.flat_buffer into world_size contiguous chunks (slices). Pass them
    // as the input to reduce_scatter; the output is this rank's slice.
    std::vector<Tensor> chunks(config_.world_size);
    for (int r = 0; r < config_.world_size; ++r) {
        chunks[r] = bucket.flat_buffer.slice(0, r * per_rank, (r + 1) * per_rank);
    }

    Tensor output = zeros({per_rank}, dt, dev);
    config_.process_group->reduce_scatter(chunks, output);

    // Replace bucket.flat_buffer with just our slice. Frees the other ranks' slices —
    // that's the actual memory win this whole refactor exists to deliver.
    bucket.flat_buffer = output;
    bucket.hooks_received = 0;  // ready for next backward
}

auto ZeROStage2Optimizer::build_rank_grad_slice() -> Tensor {
    if (config_.partitioning_mode != PartitioningMode::ElementLevel) {
        return ZeROStage1Optimizer::build_rank_grad_slice();
    }

    // Fallback: if no bucket has stashed grads (autograd hooks didn't fire — typical
    // when callers use set_grad() directly without backward(), or when bucketing is
    // disabled), delegate to the base-class path that reads from param->grad()
    // directly. Mirrors the equivalent fallback in the legacy reduce_scatter_gradients.
    bool any_bucket_has_data = false;
    for (const auto& b : element_buckets_) {
        if (b.flat_buffer.numel() > 0) { any_bucket_has_data = true; break; }
    }
    if (!any_bucket_has_data) {
        return ZeROStage1Optimizer::build_rank_grad_slice();
    }

    const auto& L = partition_layout_;
    const int64_t rs = L.rank_starts[config_.rank];
    const int64_t re = L.rank_starts[config_.rank + 1];
    const int64_t slice_n = re - rs;
    Device dev = local_partition().device;
    DType dt = local_partition().momentum[0].dtype();

    Tensor grad_slice = zeros({slice_n}, dt, dev);
    // Each bucket's flat_buffer post-reduce_scatter is per_rank-in-bucket sized; copy
    // each bucket's contribution to the right offset in grad_slice. The bucket's slice
    // corresponds to global range [b.global_start + rank*per_rank_in_bucket,
    // b.global_start + (rank+1)*per_rank_in_bucket).
    for (auto& b : element_buckets_) {
        std::lock_guard<std::mutex> bl(*b.mutex);
        if (b.flat_buffer.numel() == 0) continue;
        const int64_t bucket_n = b.global_end - b.global_start;
        const int64_t per_rank_bucket = bucket_n / config_.world_size;
        const int64_t bucket_rank_start = b.global_start + config_.rank * per_rank_bucket;
        const int64_t bucket_rank_end = bucket_rank_start + per_rank_bucket;

        int64_t lap_start = std::max(bucket_rank_start, rs);
        int64_t lap_end = std::min(bucket_rank_end, re);
        if (lap_end <= lap_start) continue;

        Tensor src = b.flat_buffer.slice(0, lap_start - bucket_rank_start,
                                            lap_end - bucket_rank_start);
        if (src.dtype() != dt) src = src.to(dt);
        Tensor dst = grad_slice.slice(0, lap_start - rs, lap_end - rs);
        add_(dst, src);
    }
    return grad_slice;
}

// =============================================================================
// ZeRO Stage 3 Optimizer Implementation
// =============================================================================

// PrefetchScheduler must be defined before constructor (unique_ptr needs complete type)
class ZeROStage3Optimizer::PrefetchScheduler {
public:
    struct Config {
        int max_concurrent{4};
        size_t max_buffer_bytes{500 * 1024 * 1024};  // 500MB
    };

    PrefetchScheduler(const Config& config, ZeROStage3Optimizer* optimizer)
        : config_(config), optimizer_(optimizer) {}

    auto schedule_prefetch(ParameterInfo& param_state, int priority) -> void {
        std::lock_guard<std::mutex> lock(queue_mutex_);

        if (in_flight_.count(param_state.param) > 0) {
            return;
        }

        if (current_buffer_size_ + param_state.size_bytes > config_.max_buffer_bytes) {
            return;
        }

        PrefetchRequest request;
        request.param_state = &param_state;
        request.priority = priority;

        queue_.push(request);
        current_buffer_size_ += param_state.size_bytes;

        execute_pending();
    }

    auto execute_pending() -> void {
        while (in_flight_.size() < static_cast<size_t>(config_.max_concurrent) && !queue_.empty()) {
            auto request = queue_.top();
            queue_.pop();

            start_async_gather(*request.param_state);

            in_flight_.insert(request.param_state->param);
        }
    }

private:
    struct PrefetchRequest {
        ParameterInfo* param_state;
        int priority;

        bool operator<(const PrefetchRequest& other) const {
            return priority < other.priority;
        }
    };

    Config config_;
    ZeROStage3Optimizer* optimizer_;
    std::priority_queue<PrefetchRequest> queue_;
    std::unordered_set<Tensor*> in_flight_;
    size_t current_buffer_size_{0};
    std::mutex queue_mutex_;

    auto start_async_gather(ParameterInfo& param_state) -> void {
        if (param_state.is_gathered || param_state.is_prefetching) {
            return;
        }

        param_state.is_prefetching = true;

        optimizer_->gather_parameter_impl(param_state);
    }
};

ZeROStage3Optimizer::ZeROStage3Optimizer(
    std::shared_ptr<Optimizer> base_optimizer,
    const Stage3Config& config
) : ZeROStage2Optimizer(std::move(base_optimizer), config),
    stage3_config_(config),
    registered_model_(nullptr) {

    perf_stats_ = PerformanceStats{};
    prefetch_scheduler_ = nullptr;
}

ZeROStage3Optimizer::~ZeROStage3Optimizer() {
    // Unregister model if registered
    if (registered_model_) {
        unregister_model();
    }
}

auto ZeROStage3Optimizer::register_model(Module& model) -> void {
    std::lock_guard<std::mutex> lock(mutex_);

    if (registered_model_) {
        throw std::runtime_error("Model already registered. Call unregister_model() first.");
    }

    registered_model_ = &model;

    // Step 1: Partition all model parameters across ranks
    partition_model_parameters(model);

    // Step 2: Register forward/backward hooks on all modules
    register_gather_scatter_hooks(model);

    // Step 3: Build the execution-order graph used by the speculative prefetcher.
    // Stamps state.layer_index, state.prefetch_priority, and state.dependent_modules
    // for every registered parameter; pin_first_layer / pin_last_layer also fire here.
    // The legacy code declined to call this because the comment in the dead helpers
    // claimed Module didn't expose its submodules — that was stale; Module::modules()
    // exists and is what we use.
    build_execution_graph(model);
}

auto ZeROStage3Optimizer::unregister_model() -> void {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!registered_model_) {
        return;
    }

    // Phase D (D1): remove the per-leaf hooks we installed via Module API.
    // The Stage-3 forward_hooks_/backward_hooks_ vectors are private bookkeeping
    // and merely need clearing; the real hooks live in each leaf Module's
    // forward_pre_hooks_/backward_post_hooks_ map and need explicit remove_hook.
    for (auto& [mod, hid] : installed_forward_hook_ids_) {
        if (mod) mod->remove_hook(hid);
    }
    installed_forward_hook_ids_.clear();
    for (auto& [mod, hid] : installed_backward_hook_ids_) {
        if (mod) mod->remove_hook(hid);
    }
    installed_backward_hook_ids_.clear();

    // Phase E (E2): clear the global recompute hooks if we set them.
    if (stage3_config_.gradient_checkpointing_aware) {
        autograd::set_recompute_hooks(autograd::RecomputeHooks{});
    }
    recompute_gathered_.clear();

    // Clear all hooks
    forward_hooks_.clear();
    backward_hooks_.clear();

    // Clear parameter states. Also drop the LRU cache list — its pointers index into
    // param_states_ which we're tearing down.
    {
        std::lock_guard<std::mutex> ps_lock(param_states_mutex_);
        param_states_.clear();
        lru_release_order_.clear();
    }

    registered_model_ = nullptr;
}

auto ZeROStage3Optimizer::step_impl() -> void {
    std::lock_guard<std::mutex> lock(mutex_);

    // Stage 3 step algorithm:
    // 1. Gradients are already reduced-scattered via backward hooks (inherited from Stage 2)
    // 2. Fetch optimizer states from CPU if offloaded
    // 3. Update local partition of parameters (parameters remain partitioned)
    // 4. Offload states back to CPU if enabled
    // 5. NO all-gather needed - parameters stay partitioned!

    // Step 2: Fetch optimizer states from CPU if offloaded
    if ((config_.offload_to_cpu && offload_engine_) || config_.offload_to_nvme) {
        fetch_states_to_gpu();
    }

    // Step 3: Update local partition of parameters
    update_local_partition();

    // Step 4: Offload states back to CPU if enabled
    if ((config_.offload_to_cpu && offload_engine_) || config_.offload_to_nvme) {
        offload_states_to_cpu();
    }

    // Phase D (D1+D4): the Stage-1 update path REBINDS param->tensor() to a fresh
    // Tensor (target = target - lr * m_hat / v_hat allocates new storage). After
    // the step, state.local_partition still references the OLD storage and any
    // cached state.full_param view references that same old storage. Without
    // re-syncing, the next forward would gather a view onto the pre-update
    // weights and silently throw away every step's worth of training.
    //
    // Sync local_partition to the post-update tensor and invalidate the
    // gathered-buffer cache so the next forward rebuilds it from the fresh data.
    {
        std::lock_guard<std::mutex> lock(param_states_mutex_);
        for (auto& [tensor_ptr, state] : param_states_) {
            if (tensor_ptr == nullptr) continue;
            // Only sync params we actually own a slice of (multi-rank: empty for
            // params not on this rank; single-rank: always populated).
            if (state.local_partition.numel() > 0) {
                state.local_partition = *tensor_ptr;
            }
            // Drop any cached full_param view -- it references the pre-update
            // storage. Free LRU entry too so the cap accounting stays correct.
            if (state.is_gathered) {
                state.full_param = Tensor();
                state.is_gathered = false;
                state.ref_count = 0;
                lru_release_order_.remove(tensor_ptr);
            }
        }
    }

    // Note: Unlike Stage 1/2, we do NOT all-gather parameters here
    // They will be gathered on-demand during next forward pass
}

auto ZeROStage3Optimizer::zero_grad() -> void {
    // Zero gradients for local partition only
    auto& partition = local_partition();
    for (auto& param : partition.params) {
        if (param->has_grad()) {
            param->zero_grad();
        }
    }
}

auto ZeROStage3Optimizer::gather_parameter(Tensor* param) -> Tensor {
    std::lock_guard<std::mutex> lock(param_states_mutex_);

    auto it = param_states_.find(param);
    if (it == param_states_.end()) {
        throw std::runtime_error("Parameter not registered with ZeROStage3Optimizer");
    }

    auto& state = it->second;

    // Case 1: Already gathered - just increment ref count.
    if (state.is_gathered) {
        state.acquire();
        perf_stats_.prefetch_hits++;
        // If this param is in the LRU release list (i.e. previously released to refcount 0
        // but kept in the cache), pull it out — it's in use again and shouldn't be evicted
        // for the duration of the new lifetime.
        lru_release_order_.remove(param);
        return state.full_param;
    }

    // Case 2: Gathering in progress - wait for it
    // Note: For now we don't implement async gather, so this case won't occur
    if (state.is_prefetching) {
        // In a full implementation, we would wait for the async gather here
        state.is_prefetching = false;
    }

    // Case 3: Not gathered - perform synchronous gather
    perf_stats_.prefetch_misses++;
    gather_parameter_impl(state);

    // Speculative prefetch: kick off gathers for the next prefetch_depth layers right
    // now, while the user's compute on this just-gathered param hasn't started. Currently
    // synchronous (ProcessGroup has no all_gather_async); the win is converting the *next*
    // gather_parameter call from a cache miss into a cache hit (LRU cache from #12 keeps
    // the buffer live across the user's release/reacquire cycle).
    //
    // The legacy code gated this on `prefetch_scheduler_` which is permanently null —
    // dead code. We gate on the user-facing `prefetch_depth` config field instead.
    //
    // We're still holding param_states_mutex_; prefetch_next_parameters knows to find the
    // freshly-gathered layer by scanning for the highest layer_index whose is_gathered
    // flag is true. The lock_guard here protects from re-entry on the same thread, but
    // because gather_parameter_impl + the speculative loop both run inside this scope,
    // we use a raw lock-free helper that assumes caller-holds-lock.
    if (stage3_config_.prefetch_depth > 0) {
        prefetch_next_parameters_locked();
    }

    return state.full_param;
}

auto ZeROStage3Optimizer::free_gathered_parameter(Tensor* param) -> void {
    std::lock_guard<std::mutex> lock(param_states_mutex_);

    auto it = param_states_.find(param);
    if (it == param_states_.end()) {
        return;
    }

    auto& state = it->second;

    // Decrement reference count
    int remaining_refs = state.release();

    // Parameter still in use by other modules
    if (remaining_refs > 0) {
        return;
    }

    // Pinned parameters are kept gathered indefinitely and never enter the LRU cache.
    if (state.pinned_in_memory) {
        return;
    }

    // Cache the gathered buffer rather than freeing it immediately. The next gather of the
    // same parameter (typical for the next iteration's forward pass on the same module)
    // gets a refcount bump instead of a fresh all-gather. The LRU list bounds peak gathered
    // memory at max_cached_params × per-param-size; anything older falls off the front and
    // gets its full_param released back to the allocator.
    if (state.is_gathered && param != nullptr) {
        // Move-or-append-to-tail: the freshest release sits at the back.
        lru_release_order_.remove(param);
        lru_release_order_.push_back(param);

        // Phase D (D3): true LRU eviction by last_access_time, with prefetch-window
        // pinning. The legacy code evicted lru_release_order_.front() (release
        // order). With prefetch_depth >= 2, the speculative gather of layer N+2
        // could push layer N+1 out of cache before forward of N+1 ran -- 1 redundant
        // all-gather per layer per step. Now we (a) pick victims by oldest
        // last_access_time across the whole list, and (b) refuse to evict any param
        // whose layer_index is within prefetch_depth of the most recently accessed
        // layer index (= "in the active prefetch window").
        const size_t cap = static_cast<size_t>(std::max(0, stage3_config_.max_cached_params));
        // Compute the active-window pin: highest layer_index across is_gathered params.
        int active_layer = -1;
        for (auto& [_, ps] : param_states_) {
            if (ps.is_gathered && ps.layer_index > active_layer) {
                active_layer = ps.layer_index;
            }
        }
        const int pf_depth = stage3_config_.prefetch_depth;

        while (lru_release_order_.size() > cap) {
            // Find oldest-access victim that is NOT in the active prefetch window
            // (layer_index in [active_layer - pf_depth, active_layer + pf_depth]).
            Tensor* victim = nullptr;
            std::chrono::steady_clock::time_point victim_time =
                std::chrono::steady_clock::time_point::max();
            for (Tensor* candidate : lru_release_order_) {
                auto c_it = param_states_.find(candidate);
                if (c_it == param_states_.end()) continue;
                const auto& cs = c_it->second;
                if (cs.pinned_in_memory || !cs.is_gathered) continue;
                // Skip params in the active prefetch window unless we have no choice.
                if (active_layer >= 0 && pf_depth > 0
                    && cs.layer_index >= active_layer - pf_depth
                    && cs.layer_index <= active_layer + pf_depth) {
                    continue;
                }
                if (cs.last_access_time < victim_time) {
                    victim = candidate;
                    victim_time = cs.last_access_time;
                }
            }
            // Fall back to oldest-overall (incl. window) if nothing else qualified --
            // we MUST evict something to honor the cap.
            if (!victim) {
                for (Tensor* candidate : lru_release_order_) {
                    auto c_it = param_states_.find(candidate);
                    if (c_it == param_states_.end()) continue;
                    const auto& cs = c_it->second;
                    if (cs.pinned_in_memory || !cs.is_gathered) continue;
                    if (cs.last_access_time < victim_time) {
                        victim = candidate;
                        victim_time = cs.last_access_time;
                    }
                }
            }
            if (!victim) break;  // Nothing evictable left; cap can't be honored.
            lru_release_order_.remove(victim);
            auto v_it = param_states_.find(victim);
            if (v_it == param_states_.end()) continue;
            auto& v_state = v_it->second;
            // Defensive: a param can be pinned after it was already in the list. Don't
            // evict pinned buffers — they're contractually kept gathered.
            if (v_state.pinned_in_memory || !v_state.is_gathered) continue;

            v_state.full_param = Tensor();
            v_state.is_gathered = false;
            perf_stats_.current_gathered_memory -= v_state.size_bytes;
        }
    }

    // Phase D (D4): the legacy code called offload_to_cpu_async and discarded
    // the returned Tensor — which made the call a no-op (state.local_partition
    // stayed GPU-resident, defeating the offload). We now synchronously
    // offload and rebind state.local_partition to the CPU copy. The gather
    // path below (gather_parameter_impl, step_impl, fetch_states_to_gpu)
    // re-uploads to GPU when the partition is needed.
    //
    // The offload itself is synchronous-by-design here: it runs at the end
    // of optimizer_step() with no outstanding compute that could overlap it.
    // D.3: an async variant would only help if a downstream consumer
    // explicitly awaited the offload completion separately from the
    // optimizer step boundary — not the case in current training loops.
    const bool wants_offload = (config_.offload_to_cpu || stage3_config_.offload_params_to_cpu)
                            && offload_engine_;
    if (wants_offload && !state.partition_on_cpu && state.local_partition.numel() > 0) {
        try {
            state.local_partition = offload_engine_->offload_to_cpu(state.local_partition);
            state.partition_on_cpu = true;
        } catch (const std::exception& e) {
            std::cerr << "ZeROStage3Optimizer: partition CPU offload failed: "
                      << e.what() << " -- continuing with GPU-resident partition\n";
        }
    }
}

// Phase E (E2): Re-gather every currently-partitioned parameter for the
// duration of a recompute pass. Tracks the gathered set in
// recompute_gathered_ so release_recompute_gathered() can undo exactly that
// set without disturbing params that were already gathered for legitimate
// reasons (e.g. pinned first/last layer).
auto ZeROStage3Optimizer::gather_for_recompute() -> void {
    std::lock_guard<std::mutex> lock(param_states_mutex_);
    recompute_gathered_.clear();

    // Snapshot which params we'll touch under the lock, but issue gathers
    // outside since gather_parameter takes the same mutex.
    std::vector<Tensor*> to_gather;
    to_gather.reserve(param_states_.size());
    for (auto& [tensor_ptr, state] : param_states_) {
        if (tensor_ptr == nullptr) continue;
        // Skip already-gathered (pinned or in-cache) -- don't double-gather.
        if (state.is_gathered) continue;
        // Skip params we own no slice of (multi-rank: numel == 0).
        if (state.local_partition.numel() == 0) continue;
        to_gather.push_back(tensor_ptr);
    }

    for (Tensor* p : to_gather) {
        auto it = param_states_.find(p);
        if (it == param_states_.end()) continue;
        auto& state = it->second;
        try {
            gather_parameter_impl(state);
            // gather_parameter_impl sets is_gathered=true and acquire()s.
            // Replace *p with the gathered full-shape tensor so the recomputed
            // forward sees full weights.
            *p = state.full_param;
            recompute_gathered_.push_back(p);
        } catch (const std::exception& e) {
            std::cerr << "ZeROStage3Optimizer::gather_for_recompute: " << e.what() << "\n";
        }
    }
}

// Phase E (E2): Free the buffers re-gathered above and restore each Variable's
// tensor to its 1-D partition slice. Mirrors the lifecycle of backward_post_hook.
auto ZeROStage3Optimizer::release_recompute_gathered() -> void {
    std::vector<Tensor*> to_release;
    {
        std::lock_guard<std::mutex> lock(param_states_mutex_);
        to_release.swap(recompute_gathered_);
    }

    for (Tensor* p : to_release) {
        // Snapshot local_partition under lock for restore-after-free.
        Tensor slice;
        {
            std::lock_guard<std::mutex> ps_lock(param_states_mutex_);
            auto it = param_states_.find(p);
            if (it != param_states_.end() && it->second.local_partition.numel() > 0) {
                slice = it->second.local_partition;
            }
        }
        free_gathered_parameter(p);
        if (slice.numel() > 0) {
            *p = slice;
        }
    }
}

auto ZeROStage3Optimizer::prefetch_parameters(const std::vector<Tensor*>& params) -> void {
    // Check if prefetching is enabled
    if (stage3_config_.prefetch_depth <= 0 || !stage3_config_.use_async_gather) {
        return;  // Prefetching disabled
    }

    // Check if distributed is initialized
    if (config_.world_size <= 1) {
        return;  // No need to prefetch for single rank
    }

    std::lock_guard<std::mutex> lock(param_states_mutex_);

    // Track number of concurrent prefetches
    int concurrent_count = 0;
    for (const auto& [param_ptr, state] : param_states_) {
        if (state.is_prefetching) {
            concurrent_count++;
        }
    }

    // Prefetch each parameter
    for (auto* param : params) {
        // Check concurrent limit
        if (concurrent_count >= stage3_config_.max_concurrent_prefetches) {
            break;  // Too many concurrent prefetches
        }

        auto it = param_states_.find(param);
        if (it == param_states_.end()) {
            continue;  // Parameter not registered
        }

        auto& state = it->second;

        // Skip if already gathered or currently prefetching
        if (state.is_gathered || state.is_prefetching) {
            continue;
        }

        // Skip pinned parameters (already in memory)
        if (state.pinned_in_memory) {
            continue;
        }

        // Mark as prefetching
        state.is_prefetching = true;
        concurrent_count++;

        // D.3: parameter prefetch.
        //
        // The gather is performed with the ProcessGroup's all_gather_async
        // when the underlying transport supports async streams (NCCL /
        // RCCL); on Gloo the same call falls through to a sync gather (the
        // ProcessGroup base method routes both paths). Either way the
        // operation is correct — the only difference is whether it
        // overlaps with default-stream compute.
        //
        // Latency hiding via overlap of gather with the next-layer
        // forward is enabled when supports_async_stream() is true
        // (NCCL backend). Gloo callers see identical results without
        // the overlap benefit, which matches PyTorch's behaviour for
        // CPU collectives.
        try {
            gather_parameter_impl(state);
            state.is_prefetching = false;
        } catch (const std::exception& e) {
            // Prefetch failed, mark as not prefetching
            state.is_prefetching = false;
            // Don't throw - prefetch failures are not fatal
        }
    }
}

// Note: get_memory_stats() inherits from base class, no need to override

auto ZeROStage3Optimizer::state_dict() const -> std::unordered_map<std::string, Tensor> {
    std::lock_guard<std::mutex> lock(mutex_);

    std::unordered_map<std::string, Tensor> state;

    // Add partition metadata
    state["rank"] = Tensor({1}, DType::Int32, Device::cpu());
    state["rank"].fill_(config_.rank);

    state["world_size"] = Tensor({1}, DType::Int32, Device::cpu());
    state["world_size"].fill_(config_.world_size);

    state["stage"] = Tensor({1}, DType::Int32, Device::cpu());
    state["stage"].fill_(3);  // Stage 3

    // Add optimizer states for local partition
    const auto& partition = local_partition();
    for (size_t i = 0; i < partition.momentum.size(); ++i) {
        std::string key = "momentum_" + std::to_string(i);
        state[key] = partition.momentum[i];
    }

    for (size_t i = 0; i < partition.variance.size(); ++i) {
        std::string key = "variance_" + std::to_string(i);
        state[key] = partition.variance[i];
    }

    // Add parameter partition info
    size_t param_idx = 0;
    for (const auto& [param, param_state] : param_states_) {
        std::string prefix = "param_" + std::to_string(param_idx) + "_";

        state[prefix + "partition_offset"] = Tensor({1}, DType::Int64, Device::cpu());
        state[prefix + "partition_offset"].fill_(static_cast<int64_t>(param_state.partition_offset));

        state[prefix + "partition_size"] = Tensor({1}, DType::Int64, Device::cpu());
        state[prefix + "partition_size"].fill_(static_cast<int64_t>(param_state.partition_size));

        param_idx++;
    }

    return state;
}

auto ZeROStage3Optimizer::load_state_dict(const std::unordered_map<std::string, Tensor>& state) -> void {
    std::lock_guard<std::mutex> lock(mutex_);

    // Verify rank and world_size match
    if (state.count("rank")) {
        int saved_rank = state.at("rank").data<int32_t>()[0];
        if (saved_rank != config_.rank) {
            throw std::runtime_error(
                "Rank mismatch: saved=" + std::to_string(saved_rank) +
                ", current=" + std::to_string(config_.rank)
            );
        }
    }

    if (state.count("world_size")) {
        int saved_world_size = state.at("world_size").data<int32_t>()[0];
        if (saved_world_size != config_.world_size) {
            throw std::runtime_error(
                "World size mismatch: saved=" + std::to_string(saved_world_size) +
                ", current=" + std::to_string(config_.world_size)
            );
        }
    }

    // Load optimizer states
    auto& partition = local_partition();

    for (size_t i = 0; i < partition.momentum.size(); ++i) {
        std::string key = "momentum_" + std::to_string(i);
        if (state.count(key)) {
            partition.momentum[i] = state.at(key).to(partition.device);
        }
    }

    for (size_t i = 0; i < partition.variance.size(); ++i) {
        std::string key = "variance_" + std::to_string(i);
        if (state.count(key)) {
            partition.variance[i] = state.at(key).to(partition.device);
        }
    }
}

// =============================================================================
// Private: Parameter Management
// =============================================================================

auto ZeROStage3Optimizer::partition_model_parameters(Module& model) -> void {
    auto params = model.parameters();

    if (params.empty()) {
        return;
    }

    // For single-rank mode, just track parameters without modifying them
    // This allows Stage 3 optimizer to work in single-process testing mode
    if (config_.world_size == 1) {
        for (const auto& param_ptr : params) {
            Tensor& param_tensor = param_ptr->tensor();
            size_t param_size = param_tensor.numel();

            ParameterInfo state;
            state.param = &param_tensor;
            state.name = "param_" + std::to_string(param_states_.size());
            state.size_bytes = param_size * dtype_size(param_tensor.dtype());
            state.owner_rank = 0;
            state.partition_offset = 0;
            state.partition_size = param_size;

            // Store original shape
            auto shape_span = param_tensor.shape();
            state.original_shape = std::vector<int64_t>(shape_span.begin(), shape_span.end());

            // In single-rank mode, local partition is the full parameter (no flattening)
            state.local_partition = param_tensor;

            // Store state (don't modify param_tensor)
            param_states_[&param_tensor] = std::move(state);
        }
        return;
    }

    // Multi-rank mode: actually partition parameters
    // Calculate total parameter count
    size_t total_params = 0;
    for (const auto& param_ptr : params) {
        total_params += param_ptr->tensor().numel();
    }

    // Calculate partition boundaries for this rank
    size_t partition_size = (total_params + config_.world_size - 1) / config_.world_size;
    size_t partition_start = config_.rank * partition_size;
    size_t partition_end = std::min(partition_start + partition_size, total_params);

    // Partition each parameter
    size_t current_offset = 0;
    for (const auto& param_ptr : params) {
        Tensor& param_tensor = param_ptr->tensor();
        size_t param_size = param_tensor.numel();

        // Skip tiny parameters (not worth partitioning)
        if (param_size * dtype_size(param_tensor.dtype()) < stage3_config_.partition_threshold) {
            continue;
        }

        // Calculate this parameter's partition boundaries
        size_t param_start = current_offset;
        size_t param_end = current_offset + param_size;

        // Find overlap with this rank's partition
        size_t overlap_start = std::max(param_start, partition_start);
        size_t overlap_end = std::min(param_end, partition_end);

        ParameterInfo state;
        state.param = &param_tensor;
        state.name = "param_" + std::to_string(param_states_.size());
        state.size_bytes = param_size * dtype_size(param_tensor.dtype());
        state.owner_rank = config_.rank;
        state.partition_offset = overlap_start - param_start;
        state.partition_size = overlap_end - overlap_start;

        // Store original shape for reshaping after gather
        auto shape_span = param_tensor.shape();
        state.original_shape = std::vector<int64_t>(shape_span.begin(), shape_span.end());

        if (overlap_end > overlap_start) {
            // This rank owns part of this parameter.
            // CRITICAL: flatten() and slice() both return *views* sharing storage
            // with the original parameter (Tensor::flatten/slice declared at
            // include/tenzor/core/tensor.hpp:694,754; Storage is IntrusiveRefCounted
            // at include/tenzor/core/storage.hpp:32). If we keep the slice as a
            // view, the original full-shape allocation is never reclaimed -- the
            // unowned (W-1)/W of the parameter stays GPU-resident on every rank,
            // defeating the entire purpose of ZeRO-3 partitioning.
            // .clone() makes an independent contiguous allocation; the original
            // storage drops to refcount 0 once param_tensor is rebound below.
            state.local_partition = param_tensor.flatten()
                                        .slice(0, overlap_start - param_start,
                                               overlap_end - param_start)
                                        .clone();

            // Replace full parameter with partition (kept flat). After this
            // assignment the original full tensor's storage has no references
            // (the local `param_tensor` reference at this scope is overwritten,
            // and there are no other holders), so it is freed.
            param_tensor = state.local_partition;
        } else {
            // This rank owns no part of this parameter
            state.local_partition = Tensor();  // Empty
            param_tensor = Tensor();  // Free GPU memory
        }

        // Store state
        param_states_[&param_tensor] = std::move(state);

        current_offset += param_size;
    }

    // FIX (A5): Stage-1's initialize_optimizer_states (called during base-class
    // construction) sized momentum / variance / master_params against each
    // parameter's ORIGINAL full shape, because partition_model_parameters had
    // not yet run. Now that we've replaced every owned-slice param with a 1-D
    // tensor, those state tensors are the wrong shape -- update_local_partition
    // will either crash on shape-mismatch or silently mutate wrong-shaped
    // masters and then `param->tensor() = target.to(...)` would re-inflate the
    // param to the master's full shape, undoing partitioning.
    //
    // Re-allocate the states for THIS rank's Stage-1 partition.params, sized to
    // the now-current param shape. We only touch params that:
    //   (a) actually had their tensor replaced with a 1-D slice by the loop
    //       above (i.e. exist in param_states_ with non-zero partition_size), and
    //   (b) belong to this rank's Stage-1 partition.
    //
    // The interaction between Stage-1 ParamLevel partitioning and Stage-3
    // element-level slicing is tracked further in Phase D — this fix gets the
    // shapes consistent so the existing Stage-1 update path runs without
    // crashing on the slices Stage-3 produces.
    if (config_.world_size > 1) {
        auto& partition = local_partition();
        for (size_t i = 0; i < partition.params.size(); ++i) {
            const auto& var = partition.params[i];
            if (!var) continue;
            Tensor& current = var->tensor();
            // Skip params that weren't partitioned (small params below threshold,
            // or params this rank owns nothing of -- their tensor would be empty).
            if (current.numel() == 0) continue;

            const auto current_shape = current.shape();
            std::vector<int64_t> shape(current_shape.begin(), current_shape.end());
            const Device dev = current.device();

            // Resize momentum / variance to match current (sliced) shape if the
            // existing allocation is mis-sized.
            if (i < partition.momentum.size()
                && partition.momentum[i].numel() != current.numel()) {
                DType state_dtype = config_.state_dtype.value_or(current.dtype());
                partition.momentum[i] = zeros(shape, state_dtype, dev);
            }
            if (i < partition.variance.size()
                && partition.variance[i].numel() != current.numel()) {
                DType state_dtype = config_.state_dtype.value_or(current.dtype());
                partition.variance[i] = zeros(shape, state_dtype, dev);
            }

            // Master FP32: resize to slice shape (only meaningful when param
            // dtype is non-fp32, matching the Stage-1 alloc rule).
            if (config_.use_master_fp32
                && i < partition.master_params.size()
                && current.dtype() != DType::Float32) {
                if (partition.master_params[i].numel() != current.numel()) {
                    // Seed with the current sliced param values cast to fp32 --
                    // matches the Stage-1 init pattern (p.to(fp32).to(device)).
                    partition.master_params[i] =
                        current.to(DType::Float32).to(dev);
                }
            } else if (config_.use_master_fp32
                       && i < partition.master_params.size()
                       && current.dtype() == DType::Float32) {
                // Param is fp32: master is wasteful, leave it empty.
                partition.master_params[i] = Tensor();
            }
        }
    }
}

auto ZeROStage3Optimizer::register_gather_scatter_hooks(Module& model) -> void {
    // Phase D (D1): install per-leaf-module hooks into Module::forward_pre_hooks_
    // and Module::backward_post_hooks_ via the public Module API. Previously the
    // Stage-3 hooks lived in the private forward_hooks_/backward_hooks_ vectors
    // and NEVER actually fired during forward/backward -- only manual callers of
    // forward_pre_hook() ran the gather. With per-leaf hooks the gather happens
    // automatically when each submodule's forward begins, and peak gathered
    // memory drops from |model| (root-only single hook would gather everything)
    // to ~max(layer_size) * (prefetch_depth + 1).
    //
    // We walk the submodule tree recursively. A "leaf" for our purposes is a
    // module that has its OWN parameters (own_parameters() non-empty); we attach
    // hooks there. Container-only modules (Sequential, ModuleList, ModuleDict)
    // typically have no own params but contain leaves -- we recurse through them.

    auto params = model.parameters();
    if (params.empty()) {
        return;
    }

    // Recursive walker. Captures `this` for hook closures and the installed-id
    // tracking vectors.
    std::function<void(Module*)> install = [this, &install](Module* m) {
        if (!m) return;

        // Attach hooks if this module owns any parameters directly.
        auto own = m->own_parameters();
        bool has_own_params = std::any_of(own.begin(), own.end(),
                                          [](const std::shared_ptr<Variable>& v) {
                                              return v && v->tensor().numel() > 0;
                                          });
        if (has_own_params) {
            // Forward pre-hook: gather this module's own parameters before
            // forward executes. The closure captures `this` (the optimizer)
            // and the module pointer; the per-call work just delegates to
            // forward_pre_hook(module, ...).
            Module* mod_ptr = m;
            size_t fid = m->register_forward_pre_hook(
                [this, mod_ptr](Module*, const Variable&) {
                    this->forward_pre_hook(mod_ptr, {});
                });
            installed_forward_hook_ids_.emplace_back(mod_ptr, fid);

            // Backward post-hook: scatter gradients + free gathered params for
            // this module after backward executes.
            size_t bid = m->register_backward_post_hook(
                [this, mod_ptr](Module*, const Variable&, const Variable&) {
                    this->backward_post_hook(mod_ptr, {}, {});
                });
            installed_backward_hook_ids_.emplace_back(mod_ptr, bid);
        }

        // Recurse through named submodules.
        for (const auto& [name, child] : m->get_submodules()) {
            install(child.get());
        }
    };

    install(&model);

    // Phase E (E2): if the user opted in via gradient_checkpointing_aware,
    // register the recompute begin/end hooks so the autograd checkpoint
    // recompute path triggers a re-gather of partitioned params before
    // running forward_fn_ and frees them after. Without this, recompute
    // sees 1-D partition slices instead of full-shape weights and crashes
    // (or produces wrong outputs).
    if (stage3_config_.gradient_checkpointing_aware) {
        autograd::RecomputeHooks rh;
        rh.on_begin = [this](autograd::CheckpointFunction*) {
            this->gather_for_recompute();
        };
        rh.on_end = [this](autograd::CheckpointFunction*) {
            this->release_recompute_gathered();
        };
        autograd::set_recompute_hooks(std::move(rh));
    }

    // Also keep the legacy single-root entries in forward_hooks_/backward_hooks_
    // for any consumer that still iterates them (tests, profiling). Their hook_fn
    // is a no-op forwarder that doesn't double-fire because the leaf hooks above
    // already do the real work.
    ForwardPreHook root_fp{};
    root_fp.module = &model;
    root_fp.hook_id = next_hook_id_++;
    forward_hooks_.push_back(std::move(root_fp));

    BackwardPostHook root_bp{};
    root_bp.module = &model;
    root_bp.hook_id = next_hook_id_++;
    backward_hooks_.push_back(std::move(root_bp));
}

auto ZeROStage3Optimizer::gather_parameter_impl(ParameterInfo& state) -> void {
    // Single-rank gather doesn't need a process group at all — the local partition is
    // already the full parameter, we just stage it into the persistent buffer below.
    // Only multi-rank gathers actually call the collective.
    if (config_.world_size > 1 && !config_.process_group) {
        throw std::runtime_error("Process group not initialized");
    }

    // Phase D (D4): if the partition was offloaded to CPU after the previous step,
    // pull it back to GPU before the gather. The offload path
    // (free_gathered_parameter -> partition_on_cpu = true) rebinds local_partition
    // to a CPU tensor; the gather needs it on the right device. The destination
    // device is the param's original device (recorded at registration time via
    // state.param->device(); for the moment use the parameter pointer's current
    // device as a safe default since the param tensor was rebound to local_partition
    // and its device may already be CPU).
    if (state.partition_on_cpu && offload_engine_
        && state.local_partition.numel() > 0
        && state.local_partition.device().type == Device::Type::CPU) {
        try {
            // Default to CUDA(0) if we have no better hint -- partition_model_parameters
            // ran in a CUDA context originally. Long-term we should record the original
            // device on ParameterInfo at registration time.
            Device target = (state.param && state.param->device().type != Device::Type::CPU)
                              ? state.param->device()
                              : Device::cuda(0);
            state.local_partition = offload_engine_->load_to_gpu(state.local_partition, target);
            state.partition_on_cpu = false;
        } catch (const std::exception& e) {
            std::cerr << "ZeROStage3Optimizer: partition CPU->GPU reload failed: "
                      << e.what() << "\n";
        }
    }

    auto start_time = std::chrono::steady_clock::now();

    // Use original shape for full parameter (stored before flattening)
    const auto& full_shape = state.original_shape;

    // Compute total element count for the gathered result. For single-rank we still want a
    // full-shaped buffer; for multi-rank the all-gather contract is partition_size × world_size.
    int64_t total_elements = 1;
    for (int64_t d : full_shape) total_elements *= d;

    const Tensor& src = state.local_partition;
    DType buf_dtype = src.dtype();
    Device buf_device = src.device();

    // Lazy-allocate (or re-use) the gather buffer in the parameter's original shape. For
    // pinned parameters the buffer survives across gathers, so subsequent gathers skip the
    // allocation entirely. For non-pinned parameters, free_gathered_parameter() resets
    // full_param to an empty Tensor and we re-allocate here on the next gather. Either way
    // we eliminate the per-gather cat() allocation + memcpy and the reshape() of the legacy
    // path, both of which were full-parameter sized.
    bool need_alloc = state.full_param.numel() != total_elements
                   || state.full_param.dtype() != buf_dtype
                   || state.full_param.device() != buf_device;
    if (need_alloc) {
        state.full_param = empty(full_shape, buf_dtype, buf_device);
    }

    // Stage the gathered data into the persistent buffer. We zero each per-rank slot then
    // add_ the rank's chunk; with no public copy_() API this is the cheapest way to write
    // into a contiguous slice without an intermediate allocation.
    Tensor flat_view = state.full_param.view({-1});

    if (config_.world_size > 1) {
        int64_t partition_n = src.numel();
        const size_t full_bytes = static_cast<size_t>(total_elements) * dtype_size(buf_dtype);

        // Decide bulk vs chunked. Chunked only kicks in when:
        //   1. The user explicitly opted in (threshold > 0), AND
        //   2. The full gathered tensor exceeds the threshold, AND
        //   3. The chunk size is smaller than the per-rank partition (otherwise
        //      "chunked" with K=1 is just bulk with extra bookkeeping overhead).
        const size_t chunk_bytes_per_rank = stage3_config_.chunked_gather_chunk_size;
        const int64_t chunk_n = (chunk_bytes_per_rank > 0)
            ? std::max<int64_t>(1, static_cast<int64_t>(chunk_bytes_per_rank / dtype_size(buf_dtype)))
            : partition_n;
        const bool use_chunked = stage3_config_.chunked_gather_threshold > 0
                              && full_bytes > stage3_config_.chunked_gather_threshold
                              && chunk_n < partition_n;

        auto run_all_gather = [&](const Tensor& tensor_in, std::vector<Tensor>& parts_out) {
            // Route through async-on-stream when the backend supports it (NCCL/RCCL),
            // otherwise the default sync fallback. Same dispatch rule as the legacy
            // bulk path — kept identical so chunked gather inherits the same
            // overlap-with-compute properties.
            if (use_gpu_comm_ && comm_stream_) {
                config_.process_group->all_gather_async(tensor_in, parts_out, comm_stream_);
#if defined(TENZOR_USE_CUDA) || defined(TENZOR_USE_ROCM)
                cudaStreamSynchronize(static_cast<cudaStream_t>(comm_stream_));
#endif
            } else {
                config_.process_group->all_gather(tensor_in, parts_out);
            }
        };

        auto stage_chunk_into_full = [&](const std::vector<Tensor>& parts,
                                         int64_t chunk_off, int64_t chunk_len) {
            // Each rank `r` contributed `chunk_len` elements that belong at offset
            // `r * partition_n + chunk_off` in the full-parameter buffer.
            for (int rank = 0; rank < config_.world_size; ++rank) {
                int64_t off = static_cast<int64_t>(rank) * partition_n + chunk_off;
                int64_t end = std::min(off + chunk_len, total_elements);
                if (end <= off) continue;
                Tensor slot = flat_view.slice(0, off, end);
                slot.zero_();
                Tensor part_flat = parts[rank].contiguous().view({-1});
                // Guard against the uneven-split tail: this rank's chunk may be
                // shorter than `chunk_len` if `partition_n * world_size` exceeds
                // `total_elements`.
                if (part_flat.numel() != (end - off)) {
                    part_flat = part_flat.slice(0, 0, end - off);
                }
                add_(slot, part_flat);
            }
        };

        if (use_chunked) {
            // Chunked gather: K rounds of `world_size × chunk_n` element collectives,
            // staged into the persistent full_param. Peak transient memory across
            // the loop is `world_size × chunk_n × dtype_size`, independent of
            // `partition_n` — the property that lets jumbo embedding tables fit.
            Tensor src_flat = src.contiguous().view({-1});
            for (int64_t chunk_off = 0; chunk_off < partition_n; chunk_off += chunk_n) {
                int64_t chunk_end = std::min(chunk_off + chunk_n, partition_n);
                int64_t chunk_len = chunk_end - chunk_off;
                Tensor src_chunk = src_flat.slice(0, chunk_off, chunk_end);

                std::vector<Tensor> chunk_parts(config_.world_size);
                run_all_gather(src_chunk, chunk_parts);
                stage_chunk_into_full(chunk_parts, chunk_off, chunk_len);
                // chunk_parts goes out of scope at the bottom of the loop; the
                // staging output (and any pinned host buffers underneath) gets
                // freed before the next round allocates its replacement.
            }
        } else {
            // Legacy bulk path: one collective sized to the full parameter.
            std::vector<Tensor> gathered_parts(config_.world_size);
            run_all_gather(src, gathered_parts);
            stage_chunk_into_full(gathered_parts, 0, partition_n);
        }
    } else {
        // Single rank: state.local_partition is already the full parameter; alias it
        // into state.full_param (view sharing storage). The legacy code copied src
        // into a fresh buffer here, but with the Phase D D1 forward_pre_hook now
        // doing `*param = full_param` (replacing the Variable's tensor with the
        // gathered version), copying causes a divergence: the optimizer step would
        // modify the gathered COPY while leaving local_partition untouched, then the
        // backward_post_hook would restore *param to the unmodified local_partition,
        // silently throwing away every weight update. Aliasing keeps both views
        // pointing at the same storage so the update lands in local_partition too.
        // For single-rank mode the gathered shape == local_partition shape (full
        // param), so a reshape view is correct.
        if (need_alloc) {
            // Drop the freshly-allocated buffer; we'll alias instead.
            state.full_param = src.view(full_shape);
        } else {
            // Buffer was reused from a previous gather. Stage data in (write-through
            // since it shares storage with state.local_partition is what we want, but
            // we don't have copy_ public; just rebind to the alias).
            state.full_param = src.view(full_shape);
        }
        flat_view = state.full_param.view({-1});  // refresh the local handle
    }

    // Update state
    state.is_gathered = true;
    state.acquire();  // First reference

    // Update statistics
    auto end_time = std::chrono::steady_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time
    ).count();

    {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        perf_stats_.total_gathers++;
        perf_stats_.total_gather_bytes += state.size_bytes;
        perf_stats_.avg_gather_time_ms =
            (perf_stats_.avg_gather_time_ms * (perf_stats_.total_gathers - 1) +
             duration_ms) / perf_stats_.total_gathers;

        perf_stats_.current_gathered_memory += state.size_bytes;
        perf_stats_.peak_gathered_memory = std::max(
            perf_stats_.peak_gathered_memory,
            perf_stats_.current_gathered_memory
        );
    }
}

auto ZeROStage3Optimizer::forward_pre_hook(Module* module, [[maybe_unused]] const std::vector<Tensor>& inputs) -> void {
    // Find parameters for this module
    auto params = module->parameters();

    // Prefetch parameters for next modules
    if (prefetch_scheduler_) {
        prefetch_next_parameters(module);
    }

    // Gather parameters for this module
    for (const auto& param_ptr : params) {
        Tensor* param = &param_ptr->tensor();
        auto it = param_states_.find(param);
        if (it != param_states_.end()) {
            // Phase D (D1+D4): refresh state.local_partition from the Variable's
            // current tensor BEFORE gather. The Variable's tensor may have been
            // rebound externally since the last forward (e.g.
            // model.load_state_dict, manual user assignment). Without this
            // refresh, gather_parameter would alias the stale local_partition
            // and forward would see pre-load weights -- silently breaking
            // checkpoint restore.
            //
            // We detect "rebound externally" by storage identity: if the Tensor's
            // current storage handle differs from local_partition's, the param
            // was updated outside our control and we re-snapshot it.
            // Cheap: this runs on a tight inner loop; per-call cost is just a
            // couple of pointer compares. We always refresh in single-rank mode
            // (where local_partition shares storage with the param tensor) so
            // the next gather alias picks up the new data.
            {
                std::lock_guard<std::mutex> ps_lock(param_states_mutex_);
                auto& state = it->second;
                if (state.local_partition.numel() > 0
                    && state.local_partition.numel() == param->numel()
                    && state.local_partition.dtype() == param->dtype()) {
                    state.local_partition = *param;
                    // Also drop stale gathered cache; rebuild on the gather below.
                    if (state.is_gathered) {
                        state.full_param = Tensor();
                        state.is_gathered = false;
                        state.ref_count = 0;
                        lru_release_order_.remove(param);
                    }
                }
            }

            // Gather parameter (handles prefetch hits)
            Tensor full_param = gather_parameter(param);

            // Replace module's parameter with gathered version
            *param = full_param;
        }
    }
}

auto ZeROStage3Optimizer::backward_post_hook(Module* module, [[maybe_unused]] const std::vector<Tensor>& inputs, [[maybe_unused]] const std::vector<Tensor>& grad_outputs) -> void {
    // Find parameters for this module
    auto params = module->parameters();

    // Reduce-scatter gradients (inherited from Stage 2)
    for (const auto& param_ptr : params) {
        Tensor* param = &param_ptr->tensor();
        if (param_ptr->has_grad()) {
            scatter_parameter_gradient(param);
        }
    }

    // Phase D (D1 + D4): free gathered parameters AND restore each Variable's
    // tensor to its 1-D local partition slice.
    //
    // forward_pre_hook does `*param = full_param` to replace the Variable's
    // tensor with the gathered full-shape buffer for forward+backward. After
    // backward, the optimizer step expects param->tensor() to be the 1-D
    // partition slice (so update_local_partition's shape-aligned momentum/
    // variance/master arithmetic works). Restoring here keeps that contract.
    //
    // free_gathered_parameter() decrements the refcount and (if 0) optionally
    // caches the full_param in the LRU. The Variable's tensor is now back to
    // the slice; if the LRU cached the full buffer, the next forward's
    // gather_parameter() returns that cached buffer (cache hit -> no
    // re-allocation).
    for (const auto& param_ptr : params) {
        Tensor* param = &param_ptr->tensor();
        // Snapshot the local_partition BEFORE free_gathered_parameter (which
        // may re-arrange state). We only restore for params we actually own
        // (local_partition.numel() > 0). On rank that owns nothing of this
        // param, local_partition is empty -- the Variable's tensor stays as
        // whatever it was post-forward, which is fine because that rank's
        // step_impl doesn't touch it.
        Tensor slice;
        {
            std::lock_guard<std::mutex> ps_lock(param_states_mutex_);
            auto it = param_states_.find(param);
            if (it != param_states_.end() && it->second.local_partition.numel() > 0) {
                slice = it->second.local_partition;
            }
        }
        free_gathered_parameter(param);
        if (slice.numel() > 0) {
            *param = slice;
        }
    }
}

auto ZeROStage3Optimizer::scatter_parameter_gradient(Tensor* param) -> void {
    std::lock_guard<std::mutex> lock(param_states_mutex_);

    // Find the parameter state
    auto it = param_states_.find(param);
    if (it == param_states_.end()) {
        return;  // Parameter not registered
    }

    auto& state = it->second;

    // Find the corresponding Variable from parameters_ list
    std::shared_ptr<Variable> param_var = nullptr;
    for (const auto& var : parameters_) {
        if (&var->tensor() == param) {
            param_var = var;
            break;
        }
    }

    if (!param_var) {
        return;  // Variable not found
    }

    // Check if gradient exists
    if (!param_var->has_grad()) {
        return;  // No gradient to scatter
    }

    auto& grad_opt = param_var->grad();
    if (!grad_opt.has_value()) {
        return;  // Gradient not computed
    }

    Tensor full_gradient = grad_opt.value();

    // Case 1: Single rank - no communication needed, just keep gradient
    if (config_.world_size == 1) {
        // Keep the gradient as-is for single process
        return;
    }

    // Case 2: Multi-rank - perform reduce-scatter
    if (!config_.process_group) {
        // No process group available - keep full gradient (testing mode)
        return;
    }

    // Flatten the gradient tensor for reduce-scatter
    Tensor flat_grad = full_gradient.contiguous().view({-1});

    // Perform reduce-scatter operation:
    // Each rank receives only its partition of the summed gradients
    size_t total_elements = flat_grad.numel();
    size_t elements_per_rank = (total_elements + config_.world_size - 1) / config_.world_size;
    size_t local_start = config_.rank * elements_per_rank;
    size_t local_end = std::min(local_start + elements_per_rank, total_elements);
    size_t local_size = local_end - local_start;

    // Allocate buffer for local partition of gradient
    Tensor local_grad = zeros({static_cast<int64_t>(local_size)}, flat_grad.dtype(), flat_grad.device());

    // Perform proper reduce-scatter to partition gradients efficiently
    // Each rank contributes its full gradient and receives only its partition of the sum

    // Split gradient into chunks (one per rank). For ranks past the end of the unevenly-
    // partitioned grad we need a zero-element placeholder; allocate it once outside the loop
    // and share the handle across all padding ranks (Tensor is a refcounted handle, so this
    // is one zero-element allocation total instead of `world_size - active_ranks` per call).
    std::vector<Tensor> gradient_chunks;
    gradient_chunks.reserve(config_.world_size);

    Tensor empty_chunk;  // lazy: only allocate if we actually need padding
    for (int rank = 0; rank < config_.world_size; ++rank) {
        size_t rank_start = rank * elements_per_rank;
        size_t rank_end = std::min(rank_start + elements_per_rank, total_elements);

        if (rank_start < total_elements) {
            gradient_chunks.push_back(flat_grad.slice(0, rank_start, rank_end));
        } else {
            if (empty_chunk.numel() == 0) {
                empty_chunk = zeros({0}, flat_grad.dtype(), flat_grad.device());
            }
            gradient_chunks.push_back(empty_chunk);
        }
    }

    // Reduce-scatter: Each rank receives the sum of its chunk from all ranks
    config_.process_group->reduce_scatter(gradient_chunks, local_grad,
                                         distributed::ReduceOp::SUM);

    // Phase D (D2): store the local gradient slice in state.local_grad (a new
    // dedicated field), NOT in state.local_partition. The latter is the param's
    // 1-D slice -- overwriting it with the grad slice (as the legacy code did)
    // silently destroyed the parameter data after every backward.
    //
    // For master-FP32 + non-FP32 param, also produce an FP32 grad slice so the
    // Stage-1 update_local_partition's master path sees a precision-aligned
    // gradient. (The reduce was done at flat_grad.dtype() so the network-side
    // is what it always was; the upcast happens after.)
    Tensor grad_slice = local_grad;
    if (config_.use_master_fp32 && grad_slice.dtype() != DType::Float32) {
        grad_slice = grad_slice.to(DType::Float32);
    }
    state.local_grad = grad_slice;

    // Update the Variable's gradient with local partition. This frees the full
    // gradient (the previous owner) and saves memory: the full gradient was the
    // only thing holding the storage that's now `flat_grad` (a view), so when
    // set_grad rebinds, the full storage drops to refcount 0.
    param_var->set_grad(grad_slice);
}

auto ZeROStage3Optimizer::prefetch_next_parameters_locked() -> void {
    // Lock-free body — caller must already hold param_states_mutex_. Used from the
    // speculative-prefetch path inside gather_parameter() which holds the mutex itself.
    if (!registered_model_ || stage3_config_.prefetch_depth <= 0) {
        return;
    }

    int current_layer = -1;
    for (const auto& [param, state] : param_states_) {
        if (state.layer_index >= 0 && state.is_gathered) {
            current_layer = std::max(current_layer, state.layer_index);
        }
    }
    if (current_layer < 0) return;  // nothing gathered yet

    const int prefetch_depth = stage3_config_.prefetch_depth;
    const int max_concurrent = stage3_config_.max_concurrent_prefetches;

    // Walk param_states_ to gather params in (current_layer, current_layer+depth]. Stop
    // early when we've issued max_concurrent gathers this call. Cache hits (already
    // gathered, refcount-bumped via the LRU path in #12) cost effectively nothing.
    int issued = 0;
    for (auto& [param, state] : param_states_) {
        (void)param;
        if (issued >= max_concurrent) break;
        if (state.layer_index <= current_layer) continue;
        if (state.layer_index > current_layer + prefetch_depth) continue;
        if (state.is_gathered) continue;
        if (state.is_prefetching) continue;

        try {
            // D.3: ProcessGroup::all_gather_async exists (NCCL/RCCL native;
            // Gloo falls through to sync). The gather here uses
            // gather_parameter_impl which routes through the active
            // ProcessGroup, so async overlap is automatically enabled on
            // backends that support it. The cache-warmup win applies on
            // every backend regardless of async support.
            gather_parameter_impl(state);
            ++issued;
        } catch (const std::exception&) {
            // Prefetch failures are not fatal — the consumer's gather_parameter() will
            // run a real (cache-miss) gather when it actually needs the data.
            continue;
        }
    }
}

auto ZeROStage3Optimizer::prefetch_next_parameters([[maybe_unused]] Module* current_module) -> void {
    // Public-API wrapper that takes the lock. Used by external callers; the speculative
    // path inside gather_parameter() bypasses this and calls the locked helper directly.
    std::lock_guard<std::mutex> lock(param_states_mutex_);
    prefetch_next_parameters_locked();
}

auto ZeROStage3Optimizer::pin_parameter(Tensor* param) -> void {
    std::lock_guard<std::mutex> lock(param_states_mutex_);

    auto it = param_states_.find(param);
    if (it != param_states_.end()) {
        it->second.pinned_in_memory = true;
    }
}

auto ZeROStage3Optimizer::unpin_parameter(Tensor* param) -> void {
    std::lock_guard<std::mutex> lock(param_states_mutex_);

    auto it = param_states_.find(param);
    if (it != param_states_.end()) {
        it->second.pinned_in_memory = false;
    }
}

auto ZeROStage3Optimizer::get_stats() -> Stats {
    std::lock_guard<std::mutex> lock(stats_mutex_);

    Stats stats;
    stats.total_all_gather_calls = perf_stats_.total_gathers;
    stats.total_all_gather_bytes = perf_stats_.total_gather_bytes;
    stats.avg_all_gather_time_ms = perf_stats_.avg_gather_time_ms;
    stats.peak_gathered_memory_bytes = perf_stats_.peak_gathered_memory;
    stats.current_gathered_memory_bytes = perf_stats_.current_gathered_memory;

    // Calculate prefetch hit rate
    size_t total_accesses = perf_stats_.prefetch_hits + perf_stats_.prefetch_misses;
    if (total_accesses > 0) {
        stats.prefetch_hit_rate = static_cast<double>(perf_stats_.prefetch_hits) / total_accesses;
    }

    return stats;
}

auto ZeROStage3Optimizer::reset_stats() -> void {
    std::lock_guard<std::mutex> lock(stats_mutex_);

    perf_stats_ = PerformanceStats{};
}

// =============================================================================
// Additional Stage 3 Methods
// =============================================================================

auto ZeROStage3Optimizer::gather_full_state() -> std::unordered_map<std::string, Tensor> {
    // Gather full optimizer state from all ranks for checkpointing
    std::unordered_map<std::string, Tensor> full_state;

    // D.3: gather state from all ranks using collective communication.
    // Each rank's state_dict() returns its local partition; ProcessGroup's
    // all_gather concatenates them into the full state on every rank.
    auto local = state_dict();
    auto pg = config_.process_group;
    if (pg && pg->world_size() > 1) {
        for (auto& [name, local_tensor] : local) {
            std::vector<Tensor> gathered(pg->world_size());
            pg->all_gather(local_tensor, gathered);
            // Concat along dim 0 to form the full parameter state.
            Tensor full = tenzor::cat(gathered, /*dim=*/0);
            full_state.emplace(name, std::move(full));
        }
    } else {
        full_state = std::move(local);
    }

    return full_state;
}

auto ZeROStage3Optimizer::load_full_state(const std::unordered_map<std::string, Tensor>& full_state) -> void {
    // Load full state and automatically partition across ranks
    // For now, just call load_state_dict
    load_state_dict(full_state);
}

auto ZeROStage3Optimizer::gather_parameter_async(Tensor* param) -> std::shared_ptr<AsyncHandle> {
    auto handle = std::make_shared<AsyncHandle>();

    // For now, perform synchronous gather and mark as ready
    // A full implementation would use async NCCL operations
    try {
        handle->result = gather_parameter(param);
        handle->ready = true;
        handle->cv.notify_all();
    } catch (const std::exception& e) {
        handle->ready = true;  // Mark as ready even on error
        handle->cv.notify_all();
        throw;
    }

    return handle;
}

auto ZeROStage3Optimizer::wait_gather(std::shared_ptr<AsyncHandle> handle) -> Tensor {
    if (!handle) {
        throw std::runtime_error("Invalid async handle");
    }

    handle->wait();
    return handle->result;
}

auto ZeROStage3Optimizer::get_parameter_state(Tensor* param) const -> ParameterState {
    std::lock_guard<std::mutex> lock(param_states_mutex_);

    auto it = param_states_.find(param);
    if (it == param_states_.end()) {
        return ParameterState::PARTITIONED;
    }

    return it->second.state;
}

auto ZeROStage3Optimizer::is_parameter_gathered(Tensor* param) const -> bool {
    std::lock_guard<std::mutex> lock(param_states_mutex_);

    auto it = param_states_.find(param);
    if (it == param_states_.end()) {
        return false;
    }

    return it->second.is_gathered;
}

auto ZeROStage3Optimizer::is_parameter_pinned(Tensor* param) const -> bool {
    std::lock_guard<std::mutex> lock(param_states_mutex_);

    auto it = param_states_.find(param);
    if (it == param_states_.end()) {
        return false;
    }

    return it->second.pinned_in_memory;
}

auto ZeROStage3Optimizer::get_prefetch_stats() const -> PrefetchStats {
    std::lock_guard<std::mutex> lock(stats_mutex_);

    PrefetchStats stats;
    stats.prefetch_hits = perf_stats_.prefetch_hits;
    stats.prefetch_misses = perf_stats_.prefetch_misses;

    size_t total = stats.prefetch_hits + stats.prefetch_misses;
    if (total > 0) {
        stats.hit_rate = static_cast<double>(stats.prefetch_hits) / total;
    }

    return stats;
}

auto ZeROStage3Optimizer::build_execution_graph(Module& model) -> void {
    // Build execution graph for prefetch scheduling by analyzing parameter usage order
    // This helps the prefetch scheduler predict which parameters will be needed next

    std::lock_guard<std::mutex> lock(param_states_mutex_);

    // Get all model parameters in their declaration order
    auto params = model.parameters();
    if (params.empty()) {
        return;
    }

    // Assign layer indices to parameters based on their order in the model
    // This provides a simple execution order approximation
    int layer_index = 0;
    for (const auto& param_var : params) {
        Tensor* param = &param_var->tensor();

        auto it = param_states_.find(param);
        if (it != param_states_.end()) {
            auto& state = it->second;

            // Assign layer index for prefetch priority
            state.layer_index = layer_index;

            // Set prefetch priority (earlier layers = higher priority)
            // Priority decreases as layer index increases
            state.prefetch_priority = 1000 - layer_index;

            layer_index++;
        }
    }

    // Group parameters by layer index to build execution order
    // This enables prefetching of upcoming layers during forward/backward passes
    std::map<int, std::vector<Tensor*>> layer_params;

    for (auto& [param, state] : param_states_) {
        if (state.layer_index >= 0) {
            layer_params[state.layer_index].push_back(param);
        }
    }

    // Build prefetch hints for each layer
    // For each parameter, identify which parameters likely come next
    for (auto& [param, state] : param_states_) {
        if (state.layer_index < 0) {
            continue;
        }

        // Find parameters in the next prefetch_depth layers
        int current_layer = state.layer_index;
        int prefetch_depth = stage3_config_.prefetch_depth;

        std::vector<Tensor*> next_params;
        for (int i = 1; i <= prefetch_depth; ++i) {
            int next_layer = current_layer + i;

            auto layer_it = layer_params.find(next_layer);
            if (layer_it != layer_params.end()) {
                // Add all parameters from this layer to prefetch list
                for (auto* next_param : layer_it->second) {
                    next_params.push_back(next_param);
                }
            }
        }

        // Store dependency information for this parameter
        // This allows prefetch_next_parameters() to know what to prefetch
        state.dependent_modules.clear();
        for (int i = 1; i <= prefetch_depth; ++i) {
            int next_layer = current_layer + i;
            if (layer_params.find(next_layer) != layer_params.end()) {
                state.dependent_modules.push_back(next_layer);
            }
        }
    }

    // Pin first and last layer parameters if configured
    if (stage3_config_.pin_first_layer && !layer_params.empty()) {
        auto first_layer_it = layer_params.begin();
        for (auto* param : first_layer_it->second) {
            auto it = param_states_.find(param);
            if (it != param_states_.end()) {
                it->second.pinned_in_memory = true;
            }
        }
    }

    if (stage3_config_.pin_last_layer && !layer_params.empty()) {
        auto last_layer_it = layer_params.rbegin();
        for (auto* param : last_layer_it->second) {
            auto it = param_states_.find(param);
            if (it != param_states_.end()) {
                it->second.pinned_in_memory = true;
            }
        }
    }
}

auto ZeROStage3Optimizer::should_partition_parameter(const Tensor& param) const -> bool {
    // Check if parameter is large enough to partition
    size_t param_bytes = param.numel() * dtype_size(param.dtype());
    return param_bytes >= stage3_config_.partition_threshold;
}

auto ZeROStage3Optimizer::free_gathered_parameter_impl([[maybe_unused]] ParameterInfo& state) -> void {
    // Internal implementation for freeing gathered parameters
    // This is already handled in free_gathered_parameter()
}

auto ZeROStage3Optimizer::flatten_tensors(const std::vector<Tensor>& tensors) -> Tensor {
    // Use the gradient_utils implementation
    return tenzor::optim::flatten_tensors(tensors);
}

auto ZeROStage3Optimizer::unflatten_into(const Tensor& flattened, std::vector<Tensor>& targets) -> void {
    // Use the gradient_utils implementation
    tenzor::optim::unflatten_into(flattened, targets);
}

// =============================================================================
// Profiling API Implementation (ZeROStage1Optimizer)
// =============================================================================

auto ZeROStage1Optimizer::ProfilingStats::print_summary() const -> void {
    std::cout << "\n=== ZeRO Optimizer Profiling Summary ===\n";
    std::cout << "Steps: " << num_steps << "\n";
    std::cout << "\nTiming Statistics (milliseconds):\n";
    std::cout << "  Total Step Time:       " << std::fixed << std::setprecision(2) << total_step_time_ms << " ms\n";
    std::cout << "  Avg Step Time:         " << avg_step_time_ms << " ms\n";
    std::cout << "  Communication Time:    " << communication_time_ms << " ms\n";
    std::cout << "  Compute Time:          " << compute_time_ms << " ms\n";
    std::cout << "  All-Reduce Time:       " << all_reduce_time_ms << " ms (avg: " << avg_all_reduce_time_ms << " ms)\n";
    std::cout << "  All-Gather Time:       " << all_gather_time_ms << " ms (avg: " << avg_gather_time_ms << " ms)\n";
    std::cout << "  Offload Time:          " << offload_time_ms << " ms\n";

    std::cout << "\nMemory Statistics:\n";
    std::cout << "  Peak Memory:           " << (peak_memory_bytes / (1024.0 * 1024.0)) << " MB\n";
    std::cout << "  Current Memory:        " << (current_memory_bytes / (1024.0 * 1024.0)) << " MB\n";
    std::cout << "  Transferred Bytes:     " << (transferred_bytes / (1024.0 * 1024.0)) << " MB\n";
    std::cout << "  Offloaded Bytes:       " << (offloaded_bytes / (1024.0 * 1024.0)) << " MB\n";

    std::cout << "\nOperation Counts:\n";
    std::cout << "  All-Reduce Ops:        " << num_all_reduces << "\n";
    std::cout << "  All-Gather Ops:        " << num_all_gathers << "\n";
    std::cout << "  Offload Ops:           " << num_offloads << "\n";

    std::cout << "\nPerformance Metrics:\n";
    std::cout << "  Comm/Compute Overlap:  " << std::fixed << std::setprecision(1)
              << (comm_compute_overlap_ratio * 100.0) << "%\n";
    std::cout << "  Effective Bandwidth:   " << effective_bandwidth_mbps << " MB/s\n";
    std::cout << "=========================================\n\n";
}

auto ZeROStage1Optimizer::ProfilingStats::to_string() const -> std::string {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2);
    oss << "ZeRO Profiling: " << num_steps << " steps, "
        << avg_step_time_ms << " ms/step, "
        << (comm_compute_overlap_ratio * 100.0) << "% overlap, "
        << effective_bandwidth_mbps << " MB/s bandwidth";
    return oss.str();
}

auto ZeROStage1Optimizer::get_profiling_stats() const -> ProfilingStats {
    std::lock_guard<std::mutex> lock(profiling_mutex_);

    // Get current memory usage
    ProfilingStats stats = profiling_stats_;
    auto mem_stats = get_memory_stats();
    stats.current_memory_bytes = mem_stats.gpu_optimizer_memory + mem_stats.cpu_optimizer_memory;
    stats.peak_memory_bytes = std::max(stats.peak_memory_bytes, stats.current_memory_bytes);

    return stats;
}

auto ZeROStage1Optimizer::reset_profiling_stats() -> void {
    std::lock_guard<std::mutex> lock(profiling_mutex_);
    profiling_stats_ = ProfilingStats{};
}

auto ZeROStage1Optimizer::enable_profiling(bool enabled) -> void {
    profiling_enabled_ = enabled;
    if (!enabled) {
        reset_profiling_stats();
    }
}

// =============================================================================
// Phase 7: Advanced Optimizations Implementation
// =============================================================================

auto ZeROStage3Optimizer::update_prefetch_depth() -> void {
    if (!stage3_config_.enable_adaptive_prefetch) {
        return;  // Adaptive prefetch disabled
    }

    std::lock_guard<std::mutex> lock(adaptive_mutex_);

    // Calculate optimal prefetch depth based on current metrics
    int optimal_depth = calculate_optimal_prefetch_depth();

    // Update if different from current
    if (optimal_depth != adaptive_metrics_.current_prefetch_depth) {
        adaptive_metrics_.current_prefetch_depth = optimal_depth;
        stage3_config_.prefetch_depth = optimal_depth;

        // Track improvement/degradation
        if (optimal_depth > stage3_config_.prefetch_depth) {
            adaptive_metrics_.consecutive_improvements++;
            adaptive_metrics_.consecutive_degradations = 0;
        } else if (optimal_depth < stage3_config_.prefetch_depth) {
            adaptive_metrics_.consecutive_degradations++;
            adaptive_metrics_.consecutive_improvements = 0;
        }
    }
}

auto ZeROStage3Optimizer::calculate_optimal_prefetch_depth() -> int {
    std::lock_guard<std::mutex> lock(adaptive_mutex_);

    // If insufficient data, return current depth
    if (adaptive_metrics_.recent_gather_times_ms.size() < 3) {
        return adaptive_metrics_.current_prefetch_depth;
    }

    // Calculate average gather and compute times
    double avg_gather_time = 0.0;
    double avg_compute_time = 0.0;

    size_t window = std::min(adaptive_metrics_.recent_gather_times_ms.size(),
                             stage3_config_.prefetch_window_size);

    for (size_t i = 0; i < window; ++i) {
        avg_gather_time += adaptive_metrics_.recent_gather_times_ms[i];
        if (i < adaptive_metrics_.recent_compute_times_ms.size()) {
            avg_compute_time += adaptive_metrics_.recent_compute_times_ms[i];
        }
    }

    avg_gather_time /= window;
    if (!adaptive_metrics_.recent_compute_times_ms.empty()) {
        size_t compute_window = std::min(adaptive_metrics_.recent_compute_times_ms.size(), window);
        avg_compute_time /= compute_window;
    }

    // Calculate actual overlap ratio
    double actual_overlap = 0.0;
    if (avg_compute_time > 0.0) {
        actual_overlap = std::min(1.0, avg_gather_time / avg_compute_time);
    }
    adaptive_metrics_.actual_overlap_ratio = actual_overlap;

    // Determine if we need more or less prefetch depth
    int current_depth = adaptive_metrics_.current_prefetch_depth;
    int new_depth = current_depth;

    // If actual overlap is below target, increase prefetch depth
    if (actual_overlap < stage3_config_.target_overlap_ratio - 0.1) {
        // Need more prefetching to hide latency
        new_depth = std::min(current_depth + 1, stage3_config_.max_prefetch_depth);
    }
    // If actual overlap is well above target, decrease prefetch depth to save memory
    else if (actual_overlap > stage3_config_.target_overlap_ratio + 0.15) {
        // Can reduce prefetch depth to save memory
        new_depth = std::max(current_depth - 1, stage3_config_.min_prefetch_depth);
    }

    // Check memory pressure - if high, reduce prefetch depth
    double memory_pressure = check_memory_pressure();
    if (memory_pressure > 0.9 && new_depth > stage3_config_.min_prefetch_depth) {
        new_depth = std::max(stage3_config_.min_prefetch_depth, new_depth - 1);
    }

    return new_depth;
}

auto ZeROStage3Optimizer::adjust_bucket_size() -> void {
    if (!stage3_config_.enable_dynamic_bucket_sizing) {
        return;  // Dynamic bucket sizing disabled
    }

    std::lock_guard<std::mutex> lock(adaptive_mutex_);

    // Calculate optimal bucket size
    size_t optimal_size = calculate_optimal_bucket_size();

    // Update if significantly different from current
    size_t current = adaptive_metrics_.current_bucket_size;
    double change_ratio = static_cast<double>(optimal_size) / current;

    // Only adjust if change is significant (>10% difference)
    if (change_ratio > 1.1 || change_ratio < 0.9) {
        adaptive_metrics_.current_bucket_size = optimal_size;
        stage3_config_.prefetch_bucket_size = optimal_size;
    }
}

auto ZeROStage3Optimizer::calculate_optimal_bucket_size() -> size_t {
    std::lock_guard<std::mutex> lock(adaptive_mutex_);

    // If insufficient data, return current size
    if (adaptive_metrics_.recent_comm_efficiency.empty()) {
        return adaptive_metrics_.current_bucket_size;
    }

    // Calculate average communication efficiency
    double avg_efficiency = 0.0;
    size_t window = std::min(adaptive_metrics_.recent_comm_efficiency.size(),
                             stage3_config_.prefetch_window_size);

    for (size_t i = 0; i < window; ++i) {
        avg_efficiency += adaptive_metrics_.recent_comm_efficiency[i];
    }
    avg_efficiency /= window;

    size_t current_size = adaptive_metrics_.current_bucket_size;
    size_t new_size = current_size;

    // If efficiency is below target, increase bucket size
    // Larger buckets = fewer messages = better amortization of latency
    if (avg_efficiency < stage3_config_.target_comm_efficiency - 0.05) {
        // Increase bucket size by 25%
        new_size = static_cast<size_t>(current_size * 1.25);
        new_size = std::min(new_size, stage3_config_.max_bucket_size);
    }
    // If efficiency is well above target, decrease bucket size to reduce memory
    else if (avg_efficiency > stage3_config_.target_comm_efficiency + 0.05) {
        // Decrease bucket size by 20%
        new_size = static_cast<size_t>(current_size * 0.8);
        new_size = std::max(new_size, stage3_config_.min_bucket_size);
    }

    // Consider memory pressure
    double memory_pressure = check_memory_pressure();
    if (memory_pressure > 0.85) {
        // High memory pressure - reduce bucket size
        new_size = std::max(stage3_config_.min_bucket_size,
                           static_cast<size_t>(new_size * 0.8));
    }

    return new_size;
}

auto ZeROStage3Optimizer::check_memory_pressure() -> double {
    std::lock_guard<std::mutex> lock(adaptive_mutex_);

    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        now - adaptive_metrics_.last_memory_check
    ).count();

    // Only check memory pressure periodically to avoid overhead
    if (elapsed < stage3_config_.memory_monitor_interval_ms &&
        adaptive_metrics_.current_memory_pressure > 0.0) {
        return adaptive_metrics_.current_memory_pressure;
    }

    adaptive_metrics_.last_memory_check = now;

    // Get GPU memory stats
    // Note: In a real implementation, this would query CUDA runtime for actual memory usage
    // Calculate memory pressure based on tracked usage and configured limits
    // Memory limits are configured via AdaptiveOffloadConfig or MemoryManager::Config
    // For production use, set gpu_memory_limit to match actual device capacity

    size_t total_gpu_memory = 0;
    size_t used_gpu_memory = perf_stats_.current_gathered_memory;

    Device param_device = !parameters_.empty() ? parameters_[0]->tensor().device() : Device::cpu();
    if (param_device.type == Device::Type::CUDA) {
        // Phase C (C7): query the actual device capacity instead of hardcoding 16 GB.
        // On an 80 GB H100 the old 16 GB constant fired the 0.85 threshold at 16% real
        // utilisation; on a 12 GB consumer card it never fired. Now we use cudaMemGetInfo
        // (also covered by hipMemGetInfo via the HIP shim) to get the real (free, total)
        // pair. Falls back to 16 GB only when the runtime call is unavailable or fails.
        size_t free_bytes = 0;
        size_t total_bytes = 0;
        bool got_real_total = false;
#if defined(TENZOR_USE_CUDA) || defined(TENZOR_USE_ROCM)
        cudaError_t err = cudaMemGetInfo(&free_bytes, &total_bytes);
        if (err == cudaSuccess && total_bytes > 0) {
            total_gpu_memory = total_bytes;
            // Use the runtime-reported free/used split as ground truth instead of our
            // tracked counters when available -- catches allocations from other parts
            // of the program (autograd cache, comm buffers, etc.) the optimizer doesn't
            // see.
            used_gpu_memory = total_bytes - free_bytes;
            got_real_total = true;
        }
#endif
        if (!got_real_total) {
            total_gpu_memory = 16ULL * 1024 * 1024 * 1024;  // 16 GB fallback
            // Track parameter memory ourselves since we don't have device-truth.
            for (const auto& param : parameters_) {
                const auto& tensor = param->tensor();
                if (tensor.device().type == Device::Type::CUDA) {
                    used_gpu_memory += tensor.numel() * dtype_size(tensor.dtype());
                }
            }
        }
    } else {
        // CPU mode - no memory pressure
        adaptive_metrics_.current_memory_pressure = 0.0;
        return 0.0;
    }

    // Calculate pressure ratio
    double pressure = total_gpu_memory > 0 ?
        static_cast<double>(used_gpu_memory) / total_gpu_memory : 0.0;

    adaptive_metrics_.current_memory_pressure = pressure;
    return pressure;
}

auto ZeROStage3Optimizer::should_offload_parameter(Tensor* param) -> bool {
    if (!stage3_config_.enable_adaptive_offload) {
        return false;  // Adaptive offload disabled
    }

    std::lock_guard<std::mutex> lock(adaptive_mutex_);

    // Check memory pressure
    double pressure = check_memory_pressure();

    // If memory pressure is below threshold, don't offload
    if (pressure < stage3_config_.memory_pressure_threshold) {
        return false;
    }

    // Find parameter state
    auto it = param_states_.find(param);
    if (it == param_states_.end()) {
        return false;  // Parameter not registered
    }

    const auto& state = it->second;

    // Don't offload pinned parameters
    if (state.pinned_in_memory) {
        return false;
    }

    // Don't offload if parameter is currently in use
    if (state.ref_count > 0) {
        return false;
    }

    // Don't offload if parameter is small (overhead not worth it)
    if (state.size_bytes < config_.cpu_offload_threshold) {
        return false;
    }

    // Apply hysteresis - check if we recently made an offload decision
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        now - adaptive_metrics_.last_offload_decision
    ).count();

    // If we just made a decision recently, don't thrash
    if (elapsed < 500) {  // 500ms cooldown
        return false;
    }

    // Check if current memory usage is significantly above last offload threshold
    size_t current_memory = perf_stats_.current_gathered_memory;
    if (current_memory > adaptive_metrics_.last_offload_memory_threshold + stage3_config_.offload_hysteresis) {
        adaptive_metrics_.last_offload_decision = now;
        adaptive_metrics_.last_offload_memory_threshold = current_memory;
        return true;
    }

    return false;
}

auto ZeROStage3Optimizer::adaptive_offload_decision() -> void {
    if (!stage3_config_.enable_adaptive_offload || !offload_engine_) {
        return;  // Adaptive offload disabled or no offload engine
    }

    std::lock_guard<std::mutex> lock(param_states_mutex_);

    // Check memory pressure
    double pressure = check_memory_pressure();

    // If pressure is low, we can prefetch more aggressively
    if (pressure < stage3_config_.memory_pressure_threshold - 0.1) {
        // Memory pressure is acceptable - can keep parameters on GPU
        return;
    }

    // High memory pressure - selectively offload gathered parameters
    std::vector<Tensor*> candidates_to_offload;

    for (auto& [param, state] : param_states_) {
        if (should_offload_parameter(param)) {
            candidates_to_offload.push_back(param);
        }
    }

    // Sort candidates by LRU (least recently used first)
    std::sort(candidates_to_offload.begin(), candidates_to_offload.end(),
        [this](Tensor* a, Tensor* b) {
            auto it_a = param_states_.find(a);
            auto it_b = param_states_.find(b);
            if (it_a == param_states_.end() || it_b == param_states_.end()) {
                return false;
            }
            return it_a->second.age_ms() > it_b->second.age_ms();
        });

    // Offload parameters until pressure is below threshold
    size_t offloaded_bytes = 0;
    size_t target_bytes = static_cast<size_t>(
        perf_stats_.current_gathered_memory * 0.2  // Offload 20% of current memory
    );

    for (auto* param : candidates_to_offload) {
        if (offloaded_bytes >= target_bytes) {
            break;  // Offloaded enough
        }

        auto it = param_states_.find(param);
        if (it == param_states_.end()) {
            continue;
        }

        auto& state = it->second;

        // Offload the gathered parameter if available
        if (state.is_gathered && !state.gathered_on_cpu) {
            offload_engine_->offload_to_cpu_async(state.full_param);
            state.gathered_on_cpu = true;
            offloaded_bytes += state.size_bytes;

            // Update statistics
            perf_stats_.current_gathered_memory -= state.size_bytes;
        }

        // Offload the local partition if not already on CPU
        if (!state.partition_on_cpu && state.local_partition.numel() > 0) {
            offload_engine_->offload_to_cpu_async(state.local_partition);
            state.partition_on_cpu = true;
            offloaded_bytes += state.partition_size;
        }
    }
}

} // namespace optim
} // namespace tenzor
