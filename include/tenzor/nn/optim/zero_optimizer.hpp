/**
 * @file zero_optimizer.hpp
 * @brief ZeRO Stage 1 Optimizer with state partitioning and CPU offload
 *
 * Implements DeepSpeed ZeRO Stage 1: Optimizer State Partitioning
 * across distributed ranks for memory-efficient training.
 *
 * @see https://arxiv.org/abs/1910.02054
 */

#pragma once

#include "optimizer.hpp"
#include "../../distributed/distributed.hpp"
#include "../../distributed/gradient_compression.hpp"
#include "../../core/offload_engine.hpp"
#include "../../nn/module.hpp"
#include <memory>
#include <vector>
#include <map>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <list>
#include <optional>
#include <chrono>
#include <unordered_set>
#include <queue>
#include <deque>
#include <functional>

namespace tenzor {

// Forward declare Module to avoid namespace issues
namespace nn {
    class Module;
}

namespace optim {

// Bring Module into this namespace for convenience
using nn::Module;

/** Partitioning strategy for optimizer states and gradients.
 *
 *  - ParamLevel (default): each rank owns a set of WHOLE parameters. Optimizer state
 *    for those parameters lives only on the owner rank; gradients are reduced to the
 *    owner. Owner-rank peak grad memory == bucket size. Legacy behaviour, preserved
 *    for checkpoint compatibility.
 *
 *  - ElementLevel: every parameter is logically split into world_size contiguous
 *    element slices (the same partitioning Stage 3 uses for its on-the-fly gather).
 *    Each rank stores momentum / variance / master only for its slice; gradients are
 *    reduce-scattered so each rank receives bucket_size / world_size bytes regardless
 *    of which "owner" the bucket nominally belongs to. Per-rank peak grad memory ==
 *    bucket_size / world_size. Required to make a true reduce_scatter give memory
 *    savings — without element-level partition the owner still needs the whole bucket
 *    to run the optimizer step.
 *
 *  Switching modes invalidates checkpoints saved in the other mode (the on-disk
 *  layout differs: ParamLevel saves whole-tensor state; ElementLevel saves per-slice
 *  state). The mode is recorded in checkpoint metadata so cross-mode loads fail with
 *  a clear error rather than silently producing garbage.
 */
enum class PartitioningMode {
    ParamLevel,
    ElementLevel,
};

/**
 * @brief Configuration for ZeRO Stage 1 Optimizer
 */
struct ZeROStage1Config {
    int world_size{1};                      ///< Number of distributed ranks
    int rank{0};                            ///< Current rank ID
    bool offload_to_cpu{false};             ///< Offload optimizer states to CPU
    size_t cpu_offload_threshold{1024};     ///< Min bytes to offload (default: 1KB)
    bool overlap_comm{true};                ///< Overlap communication with computation
    bool pin_memory{true};                  ///< Use pinned memory for transfers
    std::shared_ptr<distributed::ProcessGroup> process_group{nullptr}; ///< Communication group

    // ========================================================================
    // Mixed-precision optimizer state (DeepSpeed-style master-fp32 split)
    // ========================================================================

    /** Keep an fp32 master copy of each partitioned parameter.
     *
     *  When training with fp16 or bfloat16 parameters, the optimizer step needs
     *  fp32 precision to avoid silent gradient underflow / cancellation in the
     *  Adam moment updates and bias-correction division. The standard fix
     *  (DeepSpeed, Megatron, FSDP) is to keep an fp32 "master" copy that the
     *  optimizer reads + writes; after each step the master is downcast and
     *  copied back into the user-visible fp16/bf16 parameter tensor.
     *
     *  Memory cost per parameter (per rank, with world_size N, in bytes per
     *  fp16 element):
     *    - off:  fp16 mom + fp16 var = 2 + 2 = 4    (silently lossy)
     *    - on:   fp32 master + fp32 mom + fp32 var = 4 + 4 + 4 = 12
     *  Stage 1+ partitions all of these by N, so the on-rank cost is 12/N.
     *
     *  Default off — only enable if your training dtype is not fp32.
     */
    bool use_master_fp32{false};

    /** Dtype for momentum / variance buffers.
     *
     *  - std::nullopt (default): match the parameter dtype. This is the legacy
     *    behaviour and is correct when training in fp32 / fp64.
     *  - DType::Float32: keep optimizer state in fp32 regardless of param dtype.
     *    This is the right setting for fp16/bf16 training even if you don't
     *    enable use_master_fp32 (it fixes the moment-update precision bug —
     *    see feedback_float32_accum_bug.md — without doubling the master copy).
     */
    std::optional<DType> state_dtype{};

    /** Quantize CPU-offloaded momentum/variance to int8 (per-tensor dynamic range).
     *
     *  Cuts CPU optimizer-state memory by ~4× when offload_to_cpu is on. Only takes
     *  effect for fp32 states — fp16/bf16 states are skipped (the precision loss to
     *  int8 from those starting points is too large to be worth the savings).
     *
     *  Mechanism: on offload, scale = max(|t|)/127 and q = clamp(round(t/scale),
     *  -128, 127) cast to int8. On fetch, t = q.to(orig_dtype) * scale. Per-tensor
     *  scaling (not block-wise like bitsandbytes), so dynamic-range outliers cost
     *  some precision in the small values — fine for momentum/variance in typical
     *  transformer training, may matter for models very sensitive to small-grad
     *  accumulators. Default off.
     *
     *  Has no effect when offload_to_cpu is false.
     */
    bool quantize_offloaded_states_int8{false};

    /** Offload optimizer states to local NVMe / SSD storage instead of host RAM
     *  (DeepSpeed-Infinity-style). Use this when the partitioned optimizer states
     *  for your model don't fit in CPU memory either — common at the multi-billion-
     *  parameter scale on a single host.
     *
     *  Mutually exclusive with offload_to_cpu in this implementation: enabling NVMe
     *  routes states GPU↔NVMe with a transient host buffer for staging the IO. The
     *  pinned-memory engine used by the CPU path is bypassed.
     *
     *  Storage format: one binary blob file per state tensor, plus one per scale
     *  tensor when quantize_offloaded_states_int8 is also set. File names are
     *  deterministic (rank + param index + state name) so restarts can reuse them.
     *
     *  No effect when nvme_path is empty.
     */
    bool offload_to_nvme{false};

    /** Directory where NVMe offload files live. Created if missing. Cleared on
     *  optimizer destruction. Defaults to <std::filesystem::temp_directory_path>/
     *  tenzor_zero_offload when empty.
     */
    std::string nvme_path{};

    /** Optional gradient compressor applied around the all-reduce in step().
     *
     *  When set, ZeRO Stage 1 compresses each parameter's gradient before the
     *  collective and decompresses after — directly cutting comm bytes by the
     *  compressor's compression_ratio. The intended use case is fp16 or bf16
     *  compression of fp32 gradients (2× bandwidth) on bandwidth-bound multi-
     *  GPU training: the compressor is *linear* (cast → reduce → cast back),
     *  so naive AVG reduction composes correctly.
     *
     *  Note: non-linear compressors (e.g. TopK with sparse representation) do
     *  not compose with the AVG reduction this path uses — averaging two
     *  different rank's TopK selections yields garbage. Those compressors
     *  need error-feedback all-reduce, which is not implemented here. Stick
     *  to FP16Compressor / BFloat16Compressor with this hook.
     *
     *  Default off (nullptr) preserves the legacy uncompressed behaviour.
     */
    std::shared_ptr<distributed::GradientCompressor> grad_compressor{nullptr};

    /** Optional pre-built OffloadEngine to adopt instead of constructing a private one.
     *
     *  Wire this in when another subsystem in the same process — typically activation
     *  offload via `OffloadContext` — already has an `OffloadEngine` running. Sharing
     *  the engine means sharing its underlying `TransferEngine`, which means a single
     *  pinned-host buffer pool serves both kinds of offload instead of two pools each
     *  holding ~1–2 GB of pinned RAM hostage. The mechanics are the same as the
     *  `Config::shared_transfer_engine` knob on `OffloadEngine` itself (see
     *  include/tenzor/core/offload_engine.hpp:46-56) — this is the next layer up so
     *  callers don't have to construct the OffloadEngine twice.
     *
     *  Has no effect when `offload_to_cpu` is false: there is no offload work to route.
     *
     *  Default `nullptr` preserves the legacy behaviour of constructing a private 1 GB
     *  pinned pool inside `initialize_offload_engine()`.
     */
    std::shared_ptr<core::OffloadEngine> shared_offload_engine{nullptr};

    /** Use size-aware (greedy LPT) partitioning instead of the legacy contiguous
     *  index-slice when assigning parameters to ranks.
     *
     *  The legacy partitioner splits `parameters_` by index into `world_size`
     *  contiguous chunks. That is fine when all parameters have similar size, but
     *  produces severe per-rank memory imbalance for models with one (or a few)
     *  huge tensors among many small ones — typically: a model with one large
     *  embedding/lm_head + many small layernorms/biases. The rank that ends up
     *  owning the huge tensor pays a multi-GB tax in optimizer-state memory while
     *  peer ranks sit nearly empty. Since per-rank usable GPU memory is the
     *  *minimum* across ranks, this directly caps your usable model size.
     *
     *  When set, parameters are sorted by size descending and each is placed onto
     *  the currently least-loaded rank — a "Longest Processing Time first" (LPT)
     *  greedy schedule, provably within 4/3 of optimal makespan.
     *
     *  Default off — preserves the partition assignment that distributed unit
     *  tests (and any existing on-disk checkpoints) rely on. Note that turning
     *  this on changes which parameters live on which rank, so checkpoints saved
     *  before the flag was on are NOT compatible after enabling — save+restore is
     *  per-rank-partition and the partition assignment changed underneath. Pick
     *  the flag value once at the start of training and don't toggle it.
     */
    bool balanced_partitioning{false};

    /** Partitioning strategy. See PartitioningMode docs above for the trade-offs.
     *  Default ParamLevel preserves legacy behaviour and existing checkpoints.
     */
    PartitioningMode partitioning_mode{PartitioningMode::ParamLevel};

    ZeROStage1Config() = default;
};

/**
 * @brief ZeRO Stage 1: Optimizer State Partitioning
 *
 * Partitions optimizer states (momentum, variance) across distributed ranks
 * to reduce memory usage by N-fold where N = world_size.
 *
 * **Algorithm**:
 * 1. Parameters are replicated on all ranks
 * 2. Optimizer states are partitioned (each rank owns 1/N)
 * 3. Gradients are all-reduced before optimizer step
 * 4. Each rank updates its parameter partition
 * 5. Parameters are all-gathered after update
 *
 * **Memory Savings**:
 * - Adam: 4x reduction in optimizer states (2 states per param)
 * - SGD with momentum: 2x reduction
 *
 * **CPU Offload**:
 * - Optionally offload optimizer states to CPU RAM
 * - States are fetched to GPU before update, then offloaded back
 * - Enables training models larger than GPU memory
 *
 * @code
 * // Example: Distributed training with ZeRO Stage 1
 * distributed::init_process_group("nccl");
 * auto rank = distributed::get_rank();
 * auto world_size = distributed::get_world_size();
 *
 * // Create base optimizer
 * auto adam = std::make_unique<Adam>(model.parameters(), 1e-3);
 *
 * // Wrap with ZeRO Stage 1
 * ZeROStage1Config config;
 * config.world_size = world_size;
 * config.rank = rank;
 * config.offload_to_cpu = true;
 * auto zero_optimizer = ZeROStage1Optimizer(std::move(adam), config);
 *
 * // Training loop
 * for (auto& batch : dataloader) {
 *     zero_optimizer.zero_grad();
 *     auto output = model.forward(batch.input);
 *     auto loss = criterion(output, batch.target);
 *     loss.backward();
 *     zero_optimizer.step();  // Handles all distributed communication
 * }
 * @endcode
 *
 * @see ZeROStage2Optimizer, ZeROStage3Optimizer
 */
class ZeROStage1Optimizer : public Optimizer {
public:
    /**
     * @brief Construct ZeRO Stage 1 optimizer.
     *
     * Takes shared ownership of the base optimizer so pybind11 can keep a
     * concurrent reference alive. C++ callers can pass either a
     * `shared_ptr<Optimizer>` directly or `std::move(unique_ptr<Optimizer>)` —
     * `shared_ptr`'s implicit `unique_ptr&&` conversion handles both.
     *
     * @param base_optimizer Base optimizer (Adam, SGD, etc.)
     * @param config ZeRO configuration
     * @throws std::invalid_argument if rank >= world_size or base_optimizer is null
     */
    ZeROStage1Optimizer(
        std::shared_ptr<Optimizer> base_optimizer,
        const ZeROStage1Config& config
    );

    /**
     * @brief Destructor - cleanup resources
     */
    ~ZeROStage1Optimizer() override;

    /**
     * @brief Perform optimizer step with distributed state partitioning
     *
     * Algorithm:
     * 1. All-reduce gradients across ranks (sum)
     * 2. If CPU offload: Fetch local state partition to GPU
     * 3. Update local parameter partition with base optimizer
     * 4. If CPU offload: Offload states back to CPU
     * 5. All-gather updated parameters across ranks
     *
     * @throws std::runtime_error if distributed not initialized
     */
    auto step_impl() -> void override;

    /**
     * @brief Zero all parameter gradients
     */
    auto zero_grad() -> void;

    /**
     * @brief Get optimizer state dictionary
     *
     * Returns only the local partition of optimizer states.
     * Use save_checkpoint() to save full distributed state.
     *
     * @return Map of state variable names to tensors
     */
    auto state_dict() const -> std::unordered_map<std::string, Tensor> override;

    /**
     * @brief Load optimizer state dictionary
     *
     * Loads the local partition of optimizer states.
     * Use load_checkpoint() to load full distributed state.
     *
     * @param state State dictionary to load
     */
    auto load_state_dict(const std::unordered_map<std::string, Tensor>& state) -> void override;

    /**
     * @brief Save distributed checkpoint (all ranks)
     *
     * Each rank saves its partition. Master rank saves metadata.
     *
     * @param path_prefix Checkpoint path prefix (rank ID appended)
     */
    auto save_checkpoint(const std::string& path_prefix) const -> void;

    /**
     * @brief Load distributed checkpoint (all ranks)
     *
     * Each rank loads its partition. Master rank loads metadata.
     *
     * @param path_prefix Checkpoint path prefix
     */
    auto load_checkpoint(const std::string& path_prefix) -> void;

    // Accessors

    /**
     * @brief Get current rank ID
     */
    auto rank() const -> int { return config_.rank; }

    /**
     * @brief Get world size
     */
    auto world_size() const -> int { return config_.world_size; }

    /**
     * @brief Check if CPU offload is enabled
     */
    auto is_cpu_offload_enabled() const -> bool { return config_.offload_to_cpu; }

    /**
     * @brief Get number of parameters in local partition
     */
    auto local_param_count() const -> size_t;

    /**
     * @brief Get memory usage statistics
     */
    struct MemoryStats {
        size_t gpu_optimizer_memory{0};     ///< GPU memory for optimizer states (bytes)
        size_t cpu_optimizer_memory{0};     ///< CPU memory for optimizer states (bytes)
        size_t gpu_gradient_memory{0};      ///< GPU memory for gradients (bytes)
        size_t num_parameters{0};           ///< Total number of parameters
        size_t num_local_parameters{0};     ///< Parameters in local partition
    };

    /**
     * @brief Get memory usage statistics
     */
    auto get_memory_stats() const -> MemoryStats;

    /**
     * @brief Get base optimizer (const)
     */
    auto base_optimizer() const -> const Optimizer& {
        return *base_optimizer_;
    }

    /**
     * @brief Borrow the OffloadEngine this optimizer is currently using, if any.
     *
     * Returns nullptr when CPU offload is disabled (config_.offload_to_cpu == false)
     * or before construction has wired one up. When non-null, the engine is either
     * the one passed in via Config::shared_offload_engine or a privately-constructed
     * one — callers can't tell the difference and don't need to. Useful for tests
     * verifying the shared-engine plumbing and for downstream subsystems that want
     * to route their own transfers through the same pinned pool.
     */
    auto offload_engine() const -> std::shared_ptr<core::OffloadEngine> {
        return offload_engine_;
    }

    // ========================================================================
    // Performance Profiling
    // ========================================================================

    /**
     * @brief Performance profiling statistics
     */
    struct ProfilingStats {
        // Timing statistics (milliseconds)
        double total_step_time_ms{0.0};
        double gather_time_ms{0.0};
        double scatter_time_ms{0.0};
        double communication_time_ms{0.0};
        double compute_time_ms{0.0};
        double offload_time_ms{0.0};
        double all_reduce_time_ms{0.0};
        double all_gather_time_ms{0.0};

        // Memory statistics (bytes)
        size_t peak_memory_bytes{0};
        size_t current_memory_bytes{0};
        size_t transferred_bytes{0};
        size_t offloaded_bytes{0};

        // Operation counts
        size_t num_steps{0};
        size_t num_gathers{0};
        size_t num_scatters{0};
        size_t num_offloads{0};
        size_t num_all_reduces{0};
        size_t num_all_gathers{0};

        // Overlap metrics (0.0 to 1.0)
        double comm_compute_overlap_ratio{0.0};

        // Average times
        double avg_step_time_ms{0.0};
        double avg_gather_time_ms{0.0};
        double avg_scatter_time_ms{0.0};
        double avg_all_reduce_time_ms{0.0};

        // Bandwidth (MB/s)
        double effective_bandwidth_mbps{0.0};

        /**
         * @brief Print formatted profiling summary
         */
        auto print_summary() const -> void;

        /**
         * @brief Get profiling data as string
         */
        auto to_string() const -> std::string;
    };

    /**
     * @brief Get performance profiling statistics
     */
    auto get_profiling_stats() const -> ProfilingStats;

    /**
     * @brief Reset profiling statistics
     */
    auto reset_profiling_stats() -> void;

    /**
     * @brief Enable or disable profiling
     * @param enabled Whether to enable profiling
     */
    auto enable_profiling(bool enabled) -> void;

    /**
     * @brief Check if profiling is enabled
     */
    auto is_profiling_enabled() const -> bool { return profiling_enabled_; }

protected:
    /**
     * @brief State partition for a single rank
     */
    struct StatePartition {
        // ----- In-memory state (legacy + CPU-offload paths) -----
        int rank{0};                                ///< Rank that owns this partition
        std::vector<std::shared_ptr<Variable>> params;  ///< Parameters in partition
        std::vector<Tensor> momentum;               ///< Momentum states (if applicable)
        std::vector<Tensor> variance;               ///< Variance states (if applicable)
        std::vector<Tensor> master_params;          ///< fp32 master param copies (only populated when ZeROStage1Config::use_master_fp32 is set; one per param, in lockstep with `params`)
        std::vector<Tensor> momentum_scales;        ///< fp32 per-tensor scales when momentum is int8-quantized on CPU; empty Tensor when not quantized
        std::vector<Tensor> variance_scales;        ///< fp32 per-tensor scales when variance is int8-quantized on CPU; empty Tensor when not quantized
        Device device{Device::cpu()};               ///< Where states are stored
        size_t memory_bytes{0};                     ///< Total memory usage

        // ----- NVMe offload metadata -----
        // When ZeROStage1Config::offload_to_nvme is set, the corresponding tensor in
        // momentum/variance is held as an empty Tensor() and the data lives on disk
        // at the path below. Shape + dtype are tracked here so we can reconstruct
        // the tensor at fetch time without needing a header in the file. An entry
        // with empty `path` means the slot's data is currently in memory.
        struct DiskSlot {
            std::string path;                       ///< File path, empty if not on disk
            std::vector<int64_t> shape;             ///< Shape recorded at offload time
            DType dtype{DType::Float32};            ///< Dtype recorded at offload time
            bool on_disk() const { return !path.empty(); }
        };
        std::vector<DiskSlot> momentum_disk;        ///< Per-momentum disk metadata (empty vector when NVMe disabled)
        std::vector<DiskSlot> variance_disk;        ///< Per-variance disk metadata
        std::vector<DiskSlot> momentum_scale_disk;  ///< Per-momentum-scale disk metadata (only populated when also quantizing)
        std::vector<DiskSlot> variance_scale_disk;  ///< Per-variance-scale disk metadata
    };

    // Core components
    // Stored as shared_ptr so Python bindings can pass a shared_ptr<Optimizer>
    // without the unsafe raw-ownership transfer that a unique_ptr would require.
    // C++ callers that use the unique_ptr constructor below still work — a
    // shared_ptr is trivially constructible from a moved unique_ptr.
    std::shared_ptr<Optimizer> base_optimizer_;     ///< Wrapped optimizer
    ZeROStage1Config config_;                       ///< Configuration
    std::vector<StatePartition> partitions_;        ///< State partitions for all ranks
    std::shared_ptr<core::OffloadEngine> offload_engine_;  ///< CPU offload engine

    /** Layout metadata describing element-level partitioning of all parameters across
     *  ranks. Populated by the element-mode partitioner; consumed by the optimizer step,
     *  reduce-scatter, and all-gather paths. See `partition_layout_` for full semantics.
     */
    struct PartitionLayout {
        struct ParamEntry {
            int64_t global_offset{0};
            int64_t numel{0};
            std::vector<int64_t> original_shape;
            DType dtype{DType::Float32};
        };
        std::vector<ParamEntry> params;          // one entry per parameter, lockstep with parameters_
        std::vector<int64_t> rank_starts;        // size == world_size + 1; rank R owns [rank_starts[R], rank_starts[R+1])
        int64_t total_elements_padded{0};        // rounded up to world_size multiple

        auto rank_size(int rank) const -> int64_t {
            return rank_starts[rank + 1] - rank_starts[rank];
        }
    };

public:
    /** Test-only accessor: borrow the element partition layout. Empty in ParamLevel mode. */
    auto test_partition_layout() const -> const PartitionLayout& {
        return partition_layout_;
    }

protected:
    /** Element-level partition layout (only populated when
     *  config_.partitioning_mode == PartitioningMode::ElementLevel).
     *
     *  Conceptually: every parameter is concatenated into a single global flat buffer of
     *  total_elements rounded up to a multiple of world_size. That buffer is then split
     *  into `world_size` equal slices; rank R owns slice R. For each parameter we record:
     *
     *    - global_offset: starting position in the global flat buffer (in elements).
     *    - numel:         total element count.
     *    - original_shape: shape to restore after the all-gather rebinding step.
     *
     *  And per-rank we record:
     *    - rank_starts: CSR-style boundary vector (size == world_size + 1); rank R's
     *      half-open element range is [rank_starts[R], rank_starts[R+1]). All ranks
     *      have the same size except possibly the last on uneven divides.
     *
     *  Empty when partitioning_mode == PartitioningMode::ParamLevel.
     */
    PartitionLayout partition_layout_;

    // Communication handles for async operations
    std::vector<Tensor> gradient_buffers_;          ///< Buffers for gradient all-reduce
    std::vector<Tensor> param_buffers_;             ///< Buffers for parameter all-gather

    // Optimizer state
    int64_t step_count_{0};                         ///< Step counter for bias correction
    bool states_on_cpu_{false};                     ///< Whether optimizer states are currently on CPU

    // Async communication. When the process group supports stream-based collectives
    // (NCCL/RCCL on GPU), all_reduce_gradients can be split into a "kick off async" /
    // "wait" pair so the all-reduce overlaps with fetch_states_to_gpu in step_impl.
    // Stored as void* (cudaStream_t / hipStream_t) for opacity — see ddp.cpp for the
    // CUDA/HIP-conditional lifecycle pattern this mirrors.
    // Exposed as protected so ZeROStage3Optimizer can route gather_parameter_impl through
    // the same comm stream. Was inferred-protected previously (the surrounding section at
    // line 404 starts protected); the explicit re-declaration below is just for clarity.
    bool use_gpu_comm_{false};                      ///< Process group supports async stream
    void* comm_stream_{nullptr};                    ///< Dedicated comm stream (cudaStream_t)

    bool async_all_reduce_in_flight_{false};        ///< Set by issue_async_all_reduce_*
    // Compressor outputs survive across the issue/wait boundary so decompress can run
    // after the stream sync. Indexed in lockstep with parameters_; only populated when
    // grad_compressor is configured AND an async all-reduce is in flight.
    std::vector<distributed::CompressedGradient> in_flight_compressed_;

    // Synchronization
    mutable std::mutex mutex_;                      ///< Thread safety

    // Profiling state
    bool profiling_enabled_{false};                 ///< Enable performance profiling
    mutable ProfilingStats profiling_stats_;        ///< Accumulated profiling statistics
    mutable std::mutex profiling_mutex_;            ///< Profiling thread safety

    // Timing helpers
    std::chrono::steady_clock::time_point step_start_time_;
    std::chrono::steady_clock::time_point comm_start_time_;
    std::chrono::steady_clock::time_point compute_start_time_;

    // Initialization

    /**
     * @brief Partition parameters across ranks
     *
     * Assigns each parameter to a rank based on parameter index.
     * Tries to balance memory across ranks.
     */
    auto partition_parameters() -> void;

    /** Compute the element-level partition layout for `parameters_`. Pure function of
     *  parameter shapes/dtypes and `world_size` — does not allocate any optimizer state
     *  itself. Called by `partition_parameters()` when in ElementLevel mode.
     */
    auto compute_element_partition_layout() -> void;

    /**
     * @brief Initialize optimizer states for local partition
     *
     * Creates momentum/variance buffers for base optimizer.
     */
    auto initialize_optimizer_states() -> void;

    /**
     * @brief Initialize CPU offload engine (if enabled)
     */
    auto initialize_offload_engine() -> void;

    // Communication

    /**
     * @brief All-reduce gradients across ranks (sum)
     *
     * Synchronizes gradients before optimizer step.
     */
    auto all_reduce_gradients() -> void;

    /**
     * @brief Issue all per-parameter all-reduce calls without waiting (split form).
     *
     * Schedules each parameter's all-reduce on the dedicated comm stream when GPU comm is
     * available, otherwise falls back to synchronous in-place reduction. Caller must follow
     * up with wait_for_async_all_reduce() before reading any gradient. Designed so step_impl
     * can do useful CPU/PCIe work (fetch_states_to_gpu) between issue and wait.
     */
    auto issue_async_all_reduce_gradients() -> void;

    /**
     * @brief Block on the comm stream and finalize per-parameter gradients (split form).
     *
     * Synchronizes with the comm stream issued by issue_async_all_reduce_gradients(), then
     * runs decompress (when a grad_compressor is configured) and writes results back to
     * the parameter Variables. Idempotent — no-op when no async is in flight.
     */
    auto wait_for_async_all_reduce() -> void;

    /**
     * @brief All-gather parameters after update
     *
     * Reconstructs full parameter set from partitions.
     */
    auto all_gather_parameters() -> void;

    /** ElementLevel-mode replacement for all_gather_parameters().
     *  After update_local_partition_element_mode wrote each rank's slice into the
     *  per-param tensors, we still need the OTHER ranks' slices. Issues a single
     *  all_gather of a global flat buffer of size world_size * rank_size, then scatters
     *  per-param ranges back to each parameter's tensor.
     */
    auto all_gather_parameters_element_mode() -> void;

    // State management

    /**
     * @brief Update local partition of optimizer states
     *
     * Calls base optimizer step() on local partition only.
     */
    auto update_local_partition() -> void;

    /** ElementLevel-mode replacement for update_local_partition() / update_partition_adam.
     *  Stages every parameter's gradient into the rank's slice of a global flat buffer,
     *  then runs the optimizer math on the slice. After the math, writes the updated
     *  parameter slice back into the rank's slot in the (yet-to-be-all-gathered) global
     *  param flat buffer. The rank's master copy (if used) is the persistent state that
     *  carries fp32 precision across steps.
     */
    auto update_local_partition_element_mode() -> void;

    /**
     * @brief Apply Adam update algorithm to partition
     *
     * @param partition State partition to update
     * @param lr Learning rate
     * @param beta1 First moment decay rate
     * @param beta2 Second moment decay rate
     * @param eps Numerical stability constant
     * @param weight_decay Weight decay coefficient
     */
    auto update_partition_adam(
        StatePartition& partition,
        double lr,
        double beta1,
        double beta2,
        double eps,
        double weight_decay
    ) -> void;

    /**
     * @brief Apply AdamW update algorithm to partition
     *
     * @param partition State partition to update
     * @param lr Learning rate
     * @param beta1 First moment decay rate
     * @param beta2 Second moment decay rate
     * @param eps Numerical stability constant
     * @param weight_decay Weight decay coefficient (decoupled)
     */
    auto update_partition_adamw(
        StatePartition& partition,
        double lr,
        double beta1,
        double beta2,
        double eps,
        double weight_decay
    ) -> void;

    /**
     * @brief Apply SGD update algorithm to partition
     *
     * @param partition State partition to update
     * @param lr Learning rate
     * @param momentum_coef Momentum coefficient
     * @param weight_decay Weight decay coefficient
     */
    auto update_partition_sgd(
        StatePartition& partition,
        double lr,
        double momentum_coef,
        double weight_decay
    ) -> void;

    /**
     * @brief Fetch optimizer states from CPU to GPU (if offloaded)
     */
    auto fetch_states_to_gpu() -> void;

    /**
     * @brief Offload optimizer states from GPU to CPU (if enabled)
     */
    auto offload_states_to_cpu() -> void;

    /**
     * @brief Get local partition for current rank
     */
    auto local_partition() -> StatePartition& {
        return partitions_[config_.rank];
    }

    /**
     * @brief Get local partition for current rank (const)
     */
    auto local_partition() const -> const StatePartition& {
        return partitions_[config_.rank];
    }

private:
    // Internal helpers that don't acquire mutex_ (caller must hold lock)
    auto state_dict_unlocked() const -> std::unordered_map<std::string, Tensor>;
    auto load_state_dict_unlocked(const std::unordered_map<std::string, Tensor>& state) -> void;
};

/**
 * @brief Configuration for ZeRO Stage 2 Optimizer
 */
struct ZeROStage2Config : public ZeROStage1Config {
    size_t gradient_bucket_size{25 * 1024 * 1024};  ///< Target bucket size (default: 25MB)
    bool reduce_scatter_in_backward{true};           ///< Enable reduce-scatter during backward
    bool gradient_bucketing{true};                   ///< Enable gradient bucketing

    ZeROStage2Config() = default;
};

/**
 * @brief ZeRO Stage 2: Gradient + Optimizer State Partitioning
 *
 * Extends ZeRO Stage 1 by also partitioning gradients across ranks.
 * Uses reduce-scatter during backward pass to compute and partition
 * gradients simultaneously, eliminating the need for all-reduce.
 *
 * **Algorithm**:
 * 1. Parameters are replicated on all ranks
 * 2. During backward pass: Reduce-scatter gradients by partition
 * 3. Each rank receives sum of its gradient partition only
 * 4. Optimizer states are partitioned (inherited from Stage 1)
 * 5. Each rank updates its parameter partition
 * 6. Parameters are all-gathered after update
 *
 * **Memory Savings**:
 * - Adam: 8x reduction (4x optimizer states + 4x gradients)
 * - SGD with momentum: 4x reduction (2x optimizer states + 2x gradients)
 *
 * **Key Optimization: Gradient Bucketing**
 * - Groups small gradients into larger buckets for efficient communication
 * - Overlaps reduce-scatter with backward computation
 * - Reduces communication overhead
 *
 * @code
 * // Example: Distributed training with ZeRO Stage 2
 * distributed::init_process_group("nccl");
 * auto rank = distributed::get_rank();
 * auto world_size = distributed::get_world_size();
 *
 * // Create model and optimizer
 * auto model = MyModel();
 * auto adam = std::make_unique<Adam>(model.parameters(), 1e-3);
 *
 * // Wrap with ZeRO Stage 2
 * ZeROStage2Config config;
 * config.world_size = world_size;
 * config.rank = rank;
 * config.offload_to_cpu = true;
 * config.gradient_bucket_size = 50 * 1024 * 1024;  // 50MB buckets
 * auto zero_optimizer = ZeROStage2Optimizer(std::move(adam), config);
 *
 * // Register backward hooks for gradient reduce-scatter
 * zero_optimizer.register_backward_hooks();
 *
 * // Training loop
 * for (auto& batch : dataloader) {
 *     zero_optimizer.zero_grad();
 *     auto output = model.forward(batch.input);
 *     auto loss = criterion(output, batch.target);
 *     loss.backward();  // Gradients reduced-scattered automatically
 *     zero_optimizer.step();
 * }
 * @endcode
 *
 * @see ZeROStage1Optimizer, ZeROStage3Optimizer
 */
class ZeROStage2Optimizer : public ZeROStage1Optimizer {
public:
    /**
     * @brief Construct ZeRO Stage 2 optimizer. Accepts
     * `shared_ptr<Optimizer>` or `std::move(unique_ptr<Optimizer>)` — see
     * ZeROStage1Optimizer's constructor comment for rationale.
     *
     * @param base_optimizer Base optimizer (Adam, SGD, etc.)
     * @param config ZeRO Stage 2 configuration
     * @throws std::invalid_argument if rank >= world_size or base_optimizer is null
     */
    ZeROStage2Optimizer(
        std::shared_ptr<Optimizer> base_optimizer,
        const ZeROStage2Config& config
    );

    /**
     * @brief Destructor - cleanup hooks and resources
     */
    ~ZeROStage2Optimizer() override;

    /**
     * @brief Perform optimizer step with gradient partitioning
     *
     * Algorithm:
     * 1. Reduce-scatter gradients (already done in backward hooks)
     * 2. If CPU offload: Fetch local state partition to GPU
     * 3. Update local parameter partition with base optimizer
     * 4. If CPU offload: Offload states back to CPU
     * 5. All-gather updated parameters across ranks
     *
     * Note: Unlike Stage 1, no all-reduce is needed since gradients
     * are already reduced-scattered during backward pass.
     *
     * @throws std::runtime_error if distributed not initialized
     */
    auto step_impl() -> void override;

    /**
     * @brief Register backward hooks for gradient reduce-scatter
     *
     * Must be called after model creation to enable automatic gradient
     * partitioning during backward pass. Hooks are registered per parameter
     * and trigger reduce-scatter when gradient is computed.
     *
     * @throws std::runtime_error if parameters have no grad_fn
     */
    auto register_backward_hooks() -> void;

    /**
     * @brief Unregister all backward hooks previously installed by register_backward_hooks().
     *
     * Safe to call multiple times. Called automatically from the destructor so the optimizer
     * cannot leave dangling hooks pointing into freed memory.
     */
    auto unregister_backward_hooks() -> void;

    /**
     * @brief Get gradient bucket statistics
     */
    struct BucketStats {
        size_t num_buckets{0};           ///< Number of gradient buckets
        size_t avg_bucket_size{0};       ///< Average bucket size (bytes)
        size_t max_bucket_size{0};       ///< Maximum bucket size (bytes)
        size_t total_gradient_memory{0}; ///< Total gradient memory (bytes)
    };

    /**
     * @brief Get gradient bucket statistics
     */
    auto get_bucket_stats() const -> BucketStats;

    /**
     * @brief Check if backward hooks are registered
     */
    auto hooks_registered() const -> bool { return hooks_registered_; }

private:
    /**
     * @brief Gradient bucket for efficient reduce-scatter
     */
    struct GradientBucket {
        std::vector<std::shared_ptr<Variable>> params;  ///< Parameters in bucket
        std::vector<Tensor> gradient_buffers;           ///< Stashed grads (one per param), filled by autograd hook before accumulation
        std::vector<size_t> hook_ids;                   ///< Variable::register_hook handles, one per param, for clean unregistration
        size_t total_size{0};                           ///< Total size in bytes
        int target_rank{-1};                            ///< Rank that owns these gradients
        bool ready{false};                              ///< All gradients computed
        size_t gradients_received{0};                   ///< Count of gradients received
        std::unique_ptr<std::mutex> mutex;              ///< Thread safety for async hooks

        // --- Persistent flat staging buffers (allocated lazily on first reduce-scatter) ---
        // The legacy code path called flatten_tensors() and zeros() every step, allocating two
        // bucket-sized tensors per backward. By keeping these around for the lifetime of the
        // bucket we eliminate 2*N allocator hits per training step, where N is the number of
        // buckets — the dominant residual allocator pressure once parameter/optimizer-state
        // memory is partitioned by ZeRO.
        Tensor flat_buffer;                ///< Bucket-wide flat scratch (size = sum of param numels)
        Tensor flat_partition_buffer;      ///< Reduce-scatter output buffer (size = ceil(total / world_size))
        std::vector<int64_t> param_offsets_elem;  ///< Element offset of each param in flat_buffer
        std::vector<int64_t> param_sizes_elem;    ///< Numel of each param

        // Constructor to initialize mutex
        GradientBucket() : mutex(std::make_unique<std::mutex>()) {}

        // Move constructor
        GradientBucket(GradientBucket&&) = default;
        GradientBucket& operator=(GradientBucket&&) = default;

        // Delete copy operations (mutex is non-copyable)
        GradientBucket(const GradientBucket&) = delete;
        GradientBucket& operator=(const GradientBucket&) = delete;
    };

    /** Gradient bucket for ElementLevel mode.
     *  Represents a contiguous range of the global flat-grad buffer. Each rank receives
     *  its 1/world_size slice of the bucket via reduce_scatter; non-owner ranks NEVER
     *  hold the whole bucket on receive (the memory win that motivated this whole mode).
     */
    struct ElementBucket {
        int64_t global_start{0};
        int64_t global_end{0};
        std::vector<size_t> param_indices;
        Tensor flat_buffer;
        size_t hooks_received{0};
        std::unique_ptr<std::mutex> mutex{std::make_unique<std::mutex>()};

        ElementBucket() = default;
        ElementBucket(ElementBucket&&) = default;
        ElementBucket& operator=(ElementBucket&&) = default;
        ElementBucket(const ElementBucket&) = delete;
        ElementBucket& operator=(const ElementBucket&) = delete;
    };

    // Configuration
    ZeROStage2Config stage2_config_;

    // Gradient buckets
    std::vector<GradientBucket> gradient_buckets_;
    std::vector<ElementBucket> element_buckets_;
    mutable std::mutex buckets_mutex_;  // Mutable so const methods can lock

    // Hook management
    bool hooks_registered_{false};
    std::vector<size_t> element_hook_ids_;  ///< register_hook handles for ElementLevel mode, one per parameter

    // Initialization

    /**
     * @brief Create gradient buckets for efficient communication
     *
     * Groups parameters into buckets based on:
     * - Target size (gradient_bucket_size config)
     * - Parameter ownership (which rank owns the gradient partition)
     * - Memory alignment
     */
    auto create_gradient_buckets() -> void;
    auto create_gradient_buckets_element_mode() -> void;

    // Communication

    /**
     * @brief Reduce-scatter gradients in a bucket
     *
     * Performs reduce-scatter operation on all gradients in the bucket:
     * 1. Flatten gradients into contiguous buffer
     * 2. Reduce-scatter: Each rank gets 1/N of the summed gradients
     * 3. Unflatten into individual gradient tensors
     * 4. Free non-local gradients
     *
     * @param bucket Gradient bucket to process
     */
    auto reduce_scatter_gradients(GradientBucket& bucket) -> void;

    /**
     * @brief Gradient hook callback
     *
     * Called during backward pass when gradient is computed.
     * Marks gradient as ready and triggers reduce-scatter when
     * all gradients in bucket are available.
     *
     * @param bucket_idx Index of gradient bucket
     * @param param_idx Index of parameter in bucket
     */
    auto gradient_hook(size_t bucket_idx, size_t param_idx) -> void;

    /**
     * @brief Variant of gradient_hook used by the autograd register_hook callback.
     *
     * Stashes @p grad into the bucket's gradient_buffers[param_idx] slot before grad
     * accumulation runs (so we never have to consult param->grad(), which is not yet
     * populated when the hook fires for this param). Triggers reduce-scatter when the
     * bucket has received all of its gradients.
     */
    auto gradient_hook(size_t bucket_idx, size_t param_idx, const Tensor& grad) -> void;

    /** Handler for the autograd register_hook callback in ElementLevel mode. Stages grad
     *  into every overlapping ElementBucket's flat_buffer, fires reduce_scatter when a
     *  bucket has received all its params' grads.
     */
    auto element_gradient_hook(size_t param_idx, const Tensor& grad) -> void;
    auto reduce_scatter_element_bucket(ElementBucket& bucket) -> void;

    /**
     * @brief Check if bucket is ready for reduce-scatter
     *
     * @param bucket Gradient bucket to check
     * @return true if all gradients computed
     */
    auto is_bucket_ready(const GradientBucket& bucket) const -> bool;

    /**
     * @brief Flatten tensors into contiguous buffer
     *
     * @param tensors Vector of gradient tensors
     * @return Flattened tensor
     */
    auto flatten_tensors(const std::vector<Tensor>& tensors) -> Tensor;

    /**
     * @brief Unflatten buffer into individual tensors
     *
     * @param flattened Flattened tensor buffer
     * @param targets Target tensors to write into
     */
    auto unflatten_into(const Tensor& flattened, std::vector<Tensor>& targets) -> void;
};

/**
 * @brief Configuration for ZeRO Stage 3 Optimizer
 */
struct Stage3Config : public ZeROStage2Config {
    // ========================================================================
    // Prefetching Configuration
    // ========================================================================

    /** Size of buckets for parameter gathering (bytes).
     *  Larger buckets = fewer all-gather calls but more memory.
     *  Recommended: 100-500 MB based on model size. */
    size_t prefetch_bucket_size{100 * 1024 * 1024};  // 100 MB default

    /** Number of layers to prefetch ahead during forward/backward.
     *  Larger depth = better latency hiding but more GPU memory.
     *  Recommended: 1-4 based on available memory. */
    int prefetch_depth{2};

    /** Maximum number of concurrent prefetch operations. */
    int max_concurrent_prefetches{4};

    /** Enable overlap of communication with computation.
     *  Uses separate CUDA streams for gather/compute. */
    bool overlap_comm_compute{true};

    // ========================================================================
    // Memory Management
    // ========================================================================

    /** Maximum number of gathered parameters to cache simultaneously.
     *  Limits peak memory usage during forward/backward passes. */
    int max_cached_params{10};

    /** Enable parameter caching across forward/backward passes.
     *  Avoids re-gathering parameters if they're used in both passes. */
    bool cache_params_across_passes{true};

    /** Threshold for small parameters (bytes).
     *  Parameters smaller than this are not partitioned. */
    size_t partition_threshold{1024};  // 1 KB

    /** Pin first layer parameters in memory (keep gathered). */
    bool pin_first_layer{true};

    /** Pin last layer parameters in memory (keep gathered). */
    bool pin_last_layer{true};

    /** Maximum memory for gathered parameters (bytes). */
    size_t max_gathered_buffer_size{500 * 1024 * 1024};  // 500 MB

    // ========================================================================
    // CPU Offload Integration
    // ========================================================================

    /** Offload partitioned parameters to CPU when not in use. */
    bool offload_params_to_cpu{false};

    /** Offload gathered parameters to CPU after use. */
    bool offload_gathered_to_cpu{false};

    // ========================================================================
    // Communication Settings
    // ========================================================================

    /** Use asynchronous all-gather operations. */
    bool use_async_gather{true};

    /** Use separate CUDA streams for communication. */
    bool use_separate_streams{true};

    /** CUDA stream priority for gather operations (higher = more urgent). */
    int gather_stream_priority{-1};

    /** Use NCCL groups for parallel gather operations.
     *  Experimental: may improve bandwidth utilization. */
    bool use_nccl_groups{false};

    /** Enable gradient checkpointing integration.
     *  Manages parameter gathering during recomputation. */
    bool gradient_checkpointing_aware{false};

    /** Align parameter partitions to this byte boundary.
     *  Improves memory coalescing. Recommended: 128 or 256. */
    size_t partition_alignment{128};

    // ========================================================================
    // Adaptive Prefetch Configuration (Phase 7 Optimizations)
    // ========================================================================

    /** Enable adaptive prefetch depth adjustment based on performance */
    bool enable_adaptive_prefetch{true};

    /** Target overlap ratio: fraction of communication to hide (0.0-1.0) */
    double target_overlap_ratio{0.8};  // Target 80% overlap

    /** Minimum allowed prefetch depth */
    int min_prefetch_depth{1};

    /** Maximum allowed prefetch depth */
    int max_prefetch_depth{5};

    /** Number of recent operations to consider for adaptation */
    size_t prefetch_window_size{10};

    // ========================================================================
    // Dynamic Bucket Sizing Configuration (Phase 7 Optimizations)
    // ========================================================================

    /** Enable dynamic bucket size adjustment based on communication patterns */
    bool enable_dynamic_bucket_sizing{true};

    /** Minimum bucket size (bytes) */
    size_t min_bucket_size{1 * 1024 * 1024};   // 1MB

    /** Maximum bucket size (bytes) */
    size_t max_bucket_size{500 * 1024 * 1024}; // 500MB

    /** Target communication efficiency (0.0-1.0) */
    double target_comm_efficiency{0.9};  // Target 90% efficiency

    // ========================================================================
    // Adaptive CPU Offload Configuration (Phase 7 Optimizations)
    // ========================================================================

    /** Enable adaptive offloading based on GPU memory pressure */
    bool enable_adaptive_offload{true};

    /** GPU memory pressure threshold to trigger offload (0.0-1.0) */
    double memory_pressure_threshold{0.85};  // Offload at 85% GPU memory

    /** Hysteresis for offload decisions to prevent thrashing (bytes) */
    size_t offload_hysteresis{100 * 1024 * 1024};  // 100MB hysteresis

    /** Monitor interval for memory pressure (milliseconds) */
    int memory_monitor_interval_ms{100};

    // ========================================================================
    // Chunked all-gather (review item #9)
    // ========================================================================
    //
    // The legacy `gather_parameter_impl()` path issues a single all_gather that
    // produces `world_size * partition_n` elements of staging output before that
    // result can be folded into `state.full_param`. For typical transformer
    // params this is fine, but for jumbo tensors (a 32k×4096 fp32 LM head ≈ 512
    // MB partition × 8 ranks ≈ 4 GB transient on every GPU during the gather)
    // it breaks Stage 3's "model bigger than one device" promise on exactly the
    // few params that prompted the user to use Stage 3 in the first place.
    //
    // When `chunked_gather_threshold > 0` and the param's full byte count
    // exceeds it, the gather is decomposed into K rounds, each gathering at
    // most `chunked_gather_chunk_size` bytes per rank. Peak transient drops
    // from `world_size × partition_n` to `world_size × chunk_size` regardless
    // of partition_n, at the cost of K-1 extra collective launches (which are
    // bandwidth-amortising once the chunk is ≥ a few MB).

    /** Threshold (bytes) at and above which `gather_parameter_impl()` switches
     *  from a single bulk all-gather to a chunked all-gather. Compared against
     *  `world_size × partition_n × dtype_size` — the size of the full gathered
     *  parameter, not the per-rank partition.
     *
     *  Set to 0 (default) to disable chunked gather entirely; this preserves
     *  the legacy bulk path. Recommended starting value when enabling: 256 MB.
     */
    size_t chunked_gather_threshold{0};

    /** Target per-rank chunk size (bytes) when chunked gather is active. Each
     *  round of the chunked collective gathers at most this many bytes from
     *  each rank, so the transient gather buffer per round is `world_size ×
     *  chunked_gather_chunk_size`.
     *
     *  Smaller chunks lower peak memory but multiply collective-launch overhead;
     *  64 MB is a reasonable default that keeps NCCL ring-bandwidth saturated
     *  on common transformer params. No effect when `chunked_gather_threshold`
     *  is 0.
     */
    size_t chunked_gather_chunk_size{64ULL * 1024 * 1024};

    Stage3Config() = default;
};

/**
 * @brief ZeRO Stage 3: Full Model Partitioning
 *
 * Most aggressive memory savings. Partitions parameters, gradients, AND
 * optimizer states across distributed ranks. Parameters are gathered
 * on-demand for computation and freed immediately after use.
 *
 * **Memory Usage Per Rank (N ranks, M parameters)**:
 *   - Parameters: M/N (only local partition)
 *   - Gradients: M/N (only local partition)
 *   - Optimizer States: 2M/N (only local partition for Adam)
 *   - Temporary: M (gathered parameters during forward/backward)
 *
 * **Communication Pattern**:
 *   - Forward: All-gather parameters before each layer
 *   - Backward: All-gather parameters, reduce-scatter gradients
 *   - Optimizer: No communication (operates on local partition)
 *
 * **Key Features**:
 *   - Automatic parameter gathering via forward/backward hooks
 *   - Intelligent prefetch scheduling to hide communication latency
 *   - Parameter caching to avoid redundant gathers
 *   - Reference counting for shared parameters
 *   - CPU offload support for maximum memory savings
 *
 * @code
 * // Example: Training with ZeRO Stage 3
 * distributed::init_process_group("nccl");
 * auto rank = distributed::get_rank();
 * auto world_size = distributed::get_world_size();
 *
 * // Create model
 * auto model = GPT2Model(GPT2Config::gpt2_medium());  // 350M parameters
 * model.to(Device::cuda(rank));
 *
 * // Configure ZeRO Stage 3
 * Stage3Config config;
 * config.world_size = world_size;
 * config.rank = rank;
 * config.prefetch_bucket_size = 100 * 1024 * 1024;  // 100 MB
 * config.prefetch_depth = 2;  // Prefetch 2 layers ahead
 * config.overlap_comm_compute = true;
 * config.cache_params_across_passes = true;
 *
 * // Create optimizer
 * auto optimizer = ZeROStage3Optimizer(
 *     std::make_unique<AdamW>(model.parameters(), 1e-4),
 *     config
 * );
 *
 * // Register model for parameter partitioning
 * optimizer.register_model(model);
 *
 * // Training loop (parameters automatically gathered/freed)
 * for (auto& batch : dataloader) {
 *     optimizer.zero_grad();
 *     auto output = model.forward(batch.input);  // Parameters gathered
 *     auto loss = criterion(output, batch.labels);
 *     loss.backward();  // Gradients reduced-scattered
 *     optimizer.step();  // Update local partition only
 * }
 * @endcode
 *
 * @see ZeROStage1Optimizer, ZeROStage2Optimizer
 */
class ZeROStage3Optimizer : public ZeROStage2Optimizer {
public:
    // ========================================================================
    // Forward Declarations (public section for visibility)
    // ========================================================================

    /**
     * @brief Async handle for non-blocking operations (forward declaration)
     */
    struct AsyncHandle;

    /**
     * @brief Prefetch scheduler for predictive parameter gathering
     */
    class PrefetchScheduler;

    // ========================================================================
    // Constructor & Destructor
    // ========================================================================

    /**
     * @brief Construct ZeRO Stage 3 optimizer. Accepts
     * `shared_ptr<Optimizer>` or `std::move(unique_ptr<Optimizer>)` — see
     * ZeROStage1Optimizer's constructor comment for rationale.
     *
     * @param base_optimizer The underlying optimizer (Adam, AdamW, SGD, etc.)
     * @param config Stage 3 configuration
     * @throws std::invalid_argument if rank >= world_size or base_optimizer is null
     */
    ZeROStage3Optimizer(
        std::shared_ptr<Optimizer> base_optimizer,
        const Stage3Config& config
    );

    /**
     * @brief Destructor - cleanup hooks and resources
     */
    ~ZeROStage3Optimizer() override;

    // ========================================================================
    // Model Registration
    // ========================================================================

    /**
     * @brief Register model for parameter partitioning
     *
     * This must be called before training begins. It:
     *   1. Partitions all model parameters across ranks
     *   2. Registers forward/backward hooks for automatic gather/scatter
     *   3. Builds execution graph for prefetch scheduling
     *   4. Initializes parameter state tracking
     *
     * @param model The model to partition
     * @throws std::runtime_error if model is already registered
     */
    auto register_model(Module& model) -> void;

    /**
     * @brief Unregister model (for cleanup or re-registration)
     */
    auto unregister_model() -> void;

    // ========================================================================
    // Optimizer Interface (Override)
    // ========================================================================

    /**
     * @brief Perform optimizer step
     *
     * Stage 3 step workflow:
     *   1. Wait for all gradient reduce-scatter to complete
     *   2. Update only local partition of parameters
     *   3. NO all-gather needed (parameters remain partitioned)
     *
     * @throws std::runtime_error if distributed not initialized
     */
    auto step_impl() -> void override;

    /**
     * @brief Zero gradients
     *
     * Only zeros local partition of gradients.
     */
    auto zero_grad() -> void;

    // ========================================================================
    // State Management
    // ========================================================================

    /**
     * @brief Get optimizer state dictionary
     *
     * Returns state for only the local partition. To get full state,
     * use gather_full_state().
     *
     * @return Map of state variable names to tensors
     */
    auto state_dict() const -> std::unordered_map<std::string, Tensor> override;

    /**
     * @brief Load optimizer state dictionary
     *
     * Expects partitioned state. To load from full checkpoint,
     * use load_full_state().
     *
     * @param state State dictionary to load
     */
    auto load_state_dict(const std::unordered_map<std::string, Tensor>& state) -> void override;

    /**
     * @brief Gather full optimizer state from all ranks
     *
     * Used for checkpointing. Expensive operation that should only
     * be called periodically.
     *
     * @return Full optimizer state (gathered from all ranks)
     */
    auto gather_full_state() -> std::unordered_map<std::string, Tensor>;

    /**
     * @brief Load from full (non-partitioned) checkpoint
     *
     * Automatically partitions the state across ranks.
     *
     * @param full_state Full optimizer state to load
     */
    auto load_full_state(const std::unordered_map<std::string, Tensor>& full_state) -> void;

    // ========================================================================
    // Manual Control API
    // ========================================================================

    /**
     * @brief Manually gather a parameter
     *
     * Useful for inference or fine-grained control.
     * Returns the full (gathered) parameter.
     *
     * @param param Parameter to gather (must be registered)
     * @return Full parameter (replicated across all ranks)
     * @throws std::runtime_error if communication fails
     */
    auto gather_parameter(Tensor* param) -> Tensor;

    /**
     * @brief Manually gather a parameter asynchronously
     *
     * @param param Parameter to gather
     * @return Handle for async operation
     */
    auto gather_parameter_async(Tensor* param) -> std::shared_ptr<AsyncHandle>;

    /**
     * @brief Wait for async gather to complete
     *
     * @param handle Handle from gather_parameter_async()
     * @return Full parameter tensor
     */
    auto wait_gather(std::shared_ptr<AsyncHandle> handle) -> Tensor;

    /**
     * @brief Manually free a gathered parameter
     *
     * Releases the full parameter, keeping only the local partition.
     *
     * @param param Parameter to free
     */
    auto free_gathered_parameter(Tensor* param) -> void;

    /**
     * @brief Prefetch parameters for upcoming layers
     *
     * Manually trigger prefetch for specific parameters.
     *
     * @param params Parameters to prefetch
     */
    auto prefetch_parameters(const std::vector<Tensor*>& params) -> void;

    // ========================================================================
    // State Queries
    // ========================================================================

    /**
     * @brief Parameter state enumeration
     */
    enum class ParameterState {
        PARTITIONED,    ///< Only local partition exists
        GATHERING,      ///< All-gather in progress (async)
        GATHERED,       ///< Full parameter available
        SCATTERING,     ///< Reduce-scatter in progress (async)
    };

    /**
     * @brief Get current state of a parameter
     *
     * @param param Parameter to query
     * @return Current state (PARTITIONED, GATHERING, GATHERED, SCATTERING)
     */
    auto get_parameter_state(Tensor* param) const -> ParameterState;

    /**
     * @brief Check if parameter is currently gathered
     *
     * @param param Parameter to check
     * @return true if full parameter is available
     */
    auto is_parameter_gathered(Tensor* param) const -> bool;

    /**
     * @brief Pin parameter in memory (keep gathered)
     *
     * Useful for frequently used parameters (e.g., first/last layer).
     * Pinned parameters are never freed after gathering.
     *
     * @param param Parameter to pin
     */
    auto pin_parameter(Tensor* param) -> void;

    /**
     * @brief Unpin parameter (allow freeing)
     *
     * @param param Parameter to unpin
     */
    auto unpin_parameter(Tensor* param) -> void;

    /**
     * @brief Check if parameter is pinned
     *
     * @param param Parameter to check
     * @return true if parameter is pinned in memory
     */
    auto is_parameter_pinned(Tensor* param) const -> bool;

    // ========================================================================
    // Statistics and Monitoring
    // ========================================================================

    /**
     * @brief Performance statistics
     */
    struct Stats {
        // Communication stats
        size_t total_all_gather_calls{0};
        size_t total_all_gather_bytes{0};
        double avg_all_gather_time_ms{0.0};

        // Memory stats
        size_t peak_gathered_memory_bytes{0};
        size_t current_gathered_memory_bytes{0};
        int num_cached_params{0};

        // Prefetch efficiency
        double prefetch_hit_rate{0.0};  // % of gathers satisfied by prefetch
        int prefetch_queue_depth{0};

        // Performance metrics
        double forward_comm_time_ms{0.0};
        double backward_comm_time_ms{0.0};
        double overlap_efficiency{0.0};  // % of comm hidden by compute
    };

    /**
     * @brief Get performance statistics
     *
     * @return Current performance statistics
     */
    auto get_stats() -> Stats;

    /**
     * @brief Reset performance statistics
     */
    auto reset_stats() -> void;

    /**
     * @brief Get prefetch statistics
     */
    struct PrefetchStats {
        size_t prefetch_queue_size{0};
        size_t prefetched_bytes{0};
        double hit_rate{0.0};           ///< Fraction of gathers that hit prefetch
        double avg_prefetch_time_ms{0.0};
        size_t prefetch_hits{0};
        size_t prefetch_misses{0};
    };

    /**
     * @brief Get prefetch statistics
     *
     * @return Current prefetch statistics
     */
    auto get_prefetch_stats() const -> PrefetchStats;

    // ========================================================================
    // Phase 7: Advanced Optimizations
    // ========================================================================

    /**
     * @brief Update prefetch depth dynamically based on performance metrics
     *
     * Analyzes recent gather operations and adjusts prefetch_depth to:
     * - Maximize communication/compute overlap
     * - Minimize memory pressure
     * - Balance latency hiding vs memory consumption
     */
    auto update_prefetch_depth() -> void;

    /**
     * @brief Calculate optimal prefetch depth based on current metrics
     *
     * @return Recommended prefetch depth (within min/max bounds)
     */
    auto calculate_optimal_prefetch_depth() -> int;

    /**
     * @brief Adjust bucket size dynamically based on communication patterns
     *
     * Analyzes communication efficiency and adjusts bucket size to:
     * - Maximize bandwidth utilization
     * - Minimize communication overhead
     * - Balance message size vs frequency
     */
    auto adjust_bucket_size() -> void;

    /**
     * @brief Calculate optimal bucket size based on communication metrics
     *
     * @return Recommended bucket size in bytes
     */
    auto calculate_optimal_bucket_size() -> size_t;

    /**
     * @brief Check current GPU memory pressure level
     *
     * @return Memory pressure ratio (0.0 = empty, 1.0 = full)
     */
    auto check_memory_pressure() -> double;

    /**
     * @brief Determine if parameter should be offloaded based on memory pressure
     *
     * @param param Parameter to evaluate
     * @return true if parameter should be offloaded to CPU
     */
    auto should_offload_parameter(Tensor* param) -> bool;

    /**
     * @brief Make adaptive offload decision based on current memory state
     *
     * Evaluates memory pressure and selectively offloads parameters to CPU
     * when GPU memory usage exceeds threshold. Uses hysteresis to prevent
     * thrashing between offload/prefetch cycles.
     */
    auto adaptive_offload_decision() -> void;

private:
    // ========================================================================
    // Internal State Structures
    // ========================================================================

    /**
     * @brief State information for a single parameter
     *
     * Tracks the lifecycle of a parameter including whether it's currently
     * gathered, who is using it, and manages temporary buffers.
     */
    struct ParameterInfo {
        Tensor* param;                      ///< Original parameter pointer
        std::string name;                   ///< Parameter name (for debugging)
        size_t size_bytes;                  ///< Size of full parameter

        // Partitioning information
        int owner_rank;                     ///< Rank that owns this partition
        Tensor local_partition;             ///< Local partition (1/N of full parameter)
        size_t partition_offset;            ///< Offset in full parameter
        size_t partition_size;              ///< Size of local partition (in elements)
        std::vector<int64_t> original_shape; ///< Original shape before flattening

        // Gathered state
        Tensor full_param;                  ///< Full parameter (temporarily gathered)
        bool is_gathered{false};            ///< Is the full parameter available?
        int ref_count{0};                   ///< Reference count for shared parameters (non-atomic for simplicity)
        std::chrono::steady_clock::time_point last_access_time;  ///< For LRU eviction

        // Communication handles (using shared_ptr with forward-declared type is OK)
        std::shared_ptr<void> gather_handle;   ///< Handle for ongoing all-gather (opaque)
        std::shared_ptr<void> scatter_handle;  ///< Handle for ongoing reduce-scatter (opaque)

        // Prefetch state
        bool is_prefetching{false};         ///< Is this parameter being prefetched?
        int prefetch_priority{0};           ///< Priority for prefetch scheduling
        int64_t time_until_use_us{0};       ///< Estimated time until needed

        // CPU offload state
        bool partition_on_cpu{false};       ///< Is partition on CPU?
        bool gathered_on_cpu{false};        ///< Is gathered param on CPU?
        std::shared_ptr<core::TransferHandle> offload_handle;  ///< Offload operation handle

        // Module dependency tracking
        std::vector<int> dependent_modules; ///< Modules that use this parameter
        int layer_index{-1};                ///< Layer index in execution graph

        // Memory management
        bool pinned_in_memory{false};       ///< Keep gathered (e.g., first/last layer)
        ParameterState state{ParameterState::PARTITIONED};  ///< Current state

        /**
         * @brief Increment reference count
         */
        auto acquire() -> void {
            ref_count++;
            last_access_time = std::chrono::steady_clock::now();
        }

        /**
         * @brief Decrement reference count
         * @return New reference count value
         */
        auto release() -> int {
            return --ref_count;
        }

        /**
         * @brief Check if parameter can be freed
         * @return true if gathered and no active users
         */
        auto can_free() const -> bool {
            return is_gathered && ref_count == 0;
        }

        /**
         * @brief Get age since last access (for LRU eviction)
         * @return Age in milliseconds
         */
        auto age_ms() const -> int64_t {
            auto now = std::chrono::steady_clock::now();
            return std::chrono::duration_cast<std::chrono::milliseconds>(
                now - last_access_time
            ).count();
        }
    };

    /**
     * @brief Forward pre-hook for parameter gathering
     */
    struct ForwardPreHook {
        Module* module;                     ///< Module this hook is attached to
        std::vector<Tensor*> params;        ///< Parameters used by this module
        std::function<void(Module*, const std::vector<Tensor>&)> hook_fn;  ///< Hook callback
        int hook_id;                        ///< Unique hook identifier
    };

    /**
     * @brief Backward post-hook for gradient scattering
     */
    struct BackwardPostHook {
        Module* module;                     ///< Module this hook is attached to
        std::vector<Tensor*> params;        ///< Parameters used by this module
        std::function<void(Module*, const std::vector<Tensor>&, const std::vector<Tensor>&)> hook_fn;
        int hook_id;                        ///< Unique hook identifier
    };

    // Forward declarations moved to public section

    // ========================================================================
    // Internal State
    // ========================================================================

    Stage3Config stage3_config_;

    /** Module being managed (weak reference to avoid ownership) */
    Module* registered_model_{nullptr};

    // LRU ordering for non-pinned gather buffers. When free_gathered_parameter() drops a
    // parameter's refcount to zero, instead of immediately freeing state.full_param the
    // optimizer keeps it cached and pushes the parameter's pointer to the back of this
    // list. The next gather_parameter() call on the same param can then reuse the buffer
    // without re-running all_gather. When the list grows past Stage3Config::max_cached_params,
    // the front (oldest-released) entry is evicted — its full_param is freed, mirroring the
    // legacy free-on-release behaviour. Pinned parameters never enter this list. Same idea
    // as a fixed-size ring buffer (review item #12) but accommodates heterogeneous param
    // sizes without pre-allocating max_param_size × prefetch_depth bytes up front.
    std::list<Tensor*> lru_release_order_;

    /** Parameter state tracking map */
    std::unordered_map<Tensor*, ParameterInfo> param_states_;
    mutable std::mutex param_states_mutex_;

    /** Prefetch queue and scheduler */
    std::unique_ptr<PrefetchScheduler> prefetch_scheduler_;

    /** Hook handles for cleanup */
    std::vector<ForwardPreHook> forward_hooks_;
    std::vector<BackwardPostHook> backward_hooks_;
    int next_hook_id_{0};

    /** Communication streams for async operations
     * Note: CUDA stream support for gather/scatter operations is planned for future enhancement.
     * Currently operations execute synchronously on the default stream.
     * When stream support is added, uncomment these members:
     * CUDAStream gather_stream_;
     * CUDAStream scatter_stream_;
     */

    /** Performance statistics (internal structure) */
    struct PerformanceStats {
        size_t total_gathers{0};
        size_t total_gather_bytes{0};
        double avg_gather_time_ms{0.0};
        size_t peak_gathered_memory{0};
        size_t current_gathered_memory{0};
        size_t prefetch_hits{0};
        size_t prefetch_misses{0};
    };

    /** Statistics */
    Stats stats_;
    PerformanceStats perf_stats_;
    mutable std::mutex stats_mutex_;

    // ========================================================================
    // Phase 7: Adaptive Optimization State
    // ========================================================================

    /** Adaptive prefetch tracking */
    struct AdaptiveMetrics {
        // Gather timing metrics (rolling window)
        std::deque<double> recent_gather_times_ms;
        std::deque<double> recent_compute_times_ms;

        // Communication overlap metrics
        double actual_overlap_ratio{0.0};
        double target_overlap_ratio{0.8};

        // Bucket sizing metrics
        std::deque<double> recent_comm_efficiency;
        size_t current_bucket_size{100 * 1024 * 1024};

        // Memory pressure tracking
        double current_memory_pressure{0.0};
        size_t last_offload_memory_threshold{0};
        std::chrono::steady_clock::time_point last_memory_check;
        std::chrono::steady_clock::time_point last_offload_decision;

        // Prefetch depth tracking
        int current_prefetch_depth{2};
        int consecutive_improvements{0};
        int consecutive_degradations{0};
    };

    AdaptiveMetrics adaptive_metrics_;
    mutable std::mutex adaptive_mutex_;

    // ========================================================================
    // Internal Methods
    // ========================================================================

    /** Partition all model parameters across ranks */
    auto partition_model_parameters(Module& model) -> void;

    /** Register gather/scatter hooks on all modules */
    auto register_gather_scatter_hooks(Module& model) -> void;

    /** Build execution graph for prefetch scheduling */
    auto build_execution_graph(Module& model) -> void;

    /** All-gather parameter (internal implementation) */
    auto gather_parameter_impl(ParameterInfo& state) -> void;

    /** Free gathered parameter (internal implementation) */
    auto free_gathered_parameter_impl(ParameterInfo& state) -> void;

    /** Check if parameter should be partitioned */
    auto should_partition_parameter(const Tensor& param) const -> bool;

    /** Forward pre-hook callback */
    auto forward_pre_hook(Module* module, const std::vector<Tensor>& inputs) -> void;

    /** Backward post-hook callback */
    auto backward_post_hook(Module* module, const std::vector<Tensor>& inputs,
                           const std::vector<Tensor>& grad_outputs) -> void;

    /** Scatter (reduce-scatter) gradient for a parameter */
    auto scatter_parameter_gradient(Tensor* param) -> void;

    /** Prefetch parameters for next modules. Public wrapper that takes
     *  param_states_mutex_ internally. */
    auto prefetch_next_parameters(Module* current_module) -> void;

    /** Lock-free body of prefetch_next_parameters. Caller must already hold
     *  param_states_mutex_. Used by gather_parameter()'s speculative-prefetch
     *  path so we don't have to drop and re-acquire the mutex mid-call. */
    auto prefetch_next_parameters_locked() -> void;

    /** Get next module in execution order */
    auto get_next_module_in_execution_order(Module* current_module) -> Module*;

    /** Get next parameters in execution order */
    auto get_next_parameters_in_execution_order(const ParameterInfo& state)
        -> std::vector<Tensor*>;

    /** Flatten tensors into contiguous buffer */
    auto flatten_tensors(const std::vector<Tensor>& tensors) -> Tensor;

    /** Unflatten buffer into individual tensors */
    auto unflatten_into(const Tensor& flattened, std::vector<Tensor>& targets) -> void;
};

/**
 * @brief Async handle for non-blocking operations (definition)
 *
 * Placed here after the class definition to avoid forward declaration issues.
 */
struct ZeROStage3Optimizer::AsyncHandle {
    bool ready{false};                  ///< Operation complete?
    Tensor result;                      ///< Result tensor (when ready)
    std::shared_ptr<void> comm_handle;  ///< Underlying communication handle
    std::mutex mutex;                   ///< Thread safety
    std::condition_variable cv;         ///< Notification for completion

    /**
     * @brief Check if operation is complete
     */
    auto is_ready() -> bool {
        std::lock_guard<std::mutex> lock(mutex);
        return ready;
    }

    /**
     * @brief Wait for operation to complete
     */
    auto wait() -> void {
        std::unique_lock<std::mutex> lock(mutex);
        cv.wait(lock, [this] { return ready; });
    }
};

} // namespace optim
} // namespace tenzor
