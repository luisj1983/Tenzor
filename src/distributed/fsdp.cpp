/**
 * @file fsdp.cpp
 * @brief Implementation of Fully Sharded Data Parallel (FSDP)
 *
 * Implements ZeRO-3 style parameter sharding. Parameters are flattened into
 * contiguous buffers per FSDP unit, sharded across ranks, and all-gathered
 * on demand for forward/backward computation.
 *
 * Memory lifecycle per FSDP unit (FULL_SHARD):
 *   idle:      local_shard_ only (1/N of params)
 *   forward:   flat_param_ (full) + local_shard_
 *   backward:  flat_param_ (full) + flat_grad_ (full) + local_shard_
 *   post-bwd:  local_shard_ + grad shard (1/N of grads)
 */

#include "tenzor/distributed/fsdp.hpp"
#include "tenzor/core/dtype.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/transform.hpp"
#include "tenzor/ops/indexing.hpp"
#include <algorithm>
#include <stdexcept>
#include <numeric>
#include <cstring>

#if defined(TENZOR_USE_CUDA)
    #include <cuda_runtime.h>
    #define FSDP_CUDA_CHECK(call) \
        do { \
            cudaError_t err = call; \
            if (err != cudaSuccess) { \
                throw std::runtime_error( \
                    std::string("CUDA error in FSDP: ") + \
                    cudaGetErrorString(err) + \
                    " at " + __FILE__ + ":" + std::to_string(__LINE__) \
                ); \
            } \
        } while(0)
#elif defined(TENZOR_USE_ROCM)
    #include <hip/hip_runtime.h>
    #define cudaStream_t hipStream_t
    #define cudaEvent_t hipEvent_t
    #define cudaStreamCreateWithFlags hipStreamCreateWithFlags
    #define cudaStreamNonBlocking hipStreamNonBlocking
    #define cudaStreamDestroy hipStreamDestroy
    #define cudaStreamSynchronize hipStreamSynchronize
    #define cudaStreamWaitEvent hipStreamWaitEvent
    #define cudaEventCreate hipEventCreate
    #define cudaEventCreateWithFlags hipEventCreateWithFlags
    #define cudaEventDestroy hipEventDestroy
    #define cudaEventRecord hipEventRecord
    #define cudaEventDisableTiming hipEventDisableTiming
    #define cudaSuccess hipSuccess
    #define cudaGetErrorString hipGetErrorString
    #define FSDP_CUDA_CHECK(call) \
        do { \
            hipError_t err = call; \
            if (err != hipSuccess) { \
                throw std::runtime_error( \
                    std::string("HIP error in FSDP: ") + \
                    hipGetErrorString(err) + \
                    " at " + __FILE__ + ":" + std::to_string(__LINE__) \
                ); \
            } \
        } while(0)
#endif

namespace tenzor::distributed {

// ============================================================================
// FSDPUnit Implementation
// ============================================================================

FSDPUnit::FSDPUnit(nn::Module& module, ProcessGroup& pg, const FSDPConfig& config)
    : module_(module), pg_(&pg), config_(config) {

    use_gpu_comm_ = pg_->supports_async_stream();

    // Flatten all parameters into a single contiguous buffer
    flatten_params();

    // Shard the flat buffer across ranks
    shard_params();

    // Initialize communication resources for GPU backends
    init_comm_resources();
}

FSDPUnit::~FSDPUnit() {
    destroy_comm_resources();
}

// ============================================================================
// Parameter flattening and sharding
// ============================================================================

auto FSDPUnit::flatten_params() -> void {
    // Collect all parameters from the module (own parameters only, not submodules
    // that may be wrapped by their own FSDP unit)
    original_params_ = module_.own_parameters();

    if (original_params_.empty()) {
        return;
    }

    // Record shapes and compute total size
    total_numel_ = 0;
    for (const auto& param : original_params_) {
        if (!param) continue;
        auto shape = param->tensor().shape();
        param_shapes_.push_back(std::vector<int64_t>(shape.begin(), shape.end()));
        size_t numel = param->tensor().numel();
        param_numels_.push_back(numel);
        total_numel_ += numel;
    }

    if (total_numel_ == 0) {
        return;
    }

    // Determine device and dtype from the first parameter
    auto device = original_params_[0]->tensor().device();
    auto dtype = original_params_[0]->tensor().dtype();

    // Allocate flat buffer and copy parameter data into it using slice_scatter
    // — device-agnostic, unlike std::memcpy which is invalid for GPU pointers.
    flat_param_ = zeros({static_cast<int64_t>(total_numel_)}, dtype, device);

    size_t offset = 0;
    for (const auto& param : original_params_) {
        if (!param) continue;
        size_t numel = param->tensor().numel();
        Tensor param_flat = param->tensor().reshape({static_cast<int64_t>(numel)}).contiguous();

        flat_param_ = slice_scatter(flat_param_, param_flat, /*dim=*/0,
                                    /*start=*/static_cast<int64_t>(offset),
                                    /*end=*/static_cast<int64_t>(offset + numel));
        offset += numel;
    }

    params_full_ = true;
}

auto FSDPUnit::shard_params() -> void {
    if (total_numel_ == 0) {
        return;
    }

    int ws = pg_->world_size();
    int rank = pg_->rank();

    // Compute shard size: pad total_numel to be divisible by world_size
    shard_numel_ = (total_numel_ + ws - 1) / ws;
    shard_offset_ = rank * shard_numel_;

    // Clamp the last rank's shard if total_numel is not evenly divisible
    size_t actual_end = std::min(shard_offset_ + shard_numel_, total_numel_);
    size_t actual_shard_numel = actual_end - shard_offset_;

    // Extract this rank's shard from the flat buffer
    auto dtype = flat_param_.dtype();
    auto device = flat_param_.device();

    // Start from zeros so the padded tail (when shard extends beyond total_numel)
    // is already zero without touching raw device memory.
    local_shard_ = zeros({static_cast<int64_t>(shard_numel_)}, dtype, device);

    // Copy the valid portion of the shard via slice_scatter (device-agnostic).
    if (actual_shard_numel > 0) {
        Tensor valid_src = slice(flat_param_, /*dim=*/0,
                                 static_cast<int64_t>(shard_offset_),
                                 static_cast<int64_t>(shard_offset_ + actual_shard_numel))
                               .contiguous();
        local_shard_ = slice_scatter(local_shard_, valid_src, /*dim=*/0,
                                     /*start=*/0,
                                     /*end=*/static_cast<int64_t>(actual_shard_numel));
    }

    if (config_.strategy == ShardingStrategy::FULL_SHARD) {
        // Free the full buffer -- we only keep the local shard
        flat_param_ = Tensor{};
        params_full_ = false;
    }

    // Offload to CPU if configured
    if (config_.cpu_offload) {
        offload_to_cpu();
    }
}

auto FSDPUnit::unflatten_params() -> void {
    if (!params_full_ || total_numel_ == 0) {
        return;
    }

    // Write values from flat_param_ back into original parameter Variables
    auto dtype = flat_param_.dtype();
    size_t offset = 0;
    for (size_t i = 0; i < original_params_.size(); ++i) {
        auto& param = original_params_[i];
        if (!param) continue;

        size_t numel = param_numels_[i];
        const auto& shape = param_shapes_[i];

        // Slice the flat buffer and reshape to the parameter's shape (device-agnostic).
        Tensor param_data =
            slice(flat_param_, /*dim=*/0,
                  static_cast<int64_t>(offset),
                  static_cast<int64_t>(offset + numel))
                .contiguous()
                .reshape(shape);

        // Update the parameter's underlying tensor
        param->tensor() = param_data;
        offset += numel;
    }
}

auto FSDPUnit::collect_grads() -> void {
    if (total_numel_ == 0) {
        return;
    }

    auto dtype = original_params_[0]->tensor().dtype();
    auto device = original_params_[0]->tensor().device();

    // Allocate flat gradient buffer with padded size
    int ws = pg_->world_size();
    size_t padded_numel = shard_numel_ * ws;
    flat_grad_ = zeros({static_cast<int64_t>(padded_numel)}, dtype, device);

    // Copy each parameter's gradient into the flat buffer
    size_t offset = 0;
    for (size_t i = 0; i < original_params_.size(); ++i) {
        auto& param = original_params_[i];
        if (!param) continue;

        size_t numel = param_numels_[i];

        if (param->has_grad()) {
            Tensor grad = param->grad().value();
            Tensor grad_flat = grad.reshape({static_cast<int64_t>(numel)}).contiguous();

            flat_grad_ = slice_scatter(flat_grad_, grad_flat, /*dim=*/0,
                                       static_cast<int64_t>(offset),
                                       static_cast<int64_t>(offset + numel));
        }
        offset += numel;
    }
}

auto FSDPUnit::scatter_grads_to_params() -> void {
    if (total_numel_ == 0) {
        return;
    }

    // After reduce-scatter, flat_grad_ contains only this rank's shard
    // of the averaged gradient. Write it back to the corresponding parameters.
    auto dtype = original_params_[0]->tensor().dtype();

    // The gradient shard corresponds to flat buffer offset [shard_offset_, shard_offset_ + shard_numel_)
    // We need to figure out which parameters overlap with this range and write partial grads
    size_t shard_start = shard_offset_;
    size_t shard_end = shard_offset_ + shard_numel_;

    size_t param_offset = 0;
    for (size_t i = 0; i < original_params_.size(); ++i) {
        auto& param = original_params_[i];
        if (!param) continue;

        size_t numel = param_numels_[i];
        size_t param_start = param_offset;
        size_t param_end = param_offset + numel;

        // Check overlap between this parameter and our shard
        size_t overlap_start = std::max(param_start, shard_start);
        size_t overlap_end = std::min(param_end, shard_end);

        if (overlap_start < overlap_end) {
            size_t overlap_numel = overlap_end - overlap_start;
            size_t grad_buf_offset = overlap_start - shard_start;
            size_t param_sub_offset = overlap_start - param_start;

            // Create a gradient tensor for this parameter; only the overlapping
            // portion has valid data. Use slice_scatter for device safety.
            const auto& shape = param_shapes_[i];
            Tensor grad_flat = zeros({static_cast<int64_t>(numel)}, dtype, flat_grad_.device());
            Tensor overlap_src =
                slice(flat_grad_, /*dim=*/0,
                      static_cast<int64_t>(grad_buf_offset),
                      static_cast<int64_t>(grad_buf_offset + overlap_numel))
                    .contiguous();
            grad_flat = slice_scatter(grad_flat, overlap_src, /*dim=*/0,
                                      static_cast<int64_t>(param_sub_offset),
                                      static_cast<int64_t>(param_sub_offset + overlap_numel));

            param->set_grad(grad_flat.reshape(shape));
        }

        param_offset += numel;
    }
}

// ============================================================================
// All-gather and reduce-scatter
// ============================================================================

auto FSDPUnit::all_gather_params() -> void {
    if (params_full_ || total_numel_ == 0) {
        return;
    }

    // Reload from CPU if offloaded
    if (offloaded_) {
        reload_from_cpu();
    }

    int ws = pg_->world_size();
    auto dtype = local_shard_.dtype();
    auto device = local_shard_.device();

    // Mixed precision: cast shard to Float16 before communication for 2x bandwidth savings
    Tensor comm_shard = local_shard_;
    DType comm_dtype = dtype;
    if (config_.mixed_precision && dtype == DType::Float32) {
        comm_shard = local_shard_.to(DType::Float16);
        comm_dtype = DType::Float16;
    }

    // Allocate output tensors for all-gather (one per rank)
    std::vector<Tensor> gathered(ws);
    for (int i = 0; i < ws; ++i) {
        gathered[i] = empty({static_cast<int64_t>(shard_numel_)}, comm_dtype, device);
    }

    // All-gather: each rank sends its shard, receives all shards
    pg_->all_gather(comm_shard, gathered);

    // Cast back to original dtype after communication
    if (config_.mixed_precision && dtype == DType::Float32) {
        for (auto& g : gathered) {
            g = g.to(DType::Float32);
        }
    }

    // Concatenate all shards into the flat parameter buffer
    // Total size is shard_numel_ * ws (may be padded beyond total_numel_)
    size_t padded_numel = shard_numel_ * ws;
    flat_param_ = zeros({static_cast<int64_t>(padded_numel)}, dtype, device);

    for (int i = 0; i < ws; ++i) {
        int64_t off = static_cast<int64_t>(i) * static_cast<int64_t>(shard_numel_);
        flat_param_ = slice_scatter(flat_param_, gathered[i].contiguous(),
                                    /*dim=*/0, /*start=*/off,
                                    /*end=*/off + static_cast<int64_t>(shard_numel_));
    }

    params_full_ = true;

    // Write full parameters back to original parameter Variables
    unflatten_params();
}

auto FSDPUnit::free_full_params() -> void {
    if (!params_full_ || config_.strategy != ShardingStrategy::FULL_SHARD) {
        return;
    }

    // Free the full parameter buffer
    flat_param_ = Tensor{};
    params_full_ = false;

    // Offload local shard to CPU if configured
    if (config_.cpu_offload) {
        offload_to_cpu();
    }
}

auto FSDPUnit::reduce_scatter_grads() -> void {
    if (total_numel_ == 0) {
        return;
    }

    int ws = pg_->world_size();

    // Collect gradients from parameters into flat buffer
    collect_grads();

    auto dtype = flat_grad_.dtype();
    auto device = flat_grad_.device();

    // Split flat gradient into world_size chunks for reduce-scatter input
    std::vector<Tensor> grad_chunks(ws);
    for (int i = 0; i < ws; ++i) {
        int64_t off = static_cast<int64_t>(i) * static_cast<int64_t>(shard_numel_);
        grad_chunks[i] = slice(flat_grad_, /*dim=*/0, off,
                               off + static_cast<int64_t>(shard_numel_))
                             .contiguous();
    }

    // Output: this rank's reduced gradient shard
    Tensor grad_shard = empty({static_cast<int64_t>(shard_numel_)}, dtype, device);

    // Reduce-scatter: sum across ranks, each rank gets its shard
    pg_->reduce_scatter(grad_chunks, grad_shard, ReduceOp::SUM);

    // Divide by world_size to compute average
    if (ws > 1) {
        grad_shard = grad_shard / static_cast<float>(ws);
    }

    // Store reduced shard back into flat_grad_ for scatter_grads_to_params
    flat_grad_ = grad_shard;

    // Write gradient shards back to parameters
    scatter_grads_to_params();

    // Re-shard parameters (they were all-gathered for backward)
    if (config_.strategy == ShardingStrategy::FULL_SHARD) {
        free_full_params();
    }
}

// ============================================================================
// CPU offloading
// ============================================================================

auto FSDPUnit::offload_to_cpu() -> void {
    if (offloaded_ || total_numel_ == 0) {
        return;
    }

    // Use Tensor::to() for a device-correct host transfer (handles any backend).
    cpu_shard_ = local_shard_.to(Device::cpu());

    // Free GPU shard
    local_shard_ = Tensor{};
    offloaded_ = true;
}

auto FSDPUnit::reload_from_cpu() -> void {
    if (!offloaded_ || total_numel_ == 0) {
        return;
    }

    // Determine target device from original parameters
    auto device = original_params_[0]->tensor().device();

    local_shard_ = cpu_shard_.to(device);

    // Free CPU copy
    cpu_shard_ = Tensor{};
    offloaded_ = false;
}

// ============================================================================
// Communication resource management
// ============================================================================

auto FSDPUnit::init_comm_resources() -> void {
#if defined(TENZOR_USE_CUDA) || defined(TENZOR_USE_ROCM)
    if (!use_gpu_comm_) {
        return;
    }

    cudaStream_t stream = nullptr;
    FSDP_CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    comm_stream_ = static_cast<void*>(stream);

    cudaEvent_t event = nullptr;
    FSDP_CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    comm_event_ = static_cast<void*>(event);
#else
    (void)use_gpu_comm_;
#endif
}

auto FSDPUnit::destroy_comm_resources() -> void {
#if defined(TENZOR_USE_CUDA) || defined(TENZOR_USE_ROCM)
    if (!use_gpu_comm_) {
        return;
    }

    if (comm_event_) {
        cudaEventDestroy(static_cast<cudaEvent_t>(comm_event_));
        comm_event_ = nullptr;
    }

    if (comm_stream_) {
        cudaStreamDestroy(static_cast<cudaStream_t>(comm_stream_));
        comm_stream_ = nullptr;
    }
#endif
}

// ============================================================================
// FullyShardedDataParallel Implementation
// ============================================================================

FullyShardedDataParallel::FullyShardedDataParallel(
    nn::Module& module,
    ProcessGroup& pg,
    const FSDPConfig& config
) : module_(module), pg_(&pg), config_(config) {

    // Apply auto-wrap policy to identify FSDP units
    apply_auto_wrap();

    // Register forward hooks for automatic all-gather/free
    register_hooks();
}

FullyShardedDataParallel::~FullyShardedDataParallel() = default;

auto FullyShardedDataParallel::apply_auto_wrap() -> void {
    // Walk the module tree and wrap submodules that exceed the parameter threshold
    auto submodules = module_.get_submodules();

    bool wrapped_any = false;
    for (auto& [name, submodule] : submodules) {
        if (!submodule) continue;

        size_t param_count = count_params(*submodule);
        if (param_count >= config_.auto_wrap_min_params) {
            wrap_module(*submodule);
            wrapped_any = true;
        }
    }

    // If no submodules were wrapped (small model or no submodules),
    // wrap the root module itself as a single FSDP unit
    if (!wrapped_any) {
        wrap_module(module_);
    }
}

auto FullyShardedDataParallel::wrap_module(nn::Module& module) -> void {
    units_.push_back(std::make_unique<FSDPUnit>(module, *pg_, config_));
}

auto FullyShardedDataParallel::count_params(nn::Module& module) const -> size_t {
    size_t count = 0;
    auto params = module.own_parameters();
    for (const auto& p : params) {
        if (p && p->requires_grad()) {
            count += p->tensor().numel();
        }
    }
    return count;
}

auto FullyShardedDataParallel::forward(const Variable& input) -> Variable {
    // All-gather parameters for all units before forward
    for (auto& unit : units_) {
        unit->all_gather_params();
    }

    // Execute the wrapped module's forward pass
    auto output = module_.forward(input);

    // Free non-local shards after forward (FULL_SHARD only)
    if (config_.strategy == ShardingStrategy::FULL_SHARD) {
        for (auto& unit : units_) {
            unit->free_full_params();
        }
    }

    return output;
}

auto FullyShardedDataParallel::finalize_backward() -> void {
    // For FULL_SHARD: all-gather params again (needed for gradient computation
    // with respect to weights, which requires the full weight values)
    if (config_.strategy == ShardingStrategy::FULL_SHARD) {
        for (auto& unit : units_) {
            unit->all_gather_params();
        }
    }

    // Reduce-scatter gradients for each unit
    for (auto& unit : units_) {
        unit->reduce_scatter_grads();
    }
}

auto FullyShardedDataParallel::summon_full_params() -> void {
    for (auto& unit : units_) {
        unit->all_gather_params();
    }
}

auto FullyShardedDataParallel::release_full_params() -> void {
    for (auto& unit : units_) {
        unit->free_full_params();
    }
}

auto FullyShardedDataParallel::total_params() const -> size_t {
    size_t total = 0;
    for (const auto& unit : units_) {
        total += unit->total_numel();
    }
    return total;
}

auto FullyShardedDataParallel::sharded_param_bytes() const -> size_t {
    size_t total = 0;
    for (const auto& unit : units_) {
        if (unit->total_numel() > 0) {
            // Each rank holds shard_numel elements
            // flat_param may be empty if sharded; use module params for dtype
            auto own_params = unit->module().own_parameters();
            DType dtype = own_params.empty() ? DType::Float32
                                              : own_params[0]->tensor().dtype();
            total += unit->shard_numel() * dtype_size(dtype);
        }
    }
    return total;
}

auto FullyShardedDataParallel::register_hooks() -> void {
    // Register forward pre-hooks on each FSDP unit's module for automatic
    // all-gather when the module's forward is called
    for (size_t i = 0; i < units_.size(); ++i) {
        auto* unit_ptr = units_[i].get();

        unit_ptr->module().register_forward_pre_hook(
            [unit_ptr](nn::Module* /*module*/, const Variable& /*input*/) {
                unit_ptr->all_gather_params();
            }
        );

        // Post-hook to free full params after forward
        if (config_.strategy == ShardingStrategy::FULL_SHARD) {
            unit_ptr->module().register_forward_post_hook(
                [unit_ptr](nn::Module* /*module*/, const Variable& /*input*/,
                          const Variable& /*output*/) {
                    unit_ptr->free_full_params();
                }
            );
        }
    }
}

} // namespace tenzor::distributed
