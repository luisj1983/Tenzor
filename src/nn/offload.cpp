/**
 * @file offload.cpp
 * @brief Implementation of Parameter Offloading API for ZeRO Phase 2
 */

#include "tenzor/nn/offload.hpp"
#include "tenzor/backend/loader.hpp"    // try_get_backend (F076: multi-backend gating)
#include "tenzor/ops/math.hpp"          // for clamp, abs, round, gt, div, mul (G3)
#include "tenzor/ops/reduction.hpp"     // for tenzor::max (G3)
#include "tenzor/ops/creation.hpp"      // for zeros_like, ones_like (G3)
#include "tenzor/ops/indexing.hpp"      // for where (G3 device-side zero guard)
#include "tenzor/utils/log.hpp"         // TENZOR_LOG_WARN (D.1)
#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <iostream>

namespace tenzor {
namespace nn {

// Global offload context (for offload_param function)
namespace {
    OffloadContext* g_offload_context = nullptr;

    // A device is a compute accelerator if it is anything other than the CPU
    // host. Offload / eviction / memory-pressure logic must apply to EVERY
    // accelerator backend (CUDA, ROCm, OneAPI, Vulkan, MPS), not just CUDA —
    // otherwise the offload paths silently no-op on non-CUDA hardware while
    // reporting success (F076).
    inline bool is_accelerator_device(const Device& d) {
        return d.type != Device::Type::CPU;
    }

    // Pick the accelerator device a ComputeContext should stage CPU tensors
    // onto. Prefer the device of whichever managed tensor is already
    // accelerator-resident (the "actual" device of the workload); otherwise
    // fall back to the first available accelerator backend. Never introduces a
    // CPU compute fallback — ComputeContext exists to run on an accelerator.
    inline Device pick_compute_device(const std::vector<Tensor*>& tensors) {
        for (auto* t : tensors) {
            if (t && is_accelerator_device(t->device())) {
                return t->device();
            }
        }
        for (auto type : {Device::Type::CUDA, Device::Type::ROCm,
                          Device::Type::OneAPI, Device::Type::Vulkan,
                          Device::Type::MPS}) {
            if (auto* b = ::tenzor::try_get_backend(type); b && b->is_available()) {
                return Device{type, 0};
            }
        }
        return Device::cuda(0);  // last-resort default (e.g. CUDA-only build)
    }
}

// ============================================================================
// OffloadContext Implementation
// ============================================================================

OffloadContext::OffloadContext(Module& model, const Config& config)
    : model_(model), config_(config), enabled_(false) {

    // Adopt the caller's TransferEngine when provided so the host-pinned pool is shared
    // with whichever other subsystem (typically a core::OffloadEngine driving Stage 1+
    // optimizer-state offload) built it. Without sharing, each component keeps its own
    // ~512 MB – 2 GB pinned pool live for the duration of training, doubling host RAM
    // pressure on systems where it's already the constraining resource.
    if (config_.shared_transfer_engine) {
        transfer_engine_ = config_.shared_transfer_engine;
    } else {
        core::TransferEngine::Config engine_config;
        engine_config.num_streams = 4;  // Multiple streams for parallelism
        engine_config.use_pinned_memory = true;
        engine_config.pinned_pool_size = 512 * 1024 * 1024;  // 512 MB pinned pool
        transfer_engine_ = std::make_shared<core::TransferEngine>(engine_config);
    }

    // Initialize memory manager
    core::MemoryManager::Config mem_config;
    mem_config.cpu_memory_limit = config_.cpu_memory_limit;
    mem_config.gpu_memory_limit = config_.gpu_memory_limit;
    mem_config.eviction_threshold = 0.9f;  // Evict at 90% capacity
    mem_config.track_statistics = config_.enable_statistics;
    memory_manager_ = std::make_shared<core::MemoryManager>(mem_config);

    // Build layer order for sequential processing
    build_layer_order();

    // Collect all parameters and gradients
    collect_tensors();

    // Register hooks (even if no parameters, hooks are safe to register)
    register_hooks();
}

OffloadContext::~OffloadContext() {
    // Disable hooks first so no new transfers are issued during teardown.
    disable();

    // Unregister per-parameter backward hooks before draining any pending
    // transfers — otherwise a backward() invoked from a destructor in
    // user code (rare but possible during shutdown) could re-enter the
    // hook and queue a transfer after we've already drained.
    for (auto& ph : param_grad_hooks_) {
        if (ph.param) {
            ph.param->unregister_hook(ph.hook_id);
        }
    }
    param_grad_hooks_.clear();

    // Drain any in-flight async transfers (per-tensor wait) and restore offloaded
    // tensors to their original device on best-effort basis. Tensor::Storage is
    // IntrusiveRefCounted so the storage referenced by *tensor_ptr survives the
    // map's destruction either way -- but if we leave is_offloaded tensors
    // pointing at info.cpu_copy, the user's Variable silently sees CPU-resident
    // data even though it was originally on GPU. That's a correctness bug
    // (downstream ops dispatch to the wrong backend).
    //
    // We do this BEFORE engine.synchronize() so that finalize_pending() can use
    // the engine's per-handle wait path; the final synchronize() catches any
    // restore-side cpu_to_gpu DMAs that we issue here.
    if (transfer_engine_) {
        try {
            std::lock_guard<std::mutex> lock(tensor_map_mutex_);

            // Drain pending handles first.
            drain_all_pending();

            // Restore is_offloaded tensors to original_device. On alloc failure,
            // leave *tensor_ptr pointing at info.cpu_copy -- the IntrusiveRefCount
            // on Storage keeps the data alive (no UAF), but the device will be
            // wrong; we log so the user knows.
            for (auto& [tensor_ptr, info] : tensor_map_) {
                if (!info.is_offloaded || tensor_ptr == nullptr) continue;
                try {
                    Tensor restored = transfer_engine_->cpu_to_gpu(
                        info.cpu_copy, info.original_device);
                    // Phase C (C3): cast back to original dtype if cpu_copy was quantized.
                    // Audit G3: Int8WithScale dequant — widen + multiply by scale.
                    if (info.offload_dtype_used == DType::Int8 && info.quant_scale != 0.0f) {
                        Tensor widened = restored.to(info.original_dtype);
                        restored = widened * info.quant_scale;
                    } else if (info.original_dtype != info.offload_dtype_used) {
                        restored = restored.to(info.original_dtype);
                    }
                    *tensor_ptr = restored;
                    info.is_offloaded = false;
                } catch (const std::exception& e) {
                    // Audit I.4: route to unified logger.
                    TENZOR_LOG_WARN("OffloadContext: failed to restore offloaded "
                                    "tensor to {} on shutdown: {} -- tensor will remain on CPU",
                                    info.original_device.to_string(), e.what());
                }
            }
        } catch (const std::exception& e) {
            TENZOR_LOG_WARN("OffloadContext: error draining pending transfers on "
                            "shutdown: {}", e.what());
        }

        // Final sync to catch the restore-side DMAs (and anything finalize_pending
        // re-issued).
        transfer_engine_->synchronize();
    }
}

auto OffloadContext::enable() -> void {
    enabled_.store(true, std::memory_order_release);

    // Immediately offload all parameters to CPU when offloading is enabled
    if (config_.offload_parameters) {
        auto all_params = model_.parameters();

        for (auto& param_ptr : all_params) {
            if (param_ptr) {
                Tensor* tensor_ptr = &(param_ptr->tensor());

                // Only offload if resident on an accelerator (any non-CPU
                // backend: CUDA/ROCm/OneAPI/Vulkan/MPS).
                if (is_accelerator_device(tensor_ptr->device())) {
                    // Add to tensor map if not already there
                    {
                        std::lock_guard<std::mutex> lock(tensor_map_mutex_);
                        if (tensor_map_.find(tensor_ptr) == tensor_map_.end()) {
                            initialize_tensor_info(tensor_ptr, &model_, param_ptr);
                        }
                    }

                    // Issue an async offload — the call returns immediately with an
                    // in-flight TransferHandle stashed in TensorInfo.
                    offload_tensor(tensor_ptr);
                }
            }
        }

        // The async path lets multiple offloads overlap on the TransferEngine's streams,
        // but enable() should *feel* synchronous to outside callers (they expect get_stats
        // afterwards to report all params offloaded). Drain all pending transfers here so
        // the post-enable state is fully committed.
        std::lock_guard<std::mutex> lock(tensor_map_mutex_);
        drain_all_pending();
    }
}

auto OffloadContext::disable() -> void {
    enabled_.store(false, std::memory_order_release);
    std::lock_guard<std::mutex> lock(offloaded_gradients_mutex_);
    offloaded_gradients_.clear();
}

auto OffloadContext::is_enabled() const -> bool {
    return enabled_.load(std::memory_order_acquire);
}

auto OffloadContext::get_stats() -> OffloadStats {
    OffloadStats stats;

    // Gradient offload is now driven by per-parameter backward hooks
    // (see register_param_grad_hooks()) plus the module-level
    // backward_post_hook, so get_stats no longer needs to walk parameters
    // and trigger offloads itself. This makes stats observation pure —
    // calling get_stats() does not mutate the tensor map or kick off
    // transfers.

    // Update internal statistics first
    update_stats();

    // Copy atomic values to stats structure
    stats.total_prefetch_count = stats_.total_prefetches.load(std::memory_order_relaxed);
    stats.total_offload_count = stats_.total_offloads.load(std::memory_order_relaxed);
    stats.peak_gpu_memory_mb = static_cast<double>(stats_.peak_gpu_memory.load(std::memory_order_relaxed)) / (1024.0 * 1024.0);
    stats.current_cpu_memory_mb = static_cast<double>(stats_.current_cpu_memory.load(std::memory_order_relaxed)) / (1024.0 * 1024.0);

    // Calculate average transfer time
    size_t transfer_count = stats_.transfer_count.load(std::memory_order_relaxed);
    if (transfer_count > 0) {
        double total_time = stats_.total_transfer_time_ms.load(std::memory_order_relaxed);
        stats.avg_transfer_time_ms = total_time / transfer_count;
    }

    // Count currently offloaded tensors (separate parameters and gradients).
    // Drain any in-flight async transfers first so the snapshot is consistent — otherwise
    // a user calling get_stats() between forward_post_hook(N) and forward_post_hook(N+1)
    // would see layer N's offload as not-yet-applied even though it's effectively committed.
    std::lock_guard<std::mutex> lock(tensor_map_mutex_);
    drain_all_pending();
    for (const auto& [tensor_ptr, info] : tensor_map_) {
        if (info.is_offloaded) {
            if (info.is_gradient) {
                stats.num_gradients_offloaded++;
            } else {
                stats.num_parameters_offloaded++;
            }
        }
    }

    // Gradient offload is tracked separately (keyed by parameter identity)
    // because it is driven from the per-parameter backward hook, where the
    // gradient value — but not a stable Tensor* in tensor_map_ — is available.
    {
        std::lock_guard<std::mutex> grad_lock(offloaded_gradients_mutex_);
        stats.num_gradients_offloaded += offloaded_gradients_.size();
    }

    return stats;
}

auto OffloadContext::reset_stats() -> void {
    stats_.total_prefetches.store(0, std::memory_order_relaxed);
    stats_.total_offloads.store(0, std::memory_order_relaxed);
    stats_.peak_gpu_memory.store(0, std::memory_order_relaxed);
    stats_.current_cpu_memory.store(0, std::memory_order_relaxed);
    stats_.total_transfer_time_ms.store(0.0, std::memory_order_relaxed);
    stats_.transfer_count.store(0, std::memory_order_relaxed);
}

auto OffloadContext::get_gpu_memory_usage() const -> size_t {
    // Query the accelerator the model actually lives on, not a hardcoded CUDA.
    return memory_manager_->get_memory_usage(config_.target_device.type);
}

auto OffloadContext::get_cpu_memory_usage() const -> size_t {
    return stats_.current_cpu_memory.load(std::memory_order_relaxed);
}

// ============================================================================
// Private: Initialization Methods
// ============================================================================

auto OffloadContext::register_hooks() -> void {
    // Register hooks on ALL modules (root and all submodules).
    // With the updated Module::forward() that automatically calls hooks,
    // each module's hooks will fire when its forward() is called,
    // enabling true layer-by-layer offloading.
    register_hooks_recursive(&model_);

    // Subscribe per-parameter backward hooks via Variable::register_hook so
    // the slow→fast (CPU→GPU) upload of each parameter is kicked off as
    // soon as its gradient lands during backward(). This replaces the
    // previous lazy-offload-in-get_stats() workaround.
    register_param_grad_hooks();
}

auto OffloadContext::register_param_grad_hooks() -> void {
    auto all_params = model_.parameters();
    param_grad_hooks_.reserve(all_params.size());

    for (auto& param_ptr : all_params) {
        if (!param_ptr || !param_ptr->requires_grad()) {
            continue;
        }

        // Capture a raw `this` pointer plus a weak_ptr to the Variable.
        // The OffloadContext is required (per its design — see hook
        // contract above) to outlive the model whose modules it has
        // wired pre/post hooks onto, so `this` is valid for as long as
        // the hook can fire. The weak_ptr lets us detect a parameter
        // that was destroyed (rare, but possible if the user mutates
        // module structure after constructing the OffloadContext).
        OffloadContext* self = this;
        std::weak_ptr<Variable> param_weak = param_ptr;

        size_t hook_id = param_ptr->register_hook(
            [self, param_weak](const Tensor& grad) -> Tensor {
                if (!self->is_enabled()) {
                    return grad;
                }

                auto param = param_weak.lock();
                if (!param) {
                    // Parameter was destroyed; nothing to schedule.
                    return grad;
                }

                Tensor* param_tensor_ptr = &(param->tensor());

                // Track the parameter's data tensor for offload bookkeeping
                // if it wasn't already (CPU-start models populate the map
                // lazily through the forward_pre_hook path; here we ensure
                // the entry exists so prefetch_tensor() can find it).
                {
                    std::lock_guard<std::mutex> lock(self->tensor_map_mutex_);
                    if (self->tensor_map_.find(param_tensor_ptr) == self->tensor_map_.end()) {
                        self->initialize_tensor_info(param_tensor_ptr, &self->model_, param);
                    }
                }

                // The parameter's gradient has just been computed, so the
                // parameter is no longer needed for the rest of this backward
                // pass (a leaf weight is read only by its own layer's backward,
                // which has already run by the time its AccumulateGrad hook
                // fires). When parameter offload is enabled, push it back to the
                // host now to free device memory — the next forward's
                // forward_pre_hook prefetches it back. Without this the param
                // stayed resident and num_parameters_offloaded never grew (the
                // PrefetchForOptimizer / FullTrainingLoop regressions). When
                // parameter offload is disabled we instead prefetch so the GPU
                // copy is ready for the next forward.
                if (self->config_.offload_parameters) {
                    self->offload_tensor(param_tensor_ptr);
                } else {
                    self->prefetch_tensor(param_tensor_ptr);
                }

                // Gradient offload. This per-parameter hook fires the instant
                // the leaf's gradient is computed (BackwardEngine applies hooks
                // immediately before accumulating into param->grad()), so the
                // gradient value is available here as `grad`. The module-level
                // backward_post_hook cannot offload it: it runs via the
                // ModuleHookFunction wrapping the module OUTPUT, which in
                // reverse-mode executes BEFORE the layer's weight gradients are
                // produced — param->grad() is still empty there, so no gradient
                // was ever offloaded (num_gradients_offloaded stayed 0).
                //
                // We must NOT publish into param->grad() here: the engine
                // inspects grad_.has_value() right after this hook returns and
                // would then ADD grad_to_apply on top (doubling the gradient).
                // Instead record the offload against the gradient's own storage
                // identity, copying the payload to host. This frees nothing the
                // engine still owns (the engine keeps its own grad handle), so
                // there is no aliasing with the post-hook accumulation.
                if (self->config_.offload_gradients &&
                    is_accelerator_device(grad.device())) {
                    self->record_gradient_offload(param.get(), grad);
                }

                // Return the gradient unchanged — the optimizer consumes the
                // same gradient values; offload keeps a host copy for memory
                // accounting without perturbing the value the engine stores.
                return grad;
            }
        );

        param_grad_hooks_.push_back(ParamHook{param_ptr, hook_id});
    }
}

auto OffloadContext::record_gradient_offload(const Variable* param,
                                             const Tensor& grad) -> void {
    if (!param) return;
    // The actual offload: materialise a host-resident copy of the gradient.
    // This frees nothing the BackwardEngine still owns (it keeps its own
    // gradient handle), so there is no aliasing with the post-hook gradient
    // accumulation that runs immediately after this returns.
    Tensor host_copy = grad.to(Device::cpu());
    {
        std::lock_guard<std::mutex> lock(offloaded_gradients_mutex_);
        offloaded_gradients_[param] = std::move(host_copy);
    }
    stats_.total_offloads.fetch_add(1, std::memory_order_relaxed);
    stats_.transfer_count.fetch_add(1, std::memory_order_relaxed);
}

auto OffloadContext::register_hooks_recursive(Module* module) -> void {
    if (!module) return;

    // Register hooks on this module
    // Note: hooks now receive input/output but offload only needs module reference
    module->register_forward_pre_hook([this](Module* m, const Variable&) {
        this->forward_pre_hook(m);
    });

    module->register_forward_post_hook([this](Module* m, const Variable&, const Variable&) {
        this->forward_post_hook(m);
    });

    module->register_backward_pre_hook([this](Module* m, const Variable&) {
        this->backward_pre_hook(m);
    });

    module->register_backward_post_hook([this](Module* m, const Variable&, const Variable&) {
        this->backward_post_hook(m);
    });

    // Recursively register on all submodules
    for (auto& [name, submodule] : module->get_submodules()) {
        register_hooks_recursive(submodule.get());
    }
}

auto OffloadContext::build_layer_order() -> void {
    // Build sequential layer order by traversing model hierarchy
    layer_order_.clear();
    layer_indices_.clear();

    // Recursive lambda to traverse module tree
    std::function<void(Module*)> traverse = [&](Module* module) {
        if (!module) return;

        // Add this module to ordering
        int index = static_cast<int>(layer_order_.size());
        layer_order_.push_back(module);
        layer_indices_[module] = index;

        // Traverse submodules recursively
        for (auto& [name, submodule] : module->get_submodules()) {
            if (submodule) {
                traverse(submodule.get());
            }
        }
    };

    traverse(&model_);
}

auto OffloadContext::collect_tensors() -> void {
    std::lock_guard<std::mutex> lock(tensor_map_mutex_);

    // Collect parameters from each layer in order
    // Only process leaf modules (those without submodules) to avoid double-counting
    for (Module* layer : layer_order_) {
        if (!layer) continue;

        // Skip non-leaf modules (those with submodules)
        if (!layer->get_submodules().empty()) {
            continue;
        }

        // Get parameters from this leaf layer
        auto params = layer->parameters();

        for (auto& param_ptr : params) {
            if (param_ptr) {
                // Get the underlying tensor
                Tensor* tensor_ptr = &(param_ptr->tensor());

                // Skip if already tracked
                if (tensor_map_.find(tensor_ptr) != tensor_map_.end()) {
                    continue;
                }

                // Initialize tracking info with correct owning layer
                initialize_tensor_info(tensor_ptr, layer, param_ptr);
            }
        }
    }
}

auto OffloadContext::initialize_tensor_info(Tensor* tensor, Module* layer,
                                            std::shared_ptr<Variable> owner) -> void {
    if (!tensor) return;

    TensorInfo info;
    info.tensor = tensor;
    info.owner = std::move(owner);
    // For CPU-start models, use target_device for computation; otherwise use tensor's current device
    info.original_device = (tensor->device().type == Device::Type::CPU)
                           ? config_.target_device
                           : tensor->device();
    info.is_offloaded = false;
    info.is_pinned = false;
    info.use_count = 0;
    info.priority = OffloadPriority::NORMAL;
    info.size_bytes = tensor->numel() * tensor->dtype_size();
    info.owning_layer = layer;

    // Find first and last LEAF modules (those with no submodules)
    int first_leaf_idx = -1;
    int last_leaf_idx = -1;
    for (size_t i = 0; i < layer_order_.size(); ++i) {
        if (layer_order_[i]->get_submodules().empty()) {
            if (first_leaf_idx == -1) first_leaf_idx = i;
            last_leaf_idx = i;
        }
    }

    // Check if this layer should be pinned
    int layer_idx = layer_indices_[layer];
    bool is_first = (layer_idx == first_leaf_idx);
    bool is_last = (layer_idx == last_leaf_idx);

    if ((is_first && config_.pin_first_layer) || (is_last && config_.pin_last_layer)) {
        info.is_pinned = true;
    }

    // Add to tracking map
    tensor_map_[tensor] = std::move(info);

    // Register with memory manager
    memory_manager_->register_tensor(tensor);
}

// ============================================================================
// Private: Offload/Prefetch Operations
// ============================================================================

auto OffloadContext::offload_layer(Module* layer) -> void {
    if (!is_enabled()) return;

    // Get only this layer's own parameters (not submodules')
    // Each submodule's hooks handle their own parameters
    auto params = layer->own_parameters();

    for (auto& param_ptr : params) {
        if (param_ptr) {
            Tensor* tensor_ptr = &(param_ptr->tensor());
            offload_tensor(tensor_ptr);
        }
    }
}

auto OffloadContext::prefetch_layer(Module* layer) -> void {
    if (!is_enabled()) return;

    // Get only this layer's own parameters (not submodules')
    // Each submodule's hooks handle their own parameters
    auto params = layer->own_parameters();

    for (auto& param_ptr : params) {
        if (param_ptr) {
            Tensor* tensor_ptr = &(param_ptr->tensor());
            prefetch_tensor(tensor_ptr);
        }
    }
}

auto OffloadContext::drain_all_pending() -> void {
    // Caller must hold tensor_map_mutex_. Walks every tracked tensor and forces finalization
    // of any in-flight async transfer so subsequent state reads (e.g. get_stats) observe the
    // committed state rather than the in-progress one.
    for (auto& [tensor_ptr, info] : tensor_map_) {
        if (info.pending_handle.is_valid()) {
            finalize_pending(info, tensor_ptr);
        }
    }
}

auto OffloadContext::offload_single_tensor(Tensor* tensor_ptr,
                                           OffloadPriority priority) -> bool {
    // Public wrapper around the private offload_tensor() so
    // tenzor::nn::offload_param can drive a manual eviction without exposing
    // the whole private API. The caller-supplied priority is forwarded so the
    // per-tensor OffloadPriority is honored end-to-end: HIGH keeps the tensor
    // on the device, LOW bypasses the size threshold to force an offload, and
    // NORMAL follows the standard rules.
    return offload_tensor(tensor_ptr, priority, /*priority_explicit=*/true);
}

auto OffloadContext::finalize_completed_offloads() -> size_t {
    // Non-blocking variant of drain_all_pending: only commits transfers that
    // already finished in the background (TransferHandle::is_ready() == true).
    // Critical for releasing GPU memory after forward_post_hook-issued offloads
    // -- without this, the GPU storage stays alive until the same tensor is
    // touched next step.
    if (!is_enabled()) return 0;

    std::lock_guard<std::mutex> lock(tensor_map_mutex_);
    size_t finalized = 0;
    for (auto& [tensor_ptr, info] : tensor_map_) {
        if (info.pending_handle.is_valid() && info.pending_handle.is_ready()) {
            finalize_pending(info, tensor_ptr);
            ++finalized;
        }
    }
    return finalized;
}

auto OffloadContext::finalize_pending(TensorInfo& info, Tensor* tensor_ptr) -> void {
    if (!info.pending_handle.is_valid() || tensor_ptr == nullptr) {
        return;
    }

    auto start = std::chrono::high_resolution_clock::now();
    Tensor result = info.pending_handle.get_tensor();  // implicit wait if not ready
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();

    if (info.pending_is_offload) {
        // Commit the GPU→CPU offload: keep the CPU copy and free the GPU storage by
        // rebinding *tensor_ptr.
        info.cpu_copy = result;
        *tensor_ptr = info.cpu_copy;
        info.is_offloaded = true;
        stats_.current_cpu_memory.fetch_add(info.size_bytes, std::memory_order_relaxed);
    } else {
        // Commit the CPU→GPU prefetch.
        // Phase C (C3): if the cpu_copy was a quantized representation (Half/BF16),
        // cast back on-GPU to the original dtype so downstream ops dispatch
        // correctly. Cast happens after the DMA completes -- the bytes-on-the-wire
        // savings still apply. is_pinned and CRITICAL tensors had cast skipped at
        // offload time, so original_dtype == offload_dtype_used and no cast runs.
        // Audit G3: Int8WithScale needs an extra step — widen Int8→target then
        // multiply by the recorded scale to recover the fp32 (or original) value.
        if (info.offload_dtype_used == DType::Int8 && info.quant_scale != 0.0f) {
            Tensor widened = result.to(info.original_dtype);
            *tensor_ptr = widened * info.quant_scale;
        } else if (info.original_dtype != info.offload_dtype_used) {
            *tensor_ptr = result.to(info.original_dtype);
        } else {
            *tensor_ptr = result;
        }
        info.is_offloaded = false;
        // current_cpu_memory was already decremented when the prefetch was issued.
    }

    // Stats: only count *wait* time as user-visible transfer time. Anything that overlapped
    // with compute already happened in the background and shouldn't be billed against the
    // critical-path profile.
    stats_.total_transfer_time_ms.fetch_add(elapsed_ms, std::memory_order_relaxed);

    // Reset the handle to default-constructed state (transfer is no longer in flight).
    info.pending_handle = tenzor::core::TransferHandle{};
    info.pending_is_offload = false;
}

auto OffloadContext::offload_tensor(Tensor* tensor_ptr,
                                    OffloadPriority priority,
                                    bool priority_explicit) -> bool {
    if (!tensor_ptr) {
        return false;
    }
    if (!is_enabled()) {
        return false;
    }

    std::lock_guard<std::mutex> lock(tensor_map_mutex_);

    auto it = tensor_map_.find(tensor_ptr);
    if (it == tensor_map_.end()) {
        return false;
    }

    auto& info = it->second;

    // Honour caller-supplied priority for manual offload paths
    // (offload_single_tensor → offload_param). HIGH means "keep on device"
    // (skip offload), LOW means "force offload" (bypass size threshold),
    // NORMAL follows the standard `should_offload` predicate.
    if (priority_explicit) {
        info.priority = priority;
        if (priority == OffloadPriority::HIGH) {
            return false;  // HIGH = pin to device, never offload via this path
        }
    }

    // If a previous async transfer for this tensor is still pending, drain it so we have a
    // consistent view before deciding whether a new offload is needed.
    finalize_pending(info, tensor_ptr);

    // Check if should offload. LOW priority bypasses the size threshold so a
    // user can force a small tensor off the device.
    const bool force = priority_explicit && (priority == OffloadPriority::LOW);
    if ((!force && !should_offload(info)) || info.is_offloaded) {
        return false;
    }

    try {
        // Save the original device + dtype before offloading
        info.original_device = tensor_ptr->device();
        info.original_dtype = tensor_ptr->dtype();

        // Phase C (C3): pick the dtype for the host-side copy based on Config and
        // per-tensor exemptions. Pinned tensors and CRITICAL-priority tensors keep
        // their original dtype (no precision loss for activations whose backward
        // may be sensitive). Everything else honors Config::offload_dtype.
        DType cast_dtype = info.original_dtype;
        bool use_int8_quant = false;
        // Gradients are offloaded at LOW priority, but they must NEVER be
        // lossy-quantized by default: a Half/BF16/Int8 round-trip of an F32
        // gradient degrades every optimizer step. Exclude gradients from
        // quantization so they round-trip losslessly (F077). (There is no
        // per-config opt-in for grad quantization; grads are always exact.)
        const bool may_quant = !info.is_pinned
                            && info.priority != OffloadPriority::HIGH
                            && !info.is_gradient;
        if (may_quant && info.original_dtype == DType::Float32) {
            switch (config_.offload_dtype) {
                case Config::OffloadDType::Same:
                    break;
                case Config::OffloadDType::Half:
                    cast_dtype = DType::Float16;
                    break;
                case Config::OffloadDType::BFloat16:
                    cast_dtype = DType::BFloat16;
                    break;
                case Config::OffloadDType::Int8WithScale:
                    // Audit G3: real INT8 quantize-on-offload + dequantize-on-fetch.
                    // Per-tensor symmetric scale: scale = max(|t|) / 127.
                    // Result is `Int8` on the wire (4x smaller than Float32 vs 2x
                    // for Half/BFloat16). Reuses the same quantize math as
                    // ZeROStage1Optimizer's `quantize_to_int8` helper.
                    use_int8_quant = true;
                    cast_dtype = DType::Int8;
                    break;
            }
        }
        info.offload_dtype_used = cast_dtype;
        info.quant_scale = 0.0f;

        // Cast on-GPU (when needed) before DMA. The cast-result tensor is what
        // we send across PCIe -- so a Half cast halves the bytes-on-the-wire as
        // well as the host-side residency. Audit G3: Int8 path quantizes here
        // and stores the scale in `info.quant_scale`.
        Tensor src;
        Tensor pending_max_t;  // kept alive for the post-DMA scale readback
        if (use_int8_quant) {
            // scale = max(|t|) / 127. Pin to 1.0 if the entire tensor is zero so
            // we don't emit a NaN-laden int8 payload.
            //
            // The quantize is performed entirely with device tensor ops so that
            // *no* blocking device->host scalar readback sits on the offload
            // critical path ahead of the async DMA. The zero-guard (max == 0 ->
            // emit zeros) is applied on-device via `where`, and the host-side
            // `quant_scale` readback is deferred until *after* the async DMA is
            // issued (see below), so it overlaps the in-flight transfer instead
            // of serializing the offload.
            Tensor max_t = tenzor::max(abs(*tensor_ptr));  // 0-d scalar on device
            pending_max_t = max_t;

            // safe_max = (max > 0) ? max : 1, so the all-zero tensor produces a
            // zero int8 payload (0 / 1 * 127 = 0) rather than NaN/Inf, and the
            // deferred host scale falls back to 1.0 consistently.
            Tensor pos_mask = tenzor::gt(max_t, tenzor::zeros_like(max_t));
            Tensor safe_max = tenzor::where(pos_mask, max_t, tenzor::ones_like(max_t));

            // q = clamp(round(t / safe_max * 127), -128, 127) cast to Int8.
            // Equivalent to round(t / scale) since scale = safe_max / 127.
            Tensor scaled = tenzor::mul(tenzor::div(*tensor_ptr, safe_max), 127.0);
            Tensor rounded = round(scaled);
            Tensor clamped = clamp(rounded, -128.0f, 127.0f);
            src = clamped.to(DType::Int8);
        } else {
            src = (cast_dtype != info.original_dtype)
                    ? tensor_ptr->to(cast_dtype)
                    : *tensor_ptr;
        }

        // Issue async GPU→CPU transfer. The actual data motion runs on the TransferEngine's
        // dedicated stream, which lets it overlap with the next layer's compute on the
        // default stream. We do *not* swap *tensor_ptr yet — the GPU memory must stay alive
        // until the DMA completes. The swap and memory-free happen in finalize_pending(),
        // typically called from the next forward_post_hook / backward_post_hook.
        info.pending_handle = transfer_engine_->gpu_to_cpu_async(src);
        info.pending_is_offload = true;

        // Audit G3: read back the per-tensor quant scale *after* the async int8
        // DMA has been issued so the (unavoidable) device->host scalar transfer
        // overlaps the in-flight payload DMA rather than blocking ahead of it.
        // `scale = max(|t|) / 127`, with the same 1.0 zero-guard the on-device
        // quantize used (safe_max == 1 when max == 0).
        if (use_int8_quant) {
            float max_val = pending_max_t.item<float>();
            info.quant_scale = (max_val > 0.0f) ? (max_val / 127.0f) : 1.0f;
        }

        // Stats: count the offload at issue time (so callers see immediate feedback in
        // stats); transfer-time accounting happens in finalize_pending against actual wait.
        stats_.total_offloads.fetch_add(1, std::memory_order_relaxed);
        stats_.transfer_count.fetch_add(1, std::memory_order_relaxed);

        return true;
    } catch (const std::exception& e) {
        // Audit I.4: route to unified logger.
        TENZOR_LOG_WARN("OffloadContext::offload_tensor failed: {}", e.what());
        return false;
    }
}

auto OffloadContext::prefetch_tensor(Tensor* tensor_ptr) -> bool {
    if (!tensor_ptr || !is_enabled()) return false;

    std::lock_guard<std::mutex> lock(tensor_map_mutex_);

    auto it = tensor_map_.find(tensor_ptr);
    if (it == tensor_map_.end()) return false;

    auto& info = it->second;

    // Drain any prior pending transfer (could be an offload that finished while we were
    // doing other work; we must commit it before issuing a load in the opposite direction).
    finalize_pending(info, tensor_ptr);

    // Only prefetch if currently offloaded
    if (!info.is_offloaded) {
        return false;
    }

    try {
        // Issue async CPU→GPU transfer; commit happens in finalize_pending() (typically the
        // forward_pre_hook or backward_pre_hook of the consuming layer).
        info.pending_handle = transfer_engine_->cpu_to_gpu_async(info.cpu_copy, info.original_device);
        info.pending_is_offload = false;

        stats_.total_prefetches.fetch_add(1, std::memory_order_relaxed);
        stats_.transfer_count.fetch_add(1, std::memory_order_relaxed);

        // Eagerly decrement CPU memory at issue time (matches symmetric increment in
        // offload_tensor's finalize); we'll never re-touch info.cpu_copy after a load issues.
        stats_.current_cpu_memory.fetch_sub(info.size_bytes, std::memory_order_relaxed);

        return true;
    } catch (const std::exception& e) {
        // Audit I.4: route to unified logger.
        TENZOR_LOG_WARN("OffloadContext::prefetch_tensor failed: {}", e.what());
        return false;
    }
}

auto OffloadContext::should_offload(const TensorInfo& info) const -> bool {
    // Don't offload pinned tensors
    if (info.is_pinned) return false;

    // HIGH-priority tensors are pinned to the device by policy: they bypass
    // the auto-offload predicate entirely. (Manual offload via offload_param
    // also skips them — see offload_tensor.)
    if (info.priority == OffloadPriority::HIGH) return false;

    // Check size threshold
    if (info.size_bytes < config_.offload_threshold) return false;

    // Check if offloading is enabled for this tensor type
    if (info.is_gradient) {
        if (!config_.offload_gradients) return false;
    } else {
        if (!config_.offload_parameters) return false;
    }

    return true;
}

// Comparator for ranking offload candidates during eviction. Tensors that
// should be evicted first compare "less than" tensors that should be kept.
// Ordering: LOW evicts before NORMAL before HIGH; within the same priority,
// older (lower) use_count evicts first. Static member so the comparator has
// access to the private nested TensorInfo type.
bool OffloadContext::priority_lt(const OffloadContext::TensorInfo& a,
                                  const OffloadContext::TensorInfo& b) {
    auto rank = [](OffloadPriority p) -> int {
        switch (p) {
            case OffloadPriority::LOW:    return 0;
            case OffloadPriority::NORMAL: return 1;
            case OffloadPriority::HIGH:   return 2;
        }
        return 1;
    };
    int ra = rank(a.priority);
    int rb = rank(b.priority);
    if (ra != rb) return ra < rb;
    return a.use_count < b.use_count;
}

// ============================================================================
// Private: Hook Callbacks
// ============================================================================

auto OffloadContext::forward_pre_hook(Module* layer) -> void {
    if (!is_enabled()) return;

    // Drain any offloads from previous layers that completed in the background.
    // Without this, forward_post_hook-issued gpu_to_cpu_async transfers stay
    // half-committed (GPU memory still resident) until the same tensor is
    // touched again next step. Non-blocking: only commits transfers whose
    // TransferHandle::is_ready() returns true.
    finalize_completed_offloads();

    // Helper lambda to load a tensor to GPU
    auto load_tensor_to_gpu = [&](Tensor* tensor_ptr, const std::shared_ptr<Variable>& owner) {
        std::lock_guard<std::mutex> lock(tensor_map_mutex_);
        auto it = tensor_map_.find(tensor_ptr);

        // First: drain any in-flight async transfer for this tensor. After finalize the
        // tensor is in a known state (either committed-to-CPU after an offload, or
        // committed-to-GPU after a prefetch), and we can decide what to do next.
        if (it != tensor_map_.end()) {
            finalize_pending(it->second, tensor_ptr);
        }

        // Case 1: Tensor was offloaded from GPU - restore it
        if (it != tensor_map_.end() && it->second.is_offloaded) {
            try {
                Tensor gpu_tensor = transfer_engine_->cpu_to_gpu(
                    it->second.cpu_copy, it->second.original_device);
                // Phase C (C3): cast back to original dtype on GPU after DMA when
                // the cpu_copy was quantized.
                // Audit G3: Int8WithScale dequant — widen + multiply by scale.
                if (it->second.offload_dtype_used == DType::Int8 &&
                    it->second.quant_scale != 0.0f) {
                    Tensor widened = gpu_tensor.to(it->second.original_dtype);
                    gpu_tensor = widened * it->second.quant_scale;
                } else if (it->second.original_dtype != it->second.offload_dtype_used) {
                    gpu_tensor = gpu_tensor.to(it->second.original_dtype);
                }
                *tensor_ptr = gpu_tensor;
                it->second.is_offloaded = false;
                stats_.current_cpu_memory.fetch_sub(it->second.size_bytes, std::memory_order_relaxed);
            } catch (const std::exception& e) {
                // Audit I.4: route to unified logger.
                TENZOR_LOG_WARN("OffloadContext::load_to_gpu: failed to load offloaded "
                                "tensor to GPU: {}", e.what());
            }
        }
        // Case 2: Tensor on CPU and not in map - CPU-start model, load to target device
        else if (tensor_ptr->device().type == Device::Type::CPU) {
            try {
                // Initialize tensor info if not already tracked
                if (it == tensor_map_.end()) {
                    initialize_tensor_info(tensor_ptr, layer, owner);
                    it = tensor_map_.find(tensor_ptr);
                }
                // Store CPU copy and load to GPU
                if (it != tensor_map_.end()) {
                    it->second.cpu_copy = *tensor_ptr;  // Store original CPU tensor
                    Tensor gpu_tensor = transfer_engine_->cpu_to_gpu(*tensor_ptr, config_.target_device);
                    *tensor_ptr = gpu_tensor;
                    it->second.is_offloaded = false;  // Currently on GPU
                }
            } catch (const std::exception& e) {
                // Audit I.4: route to unified logger.
                TENZOR_LOG_WARN("OffloadContext::load_to_gpu: failed to load CPU tensor "
                                "to GPU: {}", e.what());
            }
        }
    };

    // Load this layer's own parameters to GPU
    auto params = layer->own_parameters();
    for (auto& param_ptr : params) {
        if (param_ptr) {
            load_tensor_to_gpu(&(param_ptr->tensor()), param_ptr);
        }
    }

    // Also load this layer's own buffers to GPU (e.g., BatchNorm running stats)
    auto bufs = layer->own_buffers();
    for (auto& buf_ptr : bufs) {
        if (buf_ptr) {
            load_tensor_to_gpu(&(buf_ptr->tensor()), buf_ptr);
        }
    }

    // Prefetch ahead if configured
    auto it = layer_indices_.find(layer);
    if (it != layer_indices_.end()) {
        int current_idx = it->second;

        // Prefetch next N layers
        for (int i = 1; i <= config_.prefetch_depth; ++i) {
            int next_idx = current_idx + i;
            if (next_idx < static_cast<int>(layer_order_.size())) {
                prefetch_layer(layer_order_[next_idx]);
            }
        }
    }
}

auto OffloadContext::forward_post_hook(Module* layer) -> void {
    if (!is_enabled()) return;

    // Offload parameters after forward pass
    offload_layer(layer);
}

auto OffloadContext::backward_pre_hook(Module* layer) -> void {
    if (!is_enabled()) return;

    // Same drain rationale as forward_pre_hook -- release GPU storage of any
    // tensor whose async offload completed since the last hook fire.
    finalize_completed_offloads();

    // Prefetch parameters for this layer
    prefetch_layer(layer);

    // Also prefetch any offloaded gradients for this layer
    if (config_.offload_gradients) {
        auto params = layer->own_parameters();

        // Collect gradients that need prefetching (check under lock)
        std::vector<Tensor*> gradients_to_prefetch;
        {
            std::lock_guard<std::mutex> lock(tensor_map_mutex_);
            for (auto& param_ptr : params) {
                if (!param_ptr || !param_ptr->grad().has_value()) continue;

                Tensor* grad_tensor_ptr = &(param_ptr->mutable_grad().value());
                auto it = tensor_map_.find(grad_tensor_ptr);
                if (it != tensor_map_.end() && it->second.is_offloaded) {
                    gradients_to_prefetch.push_back(grad_tensor_ptr);
                }
            }
        }

        // Prefetch gradients back to GPU (outside lock to avoid deadlock)
        for (Tensor* grad_tensor_ptr : gradients_to_prefetch) {
            prefetch_tensor(grad_tensor_ptr);
        }
    }
}

auto OffloadContext::backward_post_hook(Module* layer) -> void {
    if (!is_enabled()) return;

    // Get all parameters for this layer
    auto params = layer->own_parameters();

    for (auto& param_ptr : params) {
        if (!param_ptr) continue;

        // Offload parameter to CPU after backward pass (same as forward_post_hook)
        if (config_.offload_parameters) {
            Tensor* tensor_ptr = &(param_ptr->tensor());
            offload_tensor(tensor_ptr);
        }

        // Offload gradient to CPU if configured
        if (config_.offload_gradients && param_ptr->grad().has_value()) {
            Tensor* grad_tensor_ptr = &(param_ptr->mutable_grad().value());

            // Skip if not resident on an accelerator (any non-CPU backend).
            if (!is_accelerator_device(grad_tensor_ptr->device())) continue;

            // Track gradient if not already tracked
            {
                std::lock_guard<std::mutex> lock(tensor_map_mutex_);
                if (tensor_map_.find(grad_tensor_ptr) == tensor_map_.end()) {
                    TensorInfo info;
                    info.tensor = grad_tensor_ptr;
                    info.owner = param_ptr;  // grad lives inside this Variable
                    info.is_offloaded = false;
                    info.is_pinned = false;
                    info.is_gradient = true;
                    info.use_count = 0;
                    info.priority = OffloadPriority::LOW;  // Gradients are low priority
                    info.size_bytes = grad_tensor_ptr->numel() * grad_tensor_ptr->dtype_size();
                    info.owning_layer = layer;
                    tensor_map_[grad_tensor_ptr] = std::move(info);
                }
            }

            // Offload gradient to CPU
            offload_tensor(grad_tensor_ptr);
        }
    }
}

// ============================================================================
// Private: Memory Management
// ============================================================================

auto OffloadContext::update_stats() -> void {
    // Update peak accelerator memory (the backend the model is offloading
    // against, not a hardcoded CUDA).
    size_t current_gpu = memory_manager_->get_memory_usage(config_.target_device.type);
    size_t peak = stats_.peak_gpu_memory.load(std::memory_order_relaxed);

    while (current_gpu > peak) {
        if (stats_.peak_gpu_memory.compare_exchange_weak(peak, current_gpu,
                                                          std::memory_order_relaxed)) {
            break;
        }
    }
}

auto OffloadContext::check_memory_pressure() -> void {
    // Check accelerator memory pressure (the backend the model lives on).
    const Device::Type accel_type = config_.target_device.type;
    float gpu_pressure = memory_manager_->get_memory_pressure(accel_type);

    if (gpu_pressure > 0.85f) {  // High memory pressure
        // Capture (sort_key, tensor_ptr) pairs UNDER the lock. Previously this
        // collected only pointers, dropped the lock, and then sorted using a
        // lambda that read tensor_map_[a] -- racing with concurrent inserts in
        // forward_pre_hook (initialize_tensor_info), which can rehash the map
        // and invalidate the references. Computing sort keys eagerly while
        // holding the lock removes the race entirely; sort runs against the
        // captured POD values.
        struct Candidate {
            int priority;
            size_t size_bytes;
            Tensor* tensor_ptr;
        };
        std::vector<Candidate> candidates;

        {
            std::lock_guard<std::mutex> lock(tensor_map_mutex_);
            candidates.reserve(tensor_map_.size());
            for (auto& [tensor_ptr, info] : tensor_map_) {
                if (!info.is_offloaded && should_offload(info)) {
                    candidates.push_back(Candidate{
                        static_cast<int>(info.priority),
                        info.size_bytes,
                        tensor_ptr});
                }
            }
        }

        // Sort by priority then size (offload large, high-priority tensors first).
        std::sort(candidates.begin(), candidates.end(),
                  [](const Candidate& a, const Candidate& b) {
                      if (a.priority != b.priority) {
                          return a.priority > b.priority;
                      }
                      return a.size_bytes > b.size_bytes;
                  });

        // Offload tensors until pressure is reduced.
        for (const auto& c : candidates) {
            offload_tensor(c.tensor_ptr);

            // Recheck pressure
            gpu_pressure = memory_manager_->get_memory_pressure(accel_type);
            if (gpu_pressure < 0.75f) break;  // Target 75% usage
        }
    }
}

// ============================================================================
// ComputeContext Implementation
// ============================================================================

ComputeContext::ComputeContext(const std::vector<Tensor*>& tensors)
    : tensors_(tensors) {

    // Initialize transfer engine
    core::TransferEngine::Config config;
    config.num_streams = 2;
    config.use_pinned_memory = true;
    transfer_engine_ = std::make_shared<core::TransferEngine>(config);

    // Stage CPU tensors onto the accelerator the workload actually uses rather
    // than a hardcoded Device::cuda(0). Prefer the device of an already-resident
    // tensor; otherwise the first available accelerator backend (F076).
    const Device compute_device = pick_compute_device(tensors_);

    // Save original devices and transfer to GPU if needed
    for (auto* tensor_ptr : tensors_) {
        if (!tensor_ptr) continue;

        original_devices_.push_back(tensor_ptr->device());

        // If on CPU, transfer to GPU.  Push exactly ONE entry into
        // cpu_copies_ per CPU tensor, regardless of whether the transfer
        // succeeds (audit item D.1 — previous code double-pushed on
        // failure, desyncing cpu_copies_ from tensors_ and corrupting
        // the destructor's restore loop).
        if (tensor_ptr->device().type == Device::Type::CPU) {
            cpu_copies_.push_back(*tensor_ptr);  // Save CPU copy

            try {
                Tensor gpu_tensor = transfer_engine_->cpu_to_gpu(*tensor_ptr, compute_device);
                *tensor_ptr = gpu_tensor;
            } catch (const std::exception& e) {
                // Transfer failed — tensor_ptr still points at the CPU
                // tensor, so original_devices_[i] (CPU) is correct and
                // the destructor's "restore to original device" path is
                // a no-op for this slot.  No extra placeholder needed.
                TENZOR_LOG_WARN("ComputeContext: failed to load tensor to GPU: {}",
                                e.what());
            }
        } else {
            cpu_copies_.push_back(Tensor());  // No copy needed
        }
    }
}

ComputeContext::~ComputeContext() {
    // Restore tensors to original devices
    for (size_t i = 0; i < tensors_.size(); ++i) {
        auto* tensor_ptr = tensors_[i];
        if (!tensor_ptr) continue;

        const auto& original_device = original_devices_[i];

        // If was on CPU, restore
        if (original_device.type == Device::Type::CPU && cpu_copies_[i].numel() > 0) {
            try {
                Tensor cpu_tensor = transfer_engine_->gpu_to_cpu(*tensor_ptr);
                *tensor_ptr = cpu_tensor;
            } catch (const std::exception& e) {
                // Audit I.4: route to unified logger.
                TENZOR_LOG_WARN("ComputeContext: failed to restore tensor to CPU: {}",
                                e.what());
            }
        }
    }

    // Synchronize to ensure all transfers complete
    if (transfer_engine_) {
        transfer_engine_->synchronize();
    }
}

auto ComputeContext::synchronize() -> void {
    if (transfer_engine_) {
        transfer_engine_->synchronize();
    }
}

// ============================================================================
// Global Functions
// ============================================================================

auto offload_param(Tensor& param, OffloadPriority priority) -> void {
    // Forward the caller's priority into the per-tensor TensorInfo. The
    // semantics:
    //   HIGH   → pin to device (skip offload via this manual path).
    //   NORMAL → standard `should_offload` predicate applies.
    //   LOW    → force offload, bypassing the size threshold.
    // The priority is also recorded on the TensorInfo so eviction-time
    // comparators (priority_lt) and quantization-on-offload decisions read
    // a consistent value.
    auto* ctx = get_global_offload_context();
    if (!ctx) {
        // Audit I.4: route to unified logger.
        TENZOR_LOG_WARN("offload_param called but no global offload context set");
        return;
    }
    ctx->offload_single_tensor(&param, priority);
}

auto get_global_offload_context() -> OffloadContext* {
    return g_offload_context;
}

auto set_global_offload_context(OffloadContext* ctx) -> void {
    g_offload_context = ctx;
}

} // namespace nn
} // namespace tenzor
