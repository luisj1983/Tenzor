/**
 * @file offload.cpp
 * @brief Implementation of Parameter Offloading API for ZeRO Phase 2
 */

#include "tenzor/nn/offload.hpp"
#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <iostream>

namespace tenzor {
namespace nn {

// Global offload context (for offload_param function)
namespace {
    OffloadContext* g_offload_context = nullptr;
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
    // Disable hooks
    disable();

    // Synchronize all pending transfers
    if (transfer_engine_) {
        transfer_engine_->synchronize();
    }

    // Restore all offloaded tensors to their original device
    // NOTE: During cleanup, we don't strictly need to restore - the tensors will be
    // destroyed anyway. Skip restoration to avoid potential allocation failures.
    // The CPU copies will be cleaned up when tensor_map_ is destroyed.
}

auto OffloadContext::enable() -> void {
    enabled_.store(true, std::memory_order_release);

    // Immediately offload all parameters to CPU when offloading is enabled
    if (config_.offload_parameters) {
        auto all_params = model_.parameters();

        for (auto& param_ptr : all_params) {
            if (param_ptr) {
                Tensor* tensor_ptr = &(param_ptr->tensor());

                // Only offload if on GPU
                if (tensor_ptr->device().type == Device::Type::CUDA) {
                    // Add to tensor map if not already there
                    {
                        std::lock_guard<std::mutex> lock(tensor_map_mutex_);
                        if (tensor_map_.find(tensor_ptr) == tensor_map_.end()) {
                            initialize_tensor_info(tensor_ptr, &model_);
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
}

auto OffloadContext::is_enabled() const -> bool {
    return enabled_.load(std::memory_order_acquire);
}

auto OffloadContext::get_stats() -> OffloadStats {
    OffloadStats stats;

    // Perform gradient offloading if enabled
    // Note: Module backward hooks are not yet connected to autograd, so we do lazy offloading here
    if (is_enabled() && config_.offload_gradients) {
        auto all_params = model_.parameters();
        for (auto& param_ptr : all_params) {
            if (param_ptr && param_ptr->grad().has_value()) {
                Tensor* grad_tensor_ptr = &(param_ptr->mutable_grad().value());

                // Track gradient if not already tracked
                {
                    std::lock_guard<std::mutex> lock(tensor_map_mutex_);
                    if (tensor_map_.find(grad_tensor_ptr) == tensor_map_.end()) {
                        // Initialize gradient tracking
                        TensorInfo info;
                        info.tensor = grad_tensor_ptr;
                        info.is_offloaded = false;
                        info.is_pinned = false;
                        info.is_gradient = true;
                        info.use_count = 0;
                        info.priority = OffloadPriority::LOW;
                        info.size_bytes = grad_tensor_ptr->numel() * grad_tensor_ptr->dtype_size();
                        info.owning_layer = &model_;
                        tensor_map_[grad_tensor_ptr] = std::move(info);
                    }
                }

                // Offload gradient if on GPU
                if (grad_tensor_ptr->device().type == Device::Type::CUDA) {
                    offload_tensor(grad_tensor_ptr);
                }
            }
        }
    }

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
    return memory_manager_->get_memory_usage(Device::Type::CUDA);
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
                initialize_tensor_info(tensor_ptr, layer);
            }
        }
    }
}

auto OffloadContext::initialize_tensor_info(Tensor* tensor, Module* layer) -> void {
    if (!tensor) return;

    TensorInfo info;
    info.tensor = tensor;
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
        *tensor_ptr = result;
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

auto OffloadContext::offload_tensor(Tensor* tensor_ptr) -> bool {
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

    // If a previous async transfer for this tensor is still pending, drain it so we have a
    // consistent view before deciding whether a new offload is needed.
    finalize_pending(info, tensor_ptr);

    // Check if should offload
    if (!should_offload(info) || info.is_offloaded) {
        return false;
    }

    try {
        // Save the original device before offloading
        info.original_device = tensor_ptr->device();

        // Issue async GPU→CPU transfer. The actual data motion runs on the TransferEngine's
        // dedicated stream, which lets it overlap with the next layer's compute on the
        // default stream. We do *not* swap *tensor_ptr yet — the GPU memory must stay alive
        // until the DMA completes. The swap and memory-free happen in finalize_pending(),
        // typically called from the next forward_post_hook / backward_post_hook.
        info.pending_handle = transfer_engine_->gpu_to_cpu_async(*tensor_ptr);
        info.pending_is_offload = true;

        // Stats: count the offload at issue time (so callers see immediate feedback in
        // stats); transfer-time accounting happens in finalize_pending against actual wait.
        stats_.total_offloads.fetch_add(1, std::memory_order_relaxed);
        stats_.transfer_count.fetch_add(1, std::memory_order_relaxed);

        return true;
    } catch (const std::exception& e) {
        std::cerr << "Offload failed: " << e.what() << "\n";
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
        std::cerr << "Prefetch failed: " << e.what() << "\n";
        return false;
    }
}

auto OffloadContext::should_offload(const TensorInfo& info) const -> bool {
    // Don't offload pinned tensors
    if (info.is_pinned) return false;

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

// ============================================================================
// Private: Hook Callbacks
// ============================================================================

auto OffloadContext::forward_pre_hook(Module* layer) -> void {
    if (!is_enabled()) return;

    // Helper lambda to load a tensor to GPU
    auto load_tensor_to_gpu = [&](Tensor* tensor_ptr) {
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
                Tensor gpu_tensor = transfer_engine_->cpu_to_gpu(it->second.cpu_copy, it->second.original_device);
                *tensor_ptr = gpu_tensor;
                it->second.is_offloaded = false;
                stats_.current_cpu_memory.fetch_sub(it->second.size_bytes, std::memory_order_relaxed);
            } catch (const std::exception& e) {
                std::cerr << "Failed to load tensor to GPU: " << e.what() << "\n";
            }
        }
        // Case 2: Tensor on CPU and not in map - CPU-start model, load to target device
        else if (tensor_ptr->device().type == Device::Type::CPU) {
            try {
                // Initialize tensor info if not already tracked
                if (it == tensor_map_.end()) {
                    initialize_tensor_info(tensor_ptr, layer);
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
                std::cerr << "Failed to load CPU tensor to GPU: " << e.what() << "\n";
            }
        }
    };

    // Load this layer's own parameters to GPU
    auto params = layer->own_parameters();
    for (auto& param_ptr : params) {
        if (param_ptr) {
            load_tensor_to_gpu(&(param_ptr->tensor()));
        }
    }

    // Also load this layer's own buffers to GPU (e.g., BatchNorm running stats)
    auto bufs = layer->own_buffers();
    for (auto& buf_ptr : bufs) {
        if (buf_ptr) {
            load_tensor_to_gpu(&(buf_ptr->tensor()));
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

            // Skip if not on GPU
            if (grad_tensor_ptr->device().type != Device::Type::CUDA) continue;

            // Track gradient if not already tracked
            {
                std::lock_guard<std::mutex> lock(tensor_map_mutex_);
                if (tensor_map_.find(grad_tensor_ptr) == tensor_map_.end()) {
                    TensorInfo info;
                    info.tensor = grad_tensor_ptr;
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
    // Update peak GPU memory
    size_t current_gpu = memory_manager_->get_memory_usage(Device::Type::CUDA);
    size_t peak = stats_.peak_gpu_memory.load(std::memory_order_relaxed);

    while (current_gpu > peak) {
        if (stats_.peak_gpu_memory.compare_exchange_weak(peak, current_gpu,
                                                          std::memory_order_relaxed)) {
            break;
        }
    }
}

auto OffloadContext::check_memory_pressure() -> void {
    // Check GPU memory pressure
    float gpu_pressure = memory_manager_->get_memory_pressure(Device::Type::CUDA);

    if (gpu_pressure > 0.85f) {  // High memory pressure
        // Find candidates for offloading
        std::vector<Tensor*> candidates;

        {
            std::lock_guard<std::mutex> lock(tensor_map_mutex_);
            for (auto& [tensor_ptr, info] : tensor_map_) {
                if (!info.is_offloaded && should_offload(info)) {
                    candidates.push_back(tensor_ptr);
                }
            }
        }

        // Sort by priority and size (offload large, high-priority tensors first)
        std::sort(candidates.begin(), candidates.end(),
                  [this](Tensor* a, Tensor* b) {
                      auto& info_a = tensor_map_[a];
                      auto& info_b = tensor_map_[b];

                      if (info_a.priority != info_b.priority) {
                          return static_cast<int>(info_a.priority) > static_cast<int>(info_b.priority);
                      }
                      return info_a.size_bytes > info_b.size_bytes;
                  });

        // Offload tensors until pressure is reduced
        for (auto* tensor : candidates) {
            offload_tensor(tensor);

            // Recheck pressure
            gpu_pressure = memory_manager_->get_memory_pressure(Device::Type::CUDA);
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

    // Save original devices and transfer to GPU if needed
    for (auto* tensor_ptr : tensors_) {
        if (!tensor_ptr) continue;

        original_devices_.push_back(tensor_ptr->device());

        // If on CPU, transfer to GPU
        if (tensor_ptr->device().type == Device::Type::CPU) {
            cpu_copies_.push_back(*tensor_ptr);  // Save CPU copy

            try {
                Tensor gpu_tensor = transfer_engine_->cpu_to_gpu(*tensor_ptr, Device::cuda(0));
                *tensor_ptr = gpu_tensor;
            } catch (const std::exception& e) {
                std::cerr << "ComputeContext: Failed to load tensor to GPU: " << e.what() << "\n";
                cpu_copies_.push_back(Tensor());  // Empty placeholder
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
                std::cerr << "ComputeContext: Failed to restore tensor to CPU: " << e.what() << "\n";
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

auto offload_param([[maybe_unused]] Tensor& param, [[maybe_unused]] OffloadPriority priority) -> void {
    auto* ctx = get_global_offload_context();
    if (!ctx) {
        std::cerr << "Warning: offload_param called but no global offload context set\n";
        return;
    }

    // Manual offload through context
    // This would require making offload_tensor public or adding a friend function
    // For now, we document the intended behavior

    // In full implementation:
    // ctx->offload_tensor(&param);
}

auto get_global_offload_context() -> OffloadContext* {
    return g_offload_context;
}

auto set_global_offload_context(OffloadContext* ctx) -> void {
    g_offload_context = ctx;
}

} // namespace nn
} // namespace tenzor
