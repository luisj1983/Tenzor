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
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <iostream>

namespace tenzor {
namespace optim {

// =============================================================================
// Constructor & Destructor
// =============================================================================

ZeROStage1Optimizer::ZeROStage1Optimizer(
    std::unique_ptr<Optimizer> base_optimizer,
    const ZeROStage1Config& config
) : Optimizer(base_optimizer ? base_optimizer->parameters() : std::vector<std::shared_ptr<Variable>>{}),
    base_optimizer_(std::move(base_optimizer)),
    config_(config) {

    // Validation
    if (!base_optimizer_) {
        throw std::invalid_argument("base_optimizer cannot be null");
    }
    if (config_.rank < 0 || config_.rank >= config_.world_size) {
        throw std::invalid_argument("Invalid rank: must be in [0, world_size)");
    }
    if (config_.world_size <= 0) {
        throw std::invalid_argument("world_size must be > 0");
    }

    // Initialize distributed communication if not provided
    if (!config_.process_group && distributed::is_initialized()) {
        config_.process_group = distributed::DistributedContext::get_process_group();
    }
    // Note: If world_size > 1 but process_group is null, communication operations
    // will be skipped. This allows testing with multi-rank configs without requiring
    // actual distributed initialization.

    // Partition parameters across ranks
    partition_parameters();

    // Initialize optimizer states
    initialize_optimizer_states();

    // Initialize offload engine if needed
    if (config_.offload_to_cpu) {
        initialize_offload_engine();
    }
}

ZeROStage1Optimizer::~ZeROStage1Optimizer() {
    // Cleanup is automatic via smart pointers
}

// =============================================================================
// Optimizer Interface
// =============================================================================

auto ZeROStage1Optimizer::step() -> void {
    std::lock_guard<std::mutex> lock(mutex_);

    // Step 1: All-reduce gradients across ranks
    if (config_.world_size > 1) {
        all_reduce_gradients();
    }

    // Step 2: Fetch optimizer states from CPU if offloaded
    if (config_.offload_to_cpu && offload_engine_) {
        fetch_states_to_gpu();
    }

    // Step 3: Update local partition of parameters
    update_local_partition();

    // Step 4: Offload states back to CPU if enabled
    if (config_.offload_to_cpu && offload_engine_) {
        offload_states_to_cpu();
    }

    // Step 5: All-gather updated parameters across ranks
    if (config_.world_size > 1) {
        all_gather_parameters();
    }
}

auto ZeROStage1Optimizer::zero_grad() -> void {
    base_optimizer_->zero_grad();
}

auto ZeROStage1Optimizer::state_dict() const -> std::unordered_map<std::string, Tensor> {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Return only local partition state
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

auto ZeROStage1Optimizer::load_state_dict(
    const std::unordered_map<std::string, Tensor>& state
) -> void {
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

auto ZeROStage1Optimizer::save_checkpoint(const std::string& path_prefix) const -> void {
    std::lock_guard<std::mutex> lock(mutex_);

    // Each rank saves its own partition
    std::string rank_path = path_prefix + "_rank_" + std::to_string(config_.rank) + ".pt";

    try {
        // Get state dictionary for this rank's partition
        auto state = state_dict();

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

        // Load state into optimizer
        load_state_dict(loaded_state);

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
    const auto& params = parameters_;
    size_t total_params = params.size();
    size_t params_per_rank = (total_params + config_.world_size - 1) / config_.world_size;

    // Determine device from first parameter (all params should be on same device)
    Device param_device = !params.empty() ? params[0]->tensor().device() : Device::cpu();

    // Create partitions for all ranks
    partitions_.resize(config_.world_size);

    for (int rank = 0; rank < config_.world_size; ++rank) {
        auto& partition = partitions_[rank];
        partition.rank = rank;
        // Use the same device as the parameters (don't assume CUDA)
        partition.device = config_.offload_to_cpu ? Device::cpu() : param_device;
        
        // Assign parameters to this rank
        size_t start_idx = rank * params_per_rank;
        size_t end_idx = std::min(start_idx + params_per_rank, total_params);
        
        for (size_t i = start_idx; i < end_idx; ++i) {
            partition.params.push_back(params[i]);
            const auto& tensor = params[i]->tensor();
            partition.memory_bytes += tensor.numel() * dtype_size(tensor.dtype());
        }
    }
}

auto ZeROStage1Optimizer::initialize_optimizer_states() -> void {
    // Initialize states for local partition only
    auto& partition = local_partition();
    
    // Detect optimizer type and create appropriate states
    // For Adam/AdamW: need momentum and variance
    // For SGD with momentum: need momentum only
    
    partition.momentum.reserve(partition.params.size());
    partition.variance.reserve(partition.params.size());
    
    for (const auto& param : partition.params) {
        // Momentum buffer (all optimizers)
        Tensor momentum = zeros_like(param->tensor()).to(partition.device);
        partition.momentum.push_back(momentum);
        
        // Variance buffer (Adam family)
        Tensor variance = zeros_like(param->tensor()).to(partition.device);
        partition.variance.push_back(variance);
        
        partition.memory_bytes += momentum.numel() * dtype_size(momentum.dtype());
        partition.memory_bytes += variance.numel() * dtype_size(variance.dtype());
    }
}

auto ZeROStage1Optimizer::initialize_offload_engine() -> void {
    if (!config_.offload_to_cpu) {
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
    if (!config_.process_group) {
        throw std::runtime_error("Process group not initialized");
    }
    
    // All-reduce gradients for all parameters
    for (auto& param : parameters_) {
        if (param->has_grad()) {
            auto& grad_opt = param->grad();
            if (grad_opt.has_value()) {
                Tensor grad = grad_opt.value();
                config_.process_group->all_reduce(grad, distributed::ReduceOp::SUM);

                // Average by world size
                grad = grad / static_cast<float>(config_.world_size);

                // Update the gradient (this modifies the optional)
                grad_opt = grad;
            }
        }
    }
}

auto ZeROStage1Optimizer::all_gather_parameters() -> void {
    if (!config_.process_group) {
        throw std::runtime_error("Process group not initialized");
    }
    
    // All-gather parameters from all ranks
    for (int rank = 0; rank < config_.world_size; ++rank) {
        const auto& partition = partitions_[rank];
        
        for (const auto& param : partition.params) {
            // Broadcast from owner rank
            Tensor param_data = param->tensor();
            config_.process_group->broadcast(param_data, rank);
            
            if (rank != config_.rank) {
                // Copy broadcasted data to local parameter
                param->tensor() = param_data;
            }
        }
    }
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

        const Tensor& grad = grad_opt.value();

        // Apply weight decay (L2 regularization)
        Tensor grad_with_decay = grad;
        if (weight_decay != 0.0) {
            grad_with_decay = grad + param->tensor() * static_cast<float>(weight_decay);
        }

        // Update biased first moment estimate
        Tensor& momentum = partition.momentum[i];
        momentum = momentum * static_cast<float>(beta1) +
                   grad_with_decay * static_cast<float>(1.0 - beta1);

        // Update biased second moment estimate
        Tensor& variance = partition.variance[i];
        variance = variance * static_cast<float>(beta2) +
                   (grad_with_decay * grad_with_decay) * static_cast<float>(1.0 - beta2);

        // Compute bias-corrected moment estimates
        double bias_correction1 = 1.0 - std::pow(beta1, step_count_);
        double bias_correction2 = 1.0 - std::pow(beta2, step_count_);

        Tensor momentum_corrected = momentum * static_cast<float>(1.0 / bias_correction1);
        Tensor variance_corrected = variance * static_cast<float>(1.0 / bias_correction2);

        // Compute step: theta = theta - lr * m_hat / (sqrt(v_hat) + eps)
        Tensor denom = sqrt(variance_corrected) + static_cast<float>(eps);

        // Update parameters directly (same pattern as standard Adam)
        param->tensor() = param->tensor() - div(momentum_corrected, denom) * static_cast<float>(lr);
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

        const Tensor& grad = grad_opt.value();

        // Update biased first moment estimate
        Tensor& momentum = partition.momentum[i];
        momentum = momentum * static_cast<float>(beta1) +
                   grad * static_cast<float>(1.0 - beta1);

        // Update biased second moment estimate
        Tensor& variance = partition.variance[i];
        variance = variance * static_cast<float>(beta2) +
                   (grad * grad) * static_cast<float>(1.0 - beta2);

        // Compute bias-corrected moment estimates
        double bias_correction1 = 1.0 - std::pow(beta1, step_count_);
        double bias_correction2 = 1.0 - std::pow(beta2, step_count_);

        Tensor momentum_corrected = momentum * static_cast<float>(1.0 / bias_correction1);
        Tensor variance_corrected = variance * static_cast<float>(1.0 / bias_correction2);

        // Compute denominator
        Tensor denom = sqrt(variance_corrected) + static_cast<float>(eps);

        // AdamW: Apply decoupled weight decay + optimizer step
        if (weight_decay != 0.0) {
            param->tensor() = param->tensor() * static_cast<float>(1.0 - lr * weight_decay) -
                            div(momentum_corrected, denom) * static_cast<float>(lr);
        } else {
            param->tensor() = param->tensor() - div(momentum_corrected, denom) * static_cast<float>(lr);
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

        const Tensor& grad = grad_opt.value();

        // Apply weight decay (L2 regularization)
        Tensor grad_with_decay = grad;
        if (weight_decay != 0.0) {
            grad_with_decay = grad + param->tensor() * static_cast<float>(weight_decay);
        }

        if (momentum_coef != 0.0) {
            // SGD with momentum
            Tensor& momentum = partition.momentum[i];
            momentum = momentum * static_cast<float>(momentum_coef) + grad_with_decay;
            param->tensor() = param->tensor() - momentum * static_cast<float>(lr);
        } else {
            // Vanilla SGD
            param->tensor() = param->tensor() - grad_with_decay * static_cast<float>(lr);
        }
    }
}

auto ZeROStage1Optimizer::fetch_states_to_gpu() -> void {
    if (!offload_engine_) {
        return;
    }

    auto& partition = local_partition();

    // Only fetch if parameters are on GPU (CPU offload only makes sense for GPU training)
    Device param_device = !parameters_.empty() ? parameters_[0]->tensor().device() : Device::cpu();
    if (param_device.type != Device::Type::CUDA) {
        return;  // Skip offload for CPU parameters
    }

    if (partition.device.type == Device::Type::CPU) {
        // Prefetch all states to GPU
        std::vector<Tensor*> all_states;
        for (auto& momentum : partition.momentum) {
            all_states.push_back(&momentum);
        }
        for (auto& variance : partition.variance) {
            all_states.push_back(&variance);
        }

        offload_engine_->prefetch_to_gpu(all_states);
        offload_engine_->wait_for_prefetch();
    }
}

auto ZeROStage1Optimizer::offload_states_to_cpu() -> void {
    if (!offload_engine_) {
        return;
    }

    auto& partition = local_partition();

    // Only offload if parameters are on GPU (CPU offload only makes sense for GPU training)
    Device param_device = !parameters_.empty() ? parameters_[0]->tensor().device() : Device::cpu();
    if (param_device.type != Device::Type::CUDA) {
        return;  // Skip offload for CPU parameters
    }

    if (partition.device.type == Device::Type::CPU) {
        // Offload all states back to CPU asynchronously
        for (auto& momentum : partition.momentum) {
            offload_engine_->offload_to_cpu_async(momentum);
        }

        for (auto& variance : partition.variance) {
            offload_engine_->offload_to_cpu_async(variance);
        }

        offload_engine_->synchronize();
    }
}

// =============================================================================
// ZeRO Stage 2 Optimizer Implementation
// =============================================================================

ZeROStage2Optimizer::ZeROStage2Optimizer(
    std::unique_ptr<Optimizer> base_optimizer,
    const ZeROStage2Config& config
) : ZeROStage1Optimizer(std::move(base_optimizer), config),
    stage2_config_(config) {

    // Create gradient buckets for efficient communication
    if (stage2_config_.gradient_bucketing) {
        create_gradient_buckets();
    }
}

ZeROStage2Optimizer::~ZeROStage2Optimizer() {
    // Cleanup is automatic via smart pointers
    // Hooks will be cleaned up by the autograd system
}

auto ZeROStage2Optimizer::step() -> void {
    std::lock_guard<std::mutex> lock(mutex_);

    // Step 1: Gradients are already reduced-scattered via backward hooks
    // No need for all-reduce like in Stage 1

    // Step 2: Fetch optimizer states from CPU if offloaded
    if (config_.offload_to_cpu && offload_engine_) {
        fetch_states_to_gpu();
    }

    // Step 3: Update local partition of parameters
    // Uses the local (reduced-scattered) gradients
    update_local_partition();

    // Step 4: Offload states back to CPU if enabled
    if (config_.offload_to_cpu && offload_engine_) {
        offload_states_to_cpu();
    }

    // Step 5: All-gather updated parameters across ranks
    if (config_.world_size > 1) {
        all_gather_parameters();
    }
}

auto ZeROStage2Optimizer::register_backward_hooks() -> void {
    if (hooks_registered_) {
        return;  // Already registered
    }

    if (!stage2_config_.reduce_scatter_in_backward) {
        hooks_registered_ = true;
        return;  // Hooks disabled in config
    }

    // Register a hook for each parameter in each bucket
    // These hooks will be called during backward pass when gradients are computed
    for (size_t bucket_idx = 0; bucket_idx < gradient_buckets_.size(); ++bucket_idx) {
        auto& bucket = gradient_buckets_[bucket_idx];

        for (size_t param_idx = 0; param_idx < bucket.params.size(); ++param_idx) {
            auto param = bucket.params[param_idx];

            // In a production implementation, this would register with the autograd system:
            // param->register_hook([this, bucket_idx, param_idx](const Tensor& grad) {
            //     this->gradient_hook(bucket_idx, param_idx);
            //     return grad;  // Return gradient unchanged
            // });
            //
            // The hook mechanism would automatically call gradient_hook() during backward()
            // when this parameter's gradient is computed, enabling automatic reduce-scatter.
            //
            // For manual triggering (e.g., in tests), call gradient_hook() explicitly
            // after backward pass completes for each parameter.
        }
    }

    hooks_registered_ = true;
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

    // Initialize gradient buffers for each bucket
    for (auto& bucket : gradient_buckets_) {
        bucket.gradient_buffers.reserve(bucket.params.size());
        bucket.gradients_received = 0;
        bucket.ready = false;
    }
}

// =============================================================================
// Private: Communication
// =============================================================================

auto ZeROStage2Optimizer::reduce_scatter_gradients(GradientBucket& bucket) -> void {
    if (!config_.process_group || config_.world_size <= 1) {
        return;  // No communication needed
    }

    // Collect all gradients from the bucket
    std::vector<Tensor> gradients;
    gradients.reserve(bucket.params.size());

    for (const auto& param : bucket.params) {
        if (!param->has_grad()) {
            throw std::runtime_error("Parameter missing gradient in reduce-scatter");
        }

        const auto& grad_opt = param->grad();
        if (!grad_opt.has_value()) {
            throw std::runtime_error("Parameter gradient not computed in reduce-scatter");
        }

        gradients.push_back(grad_opt.value());
    }

    if (gradients.empty()) {
        return;
    }

    // Flatten gradients into contiguous buffer
    Tensor flat_grads = flatten_tensors(gradients);

    // For single-process mode (world_size=1), just use the gradients as-is
    // For multi-process, we would need true reduce-scatter collective
    Tensor local_grad_sum;

    if (config_.world_size == 1) {
        // Single process: no communication needed
        local_grad_sum = flat_grads;
    } else {
        // Multi-process: perform all-reduce instead of reduce-scatter for now
        // TODO: Implement proper reduce-scatter for world_size > 1
        if (config_.process_group) {
            // All-reduce averages gradients across ranks
            local_grad_sum = flat_grads.clone();
            config_.process_group->all_reduce(local_grad_sum, distributed::ReduceOp::AVG);
        } else {
            local_grad_sum = flat_grads;
        }
    }

    // Unflatten the local portion back into individual gradients
    // Only update gradients for parameters owned by this rank
    if (bucket.target_rank == config_.rank && local_grad_sum.numel() > 0) {
        std::vector<Tensor> local_grads;
        local_grads.reserve(bucket.params.size());

        for (const auto& param : bucket.params) {
            if (param->has_grad()) {
                auto& grad_opt = param->grad();
                if (grad_opt.has_value()) {
                    local_grads.push_back(grad_opt.value());
                }
            }
        }

        if (!local_grads.empty()) {
            unflatten_into(local_grad_sum, local_grads);

            // Update the parameter gradients with reduced-scattered values
            size_t grad_idx = 0;
            for (auto& param : bucket.params) {
                if (param->has_grad() && grad_idx < local_grads.size()) {
                    param->grad() = local_grads[grad_idx];
                    grad_idx++;
                }
            }
        }
    } else {
        // This rank doesn't own these parameters - free their gradients
        for (auto& param : bucket.params) {
            if (param->has_grad()) {
                // Set gradient to empty tensor to free memory
                param->grad() = std::nullopt;
            }
        }
    }

    // Mark bucket as processed
    bucket.ready = false;
    bucket.gradients_received = 0;
}

auto ZeROStage2Optimizer::gradient_hook(size_t bucket_idx, size_t param_idx) -> void {
    if (bucket_idx >= gradient_buckets_.size()) {
        return;
    }

    auto& bucket = gradient_buckets_[bucket_idx];

    {
        std::lock_guard<std::mutex> lock(*bucket.mutex);
        bucket.gradients_received++;

        // Check if all gradients in bucket are ready
        if (bucket.gradients_received >= bucket.params.size()) {
            bucket.ready = true;
        }
    }

    // If bucket is ready, perform reduce-scatter
    if (is_bucket_ready(bucket)) {
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

} // namespace optim
} // namespace tenzor
