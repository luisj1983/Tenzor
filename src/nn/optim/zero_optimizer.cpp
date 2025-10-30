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
        // Multi-process: perform reduce-scatter to partition gradients
        if (config_.process_group) {
            // Split flat gradients into world_size chunks for reduce-scatter
            // This is the core of Stage 2 - gradient partitioning!
            int64_t total_elements = flat_grads.numel();
            int64_t chunk_size = (total_elements + config_.world_size - 1) / config_.world_size;

            std::vector<Tensor> gradient_chunks;
            gradient_chunks.reserve(config_.world_size);

            for (int rank = 0; rank < config_.world_size; ++rank) {
                int64_t start_idx = rank * chunk_size;
                int64_t end_idx = std::min(start_idx + chunk_size, total_elements);

                if (start_idx < total_elements) {
                    // Extract chunk for this rank
                    Tensor chunk = flat_grads.slice(0, start_idx, end_idx);
                    gradient_chunks.push_back(chunk);
                } else {
                    // Padding chunk if we've run out of elements
                    Tensor empty_chunk = zeros({0}, flat_grads.dtype(), flat_grads.device());
                    gradient_chunks.push_back(empty_chunk);
                }
            }

            // Allocate output tensor for this rank's portion
            int64_t local_start = config_.rank * chunk_size;
            int64_t local_end = std::min(local_start + chunk_size, total_elements);
            int64_t local_size = std::max(int64_t(0), local_end - local_start);

            local_grad_sum = zeros({local_size}, flat_grads.dtype(), flat_grads.device());

            // Reduce-scatter: Each rank receives 1/N of the reduced gradients
            // After this, local_grad_sum contains the SUM of all ranks' contributions
            // for the gradient chunk owned by this rank
            config_.process_group->reduce_scatter(gradient_chunks, local_grad_sum,
                                                 distributed::ReduceOp::SUM);
        } else {
            // No process group configured - treat as single rank
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

// =============================================================================
// ZeRO Stage 3 Optimizer Implementation
// =============================================================================

ZeROStage3Optimizer::ZeROStage3Optimizer(
    std::unique_ptr<Optimizer> base_optimizer,
    const Stage3Config& config
) : ZeROStage2Optimizer(std::move(base_optimizer), config),
    stage3_config_(config),
    registered_model_(nullptr) {

    // Initialize performance stats
    perf_stats_ = PerformanceStats{};

    // Initialize CUDA streams for communication/compute overlap
    // Note: CUDAStream is used if available, but we keep the code flexible
    // if (stage3_config_.use_separate_streams) {
    //     gather_stream_ = core::CUDAStream(stage3_config_.gather_stream_priority);
    //     scatter_stream_ = core::CUDAStream(stage3_config_.gather_stream_priority);
    // }

    // Initialize prefetch scheduler
    // Note: PrefetchScheduler is defined later in the file, so we can't initialize it here
    // It will be initialized on first use if needed
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

    // Step 3: Pin first and last layer parameters if configured
    // Note: Module doesn't expose a modules() method, so we skip this for now
    // This feature can be implemented when Module's submodule traversal is available
}

auto ZeROStage3Optimizer::unregister_model() -> void {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!registered_model_) {
        return;
    }

    // Clear all hooks
    forward_hooks_.clear();
    backward_hooks_.clear();

    // Clear parameter states
    param_states_.clear();

    registered_model_ = nullptr;
}

auto ZeROStage3Optimizer::step() -> void {
    std::lock_guard<std::mutex> lock(mutex_);

    // Stage 3 step algorithm:
    // 1. Gradients are already reduced-scattered via backward hooks (inherited from Stage 2)
    // 2. Fetch optimizer states from CPU if offloaded
    // 3. Update local partition of parameters (parameters remain partitioned)
    // 4. Offload states back to CPU if enabled
    // 5. NO all-gather needed - parameters stay partitioned!

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

    // Note: Unlike Stage 1/2, we do NOT all-gather parameters here
    // They will be gathered on-demand during next forward pass
}

auto ZeROStage3Optimizer::zero_grad() -> void {
    // Zero gradients for local partition only
    auto& partition = local_partition();
    for (auto& param : partition.params) {
        if (param->has_grad()) {
            param->grad() = std::nullopt;
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

    // Case 1: Already gathered - just increment ref count
    if (state.is_gathered) {
        state.acquire();
        perf_stats_.prefetch_hits++;
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

    // Start prefetch for next parameters (speculation)
    if (prefetch_scheduler_) {
        // Unlock to allow prefetch scheduler to work
        param_states_mutex_.unlock();
        prefetch_next_parameters(nullptr);
        param_states_mutex_.lock();
    }

    // Perform synchronous all-gather
    gather_parameter_impl(state);

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

    // Check if pinned
    if (state.pinned_in_memory) {
        return;
    }

    // Free the gathered parameter
    if (state.is_gathered) {
        state.full_param = Tensor();  // Release GPU memory
        state.is_gathered = false;

        // Update statistics
        perf_stats_.current_gathered_memory -= state.size_bytes;
    }

    // Optionally offload local partition to CPU
    if (config_.offload_to_cpu && offload_engine_ && !state.partition_on_cpu) {
        offload_engine_->offload_to_cpu_async(state.local_partition);
        state.partition_on_cpu = true;
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

        // Start async gather (synchronous for now)
        // NOTE: PrefetchScheduler would be used here for async operations,
        // but it requires a complete type definition which appears later in the file.
        // Future enhancement: Move PrefetchScheduler class definition earlier
        // or refactor to use pImpl pattern for better encapsulation.
        //
        // Ideal implementation with async support:
        //   int priority = 100 - state.layer_index;  // Earlier layers = higher priority
        //   prefetch_scheduler_->schedule_prefetch(state, priority);
        //
        // For now, perform synchronous gather which achieves parameter prefetching
        // but without latency hiding through async communication overlap.
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

        if (overlap_end > overlap_start) {
            // This rank owns part of this parameter
            // Extract local partition
            state.local_partition = param_tensor.slice(0, overlap_start - param_start, overlap_end - param_start);

            // Replace full parameter with partition
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
}

auto ZeROStage3Optimizer::register_gather_scatter_hooks(Module& model) -> void {
    // Register forward pre-hook and backward post-hook for the model
    // Note: Module doesn't expose a modules() method, so we just register for the root model
    // In a full implementation, this would recursively register hooks on all submodules

    auto params = model.parameters();
    if (params.empty()) {
        return;
    }

    // Create forward pre-hook for the model
    ForwardPreHook forward_hook;
    forward_hook.module = &model;
    // Convert shared_ptr<Variable> to Tensor* for storage
    forward_hook.params.reserve(params.size());
    for (const auto& param_ptr : params) {
        forward_hook.params.push_back(&param_ptr->tensor());
    }
    forward_hook.hook_fn = [this, &model](Module*, const std::vector<Tensor>&) {
        this->forward_pre_hook(&model, {});
    };
    forward_hook.hook_id = next_hook_id_++;

    forward_hooks_.push_back(std::move(forward_hook));

    // Create backward post-hook for the model
    BackwardPostHook backward_hook;
    backward_hook.module = &model;
    // Convert shared_ptr<Variable> to Tensor* for storage
    backward_hook.params.reserve(params.size());
    for (const auto& param_ptr : params) {
        backward_hook.params.push_back(&param_ptr->tensor());
    }
    backward_hook.hook_fn = [this, &model](Module*, const std::vector<Tensor>& inputs, const std::vector<Tensor>& grad_outputs) {
        this->backward_post_hook(&model, inputs, grad_outputs);
    };
    backward_hook.hook_id = next_hook_id_++;

    backward_hooks_.push_back(std::move(backward_hook));
}

auto ZeROStage3Optimizer::gather_parameter_impl(ParameterInfo& state) -> void {
    if (!config_.process_group) {
        throw std::runtime_error("Process group not initialized");
    }

    auto start_time = std::chrono::steady_clock::now();

    // Allocate buffer for full parameter
    auto shape_span = state.param->shape();
    std::vector<int64_t> full_shape(shape_span.begin(), shape_span.end());
    state.full_param = zeros(full_shape, state.param->dtype(), state.param->device());

    // All-gather: collect partitions from all ranks
    if (config_.world_size > 1) {
        std::vector<Tensor> gathered_parts(config_.world_size);
        config_.process_group->all_gather(
            state.local_partition,  // input: local partition
            gathered_parts          // output: gathered partitions from all ranks
        );

        // Concatenate gathered parts into full parameter
        // For simplicity, assume the first gathered part is the full parameter
        // In a full implementation, this would concatenate all partitions
        if (!gathered_parts.empty()) {
            state.full_param = gathered_parts[config_.rank];
        }
    } else {
        // Single rank: just copy
        state.full_param = state.local_partition.clone();
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

auto ZeROStage3Optimizer::forward_pre_hook(Module* module, const std::vector<Tensor>& inputs) -> void {
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
            auto& state = it->second;

            // Gather parameter (handles prefetch hits)
            Tensor full_param = gather_parameter(param);

            // Replace module's parameter with gathered version
            *param = full_param;
        }
    }
}

auto ZeROStage3Optimizer::backward_post_hook(Module* module, const std::vector<Tensor>& inputs, const std::vector<Tensor>& grad_outputs) -> void {
    // Find parameters for this module
    auto params = module->parameters();

    // Reduce-scatter gradients (inherited from Stage 2)
    for (const auto& param_ptr : params) {
        Tensor* param = &param_ptr->tensor();
        if (param_ptr->has_grad()) {
            scatter_parameter_gradient(param);
        }
    }

    // Free gathered parameters
    for (const auto& param_ptr : params) {
        Tensor* param = &param_ptr->tensor();
        free_gathered_parameter(param);
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

    // Split gradient into chunks (one per rank)
    std::vector<Tensor> gradient_chunks;
    gradient_chunks.reserve(config_.world_size);

    for (int rank = 0; rank < config_.world_size; ++rank) {
        size_t rank_start = rank * elements_per_rank;
        size_t rank_end = std::min(rank_start + elements_per_rank, total_elements);

        if (rank_start < total_elements) {
            Tensor chunk = flat_grad.slice(0, rank_start, rank_end);
            gradient_chunks.push_back(chunk);
        } else {
            // For uneven partitioning, add empty chunk
            Tensor empty_chunk = zeros({0}, flat_grad.dtype(), flat_grad.device());
            gradient_chunks.push_back(empty_chunk);
        }
    }

    // Reduce-scatter: Each rank receives the sum of its chunk from all ranks
    config_.process_group->reduce_scatter(gradient_chunks, local_grad,
                                         distributed::ReduceOp::SUM);

    // Store the local gradient partition
    state.local_partition = local_grad;

    // Update the Variable's gradient with local partition
    // This frees the full gradient and saves memory
    param_var->grad() = local_grad;
}

auto ZeROStage3Optimizer::prefetch_next_parameters(Module* current_module) -> void {
    if (!prefetch_scheduler_ || !registered_model_) {
        return;
    }

    // For now, without a modules() method, we can't traverse the execution graph
    // This would require Module to expose its submodules in a traversable way
    // In a full implementation, this would prefetch parameters from upcoming modules
    // based on the execution order of the model's computational graph
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
// PrefetchScheduler Implementation
// =============================================================================

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

        // Check if already in queue or in-flight
        if (in_flight_.count(param_state.param) > 0) {
            return;
        }

        // Check buffer size limit
        if (current_buffer_size_ + param_state.size_bytes > config_.max_buffer_bytes) {
            return;  // Buffer full
        }

        // Add to priority queue
        PrefetchRequest request;
        request.param_state = &param_state;
        request.priority = priority;

        queue_.push(request);
        current_buffer_size_ += param_state.size_bytes;

        // Execute if capacity available
        execute_pending();
    }

    auto execute_pending() -> void {
        while (in_flight_.size() < static_cast<size_t>(config_.max_concurrent) && !queue_.empty()) {
            auto request = queue_.top();
            queue_.pop();

            // Start async gather
            start_async_gather(*request.param_state);

            in_flight_.insert(request.param_state->param);
        }
    }

private:
    struct PrefetchRequest {
        ParameterInfo* param_state;
        int priority;

        bool operator<(const PrefetchRequest& other) const {
            return priority < other.priority;  // Higher priority first
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

        // Mark as prefetching
        param_state.is_prefetching = true;

        // For now, perform synchronous gather
        // In a full implementation, this would use async NCCL operations
        optimizer_->gather_parameter_impl(param_state);
    }
};

// =============================================================================
// Additional Stage 3 Methods
// =============================================================================

auto ZeROStage3Optimizer::gather_full_state() -> std::unordered_map<std::string, Tensor> {
    // Gather full optimizer state from all ranks for checkpointing
    std::unordered_map<std::string, Tensor> full_state;

    // This is a simplified implementation - a full implementation would
    // need to gather state from all ranks using collective communication
    full_state = state_dict();

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

auto ZeROStage3Optimizer::free_gathered_parameter_impl(ParameterInfo& state) -> void {
    // Internal implementation for freeing gathered parameters
    // This is already handled in free_gathered_parameter()
}

auto ZeROStage3Optimizer::get_next_module_in_execution_order(Module* current_module) -> Module* {
    // Without a modules() method, we can't traverse the execution graph
    // Return nullptr for now
    return nullptr;
}

auto ZeROStage3Optimizer::get_next_parameters_in_execution_order(const ParameterInfo& state)
    -> std::vector<Tensor*> {
    // Without a modules() method, we can't traverse the execution graph
    // Return empty vector for now
    return std::vector<Tensor*>();
}

auto ZeROStage3Optimizer::flatten_tensors(const std::vector<Tensor>& tensors) -> Tensor {
    // Use the gradient_utils implementation
    return tenzor::optim::flatten_tensors(tensors);
}

auto ZeROStage3Optimizer::unflatten_into(const Tensor& flattened, std::vector<Tensor>& targets) -> void {
    // Use the gradient_utils implementation
    tenzor::optim::unflatten_into(flattened, targets);
}

} // namespace optim
} // namespace tenzor
