/**
 * @file zero_optimizer.cpp
 * @brief Implementation of ZeRO Stage 1 Optimizer
 */

#include "tenzor/nn/optim/zero_optimizer.hpp"
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
    if (!config_.process_group) {
        if (distributed::is_initialized()) {
            config_.process_group = distributed::DistributedContext::get_process_group();
        } else if (config_.world_size > 1) {
            throw std::runtime_error(
                "Distributed not initialized. Call distributed::init_process_group() first"
            );
        }
    }

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

} // namespace optim
} // namespace tenzor
