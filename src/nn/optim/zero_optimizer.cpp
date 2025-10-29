/**
 * @file zero_optimizer.cpp
 * @brief Implementation of ZeRO Stage 1 Optimizer
 */

#include "tenzor/nn/optim/zero_optimizer.hpp"
#include "tenzor/nn/optim/adam.hpp"
#include "tenzor/nn/optim/sgd.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/creation.hpp"
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>

namespace tenzor {
namespace optim {

// =============================================================================
// Constructor & Destructor
// =============================================================================

ZeROStage1Optimizer::ZeROStage1Optimizer(
    std::unique_ptr<Optimizer> base_optimizer,
    const ZeROStage1Config& config
) : Optimizer(base_optimizer->parameters()),
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
    std::string rank_path = path_prefix + "_rank_" + std::to_string(config_.rank) + ".pt";
    
    auto state = state_dict();
    // TODO: Implement proper serialization using tenzor::save()
    // For now, just save metadata
    
    if (config_.rank == 0) {
        // Master rank saves metadata
        std::string metadata_path = path_prefix + "_metadata.txt";
        std::ofstream meta_file(metadata_path);
        meta_file << "world_size=" << config_.world_size << "\n";
        meta_file << "num_partitions=" << partitions_.size() << "\n";
        meta_file.close();
    }
}

auto ZeROStage1Optimizer::load_checkpoint(const std::string& path_prefix) -> void {
    std::string rank_path = path_prefix + "_rank_" + std::to_string(config_.rank) + ".pt";
    
    // TODO: Implement proper deserialization using tenzor::load()
    // For now, this is a placeholder
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
    
    // Create partitions for all ranks
    partitions_.resize(config_.world_size);
    
    for (int rank = 0; rank < config_.world_size; ++rank) {
        auto& partition = partitions_[rank];
        partition.rank = rank;
        partition.device = config_.offload_to_cpu ? Device::cpu() : Device::cuda(rank);
        
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
    
    // Create temporary optimizer for local partition
    // We can't directly call base_optimizer_->step() because it would
    // try to update all parameters, not just our partition
    
    for (size_t i = 0; i < partition.params.size(); ++i) {
        auto& param = partition.params[i];
        
        if (!param->has_grad()) {
            continue;
        }
        
        // Simple SGD update as fallback
        // TODO: Integrate with actual base optimizer type
        Tensor& param_data = param->tensor();
        const auto& grad_opt = param->grad();
        if (!grad_opt.has_value()) {
            continue;
        }
        const Tensor& grad = grad_opt.value();
        
        // Momentum update
        Tensor& momentum = partition.momentum[i];
        float beta = 0.9f;
        float lr = 0.001f;  // Default learning rate
        
        momentum = momentum * beta + grad * (1.0f - beta);
        param_data = param_data - momentum * lr;
    }
}

auto ZeROStage1Optimizer::fetch_states_to_gpu() -> void {
    if (!offload_engine_) {
        return;
    }
    
    auto& partition = local_partition();
    
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
