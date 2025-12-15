/**
 * @file distributed_training.cpp
 * @brief Multi-GPU distributed training setup with ZeRO
 *
 * This example demonstrates:
 * 1. Setting up distributed training environment (NCCL/Gloo)
 * 2. Data parallelism with ZeRO optimization
 * 3. Distributed data loading and synchronization
 * 4. Multi-node training configuration
 * 5. Monitoring and debugging distributed training
 *
 * Run configurations:
 * - Single node, multi-GPU: mpirun -np 4 ./distributed_training
 * - Multi-node: mpirun -np 16 -H node1:8,node2:8 ./distributed_training
 *
 * Build: g++ -std=c++17 -O3 distributed_training.cpp -ltenzor -o distributed_training
 */

#include <tenzor/tenzor.hpp>
#include <tenzor/models/resnet.hpp>
#include <tenzor/nn/optim/zero_optimizer.hpp>
#include <tenzor/nn/optim/sgd.hpp>
#include <tenzor/distributed/distributed.hpp>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <random>

using namespace tenzor;
using namespace tenzor::models;
using namespace tenzor::nn;
using namespace tenzor::optim;

// ============================================================================
// Distributed Configuration
// ============================================================================

struct DistributedConfig {
    std::string backend = "nccl";  // nccl, gloo, or mpi
    std::string master_addr = "localhost";
    std::string master_port = "12355";
    int world_size = 1;
    int rank = 0;
    int local_rank = 0;  // GPU ID on current node
    bool use_zero = true;
    int zero_stage = 2;  // 1, 2, or 3
};

// ============================================================================
// Data Generation (Simulates ImageNet-like data)
// ============================================================================

auto generate_image_batch(int batch_size, int channels, int height, int width,
                         int num_classes, Device device, int rank)
    -> std::tuple<Variable, Tensor> {

    // Different random seed per rank for different data
    std::random_device rd;
    std::mt19937 gen(rd() + rank);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    // Generate images
    std::vector<float> image_data(batch_size * channels * height * width);
    for (auto& val : image_data) {
        val = dist(gen);
    }

    Tensor images_cpu({batch_size, channels, height, width}, DType::Float32, Device::cpu());
    std::memcpy(const_cast<float*>(images_cpu.data<float>()),
                image_data.data(), image_data.size() * sizeof(float));
    Tensor images = images_cpu.to(device);

    // Generate labels
    std::uniform_int_distribution<int64_t> label_dist(0, num_classes - 1);
    std::vector<int64_t> labels_data(batch_size);
    for (auto& label : labels_data) {
        label = label_dist(gen);
    }

    Tensor labels_cpu({batch_size}, DType::Int64, Device::cpu());
    std::memcpy(const_cast<int64_t*>(labels_cpu.data<int64_t>()),
                labels_data.data(), labels_data.size() * sizeof(int64_t));
    Tensor labels = labels_cpu.to(device);

    return {Variable(images, true), labels};
}

// ============================================================================
// Helper Functions
// ============================================================================

auto format_bytes(size_t bytes) -> std::string {
    const char* units[] = {"B", "KB", "MB", "GB"};
    int unit_idx = 0;
    double size = static_cast<double>(bytes);

    while (size >= 1024.0 && unit_idx < 3) {
        size /= 1024.0;
        unit_idx++;
    }

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2) << size << " " << units[unit_idx];
    return oss.str();
}

auto print_distributed_info(const DistributedConfig& config) -> void {
    if (config.rank == 0) {
        std::cout << "Distributed Training Configuration:" << std::endl;
        std::cout << "  Backend: " << config.backend << std::endl;
        std::cout << "  World size: " << config.world_size << std::endl;
        std::cout << "  Master: " << config.master_addr << ":" << config.master_port << std::endl;
        std::cout << "  ZeRO enabled: " << (config.use_zero ? "Yes" : "No") << std::endl;
        if (config.use_zero) {
            std::cout << "  ZeRO stage: " << config.zero_stage << std::endl;
        }
        std::cout << std::endl;
    }
}

// ============================================================================
// Training Loop
// ============================================================================

template<typename OptimizerType>
auto train_distributed(
    ResNet& model,
    OptimizerType& optimizer,
    const DistributedConfig& config,
    int num_epochs,
    int steps_per_epoch,
    int batch_size,
    Device device
) -> void {

    model.train();

    for (int epoch = 1; epoch <= num_epochs; ++epoch) {
        double epoch_loss = 0.0;
        auto epoch_start = std::chrono::steady_clock::now();

        for (int step = 0; step < steps_per_epoch; ++step) {
            // Generate batch (different data per rank)
            auto [images, labels] = generate_image_batch(
                batch_size, 3, 224, 224, 1000, device, config.rank
            );

            // Forward pass
            auto output = model.forward(images);

            // Compute loss
            Variable labels_var(labels, false);
            auto diff = output - labels_var;
            auto loss = Variable(mean(pow(diff.tensor(), 2.0f)), true);

            // Backward pass
            optimizer.zero_grad();
            loss.backward();

            // Optimizer step (handles gradient synchronization)
            optimizer.step();

            auto loss_cpu = loss.tensor().to(Device::cpu());
            float loss_value = const_cast<float*>(loss_cpu.data<float>())[0];
            epoch_loss += loss_value;

            if (config.rank == 0 && step % 10 == 0) {
                std::cout << "  Epoch " << epoch << " | Step " << step << "/" << steps_per_epoch
                          << " | Loss: " << std::fixed << std::setprecision(4) << loss_value
                          << std::endl;
            }
        }

        auto epoch_end = std::chrono::steady_clock::now();
        auto epoch_time = std::chrono::duration<double>(epoch_end - epoch_start).count();

        if (config.rank == 0) {
            double avg_loss = epoch_loss / steps_per_epoch;
            double global_throughput = (steps_per_epoch * batch_size * config.world_size) / epoch_time;

            std::cout << "Epoch " << epoch << " completed:" << std::endl;
            std::cout << "  Average loss: " << avg_loss << std::endl;
            std::cout << "  Epoch time: " << epoch_time << " seconds" << std::endl;
            std::cout << "  Global throughput: " << global_throughput << " images/sec" << std::endl;
            std::cout << std::string(80, '-') << std::endl;
        }

        // Synchronize all ranks
        distributed::barrier();
    }
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    // Parse environment variables for distributed setup
    DistributedConfig config;

    if (const char* world_size_env = std::getenv("WORLD_SIZE")) {
        config.world_size = std::atoi(world_size_env);
    }
    if (const char* rank_env = std::getenv("RANK")) {
        config.rank = std::atoi(rank_env);
    }
    if (const char* local_rank_env = std::getenv("LOCAL_RANK")) {
        config.local_rank = std::atoi(local_rank_env);
    }
    if (const char* master_addr = std::getenv("MASTER_ADDR")) {
        config.master_addr = master_addr;
    }
    if (const char* master_port = std::getenv("MASTER_PORT")) {
        config.master_port = master_port;
    }

    // Initialize distributed
    distributed::init_process_group(config.backend, config.master_addr,
                                   std::stoi(config.master_port), config.world_size, config.rank);

    config.world_size = distributed::get_world_size();
    config.rank = distributed::get_rank();

    if (config.rank == 0) {
        std::cout << "=== Distributed Training with ZeRO ===" << std::endl;
        std::cout << std::endl;
    }

    print_distributed_info(config);

    // Set device to local rank
    Device device = Device::cuda(config.local_rank);

    if (config.rank == 0) {
        std::cout << "Each rank using device: cuda:" << config.local_rank << std::endl;
        std::cout << std::endl;
    }

    // Create model (ResNet-50)
    auto model = ResNet::resnet50(1000);
    model.to(device);

    auto params = model.parameters();
    size_t total_params = 0;
    for (const auto& param : params) {
        total_params += param->tensor().numel();
    }

    if (config.rank == 0) {
        std::cout << "Model: ResNet-50" << std::endl;
        std::cout << "  Parameters: " << total_params << std::endl;
        std::cout << "  Model size: " << format_bytes(total_params * 4) << std::endl;
        std::cout << std::endl;
    }

    // Training configuration
    int batch_size = 32;  // Per GPU
    int num_epochs = 5;
    int steps_per_epoch = 100;
    double learning_rate = 0.1;
    double momentum = 0.9;

    if (config.rank == 0) {
        std::cout << "Training Configuration:" << std::endl;
        std::cout << "  Batch size per GPU: " << batch_size << std::endl;
        std::cout << "  Global batch size: " << (batch_size * config.world_size) << std::endl;
        std::cout << "  Epochs: " << num_epochs << std::endl;
        std::cout << "  Steps per epoch: " << steps_per_epoch << std::endl;
        std::cout << "  Learning rate: " << learning_rate << std::endl;
        std::cout << std::endl;
    }

    // Create optimizer with ZeRO
    if (config.use_zero && config.zero_stage >= 1) {
        auto base_optimizer = std::make_unique<SGD>(params, learning_rate, momentum);

        if (config.zero_stage == 1) {
            // ZeRO Stage 1
            ZeROStage1Config zero_config;
            zero_config.world_size = config.world_size;
            zero_config.rank = config.rank;
            zero_config.process_group = distributed::get_default_process_group();

            ZeROStage1Optimizer optimizer(std::move(base_optimizer), zero_config);

            if (config.rank == 0) {
                std::cout << "Using ZeRO Stage 1 Optimizer" << std::endl;
                std::cout << "  Local parameters: " << optimizer.local_param_count() << std::endl;
                std::cout << std::endl;
                std::cout << "Starting training..." << std::endl;
                std::cout << std::string(80, '=') << std::endl;
            }

            train_distributed(model, optimizer, config, num_epochs, steps_per_epoch, batch_size, device);

            if (config.rank == 0) {
                auto stats = optimizer.get_memory_stats();
                std::cout << std::endl;
                std::cout << "Memory Statistics:" << std::endl;
                std::cout << "  Optimizer memory: " << format_bytes(stats.gpu_optimizer_memory) << std::endl;
            }

        } else if (config.zero_stage == 2) {
            // ZeRO Stage 2
            ZeROStage2Config zero_config;
            zero_config.world_size = config.world_size;
            zero_config.rank = config.rank;
            zero_config.gradient_bucket_size = 25 * 1024 * 1024;
            zero_config.process_group = distributed::get_default_process_group();

            ZeROStage2Optimizer optimizer(std::move(base_optimizer), zero_config);
            optimizer.register_backward_hooks();

            if (config.rank == 0) {
                std::cout << "Using ZeRO Stage 2 Optimizer" << std::endl;
                auto bucket_stats = optimizer.get_bucket_stats();
                std::cout << "  Gradient buckets: " << bucket_stats.num_buckets << std::endl;
                std::cout << std::endl;
                std::cout << "Starting training..." << std::endl;
                std::cout << std::string(80, '=') << std::endl;
            }

            train_distributed(model, optimizer, config, num_epochs, steps_per_epoch, batch_size, device);

            if (config.rank == 0) {
                auto stats = optimizer.get_memory_stats();
                std::cout << std::endl;
                std::cout << "Memory Statistics:" << std::endl;
                std::cout << "  Optimizer memory: " << format_bytes(stats.gpu_optimizer_memory) << std::endl;
                std::cout << "  Gradient memory: " << format_bytes(stats.gpu_gradient_memory) << std::endl;
            }

        } else if (config.zero_stage == 3) {
            // ZeRO Stage 3
            Stage3Config zero_config;
            zero_config.world_size = config.world_size;
            zero_config.rank = config.rank;
            zero_config.prefetch_depth = 2;
            zero_config.process_group = distributed::get_default_process_group();

            ZeROStage3Optimizer optimizer(std::move(base_optimizer), zero_config);
            optimizer.register_model(model);

            if (config.rank == 0) {
                std::cout << "Using ZeRO Stage 3 Optimizer" << std::endl;
                std::cout << std::endl;
                std::cout << "Starting training..." << std::endl;
                std::cout << std::string(80, '=') << std::endl;
            }

            train_distributed(model, optimizer, config, num_epochs, steps_per_epoch, batch_size, device);

            if (config.rank == 0) {
                auto stats = optimizer.get_stats();
                std::cout << std::endl;
                std::cout << "Performance Statistics:" << std::endl;
                std::cout << "  Prefetch hit rate: " << (stats.prefetch_hit_rate * 100) << "%" << std::endl;
                std::cout << "  Overlap efficiency: " << (stats.overlap_efficiency * 100) << "%" << std::endl;
            }
        }
    } else {
        // Standard distributed training (no ZeRO)
        SGD optimizer(params, learning_rate, momentum);

        if (config.rank == 0) {
            std::cout << "Using standard SGD optimizer (no ZeRO)" << std::endl;
            std::cout << std::endl;
            std::cout << "Starting training..." << std::endl;
            std::cout << std::string(80, '=') << std::endl;
        }

        train_distributed(model, optimizer, config, num_epochs, steps_per_epoch, batch_size, device);
    }

    if (config.rank == 0) {
        std::cout << std::string(80, '=') << std::endl;
        std::cout << "Training completed successfully!" << std::endl;
    }

    // Cleanup
    distributed::destroy_process_group();

    return 0;
}
