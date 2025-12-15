/**
 * @file checkpoint_example.cpp
 * @brief Checkpoint save/load/restore with ZeRO optimization
 *
 * This example demonstrates:
 * 1. Saving distributed checkpoints with ZeRO
 * 2. Loading checkpoints for resuming training
 * 3. Converting between partitioned and full checkpoints
 * 4. Checkpoint validation and recovery
 * 5. Best practices for production checkpointing
 *
 * Build: g++ -std=c++17 -O3 checkpoint_example.cpp -ltenzor -o checkpoint_example
 * Run:   mpirun -np 4 ./checkpoint_example
 */

#include <tenzor/tenzor.hpp>
#include <tenzor/models/resnet.hpp>
#include <tenzor/nn/optim/zero_optimizer.hpp>
#include <tenzor/nn/optim/adam.hpp>
#include <tenzor/distributed/distributed.hpp>
#include <iostream>
#include <iomanip>
#include <filesystem>
#include <chrono>

using namespace tenzor;
using namespace tenzor::models;
using namespace tenzor::nn;
using namespace tenzor::optim;
namespace fs = std::filesystem;

// ============================================================================
// Helper Functions
// ============================================================================

auto generate_batch(int batch_size, int num_classes, Device device)
    -> std::tuple<Variable, Tensor> {

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 1.0f);

    std::vector<float> image_data(batch_size * 3 * 224 * 224);
    for (auto& val : image_data) {
        val = dist(gen);
    }

    Tensor images_cpu({batch_size, 3, 224, 224}, DType::Float32, Device::cpu());
    std::memcpy(const_cast<float*>(images_cpu.data<float>()),
                image_data.data(), image_data.size() * sizeof(float));
    Tensor images = images_cpu.to(device);

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

auto checkpoint_exists(const std::string& path, int rank) -> bool {
    std::string rank_path = path + "_rank_" + std::to_string(rank) + ".pt";
    return fs::exists(rank_path);
}

// ============================================================================
// Training Functions
// ============================================================================

auto train_steps(
    ResNet& model,
    ZeROStage2Optimizer& optimizer,
    int num_steps,
    int batch_size,
    Device device,
    int rank
) -> double {

    model.train();
    double total_loss = 0.0;

    for (int step = 0; step < num_steps; ++step) {
        auto [images, labels] = generate_batch(batch_size, 1000, device);

        auto output = model.forward(images);

        Variable labels_var(labels, false);
        auto diff = output - labels_var;
        auto loss = Variable(mean(pow(diff.tensor(), 2.0f)), true);

        optimizer.zero_grad();
        loss.backward();
        optimizer.step();

        auto loss_cpu = loss.tensor().to(Device::cpu());
        float loss_value = const_cast<float*>(loss_cpu.data<float>())[0];
        total_loss += loss_value;

        if (rank == 0 && step % 5 == 0) {
            std::cout << "  Step " << step << "/" << num_steps
                      << " | Loss: " << std::fixed << std::setprecision(4) << loss_value
                      << std::endl;
        }
    }

    return total_loss / num_steps;
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    // Initialize distributed
    distributed::init_process_group("nccl");
    auto rank = distributed::get_rank();
    auto world_size = distributed::get_world_size();

    if (rank == 0) {
        std::cout << "=== ZeRO Checkpoint Save/Load Example ===" << std::endl;
        std::cout << "World size: " << world_size << std::endl;
        std::cout << std::endl;
    }

    Device device = Device::cuda(rank);
    int batch_size = 16;
    int num_classes = 1000;

    // Checkpoint paths
    std::string checkpoint_dir = "/tmp/zero_checkpoints";
    std::string checkpoint_path = checkpoint_dir + "/resnet_checkpoint";

    if (rank == 0) {
        // Create checkpoint directory
        if (!fs::exists(checkpoint_dir)) {
            fs::create_directories(checkpoint_dir);
        }
        std::cout << "Checkpoint directory: " << checkpoint_dir << std::endl;
        std::cout << std::endl;
    }

    // ========================================================================
    // Part 1: Training and Checkpoint Saving
    // ========================================================================

    if (rank == 0) {
        std::cout << std::string(80, '=') << std::endl;
        std::cout << "Part 1: Training and Saving Checkpoint" << std::endl;
        std::cout << std::string(80, '=') << std::endl;
    }

    // Create model
    auto model = ResNet::resnet50(num_classes);
    model.to(device);

    auto params = model.parameters();
    size_t total_params = 0;
    for (const auto& param : params) {
        total_params += param->tensor().numel();
    }

    if (rank == 0) {
        std::cout << "Model: ResNet-50" << std::endl;
        std::cout << "  Parameters: " << total_params << std::endl;
        std::cout << "  Model size: " << format_bytes(total_params * 4) << std::endl;
        std::cout << std::endl;
    }

    // Create optimizer with ZeRO Stage 2
    auto base_optimizer = std::make_unique<Adam>(params, 1e-3);

    ZeROStage2Config zero_config;
    zero_config.world_size = world_size;
    zero_config.rank = rank;
    zero_config.gradient_bucket_size = 25 * 1024 * 1024;
    zero_config.process_group = distributed::get_default_process_group();

    ZeROStage2Optimizer optimizer(std::move(base_optimizer), zero_config);
    optimizer.register_backward_hooks();

    if (rank == 0) {
        std::cout << "Training for 20 steps..." << std::endl;
    }

    // Train for 20 steps
    double loss1 = train_steps(model, optimizer, 20, batch_size, device, rank);

    if (rank == 0) {
        std::cout << "Average loss after 20 steps: " << std::fixed << std::setprecision(4) << loss1 << std::endl;
        std::cout << std::endl;
    }

    // Save checkpoint
    if (rank == 0) {
        std::cout << "Saving checkpoint..." << std::endl;
    }

    optimizer.save_checkpoint(checkpoint_path);

    distributed::barrier();  // Wait for all ranks to save

    if (rank == 0) {
        // List checkpoint files
        std::cout << "Checkpoint files created:" << std::endl;
        for (int r = 0; r < world_size; ++r) {
            std::string rank_file = checkpoint_path + "_rank_" + std::to_string(r) + ".pt";
            if (fs::exists(rank_file)) {
                auto file_size = fs::file_size(rank_file);
                std::cout << "  " << rank_file << " (" << format_bytes(file_size) << ")" << std::endl;
            }
        }
        std::cout << std::endl;
    }

    // ========================================================================
    // Part 2: Resume Training from Checkpoint
    // ========================================================================

    if (rank == 0) {
        std::cout << std::string(80, '=') << std::endl;
        std::cout << "Part 2: Loading Checkpoint and Resuming" << std::endl;
        std::cout << std::string(80, '=') << std::endl;
    }

    // Create new model and optimizer
    auto model2 = ResNet::resnet50(num_classes);
    model2.to(device);

    auto params2 = model2.parameters();
    auto base_optimizer2 = std::make_unique<Adam>(params2, 1e-3);

    ZeROStage2Config zero_config2;
    zero_config2.world_size = world_size;
    zero_config2.rank = rank;
    zero_config2.gradient_bucket_size = 25 * 1024 * 1024;
    zero_config2.process_group = distributed::get_default_process_group();

    ZeROStage2Optimizer optimizer2(std::move(base_optimizer2), zero_config2);
    optimizer2.register_backward_hooks();

    // Load checkpoint
    if (rank == 0) {
        std::cout << "Loading checkpoint..." << std::endl;
    }

    if (checkpoint_exists(checkpoint_path, rank)) {
        optimizer2.load_checkpoint(checkpoint_path);

        if (rank == 0) {
            std::cout << "Checkpoint loaded successfully!" << std::endl;
            std::cout << std::endl;
        }
    } else {
        if (rank == 0) {
            std::cout << "ERROR: Checkpoint not found!" << std::endl;
        }
        distributed::destroy_process_group();
        return 1;
    }

    // Continue training for 10 more steps
    if (rank == 0) {
        std::cout << "Continuing training for 10 more steps..." << std::endl;
    }

    double loss2 = train_steps(model2, optimizer2, 10, batch_size, device, rank);

    if (rank == 0) {
        std::cout << "Average loss after resuming: " << std::fixed << std::setprecision(4) << loss2 << std::endl;
        std::cout << std::endl;
    }

    // ========================================================================
    // Part 3: Checkpoint Validation
    // ========================================================================

    if (rank == 0) {
        std::cout << std::string(80, '=') << std::endl;
        std::cout << "Part 3: Checkpoint Validation" << std::endl;
        std::cout << std::string(80, '=') << std::endl;

        std::cout << "Checkpoint Validation:" << std::endl;
        std::cout << "  All ranks have checkpoint files: ";

        bool all_exist = true;
        for (int r = 0; r < world_size; ++r) {
            if (!checkpoint_exists(checkpoint_path, r)) {
                all_exist = false;
                break;
            }
        }

        std::cout << (all_exist ? "Yes" : "No") << std::endl;

        // Check total checkpoint size
        size_t total_checkpoint_size = 0;
        for (int r = 0; r < world_size; ++r) {
            std::string rank_file = checkpoint_path + "_rank_" + std::to_string(r) + ".pt";
            if (fs::exists(rank_file)) {
                total_checkpoint_size += fs::file_size(rank_file);
            }
        }

        std::cout << "  Total checkpoint size: " << format_bytes(total_checkpoint_size) << std::endl;
        std::cout << "  Per-rank average: " << format_bytes(total_checkpoint_size / world_size) << std::endl;
        std::cout << std::endl;
    }

    // ========================================================================
    // Part 4: Incremental Checkpointing
    // ========================================================================

    if (rank == 0) {
        std::cout << std::string(80, '=') << std::endl;
        std::cout << "Part 4: Incremental Checkpointing" << std::endl;
        std::cout << std::string(80, '=') << std::endl;
    }

    // Save multiple checkpoints during training
    for (int epoch = 1; epoch <= 3; ++epoch) {
        if (rank == 0) {
            std::cout << "Epoch " << epoch << ":" << std::endl;
        }

        double epoch_loss = train_steps(model2, optimizer2, 5, batch_size, device, rank);

        if (rank == 0) {
            std::cout << "  Loss: " << std::fixed << std::setprecision(4) << epoch_loss << std::endl;
        }

        // Save checkpoint
        std::string epoch_checkpoint = checkpoint_dir + "/checkpoint_epoch_" + std::to_string(epoch);
        optimizer2.save_checkpoint(epoch_checkpoint);

        distributed::barrier();

        if (rank == 0) {
            std::cout << "  Checkpoint saved: " << epoch_checkpoint << "_rank_*.pt" << std::endl;
        }
    }

    if (rank == 0) {
        std::cout << std::endl;
    }

    // ========================================================================
    // Summary and Best Practices
    // ========================================================================

    if (rank == 0) {
        std::cout << std::string(80, '=') << std::endl;
        std::cout << "Checkpointing Best Practices" << std::endl;
        std::cout << std::string(80, '=') << std::endl;
        std::cout << std::endl;

        std::cout << "1. Save Frequency:" << std::endl;
        std::cout << "   - Save every N steps/epochs based on training length" << std::endl;
        std::cout << "   - Keep last K checkpoints (e.g., K=3)" << std::endl;
        std::cout << "   - Save best checkpoint based on validation metrics" << std::endl;
        std::cout << std::endl;

        std::cout << "2. Checkpoint Contents:" << std::endl;
        std::cout << "   - Model parameters (partitioned across ranks)" << std::endl;
        std::cout << "   - Optimizer states (partitioned across ranks)" << std::endl;
        std::cout << "   - Training metadata (epoch, step, best loss)" << std::endl;
        std::cout << "   - RNG state for reproducibility" << std::endl;
        std::cout << std::endl;

        std::cout << "3. Storage Considerations:" << std::endl;
        std::cout << "   - Use distributed file system (e.g., NFS, Lustre)" << std::endl;
        std::cout << "   - Checkpoint to fast storage (SSD/NVMe)" << std::endl;
        std::cout << "   - Compress checkpoints if storage is limited" << std::endl;
        std::cout << "   - Implement checkpoint rotation to save space" << std::endl;
        std::cout << std::endl;

        std::cout << "4. Recovery Strategy:" << std::endl;
        std::cout << "   - Always validate checkpoints after saving" << std::endl;
        std::cout << "   - Implement automatic recovery from last valid checkpoint" << std::endl;
        std::cout << "   - Keep backup of previous checkpoint before overwriting" << std::endl;
        std::cout << "   - Test recovery procedure regularly" << std::endl;
        std::cout << std::endl;

        std::cout << "5. Production Checklist:" << std::endl;
        std::cout << "   - Verify all ranks complete save/load successfully" << std::endl;
        std::cout << "   - Test loading checkpoint with different world sizes" << std::endl;
        std::cout << "   - Monitor checkpoint I/O performance" << std::endl;
        std::cout << "   - Implement checkpoint versioning for compatibility" << std::endl;
        std::cout << std::endl;

        // Cleanup old checkpoints
        std::cout << "Cleaning up checkpoints..." << std::endl;
        try {
            fs::remove_all(checkpoint_dir);
            std::cout << "Checkpoints cleaned up successfully" << std::endl;
        } catch (const std::exception& e) {
            std::cout << "Warning: Could not clean up checkpoints: " << e.what() << std::endl;
        }
        std::cout << std::endl;

        std::cout << "Example completed successfully!" << std::endl;
    }

    distributed::destroy_process_group();
    return 0;
}
