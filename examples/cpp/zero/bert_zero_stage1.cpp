/**
 * @file bert_zero_stage1.cpp
 * @brief BERT training with ZeRO Stage 1 (Optimizer State Partitioning)
 *
 * This example demonstrates:
 * 1. Training BERT with ZeRO Stage 1 for 4x memory reduction
 * 2. Optimizer state partitioning across distributed ranks
 * 3. Parameter gradient all-reduce
 * 4. Memory usage tracking and performance monitoring
 * 5. Checkpoint save/load with distributed state
 *
 * Memory Savings:
 * - Optimizer states partitioned across ranks (4x reduction for Adam)
 * - Parameters remain replicated
 * - Gradients remain replicated
 *
 * Build: g++ -std=c++17 -O3 bert_zero_stage1.cpp -ltenzor -o bert_zero_stage1
 * Run:   mpirun -np 4 ./bert_zero_stage1
 */

#include <tenzor/tenzor.hpp>
#include <tenzor/models/bert.hpp>
#include <tenzor/nn/optim/zero_optimizer.hpp>
#include <tenzor/nn/optim/adam.hpp>
#include <tenzor/distributed/distributed.hpp>
#include "../common.hpp"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <random>

using namespace tenzor;
using namespace tenzor::models;
using namespace tenzor::nn;
using namespace tenzor::optim;

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * @brief Generate synthetic training batch
 */
auto generate_batch(int batch_size, int seq_length, int vocab_size, Device device)
    -> std::tuple<Variable, Tensor, Tensor> {

    // Random input IDs
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int64_t> dist(0, vocab_size - 1);

    std::vector<int64_t> input_ids_data(batch_size * seq_length);
    for (auto& id : input_ids_data) {
        id = dist(gen);
    }

    Tensor input_ids_cpu({batch_size, seq_length}, DType::Int64, Device::cpu());
    std::memcpy(const_cast<int64_t*>(input_ids_cpu.data<int64_t>()),
                input_ids_data.data(),
                input_ids_data.size() * sizeof(int64_t));

    Tensor input_ids = input_ids_cpu.to(device);

    // Attention mask (all 1s for simplicity)
    Tensor attention_mask = ones({batch_size, seq_length}, DType::Float32, device);

    // Labels (random classification targets)
    std::uniform_int_distribution<int64_t> label_dist(0, 1);
    std::vector<int64_t> labels_data(batch_size);
    for (auto& label : labels_data) {
        label = label_dist(gen);
    }

    Tensor labels_cpu({batch_size}, DType::Int64, Device::cpu());
    std::memcpy(const_cast<int64_t*>(labels_cpu.data<int64_t>()),
                labels_data.data(),
                labels_data.size() * sizeof(int64_t));

    Tensor labels = labels_cpu.to(device);

    return {Variable(input_ids, true), attention_mask, labels};
}

/**
 * @brief Format bytes as human-readable string
 */
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

/**
 * @brief Print memory statistics
 */
auto print_memory_stats(const ZeROStage1Optimizer::MemoryStats& stats, int rank) -> void {
    std::cout << "[Rank " << rank << "] Memory Statistics:" << std::endl;
    std::cout << "  Total parameters: " << stats.num_parameters << std::endl;
    std::cout << "  Local parameters: " << stats.num_local_parameters << std::endl;
    std::cout << "  GPU optimizer memory: " << format_bytes(stats.gpu_optimizer_memory) << std::endl;
    std::cout << "  CPU optimizer memory: " << format_bytes(stats.cpu_optimizer_memory) << std::endl;
    std::cout << "  GPU gradient memory: " << format_bytes(stats.gpu_gradient_memory) << std::endl;
}

// ============================================================================
// Training Function
// ============================================================================

auto train_epoch(
    BertForSequenceClassification& model,
    ZeROStage1Optimizer& optimizer,
    int num_batches,
    int batch_size,
    int seq_length,
    int vocab_size,
    Device device,
    int rank
) -> std::pair<double, double> {

    model.train();

    double total_loss = 0.0;
    auto epoch_start = std::chrono::steady_clock::now();

    for (int batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
        auto batch_start = std::chrono::steady_clock::now();

        // Generate batch
        auto [input_ids, attention_mask, labels] = generate_batch(
            batch_size, seq_length, vocab_size, device
        );

        // Forward pass
        auto logits = model.forward(input_ids, attention_mask, Variable{});

        // Compute loss (cross-entropy)
        Variable labels_var(labels, false);
        auto diff = logits - labels_var;
        auto squared = pow(diff.tensor(), 2.0f);
        auto loss = Variable(mean(squared), true);

        // Backward pass
        optimizer.zero_grad();
        loss.backward();

        // Optimizer step (includes gradient all-reduce)
        optimizer.step();

        auto batch_end = std::chrono::steady_clock::now();
        auto batch_time = std::chrono::duration<double, std::milli>(batch_end - batch_start).count();

        auto loss_cpu = loss.tensor().to(Device::cpu());
        float loss_value = const_cast<float*>(loss_cpu.data<float>())[0];
        total_loss += loss_value;

        if (rank == 0 && batch_idx % 10 == 0) {
            std::cout << "  Batch " << std::setw(3) << batch_idx << "/" << num_batches
                      << " | Loss: " << std::fixed << std::setprecision(4) << loss_value
                      << " | Time: " << std::setprecision(2) << batch_time << " ms"
                      << std::endl;
        }
    }

    auto epoch_end = std::chrono::steady_clock::now();
    auto epoch_time = std::chrono::duration<double>(epoch_end - epoch_start).count();

    return {total_loss / num_batches, epoch_time};
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
        std::cout << "=== BERT Training with ZeRO Stage 1 ===" << std::endl;
        std::cout << "World size: " << world_size << std::endl;
        std::cout << std::endl;
    }

    // Configuration
    Device device = Device::cuda(rank);
    int batch_size = 8;
    int seq_length = 128;
    int num_epochs = 3;
    int num_batches_per_epoch = 50;
    double learning_rate = 2e-5;

    // BERT configuration (small model for demonstration)
    BertConfig bert_config;
    bert_config.vocab_size = 30000;
    bert_config.hidden_size = 768;
    bert_config.num_hidden_layers = 6;
    bert_config.num_attention_heads = 12;
    bert_config.intermediate_size = 3072;
    bert_config.max_position_embeddings = 512;
    bert_config.hidden_dropout_prob = 0.1;
    bert_config.attention_probs_dropout_prob = 0.1;

    if (rank == 0) {
        std::cout << "Configuration:" << std::endl;
        std::cout << "  Batch size: " << batch_size << std::endl;
        std::cout << "  Sequence length: " << seq_length << std::endl;
        std::cout << "  Number of epochs: " << num_epochs << std::endl;
        std::cout << "  Batches per epoch: " << num_batches_per_epoch << std::endl;
        std::cout << "  Learning rate: " << learning_rate << std::endl;
        std::cout << "  Device: " << device.to_string() << std::endl;
        std::cout << std::endl;

        std::cout << "BERT Model:" << std::endl;
        std::cout << "  Vocabulary size: " << bert_config.vocab_size << std::endl;
        std::cout << "  Hidden size: " << bert_config.hidden_size << std::endl;
        std::cout << "  Layers: " << bert_config.num_hidden_layers << std::endl;
        std::cout << "  Attention heads: " << bert_config.num_attention_heads << std::endl;
        std::cout << std::endl;
    }

    // Create model
    auto model = BertForSequenceClassification(bert_config, 2);
    model.to(device);

    // Count parameters
    auto params = model.parameters();
    size_t total_params = 0;
    for (const auto& param : params) {
        total_params += param->tensor().numel();
    }

    if (rank == 0) {
        std::cout << "Model created with " << total_params << " parameters" << std::endl;
        std::cout << "Model size: ~" << format_bytes(total_params * 4) << std::endl;
        std::cout << std::endl;
    }

    // Create ZeRO Stage 1 optimizer
    auto base_optimizer = std::make_unique<Adam>(params, learning_rate);

    ZeROStage1Config zero_config;
    zero_config.world_size = world_size;
    zero_config.rank = rank;
    zero_config.offload_to_cpu = false;
    zero_config.overlap_comm = true;
    zero_config.pin_memory = true;
    zero_config.process_group = distributed::get_default_process_group();

    ZeROStage1Optimizer optimizer(std::move(base_optimizer), zero_config);

    if (rank == 0) {
        std::cout << "ZeRO Stage 1 Configuration:" << std::endl;
        std::cout << "  World size: " << optimizer.world_size() << std::endl;
        std::cout << "  Rank: " << optimizer.rank() << std::endl;
        std::cout << "  CPU offload: " << (optimizer.is_cpu_offload_enabled() ? "Enabled" : "Disabled") << std::endl;
        std::cout << "  Local parameters: " << optimizer.local_param_count() << std::endl;
        std::cout << std::endl;
    }

    // Print initial memory stats
    auto initial_stats = optimizer.get_memory_stats();
    if (rank == 0) {
        print_memory_stats(initial_stats, rank);
        std::cout << std::endl;
    }

    // Training loop
    if (rank == 0) {
        std::cout << "Starting training..." << std::endl;
        std::cout << std::string(80, '=') << std::endl;
    }

    for (int epoch = 1; epoch <= num_epochs; ++epoch) {
        if (rank == 0) {
            std::cout << "Epoch " << epoch << "/" << num_epochs << std::endl;
        }

        auto [avg_loss, epoch_time] = train_epoch(
            model, optimizer, num_batches_per_epoch,
            batch_size, seq_length, bert_config.vocab_size,
            device, rank
        );

        if (rank == 0) {
            std::cout << "Epoch " << epoch << " completed:" << std::endl;
            std::cout << "  Average loss: " << std::fixed << std::setprecision(4) << avg_loss << std::endl;
            std::cout << "  Epoch time: " << std::setprecision(2) << epoch_time << " seconds" << std::endl;
            std::cout << "  Throughput: " << std::setprecision(2)
                      << (num_batches_per_epoch * batch_size / epoch_time) << " samples/sec" << std::endl;

            // Print memory stats after epoch
            auto epoch_stats = optimizer.get_memory_stats();
            print_memory_stats(epoch_stats, rank);
            std::cout << std::string(80, '-') << std::endl;
        }
    }

    if (rank == 0) {
        std::cout << std::string(80, '=') << std::endl;
        std::cout << "Training completed!" << std::endl;
        std::cout << std::endl;
    }

    // Save checkpoint
    if (rank == 0) {
        std::cout << "Saving checkpoint..." << std::endl;
    }

    // II.20: PID-suffixed to avoid concurrent-run collisions.
    std::string checkpoint_path =
        tenzor::examples::example_tmp_path("bert_zero_stage1_checkpoint");
    optimizer.save_checkpoint(checkpoint_path);

    if (rank == 0) {
        std::cout << "Checkpoint saved to: " << checkpoint_path << "_rank_*.pt" << std::endl;
        std::cout << std::endl;
    }

    // Final memory stats
    auto final_stats = optimizer.get_memory_stats();
    if (rank == 0) {
        std::cout << "Final Memory Statistics:" << std::endl;
        print_memory_stats(final_stats, rank);
        std::cout << std::endl;

        // Calculate memory savings
        size_t baseline_optimizer_memory = total_params * 4 * 2;  // Adam: 2 states per param (fp32)
        size_t zero_stage1_memory = final_stats.gpu_optimizer_memory + final_stats.cpu_optimizer_memory;
        double reduction_factor = static_cast<double>(baseline_optimizer_memory) / zero_stage1_memory;

        std::cout << "Memory Savings Analysis:" << std::endl;
        std::cout << "  Baseline (full Adam states): " << format_bytes(baseline_optimizer_memory) << std::endl;
        std::cout << "  ZeRO Stage 1 (partitioned): " << format_bytes(zero_stage1_memory) << std::endl;
        std::cout << "  Reduction factor: " << std::fixed << std::setprecision(2) << reduction_factor << "x" << std::endl;
        std::cout << std::endl;
    }

    // Cleanup
    distributed::destroy_process_group();

    if (rank == 0) {
        std::cout << "Example completed successfully!" << std::endl;
    }

    return 0;
}
