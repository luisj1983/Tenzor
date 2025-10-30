/**
 * @file bert_zero_stage2.cpp
 * @brief BERT training with ZeRO Stage 2 (Gradient + Optimizer State Partitioning)
 *
 * This example demonstrates:
 * 1. Training BERT with ZeRO Stage 2 for 8x memory reduction
 * 2. Gradient reduce-scatter during backward pass
 * 3. Gradient bucketing for efficient communication
 * 4. Backward hook registration for automatic gradient partitioning
 * 5. Performance monitoring and throughput measurement
 *
 * Memory Savings:
 * - Optimizer states partitioned (4x reduction for Adam)
 * - Gradients partitioned (4x reduction)
 * - Total: 8x reduction
 *
 * Build: g++ -std=c++17 -O3 bert_zero_stage2.cpp -ltenzor -o bert_zero_stage2
 * Run:   mpirun -np 4 ./bert_zero_stage2
 */

#include <tenzor/tenzor.hpp>
#include <tenzor/models/bert.hpp>
#include <tenzor/nn/optim/zero_optimizer.hpp>
#include <tenzor/nn/optim/adam.hpp>
#include <tenzor/distributed/distributed.hpp>
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

auto generate_batch(int batch_size, int seq_length, int vocab_size, Device device)
    -> std::tuple<Variable, Tensor, Tensor> {

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

    Tensor attention_mask = ones({batch_size, seq_length}, DType::Float32, device);

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

// ============================================================================
// Training Function
// ============================================================================

auto train_epoch(
    BertForSequenceClassification& model,
    ZeROStage2Optimizer& optimizer,
    int num_batches,
    int batch_size,
    int seq_length,
    int vocab_size,
    Device device,
    int rank
) -> std::pair<double, double> {

    model.train();

    double total_loss = 0.0;
    double total_forward_time = 0.0;
    double total_backward_time = 0.0;
    double total_optim_time = 0.0;

    auto epoch_start = std::chrono::steady_clock::now();

    for (int batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
        // Generate batch
        auto [input_ids, attention_mask, labels] = generate_batch(
            batch_size, seq_length, vocab_size, device
        );

        // Forward pass
        auto forward_start = std::chrono::steady_clock::now();
        auto logits = model.forward(input_ids, attention_mask, Variable{});
        auto forward_end = std::chrono::steady_clock::now();
        total_forward_time += std::chrono::duration<double, std::milli>(forward_end - forward_start).count();

        // Compute loss
        Variable labels_var(labels, false);
        auto diff = logits - labels_var;
        auto squared = pow(diff.tensor(), 2.0f);
        auto loss = Variable(mean(squared), true);

        // Backward pass (gradients automatically reduce-scattered via hooks)
        optimizer.zero_grad();
        auto backward_start = std::chrono::steady_clock::now();
        loss.backward();
        auto backward_end = std::chrono::steady_clock::now();
        total_backward_time += std::chrono::duration<double, std::milli>(backward_end - backward_start).count();

        // Optimizer step (no gradient all-reduce needed!)
        auto optim_start = std::chrono::steady_clock::now();
        optimizer.step();
        auto optim_end = std::chrono::steady_clock::now();
        total_optim_time += std::chrono::duration<double, std::milli>(optim_end - optim_start).count();

        auto loss_cpu = loss.tensor().to(Device::cpu());
        float loss_value = const_cast<float*>(loss_cpu.data<float>())[0];
        total_loss += loss_value;

        if (rank == 0 && batch_idx % 10 == 0) {
            std::cout << "  Batch " << std::setw(3) << batch_idx << "/" << num_batches
                      << " | Loss: " << std::fixed << std::setprecision(4) << loss_value
                      << std::endl;
        }
    }

    auto epoch_end = std::chrono::steady_clock::now();
    auto epoch_time = std::chrono::duration<double>(epoch_end - epoch_start).count();

    if (rank == 0) {
        std::cout << "  Timing breakdown (per batch avg):" << std::endl;
        std::cout << "    Forward: " << std::fixed << std::setprecision(2)
                  << (total_forward_time / num_batches) << " ms" << std::endl;
        std::cout << "    Backward: " << (total_backward_time / num_batches) << " ms" << std::endl;
        std::cout << "    Optimizer: " << (total_optim_time / num_batches) << " ms" << std::endl;
    }

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
        std::cout << "=== BERT Training with ZeRO Stage 2 ===" << std::endl;
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

    // BERT configuration
    BertConfig bert_config;
    bert_config.vocab_size = 30000;
    bert_config.hidden_size = 768;
    bert_config.num_hidden_layers = 6;
    bert_config.num_attention_heads = 12;
    bert_config.intermediate_size = 3072;
    bert_config.max_position_embeddings = 512;

    if (rank == 0) {
        std::cout << "Configuration:" << std::endl;
        std::cout << "  Batch size: " << batch_size << std::endl;
        std::cout << "  Sequence length: " << seq_length << std::endl;
        std::cout << "  Epochs: " << num_epochs << std::endl;
        std::cout << "  Device: " << device.to_string() << std::endl;
        std::cout << std::endl;
    }

    // Create model
    auto model = BertForSequenceClassification(bert_config, 2);
    model.to(device);

    auto params = model.parameters();
    size_t total_params = 0;
    for (const auto& param : params) {
        total_params += param->tensor().numel();
    }

    if (rank == 0) {
        std::cout << "Model: " << total_params << " parameters (~"
                  << format_bytes(total_params * 4) << ")" << std::endl;
        std::cout << std::endl;
    }

    // Create ZeRO Stage 2 optimizer
    auto base_optimizer = std::make_unique<Adam>(params, learning_rate);

    ZeROStage2Config zero_config;
    zero_config.world_size = world_size;
    zero_config.rank = rank;
    zero_config.offload_to_cpu = false;
    zero_config.overlap_comm = true;
    zero_config.pin_memory = true;
    zero_config.gradient_bucket_size = 25 * 1024 * 1024;  // 25MB buckets
    zero_config.reduce_scatter_in_backward = true;
    zero_config.gradient_bucketing = true;
    zero_config.process_group = distributed::get_default_process_group();

    ZeROStage2Optimizer optimizer(std::move(base_optimizer), zero_config);

    if (rank == 0) {
        std::cout << "ZeRO Stage 2 Configuration:" << std::endl;
        std::cout << "  Gradient bucket size: " << format_bytes(zero_config.gradient_bucket_size) << std::endl;
        std::cout << "  Reduce-scatter in backward: Enabled" << std::endl;
        std::cout << "  Gradient bucketing: Enabled" << std::endl;
        std::cout << std::endl;
    }

    // Register backward hooks for gradient reduce-scatter
    optimizer.register_backward_hooks();

    if (rank == 0) {
        std::cout << "Backward hooks registered for automatic gradient partitioning" << std::endl;

        // Print bucket statistics
        auto bucket_stats = optimizer.get_bucket_stats();
        std::cout << "Gradient Bucket Statistics:" << std::endl;
        std::cout << "  Number of buckets: " << bucket_stats.num_buckets << std::endl;
        std::cout << "  Average bucket size: " << format_bytes(bucket_stats.avg_bucket_size) << std::endl;
        std::cout << "  Max bucket size: " << format_bytes(bucket_stats.max_bucket_size) << std::endl;
        std::cout << "  Total gradient memory: " << format_bytes(bucket_stats.total_gradient_memory) << std::endl;
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
            std::cout << "Epoch " << epoch << " Summary:" << std::endl;
            std::cout << "  Average loss: " << std::fixed << std::setprecision(4) << avg_loss << std::endl;
            std::cout << "  Epoch time: " << std::setprecision(2) << epoch_time << " sec" << std::endl;
            std::cout << "  Throughput: " << (num_batches_per_epoch * batch_size / epoch_time)
                      << " samples/sec" << std::endl;

            // Memory stats
            auto stats = optimizer.get_memory_stats();
            std::cout << "  GPU optimizer memory: " << format_bytes(stats.gpu_optimizer_memory) << std::endl;
            std::cout << "  GPU gradient memory: " << format_bytes(stats.gpu_gradient_memory) << std::endl;
            std::cout << std::string(80, '-') << std::endl;
        }
    }

    if (rank == 0) {
        std::cout << std::string(80, '=') << std::endl;

        // Calculate memory savings
        size_t baseline_memory = total_params * 4 * 4;  // params + grads + 2 optimizer states
        auto final_stats = optimizer.get_memory_stats();
        size_t zero_stage2_memory = final_stats.gpu_optimizer_memory + final_stats.gpu_gradient_memory;
        double reduction = static_cast<double>(baseline_memory) / zero_stage2_memory;

        std::cout << "Memory Savings (vs baseline):" << std::endl;
        std::cout << "  Baseline: " << format_bytes(baseline_memory) << std::endl;
        std::cout << "  ZeRO Stage 2: " << format_bytes(zero_stage2_memory) << std::endl;
        std::cout << "  Reduction: " << std::fixed << std::setprecision(2) << reduction << "x" << std::endl;
        std::cout << std::endl;

        std::cout << "Training completed successfully!" << std::endl;
    }

    distributed::destroy_process_group();
    return 0;
}
