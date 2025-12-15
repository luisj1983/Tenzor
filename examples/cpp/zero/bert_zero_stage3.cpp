/**
 * @file bert_zero_stage3.cpp
 * @brief BERT training with ZeRO Stage 3 (Full Parameter Partitioning)
 *
 * This example demonstrates:
 * 1. Training BERT with ZeRO Stage 3 for maximum memory efficiency
 * 2. Parameter partitioning with automatic gather/scatter
 * 3. Prefetch scheduling to hide communication latency
 * 4. Parameter caching across forward/backward passes
 * 5. Performance monitoring and optimization
 *
 * Memory Savings:
 * - Parameters partitioned (Nx reduction)
 * - Gradients partitioned (Nx reduction)
 * - Optimizer states partitioned (Nx reduction)
 * - Enables training models N times larger than GPU memory
 *
 * Build: g++ -std=c++17 -O3 bert_zero_stage3.cpp -ltenzor -o bert_zero_stage3
 * Run:   mpirun -np 8 ./bert_zero_stage3
 */

#include <tenzor/tenzor.hpp>
#include <tenzor/models/bert.hpp>
#include <tenzor/nn/optim/zero_optimizer.hpp>
#include <tenzor/nn/optim/adamw.hpp>
#include <tenzor/distributed/distributed.hpp>
#include <iostream>
#include <iomanip>
#include <chrono>

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
                input_ids_data.data(), input_ids_data.size() * sizeof(int64_t));
    Tensor input_ids = input_ids_cpu.to(device);

    Tensor attention_mask = ones({batch_size, seq_length}, DType::Float32, device);

    std::uniform_int_distribution<int64_t> label_dist(0, 1);
    std::vector<int64_t> labels_data(batch_size);
    for (auto& label : labels_data) {
        label = label_dist(gen);
    }

    Tensor labels_cpu({batch_size}, DType::Int64, Device::cpu());
    std::memcpy(const_cast<int64_t*>(labels_cpu.data<int64_t>()),
                labels_data.data(), labels_data.size() * sizeof(int64_t));
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
    ZeROStage3Optimizer& optimizer,
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
        // Generate batch
        auto [input_ids, attention_mask, labels] = generate_batch(
            batch_size, seq_length, vocab_size, device
        );

        // Forward pass (parameters automatically gathered)
        auto logits = model.forward(input_ids, attention_mask, Variable{});

        // Compute loss
        Variable labels_var(labels, false);
        auto diff = logits - labels_var;
        auto squared = pow(diff.tensor(), 2.0f);
        auto loss = Variable(mean(squared), true);

        // Backward pass (gradients automatically scattered)
        optimizer.zero_grad();
        loss.backward();

        // Optimizer step (operates on local partition only)
        optimizer.step();

        auto loss_cpu = loss.tensor().to(Device::cpu());
        float loss_value = const_cast<float*>(loss_cpu.data<float>())[0];
        total_loss += loss_value;

        if (rank == 0 && batch_idx % 10 == 0) {
            // Get prefetch statistics
            auto prefetch_stats = optimizer.get_prefetch_stats();

            std::cout << "  Batch " << std::setw(3) << batch_idx << "/" << num_batches
                      << " | Loss: " << std::fixed << std::setprecision(4) << loss_value
                      << " | Prefetch hit rate: " << std::setprecision(1)
                      << (prefetch_stats.hit_rate * 100) << "%"
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
        std::cout << "=== BERT Training with ZeRO Stage 3 ===" << std::endl;
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

    // BERT configuration (larger model for Stage 3)
    BertConfig bert_config;
    bert_config.vocab_size = 30000;
    bert_config.hidden_size = 1024;
    bert_config.num_hidden_layers = 12;
    bert_config.num_attention_heads = 16;
    bert_config.intermediate_size = 4096;
    bert_config.max_position_embeddings = 512;

    if (rank == 0) {
        std::cout << "Configuration:" << std::endl;
        std::cout << "  Batch size: " << batch_size << std::endl;
        std::cout << "  Sequence length: " << seq_length << std::endl;
        std::cout << "  Epochs: " << num_epochs << std::endl;
        std::cout << std::endl;

        std::cout << "BERT Model (Large):" << std::endl;
        std::cout << "  Hidden size: " << bert_config.hidden_size << std::endl;
        std::cout << "  Layers: " << bert_config.num_hidden_layers << std::endl;
        std::cout << "  Attention heads: " << bert_config.num_attention_heads << std::endl;
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
        std::cout << "Model: " << total_params << " parameters" << std::endl;
        std::cout << "Model size: " << format_bytes(total_params * 4) << std::endl;
        std::cout << "Per-rank size: " << format_bytes((total_params * 4) / world_size) << std::endl;
        std::cout << std::endl;
    }

    // Create ZeRO Stage 3 optimizer
    auto base_optimizer = std::make_unique<AdamW>(
        params, learning_rate, 0.9, 0.999, 1e-8, 0.01
    );

    Stage3Config zero_config;
    zero_config.world_size = world_size;
    zero_config.rank = rank;
    zero_config.offload_to_cpu = false;
    zero_config.overlap_comm = true;
    zero_config.pin_memory = true;

    // Stage 3 specific configuration
    zero_config.prefetch_bucket_size = 100 * 1024 * 1024;  // 100MB
    zero_config.prefetch_depth = 2;
    zero_config.overlap_comm_compute = true;
    zero_config.max_cached_params = 10;
    zero_config.cache_params_across_passes = true;
    zero_config.partition_threshold = 1024;
    zero_config.pin_first_layer = true;
    zero_config.pin_last_layer = true;

    zero_config.process_group = distributed::get_default_process_group();

    ZeROStage3Optimizer optimizer(std::move(base_optimizer), zero_config);

    if (rank == 0) {
        std::cout << "ZeRO Stage 3 Configuration:" << std::endl;
        std::cout << "  Prefetch bucket size: " << format_bytes(zero_config.prefetch_bucket_size) << std::endl;
        std::cout << "  Prefetch depth: " << zero_config.prefetch_depth << std::endl;
        std::cout << "  Max cached params: " << zero_config.max_cached_params << std::endl;
        std::cout << "  Cache across passes: " << (zero_config.cache_params_across_passes ? "Yes" : "No") << std::endl;
        std::cout << "  Overlap comm/compute: " << (zero_config.overlap_comm_compute ? "Yes" : "No") << std::endl;
        std::cout << std::endl;
    }

    // Register model for parameter partitioning
    optimizer.register_model(model);

    if (rank == 0) {
        std::cout << "Model registered for parameter partitioning" << std::endl;
        std::cout << "Forward/backward hooks installed" << std::endl;
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
            // Get performance statistics
            auto stats = optimizer.get_stats();

            std::cout << "Epoch " << epoch << " Summary:" << std::endl;
            std::cout << "  Average loss: " << std::fixed << std::setprecision(4) << avg_loss << std::endl;
            std::cout << "  Epoch time: " << std::setprecision(2) << epoch_time << " sec" << std::endl;
            std::cout << "  Throughput: " << (num_batches_per_epoch * batch_size / epoch_time)
                      << " samples/sec" << std::endl;

            std::cout << "Performance Metrics:" << std::endl;
            std::cout << "  Total all-gather calls: " << stats.total_all_gather_calls << std::endl;
            std::cout << "  Total all-gather bytes: " << format_bytes(stats.total_all_gather_bytes) << std::endl;
            std::cout << "  Avg all-gather time: " << stats.avg_all_gather_time_ms << " ms" << std::endl;
            std::cout << "  Prefetch hit rate: " << std::fixed << std::setprecision(1)
                      << (stats.prefetch_hit_rate * 100) << "%" << std::endl;
            std::cout << "  Overlap efficiency: " << (stats.overlap_efficiency * 100) << "%" << std::endl;

            std::cout << "Memory Statistics:" << std::endl;
            std::cout << "  Peak gathered memory: " << format_bytes(stats.peak_gathered_memory_bytes) << std::endl;
            std::cout << "  Current gathered memory: " << format_bytes(stats.current_gathered_memory_bytes) << std::endl;
            std::cout << "  Cached parameters: " << stats.num_cached_params << std::endl;

            std::cout << std::string(80, '-') << std::endl;
        }
    }

    if (rank == 0) {
        std::cout << std::string(80, '=') << std::endl;

        // Calculate memory savings
        size_t baseline_memory = total_params * 4 * 4;  // params + grads + 2 optimizer states
        size_t stage3_memory = (total_params * 4 * 4) / world_size;  // Everything partitioned
        double reduction = static_cast<double>(baseline_memory) / stage3_memory;

        std::cout << "Memory Savings (per rank vs baseline):" << std::endl;
        std::cout << "  Baseline: " << format_bytes(baseline_memory) << std::endl;
        std::cout << "  ZeRO Stage 3: " << format_bytes(stage3_memory) << std::endl;
        std::cout << "  Reduction: " << std::fixed << std::setprecision(2) << reduction << "x" << std::endl;
        std::cout << std::endl;

        std::cout << "Training completed successfully!" << std::endl;
    }

    // Reset statistics
    optimizer.reset_stats();

    distributed::destroy_process_group();
    return 0;
}
