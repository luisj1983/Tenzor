/**
 * @file cpu_offload_example.cpp
 * @brief CPU offload configuration for training models larger than GPU memory
 *
 * This example demonstrates:
 * 1. Enabling CPU offload for optimizer states and gradients
 * 2. Training models that exceed GPU memory capacity
 * 3. Configuring offload thresholds and policies
 * 4. Monitoring CPU/GPU memory usage
 * 5. Performance trade-offs of CPU offload
 *
 * Use Cases:
 * - Training very large models (>100B parameters)
 * - Limited GPU memory scenarios
 * - Cost-effective training on cheaper GPUs
 *
 * Build: g++ -std=c++17 -O3 cpu_offload_example.cpp -ltenzor -o cpu_offload_example
 * Run:   ./cpu_offload_example
 */

#include <tenzor/tenzor.hpp>
#include <tenzor/models/gpt2.hpp>
#include <tenzor/nn/optim/zero_optimizer.hpp>
#include <tenzor/nn/optim/adam.hpp>
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
    -> std::tuple<Variable, Tensor> {

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

    std::vector<int64_t> labels_data(input_ids_data.begin() + 1, input_ids_data.end());
    labels_data.push_back(-100);

    Tensor labels_cpu({batch_size, seq_length}, DType::Int64, Device::cpu());
    std::memcpy(const_cast<int64_t*>(labels_cpu.data<int64_t>()),
                labels_data.data(), labels_data.size() * sizeof(int64_t));
    Tensor labels = labels_cpu.to(device);

    return {Variable(input_ids, true), labels};
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

auto print_memory_breakdown(const ZeROStage1Optimizer::MemoryStats& stats) -> void {
    std::cout << "Memory Breakdown:" << std::endl;
    std::cout << "  GPU Optimizer States: " << format_bytes(stats.gpu_optimizer_memory) << std::endl;
    std::cout << "  CPU Optimizer States: " << format_bytes(stats.cpu_optimizer_memory) << std::endl;
    std::cout << "  GPU Gradients: " << format_bytes(stats.gpu_gradient_memory) << std::endl;
    std::cout << "  Total GPU: " << format_bytes(stats.gpu_optimizer_memory + stats.gpu_gradient_memory) << std::endl;
    std::cout << "  Total CPU: " << format_bytes(stats.cpu_optimizer_memory) << std::endl;
}

// ============================================================================
// Training Function
// ============================================================================

template<typename OptimizerType>
auto benchmark_training(
    GPT2LMHeadModel& model,
    OptimizerType& optimizer,
    int num_steps,
    int batch_size,
    int seq_length,
    Device device,
    const std::string& config_name
) -> double {

    model.train();

    double total_loss = 0.0;
    double total_forward_time = 0.0;
    double total_backward_time = 0.0;
    double total_optim_time = 0.0;

    auto start = std::chrono::steady_clock::now();

    for (int step = 0; step < num_steps; ++step) {
        auto [input, labels] = generate_batch(batch_size, seq_length, model.config().vocab_size, device);

        // Forward
        auto fwd_start = std::chrono::steady_clock::now();
        auto output = model.forward(input);
        auto fwd_end = std::chrono::steady_clock::now();
        total_forward_time += std::chrono::duration<double, std::milli>(fwd_end - fwd_start).count();

        // Loss
        Variable labels_var(labels, false);
        auto diff = output - labels_var;
        auto loss = Variable(mean(pow(diff.tensor(), 2.0f)), true);

        // Backward
        optimizer.zero_grad();
        auto bwd_start = std::chrono::steady_clock::now();
        loss.backward();
        auto bwd_end = std::chrono::steady_clock::now();
        total_backward_time += std::chrono::duration<double, std::milli>(bwd_end - bwd_start).count();

        // Optimizer step
        auto opt_start = std::chrono::steady_clock::now();
        optimizer.step();
        auto opt_end = std::chrono::steady_clock::now();
        total_optim_time += std::chrono::duration<double, std::milli>(opt_end - opt_start).count();

        auto loss_cpu = loss.tensor().to(Device::cpu());
        total_loss += const_cast<float*>(loss_cpu.data<float>())[0];

        if (step % 10 == 0) {
            std::cout << "  Step " << step << "/" << num_steps
                      << " | Loss: " << std::fixed << std::setprecision(4)
                      << const_cast<float*>(loss_cpu.data<float>())[0]
                      << std::endl;
        }
    }

    auto end = std::chrono::steady_clock::now();
    auto total_time = std::chrono::duration<double>(end - start).count();

    std::cout << std::endl;
    std::cout << config_name << " Results:" << std::endl;
    std::cout << "  Total time: " << std::fixed << std::setprecision(2) << total_time << " sec" << std::endl;
    std::cout << "  Avg forward time: " << (total_forward_time / num_steps) << " ms" << std::endl;
    std::cout << "  Avg backward time: " << (total_backward_time / num_steps) << " ms" << std::endl;
    std::cout << "  Avg optimizer time: " << (total_optim_time / num_steps) << " ms" << std::endl;
    std::cout << "  Average loss: " << std::setprecision(4) << (total_loss / num_steps) << std::endl;
    std::cout << "  Throughput: " << (num_steps * batch_size / total_time) << " samples/sec" << std::endl;

    return total_time;
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    // Initialize (single GPU)
    tenzor::initialize();

    std::cout << "=== CPU Offload Configuration Example ===" << std::endl;
    std::cout << std::endl;

    Device device = Device::cuda(0);

    // Configuration
    int batch_size = 4;
    int seq_length = 512;
    int num_steps = 50;
    double learning_rate = 1e-4;

    // GPT-2 Medium (large enough to benefit from offload)
    GPT2Config gpt_config = GPT2Config::gpt2_medium();
    gpt_config.n_positions = seq_length;

    std::cout << "Model: GPT-2 Medium" << std::endl;
    std::cout << "  Hidden size: " << gpt_config.n_embd << std::endl;
    std::cout << "  Layers: " << gpt_config.n_layer << std::endl;
    std::cout << "  Context: " << gpt_config.n_positions << std::endl;
    std::cout << std::endl;

    // Create model
    auto model = GPT2LMHeadModel(gpt_config);
    model.to(device);

    auto params = model.parameters();
    size_t total_params = 0;
    for (const auto& param : params) {
        total_params += param->tensor().numel();
    }

    std::cout << "Parameters: " << total_params << " (~" << format_bytes(total_params * 4) << ")" << std::endl;
    std::cout << std::endl;

    // ========================================================================
    // Experiment 1: No Offload (Baseline)
    // ========================================================================

    std::cout << std::string(80, '=') << std::endl;
    std::cout << "Experiment 1: No CPU Offload (Baseline)" << std::endl;
    std::cout << std::string(80, '=') << std::endl;

    {
        auto base_opt = std::make_unique<Adam>(model.parameters(), learning_rate);

        ZeROStage1Config config;
        config.world_size = 1;
        config.rank = 0;
        config.offload_to_cpu = false;

        ZeROStage1Optimizer optimizer(std::move(base_opt), config);

        auto time1 = benchmark_training(model, optimizer, num_steps, batch_size, seq_length, device,
                                       "No Offload");

        auto stats = optimizer.get_memory_stats();
        print_memory_breakdown(stats);
        std::cout << std::endl;
    }

    // ========================================================================
    // Experiment 2: CPU Offload with Default Threshold
    // ========================================================================

    std::cout << std::string(80, '=') << std::endl;
    std::cout << "Experiment 2: CPU Offload (Default Threshold)" << std::endl;
    std::cout << std::string(80, '=') << std::endl;

    {
        auto base_opt = std::make_unique<Adam>(model.parameters(), learning_rate);

        ZeROStage1Config config;
        config.world_size = 1;
        config.rank = 0;
        config.offload_to_cpu = true;
        config.cpu_offload_threshold = 1024;  // 1KB
        config.pin_memory = true;

        ZeROStage1Optimizer optimizer(std::move(base_opt), config);

        std::cout << "Offload Configuration:" << std::endl;
        std::cout << "  Enabled: Yes" << std::endl;
        std::cout << "  Threshold: " << format_bytes(config.cpu_offload_threshold) << std::endl;
        std::cout << "  Pinned memory: Yes" << std::endl;
        std::cout << std::endl;

        auto time2 = benchmark_training(model, optimizer, num_steps, batch_size, seq_length, device,
                                       "CPU Offload (1KB)");

        auto stats = optimizer.get_memory_stats();
        print_memory_breakdown(stats);
        std::cout << std::endl;
    }

    // ========================================================================
    // Experiment 3: CPU Offload with High Threshold
    // ========================================================================

    std::cout << std::string(80, '=') << std::endl;
    std::cout << "Experiment 3: CPU Offload (High Threshold)" << std::endl;
    std::cout << std::string(80, '=') << std::endl;

    {
        auto base_opt = std::make_unique<Adam>(model.parameters(), learning_rate);

        ZeROStage1Config config;
        config.world_size = 1;
        config.rank = 0;
        config.offload_to_cpu = true;
        config.cpu_offload_threshold = 1024 * 1024;  // 1MB
        config.pin_memory = true;

        ZeROStage1Optimizer optimizer(std::move(base_opt), config);

        std::cout << "Offload Configuration:" << std::endl;
        std::cout << "  Enabled: Yes" << std::endl;
        std::cout << "  Threshold: " << format_bytes(config.cpu_offload_threshold) << std::endl;
        std::cout << "  Pinned memory: Yes" << std::endl;
        std::cout << std::endl;

        auto time3 = benchmark_training(model, optimizer, num_steps, batch_size, seq_length, device,
                                       "CPU Offload (1MB)");

        auto stats = optimizer.get_memory_stats();
        print_memory_breakdown(stats);
        std::cout << std::endl;
    }

    // ========================================================================
    // Experiment 4: CPU Offload without Pinned Memory
    // ========================================================================

    std::cout << std::string(80, '=') << std::endl;
    std::cout << "Experiment 4: CPU Offload (No Pinned Memory)" << std::endl;
    std::cout << std::string(80, '=') << std::endl;

    {
        auto base_opt = std::make_unique<Adam>(model.parameters(), learning_rate);

        ZeROStage1Config config;
        config.world_size = 1;
        config.rank = 0;
        config.offload_to_cpu = true;
        config.cpu_offload_threshold = 1024;
        config.pin_memory = false;  // Disable pinned memory

        ZeROStage1Optimizer optimizer(std::move(base_opt), config);

        std::cout << "Offload Configuration:" << std::endl;
        std::cout << "  Enabled: Yes" << std::endl;
        std::cout << "  Threshold: " << format_bytes(config.cpu_offload_threshold) << std::endl;
        std::cout << "  Pinned memory: No" << std::endl;
        std::cout << std::endl;

        auto time4 = benchmark_training(model, optimizer, num_steps, batch_size, seq_length, device,
                                       "CPU Offload (No Pin)");

        auto stats = optimizer.get_memory_stats();
        print_memory_breakdown(stats);
        std::cout << std::endl;
    }

    // ========================================================================
    // Summary
    // ========================================================================

    std::cout << std::string(80, '=') << std::endl;
    std::cout << "Summary" << std::endl;
    std::cout << std::string(80, '=') << std::endl;
    std::cout << std::endl;

    std::cout << "CPU Offload Trade-offs:" << std::endl;
    std::cout << "  Pros:" << std::endl;
    std::cout << "    - Reduces GPU memory usage" << std::endl;
    std::cout << "    - Enables training larger models" << std::endl;
    std::cout << "    - Cost-effective for memory-bound workloads" << std::endl;
    std::cout << std::endl;
    std::cout << "  Cons:" << std::endl;
    std::cout << "    - Adds CPU-GPU transfer overhead" << std::endl;
    std::cout << "    - Slightly slower training (5-15%)" << std::endl;
    std::cout << "    - Requires sufficient CPU RAM" << std::endl;
    std::cout << std::endl;

    std::cout << "Best Practices:" << std::endl;
    std::cout << "  1. Use pinned memory for faster transfers" << std::endl;
    std::cout << "  2. Set threshold to offload only large tensors" << std::endl;
    std::cout << "  3. Combine with ZeRO Stage 2/3 for maximum efficiency" << std::endl;
    std::cout << "  4. Monitor CPU RAM usage to avoid swapping" << std::endl;
    std::cout << std::endl;

    std::cout << "Example completed successfully!" << std::endl;

    return 0;
}
