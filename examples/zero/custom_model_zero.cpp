/**
 * @file custom_model_zero.cpp
 * @brief Custom model integration with ZeRO optimization
 *
 * This example demonstrates:
 * 1. Creating a custom model from scratch
 * 2. Integrating with ZeRO Stage 1, 2, and 3
 * 3. Comparing memory usage across different ZeRO stages
 * 4. Best practices for custom model architecture
 * 5. Debugging and profiling ZeRO performance
 *
 * Build: g++ -std=c++17 -O3 custom_model_zero.cpp -ltenzor -o custom_model_zero
 * Run:   mpirun -np 4 ./custom_model_zero
 */

#include <tenzor/tenzor.hpp>
#include <tenzor/nn/module.hpp>
#include <tenzor/nn/optim/zero_optimizer.hpp>
#include <tenzor/nn/optim/adam.hpp>
#include <tenzor/distributed/distributed.hpp>
#include <iostream>
#include <iomanip>

using namespace tenzor;
using namespace tenzor::nn;
using namespace tenzor::optim;

// ============================================================================
// Custom Model Definition
// ============================================================================

/**
 * @brief Custom Vision Transformer-like model
 */
class CustomVisionModel : public Module {
public:
    CustomVisionModel(int64_t input_dim, int64_t hidden_dim, int64_t num_layers, int64_t num_classes)
        : input_dim_(input_dim), hidden_dim_(hidden_dim), num_layers_(num_layers) {

        // Input projection
        register_parameter("input_proj_weight",
            Variable(randn({input_dim, hidden_dim}, DType::Float32, Device::cpu()), true));
        register_parameter("input_proj_bias",
            Variable(zeros({hidden_dim}, DType::Float32, Device::cpu()), true));

        // Transformer layers
        for (int64_t i = 0; i < num_layers; ++i) {
            std::string prefix = "layer_" + std::to_string(i) + "_";

            // Self-attention
            register_parameter(prefix + "attn_qkv_weight",
                Variable(randn({hidden_dim, hidden_dim * 3}, DType::Float32, Device::cpu()), true));
            register_parameter(prefix + "attn_qkv_bias",
                Variable(zeros({hidden_dim * 3}, DType::Float32, Device::cpu()), true));
            register_parameter(prefix + "attn_proj_weight",
                Variable(randn({hidden_dim, hidden_dim}, DType::Float32, Device::cpu()), true));
            register_parameter(prefix + "attn_proj_bias",
                Variable(zeros({hidden_dim}, DType::Float32, Device::cpu()), true));

            // Feed-forward network
            register_parameter(prefix + "ffn_fc1_weight",
                Variable(randn({hidden_dim, hidden_dim * 4}, DType::Float32, Device::cpu()), true));
            register_parameter(prefix + "ffn_fc1_bias",
                Variable(zeros({hidden_dim * 4}, DType::Float32, Device::cpu()), true));
            register_parameter(prefix + "ffn_fc2_weight",
                Variable(randn({hidden_dim * 4, hidden_dim}, DType::Float32, Device::cpu()), true));
            register_parameter(prefix + "ffn_fc2_bias",
                Variable(zeros({hidden_dim}, DType::Float32, Device::cpu()), true));

            // Layer norm
            register_parameter(prefix + "norm1_weight",
                Variable(ones({hidden_dim}, DType::Float32, Device::cpu()), true));
            register_parameter(prefix + "norm1_bias",
                Variable(zeros({hidden_dim}, DType::Float32, Device::cpu()), true));
            register_parameter(prefix + "norm2_weight",
                Variable(ones({hidden_dim}, DType::Float32, Device::cpu()), true));
            register_parameter(prefix + "norm2_bias",
                Variable(zeros({hidden_dim}, DType::Float32, Device::cpu()), true));
        }

        // Classification head
        register_parameter("head_weight",
            Variable(randn({hidden_dim, num_classes}, DType::Float32, Device::cpu()), true));
        register_parameter("head_bias",
            Variable(zeros({num_classes}, DType::Float32, Device::cpu()), true));
    }

    auto forward_impl(const Variable& input) -> Variable override {
        // Input projection
        auto x = matmul(input.tensor(), parameters_.at("input_proj_weight")->tensor())
               + parameters_.at("input_proj_bias")->tensor();
        auto out = Variable(x, input.requires_grad());

        // Transformer layers
        for (int64_t i = 0; i < num_layers_; ++i) {
            std::string prefix = "layer_" + std::to_string(i) + "_";

            // Self-attention (simplified)
            auto attn_in = out;
            auto qkv = matmul(out.tensor(), parameters_.at(prefix + "attn_qkv_weight")->tensor())
                     + parameters_.at(prefix + "attn_qkv_bias")->tensor();
            auto attn_out = matmul(qkv, parameters_.at(prefix + "attn_proj_weight")->tensor())
                          + parameters_.at(prefix + "attn_proj_bias")->tensor();

            // Residual + norm
            out = Variable(out.tensor() + attn_out, out.requires_grad());

            // Feed-forward
            auto ffn_in = out;
            auto ffn_hidden = matmul(out.tensor(), parameters_.at(prefix + "ffn_fc1_weight")->tensor())
                            + parameters_.at(prefix + "ffn_fc1_bias")->tensor();
            auto ffn_out = matmul(ffn_hidden, parameters_.at(prefix + "ffn_fc2_weight")->tensor())
                         + parameters_.at(prefix + "ffn_fc2_bias")->tensor();

            // Residual
            out = Variable(out.tensor() + ffn_out, out.requires_grad());
        }

        // Classification head
        auto logits = matmul(out.tensor(), parameters_.at("head_weight")->tensor())
                    + parameters_.at("head_bias")->tensor();

        return Variable(logits, out.requires_grad());
    }

private:
    int64_t input_dim_;
    int64_t hidden_dim_;
    int64_t num_layers_;
};

// ============================================================================
// Helper Functions
// ============================================================================

auto generate_batch(int batch_size, int input_dim, int num_classes, Device device)
    -> std::tuple<Variable, Tensor> {

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 1.0f);

    std::vector<float> input_data(batch_size * input_dim);
    for (auto& val : input_data) {
        val = dist(gen);
    }

    Tensor input_cpu({batch_size, input_dim}, DType::Float32, Device::cpu());
    std::memcpy(const_cast<float*>(input_cpu.data<float>()),
                input_data.data(), input_data.size() * sizeof(float));
    Tensor input = input_cpu.to(device);

    std::uniform_int_distribution<int64_t> label_dist(0, num_classes - 1);
    std::vector<int64_t> labels_data(batch_size);
    for (auto& label : labels_data) {
        label = label_dist(gen);
    }

    Tensor labels_cpu({batch_size}, DType::Int64, Device::cpu());
    std::memcpy(const_cast<int64_t*>(labels_cpu.data<int64_t>()),
                labels_data.data(), labels_data.size() * sizeof(int64_t));
    Tensor labels = labels_cpu.to(device);

    return {Variable(input, true), labels};
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

template<typename OptimizerType>
auto train_with_optimizer(
    CustomVisionModel& model,
    OptimizerType& optimizer,
    int num_steps,
    int batch_size,
    int input_dim,
    int num_classes,
    Device device,
    int rank,
    const std::string& optimizer_name
) -> void {

    model.train();
    double total_loss = 0.0;

    auto start = std::chrono::steady_clock::now();

    for (int step = 0; step < num_steps; ++step) {
        // Generate batch
        auto [input, labels] = generate_batch(batch_size, input_dim, num_classes, device);

        // Forward
        auto output = model.forward(input);

        // Loss
        Variable labels_var(labels, false);
        auto diff = output - labels_var;
        auto loss = Variable(mean(pow(diff.tensor(), 2.0f)), true);

        // Backward
        optimizer.zero_grad();
        loss.backward();
        optimizer.step();

        auto loss_cpu = loss.tensor().to(Device::cpu());
        total_loss += const_cast<float*>(loss_cpu.data<float>())[0];

        if (rank == 0 && step % 10 == 0) {
            std::cout << "  Step " << step << "/" << num_steps
                      << " | Loss: " << std::fixed << std::setprecision(4)
                      << const_cast<float*>(loss_cpu.data<float>())[0]
                      << std::endl;
        }
    }

    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration<double>(end - start).count();

    if (rank == 0) {
        std::cout << optimizer_name << " Results:" << std::endl;
        std::cout << "  Average loss: " << (total_loss / num_steps) << std::endl;
        std::cout << "  Training time: " << duration << " seconds" << std::endl;
        std::cout << "  Throughput: " << (num_steps * batch_size / duration) << " samples/sec" << std::endl;
    }
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
        std::cout << "=== Custom Model Training with ZeRO ===" << std::endl;
        std::cout << "World size: " << world_size << std::endl;
        std::cout << std::endl;
    }

    // Configuration
    Device device = Device::cuda(rank);
    int64_t input_dim = 1024;
    int64_t hidden_dim = 2048;
    int64_t num_layers = 8;
    int64_t num_classes = 1000;
    int batch_size = 16;
    int num_steps = 50;

    if (rank == 0) {
        std::cout << "Model Configuration:" << std::endl;
        std::cout << "  Input dimension: " << input_dim << std::endl;
        std::cout << "  Hidden dimension: " << hidden_dim << std::endl;
        std::cout << "  Number of layers: " << num_layers << std::endl;
        std::cout << "  Number of classes: " << num_classes << std::endl;
        std::cout << std::endl;
    }

    // Create model
    auto model = CustomVisionModel(input_dim, hidden_dim, num_layers, num_classes);
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

    // ========================================================================
    // Test 1: ZeRO Stage 1
    // ========================================================================

    if (rank == 0) {
        std::cout << std::string(80, '=') << std::endl;
        std::cout << "Test 1: ZeRO Stage 1 (Optimizer State Partitioning)" << std::endl;
        std::cout << std::string(80, '=') << std::endl;
    }

    {
        auto base_opt = std::make_unique<Adam>(model.parameters(), 1e-3);

        ZeROStage1Config config1;
        config1.world_size = world_size;
        config1.rank = rank;
        config1.process_group = distributed::get_default_process_group();

        ZeROStage1Optimizer optimizer(std::move(base_opt), config1);

        train_with_optimizer(model, optimizer, num_steps, batch_size,
                           input_dim, num_classes, device, rank, "ZeRO Stage 1");

        if (rank == 0) {
            auto stats = optimizer.get_memory_stats();
            std::cout << "  Optimizer memory: " << format_bytes(stats.gpu_optimizer_memory) << std::endl;
            std::cout << "  Local parameters: " << stats.num_local_parameters << std::endl;
            std::cout << std::endl;
        }
    }

    // ========================================================================
    // Test 2: ZeRO Stage 2
    // ========================================================================

    if (rank == 0) {
        std::cout << std::string(80, '=') << std::endl;
        std::cout << "Test 2: ZeRO Stage 2 (Gradient + State Partitioning)" << std::endl;
        std::cout << std::string(80, '=') << std::endl;
    }

    {
        auto base_opt = std::make_unique<Adam>(model.parameters(), 1e-3);

        ZeROStage2Config config2;
        config2.world_size = world_size;
        config2.rank = rank;
        config2.gradient_bucket_size = 25 * 1024 * 1024;
        config2.process_group = distributed::get_default_process_group();

        ZeROStage2Optimizer optimizer(std::move(base_opt), config2);
        optimizer.register_backward_hooks();

        train_with_optimizer(model, optimizer, num_steps, batch_size,
                           input_dim, num_classes, device, rank, "ZeRO Stage 2");

        if (rank == 0) {
            auto stats = optimizer.get_memory_stats();
            auto bucket_stats = optimizer.get_bucket_stats();
            std::cout << "  Optimizer memory: " << format_bytes(stats.gpu_optimizer_memory) << std::endl;
            std::cout << "  Gradient memory: " << format_bytes(stats.gpu_gradient_memory) << std::endl;
            std::cout << "  Number of buckets: " << bucket_stats.num_buckets << std::endl;
            std::cout << std::endl;
        }
    }

    // ========================================================================
    // Test 3: ZeRO Stage 3
    // ========================================================================

    if (rank == 0) {
        std::cout << std::string(80, '=') << std::endl;
        std::cout << "Test 3: ZeRO Stage 3 (Full Parameter Partitioning)" << std::endl;
        std::cout << std::string(80, '=') << std::endl;
    }

    {
        auto base_opt = std::make_unique<Adam>(model.parameters(), 1e-3);

        Stage3Config config3;
        config3.world_size = world_size;
        config3.rank = rank;
        config3.prefetch_depth = 2;
        config3.max_cached_params = 10;
        config3.process_group = distributed::get_default_process_group();

        ZeROStage3Optimizer optimizer(std::move(base_opt), config3);
        optimizer.register_model(model);

        train_with_optimizer(model, optimizer, num_steps, batch_size,
                           input_dim, num_classes, device, rank, "ZeRO Stage 3");

        if (rank == 0) {
            auto stats = optimizer.get_stats();
            std::cout << "  Peak gathered memory: " << format_bytes(stats.peak_gathered_memory_bytes) << std::endl;
            std::cout << "  Prefetch hit rate: " << std::fixed << std::setprecision(1)
                      << (stats.prefetch_hit_rate * 100) << "%" << std::endl;
            std::cout << "  Overlap efficiency: " << (stats.overlap_efficiency * 100) << "%" << std::endl;
            std::cout << std::endl;
        }
    }

    // ========================================================================
    // Summary
    // ========================================================================

    if (rank == 0) {
        std::cout << std::string(80, '=') << std::endl;
        std::cout << "Memory Comparison Summary" << std::endl;
        std::cout << std::string(80, '=') << std::endl;

        size_t baseline = total_params * 4 * 4;  // params + grads + 2 optimizer states
        std::cout << "Baseline (no ZeRO): " << format_bytes(baseline) << std::endl;
        std::cout << "ZeRO Stage 1 (~4x): " << format_bytes(baseline / 4) << std::endl;
        std::cout << "ZeRO Stage 2 (~8x): " << format_bytes(baseline / 8) << std::endl;
        std::cout << "ZeRO Stage 3 (~" << world_size << "x): "
                  << format_bytes(baseline / world_size) << std::endl;
        std::cout << std::endl;

        std::cout << "All tests completed successfully!" << std::endl;
    }

    distributed::destroy_process_group();
    return 0;
}
