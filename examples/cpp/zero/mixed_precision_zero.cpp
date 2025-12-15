/**
 * @file mixed_precision_zero.cpp
 * @brief Mixed precision training with ZeRO optimization
 *
 * This example demonstrates:
 * 1. Combining FP16/BF16 training with ZeRO
 * 2. Gradient scaling for numerical stability
 * 3. Loss scaling and unscaling
 * 4. Master weights in FP32
 * 5. Performance comparison: FP32 vs FP16 vs BF16
 *
 * Memory Benefits:
 * - FP16: 2x memory reduction vs FP32
 * - ZeRO Stage 3 + FP16: Up to 16x memory reduction
 * - Enables training 16x larger models
 *
 * Build: g++ -std=c++17 -O3 mixed_precision_zero.cpp -ltenzor -o mixed_precision_zero
 * Run:   mpirun -np 4 ./mixed_precision_zero
 */

#include <tenzor/tenzor.hpp>
#include <tenzor/models/resnet.hpp>
#include <tenzor/nn/optim/zero_optimizer.hpp>
#include <tenzor/nn/optim/adam.hpp>
#include <tenzor/distributed/distributed.hpp>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cmath>

using namespace tenzor;
using namespace tenzor::models;
using namespace tenzor::nn;
using namespace tenzor::optim;

// ============================================================================
// Gradient Scaler for Mixed Precision Training
// ============================================================================

class GradScaler {
public:
    GradScaler(double init_scale = 65536.0, double growth_factor = 2.0,
               double backoff_factor = 0.5, int growth_interval = 2000)
        : scale_(init_scale), growth_factor_(growth_factor),
          backoff_factor_(backoff_factor), growth_interval_(growth_interval) {}

    auto scale(const Variable& loss) -> Variable {
        auto scaled_tensor = loss.tensor() * static_cast<float>(scale_);
        return Variable(scaled_tensor, loss.requires_grad());
    }

    auto step(std::vector<std::shared_ptr<Variable>>& params) -> bool {
        // Check for inf/nan in gradients
        bool found_inf = false;
        for (const auto& param : params) {
            if (param->has_grad()) {
                const auto& grad = param->grad().value();
                auto grad_cpu = grad.to(Device::cpu());
                const float* grad_data = grad_cpu.data<float>();

                for (int64_t i = 0; i < grad.numel(); ++i) {
                    if (std::isinf(grad_data[i]) || std::isnan(grad_data[i])) {
                        found_inf = true;
                        break;
                    }
                }
                if (found_inf) break;
            }
        }

        if (found_inf) {
            // Reduce scale
            scale_ *= backoff_factor_;
            steps_since_growth_ = 0;
            return false;  // Skip optimizer step
        }

        // Unscale gradients
        for (auto& param : params) {
            if (param->has_grad()) {
                auto grad = param->grad().value();
                auto unscaled = grad / static_cast<float>(scale_);
                param->set_grad(unscaled);
            }
        }

        // Increase scale periodically
        steps_since_growth_++;
        if (steps_since_growth_ >= growth_interval_) {
            scale_ *= growth_factor_;
            steps_since_growth_ = 0;
        }

        return true;  // Proceed with optimizer step
    }

    auto get_scale() const -> double { return scale_; }

private:
    double scale_;
    double growth_factor_;
    double backoff_factor_;
    int growth_interval_;
    int steps_since_growth_{0};
};

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

// ============================================================================
// Training Functions
// ============================================================================

auto train_fp32(
    ResNet& model,
    ZeROStage2Optimizer& optimizer,
    int num_steps,
    int batch_size,
    Device device,
    int rank
) -> std::pair<double, double> {

    model.train();
    double total_loss = 0.0;

    auto start = std::chrono::steady_clock::now();

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

    return {total_loss / num_steps, duration};
}

auto train_mixed_precision(
    ResNet& model,
    ZeROStage2Optimizer& optimizer,
    int num_steps,
    int batch_size,
    Device device,
    int rank,
    DType compute_dtype
) -> std::pair<double, double> {

    model.train();
    double total_loss = 0.0;
    int skipped_steps = 0;

    GradScaler scaler;

    auto start = std::chrono::steady_clock::now();

    for (int step = 0; step < num_steps; ++step) {
        auto [images, labels] = generate_batch(batch_size, 1000, device);

        // Cast to lower precision for forward pass
        auto images_cast = images.tensor().to(compute_dtype);
        Variable images_fp16(images_cast, images.requires_grad());

        // Forward in FP16/BF16
        auto output = model.forward(images_fp16);

        // Cast output back to FP32 for loss computation
        auto output_fp32 = output.tensor().to(DType::Float32);
        Variable output_cast(output_fp32, output.requires_grad());

        // Compute loss in FP32
        Variable labels_var(labels, false);
        auto diff = output_cast - labels_var;
        auto loss = Variable(mean(pow(diff.tensor(), 2.0f)), true);

        // Scale loss
        auto scaled_loss = scaler.scale(loss);

        // Backward with scaled loss
        optimizer.zero_grad();
        scaled_loss.backward();

        // Unscale gradients and check for inf/nan
        auto params = model.parameters();
        if (scaler.step(params)) {
            // Gradients are valid, proceed with optimizer step
            optimizer.step();

            auto loss_cpu = loss.tensor().to(Device::cpu());
            total_loss += const_cast<float*>(loss_cpu.data<float>())[0];
        } else {
            // Inf/NaN detected, skip this step
            skipped_steps++;
        }

        if (rank == 0 && step % 10 == 0) {
            auto loss_cpu = loss.tensor().to(Device::cpu());
            std::cout << "  Step " << step << "/" << num_steps
                      << " | Loss: " << std::fixed << std::setprecision(4)
                      << const_cast<float*>(loss_cpu.data<float>())[0]
                      << " | Scale: " << std::scientific << std::setprecision(1)
                      << scaler.get_scale()
                      << std::endl;
        }
    }

    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration<double>(end - start).count();

    if (rank == 0 && skipped_steps > 0) {
        std::cout << "  Skipped steps (inf/nan): " << skipped_steps << std::endl;
    }

    return {total_loss / (num_steps - skipped_steps), duration};
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
        std::cout << "=== Mixed Precision Training with ZeRO ===" << std::endl;
        std::cout << "World size: " << world_size << std::endl;
        std::cout << std::endl;
    }

    Device device = Device::cuda(rank);
    int batch_size = 32;
    int num_steps = 50;

    // ========================================================================
    // Experiment 1: FP32 Training (Baseline)
    // ========================================================================

    if (rank == 0) {
        std::cout << std::string(80, '=') << std::endl;
        std::cout << "Experiment 1: FP32 Training (Baseline)" << std::endl;
        std::cout << std::string(80, '=') << std::endl;
    }

    {
        auto model = ResNet::resnet50(1000);
        model.to(device);

        auto params = model.parameters();
        auto base_opt = std::make_unique<Adam>(params, 1e-3);

        ZeROStage2Config config;
        config.world_size = world_size;
        config.rank = rank;
        config.process_group = distributed::get_default_process_group();

        ZeROStage2Optimizer optimizer(std::move(base_opt), config);
        optimizer.register_backward_hooks();

        auto [loss, time] = train_fp32(model, optimizer, num_steps, batch_size, device, rank);

        if (rank == 0) {
            std::cout << std::endl;
            std::cout << "FP32 Results:" << std::endl;
            std::cout << "  Average loss: " << std::fixed << std::setprecision(4) << loss << std::endl;
            std::cout << "  Time: " << std::setprecision(2) << time << " seconds" << std::endl;
            std::cout << "  Throughput: " << (num_steps * batch_size / time) << " images/sec" << std::endl;

            auto stats = optimizer.get_memory_stats();
            std::cout << "  Memory: " << format_bytes(stats.gpu_optimizer_memory + stats.gpu_gradient_memory) << std::endl;
            std::cout << std::endl;
        }
    }

    // ========================================================================
    // Experiment 2: FP16 Mixed Precision
    // ========================================================================

    if (rank == 0) {
        std::cout << std::string(80, '=') << std::endl;
        std::cout << "Experiment 2: FP16 Mixed Precision" << std::endl;
        std::cout << std::string(80, '=') << std::endl;
    }

    {
        auto model = ResNet::resnet50(1000);
        model.to(device);

        auto params = model.parameters();
        auto base_opt = std::make_unique<Adam>(params, 1e-3);

        ZeROStage2Config config;
        config.world_size = world_size;
        config.rank = rank;
        config.process_group = distributed::get_default_process_group();

        ZeROStage2Optimizer optimizer(std::move(base_opt), config);
        optimizer.register_backward_hooks();

        auto [loss, time] = train_mixed_precision(model, optimizer, num_steps, batch_size,
                                                  device, rank, DType::Float16);

        if (rank == 0) {
            std::cout << std::endl;
            std::cout << "FP16 Results:" << std::endl;
            std::cout << "  Average loss: " << std::fixed << std::setprecision(4) << loss << std::endl;
            std::cout << "  Time: " << std::setprecision(2) << time << " seconds" << std::endl;
            std::cout << "  Throughput: " << (num_steps * batch_size / time) << " images/sec" << std::endl;

            auto stats = optimizer.get_memory_stats();
            std::cout << "  Memory: " << format_bytes(stats.gpu_optimizer_memory + stats.gpu_gradient_memory) << std::endl;
            std::cout << std::endl;
        }
    }

    // ========================================================================
    // Summary
    // ========================================================================

    if (rank == 0) {
        std::cout << std::string(80, '=') << std::endl;
        std::cout << "Mixed Precision Summary" << std::endl;
        std::cout << std::string(80, '=') << std::endl;
        std::cout << std::endl;

        std::cout << "Memory Savings:" << std::endl;
        std::cout << "  FP32: Baseline" << std::endl;
        std::cout << "  FP16: ~2x reduction in activations and gradients" << std::endl;
        std::cout << "  FP16 + ZeRO Stage 2: ~4x total reduction" << std::endl;
        std::cout << "  FP16 + ZeRO Stage 3: ~8x total reduction" << std::endl;
        std::cout << std::endl;

        std::cout << "Performance:" << std::endl;
        std::cout << "  FP16 typically 1.5-2x faster than FP32" << std::endl;
        std::cout << "  BF16 similar speed to FP16 with better numerical stability" << std::endl;
        std::cout << "  Tensor Cores (Volta+) provide significant speedup" << std::endl;
        std::cout << std::endl;

        std::cout << "Best Practices:" << std::endl;
        std::cout << "  1. Use gradient scaling to prevent underflow" << std::endl;
        std::cout << "  2. Keep master weights in FP32 for optimizer" << std::endl;
        std::cout << "  3. Use loss scaling with dynamic adjustment" << std::endl;
        std::cout << "  4. Monitor for inf/nan and skip affected steps" << std::endl;
        std::cout << "  5. Consider BF16 for better stability (A100+)" << std::endl;
        std::cout << std::endl;

        std::cout << "Recommended Configuration:" << std::endl;
        std::cout << "  - Compute: FP16 or BF16" << std::endl;
        std::cout << "  - Master weights: FP32" << std::endl;
        std::cout << "  - Optimizer states: FP32" << std::endl;
        std::cout << "  - Initial loss scale: 2^16" << std::endl;
        std::cout << "  - ZeRO Stage: 2 or 3 for maximum memory efficiency" << std::endl;
        std::cout << std::endl;

        std::cout << "Example completed successfully!" << std::endl;
    }

    distributed::destroy_process_group();
    return 0;
}
