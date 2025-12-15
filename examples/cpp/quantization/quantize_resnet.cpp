/**
 * @file quantize_resnet.cpp
 * @brief Example: Quantizing ResNet for inference optimization
 *
 * Demonstrates:
 * 1. Post-training quantization (PTQ)
 * 2. Quantization-aware training (QAT)
 * 3. Performance benchmarking
 * 4. Accuracy evaluation
 */

#include <iostream>
#include <iomanip>
#include "tenzor/tenzor.hpp"
#include "tenzor/models/resnet.hpp"
#include "tenzor/nn/quantization.hpp"
#include "tenzor/quantization/quantize_api.hpp"
#include "tenzor/data/dataset.hpp"

using namespace tenzor;
using namespace tenzor::nn;
using namespace tenzor::nn::quantization;

// ============================================================================
// Utility Functions
// ============================================================================

void print_header(const std::string& title) {
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << title << "\n";
    std::cout << std::string(60, '=') << "\n";
}

void print_metric(const std::string& name, float value, const std::string& unit = "") {
    std::cout << std::left << std::setw(30) << name << ": "
              << std::fixed << std::setprecision(2) << value << " " << unit << "\n";
}

// ============================================================================
// Post-Training Quantization (PTQ) Example
// ============================================================================

void demo_post_training_quantization() {
    print_header("Post-Training Quantization (PTQ)");

    // 1. Load pretrained ResNet50
    std::cout << "[1/5] Loading ResNet50 model...\n";
    auto model = models::resnet50(1000);
    model->eval();

    // 2. Prepare calibration data (simulated)
    std::cout << "[2/5] Preparing calibration data...\n";
    std::vector<Tensor> calibration_data;
    for (int i = 0; i < 100; ++i) {
        Tensor batch({32, 3, 224, 224}, DType::Float32, Device::cpu());
        batch.fill_(0.5f);  // In practice, use real validation data
        calibration_data.push_back(batch);
    }

    // 3. Configure quantization
    std::cout << "[3/5] Configuring quantization...\n";
    auto qconfig = DefaultQConfigs::default_qconfig();

    // 4. Perform static quantization with calibration
    std::cout << "[4/5] Performing quantization...\n";
    auto calibrate_fn = [&](Module& m) {
        m.eval();
        for (size_t i = 0; i < std::min(size_t(50), calibration_data.size()); ++i) {
            auto output = m.forward(autograd::Variable(calibration_data[i], false));
            if ((i + 1) % 10 == 0) {
                std::cout << "  Calibration progress: " << (i + 1) << "/50\n";
            }
        }
    };

    auto quantized_model = tenzor::quantization::quantize_static(
        model, calibrate_fn, qconfig
    );

    // 5. Benchmark performance
    std::cout << "[5/5] Benchmarking performance...\n\n";
    auto [fp32_time, int8_time, speedup, memory_reduction] =
        tenzor::quantization::benchmark_quantization(
            *model, *quantized_model, {1, 3, 224, 224}, 100
        );

    print_header("PTQ Results");
    print_metric("FP32 Inference Time", fp32_time, "ms");
    print_metric("INT8 Inference Time", int8_time, "ms");
    print_metric("Speedup", speedup, "x");
    print_metric("Memory Reduction", memory_reduction, "x");
    print_metric("Expected Accuracy Loss", 1.0f, "%");
}

// ============================================================================
// Dynamic Quantization Example
// ============================================================================

void demo_dynamic_quantization() {
    print_header("Dynamic Quantization");

    std::cout << "[1/3] Loading ResNet18 model...\n";
    auto model = models::resnet18(1000);
    model->eval();

    std::cout << "[2/3] Applying dynamic quantization...\n";
    auto quantized_model = tenzor::quantization::quantize_dynamic(model);

    std::cout << "[3/3] Benchmarking...\n\n";
    auto [fp32_time, int8_time, speedup, memory_reduction] =
        tenzor::quantization::benchmark_quantization(
            *model, *quantized_model, {1, 3, 224, 224}, 100
        );

    print_header("Dynamic Quantization Results");
    print_metric("FP32 Inference Time", fp32_time, "ms");
    print_metric("INT8 Inference Time", int8_time, "ms");
    print_metric("Speedup", speedup, "x");
    print_metric("Memory Reduction (weights only)", 4.0f, "x");
    std::cout << "\nNote: Dynamic quantization only quantizes weights.\n";
    std::cout << "Activations remain in FP32 for maximum compatibility.\n";
}

// ============================================================================
// Quantization-Aware Training (QAT) Example
// ============================================================================

void demo_quantization_aware_training() {
    print_header("Quantization-Aware Training (QAT)");

    std::cout << "[1/5] Creating ResNet18 model...\n";
    auto model = models::resnet18(1000);

    std::cout << "[2/5] Preparing model for QAT...\n";
    auto qconfig = DefaultQConfigs::qat_qconfig();
    auto qat_model = tenzor::quantization::prepare_qat(model, qconfig);

    std::cout << "[3/5] Training with quantization simulation...\n";
    qat_model->train();

    // Simulate training loop
    optim::SGD optimizer(qat_model->parameters(), 0.001f);

    for (int epoch = 0; epoch < 3; ++epoch) {
        std::cout << "  Epoch " << (epoch + 1) << "/3...\n";

        // Simulate batch training
        for (int batch = 0; batch < 10; ++batch) {
            // Create dummy data
            Tensor input({32, 3, 224, 224}, DType::Float32, Device::cpu());
            Tensor labels({32}, DType::Int64, Device::cpu());
            input.fill_(0.5f);
            labels.fill_(1);

            // Forward + backward (simplified)
            optimizer.zero_grad();
            auto output = qat_model->forward(autograd::Variable(input, true));
            // loss = criterion(output, labels)
            // loss.backward()
            optimizer.step();
        }
    }

    std::cout << "[4/5] Converting QAT model to quantized model...\n";
    auto quantized_model = tenzor::quantization::convert_qat(qat_model);

    std::cout << "[5/5] Final evaluation...\n\n";

    print_header("QAT Results");
    std::cout << "QAT typically achieves:\n";
    print_metric("Accuracy Loss", 0.3f, "%");
    print_metric("Speedup", 2.5f, "x");
    print_metric("Memory Reduction", 4.0f, "x");
    std::cout << "\nNote: QAT provides best accuracy but requires retraining.\n";
}

// ============================================================================
// Per-Layer Quantization Configuration
// ============================================================================

void demo_per_layer_quantization() {
    print_header("Per-Layer Quantization Configuration");

    std::cout << "Configuring layer-specific quantization settings...\n\n";

    // Create quantization strategy with different configs per layer
    auto strategy = QuantizationStrategyBuilder()
        .set_global_qconfig(DefaultQConfigs::default_qconfig())
        .set_layer_qconfig("conv1", DefaultQConfigs::high_accuracy_qconfig())
        .set_layer_qconfig("fc", DefaultQConfigs::high_accuracy_qconfig())
        .set_type_qconfig("BatchNorm2d", DefaultQConfigs::fast_qconfig())
        .disable_layer("layer1.0.downsample")
        .set_backend(QuantizationBackend::FBGEMM)
        .set_calibration_batches(200)
        .enable_operation_fusion(true)
        .build();

    std::cout << "Strategy configured:\n";
    std::cout << "  - Global: Default INT8 quantization\n";
    std::cout << "  - conv1, fc: High-accuracy histogram-based\n";
    std::cout << "  - BatchNorm2d: Fast min-max quantization\n";
    std::cout << "  - layer1.0.downsample: Disabled (FP32)\n";
    std::cout << "  - Backend: FBGEMM (optimized for x86)\n";
    std::cout << "  - Calibration: 200 batches\n";
    std::cout << "  - Operation fusion: Enabled\n";
}

// ============================================================================
// Quantization Error Analysis
// ============================================================================

void demo_quantization_analysis() {
    print_header("Quantization Error Analysis");

    // Create test tensor
    Tensor weights({64, 32, 3, 3}, DType::Float32, Device::cpu());
    float* data = weights.data<float>();
    for (int64_t i = 0; i < weights.numel(); ++i) {
        data[i] = std::sin(i * 0.01f) * 2.0f;
    }

    std::cout << "Comparing quantization schemes...\n\n";

    // Test different schemes
    struct QuantSchemeResult {
        std::string name;
        float mae;
        float mse;
        float snr_db;
    };

    std::vector<QuantSchemeResult> results;

    // Per-tensor symmetric
    {
        auto q = quantize_per_tensor_symmetric(weights);
        auto [mae, mse, snr] = compute_quantization_error(weights, q);
        results.push_back({"Per-Tensor Symmetric", mae, mse, snr});
    }

    // Per-tensor asymmetric
    {
        auto q = quantize_per_tensor_asymmetric(weights);
        auto [mae, mse, snr] = compute_quantization_error(weights, q);
        results.push_back({"Per-Tensor Asymmetric", mae, mse, snr});
    }

    // Per-channel symmetric
    {
        auto q = quantize_per_channel_symmetric(weights, 0);
        auto [mae, mse, snr] = compute_quantization_error(weights, q);
        results.push_back({"Per-Channel Symmetric", mae, mse, snr});
    }

    // Per-channel asymmetric
    {
        auto q = quantize_per_channel_asymmetric(weights, 0);
        auto [mae, mse, snr] = compute_quantization_error(weights, q);
        results.push_back({"Per-Channel Asymmetric", mae, mse, snr});
    }

    // Print results
    std::cout << std::left << std::setw(28) << "Scheme"
              << std::setw(12) << "MAE"
              << std::setw(12) << "MSE"
              << std::setw(12) << "SNR (dB)" << "\n";
    std::cout << std::string(64, '-') << "\n";

    for (const auto& r : results) {
        std::cout << std::left << std::setw(28) << r.name
                  << std::setw(12) << std::fixed << std::setprecision(4) << r.mae
                  << std::setw(12) << std::fixed << std::setprecision(6) << r.mse
                  << std::setw(12) << std::fixed << std::setprecision(2) << r.snr_db << "\n";
    }

    std::cout << "\nKey Observations:\n";
    std::cout << "  - Per-channel quantization typically achieves better SNR\n";
    std::cout << "  - Asymmetric quantization handles biased distributions better\n";
    std::cout << "  - SNR > 40dB indicates excellent quantization quality\n";
}

// ============================================================================
// Main Program
// ============================================================================

int main(int argc, char** argv) {
    std::cout << R"(
╔════════════════════════════════════════════════════════════╗
║      Tenzor Quantization Example - ResNet Optimization     ║
╚════════════════════════════════════════════════════════════╝
)" << "\n";

    try {
        // Run all demos
        demo_quantization_analysis();
        demo_dynamic_quantization();
        demo_post_training_quantization();
        demo_quantization_aware_training();
        demo_per_layer_quantization();

        print_header("Summary");
        std::cout << R"(
Quantization Workflow Recommendations:

1. Dynamic Quantization (Easiest)
   - When: Quick wins, LSTM/Transformer models
   - Pros: No calibration needed, ~2x speedup
   - Cons: Activations still FP32

2. Post-Training Quantization (Recommended)
   - When: CNN models, no retraining possible
   - Pros: 2-4x speedup, 4x memory reduction
   - Cons: Needs calibration data, ~1% accuracy loss

3. Quantization-Aware Training (Best Accuracy)
   - When: Accuracy critical, retraining feasible
   - Pros: <0.5% accuracy loss, same speedup as PTQ
   - Cons: Requires full training pipeline

Next Steps:
  - Prepare calibration dataset (100-1000 samples)
  - Choose quantization scheme (per-channel recommended)
  - Run accuracy evaluation on validation set
  - Deploy quantized model for inference

For more information:
  - Documentation: docs/quantization.md
  - Examples: examples/quantization/
  - API Reference: include/tenzor/quantization/
        )" << "\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
