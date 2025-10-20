/**
 * @file quantization.hpp
 * @brief Main header for quantization support
 *
 * Includes all quantization components for easy access:
 * - Quantization operations and schemes
 * - Observers for calibration
 * - Fake quantization for QAT
 * - Quantization configuration
 * - Quantized layers
 *
 * @example
 * #include <tenzor/nn/quantization.hpp>
 *
 * using namespace tenzor::nn::quantization;
 *
 * // Post-training quantization
 * auto observer = std::make_unique<MinMaxObserver>();
 * observer->observe(activations);
 * auto qparams = observer->calculate_qparams(QuantDType::INT8,
 *                                           QuantizationScheme::PerTensorSymmetric);
 * auto q_tensor = quantize_tensor(weights, qparams);
 *
 * // Quantization-aware training
 * auto fake_quant = std::make_shared<FakeQuantize>();
 * fake_quant->train();
 * auto output = fake_quant->forward(input);
 */

#pragma once

// Core quantization operations
#include "quantization/quantize.hpp"

// Observers for calibration
#include "quantization/observer.hpp"

// Fake quantization for QAT
#include "quantization/fake_quantize.hpp"

// Configuration
#include "quantization/qconfig.hpp"

// Quantized layers
#include "quantization/quantized_layers.hpp"

namespace tenzor {
namespace nn {

/**
 * @brief Quantization support for Tenzor neural networks.
 *
 * The quantization namespace provides comprehensive support for:
 *
 * 1. **Post-Training Quantization (PTQ)**:
 *    - Calibrate on sample data
 *    - Convert trained FP32 models to INT8
 *    - Minimal accuracy loss (~1-2%)
 *    - 4x memory reduction, 2-4x speedup
 *
 * 2. **Quantization-Aware Training (QAT)**:
 *    - Train with quantization simulation
 *    - Better accuracy than PTQ (<0.5% loss)
 *    - Same deployment benefits as PTQ
 *
 * 3. **Quantization Schemes**:
 *    - Per-tensor symmetric/asymmetric
 *    - Per-channel symmetric/asymmetric
 *    - INT8 and UINT8 data types
 *
 * 4. **Observers**:
 *    - MinMaxObserver: Fast, simple statistics
 *    - MovingAverageMinMaxObserver: Smooth updates
 *    - HistogramObserver: Robust to outliers
 *
 * 5. **Quantized Operations**:
 *    - QuantizedLinear
 *    - QuantizedConv2d
 *    - QuantizedBatchNorm2d
 *    - Fused layers (Conv-BN-ReLU)
 *
 * @section usage Usage
 *
 * **Post-Training Quantization:**
 * @code
 * // 1. Create observer
 * auto observer = std::make_unique<MinMaxObserver>();
 *
 * // 2. Calibrate on representative data
 * for (auto& batch : calibration_data) {
 *     observer->observe(model.forward(batch).tensor());
 * }
 *
 * // 3. Calculate quantization parameters
 * auto qparams = observer->calculate_qparams(
 *     QuantDType::INT8,
 *     QuantizationScheme::PerTensorSymmetric
 * );
 *
 * // 4. Quantize model
 * auto q_model = quantize_model(model, qparams);
 * @endcode
 *
 * **Quantization-Aware Training:**
 * @code
 * // 1. Create model with fake quantization
 * auto model = std::make_shared<MyModel>();
 * auto fake_quant = std::make_shared<FakeQuantize>(
 *     QuantDType::INT8,
 *     QuantizationScheme::PerTensorSymmetric
 * );
 *
 * // 2. Train with quantization simulation
 * model->train();
 * for (auto& batch : training_data) {
 *     auto output = fake_quant->forward(model->forward(batch));
 *     loss = criterion(output, labels);
 *     loss.backward();
 *     optimizer.step();
 * }
 *
 * // 3. Convert to quantized model
 * fake_quant->disable_observer();
 * auto q_model = convert_to_quantized(model);
 * @endcode
 *
 * @section performance Performance
 *
 * Typical results on modern CPUs/GPUs:
 * - **Memory**: 4x reduction (FP32 → INT8)
 * - **Latency**: 2-4x speedup on CPU, 2x on GPU
 * - **Accuracy**: <1% loss with PTQ, <0.5% with QAT
 *
 * @section backends Backends
 *
 * CPU backends:
 * - FBGEMM (x86)
 * - QNNPACK (ARM)
 * - OneDNN (Intel)
 *
 * GPU backends:
 * - CUDA TensorCore INT8 (Turing+)
 *
 * @see quantization::quantize_tensor()
 * @see quantization::Observer
 * @see quantization::FakeQuantize
 * @see quantization::QConfig
 */
namespace quantization {

/**
 * @brief Version information for quantization API.
 */
struct QuantizationVersion {
    static constexpr int MAJOR = 1;
    static constexpr int MINOR = 0;
    static constexpr int PATCH = 0;

    static auto version_string() -> const char* {
        return "1.0.0";
    }
};

} // namespace quantization
} // namespace nn
} // namespace tenzor
