# Tenzor Quantization Guide

## Overview

Quantization reduces model size and improves inference speed by converting floating-point operations to lower-precision integer operations. Tenzor provides comprehensive INT8 quantization support for neural network inference optimization.

## Table of Contents

1. [Introduction](#introduction)
2. [Quantization Methods](#quantization-methods)
3. [Quick Start](#quick-start)
4. [API Reference](#api-reference)
5. [Advanced Usage](#advanced-usage)
6. [Performance Guide](#performance-guide)
7. [Troubleshooting](#troubleshooting)

## Introduction

### What is Quantization?

Quantization converts 32-bit floating-point weights and activations to 8-bit integers:
- **FP32 → INT8**: 4x memory reduction
- **Performance**: 2-4x faster inference on CPUs
- **Accuracy**: <1% loss with proper calibration

### When to Use Quantization

✅ **Good for:**
- Production deployment (CPU inference)
- Resource-constrained environments
- Batch inference workloads
- CNNs and fully-connected networks

⚠️ **Limitations:**
- May require calibration data
- Some accuracy loss (<1-2%)
- INT8 ops may not be faster on all GPUs

## Quantization Methods

### 1. Dynamic Quantization (Easiest)

**What**: Quantize weights only, activations computed in FP32
**When**: Quick wins, no calibration data available
**Speedup**: ~2x
**Accuracy Loss**: Minimal

```cpp
#include "tenzor/quantization/quantize_api.hpp"

auto model = models::resnet50(1000);
auto quantized_model = quantization::quantize_dynamic(model);
```

### 2. Post-Training Quantization (Recommended)

**What**: Quantize weights and activations
**When**: Maximum performance needed
**Speedup**: 2-4x
**Accuracy Loss**: <1%
**Requires**: Calibration dataset

```cpp
// Prepare calibration function
auto calibrate_fn = [&](nn::Module& m) {
    for (auto& batch : calibration_loader) {
        m.forward(batch.data);
    }
};

// Quantize model
auto qconfig = DefaultQConfigs::default_qconfig();
auto quantized_model = quantization::quantize_static(
    model, calibrate_fn, qconfig
);
```

### 3. Quantization-Aware Training (Best Accuracy)

**What**: Train model to be quantization-robust
**When**: Accuracy critical, can retrain
**Speedup**: 2-4x
**Accuracy Loss**: <0.5%
**Requires**: Full training pipeline

```cpp
// Prepare model for QAT
auto qat_model = quantization::prepare_qat(model);

// Train with quantization simulation
for (int epoch = 0; epoch < epochs; ++epoch) {
    for (auto& batch : train_loader) {
        auto output = qat_model->forward(batch.data);
        auto loss = criterion(output, batch.labels);
        loss.backward();
        optimizer.step();
    }
}

// Convert to quantized model
auto quantized_model = quantization::convert_qat(qat_model);
```

## Quick Start

### Example: Quantize ResNet50

```cpp
#include "tenzor/tenzor.hpp"
#include "tenzor/models/resnet.hpp"
#include "tenzor/quantization/quantize_api.hpp"

int main() {
    // 1. Load model
    auto model = models::resnet50(1000);
    model->eval();

    // 2. Prepare calibration data
    std::vector<Tensor> calibration_data;
    for (int i = 0; i < 100; ++i) {
        auto batch = load_batch(i);  // Your data loading
        calibration_data.push_back(batch);
    }

    // 3. Quantize
    auto calibrate_fn = [&](nn::Module& m) {
        for (const auto& batch : calibration_data) {
            m.forward(autograd::Variable(batch, false));
        }
    };

    auto quantized_model = quantization::quantize_static(
        model, calibrate_fn
    );

    // 4. Use quantized model
    auto output = quantized_model->forward(input);

    return 0;
}
```

## API Reference

### Quantization Schemes

#### Per-Tensor Symmetric
- **Best for**: Activations
- **Range**: [-127, 127]
- **Zero-point**: 0
- **Formula**: `q = round(x / scale)`

#### Per-Tensor Asymmetric
- **Best for**: Non-uniform distributions
- **Range**: [-128, 127]
- **Zero-point**: Variable
- **Formula**: `q = round(x / scale) + zero_point`

#### Per-Channel Symmetric
- **Best for**: Weights (convolution/linear)
- **Range**: [-127, 127] per channel
- **Zero-point**: 0 per channel
- **Formula**: `q[c] = round(x[c] / scale[c])`

#### Per-Channel Asymmetric
- **Best for**: Extreme weight distributions
- **Range**: [-128, 127] per channel
- **Zero-point**: Variable per channel

### Observers

Observers collect statistics during calibration to determine quantization parameters.

#### MinMaxObserver
```cpp
auto observer = std::make_unique<MinMaxObserver>();
observer->observe(tensor);
auto qparams = observer->calculate_qparams(
    QuantDType::INT8,
    QuantizationScheme::PerTensorSymmetric
);
```

**Characteristics:**
- Fast and simple
- Sensitive to outliers
- Good for uniform distributions

#### MovingAverageMinMaxObserver
```cpp
auto observer = std::make_unique<MovingAverageMinMaxObserver>(0.9f);
```

**Characteristics:**
- Smooths min/max over batches
- Less sensitive to outliers
- Good for QAT

#### HistogramObserver
```cpp
auto observer = std::make_unique<HistogramObserver>(2048, 0.001f, 0.999f);
```

**Characteristics:**
- Most robust to outliers
- Uses percentile clipping
- Slower but higher accuracy

### QConfig Presets

```cpp
// Default - good for most models
auto qconfig = DefaultQConfigs::default_qconfig();

// High accuracy - histogram-based
auto qconfig = DefaultQConfigs::high_accuracy_qconfig();

// Fast - minimal calibration time
auto qconfig = DefaultQConfigs::fast_qconfig();

// QAT - moving average for smooth training
auto qconfig = DefaultQConfigs::qat_qconfig();

// UINT8 activations - for ReLU networks
auto qconfig = DefaultQConfigs::uint8_activation_qconfig();
```

## Advanced Usage

### Per-Layer Quantization Configuration

```cpp
using namespace tenzor::nn::quantization;

// Create custom quantization strategy
auto strategy = QuantizationStrategyBuilder()
    .set_global_qconfig(DefaultQConfigs::default_qconfig())
    .set_layer_qconfig("conv1", DefaultQConfigs::high_accuracy_qconfig())
    .set_layer_qconfig("fc", DefaultQConfigs::high_accuracy_qconfig())
    .set_type_qconfig("BatchNorm2d", DefaultQConfigs::fast_qconfig())
    .disable_layer("layer1.0.downsample")  // Keep in FP32
    .set_backend(QuantizationBackend::FBGEMM)
    .set_calibration_batches(200)
    .enable_operation_fusion(true)
    .build();
```

### Custom Quantization Parameters

```cpp
// Create observer
auto observer = std::make_unique<MinMaxObserver>();

// Collect statistics
for (const auto& batch : calibration_data) {
    observer->observe(batch);
}

// Calculate quantization parameters
auto qparams = observer->calculate_qparams(
    QuantDType::INT8,
    QuantizationScheme::PerChannelSymmetric
);

// Quantize tensor
auto q_tensor = quantize_tensor(weights, qparams);

// Dequantize if needed
Tensor fp_tensor = dequantize_tensor(q_tensor);
```

### Quantization Error Analysis

```cpp
// Quantize tensor
auto q_tensor = quantize_per_tensor_symmetric(tensor);

// Compute error metrics
auto [mae, mse, snr_db] = compute_quantization_error(tensor, q_tensor);

std::cout << "Mean Absolute Error: " << mae << std::endl;
std::cout << "Mean Squared Error: " << mse << std::endl;
std::cout << "Signal-to-Noise Ratio: " << snr_db << " dB" << std::endl;
```

### Module Fusion

Fuse sequential operations for better quantized performance:

```cpp
// Fuse Conv-BN-ReLU patterns
auto fused_model = quantization::fuse_modules(model);

// Then quantize
auto quantized_model = quantization::quantize_static(
    fused_model, calibrate_fn
);
```

## Performance Guide

### Expected Performance Improvements

| Model Type | Speedup | Memory | Accuracy Loss |
|------------|---------|--------|---------------|
| ResNet50 (CPU) | 2.5-3.5x | 4x | <1% |
| MobileNetV2 (CPU) | 3.0-4.0x | 4x | <0.5% |
| BERT-Base (CPU) | 2.0-2.5x | 4x | <1% |

### Optimization Tips

1. **Use Per-Channel Quantization for Weights**
   ```cpp
   auto qconfig = DefaultQConfigs::default_qconfig();  // Uses per-channel
   ```

2. **Calibration Dataset Size**
   - Minimum: 50-100 representative samples
   - Recommended: 100-1000 samples
   - More is not always better (diminishing returns)

3. **Histogram vs MinMax**
   - MinMax: Faster calibration
   - Histogram: Better accuracy with outliers

4. **Backend Selection**
   ```cpp
   .set_backend(QuantizationBackend::FBGEMM)  // x86 CPUs
   .set_backend(QuantizationBackend::QNNPACK)  // ARM CPUs
   .set_backend(QuantizationBackend::OneDNN)   // Intel optimized
   ```

### Benchmarking

```cpp
auto [fp32_time, int8_time, speedup, memory_reduction] =
    quantization::benchmark_quantization(
        *fp32_model,
        *quantized_model,
        {1, 3, 224, 224},  // Input shape
        100  // Iterations
    );

std::cout << "Speedup: " << speedup << "x" << std::endl;
```

## Quantization Workflow

### Recommended Workflow

```
┌─────────────────┐
│  Train FP32     │
│     Model       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Prepare        │
│  Calibration    │  (100-1000 samples)
│     Data        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Configure     │
│  Quantization   │  (QConfig, observers)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Calibrate     │
│   & Quantize    │  (PTQ or QAT)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Evaluate      │
│   Accuracy      │  (Compare FP32 vs INT8)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Benchmark     │
│  Performance    │  (Latency, throughput)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│     Deploy      │
│ Quantized Model │
└─────────────────┘
```

## Troubleshooting

### High Accuracy Loss (>2%)

**Problem**: Quantized model performs poorly

**Solutions**:
1. Use histogram observer instead of minmax
2. Increase calibration dataset size
3. Try per-channel quantization
4. Use QAT instead of PTQ
5. Check for outliers in activations

```cpp
// Switch to histogram observer
auto qconfig = DefaultQConfigs::high_accuracy_qconfig();
```

### Low Speedup (<2x)

**Problem**: Quantization not improving performance

**Possible Causes**:
1. Model is memory-bound, not compute-bound
2. Using wrong backend
3. Model has many non-quantizable layers

**Solutions**:
1. Check backend selection
2. Ensure INT8 ops are actually being used
3. Profile to find bottlenecks

### Calibration Issues

**Problem**: Observer reports no data or crashes

**Solutions**:
1. Ensure calibration data is on correct device
2. Check data type (should be FP32)
3. Verify data range is reasonable
4. Use more calibration batches

### Numerical Instability

**Problem**: NaN or Inf values after quantization

**Solutions**:
1. Check for extreme values in weights/activations
2. Use gradient clipping during QAT
3. Normalize inputs properly
4. Consider per-channel quantization

## Best Practices

### ✅ DO

- Use calibration data similar to production data
- Start with PTQ before trying QAT
- Benchmark on target hardware
- Monitor quantization error metrics
- Use per-channel quantization for weights
- Fuse operations before quantizing
- Validate accuracy on holdout set

### ❌ DON'T

- Quantize without calibration (except dynamic)
- Use too few calibration samples (<50)
- Assume quantization always improves speed
- Ignore accuracy degradation >2%
- Quantize first and last layers aggressively
- Mix quantized and non-quantized ops carelessly

## Examples

See full examples in:
- `examples/quantization/quantize_resnet.cpp` - Complete ResNet quantization
- `tests/test_quantization.cpp` - Unit tests with usage patterns

## References

- PyTorch Quantization: https://pytorch.org/docs/stable/quantization.html
- TensorFlow Lite Quantization: https://www.tensorflow.org/lite/performance/model_optimization
- NVIDIA TensorRT INT8: https://docs.nvidia.com/deeplearning/tensorrt/
- ONNX Runtime Quantization: https://onnxruntime.ai/docs/performance/quantization.html

## Summary

**Quantization Cheat Sheet:**

| Method | Setup Time | Speedup | Accuracy | Use Case |
|--------|-----------|---------|----------|----------|
| **Dynamic** | Minutes | 2x | No loss | Quick optimization |
| **PTQ** | Hours | 3x | <1% loss | Production deployment |
| **QAT** | Days | 3x | <0.5% loss | Accuracy critical |

Choose based on your requirements and available resources!
