# HIP Activation Functions - Quick Reference Card

## File Locations
```
Source: src/backends/rocm/kernels/activations.hip.cpp
Header: (to be created) include/tenzor/backends/rocm/activations.hpp
Tests:  (to be created) tests/backends/rocm/test_activations.cpp
```

## Available Activation Functions

### 1. ReLU (Rectified Linear Unit)
```cpp
// Forward: y = max(0, x)
extern "C" void relu_forward_float(const float* input, float* output, int64_t n);
extern "C" void relu_forward_double(const double* input, double* output, int64_t n);

// Backward: dy/dx = (x > 0) ? 1 : 0
extern "C" void relu_backward_float(const float* grad_out, const float* input, float* grad_in, int64_t n);
extern "C" void relu_backward_double(const double* grad_out, const double* input, double* grad_in, int64_t n);
```

**Use Case:** Default activation for most neural networks.

---

### 2. Sigmoid
```cpp
// Forward: y = 1 / (1 + exp(-x))
extern "C" void sigmoid_forward_float(const float* input, float* output, int64_t n);
extern "C" void sigmoid_forward_double(const double* input, double* output, int64_t n);

// Backward: dy/dx = sigmoid(x) * (1 - sigmoid(x))
extern "C" void sigmoid_backward_float(const float* grad_out, const float* input, float* grad_in, int64_t n);
extern "C" void sigmoid_backward_double(const double* grad_out, const double* input, double* grad_in, int64_t n);
```

**Use Case:** Binary classification, gates in RNNs/LSTMs.

---

### 3. Tanh (Hyperbolic Tangent)
```cpp
// Forward: y = tanh(x)
extern "C" void tanh_forward_float(const float* input, float* output, int64_t n);
extern "C" void tanh_forward_double(const double* input, double* output, int64_t n);

// Backward: dy/dx = 1 - tanh(x)^2
extern "C" void tanh_backward_float(const float* grad_out, const float* input, float* grad_in, int64_t n);
extern "C" void tanh_backward_double(const double* grad_out, const double* input, double* grad_in, int64_t n);
```

**Use Case:** RNNs, normalized outputs in [-1, 1].

---

### 4. GELU (Gaussian Error Linear Unit)
```cpp
// Forward: y = x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
extern "C" void gelu_forward_float(const float* input, float* output, int64_t n);
extern "C" void gelu_forward_double(const double* input, double* output, int64_t n);

// Backward: computed gradient
extern "C" void gelu_backward_float(const float* grad_out, const float* input, float* grad_in, int64_t n);
extern "C" void gelu_backward_double(const double* grad_out, const double* input, double* grad_in, int64_t n);
```

**Use Case:** BERT, GPT, transformers (state-of-the-art NLP).

---

### 5. Leaky ReLU
```cpp
// Forward: y = (x > 0) ? x : alpha * x
extern "C" void leaky_relu_forward_float(const float* input, float* output, int64_t n, float alpha);
extern "C" void leaky_relu_forward_double(const double* input, double* output, int64_t n, double alpha);

// Backward: dy/dx = (x > 0) ? 1 : alpha
extern "C" void leaky_relu_backward_float(const float* grad_out, const float* input, float* grad_in, int64_t n, float alpha);
extern "C" void leaky_relu_backward_double(const double* grad_out, const double* input, double* grad_in, int64_t n, double alpha);
```

**Use Case:** Prevent dying ReLU problem. Typical alpha = 0.01.

---

### 6. ELU (Exponential Linear Unit)
```cpp
// Forward: y = (x > 0) ? x : alpha * (exp(x) - 1)
extern "C" void elu_forward_float(const float* input, float* output, int64_t n, float alpha);
extern "C" void elu_forward_double(const double* input, double* output, int64_t n, double alpha);

// Backward: dy/dx = (x > 0) ? 1 : alpha * exp(x)
extern "C" void elu_backward_float(const float* grad_out, const float* input, float* grad_in, int64_t n, float alpha);
extern "C" void elu_backward_double(const double* grad_out, const double* input, double* grad_in, int64_t n, double alpha);
```

**Use Case:** Smoother than ReLU, better gradient flow. Typical alpha = 1.0.

---

### 7. SELU (Scaled Exponential Linear Unit)
```cpp
// Forward: y = scale * ((x > 0) ? x : alpha * (exp(x) - 1))
// Standard: alpha = 1.6733, scale = 1.0507
extern "C" void selu_forward_float(const float* input, float* output, int64_t n);
extern "C" void selu_forward_double(const double* input, double* output, int64_t n);

// Backward: computed with standard parameters
extern "C" void selu_backward_float(const float* grad_out, const float* input, float* grad_in, int64_t n);
extern "C" void selu_backward_double(const double* grad_out, const double* input, double* grad_in, int64_t n);
```

**Use Case:** Self-normalizing networks, no batch norm needed.

---

### 8. Swish / SiLU (Sigmoid Linear Unit)
```cpp
// Forward: y = x * sigmoid(x)
extern "C" void swish_forward_float(const float* input, float* output, int64_t n);
extern "C" void swish_forward_double(const double* input, double* output, int64_t n);

// Backward: dy/dx = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
extern "C" void swish_backward_float(const float* grad_out, const float* input, float* grad_in, int64_t n);
extern "C" void swish_backward_double(const double* grad_out, const double* input, double* grad_in, int64_t n);
```

**Use Case:** Deep networks, smooth non-monotonic activation.

---

### 9. Mish
```cpp
// Forward: y = x * tanh(softplus(x))
extern "C" void mish_forward_float(const float* input, float* output, int64_t n);
extern "C" void mish_forward_double(const double* input, double* output, int64_t n);

// Backward: computed gradient with numerical stability
extern "C" void mish_backward_float(const float* grad_out, const float* input, float* grad_in, int64_t n);
extern "C" void mish_backward_double(const double* grad_out, const double* input, double* grad_in, int64_t n);
```

**Use Case:** State-of-the-art image classification, YOLOv4.

---

### 10. Softmax
```cpp
// Forward: y_i = exp(x_i - max) / sum(exp(x_j - max))
extern "C" void softmax_forward_float(const float* input, float* output, int64_t batch_size, int64_t dim_size);
extern "C" void softmax_forward_double(const double* input, double* output, int64_t batch_size, int64_t dim_size);

// With temperature scaling
extern "C" void softmax_forward_float_temperature(const float* input, float* output,
                                                   int64_t batch_size, int64_t dim_size, float temperature);
extern "C" void softmax_forward_double_temperature(const double* input, double* output,
                                                    int64_t batch_size, int64_t dim_size, double temperature);

// Backward: dy/dx_i = softmax_i * (grad_out_i - sum(grad_out * softmax))
extern "C" void softmax_backward_float(const float* grad_out, const float* output, float* grad_in,
                                       int64_t batch_size, int64_t dim_size);
extern "C" void softmax_backward_double(const double* grad_out, const double* output, double* grad_in,
                                        int64_t batch_size, int64_t dim_size);
```

**Use Case:** Multi-class classification, attention mechanisms.

**Temperature Parameter:**
- `T < 1.0`: Sharper distribution (more confident)
- `T = 1.0`: Standard softmax
- `T > 1.0`: Smoother distribution (less confident)

---

### 11. LogSoftmax
```cpp
// Forward: y = x - max - log(sum(exp(x - max)))
extern "C" void log_softmax_forward_float(const float* input, float* output, int64_t batch_size, int64_t dim_size);
extern "C" void log_softmax_forward_double(const double* input, double* output, int64_t batch_size, int64_t dim_size);

// Backward: dy/dx = grad_out - exp(log_softmax) * sum(grad_out)
extern "C" void log_softmax_backward_float(const float* grad_out, const float* output, float* grad_in,
                                           int64_t batch_size, int64_t dim_size);
extern "C" void log_softmax_backward_double(const double* grad_out, const double* output, double* grad_in,
                                            int64_t batch_size, int64_t dim_size);
```

**Use Case:** Combined with NLLLoss for classification, more numerically stable.

---

## Tensor Wrapper API (C++)

```cpp
namespace tenzor::rocm {

// Element-wise activations
auto relu_kernel(const Tensor& input, hipStream_t stream = 0) -> Tensor;
auto relu_backward_kernel(const Tensor& grad_output, const Tensor& input, hipStream_t stream = 0) -> Tensor;

auto sigmoid_kernel(const Tensor& input, hipStream_t stream = 0) -> Tensor;
auto sigmoid_backward_kernel(const Tensor& grad_output, const Tensor& input, hipStream_t stream = 0) -> Tensor;

auto tanh_kernel(const Tensor& input, hipStream_t stream = 0) -> Tensor;
auto tanh_backward_kernel(const Tensor& grad_output, const Tensor& input, hipStream_t stream = 0) -> Tensor;

auto gelu_kernel(const Tensor& input, hipStream_t stream = 0) -> Tensor;
auto gelu_backward_kernel(const Tensor& grad_output, const Tensor& input, hipStream_t stream = 0) -> Tensor;

auto leaky_relu_kernel(const Tensor& input, float alpha, hipStream_t stream = 0) -> Tensor;
auto leaky_relu_backward_kernel(const Tensor& grad_output, const Tensor& input, float alpha, hipStream_t stream = 0) -> Tensor;

// Softmax activations (specify dimension)
auto softmax_kernel(const Tensor& input, int64_t dim, hipStream_t stream = 0, float temperature = 1.0f) -> Tensor;
auto softmax_backward_kernel(const Tensor& grad_output, const Tensor& output, int64_t dim, hipStream_t stream = 0) -> Tensor;

auto log_softmax_kernel(const Tensor& input, int64_t dim, hipStream_t stream = 0) -> Tensor;
auto log_softmax_backward_kernel(const Tensor& grad_output, const Tensor& output, int64_t dim, hipStream_t stream = 0) -> Tensor;

} // namespace tenzor::rocm
```

## Usage Examples

### Example 1: Basic ReLU
```cpp
#include "tenzor/backends/rocm/activations.hpp"

// Allocate GPU tensors
Tensor input = Tensor::randn({1024, 512}, DType::Float32, Device::ROCM);
hipStream_t stream = 0;

// Forward pass
Tensor output = tenzor::rocm::relu_kernel(input, stream);

// Backward pass
Tensor grad_output = Tensor::ones_like(output);
Tensor grad_input = tenzor::rocm::relu_backward_kernel(grad_output, input, stream);
```

### Example 2: Softmax with Temperature
```cpp
// Model logits for text generation
Tensor logits = model.forward(input);  // Shape: [batch, vocab_size]

// Apply temperature scaling for diverse sampling
float temperature = 1.2f;  // Smoother distribution
Tensor probs = tenzor::rocm::softmax_kernel(logits, -1, 0, temperature);

// Sample from distribution
int token_id = sample_multinomial(probs);
```

### Example 3: GELU in Transformer
```cpp
// Transformer feed-forward network
class FeedForward {
    Tensor linear1(const Tensor& x) { /* ... */ }
    Tensor linear2(const Tensor& x) { /* ... */ }

    Tensor forward(const Tensor& x) {
        Tensor h1 = linear1(x);
        Tensor h1_activated = tenzor::rocm::gelu_kernel(h1);  // GELU activation
        Tensor output = linear2(h1_activated);
        return output;
    }
};
```

### Example 4: Leaky ReLU with Custom Alpha
```cpp
// Network with leaky ReLU (alpha = 0.2)
float alpha = 0.2f;
Tensor x = conv_layer(input);
Tensor activated = tenzor::rocm::leaky_relu_kernel(x, alpha);
```

## Performance Tips

### 1. Stream Usage
```cpp
// Overlap computation with memory transfers
hipStream_t stream1, stream2;
hipStreamCreate(&stream1);
hipStreamCreate(&stream2);

// Async operations on different streams
Tensor out1 = tenzor::rocm::relu_kernel(input1, stream1);
Tensor out2 = tenzor::rocm::gelu_kernel(input2, stream2);

hipStreamSynchronize(stream1);
hipStreamSynchronize(stream2);
```

### 2. In-Place Operations
```cpp
// For memory-constrained scenarios, consider in-place modifications
// (Note: Requires wrapper function modification)
extern "C" void relu_forward_inplace_float(float* data, int64_t n);
```

### 3. Batch Operations
```cpp
// Process multiple batches efficiently
for (const auto& batch : dataloader) {
    Tensor logits = model.forward(batch);
    Tensor probs = tenzor::rocm::softmax_kernel(logits, -1);  // Dim -1 = last dimension
    Tensor loss = criterion(probs, batch.labels);
    loss.backward();
}
```

## Memory Requirements

| Activation | Memory Usage | Notes |
|------------|--------------|-------|
| ReLU, Sigmoid, Tanh | 2x input size | Input + output |
| Leaky ReLU, ELU | 2x input size | Input + output |
| GELU, Swish, Mish | 2x input size | Input + output |
| Softmax | 2x input size + shared mem | Requires reduction |
| LogSoftmax | 2x input size + shared mem | Requires reduction |

**Shared Memory:** `256 * sizeof(float)` = 1 KB per block (softmax/log_softmax)

## Error Handling

All functions check for errors using `HIP_CHECK()` macro:

```cpp
#define HIP_CHECK(call) \
    do { \
        hipError_t err = call; \
        if (err != hipSuccess) { \
            fprintf(stderr, "HIP error at %s:%d: %s\n", \
                    __FILE__, __LINE__, hipGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)
```

For production code, consider replacing `exit()` with exception throwing:

```cpp
if (err != hipSuccess) {
    throw std::runtime_error("HIP kernel launch failed: " +
                             std::string(hipGetErrorString(err)));
}
```

## Compilation

### hipcc Command
```bash
hipcc -O3 -ffast-math \
      --amdgpu-target=gfx90a \
      -I/path/to/tenzor/include \
      src/backends/rocm/kernels/activations.hip.cpp \
      -o libtenzor_rocm_kernels.so \
      -shared -fPIC
```

### CMake
```cmake
find_package(HIP REQUIRED)

hip_add_library(tenzor_rocm_kernels SHARED
    src/backends/rocm/kernels/activations.hip.cpp
)

target_include_directories(tenzor_rocm_kernels PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

target_compile_options(tenzor_rocm_kernels PRIVATE
    -O3
    -ffast-math
    --amdgpu-target=gfx90a  # MI200 series
)
```

## Benchmarking Template

```cpp
#include <hip/hip_runtime.h>
#include <chrono>

// Benchmark helper
template<typename Func>
float benchmark_kernel(Func kernel_func, int iterations = 100) {
    // Warm-up
    kernel_func();
    hipDeviceSynchronize();

    // Timed runs
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        kernel_func();
    }
    hipDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    return duration.count() / (float)iterations;  // Average time in μs
}

// Usage
float avg_time = benchmark_kernel([&]() {
    tenzor::rocm::relu_kernel(input, stream);
});

printf("Average kernel time: %.2f μs\n", avg_time);
```

## Activation Function Selection Guide

| Task | Recommended Activation | Alternative |
|------|------------------------|-------------|
| Image Classification | ReLU, Mish | Leaky ReLU, Swish |
| Object Detection | Mish, Swish | ReLU, Leaky ReLU |
| NLP/Transformers | GELU | Swish |
| RNN/LSTM Gates | Sigmoid, Tanh | - |
| Output Layer (Binary) | Sigmoid | - |
| Output Layer (Multi-class) | Softmax | LogSoftmax + NLLLoss |
| Self-Normalizing Nets | SELU | - |
| Dying ReLU Problem | Leaky ReLU, ELU | Swish, Mish |

## Gradient Checking

```cpp
// Numerical gradient check for validation
float epsilon = 1e-5f;
Tensor x = Tensor::randn({100}, DType::Float32, Device::ROCM);

// Forward pass
Tensor y = tenzor::rocm::gelu_kernel(x);

// Analytical gradient
Tensor grad_out = Tensor::ones_like(y);
Tensor grad_analytical = tenzor::rocm::gelu_backward_kernel(grad_out, x);

// Numerical gradient
Tensor grad_numerical(x.shape(), x.dtype(), x.device());
for (int i = 0; i < x.numel(); ++i) {
    float original = x.data<float>()[i];

    // f(x + epsilon)
    x.data<float>()[i] = original + epsilon;
    Tensor y_plus = tenzor::rocm::gelu_kernel(x);

    // f(x - epsilon)
    x.data<float>()[i] = original - epsilon;
    Tensor y_minus = tenzor::rocm::gelu_kernel(x);

    // Gradient: (f(x+eps) - f(x-eps)) / (2*eps)
    grad_numerical.data<float>()[i] =
        (y_plus.data<float>()[i] - y_minus.data<float>()[i]) / (2 * epsilon);

    // Restore original value
    x.data<float>()[i] = original;
}

// Compare
float max_error = (grad_analytical - grad_numerical).abs().max().item<float>();
printf("Max gradient error: %e\n", max_error);
assert(max_error < 1e-4f);  // Acceptable for float32
```

## Troubleshooting

### Issue: Kernel Launch Fails
**Solution:** Check tensor device placement and stream validity.

### Issue: NaN/Inf in Output
**Solution:** Verify numerical stability (especially in exp/log operations).

### Issue: Low Performance
**Solution:** Profile with `rocprof`, check occupancy and memory access patterns.

### Issue: Compilation Errors
**Solution:** Ensure ROCm and HIP SDK are correctly installed and in PATH.

---

**Quick Reference Version:** 1.0
**Last Updated:** 2025-10-14
**Compatibility:** ROCm 5.0+, HIP 5.0+
