# ONNX Import Guide for Tenzor

## Overview

The Tenzor ONNX Importer provides complete functionality to import ONNX models into the Tenzor framework. This allows you to:

- Load pretrained models from PyTorch, TensorFlow, and other frameworks
- Import models for inference or fine-tuning
- Convert ONNX operators to Tenzor modules
- Support for common architectures (ResNet, VGG, BERT, etc.)

**Supported ONNX Opset**: Version 13 and higher

## Features

### ✅ Supported ONNX Operators

#### Tensor Operations
- **Add** - Element-wise addition
- **Sub** - Element-wise subtraction
- **Mul** - Element-wise multiplication
- **Div** - Element-wise division
- **MatMul** - Matrix multiplication
- **Reshape** - Tensor reshaping
- **Transpose** - Tensor transposition
- **Concat** - Concatenation along axis
- **Split** - Split tensor along axis
- **Flatten** - Flatten tensor dimensions

#### Neural Network Layers
- **Gemm** - General Matrix Multiplication (Linear/Fully-Connected layer)
  - Supports alpha, beta, transA, transB attributes
  - Converts to `tenzor::nn::Linear`
- **Conv** - Convolution (1D, 2D)
  - Supports kernel_shape, strides, pads, dilations, groups
  - Converts to `tenzor::nn::Conv1d` or `tenzor::nn::Conv2d`
- **BatchNormalization** - Batch normalization
  - Supports epsilon, momentum attributes
  - Converts to `tenzor::nn::BatchNorm1d` or `tenzor::nn::BatchNorm2d`

#### Activation Functions
- **Relu** - Rectified Linear Unit
- **LeakyRelu** - Leaky ReLU with alpha parameter
- **Sigmoid** - Sigmoid activation
- **Tanh** - Hyperbolic tangent
- **Gelu** - Gaussian Error Linear Unit
- **Softmax** - Softmax normalization
- **LogSoftmax** - Log-Softmax
- **Elu** - Exponential Linear Unit
- **Selu** - Scaled ELU

#### Pooling Layers
- **MaxPool** - Max pooling (1D, 2D)
  - Supports kernel_shape, strides, pads attributes
  - Converts to `tenzor::nn::MaxPool2d`
- **AveragePool** - Average pooling (1D, 2D)
  - Converts to `tenzor::nn::AvgPool2d`
- **GlobalAveragePool** - Global average pooling
  - Converts to `tenzor::nn::AdaptiveAvgPool2d`

## Installation

The ONNX importer is included in the Tenzor library. No additional dependencies are required.

**Header:**
```cpp
#include <tenzor/onnx/importer.hpp>
```

## Quick Start

### Basic Usage

```cpp
#include <tenzor/onnx/importer.hpp>
#include <tenzor/core/tensor.hpp>
#include <tenzor/autograd/variable.hpp>

using namespace tenzor;

// Import ONNX model
auto model = onnx::import_onnx("resnet50.onnx", true);

// Create input tensor
Tensor input({1, 3, 224, 224}, DType::Float32, Device::cpu());
input.fill_(0.5f);

// Run inference
Variable output = model->forward(Variable(input));

std::cout << "Output shape: " << output.tensor().shape() << std::endl;
```

### Advanced Usage with ONNXImporter

```cpp
#include <tenzor/onnx/importer.hpp>

using namespace tenzor;

// Create importer with configuration
onnx::ONNXImporter importer(true); // verbose=true
importer.set_device(Device::cuda(0)); // Import to GPU

// Import model
auto model = importer.import_from_file("model.onnx");

// Move model to GPU
model->to(Device::cuda(0));

// Set to evaluation mode
model->eval();

// Run inference on GPU
Tensor input({1, 3, 224, 224}, DType::Float32, Device::cuda(0));
Variable output = model->forward(Variable(input));
```

### Import from Memory

```cpp
#include <tenzor/onnx/importer.hpp>
#include <fstream>
#include <vector>

using namespace tenzor;

// Read ONNX file into memory
std::ifstream file("model.onnx", std::ios::binary);
std::vector<uint8_t> onnx_bytes(
    (std::istreambuf_iterator<char>(file)),
    std::istreambuf_iterator<char>()
);

// Import from bytes
onnx::ONNXImporter importer;
auto model = importer.import_from_bytes(onnx_bytes);
```

## API Reference

### High-Level Function

```cpp
auto import_onnx(
    const std::string& filename,
    bool verbose = false,
    Device device = Device::cpu()
) -> std::shared_ptr<nn::Module>;
```

**Parameters:**
- `filename`: Path to ONNX model file
- `verbose`: Enable detailed logging (default: false)
- `device`: Target device for imported model (default: CPU)

**Returns:** Imported Tenzor module

**Throws:** `std::runtime_error` if import fails

---

### ONNXImporter Class

#### Constructor

```cpp
explicit ONNXImporter(bool verbose = false);
```

#### Configuration Methods

```cpp
auto set_verbose(bool verbose) -> void;
auto set_device(Device device) -> void;
```

#### Import Methods

```cpp
auto import_from_file(const std::string& filepath) -> std::shared_ptr<nn::Module>;
auto import_from_bytes(const std::vector<uint8_t>& bytes) -> std::shared_ptr<nn::Module>;
```

#### Inspection

```cpp
auto get_model_data() const -> const ONNXModelData&;
```

## Examples

### Example 1: Import ResNet-50 from ONNX

```cpp
#include <tenzor/onnx/importer.hpp>
#include <tenzor/core/tensor.hpp>

using namespace tenzor;

int main() {
    // Import pretrained ResNet-50
    auto resnet = onnx::import_onnx("resnet50.onnx", true);

    // Create sample input (batch=1, channels=3, height=224, width=224)
    Tensor input({1, 3, 224, 224}, DType::Float32, Device::cpu());

    // Normalize to ImageNet mean/std
    // (In practice, load actual image data)
    input.fill_(0.5f);

    // Set to evaluation mode
    resnet->eval();

    // Run inference
    Variable output = resnet->forward(Variable(input));

    // Output shape: [1, 1000] (ImageNet classes)
    auto output_data = output.tensor().data<float>();

    // Find predicted class
    int predicted_class = 0;
    float max_score = output_data[0];
    for (int i = 1; i < 1000; ++i) {
        if (output_data[i] > max_score) {
            max_score = output_data[i];
            predicted_class = i;
        }
    }

    std::cout << "Predicted class: " << predicted_class << std::endl;
    std::cout << "Confidence: " << max_score << std::endl;

    return 0;
}
```

### Example 2: Import and Fine-Tune

```cpp
#include <tenzor/onnx/importer.hpp>
#include <tenzor/nn/optim/adam.hpp>
#include <tenzor/nn/loss/cross_entropy.hpp>

using namespace tenzor;

int main() {
    // Import model
    auto model = onnx::import_onnx("pretrained_model.onnx");

    // Set to training mode
    model->train();

    // Create optimizer
    nn::Adam optimizer(model->parameters(), 0.001);

    // Training loop
    for (int epoch = 0; epoch < 10; ++epoch) {
        // Forward pass
        Tensor input({32, 3, 224, 224}, DType::Float32, Device::cpu());
        Tensor target({32}, DType::Int64, Device::cpu());

        // Fill with training data
        // ... (load your data)

        Variable output = model->forward(Variable(input));

        // Compute loss
        auto loss = nn::cross_entropy(output, Variable(target));

        // Backward pass
        model->zero_grad();
        loss.backward();

        // Update weights
        optimizer.step();

        std::cout << "Epoch " << epoch << ", Loss: "
                  << loss.tensor().item<float>() << std::endl;
    }

    // Save fine-tuned model
    model->save("fine_tuned_model.pt");

    return 0;
}
```

### Example 3: Batch Inference on GPU

```cpp
#include <tenzor/onnx/importer.hpp>

using namespace tenzor;

int main() {
    // Import model to GPU
    onnx::ONNXImporter importer(false);
    importer.set_device(Device::cuda(0));
    auto model = importer.import_from_file("model.onnx");

    // Set to eval mode
    model->eval();

    // Batch inference
    int batch_size = 64;
    Tensor batch_input({batch_size, 3, 224, 224}, DType::Float32, Device::cuda(0));

    // Fill with data
    batch_input.fill_(0.5f);

    // Run batch inference
    Variable batch_output = model->forward(Variable(batch_input));

    // Process results
    auto output_cpu = batch_output.tensor().cpu();
    std::cout << "Batch output shape: " << output_cpu.shape() << std::endl;

    return 0;
}
```

### Example 4: Import VGG-16

```cpp
#include <tenzor/onnx/importer.hpp>

using namespace tenzor;

int main() {
    // Import VGG-16
    auto vgg = onnx::import_onnx("vgg16.onnx", true);

    // Inspect model
    auto params = vgg->parameters();
    std::cout << "Total parameters: " << params.size() << std::endl;

    // Run inference
    Tensor input({1, 3, 224, 224}, DType::Float32, Device::cpu());
    Variable output = vgg->forward(Variable(input));

    return 0;
}
```

## Operator Mapping Reference

### Reverse Operator Mapping (ONNX → Tenzor)

| ONNX Operator | Tenzor Module/Function | Notes |
|---------------|------------------------|-------|
| Add | `ops::add(a, b)` | Element-wise addition |
| Sub | `ops::sub(a, b)` | Element-wise subtraction |
| Mul | `ops::mul(a, b)` | Element-wise multiplication |
| Div | `ops::div(a, b)` | Element-wise division |
| MatMul | `ops::matmul(a, b)` | Matrix multiplication |
| Gemm | `nn::Linear` | Linear layer (Y = X @ W^T + b) |
| Conv | `nn::Conv1d` / `nn::Conv2d` | Convolution layer |
| BatchNormalization | `nn::BatchNorm1d` / `nn::BatchNorm2d` | Batch normalization |
| Relu | `tensor.relu()` | ReLU activation |
| LeakyRelu | `tensor.leaky_relu(alpha)` | Leaky ReLU |
| Sigmoid | `tensor.sigmoid()` | Sigmoid activation |
| Tanh | `tensor.tanh()` | Tanh activation |
| Gelu | `tensor.gelu()` | GELU activation |
| Softmax | `tensor.softmax(axis)` | Softmax normalization |
| LogSoftmax | `tensor.log_softmax(axis)` | Log-Softmax |
| Elu | `tensor.elu(alpha)` | ELU activation |
| Selu | `tensor.selu()` | SELU activation |
| MaxPool | `nn::MaxPool2d` | Max pooling |
| AveragePool | `nn::AvgPool2d` | Average pooling |
| GlobalAveragePool | `nn::AdaptiveAvgPool2d` | Global average pooling |
| Reshape | `tensor.reshape(shape)` | Reshape tensor |
| Transpose | `tensor.permute(perm)` | Transpose/permute |
| Concat | `ops::cat(tensors, axis)` | Concatenation |
| Split | `tensor.split(sizes, axis)` | Split tensor |
| Flatten | `tensor.flatten(axis)` | Flatten dimensions |

## Attribute Handling

### Common Attributes

**Convolution (Conv):**
- `kernel_shape`: Kernel size
- `strides`: Convolution stride
- `pads`: Padding (format: [top, left, bottom, right])
- `dilations`: Dilation factor
- `group`: Number of groups for grouped convolution

**Pooling (MaxPool, AveragePool):**
- `kernel_shape`: Pooling kernel size
- `strides`: Pooling stride
- `pads`: Padding (format: [top, left, bottom, right])

**Gemm (Linear):**
- `alpha`: Scalar multiplier for A @ B
- `beta`: Scalar multiplier for C
- `transA`: Transpose A before multiplication
- `transB`: Transpose B before multiplication

**Batch Normalization:**
- `epsilon`: Small constant for numerical stability
- `momentum`: Momentum for running statistics

**Activations:**
- `alpha`: Parameter for LeakyReLU, ELU
- `axis`: Axis for Softmax, LogSoftmax

## Data Type Support

### Supported Data Types

| ONNX Data Type | Tenzor DType |
|----------------|--------------|
| FLOAT (1) | DType::Float32 |
| DOUBLE (11) | DType::Float64 |
| FLOAT16 (10) | DType::Float16 |
| BFLOAT16 (16) | DType::BFloat16 |
| INT8 (3) | DType::Int8 |
| INT16 (5) | DType::Int16 |
| INT32 (6) | DType::Int32 |
| INT64 (7) | DType::Int64 |
| UINT8 (2) | DType::UInt8 |
| BOOL (9) | DType::Bool |

## Error Handling

The importer provides detailed error messages for common issues:

### Unsupported Operator

```cpp
try {
    auto model = onnx::import_onnx("model.onnx");
} catch (const std::runtime_error& e) {
    // Error: "Unsupported ONNX operator: CustomOp"
    std::cerr << "Import failed: " << e.what() << std::endl;
}
```

### Missing Initializer

```cpp
try {
    auto model = onnx::import_onnx("model.onnx");
} catch (const std::runtime_error& e) {
    // Error: "Input tensor not found: weight_tensor"
    std::cerr << "Import failed: " << e.what() << std::endl;
}
```

### Invalid ONNX Version

```cpp
try {
    auto model = onnx::import_onnx("old_model.onnx", true);
    // Warning logged: "ONNX opset version 9 is older than recommended version 13"
} catch (const std::runtime_error& e) {
    std::cerr << "Import failed: " << e.what() << std::endl;
}
```

## Best Practices

### 1. Use Verbose Mode During Development

```cpp
// Enable verbose logging to understand import process
auto model = onnx::import_onnx("model.onnx", true);
```

### 2. Validate Model After Import

```cpp
auto model = onnx::import_onnx("model.onnx");
model->eval();

// Test with dummy input
Tensor dummy_input({1, 3, 224, 224}, DType::Float32, Device::cpu());
Variable output = model->forward(Variable(dummy_input));

// Verify output shape
assert(output.tensor().shape()[0] == 1);
assert(output.tensor().shape()[1] == 1000); // Expected output classes
```

### 3. Handle Device Placement

```cpp
// Option 1: Import directly to GPU
onnx::ONNXImporter importer;
importer.set_device(Device::cuda(0));
auto model = importer.import_from_file("model.onnx");

// Option 2: Import to CPU then move to GPU
auto model = onnx::import_onnx("model.onnx");
model->to(Device::cuda(0));
```

### 4. Set Evaluation Mode for Inference

```cpp
auto model = onnx::import_onnx("model.onnx");
model->eval(); // Disable dropout, fix batch norm statistics
```

## Limitations

### Current Limitations

1. **Dynamic Shapes**: Limited support for dynamic input shapes
2. **Control Flow**: Does not support If, Loop, Scan operators
3. **Custom Operators**: Does not support custom ONNX operators
4. **Quantized Models**: INT8/UINT8 quantization not fully supported

### Workarounds

**For dynamic batch sizes:**
```cpp
// Import with fixed batch size
auto model = onnx::import_onnx("model.onnx");

// Run with different batch sizes (if model supports it)
Tensor batch1({1, 3, 224, 224}, DType::Float32, Device::cpu());
Tensor batch8({8, 3, 224, 224}, DType::Float32, Device::cpu());

auto output1 = model->forward(Variable(batch1));
auto output8 = model->forward(Variable(batch8));
```

## Troubleshooting

### Issue: "Unsupported ONNX operator"

**Solution:** Check if the operator is in the supported list. If not, you may need to:
1. Simplify the ONNX model
2. Use a different opset version
3. Replace unsupported operators with supported equivalents

### Issue: Model import is slow

**Solution:**
1. Disable verbose logging: `import_onnx("model.onnx", false)`
2. Import directly to target device to avoid extra copies
3. Use `import_from_bytes()` to avoid repeated file I/O

### Issue: Incorrect output values

**Solution:**
1. Verify input preprocessing matches the original framework
2. Check that the model is in eval mode: `model->eval()`
3. Compare intermediate layer outputs with original model

## Advanced Topics

### Accessing Model Metadata

```cpp
onnx::ONNXImporter importer;
auto model = importer.import_from_file("model.onnx");

// Access ONNX model metadata
auto model_data = importer.get_model_data();
std::cout << "Producer: " << model_data.producer_name << std::endl;
std::cout << "IR Version: " << model_data.ir_version << std::endl;
std::cout << "Opset Version: " << model_data.opset_version << std::endl;
std::cout << "Model Version: " << model_data.model_version << std::endl;
```

### Inspecting Imported Layers

```cpp
auto model = onnx::import_onnx("model.onnx");

// Get all parameters
auto params = model->named_parameters();
for (const auto& [name, param] : params) {
    std::cout << "Parameter: " << name
              << ", Shape: " << param->tensor().shape()
              << ", DType: " << static_cast<int>(param->tensor().dtype())
              << std::endl;
}

// Get all buffers (e.g., BatchNorm running statistics)
auto buffers = model->named_buffers();
for (const auto& [name, buffer] : buffers) {
    std::cout << "Buffer: " << name
              << ", Shape: " << buffer->tensor().shape()
              << std::endl;
}
```

## Performance Considerations

### Memory Optimization

```cpp
// Import to CPU, process in batches, move to GPU as needed
auto model = onnx::import_onnx("large_model.onnx");
model->to(Device::cuda(0));

for (auto& batch : dataset) {
    Tensor batch_gpu = batch.to(Device::cuda(0));
    auto output = model->forward(Variable(batch_gpu));
    auto result = output.tensor().cpu(); // Move result back to CPU
    // Process result...
}
```

### Inference Optimization

```cpp
auto model = onnx::import_onnx("model.onnx");
model->eval(); // Disable training-specific layers

// For FP16 inference (if supported)
// model->to(DType::Float16);

// Warmup run
Tensor dummy({1, 3, 224, 224}, DType::Float32, Device::cuda(0));
model->forward(Variable(dummy));

// Actual inference
auto output = model->forward(Variable(input));
```

## References

- [ONNX Specification](https://github.com/onnx/onnx/blob/main/docs/IR.md)
- [ONNX Operators](https://github.com/onnx/onnx/blob/main/docs/Operators.md)
- [Tenzor Documentation](../README.md)

## License

The ONNX importer is part of the Tenzor library and follows the same license.
