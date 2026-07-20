# Getting Started with Tenzor

Welcome to Tenzor! This guide will help you get up and running quickly with the library.

## Table of Contents

- [Introduction](#introduction)
- [Your First Tensor](#your-first-tensor)
- [Basic Operations](#basic-operations)
- [Automatic Differentiation](#automatic-differentiation)
- [Building Neural Networks](#building-neural-networks)
- [Training a Model](#training-a-model)
- [GPU Acceleration](#gpu-acceleration)
- [Next Steps](#next-steps)

## Introduction

Tenzor is a high-performance tensor computation and deep learning library designed with a familiar PyTorch-like API. Whether you're coming from PyTorch, TensorFlow, or starting fresh, you'll find Tenzor intuitive and powerful.

### Key Concepts

- **Tensor**: Multi-dimensional array, the fundamental data structure
- **Variable**: Tensor wrapper that tracks gradients for automatic differentiation
- **Module**: Building block for neural networks (layers, models)
- **Device**: Execution target (CPU, CUDA, ROCm, etc.)

## Your First Tensor

### C++

```cpp
#include <tenzor/tenzor.hpp>

int main() {
    using namespace tenzor;

    // Initialize the library
    initialize();

    // Create tensors
    auto a = zeros({3, 3});           // 3x3 tensor of zeros
    auto b = ones({3, 3});            // 3x3 tensor of ones
    auto c = randn({3, 3});           // 3x3 tensor of random normal values
    auto d = arange(0, 10, 1);        // [0, 1, 2, ..., 9]

    // Create from data
    auto e = tensor({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2});

    // Print tensor info
    std::cout << "Shape: " << c.shape() << "\n";
    std::cout << "DType: " << c.dtype() << "\n";
    std::cout << "Device: " << c.device() << "\n";

    return 0;
}
```

### Python

```python
import tenzor as tz

# Create tensors
a = tz.zeros([3, 3])           # 3x3 tensor of zeros
b = tz.ones([3, 3])            # 3x3 tensor of ones
c = tz.randn([3, 3])           # 3x3 tensor of random normal values
d = tz.arange(0, 10, 1)        # [0, 1, 2, ..., 9]

# Create from list
e = tz.tensor([[1.0, 2.0], [3.0, 4.0]])

# Print tensor info
print(f"Shape: {c.shape}")
print(f"DType: {c.dtype}")
print(f"Device: {c.device}")
```

### Data Types

Tenzor supports various data types:

| Type | C++ | Python | Description |
|------|-----|--------|-------------|
| Float32 | `DType::Float32` | `tz.dtype.float32` | 32-bit floating point (default) |
| Float64 | `DType::Float64` | `tz.dtype.float64` | 64-bit floating point |
| Float16 | `DType::Float16` | `tz.dtype.float16` | 16-bit floating point |
| BFloat16 | `DType::BFloat16` | `tz.dtype.bfloat16` | Brain floating point |
| Int32 | `DType::Int32` | `tz.dtype.int32` | 32-bit integer |
| Int64 | `DType::Int64` | `tz.dtype.int64` | 64-bit integer |
| Bool | `DType::Bool` | `tz.dtype.bool` | Boolean |

```cpp
// C++: Specify data type
auto x = zeros({3, 3}, DType::Float64);
auto y = ones({3, 3}, DType::Int32);
```

```python
# Python: Specify data type
x = tz.zeros([3, 3], dtype=tz.dtype.float64)
y = tz.ones([3, 3], dtype=tz.dtype.int32)
```

## Basic Operations

### Arithmetic Operations

```cpp
// C++
auto a = randn({3, 3});
auto b = randn({3, 3});

auto c = a + b;           // Element-wise addition
auto d = a - b;           // Element-wise subtraction
auto e = a * b;           // Element-wise multiplication
auto f = a / b;           // Element-wise division
auto g = pow(a, 2);       // Element-wise power
auto h = sqrt(a.abs());   // Element-wise square root
```

```python
# Python
a = tz.randn([3, 3])
b = tz.randn([3, 3])

c = a + b           # Element-wise addition
d = a - b           # Element-wise subtraction
e = a * b           # Element-wise multiplication
f = a / b           # Element-wise division
g = a ** 2          # Element-wise power
h = tz.sqrt(tz.abs(a))  # Element-wise square root
```

### Matrix Operations

```cpp
// C++
auto a = randn({3, 4});
auto b = randn({4, 5});

auto c = matmul(a, b);    // Matrix multiplication: [3, 5]
auto d = a.t();           // Transpose: [4, 3]
auto e = bmm(batch_a, batch_b);  // Batched matrix multiply
```

```python
# Python
a = tz.randn([3, 4])
b = tz.randn([4, 5])

c = tz.matmul(a, b)    # Matrix multiplication: [3, 5]
d = a.t()              # Transpose: [4, 3]
e = tz.bmm(batch_a, batch_b)  # Batched matrix multiply
```

### Reduction Operations

```cpp
// C++
auto x = randn({3, 4});

auto s = x.sum();           // Sum all elements
auto m = x.mean();          // Mean of all elements
auto row_sum = x.sum(1);    // Sum along axis 1
auto col_max = x.max(0);    // Max along axis 0
auto idx = x.argmax(1);     // Argmax along axis 1
```

```python
# Python
x = tz.randn([3, 4])

s = x.sum()           # Sum all elements
m = x.mean()          # Mean of all elements
row_sum = x.sum(dim=1)    # Sum along axis 1
col_max = x.max(dim=0)    # Max along axis 0
idx = x.argmax(dim=1)     # Argmax along axis 1
```

### Shape Operations

```cpp
// C++
auto x = randn({2, 3, 4});

auto a = x.reshape({6, 4});       // Reshape to [6, 4]
auto b = x.view({-1, 4});         // View with inferred dimension
auto c = x.transpose(0, 2);       // Swap axes 0 and 2
auto d = x.permute({2, 0, 1});    // Reorder all axes
auto e = x.squeeze();             // Remove size-1 dimensions
auto f = x.unsqueeze(0);          // Add dimension at position 0
```

```python
# Python
x = tz.randn([2, 3, 4])

a = x.reshape([6, 4])       # Reshape to [6, 4]
b = x.view([-1, 4])         # View with inferred dimension
c = x.transpose(0, 2)       # Swap axes 0 and 2
d = x.permute([2, 0, 1])    # Reorder all axes
e = x.squeeze()             # Remove size-1 dimensions
f = x.unsqueeze(0)          # Add dimension at position 0
```

## Automatic Differentiation

Tenzor includes a powerful automatic differentiation engine for computing gradients.

### C++

```cpp
#include <tenzor/tenzor.hpp>

int main() {
    using namespace tenzor;
    initialize();

    // Create variables with gradient tracking
    auto x = Variable(randn({3, 3}), true);  // requires_grad=true
    auto w = Variable(randn({3, 3}), true);

    // Forward pass
    auto y = matmul(x, w);
    auto loss = y.sum();

    // Backward pass
    loss.backward();

    // Access gradients
    std::cout << "x.grad: " << x.grad() << "\n";
    std::cout << "w.grad: " << w.grad() << "\n";

    return 0;
}
```

### Python

```python
import tenzor as tz

# Create variables with gradient tracking
x = tz.Variable(tz.randn([3, 3]), requires_grad=True)
w = tz.Variable(tz.randn([3, 3]), requires_grad=True)

# Forward pass
y = tz.matmul(x, w)
loss = y.sum()

# Backward pass
loss.backward()

# Access gradients
print(f"x.grad: {x.grad}")
print(f"w.grad: {w.grad}")
```

### Gradient Control

```cpp
// C++
{
    auto guard = no_grad();  // Disable gradient tracking in scope
    auto y = model.forward(x);  // No gradients computed
}

// Detach from computation graph
auto detached = x.detach();
```

```python
# Python
with tz.no_grad():
    y = model(x)  # No gradients computed

# Detach from computation graph
detached = x.detach()
```

## Building Neural Networks

### Using nn::Module

```cpp
// C++
#include <tenzor/tenzor.hpp>

class MLP : public nn::Module {
public:
    MLP(int64_t input_size, int64_t hidden_size, int64_t output_size)
        : fc1(std::make_shared<nn::Linear>(input_size, hidden_size))
        , fc2(std::make_shared<nn::Linear>(hidden_size, output_size))
        , relu(std::make_shared<nn::ReLU>())
    {
        register_module("fc1", fc1);
        register_module("fc2", fc2);
    }

    // Override forward_impl, not forward — Module::forward is a non-virtual
    // dispatcher that calls forward_impl() internally.
    auto forward_impl(const Variable& x) -> Variable override {
        auto h = relu->forward(fc1->forward(x));
        return fc2->forward(h);
    }

private:
    std::shared_ptr<nn::Linear> fc1, fc2;
    std::shared_ptr<nn::ReLU> relu;
};

int main() {
    auto model = std::make_shared<MLP>(784, 128, 10);

    // Forward pass
    auto x = Variable(randn({32, 784}), true);
    auto output = model->forward(x);

    std::cout << "Output shape: " << output.data().shape() << "\n";
    return 0;
}
```

### Python

```python
import tenzor as tz

class MLP(tz.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = tz.nn.Linear(input_size, hidden_size)
        self.fc2 = tz.nn.Linear(hidden_size, output_size)
        self.relu = tz.nn.ReLU()

    def forward(self, x):
        h = self.relu(self.fc1(x))
        return self.fc2(h)

model = MLP(784, 128, 10)

# Forward pass
x = tz.Variable(tz.randn([32, 784]), requires_grad=True)
output = model(x)

print(f"Output shape: {output.shape}")
```

### Using nn::Sequential

```cpp
// C++
auto model = nn::Sequential(
    std::make_shared<nn::Linear>(784, 256),
    std::make_shared<nn::ReLU>(),
    std::make_shared<nn::Dropout>(0.5),
    std::make_shared<nn::Linear>(256, 128),
    std::make_shared<nn::ReLU>(),
    std::make_shared<nn::Linear>(128, 10)
);
```

```python
# Python
model = tz.nn.Sequential(
    tz.nn.Linear(784, 256),
    tz.nn.ReLU(),
    tz.nn.Dropout(0.5),
    tz.nn.Linear(256, 128),
    tz.nn.ReLU(),
    tz.nn.Linear(128, 10)
)
```

### Available Layers

| Layer | Description |
|-------|-------------|
| `nn::Linear` | Fully connected layer |
| `nn::Conv1d/2d/3d` | Convolutional layers |
| `nn::BatchNorm1d/2d` | Batch normalization |
| `nn::LayerNorm` | Layer normalization |
| `nn::Dropout` | Dropout regularization |
| `nn::ReLU/GELU/SiLU` | Activation functions |
| `nn::LSTM/GRU` | Recurrent layers |
| `nn::MultiheadAttention` | Transformer attention |
| `nn::Embedding` | Embedding layer |

## Training a Model

### Complete Training Loop

```cpp
// C++ Training Example
#include <tenzor/tenzor.hpp>

int main() {
    using namespace tenzor;
    initialize();

    // Model
    auto model = nn::Sequential(
        std::make_shared<nn::Linear>(784, 128),
        std::make_shared<nn::ReLU>(),
        std::make_shared<nn::Linear>(128, 10)
    );

    // Optimizer
    auto optimizer = optim::Adam(model->parameters(), 1e-3);

    // Training loop
    for (int epoch = 0; epoch < 10; ++epoch) {
        float total_loss = 0.0f;

        for (auto& [images, labels] : train_loader) {
            // Forward pass
            auto x = Variable(images.reshape({-1, 784}), true);
            auto output = model->forward(x);
            auto loss = nn::cross_entropy(output, labels);

            // Backward pass
            optimizer.zero_grad();
            loss.backward();
            optimizer.step();

            total_loss += loss.data().item<float>();
        }

        std::cout << "Epoch " << epoch << " Loss: " << total_loss << "\n";
    }

    return 0;
}
```

### Python Training

```python
import tenzor as tz

# Model
model = tz.nn.Sequential(
    tz.nn.Linear(784, 128),
    tz.nn.ReLU(),
    tz.nn.Linear(128, 10)
)

# Optimizer
optimizer = tz.optim.Adam(model.parameters(), lr=1e-3)

# Training loop
for epoch in range(10):
    total_loss = 0.0

    for images, labels in train_loader:
        # Forward pass
        x = tz.Variable(images.reshape([-1, 784]), requires_grad=True)
        output = model(x)
        loss = tz.nn.cross_entropy(output, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch} Loss: {total_loss:.4f}")
```

### Loss Functions

```cpp
// C++
auto mse = nn::mse_loss(predictions, targets);
auto ce = nn::cross_entropy(logits, labels);
auto bce = nn::binary_cross_entropy_with_logits(logits, targets);
auto nll = nn::nll_loss(log_probs, labels);
```

```python
# Python
mse = tz.nn.mse_loss(predictions, targets)
ce = tz.nn.cross_entropy(logits, labels)
bce = tz.nn.functional.binary_cross_entropy_with_logits(logits, targets)
nll = tz.nn.nll_loss(log_probs, labels)
```

### Optimizers

```python
# Available optimizers
optimizer = tz.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
optimizer = tz.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999))
optimizer = tz.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
optimizer = tz.optim.RMSprop(model.parameters(), lr=1e-3)
```

## GPU Acceleration

### Moving to GPU

```cpp
// C++
auto x = randn({3, 3});
auto x_gpu = x.cuda();      // Move to GPU (device 0)
auto x_gpu1 = x.cuda(1);    // Move to GPU device 1
auto x_cpu = x_gpu.cpu();   // Move back to CPU

// Create directly on GPU
auto y = randn({3, 3}, DType::Float32, Device::cuda(0));

// Move model to GPU
model->cuda();
```

```python
# Python
x = tz.randn([3, 3])
x_gpu = x.cuda()      # Move to GPU (device 0)
x_gpu1 = x.cuda(1)    # Move to GPU device 1
x_cpu = x_gpu.cpu()   # Move back to CPU

# Create directly on GPU
y = tz.randn([3, 3], device=tz.Device.cuda(0))

# Move model to GPU
model.cuda()
```

### Check GPU Availability

```cpp
// C++
auto* cuda_backend = tenzor::backend_registry().get_backend("cuda");
if (cuda_backend != nullptr && cuda_backend->is_available()) {
    std::cout << "CUDA devices: " << cuda_backend->device_count() << "\n";
}
```

```python
# Python
if tz.cuda_is_available():
    print(f"CUDA devices: {tz.cuda_device_count()}")
```

### Mixed Precision Training

```cpp
// C++
auto scaler = nn::amp::GradScaler();

for (auto& batch : loader) {
    optimizer.zero_grad();

    // Automatic mixed precision
    {
        auto autocast = nn::amp::Autocast(true);
        auto output = model->forward(input);
        auto loss = criterion(output, target);
    }

    // Scale loss and backward
    scaler.scale(loss).backward();
    scaler.step(optimizer);
    scaler.update();
}
```

```python
# Python
scaler = tz.amp.GradScaler()

for batch in loader:
    optimizer.zero_grad()

    # Automatic mixed precision
    with tz.amp.autocast():
        output = model(input)
        loss = criterion(output, target)

    # Scale loss and backward
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

## Next Steps

Now that you understand the basics, explore these resources:

### Tutorials
- [MNIST Classification](../examples/cpp/tutorials/mnist_complete.cpp) - Complete training example
- [Custom Training Loop](../examples/cpp/tutorials/custom_training_loop.cpp) - Advanced patterns
- [Python Examples](../examples/python/) - Comprehensive Python tutorials

### Advanced Topics
- [Model Serialization](../examples/cpp/serialization_example.cpp) - Save and load models
- [Data Loading](../examples/cpp/tutorials/mnist_with_dataloader.cpp) - DataLoader and callbacks
- [Distributed Training](../examples/cpp/zero/) - ZeRO sharding and multi-GPU training
- [Model Quantization](../examples/cpp/quantization/) - Model optimization

### API Reference
- C++ API Documentation - generate locally with `doxygen Doxyfile` from the project root, then open `docs/api/html/index.html` (not pre-built or committed in this repo)

### Community
- [GitHub Issues](https://github.com/skreamz/Tenzor/issues) - Report bugs, request features
- [Discussions](https://github.com/skreamz/Tenzor/discussions) - Ask questions, share projects
- [Contributing](../CONTRIBUTING.md) - Contribute to Tenzor

---

**Happy coding with Tenzor!**
