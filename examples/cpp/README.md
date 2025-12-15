# C++ Examples

This directory contains C++ examples demonstrating Tenzor's capabilities.

## Directory Structure

```
cpp/
├── tutorials/          # Step-by-step learning tutorials
├── showcase/           # Feature demonstrations at 3 abstraction levels
├── nlp/                # Natural language processing examples
├── cv/                 # Computer vision examples
├── compression/        # Model compression (pruning, distillation)
├── quantization/       # INT8/FP16 quantization
├── zero/               # ZeRO optimizer sharding
├── autograd/           # Automatic differentiation examples
└── *.cpp               # Basic standalone examples
```

## Building

From the build directory:

```bash
cmake .. -DTENZOR_BUILD_EXAMPLES=ON
cmake --build . -j$(nproc)
```

Executables are placed in `bin/`.

## Basic Examples

### simple_example.cpp

Basic tensor operations:

```cpp
#include <tenzor/tenzor.hpp>

int main() {
    tenzor::initialize();

    auto a = tenzor::randn({3, 3});
    auto b = tenzor::randn({3, 3});
    auto c = tenzor::matmul(a, b);

    std::cout << "Result: " << c << std::endl;
    return 0;
}
```

Run: `./bin/simple_example`

### mnist_example.cpp

Train a simple neural network on MNIST:

```bash
./bin/mnist_example
```

### backend_example.cpp

Demonstrates multi-backend support:

```bash
./bin/backend_example
```

### serialization_example.cpp

Save and load models:

```bash
./bin/serialization_example
```

### tensorboard_example.cpp

Log training metrics to TensorBoard:

```bash
./bin/tensorboard_example
tensorboard --logdir=runs/
```

## Tutorials

Located in `tutorials/`. See [tutorials/README.md](tutorials/README.md).

| Tutorial | Description | Concepts |
|----------|-------------|----------|
| mnist_complete | Full MNIST training | Manual training loop |
| mnist_with_dataloader | DataLoader usage | Batching, callbacks |
| custom_training_loop | Advanced patterns | Validation, checkpoints |
| mixed_precision_training | FP16/BF16 training | AMP, GradScaler |

## Showcase

Located in `showcase/`. Each example shows three implementation styles:

1. **tensor_only.cpp** - Raw tensor math
2. **autograd.cpp** - With automatic differentiation
3. **neural_network.cpp** - High-level nn::Module

## NLP Examples

| Example | Model | Task |
|---------|-------|------|
| bert_example.cpp | BERT | Text classification |
| gpt_text_generation.cpp | GPT-2 | Text generation |

## Computer Vision

| Example | Models | Task |
|---------|--------|------|
| classic_models_example.cpp | ResNet, VGG, MobileNet | Image classification |

## Model Optimization

### Compression

- `pruning_example.cpp` - Structured/unstructured pruning
- `distillation_example.cpp` - Knowledge distillation
- `combined_compression.cpp` - Multiple techniques

### Quantization

- `quantize_resnet.cpp` - Post-training INT8 quantization

### ZeRO

Memory-efficient distributed training:
- Stage 1: Optimizer state partitioning
- Stage 2: Gradient partitioning
- Stage 3: Parameter partitioning

## Running with GPU

```bash
# CUDA
./bin/mnist_example  # Auto-detects CUDA

# Specific device
CUDA_VISIBLE_DEVICES=0 ./bin/mnist_example
```

## Adding New Examples

1. Create `.cpp` file in appropriate directory
2. Add to `CMakeLists.txt`:
   ```cmake
   add_executable(my_example cpp/my_example.cpp)
   target_link_libraries(my_example PRIVATE tenzor_core)
   ```
3. Document in this README
