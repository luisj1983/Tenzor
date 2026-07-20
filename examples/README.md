# Tenzor Examples

This directory contains example code demonstrating how to use Tenzor for various machine learning tasks.

## Directory Structure

```
examples/
├── cpp/                    # C++ examples
│   ├── tutorials/          # Step-by-step tutorials
│   ├── showcase/           # Feature demonstrations (3 abstraction levels)
│   ├── nlp/                # Natural language processing
│   ├── cv/                 # Computer vision
│   ├── compression/        # Model compression techniques
│   ├── quantization/       # Model quantization
│   ├── zero/               # ZeRO optimization examples
│   ├── autograd/           # Automatic differentiation
│   └── *.cpp               # Basic examples
│
└── python/                 # Python examples
    ├── 01-10_*.py          # Progressive tutorials
    ├── onnx_*.py           # ONNX import/export
    ├── checkpoint_*.py     # Model checkpointing
    └── README.md           # Python examples guide
```

## Quick Start

### C++ Examples

Build the examples with CMake:

```bash
cd Tenzor/build
cmake .. -DTENZOR_BUILD_EXAMPLES=ON
cmake --build .

# Run an example
./bin/simple_example
./bin/mnist_example
```

### Python Examples

```bash
cd Tenzor/examples/python

# Basic tensor operations
python 01_tensor_basics.py

# Training a model
python 04_mnist_mlp.py
```

## C++ Examples Overview

### Basic Examples

| Example | Description |
|---------|-------------|
| `simple_example.cpp` | Basic tensor creation and operations |
| `mnist_example.cpp` | MNIST digit classification |
| `backend_example.cpp` | Multi-backend usage (CPU, CUDA, etc.) |
| `serialization_example.cpp` | Save and load models |
| `tensorboard_example.cpp` | TensorBoard logging integration |
| `custom_op_example.cpp` | Creating custom operations |
| `callback_demo.cpp` | Training callback hooks |
| `demo_split_operation.cpp` | Tensor split operation demo |
| `quantization_example.cpp` | INT8 quantization walkthrough |

### Tutorials (`cpp/tutorials/`)

Progressive tutorials for learning Tenzor:

1. **mnist_complete.cpp** - Complete MNIST training with manual loop
2. **mnist_with_dataloader.cpp** - Using DataLoader and callbacks
3. **custom_training_loop.cpp** - Advanced training patterns
4. **mixed_precision_training.cpp** - FP16/BF16 mixed precision

### Showcase (`cpp/showcase/`)

Each showcase demonstrates the same task at three abstraction levels:
- **Tensor-only**: Raw tensor operations
- **Autograd**: Using automatic differentiation
- **Neural Network**: High-level nn::Module API

| Showcase | Description |
|----------|-------------|
| `01_xor/` | XOR problem (hello world of neural networks) |
| `02_linear_regression/` | Simple regression |
| `03_binary_classification/` | Binary classification |
| `04_multiclass_classification/` | Multi-class classification |
| `05_convolutional/` | Convolutional neural networks |
| `06_autoencoder/` | Autoencoder for dimensionality reduction |
| `07_rnn_sequence/` | Recurrent neural networks |
| `08_batch_normalization/` | Batch normalization |
| `09_dropout_regularization/` | Dropout for regularization |
| `10_custom_loss/` | Custom loss functions |
| `11_chat_ai/` | Simple chatbot |
| `12_residual_network/` | Residual (skip) connections |
| `13_variational_autoencoder/` | Variational autoencoder (VAE) |
| `14_gan/` | Generative adversarial network |
| `15_lstm_text/` | LSTM text generation |
| `16_self_attention/` | Self-attention mechanism |
| `17_word_embedding/` | Word embeddings |
| `18_transfer_learning/` | Transfer learning / fine-tuning |
| `19_layer_normalization/` | Layer normalization |
| `20_multitask_learning/` | Multi-task learning |
| `21_siamese_network/` | Siamese network |
| `22_hierarchical_reasoning/` | Hierarchical Reasoning Model (HRM) |

Each showcase example doubles as a regression test: `autograd_runner.{cpp,hpp}`
wires the showcase's forward/backward pass into ctest and asserts the loss
decreases end-to-end, so a showcase directory going stale or its loss
plateauing fails CI, not just a docs check.

### NLP Examples (`cpp/nlp/`)

| Example | Description |
|---------|-------------|
| `bert_example.cpp` | BERT text classification |
| `gpt_text_generation.cpp` | GPT-2 text generation |

### Computer Vision (`cpp/cv/`)

| Example | Description |
|---------|-------------|
| `classic_models_example.cpp` | ResNet, VGG, MobileNet |
| `resnet_example.cpp` | ResNet image classification |

### Model Optimization

**Compression (`cpp/compression/`):**
- `pruning_example.cpp` - Weight pruning
- `distillation_example.cpp` - Knowledge distillation
- `combined_compression.cpp` - Combined techniques

**Quantization (`cpp/quantization/`):**
- `quantize_resnet.cpp` - INT8 quantization

**ZeRO Optimization (`cpp/zero/`):**
- Stage 1, 2, 3 optimizer sharding
- CPU offloading
- Distributed training

## Python Examples Overview

See [python/README.md](python/README.md) for the complete Python examples guide.

### Learning Path

| Level | Examples | Topics |
|-------|----------|--------|
| Beginner | 01-03 | Tensors, autograd, linear regression |
| Intermediate | 04-06 | MLPs, CNNs, custom layers |
| Advanced | 07-10 | ResNet, RNNs, distributed training |

### Quick Examples

```python
import tenzor as tz

# Create and train a simple model
model = tz.nn.Sequential(
    tz.nn.Linear(784, 128),
    tz.nn.ReLU(),
    tz.nn.Linear(128, 10)
)

optimizer = tz.optim.Adam(model.parameters(), lr=1e-3)

# Training loop
for epoch in range(10):
    output = model(x)
    loss = tz.nn.cross_entropy(output, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

## Building Examples

### All Examples

```bash
cmake .. -DTENZOR_BUILD_EXAMPLES=ON
cmake --build . --target all
```

### Specific Example

```bash
cmake --build . --target mnist_example
```

### With GPU Support

```bash
cmake .. -DTENZOR_BUILD_EXAMPLES=ON -DTENZOR_BUILD_CUDA=ON
cmake --build .
```

## Requirements

- Tenzor library (built and installed)
- C++23 compatible compiler
- Python 3.9 - 3.13 (for Python examples; matches `requires-python` in `pyproject.toml`)
- CUDA 12.0+ (for GPU examples, optional)

## Contributing

We welcome new examples! Please follow these guidelines:

1. Place C++ examples in `cpp/` with appropriate subdirectory
2. Place Python examples in `python/`
3. Include comments explaining key concepts
4. Add entry to this README
5. Test on both CPU and GPU (if applicable)

## License

All examples are released under the MIT License.
