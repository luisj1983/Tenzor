# Tenzor Python Examples

This directory contains comprehensive tutorial examples demonstrating the Tenzor library's capabilities for deep learning and tensor computation.

## Getting Started

1. **Build and install Tenzor** (from project root):
   ```bash
   mkdir build && cd build
   cmake ..
   make
   make install
   ```

2. **Run any example**:
   ```bash
   python examples/python/01_tensor_basics.py
   ```

## Examples Overview

### 01_tensor_basics.py (Beginner)
**Duration**: ~30 seconds
**Topics**: Tensor creation, device management, basic operations

Learn the fundamentals of working with tensors in Tenzor:
- Creating tensors (zeros, ones, randn)
- Managing CPU/CUDA devices
- Element-wise operations (+, -, *)
- Matrix multiplication
- Shape manipulation (reshape)
- Different data types (float32, float64, int32)

**Run**:
```bash
python examples/python/01_tensor_basics.py
```

---

### 02_autograd_basics.py (Beginner)
**Duration**: ~45 seconds
**Topics**: Automatic differentiation, gradient computation

Master Tenzor's automatic differentiation system:
- Creating Variables with gradient tracking
- Building computation graphs
- Forward and backward passes
- Gradient accumulation
- Computing derivatives automatically
- Matrix operations with gradients

**Run**:
```bash
python examples/python/02_autograd_basics.py
```

---

### 03_linear_regression.py (Intermediate)
**Duration**: ~1 minute
**Topics**: Linear models, SGD optimization, training loops

Build and train a simple linear regression model:
- Generating synthetic training data
- Defining a linear model (y = wx + b)
- Mean Squared Error (MSE) loss
- Stochastic Gradient Descent (SGD) optimizer
- Complete training loop (100 epochs)
- Model evaluation and prediction

**Run**:
```bash
python examples/python/03_linear_regression.py
```

**Mathematical Background**:
- Model: `y = wx + b`
- Loss: `MSE = mean((y_pred - y_true)²)`
- Update: `w ← w - lr * ∂L/∂w`

---

### 04_mnist_mlp.py (Intermediate)
**Duration**: ~1-2 minutes
**Topics**: Classification, Multi-Layer Perceptrons, Adam optimizer

Train a neural network for digit classification:
- MNIST-like dataset (synthetic for demo)
- 2-layer MLP architecture (784 → 256 → 10)
- ReLU activation function
- Cross-Entropy loss for classification
- Adam optimizer with adaptive learning rates
- Mini-batch training
- Model evaluation and accuracy

**Run**:
```bash
python examples/python/04_mnist_mlp.py
```

**Architecture**:
```
Input (784 pixels)
    ↓
Linear(784 → 256)
    ↓
ReLU
    ↓
Linear(256 → 10)
    ↓
Softmax → Class probabilities
```

---

### 05_cnn_classification.py (Advanced)
**Duration**: ~1-2 minutes
**Topics**: Convolutional Neural Networks, image processing

Build a CNN for image classification:
- Understanding convolutions and pooling
- CNN architecture design
- Convolutional layers for feature extraction
- Max pooling for dimension reduction
- Dropout for regularization
- Model checkpointing
- Per-class performance metrics

**Run**:
```bash
python examples/python/05_cnn_classification.py
```

**Architecture**:
```
Input (1×28×28)
    ↓
Conv2d(1→32, 3×3) + ReLU
    ↓
MaxPool2d(2×2) → (32×14×14)
    ↓
Conv2d(32→64, 3×3) + ReLU
    ↓
MaxPool2d(2×2) → (64×7×7)
    ↓
Flatten → 3136
    ↓
Linear(3136→128) + ReLU
    ↓
Dropout(0.5)
    ↓
Linear(128→10)
```

---

### 06_custom_layer.py (Advanced)
**Duration**: ~45 seconds
**Topics**: Extending Tenzor, custom components

Learn to create custom neural network components:
- Understanding the Module system
- Building custom linear layers
- Implementing custom activation functions (LeakyReLU)
- Creating composite layers (Residual blocks)
- Defining custom loss functions (Huber loss)
- Combining components into networks
- Best practices for layer design

**Run**:
```bash
python examples/python/06_custom_layer.py
```

**Custom Components**:
- LeakyReLU: `f(x) = max(αx, x)`
- Huber Loss: Robust to outliers
- Residual Block: Skip connections for deep networks

---

### 07_resnet_cifar10.py (Expert)
**Duration**: ~5-10 minutes
**Topics**: ResNet architecture, modern training techniques

Advanced CNN training with state-of-the-art practices:
- ResNet-18 architecture with skip connections
- CIFAR-10 dataset (10 classes of 32x32 color images)
- Batch normalization for training stability
- Data augmentation (flips, crops)
- Learning rate scheduling
- Mixed precision training
- Model checkpointing
- Comprehensive evaluation metrics

**Run**:
```bash
python examples/python/07_resnet_cifar10.py
```

**Architecture Highlights**:
- Residual blocks with skip connections
- Deeper networks (18 layers) without vanishing gradients
- Adaptive average pooling
- SGD with momentum and weight decay

---

### 08_fashion_mnist_cnn.py (Advanced - Production Ready)
**Duration**: ~3-5 minutes
**Topics**: Complete training pipeline, best practices

A comprehensive, production-ready example for Fashion-MNIST classification:
- Modern 5-layer CNN architecture
- Fashion-MNIST dataset (70,000 images of 10 clothing categories)
- Data normalization and augmentation
- Batch normalization + dropout regularization
- Detailed progress tracking and metrics
- Per-class accuracy evaluation
- Model checkpointing (save/load)
- Educational comments throughout

**Run**:
```bash
python examples/python/08_fashion_mnist_cnn.py
```

**Architecture**:
```
Input (1×28×28)
    ↓
[Block 1: 2× Conv2d(3×3) + BN + ReLU + MaxPool + Dropout]
    → (32×14×14)
    ↓
[Block 2: 2× Conv2d(3×3) + BN + ReLU + MaxPool + Dropout]
    → (64×7×7)
    ↓
[Block 3: Conv2d(3×3) + BN + ReLU + MaxPool + Dropout]
    → (128×3×3)
    ↓
Flatten → 1152 features
    ↓
Linear(1152→256) + BN + ReLU + Dropout(0.5)
    ↓
Linear(256→10)
```

**Key Features**:
- **Data Augmentation**: Random flips, shifts, noise
- **Normalization**: Zero mean, unit variance preprocessing
- **Regularization**: Batch norm + dropout (0.25 conv, 0.5 FC)
- **Progress Tracking**: Real-time loss, accuracy, speed metrics
- **Checkpointing**: Save best model and periodic snapshots
- **Per-Class Metrics**: Detailed accuracy for each clothing category
- **Best Practices**: Production-ready code structure

**Fashion-MNIST Classes**:
0. T-shirt/top, 1. Trouser, 2. Pullover, 3. Dress, 4. Coat
5. Sandal, 6. Shirt, 7. Sneaker, 8. Bag, 9. Ankle boot

**What You'll Learn**:
- Complete deep learning training pipeline
- Data preprocessing and augmentation strategies
- Modern CNN architecture design principles
- Training loop with comprehensive monitoring
- Model evaluation with detailed statistics
- Checkpoint management for model persistence
- Production-ready code organization

---

## Learning Path

### For Beginners:
1. Start with `01_tensor_basics.py` to understand tensors
2. Move to `02_autograd_basics.py` to learn about gradients
3. Try `03_linear_regression.py` for your first model

### For Intermediate Users:
4. Explore `04_mnist_mlp.py` to build a real classifier
5. Study `05_cnn_classification.py` to understand CNNs
6. Dive into `06_custom_layer.py` to extend Tenzor

### For Advanced Users:
7. Master `08_fashion_mnist_cnn.py` for production-ready training pipelines
8. Challenge yourself with `07_resnet_cifar10.py` for state-of-the-art architectures

## Key Concepts Covered

- **Tensors**: Multi-dimensional arrays for computation
- **Autograd**: Automatic differentiation for gradient computation
- **Modules**: Building blocks of neural networks
- **Optimizers**: Algorithms for updating parameters (SGD, Adam)
- **Loss Functions**: Measuring prediction error (MSE, Cross-Entropy)
- **Layers**: Linear, Convolutional, Pooling, Dropout
- **Activations**: ReLU, LeakyReLU, Softmax
- **Training Loop**: Forward → Loss → Backward → Update

## Technical Requirements

- Python 3.7+
- NumPy (for data generation in examples)
- Tenzor library (built from source)
- Optional: CUDA for GPU acceleration

## Notes

- **Synthetic Data**: Examples use synthetic data for demonstration. For real applications, load actual datasets.
- **Simplified Implementations**: Some functions are simplified to focus on Tenzor API usage. Production code would be more robust.
- **GPU Support**: Examples automatically detect and use CUDA if available, falling back to CPU otherwise.
- **Timing**: All examples run in < 2 minutes for quick experimentation.

## Troubleshooting

**Import Error**: Ensure Tenzor is built and installed:
```bash
cd build && make install
export PYTHONPATH=/path/to/tenzor/build/python:$PYTHONPATH
```

**CUDA Not Found**: Examples work on CPU if CUDA is unavailable. No changes needed.

**Slow Execution**: Reduce `n_epochs` or `batch_size` in training examples for faster runs.

## Further Resources

- **Tenzor Documentation**: See `/docs` directory
- **C++ API**: Check `/include/tenzor` for header files
- **Tests**: See `/tests` for unit test examples
- **GitHub**: Report issues or contribute at [repository URL]

## Contributing

Found a bug or want to add an example? Contributions are welcome! Please ensure:
- Examples are self-contained and runnable
- Code is well-commented
- Execution time is < 2 minutes
- Follow existing style and structure

---

**Happy Learning with Tenzor!** 🚀
