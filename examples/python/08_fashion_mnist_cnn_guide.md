# Fashion-MNIST CNN Example - Quick Reference Guide

## Overview

**File**: `08_fashion_mnist_cnn.py`
**Lines of Code**: 861
**Difficulty**: Advanced (Production-Ready)
**Duration**: 3-5 minutes
**Dataset**: Fashion-MNIST (70,000 grayscale 28×28 images)

## Purpose

This example demonstrates a complete, production-ready deep learning pipeline for image classification. It's designed to be both educational and practical, showing best practices for real-world CNN training.

## Architecture Summary

```
Input: 1×28×28 grayscale images

Block 1 (Edge Detection):
  ├─ Conv2d(1→32, 3×3) + BatchNorm + ReLU
  ├─ Conv2d(32→32, 3×3) + BatchNorm + ReLU
  ├─ MaxPool2d(2×2) → 32×14×14
  └─ Dropout(0.25)

Block 2 (Texture Features):
  ├─ Conv2d(32→64, 3×3) + BatchNorm + ReLU
  ├─ Conv2d(64→64, 3×3) + BatchNorm + ReLU
  ├─ MaxPool2d(2×2) → 64×7×7
  └─ Dropout(0.25)

Block 3 (High-Level Features):
  ├─ Conv2d(64→128, 3×3) + BatchNorm + ReLU
  ├─ MaxPool2d(2×2) → 128×3×3
  └─ Dropout(0.25)

Classifier:
  ├─ Flatten → 1152 features
  ├─ Linear(1152→256) + BatchNorm + ReLU
  ├─ Dropout(0.5)
  └─ Linear(256→10) → Class logits
```

**Total Layers**: 5 convolutional + 2 fully connected = 7 layers

## Key Features

### 1. Data Pipeline
- **Loading**: Synthetic data (replace with real Fashion-MNIST)
- **Normalization**: Zero mean, unit variance (μ=0.286, σ=0.353)
- **Augmentation**: Random flips, shifts (±2px), Gaussian noise
- **Batch Size**: 128 samples

### 2. Architecture Design
- **Modern CNN**: Multiple conv layers per block
- **Batch Normalization**: After every conv and FC layer
- **Dropout**: 0.25 for conv layers, 0.5 for FC layers
- **Activation**: ReLU for non-linearity
- **Pooling**: 2×2 max pooling for downsampling

### 3. Training Strategy
- **Optimizer**: Adam (lr=0.001, β₁=0.9, β₂=0.999)
- **Weight Decay**: 1e-4 (L2 regularization)
- **Epochs**: 20 (configurable)
- **Loss**: Cross-entropy loss

### 4. Monitoring & Evaluation
- **Real-time Progress**: Loss, accuracy, speed (batches/sec)
- **Per-class Metrics**: Accuracy for each clothing category
- **Checkpointing**: Save best model and periodic snapshots
- **Training History**: Track loss and accuracy over time

## Function Reference

### Data Functions
```python
download_fashion_mnist()         # Load dataset (60k train, 10k test)
normalize_images(images)         # Normalize to μ=0, σ=1
augment_batch(images, training)  # Apply data augmentation
```

### Model Class
```python
class FashionMNISTCNN:
    __init__()                           # Initialize layers
    forward(x, training)                 # Forward pass
    parameters()                         # Get trainable parameters
    train()                              # Set to training mode
    eval()                               # Set to evaluation mode
    _count_parameters()                  # Count model parameters
```

### Training Functions
```python
compute_accuracy(preds, labels)          # Calculate accuracy %
train_epoch(model, optimizer, data, ep)  # Train one epoch
evaluate(model, data, class_names)       # Evaluate on test set
compute_cross_entropy_loss(logits, y)    # Compute loss
```

### Checkpointing
```python
save_checkpoint(model, opt, ep, ...)     # Save model state
load_checkpoint(model, path)             # Load model state
```

### Main
```python
main()  # Complete training pipeline
```

## Configuration

Located in `Config` class:

```python
# Data
NUM_CLASSES = 10          # Fashion categories
IMAGE_SIZE = 28           # 28×28 pixels
CHANNELS = 1              # Grayscale

# Training
BATCH_SIZE = 128          # Samples per batch
EPOCHS = 20               # Training iterations
LEARNING_RATE = 0.001     # Adam learning rate

# Regularization
DROPOUT_RATE = 0.5        # FC dropout rate
WEIGHT_DECAY = 1e-4       # L2 regularization

# Checkpointing
CHECKPOINT_DIR = "..."    # Save location
MODEL_NAME = "..."        # Model filename
```

## Fashion-MNIST Classes

| ID | Category | Description |
|----|----------|-------------|
| 0  | T-shirt/top | Simple t-shirts |
| 1  | Trouser | Pants and trousers |
| 2  | Pullover | Sweaters and pullovers |
| 3  | Dress | Various dress styles |
| 4  | Coat | Jackets and coats |
| 5  | Sandal | Open footwear |
| 6  | Shirt | Button-up shirts |
| 7  | Sneaker | Athletic shoes |
| 8  | Bag | Handbags and backpacks |
| 9  | Ankle boot | Short boots |

## Expected Results

With proper training on real Fashion-MNIST data:
- **Training Accuracy**: 95-98%
- **Test Accuracy**: 91-93%
- **Training Time**: ~3-5 minutes (CPU), ~1 minute (GPU)
- **Model Size**: ~1-2 MB

## Usage Examples

### Basic Training
```bash
python 08_fashion_mnist_cnn.py
```

### With Real Fashion-MNIST Data
Replace the `download_fashion_mnist()` function with:

```python
# Using TensorFlow
import tensorflow as tf
(x_train, y_train), (x_test, y_test) = \
    tf.keras.datasets.fashion_mnist.load_data()

# Reshape to (N, C, H, W) format
x_train = x_train[:, np.newaxis, :, :]
x_test = x_test[:, np.newaxis, :, :]
```

Or:

```python
# Using PyTorch
import torchvision
train_dataset = torchvision.datasets.FashionMNIST(
    root='./data', train=True, download=True
)
test_dataset = torchvision.datasets.FashionMNIST(
    root='./data', train=False, download=True
)
```

## Key Takeaways

1. **Architecture**: Use multiple conv layers per block for richer features
2. **Regularization**: Batch norm + dropout prevent overfitting
3. **Augmentation**: Increases effective dataset size
4. **Monitoring**: Track metrics for debugging and optimization
5. **Checkpointing**: Save best models for deployment
6. **Organization**: Separate concerns (data, model, training, evaluation)

## Common Modifications

### Increase Model Capacity
```python
# In __init__, change filter counts:
self.conv1a = tz.nn.Conv2d(1, 64, ...)      # 32 → 64
self.conv2a = tz.nn.Conv2d(64, 128, ...)    # 64 → 128
self.conv3 = tz.nn.Conv2d(128, 256, ...)    # 128 → 256
```

### Adjust Training Duration
```python
Config.EPOCHS = 50         # More training
Config.BATCH_SIZE = 256    # Larger batches
```

### Tune Regularization
```python
Config.DROPOUT_RATE = 0.3      # Less dropout
Config.WEIGHT_DECAY = 5e-4     # More L2 penalty
```

### Learning Rate Scheduling
Add after optimizer creation:
```python
scheduler = tz.optim.lr_scheduler.StepLR(
    optimizer, step_size=5, gamma=0.5
)
# Call scheduler.step() after each epoch
```

## Performance Optimization Tips

1. **GPU Acceleration**: Ensure CUDA is available
2. **Larger Batches**: Increase `BATCH_SIZE` if memory allows
3. **Mixed Precision**: Use AMP for faster training
4. **Data Parallelism**: Train on multiple GPUs
5. **Efficient Augmentation**: Use GPU-accelerated transforms

## Troubleshooting

### Loss Not Decreasing
- Check learning rate (try 1e-4 or 1e-2)
- Verify data normalization
- Ensure gradients are flowing (check backward pass)

### Overfitting
- Increase dropout rates
- Add more augmentation
- Reduce model capacity
- Add weight decay

### Out of Memory
- Reduce batch size
- Use gradient accumulation
- Reduce model size
- Enable mixed precision

## Next Steps

After mastering this example:

1. **Try Different Architectures**: ResNet, VGG, EfficientNet
2. **Advanced Augmentation**: CutOut, MixUp, AutoAugment
3. **Transfer Learning**: Use pretrained models
4. **Model Deployment**: Export to ONNX, serve with REST API
5. **Visualization**: Plot filters, feature maps, training curves
6. **Hyperparameter Tuning**: Grid search, Bayesian optimization

## References

- Fashion-MNIST: https://github.com/zalandoresearch/fashion-mnist
- Batch Normalization: Ioffe & Szegedy (2015)
- Dropout: Srivastava et al. (2014)
- Adam Optimizer: Kingma & Ba (2014)

## Code Quality Highlights

✅ **Well-commented**: Every function and section documented
✅ **Type hints**: Clear parameter and return types
✅ **Error handling**: Graceful degradation and validation
✅ **Configuration**: Easy to modify hyperparameters
✅ **Progress tracking**: Real-time training feedback
✅ **Modular design**: Reusable components
✅ **Best practices**: Follows deep learning conventions

---

**Ready to train your first production-ready CNN!** 🚀
