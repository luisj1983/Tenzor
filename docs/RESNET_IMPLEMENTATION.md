# ResNet Family Implementation - Phase 9

## Overview

Complete implementation of the ResNet (Residual Network) family of deep convolutional neural networks for computer vision tasks. This implementation includes all major variants from the original paper and subsequent improvements.

## Files Created

All files are properly organized in appropriate subdirectories:

1. **Header**: `/home/lee/Projects/Tenzor/include/tenzor/models/resnet.hpp`
2. **Implementation**: `/home/lee/Projects/Tenzor/src/models/resnet.cpp`
3. **Tests**: `/home/lee/Projects/Tenzor/tests/unit/test_resnet.cpp`
4. **Example**: `/home/lee/Projects/Tenzor/examples/cv/resnet_example.cpp`

## Architecture Components

### 1. BasicBlock
- Used in ResNet-18 and ResNet-34
- Architecture: `3x3 conv -> BN -> ReLU -> 3x3 conv -> BN -> (+skip) -> ReLU`
- Channel expansion factor: 1
- Parameters:
  - `in_channels`: Number of input channels
  - `out_channels`: Number of output channels
  - `stride`: Spatial downsampling (1 or 2)
  - Optional downsampling module for skip connections

### 2. Bottleneck
- Used in ResNet-50, ResNet-101, ResNet-152, and variants
- Architecture: `1x1 conv -> BN -> ReLU -> 3x3 conv -> BN -> ReLU -> 1x1 conv -> BN -> (+skip) -> ReLU`
- Channel expansion factor: 4
- Supports grouped convolutions for ResNeXt
- Parameters:
  - `in_channels`: Number of input channels
  - `out_channels`: Number of intermediate channels
  - `stride`: Spatial downsampling (1 or 2)
  - `groups`: Number of groups for grouped convolutions
  - `base_width`: Base width for channel calculations
  - Optional downsampling module for skip connections

### 3. ResNet Main Class
Complete ResNet architecture with:
- Initial 7x7 convolution (stride 2)
- Batch normalization
- ReLU activation
- 3x3 max pooling (stride 2)
- Four residual layer groups (layer1-4)
- Adaptive average pooling (1x1 output)
- Fully connected classification layer

## Supported Variants

### Standard ResNet Models

#### ResNet-18
- **Blocks**: [2, 2, 2, 2] BasicBlocks
- **Parameters**: ~11.7M
- **Depth**: 18 layers
- **Top-1 Accuracy (ImageNet)**: ~69.8%
- **Use case**: Fast inference, resource-constrained environments

```cpp
auto model = models::resnet18(1000, false);
```

#### ResNet-34
- **Blocks**: [3, 4, 6, 3] BasicBlocks
- **Parameters**: ~21.8M
- **Depth**: 34 layers
- **Top-1 Accuracy (ImageNet)**: ~73.3%
- **Use case**: Balance between accuracy and speed

```cpp
auto model = models::resnet34(1000, false);
```

#### ResNet-50
- **Blocks**: [3, 4, 6, 3] Bottlenecks
- **Parameters**: ~25.6M
- **Depth**: 50 layers
- **Top-1 Accuracy (ImageNet)**: ~76.1%
- **Use case**: Standard choice for most applications

```cpp
auto model = models::resnet50(1000, false);
```

#### ResNet-101
- **Blocks**: [3, 4, 23, 3] Bottlenecks
- **Parameters**: ~44.5M
- **Depth**: 101 layers
- **Top-1 Accuracy (ImageNet)**: ~77.4%
- **Use case**: Higher accuracy requirements

```cpp
auto model = models::resnet101(1000, false);
```

#### ResNet-152
- **Blocks**: [3, 8, 36, 3] Bottlenecks
- **Parameters**: ~60.2M
- **Depth**: 152 layers
- **Top-1 Accuracy (ImageNet)**: ~78.3%
- **Use case**: Maximum accuracy in standard ResNet

```cpp
auto model = models::resnet152(1000, false);
```

### ResNeXt Variants

ResNeXt uses grouped convolutions to improve representational power.

#### ResNeXt-50 (32x4d)
- **Blocks**: [3, 4, 6, 3] Bottlenecks
- **Groups**: 32, **Width per group**: 4
- **Parameters**: ~25.0M
- **Top-1 Accuracy (ImageNet)**: ~77.6%
- **Advantage**: Better accuracy than ResNet-50 with similar parameters

```cpp
auto model = models::resnext50_32x4d(1000, false);
```

#### ResNeXt-101 (32x8d)
- **Blocks**: [3, 4, 23, 3] Bottlenecks
- **Groups**: 32, **Width per group**: 8
- **Parameters**: ~88.8M
- **Top-1 Accuracy (ImageNet)**: ~79.3%
- **Advantage**: State-of-the-art accuracy

```cpp
auto model = models::resnext101_32x8d(1000, false);
```

### Wide ResNet Variants

Wide ResNet increases channel width for improved accuracy.

#### Wide ResNet-50-2
- **Blocks**: [3, 4, 6, 3] Bottlenecks
- **Width multiplier**: 2x (base_width=128)
- **Parameters**: ~68.9M
- **Top-1 Accuracy (ImageNet)**: ~78.5%
- **Advantage**: Better accuracy through wider layers

```cpp
auto model = models::wide_resnet50_2(1000, false);
```

#### Wide ResNet-101-2
- **Blocks**: [3, 4, 23, 3] Bottlenecks
- **Width multiplier**: 2x (base_width=128)
- **Parameters**: ~126.9M
- **Top-1 Accuracy (ImageNet)**: ~78.8%
- **Advantage**: Maximum accuracy in Wide ResNet

```cpp
auto model = models::wide_resnet101_2(1000, false);
```

## Key Features

### 1. Skip Connections
All models implement proper residual connections:
- Identity mapping when dimensions match
- 1x1 convolution downsampling when stride != 1 or channels change
- Enables training of very deep networks (100+ layers)

### 2. Batch Normalization
- Applied after every convolution
- Stabilizes training
- Reduces internal covariate shift

### 3. Kaiming Initialization
- Weights initialized using Kaiming uniform distribution
- Optimized for ReLU activations
- Implemented in individual layer constructors

### 4. Flexible Input Sizes
- Standard: 224x224 (ImageNet)
- Minimum recommended: 32x32 (must pass through all pooling layers)
- Adaptive average pooling allows variable input sizes

### 5. Configurable Number of Classes
```cpp
// CIFAR-10 (10 classes)
auto model = models::resnet50(10, false);

// ImageNet (1000 classes)
auto model = models::resnet50(1000, false);

// Custom dataset (100 classes)
auto model = models::resnet50(100, false);
```

### 6. Training/Evaluation Modes
```cpp
model->train();  // Enable batch norm updates, dropout, etc.
model->eval();   // Use running statistics, disable dropout
```

### 7. Device Management
```cpp
model->to(Device::cuda(0));  // Move to GPU
model->to(Device::cpu());     // Move to CPU
```

### 8. Pretrained Weight Loading
```cpp
auto model = models::resnet50(1000, true);  // Load pretrained weights
// Or manually:
model->load_pretrained("path/to/weights.pth");
```

## API Usage Examples

### Basic Inference
```cpp
#include "tenzor/models/resnet.hpp"

// Create model
auto model = models::resnet50(1000, false);
model->eval();

// Prepare input (batch_size=4, channels=3, height=224, width=224)
Variable input(Tensor({4, 3, 224, 224}, DType::Float32, Device::cpu()), false);

// Forward pass
Variable output = model->forward(input);  // Shape: [4, 1000]

// Apply softmax for probabilities
Softmax softmax(-1);
Variable probs = softmax.forward(output);
```

### Training Loop
```cpp
#include "tenzor/models/resnet.hpp"
#include "tenzor/nn/loss/losses.hpp"
#include "tenzor/nn/optim/sgd.hpp"

// Create model
auto model = models::resnet18(100, false);
model->train();

// Setup optimizer and loss
auto params = model->parameters();
optim::SGD optimizer(params, 0.01, 0.9);
loss::CrossEntropyLoss criterion;

// Training loop
for (int epoch = 0; epoch < num_epochs; ++epoch) {
    for (auto& [images, labels] : dataloader) {
        // Zero gradients
        optimizer.zero_grad();

        // Forward pass
        Variable outputs = model->forward(images);

        // Compute loss
        Variable loss = criterion.forward(outputs, labels);

        // Backward pass
        loss.backward();

        // Update weights
        optimizer.step();
    }
}
```

### Transfer Learning
```cpp
// Load pretrained ImageNet model
auto model = models::resnet50(1000, true);

// Freeze early layers (optional)
auto params = model->parameters();
for (size_t i = 0; i < params.size() - 2; ++i) {
    params[i]->set_requires_grad(false);
}

// Replace final layer for new task (20 classes)
// Note: Would need to access fc_ directly or provide a method

// Fine-tune with lower learning rate
std::vector<std::shared_ptr<Variable>> trainable_params;
for (auto& p : params) {
    if (p->requires_grad()) {
        trainable_params.push_back(p);
    }
}
optim::Adam optimizer(trainable_params, 0.001);
```

## Comprehensive Test Coverage

The test suite (`test_resnet.cpp`) includes:

### BasicBlock Tests
- ✅ Forward pass shape validation (stride=1)
- ✅ Forward pass shape validation (stride=2, downsampling)
- ✅ Gradient flow through skip connections
- ✅ Error handling for invalid parameters

### Bottleneck Tests
- ✅ Forward pass shape validation (stride=1)
- ✅ Forward pass shape validation (stride=2, downsampling)
- ✅ Grouped convolution support (ResNeXt)
- ✅ Gradient flow through all layers

### ResNet-18 Tests
- ✅ Architecture validation (parameter count ~11.7M)
- ✅ Forward pass shape [N, 3, 224, 224] -> [N, 1000]
- ✅ Custom number of classes
- ✅ Different input sizes (128x128, etc.)
- ✅ End-to-end gradient flow
- ✅ Training/evaluation mode switching

### ResNet-34/50/101/152 Tests
- ✅ Forward pass shape validation
- ✅ Parameter count validation
- ✅ Gradient flow verification

### ResNeXt Tests
- ✅ ResNeXt-50 (32x4d) forward pass
- ✅ ResNeXt-101 (32x8d) forward pass

### Wide ResNet Tests
- ✅ Wide ResNet-50-2 forward pass
- ✅ Wide ResNet-101-2 forward pass

### Additional Tests
- ✅ Batch normalization behavior (train vs eval)
- ✅ State dict save/load
- ✅ Small batch size (1)
- ✅ Large batch size (16+)
- ✅ Parameter sharing verification
- ✅ Zero gradient functionality

## Example Code

See `/home/lee/Projects/Tenzor/examples/cv/resnet_example.cpp` for comprehensive examples including:
1. Creating different ResNet variants
2. Forward pass and inference
3. Training loop
4. Transfer learning
5. Architecture comparison
6. Handling different input sizes
7. Model evaluation

## Implementation Notes

### Design Decisions

1. **Non-template Design**: Uses runtime polymorphism instead of templates for cleaner API and better compilation times
2. **Proper Module Registration**: All submodules are properly registered for parameter management
3. **Memory Efficiency**: Uses shared_ptr for modules to enable proper memory management
4. **Skip Connection Implementation**: Carefully handles dimension matching through downsampling
5. **Grouped Convolutions**: Full support for ResNeXt-style grouped convolutions

### Dependencies

Required Tenzor components (all verified present):
- ✅ `nn::Conv2d` - 2D convolution layers
- ✅ `nn::BatchNorm2d` - Batch normalization for 2D inputs
- ✅ `nn::Linear` - Fully connected layers
- ✅ `nn::MaxPool2d` - Max pooling layers
- ✅ `nn::AdaptiveAvgPool2d` - Adaptive average pooling
- ✅ `nn::ReLU` - ReLU activation
- ✅ `nn::Module` - Base module class
- ✅ `nn::Sequential` - Sequential container
- ✅ `Variable` - Autograd variable wrapper
- ✅ `Tensor` - Core tensor class

### Performance Considerations

1. **Memory Usage**:
   - ResNet-18: ~45MB (FP32), ~23MB (FP16)
   - ResNet-50: ~100MB (FP32), ~50MB (FP16)
   - ResNet-152: ~230MB (FP32), ~115MB (FP16)

2. **Computation**:
   - ResNet-18: ~1.8 GFLOPs
   - ResNet-50: ~3.8 GFLOPs
   - ResNet-101: ~7.6 GFLOPs

3. **Optimizations**:
   - Batch normalization fusion (can be added)
   - Conv-ReLU fusion (can be added)
   - Quantization support (via tenzor::nn::quantization)

## References

1. **Original ResNet Paper**:
   - He, K., Zhang, X., Ren, S., & Sun, J. (2016). "Deep Residual Learning for Image Recognition"
   - arXiv:1512.03385

2. **ResNeXt Paper**:
   - Xie, S., Girshick, R., Dollár, P., Tu, Z., & He, K. (2017). "Aggregated Residual Transformations for Deep Neural Networks"
   - arXiv:1611.05431

3. **Wide ResNet Paper**:
   - Zagoruyko, S., & Komodakis, N. (2016). "Wide Residual Networks"
   - arXiv:1605.07146

## Future Enhancements

Potential improvements:
- [ ] Pre-activation ResNet variant (ResNet-v2)
- [ ] Squeeze-and-Excitation ResNet (SE-ResNet)
- [ ] ResNeSt (Split-Attention Networks)
- [ ] EfficientNet-style compound scaling
- [ ] Automatic mixed precision training
- [ ] Model pruning and quantization
- [ ] ONNX export support

## Summary

This implementation provides:
- ✅ Complete ResNet family (ResNet-18 through ResNet-152)
- ✅ ResNeXt variants with grouped convolutions
- ✅ Wide ResNet variants with increased width
- ✅ Proper skip connections and dimension matching
- ✅ Full autograd support for training
- ✅ Comprehensive test suite (30+ tests)
- ✅ Detailed examples and documentation
- ✅ Clean, maintainable code following best practices
- ✅ NO STUBS - fully functional implementation
- ✅ Proper file organization in subdirectories

All files are ready for integration into the Tenzor framework for Phase 9 computer vision model support.
