# Changelog

All notable changes to Tenzor will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-01-15

### Initial Public Release

This is the first public release of Tenzor, a high-performance tensor computation and deep learning library.

### Core Features

#### Tensor Operations
- Multi-dimensional tensor support with efficient memory management
- Comprehensive math operations: add, sub, mul, div, matmul, pow, sqrt, exp, log
- Reduction operations: sum, mean, max, min, argmax, argmin, prod
- Shape operations: reshape, view, transpose, permute, squeeze, unsqueeze
- Indexing: slice, gather, scatter, masked_select, masked_fill, index_select
- Broadcasting support for all element-wise operations
- Multiple data types: Float32, Float64, Float16, BFloat16, Int8/16/32/64, UInt8/16/32/64, Bool, Complex64/128

#### Automatic Differentiation
- Full reverse-mode autodiff with computational graph tracking
- Gradient accumulation and zeroing
- Support for in-place operations
- Custom autograd function support
- Gradient checkpointing for memory-efficient training

#### Neural Network Layers
- **Linear layers**: Linear (fully connected)
- **Convolutional layers**: Conv1d, Conv2d, Conv3d with grouped convolution support
- **Normalization**: BatchNorm1d/2d, LayerNorm, GroupNorm, RMSNorm
- **Pooling**: MaxPool1d/2d/3d, AvgPool1d/2d/3d, AdaptiveAvgPool, AdaptiveMaxPool
- **Recurrent**: RNN, LSTM, GRU with bidirectional support
- **Attention**: MultiheadAttention, ScaledDotProductAttention
- **Activation functions**: ReLU, LeakyReLU, GELU, SiLU/Swish, Sigmoid, Tanh, Softmax, ELU, SELU
- **Regularization**: Dropout, Dropout2d, AlphaDropout
- **Embedding**: Embedding, EmbeddingBag
- **Utilities**: Flatten, Unflatten, Sequential, ModuleList, ModuleDict

#### Optimizers
- SGD with momentum and Nesterov acceleration
- Adam and AdamW with weight decay
- AdamAtan2 (for HRM training stability)
- RMSprop
- Adagrad

#### Learning Rate Schedulers
- StepLR, MultiStepLR
- ExponentialLR
- CosineAnnealingLR, CosineAnnealingWarmRestarts
- OneCycleLR
- ReduceLROnPlateau
- Linear and polynomial warmup

#### Loss Functions
- MSELoss
- CrossEntropyLoss
- NLLLoss
- BCELoss, BCEWithLogitsLoss
- L1Loss, SmoothL1Loss
- HuberLoss
- CTC Loss
- Focal Loss
- Label Smoothing Cross Entropy

### Backend Support

#### CPU Backend
- SIMD optimization (SSE4.2, AVX2, AVX-512)
- OpenMP parallelization
- Optimized BLAS operations

#### CUDA Backend (NVIDIA GPUs)
- CUDA 12.0+ support
- cuBLAS integration for matrix operations
- cuDNN support for convolutions
- Custom CUDA kernels for all operations
- Multi-GPU support
- Mixed precision training (FP16/BF16)

#### ROCm Backend (AMD GPUs)
- ROCm 5.0+ support
- hipBLAS integration
- MIOpen for optimized convolutions
- Full operation parity with CUDA

#### OneAPI Backend (Intel GPUs)
- OneAPI 2023.0+ support
- oneMKL integration
- oneDNN for neural network operations
- Support for Intel Arc, Data Center, and integrated GPUs

#### Vulkan Backend
- Cross-platform GPU compute via Vulkan 1.2+
- SPIR-V compute shaders
- Support for all major GPU vendors
- Optimized kernels for Float16, Float32, Float64

#### Metal Backend (Apple Silicon)
- macOS 12+ and iOS 15+ support
- Metal Performance Shaders integration
- Optimized for M1/M2/M3 chips

### Pre-built Models

#### Vision Models
- ResNet (18, 34, 50, 101, 152)
- VGG (11, 13, 16, 19)
- MobileNetV2, MobileNetV3
- EfficientNet (B0-B7)
- ConvNeXt
- Vision Transformer (ViT)
- Swin Transformer
- YOLO (v5, v8)
- Faster R-CNN, Mask R-CNN
- U-Net, DeepLabV3+

#### Language Models
- BERT, RoBERTa
- GPT-2
- T5
- ALBERT
- ELECTRA

#### Specialized
- Hierarchical Reasoning Model (HRM) - brain-inspired recurrent architecture

### Data Loading
- Dataset and DataLoader abstractions
- Multi-worker data loading
- Prefetching and caching
- Data transforms and augmentation
- Image transforms: Resize, RandomCrop, RandomHorizontalFlip, Normalize, ColorJitter

### Model Serialization
- Save and load model checkpoints
- State dict serialization
- ONNX export and import
- Model hub for pretrained weights

### JIT Compilation
- Function tracing
- Script compilation
- Kernel fusion optimization
- Serialization of traced models

### Quantization
- Post-training quantization (INT8)
- Quantization-aware training
- Dynamic quantization
- Mixed precision inference

### Distributed Training
- Data parallel training
- Model parallel support
- Gradient synchronization
- Process group management

### Utilities
- TensorBoard integration for logging
- Benchmark utilities
- Memory profiling
- Configuration management
- Logging framework

### Python Bindings
- Complete pybind11 bindings
- NumPy interoperability
- Python 3.8 - 3.13 support
- Pythonic API matching C++ interface

### Documentation
- Comprehensive API documentation (Doxygen)
- Getting started guide
- Architecture documentation
- Example tutorials (Python and C++)
- Installation guide

### Testing
- Extensive test suite with 500+ tests
- Multi-dtype testing
- Backend parity testing
- Gradient checking utilities
- Performance benchmarks

---

## [Unreleased]

### Planned Features
- WebGPU backend for browser deployment
- Distributed training improvements
- More pre-built model architectures
- Enhanced JIT optimization
- Model compression techniques (pruning, knowledge distillation)

---

## Version History Summary

| Version | Date | Highlights |
|---------|------|------------|
| 1.0.0 | 2025-01-15 | Initial public release |

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to contribute to Tenzor.

## License

Tenzor is licensed under the MIT License. See [LICENSE](LICENSE) for details.
