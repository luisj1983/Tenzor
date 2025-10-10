# Tenzor Library - Comprehensive TODO Roadmap
## Path to World-Class Neural Network Library

**Document Version**: 1.0
**Created**: 2025-10-10
**Status**: Phase 5 Complete → Phase 6+ Planning
**Current Completion**: 78% overall (C++: 100%, Python: 40%)

---

## Executive Summary

This document outlines the complete roadmap to transform Tenzor into a world-class tensor and neural network library comparable to PyTorch, TensorFlow, and JAX. It follows the SPARC methodology for systematic development.

**Current State**: Production-ready C++ library with 448/448 tests passing
**Target State**: Complete ecosystem with Python bindings, advanced features, and comprehensive documentation

---

## Quick Reference - Critical Path to v1.0

**Immediate Blockers (220 hours, ~6 weeks):**
1. Complete Python bindings (80h)
2. NumPy interoperability (40h)
3. API documentation (60h)
4. Tutorial examples (40h)

**v1.0 Release Criteria:**
- ✅ All core features accessible from Python
- ✅ Zero-copy NumPy integration
- ✅ Complete API documentation
- ✅ 10+ tutorial examples
- ✅ Getting started guide

---

# SPARC Phase 6: Python Ecosystem & Documentation

## 6.1 Python Bindings Completion (HIGH PRIORITY - BLOCKING v1.0)

### Status: 40% Complete → Target: 100%

### 6.1.1 Missing Layer Bindings (30 hours)

**Conv Layers:**
- [ ] Conv2d bindings with all parameters
  - [ ] in_channels, out_channels, kernel_size
  - [ ] stride, padding, dilation, groups, bias
  - [ ] Device support (CPU/CUDA)
- [ ] Conv1d bindings
- [ ] ConvTranspose2d (deconvolution) bindings
- [ ] Conv3d bindings (if 3D support added)

**Normalization Layers:**
- [ ] BatchNorm2d bindings
  - [ ] num_features, eps, momentum
  - [ ] affine, track_running_stats
  - [ ] train/eval mode switching
- [ ] BatchNorm1d bindings
- [ ] LayerNorm bindings
- [ ] GroupNorm bindings (if implemented)
- [ ] InstanceNorm bindings (if implemented)

**Regularization:**
- [ ] Dropout bindings
  - [ ] p (dropout probability)
  - [ ] inplace option
  - [ ] train/eval mode
- [ ] Dropout2d bindings
- [ ] AlphaDropout bindings (if implemented)

**Pooling Layers:**
- [ ] MaxPool2d bindings
  - [ ] kernel_size, stride, padding
  - [ ] dilation, return_indices, ceil_mode
- [ ] AvgPool2d bindings
- [ ] AdaptiveAvgPool2d bindings
- [ ] AdaptiveMaxPool2d bindings
- [ ] MaxPool1d, AvgPool1d bindings

**Recurrent Layers (To Implement):**
- [ ] RNN bindings
- [ ] LSTM bindings
- [ ] GRU bindings
- [ ] Bidirectional wrapper

**Utility Layers:**
- [ ] Flatten bindings
- [ ] Reshape bindings
- [ ] Identity bindings
- [ ] Embedding bindings (if implemented)

**Files to Modify:**
- `python/bindings.cpp` (add 200+ lines)

---

### 6.1.2 Activation Function Bindings (10 hours)

**Current:** 0/12 activations bound
**Target:** 12/12 bound

- [ ] ReLU (with inplace option)
- [ ] LeakyReLU (with negative_slope)
- [ ] PReLU (with learnable parameter)
- [ ] ELU (with alpha)
- [ ] SELU
- [ ] GELU
- [ ] Sigmoid
- [ ] Tanh
- [ ] Softmax (with dim parameter)
- [ ] LogSoftmax (with dim parameter)
- [ ] SiLU/Swish
- [ ] Mish
- [ ] Hardswish (if implemented)
- [ ] Hardsigmoid (if implemented)

**Files to Modify:**
- `python/bindings.cpp` (add 80+ lines)

---

### 6.1.3 Loss Function Bindings (8 hours)

**Current:** 0/8 losses bound
**Target:** 8/8 bound

- [ ] MSELoss (with reduction)
- [ ] L1Loss (with reduction)
- [ ] SmoothL1Loss (with beta, reduction)
- [ ] CrossEntropyLoss (with weight, ignore_index, reduction)
- [ ] NLLLoss (with weight, ignore_index, reduction)
- [ ] BCELoss (with weight, reduction)
- [ ] BCEWithLogitsLoss (with pos_weight, reduction)
- [ ] KLDivLoss (if implemented)
- [ ] CosineEmbeddingLoss (if implemented)
- [ ] HingeLoss (if implemented)

**Files to Modify:**
- `python/bindings.cpp` (add 60+ lines)

---

### 6.1.4 Optimizer Bindings Enhancement (5 hours)

**Current:** SGD, Adam bound
**Missing:**

- [ ] AdamW bindings (already implemented in C++)
- [ ] RMSprop bindings (if implemented)
- [ ] Adagrad bindings (if implemented)
- [ ] Adadelta bindings (if implemented)
- [ ] LAMB optimizer (if implemented)
- [ ] state_dict() / load_state_dict() for all optimizers
- [ ] Parameter groups support

**Files to Modify:**
- `python/bindings.cpp` (add 40+ lines)

---

### 6.1.5 Sequential Container Bindings (5 hours)

**Status:** NOT BOUND (C++ implemented)

- [ ] Sequential class binding
- [ ] add_module() method
- [ ] forward() method
- [ ] __getitem__() for Python indexing
- [ ] __len__() for Python len()
- [ ] Pythonic construction: `Sequential(layer1, layer2, ...)`

**Files to Modify:**
- `python/bindings.cpp` (add 30+ lines)

---

### 6.1.6 Tensor Operations - Missing 31/40 (20 hours)

**Current Python Coverage: 9/40 operations (22.5%)**

**Math Operations:**
- [ ] div (division)
- [ ] pow (power)
- [ ] exp (exponential)
- [ ] log (logarithm)
- [ ] sqrt (square root)
- [ ] sin, cos, tan (trigonometric)
- [ ] abs (absolute value)
- [ ] clamp (clamp values)
- [ ] sign, neg (sign/negation)

**Reduction Operations:**
- [ ] sum (with dim, keepdim)
- [ ] mean (with dim, keepdim)
- [ ] max (with dim)
- [ ] min (with dim)
- [ ] argmax, argmin
- [ ] std, var (standard deviation, variance)
- [ ] prod (product)
- [ ] all, any (boolean reductions)

**Transformation Operations:**
- [ ] transpose (2D transpose)
- [ ] permute (multi-dimensional transpose)
- [ ] squeeze (remove singleton dimensions)
- [ ] unsqueeze (add singleton dimension)
- [ ] flatten (flatten to 1D or 2D)
- [ ] unflatten
- [ ] view (alias for reshape)
- [ ] expand, repeat

**Indexing/Slicing:**
- [ ] slice (with dim, start, end, step)
- [ ] index_select
- [ ] gather, scatter
- [ ] masked_select
- [ ] __getitem__ (Python indexing: tensor[0, :, 1:5])
- [ ] __setitem__ (Python assignment: tensor[0, :] = value)

**Utility Operations:**
- [ ] clone
- [ ] detach
- [ ] contiguous
- [ ] to(dtype) overload
- [ ] cuda() method
- [ ] cpu() method
- [ ] item() (extract scalar)
- [ ] numpy() (to NumPy array)
- [ ] from_numpy() (static method)

**Files to Modify:**
- `python/bindings.cpp` (add 150+ lines)

---

### 6.1.7 Learning Rate Schedulers (5 hours)

**Current:** NOT BOUND (C++ has StepLR, ExponentialLR, CosineAnnealingLR)

- [ ] StepLR bindings
- [ ] ExponentialLR bindings
- [ ] CosineAnnealingLR bindings
- [ ] ReduceLROnPlateau bindings (if implemented)
- [ ] CyclicLR bindings (if implemented)
- [ ] OneCycleLR bindings (if implemented)

**Files to Modify:**
- `python/bindings.cpp` (add 40+ lines)

---

### 6.1.8 Autograd Enhancement (2 hours)

**Current:** Variable bound with basic backward()

- [ ] grad property (mutable access)
- [ ] requires_grad property (setter)
- [ ] is_leaf property
- [ ] grad_fn property
- [ ] retain_grad() method
- [ ] register_hook() for custom backward hooks
- [ ] Context manager: `with no_grad():` block
- [ ] Context manager: `with enable_grad():` block

**Files to Modify:**
- `python/bindings.cpp` (add 30+ lines)

---

## 6.2 NumPy Interoperability (HIGH PRIORITY - BLOCKING v1.0)

### Status: 0% Complete → Target: 100%

### 6.2.1 Zero-Copy Conversions (25 hours)

**Implementation Required:**

- [ ] `tensor_to_numpy()` function
  - [ ] Zero-copy when CPU tensor (shared memory)
  - [ ] Copy when CUDA tensor (device to host)
  - [ ] Proper dtype mapping (Float32 → np.float32, etc.)
  - [ ] Handle stride/contiguity correctly
  - [ ] Reference counting to prevent dangling pointers
  - [ ] Support for all DType variants

- [ ] `numpy_to_tensor()` function
  - [ ] Zero-copy when possible (aligned, contiguous)
  - [ ] Copy when necessary (non-contiguous, misaligned)
  - [ ] Automatic dtype detection
  - [ ] Device parameter (CPU/CUDA)
  - [ ] Memory ownership tracking

- [ ] Tensor.numpy() method
  - [ ] Returns NumPy array view
  - [ ] Automatic sync for CUDA tensors
  - [ ] Proper error messages if tensor is on CUDA

- [ ] Tensor constructor from NumPy
  - [ ] `Tensor(numpy_array)` constructor
  - [ ] `Tensor.from_numpy(numpy_array)` static method

**DType Mapping Table:**
```cpp
// Required mappings
Float32 ↔ np.float32
Float64 ↔ np.float64
Int32   ↔ np.int32
Int64   ↔ np.int64
Int8    ↔ np.int8
Int16   ↔ np.int16
UInt8   ↔ np.uint8
Bool    ↔ np.bool_
```

**Files to Create:**
- `python/numpy_interop.cpp` (400+ lines)

**Files to Modify:**
- `python/bindings.cpp` (add includes and registration)

---

### 6.2.2 Memory Safety & Lifetime Management (10 hours)

**Critical Issues to Handle:**

- [ ] Prevent dangling pointers
  - [ ] NumPy array holds reference to Tensor storage
  - [ ] Tensor holds reference to NumPy buffer
  - [ ] Proper reference counting

- [ ] Thread safety
  - [ ] GIL (Global Interpreter Lock) awareness
  - [ ] Atomic reference counting

- [ ] Device synchronization
  - [ ] Automatic CUDA sync before numpy() call
  - [ ] Error handling for CUDA tensors

**Test Coverage Required:**
- [ ] Round-trip conversion (Tensor → NumPy → Tensor)
- [ ] Memory aliasing (modifications propagate)
- [ ] Lifetime tests (ensure no crashes)
- [ ] Multi-threaded access tests

**Files to Create:**
- `tests/python/test_numpy_interop.py` (200+ lines)

---

### 6.2.3 Advanced NumPy Features (5 hours)

- [ ] Support for NumPy-style broadcasting in Tensor ops
- [ ] NumPy-compatible indexing (fancy indexing)
- [ ] astype() method (like NumPy's astype)
- [ ] Integration with scipy.sparse (if applicable)
- [ ] NumPy random seed synchronization

---

## 6.3 API Documentation (HIGH PRIORITY - BLOCKING v1.0)

### Status: ~0% Doxygen Coverage → Target: 100%

### 6.3.1 Core API Documentation (30 hours)

**Headers to Document (350+ APIs):**

**Tensor System (8 files, ~80 APIs):**
- [ ] `include/tenzor/core/tensor.hpp`
  - [ ] Class overview with usage examples
  - [ ] All constructors documented
  - [ ] All methods with @param, @return, @throws
  - [ ] Code examples for common operations
  - [ ] Performance notes

- [ ] `include/tenzor/core/dtype.hpp`
  - [ ] DType enum values
  - [ ] Type traits documentation
  - [ ] Concept definitions

- [ ] `include/tenzor/core/device.hpp`
- [ ] `include/tenzor/core/storage.hpp`
- [ ] `include/tenzor/core/shape.hpp`

**Backend System (4 files, ~30 APIs):**
- [ ] `include/tenzor/backend/backend.hpp`
- [ ] `include/tenzor/backend/loader.hpp`
- [ ] `include/tenzor/backend/registry.hpp`
- [ ] `include/tenzor/backend/dispatch.hpp`

**Operations (5 files, ~60 APIs):**
- [ ] `include/tenzor/ops/creation.hpp`
- [ ] `include/tenzor/ops/math.hpp`
- [ ] `include/tenzor/ops/reduction.hpp`
- [ ] `include/tenzor/ops/transform.hpp`
- [ ] `include/tenzor/ops/indexing.hpp`

**Autograd (4 files, ~40 APIs):**
- [ ] `include/tenzor/autograd/variable.hpp`
- [ ] `include/tenzor/autograd/function.hpp`
- [ ] `include/tenzor/autograd/graph.hpp`
- [ ] `include/tenzor/autograd/engine.hpp`

**Neural Networks (12 files, ~100 APIs):**
- [ ] `include/tenzor/nn/module.hpp`
- [ ] `include/tenzor/nn/layers/linear.hpp`
- [ ] `include/tenzor/nn/layers/conv.hpp`
- [ ] `include/tenzor/nn/layers/batchnorm.hpp`
- [ ] `include/tenzor/nn/layers/dropout.hpp`
- [ ] `include/tenzor/nn/layers/pooling.hpp`
- [ ] `include/tenzor/nn/layers/normalization.hpp`
- [ ] `include/tenzor/nn/activations/activations.hpp`
- [ ] `include/tenzor/nn/loss/losses.hpp`
- [ ] `include/tenzor/nn/optim/optimizer.hpp`
- [ ] `include/tenzor/nn/optim/sgd.hpp`
- [ ] `include/tenzor/nn/optim/adam.hpp`

**Utilities (3 files, ~20 APIs):**
- [ ] `include/tenzor/parallel/threadpool.hpp`
- [ ] `include/tenzor/utils/logging.hpp`
- [ ] `include/tenzor/utils/error.hpp`

---

### 6.3.2 Doxygen Configuration (5 hours)

- [ ] Create `Doxyfile` configuration
  - [ ] Project name, version, description
  - [ ] Input directories
  - [ ] Output formats (HTML, optionally PDF)
  - [ ] Styling and theme
  - [ ] Example code extraction
  - [ ] Graph generation

- [ ] Set up documentation build
  - [ ] Add to CMakeLists.txt
  - [ ] `make docs` target
  - [ ] CI integration for doc generation

- [ ] Host documentation
  - [ ] GitHub Pages setup
  - [ ] Automatic deployment on releases

**Files to Create:**
- `Doxyfile`
- `docs/CMakeLists.txt`

---

### 6.3.3 User Guide & Tutorials (25 hours)

**Getting Started Guide (8 hours):**
- [ ] `docs/getting_started.md`
  - [ ] Installation instructions (Linux, macOS, Windows)
  - [ ] Building from source
  - [ ] Python package installation
  - [ ] First tensor operations
  - [ ] First neural network training

**Core Concepts (10 hours):**
- [ ] `docs/tensor_operations.md`
  - [ ] Creation, manipulation, indexing
  - [ ] Broadcasting rules
  - [ ] Device management
  - [ ] Memory management

- [ ] `docs/autograd_tutorial.md`
  - [ ] How autograd works
  - [ ] Custom gradient functions
  - [ ] Gradient checkpointing

- [ ] `docs/neural_networks.md`
  - [ ] Module system
  - [ ] Building custom layers
  - [ ] Training loops
  - [ ] Saving/loading models

**Advanced Topics (7 hours):**
- [ ] `docs/backends.md`
  - [ ] Backend system architecture
  - [ ] Writing custom backends
  - [ ] Performance optimization

- [ ] `docs/cuda_programming.md`
  - [ ] CUDA tensor operations
  - [ ] Multi-GPU training
  - [ ] Memory optimization

- [ ] `docs/performance_guide.md`
  - [ ] Benchmarking
  - [ ] Profiling
  - [ ] Optimization techniques

---

## 6.4 Example Programs (BLOCKING v1.0)

### Status: 5 basic examples → Target: 30+ comprehensive examples

### 6.4.1 Basic Examples (10 hours)

**Already Have (Verify & Document):**
- [ ] Review and improve existing 5 examples
- [ ] Add detailed comments
- [ ] Add README for each example

**To Add:**
- [ ] `examples/01_tensor_basics.py`
  - [ ] Creation, operations, indexing
  - [ ] Device transfers
  - [ ] NumPy interop

- [ ] `examples/02_autograd_basics.py`
  - [ ] Variable creation
  - [ ] Gradient computation
  - [ ] Custom functions

- [ ] `examples/03_simple_linear.py`
  - [ ] Linear regression
  - [ ] MSE loss
  - [ ] SGD optimizer

- [ ] `examples/04_mnist_mlp.py`
  - [ ] MNIST data loading
  - [ ] MLP architecture
  - [ ] Training loop
  - [ ] Evaluation

- [ ] `examples/05_fashion_mnist_cnn.py`
  - [ ] CNN with Conv2d + BatchNorm + Pooling
  - [ ] Data augmentation
  - [ ] Validation set
  - [ ] Checkpointing

---

### 6.4.2 Computer Vision Examples (12 hours)

- [ ] `examples/cv/cifar10_resnet.py`
  - [ ] ResNet architecture implementation
  - [ ] Data augmentation
  - [ ] Learning rate scheduling
  - [ ] Accuracy: >90%

- [ ] `examples/cv/image_classification.py`
  - [ ] Transfer learning
  - [ ] Fine-tuning pretrained models
  - [ ] Custom dataset loading

- [ ] `examples/cv/autoencoder.py`
  - [ ] Encoder-decoder architecture
  - [ ] Reconstruction loss
  - [ ] Latent space visualization

- [ ] `examples/cv/gan_mnist.py`
  - [ ] Generator network
  - [ ] Discriminator network
  - [ ] Adversarial training

- [ ] `examples/cv/semantic_segmentation.py`
  - [ ] U-Net architecture
  - [ ] Pixel-wise classification
  - [ ] IoU metric

---

### 6.4.3 NLP Examples (10 hours - Requires RNN/LSTM)

- [ ] `examples/nlp/text_classification.py`
  - [ ] Embedding layer
  - [ ] LSTM/GRU
  - [ ] Sentiment analysis

- [ ] `examples/nlp/language_model.py`
  - [ ] Character-level RNN
  - [ ] Text generation
  - [ ] Perplexity metric

- [ ] `examples/nlp/seq2seq_translation.py`
  - [ ] Encoder-decoder
  - [ ] Attention mechanism
  - [ ] BLEU score

---

### 6.4.4 Advanced Examples (8 hours)

- [ ] `examples/advanced/multi_gpu_training.py`
  - [ ] DataParallel usage
  - [ ] Distributed data loading
  - [ ] Gradient synchronization

- [ ] `examples/advanced/mixed_precision.py`
  - [ ] FP16 training
  - [ ] Loss scaling
  - [ ] Performance benchmarks

- [ ] `examples/advanced/custom_layer.py`
  - [ ] Custom Module implementation
  - [ ] Custom forward/backward
  - [ ] Custom CUDA kernel integration

- [ ] `examples/advanced/neural_style_transfer.py`
  - [ ] Content + style loss
  - [ ] VGG feature extraction
  - [ ] Optimization-based approach

---

# SPARC Phase 7: Advanced Neural Network Components

## 7.1 Recurrent Neural Networks (MEDIUM PRIORITY)

### Status: NOT IMPLEMENTED → Target: Full RNN Support

### 7.1.1 RNN Cell (20 hours)

**C++ Implementation:**
- [ ] `include/tenzor/nn/layers/rnn.hpp`
- [ ] `src/nn/layers/rnn.cpp`

**Features:**
- [ ] RNNCell class
  - [ ] input_size, hidden_size parameters
  - [ ] nonlinearity (tanh, relu)
  - [ ] bias option
  - [ ] Weight initialization (Xavier/He)

- [ ] RNN class (multi-layer wrapper)
  - [ ] num_layers parameter
  - [ ] bidirectional flag
  - [ ] dropout between layers
  - [ ] batch_first option

**Operations:**
- [ ] rnn_cell_forward (single timestep)
- [ ] rnn_forward (sequence processing)
- [ ] rnn_backward (BPTT)

**Tests:**
- [ ] `tests/unit/test_rnn.cpp` (50+ tests)

**Python Bindings:**
- [ ] RNNCell binding
- [ ] RNN binding

---

### 7.1.2 LSTM (30 hours)

**C++ Implementation:**
- [ ] LSTMCell class
  - [ ] input_size, hidden_size
  - [ ] bias option
  - [ ] Input, forget, cell, output gates
  - [ ] Hidden state + cell state management

- [ ] LSTM class (multi-layer)
  - [ ] num_layers, bidirectional
  - [ ] dropout
  - [ ] proj_size (projection LSTM)

**CUDA Kernels:**
- [ ] `src/backends/cuda/kernels/lstm.cu`
  - [ ] Fused LSTM cell (all gates in one kernel)
  - [ ] Optimized for throughput

**CPU Kernels:**
- [ ] `src/backends/cpu/kernels/lstm.cpp`
  - [ ] OpenMP parallelization
  - [ ] SIMD optimizations

**Tests:**
- [ ] `tests/unit/test_lstm.cpp`
- [ ] Gradient checking
- [ ] Hidden state propagation tests

**Python Bindings:**
- [ ] LSTMCell binding
- [ ] LSTM binding

---

### 7.1.3 GRU (25 hours)

**C++ Implementation:**
- [ ] GRUCell class
  - [ ] Reset and update gates
  - [ ] Candidate hidden state

- [ ] GRU class (multi-layer)
  - [ ] Same interface as LSTM

**CUDA Kernels:**
- [ ] Fused GRU cell

**CPU Kernels:**
- [ ] Optimized GRU implementation

**Tests:**
- [ ] `tests/unit/test_gru.cpp`

**Python Bindings:**
- [ ] GRUCell binding
- [ ] GRU binding

---

### 7.1.4 Bidirectional Wrapper (5 hours)

- [ ] Bidirectional class
  - [ ] Wraps any RNN/LSTM/GRU
  - [ ] Forward and backward passes
  - [ ] Concatenates outputs

---

## 7.2 Transformer Components (HIGH PRIORITY for Modern NLP)

### Status: NOT IMPLEMENTED → Target: Full Transformer Support

### 7.2.1 Multi-Head Attention (25 hours)

**C++ Implementation:**
- [ ] `include/tenzor/nn/layers/attention.hpp`
- [ ] `src/nn/layers/attention.cpp`

**Features:**
- [ ] MultiheadAttention class
  - [ ] embed_dim, num_heads
  - [ ] Query, key, value projections
  - [ ] Output projection
  - [ ] Dropout
  - [ ] Attention mask support
  - [ ] key_padding_mask support

**Operations:**
- [ ] Scaled dot-product attention
- [ ] Softmax over attention scores
- [ ] Attention dropout
- [ ] Output projection

**CUDA Optimizations:**
- [ ] Flash Attention (memory-efficient attention)
- [ ] Fused scaled dot-product kernel

**Tests:**
- [ ] `tests/unit/test_attention.cpp`
- [ ] Masked attention tests
- [ ] Gradient tests

**Python Bindings:**
- [ ] MultiheadAttention binding

---

### 7.2.2 Transformer Encoder/Decoder (30 hours)

**TransformerEncoderLayer:**
- [ ] Self-attention
- [ ] Feed-forward network
- [ ] LayerNorm
- [ ] Residual connections
- [ ] Dropout

**TransformerEncoder:**
- [ ] Stack of encoder layers
- [ ] num_layers parameter
- [ ] Final normalization

**TransformerDecoderLayer:**
- [ ] Self-attention
- [ ] Cross-attention
- [ ] Feed-forward network
- [ ] LayerNorm, residuals, dropout

**TransformerDecoder:**
- [ ] Stack of decoder layers

**Transformer:**
- [ ] Complete encoder-decoder model
- [ ] Src/tgt mask generation helpers

**Files to Create:**
- [ ] `include/tenzor/nn/layers/transformer.hpp`
- [ ] `src/nn/layers/transformer.cpp`

**Tests:**
- [ ] `tests/unit/test_transformer.cpp`

**Python Bindings:**
- [ ] All transformer components

---

### 7.2.3 Positional Encoding (5 hours)

- [ ] PositionalEncoding class
  - [ ] Sinusoidal encoding
  - [ ] Learned encoding option
  - [ ] max_len parameter
  - [ ] Dropout

---

## 7.3 Advanced CNN Layers (MEDIUM PRIORITY)

### 7.3.1 3D Convolutions (15 hours)

**For Video/3D Medical Imaging:**
- [ ] Conv3d layer
  - [ ] 5D tensors (N, C, D, H, W)
  - [ ] 3D kernels
  - [ ] CPU implementation (im2col extension)
  - [ ] CUDA implementation

- [ ] ConvTranspose3d (deconvolution)
- [ ] MaxPool3d, AvgPool3d
- [ ] BatchNorm3d

**Tests:**
- [ ] `tests/unit/test_conv3d.cpp`

**Python Bindings:**
- [ ] All 3D layers

---

### 7.3.2 Depthwise Separable Convolution (10 hours)

**For MobileNet/EfficientNet:**
- [ ] DepthwiseConv2d
  - [ ] groups = in_channels
  - [ ] Optimized CUDA kernel

- [ ] SeparableConv2d
  - [ ] Depthwise + pointwise (1x1)
  - [ ] Single module combining both

**Files:**
- [ ] `include/tenzor/nn/layers/depthwise_conv.hpp`

---

### 7.3.3 Dilated/Atrous Convolution (5 hours)

**Already has `dilation` parameter in Conv2d, verify:**
- [ ] Test dilated convolutions
- [ ] CUDA kernel optimization for dilation
- [ ] Examples demonstrating receptive field

---

## 7.4 Normalization Layers (MEDIUM PRIORITY)

### 7.4.1 GroupNorm (8 hours)

- [ ] GroupNorm class
  - [ ] num_groups, num_channels
  - [ ] eps, affine parameters
  - [ ] Works for small batch sizes

**Files:**
- [ ] `include/tenzor/nn/layers/normalization.hpp` (extend)
- [ ] CPU/CUDA kernels

**Tests:**
- [ ] `tests/unit/test_normalization.cpp`

**Python Bindings:**
- [ ] GroupNorm binding

---

### 7.4.2 InstanceNorm (8 hours)

- [ ] InstanceNorm1d, InstanceNorm2d
  - [ ] Per-instance normalization
  - [ ] Style transfer applications

---

### 7.4.3 LocalResponseNorm (5 hours)

- [ ] LRN (for AlexNet compatibility)
  - [ ] size, alpha, beta, k parameters

---

## 7.5 Advanced Pooling (LOW PRIORITY)

### 7.5.1 Fractional Max Pooling (8 hours)

- [ ] FractionalMaxPool2d
  - [ ] Random pooling regions

---

### 7.5.2 LP Pooling (5 hours)

- [ ] LPPool2d
  - [ ] Generalized pooling (p-norm)

---

## 7.6 Additional Optimizers (MEDIUM PRIORITY)

### Status: 3/10 optimizers implemented

### 7.6.1 RMSprop (10 hours)

- [ ] RMSprop class
  - [ ] lr, alpha, eps, weight_decay
  - [ ] momentum option
  - [ ] centered option
  - [ ] Squared gradient moving average

**Files:**
- [ ] `include/tenzor/nn/optim/rmsprop.hpp`
- [ ] `src/nn/optim/rmsprop.cpp`

**Tests:**
- [ ] `tests/unit/test_optimizers.cpp` (extend)

**Python Bindings:**
- [ ] RMSprop binding

---

### 7.6.2 Adagrad (8 hours)

- [ ] Adagrad class
  - [ ] lr, lr_decay, weight_decay, eps
  - [ ] Accumulated squared gradients

---

### 7.6.3 Adadelta (8 hours)

- [ ] Adadelta class
  - [ ] rho (decay rate), eps
  - [ ] No learning rate required

---

### 7.6.4 AdaMax (5 hours)

- [ ] AdaMax class (Adam variant)
  - [ ] Infinity norm instead of L2

---

### 7.6.5 Nadam (5 hours)

- [ ] Nadam class (Adam + Nesterov)

---

### 7.6.6 LAMB (8 hours)

**For large-batch training:**
- [ ] LAMB optimizer
  - [ ] Layer-wise adaptive rate
  - [ ] Trust ratio computation

---

### 7.6.7 LBFGS (15 hours)

**Second-order optimizer:**
- [ ] LBFGS class
  - [ ] Line search
  - [ ] History size
  - [ ] Requires closure (re-evaluation)

---

## 7.7 Learning Rate Schedulers (MEDIUM PRIORITY)

### Status: 3/10 schedulers implemented

### 7.7.1 ReduceLROnPlateau (10 hours)

- [ ] ReduceLROnPlateau class
  - [ ] mode (min/max)
  - [ ] factor, patience, threshold
  - [ ] Metric-based reduction
  - [ ] cooldown, min_lr

**Files:**
- [ ] `include/tenzor/nn/optim/scheduler.hpp` (extend)
- [ ] `src/nn/optim/scheduler.cpp` (extend)

---

### 7.7.2 CyclicLR (8 hours)

- [ ] CyclicLR class
  - [ ] base_lr, max_lr
  - [ ] step_size_up, step_size_down
  - [ ] mode (triangular, triangular2, exp_range)
  - [ ] gamma (for exp_range)

---

### 7.7.3 OneCycleLR (10 hours)

- [ ] OneCycleLR class
  - [ ] max_lr, total_steps
  - [ ] pct_start, anneal_strategy
  - [ ] div_factor, final_div_factor

---

### 7.7.4 CosineAnnealingWarmRestarts (8 hours)

- [ ] CosineAnnealingWarmRestarts
  - [ ] T_0 (initial restart period)
  - [ ] T_mult (period multiplier)
  - [ ] eta_min

---

### 7.7.5 MultiStepLR (5 hours)

- [ ] MultiStepLR class
  - [ ] milestones (list of epochs)
  - [ ] gamma (decay factor)

---

### 7.7.6 LambdaLR (5 hours)

- [ ] LambdaLR class
  - [ ] Custom function for LR computation

---

### 7.7.7 WarmupScheduler (8 hours)

**Wrapper for warmup:**
- [ ] Linear warmup
- [ ] Exponential warmup
- [ ] Combines with another scheduler

---

## 7.8 Advanced Activation Functions (LOW PRIORITY)

### Status: 9/15 activations implemented

### 7.8.1 PReLU (5 hours)

- [ ] PReLU class
  - [ ] num_parameters (1 or num_channels)
  - [ ] init (initial value for alpha)
  - [ ] Learnable negative slope

---

### 7.8.2 SELU (3 hours)

- [ ] SELU activation
  - [ ] Self-normalizing property
  - [ ] Alpha and lambda constants

---

### 7.8.3 GLU (Gated Linear Unit) (5 hours)

- [ ] GLU class
  - [ ] dim parameter
  - [ ] Split input and apply gate

---

### 7.8.4 Swish/SiLU Variants (3 hours)

- [ ] HardSwish (for MobileNetV3)
- [ ] Verify SiLU implementation

---

### 7.8.5 Mish Verification (2 hours)

- [ ] Verify current Mish implementation
- [ ] Add tests

---

### 7.8.6 Additional Activations (5 hours each)

- [ ] Softplus
- [ ] Softsign
- [ ] Tanhshrink
- [ ] RReLU (Randomized ReLU)
- [ ] ReLU6 (for quantization)
- [ ] CELU

---

## 7.9 Embedding Layers (MEDIUM PRIORITY for NLP)

### 7.9.1 Embedding (10 hours)

- [ ] Embedding class
  - [ ] num_embeddings, embedding_dim
  - [ ] padding_idx
  - [ ] max_norm, norm_type
  - [ ] scale_grad_by_freq
  - [ ] sparse gradients option

**Files:**
- [ ] `include/tenzor/nn/layers/embedding.hpp`
- [ ] `src/nn/layers/embedding.cpp`

**Tests:**
- [ ] `tests/unit/test_embedding.cpp`

**Python Bindings:**
- [ ] Embedding binding

---

### 7.9.2 EmbeddingBag (8 hours)

- [ ] EmbeddingBag class
  - [ ] mode (sum, mean, max)
  - [ ] Efficient bag-of-words

---

## 7.10 Loss Functions (MEDIUM PRIORITY)

### Status: 7/15 losses implemented

### 7.10.1 KLDivLoss (5 hours)

- [ ] KLDivLoss class
  - [ ] reduction parameter
  - [ ] KL divergence computation

---

### 7.10.2 Hinge Losses (8 hours)

- [ ] HingeEmbeddingLoss
- [ ] MultiLabelMarginLoss
- [ ] MultiMarginLoss

---

### 7.10.3 Ranking Losses (10 hours)

- [ ] MarginRankingLoss
- [ ] TripletMarginLoss
- [ ] CosineEmbeddingLoss

---

### 7.10.4 CTC Loss (15 hours)

**For speech recognition:**
- [ ] CTCLoss class
  - [ ] blank label
  - [ ] zero_infinity option
  - [ ] Dynamic programming algorithm

---

### 7.10.5 Focal Loss (8 hours)

**For imbalanced classification:**
- [ ] FocalLoss class
  - [ ] alpha, gamma parameters
  - [ ] Addresses class imbalance

---

### 7.10.6 Dice Loss (5 hours)

**For segmentation:**
- [ ] DiceLoss class
  - [ ] Soft Dice coefficient
  - [ ] smooth parameter

---

### 7.10.7 Huber Loss (5 hours)

- [ ] HuberLoss class
  - [ ] delta parameter
  - [ ] Robust regression

---

# SPARC Phase 8: Advanced Features & Optimizations

## 8.1 Mixed Precision Training (HIGH PRIORITY)

### Status: NOT IMPLEMENTED → Target: Complete AMP Support

### 8.1.1 FP16 Support (20 hours)

**DType Extensions:**
- [ ] Verify Float16, BFloat16 in DType enum
- [ ] Complete dtype_traits for Float16, BFloat16
- [ ] Ensure all ops support FP16/BF16

**CUDA Kernels:**
- [ ] Half precision kernels
  - [ ] Use __half type
  - [ ] Tensor Core utilization (if available)

**CPU Support:**
- [ ] FP16 emulation on CPU (conversion to FP32)

**Files to Modify:**
- [ ] `include/tenzor/core/dtype.hpp`
- [ ] All CUDA kernels (add __half support)

---

### 8.1.2 Automatic Mixed Precision (25 hours)

**GradScaler:**
- [ ] `include/tenzor/nn/amp/grad_scaler.hpp`
- [ ] Loss scaling to prevent underflow
- [ ] Dynamic loss scale adjustment
- [ ] Gradient unscaling before optimizer step
- [ ] Inf/NaN gradient detection

**Autocast Context:**
- [ ] `autocast` context manager (Python)
- [ ] Automatic FP16/FP32 dtype selection
- [ ] Op-specific dtype policies
  - [ ] GEMM/Conv in FP16
  - [ ] Softmax/LayerNorm in FP32
  - [ ] Loss computation in FP32

**Implementation:**
```cpp
class GradScaler {
public:
    GradScaler(float init_scale = 65536.0f,
               float growth_factor = 2.0f,
               float backoff_factor = 0.5f,
               int growth_interval = 2000);

    auto scale(const Variable& loss) -> Variable;
    auto unscale_(Optimizer& optimizer) -> void;
    auto step(Optimizer& optimizer) -> void;
    auto update() -> void;
};
```

**Tests:**
- [ ] `tests/unit/test_amp.cpp`
- [ ] Gradient scaling tests
- [ ] Inf/NaN detection tests

**Python Bindings:**
- [ ] GradScaler binding
- [ ] autocast context manager

**Files to Create:**
- [ ] `include/tenzor/nn/amp/grad_scaler.hpp`
- [ ] `src/nn/amp/grad_scaler.cpp`

---

### 8.1.3 Tensor Core Utilization (15 hours)

**CUDA Optimizations:**
- [ ] Use Tensor Cores for GEMM operations
  - [ ] cuBLAS with Tensor Core mode
  - [ ] Ensure proper alignment (multiple of 8)

- [ ] Tensor Cores for convolutions
  - [ ] cuDNN with Tensor Core algorithms
  - [ ] Benchmark performance gains

**Verification:**
- [ ] Benchmark suite comparing FP32 vs FP16+TC
- [ ] Expected: 2-3x speedup on V100/A100

---

## 8.2 Kernel Fusion Optimizations (MEDIUM PRIORITY)

### Status: NOT IMPLEMENTED → Target: 20-30% Speedup

### 8.2.1 Element-wise Fusion (20 hours)

**Common Patterns:**
- [ ] Fused Linear + ReLU
- [ ] Fused Linear + GELU
- [ ] Fused BatchNorm + ReLU
- [ ] Fused Add + ReLU

**Implementation:**
- [ ] `include/tenzor/nn/fused/fused_ops.hpp`
- [ ] CUDA kernels for fused operations
- [ ] CPU fallback (call ops sequentially)

**Graph Optimizer:**
- [ ] Pattern matching for fusion opportunities
- [ ] Automatic fusion in eager mode (optional)

**Files to Create:**
- [ ] `include/tenzor/nn/fused/fused_ops.hpp`
- [ ] `src/backends/cuda/kernels/fused_ops.cu`

---

### 8.2.2 Memory Optimization - Operation Fusion (15 hours)

**Fused Softmax + Cross-Entropy:**
- [ ] LogSoftmax + NLLLoss in single kernel
- [ ] Prevents intermediate tensor allocation
- [ ] Numerically stable computation

**Fused LayerNorm:**
- [ ] Mean, variance, normalize in one kernel
- [ ] Single pass over data

---

## 8.3 Memory Management (HIGH PRIORITY)

### Status: Basic Allocator → Target: Advanced Caching

### 8.3.1 Caching Allocator (25 hours)

**CUDA Caching Allocator:**
- [ ] `include/tenzor/backend/caching_allocator.hpp`
- [ ] Memory pool per device
- [ ] Block splitting and merging
- [ ] Fragmentation reduction
- [ ] Memory statistics tracking

**Features:**
- [ ] Reuse freed memory blocks
- [ ] Configurable block sizes
- [ ] Empty cache API
- [ ] Memory snapshot for debugging

**Implementation:**
```cpp
class CachingAllocator {
public:
    auto allocate(size_t size, int device_id) -> void*;
    auto free(void* ptr, int device_id) -> void;
    auto empty_cache() -> void;
    auto memory_allocated() const -> size_t;
    auto memory_reserved() const -> size_t;
};
```

**Tests:**
- [ ] Allocation/deallocation patterns
- [ ] Memory leak detection
- [ ] Fragmentation tests

**Files to Create:**
- [ ] `include/tenzor/backend/caching_allocator.hpp`
- [ ] `src/backend/caching_allocator.cpp`

---

### 8.3.2 Memory Profiling Tools (10 hours)

- [ ] `memory_snapshot()` API
  - [ ] Current allocations
  - [ ] Allocation history
  - [ ] Peak memory usage

- [ ] `memory_stats()` function
  - [ ] Per-device statistics
  - [ ] Cache hit rate

**Python Bindings:**
- [ ] Expose memory profiling APIs

---

## 8.4 Multi-GPU & Distributed Training (HIGH PRIORITY)

### Status: NOT IMPLEMENTED → Target: DataParallel + DDP

### 8.4.1 DataParallel (30 hours)

**Single-Machine Multi-GPU:**
- [ ] `include/tenzor/nn/parallel/data_parallel.hpp`

**Features:**
- [ ] Model replication across GPUs
- [ ] Input batch splitting
- [ ] Forward pass on each GPU
- [ ] Gradient gathering to GPU 0
- [ ] Gradient averaging
- [ ] Parameter update on GPU 0
- [ ] Broadcast updated parameters

**Implementation:**
```cpp
class DataParallel : public Module {
public:
    DataParallel(std::shared_ptr<Module> module,
                 std::vector<int> device_ids,
                 int output_device = 0);

    auto forward(const Variable& input) -> Variable override;
};
```

**Backend Support:**
- [ ] CUDA P2P transfers (if available)
- [ ] Efficient gradient allreduce
- [ ] Asynchronous transfers

**Tests:**
- [ ] `tests/integration/test_data_parallel.cpp`
- [ ] 2-GPU training
- [ ] Gradient correctness tests

**Python Bindings:**
- [ ] DataParallel wrapper

**Files to Create:**
- [ ] `include/tenzor/nn/parallel/data_parallel.hpp`
- [ ] `src/nn/parallel/data_parallel.cpp`

---

### 8.4.2 DistributedDataParallel (40 hours)

**Multi-Machine Multi-GPU (Requires NCCL):**
- [ ] NCCL integration
  - [ ] AllReduce for gradients
  - [ ] Broadcast for parameters
  - [ ] Efficient cross-node communication

- [ ] Process group management
- [ ] Rank and world size tracking
- [ ] Gradient bucketing for efficiency

**Files to Create:**
- [ ] `include/tenzor/nn/parallel/distributed.hpp`
- [ ] `src/nn/parallel/distributed.cpp`

**External Dependencies:**
- [ ] NCCL library integration
- [ ] MPI for process initialization (optional)

---

### 8.4.3 Gradient Checkpointing (15 hours)

**Memory-Computation Tradeoff:**
- [ ] `checkpoint()` API
- [ ] Discard intermediate activations during forward
- [ ] Recompute during backward
- [ ] Configurable checkpoint segments

**Python API:**
```python
def checkpoint(function, *args):
    # Saves memory by recomputing forward pass
    pass
```

---

## 8.5 Performance Optimizations (MEDIUM PRIORITY)

### 8.5.1 SIMD Runtime Dispatch (15 hours)

**CPU Feature Detection:**
- [ ] Detect AVX-512, AVX2, SSE4.2 at runtime
- [ ] Dispatch to optimal kernel
- [ ] Fallback to scalar code

**Implementation:**
- [ ] `include/tenzor/backend/cpu_features.hpp`
- [ ] CPUID-based detection (x86)
- [ ] Function pointer dispatch table

**Files to Create:**
- [ ] `include/tenzor/backend/cpu_features.hpp`
- [ ] `src/backend/cpu_features.cpp`

---

### 8.5.2 cuBLAS/cuDNN Integration (20 hours)

**cuBLAS:**
- [ ] Use cuBLAS for GEMM operations
  - [ ] Currently partially integrated, verify all paths

**cuDNN:**
- [ ] Integrate cuDNN for convolutions
  - [ ] Forward/backward convolution
  - [ ] Algorithm selection (performance tuning)
  - [ ] Workspace management

- [ ] cuDNN for RNN/LSTM
- [ ] cuDNN for BatchNorm
- [ ] cuDNN for Pooling, Softmax, Activation

**Files to Modify:**
- [ ] `src/backends/cuda/cuda_backend.cpp`
- [ ] Add cuDNN handle management

---

### 8.5.3 JIT Compilation (Advanced - 40+ hours)

**Lazy Execution Mode:**
- [ ] Build computation graph
- [ ] Optimize graph (fusion, reordering)
- [ ] Generate optimized code
- [ ] LLVM-based code generation (long-term)

---

### 8.5.4 Benchmark Suite (15 hours)

**Comprehensive Benchmarks:**
- [ ] `benchmarks/matmul_benchmark.cpp`
  - [ ] Various matrix sizes
  - [ ] CPU vs CUDA
  - [ ] Compare with PyTorch

- [ ] `benchmarks/conv2d_benchmark.cpp`
  - [ ] ResNet layer configs
  - [ ] Measure TFLOPS

- [ ] `benchmarks/training_benchmark.cpp`
  - [ ] End-to-end training time
  - [ ] ResNet-50, BERT

**Integration:**
- [ ] Google Benchmark framework
- [ ] Automated CI benchmarking
- [ ] Performance regression detection

**Files to Create:**
- [ ] `benchmarks/CMakeLists.txt`
- [ ] `benchmarks/matmul_benchmark.cpp`
- [ ] `benchmarks/conv2d_benchmark.cpp`
- [ ] `benchmarks/rnn_benchmark.cpp`

---

## 8.6 Model Serialization & Checkpointing (MEDIUM PRIORITY)

### Status: Basic save/load → Target: Advanced Checkpointing

### 8.6.1 Enhanced Serialization (10 hours)

**Current:** Basic `save()` and `load()` in Module

**Enhancements:**
- [ ] Versioning support
- [ ] Backward compatibility
- [ ] Optimizer state serialization
- [ ] Scheduler state serialization
- [ ] Metadata (training epoch, best accuracy, etc.)

**Checkpoint Format:**
- [ ] HDF5 format (optional)
- [ ] Custom binary format
- [ ] JSON metadata + binary weights

---

### 8.6.2 ModelCheckpoint Utility (8 hours)

**Training Utilities:**
- [ ] `include/tenzor/nn/utils/checkpoint.hpp`

**Features:**
- [ ] Auto-save best model (by metric)
- [ ] Save every N epochs
- [ ] Keep only top K checkpoints
- [ ] EarlyStopping integration

**Implementation:**
```cpp
class ModelCheckpoint {
public:
    ModelCheckpoint(std::string filepath,
                    std::string monitor = "val_loss",
                    std::string mode = "min",
                    int save_top_k = 1);

    auto on_epoch_end(int epoch, const std::map<std::string, float>& metrics) -> void;
};
```

**Python Bindings:**
- [ ] ModelCheckpoint class

---

## 8.7 Data Loading & Augmentation (HIGH PRIORITY)

### Status: NOT IMPLEMENTED → Target: DataLoader + Augmentation

### 8.7.1 Dataset Interface (15 hours)

**C++ Dataset:**
- [ ] `include/tenzor/data/dataset.hpp`

**Features:**
- [ ] Dataset abstract base class
- [ ] `__len__()` and `__getitem__()` interface
- [ ] Map-style datasets
- [ ] Iterable-style datasets

**Python Integration:**
- [ ] Python dataset wrapper
- [ ] Inherit from Python classes

---

### 8.7.2 DataLoader (25 hours)

**Multi-threaded Data Loading:**
- [ ] `include/tenzor/data/dataloader.hpp`

**Features:**
- [ ] Batching
- [ ] Shuffling
- [ ] Multi-threaded workers
- [ ] Pin memory (for CUDA)
- [ ] Collate function
- [ ] Prefetching

**Implementation:**
```cpp
class DataLoader {
public:
    DataLoader(Dataset& dataset,
               size_t batch_size,
               bool shuffle = false,
               size_t num_workers = 0);

    auto begin() -> Iterator;
    auto end() -> Iterator;
};
```

**Python Bindings:**
- [ ] DataLoader class
- [ ] Pythonic iteration

**Files to Create:**
- [ ] `include/tenzor/data/dataset.hpp`
- [ ] `include/tenzor/data/dataloader.hpp`
- [ ] `src/data/dataloader.cpp`

---

### 8.7.3 Data Augmentation (20 hours)

**Image Transforms:**
- [ ] `include/tenzor/data/transforms.hpp`

**Transformations:**
- [ ] RandomCrop
- [ ] RandomHorizontalFlip
- [ ] RandomVerticalFlip
- [ ] RandomRotation
- [ ] ColorJitter (brightness, contrast, saturation, hue)
- [ ] Normalize (mean, std)
- [ ] Resize
- [ ] CenterCrop
- [ ] ToTensor (convert image to tensor)

**Compose:**
- [ ] Compose class to chain transforms

**Files to Create:**
- [ ] `include/tenzor/data/transforms.hpp`
- [ ] `src/data/transforms.cpp`

---

## 8.8 Visualization & Debugging Tools (LOW PRIORITY)

### 8.8.1 TensorBoard Integration (20 hours)

**Logging:**
- [ ] Scalar logging (loss, accuracy)
- [ ] Histogram logging (weights, gradients)
- [ ] Image logging
- [ ] Graph visualization

**Files to Create:**
- [ ] `include/tenzor/utils/tensorboard.hpp`
- [ ] External: protobuf integration

---

### 8.8.2 Gradient Checking (10 hours)

**Numerical Gradient Verification:**
- [ ] `gradcheck()` function
- [ ] Finite difference approximation
- [ ] Compare with autograd gradients
- [ ] Configurable epsilon

**Files to Create:**
- [ ] `include/tenzor/autograd/gradcheck.hpp`

---

# SPARC Phase 9: Model Zoo & Pretrained Models

## 9.1 Computer Vision Models (MEDIUM PRIORITY)

### 9.1.1 ResNet Family (20 hours)

**Implementations:**
- [ ] ResNet-18, ResNet-34
- [ ] ResNet-50, ResNet-101, ResNet-152
- [ ] ResNeXt variants
- [ ] Wide ResNet

**Files to Create:**
- [ ] `include/tenzor/models/resnet.hpp`
- [ ] `src/models/resnet.cpp`

**Pretrained Weights:**
- [ ] ImageNet-1k pretrained
- [ ] Weight loading from URL/file

---

### 9.1.2 VGG, AlexNet, GoogleNet (15 hours)

- [ ] VGG-11, VGG-13, VGG-16, VGG-19
- [ ] AlexNet
- [ ] GoogLeNet (Inception v1)

---

### 9.1.3 Modern Architectures (30 hours)

- [ ] EfficientNet family
- [ ] Vision Transformer (ViT)
- [ ] Swin Transformer
- [ ] ConvNeXt
- [ ] MobileNet v2, v3

---

### 9.1.4 Detection & Segmentation (40+ hours)

- [ ] Faster R-CNN
- [ ] Mask R-CNN
- [ ] YOLO variants
- [ ] U-Net
- [ ] DeepLab

---

## 9.2 NLP Models (HIGH PRIORITY)

### 9.2.1 BERT (30 hours)

**Requires:**
- [ ] Transformer layers (from Phase 7.2)
- [ ] Positional encoding
- [ ] Token type embeddings

**Implementation:**
- [ ] BertModel
- [ ] BertForSequenceClassification
- [ ] BertForTokenClassification
- [ ] BertForQuestionAnswering

**Pretrained:**
- [ ] Load Hugging Face weights

---

### 9.2.2 GPT Family (25 hours)

- [ ] GPT-2
- [ ] GPT-3 (architecture, training separately)
- [ ] Text generation utilities

---

### 9.2.3 Other Transformers (30 hours)

- [ ] RoBERTa
- [ ] ALBERT
- [ ] T5
- [ ] ELECTRA

---

## 9.3 Pretrained Weight Management (15 hours)

**Weight Hub:**
- [ ] Download from URLs
- [ ] Caching to local directory
- [ ] Checksum verification
- [ ] Progress bars

**Integration:**
```python
model = tenzor.models.resnet50(pretrained=True)  # Auto-downloads
```

---

# SPARC Phase 10: Ecosystem & Interoperability

## 10.1 ONNX Support (MEDIUM PRIORITY)

### Status: NOT IMPLEMENTED → Target: Export/Import

### 10.1.1 ONNX Export (30 hours)

**Model Export:**
- [ ] Convert Tenzor model to ONNX format
- [ ] Operator mapping (Tenzor ops → ONNX ops)
- [ ] Shape inference
- [ ] Dynamic shapes support

**Files to Create:**
- [ ] `include/tenzor/onnx/exporter.hpp`
- [ ] `src/onnx/exporter.cpp`

**External Dependencies:**
- [ ] ONNX library integration

**API:**
```cpp
auto export_onnx(Module& model,
                 const Tensor& dummy_input,
                 const std::string& filename,
                 bool verbose = false) -> void;
```

**Python Bindings:**
- [ ] `tenzor.onnx.export()` function

---

### 10.1.2 ONNX Import (25 hours)

**Model Import:**
- [ ] Load ONNX model
- [ ] Convert to Tenzor Module
- [ ] Weight loading

**Files to Create:**
- [ ] `include/tenzor/onnx/importer.hpp`

---

## 10.2 TorchScript/JIT Support (Advanced - 40+ hours)

**JIT Compilation:**
- [ ] Trace mode (record operations)
- [ ] Script mode (parse Python code)
- [ ] Optimize graph
- [ ] Serialize compiled model

---

## 10.3 Quantization (Advanced - 50+ hours)

### 10.3.1 Post-Training Quantization (30 hours)

**INT8 Quantization:**
- [ ] Dynamic quantization
- [ ] Static quantization (calibration)
- [ ] Quantization-aware training

**Files to Create:**
- [ ] `include/tenzor/quantization/quantize.hpp`

---

### 10.3.2 Quantized Operations (20 hours)

- [ ] Quantized Conv2d, Linear
- [ ] INT8 GEMM kernels
- [ ] Fake quantization for training

---

## 10.4 Model Compression (MEDIUM PRIORITY)

### 10.4.1 Pruning (20 hours)

**Structured Pruning:**
- [ ] Channel pruning
- [ ] Layer pruning

**Unstructured Pruning:**
- [ ] Weight pruning (magnitude-based)
- [ ] Iterative pruning

**Files to Create:**
- [ ] `include/tenzor/compression/pruning.hpp`

---

### 10.4.2 Knowledge Distillation (15 hours)

**Teacher-Student Training:**
- [ ] Soft target loss
- [ ] Temperature scaling
- [ ] Distillation utilities

---

# SPARC Phase 11: Additional Backend Support

## 11.1 ROCm Backend (HIGH PRIORITY for AMD GPUs)

### Status: STUB → Target: Full Implementation

### 11.1.1 HIP Kernels (60 hours)

**Convert CUDA to HIP:**
- [ ] Rename CUDA APIs to HIP equivalents
- [ ] Test on AMD GPUs (MI100, MI200)

**Kernels to Implement:**
- [ ] All kernels from `src/backends/cuda/kernels/`
- [ ] Conv2d, BatchNorm, Pooling
- [ ] Activations, reductions, transforms
- [ ] LSTM, GRU (if implemented)

**Files to Create:**
- [ ] `src/backends/rocm/kernels/*.hip`

**Build System:**
- [ ] HIP compiler integration in CMake
- [ ] Conditional compilation

---

### 11.1.2 rocBLAS Integration (10 hours)

- [ ] Link rocBLAS library
- [ ] GEMM operations
- [ ] Batched GEMM

---

### 11.1.3 MIOpen Integration (15 hours)

**AMD's DNN library:**
- [ ] Convolution algorithms
- [ ] BatchNorm, Pooling
- [ ] Activation functions

---

## 11.2 OneAPI Backend (MEDIUM PRIORITY for Intel GPUs)

### Status: STUB → Target: Full Implementation

### 11.2.1 SYCL Kernels (70 hours)

**Intel GPU Support:**
- [ ] SYCL kernel implementations
- [ ] All operations from CPU/CUDA

**Files to Create:**
- [ ] `src/backends/oneapi/kernels/*.cpp` (SYCL)

---

### 11.2.2 oneDNN Integration (15 hours)

**Intel's DNN library:**
- [ ] Convolution primitives
- [ ] BatchNorm, Pooling
- [ ] RNN/LSTM

---

## 11.3 Metal Backend (LOW PRIORITY for macOS)

### 11.3.1 Metal Shaders (60 hours)

**For Apple Silicon (M1/M2):**
- [ ] Metal shader implementations
- [ ] GPU acceleration on Macs

---

## 11.4 WebGPU Backend (LOW PRIORITY)

### 11.4.1 WebGPU Kernels (50 hours)

**Browser-based Inference:**
- [ ] WebGPU compute shaders
- [ ] WASM bindings

---

# SPARC Phase 12: Testing & Quality Assurance

## 12.1 Expand Test Coverage (ONGOING)

### Current: 448 tests, 85-90% coverage → Target: 95%+

### 12.1.1 Unit Tests for New Features (40 hours)

**For every new feature added, include:**
- [ ] Forward pass tests
- [ ] Backward pass tests (gradient checking)
- [ ] Edge case tests (empty tensors, broadcasting, etc.)
- [ ] Device tests (CPU/CUDA/ROCm/OneAPI)
- [ ] Dtype tests (Float32, Float64, Float16)

---

### 12.1.2 Integration Tests (20 hours)

- [ ] End-to-end training scenarios
- [ ] Multi-GPU training tests
- [ ] Serialization round-trip tests
- [ ] ONNX export/import tests

---

### 12.1.3 Performance Regression Tests (15 hours)

- [ ] Benchmark suite in CI
- [ ] Alert on performance degradation (>5%)

---

## 12.2 Continuous Integration (10 hours)

**GitHub Actions / GitLab CI:**
- [ ] Build on Linux, macOS, Windows
- [ ] CPU-only builds
- [ ] CUDA builds (if runners available)
- [ ] Run all tests automatically
- [ ] Code coverage reports
- [ ] Static analysis (clang-tidy, cppcheck)
- [ ] Documentation generation

**Files to Create:**
- [ ] `.github/workflows/ci.yml`

---

## 12.3 Static Analysis & Linting (5 hours)

- [ ] clang-tidy configuration
- [ ] clang-format for code style
- [ ] Address sanitizer (ASan)
- [ ] Memory sanitizer (MSan)
- [ ] Thread sanitizer (TSan)

---

# Priority Summary & Estimated Timeline

## Critical Path to v1.0 (6-8 weeks, 220 hours)

| Task | Priority | Hours | Dependencies |
|------|----------|-------|--------------|
| Complete Python bindings | 🔴 CRITICAL | 80 | None |
| NumPy interoperability | 🔴 CRITICAL | 40 | Python bindings |
| API documentation (Doxygen) | 🔴 CRITICAL | 60 | None |
| Tutorial examples | 🔴 CRITICAL | 40 | Python bindings, NumPy |

**Total for v1.0:** 220 hours (~6 weeks with 1 person, ~2 weeks with 3 people)

---

## High Priority for v1.1-v1.2 (3-6 months, 400+ hours)

| Feature Category | Priority | Hours | Timeline |
|------------------|----------|-------|----------|
| RNN/LSTM/GRU | 🟠 HIGH | 80 | v1.1 |
| Transformer layers | 🟠 HIGH | 60 | v1.1 |
| Mixed precision (AMP) | 🟠 HIGH | 60 | v1.1 |
| DataLoader + augmentation | 🟠 HIGH | 60 | v1.1 |
| Additional optimizers (5) | 🟡 MEDIUM | 40 | v1.2 |
| Additional schedulers (5) | 🟡 MEDIUM | 40 | v1.2 |
| Multi-GPU (DataParallel) | 🟠 HIGH | 30 | v1.2 |
| Caching allocator | 🟡 MEDIUM | 25 | v1.2 |

**Total for v1.1-v1.2:** ~400 hours

---

## Medium Priority for v1.3-v2.0 (6-12 months, 600+ hours)

| Feature Category | Priority | Hours | Timeline |
|------------------|----------|-------|----------|
| ROCm backend | 🟠 HIGH | 85 | v1.3 |
| ONNX export/import | 🟡 MEDIUM | 55 | v1.3 |
| Kernel fusion | 🟡 MEDIUM | 35 | v1.4 |
| 3D convolutions | 🟡 MEDIUM | 15 | v1.4 |
| Embedding layers | 🟡 MEDIUM | 18 | v1.5 |
| Advanced loss functions | 🟡 MEDIUM | 40 | v1.5 |
| Model zoo (ResNet, VGG, BERT) | 🟡 MEDIUM | 100 | v1.6 |
| Model compression (pruning) | 🟡 MEDIUM | 35 | v2.0 |
| OneAPI backend | 🟡 MEDIUM | 100 | v2.0 |
| Quantization | 🟡 MEDIUM | 50 | v2.0 |

**Total for v1.3-v2.0:** ~600 hours

---

## Low Priority / Future Work (12+ months)

| Feature Category | Priority | Hours | Timeline |
|------------------|----------|-------|----------|
| JIT compilation | 🟢 LOW | 80+ | v3.0 |
| TorchScript support | 🟢 LOW | 60 | v3.0 |
| Metal backend (macOS) | 🟢 LOW | 60 | v3.0 |
| WebGPU backend | 🟢 LOW | 50 | v3.0 |
| Distributed training (DDP+NCCL) | 🟡 MEDIUM | 60 | v2.5 |
| TensorBoard integration | 🟢 LOW | 20 | v2.x |

---

# Discrepancies Found (From DESIGN.md Comparison)

## 1. Python Bindings Gaps (CRITICAL)

**Issue:** Only 40% of C++ API is accessible from Python

**Missing:**
- 60% of tensor operations (31/40 operations)
- 87% of NN components (27/31 layers, losses, activations)
- 100% of NumPy interoperability
- Sequential container
- Advanced optimizer features (state_dict, parameter groups)

**Impact:** Python users cannot use most of the library

**Recommendation:** Complete before v1.0 (80 hours)

---

## 2. NumPy Interoperability Missing (CRITICAL)

**Issue:** Zero-copy NumPy conversions not implemented

**Expected (from DESIGN.md line 15):** "Seamless NumPy interoperability with zero-copy conversions"

**Current:** No NumPy integration at all

**Impact:** Cannot integrate with SciPy ecosystem, major usability gap

**Recommendation:** Implement immediately (40 hours)

---

## 3. dtype_traits Incomplete (MINOR)

**Issue:** Only 8/13 DType enum values have trait specializations

**Missing:**
- Float16, BFloat16, Int8, Int16, UInt16, UInt32, UInt64

**Impact:** These types can't use `dtype_t<>` template

**Recommendation:** Complete all specializations (2 hours)

---

## 4. High-Level Training API Missing (MEDIUM)

**Issue:** No NeuralNetwork wrapper, DataLoader, or fit() method

**Expected (DESIGN.md Section 6.7):** Complete high-level training API

**Current:** Users must write manual training loops

**Impact:** More boilerplate code for users

**Recommendation:** Implement in v1.1 (60 hours)

---

## 5. Multi-GPU Support Missing (MEDIUM)

**Issue:** DataParallel not implemented

**Expected (DESIGN.md Section 7.4):** Multi-GPU training support

**Current:** Single GPU only

**Impact:** Cannot scale to multi-GPU efficiently

**Recommendation:** v1.2 release (30 hours)

---

## 6. Performance Optimizations Not Implemented (MEDIUM)

**Issue:** Several optimization features missing

**Missing:**
- Kernel fusion (GraphOptimizer, FusedLinearReLU)
- Caching allocator for memory pooling
- Runtime SIMD dispatch
- Benchmark suite

**Impact:** Performance may not meet targets

**Recommendation:** Incremental implementation based on profiling (v1.2-v1.4)

---

## 7. Advanced Features Not Started (LOW)

**Issue:** Phase 5 (Advanced Features) from roadmap not begun

**Missing:**
- Mixed precision training
- ONNX export
- Model compression
- Distributed training (NCCL)

**Impact:** Missing cutting-edge features, but acceptable for v1.0

**Recommendation:** Plan for v1.1+ (various timelines)

---

## 8. ROCm/OneAPI Backends are Stubs (MEDIUM)

**Issue:** AMD and Intel GPU support is stub-only

**Current:**
- ROCm: Interface ready, no HIP kernels
- OneAPI: Interface ready, no SYCL kernels

**Impact:** AMD and Intel GPU users cannot use Tenzor

**Recommendation:** ROCm in v1.3 (85 hours), OneAPI in v2.0 (100 hours)

---

## 9. Documentation Gaps (CRITICAL for v1.0)

**Issue:** ~0% Doxygen coverage

**Expected:** Complete API documentation for all 350+ APIs

**Current:** Inline comments exist but not in Doxygen format

**Impact:** No generated docs, poor IDE tooltips

**Recommendation:** 60 hours for core documentation before v1.0

---

## 10. Minimal Examples (BLOCKING v1.0)

**Issue:** Only 5 basic examples, need 30+

**Expected (DESIGN.md):** Complete MNIST example, various tutorials

**Current:** Basic examples without detailed documentation

**Impact:** Harder for new users to learn

**Recommendation:** 40 hours for comprehensive examples before v1.0

---

# Development Guidelines

## Code Quality Standards

**Every new feature MUST include:**
1. ✅ Unit tests (90%+ coverage)
2. ✅ Integration tests (if applicable)
3. ✅ Doxygen documentation
4. ✅ Python bindings (if user-facing)
5. ✅ Example usage code
6. ✅ Performance tests (for kernels)

**Code Review Checklist:**
- [ ] Follows C++23 best practices
- [ ] Memory safe (no leaks, RAII)
- [ ] Thread safe (where applicable)
- [ ] Proper error handling
- [ ] Consistent with existing API
- [ ] Documented with Doxygen
- [ ] Tests pass on CPU and CUDA

---

## Contribution Priorities

**For External Contributors:**

**Easy (Good First Issues):**
- Additional activation functions (2-5 hours each)
- Additional loss functions (3-8 hours each)
- Data augmentation transforms (3-5 hours each)
- Documentation improvements (1-5 hours each)
- Examples/tutorials (5-10 hours each)

**Medium:**
- Additional optimizers (8-15 hours each)
- Additional schedulers (5-10 hours each)
- Layer implementations (10-25 hours each)

**Hard:**
- Backend implementations (ROCm, OneAPI)
- ONNX export/import
- Multi-GPU support
- JIT compilation

---

## Continuous Tracking

**Update This Document:**
- ✅ Mark completed items
- 📅 Add new features as needed
- 🔄 Re-estimate hours based on actual time
- 🎯 Adjust priorities based on user feedback

**Quarterly Reviews:**
- Assess progress against roadmap
- Reprioritize based on community needs
- Update timelines

---

# Conclusion

This TODO roadmap provides a comprehensive path to making Tenzor a **world-class tensor and neural network library**. The critical path to v1.0 focuses on:

1. **Python ecosystem completion** (120 hours)
2. **Documentation** (100 hours)

Following v1.0, the roadmap spans 12-24 months of development across:
- Advanced neural network components (RNNs, Transformers)
- Performance optimizations (AMP, kernel fusion, multi-GPU)
- Additional backends (ROCm, OneAPI)
- Model zoo and pretrained models
- Ecosystem integration (ONNX, quantization)

**Estimated Total Effort:** ~2,000-2,500 hours for full "world-class" status

**With focused effort, Tenzor can achieve feature parity with PyTorch/TensorFlow within 18-24 months.**

---

**Last Updated:** 2025-10-10
**Maintainer:** Tenzor Development Team
**Status:** Living Document - Update Quarterly
