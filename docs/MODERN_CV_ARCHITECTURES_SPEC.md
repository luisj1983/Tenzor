# Modern Computer Vision Architectures - C++ Implementation Specification

**Version:** 1.0
**Date:** 2025-10-18
**Author:** Research Analysis
**Purpose:** Detailed architectural specifications for implementing modern CV models in C++ (Tenzor framework)

---

## Table of Contents

1. [EfficientNet Family (B0-B7)](#1-efficientnet-family-b0-b7)
2. [Vision Transformer (ViT)](#2-vision-transformer-vit)
3. [Swin Transformer](#3-swin-transformer)
4. [ConvNeXt](#4-convnext)
5. [MobileNet V2 and V3](#5-mobilenet-v2-and-v3)
6. [C++ Implementation Considerations](#6-c-implementation-considerations)
7. [Reference Implementations](#7-reference-implementations)

---

## 1. EfficientNet Family (B0-B7)

### 1.1 Overview

EfficientNet uses a compound scaling method that uniformly scales network width, depth, and resolution using a compound coefficient φ. All variants (B0-B7) are scaled from the baseline EfficientNet-B0.

**Official Paper:** "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" (2019)
**Authors:** Mingxing Tan, Quoc V. Le (Google Research)
**Paper URL:** https://arxiv.org/abs/1905.11946

### 1.2 Compound Scaling Coefficients

The compound scaling method uses the formula:
- **Depth:** d = α^φ
- **Width:** w = β^φ
- **Resolution:** r = γ^φ

Where:
- α = 1.2 (depth coefficient)
- β = 1.1 (width coefficient)
- γ = 1.15 (resolution coefficient)
- α · β² · γ² ≈ 2 (resource constraint)

### 1.3 EfficientNet-B0 Baseline Architecture

**Input Resolution:** 224×224×3
**Total Layers:** 237
**Parameters:** 5.3M
**FLOPs:** 390M

#### Stem Layer
- Conv2D: 3×3, stride 2, 32 filters
- Batch Normalization
- Swish activation

#### MBConv Block Configuration

| Stage | Operator | Resolution | Channels | Layers | Stride | Kernel | Expansion |
|-------|----------|------------|----------|--------|--------|--------|-----------|
| 1 | MBConv1 | 112×112 | 16 | 1 | 1 | 3×3 | 1 |
| 2 | MBConv6 | 112×112 | 24 | 2 | 2 | 3×3 | 6 |
| 3 | MBConv6 | 56×56 | 40 | 2 | 2 | 5×5 | 6 |
| 4 | MBConv6 | 28×28 | 80 | 3 | 2 | 3×3 | 6 |
| 5 | MBConv6 | 14×14 | 112 | 3 | 1 | 5×5 | 6 |
| 6 | MBConv6 | 14×14 | 192 | 4 | 2 | 5×5 | 6 |
| 7 | MBConv6 | 7×7 | 320 | 1 | 1 | 3×3 | 6 |

**Note:** MBConv1 = expansion ratio 1, MBConv6 = expansion ratio 6

#### Head Layer
- Conv2D: 1×1, 1280 filters
- Global Average Pooling
- Dropout (rate varies by model)
- Fully Connected: 1000 classes

### 1.4 MBConv Block Structure

```
Input
  ↓
[Optional: 1×1 Conv Expansion] (if expansion_ratio != 1)
  ↓
Batch Normalization
  ↓
Swish Activation
  ↓
Depthwise 3×3 or 5×5 Convolution
  ↓
Batch Normalization
  ↓
Swish Activation
  ↓
Squeeze-and-Excitation (SE) Module
  ↓
1×1 Conv Projection (to output channels)
  ↓
Batch Normalization
  ↓
[Optional: Stochastic Depth / Drop Connect]
  ↓
[Optional: Skip Connection] (if stride==1 and in_channels==out_channels)
  ↓
Output
```

### 1.5 Squeeze-and-Excitation (SE) Module

**SE Reduction Ratio:** 0.25 (fixed for all EfficientNet variants)

```
Input (C channels)
  ↓
Global Average Pooling → [1×1×C]
  ↓
FC (Dense): C → C/4
  ↓
Swish Activation
  ↓
FC (Dense): C/4 → C
  ↓
Sigmoid Activation
  ↓
Channel-wise Multiplication with Input
  ↓
Output
```

### 1.6 Variant Specifications (B0-B7)

| Model | φ | Resolution | Depth Scale | Width Scale | Dropout | Params | FLOPs |
|-------|---|------------|-------------|-------------|---------|--------|-------|
| B0 | 0 | 224 | 1.0 | 1.0 | 0.2 | 5.3M | 0.39B |
| B1 | 0.5 | 240 | 1.1 | 1.0 | 0.2 | 7.8M | 0.70B |
| B2 | 1 | 260 | 1.2 | 1.1 | 0.3 | 9.2M | 1.0B |
| B3 | 2 | 300 | 1.4 | 1.2 | 0.3 | 12M | 1.8B |
| B4 | 3 | 380 | 1.8 | 1.4 | 0.4 | 19M | 4.2B |
| B5 | 4 | 456 | 2.2 | 1.6 | 0.4 | 30M | 9.9B |
| B6 | 5 | 528 | 2.6 | 1.8 | 0.5 | 43M | 19B |
| B7 | 6 | 600 | 3.1 | 2.0 | 0.5 | 66M | 37B |

### 1.7 Key Hyperparameters

- **Activation:** Swish (also called SiLU): `x * sigmoid(x)`
- **Normalization:** Batch Normalization with momentum=0.99, epsilon=1e-3
- **Stochastic Depth:** Linear scaling from 0.0 to drop_connect_rate (0.2 for B0, 0.5 for B7)
- **SE Reduction Ratio:** 0.25 (applied to expansion layer channels)
- **Stem Filters:** 32 (scaled by width multiplier for B1-B7)
- **Head Filters:** 1280 (scaled by width multiplier for B1-B7)

### 1.8 C++ Implementation Considerations

1. **Depthwise Separable Convolutions:**
   - Implement as two separate operations: depthwise conv + pointwise conv
   - Use efficient CUDA kernels for depthwise operations
   - Consider cuDNN's grouped convolution with groups=channels

2. **Swish Activation:**
   - Fuse sigmoid and multiplication into single kernel
   - Cache sigmoid output for backward pass

3. **SE Module:**
   - Fuse global pooling and first FC layer
   - Use optimized GEMM operations for FC layers

4. **Stochastic Depth:**
   - Only during training
   - Implement as random layer skip with probability based on depth

5. **Memory Optimization:**
   - Reuse activation buffers between MBConv blocks
   - Use in-place operations where possible for Swish

---

## 2. Vision Transformer (ViT)

### 2.1 Overview

Vision Transformer applies pure transformer architecture to image classification by treating images as sequences of patches.

**Official Paper:** "An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale" (2020)
**Authors:** Alexey Dosovitskiy et al. (Google Research)
**Paper URL:** https://arxiv.org/abs/2010.11929

### 2.2 Architecture Variants

| Variant | Layers | Hidden Size | MLP Size | Heads | Params | Patch Size |
|---------|--------|-------------|----------|-------|--------|------------|
| ViT-Base | 12 | 768 | 3072 | 12 | 86M | 16 or 32 |
| ViT-Large | 24 | 1024 | 4096 | 16 | 307M | 16 or 32 |
| ViT-Huge | 32 | 1280 | 5120 | 16 | 632M | 14 or 16 |

**Naming Convention:** ViT-{Size}/{Patch_Size}
Example: ViT-B/16 = Base model with 16×16 patches

### 2.3 Detailed Architecture (ViT-Base/16)

**Input:** 224×224×3 image
**Patch Size:** 16×16
**Sequence Length:** (224/16)² = 196 patches + 1 CLS token = 197

#### Patch Embedding

```
Input Image: [B, 3, 224, 224]
  ↓
Patch Extraction: [B, 196, 16×16×3] = [B, 196, 768]
  ↓
Linear Projection: [B, 196, 768] → [B, 196, 768]
  ↓
Add CLS Token: [B, 197, 768]
  ↓
Add Position Embeddings: [B, 197, 768]
  ↓
To Transformer Encoder
```

**Implementation:**
- Can be implemented as Conv2D with kernel=16, stride=16, output_channels=768
- Or as reshape + linear projection

#### Position Embeddings

- **Type:** Learnable 1D position embeddings
- **Shape:** [197, 768] for ViT-Base/16
- **Initialization:** Random normal or truncated normal
- Element-wise addition to patch embeddings

#### CLS Token

- **Shape:** [1, 768]
- **Purpose:** Aggregates global image information for classification
- **Initialization:** Learnable parameter initialized to zeros
- Prepended to sequence before position embeddings

### 2.4 Transformer Encoder Block

**Repeated 12 times for ViT-Base, 24 for Large, 32 for Huge**

```
Input: [B, 197, 768]
  ↓
Layer Norm
  ↓
Multi-Head Self-Attention (MHSA)
  ↓
Residual Connection
  ↓
Layer Norm
  ↓
MLP (Feed-Forward Network)
  ↓
Residual Connection
  ↓
Output: [B, 197, 768]
```

### 2.5 Multi-Head Self-Attention (MHSA)

**For ViT-Base:**
- **Num Heads:** 12
- **Head Dim:** 768 / 12 = 64
- **QKV Bias:** True (include bias in Q, K, V projections)
- **Attention Dropout:** 0.0 (standard config)
- **Projection Dropout:** 0.0 (standard config)

```
Input: [B, 197, 768]
  ↓
Linear Projections (with bias):
  Q = Input @ W_q + b_q → [B, 197, 768]
  K = Input @ W_k + b_k → [B, 197, 768]
  V = Input @ W_v + b_v → [B, 197, 768]
  ↓
Reshape to Multi-Head:
  Q, K, V → [B, 12, 197, 64]
  ↓
Attention Scores:
  Scores = (Q @ K^T) / sqrt(64) → [B, 12, 197, 197]
  ↓
Softmax + Dropout
  ↓
Attention Output:
  Output = Scores @ V → [B, 12, 197, 64]
  ↓
Reshape: [B, 197, 768]
  ↓
Projection: Output @ W_o → [B, 197, 768]
  ↓
Dropout
```

### 2.6 MLP (Feed-Forward Network)

**MLP Ratio:** 4.0 (hidden_dim = 4 × embedding_dim)

```
Input: [B, 197, 768]
  ↓
Linear 1: [B, 197, 768] → [B, 197, 3072]
  ↓
GELU Activation
  ↓
Dropout (0.0 in standard config)
  ↓
Linear 2: [B, 197, 3072] → [B, 197, 768]
  ↓
Dropout (0.0 in standard config)
  ↓
Output: [B, 197, 768]
```

### 2.7 Classification Head

```
Transformer Output: [B, 197, 768]
  ↓
Extract CLS Token: [B, 768] (index 0)
  ↓
Layer Norm: [B, 768]
  ↓
Linear: [B, 768] → [B, num_classes]
  ↓
Output Logits
```

### 2.8 Key Hyperparameters

| Parameter | ViT-Base | ViT-Large | ViT-Huge |
|-----------|----------|-----------|----------|
| Image Size | 224 or 384 | 224 or 384 | 224 or 384 |
| Patch Size | 16 or 32 | 16 or 32 | 14 or 16 |
| Embedding Dim | 768 | 1024 | 1280 |
| Depth | 12 | 24 | 32 |
| Num Heads | 12 | 16 | 16 |
| MLP Ratio | 4.0 | 4.0 | 4.0 |
| QKV Bias | True | True | True |
| Dropout | 0.0 | 0.0 | 0.0 |
| Attention Dropout | 0.0 | 0.0 | 0.0 |
| Layer Norm Epsilon | 1e-6 | 1e-6 | 1e-6 |

### 2.9 C++ Implementation Considerations

1. **Patch Embedding:**
   - Use im2col + GEMM or direct Conv2D implementation
   - Efficient for 16×16 patches: single strided convolution

2. **Multi-Head Attention:**
   - Batch QKV computation: single GEMM with 3× output channels
   - Use cuBLAS for efficient matrix multiplication
   - Consider Flash Attention for memory efficiency (see section 6.2)
   - Fuse softmax with scaling operation

3. **Position Embeddings:**
   - Store as learnable tensor parameter
   - Simple element-wise addition after patch embedding

4. **Layer Normalization:**
   - Implement epsilon for numerical stability
   - Can be fused with adjacent operations
   - Use Welford's algorithm for numerical stability

5. **Memory Optimization:**
   - Attention scores [B, H, N, N] are largest memory consumer
   - For N=197, H=12: requires significant memory
   - Consider gradient checkpointing for training
   - Use mixed precision (FP16) where possible

6. **GELU Activation:**
   - Use approximation: `0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))`
   - Or use error function-based exact implementation

---

## 3. Swin Transformer

### 3.1 Overview

Swin Transformer is a hierarchical vision transformer that computes representations using shifted windows, providing linear complexity with respect to image size.

**Official Paper:** "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows" (ICCV 2021 Best Paper)
**Authors:** Ze Liu et al. (Microsoft Research Asia)
**Paper URL:** https://arxiv.org/abs/2103.14030

### 3.2 Architecture Variants

| Variant | Embed Dim (C) | Depths | Num Heads | Params | FLOPs@224 |
|---------|---------------|--------|-----------|--------|-----------|
| Swin-Tiny | 96 | [2,2,6,2] | [3,6,12,24] | 29M | 4.5G |
| Swin-Small | 96 | [2,2,18,2] | [3,6,12,24] | 50M | 8.7G |
| Swin-Base | 128 | [2,2,18,2] | [4,8,16,32] | 88M | 15.4G |
| Swin-Large | 192 | [2,2,18,2] | [6,12,24,48] | 197M | 34.5G |

**Note:** Depths = number of Swin Transformer blocks per stage

### 3.3 Overall Architecture (Swin-Tiny)

```
Input: 224×224×3
  ↓
Patch Partition (4×4): 56×56×48
  ↓
Linear Embedding: 56×56×96
  ↓
[Stage 1] 2× Swin Blocks (W-MSA, SW-MSA): 56×56×96
  ↓
Patch Merging: 28×28×192
  ↓
[Stage 2] 2× Swin Blocks (W-MSA, SW-MSA): 28×28×192
  ↓
Patch Merging: 14×14×384
  ↓
[Stage 3] 6× Swin Blocks (W-MSA, SW-MSA): 14×14×384
  ↓
Patch Merging: 7×7×768
  ↓
[Stage 4] 2× Swin Blocks (W-MSA, SW-MSA): 7×7×768
  ↓
Global Average Pooling + FC
```

### 3.4 Patch Partition and Linear Embedding

**Patch Size:** 4×4
**Output:** H/4 × W/4 patches, each with dimension 4×4×3 = 48

```
Input: [B, 3, 224, 224]
  ↓
Patch Partition: [B, 56, 56, 48]
  ↓
Linear Projection: [B, 56, 56, 96]
```

### 3.5 Patch Merging

Downsamples resolution by 2× and doubles channels:

```
Input: [B, H, W, C]
  ↓
Concatenate 2×2 neighbors: [B, H/2, W/2, 4C]
  ↓
Layer Norm
  ↓
Linear: 4C → 2C
  ↓
Output: [B, H/2, W/2, 2C]
```

### 3.6 Swin Transformer Block

**Two consecutive blocks always pair:**
1. W-MSA (Window-based Multi-head Self Attention)
2. SW-MSA (Shifted Window-based Multi-head Self Attention)

```
Input: [B, H, W, C]
  ↓
Layer Norm
  ↓
W-MSA or SW-MSA
  ↓
Residual Connection
  ↓
Layer Norm
  ↓
MLP
  ↓
Residual Connection
  ↓
Output: [B, H, W, C]
```

### 3.7 Window-based Multi-Head Self-Attention (W-MSA)

**Window Size (M):** 7 (fixed for all variants)
**Number of Windows:** (H/M) × (W/M)

**For 56×56 feature map:**
- Number of windows: (56/7) × (56/7) = 64 windows
- Each window: 7×7 = 49 tokens
- Attention complexity per window: O(49²) = O(2401)
- Total complexity: O(H×W) (linear!)

```
Input: [B, H, W, C]
  ↓
Partition into M×M windows: [B×num_windows, M×M, C]
  ↓
Multi-Head Self-Attention (within each window)
  ↓
Merge windows: [B, H, W, C]
```

### 3.8 Shifted Window MSA (SW-MSA)

**Shift Amount:** (M/2, M/2) = (3, 3) pixels

```
Input: [B, H, W, C]
  ↓
Cyclic Shift: shift by (-3, -3)
  ↓
Partition into windows
  ↓
Multi-Head Self-Attention with Attention Mask
  ↓
Merge windows
  ↓
Reverse Cyclic Shift: shift by (+3, +3)
  ↓
Output: [B, H, W, C]
```

**Attention Mask:** Prevents attention between non-adjacent regions after cyclic shift

### 3.9 Relative Position Bias

**Bias Table Shape:** (2M-1, 2M-1, num_heads)
For M=7: (13, 13, num_heads)

```
Attention = Softmax(QK^T / sqrt(d) + B)
```

Where B is the relative position bias indexed from the bias table based on relative positions between query and key.

**Relative Position Range:**
- x-axis: [-M+1, M-1] = [-6, 6]
- y-axis: [-M+1, M-1] = [-6, 6]

### 3.10 MLP Architecture

**MLP Ratio:** 4.0

```
Input: [B, H, W, C]
  ↓
Linear: C → 4C
  ↓
GELU
  ↓
Dropout
  ↓
Linear: 4C → C
  ↓
Dropout
  ↓
Output: [B, H, W, C]
```

### 3.11 Key Hyperparameters

| Parameter | Value |
|-----------|-------|
| Window Size (M) | 7 |
| MLP Ratio | 4.0 |
| QKV Bias | True |
| Dropout | 0.0 |
| Attention Dropout | 0.0 |
| Drop Path Rate | 0.1-0.5 (linear scaling) |
| Layer Norm Epsilon | 1e-5 |
| Patch Size | 4 |
| Query Dimension per Head | 32 |

**Drop Path (Stochastic Depth):**
- Linearly scaled from 0.0 to drop_path_rate across depth
- Swin-T/S: 0.2, Swin-B: 0.5, Swin-L: 0.5

### 3.12 Stage-wise Specifications (Swin-Tiny)

| Stage | Resolution | Channels | Num Blocks | Num Heads | Head Dim |
|-------|------------|----------|------------|-----------|----------|
| 1 | 56×56 | 96 | 2 | 3 | 32 |
| 2 | 28×28 | 192 | 2 | 6 | 32 |
| 3 | 14×14 | 384 | 6 | 12 | 32 |
| 4 | 7×7 | 768 | 2 | 24 | 32 |

### 3.13 C++ Implementation Considerations

1. **Shifted Window Attention:**
   - Implement efficient cyclic shift using memory copy or index manipulation
   - Pre-compute attention masks for SW-MSA
   - Batch all windows together for parallel attention computation

2. **Relative Position Bias:**
   - Pre-compute relative position indices (H×W, H×W) → relative_coords
   - Store as lookup table for efficient indexing
   - Learnable bias table shared across all blocks in a stage

3. **Window Partitioning:**
   - Use tensor reshaping and permutation
   - Avoid actual data copying where possible
   - Optimize for contiguous memory access

4. **Attention Mask Implementation:**
   - Add large negative value (-100.0) to masked positions before softmax
   - Pre-compute and cache masks (only a few unique patterns)

5. **Memory Optimization:**
   - Window attention is memory-efficient (7×7 = 49 tokens per window)
   - Reuse attention score buffers across windows
   - Consider fusing cyclic shift with window partitioning

6. **CUDA Optimization:**
   - Parallelize attention computation across windows (independent)
   - Use warp-level operations for small window sizes
   - Flash Window Attention: recent optimization (arxiv:2501.06480)

---

## 4. ConvNeXt

### 4.1 Overview

ConvNeXt modernizes the standard ResNet architecture by incorporating design choices from Vision Transformers while remaining purely convolutional.

**Official Paper:** "A ConvNet for the 2020s" (CVPR 2022)
**Authors:** Zhuang Liu et al. (Facebook AI Research, UC Berkeley)
**Paper URL:** https://arxiv.org/abs/2201.03545

### 4.2 Architecture Variants

| Variant | Channels (C) | Blocks (B) | Params | FLOPs@224 |
|---------|--------------|------------|--------|-----------|
| ConvNeXt-Tiny | [96,192,384,768] | [3,3,9,3] | 28M | 4.5G |
| ConvNeXt-Small | [96,192,384,768] | [3,3,27,3] | 50M | 8.7G |
| ConvNeXt-Base | [128,256,512,1024] | [3,3,27,3] | 89M | 15.4G |
| ConvNeXt-Large | [192,384,768,1536] | [3,3,27,3] | 198M | 34.4G |
| ConvNeXt-XLarge | [256,512,1024,2048] | [3,3,27,3] | 350M | 60.9G |

**Note:** B = blocks per stage, C = channels at each of 4 stages

### 4.3 Overall Architecture (ConvNeXt-Tiny)

```
Input: 224×224×3
  ↓
Stem: Conv 4×4, stride=4 → 56×56×96
  ↓
LayerNorm
  ↓
[Stage 1] 3× ConvNeXt Blocks: 56×56×96
  ↓
Downsample: LN + Conv 2×2, stride=2 → 28×28×192
  ↓
[Stage 2] 3× ConvNeXt Blocks: 28×28×192
  ↓
Downsample: LN + Conv 2×2, stride=2 → 14×14×384
  ↓
[Stage 3] 9× ConvNeXt Blocks: 14×14×384
  ↓
Downsample: LN + Conv 2×2, stride=2 → 7×7×768
  ↓
[Stage 4] 3× ConvNeXt Blocks: 7×7×768
  ↓
Global Average Pooling
  ↓
Layer Norm
  ↓
FC: 768 → num_classes
```

### 4.4 ConvNeXt Block Structure

```
Input: [B, C, H, W]
  ↓
Depthwise Conv 7×7: [B, C, H, W]
  ↓
LayerNorm (channels_first)
  ↓
Pointwise Conv 1×1: C → 4C (Expansion)
  ↓
GELU
  ↓
Pointwise Conv 1×1: 4C → C (Projection)
  ↓
Layer Scale (gamma * x)
  ↓
Stochastic Depth (Drop Path)
  ↓
Residual Connection
  ↓
Output: [B, C, H, W]
```

**Key Differences from ResNet:**
1. Larger kernel size (7×7 vs 3×3)
2. Depthwise convolution moved to the beginning
3. Inverted bottleneck (expand then compress)
4. LayerNorm instead of BatchNorm
5. GELU instead of ReLU
6. Layer Scale for stable training
7. Fewer activation functions (only one GELU per block)

### 4.5 Depthwise Convolution

**Kernel Size:** 7×7
**Groups:** Equal to number of input channels (depthwise)
**Padding:** 3 (to maintain spatial dimensions)

```
For input [B, C, H, W]:
  - Use C groups with 1 filter per group
  - Each filter: 7×7×1
  - Output: [B, C, H, W]
```

### 4.6 Layer Normalization

**Format:** channels_first (NCHW)
**Epsilon:** 1e-6

```
For input [B, C, H, W]:
  - Compute mean and variance over [C, H, W] dimensions
  - Normalize: (x - mean) / sqrt(var + eps)
  - Apply learnable affine: gamma * x + beta
```

### 4.7 Layer Scale

**Initial Value:** 1e-6 (0.000001)

```
gamma = learnable_parameter(shape=[C, 1, 1], init=1e-6)
output = gamma * features
```

**Purpose:** Improves training stability, especially for deeper models

### 4.8 Downsampling Layer

```
Input: [B, C, H, W]
  ↓
LayerNorm
  ↓
Conv 2×2, stride=2: C → 2C
  ↓
Output: [B, 2C, H/2, W/2]
```

### 4.9 Stem Layer

**Aggressive downsampling (similar to ViT):**

```
Input: [B, 3, 224, 224]
  ↓
Conv 4×4, stride=4, C filters
  ↓
Output: [B, C, 56, 56]
  ↓
LayerNorm
```

**Initial Channels (C):**
- Tiny/Small: 96
- Base: 128
- Large: 192
- XLarge: 256

### 4.10 Key Hyperparameters

| Parameter | Value |
|-----------|-------|
| Depthwise Kernel Size | 7×7 |
| Expansion Ratio | 4.0 |
| Layer Scale Init | 1e-6 |
| Drop Path Rate | 0.0-0.5 (linear scaling) |
| Layer Norm Epsilon | 1e-6 |
| Stem Kernel/Stride | 4×4, stride=4 |
| Downsample Kernel/Stride | 2×2, stride=2 |
| Activation | GELU |

**Drop Path (Stochastic Depth):**
- Linearly scaled from 0.0 to drop_path_rate across depth
- Typical values: 0.1 (Tiny/Small), 0.4-0.5 (Base/Large/XLarge)

### 4.11 Stage Compute Distribution

ConvNeXt follows Swin Transformer's 1:1:3:1 ratio for blocks:

```
Stage 1: 3 blocks  (1 part)
Stage 2: 3 blocks  (1 part)
Stage 3: 9 blocks  (3 parts)
Stage 4: 3 blocks  (1 part)
```

### 4.12 C++ Implementation Considerations

1. **Depthwise Convolution:**
   - Use cuDNN's grouped convolution with groups=channels
   - 7×7 kernel requires careful padding and memory access
   - Consider Winograd or FFT-based convolution for large kernels
   - Optimize data layout for cache efficiency

2. **Layer Normalization:**
   - More computationally expensive than BatchNorm
   - Requires computing statistics over spatial and channel dimensions
   - Can be fused with adjacent Conv operations
   - Use Welford's online algorithm for numerical stability

3. **Inverted Bottleneck:**
   - Pointwise (1×1) convolutions → use optimized GEMM
   - Expansion increases memory temporarily (4× channels)
   - Fuse LayerNorm + Conv1×1 where possible

4. **GELU Activation:**
   - Use tanh approximation for efficiency
   - Can be fused with pointwise convolution

5. **Layer Scale:**
   - Simple element-wise multiplication
   - Can be fused with previous convolution
   - Very small initial value (1e-6) requires FP32 precision

6. **Memory Optimization:**
   - Inverted bottleneck temporarily uses 4× memory
   - Gradient checkpointing for training deep variants
   - Reuse buffers for expanded features

7. **Comparison to ResNet:**
   - Similar structure, easier to optimize than transformers
   - More memory-efficient than ViT (no attention)
   - Standard CNN optimization techniques apply

---

## 5. MobileNet V2 and V3

### 5.1 Overview

MobileNets are a family of efficient CNNs designed for mobile and embedded vision applications, using depthwise separable convolutions and inverted residual blocks.

**MobileNetV2 Paper:** "MobileNetV2: Inverted Residuals and Linear Bottlenecks" (CVPR 2018)
**Authors:** Mark Sandler et al. (Google)
**Paper URL:** https://arxiv.org/abs/1801.04381

**MobileNetV3 Paper:** "Searching for MobileNetV3" (ICCV 2019)
**Authors:** Andrew Howard et al. (Google)
**Paper URL:** https://arxiv.org/abs/1905.02244

---

## 5.2 MobileNetV2

### 5.2.1 Architecture Overview

**Parameters:** 3.4M
**FLOPs:** 300M (width=1.0, resolution=224)
**Input:** 224×224×3

### 5.2.2 Inverted Residual Block (MBConv)

Unlike traditional residuals (wide → narrow → wide), MobileNetV2 uses inverted residuals (narrow → wide → narrow):

```
Input: [B, H, W, C_in] (narrow)
  ↓
1×1 Conv Expansion: C_in → t*C_in (wide)
  ↓
BatchNorm + ReLU6
  ↓
3×3 Depthwise Conv: t*C_in → t*C_in
  ↓
BatchNorm + ReLU6
  ↓
1×1 Conv Projection: t*C_in → C_out (narrow)
  ↓
BatchNorm (NO activation!)
  ↓
[Residual Connection if stride==1 and C_in==C_out]
  ↓
Output: [B, H/s, W/s, C_out]
```

**Key Points:**
- **t:** expansion factor (usually 6)
- **Linear Bottleneck:** No activation after final projection
- **Residual:** Only when stride=1 and input/output channels match

### 5.2.3 MobileNetV2 Architecture Specification

| Layer | Operator | t | c | n | s | Output Size |
|-------|----------|---|---|---|---|-------------|
| 1 | Conv2D 3×3 | - | 32 | 1 | 2 | 112×112×32 |
| 2 | Bottleneck | 1 | 16 | 1 | 1 | 112×112×16 |
| 3 | Bottleneck | 6 | 24 | 2 | 2 | 56×56×24 |
| 4 | Bottleneck | 6 | 32 | 3 | 2 | 28×28×32 |
| 5 | Bottleneck | 6 | 64 | 4 | 2 | 14×14×64 |
| 6 | Bottleneck | 6 | 96 | 3 | 1 | 14×14×96 |
| 7 | Bottleneck | 6 | 160 | 3 | 2 | 7×7×160 |
| 8 | Bottleneck | 6 | 320 | 1 | 1 | 7×7×320 |
| 9 | Conv2D 1×1 | - | 1280 | 1 | 1 | 7×7×1280 |
| 10 | AvgPool 7×7 | - | - | 1 | - | 1×1×1280 |
| 11 | Conv2D 1×1 | - | k | 1 | - | 1×1×k |

**Legend:**
- **t:** expansion factor
- **c:** output channels
- **n:** number of repetitions
- **s:** stride (applied to first block only)
- **k:** number of classes

### 5.2.4 Key Features

**ReLU6 Activation:**
```
ReLU6(x) = min(max(0, x), 6)
```
Robust for low-precision (int8) quantization

**Linear Bottleneck:**
- Final 1×1 projection has NO activation
- Preserves information in low-dimensional space
- Critical for model expressiveness

**Width Multiplier (α):**
- Scales number of channels: C_out = α * C_base
- Common values: 0.35, 0.5, 0.75, 1.0, 1.3, 1.4
- Trades accuracy for efficiency

**Resolution Multiplier (ρ):**
- Scales input resolution
- Common values: 96, 128, 160, 192, 224

---

## 5.3 MobileNetV3

### 5.3.1 Key Improvements Over V2

1. **Hard-Swish (h-swish) Activation:** More efficient than Swish
2. **Squeeze-and-Excitation (SE) Modules:** Channel attention
3. **Network Architecture Search (NAS):** Automated architecture design
4. **Platform-Aware NAS:** Optimized for mobile CPUs
5. **Redesigned Expensive Layers:** Reduced latency in head

### 5.3.2 Hard-Swish Activation

**Swish:** `x * sigmoid(x)`
**Hard-Swish:** `x * ReLU6(x + 3) / 6`

```
h-swish(x) = {
    0,           if x ≤ -3
    x,           if x ≥ +3
    x(x+3)/6,    otherwise
}
```

**Advantages:**
- Avoids expensive sigmoid computation
- Quantization-friendly (piecewise linear)
- Nearly identical behavior to Swish
- Only used in deeper layers (computation cost amortized)

### 5.3.3 Hard-Sigmoid Activation (for SE)

**Sigmoid:** `1 / (1 + exp(-x))`
**Hard-Sigmoid:** `ReLU6(x + 3) / 6`

```
h-sigmoid(x) = {
    0,         if x ≤ -3
    1,         if x ≥ +3
    (x+3)/6,   otherwise
}
```

### 5.3.4 Squeeze-and-Excitation in MobileNetV3

**SE Reduction Ratio:** Fixed to expansion layer channels / 4

```
Input: [B, H, W, C_expanded]
  ↓
Global Average Pooling: [B, 1, 1, C_expanded]
  ↓
FC: C_expanded → C_expanded/4
  ↓
ReLU
  ↓
FC: C_expanded/4 → C_expanded
  ↓
Hard-Sigmoid
  ↓
Multiply with Input (channel-wise)
  ↓
Output: [B, H, W, C_expanded]
```

**Key Difference from MobileNetV2:**
- SE is applied AFTER depthwise conv, BEFORE projection
- SE bottleneck fixed to 1/4 of expansion channels
- Uses Hard-Sigmoid instead of Sigmoid

### 5.3.5 MobileNetV3-Large Architecture

**Parameters:** 5.4M
**FLOPs:** 219M
**Input:** 224×224×3

| Layer | Kernel | Exp | Out | SE | NL | Stride | Output Size |
|-------|--------|-----|-----|----|----|--------|-------------|
| conv2d | 3×3 | - | 16 | - | HS | 2 | 112×112×16 |
| bneck | 3×3 | 16 | 16 | - | RE | 1 | 112×112×16 |
| bneck | 3×3 | 64 | 24 | - | RE | 2 | 56×56×24 |
| bneck | 3×3 | 72 | 24 | - | RE | 1 | 56×56×24 |
| bneck | 5×5 | 72 | 40 | ✓ | RE | 2 | 28×28×40 |
| bneck | 5×5 | 120 | 40 | ✓ | RE | 1 | 28×28×40 |
| bneck | 5×5 | 120 | 40 | ✓ | RE | 1 | 28×28×40 |
| bneck | 3×3 | 240 | 80 | - | HS | 2 | 14×14×80 |
| bneck | 3×3 | 200 | 80 | - | HS | 1 | 14×14×80 |
| bneck | 3×3 | 184 | 80 | - | HS | 1 | 14×14×80 |
| bneck | 3×3 | 184 | 80 | - | HS | 1 | 14×14×80 |
| bneck | 3×3 | 480 | 112 | ✓ | HS | 1 | 14×14×112 |
| bneck | 3×3 | 672 | 112 | ✓ | HS | 1 | 14×14×112 |
| bneck | 5×5 | 672 | 160 | ✓ | HS | 2 | 7×7×160 |
| bneck | 5×5 | 960 | 160 | ✓ | HS | 1 | 7×7×160 |
| bneck | 5×5 | 960 | 160 | ✓ | HS | 1 | 7×7×160 |
| conv2d | 1×1 | - | 960 | - | HS | 1 | 7×7×960 |
| pool | 7×7 | - | - | - | - | 1 | 1×1×960 |
| conv2d | 1×1 | - | 1280 | - | HS | 1 | 1×1×1280 |
| conv2d | 1×1 | - | k | - | - | 1 | 1×1×k |

**Legend:**
- **Exp:** Expansion size (expanded channels)
- **Out:** Output channels
- **SE:** Squeeze-and-Excitation (✓ = present)
- **NL:** Non-linearity (RE=ReLU, HS=Hard-Swish)
- **k:** Number of classes

### 5.3.6 MobileNetV3-Small Architecture

**Parameters:** 2.9M
**FLOPs:** 66M
**Input:** 224×224×3

| Layer | Kernel | Exp | Out | SE | NL | Stride | Output Size |
|-------|--------|-----|-----|----|----|--------|-------------|
| conv2d | 3×3 | - | 16 | - | HS | 2 | 112×112×16 |
| bneck | 3×3 | 16 | 16 | ✓ | RE | 2 | 56×56×16 |
| bneck | 3×3 | 72 | 24 | - | RE | 2 | 28×28×24 |
| bneck | 3×3 | 88 | 24 | - | RE | 1 | 28×28×24 |
| bneck | 5×5 | 96 | 40 | ✓ | HS | 2 | 14×14×40 |
| bneck | 5×5 | 240 | 40 | ✓ | HS | 1 | 14×14×40 |
| bneck | 5×5 | 240 | 40 | ✓ | HS | 1 | 14×14×40 |
| bneck | 5×5 | 120 | 48 | ✓ | HS | 1 | 14×14×48 |
| bneck | 5×5 | 144 | 48 | ✓ | HS | 1 | 14×14×48 |
| bneck | 5×5 | 288 | 96 | ✓ | HS | 2 | 7×7×96 |
| bneck | 5×5 | 576 | 96 | ✓ | HS | 1 | 7×7×96 |
| bneck | 5×5 | 576 | 96 | ✓ | HS | 1 | 7×7×96 |
| conv2d | 1×1 | - | 576 | ✓ | HS | 1 | 7×7×576 |
| pool | 7×7 | - | - | - | - | 1 | 1×1×576 |
| conv2d | 1×1 | - | 1024 | - | HS | 1 | 1×1×1024 |
| conv2d | 1×1 | - | k | - | - | 1 | 1×1×k |

### 5.3.7 Efficient Last Stage

**Problem:** Original MobileNetV2 uses expensive 1×1 conv to high dimensions (1280 channels)

**Solution in V3:**
- Move final 1×1 expansion to AFTER global pooling
- Reduces spatial dimensions first (7×7 → 1×1)
- Then expand channels (saves ~7ms latency)

### 5.3.8 Key Hyperparameters

| Parameter | MobileNetV2 | MobileNetV3-Large | MobileNetV3-Small |
|-----------|-------------|-------------------|-------------------|
| Activation (early) | ReLU6 | ReLU | ReLU |
| Activation (late) | ReLU6 | Hard-Swish | Hard-Swish |
| SE Reduction | N/A | C_expanded / 4 | C_expanded / 4 |
| Dropout | 0.2 | 0.2 | 0.2 |
| Width Multiplier | 0.35-1.4 | 0.75-1.0 | 0.75-1.0 |
| BatchNorm momentum | 0.999 | 0.999 | 0.999 |
| BatchNorm epsilon | 1e-3 | 1e-3 | 1e-3 |

### 5.3.9 C++ Implementation Considerations

1. **Inverted Residual Structure:**
   - Expansion increases channels (6× typical)
   - Requires careful memory management
   - Use in-place operations where possible

2. **Depthwise Separable Convolutions:**
   - Split into depthwise + pointwise operations
   - Use cuDNN grouped convolution (groups=channels)
   - Optimize memory layout for depthwise conv
   - Consider specialized CUDA kernels (15× speedup possible)

3. **ReLU6 / Hard-Swish / Hard-Sigmoid:**
   - Implement as fused kernels with conv/matmul
   - ReLU6: `min(max(x, 0), 6)` - very efficient
   - Hard-Swish: `x * ReLU6(x + 3) / 6` - fuse multiply and divide
   - Hard-Sigmoid: `ReLU6(x + 3) / 6` - use in SE module

4. **Squeeze-and-Excitation:**
   - Global pooling can be fused with preceding conv
   - FC layers are small GEMMs (use cuBLAS)
   - Entire SE module can be fused into single kernel

5. **Linear Bottleneck:**
   - Final projection has NO activation
   - Important: don't apply activation here!
   - Preserves gradient flow and information

6. **Quantization Considerations:**
   - MobileNets designed for int8 quantization
   - ReLU6 helps with quantization (bounded output)
   - Hard-Swish/Hard-Sigmoid are quantization-friendly
   - Batch normalization can be folded into convolutions

7. **Memory Optimization:**
   - Inverted residual uses more memory during expansion
   - Reuse buffers for expanded features
   - Avoid allocating new memory for each layer
   - Consider gradient checkpointing for training

8. **Platform-Specific Optimizations:**
   - MobileNetV3 designed with mobile CPUs in mind
   - On GPU: batch operations for throughput
   - On mobile: optimize for latency and power
   - Use platform-specific libraries (NNAPI, Core ML, etc.)

---

## 6. C++ Implementation Considerations

### 6.1 General Architecture Design

#### 6.1.1 Layer Abstraction

```cpp
class Layer {
public:
    virtual Tensor forward(const Tensor& input) = 0;
    virtual Tensor backward(const Tensor& grad_output) = 0;
    virtual std::vector<Tensor> parameters() = 0;
};

class Module {
    std::vector<std::shared_ptr<Layer>> layers;
public:
    void add(std::shared_ptr<Layer> layer);
    Tensor forward(const Tensor& input);
};
```

#### 6.1.2 Memory Management

**Key Principles:**
1. **Pre-allocate buffers** for activations and gradients
2. **Reuse memory** across layers where possible
3. **Implement memory pools** for dynamic allocation
4. **Use in-place operations** when safe
5. **Reference counting** or smart pointers for automatic cleanup

```cpp
class MemoryPool {
    std::vector<void*> free_blocks;
    std::vector<void*> used_blocks;
public:
    void* allocate(size_t bytes);
    void free(void* ptr);
    void reset(); // Reuse all allocated memory
};
```

#### 6.1.3 Computational Graph

**For Automatic Differentiation:**

```cpp
class ComputeNode {
    Tensor data;
    Tensor grad;
    std::vector<ComputeNode*> parents;
    std::function<void()> backward_fn;
};

class AutogradEngine {
    std::vector<ComputeNode*> topological_order;
public:
    void backward(ComputeNode* loss);
};
```

---

### 6.2 Attention Mechanism Optimization

#### 6.2.1 Standard Attention Complexity

**Memory:** O(N² · d) for attention scores
**Compute:** O(N² · d) where N = sequence length, d = head dimension

**Problem:** For ViT-Base/16 (N=197, H=12, d=64):
- Attention scores: [B, 12, 197, 197] ≈ 465K elements per sample
- Becomes prohibitive for high-resolution images

#### 6.2.2 Flash Attention

**Key Idea:** Minimize DRAM ↔ SRAM transfers, compute attention in blocks

**Benefits:**
- 2-4× faster than standard attention
- Reduced memory usage (no materialization of full attention matrix)
- Exact attention (not an approximation)

**Algorithm:**
1. Divide Q, K, V into blocks
2. Load blocks into SRAM (on-chip fast memory)
3. Compute attention scores and outputs in blocks
4. Use online softmax to combine block results

**Implementation:**
- Requires careful CUDA kernel design
- Available in: PyTorch (torch.nn.functional.scaled_dot_product_attention)
- NVIDIA Cutlass library has primitives

**Reference:** "FlashAttention: Fast and Memory-Efficient Exact Attention" (2022)
**URL:** https://arxiv.org/abs/2205.14135

#### 6.2.3 Efficient Attention Implementation

```cpp
// Standard QKV computation
Tensor qkv = linear(input, W_qkv); // Single GEMM: [B,N,D] @ [D,3*D]
Tensor q = qkv[:, :, 0:D];
Tensor k = qkv[:, :, D:2*D];
Tensor v = qkv[:, :, 2*D:3*D];

// Reshape for multi-head
q = q.reshape({B, N, H, d}).transpose(1, 2); // [B,H,N,d]
k = k.reshape({B, N, H, d}).transpose(1, 2);
v = v.reshape({B, N, H, d}).transpose(1, 2);

// Attention scores
Tensor scores = (q @ k.transpose(-1, -2)) / sqrt(d); // [B,H,N,N]
Tensor attn = softmax(scores, dim=-1);
Tensor out = attn @ v; // [B,H,N,d]

// Reshape and project
out = out.transpose(1, 2).reshape({B, N, D});
out = linear(out, W_o);
```

**Optimizations:**
1. **Fused QKV:** Single GEMM instead of 3 separate ones
2. **Fused Softmax:** Combine scaling, exp, and normalization
3. **cuBLAS batched GEMM:** For Q@K^T and Attn@V
4. **Transpose Fusion:** Avoid explicit transpose operations

#### 6.2.4 Window Attention (Swin Transformer)

**Advantages:**
- Linear complexity: O(M² · N) where M=7 (window size)
- Windows can be processed in parallel
- Much less memory than global attention

**Implementation:**
```cpp
// Partition into windows
Tensor windows = window_partition(x, window_size=7); // [B*num_windows, 49, C]

// Compute attention within each window
Tensor attn_windows = window_attention(windows); // [B*num_windows, 49, C]

// Merge windows back
Tensor x = window_reverse(attn_windows, window_size=7); // [B, H, W, C]
```

**Shifted Window:**
```cpp
// Cyclic shift
if (shift_size > 0) {
    x = torch::roll(x, {-shift_size, -shift_size}, {1, 2});
}

// Apply attention with mask
Tensor attn = window_attention(x, mask=attn_mask);

// Reverse cyclic shift
if (shift_size > 0) {
    x = torch::roll(x, {shift_size, shift_size}, {1, 2});
}
```

---

### 6.3 Depthwise Separable Convolution

#### 6.3.1 Standard Implementation

**Depthwise Convolution:**
```cpp
// Use grouped convolution with groups = in_channels
Conv2D depthwise = Conv2D(
    in_channels=C,
    out_channels=C,
    kernel_size=K,
    groups=C,  // Key parameter!
    stride=s,
    padding=p
);
```

**Pointwise Convolution:**
```cpp
// Standard 1×1 convolution
Conv2D pointwise = Conv2D(
    in_channels=C_in,
    out_channels=C_out,
    kernel_size=1
);
```

#### 6.3.2 cuDNN Implementation

```cpp
// Configure cuDNN for depthwise convolution
cudnnConvolutionDescriptor_t conv_desc;
cudnnCreateConvolutionDescriptor(&conv_desc);
cudnnSetConvolutionGroupCount(conv_desc, in_channels);

// Forward pass
cudnnConvolutionForward(
    handle,
    &alpha,
    input_desc, input_data,
    filter_desc, filter_data,
    conv_desc,
    algo,
    workspace, workspace_size,
    &beta,
    output_desc, output_data
);
```

#### 6.3.3 Specialized CUDA Kernel

**For 3×3 depthwise:**
- Each thread computes one output element
- Load filter weights into shared memory (9 values per channel)
- Coalesce input memory access

**For 7×7 depthwise (ConvNeXt):**
- Use tiling to reuse input data
- Shared memory for input tile
- Warp-level primitives for reduction

**Potential Speedup:** 5-15× over naive implementation

#### 6.3.4 Optimization Techniques

1. **Winograd Algorithm:**
   - Reduces multiplication count for small kernels
   - Effective for 3×3 depthwise convolutions
   - Trade-off: increased addition count

2. **Im2Col + GEMM:**
   - Not ideal for depthwise (loses efficiency)
   - Better for pointwise (1×1) convolutions

3. **Direct Convolution:**
   - Specialized kernels for each kernel size
   - Optimize for specific input/output sizes
   - Use shared memory and register tiling

---

### 6.4 Normalization Layers

#### 6.4.1 Batch Normalization

**Formula:**
```
y = gamma * (x - mean) / sqrt(var + eps) + beta
```

**During Training:**
- Compute mean and variance over batch and spatial dimensions
- Update running statistics with momentum
- Normalize using batch statistics

**During Inference:**
- Use running mean and running variance
- Can be fused with preceding convolution

**Folding into Convolution:**
```cpp
// BN parameters
float scale = gamma / sqrt(running_var + eps);
float bias = beta - gamma * running_mean / sqrt(running_var + eps);

// Fused conv weights
W_fused = W_conv * scale;
b_fused = (b_conv * scale) + bias;
```

#### 6.4.2 Layer Normalization

**Formula:**
```
y = gamma * (x - mean) / sqrt(var + eps) + beta
```

**Difference from BN:**
- Normalize over features (channels + spatial for CNN)
- Independent per sample (no batch statistics)
- Same during training and inference

**Welford's Online Algorithm (Numerical Stability):**
```cpp
// Single pass computation
float mean = 0, M2 = 0;
for (int i = 0; i < n; i++) {
    float delta = x[i] - mean;
    mean += delta / (i + 1);
    M2 += delta * (x[i] - mean);
}
float variance = M2 / n;
```

**CUDA Implementation:**
- Use warp-level reduction for mean/variance
- Each block processes one sample
- Fuse normalization with affine transformation

#### 6.4.3 Group Normalization

**Used in:** Some variants of ConvNeXt, normalization-free networks

**Formula:** Same as LayerNorm, but applied per group of channels

```cpp
int channels_per_group = C / num_groups;
for (int g = 0; g < num_groups; g++) {
    // Normalize channels [g*CPG : (g+1)*CPG] together
}
```

---

### 6.5 Activation Functions

#### 6.5.1 GELU (Gaussian Error Linear Unit)

**Exact:**
```
GELU(x) = x * Φ(x)
where Φ(x) = 0.5 * (1 + erf(x / sqrt(2)))
```

**Tanh Approximation (faster):**
```
GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
```

**Sigmoid Approximation:**
```
GELU(x) ≈ x * sigmoid(1.702 * x)
```

**CUDA Implementation:**
```cpp
__device__ float gelu_tanh(float x) {
    float x3 = x * x * x;
    float inner = 0.7978845608f * (x + 0.044715f * x3);
    return 0.5f * x * (1.0f + tanhf(inner));
}
```

#### 6.5.2 Swish / SiLU

**Formula:**
```
Swish(x) = x * sigmoid(x)
```

**CUDA Implementation:**
```cpp
__device__ float swish(float x) {
    return x / (1.0f + expf(-x));
}
```

**Fused with Conv:**
- Compute conv output
- Apply Swish in same kernel (avoid separate pass)

#### 6.5.3 Hard-Swish

**Formula:**
```
h-swish(x) = x * ReLU6(x + 3) / 6
```

**CUDA Implementation:**
```cpp
__device__ float hard_swish(float x) {
    float relu6 = fminf(fmaxf(x + 3.0f, 0.0f), 6.0f);
    return x * relu6 / 6.0f;
}
```

**Optimization:**
- Multiply by 1/6 can be fused with preceding conv
- ReLU6 is very efficient (min/max operations)

#### 6.5.4 ReLU6

**Formula:**
```
ReLU6(x) = min(max(0, x), 6)
```

**CUDA Implementation:**
```cpp
__device__ float relu6(float x) {
    return fminf(fmaxf(x, 0.0f), 6.0f);
}
```

---

### 6.6 Efficient Convolution Implementations

#### 6.6.1 Im2Col + GEMM

**Traditional Approach:**
1. Im2Col: Unfold image patches into columns
2. GEMM: Matrix multiplication with weights
3. Col2Im (if needed): Reshape output

**Advantages:**
- Leverage highly optimized BLAS libraries (cuBLAS)
- Consistent performance across kernel sizes

**Disadvantages:**
- High memory overhead (Im2Col buffer)
- Memory bandwidth bound for small kernels

#### 6.6.2 Direct Convolution

**Specialized Kernels:**
- Optimized for specific kernel sizes (1×1, 3×3, 5×5, 7×7)
- Use shared memory and register tiling
- Minimize global memory access

**cuDNN Auto-Tuning:**
- cuDNN selects best algorithm based on:
  - Input/output dimensions
  - Kernel size
  - Hardware (GPU architecture)
- Use `cudnnGetConvolutionForwardAlgorithm_v7()`

#### 6.6.3 Winograd Convolution

**For 3×3 Kernels:**
- F(2×2, 3×3): Reduce multiplications by 2.25×
- F(4×4, 3×3): Reduce multiplications by 4×

**Trade-offs:**
- Increased additions
- Numerical stability concerns for deep networks
- More effective for smaller tile sizes

**cuDNN Support:**
- `CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD`
- `CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED`

#### 6.6.4 FFT-based Convolution

**For Large Kernels (≥7×7):**
- Use Fast Fourier Transform
- Convolution theorem: conv(x, w) = IFFT(FFT(x) ⊙ FFT(w))

**Libraries:**
- cuFFT (NVIDIA)
- FFTW (CPU)

**When to Use:**
- Large kernel sizes (7×7 or larger)
- Large spatial dimensions
- Not effective for small kernels or large batch sizes

---

### 6.7 Memory Optimization Strategies

#### 6.7.1 Gradient Checkpointing

**Idea:** Trade computation for memory
- Store only subset of activations during forward pass
- Recompute others during backward pass

**Implementation:**
```cpp
class CheckpointedModule {
    std::vector<Layer*> layers;
    int checkpoint_every_n;

    Tensor forward(Tensor x) {
        std::vector<Tensor> checkpoints;
        for (int i = 0; i < layers.size(); i++) {
            x = layers[i]->forward(x);
            if (i % checkpoint_every_n == 0) {
                checkpoints.push_back(x.detach()); // Save
            }
        }
        return x;
    }
};
```

**Trade-off:**
- 33% recomputation: save 1/3 memory
- Useful for training very deep networks (Swin-Large, ViT-Huge)

#### 6.7.2 In-Place Operations

**When Safe:**
- ReLU, ReLU6, Hard-Swish (element-wise, monotonic)
- Dropout during training
- Some normalization operations

**When NOT Safe:**
- Before residual connections (need original input)
- Non-monotonic activations (GELU, Swish) if gradients needed

**Example:**
```cpp
// In-place ReLU
__global__ void relu_inplace(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = fmaxf(data[idx], 0.0f);
    }
}
```

#### 6.7.3 Memory Pooling

**Pre-allocate Common Sizes:**
```cpp
class TensorPool {
    std::unordered_map<size_t, std::vector<Tensor>> free_tensors;

public:
    Tensor get(std::vector<int> shape) {
        size_t size = compute_size(shape);
        if (free_tensors[size].empty()) {
            return allocate_new(shape);
        } else {
            Tensor t = free_tensors[size].back();
            free_tensors[size].pop_back();
            return t.reshape(shape);
        }
    }

    void release(Tensor t) {
        free_tensors[t.size()].push_back(t);
    }
};
```

#### 6.7.4 Mixed Precision Training

**FP16 (Half Precision):**
- 2× memory reduction
- Faster computation on modern GPUs (Tensor Cores)
- Requires loss scaling to prevent underflow

**BF16 (Brain Float 16):**
- Same range as FP32, less precision
- No loss scaling needed
- Supported on newer GPUs (Ampere, Ada)

**Implementation:**
```cpp
// Weights in FP32, activations in FP16
Tensor forward(Tensor input_fp32) {
    Tensor input_fp16 = input_fp32.to(dtype=fp16);
    Tensor output_fp16 = conv_fp16(input_fp16, weights_fp32.to(fp16));
    return output_fp16.to(dtype=fp32);
}
```

---

### 6.8 Parallelization Strategies

#### 6.8.1 Data Parallelism

**Distribute batches across GPUs:**
```cpp
// Pseudo-code
for (int gpu = 0; gpu < num_gpus; gpu++) {
    Tensor batch_gpu = batch[gpu * batch_size_per_gpu :
                              (gpu+1) * batch_size_per_gpu];
    cudaSetDevice(gpu);
    Tensor output_gpu = model.forward(batch_gpu);
    loss_gpu = criterion(output_gpu, labels_gpu);
    gradients_gpu = loss_gpu.backward();
}

// All-reduce gradients
for (auto& param : model.parameters()) {
    all_reduce(param.grad); // Sum across GPUs
    param.grad /= num_gpus;
}
```

**Libraries:** NCCL (NVIDIA Collective Communications Library)

#### 6.8.2 Model Parallelism

**Split model across GPUs:**
- Useful when model doesn't fit on single GPU
- Pipeline parallelism: different layers on different GPUs
- Tensor parallelism: split tensor dimensions

**Example:**
```cpp
// ViT split across 2 GPUs
GPU0: Patch Embedding + Layers 0-5
GPU1: Layers 6-11 + Classification Head

Tensor forward(Tensor x) {
    x = x.to(device=0);
    x = layers_0_5(x);
    x = x.to(device=1); // Transfer
    x = layers_6_11(x);
    return classification_head(x);
}
```

#### 6.8.3 Kernel Fusion

**Combine Multiple Operations:**
- Reduce memory transfers
- Improve instruction-level parallelism
- Lower kernel launch overhead

**Example: Conv + BatchNorm + ReLU**
```cpp
__global__ void fused_conv_bn_relu(
    const float* input,
    const float* weights,
    const float* bn_gamma,
    const float* bn_beta,
    const float* bn_mean,
    const float* bn_var,
    float* output,
    int N, int C, int H, int W
) {
    // 1. Convolution
    float conv_out = 0.0f;
    for (...) {
        conv_out += input[...] * weights[...];
    }

    // 2. Batch Normalization
    float bn_out = bn_gamma[c] * (conv_out - bn_mean[c]) /
                   sqrtf(bn_var[c] + eps) + bn_beta[c];

    // 3. ReLU
    output[idx] = fmaxf(bn_out, 0.0f);
}
```

---

### 6.9 Quantization for Inference

#### 6.9.1 Post-Training Quantization (PTQ)

**INT8 Quantization:**
```cpp
// Calibration: collect statistics
std::pair<float, float> calibrate(Tensor activations) {
    float min_val = activations.min();
    float max_val = activations.max();
    return {min_val, max_val};
}

// Quantize
int8_t quantize(float x, float scale, int8_t zero_point) {
    return static_cast<int8_t>(round(x / scale) + zero_point);
}

// Dequantize
float dequantize(int8_t x_q, float scale, int8_t zero_point) {
    return (x_q - zero_point) * scale;
}
```

**Symmetric Quantization:**
```
scale = max(|min_val|, |max_val|) / 127
zero_point = 0
```

**Asymmetric Quantization:**
```
scale = (max_val - min_val) / 255
zero_point = round(-min_val / scale)
```

#### 6.9.2 Quantization-Aware Training (QAT)

**Simulate Quantization During Training:**
```cpp
Tensor fake_quantize(Tensor x, float scale, int8_t zero_point) {
    Tensor x_q = round(x / scale) + zero_point;
    x_q = clamp(x_q, -128, 127);
    Tensor x_dq = (x_q - zero_point) * scale;
    return x_dq; // Gradient flows through
}
```

#### 6.9.3 INT8 GEMM

**CUDA Implementation:**
```cpp
// Use NVIDIA Tensor Cores (INT8 → INT32 accumulation)
cublasGemmEx(
    handle,
    CUBLAS_OP_N, CUBLAS_OP_N,
    M, N, K,
    &alpha,
    B, CUDA_R_8I, ldb,
    A, CUDA_R_8I, lda,
    &beta,
    C, CUDA_R_32I, ldc,
    CUDA_R_32I,
    CUBLAS_GEMM_DEFAULT_TENSOR_OP
);
```

**Speedup:** 4-8× over FP32 on modern GPUs

---

### 6.10 Inference Optimization

#### 6.10.1 Operator Fusion

**Common Fusion Patterns:**
1. Conv + BN + ReLU
2. Linear + GELU
3. Attention (QKV + Softmax + Output)
4. SE Module (Pool + FC + Sigmoid + Multiply)

**Tools:**
- TensorRT (NVIDIA)
- ONNX Runtime
- Custom CUDA kernels

#### 6.10.2 Dynamic Shapes

**Problem:** Most models assume fixed input size

**Solutions:**
1. **Pad to fixed size:** Wastes computation
2. **Multiple model variants:** One per common size
3. **Dynamic kernels:** Handle variable sizes efficiently

**TensorRT Optimization Profiles:**
```cpp
// Define min, optimal, max input sizes
config->setProfileDimensions(
    input_name,
    min_dims,  // e.g., [1, 3, 224, 224]
    opt_dims,  // e.g., [8, 3, 224, 224]
    max_dims   // e.g., [32, 3, 224, 224]
);
```

#### 6.10.3 Caching and Pre-computation

**For Inference:**
- Fold BatchNorm into Conv weights
- Pre-compute relative position bias (Swin Transformer)
- Cache attention masks
- Pre-allocate all buffers

**Example:**
```cpp
class SwinTransformer {
    Tensor relative_position_bias_table; // Learnable
    Tensor relative_position_index; // Pre-computed

    void precompute_indices() {
        // Compute once, use in every forward pass
        for (int i = 0; i < window_size * window_size; i++) {
            for (int j = 0; j < window_size * window_size; j++) {
                int relative_coords_h = coords_h[i] - coords_h[j];
                int relative_coords_w = coords_w[i] - coords_w[j];
                relative_position_index[i][j] =
                    (relative_coords_h + window_size - 1) * (2*window_size - 1) +
                    (relative_coords_w + window_size - 1);
            }
        }
    }
};
```

---

## 7. Reference Implementations

### 7.1 Official Implementations

#### EfficientNet

**TensorFlow (Official):**
- Repository: https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet
- Author: Mingxing Tan (Google)
- Includes: AutoAugment, RandAugment, Mixup

**PyTorch (timm):**
- Repository: https://github.com/huggingface/pytorch-image-models
- File: `timm/models/efficientnet.py`
- Maintainer: Ross Wightman
- Includes: EfficientNet, EfficientNetV2, Lite variants

**Keras/TF:**
- `tf.keras.applications.EfficientNetB0` through `EfficientNetB7`
- Built-in, well-tested

#### Vision Transformer (ViT)

**JAX/Flax (Official):**
- Repository: https://github.com/google-research/vision_transformer
- Author: Google Research
- Includes: Pre-trained weights on ImageNet-21K

**PyTorch (timm):**
- Repository: https://github.com/huggingface/pytorch-image-models
- File: `timm/models/vision_transformer.py`
- Extensive variant support

**PyTorch (torchvision):**
- `torchvision.models.vit_b_16`, `vit_b_32`, `vit_l_16`, `vit_l_32`, `vit_h_14`
- Official PyTorch implementation

**Hugging Face Transformers:**
- `transformers.ViTModel`, `ViTForImageClassification`
- Unified interface with other transformers

#### Swin Transformer

**PyTorch (Official):**
- Repository: https://github.com/microsoft/Swin-Transformer
- Author: Microsoft Research Asia
- Includes: Training scripts, pre-trained weights, detection/segmentation

**Swin Transformer V2:**
- Repository: https://github.com/microsoft/Swin-Transformer (branch: main)
- Improvements: Scaled up to 3B parameters

**PyTorch (timm):**
- Repository: https://github.com/huggingface/pytorch-image-models
- File: `timm/models/swin_transformer.py`

#### ConvNeXt

**PyTorch (Official):**
- Repository: https://github.com/facebookresearch/ConvNeXt
- Author: Facebook AI Research
- File: `models/convnext.py`

**ConvNeXt V2:**
- Repository: https://github.com/facebookresearch/ConvNeXt-V2
- Improvements: FCMAE pre-training, better performance

**PyTorch (torchvision):**
- `torchvision.models.convnext_tiny`, `small`, `base`, `large`
- Pre-trained weights included

**Keras:**
- `tf.keras.applications.ConvNeXtTiny` through `ConvNeXtXLarge`

#### MobileNet V2

**TensorFlow (Official):**
- Repository: https://github.com/tensorflow/models/tree/master/research/slim/nets/mobilenet
- Author: Google Research

**PyTorch (torchvision):**
- `torchvision.models.mobilenet_v2`
- Pre-trained on ImageNet

**Keras:**
- `tf.keras.applications.MobileNetV2`

#### MobileNet V3

**TensorFlow (Official):**
- Repository: https://github.com/tensorflow/models/tree/master/research/slim/nets/mobilenet
- Includes: MobileNetV3-Large and MobileNetV3-Small

**PyTorch (torchvision):**
- `torchvision.models.mobilenet_v3_large`
- `torchvision.models.mobilenet_v3_small`

**PyTorch (timm):**
- Repository: https://github.com/huggingface/pytorch-image-models
- File: `timm/models/mobilenetv3.py`

---

### 7.2 Implementation Tutorials and Guides

#### EfficientNet

1. **Medium - Complete Architectural Details:**
   - URL: https://medium.com/data-science/complete-architectural-details-of-all-efficientnet-models-5fd5b736142
   - Comprehensive breakdown of all variants

2. **Keras Example - Fine-tuning:**
   - URL: https://keras.io/examples/vision/image_classification_efficientnet_fine_tuning/
   - Transfer learning tutorial

#### Vision Transformer

1. **Annotated PyTorch Implementation:**
   - URL: https://amaarora.github.io/posts/2021-01-18-ViT.html
   - Step-by-step explanation with code

2. **Hugging Face Tutorial:**
   - URL: https://huggingface.co/blog/fine-tune-vit
   - Fine-tuning ViT for custom datasets

3. **From Scratch (PyTorch):**
   - Repository: https://github.com/s-chh/PyTorch-Scratch-Vision-Transformer-ViT
   - Educational implementation

#### Swin Transformer

1. **Annotated Implementation:**
   - URL: https://amaarora.github.io/posts/2022-07-04-swintransformerv1.html
   - Detailed explanation of shifted windows

2. **Relative Position Bias Explanation:**
   - URL: https://medium.com/@ovularslan/breaking-down-swin-transformer-understanding-relative-position-bias-and-masked-self-attention-437d692ab7cf
   - Deep dive into key mechanisms

#### ConvNeXt

1. **Paper Walkthrough:**
   - URL: https://medium.com/augmented-startups/convnext-the-return-of-convolution-networks-e70cbe8dabcc
   - Modernization steps from ResNet

2. **From ResNet to ConvNeXt:**
   - URL: https://tokudayo.github.io/from-resnet-to-convnext-2/
   - Progressive transformation

#### MobileNet V2/V3

1. **MobileNetV2 Summary:**
   - URL: https://medium.com/codex/a-summary-of-the-mobilenetv2-inverted-residuals-and-linear-bottlenecks-paper-e19b187cb78a
   - Inverted residuals explained

2. **MobileNetV3 Implementation Guide:**
   - URL: https://medium.com/@RobuRishabh/understanding-and-implementing-mobilenetv3-422bd0bdfb5a
   - Hard-Swish and SE modules

---

### 7.3 Performance Benchmarking

#### Papers With Code

- EfficientNet: https://paperswithcode.com/method/efficientnet
- ViT: https://paperswithcode.com/method/vision-transformer
- Swin: https://paperswithcode.com/method/swin-transformer
- ConvNeXt: https://paperswithcode.com/method/convnext
- MobileNetV2: https://paperswithcode.com/method/mobilenetv2
- MobileNetV3: https://paperswithcode.com/method/mobilenetv3

**Includes:** Leaderboards, benchmark results, code implementations

---

### 7.4 C++ Deep Learning Frameworks

#### LibTorch (PyTorch C++ API)

**Website:** https://pytorch.org/cppdocs/
**Usage:** Load PyTorch models in C++

```cpp
#include <torch/torch.h>
#include <torch/script.h>

// Load torchscript model
torch::jit::script::Module module = torch::jit::load("model.pt");

// Inference
std::vector<torch::jit::IValue> inputs;
inputs.push_back(input_tensor);
auto output = module.forward(inputs).toTensor();
```

#### ONNX Runtime

**Website:** https://onnxruntime.ai/
**Usage:** Run ONNX models in C++

```cpp
#include <onnxruntime_cxx_api.h>

Ort::Env env;
Ort::SessionOptions session_options;
Ort::Session session(env, "model.onnx", session_options);

// Run inference
auto output_tensors = session.Run(
    Ort::RunOptions{nullptr},
    input_names.data(),
    input_tensors.data(),
    1,
    output_names.data(),
    1
);
```

#### TensorRT

**Website:** https://developer.nvidia.com/tensorrt
**Usage:** High-performance inference on NVIDIA GPUs

```cpp
#include <NvInfer.h>

// Build engine from ONNX
IBuilder* builder = createInferBuilder(logger);
INetworkDefinition* network = builder->createNetworkV2(0);
nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, logger);
parser->parseFromFile("model.onnx", ...);

// Build optimized engine
IBuilderConfig* config = builder->createBuilderConfig();
ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);

// Run inference
IExecutionContext* context = engine->createExecutionContext();
context->executeV2(buffers);
```

#### cuDNN

**Website:** https://developer.nvidia.com/cudnn
**Usage:** Low-level CUDA primitives for DNNs

**Key Functions:**
- `cudnnConvolutionForward`
- `cudnnBatchNormalizationForwardTraining`
- `cudnnActivationForward`
- `cudnnSoftmaxForward`
- `cudnnTransformerForward` (for attention)

#### Dlib

**Website:** http://dlib.net/
**Usage:** C++ machine learning library

**Features:**
- Pure C++ implementation
- CPU and CUDA support
- No external dependencies (except CUDA)

---

### 7.5 Recommended Development Stack for C++

#### For Research/Prototyping:
1. **PyTorch** → Design and train model
2. **Export to TorchScript** or **ONNX**
3. **Load in LibTorch or ONNX Runtime** for C++ inference

#### For Production:
1. **PyTorch** → Train model
2. **Export to ONNX**
3. **Optimize with TensorRT**
4. **Deploy with TensorRT or ONNX Runtime**

#### For Custom Kernels:
1. **CUDA** for low-level kernels
2. **cuDNN** for standard operations
3. **cuBLAS** for matrix operations
4. **Thrust** for parallel algorithms

---

## 8. Summary and Key Takeaways

### 8.1 Model Comparison

| Model | Type | Key Innovation | Complexity | Best Use Case |
|-------|------|---------------|------------|---------------|
| **EfficientNet** | CNN | Compound scaling | O(N²) spatial | Balanced accuracy/efficiency |
| **ViT** | Transformer | Patch-based attention | O(N²) sequence | High-capacity, large datasets |
| **Swin** | Transformer | Shifted windows | O(N) spatial | Hierarchical features, dense prediction |
| **ConvNeXt** | CNN | Modernized ResNet | O(N²) spatial | Pure CNN with transformer performance |
| **MobileNetV2** | CNN | Inverted residuals | O(N²) spatial | Mobile/embedded, efficiency |
| **MobileNetV3** | CNN | NAS + h-swish | O(N²) spatial | Ultra-efficient mobile deployment |

### 8.2 Implementation Priority (C++ Perspective)

**Easiest to Hardest:**

1. **ConvNeXt:** Pure CNN, standard operations, no special attention
2. **MobileNetV2/V3:** Depthwise separable convs, compact architecture
3. **EfficientNet:** SE modules, compound scaling, stochastic depth
4. **Swin Transformer:** Shifted window attention, relative position bias
5. **ViT:** Large attention matrices, heavy memory usage

### 8.3 Critical Implementation Components

**Must Implement:**
1. Efficient convolution kernels (cuDNN or custom CUDA)
2. Optimized matrix multiplication (cuBLAS)
3. Attention mechanisms (for transformers)
4. Normalization layers (BN, LN)
5. Modern activations (GELU, Swish, Hard-Swish)

**Nice to Have:**
1. Flash Attention (for large-scale transformers)
2. Gradient checkpointing (for training)
3. Mixed precision support (FP16/BF16)
4. Quantization (INT8 for inference)
5. Kernel fusion (for production)

### 8.4 Performance Optimization Checklist

- [ ] Use cuDNN for standard convolutions
- [ ] Implement efficient depthwise separable convolutions
- [ ] Optimize attention with Flash Attention or window-based approaches
- [ ] Fuse operations (Conv+BN+ReLU, etc.)
- [ ] Use mixed precision (FP16/BF16)
- [ ] Implement memory pooling/reuse
- [ ] Enable gradient checkpointing for deep models
- [ ] Pre-compute static values (position embeddings, masks)
- [ ] Batch operations for data parallelism
- [ ] Profile and identify bottlenecks (NVIDIA Nsight, nvprof)

---

## Appendix A: Common Notation

- **B:** Batch size
- **C:** Number of channels
- **H:** Height (spatial dimension)
- **W:** Width (spatial dimension)
- **N:** Sequence length (number of tokens)
- **D:** Embedding dimension / hidden dimension
- **H (context-dependent):** Number of attention heads
- **d:** Dimension per head (D / H)
- **M:** Window size (Swin Transformer)
- **K:** Kernel size
- **s:** Stride
- **p:** Padding
- **t:** Expansion ratio (MobileNet)
- **φ:** Compound coefficient (EfficientNet)

---

## Appendix B: Useful CUDA Libraries

| Library | Purpose | Documentation |
|---------|---------|---------------|
| **cuDNN** | DNN primitives | https://docs.nvidia.com/deeplearning/cudnn/ |
| **cuBLAS** | Linear algebra | https://docs.nvidia.com/cuda/cublas/ |
| **cuFFT** | FFT for convolution | https://docs.nvidia.com/cuda/cufft/ |
| **Thrust** | Parallel algorithms | https://docs.nvidia.com/cuda/thrust/ |
| **CUB** | Warp/block primitives | https://nvlabs.github.io/cub/ |
| **NCCL** | Multi-GPU communication | https://docs.nvidia.com/deeplearning/nccl/ |
| **Cutlass** | GEMM templates | https://github.com/NVIDIA/cutlass |

---

## Appendix C: Training Hyperparameters (Reference)

### EfficientNet (ImageNet)
- Optimizer: RMSprop (decay=0.9, momentum=0.9)
- Learning rate: 0.256, cosine decay
- Batch size: 4096
- Weight decay: 1e-5
- Augmentation: AutoAugment
- Regularization: Stochastic depth, dropout

### ViT (ImageNet-21K → ImageNet-1K)
- Optimizer: Adam (β1=0.9, β2=0.999)
- Learning rate: 0.0003, linear warmup + cosine decay
- Batch size: 512
- Weight decay: 0.3
- Augmentation: RandAugment, Mixup
- Pre-training: 300 epochs on ImageNet-21K
- Fine-tuning: 30 epochs on ImageNet-1K

### Swin Transformer (ImageNet-1K)
- Optimizer: AdamW (β1=0.9, β2=0.999)
- Learning rate: 0.001, cosine decay
- Batch size: 1024
- Weight decay: 0.05
- Augmentation: RandAugment, Mixup, CutMix
- Regularization: Drop path (0.1-0.5)

### ConvNeXt (ImageNet-1K)
- Optimizer: AdamW (β1=0.9, β2=0.999)
- Learning rate: 0.004, cosine decay
- Batch size: 4096
- Weight decay: 0.05
- Augmentation: RandAugment, Mixup, CutMix
- Regularization: Drop path, LayerScale

### MobileNetV3 (ImageNet-1K)
- Optimizer: RMSprop (decay=0.9, momentum=0.9)
- Learning rate: 0.1, exponential decay
- Batch size: 4096
- Weight decay: 1e-5
- Augmentation: Standard (crop, flip, color jitter)
- Regularization: Dropout

---

**Document End**

**For questions, clarifications, or contributions:**
- Open an issue in the Tenzor repository
- Consult the reference implementations listed in Section 7
- Review the official papers for mathematical details

**Version History:**
- v1.0 (2025-10-18): Initial comprehensive specification document
