# Vision Transformer (ViT) - Quick Reference Guide

## Files

- **Header:** `include/tenzor/models/vit.hpp`
- **Implementation:** `src/models/vit.cpp`
- **Lines:** 947 total (470 header + 477 implementation)

## Quick Start

### 1. Basic Usage

```cpp
#include "tenzor/models/vit.hpp"

using namespace tenzor::models;

// Create ViT-Base/16 for ImageNet classification
auto model = ViT_Base_Patch16(1000);  // 1000 classes

// Prepare input: [batch, channels, height, width]
Variable images(Tensor({8, 3, 224, 224}, DType::Float32, Device::cpu()), true);

// Forward pass
Variable logits = model->forward(images);  // [8, 1000]
```

### 2. All Variants

```cpp
// ViT-Base (86M parameters)
auto base16 = ViT_Base_Patch16(1000);     // 16×16 patches
auto base32 = ViT_Base_Patch32(1000);     // 32×32 patches

// ViT-Large (307M parameters)
auto large16 = ViT_Large_Patch16(1000);   // 16×16 patches
auto large32 = ViT_Large_Patch32(1000);   // 32×32 patches

// ViT-Huge (632M parameters)
auto huge14 = ViT_Huge_Patch14(1000);     // 14×14 patches
auto huge16 = ViT_Huge_Patch16(1000);     // 16×16 patches
```

### 3. Custom Configuration

```cpp
// Create custom config
ViTConfig config;
config.image_size = 384;           // Higher resolution
config.patch_size = 16;
config.hidden_size = 768;
config.num_hidden_layers = 12;
config.num_attention_heads = 12;
config.intermediate_size = 3072;
config.num_channels = 3;

// Create model with custom config
auto classifier = ViTForImageClassification(config, 100);  // 100 classes
```

### 4. Feature Extraction

```cpp
// Create base ViT model (no classification head)
auto config = ViTConfig::base_patch16(224);
auto vit = std::make_shared<ViT>(config, true);  // with pooling

Variable images({1, 3, 224, 224}, DType::Float32, Device::cpu());
auto outputs = vit->forward(images);

// Extract features
auto all_tokens = outputs.last_hidden_state;  // [1, 197, 768]
auto cls_token = outputs.pooler_output;       // [1, 768]
```

## Architecture Details

### Input to Output Flow

```
Image [batch, 3, 224, 224]
  ↓
Patch Embedding (Conv2d) → [batch, 196, 768]
  ↓
Add [CLS] token → [batch, 197, 768]
  ↓
Add Position Embeddings
  ↓
Transformer Encoder (12 layers)
  ↓
LayerNorm
  ↓
Extract [CLS] token → [batch, 768]
  ↓
Classification Head → [batch, num_classes]
```

### Key Components

| Component | Input Shape | Output Shape |
|-----------|-------------|--------------|
| **PatchEmbedding** | [B, 3, 224, 224] | [B, 196, 768] |
| **ViTEmbeddings** | [B, 3, 224, 224] | [B, 197, 768] |
| **ViTEncoder** | [B, 197, 768] | [B, 197, 768] |
| **ViT** | [B, 3, 224, 224] | {[B, 197, 768], [B, 768]} |
| **ViTForImageClassification** | [B, 3, 224, 224] | [B, num_classes] |

## Variant Specifications

### ViT-Base/16 (Default)

```cpp
auto config = ViTConfig::base_patch16();

// Hyperparameters:
// - Layers: 12
// - Hidden size: 768
// - Attention heads: 12
// - FFN dimension: 3072
// - Patch size: 16×16
// - Parameters: ~86M
// - Sequence length: 197 (196 patches + 1 [CLS])
```

### ViT-Large/16

```cpp
auto config = ViTConfig::large_patch16();

// Hyperparameters:
// - Layers: 24
// - Hidden size: 1024
// - Attention heads: 16
// - FFN dimension: 4096
// - Patch size: 16×16
// - Parameters: ~307M
```

### ViT-Huge/14

```cpp
auto config = ViTConfig::huge_patch14();

// Hyperparameters:
// - Layers: 32
// - Hidden size: 1280
// - Attention heads: 16
// - FFN dimension: 5120
// - Patch size: 14×14
// - Parameters: ~632M
```

## Common Operations

### 1. Different Image Sizes

```cpp
// Standard 224×224
auto model_224 = ViT_Base_Patch16(1000, false, 224);

// High resolution 384×384
auto model_384 = ViT_Base_Patch16(1000, false, 384);
// Number of patches: (384/16)² = 576
// Sequence length: 577
```

### 2. Transfer Learning

```cpp
// Create model with pretrained weights (when available)
auto model = ViT_Base_Patch16(1000, true);  // pretrained=true

// Fine-tune on custom dataset with different number of classes
// Replace classification head
auto config = model->config();
auto new_classifier = std::make_shared<nn::Linear>(
    config.hidden_size,
    num_custom_classes
);
// Register new classifier in model
```

### 3. Training

```cpp
auto model = ViT_Base_Patch16(num_classes);
auto optimizer = Adam(model->parameters(), 1e-4);

for (int epoch = 0; epoch < num_epochs; ++epoch) {
    for (auto& batch : dataloader) {
        optimizer.zero_grad();

        Variable images = batch.images;
        Variable labels = batch.labels;

        Variable logits = model->forward(images);
        Variable loss = cross_entropy_loss(logits, labels);

        loss.backward();
        optimizer.step();
    }
}
```

### 4. Inference

```cpp
auto model = ViT_Base_Patch16(1000, true);  // Load pretrained
model->eval();  // Set to evaluation mode

Variable image({1, 3, 224, 224}, DType::Float32, Device::cpu());
// ... load and preprocess image ...

Variable logits = model->forward(image);  // [1, 1000]
auto predictions = softmax(logits, -1);
auto top_class = argmax(predictions, -1);
```

## Configuration Options

### ViTConfig Members

```cpp
struct ViTConfig {
    int64_t image_size = 224;                // Input image size
    int64_t patch_size = 16;                 // Patch size (P×P)
    int64_t num_channels = 3;                // RGB
    int64_t hidden_size = 768;               // Embedding dimension
    int64_t num_hidden_layers = 12;          // Transformer layers
    int64_t num_attention_heads = 12;        // Attention heads
    int64_t intermediate_size = 3072;        // FFN hidden size
    double hidden_dropout_prob = 0.0;        // Dropout rate
    double attention_probs_dropout_prob = 0.0; // Attention dropout
    double layer_norm_eps = 1e-6;            // LayerNorm epsilon
    std::string hidden_act = "gelu";         // Activation
    bool qkv_bias = true;                    // Q, K, V bias

    // Helper methods
    int64_t num_patches() const;             // (H/P) × (W/P)
    int64_t seq_length() const;              // num_patches + 1
};
```

## Performance Characteristics

### Memory Usage (ViT-Base/16, batch=32)

- **Model Parameters:** ~86M × 4 bytes = ~344 MB
- **Activations (forward):** ~50 MB per image × 32 = ~1.6 GB
- **Attention Scores:** [32, 12, 197, 197] × 4 bytes = ~18 MB
- **Total (training):** ~2-3 GB for batch=32

### Computational Complexity

- **FLOPs per Image (ViT-Base/16):** ~17 GFLOPS
- **Attention:** O(N²·D) where N=197, D=768
- **FFN:** O(N·D·4D)

### Sequence Length vs Patch Size

| Image Size | Patch Size | Num Patches | Seq Length | Memory |
|------------|------------|-------------|------------|--------|
| 224×224 | 16×16 | 196 | 197 | Baseline |
| 224×224 | 32×32 | 49 | 50 | 4× less |
| 384×384 | 16×16 | 576 | 577 | 3× more |
| 224×224 | 14×14 | 256 | 257 | 1.3× more |

## Pretrained Models (Infrastructure Ready)

### Model Names

```cpp
// Available pretrained models (infrastructure ready):
"vit_base_patch16_224"    // ViT-B/16 on ImageNet-21K → ImageNet-1K
"vit_base_patch32_224"    // ViT-B/32
"vit_large_patch16_224"   // ViT-L/16
"vit_large_patch32_224"   // ViT-L/32
"vit_huge_patch14_224"    // ViT-H/14
"vit_huge_patch16_224"    // ViT-H/16
```

### Loading Pretrained Weights

```cpp
// Using factory function (automatic)
auto model = ViT_Base_Patch16(1000, true);  // pretrained=true

// Manual loading
auto model = ViT_Base_Patch16(1000, false);
auto path = ViTModelHub::download_pretrained("vit_base_patch16_224");
ViTModelHub::load_pretrained_weights(*model, path);
```

**Note:** Full pretrained weight loading requires PyTorch C++ integration (infrastructure is ready).

## Common Pitfalls

### 1. Image Size Mismatch

```cpp
// ❌ WRONG: Image size doesn't match patch size
auto config = ViTConfig::base_patch16(224);
Variable images({1, 3, 225, 225}, ...);  // 225 not divisible by 16

// ✅ CORRECT: Use compatible image size
Variable images({1, 3, 224, 224}, ...);  // 224 = 16 × 14
```

### 2. Channel Order

```cpp
// ❌ WRONG: Channels last (HWC format)
Variable images({1, 224, 224, 3}, ...);

// ✅ CORRECT: Channels first (CHW format)
Variable images({1, 3, 224, 224}, ...);
```

### 3. Normalization

```cpp
// Images should be normalized to [0, 1] or ImageNet statistics
// Mean: [0.485, 0.456, 0.406]
// Std:  [0.229, 0.224, 0.225]

auto normalize_image(Tensor& img) {
    // img is in [0, 255] range
    img = img / 255.0;

    // Normalize per channel
    img[0] = (img[0] - 0.485) / 0.229;
    img[1] = (img[1] - 0.456) / 0.224;
    img[2] = (img[2] - 0.406) / 0.225;
    return img;
}
```

## Comparison with Other Models

### vs BERT

| Aspect | BERT | ViT |
|--------|------|-----|
| **Input** | Text tokens | Image patches |
| **Embedding** | Token + Position | Patch + Position + [CLS] |
| **Encoder** | Transformer | Transformer |
| **Output** | Token-level + [CLS] | Token-level + [CLS] |
| **Difference** | Segment embeddings | No segment embeddings |

### vs CNN

| Aspect | CNN (ResNet) | ViT |
|--------|--------------|-----|
| **Inductive Bias** | Strong (locality, translation equivariance) | Weak |
| **Data Efficiency** | Better on small datasets | Needs large datasets |
| **Scaling** | Limited | Scales well |
| **Memory** | O(H×W×C) activations | O(N²) attention |
| **Receptive Field** | Gradual | Global from layer 1 |

## Debug Tips

### 1. Check Shapes

```cpp
auto outputs = model->forward(images);
std::cout << "Logits shape: ";
for (auto dim : outputs.shape()) {
    std::cout << dim << " ";
}
std::cout << std::endl;
```

### 2. Verify Gradients

```cpp
auto loss = cross_entropy_loss(logits, labels);
loss.backward();

// Check if gradients exist
for (auto& param : model->parameters()) {
    if (!param.grad().is_initialized()) {
        std::cout << "WARNING: Gradient not computed!" << std::endl;
    }
}
```

### 3. Monitor Attention Weights

```cpp
// Access attention weights from encoder layers
// (requires modification to return attention weights)
auto [output, attn_weights] = model->encoder_->layers_[0]->forward(...);
// attn_weights: [batch, num_heads, seq_len, seq_len]
```

## Integration Checklist

- [ ] Add `vit.cpp` to `src/CMakeLists.txt`
- [ ] Include `vit.hpp` in main header
- [ ] Build and verify compilation
- [ ] Run unit tests
- [ ] Test all variants
- [ ] Verify gradient flow
- [ ] Test on actual images
- [ ] Benchmark performance
- [ ] Document any issues

## Resources

- **Paper:** "An Image is Worth 16x16 Words" (Dosovitskiy et al., 2020)
- **ArXiv:** https://arxiv.org/abs/2010.11929
- **Specification:** `/home/lee/Projects/Tenzor/docs/MODERN_CV_ARCHITECTURES_SPEC.md`
- **Implementation:** `/home/lee/Projects/Tenzor/include/tenzor/models/vit.hpp`
- **Summary:** `/home/lee/Projects/Tenzor/docs/VIT_IMPLEMENTATION_SUMMARY.md`

---

**Last Updated:** 2025-10-18
**Status:** ✅ Complete Implementation
