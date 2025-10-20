# ConvNeXt and MobileNet Quick Reference

**Quick guide for using ConvNeXt and MobileNet models in Tenzor**

---

## Table of Contents

1. [Basic Usage](#basic-usage)
2. [Model Variants](#model-variants)
3. [API Reference](#api-reference)
4. [Common Patterns](#common-patterns)
5. [Performance Tips](#performance-tips)

---

## Basic Usage

### ConvNeXt

```cpp
#include <tenzor/tenzor.hpp>
using namespace tenzor;

// Create model
auto model = models::convnext_tiny(1000);  // 1000 classes, no pretrained

// Or with pretrained weights
auto model_pretrained = models::convnext_base(1000, true);

// Forward pass
Variable input = randn({8, 3, 224, 224});  // Batch of 8 images
Variable output = model->forward(input);    // Output: [8, 1000]
```

### MobileNetV2

```cpp
#include <tenzor/tenzor.hpp>
using namespace tenzor;

// Standard width (1.0)
auto model = models::mobilenet_v2(1000);

// Custom width multiplier (0.5, 0.75, 1.4)
auto model_small = models::mobilenet_v2_width(1000, 0.5, false);

// Forward pass
Variable input = randn({16, 3, 224, 224});
Variable output = model->forward(input);  // [16, 1000]
```

### MobileNetV3

```cpp
#include <tenzor/tenzor.hpp>
using namespace tenzor;

// Large variant (higher accuracy)
auto model_large = models::mobilenet_v3_large(1000);

// Small variant (lower latency)
auto model_small = models::mobilenet_v3_small(1000);

// Forward pass
Variable input = randn({32, 3, 224, 224});
Variable output = model_large->forward(input);  // [32, 1000]
```

---

## Model Variants

### ConvNeXt Family

| Model | Factory Function | Params | FLOPs | Top-1 Acc |
|-------|-----------------|--------|-------|-----------|
| ConvNeXt-Tiny | `convnext_tiny()` | 28M | 4.5G | 82.1% |
| ConvNeXt-Small | `convnext_small()` | 50M | 8.7G | 83.1% |
| ConvNeXt-Base | `convnext_base()` | 89M | 15.4G | 83.8% |
| ConvNeXt-Large | `convnext_large()` | 198M | 34.4G | 84.3% |
| ConvNeXt-XLarge | `convnext_xlarge()` | 350M | 60.9G | 87.0% |

### MobileNet Family

| Model | Factory Function | Params | FLOPs | Top-1 Acc |
|-------|-----------------|--------|-------|-----------|
| MobileNetV2 | `mobilenet_v2()` | 3.4M | 300M | 72.0% |
| MobileNetV2 (0.75×) | `mobilenet_v2_width(1000, 0.75)` | ~2.6M | ~220M | ~70% |
| MobileNetV2 (0.5×) | `mobilenet_v2_width(1000, 0.5)` | ~1.9M | ~100M | ~65% |
| MobileNetV3-Large | `mobilenet_v3_large()` | 5.4M | 219M | 75.2% |
| MobileNetV3-Small | `mobilenet_v3_small()` | 2.9M | 66M | 67.4% |

---

## API Reference

### ConvNeXt

```cpp
// All variants follow same signature
auto convnext_tiny(int64_t num_classes = 1000, bool pretrained = false)
    -> std::shared_ptr<ConvNeXt>;

auto convnext_small(int64_t num_classes = 1000, bool pretrained = false)
    -> std::shared_ptr<ConvNeXt>;

auto convnext_base(int64_t num_classes = 1000, bool pretrained = false)
    -> std::shared_ptr<ConvNeXt>;

auto convnext_large(int64_t num_classes = 1000, bool pretrained = false)
    -> std::shared_ptr<ConvNeXt>;

auto convnext_xlarge(int64_t num_classes = 1000, bool pretrained = false)
    -> std::shared_ptr<ConvNeXt>;
```

**Parameters:**
- `num_classes`: Number of output classes (default: 1000 for ImageNet)
- `pretrained`: Whether to load pretrained weights (default: false)

**Returns:** Shared pointer to model instance

### MobileNet

```cpp
// MobileNetV2
auto mobilenet_v2(int64_t num_classes = 1000, bool pretrained = false)
    -> std::shared_ptr<MobileNetV2>;

auto mobilenet_v2_width(int64_t num_classes, double width_mult, bool pretrained = false)
    -> std::shared_ptr<MobileNetV2>;

// MobileNetV3
auto mobilenet_v3_large(int64_t num_classes = 1000, bool pretrained = false)
    -> std::shared_ptr<MobileNetV3>;

auto mobilenet_v3_small(int64_t num_classes = 1000, bool pretrained = false)
    -> std::shared_ptr<MobileNetV3>;
```

**Parameters:**
- `num_classes`: Number of output classes
- `width_mult`: Width multiplier (0.5, 0.75, 1.0, 1.4) for MobileNetV2
- `pretrained`: Whether to load pretrained weights

---

## Common Patterns

### 1. Transfer Learning

```cpp
// Load pretrained model
auto model = models::convnext_base(1000, true);

// Replace classifier for custom dataset (e.g., 10 classes)
// ConvNeXt has 'head_' as final linear layer
// You would need to access and replace it

// Fine-tune with lower learning rate
auto optimizer = optim::Adam(model->parameters(), 1e-4);

// Training loop
for (int epoch = 0; epoch < num_epochs; ++epoch) {
    model->train();
    for (auto& batch : train_loader) {
        optimizer.zero_grad();
        auto output = model->forward(batch.data);
        auto loss = criterion(output, batch.labels);
        loss.backward();
        optimizer.step();
    }
}
```

### 2. Feature Extraction

```cpp
// Use model as feature extractor
auto model = models::mobilenet_v3_large(1000, true);
model->eval();  // Set to evaluation mode

// Forward pass without gradient
{
    NoGradGuard no_grad;
    Variable input = load_image("image.jpg");
    Variable features = model->forward(input);
    // Use features for downstream task
}
```

### 3. Inference

```cpp
// Single image inference
auto model = models::convnext_tiny(1000, true);
model->eval();
model->to(Device::cuda(0));  // Move to GPU

Variable image = preprocess_image("cat.jpg");
image = image.to(Device::cuda(0));

{
    NoGradGuard no_grad;
    Variable logits = model->forward(image);
    auto probs = softmax(logits, -1);
    auto top5 = probs.topk(5);
}
```

### 4. Mixed Precision Training

```cpp
// For faster training on modern GPUs
auto model = models::mobilenet_v3_small(1000);
model->to(Device::cuda(0));

// Use FP16 for forward/backward, FP32 for weights
for (auto& batch : train_loader) {
    auto input_fp16 = batch.data.to(DType::Float16);
    auto output = model->forward(input_fp16);
    auto loss = criterion(output, batch.labels);
    loss.backward();
    optimizer.step();
}
```

### 5. Model Ensembling

```cpp
// Ensemble multiple models for better accuracy
auto model1 = models::convnext_small(1000, true);
auto model2 = models::convnext_base(1000, true);
auto model3 = models::convnext_large(1000, true);

model1->eval();
model2->eval();
model3->eval();

Variable input = load_batch(image_paths);
Variable pred1 = softmax(model1->forward(input), -1);
Variable pred2 = softmax(model2->forward(input), -1);
Variable pred3 = softmax(model3->forward(input), -1);

// Average predictions
Variable ensemble_pred = (pred1 + pred2 + pred3) / 3.0;
```

---

## Performance Tips

### 1. Choose Right Model

**For Accuracy:**
- ConvNeXt-Base or ConvNeXt-Large
- MobileNetV3-Large

**For Speed:**
- ConvNeXt-Tiny
- MobileNetV3-Small
- MobileNetV2 with width=0.75 or 0.5

**For Mobile Deployment:**
- MobileNetV3-Small
- MobileNetV2 (width=0.5)

### 2. Batch Size Optimization

```cpp
// ConvNeXt: Use larger batches (GPU memory permitting)
auto model = models::convnext_tiny(1000);
Variable input = randn({64, 3, 224, 224});  // Batch of 64

// MobileNet: Can use very large batches
auto model_mobile = models::mobilenet_v3_small(1000);
Variable input = randn({256, 3, 224, 224});  // Batch of 256
```

### 3. Resolution Scaling

```cpp
// Higher resolution for better accuracy (if needed)
Variable input_224 = randn({8, 3, 224, 224});  // Standard
Variable input_384 = randn({8, 3, 384, 384});  // Higher resolution

// Lower resolution for faster inference
Variable input_192 = randn({16, 3, 192, 192});
Variable input_160 = randn({32, 3, 160, 160});
```

### 4. Device Management

```cpp
// Move to GPU for training
auto model = models::convnext_base(1000);
model->cuda(0);  // GPU 0

// Keep on CPU for inference on edge devices
auto model_mobile = models::mobilenet_v3_small(1000);
model_mobile->cpu();

// Multi-GPU (if supported)
auto model = models::convnext_large(1000);
// DataParallel wrapper would go here
```

### 5. Memory Optimization

```cpp
// Enable gradient checkpointing for very deep models
// (would need to be implemented in model)

// Use eval mode for inference to disable dropout/batchnorm updates
model->eval();

// Clear gradients to free memory
model->zero_grad();

// Use smaller width multipliers for MobileNet
auto model_efficient = models::mobilenet_v2_width(1000, 0.5);
```

---

## Input/Output Specifications

### Input Format

All models expect:
- **Shape:** `[N, 3, H, W]` where N=batch size
- **Type:** Float32
- **Range:** Normalized (typically mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- **Standard Resolution:** 224×224 (can vary)

```cpp
// Preprocessing example
Variable image = load_image("image.jpg");  // [3, H, W]
image = resize(image, {224, 224});
image = normalize(image, {0.485, 0.456, 0.406}, {0.229, 0.224, 0.225});
image = image.unsqueeze(0);  // Add batch dimension: [1, 3, 224, 224]
```

### Output Format

- **Shape:** `[N, num_classes]`
- **Type:** Float32
- **Values:** Logits (not probabilities)

```cpp
// Convert to probabilities
Variable logits = model->forward(input);
Variable probs = softmax(logits, -1);

// Get top prediction
auto [values, indices] = probs.max(-1);
int64_t predicted_class = indices.item<int64_t>();
```

---

## Error Handling

### Common Errors

**1. Shape Mismatch**
```cpp
// ERROR: Wrong number of channels
Variable input = randn({8, 1, 224, 224});  // Grayscale
// FIX: Convert to 3 channels or modify model

Variable input_rgb = input.repeat({1, 3, 1, 1});  // Duplicate grayscale
```

**2. Device Mismatch**
```cpp
// ERROR: Model on GPU, input on CPU
model->cuda(0);
Variable input = randn({8, 3, 224, 224});  // CPU tensor
// FIX: Move input to same device

input = input.cuda(0);
```

**3. Pretrained File Not Found**
```cpp
// ERROR: Pretrained weights not found
auto model = models::convnext_tiny(1000, true);
// FIX: Either download weights or use pretrained=false

auto model = models::convnext_tiny(1000, false);  // No pretrained
```

---

## Pretrained Weights

### Expected Files

Place pretrained weight files in the model directory:

**ConvNeXt:**
- `convnext_tiny_imagenet.pth`
- `convnext_small_imagenet.pth`
- `convnext_base_imagenet.pth`
- `convnext_large_imagenet.pth`
- `convnext_xlarge_imagenet22k.pth`

**MobileNet:**
- `mobilenet_v2_imagenet.pth`
- `mobilenet_v2_0.75_imagenet.pth`
- `mobilenet_v2_0.5_imagenet.pth`
- `mobilenet_v3_large_imagenet.pth`
- `mobilenet_v3_small_imagenet.pth`

### Manual Loading

```cpp
auto model = models::convnext_tiny(1000, false);
model->load_pretrained("/path/to/weights/convnext_tiny_imagenet.pth");
```

---

## Architecture Details

### ConvNeXt Block

```
7×7 Depthwise Conv → LayerNorm → 1×1 Conv (4× expansion) → GELU →
1×1 Conv (projection) → Layer Scale → Residual
```

**Key Features:**
- Large 7×7 kernels
- GELU activation (not ReLU)
- LayerNorm (not BatchNorm)
- Inverted bottleneck
- Layer Scale for stability

### MobileNet V2 Block

```
[1×1 Expansion] → BatchNorm → ReLU6 → k×k Depthwise → BatchNorm → ReLU6 →
1×1 Projection → BatchNorm → Residual
```

**Key Features:**
- Inverted residuals
- Linear bottlenecks (no activation after projection)
- ReLU6 for quantization
- Depthwise separable convolutions

### MobileNet V3 Block

```
[1×1 Expansion] → BatchNorm → Activation* → k×k Depthwise → BatchNorm → Activation* →
[SE Module] → 1×1 Projection → BatchNorm → Residual
```

**Key Features:**
- Hard-Swish in deeper layers
- SE modules for attention
- Efficient last stage design
- NAS-optimized structure

*Activation: ReLU in early layers, Hard-Swish in deeper layers

---

## Debugging Tips

### 1. Check Model Structure

```cpp
auto model = models::convnext_tiny(1000);
auto params = model->parameters();
std::cout << "Total parameters: " << params.size() << std::endl;

for (auto& [name, param] : model->named_parameters()) {
    std::cout << name << ": " << param->shape() << std::endl;
}
```

### 2. Verify Forward Pass

```cpp
auto model = models::mobilenet_v3_large(1000);
Variable input = randn({1, 3, 224, 224});

try {
    Variable output = model->forward(input);
    std::cout << "Output shape: " << output.shape() << std::endl;
} catch (const std::exception& e) {
    std::cerr << "Forward pass failed: " << e.what() << std::endl;
}
```

### 3. Check Gradients

```cpp
auto model = models::convnext_base(1000);
auto input = Variable(randn({2, 3, 224, 224}), true);  // requires_grad

auto output = model->forward(input);
auto loss = output.sum();
loss.backward();

// Verify gradients
for (auto& param : model->parameters()) {
    if (!param->grad().defined()) {
        std::cerr << "Gradient not computed for parameter!" << std::endl;
    }
}
```

---

## Additional Resources

- **Specification:** `/home/lee/Projects/Tenzor/docs/MODERN_CV_ARCHITECTURES_SPEC.md`
- **Implementation Report:** `/home/lee/Projects/Tenzor/docs/CONVNEXT_MOBILENET_IMPLEMENTATION.md`
- **Source Code:** `/home/lee/Projects/Tenzor/src/models/`
- **Headers:** `/home/lee/Projects/Tenzor/include/tenzor/models/`

---

**Last Updated:** 2025-10-18
**Tenzor Version:** 1.0.0
