# U-Net Implementation for Semantic Segmentation

**Date:** 2025-10-18
**Status:** ✅ Complete
**Location:** `include/tenzor/models/unet.hpp`, `src/models/unet.cpp`

## Overview

Implemented the U-Net encoder-decoder architecture for semantic segmentation following the specification in `DETECTION_SEGMENTATION_SPEC.md`. U-Net is widely used for biomedical image segmentation and general semantic segmentation tasks.

## Architecture Components

### 1. DoubleConv Block
**File:** `include/tenzor/models/unet.hpp:38-74`

The fundamental building block of U-Net, consisting of:
```
Conv2d(3x3, padding=1) → BatchNorm2d → ReLU →
Conv2d(3x3, padding=1) → BatchNorm2d → ReLU
```

**Features:**
- Maintains spatial dimensions (padding=1 for "same" convolution)
- BatchNorm for stable training
- ReLU activation for non-linearity
- Configurable mid-channels for flexibility

**Usage:**
```cpp
DoubleConv block(64, 128);  // 64 → 128 channels
Variable out = block.forward(input);
```

### 2. Down Block (Encoder)
**File:** `include/tenzor/models/unet.hpp:76-107`

Downsampling block for the encoder path:
```
MaxPool2d(2x2, stride=2) → DoubleConv
```

**Features:**
- Reduces spatial dimensions by 2x
- Increases feature channels
- Preserves important features via max pooling

**Usage:**
```cpp
Down encoder(64, 128);  // 64 → 128 channels, H/2 x W/2
Variable encoded = encoder.forward(input);
```

### 3. Up Block (Decoder)
**File:** `include/tenzor/models/unet.hpp:109-178`

Upsampling block with skip connections for the decoder path:
```
[Upsample 2x] → Concatenate(skip) → DoubleConv
```

**Features:**
- Two upsampling modes:
  - **Learned (bilinear=false):** ConvTranspose2d for learnable upsampling
  - **Bilinear (bilinear=true):** Fast bilinear interpolation + 1x1 conv
- Concatenates encoder features via skip connections
- Recovers spatial resolution

**Usage:**
```cpp
Up decoder(256, 128, false);  // Learned upsampling
Variable upsampled = decoder.forward(input, skip_connection);
```

### 4. Complete U-Net Model
**File:** `include/tenzor/models/unet.hpp:180-322`

Full encoder-decoder architecture with 4 downsampling and 4 upsampling stages:

```
Input [N, C, H, W]
  ↓ DoubleConv
[N, 64, H, W] ────────────────────→ Skip1
  ↓ Down                             ↓
[N, 128, H/2, W/2] ─────────────→ Skip2
  ↓ Down                             ↓
[N, 256, H/4, W/4] ─────────→ Skip3
  ↓ Down                             ↓
[N, 512, H/8, W/8] ─────→ Skip4
  ↓ Down                             ↓
[N, 512|1024, H/16, W/16]  (Bottleneck)
  ↑ Up + Concat(Skip4)
[N, 256|512, H/8, W/8]
  ↑ Up + Concat(Skip3)
[N, 128|256, H/4, W/4]
  ↑ Up + Concat(Skip2)
[N, 64|128, H/2, W/2]
  ↑ Up + Concat(Skip1)
[N, 64, H, W]
  ↓ Conv2d(1x1)
Output [N, num_classes, H, W]
```

**Constructor Parameters:**
- `in_channels`: Input channels (1 for grayscale, 3 for RGB)
- `num_classes`: Number of output classes
- `bilinear`: Use bilinear upsampling (faster, less memory) vs learned (better results)

**Channel Configuration:**
- **Bilinear mode:** Max 512 channels in bottleneck
- **Learned mode:** Max 1024 channels in bottleneck

## Additional Implementation

### Bilinear Interpolation Operation
**File:** `include/tenzor/ops/vision.hpp:87-123`, `src/ops/vision.cpp:195-338`

Implemented `interpolate()` function for upsampling tensors:

**Features:**
- **Modes:** "nearest", "bilinear" (bicubic planned)
- **Align corners:** PyTorch-compatible pixel alignment
- **CPU implementation:** Reference implementation
- **CUDA support:** Tagged for future GPU acceleration

**Algorithm (Bilinear):**
```cpp
// For each output pixel:
1. Compute source position in input (float coordinates)
2. Find 4 nearest neighbors
3. Calculate bilinear weights
4. Interpolate: value = Σ(weight_i × pixel_i)
```

**Usage:**
```cpp
auto upsampled = ops::interpolate(input, {64, 64}, "bilinear");
```

## Usage Examples

### Binary Segmentation
```cpp
#include "tenzor/models/unet.hpp"

// Medical imaging: grayscale input, binary output (tumor/no tumor)
UNet model(1, 1, false);

Variable input(Tensor({batch, 1, 256, 256}, DType::Float32, Device::cpu()), true);
Variable output = model.forward(input);  // [batch, 1, 256, 256]

// Apply sigmoid for probabilities
auto probs = sigmoid(output);
auto binary_mask = (probs > 0.5);
```

### Multi-Class Segmentation
```cpp
// Pascal VOC: RGB input, 21 classes
UNet model(3, 21, true);  // Bilinear for efficiency

Variable input(Tensor({batch, 3, 512, 512}, DType::Float32, Device::cpu()), true);
Variable logits = model.forward(input);  // [batch, 21, 512, 512]

// Apply softmax for class probabilities
auto probs = softmax(logits, /*dim=*/1);
auto predictions = argmax(probs, /*dim=*/1);  // [batch, 512, 512]
```

### Training Example
```cpp
#include "tenzor/models/unet.hpp"
#include "tenzor/nn/loss/losses.hpp"
#include "tenzor/nn/optim/adam.hpp"

// Create model
auto model = std::make_shared<UNet>(3, 2, false);
model->train();

// Optimizer
auto optimizer = Adam(model->parameters(), 1e-4);

// Loss: BCE + Dice
auto bce_loss = BCEWithLogitsLoss();
auto dice_loss = DiceLoss();

// Training loop
for (int epoch = 0; epoch < 100; ++epoch) {
    for (auto& batch : dataloader) {
        // Forward pass
        auto output = model->forward(batch.input);

        // Combined loss
        auto loss = 0.5 * bce_loss(output, batch.target) +
                   0.5 * dice_loss(sigmoid(output), batch.target);

        // Backward pass
        optimizer.zero_grad();
        loss.backward();
        optimizer.step();
    }
}
```

## Integration with Build System

**CMakeLists.txt:**
```cmake
# src/CMakeLists.txt
models/unet.cpp  # Added to TENZOR_CORE_SOURCES
```

The U-Net model is compiled as part of the core Tenzor library.

## Key Design Decisions

### 1. Skip Connection Implementation
- Concatenation along channel dimension (PyTorch style)
- Handles variable input sizes through dynamic shape inference
- Requires `ops::cat()` from transform.hpp

### 2. Upsampling Modes
- **Learned (ConvTranspose2d):**
  - Better quality results
  - More parameters (slower, more memory)
  - Recommended for high-quality segmentation

- **Bilinear:**
  - Faster training and inference
  - Fewer parameters
  - Recommended for large images or limited hardware

### 3. Activation Strategy
- ReLU as separate modules (not in-place)
- Enables proper autograd tracking
- Consistent with Tenzor's module design

### 4. Channel Progression
Follows standard U-Net design:
```
64 → 128 → 256 → 512 → 1024 (learned)
64 → 128 → 256 → 512 → 512  (bilinear)
```

## Testing Recommendations

### Unit Tests
1. **DoubleConv:**
   - Shape preservation
   - Channel transformation
   - Gradient flow

2. **Down:**
   - 2x downsampling
   - Feature extraction

3. **Up:**
   - 2x upsampling
   - Skip connection concatenation
   - Both bilinear and learned modes

4. **UNet:**
   - End-to-end shape consistency
   - Skip connection alignment
   - Gradient flow through entire network

### Integration Tests
1. **Forward Pass:**
   ```cpp
   auto model = UNet(3, 21, false);
   Variable input(Tensor({2, 3, 256, 256}, DType::Float32), true);
   Variable output = model.forward(input);
   assert(output.tensor().shape() == std::vector<int64_t>{2, 21, 256, 256});
   ```

2. **Backward Pass:**
   ```cpp
   auto loss = output.sum();
   loss.backward();

   // Check all parameters have gradients
   for (auto& param : model.parameters()) {
       assert(param->grad() != nullptr);
   }
   ```

3. **Device Transfer:**
   ```cpp
   model.cuda();
   Variable input_gpu(Tensor({1, 3, 256, 256}, DType::Float32, Device::cuda(0)), true);
   Variable output_gpu = model.forward(input_gpu);
   assert(output_gpu.tensor().device().type == Device::Type::CUDA);
   ```

## Performance Considerations

### Memory Usage
- **Input:** 256×256×3 RGB
- **Peak activation:** Bottleneck (16×16×1024 for learned mode)
- **Estimated peak:** ~200MB for batch_size=4

### Optimization Opportunities
1. **CUDA kernel for interpolate:** Currently CPU-only, tagged for GPU impl
2. **In-place ReLU:** Could reduce memory for large feature maps
3. **Gradient checkpointing:** For very large images
4. **Mixed precision (FP16):** Reduce memory by 2x

## Compatibility

### Dependencies
- ✅ Conv2d, ConvTranspose2d (conv.hpp)
- ✅ BatchNorm2d (batchnorm.hpp)
- ✅ MaxPool2d (pooling.hpp)
- ✅ ReLU (activations.hpp)
- ✅ cat (transform.hpp)
- ✅ interpolate (vision.hpp) - **NEWLY IMPLEMENTED**

### Device Support
- ✅ CPU: Full support
- ⚠️ CUDA: Supported via existing layer ops, interpolate needs GPU kernel
- ⚠️ ROCm: Supported via existing layer ops, interpolate needs GPU kernel
- ⚠️ OneAPI: Supported via existing layer ops, interpolate needs GPU kernel

## Comparison with Specification

| Requirement | Status | Notes |
|-------------|--------|-------|
| DoubleConv block | ✅ | Conv→BN→ReLU→Conv→BN→ReLU |
| Down block | ✅ | MaxPool→DoubleConv |
| Up block | ✅ | Both bilinear and transposed conv |
| Skip connections | ✅ | Concatenation along channels |
| 4-level encoder/decoder | ✅ | Standard U-Net depth |
| Configurable channels | ✅ | via in_channels, num_classes |
| Bilinear upsampling | ✅ | Implemented in ops::interpolate |
| Full autograd support | ✅ | All operations differentiable |
| No stubs | ✅ | Complete implementation |

## Future Enhancements

1. **Attention U-Net:** Add attention gates for better feature selection
2. **3D U-Net:** Extend to volumetric segmentation
3. **U-Net++:** Nested skip connections
4. **Residual U-Net:** Add residual connections in DoubleConv
5. **GPU kernels:** Optimize interpolate for CUDA/ROCm/OneAPI

## References

1. Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation", MICCAI 2015
2. Detection & Segmentation Specification: `docs/DETECTION_SEGMENTATION_SPEC.md`
3. PyTorch U-Net implementation (reference for compatibility)

## Files Modified

### New Files
- `include/tenzor/models/unet.hpp` (322 lines)
- `src/models/unet.cpp` (293 lines)
- `docs/UNET_IMPLEMENTATION.md` (this file)

### Modified Files
- `include/tenzor/ops/vision.hpp` (+38 lines: interpolate declaration)
- `src/ops/vision.cpp` (+144 lines: interpolate implementation)
- `src/CMakeLists.txt` (+1 line: models/unet.cpp)

### Total Addition
- **~800 lines** of implementation and documentation
- **3 new classes** (DoubleConv, Down, Up)
- **1 complete model** (UNet)
- **1 new operation** (interpolate with bilinear mode)

---

**Implementation Status:** ✅ **COMPLETE**
**Compilation Status:** ✅ **PASSES**
**Ready for Testing:** ✅ **YES**
