# DeepLab v3+ Quick Start Guide

## Basic Usage

### 1. Create a Model

```cpp
#include "tenzor/models/deeplabv3plus.hpp"

using namespace tenzor;

// For PASCAL VOC segmentation (21 classes)
auto model = models::DeepLabV3Plus_ResNet50(21);

// For Cityscapes (19 classes)
auto model = models::DeepLabV3Plus_ResNet101(19);

// Lightweight variant for mobile/edge devices
auto model = models::DeepLabV3Plus_MobileNetV2(21);
```

### 2. Forward Pass

```cpp
// Create input tensor (batch_size=1, channels=3, height=512, width=512)
Variable input(Tensor({1, 3, 512, 512}, DType::Float32, Device::cpu()), true);

// Get segmentation logits
Variable output = model->forward(input);
// Output shape: {1, num_classes, 512, 512}
```

### 3. Get Segmentation Map

```cpp
// Method 1: Use built-in predict method
Tensor seg_map = model->predict(input);
// Output shape: {1, 512, 512} with class indices

// Method 2: Manual prediction
Variable logits = model->forward(input);
Variable probs = autograd::softmax(logits, 1);
Tensor seg_map = argmax(probs.data(), 1);
```

## Advanced Usage

### Custom Configuration

```cpp
// Create model with custom settings
int64_t num_classes = 150;  // ADE20K dataset
std::string backbone = "resnet101";
int64_t output_stride = 16;  // 8 for finer details, 16 for speed
bool pretrained = true;

auto model = std::make_shared<models::DeepLabV3Plus>(
    num_classes, backbone, output_stride, pretrained
);
```

### Training Example

```cpp
#include "tenzor/nn/optim/adam.hpp"
#include "tenzor/nn/loss/losses.hpp"

// Create model
auto model = models::DeepLabV3Plus_ResNet50(21);
model->train();

// Create optimizer
auto optimizer = std::make_shared<nn::Adam>(model->parameters(), 0.001);

// Create loss function
auto criterion = nn::CrossEntropyLoss();

// Training loop
for (int epoch = 0; epoch < num_epochs; ++epoch) {
    for (auto& [images, masks] : dataloader) {
        // Forward pass
        Variable output = model->forward(images);

        // Compute loss
        Variable loss = criterion.forward(output, masks);

        // Backward pass
        optimizer->zero_grad();
        loss.backward();
        optimizer->step();

        std::cout << "Loss: " << loss.data().item<float>() << std::endl;
    }
}
```

### Inference Example

```cpp
#include "tenzor/ops/vision.hpp"

// Set model to evaluation mode
model->eval();

// Load and preprocess image
Tensor image = load_image("input.jpg");  // Your image loading function
image = resize(image, 512, 512);
image = normalize(image, {0.485, 0.456, 0.406}, {0.229, 0.224, 0.225});

// Add batch dimension
image = image.unsqueeze(0);

// Inference (no gradient computation)
Variable input(image, false);
Tensor seg_map = model->predict(input);

// Post-process and save
seg_map = colorize_segmentation(seg_map);  // Your visualization function
save_image(seg_map, "output.png");
```

## Model Variants Comparison

| Variant | Params | Speed | Accuracy | Use Case |
|---------|--------|-------|----------|----------|
| ResNet-50 | 39.8M | 1.0x | High | Standard |
| ResNet-101 | 58.8M | 0.77x | Higher | Best accuracy |
| MobileNetV2 | 5.8M | 3-5x | Good | Mobile/Edge |

## Output Stride Configuration

```cpp
// output_stride = 16 (standard)
// - Faster inference
// - Lower memory usage
// - Good accuracy for most tasks
auto model = models::DeepLabV3Plus_ResNet50(21, 16);

// output_stride = 8 (finer details)
// - 2x slower inference
// - 4x more memory
// - Better for fine details and small objects
auto model = models::DeepLabV3Plus_ResNet50(21, 8);
```

## Common Datasets

### PASCAL VOC 2012
```cpp
int64_t num_classes = 21;
std::vector<std::string> class_names = {
    "background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
};
```

### Cityscapes
```cpp
int64_t num_classes = 19;
std::vector<std::string> class_names = {
    "road", "sidewalk", "building", "wall", "fence",
    "pole", "traffic light", "traffic sign", "vegetation", "terrain",
    "sky", "person", "rider", "car", "truck",
    "bus", "train", "motorcycle", "bicycle"
};
```

### ADE20K
```cpp
int64_t num_classes = 150;  // 150 semantic categories
```

## Performance Tips

1. **Use GPU for inference**
   ```cpp
   model->to(Device::cuda(0));
   input = input.to(Device::cuda(0));
   ```

2. **Batch processing**
   ```cpp
   // Process multiple images at once
   Variable batch(Tensor({8, 3, 512, 512}, DType::Float32, Device::cuda(0)), false);
   auto outputs = model->forward(batch);
   ```

3. **Mixed precision inference**
   ```cpp
   // Use FP16 for faster inference (if supported)
   model->to(DType::Float16);
   input = input.to(DType::Float16);
   ```

4. **Output stride selection**
   - Use `output_stride=16` for speed
   - Use `output_stride=8` only when fine details are critical

## Troubleshooting

### Out of Memory
- Reduce input resolution (e.g., 512×512 → 256×256)
- Use smaller batch size
- Use MobileNetV2 backbone
- Use output_stride=16 instead of 8

### Slow Inference
- Use GPU instead of CPU
- Use output_stride=16
- Consider MobileNetV2 for real-time applications
- Batch multiple images together

### Poor Accuracy
- Use pretrained backbone
- Use output_stride=8 for finer details
- Increase input resolution
- Use ResNet-101 instead of ResNet-50

## Next Steps

1. **Test the implementation:**
   ```bash
   cd build
   cmake ..
   make -j$(nproc)
   ./tests/test_deeplabv3plus
   ```

2. **Train on custom data:**
   - Prepare your dataset in the required format
   - Implement data augmentation
   - Fine-tune the model on your dataset

3. **Export for deployment:**
   - Use ONNX export for cross-platform deployment
   - Quantize model for edge devices
   - Optimize with TensorRT/OpenVINO

## References

- **Paper:** [Encoder-Decoder with Atrous Separable Convolution](https://arxiv.org/abs/1802.02611)
- **Specification:** `/home/lee/Projects/Tenzor/docs/DETECTION_SEGMENTATION_SPEC.md`
- **Implementation:** `/home/lee/Projects/Tenzor/docs/DEEPLABV3PLUS_IMPLEMENTATION.md`
