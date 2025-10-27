# Backend Test Conversion Status

## Overview

**Goal**: Convert ALL tests to use the `BackendTest` fixture to ensure backend parity across CPU, CUDA, Vulkan, and OneAPI.

**Current Status**: 16/187 tests converted (8.5%)

## Converted Tests ✅

These tests already use `BackendTest` and run on all backends:

### Core Tests
1. `unit/test_tensor.cpp` - Tensor creation and manipulation
2. `unit/test_ops.cpp` - Tensor operations
3. `unit/test_autograd.cpp` - Automatic differentiation
4. `unit/test_broadcasting.cpp` - Broadcasting operations
5. `unit/test_device.cpp` - Device management
6. `test_backend_parity_example.cpp` - Example multi-backend test

### Neural Network Tests
7. `unit/test_linear.cpp` - Linear layer
8. `unit/test_embedding.cpp` - Embedding layer
9. `unit/test_rnn.cpp` - RNN layer
10. `unit/test_lstm.cpp` - LSTM layer
11. `unit/test_gru.cpp` - GRU layer
12. `unit/test_attention.cpp` - Attention mechanisms
13. `unit/test_transformer.cpp` - Transformer architecture

### Loss and Optimization
14. `unit/test_losses.cpp` - Loss functions
15. `unit/test_optimizers.cpp` - Optimizers (SGD, Adam, etc.)
16. `unit/test_transforms.cpp` - Data transforms

## Tests Needing Conversion 🔄

### Priority 1: Core Functionality (HIGH)
These tests are critical for backend parity:

- `test_comparison_operators.cpp` - Comparison ops (eq, ne, lt, gt, etc.)
- `test_dtype_conversion.cpp` - Data type conversions
- `test_creation_ops.cpp` - Tensor creation operations
- `test_split_operation.cpp` - Split/chunk operations
- `test_indexing_operator.cpp` - Indexing and slicing
- `test_conv1d.cpp` - 1D convolution
- `test_convtranspose2d.cpp` - Transpose convolution
- `nn/layers/test_conv2d.cpp` - 2D convolution
- `nn/layers/test_pooling.cpp` - Pooling layers
- `nn/layers/test_batchnorm2d.cpp` - Batch normalization
- `nn/layers/test_dropout.cpp` - Dropout layers
- `nn/layers/test_normalization.cpp` - Layer/Group normalization

### Priority 2: Advanced Operations (MEDIUM)
Important for production use:

- `unit/test_fused_ops.cpp` - Fused operations
- `unit/test_inplace_operations.cpp` - In-place operations
- `ops/test_advanced_ops.cpp` - Advanced tensor operations
- `unit/test_mixed_precision.cpp` - FP16/BF16 support
- `unit/test_autocast.cpp` - Automatic mixed precision
- `unit/test_grad_scaler.cpp` - Gradient scaling
- `unit/test_quantization.cpp` - Quantization support

### Priority 3: Models & Integration (MEDIUM)
End-to-end model tests:

- `unit/test_resnet.cpp` - ResNet architecture
- `unit/test_bert.cpp` - BERT model
- `unit/test_gpt.cpp` - GPT model
- `unit/test_vit.cpp` - Vision Transformer
- `unit/test_efficientnet.cpp` - EfficientNet
- `unit/test_mobilenet_v2_v3.cpp` - MobileNet variants
- `unit/test_yolo.cpp` - YOLO detection
- `unit/test_unet.cpp` - U-Net segmentation
- `unit/test_mask_rcnn.cpp` - Mask R-CNN
- `unit/test_faster_rcnn.cpp` - Faster R-CNN
- `unit/test_deeplabv3plus.cpp` - DeepLabV3+

### Priority 4: Specialized Features (LOW)
Less critical but should be supported:

- `unit/test_checkpoint.cpp` - Gradient checkpointing
- `unit/test_dataloader.cpp` - Data loading
- `unit/test_callbacks.cpp` - Training callbacks
- `nn/optim/test_schedulers.cpp` - Learning rate schedulers
- `unit/test_pruning.cpp` - Model pruning
- `unit/test_distillation.cpp` - Knowledge distillation
- `integration/test_model_persistence.cpp` - Model save/load
- `nn/test_serialization.cpp` - Serialization

### Priority 5: Backend-Specific Tests (SPECIAL)
These may need special handling:

- `backends/test_cuda_kernels.cpp` - CUDA-specific (keep as-is or adapt)
- `backends/test_fp16_kernels.cpp` - FP16 kernels (keep as-is)
- `backends/test_rocm_*` - ROCm-specific (keep as-is)
- `backend/test_oneapi_backend.cpp` - OneAPI-specific (keep as-is)
- `test_vulkan_complete_ops.cpp` - Vulkan-specific (convert or keep)

## Conversion Pattern

### Before (CPU-only):
```cpp
TEST(TensorTest, Addition) {
    auto a = ones({2, 2}, DType::Float32, Device::cpu());
    auto b = ones({2, 2}, DType::Float32, Device::cpu());
    auto c = a + b;
    // assertions...
}
```

### After (Multi-backend):
```cpp
#include "../backend_test_fixture.hpp"

class TensorBackendTest : public BackendTest {};

TEST_P(TensorBackendTest, Addition) {
    auto a = ones({2, 2}, DType::Float32, device);  // device from fixture
    auto b = ones({2, 2}, DType::Float32, device);
    auto c = a + b;

    // Move to CPU for verification
    auto c_cpu = c.to(Device::cpu());
    // assertions...
}

INSTANTIATE_BACKEND_TESTS(TensorBackendTest);
```

## Implementation Guidelines

1. **Include the fixture**: `#include "../backend_test_fixture.hpp"`
2. **Inherit from BackendTest**: `class MyTest : public BackendTest {};`
3. **Use TEST_P**: Replace `TEST(TestSuite, TestName)` with `TEST_P(TestSuite, TestName)`
4. **Use device parameter**: Replace hardcoded `Device::cpu()` with `device`
5. **Move to CPU for verification**: Use `.to(Device::cpu())` before checking values
6. **Add backend info to assertions**: Include `device.to_string()` in error messages
7. **Instantiate for all backends**: Add `INSTANTIATE_BACKEND_TESTS(TestSuite);`

## Testing Strategy

### Phase 1: Core Operations (Weeks 1-2)
- Convert all Priority 1 tests
- Ensure basic operations work on all backends
- Fix any backend-specific issues discovered

### Phase 2: Advanced Features (Weeks 3-4)
- Convert Priority 2 tests
- Handle mixed precision, quantization
- Optimize performance across backends

### Phase 3: Models & Integration (Weeks 5-6)
- Convert Priority 3 tests
- Validate end-to-end model training
- Benchmark performance

### Phase 4: Specialized Features (Week 7)
- Convert Priority 4 tests
- Handle edge cases
- Complete documentation

## Running Backend-Specific Tests

```bash
# Run all tests on CPU backend
ctest -R "cpu"

# Run all tests on CUDA backend
ctest -R "cuda"

# Run all tests on Vulkan backend
ctest -R "vulkan"

# Run all tests on OneAPI backend
ctest -R "oneapi"

# Run specific test on all backends
ctest -R "Addition" --verbose

# Run tests in parallel
ctest -j$(nproc) --output-on-failure
```

## Expected Outcomes

1. **Backend Parity**: All operations produce identical results across backends
2. **Quality Assurance**: Catch backend-specific bugs early
3. **Performance Comparison**: Identify performance differences
4. **Confidence**: Deploy to any backend with confidence
5. **Documentation**: Clear examples of multi-backend usage

## Progress Tracking

- [x] Create backend test fixture (DONE)
- [x] Convert 16 core tests (DONE)
- [ ] Convert Priority 1 tests (0/12 remaining)
- [ ] Convert Priority 2 tests (0/7 remaining)
- [ ] Convert Priority 3 tests (0/11 remaining)
- [ ] Convert Priority 4 tests (0/8 remaining)
- [ ] Handle Priority 5 special cases (0/5 remaining)
- [ ] Run full test suite on all backends
- [ ] Document any backend-specific limitations

## Notes

- Backend availability is checked automatically - tests skip if backend unavailable
- The `BackendTest` fixture provides helper methods like `expectTensorNear()`
- Each test runs 4 times (once per backend)
- Use `device.to_string()` in assertions for better error messages
- Always move tensors to CPU before reading raw data for verification

---

**Last Updated**: 2025-10-27
**Status**: In Progress - 16/187 tests converted (8.5%)
