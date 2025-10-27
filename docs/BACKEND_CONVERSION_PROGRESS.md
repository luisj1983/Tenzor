# Backend Test Conversion Progress

## Overview
Converting all unit tests to use `BackendTest` fixture for multi-backend testing. This ensures all backends (CPU, CUDA, Vulkan, OneAPI) produce consistent results.

## Conversion Status

### ✅ Completed (2/62 files)
1. `test_tensor.cpp` - Core tensor operations
2. `test_ops.cpp` - Tensor creation operations

### 🔄 In Progress (60/62 files remaining)
Files needing conversion:
- test_autograd.cpp (HIGH PRIORITY)
- test_broadcasting.cpp (HIGH PRIORITY)
- test_transforms.cpp (HIGH PRIORITY)
- test_linear.cpp
- test_losses.cpp
- test_optimizers.cpp
- test_cpu_kernels.cpp
- test_device.cpp
- test_albert_t5.cpp
- test_attention.cpp
- test_autocast.cpp
- test_bert.cpp
- test_caching_allocator.cpp
- test_chunk.cpp
- test_classic_models.cpp
- test_convnext.cpp
- test_cublas_cudnn.cpp
- test_dataloader.cpp
- test_data_parallel.cpp
- test_data_parallel_single_gpu.cpp
- test_deeplabv3plus.cpp
- test_detection_components.cpp
- test_detection_ops.cpp
- test_distillation.cpp
- test_efficientnet.cpp
- test_electra.cpp
- test_embedding.cpp
- test_faster_rcnn.cpp
- test_fp16.cpp
- test_fused_ops.cpp
- test_fusion_optimizer.cpp
- test_gpt.cpp
- test_gradient_checkpoint.cpp
- test_grad_scaler.cpp
- test_gru.cpp
- test_jit.cpp
- test_losses_advanced.cpp
- test_lstm.cpp
- test_mask_rcnn.cpp
- test_mobilenet_v2_v3.cpp
- test_model_checkpoint.cpp
- test_model_hub.cpp
- test_oneapi_backend.cpp
- test_onnx_export.cpp
- test_optimizers_extended.cpp
- test_pruning.cpp
- test_quantization.cpp
- test_resnet.cpp
- test_rnn.cpp
- test_roberta.cpp
- test_roberta_electra.cpp
- test_schedulers_advanced.cpp
- test_simd_ops.cpp
- test_swin_transformer.cpp
- test_transformer.cpp
- test_unet.cpp
- test_vision_components.cpp
- test_vit.cpp
- test_yolo.cpp

## Test Results (Converted Files)

### Current Backend Status
- **CPU**: ✅ Working (20/20 tests passing)
- **CUDA**: ✅ Working (20/20 tests passing)
- **Vulkan**: ⚠️ Backend loading issues (0/20 tests run, 20 skipped)
- **OneAPI**: ⚠️ Backend loading issues (0/20 tests run, 20 skipped)

### Test Execution Summary
```
Running 80 tests from 2 test suites
PASSED: 40 tests (20 CPU + 20 CUDA)
SKIPPED: 40 tests (20 Vulkan + 20 OneAPI)
FAILED: 0 tests
Time: 614ms total
```

### Backend-Specific Notes

#### CPU Backend
- All tests passing
- Fastest execution time
- Used as reference for correctness

#### CUDA Backend
- All tests passing
- Device transfer tests verified
- Tensor operations consistent with CPU

#### Vulkan Backend
- Backend loads but availability check fails
- Need to investigate `isBackendAvailable()` implementation
- May need Vulkan device initialization fixes

#### OneAPI Backend
- Backend fails to load
- Error: "Failed to load library: /home/lee/Projects/Tenzor/bin/tenzor_backend_oneapi.so"
- Likely dependency or compilation issue

## Conversion Pattern

### Before (CPU-only test):
```cpp
TEST(TensorTest, Creation) {
    auto t = zeros({2, 3}, DType::Float32, Device::cpu());
    EXPECT_EQ(t.ndim(), 2);
    // ... assertions ...
}
```

### After (Multi-backend test):
```cpp
class TensorBackendTest : public BackendTest {};

TEST_P(TensorBackendTest, Creation) {
    auto t = zeros({2, 3}, DType::Float32, device);  // 'device' from fixture
    EXPECT_EQ(t.ndim(), 2);
    // ... assertions ...
}

INSTANTIATE_BACKEND_TESTS(TensorBackendTest);
```

### Key Changes:
1. Include `../backend_test_fixture.hpp`
2. Change `TEST()` to `TEST_P()` with fixture class
3. Replace `Device::cpu()` with `device` variable
4. Add `.to(Device::cpu())` before accessing data pointers
5. Add device name to assertion messages for debugging
6. Remove `main()` function (use `gtest_main`)
7. Add `INSTANTIATE_BACKEND_TESTS()` macro

## Next Steps

### High Priority (Core Operations)
1. Convert `test_autograd.cpp` - Automatic differentiation
2. Convert `test_broadcasting.cpp` - Broadcasting rules
3. Convert `test_transforms.cpp` - Tensor transformations

### Medium Priority (Neural Network Components)
4. Convert `test_linear.cpp` - Linear layers
5. Convert `test_losses.cpp` - Loss functions
6. Convert `test_optimizers.cpp` - Optimizers

### Fix Backend Loading Issues
- Debug Vulkan backend availability check
- Fix OneAPI backend library loading
- Verify all backends can be initialized properly

## Performance Impact
- No significant performance degradation observed
- Test execution time: 614ms for 80 tests
- Parallel backend testing helps catch backend-specific bugs early

## Future Work
- Automate test conversion with script
- Add backend performance comparison tests
- Create CI/CD pipeline for multi-backend testing
- Add tolerance ranges for numerical precision differences across backends
