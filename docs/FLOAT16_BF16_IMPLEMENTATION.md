# Float16 and BFloat16 Support in Tenzor

## Overview

Tenzor now supports mixed precision training with IEEE 754 Float16 (half precision) and Google's BFloat16 data types. This implementation enables efficient training on modern GPUs with Tensor Cores while maintaining numerical stability.

## Implementation Summary

### 1. Core Type Definitions

**Location**: `/home/lee/Projects/Tenzor/include/tenzor/core/dtype.hpp`

- Added `Float16` struct with IEEE 754 half-precision format (1 sign, 5 exponent, 10 mantissa bits)
- Added `BFloat16` struct with Brain Float format (1 sign, 8 exponent, 7 mantissa bits)
- Implemented bidirectional conversions between FP32/FP16/BF16
- Added proper handling of special values (infinity, NaN, denormals)
- Updated `ScalarType` concept to include Float16 and BFloat16

### 2. Conversion Functions

**Location**: `/home/lee/Projects/Tenzor/src/core/dtype.cpp`

**Float16 Conversion Features**:
- IEEE 754-2008 compliant conversion
- Proper rounding to nearest even
- Overflow/underflow handling
- Denormalized number support
- Range: approximately ±65,504
- Precision: ~3 decimal digits

**BFloat16 Conversion Features**:
- Simple truncation with rounding
- Same exponent range as Float32
- Easier to use for deep learning
- Range: approximately ±3.4×10^38
- Precision: ~2 decimal digits

### 3. CUDA Native Support

**Location**: `/home/lee/Projects/Tenzor/src/backends/cuda/cuda_fp16.hpp`

Provides efficient GPU operations using CUDA intrinsics:
- Conversion to/from CUDA `__half` and `__nv_bfloat16` types
- Vectorized operations with `__half2` and `__nv_bfloat162`
- Tensor Core compatibility checks
- Native arithmetic operations (add, mul) using CUDA intrinsics

**Tensor Core Support**:
- FP16: Available on Volta+ (compute capability 7.0+)
- BF16: Available on Ampere+ (compute capability 8.0+)

### 4. Test Suite

**Location**: `/home/lee/Projects/Tenzor/tests/unit/test_fp16.cpp`

Comprehensive test coverage including:
- Basic type conversions
- Special value handling (inf, NaN, zero)
- Precision and accuracy tests
- Round-trip conversion tests
- Edge cases (subnormals, overflow/underflow)
- Tensor creation with FP16/BF16 dtypes

## Usage Examples

### Creating Half-Precision Tensors

```cpp
#include "tenzor/tenzor.hpp"

using namespace tenzor;

// Create Float16 tensor
Tensor fp16_tensor({4, 4}, DType::Float16, Device::cpu());

// Create BFloat16 tensor
Tensor bf16_tensor({4, 4}, DType::BFloat16, Device::cpu());

// Create on GPU (if available)
Tensor gpu_fp16 = fp16_tensor.cuda();
```

### Manual Type Conversion

```cpp
// Convert float to Float16
float original = 3.14159f;
Float16 fp16(original);
float recovered = static_cast<float>(fp16);

// Convert float to BFloat16
BFloat16 bf16(original);
float recovered_bf = static_cast<float>(bf16);
```

### Mixed Precision Training Pattern

```cpp
// Typical mixed precision training workflow
auto model = YourModel();
auto optimizer = optim::Adam(model.parameters(), 0.001);
auto scaler = nn::amp::GradScaler();

for (auto& batch : dataloader) {
    // Forward pass in FP16/BF16
    auto inputs_fp16 = batch.inputs.to(DType::Float16);
    auto outputs = model.forward(inputs_fp16);
    
    // Loss computation (typically in FP32)
    auto loss = criterion(outputs.to(DType::Float32), batch.targets);
    
    // Scaled backward pass
    auto scaled_loss = scaler.scale(loss);
    scaled_loss.backward();
    
    // Unscale and optimize
    scaler.unscale_(optimizer);
    optimizer.step();
    scaler.update();
}
```

### CUDA-Specific Operations

```cpp
#ifdef __CUDACC__
#include "cuda_fp16.hpp"

// Convert to CUDA native types
Float16 fp16(1.5f);
__half cuda_half = tenzor::cuda::to_cuda_half(fp16);

// Use CUDA intrinsics
Float16 a(2.0f), b(3.0f);
Float16 result = tenzor::cuda::mul_fp16(a, b);

// Check Tensor Core support
if (tenzor::cuda::supports_fp16_tensor_cores()) {
    // Use optimized FP16 matrix multiplication
}
#endif
```

## Performance Considerations

### When to Use Float16

**Advantages**:
- 2x memory reduction vs Float32
- 2-4x faster on Tensor Core GPUs
- Better precision than BFloat16 for small values
- Standard IEEE 754 format

**Disadvantages**:
- Limited range (±65,504)
- Can cause overflow/underflow in some models
- Requires gradient scaling for training

**Best for**:
- Inference workloads
- Models with small activation ranges
- Vision tasks (CNNs)
- Inference on mobile/edge devices

### When to Use BFloat16

**Advantages**:
- Same range as Float32
- Less likely to overflow/underflow
- Simpler conversion (truncation)
- Preferred by Google/TPU ecosystem

**Disadvantages**:
- Lower precision than Float16
- Only supported on newer GPUs (Ampere+)

**Best for**:
- Training large language models
- Training with minimal hyperparameter tuning
- Direct Float32 replacement
- Research and experimentation

## Technical Details

### Float16 Bit Layout
```
┌─────┬─────────┬────────────────────┐
│Sign │Exponent │     Mantissa       │
│  1  │    5    │        10          │
└─────┴─────────┴────────────────────┘
Range: ±6.55×10^4
Precision: ~3 decimal digits
```

### BFloat16 Bit Layout
```
┌─────┬─────────┬────────────────────┐
│Sign │Exponent │     Mantissa       │
│  1  │    8    │         7          │
└─────┴─────────┴────────────────────┘
Range: ±3.4×10^38 (same as Float32)
Precision: ~2 decimal digits
```

### Conversion Accuracy

**Float16**:
- Relative error: <0.1% for values in normal range
- Absolute error: <0.001 for small values

**BFloat16**:
- Relative error: <1% for values in normal range
- Direct truncation of Float32 mantissa
- No denormal number support

## Future Enhancements

1. **CPU Backend**: Add optimized FP16/BF16 kernels using SIMD instructions
2. **Autocast**: Automatic mixed precision context manager
3. **Gradient Checkpointing**: Memory-efficient training with FP16
4. **Quantization**: INT8 quantization building on FP16 infrastructure
5. **Operator Fusion**: Fused FP16 operations for better performance

## Files Modified

1. `/home/lee/Projects/Tenzor/include/tenzor/core/dtype.hpp` - Type definitions
2. `/home/lee/Projects/Tenzor/src/core/dtype.cpp` - Conversion implementations
3. `/home/lee/Projects/Tenzor/src/backends/cuda/cuda_fp16.hpp` - CUDA utilities
4. `/home/lee/Projects/Tenzor/tests/unit/test_fp16.cpp` - Test suite
5. `/home/lee/Projects/Tenzor/tests/CMakeLists.txt` - Test configuration
6. `/home/lee/Projects/Tenzor/src/backends/cuda/CMakeLists.txt` - Build configuration

## References

- IEEE 754-2008 Standard for Floating-Point Arithmetic
- [Brain Floating Point Format](https://en.wikipedia.org/wiki/Bfloat16_floating-point_format)
- [NVIDIA Tensor Cores](https://www.nvidia.com/en-us/data-center/tensor-cores/)
- [Mixed Precision Training](https://arxiv.org/abs/1710.03740)

## Testing

Run the FP16 test suite:
```bash
cd build
cmake .. -DTENZOR_BUILD_CUDA=ON  # For CUDA support
make test_fp16
./tests/test_fp16
```

Expected output:
- All Float16 conversion tests pass
- All BFloat16 conversion tests pass
- Precision tests verify <1% relative error
- Special value tests handle inf/NaN correctly
