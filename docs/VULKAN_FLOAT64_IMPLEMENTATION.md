# Vulkan Float64/Double Precision Implementation

## Summary of Changes

Added Float64 (double precision) support to Vulkan backend shaders and C++ dispatch code.

## Files Modified

### Shaders
1. `/home/lee/Projects/Tenzor/src/backends/vulkan/kernels/math.comp` - Added Float64 support
2. `/home/lee/Projects/Tenzor/src/backends/vulkan/kernels/math_broadcast.comp` - Added Float64 broadcasting
3. `/home/lee/Projects/Tenzor/shaders/vulkan/math.spv` - Recompiled with Float64
4. `/home/lee/Projects/Tenzor/shaders/vulkan/math_broadcast.spv` - Recompiled with Float64

### C++ Code
1. `/home/lee/Projects/Tenzor/src/backends/vulkan/vulkan_backend.cpp`
   - Updated `dispatchBinaryOp()` to map `DType::Float64` to `dtype_code=2`
   - Updated `dispatchUnaryOp()` to support Float64
   - Updated `dispatchUnaryOpWithParam()` to support Float64
   - Added `dtype` field to PushConstants structures

## Implementation Details

### DType Mapping
- `DType::Float32` (enum value 0) → `dtype_code = 0` in shader
- `DType::Int32` (enum value 6) → `dtype_code = 1` in shader
- `DType::Float64` (enum value 1) → `dtype_code = 2` in shader

### Shader Implementation
The shaders now use `double` arrays directly for Float64 operations:

```glsl
#version 450
#extension GL_ARB_gpu_shader_fp64 : enable

layout(binding = 0) buffer InputA { double a[]; };
layout(binding = 1) buffer InputB { double b[]; };
layout(binding = 2) buffer Output { double result[]; };

layout(push_constant) uniform PushConstants {
    uint n;
    uint op;
    float param;
    uint dtype;  // 0=float32, 2=float64
} params;
```

### Vulkan Feature Requirements
- Requires `shaderFloat64` feature to be enabled (line 207 in vulkan_backend.cpp)
- GPU must support VK_KHR_shader_float64 extension
- Both AMD RADV and NVIDIA GPUs tested support this feature

## Limitations

### GLSL Float64 Transcendental Functions
GLSL does not provide double-precision versions of transcendental functions. The following operations fall back to float precision internally:

- `sqrt()` - Uses `double(sqrt(float(val)))`
- `exp()` - Uses `double(exp(float(val)))`
- `log()` - Uses `double(log(float(val)))`
- `pow()` - Uses `double(pow(float(val), param))`

This results in reduced precision for these operations even when using Float64 inputs.

### Supported Operations with Full Float64 Precision
- Arithmetic: add, sub, mul, div
- Negation: neg
- Absolute value: abs
- Sign function: sign
- Rounding: floor, ceil, round, trunc
- Reciprocal: 1/x

## Test Results

### Before Fix
- Float64 tensors were incorrectly treated as Float32 in shaders
- `dtype_code` defaulted to 0 regardless of actual dtype
- Numerical gradients: 46.8, Analytical gradients: 25.6 (huge mismatch)

### After Fix (Current Status)
- Float64 dtype correctly mapped to `dtype_code=2`
- Shaders compiled with Float64 support
- Test still failing - investigating whether operations bypass Vulkan backend

## Known Issues

1. **Gradient Test Failures**: Float64 gradient tests still fail
   - Possible cause: Operations may be executing on CPU backend instead of Vulkan
   - Need to verify operation dispatch path

2. **Shader Linter Conflicts**: Shader files may be auto-formatted/reverted
   - Solution: Compiled .spv binaries are authoritative

## Next Steps

1. Verify that Float64 operations actually dispatch to Vulkan backend
2. Add debug logging to trace execution path
3. Check if gradcheck creates new tensors that bypass device specification
4. Consider implementing Float64 support in other backends for comparison

## Vulkan Capability Verification

```bash
$ vulkaninfo | grep shaderFloat64
shaderFloat64 = true
```

Both test GPUs support Float64:
- AMD Radeon Graphics (RADV RENOIR) - Integrated
- NVIDIA GeForce GTX 1660 Ti - Dedicated

## References

- Vulkan GLSL specification: https://www.khronos.org/registry/vulkan/specs/1.3/html/vkspec.html#spirvenv-capabilities-table
- GLSL double precision: https://www.khronos.org/opengl/wiki/Data_Type_(GLSL)#Floating-point_types
