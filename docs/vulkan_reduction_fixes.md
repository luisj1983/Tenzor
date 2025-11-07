# Vulkan Backend Reduction Operations Fixes

## Issues Identified and Fixed

### 1. Variance/Std Operations Returning Infinity

**Location:** `/home/lee/Projects/Tenzor/src/backends/vulkan/vulkan_backend.cpp:3470-3482`

**Problem:**
The `dispatchVariance` function was creating a scalar `divisor_tensor` on the Vulkan device and then directly writing to it via `data_ptr()`:

```cpp
// INCORRECT CODE (before fix):
Tensor divisor_tensor(scalar_shape, input.dtype(), input.device());
float* divisor_data = static_cast<float*>(divisor_tensor.data_ptr());
*divisor_data = divisor;  // Writing to GPU memory as if it's CPU memory!
```

This approach doesn't work for GPU tensors because:
- `divisor_tensor.data_ptr()` returns a GPU device pointer
- Writing to `*divisor_data` writes to CPU memory mapped to that address, but doesn't actually transfer data to the GPU
- The GPU tensor remains uninitialized, containing garbage values or inf

**Root Cause:**
When a tensor is allocated on a GPU device (Vulkan, CUDA, etc.), its memory is on the GPU. Direct pointer dereferencing only works for CPU tensors. For GPU tensors, data must be explicitly transferred using device-specific mechanisms or creation functions.

**Fix:**
Use the `full()` function which properly initializes tensors on any device:

```cpp
// CORRECT CODE (after fix):
Tensor divisor_tensor = full({1}, divisor, input.dtype(), input.device());
```

This ensures:
- The tensor is properly allocated on the device
- The value is correctly transferred to the device
- No undefined behavior or garbage values

**Test Affected:**
- `AllBackends/ReductionOpsTest.VarianceStd/vulkan`

### 2. Empty Tensor Reduction

**Location:** `/home/lee/Projects/Tenzor/src/backends/vulkan/vulkan_backend.cpp:1918-1928`

**Problem:**
The `dispatchReduction` function didn't handle empty tensors (numel() == 0). This caused:
- Division by zero in the reduction shader for mean operations
- Undefined behavior when dispatching with 0 workgroups
- Incorrect results for empty tensor reductions

**Fix:**
Added edge case handling before dispatching to the GPU:

```cpp
// Handle edge case: empty tensor
if (input.numel() == 0) {
    // Return appropriate value for empty tensor
    if (op_name == "sum" || op_name == "mean") {
        return full(out_shape, 0.0f, input.dtype(), input.device());
    } else if (op_name == "max") {
        return full(out_shape, -std::numeric_limits<float>::infinity(), input.dtype(), input.device());
    } else if (op_name == "min") {
        return full(out_shape, std::numeric_limits<float>::infinity(), input.dtype(), input.device());
    }
}
```

This returns mathematically correct values:
- Sum/mean of empty set: 0
- Max of empty set: -∞
- Min of empty set: +∞

**Test Affected:**
- `AllBackends/ReductionOpsTest.EmptyTensorReduction/vulkan`

### 3. Variance Edge Cases

**Location:** `/home/lee/Projects/Tenzor/src/backends/vulkan/vulkan_backend.cpp:3473-3477`

**Problem:**
Unbiased variance cannot be computed with:
- Empty tensors (reduce_size == 0)
- Single element (reduce_size == 1, would divide by 0)

**Fix:**
Added validation and return NaN for undefined cases:

```cpp
// Handle edge case: empty tensor or insufficient data for unbiased
if (reduce_size == 0 || (unbiased && reduce_size <= 1)) {
    // Return NaN for empty tensors or when unbiased variance cannot be computed
    return full(out_shape, std::numeric_limits<float>::quiet_NaN(), input.dtype(), input.device());
}
```

### 4. Norm Operation Tensor Initialization

**Location:** `/home/lee/Projects/Tenzor/src/backends/vulkan/vulkan_backend.cpp:3531-3543`

**Problem:**
Similar to variance, the `dispatchNorm` function was creating scalar tensors and directly writing to GPU pointers.

**Fix:**
Replaced manual tensor creation and pointer dereferencing with `full()`:

```cpp
// Before:
Tensor p_tensor(scalar_shape, input.dtype(), input.device());
float* p_data = static_cast<float*>(p_tensor.data_ptr());
*p_data = p;

// After:
Tensor p_tensor = full({1}, p, input.dtype(), input.device());
```

## Additional Changes

### Include Statement Added
Added `#include "../../ops/creation.hpp"` to access the `full()` function for proper tensor initialization.

## Summary of Fixes

| Issue | Root Cause | Solution | Lines Changed |
|-------|------------|----------|---------------|
| Variance returning inf | GPU tensor uninitialized | Use `full()` instead of direct pointer write | 3474-3482, 3501-3504 |
| Empty tensor reduction | No edge case handling | Return appropriate values for empty tensors | 1918-1928 |
| Variance edge cases | Division by zero | Return NaN for undefined cases | 3473-3477 |
| Norm initialization | GPU tensor uninitialized | Use `full()` for scalar tensors | 3532-3540 |

## Key Principle

**Never directly write to GPU tensor pointers obtained via `data_ptr()`**

For GPU tensors:
- Use creation functions: `zeros()`, `ones()`, `full()`
- Use tensor operations to populate values
- Use explicit copy operations: `tensor.to(device)`

Only CPU tensors can have their memory directly accessed via pointer dereferencing.

## Testing

These fixes should resolve:
1. `AllBackends/ReductionOpsTest.VarianceStd/vulkan` - variance/std now compute correctly
2. `AllBackends/ReductionOpsTest.EmptyTensorReduction/vulkan` - empty tensors handled properly

Run tests with:
```bash
./build/bin/tenzor_tests --gtest_filter="*ReductionOpsTest*vulkan*"
```
