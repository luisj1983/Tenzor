# Division by Zero Fixes - Tenzor Codebase

## Summary

This document details all fixes applied to eliminate integer division by zero undefined behavior throughout the Tenzor codebase. These fixes ensure robust error handling and prevent potential crashes or undefined behavior.

## Date: 2025-10-10

## Files Modified

### 1. `/home/lee/Projects/Tenzor/src/backends/cpu/kernels/batchnorm.cpp`

**Issue:** Division by `total_elements` in mean/variance computation could cause division by zero for empty tensors.

**Fix:** Added validation check at the beginning of `batchnorm_mean_var_impl`:
```cpp
// Check for division by zero
if (total_elements == 0) {
    throw std::runtime_error("BatchNorm2d: Cannot compute mean/variance for empty tensor (total_elements = 0)");
}
```

Also added check in `batchnorm_backward_impl`:
```cpp
// Check for division by zero
if (total_elements == 0) {
    throw std::runtime_error("BatchNorm2d backward: Cannot compute gradients for empty tensor (total_elements = 0)");
}
```

**Location:** Lines 32-35 and 336-339

---

### 2. `/home/lee/Projects/Tenzor/src/backends/cuda/kernels/batchnorm.cu`

**Issue:** Similar division by zero issues in CUDA kernels for BatchNorm operations.

**Fix:** Added validation in host functions `batchnorm2d_mean_var` and `batchnorm2d_backward`:
```cpp
// Check for division by zero
int64_t total_elements = N * H * W;
if (total_elements == 0) {
    throw std::runtime_error("BatchNorm2d CUDA: Cannot compute mean/variance for empty tensor (N*H*W = 0)");
}
```

**Location:** Lines 474-478 and 623-627

---

### 3. `/home/lee/Projects/Tenzor/src/core/tensor.cpp`

**Issue:** In `reshape()` function, when inferring dimensions (using -1), division by `total` could be zero.

**Fix:** Added check before performing division:
```cpp
if (total == 0) {
    throw std::invalid_argument("Cannot infer dimension: product of known dimensions is zero");
}
```

**Location:** Lines 440-442

---

### 4. `/home/lee/Projects/Tenzor/src/backends/cuda/kernels/conv2d.cu`

**Issue:** Multiple division operations by `stride` and `groups` parameters that could be zero.

**Fix:** Added validation checks in forward and backward functions:
```cpp
// Validate parameters to prevent division by zero
if (stride == 0) {
    throw std::invalid_argument("Conv2d: stride cannot be zero");
}
if (groups == 0) {
    throw std::invalid_argument("Conv2d: groups cannot be zero");
}
```

Also added check in `calculate_output_size` helper function:
```cpp
#ifndef __CUDA_ARCH__
// Host-side validation (not in device code)
if (stride == 0) {
    throw std::invalid_argument("Conv2d: stride cannot be zero");
}
#endif
```

**Location:** Lines 59-64, 375-381, 543-549

---

### 5. `/home/lee/Projects/Tenzor/src/nn/layers/batchnorm.cpp`

**Issue:** High-level BatchNorm layer could compute statistics on empty batches.

**Fix:** Added validation in `forward()` method:
```cpp
// Validate to prevent division by zero
int64_t batch_size = N * spatial_size;
if (training_ && batch_size == 0) {
    throw std::runtime_error("BatchNorm2d: Cannot compute statistics for empty batch (N * H * W = 0)");
}
```

**Location:** Lines 137-141

---

### 6. `/home/lee/Projects/Tenzor/src/nn/layers/pooling.cpp`

**Issue:** Division by `stride` in pooling output size calculation.

**Fix:** Added validation in `calculate_pool_output_size` helper:
```cpp
if (stride == 0) {
    throw std::invalid_argument("Pooling: stride cannot be zero");
}
```

**Location:** Lines 18-20

---

### 7. `/home/lee/Projects/Tenzor/src/nn/optim/scheduler.cpp`

**Issue:** Division by `step_size_` and `T_max_` in learning rate schedulers.

**Fixes:**

For StepLR:
```cpp
if (step_size_ == 0) {
    throw std::runtime_error("StepLR: step_size cannot be zero");
}
```
**Location:** Lines 44-46

For CosineAnnealingLR:
```cpp
if (T_max_ == 0) {
    throw std::runtime_error("CosineAnnealingLR: T_max cannot be zero");
}
```
**Location:** Lines 180-182

---

## Pre-existing Safe Division Operations

The following operations already had proper validation and were NOT modified:

### `/home/lee/Projects/Tenzor/src/ops/creation.cpp`
- `arange()` function already validates `step != 0` at line 285-287

---

## Testing

Test suite was run with `ctest --output-on-failure -j$(nproc)`:
- **Result:** No new test failures introduced by these changes
- **Pre-existing failures:** Some tests were already failing before these changes (unrelated to division by zero)
- **CUDA tests:** Permission issues prevent CUDA test execution (expected in this environment)

---

## Impact Assessment

### Safety Improvements
- **Eliminated undefined behavior**: All integer division by zero cases now throw descriptive exceptions
- **Better error messages**: Users get clear feedback about what went wrong
- **Early detection**: Problems are caught immediately rather than causing silent corruption or crashes

### Performance Impact
- **Negligible**: Validation checks are only performed once per operation at the API boundary
- **No hot path impact**: Checks are not in inner loops or frequently called code

### Backward Compatibility
- **Breaking change**: Code that previously caused undefined behavior will now throw exceptions
- **Justification**: This is a bug fix that prevents undefined behavior; code relying on division by zero was already broken

---

## Recommendations

1. **Review constructor validation**: Consider adding validation in constructors for parameters that will be used in division:
   - `MaxPool2d(stride)`, `AvgPool2d(stride)`, `AdaptiveAvgPool2d(output_size)`
   - `StepLR(step_size)`, `CosineAnnealingLR(T_max)`
   - `Conv2d(stride, groups)`

2. **Add unit tests**: Create specific tests for zero-parameter edge cases:
   ```cpp
   TEST(BatchNormTest, EmptyTensorThrows) {
       auto input = zeros({0, 3, 32, 32});
       BatchNorm2d bn(3);
       EXPECT_THROW(bn.forward(Variable(input, false)), std::runtime_error);
   }
   ```

3. **Documentation**: Update API documentation to specify that these parameters cannot be zero.

4. **Static analysis**: Consider using tools like clang-tidy with `-Wdivision-by-zero` to catch future issues.

---

## Files Summary

**Total files modified:** 7

1. `src/backends/cpu/kernels/batchnorm.cpp` (2 checks added)
2. `src/backends/cuda/kernels/batchnorm.cu` (2 checks added)
3. `src/core/tensor.cpp` (1 check added)
4. `src/backends/cuda/kernels/conv2d.cu` (4 checks added)
5. `src/nn/layers/batchnorm.cpp` (1 check added)
6. `src/nn/layers/pooling.cpp` (1 check added)
7. `src/nn/optim/scheduler.cpp` (2 checks added)

**Total validation checks added:** 13

---

## Conclusion

All identified integer division by zero issues have been addressed with proper validation and error handling. The codebase is now safer and provides better diagnostics when invalid parameters are used.
