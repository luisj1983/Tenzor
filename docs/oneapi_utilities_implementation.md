# OneAPI Backend Utility Operations Implementation

## Summary

Implemented three utility operations for the OneAPI backend: `cat`, `clamp`, and `sign`.

## Files Created

### 1. `/home/lee/Projects/Tenzor/src/backends/oneapi/kernels/utilities.cpp`

Complete SYCL implementation of utility operations with full error handling and support for Float32 and Float64 dtypes.

## Files Modified

### 1. `/home/lee/Projects/Tenzor/src/backends/oneapi/oneapi_backend.cpp`

**Added Forward Declarations (lines 107-110):**
```cpp
// Utility operations
auto cat_kernel(std::span<const Tensor> tensors, int64_t dim, sycl::queue& queue) -> Tensor;
auto clamp_kernel(const Tensor& input, float min_val, float max_val, sycl::queue& queue) -> Tensor;
auto sign_kernel(const Tensor& input, sycl::queue& queue) -> Tensor;
```

**Added Dispatch Cases (lines 652-670):**
```cpp
// Utility operations
else if (op_name == "cat") {
    if (inputs.empty()) throw std::invalid_argument("cat requires at least one input");
    int64_t dim = attrs.contains("dim") ? std::stoll(attrs.at("dim")) : 0;
    return {oneapi::cat_kernel(inputs, dim, queue)};
}
else if (op_name == "clamp") {
    if (inputs.size() != 1) throw std::invalid_argument("clamp requires 1 input");
    if (!attrs.contains("min") || !attrs.contains("max")) {
        throw std::invalid_argument("clamp requires 'min' and 'max' attributes");
    }
    float min_val = std::stof(attrs.at("min"));
    float max_val = std::stof(attrs.at("max"));
    return {oneapi::clamp_kernel(inputs[0], min_val, max_val, queue)};
}
else if (op_name == "sign") {
    if (inputs.size() != 1) throw std::invalid_argument("sign requires 1 input");
    return {oneapi::sign_kernel(inputs[0], queue)};
}
```

## Implementation Details

### 1. Cat (Concatenate) Operation

**Function Signature:**
```cpp
auto cat_kernel(std::span<const Tensor> tensors, int64_t dim, sycl::queue& queue) -> Tensor
```

**Features:**
- Concatenates an arbitrary number of input tensors along a specified dimension
- Validates that all tensors have compatible shapes (same shape except in concat dimension)
- Validates all tensors have the same dtype
- Handles negative dimension indexing
- Optimized parallel copying using SYCL parallel_for
- Supports Float32 and Float64 dtypes

**Algorithm:**
1. Validates inputs (at least one tensor required)
2. Normalizes dimension index (handles negative values)
3. Validates all tensors have compatible shapes and same dtype
4. Computes output shape (sum of sizes along concat dimension)
5. For each input tensor:
   - Decomposes data into slices
   - Parallel copies each slice to appropriate offset in output
6. Returns concatenated tensor

**Memory Layout:**
- Uses strided memory access for efficient copying
- Calculates proper offsets for each tensor based on concat dimension
- Handles multi-dimensional tensors correctly

**Example Usage:**
```cpp
// Concatenate along dimension 0
Tensor a({2, 3}, DType::Float32, Device::oneapi(0));  // Shape: [2, 3]
Tensor b({1, 3}, DType::Float32, Device::oneapi(0));  // Shape: [1, 3]
Tensor result = cat_kernel({a, b}, 0, queue);         // Shape: [3, 3]
```

### 2. Clamp Operation

**Function Signature:**
```cpp
auto clamp_kernel(const Tensor& input, float min_val, float max_val, sycl::queue& queue) -> Tensor
```

**Features:**
- Clamps tensor values to range [min_val, max_val]
- Element-wise operation with full parallelization
- Validates min_val <= max_val
- Uses SYCL's fmin/fmax for correct IEEE 754 handling
- Supports Float32 and Float64 dtypes

**Algorithm:**
```
output[i] = min(max(input[i], min_val), max_val)
```

**Example Usage:**
```cpp
Tensor input({100}, DType::Float32, Device::oneapi(0));
Tensor result = clamp_kernel(input, -1.0f, 1.0f, queue);  // Clamp to [-1, 1]
```

### 3. Sign Operation

**Function Signature:**
```cpp
auto sign_kernel(const Tensor& input, sycl::queue& queue) -> Tensor
```

**Features:**
- Returns the sign of each element (-1, 0, or +1)
- Element-wise operation with full parallelization
- Correctly handles -0.0 (returns 0.0)
- IEEE 754 compliant
- Supports Float32 and Float64 dtypes

**Algorithm:**
```
sign(x) = -1  if x < 0
        =  0  if x == 0
        = +1  if x > 0
```

**Example Usage:**
```cpp
Tensor input({100}, DType::Float32, Device::oneapi(0));
Tensor result = sign_kernel(input, queue);  // Returns signs
```

## Code Quality

### 1. Error Handling
- All operations validate inputs thoroughly
- Proper error messages for invalid arguments
- Dimension validation with negative index support
- Shape compatibility checking for cat operation
- Min/max validation for clamp operation

### 2. Performance
- Fully parallelized SYCL kernels
- Efficient memory access patterns
- Optimized stride calculations for cat operation
- Direct element-wise operations for clamp and sign

### 3. Type Safety
- Template-based typed pointer access
- Proper const correctness
- SYCL kernel naming classes for proper kernel identification

### 4. IEEE 754 Compliance
- Uses SYCL's fmin/fmax for correct NaN handling in clamp
- Correct handling of -0.0 in sign operation
- Proper floating-point comparisons

### 5. Code Patterns
- Follows existing OneAPI backend patterns
- Consistent with math.cpp, transform.cpp, and other kernel files
- Proper SYCL queue synchronization with .wait()
- Uses helper functions for common operations

## Testing

The implementation can be tested using:

```cpp
// Test cat operation
auto backend = Backend::create("oneapi");
Tensor a = Tensor({2, 3}, DType::Float32, Device::oneapi(0));
Tensor b = Tensor({1, 3}, DType::Float32, Device::oneapi(0));
OpAttributes attrs;
attrs["dim"] = "0";
auto result = backend->dispatch("cat", {a, b}, attrs);

// Test clamp operation
Tensor input = Tensor({100}, DType::Float32, Device::oneapi(0));
OpAttributes attrs_clamp;
attrs_clamp["min"] = "-1.0";
attrs_clamp["max"] = "1.0";
auto clamped = backend->dispatch("clamp", {input}, attrs_clamp);

// Test sign operation
auto signs = backend->dispatch("sign", {input}, {});
```

## Compilation

The implementation compiles successfully with Intel oneAPI DPC++/C++ Compiler:
```bash
icpx -fsycl -std=c++20 -I./include -c src/backends/oneapi/kernels/utilities.cpp
```

## Future Enhancements

1. **Additional dtypes**: Add support for Int32, Int64, and other numeric types
2. **Optimization**: Consider using SYCL local memory for cat operation on large tensors
3. **Broadcasting**: Implement broadcasting support for more flexible concatenation
4. **In-place operations**: Add in-place versions of clamp and sign for memory efficiency
5. **Half precision**: Add Float16/BFloat16 support when needed

## Notes

- All operations follow the existing OneAPI backend patterns
- No stubs or placeholders - complete implementations
- Proper SYCL error handling through queue.wait()
- Compatible with existing backend infrastructure
- Ready for integration into the full build system
