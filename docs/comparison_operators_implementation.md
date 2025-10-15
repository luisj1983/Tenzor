# Comparison Operators Implementation

## Summary

Implemented all six element-wise comparison operators for the Tensor class in Tenzor. These operators are essential for masking, filtering, and conditional logic in neural networks.

## Implementation Details

### Files Modified

1. **include/tenzor/ops/math.hpp**
   - Added declarations for comparison functions: `eq`, `ne`, `lt`, `le`, `gt`, `ge`
   - Each function returns a boolean tensor (DType::Bool)

2. **src/ops/math.cpp**
   - Implemented comparison functions following the existing dispatcher pattern
   - All functions ensure tensors are contiguous before comparison
   - Dispatch to backend kernels for actual computation

3. **include/tenzor/core/tensor.hpp**
   - Added declarations for comparison operators: `==`, `!=`, `<`, `>`, `<=`, `>=`
   - Documented all operators with parameter and return type information

4. **src/core/tensor.cpp**
   - Implemented operators to delegate to the corresponding comparison functions
   - Replaced stub implementations at lines 947-964

5. **tests/test_comparison_operators.cpp** (NEW)
   - Comprehensive test suite covering all comparison operators
   - Tests 1D and 2D tensors
   - Verifies function versions match operator versions
   - Tests return type (DType::Bool)

6. **tests/CMakeLists.txt**
   - Added test_comparison_operators target

## Operators Implemented

### 1. Equal (==)
```cpp
Tensor a = from_data<float>({1.0f, 2.0f, 3.0f}, {3});
Tensor b = from_data<float>({1.0f, 0.0f, 3.0f}, {3});
Tensor result = a == b;  // {true, false, true}
```

### 2. Not Equal (!=)
```cpp
Tensor result = a != b;  // {false, true, false}
```

### 3. Less Than (<)
```cpp
Tensor result = a < b;  // {false, false, false}
```

### 4. Less Than or Equal (<=)
```cpp
Tensor result = a <= b;  // {true, false, true}
```

### 5. Greater Than (>)
```cpp
Tensor result = a > b;  // {false, true, false}
```

### 6. Greater Than or Equal (>=)
```cpp
Tensor result = a >= b;  // {true, true, true}
```

## Features

### Broadcasting Support
All comparison operators support broadcasting, following the same rules as arithmetic operators:
```cpp
Tensor a({3, 1}, DType::Float32);  // Shape: (3, 1)
Tensor b({1, 4}, DType::Float32);  // Shape: (1, 4)
Tensor result = a > b;              // Shape: (3, 4)
```

### Boolean Return Type
All comparisons return tensors with `DType::Bool`:
```cpp
Tensor result = a == b;
assert(result.dtype() == DType::Bool);
```

### Backend Dispatch
Operations are dispatched to appropriate backends (CPU/CUDA/ROCm) through the standard dispatcher:
```cpp
// Automatically uses CUDA kernels for GPU tensors
Tensor gpu_a = cpu_a.cuda();
Tensor gpu_b = cpu_b.cuda();
Tensor gpu_result = gpu_a > gpu_b;  // Runs on GPU
```

### Contiguous Memory Handling
All comparison functions ensure tensors are contiguous before operation, handling non-contiguous views correctly:
```cpp
Tensor transposed = tensor.transpose(0, 1);  // Non-contiguous
Tensor result = transposed == other;          // Automatically made contiguous
```

## Usage Examples

### Masking
```cpp
Tensor data = from_data<float>({1.0f, 2.0f, 3.0f, 4.0f, 5.0f}, {5});
Tensor threshold = full({5}, 3.0f);
Tensor mask = data > threshold;  // {false, false, false, true, true}
```

### Filtering
```cpp
Tensor values = from_data<float>({-2.0f, -1.0f, 0.0f, 1.0f, 2.0f}, {5});
Tensor zero = zeros({5});
Tensor is_positive = values > zero;  // {false, false, false, true, true}
```

### Conditional Logic
```cpp
Tensor a = from_data<float>({1.0f, 2.0f, 3.0f}, {3});
Tensor b = from_data<float>({2.0f, 2.0f, 2.0f}, {3});
Tensor eq_mask = a == b;   // {false, true, false}
Tensor lt_mask = a < b;    // {true, false, false}
Tensor gt_mask = a > b;    // {false, false, true}
```

## Function Variants

Both operator and function forms are available:

```cpp
// Operator form
Tensor result1 = a == b;

// Function form
Tensor result2 = eq(a, b);

// Both produce identical results
assert(result1.dtype() == result2.dtype());
```

## Backend Requirements

The implementation dispatches to backend kernels with the following operation names:
- `"eq"` - Element-wise equal
- `"ne"` - Element-wise not equal
- `"lt"` - Element-wise less than
- `"le"` - Element-wise less than or equal
- `"gt"` - Element-wise greater than
- `"ge"` - Element-wise greater than or equal

Backend implementations (CPU/CUDA/ROCm) need to provide kernels for these operations that:
1. Accept two input tensors
2. Return a boolean tensor (DType::Bool)
3. Support broadcasting
4. Handle all supported dtypes

## Testing

The test suite (`test_comparison_operators.cpp`) verifies:
- Correct return type (DType::Bool)
- Proper shape handling
- Correct comparison results for all operators
- 2D tensor support
- Function vs operator equivalence

Run tests with:
```bash
cd build
cmake --build . --target test_comparison_operators
./bin/test_comparison_operators
```

## Integration with Neural Networks

These operators enable important neural network operations:

### Attention Masking
```cpp
// Create attention mask for sequence padding
Tensor lengths = from_data<int32_t>({5, 3, 4}, {3});
Tensor positions = arange(0, 5).unsqueeze(0).expand({3, 5});
Tensor mask = positions < lengths.unsqueeze(1);
```

### ReLU Alternative
```cpp
// Boolean mask for positive values
Tensor x = randn({3, 4});
Tensor positive_mask = x > zeros_like(x);
```

### Dropout Masking
```cpp
// Create dropout mask
Tensor rand_vals = rand({3, 4});
Tensor dropout_mask = rand_vals > full({3, 4}, 0.5f);
```

## Performance Considerations

1. **Contiguous Check**: All operations check if tensors are contiguous and create contiguous copies if needed
2. **Broadcasting**: Operations support full broadcasting semantics
3. **Backend Dispatch**: Operations are dispatched to hardware-accelerated backends (CUDA/ROCm)
4. **Memory Layout**: Result tensors use efficient contiguous memory layout

## Future Enhancements

Potential improvements for future versions:
1. Scalar comparison operators (e.g., `tensor > 0.5f`)
2. In-place comparison operations
3. Vectorized CPU implementations using SIMD
4. Optimized GPU kernels with coalesced memory access
5. Support for complex number comparisons (magnitude-based)

## Compatibility

- **C++ Standard**: Requires C++20 (for concepts and span)
- **Backends**: CPU, CUDA, ROCm (backend kernels required)
- **DTypes**: All numeric dtypes supported (Bool, Int8-Int64, UInt8-UInt64, Float16-Float64, Complex64-Complex128)

## References

- PyTorch comparison operators: https://pytorch.org/docs/stable/torch.html#comparison-ops
- NumPy comparison functions: https://numpy.org/doc/stable/reference/routines.logic.html
- Broadcasting semantics: https://pytorch.org/docs/stable/notes/broadcasting.html
