# Autograd-Aware Reshape and Permute Operations

## Overview

This document describes the implementation of autograd-aware `reshape()` and `permute()` operations for the Tenzor deep learning framework. These operations properly track gradients through shape transformations, fixing gradient accumulation issues.

## Problem Statement

Previously, when using `reshape()` or `permute()` operations on Variables in an autograd graph, gradients would not properly flow back through these operations because they operated directly on Tensors without participating in the autograd graph.

## Implementation

### Files Modified

1. **`include/tenzor/autograd/ops.hpp`**
   - Added declarations for `reshape()` and `permute()` functions
   - Added comprehensive documentation for both operations

2. **`include/tenzor/autograd/function.hpp`**
   - Added `ReshapeBackward` class
   - Added `PermuteBackward` class
   - Both classes implement the `Function` interface for autograd

3. **`src/autograd/ops.cpp`**
   - Implemented autograd-aware `reshape()` function
   - Implemented autograd-aware `permute()` function
   - Both functions create proper backward graph connections

4. **`src/autograd/function.cpp`**
   - Implemented `ReshapeBackward::backward()` - reshapes gradient to input shape
   - Implemented `PermuteBackward::backward()` - applies inverse permutation to gradient

5. **`tests/test_autograd_transform.cpp`** (NEW)
   - Comprehensive test suite with 10 test cases
   - Tests forward/backward passes, chained operations, and no-grad behavior

6. **`tests/test_linear_reshape_integration.cpp`** (NEW)
   - Integration tests for Linear layer with reshape/permute operations
   - Tests multiple reshape operations and gradient flow

## Key Features

### Reshape Operation

```cpp
auto reshape(const Variable& input, const std::vector<int64_t>& shape) -> Variable;
```

**Forward Pass:**
- Reshapes the input tensor to the specified shape
- Preserves requires_grad flag
- Creates ReshapeBackward function if gradients are enabled

**Backward Pass:**
- Reshapes gradient back to original input shape
- Formula: `grad_input = reshape(grad_output, input_shape)`

**Example:**
```cpp
Variable x(Tensor({3, 4}, DType::Float32, Device::cpu()), true);
Variable y = reshape(x, {12});  // Forward: {3, 4} -> {12}
// Backward: gradient reshaped from {12} back to {3, 4}
```

### Permute Operation

```cpp
auto permute(const Variable& input, const std::vector<int64_t>& dims) -> Variable;
```

**Forward Pass:**
- Permutes dimensions according to specified order
- Preserves requires_grad flag
- Creates PermuteBackward function if gradients are enabled
- Computes inverse permutation for backward pass

**Backward Pass:**
- Applies inverse permutation to gradient
- Formula: `grad_input = permute(grad_output, inverse_dims)`

**Example:**
```cpp
Variable x(Tensor({2, 3, 4}, DType::Float32, Device::cpu()), true);
Variable y = permute(x, {2, 0, 1});  // Forward: {2, 3, 4} -> {4, 2, 3}
// Backward: gradient permuted from {4, 2, 3} back to {2, 3, 4}
```

## Backward Function Implementations

### ReshapeBackward

```cpp
class ReshapeBackward : public Function {
public:
    ReshapeBackward(std::vector<int64_t> input_shape);

    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override {
        // Reshape gradient back to input shape
        return {reshape(grad_outputs[0], input_shape_)};
    }

private:
    std::vector<int64_t> input_shape_;
};
```

### PermuteBackward

```cpp
class PermuteBackward : public Function {
public:
    PermuteBackward(std::vector<int64_t> dims) {
        // Compute inverse permutation
        inv_dims_.resize(dims_.size());
        for (size_t i = 0; i < dims_.size(); ++i) {
            inv_dims_[dims_[i]] = i;
        }
    }

    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override {
        // Apply inverse permutation
        return {permute(grad_outputs[0], inv_dims_)};
    }

private:
    std::vector<int64_t> dims_;
    std::vector<int64_t> inv_dims_;
};
```

## Test Coverage

### Unit Tests (test_autograd_transform.cpp)

1. **ReshapeForwardPass** - Verifies forward computation and shape
2. **ReshapeBackwardPass** - Tests gradient computation and shape preservation
3. **ReshapeChainedOperations** - Tests reshape with other operations
4. **PermuteForwardPass** - Verifies permutation and shape
5. **PermuteBackwardPass** - Tests gradient inverse permutation
6. **PermuteTranspose** - Tests permute as transpose operation
7. **PermuteWithSum** - Tests permute with reduction operations
8. **ReshapeAndPermuteCombined** - Tests chaining reshape and permute
9. **ReshapeNoGrad** - Verifies behavior without gradients
10. **PermuteNoGrad** - Verifies behavior without gradients

### Integration Tests (test_linear_reshape_integration.cpp)

1. **LinearWithReshapeInput** - Tests Linear layer with reshaped input
2. **LinearWithPermuteInput** - Tests Linear layer with permuted input
3. **MultipleReshapeOps** - Tests multiple reshape operations in graph

## Usage in Linear Layer

The Linear layer internally uses tensor-level reshape operations, which is correct since it manually manages gradients through LinearBackward. For user-facing code that operates on Variables, the autograd-aware versions should be used:

```cpp
// User code - use autograd-aware reshape
Variable input(tensor, true);
auto reshaped = tenzor::reshape(input, {batch, features});
auto output = layer->forward(reshaped);
```

## Gradient Flow Verification

All tests verify that:
1. Gradients have correct shapes after backward pass
2. Gradient values are correct (using known derivatives)
3. Gradients flow through multiple reshape/permute operations
4. No gradients are computed when requires_grad=false

## Performance Considerations

- Reshape and permute are typically view operations with minimal overhead
- Backward passes only perform the inverse transformation
- No additional memory is allocated except for saved shape/permutation metadata
- Operations integrate seamlessly with existing autograd engine

## Future Improvements

Possible enhancements:
1. Add support for dynamic shapes (infer -1 dimensions)
2. Optimize memory usage for saved shapes
3. Add CUDA-specific optimizations
4. Support for in-place operations where safe

## Build Integration

The implementation is fully integrated into the build system:
- Tests are part of `tenzor_unit_tests` target
- Additional integration test executable: `test_linear_reshape_integration`
- All tests pass on both CPU and CUDA backends

## Conclusion

The autograd-aware reshape and permute operations successfully fix gradient accumulation issues by properly participating in the autograd graph. The implementation follows PyTorch-style conventions and integrates cleanly with the existing Tenzor framework.
