# Split Operation Implementation

## Overview

Successfully implemented the missing `split()` operation in `/home/lee/Projects/Tenzor/src/ops/transform.cpp`. The split operation divides a tensor into multiple chunks of a specified size along a given dimension, creating efficient views that share storage with the original tensor (zero-copy).

## Implementation Details

### Location
- **File**: `/home/lee/Projects/Tenzor/src/ops/transform.cpp`
- **Lines**: 197-236

### Function Signature
```cpp
auto split(const Tensor& input, int64_t split_size, int64_t dim) -> std::vector<Tensor>
```

### Parameters
- `input`: The tensor to split
- `split_size`: Size of each chunk (except possibly the last one)
- `dim`: Dimension along which to split (supports negative indexing)

### Key Features

1. **Zero-Copy Views**: Uses `Tensor::slice()` to create views that share storage with the original tensor, avoiding unnecessary data copying.

2. **Negative Dimension Support**: Handles negative dimension indices (e.g., -1 for last dimension).

3. **Uneven Splits**: Properly handles cases where the dimension size doesn't divide evenly by `split_size`. The last chunk will be smaller.

4. **Device Agnostic**: Works with both CPU and GPU tensors.

5. **Error Handling**:
   - Validates split_size is positive
   - Checks dimension is in valid range
   - Provides clear error messages

### Algorithm

```cpp
1. Validate split_size > 0
2. Normalize negative dimension indices
3. Calculate number of splits: ceil(dim_size / split_size)
4. For each split:
   - Calculate start = i * split_size
   - Calculate end = min(start + split_size, dim_size)
   - Create view using slice(dim, start, end)
5. Return vector of tensor views
```

## Testing

### Test Coverage
Created comprehensive test suite in `/home/lee/Projects/Tenzor/tests/test_split_operation.cpp` with 15 test cases:

#### Basic Functionality Tests
- ✅ Split 1D tensor with even division
- ✅ Split 1D tensor with uneven division
- ✅ Split 2D tensor along dimension 0
- ✅ Split 2D tensor along dimension 1
- ✅ Split 3D tensor along multiple dimensions
- ✅ Negative dimension indexing

#### Edge Cases
- ✅ Split with size equal to dimension size
- ✅ Split with size larger than dimension size
- ✅ Split with size = 1 (individual elements)

#### Memory and Performance
- ✅ Verify split creates views (zero-copy)
- ✅ Split on non-contiguous tensors

#### Error Handling
- ✅ Invalid split size (0 or negative)
- ✅ Dimension out of range

#### Real-World Use Cases
- ✅ Multi-head attention pattern
- ✅ Batch processing (mini-batches)

### Test Results
All 15 tests passed successfully:
```
[==========] Running 15 tests from 1 test suite.
[  PASSED  ] 15 tests.
```

## Use Cases

### 1. Multi-Head Attention
Split hidden dimension into multiple attention heads:
```cpp
// Input: [batch_size, seq_len, hidden_dim] = [2, 10, 64]
// Split into 8 heads of size 8
auto heads = split(attention_input, 8, 2);
// Result: 8 tensors each with shape [2, 10, 8]
```

### 2. Batch Processing
Divide large batch into mini-batches:
```cpp
// Input: [100, 32] - 100 samples
// Split into mini-batches of 10
auto mini_batches = split(dataset, 10, 0);
// Result: 10 tensors each with shape [10, 32]
```

### 3. Multi-Path Architectures
Split feature maps for parallel processing paths:
```cpp
// Split feature maps across different processing branches
auto branches = split(features, branch_size, 1);
```

## Performance Characteristics

### Time Complexity
- **O(n)** where n is the number of chunks
- Each chunk creation is O(1) (view creation, no copying)

### Space Complexity
- **O(n)** for storing n tensor view objects
- **O(1)** for actual data (views share storage)

### Memory Efficiency
- ✅ Zero-copy: Views share storage with original tensor
- ✅ Efficient: Only metadata (shape, strides, offset) is copied
- ✅ Fast: No data movement required

## Comparison with Related Operations

| Operation | Purpose | Copies Data | Chunk Sizes |
|-----------|---------|-------------|-------------|
| `split()` | Fixed-size chunks | No (views) | All equal (except last) |
| `chunk()` | Fixed number of chunks | No (views) | May vary |
| `slice()` | Single range | No (view) | N/A |

## Examples

### Basic Usage
```cpp
// Split 1D tensor
auto tensor = arange(0.0f, 12.0f, 1.0f);
auto chunks = split(tensor, 3, 0);
// Result: 4 chunks of size 3: [0,1,2], [3,4,5], [6,7,8], [9,10,11]
```

### Uneven Split
```cpp
// Split when size doesn't divide evenly
auto tensor = arange(0.0f, 10.0f, 1.0f);
auto chunks = split(tensor, 3, 0);
// Result: 4 chunks: [0,1,2], [3,4,5], [6,7,8], [9]
```

### Multi-Dimensional Split
```cpp
// Split 3D tensor along different dimensions
auto tensor = zeros({4, 6, 8});

auto chunks_dim0 = split(tensor, 2, 0);  // 2 chunks of shape [2, 6, 8]
auto chunks_dim1 = split(tensor, 2, 1);  // 3 chunks of shape [4, 2, 8]
auto chunks_dim2 = split(tensor, 5, 2);  // 2 chunks: [4,6,5] and [4,6,3]
```

### Negative Indexing
```cpp
// Use negative indices
auto tensor = ones({4, 6, 8});
auto chunks = split(tensor, 4, -1);  // Split along last dimension
```

## Demonstration Program

A comprehensive demonstration program is available at:
`/home/lee/Projects/Tenzor/examples/demo_split_operation.cpp`

The demo showcases:
1. Basic split operations
2. 2D tensor splitting
3. Multi-head attention use case
4. Batch processing use case
5. Zero-copy views behavior
6. Uneven split handling

To run:
```bash
cd /home/lee/Projects/Tenzor
./bin/demo_split_operation
```

## Integration

### CMake Configuration
Test added to `/home/lee/Projects/Tenzor/tests/CMakeLists.txt`:
```cmake
add_executable(test_split_operation test_split_operation.cpp)
target_link_libraries(test_split_operation PRIVATE tenzor_core GTest::gtest_main)
gtest_discover_tests(test_split_operation DISCOVERY_TIMEOUT 30)
```

### Build Instructions
```bash
cd /home/lee/Projects/Tenzor/build
cmake --build . --target test_split_operation
./bin/test_split_operation
```

## Benefits

1. **Performance**: Zero-copy implementation using views
2. **Flexibility**: Supports any dimension with positive or negative indices
3. **Robustness**: Comprehensive error handling and validation
4. **Usability**: Intuitive API matching PyTorch/NumPy conventions
5. **Versatility**: Works with CPU and GPU tensors
6. **Reliability**: Thoroughly tested with 15 test cases covering edge cases

## Future Enhancements

Potential improvements for future consideration:
1. Add support for splitting by specific indices (similar to `torch.tensor_split`)
2. Optimize for specific tensor layouts (row-major vs column-major)
3. Add GPU kernel specializations for improved performance
4. Support for more advanced splitting patterns (e.g., variable chunk sizes)

## Conclusion

The split operation is now fully implemented, tested, and ready for use in neural network operations, particularly for:
- Multi-head attention mechanisms
- Batch processing pipelines
- Multi-path network architectures
- Any scenario requiring efficient tensor partitioning

The implementation follows best practices:
- ✅ Efficient (zero-copy views)
- ✅ Safe (comprehensive validation)
- ✅ Well-tested (15+ test cases)
- ✅ Well-documented (inline comments and examples)
- ✅ Compatible with existing codebase patterns
