# OneAPI Im2col/Col2im Implementation

## Overview
This document describes the implementation of im2col and col2im operations for the OneAPI backend in the Tenzor deep learning framework.

## Files Modified

### 1. New File: `src/backends/oneapi/kernels/im2col.cpp`
**Purpose**: Standalone im2col and col2im operations extracted from conv2d.cpp

**Key Functions**:
- `im2col_kernel(const Tensor& input, const OpAttributes& attrs, sycl::queue& queue) -> Tensor`
  - Converts 4D input tensor (batch, channels, height, width) into column matrix format
  - Output shape: [batch, channels * kernel_h * kernel_w, output_h * output_w]
  - Supports Float32 and Float64 dtypes

- `col2im_kernel(const Tensor& input, const OpAttributes& attrs, sycl::queue& queue) -> Tensor`
  - Inverse operation: converts column matrix back to image format
  - Input shape: [batch, channels * kernel_h * kernel_w, output_h * output_w]
  - Output shape: [batch, channels, height, width]
  - Supports Float32 and Float64 dtypes

**SYCL Kernel Classes**:
- `Im2colKernel`: Parallel kernel functor for im2col transformation
- `Col2imKernel`: Parallel kernel functor for col2im transformation with atomic operations

### 2. Modified: `src/backends/oneapi/oneapi_backend.cpp`
**Changes**:
- Added forward declarations for `im2col_kernel` and `col2im_kernel` (lines 109-111)
- Added dispatch cases for "im2col" and "col2im" operations (lines 720-728)

**Dispatch Logic**:
```cpp
else if (op_name == "im2col") {
    if (inputs.size() != 1) throw std::invalid_argument("im2col requires 1 input");
    return {oneapi::im2col_kernel(inputs[0], attrs, queue)};
}
else if (op_name == "col2im") {
    if (inputs.size() != 1) throw std::invalid_argument("col2im requires 1 input");
    return {oneapi::col2im_kernel(inputs[0], attrs, queue)};
}
```

### 3. Modified: `src/backends/oneapi/CMakeLists.txt`
**Changes**:
- Added `kernels/im2col.cpp` to `ONEAPI_BACKEND_SOURCES` (line 36)
- Added `kernels/im2col.cpp` to `ONEAPI_SYCL_SOURCES` (line 53)

## Operation Attributes

### Im2col Attributes
Required:
- `kernel_size`: int64_t - Size of the convolution kernel (assumed square)

Optional:
- `stride`: int64_t - Stride of the convolution (default: 1)
- `padding`: int64_t - Padding size (default: 0)
- `dilation`: int64_t - Dilation factor (default: 1)

### Col2im Attributes
Required:
- `kernel_size`: int64_t - Size of the convolution kernel (assumed square)
- `output_height`: int64_t - Output tensor height
- `output_width`: int64_t - Output tensor width

Optional:
- `stride`: int64_t - Stride of the convolution (default: 1)
- `padding`: int64_t - Padding size (default: 0)
- `dilation`: int64_t - Dilation factor (default: 1)

## Implementation Details

### Im2col Algorithm
1. **Input Validation**: Ensures input is 4D tensor
2. **Dimension Calculation**: Computes output spatial dimensions based on kernel parameters
3. **Memory Allocation**: Creates output tensor with appropriate shape
4. **Parallel Execution**: Uses SYCL `parallel_for` to transform each position
5. **Batch Processing**: Processes each batch element sequentially

**Kernel Logic** (per thread):
- Computes output position (h_out, w_out)
- Computes kernel position (kh, kw)
- Computes input channel (c)
- Maps to input position with stride, padding, and dilation
- Handles boundary conditions (zero padding)

### Col2im Algorithm
1. **Input Validation**: Ensures input is 3D tensor with correct dimensions
2. **Channel Inference**: Calculates number of channels from input shape
3. **Memory Initialization**: Initializes output to zero
4. **Parallel Accumulation**: Uses SYCL `parallel_for` with atomic operations
5. **Batch Processing**: Processes each batch element sequentially

**Kernel Logic** (per thread):
- Computes output position (h_out, w_out)
- Computes kernel position (kh, kw)
- Computes output channel (c)
- Maps to output position with stride, padding, and dilation
- Uses atomic operations to safely accumulate overlapping contributions

### Thread Safety
- **Im2col**: Each output element written by single thread (no conflicts)
- **Col2im**: Multiple threads may write to same output element
  - Uses `sycl::atomic_ref` with relaxed memory order
  - Ensures correct gradient accumulation in backward pass

## Supported Data Types
- `DType::Float32` (float)
- `DType::Float64` (double)

## Error Handling
Both operations include comprehensive error checking:
- Input tensor dimensionality validation
- Required attribute presence validation
- Shape compatibility validation
- Data type support validation

## Usage Example

```cpp
// Im2col operation
OpAttributes im2col_attrs;
im2col_attrs["kernel_size"] = "3";
im2col_attrs["stride"] = "1";
im2col_attrs["padding"] = "1";
im2col_attrs["dilation"] = "1";

auto col_matrix = backend->dispatch("im2col", {input_tensor}, im2col_attrs);

// Col2im operation
OpAttributes col2im_attrs;
col2im_attrs["kernel_size"] = "3";
col2im_attrs["stride"] = "1";
col2im_attrs["padding"] = "1";
col2im_attrs["dilation"] = "1";
col2im_attrs["output_height"] = "28";
col2im_attrs["output_width"] = "28";

auto image_tensor = backend->dispatch("col2im", {col_matrix}, col2im_attrs);
```

## Relationship to Conv2d
The original im2col and col2im helper functions in `conv2d.cpp` (lines 294-340) remain intact for use by the conv2d operation. The new standalone operations:
- Use the same underlying transformation logic
- Provide explicit control over im2col/col2im transformations
- Enable custom convolution implementations
- Support debugging and testing of convolution operations

## Performance Considerations
1. **Memory Access Pattern**: Column-major layout for cache efficiency
2. **SYCL Work Distribution**: Each thread handles one output element
3. **Atomic Operations**: Col2im uses atomic adds for thread-safe accumulation
4. **Batch Processing**: Sequential batch processing maintains simplicity while allowing SIMD within batches

## Testing Recommendations
1. **Correctness Tests**:
   - Verify im2col output dimensions
   - Test col2im as inverse of im2col
   - Validate with known convolution results

2. **Edge Cases**:
   - Various padding/stride/dilation combinations
   - Different batch sizes
   - Small and large kernel sizes

3. **Data Type Tests**:
   - Float32 and Float64 operations
   - Numerical precision validation

4. **Error Cases**:
   - Invalid input dimensions
   - Missing required attributes
   - Unsupported data types

## Build Integration
The implementation integrates seamlessly with the existing OneAPI backend build system:
1. SYCL kernels compiled with `icpx` targeting spir64
2. Static library created from compiled SYCL objects
3. Linked into shared backend library
4. Loaded dynamically at runtime

## Future Enhancements
Potential improvements for future iterations:
1. Support for non-square kernels (separate kernel_h and kernel_w)
2. Batched SYCL execution for batch dimension
3. Local memory optimization for improved cache utilization
4. Support for grouped convolutions
5. Integration with oneMKL for optimized GEMM operations
