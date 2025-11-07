# Int32 Broadcast Bug Analysis

## Summary
Int32 broadcasting operations return all zeros instead of correct values. Float32 broadcasting works correctly with identical test parameters.

## Test Case
**AddBroadcast_Int32** (test_broadcasting.cpp:195-230):
- Input a: shape (2,3), Int32, values [[1,2,3],[4,5,6]]
- Input b: shape (1,3), Int32, values [[10,20,30]]
- Expected output: [[11,22,33],[14,25,36]]
- Actual output: [[0,0,0],[0,0,0]]

**AddBroadcast_RowToMatrix** (test_broadcasting.cpp:41-78):
- IDENTICAL test with Float32 instead of Int32
- **PASSES** ✓

## What Works
1. **FullInt32 test PASSES**: `full({3}, 42, DType::Int32, vulkan)` creates correct Int32 data on Vulkan
2. **Float32 broadcasting PASSES**: Same shapes, same operations, only dtype differs
3. **All other Float32 broadcast tests PASS**: ScalarToVector, ColumnToMatrix, DifferentDimensions

## What We've Verified

### 1. Shader Implementation (math_broadcast.comp)
- ✓ Has Int32 branch (lines 68-83)
- ✓ Reads with `int val_a = int(a[idx_a])`
- ✓ Performs correct Int32 operations
- ✓ Writes with `result[out_idx] = uint(res)`
- ✓ Shader recompiled and re-embedded

### 2. C++ Backend (vulkan_backend.cpp)
- ✓ dtype_code correctly set to 1 for Int32 (line 1575)
- ✓ Buffer sizes use `dtype_size()` correctly (line 1603-1605)
- ✓ Strides computed correctly with broadcasting (line 1551-1553)
- ✓ Push constants structure matches shader layout
- ✓ Output tensor created with correct dtype (line 1549)

### 3. Data Transfer
- ✓ `to()` method uses `dtype_size()` correctly (line 423, 457-458)
- ✓ `copy()` is dtype-agnostic, uses memcpy (line 316)
- ✓ `allocateDeviceMemory()` is dtype-agnostic (line 266-283)
- ✓ Synchronization present in transfer paths

### 4. Tensor Creation
- ✓ `ones()` handles Int32 correctly (line 5999-6001)
- ✓ Tensor constructor allocates correct size

## Code Paths

### Fast Path (Float32 Only)
```
Line 1487: if (same_shape && is_float32 && both_contiguous)
    → Uses "math" shader (Float32 only)
    → Int32 NEVER uses this path
```

### Broadcast Path (Int32 Uses This)
```
Line 1546: else → Uses "math_broadcast" shader
    → Supports both Float32 and Int32
    → dtype parameter controls which branch
```

## Theories Explored

### ❌ Shader Not Compiled
- Recompiled shader, re-embedded, still fails

### ❌ Stride Calculation Wrong
- Same calculation works for Float32
- `compute_broadcast_strides()` checks `shape[i] == 1` correctly

### ❌ Data Transfer Broken
- FullInt32 proves Vulkan can create/read Int32
- to() method handles all dtypes the same way

### ❌ Buffer Allocation Wrong
- Uses `dtype_size()` everywhere
- Same allocation code works for Float32

### ❌ Push Constants Mismatch
- C++ and shader structures match exactly
- dtype field in correct position

## Remaining Possibilities

1. **Shader execution issue**: Workgroups calculated wrong? Pipeline not binding correctly?
2. **Buffer binding issue**: Descriptor sets not binding Int32 buffers correctly?
3. **Synchronization bug**: GPU operations completing out of order?
4. **Buffer offset bug**: Something about how buffer offsets are calculated for Int32?
5. **GLSL int casting bug**: `int(uint_value)` not working as expected in GLSL?

## Next Steps

1. Add debug output to verify data reaches GPU correctly
2. Check if workgroup dispatch is correct for Int32 case
3. Verify descriptor sets are binding correct buffers
4. Compare Float32 vs Int32 execution paths more carefully
5. Check if there's a Vulkan validation layer warning

## File Locations

- Test: `tests/unit/test_broadcasting.cpp:195-230`
- Backend: `src/backends/vulkan/vulkan_backend.cpp:1454-1640`
- Shader: `src/backends/vulkan/kernels/math_broadcast.comp`
- Embedded: `src/backends/vulkan/embedded_shaders.cpp`
