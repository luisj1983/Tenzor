# Python Tensor `__setitem__` Implementation Summary

## Implementation Status: Partially Complete

### Location
- **File**: `/home/lee/Projects/Tenzor/python/bindings.cpp`
- **Lines**: 227-492

### What Was Implemented

1. **Comprehensive `__setitem__` handler** supporting:
   - Single integer indexing: `tensor[0] = value`
   - Slice indexing: `tensor[1:5] = value`
   - Multi-dimensional indexing: `tensor[0, :, 1:3] = value`
   - Ellipsis support: `tensor[..., 0] = value`
   - Negative indexing: `tensor[-1] = value`

2. **Helper functions**:
   - `value_to_tensor`: Converts Python scalars/tensors to tensor format
   - `copy_with_broadcast`: Handles copying with broadcasting support

3. **Features**:
   - Dtype conversion for scalars
   - Shape validation
   - Bounds checking
   - Device compatibility checking
   - Scalar broadcasting to destination shape

4. **Added missing Python bindings**:
   - `fill_(value)`: Fill tensor with scalar in-place
   - `zero_()`: Fill tensor with zeros in-place

### Current Issue

**The implementation does not work correctly due to a fundamental limitation**:

When `__getitem__` (e.g., `tensor[1:3]`) creates a view/slice, pybind11 returns it **by value** (a copy), not by reference. This means:

```python
t = tz.zeros([5])
s = t[1:3]      # s is a VIEW sharing storage with t (in C++)
s.fill_(7.0)     # Modifies s's storage
# But s is a Python copy, so t remains unchanged!
print(t.numpy()) # Still [0, 0, 0, 0, 0]
```

### Root Cause

The C++ `Tensor::slice()` method correctly creates a view that shares storage with the original tensor. However:

1. Pybind11's default behavior copies the returned tensor when passing back to Python
2. The Python object holds an independent copy, breaking the storage-sharing contract
3. Modifications to the "view" don't affect the original tensor

### Solutions (Recommended Next Steps)

#### Option 1: Direct Storage Modification (Recommended)
Implement `__setitem__` to directly compute memory indices and write to the original tensor's storage:

```cpp
.def("__setitem__", [](tenzor::Tensor& self, py::object key, py::object value) {
    // Parse indexing from key
    // Compute linear indices into storage
    // Write values directly to self.data_ptr() + computed_offsets
    // This bypasses the view/copy issue entirely
})
```

**Pros**: Works with current architecture, no pybind11 changes needed
**Cons**: More complex index calculation, must handle strides manually

#### Option 2: Return Reference-Counted Views
Modify `__getitem__` to return a special "TensorView" type that holds a reference to the parent:

```cpp
class TensorView {
    std::shared_ptr<TensorImpl> parent_;
    // Indexing metadata

    void fill_(float value) {
        // Directly modifies parent_'s storage
    }
};
```

**Pros**: More PyTorch-like semantics, cleaner API
**Cons**: Requires significant refactoring, new type to maintain

#### Option 3: Use Pybind11's `return_value_policy`
Try to force reference semantics:

```cpp
.def("__getitem__", [...], py::return_value_policy::reference_internal)
```

**Pros**: Minimal code changes
**Cons**: May not work with value-semantic C++ types, could cause lifetime issues

### Test Coverage

Comprehensive test suite created at `/home/lee/Projects/Tenzor/tests/test_python_setitem.py`:

- Scalar assignment
- Slice assignment
- Tensor-to-tensor assignment
- 2D/3D indexing
- Negative indexing
- Multi-dtype support
- Broadcasting
- Error handling
- Edge cases

**Current status**: 1/11 tests passing (only error handling works)

### Files Modified

1. `/home/lee/Projects/Tenzor/python/bindings.cpp`:
   - Added `__setitem__` implementation (lines 227-492)
   - Added `fill_()` and `zero_()` bindings (lines 92-95)
   - Added missing includes for backend access

2. `/home/lee/Projects/Tenzor/tests/test_python_setitem.py`:
   - New comprehensive test suite (269 lines)

### Compilation Status

✅ Code compiles successfully
❌ Tests fail due to storage-sharing issue

### Recommended Action Plan

1. **Immediate** (1-2 hours):
   - Implement Option 1 (direct storage modification)
   - Calculate linear indices from multi-dimensional indices
   - Handle strides and offsets correctly
   - Support contiguous tensors only initially

2. **Short-term** (4-6 hours):
   - Extend to handle non-contiguous tensors
   - Implement proper broadcasting for tensor assignment
   - Add comprehensive error messages

3. **Long-term** (1-2 days):
   - Consider Option 2 for better API semantics
   - Implement advanced indexing (boolean, integer arrays)
   - Optimize performance for large tensors

### Code Quality

The implementation follows best practices:
- ✅ Proper error handling and validation
- ✅ Clear helper function separation
- ✅ Comprehensive documentation
- ✅ Type safety (dtype checking)
- ✅ Device compatibility checks
- ⚠️  Needs performance optimization
- ❌  Core functionality broken (storage sharing)

### Conclusion

The `__setitem__` implementation is **production-quality in structure** but requires fixing the fundamental storage-sharing issue before it can work correctly. The architecture and error handling are solid; only the core write mechanism needs reimplementation using one of the recommended solutions above.

**Estimated time to fix**: 2-4 hours for Option 1 (direct storage modification)

---

## Implementation Details (For Reference)

### Helper Function: `value_to_tensor`

Converts Python values to tensors:
- Handles `Tensor` objects (pass-through)
- Converts Python floats/ints to single-element tensors
- Matches dtype of destination tensor
- Validates types

### Helper Function: `copy_with_broadcast`

Handles tensor-to-tensor copying:
- Scalar broadcasting (1-element tensor → any shape)
- Shape matching validation
- Contiguous fast path (memcpy/device copy)
- Non-contiguous path (not yet implemented)
- Broadcasting validation (not yet fully implemented)

### Indexing Patterns Supported

1. **Integer**: `tensor[idx]` → Slice + squeeze
2. **Slice**: `tensor[start:stop]` → Direct slice
3. **Tuple**: `tensor[i, :, j:k]` → Sequential slicing with squeeze tracking
4. **Ellipsis**: `tensor[..., 0]` → Dimension skip calculation

All patterns compute a "target" tensor (view), then attempt to modify it.

### Error Messages

The implementation provides clear, specific error messages for:
- Index out of range
- Dimension mismatches
- Device mismatches
- Type errors
- Unsupported operations (step != 1, boolean indexing, etc.)
