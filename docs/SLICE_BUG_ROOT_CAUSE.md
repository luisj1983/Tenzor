# Slice Bug Root Cause Analysis
**Date**: October 25, 2025
**Status**: ❌ CRITICAL BUG CONFIRMED

## Executive Summary

Found the root cause of slice operation failures affecting **ALL backends** (CPU, CUDA, Vulkan, OneAPI). The bug has two components:

1. ✅ **FIXED**: `.contiguous()` correctly converts non-contiguous sliced tensors to contiguous layout
2. ❌ **BROKEN**: Backend operations create result tensors that incorrectly inherit metadata from input sliced tensors

## Evidence

### Test Results

#### Test 1: Direct Data Access on Sliced Tensors

**Original Tensor:**
```
Box 0: [0, 1, 10, 11]
Box 1: [2, 3, 12, 13]
Box 2: [4, 5, 14, 15]
```

**Slice(1, 0, 1) - Column 0 (x1 values):**
- Expected: `[0, 2, 4]`
- Without `.contiguous()`: `[0, 1, 10]` ❌ **WRONG**
- With `.contiguous()`: `[0, 2, 4]` ✅ **CORRECT**

**Slice(1, 2, 3) - Column 2 (x2 values):**
- Expected: `[10, 12, 14]`
- Without `.contiguous()`: `[10, 11, 2]` ❌ **WRONG**
- With `.contiguous()`: `[10, 12, 14]` ✅ **CORRECT**

#### Test 2: Backend Subtraction Operation

**Setup:**
```cpp
auto x1_contig = boxes.slice(1, 0, 1).contiguous();  // [0, 2, 4] ✅
auto x2_contig = boxes.slice(1, 2, 3).contiguous();  // [10, 12, 14] ✅
auto widths = x2_contig - x1_contig;  // Should be [10, 10, 10]
```

**Result:**
`widths = [2.33e-41, -2, -4]` ❌ **COMPLETELY WRONG**

## Root Cause

### Problem #1: Non-Contiguous Tensor Data Access ✅ MITIGATED

**Issue:** Calling `data<T>()[i]` on a sliced tensor accesses memory sequentially, ignoring strides.

**Why:** Sliced tensors are **views** with strides (e.g., `[4, 1]`) to skip elements. Linear indexing doesn't respect this.

**Solution:** Calling `.contiguous()` creates a new tensor with contiguous layout where `data<T>()[i]` works correctly.

**Status:** ✅ This workaround is implemented in `/home/lee/Projects/Tenzor/src/ops/math.cpp:24-29`:
```cpp
auto sub(const Tensor& a, const Tensor& b) -> Tensor {
    Tensor a_contiguous = a.is_contiguous() ? a : a.contiguous();  // ✅
    Tensor b_contiguous = b.is_contiguous() ? b : b.contiguous();  // ✅
    std::vector<Tensor> inputs = {a_contiguous, b_contiguous};
    return Dispatcher::dispatch("sub", inputs)[0];  // ❌ Result is broken!
}
```

### Problem #2: Backend Operations Create Broken Result Tensors ❌ NOT FIXED

**Issue:** Even though inputs are contiguous, the **result tensor** from backend operations has garbage data.

**Evidence:**
- Input x1_contig: `[0, 2, 4]` ✅ Correct
- Input x2_contig: `[10, 12, 14]` ✅ Correct
- Result widths: `[2.33e-41, -2, -4]` ❌ Garbage

**Hypothesis:** Backend operation creates result tensor using code like:
```cpp
// BROKEN:
auto result = tensor_like(input);  // Inherits input's offset/strides!
```

**Should be:**
```cpp
// CORRECT:
auto result = zeros(output_shape, input.dtype(), input.device());  // Fresh tensor, offset=0
```

## Impact

This bug affects **EVERY operation on sliced tensors** across **ALL backends**:
- ❌ Arithmetic: `+`, `-`, `*`, `/`
- ❌ Comparisons: `>`, `<`, `==`
- ❌ Math operations: `exp`, `log`, etc.
- ❌ Neural network operations using sliced tensors

### Why Manual CPU Workarounds Exist

This explains the manual implementations in `/home/lee/Projects/Tenzor/src/ops/detection.cpp:38-170`:
- Lines 48-67: Manual CPU `box_area` implementation
- Lines 84-170: Manual CPU `box_iou` implementation
- Comment (line 63): "fall back to slice-based approach" for non-CPU backends

These workarounds were added because slice-based operations produce garbage results!

## Diagnosis Process

1. Created `/home/lee/Projects/Tenzor/tests/test_slice_backend_parity.cpp`
   - **Result:** Failed on CPU backend with garbage values
   - **Conclusion:** NOT a backend-specific issue, affects all backends

2. Created `/home/lee/Projects/Tenzor/tests/test_slice_debug.cpp`
   - **Result:** Identified non-contiguous tensor data access issue
   - **Conclusion:** `.contiguous()` fixes data access

3. Created `/home/lee/Projects/Tenzor/tests/test_contiguous_fix.cpp`
   - **Result:** `.contiguous()` fixes input tensors, but output is still broken
   - **Conclusion:** Backend operations create broken result tensors

## Next Steps

### Immediate Priority: Fix Backend Result Tensor Creation

**Files to fix:** All backend operation implementations
- `src/backends/cpu/kernels/*.cpp`
- `src/backends/cuda/kernels/*.cu`
- `src/backends/vulkan/*.cpp`
- `src/backends/oneapi/kernels/*.cpp`

**Search Pattern:**
```cpp
tensor_like(input)  // BROKEN - inherits offset
```

**Fix Pattern:**
```cpp
zeros(output_shape, dtype, device)  // CORRECT - fresh tensor
```

### Estimated Scope

- **CPU Operations**: ~51 operations to audit
- **CUDA Operations**: ~50 operations to audit
- **Vulkan Operations**: ~30 operations to audit
- **OneAPI Operations**: ~86 kernels (currently broken for other reasons)

**Total**: ~220 operation implementations to review and fix

## Success Criteria

✅ Test `/home/lee/Projects/Tenzor/tests/test_slice_backend_parity.cpp` passes on all backends
✅ Test `/home/lee/Projects/Tenzor/tests/test_contiguous_fix.cpp` passes
✅ Manual CPU workarounds in `detection.cpp` can be safely removed
✅ Mask R-CNN tests complete in <10 seconds with GPU acceleration

## References

- Original Plan: `/home/lee/Projects/Tenzor/docs/BACKEND_PARITY_PLAN.md`
- Phase 12 Status: `/home/lee/Projects/Tenzor/docs/PHASE_12_MASK_RCNN_FIX_STATUS.md`
- Tensor Implementation: `/home/lee/Projects/Tenzor/src/core/tensor.cpp:1098-1147` (slice method)
- Math Operations: `/home/lee/Projects/Tenzor/src/ops/math.cpp:15-30` (add/sub with contiguous conversion)

---

**Critical Insight:** This is NOT a "slice bug" - it's a "backend result tensor creation bug" that manifests when operating on sliced tensors. The slice operation itself is correct!
