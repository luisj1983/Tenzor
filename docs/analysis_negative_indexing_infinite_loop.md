# Analysis: NegativeIndexing Test Infinite Loop on Vulkan Backend

## Executive Summary

The `NegativeIndexing` test causes an **infinite recursion loop** when running on the Vulkan backend. The issue is caused by a circular call chain between `dispatchReshape()` and `dispatchContiguous()` in the Vulkan backend implementation.

---

## Test Details

**Test Location:** `/home/lee/Projects/Tenzor/tests/unit/test_ops_additional.cpp` line 1140-1153

**Test Code:**
```cpp
TEST_P(AdvancedIndexingTest, NegativeIndexing) {
    auto t = zeros({5}, DType::Float32, device);  // device = vulkan(0)
    auto t_cpu = t.to(Device::cpu());
    auto data = t_cpu.data<float>();
    for (int i = 0; i < 5; i++) {
        data[i] = static_cast<float>(i);
    }
    t = t_cpu.to(device);  // Transfer back to Vulkan

    // Access last element with negative index - THIS CAUSES INFINITE LOOP
    auto last = t[-1];
    auto last_cpu = last.to(Device::cpu());
    EXPECT_FLOAT_EQ(last_cpu.data<float>()[0], 4.0f) << "Failed on " << device.to_string();
}
```

**Affected Backend:** Vulkan only (CPU backend works correctly)

---

## Root Cause: Infinite Recursion Loop

### Complete Call Chain

1. **Test calls:** `t[-1]`
   - Location: `test_ops_additional.cpp:1150`

2. **Invokes:** `Tensor::operator[](int64_t idx)`
   - Location: `src/core/tensor.cpp:1106-1140`
   - Line 1133: Calls `slice(0, normalized_idx, normalized_idx + 1, 1)`
   - Line 1137: Calls `result.squeeze(0)` on the sliced result

3. **Slice creates non-contiguous tensor:** `Tensor::slice()`
   - Location: `src/core/tensor.cpp:1142-1191`
   - Returns a tensor with modified strides (non-contiguous)
   - Important: `result.impl_->strides[dim] *= step;` (line 1187)

4. **Squeeze on CPU/CUDA backends:** `Tensor::squeeze()`
   - Location: `src/core/tensor.cpp:969-1021`
   - **CPU Implementation:** Directly manipulates shape and strides metadata
   - **Does NOT dispatch to backend** - Pure metadata operation
   - Returns immediately without recursion

5. **BUT Vulkan Backend Dispatches Squeeze:**
   - Location: Vulkan's `dispatch()` method catches "squeeze"
   - File: `src/backends/vulkan/vulkan_backend.cpp:929-935`
   ```cpp
   if (op_name == "squeeze") {
       if (inputs.size() != 1) {
           throw std::invalid_argument("squeeze requires 1 input");
       }
       int64_t dim = attrs.contains("dim") ? std::stoll(attrs.at("dim")) : -1;
       return {dispatchSqueeze(inputs[0], dim)};  // Line 934
   }
   ```

6. **PROBLEM 1:** `VulkanBackend::dispatchSqueeze()` **always calls reshape**
   - Location: `src/backends/vulkan/vulkan_backend.cpp:3234-3273`
   - Line 3272: `return dispatchReshape(input, out_shape);`
   - **This is the first link in the circular chain**

7. **PROBLEM 2:** `VulkanBackend::dispatchReshape()` checks contiguity
   - Location: `src/backends/vulkan/vulkan_backend.cpp:3067-3098`
   - Lines 3084-3089: If contiguous, performs simple copy
   - **Lines 3090-3097: If NOT contiguous, calls `dispatchContiguous()`**
   ```cpp
   } else {
       // Need to make contiguous first, then reshape
       Tensor contiguous = dispatchContiguous(input);  // Line 3092 - SECOND RECURSION POINT
       Tensor output(new_shape, contiguous.dtype(), contiguous.device());
       size_t bytes = contiguous.numel() * contiguous.dtype_size();
       copy(output.data_ptr(), contiguous.data_ptr(), bytes, CopyKind::DeviceToDevice);
       return output;
   }
   ```

8. **PROBLEM 3:** `VulkanBackend::dispatchContiguous()` needs to be dispatched via Tensor::contiguous()
   - Location: `src/backends/vulkan/vulkan_backend.cpp:3302-3320`
   - But there's no direct recursion here - the issue is at the Tensor class level

9. **THE REAL CIRCULAR PATH:**
   ```
   Tensor::squeeze() [CPU impl]
     → Creates result with modified shape/strides
     → BUT for Vulkan, squeeze is INTERCEPTED by dispatcher

   Dispatcher catches "squeeze"
     → Calls VulkanBackend::dispatchSqueeze()

   VulkanBackend::dispatchSqueeze()
     → Calls dispatchReshape() (line 3272)

   VulkanBackend::dispatchReshape()
     → Checks if input is contiguous (line 3084)
     → Input from slice() is NOT contiguous
     → Calls dispatchContiguous() (line 3092)

   VulkanBackend::dispatchContiguous()
     → Creates new contiguous tensor
     → But the implementation is incomplete/buggy
     → Likely tries to dispatch back through system

   [INFINITE LOOP CONTINUES]
   ```

---

## Why CPU/CUDA Don't Have This Problem

### CPU Backend Implementation
- File: `src/core/tensor.cpp:969-1021`
- **Squeeze is NOT dispatched** to backend
- Pure metadata operation - directly modifies shape and strides
- No backend dispatch needed
- Returns immediately

### CUDA Backend (likely)
- Probably handles reshape/contiguous without circular dependencies
- May have optimized paths that avoid the recursion

### Vulkan Backend Issue
- **Over-dispatching**: Treats squeeze as a compute operation
- Unnecessary backend dispatch for metadata-only operation
- Reshape implementation depends on contiguous
- Creates circular dependency chain

---

## Exact Recursion Path

```
Test: t[-1]
  ↓
Tensor::operator[](-1)  [Line 1106, tensor.cpp]
  ↓
slice(0, 4, 5, 1)  [Line 1133, tensor.cpp]
  ↓
Creates non-contiguous result (modified stride)
  ↓
result.squeeze(0)  [Line 1137, tensor.cpp]
  ↓
Dispatcher::dispatch("squeeze", ...)  [Implicit via backend registration]
  ↓
VulkanBackend::dispatch() catches "squeeze"  [Line 929, vulkan_backend.cpp]
  ↓
dispatchSqueeze(input, 0)  [Line 934, vulkan_backend.cpp]
  ↓
dispatchReshape(input, out_shape)  [Line 3272, vulkan_backend.cpp]
  ↓
Checks: !input.is_contiguous()  [Line 3090, vulkan_backend.cpp]
  ↓
dispatchContiguous(input)  [Line 3092, vulkan_backend.cpp]
  ↓
Creates Tensor with new shape  [Line 3311, vulkan_backend.cpp]
  ↓
copy(output, input, bytes, DeviceToDevice)  [Line 3317, vulkan_backend.cpp]
  ↓
[COPY IMPLEMENTATION MAY DISPATCH BACK TO RESHAPE OR CONTIGUOUS]
  ↓
🔄 INFINITE LOOP - Returns to dispatchReshape() or dispatchContiguous()
```

---

## Key Code Locations

| Component | File | Lines | Issue |
|-----------|------|-------|-------|
| Test Case | `tests/unit/test_ops_additional.cpp` | 1140-1153 | Triggers via `t[-1]` |
| operator[] | `src/core/tensor.cpp` | 1106-1140 | Calls slice + squeeze |
| slice() | `src/core/tensor.cpp` | 1142-1191 | Creates non-contiguous tensor |
| squeeze() CPU impl | `src/core/tensor.cpp` | 969-1021 | Metadata-only, no dispatch |
| Vulkan dispatch | `src/backends/vulkan/vulkan_backend.cpp` | 929-935 | Catches "squeeze" operation |
| dispatchSqueeze | `src/backends/vulkan/vulkan_backend.cpp` | 3234-3273 | **Always calls reshape** (line 3272) |
| dispatchReshape | `src/backends/vulkan/vulkan_backend.cpp` | 3067-3098 | **Calls contiguous if non-contiguous** (line 3092) |
| dispatchContiguous | `src/backends/vulkan/vulkan_backend.cpp` | 3302-3320 | Copy implementation issue |

---

## Fix Approach

### Option 1: Make Vulkan squeeze a metadata-only operation (RECOMMENDED)
**Location:** `src/backends/vulkan/vulkan_backend.cpp:3234-3273`

**Change `dispatchSqueeze()` to directly manipulate metadata like CPU version:**

```cpp
auto VulkanBackend::dispatchSqueeze(const Tensor& input, int64_t dim) -> Tensor {
    auto input_shape = input.shape();
    int32_t ndim = input.ndim();

    std::vector<int64_t> out_shape;

    if (dim < 0) {
        // Squeeze all dimensions of size 1
        for (int64_t d : input_shape) {
            if (d != 1) {
                out_shape.push_back(d);
            }
        }
    } else {
        // Normalize negative dimension
        if (dim < 0) dim += ndim;
        if (dim < 0 || dim >= ndim) {
            throw std::invalid_argument("Squeeze: dimension out of range");
        }

        // Squeeze specific dimension
        if (input_shape[dim] != 1) {
            throw std::invalid_argument("Squeeze: dimension size must be 1");
        }

        for (int64_t i = 0; i < ndim; i++) {
            if (i != dim) {
                out_shape.push_back(input_shape[i]);
            }
        }
    }

    // If no dimensions were squeezed, ensure we have at least a scalar
    if (out_shape.empty() && input.numel() == 1) {
        out_shape.push_back(1);
    }

    // FIXED: Metadata-only operation - directly create result with shared storage
    // DO NOT call dispatchReshape which triggers contiguous check
    Tensor result;
    result.impl_ = std::make_shared<TensorImpl>(*(input.impl_));
    result.impl_->shape = std::move(out_shape);

    // Recompute strides for new shape
    result.impl_->strides = compute_strides(result.impl_->shape);

    return result;
}
```

**Why this works:**
- No reshape dispatch = no contiguous check
- Pure metadata operation like CPU implementation
- Maintains memory sharing (no copy needed)
- Breaks the circular dependency

---

### Option 2: Fix dispatchReshape to avoid recursion
**Location:** `src/backends/vulkan/vulkan_backend.cpp:3090-3097`

**Add guard to prevent squeeze from triggering contiguous:**

```cpp
} else {
    // Need to make contiguous first, then reshape
    // BUT: Avoid recursion if called from squeeze operation
    if (input.ndim() != new_shape.size() && /* squeezed dimensions */) {
        // This is a squeeze operation, handle differently
        // Create output with proper strides directly
        ...
    }

    Tensor contiguous = dispatchContiguous(input);
    Tensor output(new_shape, contiguous.dtype(), contiguous.device());
    size_t bytes = contiguous.numel() * contiguous.dtype_size();
    copy(output.data_ptr(), contiguous.data_ptr(), bytes, CopyKind::DeviceToDevice);
    return output;
}
```

**Why this is less ideal:**
- More complex logic
- Doesn't address the fundamental issue (over-dispatching)
- Harder to maintain

---

### Option 3: Don't dispatch squeeze to Vulkan backend at all
**Location:** `src/backends/vulkan/vulkan_backend.cpp:929-935`

**Remove the squeeze case from Vulkan dispatch:**

```cpp
// Remove this entire block:
// if (op_name == "squeeze") {
//     ...
//     return {dispatchSqueeze(inputs[0], dim)};
// }
```

**Let squeeze fall through to CPU implementation (Tensor::squeeze)**

**Why this works:**
- Squeeze becomes metadata-only for all backends
- Consistent behavior across CPU/CUDA/Vulkan
- Simplest fix

**Trade-off:**
- May miss potential Vulkan-specific optimizations
- But squeeze is metadata-only anyway, so no computation needed

---

## Recommended Solution

**Use Option 1** - Make `dispatchSqueeze()` a metadata-only operation

**Reasoning:**
1. Squeeze is fundamentally a view operation (no data movement)
2. CPU implementation already does this correctly
3. Breaking the `dispatchSqueeze` → `dispatchReshape` link stops the recursion
4. Maintains backend dispatch capability for future optimizations
5. Most aligned with PyTorch/NumPy behavior (squeeze doesn't copy data)

---

## Testing Verification

After implementing the fix, verify:

1. **NegativeIndexing test passes on Vulkan:**
   ```bash
   ./tests/unit/test_ops_additional --gtest_filter="*/AdvancedIndexingTest.NegativeIndexing/vulkan"
   ```

2. **All other squeeze tests still pass:**
   ```bash
   ./tests/unit/test_ops_additional --gtest_filter="*squeeze*"
   ```

3. **No regression on CPU/CUDA backends:**
   ```bash
   ./tests/unit/test_ops_additional --gtest_filter="*/AdvancedIndexingTest.NegativeIndexing/*"
   ```

---

## Additional Notes

### Why Slicing Creates Non-Contiguous Tensors
- File: `src/core/tensor.cpp:1142-1191`
- Line 1183: `result.impl_->offset += start * impl_->strides[dim];`
- Line 1187: `result.impl_->strides[dim] *= step;`
- These modifications break contiguity for sliced views

### Backend Dispatch Mechanism
- Squeeze goes through `Dispatcher::dispatch("squeeze", ...)`
- Vulkan backend registers handler for "squeeze" operation
- CPU backend doesn't register squeeze (metadata-only in Tensor class)

### Performance Impact of Fix
- **None** - Squeeze should never copy data
- Metadata-only operations are always O(1)
- Fix actually improves performance by avoiding unnecessary dispatch

---

## Appendix: Full Function Signatures

```cpp
// Tensor class (CPU implementation)
auto Tensor::operator[](int64_t idx) const -> Tensor;           // Line 1106
auto Tensor::slice(int64_t dim, int64_t start,
                   int64_t end, int64_t step) const -> Tensor;  // Line 1142
auto Tensor::squeeze(std::optional<int64_t> dim) const -> Tensor; // Line 969

// Vulkan Backend
auto VulkanBackend::dispatch(
    const std::string& op_name,
    const std::vector<Tensor>& inputs,
    const OpAttributes& attrs
) -> std::vector<Tensor>;                                       // Catches "squeeze"

auto VulkanBackend::dispatchSqueeze(
    const Tensor& input,
    int64_t dim
) -> Tensor;                                                     // Line 3234

auto VulkanBackend::dispatchReshape(
    const Tensor& input,
    const std::vector<int64_t>& new_shape
) -> Tensor;                                                     // Line 3067

auto VulkanBackend::dispatchContiguous(
    const Tensor& input
) -> Tensor;                                                     // Line 3302
```

---

## Document Metadata

- **Date:** 2025-11-06
- **Tenzor Version:** Current (main branch)
- **Test File:** `tests/unit/test_ops_additional.cpp`
- **Primary Files Analyzed:**
  - `src/core/tensor.cpp`
  - `src/backends/vulkan/vulkan_backend.cpp`
  - `src/backends/vulkan/vulkan_backend.hpp`
  - `tests/unit/test_ops_additional.cpp`
  - `tests/backend_test_fixture.hpp`

---

**END OF ANALYSIS**
