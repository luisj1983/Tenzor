# Negative Indexing and dispatchContiguous Bug Analysis

## Issue Summary
The `AllBackends/AdvancedIndexingTest.NegativeIndexing/vulkan` test was reported to hang indefinitely during execution.

## Root Cause Investigation

### Code Flow for Negative Indexing

1. **`t[-1]`** (tensor.cpp:1106-1140)
   - Calls `operator[]` with negative index
   - Normalizes index: `normalized_idx = idx + dim0_size`
   - Calls `slice(0, normalized_idx, normalized_idx + 1, 1)`
   - Calls `squeeze(0)` on the result

2. **`slice()`** (tensor.cpp:1142-1191)
   - Creates a **metadata-only view** that shares storage
   - Updates shape, offset, and strides
   - Returns non-contiguous tensor view

3. **`squeeze()`** (vulkan_backend.cpp:3325-3377)
   - Already implemented as **metadata-only operation**
   - Does not trigger any backend operations
   - Returns a view with updated shape/strides

4. **`.to(Device::cpu())`** (tensor.cpp:324-425)
   - Detects tensor is non-contiguous and on GPU (line 339)
   - **Special handling path** for non-contiguous GPU tensors:
     - Creates CPU result tensor
     - Copies entire GPU buffer to temporary buffer
     - Manually reorders elements using stride information (lines 367-385)
     - Copies reordered data to CPU tensor
   - This should work correctly without recursion

### Critical Bug in dispatchContiguous()

The `dispatchContiguous()` implementation (vulkan_backend.cpp:3406-3424) has a critical bug:

```cpp
auto VulkanBackend::dispatchContiguous(const Tensor& input) -> Tensor {
    if (input.is_contiguous()) {
        return input;
    }

    // Creates output with same shape
    Tensor output(out_shape, input.dtype(), input.device());

    // BUG: Simple memory copy doesn't handle strides!
    size_t bytes = input.numel() * input.dtype_size();
    copy(output.data_ptr(), input.data_ptr(), bytes, CopyKind::DeviceToDevice);

    return output;
}
```

**The Issue:** The function performs a raw memory copy without accounting for non-contiguous strides. This means:
- For a sliced tensor with offset and non-standard strides
- The copy starts from the wrong offset (input.data_ptr() ignores offset)
- Elements are not reordered correctly
- The "contiguous" tensor is still actually non-contiguous

**Affected Operations:**
- `dispatchReshape()` calls `dispatchContiguous()` at line 3183
- Any code that calls `.contiguous()` on Vulkan tensors

## Test Results

After investigation, the tests were run to completion:
- **Total tests:** 715 Vulkan backend tests
- **Passed:** 469 tests (66%)
- **Failed:** 246 tests (34%)
- **Total time:** 1140.67 seconds (~19 minutes)

**The negative indexing test did NOT hang during this run.**

## Possible Explanations for Intermittent Hanging

1. **Race Condition:** The hanging may be timing-dependent
2. **Resource Exhaustion:** Previous test runs may have left GPU resources in bad state
3. **Already Fixed:** The friend class fix or other changes may have resolved it
4. **Test Infrastructure:** The test framework itself may have issues

## Recommended Fix for dispatchContiguous()

### Fix Implementation

The function must properly handle strides when copying non-contiguous tensors. We need to use the CPU as an intermediary since it has proper stride handling:

```cpp
auto VulkanBackend::dispatchContiguous(const Tensor& input) -> Tensor {
    if (input.is_contiguous()) {
        return input;
    }

    // For non-contiguous GPU tensors, use CPU as intermediary
    // The to(Device::cpu()) method has stride-aware copying (lines 339-393)
    Tensor cpu_temp = input.to(Device::cpu());

    // Transfer back to GPU - now contiguous
    return cpu_temp.to(input.device());
}
```

**Why This Works:**
- `input.to(Device::cpu())` uses the special stride-handling path (tensor.cpp:339-393)
- It properly copies and reorders elements according to strides
- `cpu_temp.to(input.device())` transfers the now-contiguous data back to GPU
- The result is truly contiguous

### Alternative: Strided Copy Shader (Better Performance)

For production use, implement a dedicated Vulkan compute shader:

```glsl
// strided_copy.comp
layout(push_constant) uniform PushConstants {
    uint ndims;
    uint offset;
    uint strides[8];  // Max 8 dimensions
    uint shape[8];
};

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= total_elements) return;

    // Convert linear index to multi-dimensional indices
    uint multi_idx[8];
    uint remaining = idx;
    for (int d = ndims - 1; d >= 0; d--) {
        multi_idx[d] = remaining % shape[d];
        remaining /= shape[d];
    }

    // Compute source offset using strides
    uint src_offset = offset;
    for (int d = 0; d < ndims; d++) {
        src_offset += multi_idx[d] * strides[d];
    }

    // Copy element from strided source to contiguous destination
    output_buffer[idx] = input_buffer[src_offset];
}
```

## Backend Agnosticism Maintained

All identified issues are in backend-specific code:
- `src/backends/vulkan/vulkan_backend.cpp` (dispatchContiguous at line 3406)
- No changes needed to frontend tensor API
- Other backends (CPU, CUDA) are unaffected

## Status

- ✅ Identified root cause of dispatchContiguous bug
- ✅ Verified negative indexing no longer hangs in current test run
- ⚠️ Bug still exists in dispatchContiguous but may not always trigger
- 📋 Fix documented for implementation
- 📊 Current pass rate: 469/715 (66%)

## Next Steps

1. Implement proper strided copy in dispatchContiguous (CPU intermediary approach)
2. Re-run negative indexing test multiple times to confirm stability
3. Fix remaining 246 test failures
4. Consider implementing dedicated strided copy shader for better performance
