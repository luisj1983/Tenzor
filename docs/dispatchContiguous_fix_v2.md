# dispatchContiguous() Fix - Version 2

## Problem with First Fix

The initial fix attempted to use `.to()` method to transfer via CPU:

```cpp
Tensor cpu_temp = input.to(Device::cpu());
return cpu_temp.to(input.device());
```

**Issue:** This created infinite recursion because:
1. `cpu_temp.to(input.device())` goes through the normal `.to()` path (line 413 in tensor.cpp)
2. That path calls `.contiguous()` if the tensor is not contiguous
3. `.contiguous()` dispatches to `dispatchContiguous()`
4. Creates infinite loop

## Final Fix - Direct Stride-Aware Copy

The corrected implementation manually handles stride reordering without calling `.to()`:

```cpp
auto VulkanBackend::dispatchContiguous(const Tensor& input) -> Tensor {
    if (input.is_contiguous()) {
        return input;
    }

    // Create output tensor with contiguous layout
    Tensor output(out_shape, input.dtype(), input.device());

    // 1. Download entire GPU buffer to CPU (includes non-contiguous data)
    const size_t total_bytes = input.impl_->storage->size_bytes();
    std::vector<uint8_t> gpu_buffer(total_bytes);
    copy(gpu_buffer.data(), input.impl_->storage->data(),
         total_bytes, CopyKind::DeviceToHost);

    // 2. Rearrange into contiguous layout using stride information
    const int64_t ndims = input.ndim();
    const size_t element_size = input.dtype_size();
    std::vector<int64_t> indices(ndims, 0);
    std::vector<uint8_t> contiguous_buffer(input.numel() * element_size);

    for (int64_t i = 0; i < input.numel(); ++i) {
        // Calculate source offset using strides and offset
        int64_t src_offset = input.impl_->offset;
        for (int64_t dim = 0; dim < ndims; ++dim) {
            src_offset += indices[dim] * input.impl_->strides[dim];
        }

        // Copy element from strided source to contiguous destination
        std::memcpy(contiguous_buffer.data() + dst_offset * element_size,
                   gpu_buffer.data() + src_offset * element_size,
                   element_size);

        // Increment multi-dimensional indices
        for (int64_t dim = ndims - 1; dim >= 0; --dim) {
            if (++indices[dim] < input.impl_->shape[dim]) {
                break;
            }
            indices[dim] = 0;
        }
    }

    // 3. Upload contiguous data back to GPU
    copy(output.data_ptr(), contiguous_buffer.data(),
         contiguous_buffer.size(), CopyKind::HostToDevice);

    return output;
}
```

## How It Works

1. **Download Full Buffer:** Copies entire GPU storage to CPU, including padding
2. **Reorder Elements:** Uses stride and offset information to correctly locate each element
3. **Upload Contiguous:** Transfers the reordered, contiguous data back to GPU

## Key Differences from First Fix

- **No `.to()` calls:** Avoids triggering any dispatches that might call `.contiguous()`
- **Direct memory operations:** Uses only `copy()` with explicit `CopyKind`
- **Manual stride calculation:** Implements the same logic as tensor.cpp lines 367-385
- **No recursion risk:** Pure implementation with no dispatch calls

## Performance Considerations

This implementation requires:
- 1x GPU → CPU transfer (full buffer)
- 1x CPU-side reordering (element by element)
- 1x CPU → GPU transfer (contiguous buffer)

For small tensors, this is acceptable. For production use with large tensors, consider:
- Implementing a Vulkan compute shader for strided copy
- Caching temporary buffers
- Using async transfers

## Testing

The negative indexing test that was hanging should now pass:
- Test: `AllBackends/AdvancedIndexingTest.NegativeIndexing/vulkan`
- Code path: `t[-1]` → `slice()` → `squeeze()` → `.to(Device::cpu())`
- No longer hangs because `dispatchContiguous()` doesn't recurse

## Remaining Work

- Verify negative indexing test passes
- Run full test suite to check for improvements
- Consider performance optimizations for large tensors
