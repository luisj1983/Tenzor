# Negative Indexing Hang - Final Analysis

## Key Finding

**The hang is NOT in dispatchContiguous!**

By replacing dispatchContiguous with a simple exception throw, I confirmed that:
- The test still hangs
- dispatch Contiguous is never called during the hang
- Therefore, the hang must be occurring earlier in the call chain

## Hang Location

Since dispatchContiguous is not involved, the hang must be in one of:

1. **The `.to(Device::cpu())` call itself** (tensor.cpp:339-393)
   - This is the special path for non-contiguous GPU tensors
   - It does NOT call `.contiguous()`
   - But it might be hanging in the copy operations

2. **The slice() or squeeze() operations** (unlikely, as they're metadata-only)

3. **Device synchronization** in the Vulkan backend's copy operations

## Most Likely Cause

The `.to(Device::cpu())` method at line 358-359 calls:
```cpp
src_backend->copy(gpu_buffer.data(), impl_->storage->data(), total_bytes, CopyKind::DeviceToHost);
```

The Vulkan backend's copy() method (vulkan_backend.cpp:328-340) for DeviceToHost:
1. Gets a staging buffer
2. Calls `beginSingleTimeCommands(device_id)`
3. Issues vkCmdCopyBuffer
4. Calls `endSingleTimeCommands(cmdBuffer, device_id)`

**The hang is likely in `endSingleTimeCommands`** which probably waits for the GPU operation to complete.

## Possible Root Causes

1. **Vulkan command buffer not properly submitted/completed**
2. **Device synchronization deadlock**
3. **Staging buffer issue**
4. **Queue submission problem**

## Recommended Next Steps

1. Check the implementation of `endSingleTimeCommands` and `beginSingleTimeCommands`
2. Verify Vulkan queue is properly configured
3. Check if there's a fence or semaphore issue
4. Consider adding debug logging to pinpoint exact hang location

## Test Case

```cpp
// From test_ops_additional.cpp lines 1145-1155
TEST_P(AdvancedIndexingTest, NegativeIndexing) {
    auto t = create({5, 10});  // Create 5x10 tensor on Vulkan

    // This line completes fine (metadata-only)
    auto last = t[-1];  // slice + squeeze

    // This line HANGS
    auto last_cpu = last.to(Device::cpu());  // Transfer to CPU

    // Never reaches here
    EXPECT_EQ(last_cpu.shape(), std::vector<int64_t>{10});
}
```

## Status

- ❌ Not a dispatchContiguous issue
- ❓ Likely a Vulkan synchronization issue in copy operations
- ⏸️ Further investigation needed in Vulkan command submission code

## Workaround

For now, the negative indexing test should be disabled or skipped for the Vulkan backend until the underlying synchronization issue is resolved.
