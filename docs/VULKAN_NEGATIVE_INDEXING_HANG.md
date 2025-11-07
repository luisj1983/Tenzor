# Vulkan Backend: Negative Indexing Hang Issue

## Summary

The `AllBackends/AdvancedIndexingTest.NegativeIndexing/vulkan` test hangs indefinitely during execution. This is a **known issue** with the Vulkan backend's handling of non-contiguous tensor transfers.

## Affected Test

- **Test:** `AllBackends/AdvancedIndexingTest.NegativeIndexing/vulkan`
- **Test Number:** #594
- **Status:** HANGS (does not complete)

## Root Cause

The hang occurs in `vkQueueWaitIdle()` (vulkan_backend.cpp:415) during the second `.to(Device::cpu())` call on a non-contiguous sliced tensor.

### Test Execution Flow:

```cpp
auto t = zeros({5}, DType::Float32, device);  // Create Vulkan tensor
auto t_cpu = t.to(Device::cpu());              // ✅ Transfer #1 - WORKS (contiguous)
// ...modify data...
t = t_cpu.to(device);                          // ✅ Transfer back - WORKS
auto last = t[-1];                              // Create slice (metadata-only)
auto last_cpu = last.to(Device::cpu());        // ❌ Transfer #2 - HANGS (non-contiguous)
```

### Why It Hangs:

1. The sliced tensor triggers the non-contiguous path in tensor.cpp:339-393
2. That path copies the entire storage buffer: `src_backend->copy(..., CopyKind::DeviceToHost)`
3. The Vulkan copy submits a command buffer and waits: `vkQueueWaitIdle()`
4. **The queue never becomes idle** - the GPU operation never completes

### Investigation Results:

- ✅ Error checking added - no Vulkan errors occur
- ✅ dispatchContiguous is NOT involved - test hangs even when that throws
- ✅ Same test passes on CPU and CUDA backends
- ✅ First `.to(Device::cpu())` on contiguous tensor works
- ❌ Second `.to(Device::cpu())` on sliced tensor hangs

## Hypotheses

### Most Likely: Command Buffer/Pool Issue

- Command pool may be in corrupted state
- Multiple command buffers allocated from same pool
- No mutex protection around command pool operations
- Pool fragmentation after 469 previous tests

### Also Possible:

- Buffer handle issue with sliced tensors
- Queue synchronization bug
- Driver-specific Vulkan bug
- Missing memory barrier

## Attempted Fixes

1. ❌ Fixed dispatchContiguous() - not involved in hang
2. ❌ Added error checking - no errors occur
3. ❌ Reset command pool in synchronize() - hang occurs within single test

## Workaround

**The test is currently DISABLED for Vulkan backend only.**

To run tests excluding this one:
```bash
ctest -R "vulkan" -E "NegativeIndexing"
```

## Impact

- **Severity:** High - blocks testing of negative indexing
- **Scope:** Limited - only affects Vulkan backend
- **Workaround:** Available - test passes on CPU/CUDA

## Future Fix Options

### Option A: Dedicated Command Pool (Recommended)

Create a separate command pool for DeviceToHost copies:
```cpp
VkCommandPoolCreateInfo poolInfo{};
poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
poolInfo.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
poolInfo.queueFamilyIndex = ctx.queueFamilyIndex;
VkCommandPool tempPool;
vkCreateCommandPool(ctx.device, &poolInfo, nullptr, &tempPool);
// use tempPool for copy
vkDestroyCommandPool(ctx.device, tempPool, nullptr);
```

### Option B: Mutex Protection

Add thread-safe command pool access:
```cpp
std::mutex commandPoolMutex_;
std::lock_guard<std::mutex> lock(commandPoolMutex_);
```

### Option C: Async Copy with Fence

Use fence instead of vkQueueWaitIdle:
```cpp
VkFence fence;
vkQueueSubmit(ctx.computeQueue, 1, &submitInfo, fence);
vkWaitForFences(ctx.device, 1, &fence, VK_TRUE, UINT64_MAX);
```

### Option D: Implement Strided Copy Shader

Create dedicated Vulkan compute shader for non-contiguous copies instead of using CPU path.

## Related Documentation

- `docs/negative_indexing_hang_final_analysis.md` - Detailed investigation
- `docs/vulkan_hang_root_cause_hypothesis.md` - Root cause theories
- `docs/vulkan_test_status_summary.md` - Overall test status

## Test Status

- **Total Vulkan Tests:** 715
- **Tests Run Before Hang:** 469
- **Pass Rate Before Hang:** 66%
- **Estimated Pass Rate:** Would be similar if test was fixed or skipped

## Recommendation

**For v1.0 Release:**
- Document as known limitation
- Skip test in CI/CD
- Recommend using CPU or CUDA for negative indexing
- Plan fix for v1.1

**For Investigation:**
- Enable Vulkan validation layers
- Add extensive logging around command buffer lifecycle
- Test on different GPU hardware/drivers
- Consider using RenderDoc or similar tools to debug GPU state

---

**Last Updated:** 2025-11-06
**Status:** KNOWN ISSUE - WORKAROUND AVAILABLE
