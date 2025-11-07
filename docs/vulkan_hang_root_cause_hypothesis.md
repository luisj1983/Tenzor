# Vulkan Hang - Root Cause Hypothesis

## Findings from Investigation

After extensive investigation, I've confirmed:

1. ✅ **Hang is NOT in `dispatchContiguous()`** - Test still hangs when that function throws immediately
2. ✅ **Hang is in `vkQueueWaitIdle()`** at line 415 of vulkan_backend.cpp
3. ✅ **Vulkan calls are NOT failing** - Added error checking, no exceptions thrown
4. ❌ **No thread safety mechanisms** - No mutexes protecting command pool or queue

## Current Hypothesis

The most likely cause is a **Vulkan command pool threading issue** or **command buffer re-use problem**.

### Evidence:

1. **Command Pool is Shared**: Each device has ONE command pool (line 62 of vulkan_backend.hpp)
2. **No Synchronization**: No mutexes protecting `beginSingleTimeCommands()` or `endSingleTimeCommands()`
3. **Sequential Test Execution**: Previous tests may have left the command pool in a bad state
4. **Command Buffer Lifecycle**: `beginSingleTimeCommands()` allocates from pool, `endSingleTimeCommands()` frees

### Potential Issues:

**Option A: Command Pool Exhaustion**
- Many tests run before this one (469 passed)
- Each test allocates/frees command buffers from the same pool
- Pool may be fragmented or exhausted
- vkAllocateCommandBuffers might succeed but pool is corrupt

**Option B: Queue State Corruption**
- Previous operations left queue in bad state
- No proper cleanup between tests
- Queue has pending operations that never completed

**Option C: Buffer Handle Issue**
- Sliced tensor's storage pointer might be invalid
- `impl_->storage->data()` returns wrong VkBuffer for sliced tensors
- Copy command references invalid buffer, queue never completes

## Test to Confirm

### Test A: Add Mutex Protection

Add mutex around command pool operations:
```cpp
std::mutex commandPoolMutex_;

VkCommandBuffer beginSingleTimeCommands(int32_t device_id) {
    std::lock_guard<std::mutex> lock(commandPoolMutex_);
    // existing code
}

void endSingleTimeCommands(VkCommandBuffer cmd, int32_t device_id) {
    std::lock_guard<std::mutex> lock(commandPoolMutex_);
    // existing code
}
```

### Test B: Reset Command Pool

Add command pool reset after each test or periodically:
```cpp
vkResetCommandPool(ctx.device, ctx.commandPool, 0);
```

### Test C: Use Dedicated Command Pool

Create a new command pool for each copy operation:
```cpp
VkCommandPoolCreateInfo poolInfo{};
poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
poolInfo.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
poolInfo.queueFamilyIndex = ctx.queueFamilyIndex;
VkCommandPool tempPool;
vkCreateCommandPool(ctx.device, &poolInfo, nullptr, &tempPool);
// use tempPool
vkDestroyCommandPool(ctx.device, tempPool, nullptr);
```

### Test D: Validate Buffer Handle

Add logging before copy:
```cpp
auto bufferHandle = reinterpret_cast<VkBuffer>(const_cast<void*>(src));
std::cerr << "Copying from buffer: " << (void*)bufferHandle
          << " size: " << bytes << std::endl;
```

## Next Steps

1. Try Test B first (command pool reset) - simplest
2. If that doesn't work, try Test C (dedicated pool)
3. If still hanging, add Test D logging to check buffer validity
4. As last resort, add Test A (mutex protection)

## Temporary Workaround

Until root cause is fixed:
- Skip the NegativeIndexing test for Vulkan backend
- Add test to CTest exclude list
- Document as known issue in KNOWN_ISSUES.md

