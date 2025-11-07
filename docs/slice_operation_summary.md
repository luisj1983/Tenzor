# Slice Operation Analysis - Quick Reference

## Key Findings

### ✅ No Recursion Detected
The `operator[]` → `slice()` → `squeeze()` chain does **NOT** contain any recursion:
- Core operations are metadata-only
- No circular calls between operations
- Clear separation between core and backend layers

### ⚠️ Potential Issue: Vulkan Backend Inefficiency

**Location**: `/home/lee/Projects/Tenzor/src/backends/vulkan/vulkan_backend.cpp`

**Issue**: Vulkan backend performs unnecessary data copies for contiguous tensors

---

## Call Chains

### 1. Tensor::operator[] (User-Facing)
```
operator[](idx)                    [tensor.cpp:1106]
  ├─> slice(0, idx, idx+1, 1)     [tensor.cpp:1133]  ✓ Metadata-only
  └─> squeeze(0)                   [tensor.cpp:1137]  ✓ Metadata-only
       Returns: New tensor sharing original storage
```

### 2. Tensor::slice() (Core)
```
slice(dim, start, end, step)       [tensor.cpp:1142]
  └─> Creates TensorImpl copy      [tensor.cpp:1177]
      ├─> Shares storage            ✓ Zero-copy
      ├─> Updates shape[dim]
      ├─> Updates offset
      └─> Updates strides[dim]
```

### 3. Tensor::squeeze() (Core)
```
squeeze(dim)                       [tensor.cpp:969]
  └─> Creates TensorImpl copy      [tensor.cpp:991]
      ├─> Shares storage            ✓ Zero-copy
      ├─> Erases shape[dim]
      └─> Erases strides[dim]
```

### 4. VulkanBackend::dispatchSqueeze (Backend)
```
dispatchSqueeze(input, dim)        [vulkan_backend.cpp:3234]
  └─> dispatchReshape(input, out_shape)  [vulkan_backend.cpp:3272]
       └─> Creates new tensor       [vulkan_backend.cpp:3086]
           └─> copy() DeviceToDevice  [vulkan_backend.cpp:3088]  ⚠️ Always copies
```

### 5. Tensor::reshape() (User-Facing → Backend)
```
reshape(new_shape)                 [tensor.cpp:789]
  └─> Dispatcher::dispatch("reshape", ...)  [tensor.cpp:855]
       └─> operation_registry().dispatch()  [dispatch.cpp:17]
            └─> VulkanBackend::dispatch("reshape", ...)  [vulkan_backend.cpp:900]
                 └─> dispatchReshape()     [vulkan_backend.cpp:3067]
```

---

## Architecture Layers

```
┌────────────────────────────────────────┐
│  USER API LAYER                        │
│  • operator[](), slice(), squeeze()    │
│  • Metadata-only, zero-copy            │
│  • Share storage with source tensor    │
└──────────────┬─────────────────────────┘
               │
               │ reshape() explicitly dispatches
               ↓
┌────────────────────────────────────────┐
│  DISPATCHER LAYER                      │
│  • Dispatcher::dispatch()              │
│  • Routes operations to backend        │
│  • No circular dependencies            │
└──────────────┬─────────────────────────┘
               │
               ↓
┌────────────────────────────────────────┐
│  BACKEND LAYER                         │
│  • VulkanBackend::dispatch()           │
│  • dispatchSqueeze() → dispatchReshape()│
│  ⚠️ Always creates new tensor + copy    │
└────────────────────────────────────────┘
```

---

## Core vs Backend Behavior

| Operation | Core (tensor.cpp) | Vulkan Backend |
|-----------|-------------------|----------------|
| `slice()` | Metadata-only, shares storage | N/A (not dispatched) |
| `squeeze()` | Metadata-only, shares storage | Creates new tensor, copies data |
| `reshape()` | Dispatches to backend | Creates new tensor, copies data |
| Memory | Zero-copy for contiguous | Always copies |

---

## Investigation Checklist

### ✓ Completed Analysis
- [x] Traced operator[] implementation
- [x] Analyzed slice() call chain
- [x] Examined squeeze() implementation
- [x] Checked Vulkan backend dispatch
- [x] Verified no recursion exists
- [x] Identified backend inefficiency

### → Next Steps for Debugging
- [ ] Test with CPU backend to compare behavior
- [ ] Add logging to track data copies
- [ ] Verify storage sharing in Vulkan backend
- [ ] Check if issue is specific to Vulkan
- [ ] Measure performance impact of unnecessary copies
- [ ] Review VulkanBackend memory management

---

## Recommended Test Case

```cpp
// Test storage sharing behavior
Tensor t = ones({3, 1, 4}, DType::Float32, Device::vulkan(0));
Tensor squeezed = t.squeeze(1);

// Check if storage is shared
void* original_ptr = t.data_ptr();
void* squeezed_ptr = squeezed.data_ptr();

bool shares_storage = (original_ptr == squeezed_ptr);
std::cout << "Storage shared: " << shares_storage << std::endl;

// Expected: true (metadata-only operation)
// If false: Vulkan backend is copying unnecessarily
```

---

## File Locations

### Core Implementation
```
/home/lee/Projects/Tenzor/src/core/tensor.cpp
  Lines 1106-1140: operator[]
  Lines 1142-1191: slice()
  Lines 969-1021: squeeze()
```

### Dispatcher
```
/home/lee/Projects/Tenzor/src/backend/dispatch.cpp
  Lines 8-18: Dispatcher::dispatch()
```

### Vulkan Backend
```
/home/lee/Projects/Tenzor/src/backends/vulkan/vulkan_backend.cpp
  Lines 595-942: VulkanBackend::dispatch()
  Lines 3234-3273: dispatchSqueeze()
  Lines 3067-3098: dispatchReshape()
```

---

## Conclusion

**No recursion exists**. The issue, if present, is in:
1. Vulkan backend performing unnecessary copies
2. Lack of metadata-only optimization in backend
3. Inconsistency between core and backend implementations

**Not a recursion problem** - investigation should focus on:
- Vulkan backend memory management
- Storage sharing behavior
- Performance optimization opportunities
