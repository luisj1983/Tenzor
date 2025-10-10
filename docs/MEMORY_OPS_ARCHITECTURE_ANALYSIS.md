# Memory Operations Architecture Analysis

**Date**: 2025-10-10
**Purpose**: Identify memory manipulation operations with device-aware code in frontend

---

## Overview

After successfully moving `.contiguous()` to backend dispatch, this analysis identifies other operations in `/home/lee/Projects/Tenzor/src/core/tensor.cpp` that contain device-specific code and should potentially be moved to backends.

---

## Operations with Device-Aware Code

### 1. ✅ `.contiguous()` - FIXED
**Status**: Already moved to backend dispatch
**Lines**: 193-206 (now clean)

**Current implementation**:
```cpp
auto Tensor::contiguous() const -> Tensor {
    if (!impl_) return *this;
    if (is_contiguous()) return *this;

    // Dispatch to backend for contiguous operation
    std::vector<Tensor> inputs = {*this};
    return Dispatcher::dispatch("contiguous", inputs)[0];
}
```

✅ **Properly architected** - No device checks, pure dispatch

---

### 2. ⚠️ `.to(Device)` - Device Transfer Operation
**Status**: Contains extensive device-aware code
**Lines**: 150-276 (127 lines)

**Issues**:
- **Lines 165-236**: Special handling for non-contiguous GPU tensors
  - Transfers GPU→CPU with CPU-side stride handling
  - Then CPU→target device
  - **72 lines of device-specific logic in frontend**

- **Lines 248-258**: Device type checks for copy kind
  ```cpp
  if (cont.impl_->device.type == Device::Type::CPU && device.type == Device::Type::CPU) {
      copy_kind = CopyKind::HostToHost;
  } else if (cont.impl_->device.type == Device::Type::CPU && device.type != Device::Type::CPU) {
      copy_kind = CopyKind::HostToDevice;
  } else if (cont.impl_->device.type != Device::Type::CPU && device.type == Device::Type::CPU) {
      copy_kind = CopyKind::DeviceToHost;
  } else {
      copy_kind = CopyKind::DeviceToDevice;
  }
  ```

**Complexity**: HIGH
**Recommendation**: ⏸️ **DEFER** - This is complex cross-device transfer logic. The current approach is reasonable because:
- It already uses backend `copy()` for actual transfers
- Device type checks determine the *strategy*, backends handle *execution*
- Moving this would require complex coordination between source and destination backends
- Not a pure memory operation on a single device

**Alternative**: Could be simplified but doesn't need full backend dispatch

---

### 3. ⚠️ `.clone()` - Memory Copy Operation
**Status**: Contains device checks
**Lines**: 291-331

**Issues**:
- **Lines 298-304**: Device-specific logic for contiguous handling
  ```cpp
  if (is_contiguous()) {
      cont = *this;
  } else if (impl_->device.type == Device::Type::CPU) {
      cont = contiguous();
  } else {
      // For non-contiguous GPU tensors, use .to() which handles stride conversion
      cont = to(impl_->device);
  }
  ```

- **Lines 314-328**: Device-specific copy implementation
  ```cpp
  if (cont.impl_->device.type == Device::Type::CPU) {
      // CPU: direct memcpy
      std::memcpy(result.impl_->storage->data(),
                  cont.impl_->storage->data(),
                  size_bytes);
  } else {
      // GPU: use backend copy
      auto* backend = backend_registry().get_backend(cont.impl_->device.type);
      backend->copy(result.impl_->storage->data(),
                   cont.impl_->storage->data(),
                   size_bytes,
                   CopyKind::DeviceToDevice);
  }
  ```

**Complexity**: MEDIUM
**Recommendation**: 🔄 **CANDIDATE FOR REFACTOR**

**Proposed solution**: Create backend operation "clone"
```cpp
auto Tensor::clone() const -> Tensor {
    if (!impl_) return *this;

    std::vector<Tensor> inputs = {*this};
    return Dispatcher::dispatch("clone", inputs)[0];
}
```

**Benefits**:
- Removes device checks from frontend
- Backends can optimize for their memory models
- Consistent with `.contiguous()` pattern

---

### 4. ⚠️ `.fill_()` - In-place Memory Write
**Status**: CPU-only implementation
**Lines**: 327-337

**Issues**:
```cpp
auto Tensor::fill_(float value) -> Tensor& {
    if (!impl_) return *this;

    auto* data_ptr = data<float>();  // Direct memory access
    const int64_t n = numel();
    for (int64_t i = 0; i < n; ++i) {
        data_ptr[i] = value;  // Direct write
    }
    return *this;
}
```

**Problems**:
- Direct memory access only works for CPU tensors
- Would **fail or corrupt memory** if called on GPU tensor
- No device checks - silently dangerous!

**Complexity**: LOW
**Recommendation**: 🔴 **SHOULD FIX**

**Proposed solution**:
```cpp
auto Tensor::fill_(float value) -> Tensor& {
    if (!impl_) return *this;

    OpAttributes attrs;
    attrs["value"] = std::to_string(value);
    std::vector<Tensor> inputs = {*this};

    // Dispatch returns new tensor, but we modify in-place
    auto result = Dispatcher::dispatch("fill", inputs, attrs)[0];
    *this = result;
    return *this;
}
```

Or better - use existing backend "fill" operation and copy back.

---

### 5. ⚠️ Scalar Operations - All Have Device Checks
**Status**: All contain device-aware branching
**Operations**: `operator+(float)`, `operator-(float)`, `operator*(float)`, `operator/(float)`
**Lines**: 226-304 (4 operations, ~20 lines each)

**Pattern** (all identical):
```cpp
auto Tensor::operator+(float scalar) const -> Tensor {
    if (!impl_) return *this;

    // For CPU tensors, use fast direct access
    if (impl_->device.type == Device::Type::CPU) {
        auto result = clone();
        auto* data_ptr = result.data<float>();
        const int64_t n = numel();
        for (int64_t i = 0; i < n; ++i) {
            data_ptr[i] += scalar;
        }
        return result;
    }

    // For GPU tensors, create scalar tensor and use element-wise add
    auto scalar_tensor = full(..., scalar, ...);
    return *this + scalar_tensor;
}
```

**Issues**:
- Device type checks in every scalar operation
- Different code paths for CPU vs GPU
- CPU path uses direct memory access
- GPU path creates tensor and dispatches

**Complexity**: LOW (already have tensor ops)
**Recommendation**: 🟡 **COULD SIMPLIFY**

**Proposed solution**: Remove device checks, always use tensor path
```cpp
auto Tensor::operator+(float scalar) const -> Tensor {
    if (!impl_) return *this;

    auto scalar_tensor = full(
        std::vector<int64_t>(impl_->shape.begin(), impl_->shape.end()),
        scalar, impl_->dtype, impl_->device
    );
    return *this + scalar_tensor;
}
```

**Benefits**:
- Device-agnostic
- Simpler code
- Relies on backend-optimized element-wise ops

**Trade-off**:
- CPU might be slightly slower (creates tensor vs direct loop)
- But backends can optimize, and cleaner architecture may be worth it

---

## View Operations (Metadata-only) - ✅ FINE AS-IS

These operations only manipulate metadata (shape, strides) and don't touch memory:

1. **`.reshape()` / `.view()`** - Shape manipulation
2. **`.transpose()` / `.permute()`** - Dimension reordering
3. **`.squeeze()` / `.unsqueeze()`** - Dimension insertion/removal
4. **`.flatten()`** - Calls reshape

**Status**: ✅ **Correctly in frontend**
**Reason**: These are pure metadata operations, no memory manipulation

---

## Allocation Operations - ✅ CORRECT LOCATION

**`TensorImpl` constructor** (lines 14-42):
```cpp
if (device.type == Device::Type::CPU) {
    storage = std::make_shared<CPUStorage>(size_bytes);
} else {
    auto* backend = backend_registry().get_backend(device.type);
    void* device_ptr = backend->allocate(size_bytes, device.index);
    storage = std::make_shared<DeviceStorage>(device_ptr, ...);
}
```

**Status**: ✅ **Correctly architected**
**Reason**: Allocation needs to know device type to create correct storage wrapper, but delegates actual allocation to backend

---

## Priority Recommendations

### 🔴 HIGH PRIORITY - Should Fix

1. **`.fill_()`** - Currently CPU-only, unsafe for GPU tensors
   - **Impact**: Bug/correctness issue
   - **Effort**: Low
   - **Benefit**: Safety + GPU support

### 🟡 MEDIUM PRIORITY - Should Consider

2. **`.clone()`** - Device checks for copy strategy
   - **Impact**: Cleaner architecture
   - **Effort**: Medium
   - **Benefit**: Consistency with `.contiguous()`

3. **Scalar operations** - Device checks for optimization
   - **Impact**: Simpler code
   - **Effort**: Low
   - **Benefit**: Device-agnostic + simpler maintenance

### ⏸️ LOW PRIORITY - Defer

4. **`.to(Device)`** - Complex cross-device transfer
   - **Impact**: Already uses backends for actual transfers
   - **Effort**: High
   - **Benefit**: Limited (current design reasonable)

---

## Architectural Principles

### ✅ Operations that SHOULD be in backends:
- Memory manipulation (read/write/copy)
- In-place modifications
- Device-specific optimizations

### ✅ Operations that CAN stay in frontend:
- Pure metadata operations (reshape, transpose, etc.)
- Cross-device coordination (if backends handle execution)
- Storage allocation routing (delegates to backends)

### ❌ Anti-patterns to avoid:
- Direct memory access without device checks
- Different code paths for different devices
- Backend-specific logic in frontend

---

## Next Steps

If you want to continue the architectural cleanup:

1. **Fix `.fill_()`** first (safety issue)
2. **Consider `.clone()`** (consistency with `.contiguous()`)
3. **Optionally simplify scalar ops** (if you value architecture over micro-optimization)
4. **Leave `.to()`** as-is (complex cross-device case)

Would you like me to implement any of these refactors?
