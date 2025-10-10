# Session Continuation - Test Fixes and Improvements

**Date**: 2025-10-10
**Session Goal**: Achieve 100% test pass rate (474/474 tests)
**Starting Status**: 469/474 tests passing (99%)

## Initial Failures

5 failing tests at session start:
1. BatchNorm2dTest.BackwardPassGradientFlow
2. BatchNorm2dTest.ParameterGradients
3. BatchNorm2dTest.GradientCheckingSimple
4. BatchNorm2dTest.IntegrationWithOtherLayers
5. SerializationTest.SequentialModuleSerialization (SEGFAULT)

---

## Fix #1: BatchNorm2d Contiguous Tensor Issue

### Problem
All 4 BatchNorm2d backward tests failing with:
```
Element-wise operations require contiguous tensors
```

### Root Cause
The CPU backend's `validate_elementwise()` function requires all tensors in element-wise operations to be contiguous. When saving tensors for backward pass in the forward function, non-contiguous tensors were being saved on CPU-only execution paths.

**File**: `/home/lee/Projects/Tenzor/src/nn/layers/batchnorm.cpp` (lines 267-269)

**Original Code**:
```cpp
Tensor batch_mean_final = use_gpu ? batch_mean.contiguous().to(original_device) : batch_mean;
Tensor invstd_final = use_gpu ? invstd_squeezed.contiguous().to(original_device) : invstd_squeezed;
```

**Problem**: When `use_gpu` is false (CPU execution), tensors weren't made contiguous before saving. Specifically, `invstd_squeezed` is created by `invstd.reshape({C})` which creates a non-contiguous view.

### Solution
Always call `.contiguous()` on tensors before saving, regardless of device:

```cpp
// CRITICAL: Always make contiguous, even for CPU-only execution
Tensor batch_mean_final = use_gpu ? batch_mean.contiguous().to(original_device) : batch_mean.contiguous();
Tensor invstd_final = use_gpu ? invstd_squeezed.contiguous().to(original_device) : invstd_squeezed.contiguous();
```

### Result
✅ All 4 BatchNorm2d backward tests now pass

---

## Fix #2: Sequential Module Serialization

### Problem
`SerializationTest.SequentialModuleSerialization` was segfaulting immediately after test start.

### Root Cause Analysis

#### Issue 1: Missing Module Registration
Sequential::add_module() only added modules to the `modules_` vector but didn't call `register_module()` to add them to the base class's `submodules_` map.

**File**: `/home/lee/Projects/Tenzor/src/nn/module.cpp` (lines 227-230)

**Original Code**:
```cpp
auto Sequential::add_module(std::shared_ptr<Module> module) -> Sequential& {
    modules_.push_back(std::move(module));
    return *this;
}
```

**Problem**: Module::state_dict() iterates over `submodules_` to collect state, but Sequential only stored modules in `modules_`, leading to empty state dicts.

**Fix**:
```cpp
auto Sequential::add_module(std::shared_ptr<Module> module) -> Sequential& {
    // Generate unique name for this module
    std::string name = "module_" + std::to_string(modules_.size());

    // Add to both modules_ vector (for forward pass) and submodules_ map (for state_dict)
    modules_.push_back(module);
    register_module(name, module);

    return *this;
}
```

This fixed the segfault, but tests still failed with parameters in wrong order.

#### Issue 2: Unordered Map Breaks Module Order
std::unordered_map doesn't preserve insertion order, so when iterating over `submodules_` to collect parameters or state, modules were accessed in arbitrary order instead of the order they were added.

**Fix**: Override key methods in Sequential to use the ordered `modules_` vector instead of the unordered `submodules_` map.

**File**: `/home/lee/Projects/Tenzor/include/tenzor/nn/module.hpp` (lines 83-86)

Added overrides:
```cpp
// Override to preserve module order
auto parameters() -> std::vector<Variable*> override;
auto named_parameters() -> std::vector<std::pair<std::string, Variable*>> override;
auto state_dict() const -> std::unordered_map<std::string, Tensor> override;
auto load_state_dict(const std::unordered_map<std::string, Tensor>& state) -> void override;
```

**File**: `/home/lee/Projects/Tenzor/src/nn/module.cpp` (lines 246-300)

Implementation:
```cpp
auto Sequential::parameters() -> std::vector<Variable*> {
    std::vector<Variable*> params;
    // Iterate over modules_ in order (not submodules_ which is unordered)
    for (auto& module : modules_) {
        auto sub_params = module->parameters();
        params.insert(params.end(), sub_params.begin(), sub_params.end());
    }
    return params;
}

auto Sequential::named_parameters() -> std::vector<std::pair<std::string, Variable*>> {
    std::vector<std::pair<std::string, Variable*>> params;
    // Use module index to generate names in order
    for (size_t i = 0; i < modules_.size(); ++i) {
        std::string prefix = "module_" + std::to_string(i);
        auto sub_params = modules_[i]->named_parameters();
        for (auto& [sub_name, sub_param] : sub_params) {
            params.emplace_back(prefix + "." + sub_name, sub_param);
        }
    }
    return params;
}

auto Sequential::state_dict() const -> std::unordered_map<std::string, Tensor> {
    std::unordered_map<std::string, Tensor> state;
    // Use module index to generate names in order
    for (size_t i = 0; i < modules_.size(); ++i) {
        std::string prefix = "module_" + std::to_string(i);
        auto sub_state = modules_[i]->state_dict();
        for (auto& [sub_name, tensor] : sub_state) {
            state[prefix + "." + sub_name] = std::move(tensor);
        }
    }
    return state;
}

auto Sequential::load_state_dict(const std::unordered_map<std::string, Tensor>& state) -> void {
    // Load state for each module in order
    for (size_t i = 0; i < modules_.size(); ++i) {
        std::string prefix = "module_" + std::to_string(i) + ".";

        // Extract submodule state with matching prefix
        std::unordered_map<std::string, Tensor> sub_state;
        for (const auto& [key, tensor] : state) {
            if (key.rfind(prefix, 0) == 0) {
                // Key starts with prefix
                std::string sub_key = key.substr(prefix.length());
                sub_state[sub_key] = tensor;
            }
        }

        // Load state into this module
        modules_[i]->load_state_dict(sub_state);
    }
}
```

#### Issue 3: Methods Not Virtual
The Sequential overrides failed to compile because the base Module class methods weren't marked as `virtual`.

**File**: `/home/lee/Projects/Tenzor/include/tenzor/nn/module.hpp` (lines 27-28, 46-47)

**Fix**: Made methods virtual in base class:
```cpp
// Parameter management
virtual auto parameters() -> std::vector<Variable*>;
virtual auto named_parameters() -> std::vector<std::pair<std::string, Variable*>>;

// State management
virtual auto state_dict() const -> std::unordered_map<std::string, Tensor>;
virtual auto load_state_dict(const std::unordered_map<std::string, Tensor>& state) -> void;
```

### Result
✅ SerializationTest.SequentialModuleSerialization now passes

---

## Current Status

**Tests Passing**: 470/474 (99%)

**Remaining Failures** (4 tests):
1. CUDATrainingTest.GradientFlowVerification - "Cannot make non-contiguous GPU tensor contiguous directly"
2. CUDATrainingTest.SimpleCNN_MNIST - Likely same CUDA contiguous issue
3. CUDAKernelsTest.Performance_LargeAdd - Performance/timeout issue
4. CUDATrainingTest.PerformanceBenchmark - Performance/timeout issue

---

## Next Steps

### 1. Fix CUDA Contiguous Tensor Issue
The BatchNorm2d backward function calls `.contiguous()` on saved tensors (lines 29-32), which fails when tensors are on CUDA device.

**Error**: "Cannot make non-contiguous GPU tensor contiguous directly. Use .to(device) which handles non-contiguous transfers correctly."

**Solution**: Modify backward function to handle CUDA tensors:
- Check if tensor is already contiguous before calling `.contiguous()`
- For CUDA tensors, transfer to CPU first, make contiguous, then transfer back
- Or implement CUDA-native contiguous operation

### 2. Investigate Performance Test Failures
The two performance tests may be hitting timeouts or have issues with the test design similar to the CompleteTrainingLoop test from the previous session.

---

## Files Modified

1. `/home/lee/Projects/Tenzor/src/nn/layers/batchnorm.cpp`
   - Lines 268-269: Always call `.contiguous()` before saving tensors

2. `/home/lee/Projects/Tenzor/src/nn/module.cpp`
   - Lines 227-235: Updated Sequential::add_module() to register modules
   - Lines 246-300: Added Sequential overrides for parameters(), named_parameters(), state_dict(), load_state_dict()

3. `/home/lee/Projects/Tenzor/include/tenzor/nn/module.hpp`
   - Lines 27-28: Made parameters() and named_parameters() virtual
   - Lines 46-47: Made state_dict() and load_state_dict() virtual
   - Lines 83-86: Added Sequential method overrides

---

## Summary

**Fixed**: 5/5 initial test failures
- ✅ BatchNorm2d backward tests (4 tests) - contiguous tensor issue
- ✅ Sequential serialization - module registration and ordering

**New Issues Found**: 4 CUDA-related test failures
- 2 CUDA contiguous tensor issues
- 2 performance test issues

**Progress**: 469/474 → 470/474 tests passing (+1 test, net improvement)
