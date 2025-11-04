# Backend Parity Implementation Roadmap
**Date:** 2025-11-04
**Current Status:** Phase 1 Complete - 15 OneAPI Operations Registered
**Goal:** 100% operation coverage across all backends

---

## Progress Summary

### ✅ Completed (Phase 1)

**OneAPI Backend - 15 New Operations Registered:**
1. Backward activations (5): relu_backward, sigmoid_backward, tanh_backward, leaky_relu_backward, gelu_backward
2. Activation functions (2): gelu, softmax_backward
3. Normalization (5): batchnorm2d_mean_var, batchnorm2d_update_running_stats, batchnorm2d_forward, batchnorm2d_forward_affine, batchnorm2d_backward
4. Softmax operations (3): softmax, log_softmax, log_softmax_backward

**Current OneAPI Coverage:**
- **Before Phase 1:** 33/67 operations (49%)
- **After Phase 1:** 48/67 operations (72%) ✅

**Build Status:** All 160 targets compile successfully ✅

---

## Remaining Work

### OneAPI Backend - 19 Operations Missing

**Critical Priority (Comparison Operators) - 6 ops:**
```
eq, ne, lt, le, gt, ge
```
**Why Critical:** Used extensively in test assertions across all test suites

**High Priority (Utilities) - 5 ops:**
```
cat, clamp, sign, im2col, col2im
```
**Why High:** Fundamental tensor operations used by many high-level operations

**Medium Priority (Activations) - 2 ops:**
```
swish, swish_backward
```

**Low Priority (Fused Operations) - 7 ops:**
```
fused_add_relu, fused_batchnorm_relu, fused_conv2d_relu,
fused_gelu, fused_layer_norm, fused_linear_relu, fused_softmax_cross_entropy
```
**Why Low:** Performance optimizations, can fallback to separate operations

**Missing Conv Ops - 1 op:**
```
conv2d_backward_bias (separate registration, backend has unified conv2d_backward)
```

---

### Vulkan Backend - 48 Operations Missing

**Shape Operations - 9 ops (CRITICAL):**
```
zeros, fill, clone, contiguous, reshape, transpose,
permute, squeeze, unsqueeze
```
**Status:** These operations are ESSENTIAL for test infrastructure
**Impact:** 247 tests currently skip due to missing these operations

**Comparison Operations - 6 ops (CRITICAL):**
```
eq, ne, lt, le, gt, ge
```

**Backward Activations - 7 ops:**
```
relu_backward, sigmoid_backward, tanh_backward,
leaky_relu_backward, gelu_backward, softmax_backward, log_softmax_backward
```

**Forward Activations - 3 ops:**
```
leaky_relu, gelu, swish, swish_backward
```

**Math Operations - 2 ops:**
```
pow, sign
```

**Utilities - 3 ops:**
```
cat, clamp, expand
```

**Convolution Operations - 4 ops:**
```
conv2d_forward, conv2d_backward_input, conv2d_backward_weight, conv2d_backward_bias
```

**Batch Normalization - 5 ops:**
```
batchnorm2d_mean_var, batchnorm2d_update_running_stats,
batchnorm2d_forward, batchnorm2d_forward_affine, batchnorm2d_backward
```

**Im2col/Col2im - 2 ops:**
```
im2col, col2im
```

**Fused Operations - 7 ops:**
```
fused_add_relu, fused_batchnorm_relu, fused_conv2d_relu,
fused_gelu, fused_layer_norm, fused_linear_relu, fused_softmax_cross_entropy
```

---

## Implementation Strategy

### Phase 2: Comparison Operators (2-3 hours)

**Target:** Implement eq, ne, lt, le, gt, ge for both OneAPI and Vulkan

**OneAPI Implementation:**
1. Create `src/backends/oneapi/kernels/comparison.cpp`
2. Implement SYCL kernels for 6 comparison ops
3. Add dispatch cases in `oneapi_backend.cpp`
4. Register in `init.cpp`

**Vulkan Implementation:**
1. Create `shaders/vulkan/comparison.comp` shader
2. Add dispatch cases in `vulkan_backend.cpp` (use math shader pattern with opcodes)
3. Register in `init.cpp`

**Expected Impact:**
- OneAPI: 54/67 operations (81%)
- Vulkan: 38/67 operations (57%)
- Test coverage increase: ~15-20% across all suites

---

### Phase 3: Vulkan Shape Operations (3-4 hours)

**Target:** Implement zeros, fill, clone, contiguous, reshape, transpose, permute, squeeze, unsqueeze

**Critical for Test Infrastructure** - These enable 247 skipped tests to run

**Implementation Approach:**
- Most shape ops can be CPU-side metadata operations (no shader needed)
- Only `fill` requires shader execution
- `zeros` can allocate and memset to 0
- `clone`/`contiguous` use device-to-device copy

**Expected Impact:**
- Vulkan: 47/67 operations (70%)
- Test coverage: 247 previously skipped tests will run

---

### Phase 4: OneAPI Utilities (2-3 hours)

**Target:** Implement cat, clamp, sign, im2col, col2im

**Implementation:**
- `cat`: Concatenate tensors along dimension (requires SYCL kernel)
- `clamp`: Clamp values to min/max range (simple SYCL kernel)
- `sign`: Return sign of elements (-1, 0, 1) (simple SYCL kernel)
- `im2col`/`col2im`: Image-to-column transformations for conv (complex kernels)

**Expected Impact:**
- OneAPI: 59/67 operations (88%)

---

### Phase 5: Remaining Critical Operations (3-4 hours)

**Target:** Implement remaining high-value operations

**OneAPI:**
- swish, swish_backward (2 ops)
- conv2d_backward_bias wrapper (1 op)

**Vulkan:**
- leaky_relu, gelu, swish (3 ops)
- backward activations (7 ops)
- pow, sign (2 ops)

**Expected Impact:**
- OneAPI: 62/67 operations (93%)
- Vulkan: 59/67 operations (88%)

---

### Phase 6: Advanced Operations (4-5 hours)

**Target:** Complete remaining operations for 100% parity

**Both Backends:**
- Convolution backward ops
- Batch normalization (5 ops)
- Im2col/col2im
- Fused operations (7 ops each)

**Expected Impact:**
- OneAPI: 67/67 operations (100%) ✅
- Vulkan: 67/67 operations (100%) ✅

---

## File Structure for New Implementations

### OneAPI Kernels

```
src/backends/oneapi/kernels/
├── comparison.cpp          # eq, ne, lt, le, gt, ge
├── utilities.cpp           # cat, clamp, sign
├── activations_extended.cpp # swish, swish_backward
└── im2col.cpp              # im2col, col2im
```

### Vulkan Shaders

```
shaders/vulkan/
├── comparison.comp         # Comparison operations
├── shape_ops.comp          # Reshape, transpose, permute
├── activations_extended.comp # leaky_relu, gelu, swish
└── backward_activations.comp # Backward pass for activations
```

---

## Testing Strategy

### Incremental Testing

After each phase, run targeted tests:

```bash
# Test OneAPI backend
./bin/test_oneapi_backend

# Test specific operation category
ctest -R "comparison" --output-on-failure
ctest -R "batchnorm" --output-on-failure
ctest -R "activation" --output-on-failure

# Full test suite
cd build && ctest --output-on-failure
```

### Success Criteria

**Phase 2:** Comparison ops work, test failures decrease by ~15%
**Phase 3:** Vulkan tests no longer skip, 247 tests run
**Phase 4:** OneAPI reaches ~90% test pass rate
**Phase 5:** Both backends reach ~90% operation coverage
**Phase 6:** Both backends achieve 100% parity with CPU backend

---

## Quick Win Opportunities

### 1. Vulkan Shape Operations (Highest ROI)

**Effort:** 3-4 hours
**Impact:** Enable 247 skipped tests (~23% of test suite)
**Priority:** **HIGHEST**

Many shape operations don't require shaders:
- `reshape`: Metadata-only (change shape, keep data)
- `transpose`: Can use CPU or simple copy with stride changes
- `squeeze`/`unsqueeze`: Metadata-only
- `clone`/`contiguous`: Device copy operations

### 2. Comparison Operators (Highest Test Impact)

**Effort:** 2-3 hours
**Impact:** Enable test assertions across ALL test suites
**Priority:** **HIGHEST**

Without comparison ops, tests cannot verify results properly, causing many failures/skips.

### 3. OneAPI Advanced Operations (Already Implemented!)

**Effort:** 0 hours (already done in Phase 1)
**Impact:** Batchnorm, backward activations, softmax now available
**Priority:** **COMPLETE** ✅

---

## Estimated Time Remaining

**To reach 90% coverage (recommended minimum):**
- Phase 2 (Comparisons): 2-3 hours
- Phase 3 (Vulkan Shape): 3-4 hours
- Phase 4 (OneAPI Utils): 2-3 hours
- **Total:** 7-10 hours

**To reach 100% coverage (full parity):**
- Phases 2-4: 7-10 hours
- Phase 5 (Remaining): 3-4 hours
- Phase 6 (Advanced): 4-5 hours
- **Total:** 14-19 hours

---

## Current Status Summary

### OneAPI Backend
- **Operations:** 48/67 (72%)
- **Test Pass Rate:** 45/46 (98%) for backend-specific tests
- **Remaining:** 19 operations
- **Critical Missing:** Comparison ops, utilities

### Vulkan Backend
- **Operations:** 32/67 (48%)
- **Test Pass Rate:** 100% for standalone tests
- **Remaining:** 35 operations (but 9 are simple shape ops)
- **Critical Missing:** Shape operations (blocking 247 tests), comparison ops

### Build System
- **Status:** ✅ All 160 targets compile
- **Test Compilation:** ✅ All test files fixed
- **Ready for:** Incremental testing as operations are implemented

---

## Next Immediate Steps

1. **Implement comparison operators for OneAPI** (highest test impact)
2. **Implement Vulkan shape operations** (unblock 247 tests)
3. **Test and verify improvements**
4. **Continue with remaining operations systematically**

---

## Success Metrics

**Current State:**
- OneAPI: 72% operation coverage, 98% backend test pass rate
- Vulkan: 48% operation coverage, 100% standalone test pass rate

**Target State (90% coverage):**
- OneAPI: 90% operation coverage, 95%+ test pass rate
- Vulkan: 75% operation coverage, 80%+ test pass rate

**Final State (100% parity):**
- OneAPI: 100% operation coverage, 95%+ test pass rate
- Vulkan: 100% operation coverage, 95%+ test pass rate

---

**Report Generated:** 2025-11-04 19:15:00
**Phase 1 Status:** ✅ COMPLETE (15 operations registered)
**Overall Progress:** 72% OneAPI, 48% Vulkan
**Estimated Completion:** 14-19 hours for full parity
