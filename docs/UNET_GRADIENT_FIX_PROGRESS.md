# UNet Gradient Flow Fix - Progress Report

**Date:** October 20, 2025
**Status:** 🟡 PARTIAL SUCCESS - Gradient graph connected, but hitting dimension error

---

## 📊 Summary

Fixed the UNet gradient flow by using gradient-aware `cat()` function instead of raw tensor concatenation. The autograd graph is now properly connected and backward pass executes, but hits a dimension mismatch error.

---

## ✅ What Was Fixed

### Root Cause Identified

**File:** `/home/lee/Projects/Tenzor/src/models/unet.cpp:130-139`

**Problem:** UNet's `Up` layer was using raw tensor `cat()` which creates Variables without `grad_fn`, breaking the computational graph:

```cpp
// BEFORE (BROKEN) - No gradient function attached
std::vector<Tensor> tensors = {x.tensor(), skip.tensor()};
auto concatenated = cat(tensors, 1);
auto concat_var = Variable(concatenated, requires_grad);  // No grad_fn!
```

**Solution:** Use gradient-aware `cat()` from `tenzor/autograd/ops.hpp`:

```cpp
// AFTER (FIXED) - Uses CatBackward for gradients
std::vector<Variable> vars = {x, skip};
auto concat_var = tenzor::cat(vars, 1);  // Has CatBackward grad_fn
```

### Changes Made

| File | Change | Lines |
|------|--------|-------|
| `src/models/unet.cpp` | Added `#include "tenzor/autograd/ops.hpp"` | 10 |
| `src/models/unet.cpp` | Changed concatenation to use gradient-aware cat() | 130-135 |

---

## 🎯 Test Results

### Before Fix
```
[ RUN      ] UNetTest.UNetGradientFlow
/home/lee/Projects/Tenzor/tests/unit/test_unet.cpp:38: Failure
Value of: images.grad().has_value()
  Actual: false
Expected: true

[  FAILED  ] UNetTest.UNetGradientFlow (144324 ms)
```

**Issue:** Gradients never accumulated to leaf variable because computational graph was broken.

### After Fix
```
[ RUN      ] UNetTest.UNetGradientFlow
unknown file: Failure
C++ exception with description "matmul dimension mismatch: (512×64) @ (256×16384)" thrown in the test body.

[  FAILED  ] UNetTest.UNetGradientFlow (143983 ms)
```

**Status:** Backward pass IS executing (graph connected!), but hits dimension error.

---

## 🔍 Analysis

### Progress Made ✅

1. **Autograd graph connected** - Using `tenzor::cat()` with `CatBackward` properly chains gradient functions
2. **Backward pass executes** - The error occurs DURING backward, not before it starts
3. **Test runs to completion** - No crashes, clean error message

### New Issue ❌

**Error:** `"matmul dimension mismatch: (512×64) @ (256×16384)"`

**Interpretation:**
- Forward pass completes successfully
- Backward pass starts and propagates gradients
- Hits a matmul operation with incompatible dimensions
- 16384 = 128 × 128, suggesting flattened spatial dimensions

**Possible Causes:**
1. **ConvTranspose2d backward bug** - UNet uses `ConvTranspose2d` when `bilinear=false`
2. **BatchNorm2d backward issue** - Used extensively in UNet
3. **Dimension mismatch in DoubleConv** - Two conv layers with batch norm
4. **Pre-existing bug** - May have been masked because gradients weren't flowing before

---

## 🧪 Investigation Strategy

### Hypothesis 1: ConvTranspose2d Backward Bug

**Test:** Check if Conv Transpose2d gradients work in isolation
```cpp
// Simple test
Variable x(Tensor({1, 64, 32, 32}, DType::Float32, Device::cpu()), true);
auto layer = ConvTranspose2d(64, 128, 2, 2, 0);
auto y = layer->forward(x);
auto loss = sum(y);
loss.backward();
// Does x.grad() exist and have correct shape?
```

### Hypothesis 2: Dimension Mismatch in UNet Architecture

**Evidence:**
- Error shape: (512×64) @ (256×16384)
- UNet uses channels: 64, 128, 256, 512, 1024
- The 512 and 256 suggest mismatch between encoder and decoder

**Test:** Run with smaller input size to see if error persists

### Hypothesis 3: Issue Only with Full UNet

**Test:** Try simpler models:
- Single Up layer with skip connection
- DoubleConv → Down → Up sequence

---

## 📝 Key Learnings

### Pattern: Gradient-Aware vs Raw Tensor Operations

**Always use gradient-aware operations when working with Variables:**

| Raw Tensor Op | Gradient-Aware Op | Grad Function |
|---------------|-------------------|---------------|
| `ops::cat(tensors, dim)` | `tenzor::cat(variables, dim)` | `CatBackward` |
| `ops::slice(tensor, ...)` | `tenzor::slice(variable, ...)` | `SliceBackward` |
| `ops::interpolate(tensor, ...)` | *NOT IMPLEMENTED YET* | Need `InterpolateBackward` |

### Similar Issues Fixed Previously

From git history:
- **CatBackward** - Fixed ViT gradient flow (previous session)
- **SliceBackward** - Fixed Swin Transformer gradient flow
- **UpsampleBilinearBackward** - Fixed DeepLabV3Plus (this session)

### UNet-Specific Pattern

UNet's decoder uses **skip connections** which concatenate encoder features with upsampled decoder features:

```
Encoder (x1) ----skip---→ Concat with Decoder
                            ↓
                       DoubleConv
                            ↓
                         Output
```

This requires gradient-aware `cat()` to maintain the computation graph through skip connections.

---

## 🚧 Next Steps

### Immediate (Debug Dimension Error)

1. **Add debug logging** to ConvTranspose2d backward to see dimensions
2. **Create minimal reproduction** - Single Up layer test
3. **Check BatchNorm2d** backward for dimension handling
4. **Review DoubleConv** backward chain

### If ConvTranspose2d is Broken

5. **Check ConvTranspose2d implementation** in `/home/lee/Projects/Tenzor/src/nn/layers/conv.cpp`
6. **Review existing tests** for ConvTranspose2d gradients
7. **Implement ConvTranspose2dBackward** if missing

### Alternative Approach

8. **Test with bilinear=true** - This uses `ops::interpolate()` instead of ConvTranspose2d
9. **Compare with PyTorch** - Verify expected behavior
10. **Check other models** - See if ConvTranspose2d works elsewhere

---

## 🎯 Success Criteria

For UNet gradient flow to fully work:

- [ ] ✅ Autograd graph connected (DONE - using gradient-aware cat())
- [ ] ❌ Backward pass completes without errors
- [ ] ❌ `images.grad().has_value()` returns true
- [ ] ❌ Gradients have correct shapes
- [ ] ❌ Gradients are non-zero
- [ ] ❌ All UNet tests pass

---

## 📊 Related Work

### Other Gradient Flow Fixes Needed

1. **DeepLabV3Plus** - Uses `nn::upsample_bilinear()` (already fixed with UpsampleBilinearBackward)
2. **ops::interpolate()** - Need `InterpolateBackward` for models using bilinear upsampling
3. **YOLO** - Also uses `ops::interpolate()` in FPN

### Files Modified (This Session)

1. `/home/lee/Projects/Tenzor/include/tenzor/autograd/function.hpp` - Added `UpsampleBilinearBackward`
2. `/home/lee/Projects/Tenzor/src/autograd/function.cpp` - Implemented `UpsampleBilinearBackward::backward()`
3. `/home/lee/Projects/Tenzor/src/nn/layers/segmentation.cpp` - Modified `upsample_bilinear()` to use autograd
4. `/home/lee/Projects/Tenzor/src/models/unet.cpp` - Fixed concatenation to use gradient-aware `cat()`

---

## 💡 Code Quality Notes

### ✅ Strengths
- Follows established autograd patterns exactly
- Clean, minimal changes
- Proper use of existing CatBackward infrastructure
- No memory leaks (uses smart pointers)

### ⚠️ Limitations
- Dimension error needs investigation
- ConvTranspose2d backward may need fixes
- ops::interpolate() still needs gradient support

### 🐛 Known Issues
- **Critical:** Matmul dimension mismatch during backward
- Needs deeper investigation of ConvTranspose2d or UNet architecture

---

**Investigation Status:** 🔄 **IN PROGRESS**
**Confidence Level:** 🟢 HIGH (graph fix correct, dimension error is separate issue)
**Time Spent:** ~3 hours (analysis + implementation + debugging)

---

*Last Updated: October 20, 2025 - Gradient graph connected, investigating dimension error*
