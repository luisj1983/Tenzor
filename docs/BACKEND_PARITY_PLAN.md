# Backend Parity Implementation Plan

**Date**: October 25, 2025
**Objective**: Achieve 100% backend parity - all operations work identically on CPU, CUDA, Vulkan, and OneAPI backends
**Principle**: No workarounds. Fix root causes properly.

## Current Problems

### 1. ❌ Tensor Slice Offset Bug (CRITICAL)
**Location**: All backend operations
**Problem**: Backend operations creating result tensors inherit slice offsets, causing garbage output
**Evidence**:
- Manual CPU workarounds in `box_iou()` and `box_area()`
- Comment: "Backend operations may still have issues with sliced result tensors"

**Impact**:
- Affects ALL sliced tensor operations across ALL backends
- Forces fallback to CPU for critical operations
- Masks real backend performance

### 2. ❌ OneAPI Backend Completely Broken
**Problem**: Runtime kernel loading fails
**Error**: "No kernel named [...] was found"
**Status**: 86 kernels compile successfully but can't execute
**Impact**: 0% OneAPI functionality despite architectural work

### 3. ❌ No Backend Parity Testing
**Problem**: Tests only run on CPU backend
**Impact**: Backend-specific bugs go undetected
**Example**: CUDA/Vulkan may have same slice bug as manual CPU workarounds suggest

## Implementation Plan

### Phase 1: Fix Core Slice Offset Bug ⚡ CRITICAL

#### Task 1.1: Identify Root Cause
**File to investigate**: Backend operation implementations
- CUDA: `src/backends/cuda/kernels/*.cu`
- Vulkan: `src/backends/vulkan/*.cpp`
- OneAPI: `src/backends/oneapi/kernels/*.cpp`

**Expected issue**: Result tensor creation code like:
```cpp
// BROKEN:
auto result = tensor_like(input);  // Inherits input's slice offset!

// SHOULD BE:
auto result = zeros(input.shape(), input.dtype(), input.device());  // Fresh tensor
```

#### Task 1.2: Fix All Backend Implementations
1. Search for all `tensor_like()` calls creating result tensors
2. Replace with proper tensor creation that doesn't inherit offsets
3. Ensure result tensors always start at offset 0

#### Task 1.3: Remove CPU Workarounds
**Files to fix**:
- `/home/lee/Projects/Tenzor/src/ops/detection.cpp:38-170`
  - Remove manual CPU `box_area()` implementation
  - Remove manual CPU `box_iou()` implementation
  - Let backend dispatch handle these operations

### Phase 2: Fix OneAPI Backend ⚡ HIGH PRIORITY

#### Task 2.1: Debug Kernel Loading
**Investigation steps**:
1. Check if kernels use functor pattern vs empty structs
2. Verify symbol visibility in shared library
3. Test with simple standalone SYCL program
4. Check SYCL 2025.2 documentation for kernel registration changes

#### Task 2.2: Fix Kernel Registration
**Likely solutions** (try in order):
1. **Functor Pattern**: Convert to proper functor classes
   ```cpp
   struct MatMulKernel {
       void operator()(sycl::id<2> idx) const {
           // Kernel body here
       }
   };
   ```

2. **Anonymous Lambdas**: Remove explicit kernel names
   ```cpp
   queue.parallel_for(range, [=](id idx) {
       // SYCL auto-generates kernel name
   });
   ```

3. **Symbol Export**: Add `-fvisibility=default` to CMake flags

#### Task 2.3: Verify OneAPI CPU Device
Ensure OneAPI backend works on CPU device (opencl:cpu) as fallback:
```bash
export ONEAPI_DEVICE_SELECTOR=opencl:cpu
./test_oneapi_basic
```

### Phase 3: Backend Test Matrix 🎯 INFRASTRUCTURE

#### Task 3.1: Create Backend Test Fixture
**New file**: `tests/backend_parity_fixture.hpp`

```cpp
class BackendParityTest : public ::testing::TestWithParam<std::string> {
protected:
    Device device;

    void SetUp() override {
        std::string backend = GetParam();
        if (backend == "cpu") {
            device = Device::cpu();
        } else if (backend == "cuda") {
            if (!has_cuda()) GTEST_SKIP();
            device = Device::cuda(0);
        } else if (backend == "vulkan") {
            if (!has_vulkan()) GTEST_SKIP();
            device = Device::vulkan(0);
        } else if (backend == "oneapi") {
            if (!has_oneapi()) GTEST_SKIP();
            device = Device::oneapi(0);
        }
    }
};

INSTANTIATE_TEST_SUITE_P(
    AllBackends,
    BackendParityTest,
    ::testing::Values("cpu", "cuda", "vulkan", "oneapi")
);
```

#### Task 3.2: Convert Existing Tests to Parameterized
**Priority test files to convert**:
1. `tests/unit/test_tensor_ops.cpp` - Basic operations
2. `tests/unit/test_math_ops.cpp` - Math operations
3. `tests/test_ciou_loss.cpp` - Detection operations (uses box_iou)
4. `tests/unit/test_conv2d.cpp` - Convolution operations
5. `tests/unit/test_matmul.cpp` - Matrix operations

**Pattern**:
```cpp
// BEFORE:
TEST(TensorOpsTest, Addition) {
    auto a = ones({10, 10}, DType::Float32, Device::cpu());
    // ...
}

// AFTER:
TEST_P(BackendParityTest, Addition) {
    auto a = ones({10, 10}, DType::Float32, device);  // Use parameterized device
    // ...
}
```

#### Task 3.3: Backend Comparison Tests
Create explicit comparison tests:
```cpp
TEST(BackendParityTest, ResultsMatch) {
    auto input_cpu = randn({100, 100}, DType::Float32, Device::cpu());

    for (auto backend : {"cuda", "vulkan", "oneapi"}) {
        auto input_gpu = input_cpu.to(get_device(backend));
        auto result_gpu = matmul(input_gpu, input_gpu);
        auto result_cpu = matmul(input_cpu, input_cpu);

        EXPECT_TENSOR_NEAR(result_gpu.to(Device::cpu()), result_cpu, 1e-5);
    }
}
```

### Phase 4: Comprehensive Backend Testing 🧪

#### Task 4.1: Run Full Test Suite Per Backend
Create test runner script:

**File**: `scripts/test_all_backends.sh`
```bash
#!/bin/bash

BACKENDS=("cpu" "cuda" "vulkan" "oneapi")
RESULTS_DIR="test_results"

mkdir -p $RESULTS_DIR

for backend in "${BACKENDS[@]}"; do
    echo "Testing backend: $backend"

    # Set environment for backend
    export TENZOR_DEFAULT_DEVICE=$backend

    # Run all test suites
    ./bin/tenzor_unit_tests \
        --gtest_output=xml:$RESULTS_DIR/${backend}_unit.xml \
        2>&1 | tee $RESULTS_DIR/${backend}_unit.log

    ./bin/tenzor_integration_tests \
        --gtest_output=xml:$RESULTS_DIR/${backend}_integration.xml \
        2>&1 | tee $RESULTS_DIR/${backend}_integration.log
done

# Generate parity report
python3 scripts/generate_parity_report.py $RESULTS_DIR
```

#### Task 4.2: Parity Report Generator
**File**: `scripts/generate_parity_report.py`

Generates markdown report showing:
- Tests passing on all backends ✅
- Tests passing on some backends ⚠️
- Tests failing on all backends ❌
- Backend-specific failures with details

### Phase 5: Performance Verification 🚀

#### Task 5.1: GPU Acceleration Verification
Verify operations actually run on GPU:

```cpp
TEST(PerformanceTest, GPUAcceleration) {
    auto input = randn({1000, 1000}, DType::Float32, Device::cuda());

    auto start = std::chrono::high_resolution_clock::now();
    auto result = matmul(input, input);
    cudaDeviceSynchronize();  // Ensure GPU work completes
    auto end = std::chrono::high_resolution_clock::now();

    auto gpu_time = std::chrono::duration<double>(end - start).count();

    // CPU version for comparison
    auto input_cpu = input.to(Device::cpu());
    start = std::chrono::high_resolution_clock::now();
    auto result_cpu = matmul(input_cpu, input_cpu);
    end = std::chrono::high_resolution_clock::now();

    auto cpu_time = std::chrono::duration<double>(end - start).count();

    // GPU should be significantly faster for large matrices
    EXPECT_LT(gpu_time, cpu_time * 0.5);
}
```

#### Task 5.2: Fix Mask R-CNN Performance
Once box_iou uses GPU:
- Expect 100-1000x speedup for 2000×5 box comparisons
- Tests should complete in <5 seconds instead of 600+ seconds

## Success Criteria

### ✅ Phase 1 Complete When:
- [ ] No manual CPU workarounds in codebase
- [ ] All backend operations handle sliced tensors correctly
- [ ] CIoU tests pass on CUDA backend without manual implementations

### ✅ Phase 2 Complete When:
- [ ] OneAPI backend loads successfully (no "Failed to load" warning)
- [ ] OneAPI kernels execute without "kernel not found" errors
- [ ] At least simple operations (add, matmul) work on OneAPI CPU device

### ✅ Phase 3 Complete When:
- [ ] Backend parity test infrastructure exists
- [ ] At least 50 tests converted to parameterized backend tests
- [ ] Test matrix runs automatically in CI

### ✅ Phase 4 Complete When:
- [ ] All unit tests run on all 4 backends
- [ ] Parity report shows 100% pass rate on CPU, CUDA, Vulkan
- [ ] Parity report shows 100% pass rate on OneAPI (or clear tracking of failures)

### ✅ Phase 5 Complete When:
- [ ] GPU operations verified to actually use GPU (not CPU fallback)
- [ ] Mask R-CNN tests complete in <10 seconds on CUDA
- [ ] Performance tests show expected GPU speedups (>10x for large ops)

## Estimated Timeline

- **Phase 1** (Fix slice bug): 4-6 hours
- **Phase 2** (Fix OneAPI): 6-8 hours
- **Phase 3** (Test infrastructure): 3-4 hours
- **Phase 4** (Run all tests): 2-3 hours
- **Phase 5** (Performance verification): 2-3 hours

**Total**: 17-24 hours of focused work

## Priority Order

1. **CRITICAL**: Phase 1 - Fix slice offset bug (blocks everything)
2. **HIGH**: Phase 3.1 - Create test infrastructure (enables verification)
3. **HIGH**: Phase 2 - Fix OneAPI backend (major feature broken)
4. **MEDIUM**: Phase 3.2-3.3 - Convert tests to matrix
5. **MEDIUM**: Phase 4 - Run comprehensive tests
6. **LOW**: Phase 5 - Performance verification (nice to have)

## Notes

- **No Workarounds**: Every fix must address root cause
- **Test Everything**: If it's not tested on all backends, it's broken on some
- **GPU is Required**: A tensor library without GPU acceleration is not world-class
- **Parity is Non-Negotiable**: One codebase, identical behavior on all backends

---

**Status**: 📋 **PLANNING COMPLETE** - Ready to begin implementation
**Next Step**: Start Phase 1 - Fix tensor slice offset bug
