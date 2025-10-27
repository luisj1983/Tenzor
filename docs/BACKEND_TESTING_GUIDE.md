# Backend Parity Testing Guide
**Date**: October 25, 2025

## Overview

This guide shows how to ensure all operations work identically across all backends (CPU, CUDA, Vulkan, OneAPI). There are two approaches:

## Approach 1: Parameterized Tests (Recommended for New Tests)

### Setup

Include the backend test fixture:

```cpp
#include "backend_test_fixture.hpp"

using namespace tenzor::testing;
```

### Writing a Multi-Backend Test

```cpp
class MyOperationTest : public BackendTest {};

TEST_P(MyOperationTest, Addition) {
    // The 'device' member is automatically set based on test parameter
    auto a = ones({10, 10}, DType::Float32, device);
    auto b = ones({10, 10}, DType::Float32, device);

    auto c = a + b;

    // Verify result
    auto c_cpu = c.to(Device::cpu());
    auto* data = c_cpu.data<float>();
    for (int64_t i = 0; i < c_cpu.numel(); ++i) {
        EXPECT_NEAR(data[i], 2.0f, 1e-5f);
    }
}

// This runs the test on ALL backends
INSTANTIATE_BACKEND_TESTS(MyOperationTest);
```

### Running the Tests

```bash
cd bin
./test_backend_parity_example

# Run only CPU tests
./test_backend_parity_example --gtest_filter="*/cpu"

# Run only CUDA tests
./test_backend_parity_example --gtest_filter="*/cuda"

# Run only specific test on all backends
./test_backend_parity_example --gtest_filter="*.Addition*"
```

### Example Output

```
[==========] Running 16 tests from 4 test suites.
[----------] 4 tests from MathOpsBackendTest/cpu
[ RUN      ] MathOpsBackendTest/cpu.Addition
[       OK ] MathOpsBackendTest/cpu.Addition (0 ms)
[ RUN      ] MathOpsBackendTest/cpu.Subtraction
[       OK ] MathOpsBackendTest/cpu.Subtraction (0 ms)
...
[----------] 4 tests from MathOpsBackendTest/cuda
[ RUN      ] MathOpsBackendTest/cuda.Addition
[       OK ] MathOpsBackendTest/cuda.Addition (15 ms)
...
```

## Approach 2: Test Runner Script (For Existing Tests)

### Running All Tests on All Backends

```bash
cd scripts
./test_all_backends.sh
```

This script:
1. Runs each test suite on CPU, CUDA, Vulkan, and OneAPI
2. Captures output and XML results for each backend
3. Generates a parity report showing which tests pass on which backends

### Example Report

```markdown
# Backend Parity Test Report

| Test Suite              | CPU | CUDA | Vulkan | OneAPI |
|------------------------|-----|------|--------|--------|
| tenzor_unit_tests      | ✅  | ✅   | ✅     | ❌     |
| test_ciou_loss         | ✅  | ✅   | ✅     | ⏭️     |
| test_slice_parity      | ✅  | ✅   | ✅     | ⏭️     |

## Backend Parity Analysis

### Full Parity (all backends pass)
- ✅ tenzor_unit_tests (3/4 backends)
- ✅ test_ciou_loss

### Partial Parity
- ⚠️ test_complex_ops (2/4 backends passing)
```

## Approach 3: Helper Functions for Manual Testing

The fixture provides helpers:

```cpp
TEST_P(MyTest, CompareAcrossBackends) {
    auto a = ones({10, 10}, DType::Float32, device);
    auto b = ones({10, 10}, DType::Float32, device);

    auto result = a + b;

    // Compare with expected CPU result
    auto expected = ones({10, 10}, DType::Float32, Device::cpu()) * 2.0f;

    // Helper automatically moves both to CPU and compares
    expectTensorNear(result, expected, 1e-5f);
}
```

## Converting Existing Tests

### Before (CPU-only)

```cpp
TEST(TensorOpsTest, Addition) {
    auto a = ones({10, 10}, DType::Float32, Device::cpu());
    auto b = ones({10, 10}, DType::Float32, Device::cpu());
    auto c = a + b;
    // ... verify ...
}
```

### After (Multi-backend)

```cpp
class TensorOpsTest : public BackendTest {};

TEST_P(TensorOpsTest, Addition) {
    auto a = ones({10, 10}, DType::Float32, device);  // Use 'device' from fixture
    auto b = ones({10, 10}, DType::Float32, device);
    auto c = a + b;
    // ... same verification code ...
}

INSTANTIATE_BACKEND_TESTS(TensorOpsTest);
```

### Changes Required
1. Change `class TensorOpsTest : public ::testing::Test` → `class TensorOpsTest : public BackendTest`
2. Change `TEST(...)` → `TEST_P(...)`
3. Replace `Device::cpu()` with `device`
4. Add `INSTANTIATE_BACKEND_TESTS(TensorOpsTest);` at the end

## Best Practices

### 1. Always Test Data Correctness on CPU

```cpp
TEST_P(MyTest, Operation) {
    auto result = some_operation(device);

    // Move to CPU for verification
    auto result_cpu = result.to(Device::cpu());
    auto* data = result_cpu.data<float>();

    // Verify values
    for (int64_t i = 0; i < result_cpu.numel(); ++i) {
        EXPECT_NEAR(data[i], expected[i], 1e-5f);
    }
}
```

### 2. Use Appropriate Tolerances

```cpp
// For float32
EXPECT_NEAR(actual, expected, 1e-5f);

// For float16 (less precision)
EXPECT_NEAR(actual, expected, 1e-3f);

// For integer operations
EXPECT_EQ(actual, expected);
```

### 3. Skip Unavailable Backends Gracefully

The fixture automatically skips tests if a backend isn't available:

```cpp
TEST_P(MyTest, Operation) {
    // If device is unavailable, test is automatically skipped
    // You don't need to write skip logic
    auto result = operation(device);
    // ...
}
```

### 4. Test Both Contiguous and Sliced Tensors

```cpp
TEST_P(MyTest, SlicedOperation) {
    auto tensor = zeros({10, 10}, DType::Float32, device);

    // Test on contiguous tensor
    auto result1 = operation(tensor);

    // Test on sliced tensor (non-contiguous)
    auto slice = tensor.slice(0, 2, 8);
    auto result2 = operation(slice);

    // Both should produce same results
    expectTensorNear(result1.slice(0, 2, 8), result2, 1e-5f);
}
```

## Test Organization

```
tests/
├── backend_test_fixture.hpp      # Base fixture for multi-backend tests
├── test_backend_parity_example.cpp   # Example tests
├── unit/
│   ├── test_tensor_ops.cpp       # Convert to use BackendTest
│   ├── test_math_ops.cpp         # Convert to use BackendTest
│   └── test_reduction_ops.cpp    # Convert to use BackendTest
└── integration/
    └── test_models.cpp           # End-to-end tests on multiple backends
```

## Continuous Integration

Add to your CI pipeline:

```yaml
- name: Backend Parity Tests
  run: |
    cd bin
    ./test_backend_parity_example
    ../scripts/test_all_backends.sh

- name: Upload Parity Report
  uses: actions/upload-artifact@v2
  with:
    name: backend-parity-report
    path: test_results/backend_parity/parity_report.md
```

## Debugging Backend-Specific Failures

### Enable Verbose Output

```bash
./test_backend_parity_example --gtest_filter="*/cuda" --verbose
```

### Compare CPU vs GPU Output

```cpp
TEST_P(MyTest, DebugDifference) {
    // Run on current device
    auto result_device = operation(device);
    auto result_cpu_copy = result_device.to(Device::cpu());

    // Run same operation on CPU for reference
    auto input_cpu = input.to(Device::cpu());
    auto result_cpu_ref = operation(Device::cpu());

    // Compare element by element
    auto* device_data = result_cpu_copy.data<float>();
    auto* cpu_data = result_cpu_ref.data<float>();

    for (int64_t i = 0; i < result_cpu_copy.numel(); ++i) {
        if (std::abs(device_data[i] - cpu_data[i]) > 1e-5f) {
            std::cout << "Mismatch at index " << i
                      << ": device=" << device_data[i]
                      << " cpu=" << cpu_data[i] << "\n";
        }
    }
}
```

## Performance Comparison

```cpp
TEST_P(MyTest, Performance) {
    auto large_tensor = randn({1000, 1000}, DType::Float32, device);

    auto start = std::chrono::high_resolution_clock::now();
    auto result = operation(large_tensor);

    // Synchronize if needed (GPU operations are async)
    if (device.type == Device::Type::CUDA) {
        cudaDeviceSynchronize();
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << "Operation took " << duration.count()
              << "ms on " << device.to_string() << "\n";
}
```

## Summary

1. **For new tests**: Use `BackendTest` fixture for automatic multi-backend testing
2. **For existing tests**: Use `test_all_backends.sh` script to run on all backends
3. **For debugging**: Use helper functions and verbose output
4. **For CI/CD**: Integrate both approaches to ensure backend parity

This infrastructure ensures that Tenzor maintains **true backend parity** - the same operation produces the same results on any backend.
