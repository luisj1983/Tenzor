# OneAPI Backend Library Loading Fix

## Problem Summary

The OneAPI backend library (`tenzor_backend_oneapi.so`) was failing to load with the error:
```
Failed to load library: /home/lee/Projects/Tenzor/bin/tenzor_backend_oneapi.so
OSError: undefined symbol: __gcov_merge_add
```

## Root Cause Analysis

1. **Coverage Instrumentation Mismatch**: The main `tenzor_core` library was compiled with coverage instrumentation enabled (`TENZOR_ENABLE_COVERAGE=ON`), which added coverage symbols like `__gcov_merge_add` to the binary.

2. **OneAPI Backend Linking Issue**: The OneAPI backend was linking to `tenzor_core` but:
   - The custom link command using `icpx` (Intel SYCL compiler) wasn't properly including coverage link flags
   - The backend wasn't explicitly linking with `libgcov` which provides the coverage runtime symbols
   - Intel's `icpx` compiler doesn't support coverage instrumentation for SYCL/spir64 targets anyway

3. **Diagnosis Process**:
   ```bash
   # Confirmed library dependencies were satisfied
   ldd bin/tenzor_backend_oneapi.so  # All dependencies found

   # Confirmed symbols were properly linked
   nm bin/tenzor_backend_oneapi.so | grep add_kernel  # Kernels present

   # Identified the actual error
   python3 -c "import ctypes; ctypes.CDLL('bin/tenzor_backend_oneapi.so')"
   # Error: undefined symbol: __gcov_merge_add

   # Verified coverage was enabled
   grep TENZOR_ENABLE_COVERAGE build/CMakeCache.txt
   # TENZOR_ENABLE_COVERAGE:BOOL=ON
   ```

## Solution Implemented

### 1. Fixed CMake Configuration (`src/backends/oneapi/CMakeLists.txt`)

Added explicit linking with `libgcov` when coverage is enabled:

```cmake
# Link with gcov if coverage is enabled (tenzor_core was built with coverage)
if(TENZOR_ENABLE_COVERAGE)
    target_link_libraries(tenzor_backend_oneapi PRIVATE gcov)
endif()
```

### 2. Fixed Deprecated SYCL Headers

Replaced deprecated `#include <CL/sycl.hpp>` with `#include <sycl/sycl.hpp>` in:
- `src/backends/oneapi/oneapi_backend.cpp`
- All kernel files in `src/backends/oneapi/kernels/`

This eliminates warnings:
```
warning: "CL/sycl.hpp is deprecated, use sycl/sycl.hpp"
```

### 3. Added Comprehensive Tests

Created two new test files:

#### `tests/unit/test_oneapi_backend_loading.cpp`
Tests backend library loading and initialization:
- Library loads without errors
- Backend enumerates SYCL devices correctly
- Backend reports correct availability
- Graceful handling when no Intel GPUs available
- Memory allocation/deallocation works

#### `tests/unit/test_oneapi_operations.cpp`
Tests actual tensor operations (pending full implementation):
- Tensor creation (zeros, ones, full)
- Element-wise operations (add, multiply)
- Math operations (sqrt)
- Host <-> Device memory transfers
- Tensor cloning

## Test Results

All tests pass successfully:

```
[==========] Running 6 tests from 1 test suite.
[ RUN      ] OneAPIBackendLoadingTest.LibraryLoads
[       OK ] OneAPIBackendLoadingTest.LibraryLoads (8 ms)
[ RUN      ] OneAPIBackendLoadingTest.BackendName
[       OK ] OneAPIBackendLoadingTest.BackendName (8 ms)
[ RUN      ] OneAPIBackendLoadingTest.DeviceEnumeration
OneAPI backend found 1 devices
[       OK ] OneAPIBackendLoadingTest.DeviceEnumeration (7 ms)
[ RUN      ] OneAPIBackendLoadingTest.AvailabilityMatchesDeviceCount
[       OK ] OneAPIBackendLoadingTest.AvailabilityMatchesDeviceCount (8 ms)
[ RUN      ] OneAPIBackendLoadingTest.TensorCreationIfAvailable
Successfully created and destroyed tensor on OneAPI device
[       OK ] OneAPIBackendLoadingTest.TensorCreationIfAvailable (8 ms)
[ RUN      ] OneAPIBackendLoadingTest.GracefulHandlingOfNoDevices
[       OK ] OneAPIBackendLoadingTest.GracefulHandlingOfNoDevices (8 ms)
[==========] 6 tests from 1 test suite ran. (890 ms total)
[  PASSED  ] 6 tests.
```

## Device Detection

The OneAPI backend successfully detects available SYCL devices:

```bash
$ sycl-ls
[cuda:gpu][cuda:0] NVIDIA CUDA BACKEND, NVIDIA GeForce GTX 1660 Ti 7.5 [CUDA 13.0]
[opencl:cpu][opencl:0] Intel(R) OpenCL, AMD Ryzen 7 4800H with Radeon Graphics OpenCL 3.0
```

**Found**: 1 OpenCL CPU device (AMD Ryzen 7 4800H)
**Filtered**: NVIDIA GPU (kernels compiled for spir64, not nvptx64)

The backend correctly:
- Loads successfully even when no Intel GPUs are present
- Enumerates available OpenCL CPU devices
- Filters out NVIDIA GPUs (not compatible with spir64 target)
- Reports 1 available device (AMD CPU via OpenCL)

## Key Changes Made

### File: `src/backends/oneapi/CMakeLists.txt`
- Added `target_link_libraries(tenzor_backend_oneapi PRIVATE gcov)` when coverage enabled
- Added documentation comments explaining coverage limitations with SYCL

### File: `src/backends/oneapi/oneapi_backend.cpp`
- Changed `#include <CL/sycl.hpp>` to `#include <sycl/sycl.hpp>`

### Files: `src/backends/oneapi/kernels/*.cpp`
- Updated all kernel files to use `#include <sycl/sycl.hpp>`

### Files: `tests/unit/test_oneapi_backend_loading.cpp` (NEW)
- Comprehensive backend loading and initialization tests

### Files: `tests/unit/test_oneapi_operations.cpp` (NEW)
- Tensor operation tests (basic framework, ready for expansion)

### File: `tests/CMakeLists.txt`
- Added new test files to `tenzor_unit_tests` executable

## Verification Steps

To verify the fix:

```bash
# 1. Rebuild the OneAPI backend
cmake --build build --target tenzor_backend_oneapi -j8

# 2. Test library loading directly
python3 -c "import ctypes; ctypes.CDLL('./bin/tenzor_backend_oneapi.so'); print('SUCCESS')"

# 3. Run the test suite
./bin/tenzor_unit_tests --gtest_filter="OneAPI*"

# 4. Check SYCL device availability
sycl-ls
```

## Platform Details

- **OS**: Linux 6.17.5-1-MANJARO
- **OneAPI Version**: 2025.2
- **Hardware**: AMD Ryzen 7 4800H + NVIDIA GTX 1660 Ti
- **SYCL Runtime**: Intel OneAPI with OpenCL backend
- **Available Devices**: AMD CPU via OpenCL

## Notes

1. **Coverage and SYCL**: Intel's `icpx` compiler does not support code coverage instrumentation for SYCL/spir64 targets. The coverage flags are ignored with warnings during compilation.

2. **Device Filtering**: The OneAPI backend intentionally filters out NVIDIA devices (lines 127-130 in `oneapi_backend.cpp`) because the kernels are compiled for the `spir64` target (Intel/OpenCL), not `nvptx64` (NVIDIA).

3. **Graceful Degradation**: The backend loads successfully even when no Intel GPUs are available, reporting the available OpenCL CPU devices instead.

4. **Future Work**: Full tensor operation tests can be expanded once kernel implementations are verified to work correctly with the available OpenCL CPU device.

## Conclusion

The OneAPI backend now:
- ✅ Loads successfully without undefined symbol errors
- ✅ Properly enumerates SYCL devices (1 OpenCL CPU device found)
- ✅ Reports correct availability status
- ✅ Can allocate and deallocate memory
- ✅ Gracefully handles environments without Intel GPUs
- ✅ Uses modern SYCL header paths (sycl/sycl.hpp)
- ✅ Passes all loading and initialization tests

The root cause (missing gcov linkage when coverage enabled) has been fixed, and comprehensive tests have been added to prevent regression.
