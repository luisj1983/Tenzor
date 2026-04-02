# Sanitizer Testing Guide

Tenzor supports AddressSanitizer (ASan), ThreadSanitizer (TSan), and UndefinedBehaviorSanitizer (UBSan) for detecting memory errors, data races, and undefined behavior.

## Building with Sanitizers

ASan and TSan are **mutually exclusive** — you cannot enable both simultaneously.

### AddressSanitizer + UBSan (recommended for memory bugs)

```bash
cmake -B build-asan -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug \
  -DTENZOR_ENABLE_ASAN=ON \
  -DTENZOR_ENABLE_UBSAN=ON \
  -DTENZOR_BUILD_PYTHON=OFF \
  -DTENZOR_BUILD_TESTS=ON \
  -DTENZOR_BUILD_BENCHMARKS=OFF

ninja -C build-asan -j4
```

### ThreadSanitizer (for data races)

```bash
cmake -B build-tsan -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug \
  -DTENZOR_ENABLE_TSAN=ON \
  -DTENZOR_BUILD_PYTHON=OFF \
  -DTENZOR_BUILD_TESTS=ON \
  -DTENZOR_BUILD_BENCHMARKS=OFF

ninja -C build-tsan -j4
```

## Running Tests

### Environment Variables

```bash
# ASan options
export ASAN_OPTIONS="detect_leaks=1,halt_on_error=1,suppressions=$(pwd)/sanitizer_suppressions.txt"

# TSan options
export TSAN_OPTIONS="suppressions=$(pwd)/sanitizer_suppressions.txt,second_deadlock_stack=1"
```

### Quick Sanitizer Test Suites (~5 minutes each)

**ASan quick** — core correctness:
```bash
cd build-asan
ASAN_OPTIONS="detect_leaks=1,halt_on_error=1,suppressions=../sanitizer_suppressions.txt" \
  ctest -R "TensorBasicTest|AutogradTest|MemoryTest|IndexingTest" -E "cuda|vulkan" \
  -j1 --output-on-failure
```

**TSan quick** — thread safety:
```bash
cd build-tsan
TSAN_OPTIONS="suppressions=../sanitizer_suppressions.txt" \
  ctest -R "ThreadSafety|test_hooks|test_thread|Autograd" -E "cuda|vulkan" \
  -j1 --output-on-failure
```

### Full Sanitizer Run (~30 minutes)

```bash
cd build-asan
ASAN_OPTIONS="detect_leaks=1,halt_on_error=1,suppressions=../sanitizer_suppressions.txt" \
  ctest -E "cuda|vulkan|rocm|oneapi" -j1 --output-on-failure --timeout 300
```

## Suppressions

The `sanitizer_suppressions.txt` file in the project root contains known false positives from:
- **Intel TBB** — internal memory pool patterns
- **Intel MKL** — known benign races in thread pool
- **OpenMP** — runtime initialization races
- **GPU drivers** — driver-internal memory patterns

## Interpreting Results

### ASan Report Format

```
==12345==ERROR: AddressSanitizer: heap-buffer-overflow on address 0x...
READ of size 4 at 0x... thread T0
    #0 0x... in tenzor::Tensor::data_ptr() tensor.cpp:245
    #1 0x... in ...
```

Key information:
- **Error type**: heap-buffer-overflow, use-after-free, stack-buffer-overflow, etc.
- **Stack trace**: Shows exact code path
- **Thread info**: Which thread hit the error

### TSan Report Format

```
WARNING: ThreadSanitizer: data race (pid=12345)
  Write of size 8 at 0x... by thread T1:
    #0 ...
  Previous read of size 8 at 0x... by main thread:
    #0 ...
```

Key information:
- **Two stack traces**: One for each conflicting access
- **Access types**: Read vs Write
- Check if both accesses touch the same variable without synchronization

## Adding Suppressions

If you encounter a false positive (typically in third-party code), add to `sanitizer_suppressions.txt`:

```
# ASan suppressions
interceptor_via_fun:__tbb_malloc_safer_mmap

# TSan suppressions
race:libgomp  # OpenMP runtime
race:mkl_     # MKL internals
```

## Notes

- Sanitizer builds are 2-5x slower and use 2-3x more memory
- GPU-specific tests are typically excluded (ASan doesn't instrument GPU code)
- Always use `-j1` for ctest to avoid intermittent failures from resource contention
- Debug builds disable optimizations, so performance-related tests may timeout
