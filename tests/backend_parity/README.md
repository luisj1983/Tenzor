# Backend Parity Test Suite

Comprehensive test suite ensuring all Tenzor backends (CPU, CUDA, OneAPI, Vulkan) produce identical numerical results.

## Quick Start

```bash
ninja -C build test_parity_full
```

## Test Modules

| Module | Tests | Purpose |
|--------|-------|---------|
| `test_operation_parity` | 50+ | Math operations, reductions, broadcasting |
| `test_nn_parity` | 40+ | Neural network operations (conv, pool, norm, etc.) |
| `test_gradient_parity` | 15+ | Gradient verification |
| `test_dtype_parity` | 25+ | Data type handling (Float32/64, Int32/64) |
| `test_backend_stress` | 20+ | Stress tests (large tensors, deep graphs) |
| `test_numerical_stability` | 35+ | Edge cases (NaN, Inf, overflow) |
| `test_performance_regression` | 15+ | Performance baselines |

**Total**: 300+ tests

## Running Tests

### All Tests:
```bash
ninja -C build test_parity_full
```

### Quick Tests (2-3 min):
```bash
ninja -C build test_parity_quick
```

### Individual Suites:
```bash
./test_operation_parity
./test_nn_parity
./test_gradient_parity
```

### With CTest:
```bash
ctest -L backend_parity --output-on-failure
```

## Test Groups

```bash
ninja -C build test_parity_quick       # Basic operations only
ninja -C build test_parity_full        # All tests
ninja -C build test_parity_stress      # Stress tests only
ninja -C build test_parity_performance # Performance tests only
ninja -C build test_parity_gradients   # Gradient tests only
```

(These are custom CMake targets; `make` also works if you configured with the
Makefiles generator instead of Ninja.)

## Supported Backends

- ✅ **CPU**: Reference implementation (all tests)
- ✅ **CUDA**: NVIDIA GPU backend
- ✅ **OneAPI**: Intel GPU/CPU backend
- ✅ **Vulkan**: Cross-platform GPU backend
- ✅ **ROCm**: AMD GPU backend — included in `STANDARD_BACKENDS`. Set
  `TENZOR_SKIP_BACKENDS=rocm` on hosts where the driver is unstable.

Tests automatically skip unavailable backends.

### Opting out of backends at runtime

Set `TENZOR_SKIP_BACKENDS` to a comma-separated list of backend names
(e.g. `TENZOR_SKIP_BACKENDS=rocm,vulkan`) to skip those backends without
recompiling. Every parity fixture (`BackendTest`, `MultiBackendDTypeTest`,
the `SKIP_IF_NO_*` macros, and the Python `device` fixture in
`tests/python/conftest.py`) honors the same variable, so one setting
silences a backend across the whole suite.

## Documentation

- **Full Documentation**: `docs/BACKEND_PARITY_TESTS_COMPLETE.md`
- **Implementation Summary**: `docs/BACKEND_PARITY_IMPLEMENTATION_SUMMARY.md`

## Test Coverage

### Operations Tested:
- **Math**: add, sub, mul, div, matmul, exp, log, sin, cos, etc. (40+)
- **Reduction**: sum, mean, max, min, argmax, argmin, etc. (15+)
- **NN Operations**: conv2d, pooling, batchnorm, activations, losses (30+)
- **Gradients**: Forward/backward pass verification (15+)

### Scenarios Tested:
- Different tensor sizes (8x8 to 2048x2048)
- Various data types (Float32, Float64, Int32, Int64)
- Edge cases (NaN, Inf, very small/large values)
- Large tensors (>1GB)
- Deep computation graphs (100+ layers)

## Tolerance Guidelines

| Operation | Float32 rtol | Float32 atol |
|-----------|-------------|-------------|
| Element-wise | 1e-6 | 1e-8 |
| MatMul (small) | 1e-5 | 1e-7 |
| MatMul (large) | 1e-4 | 1e-6 |
| Convolution | 1e-4 | 1e-6 |
| Activation | 1e-6 | 1e-8 |

## Example Output

```
=== MatMul 1024x1024 Performance ===
cpu: 245.123 ms
cuda:0: 12.456 ms
oneapi:0: 18.789 ms
vulkan:0: 22.345 ms

GPU Speedup: 19.7x
```

## Debugging Failed Tests

Tests print detailed information on failure:

```
Test Parity failed:
  Reference backend: cpu
  Test backend: cuda:0
  Max absolute difference: 1.234e-05
  Tolerance: rtol=1e-05, atol=1e-07
```

To debug:
```bash
./test_operation_parity --gtest_filter="*MatMul*" --gtest_verbose
```

## Adding New Tests

1. Choose appropriate test file (operation, nn, gradient, etc.)
2. Add test case using `TEST()` macro
3. Use `test_operation_parity()` helper for automatic cross-backend testing
4. Set appropriate tolerances based on operation complexity

Example:
```cpp
TEST(MathOperationParity, NewOperation) {
    REQUIRE_MULTI_BACKEND_OR_SKIP("NewOperation parity");
    auto backends = get_available_backends();

    auto input = randn({32, 32}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return my_new_operation(inputs[0]);
    }, {input}, 1e-5f, 1e-7f, "New Operation");
}
```

`REQUIRE_MULTI_BACKEND_OR_SKIP(reason)` normally skips the test when fewer than 2
backends are available, but escalates to a hard `FAIL()` when the env var
`TENZOR_REQUIRE_MULTI_BACKEND=1` is set. CI jobs that build a GPU backend should
export that variable so a silently broken backend cannot pass as "skipped".

## Integration

To integrate into main build system:

```cmake
# In tests/CMakeLists.txt
add_subdirectory(backend_parity)
```

## Requirements

- CMake 3.18+
- Google Test
- Tenzor library
- At least 2 available backends for meaningful tests

## License

Same as main Tenzor project.

## Status

✅ **IMPLEMENTATION COMPLETE** - Ready for integration and testing

**Version**: 1.0.0
**Date**: October 24, 2025
**Total Lines**: ~4,200
**Total Tests**: 300+

---

## Golden-tensor fallback (single-backend hosts)

Parity tests historically skipped when only one backend was available. The
golden-tensor path lets CI produce signal on CPU-only jobs by comparing
against a pre-recorded reference tensor.

Binary format, helpers, and the fingerprint scheme are documented in
`golden_util.hpp`. Directory: `tests/backend_parity/golden/`.

### Recording goldens

On a multi-backend host (e.g. CPU + CUDA, CPU + ROCm):

```sh
TENZOR_RECORD_GOLDENS=1 ctest -R "backend_parity" -j1 -V
```

New `.gold` files appear in `tests/backend_parity/golden/`. Review the diff
and commit them alongside any parity-test change that affected inputs.

### Replaying goldens

Nothing required — running `ctest` without `TENZOR_RECORD_GOLDENS` consults
the committed goldens whenever only one backend is available.

### Enforcement

- `TENZOR_REQUIRE_MULTI_BACKEND=1` turns "no golden on CPU-only host" from a
  skip into a hard failure. Use this in CI jobs where a GPU is supposed to
  be present.
- `TENZOR_SKIP_BACKENDS=cuda,rocm,...` excludes named backends from the
  available set without a rebuild.

### When to re-record

If a parity test's inputs change (different shape, seed, or dtype), its
fingerprint changes, the old golden becomes orphaned, and the new test
path falls back to the skip arm. Re-record on a multi-backend host.

---

## Skip-reason taxonomy

The `SkipReason` enum in `../multi_backend_dtype_fixture.hpp` is the
machine-readable vocabulary for every skip that isn't pure backend-availability.
`scripts/count_skips.py` tallies them and ships a `--max-untagged N` CI gate.

Use `SKIP_WITH_REASON(::tenzor::testing::SkipReason::Kind, "detail")` instead
of raw `GTEST_SKIP()`.

Categories:

- `BackendUnavailable` — device absent.
- `BackendExcludedByEnv` — `TENZOR_SKIP_BACKENDS`.
- `NumericalDivergence` — algorithm exceeds FP16 precision.
- `DtypeUnsupportedOnBackend` — kernel doesn't register the dtype.
- `ComplexFP16Unrepresentable` — no Float16 complex type.
- `GradcheckFDPrecision` — finite-difference noise dominates at FP16.
- `KernelNotImplemented` — feature is genuinely TODO.
- `RequiresMultiGPU` — multi-device distributed coverage.
- `KnownBug` — tracked by issue number.

---

## Adding a new parity test

1. Drop a `.cpp` into `tests/backend_parity/` using `BackendTest` (single-dtype)
   or `MultiBackendDTypeTest` (per-dtype).
2. Register it with `add_parity_test(<name> <file>.cpp)` in `CMakeLists.txt`.
3. Use `test_operation_parity` / `test_operation_parity_single` /
   `test_operation_parity_cross_backend` — the helpers route goldens automatically.
4. Run once under `TENZOR_RECORD_GOLDENS=1` on a multi-backend host to seed
   the golden directory, then commit.
