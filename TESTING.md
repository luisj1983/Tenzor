# Testing Contract

This document pins the invariants every contributor must maintain when adding
ops, layers, or tests to Tenzor. It exists to keep the suite honest over time —
silent skips, EXPECT_NO_THROW-only backward tests, and hardcoded single-backend
assumptions have caused real bugs to ship.

**See also:** `CLAUDE.md` (project guidance), `tests/backend_parity/README.md`
(parity-test mechanics), `tests/autograd/GRADCHECK_COVERAGE_ANALYSIS.md`
(gradcheck coverage status).

---

## Runtime constraints

- **Never run the full ctest suite** — it takes 15+ hours. Scope every
  `ctest -R <pattern>` to the code you touched.
- Always use `ctest -j1`. The AMD driver is fragile under parallelism.
- Only one shell should run tests at a time. Finish one invocation before
  starting another.
- On machines with ≥ 2 backends, set `TENZOR_REQUIRE_MULTI_BACKEND=1`. This
  flips silent "backend unavailable" skips into hard failures so regressions
  surface immediately.
- Use `TENZOR_SKIP_BACKENDS=<csv>` (e.g. `TENZOR_SKIP_BACKENDS=rocm`) to
  exclude a specific backend without editing code. It wins over
  `TENZOR_REQUIRE_MULTI_BACKEND`.

---

## New op requirements

Every new `OpId::Xxx` must land with:

1. **Kernel on every backend** (CPU, CUDA, ROCm, Vulkan, OneAPI) — or an
   explicit entry in the "excluded ops" list with a documented rationale in
   the OpId header. "Implemented only on CUDA for now" breaks the library's
   portability promise.
2. **Registration** in each backend's `kernel_registry.cpp`.
3. **Parity test** in `tests/backend_parity/` covering at least one
   canonical shape per dtype the op claims to support. Use
   `MultiBackendDTypeTest` or `BackendTest` — never a custom fixture.
4. **`required_ops.hpp` entry** — grow the floor when a new parity test
   lands. `test_kernel_completeness.cpp` enforces that every entry in the
   floor is registered on every backend.
5. **Gradcheck** — if the op defines a `backward`, add a case to
   `tests/autograd/test_gradcheck_missing.cpp` or an appropriate sibling
   exercising Float32 **and** Float64. Float64 is the gold standard: it
   catches Float32-accumulator-in-Float64-codepath bugs that Float32 masks.
6. **Multibackend gradcheck** — register the op in
   `test_gradcheck_multibackend.cpp` so GPU backward paths are exercised.
   Backend-specific bugs (stride-ignoring kernels, FP16 rounding, async
   misordering) only surface when gradcheck runs on that backend.
7. **Stride-variant parity** when the op's gradient is shape- or
   stride-sensitive (Conv, Pool, Norm, LinAlg, Sparse). See
   `tests/backend_parity/test_stride_parity.cpp` for the 5-variant pattern
   (contiguous / transposed / permuted / narrowed / unsqueezed).

---

## New layer requirements

Every new layer in `include/tenzor/nn/` must land with:

1. **Forward + backward multi-backend multi-dtype test** using
   `MultiBackendDTypeTest`. The test must run on all 5 backends × Float32,
   Float64, Float16 (+ BFloat16 when the build has
   `TENZOR_TEST_BFLOAT16=ON`).
2. **Gradient-flow assertion that actually verifies gradients** — never
   `EXPECT_NO_THROW({ loss.backward(); })` alone. Required minimum:
   - `ASSERT_TRUE(input.has_grad())`
   - `EXPECT_EQ(input.grad()->numel(), input.tensor().numel())`
   - `EXPECT_GT(max(abs(input.grad()->to(cpu).to(Float32))).item<float>(), 0.0f)`
   A backward that silently zeroes gradients passes the no-throw check but
   fails the non-zero-grad check — that is the exact bug pattern we need to
   catch.
3. **Cross-backend parity test** in `tests/backend_parity/` for the layer,
   comparing forward AND backward gradients across backends within
   `parity_tolerances.hpp` bounds.
4. **Training-mode coverage** for any layer whose behavior differs in
   `.train()` vs `.eval()` (Dropout, BatchNorm, SyncBatchNorm,
   VariationalDropout, etc.). The test must exercise both modes.

---

## Autograd invariants

Code that constructs a `Variable` must preserve the upstream graph. Banned
patterns (these silently zero gradients):

```cpp
// BAD — .tensor() extracts the raw tensor and discards grad_fn.
Variable(x.tensor() * y.tensor(), /*requires_grad=*/true);

// BAD — tenzor::squeeze(tensor, dim) is not autograd-aware.
Variable(tenzor::squeeze(logits.tensor(), -1), logits.requires_grad());

// BAD — re-wrapping the output of a Variable-returning op.
Variable(avg_pool2d(x).tensor(), x.requires_grad());

// BAD — raw tensor op in a forward pass.
const float* p = slopes.data_ptr();  // crashes on any GPU
```

Instead:

```cpp
// Use Variable-aware ops that build the autograd chain.
auto z = x * y;                                     // Variable * Variable
auto squeezed = ::tenzor::squeeze(logits, -1);      // Variable-returning squeeze
auto pooled = avg_pool2d(x);                        // returns Variable
```

Inside `backward()`/`backward_with_variables()` implementations, wrapping
saved Tensors as `Variable(t, /*requires_grad=*/false)` is acceptable and
expected — those are constants w.r.t. the upstream graph.

For higher-order gradients, every `*Backward` class that has a real non-zero
second derivative must implement `backward_with_variables(...)` returning an
expression in Variable-level ops (not dispatch calls). See
`src/autograd/function_activations.cpp::SigmoidBackward_AG` for a canonical
example. Classes whose second derivative is structurally zero
(piecewise-linear ops, non-differentiable pass-throughs) should use
`TENZOR_HIGHER_ORDER_STRUCTURAL_ZERO_STUB()` and be pinned in
`tests/autograd/test_higher_order_stubs_regression.cpp`.

---

## Fixture hygiene

- `MultiBackendDTypeTest` (`tests/multi_backend_dtype_fixture.hpp`) is the
  canonical fixture for multi-backend × multi-dtype tests. It already honors
  `TENZOR_REQUIRE_MULTI_BACKEND` and `TENZOR_SKIP_BACKENDS`.
- `BackendTest` (`tests/backend_test_fixture.hpp`) is the canonical fixture
  when dtype is not a parameter.
- Do not reinvent a per-file `struct BackendDTypeParam { ... }`; the existing
  fixture already parameterizes correctly.
- Do not write `if (backends.size() < 2) GTEST_SKIP() << "...";` inline —
  use `REQUIRE_MULTI_BACKEND_OR_SKIP("op name")` from
  `tests/backend_parity/parity_test_utils.hpp` so the env-var escalation to
  hard failure is consistent.
- `STANDARD_BACKENDS` and `ALL_BACKENDS` are hardcoded to `{cpu, cuda,
  vulkan, oneapi, rocm}`. `INSTANTIATE_TEST_SUITE_P` is evaluated at
  static-init time — before `tenzor::initialize()` registers backends — so
  runtime discovery there collapses to CPU-only. The fixture's SetUp()
  skips unavailable backends at runtime, honoring
  `TENZOR_REQUIRE_MULTI_BACKEND` / `TENZOR_SKIP_BACKENDS`. Adding a new
  backend (Metal, WebGPU, etc.) requires editing this list in
  `multi_backend_dtype_fixture.hpp` and teaching the SetUp how to build
  its `Device`.

---

## Skips and disabled tests

A skip is acceptable only when:

- A backend is genuinely unavailable (covered by the fixture — don't write
  your own skip).
- A dtype is genuinely unsupported on the given backend (e.g., Float16 on
  backends without native FP16 compute). Prefer a `should_skip()` helper on
  the fixture over inline `GTEST_SKIP()`.
- An external service / file / environment is missing (init_process_group
  without gloo, MIOpen JIT on an unconfigured ROCm host, etc.).

A skip is **not** acceptable when:

- The underlying code is broken and the skip hides the failure. Fix the code.
- A binding is missing. Bind it.
- A function isn't implemented. Implement it.
- The assertion would "fail due to precision" — check whether the test design
  is correct (e.g., `sum()` vs `mean()` loss for deep Float16 models) before
  adding a skip.

Disabled tests (`DISABLED_*` prefix, `#if 0`, entire bodies commented out) are
banned. If a test cannot run today, fix the implementation it covers or delete
the test outright.

---

## CI / release gates

- `test_kernel_completeness` — hard gate. Every backend must register every
  OpId in `required_ops.hpp`.
- `test_registration_report` — currently informational; will be flipped to
  FAIL-on-gap once ROCm and OneAPI reach 462/462 parity (tracked in the
  cross-backend-port work).
- Parity tests with a committed golden baseline refuse goldens older than
  30 days (`TENZOR_GOLDEN_MAX_AGE_DAYS`). Re-record on the reference hardware
  when bumping the library.

---

## Common commands

```bash
# Configure + build (one-time; ninja caches afterwards)
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DTENZOR_TEST_BFLOAT16=ON
ninja -C build

# Run only the tests touching a specific area
ctest --test-dir build -j1 -R <pattern>

# Run a single parameterized test variant verbosely
ctest --test-dir build -j1 -R '<suite>\.<test>/<backend>_<dtype>' -V

# Regenerate the backend-registration baseline (after adding kernels)
ctest --test-dir build -j1 -R test_registration_report -V

# Python test subset
cd <repo_root> && PYTHONPATH=python:build/python python -m pytest tests/python/<file>.py -v
```

## Smoke set

A subset of fast tests is tagged with `LABELS smoke` so they can be run as
a quick-feedback gate before the full suite (the full suite runs 15+ hours
and is reserved for CI; see `MEMORY.md` -> `feedback_testing.md`).

```bash
# Run the smoke set — completes in < 5 minutes on a single shell.
ctest --test-dir build -j1 -L smoke --output-on-failure
```

The smoke set today covers:
- `test_kernel_completeness` and `test_registration_report` — catch dispatch-table regressions on every backend.
- `test_inplace_ops_parity` — ~50 cross-backend in-place op cases.
- `test_dropout`, `test_batchnorm2d`, `test_pooling` — common nn layers.
- `test_gradcheck` — basic autograd correctness.

To add a new test to the smoke set, pass `LABELS "<existing>\;smoke"` to
`tenzor_discover_tests` (note the **backslash-escaped semicolon** — see the
helper's comment in `tests/CMakeLists.txt` for why this form is required).
For tests that go through `add_parity_test`, follow up with an explicit
`set_tests_properties(<target> PROPERTIES LABELS "backend_parity;smoke")`.

## Performance regression check

`tests/backend_parity/test_performance_regression.cpp` benchmarks a small
matmul on every backend and prints timings every run. A second test
(`DISABLED_BaselineRegressionCheck_MatMul512`) compares those timings
against `tests/backend_parity/baselines/perf_baseline.json` and fails
when current median exceeds `baseline × TENZOR_PERF_REGRESSION_RTOL` or
p99 exceeds `baseline × TENZOR_PERF_REGRESSION_P99_RTOL` (defaults: 1.5
and 3.0).

The check is **disabled by default** because consumer-GPU run-to-run
variance (cache warmth, thermal throttling, scheduler jitter) routinely
exceeds those thresholds and would make CI flaky. Opt in on a
controlled benchmark host:

```bash
# Regenerate baseline after a known-good change.
python tools/regen_perf_baseline.py

# Run the regression check (disabled tests must be explicitly enabled).
build/bin/test_performance_regression \
    --gtest_filter='*BaselineRegressionCheck*' \
    --gtest_also_run_disabled_tests

# Loosen thresholds for noisier hardware:
TENZOR_PERF_REGRESSION_RTOL=2.0 TENZOR_PERF_REGRESSION_P99_RTOL=5.0 \
    build/bin/test_performance_regression \
        --gtest_filter='*BaselineRegressionCheck*' \
        --gtest_also_run_disabled_tests
```

Baselines are hardware-specific — the check skips with a clear message
when the recorded `host` field doesn't match the current machine.

## Re-enabling DISABLED_ benchmark tests (P.8)

Two perf-only gtests are intentionally prefixed `DISABLED_` so they don't
slow down or destabilise normal `ctest` runs:

- `SIMDOpsTest.DISABLED_AddPerformance`
  (`tests/unit/test_simd_ops.cpp`) — measures the SIMD `add` throughput
  ceiling on the current host.
- `PerformanceRegression.DISABLED_BaselineRegressionCheck_MatMul512`
  (`tests/backend_parity/test_performance_regression.cpp`) — compares
  current matmul latency against the recorded host baseline.

Both are exercised via the `tenzor_benchmarks` umbrella target which is
only emitted when CMake is configured with `-DTENZOR_BUILD_BENCHMARKS=ON`:

```bash
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DTENZOR_BUILD_BENCHMARKS=ON
ninja -C build tenzor_benchmarks            # build all benchmarks + perf gtests
ninja -C build run_disabled_perf_benchmarks # run the DISABLED_ perf gtests
```

To re-enable one as a regular ctest (e.g. while iterating on it), drop the
`DISABLED_` prefix in the test name in its source file.

### DISABLED_ semantics in this project (audit-3 T.16)

`tests/CMakeLists.txt::tenzor_discover_tests` invokes
`gtest_discover_tests`, which in turn registers each generated `add_test()`
line in the per-target `*_tests.cmake` files with
`--gtest_also_run_disabled_tests`. That flag applies to **every** target
discovered through `tenzor_discover_tests`, not just the SIMD / perf
benchmarks — `grep -l gtest_also_run_disabled_tests tests/*_tests.cmake`
shows the full list.

The practical contract:

- `DISABLED_` in this project means **"performance-ceiling assertion test"**,
  not "skipped". The test runs by default; the `DISABLED_` prefix is a
  documentation tag indicating it asserts a regression-guard threshold
  (typically `EXPECT_LT(latency_ms, ceiling_ms)` per audit-3 T.9), rather
  than a per-call correctness invariant.
- Tests prefixed `DISABLED_` that do NOT contain at least one `EXPECT_*`
  assertion are a bug — they will always pass even when the performance
  characteristic they exist to monitor has regressed. See audit-3 T.9 for
  the regression-guard assertion pattern that every DISABLED_ test must
  carry.
- To truly skip a test (e.g. while debugging an unrelated bug), use
  `GTEST_SKIP() << "reason"` inside the test body — the `DISABLED_` prefix
  no longer provides that effect in this build.

## Reproducibility — the global generator seed (audit-3 T.19)

Stochastic ops (`rand`, `randn`, `randint`, `dropout`, `bernoulli`, etc.)
on every backend pull from a single thread-local generator state managed
by `tenzor::manual_seed(uint64_t)`. To make a test reproducible:

```cpp
tenzor::manual_seed(42);            // once per test fixture SetUp()
auto x = tenzor::randn({4, 4});     // deterministic given the seed above
auto y = tenzor::nn::dropout(x, 0.5);  // also deterministic — same stream
```

Critical: **`std::srand()` does NOT affect tenzor RNG.** The C library's
PRNG and tenzor's generator are independent. Tests that seed `std::srand`
and then call `tenzor::randn` (or any backend kernel that pulls
`get_global_seed()`) will appear nondeterministic across runs. The audit-3
T.19 fix wires the CPU `dropout` kernel's thread-local `mt19937` into the
same `tenzor::detail::get_global_manual_seed_*` accessors that `rand` and
`randn` use, so `manual_seed` now genuinely controls every CPU stochastic
path. The other backends (CUDA, ROCm, Vulkan, OneAPI) already routed
through `get_global_seed()` and were unaffected.

If a test does not call `manual_seed`, the generator falls back to a
`std::random_device`-derived seed and the run is intentionally
non-reproducible. Prefer `manual_seed(42)` (or another fixed value) in
every fixture's `SetUp()` — see audit-3 T.7 for the pattern.
