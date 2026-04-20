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
