# Gradient Check Coverage Analysis Report

## Status — Updated

**This document was originally written when gradcheck tests reported 0% coverage.** The root cause — that `INSTANTIATE_BACKEND_TESTS` tests registered 0 tests when `tenzor::initialize()` failed device enumeration during CMake's `POST_BUILD` discovery — has been resolved. `tests/autograd/CMakeLists.txt:61-68` now uses `DISCOVERY_MODE PRE_TEST` for all parameterized gradcheck binaries, which defers test listing to ctest time.

Verified 2026-04-17 by running `ctest -R "^GradCheck" -j1` from `build/`:
- 102 gradcheck tests execute across `test_gradcheck`, `test_gradcheck_comprehensive`, `test_gradcheck_direct`, `test_gradcheck_extended`, `test_gradcheck_missing`.
- 84 pass, 18 fail (real backward-pass bugs, tracked separately).
- `gradcheck()`, `gradcheck_detailed()`, `gradcheck_verbose()`, `numerical_gradient()`, `compare_gradients()`, `GradCheckResult`, and `GradCheckError` all have direct call sites in `test_gradcheck_direct.cpp`.

## Remaining gap — gradgradcheck

`gradgradcheck()` and `gradgradcheck_detailed()` are declared in `gradcheck.hpp` and defined in `gradcheck.cpp:491-557`, but until Phase 0.2 of the test-coverage plan had **zero callers in the repo**. Tests were added to `test_gradcheck_direct.cpp` covering:

- Linear `f(x) = ax + b` (second derivative zero, cleanly passes).
- Quadratic `f(x) = x^2` and cubic `f(x) = x^3` (API surface, no-throw).
- `requires_grad=false` early-return path.
- `GradCheckError` thrown path with `raise_exception=true`.

## gradgradcheck_detailed implementation (2026-04-17)

`gradcheck.cpp:499-650` uses a dual-method comparison to verify the second-derivative pipeline without requiring the autograd engine to retain a graph through the leaf gradient:

1. Compute the Hessian `H = d²f/dx²` via `hessian()` (numerical Jacobian of the gradient function).
2. Independently estimate the Hessian diagonal via a direct 2nd-order central difference of `f`:
   `h_ii ≈ (f(x+ε·e_i) − 2·f(x) + f(x−ε·e_i)) / ε²`.
3. Compare `H.diag` against the direct estimate element-wise.

The Float32 eps is clamped to `5e-4` to avoid catastrophic cancellation. Both methods are finite-difference based, so this does not yet verify the engine's `create_graph=true` path end-to-end — that is covered by the `test_higher_order_gradients`, `test_higher_order_activations`, and `test_higher_order_nn` suites, which exercise real Variable-level backward chaining (`grad_variable()`).

## Higher-order (create_graph=true) op coverage — audit 2026-04-18

`grep -rE "class \w+Backward" include/ src/` identifies 189 Backward function classes. Split:

- **135 classes** have a real `backward_with_variables()` in `src/autograd/*.cpp` that builds a Variable-level gradient graph.
- **8 classes** use `TENZOR_HIGHER_ORDER_STRUCTURAL_ZERO_STUB()` (linear/piecewise-linear forward; 2nd derivative = 0 by construction):
  AsStridedBackward, AvgPool2dBackward, ViewAsComplexBackward, ViewAsRealBackward,
  FoldBackward, UnfoldBackward, SparseEmbeddingBackward,
  plus CholeskySolveBackward / LinalgHouseholderBackward / LinalgLDLFactorBackward
  (mathematically non-linear but the stub is an intentional structural-zero fallback).
- **~35 inline `backward_with_variables` definitions** in `src/nn/**.cpp` (pooling, normalization,
  activations) that my `grep ::backward_with_variables` pattern did not capture because they are
  defined inside the class body rather than as out-of-class member functions.
- **~6 classes** (`ELUBackward`, `GeLUBackward`, `MishBackward`, `SELUBackward`, `SwishBackward`,
  `TanhBackward`) are declared but never dispatched — superseded by the corresponding
  `*Backward_AG` variants that have full higher-order support.
- **Genuinely missing implementations** (dispatched, non-linear, no stub): `SigmoidBackward` and
  `LogSigmoidBackward` in `src/nn/activations/activations.cpp` (dead-code paths; production code
  goes through `SigmoidBackward_AG` / `LogSigmoidBackward_AG` via `tenzor::sigmoid`), plus
  `CTCLossBackward`, `MultiLabelMarginLossBackward`, `RReLUBackward`, and the `Nested*Backward`
  family. These throw when `create_graph=true` is requested.

**Ongoing work:** the genuinely-missing loss/nested ops need either real implementations or a
structural-zero stub with a documented rationale. Tracked as a follow-up in the coverage plan's
Phase 7 (higher-order autograd expansion).
