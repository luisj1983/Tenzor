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

## Update — 2026-04-20 (testing-audit pass)

Status changes since the 2026-04-18 audit:

- **`LogSigmoidBackward`** is no longer dead code — `nn::log_sigmoid` production path goes
  through it, and it now has a real `backward_with_variables()` override so higher-order
  works. Regression test: `HigherOrderActivationsTest.LogSigmoidDoubleBackwardNonZero`.
- **Dead nn-side backward classes deleted:** `SigmoidBackward`, `TanhBackward`,
  `LeakyReLUBackward`, `GeLUBackward`, `SwishBackward`, `ELUBackward`, `SELUBackward`,
  `MishBackward` are all removed from `src/nn/activations/activations.cpp`. Production goes
  through the `*Backward_AG` variants in `src/autograd/function_activations.cpp`. The
  `sigmoid_stub_legacy` helper (legacy `[[maybe_unused]]`) is also gone.
- **Multibackend gradcheck expanded** from 15 → 25 ops. Added: `Neg`, `Abs`, `Reciprocal`,
  `Sin`, `Cos`, `Pow`, `Transpose`, `Reshape`, `LeakyRelu`, `Softplus`. All 25 ops run
  across CPU/CUDA/ROCm/Vulkan/OneAPI on Float32+Float64. See
  `tests/autograd/test_gradcheck_multibackend.cpp`.
- **Autograd-break bugs fixed** (discovered via raw-tensor-op grep sweep):
  `gqa_attention.cpp` RoPE re-wrap, `hrm.cpp` ACT squeeze, `nn/functional.cpp` `lp_pool2d`
  chain break, `nn/quantization/quantized_layers.cpp` DeQuantStub bare re-wrap. All four
  now preserve grad_fn. `lp_pool1d` still needs an `avg_pool1d(Variable)` overload to match.
- **New hard-gate regression tests** for gradient flow (replacing EXPECT_NO_THROW-only):
  `LSTMGradientFlow`, `RNNGradientFlow`, `GRUGradientFlow`,
  `AttentionIntegrationMultiDTypeTest.ForwardBackward`,
  `TransformerIntegrationMultiDTypeTest.ForwardBackward`. They assert non-zero input grads
  across all five backends; CPU Float32/Float64 and GPU Float32 are all green.

## Update — 2026-04-29 (extensions — close the remaining plan gaps)

After the initial PR1+PR2+PR3 landing, a follow-up pass closed the
remaining open items from the plan:

**PR 2.4 — CI guard**: `tools/check_layer_tests.py` flags any
`tests/nn/layers/test_*.cpp` file that calls `.backward()` without a
sufficient non-zero gradient assertion (`EXPECT_GRAD_FLOWS`,
`numerical_gradient`, an explicit `max(abs(grad))>0` check, or a
per-element `EXPECT_NEAR` pattern). Wired into ctest as
`LayerTestGradFlowGuard` (label `lint;layer_tests`). Files can opt out
with a `// grad-check-exempt: <reason>` comment when a non-zero check is
genuinely inappropriate.

**PR 3.4 (extended)**: `Sort`, `TopK`, `GridSample`, `AffineGrid`
gradchecks added. Required adding Variable-level wrappers for
`grid_sample` and `affine_grid` (in `include/tenzor/autograd/ops.hpp` +
`src/autograd/ops.cpp`) since only the Backward classes existed; now
exposed as public API alongside the gradchecks.

**PR 3.1 (extended)**: `SparseTriSolve` Variable wrapper added (sparse L
matrix is constant, dense `b` is the differentiated input). Gradcheck
passes on CPU. SpGEMM is intentionally not exposed as a Variable op —
its dense-via-sparse use case is mathematically equivalent to `matmul`
which is already covered.

**PR 3.5 (extended)**: `view_as_real` and `view_as_complex` Variable
wrappers added — these were the missing piece for complex-output
gradcheck. With them, dedicated `FFTGradcheck` and `IFFTGradcheck` tests
now pass via `f(x) = sum(view_as_real(fft(x)))` (and the symmetric form
for IFFT through `view_as_complex` of a real `[..., 2]` input). The
RFFT/IRFFT round-trip variants are kept for negative-dim and norm-scale
coverage.

**PR 2.1 (extended) — chat_ai re-evaluation**: 11_chat_ai remains Tier B
intentionally; full re-evaluation rationale in `tests/examples/SKIP_NOTES.md`.
Summary: chat_ai already has a synthetic-pair fallback in `load_pairs`,
so disk-IO isn't the blocker; the autograd surface (GRU, Bahdanau
attention via matmul+softmax, log_softmax cross-entropy) is already
covered by other wired examples (07_rnn_sequence, 16_self_attention).

After all extensions: **116/116 tests pass on CPU**
(94 gradcheck tests + 21 example regressions + 1 lint guard).
RFFT/IRFFT, FFT, IFFT, sparse_triangular_solve, grid_sample, affine_grid,
sort, topk are now all covered by gradcheck.

## Update — 2026-04-29 (audit fix PR2/PR3 — example regressions, neg-dim gradcheck, op coverage)

PR 2 + PR 3 of the audit fix landed:

**PR 2.3 — parameterized negative-dim gradcheck** (`tests/autograd/test_gradcheck_negative_dim.cpp`):
- 18 tests covering every dim-taking op with both positive and negative
  dim values: index_select, gather, narrow, sum, prod, max(dim), min(dim),
  logsumexp, cumsum, cumprod, var, std, softmax, log_softmax, flip, roll,
  cat. 17 pass on CPU.
- Mean(dim) is skipped — pre-existing crash in MeanBackward when dim is
  passed as `int64_t` (libstdc++ span out-of-bounds assertion). Tracked
  separately as a real bug.

**PR 3.1 — sparse op gradchecks** (3 ops):
- `SpMMGradcheck`, `SpMVGradcheck`, `SparseAddGradcheck` cover the dense-
  side gradient. `SpGEMMBackward` and `SparseTriSolveBackward` are
  declared but have no Variable-level autograd entry point in the public
  API — left as a follow-up if the user wants them exposed.

**PR 3.2 — linalg gradchecks** (5 ops):
- `VecdotGradcheck`, `VectorNormGradcheck`, `MatrixNormGradcheck`
  (Frobenius — operator 2-norm goes through SVD with delicate backward
  near degenerate singular values), `EigvalshGradcheck`, `SolveGradcheck`.

**PR 3.3 — special-math gradchecks** (6 ops):
- `ErfGradcheck`, `ErfcGradcheck`, `ErfInvGradcheck` (input clamped to
  |x|<0.3 to keep curvature bounded), `I0eGradcheck`, `I1eGradcheck`,
  `MultigammalnGradcheck`.

**PR 3.4 — indexing/shape leftovers** (3 ops):
- `ScatterGradcheck` (plain scatter, distinct from ScatterAdd which was
  already covered), `TrilGradcheck`, `TriuGradcheck`.
- `SortBackward`, `TopKBackward`, `GridSampleBackward`, `AffineGridBackward`
  remain uncovered — Sort/TopK gradients flow only through values
  (deferred until needed); GridSample/AffineGrid require parameterized
  setup (deferred).

**PR 3.5 — FFT gradchecks** (RFFT/IRFFT round-trip variants):
- `RFFTIRFFT_RoundTrip_DefaultDim`, `_NegativeDim`, `_OrthoNorm`. Pure
  FFT/IFFT explicit gradchecks remain skipped pending a complex-output
  gradcheck infrastructure (the existing `tenzor::sum` doesn't reduce
  complex tensors). The round-trip variants exercise both RFFTBackward
  and IRFFTBackward, including the negative-dim and norm-scaling code
  paths.

After PR1+PR2+PR3:
- 87/87 gradcheck tests pass on CPU (`GradCheckMissingTest +
  GradCheckNegativeDimTest` filtered to cpu only).
- `tests/examples/test_hrm_example.cpp` (PR1) plus the bulk
  `tests/examples/test_all_autograd_examples.cpp` (PR2.1) provide
  end-to-end regression coverage; loss-decrease assertions catch
  zero-gradient bugs at training time.
- Layer tests across `tests/nn/layers/` now use `EXPECT_GRAD_FLOWS`
  consistently for backward-running tests, replacing the prior
  `has_value()` and `EXPECT_NO_THROW(backward())` patterns that masked
  silently-zeroed gradients.

## Update — 2026-04-28 (audit fix PR1 — functional wiring + grad-flow macro)

PR 1 of the testing-audit fix landed (see `~/.claude/plans/create-and-implement-a-wild-pearl.md`). Net changes:

- **Live Python-reachable autograd-break bugs fixed** in `src/nn/functional.cpp`:
  - `functional::group_norm`, `functional::instance_norm`, `functional::embedding`
    now properly wire their `*Backward` Function via new factories
    (`internal::make_group_norm_backward`, `make_instance_norm_backward` in
    `normalization.hpp`; `make_embedding_backward` in `embedding.hpp`).
  - `functional::nll_loss` rewritten to use Variable-level ops (gather / neg /
    mean / sum) — no custom Function needed; grad flows automatically.
  - `functional::dropout`, `functional::normalize`, and `functional::pad`
    (reflect/replicate/circular path) refactored to Variable-level ops —
    same raw-tensor-op-breaks-grad-fn pattern, fixed identically.
  - `functional::interpolate` made explicitly `requires_grad=false` — no
    `InterpolateBackward` exists yet; previous code was lying about
    differentiability. Tracked as a follow-up.
  - `Parametrization::forward_impl` documented as non-grad-flowing (the
    base `forward(Tensor)→Tensor` signature can't preserve grad_fn; design
    fix is out of PR1 scope).

- **New gradchecks** in `test_gradcheck_missing.cpp`:
  - `FunctionalGroupNormGradcheck`, `FunctionalInstanceNormGradcheck`,
    `FunctionalEmbeddingGradcheck`, `FunctionalNllLossGradcheck` —
    CPU-only at Float32 (the GroupNorm/InstanceNorm CPU backward kernels
    internally downcast to Float32; documented in the tests).

- **Layer-level grad-flow assertion macro** added at
  `tests/grad_flow_helpers.hpp`: `EXPECT_GRAD_FLOWS(var)` asserts the
  Variable's grad has at least one non-zero element after backward. Catches
  the "severed grad_fn returns zero gradients" failure class that the
  prior `has_value()` / `EXPECT_NO_THROW(backward())` patterns miss.

- **High-risk layer tests retrofitted** with EXPECT_GRAD_FLOWS or new
  grad-flow tests across these files: `test_gated_activations.cpp`,
  `test_glu_multidtype.cpp`, `test_lazy_backward.cpp`,
  `test_lazy_backward_multidtype.cpp`, `test_sparse_linear.cpp`,
  `test_sparse_linear_multidtype.cpp`, `test_dropout.cpp`,
  `test_dropout_multidtype.cpp`, `test_multihead_attention_multidtype.cpp`,
  `test_window_attention.cpp`, `test_window_attention_multidtype.cpp`.

- **HRM example wired as ctest regression** in
  `tests/examples/test_hrm_example.cpp`. The training body of
  `examples/cpp/showcase/22_hierarchical_reasoning/autograd.cpp` was
  extracted into `autograd_runner.{cpp,hpp}` so the test drives the same
  code path as the showcase exe. The test asserts that loss decreases by
  at least 0.10 absolute over 200 epochs — would have caught the
  IndexSelect/Narrow negative-dim bug that originally surfaced here.

- **Pre-existing oneapi backend crashes** in SparseLinear, LazyBackward,
  and a few other layers were exposed by the new tests (not introduced
  by them — `SparseLinear::ForwardShape` crashes on oneapi too, and that
  test predates PR1). These remain on the open-issue list and are
  excluded from PR1 verification via `-E "oneapi"`.

PR2 (full ~45-file layer-test sweep, all 22 examples wired as ctest,
negative-dim parameterized gradcheck) and PR3 (sparse/FFT/linalg/special-
math gradcheck holes) follow per the plan.

## Update — 2026-04-20 (E1 expansion, 60+ ops covered)

Additional gradcheck coverage landed in `tests/autograd/test_gradcheck_missing.cpp`
after the 2026-04-18 audit pass:

**Arithmetic / element-wise:** `neg`, `abs`, `reciprocal`, `sqrt`, `pow`,
`log1p`, `log2`, `log10`, `exp2`, `expm1`, `atan2`, `bmm`, `prod`.

**Trigonometric / hyperbolic:** `sin`, `cos`, `tan`, `asin`, `acos`, `atan`,
`sinh`, `cosh`, `tanh`. Vulkan skipped for the whole family per
tracked task J4 (SPIR-V/GLSL shader precision insufficient for Float64
gradcheck).

**Reductions:** `var`, `std`, `cumsum`, `cumprod`, `max(dim)`, `min(dim)`,
`logsumexp` (Vulkan skipped per J4).

**Indexing / shape:** `index_select`, `gather`, `expand`, `flip`, `roll`,
`cat`, `transpose`, `reshape`, `diag`, `trace`.

**Activations:** `leaky_relu`, `softplus` (added to the multibackend file
too).

## Update — 2026-07-18 (Finding 31 — regenerated stale gap list)

`tests/autograd/test_gradcheck_multibackend.cpp` had grown to **176**
`TEST_P(GradCheckMultiBackendTest, ...)` cases by this update (`grep -c
"^TEST_P(GradCheckMultiBackendTest" tests/autograd/test_gradcheck_multibackend.cpp`),
not the 25 the previous revision of this doc claimed — most of the
"gaps still open" list below (as of the 2026-04-29 update) had since been
closed under later audit-phase markers without this doc being updated to
match. Verified op-by-op via `grep "^TEST_P(GradCheckMultiBackendTest"` and
re-derived which entries are genuinely still absent vs. now present:

**Now covered** (previously listed as gaps, confirmed present in this file):
LU (`LinalgLU*`), QR (`LinalgQR*`), eigh/eigvalsh
(`LinalgEigh*`/`LinalgEigvalsh*`), SVD (`LinalgSVD*`),
`linalg_matrix_norm` (`LinalgMatrixNorm`), `linalg_ldl_factor`
(`LinalgLDLFactor`), Bessel functions (`Bessel*`), `i0e`/`i1e`
(`I0e`/`I1e`), `multigammaln` (`Multigammaln`), `erf_inv` (`ErfInv`),
`sparse_add`/`sp_mm`/`sp_mv` (`SparseAdd`/`SparseSpMM`/`SparseSpMV`),
FFT/IFFT round-trip forms (`FFTRoundTrip`, `FFT2RoundTrip`,
`FFTNRoundTrip`, `IFFT2RoundTrip`, `IFFTNRoundTrip`).

**Still genuinely absent** (re-confirmed by grep, zero matching
`TEST_P` cases):
- `linalg_vecdot`, `linalg_vector_norm`, `linalg_ldl_solve`.
- Direct (non-round-trip) `rfft`/`irfft` gradcheck — only the
  `fft`/`ifft` round-trip forms above exist; a real-input rfft/irfft pair
  has no dedicated case.
- `sp_gemm` (sparse-sparse matmul), `sparse_tri_solve`.

Real, current pass/fail tally for this file (`./bin/test_gradcheck_multibackend`,
run 2026-07-18, no filter): **3168 tests, 1548 passed, 0 failed, 1620
skipped** (Float16 — gradcheck only supports Float32/Float64 — MPS
unavailable on this Linux host, and the tracked J4 Vulkan-Float64
trig/hyperbolic skips account for the skips). This supersedes the
2026-04-17 tally of "84 pass, 18 fail" from `test_gradcheck*` binaries
predating this file's growth to 176 ops — that tally is retained above
purely as a historical note, not current status.

Gaps still open:
- `linalg_vecdot`, `linalg_vector_norm`, `linalg_ldl_solve`, direct
  `rfft`/`irfft`, `sp_gemm`, `sparse_tri_solve` (see the itemized list
  above).
- Stride-variant parity for Conv/Pool/Norm surfaced J6 (stride-ignoring
  kernels) and J5 (Float64 downcast); both tracked.
- `functional::avg_pool2d`/`max_pool2d`/`adaptive_avg_pool2d`/`lp_pool1d`
  autograd breaks fixed (J7 / B4b).
- Trig/hyperbolic Vulkan Float64 precision issues consolidated under J4.
- MoE backward parity now cross-backend (D1). HRM blocked by pre-existing
  forward divergence (J3). FlexAttention takes Tensor not Variable so
  autograd backward doesn't apply directly at the functional level.
- Float64 parity for Conv is green across all backends; LayerNorm/Pool
  diverge on every GPU backend (J5).
- ROCm 126 ops and OneAPI 65 ops still missing from dispatch tables (F1/F2)
  as of the 2026-04-29 audit — not re-verified by this update (out of
  scope for the gradcheck-file gap-list regeneration Finding 31 covers).
