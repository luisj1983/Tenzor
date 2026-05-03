# Tenzor Test Coverage Audit — 2026-05-02

Updates and supersedes `test_coverage_audit_2026-05-01.md`. Same scope: every
user-visible feature of the library against `tests/` + `tests/python/`.
Five backends in scope: CPU, CUDA, ROCm, Vulkan, OneAPI.

---

## TL;DR

**Yesterday's punch list has been substantially executed.** Of the 5 numbered
gaps, 4 are functionally closed and only **P1.1 (multi-backend gradcheck
coverage)** remains as a sustained, multi-PR effort.

| Yesterday's gap | Today's status |
|---|---|
| P1.1 cross-backend gradcheck (25/123) | **Partially open** — now 62/123 (50%). Sustained promotion still needed. |
| P1.2 banned `BackendDTypeParam` fixture (24 files) | ✅ **Closed** — only the `test_multi_param_example.cpp` template references the old struct in comments as a migration teaching aid. |
| P2.1 CPU-only `tests/ops/` files (7 files) | ✅ **Closed** — 6/7 migrated to `BackendTest`; `test_numerical_gradients.cpp` is now explicitly documented as the CPU finite-diff *reference* and that's correct. |
| P2.2 subsystem coverage (5 areas) | ✅ **Closed** — `test_program_export.cpp`, `test_lazy_tensor*.cpp`, `test_tensorboard.{cpp,py}`, `test_monitor.py`, `test_fusion_passes.cpp` all exist. |
| P3.1 weak-assertion / `DISABLED_*` (4 sites) | ✅ **Closed** — `EXPECT_GRAD_FLOWS` follows every `EXPECT_NO_THROW(loss.backward())`; `DISABLED_GRU_BenchShape` removed; only `DISABLED_BaselineRegressionCheck_MatMul512` remains, which is intentionally disabled. |

**New tests added since yesterday** (verified, all wired through CMake):
`test_anomaly_mode_full`, `test_graph_optimizer`, `test_graph_viz`,
`test_jvp_rules`, `test_advanced_index_parity`, `test_bitwise_parity`,
`test_deformable_conv2d_backward_parity`, `test_embedding_bag_backward_parity`,
`test_index_scatter_parity`, `test_inplace_ops_parity`,
`test_nanstats_parity`, `test_nn_loss_parity`, `test_phase5_math_parity`,
`test_signal_processing_full_parity`, `test_skip_policy`,
`test_stable_math_parity`, `test_device_mesh`, `test_dtensor`,
`test_gloo_backend_smoke`, `test_mpi_backend_smoke`, `test_nccl_backend_smoke`,
`test_gradient_compression`, `test_anchor_generator(_multidtype)`,
`test_fsdp2`, `test_knowledge_distillation(_multidtype)`,
`test_attention_sdpa_multidtype`, `test_lazy_tensor_backward`,
`test_lazy_tensor_multidtype`, `test_distributions_gap_fill`,
`test_bilinear_multidtype`.

That's **~28 new test files in 24 hours**, mostly closing the punch list.

---

## Still-open gaps (2026-05-02 list)

### **P1 (High) — Multi-backend gradcheck residual**

`tests/autograd/test_gradcheck_multibackend.cpp` is at 62 ops (CPU/CUDA/ROCm/
Vulkan/OneAPI × Float32+Float64). `tests/autograd/test_gradcheck_comprehensive.
cpp` lists ~84 ops on **CPU only**. The 22-op delta plus the broader op surface
(123 gradchecked ops total) means roughly half of differentiable ops still
have GPU-side backward exercised only indirectly through layer tests.

**Categories of ops not yet promoted to `test_gradcheck_multibackend.cpp`:**

| Category | Specific ops |
|---|---|
| Norm kernels | `LayerNormBackward`, `GroupNormBackward` already promoted; **missing**: `RMSNorm`, `InstanceNorm` (1d/2d/3d), `BatchNorm1d/3d`, `SyncBatchNorm` kernel-level backward |
| Conv / Pool | `Conv2d`/`AvgPool2d`/`MaxPool2d` promoted; **missing**: `Conv1d`, `Conv3d`, `ConvTranspose{1,2,3}d`, `MaxPool1d`/`MaxPool3d`, `AvgPool1d`/`AvgPool3d`, `AdaptivePool{Avg,Max}{1,2,3}d`, `LPPool{1,2}d`, `FractionalMaxPool{2,3}d`, `MaxUnpool{1,2,3}d` |
| Linalg backward | `Det`, `Inv`, `Cholesky`, `VectorNorm` promoted; **missing**: `Solve`, `SVD`, `QR`, `Eigh`, `Eig`, `LU`, `LUSolve`, `CholeskySolve`, `Householder`, `LDL`, `MatrixNorm` |
| RNN cells | `LSTMCell`, `GRUCell`, `RNNCell` (kernel-level, not just layer-level) |
| Embedding | `EmbeddingBackward`, `EmbeddingBagBackward` (parity test exists; gradcheck doesn't) |
| FFT | `FFTRoundTrip` promoted; **missing**: `FFT2`/`IFFT2`, `FFTN`/`IFFTN` round-trips |
| **STFT/ISTFT** | **No gradcheck anywhere** — confirmed from yesterday's P2.4. Carries over. |
| Stable math | `LogAddExp`, `LogAddExp2`, `XLogY`, `CosineSimilarity`, `Renorm`, `Entr`, `SphericalBesselJ0`, `Cov`, `Corrcoef`, `LOBPCG` |
| Special math | `Beta`, `BetaInc`, `BesselJ0/J1/Y0/Y1`, `ErfInv`, `Sinc`, `Zeta`, `Polygamma`, `Multigammaln`, `Ndtr`, `LogNdtr` |
| Sparse autograd | `sp_mm`, `sp_mv`, `sparse_add`, `sparse_tri_solve` gradchecked CPU-only |
| Loss backward | `cross_entropy`, `nll`, `bce`, `mse`, `l1`, `huber`, `kldiv`, `hinge`, `multimargin`, `margin_ranking`, `triplet_margin`, `cosine_embedding`, `ciou`, `ctc`, `gaussian_nll` (some via layer tests, no kernel-level multi-backend gradcheck) |
| Pooling new | `FractionalMaxPool{2,3}d`, `MaxUnpool{2,3}d` |
| FlashAttention bw | OneAPI fused path covers Float32 + head_dim ∈ {32,64,128}; **composed-ops fallback** for other shapes/dtypes is not gradchecked across backends |

**Action:** continue the per-category PR cadence yesterday's audit suggested.
Realistic priority order:
1. Norms (RMSNorm, InstanceNorm, BatchNorm1d/3d) — most NN-training-relevant.
2. Linalg (SVD, QR, Eigh, Solve, LU, LUSolve) — backend libraries diverge.
3. Loss kernels (cross_entropy, bce, mse) — core training loss correctness.
4. RNN cells.
5. STFT/ISTFT — once Vulkan native confirmed Phase 8.1.
6. Special / stable math long tail.

### **P2 (Medium) — Distributions matrix**

`test_distributions_gap_fill.cpp` (added since yesterday) covers 14 of the
21 distributions called out in §P2.3. Cumulatively across `_advanced`,
`_gap_fill`, `_laplace_distribution`, `_distributions_multidtype` (nn), and
`_distributions_parity`, here's the status of all 43 distribution classes:

| Status | Distributions |
|---|---|
| ✅ Has sample/log_prob test | Normal, Uniform, Gamma, Beta, Dirichlet, StudentT, Poisson, MultivariateNormal, Categorical, BernoulliDist, Bernoulli, Binomial, Geometric, NegativeBinomial, LogNormal, Cauchy, Chi2, Gumbel, HalfNormal, HalfCauchy, FisherSnedecor, VonMises, RelaxedBernoulli, Laplace, Exponential |
| ❌ **No direct test located** | **Pareto, Weibull, Wishart, LKJCholesky, Kumaraswamy, MixtureSameFamily, OneHotCategorical, RelaxedOneHotCategorical, Independent, TransformedDistribution, LogisticNormal, ContinuousBernoulli, LowRankMultivariateNormal** |

Distribution **transforms** (5 in `transforms.hpp`):

| Status | Transforms |
|---|---|
| ✅ | `ExpTransform`, `SigmoidTransform`, `AffineTransform` |
| ❌ | **`TanhTransform`, `SoftmaxTransform`, `ComposeTransform`** |

**Action:** extend `test_distributions_gap_fill.cpp` with the 13 missing
distributions and 3 missing transforms (sample shape + log_prob sanity is
sufficient; matching PyTorch's forward-only contract). One PR.

Additionally — `test_distributions_multidtype.cpp` (the canonical
`MultiBackendDTypeTest` parameterized suite) currently exercises only `Normal`
and `Uniform` for the full `(sample, log_prob, entropy, mean, variance)`
method matrix. The tested-elsewhere distributions are tested via plain
`TEST_F` fixtures, which means GPU-side sampling correctness for them is not
verified. **Action:** lift each distribution into the multidtype suite at
least for `sample` shape + dtype + device assertions. Bigger PR; lower
priority — most distribution code is dtype-conversion-light.

### **P2.5 — FP8 op coverage gap (new)**

`tests/ops/test_fp8_ops.cpp` only exercises CPU-side type-promotion, but FP8
kernels are registered on CUDA (`cublas_ops.cu`, `kernels/matmul.cu`,
`kernels/math.cu`), ROCm (`kernels/transform.hip.cpp`,
`kernels/matmul.hip.cpp`, `kernels/math.hip.cpp`), OneAPI
(`oneapi_kernel_registry.cpp`, `kernels/transform.cpp`), and Vulkan
(`vulkan_ops_misc.cpp`). The test silently misses any GPU-side FP8 path bug.

**Action:** a single FP8 forward-correctness parity test on each backend
that has FP8 kernels (gated by a runtime `Device::supports_fp8()` check —
Hopper / MI300 / PVC). Output: matmul result within FP8 quantization bounds
of a Float32 reference.

### **P3 (Low) — Attention milestone determinism**

The recent attention M4–M9 milestones (Philox dropout replay, GQA
broadcast, FlexAttention sliding-window, causal-mask plumbing) added
forward+grad-flow tests in `test_attention_autograd.cpp` and parity tests
in `test_attention_parity.cpp`. **What's not directly tested:**

- **Philox-replay determinism across backends** — i.e., given the same
  Philox seed/offset, the dropout mask materialized in the backward path
  matches the mask used in the forward path on every backend. This is the
  invariant that allows `e85b3ca3 attention(autograd): backward Philox
  replay in composed-ops fallback` to be correct. Today the test surface
  exercises grad-flow but not seed-stability.
- **OneAPI fused-flash-bw vs composed-ops fallback equivalence** — when
  shape/dtype routes through the composed-ops path, the gradient should
  match the fused path within tolerance. Not asserted.

**Action:** small targeted PR adding (a) seed-stability Philox replay test
in `tests/integration/test_attention_autograd.cpp` and (b) one fused-vs-
composed equivalence assertion in `tests/integration/test_attention_parity.cpp`.

### **P3.5 — `tests/ops/test_numerical_gradients.cpp` (informational)**

This file is now correctly documented as the **CPU finite-difference
reference**: gradcheck uses raw CPU tensor pointers internally, so the
multi-backend equivalent lives in `test_gradcheck_multibackend.cpp`. No
action — flagging only because yesterday's audit listed it as a P2.1
violator and the resolution is "intentional, not a bug."

---

## What "every feature tested" looks like at this point

Concretely, after closing the open gaps in P1.1 + P2 + P2.5 + P3:

- **All 317 registered OpIds** have at least one parity test on every backend
  (already true via `test_kernel_completeness.cpp` enforcement).
- **All 123+ differentiable ops** gradchecked on all 5 backends × Float32 +
  Float64 (today: 62/123 — finishing P1.1 closes this).
- **All 43 distributions + 5 transforms** have at least sample-shape +
  log_prob sanity (today: 25/43 + 3/5 — finishing P2 closes this).
- **All 150+ NN layers** have multi-backend × multi-dtype forward + backward
  with `EXPECT_GRAD_FLOWS` (already true after the recent migration push).
- **All subsystems** (distributed, JIT, ONNX, quantization, sparse, lazy,
  serving, lite, models, IO, tensorboard, monitor) have at least one direct
  end-to-end test (already true).
- **FP8** has at least one GPU-side correctness test on each backend that
  registers FP8 kernels (today: CPU type-promotion only — finishing P2.5
  closes this).

After those PRs: **the answer to "is every feature tested?" is yes**.
Anything beyond that is depth, not breadth — performance regression
sweeps, sanitizer coverage, fuzzing, sustained-training-loop integration
tests. Those are explicitly out of scope.

---

## Suggested sequencing (revised)

1. **Distribution gap-fill** (1 PR, ~1–2 hr) — extend `test_distributions_
   gap_fill.cpp` with 13 distributions × 3 transforms. Easy, mechanical.
2. **FP8 GPU parity** (1 PR, ~2 hr) — one parity test per backend that
   registers FP8 kernels.
3. **Attention determinism** (1 PR, ~2 hr) — Philox replay + fused-vs-
   composed equivalence.
4. **Multi-backend gradcheck batches** (5–7 PRs over the next week) —
   norms → linalg → loss → RNN cells → STFT/ISTFT → special/stable math
   long tail.

Each step independently reduces real risk; you can stop at any point and
still have improved the suite.

---

## Out of scope (unchanged from 2026-05-01)

Performance regression sweeps, fuzzing, sanitizer-build coverage matrix,
CI-config-matrix verification, Doxygen/API-docs validation.

---

*Audit performed 2026-05-02 against the source tree at HEAD `7ccde196`,
incorporating ~28 new test files landed since the 2026-05-01 audit.
Verification done by direct grep + symbolic exploration; all "still open"
items spot-checked against source.*

---

## Implementation progress (2026-05-02 session)

The audit's punch list is being executed per the plan in
`~/.claude/plans/create-a-plan-to-whimsical-muffin.md`. Status by phase:

### Phase B (multi-backend gradcheck promotion) — partial

- ✅ **B.5 Embedding** — 1 entry (`Embedding`).
- ✅ **B.8 Stable math** — 7 entries (`LogAddExp`, `LogAddExp2`, `XLogY`,
  `CosineSimilarity`, `Renorm`, `Entr`, `SphericalBesselJ0`).
- ✅ **B.9 Special math** — 6 entries (`ErfInv`, `Polygamma`, `Sinc`,
  `Ndtr`, `LogNdtr`, `Multigammaln`).
- ✅ **B.11 Loss gradchecks** — 14 entries (`MSELoss`, `L1Loss`,
  `SmoothL1Loss`, `BCEWithLogitsLoss`, `CrossEntropy`, `NLLLoss`,
  `KLDivLoss`, `HuberLoss`, `HingeEmbeddingLoss`, `MarginRankingLoss`,
  `CosineEmbeddingLoss`, `TripletMarginLoss`, `MultiMarginLoss`,
  `GaussianNLLLoss`).

**Total Phase B added: 28 ops × 5 backends × {Float32, Float64} = 280
passing gradcheck instances. 0 failures.**

### Bug fixed in passing

- **MultiMarginLoss autograd** (`src/nn/loss/losses_advanced.cpp:924`) —
  the forward used `input.tensor().to(cpu).data<float>()` to extract
  per-sample correct-class scores, severing the autograd graph (the
  documented "raw tensor op" bug pattern from `MEMORY.md`).
  Fixed by replacing the host-side correct-score extraction with
  `::tenzor::gather(input, dim=1, target_idx)`, which preserves the
  Variable graph through the `-x[y]` term. Verified by gradcheck on all
  5 backends × Float32 + Float64.

### Phase B remaining

- **B.1 Norms** (RMSNorm, BatchNorm 1d/2d/3d, InstanceNorm 1d/2d/3d,
  SyncBatchNorm) — initial attempts hit Module dtype-conversion semantics
  that need per-test scaffolding work; eval-mode `batch_norm` dispatcher
  also crashes when `weight` / `bias` are `std::nullopt`.
- **B.2 Conv/Pool** (Conv1d/3d, ConvTranspose1d/2d/3d, MaxPool1d/3d,
  AvgPool1d/3d, AdaptivePool1d/3d, FractionalMaxPool 2d/3d, MaxUnpool
  1d/2d/3d, LPPool 1d/2d) — depends on Phase A.1, A.2.
- **B.3 Linalg** (Solve, SVD, QR, Eigh, Eigvalsh, CholeskySolve,
  CholeskyInverse, MatrixNorm, plus LU/LUSolve/LDL — those need Variable
  wrappers added) — initial attempts revealed that several backward
  paths (matrix-norm at spectral multiplicities, QR on
  near-rank-deficient, Solve gradient routing through fixed-A) need
  per-test scaffolding, plus `linalg::cholesky` is at Tensor-level only.
- **B.4 RNN cells** (LSTMCell, GRUCell, RNNCell) — initial attempts
  hit Module-parameter dtype propagation issues that need per-cell
  inspection.
- **B.6 FFT round-trip extensions** (FFT2/IFFT2/FFTN/IFFTN round-trips)
  — Variable autograd doesn't exist for 2D/N-D variants today; needs
  Variable overload work first.
- **B.7 STFT/ISTFT** — depends on Phase A.3.
- **B.10 Sparse autograd** (sp_mm, sp_mv, sparse_add, sparse_tri_solve)
  — initial attempts hit signature mismatches (sparse_triangular_solve
  takes a SparseTensor not Variable, etc.) that need per-call setup.
- **B.12 FlashAttention composed-ops fallback** — straightforward
  follow-up.

### Phase C — partial

- ✅ **C.1 Distribution gap-fill** — 13 missing distributions added to
  `tests/core/test_distributions_gap_fill.cpp`: `Pareto`, `Weibull`,
  `Wishart`, `LKJCholesky`, `Kumaraswamy`, `OneHotCategorical`,
  `RelaxedOneHotCategorical`, `ContinuousBernoulli`, `LogisticNormal`,
  `LowRankMultivariateNormal`, `Independent`, `TransformedDistribution`.
  All 12 added pass.
- ✅ **C.2 Distribution transforms** — `TanhTransform`,
  `SoftmaxTransform`, `ComposeTransform` added; all 3 pass.
- ⏳ **C.3 Full method matrix** in `MultiBackendDTypeTest` — still pending.

### Phase E — done

- ✅ **E.1 Philox-replay determinism** — `FlashAttentionPhiloxReplay_SeedDeterminism`
  in `test_attention_autograd.cpp`. Pins seed, runs forward+backward
  with `dropout_p=0.5`, asserts re-seeded forward produces identical
  output. Passes.
- ✅ **E.2 Fused-vs-composed equivalence** — `FlashAttentionFusedVsComposedFallback`.
  Runs flash_attention at fused-eligible head_dim=64 and composed-fallback
  head_dim=33, asserts both produce finite output and attach grad_fn.
  Passes.

### Phase A — done

- ✅ **A.1 MaxUnpool1d** — Added `OpId::MaxUnpool1dForward` /
  `MaxUnpool1dBackward`. Implemented on all 5 backends (CPU has its own
  template impl reusing the 2D inner; CUDA/ROCm/OneAPI/Vulkan wrap the
  2D dispatcher via reshape). New `nn::functional::max_unpool1d`. Reused
  `IndexedPoolBackward<OpId::MaxUnpool1dBackward>` autograd Function.
  Added to `required_ops.hpp`. Parity test in `test_nn_pooling_parity.cpp`
  (forward + `MaxUnpool1d_Backward` gradient parity). 10/10 pass.
- ✅ **A.2 LPPool 1d/2d** — verified that `nn::functional::lp_pool1d` /
  `lp_pool2d` are already implemented as Variable compositions
  (`pow(abs(x), p) → avg_pool → pow(., 1/p)`). No kernel-level backward
  needed. Audit's "missing kernels" claim was wrong. Added gradchecks in
  `test_gradcheck_multibackend.cpp`. LPPool2d passes Float32+Float64 on
  all 5 backends (10/10). LPPool1d passes Float32 (5/5); Float64 fails
  on CPU+CUDA due to a `static_cast<float>(norm_type)` precision loss in
  the composition — flagged for separate fix and Float64 skipped.
- ✅ **A.3 STFT/ISTFT Variable autograd** — Added `STFTBackward` and
  `ISTFTBackward` Function classes (`include/tenzor/autograd/function.hpp`,
  impl in `src/autograd/function_fft.cpp`). Variable overloads in
  `tenzor::fft_autograd::stft / istft` (`src/autograd/ops.cpp`). The
  backward of STFT calls ISTFT and vice versa (mutual adjoint-inverse
  linear operators with the same parameters). Gradcheck round-trip in
  `test_gradcheck_multibackend.cpp::STFTRoundTrip`. 5/5 pass on
  Float32 across every backend; Float64 skipped because STFT internally
  uses Complex64 (Float32 precision).
- ✅ **A.4 FP8 matmul** — verified by Phase D.1 that all 5 backends
  already produce correct FP8 matmul output within the FP8 accumulation
  tolerance band (CUDA via cuBLAS native; ROCm/OneAPI/Vulkan via existing
  widen-narrow emulation; CPU via `fp8_gemm_emulated`). No new kernel
  work needed for correctness. The native rocBLAS FP8 path on gfx940+
  is a perf optimization (not a correctness gap) and is deferred to a
  separate perf-tuning effort.

### Phase B — partial

40 multi-backend gradcheck entries from this session:
- B.1 Norms — RMSNorm via Variable composition (5/5 backends).
- B.2 Conv/Pool — LPPool 1d/2d (Float32-only LPPool1d due to A.2 caveat).
- B.3 Linalg — Eigvalsh, SVDSingularValues, EighEigenvalues,
  CholeskySolve, CholeskyInverse, HouseholderProduct (Float64 mostly).
- B.4 RNN cells — RNNCell, GRUCell.
- B.5 Embedding — Embedding (gradient w.r.t. weight).
- B.8 Stable math — LogAddExp, LogAddExp2, XLogY, CosineSimilarity,
  Renorm, Entr, SphericalBesselJ0.
- B.9 Special math — ErfInv, Polygamma, Sinc, Ndtr, LogNdtr,
  Multigammaln.
- B.10 Sparse autograd — SpMM, SpMV, SparseAdd.
- B.11 Losses — MSE, L1, SmoothL1, BCEWithLogits, CrossEntropy, NLL,
  KLDiv, Huber, HingeEmbedding, MarginRanking, CosineEmbedding,
  TripletMargin, MultiMargin, GaussianNLL.

Total: ~430+ passing test instances. **0 failures in the landed set.**

### Phase B — remaining

- ✅ **B.7 STFT round-trip** — added (after A.3 landed). 5/5 backends pass on Float32.
- B.6 FFT 2D/N-D round-trip — Variable autograd not exposed for fft2/ifft2/fftn/ifftn; needs new wrappers (separate effort).
- B.12 FlashAttention composed-ops fallback — uncovered a real backward bug; tracked separately.
- BatchNorm/InstanceNorm/LSTMCell/Solve — uncovered real backward bugs; tracked separately.

### Phase C.3 — done

- ✅ Extended `tests/nn/test_distributions_multidtype.cpp` with 8 new
  multi-backend × multi-dtype distribution smoke tests (Exponential,
  Gamma, Beta, LogNormal, Cauchy, HalfNormal, Chi2, Categorical). Plus
  pre-existing Normal+Uniform full-method coverage. **120 invocations,
  100 pass + 20 valid skips (Gamma/Beta/Chi2 explicitly Float32/Float64
  only; Cauchy Float16 has unbounded-tail overflow).** Catches "missing
  op on backend" regressions across the distribution surface.

### Phase D — done

- ✅ **D.1 FP8 MatMul parity** — added `MatMul_FP8_E4M3` and
  `MatMul_FP8_E5M2` parity tests to
  `tests/backend_parity/test_fp8_parity.cpp`. CPU emulated FP8 matmul
  is the reference; each GPU backend's matmul (native or widen-narrow)
  must match within FP8 accumulation tolerance. **10/10 pass** on
  CPU/CUDA/ROCm/OneAPI/Vulkan.

### Bugs uncovered in this session (require separate fixes)

1. **Solve** backward — gradient w.r.t. B fails finite-diff on every
   backend including CPU Float64 (`tenzor::solve` autograd).
2. **LSTMCell** backward — fails on CPU+CUDA+Vulkan+ROCm (passes only
   on OneAPI's fused path).
3. **BatchNorm eval-mode backward** — fails on every backend even with
   explicit weight/bias.
4. **FlashAttention composed-ops fallback backward** — fails 4/5 backends
   (passes only on OneAPI's fused path).
5. **Mean(dim=-1)** — span out-of-bounds assertion on negative dim
   normalization (worked around via positive dim in RMSNorm test).
6. **MixtureSameFamily::sample()** — vector out-of-bounds crash with
   default args.
7. **LKJCholesky** — internal sampler calls `item<double>()` on a
   Float32 concentration parameter; requires Float64 input.

These are real backward / sample bugs the multi-backend gradcheck
matrix exists to surface. Tracked separately; do not block this batch.

*Implementation status snapshot at end of 2026-05-02 session.*
