# Multi-dtype / Multi-backend Coverage Gaps

Generated as part of the test audit (2026-04-19). The main focused-bug pass
does not write new test companions; this file tracks the gap list so the
follow-up is not lost.

## Convention

A `test_foo.cpp` that tests operations, autograd, or nn layers is expected
to have a sibling `test_foo_multidtype.cpp` that exercises the same
surface via `MultiBackendDTypeTest` (see `tests/multi_backend_dtype_fixture.hpp`)
— 5 backends × {Float32, Float64, Float16}. The plain file is either a
subset (forward-only on CPU) or covers behaviour that is dtype-agnostic.

## Closed candidates (companion landed)

- `tests/autograd/test_higher_order_stubs_regression.cpp` — closed. New
  companion `tests/autograd/test_higher_order_stubs_regression_multidtype.cpp`
  inherits `MultiBackendDTypeTest`, re-runs the higher-order-stub passthrough
  surface (Conv2d, BatchNorm2d, LayerNorm, MaxPool2d, AvgPool2d, Dropout/eval,
  RReLU, CTCLoss, MultiLabelMarginLoss, Embedding) across {Float32, Float64,
  Float16} × 5 backends, asserting the stub still passes through (no throw,
  finite grad) in non-Float32 dtypes. Verified CPU + CUDA (30/30 each).
- `tests/test_linear_reshape_integration.cpp` — closed. New companion
  `tests/test_linear_reshape_integration_multidtype.cpp` inherits
  `MultiBackendDTypeTest`, re-runs the Linear-fed-by-reshape/permute-chain
  integration surface (LinearWithReshapeInput, LinearWithPermuteInput,
  MultipleReshapeOps) across {Float32, Float64, Float16} × 5 backends with
  EXPECT_GRAD_FLOWS backward coverage. Verified CPU + CUDA (9/9 each).
- `tests/autograd/test_strict_linalg_grad.cpp` — closed. New companion
  `tests/autograd/test_strict_linalg_grad_multidtype.cpp` inherits
  `MultiBackendDTypeTest`, re-runs the LDL-factor backward contract
  (meaningful gradient, STRICT_LINALG_GRAD no-op, pivoting throws
  NonDifferentiable) across {Float32, Float64, Float16} × 5 backends.
  Float16/BFloat16 skip categorically (DtypeUnsupportedOnBackend — no
  backend registers a Float16 linalg factorization kernel); the real
  coverage is Float32 + Float64 exercising the closed-form LDL adjoint on
  GPU backends. Verified CPU + CUDA (8 passed + 4 F16 skipped each).
- `tests/nn/quantization/test_awq_quantizer.cpp` — closed audit-10 OO.19. New
  companion `tests/nn/quantization/test_awq_quantizer_multidtype.cpp`
  inherits `MultiBackendDTypeTest`, sweeps {Float16, BFloat16, Float32} ×
  5 backends, asserts compute_act_scales shape/non-negativity and that
  quantize_layer round-trips for the dtype-converted weight.
- `tests/nn/quantization/test_gptq_quantizer.cpp` — closed audit-11 RR.21.
  New companion `tests/nn/quantization/test_gptq_quantizer_multidtype.cpp`
  mirrors the OO.19 AWQ pattern: `MultiBackendDTypeTest`,
  {Float16, BFloat16, Float32} × 5 backends, exercises
  `compute_hessian` shape/symmetry and the `quantize_layer` packed-INT4
  round-trip on dtype-converted weights.
- `tests/test_mask_rcnn_losses.cpp` — closed. New companion
  `tests/test_mask_rcnn_losses_multidtype.cpp` inherits
  `MultiBackendDTypeTest`, re-runs all 9 Mask R-CNN head-loss tests
  (RPN/ROI/mask finiteness + range, end-to-end loop, gradient-flow
  regression guard) across {Float32, Float64, Float16} × 5 backends with
  the full detector dtype-converted via `convert_model`. Image sizes and
  box coords are kept identical to the plain file (it already proved those
  fit every backend at F32; GradientFlow keeps 512×512). Float16/BFloat16
  skip categorically (NumericalDivergence — end-to-end not validated in
  half precision); Float64 skips on non-CPU backends (NumericalDivergence —
  2× footprint exceeds GPU memory); Float64 on CPU is retained. Loss
  scalars read back via a dtype-safe helper (cast to Float32 before
  `.item<float>()`). Verified F32 CPU, F64 CPU, F16 cuda skip, and
  GradientFlow cuda F32.
- `tests/nn/quantization/test_observers_extended.cpp` — closed. New companion
  `tests/nn/quantization/test_observers_extended_multidtype.cpp` inherits
  `MultiBackendDTypeTest`, re-runs all 23 extended-observer tests
  (KLDivergence, Percentile, MSE, per-channel MinMax: construction,
  observe-does-not-crash, qparam validity, multi-observe, reset,
  tighter-percentile) across {Float32, Float64, Float16} × 5 backends with
  the observed input created in the test dtype on the test device. Dtype
  coverage is split by what each observer supports: KL & MinMax upcast in
  observe() so they run the full {F32,F64,F16} sweep (new coverage = the
  device-side `.to(Float32)` cast kernel for F16/F64 inputs); Percentile &
  MSE are test-only Float32 observers (no upcast — `.data<float>()` on a
  non-F32 tensor) so their observe-based tests skip non-F32 with
  DtypeUnsupportedOnBackend; the pure-construction tests are dtype-orthogonal
  and run on every combo. Verified CPU (51 passed / 18 skipped / 0 failed
  across 3 dtypes) and cuda F16 (KL passes, Percentile skips).

## High-value candidates missing a companion

These tests cover operations / layers / autograd that should plausibly run
on every backend and every supported dtype. They are outside the
`backend_parity/`, `jit/`, `backends/`, `distributed/`, `serving/`,
`lite/`, `nested/`, `utils/`, and `examples/` subtrees that are either
backend-agnostic infrastructure or already parity-parameterized.

| File | Notes |
|------|-------|
| `tests/test_quantization_conversion.cpp` | TODO: add `test_quantization_conversion_multidtype.cpp` using `MultiBackendDTypeTest`. |
| `tests/integration/test_training_loops.cpp` | TODO: add `test_training_loops_multidtype.cpp` using `MultiBackendDTypeTest`. |

<!--
audit-11 RR.21: removed `test_ciou_loss.cpp`, `test_contiguous_fix.cpp`,
`test_inference_mode_guard.cpp`, and `test_gptq_quantizer.cpp` from this
candidate table — the first three are already documented as
KNOWN-INTENTIONAL below (so they were self-contradicting); the fourth
(gptq) is now closed by `test_gptq_quantizer_multidtype.cpp`.
-->


<!--
audit-8 II.18: the 5 entries below were marked "audit-6 CC.21: moved to
KNOWN-INTENTIONAL" inline in this table and have been moved out — their
canonical home is the machine-readable KNOWN-INTENTIONAL block further
down. The CC.21 rationale is preserved verbatim per-entry there.

  - tests/nn/optim/test_adamw.cpp
  - tests/test_autograd_transform.cpp
  - tests/test_grad_accumulation.cpp
  - tests/autograd/test_higher_order_contract.cpp
  - tests/test_minimal_training.cpp
-->


## Known-intentional (kept CPU-only, no action)

- Gradcheck (`test_gradcheck_*.cpp`) — finite-difference is O(N·inputs) and
  too slow on GPU; parity coverage comes from the per-op parity tests.
- SIMD / kernel-specific files — by definition CPU-only.
- Allocator / memory / pinned / caching infrastructure — device-agnostic.
- JIT / nested / lite / serving / distributed / utils — either
  parity-parameterized themselves or not per-backend.

## Machine-readable known-intentional list (audit-3 T.14)

`tools/check_multidtype_coverage.py` reads the block below — every line
between the two markers is treated as a path (relative to the repo root)
that is known to lack a `*_multidtype.cpp` companion **on purpose**. Any
non-multidtype `test_*.cpp` that does not appear here AND has no sibling
multidtype companion is a NEW gap and fails CI.

To accept a new gap as intentional, append it to this block with a
one-line trailing comment justifying CPU-only coverage.

<!-- KNOWN-INTENTIONAL-START -->
# audit-3 high-value candidates (entries from the table above) — still
# aspirational, currently grandfathered so the ratchet passes. Move out of
# this block once a companion lands.
#
# audit-6 CC.21 (2026-05-25): the five highest-priority entries below
# (test_adamw, test_grad_accumulation, test_higher_order_contract,
# test_minimal_training, test_autograd_transform) reviewed in this pass.
# Justifications appended inline; remaining entries unchanged.
tests/nn/optim/test_adamw.cpp                # audit-6 CC.21: justified as out of scope for audit-6 — optimizer state buffers live on CPU regardless of param device; cross-backend AdamW parity is covered by backend_parity/ training-loop tests. File doc-comment already documents CPU-only-by-design.
tests/test_autograd_transform.cpp            # audit-6 CC.21: justified as out of scope for audit-6 — already runs as multi-backend (TEST_P + BackendTest fixture). Adding a dtype axis would re-test the autograd-graph-stability surface, not new numerics; covered by reshape/permute parity tests in tests/backend_parity/.
tests/test_ciou_loss.cpp                     # audit-7 FF.31: justified — CIoU is a detection loss with a known PyTorch-equivalent CPU reference; cross-backend correctness comes from the per-op (matmul, exp, log) parity tests its forward path uses. Adding a 5-backend × 3-dtype sweep would re-test those ops, not the loss math.
tests/test_contiguous_fix.cpp                # audit-7 FF.31: justified — regression test for a stride/contiguity bug in tensor.slice() that surfaces in scalar arithmetic. The fix lives in backend-agnostic Tensor layout code; per-backend stride parity is already covered by tests/backend_parity/test_stride_parity.cpp.
tests/integration/test_cross_backend.cpp     # audit-9 LL.16: confirmed cross-backend via TEST_P fixture (CrossBackendTest : public BackendTest). Fixture sweeps all built backends; dtype axis is intentionally CPU-anchored since the test verifies device-transfer round-trips and per-op consistency, not numeric parity across dtypes.
# audit-11 RR.21: test_gptq_quantizer.cpp removed from this list — closed
# by tests/nn/quantization/test_gptq_quantizer_multidtype.cpp.
tests/test_grad_accumulation.cpp             # audit-6 CC.21: justified as out of scope for audit-6 — already runs as multi-backend (TEST_P + BackendTest fixture). Tests GradientAccumulator state-machine (step count, flush, reset) which is dtype-orthogonal; numeric correctness comes from the wrapped optimizer's parity tests.
tests/autograd/test_higher_order_contract.cpp # audit-6 CC.21: justified as out of scope for audit-6 — engine-level contract test for HigherOrderGradMode stub backwards (Error throws, Warn logs); CPU-only by design per file doc-comment, no backend or dtype variance.
tests/autograd/test_inference_mode_guard.cpp # audit-7 FF.31: justified — file header already documents "CPU-only infrastructure tests verifying RAII guard semantics, nesting behaviour, and effect on Variable grad_fn attachment". The InferenceModeGuard / NoGradGuard contract is dtype-orthogonal and the guard mechanism lives in autograd/variable.cpp, not in any backend kernel.
tests/test_minimal_training.cpp              # audit-6 CC.21 + audit-7 FF.31 re-review: confirmed as KNOWN-INTENTIONAL — single-run NaN-debug smoke test for an Adam training loop; cross-backend training is covered by integration/test_training.cpp and the backend_parity/ training-loop tests. No action needed.
tests/ops/test_new_ops.cpp                   # audit-9 LL.16: confirmed cross-backend via TEST_P fixture (NewOpsTest : public MultiBackendDTypeTest). Already parametrised over 5 backends × 3 dtypes.
tests/integration/test_nn.cpp                # audit-9 LL.16: confirmed cross-backend via TEST_P fixture (NNTest : public BackendTest). Integration coverage parametrised over backends; per-layer numeric parity comes from tests/nn/layers/*_multidtype.cpp companions.
tests/test_quantization_conversion.cpp
tests/integration/test_training.cpp          # audit-9 LL.16: confirmed cross-backend via TEST_P fixture (TrainingTest : public BackendTest). Integration coverage parametrised over backends; per-op numeric parity comes from backend_parity/ tests.
tests/integration/test_training_loops.cpp

# Grandfathered pre-existing gaps as of audit-3 baseline (2026-05-22). These
# are either infrastructure / backend-specific / regression-guard files
# whose surface area does not benefit from a 5-backend × dtype sweep, or
# files awaiting a follow-up companion. Adding a companion is welcome; the
# ratchet only blocks NEW non-multidtype tests landing without one.
#
# audit-4 W.27 (2026-05-24): cross-checked all 82 entries against the
# tests/*_multidtype.cpp inventory.  The ratchet script
# tools/check_multidtype_coverage.py matches by *exact* base name
# (test_foo.cpp ↔ test_foo_multidtype.cpp) — topical matches (e.g.
# test_lstm_proj_size.cpp covered by test_lstm_multidtype.cpp) do not
# satisfy the script.  None of the entries gained an exact-name companion
# since audit-3, so the ratchet is unchanged in this pass; the entries
# stay grandfathered until either an exact-name companion lands or the
# script is taught to consider topical coverage.
tests/autograd/test_anomaly_mode_full.cpp
tests/autograd/test_custom_op_higher_order.cpp
tests/autograd/test_fused_linear_relu_higher_order.cpp
tests/autograd/test_graph_viz.cpp
tests/autograd/test_jvp_rules.cpp
tests/autograd/test_linalg_matrix_norm_higher_order.cpp
tests/autograd/test_linalg_vector_norm_higher_order.cpp
tests/autograd/test_matrix_norm_backward.cpp
tests/autograd/test_profiler.cpp
tests/autograd/test_requires_grad_and_detach.cpp
tests/autograd/test_upsample_bilinear_higher_order.cpp
tests/backend/test_oneapi_backend.cpp
tests/backend/test_oneapi_caching_allocator.cpp
tests/backend/test_rocm_caching_allocator.cpp
tests/backend/test_vulkan_caching_allocator.cpp
tests/core/test_distributions_gap_fill.cpp
tests/core/test_offload_engine.cpp
tests/core/test_offload_engine_diagnostic.cpp
tests/core/test_offload_engine_stress.cpp
tests/core/test_transfer_benchmark.cpp
tests/core/test_transfer_engine.cpp
tests/core/test_vulkan_flash_attention_parity.cpp
tests/integration/test_attention_autograd.cpp
tests/integration/test_attention_contract.cpp
tests/integration/test_attention_parity.cpp
tests/integration/test_attention_philox_multibackend.cpp
tests/integration/test_cuda_training.cpp
tests/integration/test_debug_mlp.cpp
tests/integration/test_distributed.cpp
tests/integration/test_eager_parity.cpp
tests/integration/test_model_backend_parity.cpp
tests/integration/test_multi_gpu.cpp
tests/integration/test_program_export.cpp
tests/integration/test_quantization_e2e.cpp
tests/nn/layers/test_interpolate_backward.cpp
tests/nn/optim/test_optim_grad_flow.cpp
tests/nn/optim/test_zero_partitioning.cpp
tests/nn/test_fsdp2.cpp
tests/nn/test_torch_pickle.cpp
tests/sparse/test_sparse_triangular_solve.cpp
tests/test_backend_parity_example.cpp
tests/test_checkpoint_leaf_fix.cpp
tests/test_convtranspose1d_dilation.cpp
tests/test_cuda_cat.cpp
tests/test_cuda_scalar_debug.cpp
tests/test_flash_attention_dropout_backward.cpp
tests/test_function_op_id.cpp
tests/test_graph_optimizer_op_id.cpp
tests/test_grid_sample_bicubic.cpp
tests/test_group_instance_norm_f64_precision.cpp
tests/test_layer_norm_f64_precision.cpp
tests/test_layer_norm_variance_stability.cpp
tests/test_lazy_tensor_backward.cpp
tests/test_median_f16_bf16.cpp
tests/test_non_differentiable_ops.cpp
tests/test_onnx_import.cpp
tests/test_optimizer_step_hooks.cpp
tests/test_param_group_contract.cpp
tests/test_phase11_backends.cpp
tests/test_slice_backend_parity.cpp
tests/test_slice_debug.cpp
tests/test_tensor_lifetime.cpp
tests/test_training_callbacks_lifecycle.cpp
tests/test_vmap_opid_dispatch.cpp
tests/test_vulkan_complete_ops.cpp
tests/test_vulkan_layer_norm_saved_stats.cpp
tests/unit/test_bf16_mkl_gemm.cpp
tests/unit/test_cat_slice_negative_dim.cpp
tests/unit/test_color_jitter.cpp
tests/unit/test_complex_matmul_mkl.cpp
tests/unit/test_cpu_kernels.cpp
tests/unit/test_creation_dtype_coverage.cpp
tests/unit/test_deterministic_mode.cpp
tests/unit/test_distributions_samplers.cpp
tests/unit/test_fill_kernel_dtype_coverage.cpp
tests/unit/test_foreach_ops.cpp
tests/unit/test_fp8_quantize_grad.cpp
tests/unit/test_fused_layer_norm_backward_f64.cpp
tests/unit/test_fusion_optimizer_math.cpp
tests/unit/test_int8_mkl_gemm.cpp
tests/unit/test_int8_overflow_semantics.cpp
tests/unit/test_linalg_complex.cpp
tests/unit/test_lstm_cuda_forward_regression.cpp
tests/unit/test_lstm_proj_size.cpp
tests/unit/test_masked_tensor.cpp
tests/unit/test_math_dtype_coverage.cpp
tests/unit/test_metrics_extended.cpp
tests/unit/test_mkl_vml.cpp
tests/unit/test_model_hub.cpp
tests/unit/test_norm_eps_cache.cpp
tests/unit/test_oneapi_backend_loading.cpp
tests/unit/test_oneapi_operations.cpp
tests/unit/test_onednn_format_any.cpp
tests/unit/test_onnx_audit_fixes.cpp
tests/unit/test_onnx_export.cpp
tests/unit/test_onnx_exporter_coverage.cpp
tests/unit/test_onnx_roundtrip.cpp
tests/unit/test_philox_dropout.cpp
tests/unit/test_philox_reproducibility.cpp
tests/unit/test_reduction_half_dtype.cpp
tests/unit/test_reduction_numerical_fixes.cpp
tests/unit/test_reductions_dtype_coverage.cpp
tests/unit/test_scatter_add_dtype_coverage.cpp
tests/unit/test_simd.cpp
tests/unit/test_softmax_stride_audit.cpp
tests/unit/test_strided_fill_dtype_coverage.cpp
# audit-6 BB.26 — dtype-orthogonal regression coverage from the audit-5
# closure pass; the companions below test slice-semantics / multi-device
# wiring, not dtype-specific numerics.
tests/autograd/test_chunk_split.cpp                   # dtype-agnostic: tests chunk/split slice semantics on empty/non-empty dims, not arithmetic
tests/autograd/test_save_backward_multi_device.cpp    # multi-device-only by construction (GPU↔CPU traversal); dtype-orthogonal

# Release-prep S-stream regression suites (added 2026-06). Every entry below
# already runs cross-backend via the BackendTest TEST_P fixture (all 5
# backends) — the multidtype convention's *backend* axis is already covered.
# The per-file note explains why a separate {F32,F64,F16} *_multidtype.cpp
# companion would add no coverage (dtype-orthogonal behaviour, the dtype axis
# already exercised in-file, or an inapplicable F16 path).
tests/autograd/test_typed_stub_backwards.cpp   # cross-backend (BackendTest); validates the differentiability *contract* of 5 formerly-NonDifferentiable typed stubs (ROIAlign/DeformableConv2d/MelScale/DCT/MFCC) — asserts grads finite & non-zero. dtype-orthogonal; per-dtype numerics of the underlying ops are covered by their own op tests.
tests/nn/test_optimizer_holes.cpp              # cross-backend (BackendTest); four dtype-orthogonal regression items (LBFGS state-dict key validation, LazyLinear pre-forward .to(device), HRM participation-ratio rank structure, Adam sparse-grad densify dispatch). Same rationale as test_adamw — optimizer/diagnostic state is dtype-agnostic.
tests/nn/test_parametrize_lifetime.cpp         # no tensors/backends/dtypes — pure Module UID / parametrize-registry lifetime test (heap-address reuse must not leak parametrization state). A dtype sweep is meaningless.
tests/nn/test_quantization_s20.cpp             # cross-backend (BackendTest); INT8 quantization fixes (HistogramObserver re-bin, QuantizedConv1d real-INT8, QuantStub Q/DQ + STE backward). The relevant dtype axis is the INT8 quant path itself; a {F32,F64,F16} float sweep does not apply. Same rationale as test_quantization_conversion.
tests/nn/test_s5_surgical_fixes.cpp            # cross-backend (BackendTest); autograd-graph-severance / diagnostic regressions (DataParallel gather grad-chain, PANet single-input diagnostic, DeepLabV3+ decoder non-identity). Structural/contract, dtype-orthogonal.
tests/nn/test_severance_sweep.cpp              # cross-backend (BackendTest); grad-flow severance sweep — each layer asserts EXPECT_GRAD_FLOWS after a forward/backward round-trip. Tests graph connectivity, not per-dtype numerics; dtype-orthogonal by construction.
tests/ops/test_griffin_lim.cpp                 # cross-backend (BackendTest); already exercises BOTH Float32 and Float64 (AcceptsFloat64Magnitude round-trips F64 through the algorithm). Float16 is not a supported dtype for iterative STFT/ISTFT phase reconstruction, so a {F32,F64,F16} companion adds nothing.
tests/ops/test_linalg_norm_ords.cpp            # cross-backend (BackendTest); already covers Float32 + Float64 explicitly (Frobenius F32/F64, L0 exact-count F32/F64, F64 gradchecks). The spectral/nuclear ords route through SVD, which has no Float16 path, so the only missing companion axis (F16) is inapplicable.
tests/test_fused_conv_activation_dtype.cpp     # THIS FILE IS the dtype-coverage test: it sweeps Float16 AND BFloat16 against a Float32 reference for fused conv+{ReLU,Sigmoid,Tanh,Swish} across all backends (BackendTest). A separate *_multidtype.cpp companion would be fully redundant.
tests/test_math_half_dispatch.cpp              # THIS FILE IS the dtype-coverage test: regression net asserting ~40 math ops accept Float16 AND BFloat16 (vs a Float32 reference) across all backends (BackendTest). The half-precision dtype axis is the whole point; a companion would duplicate it.
<!-- KNOWN-INTENTIONAL-END -->

## How to add a companion

Copy `tests/nn/layers/test_batchnorm1d_multidtype.cpp` as a template,
replace the operation under test, keep the `MultiBackendDTypeTest` fixture
plus `INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(YourSuite);` at the bottom,
and register in the nearest `CMakeLists.txt`. Include backward coverage
(gradient checks) where the op is differentiable — forward-only companions
hide the known Float32-accumulator / autograd-graph-drop bug patterns.
