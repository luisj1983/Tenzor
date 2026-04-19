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

## High-value candidates missing a companion

These tests cover operations / layers / autograd that should plausibly run
on every backend and every supported dtype. They are outside the
`backend_parity/`, `jit/`, `backends/`, `distributed/`, `serving/`,
`lite/`, `nested/`, `utils/`, and `examples/` subtrees that are either
backend-agnostic infrastructure or already parity-parameterized.

| File | Notes |
|------|-------|
| `tests/nn/optim/test_adamw.cpp` | Optimizer — step/param update should be exercised per backend/dtype. |
| `tests/test_autograd_transform.cpp` | Autograd transforms — critical for create_graph. |
| `tests/nn/quantization/test_awq_quantizer.cpp` | INT4/INT8 quantization. |
| `tests/test_ciou_loss.cpp` | Vision loss — float32/float16 |
| `tests/test_contiguous_fix.cpp` | Stride-pattern regression — especially valuable cross-backend. |
| `tests/integration/test_cross_backend.cpp` | Already cross-backend conceptually; confirm fixture use. |
| `tests/nn/quantization/test_gptq_quantizer.cpp` | |
| `tests/test_grad_accumulation.cpp` | Multi-step accumulation is a high-signal correctness test. |
| `tests/autograd/test_higher_order_contract.cpp` | create_graph=true paths. |
| `tests/autograd/test_higher_order_stubs_regression.cpp` | |
| `tests/autograd/test_inference_mode_guard.cpp` | Per-backend guard semantics. |
| `tests/test_linear_reshape_integration.cpp` | |
| `tests/test_mask_rcnn_losses.cpp` | |
| `tests/test_minimal_training.cpp` | End-to-end smoke; per-backend run is small/fast. |
| `tests/ops/test_new_ops.cpp` | New op smoke — add dtype axis. |
| `tests/integration/test_nn.cpp` | |
| `tests/nn/quantization/test_observers_extended.cpp` | |
| `tests/test_quantization_conversion.cpp` | |
| `tests/autograd/test_strict_linalg_grad.cpp` | |
| `tests/integration/test_training.cpp` | |
| `tests/integration/test_training_loops.cpp` | |

## Known-intentional (kept CPU-only, no action)

- Gradcheck (`test_gradcheck_*.cpp`) — finite-difference is O(N·inputs) and
  too slow on GPU; parity coverage comes from the per-op parity tests.
- SIMD / kernel-specific files — by definition CPU-only.
- Allocator / memory / pinned / caching infrastructure — device-agnostic.
- JIT / nested / lite / serving / distributed / utils — either
  parity-parameterized themselves or not per-backend.

## How to add a companion

Copy `tests/nn/layers/test_batchnorm1d_multidtype.cpp` as a template,
replace the operation under test, keep the `MultiBackendDTypeTest` fixture
plus `INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(YourSuite);` at the bottom,
and register in the nearest `CMakeLists.txt`. Include backward coverage
(gradient checks) where the op is differentiable — forward-only companions
hide the known Float32-accumulator / autograd-graph-drop bug patterns.
