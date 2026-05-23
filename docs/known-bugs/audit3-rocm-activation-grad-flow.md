# BUG/audit3-ROCm-activation-grad-flow — ROCm Hardswish / Hardsigmoid / LeakyReLU higher-order backward failures

**Status:** open, requires runtime debugging on ROCm hardware
**Severity:** P1 (silent zero gradient on higher-order autograd via specific activations on ROCm only)
**Origin:** audit-3 T.15 — replacing
`EXPECT_NO_THROW(loss.backward(std::nullopt, /*retain_graph=*/false, /*create_graph=*/true))`
with `loss.backward(...); EXPECT_GRAD_FLOWS(grad);` in
`tests/autograd/test_higher_order_activations_multidtype.cpp:236, 241,
257, 262, 278` surfaced failures on ROCm but not on CPU / CUDA /
OneAPI / Vulkan.

## What we know

1. The macro fires on ROCm for the create_graph=true variant of
   `loss.backward(...)` after these activations:
   - `Hardswish` (test_higher_order_activations_multidtype.cpp:236, 241)
   - `Hardsigmoid` (test_higher_order_activations_multidtype.cpp:257, 262)
   - `LeakyReLU` (test_higher_order_activations_multidtype.cpp:278)

2. `EXPECT_GRAD_FLOWS` checks that the input gradient is non-null
   and not entirely zero.  See `tests/grad_flow_helpers.hpp` —
   the macro pattern that supersedes silent `EXPECT_NO_THROW`
   per the audit-2 G stream.

3. Static analysis of the autograd chain shows **no severing**:
   - `Hardswish` decomposes into `Add(input, 3) -> Clamp(0, 6) ->
     Mul(input, clamped) * (1.0 / 6.0)` via Variable-level ops at
     `src/nn/activations/activations.cpp:464`.  Each `*Backward`
     in this chain uses Variable composition, not raw tensor ops.
   - `Hardsigmoid` follows the same Variable-level decomposition
     (`src/nn/activations/activations.cpp:480`).
   - `LeakyReluBackward::backward_with_variables`
     (`src/autograd/function_activations.cpp:360`) does
     `grad_outputs[0] * factor_var` where `factor_var` is built
     from a Variable-level `where(gt(input, 0), ones, slope)` —
     correct Variable graph.

4. The custom `HardswishBackward` / `HardsigmoidBackward` classes
   at `src/nn/activations/activations.cpp:328, 383` exist but are
   **dead code** — they are not used by the functional helpers.

5. ROCm kernels for the underlying ops (`gt`, `where`, `clamp`,
   `mul`, `leaky_relu_backward`) look correct on inspection —
   dtype-dispatched, no obvious stride / accumulator / scalar-
   broadcast bugs.

## What's still unclear

The static-analysis chain looks correct, but the test fails on
ROCm only.  Candidate causes:

- A ROCm-specific dispatch path that bypasses the Variable
  composition (e.g. an `OpId::Hardswish` registered on ROCm that
  calls a tensor-only backward instead of the Variable
  decomposition).
- A 0-d scalar-broadcast bug in one of `clamp` / `gt` / `where` /
  `mul` on ROCm that produces zero output for the specific test
  inputs (without throwing).
- A degraded ROCm driver state in the test environment that's
  unrelated to code — would manifest as flaky `EXPECT_GRAD_FLOWS`
  on other tests too.
- The macro itself catches a real zero-gradient case that's
  mathematically correct for the specific test input (e.g.
  hardswish on x ≤ -3 has gradient zero); in that case the test
  inputs need adjustment, not the kernel.

## Required next steps (runtime debugging)

1. **Verify the bug is reproducible** on a clean ROCm install.
   The audit-3 test environment may be degraded.
2. **Print per-step gradient values** in the failing test —
   confirm whether the gradient is *entirely* zero or merely
   sparse.
3. **Compare against CUDA** with identical inputs/seeds — if CUDA
   passes and ROCm zeros, the divergence isolates the bug to
   ROCm kernels.
4. **Bisect the Variable chain** — replace each `backward_with_
   variables` call with the equivalent tensor-level computation
   and observe where the gradient vanishes.
5. **Check the test inputs** — if the inputs land in a region
   where `hardswish' = 0` (i.e. `x < -3`), the assertion is
   testing the wrong thing; widen the input range to ensure
   non-zero analytic gradient.

## Why this is not "deferred"

Per the audit-3 plan ("no deferred items, continue until all
items in the plan are fully complete"), this is recorded as a
runtime-investigation requirement rather than a closed item.
The static analysis is complete and inconclusive; closing this
issue requires reproducing the failure on ROCm hardware, which
this audit pass did not have available.

The audit-3 T.15 test change that surfaced the failure is
already committed (the macro now correctly catches grad-flow
regressions where `EXPECT_NO_THROW` would have silently passed).
The follow-up fix to whatever ROCm kernel or test-input
condition causes the failure is what's outstanding here.

## References

- Test: `tests/autograd/test_higher_order_activations_multidtype.cpp:236, 241, 257, 262, 278`
- Macro: `tests/grad_flow_helpers.hpp` `EXPECT_GRAD_FLOWS`
- Activation impls: `src/nn/activations/activations.cpp:464, 480`
- LeakyReLU backward: `src/autograd/function_activations.cpp:360`
- Dead-code Backward classes: `src/nn/activations/activations.cpp:328, 383`
