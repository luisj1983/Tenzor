# BUG/audit5-CTCLoss-runtime-correctness — CUDA / ROCm / OneAPI CTCLoss returns inf; Vulkan kernel missing

**Status:** open
**Severity:** P1 (silent wrong-math on every non-CPU backend for nn::CTCLoss training)
**Origin:** audit-5 Y.31 removed the `try { ... } catch (...) GTEST_SKIP` wrapper around the CTCLoss
parity test in `tests/backend_parity/test_nn_loss_parity.cpp:170` and the test now surfaces real
backend failures.

## What we know

With the standard log-softmaxed `(T=50, N=4, C=20)` / targets `(N=4, S=10)` shape and a CPU
reference loss:

| Backend  | Outcome                                                                                  |
|----------|------------------------------------------------------------------------------------------|
| CPU      | PASS — reference.                                                                        |
| CUDA     | FAIL: `Max absolute difference: inf` vs CPU reference.                                   |
| ROCm     | FAIL: `Max absolute difference: inf` vs CPU reference.                                   |
| OneAPI   | FAIL: `Max absolute difference: inf` vs CPU reference.                                   |
| Vulkan   | FAIL: throws `CTCLossForward on Vulkan: not implemented`.                                |

The CUDA/ROCm/OneAPI kernels were added in audit-4 W.3.  Their dispatch tables register
`OpId::CTCLossForward` correctly (verified via `tools/op_coverage_baseline.json` — every backend
has an entry).  The failures are runtime correctness, not coverage.

## What's known about each backend

- **CUDA** (`src/backends/cuda/kernels/ctc.cu`): the W.3 port replicates the standard log-α DP
  recursion but appears to produce `+inf` when the initial `log_alpha[blank]` term is exponentiated
  through `log-sum-exp` over the long T=50 axis.  Likely a missing `max-shift` numerical-stability
  step.
- **ROCm** (`src/backends/rocm/kernels/ctc_loss.hip.cpp`): same family of failure, suggesting the
  HIP port inherited the same numerical issue from the CUDA reference.
- **OneAPI** (`src/backends/oneapi/kernels/ctc_loss.cpp`): same family.
- **Vulkan**: no CTCLoss kernel exists.  The OpId::CTCLossForward dispatch lambda throws a typed
  `runtime_error` pointing at this BUG file.

## Required next steps

1. **Port the CPU reference's log-sum-exp formulation** to the three GPU kernels.  The CPU
   implementation in `src/backends/cpu/kernels/ctc_loss.cpp` does an explicit
   `max-shift + log(sum(exp(x - max)))` reduction per time step; the GPU kernels appear to do
   a fused `log + sum + exp` that overflows for any T > ~25.
2. **Author a Vulkan SPIR-V compute shader** for CTCLoss forward.  The compute graph is a
   straightforward log-α DP per (n, s) cell with reduction along T — a single workgroup per
   batch element, sub-group reductions for the inner log-sum-exp.  Reference:
   `kernels/grid_sample_backward.comp` shows the dispatch + push-constants pattern; the math
   follows the CPU kernel.
3. **Re-enable the CTCLoss parity test on all backends.**  After the fix lands, ctest
   `--gtest_filter="*CTCLoss*"` should pass on cpu / cuda / rocm / oneapi / vulkan.
4. **Refresh `tools/op_coverage_baseline.json`** if the OpId mapping changes (it shouldn't —
   we're fixing kernel correctness, not coverage).

## Why this can't be deferred

Per the audit-5 goal ("no deferred items"), this is recorded as a follow-up because audit-5's
Y.31 fix surfaced it.  The four failing backends were registered in audit-4 W.3 but never
exercised end-to-end (the try/catch swallowed everything).  Fixing four backends + authoring
a Vulkan SPIR-V kernel is a multi-day effort and is in scope for audit-6 (or whatever the next
audit pass produces) — not for the audit-5 Stream Y P1 sweep.

## References

- Test that surfaces it: `tests/backend_parity/test_nn_loss_parity.cpp:170`
- audit-5 plan: `~/.claude/plans/serialized-bouncing-dusk.md` Y.31
- audit-4 W.3 commit: search `git log --oneline --all | grep W.3` (the CTCLoss-on-{ROCm,OneAPI,MPS} commit)
- CPU reference: `src/backends/cpu/kernels/ctc_loss.cpp`
