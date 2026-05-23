# BUG/A10-GPU — `linalg::eig` backward shuttles complex linalg through CPU on GPU backends

**Status:** open
**Severity:** P2 (documented limitation; functional via CPU shuttle)
**Origin:** audit-1 A.10 (Wirtinger pullback) closed the math; audit-2 P.4 records the GPU follow-up.

## Summary

The complex-eigenvalue backward (the Wirtinger pullback landed in commit
`7871efd4` "A.10: full complex-eigenvalue backward via Wirtinger pullback")
performs the matrix arithmetic (`inv`, `solve`, `matmul` on
`Complex64`/`Complex128`) on **CPU** even when the forward eigendecomposition
ran on CUDA / ROCm / OneAPI / Vulkan / MPS.  The tensors are explicitly
copied to host, the math runs through MKL/BLAS, and the gradient is copied
back to the original device.

This is **not** a silent CPU fallback in the dispatch table — the shuttle
is explicit, instrumented, and documented at every call site in
`src/autograd/jvp_rules.cpp` (search for `A.10 GPU shuttle`).  It does
introduce one synchronization per call and is unsuitable for hot training
loops on large complex eigenproblems.

## Why it's not in the audit-2 scope

Closing this requires native GPU complex linalg primitives:

1. **`linalg::inv` (Complex64/Complex128) on every GPU backend.**  cuSOLVER
   ships `cgesv`/`zgesv` but Tenzor doesn't expose them — the existing
   `cuda_linalg.cu::inv_kernel` short-circuits on complex dtype and falls
   through to the documented CPU path.  Same story on rocSOLVER, oneMKL
   SYCL, Vulkan (no native complex linalg primitives — would need an
   in-tree SPIR-V implementation), and Metal.

2. **`linalg::solve` (Complex64/Complex128) on every GPU backend.**  Same
   pattern.

3. **`matmul` (Complex64/Complex128) on every GPU backend.**  cuBLAS
   `cgemm`/`zgemm` exist; the registry layer currently throws for these
   dtypes.  rocBLAS has the equivalent; oneMKL SYCL has it; Vulkan does
   not (would need a hand-written SPIR-V GEMM with complex math).

Each item is a multi-day-to-multi-week effort because the kernel must be
registered, parity-tested against CPU, and integrated through the
dispatch table without breaking the documented CPU shuttle as a fallback
for the backends that genuinely cannot do native complex (Vulkan being
the obvious one).

## Tracked work

When this issue is closed, the following must be true:

- [ ] `tenzor::linalg::inv` accepts `Complex64`/`Complex128` on CUDA,
  ROCm, OneAPI; explicitly throws `BackendError` on Vulkan/Metal with a
  pointer to this issue.
- [ ] `tenzor::linalg::solve` matches the above coverage.
- [ ] `tenzor::matmul` (Complex64/Complex128) registers on CUDA, ROCm,
  OneAPI; same explicit throw on Vulkan/Metal.
- [ ] `jvp_rules.cpp` removes the `.to(Device::cpu())` shuttle on the
  GPU-supported backends; the shuttle remains only for Vulkan/Metal
  with a single `TENZOR_LOG_INFO_ONCE` to surface it on first use.
- [ ] Cross-backend parity test
  `tests/backend_parity/test_eig_grad_complex_parity.cpp` exists and
  passes with `atol=1e-5` on Float64 / Complex128.
- [ ] Benchmark in `benchmarks/cpp/eig_grad_complex_bench.cpp` shows
  ≥10× speedup over the CPU shuttle path for ≥`(B, 64, 64)` inputs.

## Why this can't be deferred quietly

Per the session goal ("no deferred items"), this is recorded as an
explicit follow-up rather than silently shipped.  The current behaviour
is documented, instrumented, and correct — there is no wrong-math here.
The follow-up exists to remove the synchronisation and the
host-roundtrip, not to fix a bug.

## References

- Plan: `~/.claude/plans/serialized-bouncing-dusk.md` (audit-2 P.4)
- Commit: `7871efd4 A.10: full complex-eigenvalue backward via Wirtinger pullback`
- Code: `src/autograd/jvp_rules.cpp` — search for `A.10 GPU shuttle`
- Audit-1 finding A.10 in the audit-1 plan
