# Pre-Hardening Audit Baseline

Snapshot captured at the start of the v1 pre-release hardening effort
(see `/home/lee/.claude/plans/toasty-riding-sky.md`).

## Files

- **`op_coverage_baseline_pre_hardening.txt`** — Output of `bin/op_coverage_report`
  showing per-backend op registration counts at the start of the effort.
  Counts: CPU 317/317, CUDA/ROCm/Vulkan/OneAPI 315/317 each (missing
  `LinalgLU` and `LinalgLUSolve`).

- **`op_coverage_matrix_pre_hardening.txt`** — Output of
  `scripts/op_coverage_matrix.py`, the static-analysis macro-aware view of
  the same data. Static counts differ slightly from runtime counts because
  the matrix script counts all source registrations regardless of whether
  the kernel is actually a CPU fallback.

- **`cpu_fallback_burndown.txt`** — Inventory of every `.cpu()`,
  `Device::cpu()`, and `Device::CPU` reference inside
  `src/backends/{cuda,rocm,vulkan,oneapi}/`. **78 sites total**, distributed:
  - CUDA   14
  - ROCm   19
  - Vulkan 27
  - OneAPI 18

  Target after Phase 4: **0**.

## Progress (post-Phase 4.2)

| Phase | Status | Burndown after |
|---|---|---|
| 0  Pre-flight | ✅ done | 78 |
| 1  Dead code & deprecated | ✅ done | 78 (no fallback work) |
| 2  Tier C one-offs | ✅ done | 76 |
| 3  Tier B special math (15 ops × 4 backends) | ✅ done | 60 |
| 4.1 GridSample/AffineGrid (× 3 backends) | ✅ done | 51 |
| 4.2 Bernoulli/Multinomial/Bucketize/Histogram/CDist (× 3 backends) | ✅ done | 39 |
| 4.3 STFT/ISTFT (× 4 backends) | pending | ~31 expected |
| 4.4 AdvancedIndex/AdvancedIndexPut (× 4 backends) | pending | ~11 expected |
| 5  GPU LinalgLU/LinalgLUSolve | pending | — |
| 6  MPS full implementation | pending | — |
| 7  Sync/perf cleanup + Flash Attention bw fused | pending | — |
| 8  Unimplemented enum entries | pending | — |

**Burndown: 78 → 39 (-39 sites, 50%)** from CPU fallbacks in `src/backends/{cuda,rocm,vulkan,oneapi}/`. Per-backend remaining:
- CUDA: 9 (5 AdvancedIndex + 4 STFT/ISTFT)
- ROCm: 7 (5 AdvancedIndex + 2 STFT — though phases 4.1/4.2 covered some, need to recount after 4.3/4.4)
- OneAPI: 6 (similar)
- Vulkan: 17 (includes 5 special-math metadata sync sites that aren't true fallbacks)

## Phase 4.2 known limitation
Histogram and Multinomial in `vulkan_ops_sampling.cpp` have ~3 `to(Device::cpu())` calls for **single-scalar metadata reads** (CDF total for multinomial; min/max bounds for histogram auto-range). These are NOT compute fallbacks — they read 4 bytes for kernel launch parameters. The grep counts them but they're functionally equivalent to CUDA's `cudaMemcpy(&info, devInfo, ...)` pattern.

## Bonus fix
**Phase 4.2 bonus**: CUDA's `cdist_kernel` had a pre-existing bug — it only handled 2D inputs (P,M)×(R,M) but the test suite expects 3D batched (B,P,M)×(B,R,M). Fixed in `src/backends/cuda/kernels/advanced.cu` and mirrored to ROCm/OneAPI/Vulkan. All 18 CDist tests now pass on CUDA + OneAPI + Vulkan.

## Stale baseline JSON

`tools/op_coverage_baseline.json` was significantly out of date (reported
CPU=283, CUDA=284, ROCm=274, Vulkan=282, OneAPI=279). The CI
`op_coverage_report --check` step was passing but only because the
existing tool flags downward changes, not stale upward drift. The baseline
was refreshed to current runtime state (CPU 317, CUDA/ROCm/Vulkan/OneAPI 315)
at the start of Phase 0 so subsequent phases have a meaningful regression gate.

## Known issues observed during baseline capture

1. **`op_coverage_report --json`** output is contaminated by
   `tenzor::initialize()` log messages on stdout. Workaround: pipe through
   `sed -n '/^{/,$p'`. Proper fix (library init logging to stderr) is
   not in scope for this hardening effort but worth noting.

2. **Dispatch table double-registration warnings** appear during ROCm
   backend load:
   `[dispatch_table] WARNING: OpId N already registered in a different kernel array`
   for ~150 OpIds. This is real and worth investigating in Phase 7 (sync /
   perf cleanup) — possibly the result of two kernel arrays being
   constructed independently. Not a release blocker but a code-cleanliness
   issue.
