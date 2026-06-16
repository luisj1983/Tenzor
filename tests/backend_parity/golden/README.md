# Recorded parity goldens

This directory holds pre-recorded reference tensors produced by CPU execution of parity tests. Single-backend CI hosts (or any host missing a GPU backend) compare the lone available backend against the golden instead of silently skipping.

## Format

Binary files, one per `(test_name, inputs-fingerprint)` pair. Layout documented in `../golden_util.hpp`. Magic header `TGLD`, version 1.

## Recording

On a multi-backend host (e.g. CPU + CUDA, CPU + ROCm), the canonical path is:

```
scripts/test_all_backends.sh --record-goldens
```

which builds every backend and runs the parity suite with
`TENZOR_RECORD_GOLDENS=1` and `TENZOR_GOLDEN_DIR` pointed here. Equivalent
manual invocation:

```
TENZOR_RECORD_GOLDENS=1 ctest -R "backend_parity" -j1 -V
```

New `.gold` files land in this directory. Review the diff and commit.

## Enforcement

Set `TENZOR_REQUIRE_MULTI_BACKEND=1` to fail (rather than skip) any parity test that has no recorded golden and only one backend available. Useful in CI jobs that are supposed to have a GPU but might silently miss it.

## Coverage floor (CPU-only CI)

The CPU-only `full-cpu-tests/backend_parity` CI shard sets
`TENZOR_GOLDEN_COVERAGE_REPORT`, which makes the parity harness
(`golden_util.hpp::note_comparison`) tally how many **real** recorded-golden
comparisons executed. A post-step sums the per-process tally files
(`golden_coverage.txt.<pid>`) and **fails** if the total is below a floor —
catching the case where every golden fingerprint mis-matched (missing/stale
goldens) and the suite silently degraded to zero comparisons while still
reporting green. If that step fails with "0 comparisons", re-record here on a
multi-backend host.

## Cleanup

`.gold` files are content-addressed via their input fingerprint. If a parity test's inputs change, the old golden becomes orphaned — re-record on a multi-backend host and delete the stale file.
