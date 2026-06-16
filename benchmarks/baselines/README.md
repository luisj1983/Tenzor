# Version-controlled benchmark baselines

This directory holds the **committed** benchmark baselines that the nightly
`benchmarks` CI job compares against. Each file is named after the host that
produced it:

```
benchmarks/baselines/<host>.json
```

where `<host>` is the lowercased `runs-on` runner identity used by the
benchmark job (for GitHub-hosted Ubuntu runners this is `ubuntu-24.04`).

## Format

The file is the merged `all.json` produced by `scripts/ci_benchmark.sh`
(`{ "timestamp": ..., "suites": [ { "suite_name": ..., "benchmarks": [...] } ] }`).
`scripts/compare_benchmarks.py` reads it directly.

## Why version-controlled

The old flow uploaded results as a 90-day ephemeral artifact and compared the
current run against "whatever the last run uploaded". That makes the gate
non-reproducible (depends on artifact retention) and impossible to review in a
PR diff. Committing the baseline makes every threshold change show up in code
review.

## Update flow (bot / PR)

The baseline is **only** updated through a reviewed PR — never auto-committed
on the main branch by CI. The intended flow:

1. The nightly `benchmarks` job runs `scripts/compare_benchmarks.py` against
   the committed `benchmarks/baselines/<host>.json` and **fails** on a >7%
   regression (default `FAILURE_THRESHOLD`, override with `--threshold`).
2. When a regression is *intended* (e.g. a correctness fix that costs perf) or
   an *improvement* should become the new floor, a maintainer (or a scheduled
   bot) regenerates the baseline and opens a PR:

   ```bash
   bash scripts/ci_benchmark.sh
   cp build/benchmark_results/all.json benchmarks/baselines/ubuntu-24.04.json
   git checkout -b chore/refresh-bench-baseline
   git add benchmarks/baselines/ubuntu-24.04.json
   git commit -m "chore: refresh benchmark baseline (ubuntu-24.04)"
   gh pr create --fill
   ```

3. The PR diff shows the per-benchmark delta; a reviewer approves the new floor.

A scheduled "refresh benchmark baseline" bot can automate step 2 by opening
that PR automatically; it must NOT push directly to a protected branch.
