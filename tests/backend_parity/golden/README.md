# Recorded parity goldens

This directory holds pre-recorded reference tensors produced by CPU execution of parity tests. Single-backend CI hosts (or any host missing a GPU backend) compare the lone available backend against the golden instead of silently skipping.

## Format

Binary files, one per `(test_name, inputs-fingerprint)` pair. Layout documented in `../golden_util.hpp`. Magic header `TGLD`, version 1.

## Recording

On a multi-backend host (e.g. CPU + CUDA, CPU + ROCm):

```
TENZOR_RECORD_GOLDENS=1 ctest -R "backend_parity" -j1 -V
```

New `.gold` files land in this directory. Review the diff and commit.

## Enforcement

Set `TENZOR_REQUIRE_MULTI_BACKEND=1` to fail (rather than skip) any parity test that has no recorded golden and only one backend available. Useful in CI jobs that are supposed to have a GPU but might silently miss it.

## Cleanup

`.gold` files are content-addressed via their input fingerprint. If a parity test's inputs change, the old golden becomes orphaned — re-record on a multi-backend host and delete the stale file.
