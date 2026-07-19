#!/usr/bin/env bash
# Multi-process launcher for real (not single-process-fake-PG) distributed
# correctness tests -- FINDING 21 "plan K3": the multi-process C++ test job
# tests/distributed/test_device_mesh.cpp and friends deferred to but which
# never existed. Launches `world_size` copies of `binary` as separate OS
# processes, each with RANK/WORLD_SIZE/MASTER_ADDR/MASTER_PORT set so
# init_process_group("gloo") (see src/distributed/distributed.cpp) picks them
# up, waits for all of them, and propagates failure if any rank failed.
#
# Usage: run_multirank_test.sh <world_size> <binary> [gtest_args...]
#
# Mirrors tests/python/test_collective_multirank.py's spawn+trampoline
# pattern (see that file's own docstring for why a fresh process per rank is
# required rather than threads: multiple backend .so files load helper
# threads that don't survive a fork).
set -uo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <world_size> <binary> [gtest_args...]" >&2
    exit 2
fi

world_size="$1"; shift
binary="$1"; shift

if [[ ! -x "$binary" ]]; then
    echo "run_multirank_test.sh: '$binary' is not an executable file" >&2
    exit 2
fi

# Dynamic free-port allocation (bind to port 0, read back what the OS
# assigned, close immediately) instead of a fixed port: a fixed port breaks
# under ctest's parallel test execution (-j) when two multi-rank tests run
# concurrently, and leaves stale TIME_WAIT sockets across repeated local runs.
port=$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1]); s.close()')
if [[ -z "$port" ]]; then
    echo "run_multirank_test.sh: failed to allocate a free port" >&2
    exit 2
fi

tmpdir=$(mktemp -d)
cleanup() { rm -rf "$tmpdir"; }
trap cleanup EXIT

pids=()
for ((rank = 0; rank < world_size; rank++)); do
    RANK=$rank WORLD_SIZE=$world_size MASTER_ADDR=127.0.0.1 MASTER_PORT=$port \
        "$binary" "$@" > "$tmpdir/rank_${rank}.log" 2>&1 &
    pids+=("$!")
done

overall_rc=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        overall_rc=1
    fi
done

for ((rank = 0; rank < world_size; rank++)); do
    echo "===== rank $rank (world_size=$world_size) ====="
    cat "$tmpdir/rank_${rank}.log"
    echo
done

exit "$overall_rc"
