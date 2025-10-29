#!/bin/bash
cd /home/lee/Projects/Tenzor/bin

# Run rank 0 in background
RANK=0 WORLD_SIZE=2 MASTER_ADDR=localhost MASTER_PORT=29500 ./test_distributed --gtest_filter="GlooBackendTest.*" > /tmp/rank0.log 2>&1 &
PID0=$!

# Small delay to let rank 0 initialize
sleep 1

# Run rank 1 in background
RANK=1 WORLD_SIZE=2 MASTER_ADDR=localhost MASTER_PORT=29500 ./test_distributed --gtest_filter="GlooBackendTest.*" > /tmp/rank1.log 2>&1 &
PID1=$!

# Wait for both to complete
wait $PID0
EXIT0=$?
wait $PID1
EXIT1=$?

echo "=== RANK 0 OUTPUT ==="
cat /tmp/rank0.log
echo ""
echo "=== RANK 1 OUTPUT ==="
cat /tmp/rank1.log
echo ""
echo "=== EXIT CODES ==="
echo "Rank 0: $EXIT0"
echo "Rank 1: $EXIT1"

if [ $EXIT0 -eq 0 ] && [ $EXIT1 -eq 0 ]; then
    echo "✅ ALL TESTS PASSED"
    exit 0
else
    echo "❌ TESTS FAILED"
    exit 1
fi
