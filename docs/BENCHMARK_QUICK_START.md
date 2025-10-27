# Tenzor Benchmark Suite - Quick Start Guide

## Build Instructions

```bash
cd /home/lee/Projects/Tenzor/build

# Configure with benchmarks enabled
cmake -DTENZOR_BUILD_BENCHMARKS=ON ..

# Build all benchmarks
ninja benchmark_ops benchmark_convolutions benchmark_memory

# Or build with make
make benchmark_ops benchmark_convolutions benchmark_memory
```

## Run Benchmarks

```bash
# Set library path
export LD_LIBRARY_PATH=/home/lee/Projects/Tenzor/bin:$LD_LIBRARY_PATH
cd /home/lee/Projects/Tenzor/bin

# Run individual benchmarks
./benchmark_ops              # Matrix operations
./benchmark_convolutions     # Convolution operations  
./benchmark_memory           # Memory operations

# Or use make targets from build directory
cd /home/lee/Projects/Tenzor/build
make run_ops_benchmark
make run_conv_benchmark
make run_memory_benchmark
make run_benchmarks          # Run all
```

## File Locations

### Benchmark Executables
- `/home/lee/Projects/Tenzor/bin/benchmark_ops` (61 KB)
- `/home/lee/Projects/Tenzor/bin/benchmark_convolutions` (53 KB)
- `/home/lee/Projects/Tenzor/bin/benchmark_memory` (67 KB)
- `/home/lee/Projects/Tenzor/bin/bench_fused_ops` (43 KB)

### Source Files
- `/home/lee/Projects/Tenzor/benchmarks/benchmark_ops.cpp`
- `/home/lee/Projects/Tenzor/benchmarks/benchmark_convolutions.cpp`
- `/home/lee/Projects/Tenzor/benchmarks/benchmark_memory.cpp`
- `/home/lee/Projects/Tenzor/benchmarks/CMakeLists.txt`

### Infrastructure
- `/home/lee/Projects/Tenzor/include/tenzor/utils/benchmark.hpp`
- `/home/lee/Projects/Tenzor/src/utils/benchmark.cpp`

### Documentation
- `/home/lee/Projects/Tenzor/docs/BENCHMARK_SUITE_COMPLETE.md` (Full report)
- `/home/lee/Projects/Tenzor/docs/BENCHMARK_QUICK_START.md` (This file)

## Performance Targets

| Operation | Target | Competitor |
|-----------|--------|------------|
| MatMul (4096x4096) | < 20ms | PyTorch: 22ms |
| Conv2d (ResNet50) | < 1ms/layer | PyTorch: 1.2ms |
| Backward Pass | < 2x forward | PyTorch: 2.5x |
| Memory Overhead | < 10% | PyTorch: 15% |

## Quick Verification

```bash
# Quick test - should complete in ~30 seconds
cd /home/lee/Projects/Tenzor/bin
LD_LIBRARY_PATH=.:$LD_LIBRARY_PATH timeout 30 ./benchmark_memory | head -60
```

Expected output:
```
========================================
  Tenzor Memory Benchmark Suite
========================================

Target Performance Metrics:
  Memory Overhead:  < 10% (PyTorch: 15%)

Initializing Tenzor library v1.0.0
[backends loading...]

========================================
  Tensor Allocation Benchmarks
========================================

=== Benchmark: Alloc - Vector 1K ===
  Runs:        100
  Mean:        0.00X ms
  [statistics...]
```

## Next Steps

1. Run full benchmark suite (takes longer):
   ```bash
   make run_benchmarks > /home/lee/Projects/Tenzor/benchmark_results.txt 2>&1
   ```

2. Analyze results and compare to targets

3. Identify optimization opportunities

4. Integrate into CI/CD pipeline
