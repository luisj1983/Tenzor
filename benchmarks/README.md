# Tenzor Benchmark Suite

Comprehensive performance benchmarking suite for the Tenzor deep learning framework.

## Quick Start

```bash
# Build with benchmarks enabled
cd /home/lee/Projects/Tenzor/build
cmake -DTENZOR_BUILD_BENCHMARKS=ON ..
cmake --build .

# Run all benchmarks
make run_benchmarks

# Run individual benchmark
./bin/benchmark_ops
./bin/benchmark_convolutions
./bin/benchmark_memory
./bin/benchmark_training
```

## Benchmark Files

### 1. benchmark_ops.cpp
**Matrix and tensor operations**
- Matrix multiplication (128×128 to 4096×4096)
- Batched matrix multiplication
- Element-wise operations (add, mul, exp, tanh)
- Reductions (sum, mean)
- Backward pass (autograd)

**Target:** < 20ms for 4096×4096 matmul (PyTorch: 22ms)

### 2. benchmark_convolutions.cpp
**Convolutional neural network operations**
- Conv2d (1×1, 3×3, 5×5, 7×7 kernels)
- Complete ResNet50 layer configurations
- Strided convolutions
- MaxPool2d and AvgPool2d
- Batch processing

**Target:** < 1ms per ResNet50 layer (PyTorch: 1.2ms)

### 3. benchmark_memory.cpp
**Memory operations and allocator performance**
- Tensor allocation/deallocation
- Clone (deep copy)
- Reshape and view operations
- Transpose and contiguous conversion
- Slicing and indexing
- Concatenation and stacking
- Memory bandwidth tests

**Target:** Memory overhead < 10% (PyTorch: 15%)

### 4. benchmark_training.cpp
**End-to-end training workflows**
- Complete training iterations (forward + backward + update)
- MLP training (various sizes)
- CNN training (image classification)
- Optimizer comparison (SGD vs Adam)
- Batch processing
- Gradient accumulation

**Target:** < 5ms per iteration for small models

## Build Options

```bash
# Enable benchmarks
cmake -DTENZOR_BUILD_BENCHMARKS=ON ..

# Release build for accurate measurements
cmake -DCMAKE_BUILD_TYPE=Release ..

# Specific backend
cmake -DTENZOR_ENABLE_CUDA=ON ..
```

## Individual Benchmark Targets

```bash
# Build specific benchmarks
make benchmark_ops
make benchmark_convolutions
make benchmark_memory
make benchmark_training

# Run specific benchmarks
make run_ops_benchmark
make run_conv_benchmark
make run_memory_benchmark
make run_training_benchmark
```

## Output Format

Each benchmark outputs:
- **Mean execution time** (primary metric)
- **Standard deviation** (consistency)
- **Min/Max times** (range)
- **Percentiles** (P95, P99 for tail latency)
- **GFLOPS/TFLOPS** (computational throughput)
- **Memory bandwidth** (GB/s for memory-bound operations)

Example output:
```
========================================
  Matrix Multiplication Benchmarks
========================================

Name:          Large (1024x1024 x 1024x1024)
Mean:          4.235 ms
Std Dev:       0.123 ms
Min:           4.102 ms
Max:           4.891 ms
Median:        4.220 ms
P95:           4.456 ms
P99:           4.789 ms
GFLOPS:        508.32
Bandwidth:     48.12 GB/s
```

## Performance Targets

### Computation
- **Small matrices (128×128):** < 0.1ms
- **Large matrices (4096×4096):** < 20ms
- **Conv2d (ResNet50 layers):** < 1ms/layer
- **Training iteration (small MLP):** < 5ms

### Memory
- **Allocation overhead:** < 1%
- **Clone operations:** > 20 GB/s
- **View operations:** < 1μs
- **Sequential read:** > 10 GB/s

### Training
- **Batch=32:** < 100ms for 10 batches
- **Optimizer overhead:** < 10%
- **Linear scaling** with batch size

## Backend Support

Benchmarks automatically use the default device. All backends are supported:
- **CPU:** Optimized with SIMD
- **CUDA:** GPU acceleration
- **OneAPI:** Intel hardware
- **Vulkan:** Cross-platform GPU
- **WebGPU:** Browser-based (future)

## Statistical Rigor

All benchmarks include:
- **Warmup iterations** (3-5) to eliminate cold-start effects
- **Multiple runs** (20-100) for statistical significance
- **Percentile analysis** for tail latency
- **Standard deviation** tracking for consistency

## Best Practices

### For Accurate Measurements
1. Use Release builds (`-DCMAKE_BUILD_TYPE=Release`)
2. Close background applications
3. Set CPU governor to "performance"
4. Run multiple times and average results
5. Check for thermal throttling

### For Development
1. Use Debug builds for troubleshooting
2. Run individual benchmarks for focused testing
3. Compare before/after optimization
4. Track regression across commits

## Integration with CI/CD

```bash
# Run benchmarks in CI
./bin/benchmark_ops --output ops_results.json
./bin/benchmark_convolutions --output conv_results.json
./bin/benchmark_memory --output mem_results.json
./bin/benchmark_training --output train_results.json

# Compare with baseline
python compare_benchmarks.py baseline.json current.json
```

## Troubleshooting

### Build Issues
```bash
# Clean build
cmake --build . --clean-first

# Check benchmark option
cmake -LAH | grep BENCHMARK
```

### Runtime Issues
```bash
# CUDA device selection
export CUDA_VISIBLE_DEVICES=0

# Reduce memory usage
# Edit benchmark source to reduce batch sizes
```

### Performance Issues
- Verify release build
- Check CPU frequency scaling
- Monitor thermal throttling
- Ensure no background processes

## Documentation

Full documentation: `/home/lee/Projects/Tenzor/docs/BENCHMARK_SUITE_SUMMARY.md`

## File Structure

```
benchmarks/
├── README.md                      (this file)
├── CMakeLists.txt                 (build configuration)
├── benchmark_ops.cpp              (matrix operations)
├── benchmark_convolutions.cpp     (convolution ops)
├── benchmark_memory.cpp           (memory operations)
├── benchmark_training.cpp         (training workflows)
└── bench_fused_ops.cpp           (fused operations)
```

## Contributing

When adding new benchmarks:
1. Follow existing structure and naming
2. Use the Benchmark utility class
3. Include warmup and multiple iterations
4. Calculate FLOPS/bandwidth where applicable
5. Add to CMakeLists.txt
6. Update this README

## License

Same as Tenzor framework license.

## Contact

See main Tenzor repository for contact information.
