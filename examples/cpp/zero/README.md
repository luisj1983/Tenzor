# ZeRO Optimization Examples

This directory contains comprehensive, production-ready examples demonstrating ZeRO (Zero Redundancy Optimizer) optimization for memory-efficient distributed training.

## Overview

ZeRO is a memory optimization technique that partitions optimizer states, gradients, and parameters across distributed ranks, enabling training of models significantly larger than GPU memory.

### Memory Savings

| Stage | Partitioned | Memory Reduction |
|-------|-------------|------------------|
| Stage 1 | Optimizer States | 4x (Adam), 2x (SGD) |
| Stage 2 | States + Gradients | 8x (Adam), 4x (SGD) |
| Stage 3 | States + Gradients + Parameters | Nx (N = world size) |

## Examples

### 1. BERT Training Examples

#### `bert_zero_stage1.cpp`
Training BERT with ZeRO Stage 1 (Optimizer State Partitioning).

**Features:**
- 4x memory reduction for optimizer states
- Gradient all-reduce synchronization
- Checkpoint save/load
- Memory usage tracking

**Build & Run:**
```bash
g++ -std=c++23 -O3 bert_zero_stage1.cpp -ltenzor -o bert_zero_stage1
mpirun -np 4 ./bert_zero_stage1
```

**Expected Output:**
- Training progress with loss
- Memory statistics per rank
- Throughput measurements
- Checkpoint saved to `/tmp/bert_zero_stage1_checkpoint_rank_*.pt`

---

#### `bert_zero_stage2.cpp`
Training BERT with ZeRO Stage 2 (Gradient + Optimizer State Partitioning).

**Features:**
- 8x total memory reduction
- Gradient bucketing for efficient communication
- Reduce-scatter during backward pass
- Backward hook registration

**Build & Run:**
```bash
g++ -std=c++23 -O3 bert_zero_stage2.cpp -ltenzor -o bert_zero_stage2
mpirun -np 4 ./bert_zero_stage2
```

**Expected Output:**
- Gradient bucket statistics
- Communication timing breakdown
- 8x memory reduction vs baseline

---

#### `bert_zero_stage3.cpp`
Training BERT with ZeRO Stage 3 (Full Parameter Partitioning).

**Features:**
- Nx memory reduction (N = world size)
- Parameter prefetch scheduling
- Automatic gather/scatter hooks
- Maximum memory efficiency

**Build & Run:**
```bash
g++ -std=c++23 -O3 bert_zero_stage3.cpp -ltenzor -o bert_zero_stage3
mpirun -np 8 ./bert_zero_stage3
```

**Expected Output:**
- Prefetch hit rate statistics
- Communication overlap efficiency
- Enables training models 8x larger

---

### 2. GPT Training Example

#### `gpt_zero_training.cpp`
Training GPT-2 Medium (350M parameters) with ZeRO Stage 3.

**Features:**
- Large language model training
- Gradient accumulation
- Learning rate warmup
- Checkpoint save/restore
- Performance profiling

**Build & Run:**
```bash
g++ -std=c++23 -O3 gpt_zero_training.cpp -ltenzor -o gpt_zero_training
mpirun -np 8 ./gpt_zero_training
```

**Expected Output:**
- Training with 350M parameters
- Per-rank memory: ~175MB (from 1.4GB)
- Checkpoint every 100 steps
- Tokens per second throughput

---

### 3. Custom Model Integration

#### `custom_model_zero.cpp`
Integrating custom models with ZeRO optimization.

**Features:**
- Custom Vision Transformer-like model
- Comparison across all ZeRO stages
- Memory usage analysis
- Best practices demonstration

**Build & Run:**
```bash
g++ -std=c++23 -O3 custom_model_zero.cpp -ltenzor -o custom_model_zero
mpirun -np 4 ./custom_model_zero
```

**Expected Output:**
- Side-by-side comparison of Stage 1/2/3
- Memory savings breakdown
- Performance metrics

---

### 4. Distributed Training Setup

#### `distributed_training.cpp`
Multi-GPU distributed training configuration.

**Features:**
- NCCL/Gloo backend setup
- Data parallel training
- Multi-node configuration
- Environment variable parsing

**Build & Run:**
```bash
# Single node, 4 GPUs
g++ -std=c++23 -O3 distributed_training.cpp -ltenzor -o distributed_training
mpirun -np 4 ./distributed_training

# Multi-node (2 nodes, 8 GPUs each)
mpirun -np 16 -H node1:8,node2:8 ./distributed_training
```

**Environment Variables:**
```bash
export MASTER_ADDR=localhost
export MASTER_PORT=12355
export WORLD_SIZE=4
export RANK=0
export LOCAL_RANK=0
```

---

### 5. CPU Offload Configuration

#### `cpu_offload_example.cpp`
Training models larger than GPU memory using CPU offload.

**Features:**
- Optimizer state offload to CPU
- Configurable offload thresholds
- Pinned memory optimization
- Performance trade-off analysis

**Build & Run:**
```bash
g++ -std=c++23 -O3 cpu_offload_example.cpp -ltenzor -o cpu_offload_example
./cpu_offload_example  # Single GPU
```

**Expected Output:**
- Comparison of 4 configurations
- Memory breakdown (GPU/CPU)
- Performance impact analysis
- Best practices recommendations

---

### 6. Checkpoint Save/Load

#### `checkpoint_example.cpp`
Production-grade checkpoint management with ZeRO.

**Features:**
- Distributed checkpoint save
- Resume training from checkpoint
- Checkpoint validation
- Incremental checkpointing
- Best practices guide

**Build & Run:**
```bash
g++ -std=c++23 -O3 checkpoint_example.cpp -ltenzor -o checkpoint_example
mpirun -np 4 ./checkpoint_example
```

**Expected Output:**
- Training with periodic checkpoints
- Resume training demonstration
- Checkpoint file listing
- Best practices summary

---

### 7. Mixed Precision Training

#### `mixed_precision_zero.cpp`
Combining FP16/BF16 with ZeRO optimization.

**Features:**
- FP16 mixed precision training
- Gradient scaling for stability
- Master weights in FP32
- Performance comparison

**Build & Run:**
```bash
g++ -std=c++23 -O3 mixed_precision_zero.cpp -ltenzor -o mixed_precision_zero
mpirun -np 4 ./mixed_precision_zero
```

**Expected Output:**
- FP32 baseline metrics
- FP16 mixed precision results
- 2x additional memory savings
- Best practices for mixed precision

---

### 8. Performance Comparison

#### `performance_comparison.cpp`
Comprehensive benchmarking of all ZeRO stages.

**Features:**
- Side-by-side comparison
- Throughput measurements
- Memory usage analysis
- Scalability metrics

**Build & Run:**
```bash
g++ -std=c++23 -O3 performance_comparison.cpp -ltenzor -o performance_comparison
mpirun -np 8 ./performance_comparison
```

**Expected Output:**
```
=== ZeRO Performance Comparison ===
Configuration         Step Time      Throughput     GPU Memory          CPU Memory          Avg Loss
---------------------------------------------------------------------------------------------------------------------
Baseline (No ZeRO)    45.23 ms       709 img/s      4.50 GB            0 B                 0.8234
ZeRO Stage 1          47.12 ms       682 img/s      1.12 GB            0 B                 0.8231
ZeRO Stage 2          49.56 ms       648 img/s      562 MB             0 B                 0.8229
ZeRO Stage 3          53.78 ms       597 img/s      281 MB             0 B                 0.8227
```

---

## Quick Start Guide

### Prerequisites

1. **Install Tenzor:**
```bash
git clone https://github.com/your-org/tenzor.git
cd Tenzor
mkdir build && cd build
cmake ..
make -j$(nproc)
sudo make install
```

2. **Install MPI:**
```bash
# Ubuntu/Debian
sudo apt-get install libopenmpi-dev

# CentOS/RHEL
sudo yum install openmpi-devel
```

3. **Install NCCL (for GPU training):**
```bash
# Follow NVIDIA NCCL installation guide
# https://developer.nvidia.com/nccl
```

### Running Examples

1. **Choose an example** based on your use case
2. **Compile** with g++ (C++17 or later)
3. **Run** with mpirun

```bash
# Example workflow
cd examples/zero
g++ -std=c++23 -O3 bert_zero_stage2.cpp -ltenzor -o bert_zero_stage2
mpirun -np 4 ./bert_zero_stage2
```

### Troubleshooting

**Issue:** `cannot find -ltenzor`
**Solution:** Ensure Tenzor is installed and `LD_LIBRARY_PATH` is set:
```bash
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
```

**Issue:** MPI initialization fails
**Solution:** Check MPI installation and network configuration:
```bash
mpirun --version
mpirun -np 2 hostname
```

**Issue:** CUDA out of memory
**Solution:** Reduce batch size or enable CPU offload:
```cpp
config.batch_size = 8;  // Reduce from 32
config.offload_to_cpu = true;
```

---

## Best Practices

### Choosing the Right ZeRO Stage

| Scenario | Recommended Stage | Reason |
|----------|-------------------|--------|
| Model fits in GPU memory | Stage 1 | Minimal overhead, good speedup |
| Model + gradients fit | Stage 2 | Best memory/speed trade-off |
| Very large models | Stage 3 | Maximum memory savings |
| Limited GPU memory | Stage 3 + CPU offload | Train models larger than GPU |

### Configuration Tips

1. **Batch Size:**
   - Start with 32 per GPU
   - Increase until memory full
   - Use gradient accumulation if needed

2. **Gradient Bucket Size (Stage 2):**
   - Default: 25MB
   - Larger buckets: fewer comm calls, more memory
   - Smaller buckets: more overlap, less memory

3. **Prefetch Depth (Stage 3):**
   - Default: 2
   - Increase for better latency hiding
   - Decrease to save memory

4. **CPU Offload:**
   - Enable for models >GPU memory
   - Use pinned memory for faster transfers
   - Set threshold to offload only large tensors

### Performance Optimization

1. **Communication Overlap:**
```cpp
config.overlap_comm = true;
config.overlap_comm_compute = true;
```

2. **Prefetch Scheduling:**
```cpp
config.prefetch_depth = 3;
config.max_cached_params = 15;
```

3. **Mixed Precision:**
```cpp
// Use FP16 for 2x additional memory savings
GradScaler scaler;
auto scaled_loss = scaler.scale(loss);
```

---

## Architecture Overview

### ZeRO Stage 1
```
┌─────────────────────────────────────┐
│ GPU 0                               │
│ ┌─────────┐ ┌───────┐ ┌──────────┐│
│ │Parameters│ │Gradients│ │Opt States││
│ │ (Full)  │ │ (Full)  │ │(1/N Shard)││
│ └─────────┘ └───────┘ └──────────┘│
└─────────────────────────────────────┘
```

### ZeRO Stage 2
```
┌─────────────────────────────────────┐
│ GPU 0                               │
│ ┌─────────┐ ┌───────┐ ┌──────────┐│
│ │Parameters│ │Gradients│ │Opt States││
│ │ (Full)  │ │(1/N Shard)│(1/N Shard)││
│ └─────────┘ └───────┘ └──────────┘│
└─────────────────────────────────────┘
```

### ZeRO Stage 3
```
┌─────────────────────────────────────┐
│ GPU 0                               │
│ ┌─────────┐ ┌───────┐ ┌──────────┐│
│ │Parameters│ │Gradients│ │Opt States││
│ │(1/N Shard)│(1/N Shard)│(1/N Shard)││
│ └─────────┘ └───────┘ └──────────┘│
└─────────────────────────────────────┘
```

---

## References

- [ZeRO Paper (Rajbhandari et al., 2020)](https://arxiv.org/abs/1910.02054)
- [DeepSpeed Documentation](https://www.deepspeed.ai/)
- [Tenzor API Documentation](https://tenzor.readthedocs.io/)

---

## Support

For questions or issues:
1. Check the [main documentation](../../docs/)
2. Open an issue on GitHub
3. Join the community Discord

---

## License

These examples are part of the Tenzor project and follow the same license.
