# Phase 4: ZeRO Stage 1 Optimizer - Comprehensive Test Specification

**Document Version**: 1.0
**Date**: 2025-10-29
**Phase**: ZeRO Stage 1 Implementation
**Target**: Complete test coverage for optimizer state partitioning and CPU offload

---

## Table of Contents

1. [Overview](#overview)
2. [Test Environment Setup](#test-environment-setup)
3. [Unit Tests](#unit-tests)
4. [Integration Tests](#integration-tests)
5. [Distributed Training Tests](#distributed-training-tests)
6. [CPU Offload Tests](#cpu-offload-tests)
7. [State Management Tests](#state-management-tests)
8. [Performance Tests](#performance-tests)
9. [Memory Tests](#memory-tests)
10. [Optimizer Wrapper Tests](#optimizer-wrapper-tests)
11. [Edge Case Tests](#edge-case-tests)
12. [Test Success Criteria](#test-success-criteria)

---

## Overview

### Goals
- Verify ZeROStage1Optimizer correctly partitions optimizer states across ranks
- Validate all-reduce gradient synchronization
- Test all-gather parameter reconstruction
- Verify CPU offload functionality
- Ensure state dict save/load compatibility
- Validate <10% performance overhead target
- Confirm 4x memory reduction for optimizer states

### Test Coverage Requirements
- **Line Coverage**: >90%
- **Branch Coverage**: >85%
- **Critical Path Coverage**: 100%
- **Multi-GPU Testing**: 2, 4, and 8 GPU configurations
- **CPU Offload Testing**: On/Off configurations
- **Optimizer Coverage**: Adam, SGD, AdamW

---

## Test Environment Setup

### Required Hardware
- Multi-GPU system with CUDA support (minimum 2 GPUs)
- PCIe 3.0 or higher for offload tests
- Minimum 32GB system RAM for CPU offload tests

### Required Software
- CUDA Toolkit 11.8+
- NCCL 2.12+
- Google Test (gtest)
- MPI or process group backend

### Test Fixture Template
```cpp
class ZeROStage1Test : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize Tenzor
        tenzor::initialize();

        // Check CUDA availability
        auto* backend = backend_registry().get_backend("cuda");
        cuda_available = (backend != nullptr && backend->is_available());

        if (cuda_available) {
            cudaGetDeviceCount(&num_gpus);
            multi_gpu_available = (num_gpus >= 2);
        }

        // Default config
        default_config.world_size = num_gpus;
        default_config.rank = 0;
        default_config.offload_to_cpu = false;
        default_config.cpu_offload_threshold = 1024 * 1024;  // 1MB
        default_config.overlap_comm = true;
    }

    bool cuda_available = false;
    bool multi_gpu_available = false;
    int num_gpus = 0;
    ZeROStage1Optimizer::Config default_config;
};
```

---

## Unit Tests

### 1. Constructor and Initialization Tests

#### Test 1.1: ConstructorWithValidConfig
**File**: `tests/nn/optim/test_zero_stage1.cpp`
**What it tests**: ZeROStage1Optimizer constructor accepts valid configuration
**Setup**:
```cpp
auto base_optimizer = std::make_unique<Adam>(params, 0.001);
ZeROStage1Optimizer::Config config;
config.world_size = 4;
config.rank = 0;
```
**Expected Behavior**: Constructor succeeds without throwing
**Success Criteria**: `ASSERT_NO_THROW(ZeROStage1Optimizer(std::move(base_optimizer), config))`

---

#### Test 1.2: ConstructorWithInvalidWorldSize
**What it tests**: Constructor validates world_size parameter
**Setup**:
```cpp
auto base_optimizer = std::make_unique<Adam>(params, 0.001);
ZeROStage1Optimizer::Config config;
config.world_size = 0;  // Invalid
config.rank = 0;
```
**Expected Behavior**: Constructor throws std::invalid_argument
**Success Criteria**: Exception message contains "world_size must be >= 1"

---

#### Test 1.3: ConstructorWithInvalidRank
**What it tests**: Constructor validates rank parameter
**Setup**:
```cpp
config.world_size = 4;
config.rank = 4;  // Invalid: rank >= world_size
```
**Expected Behavior**: Constructor throws std::invalid_argument
**Success Criteria**: Exception message contains "rank must be < world_size"

---

#### Test 1.4: ConstructorInitializesPartitions
**What it tests**: Constructor correctly partitions parameters across ranks
**Setup**:
```cpp
std::vector<Tensor> params = createTestParameters(100);  // 100 params
config.world_size = 4;
config.rank = 1;
auto optimizer = ZeROStage1Optimizer(std::move(base_opt), config);
```
**Expected Behavior**:
- Rank 0 owns params[0:25]
- Rank 1 owns params[25:50]
- Rank 2 owns params[50:75]
- Rank 3 owns params[75:100]
**Success Criteria**: `EXPECT_EQ(optimizer.local_partition_size(), 25)`

---

### 2. Parameter Partitioning Tests

#### Test 2.1: PartitionParametersEvenSplit
**What it tests**: Parameters divide evenly across ranks
**Setup**:
```cpp
// 100 parameters, 4 ranks = 25 params per rank
std::vector<Tensor> params(100);
for (int i = 0; i < 100; ++i) {
    params[i] = Tensor({128, 128}, DType::Float32, Device::cuda(0));
}
```
**Expected Behavior**: Each rank gets exactly 25 parameters
**Success Criteria**: `EXPECT_EQ(local_partition.params.size(), 25)` for all ranks

---

#### Test 2.2: PartitionParametersUnevenSplit
**What it tests**: Handles parameter count not divisible by world_size
**Setup**:
```cpp
// 103 parameters, 4 ranks
// Rank 0: 26, Rank 1: 26, Rank 2: 26, Rank 3: 25
std::vector<Tensor> params(103);
```
**Expected Behavior**:
- First 3 ranks get ceil(103/4) = 26 params
- Last rank gets remaining 25 params
**Success Criteria**: Total params = 26 + 26 + 26 + 25 = 103

---

#### Test 2.3: PartitionEmptyParameterList
**What it tests**: Handles edge case of no parameters
**Setup**: `std::vector<Tensor> params;  // Empty`
**Expected Behavior**: No crash, optimizer works but does nothing
**Success Criteria**: `EXPECT_NO_THROW(optimizer.step())`

---

#### Test 2.4: PartitionMixedParameterSizes
**What it tests**: Partitions parameters of different sizes correctly
**Setup**:
```cpp
params[0] = Tensor({1000, 1000}, DType::Float32);   // 4MB
params[1] = Tensor({10, 10}, DType::Float32);       // 400B
params[2] = Tensor({500, 2000}, DType::Float32);    // 4MB
```
**Expected Behavior**: Partitioning based on parameter count, not size
**Success Criteria**: Verify each rank's partition contains correct parameters

---

### 3. Optimizer State Management Tests

#### Test 3.1: InitializeOptimizerStates
**What it tests**: Optimizer states created only for local partition
**Setup**:
```cpp
// Adam optimizer has 2 states per param: momentum + variance
auto adam = std::make_unique<Adam>(params, 0.001);
auto zero_opt = ZeROStage1Optimizer(std::move(adam), config);
```
**Expected Behavior**:
- Each rank stores states only for its partition
- Rank i: states for params[i*M/N : (i+1)*M/N]
**Success Criteria**: `EXPECT_EQ(zero_opt.state_count(), local_params * 2)`

---

#### Test 3.2: OptimizerStatesOnGPU
**What it tests**: States reside on GPU by default (no offload)
**Setup**:
```cpp
config.offload_to_cpu = false;
auto zero_opt = ZeROStage1Optimizer(std::move(adam), config);
```
**Expected Behavior**: All optimizer states on GPU
**Success Criteria**: `EXPECT_EQ(zero_opt.state_device(), Device::cuda(0))`

---

#### Test 3.3: OptimizerStatesPreserveAfterStep
**What it tests**: Optimizer states updated correctly after step
**Setup**:
```cpp
// Initialize with zero gradients
for (auto& p : params) { p.set_grad(zeros_like(p)); }
zero_opt.step();
```
**Expected Behavior**: Momentum and variance states updated
**Success Criteria**: Verify state values changed from initial values

---

### 4. Gradient Synchronization Tests

#### Test 4.1: AllReduceGradients
**What it tests**: Gradients all-reduced across ranks before optimizer step
**Setup**:
```cpp
// Rank 0: set grad = 1.0
// Rank 1: set grad = 2.0
// Rank 2: set grad = 3.0
// Rank 3: set grad = 4.0
params[0].set_grad(full_like(params[0], rank + 1.0f));
zero_opt.step();
```
**Expected Behavior**: After all-reduce, grad = (1+2+3+4) = 10.0
**Success Criteria**: `EXPECT_NEAR(params[0].grad().item<float>(), 10.0f, 1e-5)`

---

#### Test 4.2: AllReduceWithZeroGradients
**What it tests**: Handles tensors with zero gradients
**Setup**: Set all gradients to zero
**Expected Behavior**: All-reduce completes, parameters unchanged
**Success Criteria**: Parameters match initial values

---

#### Test 4.3: AllReduceSkipIfAlreadySynced
**What it tests**: Optimization: skip redundant all-reduce
**Setup**: Call `sync_gradients()` twice without new gradients
**Expected Behavior**: Second call is no-op
**Success Criteria**: Verify comm counter doesn't increment

---

### 5. Parameter All-Gather Tests

#### Test 5.1: AllGatherParametersAfterUpdate
**What it tests**: Updated parameters all-gathered to reconstruct full model
**Setup**:
```cpp
// Each rank updates its partition
zero_opt.step();
```
**Expected Behavior**:
- All ranks have identical full parameter set
- Parameters match as if trained on single GPU
**Success Criteria**: Compare parameters across ranks, all equal

---

#### Test 5.2: AllGatherHandlesLargeParameters
**What it tests**: All-gather works with large tensors (>1GB)
**Setup**: `Tensor param({10000, 10000}, DType::Float32);  // 400MB`
**Expected Behavior**: All-gather completes without OOM
**Success Criteria**: Memory usage acceptable, values correct

---

#### Test 5.3: AllGatherPreservesGradientGraph
**What it tests**: All-gathered parameters maintain autograd graph
**Setup**: Run forward/backward after all-gather
**Expected Behavior**: Gradients flow correctly
**Success Criteria**: Gradient values correct

---

## Integration Tests

### 6. End-to-End Training Tests

#### Test 6.1: SingleEpochTrainingWithAdam
**What it tests**: Complete training loop with ZeRO Stage 1 + Adam
**File**: `tests/nn/optim/test_zero_stage1_integration.cpp`
**Setup**:
```cpp
// Simple 2-layer MLP
auto model = Sequential(
    Linear(784, 256),
    ReLU(),
    Linear(256, 10)
);
auto params = model.parameters();
auto optimizer = ZeROStage1Optimizer(
    std::make_unique<Adam>(params, 0.001),
    config
);
```
**Expected Behavior**:
- Loss decreases over iterations
- Model trains successfully
- Final accuracy > baseline
**Success Criteria**:
```cpp
EXPECT_LT(final_loss, initial_loss);
EXPECT_GT(accuracy, 0.90);  // MNIST baseline
```

---

#### Test 6.2: MultiEpochConvergence
**What it tests**: Model converges over multiple epochs
**Setup**: Train for 10 epochs on synthetic data
**Expected Behavior**:
- Loss curve smooth and decreasing
- No divergence or NaN values
**Success Criteria**: `EXPECT_FALSE(std::isnan(final_loss))`

---

#### Test 6.3: CompareWithStandardOptimizer
**What it tests**: ZeRO Stage 1 produces same results as standard optimizer
**Setup**:
```cpp
// Train identical models: one with ZeRO, one standard
auto model1 = createModel();
auto model2 = createModel();  // Same initialization
trainWithZeRO(model1);
trainWithStandard(model2);
```
**Expected Behavior**: Final parameters approximately equal
**Success Criteria**: `EXPECT_NEAR(model1_params, model2_params, 1e-4)`

---

## Distributed Training Tests

### 7. Multi-GPU Communication Tests

#### Test 7.1: TwoGPUTraining
**What it tests**: Basic 2-GPU distributed training
**File**: `tests/nn/optim/test_zero_stage1_distributed.cpp`
**Setup**:
```cpp
// Spawn 2 processes
if (!multi_gpu_available) GTEST_SKIP();
config.world_size = 2;
// Process 0: rank=0, Process 1: rank=1
```
**Expected Behavior**:
- Both ranks converge to same model
- Gradient all-reduce successful
**Success Criteria**: Compare final parameters across ranks

---

#### Test 7.2: FourGPUTraining
**What it tests**: Scales to 4 GPUs
**Setup**: Same as 7.1 but with 4 processes
**Expected Behavior**:
- Each GPU uses 1/4 optimizer memory
- Training time < 1.3x single GPU (accounting for overhead)
**Success Criteria**: Memory reduction verified, time acceptable

---

#### Test 7.3: EightGPUTraining
**What it tests**: Scales to 8 GPUs
**Setup**: 8-process training
**Expected Behavior**:
- Each GPU uses 1/8 optimizer memory
- Strong scaling efficiency >70%
**Success Criteria**: Measure throughput (samples/sec)

---

#### Test 7.4: NCCLCommunicationErrors
**What it tests**: Handles NCCL communication failures gracefully
**Setup**: Simulate network partition or GPU failure
**Expected Behavior**: Appropriate error thrown, no deadlock
**Success Criteria**: Test completes (doesn't hang)

---

#### Test 7.5: GradientAllReduceCorrectness
**What it tests**: All-reduce produces correct gradient values
**Setup**:
```cpp
// Each rank: different gradient values
// Verify sum across ranks
```
**Expected Behavior**: Gradients = sum of per-rank gradients
**Success Criteria**: Numerical correctness within 1e-5

---

## CPU Offload Tests

### 8. CPU Offload Functionality Tests

#### Test 8.1: EnableCPUOffload
**What it tests**: CPU offload mode activates correctly
**File**: `tests/nn/optim/test_zero_stage1_offload.cpp`
**Setup**:
```cpp
config.offload_to_cpu = true;
config.cpu_offload_threshold = 1024;  // 1KB threshold
auto zero_opt = ZeROStage1Optimizer(std::move(adam), config);
```
**Expected Behavior**:
- Optimizer states stored in CPU memory
- Parameters remain on GPU
**Success Criteria**: `EXPECT_EQ(zero_opt.state_device(), Device::cpu())`

---

#### Test 8.2: OffloadThresholdRespected
**What it tests**: Only parameters above threshold are offloaded
**Setup**:
```cpp
params[0] = Tensor({10, 10}, DType::Float32);      // 400B - not offloaded
params[1] = Tensor({1000, 1000}, DType::Float32);  // 4MB - offloaded
config.cpu_offload_threshold = 1024;
```
**Expected Behavior**:
- Small params stay on GPU
- Large params offloaded to CPU
**Success Criteria**: Check device of each state tensor

---

#### Test 8.3: AsyncTransfersDuringStep
**What it tests**: Asynchronous CPU↔GPU transfers during optimizer step
**Setup**:
```cpp
config.offload_to_cpu = true;
config.overlap_comm = true;
```
**Expected Behavior**:
- States transferred to GPU
- Optimizer step applied
- States transferred back to CPU
- All async
**Success Criteria**: Step completes, values correct

---

#### Test 8.4: PrefetchNextBatch
**What it tests**: Prefetch scheduler brings next batch to GPU early
**Setup**: Train with prefetch enabled
**Expected Behavior**:
- Reduced wait time for CPU→GPU transfer
- Overlap with forward/backward
**Success Criteria**: Measure transfer wait time < 5ms

---

#### Test 8.5: CPUOffloadMemorySavings
**What it tests**: Verify GPU memory savings with offload
**Setup**:
```cpp
// Measure GPU memory before/after offload
auto mem_before = cudaMemGetInfo();
config.offload_to_cpu = true;
auto zero_opt = ZeROStage1Optimizer(std::move(adam), config);
auto mem_after = cudaMemGetInfo();
```
**Expected Behavior**: GPU memory freed = optimizer state size
**Success Criteria**: `EXPECT_GT(mem_after - mem_before, state_size * 0.9)`

---

#### Test 8.6: CPUOffloadWithPinnedMemory
**What it tests**: Pinned memory accelerates transfers
**Setup**:
```cpp
config.use_pinned_memory = true;
```
**Expected Behavior**: Transfer bandwidth > non-pinned
**Success Criteria**: Measure bandwidth, expect >10 GB/s

---

## State Management Tests

### 9. State Dict Save/Load Tests

#### Test 9.1: SaveStateDictToFile
**What it tests**: State dict saved correctly
**File**: `tests/nn/optim/test_zero_stage1_checkpoint.cpp`
**Setup**:
```cpp
zero_opt.step();  // Create some state
auto state_dict = zero_opt.state_dict();
save_checkpoint(state_dict, "checkpoint.pt");
```
**Expected Behavior**: File created, contains all states
**Success Criteria**: File exists, size > 0

---

#### Test 9.2: LoadStateDictFromFile
**What it tests**: State dict loaded correctly
**Setup**:
```cpp
auto state_dict = load_checkpoint("checkpoint.pt");
zero_opt.load_state_dict(state_dict);
```
**Expected Behavior**:
- Optimizer states restored
- Training resumes correctly
**Success Criteria**: State values match saved values

---

#### Test 9.3: SaveLoadPreservesOptimizationState
**What it tests**: Momentum/variance preserved across save/load
**Setup**:
```cpp
// Train 5 steps, save, load, train 5 more steps
trainSteps(zero_opt, 5);
auto state = zero_opt.state_dict();
zero_opt.load_state_dict(state);
trainSteps(zero_opt, 5);
```
**Expected Behavior**: Training continues smoothly
**Success Criteria**: Loss continues decreasing

---

#### Test 9.4: StateDictCompatibilityAcrossRanks
**What it tests**: State dict from rank 0 loadable on all ranks
**Setup**:
```cpp
// Rank 0 saves full state dict
if (rank == 0) { save_checkpoint(state_dict, "checkpoint.pt"); }
// All ranks load
all_ranks_load(state_dict);
```
**Expected Behavior**: All ranks load successfully
**Success Criteria**: No errors, training resumes

---

#### Test 9.5: PartialStateDictLoad
**What it tests**: Loading state dict with missing keys
**Setup**:
```cpp
// State dict has fewer parameters than optimizer
auto partial_state = createPartialStateDict();
```
**Expected Behavior**: Missing states initialized to defaults
**Success Criteria**: No crash, warning logged

---

## Performance Tests

### 10. Overhead and Throughput Tests

#### Test 10.1: ComputeOverheadVsStandardOptimizer
**What it tests**: ZeRO Stage 1 overhead < 10%
**File**: `tests/nn/optim/test_zero_stage1_performance.cpp`
**Setup**:
```cpp
// Benchmark standard Adam
auto time_standard = benchmarkTraining(standard_adam, 100);
// Benchmark ZeRO Stage 1 Adam
auto time_zero = benchmarkTraining(zero_adam, 100);
auto overhead = (time_zero - time_standard) / time_standard * 100.0;
```
**Expected Behavior**: `overhead < 10.0%`
**Success Criteria**: `EXPECT_LT(overhead, 10.0)`

---

#### Test 10.2: ThroughputWithCPUOffload
**What it tests**: CPU offload maintains reasonable throughput
**Setup**:
```cpp
config.offload_to_cpu = true;
auto throughput = measureThroughput(zero_opt);  // samples/sec
```
**Expected Behavior**: Throughput > 80% of GPU-only
**Success Criteria**: `EXPECT_GT(throughput, baseline_throughput * 0.8)`

---

#### Test 10.3: CommunicationOverlap
**What it tests**: All-reduce overlaps with optimizer step
**Setup**:
```cpp
config.overlap_comm = true;
auto time_with_overlap = benchmarkStep();
config.overlap_comm = false;
auto time_no_overlap = benchmarkStep();
```
**Expected Behavior**: Overlap reduces time by >20%
**Success Criteria**: `EXPECT_LT(time_with_overlap, time_no_overlap * 0.8)`

---

#### Test 10.4: ScalabilityAcrossGPUs
**What it tests**: Strong scaling efficiency with more GPUs
**Setup**:
```cpp
// Benchmark 1, 2, 4, 8 GPUs
// Fixed model size
std::vector<int> gpu_counts = {1, 2, 4, 8};
for (auto n : gpu_counts) {
    auto time = benchmarkTraining(n);
    auto efficiency = time_1gpu / (n * time_n);
}
```
**Expected Behavior**: Efficiency > 70% for 8 GPUs
**Success Criteria**: `EXPECT_GT(efficiency, 0.7)`

---

## Memory Tests

### 11. Memory Profiling Tests

#### Test 11.1: MemoryReductionWithPartitioning
**What it tests**: 4x memory reduction for optimizer states
**File**: `tests/nn/optim/test_zero_stage1_memory.cpp`
**Setup**:
```cpp
// Measure memory with standard optimizer
auto mem_standard = measureGPUMemory(standard_adam);
// Measure memory with ZeRO Stage 1 (4 GPUs)
auto mem_zero = measureGPUMemory(zero_adam);
auto reduction = mem_standard / mem_zero;
```
**Expected Behavior**: `reduction ≈ 4.0` (for Adam)
**Success Criteria**: `EXPECT_NEAR(reduction, 4.0, 0.5)`

---

#### Test 11.2: MemorySavingsWithCPUOffload
**What it tests**: CPU offload frees GPU memory
**Setup**:
```cpp
auto mem_gpu_only = measureGPUMemory(config.offload_to_cpu = false);
auto mem_cpu_offload = measureGPUMemory(config.offload_to_cpu = true);
auto savings = mem_gpu_only - mem_cpu_offload;
```
**Expected Behavior**: Savings = optimizer state size
**Success Criteria**: `EXPECT_GT(savings, state_size * 0.9)`

---

#### Test 11.3: PeakMemoryUsage
**What it tests**: Peak memory during training acceptable
**Setup**:
```cpp
auto peak_mem = profilePeakMemory(zero_opt, 1000 iterations);
```
**Expected Behavior**: No OOM errors, peak < GPU capacity
**Success Criteria**: `EXPECT_LT(peak_mem, gpu_total_memory * 0.95)`

---

#### Test 11.4: MemoryFragmentation
**What it tests**: No excessive memory fragmentation
**Setup**: Run 10,000 training steps
**Expected Behavior**:
- Memory usage stable
- No gradual memory increase
**Success Criteria**: Final memory ≈ initial memory

---

## Optimizer Wrapper Tests

### 12. Optimizer Integration Tests

#### Test 12.1: WrapAdamOptimizer
**What it tests**: ZeRO Stage 1 wraps Adam correctly
**File**: `tests/nn/optim/test_zero_stage1_wrappers.cpp`
**Setup**:
```cpp
auto adam = std::make_unique<Adam>(params, 0.001, 0.9, 0.999);
auto zero_adam = ZeROStage1Optimizer(std::move(adam), config);
```
**Expected Behavior**:
- Adam update rule preserved
- Momentum and variance tracked
**Success Criteria**: Parameter updates match Adam formula

---

#### Test 12.2: WrapSGDOptimizer
**What it tests**: ZeRO Stage 1 wraps SGD correctly
**Setup**:
```cpp
auto sgd = std::make_unique<SGD>(params, 0.01, 0.9);  // With momentum
auto zero_sgd = ZeROStage1Optimizer(std::move(sgd), config);
```
**Expected Behavior**: SGD with momentum works correctly
**Success Criteria**: Momentum buffer updated correctly

---

#### Test 12.3: WrapAdamWOptimizer
**What it tests**: ZeRO Stage 1 wraps AdamW correctly
**Setup**:
```cpp
auto adamw = std::make_unique<AdamW>(params, 0.001, 0.9, 0.999, 0.01);
auto zero_adamw = ZeROStage1Optimizer(std::move(adamw), config);
```
**Expected Behavior**: Weight decay applied correctly
**Success Criteria**: Parameters decay as expected

---

#### Test 12.4: WrapOptimizerWithScheduler
**What it tests**: Learning rate scheduler works with ZeRO
**Setup**:
```cpp
auto zero_opt = ZeROStage1Optimizer(...);
auto scheduler = StepLR(zero_opt, step_size=10, gamma=0.1);
```
**Expected Behavior**: LR updates correctly
**Success Criteria**: `EXPECT_NEAR(zero_opt.lr(), expected_lr, 1e-6)`

---

## Edge Case Tests

### 13. Robustness Tests

#### Test 13.1: SingleGPUConfiguration
**What it tests**: Works correctly with world_size=1
**Setup**:
```cpp
config.world_size = 1;
config.rank = 0;
```
**Expected Behavior**: Behaves like standard optimizer (no partitioning)
**Success Criteria**: Results identical to base optimizer

---

#### Test 13.2: VeryLargeModelParameters
**What it tests**: Handles models with 1B+ parameters
**Setup**: Create model with 1000 layers
**Expected Behavior**: Partitioning works, no overflow
**Success Criteria**: Training completes successfully

---

#### Test 13.3: VerySmallParameters
**What it tests**: Handles tiny parameters (<100 floats)
**Setup**: Model with many small tensors
**Expected Behavior**: Partitioning works, no edge cases
**Success Criteria**: All parameters updated

---

#### Test 13.4: MixedPrecisionTraining
**What it tests**: ZeRO works with FP16/BF16 training
**Setup**:
```cpp
auto scaler = GradScaler();
// Train with mixed precision
```
**Expected Behavior**:
- Gradients scaled correctly
- No overflow/underflow
**Success Criteria**: Model converges

---

#### Test 13.5: GradientAccumulation
**What it tests**: Works with gradient accumulation
**Setup**:
```cpp
// Accumulate gradients over 4 steps before optimizer step
for (int i = 0; i < 4; ++i) {
    loss.backward();
}
zero_opt.step();
zero_opt.zero_grad();
```
**Expected Behavior**: Accumulated gradients synced correctly
**Success Criteria**: Effective batch size = 4x

---

#### Test 13.6: DynamicParameterRegistration
**What it tests**: Adding parameters after initialization
**Setup**:
```cpp
auto zero_opt = ZeROStage1Optimizer(...);
// Later: add new layer
model.add_module(new_layer);
```
**Expected Behavior**: Error or warning (not supported)
**Success Criteria**: Appropriate error handling

---

## Test Success Criteria

### Overall Requirements

#### Functional Requirements
- ✅ All unit tests pass (100%)
- ✅ All integration tests pass (100%)
- ✅ All distributed tests pass on 2, 4, 8 GPUs
- ✅ No memory leaks detected (Valgrind/CUDA sanitizer)
- ✅ No race conditions (ThreadSanitizer)

#### Performance Requirements
- ✅ Overhead < 10% vs standard optimizer (single GPU)
- ✅ Strong scaling efficiency > 70% (8 GPUs)
- ✅ CPU offload overhead < 20%
- ✅ All-reduce bandwidth > 80% of theoretical max

#### Memory Requirements
- ✅ 4x memory reduction for optimizer states (Adam)
- ✅ CPU offload frees >90% of optimizer state GPU memory
- ✅ Peak memory < 95% GPU capacity
- ✅ No memory fragmentation over 10K iterations

#### Correctness Requirements
- ✅ Numerical accuracy vs standard optimizer (error < 1e-4)
- ✅ Convergence on MNIST, CIFAR-10 benchmarks
- ✅ State dict save/load preserves training state
- ✅ Checkpoint compatibility across PyTorch/DeepSpeed

#### Code Quality Requirements
- ✅ Line coverage > 90%
- ✅ Branch coverage > 85%
- ✅ No compiler warnings (-Wall -Wextra)
- ✅ Passes clang-tidy static analysis
- ✅ Documentation for all public APIs

---

## Test Execution Plan

### Phase 1: Unit Tests (Week 1)
1. Constructor and initialization (Tests 1.1-1.4)
2. Parameter partitioning (Tests 2.1-2.4)
3. Optimizer state management (Tests 3.1-3.3)
4. Gradient synchronization (Tests 4.1-4.3)
5. Parameter all-gather (Tests 5.1-5.3)

**Deliverable**: All unit tests passing, 85% coverage

### Phase 2: Integration Tests (Week 2)
1. End-to-end training (Tests 6.1-6.3)
2. Multi-GPU communication (Tests 7.1-7.5)
3. Optimizer wrappers (Tests 12.1-12.4)

**Deliverable**: Integration tests passing, single GPU + multi-GPU

### Phase 3: Advanced Features (Week 3)
1. CPU offload functionality (Tests 8.1-8.6)
2. State dict save/load (Tests 9.1-9.5)
3. Edge cases (Tests 13.1-13.6)

**Deliverable**: All feature tests passing

### Phase 4: Performance & Memory (Week 4)
1. Performance benchmarks (Tests 10.1-10.4)
2. Memory profiling (Tests 11.1-11.4)
3. Optimization and tuning

**Deliverable**: Meet all performance targets (<10% overhead, 4x memory reduction)

---

## Test Data and Fixtures

### Synthetic Datasets
```cpp
// Small dataset for quick tests
auto quick_dataset = SyntheticDataset(100, {28, 28}, 10);  // 100 samples

// Large dataset for stress tests
auto stress_dataset = SyntheticDataset(10000, {224, 224, 3}, 1000);
```

### Model Architectures
```cpp
// Tiny model (fast tests)
auto tiny_model = Sequential(Linear(10, 10), ReLU(), Linear(10, 10));

// Small model (unit tests)
auto small_model = SimpleCNN(28*28, 128, 10);

// Medium model (integration tests)
auto medium_model = ResNet18();

// Large model (stress tests)
auto large_model = GPT2(layers=24, hidden=1024);
```

### Reference Implementations
- Compare against PyTorch DistributedDataParallel
- Compare against DeepSpeed ZeRO Stage 1
- Numerical accuracy validation

---

## CI/CD Integration

### Automated Testing
```yaml
# .github/workflows/test_zero_stage1.yml
name: ZeRO Stage 1 Tests

on: [push, pull_request]

jobs:
  unit_tests:
    runs-on: ubuntu-latest
    steps:
      - name: Run unit tests
        run: ./build/tests/nn/optim/test_zero_stage1

  multi_gpu_tests:
    runs-on: self-hosted-4gpu
    steps:
      - name: Run 2-GPU tests
        run: mpirun -n 2 ./build/tests/nn/optim/test_zero_stage1_distributed

      - name: Run 4-GPU tests
        run: mpirun -n 4 ./build/tests/nn/optim/test_zero_stage1_distributed

  performance_tests:
    runs-on: self-hosted-gpu
    steps:
      - name: Benchmark performance
        run: ./build/tests/nn/optim/test_zero_stage1_performance

      - name: Check overhead
        run: python scripts/check_performance.py --threshold 10.0
```

---

## Appendix: Test File Organization

```
tests/
├── nn/
│   └── optim/
│       ├── test_zero_stage1.cpp              # Unit tests
│       ├── test_zero_stage1_integration.cpp  # Integration tests
│       ├── test_zero_stage1_distributed.cpp  # Multi-GPU tests
│       ├── test_zero_stage1_offload.cpp      # CPU offload tests
│       ├── test_zero_stage1_checkpoint.cpp   # State dict tests
│       ├── test_zero_stage1_performance.cpp  # Performance tests
│       ├── test_zero_stage1_memory.cpp       # Memory profiling tests
│       └── test_zero_stage1_wrappers.cpp     # Optimizer wrapper tests
└── integration/
    └── test_zero_end_to_end.cpp              # Full training pipeline
```

---

## Summary

This test specification provides comprehensive coverage for ZeRO Stage 1 Optimizer implementation:

- **Unit Tests**: 50+ tests covering core functionality
- **Integration Tests**: 15+ tests for real-world scenarios
- **Distributed Tests**: Multi-GPU validation (2, 4, 8 GPUs)
- **Performance Tests**: Verify <10% overhead target
- **Memory Tests**: Verify 4x memory reduction
- **Edge Cases**: Robustness and error handling

**Total Test Count**: 80+ tests
**Estimated Implementation Time**: 4 weeks
**Coverage Target**: >90% line coverage, >85% branch coverage

---

**Document Status**: Ready for Implementation
**Next Steps**:
1. Create test files following this specification
2. Implement ZeROStage1Optimizer to pass tests (TDD approach)
3. Run performance benchmarks
4. Document results in Phase 4 completion report
