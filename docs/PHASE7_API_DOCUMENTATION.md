# ZeRO Optimizer API Documentation

**Version**: 1.0
**Last Updated**: 2025-10-30
**Status**: Production Ready

---

## Table of Contents

1. [Overview](#overview)
2. [ZeRO Stage 1 API](#zero-stage-1-api)
3. [ZeRO Stage 2 API](#zero-stage-2-api)
4. [ZeRO Stage 3 API](#zero-stage-3-api)
5. [Configuration Options](#configuration-options)
6. [Error Handling](#error-handling)
7. [Performance Monitoring](#performance-monitoring)
8. [Examples](#examples)

---

## Overview

The Tenzor ZeRO optimizer provides memory-efficient distributed training through three progressive stages:

| Stage | Memory Savings | Partitioned Components | Communication Overhead |
|-------|---------------|------------------------|----------------------|
| **Stage 1** | 4x | Optimizer States | Low |
| **Stage 2** | 8x | Optimizer States + Gradients | Medium |
| **Stage 3** | Nx (N=GPUs) | Parameters + Gradients + States | High |

### Key Features

- **Memory Efficiency**: Train models 10-100x larger than GPU memory
- **CPU Offloading**: Automatic offload of states/gradients/parameters to CPU
- **Communication Overlap**: Overlap computation with data transfers
- **Automatic Hooks**: No manual forward/backward modifications needed
- **Checkpoint Support**: Save and restore distributed training state

---

## ZeRO Stage 1 API

### ZeROStage1Optimizer

**File**: `include/tenzor/nn/optim/zero_optimizer.hpp`

Partitions optimizer states (momentum, variance) across distributed ranks while keeping parameters and gradients replicated.

#### Constructor

```cpp
ZeROStage1Optimizer(
    std::unique_ptr<Optimizer> base_optimizer,
    const ZeROStage1Config& config
);
```

**Parameters**:
- `base_optimizer`: Base optimizer (Adam, SGD, AdamW) - ownership transferred
- `config`: Configuration for Stage 1 (see [Configuration](#zerostage1config))

**Throws**:
- `std::invalid_argument`: If `rank >= world_size` or `base_optimizer` is null

**Example**:
```cpp
auto adam = std::make_unique<Adam>(model.parameters(), 1e-3);

ZeROStage1Config config;
config.world_size = 4;
config.rank = distributed::get_rank();
config.offload_to_cpu = true;

auto zero_opt = ZeROStage1Optimizer(std::move(adam), config);
```

#### Methods

##### `step() -> void`

Performs optimizer step with distributed state partitioning.

**Algorithm**:
1. All-reduce gradients across ranks (sum)
2. Fetch local state partition from CPU (if offloaded)
3. Update local parameter partition with base optimizer
4. Offload states back to CPU (if enabled)
5. All-gather updated parameters across ranks

**Throws**:
- `std::runtime_error`: If distributed not initialized

**Example**:
```cpp
// Training loop
for (auto& batch : dataloader) {
    auto loss = model.forward(batch.input);
    loss.backward();
    zero_opt.step();  // Handles all communication
    zero_opt.zero_grad();
}
```

##### `zero_grad() -> void`

Zeros all parameter gradients.

**Example**:
```cpp
zero_opt.zero_grad();
```

##### `state_dict() const -> std::unordered_map<std::string, Tensor>`

Returns optimizer state dictionary for the local partition only.

**Returns**: Map of state variable names to tensors

**Keys**:
- `"rank"`: Current rank ID (Int32 tensor)
- `"world_size"`: Total number of ranks (Int32 tensor)
- `"momentum_i"`: Momentum state for parameter i
- `"variance_i"`: Variance state for parameter i (Adam/AdamW only)

**Note**: Use `save_checkpoint()` to save full distributed state.

**Example**:
```cpp
auto state = zero_opt.state_dict();
// Save only local partition
serialize(state, "checkpoint_rank_" + std::to_string(rank) + ".pt");
```

##### `load_state_dict(const std::unordered_map<std::string, Tensor>& state) -> void`

Loads optimizer state dictionary for the local partition.

**Parameters**:
- `state`: State dictionary to load

**Throws**:
- `std::runtime_error`: If rank or world_size mismatch

**Note**: Use `load_checkpoint()` to load full distributed state.

**Example**:
```cpp
auto state = deserialize("checkpoint_rank_" + std::to_string(rank) + ".pt");
zero_opt.load_state_dict(state);
```

##### `save_checkpoint(const std::string& path_prefix) const -> void`

Saves distributed checkpoint. Each rank saves its partition; master rank saves metadata.

**Parameters**:
- `path_prefix`: Checkpoint path prefix (rank ID will be appended)

**Creates Files**:
- `{path_prefix}_rank_0.pt`: Rank 0 partition
- `{path_prefix}_rank_1.pt`: Rank 1 partition
- ...
- `{path_prefix}_metadata.txt`: Checkpoint metadata (master rank only)

**Example**:
```cpp
// All ranks must call this
zero_opt.save_checkpoint("checkpoints/model_step_1000");

// Creates:
//   checkpoints/model_step_1000_rank_0.pt
//   checkpoints/model_step_1000_rank_1.pt
//   ...
//   checkpoints/model_step_1000_metadata.txt
```

##### `load_checkpoint(const std::string& path_prefix) -> void`

Loads distributed checkpoint. Each rank loads its partition; master rank validates metadata.

**Parameters**:
- `path_prefix`: Checkpoint path prefix

**Throws**:
- `std::runtime_error`: If checkpoint files missing or incompatible

**Example**:
```cpp
// All ranks must call this
zero_opt.load_checkpoint("checkpoints/model_step_1000");
```

#### Accessors

##### `rank() const -> int`

Returns current rank ID.

##### `world_size() const -> int`

Returns total number of ranks.

##### `is_cpu_offload_enabled() const -> bool`

Returns true if CPU offload is enabled.

##### `local_param_count() const -> size_t`

Returns number of parameters in local partition.

##### `get_memory_stats() const -> MemoryStats`

Returns memory usage statistics.

**MemoryStats Structure**:
```cpp
struct MemoryStats {
    size_t gpu_optimizer_memory;     // GPU memory for optimizer states (bytes)
    size_t cpu_optimizer_memory;     // CPU memory for optimizer states (bytes)
    size_t gpu_gradient_memory;      // GPU memory for gradients (bytes)
    size_t num_parameters;           // Total number of parameters
    size_t num_local_parameters;     // Parameters in local partition
};
```

**Example**:
```cpp
auto stats = zero_opt.get_memory_stats();
std::cout << "GPU optimizer memory: "
          << stats.gpu_optimizer_memory / (1024*1024) << " MB\n";
std::cout << "Local parameters: "
          << stats.num_local_parameters << "\n";
```

---

## ZeRO Stage 2 API

### ZeROStage2Optimizer

**File**: `include/tenzor/nn/optim/zero_optimizer.hpp`

Extends Stage 1 by also partitioning gradients. Uses reduce-scatter during backward pass to compute and partition gradients simultaneously.

#### Constructor

```cpp
ZeROStage2Optimizer(
    std::unique_ptr<Optimizer> base_optimizer,
    const ZeROStage2Config& config
);
```

**Parameters**:
- `base_optimizer`: Base optimizer - ownership transferred
- `config`: Configuration for Stage 2 (see [Configuration](#zerostage2config))

**Example**:
```cpp
auto adamw = std::make_unique<AdamW>(model.parameters(), 2e-5);

ZeROStage2Config config;
config.world_size = 8;
config.rank = distributed::get_rank();
config.gradient_bucket_size = 25 * 1024 * 1024;  // 25MB buckets
config.reduce_scatter_in_backward = true;

auto zero_opt = ZeROStage2Optimizer(std::move(adamw), config);
```

#### Methods

##### `step() -> void`

Performs optimizer step with gradient and state partitioning.

**Algorithm**:
1. Gradients already reduced-scattered via backward hooks (no all-reduce needed)
2. Fetch local state partition from CPU (if offloaded)
3. Update local parameter partition
4. Offload states back to CPU (if enabled)
5. All-gather updated parameters

**Note**: Unlike Stage 1, no gradient all-reduce is needed since gradients are already partitioned during backward pass.

##### `register_backward_hooks() -> void`

Registers backward hooks for automatic gradient reduce-scatter.

**Must be called after model creation** to enable automatic gradient partitioning during backward pass.

**Throws**:
- `std::runtime_error`: If parameters have no grad_fn

**Example**:
```cpp
// Create optimizer
auto zero_opt = ZeROStage2Optimizer(std::move(base_opt), config);

// Register hooks (REQUIRED for Stage 2)
zero_opt.register_backward_hooks();

// Training loop - gradients automatically partitioned
for (auto& batch : dataloader) {
    auto loss = model.forward(batch.input);
    loss.backward();  // Gradients reduced-scattered automatically
    zero_opt.step();
}
```

##### `get_bucket_stats() const -> BucketStats`

Returns gradient bucket statistics.

**BucketStats Structure**:
```cpp
struct BucketStats {
    size_t num_buckets;           // Number of gradient buckets
    size_t avg_bucket_size;       // Average bucket size (bytes)
    size_t max_bucket_size;       // Maximum bucket size (bytes)
    size_t total_gradient_memory; // Total gradient memory (bytes)
};
```

**Example**:
```cpp
auto stats = zero_opt.get_bucket_stats();
std::cout << "Number of gradient buckets: " << stats.num_buckets << "\n";
std::cout << "Average bucket size: "
          << stats.avg_bucket_size / (1024*1024) << " MB\n";
```

##### `hooks_registered() const -> bool`

Returns true if backward hooks are registered.

---

## ZeRO Stage 3 API

### ZeROStage3Optimizer

**File**: `include/tenzor/nn/optim/zero_optimizer.hpp`

Most aggressive memory savings. Partitions parameters, gradients, AND optimizer states. Parameters are gathered on-demand for computation.

#### Constructor

```cpp
ZeROStage3Optimizer(
    std::unique_ptr<Optimizer> base_optimizer,
    const Stage3Config& config
);
```

**Parameters**:
- `base_optimizer`: Base optimizer - ownership transferred
- `config`: Configuration for Stage 3 (see [Configuration](#stage3config))

**Example**:
```cpp
auto adamw = std::make_unique<AdamW>(model.parameters(), 1e-4);

Stage3Config config;
config.world_size = 8;
config.rank = distributed::get_rank();
config.prefetch_bucket_size = 100 * 1024 * 1024;  // 100MB
config.prefetch_depth = 2;
config.overlap_comm_compute = true;

auto zero_opt = ZeROStage3Optimizer(std::move(adamw), config);
```

#### Model Registration

##### `register_model(Module& model) -> void`

Registers model for parameter partitioning. **Must be called before training.**

**Operations**:
1. Partitions all model parameters across ranks
2. Registers forward/backward hooks for automatic gather/scatter
3. Builds execution graph for prefetch scheduling
4. Initializes parameter state tracking

**Throws**:
- `std::runtime_error`: If model already registered

**Example**:
```cpp
auto model = GPT2Model(config);
auto zero_opt = ZeROStage3Optimizer(std::move(adamw), stage3_config);

// REQUIRED: Register model
zero_opt.register_model(model);

// Now ready for training
for (auto& batch : dataloader) {
    auto output = model.forward(batch.input);  // Parameters auto-gathered
    // ...
}
```

##### `unregister_model() -> void`

Unregisters model and cleans up hooks.

#### Optimizer Interface

##### `step() -> void`

Performs optimizer step on local partition only.

**Algorithm**:
1. Wait for gradient reduce-scatter to complete
2. Update only local partition of parameters
3. NO all-gather needed (parameters remain partitioned)

**Note**: Parameters stay partitioned after step. They will be gathered on-demand during next forward pass.

##### `zero_grad() -> void`

Zeros gradients for local partition only.

#### State Management

##### `state_dict() const -> std::unordered_map<std::string, Tensor>`

Returns state for only the local partition.

**To get full state**, use `gather_full_state()`.

##### `load_state_dict(const std::unordered_map<std::string, Tensor>& state) -> void`

Loads partitioned state.

**To load from full checkpoint**, use `load_full_state()`.

##### `gather_full_state() -> std::unordered_map<std::string, Tensor>`

Gathers full optimizer state from all ranks for checkpointing.

**Warning**: Expensive operation that should only be called periodically.

**Example**:
```cpp
if (step % 1000 == 0) {
    auto full_state = zero_opt.gather_full_state();
    if (rank == 0) {
        serialize(full_state, "checkpoint_full.pt");
    }
}
```

##### `load_full_state(const std::unordered_map<std::string, Tensor>& full_state) -> void`

Loads from full (non-partitioned) checkpoint. Automatically partitions state across ranks.

#### Manual Control API

##### `gather_parameter(Tensor* param) -> Tensor`

Manually gathers a parameter. Returns the full (gathered) parameter.

**Parameters**:
- `param`: Parameter to gather (must be registered)

**Returns**: Full parameter (replicated across all ranks)

**Throws**:
- `std::runtime_error`: If communication fails

**Example**:
```cpp
// For inference, manually gather parameters
Tensor* weight = &model.layer1.weight;
Tensor full_weight = zero_opt.gather_parameter(weight);
// Use full_weight for computation
```

##### `gather_parameter_async(Tensor* param) -> std::shared_ptr<AsyncHandle>`

Asynchronously gathers a parameter.

**Returns**: Handle for async operation

**Example**:
```cpp
auto handle = zero_opt.gather_parameter_async(param);
// Do other work...
Tensor full_param = zero_opt.wait_gather(handle);
```

##### `wait_gather(std::shared_ptr<AsyncHandle> handle) -> Tensor`

Waits for async gather to complete.

**Returns**: Full parameter tensor

##### `free_gathered_parameter(Tensor* param) -> void`

Manually frees a gathered parameter, keeping only local partition.

**Example**:
```cpp
// After using gathered parameter
zero_opt.free_gathered_parameter(param);
```

##### `prefetch_parameters(const std::vector<Tensor*>& params) -> void`

Manually triggers prefetch for specific parameters.

**Example**:
```cpp
// Prefetch next layer's parameters
std::vector<Tensor*> next_params = {
    &model.layer2.weight,
    &model.layer2.bias
};
zero_opt.prefetch_parameters(next_params);
```

#### Parameter State Queries

##### `get_parameter_state(Tensor* param) const -> ParameterState`

Returns current state of a parameter.

**ParameterState Enum**:
```cpp
enum class ParameterState {
    PARTITIONED,    // Only local partition exists
    GATHERING,      // All-gather in progress (async)
    GATHERED,       // Full parameter available
    SCATTERING,     // Reduce-scatter in progress (async)
};
```

**Example**:
```cpp
auto state = zero_opt.get_parameter_state(&weight);
if (state == ParameterState::GATHERED) {
    // Parameter is ready for use
}
```

##### `is_parameter_gathered(Tensor* param) const -> bool`

Returns true if full parameter is currently available.

##### `pin_parameter(Tensor* param) -> void`

Pins parameter in memory (keeps it gathered). Useful for frequently used parameters.

**Example**:
```cpp
// Pin first and last layer for faster access
zero_opt.pin_parameter(&model.first_layer.weight);
zero_opt.pin_parameter(&model.last_layer.weight);
```

##### `unpin_parameter(Tensor* param) -> void`

Unpins parameter (allows freeing).

##### `is_parameter_pinned(Tensor* param) const -> bool`

Returns true if parameter is pinned in memory.

#### Performance Monitoring

##### `get_stats() -> Stats`

Returns performance statistics.

**Stats Structure**:
```cpp
struct Stats {
    // Communication stats
    size_t total_all_gather_calls;
    size_t total_all_gather_bytes;
    double avg_all_gather_time_ms;

    // Memory stats
    size_t peak_gathered_memory_bytes;
    size_t current_gathered_memory_bytes;
    int num_cached_params;

    // Prefetch efficiency
    double prefetch_hit_rate;  // % of gathers satisfied by prefetch
    int prefetch_queue_depth;

    // Performance metrics
    double forward_comm_time_ms;
    double backward_comm_time_ms;
    double overlap_efficiency;  // % of comm hidden by compute
};
```

**Example**:
```cpp
auto stats = zero_opt.get_stats();
std::cout << "Prefetch hit rate: "
          << (stats.prefetch_hit_rate * 100) << "%\n";
std::cout << "Overlap efficiency: "
          << (stats.overlap_efficiency * 100) << "%\n";
```

##### `reset_stats() -> void`

Resets performance statistics.

##### `get_prefetch_stats() const -> PrefetchStats`

Returns prefetch-specific statistics.

**PrefetchStats Structure**:
```cpp
struct PrefetchStats {
    size_t prefetch_queue_size;
    size_t prefetched_bytes;
    double hit_rate;           // Fraction of gathers that hit prefetch
    double avg_prefetch_time_ms;
    size_t prefetch_hits;
    size_t prefetch_misses;
};
```

---

## Configuration Options

### ZeROStage1Config

```cpp
struct ZeROStage1Config {
    int world_size{1};                      // Number of distributed ranks
    int rank{0};                            // Current rank ID
    bool offload_to_cpu{false};             // Offload optimizer states to CPU
    size_t cpu_offload_threshold{1024};     // Min bytes to offload (default: 1KB)
    bool overlap_comm{true};                // Overlap communication with computation
    bool pin_memory{true};                  // Use pinned memory for transfers
    std::shared_ptr<distributed::ProcessGroup> process_group{nullptr};
};
```

**Parameter Details**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `world_size` | int | 1 | Total number of distributed ranks (GPUs) |
| `rank` | int | 0 | Current rank ID (0 to world_size-1) |
| `offload_to_cpu` | bool | false | Offload optimizer states to CPU RAM |
| `cpu_offload_threshold` | size_t | 1024 | Minimum tensor size (bytes) to offload |
| `overlap_comm` | bool | true | Overlap communication with computation |
| `pin_memory` | bool | true | Use pinned memory for faster CPU<->GPU transfers |
| `process_group` | shared_ptr | nullptr | Communication group (auto-detected if null) |

**Recommendations**:
- **Small models (<1B params)**: `offload_to_cpu=false` for best speed
- **Medium models (1-10B)**: `offload_to_cpu=true` to save memory
- **Large models (>10B)**: `offload_to_cpu=true` + increase threshold to 10KB

### ZeROStage2Config

Extends `ZeROStage1Config` with additional fields:

```cpp
struct ZeROStage2Config : public ZeROStage1Config {
    size_t gradient_bucket_size{25 * 1024 * 1024};  // Target bucket size (default: 25MB)
    bool reduce_scatter_in_backward{true};           // Enable reduce-scatter during backward
    bool gradient_bucketing{true};                   // Enable gradient bucketing
};
```

**Additional Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `gradient_bucket_size` | size_t | 25 MB | Target size for gradient buckets |
| `reduce_scatter_in_backward` | bool | true | Enable automatic reduce-scatter during backward |
| `gradient_bucketing` | bool | true | Group small gradients into buckets |

**Recommendations**:
- **High bandwidth networks (NVLink)**: Increase bucket size to 50-100MB
- **Low bandwidth networks (PCIe)**: Use default 25MB
- **Small models**: Decrease to 10-15MB to reduce memory overhead

### Stage3Config

Extends `ZeROStage2Config` with extensive prefetching and memory options:

```cpp
struct Stage3Config : public ZeROStage2Config {
    // Prefetching Configuration
    size_t prefetch_bucket_size{100 * 1024 * 1024};  // 100 MB default
    int prefetch_depth{2};
    int max_concurrent_prefetches{4};
    bool overlap_comm_compute{true};

    // Memory Management
    int max_cached_params{10};
    bool cache_params_across_passes{true};
    size_t partition_threshold{1024};  // 1 KB
    bool pin_first_layer{true};
    bool pin_last_layer{true};
    size_t max_gathered_buffer_size{500 * 1024 * 1024};  // 500 MB

    // CPU Offload Integration
    bool offload_params_to_cpu{false};
    bool offload_gathered_to_cpu{false};

    // Communication Settings
    bool use_async_gather{true};
    bool use_separate_streams{true};
    int gather_stream_priority{-1};
    bool use_nccl_groups{false};
    bool gradient_checkpointing_aware{false};
    size_t partition_alignment{128};
};
```

**Critical Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prefetch_bucket_size` | size_t | 100 MB | Size of parameter prefetch buckets |
| `prefetch_depth` | int | 2 | Number of layers to prefetch ahead |
| `max_concurrent_prefetches` | int | 4 | Maximum concurrent prefetch operations |
| `overlap_comm_compute` | bool | true | Overlap all-gather with computation |
| `max_cached_params` | int | 10 | Maximum cached gathered parameters |
| `cache_params_across_passes` | bool | true | Cache parameters between forward/backward |
| `partition_threshold` | size_t | 1 KB | Minimum size to partition |
| `pin_first_layer` | bool | true | Keep first layer gathered |
| `pin_last_layer` | bool | true | Keep last layer gathered |
| `max_gathered_buffer_size` | size_t | 500 MB | Maximum memory for gathered parameters |
| `offload_params_to_cpu` | bool | false | Offload partitioned parameters to CPU |
| `offload_gathered_to_cpu` | bool | false | Offload gathered parameters to CPU |

**Tuning Guidelines**:

**For Maximum Speed** (sufficient GPU memory):
```cpp
config.prefetch_depth = 3;
config.prefetch_bucket_size = 200 * 1024 * 1024;  // 200MB
config.overlap_comm_compute = true;
config.max_cached_params = 20;
config.offload_params_to_cpu = false;
```

**For Maximum Memory Efficiency** (limited GPU memory):
```cpp
config.prefetch_depth = 1;
config.prefetch_bucket_size = 50 * 1024 * 1024;  // 50MB
config.max_cached_params = 5;
config.offload_params_to_cpu = true;
config.offload_gathered_to_cpu = true;
```

**Balanced Configuration** (recommended starting point):
```cpp
config.prefetch_depth = 2;
config.prefetch_bucket_size = 100 * 1024 * 1024;  // 100MB
config.overlap_comm_compute = true;
config.max_cached_params = 10;
config.offload_params_to_cpu = false;
```

---

## Error Handling

### Common Exceptions

#### `std::invalid_argument`

Thrown when invalid configuration is provided.

**Common Causes**:
- `rank >= world_size`
- `base_optimizer` is null
- Negative or zero `world_size`

**Example**:
```cpp
try {
    auto zero_opt = ZeROStage1Optimizer(std::move(opt), config);
} catch (const std::invalid_argument& e) {
    std::cerr << "Invalid configuration: " << e.what() << "\n";
}
```

#### `std::runtime_error`

Thrown during runtime failures.

**Common Causes**:
- Distributed not initialized
- Communication failures
- Rank/world_size mismatch during checkpoint load
- Model already registered (Stage 3)

**Example**:
```cpp
try {
    zero_opt.step();
} catch (const std::runtime_error& e) {
    std::cerr << "Runtime error: " << e.what() << "\n";
    // Handle error (e.g., retry, save checkpoint)
}
```

### Best Practices

1. **Check distributed initialization**:
```cpp
if (!distributed::is_initialized()) {
    distributed::init_process_group("nccl");
}
```

2. **Validate configuration**:
```cpp
if (config.rank >= config.world_size) {
    throw std::invalid_argument("Invalid rank");
}
```

3. **Handle checkpoint loading gracefully**:
```cpp
try {
    zero_opt.load_checkpoint(path);
} catch (const std::runtime_error& e) {
    std::cerr << "Failed to load checkpoint: " << e.what() << "\n";
    std::cerr << "Starting from scratch...\n";
}
```

---

## Performance Monitoring

### Memory Statistics

Track memory usage across stages:

```cpp
auto stats = zero_opt.get_memory_stats();

std::cout << "Memory Report:\n";
std::cout << "  GPU optimizer memory: "
          << stats.gpu_optimizer_memory / (1024*1024) << " MB\n";
std::cout << "  CPU optimizer memory: "
          << stats.cpu_optimizer_memory / (1024*1024) << " MB\n";
std::cout << "  GPU gradient memory: "
          << stats.gpu_gradient_memory / (1024*1024) << " MB\n";
std::cout << "  Total parameters: " << stats.num_parameters << "\n";
std::cout << "  Local parameters: " << stats.num_local_parameters << "\n";

// Calculate memory savings
double memory_reduction = static_cast<double>(stats.num_parameters) /
                         stats.num_local_parameters;
std::cout << "Memory reduction factor: " << memory_reduction << "x\n";
```

### Communication Statistics (Stage 3)

Monitor communication efficiency:

```cpp
auto stats = zero_opt.get_stats();

std::cout << "Communication Report:\n";
std::cout << "  Total all-gather calls: " << stats.total_all_gather_calls << "\n";
std::cout << "  Total data transferred: "
          << stats.total_all_gather_bytes / (1024*1024*1024) << " GB\n";
std::cout << "  Average gather time: "
          << stats.avg_all_gather_time_ms << " ms\n";
std::cout << "  Prefetch hit rate: "
          << (stats.prefetch_hit_rate * 100) << "%\n";
std::cout << "  Communication overlap: "
          << (stats.overlap_efficiency * 100) << "%\n";
```

### Prefetch Efficiency (Stage 3)

Analyze prefetch performance:

```cpp
auto prefetch_stats = zero_opt.get_prefetch_stats();

std::cout << "Prefetch Report:\n";
std::cout << "  Hits: " << prefetch_stats.prefetch_hits << "\n";
std::cout << "  Misses: " << prefetch_stats.prefetch_misses << "\n";
std::cout << "  Hit rate: "
          << (prefetch_stats.hit_rate * 100) << "%\n";
std::cout << "  Queue size: " << prefetch_stats.prefetch_queue_size << "\n";

// Tune prefetch_depth based on hit rate
if (prefetch_stats.hit_rate < 0.8) {
    std::cout << "Consider increasing prefetch_depth\n";
}
```

---

## Examples

### Example 1: Basic ZeRO Stage 1 Training

```cpp
#include <tenzor/distributed/distributed.hpp>
#include <tenzor/nn/optim/zero_optimizer.hpp>

int main(int argc, char** argv) {
    // Initialize distributed
    distributed::init_process_group("nccl");
    auto rank = distributed::get_rank();
    auto world_size = distributed::get_world_size();

    // Create model
    auto model = BertModel(config);
    model.to(Device::cuda(rank));

    // Create base optimizer
    auto adam = std::make_unique<Adam>(model.parameters(), 1e-3);

    // Wrap with ZeRO Stage 1
    ZeROStage1Config zero_config;
    zero_config.world_size = world_size;
    zero_config.rank = rank;
    zero_config.offload_to_cpu = true;

    auto zero_opt = ZeROStage1Optimizer(std::move(adam), zero_config);

    // Training loop
    for (int epoch = 0; epoch < 10; ++epoch) {
        for (auto& batch : dataloader) {
            zero_opt.zero_grad();
            auto output = model.forward(batch.input);
            auto loss = criterion(output, batch.target);
            loss.backward();
            zero_opt.step();
        }
    }

    return 0;
}
```

### Example 2: ZeRO Stage 2 with Gradient Bucketing

```cpp
#include <tenzor/nn/optim/zero_optimizer.hpp>

// Initialize distributed (already done)

// Create model
auto model = GPT2Model(gpt2_config);
model.to(Device::cuda());

// Create ZeRO Stage 2 optimizer
auto adamw = std::make_unique<AdamW>(model.parameters(), 2e-5);

ZeROStage2Config config;
config.world_size = 8;
config.rank = distributed::get_rank();
config.gradient_bucket_size = 50 * 1024 * 1024;  // 50MB buckets
config.reduce_scatter_in_backward = true;

auto zero_opt = ZeROStage2Optimizer(std::move(adamw), config);

// REQUIRED: Register backward hooks
zero_opt.register_backward_hooks();

// Training loop
for (auto& batch : dataloader) {
    zero_opt.zero_grad();
    auto output = model.forward(batch.input);
    auto loss = criterion(output, batch.target);

    // Gradients automatically reduced-scattered during backward
    loss.backward();

    // Optimizer step (no gradient all-reduce needed)
    zero_opt.step();

    // Print statistics periodically
    if (step % 100 == 0) {
        auto stats = zero_opt.get_bucket_stats();
        std::cout << "Buckets: " << stats.num_buckets
                  << ", Avg size: " << stats.avg_bucket_size / (1024*1024)
                  << " MB\n";
    }
}
```

### Example 3: ZeRO Stage 3 with Prefetching

```cpp
#include <tenzor/nn/optim/zero_optimizer.hpp>

// Create model
auto model = GPT3Model(gpt3_config);  // 175B parameters

// Create ZeRO Stage 3 optimizer
auto adamw = std::make_unique<AdamW>(model.parameters(), 1e-4);

Stage3Config config;
config.world_size = 8;
config.rank = distributed::get_rank();
config.prefetch_bucket_size = 200 * 1024 * 1024;  // 200MB for high bandwidth
config.prefetch_depth = 3;  // Prefetch 3 layers ahead
config.overlap_comm_compute = true;
config.max_cached_params = 15;
config.pin_first_layer = true;
config.pin_last_layer = true;

auto zero_opt = ZeROStage3Optimizer(std::move(adamw), config);

// REQUIRED: Register model
zero_opt.register_model(model);

// Training loop
for (int step = 0; step < max_steps; ++step) {
    zero_opt.zero_grad();

    // Parameters automatically gathered/freed
    auto output = model.forward(batch.input);
    auto loss = criterion(output, batch.target);

    // Gradients automatically reduced-scattered
    loss.backward();

    // Update local partition
    zero_opt.step();

    // Monitor performance every 100 steps
    if (step % 100 == 0) {
        auto stats = zero_opt.get_stats();
        std::cout << "Step " << step << ":\n";
        std::cout << "  Prefetch hit rate: "
                  << (stats.prefetch_hit_rate * 100) << "%\n";
        std::cout << "  Overlap efficiency: "
                  << (stats.overlap_efficiency * 100) << "%\n";
        std::cout << "  Peak gathered memory: "
                  << stats.peak_gathered_memory_bytes / (1024*1024) << " MB\n";
    }

    // Save checkpoint every 1000 steps
    if (step % 1000 == 0) {
        zero_opt.save_checkpoint("checkpoints/step_" + std::to_string(step));
    }
}
```

### Example 4: ZeRO Stage 3 with Maximum Memory Efficiency

```cpp
// For training on limited GPU memory (e.g., 16GB GPUs)
Stage3Config config;
config.world_size = 8;
config.rank = distributed::get_rank();

// Aggressive CPU offloading
config.offload_params_to_cpu = true;
config.offload_gathered_to_cpu = true;
config.offload_to_cpu = true;  // Also offload optimizer states

// Conservative memory settings
config.prefetch_depth = 1;  // Minimal prefetch
config.prefetch_bucket_size = 50 * 1024 * 1024;  // 50MB
config.max_cached_params = 5;  // Limit cache
config.max_gathered_buffer_size = 200 * 1024 * 1024;  // 200MB max

// Enable all overlap features
config.overlap_comm_compute = true;
config.use_async_gather = true;

auto zero_opt = ZeROStage3Optimizer(std::move(base_opt), config);
zero_opt.register_model(model);

// Can now train 70B+ models on 8x 16GB GPUs
```

### Example 5: Checkpointing and Resuming

```cpp
// Save checkpoint
void save_training_state(
    const std::string& checkpoint_dir,
    int step,
    ZeROStage3Optimizer& optimizer,
    int rank
) {
    std::string prefix = checkpoint_dir + "/step_" + std::to_string(step);

    // Save optimizer state (all ranks)
    optimizer.save_checkpoint(prefix);

    // Master rank saves additional metadata
    if (rank == 0) {
        std::ofstream meta(prefix + "_train_meta.txt");
        meta << "step=" << step << "\n";
        meta << "timestamp=" << std::time(nullptr) << "\n";
        meta.close();
    }
}

// Load checkpoint
void load_training_state(
    const std::string& checkpoint_dir,
    int step,
    ZeROStage3Optimizer& optimizer,
    int rank
) {
    std::string prefix = checkpoint_dir + "/step_" + std::to_string(step);

    // Load optimizer state (all ranks)
    optimizer.load_checkpoint(prefix);

    // Master rank reads metadata
    if (rank == 0) {
        std::ifstream meta(prefix + "_train_meta.txt");
        std::string line;
        while (std::getline(meta, line)) {
            std::cout << "Loaded: " << line << "\n";
        }
    }
}

// Usage
int resume_step = 5000;
load_training_state("checkpoints", resume_step, zero_opt, rank);

// Continue training from step 5000
for (int step = resume_step; step < max_steps; ++step) {
    // Training loop...
}
```

---

## API Quick Reference

### ZeRO Stage 1

| Method | Description |
|--------|-------------|
| `ZeROStage1Optimizer(base_opt, config)` | Constructor |
| `step()` | Optimizer step with state partitioning |
| `zero_grad()` | Zero gradients |
| `state_dict()` | Get local state |
| `load_state_dict(state)` | Load local state |
| `save_checkpoint(prefix)` | Save distributed checkpoint |
| `load_checkpoint(prefix)` | Load distributed checkpoint |
| `get_memory_stats()` | Memory usage statistics |

### ZeRO Stage 2

Inherits all Stage 1 methods plus:

| Method | Description |
|--------|-------------|
| `register_backward_hooks()` | Enable gradient reduce-scatter |
| `get_bucket_stats()` | Gradient bucket statistics |

### ZeRO Stage 3

Inherits all Stage 2 methods plus:

| Method | Description |
|--------|-------------|
| `register_model(model)` | Register model (REQUIRED) |
| `unregister_model()` | Unregister model |
| `gather_parameter(param)` | Manually gather parameter |
| `gather_parameter_async(param)` | Async gather parameter |
| `free_gathered_parameter(param)` | Free gathered parameter |
| `prefetch_parameters(params)` | Manual prefetch |
| `get_parameter_state(param)` | Query parameter state |
| `is_parameter_gathered(param)` | Check if gathered |
| `pin_parameter(param)` | Pin parameter in memory |
| `unpin_parameter(param)` | Unpin parameter |
| `get_stats()` | Performance statistics |
| `get_prefetch_stats()` | Prefetch statistics |
| `gather_full_state()` | Gather full state for checkpoint |
| `load_full_state(state)` | Load full checkpoint |

---

**For more information, see**:
- [Best Practices Guide](PHASE7_BEST_PRACTICES.md)
- [Performance Tuning Guide](PHASE7_PERFORMANCE_TUNING.md)
- [Migration Guide](PHASE7_MIGRATION_GUIDE.md)
