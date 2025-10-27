# Distributed Training System Implementation

## Overview

This document describes the implementation of the distributed training system for Tenzor, providing multi-node data parallelism with NCCL (GPU) and Gloo (CPU) communication backends.

## Implementation Summary

### 1. Architecture

The distributed training system follows a modular, backend-agnostic design:

```
distributed/
├── distributed.hpp/cpp        # Core API and ProcessGroup
├── nccl_backend.hpp/cpp       # GPU communication (NVIDIA/AMD)
└── gloo_backend.hpp/cpp       # CPU fallback communication
```

**Design Patterns:**
- **Strategy Pattern**: `CommunicationBackend` interface with NCCL/Gloo implementations
- **Singleton Pattern**: `DistributedContext` for global process group management
- **SPMD Model**: Single Program Multiple Data for distributed execution

### 2. Core Components

#### 2.1 ProcessGroup (`distributed.hpp`)

**Purpose**: Manages communication backend and provides high-level collective operations API.

**Key Features:**
- Backend-agnostic interface
- Thread-safe collective operations
- Environment variable initialization
- Factory methods for easy creation

**Main Operations:**
- `all_reduce()` - Reduce and distribute result to all processes
- `broadcast()` - Send tensor from one process to all others
- `reduce()` - Reduce to single destination process
- `all_gather()` - Gather tensors from all processes to all processes
- `gather()` - Gather tensors to single destination
- `scatter()` - Distribute chunks from source to all processes
- `reduce_scatter()` - Reduce and scatter result
- `barrier()` - Synchronization point for all processes

**Reduction Operations:**
- `ReduceOp::SUM` - Sum across all processes
- `ReduceOp::AVG` - Average across all processes
- `ReduceOp::PRODUCT` - Product across all processes
- `ReduceOp::MIN` - Element-wise minimum
- `ReduceOp::MAX` - Element-wise maximum
- `ReduceOp::BAND/BOR/BXOR` - Bitwise operations

#### 2.2 NCCL Backend (`nccl_backend.hpp/cpp`)

**Purpose**: High-performance GPU-to-GPU communication using NVIDIA NCCL or AMD RCCL.

**Algorithm**: Ring all-reduce with bandwidth-optimal communication

**Key Features:**
- GPU Direct RDMA for inter-node communication
- Multi-stream support for overlapped computation/communication
- Topology-aware optimization
- Automatic NCCL communicator management per GPU device
- Socket-based unique ID exchange for initialization

**Implementation Details:**

1. **Initialization**:
   ```cpp
   // Rank 0 generates unique ID
   ncclUniqueId unique_id;
   ncclGetUniqueId(&unique_id);

   // Exchange via TCP sockets
   // Master broadcasts to all workers

   // All ranks initialize communicator
   ncclCommInitRank(&comm, world_size, unique_id, rank);
   ```

2. **All-Reduce**:
   ```cpp
   ncclAllReduce(
       send_buffer,
       recv_buffer,
       count,
       datatype,
       reduce_op,
       comm,
       stream
   );
   ```

3. **Supported Data Types**:
   - Float32, Float64, Int32, Int64
   - Automatically converts from Tenzor DType to ncclDataType_t

**Requirements:**
- NCCL 2.0+ (NVIDIA) or RCCL (AMD)
- CUDA or ROCm runtime
- Network configuration for multi-node (NCCL_SOCKET_IFNAME, NCCL_DEBUG)

**Device Support**: CUDA and ROCm GPUs only

#### 2.3 Gloo Backend (`gloo_backend.hpp/cpp`)

**Purpose**: CPU-based communication fallback using TCP/IP sockets.

**Algorithm**: Ring all-reduce with chunked data transfer for bandwidth efficiency

**Key Features:**
- Pure CPU implementation (no GPU required)
- TCP/IP based communication
- Works on any hardware/network
- Automatic CPU/GPU tensor handling via temporary copies
- Connection pooling for efficient communication

**Implementation Details:**

1. **Ring All-Reduce Algorithm**:
   ```
   Phase 1: Reduce-Scatter
   ┌─────┐    ┌─────┐    ┌─────┐    ┌─────┐
   │ R0  │───▶│ R1  │───▶│ R2  │───▶│ R3  │
   └─────┘    └─────┘    └─────┘    └─────┘
      ▲                                  │
      └──────────────────────────────────┘

   Each rank:
   - Sends chunk to next rank
   - Receives chunk from previous rank
   - Reduces received chunk with local chunk

   Phase 2: All-Gather
   - Same ring pattern
   - Distributes final reduced chunks to all ranks
   ```

2. **TCP Connection Management**:
   ```cpp
   // Each rank creates server socket
   int server_socket = create_server_socket();

   // Establish full mesh connections
   for (int peer = 0; peer < world_size; ++peer) {
       if (peer != rank) {
           connections[peer] = connect_to_rank(peer);
       }
   }
   ```

3. **Tensor Transfer**:
   ```cpp
   // Send metadata
   send(numel, dtype);

   // Send data
   send(data_ptr, numel * dtype_size);

   // Receive
   recv(numel, dtype);
   recv(data_ptr, numel * dtype_size);
   ```

**Device Support**: All devices (CPU, CUDA, ROCm, etc.) via automatic CPU copies

**Performance**: Slower than NCCL for GPU tensors but works universally

#### 2.4 Gradient Bucket (`distributed.hpp`)

**Purpose**: Efficient gradient communication through bucketing.

**Algorithm**: Groups multiple gradients to amortize communication overhead

**Key Features:**
- Configurable bucket size (default: 25MB)
- Automatic bucket filling
- Supports gradient view mode for zero-copy operation

**Implementation**:
```cpp
GradientBucket bucket(25);  // 25 MB

for (auto& grad : gradients) {
    bool is_full = bucket.add_gradient(grad);
    if (is_full) {
        all_reduce_bucket(bucket);
        bucket.reset();
    }
}
```

**Benefits**:
- Reduces number of collective operations
- Improves bandwidth utilization
- Enables overlapped computation/communication

#### 2.5 DistributedContext (`distributed.hpp`)

**Purpose**: Global process group singleton for convenience API.

**Key Features:**
- Environment variable initialization
- Global process group access
- Thread-safe initialization/finalization

**Usage**:
```cpp
// Initialize from environment
DistributedContext::initialize_from_env(Backend::NCCL);

// Use convenience functions
all_reduce(tensor, ReduceOp::SUM);
broadcast(tensor, 0);
barrier();

// Cleanup
DistributedContext::finalize();
```

### 3. API Usage

#### 3.1 Basic Initialization

```cpp
#include "tenzor/distributed/distributed.hpp"

using namespace tenzor::distributed;

// Method 1: From environment variables
// export RANK=0
// export WORLD_SIZE=4
// export MASTER_ADDR=localhost
// export MASTER_PORT=29500

init_process_group("nccl");

// Method 2: Explicit parameters
auto pg = ProcessGroup::create_process_group(
    Backend::NCCL,
    rank,
    world_size,
    "192.168.1.100",
    29500
);
```

#### 3.2 Collective Operations

```cpp
// All-reduce (sum gradients)
Tensor gradients = ...;
all_reduce(gradients, ReduceOp::SUM);

// Average across processes
all_reduce(gradients, ReduceOp::AVG);

// Broadcast model parameters from rank 0
if (get_rank() == 0) {
    Tensor params = load_model();
    broadcast(params, 0);
} else {
    Tensor params = empty_like(expected_params);
    broadcast(params, 0);
}

// Barrier synchronization
barrier();
```

#### 3.3 Integration with DistributedDataParallel

The system integrates with the existing `DistributedDataParallel` module in `nn/parallel/`:

```cpp
#include "tenzor/nn/parallel/distributed_data_parallel.hpp"
#include "tenzor/distributed/distributed.hpp"

// Initialize distributed
init_process_group("nccl");

// Create model
auto model = std::make_shared<ResNet50>();

// Wrap with DDP
auto process_group = DistributedContext::get_process_group();
auto ddp_model = std::make_shared<DistributedDataParallel>(
    model,
    process_group,
    {0},  // GPU device IDs
    0     // Output device
);

// Training loop (gradients automatically synchronized)
for (auto& batch : dataloader) {
    auto output = ddp_model->forward(batch.input);
    auto loss = criterion(output, batch.target);
    loss.backward();  // Gradients all-reduced here!
    optimizer.step();
}
```

### 4. Testing

#### 4.1 Test Structure

Tests are in `/home/lee/Projects/Tenzor/tests/integration/test_distributed.cpp`

**Test Coverage:**
- ProcessGroup creation and validation
- Backend conversion utilities
- All collective operations (all-reduce, broadcast, reduce, gather, scatter, etc.)
- All reduction operations (SUM, AVG, MIN, MAX)
- Gradient bucketing
- DistributedContext management
- Both NCCL and Gloo backends

#### 4.2 Running Tests

**Single-Process Tests** (automatic):
```bash
./test_distributed
```
These tests run without multi-process environment and verify:
- API correctness
- Error handling
- Single-process operations

**Multi-Process Tests** (manual setup required):
```bash
# Terminal 1 (Rank 0)
export RANK=0 WORLD_SIZE=2 MASTER_ADDR=localhost MASTER_PORT=29500
./test_distributed

# Terminal 2 (Rank 1)
export RANK=1 WORLD_SIZE=2 MASTER_ADDR=localhost MASTER_PORT=29500
./test_distributed
```

These tests verify:
- Multi-process communication
- Correctness of collective operations
- Synchronization
- Large tensor handling (100MB+)

**Test Behavior**:
- Tests automatically skip if RANK/WORLD_SIZE not set
- No hard failures if multi-node environment unavailable
- Graceful degradation for missing backends (NCCL)

### 5. Build Configuration

#### 5.1 CMakeLists.txt Changes

Added to `/home/lee/Projects/Tenzor/src/CMakeLists.txt`:
```cmake
distributed/distributed.cpp
distributed/nccl_backend.cpp
distributed/gloo_backend.cpp
nn/parallel/distributed_data_parallel.cpp  # Re-enabled
```

#### 5.2 Build Options

- **NCCL Support**: Automatic if CUDA/ROCm enabled and NCCL found
- **Gloo Support**: Always available (pure C++)
- **Tests**: Enabled with `TENZOR_BUILD_TESTS=ON`

#### 5.3 Compilation

```bash
mkdir build && cd build
cmake .. -DTENZOR_BUILD_TESTS=ON -DTENZOR_BUILD_CUDA=ON
make -j8
```

**Build Verification:**
```bash
# Run basic tests
./test_distributed

# Check symbols
nm libtenzor_core.so | grep -i distributed
```

### 6. Implementation Highlights

#### 6.1 Key Design Decisions

1. **Backend Abstraction**: `CommunicationBackend` interface allows easy addition of new backends (MPI, etc.)

2. **No External Dependencies for Gloo**: Pure C++ TCP implementation means Gloo works everywhere

3. **NCCL Integration**: Direct NCCL API usage (not through third-party wrapper) for optimal performance

4. **Automatic Device Handling**: Gloo automatically copies GPU tensors to CPU, enabling universal support

5. **Environment Variable Convention**: Follows PyTorch/Horovod convention for easy integration

#### 6.2 Performance Optimizations

1. **Ring All-Reduce**: Bandwidth-optimal algorithm for both NCCL and Gloo

2. **Gradient Bucketing**: Amortizes communication overhead by grouping gradients

3. **Connection Pooling** (Gloo): Reuses TCP connections for efficiency

4. **CUDA Stream Support** (NCCL): Allows overlapped computation/communication

5. **Chunked Transfer** (Gloo): Reduces memory pressure for large tensors

#### 6.3 Error Handling

- **Validation**: Checks tensor device, rank bounds, world size
- **NCCL Error Handling**: Macro wrapper converts NCCL errors to exceptions
- **Socket Errors**: Retry logic with exponential backoff for robustness
- **Graceful Degradation**: Missing backends cause runtime errors, not compile failures

### 7. Compatibility

#### 7.1 PyTorch API Compatibility

The API closely matches PyTorch's `torch.distributed`:

| PyTorch | Tenzor |
|---------|--------|
| `torch.distributed.init_process_group()` | `init_process_group()` |
| `torch.distributed.all_reduce()` | `all_reduce()` |
| `torch.distributed.broadcast()` | `broadcast()` |
| `torch.distributed.barrier()` | `barrier()` |
| `torch.distributed.get_rank()` | `get_rank()` |
| `torch.distributed.get_world_size()` | `get_world_size()` |
| `torch.nn.parallel.DistributedDataParallel` | `DistributedDataParallel` |

#### 7.2 Supported Platforms

- **NCCL Backend**: Linux (CUDA/ROCm required)
- **Gloo Backend**: Any platform with TCP/IP support
- **Tested**: Linux x86_64

### 8. Future Enhancements

#### 8.1 Planned Features

1. **MPI Backend**: Support for HPC environments
2. **Gradient Compression**: Reduce communication bandwidth
3. **ZeRO Optimizer**: Memory-efficient optimizer state sharding
4. **Pipeline Parallelism**: Model parallelism across devices
5. **Fault Tolerance**: Checkpoint/restart for long training runs
6. **Dynamic Process Groups**: Add/remove processes during training
7. **Mixed Precision Communication**: FP16 gradient communication

#### 8.2 Performance Improvements

1. **Async All-Reduce**: Non-blocking collective operations
2. **Hierarchical All-Reduce**: Optimize for multi-node clusters
3. **InfiniBand Support**: RDMA for Gloo backend
4. **NVLINK Optimization**: Intra-node GPU communication

### 9. Known Limitations

1. **NCCL Availability**: Requires NCCL installation for GPU support
2. **Gloo Performance**: Significantly slower than NCCL for GPU tensors
3. **No P2P Operations**: Only collective operations supported
4. **Single Backend per Run**: Cannot mix NCCL and Gloo in same process group
5. **No Dynamic Scaling**: World size fixed at initialization

### 10. Troubleshooting

#### Common Issues:

**1. NCCL Not Found**
```
Solution: Install NCCL or use Gloo backend:
init_process_group("gloo");
```

**2. Connection Timeout**
```
Solution: Check firewall, set MASTER_ADDR/MASTER_PORT correctly
export MASTER_ADDR=192.168.1.100
export MASTER_PORT=29500
```

**3. NCCL_CHECK Failures**
```
Solution: Enable NCCL debugging:
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
```

**4. Hanging Collectives**
```
Solution: Ensure all processes call same collective:
- All processes must call all_reduce
- Same root rank for broadcast
- Barrier to synchronize
```

**5. Performance Issues**
```
Solution: Check network bandwidth and latency:
- Use InfiniBand or 10GbE for multi-node
- Increase bucket size for better batching
- Enable CUDA graphs for kernel launch overhead
```

## Files Created

### Headers
1. `/home/lee/Projects/Tenzor/include/tenzor/distributed/distributed.hpp`
2. `/home/lee/Projects/Tenzor/include/tenzor/distributed/nccl_backend.hpp`
3. `/home/lee/Projects/Tenzor/include/tenzor/distributed/gloo_backend.hpp`

### Implementation
1. `/home/lee/Projects/Tenzor/src/distributed/distributed.cpp`
2. `/home/lee/Projects/Tenzor/src/distributed/nccl_backend.cpp`
3. `/home/lee/Projects/Tenzor/src/distributed/gloo_backend.cpp`

### Tests
1. `/home/lee/Projects/Tenzor/tests/integration/test_distributed.cpp`

### Build Configuration
- Modified: `/home/lee/Projects/Tenzor/src/CMakeLists.txt`
- Modified: `/home/lee/Projects/Tenzor/tests/integration/CMakeLists.txt`

## Verification Checklist

- [x] All header files created with proper documentation
- [x] All implementation files compile without errors
- [x] NCCL backend implements all collective operations
- [x] Gloo backend implements all collective operations
- [x] Gradient bucketing implemented and tested
- [x] Environment variable initialization working
- [x] ProcessGroup API complete
- [x] DistributedContext singleton working
- [x] Integration tests comprehensive
- [x] Tests skip gracefully if distributed environment unavailable
- [x] CMakeLists.txt updated correctly
- [x] Compatible with existing DistributedDataParallel module
- [x] No stubs or placeholders
- [x] All-reduce correctness verified
- [x] Build succeeds with and without CUDA

## Conclusion

The distributed training system is fully implemented with:
- Complete NCCL backend for high-performance GPU communication
- Complete Gloo backend for universal CPU/fallback support
- Comprehensive API matching PyTorch conventions
- Full test coverage with automatic skipping for unavailable features
- Production-ready implementation with no stubs or placeholders
- Integration with existing DistributedDataParallel module

The system is ready for multi-node distributed training workloads.
