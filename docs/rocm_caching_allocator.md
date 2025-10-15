# ROCm Caching Allocator

## Overview

The ROCm Caching Allocator is a production-grade memory management system for AMD GPUs that provides efficient allocation and reuse of GPU memory. It mirrors the CUDA implementation but is optimized for AMD's ROCm platform and HBM memory architecture.

## Features

### Core Features
- **Block-based Memory Management**: Efficient tracking and reuse of memory blocks
- **Best-fit Allocation Strategy**: Minimizes memory fragmentation
- **Block Splitting and Merging**: Optimizes memory usage
- **Per-stream Memory Pools**: Improves locality for concurrent kernel execution
- **Multi-GPU Support**: Per-device memory pools
- **Memory Statistics Tracking**: Comprehensive usage metrics
- **Configurable Memory Limits**: Prevents excessive memory usage
- **Garbage Collection**: Reduces fragmentation over time
- **Thread-safe Operations**: Full mutex protection

### AMD GPU Optimizations
- **HBM Memory Support**: Optimized for MI series GPUs (MI50, MI100, MI200, MI300)
- **256-byte Alignment**: Optimal for HBM memory coalescing
- **Large Allocation Handling**: Efficient handling of >2GB allocations
- **Device-specific Properties**: Auto-detection of HBM and compute capabilities
- **Memory Pressure Handling**: Automatic retry with garbage collection

## Architecture

### Memory Block Structure

```cpp
struct Block {
    void* ptr;              // Device pointer
    size_t size;            // Block size in bytes
    bool allocated;         // Whether block is currently allocated
    int device;             // HIP device ID
    hipStream_t stream;     // Associated stream
    size_t alignment;       // Block alignment (256 for HBM)
};
```

### Per-Device Allocator

Each device maintains:
- **Free blocks**: Sorted by size for best-fit allocation
- **All blocks**: Hash map of all blocks (free and allocated)
- **Statistics**: Memory usage tracking
- **Device properties**: GPU capabilities and memory info
- **Stream blocks**: Per-stream block tracking

## API Reference

### Singleton Access

```cpp
RocmCachingAllocator& allocator = RocmCachingAllocator::get();
```

### Memory Allocation

```cpp
void* allocate(size_t size, int device = 0, hipStream_t stream = nullptr);
```

Allocates memory from the cache or device. Features:
- Returns nullptr for zero-size requests
- Rounds size to alignment boundary
- Tries cache first (best-fit)
- Falls back to device allocation
- Retries with garbage collection on failure
- Thread-safe

**Example:**
```cpp
// Allocate 1 MB on device 0
void* ptr = allocator.allocate(1024 * 1024, 0);

// Use memory...

allocator.free(ptr, 0);
```

### Memory Deallocation

```cpp
void free(void* ptr, int device = 0);
```

Returns memory to the cache. Features:
- Validates pointer belongs to allocator
- Detects double-free
- Attempts block merging
- Enforces cache limits
- Thread-safe

### Cache Management

```cpp
// Empty all cached memory
void empty_cache(int device = -1);  // -1 = all devices

// Garbage collect to reduce fragmentation
void garbage_collect(int device = -1, bool aggressive = false);
```

**Garbage Collection Strategies:**
- **Normal**: Frees large blocks (>2GB) and blocks over cache limit
- **Aggressive**: Frees all cached blocks

### Statistics

```cpp
MemoryStats get_stats(int device = -1) const;

struct MemoryStats {
    size_t allocated_bytes;     // Currently allocated
    size_t reserved_bytes;      // Total reserved (allocated + cached)
    size_t cached_bytes;        // Cached but not allocated
    size_t num_allocations;     // Total allocation calls
    size_t num_frees;           // Total free calls
    size_t num_cache_hits;      // Cache hit count
    size_t num_splits;          // Block split count
    size_t num_merges;          // Block merge count
    size_t peak_allocated;      // Peak allocated memory
    size_t peak_reserved;       // Peak reserved memory
    size_t num_oom_errors;      // Out-of-memory errors
    size_t hbm_bytes;           // HBM-optimized allocations
};
```

### Device Properties

```cpp
DeviceProperties get_device_properties(int device) const;

struct DeviceProperties {
    size_t total_memory;        // Total device memory
    size_t available_memory;    // Available memory
    bool has_hbm;               // Has HBM (MI series)
    int compute_units;          // Number of compute units
    size_t max_shared_memory;   // Max shared memory per block
    int warp_size;              // Warp size (typically 64 for AMD)
    std::string device_name;    // Device name
};
```

### Configuration

```cpp
// Memory alignment (default: 256 bytes for HBM)
void set_alignment(size_t alignment);

// Maximum cached memory per device (default: unlimited)
void set_max_cached_memory(size_t max_bytes);

// Enable/disable block merging (default: enabled)
void set_merge_enabled(bool enable);

// Minimum block size for splitting (default: 512 bytes)
void set_min_split_size(size_t min_size);

// Enable/disable logging (default: disabled)
void set_logging_enabled(bool enable);

// Large allocation threshold (default: 2GB)
void set_large_alloc_threshold(size_t threshold);
```

### RAII Wrapper

```cpp
RocmCachedMemoryGuard guard(size_t size, int device = 0, hipStream_t stream = nullptr);

// Automatic memory management
void* ptr = guard.get();
size_t size = guard.size();
// Memory automatically freed when guard goes out of scope
```

## CUDA to HIP API Conversion

| CUDA API | HIP API |
|----------|---------|
| `cudaMalloc` | `hipMalloc` |
| `cudaFree` | `hipFree` |
| `cudaMemGetInfo` | `hipMemGetInfo` |
| `cudaStreamSynchronize` | `hipStreamSynchronize` |
| `cudaDeviceSynchronize` | `hipDeviceSynchronize` |
| `cudaSetDevice` | `hipSetDevice` |
| `cudaGetDeviceProperties` | `hipGetDeviceProperties` |
| `cudaGetErrorString` | `hipGetErrorString` |
| `cudaStream_t` | `hipStream_t` |
| `cudaError_t` | `hipError_t` |

## AMD GPU Support

### Supported Architectures

| Architecture | GPU Series | HBM Support |
|-------------|-----------|-------------|
| gfx900 | Vega (MI25, RX Vega) | No |
| gfx906 | Vega 7nm (MI50/MI60) | Yes |
| gfx908 | CDNA (MI100) | Yes |
| gfx90a | CDNA2 (MI200 series) | Yes |
| gfx940 | CDNA3 (MI300) | Yes |
| gfx1030 | RDNA2 (RX 6000) | No |
| gfx1100 | RDNA3 (RX 7000) | No |

### HBM Optimization

The allocator automatically detects HBM-equipped GPUs (MI series) and applies:
- 256-byte alignment for optimal memory coalescing
- Large block allocation strategies
- HBM-specific statistics tracking

**Detection Logic:**
```cpp
bool has_hbm = device_name.find("MI") != std::string::npos ||
               device_name.find("Instinct") != std::string::npos;
```

## Performance Characteristics

### Memory Allocation Strategy

1. **Request arrives** → Round to alignment (256 bytes)
2. **Check cache** → Best-fit search in free blocks
3. **Cache hit** → Return cached block (split if needed)
4. **Cache miss** → Allocate from device with hipMalloc
5. **Failure** → Trigger garbage collection and retry

### Block Management

- **Splitting**: Occurs when cached block is ≥ requested size + min_split_size
- **Merging**: Forward merging with adjacent free blocks
- **Alignment**: All blocks aligned to 256 bytes (HBM optimal)
- **Minimum block**: 512 bytes (prevents excessive fragmentation)

### Thread Safety

- Global mutex protects all operations
- Lock-free reads not supported (requires synchronization)
- Thread-local caches not implemented (future optimization)

## Usage Examples

### Basic Usage

```cpp
#include "tenzor/backend/rocm_caching_allocator.hip.hpp"

using namespace tenzor::backend::rocm;

// Get allocator instance
auto& allocator = RocmCachingAllocator::get();

// Allocate 10 MB
void* ptr = allocator.allocate(10 * 1024 * 1024, 0);

// Use memory...
hipMemset(ptr, 0, 10 * 1024 * 1024);

// Free memory
allocator.free(ptr, 0);
```

### RAII Pattern

```cpp
{
    RocmCachedMemoryGuard guard(1024 * 1024, 0);
    void* ptr = guard.get();

    // Use memory...
    hipMemset(ptr, 0, guard.size());

} // Automatically freed
```

### Multi-stream Usage

```cpp
hipStream_t stream1, stream2;
hipStreamCreate(&stream1);
hipStreamCreate(&stream2);

// Allocate for different streams
void* ptr1 = allocator.allocate(1024, 0, stream1);
void* ptr2 = allocator.allocate(1024, 0, stream2);

// Use in parallel...

allocator.free(ptr1, 0);
allocator.free(ptr2, 0);

hipStreamDestroy(stream1);
hipStreamDestroy(stream2);
```

### Memory Management

```cpp
// Configure allocator
allocator.set_alignment(256);              // HBM optimal
allocator.set_max_cached_memory(1ULL << 30);  // 1 GB cache limit
allocator.set_logging_enabled(true);       // Enable debug logging

// Perform allocations...

// Monitor usage
auto stats = allocator.get_stats(0);
std::cout << "Allocated: " << stats.allocated_bytes / (1024*1024) << " MB\n";
std::cout << "Cached: " << stats.cached_bytes / (1024*1024) << " MB\n";
std::cout << "Cache hit rate: "
          << (100.0 * stats.num_cache_hits / stats.num_allocations) << "%\n";

// Cleanup
allocator.garbage_collect(0, false);  // Normal GC
allocator.empty_cache(0);             // Clear all cached memory
```

### Error Handling

```cpp
try {
    void* ptr = allocator.allocate(large_size, 0);
    // Use memory...
    allocator.free(ptr, 0);
} catch (const std::runtime_error& e) {
    std::cerr << "Allocation failed: " << e.what() << "\n";

    // Check device memory
    auto props = allocator.get_device_properties(0);
    std::cerr << "Available memory: "
              << props.available_memory / (1024*1024) << " MB\n";

    // Try garbage collection
    allocator.garbage_collect(0, true);
}
```

## Testing

Comprehensive test suite available in `tests/backend/test_rocm_caching_allocator.cpp`:

- Basic allocation and deallocation
- Cache reuse and hit rate
- Block splitting and merging
- Memory alignment verification
- Cache limits enforcement
- Garbage collection
- Thread safety (multi-threaded stress test)
- RAII wrapper
- Error handling (double-free, invalid pointers)
- Statistics tracking
- Large allocations
- HBM detection
- Multi-GPU support

**Run tests:**
```bash
cd build
ctest -R test_rocm_caching_allocator -V
```

## Performance Tips

1. **Alignment**: Use 256-byte alignment for HBM GPUs
2. **Cache Limits**: Set reasonable limits to prevent memory exhaustion
3. **Garbage Collection**: Run periodically during idle periods
4. **Logging**: Disable in production for performance
5. **Block Splitting**: Adjust min_split_size based on allocation patterns
6. **Stream Pooling**: Allocate per-stream for better locality

## Troubleshooting

### Out of Memory Errors

```cpp
auto props = allocator.get_device_properties(0);
auto stats = allocator.get_stats(0);

std::cout << "Total memory: " << props.total_memory / (1024*1024) << " MB\n";
std::cout << "Available: " << props.available_memory / (1024*1024) << " MB\n";
std::cout << "Reserved: " << stats.reserved_bytes / (1024*1024) << " MB\n";
std::cout << "Allocated: " << stats.allocated_bytes / (1024*1024) << " MB\n";
std::cout << "Cached: " << stats.cached_bytes / (1024*1024) << " MB\n";
std::cout << "OOM errors: " << stats.num_oom_errors << "\n";

// Try garbage collection
allocator.garbage_collect(0, true);
```

### Memory Fragmentation

```cpp
auto stats = allocator.get_stats(0);
double fragmentation = 1.0 - (double)stats.allocated_bytes / stats.reserved_bytes;
std::cout << "Fragmentation: " << (100.0 * fragmentation) << "%\n";

// Reduce fragmentation
allocator.set_merge_enabled(true);
allocator.garbage_collect(0, true);
```

### Poor Cache Hit Rate

```cpp
auto stats = allocator.get_stats(0);
double hit_rate = (double)stats.num_cache_hits / stats.num_allocations;
std::cout << "Cache hit rate: " << (100.0 * hit_rate) << "%\n";

// Improve hit rate
allocator.set_min_split_size(256);  // Reduce splitting
allocator.set_merge_enabled(true);  // Enable merging
```

## Implementation Notes

### Differences from CUDA Implementation

1. **API Names**: CUDA → HIP conversion (cudaMalloc → hipMalloc)
2. **Alignment**: 256 bytes (HBM optimal) vs 512 bytes (CUDA default)
3. **HBM Detection**: Automatic detection of MI series GPUs
4. **Error Messages**: Use hipGetErrorString instead of cudaGetErrorString
5. **Device Properties**: hipDeviceProp_t instead of cudaDeviceProp
6. **Warp Size**: Typically 64 for AMD vs 32 for NVIDIA

### Thread Safety

- Global mutex protects all public methods
- Device allocators are not thread-local (future optimization)
- Statistics updates are atomic via mutex
- Block operations (split/merge) are protected

### Memory Overhead

- Per-block overhead: ~64 bytes (Block struct + unique_ptr)
- Set node overhead: ~32 bytes (std::set internal node)
- Map entry overhead: ~48 bytes (std::unordered_map entry)
- Total overhead: ~0.1-0.5% for typical workloads

## Future Enhancements

- [ ] Thread-local caches to reduce lock contention
- [ ] Support for unified memory (managed memory)
- [ ] Integration with rocProf for profiling
- [ ] Dynamic alignment based on allocation size
- [ ] Backward block merging (currently only forward)
- [ ] Memory compaction for highly fragmented scenarios
- [ ] Support for memory pools per kernel
- [ ] Integration with MIOpen memory management

## References

- [ROCm Documentation](https://rocm.docs.amd.com/)
- [HIP Programming Guide](https://rocm.docs.amd.com/projects/HIP/)
- [AMD GPU Architecture](https://www.amd.com/en/technologies/cdna)
- [HBM Memory Technology](https://en.wikipedia.org/wiki/High_Bandwidth_Memory)
