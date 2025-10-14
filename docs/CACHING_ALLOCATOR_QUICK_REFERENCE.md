# CachingAllocator Quick Reference

## Enable/Disable

```bash
# Enable caching allocator
export TENZOR_ENABLE_CACHING_ALLOCATOR=1

# Disable (use standard cudaMalloc/cudaFree)
unset TENZOR_ENABLE_CACHING_ALLOCATOR
```

## Basic API

```cpp
#include "tenzor/backend/caching_allocator.hpp"
using namespace tenzor::backend;

// Get singleton
auto& allocator = CachingAllocator::get();

// Allocate
void* ptr = allocator.allocate(size_bytes, device_id);

// Free (returns to cache)
allocator.free(ptr, device_id);

// Empty cache
allocator.empty_cache(device_id);  // or -1 for all devices
```

## Statistics

```cpp
// Get statistics
auto stats = allocator.get_stats(device_id);  // or -1 for all devices

std::cout << "Allocated: " << stats.allocated_bytes << " bytes\n";
std::cout << "Reserved:  " << stats.reserved_bytes << " bytes\n";
std::cout << "Cached:    " << stats.cached_bytes << " bytes\n";
std::cout << "Cache hit rate: "
          << (100.0 * stats.num_cache_hits / stats.num_allocations) << "%\n";
std::cout << "Splits: " << stats.num_splits << "\n";
std::cout << "Merges: " << stats.num_merges << "\n";

// Reset counters
allocator.reset_stats();
```

## Configuration

```cpp
// Set alignment (must be power of 2)
allocator.set_alignment(1024);  // 1KB

// Set maximum cached memory per device (0 = unlimited)
allocator.set_max_cached_memory(1024 * 1024 * 1024);  // 1GB

// Enable/disable block merging
allocator.set_merge_enabled(true);

// Set minimum block size for splitting
allocator.set_min_split_size(512);  // 512 bytes
```

## RAII Wrapper

```cpp
{
    CachedMemoryGuard guard(size_bytes, device_id);
    void* ptr = guard.get();
    size_t size = guard.size();

    // Use memory...

} // Automatically freed here
```

## Build and Test

```bash
cd /home/lee/Projects/Tenzor/build

# Configure
cmake .. -DTENZOR_BUILD_CUDA=ON

# Build
make -j$(nproc)

# Run tests
ctest -R test_caching_allocator -V

# Run with caching enabled
TENZOR_ENABLE_CACHING_ALLOCATOR=1 ./tests/test_caching_allocator

# Run benchmarks only
./tests/test_caching_allocator --gtest_filter="*Benchmark*"
```

## Common Use Cases

### Training Loop
```cpp
auto& allocator = CachingAllocator::get();

for (int epoch = 0; epoch < num_epochs; epoch++) {
    for (auto& batch : dataloader) {
        // Allocations will be cached and reused
        auto output = model->forward(batch.input);
        auto loss = criterion->forward(output, batch.target);

        optimizer->zero_grad();
        loss.backward();
        optimizer->step();
    }

    // Optional: empty cache between epochs
    allocator.empty_cache();
}
```

### Dynamic Batch Sizes
```cpp
// Allocator automatically handles varying sizes
for (size_t batch_size : {16, 32, 64, 128}) {
    Tensor input({batch_size, 3, 224, 224});
    auto output = model->forward(input);
    // ...
}
```

### Memory Monitoring
```cpp
auto stats_before = allocator.get_stats();

// Run model
auto output = model->forward(input);

auto stats_after = allocator.get_stats();

std::cout << "Memory allocated: "
          << (stats_after.allocated_bytes - stats_before.allocated_bytes)
          << " bytes\n";
std::cout << "Cache hits: "
          << (stats_after.num_cache_hits - stats_before.num_cache_hits)
          << "\n";
```

## Performance Tips

1. **Pre-warm the cache**: Run a few iterations before benchmarking
2. **Monitor hit rates**: Aim for >70% for repetitive workloads
3. **Tune cache limits**: Balance memory usage with hit rate
4. **Empty cache strategically**: Between major model changes, not every iteration
5. **Profile first**: Measure actual speedup for your workload

## Troubleshooting

### Out of Memory
```cpp
try {
    void* ptr = allocator.allocate(large_size, device_id);
} catch (const std::runtime_error& e) {
    // Try emptying cache
    allocator.empty_cache(device_id);

    // Retry
    void* ptr = allocator.allocate(large_size, device_id);
}
```

### Memory Leaks
```cpp
// Check for leaks
auto stats = allocator.get_stats();
if (stats.allocated_bytes > 0) {
    std::cerr << "Warning: " << stats.allocated_bytes
              << " bytes still allocated\n";
}

// Force cleanup
allocator.empty_cache();
```

### Low Hit Rate
```cpp
auto stats = allocator.get_stats();
float hit_rate = 100.0 * stats.num_cache_hits / stats.num_allocations;

if (hit_rate < 50.0) {
    std::cout << "Low hit rate (" << hit_rate << "%).\n";
    std::cout << "Consider:\n";
    std::cout << "  - Increasing max_cached_memory\n";
    std::cout << "  - Disabling merging for specific patterns\n";
    std::cout << "  - Adjusting min_split_size\n";
}
```

## Architecture Summary

```
CachingAllocator (Singleton)
├── Per-Device Pools (DeviceAllocator)
│   ├── Free Blocks (std::set, ordered by size)
│   ├── All Blocks (std::unordered_map, by pointer)
│   └── Statistics (MemoryStats)
│
├── Allocation Flow
│   1. Check cache (best-fit)
│   2. Split if needed
│   3. Allocate new if no match
│
├── Deallocation Flow
│   1. Mark block as free
│   2. Merge with adjacent blocks
│   3. Add to cache
│   4. Enforce cache limit
│
└── Thread Safety
    └── Global mutex for all operations
```

## Files

- **Header**: `/home/lee/Projects/Tenzor/include/tenzor/backend/caching_allocator.hpp`
- **Implementation**: `/home/lee/Projects/Tenzor/src/backend/caching_allocator.cpp`
- **Tests**: `/home/lee/Projects/Tenzor/tests/unit/test_caching_allocator.cpp`
- **Documentation**: `/home/lee/Projects/Tenzor/docs/CACHING_ALLOCATOR_IMPLEMENTATION.md`
- **Summary**: `/home/lee/Projects/Tenzor/docs/CACHING_ALLOCATOR_SUMMARY.txt`

## Further Reading

- Full implementation details: `docs/CACHING_ALLOCATOR_IMPLEMENTATION.md`
- Complete summary: `docs/CACHING_ALLOCATOR_SUMMARY.txt`
- Test examples: `tests/unit/test_caching_allocator.cpp`
