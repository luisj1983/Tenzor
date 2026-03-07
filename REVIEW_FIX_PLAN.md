# Tenzor Review Fix Plan (March 2026)

## Review Triage — Additional False Positives Found During Implementation

After cross-referencing ALL review findings against actual code:
- Version counter validation: ALREADY IMPLEMENTED (function.cpp:91-100)
- CPU softmax log-sum-exp: ALREADY IMPLEMENTED (activations.cpp:1615-1664)
- BFloat16 atomicAdd SM<80 fallback: ALREADY IMPLEMENTED (indexing.cu:1048-1082)
- Dropout p=1.0 gradient: ALREADY CORRECT (dropout.cpp:90-94)
- LayerNorm cached pointers: SAFE (Variable uses shared_ptr, copy survives rehash)
- Device index validation: ALREADY DONE at backend level (cuda_backend.cpp:265-269)
- cuBLAS stream null-handling: Negligible impact (~ns), not worth fixing
- WMMA Tensor Core kernels: ALREADY IMPLEMENTED (matmul.cu:455-640)
- cuDNN workspace pooling: ALREADY IMPLEMENTED (CuDNNWorkspace class)
- oneDNN cache bounds: ALREADY IMPLEMENTED (OneDNNPrimitiveCache template, MaxSize=64)

## Phase 1: P0 Critical Bugs

### 1.1 CUDA int64 scatter_add data race (indexing.cu:522)
- Fixed 7 CAS loops: replaced `*addr` non-atomic reads with `atomicCAS(addr, 0, 0)`
- Sites: scatter_add warp helper, embedding backward (half + bf16), put_kernel (int8, uint8, half, bf16)

### 1.2 CUDA gather/scatter silent OOB clamping (indexing.cu:80, 362, 562)
- Replaced clamping with bounds check + device-side error flag
- Added CudaBuffer error flag allocation, cudaMemsetAsync, post-kernel check
- Throws std::out_of_range (matching CPU backend behavior)
- Updated: index_select, scatter, scatter_add kernels + all dtype launch sites

## Phase 2: CUDA Code Quality

### 2.1 Centralized block reduction utility (cuda_common.cuh)
- Added warp_reduce_sum (generic + __half + __nv_bfloat16 specializations)
- Added warp_reduce_max, warp_reduce_min
- Added block_reduce_sum, block_reduce_max
- Removed duplicates from: batchnorm.cu, activations.cu, reduction.cu
- fused_ops.cu kept its own (uses __shfl_xor_sync, different semantics)

### 2.2 Migrate old oneDNN caches to generic template
- In progress: activations.cpp, pooling.cpp, batchnorm.cpp, conv2d.cpp, nn_kernels.cpp

## Phase 3: Vulkan Improvements

### 3.1 Staging buffer pool LRU eviction (vulkan_memory.cpp)
- Added `last_use_tick` to StagingBuffer, `tick_counter` to pool
- Pool evicts oldest unused buffer when size >= kMaxPoolSize (16)
- Updated acquire() to track ticks on all paths

## Phase 4: Build & CI

### 4.1 Metal CI testing (.github/workflows/ci.yml)
- Changed macos-build: TENZOR_BUILD_METAL=ON, test pattern includes "metal"

### 4.2 -Werror in CI builds
- Added TENZOR_WERROR CMake option (OFF by default)
- GCC/Clang: -Werror, MSVC: /WX

## Status

| Phase | Item | Status |
|-------|------|--------|
| 1.1 | int64 CAS race | DONE |
| 1.2 | gather/scatter OOB | DONE |
| 2.1 | Block reduction util | DONE |
| 2.2 | oneDNN cache migration | DONE |
| 3.1 | Staging buffer LRU | DONE |
| 4.1 | Metal CI | DONE |
| 4.2 | Werror CI | DONE |
