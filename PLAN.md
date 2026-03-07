# Tenzor Comprehensive Fix & Feature Plan

Based on thorough review of 525K LOC across 1,226 source files, with all issues
verified against actual code. Items marked NOT-A-BUG were confirmed as intentional
design or already fixed and are excluded.

## Completion Status

| Phase | Status | Notes |
|-------|--------|-------|
| 1 (P0 Critical Bugs) | ✅ COMPLETE | OneAPI copy, Metal sync, ROCm NMS RAII |
| 2 (P1 High Priority) | ✅ COMPLETE | Metal MPS already handled; ROCm async+tracking; OneAPI logging |
| 3.1 (EmbeddingBag GPU) | ✅ COMPLETE | CUDA+CPU kernels, OpId, embedding.cpp dispatch |
| 3.2 (Vulkan Shaders) | ✅ ALREADY DONE | Ninja parallelizes add_custom_command natively |
| 4 (Higher-Order Grad) | ✅ ALREADY DONE | All ~65 backward ops have backward_with_variables |
| 5 (SyncBatchNorm) | ⏭ SKIPPED | Requires distributed testing infrastructure |
| 6 (Sparse NN) | ✅ COMPLETE | SparseLinear + SparseEmbedding implemented |
| 7 (Quantization+QDQ) | ✅ COMPLETE | 5 quantized layers (Embedding,LSTM,GRU,Conv3d,MHA) + ONNX QDQ |
| 8.1 (OneAPI CMake) | ✅ COMPLETE | Centralized compile function, common flags |
| 8.2 (GPU C++ Docs) | ✅ COMPLETE | Comments added to CUDA/ROCm/OneAPI CMakeLists |
| 9.1 (CUDA RAII) | ✅ COMPLETE | CudaAsyncBuffer wrapper, matmul.cu updated |
| 9.2 (Reduction Errors) | ✅ COMPLETE | CUDA_PEEK_AND_THROW macro, 10 sites updated |

---

## Phase 1: Critical Backend Bug Fixes (P0)

### 1.1 OneAPI Cross-Device Copy Queue Selection
- **File:** `src/backends/oneapi/oneapi_backend.cpp:543-547`
- **Bug:** All copy ops hardcode `devices_[0].queue` regardless of source/dest device
- **Fix:** Determine correct queue from pointer device attributes or destination device_id
  - For H2D: use destination device queue
  - For D2H: use source device queue
  - For D2D: use destination device queue, stage through host if different platforms
- **Test:** Add backend test with multi-device copy (if 2+ OneAPI devices available)

### 1.2 Metal Shared Buffer GPU-CPU Synchronization
- **File:** `src/backends/metal/metal_backend.mm:122-134`
- **Bug:** `memcpy_h2d`/`memcpy_d2h` call `[buffer contents]` without GPU synchronization
- **Fix:**
  - Add `[commandBuffer waitUntilCompleted]` or equivalent fence before D2H reads
  - For H2D, add memory barrier after write to ensure GPU sees updated data
  - Consider switching to private buffers + blit encoder for correctness
- **Test:** Add test that writes GPU data then immediately reads back

### 1.3 ROCm NMS Kernel Resource Leak
- **File:** `src/backends/rocm/kernels/nms.hip.cpp:111-180`
- **Bug:** hipMalloc/hipMemcpy/hipMemset have no error checks; leak on failure
- **Fix:**
  - Add `HIP_CHECK()` macro around all hip* calls
  - Use RAII wrapper or goto-cleanup pattern for d_sorted_indices, d_suppression_mask
  - On hipMalloc failure, throw before using uninitialized pointer
- **Test:** Existing NMS tests should pass; add edge case with 0 boxes

---

## Phase 2: High-Priority Backend Fixes (P1)

### 2.1 Metal MPS Error Handling
- **File:** `src/backends/metal/metal_backend.mm:240-277`
- **Bug:** MPS matmul operations don't check MTLCommandBuffer errors
- **Fix:**
  - After `[commandBuffer waitUntilCompleted]`, check `[commandBuffer status]`
  - If `MTLCommandBufferStatusError`, throw with `[commandBuffer error].localizedDescription`
  - Apply same pattern to all MPS-dispatched operations
- **Test:** Verify error propagation with invalid inputs

### 2.2 ROCm Async Memory Copies
- **File:** `src/backends/rocm/rocm_backend.cpp:167-172`
- **Bug:** Uses blocking `hipMemcpy()` instead of `hipMemcpyAsync()`
- **Fix:**
  - Replace `hipMemcpy()` with `hipMemcpyAsync()` using device stream
  - Add stream parameter plumbing if not already available
  - Keep synchronous fallback for cases where no stream is available
- **Test:** Verify async copy correctness with existing ROCm tests

### 2.3 ROCm Device ID Fallback
- **File:** `src/backends/rocm/rocm_backend.cpp:130-139`
- **Bug:** Falls back to device 0 silently when hipPointerGetAttributes fails
- **Fix:**
  - Log warning when pointer attribute lookup fails
  - Store device_id in allocation tracking map (like CUDA caching allocator does)
  - Look up device_id from tracking map first, fall back to hipPointerGetAttributes
- **Test:** Verify correct device deallocation with multi-device scenario

### 2.4 OneAPI Silent Device Skip Logging
- **File:** `src/backends/oneapi/oneapi_backend.cpp:389-392`
- **Bug:** Silently catches exception and skips device with no log
- **Fix:**
  - Add `LOG_WARNING("Skipping SYCL device '{}': {}", device_name, e.what())`
  - After loop, if no devices initialized, throw descriptive error
- **Test:** Manual verification with logging output

---

## Phase 3: Performance Fixes (P1)

### 3.1 EmbeddingBag Fused GPU Kernel
- **File:** `src/nn/layers/embedding.cpp:889-892`
- **Bug:** GPU embeddings transferred to CPU for aggregation, then back to GPU
- **Fix:**
  - Implement CUDA kernel `embedding_bag_forward_kernel` that:
    - Takes embedding table, indices, offsets on device
    - Computes per-bag sum/mean/max directly on GPU
    - Avoids D2H/H2D round-trip
  - Register as OpId (e.g., `EmbeddingBagForward`)
  - CPU path remains as-is (no transfer needed)
  - Add Vulkan compute shader variant
- **Files to change:**
  - `include/tenzor/ops/op_id.hpp` - add EmbeddingBagForward, EmbeddingBagBackward
  - `src/backends/cuda/kernels/embedding.cu` - new kernel
  - `src/backends/cpu/kernels/embedding.cpp` - register CPU kernel
  - `src/nn/layers/embedding.cpp` - dispatch to new op
- **Test:** Compare GPU EmbeddingBag output with CPU reference

### 3.2 Vulkan Shader Parallel Compilation
- **File:** `src/backends/vulkan/CMakeLists.txt`
- **Issue:** 235+ shaders compiled serially
- **Fix:**
  - Each shader already has its own `add_custom_command()` which Ninja can parallelize
  - Verify Ninja actually parallelizes these (it should via dependency graph)
  - If not, consider `CMAKE_JOB_POOL_COMPILE` or a batch compilation script
  - Alternative: group shaders into a single `glslc` invocation with `--depfile`
- **Test:** Measure build time before/after

---

## Phase 4: Higher-Order Gradient Support (P2)

### 4.1 Extend backward_with_variables for Common Ops
- **Files:** `src/autograd/ops.cpp`, `include/tenzor/autograd/ops.hpp`
- **Current:** Only Add, Sub, Mul, Div, MatMul support create_graph=true (5/79)
- **Fix:** Implement proper `backward_with_variables()` for most-used ops:
  - **Batch 1 (elementwise):** Neg, Abs, Sqrt, Exp, Log, Pow, Clamp
  - **Batch 2 (activations):** ReLU, Sigmoid, Tanh, GELU, Softmax
  - **Batch 3 (reductions):** Sum, Mean
  - **Batch 4 (transforms):** Transpose, Reshape, Slice, Cat
  - **Batch 5 (advanced):** BatchNorm, LayerNorm, Conv2d (if feasible)
- **Pattern for each op:**
  ```cpp
  auto XXXBackward::backward_with_variables(
      const std::vector<Variable>& grad_outputs) -> std::vector<Variable> {
      // Use autograd::ops (Variable-level) instead of raw tensor ops
      // This preserves the computation graph for higher-order gradients
  }
  ```
- **Test:** Extend `tests/autograd/test_higher_order_gradients.cpp` for each new op
- **Priority:** Batch 1-2 first (most commonly needed for meta-learning, MAML)

---

## Phase 5: SyncBatchNorm (P2)

### 5.1 Implement SyncBatchNorm Layer
- **New files:**
  - `include/tenzor/nn/layers/sync_batchnorm.hpp`
  - `src/nn/layers/sync_batchnorm.cpp`
- **Design:**
  - Subclass of Module (not BatchNorm2d, to avoid virtual dispatch overhead)
  - Forward pass:
    1. Compute local mean and variance per GPU
    2. `all_reduce(mean)` and `all_reduce(var)` across process group
    3. Normalize using global statistics
    4. Apply affine transform (gamma, beta)
  - Backward pass: synchronized gradient reduction
  - Requires `ProcessGroup` reference (passed at construction or via context)
- **API:**
  ```cpp
  SyncBatchNorm(int64_t num_features, double eps=1e-5, double momentum=0.1,
                bool affine=true, bool track_running_stats=true,
                std::shared_ptr<ProcessGroup> process_group=nullptr);

  static auto convert_sync_batchnorm(std::shared_ptr<Module> module,
                                      std::shared_ptr<ProcessGroup> pg)
      -> std::shared_ptr<Module>;  // Converts all BN layers in-place
  ```
- **Python binding:** Add to `bindings.cpp` and `nn.py`
- **Test:** Multi-process test comparing SyncBN output with single-GPU BN on full batch

---

## Phase 6: Sparse NN Layers (P2)

### 6.1 SparseLinear
- **New files:**
  - `include/tenzor/nn/layers/sparse_linear.hpp`
  - `src/nn/layers/sparse_linear.cpp`
- **Design:**
  - Weight stored as sparse tensor (CSR format for efficient SpMM)
  - Forward: `output = sparse_matmul(input, weight.t()) + bias`
  - Backward: gradient w.r.t. input via sparse transpose matmul
  - Constructor accepts density ratio or pre-built sparse weight
- **Test:** Compare with dense Linear on same weights

### 6.2 SparseEmbedding
- **New files:**
  - `include/tenzor/nn/layers/sparse_embedding.hpp`
  - `src/nn/layers/sparse_embedding.cpp`
- **Design:**
  - Sparse gradient accumulation for embedding lookups
  - Only accessed rows get gradients (no full-table gradient)
  - Uses COO format for gradient accumulation
- **Test:** Verify sparse gradients match dense embedding gradients

---

## Phase 7: Extended Quantization & ONNX QDQ (P2)

### 7.1 Additional Quantized Layers
- **Files to add/modify:**
  - `include/tenzor/nn/quantization/quantized_layers.hpp` - add new classes
  - New kernel implementations in CPU/CUDA backends
- **New layers:**
  - `QuantizedLSTM` - INT8 LSTM cell with dequantized gates
  - `QuantizedGRU` - INT8 GRU cell
  - `QuantizedConv3d` - INT8 3D convolution
  - `QuantizedEmbedding` - INT8/INT4 embedding table (reduces memory 4-8x)
  - `QuantizedMultiheadAttention` - INT8 attention with FP32 softmax
- **Test:** Accuracy comparison with FP32 reference for each layer

### 7.2 ONNX QDQ Node Support
- **Files:**
  - `src/onnx/exporter.cpp` - add QuantizeLinear/DequantizeLinear export
  - `src/onnx/importer.cpp` - add QDQ node parsing
- **Export changes:**
  - When exporting quantized layers, emit QDQ pattern:
    `input -> QuantizeLinear -> DequantizeLinear -> Conv/Linear -> ...`
  - Include scale/zero_point as initializers
  - Support per-tensor and per-channel quantization parameters
- **Import changes:**
  - Detect QDQ pattern and reconstruct quantized layer
  - Parse scale, zero_point, axis attributes
  - Map to Tenzor's QuantizedLinear/QuantizedConv2d
- **Test:** Round-trip test: export quantized model -> import -> verify outputs match

---

## Phase 8: Build System Improvements (P3)

### 8.1 OneAPI CMake Modernization
- **File:** `src/backends/oneapi/CMakeLists.txt`
- **Fix:**
  - Replace manual `add_custom_command()` loops with `add_library(... OBJECT ...)`
  - Use `target_compile_options()` for SYCL flags instead of manual command construction
  - If icpx doesn't integrate with CMake's SYCL support, wrap in a function:
    ```cmake
    function(tenzor_add_sycl_library target)
        add_library(${target} SHARED ${ARGN})
        target_compile_options(${target} PRIVATE -fsycl ...)
        target_link_options(${target} PRIVATE -fsycl ...)
    endfunction()
    ```
  - Preserve functional equivalence with current build
- **Test:** Full OneAPI backend build + test suite passes

### 8.2 GPU Backend C++ Standard Documentation
- **Files:** CUDA, ROCm, OneAPI CMakeLists.txt files
- **Issue:** CUDA=C++20, ROCm=C++20, OneAPI=C++23 (inconsistent)
- **Fix:**
  - Document why CUDA/ROCm use C++20 (nvcc/hipcc don't fully support C++23)
  - Add comments in each CMakeLists.txt explaining the constraint
  - Verify OneAPI C++23 doesn't cause ABI issues with C++23 core library

---

## Phase 9: Code Hardening (P3)

### 9.1 CUDA Async Allocation RAII Wrapper
- **File:** `src/backends/cuda/kernels/matmul.cu:1397-1441`
- **Issue:** Not a bug currently, but exception-unsafe pattern
- **Fix:** Add simple RAII wrapper for CUDA async allocations:
  ```cpp
  struct CudaAsyncBuffer {
      void* ptr = nullptr;
      cudaStream_t stream;
      CudaAsyncBuffer(size_t bytes, cudaStream_t s) : stream(s) {
          TENZOR_CUDA_CHECK(cudaMallocAsync(&ptr, bytes, stream));
      }
      ~CudaAsyncBuffer() { if (ptr) cudaFreeAsync(ptr, stream); }
      CudaAsyncBuffer(const CudaAsyncBuffer&) = delete;
      CudaAsyncBuffer& operator=(const CudaAsyncBuffer&) = delete;
  };
  ```
  - Apply to matmul.cu Tensor Core path and any similar patterns
- **Test:** Existing matmul tests

### 9.2 CUDA Reduction Error Checking Improvement
- **File:** `src/backends/cuda/kernels/reduction.cu` (8 occurrences)
- **Issue:** `cudaGetLastError()` without sync may miss async errors in release
- **Fix:**
  - Replace `#ifndef NDEBUG` pattern with:
    ```cpp
    cudaError_t err = cudaPeekAtLastError();  // Non-blocking check
    if (err != cudaSuccess) {
        cudaStreamSynchronize(stream);  // Only sync on error
        throw std::runtime_error(...);
    }
    ```
  - This catches launch failures without the cost of full synchronization
- **Test:** Existing reduction tests

---

## Dependency Graph

```
Phase 1 (Critical bugs)     -- no dependencies, start immediately
Phase 2 (High-priority)     -- no dependencies, can parallel with Phase 1
Phase 3 (Performance)       -- Phase 1.1 for OneAPI, otherwise independent
Phase 4 (Higher-order grad) -- independent
Phase 5 (SyncBatchNorm)     -- requires working distributed backend (already done)
Phase 6 (Sparse NN)         -- requires working sparse ops (already done)
Phase 7 (Quantization)      -- Phase 7.2 (ONNX QDQ) independent; layers independent
Phase 8 (Build system)      -- independent, low risk
Phase 9 (Hardening)         -- independent, low risk
```

## Estimated Scope

| Phase | Items | Size | Risk |
|-------|-------|------|------|
| 1 | 3 critical fixes | Small | Low - well-understood bugs |
| 2 | 4 high-priority fixes | Small-Medium | Low |
| 3 | 2 perf improvements | Medium | Medium - new GPU kernels |
| 4 | ~20 backward ops | Medium-Large | Medium - correctness critical |
| 5 | SyncBatchNorm | Medium | Medium - distributed coordination |
| 6 | 2 sparse layers | Small-Medium | Low |
| 7 | 5 quant layers + ONNX QDQ | Medium-Large | Medium |
| 8 | 2 build fixes | Small | Low |
| 9 | 2 hardening items | Small | Low |

---

## Items Verified as NOT Needing Fixes

The following items from the review were verified and require no action:

- CUDA thread-local cudaMalloc (math.cu:128) - intentional per-thread persistence
- CUDA Float16 matmul allocation pattern - functional, just not RAII-wrapped (Phase 9.1 adds wrapper as improvement)
- CUDA reduction sync pattern - intentional async checking in release (Phase 9.2 improves it)
- int32_t->int cast in indexing.cu - already documented as the fix in MEMORY.md
- OMP threshold static init - C++11 guarantees thread-safe function-local static
- BFloat16 shuffle packing - correct AVX2 implementation verified
- Vulkan fill buffer truncation - already fixed with proper clamping
- Packed sequences - fully implemented (pack_padded_sequence, pad_packed_sequence)
- JIT compilation pipeline - fully implemented (tracing, optimization, execution)
- Distributed/NCCL - fully implemented (all_reduce, DDP, gradient compression)
- Sparse tensor core ops - fully implemented (COO, CSR, CSC, BSR formats)
