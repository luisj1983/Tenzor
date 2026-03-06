# Tenzor Comprehensive Fix & Feature Plan

Based on a thorough review of 520k LOC across 1,220 source files, verified by targeted
audits to eliminate false positives. Items are ordered by priority within each phase.

---

## Phase 1: Confirmed Bug Fixes (P0 - Critical)

### 1.1 Vulkan: Missing VkResult Check on vkCreatePipelineCache
- **File:** `src/backends/vulkan/vulkan_backend.cpp:528`
- **Issue:** `vkCreatePipelineCache()` return value is not checked. If it fails (e.g.,
  corrupt cache data from disk), `ctx.pipelineCache` is invalid, causing crashes on any
  subsequent pipeline creation.
- **Fix:** Wrap with `vulkan::checkVk()`. On failure, log warning and proceed with
  `VK_NULL_HANDLE` pipeline cache (pipelines work without cache, just slower).
- **Test:** Add unit test that loads a corrupt pipeline cache file and verifies graceful
  fallback.

### 1.2 Vulkan: Missing VkResult Check on vkResetCommandPool
- **File:** `src/backends/vulkan/vulkan_backend.cpp:908`
- **Issue:** `vkResetCommandPool()` in the descriptor pool exhaustion recovery path is
  unchecked. If reset fails, the recovery silently continues with a dirty command pool.
- **Fix:** Wrap with `vulkan::checkVk()`. On failure, throw with context about the
  recovery path.
- **Test:** Existing descriptor exhaustion tests should cover this path.

### 1.3 Non-Contiguous GPU fill_() Round-Trips Through CPU
- **File:** `src/core/tensor.cpp:774-826`
- **Issue:** For non-contiguous GPU tensors, `fill_()` copies the *entire storage* to CPU,
  modifies target elements with stride iteration, then copies back. Filling 10 elements
  of a 1GB tensor copies 2GB.
- **Fix:** Implement per-backend strided fill kernels:
  - **CUDA:** Trivial kernel — one thread per element, compute strided offset, write value.
    Register as `OpId::StridedFill` or handle inside existing `Fill` kernel with stride args.
  - **Vulkan:** Add `strided_fill.comp` shader with push constants for shape/strides/value.
  - **OneAPI:** SYCL parallel_for with strided offset computation.
  - Update `tensor.cpp:fill_()` to dispatch `OpId::StridedFill` when non-contiguous on GPU
    instead of the CPU round-trip path.
- **Test:** `fill_(value)` on a non-contiguous GPU slice; verify correctness and that no
  D2H/H2D copies occur (check via profiling or mock allocator).

### 1.4 Int8 Multiply Uses Truncation Instead of Saturation
- **File:** `src/backends/cpu/kernels/int_simd.hpp:146-178`
- **Issue:** Int8 multiply truncates results via `AND 0xFF` mask and `static_cast<int8_t>`.
  `127 * 2` produces `-2` instead of `127` (saturated). This affects quantized inference
  correctness.
- **Fix:**
  - SIMD path: After `_mm256_mullo_epi16`, clamp with `_mm256_max_epi16(result, min_val)`
    and `_mm256_min_epi16(result, max_val)` where min=-128, max=127, then pack.
  - Scalar path: Replace `static_cast<int8_t>(a*b)` with
    `static_cast<int8_t>(std::clamp(a*b, -128, 127))`.
- **Test:** Add test: `mul_i8({127, -128, 64}, {2, 2, 3})` expects `{127, -128, 127}`.

---

## Phase 2: Core API Improvements (P1 - High)

### 2.1 Add Tensor::to(Device, DType) Combo Method
- **File:** `include/tenzor/core/tensor.hpp`, `src/core/tensor.cpp`
- **Issue:** Users must chain `.to(device).to(dtype)` causing two allocations and two
  transfers instead of one.
- **Fix:** Add overload:
  ```cpp
  auto to(Device device, DType dtype) const -> Tensor;
  ```
  Implementation: If device differs, transfer first then cast on target device (avoids
  transferring in wrong dtype). If same device, just cast. Check for no-op (same device
  AND same dtype).
- **Also add:** `auto to(Device device, DType dtype, MemoryFormat fmt) const -> Tensor;`
- **Python binding:** Add `tensor.to(device, dtype)` overload in `bindings.cpp`.
- **Test:** Verify single-allocation path; verify no-op returns self.

### 2.2 Validate Shape/Stride Mutations in Internal API
- **File:** `include/tenzor/core/tensor.hpp:1049-1063`
- **Issue:** `mutable_shape()` and `mutable_strides()` allow arbitrary changes without
  validating that the new shape fits within storage bounds.
- **Fix:** Replace raw accessors with:
  ```cpp
  void set_sizes_and_strides(std::span<const int64_t> sizes,
                             std::span<const int64_t> strides);
  ```
  This method validates: (a) all dimensions non-negative, (b) max reachable offset
  `<= storage_size / dtype_size`, (c) no integer overflow in offset computation.
  Keep `mutable_shape()`/`mutable_strides()` but add `TENZOR_DEBUG` assertions.
- **Test:** Verify that invalid shapes throw; valid shapes succeed.

### 2.3 PositionalEncoding: Compute on Target Device
- **File:** `src/nn/layers/transformer.cpp:23-151`
- **Issue:** PE is computed on CPU at init and sliced/transferred on every forward pass.
- **Fix:**
  - Cache the PE tensor on the input's device after first forward call.
  - Use a `device_` member tracking where PE is cached.
  - On forward: if `pe_.device() != input.device()`, transfer and cache.
  - On subsequent calls with same device: zero-copy slice from cached tensor.
- **Test:** Verify PE on GPU after first forward; verify no CPU→GPU transfer on second call.

---

## Phase 3: Missing NN Layers (P1 - High)

### 3.1 BatchNorm3d
- **Files:** `include/tenzor/nn/layers/batchnorm.hpp`, `src/nn/layers/batchnorm.cpp`
- **Design:** Same as BatchNorm2d but for 5D input (N, C, D, H, W). Reshape to
  (N, C, D*H*W), delegate to existing BatchNorm computation, reshape back.
- **Backend support:** CPU dispatch through existing BatchNorm kernel with flattened spatial
  dims. CUDA: cuDNN supports 5D BatchNorm natively.
- **Test:** Forward/backward with 5D input; parity with PyTorch `nn.BatchNorm3d`.

### 3.2 InstanceNorm3d
- **Files:** `include/tenzor/nn/layers/normalization.hpp`, `src/nn/layers/normalization.cpp`
- **Design:** Extend existing InstanceNorm pattern. Compute mean/var over (D, H, W) dims
  per (N, C) pair.
- **Test:** Forward/backward with 5D input; parity with PyTorch `nn.InstanceNorm3d`.

### 3.3 SyncBatchNorm
- **Files:** New `include/tenzor/nn/layers/sync_batchnorm.hpp`,
  `src/nn/layers/sync_batchnorm.cpp`
- **Design:** Extends BatchNorm2d. During training:
  1. Compute local mean and var per GPU.
  2. All-reduce mean and var across process group.
  3. Normalize using global statistics.
- **Dependencies:** Requires `distributed::ProcessGroup` for all-reduce.
- **Test:** Multi-GPU test with 2+ processes verifying synchronized statistics.

---

## Phase 4: Missing Operations (P1 - High)

### 4.1 einsum
- **Files:** New `include/tenzor/ops/einsum.hpp`, `src/ops/einsum.cpp`
- **Design:**
  1. Parse Einstein notation string (e.g., `"ij,jk->ik"`).
  2. Identify contraction, batch, and free dimensions.
  3. Decompose into sequence of: transpose, reshape, matmul/bmm, reduce.
  4. Optimize common patterns: matmul (`ij,jk->ik`), batch matmul (`bij,bjk->bik`),
     trace (`ii->`), outer product (`i,j->ij`), dot (`i,i->`).
- **OpId:** Add `OpId::Einsum` for potential backend-specific fast paths.
- **Python:** Expose as `tz.einsum("ij,jk->ik", a, b)`.
- **Test:** All common patterns; verify gradient flow; parity with `numpy.einsum`.

### 4.2 median and mode
- **Files:** `include/tenzor/ops/reduction.hpp`, `src/ops/reduction.cpp`,
  CPU/CUDA kernel files.
- **Design:**
  - `median(tensor, dim)` → partial sort (nth_element on CPU, CUB radix select on CUDA).
  - `mode(tensor, dim)` → sort + adjacent count (CPU), sort + reduce-by-key (CUDA).
  - Both return `(values, indices)` tuple like PyTorch.
- **OpId:** Add `OpId::Median`, `OpId::Mode`.
- **Test:** Various shapes and dtypes; edge cases (even-length for median, ties for mode).

### 4.3 linalg.matrix_power
- **File:** `include/tenzor/ops/linalg.hpp`, `src/ops/linalg.cpp`
- **Design:** Binary exponentiation: `A^n` via repeated squaring. Handle n=0 (identity),
  n<0 (invert then exponentiate). Delegate to existing `matmul` and `linalg_inv`.
- **Test:** A^0=I, A^1=A, A^(-1)=inv(A), A^4 = (A^2)^2.

### 4.4 Trilinear Interpolation
- **File:** `src/ops/vision.cpp`, backend kernel files.
- **Design:** Extend existing `interpolate` with `mode="trilinear"` for 5D inputs (N,C,D,H,W).
  Follow the same pattern as bilinear but in 3 spatial dimensions (8-point interpolation).
- **OpId:** Reuse `OpId::Interpolate` with `AttrKey::Mode = "trilinear"`.
- **CPU:** Nested loop with 8-point weight computation.
- **CUDA:** Grid-stride kernel, one thread per output element.
- **Vulkan:** `interpolate_trilinear.comp` shader.
- **Test:** Upsample and downsample 5D tensors; parity with PyTorch
  `F.interpolate(mode='trilinear')`.

---

## Phase 5: CPU Performance Optimizations (P2 - Medium)

### 5.1 Fused Conv+Bias+ReLU CPU Kernel
- **File:** `src/backends/cpu/kernels/fused_ops.cpp` (currently only forward declarations)
- **Design:** After im2col + GEMM, fuse bias addition and ReLU into a single pass over the
  output buffer. Avoids 2 extra full-tensor passes.
  ```cpp
  // Instead of: gemm(output); add_bias(output); relu(output);
  // Do:         gemm(output); for each elem: output[i] = max(0, output[i] + bias[c]);
  ```
- **SIMD:** Vectorize with `_mm256_max_ps(_mm256_add_ps(out, bias), zero)`.
- **Registration:** `OpId::FusedConv2dReLU` and `OpId::FusedConvBiasReLU`.
- **Test:** Numerical parity with separate conv → bias → relu.

### 5.2 Winograd F(4x4, 3x3) Larger Tile
- **File:** `src/backends/cpu/kernels/winograd.hpp`
- **Current:** Only F(2x2, 3x3) implemented (output tile 2x2).
- **Add:** F(4x4, 3x3) for 6x6 transform tiles. Reduces multiplies further for large
  spatial dimensions. Published transform matrices available in literature.
- **Gate:** Use F(4x4) for spatial dims >= 8, F(2x2) for smaller.
- **Test:** Numerical parity with im2col path; benchmark improvement.

### 5.3 Depthwise Conv GEMM Optimization
- **File:** `src/backends/cpu/kernels/conv2d.cpp`
- **Issue:** Depthwise conv currently uses naive loop.
- **Fix:** For depthwise (groups == in_channels), use per-channel GEMM with smaller
  matrices. Alternatively, use direct SIMD convolution for common 3x3 depthwise.
- **Test:** MobileNet-style depthwise conv benchmark.

---

## Phase 6: ONNX Completeness (P2 - Medium)

### 6.1 ONNX Importer: Expand Op Coverage (27 → 60+ ops)
- **File:** `src/onnx/importer.cpp`
- **Priority ops to add (by frequency in real models):**
  1. **LSTM, GRU** — Map to existing RNN layers
  2. **Embedding/Gather** — Map to `OpId::Embedding` / `OpId::Gather`
  3. **ReduceSum, ReduceMean, ReduceMax** — Map to existing reduction ops
  4. **Unsqueeze, Squeeze** — Map to existing shape ops
  5. **Slice, Pad** — Map to existing indexing ops
  6. **Clip** — Map to `OpId::Clamp`
  7. **Cast** — Map to `Tensor::to(DType)`
  8. **Dropout** — Map to `OpId::Dropout` (identity in eval mode)
  9. **Resize** — Map to `OpId::Interpolate`
  10. **LayerNormalization, GroupNormalization** — Map to existing norm layers
- **Test:** Round-trip test: export model → import → compare forward pass outputs.

### 6.2 ONNX QDQ Node Support
- **Files:** `src/onnx/exporter.cpp`, `src/onnx/importer.cpp`
- **Design:** Add `QuantizeLinear` and `DequantizeLinear` ONNX ops.
  - Export: Convert Tenzor quantized layers to QDQ pattern.
  - Import: Map QDQ nodes to Tenzor's quantized ops.
- **Test:** Export quantized model; verify loadable by ONNX Runtime.

---

## Phase 7: Sparse Tensor Operations (P2 - Medium)

### 7.1 Expand Sparse Op Coverage
- **Files:** `src/sparse/sparse_ops.cpp`, `include/tenzor/sparse/sparse_ops.hpp`
- **Add operations:**
  1. **sparse_mm (sparse @ sparse → sparse)** — Merge-based COO multiplication
  2. **sparse_sum / sparse_mean** — Reduction along dimensions
  3. **sparse_softmax** — Over non-zero entries per row
  4. **sparse_to_dense_backward** — Gradient for dense→sparse conversion
  5. **sparse_index_select** — Select rows/columns from sparse matrix
- **Test:** Each op with COO and CSR inputs; verify against dense equivalent.

### 7.2 Sparse Float16/BFloat16 Support
- **Files:** `src/sparse/sparse_ops.cpp`, `src/backends/cuda/kernels/sparse.cu`
- **Issue:** Only Float32/Float64 supported in cuSPARSE path.
- **Fix:** Add FP16→FP32 conversion wrapper for sparse ops (cuSPARSE doesn't natively
  support FP16 for most ops).
- **Test:** spmm with FP16 inputs; verify numerical parity with FP32.

---

## Phase 8: Distributed Training Completeness (P2 - Medium)

### 8.1 DistributedSampler
- **Files:** New `include/tenzor/data/distributed_sampler.hpp`,
  `src/data/distributed_sampler.cpp`
- **Design:**
  - Partition dataset indices across `world_size` ranks.
  - Each rank gets `ceil(len(dataset) / world_size)` samples.
  - Shuffle with rank-specific seed per epoch.
  - Pad last rank if uneven split (configurable: pad or drop).
- **Integration:** DataLoader accepts optional Sampler; DistributedSampler is one impl.
- **Test:** 2-rank test verifying non-overlapping indices covering full dataset.

### 8.2 Gradient Compression
- **File:** `include/tenzor/distributed/gradient_compression.hpp` (header exists)
- **Implement:**
  1. **Top-K sparsification** — Only communicate largest K% of gradients.
  2. **Error feedback** — Accumulate residuals for next iteration.
  3. **Quantized all-reduce** — Compress gradients to INT8 before communication.
- **Test:** Convergence test on small model with compression vs baseline.

---

## Phase 9: JIT Enhancements (P3 - Low)

### 9.1 Control Flow Support (If/Loop)
- **File:** `src/jit/compiler.cpp`
- **Design:**
  - `If` node: Two subgraphs (then/else), condition tensor, merge outputs.
  - `Loop` node: Body subgraph, trip count, loop-carried dependencies.
  - Optimization: constant-fold conditions, unroll small loops.
- **Tracing limitation:** Trace-based JIT captures one execution path. For control flow,
  need scripting mode or symbolic tracing.
- **Test:** Model with conditional branch; model with fixed-iteration loop.

### 9.2 Cross-Kernel Fusion
- **File:** `src/jit/compiler.cpp`
- **Current:** Conv+BN, Conv+ReLU, Linear+ReLU, MatMul+Add fusions exist.
- **Add:**
  1. **Conv+BN+ReLU** triple fusion (already exists per audit — verify)
  2. **LayerNorm+Dropout** fusion
  3. **Attention fusion** (Q*K^T/sqrt(d) + mask + softmax + V multiply)
- **Test:** Benchmark fused vs unfused; verify numerical parity.

---

## Phase 10: Additional Missing Features (P3 - Low)

### 10.1 Autograd: Checkpoint Determinism Warning
- **File:** `src/autograd/checkpoint.cpp`
- **Issue:** Checkpoint assumes deterministic functions but doesn't warn or validate.
- **Fix:** Add optional `verify=True` parameter that recomputes forward twice during the
  first backward pass and compares outputs. If mismatch detected, emit warning.
- **Default:** `verify=False` (no overhead in production).

### 10.2 View Aliasing: Overlap Detection
- **File:** `src/core/tensor.cpp:724-766`
- **Current:** Detects same-storage aliasing and clones. But overlapping views from
  different slices of same storage are also detected (same storage pointer check).
- **Enhancement:** For extra safety, could add range-overlap check:
  ```cpp
  bool ranges_overlap(offset_a, size_a, offset_b, size_b);
  ```
  Only clone when ranges actually overlap, not just when storage matches.
  This is an optimization (avoids unnecessary clones), not a correctness fix.

### 10.3 Thread Safety Documentation
- **File:** `include/tenzor/core/tensor.hpp`
- **Add:** Prominent section documenting:
  - Read operations are thread-safe.
  - Concurrent read + write requires external synchronization.
  - `mutable_shape()`/`mutable_strides()` are NOT thread-safe.
  - Autograd forward pass is NOT thread-safe on shared Variables.
  - Autograd backward pass is thread-safe with per-Variable `make_thread_safe()`.

---

## Dependency Graph

```
Phase 1 (Bug Fixes)     ─── no dependencies, start immediately
Phase 2 (Core API)      ─── no dependencies, can parallelize with Phase 1
Phase 3 (NN Layers)     ─── 3.3 SyncBatchNorm depends on distributed (Phase 8)
Phase 4 (Missing Ops)   ─── 4.1 einsum depends on existing matmul/transpose
Phase 5 (CPU Perf)      ─── no dependencies
Phase 6 (ONNX)          ─── 6.1 depends on ops existing (Phases 3-4)
Phase 7 (Sparse)        ─── no dependencies
Phase 8 (Distributed)   ─── no dependencies
Phase 9 (JIT)           ─── depends on fused op kernels (Phase 5)
Phase 10 (Misc)         ─── no dependencies
```

## Recommended Execution Order

1. **Phase 1** — Bug fixes first (small scope, high impact)
2. **Phase 2** — Core API improvements (enables cleaner code in later phases)
3. **Phases 3 + 4 + 5 in parallel** — Independent workstreams (NN layers, ops, CPU perf)
4. **Phase 6** — ONNX after new ops exist
5. **Phases 7 + 8 in parallel** — Sparse and distributed (independent)
6. **Phase 9** — JIT after fused kernels exist
7. **Phase 10** — Polish items last

## Estimated Scope

| Phase | Files Modified | Files Created | Approx LOC |
|-------|---------------|---------------|------------|
| 1     | 4             | 0             | ~200       |
| 2     | 5             | 0             | ~300       |
| 3     | 4             | 2             | ~800       |
| 4     | 8             | 4             | ~2,500     |
| 5     | 3             | 1             | ~600       |
| 6     | 2             | 0             | ~1,500     |
| 7     | 3             | 0             | ~600       |
| 8     | 2             | 2             | ~500       |
| 9     | 1             | 0             | ~800       |
| 10    | 3             | 0             | ~200       |
| **Total** | **~35**   | **~9**        | **~8,000** |
