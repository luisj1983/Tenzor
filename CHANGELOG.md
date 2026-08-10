# Changelog

All notable changes to Tenzor will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-04-29

First public alpha release. The library is feature-complete for the listed scope but has had no public CI exposure on GPU hardware yet, and CPU benchmarks against PyTorch are competitive in some shapes and slower in others. Treat this release as experimental.

### Added

#### Tensor operations
- Multi-dimensional tensors with strided storage and broadcasting.
- Math: add, sub, mul, div, matmul, pow, sqrt, exp, log, plus the usual elementwise unary/binary set.
- Reductions: sum, mean, max, min, argmax, argmin, prod (with keepdim and axis).
- Shape ops: reshape, view, transpose, permute, squeeze, unsqueeze, expand, contiguous.
- Indexing: slice, gather, scatter, masked_select, masked_fill, index_select, advanced (fancy) indexing.
- Dtypes: Float32, Float64, Float16, BFloat16, Int8/16/32/64, UInt8/16/32/64, Bool, Complex64/128.

#### Automatic differentiation
- Reverse-mode autodiff with explicit computation graph.
- In-place op support, gradient accumulation, gradient checkpointing.
- Custom `autograd::Function` extension point.
- Higher-order gradients for the activation set and core math ops.

#### Neural network layers
- Linear, Conv1d/2d/3d (grouped + dilated).
- BatchNorm1d/2d/3d, LayerNorm, GroupNorm, RMSNorm, InstanceNorm.
- MaxPool1d/2d/3d, AvgPool1d/2d/3d, AdaptiveAvgPool, AdaptiveMaxPool.
- RNN, LSTM, GRU (bidirectional).
- MultiheadAttention, ScaledDotProductAttention.
- Activations: ReLU, LeakyReLU, GELU, SiLU/Swish, Sigmoid, Tanh, Softmax, ELU, SELU, GatedLinearUnit.
- Dropout, Dropout2d, AlphaDropout.
- Embedding, EmbeddingBag.
- Containers: Sequential, ModuleList, ModuleDict; Flatten / Unflatten utilities.

#### Optimizers
- SGD (with momentum and Nesterov).
- Adam, AdamW, AdamAtan2.
- RMSprop, Adagrad.

#### Learning rate schedulers
- StepLR, MultiStepLR, ExponentialLR.
- CosineAnnealingLR, CosineAnnealingWarmRestarts.
- OneCycleLR, ReduceLROnPlateau.
- Linear and polynomial warmup.

#### Loss functions
- MSELoss, CrossEntropyLoss, NLLLoss, BCELoss, BCEWithLogitsLoss.
- L1Loss, SmoothL1Loss, HuberLoss.
- CTC Loss, Focal Loss, Label-smoothing CE.

#### Backends
- **CPU** with SSE4.2 / AVX2 / AVX-512 paths, OpenMP parallelism, MKL/oneDNN where available.
- **CUDA 12.0+** via cuBLAS / cuDNN / cuSOLVER / cuSPARSE plus custom kernels.
- **ROCm 5.0+** via hipBLAS / MIOpen / rocSOLVER / rocSPARSE.
- **OneAPI 2023.0+** via oneMKL / oneDNN.
- **Vulkan 1.2+** with SPIR-V compute shaders (cross-vendor).
- All five backends register the same 317 operations (op-count parity verified by `bin/op_coverage_report`).

#### Reference model implementations (`src/models/`)
- Vision: ResNet, VGG, MobileNet (V2/V3), EfficientNet, ConvNeXt, ViT, Swin Transformer, YOLO, Faster R-CNN, Mask R-CNN, U-Net, DeepLabV3+, AlexNet, GoogLeNet.
- Language: BERT, RoBERTa, ALBERT, ELECTRA, GPT-2, T5.
- Specialized: Hierarchical Reasoning Model (HRM).
- Note: model definitions only — no pretrained weights are distributed in this release.

#### Other
- ONNX import and export (subset of operators; coverage tracked under `tests/`).
- Mixed-precision training (FP16 / BF16) and INT8 quantization (PTQ + QAT).
- JIT tracing and scripting with a kernel-fusion pass.
- Dataset / DataLoader with multi-worker prefetch and basic image transforms.
- Distributed training primitives (data-parallel and model-parallel scaffolding).
- Python bindings via pybind11; NumPy interop; Python 3.9–3.13.

### Known limitations

- **No public CI proof for GPU backends.** GitHub Actions currently runs CPU smoke tests only.
- **MPS backend** (Apple Metal) is partial and not at parity with the four primary GPU backends.
- **CPU performance** is below PyTorch on the published benchmark suite (see `benchmarks/baselines/README.md` and `scripts/ci_benchmark.sh` for how to regenerate current numbers). One Conv2D shape measures 0.07× and is almost certainly a measurement artifact pending re-run.
- **Single maintainer.** Reviews and contributions welcome.

---

## [Unreleased]

### Added

- Op count has grown to 702 registered operations across all five backends (up from
  317 at v0.1.0) — includes sparse ops (SpMM/SpGEMM/Trsv), cuDNN RNN training paths,
  fused/flash/flex attention variants, and additional distribution sampling ops.
  Re-verify with `bin/op_coverage_report --json` after configuring the backend set
  you want to inspect.
- **MPS backend** (Apple Metal) gained new coverage: FlashAttention GQA tests now
  run on MPS, and `DataParallel` was extended for multi-backend (including MPS)
  compatibility.
- **CachingAllocator** for efficient GPU memory management, reducing allocation
  overhead across backend GPU paths.

### Fixed

- **CUDA LSTM / GRU forward** no longer throws `cuBLAS INVALID_VALUE`. Root cause was inverted `OP_T` / `OP_N` flags in `cublas_gemm_ex` for the no-transpose default case. Only `lstm_forward_cuda` / `gru_forward_cuda` / `bilstm_forward_cuda` used the broken wrapper; Linear / matmul went through a different known-good path.
- **CUDA GRU forward kernel-launch bug** (the follow-on to the cuBLAS fix above) resolved. The per-timestep `std::swap` on `Tensor` handles rebound the storage handle and the next iteration's `h_out.data_ptr()` tripped a CUDA driver "invalid argument" on `cudaMemcpyAsync` at `t=1` for the BenchShape inputs (batch=32, hidden=256, seq=128). Root-cause fix in `rnn_sequence.cu`: an explicit two-buffer ping-pong (`h_buf[t & 1]` / `h_buf[(t + 1) & 1]`) gives the driver stable, never-rebinding device pointers across the whole loop; a duplicate transpose in the `gru.cpp` fast path (feeding the kernel batch-major input it misread as time-major) and an explicit `gates_ih_t.contiguous()` guard were also corrected. The regression test `GRU_BenchShape` is re-enabled and passes. LSTM forward was unaffected.
- **CUDA cuDNN SDPA path** now supports FP32 (and BF16) end-to-end on Ampere / Hopper, not just FP16. The `create_sdpa_graph` builder is dtype-parameterized; the dispatch path picks `HALF` / `BFLOAT16` / `FLOAT` from `Q.dtype()`. Added a per-process capability cache so combos cuDNN reports as unsupported on a given device only pay the build/check cost once.
- **Memory offloading / ZeRO subsystem** backend-parity bugs — cross-backend equality gaps in the distributed and offload test subsystems resolved.
- **SparseAdd** multi-backend test coverage gaps closed.
- **Vulkan native STFT / ISTFT** re-enabled. The forward pass was producing wrong-valued spectra because a Complex64 output buffer was sized with `output.numel() * 4` (half the required byte count — `Complex64` is 8 bytes/element); on permissive drivers this silently corrupted the spectrum and cascaded into the STFT round trip. Fixed in `vulkan_ops_fft.cpp` by sizing with `complex_elem_size`. The native `dispatchSTFT` / `dispatchISTFT` shaders (`stft_frame_window.comp`, `istft_overlap_add.comp`, `istft_normalize.comp`) are now registered and active on Vulkan with no CPU fallback; cross-backend parity confirmed (`FFTParity.STFT` and `FFTParity.ISTFT_Roundtrip` pass on Vulkan vs CPU).
- **NumPy interop for complex dtypes** works end-to-end. `dtype_to_numpy_format` now maps `Complex64` / `Complex128` to `py::format_descriptor<std::complex<float>>::format()` / `std::complex<double>` (the NumPy `Zf` / `Zd` buffer formats), so `from_numpy` / `Tensor.numpy()` round-trip complex arrays. Previously `.numpy()` on a complex tensor raised a buffer-protocol `'Zf'` error. Verified for `from_numpy`→`.numpy()`, op-produced complex (`fft.rfft` output), and CUDA→CPU→`.numpy()`.
- **Vulkan integer-valued sparse roundtrip** works on Vulkan. `SparseTensor::from_dense` / `to_dense` were rewritten to dtype-agnostic `nonzero` / `index_select` / `scatter_add` primitives, replacing the previous host round-trip and bypassing the float-only raw extract/scatter shaders — so integer-valued (`Int32`, etc.) sparse tensors round-trip on every backend. Regression coverage added (`SparseGPUTest.ToDenseRoundtrip_Int32`, all 4 GPU backends). The raw `OpId::DenseToSparse` / `SparseToDense` kernels still throw for integer dtypes, but those are no longer on the user-facing path (autograd dispatches them only for differentiable / float dtypes).
- **`C10dRendezvous::join()`** is a full implementation, no longer a stub. It now builds on the shared gloo `RendezvousStore`: each worker claims a unique slot via an atomic counter, barriers to `min_workers` (with a grace window up to `max_workers`) or throws on timeout, the coordinator (slot 0) freezes `world_size`, and `rank` is assigned as the worker's slot. `leave()` writes a best-effort departure marker keyed by the joined round. Previously `join()` only advanced `round_` and returned `rank=-1` / `world_size=0`, making multi-worker elastic coordination impossible. Verified by `test_rendezvous` (5/5, incl. `SingleWorkerJoinAssignsRankZero` and `LeaveResetsState`).
- **ONNX importer external-data loading** no longer crashes. The two external-data round-trip tests (`ImporterReadsExternalDataRoundTrip`, `ImporterClearsExternalDataDirOnBytesEntry`) pass, fixed by the importer's host-staging rewrite — `Slice` / `Pad` / `Resize` / `Reduce` control constants are now read from host proto bytes (`get_host_input` / `initializers_ptr_`) and never uploaded to the GPU, eliminating the SYCL crash. `test_onnx_import` 29/29.

### Known issues

- **CUDA LSTM / GRU training is ~50× slower than PyTorch.** The autograd path bypasses the fused kernel (no backward kernel exists yet) and steps through the cell per timestep, building thousands of Variable nodes. Tracked for v0.2 — cuDNN RNN integration. Until then, use eval-mode forward, or train on CPU. A one-shot `WARNING` is printed the first time the slow path runs on CUDA so users notice the perf cliff.
- **FP32 cuDNN SDPA on Blackwell (sm ≥ 100)** is gated off — cuDNN frontend reports `check_support` OK but `execute()` triggers an illegal memory access (verified on RTX 5070 / sm_120). Blackwell falls through to the manual BMM path for FP32 attention, preserving the v0.1.0 behavior. FP16 / BF16 are unaffected.

### Planned

- Public GPU CI on self-hosted runners.
- Re-run benchmarks with proper warmup / median-of-N; publish GPU numbers.
- Pretrained weight hub for the reference models.
- Distributed training improvements (gradient bucketing, NCCL/RCCL collectives).
- cuDNN RNN forward+backward integration (closes the LSTM/GRU training perf gap).

---

## Version history

| Version | Date       | Highlights                          |
|---------|------------|-------------------------------------|
| 0.1.0   | 2026-04-29 | First public alpha release.         |

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT — see [LICENSE](LICENSE).
