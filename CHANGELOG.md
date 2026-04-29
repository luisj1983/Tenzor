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
- **Vulkan STFT / ISTFT** dispatch to a CPU fallback in this release; the native compute path is built but a forward-pass shape-value bug is open. Tracked in [`audit/README.md`](audit/README.md) Phase 4.3.
- **MPS backend** (Apple Metal) is partial and not at parity with the four primary GPU backends.
- **CPU performance** is below PyTorch on the published benchmark suite (see [`reports/combined_benchmark.md`](reports/combined_benchmark.md)). One Conv2D shape measures 0.07× and is almost certainly a measurement artifact pending re-run.
- **Single maintainer.** Reviews and contributions welcome.

---

## [Unreleased]

### Planned

- Public GPU CI on self-hosted runners.
- Re-run benchmarks with proper warmup / median-of-N; publish GPU numbers.
- Resolve the Vulkan STFT / ISTFT regression and re-enable the native pipeline.
- Pretrained weight hub for the reference models.
- Distributed training improvements (gradient bucketing, NCCL/RCCL collectives).

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
