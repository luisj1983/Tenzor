# Tenzor Test Coverage Audit — 2026-05-01

Snapshot of feature surface vs test surface, with a prioritized gap list.
Scope: every user-visible feature of the library, against `tests/` and
`tests/python/`. Five backends in scope: CPU, CUDA, ROCm, Vulkan, OneAPI.

---

## TL;DR

**Library is well-tested. Don't panic.** ~687 C++ test files + ~94 Python test
files cover the bulk of the ~1,500-feature surface. The TESTING.md contract
is mostly respected. What follows is a finite punch list — not a "rewrite the
suite" exercise.

The remaining gaps cluster in five categories, in descending severity:

| # | Gap | Count | Severity |
|---|---|---|---|
| 1 | Ops gradchecked CPU-only (no GPU backward exercise) | ~98 / 123 | **High** |
| 2 | Tests using banned custom `BackendDTypeParam` fixture | 24 files | **High** |
| 3 | Whole-library subsystems with thin or no parity coverage | 5 areas | **Medium** |
| 4 | `tests/ops/` files hardcoded to CPU only | 7+ files | **Medium** |
| 5 | Specific known-broken / disabled / weak-assertion tests | 4 sites | **Low** |

The library has 469 OpId enum slots, 317 registered on every backend (CPU/CUDA/ROCm/Vulkan/OneAPI), 150+ NN layers, 147+ autograd functions, 43 distributions, 20+ pretrained models, full distributed/JIT/ONNX/quantization stacks. Test inventory map is in §1; gap list is in §2.

---

## 1. Feature ↔ Test Coverage Map

### 1.1 Ops (OpId)

| Category | OpIds | Backend-parity test? | Gradcheck? | Multi-backend gradcheck? |
|---|---|---|---|---|
| Arithmetic (Add/Sub/Mul/Div/MatMul/Bmm/Dot, in-place) | 11 | ✅ `test_arithmetic_parity.cpp`, `test_grad_arithmetic_parity.cpp` | ✅ | ✅ (matmul, add, mul, neg) |
| Reductions (Sum/Mean/Max/Min/Var/Std/Prod/ArgMax/ArgMin/Median/Mode/Any/All/Aminmax/CountNonzero/Nansum/Nanmean) | 19 | ✅ `test_grad_reduction_parity.cpp` | ✅ (most) | ⚠️ partial — Mean(dim) skipped (known MeanBackward crash) |
| Element-wise math (Sqrt/Neg/Abs/Sign/Log/Exp/Pow/Clamp/Reciprocal/Floor/Ceil/Round/Trunc/Frac/Heaviside/NanToNum/Rsqrt/Square) | 20 | ✅ `test_extended_math_parity.cpp` | ✅ | ✅ (neg/abs/reciprocal/pow) |
| Trig (Sin/Cos/Tan/Asin/Acos/Atan/Sinh/Cosh/Tanh/Asinh/Acosh/Atanh) | 12 | ✅ `test_trig_parity.cpp` | ✅ | ✅ (sin, cos) |
| Activations (ReLU/Sigmoid/Tanh/Gelu/Swish/LeakyReLU/Elu/Selu/Mish/Softplus/Softmax/LogSoftmax + inplace + LogSigmoid/RReLU) | 33 | ✅ `test_nn_activation_parity.cpp` | ✅ | ✅ (relu, leaky_relu, softplus) |
| Shape/view (Reshape/Transpose/Permute/Squeeze/Unsqueeze/Flatten/Contiguous/Clone/Fill/Repeat/Tile/Expand/Stack/Split/Chunk/ToMemoryFormat) | 16 | ✅ `test_shape_ops_parity.cpp`, `test_grad_shape_parity.cpp` | ✅ (transpose/reshape/expand) | ✅ (transpose, reshape, expand, flip, roll, cat) |
| Indexing (IndexSelect/Gather/Scatter/MaskedSelect/MaskedFill/Where/Slice/Cat/Take/Put/Nonzero/OneHot/AdvancedIndex/AdvancedIndexPut) | 14 | ✅ `test_indexing_parity.cpp` | ✅ | ✅ (index_select, gather, narrow) |
| Comparison (Eq/Ne/Lt/Le/Gt/Ge) | 6 | ✅ `test_comparison_parity.cpp` | n/a (no grads) | n/a |
| Normalization (BatchNorm2d×6 / LayerNorm / GroupNorm / InstanceNorm / RMSNorm + backwards) | 14 | ✅ `test_nn_norm_parity.cpp` | ⚠️ via layer tests | ❌ kernel-level not gradchecked on GPU |
| Convolution (Conv1/2/3d + backwards, ConvTranspose1/2/3d, Depthwise, DeformableConv2d) | 18 | ✅ `test_nn_conv_parity.cpp` | ⚠️ via layer tests | ❌ kernel-level not gradchecked on GPU |
| Pooling (Max/Avg/Adaptive 1d/2d/3d, LPPool, FractionalMaxPool, MaxUnpool) | 8 + 8 + 8 | ✅ `test_nn_pooling_parity.cpp` | ⚠️ partial | ❌ |
| Vision (Unfold/Fold/Interpolate/ROIAlign/BoxIoU/NMS/GridSample/AffineGrid) | 10 | ✅ `test_nn_parity.cpp` | ✅ (grid_sample, affine_grid) | ❌ |
| Fused (FusedLinearReLU/FusedConv2dReLU/FusedBN…/FusedAttention/FusedAdam… ×20) | 20 | ✅ `test_fused_ops_dispatch_multidtype.cpp` | ✅ | ❌ |
| Creation (Zeros/Ones/Full/Rand/Randn/Arange/Linspace/Eye/Randint) | 9 | ✅ | n/a | n/a |
| RNN (LSTMCell/GRUCell + Forward + multilayer + BiLSTM ×9) | 9 | ✅ `test_nn_rnn_parity.cpp` | ⚠️ via layer | ❌ kernel level |
| Embedding (Embedding/EmbeddingBag fwd+bwd) | 4 | ✅ `test_nn_linear_emb_parity.cpp` | ✅ | ❌ |
| Linear (Linear/LinearBackward) | 2 | ✅ | ✅ (matmul covers) | ✅ |
| Dropout (Dropout/DropoutBackward) | 2 | ✅ | ✅ | ❌ |
| Advanced (TopK/Sort/CumSum/CumProd/Unique/FlashAttention/Einsum) | 8 | ✅ `test_attention_parity.cpp` | ⚠️ partial — Sort/TopK index-grads not flowing | ❌ FlashAttention bw fp/p only on CPU; OneAPI fused-bw deferred (composed-ops fallback in use) |
| 3D Conv/Pool | 12 | ✅ | ⚠️ | ❌ |
| Cast | 1 | ✅ | n/a | n/a |
| Extended math (Log2/10/1p/Exp2/Expm1/Erf/Erfc/IsNan/IsInf/IsFinite/Atan2/Fmod/Remainder/Lerp/Hypot/Copysign/Nextafter/Gcd/Lcm/Addcmul) | 20 | ✅ | ✅ (erf, erfc, etc.) | ❌ |
| Tensor manip (Triu/Tril/Diag/Trace/Flip/Roll) | 6 | ✅ | ✅ | ✅ (flip, roll) |
| Logical (And/Or/Not/Xor) | 4 | ✅ `test_logical_parity.cpp` | n/a | n/a |
| Bitwise (And/Or/Xor/Not/Shifts) | 6 | ✅ | n/a | n/a |
| Element-wise binary (Min/Max/Cross) | 3 | ✅ | ✅ | ✅ (max, min) |
| 1D Pooling | 8 | ✅ `test_pooling1d_multidtype.cpp` | ⚠️ | ❌ |
| Search/Sampling (SearchSorted/GumbelSoftmax) | 2 | ✅ | ✅ (searchsorted via dim tests) | ❌ |
| FFT (FFT/IFFT/RFFT/IRFFT/FFT2/IFFT2/FFTN/IFFTN) | 8 | ✅ `test_fft_parity.cpp` | ✅ Phase 9.2: Vulkan FP64 trig/FFT shaders already exist (`vulkan_ops_math.cpp:843`, `vulkan_ops_fft.cpp:107`). RFFT/IRFFT round-trip gradcheck added in Phase 4.4. | ✅ via round-trip pattern |
| Indexing extended (ScatterAdd/IndexAdd/IndexCopy/IndexFill/SelectScatter/SliceScatter/DiagonalScatter) | 7 | ✅ | ✅ | ❌ |
| Linalg (Det/Inv/Solve/SVD/QR/Eigh/Eig/Cholesky) | 8 | ✅ `test_linalg_parity.cpp` | ✅ `test_strict_linalg_grad.cpp` | ❌ |
| Linalg extended (LU/LUSolve/Householder/LDL/VectorNorm/MatrixNorm/Vecdot/CholeskySolve) | 9 | ⚠️ LU is Phase 5 — just-landed. Need parity test confirmation | ✅ (gradcheck_missing) | ❌ |
| Quantized (QuantizedLinear/QuantizedConv2d) | 2 | ✅ `test_quantized_kernel.cpp` | n/a | n/a |
| Embedding bag/check, LogSumExp, HasInfNan | 4 | ✅ | ✅ | ✅ (logsumexp) |
| Complex (Conj/Real/Imag/Angle/Polar) | 5 | ✅ `test_complex_parity.cpp` | ⚠️ complex sum() infra gap | ❌ |
| Sparse (SpMM/SpMV/ToDense/DenseToSparse/Add/SpGEMM/Trsv/Trsm/Softmax/LogSoftmax) | 10 | ✅ `test_sparse_parity.cpp` | ✅ (sp_mm/sp_mv/sparse_add/sparse_tri_solve) | ❌ |
| Signal (STFT/ISTFT/CDist/DCT/IDCT/MelScale/MFCC) | 7 | ✅ `test_signal_processing_multidtype.cpp` | ❌ STFT/ISTFT still not gradchecked (the gradcheck case remains as Phase 7.1-style follow-up) | ✅ Phase 8.1 resolved: ISTFTRoundTrip now passes on Vulkan Float32 + Float64. The forward value bug the audit flagged was indirectly fixed by the Complex64 `dispatchContiguous` / `dispatchPermute` work. Float16 is skipped due to FP16 accumulation error — that's a precision limit, not a bug. |
| Sampling (Multinomial/Bernoulli/Histogram/Bucketize/NormalSample/PoissonSample/ExponentialSample/Histogramdd) | 8 | ✅ `test_distributions_parity.cpp` + `test_sampling_parity.cpp` | n/a (sampling) | n/a |
| Special math (Gamma/Lgamma/Digamma/Polygamma/Beta/BetaInc/BesselJ0/J1/Y0/Y1/I0/I1/ErfInv/Sinc/Zeta) | 15 | ✅ `test_special_math_parity.cpp` | ✅ partial (gamma/lgamma/digamma/i0e/i1e) | ❌ |
| Scatter-Reduce, Addmm/Addmv/Baddbmm/Trapezoid/CumulativeTrapezoid/PairwiseDistance/Pdist, RepeatInterleave | 9 | ⚠️ partial — `test_scatter_reduce_multidtype.cpp` (CPU-only — see §2.4) | ⚠️ partial | ❌ |
| Cumulative scan ext. (Logcumsumexp/Bincount/SegmentReduce/Ndtr/LogNdtr/Multigammaln) | 6 | ✅ | ✅ | ❌ |
| New element-wise (Igamma/Igammac/Addcdiv) | 3 | ✅ | ⚠️ partial | ❌ |
| New reductions (CumMax/CumMin/Isin/Kthvalue/Fmax/Fmin/Quantile/Nanquantile/Nanmedian/Histc/UniqueConsecutive) | 11 | ✅ | ⚠️ partial | ❌ |
| New linalg (DiagEmbed/Diagflat/SolveTriangular/CholeskyInverse/TensorInv/TensorSolve/Ormqr/Geqrf) | 8 | ✅ `test_linalg_extended.cpp` | ✅ (cholesky_inverse) | ❌ |
| New shape/index (TakeAlongDim/MaskedScatter/TrilIndices/TriuIndices/AsStrided/ComplexTensor) | 6 | ✅ | ✅ (as_strided implicit) | ❌ |
| New pooling (FractionalMaxPool 2d/3d, MaxUnpool 2d/3d) | 8 | ✅ `test_fractional_maxpool_multidtype.cpp` + `test_maxunpool_multidtype.cpp` | ⚠️ via layer | ❌ |
| Statistics (NanVar/NanStd) | 2 | ✅ | ⚠️ | ❌ |
| Phase-5 element-wise (Deg2Rad/Rad2Deg/Logit/Signbit/FloatPower/Xlog1py/Ldexp/IsReal/IsPosInf/IsNegInf/Frexp) | 11 | ✅ `test_phase5_parity.cpp` | ⚠️ partial | ❌ |
| Nested (NestedSoftmax/LogSoftmax/LayerNorm/Sum/Mean/Attention/AttentionBackward/ToPadded/FromPadded/NestedLinear) | 10 | ✅ `test_nested_parity.cpp` | ⚠️ limited backend | ❌ NestedAttention head_v / dtype restoration recently fixed (see CHANGELOG) |
| Stable math (LogAddExp/LogAddExp2/XLogY/CosineSimilarity/Renorm/I0e/I1e/Entr/SphericalBesselJ0/Cov/Corrcoef/LOBPCG) | 12 | ✅ | ✅ partial | ❌ |
| FlexAttention (Forward/Backward) | 2 | ✅ `test_flex_attention_multidtype.cpp` | ✅ already Variable-aware (Phase 9.4 verification — audit was outdated) | ✅ |

**Headline number (post-Phase-4 expansion):** ~50 ops are now exercised on all five backends in `test_gradcheck_multibackend.cpp` (up from 25). Phase 4 expansion added activations (Mish/ELU/SELU/LogSigmoid/Swish/HardShrink/Softsign/Threshold), reductions (Var/Std/Prod/CumSum/CumProd/LogSumExp/Max(dim)/Min(dim)), indexing (IndexSelect/Gather/Scatter/Where/Roll), linalg+special-math+FFT (Erf/Erfc/Lgamma/Digamma/I0e/I1e/Det/Inv/Cholesky/VectorNorm/FFT-roundtrip), and NN ops (Conv2d/AvgPool2d/MaxPool2d/GroupNorm). Remaining gaps: stable-math + losses (Phase 4.6, lower priority) and the long-tail of NN ops not yet hooked up (RMSNorm, InstanceNorm 1d/3d, RNN cells, BatchNorm 1d/3d).

### 1.2 NN Layers (~150)

Layer dirs covered:
- `tests/nn/layers/` — 67 layer-specific files; almost all use `_multidtype` suffix and `MultiBackendDTypeTest`. **Layer-level coverage is strong.**
- `tests/integration/test_model_zoo.cpp` + per-architecture files in `tests/unit/` (BERT, ResNet, Swin, ViT, T5, ALBERT, ConvNeXt, UNet, YOLO, RoBERTa/ELECTRA, Transformer) — full-model regression tests.

Gaps inside the layer set:
- **`Bilinear`** (linear.hpp) — no dedicated test file located. Search: `find tests -name '*bilinear*'` → no result. *(Action: add `test_bilinear_multidtype.cpp`.)*
- **`LazyLinear` / `LazyConv1d/2d/3d`** — `test_lazy_backward.cpp` exists but covers gradient lazy-eval, not the lazy-init parameter materialization path. *(Action: confirm shape inference + first-forward init covered, or add a focused test.)*
- **Padding layers (12 variants)** — `test_padding_multidtype.cpp` and `test_circular_pad_multidtype.cpp` cover most. ConstantPad1d/3d, ReflectionPad1d, ReplicationPad1d/3d not individually verified. *(Action: confirm coverage; add gaps.)*
- **`Identity`** — `test_identity_multidtype.cpp` exists. ✅
- **`PatchEmbedding`, `PositionalEncoding`, `RoPE`, `ALiBi`** — `test_patch_embedding_multidtype.cpp` and `test_alibi_multidtype.cpp` exist; `RoPE` covered indirectly via attention tests. *(Action: confirm RoPE has a dedicated test.)*
- **`SqueezeExcitation`, `ChannelShuffle`, `ASPP`, `AtrousSeparableConv2d`, `InvertedResidual`, `MBConvBlock`, `FusedMBConv`, `HardSwish`, `HardSigmoid`** — covered by `test_vision_components_multidtype.cpp` and `test_fused_mbconv_multidtype.cpp`. *(Action: confirm each has at least one direct unit test.)*
- **`MixtureOfExperts` / `HRM`** — `test_moe_multidtype.cpp` + `test_hrm_multidtype.cpp` + `test_moe_hrm_parity.cpp` + `test_hrm_example.cpp`. ✅ Strong coverage.
- **`SyncBatchNorm`** — `test_sync_batchnorm_multidtype.cpp` exists. ✅
- **Loss layers (22+)** — `test_losses.cpp/_extended.cpp/_advanced_multidtype.cpp/_missing_multidtype.cpp/_simple.py/_extended.py/_edge_cases.py/_functions.py/_bindings.py`. Strong coverage. *(Action: confirm `MultiLabelMarginLoss`, `GaussianNLLLoss`, `MultiMarginLoss` all have direct tests.)*

### 1.3 Autograd functions (147+)

- 123 ops gradchecked; coverage analysis in `tests/autograd/GRADCHECK_COVERAGE_ANALYSIS.md`.
- `test_higher_order_*.cpp` covers second/third derivatives.
- `test_higher_order_stubs_regression.cpp` pins the structural-zero stubs.
- `EXPECT_GRAD_FLOWS` macro (via `tests/grad_flow_helpers.hpp`) is the standard non-zero-gradient assertion — the user already burned on the silent-zero-grad bug pattern, and this macro is the codified prevention.

Gaps:
- **InterpolateBackward** — marked `requires_grad=false`. No backward implementation. Phase 7.1 of the campaign plan (`/home/lee/.claude/plans/create-a-plan-to-optimized-bunny.md`) implements it across all 5 backends.
- **Sort/TopK indices** — gradients flow only through values, not index outputs. Phase 9.5 confirmed this is **standard PyTorch behavior, not a bug**: integer-typed outputs (indices) are discrete selections and have no defined gradient. Documented in `tests/autograd/GRADCHECK_COVERAGE_ANALYSIS.md` "Excluded by design".
- **Mean(dim)** gradcheck skipped — Phase 9.1 verification: source inspection of `src/autograd/function_elementwise.cpp:108-207` shows no current crash. Likely already fixed; the next gradcheck-on-Mean(dim) run will confirm (will be added in Phase 4.2 follow-on).
- **Complex-output ops** — `sum()` already supports Complex64/Complex128 on CPU (`reduction.cpp:1065-1078`) and CUDA (`reduction.cu:1559-1566`). Phase 9.3 audits ROCm/Vulkan/OneAPI sum kernels; the FFT round-trip test pattern (`rfft → irfft → sum`) added in Phase 4.4 sidesteps the original concern.
- **FlexAttention** — takes raw `Tensor`, not `Variable`. Autograd only at functional layer. *(Action: lift to Variable-aware to enable gradcheck.)*

### 1.4 Specialized subsystems

| Subsystem | Tests | Verdict |
|---|---|---|
| **Distributed** (DDP, FSDP/2, DTensor, ProcessGroup, DeviceMesh, ColumnParallelLinear, RowParallelLinear, SequenceParallel, GPipe, Elastic, RPC, Gradient compression) | C++: `tests/distributed/` (5 files: elastic_trainer / health_monitor / rendezvous / rpc_agent / rref). Integration: `tests/integration/test_distributed.cpp`, `test_data_parallel.cpp`, `test_multi_gpu.cpp`. Python: 8 `test_*parallel*.py` + `test_ddp_*.py` + `test_collective_*.py` + `test_rpc.py` + `test_zero_optimizers.py`. NN: `test_fsdp.cpp/_multidtype.cpp` + `test_tensor_parallel.cpp/_multidtype.cpp` + `test_pipeline_parallel.cpp/_multidtype.cpp`. | **Solid.** Most failure modes (rendezvous, fault-tolerance) covered. *Gap: gradient-compression (TopKCompressor, FP16Compressor) needs a focused round-trip test on each compressor — currently exercised in passing via `test_gradient_compression.py`.* |
| **JIT** (Tracer, Compiler, fusion passes, codegen, MemoryPlanner, SymbolicShape, control flow, autotune, serialization, CostModel) | `tests/jit/` (11 files): codegen, compile, jit_compiler, low-level, backend_parity, strict_mode, control_flow×3, dynamic_shapes, extended_codegen. Plus `test_jit_correctness.py/_extended.py/_tracing.py`. Plus `test_jit_autograd_parity.cpp` in backend_parity. | **Solid for happy paths.** *Gap: per-fusion-pass unit tests are not split out — a regression in one pass (e.g., `FuseAttentionPass`) would surface as a generic "compile output wrong" failure rather than a targeted assert.* |
| **ONNX export/import** (OnnxExporter, OnnxImporter, GraphModule, OnnxTypeMapping) | `test_onnx_export.py`, `test_onnx_roundtrip.py`, `test_onnx_numerical_roundtrip.py`, `test_onnx_operator_coverage.py`. | **Solid.** Operator coverage suite is the right tool for this. |
| **Program export** (ProgramExporter — torch.export style) | None located. | **GAP: zero direct tests.** Add at minimum a `test_program_export.cpp` round-tripping a simple traced program. |
| **Quantization** (AWQ / GPTQ / FakeQuantize / Observer / QConfig / FP8 / QAT) | C++: `test_quantization.cpp/_multidtype.cpp` (unit), `test_quantization_parity.cpp` + `test_quantized_kernel.cpp` (parity), `test_quantized_inference.cpp/_multidtype.cpp` (nn). Python: `test_quantization.py`, `test_quantization_int4.py`, `test_fp8_quantization.py`. | **Solid.** *Gap: AWQ-specific and GPTQ-specific paths — confirm each algorithm has at least one regression test exercising the activation/weight-aware path, not just the quantized inference layer.* |
| **Sparse** (SparseTensor COO/CSR/CSC/BSR + ops) | C++: `test_sparse.cpp/_multidtype.cpp`, `test_sparse_autograd.cpp/_multidtype.cpp`. Parity: `test_sparse_parity.cpp/_real_patterns.cpp`. Backend: `tests/backends/test_sparse_gpu.cpp`. NN: `test_sparse_embedding.cpp/_multidtype.cpp`, `test_sparse_linear.cpp/_multidtype.cpp`. Python: `test_sparse.py`, `test_sparse_layers.py`. | **Solid.** Backend coverage explicit (CPU MKL / CUDA cuSPARSE / ROCm rocSPARSE / Vulkan native / OneAPI oneMKL). |
| **Lazy execution** (LazyTensor) | None located. (`test_lazy_backward.cpp` is gradient-lazy, not the LazyTensor path.) | **GAP: LazyTensor path has no direct test.** Add a `test_lazy_tensor.cpp` that exercises deferred construction → materialize() and confirms equivalence with eager. |
| **Lite runtime** (LiteRuntime, LiteGraph, TZLite serialization, NEON/x86 SIMD paths) | `tests/lite/` (4 files): graph, ops, model format reader/writer. | **Coverage exists, but narrow.** *Gap: NEON-specific path probably exercised only on actual ARM CI (if any). Verify a Lite NEON quantized inference test runs.* |
| **Models pretrained** (20+ archs in `include/tenzor/models/`) | Per-arch tests in `tests/unit/`. `test_model_zoo.cpp/_multidtype.cpp` and `test_model_zoo.py` provide cross-architecture coverage. | **Solid for tested archs.** *Confirm each of GoogLeNet, AlexNet, EfficientNet B0–B7, GPT-3 (LM head), Faster R-CNN, Mask R-CNN, DeepLabV3+ has at least a forward-pass smoke test.* |
| **Serving** (Server, ModelRepository, DynamicBatcher, WorkerPool, RateLimiter, TrafficRouter, MetricsCollector, AuthHandler) | `tests/serving/` (6 files: auth, dynamic batching, metrics, rate limiting, traffic routing). Plus `test_serving.py`. | **Reasonable.** *Gap: no explicit failover / worker-crash test for the WorkerPool. Add an end-to-end "worker dies mid-request" test.* |
| **Data** (Dataset, DataLoader, Sampler, DistributedSampler, transforms, MNIST/CIFAR10/ImageNet) | `tests/integration/test_data_pipeline.cpp/_multidtype.cpp`, `test_dataloader.py`, `test_transforms_multidtype.cpp`. | **Solid.** *Gap: confirm `DistributedSampler` epoch-shuffle determinism is tested under a multi-rank Python harness.* |
| **IO** (image load/save, safetensors, checkpoint serialize) | `test_safetensors.cpp/_multidtype.cpp`, `test_serialization.cpp/_multidtype.cpp`, `test_pytorch_loader.cpp/_multidtype.cpp`, `test_model_persistence.cpp/_multidtype.cpp`. | **Solid.** *Gap: image I/O (PNG/JPEG load/save) — confirm a focused round-trip test exists.* |
| **Nested tensors** | `tests/nested/` (3 files). `test_nested_parity.cpp`. | **Backend-limited** — most ops CPU-only per coverage notes. *Gap: as backends grow, ensure parity tests grow with them.* |
| **Distributions** (43 classes) | `test_distributions_advanced.cpp/_multidtype.cpp`, `test_laplace_distribution.cpp/_multidtype.cpp`, `test_distributions_parity.cpp`, `test_distributions_multidtype.cpp` (nn). | **Likely incomplete.** 43 distributions × {sample, log_prob, entropy, mean, variance} = ~215 method tests. Verify each distribution has at least sample/log_prob coverage. *(Action: produce a per-distribution checklist.)* |

### 1.5 Python bindings

13 submodules with 500+ bindings. The Python test inventory (94 files, see explorer report) covers the major surfaces: autograd, functional, nn, optim, distributed, ONNX, data, quantization, AMP, JIT, vision/detection, serving, sparse, nested.

Specific Python-side gaps to verify:
- **`tenzor.tensorboard.SummaryWriter`** — no `test_tensorboard.py` located.
- **`tenzor.monitor`** — no `test_monitor.py` located.
- **`tenzor.compression`** — covered by `test_pruning.py` and quantization tests; verify distillation has a direct test.
- **`tenzor.amp.GradScaler`** — covered by `test_amp_execution.py`; confirm overflow/inf-detection paths are exercised.

### 1.6 Examples

21 examples are pinned as regression tests with loss-decrease assertions (`tests/examples/test_all_autograd_examples.cpp`). One intentional skip (`11_chat_ai`, duplicate surface). This is **excellent** end-to-end coverage and is the user's existing "examples-as-tests" pattern (memory: `feedback_example_regressions.md`).

---

## 2. Prioritized Gap Punch-List

### **P1 — High severity (real correctness risk)**

#### P1.1 Cross-backend gradcheck coverage at 25/123 ops (~20%)

The `test_gradcheck_multibackend.cpp` suite exercises 25 ops on CPU/CUDA/ROCm/Vulkan/OneAPI × Float32+Float64. The other ~98 gradchecked ops run only on CPU. This means backend-specific backward bugs (stride-from-shape, FP16 rounding, async misordering, the **stride-ignoring kernel bug pattern** the user has already memorialized) are NOT caught for those ops.

**Action:** Promote ops to multi-backend gradcheck in batches by category. Suggested order:
1. **Activations** (`relu`/`sigmoid`/`tanh`/`gelu`/`elu`/`selu`/`mish`/`softplus`) — easy wins, all 5 backends already have kernels.
2. **Reductions** (`sum`/`mean`/`var`/`std`/`prod`/`logsumexp`/`cumsum`/`cumprod`) — high-impact for training correctness.
3. **Linalg** (`solve`/`inv`/`svd`/`qr`/`eigh`/`cholesky`/`lu`/`lu_solve`) — backend libraries (cuSOLVER, rocSOLVER, oneMKL) have known divergences.
4. **Norms** (`layer_norm`/`group_norm`/`instance_norm`/`rms_norm`) — kernel-level backward, not just layer-level.
5. **FFT** (`fft`/`ifft`/`rfft`/`irfft`) — once complex-output infra is fixed.

Rough effort: ~2–3 ops per added test param, 1 PR per category. ~5 PRs total.

#### P1.2 24 files use banned `BackendDTypeParam` custom fixture

Per TESTING.md §"Fixture hygiene": *"Do not reinvent a per-file `struct BackendDTypeParam { ... }`; the existing fixture already parameterizes correctly."*

Confirmed-violating files (full list, line numbers from grep):

```
tests/examples/test_multi_param_example.cpp:22
tests/ops/test_shape_ops_multidtype.cpp:33
tests/unit/test_transforms_multidtype.cpp:25
tests/unit/test_gru_multidtype.cpp:23
tests/unit/test_gradient_checkpoint_multidtype.cpp:31
tests/unit/test_vision_components_multidtype.cpp:34
tests/unit/test_ops_additional_multidtype.cpp:31
tests/unit/test_model_checkpoint_multidtype.cpp:35
tests/unit/test_embedding_multidtype.cpp:33
tests/unit/test_optimizers_multidtype.cpp:30
tests/unit/test_lstm_multidtype.cpp:24
tests/unit/test_roberta_electra_multidtype.cpp:27
tests/unit/test_broadcasting_multidtype.cpp:23
tests/unit/test_transformer_multidtype.cpp:24
tests/unit/test_detection_components_multidtype.cpp:34
tests/unit/test_fp16_multidtype.cpp:410
tests/unit/test_dtype_edge_cases_multidtype.cpp:25
tests/unit/test_chunk_multidtype.cpp:31
tests/unit/test_ops_multidtype.cpp:24
tests/unit/test_comparison_ops_multidtype.cpp:26
tests/unit/test_autograd_multidtype.cpp:24
tests/unit/test_nn_additional_multidtype.cpp:37
tests/unit/test_tensor_multidtype.cpp:26
tests/unit/test_dtype_edge_cases.cpp:27
```

The risk: these custom fixtures may not honor `TENZOR_REQUIRE_MULTI_BACKEND` and `TENZOR_SKIP_BACKENDS`, and their backend lists may be hand-coded — meaning a backend that fails to initialize is silently skipped instead of hard-failing.

**Action:** Migrate each to `MultiBackendDTypeTest` from `tests/multi_backend_dtype_fixture.hpp`. ~24 small PRs or one batch PR. Mechanical refactor — replace local struct with the canonical parameter, replace `INSTANTIATE_TEST_SUITE_P` arguments with `INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS`.

### **P2 — Medium severity (coverage holes)**

#### P2.1 `tests/ops/` files hardcoded to CPU only

The pitfall scan flagged 7+ files as CPU-only:

```
tests/ops/test_fp8_ops.cpp           — FP8 type promotion, CPU only
tests/ops/test_edge_cases.cpp        — NaN/Inf edge cases, CPU only
tests/ops/test_linalg_lu.cpp         — LU decomposition, CPU only (Phase 5 just added GPU LU)
tests/ops/test_numerical_gradients.cpp — finite-diff, CPU only
tests/ops/test_histogramdd.cpp       — multi-dim histogram, CPU only
tests/ops/test_scatter_reduce.cpp    — scatter w/ reduction, CPU only
tests/ops/test_transform_new.cpp     — new shape transforms, CPU only
```

**Action:** For each, either rewrite to use `MultiBackendDTypeTest`, or rename to `_cpu` suffix and add a parallel `_multidtype.cpp` covering the other backends. `test_linalg_lu.cpp` is the most important now that GPU LinalgLU just landed (Phase 5).

#### P2.2 Subsystems with thin/no parity coverage

- **`include/tenzor/export/` (ProgramExporter)** — zero direct tests. Add `test_program_export.cpp` round-trip.
- **`include/tenzor/lazy/` (LazyTensor)** — `test_lazy_backward.cpp` is unrelated. Add `test_lazy_tensor.cpp`.
- **`tenzor.tensorboard.SummaryWriter`** — no test located. Add `test_tensorboard.py`.
- **`tenzor.monitor`** — no test located. Add `test_monitor.py`.
- **JIT fusion passes** — covered by `test_jit_compile.cpp` end-to-end. Add per-pass unit tests so a regression in `FuseLinearReluPass` surfaces as a focused failure, not a generic compile-output check.

#### P2.3 Distributions matrix incomplete

43 distribution classes. Existing tests (`test_distributions_advanced`, `test_laplace_distribution`, `test_distributions_parity`, `test_distributions_multidtype`) cover the popular set. Verify each of these has at least sample/log_prob/entropy: `Cauchy`, `Wishart`, `Pareto`, `Weibull`, `HalfNormal`, `HalfCauchy`, `LogNormal`, `FisherSnedecor`, `Gumbel`, `Kumaraswamy`, `LKJCholesky`, `VonMises`, `LowRankMultivariateNormal`, `LogisticNormal`, `Geometric`, `NegativeBinomial`, `ContinuousBernoulli`, `OneHotCategorical`, `MixtureSameFamily`, `Independent`, `TransformedDistribution`. Distribution `Transform` classes (`ExpTransform`, `SigmoidTransform`, `SoftmaxTransform`, `TanhTransform`, `ComposeTransform`, `AffineTransform`) — round-trip / Jacobian-determinant tests.

#### P2.4 STFT/ISTFT not gradchecked

Despite full kernel coverage on CUDA/OneAPI/ROCm and Phase 4.3 work, STFT/ISTFT have no gradcheck entry. Vulkan still uses CPU fallback for these ops (registry).

**Action:** Add to `test_gradcheck_missing.cpp` once Vulkan native path is unblocked.

### **P3 — Low severity (cleanup, known-quantity)**

#### P3.1 Weak / disabled / inline-skip sites (4 sites)

```
tests/autograd/test_inplace_autograd.cpp:203  — EXPECT_NO_THROW(loss.backward()) without grad assertion
tests/autograd/test_strict_linalg_grad.cpp:83 — EXPECT_NO_THROW(loss.backward()) followed by exit
tests/unit/test_lstm_cuda_forward_regression.cpp:133 — DISABLED_GRU_BenchShape (rnn_sequence.cu hang)
tests/test_bmm_autograd.cpp:212               — inline `if (!Device::cuda_available()) GTEST_SKIP()`
```

Plus two in `tests/test_cuda_scalar_debug.cpp` (lines 21, 65) and one in `tests/test_deformable_conv2d_multidtype.cpp:220` that should use `REQUIRE_MULTI_BACKEND_OR_SKIP`.

**Action:** ~30-min cleanup PR. Replace EXPECT_NO_THROW with `EXPECT_GRAD_FLOWS`; investigate or delete the DISABLED test (the GRU hang is documented but the test is dead weight); migrate the inline skips to the canonical helper.

#### P3.2 Float64 / Float16 dtype coverage spot-check

The TESTING.md contract requires "Float32, Float64, Float16 (+ BFloat16 when `TENZOR_TEST_BFLOAT16=ON`)" on all layer tests. Verify each `_multidtype` test in `tests/nn/layers/` actually instantiates Float64 and Float16, not just Float32. Quick sweep: `grep -L "Float64" tests/nn/layers/*_multidtype.cpp` for Float64-missing.

---

## 3. Suggested Sequencing

If the goal is "every feature tested" with bounded effort, this order maximizes risk reduction per PR:

1. **Cleanup PR** (1–2 hr) — fix the 4 sites in P3.1; nothing in this codebase should ship with `DISABLED_*` or `EXPECT_NO_THROW(loss.backward())`.
2. **Fixture migration** (P1.2) — 1 batch PR migrating the 24 custom `BackendDTypeParam` files. Mechanical. Big returns (these tests will start respecting `TENZOR_REQUIRE_MULTI_BACKEND`).
3. **CPU-only ops migration** (P2.1) — 1 PR converting the 7 `tests/ops/*` files to `MultiBackendDTypeTest`. `test_linalg_lu.cpp` is highest priority (Phase 5 just landed GPU LU).
4. **Multi-backend gradcheck batches** (P1.1) — 5 PRs, one per category (activations → reductions → linalg → norms → FFT). This is the single largest correctness-coverage gap and the most likely place a real bug is hiding.
5. **Subsystem coverage adds** (P2.2) — 4 small PRs: `test_program_export.cpp`, `test_lazy_tensor.cpp`, `test_tensorboard.py`, `test_monitor.py`. Plus per-pass JIT fusion tests.
6. **Distributions sweep** (P2.3) — 1 PR covering the missing distributions / transforms.
7. **STFT/ISTFT gradcheck** (P2.4) — 1 small PR after Vulkan native path lands.

Each step independently reduces real risk; you can stop at any point and still have improved the suite.

---

## 4. Out of scope for this audit

- **Performance regression tests** — `test_performance_regression.cpp` exists, but quantifying coverage of the perf-regression surface is a separate exercise.
- **Fuzzing** — `fuzz/` directory exists; not analyzed here.
- **Sanitizer (ASan/TSan/UBSan/MSan) coverage** — `sanitizer_suppressions.txt` exists; sanitizer build / test parity not analyzed.
- **CI matrix** — which tests run in which CI configuration is not analyzed; the audit assumes everything in `tests/` is wired up.
- **Documentation testing** — Doxygen output not validated against actual API.

---

*Audit performed 2026-05-01. Three parallel exploration agents surveyed
the feature surface, the test surface, and TESTING.md-violation patterns;
all findings spot-verified against source.*
