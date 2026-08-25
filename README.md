# Tenzor

A multi-backend tensor computation and deep learning library written in modern C++23, with full reverse-mode autograd and a PyTorch-like API exposed in both C++ and Python.

> **Status: alpha (v0.2.0).** Single-developer research project. The API is reasonably stable but has had no public-CI exposure on GPU hardware yet. Treat it as experimental, not as a production replacement for PyTorch or TensorFlow.

[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![C++](https://img.shields.io/badge/C%2B%2B-23-blue)]()
[![Python](https://img.shields.io/badge/Python-3.9--3.13-blue)]()

## What/How/Why

I started this as a way to learn how AI and neural networks work, especially how tensors are involved. It started off with just me and a lot of googling, and then came Claude — so I have to admit probably 85-90% of this library is Claude's work. It basically ended up teaching me an awful lot; I led the direction of the project. It's taken me 2 years to get to the point where I feel comfortable using it. A few caveats: distributed training/inference hasn't been tested since I only have one GPU, ROCm has only been tested on my APU, oneAPI has only been tested on my CPU, I have only tested on Arch Linux(native), and on Windows11(virtual machine with CPU backend only) and for Apple users, please dont get your hopes up, I have no apple hardware. 


## What's in the box

- **Multi-backend autograd** with native dispatch tables across CPU, CUDA, ROCm, OneAPI, and Vulkan. Generate current per-backend registration counts with `bin/op_coverage_report --json` after configuring the backend set you want to inspect.
- **Multi-dtype dispatch**: Float32/64/16, BFloat16, Int8/16/32/64, UInt8/16/32/64, Bool, Complex64/128.
- **NN module library**: Linear, Conv1d/2d/3d, BatchNorm/LayerNorm/GroupNorm/RMSNorm, MaxPool/AvgPool/Adaptive, RNN/LSTM/GRU (bidirectional), MultiheadAttention/ScaledDotProductAttention, dropout variants, embedding/embedding-bag, activations, sequential containers.
- **Optimizers and schedulers**: SGD (+ Nesterov), Adam, AdamW, AdamAtan2, RMSprop, Adagrad. StepLR, MultiStepLR, ExponentialLR, CosineAnnealing(WarmRestarts), OneCycleLR, ReduceLROnPlateau, polynomial/linear warmup.
- **Reference model implementations** in `src/models/`: ResNet, VGG, MobileNet (V2/V3), EfficientNet, ConvNeXt, ViT, Swin Transformer, YOLO, Faster R-CNN, Mask R-CNN, U-Net, DeepLabV3+, BERT, RoBERTa, ALBERT, ELECTRA, GPT-2, T5, AlexNet, GoogLeNet, plus an HRM (Hierarchical Reasoning Model) example. These are model definitions; pretrained weights are not currently distributed.
- **Python bindings** via pybind11, with NumPy interop.
- **ONNX import/export**.
- **Mixed precision** (FP16/BF16) and **INT8 quantization** code paths (post-training and QAT).
- **JIT** tracing/scripting and a kernel-fusion pass.

## Quick start

### Python

```python
import tenzor as tz  # backends auto-initialize on import (set TENZOR_AUTO_INIT=0 to defer)

x = tz.randn(32, 784)
model = tz.nn.Sequential(
    tz.nn.Linear(784, 128),
    tz.nn.ReLU(),
    tz.nn.Linear(128, 10),
)
if tz.cuda_is_available():
    x, model = x.cuda(), model.cuda()

opt = tz.optim.Adam(model.parameters(), lr=1e-3)
pred = model(x)
loss = tz.nn.cross_entropy(pred, tz.zeros([32], dtype=tz.int64))
loss.backward()
opt.step()
```

### C++

```cpp
#include <tenzor/tenzor.hpp>

int main() {
    using namespace tenzor;

    auto model = nn::Sequential(
        std::make_shared<nn::Linear>(784, 128),
        std::make_shared<nn::ReLU>(),
        std::make_shared<nn::Linear>(128, 10)
    );

    auto optimizer = optim::Adam(model.parameters(), 1e-3);

    for (auto [x, y] : dataloader) {
        auto pred = model(Variable(x, true));
        auto loss = nn::cross_entropy(pred, y);
        optimizer.zero_grad();
        loss.backward();
        optimizer.step();
    }
}
```

## Building

### Prerequisites

- CMake 3.25+
- A C++23 compiler (GCC 13+, Clang 15+, MSVC 19.34+)
- Optional: CUDA 12.0+, ROCm 5.0+, oneAPI 2023.0+, Vulkan SDK 1.2+
- Optional: Python 3.9+ (for the bindings)

### From source

```bash
git clone https://github.com/skreamz/Tenzor.git
cd Tenzor
cmake -B build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DTENZOR_BUILD_PYTHON=ON \
    -DTENZOR_BUILD_TESTS=ON
cmake --build build -j
```

### Build options

| Option | Default | Notes |
|---|---|---|
| `TENZOR_BUILD_CUDA`       | ON  | Requires CUDA Toolkit |
| `TENZOR_BUILD_ROCM`       | ON  | Requires ROCm + HIP   |
| `TENZOR_BUILD_ONEAPI`     | ON  | Requires Intel oneAPI |
| `TENZOR_BUILD_VULKAN`     | ON  | Requires Vulkan SDK + glslc |
| `TENZOR_BUILD_MPS`        | ON (Apple only) | Apple Metal/MPS backend; option only defined when `APPLE` |
| `TENZOR_BUILD_PYTHON`     | ON  | Requires Python dev headers |
| `TENZOR_BUILD_TESTS`      | ON  | GoogleTest (fetched automatically) |
| `TENZOR_BUILD_BENCHMARKS` | ON  | |
| `TENZOR_USE_MLIR_JIT`     | autodetected | ON if an IREE distribution is found (`third_party/iree_dist/`, system install, or `$IREE_DIR`), else OFF; always overridable explicitly. See `docs/jit-mlir-setup.md` |

See [INSTALL.md](INSTALL.md) for per-backend setup details.

## Documentation

- [Getting Started](docs/GETTING_STARTED.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Installation Guide](INSTALL.md)
- [Testing Guide](TESTING.md)
- [FAQ](docs/FAQ.md)
- [Contributing](CONTRIBUTING.md)
- [Changelog](CHANGELOG.md)

## Performance

The benchmark harness is available in `scripts/ci_benchmark.sh` and `tools/regen_perf_baseline.py`, with a PyTorch-comparison suite in `benchmarks/python/` (`run_benchmarks.py --device cpu|cuda`). Numbers below are from the maintainer's own machine and are meant as orientation, not a competitive claim — re-run the harness on your own hardware before relying on any of this for a decision.

**Test machine:** (Framework 16) AMD Ryzen AI 9 HX 370 (12C/24T), 96 GiB RAM, NVIDIA RTX 5070 Laptop GPU (8 GB), Linux, PyTorch 2.13.0. Both sides re-run together on 2026-08-13: 275 CPU results (126 op configurations with a matching PyTorch counterpart) and 300 CUDA results (135 matching configurations), 5 timed iterations each after 10 warmup iterations.

Reference point — 4096×4096 FP32 matmul, wall-clock mean:

| | CPU | CUDA |
|---|---|---|
| Tenzor  | 188.5 ms (729.3 GFLOPS) | 10.08 ms (13.6 TFLOPS) |
| PyTorch | 197.5 ms (696.0 GFLOPS) | 10.11 ms (13.6 TFLOPS) |

Median Tenzor/PyTorch time ratio by op category (>1.0 = Tenzor slower; based on the run above, not a guarantee for other shapes or hardware):

| Category | CPU | CUDA |
|---|---|---|
| matmul | 1.3x | 1.0x |
| conv2d | 0.7x | 1.1x |
| linear | 1.1x | 0.9x |
| attention | 2.2x | 1.3x |
| layernorm | 2.8x | 1.0x |
| batchnorm | 0.8x | 1.1x |
| rmsnorm | 2.0x | 0.3x |
| embedding | 1.4x | 1.0x |
| lstm | 1.5x (train: 3.9x) | 1.4x (train: 1.2x) |
| gru | 0.4x | 1.1x |
| elementwise activations | 1.5x | 0.9x |
| full training step (end-to-end) | 1.1x | 0.6x |

Four things worth calling out rather than glossing over:
- **CUDA LayerNorm and RMSNorm are transformed.** LayerNorm went from 7.8x slower to parity (1.0x), and RMSNorm went from 1.3x slower to 3-4x *faster* than PyTorch (0.3x median). This tracks with the recent OMP-gating and kernel fixes (`58607880`, `aa568ba9`, `984acde6`) and the cuDNN RNN weight/descriptor caching work — CUDA normalization is no longer the worst-tuned part of the library.
- **CPU LayerNorm regressed** in this same window, from 1.7x to 2.8x slower — worth root-causing before the next release, since it moves in the opposite direction from the CUDA fix. CPU elementwise activations (Sigmoid/ReLU/GELU/Tanh) improved instead, from 2.6x to 1.5x median.
- **LSTM vs. GRU still diverge sharply on CPU**: GRU is consistently *faster* than PyTorch at every size tested (2.3-3.3x), while LSTM is consistently *slower* and gets worse with size (1.2x at the smallest size, 2.3x at the largest). LSTM training is now the single largest regression found: `LSTM Medium 2L (train)` is 3.9x slower (637 ms vs. PyTorch's 163 ms), up from 2.9x in the previous measurement — this points at an LSTM-specific kernel/backward inefficiency, not generic RNN or autograd overhead. CUDA doesn't show the same divergence (LSTM 1.4x, GRU 1.1x, both close to parity).
- **CNN training is still 5-10x slower end-to-end on CPU** (`Train CNN (LeNet-style) B=64`: 99 ms vs. 11 ms) even though standalone conv2d forward kernels are now at parity or faster in isolation (0.7x median). The gap lives in the backward path (conv2d backward / MaxPool2d backward), not the forward conv kernel — and it's CPU-specific: the same CNN training benchmarks on CUDA are within 0.8-1.2x of PyTorch. CUDA's full training step is now measured end-to-end for the first time and comes out *faster* than PyTorch overall (0.6x median).

The previously-flagged `Train small_mlp` CPU anomaly (Tenzor ~20-55x faster) reproduces consistently across all three batch sizes tested (32/64/128) with low variance, using the same benchmark harness path as every other entry in this table — it's very likely a genuine per-step Python/dispatch-overhead difference for tiny models rather than a measurement artifact, but it's still overhead-dominated rather than a kernel-throughput result, so it's excluded from the category medians above. The CUDA `Train small_mlp` benchmarks don't show this anomaly (0.4-0.6x, in line with the rest of the training category).

Fused and vendor-library-heavy paths (cuDNN/cuBLAS-backed ops in particular) are where an established library's engineering investment shows most. See [`CHANGELOG.md`](CHANGELOG.md) for release-specific performance notes, and treat any single number here as a snapshot of one machine on one day, not a general claim about either library.

## Architecture

```
Python bindings (pybind11)
        │
High-level NN API
        │
Autograd engine
        │
Core tensor ops (O(1) OpId dispatch)
        │
Backend abstraction
        │
[CPU] [CUDA] [ROCm] [OneAPI] [Vulkan]
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full diagram and dispatch internals.

## Known limitations

- **No public CI proof for GPU backends.** GitHub Actions runs CPU smoke tests only; CUDA/ROCm/OneAPI/Vulkan are tested locally on the maintainer's hardware.
- **GPU backend ops are expected to be native.** Backend gaps should fail clearly rather than silently dispatching through CPU behavior.
- **MPS backend** (Apple Metal) is partial. Not yet at parity with the four primary GPU backends.
- **LayerNorm is unoptimized on CPU** relative to PyTorch (2.8x slower in the current benchmark run, a regression from a prior 1.7x); the CUDA side has closed its gap entirely (down from 7.8x to parity) after recent OMP-gating and caching fixes. See [Performance](#performance).
- **LSTM is notably slower than PyTorch on CPU**, especially in training (3.9x on the median, worst case 3.9x wall-clock on a 2-layer model) — GRU does not show the same gap (2.3-3.3x *faster* than PyTorch), so this looks like an LSTM-specific kernel/backward issue rather than general RNN overhead; CUDA LSTM/GRU don't show it either. CNN training is also 5-10x slower end-to-end on CPU despite conv2d forward kernels being at parity, pointing at the conv2d/MaxPool2d backward path — CUDA CNN training doesn't show this gap. Neither has been root-caused yet. See [Performance](#performance).
- **Pretrained weights** are not distributed. The model files in `src/models/` are architectures only.
- **Single maintainer.** Contributions, issues, and reviews are welcome.

## Roadmap

- Public GPU CI on self-hosted runners.
- Re-run benchmarks with proper warmup/median-of-N and publish GPU numbers.
- Distributed training improvements.
- Pretrained weight hub for the reference models.

## License

MIT — see [LICENSE](LICENSE).

## Citation

If you use Tenzor in your research:

```bibtex
@software{tenzor,
  title  = {Tenzor: A multi-backend tensor and deep learning library in C++23},
  author = {Morton, Lee},
  year   = {2026},
  url    = {https://github.com/skreamz/Tenzor}
}
```

## Acknowledgments

The API surface is heavily inspired by PyTorch. The internal design borrows ideas from PyTorch's ATen, oneDNN, and the cuDNN/cuBLAS reference patterns.

## Contact

- [GitHub Issues](https://github.com/skreamz/Tenzor/issues) — bug reports and feature requests.
- [Discussions](https://github.com/skreamz/Tenzor/discussions) — questions and design.
