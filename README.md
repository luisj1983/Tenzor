# Tenzor

A multi-backend tensor computation and deep learning library written in modern C++23, with full reverse-mode autograd and a PyTorch-like API exposed in both C++ and Python.

> **Status: alpha (v0.1.0).** Single-developer research project. The API is reasonably stable but has had no public-CI exposure on GPU hardware yet. Treat it as experimental, not as a production replacement for PyTorch or TensorFlow.

[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![C++](https://img.shields.io/badge/C%2B%2B-23-blue)]()
[![Python](https://img.shields.io/badge/Python-3.9--3.13-blue)]()

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

The benchmark harness is available in `scripts/ci_benchmark.sh` and `tools/regen_perf_baseline.py`. Publish benchmark reports from generated artifacts for the exact hardware, backend options, and dependency versions used.

**Honest read:** Tenzor is competitive on some unfused-compute primitives, but fused and vendor-library-heavy paths need per-backend benchmarking before making broad performance claims. See [`CHANGELOG.md`](CHANGELOG.md) for release-specific performance notes.

If you need raw production throughput today, use PyTorch. If you want a clean, hackable C++23 codebase that's competitive on bread-and-butter linear algebra and that you can extend — Tenzor is for you.

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
- **CPU performance** is below PyTorch on the published benchmark suite. Conv2D in particular has a regression that needs investigation (one shape measures 0.07× — almost certainly a dispatch/warmup artifact).
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
