# Contributing to Tenzor

Thank you for your interest in contributing to Tenzor! This document provides guidelines for contributing to the project.

## Development Setup

### Prerequisites

- CMake 3.25 or higher
- C++23 compatible compiler (GCC 13+, Clang 15+, MSVC 2022+)
- CUDA 12.0+ (optional, for GPU support)
- Python 3.9+ (optional, for bindings)

### Building from Source

```bash
# Clone the repository
git clone https://github.com/skreamz/Tenzor.git
cd Tenzor

# Create build directory
mkdir build && cd build

# Configure with desired options
cmake .. \
    -DCMAKE_BUILD_TYPE=Debug \
    -DTENZOR_BUILD_CUDA=ON \
    -DTENZOR_BUILD_PYTHON=ON \
    -DTENZOR_BUILD_TESTS=ON \
    -DTENZOR_BUILD_EXAMPLES=ON

# Build
cmake --build . -j$(nproc)

# Run tests
ctest --output-on-failure
```

## Project Structure

```
tenzor/
├── include/tenzor/       # Public headers
├── src/                  # Implementation files
│   ├── core/            # Core tensor infrastructure
│   ├── ops/             # Tensor operations
│   ├── autograd/        # Automatic differentiation
│   ├── nn/              # Neural network components
│   └── backends/        # Backend implementations
├── tests/               # Test files
│   ├── unit/           # Unit tests
│   └── integration/    # Integration tests
├── examples/            # Example programs
├── python/              # Python bindings
└── docs/                # Documentation
```

## Coding Standards

### C++ Style

- Use modern C++23 features
- Follow the project's naming conventions:
  - `snake_case` for functions and variables
  - `PascalCase` for classes and types
  - `UPPER_CASE` for constants and macros
- Use `auto` where type is obvious
- Prefer smart pointers over raw pointers
- Use RAII for resource management

### Example

```cpp
// Good
auto create_tensor(std::vector<int64_t> shape) -> Tensor {
    return Tensor(std::move(shape), DType::Float32, Device::cpu());
}

// Avoid
Tensor* create_tensor(vector<int64_t> shape) {
    return new Tensor(shape, DType::Float32, Device::cpu());
}
```

### Header Guards

Use `#pragma once` instead of traditional include guards.

### Documentation

- Document all public APIs with Doxygen-style comments
- Include usage examples for complex functions
- Explain non-obvious implementation details

```cpp
/**
 * @brief Performs matrix multiplication on two tensors
 *
 * @param a Left-hand side tensor of shape [..., M, K]
 * @param b Right-hand side tensor of shape [..., K, N]
 * @return Tensor Result tensor of shape [..., M, N]
 *
 * @code
 * auto a = randn({2, 3});
 * auto b = randn({3, 4});
 * auto c = matmul(a, b);  // Shape: [2, 4]
 * @endcode
 */
auto matmul(const Tensor& a, const Tensor& b) -> Tensor;
```

## Testing

### Writing Tests

- Use Google Test framework
- Place unit tests in `tests/unit/`
- Place integration tests in `tests/integration/`
- Name test files `test_<component>.cpp`

### Example Test

```cpp
#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>

TEST(TensorTest, Addition) {
    auto a = tenzor::ones({2, 2});
    auto b = tenzor::ones({2, 2});
    auto c = a + b;

    auto data = c.data<float>();
    for (int i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ(data[i], 2.0f);
    }
}
```

### Running Tests

```bash
cd build
ctest --output-on-failure

# Run specific test
./tests/tenzor_unit_tests --gtest_filter=TensorTest.Addition
```

## Pull Request Process

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Make** your changes
4. **Add** tests for new functionality
5. **Ensure** all tests pass
6. **Format** code consistently
7. **Commit** changes (`git commit -m 'Add amazing feature'`)
8. **Push** to branch (`git push origin feature/amazing-feature`)
9. **Open** a Pull Request

### PR Guidelines

- Keep PRs focused on a single feature or fix
- Update documentation for API changes
- Add tests for new features
- Ensure CI passes before requesting review
- Reference related issues in PR description

## Adding New Operations

Ops are dispatched by `OpId` (see `include/tenzor/ops/op_id.hpp`), not by
string-keyed registry lookup — the walkthrough below matches the real dispatch
path. See also the "Adding a new op: coverage checklist" section further down
for the full per-PR requirement list.

### 1. Add the OpId

Add an entry to the `OpId` enum in `include/tenzor/ops/op_id.hpp`.

### 2. Define Interface

Add declaration to the appropriate header (e.g., `include/tenzor/ops/math.hpp`):

```cpp
auto my_operation(const Tensor& input, float param) -> Tensor;
```

### 3. Implement Dispatcher

Add implementation in the corresponding source file using `OpAttributes` +
OpId-based dispatch:

```cpp
auto my_operation(const Tensor& input, float param) -> Tensor {
    OpAttributes attrs;
    attrs.set(AttrKey::Alpha, param);  // pick the AttrKey that fits your op's semantics — see include/tenzor/backend/op_attributes.hpp for the full list
    return dispatch<OpId::MyOperation>({input}, attrs)[0];
}
```

### 4. Implement Backend Kernels

Add a CPU implementation in `src/backends/cpu/kernels/`:

```cpp
auto my_operation_cpu(const Tensor& input, float param) -> Tensor {
    // CPU-optimized implementation
}
```

Add a CUDA implementation in `src/backends/cuda/kernels/`:

```cuda
__global__ void my_operation_kernel(const float* input, float* output,
                                    float param, int64_t n) {
    // CUDA kernel implementation
}
```

Do the same for every other backend the op is expected to support (ROCm,
OneAPI, Vulkan, MPS) — see `TESTING.md`'s "New op requirements" for when a gap
is acceptable versus a hard failure.

### 5. Register Kernels

Register in each backend's kernel registry (e.g.
`src/backends/cpu/cpu_kernel_registry.cpp`) using the
`TENZOR_REGISTER_UNARY_KERNEL` / `TENZOR_REGISTER_BINARY_KERNEL` /
`TENZOR_REGISTER_KERNEL` macros from `include/tenzor/backend/kernel_registry.hpp`:

```cpp
TENZOR_REGISTER_UNARY_KERNEL(table, MyOperation, cpu::my_operation_cpu);
```

For kernels that don't fit the unary/binary lambda shape, register directly
against the `BackendDispatchTable`:

```cpp
table.register_kernel(OpId::MyOperation,
    [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
        return {my_operation_cpu(inputs[0], attrs.get_float(AttrKey::Alpha))};
    });
```

### 6. Add Tests

Create comprehensive tests in `tests/unit/` or `tests/ops/` using
`MultiBackendDTypeTest` (see `TESTING.md`):

```cpp
TEST(OpsTest, MyOperation) {
    auto input = randn({2, 3});
    auto output = my_operation(input, 1.5f);

    EXPECT_EQ(output.shape(), input.shape());
    // Add more assertions
}
```

### 7. Add Autograd Support

If the operation needs gradient support, implement an autograd `Function`
that dispatches to the backward OpId — see `tests/backend_parity/README.md`'s
gradient-parity section and `TESTING.md`'s "Autograd invariants" for the
patterns that must (and must not) be used.

## Performance Optimization

### Profiling

- Use appropriate profiling tools:
  - **CPU**: Valgrind, perf, Intel VTune
  - **CUDA**: nsys, ncu

### Benchmarking

Add benchmarks in `benchmarks/`:

```cpp
auto benchmark_matmul() -> void {
    constexpr int iterations = 100;
    auto a = randn({1024, 1024});
    auto b = randn({1024, 1024});

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        auto c = matmul(a, b);
    }
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
        end - start).count();

    std::cout << "Average time: " << (duration / iterations) << " μs\n";
}
```

## Backend Development

### Creating a New Backend

1. Create backend class in `src/backends/<backend_name>/`
2. Implement `Backend` interface
3. Implement kernel functions
4. Add to CMake build system
5. Export factory function

See `src/backends/cpu/cpu_backend.cpp` for reference implementation.

## Documentation

- Keep README.md up to date
- Update docs/ARCHITECTURE.md for architectural changes
- Generate API documentation with Doxygen

```bash
# Generate documentation
doxygen Doxyfile
```

## Getting Help

- **Issues**: [GitHub Issues](https://github.com/skreamz/Tenzor/issues)
- **Discussions**: [GitHub Discussions](https://github.com/skreamz/Tenzor/discussions)

## Adding a new op: coverage checklist

When you introduce a new `OpId`, make sure every item below is in your PR
before requesting review. The two audit scripts ship in `scripts/` and are
invoked in CI; use them locally to preview coverage deltas.

1. **Implementation** in `src/ops/` or `src/nn/` as appropriate.
2. **Kernel registration** on every backend the op supports: `src/backends/
   {cpu,cuda,rocm,vulkan,oneapi,mps}/<backend>_kernel_registry.{cpp,mm}`.
3. **Unit test** in `tests/ops/` that covers the op's happy path for every
   supported dtype via `MultiBackendDTypeTest`.
4. **Parity test** in `tests/backend_parity/` using `test_operation_parity`
   / `test_operation_parity_single`. If the op is non-obvious, also record
   goldens so single-backend CI verifies it (see
   `tests/backend_parity/README.md`).
5. **Python binding** in the appropriate `python/bindings/bindings_*.cpp`
   submodule + a functional test in `tests/python/`.
6. **ONNX mapping** in the exporter and a coverage entry in
   `tests/python/test_onnx_operator_coverage.py` (if the op is expressible
   in ONNX).
7. **Audit** — `scripts/audit_op_coverage.py` and `scripts/count_skips.py`
   locally. Neither should regress (the CI job enforces this).

For behavior changes on an existing op, still run both audit scripts and
re-record any affected goldens.

## Required status checks & branch protection

The cross-backend **parity** and **perf** gates only protect `main` if branch
protection lists them as *required status checks*. Branch protection lives in
GitHub repo settings, **not** in the repo, so it has to be configured once via
the UI or the API — adding a job to `ci.yml` alone does not make it gating.

### Required check names

Use the job **`name:`** values from `.github/workflows/ci.yml` (GitHub keys
required checks by the displayed check name):

| Job key (yaml)   | Required check name                | Gates on            |
|------------------|------------------------------------|---------------------|
| `build-and-smoke`| `Build & Smoke Tests`              | every PR            |
| `full-cpu-tests` | `CPU Tests (unit)`, `CPU Tests (nn)`, `CPU Tests (autograd)`, `CPU Tests (ops)`, `CPU Tests (jit)`, `CPU Tests (core)`, `CPU Tests (backend_parity)`, `CPU Tests (integration)` | every PR (golden floor) — this job is sharded by `matrix.shard`, so each shard is its own check; list all eight if you want full-suite gating |
| `gpu-smoke`      | `GPU Smoke (self-hosted)`          | every PR            |
| `gpu-parity`     | `GPU Backend Parity (self-hosted)` | every PR (fast tier)|
| `python-tests`   | `Python Tests`                     | every PR            |
| `macos-mps`      | `macOS MPS Backend`                | every PR            |

> The two GPU jobs require a **registered self-hosted runner** with labels
> `self-hosted, linux, gpu` that has CUDA + ROCm + Vulkan + OneAPI installed
> (repo **Settings → Actions → Runners**). Until that runner exists the GPU
> checks stay pending; do not mark them required before the runner is online,
> or every PR will block forever.

### Configure with `gh api`

```bash
# Requires: gh auth login  (admin on the repo)
OWNER=your-org REPO=tenzor BRANCH=main

gh api -X PUT \
  "repos/$OWNER/$REPO/branches/$BRANCH/protection" \
  -H "Accept: application/vnd.github+json" \
  --input - <<'JSON'
{
  "required_status_checks": {
    "strict": true,
    "checks": [
      { "context": "Build & Smoke Tests" },
      { "context": "CPU Tests (unit)" },
      { "context": "CPU Tests (nn)" },
      { "context": "CPU Tests (autograd)" },
      { "context": "CPU Tests (ops)" },
      { "context": "CPU Tests (jit)" },
      { "context": "CPU Tests (core)" },
      { "context": "CPU Tests (backend_parity)" },
      { "context": "CPU Tests (integration)" },
      { "context": "GPU Smoke (self-hosted)" },
      { "context": "GPU Backend Parity (self-hosted)" },
      { "context": "Python Tests" },
      { "context": "macOS MPS Backend" }
    ]
  },
  "enforce_admins": true,
  "required_pull_request_reviews": { "required_approving_review_count": 1 },
  "restrictions": null,
  "required_linear_history": true,
  "allow_force_pushes": false,
  "allow_deletions": false
}
JSON
```

> `strict: true` means a PR must be up to date with `main` before merge, so a
> green parity run can't be invalidated by an interim merge. Drop the GPU rows
> from the `checks` list if you need to land changes before the self-hosted
> runner is registered, then add them back.

### Perf and benchmark gates

* **Cross-backend perf regression** (`tests/backend_parity/`): enforced inside
  `gpu-parity` on the full/main tier via `TENZOR_PERF_ENFORCE=1` against the
  per-host entry in `tests/backend_parity/baselines/perf_baseline.json`.
  Regenerate per host with `scripts/test_all_backends.sh --perf-baseline`.
* **Nightly benchmarks** (`benchmarks`): `scripts/compare_benchmarks.py` fails
  on a >7% regression against the committed `benchmarks/baselines/<host>.json`.
  Refresh the floor via a reviewed PR (see `benchmarks/baselines/README.md`).

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
