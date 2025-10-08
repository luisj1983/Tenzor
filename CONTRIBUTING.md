# Contributing to Tenzor

Thank you for your interest in contributing to Tenzor! This document provides guidelines for contributing to the project.

## Development Setup

### Prerequisites

- CMake 3.25 or higher
- C++23 compatible compiler (GCC 12+, Clang 15+, MSVC 2022+)
- CUDA 12.0+ (optional, for GPU support)
- Python 3.8+ (optional, for bindings)

### Building from Source

```bash
# Clone the repository
git clone https://github.com/yourusername/tenzor.git
cd tenzor

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

### 1. Define Interface

Add declaration to appropriate header (e.g., `include/tenzor/ops/math.hpp`):

```cpp
auto my_operation(const Tensor& input, float param) -> Tensor;
```

### 2. Implement Dispatcher

Add implementation in corresponding source file:

```cpp
auto my_operation(const Tensor& input, float param) -> Tensor {
    OpAttributes attrs;
    attrs["param"] = std::to_string(param);
    return Dispatcher::dispatch("my_operation", {input}, attrs)[0];
}
```

### 3. Implement Backend Kernels

Add CPU implementation in `src/backends/cpu/kernels/`:

```cpp
auto my_operation_cpu(const Tensor& input, float param) -> Tensor {
    // CPU-optimized implementation
}
```

Add CUDA implementation in `src/backends/cuda/kernels/`:

```cuda
__global__ void my_operation_kernel(const float* input, float* output,
                                    float param, int64_t n) {
    // CUDA kernel implementation
}
```

### 4. Register Kernels

Register with operation registry:

```cpp
operation_registry().register_kernel(
    "my_operation",
    Device::Type::CPU,
    my_operation_cpu
);
```

### 5. Add Tests

Create comprehensive tests in `tests/unit/test_ops.cpp`:

```cpp
TEST(OpsTest, MyOperation) {
    auto input = randn({2, 3});
    auto output = my_operation(input, 1.5f);

    EXPECT_EQ(output.shape(), input.shape());
    // Add more assertions
}
```

### 6. Add Autograd Support

If operation needs gradient support, implement autograd function:

```cpp
class MyOperationBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;
};
```

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
- Update DESIGN.md for architectural changes
- Generate API documentation with Doxygen

```bash
# Generate documentation
doxygen Doxyfile
```

## Getting Help

- **Issues**: [GitHub Issues](https://github.com/yourusername/tenzor/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/tenzor/discussions)
- **Email**: your.email@example.com

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
