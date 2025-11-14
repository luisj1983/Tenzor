#include "tenzor/ops/math.hpp"
#include "tenzor/backend/dispatch.hpp"
#include "tenzor/ops/indexing.hpp"
#include "tenzor/ops/transform.hpp"
#include "tenzor/ops/creation.hpp"
#include <cstring>
#include <stdexcept>
#include <iostream>
#include <vector>

namespace tenzor {

// Math operation implementations - dispatched to backend kernels

auto add(const Tensor& a, const Tensor& b) -> Tensor {
    // Ensure tensors are contiguous before element-wise operation
    // Permute and reshape can create non-contiguous views
    Tensor a_contiguous = a.is_contiguous() ? a : a.contiguous();
    Tensor b_contiguous = b.is_contiguous() ? b : b.contiguous();
    std::vector<Tensor> inputs = {a_contiguous, b_contiguous};
    return Dispatcher::dispatch("add", inputs)[0];
}

auto sub(const Tensor& a, const Tensor& b) -> Tensor {
    // Ensure tensors are contiguous before element-wise operation
    Tensor a_contiguous = a.is_contiguous() ? a : a.contiguous();
    Tensor b_contiguous = b.is_contiguous() ? b : b.contiguous();
    std::vector<Tensor> inputs = {a_contiguous, b_contiguous};
    return Dispatcher::dispatch("sub", inputs)[0];
}

auto mul(const Tensor& a, const Tensor& b) -> Tensor {
    // Ensure tensors are contiguous before element-wise operation
    Tensor a_contiguous = a.is_contiguous() ? a : a.contiguous();
    Tensor b_contiguous = b.is_contiguous() ? b : b.contiguous();
    std::vector<Tensor> inputs = {a_contiguous, b_contiguous};
    return Dispatcher::dispatch("mul", inputs)[0];
}

auto div(const Tensor& a, const Tensor& b) -> Tensor {
    // Ensure tensors are contiguous before element-wise operation
    Tensor a_contiguous = a.is_contiguous() ? a : a.contiguous();
    Tensor b_contiguous = b.is_contiguous() ? b : b.contiguous();
    std::vector<Tensor> inputs = {a_contiguous, b_contiguous};
    return Dispatcher::dispatch("div", inputs)[0];
}

auto matmul(const Tensor& a, const Tensor& b) -> Tensor {
    std::vector<Tensor> inputs = {a, b};
    return Dispatcher::dispatch("matmul", inputs)[0];
}

auto bmm(const Tensor& a, const Tensor& b) -> Tensor {
    // Validate inputs are 3D
    if (a.shape().size() != 3 || b.shape().size() != 3) {
        throw std::runtime_error(
            "bmm requires 3D tensors, got shapes: [" +
            std::to_string(a.shape().size()) + "D] and [" +
            std::to_string(b.shape().size()) + "D]");
    }

    int64_t batch_size = a.shape()[0];
    int64_t n = a.shape()[1];
    int64_t m = a.shape()[2];

    if (b.shape()[0] != batch_size || b.shape()[1] != m) {
        throw std::runtime_error(
            "bmm dimension mismatch: expected b.shape=[" +
            std::to_string(batch_size) + ", " + std::to_string(m) + ", *], got [" +
            std::to_string(b.shape()[0]) + ", " + std::to_string(b.shape()[1]) + ", " +
            std::to_string(b.shape()[2]) + "]");
    }

    int64_t p = b.shape()[2];

    // Validate dtype support
    if (a.dtype() != DType::Float16 && a.dtype() != DType::Float32 && a.dtype() != DType::Float64) {
        throw std::runtime_error(
            "bmm currently only supports Float16, Float32, and Float64 dtypes, got: " +
            std::to_string(static_cast<int>(a.dtype())));
    }

    // Process each batch by extracting 2D slices and performing matmul
    // This preserves the computational graph for autograd operations
    std::vector<Tensor> batch_results;
    batch_results.reserve(batch_size);

    for (int64_t batch = 0; batch < batch_size; ++batch) {
        // Extract 2D slices from 3D tensors using slice and reshape operations
        // slice(input, dim, start, end) extracts input[start:end] along dimension dim
        // Then reshape removes the singleton dimension to get proper 2D tensors

        // Extract a_batch: slice to (1, n, m) then reshape to (n, m)
        Tensor a_slice = slice(a, 0, batch, batch + 1);  // Shape: (1, n, m)
        Tensor a_batch = reshape(a_slice, {n, m});        // Shape: (n, m)

        // Extract b_batch: slice to (1, m, p) then reshape to (m, p)
        Tensor b_slice = slice(b, 0, batch, batch + 1);  // Shape: (1, m, p)
        Tensor b_batch = reshape(b_slice, {m, p});        // Shape: (m, p)

        // Perform 2D matrix multiplication on this batch
        // This properly handles both contiguous and non-contiguous tensors
        // and maintains the autograd computational graph
        Tensor result_batch = matmul(a_batch, b_batch);  // Shape: (n, p)

        // Store the result for this batch
        batch_results.push_back(result_batch);
    }

    // Stack all batch results along dimension 0 to create the 3D output tensor
    // stack() concatenates tensors and adds a new dimension, creating shape (batch_size, n, p)
    // This maintains the computational graph connection for autograd
    return stack(batch_results, 0);
}

auto dot(const Tensor& a, const Tensor& b) -> Tensor {
    std::vector<Tensor> inputs = {a, b};
    return Dispatcher::dispatch("dot", inputs)[0];
}

auto pow(const Tensor& input, float exponent) -> Tensor {
    OpAttributes attrs;
    // Use scientific notation to preserve precision
    char exp_buf[32];
    snprintf(exp_buf, sizeof(exp_buf), "%.9e", exponent);
    attrs["exponent"] = std::string(exp_buf);
    std::vector<Tensor> inputs = {input};
    return Dispatcher::dispatch("pow", inputs, attrs)[0];
}

auto exp(const Tensor& input) -> Tensor {
    std::vector<Tensor> inputs = {input};
    return Dispatcher::dispatch("exp", inputs)[0];
}

auto log(const Tensor& input) -> Tensor {
    std::vector<Tensor> inputs = {input};
    return Dispatcher::dispatch("log", inputs)[0];
}

auto sqrt(const Tensor& input) -> Tensor {
    std::vector<Tensor> inputs = {input};
    return Dispatcher::dispatch("sqrt", inputs)[0];
}

auto sin(const Tensor& input) -> Tensor {
    std::vector<Tensor> inputs = {input};
    return Dispatcher::dispatch("sin", inputs)[0];
}

auto cos(const Tensor& input) -> Tensor {
    std::vector<Tensor> inputs = {input};
    return Dispatcher::dispatch("cos", inputs)[0];
}

auto tan(const Tensor& input) -> Tensor {
    std::vector<Tensor> inputs = {input};
    return Dispatcher::dispatch("tan", inputs)[0];
}

auto tanh(const Tensor& input) -> Tensor {
    std::vector<Tensor> inputs = {input};
    return Dispatcher::dispatch("tanh", inputs)[0];
}

auto abs(const Tensor& input) -> Tensor {
    std::vector<Tensor> inputs = {input};
    return Dispatcher::dispatch("abs", inputs)[0];
}

auto neg(const Tensor& input) -> Tensor {
    std::vector<Tensor> inputs = {input};
    return Dispatcher::dispatch("neg", inputs)[0];
}

auto reciprocal(const Tensor& input) -> Tensor {
    std::vector<Tensor> inputs = {input};
    return Dispatcher::dispatch("reciprocal", inputs)[0];
}

auto sign(const Tensor& input) -> Tensor {
    std::vector<Tensor> inputs = {input};
    return Dispatcher::dispatch("sign", inputs)[0];
}

auto floor(const Tensor& input) -> Tensor {
    std::vector<Tensor> inputs = {input};
    return Dispatcher::dispatch("floor", inputs)[0];
}

auto ceil(const Tensor& input) -> Tensor {
    std::vector<Tensor> inputs = {input};
    return Dispatcher::dispatch("ceil", inputs)[0];
}

auto round(const Tensor& input) -> Tensor {
    std::vector<Tensor> inputs = {input};
    return Dispatcher::dispatch("round", inputs)[0];
}

auto clamp(const Tensor& input, float min, float max) -> Tensor {
    OpAttributes attrs;
    // Use scientific notation to preserve precision
    char min_buf[32], max_buf[32];
    snprintf(min_buf, sizeof(min_buf), "%.9e", min);
    snprintf(max_buf, sizeof(max_buf), "%.9e", max);
    attrs["min"] = std::string(min_buf);
    attrs["max"] = std::string(max_buf);
    std::vector<Tensor> inputs = {input};
    return Dispatcher::dispatch("clamp", inputs, attrs)[0];
}

auto clamp_min(const Tensor& input, float min) -> Tensor {
    OpAttributes attrs;
    char min_buf[32];
    snprintf(min_buf, sizeof(min_buf), "%.9e", min);
    attrs["min"] = std::string(min_buf);
    std::vector<Tensor> inputs = {input};
    return Dispatcher::dispatch("clamp_min", inputs, attrs)[0];
}

auto clamp_max(const Tensor& input, float max) -> Tensor {
    OpAttributes attrs;
    char max_buf[32];
    snprintf(max_buf, sizeof(max_buf), "%.9e", max);
    attrs["max"] = std::string(max_buf);
    std::vector<Tensor> inputs = {input};
    return Dispatcher::dispatch("clamp_max", inputs, attrs)[0];
}

auto sinh(const Tensor& input) -> Tensor {
    std::vector<Tensor> inputs = {input};
    return Dispatcher::dispatch("sinh", inputs)[0];
}

auto cosh(const Tensor& input) -> Tensor {
    std::vector<Tensor> inputs = {input};
    return Dispatcher::dispatch("cosh", inputs)[0];
}

auto atan(const Tensor& input) -> Tensor {
    std::vector<Tensor> inputs = {input};
    return Dispatcher::dispatch("atan", inputs)[0];
}

auto asin(const Tensor& input) -> Tensor {
    std::vector<Tensor> inputs = {input};
    return Dispatcher::dispatch("asin", inputs)[0];
}

auto acos(const Tensor& input) -> Tensor {
    std::vector<Tensor> inputs = {input};
    return Dispatcher::dispatch("acos", inputs)[0];
}

// Comparison operations
auto eq(const Tensor& a, const Tensor& b) -> Tensor {
    // Ensure tensors are contiguous before element-wise comparison
    Tensor a_contiguous = a.is_contiguous() ? a : a.contiguous();
    Tensor b_contiguous = b.is_contiguous() ? b : b.contiguous();
    std::vector<Tensor> inputs = {a_contiguous, b_contiguous};
    return Dispatcher::dispatch("eq", inputs)[0];
}

auto ne(const Tensor& a, const Tensor& b) -> Tensor {
    // Ensure tensors are contiguous before element-wise comparison
    Tensor a_contiguous = a.is_contiguous() ? a : a.contiguous();
    Tensor b_contiguous = b.is_contiguous() ? b : b.contiguous();
    std::vector<Tensor> inputs = {a_contiguous, b_contiguous};
    return Dispatcher::dispatch("ne", inputs)[0];
}

auto lt(const Tensor& a, const Tensor& b) -> Tensor {
    // Ensure tensors are contiguous before element-wise comparison
    Tensor a_contiguous = a.is_contiguous() ? a : a.contiguous();
    Tensor b_contiguous = b.is_contiguous() ? b : b.contiguous();
    std::vector<Tensor> inputs = {a_contiguous, b_contiguous};
    return Dispatcher::dispatch("lt", inputs)[0];
}

auto le(const Tensor& a, const Tensor& b) -> Tensor {
    // Ensure tensors are contiguous before element-wise comparison
    Tensor a_contiguous = a.is_contiguous() ? a : a.contiguous();
    Tensor b_contiguous = b.is_contiguous() ? b : b.contiguous();
    std::vector<Tensor> inputs = {a_contiguous, b_contiguous};
    return Dispatcher::dispatch("le", inputs)[0];
}

auto gt(const Tensor& a, const Tensor& b) -> Tensor {
    // Ensure tensors are contiguous before element-wise comparison
    Tensor a_contiguous = a.is_contiguous() ? a : a.contiguous();
    Tensor b_contiguous = b.is_contiguous() ? b : b.contiguous();
    std::vector<Tensor> inputs = {a_contiguous, b_contiguous};
    return Dispatcher::dispatch("gt", inputs)[0];
}

auto ge(const Tensor& a, const Tensor& b) -> Tensor {
    // Ensure tensors are contiguous before element-wise comparison
    Tensor a_contiguous = a.is_contiguous() ? a : a.contiguous();
    Tensor b_contiguous = b.is_contiguous() ? b : b.contiguous();
    std::vector<Tensor> inputs = {a_contiguous, b_contiguous};
    return Dispatcher::dispatch("ge", inputs)[0];
}

// In-place operations
auto add_(Tensor& self, const Tensor& other) -> Tensor& {
    // Ensure self is contiguous for in-place modification
    if (!self.is_contiguous()) {
        throw std::runtime_error("In-place add requires contiguous tensor");
    }

    Tensor other_contiguous = other.is_contiguous() ? other : other.contiguous();
    std::vector<Tensor> inputs = {self, other_contiguous};

    // Dispatch to backend in-place operation
    auto result = Dispatcher::dispatch("add_inplace", inputs);

    // Result should be same tensor modified in-place
    // Copy result data back to self if backend created new tensor
    if (result[0].data<float>() != self.data<float>()) {
        self = result[0];
    }

    return self;
}

auto mul_(Tensor& self, const Tensor& other) -> Tensor& {
    // Ensure self is contiguous for in-place modification
    if (!self.is_contiguous()) {
        throw std::runtime_error("In-place mul requires contiguous tensor");
    }

    Tensor other_contiguous = other.is_contiguous() ? other : other.contiguous();
    std::vector<Tensor> inputs = {self, other_contiguous};

    // Dispatch to backend in-place operation
    auto result = Dispatcher::dispatch("mul_inplace", inputs);

    // Result should be same tensor modified in-place
    if (result[0].data<float>() != self.data<float>()) {
        self = result[0];
    }

    return self;
}

auto sub_(Tensor& self, const Tensor& other) -> Tensor& {
    // Ensure self is contiguous for in-place modification
    if (!self.is_contiguous()) {
        throw std::runtime_error("In-place sub requires contiguous tensor");
    }

    Tensor other_contiguous = other.is_contiguous() ? other : other.contiguous();
    std::vector<Tensor> inputs = {self, other_contiguous};

    // Dispatch to backend in-place operation
    auto result = Dispatcher::dispatch("sub_inplace", inputs);

    // Result should be same tensor modified in-place
    if (result[0].data<float>() != self.data<float>()) {
        self = result[0];
    }

    return self;
}

auto div_(Tensor& self, const Tensor& other) -> Tensor& {
    // Ensure self is contiguous for in-place modification
    if (!self.is_contiguous()) {
        throw std::runtime_error("In-place div requires contiguous tensor");
    }

    Tensor other_contiguous = other.is_contiguous() ? other : other.contiguous();
    std::vector<Tensor> inputs = {self, other_contiguous};

    // Dispatch to backend in-place operation
    auto result = Dispatcher::dispatch("div_inplace", inputs);

    // Result should be same tensor modified in-place
    if (result[0].data<float>() != self.data<float>()) {
        self = result[0];
    }

    return self;
}

} // namespace tenzor
