#include "tenzor/ops/math.hpp"
#include "tenzor/backend/fast_dispatch.hpp"
#include "tenzor/ops/op_id.hpp"
#include "tenzor/ops/indexing.hpp"
#include "tenzor/ops/transform.hpp"
#include "tenzor/ops/creation.hpp"
#include <array>
#include <cstring>
#include <stdexcept>
#include <iostream>
#include <vector>
#include <chrono>
#ifdef _OPENMP
#include <omp.h>
#endif


namespace tenzor {

// Validate that two shapes are broadcast-compatible
static void validate_broadcast_shapes(const char* op_name,
                                       std::span<const int64_t> a_shape,
                                       std::span<const int64_t> b_shape) {
    // Walk from trailing dimensions
    auto a_it = a_shape.rbegin();
    auto b_it = b_shape.rbegin();
    for (; a_it != a_shape.rend() && b_it != b_shape.rend(); ++a_it, ++b_it) {
        if (*a_it != *b_it && *a_it != 1 && *b_it != 1) {
            std::string a_str = "[", b_str = "[";
            for (size_t i = 0; i < a_shape.size(); ++i) {
                if (i) a_str += ",";
                a_str += std::to_string(a_shape[i]);
            }
            for (size_t i = 0; i < b_shape.size(); ++i) {
                if (i) b_str += ",";
                b_str += std::to_string(b_shape[i]);
            }
            throw std::invalid_argument(
                std::string(op_name) + ": shapes " + a_str + "] and " + b_str +
                "] are not broadcast-compatible");
        }
    }
}

// Math operation implementations - dispatched to backend kernels

auto add(const Tensor& a, const Tensor& b) -> Tensor {
    // Validate tensors are initialized
    if (!a.impl() || !b.impl()) {
        throw std::runtime_error("Cannot add uninitialized tensors");
    }
    // Validate dtype compatibility
    if (a.dtype() != b.dtype()) {
        throw std::invalid_argument("add: dtype mismatch (" +
            std::string(dtype_name(a.dtype())) + " vs " +
            std::string(dtype_name(b.dtype())) + ")");
    }
    validate_broadcast_shapes("add", a.shape(), b.shape());
    // Ensure tensors are contiguous before element-wise operation
    // Permute and reshape can create non-contiguous views
    Tensor a_contiguous = a.is_contiguous() ? a : a.contiguous();
    Tensor b_contiguous = b.is_contiguous() ? b : b.contiguous();
    std::vector<Tensor> inputs = {a_contiguous, b_contiguous};
    return dispatch<OpId::Add>(inputs)[0];
}

auto sub(const Tensor& a, const Tensor& b) -> Tensor {
    // Validate dtype compatibility
    if (a.dtype() != b.dtype()) {
        throw std::invalid_argument("sub: dtype mismatch (" +
            std::string(dtype_name(a.dtype())) + " vs " +
            std::string(dtype_name(b.dtype())) + ")");
    }
    validate_broadcast_shapes("sub", a.shape(), b.shape());
    // Ensure tensors are contiguous before element-wise operation
    Tensor a_contiguous = a.is_contiguous() ? a : a.contiguous();
    Tensor b_contiguous = b.is_contiguous() ? b : b.contiguous();
    std::vector<Tensor> inputs = {a_contiguous, b_contiguous};
    return dispatch<OpId::Sub>(inputs)[0];
}

auto mul(const Tensor& a, const Tensor& b) -> Tensor {
    // Validate dtype compatibility
    if (a.dtype() != b.dtype()) {
        throw std::invalid_argument("mul: dtype mismatch (" +
            std::string(dtype_name(a.dtype())) + " vs " +
            std::string(dtype_name(b.dtype())) + ")");
    }
    validate_broadcast_shapes("mul", a.shape(), b.shape());
    // Ensure tensors are contiguous before element-wise operation
    Tensor a_contiguous = a.is_contiguous() ? a : a.contiguous();
    Tensor b_contiguous = b.is_contiguous() ? b : b.contiguous();
    std::vector<Tensor> inputs = {a_contiguous, b_contiguous};
    return dispatch<OpId::Mul>(inputs)[0];
}

auto div(const Tensor& a, const Tensor& b) -> Tensor {
    // Validate dtype compatibility
    if (a.dtype() != b.dtype()) {
        throw std::invalid_argument("div: dtype mismatch (" +
            std::string(dtype_name(a.dtype())) + " vs " +
            std::string(dtype_name(b.dtype())) + ")");
    }
    validate_broadcast_shapes("div", a.shape(), b.shape());
    // Ensure tensors are contiguous before element-wise operation
    Tensor a_contiguous = a.is_contiguous() ? a : a.contiguous();
    Tensor b_contiguous = b.is_contiguous() ? b : b.contiguous();
    std::vector<Tensor> inputs = {a_contiguous, b_contiguous};
    return dispatch<OpId::Div>(inputs)[0];
}

// Scalar arithmetic overloads — avoid creating temporary scalar tensors.
// Creates a size-{1} scalar tensor once and delegates to the tensor overload,
// but backends can optimize this path internally.
auto add(const Tensor& a, double scalar) -> Tensor {
    if (!a.impl()) {
        throw std::runtime_error("Cannot add to uninitialized tensor");
    }
    auto scalar_tensor = full({1}, scalar, a.dtype(), a.device());
    std::vector<Tensor> inputs = {a.is_contiguous() ? a : a.contiguous(), scalar_tensor};
    return dispatch<OpId::Add>(inputs)[0];
}

auto sub(const Tensor& a, double scalar) -> Tensor {
    if (!a.impl()) {
        throw std::runtime_error("Cannot subtract from uninitialized tensor");
    }
    auto scalar_tensor = full({1}, scalar, a.dtype(), a.device());
    std::vector<Tensor> inputs = {a.is_contiguous() ? a : a.contiguous(), scalar_tensor};
    return dispatch<OpId::Sub>(inputs)[0];
}

auto mul(const Tensor& a, double scalar) -> Tensor {
    if (!a.impl()) {
        throw std::runtime_error("Cannot multiply uninitialized tensor");
    }
    auto scalar_tensor = full({1}, scalar, a.dtype(), a.device());
    std::vector<Tensor> inputs = {a.is_contiguous() ? a : a.contiguous(), scalar_tensor};
    return dispatch<OpId::Mul>(inputs)[0];
}

auto div(const Tensor& a, double scalar) -> Tensor {
    if (!a.impl()) {
        throw std::runtime_error("Cannot divide uninitialized tensor");
    }
    auto scalar_tensor = full({1}, scalar, a.dtype(), a.device());
    std::vector<Tensor> inputs = {a.is_contiguous() ? a : a.contiguous(), scalar_tensor};
    return dispatch<OpId::Div>(inputs)[0];
}

auto matmul(const Tensor& a, const Tensor& b) -> Tensor {
    // Validate dtype compatibility
    if (a.dtype() != b.dtype()) {
        throw std::invalid_argument("matmul: dtype mismatch (" +
            std::string(dtype_name(a.dtype())) + " vs " +
            std::string(dtype_name(b.dtype())) + ")");
    }
    // Handle batched matrix multiplication (3D+ tensors)
    if (a.shape().size() >= 3 && b.shape().size() >= 3) {
        // Both are batched - use bmm
        return bmm(a, b);
    }
    // Standard 2D matmul - use dispatch_single to avoid vector allocation overhead
    // This saves ~0.5-2us per call, significant for small matrices
    std::array<Tensor, 2> inputs = {a, b};
    return dispatch_single<OpId::MatMul>(inputs);
}

auto bmm(const Tensor& a, const Tensor& b) -> Tensor {
    // Validate dtype compatibility
    if (a.dtype() != b.dtype()) {
        throw std::invalid_argument("bmm: dtype mismatch (" +
            std::string(dtype_name(a.dtype())) + " vs " +
            std::string(dtype_name(b.dtype())) + ")");
    }
    // Validate inputs are 3D
    if (a.shape().size() != 3 || b.shape().size() != 3) {
        throw std::runtime_error(
            "bmm requires 3D tensors, got shapes: [" +
            std::to_string(a.shape().size()) + "D] and [" +
            std::to_string(b.shape().size()) + "D]");
    }

    // Validate batch sizes and inner dimensions match
    int64_t batch_size = a.shape()[0];
    int64_t M = a.shape()[1];
    int64_t K = a.shape()[2];
    int64_t N = b.shape()[2];

    if (b.shape()[0] != batch_size || b.shape()[1] != K) {
        throw std::runtime_error(
            "bmm dimension mismatch: expected b.shape=[" +
            std::to_string(batch_size) + ", " + std::to_string(K) + ", *], got [" +
            std::to_string(b.shape()[0]) + ", " + std::to_string(b.shape()[1]) + ", " +
            std::to_string(b.shape()[2]) + "]");
    }

    // Dispatch to registered bmm kernel (CPU uses MKL, others use slice/matmul)
    // Use dispatch_single to avoid vector allocation overhead
    std::array<Tensor, 2> inputs = {a, b};
    return dispatch_single<OpId::Bmm>(inputs);
}

auto dot(const Tensor& a, const Tensor& b) -> Tensor {
    // Validate dtype compatibility
    if (a.dtype() != b.dtype()) {
        throw std::invalid_argument("dot: dtype mismatch (" +
            std::string(dtype_name(a.dtype())) + " vs " +
            std::string(dtype_name(b.dtype())) + ")");
    }
    std::vector<Tensor> inputs = {a, b};
    return dispatch<OpId::Dot>(inputs)[0];
}

auto pow(const Tensor& input, float exponent) -> Tensor {
    OpAttributes attrs;
    // Use full float precision (9 significant digits for IEEE 754 float)
    char exp_buf[32];
    snprintf(exp_buf, sizeof(exp_buf), "%.9g", static_cast<double>(exponent));
    attrs["exponent"] = std::string(exp_buf);
    std::vector<Tensor> inputs = {input};
    return dispatch(OpId::Pow, inputs, attrs)[0];
}

auto exp(const Tensor& input) -> Tensor {
    std::vector<Tensor> inputs = {input};
    return dispatch<OpId::Exp>(inputs)[0];
}

auto log(const Tensor& input) -> Tensor {
    std::vector<Tensor> inputs = {input};
    return dispatch<OpId::Log>(inputs)[0];
}

auto sqrt(const Tensor& input) -> Tensor {
    std::vector<Tensor> inputs = {input};
    return dispatch<OpId::Sqrt>(inputs)[0];
}

auto sin(const Tensor& input) -> Tensor {
    std::vector<Tensor> inputs = {input};
    return dispatch<OpId::Sin>(inputs)[0];
}

auto cos(const Tensor& input) -> Tensor {
    std::vector<Tensor> inputs = {input};
    return dispatch<OpId::Cos>(inputs)[0];
}

auto tan(const Tensor& input) -> Tensor {
    std::vector<Tensor> inputs = {input};
    return dispatch<OpId::Tan>(inputs)[0];
}

auto tanh(const Tensor& input) -> Tensor {
    std::vector<Tensor> inputs = {input};
    return dispatch<OpId::Tanh>(inputs)[0];
}

auto abs(const Tensor& input) -> Tensor {
    std::vector<Tensor> inputs = {input};
    return dispatch<OpId::Abs>(inputs)[0];
}

auto neg(const Tensor& input) -> Tensor {
    std::vector<Tensor> inputs = {input};
    return dispatch<OpId::Neg>(inputs)[0];
}

auto reciprocal(const Tensor& input) -> Tensor {
    std::vector<Tensor> inputs = {input};
    return dispatch<OpId::Reciprocal>(inputs)[0];
}

auto sign(const Tensor& input) -> Tensor {
    std::vector<Tensor> inputs = {input};
    return dispatch<OpId::Sign>(inputs)[0];
}

auto sigmoid(const Tensor& input) -> Tensor {
    std::vector<Tensor> inputs = {input};
    return dispatch<OpId::Sigmoid>(inputs)[0];
}

auto minimum(const Tensor& a, const Tensor& b) -> Tensor {
    // minimum(a, b) = where(a <= b, a, b)
    Tensor a_contiguous = a.is_contiguous() ? a : a.contiguous();
    Tensor b_contiguous = b.is_contiguous() ? b : b.contiguous();
    auto mask = le(a_contiguous, b_contiguous);
    return where(mask, a_contiguous, b_contiguous);
}

auto maximum(const Tensor& a, const Tensor& b) -> Tensor {
    // maximum(a, b) = where(a >= b, a, b)
    Tensor a_contiguous = a.is_contiguous() ? a : a.contiguous();
    Tensor b_contiguous = b.is_contiguous() ? b : b.contiguous();
    auto mask = ge(a_contiguous, b_contiguous);
    return where(mask, a_contiguous, b_contiguous);
}

auto floor(const Tensor& input) -> Tensor {
    std::vector<Tensor> inputs = {input};
    return dispatch<OpId::Floor>(inputs)[0];
}

auto ceil(const Tensor& input) -> Tensor {
    std::vector<Tensor> inputs = {input};
    return dispatch<OpId::Ceil>(inputs)[0];
}

auto round(const Tensor& input) -> Tensor {
    std::vector<Tensor> inputs = {input};
    return dispatch<OpId::Round>(inputs)[0];
}

auto clamp(const Tensor& input, float min, float max) -> Tensor {
    OpAttributes attrs;
    char min_buf[32], max_buf[32];
    snprintf(min_buf, sizeof(min_buf), "%.9g", static_cast<double>(min));
    snprintf(max_buf, sizeof(max_buf), "%.9g", static_cast<double>(max));
    attrs["min"] = std::string(min_buf);
    attrs["max"] = std::string(max_buf);
    std::vector<Tensor> inputs = {input};
    return dispatch(OpId::Clamp, inputs, attrs)[0];
}

auto clamp_min(const Tensor& input, float min) -> Tensor {
    OpAttributes attrs;
    char min_buf[32];
    snprintf(min_buf, sizeof(min_buf), "%.9g", static_cast<double>(min));
    attrs["min"] = std::string(min_buf);
    std::vector<Tensor> inputs = {input};
    return dispatch(OpId::ClampMin, inputs, attrs)[0];
}

auto clamp_max(const Tensor& input, float max) -> Tensor {
    OpAttributes attrs;
    char max_buf[32];
    snprintf(max_buf, sizeof(max_buf), "%.9g", static_cast<double>(max));
    attrs["max"] = std::string(max_buf);
    std::vector<Tensor> inputs = {input};
    return dispatch(OpId::ClampMax, inputs, attrs)[0];
}

auto sinh(const Tensor& input) -> Tensor {
    std::vector<Tensor> inputs = {input};
    return dispatch<OpId::Sinh>(inputs)[0];
}

auto cosh(const Tensor& input) -> Tensor {
    std::vector<Tensor> inputs = {input};
    return dispatch<OpId::Cosh>(inputs)[0];
}

auto atan(const Tensor& input) -> Tensor {
    std::vector<Tensor> inputs = {input};
    return dispatch<OpId::Atan>(inputs)[0];
}

auto asin(const Tensor& input) -> Tensor {
    std::vector<Tensor> inputs = {input};
    return dispatch<OpId::Asin>(inputs)[0];
}

auto acos(const Tensor& input) -> Tensor {
    std::vector<Tensor> inputs = {input};
    return dispatch<OpId::Acos>(inputs)[0];
}

// Comparison operations
auto eq(const Tensor& a, const Tensor& b) -> Tensor {
    // Ensure tensors are contiguous before element-wise comparison
    Tensor a_contiguous = a.is_contiguous() ? a : a.contiguous();
    Tensor b_contiguous = b.is_contiguous() ? b : b.contiguous();
    std::vector<Tensor> inputs = {a_contiguous, b_contiguous};
    return dispatch<OpId::Eq>(inputs)[0];
}

auto ne(const Tensor& a, const Tensor& b) -> Tensor {
    // Ensure tensors are contiguous before element-wise comparison
    Tensor a_contiguous = a.is_contiguous() ? a : a.contiguous();
    Tensor b_contiguous = b.is_contiguous() ? b : b.contiguous();
    std::vector<Tensor> inputs = {a_contiguous, b_contiguous};
    return dispatch<OpId::Ne>(inputs)[0];
}

auto lt(const Tensor& a, const Tensor& b) -> Tensor {
    // Ensure tensors are contiguous before element-wise comparison
    Tensor a_contiguous = a.is_contiguous() ? a : a.contiguous();
    Tensor b_contiguous = b.is_contiguous() ? b : b.contiguous();
    std::vector<Tensor> inputs = {a_contiguous, b_contiguous};
    return dispatch<OpId::Lt>(inputs)[0];
}

auto le(const Tensor& a, const Tensor& b) -> Tensor {
    // Ensure tensors are contiguous before element-wise comparison
    Tensor a_contiguous = a.is_contiguous() ? a : a.contiguous();
    Tensor b_contiguous = b.is_contiguous() ? b : b.contiguous();
    std::vector<Tensor> inputs = {a_contiguous, b_contiguous};
    return dispatch<OpId::Le>(inputs)[0];
}

auto gt(const Tensor& a, const Tensor& b) -> Tensor {
    // Ensure tensors are contiguous before element-wise comparison
    Tensor a_contiguous = a.is_contiguous() ? a : a.contiguous();
    Tensor b_contiguous = b.is_contiguous() ? b : b.contiguous();
    std::vector<Tensor> inputs = {a_contiguous, b_contiguous};
    return dispatch<OpId::Gt>(inputs)[0];
}

auto ge(const Tensor& a, const Tensor& b) -> Tensor {
    // Ensure tensors are contiguous before element-wise comparison
    Tensor a_contiguous = a.is_contiguous() ? a : a.contiguous();
    Tensor b_contiguous = b.is_contiguous() ? b : b.contiguous();
    std::vector<Tensor> inputs = {a_contiguous, b_contiguous};
    return dispatch<OpId::Ge>(inputs)[0];
}

// Helper to validate in-place operations don't break autograd
static void check_inplace_autograd(const Tensor& self) {
    if (self.requires_grad()) {
        throw std::runtime_error(
            "In-place operation not allowed on a tensor that requires grad. "
            "Use the non-inplace version instead.");
    }
}

// In-place operations
auto add_(Tensor& self, const Tensor& other) -> Tensor& {
    check_inplace_autograd(self);
    // Validate dtype compatibility
    if (self.dtype() != other.dtype()) {
        throw std::invalid_argument("add_: dtype mismatch (" +
            std::string(dtype_name(self.dtype())) + " vs " +
            std::string(dtype_name(other.dtype())) + ")");
    }
    // Ensure self is contiguous for in-place modification
    if (!self.is_contiguous()) {
        throw std::runtime_error("In-place add requires contiguous tensor");
    }

    Tensor other_contiguous = other.is_contiguous() ? other : other.contiguous();
    std::array<Tensor, 1> others = {other_contiguous};

    // Dispatch directly to inplace kernel (no const_cast needed)
    dispatch_inplace(OpId::AddInplace, self, others);

    return self;
}

auto mul_(Tensor& self, const Tensor& other) -> Tensor& {
    check_inplace_autograd(self);
    // Validate dtype compatibility
    if (self.dtype() != other.dtype()) {
        throw std::invalid_argument("mul_: dtype mismatch (" +
            std::string(dtype_name(self.dtype())) + " vs " +
            std::string(dtype_name(other.dtype())) + ")");
    }
    // Ensure self is contiguous for in-place modification
    if (!self.is_contiguous()) {
        throw std::runtime_error("In-place mul requires contiguous tensor");
    }

    Tensor other_contiguous = other.is_contiguous() ? other : other.contiguous();
    std::array<Tensor, 1> others = {other_contiguous};

    dispatch_inplace(OpId::MulInplace, self, others);

    return self;
}

auto sub_(Tensor& self, const Tensor& other) -> Tensor& {
    check_inplace_autograd(self);
    // Validate dtype compatibility
    if (self.dtype() != other.dtype()) {
        throw std::invalid_argument("sub_: dtype mismatch (" +
            std::string(dtype_name(self.dtype())) + " vs " +
            std::string(dtype_name(other.dtype())) + ")");
    }
    // Ensure self is contiguous for in-place modification
    if (!self.is_contiguous()) {
        throw std::runtime_error("In-place sub requires contiguous tensor");
    }

    Tensor other_contiguous = other.is_contiguous() ? other : other.contiguous();
    std::array<Tensor, 1> others = {other_contiguous};

    dispatch_inplace(OpId::SubInplace, self, others);

    return self;
}

auto div_(Tensor& self, const Tensor& other) -> Tensor& {
    check_inplace_autograd(self);
    // Validate dtype compatibility
    if (self.dtype() != other.dtype()) {
        throw std::invalid_argument("div_: dtype mismatch (" +
            std::string(dtype_name(self.dtype())) + " vs " +
            std::string(dtype_name(other.dtype())) + ")");
    }
    // Ensure self is contiguous for in-place modification
    if (!self.is_contiguous()) {
        throw std::runtime_error("In-place div requires contiguous tensor");
    }

    Tensor other_contiguous = other.is_contiguous() ? other : other.contiguous();
    std::array<Tensor, 1> others = {other_contiguous};

    dispatch_inplace(OpId::DivInplace, self, others);

    return self;
}

} // namespace tenzor
