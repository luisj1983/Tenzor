#include "tenzor/ops/math.hpp"
#include "tenzor/backend/fast_dispatch.hpp"
#include "tenzor/ops/op_id.hpp"
#include "tenzor/ops/indexing.hpp"
#include "tenzor/ops/transform.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/type_promotion.hpp"
#include <algorithm>
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
    // Auto-promote dtypes if mismatched
    auto [ap, bp] = promote_inputs(a, b);
    validate_broadcast_shapes("add", ap.shape(), bp.shape());
    Tensor a_contiguous = ap.is_contiguous() ? ap : ap.contiguous();
    Tensor b_contiguous = bp.is_contiguous() ? bp : bp.contiguous();
    std::vector<Tensor> inputs = {a_contiguous, b_contiguous};
    return dispatch<OpId::Add>(inputs)[0];
}

auto sub(const Tensor& a, const Tensor& b) -> Tensor {
    auto [ap, bp] = promote_inputs(a, b);
    validate_broadcast_shapes("sub", ap.shape(), bp.shape());
    Tensor a_contiguous = ap.is_contiguous() ? ap : ap.contiguous();
    Tensor b_contiguous = bp.is_contiguous() ? bp : bp.contiguous();
    std::vector<Tensor> inputs = {a_contiguous, b_contiguous};
    return dispatch<OpId::Sub>(inputs)[0];
}

auto mul(const Tensor& a, const Tensor& b) -> Tensor {
    auto [ap, bp] = promote_inputs(a, b);
    validate_broadcast_shapes("mul", ap.shape(), bp.shape());
    Tensor a_contiguous = ap.is_contiguous() ? ap : ap.contiguous();
    Tensor b_contiguous = bp.is_contiguous() ? bp : bp.contiguous();
    std::vector<Tensor> inputs = {a_contiguous, b_contiguous};
    return dispatch<OpId::Mul>(inputs)[0];
}

auto div(const Tensor& a, const Tensor& b) -> Tensor {
    auto [ap, bp] = promote_inputs(a, b);
    validate_broadcast_shapes("div", ap.shape(), bp.shape());
    Tensor a_contiguous = ap.is_contiguous() ? ap : ap.contiguous();
    Tensor b_contiguous = bp.is_contiguous() ? bp : bp.contiguous();
    std::vector<Tensor> inputs = {a_contiguous, b_contiguous};
    return dispatch<OpId::Div>(inputs)[0];
}

// Scalar arithmetic overloads — avoid creating temporary scalar tensors.
// Creates a size-{1} scalar tensor once and delegates to the tensor overload,
// but backends can optimize this path internally.
// Helper: format scalar with full double precision for OpAttributes
static auto scalar_to_string(double scalar) -> std::string {
    char buf[32];
    snprintf(buf, sizeof(buf), "%.17g", scalar);
    return std::string(buf);
}

auto add(const Tensor& a, double scalar) -> Tensor {
    if (!a.impl()) {
        throw std::runtime_error("Cannot add to uninitialized tensor");
    }
    auto scalar_tensor = full({1}, scalar, a.dtype(), a.device());
    std::vector<Tensor> inputs = {a.is_contiguous() ? a : a.contiguous(), scalar_tensor};
    OpAttributes attrs{{"scalar_b", scalar_to_string(scalar)}};
    return dispatch(OpId::Add, inputs, attrs)[0];
}

auto sub(const Tensor& a, double scalar) -> Tensor {
    if (!a.impl()) {
        throw std::runtime_error("Cannot subtract from uninitialized tensor");
    }
    auto scalar_tensor = full({1}, scalar, a.dtype(), a.device());
    std::vector<Tensor> inputs = {a.is_contiguous() ? a : a.contiguous(), scalar_tensor};
    OpAttributes attrs{{"scalar_b", scalar_to_string(scalar)}};
    return dispatch(OpId::Sub, inputs, attrs)[0];
}

auto mul(const Tensor& a, double scalar) -> Tensor {
    if (!a.impl()) {
        throw std::runtime_error("Cannot multiply uninitialized tensor");
    }
    auto scalar_tensor = full({1}, scalar, a.dtype(), a.device());
    std::vector<Tensor> inputs = {a.is_contiguous() ? a : a.contiguous(), scalar_tensor};
    OpAttributes attrs{{"scalar_b", scalar_to_string(scalar)}};
    return dispatch(OpId::Mul, inputs, attrs)[0];
}

auto div(const Tensor& a, double scalar) -> Tensor {
    if (!a.impl()) {
        throw std::runtime_error("Cannot divide uninitialized tensor");
    }
    auto scalar_tensor = full({1}, scalar, a.dtype(), a.device());
    std::vector<Tensor> inputs = {a.is_contiguous() ? a : a.contiguous(), scalar_tensor};
    OpAttributes attrs{{"scalar_b", scalar_to_string(scalar)}};
    return dispatch(OpId::Div, inputs, attrs)[0];
}

auto matmul(const Tensor& a, const Tensor& b) -> Tensor {
    // Auto-promote dtypes if mismatched
    auto [ap, bp] = promote_inputs(a, b);
    // Handle batched matrix multiplication (3D+ tensors)
    if (ap.shape().size() >= 3 || bp.shape().size() >= 3) {
        // Broadcast 2D input to 3D for mixed 2D/3D case
        auto ac = ap.is_contiguous() ? ap : ap.contiguous();
        auto bc = bp.is_contiguous() ? bp : bp.contiguous();
        if (ac.shape().size() == 2 && bc.shape().size() >= 3) {
            ac = expand(ac.unsqueeze(0), {bc.shape()[0], ac.shape()[0], ac.shape()[1]});
            auto result = bmm(ac, bc);
            return result;
        }
        if (bc.shape().size() == 2 && ac.shape().size() >= 3) {
            bc = expand(bc.unsqueeze(0), {ac.shape()[0], bc.shape()[0], bc.shape()[1]});
            auto result = bmm(ac, bc);
            return result;
        }
        return bmm(ac, bc);
    }
    auto ac = ap.is_contiguous() ? ap : ap.contiguous();
    auto bc = bp.is_contiguous() ? bp : bp.contiguous();
    std::array<Tensor, 2> inputs = {ac, bc};
    return dispatch_single<OpId::MatMul>(inputs);
}

auto bmm(const Tensor& a, const Tensor& b) -> Tensor {
    auto [ap, bp] = promote_inputs(a, b);
    // Validate inputs are 3D
    if (ap.shape().size() != 3 || bp.shape().size() != 3) {
        throw std::runtime_error(
            "bmm requires 3D tensors, got shapes: [" +
            std::to_string(ap.shape().size()) + "D] and [" +
            std::to_string(bp.shape().size()) + "D]");
    }

    // Validate batch sizes and inner dimensions match
    int64_t batch_size = ap.shape()[0];
    int64_t K = ap.shape()[2];

    if (bp.shape()[0] != batch_size || bp.shape()[1] != K) {
        throw std::runtime_error(
            "bmm dimension mismatch: expected b.shape=[" +
            std::to_string(batch_size) + ", " + std::to_string(K) + ", *], got [" +
            std::to_string(bp.shape()[0]) + ", " + std::to_string(bp.shape()[1]) + ", " +
            std::to_string(bp.shape()[2]) + "]");
    }

    auto ac = ap.is_contiguous() ? ap : ap.contiguous();
    auto bc = bp.is_contiguous() ? bp : bp.contiguous();
    std::array<Tensor, 2> inputs = {ac, bc};
    return dispatch_single<OpId::Bmm>(inputs);
}

auto dot(const Tensor& a, const Tensor& b) -> Tensor {
    auto [ap, bp] = promote_inputs(a, b);
    auto ac = ap.is_contiguous() ? ap : ap.contiguous();
    auto bc = bp.is_contiguous() ? bp : bp.contiguous();
    std::vector<Tensor> inputs = {ac, bc};
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
    auto [ap, bp] = promote_inputs(a, b);
    validate_broadcast_shapes("minimum", ap.shape(), bp.shape());
    Tensor a_cont = ap.is_contiguous() ? ap : ap.contiguous();
    Tensor b_cont = bp.is_contiguous() ? bp : bp.contiguous();
    std::vector<Tensor> inputs = {a_cont, b_cont};
    return dispatch<OpId::Minimum>(inputs)[0];
}

auto maximum(const Tensor& a, const Tensor& b) -> Tensor {
    auto [ap, bp] = promote_inputs(a, b);
    validate_broadcast_shapes("maximum", ap.shape(), bp.shape());
    Tensor a_cont = ap.is_contiguous() ? ap : ap.contiguous();
    Tensor b_cont = bp.is_contiguous() ? bp : bp.contiguous();
    std::vector<Tensor> inputs = {a_cont, b_cont};
    return dispatch<OpId::Maximum>(inputs)[0];
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

// Comparison operations — auto-promote dtypes before comparing
auto eq(const Tensor& a, const Tensor& b) -> Tensor {
    auto [ap, bp] = promote_inputs(a, b);
    validate_broadcast_shapes("eq", ap.shape(), bp.shape());
    Tensor a_contiguous = ap.is_contiguous() ? ap : ap.contiguous();
    Tensor b_contiguous = bp.is_contiguous() ? bp : bp.contiguous();
    std::vector<Tensor> inputs = {a_contiguous, b_contiguous};
    return dispatch<OpId::Eq>(inputs)[0];
}

auto ne(const Tensor& a, const Tensor& b) -> Tensor {
    auto [ap, bp] = promote_inputs(a, b);
    validate_broadcast_shapes("ne", ap.shape(), bp.shape());
    Tensor a_contiguous = ap.is_contiguous() ? ap : ap.contiguous();
    Tensor b_contiguous = bp.is_contiguous() ? bp : bp.contiguous();
    std::vector<Tensor> inputs = {a_contiguous, b_contiguous};
    return dispatch<OpId::Ne>(inputs)[0];
}

auto lt(const Tensor& a, const Tensor& b) -> Tensor {
    auto [ap, bp] = promote_inputs(a, b);
    validate_broadcast_shapes("lt", ap.shape(), bp.shape());
    Tensor a_contiguous = ap.is_contiguous() ? ap : ap.contiguous();
    Tensor b_contiguous = bp.is_contiguous() ? bp : bp.contiguous();
    std::vector<Tensor> inputs = {a_contiguous, b_contiguous};
    return dispatch<OpId::Lt>(inputs)[0];
}

auto le(const Tensor& a, const Tensor& b) -> Tensor {
    auto [ap, bp] = promote_inputs(a, b);
    validate_broadcast_shapes("le", ap.shape(), bp.shape());
    Tensor a_contiguous = ap.is_contiguous() ? ap : ap.contiguous();
    Tensor b_contiguous = bp.is_contiguous() ? bp : bp.contiguous();
    std::vector<Tensor> inputs = {a_contiguous, b_contiguous};
    return dispatch<OpId::Le>(inputs)[0];
}

auto gt(const Tensor& a, const Tensor& b) -> Tensor {
    auto [ap, bp] = promote_inputs(a, b);
    validate_broadcast_shapes("gt", ap.shape(), bp.shape());
    Tensor a_contiguous = ap.is_contiguous() ? ap : ap.contiguous();
    Tensor b_contiguous = bp.is_contiguous() ? bp : bp.contiguous();
    std::vector<Tensor> inputs = {a_contiguous, b_contiguous};
    return dispatch<OpId::Gt>(inputs)[0];
}

auto ge(const Tensor& a, const Tensor& b) -> Tensor {
    auto [ap, bp] = promote_inputs(a, b);
    validate_broadcast_shapes("ge", ap.shape(), bp.shape());
    Tensor a_contiguous = ap.is_contiguous() ? ap : ap.contiguous();
    Tensor b_contiguous = bp.is_contiguous() ? bp : bp.contiguous();
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

// In-place operations — convert other to self's dtype (in-place can't change self's type)
auto add_(Tensor& self, const Tensor& other) -> Tensor& {
    check_inplace_autograd(self);
    validate_broadcast_shapes("add_", self.shape(), other.shape());
    if (!self.is_contiguous()) {
        throw std::runtime_error("In-place add requires contiguous tensor");
    }
    Tensor other_cast = (other.dtype() != self.dtype()) ? other.to(self.dtype()) : other;
    Tensor other_contiguous = other_cast.is_contiguous() ? other_cast : other_cast.contiguous();
    std::array<Tensor, 1> others = {other_contiguous};
    dispatch_inplace(OpId::AddInplace, self, others);
    return self;
}

auto mul_(Tensor& self, const Tensor& other) -> Tensor& {
    check_inplace_autograd(self);
    validate_broadcast_shapes("mul_", self.shape(), other.shape());
    if (!self.is_contiguous()) {
        throw std::runtime_error("In-place mul requires contiguous tensor");
    }
    Tensor other_cast = (other.dtype() != self.dtype()) ? other.to(self.dtype()) : other;
    Tensor other_contiguous = other_cast.is_contiguous() ? other_cast : other_cast.contiguous();
    std::array<Tensor, 1> others = {other_contiguous};
    dispatch_inplace(OpId::MulInplace, self, others);
    return self;
}

auto sub_(Tensor& self, const Tensor& other) -> Tensor& {
    check_inplace_autograd(self);
    validate_broadcast_shapes("sub_", self.shape(), other.shape());
    if (!self.is_contiguous()) {
        throw std::runtime_error("In-place sub requires contiguous tensor");
    }
    Tensor other_cast = (other.dtype() != self.dtype()) ? other.to(self.dtype()) : other;
    Tensor other_contiguous = other_cast.is_contiguous() ? other_cast : other_cast.contiguous();
    std::array<Tensor, 1> others = {other_contiguous};
    dispatch_inplace(OpId::SubInplace, self, others);
    return self;
}

auto div_(Tensor& self, const Tensor& other) -> Tensor& {
    check_inplace_autograd(self);
    validate_broadcast_shapes("div_", self.shape(), other.shape());
    if (!self.is_contiguous()) {
        throw std::runtime_error("In-place div requires contiguous tensor");
    }
    Tensor other_cast = (other.dtype() != self.dtype()) ? other.to(self.dtype()) : other;
    Tensor other_contiguous = other_cast.is_contiguous() ? other_cast : other_cast.contiguous();
    std::array<Tensor, 1> others = {other_contiguous};

    dispatch_inplace(OpId::DivInplace, self, others);

    return self;
}

// =========================================================================
// Extended Math Operations
// =========================================================================

auto log2(const Tensor& input) -> Tensor {
    std::vector<Tensor> inputs = {input};
    return dispatch<OpId::Log2>(inputs)[0];
}

auto log10(const Tensor& input) -> Tensor {
    std::vector<Tensor> inputs = {input};
    return dispatch<OpId::Log10>(inputs)[0];
}

auto log1p(const Tensor& input) -> Tensor {
    std::vector<Tensor> inputs = {input};
    return dispatch<OpId::Log1p>(inputs)[0];
}

auto exp2(const Tensor& input) -> Tensor {
    std::vector<Tensor> inputs = {input};
    return dispatch<OpId::Exp2>(inputs)[0];
}

auto expm1(const Tensor& input) -> Tensor {
    std::vector<Tensor> inputs = {input};
    return dispatch<OpId::Expm1>(inputs)[0];
}

auto erf(const Tensor& input) -> Tensor {
    std::vector<Tensor> inputs = {input};
    return dispatch<OpId::Erf>(inputs)[0];
}

auto erfc(const Tensor& input) -> Tensor {
    std::vector<Tensor> inputs = {input};
    return dispatch<OpId::Erfc>(inputs)[0];
}

auto isnan(const Tensor& input) -> Tensor {
    std::vector<Tensor> inputs = {input};
    return dispatch<OpId::IsNan>(inputs)[0];
}

auto isinf(const Tensor& input) -> Tensor {
    std::vector<Tensor> inputs = {input};
    return dispatch<OpId::IsInf>(inputs)[0];
}

auto isfinite(const Tensor& input) -> Tensor {
    std::vector<Tensor> inputs = {input};
    return dispatch<OpId::IsFinite>(inputs)[0];
}

auto atan2(const Tensor& y, const Tensor& x) -> Tensor {
    auto [yp, xp] = promote_inputs(y, x);
    validate_broadcast_shapes("atan2", yp.shape(), xp.shape());
    Tensor y_c = yp.is_contiguous() ? yp : yp.contiguous();
    Tensor x_c = xp.is_contiguous() ? xp : xp.contiguous();
    std::vector<Tensor> inputs = {y_c, x_c};
    return dispatch(OpId::Atan2, inputs)[0];
}

auto fmod(const Tensor& a, const Tensor& b) -> Tensor {
    auto [ap, bp] = promote_inputs(a, b);
    validate_broadcast_shapes("fmod", ap.shape(), bp.shape());
    Tensor a_c = ap.is_contiguous() ? ap : ap.contiguous();
    Tensor b_c = bp.is_contiguous() ? bp : bp.contiguous();
    std::vector<Tensor> inputs = {a_c, b_c};
    return dispatch(OpId::Fmod, inputs)[0];
}

auto remainder(const Tensor& a, const Tensor& b) -> Tensor {
    auto [ap, bp] = promote_inputs(a, b);
    validate_broadcast_shapes("remainder", ap.shape(), bp.shape());
    Tensor a_c = ap.is_contiguous() ? ap : ap.contiguous();
    Tensor b_c = bp.is_contiguous() ? bp : bp.contiguous();
    std::vector<Tensor> inputs = {a_c, b_c};
    return dispatch(OpId::Remainder, inputs)[0];
}

auto lerp(const Tensor& start, const Tensor& end, const Tensor& weight) -> Tensor {
    auto [sp, ep] = promote_inputs(start, end);
    Tensor wp = (weight.dtype() != sp.dtype()) ? weight.to(sp.dtype()) : weight;
    std::vector<Tensor> inputs = {sp, ep, wp};
    return dispatch(OpId::Lerp, inputs)[0];
}

auto lerp(const Tensor& start, const Tensor& end, double weight) -> Tensor {
    auto [sp, ep] = promote_inputs(start, end);
    Tensor w = full({1}, weight, sp.dtype(), sp.device());
    std::vector<Tensor> inputs = {sp, ep, w};
    return dispatch(OpId::Lerp, inputs)[0];
}

auto logical_and(const Tensor& a, const Tensor& b) -> Tensor {
    validate_broadcast_shapes("logical_and", a.shape(), b.shape());
    auto ac = a.is_contiguous() ? a : a.contiguous();
    auto bc = b.is_contiguous() ? b : b.contiguous();
    std::vector<Tensor> inputs = {ac, bc};
    return dispatch(OpId::LogicalAnd, inputs)[0];
}

auto logical_or(const Tensor& a, const Tensor& b) -> Tensor {
    validate_broadcast_shapes("logical_or", a.shape(), b.shape());
    auto ac = a.is_contiguous() ? a : a.contiguous();
    auto bc = b.is_contiguous() ? b : b.contiguous();
    std::vector<Tensor> inputs = {ac, bc};
    return dispatch(OpId::LogicalOr, inputs)[0];
}

auto logical_not(const Tensor& input) -> Tensor {
    std::vector<Tensor> inputs = {input};
    return dispatch(OpId::LogicalNot, inputs)[0];
}

auto logical_xor(const Tensor& a, const Tensor& b) -> Tensor {
    validate_broadcast_shapes("logical_xor", a.shape(), b.shape());
    auto ac = a.is_contiguous() ? a : a.contiguous();
    auto bc = b.is_contiguous() ? b : b.contiguous();
    std::vector<Tensor> inputs = {ac, bc};
    return dispatch(OpId::LogicalXor, inputs)[0];
}

auto cross(const Tensor& input, const Tensor& other, int64_t dim) -> Tensor {
    if (input.device().type != Device::Type::CPU)
        throw std::runtime_error("cross: only CPU tensors supported");

    auto shape_a = input.shape();
    auto shape_b = other.shape();
    int64_t ndim = shape_a.size();

    // Resolve negative dim
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim)
        throw std::invalid_argument("cross: dim out of range");

    if (shape_a[dim] != 3 || shape_b[dim] != 3)
        throw std::invalid_argument("cross: dimension " + std::to_string(dim) +
            " must have size 3, got " + std::to_string(shape_a[dim]) +
            " and " + std::to_string(shape_b[dim]));

    if (!std::equal(shape_a.begin(), shape_a.end(), shape_b.begin(), shape_b.end()))
        throw std::invalid_argument("cross: input tensors must have same shape");

    // Use indexing: c[...,0,...] = a[...,1,...]*b[...,2,...] - a[...,2,...]*b[...,1,...]
    auto a_cont = input.contiguous();
    auto b_cont = other.contiguous();
    auto result = empty(std::vector<int64_t>(shape_a.begin(), shape_a.end()),
                        input.dtype(), input.device());

    // Compute total iterations (product of all dims except dim)
    int64_t outer = 1, inner = 1;
    for (int64_t d = 0; d < dim; ++d) outer *= shape_a[d];
    for (int64_t d = dim + 1; d < ndim; ++d) inner *= shape_a[d];
    int64_t dim_stride = inner; // stride along the cross dimension

    auto cross_typed = [&](auto* a_ptr, auto* b_ptr, auto* c_ptr) {
        for (int64_t o = 0; o < outer; ++o) {
            for (int64_t i = 0; i < inner; ++i) {
                int64_t base = o * 3 * inner + i;
                auto a0 = a_ptr[base], a1 = a_ptr[base + dim_stride], a2 = a_ptr[base + 2*dim_stride];
                auto b0 = b_ptr[base], b1 = b_ptr[base + dim_stride], b2 = b_ptr[base + 2*dim_stride];
                c_ptr[base]                  = a1*b2 - a2*b1;
                c_ptr[base + dim_stride]     = a2*b0 - a0*b2;
                c_ptr[base + 2*dim_stride]   = a0*b1 - a1*b0;
            }
        }
    };

    if (input.dtype() == DType::Float32) {
        cross_typed(a_cont.data<float>(), b_cont.data<float>(), result.data<float>());
    } else if (input.dtype() == DType::Float64) {
        cross_typed(a_cont.data<double>(), b_cont.data<double>(), result.data<double>());
    } else {
        throw std::runtime_error("cross: only Float32 and Float64 supported");
    }

    return result;
}

} // namespace tenzor
