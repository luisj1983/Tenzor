#include "tenzor/ops/math.hpp"
#include "tenzor/backend/fast_dispatch.hpp"
#include "tenzor/backend/op_attributes.hpp"
#include "tenzor/ops/op_id.hpp"
#include "tenzor/ops/indexing.hpp"
#include "tenzor/ops/transform.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/reduction.hpp"
#include "tenzor/ops/type_promotion.hpp"
#include "tenzor/utils/profiling.hpp"
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
            throw std::runtime_error(
                std::string(op_name) + ": shapes " + a_str + "] and " + b_str +
                "] are not broadcast-compatible");
        }
    }
}

// ---------------------------------------------------------------------------
// Helper templates to eliminate repeated promote→validate→contiguous→dispatch
// ---------------------------------------------------------------------------
namespace detail {

// Standard binary op: promote dtypes, validate broadcast, make contiguous, dispatch
template<OpId Op>
auto binary_op_promoted(const char* name, const Tensor& a, const Tensor& b) -> Tensor {
    auto [ap, bp] = promote_inputs(a, b);
    validate_broadcast_shapes(name, ap.shape(), bp.shape());
    Tensor ac = ap.is_contiguous() ? ap : ap.contiguous();
    Tensor bc = bp.is_contiguous() ? bp : bp.contiguous();
    std::vector<Tensor> inputs = {ac, bc};
    return dispatch<Op>(inputs)[0];
}

// Binary op without type promotion (logical ops)
template<OpId Op>
auto binary_op_raw(const char* name, const Tensor& a, const Tensor& b) -> Tensor {
    validate_broadcast_shapes(name, a.shape(), b.shape());
    auto ac = a.is_contiguous() ? a : a.contiguous();
    auto bc = b.is_contiguous() ? b : b.contiguous();
    std::vector<Tensor> inputs = {ac, bc};
    return dispatch<Op>(inputs)[0];
}

// Simple unary op: just dispatch
template<OpId Op>
auto unary_op(const Tensor& input) -> Tensor {
    std::vector<Tensor> inputs = {input};
    return dispatch<Op>(inputs)[0];
}

} // namespace detail

// Math operation implementations - dispatched to backend kernels

auto add(const Tensor& a, const Tensor& b) -> Tensor {
    TENZOR_PROFILE_RANGE("add");
    // Validate tensors are initialized
    if (!a.impl() || !b.impl()) {
        throw std::runtime_error("Cannot add uninitialized tensors");
    }
    // Auto-promote dtypes if mismatched
    auto [ap, bp] = promote_inputs(a, b);
    validate_broadcast_shapes("add", ap.shape(), bp.shape());
    // Early return for empty tensor broadcasts (e.g. {0,5} + {1,5} -> {0,5})
    if (ap.numel() == 0 || bp.numel() == 0) {
        auto sa = ap.shape(); auto sb = bp.shape();
        size_t nd = std::max(sa.size(), sb.size());
        std::vector<int64_t> out(nd);
        for (size_t i = 0; i < nd; ++i) {
            int64_t da = i < sa.size() ? sa[sa.size()-1-i] : 1;
            int64_t db = i < sb.size() ? sb[sb.size()-1-i] : 1;
            out[nd-1-i] = (da == 1) ? db : da;
        }
        return Tensor(out, ap.dtype(), ap.device());
    }
    Tensor a_contiguous = ap.is_contiguous() ? ap : ap.contiguous();
    Tensor b_contiguous = bp.is_contiguous() ? bp : bp.contiguous();
    std::vector<Tensor> inputs = {a_contiguous, b_contiguous};
    return dispatch<OpId::Add>(inputs)[0];
}

auto sub(const Tensor& a, const Tensor& b) -> Tensor {
    return detail::binary_op_promoted<OpId::Sub>("sub", a, b);
}

auto mul(const Tensor& a, const Tensor& b) -> Tensor {
    return detail::binary_op_promoted<OpId::Mul>("mul", a, b);
}

auto div(const Tensor& a, const Tensor& b) -> Tensor {
    return detail::binary_op_promoted<OpId::Div>("div", a, b);
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
    NewOpAttributes attrs;
    attrs.set(AttrKey::ScalarB, scalar);
    return dispatch(OpId::Add, inputs, attrs)[0];
}

auto sub(const Tensor& a, double scalar) -> Tensor {
    if (!a.impl()) {
        throw std::runtime_error("Cannot subtract from uninitialized tensor");
    }
    auto scalar_tensor = full({1}, scalar, a.dtype(), a.device());
    std::vector<Tensor> inputs = {a.is_contiguous() ? a : a.contiguous(), scalar_tensor};
    NewOpAttributes attrs;
    attrs.set(AttrKey::ScalarB, scalar);
    return dispatch(OpId::Sub, inputs, attrs)[0];
}

auto mul(const Tensor& a, double scalar) -> Tensor {
    if (!a.impl()) {
        throw std::runtime_error("Cannot multiply uninitialized tensor");
    }
    auto scalar_tensor = full({1}, scalar, a.dtype(), a.device());
    std::vector<Tensor> inputs = {a.is_contiguous() ? a : a.contiguous(), scalar_tensor};
    NewOpAttributes attrs;
    attrs.set(AttrKey::ScalarB, scalar);
    return dispatch(OpId::Mul, inputs, attrs)[0];
}

auto div(const Tensor& a, double scalar) -> Tensor {
    if (!a.impl()) {
        throw std::runtime_error("Cannot divide uninitialized tensor");
    }
    auto scalar_tensor = full({1}, scalar, a.dtype(), a.device());
    std::vector<Tensor> inputs = {a.is_contiguous() ? a : a.contiguous(), scalar_tensor};
    NewOpAttributes attrs;
    attrs.set(AttrKey::ScalarB, scalar);
    return dispatch(OpId::Div, inputs, attrs)[0];
}

auto matmul(const Tensor& a, const Tensor& b) -> Tensor {
    TENZOR_PROFILE_RANGE("matmul");
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
        // 4D+ inputs: collapse leading batch dims into a single batch
        // dimension, bmm, then restore the original leading shape. Requires
        // both operands to share the same leading dims (standard PyTorch
        // torch.matmul contract — no broadcasting across batch dims here).
        if (ac.shape().size() > 3 || bc.shape().size() > 3) {
            const auto& a_shape = ac.shape();
            const auto& b_shape = bc.shape();
            if (a_shape.size() != b_shape.size()) {
                throw std::runtime_error(
                    "matmul: 4D+ operands must have the same number of "
                    "dimensions (got " + std::to_string(a_shape.size()) +
                    "D and " + std::to_string(b_shape.size()) + "D)");
            }
            const int64_t ndim = static_cast<int64_t>(a_shape.size());
            int64_t batch = 1;
            for (int64_t i = 0; i < ndim - 2; ++i) {
                if (a_shape[i] != b_shape[i]) {
                    throw std::runtime_error(
                        "matmul: batch dimensions must match, got " +
                        std::to_string(a_shape[i]) + " vs " +
                        std::to_string(b_shape[i]) + " at dim " +
                        std::to_string(i));
                }
                batch *= a_shape[i];
            }
            const int64_t M = a_shape[ndim - 2];
            const int64_t K = a_shape[ndim - 1];
            const int64_t N = b_shape[ndim - 1];
            auto ar = ac.reshape({batch, M, K});
            auto br = bc.reshape({batch, K, N});
            auto cr = bmm(ar, br);
            // Rebuild the output shape: leading batch dims + (M, N).
            std::vector<int64_t> out_shape(a_shape.begin(),
                                           a_shape.begin() + (ndim - 2));
            out_shape.push_back(M);
            out_shape.push_back(N);
            return cr.reshape(out_shape);
        }
        return bmm(ac, bc);
    }
    auto ac = ap.is_contiguous() ? ap : ap.contiguous();
    auto bc = bp.is_contiguous() ? bp : bp.contiguous();
    std::array<Tensor, 2> inputs = {ac, bc};
    return dispatch_single<OpId::MatMul>(inputs);
}

auto bmm(const Tensor& a, const Tensor& b) -> Tensor {
    TENZOR_PROFILE_RANGE("bmm");
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
    NewOpAttributes attrs;
    attrs.set(AttrKey::Exponent, static_cast<double>(exponent));
    std::vector<Tensor> inputs = {input};
    return dispatch(OpId::Pow, inputs, attrs)[0];
}

auto exp(const Tensor& input) -> Tensor { return detail::unary_op<OpId::Exp>(input); }
auto log(const Tensor& input) -> Tensor { return detail::unary_op<OpId::Log>(input); }
auto sqrt(const Tensor& input) -> Tensor { return detail::unary_op<OpId::Sqrt>(input); }
auto sin(const Tensor& input) -> Tensor { return detail::unary_op<OpId::Sin>(input); }
auto cos(const Tensor& input) -> Tensor { return detail::unary_op<OpId::Cos>(input); }
auto tan(const Tensor& input) -> Tensor { return detail::unary_op<OpId::Tan>(input); }
auto tanh(const Tensor& input) -> Tensor { return detail::unary_op<OpId::Tanh>(input); }
auto abs(const Tensor& input) -> Tensor { return detail::unary_op<OpId::Abs>(input); }
auto neg(const Tensor& input) -> Tensor { return detail::unary_op<OpId::Neg>(input); }
auto reciprocal(const Tensor& input) -> Tensor { return detail::unary_op<OpId::Reciprocal>(input); }
auto sign(const Tensor& input) -> Tensor { return detail::unary_op<OpId::Sign>(input); }
auto sigmoid(const Tensor& input) -> Tensor { return detail::unary_op<OpId::Sigmoid>(input); }

auto minimum(const Tensor& a, const Tensor& b) -> Tensor {
    return detail::binary_op_promoted<OpId::Minimum>("minimum", a, b);
}

auto maximum(const Tensor& a, const Tensor& b) -> Tensor {
    return detail::binary_op_promoted<OpId::Maximum>("maximum", a, b);
}

auto floor(const Tensor& input) -> Tensor { return detail::unary_op<OpId::Floor>(input); }
auto ceil(const Tensor& input) -> Tensor { return detail::unary_op<OpId::Ceil>(input); }
auto round(const Tensor& input) -> Tensor { return detail::unary_op<OpId::Round>(input); }

auto clamp(const Tensor& input, float min, float max) -> Tensor {
    NewOpAttributes attrs;
    attrs.set(AttrKey::Min, static_cast<double>(min));
    attrs.set(AttrKey::Max, static_cast<double>(max));
    std::vector<Tensor> inputs = {input};
    return dispatch(OpId::Clamp, inputs, attrs)[0];
}

auto clamp_min(const Tensor& input, float min) -> Tensor {
    NewOpAttributes attrs;
    attrs.set(AttrKey::Min, static_cast<double>(min));
    std::vector<Tensor> inputs = {input};
    return dispatch(OpId::ClampMin, inputs, attrs)[0];
}

auto clamp_max(const Tensor& input, float max) -> Tensor {
    NewOpAttributes attrs;
    attrs.set(AttrKey::Max, static_cast<double>(max));
    std::vector<Tensor> inputs = {input};
    return dispatch(OpId::ClampMax, inputs, attrs)[0];
}

auto sinh(const Tensor& input) -> Tensor { return detail::unary_op<OpId::Sinh>(input); }
auto cosh(const Tensor& input) -> Tensor { return detail::unary_op<OpId::Cosh>(input); }
auto atan(const Tensor& input) -> Tensor { return detail::unary_op<OpId::Atan>(input); }
auto asin(const Tensor& input) -> Tensor { return detail::unary_op<OpId::Asin>(input); }
auto acos(const Tensor& input) -> Tensor { return detail::unary_op<OpId::Acos>(input); }

// Comparison operations — auto-promote dtypes before comparing
auto eq(const Tensor& a, const Tensor& b) -> Tensor { return detail::binary_op_promoted<OpId::Eq>("eq", a, b); }
auto ne(const Tensor& a, const Tensor& b) -> Tensor { return detail::binary_op_promoted<OpId::Ne>("ne", a, b); }
auto lt(const Tensor& a, const Tensor& b) -> Tensor { return detail::binary_op_promoted<OpId::Lt>("lt", a, b); }
auto le(const Tensor& a, const Tensor& b) -> Tensor { return detail::binary_op_promoted<OpId::Le>("le", a, b); }
auto gt(const Tensor& a, const Tensor& b) -> Tensor { return detail::binary_op_promoted<OpId::Gt>("gt", a, b); }
auto ge(const Tensor& a, const Tensor& b) -> Tensor { return detail::binary_op_promoted<OpId::Ge>("ge", a, b); }

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

auto log2(const Tensor& input) -> Tensor { return detail::unary_op<OpId::Log2>(input); }
auto log10(const Tensor& input) -> Tensor { return detail::unary_op<OpId::Log10>(input); }
auto log1p(const Tensor& input) -> Tensor { return detail::unary_op<OpId::Log1p>(input); }
auto exp2(const Tensor& input) -> Tensor { return detail::unary_op<OpId::Exp2>(input); }
auto expm1(const Tensor& input) -> Tensor { return detail::unary_op<OpId::Expm1>(input); }
auto erf(const Tensor& input) -> Tensor { return detail::unary_op<OpId::Erf>(input); }
auto erfc(const Tensor& input) -> Tensor { return detail::unary_op<OpId::Erfc>(input); }
auto erfinv(const Tensor& input) -> Tensor { return detail::unary_op<OpId::ErfInv>(input); }

auto gamma(const Tensor& input) -> Tensor { return detail::unary_op<OpId::Gamma>(input); }
auto lgamma(const Tensor& input) -> Tensor { return detail::unary_op<OpId::Lgamma>(input); }
auto digamma(const Tensor& input) -> Tensor { return detail::unary_op<OpId::Digamma>(input); }

auto polygamma(int64_t n, const Tensor& input) -> Tensor {
    OpAttributes attrs;
    attrs.set(AttrKey::Order, static_cast<double>(n));
    std::vector<Tensor> inputs = {input};
    return dispatch(OpId::Polygamma, inputs, attrs)[0];
}

auto beta(const Tensor& a, const Tensor& b) -> Tensor {
    return detail::binary_op_promoted<OpId::Beta>("beta", a, b);
}

auto betainc(const Tensor& a, const Tensor& b, const Tensor& x) -> Tensor {
    auto [ap, bp] = promote_inputs(a, b);
    auto [ap2, xp] = promote_inputs(ap, x);
    Tensor ac = ap2.is_contiguous() ? ap2 : ap2.contiguous();
    Tensor bc = bp.is_contiguous() ? bp : bp.contiguous();
    Tensor xc = xp.is_contiguous() ? xp : xp.contiguous();
    std::vector<Tensor> inputs = {ac, bc, xc};
    return dispatch<OpId::BetaInc>(inputs)[0];
}

auto bessel_j0(const Tensor& input) -> Tensor { return detail::unary_op<OpId::BesselJ0>(input); }
auto bessel_j1(const Tensor& input) -> Tensor { return detail::unary_op<OpId::BesselJ1>(input); }
auto bessel_y0(const Tensor& input) -> Tensor { return detail::unary_op<OpId::BesselY0>(input); }
auto bessel_y1(const Tensor& input) -> Tensor { return detail::unary_op<OpId::BesselY1>(input); }
auto bessel_i0(const Tensor& input) -> Tensor { return detail::unary_op<OpId::BesselI0>(input); }
auto bessel_i1(const Tensor& input) -> Tensor { return detail::unary_op<OpId::BesselI1>(input); }

auto sinc(const Tensor& input) -> Tensor { return detail::unary_op<OpId::Sinc>(input); }

auto zeta(const Tensor& x, const Tensor& q) -> Tensor {
    return detail::binary_op_promoted<OpId::Zeta>("zeta", x, q);
}

auto isnan(const Tensor& input) -> Tensor { return detail::unary_op<OpId::IsNan>(input); }
auto isinf(const Tensor& input) -> Tensor { return detail::unary_op<OpId::IsInf>(input); }
auto isfinite(const Tensor& input) -> Tensor { return detail::unary_op<OpId::IsFinite>(input); }

auto atan2(const Tensor& y, const Tensor& x) -> Tensor {
    return detail::binary_op_promoted<OpId::Atan2>("atan2", y, x);
}

auto fmod(const Tensor& a, const Tensor& b) -> Tensor {
    return detail::binary_op_promoted<OpId::Fmod>("fmod", a, b);
}

auto remainder(const Tensor& a, const Tensor& b) -> Tensor {
    return detail::binary_op_promoted<OpId::Remainder>("remainder", a, b);
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

auto logical_and(const Tensor& a, const Tensor& b) -> Tensor { return detail::binary_op_raw<OpId::LogicalAnd>("logical_and", a, b); }
auto logical_or(const Tensor& a, const Tensor& b) -> Tensor { return detail::binary_op_raw<OpId::LogicalOr>("logical_or", a, b); }
auto logical_not(const Tensor& input) -> Tensor { return detail::unary_op<OpId::LogicalNot>(input); }
auto logical_xor(const Tensor& a, const Tensor& b) -> Tensor { return detail::binary_op_raw<OpId::LogicalXor>("logical_xor", a, b); }

auto cross(const Tensor& input, const Tensor& other, int64_t dim) -> Tensor {
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

    auto a_cont = input.contiguous();
    auto b_cont = other.contiguous();
    std::array<Tensor, 2> inputs = {a_cont, b_cont};
    NewOpAttributes attrs;
    attrs.set(AttrKey::Dim, dim);
    return dispatch<OpId::Cross>(inputs, attrs)[0];
}

// ============================================================================
// Search Operations
// ============================================================================

auto searchsorted(const Tensor& sorted_sequence, const Tensor& values, bool right) -> Tensor {
    if (sorted_sequence.ndim() != 1) {
        throw std::runtime_error("searchsorted: sorted_sequence must be 1-D");
    }

    std::array<Tensor, 2> inputs = {sorted_sequence, values};
    NewOpAttributes attrs;
    attrs.set(AttrKey::Right, right);
    return dispatch<OpId::SearchSorted>(inputs, attrs)[0];
}

// ============================================================================
// Sampling Operations
// ============================================================================

auto gumbel_softmax(const Tensor& logits, double tau, bool hard, int64_t dim) -> Tensor {
    std::array<Tensor, 1> inputs = {logits};
    NewOpAttributes attrs;
    attrs.set(AttrKey::Tau, tau);
    attrs.set(AttrKey::Hard, hard);
    attrs.set(AttrKey::Dim, dim);
    return dispatch<OpId::GumbelSoftmax>(inputs, attrs)[0];
}

// =========================================================================
// Complex Number Operations
// =========================================================================

auto conj(const Tensor& input) -> Tensor {
    std::vector<Tensor> inputs = {input};
    return dispatch<OpId::Conj>(inputs)[0];
}

auto real(const Tensor& input) -> Tensor {
    std::vector<Tensor> inputs = {input};
    return dispatch<OpId::Real>(inputs)[0];
}

auto imag(const Tensor& input) -> Tensor {
    std::vector<Tensor> inputs = {input};
    return dispatch<OpId::Imag>(inputs)[0];
}

auto angle(const Tensor& input) -> Tensor {
    std::vector<Tensor> inputs = {input};
    return dispatch<OpId::Angle>(inputs)[0];
}

auto polar(const Tensor& abs, const Tensor& angle) -> Tensor {
    std::vector<Tensor> inputs = {abs, angle};
    return dispatch<OpId::Polar>(inputs)[0];
}

auto cdist(const Tensor& x1, const Tensor& x2, double p) -> Tensor {
    if (x1.ndim() < 2 || x2.ndim() < 2) {
        throw std::invalid_argument("cdist: inputs must have at least 2 dimensions");
    }
    if (x1.shape()[x1.ndim() - 1] != x2.shape()[x2.ndim() - 1]) {
        throw std::invalid_argument("cdist: last dimension of x1 and x2 must match");
    }
    if (p < 0) {
        throw std::invalid_argument("cdist: p must be non-negative");
    }

    auto a = x1.contiguous();
    auto b = x2.contiguous();
    std::array<Tensor, 2> inputs = {a, b};
    NewOpAttributes attrs;
    attrs.set(AttrKey::DistP, p);
    return dispatch<OpId::CDist>(inputs, attrs)[0];
}

} // namespace tenzor
