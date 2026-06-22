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
#include <limits>
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

// Binary op with promotion + EXPLICIT broadcast expansion. For backend kernels
// that index both operands with a single linear index (no per-operand broadcast
// strides) — e.g. hypot/copysign/nextafter/gcd/lcm and the special functions.
// Those kernels would otherwise read the smaller operand out of bounds and
// return a wrong-shaped result. (add/mul keep binary_op_promoted; their kernels
// broadcast via strides.) Mirrors the addcmul broadcast handling.
template<OpId Op>
auto binary_op_promoted_broadcast(const char* name, const Tensor& a, const Tensor& b) -> Tensor {
    auto [ap, bp] = promote_inputs(a, b);
    validate_broadcast_shapes(name, ap.shape(), bp.shape());
    auto bshape = broadcast_shapes(ap.shape(), bp.shape());
    std::vector<Tensor> inputs = {broadcast_to(ap, bshape).contiguous(),
                                  broadcast_to(bp, bshape).contiguous()};
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
    // True division of two Bool tensors must promote to a float dtype (PyTorch
    // `/` semantics). promote_types(Bool, Bool) == Bool, and the Div kernel would
    // otherwise perform integer division on bool (`x / false` is UB / SIGFPE).
    if (a.dtype() == DType::Bool && b.dtype() == DType::Bool) {
        Tensor af = a.to(DType::Float32);
        Tensor bf = b.to(DType::Float32);
        return detail::binary_op_promoted<OpId::Div>("div", af, bf);
    }
    return detail::binary_op_promoted<OpId::Div>("div", a, b);
}

// Scalar arithmetic overloads — avoid creating temporary scalar tensors.
// Creates a size-{1} scalar tensor once and delegates to the tensor overload,
// but backends can optimize this path internally.

namespace {
// Dispatch a tensor-scalar elementwise op. The materialized scalar tensor and
// the ScalarB attribute must encode the SAME value, else backends diverge
// depending on which they read. We also apply PyTorch scalar type-promotion: a
// non-integral scalar applied to an integer tensor promotes to the default
// float dtype instead of silently truncating the fraction (e.g. int + 0.5).
auto dispatch_scalar_binop(OpId op, const Tensor& a, double scalar) -> Tensor {
    DType rdt = a.dtype();
    const bool scalar_is_integral =
        std::isfinite(scalar) && scalar == std::floor(scalar);
    // True division always yields a float (PyTorch `/` semantics): int / scalar
    // must promote even for an integer-valued scalar, otherwise the Div kernel
    // truncates (e.g. Int32 / 2.0 -> truncated quotient). Add/Sub/Mul keep the
    // integer fast-path when the scalar is integral.
    const bool promote_for_div = (op == OpId::Div) && is_integer_type(rdt);
    if (is_integer_type(rdt) && (!scalar_is_integral || promote_for_div)) {
        rdt = DType::Float32;
    }
    // Bool tensors: true division must promote to float (PyTorch `/` semantics);
    // also avoids bool/scalar integer-divide UB in DivOp::scalar<bool>. A
    // non-integral scalar applied to a Bool tensor likewise promotes (e.g.
    // bool + 0.5) rather than truncating.
    if (rdt == DType::Bool && (op == OpId::Div || !scalar_is_integral)) {
        rdt = DType::Float32;
    }
    Tensor a_use = (a.dtype() != rdt) ? a.to(rdt) : a;
    a_use = a_use.is_contiguous() ? a_use : a_use.contiguous();
    Tensor scalar_tensor = full({1}, scalar, rdt, a.device());
    std::vector<Tensor> inputs = {a_use, scalar_tensor};
    NewOpAttributes attrs;
    attrs.set(AttrKey::ScalarB, scalar);
    return dispatch(op, inputs, attrs)[0];
}
}  // namespace

auto add(const Tensor& a, double scalar) -> Tensor {
    if (!a.impl()) {
        throw std::runtime_error("Cannot add to uninitialized tensor");
    }
    return dispatch_scalar_binop(OpId::Add, a, scalar);
}

auto sub(const Tensor& a, double scalar) -> Tensor {
    if (!a.impl()) {
        throw std::runtime_error("Cannot subtract from uninitialized tensor");
    }
    return dispatch_scalar_binop(OpId::Sub, a, scalar);
}

auto mul(const Tensor& a, double scalar) -> Tensor {
    if (!a.impl()) {
        throw std::runtime_error("Cannot multiply uninitialized tensor");
    }
    return dispatch_scalar_binop(OpId::Mul, a, scalar);
}

auto div(const Tensor& a, double scalar) -> Tensor {
    if (!a.impl()) {
        throw std::runtime_error("Cannot divide uninitialized tensor");
    }
    return dispatch_scalar_binop(OpId::Div, a, scalar);
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
    // 2D @ 2D: validate the contracted dimension at the op layer so a mismatch
    // is a uniform error rather than a backend-specific failure (bmm and the
    // 4D+ path above already validate; this keeps the contract consistent).
    if (ap.ndim() == 2 && bp.ndim() == 2 && ap.shape()[1] != bp.shape()[0]) {
        throw std::runtime_error(
            "matmul: inner dimensions don't match (" +
            std::to_string(ap.shape()[0]) + "x" + std::to_string(ap.shape()[1]) +
            " @ " + std::to_string(bp.shape()[0]) + "x" +
            std::to_string(bp.shape()[1]) + ")");
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

auto addmm(const Tensor& input, const Tensor& mat1, const Tensor& mat2,
           double beta, double alpha) -> Tensor {
    TENZOR_PROFILE_RANGE("addmm");
    // Promote mat1/mat2 dtypes, then promote input to match
    auto [m1p, m2p] = promote_inputs(mat1, mat2);
    Tensor inp = (input.dtype() != m1p.dtype()) ? input.to(m1p.dtype()) : input;

    // Validate dimensions
    if (m1p.ndim() != 2 || m2p.ndim() != 2) {
        throw std::runtime_error("addmm: mat1 and mat2 must be 2D tensors");
    }
    if (m1p.shape()[1] != m2p.shape()[0]) {
        throw std::runtime_error(
            "addmm: mat1 (" + std::to_string(m1p.shape()[0]) + "x" +
            std::to_string(m1p.shape()[1]) + ") and mat2 (" +
            std::to_string(m2p.shape()[0]) + "x" + std::to_string(m2p.shape()[1]) +
            ") inner dimensions don't match");
    }
    // `input` must be broadcast-compatible with the (M, N) result.
    {
        const int64_t M = m1p.shape()[0];
        const int64_t N = m2p.shape()[1];
        const auto& is = inp.shape();
        bool ok = is.size() <= 2;
        if (ok) {
            const int64_t in_n = is.empty() ? 1 : is.back();
            const int64_t in_m = (is.size() == 2) ? is[0] : 1;
            ok = (in_n == N || in_n == 1) && (in_m == M || in_m == 1);
        }
        if (!ok) {
            throw std::runtime_error(
                "addmm: input shape is not broadcastable to the result (" +
                std::to_string(M) + "x" + std::to_string(N) + ")");
        }
    }

    auto ic = inp.is_contiguous() ? inp : inp.contiguous();
    auto m1c = m1p.is_contiguous() ? m1p : m1p.contiguous();
    auto m2c = m2p.is_contiguous() ? m2p : m2p.contiguous();

    NewOpAttributes attrs;
    attrs.set(AttrKey::Alpha, alpha);
    attrs.set(AttrKey::Beta, beta);

    std::array<Tensor, 3> inputs = {ic, m1c, m2c};
    return dispatch_single(OpId::Addmm, inputs, attrs);
}

auto addmv(const Tensor& input, const Tensor& mat, const Tensor& vec,
           double beta, double alpha) -> Tensor {
    TENZOR_PROFILE_RANGE("addmv");
    auto [mp, vp] = promote_inputs(mat, vec);
    Tensor inp = (input.dtype() != mp.dtype()) ? input.to(mp.dtype()) : input;

    if (mp.ndim() != 2) {
        throw std::runtime_error("addmv: mat must be a 2D tensor");
    }
    if (vp.ndim() != 1) {
        throw std::runtime_error("addmv: vec must be a 1D tensor");
    }
    if (mp.shape()[1] != vp.shape()[0]) {
        throw std::runtime_error(
            "addmv: mat (" + std::to_string(mp.shape()[0]) + "x" +
            std::to_string(mp.shape()[1]) + ") and vec (" +
            std::to_string(vp.shape()[0]) + ") inner dimensions don't match");
    }

    auto ic = inp.is_contiguous() ? inp : inp.contiguous();
    auto mc = mp.is_contiguous() ? mp : mp.contiguous();
    auto vc = vp.is_contiguous() ? vp : vp.contiguous();

    NewOpAttributes attrs;
    attrs.set(AttrKey::Alpha, alpha);
    attrs.set(AttrKey::Beta, beta);

    std::array<Tensor, 3> inputs = {ic, mc, vc};
    return dispatch_single(OpId::Addmv, inputs, attrs);
}

auto baddbmm(const Tensor& input, const Tensor& batch1, const Tensor& batch2,
             double beta, double alpha) -> Tensor {
    TENZOR_PROFILE_RANGE("baddbmm");
    auto [b1p, b2p] = promote_inputs(batch1, batch2);
    Tensor inp = (input.dtype() != b1p.dtype()) ? input.to(b1p.dtype()) : input;

    if (b1p.ndim() != 3 || b2p.ndim() != 3) {
        throw std::runtime_error("baddbmm: batch1 and batch2 must be 3D tensors");
    }
    if (b1p.shape()[0] != b2p.shape()[0]) {
        throw std::runtime_error("baddbmm: batch sizes must match");
    }
    if (b1p.shape()[2] != b2p.shape()[1]) {
        throw std::runtime_error(
            "baddbmm: inner dimensions don't match (" +
            std::to_string(b1p.shape()[2]) + " vs " +
            std::to_string(b2p.shape()[1]) + ")");
    }

    auto ic = inp.is_contiguous() ? inp : inp.contiguous();
    auto b1c = b1p.is_contiguous() ? b1p : b1p.contiguous();
    auto b2c = b2p.is_contiguous() ? b2p : b2p.contiguous();

    NewOpAttributes attrs;
    attrs.set(AttrKey::Alpha, alpha);
    attrs.set(AttrKey::Beta, beta);

    std::array<Tensor, 3> inputs = {ic, b1c, b2c};
    return dispatch_single(OpId::Baddbmm, inputs, attrs);
}

auto pow(const Tensor& input, double exponent) -> Tensor {
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
auto trunc(const Tensor& input) -> Tensor { return detail::unary_op<OpId::Trunc>(input); }

auto clamp(const Tensor& input, double min, double max) -> Tensor {
    NewOpAttributes attrs;
    attrs.set(AttrKey::Min, static_cast<double>(min));
    attrs.set(AttrKey::Max, static_cast<double>(max));
    std::vector<Tensor> inputs = {input};
    return dispatch(OpId::Clamp, inputs, attrs)[0];
}

auto clamp_min(const Tensor& input, double min) -> Tensor {
    NewOpAttributes attrs;
    attrs.set(AttrKey::Min, static_cast<double>(min));
    std::vector<Tensor> inputs = {input};
    return dispatch(OpId::ClampMin, inputs, attrs)[0];
}

auto clamp_max(const Tensor& input, double max) -> Tensor {
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
    // True division produces a floating result; it cannot be stored back into a
    // Bool tensor in place (matches PyTorch: "result type Float can't be cast to
    // the desired output type Bool"). Also avoids bool/bool integer-divide UB.
    if (self.dtype() == DType::Bool) {
        throw std::runtime_error(
            "div_: in-place true division is not supported for Bool tensors "
            "(result type cannot be cast back to Bool)");
    }
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
    return detail::binary_op_promoted_broadcast<OpId::Beta>("beta", a, b);
}

auto betainc(const Tensor& a, const Tensor& b, const Tensor& x) -> Tensor {
    // Compute a single common dtype across all three operands. The kernel keys
    // its dtype branch off a.dtype() and reads b/x with that same element type,
    // so every operand must share the promoted dtype; otherwise a narrower
    // operand's buffer is reinterpreted as a wider type (out-of-bounds read).
    DType common = promote_types(promote_types(a.dtype(), b.dtype()), x.dtype());
    Tensor ap = (a.dtype() != common) ? a.to(common) : a;
    Tensor bp = (b.dtype() != common) ? b.to(common) : b;
    Tensor xp = (x.dtype() != common) ? x.to(common) : x;
    Tensor ac = ap.is_contiguous() ? ap : ap.contiguous();
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
    return detail::binary_op_promoted_broadcast<OpId::Zeta>("zeta", x, q);
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
    // Broadcast + contiguify all three operands; the Lerp kernels index
    // elementwise and would otherwise read out of bounds on unequal shapes.
    auto bshape = broadcast_shapes(sp.shape(), ep.shape());
    bshape = broadcast_shapes(bshape, wp.shape());
    std::vector<Tensor> inputs = {broadcast_to(sp, bshape).contiguous(),
                                  broadcast_to(ep, bshape).contiguous(),
                                  broadcast_to(wp, bshape).contiguous()};
    return dispatch(OpId::Lerp, inputs)[0];
}

auto lerp(const Tensor& start, const Tensor& end, double weight) -> Tensor {
    auto [sp, ep] = promote_inputs(start, end);
    auto bshape = broadcast_shapes(sp.shape(), ep.shape());
    Tensor w = full(bshape, weight, sp.dtype(), sp.device());
    std::vector<Tensor> inputs = {broadcast_to(sp, bshape).contiguous(),
                                  broadcast_to(ep, bshape).contiguous(), w};
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

    // Validate equal rank BEFORE indexing shape_b[dim]: if `other` has fewer
    // dims than `input`, shape_b[dim] would read past the end of its shape
    // vector (the full-shape equality check below runs too late to catch it).
    if (static_cast<int64_t>(shape_b.size()) != ndim)
        throw std::invalid_argument(
            "cross: input tensors must have the same number of dimensions");

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

    // The kernel keys its dtype branch off sorted_sequence.dtype() and reads
    // the values buffer with that same element type. A mismatched values dtype
    // would be reinterpreted as the wrong type (and, when values is wider,
    // read out of bounds). Cast values to the sequence dtype before dispatch.
    Tensor values_cast = (values.dtype() != sorted_sequence.dtype())
                             ? values.to(sorted_sequence.dtype())
                             : values;
    std::array<Tensor, 2> inputs = {sorted_sequence, values_cast};
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

auto frac(const Tensor& input) -> Tensor {
    return detail::unary_op<OpId::Frac>(input);
}

auto heaviside(const Tensor& input, const Tensor& values) -> Tensor {
    return detail::binary_op_promoted<OpId::Heaviside>("heaviside", input, values);
}

auto nan_to_num(const Tensor& input, double nan, double posinf, double neginf) -> Tensor {
    std::array<Tensor, 1> inputs = {input.contiguous()};
    NewOpAttributes attrs;
    attrs.set(AttrKey::NanValue, nan);
    attrs.set(AttrKey::PosInfValue, posinf);
    attrs.set(AttrKey::NegInfValue, neginf);
    return dispatch<OpId::NanToNum>(inputs, attrs)[0];
}

auto bitwise_and(const Tensor& a, const Tensor& b) -> Tensor {
    return detail::binary_op_promoted<OpId::BitwiseAnd>("bitwise_and", a, b);
}

auto bitwise_or(const Tensor& a, const Tensor& b) -> Tensor {
    return detail::binary_op_promoted<OpId::BitwiseOr>("bitwise_or", a, b);
}

auto bitwise_xor(const Tensor& a, const Tensor& b) -> Tensor {
    return detail::binary_op_promoted<OpId::BitwiseXor>("bitwise_xor", a, b);
}

auto bitwise_not(const Tensor& input) -> Tensor {
    return detail::unary_op<OpId::BitwiseNot>(input);
}

auto bitwise_left_shift(const Tensor& input, const Tensor& shift) -> Tensor {
    return detail::binary_op_promoted<OpId::BitwiseLeftShift>("bitwise_left_shift", input, shift);
}

auto bitwise_right_shift(const Tensor& input, const Tensor& shift) -> Tensor {
    return detail::binary_op_promoted<OpId::BitwiseRightShift>("bitwise_right_shift", input, shift);
}

// ---------------------------------------------------------------------------
// Composed math operations (no new backend kernels needed)
// ---------------------------------------------------------------------------

auto diff(const Tensor& input, int64_t n, int64_t dim) -> Tensor {
    if (n < 1) {
        throw std::runtime_error("diff: n must be >= 1, got " + std::to_string(n));
    }

    const int64_t ndim = input.ndim();
    if (ndim == 0) {
        throw std::runtime_error("diff: input must have at least 1 dimension");
    }

    // Normalize dim
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) {
        throw std::runtime_error("diff: dimension out of range");
    }

    Tensor result = input;
    for (int64_t i = 0; i < n; ++i) {
        int64_t dim_size = result.shape()[dim];
        if (dim_size < 2) {
            throw std::runtime_error("diff: dimension size must be >= 2 for each application, got " +
                                     std::to_string(dim_size));
        }
        auto front = narrow(result, dim, 1, dim_size - 1);
        auto back = narrow(result, dim, 0, dim_size - 1);
        result = tenzor::sub(front, back);
    }
    return result;
}

auto logaddexp(const Tensor& a, const Tensor& b) -> Tensor {
    return detail::binary_op_promoted_broadcast<OpId::LogAddExp>("logaddexp", a, b);
}

auto logaddexp2(const Tensor& a, const Tensor& b) -> Tensor {
    return detail::binary_op_promoted_broadcast<OpId::LogAddExp2>("logaddexp2", a, b);
}

auto xlogy(const Tensor& x, const Tensor& y) -> Tensor {
    return detail::binary_op_promoted_broadcast<OpId::XLogY>("xlogy", x, y);
}

auto i0e(const Tensor& x) -> Tensor {
    std::array<Tensor, 1> inputs = {x};
    return dispatch<OpId::I0e>(inputs)[0];
}

auto i1e(const Tensor& x) -> Tensor {
    std::array<Tensor, 1> inputs = {x};
    return dispatch<OpId::I1e>(inputs)[0];
}

auto entr(const Tensor& x) -> Tensor {
    std::array<Tensor, 1> inputs = {x};
    return dispatch<OpId::Entr>(inputs)[0];
}

auto spherical_bessel_j0(const Tensor& x) -> Tensor {
    std::array<Tensor, 1> inputs = {x};
    return dispatch<OpId::SphericalBesselJ0>(inputs)[0];
}

auto cosine_similarity(const Tensor& x1, const Tensor& x2,
                       int64_t dim, double eps) -> Tensor {
    // Normalise the dim once at the dispatch layer so every backend kernel
    // (and every step of the GPU composed-ops fallback that builds cosine
    // similarity from sum + norm + div) receives the same non-negative
    // index. Without this, only CPU + CUDA happened to normalise the
    // negative dim correctly; Vulkan/OneAPI/ROCm passed the raw negative
    // through their internal `dispatchReduction`/`dispatchNorm` chain
    // which reduced the wrong axis. See [Negative-dim normalization gaps]
    // in MEMORY.md.
    int64_t ndim = static_cast<int64_t>(x1.shape().size());
    int64_t actual_dim = (dim < 0) ? dim + ndim : dim;
    if (actual_dim < 0 || actual_dim >= ndim) {
        throw std::invalid_argument(
            "cosine_similarity: dim " + std::to_string(dim) +
            " is out of range for a rank-" + std::to_string(ndim) + " input");
    }
    OpAttributes attrs;
    attrs.set(AttrKey::Dim, actual_dim);
    attrs.set(AttrKey::Eps, eps);
    std::array<Tensor, 2> inputs = {x1, x2};
    return dispatch_single(OpId::CosineSimilarity, inputs, attrs);
}

auto renorm(const Tensor& input, double p, int64_t dim, double maxnorm) -> Tensor {
    OpAttributes attrs;
    attrs.set(AttrKey::P, p);
    attrs.set(AttrKey::Dim, dim);
    attrs.set(AttrKey::MaxNorm, maxnorm);
    std::array<Tensor, 1> inputs = {input};
    return dispatch_single(OpId::Renorm, inputs, attrs);
}

auto isclose(const Tensor& a, const Tensor& b, double rtol, double atol) -> Tensor {
    // For integral/bool inputs the rtol/atol scalars would be truncated to 0 in
    // the input dtype, degenerating isclose into exact equality. For low-
    // precision floats (Float16/BFloat16) atol=1e-8 underflows below the
    // smallest representable subnormal (collapsing to 0) and the diff/tol math
    // loses precision. In both cases promote to a higher floating compute dtype
    // (matching PyTorch's promote-to-compute-type) so tolerances are honoured.
    auto needs_promote = [](DType d) {
        return d != DType::Float32 && d != DType::Float64;
    };
    Tensor ca = a;
    Tensor cb = b;
    if (needs_promote(a.dtype()) || needs_promote(b.dtype())) {
        DType compute = (a.dtype() == DType::Float64 || b.dtype() == DType::Float64)
                            ? DType::Float64
                            : DType::Float32;
        ca = (a.dtype() != compute) ? a.to(compute) : a;
        cb = (b.dtype() != compute) ? b.to(compute) : b;
    }
    auto diff = tenzor::abs(tenzor::sub(ca, cb));
    auto tol = tenzor::add(
        tenzor::full({1}, atol, ca.dtype(), ca.device()),
        tenzor::mul(
            tenzor::full({1}, rtol, ca.dtype(), ca.device()),
            tenzor::abs(cb)));
    return tenzor::le(diff, tol);
}

auto allclose(const Tensor& a, const Tensor& b, double rtol, double atol) -> bool {
    auto close = isclose(a, b, rtol, atol);
    std::array<Tensor, 1> inputs = {close};
    auto all_result = dispatch<OpId::All>(inputs)[0];
    auto cpu_result = all_result.to(Device::cpu());
    // Read scalar — use data_ptr to avoid template parse issues
    if (cpu_result.dtype() == DType::Bool) {
        return *static_cast<const bool*>(cpu_result.data_ptr());
    }
    return *static_cast<const float*>(cpu_result.data_ptr()) != 0.0f;
}

// =========================================================================
// New element-wise math operations for PyTorch parity
// =========================================================================

auto rsqrt(const Tensor& input) -> Tensor {
    return detail::unary_op<OpId::Rsqrt>(input);
}

auto square(const Tensor& input) -> Tensor {
    return detail::unary_op<OpId::Square>(input);
}

auto asinh(const Tensor& input) -> Tensor {
    return detail::unary_op<OpId::Asinh>(input);
}

auto acosh(const Tensor& input) -> Tensor {
    return detail::unary_op<OpId::Acosh>(input);
}

auto atanh(const Tensor& input) -> Tensor {
    return detail::unary_op<OpId::Atanh>(input);
}

auto hypot(const Tensor& x, const Tensor& y) -> Tensor {
    return detail::binary_op_promoted_broadcast<OpId::Hypot>("hypot", x, y);
}

auto copysign(const Tensor& magnitude, const Tensor& sign) -> Tensor {
    return detail::binary_op_promoted_broadcast<OpId::Copysign>("copysign", magnitude, sign);
}

auto nextafter(const Tensor& from, const Tensor& to) -> Tensor {
    return detail::binary_op_promoted_broadcast<OpId::Nextafter>("nextafter", from, to);
}

auto gcd(const Tensor& a, const Tensor& b) -> Tensor {
    return detail::binary_op_promoted_broadcast<OpId::Gcd>("gcd", a, b);
}

auto lcm(const Tensor& a, const Tensor& b) -> Tensor {
    return detail::binary_op_promoted_broadcast<OpId::Lcm>("lcm", a, b);
}

auto addcmul(const Tensor& input, const Tensor& tensor1, const Tensor& tensor2,
             double value) -> Tensor {
    // Broadcast all three operands to their common shape; the backend kernels
    // index elementwise and would read out of bounds on unequal shapes.
    auto bshape = broadcast_shapes(input.shape(), tensor1.shape());
    bshape = broadcast_shapes(bshape, tensor2.shape());
    std::array<Tensor, 3> inputs = {broadcast_to(input, bshape).contiguous(),
                                    broadcast_to(tensor1, bshape).contiguous(),
                                    broadcast_to(tensor2, bshape).contiguous()};
    NewOpAttributes attrs;
    attrs.set(AttrKey::Alpha, value);
    return dispatch<OpId::Addcmul>(inputs, attrs)[0];
}

auto addcdiv(const Tensor& input, const Tensor& tensor1, const Tensor& tensor2,
             double value) -> Tensor {
    // Broadcast all three operands to their common shape (see addcmul).
    auto bshape = broadcast_shapes(input.shape(), tensor1.shape());
    bshape = broadcast_shapes(bshape, tensor2.shape());
    std::array<Tensor, 3> inputs = {broadcast_to(input, bshape).contiguous(),
                                    broadcast_to(tensor1, bshape).contiguous(),
                                    broadcast_to(tensor2, bshape).contiguous()};
    NewOpAttributes attrs;
    attrs.set(AttrKey::Alpha, value);
    return dispatch<OpId::Addcdiv>(inputs, attrs)[0];
}

auto igamma(const Tensor& a, const Tensor& x) -> Tensor {
    return detail::binary_op_promoted_broadcast<OpId::Igamma>("igamma", a, x);
}

auto igammac(const Tensor& a, const Tensor& x) -> Tensor {
    return detail::binary_op_promoted_broadcast<OpId::Igammac>("igammac", a, x);
}

auto gammainc(const Tensor& a, const Tensor& x) -> Tensor {
    // Lower incomplete gamma (non-regularized) = igamma(a, x) * Gamma(a)
    return tenzor::mul(igamma(a, x), tenzor::gamma(a));
}

auto gammaincc(const Tensor& a, const Tensor& x) -> Tensor {
    // Upper incomplete gamma (non-regularized) = igammac(a, x) * Gamma(a)
    return tenzor::mul(igammac(a, x), tenzor::gamma(a));
}

// =========================================================================
// Extended math operations (PyTorch parity)
// =========================================================================

auto deg2rad(const Tensor& input) -> Tensor {
    // x * pi / 180
    Tensor scale = full_like(input, 3.14159265358979323846f / 180.0f);
    return input * scale;
}

auto rad2deg(const Tensor& input) -> Tensor {
    // x * 180 / pi
    Tensor scale = full_like(input, 180.0f / 3.14159265358979323846f);
    return input * scale;
}

auto logit(const Tensor& input, double eps) -> Tensor {
    // logit(x) = log(x / (1-x))
    // With eps > 0: clamp x to [eps, 1-eps] first
    Tensor x = input;
    if (eps > 0) {
        // clamp takes double bounds; casting to float here corrupted the
        // [eps, 1-eps] window for Float64 inputs (1-1e-6 is not representable
        // in float, shifting the result near the boundary by ~1e-2).
        x = clamp(x, eps, 1.0 - eps);
    }
    Tensor one = ones_like(x);
    return log(x / (one - x));
}

auto signbit(const Tensor& input) -> Tensor {
    // Reflects the sign BIT, not value < 0: signbit(-0.0) is true and
    // signbit(negative NaN) is true, neither of which `lt(input, 0)` captures.
    // Dispatch the dedicated Signbit kernel (std::signbit semantics).
    return detail::unary_op<OpId::Signbit>(input);
}

auto float_power(const Tensor& base, const Tensor& exponent) -> Tensor {
    // Promote to Float64 for accuracy. The magnitude is computed stably as
    // |base|^exp = exp(exp * log|base|); we then restore the sign so that
    // negative bases behave like PyTorch's torch.float_power:
    //   - negative base, odd integer exponent  -> negative result
    //   - negative base, even integer exponent  -> positive result
    //   - negative base, non-integer exponent   -> NaN (real-valued result)
    auto base_f64 = base.to(DType::Float64);
    auto exp_f64 = exponent.to(DType::Float64);

    auto magnitude = exp(exp_f64 * log(abs(base_f64)));

    Tensor zero = zeros_like(base_f64);
    Tensor neg_mask = lt(base_f64, zero);  // base < 0

    // Integer exponent? round(exp) == exp
    Tensor exp_is_int = eq(round(exp_f64), exp_f64);
    // Odd integer? fmod(exp, 2) != 0  (only meaningful when exp is integral)
    Tensor two = full_like(exp_f64, 2.0);
    Tensor exp_mod2 = fmod(exp_f64, two);
    Tensor exp_is_odd = eq(abs(exp_mod2), full_like(exp_f64, 1.0));

    // Sign for negative bases: -1 for odd-integer exponent, +1 for even-integer,
    // NaN for non-integer exponent (real branch produces complex -> NaN).
    Tensor nan_t = full_like(base_f64, std::numeric_limits<double>::quiet_NaN());
    Tensor neg_one = full_like(base_f64, -1.0);
    Tensor pos_one = ones_like(base_f64);

    // sign_for_int: odd -> -1, even -> +1
    Tensor sign_for_int = where(exp_is_odd, neg_one, pos_one);
    // sign_for_neg_base: integer exponent -> sign_for_int, else NaN
    Tensor sign_for_neg_base = where(exp_is_int, sign_for_int, nan_t);

    // For non-negative bases the magnitude is already correct (sign = +1).
    Tensor sign = where(neg_mask, sign_for_neg_base, pos_one);

    return magnitude * sign;
}

auto xlog1py(const Tensor& x, const Tensor& y) -> Tensor {
    // x * log1p(y), with 0 * log1p(y) = 0
    Tensor log1p_y = log1p(y);
    Tensor result = x * log1p_y;
    Tensor zero = zeros_like(x);
    Tensor zero_mask = eq(x, zero);
    return where(zero_mask, zeros_like(result), result);
}

auto ldexp(const Tensor& x, const Tensor& n) -> Tensor {
    // ldexp(x, n) = x * 2^n = x * exp2(n)
    return x * exp2(n.to(x.dtype()));
}

auto isreal(const Tensor& input) -> Tensor {
    // For real dtypes, all elements are real
    // For complex dtypes, check if imaginary part is zero
    if (input.dtype() != DType::Complex64 && input.dtype() != DType::Complex128) {
        return full(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                    1.0f, DType::Bool, input.device());
    }
    // Complex: check imag(x) == 0
    Tensor im = imag(input);
    return (im == zeros_like(im));
}

auto isposinf(const Tensor& input) -> Tensor {
    return logical_and(isinf(input), (input > zeros_like(input)));
}

auto isneginf(const Tensor& input) -> Tensor {
    return logical_and(isinf(input), (input < zeros_like(input)));
}

// =========================================================================
// Normal distribution functions (composed from existing ops)
// =========================================================================

auto ndtr(const Tensor& input) -> Tensor {
    // Phi(x) = 0.5 * erfc(-x / sqrt(2))
    auto neg_x_over_sqrt2 = tenzor::mul(input, -0.7071067811865475);  // -1/sqrt(2)
    auto erfc_val = tenzor::erfc(neg_x_over_sqrt2);
    return tenzor::mul(erfc_val, 0.5);
}

auto ndtri(const Tensor& p) -> Tensor {
    // Inverse normal CDF (probit function):
    // ndtri(p) = sqrt(2) * erfinv(2p - 1)
    constexpr double SQRT2 = 1.4142135623730951;
    auto two_p_minus_one = tenzor::sub(tenzor::mul(p, 2.0), full_like(p, 1.0));
    return tenzor::mul(tenzor::erfinv(two_p_minus_one), SQRT2);
}

auto log_ndtr(const Tensor& input) -> Tensor {
    // Numerically stable log(Phi(x)):
    // For x >= 0 (large positive): log1p(-0.5 * erfc(x / sqrt(2)))
    // For x < 0 (large negative):  log(0.5 * erfc(-x / sqrt(2)))
    //
    // We compute both branches and select via where() for stability.
    constexpr double SQRT1_2 = 0.7071067811865475;

    // Positive branch: log1p(-0.5 * erfc(x / sqrt(2)))
    auto x_over_sqrt2 = tenzor::mul(input, SQRT1_2);
    auto erfc_pos = tenzor::erfc(x_over_sqrt2);
    auto half_erfc_pos = tenzor::mul(erfc_pos, 0.5);
    auto pos_branch = tenzor::log1p(tenzor::neg(half_erfc_pos));

    // Negative branch: log(0.5 * erfc(-x / sqrt(2)))
    auto neg_x_over_sqrt2 = tenzor::mul(input, -SQRT1_2);
    auto erfc_neg = tenzor::erfc(neg_x_over_sqrt2);
    auto half_erfc_neg = tenzor::mul(erfc_neg, 0.5);
    auto neg_branch = tenzor::log(half_erfc_neg);

    // Select: use negative branch where x < 0, positive branch otherwise
    auto zero = zeros_like(input);
    auto condition = ge(input, zero);
    return where(condition, pos_branch, neg_branch);
}

auto multigammaln(const Tensor& input, int64_t p) -> Tensor {
    // log(Gamma_p(a)) = p*(p-1)/4 * log(pi) + sum_{i=1}^{p} lgamma(a + (1-i)/2)
    if (p < 1) {
        throw std::invalid_argument("multigammaln: p must be >= 1, got " + std::to_string(p));
    }

    constexpr double LOG_PI = 1.1447298858494002;  // log(pi)
    double prefix = static_cast<double>(p) * static_cast<double>(p - 1) / 4.0 * LOG_PI;

    // Start with the prefix as a scalar added to the first lgamma term
    Tensor result = full_like(input, prefix);

    for (int64_t i = 1; i <= p; ++i) {
        double offset = (1.0 - static_cast<double>(i)) / 2.0;
        auto shifted = tenzor::add(input, offset);
        result = tenzor::add(result, tenzor::lgamma(shifted));
    }

    return result;
}

// =========================================================================
// Pairwise distance operations (dispatched to backend kernels)
// =========================================================================

auto pairwise_distance(const Tensor& x1, const Tensor& x2, double p) -> Tensor {
    if (x1.ndim() != 2 || x2.ndim() != 2) {
        throw std::invalid_argument("pairwise_distance: inputs must be 2D (N, D)");
    }
    if (x1.shape()[0] != x2.shape()[0] || x1.shape()[1] != x2.shape()[1]) {
        throw std::invalid_argument("pairwise_distance: x1 and x2 must have the same shape");
    }
    if (p < 0) {
        throw std::invalid_argument("pairwise_distance: p must be non-negative");
    }

    auto a = x1.contiguous();
    auto b = x2.contiguous();
    std::array<Tensor, 2> inputs = {a, b};
    NewOpAttributes attrs;
    attrs.set(AttrKey::DistP, p);
    return dispatch<OpId::PairwiseDistance>(inputs, attrs)[0];
}

auto pdist(const Tensor& input, double p) -> Tensor {
    if (input.ndim() != 2) {
        throw std::invalid_argument("pdist: input must be 2D (N, D)");
    }
    if (p < 0) {
        throw std::invalid_argument("pdist: p must be non-negative");
    }

    auto a = input.contiguous();
    std::array<Tensor, 1> inputs = {a};
    NewOpAttributes attrs;
    attrs.set(AttrKey::DistP, p);
    return dispatch<OpId::Pdist>(inputs, attrs)[0];
}

// =========================================================================
// Extended math continued
// =========================================================================

auto frexp(const Tensor& input) -> std::pair<Tensor, Tensor> {
    // frexp(x) = (mantissa, exponent) where x = mantissa * 2^exponent
    // mantissa in [0.5, 1.0), exponent is integer
    // Implemented via: exponent = floor(log2(|x|)) + 1, mantissa = x / 2^exponent
    Tensor abs_input = abs(input);
    // Avoid log2(0) by clamping to a tiny positive value. The clamp floor must
    // be below any representable magnitude for the input dtype so it only
    // affects exact zeros (already masked out below) and never rescales a
    // legitimately small value. A fixed 1e-38f is the Float32 smallest-normal
    // scale; for Float64 it would force magnitudes < 1e-38 (e.g. 1e-300) up to
    // 1e-38, yielding exponent ~-126 instead of ~-996. Pick the floor from the
    // compute dtype's smallest denormal instead.
    double clamp_floor;
    switch (input.dtype()) {
        case DType::Float64:
            clamp_floor = std::numeric_limits<double>::denorm_min();
            break;
        default:
            // Float32 / Float16 / BFloat16 all widen into the Float32 range here.
            clamp_floor = static_cast<double>(std::numeric_limits<float>::denorm_min());
            break;
    }
    Tensor safe_abs = clamp_min(abs_input, clamp_floor);
    Tensor log2_val = log2(safe_abs);
    Tensor exponent_f = floor(log2_val) + ones_like(log2_val);
    // For zero input, exponent should be 0
    Tensor is_zero = eq(input, zeros_like(input));
    exponent_f = where(is_zero, zeros_like(exponent_f), exponent_f);

    // mantissa = x * 2^(-exponent)
    Tensor mantissa = input * exp2(neg(exponent_f));

    // log2-based reconstruction can land mantissa exactly on a boundary for
    // values near a power of two (e.g. log2(4.0) rounding to 1.9999999 yields
    // floor=1, exponent=2, mantissa=1.0). std::frexp guarantees the mantissa
    // magnitude lies in [0.5, 1.0); correct the boundary cases so we match.
    //   |mantissa| >= 1.0  -> mantissa /= 2, exponent += 1
    //   0 < |mantissa| < 0.5 -> mantissa *= 2, exponent -= 1
    Tensor abs_mant = abs(mantissa);
    Tensor too_big = ge(abs_mant, full_like(abs_mant, 1.0));
    mantissa = where(too_big, mantissa * full_like(mantissa, 0.5), mantissa);
    exponent_f = where(too_big, exponent_f + ones_like(exponent_f), exponent_f);

    abs_mant = abs(mantissa);
    Tensor too_small = logical_and(lt(abs_mant, full_like(abs_mant, 0.5)),
                                   logical_not(is_zero));
    mantissa = where(too_small, mantissa * full_like(mantissa, 2.0), mantissa);
    exponent_f = where(too_small, exponent_f - ones_like(exponent_f), exponent_f);

    // For zero input, exponent and mantissa should be 0
    exponent_f = where(is_zero, zeros_like(exponent_f), exponent_f);
    Tensor exponent = exponent_f.to(DType::Int32);
    // For zero input, mantissa should be 0
    mantissa = where(is_zero, zeros_like(mantissa), mantissa);

    return {mantissa, exponent};
}

} // namespace tenzor
