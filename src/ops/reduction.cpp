#include "tenzor/ops/reduction.hpp"
#include "tenzor/backend/fast_dispatch.hpp"
#include "tenzor/backend/op_attributes.hpp"
#include "tenzor/ops/op_id.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/transform.hpp"

namespace tenzor {

// Type promotion helpers matching PyTorch semantics:
// - Integer sum/prod accumulate in Int64 to prevent overflow
// - mean/var/std/norm on integers produce Float32 results
static bool is_small_int_dtype(DType dt) {
    return dt == DType::Int8 || dt == DType::UInt8 || dt == DType::Int16 || dt == DType::Bool;
}

static bool is_integer_dtype(DType dt) {
    return dt == DType::Int8 || dt == DType::UInt8 || dt == DType::Int16 ||
           dt == DType::Int32 || dt == DType::Int64 || dt == DType::Bool;
}

auto sum(const Tensor& input, std::optional<int64_t> dim, bool keepdim) -> Tensor {
    // Promote small integer types to Int64 to prevent overflow
    Tensor promoted = is_small_int_dtype(input.dtype()) ? input.to(DType::Int64) : input;
    NewOpAttributes attrs;
    if (dim.has_value()) attrs.set(AttrKey::Dim, *dim);
    attrs.set(AttrKey::Keepdim, keepdim);
    std::vector<Tensor> inputs = {promoted};
    return dispatch(OpId::Sum, inputs, attrs)[0];
}

auto mean(const Tensor& input, std::optional<int64_t> dim, bool keepdim) -> Tensor {
    // Integer/bool mean must produce floating-point results
    Tensor promoted = is_integer_dtype(input.dtype()) ? input.to(DType::Float32) : input;
    NewOpAttributes attrs;
    if (dim.has_value()) attrs.set(AttrKey::Dim, *dim);
    attrs.set(AttrKey::Keepdim, keepdim);
    std::vector<Tensor> inputs = {promoted};
    return dispatch(OpId::Mean, inputs, attrs)[0];
}

auto max(const Tensor& input, std::optional<int64_t> dim, bool keepdim) -> Tensor {
    NewOpAttributes attrs;
    if (dim.has_value()) attrs.set(AttrKey::Dim, *dim);
    attrs.set(AttrKey::Keepdim, keepdim);
    std::vector<Tensor> inputs = {input};
    return dispatch(OpId::Max, inputs, attrs)[0];
}

auto min(const Tensor& input, std::optional<int64_t> dim, bool keepdim) -> Tensor {
    NewOpAttributes attrs;
    if (dim.has_value()) attrs.set(AttrKey::Dim, *dim);
    attrs.set(AttrKey::Keepdim, keepdim);
    std::vector<Tensor> inputs = {input};
    return dispatch(OpId::Min, inputs, attrs)[0];
}

auto argmax(const Tensor& input, std::optional<int64_t> dim, bool keepdim) -> Tensor {
    NewOpAttributes attrs;
    if (dim.has_value()) attrs.set(AttrKey::Dim, *dim);
    attrs.set(AttrKey::Keepdim, keepdim);
    std::vector<Tensor> inputs = {input};
    return dispatch(OpId::ArgMax, inputs, attrs)[0];
}

auto argmin(const Tensor& input, std::optional<int64_t> dim, bool keepdim) -> Tensor {
    NewOpAttributes attrs;
    if (dim.has_value()) attrs.set(AttrKey::Dim, *dim);
    attrs.set(AttrKey::Keepdim, keepdim);
    std::vector<Tensor> inputs = {input};
    return dispatch(OpId::ArgMin, inputs, attrs)[0];
}

auto argsort(const Tensor& input, int64_t dim, bool descending) -> Tensor {
    NewOpAttributes attrs;
    attrs.set(AttrKey::Dim, dim);
    attrs.set(AttrKey::Descending, descending);
    std::vector<Tensor> inputs = {input};
    return dispatch(OpId::ArgSort, inputs, attrs)[0];
}

auto prod(const Tensor& input, std::optional<int64_t> dim, bool keepdim) -> Tensor {
    // Promote small integer types to Int64 to prevent overflow
    Tensor promoted = is_small_int_dtype(input.dtype()) ? input.to(DType::Int64) : input;
    NewOpAttributes attrs;
    if (dim.has_value()) attrs.set(AttrKey::Dim, *dim);
    attrs.set(AttrKey::Keepdim, keepdim);
    std::vector<Tensor> inputs = {promoted};
    return dispatch(OpId::Prod, inputs, attrs)[0];
}

auto std(const Tensor& input, std::optional<int64_t> dim, bool keepdim, bool unbiased) -> Tensor {
    // Integer/bool std must produce floating-point results
    Tensor promoted = is_integer_dtype(input.dtype()) ? input.to(DType::Float32) : input;
    NewOpAttributes attrs;
    if (dim.has_value()) attrs.set(AttrKey::Dim, *dim);
    attrs.set(AttrKey::Keepdim, keepdim);
    attrs.set(AttrKey::Unbiased, unbiased);
    std::vector<Tensor> inputs = {promoted};
    return dispatch(OpId::Std, inputs, attrs)[0];
}

auto var(const Tensor& input, std::optional<int64_t> dim, bool keepdim, bool unbiased) -> Tensor {
    // Integer/bool var must produce floating-point results
    Tensor promoted = is_integer_dtype(input.dtype()) ? input.to(DType::Float32) : input;
    NewOpAttributes attrs;
    if (dim.has_value()) attrs.set(AttrKey::Dim, *dim);
    attrs.set(AttrKey::Keepdim, keepdim);
    attrs.set(AttrKey::Unbiased, unbiased);
    std::vector<Tensor> inputs = {promoted};
    return dispatch(OpId::Var, inputs, attrs)[0];
}

auto norm(const Tensor& input, float p, std::optional<int64_t> dim, bool keepdim) -> Tensor {
    // Integer/bool norm must produce floating-point results
    Tensor promoted = is_integer_dtype(input.dtype()) ? input.to(DType::Float32) : input;
    NewOpAttributes attrs;
    attrs.set(AttrKey::P, static_cast<double>(p));
    if (dim.has_value()) attrs.set(AttrKey::Dim, *dim);
    attrs.set(AttrKey::Keepdim, keepdim);
    std::vector<Tensor> inputs = {promoted};
    return dispatch(OpId::Norm, inputs, attrs)[0];
}

auto any(const Tensor& input, std::optional<int64_t> dim, bool keepdim) -> Tensor {
    NewOpAttributes attrs;
    if (dim.has_value()) attrs.set(AttrKey::Dim, *dim);
    attrs.set(AttrKey::Keepdim, keepdim);
    std::vector<Tensor> inputs = {input};
    return dispatch(OpId::Any, inputs, attrs)[0];
}

auto all(const Tensor& input, std::optional<int64_t> dim, bool keepdim) -> Tensor {
    NewOpAttributes attrs;
    if (dim.has_value()) attrs.set(AttrKey::Dim, *dim);
    attrs.set(AttrKey::Keepdim, keepdim);
    std::vector<Tensor> inputs = {input};
    return dispatch(OpId::All, inputs, attrs)[0];
}

auto logsumexp(const Tensor& input, int64_t dim, bool keepdim) -> Tensor {
    // Numerically stable logsumexp: log(sum(exp(x))) = max(x) + log(sum(exp(x - max(x))))
    // Subtracting the max prevents overflow in exp() for large values.
    auto max_val = tenzor::max(input, dim, /*keepdim=*/true);  // keepdim=true for broadcasting
    auto shifted = input - max_val;                             // subtract max for stability
    auto exp_shifted = tenzor::exp(shifted);
    auto sum_exp = tenzor::sum(exp_shifted, dim, keepdim);
    auto log_sum = tenzor::log(sum_exp);
    if (!keepdim) {
        max_val = tenzor::squeeze(max_val, dim);
    }
    return max_val + log_sum;
}

} // namespace tenzor
