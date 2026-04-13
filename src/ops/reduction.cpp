#include "tenzor/ops/reduction.hpp"
#include "tenzor/backend/fast_dispatch.hpp"
#include "tenzor/backend/op_attributes.hpp"
#include "tenzor/ops/op_id.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/transform.hpp"
#include "tenzor/ops/indexing.hpp"
#include "tenzor/utils/profiling.hpp"

namespace tenzor {

// Type promotion helpers matching PyTorch semantics:
// - Integer sum/prod accumulate in Int64 to prevent overflow
// - mean/var/std/norm on integers produce Float32 results
static bool is_small_int_dtype(DType dt) {
    return dt == DType::Int8 || dt == DType::UInt8 || dt == DType::Int16 ||
           dt == DType::Int32 || dt == DType::UInt16 || dt == DType::UInt32 || dt == DType::Bool;
}

static bool is_integer_dtype(DType dt) {
    return dt == DType::Int8 || dt == DType::UInt8 || dt == DType::Int16 ||
           dt == DType::Int32 || dt == DType::Int64 || dt == DType::Bool;
}

auto sum(const Tensor& input, std::optional<int64_t> dim, bool keepdim) -> Tensor {
    TENZOR_PROFILE_RANGE("sum");
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

auto has_inf_nan(const Tensor& input) -> Tensor {
    NewOpAttributes attrs;
    std::vector<Tensor> inputs = {input};
    return dispatch(OpId::HasInfNan, inputs, attrs)[0];
}

auto logsumexp(const Tensor& input, int64_t dim, bool keepdim) -> Tensor {
    // Empty tensor: return empty with appropriate shape
    if (input.numel() == 0) {
        // For empty input, result shape collapses dim to 1 (keepdim) or removes it
        auto shape = std::vector<int64_t>(input.shape().begin(), input.shape().end());
        if (dim < 0) dim += static_cast<int64_t>(shape.size());
        if (keepdim) {
            shape[dim] = 1;
        } else {
            shape.erase(shape.begin() + dim);
        }
        // logsumexp of empty set is -inf (log(0))
        return full(shape, -std::numeric_limits<double>::infinity(), input.dtype(), input.device());
    }

    // Try fused kernel dispatch (avoids multiple passes over data)
    if (is_op_supported(OpId::LogSumExp, input.device().type)) {
        NewOpAttributes attrs;
        attrs.set(AttrKey::Dim, dim);
        attrs.set(AttrKey::Keepdim, keepdim);
        std::vector<Tensor> inputs = {input};
        return dispatch(OpId::LogSumExp, inputs, attrs)[0];
    }

    // Composite fallback: numerically stable logsumexp
    // log(sum(exp(x))) = max(x) + log(sum(exp(x - max(x))))
    // Subtracting the max prevents overflow in exp() for large values.
    auto max_val = tenzor::max(input, dim, /*keepdim=*/true);  // keepdim=true for broadcasting
    auto shifted = input - max_val;                             // subtract max for stability
    auto exp_shifted = tenzor::exp(shifted);
    auto sum_exp = tenzor::sum(exp_shifted, dim, keepdim);
    auto log_sum = tenzor::log(sum_exp);
    if (!keepdim) {
        max_val = tenzor::squeeze(max_val, dim);
    }
    auto result = max_val + log_sum;
    // Where max_val was -inf (all elements in a slice were -inf), the subtraction
    // -inf - (-inf) produces NaN, propagating through exp/sum/log.
    // The correct result is -inf (and +inf for +inf max_val).
    auto inf_mask = tenzor::isinf(max_val);
    return tenzor::where(inf_mask, max_val, result);
}

auto histogram(const Tensor& input, int64_t bins, double min_val, double max_val)
    -> std::pair<Tensor, Tensor> {
    if (bins <= 0) {
        throw std::invalid_argument("histogram: bins must be positive");
    }

    auto inp = input.contiguous();
    std::array<Tensor, 1> inputs = {inp};
    NewOpAttributes attrs;
    attrs.set(AttrKey::NumBins, bins);
    attrs.set(AttrKey::Min, min_val);
    attrs.set(AttrKey::Max, max_val);
    auto results = dispatch<OpId::Histogram>(inputs, attrs);
    return {results[0], results[1]};
}

auto count_nonzero(const Tensor& input, std::optional<int64_t> dim) -> Tensor {
    std::array<Tensor, 1> inputs = {input.contiguous()};
    NewOpAttributes attrs;
    if (dim.has_value()) attrs.set(AttrKey::Dim, dim.value());
    return dispatch<OpId::CountNonzero>(inputs, attrs)[0];
}

auto nansum(const Tensor& input, std::optional<int64_t> dim, bool keepdim) -> Tensor {
    std::array<Tensor, 1> inputs = {input.contiguous()};
    NewOpAttributes attrs;
    if (dim.has_value()) attrs.set(AttrKey::Dim, dim.value());
    attrs.set(AttrKey::Keepdim, keepdim);
    return dispatch<OpId::Nansum>(inputs, attrs)[0];
}

auto nanmean(const Tensor& input, std::optional<int64_t> dim, bool keepdim) -> Tensor {
    std::array<Tensor, 1> inputs = {input.contiguous()};
    NewOpAttributes attrs;
    if (dim.has_value()) attrs.set(AttrKey::Dim, dim.value());
    attrs.set(AttrKey::Keepdim, keepdim);
    return dispatch<OpId::Nanmean>(inputs, attrs)[0];
}

auto aminmax(const Tensor& input, std::optional<int64_t> dim, bool keepdim) -> std::pair<Tensor, Tensor> {
    std::array<Tensor, 1> inputs = {input.contiguous()};
    NewOpAttributes attrs;
    if (dim.has_value()) attrs.set(AttrKey::Dim, dim.value());
    attrs.set(AttrKey::Keepdim, keepdim);
    auto results = dispatch<OpId::Aminmax>(inputs, attrs);
    return {results[0], results[1]};
}

// =========================================================================
// Fused reductions (compositions, no new OpIds)
// =========================================================================

auto std_mean(const Tensor& input, std::optional<int64_t> dim,
              bool keepdim, bool unbiased) -> std::pair<Tensor, Tensor> {
    auto m = mean(input, dim, keepdim);
    auto s = tenzor::std(input, dim, keepdim, unbiased);
    return {s, m};
}

auto var_mean(const Tensor& input, std::optional<int64_t> dim,
              bool keepdim, bool unbiased) -> std::pair<Tensor, Tensor> {
    auto m = mean(input, dim, keepdim);
    auto v = var(input, dim, keepdim, unbiased);
    return {v, m};
}

auto dist(const Tensor& a, const Tensor& b, float p) -> Tensor {
    auto diff = tenzor::sub(a, b);
    return norm(diff, p);
}

// =========================================================================
// New reduction operations for PyTorch parity
// =========================================================================

auto cummax(const Tensor& input, int64_t dim) -> std::pair<Tensor, Tensor> {
    std::array<Tensor, 1> inputs = {input.contiguous()};
    NewOpAttributes attrs;
    attrs.set(AttrKey::Dim, dim);
    auto results = dispatch<OpId::CumMax>(inputs, attrs);
    return {results[0], results[1]};
}

auto cummin(const Tensor& input, int64_t dim) -> std::pair<Tensor, Tensor> {
    std::array<Tensor, 1> inputs = {input.contiguous()};
    NewOpAttributes attrs;
    attrs.set(AttrKey::Dim, dim);
    auto results = dispatch<OpId::CumMin>(inputs, attrs);
    return {results[0], results[1]};
}

auto isin(const Tensor& elements, const Tensor& test_elements) -> Tensor {
    std::array<Tensor, 2> inputs = {elements.contiguous(), test_elements.contiguous()};
    return dispatch<OpId::Isin>(inputs)[0];
}

auto kthvalue(const Tensor& input, int64_t k, int64_t dim, bool keepdim) -> std::pair<Tensor, Tensor> {
    std::array<Tensor, 1> inputs = {input.contiguous()};
    NewOpAttributes attrs;
    attrs.set(AttrKey::Dim, dim);
    attrs.set(AttrKey::Keepdim, keepdim);
    attrs.set(AttrKey::K, k);
    auto results = dispatch<OpId::Kthvalue>(inputs, attrs);
    return {results[0], results[1]};
}

auto fmax(const Tensor& a, const Tensor& b) -> Tensor {
    std::array<Tensor, 2> inputs = {a.contiguous(), b.contiguous()};
    return dispatch<OpId::Fmax>(inputs)[0];
}

auto fmin(const Tensor& a, const Tensor& b) -> Tensor {
    std::array<Tensor, 2> inputs = {a.contiguous(), b.contiguous()};
    return dispatch<OpId::Fmin>(inputs)[0];
}

auto quantile(const Tensor& input, double q, std::optional<int64_t> dim,
              bool keepdim) -> Tensor {
    std::array<Tensor, 1> inputs = {input.contiguous()};
    NewOpAttributes attrs;
    attrs.set(AttrKey::Alpha, q);  // reuse Alpha for the quantile value
    if (dim.has_value()) attrs.set(AttrKey::Dim, dim.value());
    attrs.set(AttrKey::Keepdim, keepdim);
    return dispatch<OpId::Quantile>(inputs, attrs)[0];
}

auto nanquantile(const Tensor& input, double q, std::optional<int64_t> dim,
                 bool keepdim) -> Tensor {
    std::array<Tensor, 1> inputs = {input.contiguous()};
    NewOpAttributes attrs;
    attrs.set(AttrKey::Alpha, q);
    if (dim.has_value()) attrs.set(AttrKey::Dim, dim.value());
    attrs.set(AttrKey::Keepdim, keepdim);
    return dispatch<OpId::Nanquantile>(inputs, attrs)[0];
}

auto nanmedian(const Tensor& input, std::optional<int64_t> dim) -> Tensor {
    std::array<Tensor, 1> inputs = {input.contiguous()};
    NewOpAttributes attrs;
    if (dim.has_value()) attrs.set(AttrKey::Dim, dim.value());
    return dispatch<OpId::Nanmedian>(inputs, attrs)[0];
}

auto histc(const Tensor& input, int64_t bins, double min_val, double max_val) -> Tensor {
    std::array<Tensor, 1> inputs = {input.contiguous()};
    NewOpAttributes attrs;
    attrs.set(AttrKey::N, bins);
    attrs.set(AttrKey::Start, static_cast<int64_t>(0));  // reuse for min/max
    attrs.set(AttrKey::Alpha, min_val);
    attrs.set(AttrKey::Beta, max_val);
    return dispatch<OpId::Histc>(inputs, attrs)[0];
}

auto unique_consecutive(const Tensor& input, bool return_inverse, bool return_counts,
                        std::optional<int64_t> dim)
    -> std::tuple<Tensor, Tensor, Tensor> {
    std::array<Tensor, 1> inputs = {input.contiguous()};
    NewOpAttributes attrs;
    if (dim.has_value()) attrs.set(AttrKey::Dim, dim.value());
    attrs.set(AttrKey::Keepdim, return_inverse);  // reuse for return_inverse flag
    auto results = dispatch<OpId::UniqueConsecutive>(inputs, attrs);
    // results[0] = unique values, results[1] = inverse indices, results[2] = counts
    Tensor inverse = (results.size() > 1 && return_inverse) ? results[1] : Tensor();
    Tensor counts = (results.size() > 2 && return_counts) ? results[2] : Tensor();
    return {results[0], inverse, counts};
}

} // namespace tenzor
