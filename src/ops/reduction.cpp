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

auto nanvar(const Tensor& input, std::optional<int64_t> dim, bool keepdim, int64_t correction) -> Tensor {
    // Composition: nanvar = sum((x - nanmean(x))^2, ignoring NaN) / (N_valid - correction)
    Tensor mean = nanmean(input, dim, /*keepdim=*/true);
    Tensor diff = input - mean;
    // Zero out NaN positions so they don't contribute
    Tensor nan_mask = isnan(input);
    Tensor diff_sq = diff * diff;
    // Replace NaN-originated values with 0
    Tensor clean = where(nan_mask, zeros_like(diff_sq), diff_sq);
    Tensor sum_sq = nansum(clean, dim, keepdim);
    // Count valid (non-NaN) elements
    Tensor valid_mask = logical_not(nan_mask);
    Tensor count = sum(valid_mask.to(input.dtype()), dim, keepdim);
    Tensor denom = count - static_cast<float>(correction);
    return sum_sq / denom;
}

auto nanstd(const Tensor& input, std::optional<int64_t> dim, bool keepdim, int64_t correction) -> Tensor {
    return sqrt(nanvar(input, dim, keepdim, correction));
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

auto cov(const Tensor& input, int64_t correction) -> Tensor {
    // For 1D input: compute variance
    if (input.ndim() == 1) {
        auto m = tenzor::mean(input);
        auto diff = tenzor::sub(input, m);
        auto n = static_cast<double>(input.shape()[0] - correction);
        auto dot_result = tenzor::dot(diff, diff);
        return tenzor::div(dot_result, tenzor::full({}, n, input.dtype(), input.device()));
    }

    // For 2D input (N, M): each row is a variable, each column is an observation
    if (input.ndim() != 2) {
        throw std::invalid_argument("cov: input must be 1D or 2D");
    }

    int64_t M = input.shape()[1];
    auto m_val = static_cast<double>(M - correction);

    // Center: subtract row means
    auto row_means = tenzor::mean(input, 1, true);  // (N, 1)
    auto centered = tenzor::sub(input, row_means);   // (N, M)

    // Cov = centered @ centered^T / (M - correction)
    auto centered_t = tenzor::transpose(centered, 0, 1);  // (M, N)
    auto product = tenzor::matmul(centered, centered_t);    // (N, N)
    return tenzor::div(product, tenzor::full({}, m_val, input.dtype(), input.device()));
}

// =========================================================================
// Numerical integration and gradient
// =========================================================================

auto trapezoid(const Tensor& y, const Tensor& x, int64_t dim) -> Tensor {
    if (y.ndim() == 0) {
        throw std::invalid_argument("trapezoid: input must be at least 1D");
    }
    auto yc = y.contiguous();
    auto xc = x.contiguous();
    std::array<Tensor, 2> inputs = {yc, xc};
    NewOpAttributes attrs;
    attrs.set(AttrKey::Dim, dim);
    return dispatch(OpId::Trapezoid, inputs, attrs)[0];
}

auto trapezoid(const Tensor& y, double dx, int64_t dim) -> Tensor {
    if (y.ndim() == 0) {
        throw std::invalid_argument("trapezoid: input must be at least 1D");
    }
    auto yc = y.contiguous();
    std::array<Tensor, 1> inputs = {yc};
    NewOpAttributes attrs;
    attrs.set(AttrKey::Dim, dim);
    attrs.set(AttrKey::Dx, dx);
    return dispatch(OpId::Trapezoid, inputs, attrs)[0];
}

auto cumulative_trapezoid(const Tensor& y, const Tensor& x, int64_t dim) -> Tensor {
    if (y.ndim() == 0) {
        throw std::invalid_argument("cumulative_trapezoid: input must be at least 1D");
    }
    auto yc = y.contiguous();
    auto xc = x.contiguous();
    std::array<Tensor, 2> inputs = {yc, xc};
    NewOpAttributes attrs;
    attrs.set(AttrKey::Dim, dim);
    return dispatch(OpId::CumulativeTrapezoid, inputs, attrs)[0];
}

auto cumulative_trapezoid(const Tensor& y, double dx, int64_t dim) -> Tensor {
    if (y.ndim() == 0) {
        throw std::invalid_argument("cumulative_trapezoid: input must be at least 1D");
    }
    auto yc = y.contiguous();
    std::array<Tensor, 1> inputs = {yc};
    NewOpAttributes attrs;
    attrs.set(AttrKey::Dim, dim);
    attrs.set(AttrKey::Dx, dx);
    return dispatch(OpId::CumulativeTrapezoid, inputs, attrs)[0];
}

auto gradient(const Tensor& input, int64_t dim, double spacing) -> Tensor {
    if (input.ndim() == 0) {
        throw std::invalid_argument("gradient: input must be at least 1D");
    }
    auto ic = input.contiguous();
    std::array<Tensor, 1> inputs = {ic};
    NewOpAttributes attrs;
    attrs.set(AttrKey::Dim, dim);
    attrs.set(AttrKey::Spacing, spacing);
    return dispatch(OpId::NumericalGradient, inputs, attrs)[0];
}

// =========================================================================
// Covariance / Correlation
// =========================================================================

auto corrcoef(const Tensor& input) -> Tensor {
    auto c = tenzor::cov(input);

    if (input.ndim() == 1) {
        return tenzor::full({}, 1.0, input.dtype(), input.device());
    }

    // Normalize: corr[i,j] = cov[i,j] / (std[i] * std[j])
    auto diag_vals = tenzor::diag(c);              // (N,)
    auto stds = tenzor::sqrt(diag_vals);           // (N,)
    auto stds_col = tenzor::reshape(stds, {-1, 1}); // (N, 1)
    auto stds_row = tenzor::reshape(stds, {1, -1}); // (1, N)
    auto norm = tenzor::matmul(stds_col, stds_row);  // (N, N)
    return tenzor::div(c, norm);
}

auto segment_reduce(const Tensor& data, const Tensor& offsets,
                    const std::string& reduce, int64_t axis) -> Tensor {
    TENZOR_PROFILE_RANGE("segment_reduce");
    if (offsets.ndim() != 1) {
        throw std::invalid_argument("segment_reduce: offsets must be 1-D");
    }
    if (offsets.numel() < 2) {
        throw std::invalid_argument("segment_reduce: offsets must have at least 2 elements");
    }
    if (reduce != "sum" && reduce != "mean" && reduce != "max" &&
        reduce != "min" && reduce != "prod") {
        throw std::invalid_argument("segment_reduce: unsupported reduce mode '" + reduce + "'");
    }

    auto data_cont = data.contiguous();
    auto offsets_cont = offsets.to(DType::Int64).contiguous();

    NewOpAttributes attrs;
    attrs.set(AttrKey::Dim, axis);
    attrs.set(AttrKey::Reduction, reduce);
    std::vector<Tensor> inputs = {data_cont, offsets_cont};
    return dispatch(OpId::SegmentReduce, inputs, attrs)[0];
}

} // namespace tenzor
