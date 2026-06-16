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

// audit-5 Y.7: normalise a (possibly negative) reduction dim against ndim so
// that ROCm / OneAPI / CUDA backends (which use the attribute raw) don't
// underflow on `shape[dim]`. Mirrors the inline prologue used by cummax/cummin/
// kthvalue. The op name is threaded through for a precise out-of-range message.
static int64_t normalize_reduce_dim(int64_t dim, int64_t ndim, const char* op) {
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) {
        throw std::out_of_range(std::string(op) + ": dim out of range");
    }
    return dim;
}

auto sum(const Tensor& input, std::optional<int64_t> dim, bool keepdim) -> Tensor {
    TENZOR_PROFILE_RANGE("sum");
    // Promote small integer types to Int64 to prevent overflow
    Tensor promoted = is_small_int_dtype(input.dtype()) ? input.to(DType::Int64) : input;
    NewOpAttributes attrs;
    if (dim.has_value()) {
        attrs.set(AttrKey::Dim, normalize_reduce_dim(*dim, static_cast<int64_t>(input.shape().size()), "sum"));
    }
    attrs.set(AttrKey::Keepdim, keepdim);
    std::vector<Tensor> inputs = {promoted};
    return dispatch(OpId::Sum, inputs, attrs)[0];
}

auto mean(const Tensor& input, std::optional<int64_t> dim, bool keepdim) -> Tensor {
    // Integer/bool mean must produce floating-point results
    Tensor promoted = is_integer_dtype(input.dtype()) ? input.to(DType::Float32) : input;
    NewOpAttributes attrs;
    if (dim.has_value()) {
        attrs.set(AttrKey::Dim, normalize_reduce_dim(*dim, static_cast<int64_t>(input.shape().size()), "mean"));
    }
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
    if (dim.has_value()) {
        attrs.set(AttrKey::Dim, normalize_reduce_dim(*dim, static_cast<int64_t>(input.shape().size()), "std"));
    }
    attrs.set(AttrKey::Keepdim, keepdim);
    // Set BOTH AttrKey::Unbiased (bool) and AttrKey::Correction (int):
    // the CPU kernel reads Correction with default 1 (→ N-1), while the
    // GPU backends read Unbiased. Without both, cross-backend parity
    // silently diverges whenever the caller passes unbiased=false —
    // CPU stays at correction=1 and returns the unbiased estimator,
    // while the GPU returns the biased one.
    attrs.set(AttrKey::Unbiased, unbiased);
    attrs.set(AttrKey::Correction, static_cast<int64_t>(unbiased ? 1 : 0));
    std::vector<Tensor> inputs = {promoted};
    return dispatch(OpId::Std, inputs, attrs)[0];
}

auto var(const Tensor& input, std::optional<int64_t> dim, bool keepdim, bool unbiased) -> Tensor {
    // Integer/bool var must produce floating-point results
    Tensor promoted = is_integer_dtype(input.dtype()) ? input.to(DType::Float32) : input;
    NewOpAttributes attrs;
    if (dim.has_value()) {
        attrs.set(AttrKey::Dim, normalize_reduce_dim(*dim, static_cast<int64_t>(input.shape().size()), "var"));
    }
    attrs.set(AttrKey::Keepdim, keepdim);
    attrs.set(AttrKey::Unbiased, unbiased);
    attrs.set(AttrKey::Correction, static_cast<int64_t>(unbiased ? 1 : 0));
    std::vector<Tensor> inputs = {promoted};
    return dispatch(OpId::Var, inputs, attrs)[0];
}

auto norm(const Tensor& input, float p, std::optional<int64_t> dim, bool keepdim) -> Tensor {
    // Complex norm = norm of the elementwise magnitudes (PyTorch semantics).
    // Reduce to the real magnitude here so every backend computes it uniformly
    // (the per-backend kernels operate on real dtypes only).
    Tensor promoted = input;
    if (input.dtype() == DType::Complex64 || input.dtype() == DType::Complex128) {
        promoted = tenzor::abs(input);  // |z| -> Float32 / Float64
    } else if (is_integer_dtype(input.dtype())) {
        promoted = input.to(DType::Float32);
    }
    NewOpAttributes attrs;
    attrs.set(AttrKey::P, static_cast<double>(p));
    if (dim.has_value()) {
        attrs.set(AttrKey::Dim, normalize_reduce_dim(*dim, static_cast<int64_t>(input.shape().size()), "norm"));
    }
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

auto logsumexp(const Tensor& input_arg, int64_t dim, bool keepdim) -> Tensor {
    // logsumexp produces a floating-point result. Promote integer/bool inputs to
    // a float compute dtype so the (possibly -inf) result is representable and the
    // exp/log composite path operates in floating arithmetic (matches PyTorch).
    Tensor input = is_floating_type(input_arg.dtype())
                       ? input_arg
                       : input_arg.to(DType::Float32);

    // Empty tensor: return empty with appropriate shape
    if (input.numel() == 0) {
        // For empty input, result shape collapses dim to 1 (keepdim) or removes it
        auto shape = std::vector<int64_t>(input.shape().begin(), input.shape().end());
        if (dim < 0) dim += static_cast<int64_t>(shape.size());
        if (dim < 0 || dim >= static_cast<int64_t>(shape.size())) {
            throw std::out_of_range("logsumexp: dim out of range");
        }
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

auto histogramdd(const Tensor& input, std::vector<int64_t> bins,
                 std::optional<std::vector<std::pair<double,double>>> ranges,
                 bool density) -> std::pair<Tensor, std::vector<Tensor>> {
    if (input.dim() != 2) {
        throw std::invalid_argument("histogramdd: input must be 2-D (N, D)");
    }
    int64_t D = input.shape()[1];
    if (static_cast<int64_t>(bins.size()) != D) {
        throw std::invalid_argument("histogramdd: bins length must match input dimension D");
    }
    for (auto b : bins) {
        if (b <= 0) {
            throw std::invalid_argument("histogramdd: all bin counts must be positive");
        }
    }

    auto inp = input.contiguous();
    std::array<Tensor, 1> inputs = {inp};
    NewOpAttributes attrs;

    // Encode bins as comma-separated string
    std::string bins_str;
    for (size_t i = 0; i < bins.size(); ++i) {
        if (i > 0) bins_str += ',';
        bins_str += std::to_string(bins[i]);
    }
    attrs.set(AttrKey::BinsList, bins_str);
    attrs.set(AttrKey::Density, density);

    // Encode ranges as comma-separated pairs: min0,max0,min1,max1,...
    if (ranges.has_value()) {
        const auto& r = ranges.value();
        if (static_cast<int64_t>(r.size()) != D) {
            throw std::invalid_argument("histogramdd: ranges length must match input dimension D");
        }
        std::string ranges_str;
        for (size_t i = 0; i < r.size(); ++i) {
            if (i > 0) ranges_str += ',';
            ranges_str += std::to_string(r[i].first) + ',' + std::to_string(r[i].second);
        }
        attrs.set(AttrKey::RangesList, ranges_str);
    }

    auto results = dispatch<OpId::Histogramdd>(inputs, attrs);
    // results[0] = counts, results[1..D] = edge tensors
    std::vector<Tensor> edges(results.begin() + 1, results.end());
    return {results[0], edges};
}

auto count_nonzero(const Tensor& input, std::optional<int64_t> dim) -> Tensor {
    std::array<Tensor, 1> inputs = {input.contiguous()};
    NewOpAttributes attrs;
    if (dim.has_value()) {
        attrs.set(AttrKey::Dim, normalize_reduce_dim(dim.value(), static_cast<int64_t>(input.shape().size()), "count_nonzero"));
    }
    return dispatch<OpId::CountNonzero>(inputs, attrs)[0];
}

auto nansum(const Tensor& input, std::optional<int64_t> dim, bool keepdim) -> Tensor {
    std::array<Tensor, 1> inputs = {input.contiguous()};
    NewOpAttributes attrs;
    if (dim.has_value()) {
        attrs.set(AttrKey::Dim, normalize_reduce_dim(dim.value(), static_cast<int64_t>(input.shape().size()), "nansum"));
    }
    attrs.set(AttrKey::Keepdim, keepdim);
    return dispatch<OpId::Nansum>(inputs, attrs)[0];
}

auto nanmean(const Tensor& input, std::optional<int64_t> dim, bool keepdim) -> Tensor {
    std::array<Tensor, 1> inputs = {input.contiguous()};
    NewOpAttributes attrs;
    if (dim.has_value()) {
        attrs.set(AttrKey::Dim, normalize_reduce_dim(dim.value(), static_cast<int64_t>(input.shape().size()), "nanmean"));
    }
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
    // Count valid (non-NaN) elements. Accumulate the count in a wide float
    // dtype (Float64 for Float64 input, Float32 otherwise) so half-precision
    // inputs don't saturate the count — mirrors mean's integer->Float32
    // promotion convention.
    DType count_dtype = (input.dtype() == DType::Float64) ? DType::Float64 : DType::Float32;
    Tensor valid_mask = logical_not(nan_mask);
    Tensor count = sum(valid_mask.to(count_dtype), dim, keepdim);
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
    // audit-5 Y.7: normalise negative dim at the dispatcher so ROCm / OneAPI /
    // CUDA backends (which use the attribute raw) don't underflow on
    // `shape[dim]`.  CPU has its own normalisation today, but the parity
    // requirement is that every backend sees a non-negative dim.
    const int64_t ndim = static_cast<int64_t>(input.shape().size());
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) {
        throw std::out_of_range("cummax: dim out of range");
    }
    std::array<Tensor, 1> inputs = {input.contiguous()};
    NewOpAttributes attrs;
    attrs.set(AttrKey::Dim, dim);
    auto results = dispatch<OpId::CumMax>(inputs, attrs);
    return {results[0], results[1]};
}

auto cummin(const Tensor& input, int64_t dim) -> std::pair<Tensor, Tensor> {
    // audit-5 Y.7: same negative-dim normalisation as cummax above.
    const int64_t ndim = static_cast<int64_t>(input.shape().size());
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) {
        throw std::out_of_range("cummin: dim out of range");
    }
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
    // audit-5 Y.7: same negative-dim normalisation as cummax / cummin above.
    const int64_t ndim = static_cast<int64_t>(input.shape().size());
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) {
        throw std::out_of_range("kthvalue: dim out of range");
    }
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
    if (input.ndim() != 1 && input.ndim() != 2) {
        throw std::invalid_argument("cov: input must be 1D or 2D");
    }
    // Promote 1D -> (1, N). Keeps the output as a (1,1) matrix and
    // routes all dtypes (Float16/BFloat16 included) through matmul,
    // which has full dtype coverage — dot_kernel is Float32/Float64-only.
    Tensor x2d = (input.ndim() == 1)
        ? input.reshape({1, input.shape()[0]})
        : input;

    int64_t M = x2d.shape()[1];
    auto m_val = static_cast<double>(M - correction);

    auto row_means = tenzor::mean(x2d, 1, true);             // (N, 1)
    auto centered = tenzor::sub(x2d, row_means);             // (N, M)
    auto centered_t = tenzor::transpose(centered, 0, 1);     // (M, N)
    auto product = tenzor::matmul(centered, centered_t);     // (N, N)
    return tenzor::div(product, tenzor::full({}, m_val, x2d.dtype(), x2d.device()));
}

auto cov(const Tensor& input, int64_t correction,
         const Tensor& fweights, const Tensor& aweights) -> Tensor {
    const bool has_fw = fweights.numel() > 0;
    const bool has_aw = aweights.numel() > 0;
    if (!has_fw && !has_aw) {
        return cov(input, correction);
    }
    if (input.ndim() != 1 && input.ndim() != 2) {
        throw std::invalid_argument("cov: input must be 1D or 2D");
    }
    Tensor x2d = (input.ndim() == 1)
        ? input.reshape({1, input.shape()[0]})
        : input;
    const int64_t M = x2d.shape()[1];
    const DType dt = x2d.dtype();

    // Combined per-observation weights w (length M), promoted to input dtype:
    // w = fweights * aweights (each defaulting to 1 when absent).
    Tensor w;
    if (has_fw) w = fweights.to(dt);
    if (has_aw) {
        Tensor aw = aweights.to(dt);
        w = has_fw ? tenzor::mul(w, aw) : aw;
    }
    Tensor w_row = w.reshape({1, M});                 // (1, M)
    Tensor w_sum = tenzor::sum(w);                    // scalar

    // Normalization (PyTorch torch.cov):
    //   no aweights: fact = w_sum - correction
    //   aweights:    fact = w_sum - correction * sum(w * aweights) / w_sum
    Tensor fact;
    if (has_aw) {
        Tensor wa_sum = tenzor::sum(tenzor::mul(w, aweights.to(dt)));
        Tensor term = tenzor::div(wa_sum, w_sum);
        fact = tenzor::sub(w_sum, tenzor::mul(term, static_cast<double>(correction)));
    } else {
        fact = tenzor::add(w_sum, static_cast<double>(-correction));
    }

    // Weighted row means: sum(x * w, dim=1, keepdim) / w_sum  -> (N, 1)
    Tensor mu = tenzor::div(tenzor::sum(tenzor::mul(x2d, w_row), 1, true), w_sum);
    Tensor centered = tenzor::sub(x2d, mu);           // (N, M)
    Tensor x_w = tenzor::mul(centered, w_row);        // (N, M)
    Tensor centered_t = tenzor::transpose(centered, 0, 1);  // (M, N)
    Tensor product = tenzor::matmul(x_w, centered_t);       // (N, N)
    return tenzor::div(product, fact);
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
    // Mathematically corr ∈ [-1, 1], but in Float16/BFloat16 the
    // sqrt → matmul → div chain accumulates enough round-off that the
    // result drifts outside the interval. Widen to Float32 for those
    // dtypes so the bound holds, then cast back — matching the widen-
    // then-narrow pattern used elsewhere for reduced-precision floats.
    auto src_dtype = input.dtype();
    const bool needs_widen = (src_dtype == DType::Float16 ||
                              src_dtype == DType::BFloat16);
    Tensor work = needs_widen ? input.to(DType::Float32) : input;

    auto c = tenzor::cov(work);

    if (work.ndim() == 1) {
        // cov(1D) is (1,1); correlation of a single variable with itself
        // is 1. Preserve that shape so the result is parallel to the
        // 2D case (N×N) with N=1.
        auto r = tenzor::full({1, 1}, 1.0, work.dtype(), work.device());
        return needs_widen ? r.to(src_dtype) : r;
    }

    auto diag_vals = tenzor::diag(c);                // (N,)
    auto stds = tenzor::sqrt(diag_vals);             // (N,)
    auto stds_col = tenzor::reshape(stds, {-1, 1});  // (N, 1)
    auto stds_row = tenzor::reshape(stds, {1, -1});  // (1, N)
    auto norm = tenzor::matmul(stds_col, stds_row);  // (N, N)
    auto result = tenzor::div(c, norm);
    return needs_widen ? result.to(src_dtype) : result;
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
