#include "tenzor/ops/reduction.hpp"
#include "tenzor/backend/fast_dispatch.hpp"
#include "tenzor/backend/op_attributes.hpp"
#include "tenzor/ops/op_id.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/transform.hpp"
#include "tenzor/ops/indexing.hpp"
#include "tenzor/ops/type_promotion.hpp"
#include "tenzor/utils/profiling.hpp"

#include <iomanip>
#include <sstream>

namespace tenzor {

// Serialise a double with full round-trip precision (17 significant digits) so
// that range bounds passed to backends via string attributes are not truncated
// (std::to_string(double) emits only 6 fractional digits). std::stod on the
// backend side accepts this format, including scientific notation.
static std::string double_to_string_full(double v) {
    std::ostringstream oss;
    oss << std::setprecision(17) << v;
    return oss.str();
}

// Type promotion helpers matching PyTorch semantics:
// - Integer sum/prod accumulate in Int64 to prevent overflow
// - mean/var/std/norm on integers produce Float32 results
static bool is_small_int_dtype(DType dt) {
    return dt == DType::Int8 || dt == DType::UInt8 || dt == DType::Int16 ||
           dt == DType::Int32 || dt == DType::UInt16 || dt == DType::UInt32 || dt == DType::Bool;
}

static bool is_integer_dtype(DType dt) {
    return dt == DType::Int8 || dt == DType::UInt8 || dt == DType::Int16 ||
           dt == DType::Int32 || dt == DType::Int64 || dt == DType::Bool ||
           dt == DType::UInt16 || dt == DType::UInt32 || dt == DType::UInt64;
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
    if (dim.has_value()) attrs.set(AttrKey::Dim, normalize_reduce_dim(*dim, static_cast<int64_t>(input.shape().size()), "max"));
    attrs.set(AttrKey::Keepdim, keepdim);
    std::vector<Tensor> inputs = {input};
    return dispatch(OpId::Max, inputs, attrs)[0];
}

auto min(const Tensor& input, std::optional<int64_t> dim, bool keepdim) -> Tensor {
    NewOpAttributes attrs;
    if (dim.has_value()) attrs.set(AttrKey::Dim, normalize_reduce_dim(*dim, static_cast<int64_t>(input.shape().size()), "min"));
    attrs.set(AttrKey::Keepdim, keepdim);
    std::vector<Tensor> inputs = {input};
    return dispatch(OpId::Min, inputs, attrs)[0];
}

auto argmax(const Tensor& input, std::optional<int64_t> dim, bool keepdim) -> Tensor {
    NewOpAttributes attrs;
    if (dim.has_value()) attrs.set(AttrKey::Dim, normalize_reduce_dim(*dim, static_cast<int64_t>(input.shape().size()), "argmax"));
    attrs.set(AttrKey::Keepdim, keepdim);
    std::vector<Tensor> inputs = {input};
    return dispatch(OpId::ArgMax, inputs, attrs)[0];
}

auto argmin(const Tensor& input, std::optional<int64_t> dim, bool keepdim) -> Tensor {
    NewOpAttributes attrs;
    if (dim.has_value()) attrs.set(AttrKey::Dim, normalize_reduce_dim(*dim, static_cast<int64_t>(input.shape().size()), "argmin"));
    attrs.set(AttrKey::Keepdim, keepdim);
    std::vector<Tensor> inputs = {input};
    return dispatch(OpId::ArgMin, inputs, attrs)[0];
}

auto argsort(const Tensor& input, int64_t dim, bool descending) -> Tensor {
    NewOpAttributes attrs;
    attrs.set(AttrKey::Dim, normalize_reduce_dim(dim, static_cast<int64_t>(input.shape().size()), "argsort"));
    attrs.set(AttrKey::Descending, descending);
    std::vector<Tensor> inputs = {input};
    return dispatch(OpId::ArgSort, inputs, attrs)[0];
}

auto prod(const Tensor& input, std::optional<int64_t> dim, bool keepdim) -> Tensor {
    // Complex prod on GPU backends without a native complex reduction kernel
    // (ROCm/OneAPI/Vulkan throw "prod: unsupported dtype"). Compute via the EXACT
    // polar decomposition |Πz| = Π|z|, arg(Πz) = Σ arg(z), using SAME-DEVICE real
    // reductions — no CPU round-trip and no native complex kernel required. CPU
    // and CUDA have native complex prod, so try the native dispatch first and only
    // fall back when the backend reports the dtype unsupported (mirrors the
    // native-then-composed pattern used by the attention backward).
    if (is_complex_type(input.dtype()) && input.device().type != Device::Type::CPU) {
        try {
            NewOpAttributes cattrs;
            if (dim.has_value()) cattrs.set(AttrKey::Dim, normalize_reduce_dim(*dim, static_cast<int64_t>(input.shape().size()), "prod"));
            cattrs.set(AttrKey::Keepdim, keepdim);
            std::vector<Tensor> cinputs = {input};
            return dispatch(OpId::Prod, cinputs, cattrs)[0];
        } catch (const std::exception&) {
            Tensor mag = abs(input);      // |z|  (real)
            Tensor ang = angle(input);    // atan2(im, re)  (real)
            Tensor pmag = prod(mag, dim, keepdim);   // Π|z|  (real prod, native)
            Tensor sang = sum(ang, dim, keepdim);    // Σ arg  (real sum, native)
            return complex(pmag * cos(sang), pmag * sin(sang));
        }
    }
    // Promote small integer types to Int64 to prevent overflow
    Tensor promoted = is_small_int_dtype(input.dtype()) ? input.to(DType::Int64) : input;
    NewOpAttributes attrs;
    if (dim.has_value()) attrs.set(AttrKey::Dim, normalize_reduce_dim(*dim, static_cast<int64_t>(input.shape().size()), "prod"));
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

auto norm(const Tensor& input, double p, std::optional<int64_t> dim, bool keepdim) -> Tensor {
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
    if (dim.has_value()) attrs.set(AttrKey::Dim, normalize_reduce_dim(*dim, static_cast<int64_t>(input.shape().size()), "any"));
    attrs.set(AttrKey::Keepdim, keepdim);
    std::vector<Tensor> inputs = {input};
    return dispatch(OpId::Any, inputs, attrs)[0];
}

auto all(const Tensor& input, std::optional<int64_t> dim, bool keepdim) -> Tensor {
    NewOpAttributes attrs;
    if (dim.has_value()) attrs.set(AttrKey::Dim, normalize_reduce_dim(*dim, static_cast<int64_t>(input.shape().size()), "all"));
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

    // Normalize the (possibly negative) reduction dim once up front so every
    // downstream consumer — the fused-kernel dispatch (which passes dim raw to
    // backends that assume non-negative) and the composite fallback — sees a
    // guaranteed in-range axis.
    dim = normalize_reduce_dim(dim, static_cast<int64_t>(input.shape().size()), "logsumexp");

    // Empty tensor: return empty with appropriate shape
    if (input.numel() == 0) {
        // For empty input, result shape collapses dim to 1 (keepdim) or removes it
        auto shape = std::vector<int64_t>(input.shape().begin(), input.shape().end());
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

    // Resolve the range. Per the documented contract, min == max means "use the
    // data range"; compute it here so the range is concrete before we can guard
    // against a degenerate (zero-width) range, which would otherwise produce a
    // div-by-zero bin width in the backend Histogram kernel.
    if (min_val == max_val) {
        if (inp.numel() == 0) {
            min_val = 0.0;
            max_val = 1.0;
        } else {
            min_val = tenzor::min(inp).to(DType::Float64).item<double>();
            max_val = tenzor::max(inp).to(DType::Float64).item<double>();
        }
    }
    // Degenerate range (all values equal, or the resolved data range has zero
    // width): widen by 0.5 on each side, matching torch.histogram.
    if (min_val == max_val) {
        min_val -= 0.5;
        max_val += 0.5;
    }

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
            ranges_str += double_to_string_full(r[i].first) + ',' +
                          double_to_string_full(r[i].second);
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
    // Half precision: widen to Float32 for the reduction (avoids catastrophic
    // F16 rounding in the squared differences) and narrow the result back so
    // nanvar(Float16) returns Float16, matching mean/var dtype conventions.
    const bool half = (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16);
    Tensor x = half ? input.to(DType::Float32) : input;
    Tensor mean = nanmean(x, dim, /*keepdim=*/true);
    Tensor diff = x - mean;
    // Zero out NaN positions so they don't contribute
    Tensor nan_mask = isnan(x);
    Tensor diff_sq = diff * diff;
    // Replace NaN-originated values with 0
    Tensor clean = where(nan_mask, zeros_like(diff_sq), diff_sq);
    Tensor sum_sq = nansum(clean, dim, keepdim);
    // Count valid (non-NaN) elements. Accumulate the count in a wide float
    // dtype (Float64 for Float64 input, Float32 otherwise) so half-precision
    // inputs don't saturate the count — mirrors mean's integer->Float32
    // promotion convention.
    DType count_dtype = (x.dtype() == DType::Float64) ? DType::Float64 : DType::Float32;
    Tensor valid_mask = logical_not(nan_mask);
    Tensor count = sum(valid_mask.to(count_dtype), dim, keepdim);
    Tensor denom = count - static_cast<float>(correction);
    Tensor result = sum_sq / denom;
    return half ? result.to(input.dtype()) : result;
}

auto nanstd(const Tensor& input, std::optional<int64_t> dim, bool keepdim, int64_t correction) -> Tensor {
    return sqrt(nanvar(input, dim, keepdim, correction));
}

auto aminmax(const Tensor& input, std::optional<int64_t> dim, bool keepdim) -> std::pair<Tensor, Tensor> {
    std::array<Tensor, 1> inputs = {input.contiguous()};
    NewOpAttributes attrs;
    if (dim.has_value()) {
        attrs.set(AttrKey::Dim,
                  normalize_reduce_dim(*dim, static_cast<int64_t>(input.shape().size()), "aminmax"));
    }
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

auto dist(const Tensor& a, const Tensor& b, double p) -> Tensor {
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
    // The kernel keys its dtype branch off elements.dtype() and reads the
    // test_elements buffer with that same element type. If test_elements is
    // wider its buffer is reinterpreted and read out of bounds. Promote both
    // to a common dtype before dispatch.
    DType common = promote_types(elements.dtype(), test_elements.dtype());
    Tensor el = (elements.dtype() != common) ? elements.to(common) : elements;
    Tensor te = (test_elements.dtype() != common) ? test_elements.to(common)
                                                  : test_elements;
    std::array<Tensor, 2> inputs = {el.contiguous(), te.contiguous()};
    return dispatch<OpId::Isin>(inputs)[0];
}

auto kthvalue(const Tensor& input, int64_t k, int64_t dim, bool keepdim) -> std::pair<Tensor, Tensor> {
    // audit-5 Y.7: same negative-dim normalisation as cummax / cummin above.
    const int64_t ndim = static_cast<int64_t>(input.shape().size());
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) {
        throw std::out_of_range("kthvalue: dim out of range");
    }
    // F-034: validate k against the reduced dim's size BEFORE dispatch, on
    // every backend — not just CPU (advanced.cpp:1051-1053). The CUDA
    // selection-sort kernel indexes a per-slice workspace sized exactly
    // dim_size; k outside [1, dim_size] reads/writes past that slice's
    // segment. Throwing here (host-side, pre-launch) is strictly better than
    // a device-side check and covers every backend uniformly.
    const int64_t dim_size = input.shape()[static_cast<size_t>(dim)];
    if (k < 1 || k > dim_size) {
        throw std::runtime_error("kthvalue: k out of range");
    }
    std::array<Tensor, 1> inputs = {input.contiguous()};
    NewOpAttributes attrs;
    attrs.set(AttrKey::Dim, dim);
    attrs.set(AttrKey::Keepdim, keepdim);
    attrs.set(AttrKey::K, k);
    auto results = dispatch<OpId::Kthvalue>(inputs, attrs);
    return {results[0], results[1]};
}

// F-005/F-020/F-021/F-037: fmax/fmin kernels (CPU and CUDA) index both
// operands with a single flat linear index over a.numel() elements — unlike
// minimum/maximum, which broadcast internally via per-operand strides, these
// kernels have no broadcast support at all. Mirror the explicit
// promote+broadcast_to+contiguous pattern used for hypot/copysign/nextafter/
// gcd/lcm (see detail::binary_op_promoted_broadcast in src/ops/math.cpp) so
// that by the time the kernel sees the tensors they are guaranteed to share
// an identical (broadcast) shape and dtype. This both adds broadcasting
// support (F-020) and eliminates the CUDA/CPU out-of-bounds read on
// mismatched operand sizes at its root (F-005/F-037), since a.numel() ==
// b.numel() is now guaranteed for any shapes that are broadcast-compatible.
auto fmax(const Tensor& a, const Tensor& b) -> Tensor {
    auto [ap, bp] = promote_inputs(a, b);
    auto bshape = broadcast_shapes(ap.shape(), bp.shape());
    std::array<Tensor, 2> inputs = {broadcast_to(ap, bshape).contiguous(),
                                     broadcast_to(bp, bshape).contiguous()};
    return dispatch<OpId::Fmax>(inputs)[0];
}

auto fmin(const Tensor& a, const Tensor& b) -> Tensor {
    auto [ap, bp] = promote_inputs(a, b);
    auto bshape = broadcast_shapes(ap.shape(), bp.shape());
    std::array<Tensor, 2> inputs = {broadcast_to(ap, bshape).contiguous(),
                                     broadcast_to(bp, bshape).contiguous()};
    return dispatch<OpId::Fmin>(inputs)[0];
}

// When dim is None, PyTorch flattens the whole tensor and reduces it. We must
// decide this at the op layer and always pass an explicit, canonical Dim so the
// backends do not fall back to conflicting registry defaults (CPU=0, CUDA=-1),
// which silently diverge on rank>=2 tensors. Returns the (possibly flattened)
// contiguous input and a definite reduction dim, normalising negative dims.
static auto prepare_flatten_reduce(const Tensor& input, std::optional<int64_t> dim,
                                   const char* op) -> std::pair<Tensor, int64_t> {
    if (dim.has_value()) {
        int64_t d = normalize_reduce_dim(dim.value(),
                                         static_cast<int64_t>(input.shape().size()), op);
        return {input.contiguous(), d};
    }
    // No dim: flatten to 1-D and reduce along axis 0 (matches PyTorch dim=None).
    return {reshape(input.contiguous(), {-1}), 0};
}

auto quantile(const Tensor& input, double q, std::optional<int64_t> dim,
              bool keepdim) -> Tensor {
    if (!(q >= 0.0 && q <= 1.0)) {  // NaN-safe
        throw std::invalid_argument("quantile: q must be in [0, 1]");
    }
    auto [prepared, d] = prepare_flatten_reduce(input, dim, "quantile");
    std::array<Tensor, 1> inputs = {prepared};
    NewOpAttributes attrs;
    attrs.set(AttrKey::Q, q);  // dedicated quantile key (avoids generic Alpha bleed)
    attrs.set(AttrKey::Dim, d);
    attrs.set(AttrKey::Keepdim, keepdim);
    return dispatch<OpId::Quantile>(inputs, attrs)[0];
}

auto nanquantile(const Tensor& input, double q, std::optional<int64_t> dim,
                 bool keepdim) -> Tensor {
    if (!(q >= 0.0 && q <= 1.0)) {  // NaN-safe
        throw std::invalid_argument("nanquantile: q must be in [0, 1]");
    }
    auto [prepared, d] = prepare_flatten_reduce(input, dim, "nanquantile");
    std::array<Tensor, 1> inputs = {prepared};
    NewOpAttributes attrs;
    attrs.set(AttrKey::Q, q);  // dedicated quantile key (avoids generic Alpha bleed)
    attrs.set(AttrKey::Dim, d);
    attrs.set(AttrKey::Keepdim, keepdim);
    return dispatch<OpId::Nanquantile>(inputs, attrs)[0];
}

auto nanmedian(const Tensor& input, std::optional<int64_t> dim) -> Tensor {
    auto [prepared, d] = prepare_flatten_reduce(input, dim, "nanmedian");
    std::array<Tensor, 1> inputs = {prepared};
    NewOpAttributes attrs;
    attrs.set(AttrKey::Dim, d);
    return dispatch<OpId::Nanmedian>(inputs, attrs)[0];
}

auto histc(const Tensor& input, int64_t bins, double min_val, double max_val) -> Tensor {
    if (bins <= 0) {
        throw std::invalid_argument("histc: bins must be positive");
    }
    std::array<Tensor, 1> inputs = {input.contiguous()};
    NewOpAttributes attrs;
    // Use the dedicated histogram keys (shared with OpId::Histogram) rather than
    // the generic N (FFT length) / Alpha / Beta keys, to avoid cross-op bleed.
    attrs.set(AttrKey::NumBins, bins);
    attrs.set(AttrKey::Min, min_val);
    attrs.set(AttrKey::Max, max_val);
    return dispatch<OpId::Histc>(inputs, attrs)[0];
}

auto unique_consecutive(const Tensor& input, bool return_inverse, bool return_counts,
                        std::optional<int64_t> dim)
    -> std::tuple<Tensor, Tensor, Tensor> {
    std::array<Tensor, 1> inputs = {input.contiguous()};
    NewOpAttributes attrs;
    if (dim.has_value()) attrs.set(AttrKey::Dim, normalize_reduce_dim(dim.value(), static_cast<int64_t>(input.shape().size()), "unique_consecutive"));
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
    attrs.set(AttrKey::Dim, normalize_reduce_dim(dim, static_cast<int64_t>(y.ndim()), "trapezoid"));
    return dispatch(OpId::Trapezoid, inputs, attrs)[0];
}

auto trapezoid(const Tensor& y, double dx, int64_t dim) -> Tensor {
    if (y.ndim() == 0) {
        throw std::invalid_argument("trapezoid: input must be at least 1D");
    }
    auto yc = y.contiguous();
    std::array<Tensor, 1> inputs = {yc};
    NewOpAttributes attrs;
    attrs.set(AttrKey::Dim, normalize_reduce_dim(dim, static_cast<int64_t>(y.ndim()), "trapezoid"));
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
    attrs.set(AttrKey::Dim, normalize_reduce_dim(dim, static_cast<int64_t>(y.ndim()), "cumulative_trapezoid"));
    return dispatch(OpId::CumulativeTrapezoid, inputs, attrs)[0];
}

auto cumulative_trapezoid(const Tensor& y, double dx, int64_t dim) -> Tensor {
    if (y.ndim() == 0) {
        throw std::invalid_argument("cumulative_trapezoid: input must be at least 1D");
    }
    auto yc = y.contiguous();
    std::array<Tensor, 1> inputs = {yc};
    NewOpAttributes attrs;
    attrs.set(AttrKey::Dim, normalize_reduce_dim(dim, static_cast<int64_t>(y.ndim()), "cumulative_trapezoid"));
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
    attrs.set(AttrKey::Dim, normalize_reduce_dim(dim, static_cast<int64_t>(input.ndim()), "gradient"));
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
        // Correlation of a single variable with itself is 1. torch.corrcoef
        // returns a 0-dim scalar (not a (1,1) matrix) for 1-D input, so emit a
        // scalar to match.
        auto r = tenzor::full({}, 1.0, work.dtype(), work.device());
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
    attrs.set(AttrKey::Dim, normalize_reduce_dim(axis, static_cast<int64_t>(data_cont.ndim()), "segment_reduce"));
    attrs.set(AttrKey::Reduction, reduce);
    std::vector<Tensor> inputs = {data_cont, offsets_cont};
    return dispatch(OpId::SegmentReduce, inputs, attrs)[0];
}

} // namespace tenzor
