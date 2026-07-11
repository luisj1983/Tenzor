/**
 * @file advanced.cpp
 * @brief Implementation of advanced tensor operations
 */

#include "tenzor/ops/advanced.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/transform.hpp"
#include "tenzor/ops/reduction.hpp"
#include "tenzor/ops/indexing.hpp"
#include "tenzor/backend/dispatch.hpp"
#include "tenzor/backend/fast_dispatch.hpp"
#include "tenzor/backend/op_attributes.hpp"
#include "tenzor/ops/op_id.hpp"
#include "tenzor/core/shape.hpp"
#include "tenzor/ops/math.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <string>
#include <unordered_map>
#include <cstring>
#include <functional>
#include <functional>
#include <type_traits>

// Hash specializations for Float16/BFloat16 so they work with std::unordered_map
template<>
struct std::hash<tenzor::Float16> {
    size_t operator()(const tenzor::Float16& v) const noexcept {
        return std::hash<float>{}(static_cast<float>(v));
    }
};

template<>
struct std::hash<tenzor::BFloat16> {
    size_t operator()(const tenzor::BFloat16& v) const noexcept {
        return std::hash<float>{}(static_cast<float>(v));
    }
};

namespace tenzor {

auto topk(const Tensor& input,
          int64_t k,
          int64_t dim,
          bool largest,
          bool sorted) -> std::tuple<Tensor, Tensor> {

    const int64_t ndim = input.ndim();

    if (ndim == 0) {
        throw std::runtime_error("topk not supported for 0-dimensional tensors");
    }

    // Normalize dimension
    if (dim < 0) {
        dim += ndim;
    }
    if (dim < 0 || dim >= ndim) {
        throw std::runtime_error("Dimension out of range for topk");
    }

    const int64_t dim_size = input.shape()[dim];
    if (k <= 0 || k > dim_size) {
        throw std::runtime_error("k must be between 1 and dimension size");
    }

    // Float16 / BFloat16: widen to Float32 for the partial_sort path,
    // then narrow the values tensor back. Indices are always Int64 so
    // they pass through unchanged. Half-precision comparisons are
    // exactly order-preserving when widened to Float32, so the result
    // is bit-identical to a native half topk modulo the final narrow.
    if (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16) {
        const DType orig = input.dtype();
        auto [values_f32, indices] =
            topk(input.to(DType::Float32), k, dim, largest, sorted);
        return {values_f32.to(orig), indices};
    }

    // Narrow integer / boolean / unsigned dtypes are not natively supported on
    // every backend (nor on the CPU reference switch below). Widen losslessly,
    // run topk, then narrow the values back; indices are Int64 and pass through
    // unchanged. This keeps topk dtype coverage identical to sort across all
    // backends.
    if (input.dtype() == DType::Int8 || input.dtype() == DType::Int16 ||
        input.dtype() == DType::UInt8 || input.dtype() == DType::UInt16 ||
        input.dtype() == DType::Bool) {
        const DType orig = input.dtype();
        auto [values_i32, indices] =
            topk(input.to(DType::Int32), k, dim, largest, sorted);
        return {values_i32.to(orig), indices};
    }
    if (input.dtype() == DType::UInt32) {
        auto [values_i64, indices] =
            topk(input.to(DType::Int64), k, dim, largest, sorted);
        return {values_i64.to(DType::UInt32), indices};
    }

    // JIT-R050: dispatch on EVERY device, including CPU. This used to
    // special-case CPU with a hand-duplicated reference implementation
    // below that bypassed dispatch() entirely -- invisible to
    // DispatchInterceptorStack (the JIT tracer's only hook into the graph)
    // by construction, unlike every other device. The CPU-registered
    // OpId::TopK kernel (cpu_kernel_registry.cpp -> cpu::topk_kernel,
    // backends/cpu/kernels/advanced.cpp) implements the identical
    // partial_sort + optional full-sort algorithm the deleted reference
    // path duplicated, with equal-or-broader dtype coverage (also handles
    // Float16/BFloat16 natively; this function still pre-widens those two
    // above for parity with the pre-existing narrow-int widening).
    OpAttributes attrs;
    attrs.set(AttrKey::K, k);
    attrs.set(AttrKey::Dim, dim);
    attrs.set(AttrKey::Largest, largest);
    attrs.set(AttrKey::Sorted, sorted);
    std::vector<Tensor> inputs = {input};
    auto results = dispatch<OpId::TopK>(inputs, attrs);
    return {results[0], results[1]};
}

auto sort(const Tensor& input,
          int64_t dim,
          bool descending) -> std::tuple<Tensor, Tensor> {

    const int64_t ndim = input.ndim();

    if (ndim == 0) {
        throw std::runtime_error("sort not supported for 0-dimensional tensors");
    }

    // Normalize dimension
    if (dim < 0) {
        dim += ndim;
    }
    if (dim < 0 || dim >= ndim) {
        throw std::runtime_error("Dimension out of range for sort");
    }

    // Float16 / BFloat16: widen to Float32, sort, then narrow the values
    // tensor back. Indices are Int64 and pass through unchanged. Half-precision
    // comparisons are exactly order-preserving when widened to Float32, so the
    // ordering is identical to a native half sort.
    if (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16) {
        const DType orig = input.dtype();
        auto [values_f32, indices] = sort(input.to(DType::Float32), dim, descending);
        return {values_f32.to(orig), indices};
    }

    // Narrow integer / boolean dtypes are not natively sortable on every backend
    // (nor on the CPU reference path). Widen losslessly, sort, then narrow the
    // values back; indices are Int64 and pass through unchanged. This keeps sort
    // dtype coverage identical across CPU/CUDA/ROCm/Vulkan/OneAPI.
    if (input.dtype() == DType::Int8 || input.dtype() == DType::Int16 ||
        input.dtype() == DType::UInt8 || input.dtype() == DType::UInt16 ||
        input.dtype() == DType::Bool) {
        const DType orig = input.dtype();
        auto [values_i32, indices] = sort(input.to(DType::Int32), dim, descending);
        return {values_i32.to(orig), indices};
    }
    if (input.dtype() == DType::UInt32) {
        auto [values_i64, indices] = sort(input.to(DType::Int64), dim, descending);
        return {values_i64.to(DType::UInt32), indices};
    }

    // JIT-R050: dispatch on EVERY device, including CPU — see topk()'s
    // identical fix above for the full rationale. cpu::sort_kernel
    // (backends/cpu/kernels/advanced.cpp) implements the same algorithm
    // this deleted reference path duplicated, with equal-or-broader dtype
    // coverage.
    OpAttributes attrs;
    attrs.set(AttrKey::Dim, dim);
    attrs.set(AttrKey::Descending, descending);
    std::vector<Tensor> inputs = {input};
    auto results = dispatch<OpId::Sort>(inputs, attrs);
    return {results[0], results[1]};
}

auto unique(const Tensor& input,
            bool sorted_output,
            bool return_inverse,
            bool return_counts) -> std::tuple<Tensor, Tensor, Tensor> {

    // JIT-R050: dispatch on EVERY device, including CPU — see topk()'s fix
    // above for the general rationale. Unlike topk/sort, this is NOT a
    // simple duplicate: the GPU Unique kernels (ROCm/OneAPI/Vulkan) always
    // emit values in SORTED order, while the deleted CPU-only reference
    // implementation returned FIRST-APPEARANCE order for
    // sorted_output==false — so this reorder step (originally "for non-CPU
    // tensors only") is still needed, now unconditionally, to give EVERY
    // backend (CPU included) matching first-appearance-order semantics.
    // cpu::unique_kernel (registered for OpId::Unique) sorts when asked
    // (attrs Sorted=true below), matching the GPU kernels' behavior, so
    // this reorder produces the identical first-appearance result CPU used
    // to compute directly, regardless of which backend ran the dispatch.
    const bool need_reorder = !sorted_output;
    OpAttributes attrs;
    attrs.set(AttrKey::Sorted, true);
    attrs.set(AttrKey::ReturnInverse, need_reorder ? true : return_inverse);
    attrs.set(AttrKey::ReturnCounts, need_reorder ? true : return_counts);
    std::vector<Tensor> inputs = {input};
    auto results = dispatch<OpId::Unique>(inputs, attrs);
    if (!need_reorder) {
        return {results[0], results[1], results[2]};
    }
    Tensor values = results[0];
    Tensor inverse = results[1];
    Tensor counts = results[2];
    const int64_t U = values.shape().empty() ? 0 : values.shape()[0];
    const int64_t N = inverse.numel();
    if (U <= 1 || N == 0) {
        // 0/1 unique value: order is already identical to first-appearance.
        return {values,
                return_inverse ? inverse : Tensor{},
                return_counts ? counts : Tensor{}};
    }
    Tensor inverse_flat = inverse.reshape({N});
    // first_occ[g] = min input position whose group is g (identity = N).
    Tensor pos = arange(0.0, static_cast<double>(N), 1.0, DType::Int64, inverse.device());
    Tensor init = full({U}, static_cast<double>(N), DType::Int64, values.device());
    Tensor first_occ = scatter_reduce(init, 0, inverse_flat, pos, "amin", /*include_self=*/true);
    Tensor perm = argsort(first_occ, 0, /*descending=*/false);   // groups by first occurrence
    Tensor inv_perm = argsort(perm, 0, /*descending=*/false);    // inverse permutation
    values = index_select(values, 0, perm);
    if (return_counts) {
        counts = index_select(counts, 0, perm);
    }
    Tensor new_inverse;
    if (return_inverse) {
        std::vector<int64_t> inv_shape(inverse.shape().begin(), inverse.shape().end());
        new_inverse = index_select(inv_perm, 0, inverse_flat).reshape(inv_shape);
    }
    return {values,
            return_inverse ? new_inverse : Tensor{},
            return_counts ? counts : Tensor{}};
}

auto cumsum(const Tensor& input, int64_t dim) -> Tensor {
    const int64_t ndim = input.ndim();

    if (ndim == 0) {
        throw std::runtime_error("cumsum not supported for 0-dimensional tensors");
    }

    // Normalize dimension
    if (dim < 0) {
        dim += ndim;
    }
    if (dim < 0 || dim >= ndim) {
        throw std::runtime_error("Dimension out of range for cumsum");
    }

    // Float16 / BFloat16: accumulate in Float32 for numerical stability, then
    // narrow back to the original dtype.
    if (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16) {
        const DType orig = input.dtype();
        return cumsum(input.to(DType::Float32), dim).to(orig);
    }

    // JIT-R050: dispatch on EVERY device, including CPU — see topk()'s
    // identical fix above for the full rationale. cpu::cumsum_kernel
    // (backends/cpu/kernels/advanced.cpp) implements the identical
    // algorithm this deleted reference path duplicated.
    NewOpAttributes attrs;
    attrs.set(AttrKey::Dim, dim);
    std::vector<Tensor> inputs = {input};
    return dispatch<OpId::CumSum>(inputs, attrs)[0];
}

auto cumprod(const Tensor& input, int64_t dim) -> Tensor {
    const int64_t ndim = input.ndim();

    if (ndim == 0) {
        throw std::runtime_error("cumprod not supported for 0-dimensional tensors");
    }

    // Normalize dimension
    if (dim < 0) {
        dim += ndim;
    }
    if (dim < 0 || dim >= ndim) {
        throw std::runtime_error("Dimension out of range for cumprod");
    }

    // Float16 / BFloat16: accumulate in Float32 for numerical stability, then
    // narrow back to the original dtype.
    if (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16) {
        const DType orig = input.dtype();
        return cumprod(input.to(DType::Float32), dim).to(orig);
    }

    // JIT-R050: dispatch on EVERY device, including CPU — see topk()'s
    // identical fix above for the full rationale. cpu::cumprod_kernel
    // (backends/cpu/kernels/advanced.cpp) implements the identical
    // algorithm this deleted reference path duplicated.
    NewOpAttributes attrs;
    attrs.set(AttrKey::Dim, dim);
    std::vector<Tensor> inputs = {input};
    return dispatch<OpId::CumProd>(inputs, attrs)[0];
}

auto logcumsumexp(const Tensor& input, int64_t dim) -> Tensor {
    const int64_t ndim = input.ndim();

    if (ndim == 0) {
        throw std::runtime_error("logcumsumexp not supported for 0-dimensional tensors");
    }

    // Normalize dimension
    if (dim < 0) {
        dim += ndim;
    }
    if (dim < 0 || dim >= ndim) {
        throw std::runtime_error("Dimension out of range for logcumsumexp");
    }

    // JIT-R050: dispatch on EVERY device, including CPU — see topk()'s
    // identical fix above for the full rationale. cpu::logcumsumexp_kernel
    // (backends/cpu/kernels/reduction.cpp) implements the identical
    // numerically-stable running-max algorithm this deleted reference path
    // duplicated, and natively widens Float16/BFloat16 itself.
    NewOpAttributes attrs;
    attrs.set(AttrKey::Dim, dim);
    std::vector<Tensor> inputs = {input};
    return dispatch<OpId::Logcumsumexp>(inputs, attrs)[0];
}

// ============================================================================
// einsum — Einstein summation convention
// ============================================================================

// Parse an einsum equation like "ij,jk->ik" into input subscripts and output subscript
static auto parse_einsum_equation(const std::string& equation)
    -> std::pair<std::vector<std::string>, std::string> {
    // Split on "->"
    auto arrow = equation.find("->");
    std::string inputs_str, output_str;
    if (arrow != std::string::npos) {
        inputs_str = equation.substr(0, arrow);
        output_str = equation.substr(arrow + 2);
    } else {
        inputs_str = equation;
        // Implicit output: sorted unique labels not appearing in contractions
        // (labels that appear exactly once across all inputs)
        std::unordered_map<char, int> counts;
        for (char c : inputs_str) {
            if (c != ',' && c != ' ') counts[c]++;
        }
        // NumPy/PyTorch emit every single-occurrence label in canonical order
        // (uppercase A-Z before lowercase a-z). Scanning only 'a'..'z' would
        // drop a free uppercase label, which would then be summed away as a
        // contraction label and yield the wrong output rank.
        for (char c = 'A'; c <= 'Z'; ++c) {
            if (counts.count(c) && counts[c] == 1) output_str += c;
        }
        for (char c = 'a'; c <= 'z'; ++c) {
            if (counts.count(c) && counts[c] == 1) output_str += c;
        }
    }

    // Split inputs on ','
    std::vector<std::string> input_subs;
    std::string current;
    for (char c : inputs_str) {
        if (c == ',') {
            input_subs.push_back(current);
            current.clear();
        } else if (c != ' ') {
            current += c;
        }
    }
    if (!current.empty()) input_subs.push_back(current);

    return {input_subs, output_str};
}

auto einsum_composed(const std::string& equation,
                     std::span<const Tensor> tensors) -> Tensor {
    auto [input_subs, output_sub] = parse_einsum_equation(equation);

    if (input_subs.size() != tensors.size()) {
        throw std::invalid_argument("einsum: number of subscripts (" +
            std::to_string(input_subs.size()) + ") does not match number of tensors (" +
            std::to_string(tensors.size()) + ")");
    }

    // Canonicalize label letters so that equivalent contractions written with
    // different letters still hit the optimized (BLAS-backed) fast paths below.
    // Labels are renamed to a fixed alphabet 'a','b','c',... in order of first
    // appearance across all input subscripts followed by the output subscript.
    // This is a pure bijective relabeling and preserves einsum semantics, so
    // e.g. "ik,kj->ij" canonicalizes to "ab,bc->ac" == "ij,jk->ik" form.
    auto canonicalize_labels = [&]() {
        std::unordered_map<char, char> remap;
        char next = 'a';
        auto remap_str = [&](const std::string& s) {
            std::string out;
            out.reserve(s.size());
            for (char c : s) {
                auto it = remap.find(c);
                if (it == remap.end()) {
                    char nc = next++;
                    remap.emplace(c, nc);
                    out.push_back(nc);
                } else {
                    out.push_back(it->second);
                }
            }
            return out;
        };
        std::vector<std::string> csubs;
        csubs.reserve(input_subs.size());
        for (const auto& s : input_subs) {
            csubs.push_back(remap_str(s));
        }
        std::string cout = remap_str(output_sub);
        return std::make_pair(csubs, cout);
    };
    auto [canon_subs, canon_out] = canonicalize_labels();

    // Fast paths for common patterns (matched against canonical label forms)
    if (tensors.size() == 2) {
        const auto& a = tensors[0];
        const auto& b = tensors[1];
        const auto& sa = canon_subs[0];
        const auto& sb = canon_subs[1];

        // Matrix multiply: ij,jk->ik  (canonical: ab,bc->ac)
        if (sa == "ab" && sb == "bc" && canon_out == "ac") {
            return matmul(a, b);
        }
        // Batch matmul: bij,bjk->bik  (canonical: abc,acd->abd)
        if (sa == "abc" && sb == "acd" && canon_out == "abd") {
            return bmm(a, b);
        }
        // Dot product: i,i->  (canonical: a,a->)
        if (sa == "a" && sb == "a" && canon_out.empty()) {
            return dot(a, b);
        }
        // Outer product: i,j->ij  (canonical: a,b->ab)
        if (sa == "a" && sb == "b" && canon_out == "ab") {
            auto a_col = reshape(a, {a.numel(), 1});
            auto b_row = reshape(b, {1, b.numel()});
            return matmul(a_col, b_row);
        }
    }
    if (tensors.size() == 1) {
        const auto& a = tensors[0];
        const auto& sa = canon_subs[0];

        // Trace: ii->  (canonical: aa->)
        if (sa == "aa" && canon_out.empty()) {
            return trace(a);
        }
        // Diagonal: ii->i  (canonical: aa->a)
        if (sa == "aa" && canon_out == "a") {
            return diag(a);
        }
    }

    // General path: use the transpose-reshape-contract approach
    // 0. Validate subscript ranks against tensor ranks before any rewriting.
    for (size_t t = 0; t < tensors.size(); ++t) {
        if (input_subs[t].size() != static_cast<size_t>(tensors[t].ndim())) {
            throw std::invalid_argument("einsum: subscript '" + input_subs[t] +
                "' has " + std::to_string(input_subs[t].size()) +
                " labels but tensor has " + std::to_string(tensors[t].ndim()) + " dims");
        }
    }

    // 0b. Collapse repeated labels *within a single* input subscript by
    //     extracting the diagonal so every label maps to exactly one axis.
    //     align_tensor below relies on sub.find()/all_labels.find() which only
    //     return the first occurrence; without this, a subscript like "ii" or
    //     "iij" would carry two axes for one label, and reshape() would request
    //     fewer elements than present (size-mismatch throw) or silently contract
    //     the wrong thing. Collapse two equal-sized axes (a<b, same label) by
    //     moving them to the trailing two positions, flattening that NxN block,
    //     and index_select-ing the k*(N+1) diagonal entries.
    auto collapse_diagonal = [](Tensor t, std::string& sub) {
        bool changed = true;
        while (changed) {
            changed = false;
            for (size_t a = 0; a < sub.size() && !changed; ++a) {
                for (size_t b = a + 1; b < sub.size(); ++b) {
                    if (sub[a] != sub[b]) continue;
                    auto shape = t.shape();
                    const int64_t na = shape[a];
                    const int64_t nb = shape[b];
                    if (na != nb) {
                        throw std::invalid_argument(
                            "einsum: repeated label '" + std::string(1, sub[a]) +
                            "' refers to axes of differing size");
                    }
                    const int64_t ndim = t.ndim();
                    // Move axes a,b to the last two positions (b last, a second-to-last).
                    std::vector<int64_t> src = {static_cast<int64_t>(a), static_cast<int64_t>(b)};
                    std::vector<int64_t> dst = {ndim - 2, ndim - 1};
                    Tensor moved = movedim(t, src, dst).contiguous();
                    // Flatten the trailing NxN block to N*N, then pick the diagonal.
                    auto mshape = moved.shape();
                    std::vector<int64_t> flat_shape(mshape.begin(), mshape.end() - 2);
                    flat_shape.push_back(na * nb);
                    Tensor flat = reshape(moved, flat_shape);
                    // Diagonal offsets within the flattened NxN block are
                    // k*(N+1) for k in [0,N): the arithmetic sequence
                    // [0, N+1, 2(N+1), ...] of N elements (step N+1).
                    Tensor idx_t = arange(0.0,
                                          static_cast<double>(na) * static_cast<double>(nb + 1),
                                          static_cast<double>(nb + 1),
                                          DType::Int64, t.device());
                    Tensor picked = index_select(flat, static_cast<int64_t>(flat_shape.size()) - 1, idx_t);
                    // picked now has the merged label as its last axis; rewrite sub:
                    // remove positions a and b, append the single merged label.
                    char label = sub[a];
                    std::string new_sub;
                    for (size_t i = 0; i < sub.size(); ++i) {
                        if (i == a || i == b) continue;
                        new_sub += sub[i];
                    }
                    new_sub += label;
                    sub = new_sub;
                    t = picked;
                    changed = true;
                    break;
                }
            }
        }
        return t;
    };

    std::vector<Tensor> work_tensors;
    work_tensors.reserve(tensors.size());
    std::vector<std::string> work_subs = input_subs;
    for (size_t t = 0; t < tensors.size(); ++t) {
        work_tensors.push_back(collapse_diagonal(tensors[t], work_subs[t]));
    }

    // 1. Build label→dimension size mapping
    std::unordered_map<char, int64_t> label_sizes;
    for (size_t t = 0; t < work_tensors.size(); ++t) {
        auto shape = work_tensors[t].shape();
        for (size_t d = 0; d < work_subs[t].size(); ++d) {
            char label = work_subs[t][d];
            if (label_sizes.count(label)) {
                if (label_sizes[label] != shape[d]) {
                    throw std::invalid_argument("einsum: dimension mismatch for label '" +
                        std::string(1, label) + "'");
                }
            } else {
                label_sizes[label] = shape[d];
            }
        }
    }

    // 2. Identify contraction labels (in inputs but not in output)
    std::string contract_labels;
    for (auto& [label, _] : label_sizes) {
        if (output_sub.find(label) == std::string::npos) {
            contract_labels += label;
        }
    }

    // 3. Build unified label ordering: output labels + contraction labels
    std::string all_labels = output_sub + contract_labels;

    // 4. For each tensor, permute dims to align with all_labels order,
    //    unsqueezing missing dims to size 1.
    auto align_tensor = [&](const Tensor& t, const std::string& sub) -> Tensor {
        // Build shape with all_labels, inserting size-1 for missing labels
        std::vector<int64_t> new_shape(all_labels.size(), 1);
        std::vector<int64_t> perm;

        for (size_t i = 0; i < all_labels.size(); ++i) {
            auto pos = sub.find(all_labels[i]);
            if (pos != std::string::npos) {
                new_shape[i] = t.shape()[static_cast<int64_t>(pos)];
            }
        }

        // Build permutation: reorder tensor dims to match their position in all_labels
        std::vector<int64_t> src_to_target(sub.size());
        for (size_t i = 0; i < sub.size(); ++i) {
            src_to_target[i] = static_cast<int64_t>(all_labels.find(sub[i]));
        }

        // Sort source dims by target position
        std::vector<int64_t> sorted_src(sub.size());
        std::iota(sorted_src.begin(), sorted_src.end(), 0);
        std::sort(sorted_src.begin(), sorted_src.end(),
                  [&](int64_t a, int64_t b) { return src_to_target[a] < src_to_target[b]; });

        Tensor permuted = permute(t, sorted_src);
        return reshape(permuted, new_shape);
    };

    // 5. Align all tensors, multiply element-wise, reduce contraction dims
    Tensor result = align_tensor(work_tensors[0], work_subs[0]);
    for (size_t t = 1; t < work_tensors.size(); ++t) {
        Tensor aligned = align_tensor(work_tensors[t], work_subs[t]);
        result = mul(result, aligned);
    }

    // 6. Sum over contraction dimensions (from the end to avoid index shifting)
    std::vector<int64_t> reduce_dims;
    for (size_t i = output_sub.size(); i < all_labels.size(); ++i) {
        reduce_dims.push_back(static_cast<int64_t>(i));
    }
    // Sort descending to reduce from back
    std::sort(reduce_dims.rbegin(), reduce_dims.rend());
    for (int64_t dim : reduce_dims) {
        result = sum(result, dim, false);
    }

    return result;
}

auto einsum(const std::string& equation,
            std::span<const Tensor> tensors) -> Tensor {
    // Try OpId dispatch first (for backends with optimized einsum, e.g. cuTENSOR)
    if (!tensors.empty()) {
        auto dev_type = tensors[0].device().type;
        auto& table = DispatchTableRegistry::get_table(dev_type);
        if (table.has_kernel(OpId::Einsum)) {
            OpAttributes attrs;
            attrs.set(AttrKey::EinsumEquation, equation);
            std::vector<Tensor> inputs(tensors.begin(), tensors.end());
            auto result = table.dispatch(OpId::Einsum, inputs, attrs);
            return result[0];
        }
    }

    // Fall back to composed implementation
    return einsum_composed(equation, tensors);
}

// ============================================================================
// median — along a dimension
// ============================================================================

auto median(const Tensor& input, int64_t dim, bool keepdim)
    -> std::tuple<Tensor, Tensor> {
    if (!input.is_valid()) {
        throw std::runtime_error("median: uninitialized tensor");
    }

    int64_t ndim = input.ndim();
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) {
        throw std::invalid_argument("median: dim " + std::to_string(dim) + " out of range");
    }

    OpAttributes attrs;
    attrs.set(AttrKey::Dim, dim);
    attrs.set(AttrKey::Keepdim, keepdim);
    std::vector<Tensor> inputs = {input};
    auto results = dispatch<OpId::Median>(inputs, attrs);
    return {results[0], results[1]};
}

// ============================================================================
// mode — most frequent value along a dimension
// ============================================================================

auto mode(const Tensor& input, int64_t dim, bool keepdim)
    -> std::tuple<Tensor, Tensor> {
    if (!input.is_valid()) {
        throw std::runtime_error("mode: uninitialized tensor");
    }

    int64_t ndim = input.ndim();
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) {
        throw std::invalid_argument("mode: dim " + std::to_string(dim) + " out of range");
    }

    OpAttributes attrs;
    attrs.set(AttrKey::Dim, dim);
    attrs.set(AttrKey::Keepdim, keepdim);
    std::vector<Tensor> inputs = {input};
    auto results = dispatch<OpId::Mode>(inputs, attrs);
    return {results[0], results[1]};
}

auto bucketize(const Tensor& input, const Tensor& boundaries, bool right) -> Tensor {
    if (boundaries.ndim() != 1) {
        throw std::invalid_argument("bucketize: boundaries must be a 1-D tensor");
    }

    auto inp = input.contiguous();
    auto bounds = boundaries.contiguous();
    std::array<Tensor, 2> inputs = {inp, bounds};
    NewOpAttributes attrs;
    attrs.set(AttrKey::Right, right);
    return dispatch<OpId::Bucketize>(inputs, attrs)[0];
}

// =========================================================================
// Matrix Construction Operations
// =========================================================================

auto kron(const Tensor& a, const Tensor& b) -> Tensor {
    if (a.ndim() != 2 || b.ndim() != 2) {
        throw std::runtime_error("kron: both tensors must be 2-D");
    }
    int64_t m = a.shape()[0], n = a.shape()[1];
    int64_t p = b.shape()[0], q = b.shape()[1];

    // Overflow guard on the output extents m*p and n*q before they feed
    // reshape(); a wrapped-negative extent is undefined downstream.
    if ((p != 0 && m > std::numeric_limits<int64_t>::max() / p) ||
        (q != 0 && n > std::numeric_limits<int64_t>::max() / q)) {
        throw std::overflow_error("kron: output dimensions overflow int64_t");
    }

    // kron(A, B) = A ⊗ B
    // Reshape A to (m, 1, n, 1), B to (1, p, 1, q), multiply, reshape to (m*p, n*q)
    auto a4 = a.reshape({m, 1, n, 1});
    auto b4 = b.reshape({1, p, 1, q});
    auto product = mul(a4, b4);  // Broadcasting: (m, p, n, q)
    // Need (m, p, n, q) -> (m, n, p, q) -> transpose to interleave correctly
    // Actually kron layout: row (i*p+r), col (j*q+s) = a[i,j]*b[r,s]
    // product[i,r,j,s] = a[i,j]*b[r,s] — already correct for (m,p,n,q)
    // Reshape directly: group (m,p) -> m*p rows, (n,q) -> n*q cols
    // But (m,p,n,q) reshaped to (m*p, n*q) interleaves wrong.
    // Correct: permute to (m,p,n,q) -> already is. reshape to (m*p, n*q) IS correct
    // because C-order layout groups last dims first:
    // [i][r][j][s] -> linear index i*p*n*q + r*n*q + j*q + s
    // Result row = i*p + r, col = j*q + s -> row*n*q + col = (i*p+r)*n*q + j*q + s
    // = i*p*n*q + r*n*q + j*q + s  ✓  Same!
    return product.contiguous().reshape({m * p, n * q});
}

auto block_diag(std::span<const Tensor> tensors) -> Tensor {
    if (tensors.empty()) {
        return zeros({0, 0}, DType::Float32, Device::cpu());
    }

    // Compute total dimensions
    int64_t total_cols = 0;
    std::vector<int64_t> col_sizes;
    for (const auto& t : tensors) {
        if (t.ndim() != 2) {
            throw std::runtime_error("block_diag: all tensors must be 2-D");
        }
        col_sizes.push_back(t.shape()[1]);
        total_cols += t.shape()[1];
    }

    // Build row-by-row: for each tensor, create [zeros_left | tensor_rows | zeros_right]
    std::vector<Tensor> row_blocks;
    int64_t col_offset = 0;
    for (size_t i = 0; i < tensors.size(); ++i) {
        const auto& t = tensors[i];
        int64_t r = t.shape()[0], c = t.shape()[1];
        int64_t left = col_offset;
        int64_t right = total_cols - col_offset - c;

        std::vector<Tensor> parts;
        if (left > 0) parts.push_back(zeros({r, left}, t.dtype(), t.device()));
        parts.push_back(t);
        if (right > 0) parts.push_back(zeros({r, right}, t.dtype(), t.device()));

        row_blocks.push_back(cat(parts, 1));
        col_offset += c;
    }
    return cat(row_blocks, 0);
}

auto vander(const Tensor& x, int64_t N, bool increasing) -> Tensor {
    if (x.ndim() != 1) {
        throw std::runtime_error("vander: input must be 1-D");
    }
    int64_t n = x.shape()[0];
    if (N < 0) N = n;
    if (N == 0) return zeros({n, 0}, x.dtype(), x.device());

    // Build columns: col k = x^k (increasing) or x^(N-1-k) (decreasing)
    std::vector<Tensor> cols;
    cols.reserve(N);
    for (int64_t k = 0; k < N; ++k) {
        int64_t exp = increasing ? k : (N - 1 - k);
        if (exp == 0) {
            cols.push_back(ones({n}, x.dtype(), x.device()));
        } else if (exp == 1) {
            cols.push_back(x.clone());
        } else {
            cols.push_back(pow(x, static_cast<double>(exp)));
        }
    }

    // Stack columns into (n, N)
    return stack(cols, 1);
}

auto cartesian_prod(std::span<const Tensor> tensors) -> Tensor {
    if (tensors.empty()) {
        return zeros({0, 0}, DType::Float32, Device::cpu());
    }
    // Validate that every input is 1-D before any reshaping so the single- and
    // multi-tensor paths share the same clear diagnostic (instead of a confusing
    // reshape size-mismatch on a 2-D-or-higher single input).
    for (const auto& t : tensors) {
        if (t.ndim() != 1) {
            throw std::runtime_error("cartesian_prod: all tensors must be 1-D");
        }
    }
    if (tensors.size() == 1) {
        return tensors[0].reshape({tensors[0].shape()[0], 1});
    }

    // Compute total number of rows, guarding against int64 overflow (a wrapped
    // negative total would feed a corrupt extent into zeros()).
    int64_t total = 1;
    for (const auto& t : tensors) {
        int64_t dim0 = t.shape()[0];
        if (dim0 != 0 && total > std::numeric_limits<int64_t>::max() / dim0) {
            throw std::overflow_error("cartesian_prod: number of rows overflows int64_t");
        }
        total *= dim0;
    }

    // If any input is empty, the cartesian product is empty. Returning before
    // the column loop also avoids integer division-by-zero (SIGFPE) at
    // total / (n * repeat_inner) when some n == 0.
    if (total == 0) {
        return zeros({0, static_cast<int64_t>(tensors.size())},
                     tensors[0].dtype(), tensors[0].device());
    }

    // Build each column using repeat+tile pattern, then stack
    std::vector<Tensor> columns;
    int64_t repeat_inner = 1;
    int64_t num_tensors = static_cast<int64_t>(tensors.size());

    // Process from last to first
    for (int64_t i = num_tensors - 1; i >= 0; --i) {
        int64_t n = tensors[i].shape()[0];
        int64_t repeat_outer = total / (n * repeat_inner);

        // Each element repeated repeat_inner times, then the block repeated repeat_outer times
        auto expanded = tensors[i].unsqueeze(1).expand({n, repeat_inner}).contiguous().reshape({n * repeat_inner});
        auto col = repeat(expanded, {repeat_outer});
        columns.push_back(col);
        repeat_inner *= n;
    }

    // Reverse to get correct order (we built from last to first)
    std::reverse(columns.begin(), columns.end());

    // Stack as columns -> (total, num_tensors)
    return stack(columns, 1);
}

auto combinations(const Tensor& input, int64_t r, bool with_replacement) -> Tensor {
    if (input.ndim() != 1) {
        throw std::runtime_error("combinations: input must be 1-D");
    }
    int64_t n = input.shape()[0];
    if (r < 0) {
        throw std::invalid_argument("combinations: r must be non-negative");
    }
    // r == 0 yields exactly one empty combination (numpy/itertools semantics).
    if (r == 0) {
        return zeros({1, 0}, input.dtype(), input.device());
    }
    if (n == 0) {
        return zeros({0, r}, input.dtype(), input.device());
    }
    // Float16 / BFloat16 aren't supported by the gather; widen and narrow.
    if (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16) {
        const DType orig = input.dtype();
        return combinations(input.to(DType::Float32), r, with_replacement).to(orig);
    }

    // Combinatorial enumeration is inherently host work — the index list is
    // metadata about which input positions form each combination. Build the
    // flat (num_combos * r) Int64 index tensor on CPU, upload to the input's
    // device, then perform the actual data gather as a registered
    // index_select kernel on whichever backend the input lives on. This is
    // the same pattern the rest of the codebase uses for host-built strides
    // / offsets / topology metadata; the *output* tensor is produced on the
    // input device by a real GPU kernel.
    // Bound the number of combinations BEFORE enumerating them into a flat
    // vector — otherwise a large (n, r) silently allocates an enormous buffer
    // (OOM/DoS). Compute C(n,r) (or, with replacement, C(n+r-1, r)) with
    // overflow detection and reject anything that overflows int64.
    {
        int64_t nn = with_replacement ? (n + r - 1) : n;
        // With replacement is always valid; without, r > n yields zero combos.
        if (with_replacement || r <= n) {
            int64_t kk = std::min(r, nn - r);
            int64_t count = 1;
            bool overflow = false;
            for (int64_t i = 0; i < kk && !overflow; ++i) {
                // count = count * (nn - kk + 1 + i) / (i + 1), kept exact.
                int64_t factor = nn - kk + 1 + i;
                if (factor != 0 && count > std::numeric_limits<int64_t>::max() / factor) {
                    overflow = true;
                    break;
                }
                count = count * factor / (i + 1);
            }
            if (overflow ||
                (count != 0 && count > std::numeric_limits<int64_t>::max() / r)) {
                throw std::overflow_error(
                    "combinations: number of combinations is too large to enumerate");
            }
        }
    }

    std::vector<int64_t> flat_indices;
    {
        std::vector<int64_t> current;
        current.reserve(static_cast<size_t>(r));
        std::function<void(int64_t)> generate;
        generate = [&](int64_t start) {
            if (static_cast<int64_t>(current.size()) == r) {
                flat_indices.insert(flat_indices.end(),
                                    current.begin(), current.end());
                return;
            }
            for (int64_t i = start; i < n; ++i) {
                current.push_back(i);
                generate(with_replacement ? i : i + 1);
                current.pop_back();
            }
        };
        generate(0);
    }

    if (flat_indices.empty()) {
        return zeros({0, r}, input.dtype(), input.device());
    }

    const int64_t num_indices = static_cast<int64_t>(flat_indices.size());
    const int64_t num_combos = num_indices / r;

    // Stage indices on CPU, upload, then gather on input.device().
    Tensor idx_cpu = zeros({num_indices}, DType::Int64, Device::cpu());
    std::memcpy(idx_cpu.data<int64_t>(), flat_indices.data(),
                flat_indices.size() * sizeof(int64_t));
    Tensor idx_dev = (input.device().type == Device::Type::CPU)
        ? idx_cpu : idx_cpu.to(input.device());

    // index_select along the only dim of the 1-D input → flat (num_indices,);
    // reshape to (num_combos, r). Both ops have per-backend kernels.
    Tensor gathered = tenzor::index_select(input.contiguous(), /*dim=*/0, idx_dev);
    return tenzor::reshape(gathered, {num_combos, r});
}

// ---------------------------------------------------------------------------
// tensordot — generalized tensor contraction
// ---------------------------------------------------------------------------

auto tensordot(const Tensor& a, const Tensor& b,
               std::vector<int64_t> dims_a,
               std::vector<int64_t> dims_b) -> Tensor {
    if (dims_a.size() != dims_b.size()) {
        throw std::runtime_error(
            "tensordot: dims_a and dims_b must have the same length");
    }

    const int64_t ndim_a = a.ndim();
    const int64_t ndim_b = b.ndim();
    const int64_t n_contract = static_cast<int64_t>(dims_a.size());

    // Normalize negative dims
    for (auto& d : dims_a) {
        if (d < 0) d += ndim_a;
        if (d < 0 || d >= ndim_a) {
            throw std::runtime_error("tensordot: dim out of range for tensor a");
        }
    }
    for (auto& d : dims_b) {
        if (d < 0) d += ndim_b;
        if (d < 0 || d >= ndim_b) {
            throw std::runtime_error("tensordot: dim out of range for tensor b");
        }
    }

    // Validate contracted dimensions match in size
    for (int64_t i = 0; i < n_contract; ++i) {
        if (a.shape()[dims_a[i]] != b.shape()[dims_b[i]]) {
            throw std::runtime_error(
                "tensordot: contracted dimensions must match, got " +
                std::to_string(a.shape()[dims_a[i]]) + " vs " +
                std::to_string(b.shape()[dims_b[i]]));
        }
    }

    // Build permutation for a: free dims first, then contracted dims
    std::vector<bool> a_contracted(ndim_a, false);
    for (auto d : dims_a) a_contracted[d] = true;

    std::vector<int64_t> perm_a;
    std::vector<int64_t> free_shape_a;
    for (int64_t i = 0; i < ndim_a; ++i) {
        if (!a_contracted[i]) {
            perm_a.push_back(i);
            free_shape_a.push_back(a.shape()[i]);
        }
    }
    for (auto d : dims_a) perm_a.push_back(d);

    // Build permutation for b: contracted dims first, then free dims
    std::vector<bool> b_contracted(ndim_b, false);
    for (auto d : dims_b) b_contracted[d] = true;

    std::vector<int64_t> perm_b;
    std::vector<int64_t> free_shape_b;
    for (auto d : dims_b) perm_b.push_back(d);
    for (int64_t i = 0; i < ndim_b; ++i) {
        if (!b_contracted[i]) {
            perm_b.push_back(i);
            free_shape_b.push_back(b.shape()[i]);
        }
    }

    // Compute product of free and contracted dimensions
    int64_t free_size_a = 1;
    for (auto s : free_shape_a) free_size_a *= s;
    int64_t free_size_b = 1;
    for (auto s : free_shape_b) free_size_b *= s;
    int64_t contract_size = 1;
    for (auto d : dims_a) contract_size *= a.shape()[d];

    // Permute -> reshape to 2D -> matmul -> reshape back
    auto a_perm = a.permute(perm_a).contiguous().reshape({free_size_a, contract_size});
    auto b_perm = b.permute(perm_b).contiguous().reshape({contract_size, free_size_b});

    auto result_2d = tenzor::matmul(a_perm, b_perm);

    // Build output shape: free_shape_a + free_shape_b
    std::vector<int64_t> out_shape;
    out_shape.insert(out_shape.end(), free_shape_a.begin(), free_shape_a.end());
    out_shape.insert(out_shape.end(), free_shape_b.begin(), free_shape_b.end());

    if (out_shape.empty()) {
        // Scalar result
        return result_2d.reshape({});
    }
    return result_2d.reshape(out_shape);
}

auto tensordot(const Tensor& a, const Tensor& b, int64_t dims) -> Tensor {
    if (dims < 0) {
        throw std::runtime_error("tensordot: dims must be >= 0");
    }

    const int64_t ndim_a = a.ndim();
    const int64_t ndim_b = b.ndim();

    if (dims > ndim_a || dims > ndim_b) {
        throw std::runtime_error(
            "tensordot: dims cannot exceed number of dimensions of either tensor");
    }

    // Contract last `dims` of a with first `dims` of b
    std::vector<int64_t> dims_a, dims_b;
    for (int64_t i = 0; i < dims; ++i) {
        dims_a.push_back(ndim_a - dims + i);
        dims_b.push_back(i);
    }

    return tensordot(a, b, std::move(dims_a), std::move(dims_b));
}

} // namespace tenzor
