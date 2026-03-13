/**
 * @file advanced.cpp
 * @brief Implementation of advanced tensor operations
 */

#include "tenzor/ops/advanced.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/transform.hpp"
#include "tenzor/ops/reduction.hpp"
#include "tenzor/backend/dispatch.hpp"
#include "tenzor/backend/fast_dispatch.hpp"
#include "tenzor/backend/op_attributes.hpp"
#include "tenzor/ops/op_id.hpp"
#include "tenzor/core/shape.hpp"
#include "tenzor/ops/math.hpp"
#include <algorithm>
#include <numeric>
#include <string>
#include <unordered_map>
#include <cstring>
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

    // For non-CPU tensors, dispatch to backend kernel
    if (input.device().type != Device::Type::CPU) {
        OpAttributes attrs;
        attrs.set(AttrKey::K, k);
        attrs.set(AttrKey::Dim, dim);
        attrs.set(AttrKey::Largest, largest);
        attrs.set(AttrKey::Sorted, sorted);
        std::vector<Tensor> inputs = {input};
        auto results = dispatch<OpId::TopK>(inputs, attrs);
        return {results[0], results[1]};
    }

    // Get contiguous input
    Tensor input_cont = input.is_contiguous() ? input : input.contiguous();

    // Create output tensors
    std::vector<int64_t> output_shape(input.shape().begin(), input.shape().end());
    output_shape[dim] = k;

    Tensor values(output_shape, input.dtype(), input.device());
    Tensor indices(output_shape, DType::Int64, input.device());

    // Number of slices to process
    int64_t outer_size = 1;
    for (int64_t i = 0; i < dim; ++i) {
        outer_size *= input.shape()[i];
    }

    int64_t inner_size = 1;
    for (int64_t i = dim + 1; i < ndim; ++i) {
        inner_size *= input.shape()[i];
    }

    // Helper lambda to process topk for any numeric type
    auto process_topk = [&]<typename T>(T*) {
        const T* input_data = input_cont.data<T>();
        T* values_data = values.data<T>();
        int64_t* indices_data = indices.data<int64_t>();

        for (int64_t outer = 0; outer < outer_size; ++outer) {
            for (int64_t inner = 0; inner < inner_size; ++inner) {
                // Create index-value pairs for this slice
                std::vector<std::pair<T, int64_t>> pairs;
                pairs.reserve(dim_size);

                for (int64_t i = 0; i < dim_size; ++i) {
                    int64_t offset = outer * dim_size * inner_size + i * inner_size + inner;
                    pairs.emplace_back(input_data[offset], i);
                }

                // Partial sort to get top-k
                if (largest) {
                    std::partial_sort(pairs.begin(), pairs.begin() + k, pairs.end(),
                        [](const auto& a, const auto& b) { return a.first > b.first; });
                } else {
                    std::partial_sort(pairs.begin(), pairs.begin() + k, pairs.end(),
                        [](const auto& a, const auto& b) { return a.first < b.first; });
                }

                // Optionally sort the top-k
                if (sorted) {
                    if (largest) {
                        std::sort(pairs.begin(), pairs.begin() + k,
                            [](const auto& a, const auto& b) { return a.first > b.first; });
                    } else {
                        std::sort(pairs.begin(), pairs.begin() + k,
                            [](const auto& a, const auto& b) { return a.first < b.first; });
                    }
                }

                // Write results
                for (int64_t i = 0; i < k; ++i) {
                    int64_t out_offset = outer * k * inner_size + i * inner_size + inner;
                    values_data[out_offset] = pairs[i].first;
                    indices_data[out_offset] = pairs[i].second;
                }
            }
        }
    };

    // Process each slice along the specified dimension
    switch (input.dtype()) {
        case DType::Float32:
            process_topk(static_cast<float*>(nullptr));
            break;
        case DType::Float64:
            process_topk(static_cast<double*>(nullptr));
            break;
        case DType::Int32:
            process_topk(static_cast<int32_t*>(nullptr));
            break;
        case DType::Int64:
            process_topk(static_cast<int64_t*>(nullptr));
            break;
        default:
            throw std::runtime_error("topk only supports Float32, Float64, Int32, and Int64 dtypes");
    }

    return {values, indices};
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

    // For non-CPU tensors, dispatch to backend kernel
    if (input.device().type != Device::Type::CPU) {
        OpAttributes attrs;
        attrs.set(AttrKey::Dim, dim);
        attrs.set(AttrKey::Descending, descending);
        std::vector<Tensor> inputs = {input};
        auto results = dispatch<OpId::Sort>(inputs, attrs);
        return {results[0], results[1]};
    }

    // Get contiguous input
    Tensor input_cont = input.is_contiguous() ? input : input.contiguous();

    // Create output tensors with same shape as input
    Tensor values(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                  input.dtype(), input.device());
    Tensor indices(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                   DType::Int64, input.device());

    const int64_t dim_size = input.shape()[dim];

    // Number of slices to process
    int64_t outer_size = 1;
    for (int64_t i = 0; i < dim; ++i) {
        outer_size *= input.shape()[i];
    }

    int64_t inner_size = 1;
    for (int64_t i = dim + 1; i < ndim; ++i) {
        inner_size *= input.shape()[i];
    }

    // Helper lambda to process sort for any numeric type
    auto process_sort = [&]<typename T>(T*) {
        const T* input_data = input_cont.data<T>();
        T* values_data = values.data<T>();
        int64_t* indices_data = indices.data<int64_t>();

        for (int64_t outer = 0; outer < outer_size; ++outer) {
            for (int64_t inner = 0; inner < inner_size; ++inner) {
                // Create index-value pairs
                std::vector<std::pair<T, int64_t>> pairs;
                pairs.reserve(dim_size);

                for (int64_t i = 0; i < dim_size; ++i) {
                    int64_t offset = outer * dim_size * inner_size + i * inner_size + inner;
                    pairs.emplace_back(input_data[offset], i);
                }

                // Sort
                if (descending) {
                    std::sort(pairs.begin(), pairs.end(),
                        [](const auto& a, const auto& b) { return a.first > b.first; });
                } else {
                    std::sort(pairs.begin(), pairs.end(),
                        [](const auto& a, const auto& b) { return a.first < b.first; });
                }

                // Write results
                for (int64_t i = 0; i < dim_size; ++i) {
                    int64_t offset = outer * dim_size * inner_size + i * inner_size + inner;
                    values_data[offset] = pairs[i].first;
                    indices_data[offset] = pairs[i].second;
                }
            }
        }
    };

    // Process each slice based on dtype
    switch (input.dtype()) {
        case DType::Float32:
            process_sort(static_cast<float*>(nullptr));
            break;
        case DType::Float64:
            process_sort(static_cast<double*>(nullptr));
            break;
        case DType::Int32:
            process_sort(static_cast<int32_t*>(nullptr));
            break;
        case DType::Int64:
            process_sort(static_cast<int64_t*>(nullptr));
            break;
        default:
            throw std::runtime_error("sort only supports Float32, Float64, Int32, and Int64 dtypes");
    }

    return {values, indices};
}

auto unique(const Tensor& input,
            bool sorted_output,
            bool return_inverse,
            bool return_counts) -> std::tuple<Tensor, Tensor, Tensor> {

    // For non-CPU tensors, dispatch to backend kernel
    if (input.device().type != Device::Type::CPU) {
        OpAttributes attrs;
        attrs.set(AttrKey::Sorted, sorted_output);
        attrs.set(AttrKey::ReturnInverse, return_inverse);
        attrs.set(AttrKey::ReturnCounts, return_counts);
        std::vector<Tensor> inputs = {input};
        auto results = dispatch<OpId::Unique>(inputs, attrs);
        return {results[0], results[1], results[2]};
    }

    // Get contiguous flattened input
    Tensor input_flat = input.flatten().contiguous();
    const int64_t numel = input_flat.numel();
    const DType dtype = input.dtype();

    // Helper lambda to process unique for any numeric type
    auto process_unique = [&]<typename T>(T*) -> std::tuple<Tensor, Tensor, Tensor> {
        const T* data = input_flat.data<T>();

        // Map from value to (first_index, count)
        std::vector<std::pair<T, std::pair<int64_t, int64_t>>> value_info;
        std::unordered_map<T, size_t> value_to_idx;

        for (int64_t i = 0; i < numel; ++i) {
            T val = data[i];
            auto it = value_to_idx.find(val);
            if (it == value_to_idx.end()) {
                value_to_idx[val] = value_info.size();
                value_info.emplace_back(val, std::make_pair(i, 1));
            } else {
                value_info[it->second].second.second++;
            }
        }

        const int64_t num_unique = static_cast<int64_t>(value_info.size());

        // Sort if requested
        if (sorted_output) {
            std::sort(value_info.begin(), value_info.end(),
                [](const auto& a, const auto& b) {
                    if constexpr (std::is_same_v<T, Float16> || std::is_same_v<T, BFloat16>) {
                        return static_cast<float>(a.first) < static_cast<float>(b.first);
                    } else {
                        return a.first < b.first;
                    }
                });
        }

        // Create unique values tensor
        Tensor unique_vals({num_unique}, dtype, Device::cpu());
        T* unique_data = unique_vals.data<T>();

        for (int64_t i = 0; i < num_unique; ++i) {
            unique_data[i] = value_info[i].first;
        }

        // Create inverse indices if requested
        Tensor inverse_indices;
        if (return_inverse) {
            inverse_indices = Tensor({numel}, DType::Int64, Device::cpu());
            int64_t* inverse_data = inverse_indices.data<int64_t>();

            // Build value to output index map
            std::unordered_map<T, int64_t> val_to_out_idx;
            for (int64_t i = 0; i < num_unique; ++i) {
                val_to_out_idx[value_info[i].first] = i;
            }

            for (int64_t i = 0; i < numel; ++i) {
                inverse_data[i] = val_to_out_idx[data[i]];
            }
        }

        // Create counts if requested
        Tensor counts;
        if (return_counts) {
            counts = Tensor({num_unique}, DType::Int64, Device::cpu());
            int64_t* counts_data = counts.data<int64_t>();

            for (int64_t i = 0; i < num_unique; ++i) {
                counts_data[i] = value_info[i].second.second;
            }
        }

        return {unique_vals, inverse_indices, counts};
    };

    switch (dtype) {
        case DType::Float32:
            return process_unique(static_cast<float*>(nullptr));
        case DType::Float64:
            return process_unique(static_cast<double*>(nullptr));
        case DType::Int32:
            return process_unique(static_cast<int32_t*>(nullptr));
        case DType::Int64:
            return process_unique(static_cast<int64_t*>(nullptr));
        case DType::Bool:
            return process_unique(static_cast<bool*>(nullptr));
        case DType::Float16:
            return process_unique(static_cast<Float16*>(nullptr));
        case DType::BFloat16:
            return process_unique(static_cast<BFloat16*>(nullptr));
        default:
            throw std::runtime_error("unique: unsupported dtype");
    }
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

    // For non-CPU tensors, dispatch to backend kernel
    if (input.device().type != Device::Type::CPU) {
        NewOpAttributes attrs;
        attrs.set(AttrKey::Dim, dim);
        std::vector<Tensor> inputs = {input};
        return dispatch<OpId::CumSum>(inputs, attrs)[0];
    }

    // Get contiguous input
    Tensor input_cont = input.is_contiguous() ? input : input.contiguous();

    // Create output tensor
    Tensor output(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                  input.dtype(), input.device());

    const int64_t dim_size = input.shape()[dim];

    // Number of slices to process
    int64_t outer_size = 1;
    for (int64_t i = 0; i < dim; ++i) {
        outer_size *= input.shape()[i];
    }

    int64_t inner_size = 1;
    for (int64_t i = dim + 1; i < ndim; ++i) {
        inner_size *= input.shape()[i];
    }

    // Helper lambda for cumsum
    auto process_cumsum = [&]<typename T>(T*) {
        const T* input_data = input_cont.data<T>();
        T* output_data = output.data<T>();

        for (int64_t outer = 0; outer < outer_size; ++outer) {
            for (int64_t inner = 0; inner < inner_size; ++inner) {
                T cumsum_val = static_cast<T>(0);

                for (int64_t i = 0; i < dim_size; ++i) {
                    int64_t offset = outer * dim_size * inner_size + i * inner_size + inner;
                    cumsum_val += input_data[offset];
                    output_data[offset] = cumsum_val;
                }
            }
        }
    };

    switch (input.dtype()) {
        case DType::Float32:
            process_cumsum(static_cast<float*>(nullptr));
            break;
        case DType::Float64:
            process_cumsum(static_cast<double*>(nullptr));
            break;
        case DType::Int32:
            process_cumsum(static_cast<int32_t*>(nullptr));
            break;
        case DType::Int64:
            process_cumsum(static_cast<int64_t*>(nullptr));
            break;
        default:
            throw std::runtime_error("cumsum only supports Float32, Float64, Int32, and Int64 dtypes");
    }

    return output;
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

    // For non-CPU tensors, dispatch to backend kernel
    if (input.device().type != Device::Type::CPU) {
        NewOpAttributes attrs;
        attrs.set(AttrKey::Dim, dim);
        std::vector<Tensor> inputs = {input};
        return dispatch<OpId::CumProd>(inputs, attrs)[0];
    }

    // Get contiguous input
    Tensor input_cont = input.is_contiguous() ? input : input.contiguous();

    // Create output tensor
    Tensor output(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                  input.dtype(), input.device());

    const int64_t dim_size = input.shape()[dim];

    // Number of slices to process
    int64_t outer_size = 1;
    for (int64_t i = 0; i < dim; ++i) {
        outer_size *= input.shape()[i];
    }

    int64_t inner_size = 1;
    for (int64_t i = dim + 1; i < ndim; ++i) {
        inner_size *= input.shape()[i];
    }

    // Helper lambda for cumprod
    auto process_cumprod = [&]<typename T>(T*) {
        const T* input_data = input_cont.data<T>();
        T* output_data = output.data<T>();

        for (int64_t outer = 0; outer < outer_size; ++outer) {
            for (int64_t inner = 0; inner < inner_size; ++inner) {
                T cumprod_val = static_cast<T>(1);

                for (int64_t i = 0; i < dim_size; ++i) {
                    int64_t offset = outer * dim_size * inner_size + i * inner_size + inner;
                    cumprod_val *= input_data[offset];
                    output_data[offset] = cumprod_val;
                }
            }
        }
    };

    switch (input.dtype()) {
        case DType::Float32:
            process_cumprod(static_cast<float*>(nullptr));
            break;
        case DType::Float64:
            process_cumprod(static_cast<double*>(nullptr));
            break;
        case DType::Int32:
            process_cumprod(static_cast<int32_t*>(nullptr));
            break;
        case DType::Int64:
            process_cumprod(static_cast<int64_t*>(nullptr));
            break;
        default:
            throw std::runtime_error("cumprod only supports Float32, Float64, Int32, and Int64 dtypes");
    }

    return output;
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

auto einsum(const std::string& equation,
            std::span<const Tensor> tensors) -> Tensor {
    auto [input_subs, output_sub] = parse_einsum_equation(equation);

    if (input_subs.size() != tensors.size()) {
        throw std::invalid_argument("einsum: number of subscripts (" +
            std::to_string(input_subs.size()) + ") does not match number of tensors (" +
            std::to_string(tensors.size()) + ")");
    }

    // Fast paths for common patterns
    if (tensors.size() == 2) {
        const auto& a = tensors[0];
        const auto& b = tensors[1];
        const auto& sa = input_subs[0];
        const auto& sb = input_subs[1];

        // Matrix multiply: ij,jk->ik
        if (sa == "ij" && sb == "jk" && output_sub == "ik") {
            return matmul(a, b);
        }
        // Batch matmul: bij,bjk->bik
        if (sa == "bij" && sb == "bjk" && output_sub == "bik") {
            return bmm(a, b);
        }
        // Dot product: i,i->
        if (sa == "i" && sb == "i" && output_sub.empty()) {
            return dot(a, b);
        }
        // Outer product: i,j->ij
        if (sa == "i" && sb == "j" && output_sub == "ij") {
            auto a_col = reshape(a, {a.numel(), 1});
            auto b_row = reshape(b, {1, b.numel()});
            return matmul(a_col, b_row);
        }
    }
    if (tensors.size() == 1) {
        const auto& a = tensors[0];
        const auto& sa = input_subs[0];

        // Trace: ii->
        if (sa == "ii" && output_sub.empty()) {
            return trace(a);
        }
        // Diagonal: ii->i
        if (sa == "ii" && output_sub == "i") {
            return diag(a);
        }
    }

    // General path: use the transpose-reshape-contract approach
    // 1. Build label→dimension size mapping
    std::unordered_map<char, int64_t> label_sizes;
    for (size_t t = 0; t < tensors.size(); ++t) {
        auto shape = tensors[t].shape();
        if (input_subs[t].size() != static_cast<size_t>(tensors[t].ndim())) {
            throw std::invalid_argument("einsum: subscript '" + input_subs[t] +
                "' has " + std::to_string(input_subs[t].size()) +
                " labels but tensor has " + std::to_string(tensors[t].ndim()) + " dims");
        }
        for (size_t d = 0; d < input_subs[t].size(); ++d) {
            char label = input_subs[t][d];
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
    Tensor result = align_tensor(tensors[0], input_subs[0]);
    for (size_t t = 1; t < tensors.size(); ++t) {
        Tensor aligned = align_tensor(tensors[t], input_subs[t]);
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

} // namespace tenzor
