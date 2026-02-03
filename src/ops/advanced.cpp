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
#include "tenzor/ops/op_id.hpp"
#include "tenzor/core/shape.hpp"
#include <algorithm>
#include <numeric>
#include <unordered_map>
#include <cstring>

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
        attrs["k"] = std::to_string(k);
        attrs["dim"] = std::to_string(dim);
        attrs["largest"] = largest ? "1" : "0";
        attrs["sorted"] = sorted ? "1" : "0";
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
        attrs["dim"] = std::to_string(dim);
        attrs["descending"] = descending ? "1" : "0";
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
        attrs["sorted"] = sorted_output ? "1" : "0";
        attrs["return_inverse"] = return_inverse ? "1" : "0";
        attrs["return_counts"] = return_counts ? "1" : "0";
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
                [](const auto& a, const auto& b) { return a.first < b.first; });
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
        default:
            throw std::runtime_error("unique only supports Float32, Float64, Int32, Int64, and Bool dtypes");
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
        OpAttributes attrs;
        attrs["dim"] = std::to_string(dim);
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
        OpAttributes attrs;
        attrs["dim"] = std::to_string(dim);
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

} // namespace tenzor
