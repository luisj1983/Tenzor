/**
 * @file advanced.cpp
 * @brief CPU kernels for advanced operations (TopK, Sort, CumSum, CumProd, Unique)
 *
 * These operations provide parity with the CUDA backend's advanced.cu.
 */

#include "tenzor/core/tensor.hpp"
#include "tenzor/core/dtype.hpp"
#include "tenzor/core/shape.hpp"
#include <algorithm>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#ifdef _OPENMP
    #include <omp.h>
#endif

namespace tenzor {
namespace cpu {

namespace {

template<typename T>
auto topk_impl(const T* data, int64_t numel, int64_t dim_size,
               int64_t outer_size, int64_t inner_size,
               int64_t k, bool largest, bool sorted,
               T* out_values, int64_t* out_indices) -> void {

    #pragma omp parallel for if(outer_size * inner_size > 1024)
    for (int64_t outer = 0; outer < outer_size; ++outer) {
        for (int64_t inner = 0; inner < inner_size; ++inner) {
            // Collect (value, index) pairs along the dim
            std::vector<std::pair<T, int64_t>> pairs(dim_size);
            for (int64_t d = 0; d < dim_size; ++d) {
                int64_t idx = (outer * dim_size + d) * inner_size + inner;
                pairs[d] = {data[idx], d};
            }

            // Partial sort for top-k
            if (largest) {
                std::partial_sort(pairs.begin(), pairs.begin() + k, pairs.end(),
                    [](const auto& a, const auto& b) { return a.first > b.first; });
            } else {
                std::partial_sort(pairs.begin(), pairs.begin() + k, pairs.end(),
                    [](const auto& a, const auto& b) { return a.first < b.first; });
            }

            if (sorted && largest) {
                std::sort(pairs.begin(), pairs.begin() + k,
                    [](const auto& a, const auto& b) { return a.first > b.first; });
            } else if (sorted) {
                std::sort(pairs.begin(), pairs.begin() + k,
                    [](const auto& a, const auto& b) { return a.first < b.first; });
            }

            // Write output
            for (int64_t i = 0; i < k; ++i) {
                int64_t out_idx = (outer * k + i) * inner_size + inner;
                out_values[out_idx] = pairs[i].first;
                out_indices[out_idx] = pairs[i].second;
            }
        }
    }
}

template<typename T>
auto sort_impl(const T* data, int64_t dim_size,
               int64_t outer_size, int64_t inner_size,
               bool descending,
               T* out_values, int64_t* out_indices) -> void {

    #pragma omp parallel for if(outer_size * inner_size > 1024)
    for (int64_t outer = 0; outer < outer_size; ++outer) {
        for (int64_t inner = 0; inner < inner_size; ++inner) {
            std::vector<std::pair<T, int64_t>> pairs(dim_size);
            for (int64_t d = 0; d < dim_size; ++d) {
                int64_t idx = (outer * dim_size + d) * inner_size + inner;
                pairs[d] = {data[idx], d};
            }

            if (descending) {
                std::stable_sort(pairs.begin(), pairs.end(),
                    [](const auto& a, const auto& b) { return a.first > b.first; });
            } else {
                std::stable_sort(pairs.begin(), pairs.end(),
                    [](const auto& a, const auto& b) { return a.first < b.first; });
            }

            for (int64_t d = 0; d < dim_size; ++d) {
                int64_t idx = (outer * dim_size + d) * inner_size + inner;
                out_values[idx] = pairs[d].first;
                out_indices[idx] = pairs[d].second;
            }
        }
    }
}

template<typename T>
auto cumsum_impl(const T* data, T* output, int64_t dim_size,
                 int64_t outer_size, int64_t inner_size) -> void {
    #pragma omp parallel for if(outer_size * inner_size > 4096)
    for (int64_t outer = 0; outer < outer_size; ++outer) {
        for (int64_t inner = 0; inner < inner_size; ++inner) {
            T running = T(0);
            for (int64_t d = 0; d < dim_size; ++d) {
                int64_t idx = (outer * dim_size + d) * inner_size + inner;
                running += data[idx];
                output[idx] = running;
            }
        }
    }
}

template<typename T>
auto cumprod_impl(const T* data, T* output, int64_t dim_size,
                  int64_t outer_size, int64_t inner_size) -> void {
    #pragma omp parallel for if(outer_size * inner_size > 4096)
    for (int64_t outer = 0; outer < outer_size; ++outer) {
        for (int64_t inner = 0; inner < inner_size; ++inner) {
            T running = T(1);
            for (int64_t d = 0; d < dim_size; ++d) {
                int64_t idx = (outer * dim_size + d) * inner_size + inner;
                running *= data[idx];
                output[idx] = running;
            }
        }
    }
}

// Helper to get dim decomposition for axis operations
struct DimDecomp {
    int64_t outer_size;
    int64_t dim_size;
    int64_t inner_size;
};

auto decompose_dim(const Tensor& t, int64_t dim) -> DimDecomp {
    const auto& shape = t.shape();
    int64_t ndim = t.ndim();
    if (dim < 0) dim += ndim;

    DimDecomp d;
    d.dim_size = shape[dim];
    d.outer_size = 1;
    for (int64_t i = 0; i < dim; ++i) d.outer_size *= shape[i];
    d.inner_size = 1;
    for (int64_t i = dim + 1; i < ndim; ++i) d.inner_size *= shape[i];
    return d;
}

} // anonymous namespace

auto topk_kernel(const Tensor& input, int64_t k, int64_t dim,
                 bool largest, bool sorted) -> std::pair<Tensor, Tensor> {
    if (dim < 0) dim += input.ndim();
    auto [outer_size, dim_size, inner_size] = decompose_dim(input, dim);

    if (k > dim_size) {
        throw std::runtime_error("topk: k is too large for the given dimension");
    }

    Tensor cont = input.is_contiguous() ? input : input.contiguous();
    std::vector<int64_t> out_shape(input.shape().begin(), input.shape().end());
    out_shape[dim] = k;

    Tensor values(out_shape, input.dtype(), input.device());
    Tensor indices(out_shape, DType::Int64, input.device());

    if (input.dtype() == DType::Float32) {
        topk_impl(cont.data<float>(), cont.numel(), dim_size,
                  outer_size, inner_size, k, largest, sorted,
                  values.data<float>(), indices.data<int64_t>());
    } else if (input.dtype() == DType::Float64) {
        topk_impl(cont.data<double>(), cont.numel(), dim_size,
                  outer_size, inner_size, k, largest, sorted,
                  values.data<double>(), indices.data<int64_t>());
    } else if (input.dtype() == DType::Int32) {
        topk_impl(cont.data<int32_t>(), cont.numel(), dim_size,
                  outer_size, inner_size, k, largest, sorted,
                  values.data<int32_t>(), indices.data<int64_t>());
    } else if (input.dtype() == DType::Int64) {
        topk_impl(cont.data<int64_t>(), cont.numel(), dim_size,
                  outer_size, inner_size, k, largest, sorted,
                  values.data<int64_t>(), indices.data<int64_t>());
    } else if (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16) {
        DType orig = input.dtype();
        Tensor cont_f32 = cont.to(DType::Float32);
        Tensor values_f32(out_shape, DType::Float32, input.device());
        topk_impl(cont_f32.data<float>(), cont_f32.numel(), dim_size,
                  outer_size, inner_size, k, largest, sorted,
                  values_f32.data<float>(), indices.data<int64_t>());
        values = values_f32.to(orig);
    } else {
        throw std::runtime_error("topk: unsupported dtype");
    }

    return {values, indices};
}

auto sort_kernel(const Tensor& input, int64_t dim,
                 bool descending) -> std::pair<Tensor, Tensor> {
    if (dim < 0) dim += input.ndim();
    auto [outer_size, dim_size, inner_size] = decompose_dim(input, dim);

    Tensor cont = input.is_contiguous() ? input : input.contiguous();
    Tensor values(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                  input.dtype(), input.device());
    Tensor indices(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                   DType::Int64, input.device());

    if (input.dtype() == DType::Float32) {
        sort_impl(cont.data<float>(), dim_size, outer_size, inner_size, descending,
                  values.data<float>(), indices.data<int64_t>());
    } else if (input.dtype() == DType::Float64) {
        sort_impl(cont.data<double>(), dim_size, outer_size, inner_size, descending,
                  values.data<double>(), indices.data<int64_t>());
    } else if (input.dtype() == DType::Int32) {
        sort_impl(cont.data<int32_t>(), dim_size, outer_size, inner_size, descending,
                  values.data<int32_t>(), indices.data<int64_t>());
    } else if (input.dtype() == DType::Int64) {
        sort_impl(cont.data<int64_t>(), dim_size, outer_size, inner_size, descending,
                  values.data<int64_t>(), indices.data<int64_t>());
    } else if (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16) {
        DType orig = input.dtype();
        Tensor cont_f32 = cont.to(DType::Float32);
        Tensor values_f32(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                          DType::Float32, input.device());
        sort_impl(cont_f32.data<float>(), dim_size, outer_size, inner_size, descending,
                  values_f32.data<float>(), indices.data<int64_t>());
        values = values_f32.to(orig);
    } else {
        throw std::runtime_error("sort: unsupported dtype");
    }

    return {values, indices};
}

auto cumsum_kernel(const Tensor& input, int64_t dim) -> Tensor {
    if (dim < 0) dim += input.ndim();
    auto [outer_size, dim_size, inner_size] = decompose_dim(input, dim);

    Tensor cont = input.is_contiguous() ? input : input.contiguous();
    Tensor output(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                  input.dtype(), input.device());

    if (input.dtype() == DType::Float32) {
        cumsum_impl(cont.data<float>(), output.data<float>(),
                    dim_size, outer_size, inner_size);
    } else if (input.dtype() == DType::Float64) {
        cumsum_impl(cont.data<double>(), output.data<double>(),
                    dim_size, outer_size, inner_size);
    } else if (input.dtype() == DType::Int32) {
        cumsum_impl(cont.data<int32_t>(), output.data<int32_t>(),
                    dim_size, outer_size, inner_size);
    } else if (input.dtype() == DType::Int64) {
        cumsum_impl(cont.data<int64_t>(), output.data<int64_t>(),
                    dim_size, outer_size, inner_size);
    } else if (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16) {
        DType orig = input.dtype();
        Tensor cont_f32 = cont.to(DType::Float32);
        Tensor output_f32(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                          DType::Float32, input.device());
        cumsum_impl(cont_f32.data<float>(), output_f32.data<float>(),
                    dim_size, outer_size, inner_size);
        output = output_f32.to(orig);
    } else {
        throw std::runtime_error("cumsum: unsupported dtype");
    }

    return output;
}

auto cumprod_kernel(const Tensor& input, int64_t dim) -> Tensor {
    if (dim < 0) dim += input.ndim();
    auto [outer_size, dim_size, inner_size] = decompose_dim(input, dim);

    Tensor cont = input.is_contiguous() ? input : input.contiguous();
    Tensor output(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                  input.dtype(), input.device());

    if (input.dtype() == DType::Float32) {
        cumprod_impl(cont.data<float>(), output.data<float>(),
                     dim_size, outer_size, inner_size);
    } else if (input.dtype() == DType::Float64) {
        cumprod_impl(cont.data<double>(), output.data<double>(),
                     dim_size, outer_size, inner_size);
    } else if (input.dtype() == DType::Int32) {
        cumprod_impl(cont.data<int32_t>(), output.data<int32_t>(),
                     dim_size, outer_size, inner_size);
    } else if (input.dtype() == DType::Int64) {
        cumprod_impl(cont.data<int64_t>(), output.data<int64_t>(),
                     dim_size, outer_size, inner_size);
    } else if (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16) {
        DType orig = input.dtype();
        Tensor cont_f32 = cont.to(DType::Float32);
        Tensor output_f32(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                          DType::Float32, input.device());
        cumprod_impl(cont_f32.data<float>(), output_f32.data<float>(),
                     dim_size, outer_size, inner_size);
        output = output_f32.to(orig);
    } else {
        throw std::runtime_error("cumprod: unsupported dtype");
    }

    return output;
}

auto unique_kernel(const Tensor& input, bool sorted_output,
                   bool return_inverse, bool return_counts)
    -> std::tuple<Tensor, Tensor, Tensor> {

    Tensor cont = input.is_contiguous() ? input : input.contiguous();
    const int64_t numel = cont.numel();

    // Work with Float32 for simplicity (most common case)
    // This handles all numeric dtypes by working on sorted copies
    auto do_unique = [&]<typename T>(const T* data) -> std::tuple<Tensor, Tensor, Tensor> {
        // Build sorted index array
        std::vector<int64_t> sort_idx(numel);
        std::iota(sort_idx.begin(), sort_idx.end(), 0);

        if (sorted_output) {
            std::stable_sort(sort_idx.begin(), sort_idx.end(),
                [data](int64_t a, int64_t b) { return data[a] < data[b]; });
        }

        // Find unique elements
        std::vector<T> unique_vals;
        std::vector<int64_t> inverse_map(numel);
        std::vector<int64_t> counts_vec;

        unique_vals.reserve(numel);

        if (sorted_output) {
            unique_vals.push_back(data[sort_idx[0]]);
            int64_t cur_count = 1;

            // Map from original index to unique index
            std::vector<int64_t> sorted_unique_idx(numel);
            sorted_unique_idx[0] = 0;

            for (int64_t i = 1; i < numel; ++i) {
                if (data[sort_idx[i]] != data[sort_idx[i - 1]]) {
                    if (return_counts) counts_vec.push_back(cur_count);
                    cur_count = 1;
                    unique_vals.push_back(data[sort_idx[i]]);
                } else {
                    ++cur_count;
                }
                sorted_unique_idx[i] = static_cast<int64_t>(unique_vals.size()) - 1;
            }
            if (return_counts) counts_vec.push_back(cur_count);

            // Build inverse map in original order
            if (return_inverse) {
                for (int64_t i = 0; i < numel; ++i) {
                    inverse_map[sort_idx[i]] = sorted_unique_idx[i];
                }
            }
        } else {
            // Unsorted unique: preserve order of first appearance
            // Use unordered_map with bit-pattern keys for O(n) amortized lookup
            std::unordered_map<uint64_t, int64_t> seen;
            seen.reserve(std::min(numel, int64_t(65536)));
            for (int64_t i = 0; i < numel; ++i) {
                uint64_t key = 0;
                std::memcpy(&key, &data[i], sizeof(T));
                auto [it, inserted] = seen.try_emplace(key, static_cast<int64_t>(unique_vals.size()));
                if (inserted) {
                    unique_vals.push_back(data[i]);
                    if (return_counts) counts_vec.push_back(1);
                } else {
                    if (return_counts) counts_vec[it->second]++;
                }
                if (return_inverse) inverse_map[i] = it->second;
            }
        }

        int64_t n_unique = static_cast<int64_t>(unique_vals.size());

        // Build output tensors
        Tensor unique_out({n_unique}, input.dtype(), input.device());
        T* unique_data = unique_out.template data<T>();
        std::memcpy(unique_data, unique_vals.data(), n_unique * sizeof(T));

        Tensor inverse_out;
        if (return_inverse) {
            inverse_out = Tensor({numel}, DType::Int64, input.device());
            std::memcpy(inverse_out.data<int64_t>(), inverse_map.data(), numel * sizeof(int64_t));
        } else {
            inverse_out = Tensor({0}, DType::Int64, input.device());
        }

        Tensor counts_out;
        if (return_counts) {
            counts_out = Tensor({n_unique}, DType::Int64, input.device());
            std::memcpy(counts_out.data<int64_t>(), counts_vec.data(), n_unique * sizeof(int64_t));
        } else {
            counts_out = Tensor({0}, DType::Int64, input.device());
        }

        return {unique_out, inverse_out, counts_out};
    };

    if (input.dtype() == DType::Float32) {
        return do_unique(cont.data<float>());
    } else if (input.dtype() == DType::Float64) {
        return do_unique(cont.data<double>());
    } else if (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16) {
        DType orig = input.dtype();
        Tensor cont_f32 = cont.to(DType::Float32);
        auto [unique_f32, inverse_out, counts_out] = do_unique(cont_f32.data<float>());
        return {unique_f32.to(orig), inverse_out, counts_out};
    } else if (input.dtype() == DType::Int32) {
        return do_unique(cont.data<int32_t>());
    } else if (input.dtype() == DType::Int64) {
        return do_unique(cont.data<int64_t>());
    } else if (input.dtype() == DType::Int8) {
        return do_unique(cont.data<int8_t>());
    } else if (input.dtype() == DType::UInt8) {
        return do_unique(cont.data<uint8_t>());
    } else if (input.dtype() == DType::Bool) {
        // vector<bool> is special in C++, use uint8_t internally
        const bool* bdata = cont.data<bool>();
        // Bool has at most 2 unique values
        std::vector<uint8_t> seen_vals;
        std::vector<int64_t> inverse_map(numel);
        std::vector<int64_t> counts_vec;

        std::unordered_map<int, int64_t> seen;
        for (int64_t i = 0; i < numel; ++i) {
            int key = bdata[i] ? 1 : 0;
            auto [it, inserted] = seen.try_emplace(key, static_cast<int64_t>(seen_vals.size()));
            if (inserted) {
                seen_vals.push_back(static_cast<uint8_t>(bdata[i]));
                if (return_counts) counts_vec.push_back(1);
            } else {
                if (return_counts) counts_vec[it->second]++;
            }
            if (return_inverse) inverse_map[i] = it->second;
        }

        if (sorted_output && seen_vals.size() == 2 && seen_vals[0]) {
            // Swap so false comes first
            std::swap(seen_vals[0], seen_vals[1]);
            if (return_counts) std::swap(counts_vec[0], counts_vec[1]);
            if (return_inverse) {
                for (auto& v : inverse_map) v = 1 - v;
            }
        }

        int64_t n_unique = static_cast<int64_t>(seen_vals.size());
        Tensor unique_out({n_unique}, DType::Bool, input.device());
        bool* udata = unique_out.data<bool>();
        for (int64_t i = 0; i < n_unique; ++i) udata[i] = static_cast<bool>(seen_vals[i]);

        Tensor inverse_out = return_inverse
            ? Tensor({numel}, DType::Int64, input.device())
            : Tensor({0}, DType::Int64, input.device());
        if (return_inverse) {
            std::memcpy(inverse_out.data<int64_t>(), inverse_map.data(), numel * sizeof(int64_t));
        }

        Tensor counts_out = return_counts
            ? Tensor({n_unique}, DType::Int64, input.device())
            : Tensor({0}, DType::Int64, input.device());
        if (return_counts) {
            std::memcpy(counts_out.data<int64_t>(), counts_vec.data(), n_unique * sizeof(int64_t));
        }

        return {unique_out, inverse_out, counts_out};
    } else {
        throw std::runtime_error("unique: unsupported dtype");
    }
}

} // namespace cpu
} // namespace tenzor
