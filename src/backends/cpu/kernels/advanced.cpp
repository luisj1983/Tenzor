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
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "tenzor/backend/omp_thresholds.hpp"

#ifdef _OPENMP
    #include <omp.h>
#endif

#if defined(__x86_64__) || defined(_M_X64)
    #include <immintrin.h>
#endif

namespace tenzor {
namespace cpu {

namespace {

// ============================================================================
// SIMD-optimized CumSum for Float32 when inner_size >= SIMD width.
// Processes multiple independent prefix sums in parallel across the inner
// dimension using AVX2 (8 lanes) or AVX-512 (16 lanes).
// ============================================================================

#if defined(__AVX512F__)
static void cumsum_f32_avx512(const float* data, float* output, int64_t dim_size,
                               int64_t outer_size, int64_t inner_size) {
    constexpr int64_t VEC = 16;
    #pragma omp parallel for if(outer_size > 64)
    for (int64_t outer = 0; outer < outer_size; ++outer) {
        const float* base_in = data + outer * dim_size * inner_size;
        float* base_out = output + outer * dim_size * inner_size;

        // Vectorized prefix sum across inner dimension
        int64_t inner = 0;
        for (; inner + VEC <= inner_size; inner += VEC) {
            __m512 running = _mm512_setzero_ps();
            for (int64_t d = 0; d < dim_size; ++d) {
                int64_t off = d * inner_size + inner;
                __m512 val = _mm512_loadu_ps(base_in + off);
                running = _mm512_add_ps(running, val);
                _mm512_storeu_ps(base_out + off, running);
            }
        }
        // Scalar tail
        for (; inner < inner_size; ++inner) {
            float running = 0.0f;
            for (int64_t d = 0; d < dim_size; ++d) {
                int64_t idx = d * inner_size + inner;
                running += base_in[idx];
                base_out[idx] = running;
            }
        }
    }
}
#endif

#if defined(__AVX2__)
static void cumsum_f32_avx2(const float* data, float* output, int64_t dim_size,
                             int64_t outer_size, int64_t inner_size) {
    constexpr int64_t VEC = 8;
    #pragma omp parallel for if(outer_size > 64)
    for (int64_t outer = 0; outer < outer_size; ++outer) {
        const float* base_in = data + outer * dim_size * inner_size;
        float* base_out = output + outer * dim_size * inner_size;

        int64_t inner = 0;
        for (; inner + VEC <= inner_size; inner += VEC) {
            __m256 running = _mm256_setzero_ps();
            for (int64_t d = 0; d < dim_size; ++d) {
                int64_t off = d * inner_size + inner;
                __m256 val = _mm256_loadu_ps(base_in + off);
                running = _mm256_add_ps(running, val);
                _mm256_storeu_ps(base_out + off, running);
            }
        }
        for (; inner < inner_size; ++inner) {
            float running = 0.0f;
            for (int64_t d = 0; d < dim_size; ++d) {
                int64_t idx = d * inner_size + inner;
                running += base_in[idx];
                base_out[idx] = running;
            }
        }
    }
}
#endif

template<typename T>
auto topk_impl(const T* data, [[maybe_unused]] int64_t numel, int64_t dim_size,
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
    // Use SIMD-optimized paths for Float32 when inner_size is large enough
    if constexpr (std::is_same_v<T, float>) {
#if defined(__AVX512F__)
        if (inner_size >= 16) {
            cumsum_f32_avx512(data, output, dim_size, outer_size, inner_size);
            return;
        }
#endif
#if defined(__AVX2__)
        if (inner_size >= 8) {
            cumsum_f32_avx2(data, output, dim_size, outer_size, inner_size);
            return;
        }
#endif
    }

    // Scalar fallback for all types and small inner sizes
    #pragma omp parallel for if(outer_size * inner_size > ::tenzor::OmpThresholds::complex())
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
    // SIMD-optimized paths for Float32: vectorize across inner dimension
    if constexpr (std::is_same_v<T, float>) {
#if defined(__AVX512F__)
        if (inner_size >= 16) {
            constexpr int64_t VEC = 16;
            #pragma omp parallel for if(outer_size > 64)
            for (int64_t outer = 0; outer < outer_size; ++outer) {
                const float* base_in = data + outer * dim_size * inner_size;
                float* base_out = output + outer * dim_size * inner_size;
                int64_t inner = 0;
                for (; inner + VEC <= inner_size; inner += VEC) {
                    __m512 running = _mm512_set1_ps(1.0f);
                    for (int64_t d = 0; d < dim_size; ++d) {
                        int64_t off = d * inner_size + inner;
                        running = _mm512_mul_ps(running, _mm512_loadu_ps(base_in + off));
                        _mm512_storeu_ps(base_out + off, running);
                    }
                }
                for (; inner < inner_size; ++inner) {
                    float running = 1.0f;
                    for (int64_t d = 0; d < dim_size; ++d) {
                        int64_t idx = d * inner_size + inner;
                        running *= base_in[idx];
                        base_out[idx] = running;
                    }
                }
            }
            return;
        }
#endif
#if defined(__AVX2__)
        if (inner_size >= 8) {
            constexpr int64_t VEC = 8;
            #pragma omp parallel for if(outer_size > 64)
            for (int64_t outer = 0; outer < outer_size; ++outer) {
                const float* base_in = data + outer * dim_size * inner_size;
                float* base_out = output + outer * dim_size * inner_size;
                int64_t inner = 0;
                for (; inner + VEC <= inner_size; inner += VEC) {
                    __m256 running = _mm256_set1_ps(1.0f);
                    for (int64_t d = 0; d < dim_size; ++d) {
                        int64_t off = d * inner_size + inner;
                        running = _mm256_mul_ps(running, _mm256_loadu_ps(base_in + off));
                        _mm256_storeu_ps(base_out + off, running);
                    }
                }
                for (; inner < inner_size; ++inner) {
                    float running = 1.0f;
                    for (int64_t d = 0; d < dim_size; ++d) {
                        int64_t idx = d * inner_size + inner;
                        running *= base_in[idx];
                        base_out[idx] = running;
                    }
                }
            }
            return;
        }
#endif
    }

    // Scalar fallback
    #pragma omp parallel for if(outer_size * inner_size > ::tenzor::OmpThresholds::complex())
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
    if (dim < 0 || dim >= ndim) {
        throw std::out_of_range(
            "dim " + std::to_string(dim) + " is out of range for tensor with " +
            std::to_string(ndim) + " dimensions");
    }

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
    if (k < 0) {
        throw std::runtime_error("topk: k must be non-negative");
    }
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
        // Empty input: no unique values. The sorted_output branch below
        // unconditionally reads data[sort_idx[0]] with an empty sort_idx, so
        // guard here instead of indexing out of bounds.
        if (numel == 0) {
            return {Tensor({0}, input.dtype(), input.device()),
                    Tensor({0}, DType::Int64, input.device()),
                    Tensor({0}, DType::Int64, input.device())};
        }

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

// ============================================================================
// Bucketize Kernel - Binary search in sorted boundaries
// ============================================================================

auto bucketize_kernel(const Tensor& input, const Tensor& boundaries, bool right) -> Tensor {
    // input: any shape, boundaries: 1D sorted tensor
    // Returns: same shape as input, Int64 indices
    if (boundaries.ndim() != 1) {
        throw std::runtime_error("bucketize: boundaries must be 1D");
    }

    Tensor result(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                  DType::Int64, input.device());
    int64_t* out_data = result.data<int64_t>();

    // Templated binary search; searching in the native dtype avoids the
    // Float32 down-cast that collapses tightly-spaced Float64 values/boundaries
    // (PyTorch performs the search at the input precision).
    auto search = [&](const auto* in_data, const auto* b_data, int64_t n, int64_t nb) {
        #pragma omp parallel for schedule(static) if(n > ::tenzor::OmpThresholds::simple())
        for (int64_t i = 0; i < n; ++i) {
            auto val = in_data[i];
            int64_t lo = 0, hi = nb;
            while (lo < hi) {
                int64_t mid = lo + (hi - lo) / 2;
                bool go_right = right ? (b_data[mid] <= val) : (b_data[mid] < val);
                if (go_right) {
                    lo = mid + 1;
                } else {
                    hi = mid;
                }
            }
            out_data[i] = lo;
        }
    };

    if (input.dtype() == DType::Float64 || boundaries.dtype() == DType::Float64) {
        // Native Float64 path: widen both operands to double (lossless for
        // Float32/integer inputs) and search at double precision.
        Tensor input_f64 = (input.dtype() != DType::Float64) ? input.to(DType::Float64) : input;
        Tensor bound_f64 = (boundaries.dtype() != DType::Float64) ? boundaries.to(DType::Float64) : boundaries;
        search(input_f64.data<double>(), bound_f64.data<double>(),
               input_f64.numel(), bound_f64.numel());
    } else {
        // Float32/Float16/BFloat16/integer: search in Float32.
        Tensor input_f32 = (input.dtype() != DType::Float32) ? input.to(DType::Float32) : input;
        Tensor bound_f32 = (boundaries.dtype() != DType::Float32) ? boundaries.to(DType::Float32) : boundaries;
        search(input_f32.data<float>(), bound_f32.data<float>(),
               input_f32.numel(), bound_f32.numel());
    }

    return result;
}

// ============================================================================
// CumMax / CumMin Kernels
// ============================================================================

template<typename T>
auto cummax_impl(const T* data, T* out_values, int64_t* out_indices,
                 int64_t dim_size, int64_t outer_size, int64_t inner_size) -> void {
    #pragma omp parallel for if(outer_size * inner_size > ::tenzor::OmpThresholds::complex())
    for (int64_t outer = 0; outer < outer_size; ++outer) {
        for (int64_t inner = 0; inner < inner_size; ++inner) {
            int64_t first_idx = (outer * dim_size + 0) * inner_size + inner;
            T running_max = data[first_idx];
            int64_t running_idx = 0;
            out_values[first_idx] = running_max;
            out_indices[first_idx] = running_idx;
            for (int64_t d = 1; d < dim_size; ++d) {
                int64_t idx = (outer * dim_size + d) * inner_size + inner;
                T val = data[idx];
                if (val > running_max) {
                    running_max = val;
                    running_idx = d;
                }
                out_values[idx] = running_max;
                out_indices[idx] = running_idx;
            }
        }
    }
}

template<typename T>
auto cummin_impl(const T* data, T* out_values, int64_t* out_indices,
                 int64_t dim_size, int64_t outer_size, int64_t inner_size) -> void {
    #pragma omp parallel for if(outer_size * inner_size > ::tenzor::OmpThresholds::complex())
    for (int64_t outer = 0; outer < outer_size; ++outer) {
        for (int64_t inner = 0; inner < inner_size; ++inner) {
            int64_t first_idx = (outer * dim_size + 0) * inner_size + inner;
            T running_min = data[first_idx];
            int64_t running_idx = 0;
            out_values[first_idx] = running_min;
            out_indices[first_idx] = running_idx;
            for (int64_t d = 1; d < dim_size; ++d) {
                int64_t idx = (outer * dim_size + d) * inner_size + inner;
                T val = data[idx];
                if (val < running_min) {
                    running_min = val;
                    running_idx = d;
                }
                out_values[idx] = running_min;
                out_indices[idx] = running_idx;
            }
        }
    }
}

auto cummax_kernel(const Tensor& input, int64_t dim) -> std::pair<Tensor, Tensor> {
    if (dim < 0) dim += input.ndim();
    auto [outer_size, dim_size, inner_size] = decompose_dim(input, dim);

    Tensor cont = input.is_contiguous() ? input : input.contiguous();
    auto shape_vec = std::vector<int64_t>(input.shape().begin(), input.shape().end());
    Tensor values(shape_vec, input.dtype(), input.device());
    Tensor indices(shape_vec, DType::Int64, input.device());

    if (input.dtype() == DType::Float32) {
        cummax_impl(cont.data<float>(), values.data<float>(), indices.data<int64_t>(),
                    dim_size, outer_size, inner_size);
    } else if (input.dtype() == DType::Float64) {
        cummax_impl(cont.data<double>(), values.data<double>(), indices.data<int64_t>(),
                    dim_size, outer_size, inner_size);
    } else if (input.dtype() == DType::Int32) {
        cummax_impl(cont.data<int32_t>(), values.data<int32_t>(), indices.data<int64_t>(),
                    dim_size, outer_size, inner_size);
    } else if (input.dtype() == DType::Int64) {
        cummax_impl(cont.data<int64_t>(), values.data<int64_t>(), indices.data<int64_t>(),
                    dim_size, outer_size, inner_size);
    } else if (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16) {
        DType orig = input.dtype();
        Tensor cont_f32 = cont.to(DType::Float32);
        Tensor values_f32(shape_vec, DType::Float32, input.device());
        cummax_impl(cont_f32.data<float>(), values_f32.data<float>(), indices.data<int64_t>(),
                    dim_size, outer_size, inner_size);
        values = values_f32.to(orig);
    } else {
        throw std::runtime_error("cummax: unsupported dtype");
    }

    return {values, indices};
}

auto cummin_kernel(const Tensor& input, int64_t dim) -> std::pair<Tensor, Tensor> {
    if (dim < 0) dim += input.ndim();
    auto [outer_size, dim_size, inner_size] = decompose_dim(input, dim);

    Tensor cont = input.is_contiguous() ? input : input.contiguous();
    auto shape_vec = std::vector<int64_t>(input.shape().begin(), input.shape().end());
    Tensor values(shape_vec, input.dtype(), input.device());
    Tensor indices(shape_vec, DType::Int64, input.device());

    if (input.dtype() == DType::Float32) {
        cummin_impl(cont.data<float>(), values.data<float>(), indices.data<int64_t>(),
                    dim_size, outer_size, inner_size);
    } else if (input.dtype() == DType::Float64) {
        cummin_impl(cont.data<double>(), values.data<double>(), indices.data<int64_t>(),
                    dim_size, outer_size, inner_size);
    } else if (input.dtype() == DType::Int32) {
        cummin_impl(cont.data<int32_t>(), values.data<int32_t>(), indices.data<int64_t>(),
                    dim_size, outer_size, inner_size);
    } else if (input.dtype() == DType::Int64) {
        cummin_impl(cont.data<int64_t>(), values.data<int64_t>(), indices.data<int64_t>(),
                    dim_size, outer_size, inner_size);
    } else if (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16) {
        DType orig = input.dtype();
        Tensor cont_f32 = cont.to(DType::Float32);
        Tensor values_f32(shape_vec, DType::Float32, input.device());
        cummin_impl(cont_f32.data<float>(), values_f32.data<float>(), indices.data<int64_t>(),
                    dim_size, outer_size, inner_size);
        values = values_f32.to(orig);
    } else {
        throw std::runtime_error("cummin: unsupported dtype");
    }

    return {values, indices};
}

// ============================================================================
// Isin Kernel — set membership test
// ============================================================================

auto isin_kernel(const Tensor& elements, const Tensor& test_elements) -> Tensor {
    // Hash/compare at the native bit-width. The previous implementation widened
    // everything to Float32 and hashed 32 bits, which collides distinct Int64
    // values beyond 2^24 and loses Float64 precision (wrong membership).
    auto is_integral = [](DType d) {
        return d == DType::Int8 || d == DType::Int16 || d == DType::Int32 ||
               d == DType::Int64 || d == DType::UInt8 || d == DType::UInt16 ||
               d == DType::UInt32 || d == DType::UInt64 || d == DType::Bool;
    };
    const bool integral =
        is_integral(elements.dtype()) && is_integral(test_elements.dtype());

    auto out_shape = std::vector<int64_t>(elements.shape().begin(), elements.shape().end());
    Tensor output(out_shape, DType::Bool, elements.device());
    bool* out_data = output.data<bool>();
    const int64_t n_elem = elements.numel();

    if (integral) {
        // Exact integer comparison (covers all int widths and Bool).
        Tensor e = (elements.dtype() == DType::Int64) ? elements : elements.to(DType::Int64);
        Tensor t = (test_elements.dtype() == DType::Int64) ? test_elements : test_elements.to(DType::Int64);
        const int64_t* ed = e.data<int64_t>();
        const int64_t* td = t.data<int64_t>();
        const int64_t n_test = t.numel();

        std::unordered_set<int64_t> test_set;
        test_set.reserve(static_cast<size_t>(n_test));
        for (int64_t i = 0; i < n_test; ++i) test_set.insert(td[i]);

        #pragma omp parallel for schedule(static) if(n_elem > ::tenzor::OmpThresholds::simple())
        for (int64_t i = 0; i < n_elem; ++i) {
            out_data[i] = test_set.count(ed[i]) > 0;
        }
    } else {
        // Floating comparison via 64-bit bit pattern (exact for Float64/Float32/
        // Float16/BFloat16 since all widen losslessly to double).
        Tensor e = (elements.dtype() == DType::Float64) ? elements : elements.to(DType::Float64);
        Tensor t = (test_elements.dtype() == DType::Float64) ? test_elements : test_elements.to(DType::Float64);
        const double* ed = e.data<double>();
        const double* td = t.data<double>();
        const int64_t n_test = t.numel();

        std::unordered_set<uint64_t> test_set;
        test_set.reserve(static_cast<size_t>(n_test));
        for (int64_t i = 0; i < n_test; ++i) {
            uint64_t bits;
            std::memcpy(&bits, &td[i], sizeof(uint64_t));
            test_set.insert(bits);
        }

        #pragma omp parallel for schedule(static) if(n_elem > ::tenzor::OmpThresholds::simple())
        for (int64_t i = 0; i < n_elem; ++i) {
            uint64_t bits;
            std::memcpy(&bits, &ed[i], sizeof(uint64_t));
            out_data[i] = test_set.count(bits) > 0;
        }
    }

    return output;
}

// ============================================================================
// Kthvalue Kernel — k-th smallest value along dim
// ============================================================================

template<typename T>
auto kthvalue_impl(const T* data, int64_t dim_size,
                   int64_t outer_size, int64_t inner_size, int64_t k,
                   T* out_values, int64_t* out_indices) -> void {
    #pragma omp parallel for if(outer_size * inner_size > 1024)
    for (int64_t outer = 0; outer < outer_size; ++outer) {
        for (int64_t inner = 0; inner < inner_size; ++inner) {
            std::vector<std::pair<T, int64_t>> pairs(dim_size);
            for (int64_t d = 0; d < dim_size; ++d) {
                int64_t idx = (outer * dim_size + d) * inner_size + inner;
                pairs[d] = {data[idx], d};
            }
            // Partial sort so that pairs[k-1] is the k-th smallest
            std::nth_element(pairs.begin(), pairs.begin() + (k - 1), pairs.end(),
                [](const auto& a, const auto& b) { return a.first < b.first; });

            int64_t out_idx = outer * inner_size + inner;
            out_values[out_idx] = pairs[k - 1].first;
            out_indices[out_idx] = pairs[k - 1].second;
        }
    }
}

auto kthvalue_kernel(const Tensor& input, int64_t k, int64_t dim,
                     bool keepdim) -> std::pair<Tensor, Tensor> {
    if (dim < 0) dim += input.ndim();
    auto [outer_size, dim_size, inner_size] = decompose_dim(input, dim);

    if (k < 1 || k > dim_size) {
        throw std::runtime_error("kthvalue: k out of range");
    }

    Tensor cont = input.is_contiguous() ? input : input.contiguous();

    // Output shape: same as input but dim is reduced to 1
    std::vector<int64_t> out_shape(input.shape().begin(), input.shape().end());
    out_shape[dim] = 1;

    Tensor values(out_shape, input.dtype(), input.device());
    Tensor indices(out_shape, DType::Int64, input.device());

    if (input.dtype() == DType::Float32) {
        kthvalue_impl(cont.data<float>(), dim_size, outer_size, inner_size, k,
                      values.data<float>(), indices.data<int64_t>());
    } else if (input.dtype() == DType::Float64) {
        kthvalue_impl(cont.data<double>(), dim_size, outer_size, inner_size, k,
                      values.data<double>(), indices.data<int64_t>());
    } else if (input.dtype() == DType::Int32) {
        kthvalue_impl(cont.data<int32_t>(), dim_size, outer_size, inner_size, k,
                      values.data<int32_t>(), indices.data<int64_t>());
    } else if (input.dtype() == DType::Int64) {
        kthvalue_impl(cont.data<int64_t>(), dim_size, outer_size, inner_size, k,
                      values.data<int64_t>(), indices.data<int64_t>());
    } else if (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16) {
        DType orig = input.dtype();
        Tensor cont_f32 = cont.to(DType::Float32);
        Tensor values_f32(out_shape, DType::Float32, input.device());
        kthvalue_impl(cont_f32.data<float>(), dim_size, outer_size, inner_size, k,
                      values_f32.data<float>(), indices.data<int64_t>());
        values = values_f32.to(orig);
    } else {
        throw std::runtime_error("kthvalue: unsupported dtype");
    }

    if (!keepdim) {
        // Squeeze the dim dimension
        std::vector<int64_t> squeezed;
        for (int64_t i = 0; i < static_cast<int64_t>(out_shape.size()); ++i) {
            if (i != dim) squeezed.push_back(out_shape[i]);
        }
        if (squeezed.empty()) squeezed.push_back(1);
        values = values.reshape(squeezed);
        indices = indices.reshape(squeezed);
    }

    return {values, indices};
}

// ============================================================================
// Fmax / Fmin Kernels — element-wise max/min ignoring NaN
// ============================================================================

template<typename T>
auto fmax_impl(const T* a, const T* b, T* out, int64_t n) -> void {
    #pragma omp parallel for schedule(static) if(n > ::tenzor::OmpThresholds::simple())
    for (int64_t i = 0; i < n; ++i) {
        if constexpr (std::is_floating_point_v<T>) {
            if (std::isnan(a[i])) { out[i] = b[i]; }
            else if (std::isnan(b[i])) { out[i] = a[i]; }
            else { out[i] = (a[i] >= b[i]) ? a[i] : b[i]; }
        } else {
            out[i] = (a[i] >= b[i]) ? a[i] : b[i];
        }
    }
}

template<typename T>
auto fmin_impl(const T* a, const T* b, T* out, int64_t n) -> void {
    #pragma omp parallel for schedule(static) if(n > ::tenzor::OmpThresholds::simple())
    for (int64_t i = 0; i < n; ++i) {
        if constexpr (std::is_floating_point_v<T>) {
            if (std::isnan(a[i])) { out[i] = b[i]; }
            else if (std::isnan(b[i])) { out[i] = a[i]; }
            else { out[i] = (a[i] <= b[i]) ? a[i] : b[i]; }
        } else {
            out[i] = (a[i] <= b[i]) ? a[i] : b[i];
        }
    }
}

auto fmax_kernel(const Tensor& a, const Tensor& b) -> Tensor {
    if (a.numel() != b.numel()) {
        throw std::runtime_error("fmax: tensors must have the same number of elements");
    }
    Tensor ca = a.is_contiguous() ? a : a.contiguous();
    Tensor cb = b.is_contiguous() ? b : b.contiguous();
    auto shape_vec = std::vector<int64_t>(a.shape().begin(), a.shape().end());
    Tensor output(shape_vec, a.dtype(), a.device());
    int64_t n = a.numel();

    if (a.dtype() == DType::Float32) {
        fmax_impl(ca.data<float>(), cb.data<float>(), output.data<float>(), n);
    } else if (a.dtype() == DType::Float64) {
        fmax_impl(ca.data<double>(), cb.data<double>(), output.data<double>(), n);
    } else if (a.dtype() == DType::Int32) {
        fmax_impl(ca.data<int32_t>(), cb.data<int32_t>(), output.data<int32_t>(), n);
    } else if (a.dtype() == DType::Int64) {
        fmax_impl(ca.data<int64_t>(), cb.data<int64_t>(), output.data<int64_t>(), n);
    } else if (a.dtype() == DType::Float16 || a.dtype() == DType::BFloat16) {
        DType orig = a.dtype();
        Tensor af32 = ca.to(DType::Float32);
        Tensor bf32 = cb.to(DType::Float32);
        Tensor out_f32(shape_vec, DType::Float32, a.device());
        fmax_impl(af32.data<float>(), bf32.data<float>(), out_f32.data<float>(), n);
        output = out_f32.to(orig);
    } else {
        throw std::runtime_error("fmax: unsupported dtype");
    }
    return output;
}

auto fmin_kernel(const Tensor& a, const Tensor& b) -> Tensor {
    if (a.numel() != b.numel()) {
        throw std::runtime_error("fmin: tensors must have the same number of elements");
    }
    Tensor ca = a.is_contiguous() ? a : a.contiguous();
    Tensor cb = b.is_contiguous() ? b : b.contiguous();
    auto shape_vec = std::vector<int64_t>(a.shape().begin(), a.shape().end());
    Tensor output(shape_vec, a.dtype(), a.device());
    int64_t n = a.numel();

    if (a.dtype() == DType::Float32) {
        fmin_impl(ca.data<float>(), cb.data<float>(), output.data<float>(), n);
    } else if (a.dtype() == DType::Float64) {
        fmin_impl(ca.data<double>(), cb.data<double>(), output.data<double>(), n);
    } else if (a.dtype() == DType::Int32) {
        fmin_impl(ca.data<int32_t>(), cb.data<int32_t>(), output.data<int32_t>(), n);
    } else if (a.dtype() == DType::Int64) {
        fmin_impl(ca.data<int64_t>(), cb.data<int64_t>(), output.data<int64_t>(), n);
    } else if (a.dtype() == DType::Float16 || a.dtype() == DType::BFloat16) {
        DType orig = a.dtype();
        Tensor af32 = ca.to(DType::Float32);
        Tensor bf32 = cb.to(DType::Float32);
        Tensor out_f32(shape_vec, DType::Float32, a.device());
        fmin_impl(af32.data<float>(), bf32.data<float>(), out_f32.data<float>(), n);
        output = out_f32.to(orig);
    } else {
        throw std::runtime_error("fmin: unsupported dtype");
    }
    return output;
}

// ============================================================================
// Quantile / Nanquantile Kernels
// ============================================================================

template<typename T>
auto quantile_impl(const T* data, int64_t dim_size,
                   int64_t outer_size, int64_t inner_size,
                   double q, T* out_values) -> void {
    #pragma omp parallel for if(outer_size * inner_size > 1024)
    for (int64_t outer = 0; outer < outer_size; ++outer) {
        for (int64_t inner = 0; inner < inner_size; ++inner) {
            // A zero-length reduction axis has no values to interpolate; emit NaN
            // instead of indexing the empty slice (negative/OOB access).
            if (dim_size == 0) {
                out_values[outer * inner_size + inner] =
                    std::numeric_limits<T>::quiet_NaN();
                continue;
            }
            std::vector<T> slice(dim_size);
            for (int64_t d = 0; d < dim_size; ++d) {
                slice[d] = data[(outer * dim_size + d) * inner_size + inner];
            }
            std::sort(slice.begin(), slice.end());

            // Linear interpolation at the q-th position
            double idx_f = q * static_cast<double>(dim_size - 1);
            int64_t lo = static_cast<int64_t>(idx_f);
            if (lo < 0) lo = 0;
            if (lo >= dim_size) lo = dim_size - 1;
            int64_t hi = lo + 1;
            if (hi >= dim_size) hi = dim_size - 1;
            double frac_part = idx_f - static_cast<double>(lo);

            T result = static_cast<T>(
                static_cast<double>(slice[lo]) * (1.0 - frac_part) +
                static_cast<double>(slice[hi]) * frac_part);

            out_values[outer * inner_size + inner] = result;
        }
    }
}

template<typename T>
auto nanquantile_impl(const T* data, int64_t dim_size,
                      int64_t outer_size, int64_t inner_size,
                      double q, T* out_values) -> void {
    #pragma omp parallel for if(outer_size * inner_size > 1024)
    for (int64_t outer = 0; outer < outer_size; ++outer) {
        for (int64_t inner = 0; inner < inner_size; ++inner) {
            std::vector<T> slice;
            slice.reserve(dim_size);
            for (int64_t d = 0; d < dim_size; ++d) {
                T val = data[(outer * dim_size + d) * inner_size + inner];
                if constexpr (std::is_floating_point_v<T>) {
                    if (!std::isnan(val)) slice.push_back(val);
                } else {
                    slice.push_back(val);
                }
            }

            if (slice.empty()) {
                out_values[outer * inner_size + inner] = std::numeric_limits<T>::quiet_NaN();
                continue;
            }

            std::sort(slice.begin(), slice.end());
            int64_t n = static_cast<int64_t>(slice.size());

            double idx_f = q * static_cast<double>(n - 1);
            int64_t lo = static_cast<int64_t>(idx_f);
            if (lo < 0) lo = 0;
            if (lo >= n) lo = n - 1;
            int64_t hi = lo + 1;
            if (hi >= n) hi = n - 1;
            double frac_part = idx_f - static_cast<double>(lo);

            T result = static_cast<T>(
                static_cast<double>(slice[lo]) * (1.0 - frac_part) +
                static_cast<double>(slice[hi]) * frac_part);

            out_values[outer * inner_size + inner] = result;
        }
    }
}

auto quantile_kernel(const Tensor& input, double q, int64_t dim,
                     bool keepdim) -> Tensor {
    if (dim < 0) dim += input.ndim();
    auto [outer_size, dim_size, inner_size] = decompose_dim(input, dim);

    Tensor cont = input.is_contiguous() ? input : input.contiguous();

    std::vector<int64_t> out_shape(input.shape().begin(), input.shape().end());
    out_shape[dim] = 1;

    Tensor output(out_shape, input.dtype(), input.device());

    if (input.dtype() == DType::Float32) {
        quantile_impl(cont.data<float>(), dim_size, outer_size, inner_size, q,
                      output.data<float>());
    } else if (input.dtype() == DType::Float64) {
        quantile_impl(cont.data<double>(), dim_size, outer_size, inner_size, q,
                      output.data<double>());
    } else if (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16) {
        DType orig = input.dtype();
        Tensor cont_f32 = cont.to(DType::Float32);
        Tensor out_f32(out_shape, DType::Float32, input.device());
        quantile_impl(cont_f32.data<float>(), dim_size, outer_size, inner_size, q,
                      out_f32.data<float>());
        output = out_f32.to(orig);
    } else {
        throw std::runtime_error("quantile: unsupported dtype (requires floating-point)");
    }

    if (!keepdim) {
        std::vector<int64_t> squeezed;
        for (int64_t i = 0; i < static_cast<int64_t>(out_shape.size()); ++i) {
            if (i != dim) squeezed.push_back(out_shape[i]);
        }
        if (squeezed.empty()) squeezed.push_back(1);
        output = output.reshape(squeezed);
    }

    return output;
}

auto nanquantile_kernel(const Tensor& input, double q, int64_t dim,
                        bool keepdim) -> Tensor {
    if (dim < 0) dim += input.ndim();
    auto [outer_size, dim_size, inner_size] = decompose_dim(input, dim);

    Tensor cont = input.is_contiguous() ? input : input.contiguous();

    std::vector<int64_t> out_shape(input.shape().begin(), input.shape().end());
    out_shape[dim] = 1;

    Tensor output(out_shape, input.dtype(), input.device());

    if (input.dtype() == DType::Float32) {
        nanquantile_impl(cont.data<float>(), dim_size, outer_size, inner_size, q,
                         output.data<float>());
    } else if (input.dtype() == DType::Float64) {
        nanquantile_impl(cont.data<double>(), dim_size, outer_size, inner_size, q,
                         output.data<double>());
    } else if (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16) {
        DType orig = input.dtype();
        Tensor cont_f32 = cont.to(DType::Float32);
        Tensor out_f32(out_shape, DType::Float32, input.device());
        nanquantile_impl(cont_f32.data<float>(), dim_size, outer_size, inner_size, q,
                         out_f32.data<float>());
        output = out_f32.to(orig);
    } else {
        throw std::runtime_error("nanquantile: unsupported dtype (requires floating-point)");
    }

    if (!keepdim) {
        std::vector<int64_t> squeezed;
        for (int64_t i = 0; i < static_cast<int64_t>(out_shape.size()); ++i) {
            if (i != dim) squeezed.push_back(out_shape[i]);
        }
        if (squeezed.empty()) squeezed.push_back(1);
        output = output.reshape(squeezed);
    }

    return output;
}

// ============================================================================
// Nanmedian Kernel — NaN-ignoring median
// ============================================================================

template<typename T>
auto nanmedian_impl(const T* data, int64_t dim_size,
                    int64_t outer_size, int64_t inner_size,
                    T* out_values) -> void {
    #pragma omp parallel for if(outer_size * inner_size > 1024)
    for (int64_t outer = 0; outer < outer_size; ++outer) {
        for (int64_t inner = 0; inner < inner_size; ++inner) {
            std::vector<T> slice;
            slice.reserve(dim_size);
            for (int64_t d = 0; d < dim_size; ++d) {
                T val = data[(outer * dim_size + d) * inner_size + inner];
                if constexpr (std::is_floating_point_v<T>) {
                    if (!std::isnan(val)) slice.push_back(val);
                } else {
                    slice.push_back(val);
                }
            }

            if (slice.empty()) {
                out_values[outer * inner_size + inner] = std::numeric_limits<T>::quiet_NaN();
                continue;
            }

            std::sort(slice.begin(), slice.end());
            int64_t n = static_cast<int64_t>(slice.size());
            // PyTorch nanmedian returns the lower median (no interpolation)
            out_values[outer * inner_size + inner] = slice[n / 2];
        }
    }
}

auto nanmedian_kernel(const Tensor& input, int64_t dim) -> Tensor {
    if (dim < 0) dim += input.ndim();
    auto [outer_size, dim_size, inner_size] = decompose_dim(input, dim);

    Tensor cont = input.is_contiguous() ? input : input.contiguous();

    std::vector<int64_t> out_shape(input.shape().begin(), input.shape().end());
    out_shape[dim] = 1;

    Tensor output(out_shape, input.dtype(), input.device());

    if (input.dtype() == DType::Float32) {
        nanmedian_impl(cont.data<float>(), dim_size, outer_size, inner_size,
                       output.data<float>());
    } else if (input.dtype() == DType::Float64) {
        nanmedian_impl(cont.data<double>(), dim_size, outer_size, inner_size,
                       output.data<double>());
    } else if (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16) {
        DType orig = input.dtype();
        Tensor cont_f32 = cont.to(DType::Float32);
        Tensor out_f32(out_shape, DType::Float32, input.device());
        nanmedian_impl(cont_f32.data<float>(), dim_size, outer_size, inner_size,
                       out_f32.data<float>());
        output = out_f32.to(orig);
    } else {
        throw std::runtime_error("nanmedian: unsupported dtype (requires floating-point)");
    }

    // Squeeze the dim dimension
    std::vector<int64_t> squeezed;
    for (int64_t i = 0; i < static_cast<int64_t>(out_shape.size()); ++i) {
        if (i != dim) squeezed.push_back(out_shape[i]);
    }
    if (squeezed.empty()) squeezed.push_back(1);
    output = output.reshape(squeezed);

    return output;
}

// ============================================================================
// Histc Kernel — fixed-bin histogram
// ============================================================================

auto histc_kernel(const Tensor& input, int64_t bins, double min_val, double max_val) -> Tensor {
    if (bins <= 0) {
        throw std::runtime_error("histc: bins must be positive");
    }
    Tensor cont = input.is_contiguous() ? input : input.contiguous();

    // PyTorch's torch.histc returns a tensor of the input dtype and bins at the
    // input precision. Use a native Float64 path for Float64 input so values
    // near bin edges or beyond float32 precision land in the correct bin and the
    // returned dtype matches; keep the Float32 path for Float32/half inputs.
    auto histogram = [&]<typename T>(const T* data, int64_t n, T* out_data) {
        // Auto-detect min/max if both are zero (PyTorch convention)
        T lo = static_cast<T>(min_val);
        T hi = static_cast<T>(max_val);
        if (lo == T(0) && hi == T(0)) {
            lo = std::numeric_limits<T>::max();
            hi = std::numeric_limits<T>::lowest();
            for (int64_t i = 0; i < n; ++i) {
                if (data[i] < lo) lo = data[i];
                if (data[i] > hi) hi = data[i];
            }
        }

        if (lo >= hi) {
            // When min == max, all elements go into the first bin
            hi = lo + T(1);
        }

        T bin_width = (hi - lo) / static_cast<T>(bins);

        // Accumulate counts in int64 so the bin totals stay integer-exact: a
        // running ``out_data[bin] += T(1)`` in Float32 stops incrementing past
        // 2^24 elements per bin and would silently undercount large inputs.
        std::vector<int64_t> counts(static_cast<size_t>(bins), 0);
        for (int64_t i = 0; i < n; ++i) {
            T val = data[i];
            if (val < lo || val > hi) continue;
            int64_t bin = static_cast<int64_t>((val - lo) / bin_width);
            if (bin >= bins) bin = bins - 1;  // clamp right edge
            ++counts[static_cast<size_t>(bin)];
        }
        for (int64_t b = 0; b < bins; ++b) {
            out_data[b] = static_cast<T>(counts[static_cast<size_t>(b)]);
        }
    };

    if (cont.dtype() == DType::Float64) {
        Tensor output({bins}, DType::Float64, input.device());
        histogram.template operator()<double>(cont.data<double>(), cont.numel(),
                                               output.data<double>());
        return output;
    }

    Tensor input_f32 = (cont.dtype() != DType::Float32) ? cont.to(DType::Float32) : cont;
    Tensor output({bins}, DType::Float32, input.device());
    histogram.template operator()<float>(input_f32.data<float>(), input_f32.numel(),
                                         output.data<float>());
    return output;
}

// ============================================================================
// UniqueConsecutive Kernel
// ============================================================================

template<typename T>
auto unique_consecutive_impl(const T* data, int64_t numel,
                             bool return_inverse, bool return_counts)
    -> std::tuple<std::vector<T>, std::vector<int64_t>, std::vector<int64_t>> {

    std::vector<T> unique_vals;
    std::vector<int64_t> inverse_map;
    std::vector<int64_t> counts_vec;

    if (numel == 0) {
        return {unique_vals, inverse_map, counts_vec};
    }

    unique_vals.push_back(data[0]);
    if (return_inverse) inverse_map.resize(numel);
    if (return_inverse) inverse_map[0] = 0;
    int64_t cur_count = 1;

    for (int64_t i = 1; i < numel; ++i) {
        if (data[i] != data[i - 1]) {
            if (return_counts) counts_vec.push_back(cur_count);
            cur_count = 1;
            unique_vals.push_back(data[i]);
        } else {
            ++cur_count;
        }
        if (return_inverse) {
            inverse_map[i] = static_cast<int64_t>(unique_vals.size()) - 1;
        }
    }
    if (return_counts) counts_vec.push_back(cur_count);

    return {unique_vals, inverse_map, counts_vec};
}

auto unique_consecutive_kernel(const Tensor& input, bool return_inverse,
                               bool return_counts)
    -> std::tuple<Tensor, Tensor, Tensor> {

    Tensor cont = input.is_contiguous() ? input : input.contiguous();
    int64_t numel = cont.numel();

    auto do_unique = [&]<typename T>(const T* data) -> std::tuple<Tensor, Tensor, Tensor> {
        auto [unique_vals, inverse_map, counts_vec] =
            unique_consecutive_impl(data, numel, return_inverse, return_counts);

        int64_t n_unique = static_cast<int64_t>(unique_vals.size());

        Tensor unique_out({n_unique}, input.dtype(), input.device());
        T* udata = unique_out.template data<T>();
        std::memcpy(udata, unique_vals.data(), n_unique * sizeof(T));

        Tensor inverse_out;
        if (return_inverse && numel > 0) {
            inverse_out = Tensor({numel}, DType::Int64, input.device());
            std::memcpy(inverse_out.data<int64_t>(), inverse_map.data(), numel * sizeof(int64_t));
        } else {
            inverse_out = Tensor({0}, DType::Int64, input.device());
        }

        Tensor counts_out;
        if (return_counts && n_unique > 0) {
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
    } else if (input.dtype() == DType::Int32) {
        return do_unique(cont.data<int32_t>());
    } else if (input.dtype() == DType::Int64) {
        return do_unique(cont.data<int64_t>());
    } else if (input.dtype() == DType::Int8) {
        return do_unique(cont.data<int8_t>());
    } else if (input.dtype() == DType::UInt8) {
        return do_unique(cont.data<uint8_t>());
    } else if (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16) {
        DType orig = input.dtype();
        Tensor cont_f32 = cont.to(DType::Float32);
        auto [unique_f32, inverse_out, counts_out] = do_unique(cont_f32.data<float>());
        return {unique_f32.to(orig), inverse_out, counts_out};
    } else {
        throw std::runtime_error("unique_consecutive: unsupported dtype");
    }
}

// ============================================================================
// SegmentReduce — reduce over segments defined by offsets
// ============================================================================

template<typename T>
static auto segment_reduce_impl(const T* data, const int64_t* offsets,
                                 int64_t num_segments, int64_t outer_size,
                                 int64_t axis_size, int64_t inner_size,
                                 const std::string& reduce, DType dtype, Device device,
                                 const std::vector<int64_t>& out_shape) -> Tensor {
    // Output layout is logically [outer, num_segments, inner]; allocate with the
    // caller's full N-D shape so no post-kernel reshape is needed.
    Tensor output(out_shape, dtype, device);
    T* out_ptr = output.data<T>();
    const int64_t out_numel = output.numel();

    // Resolve the reduce mode to an enum ONCE up front, both to validate it (an
    // unknown mode is a hard error, not a silent identity-fill) and to keep the
    // hot inner loop free of repeated std::string comparisons.
    enum class Mode { Sum, Mean, Prod, Max, Min };
    Mode mode;
    if (reduce == "sum")        mode = Mode::Sum;
    else if (reduce == "mean")  mode = Mode::Mean;
    else if (reduce == "prod")  mode = Mode::Prod;
    else if (reduce == "max")   mode = Mode::Max;
    else if (reduce == "min")   mode = Mode::Min;
    else throw std::invalid_argument("segment_reduce: unsupported reduce mode '" + reduce + "'");

    // Determine identity values for each reduce mode
    const T identity = [&]() -> T {
        switch (mode) {
            case Mode::Sum:
            case Mode::Mean: return T(0);
            case Mode::Prod: return T(1);
            case Mode::Max:  return std::numeric_limits<T>::lowest();
            case Mode::Min:  return std::numeric_limits<T>::max();
        }
        return T(0);
    }();

    // Initialize output with identity
    for (int64_t i = 0; i < out_numel; ++i) {
        out_ptr[i] = identity;
    }

    #pragma omp parallel for collapse(2) if(outer_size * num_segments > 64)
    for (int64_t outer = 0; outer < outer_size; ++outer) {
        for (int64_t seg = 0; seg < num_segments; ++seg) {
            int64_t seg_start = offsets[seg];
            int64_t seg_end = offsets[seg + 1];
            int64_t seg_len = seg_end - seg_start;
            if (seg_len <= 0) continue;

            for (int64_t inner = 0; inner < inner_size; ++inner) {
                int64_t out_idx = (outer * num_segments + seg) * inner_size + inner;
                T acc = identity;

                for (int64_t d = seg_start; d < seg_end; ++d) {
                    int64_t in_idx = (outer * axis_size + d) * inner_size + inner;
                    T val = data[in_idx];
                    switch (mode) {
                        case Mode::Sum:
                        case Mode::Mean: acc += val; break;
                        case Mode::Prod: acc *= val; break;
                        case Mode::Max:  acc = acc > val ? acc : val; break;
                        case Mode::Min:  acc = acc < val ? acc : val; break;
                    }
                }

                if (mode == Mode::Mean && seg_len > 0) {
                    acc /= static_cast<T>(seg_len);
                }

                out_ptr[out_idx] = acc;
            }
        }
    }

    return output;
}

auto segment_reduce_kernel(const Tensor& data, const Tensor& offsets,
                           const std::string& reduce, int64_t axis) -> Tensor {
    auto cont = data.is_contiguous() ? data : data.contiguous();
    auto offs = offsets.is_contiguous() ? offsets : offsets.contiguous();

    if (reduce != "sum" && reduce != "mean" && reduce != "prod" &&
        reduce != "max" && reduce != "min") {
        throw std::invalid_argument(
            "segment_reduce: unsupported reduce mode '" + reduce +
            "' (expected one of sum, mean, prod, max, min)");
    }

    int64_t ndim = cont.ndim();
    if (axis < 0) axis += ndim;
    if (axis < 0 || axis >= ndim) {
        throw std::runtime_error("segment_reduce: axis out of range");
    }

    const auto& shape = cont.shape();
    int64_t axis_size = shape[axis];
    int64_t num_segments = offs.numel() - 1;

    // Compute outer and inner sizes
    int64_t outer_size = 1;
    for (int64_t i = 0; i < axis; ++i) outer_size *= shape[i];
    int64_t inner_size = 1;
    for (int64_t i = axis + 1; i < ndim; ++i) inner_size *= shape[i];

    // Build the final N-D output shape up front: identical to input except the
    // reduction axis is replaced by num_segments.
    std::vector<int64_t> out_shape;
    out_shape.reserve(ndim);
    for (int64_t i = 0; i < ndim; ++i) {
        out_shape.push_back(i == axis ? num_segments : shape[i]);
    }

    const int64_t* offsets_ptr = offs.data<int64_t>();
    // Validate the (untrusted) offsets so segment_reduce_impl cannot index the
    // input out of bounds: in-range endpoints and non-decreasing.
    if (num_segments < 0) {
        throw std::invalid_argument("segment_reduce: offsets must have at least 1 element");
    }
    if (num_segments > 0) {
        if (offsets_ptr[0] < 0 || offsets_ptr[num_segments] > axis_size) {
            throw std::invalid_argument("segment_reduce: offsets out of range [0, axis_size]");
        }
        for (int64_t i = 0; i < num_segments; ++i) {
            if (offsets_ptr[i] > offsets_ptr[i + 1]) {
                throw std::invalid_argument("segment_reduce: offsets must be non-decreasing");
            }
        }
    }
    auto dtype = cont.dtype();
    auto device = cont.device();

    if (dtype == DType::Float32) {
        return segment_reduce_impl(cont.data<float>(), offsets_ptr, num_segments,
                                   outer_size, axis_size, inner_size, reduce, dtype, device, out_shape);
    } else if (dtype == DType::Float64) {
        return segment_reduce_impl(cont.data<double>(), offsets_ptr, num_segments,
                                   outer_size, axis_size, inner_size, reduce, dtype, device, out_shape);
    } else if (dtype == DType::Int32) {
        return segment_reduce_impl(cont.data<int32_t>(), offsets_ptr, num_segments,
                                   outer_size, axis_size, inner_size, reduce, dtype, device, out_shape);
    } else if (dtype == DType::Int64) {
        return segment_reduce_impl(cont.data<int64_t>(), offsets_ptr, num_segments,
                                   outer_size, axis_size, inner_size, reduce, dtype, device, out_shape);
    } else if (dtype == DType::Float16 || dtype == DType::BFloat16) {
        DType orig = dtype;
        Tensor cont_f32 = cont.to(DType::Float32);
        auto res_f32 = segment_reduce_impl(cont_f32.data<float>(), offsets_ptr, num_segments,
                                           outer_size, axis_size, inner_size, reduce,
                                           DType::Float32, device, out_shape);
        return res_f32.to(orig);
    }
    throw std::runtime_error("segment_reduce: unsupported dtype");
}

} // namespace cpu
} // namespace tenzor
