/**
 * @file indexing.cpp
 * @brief CPU indexing operation kernels
 */

#include "tenzor/core/tensor.hpp"
#include "tenzor/core/dtype.hpp"
#include "tenzor/core/shape.hpp"
#include "tenzor/backend/backend.hpp"
#include "tenzor/backend/op_attributes.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/transform.hpp"  // broadcast_to (C.6 masked_fill)
#include "simd_fast_math.hpp"
#include <complex>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <span>
#ifdef _OPENMP
#include <omp.h>
#endif

// SIMD support detection
#if defined(__AVX512F__)
    #include <immintrin.h>
    #define INDEXING_HAS_AVX512 1
#elif defined(__AVX2__)
    #include <immintrin.h>
    #define INDEXING_HAS_AVX2 1
#endif

#ifdef _OPENMP
    #include <omp.h>
#endif

namespace tenzor {
namespace cpu {

namespace detail {

// Template helper for index_select inner loop (pure copy, no arithmetic)
template<typename T>
void index_select_impl(const T* in_data, T* out_data,
                       const int64_t* index_data, int64_t num_indices,
                       int64_t outer_size, int64_t inner_size,
                       std::span<const int64_t> in_shape,
                       std::span<const int64_t> in_strides,
                       std::span<const int64_t> out_strides,
                       int64_t dim) {
    // Validate indices first (can't throw in OpenMP region)
    for (int64_t idx = 0; idx < num_indices; ++idx) {
        int64_t src_idx = index_data[idx];
        if (src_idx < 0) src_idx += in_shape[dim];
        if (src_idx < 0 || src_idx >= in_shape[dim]) {
            throw std::out_of_range("index_select: index out of range");
        }
    }

    #pragma omp parallel for collapse(2) if(outer_size * num_indices > 65536)
    for (int64_t outer = 0; outer < outer_size; ++outer) {
        for (int64_t idx = 0; idx < num_indices; ++idx) {
            int64_t src_idx = index_data[idx];
            if (src_idx < 0) src_idx += in_shape[dim];
            for (int64_t inner = 0; inner < inner_size; ++inner) {
                int64_t in_offset = outer * in_strides[dim] * in_shape[dim] +
                                   src_idx * in_strides[dim] + inner;
                int64_t out_offset = outer * out_strides[dim] * num_indices +
                                    idx * out_strides[dim] + inner;
                out_data[out_offset] = in_data[in_offset];
            }
        }
    }
}

// Template helper for gather inner loop
template<typename T>
void gather_impl(const T* input_ptr, T* output_ptr,
                 const int64_t* index_ptr, int64_t numel,
                 const std::vector<int64_t>& input_shape,
                 const std::vector<int64_t>& input_strides,
                 const std::vector<int64_t>& index_strides,
                 size_t ndims, int64_t dim) {
    // Validate gather-dim indices sequentially (throwing from inside OMP parallel is UB)
    {
        const int64_t dim_size = input_shape[dim];
        for (int64_t i = 0; i < numel; ++i) {
            int64_t idx_val = index_ptr[i];
            if (idx_val < 0) idx_val += dim_size;
            if (idx_val < 0 || idx_val >= dim_size) {
                throw std::out_of_range("gather: index " + std::to_string(index_ptr[i]) +
                    " out of range for dimension " + std::to_string(dim) +
                    " with size " + std::to_string(dim_size));
            }
        }
    }

    // Gather in parallel (indices already validated)
    #pragma omp parallel for if(numel > 65536)
    for (int64_t flat_idx = 0; flat_idx < numel; ++flat_idx) {
        int64_t temp = flat_idx;
        int64_t input_idx = 0;
        for (size_t d = 0; d < ndims; ++d) {
            int64_t coord = temp / index_strides[d];
            temp %= index_strides[d];
            if (static_cast<int64_t>(d) == dim) {
                int64_t idx_val = index_ptr[flat_idx];
                if (idx_val < 0) idx_val += input_shape[d];
                input_idx += idx_val * input_strides[d];
            } else {
                input_idx += coord * input_strides[d];
            }
        }
        output_ptr[flat_idx] = input_ptr[input_idx];
    }
}

// Template helper for scatter inner loop
template<typename T>
void scatter_impl(const T* input_ptr, T* output_ptr, const T* src_ptr,
                  const int64_t* index_ptr, int64_t input_numel, int64_t index_numel,
                  const std::vector<int64_t>& input_shape,
                  const std::vector<int64_t>& input_strides,
                  const std::vector<int64_t>& index_strides,
                  size_t ndims, int64_t dim) {
    std::memcpy(output_ptr, input_ptr, input_numel * sizeof(T));
    for (int64_t flat_idx = 0; flat_idx < index_numel; ++flat_idx) {
        int64_t temp = flat_idx;
        int64_t output_idx = 0;
        for (size_t d = 0; d < ndims; ++d) {
            int64_t coord = temp / index_strides[d];
            temp %= index_strides[d];
            if (static_cast<int64_t>(d) == dim) {
                int64_t idx_val = index_ptr[flat_idx];
                if (idx_val < 0) idx_val += input_shape[d];
                if (idx_val < 0 || idx_val >= input_shape[d]) {
                    throw std::out_of_range("scatter: index " + std::to_string(index_ptr[flat_idx]) +
                        " out of range for dimension " + std::to_string(d) +
                        " with size " + std::to_string(input_shape[d]));
                }
                output_idx += idx_val * input_strides[d];
            } else {
                output_idx += coord * input_strides[d];
            }
        }
        output_ptr[output_idx] = src_ptr[flat_idx];
    }
}

// Template helper for masked_select
template<typename T, typename MaskFn>
void masked_select_impl(const T* input_ptr, T* output_ptr,
                        int64_t numel, MaskFn is_mask_true) {
    int64_t out_idx = 0;
    for (int64_t i = 0; i < numel; ++i) {
        if (is_mask_true(i)) {
            output_ptr[out_idx++] = input_ptr[i];
        }
    }
}

// Template helper for where
template<typename T, typename CondFn>
void where_impl(const T* x_ptr, const T* y_ptr, T* output_ptr,
                int64_t numel, CondFn is_cond_true) {
    for (int64_t i = 0; i < numel; ++i) {
        output_ptr[i] = is_cond_true(i) ? x_ptr[i] : y_ptr[i];
    }
}

// Template helper for nonzero - default uses memcmp against zero
template<typename T>
void nonzero_impl(const T* data, int64_t numel, std::vector<int64_t>& nz_indices) {
    if constexpr (std::is_same_v<T, bool>) {
        for (int64_t i = 0; i < numel; ++i) {
            if (data[i]) nz_indices.push_back(i);
        }
    } else if constexpr (std::is_same_v<T, Float16> || std::is_same_v<T, BFloat16>) {
        for (int64_t i = 0; i < numel; ++i) {
            if (static_cast<float>(data[i]) != 0.0f) nz_indices.push_back(i);
        }
    } else {
        for (int64_t i = 0; i < numel; ++i) {
            if (data[i] != T(0)) nz_indices.push_back(i);
        }
    }
}

} // namespace detail

/**
 * @brief Select elements along a dimension using an index tensor
 * @param input Input tensor
 * @param dim Dimension to select along
 * @param index Index tensor (must be Int64)
 * @return Tensor with selected elements
 */
auto index_select_kernel(const Tensor& input, int64_t dim, const Tensor& index) -> Tensor {
    // Validate index tensor
    if (index.dtype() != DType::Int64) {
        throw std::invalid_argument("index_select: index tensor must have dtype Int64");
    }

    // Normalize dimension
    const int64_t ndim = input.ndim();
    if (dim < 0) {
        dim += ndim;
    }
    if (dim < 0 || dim >= ndim) {
        throw std::out_of_range("index_select: dimension out of range");
    }

    // Get number of indices
    const int64_t num_indices = index.numel();

    // Compute output shape
    std::vector<int64_t> output_shape(input.shape().begin(), input.shape().end());
    output_shape[dim] = num_indices;

    // Handle empty index tensor - return empty tensor with correct shape
    // This matches PyTorch behavior where index_select with empty indices returns empty tensor
    if (num_indices == 0) {
        return Tensor(output_shape, input.dtype(), input.device());
    }

    // Create output tensor
    Tensor output(output_shape, input.dtype(), input.device());

    // Get index data (move to CPU if needed)
    auto index_cpu = index.device().type == Device::Type::CPU ? index : index.to(Device::cpu());
    const int64_t* index_data = index_cpu.data<int64_t>();

    // Compute strides for iteration
    const auto& in_shape = input.shape();
    const auto& in_strides = input.strides();
    const auto& out_strides = output.strides();

    // Compute size of inner dimensions (after dim)
    int64_t inner_size = 1;
    for (int64_t d = dim + 1; d < ndim; ++d) {
        inner_size *= in_shape[d];
    }

    // Compute size of outer dimensions (before dim)
    int64_t outer_size = 1;
    for (int64_t d = 0; d < dim; ++d) {
        outer_size *= in_shape[d];
    }

    // Perform the selection based on dtype using template helper
    // Macro to reduce repetition for pure-copy index_select dispatch
    #define INDEX_SELECT_DISPATCH(DTYPE_ENUM, CPP_TYPE) \
        if (input.dtype() == DTYPE_ENUM) { \
            detail::index_select_impl(input.data<CPP_TYPE>(), output.data<CPP_TYPE>(), \
                index_data, num_indices, outer_size, inner_size, \
                in_shape, in_strides, out_strides, dim); \
        }

    if (false) {}
    else INDEX_SELECT_DISPATCH(DType::Float32, float)
    else INDEX_SELECT_DISPATCH(DType::Float64, double)
    else INDEX_SELECT_DISPATCH(DType::Float16, Float16)
    else INDEX_SELECT_DISPATCH(DType::BFloat16, BFloat16)
    else INDEX_SELECT_DISPATCH(DType::Int32, int32_t)
    else INDEX_SELECT_DISPATCH(DType::Int64, int64_t)
    else INDEX_SELECT_DISPATCH(DType::Int8, int8_t)
    else INDEX_SELECT_DISPATCH(DType::UInt8, uint8_t)
    else INDEX_SELECT_DISPATCH(DType::Bool, bool)
    else {
        throw std::runtime_error("index_select: unsupported dtype");
    }

    #undef INDEX_SELECT_DISPATCH

    return output;
}

// Helper function to calculate strides from shape
static auto calculate_strides(const std::vector<int64_t>& shape) -> std::vector<int64_t> {
    std::vector<int64_t> strides(shape.size());
    int64_t stride = 1;
    for (int64_t i = shape.size() - 1; i >= 0; --i) {
        strides[i] = stride;
        stride *= shape[i];
    }
    return strides;
}

// Gather operation - select values at specified indices along a dimension
auto gather_kernel(const Tensor& input, int64_t dim, const Tensor& index) -> Tensor {
    // Validate index tensor dtype
    if (index.dtype() != DType::Int64) {
        throw std::invalid_argument("gather: index tensor must have dtype Int64");
    }

    // Ensure input and index are contiguous - strides are computed from shape
    // so non-contiguous tensors (from transpose/slice) would produce wrong results
    auto input_c = input.contiguous();
    auto index_c = index.contiguous();

    auto input_shape_span = input_c.shape();
    auto index_shape_span = index_c.shape();

    std::vector<int64_t> input_shape(input_shape_span.begin(), input_shape_span.end());
    std::vector<int64_t> index_shape(index_shape_span.begin(), index_shape_span.end());

    // Validate dimension
    if (dim < 0) dim += input_shape.size();
    if (dim < 0 || dim >= static_cast<int64_t>(input_shape.size())) {
        throw std::invalid_argument("gather: invalid dimension");
    }

    // Validate shapes match except for the gather dimension
    if (input_shape.size() != index_shape.size()) {
        throw std::invalid_argument("gather: input and index must have same number of dimensions");
    }

    // Output has same shape as index
    Tensor output(index_shape, input_c.dtype(), input_c.device());

    auto input_strides = calculate_strides(input_shape);
    auto index_strides = calculate_strides(index_shape);
    const int64_t numel = output.numel();
    const size_t ndims = index_shape.size();

    const int64_t* index_ptr = index_c.data<int64_t>();

    #define GATHER_DISPATCH(DTYPE_ENUM, CPP_TYPE) \
        if (input_c.dtype() == DTYPE_ENUM) { \
            detail::gather_impl(input_c.data<CPP_TYPE>(), output.data<CPP_TYPE>(), \
                index_ptr, numel, input_shape, input_strides, index_strides, ndims, dim); \
        }

    if (false) {}
    else GATHER_DISPATCH(DType::Float32, float)
    else GATHER_DISPATCH(DType::Float64, double)
    else GATHER_DISPATCH(DType::Float16, Float16)
    else GATHER_DISPATCH(DType::BFloat16, BFloat16)
    else GATHER_DISPATCH(DType::Int32, int32_t)
    else GATHER_DISPATCH(DType::Int64, int64_t)
    else GATHER_DISPATCH(DType::Int8, int8_t)
    else GATHER_DISPATCH(DType::UInt8, uint8_t)
    else GATHER_DISPATCH(DType::Bool, bool)
    else {
        throw std::runtime_error("gather: unsupported dtype");
    }

    #undef GATHER_DISPATCH

    return output;
}

// Scatter operation - distribute values at specified indices along a dimension
auto scatter_kernel(const Tensor& input, int64_t dim, const Tensor& index, const Tensor& src) -> Tensor {
    // Validate index tensor dtype
    if (index.dtype() != DType::Int64) {
        throw std::invalid_argument("scatter: index tensor must have dtype Int64");
    }

    // Ensure tensors are contiguous - strides are computed from shape
    auto input_c = input.contiguous();
    auto index_c = index.contiguous();
    auto src_c = src.contiguous();

    auto input_shape_span = input_c.shape();
    auto index_shape_span = index_c.shape();
    auto src_shape_span = src_c.shape();

    std::vector<int64_t> input_shape(input_shape_span.begin(), input_shape_span.end());
    std::vector<int64_t> index_shape(index_shape_span.begin(), index_shape_span.end());
    std::vector<int64_t> src_shape(src_shape_span.begin(), src_shape_span.end());

    // Validate dimension
    if (dim < 0) dim += input_shape.size();
    if (dim < 0 || dim >= static_cast<int64_t>(input_shape.size())) {
        throw std::invalid_argument("scatter: invalid dimension");
    }

    // Validate shapes
    if (index_shape != src_shape) {
        throw std::invalid_argument("scatter: index and src must have the same shape");
    }

    if (input_shape.size() != index_shape.size()) {
        throw std::invalid_argument("scatter: input and index must have same number of dimensions");
    }

    // Create output as copy of input
    Tensor output(input_shape, input_c.dtype(), input_c.device());

    auto input_strides = calculate_strides(input_shape);
    auto index_strides = calculate_strides(index_shape);
    const int64_t numel = index_c.numel();
    const size_t ndims = index_shape.size();

    const int64_t* index_ptr = index_c.data<int64_t>();

    #define SCATTER_DISPATCH(DTYPE_ENUM, CPP_TYPE) \
        if (input_c.dtype() == DTYPE_ENUM) { \
            detail::scatter_impl(input_c.data<CPP_TYPE>(), output.data<CPP_TYPE>(), \
                src_c.data<CPP_TYPE>(), index_ptr, input_c.numel(), numel, \
                input_shape, input_strides, index_strides, ndims, dim); \
        }

    if (false) {}
    else SCATTER_DISPATCH(DType::Float32, float)
    else SCATTER_DISPATCH(DType::Float64, double)
    else SCATTER_DISPATCH(DType::Float16, Float16)
    else SCATTER_DISPATCH(DType::BFloat16, BFloat16)
    else SCATTER_DISPATCH(DType::Int32, int32_t)
    else SCATTER_DISPATCH(DType::Int64, int64_t)
    else SCATTER_DISPATCH(DType::Int8, int8_t)
    else SCATTER_DISPATCH(DType::UInt8, uint8_t)
    else SCATTER_DISPATCH(DType::Bool, bool)
    else {
        throw std::runtime_error("scatter: unsupported dtype");
    }

    #undef SCATTER_DISPATCH

    return output;
}

// Scatter-add: accumulate source elements into output at indexed positions
auto scatter_add_kernel(const Tensor& input, int64_t dim, const Tensor& index, const Tensor& src) -> Tensor {
    if (index.dtype() != DType::Int64) {
        throw std::invalid_argument("scatter_add: index tensor must have dtype Int64");
    }

    auto input_c = input.contiguous();
    auto index_c = index.contiguous();
    auto src_c = src.contiguous();

    auto input_shape_span = input_c.shape();
    auto index_shape_span = index_c.shape();
    auto src_shape_span = src_c.shape();

    std::vector<int64_t> input_shape(input_shape_span.begin(), input_shape_span.end());
    std::vector<int64_t> index_shape(index_shape_span.begin(), index_shape_span.end());
    std::vector<int64_t> src_shape(src_shape_span.begin(), src_shape_span.end());

    if (dim < 0) dim += input_shape.size();
    if (dim < 0 || dim >= static_cast<int64_t>(input_shape.size())) {
        throw std::invalid_argument("scatter_add: invalid dimension");
    }
    if (index_shape != src_shape) {
        throw std::invalid_argument("scatter_add: index and src must have the same shape");
    }
    if (input_shape.size() != index_shape.size()) {
        throw std::invalid_argument("scatter_add: input and index must have same number of dimensions");
    }

    Tensor output(input_shape, input_c.dtype(), input_c.device());

    auto input_strides = calculate_strides(input_shape);
    auto index_strides = calculate_strides(index_shape);
    const int64_t numel = index_c.numel();
    const size_t ndims = index_shape.size();
    const int64_t* index_ptr = index_c.data<int64_t>();

    auto scatter_add_impl = [&]<typename T>() {
        const T* in_ptr = input_c.data<T>();
        T* out_ptr = output.data<T>();
        const T* src_ptr = src_c.data<T>();

        // Copy input to output first
        std::memcpy(out_ptr, in_ptr, input_c.numel() * sizeof(T));

        // Add source elements at indexed positions
        for (int64_t flat_idx = 0; flat_idx < numel; ++flat_idx) {
            int64_t temp = flat_idx;
            int64_t output_idx = 0;
            for (size_t d = 0; d < ndims; ++d) {
                int64_t coord = temp / index_strides[d];
                temp %= index_strides[d];
                if (static_cast<int64_t>(d) == dim) {
                    int64_t idx_val = index_ptr[flat_idx];
                    if (idx_val < 0) idx_val += input_shape[d];
                    if (idx_val < 0 || idx_val >= input_shape[d]) {
                        throw std::out_of_range("scatter_add: index " + std::to_string(index_ptr[flat_idx]) +
                            " out of range for dimension " + std::to_string(d) +
                            " with size " + std::to_string(input_shape[d]));
                    }
                    output_idx += idx_val * input_strides[d];
                } else {
                    output_idx += coord * input_strides[d];
                }
            }
            out_ptr[output_idx] += src_ptr[flat_idx];
        }
    };

    if (input_c.dtype() == DType::Float32) { scatter_add_impl.template operator()<float>(); }
    else if (input_c.dtype() == DType::Float64) { scatter_add_impl.template operator()<double>(); }
    else if (input_c.dtype() == DType::Float16) {
        // Float16: accumulate in Float32 to avoid precision loss, then convert back
        auto input_f32 = input_c.to(DType::Float32);
        auto src_f32 = src_c.to(DType::Float32);
        // Temporarily swap to Float32 tensors for accumulation
        auto saved_input = input_c;
        input_c = input_f32;
        src_c = src_f32;
        output = Tensor(input_shape, DType::Float32, input_f32.device());
        scatter_add_impl.template operator()<float>();
        output = output.to(DType::Float16);
    }
    else if (input_c.dtype() == DType::BFloat16) {
        // BFloat16: accumulate in Float32 to avoid precision loss, then convert back
        auto input_f32 = input_c.to(DType::Float32);
        auto src_f32 = src_c.to(DType::Float32);
        auto saved_input = input_c;
        input_c = input_f32;
        src_c = src_f32;
        output = Tensor(input_shape, DType::Float32, input_f32.device());
        scatter_add_impl.template operator()<float>();
        output = output.to(DType::BFloat16);
    }
    else if (input_c.dtype() == DType::Int8)  { scatter_add_impl.template operator()<int8_t>(); }
    else if (input_c.dtype() == DType::Int16) { scatter_add_impl.template operator()<int16_t>(); }
    else if (input_c.dtype() == DType::Int32) { scatter_add_impl.template operator()<int32_t>(); }
    else if (input_c.dtype() == DType::Int64) { scatter_add_impl.template operator()<int64_t>(); }
    else if (input_c.dtype() == DType::UInt8)  { scatter_add_impl.template operator()<uint8_t>(); }
    else if (input_c.dtype() == DType::UInt16) { scatter_add_impl.template operator()<uint16_t>(); }
    else if (input_c.dtype() == DType::UInt32) { scatter_add_impl.template operator()<uint32_t>(); }
    else if (input_c.dtype() == DType::UInt64) { scatter_add_impl.template operator()<uint64_t>(); }
    else if (input_c.dtype() == DType::Complex64) {
        scatter_add_impl.template operator()<std::complex<float>>();
    }
    else if (input_c.dtype() == DType::Complex128) {
        scatter_add_impl.template operator()<std::complex<double>>();
    }
    else {
        // Bool: += on bool is UB-prone (no arithmetic semantics).
        // FP8, quantized: dtype-specific accumulator and scaling required;
        // intentionally not supported here. Complex64/128 are handled above.
        throw std::runtime_error(
            "scatter_add: unsupported dtype " +
            std::string(dtype_name(input_c.dtype())));
    }

    return output;
}

// ============================================================================
// Scatter-reduce: scatter with configurable reduction
// ============================================================================

auto scatter_reduce_kernel(const Tensor& input, int64_t dim, const Tensor& index,
                           const Tensor& src, const std::string& reduce,
                           bool include_self) -> Tensor {
    if (index.dtype() != DType::Int64) {
        throw std::invalid_argument("scatter_reduce: index tensor must have dtype Int64");
    }

    auto input_c = input.contiguous();
    auto index_c = index.contiguous();
    auto src_c = src.contiguous();

    auto input_shape_span = input_c.shape();
    auto index_shape_span = index_c.shape();
    auto src_shape_span = src_c.shape();

    std::vector<int64_t> input_shape(input_shape_span.begin(), input_shape_span.end());
    std::vector<int64_t> index_shape(index_shape_span.begin(), index_shape_span.end());
    std::vector<int64_t> src_shape(src_shape_span.begin(), src_shape_span.end());

    if (dim < 0) dim += static_cast<int64_t>(input_shape.size());
    if (dim < 0 || dim >= static_cast<int64_t>(input_shape.size())) {
        throw std::invalid_argument("scatter_reduce: invalid dimension");
    }
    if (index_shape != src_shape) {
        throw std::invalid_argument("scatter_reduce: index and src must have the same shape");
    }
    if (input_shape.size() != index_shape.size()) {
        throw std::invalid_argument("scatter_reduce: input and index must have same number of dimensions");
    }

    // Validate reduce mode early
    if (reduce != "sum" && reduce != "prod" && reduce != "mean" &&
        reduce != "amax" && reduce != "amin") {
        throw std::invalid_argument("scatter_reduce: unknown reduce mode '" + reduce + "'");
    }

    // Handle Float16/BFloat16 by upcasting
    if (input_c.dtype() == DType::Float16 || input_c.dtype() == DType::BFloat16) {
        DType orig_dtype = input_c.dtype();
        auto f32_in = input_c.to(DType::Float32);
        auto f32_src = src_c.to(DType::Float32);
        auto result = scatter_reduce_kernel(f32_in, dim, index_c, f32_src, reduce, include_self);
        return result.to(orig_dtype);
    }

    auto output = input_c.clone();

    auto input_strides = calculate_strides(input_shape);
    auto index_strides = calculate_strides(index_shape);
    const int64_t numel = index_c.numel();
    const size_t ndims = index_shape.size();
    const int64_t* index_ptr = index_c.data<int64_t>();

    // Helper: compute output flat index for a given index-space flat_idx
    auto compute_output_idx = [&](int64_t flat_idx) -> int64_t {
        int64_t temp = flat_idx;
        int64_t output_idx = 0;
        for (size_t d = 0; d < ndims; ++d) {
            int64_t coord = temp / index_strides[d];
            temp %= index_strides[d];
            if (static_cast<int64_t>(d) == dim) {
                int64_t idx_val = index_ptr[flat_idx];
                if (idx_val < 0) idx_val += input_shape[d];
                if (idx_val < 0 || idx_val >= input_shape[d]) {
                    throw std::out_of_range("scatter_reduce: index " + std::to_string(index_ptr[flat_idx]) +
                        " out of range for dimension " + std::to_string(d) +
                        " with size " + std::to_string(input_shape[d]));
                }
                output_idx += idx_val * input_strides[d];
            } else {
                output_idx += coord * input_strides[d];
            }
        }
        return output_idx;
    };

    auto do_scatter_reduce = [&]<typename T>() {
        T* out_ptr = output.data<T>();
        const T* src_ptr = src_c.data<T>();
        int64_t out_numel = output.numel();

        // If !include_self, we need to set touched output positions to the
        // reduction identity value before applying the reduction.
        // Track which positions are touched.
        std::vector<bool> touched;
        if (!include_self) {
            touched.resize(out_numel, false);
            // First pass: mark touched positions
            for (int64_t flat_idx = 0; flat_idx < numel; ++flat_idx) {
                int64_t oidx = compute_output_idx(flat_idx);
                touched[oidx] = true;
            }
            // Set touched positions to identity values
            T identity;
            if (reduce == "sum" || reduce == "mean") {
                identity = T(0);
            } else if (reduce == "prod") {
                identity = T(1);
            } else if (reduce == "amax") {
                identity = std::numeric_limits<T>::lowest();
            } else if (reduce == "amin") {
                identity = std::numeric_limits<T>::max();
            } else {
                throw std::invalid_argument("scatter_reduce: unknown reduce mode '" + reduce + "'");
            }
            for (int64_t i = 0; i < out_numel; ++i) {
                if (touched[i]) out_ptr[i] = identity;
            }
        }

        // For "mean" mode, we need to count how many values are accumulated per output position.
        std::vector<int64_t> counts;
        if (reduce == "mean") {
            counts.resize(out_numel, include_self ? 1 : 0);
        }

        // Apply scatter reduction
        for (int64_t flat_idx = 0; flat_idx < numel; ++flat_idx) {
            int64_t oidx = compute_output_idx(flat_idx);
            T val = src_ptr[flat_idx];

            if (reduce == "sum" || reduce == "mean") {
                out_ptr[oidx] += val;
            } else if (reduce == "prod") {
                out_ptr[oidx] *= val;
            } else if (reduce == "amax") {
                if (val > out_ptr[oidx]) out_ptr[oidx] = val;
            } else if (reduce == "amin") {
                if (val < out_ptr[oidx]) out_ptr[oidx] = val;
            }

            if (reduce == "mean") {
                counts[oidx]++;
            }
        }

        // For "mean", divide by counts
        if (reduce == "mean") {
            for (int64_t i = 0; i < out_numel; ++i) {
                if (counts[i] > (include_self ? 1 : 0)) {
                    out_ptr[i] /= static_cast<T>(counts[i]);
                }
            }
        }
    };

    if (input_c.dtype() == DType::Float32) { do_scatter_reduce.template operator()<float>(); }
    else if (input_c.dtype() == DType::Float64) { do_scatter_reduce.template operator()<double>(); }
    else if (input_c.dtype() == DType::Int32) { do_scatter_reduce.template operator()<int32_t>(); }
    else if (input_c.dtype() == DType::Int64) { do_scatter_reduce.template operator()<int64_t>(); }
    else {
        throw std::runtime_error("scatter_reduce: unsupported dtype " +
            std::string(dtype_name(input_c.dtype())));
    }

    return output;
}

// Masked select operation - select elements where mask is true
auto masked_select_kernel(const Tensor& input, const Tensor& mask) -> Tensor {
    auto input_shape = input.shape();
    auto mask_shape = mask.shape();

    // Validate shapes match
    if (!std::equal(input_shape.begin(), input_shape.end(), mask_shape.begin(), mask_shape.end())) {
        throw std::invalid_argument("masked_select: input and mask must have same shape");
    }

    const int64_t numel = input.numel();

    // Support both Bool dtype and Float32 dtype (CUDA may convert Bool to Float32 during device transfers)
    int64_t true_count = 0;
    const bool use_float_mask = (mask.dtype() == DType::Float32);
    const bool* bool_mask_ptr = nullptr;
    const float* float_mask_ptr = nullptr;

    if (mask.dtype() == DType::Bool) {
        bool_mask_ptr = mask.data<bool>();
        // First pass: count true values in mask
        for (int64_t i = 0; i < numel; ++i) {
            if (bool_mask_ptr[i]) ++true_count;
        }
    } else if (use_float_mask) {
        float_mask_ptr = mask.data<float>();
        // First pass: count non-zero values (treating Float32 as boolean)
        for (int64_t i = 0; i < numel; ++i) {
            if (float_mask_ptr[i] != 0.0f) ++true_count;
        }
    } else {
        char msg[256];
        snprintf(msg, sizeof(msg), "masked_select: mask tensor must have dtype Bool or Float32, but got dtype %d",
                 static_cast<int>(mask.dtype()));
        throw std::invalid_argument(msg);
    }

    // Create output with size = number of true values
    Tensor output({true_count}, input.dtype(), input.device());

    if (true_count == 0) {
        return output;
    }

    // Helper to check mask value for both Bool and Float32 masks
    auto is_mask_true = [use_float_mask, bool_mask_ptr, float_mask_ptr](int64_t i) -> bool {
        return use_float_mask ? (float_mask_ptr[i] != 0.0f) : bool_mask_ptr[i];
    };

    // Second pass: copy selected elements
    #define MASKED_SELECT_DISPATCH(DTYPE_ENUM, CPP_TYPE) \
        if (input.dtype() == DTYPE_ENUM) { \
            detail::masked_select_impl(input.data<CPP_TYPE>(), output.data<CPP_TYPE>(), \
                numel, is_mask_true); \
        }

    if (false) {}
    else MASKED_SELECT_DISPATCH(DType::Float32, float)
    else MASKED_SELECT_DISPATCH(DType::Float64, double)
    else MASKED_SELECT_DISPATCH(DType::Float16, Float16)
    else MASKED_SELECT_DISPATCH(DType::BFloat16, BFloat16)
    else MASKED_SELECT_DISPATCH(DType::Int32, int32_t)
    else MASKED_SELECT_DISPATCH(DType::Int64, int64_t)
    else MASKED_SELECT_DISPATCH(DType::Int8, int8_t)
    else MASKED_SELECT_DISPATCH(DType::UInt8, uint8_t)
    else MASKED_SELECT_DISPATCH(DType::Bool, bool)
    else {
        throw std::runtime_error("masked_select: unsupported dtype");
    }

    #undef MASKED_SELECT_DISPATCH

    return output;
}

// Masked fill operation - fill elements with value where mask is true
auto masked_fill_kernel(const Tensor& input, const Tensor& mask_in, float value) -> Tensor {
    auto input_shape = input.shape();

    // C.6: broadcast the mask to input shape (matches the CUDA wrapper).
    // The common attention-mask pattern is (B, 1, S, S) broadcasting across
    // H heads to (B, H, S, S); previously this kernel required exact-shape
    // masks and forced the caller to allocate a B*H*S*S boolean buffer.
    std::vector<int64_t> input_shape_vec(input_shape.begin(), input_shape.end());
    Tensor mask = mask_in;
    std::vector<int64_t> mask_shape_vec(mask_in.shape().begin(), mask_in.shape().end());
    if (mask_shape_vec != input_shape_vec) {
        auto broadcast_shape =
            tenzor::broadcast_shapes(mask_in.shape(), input_shape);
        if (broadcast_shape != input_shape_vec) {
            throw std::invalid_argument(
                "masked_fill: mask shape is not broadcast-compatible with "
                "input shape");
        }
        mask = tenzor::broadcast_to(mask_in, input_shape_vec).contiguous();
    }
    auto mask_shape = mask.shape();

    const int64_t numel = input.numel();

    // Create output as copy of input (convert span to vector for constructor)
    std::vector<int64_t> shape_vec(input_shape.begin(), input_shape.end());
    Tensor output(shape_vec, input.dtype(), input.device());

    // Support both Bool dtype and Float32 dtype (for CUDA compatibility)
    const bool use_float_mask = (mask.dtype() == DType::Float32);
    const bool* bool_mask_ptr = nullptr;
    const float* float_mask_ptr = nullptr;

    if (mask.dtype() == DType::Bool) {
        bool_mask_ptr = mask.data<bool>();
    } else if (use_float_mask) {
        float_mask_ptr = mask.data<float>();
    } else {
        char msg[256];
        snprintf(msg, sizeof(msg), "masked_fill: mask tensor must have dtype Bool or Float32, but got dtype %d",
                 static_cast<int>(mask.dtype()));
        throw std::invalid_argument(msg);
    }

    // Helper to check mask value
    auto is_mask_true = [use_float_mask, bool_mask_ptr, float_mask_ptr](int64_t i) -> bool {
        return use_float_mask ? (float_mask_ptr[i] != 0.0f) : bool_mask_ptr[i];
    };

    // Fill elements based on dtype
    if (input.dtype() == DType::Float32) {
        const float* input_ptr = input.data<float>();
        float* output_ptr = output.data<float>();
        const float fill_value = value;

        for (int64_t i = 0; i < numel; ++i) {
            output_ptr[i] = is_mask_true(i) ? fill_value : input_ptr[i];
        }
    } else if (input.dtype() == DType::Float64) {
        const double* input_ptr = input.data<double>();
        double* output_ptr = output.data<double>();
        const double fill_value = static_cast<double>(value);

        for (int64_t i = 0; i < numel; ++i) {
            output_ptr[i] = is_mask_true(i) ? fill_value : input_ptr[i];
        }
    } else if (input.dtype() == DType::Int32) {
        const int32_t* input_ptr = input.data<int32_t>();
        int32_t* output_ptr = output.data<int32_t>();
        const int32_t fill_value = static_cast<int32_t>(value);

        for (int64_t i = 0; i < numel; ++i) {
            output_ptr[i] = is_mask_true(i) ? fill_value : input_ptr[i];
        }
    } else if (input.dtype() == DType::Int64) {
        const int64_t* input_ptr = input.data<int64_t>();
        int64_t* output_ptr = output.data<int64_t>();
        const int64_t fill_value = static_cast<int64_t>(value);

        for (int64_t i = 0; i < numel; ++i) {
            output_ptr[i] = is_mask_true(i) ? fill_value : input_ptr[i];
        }
    } else if (input.dtype() == DType::Float16) {
        const Float16* input_ptr = input.data<Float16>();
        Float16* output_ptr = output.data<Float16>();
        const Float16 fill_value = Float16(value);

        for (int64_t i = 0; i < numel; ++i) {
            output_ptr[i] = is_mask_true(i) ? fill_value : input_ptr[i];
        }
    } else if (input.dtype() == DType::BFloat16) {
        const BFloat16* input_ptr = input.data<BFloat16>();
        BFloat16* output_ptr = output.data<BFloat16>();
        const BFloat16 fill_value = BFloat16(value);

        for (int64_t i = 0; i < numel; ++i) {
            output_ptr[i] = is_mask_true(i) ? fill_value : input_ptr[i];
        }
    } else if (input.dtype() == DType::Bool) {
        const bool* input_ptr = input.data<bool>();
        bool* output_ptr = output.data<bool>();
        const bool fill_value = (value != 0.0f);

        for (int64_t i = 0; i < numel; ++i) {
            output_ptr[i] = is_mask_true(i) ? fill_value : input_ptr[i];
        }
    } else {
        throw std::runtime_error("masked_fill: unsupported dtype");
    }

    return output;
}

// Where operation - select elements from x or y based on condition
auto where_kernel(const Tensor& condition, const Tensor& x, const Tensor& y) -> Tensor {
    auto cond_shape = condition.shape();
    auto x_shape = x.shape();
    auto y_shape = y.shape();

    // Validate shapes match
    if (!std::equal(cond_shape.begin(), cond_shape.end(), x_shape.begin(), x_shape.end()) ||
        !std::equal(cond_shape.begin(), cond_shape.end(), y_shape.begin(), y_shape.end())) {
        throw std::invalid_argument("where: condition, x, and y must have same shape");
    }

    // Validate x and y have same dtype
    if (x.dtype() != y.dtype()) {
        throw std::invalid_argument("where: x and y must have same dtype");
    }

    const int64_t numel = condition.numel();

    // Create output with same shape and dtype as x
    std::vector<int64_t> shape_vec(x_shape.begin(), x_shape.end());
    Tensor output(shape_vec, x.dtype(), x.device());

    // Support both Bool dtype and Float32 dtype for condition (for CUDA compatibility)
    const bool use_float_cond = (condition.dtype() == DType::Float32);
    const bool* bool_cond_ptr = nullptr;
    const float* float_cond_ptr = nullptr;

    if (condition.dtype() == DType::Bool) {
        bool_cond_ptr = condition.data<bool>();
    } else if (use_float_cond) {
        float_cond_ptr = condition.data<float>();
    } else {
        char msg[256];
        snprintf(msg, sizeof(msg), "where: condition tensor must have dtype Bool or Float32, but got dtype %d",
                 static_cast<int>(condition.dtype()));
        throw std::invalid_argument(msg);
    }

    // Helper to check condition value
    auto is_cond_true = [use_float_cond, bool_cond_ptr, float_cond_ptr](int64_t i) -> bool {
        return use_float_cond ? (float_cond_ptr[i] != 0.0f) : bool_cond_ptr[i];
    };

    // Select elements based on dtype
    if (x.dtype() == DType::Float32) {
        const float* x_ptr = x.data<float>();
        const float* y_ptr = y.data<float>();
        float* output_ptr = output.data<float>();

        // Use SIMD optimization for Float32 condition with Float32 x,y
        if (use_float_cond) {
#if defined(INDEXING_HAS_AVX512)
            fast_math::where_batch_avx512(float_cond_ptr, x_ptr, y_ptr, output_ptr, static_cast<size_t>(numel));
#elif defined(INDEXING_HAS_AVX2)
            fast_math::where_batch_avx2(float_cond_ptr, x_ptr, y_ptr, output_ptr, static_cast<size_t>(numel));
#else
            #pragma omp parallel for if(numel > 100000)
            for (int64_t i = 0; i < numel; ++i) {
                output_ptr[i] = is_cond_true(i) ? x_ptr[i] : y_ptr[i];
            }
#endif
        } else {
            // Bool condition - scalar loop with OpenMP
            #pragma omp parallel for if(numel > 100000)
            for (int64_t i = 0; i < numel; ++i) {
                output_ptr[i] = bool_cond_ptr[i] ? x_ptr[i] : y_ptr[i];
            }
        }
    } else {
        // Generic path for all other dtypes
        #define WHERE_DISPATCH(DTYPE_ENUM, CPP_TYPE) \
            if (x.dtype() == DTYPE_ENUM) { \
                detail::where_impl(x.data<CPP_TYPE>(), y.data<CPP_TYPE>(), \
                    output.data<CPP_TYPE>(), numel, is_cond_true); \
            }

        if (false) {}
        else WHERE_DISPATCH(DType::Float64, double)
        else WHERE_DISPATCH(DType::Float16, Float16)
        else WHERE_DISPATCH(DType::BFloat16, BFloat16)
        else WHERE_DISPATCH(DType::Int32, int32_t)
        else WHERE_DISPATCH(DType::Int64, int64_t)
        else WHERE_DISPATCH(DType::Int8, int8_t)
        else WHERE_DISPATCH(DType::UInt8, uint8_t)
        else WHERE_DISPATCH(DType::Bool, bool)
        else {
            throw std::runtime_error("where: unsupported dtype");
        }

        #undef WHERE_DISPATCH
    }

    return output;
}

auto nonzero_kernel(const Tensor& input) -> Tensor {
    const int64_t numel = input.numel();
    const int64_t ndim = input.ndim();

    // First pass: count nonzero elements
    std::vector<int64_t> nz_indices;
    nz_indices.reserve(numel / 4);  // Heuristic

    #define NONZERO_DISPATCH(DTYPE_ENUM, CPP_TYPE) \
        if (input.dtype() == DTYPE_ENUM) { \
            detail::nonzero_impl(input.data<CPP_TYPE>(), numel, nz_indices); \
        }

    if (false) {}
    else NONZERO_DISPATCH(DType::Float32, float)
    else NONZERO_DISPATCH(DType::Float64, double)
    else NONZERO_DISPATCH(DType::Float16, Float16)
    else NONZERO_DISPATCH(DType::BFloat16, BFloat16)
    else NONZERO_DISPATCH(DType::Int32, int32_t)
    else NONZERO_DISPATCH(DType::Int64, int64_t)
    else NONZERO_DISPATCH(DType::Int8, int8_t)
    else NONZERO_DISPATCH(DType::UInt8, uint8_t)
    else NONZERO_DISPATCH(DType::Bool, bool)
    else {
        throw std::runtime_error("nonzero: unsupported dtype");
    }

    #undef NONZERO_DISPATCH

    int64_t nnz = static_cast<int64_t>(nz_indices.size());

    // Output shape: (nnz, ndim)
    Tensor output({nnz, ndim}, DType::Int64, input.device());
    int64_t* out_data = output.data<int64_t>();

    std::vector<int64_t> shape_vec(input.shape().begin(), input.shape().end());
    auto strides = calculate_strides(shape_vec);

    // Convert linear indices to multi-dimensional indices
    for (int64_t i = 0; i < nnz; ++i) {
        int64_t linear = nz_indices[i];
        for (int64_t d = 0; d < ndim; ++d) {
            out_data[i * ndim + d] = linear / strides[d];
            linear %= strides[d];
        }
    }

    return output;
}

auto one_hot_kernel(const Tensor& indices, int64_t num_classes) -> Tensor {
    if (indices.dtype() != DType::Int64 && indices.dtype() != DType::Int32) {
        throw std::runtime_error("one_hot: indices must be integer type");
    }

    const int64_t numel = indices.numel();

    // If num_classes not specified, infer from max value
    if (num_classes <= 0) {
        if (indices.dtype() == DType::Int64) {
            const int64_t* data = indices.data<int64_t>();
            for (int64_t i = 0; i < numel; ++i) {
                num_classes = std::max(num_classes, data[i] + 1);
            }
        } else {
            const int32_t* data = indices.data<int32_t>();
            for (int64_t i = 0; i < numel; ++i) {
                num_classes = std::max(num_classes, static_cast<int64_t>(data[i]) + 1);
            }
        }
    }

    // Output shape: indices_shape + [num_classes]
    auto idx_shape = indices.shape();
    std::vector<int64_t> out_shape(idx_shape.begin(), idx_shape.end());
    out_shape.push_back(num_classes);

    // Create zero-filled output (Float32)
    Tensor output(out_shape, DType::Float32, indices.device());
    float* out_data = output.data<float>();
    std::memset(out_data, 0, output.numel() * sizeof(float));

    // Set the appropriate positions to 1
    if (indices.dtype() == DType::Int64) {
        const int64_t* idx_data = indices.data<int64_t>();
        for (int64_t i = 0; i < numel; ++i) {
            int64_t cls = idx_data[i];
            if (cls >= 0 && cls < num_classes) {
                out_data[i * num_classes + cls] = 1.0f;
            }
        }
    } else {
        const int32_t* idx_data = indices.data<int32_t>();
        for (int64_t i = 0; i < numel; ++i) {
            int64_t cls = static_cast<int64_t>(idx_data[i]);
            if (cls >= 0 && cls < num_classes) {
                out_data[i * num_classes + cls] = 1.0f;
            }
        }
    }

    return output;
}

auto take_kernel(const Tensor& input, const Tensor& indices) -> Tensor {
    // Take elements from flattened input using indices
    if (indices.dtype() != DType::Int64) {
        throw std::runtime_error("take: indices must have dtype Int64");
    }

    const int64_t num_indices = indices.numel();
    const int64_t input_numel = input.numel();
    const int64_t* idx_data = indices.data<int64_t>();

    // Output shape matches indices shape
    std::vector<int64_t> out_shape(indices.shape().begin(), indices.shape().end());
    Tensor output(out_shape, input.dtype(), input.device());

    const size_t elem_size = dtype_size(input.dtype());
    Tensor cont = input.is_contiguous() ? input : input.contiguous();
    const auto* src = static_cast<const uint8_t*>(cont.data<uint8_t>());
    auto* dst = output.data<uint8_t>();

    // Pre-validate all indices sequentially — throwing inside an OMP parallel
    // region is undefined behavior, so we validate before entering the loop.
    for (int64_t i = 0; i < num_indices; ++i) {
        int64_t idx = idx_data[i];
        if (idx < 0) idx += input_numel;
        if (idx < 0 || idx >= input_numel) {
            throw std::out_of_range("take: index " + std::to_string(idx_data[i]) +
                " out of range [" + std::to_string(-input_numel) + ", " +
                std::to_string(input_numel) + ")");
        }
    }

    #pragma omp parallel for if(num_indices > 65536)
    for (int64_t i = 0; i < num_indices; ++i) {
        int64_t idx = idx_data[i];
        if (idx < 0) idx += input_numel;
        std::memcpy(dst + i * elem_size, src + idx * elem_size, elem_size);
    }

    return output;
}

auto put_kernel(Tensor& input, const Tensor& indices, const Tensor& source,
                bool accumulate) -> Tensor {
    if (indices.dtype() != DType::Int64) {
        throw std::runtime_error("put: indices must have dtype Int64");
    }

    const int64_t num_indices = indices.numel();
    const int64_t input_numel = input.numel();
    const int64_t* idx_data = indices.data<int64_t>();

    Tensor result = input.is_contiguous() ? input : input.contiguous();
    const size_t elem_size = dtype_size(result.dtype());
    auto* dst = result.data<uint8_t>();
    const auto* src_data = static_cast<const uint8_t*>(source.data<uint8_t>());

    // Pre-validate all indices sequentially to avoid OOB writes.
    for (int64_t i = 0; i < num_indices; ++i) {
        int64_t idx = idx_data[i];
        if (idx < 0) idx += input_numel;
        if (idx < 0 || idx >= input_numel) {
            throw std::out_of_range("put: index " + std::to_string(idx_data[i]) +
                " out of range [" + std::to_string(-input_numel) + ", " +
                std::to_string(input_numel) + ")");
        }
    }

    if (accumulate && result.dtype() == DType::Float32) {
        float* dst_f = result.data<float>();
        const float* src_f = source.data<float>();
        for (int64_t i = 0; i < num_indices; ++i) {
            int64_t idx = idx_data[i];
            if (idx < 0) idx += input_numel;
            dst_f[idx] += src_f[i];
        }
    } else if (accumulate && result.dtype() == DType::Float64) {
        double* dst_d = result.data<double>();
        const double* src_d = source.data<double>();
        for (int64_t i = 0; i < num_indices; ++i) {
            int64_t idx = idx_data[i];
            if (idx < 0) idx += input_numel;
            dst_d[idx] += src_d[i];
        }
    } else {
        for (int64_t i = 0; i < num_indices; ++i) {
            int64_t idx = idx_data[i];
            if (idx < 0) idx += input_numel;
            std::memcpy(dst + idx * elem_size, src_data + i * elem_size, elem_size);
        }
    }

    return result;
}

// ============================================================================
// SearchSorted: binary search per element in a sorted 1-D sequence
// ============================================================================

auto searchsorted_kernel(std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
    const Tensor& sorted_sequence = inputs[0];
    const Tensor& values = inputs[1];
    bool right = attrs.get_bool(AttrKey::Right, false);

    if (sorted_sequence.ndim() != 1) {
        throw std::runtime_error("searchsorted: sorted_sequence must be 1-D");
    }

    Tensor seq_cont = sorted_sequence.contiguous();
    Tensor val_cont = values.contiguous();
    int64_t seq_len = seq_cont.shape()[0];
    int64_t num_values = val_cont.numel();

    Tensor result(std::vector<int64_t>(values.shape().begin(), values.shape().end()),
                  DType::Int64, values.device());

    auto search_typed = [&](const auto* seq_ptr, const auto* val_ptr, int64_t* out_ptr) {
        #ifdef _OPENMP
        #pragma omp parallel for if(num_values > 4096)
        #endif
        for (int64_t i = 0; i < num_values; ++i) {
            auto v = val_ptr[i];
            int64_t lo = 0, hi = seq_len;
            while (lo < hi) {
                int64_t mid = lo + (hi - lo) / 2;
                bool go_right = right ? (seq_ptr[mid] <= v) : (seq_ptr[mid] < v);
                if (go_right) {
                    lo = mid + 1;
                } else {
                    hi = mid;
                }
            }
            out_ptr[i] = lo;
        }
    };

    auto* out_ptr = result.data<int64_t>();
    switch (sorted_sequence.dtype()) {
        case DType::Float32:
            search_typed(seq_cont.data<float>(), val_cont.data<float>(), out_ptr);
            break;
        case DType::Float64:
            search_typed(seq_cont.data<double>(), val_cont.data<double>(), out_ptr);
            break;
        case DType::Int32:
            search_typed(seq_cont.data<int32_t>(), val_cont.data<int32_t>(), out_ptr);
            break;
        case DType::Int64:
            search_typed(seq_cont.data<int64_t>(), val_cont.data<int64_t>(), out_ptr);
            break;
        case DType::Float16:
        case DType::BFloat16: {
            // Convert to Float32 for search
            auto seq_f32 = sorted_sequence.to(DType::Float32).contiguous();
            auto val_f32 = values.to(DType::Float32).contiguous();
            search_typed(seq_f32.data<float>(), val_f32.data<float>(), out_ptr);
            break;
        }
        case DType::Int8:
            search_typed(seq_cont.data<int8_t>(), val_cont.data<int8_t>(), out_ptr);
            break;
        case DType::UInt8:
            search_typed(seq_cont.data<uint8_t>(), val_cont.data<uint8_t>(), out_ptr);
            break;
        case DType::Int16:
            search_typed(seq_cont.data<int16_t>(), val_cont.data<int16_t>(), out_ptr);
            break;
        default:
            throw std::runtime_error("searchsorted: unsupported dtype " +
                                     std::string(dtype_name(sorted_sequence.dtype())));
    }

    return result;
}

// ============================================================================
// bincount — Count occurrences of each value in an integer tensor
// ============================================================================

auto bincount_kernel(const Tensor& input, const Tensor* weights, int64_t minlength) -> Tensor {
    if (input.ndim() != 1) {
        throw std::runtime_error("bincount: input must be 1D");
    }

    const int64_t n = input.numel();

    // Determine output size: max(max(input)+1, minlength)
    int64_t max_val = -1;
    if (input.dtype() == DType::Int64) {
        const auto* data = input.data<int64_t>();
        for (int64_t i = 0; i < n; ++i) {
            if (data[i] < 0) {
                throw std::runtime_error("bincount: input must contain non-negative integers");
            }
            if (data[i] > max_val) max_val = data[i];
        }
    } else if (input.dtype() == DType::Int32) {
        const auto* data = input.data<int32_t>();
        for (int64_t i = 0; i < n; ++i) {
            if (data[i] < 0) {
                throw std::runtime_error("bincount: input must contain non-negative integers");
            }
            if (data[i] > max_val) max_val = data[i];
        }
    } else {
        throw std::runtime_error("bincount: input must be Int32 or Int64");
    }

    int64_t output_size = std::max(max_val + 1, minlength);

    bool has_weights = (weights != nullptr);
    DType out_dtype = has_weights ? DType::Float64 : DType::Int64;

    Tensor output({output_size}, out_dtype, input.device());

    if (has_weights) {
        auto* out = output.data<double>();
        std::memset(out, 0, static_cast<size_t>(output_size) * sizeof(double));

        // Support Float32 and Float64 weights
        auto get_idx = [&](int64_t i) -> int64_t {
            if (input.dtype() == DType::Int64)
                return input.data<int64_t>()[i];
            return static_cast<int64_t>(input.data<int32_t>()[i]);
        };

        if (weights->dtype() == DType::Float32) {
            const auto* w = weights->data<float>();
            for (int64_t i = 0; i < n; ++i) {
                out[get_idx(i)] += static_cast<double>(w[i]);
            }
        } else if (weights->dtype() == DType::Float64) {
            const auto* w = weights->data<double>();
            for (int64_t i = 0; i < n; ++i) {
                out[get_idx(i)] += w[i];
            }
        } else {
            // Convert weights to Float64
            auto w_f64 = weights->to(DType::Float64);
            const auto* w = w_f64.data<double>();
            for (int64_t i = 0; i < n; ++i) {
                out[get_idx(i)] += w[i];
            }
        }
    } else {
        auto* out = output.data<int64_t>();
        std::memset(out, 0, static_cast<size_t>(output_size) * sizeof(int64_t));

        if (input.dtype() == DType::Int64) {
            const auto* data = input.data<int64_t>();
            for (int64_t i = 0; i < n; ++i) {
                out[data[i]]++;
            }
        } else {
            const auto* data = input.data<int32_t>();
            for (int64_t i = 0; i < n; ++i) {
                out[data[i]]++;
            }
        }
    }

    return output;
}

// =============================================================================
// TakeAlongDim kernel
// =============================================================================

auto take_along_dim_kernel(const Tensor& input, const Tensor& indices, int64_t dim) -> Tensor {
    auto in_shape = input.shape();
    auto idx_shape = indices.shape();
    int64_t ndim = in_shape.size();

    // Normalize dim
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) {
        throw std::out_of_range("take_along_dim: dim out of range");
    }

    // Output has same shape as indices
    Tensor output(std::vector<int64_t>(idx_shape.begin(), idx_shape.end()),
                  input.dtype(), input.device());

    int64_t numel = indices.numel();
    if (numel == 0) return output;

    // Compute outer_size (product of dims before dim), dim_size, inner_size (product of dims after dim)
    int64_t outer_size = 1;
    for (int64_t d = 0; d < dim; ++d) outer_size *= idx_shape[d];
    int64_t idx_dim_size = idx_shape[dim];
    int64_t in_dim_size = in_shape[dim];
    int64_t inner_size = 1;
    for (int64_t d = dim + 1; d < ndim; ++d) inner_size *= idx_shape[d];

    const int64_t* idx_ptr = indices.data<int64_t>();

    auto copy_elements = [&](auto* in_ptr, auto* out_ptr) {
        #pragma omp parallel for if(numel > 65536)
        for (int64_t i = 0; i < numel; ++i) {
            int64_t outer = i / (idx_dim_size * inner_size);
            int64_t rem = i % (idx_dim_size * inner_size);
            int64_t inner = rem % inner_size;

            int64_t src_idx = idx_ptr[i];
            if (src_idx < 0) src_idx += in_dim_size;

            int64_t in_offset = outer * (in_dim_size * inner_size) + src_idx * inner_size + inner;
            out_ptr[i] = in_ptr[in_offset];
        }
    };

    switch (input.dtype()) {
        case DType::Float32: copy_elements(input.data<float>(), output.data<float>()); break;
        case DType::Float64: copy_elements(input.data<double>(), output.data<double>()); break;
        case DType::Int32:   copy_elements(input.data<int32_t>(), output.data<int32_t>()); break;
        case DType::Int64:   copy_elements(input.data<int64_t>(), output.data<int64_t>()); break;
        case DType::Int16:   copy_elements(input.data<int16_t>(), output.data<int16_t>()); break;
        case DType::Int8:    copy_elements(input.data<int8_t>(), output.data<int8_t>()); break;
        case DType::UInt8:   copy_elements(input.data<uint8_t>(), output.data<uint8_t>()); break;
        case DType::Bool:    copy_elements(input.data<bool>(), output.data<bool>()); break;
        default:
            // Float16/BFloat16: use byte-level copy
            {
                auto elem_size = input.dtype_size();
                const uint8_t* in_bytes = static_cast<const uint8_t*>(input.data_ptr());
                uint8_t* out_bytes = static_cast<uint8_t*>(output.data_ptr());
                for (int64_t i = 0; i < numel; ++i) {
                    int64_t outer = i / (idx_dim_size * inner_size);
                    int64_t rem = i % (idx_dim_size * inner_size);
                    int64_t inner = rem % inner_size;
                    int64_t src_idx = idx_ptr[i];
                    if (src_idx < 0) src_idx += in_dim_size;
                    int64_t in_offset = outer * (in_dim_size * inner_size) + src_idx * inner_size + inner;
                    std::memcpy(out_bytes + i * elem_size, in_bytes + in_offset * elem_size, elem_size);
                }
            }
            break;
    }

    return output;
}

// =============================================================================
// MaskedScatter kernel
// =============================================================================

auto masked_scatter_kernel(const Tensor& input, const Tensor& mask, const Tensor& source) -> Tensor {
    Tensor output = input.clone();
    int64_t numel = input.numel();
    const bool* mask_ptr = mask.data<bool>();

    auto scatter_values = [&](auto* out_ptr, const auto* src_ptr, int64_t src_numel) {
        int64_t src_idx = 0;
        for (int64_t i = 0; i < numel; ++i) {
            if (mask_ptr[i]) {
                if (src_idx >= src_numel) {
                    throw std::runtime_error("masked_scatter: source has fewer elements than mask true count");
                }
                out_ptr[i] = src_ptr[src_idx++];
            }
        }
    };

    switch (input.dtype()) {
        case DType::Float32: scatter_values(output.data<float>(), source.data<float>(), source.numel()); break;
        case DType::Float64: scatter_values(output.data<double>(), source.data<double>(), source.numel()); break;
        case DType::Int32:   scatter_values(output.data<int32_t>(), source.data<int32_t>(), source.numel()); break;
        case DType::Int64:   scatter_values(output.data<int64_t>(), source.data<int64_t>(), source.numel()); break;
        case DType::Int16:   scatter_values(output.data<int16_t>(), source.data<int16_t>(), source.numel()); break;
        case DType::Int8:    scatter_values(output.data<int8_t>(), source.data<int8_t>(), source.numel()); break;
        case DType::UInt8:   scatter_values(output.data<uint8_t>(), source.data<uint8_t>(), source.numel()); break;
        default:
            throw std::runtime_error("masked_scatter: unsupported dtype");
    }

    return output;
}

// =============================================================================
// TrilIndices kernel
// =============================================================================

auto tril_indices_kernel(int64_t row, int64_t col, int64_t offset) -> Tensor {
    // Count lower-triangular elements
    std::vector<int64_t> row_indices;
    std::vector<int64_t> col_indices;
    row_indices.reserve(row * col);
    col_indices.reserve(row * col);

    for (int64_t r = 0; r < row; ++r) {
        int64_t max_c = std::min(col, r + offset + 1);
        for (int64_t c = 0; c < max_c; ++c) {
            row_indices.push_back(r);
            col_indices.push_back(c);
        }
    }

    int64_t n = static_cast<int64_t>(row_indices.size());
    if (n == 0) {
        return tenzor::empty({2, 0}, DType::Int64, Device::cpu());
    }

    // Build (2, N) output
    Tensor output({2, n}, DType::Int64, Device::cpu());
    int64_t* out = output.data<int64_t>();
    std::memcpy(out, row_indices.data(), n * sizeof(int64_t));
    std::memcpy(out + n, col_indices.data(), n * sizeof(int64_t));

    return output;
}

// =============================================================================
// TriuIndices kernel
// =============================================================================

auto triu_indices_kernel(int64_t row, int64_t col, int64_t offset) -> Tensor {
    std::vector<int64_t> row_indices;
    std::vector<int64_t> col_indices;
    row_indices.reserve(row * col);
    col_indices.reserve(row * col);

    for (int64_t r = 0; r < row; ++r) {
        int64_t min_c = std::max(static_cast<int64_t>(0), r + offset);
        for (int64_t c = min_c; c < col; ++c) {
            row_indices.push_back(r);
            col_indices.push_back(c);
        }
    }

    int64_t n = static_cast<int64_t>(row_indices.size());
    if (n == 0) {
        return tenzor::empty({2, 0}, DType::Int64, Device::cpu());
    }

    Tensor output({2, n}, DType::Int64, Device::cpu());
    int64_t* out = output.data<int64_t>();
    std::memcpy(out, row_indices.data(), n * sizeof(int64_t));
    std::memcpy(out + n, col_indices.data(), n * sizeof(int64_t));

    return output;
}

} // namespace cpu
} // namespace tenzor
