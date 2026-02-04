/**
 * @file indexing.cpp
 * @brief CPU indexing operation kernels
 */

#include "tenzor/core/tensor.hpp"
#include "tenzor/core/dtype.hpp"
#include "tenzor/core/shape.hpp"
#include "simd_fast_math.hpp"
#include <cstring>
#include <stdexcept>

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

    // Size of one element along the selected dimension
    const int64_t elem_size = inner_size * dtype_size(input.dtype());

    // Perform the selection based on dtype
    if (input.dtype() == DType::Float32) {
        const float* in_data = input.data<float>();
        float* out_data = output.data<float>();

        for (int64_t outer = 0; outer < outer_size; ++outer) {
            for (int64_t idx = 0; idx < num_indices; ++idx) {
                int64_t src_idx = index_data[idx];

                // Handle negative indices
                if (src_idx < 0) {
                    src_idx += in_shape[dim];
                }

                // Validate index
                if (src_idx < 0 || src_idx >= in_shape[dim]) {
                    std::string error_msg = "index_select: index out of range. ";
                    error_msg += "Index: " + std::to_string(src_idx) + ", ";
                    error_msg += "Valid range: [0, " + std::to_string(in_shape[dim]) + "), ";
                    error_msg += "Input shape: [";
                    for (size_t i = 0; i < in_shape.size(); ++i) {
                        if (i > 0) error_msg += ", ";
                        error_msg += std::to_string(in_shape[i]);
                    }
                    error_msg += "], dim: " + std::to_string(dim);
                    throw std::out_of_range(error_msg);
                }

                // Copy elements
                for (int64_t inner = 0; inner < inner_size; ++inner) {
                    int64_t in_offset = outer * in_strides[dim] * in_shape[dim] +
                                       src_idx * in_strides[dim] + inner;
                    int64_t out_offset = outer * out_strides[dim] * num_indices +
                                        idx * out_strides[dim] + inner;
                    out_data[out_offset] = in_data[in_offset];
                }
            }
        }
    } else if (input.dtype() == DType::Float64) {
        const double* in_data = input.data<double>();
        double* out_data = output.data<double>();

        for (int64_t outer = 0; outer < outer_size; ++outer) {
            for (int64_t idx = 0; idx < num_indices; ++idx) {
                int64_t src_idx = index_data[idx];
                if (src_idx < 0) src_idx += in_shape[dim];
                if (src_idx < 0 || src_idx >= in_shape[dim]) {
                    throw std::out_of_range("index_select: index out of range");
                }

                for (int64_t inner = 0; inner < inner_size; ++inner) {
                    int64_t in_offset = outer * in_strides[dim] * in_shape[dim] +
                                       src_idx * in_strides[dim] + inner;
                    int64_t out_offset = outer * out_strides[dim] * num_indices +
                                        idx * out_strides[dim] + inner;
                    out_data[out_offset] = in_data[in_offset];
                }
            }
        }
    } else if (input.dtype() == DType::Int32) {
        const int32_t* in_data = input.data<int32_t>();
        int32_t* out_data = output.data<int32_t>();

        for (int64_t outer = 0; outer < outer_size; ++outer) {
            for (int64_t idx = 0; idx < num_indices; ++idx) {
                int64_t src_idx = index_data[idx];
                if (src_idx < 0) src_idx += in_shape[dim];
                if (src_idx < 0 || src_idx >= in_shape[dim]) {
                    throw std::out_of_range("index_select: index out of range");
                }

                for (int64_t inner = 0; inner < inner_size; ++inner) {
                    int64_t in_offset = outer * in_strides[dim] * in_shape[dim] +
                                       src_idx * in_strides[dim] + inner;
                    int64_t out_offset = outer * out_strides[dim] * num_indices +
                                        idx * out_strides[dim] + inner;
                    out_data[out_offset] = in_data[in_offset];
                }
            }
        }
    } else if (input.dtype() == DType::Int64) {
        const int64_t* in_data = input.data<int64_t>();
        int64_t* out_data = output.data<int64_t>();

        for (int64_t outer = 0; outer < outer_size; ++outer) {
            for (int64_t idx = 0; idx < num_indices; ++idx) {
                int64_t src_idx = index_data[idx];
                if (src_idx < 0) src_idx += in_shape[dim];
                if (src_idx < 0 || src_idx >= in_shape[dim]) {
                    throw std::out_of_range("index_select: index out of range");
                }

                for (int64_t inner = 0; inner < inner_size; ++inner) {
                    int64_t in_offset = outer * in_strides[dim] * in_shape[dim] +
                                       src_idx * in_strides[dim] + inner;
                    int64_t out_offset = outer * out_strides[dim] * num_indices +
                                        idx * out_strides[dim] + inner;
                    out_data[out_offset] = in_data[in_offset];
                }
            }
        }
    } else if (input.dtype() == DType::Bool) {
        const bool* in_data = input.data<bool>();
        bool* out_data = output.data<bool>();

        for (int64_t outer = 0; outer < outer_size; ++outer) {
            for (int64_t idx = 0; idx < num_indices; ++idx) {
                int64_t src_idx = index_data[idx];
                if (src_idx < 0) src_idx += in_shape[dim];
                if (src_idx < 0 || src_idx >= in_shape[dim]) {
                    throw std::out_of_range("index_select: index out of range");
                }

                for (int64_t inner = 0; inner < inner_size; ++inner) {
                    int64_t in_offset = outer * in_strides[dim] * in_shape[dim] +
                                       src_idx * in_strides[dim] + inner;
                    int64_t out_offset = outer * out_strides[dim] * num_indices +
                                        idx * out_strides[dim] + inner;
                    out_data[out_offset] = in_data[in_offset];
                }
            }
        }
    } else if (input.dtype() == DType::Float16) {
        const Float16* in_data = input.data<Float16>();
        Float16* out_data = output.data<Float16>();

        for (int64_t outer = 0; outer < outer_size; ++outer) {
            for (int64_t idx = 0; idx < num_indices; ++idx) {
                int64_t src_idx = index_data[idx];
                if (src_idx < 0) src_idx += in_shape[dim];
                if (src_idx < 0 || src_idx >= in_shape[dim]) {
                    throw std::out_of_range("index_select: index out of range");
                }

                for (int64_t inner = 0; inner < inner_size; ++inner) {
                    int64_t in_offset = outer * in_strides[dim] * in_shape[dim] +
                                       src_idx * in_strides[dim] + inner;
                    int64_t out_offset = outer * out_strides[dim] * num_indices +
                                        idx * out_strides[dim] + inner;
                    out_data[out_offset] = in_data[in_offset];
                }
            }
        }
    } else {
        throw std::runtime_error("index_select: unsupported dtype");
    }

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

    auto input_shape_span = input.shape();
    auto index_shape_span = index.shape();

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
    Tensor output(index_shape, input.dtype(), input.device());

    auto input_strides = calculate_strides(input_shape);
    auto index_strides = calculate_strides(index_shape);
    const int64_t numel = output.numel();
    const size_t ndims = index_shape.size();

    const int64_t* index_ptr = index.data<int64_t>();

    if (input.dtype() == DType::Float32) {
        const float* input_ptr = input.data<float>();
        float* output_ptr = output.data<float>();

        for (int64_t flat_idx = 0; flat_idx < numel; ++flat_idx) {
            // Compute multi-dimensional index in output
            int64_t temp = flat_idx;
            int64_t input_idx = 0;

            for (size_t d = 0; d < ndims; ++d) {
                int64_t coord = temp / index_strides[d];
                temp %= index_strides[d];

                if (static_cast<int64_t>(d) == dim) {
                    // Use index tensor to determine coordinate
                    int64_t idx_val = index_ptr[flat_idx];
                    if (idx_val < 0) idx_val += input_shape[d];
                    input_idx += idx_val * input_strides[d];
                } else {
                    input_idx += coord * input_strides[d];
                }
            }

            output_ptr[flat_idx] = input_ptr[input_idx];
        }
    } else if (input.dtype() == DType::Float64) {
        const double* input_ptr = input.data<double>();
        double* output_ptr = output.data<double>();

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
    } else if (input.dtype() == DType::Int32) {
        const int32_t* input_ptr = input.data<int32_t>();
        int32_t* output_ptr = output.data<int32_t>();

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
    } else if (input.dtype() == DType::Int64) {
        const int64_t* input_ptr = input.data<int64_t>();
        int64_t* output_ptr = output.data<int64_t>();

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
    } else if (input.dtype() == DType::Bool) {
        const bool* input_ptr = input.data<bool>();
        bool* output_ptr = output.data<bool>();

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
    } else {
        throw std::runtime_error("gather: unsupported dtype");
    }

    return output;
}

// Scatter operation - distribute values at specified indices along a dimension
auto scatter_kernel(const Tensor& input, int64_t dim, const Tensor& index, const Tensor& src) -> Tensor {
    // Validate index tensor dtype
    if (index.dtype() != DType::Int64) {
        throw std::invalid_argument("scatter: index tensor must have dtype Int64");
    }

    auto input_shape_span = input.shape();
    auto index_shape_span = index.shape();
    auto src_shape_span = src.shape();

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
    Tensor output(input_shape, input.dtype(), input.device());

    auto input_strides = calculate_strides(input_shape);
    auto index_strides = calculate_strides(index_shape);
    const int64_t numel = index.numel();
    const size_t ndims = index_shape.size();

    const int64_t* index_ptr = index.data<int64_t>();

    if (input.dtype() == DType::Float32) {
        const float* input_ptr = input.data<float>();
        float* output_ptr = output.data<float>();
        const float* src_ptr = src.data<float>();

        // Copy input to output first
        std::memcpy(output_ptr, input_ptr, input.numel() * sizeof(float));

        // Scatter src values into output
        for (int64_t flat_idx = 0; flat_idx < numel; ++flat_idx) {
            int64_t temp = flat_idx;
            int64_t output_idx = 0;

            for (size_t d = 0; d < ndims; ++d) {
                int64_t coord = temp / index_strides[d];
                temp %= index_strides[d];

                if (static_cast<int64_t>(d) == dim) {
                    int64_t idx_val = index_ptr[flat_idx];
                    if (idx_val < 0) idx_val += input_shape[d];
                    output_idx += idx_val * input_strides[d];
                } else {
                    output_idx += coord * input_strides[d];
                }
            }

            output_ptr[output_idx] = src_ptr[flat_idx];
        }
    } else if (input.dtype() == DType::Float64) {
        const double* input_ptr = input.data<double>();
        double* output_ptr = output.data<double>();
        const double* src_ptr = src.data<double>();

        std::memcpy(output_ptr, input_ptr, input.numel() * sizeof(double));

        for (int64_t flat_idx = 0; flat_idx < numel; ++flat_idx) {
            int64_t temp = flat_idx;
            int64_t output_idx = 0;

            for (size_t d = 0; d < ndims; ++d) {
                int64_t coord = temp / index_strides[d];
                temp %= index_strides[d];

                if (static_cast<int64_t>(d) == dim) {
                    int64_t idx_val = index_ptr[flat_idx];
                    if (idx_val < 0) idx_val += input_shape[d];
                    output_idx += idx_val * input_strides[d];
                } else {
                    output_idx += coord * input_strides[d];
                }
            }

            output_ptr[output_idx] = src_ptr[flat_idx];
        }
    } else if (input.dtype() == DType::Int32) {
        const int32_t* input_ptr = input.data<int32_t>();
        int32_t* output_ptr = output.data<int32_t>();
        const int32_t* src_ptr = src.data<int32_t>();

        std::memcpy(output_ptr, input_ptr, input.numel() * sizeof(int32_t));

        for (int64_t flat_idx = 0; flat_idx < numel; ++flat_idx) {
            int64_t temp = flat_idx;
            int64_t output_idx = 0;

            for (size_t d = 0; d < ndims; ++d) {
                int64_t coord = temp / index_strides[d];
                temp %= index_strides[d];

                if (static_cast<int64_t>(d) == dim) {
                    int64_t idx_val = index_ptr[flat_idx];
                    if (idx_val < 0) idx_val += input_shape[d];
                    output_idx += idx_val * input_strides[d];
                } else {
                    output_idx += coord * input_strides[d];
                }
            }

            output_ptr[output_idx] = src_ptr[flat_idx];
        }
    } else if (input.dtype() == DType::Int64) {
        const int64_t* input_ptr = input.data<int64_t>();
        int64_t* output_ptr = output.data<int64_t>();
        const int64_t* src_ptr = src.data<int64_t>();

        std::memcpy(output_ptr, input_ptr, input.numel() * sizeof(int64_t));

        for (int64_t flat_idx = 0; flat_idx < numel; ++flat_idx) {
            int64_t temp = flat_idx;
            int64_t output_idx = 0;

            for (size_t d = 0; d < ndims; ++d) {
                int64_t coord = temp / index_strides[d];
                temp %= index_strides[d];

                if (static_cast<int64_t>(d) == dim) {
                    int64_t idx_val = index_ptr[flat_idx];
                    if (idx_val < 0) idx_val += input_shape[d];
                    output_idx += idx_val * input_strides[d];
                } else {
                    output_idx += coord * input_strides[d];
                }
            }

            output_ptr[output_idx] = src_ptr[flat_idx];
        }
    } else if (input.dtype() == DType::Bool) {
        const bool* input_ptr = input.data<bool>();
        bool* output_ptr = output.data<bool>();
        const bool* src_ptr = src.data<bool>();

        std::memcpy(output_ptr, input_ptr, input.numel() * sizeof(bool));

        for (int64_t flat_idx = 0; flat_idx < numel; ++flat_idx) {
            int64_t temp = flat_idx;
            int64_t output_idx = 0;

            for (size_t d = 0; d < ndims; ++d) {
                int64_t coord = temp / index_strides[d];
                temp %= index_strides[d];

                if (static_cast<int64_t>(d) == dim) {
                    int64_t idx_val = index_ptr[flat_idx];
                    if (idx_val < 0) idx_val += input_shape[d];
                    output_idx += idx_val * input_strides[d];
                } else {
                    output_idx += coord * input_strides[d];
                }
            }

            output_ptr[output_idx] = src_ptr[flat_idx];
        }
    } else if (input.dtype() == DType::Float16) {
        // Convert to Float32 for computation
        auto input_f32 = input.to(DType::Float32);
        auto src_f32 = src.to(DType::Float32);
        Tensor output_f32(input_shape, DType::Float32, input.device());

        const float* input_ptr = input_f32.data<float>();
        float* output_ptr = output_f32.data<float>();
        const float* src_ptr = src_f32.data<float>();

        std::memcpy(output_ptr, input_ptr, input_f32.numel() * sizeof(float));

        for (int64_t flat_idx = 0; flat_idx < numel; ++flat_idx) {
            int64_t temp = flat_idx;
            int64_t output_idx = 0;

            for (size_t d = 0; d < ndims; ++d) {
                int64_t coord = temp / index_strides[d];
                temp %= index_strides[d];

                if (static_cast<int64_t>(d) == dim) {
                    int64_t idx_val = index_ptr[flat_idx];
                    if (idx_val < 0) idx_val += input_shape[d];
                    output_idx += idx_val * input_strides[d];
                } else {
                    output_idx += coord * input_strides[d];
                }
            }

            output_ptr[output_idx] = src_ptr[flat_idx];
        }

        // Convert back to Float16
        output = output_f32.to(DType::Float16);
    } else {
        throw std::runtime_error("scatter: unsupported dtype");
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
    if (input.dtype() == DType::Float32) {
        const float* input_ptr = input.data<float>();
        float* output_ptr = output.data<float>();

        int64_t out_idx = 0;
        for (int64_t i = 0; i < numel; ++i) {
            if (is_mask_true(i)) {
                output_ptr[out_idx++] = input_ptr[i];
            }
        }
    } else if (input.dtype() == DType::Float64) {
        const double* input_ptr = input.data<double>();
        double* output_ptr = output.data<double>();

        int64_t out_idx = 0;
        for (int64_t i = 0; i < numel; ++i) {
            if (is_mask_true(i)) {
                output_ptr[out_idx++] = input_ptr[i];
            }
        }
    } else if (input.dtype() == DType::Int32) {
        const int32_t* input_ptr = input.data<int32_t>();
        int32_t* output_ptr = output.data<int32_t>();

        int64_t out_idx = 0;
        for (int64_t i = 0; i < numel; ++i) {
            if (is_mask_true(i)) {
                output_ptr[out_idx++] = input_ptr[i];
            }
        }
    } else if (input.dtype() == DType::Int64) {
        const int64_t* input_ptr = input.data<int64_t>();
        int64_t* output_ptr = output.data<int64_t>();

        int64_t out_idx = 0;
        for (int64_t i = 0; i < numel; ++i) {
            if (is_mask_true(i)) {
                output_ptr[out_idx++] = input_ptr[i];
            }
        }
    } else if (input.dtype() == DType::Bool) {
        const bool* input_ptr = input.data<bool>();
        bool* output_ptr = output.data<bool>();

        int64_t out_idx = 0;
        for (int64_t i = 0; i < numel; ++i) {
            if (is_mask_true(i)) {
                output_ptr[out_idx++] = input_ptr[i];
            }
        }
    } else {
        throw std::runtime_error("masked_select: unsupported dtype");
    }

    return output;
}

// Masked fill operation - fill elements with value where mask is true
auto masked_fill_kernel(const Tensor& input, const Tensor& mask, float value) -> Tensor {
    auto input_shape = input.shape();
    auto mask_shape = mask.shape();

    // Validate shapes match
    if (!std::equal(input_shape.begin(), input_shape.end(), mask_shape.begin(), mask_shape.end())) {
        throw std::invalid_argument("masked_fill: input and mask must have same shape");
    }

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
    } else if (x.dtype() == DType::Float64) {
        const double* x_ptr = x.data<double>();
        const double* y_ptr = y.data<double>();
        double* output_ptr = output.data<double>();

        for (int64_t i = 0; i < numel; ++i) {
            output_ptr[i] = is_cond_true(i) ? x_ptr[i] : y_ptr[i];
        }
    } else if (x.dtype() == DType::Int32) {
        const int32_t* x_ptr = x.data<int32_t>();
        const int32_t* y_ptr = y.data<int32_t>();
        int32_t* output_ptr = output.data<int32_t>();

        for (int64_t i = 0; i < numel; ++i) {
            output_ptr[i] = is_cond_true(i) ? x_ptr[i] : y_ptr[i];
        }
    } else if (x.dtype() == DType::Int64) {
        const int64_t* x_ptr = x.data<int64_t>();
        const int64_t* y_ptr = y.data<int64_t>();
        int64_t* output_ptr = output.data<int64_t>();

        for (int64_t i = 0; i < numel; ++i) {
            output_ptr[i] = is_cond_true(i) ? x_ptr[i] : y_ptr[i];
        }
    } else if (x.dtype() == DType::Bool) {
        const bool* x_ptr = x.data<bool>();
        const bool* y_ptr = y.data<bool>();
        bool* output_ptr = output.data<bool>();

        for (int64_t i = 0; i < numel; ++i) {
            output_ptr[i] = is_cond_true(i) ? x_ptr[i] : y_ptr[i];
        }
    } else {
        throw std::runtime_error("where: unsupported dtype");
    }

    return output;
}

auto nonzero_kernel(const Tensor& input) -> Tensor {
    const int64_t numel = input.numel();
    const int64_t ndim = input.ndim();

    // First pass: count nonzero elements
    std::vector<int64_t> nz_indices;
    nz_indices.reserve(numel / 4);  // Heuristic

    if (input.dtype() == DType::Float32) {
        const float* data = input.data<float>();
        for (int64_t i = 0; i < numel; ++i) {
            if (data[i] != 0.0f) nz_indices.push_back(i);
        }
    } else if (input.dtype() == DType::Float64) {
        const double* data = input.data<double>();
        for (int64_t i = 0; i < numel; ++i) {
            if (data[i] != 0.0) nz_indices.push_back(i);
        }
    } else if (input.dtype() == DType::Int32) {
        const int32_t* data = input.data<int32_t>();
        for (int64_t i = 0; i < numel; ++i) {
            if (data[i] != 0) nz_indices.push_back(i);
        }
    } else if (input.dtype() == DType::Int64) {
        const int64_t* data = input.data<int64_t>();
        for (int64_t i = 0; i < numel; ++i) {
            if (data[i] != 0) nz_indices.push_back(i);
        }
    } else if (input.dtype() == DType::Bool) {
        const bool* data = input.data<bool>();
        for (int64_t i = 0; i < numel; ++i) {
            if (data[i]) nz_indices.push_back(i);
        }
    } else {
        throw std::runtime_error("nonzero: unsupported dtype");
    }

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

    #pragma omp parallel for if(num_indices > 65536)
    for (int64_t i = 0; i < num_indices; ++i) {
        int64_t idx = idx_data[i];
        if (idx < 0) idx += input_numel;
        if (idx < 0 || idx >= input_numel) {
            throw std::out_of_range("take: index out of range");
        }
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

} // namespace cpu
} // namespace tenzor
