/**
 * @file indexing.cpp
 * @brief CPU indexing operation kernels
 */

#include "tenzor/core/tensor.hpp"
#include "tenzor/core/dtype.hpp"
#include "tenzor/core/shape.hpp"
#include <cstring>
#include <stdexcept>

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
    if (num_indices == 0) {
        throw std::invalid_argument("index_select: index tensor cannot be empty");
    }

    // Compute output shape
    std::vector<int64_t> output_shape(input.shape().begin(), input.shape().end());
    output_shape[dim] = num_indices;

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
    } else {
        throw std::runtime_error("index_select: unsupported dtype");
    }

    return output;
}

} // namespace cpu
} // namespace tenzor
