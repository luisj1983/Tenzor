#include "tenzor/core/tensor.hpp"
#include "../sycl_prefix_sum.hpp"
#include <sycl/sycl.hpp>
#include <stdexcept>
#include <algorithm>
#include <array>
#include <numeric>
#include <vector>

#ifdef TENZOR_HAS_ONEDPL
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/iterator>
#endif

namespace tenzor {
namespace oneapi {

// Kernel class declarations for SYCL
class GatherKernelFloat32;
class GatherKernelFloat64;
class GatherKernelFloat16;
class ScatterKernelFloat32;
class ScatterKernelFloat64;
class ScatterKernelFloat16;
class IndexSelectKernelFloat32;
class IndexSelectKernelFloat64;
class IndexSelectKernelFloat16;
class IndexSelectKernelInt32;
class IndexSelectKernelInt64;
class IndexSelectKernelBool;
class MaskedFillKernelFloat32;
class MaskedFillKernelFloat64;
class MaskedFillKernelFloat16;
class GatherKernelBFloat16;
class ScatterKernelBFloat16;
class IndexSelectKernelBFloat16;
class MaskedFillKernelBFloat16;

// Kernel names for device-side masked_select and nonzero
class MaskedSelectPrefixSumUpSweep;
class MaskedSelectPrefixSumDownSweep;
class MaskedSelectScatterFloat32;
class MaskedSelectScatterFloat64;
class MaskedSelectScatterFloat16;
class MaskedSelectScatterBFloat16;
class NonzeroBinaryMaskFloat32;
class NonzeroBinaryMaskFloat64;
class NonzeroBinaryMaskFloat16;
class NonzeroBinaryMaskBFloat16;
class NonzeroBinaryMaskInt32;
class NonzeroBinaryMaskInt64;
class NonzeroBinaryMaskBool;
class NonzeroScatterIndices;

// BFloat16 conversion helpers
inline float bf16_to_f32(uint16_t bf16) {
    uint32_t bits = static_cast<uint32_t>(bf16) << 16;
    float result;
    __builtin_memcpy(&result, &bits, sizeof(float));
    return result;
}
inline uint16_t f32_to_bf16(float f32) {
    uint32_t bits;
    __builtin_memcpy(&bits, &f32, sizeof(uint32_t));
    // Round to nearest even (banker's rounding) for BFloat16
    uint32_t lsb = (bits >> 16) & 1;
    uint32_t rounding_bias = 0x7FFF + lsb;
    bits += rounding_bias;
    return static_cast<uint16_t>(bits >> 16);
}

// Helper function to get typed pointer from tensor
template<typename T>
inline auto get_data_ptr(const Tensor& t) -> T* {
    return static_cast<T*>(const_cast<void*>(t.data_ptr()));
}

// Helper to calculate strides from shape
inline auto calculate_strides(const std::vector<int64_t>& shape) -> std::vector<int64_t> {
    std::vector<int64_t> strides(shape.size());
    int64_t stride = 1;
    for (int64_t i = shape.size() - 1; i >= 0; --i) {
        strides[i] = stride;
        stride *= shape[i];
    }
    return strides;
}

// Gather operation - collect values at specified indices along a dimension
auto gather_kernel(const Tensor& input, int64_t dim, const Tensor& index, sycl::queue& queue) -> Tensor {
    auto input_shape_span = input.shape();
    auto index_shape_span = index.shape();

    // Convert spans to vectors
    std::vector<int64_t> input_shape(input_shape_span.begin(), input_shape_span.end());
    std::vector<int64_t> index_shape(index_shape_span.begin(), index_shape_span.end());

    // Validate dimension
    if (dim < 0) dim += input_shape.size();
    if (dim < 0 || dim >= static_cast<int64_t>(input_shape.size())) {
        throw std::invalid_argument("Gather: invalid dimension");
    }

    // Output has same shape as index
    Tensor output(index_shape, input.dtype(), input.device());

    auto input_strides = calculate_strides(input_shape);
    auto index_strides = calculate_strides(index_shape);

    const int64_t numel = output.numel();

    // Copy vectors to arrays for device copyability (max 8 dimensions)
    int64_t input_shape_arr[8], input_strides_arr[8], index_strides_arr[8];
    const size_t ndims = index_shape.size();
    for (size_t i = 0; i < ndims && i < 8; ++i) {
        input_shape_arr[i] = input_shape[i];
        input_strides_arr[i] = input_strides[i];
        index_strides_arr[i] = index_strides[i];
    }

    if (input.dtype() == DType::Float32) {
        const float* input_ptr = get_data_ptr<const float>(input);
        const int64_t* index_ptr = get_data_ptr<const int64_t>(index);
        float* output_ptr = get_data_ptr<float>(output);

        queue.parallel_for<GatherKernelFloat32>(sycl::range<1>(numel), [=](sycl::id<1> flat_idx) {
            // Compute multi-dimensional index in output
            int64_t temp = flat_idx;
            int64_t input_idx = 0;

            for (size_t d = 0; d < ndims; ++d) {
                int64_t coord = temp / index_strides_arr[d];
                temp %= index_strides_arr[d];

                if (static_cast<int64_t>(d) == dim) {
                    // Use index tensor to determine coordinate
                    int64_t idx_val = index_ptr[flat_idx];
                    if (idx_val < 0) idx_val += input_shape_arr[d];
                    input_idx += idx_val * input_strides_arr[d];
                } else {
                    input_idx += coord * input_strides_arr[d];
                }
            }

            output_ptr[flat_idx] = input_ptr[input_idx];
        });
    }
    else if (input.dtype() == DType::Float64) {
        const double* input_ptr = get_data_ptr<const double>(input);
        const int64_t* index_ptr = get_data_ptr<const int64_t>(index);
        double* output_ptr = get_data_ptr<double>(output);

        queue.parallel_for<GatherKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> flat_idx) {
            int64_t temp = flat_idx;
            int64_t input_idx = 0;

            for (size_t d = 0; d < ndims; ++d) {
                int64_t coord = temp / index_strides_arr[d];
                temp %= index_strides_arr[d];

                if (static_cast<int64_t>(d) == dim) {
                    int64_t idx_val = index_ptr[flat_idx];
                    if (idx_val < 0) idx_val += input_shape_arr[d];
                    input_idx += idx_val * input_strides_arr[d];
                } else {
                    input_idx += coord * input_strides_arr[d];
                }
            }

            output_ptr[flat_idx] = input_ptr[input_idx];
        });
    }
    else if (input.dtype() == DType::Float16) {
        const sycl::half* input_ptr = get_data_ptr<const sycl::half>(input);
        const int64_t* index_ptr = get_data_ptr<const int64_t>(index);
        sycl::half* output_ptr = get_data_ptr<sycl::half>(output);

        queue.parallel_for<GatherKernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> flat_idx) {
            int64_t temp = flat_idx;
            int64_t input_idx = 0;

            for (size_t d = 0; d < ndims; ++d) {
                int64_t coord = temp / index_strides_arr[d];
                temp %= index_strides_arr[d];

                if (static_cast<int64_t>(d) == dim) {
                    int64_t idx_val = index_ptr[flat_idx];
                    if (idx_val < 0) idx_val += input_shape_arr[d];
                    input_idx += idx_val * input_strides_arr[d];
                } else {
                    input_idx += coord * input_strides_arr[d];
                }
            }

            output_ptr[flat_idx] = input_ptr[input_idx];
        });
    }
    else if (input.dtype() == DType::BFloat16) {
        const uint16_t* input_ptr = get_data_ptr<const uint16_t>(input);
        const int64_t* index_ptr = get_data_ptr<const int64_t>(index);
        uint16_t* output_ptr = get_data_ptr<uint16_t>(output);

        queue.parallel_for<GatherKernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> flat_idx) {
            int64_t temp = flat_idx;
            int64_t input_idx = 0;

            for (size_t d = 0; d < ndims; ++d) {
                int64_t coord = temp / index_strides_arr[d];
                temp %= index_strides_arr[d];

                if (static_cast<int64_t>(d) == dim) {
                    int64_t idx_val = index_ptr[flat_idx];
                    if (idx_val < 0) idx_val += input_shape_arr[d];
                    input_idx += idx_val * input_strides_arr[d];
                } else {
                    input_idx += coord * input_strides_arr[d];
                }
            }

            output_ptr[flat_idx] = input_ptr[input_idx];
        });
    }
    else {
        throw std::runtime_error("Unsupported dtype for gather");
    }

    return output;
}

// Scatter operation - distribute values at specified indices along a dimension
auto scatter_kernel(const Tensor& input, int64_t dim, const Tensor& index,
                    const Tensor& src, sycl::queue& queue) -> Tensor {
    auto input_shape_span = input.shape();
    std::vector<int64_t> input_shape(input_shape_span.begin(), input_shape_span.end());

    // Validate dimension
    if (dim < 0) dim += input_shape.size();
    if (dim < 0 || dim >= static_cast<int64_t>(input_shape.size())) {
        throw std::invalid_argument("Scatter: invalid dimension");
    }

    // Create output as copy of input
    Tensor output(input_shape, input.dtype(), input.device());

    // Copy input to output first
    const size_t bytes = input.numel() * input.dtype_size();
    if (input.dtype() == DType::Float32) {
        const float* in_ptr = get_data_ptr<const float>(input);
        float* out_ptr = get_data_ptr<float>(output);
        queue.memcpy(out_ptr, in_ptr, bytes);
    } else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        double* out_ptr = get_data_ptr<double>(output);
        queue.memcpy(out_ptr, in_ptr, bytes);
    } else if (input.dtype() == DType::Float16) {
        const sycl::half* in_ptr = get_data_ptr<const sycl::half>(input);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);
        queue.memcpy(out_ptr, in_ptr, bytes);
    } else if (input.dtype() == DType::BFloat16) {
        const uint16_t* in_ptr = get_data_ptr<const uint16_t>(input);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);
        queue.memcpy(out_ptr, in_ptr, bytes);
    }

    auto index_shape_span = index.shape();
    std::vector<int64_t> index_shape(index_shape_span.begin(), index_shape_span.end());
    auto input_strides = calculate_strides(input_shape);
    auto index_strides = calculate_strides(index_shape);

    const int64_t numel = index.numel();

    // Copy vectors to arrays for device copyability (max 8 dimensions)
    int64_t input_shape_arr[8], input_strides_arr[8], index_strides_arr[8];
    const size_t ndims = index_shape.size();
    for (size_t i = 0; i < ndims && i < 8; ++i) {
        input_shape_arr[i] = input_shape[i];
        input_strides_arr[i] = input_strides[i];
        index_strides_arr[i] = index_strides[i];
    }

    if (input.dtype() == DType::Float32) {
        float* output_ptr = get_data_ptr<float>(output);
        const int64_t* index_ptr = get_data_ptr<const int64_t>(index);
        const float* src_ptr = get_data_ptr<const float>(src);

        queue.parallel_for<ScatterKernelFloat32>(sycl::range<1>(numel), [=](sycl::id<1> flat_idx) {
            // Compute multi-dimensional index
            int64_t temp = flat_idx;
            int64_t output_idx = 0;

            for (size_t d = 0; d < ndims; ++d) {
                int64_t coord = temp / index_strides_arr[d];
                temp %= index_strides_arr[d];

                if (static_cast<int64_t>(d) == dim) {
                    int64_t idx_val = index_ptr[flat_idx];
                    if (idx_val < 0) idx_val += input_shape_arr[d];
                    output_idx += idx_val * input_strides_arr[d];
                } else {
                    output_idx += coord * input_strides_arr[d];
                }
            }

            // Atomic write to avoid race conditions
            output_ptr[output_idx] = src_ptr[flat_idx];
        });
    }
    else if (input.dtype() == DType::Float64) {
        double* output_ptr = get_data_ptr<double>(output);
        const int64_t* index_ptr = get_data_ptr<const int64_t>(index);
        const double* src_ptr = get_data_ptr<const double>(src);

        queue.parallel_for<ScatterKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> flat_idx) {
            int64_t temp = flat_idx;
            int64_t output_idx = 0;

            for (size_t d = 0; d < ndims; ++d) {
                int64_t coord = temp / index_strides_arr[d];
                temp %= index_strides_arr[d];

                if (static_cast<int64_t>(d) == dim) {
                    int64_t idx_val = index_ptr[flat_idx];
                    if (idx_val < 0) idx_val += input_shape_arr[d];
                    output_idx += idx_val * input_strides_arr[d];
                } else {
                    output_idx += coord * input_strides_arr[d];
                }
            }

            output_ptr[output_idx] = src_ptr[flat_idx];
        });
    }
    else if (input.dtype() == DType::Float16) {
        sycl::half* output_ptr = get_data_ptr<sycl::half>(output);
        const int64_t* index_ptr = get_data_ptr<const int64_t>(index);
        const sycl::half* src_ptr = get_data_ptr<const sycl::half>(src);

        queue.parallel_for<ScatterKernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> flat_idx) {
            int64_t temp = flat_idx;
            int64_t output_idx = 0;

            for (size_t d = 0; d < ndims; ++d) {
                int64_t coord = temp / index_strides_arr[d];
                temp %= index_strides_arr[d];

                if (static_cast<int64_t>(d) == dim) {
                    int64_t idx_val = index_ptr[flat_idx];
                    if (idx_val < 0) idx_val += input_shape_arr[d];
                    output_idx += idx_val * input_strides_arr[d];
                } else {
                    output_idx += coord * input_strides_arr[d];
                }
            }

            output_ptr[output_idx] = src_ptr[flat_idx];
        });
    }
    else if (input.dtype() == DType::BFloat16) {
        uint16_t* output_ptr = get_data_ptr<uint16_t>(output);
        const int64_t* index_ptr = get_data_ptr<const int64_t>(index);
        const uint16_t* src_ptr = get_data_ptr<const uint16_t>(src);

        queue.parallel_for<ScatterKernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> flat_idx) {
            int64_t temp = flat_idx;
            int64_t output_idx = 0;

            for (size_t d = 0; d < ndims; ++d) {
                int64_t coord = temp / index_strides_arr[d];
                temp %= index_strides_arr[d];

                if (static_cast<int64_t>(d) == dim) {
                    int64_t idx_val = index_ptr[flat_idx];
                    if (idx_val < 0) idx_val += input_shape_arr[d];
                    output_idx += idx_val * input_strides_arr[d];
                } else {
                    output_idx += coord * input_strides_arr[d];
                }
            }

            output_ptr[output_idx] = src_ptr[flat_idx];
        });
    }
    else {
        throw std::runtime_error("Unsupported dtype for scatter");
    }

    return output;
}

// Index select - select elements along dimension using 1D index tensor
auto index_select_kernel(const Tensor& input, int64_t dim, const Tensor& index, sycl::queue& queue) -> Tensor {
    auto input_shape_span = input.shape();
    std::vector<int64_t> input_shape(input_shape_span.begin(), input_shape_span.end());

    // Validate dimension
    if (dim < 0) dim += input_shape.size();
    if (dim < 0 || dim >= static_cast<int64_t>(input_shape.size())) {
        throw std::invalid_argument("IndexSelect: invalid dimension");
    }

    // Output shape: replace dimension size with index size
    std::vector<int64_t> output_shape = input_shape;
    output_shape[dim] = index.numel();

    Tensor output(output_shape, input.dtype(), input.device());

    auto input_strides = calculate_strides(input_shape);
    auto output_strides = calculate_strides(output_shape);

    const int64_t numel = output.numel();

    // Copy vectors to arrays for device copyability (max 8 dimensions)
    int64_t input_shape_arr[8], input_strides_arr[8], output_strides_arr[8];
    const size_t ndims = output_shape.size();
    for (size_t i = 0; i < ndims && i < 8; ++i) {
        input_shape_arr[i] = input_shape[i];
        input_strides_arr[i] = input_strides[i];
        output_strides_arr[i] = output_strides[i];
    }

    if (input.dtype() == DType::Float32) {
        const float* input_ptr = get_data_ptr<const float>(input);
        const int64_t* index_ptr = get_data_ptr<const int64_t>(index);
        float* output_ptr = get_data_ptr<float>(output);

        queue.parallel_for<IndexSelectKernelFloat32>(sycl::range<1>(numel), [=](sycl::id<1> flat_idx) {
            // Compute multi-dimensional index in output
            int64_t temp = flat_idx;
            int64_t input_idx = 0;

            for (size_t d = 0; d < ndims; ++d) {
                int64_t coord = temp / output_strides_arr[d];
                temp %= output_strides_arr[d];

                if (static_cast<int64_t>(d) == dim) {
                    // Use index tensor
                    int64_t idx_val = index_ptr[coord];
                    if (idx_val < 0) idx_val += input_shape_arr[d];
                    input_idx += idx_val * input_strides_arr[d];
                } else {
                    input_idx += coord * input_strides_arr[d];
                }
            }

            output_ptr[flat_idx] = input_ptr[input_idx];
        });
    }
    else if (input.dtype() == DType::Float64) {
        const double* input_ptr = get_data_ptr<const double>(input);
        const int64_t* index_ptr = get_data_ptr<const int64_t>(index);
        double* output_ptr = get_data_ptr<double>(output);

        queue.parallel_for<IndexSelectKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> flat_idx) {
            int64_t temp = flat_idx;
            int64_t input_idx = 0;

            for (size_t d = 0; d < ndims; ++d) {
                int64_t coord = temp / output_strides_arr[d];
                temp %= output_strides_arr[d];

                if (static_cast<int64_t>(d) == dim) {
                    int64_t idx_val = index_ptr[coord];
                    if (idx_val < 0) idx_val += input_shape_arr[d];
                    input_idx += idx_val * input_strides_arr[d];
                } else {
                    input_idx += coord * input_strides_arr[d];
                }
            }

            output_ptr[flat_idx] = input_ptr[input_idx];
        });
    }
    else if (input.dtype() == DType::Float16) {
        const sycl::half* input_ptr = get_data_ptr<const sycl::half>(input);
        const int64_t* index_ptr = get_data_ptr<const int64_t>(index);
        sycl::half* output_ptr = get_data_ptr<sycl::half>(output);

        queue.parallel_for<IndexSelectKernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> flat_idx) {
            int64_t temp = flat_idx;
            int64_t input_idx = 0;

            for (size_t d = 0; d < ndims; ++d) {
                int64_t coord = temp / output_strides_arr[d];
                temp %= output_strides_arr[d];

                if (static_cast<int64_t>(d) == dim) {
                    int64_t idx_val = index_ptr[coord];
                    if (idx_val < 0) idx_val += input_shape_arr[d];
                    input_idx += idx_val * input_strides_arr[d];
                } else {
                    input_idx += coord * input_strides_arr[d];
                }
            }

            output_ptr[flat_idx] = input_ptr[input_idx];
        });
    }
    else if (input.dtype() == DType::BFloat16) {
        const uint16_t* input_ptr = get_data_ptr<const uint16_t>(input);
        const int64_t* index_ptr = get_data_ptr<const int64_t>(index);
        uint16_t* output_ptr = get_data_ptr<uint16_t>(output);

        queue.parallel_for<IndexSelectKernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> flat_idx) {
            int64_t temp = flat_idx;
            int64_t input_idx = 0;

            for (size_t d = 0; d < ndims; ++d) {
                int64_t coord = temp / output_strides_arr[d];
                temp %= output_strides_arr[d];

                if (static_cast<int64_t>(d) == dim) {
                    int64_t idx_val = index_ptr[coord];
                    if (idx_val < 0) idx_val += input_shape_arr[d];
                    input_idx += idx_val * input_strides_arr[d];
                } else {
                    input_idx += coord * input_strides_arr[d];
                }
            }

            output_ptr[flat_idx] = input_ptr[input_idx];
        });
    }
    else if (input.dtype() == DType::Int32) {
        const int32_t* input_ptr = get_data_ptr<const int32_t>(input);
        const int64_t* index_ptr = get_data_ptr<const int64_t>(index);
        int32_t* output_ptr = get_data_ptr<int32_t>(output);

        queue.parallel_for<IndexSelectKernelInt32>(sycl::range<1>(numel), [=](sycl::id<1> flat_idx) {
            int64_t temp = flat_idx;
            int64_t input_idx = 0;
            for (size_t d = 0; d < ndims; ++d) {
                int64_t coord = temp / output_strides_arr[d];
                temp %= output_strides_arr[d];
                if (static_cast<int64_t>(d) == dim) {
                    int64_t idx_val = index_ptr[coord];
                    if (idx_val < 0) idx_val += input_shape_arr[d];
                    input_idx += idx_val * input_strides_arr[d];
                } else {
                    input_idx += coord * input_strides_arr[d];
                }
            }
            output_ptr[flat_idx] = input_ptr[input_idx];
        });
    }
    else if (input.dtype() == DType::Int64) {
        const int64_t* input_ptr = get_data_ptr<const int64_t>(input);
        const int64_t* index_ptr = get_data_ptr<const int64_t>(index);
        int64_t* output_ptr = get_data_ptr<int64_t>(output);

        queue.parallel_for<IndexSelectKernelInt64>(sycl::range<1>(numel), [=](sycl::id<1> flat_idx) {
            int64_t temp = flat_idx;
            int64_t input_idx = 0;
            for (size_t d = 0; d < ndims; ++d) {
                int64_t coord = temp / output_strides_arr[d];
                temp %= output_strides_arr[d];
                if (static_cast<int64_t>(d) == dim) {
                    int64_t idx_val = index_ptr[coord];
                    if (idx_val < 0) idx_val += input_shape_arr[d];
                    input_idx += idx_val * input_strides_arr[d];
                } else {
                    input_idx += coord * input_strides_arr[d];
                }
            }
            output_ptr[flat_idx] = input_ptr[input_idx];
        });
    }
    else if (input.dtype() == DType::Bool) {
        const bool* input_ptr = get_data_ptr<const bool>(input);
        const int64_t* index_ptr = get_data_ptr<const int64_t>(index);
        bool* output_ptr = get_data_ptr<bool>(output);

        queue.parallel_for<IndexSelectKernelBool>(sycl::range<1>(numel), [=](sycl::id<1> flat_idx) {
            int64_t temp = flat_idx;
            int64_t input_idx = 0;
            for (size_t d = 0; d < ndims; ++d) {
                int64_t coord = temp / output_strides_arr[d];
                temp %= output_strides_arr[d];
                if (static_cast<int64_t>(d) == dim) {
                    int64_t idx_val = index_ptr[coord];
                    if (idx_val < 0) idx_val += input_shape_arr[d];
                    input_idx += idx_val * input_strides_arr[d];
                } else {
                    input_idx += coord * input_strides_arr[d];
                }
            }
            output_ptr[flat_idx] = input_ptr[input_idx];
        });
    }
    else {
        throw std::runtime_error("Unsupported dtype for index_select");
    }

    return output;
}

// Masked fill - fill elements where mask is true with value
auto masked_fill_kernel(const Tensor& input, const Tensor& mask, float value, sycl::queue& queue) -> Tensor {
    auto input_shape = input.shape();
    auto mask_shape = mask.shape();
    if (!std::equal(input_shape.begin(), input_shape.end(), mask_shape.begin(), mask_shape.end())) {
        throw std::invalid_argument("MaskedFill: input and mask must have same shape");
    }

    Tensor output(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                  input.dtype(), input.device());

    const int64_t numel = input.numel();

    if (input.dtype() == DType::Float32) {
        const float* input_ptr = get_data_ptr<const float>(input);
        const bool* mask_ptr = get_data_ptr<const bool>(mask);
        float* output_ptr = get_data_ptr<float>(output);

        queue.parallel_for<MaskedFillKernelFloat32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            output_ptr[idx] = mask_ptr[idx] ? value : input_ptr[idx];
        });
    }
    else if (input.dtype() == DType::Float64) {
        const double* input_ptr = get_data_ptr<const double>(input);
        const bool* mask_ptr = get_data_ptr<const bool>(mask);
        double* output_ptr = get_data_ptr<double>(output);
        const double value_d = static_cast<double>(value);

        queue.parallel_for<MaskedFillKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            output_ptr[idx] = mask_ptr[idx] ? value_d : input_ptr[idx];
        });
    }
    else if (input.dtype() == DType::Float16) {
        const sycl::half* input_ptr = get_data_ptr<const sycl::half>(input);
        const bool* mask_ptr = get_data_ptr<const bool>(mask);
        sycl::half* output_ptr = get_data_ptr<sycl::half>(output);
        const sycl::half value_h = static_cast<sycl::half>(value);

        queue.parallel_for<MaskedFillKernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            output_ptr[idx] = mask_ptr[idx] ? value_h : input_ptr[idx];
        });
    }
    else if (input.dtype() == DType::BFloat16) {
        const uint16_t* input_ptr = get_data_ptr<const uint16_t>(input);
        const bool* mask_ptr = get_data_ptr<const bool>(mask);
        uint16_t* output_ptr = get_data_ptr<uint16_t>(output);
        const uint16_t value_bf16 = f32_to_bf16(value);

        queue.parallel_for<MaskedFillKernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            output_ptr[idx] = mask_ptr[idx] ? value_bf16 : input_ptr[idx];
        });
    }
    else {
        throw std::runtime_error("Unsupported dtype for masked_fill");
    }

    return output;
}

// sycl_exclusive_prefix_sum is now in ../sycl_prefix_sum.hpp

// Masked select - select elements where mask is true
// Uses device-side prefix sum to avoid host roundtrips
auto masked_select_kernel(const Tensor& input, const Tensor& mask, sycl::queue& queue) -> Tensor {
    auto input_shape = input.shape();
    auto mask_shape = mask.shape();
    if (!std::equal(input_shape.begin(), input_shape.end(), mask_shape.begin(), mask_shape.end())) {
        throw std::invalid_argument("MaskedSelect: input and mask must have same shape");
    }

    const int64_t numel = input.numel();
    if (numel == 0) {
        return Tensor({0}, input.dtype(), input.device());
    }

    const bool* mask_ptr = get_data_ptr<const bool>(mask);

    // Phase 1: Create int32 mask on device (bool -> 0/1)
    int32_t* d_mask_int = sycl::malloc_device<int32_t>(numel, queue);
    queue.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> i) {
        d_mask_int[i] = mask_ptr[i] ? 1 : 0;
    }).wait();

    // Phase 2: Exclusive prefix sum — returns total count
    // We need a copy for the scatter step (prefix sum modifies in-place)
    int32_t* d_offsets = sycl::malloc_device<int32_t>(numel, queue);
    queue.memcpy(d_offsets, d_mask_int, numel * sizeof(int32_t)).wait();

    int64_t true_count = sycl_exclusive_prefix_sum(d_offsets, numel, queue);

    // Create output
    Tensor output({true_count}, input.dtype(), input.device());

    if (true_count == 0) {
        sycl::free(d_mask_int, queue);
        sycl::free(d_offsets, queue);
        return output;
    }

    // Phase 3: Parallel scatter using prefix-sum offsets
    auto scatter_impl = [&](auto* out_ptr, const auto* in_ptr) {
        queue.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> i) {
            if (d_mask_int[i]) {
                out_ptr[d_offsets[i]] = in_ptr[i];
            }
        }).wait();
    };

    if (input.dtype() == DType::Float32) {
        scatter_impl(get_data_ptr<float>(output), get_data_ptr<const float>(input));
    }
    else if (input.dtype() == DType::Float64) {
        scatter_impl(get_data_ptr<double>(output), get_data_ptr<const double>(input));
    }
    else if (input.dtype() == DType::Float16) {
        scatter_impl(get_data_ptr<sycl::half>(output), get_data_ptr<const sycl::half>(input));
    }
    else if (input.dtype() == DType::BFloat16) {
        scatter_impl(get_data_ptr<uint16_t>(output), get_data_ptr<const uint16_t>(input));
    }
    else {
        sycl::free(d_mask_int, queue);
        sycl::free(d_offsets, queue);
        throw std::runtime_error("Unsupported dtype for masked_select");
    }

    sycl::free(d_mask_int, queue);
    sycl::free(d_offsets, queue);
    return output;
}

// Nonzero operation - find indices of non-zero elements
// Returns shape (num_nonzero, ndim) with Int64 dtype
// Uses device-side prefix sum to avoid host roundtrips
auto nonzero_kernel(const Tensor& input, sycl::queue& queue) -> Tensor {
    const int64_t numel = input.numel();
    auto input_shape_span = input.shape();
    std::vector<int64_t> input_shape(input_shape_span.begin(), input_shape_span.end());
    const int64_t ndim = static_cast<int64_t>(input_shape.size());

    if (numel == 0) {
        return Tensor({0, ndim}, DType::Int64, input.device());
    }

    // Phase 1: Create binary mask on device (nonzero -> 1, zero -> 0)
    int32_t* d_mask = sycl::malloc_device<int32_t>(numel, queue);

    auto create_mask = [&](const auto* ptr, auto zero_val) {
        using ValT = std::remove_const_t<std::remove_pointer_t<decltype(ptr)>>;
        auto zv = zero_val;
        queue.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> i) {
            d_mask[i] = (ptr[i] != zv) ? 1 : 0;
        }).wait();
    };

    // BFloat16 needs special comparison
    auto create_mask_bf16 = [&](const uint16_t* ptr) {
        queue.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> i) {
            // BFloat16: zero is 0x0000
            d_mask[i] = (ptr[i] != 0) ? 1 : 0;
        }).wait();
    };

    // Float16 needs cast for zero comparison
    auto create_mask_f16 = [&](const sycl::half* ptr) {
        sycl::half zero_h{0.0f};
        queue.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> i) {
            d_mask[i] = (ptr[i] != zero_h) ? 1 : 0;
        }).wait();
    };

    if (input.dtype() == DType::Float32) {
        create_mask(get_data_ptr<const float>(input), 0.0f);
    } else if (input.dtype() == DType::Float64) {
        create_mask(get_data_ptr<const double>(input), 0.0);
    } else if (input.dtype() == DType::Float16) {
        create_mask_f16(get_data_ptr<const sycl::half>(input));
    } else if (input.dtype() == DType::BFloat16) {
        create_mask_bf16(get_data_ptr<const uint16_t>(input));
    } else if (input.dtype() == DType::Int32) {
        create_mask(get_data_ptr<const int32_t>(input), int32_t{0});
    } else if (input.dtype() == DType::Int64) {
        create_mask(get_data_ptr<const int64_t>(input), int64_t{0});
    } else if (input.dtype() == DType::Bool) {
        const uint8_t* bool_ptr = reinterpret_cast<const uint8_t*>(get_data_ptr<const bool>(input));
        queue.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> i) {
            d_mask[i] = bool_ptr[i] ? 1 : 0;
        }).wait();
    } else {
        sycl::free(d_mask, queue);
        throw std::runtime_error("nonzero: unsupported dtype");
    }

    // Phase 2: Exclusive prefix sum over mask — returns total nonzero count
    int32_t* d_offsets = sycl::malloc_device<int32_t>(numel, queue);
    queue.memcpy(d_offsets, d_mask, numel * sizeof(int32_t)).wait();

    int64_t nonzero_count = sycl_exclusive_prefix_sum(d_offsets, numel, queue);

    // Create output tensor of shape (nonzero_count, ndim)
    Tensor output({nonzero_count, ndim}, DType::Int64, input.device());

    if (nonzero_count == 0) {
        sycl::free(d_mask, queue);
        sycl::free(d_offsets, queue);
        return output;
    }

    // Phase 3: Parallel kernel computes multi-dimensional indices using strides
    // Copy strides to device
    int64_t* d_strides = sycl::malloc_device<int64_t>(ndim, queue);
    auto strides = calculate_strides(input_shape);
    queue.memcpy(d_strides, strides.data(), ndim * sizeof(int64_t)).wait();

    int64_t* output_ptr = get_data_ptr<int64_t>(output);

    queue.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> gid) {
        int64_t i = gid;
        if (d_mask[i]) {
            int64_t out_row = d_offsets[i];
            int64_t flat = i;
            for (int64_t d = 0; d < ndim; ++d) {
                output_ptr[out_row * ndim + d] = flat / d_strides[d];
                flat %= d_strides[d];
            }
        }
    }).wait();

    sycl::free(d_mask, queue);
    sycl::free(d_offsets, queue);
    sycl::free(d_strides, queue);

    return output;
}

// One-hot encoding operation
// Input: Int64 indices tensor of shape (N,)
// Output: Float tensor of shape (N, num_classes)
class OneHotKernel {};

auto one_hot_kernel(const Tensor& indices, int64_t num_classes, DType output_dtype,
                    sycl::queue& queue) -> Tensor {
    auto indices_shape = indices.shape();
    int64_t numel = indices.numel();

    // Output shape: indices_shape + [num_classes]
    std::vector<int64_t> output_shape(indices_shape.begin(), indices_shape.end());
    output_shape.push_back(num_classes);

    Tensor output(output_shape, output_dtype, indices.device());
    int64_t total = numel * num_classes;

    // Zero-initialize
    if (output_dtype == DType::Float32) {
        float* out_ptr = get_data_ptr<float>(output);
        queue.fill(out_ptr, 0.0f, total);

        const int64_t* idx_ptr = get_data_ptr<const int64_t>(indices);

        queue.parallel_for<OneHotKernel>(sycl::range<1>(numel), [=](sycl::id<1> i) {
            int64_t cls = idx_ptr[i];
            if (cls >= 0 && cls < num_classes) {
                out_ptr[i * num_classes + cls] = 1.0f;
            }
        });
    }
    else if (output_dtype == DType::Float64) {
        double* out_ptr = get_data_ptr<double>(output);
        queue.fill(out_ptr, 0.0, total);

        const int64_t* idx_ptr = get_data_ptr<const int64_t>(indices);

        queue.parallel_for<class OneHotKernelF64>(sycl::range<1>(numel), [=](sycl::id<1> i) {
            int64_t cls = idx_ptr[i];
            if (cls >= 0 && cls < num_classes) {
                out_ptr[i * num_classes + cls] = 1.0;
            }
        });
    }
    else if (output_dtype == DType::Float16) {
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);
        queue.fill(out_ptr, sycl::half(0.0f), total);

        const int64_t* idx_ptr = get_data_ptr<const int64_t>(indices);

        queue.parallel_for<class OneHotKernelF16>(sycl::range<1>(numel), [=](sycl::id<1> i) {
            int64_t cls = idx_ptr[i];
            if (cls >= 0 && cls < num_classes) {
                out_ptr[i * num_classes + cls] = sycl::half(1.0f);
            }
        });
    }
    else if (output_dtype == DType::BFloat16) {
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);
        const uint16_t zero_bf16 = f32_to_bf16(0.0f);
        queue.fill(out_ptr, zero_bf16, total);

        const int64_t* idx_ptr = get_data_ptr<const int64_t>(indices);
        const uint16_t one_bf16 = f32_to_bf16(1.0f);

        queue.parallel_for<class OneHotKernelBF16>(sycl::range<1>(numel), [=](sycl::id<1> i) {
            int64_t cls = idx_ptr[i];
            if (cls >= 0 && cls < num_classes) {
                out_ptr[i * num_classes + cls] = one_bf16;
            }
        });
    }
    else {
        throw std::runtime_error("one_hot: unsupported output dtype");
    }

    return output;
}

// ============================================================================
// Argsort
// ============================================================================

/**
 * @brief Returns indices that would sort a tensor along a given dimension.
 *
 * For each 1D slice along the specified dimension, computes the indices
 * that would sort the elements.
 *
 * @param input Input tensor of any shape
 * @param dim Dimension along which to sort (negative indexing supported)
 * @param descending If true, sort in descending order
 * @param queue SYCL queue for execution
 * @return Int64 tensor of same shape with sorted indices
 */
auto argsort_kernel(const Tensor& input, int64_t dim, bool descending, sycl::queue& queue) -> Tensor {
    const int64_t ndim = input.ndim();

    // Normalize dimension
    if (dim < 0) {
        dim += ndim;
    }
    if (dim < 0 || dim >= ndim) {
        throw std::out_of_range("argsort: dimension out of range");
    }

    auto input_shape = input.shape();
    std::vector<int64_t> shape_vec(input_shape.begin(), input_shape.end());

    // Output has same shape as input but with Int64 dtype
    Tensor output(shape_vec, DType::Int64, input.device());

    const int64_t dim_size = shape_vec[dim];

    if (dim_size == 0) {
        return output;
    }

    // Compute inner_size to decide contiguous vs transpose path
    int64_t inner_size = 1;
    for (int64_t d = dim + 1; d < ndim; ++d) {
        inner_size *= shape_vec[d];
    }

#ifdef TENZOR_HAS_ONEDPL
    // Device-side argsort for contiguous sort dimension (inner_size == 1)
    if (inner_size == 1) {
        int64_t total_elems = input.numel();
        int64_t outer_size = total_elems / (dim_size * inner_size);
        auto policy = ::oneapi::dpl::execution::make_device_policy(queue);

        auto device_argsort_impl = [&](const auto* in_ptr) {
            using T = std::remove_const_t<std::remove_pointer_t<decltype(in_ptr)>>;
            int64_t* idx_ptr = get_data_ptr<int64_t>(output);

            // Allocate temp device buffer for values (we sort values+indices together)
            T* tmp_vals = sycl::malloc_device<T>(total_elems, queue);
            queue.memcpy(tmp_vals, in_ptr, total_elems * sizeof(T)).wait();

            // Initialize indices: each slice gets 0..dim_size-1
            queue.parallel_for(sycl::range<1>(total_elems), [=](sycl::id<1> gid) {
                idx_ptr[gid] = static_cast<int64_t>(gid[0]) % dim_size;
            }).wait();

            // Sort each contiguous slice using oneDPL sort_by_key
            for (int64_t o = 0; o < outer_size; ++o) {
                T* slice_vals = tmp_vals + o * dim_size;
                int64_t* slice_idx = idx_ptr + o * dim_size;
                if (descending) {
                    ::oneapi::dpl::sort_by_key(policy, slice_vals, slice_vals + dim_size,
                                              slice_idx, std::greater<T>());
                } else {
                    ::oneapi::dpl::sort_by_key(policy, slice_vals, slice_vals + dim_size,
                                              slice_idx);
                }
            }

            sycl::free(tmp_vals, queue);
        };

        if (input.dtype() == DType::Float32) {
            device_argsort_impl(get_data_ptr<const float>(input));
        } else if (input.dtype() == DType::Float64) {
            device_argsort_impl(get_data_ptr<const double>(input));
        } else if (input.dtype() == DType::Int32) {
            device_argsort_impl(get_data_ptr<const int32_t>(input));
        } else if (input.dtype() == DType::Int64) {
            device_argsort_impl(get_data_ptr<const int64_t>(input));
        } else if (input.dtype() == DType::Float16) {
            // Upcast to Float32 for sorting
            Tensor input_f32 = input.to(DType::Float32);
            device_argsort_impl(get_data_ptr<const float>(input_f32));
        } else if (input.dtype() == DType::BFloat16) {
            Tensor input_f32 = input.to(DType::Float32);
            device_argsort_impl(get_data_ptr<const float>(input_f32));
        } else {
            throw std::runtime_error("argsort: unsupported dtype");
        }

        return output;
    }
    // Fall through: transpose so sort dim is last, argsort on device, transpose back
    {
        std::vector<int64_t> perm(ndim);
        std::iota(perm.begin(), perm.end(), 0);
        std::swap(perm[dim], perm[ndim - 1]);

        std::vector<int64_t> inv_perm(ndim);
        for (int64_t i = 0; i < ndim; ++i) inv_perm[perm[i]] = i;

        Tensor transposed = input.permute(perm).contiguous();
        output = argsort_kernel(transposed, ndim - 1, descending, queue);
        output = output.permute(inv_perm).contiguous();
    }

    return output;
#else
    // Without oneDPL, no device-side argsort is available
    throw std::runtime_error("argsort: oneDPL required for device-side argsort");
#endif
}

// ============================================================================
// ScatterAdd kernel - scatter with addition
// ============================================================================
class ScatterAddKernelFloat32;
class ScatterAddKernelFloat64;
class ScatterAddKernelFloat16;
class ScatterAddKernelFloat16Convert;
class ScatterAddKernelBFloat16;
class ScatterAddKernelBFloat16Convert;
class ScatterAddKernelInt32;
class ScatterAddKernelInt64;

auto scatter_add_kernel(const Tensor& self, int64_t dim, const Tensor& index, const Tensor& src,
                        sycl::queue& queue) -> Tensor {
    auto self_shape_span = self.shape();
    std::vector<int64_t> shape(self_shape_span.begin(), self_shape_span.end());
    int64_t ndim = shape.size();
    if (dim < 0) dim += ndim;

    // Clone self as output (scatter_add modifies in-place semantically)
    Tensor output(shape, self.dtype(), self.device());
    queue.memcpy(const_cast<void*>(output.data_ptr()), self.data_ptr(),
                 self.numel() * self.dtype_size());

    // Host-side implementation for atomicity correctness
    int64_t idx_numel = index.numel();
    if (idx_numel == 0) return output;

    auto idx_shape = index.shape();
    std::vector<int64_t> idx_shape_vec(idx_shape.begin(), idx_shape.end());

    // Compute strides for self/output
    std::vector<int64_t> out_strides(ndim);
    { int64_t s = 1; for (int64_t i = ndim - 1; i >= 0; --i) { out_strides[i] = s; s *= shape[i]; } }
    std::vector<int64_t> idx_strides(ndim);
    { int64_t s = 1; for (int64_t i = ndim - 1; i >= 0; --i) { idx_strides[i] = s; s *= idx_shape_vec[i]; } }

    if (self.dtype() == DType::Float32) {
        float* out_ptr = get_data_ptr<float>(output);
        const float* src_ptr = get_data_ptr<const float>(src);
        const int64_t* idx_ptr = get_data_ptr<const int64_t>(index);

        std::array<int64_t, 8> d_out_strides{}, d_idx_strides{};
        for (int64_t d = 0; d < ndim; ++d) {
            d_out_strides[d] = out_strides[d];
            d_idx_strides[d] = idx_strides[d];
        }

        queue.parallel_for(sycl::range<1>(idx_numel), [=](sycl::id<1> id) {
            int64_t flat = id[0];
            int64_t remaining = flat;
            int64_t out_offset = 0;
            for (int64_t d = 0; d < ndim; ++d) {
                int64_t coord = remaining / d_idx_strides[d];
                remaining %= d_idx_strides[d];
                if (d == dim) {
                    out_offset += idx_ptr[flat] * d_out_strides[d];
                } else {
                    out_offset += coord * d_out_strides[d];
                }
            }
            sycl::atomic_ref<float, sycl::memory_order::relaxed,
                            sycl::memory_scope::device,
                            sycl::access::address_space::global_space>
                atomic_out(out_ptr[out_offset]);
            atomic_out.fetch_add(src_ptr[flat]);
        }).wait();
    } else if (self.dtype() == DType::Float64) {
        double* out_ptr = get_data_ptr<double>(output);
        const double* src_ptr = get_data_ptr<const double>(src);
        const int64_t* idx_ptr = get_data_ptr<const int64_t>(index);

        std::array<int64_t, 8> d_out_strides{}, d_idx_strides{};
        for (int64_t d = 0; d < ndim; ++d) {
            d_out_strides[d] = out_strides[d];
            d_idx_strides[d] = idx_strides[d];
        }

        queue.parallel_for(sycl::range<1>(idx_numel), [=](sycl::id<1> id) {
            int64_t flat = id[0];
            int64_t remaining = flat;
            int64_t out_offset = 0;
            for (int64_t d = 0; d < ndim; ++d) {
                int64_t coord = remaining / d_idx_strides[d];
                remaining %= d_idx_strides[d];
                if (d == dim) {
                    out_offset += idx_ptr[flat] * d_out_strides[d];
                } else {
                    out_offset += coord * d_out_strides[d];
                }
            }
            sycl::atomic_ref<double, sycl::memory_order::relaxed,
                            sycl::memory_scope::device,
                            sycl::access::address_space::global_space>
                atomic_out(out_ptr[out_offset]);
            atomic_out.fetch_add(src_ptr[flat]);
        }).wait();
    } else if (self.dtype() == DType::Int32) {
        int32_t* out_ptr = get_data_ptr<int32_t>(output);
        const int32_t* src_ptr = get_data_ptr<const int32_t>(src);
        const int64_t* idx_ptr = get_data_ptr<const int64_t>(index);

        std::array<int64_t, 8> d_out_strides{}, d_idx_strides{};
        for (int64_t d = 0; d < ndim; ++d) {
            d_out_strides[d] = out_strides[d];
            d_idx_strides[d] = idx_strides[d];
        }

        queue.parallel_for(sycl::range<1>(idx_numel), [=](sycl::id<1> id) {
            int64_t flat = id[0];
            int64_t remaining = flat;
            int64_t out_offset = 0;
            for (int64_t d = 0; d < ndim; ++d) {
                int64_t coord = remaining / d_idx_strides[d];
                remaining %= d_idx_strides[d];
                if (d == dim) {
                    out_offset += idx_ptr[flat] * d_out_strides[d];
                } else {
                    out_offset += coord * d_out_strides[d];
                }
            }
            sycl::atomic_ref<int32_t, sycl::memory_order::relaxed,
                            sycl::memory_scope::device,
                            sycl::access::address_space::global_space>
                atomic_out(out_ptr[out_offset]);
            atomic_out.fetch_add(src_ptr[flat]);
        }).wait();
    } else if (self.dtype() == DType::Int64) {
        int64_t* out_ptr = get_data_ptr<int64_t>(output);
        const int64_t* src_ptr = get_data_ptr<const int64_t>(src);
        const int64_t* idx_ptr = get_data_ptr<const int64_t>(index);

        std::array<int64_t, 8> d_out_strides{}, d_idx_strides{};
        for (int64_t d = 0; d < ndim; ++d) {
            d_out_strides[d] = out_strides[d];
            d_idx_strides[d] = idx_strides[d];
        }

        queue.parallel_for(sycl::range<1>(idx_numel), [=](sycl::id<1> id) {
            int64_t flat = id[0];
            int64_t remaining = flat;
            int64_t out_offset = 0;
            for (int64_t d = 0; d < ndim; ++d) {
                int64_t coord = remaining / d_idx_strides[d];
                remaining %= d_idx_strides[d];
                if (d == dim) {
                    out_offset += idx_ptr[flat] * d_out_strides[d];
                } else {
                    out_offset += coord * d_out_strides[d];
                }
            }
            sycl::atomic_ref<int64_t, sycl::memory_order::relaxed,
                            sycl::memory_scope::device,
                            sycl::access::address_space::global_space>
                atomic_out(out_ptr[out_offset]);
            atomic_out.fetch_add(src_ptr[flat]);
        }).wait();
    } else if (self.dtype() == DType::Float16) {
        // Float16: use float32 accumulator since atomic_ref<half> not widely supported
        float* acc = sycl::malloc_device<float>(self.numel(), queue);
        const sycl::half* self_ptr = get_data_ptr<const sycl::half>(output);
        // Convert output to float32
        queue.parallel_for(sycl::range<1>(self.numel()), [=](sycl::id<1> i) {
            acc[i] = static_cast<float>(self_ptr[i]);
        }).wait();

        const sycl::half* src_ptr = get_data_ptr<const sycl::half>(src);
        const int64_t* idx_ptr = get_data_ptr<const int64_t>(index);
        std::array<int64_t, 8> d_out_strides{}, d_idx_strides{};
        for (int64_t d = 0; d < ndim; ++d) {
            d_out_strides[d] = out_strides[d];
            d_idx_strides[d] = idx_strides[d];
        }

        queue.parallel_for<ScatterAddKernelFloat16>(sycl::range<1>(idx_numel), [=](sycl::id<1> id) {
            int64_t flat = id[0];
            int64_t remaining = flat;
            int64_t out_offset = 0;
            for (int64_t d = 0; d < ndim; ++d) {
                int64_t coord = remaining / d_idx_strides[d];
                remaining %= d_idx_strides[d];
                if (d == dim) {
                    out_offset += idx_ptr[flat] * d_out_strides[d];
                } else {
                    out_offset += coord * d_out_strides[d];
                }
            }
            sycl::atomic_ref<float, sycl::memory_order::relaxed,
                            sycl::memory_scope::device,
                            sycl::access::address_space::global_space>
                atomic_out(acc[out_offset]);
            atomic_out.fetch_add(static_cast<float>(src_ptr[flat]));
        }).wait();

        // Convert back to half
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);
        queue.parallel_for<ScatterAddKernelFloat16Convert>(sycl::range<1>(self.numel()), [=](sycl::id<1> i) {
            out_ptr[i] = sycl::half(acc[i]);
        }).wait();
        sycl::free(acc, queue);
    } else if (self.dtype() == DType::BFloat16) {
        float* acc = sycl::malloc_device<float>(self.numel(), queue);
        const uint16_t* self_ptr = get_data_ptr<const uint16_t>(output);
        queue.parallel_for(sycl::range<1>(self.numel()), [=](sycl::id<1> i) {
            acc[i] = bf16_to_f32(self_ptr[i]);
        }).wait();

        const uint16_t* src_ptr = get_data_ptr<const uint16_t>(src);
        const int64_t* idx_ptr = get_data_ptr<const int64_t>(index);
        std::array<int64_t, 8> d_out_strides{}, d_idx_strides{};
        for (int64_t d = 0; d < ndim; ++d) {
            d_out_strides[d] = out_strides[d];
            d_idx_strides[d] = idx_strides[d];
        }

        queue.parallel_for<ScatterAddKernelBFloat16>(sycl::range<1>(idx_numel), [=](sycl::id<1> id) {
            int64_t flat = id[0];
            int64_t remaining = flat;
            int64_t out_offset = 0;
            for (int64_t d = 0; d < ndim; ++d) {
                int64_t coord = remaining / d_idx_strides[d];
                remaining %= d_idx_strides[d];
                if (d == dim) {
                    out_offset += idx_ptr[flat] * d_out_strides[d];
                } else {
                    out_offset += coord * d_out_strides[d];
                }
            }
            sycl::atomic_ref<float, sycl::memory_order::relaxed,
                            sycl::memory_scope::device,
                            sycl::access::address_space::global_space>
                atomic_out(acc[out_offset]);
            atomic_out.fetch_add(bf16_to_f32(src_ptr[flat]));
        }).wait();

        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);
        queue.parallel_for<ScatterAddKernelBFloat16Convert>(sycl::range<1>(self.numel()), [=](sycl::id<1> i) {
            out_ptr[i] = f32_to_bf16(acc[i]);
        }).wait();
        sycl::free(acc, queue);
    } else {
        throw std::runtime_error("scatter_add: unsupported dtype");
    }

    return output;
}

// ============================================================================
// Put Operation
// ============================================================================
// Kernel name classes
class PutKernelFloat32;
class PutKernelFloat64;
class PutKernelInt32;
class PutKernelInt64;
class PutKernelFloat32Acc;
class PutKernelFloat64Acc;
class PutKernelInt32Acc;
class PutKernelInt64Acc;

auto put_kernel(
    const Tensor& input,
    const Tensor& indices,
    const Tensor& source,
    bool accumulate,
    sycl::queue& queue
) -> Tensor {
    Tensor output = input.clone();

    int64_t num_indices = indices.numel();
    int64_t total_size = input.numel();

    if (num_indices == 0) return output;

    if (accumulate) {
        // Accumulate mode: use device-side parallel_for with atomic operations
        if (input.dtype() == DType::Float32) {
            float* out_ptr = get_data_ptr<float>(output);
            const float* src_ptr = get_data_ptr<const float>(source);
            const int64_t* idx_ptr = get_data_ptr<const int64_t>(indices);

            queue.parallel_for<PutKernelFloat32Acc>(sycl::range<1>(num_indices), [=](sycl::id<1> id) {
                int64_t i = id[0];
                int64_t target_idx = idx_ptr[i];
                if (target_idx < 0) target_idx += total_size;
                if (target_idx >= 0 && target_idx < total_size) {
                    sycl::atomic_ref<float, sycl::memory_order::relaxed,
                                    sycl::memory_scope::device,
                                    sycl::access::address_space::global_space>
                        atomic_out(out_ptr[target_idx]);
                    atomic_out.fetch_add(src_ptr[i]);
                }
            }).wait();
        } else if (input.dtype() == DType::Float64) {
            double* out_ptr = get_data_ptr<double>(output);
            const double* src_ptr = get_data_ptr<const double>(source);
            const int64_t* idx_ptr = get_data_ptr<const int64_t>(indices);

            queue.parallel_for<PutKernelFloat64Acc>(sycl::range<1>(num_indices), [=](sycl::id<1> id) {
                int64_t i = id[0];
                int64_t target_idx = idx_ptr[i];
                if (target_idx < 0) target_idx += total_size;
                if (target_idx >= 0 && target_idx < total_size) {
                    sycl::atomic_ref<double, sycl::memory_order::relaxed,
                                    sycl::memory_scope::device,
                                    sycl::access::address_space::global_space>
                        atomic_out(out_ptr[target_idx]);
                    atomic_out.fetch_add(src_ptr[i]);
                }
            }).wait();
        } else if (input.dtype() == DType::Int32) {
            int32_t* out_ptr = get_data_ptr<int32_t>(output);
            const int32_t* src_ptr = get_data_ptr<const int32_t>(source);
            const int64_t* idx_ptr = get_data_ptr<const int64_t>(indices);

            queue.parallel_for<PutKernelInt32Acc>(sycl::range<1>(num_indices), [=](sycl::id<1> id) {
                int64_t i = id[0];
                int64_t target_idx = idx_ptr[i];
                if (target_idx < 0) target_idx += total_size;
                if (target_idx >= 0 && target_idx < total_size) {
                    sycl::atomic_ref<int32_t, sycl::memory_order::relaxed,
                                    sycl::memory_scope::device,
                                    sycl::access::address_space::global_space>
                        atomic_out(out_ptr[target_idx]);
                    atomic_out.fetch_add(src_ptr[i]);
                }
            }).wait();
        } else if (input.dtype() == DType::Int64) {
            int64_t* out_ptr = get_data_ptr<int64_t>(output);
            const int64_t* src_ptr = get_data_ptr<const int64_t>(source);
            const int64_t* idx_ptr = get_data_ptr<const int64_t>(indices);

            queue.parallel_for<PutKernelInt64Acc>(sycl::range<1>(num_indices), [=](sycl::id<1> id) {
                int64_t i = id[0];
                int64_t target_idx = idx_ptr[i];
                if (target_idx < 0) target_idx += total_size;
                if (target_idx >= 0 && target_idx < total_size) {
                    sycl::atomic_ref<int64_t, sycl::memory_order::relaxed,
                                    sycl::memory_scope::device,
                                    sycl::access::address_space::global_space>
                        atomic_out(out_ptr[target_idx]);
                    atomic_out.fetch_add(src_ptr[i]);
                }
            }).wait();
        } else {
            throw std::runtime_error("put_kernel: unsupported dtype for accumulate mode");
        }
    } else {
        // Non-accumulate mode: safe to use parallel_for (last write wins semantics)
        if (input.dtype() == DType::Float32) {
            float* out_ptr = get_data_ptr<float>(output);
            const int64_t* idx_ptr = get_data_ptr<const int64_t>(indices);
            const float* src_ptr = get_data_ptr<const float>(source);

            queue.parallel_for<PutKernelFloat32>(sycl::range<1>(num_indices), [=](sycl::id<1> i) {
                int64_t target_idx = idx_ptr[i];
                if (target_idx < 0) target_idx += total_size;
                if (target_idx >= 0 && target_idx < total_size) {
                    out_ptr[target_idx] = src_ptr[i];
                }
            });
        } else if (input.dtype() == DType::Float64) {
            double* out_ptr = get_data_ptr<double>(output);
            const int64_t* idx_ptr = get_data_ptr<const int64_t>(indices);
            const double* src_ptr = get_data_ptr<const double>(source);

            queue.parallel_for<PutKernelFloat64>(sycl::range<1>(num_indices), [=](sycl::id<1> i) {
                int64_t target_idx = idx_ptr[i];
                if (target_idx < 0) target_idx += total_size;
                if (target_idx >= 0 && target_idx < total_size) {
                    out_ptr[target_idx] = src_ptr[i];
                }
            });
        } else if (input.dtype() == DType::Int32) {
            int32_t* out_ptr = get_data_ptr<int32_t>(output);
            const int64_t* idx_ptr = get_data_ptr<const int64_t>(indices);
            const int32_t* src_ptr = get_data_ptr<const int32_t>(source);

            queue.parallel_for<PutKernelInt32>(sycl::range<1>(num_indices), [=](sycl::id<1> i) {
                int64_t target_idx = idx_ptr[i];
                if (target_idx < 0) target_idx += total_size;
                if (target_idx >= 0 && target_idx < total_size) {
                    out_ptr[target_idx] = src_ptr[i];
                }
            });
        } else if (input.dtype() == DType::Int64) {
            int64_t* out_ptr = get_data_ptr<int64_t>(output);
            const int64_t* idx_ptr = get_data_ptr<const int64_t>(indices);
            const int64_t* src_ptr = get_data_ptr<const int64_t>(source);

            queue.parallel_for<PutKernelInt64>(sycl::range<1>(num_indices), [=](sycl::id<1> i) {
                int64_t target_idx = idx_ptr[i];
                if (target_idx < 0) target_idx += total_size;
                if (target_idx >= 0 && target_idx < total_size) {
                    out_ptr[target_idx] = src_ptr[i];
                }
            });
        } else {
            throw std::runtime_error("put_kernel: unsupported dtype");
        }
    }

    return output;
}

// ============================================================================
// SearchSorted: binary search per element in a sorted 1-D sequence
// ============================================================================

class SearchSortedKernelFloat32;
class SearchSortedKernelFloat64;
class SearchSortedKernelInt32;
class SearchSortedKernelInt64;

auto searchsorted_kernel(const Tensor& sorted_sequence, const Tensor& values,
                          bool right, sycl::queue& queue) -> Tensor {
    if (sorted_sequence.ndim() != 1) {
        throw std::runtime_error("searchsorted: sorted_sequence must be 1-D");
    }

    Tensor seq_cont = sorted_sequence.contiguous();
    Tensor val_cont = values.contiguous();
    int64_t seq_len = seq_cont.shape()[0];
    int64_t num_values = val_cont.numel();

    Tensor result(std::vector<int64_t>(values.shape().begin(), values.shape().end()),
                  DType::Int64, values.device());

    if (num_values == 0) return result;

    int64_t* out_ptr = get_data_ptr<int64_t>(result);

    auto launch_search = [&]<typename T, typename KernelName>(const T* seq_ptr, const T* val_ptr) {
        queue.parallel_for<KernelName>(sycl::range<1>(num_values), [=](sycl::id<1> i) {
            T v = val_ptr[i];
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
        });
        queue.wait();
    };

    switch (sorted_sequence.dtype()) {
        case DType::Float32:
            launch_search.template operator()<float, SearchSortedKernelFloat32>(
                get_data_ptr<const float>(seq_cont), get_data_ptr<const float>(val_cont));
            break;
        case DType::Float64:
            launch_search.template operator()<double, SearchSortedKernelFloat64>(
                get_data_ptr<const double>(seq_cont), get_data_ptr<const double>(val_cont));
            break;
        case DType::Int32:
            launch_search.template operator()<int32_t, SearchSortedKernelInt32>(
                get_data_ptr<const int32_t>(seq_cont), get_data_ptr<const int32_t>(val_cont));
            break;
        case DType::Int64:
            launch_search.template operator()<int64_t, SearchSortedKernelInt64>(
                get_data_ptr<const int64_t>(seq_cont), get_data_ptr<const int64_t>(val_cont));
            break;
        case DType::Float16:
        case DType::BFloat16: {
            auto seq_f32 = sorted_sequence.to(DType::Float32).contiguous();
            auto val_f32 = values.to(DType::Float32).contiguous();
            launch_search.template operator()<float, SearchSortedKernelFloat32>(
                get_data_ptr<const float>(seq_f32), get_data_ptr<const float>(val_f32));
            break;
        }
        default:
            throw std::runtime_error("searchsorted: unsupported dtype " +
                                     std::string(dtype_name(sorted_sequence.dtype())));
    }

    return result;
}

} // namespace oneapi
} // namespace tenzor
