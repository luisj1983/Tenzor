#include "tenzor/core/tensor.hpp"
#include <sycl/sycl.hpp>
#include <stdexcept>
#include <algorithm>
#include <numeric>
#include <vector>

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

// Masked select - select elements where mask is true
auto masked_select_kernel(const Tensor& input, const Tensor& mask, sycl::queue& queue) -> Tensor {
    auto input_shape = input.shape();
    auto mask_shape = mask.shape();
    if (!std::equal(input_shape.begin(), input_shape.end(), mask_shape.begin(), mask_shape.end())) {
        throw std::invalid_argument("MaskedSelect: input and mask must have same shape");
    }

    const int64_t numel = input.numel();

    // First pass: count true values
    Tensor count_buffer({1}, DType::Int64, input.device());
    int64_t* count_ptr = get_data_ptr<int64_t>(count_buffer);
    *count_ptr = 0;

    const bool* mask_ptr = get_data_ptr<const bool>(mask);

    // Count on host for simplicity (could be optimized with parallel reduction)
    bool* mask_host = new bool[numel];
    queue.memcpy(mask_host, mask_ptr, numel * sizeof(bool)).wait();

    int64_t true_count = 0;
    for (int64_t i = 0; i < numel; ++i) {
        if (mask_host[i]) true_count++;
    }
    delete[] mask_host;

    // Create output with size = number of true values
    Tensor output({true_count}, input.dtype(), input.device());

    if (true_count == 0) {
        return output;
    }

    // Second pass: copy selected elements
    if (input.dtype() == DType::Float32) {
        const float* input_ptr = get_data_ptr<const float>(input);
        float* output_ptr = get_data_ptr<float>(output);

        // Serial copy (could be optimized with prefix sum)
        float* input_host = new float[numel];
        bool* mask_host2 = new bool[numel];
        queue.memcpy(input_host, input_ptr, numel * sizeof(float));
        queue.memcpy(mask_host2, mask_ptr, numel * sizeof(bool)).wait();

        int64_t out_idx = 0;
        for (int64_t i = 0; i < numel; ++i) {
            if (mask_host2[i]) {
                output_ptr[out_idx++] = input_host[i];
            }
        }

        delete[] input_host;
        delete[] mask_host2;
    }
    else if (input.dtype() == DType::Float64) {
        const double* input_ptr = get_data_ptr<const double>(input);
        double* output_ptr = get_data_ptr<double>(output);

        double* input_host = new double[numel];
        bool* mask_host2 = new bool[numel];
        queue.memcpy(input_host, input_ptr, numel * sizeof(double));
        queue.memcpy(mask_host2, mask_ptr, numel * sizeof(bool)).wait();

        int64_t out_idx = 0;
        for (int64_t i = 0; i < numel; ++i) {
            if (mask_host2[i]) {
                output_ptr[out_idx++] = input_host[i];
            }
        }

        delete[] input_host;
        delete[] mask_host2;
    }
    else if (input.dtype() == DType::Float16) {
        const sycl::half* input_ptr = get_data_ptr<const sycl::half>(input);
        sycl::half* output_ptr = get_data_ptr<sycl::half>(output);

        sycl::half* input_host = new sycl::half[numel];
        bool* mask_host2 = new bool[numel];
        queue.memcpy(input_host, input_ptr, numel * sizeof(sycl::half));
        queue.memcpy(mask_host2, mask_ptr, numel * sizeof(bool)).wait();

        int64_t out_idx = 0;
        for (int64_t i = 0; i < numel; ++i) {
            if (mask_host2[i]) {
                output_ptr[out_idx++] = input_host[i];
            }
        }

        delete[] input_host;
        delete[] mask_host2;
    }
    else if (input.dtype() == DType::BFloat16) {
        const uint16_t* input_ptr = get_data_ptr<const uint16_t>(input);
        uint16_t* output_ptr = get_data_ptr<uint16_t>(output);

        uint16_t* input_host = new uint16_t[numel];
        bool* mask_host2 = new bool[numel];
        queue.memcpy(input_host, input_ptr, numel * sizeof(uint16_t));
        queue.memcpy(mask_host2, mask_ptr, numel * sizeof(bool)).wait();

        int64_t out_idx = 0;
        for (int64_t i = 0; i < numel; ++i) {
            if (mask_host2[i]) {
                output_ptr[out_idx++] = input_host[i];
            }
        }

        delete[] input_host;
        delete[] mask_host2;
    }
    else {
        throw std::runtime_error("Unsupported dtype for masked_select");
    }

    return output;
}

// Nonzero operation - find indices of non-zero elements
// Returns shape (num_nonzero, ndim) with Int64 dtype
auto nonzero_kernel(const Tensor& input, sycl::queue& queue) -> Tensor {
    const int64_t numel = input.numel();
    auto input_shape_span = input.shape();
    std::vector<int64_t> input_shape(input_shape_span.begin(), input_shape_span.end());
    const size_t ndim = input_shape.size();

    auto strides = calculate_strides(input_shape);

    // First pass: count nonzero elements on host
    int64_t nonzero_count = 0;

    if (input.dtype() == DType::Float32) {
        std::vector<float> host_data(numel);
        queue.memcpy(host_data.data(), get_data_ptr<const float>(input), numel * sizeof(float)).wait();
        for (int64_t i = 0; i < numel; ++i) {
            if (host_data[i] != 0.0f) nonzero_count++;
        }
    }
    else if (input.dtype() == DType::Float64) {
        std::vector<double> host_data(numel);
        queue.memcpy(host_data.data(), get_data_ptr<const double>(input), numel * sizeof(double)).wait();
        for (int64_t i = 0; i < numel; ++i) {
            if (host_data[i] != 0.0) nonzero_count++;
        }
    }
    else if (input.dtype() == DType::Float16) {
        std::vector<sycl::half> host_data(numel);
        queue.memcpy(host_data.data(), get_data_ptr<const sycl::half>(input), numel * sizeof(sycl::half)).wait();
        for (int64_t i = 0; i < numel; ++i) {
            if (static_cast<float>(host_data[i]) != 0.0f) nonzero_count++;
        }
    }
    else if (input.dtype() == DType::BFloat16) {
        std::vector<uint16_t> host_data(numel);
        queue.memcpy(host_data.data(), get_data_ptr<const uint16_t>(input), numel * sizeof(uint16_t)).wait();
        for (int64_t i = 0; i < numel; ++i) {
            if (bf16_to_f32(host_data[i]) != 0.0f) nonzero_count++;
        }
    }
    else if (input.dtype() == DType::Int32) {
        std::vector<int32_t> host_data(numel);
        queue.memcpy(host_data.data(), get_data_ptr<const int32_t>(input), numel * sizeof(int32_t)).wait();
        for (int64_t i = 0; i < numel; ++i) {
            if (host_data[i] != 0) nonzero_count++;
        }
    }
    else if (input.dtype() == DType::Int64) {
        std::vector<int64_t> host_data(numel);
        queue.memcpy(host_data.data(), get_data_ptr<const int64_t>(input), numel * sizeof(int64_t)).wait();
        for (int64_t i = 0; i < numel; ++i) {
            if (host_data[i] != 0) nonzero_count++;
        }
    }
    else if (input.dtype() == DType::Bool) {
        std::vector<bool> host_data(numel);
        const bool* bool_ptr = get_data_ptr<const bool>(input);
        // std::vector<bool> is special, copy element by element
        std::vector<uint8_t> raw_data(numel);
        queue.memcpy(raw_data.data(), bool_ptr, numel * sizeof(uint8_t)).wait();
        for (int64_t i = 0; i < numel; ++i) {
            if (raw_data[i]) nonzero_count++;
        }
    }
    else {
        throw std::runtime_error("nonzero: unsupported dtype");
    }

    // Create output tensor of shape (nonzero_count, ndim)
    Tensor output({nonzero_count, static_cast<int64_t>(ndim)}, DType::Int64, input.device());

    if (nonzero_count == 0) {
        return output;
    }

    // Second pass: compute multi-dimensional indices
    std::vector<int64_t> host_output(nonzero_count * ndim);
    int64_t out_idx = 0;

    // Lambda to convert flat index to multi-dimensional indices
    auto flat_to_multi = [&](int64_t flat) {
        int64_t temp = flat;
        for (size_t d = 0; d < ndim; ++d) {
            host_output[out_idx * ndim + d] = temp / strides[d];
            temp %= strides[d];
        }
        out_idx++;
    };

    if (input.dtype() == DType::Float32) {
        std::vector<float> host_data(numel);
        queue.memcpy(host_data.data(), get_data_ptr<const float>(input), numel * sizeof(float)).wait();
        for (int64_t i = 0; i < numel; ++i) {
            if (host_data[i] != 0.0f) flat_to_multi(i);
        }
    }
    else if (input.dtype() == DType::Float64) {
        std::vector<double> host_data(numel);
        queue.memcpy(host_data.data(), get_data_ptr<const double>(input), numel * sizeof(double)).wait();
        for (int64_t i = 0; i < numel; ++i) {
            if (host_data[i] != 0.0) flat_to_multi(i);
        }
    }
    else if (input.dtype() == DType::Float16) {
        std::vector<sycl::half> host_data(numel);
        queue.memcpy(host_data.data(), get_data_ptr<const sycl::half>(input), numel * sizeof(sycl::half)).wait();
        for (int64_t i = 0; i < numel; ++i) {
            if (static_cast<float>(host_data[i]) != 0.0f) flat_to_multi(i);
        }
    }
    else if (input.dtype() == DType::BFloat16) {
        std::vector<uint16_t> host_data(numel);
        queue.memcpy(host_data.data(), get_data_ptr<const uint16_t>(input), numel * sizeof(uint16_t)).wait();
        for (int64_t i = 0; i < numel; ++i) {
            if (bf16_to_f32(host_data[i]) != 0.0f) flat_to_multi(i);
        }
    }
    else if (input.dtype() == DType::Int32) {
        std::vector<int32_t> host_data(numel);
        queue.memcpy(host_data.data(), get_data_ptr<const int32_t>(input), numel * sizeof(int32_t)).wait();
        for (int64_t i = 0; i < numel; ++i) {
            if (host_data[i] != 0) flat_to_multi(i);
        }
    }
    else if (input.dtype() == DType::Int64) {
        std::vector<int64_t> host_data(numel);
        queue.memcpy(host_data.data(), get_data_ptr<const int64_t>(input), numel * sizeof(int64_t)).wait();
        for (int64_t i = 0; i < numel; ++i) {
            if (host_data[i] != 0) flat_to_multi(i);
        }
    }
    else if (input.dtype() == DType::Bool) {
        std::vector<uint8_t> raw_data(numel);
        queue.memcpy(raw_data.data(), get_data_ptr<const bool>(input), numel * sizeof(uint8_t)).wait();
        for (int64_t i = 0; i < numel; ++i) {
            if (raw_data[i]) flat_to_multi(i);
        }
    }

    // Copy result to device
    int64_t* output_ptr = get_data_ptr<int64_t>(output);
    queue.memcpy(output_ptr, host_output.data(), nonzero_count * ndim * sizeof(int64_t)).wait();

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

    // Compute outer_size and inner_size
    int64_t inner_size = 1;
    for (int64_t d = dim + 1; d < ndim; ++d) {
        inner_size *= shape_vec[d];
    }

    int64_t total_elems = input.numel();
    int64_t outer_size = total_elems / (dim_size * inner_size);
    int64_t num_slices = outer_size * inner_size;

    // Copy input data to host for sorting
    // Argsort is inherently comparison-based and benefits from host-side std::sort
    // The parallelism comes from processing many independent slices

    auto sort_slices = [&](auto* host_input) {
        using T = std::remove_const_t<std::remove_pointer_t<decltype(host_input)>>;

        // Copy input to host
        std::vector<T> h_input(total_elems);
        queue.memcpy(h_input.data(), host_input, total_elems * sizeof(T)).wait();

        // Allocate host output
        std::vector<int64_t> h_output(total_elems);

        // Sort each slice independently
        for (int64_t slice = 0; slice < num_slices; ++slice) {
            int64_t outer = slice / inner_size;
            int64_t inner = slice % inner_size;

            // Collect values along the dimension
            std::vector<std::pair<T, int64_t>> values(dim_size);
            for (int64_t i = 0; i < dim_size; ++i) {
                int64_t offset = outer * dim_size * inner_size + i * inner_size + inner;
                values[i] = {h_input[offset], i};
            }

            // Sort by value
            if (descending) {
                std::sort(values.begin(), values.end(),
                    [](const auto& a, const auto& b) { return a.first > b.first; });
            } else {
                std::sort(values.begin(), values.end(),
                    [](const auto& a, const auto& b) { return a.first < b.first; });
            }

            // Write sorted indices
            for (int64_t i = 0; i < dim_size; ++i) {
                int64_t offset = outer * dim_size * inner_size + i * inner_size + inner;
                h_output[offset] = values[i].second;
            }
        }

        // Copy output to device
        int64_t* output_ptr = get_data_ptr<int64_t>(output);
        queue.memcpy(output_ptr, h_output.data(), total_elems * sizeof(int64_t)).wait();
    };

    if (input.dtype() == DType::Float32) {
        sort_slices(get_data_ptr<const float>(input));
    }
    else if (input.dtype() == DType::Float64) {
        sort_slices(get_data_ptr<const double>(input));
    }
    else if (input.dtype() == DType::Int32) {
        sort_slices(get_data_ptr<const int32_t>(input));
    }
    else if (input.dtype() == DType::Int64) {
        sort_slices(get_data_ptr<const int64_t>(input));
    }
    else if (input.dtype() == DType::Float16) {
        // Convert Float16 to Float32 for sorting
        const sycl::half* h_input_ptr = get_data_ptr<const sycl::half>(input);
        std::vector<sycl::half> h_input_half(total_elems);
        queue.memcpy(h_input_half.data(), h_input_ptr, total_elems * sizeof(sycl::half)).wait();

        // Convert to float
        std::vector<float> h_input_f32(total_elems);
        for (int64_t i = 0; i < total_elems; ++i) {
            h_input_f32[i] = float(h_input_half[i]);
        }

        // Allocate host output
        std::vector<int64_t> h_output(total_elems);

        for (int64_t slice = 0; slice < num_slices; ++slice) {
            int64_t outer = slice / inner_size;
            int64_t inner = slice % inner_size;

            std::vector<std::pair<float, int64_t>> values(dim_size);
            for (int64_t i = 0; i < dim_size; ++i) {
                int64_t offset = outer * dim_size * inner_size + i * inner_size + inner;
                values[i] = {h_input_f32[offset], i};
            }

            if (descending) {
                std::sort(values.begin(), values.end(),
                    [](const auto& a, const auto& b) { return a.first > b.first; });
            } else {
                std::sort(values.begin(), values.end(),
                    [](const auto& a, const auto& b) { return a.first < b.first; });
            }

            for (int64_t i = 0; i < dim_size; ++i) {
                int64_t offset = outer * dim_size * inner_size + i * inner_size + inner;
                h_output[offset] = values[i].second;
            }
        }

        int64_t* output_ptr = get_data_ptr<int64_t>(output);
        queue.memcpy(output_ptr, h_output.data(), total_elems * sizeof(int64_t)).wait();
    }
    else if (input.dtype() == DType::BFloat16) {
        // Convert BFloat16 to Float32 for sorting
        const uint16_t* h_input_ptr = get_data_ptr<const uint16_t>(input);
        std::vector<uint16_t> h_input_bf16(total_elems);
        queue.memcpy(h_input_bf16.data(), h_input_ptr, total_elems * sizeof(uint16_t)).wait();

        // Convert to float
        std::vector<float> h_input_f32(total_elems);
        for (int64_t i = 0; i < total_elems; ++i) {
            h_input_f32[i] = bf16_to_f32(h_input_bf16[i]);
        }

        // Allocate host output
        std::vector<int64_t> h_output(total_elems);

        for (int64_t slice = 0; slice < num_slices; ++slice) {
            int64_t outer = slice / inner_size;
            int64_t inner = slice % inner_size;

            std::vector<std::pair<float, int64_t>> values(dim_size);
            for (int64_t i = 0; i < dim_size; ++i) {
                int64_t offset = outer * dim_size * inner_size + i * inner_size + inner;
                values[i] = {h_input_f32[offset], i};
            }

            if (descending) {
                std::sort(values.begin(), values.end(),
                    [](const auto& a, const auto& b) { return a.first > b.first; });
            } else {
                std::sort(values.begin(), values.end(),
                    [](const auto& a, const auto& b) { return a.first < b.first; });
            }

            for (int64_t i = 0; i < dim_size; ++i) {
                int64_t offset = outer * dim_size * inner_size + i * inner_size + inner;
                h_output[offset] = values[i].second;
            }
        }

        int64_t* output_ptr = get_data_ptr<int64_t>(output);
        queue.memcpy(output_ptr, h_output.data(), total_elems * sizeof(int64_t)).wait();
    }
    else {
        throw std::runtime_error("argsort: unsupported dtype");
    }

    return output;
}

// ============================================================================
// ScatterAdd kernel - scatter with addition
// ============================================================================
class ScatterAddKernelFloat32;
class ScatterAddKernelFloat64;
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
        std::vector<float> h_out(self.numel());
        std::vector<float> h_src(src.numel());
        std::vector<int64_t> h_idx(idx_numel);
        queue.memcpy(h_out.data(), output.data_ptr(), self.numel() * sizeof(float));
        queue.memcpy(h_src.data(), src.data_ptr(), src.numel() * sizeof(float));
        queue.memcpy(h_idx.data(), index.data_ptr(), idx_numel * sizeof(int64_t)).wait();

        for (int64_t flat = 0; flat < idx_numel; ++flat) {
            // Decompose flat index to multi-dim
            int64_t remaining = flat;
            int64_t out_offset = 0;
            int64_t src_offset = flat;
            for (int64_t d = 0; d < ndim; ++d) {
                int64_t coord = remaining / idx_strides[d];
                remaining %= idx_strides[d];
                if (d == dim) {
                    out_offset += h_idx[flat] * out_strides[d];
                } else {
                    out_offset += coord * out_strides[d];
                }
            }
            h_out[out_offset] += h_src[src_offset];
        }
        queue.memcpy(const_cast<void*>(output.data_ptr()), h_out.data(), self.numel() * sizeof(float)).wait();
    } else if (self.dtype() == DType::Float64) {
        std::vector<double> h_out(self.numel());
        std::vector<double> h_src(src.numel());
        std::vector<int64_t> h_idx(idx_numel);
        queue.memcpy(h_out.data(), output.data_ptr(), self.numel() * sizeof(double));
        queue.memcpy(h_src.data(), src.data_ptr(), src.numel() * sizeof(double));
        queue.memcpy(h_idx.data(), index.data_ptr(), idx_numel * sizeof(int64_t)).wait();

        for (int64_t flat = 0; flat < idx_numel; ++flat) {
            int64_t remaining = flat;
            int64_t out_offset = 0;
            for (int64_t d = 0; d < ndim; ++d) {
                int64_t coord = remaining / idx_strides[d];
                remaining %= idx_strides[d];
                if (d == dim) {
                    out_offset += h_idx[flat] * out_strides[d];
                } else {
                    out_offset += coord * out_strides[d];
                }
            }
            h_out[out_offset] += h_src[flat];
        }
        queue.memcpy(const_cast<void*>(output.data_ptr()), h_out.data(), self.numel() * sizeof(double)).wait();
    } else if (self.dtype() == DType::Int32) {
        std::vector<int32_t> h_out(self.numel());
        std::vector<int32_t> h_src(src.numel());
        std::vector<int64_t> h_idx(idx_numel);
        queue.memcpy(h_out.data(), output.data_ptr(), self.numel() * sizeof(int32_t));
        queue.memcpy(h_src.data(), src.data_ptr(), src.numel() * sizeof(int32_t));
        queue.memcpy(h_idx.data(), index.data_ptr(), idx_numel * sizeof(int64_t)).wait();

        for (int64_t flat = 0; flat < idx_numel; ++flat) {
            int64_t remaining = flat;
            int64_t out_offset = 0;
            for (int64_t d = 0; d < ndim; ++d) {
                int64_t coord = remaining / idx_strides[d];
                remaining %= idx_strides[d];
                if (d == dim) {
                    out_offset += h_idx[flat] * out_strides[d];
                } else {
                    out_offset += coord * out_strides[d];
                }
            }
            h_out[out_offset] += h_src[flat];
        }
        queue.memcpy(const_cast<void*>(output.data_ptr()), h_out.data(), self.numel() * sizeof(int32_t)).wait();
    } else if (self.dtype() == DType::Int64) {
        std::vector<int64_t> h_out(self.numel());
        std::vector<int64_t> h_src(src.numel());
        std::vector<int64_t> h_idx(idx_numel);
        queue.memcpy(h_out.data(), output.data_ptr(), self.numel() * sizeof(int64_t));
        queue.memcpy(h_src.data(), src.data_ptr(), src.numel() * sizeof(int64_t));
        queue.memcpy(h_idx.data(), index.data_ptr(), idx_numel * sizeof(int64_t)).wait();

        for (int64_t flat = 0; flat < idx_numel; ++flat) {
            int64_t remaining = flat;
            int64_t out_offset = 0;
            for (int64_t d = 0; d < ndim; ++d) {
                int64_t coord = remaining / idx_strides[d];
                remaining %= idx_strides[d];
                if (d == dim) {
                    out_offset += h_idx[flat] * out_strides[d];
                } else {
                    out_offset += coord * out_strides[d];
                }
            }
            h_out[out_offset] += h_src[flat];
        }
        queue.memcpy(const_cast<void*>(output.data_ptr()), h_out.data(), self.numel() * sizeof(int64_t)).wait();
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
        // Accumulate mode: must use host-side loop to avoid data races
        // (SYCL atomics on float are limited)
        if (input.dtype() == DType::Float32) {
            std::vector<float> h_out(total_size);
            std::vector<float> h_src(source.numel());
            std::vector<int64_t> h_idx(num_indices);
            queue.memcpy(h_out.data(), output.data_ptr(), total_size * sizeof(float));
            queue.memcpy(h_src.data(), source.data_ptr(), source.numel() * sizeof(float));
            queue.memcpy(h_idx.data(), indices.data_ptr(), num_indices * sizeof(int64_t)).wait();

            for (int64_t i = 0; i < num_indices; ++i) {
                int64_t target_idx = h_idx[i];
                if (target_idx < 0) target_idx += total_size;
                if (target_idx >= 0 && target_idx < total_size) {
                    h_out[target_idx] += h_src[i];
                }
            }
            queue.memcpy(const_cast<void*>(output.data_ptr()), h_out.data(), total_size * sizeof(float)).wait();
        } else if (input.dtype() == DType::Float64) {
            std::vector<double> h_out(total_size);
            std::vector<double> h_src(source.numel());
            std::vector<int64_t> h_idx(num_indices);
            queue.memcpy(h_out.data(), output.data_ptr(), total_size * sizeof(double));
            queue.memcpy(h_src.data(), source.data_ptr(), source.numel() * sizeof(double));
            queue.memcpy(h_idx.data(), indices.data_ptr(), num_indices * sizeof(int64_t)).wait();

            for (int64_t i = 0; i < num_indices; ++i) {
                int64_t target_idx = h_idx[i];
                if (target_idx < 0) target_idx += total_size;
                if (target_idx >= 0 && target_idx < total_size) {
                    h_out[target_idx] += h_src[i];
                }
            }
            queue.memcpy(const_cast<void*>(output.data_ptr()), h_out.data(), total_size * sizeof(double)).wait();
        } else if (input.dtype() == DType::Int32) {
            std::vector<int32_t> h_out(total_size);
            std::vector<int32_t> h_src(source.numel());
            std::vector<int64_t> h_idx(num_indices);
            queue.memcpy(h_out.data(), output.data_ptr(), total_size * sizeof(int32_t));
            queue.memcpy(h_src.data(), source.data_ptr(), source.numel() * sizeof(int32_t));
            queue.memcpy(h_idx.data(), indices.data_ptr(), num_indices * sizeof(int64_t)).wait();

            for (int64_t i = 0; i < num_indices; ++i) {
                int64_t target_idx = h_idx[i];
                if (target_idx < 0) target_idx += total_size;
                if (target_idx >= 0 && target_idx < total_size) {
                    h_out[target_idx] += h_src[i];
                }
            }
            queue.memcpy(const_cast<void*>(output.data_ptr()), h_out.data(), total_size * sizeof(int32_t)).wait();
        } else if (input.dtype() == DType::Int64) {
            std::vector<int64_t> h_out(total_size);
            std::vector<int64_t> h_src(source.numel());
            std::vector<int64_t> h_idx(num_indices);
            queue.memcpy(h_out.data(), output.data_ptr(), total_size * sizeof(int64_t));
            queue.memcpy(h_src.data(), source.data_ptr(), source.numel() * sizeof(int64_t));
            queue.memcpy(h_idx.data(), indices.data_ptr(), num_indices * sizeof(int64_t)).wait();

            for (int64_t i = 0; i < num_indices; ++i) {
                int64_t target_idx = h_idx[i];
                if (target_idx < 0) target_idx += total_size;
                if (target_idx >= 0 && target_idx < total_size) {
                    h_out[target_idx] += h_src[i];
                }
            }
            queue.memcpy(const_cast<void*>(output.data_ptr()), h_out.data(), total_size * sizeof(int64_t)).wait();
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
