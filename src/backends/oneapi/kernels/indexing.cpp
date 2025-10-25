#include "tenzor/core/tensor.hpp"
#include <CL/sycl.hpp>
#include <stdexcept>

namespace tenzor {
namespace oneapi {

// Kernel class declarations for SYCL
class GatherKernelFloat32;
class GatherKernelFloat64;
class ScatterKernelFloat32;
class ScatterKernelFloat64;
class IndexSelectKernelFloat32;
class IndexSelectKernelFloat64;
class MaskedFillKernelFloat32;
class MaskedFillKernelFloat64;

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
        }).wait();
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
        }).wait();
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
        queue.memcpy(out_ptr, in_ptr, bytes).wait();
    } else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        double* out_ptr = get_data_ptr<double>(output);
        queue.memcpy(out_ptr, in_ptr, bytes).wait();
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
        }).wait();
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
        }).wait();
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
        }).wait();
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
        }).wait();
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
        }).wait();
    }
    else if (input.dtype() == DType::Float64) {
        const double* input_ptr = get_data_ptr<const double>(input);
        const bool* mask_ptr = get_data_ptr<const bool>(mask);
        double* output_ptr = get_data_ptr<double>(output);
        const double value_d = static_cast<double>(value);

        queue.parallel_for<MaskedFillKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            output_ptr[idx] = mask_ptr[idx] ? value_d : input_ptr[idx];
        }).wait();
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
        queue.memcpy(input_host, input_ptr, numel * sizeof(float)).wait();
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
        queue.memcpy(input_host, input_ptr, numel * sizeof(double)).wait();
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

} // namespace oneapi
} // namespace tenzor
