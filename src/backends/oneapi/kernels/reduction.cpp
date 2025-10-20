#include "tenzor/core/tensor.hpp"
#include <CL/sycl.hpp>
#include <limits>
#include <numeric>
#include <algorithm>
#include <stdexcept>

namespace tenzor {
namespace oneapi {

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

// Sum reduction kernel
auto sum_kernel(const Tensor& input, int64_t dim, bool keepdim, sycl::queue& queue) -> Tensor {
    auto shape = input.shape();

    // Handle negative dimension
    if (dim < 0) {
        dim += shape.size();
    }

    if (dim < 0 || dim >= static_cast<int64_t>(shape.size())) {
        throw std::invalid_argument("Invalid dimension for sum reduction");
    }

    // Calculate output shape
    std::vector<int64_t> out_shape;
    for (size_t i = 0; i < shape.size(); ++i) {
        if (static_cast<int64_t>(i) == dim) {
            if (keepdim) {
                out_shape.push_back(1);
            }
        } else {
            out_shape.push_back(shape[i]);
        }
    }

    if (out_shape.empty()) {
        out_shape.push_back(1);
    }

    Tensor output(out_shape, input.dtype(), input.device());

    // Calculate reduction parameters
    const int64_t outer_size = std::accumulate(shape.begin(), shape.begin() + dim, 1LL, std::multiplies<>());
    const int64_t dim_size = shape[dim];
    const int64_t inner_size = std::accumulate(shape.begin() + dim + 1, shape.end(), 1LL, std::multiplies<>());

    if (input.dtype() == DType::Float32) {
        const float* in_ptr = get_data_ptr<const float>(input);
        float* out_ptr = get_data_ptr<float>(output);

        queue.parallel_for(sycl::range<2>(outer_size, inner_size), [=](sycl::id<2> idx) {
            const int64_t outer_idx = idx[0];
            const int64_t inner_idx = idx[1];
            const int64_t base_offset = outer_idx * dim_size * inner_size + inner_idx;

            float sum = 0.0f;
            for (int64_t d = 0; d < dim_size; ++d) {
                sum += in_ptr[base_offset + d * inner_size];
            }

            out_ptr[outer_idx * inner_size + inner_idx] = sum;
        }).wait();
    }
    else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        double* out_ptr = get_data_ptr<double>(output);

        queue.parallel_for(sycl::range<2>(outer_size, inner_size), [=](sycl::id<2> idx) {
            const int64_t outer_idx = idx[0];
            const int64_t inner_idx = idx[1];
            const int64_t base_offset = outer_idx * dim_size * inner_size + inner_idx;

            double sum = 0.0;
            for (int64_t d = 0; d < dim_size; ++d) {
                sum += in_ptr[base_offset + d * inner_size];
            }

            out_ptr[outer_idx * inner_size + inner_idx] = sum;
        }).wait();
    }
    else {
        throw std::runtime_error("Unsupported dtype for sum reduction");
    }

    return output;
}

// Mean reduction kernel
auto mean_kernel(const Tensor& input, int64_t dim, bool keepdim, sycl::queue& queue) -> Tensor {
    auto shape = input.shape();

    // Handle negative dimension
    if (dim < 0) {
        dim += shape.size();
    }

    if (dim < 0 || dim >= static_cast<int64_t>(shape.size())) {
        throw std::invalid_argument("Invalid dimension for mean reduction");
    }

    // Calculate output shape
    std::vector<int64_t> out_shape;
    for (size_t i = 0; i < shape.size(); ++i) {
        if (static_cast<int64_t>(i) == dim) {
            if (keepdim) {
                out_shape.push_back(1);
            }
        } else {
            out_shape.push_back(shape[i]);
        }
    }

    if (out_shape.empty()) {
        out_shape.push_back(1);
    }

    Tensor output(out_shape, input.dtype(), input.device());

    // Calculate reduction parameters
    const int64_t outer_size = std::accumulate(shape.begin(), shape.begin() + dim, 1LL, std::multiplies<>());
    const int64_t dim_size = shape[dim];
    const int64_t inner_size = std::accumulate(shape.begin() + dim + 1, shape.end(), 1LL, std::multiplies<>());

    if (input.dtype() == DType::Float32) {
        const float* in_ptr = get_data_ptr<const float>(input);
        float* out_ptr = get_data_ptr<float>(output);
        const float scale = 1.0f / static_cast<float>(dim_size);

        queue.parallel_for(sycl::range<2>(outer_size, inner_size), [=](sycl::id<2> idx) {
            const int64_t outer_idx = idx[0];
            const int64_t inner_idx = idx[1];
            const int64_t base_offset = outer_idx * dim_size * inner_size + inner_idx;

            float sum = 0.0f;
            for (int64_t d = 0; d < dim_size; ++d) {
                sum += in_ptr[base_offset + d * inner_size];
            }

            out_ptr[outer_idx * inner_size + inner_idx] = sum * scale;
        }).wait();
    }
    else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        double* out_ptr = get_data_ptr<double>(output);
        const double scale = 1.0 / static_cast<double>(dim_size);

        queue.parallel_for(sycl::range<2>(outer_size, inner_size), [=](sycl::id<2> idx) {
            const int64_t outer_idx = idx[0];
            const int64_t inner_idx = idx[1];
            const int64_t base_offset = outer_idx * dim_size * inner_size + inner_idx;

            double sum = 0.0;
            for (int64_t d = 0; d < dim_size; ++d) {
                sum += in_ptr[base_offset + d * inner_size];
            }

            out_ptr[outer_idx * inner_size + inner_idx] = sum * scale;
        }).wait();
    }
    else {
        throw std::runtime_error("Unsupported dtype for mean reduction");
    }

    return output;
}

// Max reduction kernel
auto max_kernel(const Tensor& input, int64_t dim, bool keepdim, sycl::queue& queue) -> Tensor {
    auto shape = input.shape();

    // Handle negative dimension
    if (dim < 0) {
        dim += shape.size();
    }

    if (dim < 0 || dim >= static_cast<int64_t>(shape.size())) {
        throw std::invalid_argument("Invalid dimension for max reduction");
    }

    // Calculate output shape
    std::vector<int64_t> out_shape;
    for (size_t i = 0; i < shape.size(); ++i) {
        if (static_cast<int64_t>(i) == dim) {
            if (keepdim) {
                out_shape.push_back(1);
            }
        } else {
            out_shape.push_back(shape[i]);
        }
    }

    if (out_shape.empty()) {
        out_shape.push_back(1);
    }

    Tensor output(out_shape, input.dtype(), input.device());

    // Calculate reduction parameters
    const int64_t outer_size = std::accumulate(shape.begin(), shape.begin() + dim, 1LL, std::multiplies<>());
    const int64_t dim_size = shape[dim];
    const int64_t inner_size = std::accumulate(shape.begin() + dim + 1, shape.end(), 1LL, std::multiplies<>());

    if (input.dtype() == DType::Float32) {
        const float* in_ptr = get_data_ptr<const float>(input);
        float* out_ptr = get_data_ptr<float>(output);

        queue.parallel_for(sycl::range<2>(outer_size, inner_size), [=](sycl::id<2> idx) {
            const int64_t outer_idx = idx[0];
            const int64_t inner_idx = idx[1];
            const int64_t base_offset = outer_idx * dim_size * inner_size + inner_idx;

            float max_val = -std::numeric_limits<float>::infinity();
            for (int64_t d = 0; d < dim_size; ++d) {
                float val = in_ptr[base_offset + d * inner_size];
                max_val = sycl::fmax(max_val, val);
            }

            out_ptr[outer_idx * inner_size + inner_idx] = max_val;
        }).wait();
    }
    else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        double* out_ptr = get_data_ptr<double>(output);

        queue.parallel_for(sycl::range<2>(outer_size, inner_size), [=](sycl::id<2> idx) {
            const int64_t outer_idx = idx[0];
            const int64_t inner_idx = idx[1];
            const int64_t base_offset = outer_idx * dim_size * inner_size + inner_idx;

            double max_val = -std::numeric_limits<double>::infinity();
            for (int64_t d = 0; d < dim_size; ++d) {
                double val = in_ptr[base_offset + d * inner_size];
                max_val = sycl::fmax(max_val, val);
            }

            out_ptr[outer_idx * inner_size + inner_idx] = max_val;
        }).wait();
    }
    else {
        throw std::runtime_error("Unsupported dtype for max reduction");
    }

    return output;
}

// Min reduction kernel
auto min_kernel(const Tensor& input, int64_t dim, bool keepdim, sycl::queue& queue) -> Tensor {
    auto shape = input.shape();

    // Handle negative dimension
    if (dim < 0) {
        dim += shape.size();
    }

    if (dim < 0 || dim >= static_cast<int64_t>(shape.size())) {
        throw std::invalid_argument("Invalid dimension for min reduction");
    }

    // Calculate output shape
    std::vector<int64_t> out_shape;
    for (size_t i = 0; i < shape.size(); ++i) {
        if (static_cast<int64_t>(i) == dim) {
            if (keepdim) {
                out_shape.push_back(1);
            }
        } else {
            out_shape.push_back(shape[i]);
        }
    }

    if (out_shape.empty()) {
        out_shape.push_back(1);
    }

    Tensor output(out_shape, input.dtype(), input.device());

    // Calculate reduction parameters
    const int64_t outer_size = std::accumulate(shape.begin(), shape.begin() + dim, 1LL, std::multiplies<>());
    const int64_t dim_size = shape[dim];
    const int64_t inner_size = std::accumulate(shape.begin() + dim + 1, shape.end(), 1LL, std::multiplies<>());

    if (input.dtype() == DType::Float32) {
        const float* in_ptr = get_data_ptr<const float>(input);
        float* out_ptr = get_data_ptr<float>(output);

        queue.parallel_for(sycl::range<2>(outer_size, inner_size), [=](sycl::id<2> idx) {
            const int64_t outer_idx = idx[0];
            const int64_t inner_idx = idx[1];
            const int64_t base_offset = outer_idx * dim_size * inner_size + inner_idx;

            float min_val = std::numeric_limits<float>::infinity();
            for (int64_t d = 0; d < dim_size; ++d) {
                float val = in_ptr[base_offset + d * inner_size];
                min_val = sycl::fmin(min_val, val);
            }

            out_ptr[outer_idx * inner_size + inner_idx] = min_val;
        }).wait();
    }
    else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        double* out_ptr = get_data_ptr<double>(output);

        queue.parallel_for(sycl::range<2>(outer_size, inner_size), [=](sycl::id<2> idx) {
            const int64_t outer_idx = idx[0];
            const int64_t inner_idx = idx[1];
            const int64_t base_offset = outer_idx * dim_size * inner_size + inner_idx;

            double min_val = std::numeric_limits<double>::infinity();
            for (int64_t d = 0; d < dim_size; ++d) {
                double val = in_ptr[base_offset + d * inner_size];
                min_val = sycl::fmin(min_val, val);
            }

            out_ptr[outer_idx * inner_size + inner_idx] = min_val;
        }).wait();
    }
    else {
        throw std::runtime_error("Unsupported dtype for min reduction");
    }

    return output;
}

} // namespace oneapi
} // namespace tenzor
