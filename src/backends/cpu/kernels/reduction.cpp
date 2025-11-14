#include "tenzor/core/tensor.hpp"
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <cmath>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace tenzor {
namespace cpu {

// Helper to compute output shape for reduction
static auto compute_reduction_shape(const std::vector<int64_t>& input_shape,
                                    int64_t dim,
                                    bool keepdim) -> std::vector<int64_t> {
    if (dim < 0) {
        // Full reduction - return scalar or [1,1,...] if keepdim
        if (keepdim) {
            return std::vector<int64_t>(input_shape.size(), 1);
        }
        return {};  // Scalar (0D tensor)
    }

    std::vector<int64_t> output_shape = input_shape;
    if (keepdim) {
        output_shape[dim] = 1;
    } else {
        output_shape.erase(output_shape.begin() + dim);
        // Keep empty shape for scalar result
    }
    return output_shape;
}

// Template for sum reduction with Kahan summation for floating point types
template<typename T>
auto sum_impl(const T* input_data, int64_t n) -> T {
    if (n == 0) return T(0);

    if constexpr (std::is_floating_point_v<T>) {
        // Kahan summation for improved numerical stability
        T sum = 0;
        T c = 0;  // Running compensation for lost low-order bits

        #pragma omp parallel if(n > 10000)
        {
            T local_sum = 0;
            T local_c = 0;

            #pragma omp for nowait
            for (int64_t i = 0; i < n; i++) {
                T y = input_data[i] - local_c;
                T t = local_sum + y;
                local_c = (t - local_sum) - y;
                local_sum = t;
            }

            #pragma omp critical
            {
                T y = local_sum - c;
                T t = sum + y;
                c = (t - sum) - y;
                sum = t;
            }
        }
        return sum;
    } else {
        // Simple sum for integer types
        T sum = 0;
        #pragma omp parallel for reduction(+:sum) if(n > 10000)
        for (int64_t i = 0; i < n; i++) {
            sum += input_data[i];
        }
        return sum;
    }
}

// Sum along a specific dimension
template<typename T>
void sum_along_dim(const T* input_data,
                   T* output_data,
                   const std::vector<int64_t>& input_shape,
                   const std::vector<int64_t>& input_strides,
                   int64_t dim) {
    const int64_t ndim = input_shape.size();
    const int64_t dim_size = input_shape[dim];

    // Compute output size
    int64_t output_size = 1;
    for (int64_t i = 0; i < ndim; i++) {
        if (i != dim) {
            output_size *= input_shape[i];
        }
    }

    // Initialize output to zero
    std::fill(output_data, output_data + output_size, T(0));

    // Reduction along dimension
    #pragma omp parallel for if(output_size > 1000)
    for (int64_t out_idx = 0; out_idx < output_size; out_idx++) {
        // Compute multi-dimensional index for output
        std::vector<int64_t> indices(ndim, 0);
        int64_t tmp = out_idx;
        int64_t out_i = 0;

        for (int64_t d = 0; d < ndim; d++) {
            if (d == dim) continue;
            int64_t size = input_shape[d];
            indices[d] = tmp % size;
            tmp /= size;
        }

        // Sum along the reduction dimension
        if constexpr (std::is_floating_point_v<T>) {
            T sum = 0, c = 0;  // Kahan summation
            for (int64_t i = 0; i < dim_size; i++) {
                indices[dim] = i;
                int64_t in_idx = 0;
                for (int64_t d = 0; d < ndim; d++) {
                    in_idx += indices[d] * input_strides[d];
                }
                T y = input_data[in_idx] - c;
                T t = sum + y;
                c = (t - sum) - y;
                sum = t;
            }
            output_data[out_idx] = sum;
        } else {
            T sum = 0;
            for (int64_t i = 0; i < dim_size; i++) {
                indices[dim] = i;
                int64_t in_idx = 0;
                for (int64_t d = 0; d < ndim; d++) {
                    in_idx += indices[d] * input_strides[d];
                }
                sum += input_data[in_idx];
            }
            output_data[out_idx] = sum;
        }
    }
}

auto sum_kernel(const Tensor& input, int64_t dim, bool keepdim) -> Tensor {
    const auto dtype = input.dtype();
    const auto& device = input.device();
    const auto& input_shape = input.shape();
    const auto& input_strides = input.strides();

    // Compute output shape
    auto output_shape = compute_reduction_shape(
        std::vector<int64_t>(input_shape.begin(), input_shape.end()),
        dim, keepdim
    );

    Tensor output(output_shape, dtype, device);

    // Dispatch based on dtype
    switch (dtype) {
        case DType::Float16: {
            auto* input_data = input.data<Float16>();
            auto* output_data = output.data<Float16>();

            if (dim < 0) {
                // Full reduction - compute in Float32 for precision
                const int64_t n = input.numel();
                float sum = 0.0f;
                float c = 0.0f;  // Kahan summation

                #pragma omp parallel if(n > 10000)
                {
                    float local_sum = 0.0f;
                    float local_c = 0.0f;

                    #pragma omp for nowait
                    for (int64_t i = 0; i < n; i++) {
                        float val = static_cast<float>(input_data[i]);
                        float y = val - local_c;
                        float t = local_sum + y;
                        local_c = (t - local_sum) - y;
                        local_sum = t;
                    }

                    #pragma omp critical
                    {
                        float y = local_sum - c;
                        float t = sum + y;
                        c = (t - sum) - y;
                        sum = t;
                    }
                }
                output_data[0] = Float16(sum);
            } else {
                // Dimensional reduction - compute in Float32
                const int64_t ndim = input_shape.size();
                const int64_t dim_size = input_shape[dim];

                int64_t output_size = 1;
                for (int64_t i = 0; i < ndim; i++) {
                    if (i != dim) {
                        output_size *= input_shape[i];
                    }
                }

                std::fill(output_data, output_data + output_size, Float16(0.0f));

                #pragma omp parallel for if(output_size > 1000)
                for (int64_t out_idx = 0; out_idx < output_size; out_idx++) {
                    std::vector<int64_t> indices(ndim, 0);
                    int64_t tmp = out_idx;

                    for (int64_t d = 0; d < ndim; d++) {
                        if (d == dim) continue;
                        int64_t size = input_shape[d];
                        indices[d] = tmp % size;
                        tmp /= size;
                    }

                    // Sum along dimension with Kahan summation
                    float sum = 0.0f, c = 0.0f;
                    for (int64_t i = 0; i < dim_size; i++) {
                        indices[dim] = i;
                        int64_t in_idx = 0;
                        for (int64_t d = 0; d < ndim; d++) {
                            in_idx += indices[d] * input_strides[d];
                        }
                        float val = static_cast<float>(input_data[in_idx]);
                        float y = val - c;
                        float t = sum + y;
                        c = (t - sum) - y;
                        sum = t;
                    }
                    output_data[out_idx] = Float16(sum);
                }
            }
            break;
        }
        case DType::Float32: {
            auto* input_data = input.data<float>();
            auto* output_data = output.data<float>();

            if (dim < 0) {
                // Full reduction
                output_data[0] = sum_impl(input_data, input.numel());
            } else {
                sum_along_dim(input_data, output_data,
                            std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                            std::vector<int64_t>(input_strides.begin(), input_strides.end()),
                            dim);
            }
            break;
        }
        case DType::Float64: {
            auto* input_data = input.data<double>();
            auto* output_data = output.data<double>();

            if (dim < 0) {
                output_data[0] = sum_impl(input_data, input.numel());
            } else {
                sum_along_dim(input_data, output_data,
                            std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                            std::vector<int64_t>(input_strides.begin(), input_strides.end()),
                            dim);
            }
            break;
        }
        case DType::Int32: {
            auto* input_data = input.data<int32_t>();
            auto* output_data = output.data<int32_t>();

            if (dim < 0) {
                output_data[0] = sum_impl(input_data, input.numel());
            } else {
                sum_along_dim(input_data, output_data,
                            std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                            std::vector<int64_t>(input_strides.begin(), input_strides.end()),
                            dim);
            }
            break;
        }
        case DType::Int64: {
            auto* input_data = input.data<int64_t>();
            auto* output_data = output.data<int64_t>();

            if (dim < 0) {
                output_data[0] = sum_impl(input_data, input.numel());
            } else {
                sum_along_dim(input_data, output_data,
                            std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                            std::vector<int64_t>(input_strides.begin(), input_strides.end()),
                            dim);
            }
            break;
        }
        default:
            throw std::runtime_error("sum: unsupported dtype");
    }

    return output;
}

auto mean_kernel(const Tensor& input, int64_t dim, bool keepdim) -> Tensor {
    const auto dtype = input.dtype();

    // Mean only supports floating point types
    if (dtype != DType::Float16 && dtype != DType::Float32 && dtype != DType::Float64) {
        throw std::runtime_error("mean: only Float16, Float32, and Float64 are supported");
    }

    // Compute sum first
    auto sum_result = sum_kernel(input, dim, keepdim);

    // Compute the count for averaging
    int64_t count;
    if (dim < 0) {
        count = input.numel();
    } else {
        count = input.shape()[dim];
    }

    // Divide sum by count
    if (dtype == DType::Float16) {
        auto* data = sum_result.data<Float16>();
        const float scale = 1.0f / static_cast<float>(count);
        const int64_t n = sum_result.numel();

        #pragma omp parallel for if(n > 10000)
        for (int64_t i = 0; i < n; i++) {
            data[i] = Float16(static_cast<float>(data[i]) * scale);
        }
    } else if (dtype == DType::Float32) {
        auto* data = sum_result.data<float>();
        const float scale = 1.0f / static_cast<float>(count);
        const int64_t n = sum_result.numel();

        #pragma omp parallel for if(n > 10000)
        for (int64_t i = 0; i < n; i++) {
            data[i] *= scale;
        }
    } else {  // Float64
        auto* data = sum_result.data<double>();
        const double scale = 1.0 / static_cast<double>(count);
        const int64_t n = sum_result.numel();

        #pragma omp parallel for if(n > 10000)
        for (int64_t i = 0; i < n; i++) {
            data[i] *= scale;
        }
    }

    return sum_result;
}

// Template for max reduction
template<typename T>
auto max_impl(const T* input_data, int64_t n) -> T {
    if (n == 0) throw std::runtime_error("max: input tensor is empty");

    T max_val = input_data[0];

    #pragma omp parallel if(n > 10000)
    {
        T local_max = input_data[0];

        #pragma omp for nowait
        for (int64_t i = 1; i < n; i++) {
            if (input_data[i] > local_max) {
                local_max = input_data[i];
            }
        }

        #pragma omp critical
        {
            if (local_max > max_val) {
                max_val = local_max;
            }
        }
    }

    return max_val;
}

// Max along a specific dimension
template<typename T>
void max_along_dim(const T* input_data,
                   T* output_data,
                   const std::vector<int64_t>& input_shape,
                   const std::vector<int64_t>& input_strides,
                   int64_t dim) {
    const int64_t ndim = input_shape.size();
    const int64_t dim_size = input_shape[dim];

    // Compute output size
    int64_t output_size = 1;
    for (int64_t i = 0; i < ndim; i++) {
        if (i != dim) {
            output_size *= input_shape[i];
        }
    }

    // Reduction along dimension
    #pragma omp parallel for if(output_size > 1000)
    for (int64_t out_idx = 0; out_idx < output_size; out_idx++) {
        std::vector<int64_t> indices(ndim, 0);
        int64_t tmp = out_idx;

        for (int64_t d = 0; d < ndim; d++) {
            if (d == dim) continue;
            int64_t size = input_shape[d];
            indices[d] = tmp % size;
            tmp /= size;
        }

        // Find max along the reduction dimension
        indices[dim] = 0;
        int64_t in_idx = 0;
        for (int64_t d = 0; d < ndim; d++) {
            in_idx += indices[d] * input_strides[d];
        }
        T max_val = input_data[in_idx];

        for (int64_t i = 1; i < dim_size; i++) {
            indices[dim] = i;
            in_idx = 0;
            for (int64_t d = 0; d < ndim; d++) {
                in_idx += indices[d] * input_strides[d];
            }
            if (input_data[in_idx] > max_val) {
                max_val = input_data[in_idx];
            }
        }
        output_data[out_idx] = max_val;
    }
}

auto max_kernel(const Tensor& input, int64_t dim, bool keepdim) -> Tensor {
    const auto dtype = input.dtype();
    const auto& device = input.device();
    const auto& input_shape = input.shape();
    const auto& input_strides = input.strides();

    auto output_shape = compute_reduction_shape(
        std::vector<int64_t>(input_shape.begin(), input_shape.end()),
        dim, keepdim
    );

    Tensor output(output_shape, dtype, device);

    switch (dtype) {
        case DType::Float32: {
            auto* input_data = input.data<float>();
            auto* output_data = output.data<float>();

            if (dim < 0) {
                output_data[0] = max_impl(input_data, input.numel());
            } else {
                max_along_dim(input_data, output_data,
                            std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                            std::vector<int64_t>(input_strides.begin(), input_strides.end()),
                            dim);
            }
            break;
        }
        case DType::Float64: {
            auto* input_data = input.data<double>();
            auto* output_data = output.data<double>();

            if (dim < 0) {
                output_data[0] = max_impl(input_data, input.numel());
            } else {
                max_along_dim(input_data, output_data,
                            std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                            std::vector<int64_t>(input_strides.begin(), input_strides.end()),
                            dim);
            }
            break;
        }
        case DType::Int32: {
            auto* input_data = input.data<int32_t>();
            auto* output_data = output.data<int32_t>();

            if (dim < 0) {
                output_data[0] = max_impl(input_data, input.numel());
            } else {
                max_along_dim(input_data, output_data,
                            std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                            std::vector<int64_t>(input_strides.begin(), input_strides.end()),
                            dim);
            }
            break;
        }
        case DType::Int64: {
            auto* input_data = input.data<int64_t>();
            auto* output_data = output.data<int64_t>();

            if (dim < 0) {
                output_data[0] = max_impl(input_data, input.numel());
            } else {
                max_along_dim(input_data, output_data,
                            std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                            std::vector<int64_t>(input_strides.begin(), input_strides.end()),
                            dim);
            }
            break;
        }
        default:
            throw std::runtime_error("max: unsupported dtype");
    }

    return output;
}

// Template for min reduction
template<typename T>
auto min_impl(const T* input_data, int64_t n) -> T {
    if (n == 0) throw std::runtime_error("min: input tensor is empty");

    T min_val = input_data[0];

    #pragma omp parallel if(n > 10000)
    {
        T local_min = input_data[0];

        #pragma omp for nowait
        for (int64_t i = 1; i < n; i++) {
            if (input_data[i] < local_min) {
                local_min = input_data[i];
            }
        }

        #pragma omp critical
        {
            if (local_min < min_val) {
                min_val = local_min;
            }
        }
    }

    return min_val;
}

// Min along a specific dimension
template<typename T>
void min_along_dim(const T* input_data,
                   T* output_data,
                   const std::vector<int64_t>& input_shape,
                   const std::vector<int64_t>& input_strides,
                   int64_t dim) {
    const int64_t ndim = input_shape.size();
    const int64_t dim_size = input_shape[dim];

    // Compute output size
    int64_t output_size = 1;
    for (int64_t i = 0; i < ndim; i++) {
        if (i != dim) {
            output_size *= input_shape[i];
        }
    }

    // Reduction along dimension
    #pragma omp parallel for if(output_size > 1000)
    for (int64_t out_idx = 0; out_idx < output_size; out_idx++) {
        std::vector<int64_t> indices(ndim, 0);
        int64_t tmp = out_idx;

        for (int64_t d = 0; d < ndim; d++) {
            if (d == dim) continue;
            int64_t size = input_shape[d];
            indices[d] = tmp % size;
            tmp /= size;
        }

        // Find min along the reduction dimension
        indices[dim] = 0;
        int64_t in_idx = 0;
        for (int64_t d = 0; d < ndim; d++) {
            in_idx += indices[d] * input_strides[d];
        }
        T min_val = input_data[in_idx];

        for (int64_t i = 1; i < dim_size; i++) {
            indices[dim] = i;
            in_idx = 0;
            for (int64_t d = 0; d < ndim; d++) {
                in_idx += indices[d] * input_strides[d];
            }
            if (input_data[in_idx] < min_val) {
                min_val = input_data[in_idx];
            }
        }
        output_data[out_idx] = min_val;
    }
}

auto min_kernel(const Tensor& input, int64_t dim, bool keepdim) -> Tensor {
    const auto dtype = input.dtype();
    const auto& device = input.device();
    const auto& input_shape = input.shape();
    const auto& input_strides = input.strides();

    auto output_shape = compute_reduction_shape(
        std::vector<int64_t>(input_shape.begin(), input_shape.end()),
        dim, keepdim
    );

    Tensor output(output_shape, dtype, device);

    switch (dtype) {
        case DType::Float32: {
            auto* input_data = input.data<float>();
            auto* output_data = output.data<float>();

            if (dim < 0) {
                output_data[0] = min_impl(input_data, input.numel());
            } else {
                min_along_dim(input_data, output_data,
                            std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                            std::vector<int64_t>(input_strides.begin(), input_strides.end()),
                            dim);
            }
            break;
        }
        case DType::Float64: {
            auto* input_data = input.data<double>();
            auto* output_data = output.data<double>();

            if (dim < 0) {
                output_data[0] = min_impl(input_data, input.numel());
            } else {
                min_along_dim(input_data, output_data,
                            std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                            std::vector<int64_t>(input_strides.begin(), input_strides.end()),
                            dim);
            }
            break;
        }
        case DType::Int32: {
            auto* input_data = input.data<int32_t>();
            auto* output_data = output.data<int32_t>();

            if (dim < 0) {
                output_data[0] = min_impl(input_data, input.numel());
            } else {
                min_along_dim(input_data, output_data,
                            std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                            std::vector<int64_t>(input_strides.begin(), input_strides.end()),
                            dim);
            }
            break;
        }
        case DType::Int64: {
            auto* input_data = input.data<int64_t>();
            auto* output_data = output.data<int64_t>();

            if (dim < 0) {
                output_data[0] = min_impl(input_data, input.numel());
            } else {
                min_along_dim(input_data, output_data,
                            std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                            std::vector<int64_t>(input_strides.begin(), input_strides.end()),
                            dim);
            }
            break;
        }
        default:
            throw std::runtime_error("min: unsupported dtype");
    }

    return output;
}

// Template for argmax reduction - returns index of maximum value
template<typename T>
auto argmax_impl(const T* input_data, int64_t n) -> int64_t {
    if (n == 0) throw std::runtime_error("argmax: input tensor is empty");

    int64_t max_idx = 0;
    T max_val = input_data[0];

    for (int64_t i = 1; i < n; i++) {
        if (input_data[i] > max_val) {
            max_val = input_data[i];
            max_idx = i;
        }
    }

    return max_idx;
}

// Argmax along a specific dimension
template<typename T>
void argmax_along_dim(const T* input_data,
                      int64_t* output_data,
                      const std::vector<int64_t>& input_shape,
                      const std::vector<int64_t>& input_strides,
                      int64_t dim) {
    const int64_t ndim = input_shape.size();
    const int64_t dim_size = input_shape[dim];

    // Compute output size
    int64_t output_size = 1;
    for (int64_t i = 0; i < ndim; i++) {
        if (i != dim) {
            output_size *= input_shape[i];
        }
    }

    // Find argmax along dimension
    #pragma omp parallel for if(output_size > 1000)
    for (int64_t out_idx = 0; out_idx < output_size; out_idx++) {
        std::vector<int64_t> indices(ndim, 0);
        int64_t tmp = out_idx;

        for (int64_t d = 0; d < ndim; d++) {
            if (d == dim) continue;
            int64_t size = input_shape[d];
            indices[d] = tmp % size;
            tmp /= size;
        }

        // Find index of max along the reduction dimension
        indices[dim] = 0;
        int64_t in_idx = 0;
        for (int64_t d = 0; d < ndim; d++) {
            in_idx += indices[d] * input_strides[d];
        }

        T max_val = input_data[in_idx];
        int64_t max_idx = 0;

        for (int64_t i = 1; i < dim_size; i++) {
            indices[dim] = i;
            in_idx = 0;
            for (int64_t d = 0; d < ndim; d++) {
                in_idx += indices[d] * input_strides[d];
            }

            if (input_data[in_idx] > max_val) {
                max_val = input_data[in_idx];
                max_idx = i;
            }
        }

        output_data[out_idx] = max_idx;
    }
}

auto argmax_kernel(const Tensor& input, int64_t dim, bool keepdim) -> Tensor {
    // Argmax always returns Int64 indices
    auto input_shape_span = input.shape();
    std::vector<int64_t> input_shape_vec(input_shape_span.begin(), input_shape_span.end());
    auto output_shape = compute_reduction_shape(input_shape_vec, dim, keepdim);
    auto output = Tensor(output_shape, DType::Int64, input.device());

    auto input_shape = input.shape();
    auto input_strides = input.strides();

    // Output is always int64
    auto* output_data = output.data<int64_t>();

    // Handle different input dtypes
    switch (input.dtype()) {
        case DType::Float32: {
            auto* input_data = input.data<float>();
            if (dim < 0) {
                output_data[0] = argmax_impl(input_data, input.numel());
            } else {
                argmax_along_dim(input_data, output_data,
                               std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                               std::vector<int64_t>(input_strides.begin(), input_strides.end()),
                               dim);
            }
            break;
        }
        case DType::Float64: {
            auto* input_data = input.data<double>();
            if (dim < 0) {
                output_data[0] = argmax_impl(input_data, input.numel());
            } else {
                argmax_along_dim(input_data, output_data,
                               std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                               std::vector<int64_t>(input_strides.begin(), input_strides.end()),
                               dim);
            }
            break;
        }
        case DType::Int32: {
            auto* input_data = input.data<int32_t>();
            if (dim < 0) {
                output_data[0] = argmax_impl(input_data, input.numel());
            } else {
                argmax_along_dim(input_data, output_data,
                               std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                               std::vector<int64_t>(input_strides.begin(), input_strides.end()),
                               dim);
            }
            break;
        }
        case DType::Int64: {
            auto* input_data = input.data<int64_t>();
            if (dim < 0) {
                output_data[0] = argmax_impl(input_data, input.numel());
            } else {
                argmax_along_dim(input_data, output_data,
                               std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                               std::vector<int64_t>(input_strides.begin(), input_strides.end()),
                               dim);
            }
            break;
        }
        default:
            throw std::runtime_error("argmax: unsupported dtype");
    }

    return output;
}

// Argmin implementation - find index of minimum value
template<typename T>
auto argmin_impl(const T* input_data, int64_t n) -> int64_t {
    if (n == 0) throw std::runtime_error("argmin: input tensor is empty");

    int64_t min_idx = 0;
    T min_val = input_data[0];

    for (int64_t i = 1; i < n; i++) {
        if (input_data[i] < min_val) {
            min_val = input_data[i];
            min_idx = i;
        }
    }

    return min_idx;
}

// Argmin along a specific dimension
template<typename T>
void argmin_along_dim(const T* input_data,
                      int64_t* output_data,
                      const std::vector<int64_t>& input_shape,
                      const std::vector<int64_t>& input_strides,
                      int64_t dim) {
    const int64_t ndim = input_shape.size();
    const int64_t dim_size = input_shape[dim];

    // Compute output size
    int64_t output_size = 1;
    for (int64_t i = 0; i < ndim; i++) {
        if (i != dim) {
            output_size *= input_shape[i];
        }
    }

    // Find argmin along dimension
    #pragma omp parallel for if(output_size > 1000)
    for (int64_t out_idx = 0; out_idx < output_size; out_idx++) {
        std::vector<int64_t> indices(ndim, 0);
        int64_t tmp = out_idx;

        for (int64_t d = 0; d < ndim; d++) {
            if (d == dim) continue;
            int64_t size = input_shape[d];
            indices[d] = tmp % size;
            tmp /= size;
        }

        // Find index of min along the reduction dimension
        indices[dim] = 0;
        int64_t in_idx = 0;
        for (int64_t d = 0; d < ndim; d++) {
            in_idx += indices[d] * input_strides[d];
        }

        T min_val = input_data[in_idx];
        int64_t min_idx = 0;

        for (int64_t i = 1; i < dim_size; i++) {
            indices[dim] = i;
            in_idx = 0;
            for (int64_t d = 0; d < ndim; d++) {
                in_idx += indices[d] * input_strides[d];
            }

            if (input_data[in_idx] < min_val) {
                min_val = input_data[in_idx];
                min_idx = i;
            }
        }

        output_data[out_idx] = min_idx;
    }
}

// Argmin kernel - returns indices of minimum values
auto argmin_kernel(const Tensor& input, int64_t dim, bool keepdim) -> Tensor {
    // Argmin always returns Int64 indices
    auto input_shape_span = input.shape();
    std::vector<int64_t> input_shape_vec(input_shape_span.begin(), input_shape_span.end());
    auto output_shape = compute_reduction_shape(input_shape_vec, dim, keepdim);
    auto output = Tensor(output_shape, DType::Int64, input.device());

    auto input_shape = input.shape();
    auto input_strides = input.strides();

    // Output is always int64
    auto* output_data = output.data<int64_t>();

    // Handle different input dtypes
    switch (input.dtype()) {
        case DType::Float32: {
            auto* input_data = input.data<float>();
            if (dim < 0) {
                output_data[0] = argmin_impl(input_data, input.numel());
            } else {
                argmin_along_dim(input_data, output_data,
                               std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                               std::vector<int64_t>(input_strides.begin(), input_strides.end()),
                               dim);
            }
            break;
        }
        case DType::Float64: {
            auto* input_data = input.data<double>();
            if (dim < 0) {
                output_data[0] = argmin_impl(input_data, input.numel());
            } else {
                argmin_along_dim(input_data, output_data,
                               std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                               std::vector<int64_t>(input_strides.begin(), input_strides.end()),
                               dim);
            }
            break;
        }
        case DType::Int32: {
            auto* input_data = input.data<int32_t>();
            if (dim < 0) {
                output_data[0] = argmin_impl(input_data, input.numel());
            } else {
                argmin_along_dim(input_data, output_data,
                               std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                               std::vector<int64_t>(input_strides.begin(), input_strides.end()),
                               dim);
            }
            break;
        }
        case DType::Int64: {
            auto* input_data = input.data<int64_t>();
            if (dim < 0) {
                output_data[0] = argmin_impl(input_data, input.numel());
            } else {
                argmin_along_dim(input_data, output_data,
                               std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                               std::vector<int64_t>(input_strides.begin(), input_strides.end()),
                               dim);
            }
            break;
        }
        default:
            throw std::runtime_error("argmin: unsupported dtype");
    }

    return output;
}

// Argsort kernel - returns indices that would sort the input
template<typename T>
auto argsort_impl(const T* data, int64_t n, bool descending) -> std::vector<int64_t> {
    // Create index array
    std::vector<int64_t> indices(n);
    for (int64_t i = 0; i < n; ++i) {
        indices[i] = i;
    }

    // Sort indices based on data values
    if (descending) {
        std::sort(indices.begin(), indices.end(),
                 [data](int64_t a, int64_t b) { return data[a] > data[b]; });
    } else {
        std::sort(indices.begin(), indices.end(),
                 [data](int64_t a, int64_t b) { return data[a] < data[b]; });
    }

    return indices;
}

template<typename T>
void argsort_along_dim(const T* input_data, int64_t* output_data,
                      const std::vector<int64_t>& shape,
                      const std::vector<int64_t>& strides,
                      int64_t dim, bool descending) {
    const int64_t ndim = shape.size();
    const int64_t dim_size = shape[dim];

    // Compute total number of elements
    int64_t total_elems = 1;
    for (auto s : shape) total_elems *= s;

    // Compute size of inner dimensions (after dim)
    int64_t inner_size = 1;
    for (int64_t d = dim + 1; d < ndim; ++d) {
        inner_size *= shape[d];
    }

    // Compute size of outer dimensions (before dim)
    int64_t outer_size = total_elems / (dim_size * inner_size);

    // For each outer x inner combination, sort along dim
    #pragma omp parallel for if(outer_size * inner_size > 1000)
    for (int64_t outer = 0; outer < outer_size; ++outer) {
        for (int64_t inner = 0; inner < inner_size; ++inner) {
            // Collect values along the dimension
            std::vector<T> values(dim_size);
            for (int64_t i = 0; i < dim_size; ++i) {
                int64_t offset = outer * dim_size * inner_size + i * inner_size + inner;
                values[i] = input_data[offset];
            }

            // Get sorted indices
            auto sorted_indices = argsort_impl(values.data(), dim_size, descending);

            // Write sorted indices to output
            for (int64_t i = 0; i < dim_size; ++i) {
                int64_t offset = outer * dim_size * inner_size + i * inner_size + inner;
                output_data[offset] = sorted_indices[i];
            }
        }
    }
}

auto argsort_kernel(const Tensor& input, int64_t dim, bool descending) -> Tensor {
    const int64_t ndim = input.ndim();

    // Normalize dimension
    if (dim < 0) {
        dim += ndim;
    }
    if (dim < 0 || dim >= ndim) {
        throw std::out_of_range("argsort: dimension out of range");
    }

    // Output has same shape as input but with Int64 dtype
    std::vector<int64_t> output_shape(input.shape().begin(), input.shape().end());
    Tensor output(output_shape, DType::Int64, input.device());
    int64_t* output_data = output.data<int64_t>();

    const auto& input_shape = input.shape();
    const auto& input_strides = input.strides();

    // Dispatch based on input dtype
    switch (input.dtype()) {
        case DType::Float32: {
            auto* input_data = input.data<float>();
            argsort_along_dim(input_data, output_data,
                            std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                            std::vector<int64_t>(input_strides.begin(), input_strides.end()),
                            dim, descending);
            break;
        }
        case DType::Float64: {
            auto* input_data = input.data<double>();
            argsort_along_dim(input_data, output_data,
                            std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                            std::vector<int64_t>(input_strides.begin(), input_strides.end()),
                            dim, descending);
            break;
        }
        case DType::Int32: {
            auto* input_data = input.data<int32_t>();
            argsort_along_dim(input_data, output_data,
                            std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                            std::vector<int64_t>(input_strides.begin(), input_strides.end()),
                            dim, descending);
            break;
        }
        case DType::Int64: {
            auto* input_data = input.data<int64_t>();
            argsort_along_dim(input_data, output_data,
                            std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                            std::vector<int64_t>(input_strides.begin(), input_strides.end()),
                            dim, descending);
            break;
        }
        default:
            throw std::runtime_error("argsort: unsupported dtype");
    }

    return output;
}

// ============================================================================
// Product, Variance, and Standard Deviation operations
// ============================================================================

// Template for product reduction
template<typename T>
auto prod_impl(const T* input_data, int64_t n) -> T {
    if (n == 0) return T(1);  // Empty product is 1

    T result = T(1);
    #pragma omp parallel for reduction(*:result) if(n > 10000)
    for (int64_t i = 0; i < n; i++) {
        result *= input_data[i];
    }
    return result;
}

// Product along a specific dimension
template<typename T>
void prod_along_dim(const T* input_data,
                    T* output_data,
                    const std::vector<int64_t>& input_shape,
                    const std::vector<int64_t>& input_strides,
                    int64_t dim) {
    const int64_t ndim = input_shape.size();
    const int64_t dim_size = input_shape[dim];

    int64_t output_size = 1;
    for (int64_t i = 0; i < ndim; i++) {
        if (i != dim) {
            output_size *= input_shape[i];
        }
    }

    // Initialize output to 1 (identity for multiplication)
    std::fill(output_data, output_data + output_size, T(1));

    #pragma omp parallel for if(output_size > 1000)
    for (int64_t out_idx = 0; out_idx < output_size; out_idx++) {
        // Compute indices for this output position
        std::vector<int64_t> indices(ndim);
        int64_t tmp = out_idx;
        for (int64_t d = ndim - 1; d >= 0; d--) {
            if (d == dim) continue;
            indices[d] = tmp % input_shape[d];
            tmp /= input_shape[d];
        }

        // Product along dimension
        T prod_val = T(1);
        for (int64_t i = 0; i < dim_size; i++) {
            indices[dim] = i;
            int64_t in_idx = 0;
            for (int64_t d = 0; d < ndim; d++) {
                in_idx += indices[d] * input_strides[d];
            }
            prod_val *= input_data[in_idx];
        }
        output_data[out_idx] = prod_val;
    }
}

// Public API for product
auto prod_kernel(const Tensor& input, int64_t dim, bool keepdim) -> Tensor {
    auto shape_span = input.shape();
    std::vector<int64_t> input_shape(shape_span.begin(), shape_span.end());
    auto output_shape = compute_reduction_shape(input_shape, dim, keepdim);

    Tensor output(output_shape, input.dtype(), input.device());

    switch (input.dtype()) {
        case DType::Float32: {
            auto* input_data = input.data<float>();
            auto* output_data = output.data<float>();
            if (dim < 0) {
                output_data[0] = prod_impl(input_data, input.numel());
            } else {
                auto strides_span = input.strides();
                std::vector<int64_t> input_strides(strides_span.begin(), strides_span.end());
                prod_along_dim(input_data, output_data, input_shape, input_strides, dim);
            }
            break;
        }
        case DType::Float64: {
            auto* input_data = input.data<double>();
            auto* output_data = output.data<double>();
            if (dim < 0) {
                output_data[0] = prod_impl(input_data, input.numel());
            } else {
                auto strides_span = input.strides();
                std::vector<int64_t> input_strides(strides_span.begin(), strides_span.end());
                prod_along_dim(input_data, output_data, input_shape, input_strides, dim);
            }
            break;
        }
        default:
            throw std::runtime_error("prod: unsupported dtype");
    }

    return output;
}

// Variance using two-pass algorithm for numerical stability
auto var_kernel(const Tensor& input, int64_t dim, bool keepdim, int64_t correction) -> Tensor {
    auto shape_span = input.shape();
    std::vector<int64_t> input_shape(shape_span.begin(), shape_span.end());
    auto output_shape = compute_reduction_shape(input_shape, dim, keepdim);

    Tensor output(output_shape, input.dtype(), input.device());

    if (dim != -1) {
        throw std::runtime_error("var: only full reduction (dim=-1) is currently supported for CPU");
    }

    const int64_t n = input.numel();
    if (n == 0) {
        throw std::runtime_error("var: input tensor is empty");
    }

    switch (input.dtype()) {
        case DType::Float32: {
            auto* input_data = input.data<float>();
            auto* output_data = output.data<float>();

            // Pass 1: Compute mean
            float mean = sum_impl(input_data, n) / static_cast<float>(n);

            // Pass 2: Compute variance
            float var_sum = 0.0f;
            #pragma omp parallel for reduction(+:var_sum) if(n > 10000)
            for (int64_t i = 0; i < n; i++) {
                float diff = input_data[i] - mean;
                var_sum += diff * diff;
            }

            int64_t divisor = n - correction;
            if (divisor <= 0) divisor = 1;
            output_data[0] = var_sum / static_cast<float>(divisor);
            break;
        }
        case DType::Float64: {
            auto* input_data = input.data<double>();
            auto* output_data = output.data<double>();

            // Pass 1: Compute mean
            double mean = sum_impl(input_data, n) / static_cast<double>(n);

            // Pass 2: Compute variance
            double var_sum = 0.0;
            #pragma omp parallel for reduction(+:var_sum) if(n > 10000)
            for (int64_t i = 0; i < n; i++) {
                double diff = input_data[i] - mean;
                var_sum += diff * diff;
            }

            int64_t divisor = n - correction;
            if (divisor <= 0) divisor = 1;
            output_data[0] = var_sum / static_cast<double>(divisor);
            break;
        }
        default:
            throw std::runtime_error("var: unsupported dtype");
    }

    return output;
}

// Standard deviation (sqrt of variance)
auto std_kernel(const Tensor& input, int64_t dim, bool keepdim, int64_t correction) -> Tensor {
    auto var_result = var_kernel(input, dim, keepdim, correction);

    // Apply sqrt element-wise
    auto shape_span = var_result.shape();
    std::vector<int64_t> output_shape(shape_span.begin(), shape_span.end());
    Tensor output(output_shape, var_result.dtype(), var_result.device());

    const int64_t n = var_result.numel();

    switch (var_result.dtype()) {
        case DType::Float32: {
            auto* var_data = var_result.data<float>();
            auto* output_data = output.data<float>();
            #pragma omp parallel for if(n > 10000)
            for (int64_t i = 0; i < n; i++) {
                output_data[i] = std::sqrt(var_data[i]);
            }
            break;
        }
        case DType::Float64: {
            auto* var_data = var_result.data<double>();
            auto* output_data = output.data<double>();
            #pragma omp parallel for if(n > 10000)
            for (int64_t i = 0; i < n; i++) {
                output_data[i] = std::sqrt(var_data[i]);
            }
            break;
        }
        default:
            throw std::runtime_error("std: unsupported dtype");
    }

    return output;
}

// Norm operation - compute Lp norm
auto norm_kernel(const Tensor& input, float p, int64_t dim, bool keepdim) -> Tensor {
    auto shape_span = input.shape();
    std::vector<int64_t> input_shape(shape_span.begin(), shape_span.end());

    if (dim != -1) {
        throw std::runtime_error("norm: only full reduction (dim=-1) is currently supported for CPU");
    }

    auto output_shape = compute_reduction_shape(input_shape, dim, keepdim);
    Tensor output(output_shape, input.dtype(), input.device());

    const int64_t n = input.numel();
    if (n == 0) {
        throw std::runtime_error("norm: input tensor is empty");
    }

    switch (input.dtype()) {
        case DType::Float32: {
            auto* input_data = input.data<float>();
            auto* output_data = output.data<float>();

            float norm_value = 0.0f;

            if (p == 1.0f) {
                // L1 norm: sum of absolute values
                #pragma omp parallel for reduction(+:norm_value) if(n > 10000)
                for (int64_t i = 0; i < n; i++) {
                    norm_value += std::abs(input_data[i]);
                }
            } else if (p == 2.0f) {
                // L2 norm: sqrt of sum of squares
                #pragma omp parallel for reduction(+:norm_value) if(n > 10000)
                for (int64_t i = 0; i < n; i++) {
                    norm_value += input_data[i] * input_data[i];
                }
                norm_value = std::sqrt(norm_value);
            } else if (std::isinf(p)) {
                // L-inf norm: max absolute value
                norm_value = std::abs(input_data[0]);
                for (int64_t i = 1; i < n; i++) {
                    float abs_val = std::abs(input_data[i]);
                    if (abs_val > norm_value) {
                        norm_value = abs_val;
                    }
                }
            } else {
                // General Lp norm: (sum(|x|^p))^(1/p)
                #pragma omp parallel for reduction(+:norm_value) if(n > 10000)
                for (int64_t i = 0; i < n; i++) {
                    norm_value += std::pow(std::abs(input_data[i]), p);
                }
                norm_value = std::pow(norm_value, 1.0f / p);
            }

            output_data[0] = norm_value;
            break;
        }
        case DType::Float64: {
            auto* input_data = input.data<double>();
            auto* output_data = output.data<double>();

            double norm_value = 0.0;

            if (p == 1.0) {
                // L1 norm: sum of absolute values
                #pragma omp parallel for reduction(+:norm_value) if(n > 10000)
                for (int64_t i = 0; i < n; i++) {
                    norm_value += std::abs(input_data[i]);
                }
            } else if (p == 2.0) {
                // L2 norm: sqrt of sum of squares
                #pragma omp parallel for reduction(+:norm_value) if(n > 10000)
                for (int64_t i = 0; i < n; i++) {
                    norm_value += input_data[i] * input_data[i];
                }
                norm_value = std::sqrt(norm_value);
            } else if (std::isinf(p)) {
                // L-inf norm: max absolute value
                norm_value = std::abs(input_data[0]);
                for (int64_t i = 1; i < n; i++) {
                    double abs_val = std::abs(input_data[i]);
                    if (abs_val > norm_value) {
                        norm_value = abs_val;
                    }
                }
            } else {
                // General Lp norm: (sum(|x|^p))^(1/p)
                #pragma omp parallel for reduction(+:norm_value) if(n > 10000)
                for (int64_t i = 0; i < n; i++) {
                    norm_value += std::pow(std::abs(input_data[i]), p);
                }
                norm_value = std::pow(norm_value, 1.0 / p);
            }

            output_data[0] = norm_value;
            break;
        }
        default:
            throw std::runtime_error("norm: unsupported dtype");
    }

    return output;
}

} // namespace cpu
} // namespace tenzor
