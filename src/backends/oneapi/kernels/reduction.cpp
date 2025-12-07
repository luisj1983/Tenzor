#include "tenzor/core/tensor.hpp"
#include <sycl/sycl.hpp>
#include <limits>
#include <numeric>
#include <algorithm>
#include <stdexcept>

// Forward declaration for contiguous kernel
namespace tenzor {
namespace oneapi {
    auto contiguous_kernel(const Tensor& input, sycl::queue& queue) -> Tensor;
}
}

namespace tenzor {
namespace oneapi {

// Kernel class declarations for SYCL
class SumKernelFloat32;
class SumKernelFloat64;
class MeanKernelFloat32;
class MeanKernelFloat64;
class MaxKernelFloat32;
class MaxKernelFloat64;
class MinKernelFloat32;
class MinKernelFloat64;
class ArgmaxKernelFloat32;
class ArgmaxKernelFloat64;
class ArgminKernelFloat32;
class ArgminKernelFloat64;

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

// Helper function to compute output shape for reduction
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
    }
    return output_shape;
}

// Sum reduction kernel
// IMPORTANT: Must ensure contiguous input for direct memory access
auto sum_kernel(const Tensor& input, int64_t dim, bool keepdim, sycl::queue& queue) -> Tensor {
    // Ensure input is contiguous for correct memory access
    Tensor in_cont = input.is_contiguous() ? input : contiguous_kernel(input, queue);

    auto shape = in_cont.shape();
    std::vector<int64_t> shape_vec(shape.begin(), shape.end());

    // Check if this is a full reduction (dim < 0 means reduce all dimensions)
    bool is_full_reduction = (dim < 0);

    if (!is_full_reduction) {
        // Normalize dimension for partial reduction
        if (dim < 0) {
            dim += shape.size();
        }
        if (dim < 0 || dim >= static_cast<int64_t>(shape.size())) {
            throw std::invalid_argument("Invalid dimension for sum reduction");
        }
    }

    // Calculate output shape
    auto out_shape = compute_reduction_shape(shape_vec, is_full_reduction ? -1 : dim, keepdim);
    Tensor output(out_shape, in_cont.dtype(), in_cont.device());

    if (is_full_reduction) {
        // Full reduction: sum all elements
        const int64_t total_size = in_cont.numel();

        // Handle empty tensor: sum of empty set is 0 (additive identity)
        if (total_size == 0) {
            if (in_cont.dtype() == DType::Float32) {
                float* out_ptr = get_data_ptr<float>(output);
                queue.single_task([=]() { out_ptr[0] = 0.0f; }).wait();
            } else if (in_cont.dtype() == DType::Float64) {
                double* out_ptr = get_data_ptr<double>(output);
                queue.single_task([=]() { out_ptr[0] = 0.0; }).wait();
            }
            return output;
        }

        if (in_cont.dtype() == DType::Float32) {
            const float* in_ptr = get_data_ptr<const float>(in_cont);
            float* out_ptr = get_data_ptr<float>(output);

            // Use parallel_for with reduction
            auto sum_buf = sycl::malloc_shared<float>(1, queue);
            sum_buf[0] = 0.0f;

            queue.parallel_for(sycl::range<1>(total_size), sycl::reduction(sum_buf, sycl::plus<float>()),
                              [=](sycl::id<1> idx, auto& sum) {
                sum += in_ptr[idx];
            }).wait();

            out_ptr[0] = sum_buf[0];
            queue.wait();
            sycl::free(sum_buf, queue);
        }
        else if (in_cont.dtype() == DType::Float64) {
            const double* in_ptr = get_data_ptr<const double>(in_cont);
            double* out_ptr = get_data_ptr<double>(output);

            auto sum_buf = sycl::malloc_shared<double>(1, queue);
            sum_buf[0] = 0.0;

            queue.parallel_for(sycl::range<1>(total_size), sycl::reduction(sum_buf, sycl::plus<double>()),
                              [=](sycl::id<1> idx, auto& sum) {
                sum += in_ptr[idx];
            }).wait();

            out_ptr[0] = sum_buf[0];
            queue.wait();
            sycl::free(sum_buf, queue);
        }
        else {
            throw std::runtime_error("Unsupported dtype for sum reduction");
        }
    } else {
        // Partial reduction along a specific dimension
        const int64_t outer_size = std::accumulate(shape.begin(), shape.begin() + dim, 1LL, std::multiplies<>());
        const int64_t dim_size = shape[dim];
        const int64_t inner_size = std::accumulate(shape.begin() + dim + 1, shape.end(), 1LL, std::multiplies<>());

        if (in_cont.dtype() == DType::Float32) {
            const float* in_ptr = get_data_ptr<const float>(in_cont);
            float* out_ptr = get_data_ptr<float>(output);

            queue.parallel_for<SumKernelFloat32>(sycl::range<2>(outer_size, inner_size), [=](sycl::id<2> idx) {
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
        else if (in_cont.dtype() == DType::Float64) {
            const double* in_ptr = get_data_ptr<const double>(in_cont);
            double* out_ptr = get_data_ptr<double>(output);

            queue.parallel_for<SumKernelFloat64>(sycl::range<2>(outer_size, inner_size), [=](sycl::id<2> idx) {
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
    }

    return output;
}

// Mean reduction kernel
// IMPORTANT: Must ensure contiguous input for direct memory access
auto mean_kernel(const Tensor& input, int64_t dim, bool keepdim, sycl::queue& queue) -> Tensor {
    // Ensure input is contiguous for correct memory access
    Tensor in_cont = input.is_contiguous() ? input : contiguous_kernel(input, queue);

    auto shape = in_cont.shape();
    std::vector<int64_t> shape_vec(shape.begin(), shape.end());

    // Check if this is a full reduction (dim < 0 means reduce all dimensions)
    bool is_full_reduction = (dim < 0);

    if (!is_full_reduction) {
        // Normalize dimension for partial reduction
        if (dim < 0) {
            dim += shape.size();
        }
        if (dim < 0 || dim >= static_cast<int64_t>(shape.size())) {
            throw std::invalid_argument("Invalid dimension for mean reduction");
        }
    }

    // Calculate output shape
    auto out_shape = compute_reduction_shape(shape_vec, is_full_reduction ? -1 : dim, keepdim);
    Tensor output(out_shape, in_cont.dtype(), in_cont.device());

    if (is_full_reduction) {
        // Full reduction: mean of all elements
        const int64_t total_size = in_cont.numel();

        if (in_cont.dtype() == DType::Float32) {
            const float* in_ptr = get_data_ptr<const float>(in_cont);
            float* out_ptr = get_data_ptr<float>(output);
            const float scale = 1.0f / static_cast<float>(total_size);

            auto sum_buf = sycl::malloc_shared<float>(1, queue);
            sum_buf[0] = 0.0f;

            queue.parallel_for(sycl::range<1>(total_size), sycl::reduction(sum_buf, sycl::plus<float>()),
                              [=](sycl::id<1> idx, auto& sum) {
                sum += in_ptr[idx];
            }).wait();

            out_ptr[0] = sum_buf[0] * scale;
            queue.wait();
            sycl::free(sum_buf, queue);
        }
        else if (in_cont.dtype() == DType::Float64) {
            const double* in_ptr = get_data_ptr<const double>(in_cont);
            double* out_ptr = get_data_ptr<double>(output);
            const double scale = 1.0 / static_cast<double>(total_size);

            auto sum_buf = sycl::malloc_shared<double>(1, queue);
            sum_buf[0] = 0.0;

            queue.parallel_for(sycl::range<1>(total_size), sycl::reduction(sum_buf, sycl::plus<double>()),
                              [=](sycl::id<1> idx, auto& sum) {
                sum += in_ptr[idx];
            }).wait();

            out_ptr[0] = sum_buf[0] * scale;
            queue.wait();
            sycl::free(sum_buf, queue);
        }
        else {
            throw std::runtime_error("Unsupported dtype for mean reduction");
        }
    } else {
        // Partial reduction along a specific dimension
        const int64_t outer_size = std::accumulate(shape.begin(), shape.begin() + dim, 1LL, std::multiplies<>());
        const int64_t dim_size = shape[dim];
        const int64_t inner_size = std::accumulate(shape.begin() + dim + 1, shape.end(), 1LL, std::multiplies<>());

        if (in_cont.dtype() == DType::Float32) {
            const float* in_ptr = get_data_ptr<const float>(in_cont);
            float* out_ptr = get_data_ptr<float>(output);
            const float scale = 1.0f / static_cast<float>(dim_size);

            queue.parallel_for<MeanKernelFloat32>(sycl::range<2>(outer_size, inner_size), [=](sycl::id<2> idx) {
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
        else if (in_cont.dtype() == DType::Float64) {
            const double* in_ptr = get_data_ptr<const double>(in_cont);
            double* out_ptr = get_data_ptr<double>(output);
            const double scale = 1.0 / static_cast<double>(dim_size);

            queue.parallel_for<MeanKernelFloat64>(sycl::range<2>(outer_size, inner_size), [=](sycl::id<2> idx) {
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
    }

    return output;
}

// Max reduction kernel
auto max_kernel(const Tensor& input, int64_t dim, bool keepdim, sycl::queue& queue) -> Tensor {
    auto shape = input.shape();
    std::vector<int64_t> shape_vec(shape.begin(), shape.end());

    // Check if this is a full reduction (dim < 0 means reduce all dimensions)
    bool is_full_reduction = (dim < 0);

    if (!is_full_reduction) {
        // Normalize dimension for partial reduction
        if (dim < 0) {
            dim += shape.size();
        }
        if (dim < 0 || dim >= static_cast<int64_t>(shape.size())) {
            throw std::invalid_argument("Invalid dimension for max reduction");
        }
    }

    // Calculate output shape
    auto out_shape = compute_reduction_shape(shape_vec, is_full_reduction ? -1 : dim, keepdim);
    Tensor output(out_shape, input.dtype(), input.device());

    if (is_full_reduction) {
        // Full reduction: max of all elements
        const int64_t total_size = input.numel();

        if (input.dtype() == DType::Float32) {
            const float* in_ptr = get_data_ptr<const float>(input);
            float* out_ptr = get_data_ptr<float>(output);

            queue.single_task([=]() {
                float max_val = -std::numeric_limits<float>::infinity();
                for (int64_t i = 0; i < total_size; ++i) {
                    max_val = sycl::fmax(max_val, in_ptr[i]);
                }
                out_ptr[0] = max_val;
            }).wait();
        }
        else if (input.dtype() == DType::Float64) {
            const double* in_ptr = get_data_ptr<const double>(input);
            double* out_ptr = get_data_ptr<double>(output);

            queue.single_task([=]() {
                double max_val = -std::numeric_limits<double>::infinity();
                for (int64_t i = 0; i < total_size; ++i) {
                    max_val = sycl::fmax(max_val, in_ptr[i]);
                }
                out_ptr[0] = max_val;
            }).wait();
        }
        else {
            throw std::runtime_error("Unsupported dtype for max reduction");
        }
    } else {
        // Partial reduction along a specific dimension
        const int64_t outer_size = std::accumulate(shape.begin(), shape.begin() + dim, 1LL, std::multiplies<>());
        const int64_t dim_size = shape[dim];
        const int64_t inner_size = std::accumulate(shape.begin() + dim + 1, shape.end(), 1LL, std::multiplies<>());

        if (input.dtype() == DType::Float32) {
            const float* in_ptr = get_data_ptr<const float>(input);
            float* out_ptr = get_data_ptr<float>(output);

            queue.parallel_for<MaxKernelFloat32>(sycl::range<2>(outer_size, inner_size), [=](sycl::id<2> idx) {
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

            queue.parallel_for<MaxKernelFloat64>(sycl::range<2>(outer_size, inner_size), [=](sycl::id<2> idx) {
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
    }

    return output;
}

// Min reduction kernel
auto min_kernel(const Tensor& input, int64_t dim, bool keepdim, sycl::queue& queue) -> Tensor {
    auto shape = input.shape();
    std::vector<int64_t> shape_vec(shape.begin(), shape.end());

    // Check if this is a full reduction (dim < 0 means reduce all dimensions)
    bool is_full_reduction = (dim < 0);

    if (!is_full_reduction) {
        // Normalize dimension for partial reduction
        if (dim < 0) {
            dim += shape.size();
        }
        if (dim < 0 || dim >= static_cast<int64_t>(shape.size())) {
            throw std::invalid_argument("Invalid dimension for min reduction");
        }
    }

    // Calculate output shape
    auto out_shape = compute_reduction_shape(shape_vec, is_full_reduction ? -1 : dim, keepdim);
    Tensor output(out_shape, input.dtype(), input.device());

    if (is_full_reduction) {
        // Full reduction: min of all elements
        const int64_t total_size = input.numel();

        if (input.dtype() == DType::Float32) {
            const float* in_ptr = get_data_ptr<const float>(input);
            float* out_ptr = get_data_ptr<float>(output);

            queue.single_task([=]() {
                float min_val = std::numeric_limits<float>::infinity();
                for (int64_t i = 0; i < total_size; ++i) {
                    min_val = sycl::fmin(min_val, in_ptr[i]);
                }
                out_ptr[0] = min_val;
            }).wait();
        }
        else if (input.dtype() == DType::Float64) {
            const double* in_ptr = get_data_ptr<const double>(input);
            double* out_ptr = get_data_ptr<double>(output);

            queue.single_task([=]() {
                double min_val = std::numeric_limits<double>::infinity();
                for (int64_t i = 0; i < total_size; ++i) {
                    min_val = sycl::fmin(min_val, in_ptr[i]);
                }
                out_ptr[0] = min_val;
            }).wait();
        }
        else {
            throw std::runtime_error("Unsupported dtype for min reduction");
        }
    } else {
        // Partial reduction along a specific dimension
        const int64_t outer_size = std::accumulate(shape.begin(), shape.begin() + dim, 1LL, std::multiplies<>());
        const int64_t dim_size = shape[dim];
        const int64_t inner_size = std::accumulate(shape.begin() + dim + 1, shape.end(), 1LL, std::multiplies<>());

        if (input.dtype() == DType::Float32) {
            const float* in_ptr = get_data_ptr<const float>(input);
            float* out_ptr = get_data_ptr<float>(output);

            queue.parallel_for<MinKernelFloat32>(sycl::range<2>(outer_size, inner_size), [=](sycl::id<2> idx) {
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

            queue.parallel_for<MinKernelFloat64>(sycl::range<2>(outer_size, inner_size), [=](sycl::id<2> idx) {
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
    }

    return output;
}

/**
 * @brief Argmax operation - returns indices of maximum values.
 *
 * Finds the indices of maximum values along a dimension.
 * Currently supports reduction over all dimensions (dim=-1) or last dimension.
 *
 * @param input Input tensor
 * @param dim Dimension to reduce along (-1 for all dimensions)
 * @param keepdim Whether to keep the reduced dimension
 * @param queue SYCL queue for execution
 * @return Tensor containing indices (Int64 dtype)
 */
auto argmax_kernel(const Tensor& input, int64_t dim, bool keepdim, sycl::queue& queue) -> Tensor {
    const auto& shape = input.shape();

    // For full reduction (dim=-1)
    if (dim == -1) {
        int64_t total_size = 1;
        for (auto s : shape) {
            total_size *= s;
        }

        std::vector<int64_t> out_shape = keepdim ? std::vector<int64_t>(shape.size(), 1) : std::vector<int64_t>{1};
        Tensor output(out_shape, DType::Int64, input.device());
        int64_t* out_ptr = get_data_ptr<int64_t>(output);

        if (input.dtype() == DType::Float32) {
            const float* in_ptr = get_data_ptr<const float>(input);

            // Copy data to host and find argmax
            std::vector<float> host_data(total_size);
            queue.memcpy(host_data.data(), in_ptr, total_size * sizeof(float)).wait();

            int64_t max_idx = 0;
            float max_val = host_data[0];
            for (int64_t i = 1; i < total_size; ++i) {
                if (host_data[i] > max_val) {
                    max_val = host_data[i];
                    max_idx = i;
                }
            }

            queue.memcpy(out_ptr, &max_idx, sizeof(int64_t)).wait();
        }
        else if (input.dtype() == DType::Float64) {
            const double* in_ptr = get_data_ptr<const double>(input);

            // Copy data to host and find argmax
            std::vector<double> host_data(total_size);
            queue.memcpy(host_data.data(), in_ptr, total_size * sizeof(double)).wait();

            int64_t max_idx = 0;
            double max_val = host_data[0];
            for (int64_t i = 1; i < total_size; ++i) {
                if (host_data[i] > max_val) {
                    max_val = host_data[i];
                    max_idx = i;
                }
            }

            queue.memcpy(out_ptr, &max_idx, sizeof(int64_t)).wait();
        }
        else {
            throw std::runtime_error("Unsupported dtype for argmax");
        }

        return output;
    }

    // For reduction along a specific dimension
    if (dim < 0 || dim >= static_cast<int64_t>(shape.size())) {
        throw std::invalid_argument("Invalid dimension for argmax");
    }

    std::vector<int64_t> shape_vec(shape.begin(), shape.end());
    auto output_shape = compute_reduction_shape(shape_vec, dim, keepdim);
    Tensor output(output_shape, DType::Int64, input.device());

    const int64_t dim_size = shape[dim];
    int64_t outer_size = 1;
    for (int64_t i = 0; i < dim; ++i) {
        outer_size *= shape[i];
    }
    int64_t inner_size = 1;
    for (int64_t i = dim + 1; i < static_cast<int64_t>(shape.size()); ++i) {
        inner_size *= shape[i];
    }

    int64_t* out_ptr = get_data_ptr<int64_t>(output);

    if (input.dtype() == DType::Float32) {
        const float* in_ptr = get_data_ptr<const float>(input);

        queue.parallel_for<ArgmaxKernelFloat32>(sycl::range<2>(outer_size, inner_size), [=](sycl::id<2> idx) {
            const int64_t outer_idx = idx[0];
            const int64_t inner_idx = idx[1];
            const int64_t base_offset = outer_idx * dim_size * inner_size + inner_idx;

            float max_val = -std::numeric_limits<float>::infinity();
            int64_t max_idx = 0;
            for (int64_t d = 0; d < dim_size; ++d) {
                float val = in_ptr[base_offset + d * inner_size];
                if (val > max_val) {
                    max_val = val;
                    max_idx = d;
                }
            }

            out_ptr[outer_idx * inner_size + inner_idx] = max_idx;
        }).wait();
    }
    else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);

        queue.parallel_for<ArgmaxKernelFloat64>(sycl::range<2>(outer_size, inner_size), [=](sycl::id<2> idx) {
            const int64_t outer_idx = idx[0];
            const int64_t inner_idx = idx[1];
            const int64_t base_offset = outer_idx * dim_size * inner_size + inner_idx;

            double max_val = -std::numeric_limits<double>::infinity();
            int64_t max_idx = 0;
            for (int64_t d = 0; d < dim_size; ++d) {
                double val = in_ptr[base_offset + d * inner_size];
                if (val > max_val) {
                    max_val = val;
                    max_idx = d;
                }
            }

            out_ptr[outer_idx * inner_size + inner_idx] = max_idx;
        }).wait();
    }
    else {
        throw std::runtime_error("Unsupported dtype for argmax");
    }

    return output;
}

/**
 * @brief Argmin operation - returns indices of minimum values.
 *
 * Finds the indices of minimum values along a dimension.
 * Currently supports reduction over all dimensions (dim=-1) or last dimension.
 *
 * @param input Input tensor
 * @param dim Dimension to reduce along (-1 for all dimensions)
 * @param keepdim Whether to keep the reduced dimension
 * @param queue SYCL queue for execution
 * @return Tensor containing indices (Int64 dtype)
 */
auto argmin_kernel(const Tensor& input, int64_t dim, bool keepdim, sycl::queue& queue) -> Tensor {
    const auto& shape = input.shape();

    // For full reduction (dim=-1)
    if (dim == -1) {
        int64_t total_size = 1;
        for (auto s : shape) {
            total_size *= s;
        }

        std::vector<int64_t> out_shape = keepdim ? std::vector<int64_t>(shape.size(), 1) : std::vector<int64_t>{1};
        Tensor output(out_shape, DType::Int64, input.device());
        int64_t* out_ptr = get_data_ptr<int64_t>(output);

        if (input.dtype() == DType::Float32) {
            const float* in_ptr = get_data_ptr<const float>(input);

            // Copy data to host and find argmin
            std::vector<float> host_data(total_size);
            queue.memcpy(host_data.data(), in_ptr, total_size * sizeof(float)).wait();

            int64_t min_idx = 0;
            float min_val = host_data[0];
            for (int64_t i = 1; i < total_size; ++i) {
                if (host_data[i] < min_val) {
                    min_val = host_data[i];
                    min_idx = i;
                }
            }

            queue.memcpy(out_ptr, &min_idx, sizeof(int64_t)).wait();
        }
        else if (input.dtype() == DType::Float64) {
            const double* in_ptr = get_data_ptr<const double>(input);

            // Copy data to host and find argmin
            std::vector<double> host_data(total_size);
            queue.memcpy(host_data.data(), in_ptr, total_size * sizeof(double)).wait();

            int64_t min_idx = 0;
            double min_val = host_data[0];
            for (int64_t i = 1; i < total_size; ++i) {
                if (host_data[i] < min_val) {
                    min_val = host_data[i];
                    min_idx = i;
                }
            }

            queue.memcpy(out_ptr, &min_idx, sizeof(int64_t)).wait();
        }
        else {
            throw std::runtime_error("Unsupported dtype for argmin");
        }

        return output;
    }

    // For reduction along a specific dimension
    if (dim < 0 || dim >= static_cast<int64_t>(shape.size())) {
        throw std::invalid_argument("Invalid dimension for argmin");
    }

    std::vector<int64_t> shape_vec(shape.begin(), shape.end());
    auto output_shape = compute_reduction_shape(shape_vec, dim, keepdim);
    Tensor output(output_shape, DType::Int64, input.device());

    const int64_t dim_size = shape[dim];
    int64_t outer_size = 1;
    for (int64_t i = 0; i < dim; ++i) {
        outer_size *= shape[i];
    }
    int64_t inner_size = 1;
    for (int64_t i = dim + 1; i < static_cast<int64_t>(shape.size()); ++i) {
        inner_size *= shape[i];
    }

    int64_t* out_ptr = get_data_ptr<int64_t>(output);

    if (input.dtype() == DType::Float32) {
        const float* in_ptr = get_data_ptr<const float>(input);

        queue.parallel_for<ArgminKernelFloat32>(sycl::range<2>(outer_size, inner_size), [=](sycl::id<2> idx) {
            const int64_t outer_idx = idx[0];
            const int64_t inner_idx = idx[1];
            const int64_t base_offset = outer_idx * dim_size * inner_size + inner_idx;

            float min_val = std::numeric_limits<float>::infinity();
            int64_t min_idx = 0;
            for (int64_t d = 0; d < dim_size; ++d) {
                float val = in_ptr[base_offset + d * inner_size];
                if (val < min_val) {
                    min_val = val;
                    min_idx = d;
                }
            }

            out_ptr[outer_idx * inner_size + inner_idx] = min_idx;
        }).wait();
    }
    else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);

        queue.parallel_for<ArgminKernelFloat64>(sycl::range<2>(outer_size, inner_size), [=](sycl::id<2> idx) {
            const int64_t outer_idx = idx[0];
            const int64_t inner_idx = idx[1];
            const int64_t base_offset = outer_idx * dim_size * inner_size + inner_idx;

            double min_val = std::numeric_limits<double>::infinity();
            int64_t min_idx = 0;
            for (int64_t d = 0; d < dim_size; ++d) {
                double val = in_ptr[base_offset + d * inner_size];
                if (val < min_val) {
                    min_val = val;
                    min_idx = d;
                }
            }

            out_ptr[outer_idx * inner_size + inner_idx] = min_idx;
        }).wait();
    }
    else {
        throw std::runtime_error("Unsupported dtype for argmin");
    }

    return output;
}

} // namespace oneapi
} // namespace tenzor
