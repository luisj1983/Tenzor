#include "tenzor/core/tensor.hpp"
#include "tenzor/backend/backend.hpp"
#include "tenzor/backend/op_attributes.hpp"
#include <sycl/sycl.hpp>
#include <limits>
#include <stdexcept>
#include <cmath>

namespace tenzor {
namespace oneapi {

// Kernel class declarations for SYCL
class StdAllKernelFloat32;
class StdAllKernelFloat64;
class VarKernelFloat32;
class VarKernelFloat64;
class ProdKernelFloat32;
class ProdKernelFloat64;

// Helper function to get typed pointer from tensor
template<typename T>
inline auto get_data_ptr(const Tensor& t) -> T* {
    return static_cast<T*>(const_cast<void*>(t.data_ptr()));
}

/**
 * @brief Standard deviation operation wrapper.
 *
 * Computes the standard deviation of tensor elements.
 * Currently supports reduction over all dimensions (dim=-1).
 * OpAttributes wrapper for std operation.
 *
 * Formula: std = sqrt(sum((x - mean)^2) / (n - correction))
 *
 * @param input Input tensor
 * @param attrs Operation attributes containing dim, keepdim, correction
 * @param queue SYCL queue for execution
 * @return Tensor Standard deviation tensor
 */
auto std_kernel(const Tensor& input, const OpAttributes& attrs, sycl::queue& queue) -> Tensor {
    int64_t dim = attrs.get_int(AttrKey::Dim, -1);
    bool keepdim = attrs.get_bool(AttrKey::Keepdim, false);
    int64_t correction = attrs.get_int(AttrKey::Correction, 1);

    auto shape = input.shape();

    // For now, only support reduction over all dimensions
    if (dim != -1 && dim != static_cast<int64_t>(shape.size()) - 1) {
        throw std::invalid_argument("std: only dim=-1 (all dimensions) is currently supported for OneAPI");
    }

    // Compute total number of elements
    int64_t total_size = 1;
    for (auto s : shape) {
        total_size *= s;
    }

    if (input.dtype() == DType::Float32) {
        const float* in_ptr = get_data_ptr<const float>(input);
        float eps = 1e-8f;

        // Step 1: Compute mean using atomic operations
        float* mean_device = sycl::malloc_device<float>(1, queue);
        queue.fill(mean_device, 0.0f, 1);

        queue.parallel_for<class MeanAllKernelFloat32>(sycl::range<1>(total_size),
            [=](sycl::id<1> idx) {
                sycl::atomic_ref<float, sycl::memory_order::relaxed,
                               sycl::memory_scope::device> atomic_mean(mean_device[0]);
                atomic_mean += in_ptr[idx];
        });

        float mean_host;
        queue.memcpy(&mean_host, mean_device, sizeof(float)).wait();
        mean_host /= static_cast<float>(total_size);

        // Step 2: Compute variance
        float* var_device = sycl::malloc_device<float>(1, queue);
        queue.fill(var_device, 0.0f, 1);

        queue.parallel_for<class VarAllKernelFloat32>(sycl::range<1>(total_size),
            [=](sycl::id<1> idx) {
                float diff = in_ptr[idx] - mean_host;
                sycl::atomic_ref<float, sycl::memory_order::relaxed,
                               sycl::memory_scope::device> atomic_var(var_device[0]);
                atomic_var += diff * diff;
        });

        float var_host;
        queue.memcpy(&var_host, var_device, sizeof(float)).wait();
        int64_t divisor = total_size - correction;
        if (divisor <= 0) divisor = 1;
        var_host /= static_cast<float>(divisor);

        // Step 3: Compute std
        float std_val = sycl::sqrt(var_host + eps);

        sycl::free(mean_device, queue);
        sycl::free(var_device, queue);

        // Create output tensor
        std::vector<int64_t> out_shape;
        if (keepdim) {
            out_shape.resize(shape.size(), 1);
        } else {
            out_shape = {1};
        }

        Tensor output(out_shape, input.dtype(), input.device());
        float* out_ptr = get_data_ptr<float>(output);
        queue.fill(out_ptr, std_val, 1).wait();

        return output;
    }
    else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        double eps = 1e-8;

        // Step 1: Compute mean
        double* mean_device = sycl::malloc_device<double>(1, queue);
        queue.fill(mean_device, 0.0, 1);

        queue.parallel_for<class MeanAllKernelFloat64>(sycl::range<1>(total_size),
            [=](sycl::id<1> idx) {
                sycl::atomic_ref<double, sycl::memory_order::relaxed,
                               sycl::memory_scope::device> atomic_mean(mean_device[0]);
                atomic_mean += in_ptr[idx];
        });

        double mean_host;
        queue.memcpy(&mean_host, mean_device, sizeof(double)).wait();
        mean_host /= static_cast<double>(total_size);

        // Step 2: Compute variance
        double* var_device = sycl::malloc_device<double>(1, queue);
        queue.fill(var_device, 0.0, 1);

        queue.parallel_for<class VarAllKernelFloat64>(sycl::range<1>(total_size),
            [=](sycl::id<1> idx) {
                double diff = in_ptr[idx] - mean_host;
                sycl::atomic_ref<double, sycl::memory_order::relaxed,
                               sycl::memory_scope::device> atomic_var(var_device[0]);
                atomic_var += diff * diff;
        });

        double var_host;
        queue.memcpy(&var_host, var_device, sizeof(double)).wait();
        int64_t divisor = total_size - correction;
        if (divisor <= 0) divisor = 1;
        var_host /= static_cast<double>(divisor);

        // Step 3: Compute std
        double std_val = sycl::sqrt(var_host + eps);

        sycl::free(mean_device, queue);
        sycl::free(var_device, queue);

        // Create output tensor
        std::vector<int64_t> out_shape;
        if (keepdim) {
            out_shape.resize(shape.size(), 1);
        } else {
            out_shape = {1};
        }

        Tensor output(out_shape, input.dtype(), input.device());
        double* out_ptr = get_data_ptr<double>(output);
        queue.fill(out_ptr, std_val, 1).wait();

        return output;
    }
    else {
        throw std::runtime_error("Unsupported dtype for std_kernel");
    }
}

/**
 * @brief Variance operation wrapper.
 *
 * Computes the variance of tensor elements.
 * Currently supports reduction over all dimensions (dim=-1).
 * OpAttributes wrapper for var operation.
 *
 * Formula: var = sum((x - mean)^2) / (n - correction)
 *
 * @param input Input tensor
 * @param attrs Operation attributes containing dim, keepdim, correction
 * @param queue SYCL queue for execution
 * @return Tensor Variance tensor
 */
auto var_kernel(const Tensor& input, const OpAttributes& attrs, sycl::queue& queue) -> Tensor {
    int64_t dim = attrs.get_int(AttrKey::Dim, -1);
    bool keepdim = attrs.get_bool(AttrKey::Keepdim, false);
    int64_t correction = attrs.get_int(AttrKey::Correction, 1);

    auto shape = input.shape();

    // For now, only support reduction over all dimensions
    if (dim != -1 && dim != static_cast<int64_t>(shape.size()) - 1) {
        throw std::invalid_argument("var: only dim=-1 (all dimensions) is currently supported for OneAPI");
    }

    // Compute total number of elements
    int64_t total_size = 1;
    for (auto s : shape) {
        total_size *= s;
    }

    if (input.dtype() == DType::Float32) {
        const float* in_ptr = get_data_ptr<const float>(input);

        // Step 1: Compute mean using atomic operations
        float* mean_device = sycl::malloc_device<float>(1, queue);
        queue.fill(mean_device, 0.0f, 1);

        queue.parallel_for<class MeanVarKernelFloat32>(sycl::range<1>(total_size),
            [=](sycl::id<1> idx) {
                sycl::atomic_ref<float, sycl::memory_order::relaxed,
                               sycl::memory_scope::device> atomic_mean(mean_device[0]);
                atomic_mean += in_ptr[idx];
        });

        float mean_host;
        queue.memcpy(&mean_host, mean_device, sizeof(float)).wait();
        mean_host /= static_cast<float>(total_size);

        // Step 2: Compute variance
        float* var_device = sycl::malloc_device<float>(1, queue);
        queue.fill(var_device, 0.0f, 1);

        queue.parallel_for<class VarKernelFloat32>(sycl::range<1>(total_size),
            [=](sycl::id<1> idx) {
                float diff = in_ptr[idx] - mean_host;
                sycl::atomic_ref<float, sycl::memory_order::relaxed,
                               sycl::memory_scope::device> atomic_var(var_device[0]);
                atomic_var += diff * diff;
        });

        float var_host;
        queue.memcpy(&var_host, var_device, sizeof(float)).wait();
        int64_t divisor = total_size - correction;
        if (divisor <= 0) divisor = 1;
        var_host /= static_cast<float>(divisor);

        sycl::free(mean_device, queue);
        sycl::free(var_device, queue);

        // Create output tensor
        std::vector<int64_t> out_shape;
        if (keepdim) {
            out_shape.resize(shape.size(), 1);
        } else {
            out_shape = {1};
        }

        Tensor output(out_shape, input.dtype(), input.device());
        float* out_ptr = get_data_ptr<float>(output);
        queue.fill(out_ptr, var_host, 1).wait();

        return output;
    }
    else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);

        // Step 1: Compute mean
        double* mean_device = sycl::malloc_device<double>(1, queue);
        queue.fill(mean_device, 0.0, 1);

        queue.parallel_for<class MeanVarKernelFloat64>(sycl::range<1>(total_size),
            [=](sycl::id<1> idx) {
                sycl::atomic_ref<double, sycl::memory_order::relaxed,
                               sycl::memory_scope::device> atomic_mean(mean_device[0]);
                atomic_mean += in_ptr[idx];
        });

        double mean_host;
        queue.memcpy(&mean_host, mean_device, sizeof(double)).wait();
        mean_host /= static_cast<double>(total_size);

        // Step 2: Compute variance
        double* var_device = sycl::malloc_device<double>(1, queue);
        queue.fill(var_device, 0.0, 1);

        queue.parallel_for<class VarKernelFloat64>(sycl::range<1>(total_size),
            [=](sycl::id<1> idx) {
                double diff = in_ptr[idx] - mean_host;
                sycl::atomic_ref<double, sycl::memory_order::relaxed,
                               sycl::memory_scope::device> atomic_var(var_device[0]);
                atomic_var += diff * diff;
        });

        double var_host;
        queue.memcpy(&var_host, var_device, sizeof(double)).wait();
        int64_t divisor = total_size - correction;
        if (divisor <= 0) divisor = 1;
        var_host /= static_cast<double>(divisor);

        sycl::free(mean_device, queue);
        sycl::free(var_device, queue);

        // Create output tensor
        std::vector<int64_t> out_shape;
        if (keepdim) {
            out_shape.resize(shape.size(), 1);
        } else {
            out_shape = {1};
        }

        Tensor output(out_shape, input.dtype(), input.device());
        double* out_ptr = get_data_ptr<double>(output);
        queue.fill(out_ptr, var_host, 1).wait();

        return output;
    }
    else {
        throw std::runtime_error("Unsupported dtype for var_kernel");
    }
}

/**
 * @brief Product operation wrapper.
 *
 * Computes the product of all tensor elements.
 * Currently supports reduction over all dimensions (dim=-1).
 * OpAttributes wrapper for prod operation.
 *
 * Formula: prod = x[0] * x[1] * x[2] * ... * x[n-1]
 *
 * @param input Input tensor
 * @param attrs Operation attributes containing dim, keepdim
 * @param queue SYCL queue for execution
 * @return Tensor Product tensor
 */
auto prod_kernel(const Tensor& input, const OpAttributes& attrs, sycl::queue& queue) -> Tensor {
    int64_t dim = attrs.get_int(AttrKey::Dim, -1);
    bool keepdim = attrs.get_bool(AttrKey::Keepdim, false);

    auto shape_span = input.shape();
    std::vector<int64_t> shape(shape_span.begin(), shape_span.end());
    auto strides_span = input.strides();
    std::vector<int64_t> strides(strides_span.begin(), strides_span.end());

    // Compute output shape
    std::vector<int64_t> out_shape;
    if (dim == -1) {
        // Full reduction
        out_shape = keepdim ? std::vector<int64_t>(shape.size(), 1) : std::vector<int64_t>{1};
    } else {
        // Dimensional reduction
        if (dim < 0 || dim >= static_cast<int64_t>(shape.size())) {
            throw std::invalid_argument("prod: dim out of range");
        }
        out_shape.reserve(keepdim ? shape.size() : shape.size() - 1);
        for (size_t i = 0; i < shape.size(); ++i) {
            if (static_cast<int64_t>(i) == dim) {
                if (keepdim) out_shape.push_back(1);
            } else {
                out_shape.push_back(shape[i]);
            }
        }
    }

    // Compute total number of elements
    int64_t total_size = 1;
    for (auto s : shape) {
        total_size *= s;
    }

    // Compute output size
    int64_t output_size = 1;
    for (auto s : out_shape) {
        output_size *= s;
    }

    if (input.dtype() == DType::Float32) {
        const float* in_ptr = get_data_ptr<const float>(input);

        // Copy to host and compute product
        std::vector<float> host_data(total_size);
        queue.memcpy(host_data.data(), in_ptr, total_size * sizeof(float)).wait();

        std::vector<float> result(output_size);

        if (dim == -1) {
            // Full reduction
            float prod_value = 1.0f;
            for (int64_t i = 0; i < total_size; ++i) {
                prod_value *= host_data[i];
            }
            result[0] = prod_value;
        } else {
            // Dimensional reduction
            const int64_t ndim = shape.size();
            const int64_t dim_size = shape[dim];

            for (int64_t out_idx = 0; out_idx < output_size; ++out_idx) {
                std::vector<int64_t> indices(ndim, 0);
                int64_t tmp = out_idx;

                // Convert flat output index to multi-dimensional indices
                for (int64_t d = ndim - 1; d >= 0; --d) {
                    if (d == dim) continue;
                    int64_t size = shape[d];
                    indices[d] = tmp % size;
                    tmp /= size;
                }

                // Compute product along the reduction dimension
                float prod_value = 1.0f;
                for (int64_t i = 0; i < dim_size; ++i) {
                    indices[dim] = i;
                    int64_t in_idx = 0;
                    for (int64_t d = 0; d < ndim; ++d) {
                        in_idx += indices[d] * strides[d];
                    }
                    prod_value *= host_data[in_idx];
                }
                result[out_idx] = prod_value;
            }
        }

        // Create output tensor and copy result back
        Tensor output(out_shape, input.dtype(), input.device());
        float* out_ptr = get_data_ptr<float>(output);
        queue.memcpy(out_ptr, result.data(), output_size * sizeof(float)).wait();

        return output;
    }
    else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);

        // Copy to host and compute product
        std::vector<double> host_data(total_size);
        queue.memcpy(host_data.data(), in_ptr, total_size * sizeof(double)).wait();

        std::vector<double> result(output_size);

        if (dim == -1) {
            // Full reduction
            double prod_value = 1.0;
            for (int64_t i = 0; i < total_size; ++i) {
                prod_value *= host_data[i];
            }
            result[0] = prod_value;
        } else {
            // Dimensional reduction
            const int64_t ndim = shape.size();
            const int64_t dim_size = shape[dim];

            for (int64_t out_idx = 0; out_idx < output_size; ++out_idx) {
                std::vector<int64_t> indices(ndim, 0);
                int64_t tmp = out_idx;

                // Convert flat output index to multi-dimensional indices
                for (int64_t d = ndim - 1; d >= 0; --d) {
                    if (d == dim) continue;
                    int64_t size = shape[d];
                    indices[d] = tmp % size;
                    tmp /= size;
                }

                // Compute product along the reduction dimension
                double prod_value = 1.0;
                for (int64_t i = 0; i < dim_size; ++i) {
                    indices[dim] = i;
                    int64_t in_idx = 0;
                    for (int64_t d = 0; d < ndim; ++d) {
                        in_idx += indices[d] * strides[d];
                    }
                    prod_value *= host_data[in_idx];
                }
                result[out_idx] = prod_value;
            }
        }

        // Create output tensor and copy result back
        Tensor output(out_shape, input.dtype(), input.device());
        double* out_ptr = get_data_ptr<double>(output);
        queue.memcpy(out_ptr, result.data(), output_size * sizeof(double)).wait();

        return output;
    }
    else if (input.dtype() == DType::Int32) {
        const int32_t* in_ptr = get_data_ptr<const int32_t>(input);

        // Copy to host and compute product
        std::vector<int32_t> host_data(total_size);
        queue.memcpy(host_data.data(), in_ptr, total_size * sizeof(int32_t)).wait();

        std::vector<int32_t> result(output_size);

        if (dim == -1) {
            // Full reduction - use int64_t accumulator to avoid overflow
            int64_t prod_value = 1;
            for (int64_t i = 0; i < total_size; ++i) {
                prod_value *= static_cast<int64_t>(host_data[i]);
            }
            result[0] = static_cast<int32_t>(prod_value);
        } else {
            // Dimensional reduction
            const int64_t ndim = shape.size();
            const int64_t dim_size = shape[dim];

            for (int64_t out_idx = 0; out_idx < output_size; ++out_idx) {
                std::vector<int64_t> indices(ndim, 0);
                int64_t tmp = out_idx;

                // Convert flat output index to multi-dimensional indices
                for (int64_t d = ndim - 1; d >= 0; --d) {
                    if (d == dim) continue;
                    int64_t size = shape[d];
                    indices[d] = tmp % size;
                    tmp /= size;
                }

                // Compute product along the reduction dimension
                int64_t prod_value = 1;
                for (int64_t i = 0; i < dim_size; ++i) {
                    indices[dim] = i;
                    int64_t in_idx = 0;
                    for (int64_t d = 0; d < ndim; ++d) {
                        in_idx += indices[d] * strides[d];
                    }
                    prod_value *= static_cast<int64_t>(host_data[in_idx]);
                }
                result[out_idx] = static_cast<int32_t>(prod_value);
            }
        }

        // Create output tensor and copy result back
        Tensor output(out_shape, input.dtype(), input.device());
        int32_t* out_ptr = get_data_ptr<int32_t>(output);
        queue.memcpy(out_ptr, result.data(), output_size * sizeof(int32_t)).wait();

        return output;
    }
    else {
        throw std::runtime_error("Unsupported dtype for prod_kernel");
    }
}

// Norm kernel - compute Lp norm
auto norm_kernel(const Tensor& input, const OpAttributes& attrs, sycl::queue& queue) -> Tensor {
    float p = static_cast<float>(attrs.get_float(AttrKey::P, 2.0));
    int64_t dim = attrs.get_int(AttrKey::Dim, -1);
    bool keepdim = attrs.get_bool(AttrKey::Keepdim, false);

    auto shape_span = input.shape();
    std::vector<int64_t> shape(shape_span.begin(), shape_span.end());

    if (dim != -1) {
        throw std::runtime_error("norm: only full reduction (dim=-1) is currently supported for OneAPI");
    }

    // Compute output shape
    std::vector<int64_t> out_shape;
    if (keepdim) {
        out_shape.resize(shape.size(), 1);
    } else {
        out_shape = {1};
    }

    int64_t n = input.numel();
    if (n == 0) {
        throw std::runtime_error("norm: input tensor is empty");
    }

    if (input.dtype() == DType::Float32) {
        const float* in_ptr = get_data_ptr<const float>(input);

        // Copy to host and compute norm
        std::vector<float> host_data(n);
        queue.memcpy(host_data.data(), in_ptr, n * sizeof(float)).wait();

        float norm_value = 0.0f;

        if (p == 1.0f) {
            // L1 norm
            for (int64_t i = 0; i < n; ++i) {
                norm_value += std::abs(host_data[i]);
            }
        } else if (p == 2.0f) {
            // L2 norm
            for (int64_t i = 0; i < n; ++i) {
                norm_value += host_data[i] * host_data[i];
            }
            norm_value = std::sqrt(norm_value);
        } else {
            // General Lp norm
            for (int64_t i = 0; i < n; ++i) {
                norm_value += std::pow(std::abs(host_data[i]), p);
            }
            norm_value = std::pow(norm_value, 1.0f / p);
        }

        // Create output and copy result
        Tensor output(out_shape, input.dtype(), input.device());
        float* out_ptr = get_data_ptr<float>(output);
        queue.fill(out_ptr, norm_value, 1).wait();

        return output;
    }
    else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);

        // Copy to host and compute norm
        std::vector<double> host_data(n);
        queue.memcpy(host_data.data(), in_ptr, n * sizeof(double)).wait();

        double norm_value = 0.0;

        if (p == 1.0) {
            // L1 norm
            for (int64_t i = 0; i < n; ++i) {
                norm_value += std::abs(host_data[i]);
            }
        } else if (p == 2.0) {
            // L2 norm
            for (int64_t i = 0; i < n; ++i) {
                norm_value += host_data[i] * host_data[i];
            }
            norm_value = std::sqrt(norm_value);
        } else {
            // General Lp norm
            for (int64_t i = 0; i < n; ++i) {
                norm_value += std::pow(std::abs(host_data[i]), p);
            }
            norm_value = std::pow(norm_value, 1.0 / p);
        }

        // Create output and copy result
        Tensor output(out_shape, input.dtype(), input.device());
        double* out_ptr = get_data_ptr<double>(output);
        queue.fill(out_ptr, norm_value, 1).wait();

        return output;
    }
    else {
        throw std::runtime_error("Unsupported dtype for norm_kernel");
    }
}

} // namespace oneapi
} // namespace tenzor
