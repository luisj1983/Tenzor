#include "tenzor/core/tensor.hpp"
#include "tenzor/backend/backend.hpp"
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
    int64_t dim = attrs.contains("dim") ? std::stoll(attrs.at("dim")) : -1;
    bool keepdim = attrs.contains("keepdim") && attrs.at("keepdim") == "1";
    int64_t correction = attrs.contains("correction") ? std::stoll(attrs.at("correction")) : 1;

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
        queue.fill(mean_device, 0.0f, 1).wait();

        queue.parallel_for<class MeanAllKernelFloat32>(sycl::range<1>(total_size),
            [=](sycl::id<1> idx) {
                sycl::atomic_ref<float, sycl::memory_order::relaxed,
                               sycl::memory_scope::device> atomic_mean(mean_device[0]);
                atomic_mean += in_ptr[idx];
        }).wait();

        float mean_host;
        queue.memcpy(&mean_host, mean_device, sizeof(float)).wait();
        mean_host /= static_cast<float>(total_size);

        // Step 2: Compute variance
        float* var_device = sycl::malloc_device<float>(1, queue);
        queue.fill(var_device, 0.0f, 1).wait();

        queue.parallel_for<class VarAllKernelFloat32>(sycl::range<1>(total_size),
            [=](sycl::id<1> idx) {
                float diff = in_ptr[idx] - mean_host;
                sycl::atomic_ref<float, sycl::memory_order::relaxed,
                               sycl::memory_scope::device> atomic_var(var_device[0]);
                atomic_var += diff * diff;
        }).wait();

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
        queue.fill(mean_device, 0.0, 1).wait();

        queue.parallel_for<class MeanAllKernelFloat64>(sycl::range<1>(total_size),
            [=](sycl::id<1> idx) {
                sycl::atomic_ref<double, sycl::memory_order::relaxed,
                               sycl::memory_scope::device> atomic_mean(mean_device[0]);
                atomic_mean += in_ptr[idx];
        }).wait();

        double mean_host;
        queue.memcpy(&mean_host, mean_device, sizeof(double)).wait();
        mean_host /= static_cast<double>(total_size);

        // Step 2: Compute variance
        double* var_device = sycl::malloc_device<double>(1, queue);
        queue.fill(var_device, 0.0, 1).wait();

        queue.parallel_for<class VarAllKernelFloat64>(sycl::range<1>(total_size),
            [=](sycl::id<1> idx) {
                double diff = in_ptr[idx] - mean_host;
                sycl::atomic_ref<double, sycl::memory_order::relaxed,
                               sycl::memory_scope::device> atomic_var(var_device[0]);
                atomic_var += diff * diff;
        }).wait();

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
    int64_t dim = attrs.contains("dim") ? std::stoll(attrs.at("dim")) : -1;
    bool keepdim = attrs.contains("keepdim") && attrs.at("keepdim") == "1";
    int64_t correction = attrs.contains("correction") ? std::stoll(attrs.at("correction")) : 1;

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
        queue.fill(mean_device, 0.0f, 1).wait();

        queue.parallel_for<class MeanVarKernelFloat32>(sycl::range<1>(total_size),
            [=](sycl::id<1> idx) {
                sycl::atomic_ref<float, sycl::memory_order::relaxed,
                               sycl::memory_scope::device> atomic_mean(mean_device[0]);
                atomic_mean += in_ptr[idx];
        }).wait();

        float mean_host;
        queue.memcpy(&mean_host, mean_device, sizeof(float)).wait();
        mean_host /= static_cast<float>(total_size);

        // Step 2: Compute variance
        float* var_device = sycl::malloc_device<float>(1, queue);
        queue.fill(var_device, 0.0f, 1).wait();

        queue.parallel_for<class VarKernelFloat32>(sycl::range<1>(total_size),
            [=](sycl::id<1> idx) {
                float diff = in_ptr[idx] - mean_host;
                sycl::atomic_ref<float, sycl::memory_order::relaxed,
                               sycl::memory_scope::device> atomic_var(var_device[0]);
                atomic_var += diff * diff;
        }).wait();

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
        queue.fill(mean_device, 0.0, 1).wait();

        queue.parallel_for<class MeanVarKernelFloat64>(sycl::range<1>(total_size),
            [=](sycl::id<1> idx) {
                sycl::atomic_ref<double, sycl::memory_order::relaxed,
                               sycl::memory_scope::device> atomic_mean(mean_device[0]);
                atomic_mean += in_ptr[idx];
        }).wait();

        double mean_host;
        queue.memcpy(&mean_host, mean_device, sizeof(double)).wait();
        mean_host /= static_cast<double>(total_size);

        // Step 2: Compute variance
        double* var_device = sycl::malloc_device<double>(1, queue);
        queue.fill(var_device, 0.0, 1).wait();

        queue.parallel_for<class VarKernelFloat64>(sycl::range<1>(total_size),
            [=](sycl::id<1> idx) {
                double diff = in_ptr[idx] - mean_host;
                sycl::atomic_ref<double, sycl::memory_order::relaxed,
                               sycl::memory_scope::device> atomic_var(var_device[0]);
                atomic_var += diff * diff;
        }).wait();

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
    int64_t dim = attrs.contains("dim") ? std::stoll(attrs.at("dim")) : -1;
    bool keepdim = attrs.contains("keepdim") && attrs.at("keepdim") == "1";

    auto shape = input.shape();

    // For now, only support reduction over all dimensions
    if (dim != -1 && dim != static_cast<int64_t>(shape.size()) - 1) {
        throw std::invalid_argument("prod: only dim=-1 (all dimensions) is currently supported for OneAPI");
    }

    // Compute total number of elements
    int64_t total_size = 1;
    for (auto s : shape) {
        total_size *= s;
    }

    if (input.dtype() == DType::Float32) {
        const float* in_ptr = get_data_ptr<const float>(input);

        // Use logarithmic reduction for better numerical stability
        // prod = exp(sum(log(x))) for positive values
        // For general case, we use sequential multiplication on host
        float prod_value = 1.0f;

        // Copy to host and compute product sequentially
        std::vector<float> host_data(total_size);
        queue.memcpy(host_data.data(), in_ptr, total_size * sizeof(float)).wait();

        for (int64_t i = 0; i < total_size; ++i) {
            prod_value *= host_data[i];
        }

        // Create output tensor
        std::vector<int64_t> out_shape;
        if (keepdim) {
            out_shape.resize(shape.size(), 1);
        } else {
            out_shape = {1};
        }

        Tensor output(out_shape, input.dtype(), input.device());
        float* out_ptr = get_data_ptr<float>(output);
        queue.fill(out_ptr, prod_value, 1).wait();

        return output;
    }
    else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);

        // Copy to host and compute product sequentially
        double prod_value = 1.0;

        std::vector<double> host_data(total_size);
        queue.memcpy(host_data.data(), in_ptr, total_size * sizeof(double)).wait();

        for (int64_t i = 0; i < total_size; ++i) {
            prod_value *= host_data[i];
        }

        // Create output tensor
        std::vector<int64_t> out_shape;
        if (keepdim) {
            out_shape.resize(shape.size(), 1);
        } else {
            out_shape = {1};
        }

        Tensor output(out_shape, input.dtype(), input.device());
        double* out_ptr = get_data_ptr<double>(output);
        queue.fill(out_ptr, prod_value, 1).wait();

        return output;
    }
    else {
        throw std::runtime_error("Unsupported dtype for prod_kernel");
    }
}

} // namespace oneapi
} // namespace tenzor
