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
class ProdReducePhase1Float32;
class ProdReducePhase2Float32;
class ProdReducePhase1Float64;
class ProdReducePhase2Float64;
class ProdDimKernelFloat32;
class ProdDimKernelFloat64;
class NormReducePhase1Float32;
class NormReducePhase2Float32;
class NormReducePhase1Float64;
class NormReducePhase2Float64;

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

    // Device-side product reduction using two-phase SYCL work-group reduction
    // Phase 1: nd_range kernel with work-groups of 256, grid-stride accumulation,
    //          work-group reduce to partial buffer (identity = 1, not 0)
    // Phase 2: Single work-group reduces partial results

    if (input.dtype() == DType::Float32) {
        const float* in_ptr = get_data_ptr<const float>(input);
        Tensor output(out_shape, input.dtype(), input.device());
        float* out_ptr = get_data_ptr<float>(output);

        if (dim == -1) {
            // Full reduction on device
            constexpr int64_t WG_SIZE = 256;
            int64_t num_wgs = (total_size + WG_SIZE - 1) / WG_SIZE;
            auto partial_buf = sycl::malloc_device<float>(num_wgs, queue);

            // Phase 1: each work-group computes partial product
            queue.parallel_for<ProdReducePhase1Float32>(
                sycl::nd_range<1>(num_wgs * WG_SIZE, WG_SIZE),
                [=](sycl::nd_item<1> item) {
                    int64_t gid = item.get_global_id(0);
                    float val = (gid < total_size) ? in_ptr[gid] : 1.0f;
                    float prod = sycl::reduce_over_group(
                        item.get_group(), val, 1.0f, std::multiplies<float>());
                    if (item.get_local_id(0) == 0) {
                        partial_buf[item.get_group(0)] = prod;
                    }
                }
            );

            // Phase 2: single work-group reduces partial products
            queue.parallel_for<ProdReducePhase2Float32>(
                sycl::nd_range<1>(WG_SIZE, WG_SIZE),
                [=](sycl::nd_item<1> item) {
                    int64_t lid = item.get_local_id(0);
                    float val = (lid < num_wgs) ? partial_buf[lid] : 1.0f;
                    float prod = sycl::reduce_over_group(
                        item.get_group(), val, 1.0f, std::multiplies<float>());
                    if (lid == 0) out_ptr[0] = prod;
                }
            );
            queue.wait();
            sycl::free(partial_buf, queue);
        } else {
            // Dimensional reduction on device
            const int64_t ndim = shape.size();
            const int64_t dim_size = shape[dim];

            // Precompute strides on device
            auto d_strides = sycl::malloc_device<int64_t>(ndim, queue);
            auto d_shape = sycl::malloc_device<int64_t>(ndim, queue);
            queue.memcpy(d_strides, strides.data(), ndim * sizeof(int64_t));
            queue.memcpy(d_shape, shape.data(), ndim * sizeof(int64_t));
            queue.wait();

            queue.parallel_for<ProdDimKernelFloat32>(
                sycl::range<1>(output_size), [=](sycl::id<1> gid) {
                    int64_t out_idx = gid[0];
                    int64_t tmp = out_idx;

                    // Compute base input index (with dim-th index = 0)
                    int64_t base_in_idx = 0;
                    for (int64_t d = ndim - 1; d >= 0; --d) {
                        if (d == dim) continue;
                        int64_t s = d_shape[d];
                        int64_t coord = tmp % s;
                        tmp /= s;
                        base_in_idx += coord * d_strides[d];
                    }

                    // Accumulate product along reduction dimension
                    float prod_value = 1.0f;
                    for (int64_t i = 0; i < dim_size; ++i) {
                        prod_value *= in_ptr[base_in_idx + i * d_strides[dim]];
                    }
                    out_ptr[out_idx] = prod_value;
                }
            );
            queue.wait();
            sycl::free(d_strides, queue);
            sycl::free(d_shape, queue);
        }

        return output;
    }
    else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        Tensor output(out_shape, input.dtype(), input.device());
        double* out_ptr = get_data_ptr<double>(output);

        if (dim == -1) {
            // Full reduction on device
            constexpr int64_t WG_SIZE = 256;
            int64_t num_wgs = (total_size + WG_SIZE - 1) / WG_SIZE;
            auto partial_buf = sycl::malloc_device<double>(num_wgs, queue);

            // Phase 1: each work-group computes partial product
            queue.parallel_for<ProdReducePhase1Float64>(
                sycl::nd_range<1>(num_wgs * WG_SIZE, WG_SIZE),
                [=](sycl::nd_item<1> item) {
                    int64_t gid = item.get_global_id(0);
                    double val = (gid < total_size) ? in_ptr[gid] : 1.0;
                    double prod = sycl::reduce_over_group(
                        item.get_group(), val, 1.0, std::multiplies<double>());
                    if (item.get_local_id(0) == 0) {
                        partial_buf[item.get_group(0)] = prod;
                    }
                }
            );

            // Phase 2: single work-group reduces partial products
            queue.parallel_for<ProdReducePhase2Float64>(
                sycl::nd_range<1>(WG_SIZE, WG_SIZE),
                [=](sycl::nd_item<1> item) {
                    int64_t lid = item.get_local_id(0);
                    double val = (lid < num_wgs) ? partial_buf[lid] : 1.0;
                    double prod = sycl::reduce_over_group(
                        item.get_group(), val, 1.0, std::multiplies<double>());
                    if (lid == 0) out_ptr[0] = prod;
                }
            );
            queue.wait();
            sycl::free(partial_buf, queue);
        } else {
            // Dimensional reduction on device
            const int64_t ndim = shape.size();
            const int64_t dim_size = shape[dim];

            auto d_strides = sycl::malloc_device<int64_t>(ndim, queue);
            auto d_shape = sycl::malloc_device<int64_t>(ndim, queue);
            queue.memcpy(d_strides, strides.data(), ndim * sizeof(int64_t));
            queue.memcpy(d_shape, shape.data(), ndim * sizeof(int64_t));
            queue.wait();

            queue.parallel_for<ProdDimKernelFloat64>(
                sycl::range<1>(output_size), [=](sycl::id<1> gid) {
                    int64_t out_idx = gid[0];
                    int64_t tmp = out_idx;

                    int64_t base_in_idx = 0;
                    for (int64_t d = ndim - 1; d >= 0; --d) {
                        if (d == dim) continue;
                        int64_t s = d_shape[d];
                        int64_t coord = tmp % s;
                        tmp /= s;
                        base_in_idx += coord * d_strides[d];
                    }

                    double prod_value = 1.0;
                    for (int64_t i = 0; i < dim_size; ++i) {
                        prod_value *= in_ptr[base_in_idx + i * d_strides[dim]];
                    }
                    out_ptr[out_idx] = prod_value;
                }
            );
            queue.wait();
            sycl::free(d_strides, queue);
            sycl::free(d_shape, queue);
        }

        return output;
    }
    else if (input.dtype() == DType::Int32) {
        const int32_t* in_ptr = get_data_ptr<const int32_t>(input);
        Tensor output(out_shape, input.dtype(), input.device());
        int32_t* out_ptr = get_data_ptr<int32_t>(output);

        if (dim == -1) {
            // Full reduction — use shared memory for int64 accumulator
            auto prod_buf = sycl::malloc_shared<int64_t>(1, queue);
            prod_buf[0] = 1;

            queue.parallel_for(sycl::range<1>(total_size),
                sycl::reduction(prod_buf, int64_t(1), std::multiplies<int64_t>()),
                [=](sycl::id<1> idx, auto& prod) {
                    prod.combine(static_cast<int64_t>(in_ptr[idx]));
                }
            );
            queue.wait();
            out_ptr[0] = static_cast<int32_t>(prod_buf[0]);
            queue.wait();
            sycl::free(prod_buf, queue);
        } else {
            // Dimensional reduction on device
            const int64_t ndim = shape.size();
            const int64_t dim_size = shape[dim];

            auto d_strides = sycl::malloc_device<int64_t>(ndim, queue);
            auto d_shape = sycl::malloc_device<int64_t>(ndim, queue);
            queue.memcpy(d_strides, strides.data(), ndim * sizeof(int64_t));
            queue.memcpy(d_shape, shape.data(), ndim * sizeof(int64_t));
            queue.wait();

            queue.parallel_for(
                sycl::range<1>(output_size), [=](sycl::id<1> gid) {
                    int64_t out_idx = gid[0];
                    int64_t tmp = out_idx;

                    int64_t base_in_idx = 0;
                    for (int64_t d = ndim - 1; d >= 0; --d) {
                        if (d == dim) continue;
                        int64_t s = d_shape[d];
                        int64_t coord = tmp % s;
                        tmp /= s;
                        base_in_idx += coord * d_strides[d];
                    }

                    int64_t prod_value = 1;
                    for (int64_t i = 0; i < dim_size; ++i) {
                        prod_value *= static_cast<int64_t>(
                            in_ptr[base_in_idx + i * d_strides[dim]]);
                    }
                    out_ptr[out_idx] = static_cast<int32_t>(prod_value);
                }
            );
            queue.wait();
            sycl::free(d_strides, queue);
            sycl::free(d_shape, queue);
        }

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

    // Device-side norm reduction using two-phase SYCL work-group reduction
    // Phase 1: nd_range kernel, work-groups of 256, apply norm transform, reduce to partials
    // Phase 2: Single work-group reduces partial results, apply final root

    if (input.dtype() == DType::Float32) {
        const float* in_ptr = get_data_ptr<const float>(input);
        Tensor output(out_shape, input.dtype(), input.device());
        float* out_ptr = get_data_ptr<float>(output);

        constexpr int64_t WG_SIZE = 256;
        int64_t num_wgs = (n + WG_SIZE - 1) / WG_SIZE;
        auto partial_buf = sycl::malloc_device<float>(num_wgs, queue);
        float p_val = p;

        // Phase 1: each work-group transforms and sums
        queue.parallel_for<NormReducePhase1Float32>(
            sycl::nd_range<1>(num_wgs * WG_SIZE, WG_SIZE),
            [=](sycl::nd_item<1> item) {
                int64_t gid = item.get_global_id(0);
                float val = 0.0f;
                if (gid < n) {
                    float x = in_ptr[gid];
                    if (p_val == 1.0f) {
                        val = sycl::fabs(x);
                    } else if (p_val == 2.0f) {
                        val = x * x;
                    } else {
                        val = sycl::pow(sycl::fabs(x), p_val);
                    }
                }
                float sum = sycl::reduce_over_group(
                    item.get_group(), val, sycl::plus<float>());
                if (item.get_local_id(0) == 0) {
                    partial_buf[item.get_group(0)] = sum;
                }
            }
        );

        // Phase 2: single work-group reduces partial sums, apply root
        queue.parallel_for<NormReducePhase2Float32>(
            sycl::nd_range<1>(WG_SIZE, WG_SIZE),
            [=](sycl::nd_item<1> item) {
                int64_t lid = item.get_local_id(0);
                float val = (lid < num_wgs) ? partial_buf[lid] : 0.0f;
                float sum = sycl::reduce_over_group(
                    item.get_group(), val, sycl::plus<float>());
                if (lid == 0) {
                    if (p_val == 2.0f) {
                        out_ptr[0] = sycl::sqrt(sum);
                    } else if (p_val == 1.0f) {
                        out_ptr[0] = sum;
                    } else {
                        out_ptr[0] = sycl::pow(sum, 1.0f / p_val);
                    }
                }
            }
        );
        queue.wait();
        sycl::free(partial_buf, queue);

        return output;
    }
    else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        Tensor output(out_shape, input.dtype(), input.device());
        double* out_ptr = get_data_ptr<double>(output);

        constexpr int64_t WG_SIZE = 256;
        int64_t num_wgs = (n + WG_SIZE - 1) / WG_SIZE;
        auto partial_buf = sycl::malloc_device<double>(num_wgs, queue);
        double p_val = static_cast<double>(p);

        // Phase 1: each work-group transforms and sums
        queue.parallel_for<NormReducePhase1Float64>(
            sycl::nd_range<1>(num_wgs * WG_SIZE, WG_SIZE),
            [=](sycl::nd_item<1> item) {
                int64_t gid = item.get_global_id(0);
                double val = 0.0;
                if (gid < n) {
                    double x = in_ptr[gid];
                    if (p_val == 1.0) {
                        val = sycl::fabs(x);
                    } else if (p_val == 2.0) {
                        val = x * x;
                    } else {
                        val = sycl::pow(sycl::fabs(x), p_val);
                    }
                }
                double sum = sycl::reduce_over_group(
                    item.get_group(), val, sycl::plus<double>());
                if (item.get_local_id(0) == 0) {
                    partial_buf[item.get_group(0)] = sum;
                }
            }
        );

        // Phase 2: single work-group reduces partial sums, apply root
        queue.parallel_for<NormReducePhase2Float64>(
            sycl::nd_range<1>(WG_SIZE, WG_SIZE),
            [=](sycl::nd_item<1> item) {
                int64_t lid = item.get_local_id(0);
                double val = (lid < num_wgs) ? partial_buf[lid] : 0.0;
                double sum = sycl::reduce_over_group(
                    item.get_group(), val, sycl::plus<double>());
                if (lid == 0) {
                    if (p_val == 2.0) {
                        out_ptr[0] = sycl::sqrt(sum);
                    } else if (p_val == 1.0) {
                        out_ptr[0] = sum;
                    } else {
                        out_ptr[0] = sycl::pow(sum, 1.0 / p_val);
                    }
                }
            }
        );
        queue.wait();
        sycl::free(partial_buf, queue);

        return output;
    }
    else {
        throw std::runtime_error("Unsupported dtype for norm_kernel");
    }
}

} // namespace oneapi
} // namespace tenzor
