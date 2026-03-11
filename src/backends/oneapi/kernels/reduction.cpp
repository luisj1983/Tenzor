#include "tenzor/core/tensor.hpp"
#include <sycl/sycl.hpp>
#include <limits>
#include <numeric>
#include <algorithm>
#include <stdexcept>

#ifdef TENZOR_HAS_ONEDPL
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/iterator>
#endif

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
class SumKernelFloat16;
class SumKernelBFloat16;
class SumKernelInt32;
class SumKernelInt64;
class SumKernelBool;
class MeanKernelFloat32;
class MeanKernelFloat64;
class MeanKernelFloat16;
class MeanKernelBFloat16;
class MaxKernelFloat32;
class MaxKernelFloat64;
class MaxKernelFloat16;
class MaxKernelBFloat16;
class MaxKernelInt32;
class MinKernelFloat32;
class MinKernelFloat64;
class MinKernelFloat16;
class MinKernelBFloat16;
class MinKernelInt32;
class ArgmaxKernelFloat32;
class ArgmaxKernelFloat64;
class ArgmaxKernelFloat16;
class ArgmaxKernelInt32;
class ArgminKernelFloat32;
class ArgminKernelFloat64;
class ArgminKernelFloat16;
class ArgminKernelInt32;

// Helper function to get typed pointer from tensor
template<typename T>
inline auto get_data_ptr(const Tensor& t) -> T* {
    return static_cast<T*>(const_cast<void*>(t.data_ptr()));
}

// BFloat16 <-> Float32 conversion helpers (device-compatible)
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
    const int64_t ndim = static_cast<int64_t>(shape.size());

    // INT64_MIN indicates full reduction (sum all elements)
    bool is_full_reduction = (dim == INT64_MIN);

    if (!is_full_reduction) {
        // Normalize negative dimensions (e.g., -1 means last dimension)
        if (dim < 0) {
            dim += ndim;
        }
        // Validate dimension is within bounds
        if (dim < 0 || dim >= ndim) {
            throw std::runtime_error("Dimension " + std::to_string(dim) +
                " out of range for tensor with " + std::to_string(ndim) + " dimensions");
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
                queue.single_task([=]() { out_ptr[0] = 0.0f; });
            } else if (in_cont.dtype() == DType::Float64) {
                double* out_ptr = get_data_ptr<double>(output);
                queue.single_task([=]() { out_ptr[0] = 0.0; });
            } else if (in_cont.dtype() == DType::Int32) {
                int32_t* out_ptr = get_data_ptr<int32_t>(output);
                queue.single_task([=]() { out_ptr[0] = 0; });
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
            });

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
            });

            out_ptr[0] = sum_buf[0];
            queue.wait();
            sycl::free(sum_buf, queue);
        }
        else if (in_cont.dtype() == DType::Float16) {
            const sycl::half* in_ptr = get_data_ptr<const sycl::half>(in_cont);
            sycl::half* out_ptr = get_data_ptr<sycl::half>(output);

            // Use float accumulation for precision
            auto sum_buf = sycl::malloc_shared<float>(1, queue);
            sum_buf[0] = 0.0f;

            queue.parallel_for(sycl::range<1>(total_size), sycl::reduction(sum_buf, sycl::plus<float>()),
                              [=](sycl::id<1> idx, auto& sum) {
                sum += static_cast<float>(in_ptr[idx]);
            });

            out_ptr[0] = sycl::half(sum_buf[0]);
            queue.wait();
            sycl::free(sum_buf, queue);
        }
        else if (in_cont.dtype() == DType::BFloat16) {
            const uint16_t* in_ptr = get_data_ptr<const uint16_t>(in_cont);
            uint16_t* out_ptr = get_data_ptr<uint16_t>(output);

            // Use float accumulation for precision
            auto sum_buf = sycl::malloc_shared<float>(1, queue);
            sum_buf[0] = 0.0f;

            queue.parallel_for(sycl::range<1>(total_size), sycl::reduction(sum_buf, sycl::plus<float>()),
                              [=](sycl::id<1> idx, auto& sum) {
                sum += bf16_to_f32(in_ptr[idx]);
            });

            out_ptr[0] = f32_to_bf16(sum_buf[0]);
            queue.wait();
            sycl::free(sum_buf, queue);
        }
        else if (in_cont.dtype() == DType::Int32) {
            const int32_t* in_ptr = get_data_ptr<const int32_t>(in_cont);
            int32_t* out_ptr = get_data_ptr<int32_t>(output);

            // Use int64 accumulation to avoid overflow
            auto sum_buf = sycl::malloc_shared<int64_t>(1, queue);
            sum_buf[0] = 0;

            queue.parallel_for(sycl::range<1>(total_size), sycl::reduction(sum_buf, sycl::plus<int64_t>()),
                              [=](sycl::id<1> idx, auto& sum) {
                sum += static_cast<int64_t>(in_ptr[idx]);
            });

            out_ptr[0] = static_cast<int32_t>(sum_buf[0]);
            queue.wait();
            sycl::free(sum_buf, queue);
        }
        else if (in_cont.dtype() == DType::Int64) {
            const int64_t* in_ptr = get_data_ptr<const int64_t>(in_cont);
            int64_t* out_ptr = get_data_ptr<int64_t>(output);

            // Host-side summation for Int64 (avoids SYCL reduction issues with int64)
            std::vector<int64_t> host_data(total_size);
            queue.memcpy(host_data.data(), in_ptr, total_size * sizeof(int64_t)).wait();

            int64_t sum = 0;
            for (int64_t i = 0; i < total_size; ++i) {
                sum += host_data[i];
            }

            queue.memcpy(out_ptr, &sum, sizeof(int64_t)).wait();
        }
        else if (in_cont.dtype() == DType::Bool) {
            const bool* in_ptr = get_data_ptr<const bool>(in_cont);
            // Sum of bools returns Int64
            Tensor bool_output({}, DType::Int64, output.device());
            int64_t* out_ptr = get_data_ptr<int64_t>(bool_output);

            std::vector<bool> host_data(total_size);
            // Copy bool data to host
            std::vector<uint8_t> host_raw(total_size);
            queue.memcpy(host_raw.data(), in_ptr, total_size * sizeof(bool)).wait();

            int64_t sum = 0;
            for (int64_t i = 0; i < total_size; ++i) {
                sum += host_raw[i] ? 1 : 0;
            }

            queue.memcpy(out_ptr, &sum, sizeof(int64_t)).wait();
            return bool_output;
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
            });
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
            });
        }
        else if (in_cont.dtype() == DType::Float16) {
            const sycl::half* in_ptr = get_data_ptr<const sycl::half>(in_cont);
            sycl::half* out_ptr = get_data_ptr<sycl::half>(output);

            queue.parallel_for<SumKernelFloat16>(sycl::range<2>(outer_size, inner_size), [=](sycl::id<2> idx) {
                const int64_t outer_idx = idx[0];
                const int64_t inner_idx = idx[1];
                const int64_t base_offset = outer_idx * dim_size * inner_size + inner_idx;

                // Use float accumulation for precision
                float sum = 0.0f;
                for (int64_t d = 0; d < dim_size; ++d) {
                    sum += static_cast<float>(in_ptr[base_offset + d * inner_size]);
                }

                out_ptr[outer_idx * inner_size + inner_idx] = sycl::half(sum);
            });
        }
        else if (in_cont.dtype() == DType::BFloat16) {
            const uint16_t* in_ptr = get_data_ptr<const uint16_t>(in_cont);
            uint16_t* out_ptr = get_data_ptr<uint16_t>(output);

            queue.parallel_for<SumKernelBFloat16>(sycl::range<2>(outer_size, inner_size), [=](sycl::id<2> idx) {
                const int64_t outer_idx = idx[0];
                const int64_t inner_idx = idx[1];
                const int64_t base_offset = outer_idx * dim_size * inner_size + inner_idx;

                // Use float accumulation for precision
                float sum = 0.0f;
                for (int64_t d = 0; d < dim_size; ++d) {
                    sum += bf16_to_f32(in_ptr[base_offset + d * inner_size]);
                }

                out_ptr[outer_idx * inner_size + inner_idx] = f32_to_bf16(sum);
            });
        }
        else if (in_cont.dtype() == DType::Int32) {
            const int32_t* in_ptr = get_data_ptr<const int32_t>(in_cont);
            int32_t* out_ptr = get_data_ptr<int32_t>(output);

            queue.parallel_for<SumKernelInt32>(sycl::range<2>(outer_size, inner_size), [=](sycl::id<2> idx) {
                const int64_t outer_idx = idx[0];
                const int64_t inner_idx = idx[1];
                const int64_t base_offset = outer_idx * dim_size * inner_size + inner_idx;

                // Use int64 accumulation to avoid overflow
                int64_t sum = 0;
                for (int64_t d = 0; d < dim_size; ++d) {
                    sum += static_cast<int64_t>(in_ptr[base_offset + d * inner_size]);
                }

                out_ptr[outer_idx * inner_size + inner_idx] = static_cast<int32_t>(sum);
            });
        }
        else if (in_cont.dtype() == DType::Int64) {
            const int64_t* in_ptr = get_data_ptr<const int64_t>(in_cont);
            int64_t* out_ptr = get_data_ptr<int64_t>(output);

            queue.parallel_for<SumKernelInt64>(sycl::range<2>(outer_size, inner_size), [=](sycl::id<2> idx) {
                const int64_t outer_idx = idx[0];
                const int64_t inner_idx = idx[1];
                const int64_t base_offset = outer_idx * dim_size * inner_size + inner_idx;

                int64_t sum = 0;
                for (int64_t d = 0; d < dim_size; ++d) {
                    sum += in_ptr[base_offset + d * inner_size];
                }

                out_ptr[outer_idx * inner_size + inner_idx] = sum;
            });
        }
        else if (in_cont.dtype() == DType::Bool) {
            const bool* in_ptr = get_data_ptr<const bool>(in_cont);
            // Sum of bools along dim returns Int64
            std::vector<int64_t> bool_out_shape(out_shape.begin(), out_shape.end());
            Tensor bool_output(bool_out_shape, DType::Int64, output.device());
            int64_t* out_ptr = get_data_ptr<int64_t>(bool_output);

            queue.parallel_for<SumKernelBool>(sycl::range<2>(outer_size, inner_size), [=](sycl::id<2> idx) {
                const int64_t outer_idx = idx[0];
                const int64_t inner_idx = idx[1];
                const int64_t base_offset = outer_idx * dim_size * inner_size + inner_idx;

                int64_t sum = 0;
                for (int64_t d = 0; d < dim_size; ++d) {
                    sum += in_ptr[base_offset + d * inner_size] ? 1 : 0;
                }

                out_ptr[outer_idx * inner_size + inner_idx] = sum;
            });
            return bool_output;
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

    // Normalize negative dimensions first (e.g., -1 means last dimension)
    if (dim < 0 && dim >= -static_cast<int64_t>(shape.size())) {
        dim += shape.size();
    }

    // Check if this is a full reduction (dim still negative after normalization means full reduction)
    bool is_full_reduction = (dim < 0 || dim >= static_cast<int64_t>(shape.size()));

    if (!is_full_reduction) {
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
            });

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
            });

            out_ptr[0] = sum_buf[0] * scale;
            queue.wait();
            sycl::free(sum_buf, queue);
        }
        else if (in_cont.dtype() == DType::Float16) {
            const sycl::half* in_ptr = get_data_ptr<const sycl::half>(in_cont);
            sycl::half* out_ptr = get_data_ptr<sycl::half>(output);
            const float scale = 1.0f / static_cast<float>(total_size);

            auto sum_buf = sycl::malloc_shared<float>(1, queue);
            sum_buf[0] = 0.0f;

            queue.parallel_for(sycl::range<1>(total_size), sycl::reduction(sum_buf, sycl::plus<float>()),
                              [=](sycl::id<1> idx, auto& sum) {
                sum += static_cast<float>(in_ptr[idx]);
            });

            out_ptr[0] = sycl::half(sum_buf[0] * scale);
            queue.wait();
            sycl::free(sum_buf, queue);
        }
        else if (in_cont.dtype() == DType::BFloat16) {
            const uint16_t* in_ptr = get_data_ptr<const uint16_t>(in_cont);
            uint16_t* out_ptr = get_data_ptr<uint16_t>(output);
            const float scale = 1.0f / static_cast<float>(total_size);

            auto sum_buf = sycl::malloc_shared<float>(1, queue);
            sum_buf[0] = 0.0f;

            queue.parallel_for(sycl::range<1>(total_size), sycl::reduction(sum_buf, sycl::plus<float>()),
                              [=](sycl::id<1> idx, auto& sum) {
                sum += bf16_to_f32(in_ptr[idx]);
            });

            out_ptr[0] = f32_to_bf16(sum_buf[0] * scale);
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
            });
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
            });
        }
        else if (in_cont.dtype() == DType::Float16) {
            const sycl::half* in_ptr = get_data_ptr<const sycl::half>(in_cont);
            sycl::half* out_ptr = get_data_ptr<sycl::half>(output);
            const float scale = 1.0f / static_cast<float>(dim_size);

            queue.parallel_for<MeanKernelFloat16>(sycl::range<2>(outer_size, inner_size), [=](sycl::id<2> idx) {
                const int64_t outer_idx = idx[0];
                const int64_t inner_idx = idx[1];
                const int64_t base_offset = outer_idx * dim_size * inner_size + inner_idx;

                float sum = 0.0f;
                for (int64_t d = 0; d < dim_size; ++d) {
                    sum += static_cast<float>(in_ptr[base_offset + d * inner_size]);
                }

                out_ptr[outer_idx * inner_size + inner_idx] = sycl::half(sum * scale);
            });
        }
        else if (in_cont.dtype() == DType::BFloat16) {
            const uint16_t* in_ptr = get_data_ptr<const uint16_t>(in_cont);
            uint16_t* out_ptr = get_data_ptr<uint16_t>(output);
            const float scale = 1.0f / static_cast<float>(dim_size);

            queue.parallel_for<MeanKernelBFloat16>(sycl::range<2>(outer_size, inner_size), [=](sycl::id<2> idx) {
                const int64_t outer_idx = idx[0];
                const int64_t inner_idx = idx[1];
                const int64_t base_offset = outer_idx * dim_size * inner_size + inner_idx;

                float sum = 0.0f;
                for (int64_t d = 0; d < dim_size; ++d) {
                    sum += bf16_to_f32(in_ptr[base_offset + d * inner_size]);
                }

                out_ptr[outer_idx * inner_size + inner_idx] = f32_to_bf16(sum * scale);
            });
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

    // Normalize negative dimensions first (e.g., -1 means last dimension)
    if (dim < 0 && dim >= -static_cast<int64_t>(shape.size())) {
        dim += shape.size();
    }

    // Check if this is a full reduction (dim still negative after normalization means full reduction)
    bool is_full_reduction = (dim < 0 || dim >= static_cast<int64_t>(shape.size()));

    if (!is_full_reduction) {
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
            });
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
            });
        }
        else if (input.dtype() == DType::Float16) {
            const sycl::half* in_ptr = get_data_ptr<const sycl::half>(input);
            sycl::half* out_ptr = get_data_ptr<sycl::half>(output);

            queue.single_task([=]() {
                float max_val = -std::numeric_limits<float>::infinity();
                for (int64_t i = 0; i < total_size; ++i) {
                    max_val = sycl::fmax(max_val, static_cast<float>(in_ptr[i]));
                }
                out_ptr[0] = sycl::half(max_val);
            });
        }
        else if (input.dtype() == DType::BFloat16) {
            const uint16_t* in_ptr = get_data_ptr<const uint16_t>(input);
            uint16_t* out_ptr = get_data_ptr<uint16_t>(output);

            queue.single_task([=]() {
                float max_val = -std::numeric_limits<float>::infinity();
                for (int64_t i = 0; i < total_size; ++i) {
                    max_val = sycl::fmax(max_val, bf16_to_f32(in_ptr[i]));
                }
                out_ptr[0] = f32_to_bf16(max_val);
            });
        }
        else if (input.dtype() == DType::Int32) {
            const int32_t* in_ptr = get_data_ptr<const int32_t>(input);
            int32_t* out_ptr = get_data_ptr<int32_t>(output);

            queue.single_task([=]() {
                int32_t max_val = std::numeric_limits<int32_t>::min();
                for (int64_t i = 0; i < total_size; ++i) {
                    max_val = sycl::max(max_val, in_ptr[i]);
                }
                out_ptr[0] = max_val;
            });
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
            });
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
            });
        }
        else if (input.dtype() == DType::Float16) {
            const sycl::half* in_ptr = get_data_ptr<const sycl::half>(input);
            sycl::half* out_ptr = get_data_ptr<sycl::half>(output);

            queue.parallel_for<MaxKernelFloat16>(sycl::range<2>(outer_size, inner_size), [=](sycl::id<2> idx) {
                const int64_t outer_idx = idx[0];
                const int64_t inner_idx = idx[1];
                const int64_t base_offset = outer_idx * dim_size * inner_size + inner_idx;

                float max_val = -std::numeric_limits<float>::infinity();
                for (int64_t d = 0; d < dim_size; ++d) {
                    float val = static_cast<float>(in_ptr[base_offset + d * inner_size]);
                    max_val = sycl::fmax(max_val, val);
                }

                out_ptr[outer_idx * inner_size + inner_idx] = sycl::half(max_val);
            });
        }
        else if (input.dtype() == DType::BFloat16) {
            const uint16_t* in_ptr = get_data_ptr<const uint16_t>(input);
            uint16_t* out_ptr = get_data_ptr<uint16_t>(output);

            queue.parallel_for<MaxKernelBFloat16>(sycl::range<2>(outer_size, inner_size), [=](sycl::id<2> idx) {
                const int64_t outer_idx = idx[0];
                const int64_t inner_idx = idx[1];
                const int64_t base_offset = outer_idx * dim_size * inner_size + inner_idx;

                float max_val = -std::numeric_limits<float>::infinity();
                for (int64_t d = 0; d < dim_size; ++d) {
                    float val = bf16_to_f32(in_ptr[base_offset + d * inner_size]);
                    max_val = sycl::fmax(max_val, val);
                }

                out_ptr[outer_idx * inner_size + inner_idx] = f32_to_bf16(max_val);
            });
        }
        else if (input.dtype() == DType::Int32) {
            const int32_t* in_ptr = get_data_ptr<const int32_t>(input);
            int32_t* out_ptr = get_data_ptr<int32_t>(output);

            queue.parallel_for<MaxKernelInt32>(sycl::range<2>(outer_size, inner_size), [=](sycl::id<2> idx) {
                const int64_t outer_idx = idx[0];
                const int64_t inner_idx = idx[1];
                const int64_t base_offset = outer_idx * dim_size * inner_size + inner_idx;

                int32_t max_val = std::numeric_limits<int32_t>::min();
                for (int64_t d = 0; d < dim_size; ++d) {
                    int32_t val = in_ptr[base_offset + d * inner_size];
                    max_val = sycl::max(max_val, val);
                }

                out_ptr[outer_idx * inner_size + inner_idx] = max_val;
            });
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

    // Normalize negative dimensions first (e.g., -1 means last dimension)
    if (dim < 0 && dim >= -static_cast<int64_t>(shape.size())) {
        dim += shape.size();
    }

    // Check if this is a full reduction (dim still negative after normalization means full reduction)
    bool is_full_reduction = (dim < 0 || dim >= static_cast<int64_t>(shape.size()));

    if (!is_full_reduction) {
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
            });
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
            });
        }
        else if (input.dtype() == DType::Float16) {
            const sycl::half* in_ptr = get_data_ptr<const sycl::half>(input);
            sycl::half* out_ptr = get_data_ptr<sycl::half>(output);

            queue.single_task([=]() {
                float min_val = std::numeric_limits<float>::infinity();
                for (int64_t i = 0; i < total_size; ++i) {
                    min_val = sycl::fmin(min_val, static_cast<float>(in_ptr[i]));
                }
                out_ptr[0] = sycl::half(min_val);
            });
        }
        else if (input.dtype() == DType::BFloat16) {
            const uint16_t* in_ptr = get_data_ptr<const uint16_t>(input);
            uint16_t* out_ptr = get_data_ptr<uint16_t>(output);

            queue.single_task([=]() {
                float min_val = std::numeric_limits<float>::infinity();
                for (int64_t i = 0; i < total_size; ++i) {
                    min_val = sycl::fmin(min_val, bf16_to_f32(in_ptr[i]));
                }
                out_ptr[0] = f32_to_bf16(min_val);
            });
        }
        else if (input.dtype() == DType::Int32) {
            const int32_t* in_ptr = get_data_ptr<const int32_t>(input);
            int32_t* out_ptr = get_data_ptr<int32_t>(output);

            queue.single_task([=]() {
                int32_t min_val = std::numeric_limits<int32_t>::max();
                for (int64_t i = 0; i < total_size; ++i) {
                    min_val = sycl::min(min_val, in_ptr[i]);
                }
                out_ptr[0] = min_val;
            });
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
            });
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
            });
        }
        else if (input.dtype() == DType::Float16) {
            const sycl::half* in_ptr = get_data_ptr<const sycl::half>(input);
            sycl::half* out_ptr = get_data_ptr<sycl::half>(output);

            queue.parallel_for<MinKernelFloat16>(sycl::range<2>(outer_size, inner_size), [=](sycl::id<2> idx) {
                const int64_t outer_idx = idx[0];
                const int64_t inner_idx = idx[1];
                const int64_t base_offset = outer_idx * dim_size * inner_size + inner_idx;

                float min_val = std::numeric_limits<float>::infinity();
                for (int64_t d = 0; d < dim_size; ++d) {
                    float val = static_cast<float>(in_ptr[base_offset + d * inner_size]);
                    min_val = sycl::fmin(min_val, val);
                }

                out_ptr[outer_idx * inner_size + inner_idx] = sycl::half(min_val);
            });
        }
        else if (input.dtype() == DType::BFloat16) {
            const uint16_t* in_ptr = get_data_ptr<const uint16_t>(input);
            uint16_t* out_ptr = get_data_ptr<uint16_t>(output);

            queue.parallel_for<MinKernelBFloat16>(sycl::range<2>(outer_size, inner_size), [=](sycl::id<2> idx) {
                const int64_t outer_idx = idx[0];
                const int64_t inner_idx = idx[1];
                const int64_t base_offset = outer_idx * dim_size * inner_size + inner_idx;

                float min_val = std::numeric_limits<float>::infinity();
                for (int64_t d = 0; d < dim_size; ++d) {
                    float val = bf16_to_f32(in_ptr[base_offset + d * inner_size]);
                    min_val = sycl::fmin(min_val, val);
                }

                out_ptr[outer_idx * inner_size + inner_idx] = f32_to_bf16(min_val);
            });
        }
        else if (input.dtype() == DType::Int32) {
            const int32_t* in_ptr = get_data_ptr<const int32_t>(input);
            int32_t* out_ptr = get_data_ptr<int32_t>(output);

            queue.parallel_for<MinKernelInt32>(sycl::range<2>(outer_size, inner_size), [=](sycl::id<2> idx) {
                const int64_t outer_idx = idx[0];
                const int64_t inner_idx = idx[1];
                const int64_t base_offset = outer_idx * dim_size * inner_size + inner_idx;

                int32_t min_val = std::numeric_limits<int32_t>::max();
                for (int64_t d = 0; d < dim_size; ++d) {
                    int32_t val = in_ptr[base_offset + d * inner_size];
                    min_val = sycl::min(min_val, val);
                }

                out_ptr[outer_idx * inner_size + inner_idx] = min_val;
            });
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
        else if (input.dtype() == DType::Float16) {
            const sycl::half* in_ptr = get_data_ptr<const sycl::half>(input);

            // Copy data to host and find argmax
            std::vector<sycl::half> host_data(total_size);
            queue.memcpy(host_data.data(), in_ptr, total_size * sizeof(sycl::half)).wait();

            int64_t max_idx = 0;
            float max_val = static_cast<float>(host_data[0]);
            for (int64_t i = 1; i < total_size; ++i) {
                float val = static_cast<float>(host_data[i]);
                if (val > max_val) {
                    max_val = val;
                    max_idx = i;
                }
            }

            queue.memcpy(out_ptr, &max_idx, sizeof(int64_t)).wait();
        }
        else if (input.dtype() == DType::Int32) {
            const int32_t* in_ptr = get_data_ptr<const int32_t>(input);

            // Copy data to host and find argmax
            std::vector<int32_t> host_data(total_size);
            queue.memcpy(host_data.data(), in_ptr, total_size * sizeof(int32_t)).wait();

            int64_t max_idx = 0;
            int32_t max_val = host_data[0];
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
        });
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
        });
    }
    else if (input.dtype() == DType::Float16) {
        const sycl::half* in_ptr = get_data_ptr<const sycl::half>(input);

        queue.parallel_for<ArgmaxKernelFloat16>(sycl::range<2>(outer_size, inner_size), [=](sycl::id<2> idx) {
            const int64_t outer_idx = idx[0];
            const int64_t inner_idx = idx[1];
            const int64_t base_offset = outer_idx * dim_size * inner_size + inner_idx;

            float max_val = -std::numeric_limits<float>::infinity();
            int64_t max_idx = 0;
            for (int64_t d = 0; d < dim_size; ++d) {
                float val = static_cast<float>(in_ptr[base_offset + d * inner_size]);
                if (val > max_val) {
                    max_val = val;
                    max_idx = d;
                }
            }

            out_ptr[outer_idx * inner_size + inner_idx] = max_idx;
        });
    }
    else if (input.dtype() == DType::Int32) {
        const int32_t* in_ptr = get_data_ptr<const int32_t>(input);

        queue.parallel_for<ArgmaxKernelInt32>(sycl::range<2>(outer_size, inner_size), [=](sycl::id<2> idx) {
            const int64_t outer_idx = idx[0];
            const int64_t inner_idx = idx[1];
            const int64_t base_offset = outer_idx * dim_size * inner_size + inner_idx;

            int32_t max_val = std::numeric_limits<int32_t>::min();
            int64_t max_idx = 0;
            for (int64_t d = 0; d < dim_size; ++d) {
                int32_t val = in_ptr[base_offset + d * inner_size];
                if (val > max_val) {
                    max_val = val;
                    max_idx = d;
                }
            }

            out_ptr[outer_idx * inner_size + inner_idx] = max_idx;
        });
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
        else if (input.dtype() == DType::Float16) {
            const sycl::half* in_ptr = get_data_ptr<const sycl::half>(input);

            // Copy data to host and find argmin
            std::vector<sycl::half> host_data(total_size);
            queue.memcpy(host_data.data(), in_ptr, total_size * sizeof(sycl::half)).wait();

            int64_t min_idx = 0;
            float min_val = static_cast<float>(host_data[0]);
            for (int64_t i = 1; i < total_size; ++i) {
                float val = static_cast<float>(host_data[i]);
                if (val < min_val) {
                    min_val = val;
                    min_idx = i;
                }
            }

            queue.memcpy(out_ptr, &min_idx, sizeof(int64_t)).wait();
        }
        else if (input.dtype() == DType::Int32) {
            const int32_t* in_ptr = get_data_ptr<const int32_t>(input);

            // Copy data to host and find argmin
            std::vector<int32_t> host_data(total_size);
            queue.memcpy(host_data.data(), in_ptr, total_size * sizeof(int32_t)).wait();

            int64_t min_idx = 0;
            int32_t min_val = host_data[0];
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
        });
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
        });
    }
    else if (input.dtype() == DType::Float16) {
        const sycl::half* in_ptr = get_data_ptr<const sycl::half>(input);

        queue.parallel_for<ArgminKernelFloat16>(sycl::range<2>(outer_size, inner_size), [=](sycl::id<2> idx) {
            const int64_t outer_idx = idx[0];
            const int64_t inner_idx = idx[1];
            const int64_t base_offset = outer_idx * dim_size * inner_size + inner_idx;

            float min_val = std::numeric_limits<float>::infinity();
            int64_t min_idx = 0;
            for (int64_t d = 0; d < dim_size; ++d) {
                float val = static_cast<float>(in_ptr[base_offset + d * inner_size]);
                if (val < min_val) {
                    min_val = val;
                    min_idx = d;
                }
            }

            out_ptr[outer_idx * inner_size + inner_idx] = min_idx;
        });
    }
    else if (input.dtype() == DType::Int32) {
        const int32_t* in_ptr = get_data_ptr<const int32_t>(input);

        queue.parallel_for<ArgminKernelInt32>(sycl::range<2>(outer_size, inner_size), [=](sycl::id<2> idx) {
            const int64_t outer_idx = idx[0];
            const int64_t inner_idx = idx[1];
            const int64_t base_offset = outer_idx * dim_size * inner_size + inner_idx;

            int32_t min_val = std::numeric_limits<int32_t>::max();
            int64_t min_idx = 0;
            for (int64_t d = 0; d < dim_size; ++d) {
                int32_t val = in_ptr[base_offset + d * inner_size];
                if (val < min_val) {
                    min_val = val;
                    min_idx = d;
                }
            }

            out_ptr[outer_idx * inner_size + inner_idx] = min_idx;
        });
    }
    else {
        throw std::runtime_error("Unsupported dtype for argmin");
    }

    return output;
}

// ============================================================================
// Sort kernel - sort along a dimension, returns (values, indices)
// ============================================================================
auto sort_kernel(const Tensor& input, int64_t dim, bool descending, sycl::queue& queue)
    -> std::pair<Tensor, Tensor> {
    auto shape_span = input.shape();
    std::vector<int64_t> shape(shape_span.begin(), shape_span.end());
    int64_t ndim = shape.size();
    if (dim < 0) dim += ndim;

    Tensor values(shape, input.dtype(), input.device());
    Tensor indices(shape, DType::Int64, input.device());
    int64_t numel = input.numel();
    if (numel == 0) return {values, indices};

    int64_t outer_size = 1, inner_size = 1;
    for (int64_t i = 0; i < dim; ++i) outer_size *= shape[i];
    for (int64_t i = dim + 1; i < ndim; ++i) inner_size *= shape[i];
    int64_t dim_size = shape[dim];

#ifdef TENZOR_HAS_ONEDPL
    // Device-side sort for contiguous sort dimension (inner_size == 1)
    if (inner_size == 1) {
        auto policy = oneapi::dpl::execution::make_device_policy(queue);

        auto device_sort_impl = [&](auto* val_ptr, const auto* in_ptr) {
            using T = std::remove_const_t<std::remove_pointer_t<decltype(in_ptr)>>;
            int64_t* idx_ptr = get_data_ptr<int64_t>(indices);

            // Copy input to values buffer
            queue.memcpy(val_ptr, in_ptr, numel * sizeof(T)).wait();

            // Initialize indices: each slice gets 0..dim_size-1
            queue.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> gid) {
                idx_ptr[gid] = static_cast<int64_t>(gid[0]) % dim_size;
            }).wait();

            // Sort each contiguous slice using oneDPL sort_by_key
            for (int64_t o = 0; o < outer_size; ++o) {
                T* slice_vals = val_ptr + o * dim_size;
                int64_t* slice_idx = idx_ptr + o * dim_size;
                if (descending) {
                    oneapi::dpl::sort_by_key(policy, slice_vals, slice_vals + dim_size,
                                              slice_idx, std::greater<T>());
                } else {
                    oneapi::dpl::sort_by_key(policy, slice_vals, slice_vals + dim_size,
                                              slice_idx);
                }
            }
        };

        if (input.dtype() == DType::Float32) {
            device_sort_impl(get_data_ptr<float>(values), get_data_ptr<const float>(input));
        } else if (input.dtype() == DType::Float64) {
            device_sort_impl(get_data_ptr<double>(values), get_data_ptr<const double>(input));
        } else if (input.dtype() == DType::Int32) {
            device_sort_impl(get_data_ptr<int32_t>(values), get_data_ptr<const int32_t>(input));
        } else if (input.dtype() == DType::Int64) {
            device_sort_impl(get_data_ptr<int64_t>(values), get_data_ptr<const int64_t>(input));
        } else if (input.dtype() == DType::Float16) {
            // FP16: upcast to Float32, sort on device, downcast
            Tensor input_f32 = input.to(DType::Float32);
            auto [vals_f32, idx] = sort_kernel(input_f32, dim, descending, queue);
            queue.memcpy(get_data_ptr<int64_t>(indices),
                         get_data_ptr<const int64_t>(idx), numel * sizeof(int64_t)).wait();
            Tensor vals_f16 = vals_f32.to(DType::Float16);
            queue.memcpy(values.data_ptr(), vals_f16.data_ptr(),
                         numel * dtype_size(DType::Float16)).wait();
            return {values, indices};
        } else if (input.dtype() == DType::BFloat16) {
            Tensor input_f32 = input.to(DType::Float32);
            auto [vals_f32, idx] = sort_kernel(input_f32, dim, descending, queue);
            queue.memcpy(get_data_ptr<int64_t>(indices),
                         get_data_ptr<const int64_t>(idx), numel * sizeof(int64_t)).wait();
            Tensor vals_bf16 = vals_f32.to(DType::BFloat16);
            queue.memcpy(values.data_ptr(), vals_bf16.data_ptr(),
                         numel * dtype_size(DType::BFloat16)).wait();
            return {values, indices};
        } else {
            throw std::runtime_error("sort: unsupported dtype");
        }

        return {values, indices};
    }
    // Fall through to host-side sort for non-contiguous sort dimension
#endif

    // Host-side sort fallback
    auto host_sort_impl = [&](auto dummy) {
        using T = decltype(dummy);
        std::vector<T> h_in(numel), h_out(numel);
        std::vector<int64_t> h_idx(numel);
        queue.memcpy(h_in.data(), input.data_ptr(), numel * sizeof(T)).wait();

        for (int64_t o = 0; o < outer_size; ++o) {
            for (int64_t i = 0; i < inner_size; ++i) {
                std::vector<std::pair<T, int64_t>> pairs(dim_size);
                for (int64_t d = 0; d < dim_size; ++d) {
                    int64_t idx = o * dim_size * inner_size + d * inner_size + i;
                    pairs[d] = {h_in[idx], d};
                }
                if (descending) {
                    std::sort(pairs.begin(), pairs.end(), [](auto& a, auto& b) { return a.first > b.first; });
                } else {
                    std::sort(pairs.begin(), pairs.end(), [](auto& a, auto& b) { return a.first < b.first; });
                }
                for (int64_t d = 0; d < dim_size; ++d) {
                    int64_t idx = o * dim_size * inner_size + d * inner_size + i;
                    h_out[idx] = pairs[d].first;
                    h_idx[idx] = pairs[d].second;
                }
            }
        }
        queue.memcpy(const_cast<void*>(values.data_ptr()), h_out.data(), numel * sizeof(T)).wait();
        queue.memcpy(const_cast<void*>(indices.data_ptr()), h_idx.data(), numel * sizeof(int64_t)).wait();
    };

    if (input.dtype() == DType::Float32) {
        host_sort_impl(float{});
    } else if (input.dtype() == DType::Float64) {
        host_sort_impl(double{});
    } else {
        throw std::runtime_error("sort: unsupported dtype");
    }

    return {values, indices};
}

// ============================================================================
// TopK kernel - find top K values and indices along a dimension
// ============================================================================
auto topk_kernel(const Tensor& input, int64_t k, int64_t dim, bool largest, bool sorted,
                 sycl::queue& queue) -> std::pair<Tensor, Tensor> {
    auto shape_span = input.shape();
    std::vector<int64_t> shape(shape_span.begin(), shape_span.end());
    int64_t ndim = shape.size();
    if (dim < 0) dim += ndim;

    std::vector<int64_t> out_shape = shape;
    out_shape[dim] = k;

    Tensor values(out_shape, input.dtype(), input.device());
    Tensor indices(out_shape, DType::Int64, input.device());

    int64_t outer_size = 1, inner_size = 1;
    for (int64_t i = 0; i < dim; ++i) outer_size *= shape[i];
    for (int64_t i = dim + 1; i < ndim; ++i) inner_size *= shape[i];
    int64_t dim_size = shape[dim];

#ifdef TENZOR_HAS_ONEDPL
    // Device-side topk for contiguous sort dimension (inner_size == 1)
    // Strategy: sort each slice on device via oneDPL, then copy first k elements
    if (inner_size == 1) {
        auto policy = oneapi::dpl::execution::make_device_policy(queue);
        int64_t in_numel = input.numel();
        int64_t out_numel = outer_size * k;

        auto device_topk_impl = [&](auto* out_val_ptr, const auto* in_ptr) {
            using T = std::remove_const_t<std::remove_pointer_t<decltype(in_ptr)>>;
            int64_t* out_idx_ptr = get_data_ptr<int64_t>(indices);

            // Allocate temporary device buffers for full sort
            T* tmp_vals = sycl::malloc_device<T>(in_numel, queue);
            int64_t* tmp_idx = sycl::malloc_device<int64_t>(in_numel, queue);

            // Copy input to temp values
            queue.memcpy(tmp_vals, in_ptr, in_numel * sizeof(T)).wait();

            // Initialize indices: each slice gets 0..dim_size-1
            queue.parallel_for(sycl::range<1>(in_numel), [=](sycl::id<1> gid) {
                tmp_idx[gid] = static_cast<int64_t>(gid[0]) % dim_size;
            }).wait();

            // Sort each slice, then copy top-k
            for (int64_t o = 0; o < outer_size; ++o) {
                T* slice_vals = tmp_vals + o * dim_size;
                int64_t* slice_idx = tmp_idx + o * dim_size;
                if (largest) {
                    oneapi::dpl::sort_by_key(policy, slice_vals, slice_vals + dim_size,
                                              slice_idx, std::greater<T>());
                } else {
                    oneapi::dpl::sort_by_key(policy, slice_vals, slice_vals + dim_size,
                                              slice_idx);
                }
                // Copy first k elements to output
                queue.memcpy(out_val_ptr + o * k, slice_vals, k * sizeof(T)).wait();
                queue.memcpy(out_idx_ptr + o * k, slice_idx, k * sizeof(int64_t)).wait();
            }

            sycl::free(tmp_vals, queue);
            sycl::free(tmp_idx, queue);
        };

        if (input.dtype() == DType::Float32) {
            device_topk_impl(get_data_ptr<float>(values), get_data_ptr<const float>(input));
        } else if (input.dtype() == DType::Float64) {
            device_topk_impl(get_data_ptr<double>(values), get_data_ptr<const double>(input));
        } else if (input.dtype() == DType::Int32) {
            device_topk_impl(get_data_ptr<int32_t>(values), get_data_ptr<const int32_t>(input));
        } else if (input.dtype() == DType::Int64) {
            device_topk_impl(get_data_ptr<int64_t>(values), get_data_ptr<const int64_t>(input));
        } else if (input.dtype() == DType::Float16) {
            Tensor input_f32 = input.to(DType::Float32);
            auto [vals_f32, idx] = topk_kernel(input_f32, k, dim, largest, sorted, queue);
            queue.memcpy(get_data_ptr<int64_t>(indices),
                         get_data_ptr<const int64_t>(idx), out_numel * sizeof(int64_t)).wait();
            Tensor vals_f16 = vals_f32.to(DType::Float16);
            queue.memcpy(values.data_ptr(), vals_f16.data_ptr(),
                         out_numel * dtype_size(DType::Float16)).wait();
            return {values, indices};
        } else if (input.dtype() == DType::BFloat16) {
            Tensor input_f32 = input.to(DType::Float32);
            auto [vals_f32, idx] = topk_kernel(input_f32, k, dim, largest, sorted, queue);
            queue.memcpy(get_data_ptr<int64_t>(indices),
                         get_data_ptr<const int64_t>(idx), out_numel * sizeof(int64_t)).wait();
            Tensor vals_bf16 = vals_f32.to(DType::BFloat16);
            queue.memcpy(values.data_ptr(), vals_bf16.data_ptr(),
                         out_numel * dtype_size(DType::BFloat16)).wait();
            return {values, indices};
        } else {
            throw std::runtime_error("topk: unsupported dtype");
        }

        return {values, indices};
    }
    // Fall through to host-side topk for non-contiguous sort dimension
#endif

    // Host-side topk fallback
    auto host_topk_impl = [&](auto dummy) {
        using T = decltype(dummy);
        int64_t in_numel = input.numel();
        int64_t out_numel = outer_size * k * inner_size;
        std::vector<T> h_in(in_numel), h_out(out_numel);
        std::vector<int64_t> h_idx(out_numel);
        queue.memcpy(h_in.data(), input.data_ptr(), in_numel * sizeof(T)).wait();

        for (int64_t o = 0; o < outer_size; ++o) {
            for (int64_t i = 0; i < inner_size; ++i) {
                std::vector<std::pair<T, int64_t>> pairs(dim_size);
                for (int64_t d = 0; d < dim_size; ++d) {
                    int64_t idx = o * dim_size * inner_size + d * inner_size + i;
                    pairs[d] = {h_in[idx], d};
                }
                if (largest) {
                    std::partial_sort(pairs.begin(), pairs.begin() + k, pairs.end(),
                                     [](auto& a, auto& b) { return a.first > b.first; });
                } else {
                    std::partial_sort(pairs.begin(), pairs.begin() + k, pairs.end(),
                                     [](auto& a, auto& b) { return a.first < b.first; });
                }
                if (sorted && largest) {
                    std::sort(pairs.begin(), pairs.begin() + k,
                              [](auto& a, auto& b) { return a.first > b.first; });
                } else if (sorted) {
                    std::sort(pairs.begin(), pairs.begin() + k,
                              [](auto& a, auto& b) { return a.first < b.first; });
                }
                for (int64_t d = 0; d < k; ++d) {
                    int64_t out_idx = o * k * inner_size + d * inner_size + i;
                    h_out[out_idx] = pairs[d].first;
                    h_idx[out_idx] = pairs[d].second;
                }
            }
        }
        queue.memcpy(const_cast<void*>(values.data_ptr()), h_out.data(), out_numel * sizeof(T)).wait();
        queue.memcpy(const_cast<void*>(indices.data_ptr()), h_idx.data(), out_numel * sizeof(int64_t)).wait();
    };

    if (input.dtype() == DType::Float32) {
        host_topk_impl(float{});
    } else if (input.dtype() == DType::Float64) {
        host_topk_impl(double{});
    } else {
        throw std::runtime_error("topk: unsupported dtype");
    }

    return {values, indices};
}

// ============================================================================
// Unique kernel - find unique values
// ============================================================================
auto unique_kernel(const Tensor& input, bool sorted, bool return_inverse, bool return_counts,
                   sycl::queue& queue) -> std::tuple<Tensor, Tensor, Tensor> {
    int64_t numel = input.numel();

#ifdef TENZOR_HAS_ONEDPL
    // Device-side unique using oneDPL: sort + unique on device
    // Unique operates on flattened 1D input, so always contiguous
    if (sorted && numel > 0) {
        auto policy = oneapi::dpl::execution::make_device_policy(queue);

        auto device_unique_impl = [&](const auto* in_ptr) {
            using T = std::remove_const_t<std::remove_pointer_t<decltype(in_ptr)>>;

            // Allocate device buffer for sorted copy
            T* d_sorted = sycl::malloc_device<T>(numel, queue);
            queue.memcpy(d_sorted, in_ptr, numel * sizeof(T)).wait();

            // Sort on device
            oneapi::dpl::sort(policy, d_sorted, d_sorted + numel);

            // Find unique elements using oneDPL unique
            auto new_end = oneapi::dpl::unique(policy, d_sorted, d_sorted + numel);
            int64_t n_unique = std::distance(d_sorted, new_end);

            // Create output tensor and copy unique values
            Tensor out_vals({n_unique}, input.dtype(), input.device());
            queue.memcpy(out_vals.data_ptr(), d_sorted, n_unique * sizeof(T)).wait();

            // Compute inverse mapping if needed
            Tensor out_inverse({numel}, DType::Int64, input.device());
            if (return_inverse) {
                int64_t* inv_ptr = get_data_ptr<int64_t>(out_inverse);
                const T* unique_ptr = get_data_ptr<const T>(out_vals);
                // For each input element, binary search in sorted unique values
                queue.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> gid) {
                    T val = in_ptr[gid];
                    // Binary search in unique_ptr[0..n_unique)
                    int64_t lo = 0, hi = n_unique;
                    while (lo < hi) {
                        int64_t mid = lo + (hi - lo) / 2;
                        if (unique_ptr[mid] < val) {
                            lo = mid + 1;
                        } else {
                            hi = mid;
                        }
                    }
                    inv_ptr[gid] = lo;
                }).wait();
            }

            // Compute counts if needed
            Tensor out_counts({return_counts ? n_unique : 0}, DType::Int64, input.device());
            if (return_counts && n_unique > 0) {
                int64_t* cnt_ptr = get_data_ptr<int64_t>(out_counts);
                const T* unique_ptr = get_data_ptr<const T>(out_vals);
                // Initialize counts to 0
                queue.memset(cnt_ptr, 0, n_unique * sizeof(int64_t)).wait();
                // Count occurrences: for each input element, find its unique index and increment
                // Use host-side counting since atomic int64 add isn't universally supported
                std::vector<int64_t> h_counts(n_unique, 0);
                std::vector<int64_t> h_inverse(numel);
                if (return_inverse) {
                    queue.memcpy(h_inverse.data(), get_data_ptr<int64_t>(out_inverse),
                                 numel * sizeof(int64_t)).wait();
                } else {
                    // Need to compute inverse for counting
                    std::vector<T> h_unique(n_unique), h_input(numel);
                    queue.memcpy(h_unique.data(), get_data_ptr<const T>(out_vals),
                                 n_unique * sizeof(T)).wait();
                    queue.memcpy(h_input.data(), in_ptr, numel * sizeof(T)).wait();
                    for (int64_t i = 0; i < numel; ++i) {
                        auto it = std::lower_bound(h_unique.begin(), h_unique.end(), h_input[i]);
                        h_inverse[i] = std::distance(h_unique.begin(), it);
                    }
                }
                for (int64_t i = 0; i < numel; ++i) {
                    h_counts[h_inverse[i]]++;
                }
                queue.memcpy(cnt_ptr, h_counts.data(), n_unique * sizeof(int64_t)).wait();
            }

            sycl::free(d_sorted, queue);
            return std::make_tuple(out_vals, out_inverse, out_counts);
        };

        if (input.dtype() == DType::Float32) {
            return device_unique_impl(get_data_ptr<const float>(input));
        } else if (input.dtype() == DType::Int64) {
            return device_unique_impl(get_data_ptr<const int64_t>(input));
        } else if (input.dtype() == DType::Float64) {
            return device_unique_impl(get_data_ptr<const double>(input));
        } else if (input.dtype() == DType::Int32) {
            return device_unique_impl(get_data_ptr<const int32_t>(input));
        }
        // Unsupported dtypes fall through to host path
    }
    // Fall through to host-side unique for unsorted or unsupported dtypes
#endif

    // Host-side unique fallback
    auto host_unique_impl = [&](auto dummy) {
        using T = decltype(dummy);
        std::vector<T> h_in(numel);
        queue.memcpy(h_in.data(), input.data_ptr(), numel * sizeof(T)).wait();

        std::vector<T> unique_vals;
        std::vector<int64_t> inverse(numel, 0);
        std::vector<int64_t> counts;

        if (sorted) {
            unique_vals = h_in;
            std::sort(unique_vals.begin(), unique_vals.end());
            unique_vals.erase(std::unique(unique_vals.begin(), unique_vals.end()), unique_vals.end());
        } else {
            for (auto v : h_in) {
                if (std::find(unique_vals.begin(), unique_vals.end(), v) == unique_vals.end()) {
                    unique_vals.push_back(v);
                }
            }
        }

        if (return_inverse) {
            for (int64_t i = 0; i < numel; ++i) {
                for (size_t j = 0; j < unique_vals.size(); ++j) {
                    if (h_in[i] == unique_vals[j]) { inverse[i] = j; break; }
                }
            }
        }

        if (return_counts) {
            counts.resize(unique_vals.size(), 0);
            for (int64_t i = 0; i < numel; ++i) {
                for (size_t j = 0; j < unique_vals.size(); ++j) {
                    if (h_in[i] == unique_vals[j]) { counts[j]++; break; }
                }
            }
        }

        int64_t n_unique = unique_vals.size();
        Tensor out_vals({n_unique}, input.dtype(), input.device());
        queue.memcpy(const_cast<void*>(out_vals.data_ptr()), unique_vals.data(), n_unique * sizeof(T)).wait();

        Tensor out_inverse({numel}, DType::Int64, input.device());
        queue.memcpy(const_cast<void*>(out_inverse.data_ptr()), inverse.data(), numel * sizeof(int64_t)).wait();

        Tensor out_counts({return_counts ? n_unique : 0}, DType::Int64, input.device());
        if (return_counts && n_unique > 0) {
            queue.memcpy(const_cast<void*>(out_counts.data_ptr()), counts.data(), n_unique * sizeof(int64_t)).wait();
        }

        return std::make_tuple(out_vals, out_inverse, out_counts);
    };

    if (input.dtype() == DType::Float32) {
        return host_unique_impl(float{});
    } else if (input.dtype() == DType::Int64) {
        return host_unique_impl(int64_t{});
    } else {
        throw std::runtime_error("unique: unsupported dtype (only Float32 and Int64 supported)");
    }
}

// ============================================================================
// Any reduction kernel - check if any element is non-zero along a dimension
// ============================================================================

// Kernel name structs for any_kernel
class AnyKernelFloat32;
class AnyKernelFloat64;
class AnyKernelFloat16;
class AnyKernelBFloat16;
class AnyKernelInt32;
class AnyKernelInt64;
class AnyKernelBool;
class AnyFullKernel;

auto any_kernel(const Tensor& input, int64_t dim, bool keepdim, sycl::queue& queue) -> Tensor {
    Tensor in_cont = input.is_contiguous() ? input : contiguous_kernel(input, queue);

    auto shape = in_cont.shape();
    std::vector<int64_t> shape_vec(shape.begin(), shape.end());
    const int64_t ndim = static_cast<int64_t>(shape.size());

    bool is_full_reduction = (dim == INT64_MIN || (dim == -1 && ndim == 0));

    if (!is_full_reduction) {
        if (dim < 0) dim += ndim;
        if (dim < 0 || dim >= ndim) {
            throw std::runtime_error("Dimension " + std::to_string(dim) +
                " out of range for tensor with " + std::to_string(ndim) + " dimensions");
        }
    }

    auto out_shape = compute_reduction_shape(shape_vec, is_full_reduction ? -1 : dim, keepdim);
    Tensor output(out_shape, DType::Bool, in_cont.device());

    if (is_full_reduction) {
        const int64_t total_size = in_cont.numel();
        bool* out_ptr = get_data_ptr<bool>(output);

        if (total_size == 0) {
            queue.single_task([=]() { out_ptr[0] = false; });
            return output;
        }

        // Use int32 flag for reduction (1 = found non-zero)
        auto flag_buf = sycl::malloc_shared<int32_t>(1, queue);
        flag_buf[0] = 0;

        if (in_cont.dtype() == DType::Float32) {
            const float* in_ptr = get_data_ptr<const float>(in_cont);
            queue.parallel_for(sycl::range<1>(total_size), sycl::reduction(flag_buf, sycl::maximum<int32_t>()),
                [=](sycl::id<1> idx, auto& flag) {
                    if (in_ptr[idx] != 0.0f) flag.combine(1);
                });
        } else if (in_cont.dtype() == DType::Float64) {
            const double* in_ptr = get_data_ptr<const double>(in_cont);
            queue.parallel_for(sycl::range<1>(total_size), sycl::reduction(flag_buf, sycl::maximum<int32_t>()),
                [=](sycl::id<1> idx, auto& flag) {
                    if (in_ptr[idx] != 0.0) flag.combine(1);
                });
        } else if (in_cont.dtype() == DType::Float16) {
            const sycl::half* in_ptr = get_data_ptr<const sycl::half>(in_cont);
            queue.parallel_for(sycl::range<1>(total_size), sycl::reduction(flag_buf, sycl::maximum<int32_t>()),
                [=](sycl::id<1> idx, auto& flag) {
                    if (static_cast<float>(in_ptr[idx]) != 0.0f) flag.combine(1);
                });
        } else if (in_cont.dtype() == DType::BFloat16) {
            const uint16_t* in_ptr = get_data_ptr<const uint16_t>(in_cont);
            queue.parallel_for(sycl::range<1>(total_size), sycl::reduction(flag_buf, sycl::maximum<int32_t>()),
                [=](sycl::id<1> idx, auto& flag) {
                    if (bf16_to_f32(in_ptr[idx]) != 0.0f) flag.combine(1);
                });
        } else if (in_cont.dtype() == DType::Int32) {
            const int32_t* in_ptr = get_data_ptr<const int32_t>(in_cont);
            queue.parallel_for(sycl::range<1>(total_size), sycl::reduction(flag_buf, sycl::maximum<int32_t>()),
                [=](sycl::id<1> idx, auto& flag) {
                    if (in_ptr[idx] != 0) flag.combine(1);
                });
        } else if (in_cont.dtype() == DType::Int64) {
            const int64_t* in_ptr = get_data_ptr<const int64_t>(in_cont);
            queue.parallel_for(sycl::range<1>(total_size), sycl::reduction(flag_buf, sycl::maximum<int32_t>()),
                [=](sycl::id<1> idx, auto& flag) {
                    if (in_ptr[idx] != 0) flag.combine(1);
                });
        } else if (in_cont.dtype() == DType::Bool) {
            const bool* in_ptr = get_data_ptr<const bool>(in_cont);
            queue.parallel_for(sycl::range<1>(total_size), sycl::reduction(flag_buf, sycl::maximum<int32_t>()),
                [=](sycl::id<1> idx, auto& flag) {
                    if (in_ptr[idx]) flag.combine(1);
                });
        } else {
            sycl::free(flag_buf, queue);
            throw std::runtime_error("Unsupported dtype for any reduction");
        }

        bool result = (flag_buf[0] != 0);
        queue.memcpy(out_ptr, &result, sizeof(bool)).wait();
        sycl::free(flag_buf, queue);
    } else {
        // Partial reduction along a specific dimension
        const int64_t outer_size = std::accumulate(shape.begin(), shape.begin() + dim, 1LL, std::multiplies<>());
        const int64_t dim_size = shape[dim];
        const int64_t inner_size = std::accumulate(shape.begin() + dim + 1, shape.end(), 1LL, std::multiplies<>());
        bool* out_ptr = get_data_ptr<bool>(output);

        if (in_cont.dtype() == DType::Float32) {
            const float* in_ptr = get_data_ptr<const float>(in_cont);
            queue.parallel_for<AnyKernelFloat32>(sycl::range<2>(outer_size, inner_size), [=](sycl::id<2> idx) {
                const int64_t outer_idx = idx[0];
                const int64_t inner_idx = idx[1];
                const int64_t base_offset = outer_idx * dim_size * inner_size + inner_idx;
                bool found = false;
                for (int64_t d = 0; d < dim_size; ++d) {
                    if (in_ptr[base_offset + d * inner_size] != 0.0f) { found = true; break; }
                }
                out_ptr[outer_idx * inner_size + inner_idx] = found;
            });
        } else if (in_cont.dtype() == DType::Float64) {
            const double* in_ptr = get_data_ptr<const double>(in_cont);
            queue.parallel_for<AnyKernelFloat64>(sycl::range<2>(outer_size, inner_size), [=](sycl::id<2> idx) {
                const int64_t outer_idx = idx[0];
                const int64_t inner_idx = idx[1];
                const int64_t base_offset = outer_idx * dim_size * inner_size + inner_idx;
                bool found = false;
                for (int64_t d = 0; d < dim_size; ++d) {
                    if (in_ptr[base_offset + d * inner_size] != 0.0) { found = true; break; }
                }
                out_ptr[outer_idx * inner_size + inner_idx] = found;
            });
        } else if (in_cont.dtype() == DType::Float16) {
            const sycl::half* in_ptr = get_data_ptr<const sycl::half>(in_cont);
            queue.parallel_for<AnyKernelFloat16>(sycl::range<2>(outer_size, inner_size), [=](sycl::id<2> idx) {
                const int64_t outer_idx = idx[0];
                const int64_t inner_idx = idx[1];
                const int64_t base_offset = outer_idx * dim_size * inner_size + inner_idx;
                bool found = false;
                for (int64_t d = 0; d < dim_size; ++d) {
                    if (static_cast<float>(in_ptr[base_offset + d * inner_size]) != 0.0f) { found = true; break; }
                }
                out_ptr[outer_idx * inner_size + inner_idx] = found;
            });
        } else if (in_cont.dtype() == DType::BFloat16) {
            const uint16_t* in_ptr = get_data_ptr<const uint16_t>(in_cont);
            queue.parallel_for<AnyKernelBFloat16>(sycl::range<2>(outer_size, inner_size), [=](sycl::id<2> idx) {
                const int64_t outer_idx = idx[0];
                const int64_t inner_idx = idx[1];
                const int64_t base_offset = outer_idx * dim_size * inner_size + inner_idx;
                bool found = false;
                for (int64_t d = 0; d < dim_size; ++d) {
                    if (bf16_to_f32(in_ptr[base_offset + d * inner_size]) != 0.0f) { found = true; break; }
                }
                out_ptr[outer_idx * inner_size + inner_idx] = found;
            });
        } else if (in_cont.dtype() == DType::Int32) {
            const int32_t* in_ptr = get_data_ptr<const int32_t>(in_cont);
            queue.parallel_for<AnyKernelInt32>(sycl::range<2>(outer_size, inner_size), [=](sycl::id<2> idx) {
                const int64_t outer_idx = idx[0];
                const int64_t inner_idx = idx[1];
                const int64_t base_offset = outer_idx * dim_size * inner_size + inner_idx;
                bool found = false;
                for (int64_t d = 0; d < dim_size; ++d) {
                    if (in_ptr[base_offset + d * inner_size] != 0) { found = true; break; }
                }
                out_ptr[outer_idx * inner_size + inner_idx] = found;
            });
        } else if (in_cont.dtype() == DType::Int64) {
            const int64_t* in_ptr = get_data_ptr<const int64_t>(in_cont);
            queue.parallel_for<AnyKernelInt64>(sycl::range<2>(outer_size, inner_size), [=](sycl::id<2> idx) {
                const int64_t outer_idx = idx[0];
                const int64_t inner_idx = idx[1];
                const int64_t base_offset = outer_idx * dim_size * inner_size + inner_idx;
                bool found = false;
                for (int64_t d = 0; d < dim_size; ++d) {
                    if (in_ptr[base_offset + d * inner_size] != 0) { found = true; break; }
                }
                out_ptr[outer_idx * inner_size + inner_idx] = found;
            });
        } else if (in_cont.dtype() == DType::Bool) {
            const bool* in_ptr = get_data_ptr<const bool>(in_cont);
            queue.parallel_for<AnyKernelBool>(sycl::range<2>(outer_size, inner_size), [=](sycl::id<2> idx) {
                const int64_t outer_idx = idx[0];
                const int64_t inner_idx = idx[1];
                const int64_t base_offset = outer_idx * dim_size * inner_size + inner_idx;
                bool found = false;
                for (int64_t d = 0; d < dim_size; ++d) {
                    if (in_ptr[base_offset + d * inner_size]) { found = true; break; }
                }
                out_ptr[outer_idx * inner_size + inner_idx] = found;
            });
        } else {
            throw std::runtime_error("Unsupported dtype for any reduction");
        }
    }

    return output;
}

// ============================================================================
// All reduction kernel - check if all elements are non-zero along a dimension
// ============================================================================

// Kernel name structs for all_kernel
class AllKernelFloat32;
class AllKernelFloat64;
class AllKernelFloat16;
class AllKernelBFloat16;
class AllKernelInt32;
class AllKernelInt64;
class AllKernelBool;

auto all_kernel(const Tensor& input, int64_t dim, bool keepdim, sycl::queue& queue) -> Tensor {
    Tensor in_cont = input.is_contiguous() ? input : contiguous_kernel(input, queue);

    auto shape = in_cont.shape();
    std::vector<int64_t> shape_vec(shape.begin(), shape.end());
    const int64_t ndim = static_cast<int64_t>(shape.size());

    bool is_full_reduction = (dim == INT64_MIN || (dim == -1 && ndim == 0));

    if (!is_full_reduction) {
        if (dim < 0) dim += ndim;
        if (dim < 0 || dim >= ndim) {
            throw std::runtime_error("Dimension " + std::to_string(dim) +
                " out of range for tensor with " + std::to_string(ndim) + " dimensions");
        }
    }

    auto out_shape = compute_reduction_shape(shape_vec, is_full_reduction ? -1 : dim, keepdim);
    Tensor output(out_shape, DType::Bool, in_cont.device());

    if (is_full_reduction) {
        const int64_t total_size = in_cont.numel();
        bool* out_ptr = get_data_ptr<bool>(output);

        if (total_size == 0) {
            // all() of empty set is true (vacuous truth)
            queue.single_task([=]() { out_ptr[0] = true; });
            return output;
        }

        // Use int32 flag for reduction: minimum. Start at 1, set to 0 if any zero found.
        auto flag_buf = sycl::malloc_shared<int32_t>(1, queue);
        flag_buf[0] = 1;

        if (in_cont.dtype() == DType::Float32) {
            const float* in_ptr = get_data_ptr<const float>(in_cont);
            queue.parallel_for(sycl::range<1>(total_size), sycl::reduction(flag_buf, sycl::minimum<int32_t>()),
                [=](sycl::id<1> idx, auto& flag) {
                    if (in_ptr[idx] == 0.0f) flag.combine(0);
                });
        } else if (in_cont.dtype() == DType::Float64) {
            const double* in_ptr = get_data_ptr<const double>(in_cont);
            queue.parallel_for(sycl::range<1>(total_size), sycl::reduction(flag_buf, sycl::minimum<int32_t>()),
                [=](sycl::id<1> idx, auto& flag) {
                    if (in_ptr[idx] == 0.0) flag.combine(0);
                });
        } else if (in_cont.dtype() == DType::Float16) {
            const sycl::half* in_ptr = get_data_ptr<const sycl::half>(in_cont);
            queue.parallel_for(sycl::range<1>(total_size), sycl::reduction(flag_buf, sycl::minimum<int32_t>()),
                [=](sycl::id<1> idx, auto& flag) {
                    if (static_cast<float>(in_ptr[idx]) == 0.0f) flag.combine(0);
                });
        } else if (in_cont.dtype() == DType::BFloat16) {
            const uint16_t* in_ptr = get_data_ptr<const uint16_t>(in_cont);
            queue.parallel_for(sycl::range<1>(total_size), sycl::reduction(flag_buf, sycl::minimum<int32_t>()),
                [=](sycl::id<1> idx, auto& flag) {
                    if (bf16_to_f32(in_ptr[idx]) == 0.0f) flag.combine(0);
                });
        } else if (in_cont.dtype() == DType::Int32) {
            const int32_t* in_ptr = get_data_ptr<const int32_t>(in_cont);
            queue.parallel_for(sycl::range<1>(total_size), sycl::reduction(flag_buf, sycl::minimum<int32_t>()),
                [=](sycl::id<1> idx, auto& flag) {
                    if (in_ptr[idx] == 0) flag.combine(0);
                });
        } else if (in_cont.dtype() == DType::Int64) {
            const int64_t* in_ptr = get_data_ptr<const int64_t>(in_cont);
            queue.parallel_for(sycl::range<1>(total_size), sycl::reduction(flag_buf, sycl::minimum<int32_t>()),
                [=](sycl::id<1> idx, auto& flag) {
                    if (in_ptr[idx] == 0) flag.combine(0);
                });
        } else if (in_cont.dtype() == DType::Bool) {
            const bool* in_ptr = get_data_ptr<const bool>(in_cont);
            queue.parallel_for(sycl::range<1>(total_size), sycl::reduction(flag_buf, sycl::minimum<int32_t>()),
                [=](sycl::id<1> idx, auto& flag) {
                    if (!in_ptr[idx]) flag.combine(0);
                });
        } else {
            sycl::free(flag_buf, queue);
            throw std::runtime_error("Unsupported dtype for all reduction");
        }

        bool result = (flag_buf[0] != 0);
        queue.memcpy(out_ptr, &result, sizeof(bool)).wait();
        sycl::free(flag_buf, queue);
    } else {
        // Partial reduction along a specific dimension
        const int64_t outer_size = std::accumulate(shape.begin(), shape.begin() + dim, 1LL, std::multiplies<>());
        const int64_t dim_size = shape[dim];
        const int64_t inner_size = std::accumulate(shape.begin() + dim + 1, shape.end(), 1LL, std::multiplies<>());
        bool* out_ptr = get_data_ptr<bool>(output);

        if (in_cont.dtype() == DType::Float32) {
            const float* in_ptr = get_data_ptr<const float>(in_cont);
            queue.parallel_for<AllKernelFloat32>(sycl::range<2>(outer_size, inner_size), [=](sycl::id<2> idx) {
                const int64_t outer_idx = idx[0];
                const int64_t inner_idx = idx[1];
                const int64_t base_offset = outer_idx * dim_size * inner_size + inner_idx;
                bool all_nonzero = true;
                for (int64_t d = 0; d < dim_size; ++d) {
                    if (in_ptr[base_offset + d * inner_size] == 0.0f) { all_nonzero = false; break; }
                }
                out_ptr[outer_idx * inner_size + inner_idx] = all_nonzero;
            });
        } else if (in_cont.dtype() == DType::Float64) {
            const double* in_ptr = get_data_ptr<const double>(in_cont);
            queue.parallel_for<AllKernelFloat64>(sycl::range<2>(outer_size, inner_size), [=](sycl::id<2> idx) {
                const int64_t outer_idx = idx[0];
                const int64_t inner_idx = idx[1];
                const int64_t base_offset = outer_idx * dim_size * inner_size + inner_idx;
                bool all_nonzero = true;
                for (int64_t d = 0; d < dim_size; ++d) {
                    if (in_ptr[base_offset + d * inner_size] == 0.0) { all_nonzero = false; break; }
                }
                out_ptr[outer_idx * inner_size + inner_idx] = all_nonzero;
            });
        } else if (in_cont.dtype() == DType::Float16) {
            const sycl::half* in_ptr = get_data_ptr<const sycl::half>(in_cont);
            queue.parallel_for<AllKernelFloat16>(sycl::range<2>(outer_size, inner_size), [=](sycl::id<2> idx) {
                const int64_t outer_idx = idx[0];
                const int64_t inner_idx = idx[1];
                const int64_t base_offset = outer_idx * dim_size * inner_size + inner_idx;
                bool all_nonzero = true;
                for (int64_t d = 0; d < dim_size; ++d) {
                    if (static_cast<float>(in_ptr[base_offset + d * inner_size]) == 0.0f) { all_nonzero = false; break; }
                }
                out_ptr[outer_idx * inner_size + inner_idx] = all_nonzero;
            });
        } else if (in_cont.dtype() == DType::BFloat16) {
            const uint16_t* in_ptr = get_data_ptr<const uint16_t>(in_cont);
            queue.parallel_for<AllKernelBFloat16>(sycl::range<2>(outer_size, inner_size), [=](sycl::id<2> idx) {
                const int64_t outer_idx = idx[0];
                const int64_t inner_idx = idx[1];
                const int64_t base_offset = outer_idx * dim_size * inner_size + inner_idx;
                bool all_nonzero = true;
                for (int64_t d = 0; d < dim_size; ++d) {
                    if (bf16_to_f32(in_ptr[base_offset + d * inner_size]) == 0.0f) { all_nonzero = false; break; }
                }
                out_ptr[outer_idx * inner_size + inner_idx] = all_nonzero;
            });
        } else if (in_cont.dtype() == DType::Int32) {
            const int32_t* in_ptr = get_data_ptr<const int32_t>(in_cont);
            queue.parallel_for<AllKernelInt32>(sycl::range<2>(outer_size, inner_size), [=](sycl::id<2> idx) {
                const int64_t outer_idx = idx[0];
                const int64_t inner_idx = idx[1];
                const int64_t base_offset = outer_idx * dim_size * inner_size + inner_idx;
                bool all_nonzero = true;
                for (int64_t d = 0; d < dim_size; ++d) {
                    if (in_ptr[base_offset + d * inner_size] == 0) { all_nonzero = false; break; }
                }
                out_ptr[outer_idx * inner_size + inner_idx] = all_nonzero;
            });
        } else if (in_cont.dtype() == DType::Int64) {
            const int64_t* in_ptr = get_data_ptr<const int64_t>(in_cont);
            queue.parallel_for<AllKernelInt64>(sycl::range<2>(outer_size, inner_size), [=](sycl::id<2> idx) {
                const int64_t outer_idx = idx[0];
                const int64_t inner_idx = idx[1];
                const int64_t base_offset = outer_idx * dim_size * inner_size + inner_idx;
                bool all_nonzero = true;
                for (int64_t d = 0; d < dim_size; ++d) {
                    if (in_ptr[base_offset + d * inner_size] == 0) { all_nonzero = false; break; }
                }
                out_ptr[outer_idx * inner_size + inner_idx] = all_nonzero;
            });
        } else if (in_cont.dtype() == DType::Bool) {
            const bool* in_ptr = get_data_ptr<const bool>(in_cont);
            queue.parallel_for<AllKernelBool>(sycl::range<2>(outer_size, inner_size), [=](sycl::id<2> idx) {
                const int64_t outer_idx = idx[0];
                const int64_t inner_idx = idx[1];
                const int64_t base_offset = outer_idx * dim_size * inner_size + inner_idx;
                bool all_nonzero = true;
                for (int64_t d = 0; d < dim_size; ++d) {
                    if (!in_ptr[base_offset + d * inner_size]) { all_nonzero = false; break; }
                }
                out_ptr[outer_idx * inner_size + inner_idx] = all_nonzero;
            });
        } else {
            throw std::runtime_error("Unsupported dtype for all reduction");
        }
    }

    return output;
}

// ============================================================================
// LogSumExp - Numerically stable log(sum(exp(x)))
// ============================================================================

struct LogSumExpKernelFloat32Max {};
struct LogSumExpKernelFloat32SumExp {};
struct LogSumExpKernelFloat64Max {};
struct LogSumExpKernelFloat64SumExp {};

auto logsumexp_kernel(const Tensor& input, int64_t dim, bool keepdim, sycl::queue& queue) -> Tensor {
    // Ensure input is contiguous for direct memory access
    Tensor in_cont = input.is_contiguous() ? input : contiguous_kernel(input, queue);

    auto shape = in_cont.shape();
    std::vector<int64_t> shape_vec(shape.begin(), shape.end());
    int64_t ndim = static_cast<int64_t>(shape.size());
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) {
        throw std::runtime_error("Dimension " + std::to_string(dim) +
            " out of range for tensor with " + std::to_string(ndim) + " dimensions");
    }

    int64_t dim_size = shape[dim];
    int64_t outer = 1, inner = 1;
    for (int64_t i = 0; i < dim; ++i) outer *= shape[i];
    for (int64_t i = dim + 1; i < ndim; ++i) inner *= shape[i];

    // Output shape with reduced dimension
    auto out_shape = compute_reduction_shape(shape_vec, dim, keepdim);
    int64_t out_numel = outer * inner;
    Tensor output(out_shape, in_cont.dtype(), in_cont.device());

    if (in_cont.dtype() == DType::Float32) {
        const float* in_ptr = get_data_ptr<const float>(in_cont);
        float* out_ptr = get_data_ptr<float>(output);

        // Step 1: Find max along dim
        Tensor max_buf({out_numel}, DType::Float32, in_cont.device());
        float* max_ptr = get_data_ptr<float>(max_buf);

        queue.parallel_for<LogSumExpKernelFloat32Max>(sycl::range<1>(out_numel), [=](sycl::id<1> idx) {
            int64_t o = idx[0] / inner;
            int64_t i = idx[0] % inner;
            float m = -std::numeric_limits<float>::infinity();
            for (int64_t d = 0; d < dim_size; ++d) {
                float val = in_ptr[(o * dim_size + d) * inner + i];
                m = sycl::fmax(m, val);
            }
            max_ptr[idx] = m;
        });

        // Step 2: sum(exp(x - max)) and compute log(...) + max
        queue.parallel_for<LogSumExpKernelFloat32SumExp>(sycl::range<1>(out_numel), [=](sycl::id<1> idx) {
            int64_t o = idx[0] / inner;
            int64_t i = idx[0] % inner;
            float m = max_ptr[idx];
            // Handle inf: if max is +/-inf, result should be the same inf
            if (sycl::isinf(m)) {
                out_ptr[idx] = m;
                return;
            }
            float sum = 0.0f;
            for (int64_t d = 0; d < dim_size; ++d) {
                sum += sycl::exp(in_ptr[(o * dim_size + d) * inner + i] - m);
            }
            out_ptr[idx] = sycl::log(sum) + m;
        });
    } else if (in_cont.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(in_cont);
        double* out_ptr = get_data_ptr<double>(output);

        Tensor max_buf({out_numel}, DType::Float64, in_cont.device());
        double* max_ptr = get_data_ptr<double>(max_buf);

        queue.parallel_for<LogSumExpKernelFloat64Max>(sycl::range<1>(out_numel), [=](sycl::id<1> idx) {
            int64_t o = idx[0] / inner;
            int64_t i = idx[0] % inner;
            double m = -std::numeric_limits<double>::infinity();
            for (int64_t d = 0; d < dim_size; ++d) {
                double val = in_ptr[(o * dim_size + d) * inner + i];
                m = sycl::fmax(m, val);
            }
            max_ptr[idx] = m;
        });

        queue.parallel_for<LogSumExpKernelFloat64SumExp>(sycl::range<1>(out_numel), [=](sycl::id<1> idx) {
            int64_t o = idx[0] / inner;
            int64_t i = idx[0] % inner;
            double m = max_ptr[idx];
            if (sycl::isinf(m)) {
                out_ptr[idx] = m;
                return;
            }
            double sum = 0.0;
            for (int64_t d = 0; d < dim_size; ++d) {
                sum += sycl::exp(in_ptr[(o * dim_size + d) * inner + i] - m);
            }
            out_ptr[idx] = sycl::log(sum) + m;
        });
    } else if (in_cont.dtype() == DType::Float16) {
        const sycl::half* in_ptr = get_data_ptr<const sycl::half>(in_cont);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);

        Tensor max_buf({out_numel}, DType::Float32, in_cont.device());
        float* max_ptr = get_data_ptr<float>(max_buf);

        queue.parallel_for(sycl::range<1>(out_numel), [=](sycl::id<1> idx) {
            int64_t o = idx[0] / inner;
            int64_t i = idx[0] % inner;
            float m = -std::numeric_limits<float>::infinity();
            for (int64_t d = 0; d < dim_size; ++d) {
                float val = static_cast<float>(in_ptr[(o * dim_size + d) * inner + i]);
                m = sycl::fmax(m, val);
            }
            max_ptr[idx] = m;
        });

        queue.parallel_for(sycl::range<1>(out_numel), [=](sycl::id<1> idx) {
            int64_t o = idx[0] / inner;
            int64_t i = idx[0] % inner;
            float m = max_ptr[idx];
            if (sycl::isinf(m)) {
                out_ptr[idx] = sycl::half(m);
                return;
            }
            float sum = 0.0f;
            for (int64_t d = 0; d < dim_size; ++d) {
                sum += sycl::exp(static_cast<float>(in_ptr[(o * dim_size + d) * inner + i]) - m);
            }
            out_ptr[idx] = sycl::half(sycl::log(sum) + m);
        });
    } else if (in_cont.dtype() == DType::BFloat16) {
        const uint16_t* in_ptr = get_data_ptr<const uint16_t>(in_cont);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);

        Tensor max_buf({out_numel}, DType::Float32, in_cont.device());
        float* max_ptr = get_data_ptr<float>(max_buf);

        queue.parallel_for(sycl::range<1>(out_numel), [=](sycl::id<1> idx) {
            int64_t o = idx[0] / inner;
            int64_t i = idx[0] % inner;
            float m = -std::numeric_limits<float>::infinity();
            for (int64_t d = 0; d < dim_size; ++d) {
                float val = bf16_to_f32(in_ptr[(o * dim_size + d) * inner + i]);
                m = sycl::fmax(m, val);
            }
            max_ptr[idx] = m;
        });

        queue.parallel_for(sycl::range<1>(out_numel), [=](sycl::id<1> idx) {
            int64_t o = idx[0] / inner;
            int64_t i = idx[0] % inner;
            float m = max_ptr[idx];
            if (sycl::isinf(m)) {
                out_ptr[idx] = f32_to_bf16(m);
                return;
            }
            float sum = 0.0f;
            for (int64_t d = 0; d < dim_size; ++d) {
                sum += sycl::exp(bf16_to_f32(in_ptr[(o * dim_size + d) * inner + i]) - m);
            }
            out_ptr[idx] = f32_to_bf16(sycl::log(sum) + m);
        });
    } else {
        throw std::runtime_error("logsumexp_kernel: unsupported dtype");
    }

    return output;
}

} // namespace oneapi
} // namespace tenzor
