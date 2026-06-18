#include "tenzor/core/tensor.hpp"
#include "oneapi_kernel_utils.hpp"
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

// Forward declarations
namespace tenzor {
namespace oneapi {
    auto contiguous_kernel(const Tensor& input, sycl::queue& queue) -> Tensor;
    auto cast_kernel(const Tensor& input, DType target_dtype, sycl::queue& queue) -> Tensor;
}
}

namespace tenzor {
namespace oneapi {

// NaN-propagating max/min for reductions. IEEE sycl::fmax/fmin return the
// non-NaN operand when one is NaN, which silently drops NaNs — but max()/min()
// must propagate NaN (PyTorch / CPU contract). These are sticky: once the
// accumulator is NaN it stays NaN. (oneAPI build does not use fast-math, so
// sycl::isnan is reliable here.)
template <typename T>
static inline T nan_fmax(T acc, T val) {
    return (sycl::isnan(val) || sycl::isnan(acc)) ? val + acc : sycl::fmax(acc, val);
}
template <typename T>
static inline T nan_fmin(T acc, T val) {
    return (sycl::isnan(val) || sycl::isnan(acc)) ? val + acc : sycl::fmin(acc, val);
}

// Kernel class declarations for SYCL
class SumKernelFloat32;
class SumKernelFloat64;
class SumKernelFloat16;
class SumKernelBFloat16;
class SumKernelInt32;
class SumKernelInt64;
class SumKernelUInt64;
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
class MaxKernelInt64;
class MinKernelFloat32;
class MinKernelFloat64;
class MinKernelFloat16;
class MinKernelBFloat16;
class MinKernelInt32;
class MinKernelInt64;
class ArgmaxKernelFloat32;
class ArgmaxKernelFloat64;
class ArgmaxKernelFloat16;
class ArgmaxKernelInt32;
class ArgminKernelFloat32;
class ArgminKernelFloat64;
class ArgminKernelFloat16;
class ArgminKernelInt32;

// Two-phase argmax/argmin full-reduction kernel names
class ArgmaxFullPhase1Float32;
class ArgmaxFullPhase1Float64;
class ArgmaxFullPhase1Float16;
class ArgmaxFullPhase1Int32;
class ArgmaxFullPhase2Float32;
class ArgmaxFullPhase2Float64;
class ArgmaxFullPhase2Float16;
class ArgmaxFullPhase2Int32;
class ArgminFullPhase1Float32;
class ArgminFullPhase1Float64;
class ArgminFullPhase1Float16;
class ArgminFullPhase1Int32;
class ArgminFullPhase2Float32;
class ArgminFullPhase2Float64;
class ArgminFullPhase2Float16;
class ArgminFullPhase2Int32;



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
            } else if (in_cont.dtype() == DType::Int64) {
                int64_t* out_ptr = get_data_ptr<int64_t>(output);
                queue.single_task([=]() { out_ptr[0] = 0; });
            } else if (in_cont.dtype() == DType::Bool) {
                output = Tensor({}, DType::Int64, output.device());
                int64_t* out_ptr = get_data_ptr<int64_t>(output);
                queue.single_task([=]() { out_ptr[0] = 0; });
            } else if (in_cont.dtype() == DType::Float16) {
                // Additive identity 0; IEEE-754 half +0.0 bit pattern. The caching
                // allocator does not zero memory, so write it explicitly.
                uint16_t zero_bits = 0x0000;
                queue.memcpy(get_data_ptr<sycl::half>(output), &zero_bits, sizeof(uint16_t)).wait();
            } else if (in_cont.dtype() == DType::BFloat16) {
                // Additive identity 0; bfloat16 +0.0 bit pattern.
                uint16_t zero_bits = 0x0000;
                queue.memcpy(get_data_ptr<uint16_t>(output), &zero_bits, sizeof(uint16_t)).wait();
            } else if (in_cont.dtype() == DType::Complex64) {
                float* out_ptr = get_data_ptr<float>(output);
                float zero_pair[2] = {0.0f, 0.0f};
                queue.memcpy(out_ptr, zero_pair, 2 * sizeof(float)).wait();
            } else if (in_cont.dtype() == DType::Complex128) {
                double* out_ptr = get_data_ptr<double>(output);
                double zero_pair[2] = {0.0, 0.0};
                queue.memcpy(out_ptr, zero_pair, 2 * sizeof(double)).wait();
            }
            return output;
        }

        if (in_cont.dtype() == DType::Float32) {
            const float* in_ptr = get_data_ptr<const float>(in_cont);
            float* out_ptr = get_data_ptr<float>(output);

            // Use parallel_for with reduction
            auto sum_buf = sycl::malloc_shared<float>(1, queue);
            sum_buf[0] = 0.0f;

            auto event = queue.parallel_for(sycl::range<1>(total_size), sycl::reduction(sum_buf, sycl::plus<float>()),
                              [=](sycl::id<1> idx, auto& sum) {
                sum += in_ptr[idx];
            });

            event.wait();
            out_ptr[0] = sum_buf[0];
            sycl::free(sum_buf, queue);
        }
        else if (in_cont.dtype() == DType::Float64) {
            const double* in_ptr = get_data_ptr<const double>(in_cont);
            double* out_ptr = get_data_ptr<double>(output);

            auto sum_buf = sycl::malloc_shared<double>(1, queue);
            sum_buf[0] = 0.0;

            auto event = queue.parallel_for(sycl::range<1>(total_size), sycl::reduction(sum_buf, sycl::plus<double>()),
                              [=](sycl::id<1> idx, auto& sum) {
                sum += in_ptr[idx];
            });

            event.wait();
            out_ptr[0] = sum_buf[0];
            sycl::free(sum_buf, queue);
        }
        else if (in_cont.dtype() == DType::Float16) {
            const sycl::half* in_ptr = get_data_ptr<const sycl::half>(in_cont);
            sycl::half* out_ptr = get_data_ptr<sycl::half>(output);

            // Use float accumulation for precision
            auto sum_buf = sycl::malloc_shared<float>(1, queue);
            sum_buf[0] = 0.0f;

            auto event = queue.parallel_for(sycl::range<1>(total_size), sycl::reduction(sum_buf, sycl::plus<float>()),
                              [=](sycl::id<1> idx, auto& sum) {
                sum += static_cast<float>(in_ptr[idx]);
            });

            event.wait();
            out_ptr[0] = sycl::half(sum_buf[0]);
            sycl::free(sum_buf, queue);
        }
        else if (in_cont.dtype() == DType::BFloat16) {
            const uint16_t* in_ptr = get_data_ptr<const uint16_t>(in_cont);
            uint16_t* out_ptr = get_data_ptr<uint16_t>(output);

            // Use float accumulation for precision
            auto sum_buf = sycl::malloc_shared<float>(1, queue);
            sum_buf[0] = 0.0f;

            auto event = queue.parallel_for(sycl::range<1>(total_size), sycl::reduction(sum_buf, sycl::plus<float>()),
                              [=](sycl::id<1> idx, auto& sum) {
                sum += bf16_to_f32(in_ptr[idx]);
            });

            event.wait();
            out_ptr[0] = f32_to_bf16(sum_buf[0]);
            sycl::free(sum_buf, queue);
        }
        else if (in_cont.dtype() == DType::Int32) {
            const int32_t* in_ptr = get_data_ptr<const int32_t>(in_cont);
            int32_t* out_ptr = get_data_ptr<int32_t>(output);

            // Use int64 accumulation to avoid overflow
            auto sum_buf = sycl::malloc_shared<int64_t>(1, queue);
            sum_buf[0] = 0;

            auto event = queue.parallel_for(sycl::range<1>(total_size), sycl::reduction(sum_buf, sycl::plus<int64_t>()),
                              [=](sycl::id<1> idx, auto& sum) {
                sum += static_cast<int64_t>(in_ptr[idx]);
            });

            event.wait();
            out_ptr[0] = static_cast<int32_t>(sum_buf[0]);
            sycl::free(sum_buf, queue);
        }
        else if (in_cont.dtype() == DType::Int64) {
            const int64_t* in_ptr = get_data_ptr<const int64_t>(in_cont);
            int64_t* out_ptr = get_data_ptr<int64_t>(output);

            // Device-side reduction using work-group reduce
            constexpr int64_t WG_SIZE = 256;
            int64_t num_wgs = (total_size + WG_SIZE - 1) / WG_SIZE;
            auto partial_buf = sycl::malloc_device<int64_t>(num_wgs, queue);

            queue.parallel_for(
                sycl::nd_range<1>(num_wgs * WG_SIZE, WG_SIZE),
                [=](sycl::nd_item<1> item) {
                    int64_t gid = item.get_global_id(0);
                    int64_t val = (gid < total_size) ? in_ptr[gid] : int64_t(0);
                    int64_t sum = sycl::reduce_over_group(item.get_group(), val, sycl::plus<int64_t>());
                    if (item.get_local_id(0) == 0) {
                        partial_buf[item.get_group(0)] = sum;
                    }
                }
            ).wait();

            // Second pass: reduce partial sums
            auto event2 = queue.parallel_for(
                sycl::nd_range<1>(WG_SIZE, WG_SIZE),
                [=](sycl::nd_item<1> item) {
                    int64_t lid = item.get_local_id(0);
                    int64_t val = (lid < num_wgs) ? partial_buf[lid] : int64_t(0);
                    int64_t sum = sycl::reduce_over_group(item.get_group(), val, sycl::plus<int64_t>());
                    if (lid == 0) out_ptr[0] = sum;
                }
            );
            event2.wait();
            sycl::free(partial_buf, queue);
        }
        else if (in_cont.dtype() == DType::UInt64) {
            // UInt64 is not in the public-op small-int promotion list, so the
            // kernel preserves UInt64. Native two-pass work-group reduction.
            const uint64_t* in_ptr = get_data_ptr<const uint64_t>(in_cont);
            uint64_t* out_ptr = get_data_ptr<uint64_t>(output);

            constexpr int64_t WG_SIZE = 256;
            int64_t num_wgs = (total_size + WG_SIZE - 1) / WG_SIZE;
            auto partial_buf = sycl::malloc_device<uint64_t>(num_wgs, queue);

            queue.parallel_for(
                sycl::nd_range<1>(num_wgs * WG_SIZE, WG_SIZE),
                [=](sycl::nd_item<1> item) {
                    int64_t gid = item.get_global_id(0);
                    uint64_t val = (gid < total_size) ? in_ptr[gid] : uint64_t(0);
                    uint64_t sum = sycl::reduce_over_group(item.get_group(), val, sycl::plus<uint64_t>());
                    if (item.get_local_id(0) == 0) {
                        partial_buf[item.get_group(0)] = sum;
                    }
                }
            ).wait();

            auto event2 = queue.parallel_for(
                sycl::nd_range<1>(WG_SIZE, WG_SIZE),
                [=](sycl::nd_item<1> item) {
                    int64_t lid = item.get_local_id(0);
                    uint64_t val = (lid < num_wgs) ? partial_buf[lid] : uint64_t(0);
                    uint64_t sum = sycl::reduce_over_group(item.get_group(), val, sycl::plus<uint64_t>());
                    if (lid == 0) out_ptr[0] = sum;
                }
            );
            event2.wait();
            sycl::free(partial_buf, queue);
        }
        else if (in_cont.dtype() == DType::Bool) {
            const uint8_t* in_ptr = get_data_ptr<const uint8_t>(in_cont);
            // Sum of bools returns Int64 — reassign output with correct dtype
            output = Tensor({}, DType::Int64, output.device());
            int64_t* out_ptr = get_data_ptr<int64_t>(output);

            auto sum_buf = sycl::malloc_shared<int32_t>(1, queue);
            sum_buf[0] = 0;

            auto event = queue.parallel_for(sycl::range<1>(total_size), sycl::reduction(sum_buf, sycl::plus<int32_t>()),
                              [=](sycl::id<1> idx, auto& s) {
                s += static_cast<int32_t>(in_ptr[idx] ? 1 : 0);
            });
            event.wait();

            int64_t sum = static_cast<int64_t>(sum_buf[0]);
            queue.memcpy(out_ptr, &sum, sizeof(int64_t)).wait();
            sycl::free(sum_buf, queue);
            return output;
        }
        else if (in_cont.dtype() == DType::Complex64) {
            // Complex stored as interleaved (re, im) floats.
            // Sum real and imag parts separately; SYCL's reduction infrastructure
            // doesn't have a built-in plus for std::complex.
            const float* in_ptr = reinterpret_cast<const float*>(in_cont.data_ptr());
            float* out_ptr = reinterpret_cast<float*>(const_cast<void*>(output.data_ptr()));
            auto re_buf = sycl::malloc_shared<float>(1, queue);
            auto im_buf = sycl::malloc_shared<float>(1, queue);
            re_buf[0] = 0.0f;
            im_buf[0] = 0.0f;
            queue.parallel_for(sycl::range<1>(total_size),
                sycl::reduction(re_buf, sycl::plus<float>()),
                sycl::reduction(im_buf, sycl::plus<float>()),
                [=](sycl::id<1> idx, auto& re, auto& im) {
                    re += in_ptr[2 * idx[0]];
                    im += in_ptr[2 * idx[0] + 1];
                }).wait();
            out_ptr[0] = re_buf[0];
            out_ptr[1] = im_buf[0];
            sycl::free(re_buf, queue);
            sycl::free(im_buf, queue);
        }
        else if (in_cont.dtype() == DType::Complex128) {
            const double* in_ptr = reinterpret_cast<const double*>(in_cont.data_ptr());
            double* out_ptr = reinterpret_cast<double*>(const_cast<void*>(output.data_ptr()));
            auto re_buf = sycl::malloc_shared<double>(1, queue);
            auto im_buf = sycl::malloc_shared<double>(1, queue);
            re_buf[0] = 0.0;
            im_buf[0] = 0.0;
            queue.parallel_for(sycl::range<1>(total_size),
                sycl::reduction(re_buf, sycl::plus<double>()),
                sycl::reduction(im_buf, sycl::plus<double>()),
                [=](sycl::id<1> idx, auto& re, auto& im) {
                    re += in_ptr[2 * idx[0]];
                    im += in_ptr[2 * idx[0] + 1];
                }).wait();
            out_ptr[0] = re_buf[0];
            out_ptr[1] = im_buf[0];
            sycl::free(re_buf, queue);
            sycl::free(im_buf, queue);
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
        else if (in_cont.dtype() == DType::UInt64) {
            const uint64_t* in_ptr = get_data_ptr<const uint64_t>(in_cont);
            uint64_t* out_ptr = get_data_ptr<uint64_t>(output);

            queue.parallel_for<SumKernelUInt64>(sycl::range<2>(outer_size, inner_size), [=](sycl::id<2> idx) {
                const int64_t outer_idx = idx[0];
                const int64_t inner_idx = idx[1];
                const int64_t base_offset = outer_idx * dim_size * inner_size + inner_idx;

                uint64_t sum = 0;
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
        else if (in_cont.dtype() == DType::Complex64) {
            const float* in_ptr = reinterpret_cast<const float*>(in_cont.data_ptr());
            float* out_ptr = reinterpret_cast<float*>(const_cast<void*>(output.data_ptr()));
            queue.parallel_for(sycl::range<2>(outer_size, inner_size), [=](sycl::id<2> idx) {
                const int64_t outer_idx = idx[0];
                const int64_t inner_idx = idx[1];
                const int64_t base_offset = outer_idx * dim_size * inner_size + inner_idx;
                float re = 0.0f;
                float im = 0.0f;
                for (int64_t d = 0; d < dim_size; ++d) {
                    int64_t src = 2 * (base_offset + d * inner_size);
                    re += in_ptr[src];
                    im += in_ptr[src + 1];
                }
                int64_t dst = 2 * (outer_idx * inner_size + inner_idx);
                out_ptr[dst]     = re;
                out_ptr[dst + 1] = im;
            });
        }
        else if (in_cont.dtype() == DType::Complex128) {
            const double* in_ptr = reinterpret_cast<const double*>(in_cont.data_ptr());
            double* out_ptr = reinterpret_cast<double*>(const_cast<void*>(output.data_ptr()));
            queue.parallel_for(sycl::range<2>(outer_size, inner_size), [=](sycl::id<2> idx) {
                const int64_t outer_idx = idx[0];
                const int64_t inner_idx = idx[1];
                const int64_t base_offset = outer_idx * dim_size * inner_size + inner_idx;
                double re = 0.0;
                double im = 0.0;
                for (int64_t d = 0; d < dim_size; ++d) {
                    int64_t src = 2 * (base_offset + d * inner_size);
                    re += in_ptr[src];
                    im += in_ptr[src + 1];
                }
                int64_t dst = 2 * (outer_idx * inner_size + inner_idx);
                out_ptr[dst]     = re;
                out_ptr[dst + 1] = im;
            });
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

        // Handle empty tensor: mean of empty = NaN (0/0). The caching allocator
        // does not zero memory, so every floating/complex dtype must explicitly
        // write NaN (PyTorch parity) rather than leaving garbage.
        if (total_size == 0) {
            if (in_cont.dtype() == DType::Float32) {
                float* out_ptr = get_data_ptr<float>(output);
                float nan_val = std::numeric_limits<float>::quiet_NaN();
                queue.memcpy(out_ptr, &nan_val, sizeof(float)).wait();
            } else if (in_cont.dtype() == DType::Float64) {
                double* out_ptr = get_data_ptr<double>(output);
                double nan_val = std::numeric_limits<double>::quiet_NaN();
                queue.memcpy(out_ptr, &nan_val, sizeof(double)).wait();
            } else if (in_cont.dtype() == DType::Float16) {
                // IEEE-754 half quiet-NaN bit pattern (sign 0, exp all 1s, msb of mantissa set)
                uint16_t nan_bits = 0x7E00;
                queue.memcpy(get_data_ptr<sycl::half>(output), &nan_bits, sizeof(uint16_t)).wait();
            } else if (in_cont.dtype() == DType::BFloat16) {
                // bfloat16 quiet-NaN bit pattern (exp all 1s, msb of mantissa set)
                uint16_t nan_bits = 0x7FC0;
                queue.memcpy(get_data_ptr<uint16_t>(output), &nan_bits, sizeof(uint16_t)).wait();
            } else if (in_cont.dtype() == DType::Complex64) {
                float* out_ptr = get_data_ptr<float>(output);
                float nan_pair[2] = {std::numeric_limits<float>::quiet_NaN(),
                                     std::numeric_limits<float>::quiet_NaN()};
                queue.memcpy(out_ptr, nan_pair, 2 * sizeof(float)).wait();
            } else if (in_cont.dtype() == DType::Complex128) {
                double* out_ptr = get_data_ptr<double>(output);
                double nan_pair[2] = {std::numeric_limits<double>::quiet_NaN(),
                                      std::numeric_limits<double>::quiet_NaN()};
                queue.memcpy(out_ptr, nan_pair, 2 * sizeof(double)).wait();
            }
            return output;
        }

        if (in_cont.dtype() == DType::Float32) {
            const float* in_ptr = get_data_ptr<const float>(in_cont);
            float* out_ptr = get_data_ptr<float>(output);
            const float scale = 1.0f / static_cast<float>(total_size);

            auto sum_buf = sycl::malloc_shared<float>(1, queue);
            sum_buf[0] = 0.0f;

            auto event = queue.parallel_for(sycl::range<1>(total_size), sycl::reduction(sum_buf, sycl::plus<float>()),
                              [=](sycl::id<1> idx, auto& sum) {
                sum += in_ptr[idx];
            });

            event.wait();
            out_ptr[0] = sum_buf[0] * scale;
            sycl::free(sum_buf, queue);
        }
        else if (in_cont.dtype() == DType::Float64) {
            const double* in_ptr = get_data_ptr<const double>(in_cont);
            double* out_ptr = get_data_ptr<double>(output);
            const double scale = 1.0 / static_cast<double>(total_size);

            auto sum_buf = sycl::malloc_shared<double>(1, queue);
            sum_buf[0] = 0.0;

            auto event = queue.parallel_for(sycl::range<1>(total_size), sycl::reduction(sum_buf, sycl::plus<double>()),
                              [=](sycl::id<1> idx, auto& sum) {
                sum += in_ptr[idx];
            });

            event.wait();
            out_ptr[0] = sum_buf[0] * scale;
            sycl::free(sum_buf, queue);
        }
        else if (in_cont.dtype() == DType::Float16) {
            const sycl::half* in_ptr = get_data_ptr<const sycl::half>(in_cont);
            sycl::half* out_ptr = get_data_ptr<sycl::half>(output);
            const float scale = 1.0f / static_cast<float>(total_size);

            auto sum_buf = sycl::malloc_shared<float>(1, queue);
            sum_buf[0] = 0.0f;

            auto event = queue.parallel_for(sycl::range<1>(total_size), sycl::reduction(sum_buf, sycl::plus<float>()),
                              [=](sycl::id<1> idx, auto& sum) {
                sum += static_cast<float>(in_ptr[idx]);
            });

            event.wait();
            out_ptr[0] = sycl::half(sum_buf[0] * scale);
            sycl::free(sum_buf, queue);
        }
        else if (in_cont.dtype() == DType::BFloat16) {
            const uint16_t* in_ptr = get_data_ptr<const uint16_t>(in_cont);
            uint16_t* out_ptr = get_data_ptr<uint16_t>(output);
            const float scale = 1.0f / static_cast<float>(total_size);

            auto sum_buf = sycl::malloc_shared<float>(1, queue);
            sum_buf[0] = 0.0f;

            auto event = queue.parallel_for(sycl::range<1>(total_size), sycl::reduction(sum_buf, sycl::plus<float>()),
                              [=](sycl::id<1> idx, auto& sum) {
                sum += bf16_to_f32(in_ptr[idx]);
            });

            event.wait();
            out_ptr[0] = f32_to_bf16(sum_buf[0] * scale);
            sycl::free(sum_buf, queue);
        }
        else if (in_cont.dtype() == DType::Complex64 || in_cont.dtype() == DType::Complex128) {
            // Complex mean: separate real/imag reductions over the interleaved
            // (re, im) storage, then divide by element count.
            auto run_c = [&]<typename R>() {
                const R* in_ptr = get_data_ptr<const R>(in_cont);
                R* out_ptr = get_data_ptr<R>(output);
                const R scale = R(1) / static_cast<R>(total_size);
                auto re_buf = sycl::malloc_shared<R>(1, queue);
                auto im_buf = sycl::malloc_shared<R>(1, queue);
                re_buf[0] = R(0); im_buf[0] = R(0);
                queue.parallel_for(sycl::range<1>(total_size),
                                   sycl::reduction(re_buf, sycl::plus<R>()),
                                   sycl::reduction(im_buf, sycl::plus<R>()),
                                   [=](auto idx, auto& re_s, auto& im_s) {
                    re_s += in_ptr[idx[0] * 2];
                    im_s += in_ptr[idx[0] * 2 + 1];
                }).wait();
                queue.wait_and_throw();
                out_ptr[0] = re_buf[0] * scale;
                out_ptr[1] = im_buf[0] * scale;
                sycl::free(re_buf, queue);
                sycl::free(im_buf, queue);
            };
            if (in_cont.dtype() == DType::Complex64) run_c.template operator()<float>();
            else run_c.template operator()<double>();
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
        else if (in_cont.dtype() == DType::Complex64) {
            const float* in_ptr = reinterpret_cast<const float*>(in_cont.data_ptr());
            float* out_ptr = reinterpret_cast<float*>(const_cast<void*>(output.data_ptr()));
            const float scale = 1.0f / static_cast<float>(dim_size);

            queue.parallel_for(sycl::range<2>(outer_size, inner_size), [=](sycl::id<2> idx) {
                const int64_t outer_idx = idx[0];
                const int64_t inner_idx = idx[1];
                const int64_t base_offset = outer_idx * dim_size * inner_size + inner_idx;
                float re = 0.0f;
                float im = 0.0f;
                for (int64_t d = 0; d < dim_size; ++d) {
                    int64_t src = 2 * (base_offset + d * inner_size);
                    re += in_ptr[src];
                    im += in_ptr[src + 1];
                }
                int64_t dst = 2 * (outer_idx * inner_size + inner_idx);
                out_ptr[dst]     = re * scale;
                out_ptr[dst + 1] = im * scale;
            });
        }
        else if (in_cont.dtype() == DType::Complex128) {
            const double* in_ptr = reinterpret_cast<const double*>(in_cont.data_ptr());
            double* out_ptr = reinterpret_cast<double*>(const_cast<void*>(output.data_ptr()));
            const double scale = 1.0 / static_cast<double>(dim_size);

            queue.parallel_for(sycl::range<2>(outer_size, inner_size), [=](sycl::id<2> idx) {
                const int64_t outer_idx = idx[0];
                const int64_t inner_idx = idx[1];
                const int64_t base_offset = outer_idx * dim_size * inner_size + inner_idx;
                double re = 0.0;
                double im = 0.0;
                for (int64_t d = 0; d < dim_size; ++d) {
                    int64_t src = 2 * (base_offset + d * inner_size);
                    re += in_ptr[src];
                    im += in_ptr[src + 1];
                }
                int64_t dst = 2 * (outer_idx * inner_size + inner_idx);
                out_ptr[dst]     = re * scale;
                out_ptr[dst + 1] = im * scale;
            });
        }
        else {
            throw std::runtime_error("Unsupported dtype for mean reduction");
        }
    }

    return output;
}

// Max reduction kernel
auto max_kernel(const Tensor& input_raw, int64_t dim, bool keepdim, sycl::queue& queue) -> Tensor {
    // Ensure contiguous input: the shape-derived offset math below assumes row-major
    // contiguous layout, so a non-contiguous (transpose/permute/slice) view must be
    // materialized first to avoid reading the wrong elements.
    Tensor input = input_raw.is_contiguous() ? input_raw : contiguous_kernel(input_raw, queue);
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

        // max has no identity element; PyTorch raises on an empty input. Without
        // this guard the single_task loop never runs, leaving the in-kernel
        // sentinel / uninitialized USM garbage in out_ptr[0].
        if (total_size == 0) {
            throw std::runtime_error("max: cannot reduce empty tensor");
        }

        if (input.dtype() == DType::Float32) {
            const float* in_ptr = get_data_ptr<const float>(input);
            float* out_ptr = get_data_ptr<float>(output);

            queue.single_task([=]() {
                float max_val = -std::numeric_limits<float>::infinity();
                for (int64_t i = 0; i < total_size; ++i) {
                    max_val = nan_fmax(max_val, in_ptr[i]);
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
                    max_val = nan_fmax(max_val, in_ptr[i]);
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
                    max_val = nan_fmax(max_val, static_cast<float>(in_ptr[i]));
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
                    max_val = nan_fmax(max_val, bf16_to_f32(in_ptr[i]));
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
        else if (input.dtype() == DType::Int64) {
            const int64_t* in_ptr = get_data_ptr<const int64_t>(input);
            int64_t* out_ptr = get_data_ptr<int64_t>(output);

            queue.single_task([=]() {
                int64_t max_val = std::numeric_limits<int64_t>::min();
                for (int64_t i = 0; i < total_size; ++i) {
                    if (in_ptr[i] > max_val) max_val = in_ptr[i];
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
                    max_val = nan_fmax(max_val, val);
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
                    max_val = nan_fmax(max_val, val);
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
                    max_val = nan_fmax(max_val, val);
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
                    max_val = nan_fmax(max_val, val);
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
        else if (input.dtype() == DType::Int64) {
            const int64_t* in_ptr = get_data_ptr<const int64_t>(input);
            int64_t* out_ptr = get_data_ptr<int64_t>(output);

            queue.parallel_for<MaxKernelInt64>(sycl::range<2>(outer_size, inner_size), [=](sycl::id<2> idx) {
                const int64_t outer_idx = idx[0];
                const int64_t inner_idx = idx[1];
                const int64_t base_offset = outer_idx * dim_size * inner_size + inner_idx;

                int64_t max_val = std::numeric_limits<int64_t>::min();
                for (int64_t d = 0; d < dim_size; ++d) {
                    int64_t val = in_ptr[base_offset + d * inner_size];
                    if (val > max_val) max_val = val;
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
auto min_kernel(const Tensor& input_raw, int64_t dim, bool keepdim, sycl::queue& queue) -> Tensor {
    // Ensure contiguous input: the shape-derived offset math below assumes row-major
    // contiguous layout, so a non-contiguous view must be materialized first.
    Tensor input = input_raw.is_contiguous() ? input_raw : contiguous_kernel(input_raw, queue);
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

        // min has no identity element; PyTorch raises on an empty input. Without
        // this guard the single_task loop never runs, leaving the in-kernel
        // sentinel / uninitialized USM garbage in out_ptr[0].
        if (total_size == 0) {
            throw std::runtime_error("min: cannot reduce empty tensor");
        }

        if (input.dtype() == DType::Float32) {
            const float* in_ptr = get_data_ptr<const float>(input);
            float* out_ptr = get_data_ptr<float>(output);

            queue.single_task([=]() {
                float min_val = std::numeric_limits<float>::infinity();
                for (int64_t i = 0; i < total_size; ++i) {
                    min_val = nan_fmin(min_val, in_ptr[i]);
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
                    min_val = nan_fmin(min_val, in_ptr[i]);
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
                    min_val = nan_fmin(min_val, static_cast<float>(in_ptr[i]));
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
                    min_val = nan_fmin(min_val, bf16_to_f32(in_ptr[i]));
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
        else if (input.dtype() == DType::Int64) {
            const int64_t* in_ptr = get_data_ptr<const int64_t>(input);
            int64_t* out_ptr = get_data_ptr<int64_t>(output);

            queue.single_task([=]() {
                int64_t min_val = std::numeric_limits<int64_t>::max();
                for (int64_t i = 0; i < total_size; ++i) {
                    if (in_ptr[i] < min_val) min_val = in_ptr[i];
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
                    min_val = nan_fmin(min_val, val);
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
                    min_val = nan_fmin(min_val, val);
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
                    min_val = nan_fmin(min_val, val);
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
                    min_val = nan_fmin(min_val, val);
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
        else if (input.dtype() == DType::Int64) {
            const int64_t* in_ptr = get_data_ptr<const int64_t>(input);
            int64_t* out_ptr = get_data_ptr<int64_t>(output);

            queue.parallel_for<MinKernelInt64>(sycl::range<2>(outer_size, inner_size), [=](sycl::id<2> idx) {
                const int64_t outer_idx = idx[0];
                const int64_t inner_idx = idx[1];
                const int64_t base_offset = outer_idx * dim_size * inner_size + inner_idx;

                int64_t min_val = std::numeric_limits<int64_t>::max();
                for (int64_t d = 0; d < dim_size; ++d) {
                    int64_t val = in_ptr[base_offset + d * inner_size];
                    if (val < min_val) min_val = val;
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
 * @brief Two-phase device-side argmax/argmin full reduction.
 *
 * Phase 1: Each work-group reduces its portion into a (value, index) pair
 *          stored in partial result buffers using local memory + group barriers.
 * Phase 2: A single work-group reduces the partial results to the final answer.
 *
 * Template parameters:
 *   T         - element type (float, double, int32_t, or float for half promotion)
 *   Phase1Name, Phase2Name - SYCL kernel name types
 *   IsMax     - true for argmax, false for argmin
 *   ReadT     - the actual device pointer type (may differ from T for Float16)
 */
template<typename T, typename Phase1Name, typename Phase2Name, bool IsMax, typename ReadT = T>
static void sycl_arg_reduce_full(const ReadT* in_ptr, int64_t* out_ptr,
                                 int64_t total_size, sycl::queue& queue) {
    constexpr int64_t WG_SIZE = 256;
    const int64_t num_wgs = std::min((total_size + WG_SIZE - 1) / WG_SIZE, int64_t{1024});

    // Allocate partial results on device
    T* partial_vals = sycl::malloc_device<T>(num_wgs, queue);
    int64_t* partial_idxs = sycl::malloc_device<int64_t>(num_wgs, queue);

    // Phase 1: Each work-group reduces its chunk
    auto phase1_event = queue.submit([&](sycl::handler& cgh) {
        sycl::local_accessor<T, 1> local_vals(sycl::range<1>(WG_SIZE), cgh);
        sycl::local_accessor<int64_t, 1> local_idxs(sycl::range<1>(WG_SIZE), cgh);

        cgh.parallel_for<Phase1Name>(
            sycl::nd_range<1>(num_wgs * WG_SIZE, WG_SIZE),
            [=](sycl::nd_item<1> item) {
                const int64_t gid = item.get_global_id(0);
                const int64_t lid = item.get_local_id(0);
                const int64_t wg_id = item.get_group(0);

                // Each thread processes multiple elements with stride
                T best_val;
                if constexpr (IsMax) {
                    best_val = std::numeric_limits<T>::lowest();
                } else {
                    best_val = std::numeric_limits<T>::max();
                }
                // INT64_MAX sentinel: a thread whose grid-stride range is empty
                // (gid >= total_size) never wins a value-tie against a real
                // element on the index tie-break below.
                int64_t best_idx = std::numeric_limits<int64_t>::max();

                for (int64_t i = gid; i < total_size; i += num_wgs * WG_SIZE) {
                    T val;
                    if constexpr (std::is_same_v<ReadT, T>) {
                        val = in_ptr[i];
                    } else {
                        // Float16 path: read sycl::half, promote to float
                        val = static_cast<T>(in_ptr[i]);
                    }
                    if constexpr (IsMax) {
                        if (val > best_val) { best_val = val; best_idx = i; }
                    } else {
                        if (val < best_val) { best_val = val; best_idx = i; }
                    }
                }

                local_vals[lid] = best_val;
                local_idxs[lid] = best_idx;
                sycl::group_barrier(item.get_group());

                // Tree reduction in shared memory. Tie-break by smaller flat
                // index so the result matches the CPU reference (first
                // occurrence) regardless of which thread held the duplicate.
                for (int64_t stride = WG_SIZE / 2; stride > 0; stride >>= 1) {
                    if (lid < stride) {
                        const T ov = local_vals[lid + stride];
                        const int64_t oidx = local_idxs[lid + stride];
                        if constexpr (IsMax) {
                            if (ov > local_vals[lid] ||
                                (ov == local_vals[lid] && oidx < local_idxs[lid])) {
                                local_vals[lid] = ov;
                                local_idxs[lid] = oidx;
                            }
                        } else {
                            if (ov < local_vals[lid] ||
                                (ov == local_vals[lid] && oidx < local_idxs[lid])) {
                                local_vals[lid] = ov;
                                local_idxs[lid] = oidx;
                            }
                        }
                    }
                    sycl::group_barrier(item.get_group());
                }

                if (lid == 0) {
                    partial_vals[wg_id] = local_vals[0];
                    partial_idxs[wg_id] = local_idxs[0];
                }
            });
    });

    // Phase 2: A single work-group reduces ALL `num_wgs` partial results.
    //
    // The work-group is always WG_SIZE wide; each thread first folds an
    // arbitrary number of partials via a grid-stride loop (lid, lid+WG_SIZE,
    // lid+2*WG_SIZE, ...). This guarantees every partial is consumed even when
    // num_wgs > WG_SIZE (input > WG_SIZE*WG_SIZE = 65536 elements) — the old
    // code capped the work-group at WG_SIZE and read only the first WG_SIZE
    // partials via `local_vals[lid] = partial_vals[lid]`, silently dropping the
    // tail and returning a wrong index for large tensors.
    //
    // Tie-breaking matches the CPU reference (first occurrence = smallest flat
    // index): on equal values we keep the candidate with the smaller stored
    // index, in both the grid-stride fold and the in-group tree reduction.
    const int64_t phase2_wg = WG_SIZE;

    queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(phase1_event);
        sycl::local_accessor<T, 1> local_vals(sycl::range<1>(phase2_wg), cgh);
        sycl::local_accessor<int64_t, 1> local_idxs(sycl::range<1>(phase2_wg), cgh);

        const int64_t n_partials = num_wgs;

        cgh.parallel_for<Phase2Name>(
            sycl::nd_range<1>(phase2_wg, phase2_wg),
            [=](sycl::nd_item<1> item) {
                const int64_t lid = item.get_local_id(0);

                T best_val;
                if constexpr (IsMax) {
                    best_val = std::numeric_limits<T>::lowest();
                } else {
                    best_val = std::numeric_limits<T>::max();
                }
                int64_t best_idx = std::numeric_limits<int64_t>::max();

                // Grid-stride fold: every partial in [0, n_partials) is visited
                // by exactly one thread, regardless of how large n_partials is.
                for (int64_t p = lid; p < n_partials; p += phase2_wg) {
                    const T v = partial_vals[p];
                    const int64_t pidx = partial_idxs[p];
                    if constexpr (IsMax) {
                        if (v > best_val || (v == best_val && pidx < best_idx)) {
                            best_val = v; best_idx = pidx;
                        }
                    } else {
                        if (v < best_val || (v == best_val && pidx < best_idx)) {
                            best_val = v; best_idx = pidx;
                        }
                    }
                }

                local_vals[lid] = best_val;
                local_idxs[lid] = best_idx;
                sycl::group_barrier(item.get_group());

                for (int64_t stride = phase2_wg / 2; stride > 0; stride >>= 1) {
                    if (lid < stride) {
                        const T ov = local_vals[lid + stride];
                        const int64_t oidx = local_idxs[lid + stride];
                        if constexpr (IsMax) {
                            if (ov > local_vals[lid] ||
                                (ov == local_vals[lid] && oidx < local_idxs[lid])) {
                                local_vals[lid] = ov;
                                local_idxs[lid] = oidx;
                            }
                        } else {
                            if (ov < local_vals[lid] ||
                                (ov == local_vals[lid] && oidx < local_idxs[lid])) {
                                local_vals[lid] = ov;
                                local_idxs[lid] = oidx;
                            }
                        }
                    }
                    sycl::group_barrier(item.get_group());
                }

                if (lid == 0) {
                    out_ptr[0] = local_idxs[0];
                }
            });
    }).wait();

    sycl::free(partial_vals, queue);
    sycl::free(partial_idxs, queue);
}

/**
 * @brief Argmax operation - returns indices of maximum values.
 *
 * Finds the indices of maximum values along a dimension.
 * Uses two-phase device-side reduction for full (dim=-1) case.
 *
 * @param input Input tensor
 * @param dim Dimension to reduce along (-1 for all dimensions)
 * @param keepdim Whether to keep the reduced dimension
 * @param queue SYCL queue for execution
 * @return Tensor containing indices (Int64 dtype)
 */
auto argmax_kernel(const Tensor& input_raw, int64_t dim, bool keepdim, sycl::queue& queue) -> Tensor {
    // Argmax returns INDICES, invariant under a lossless integer widen. oneAPI
    // has native argmax kernels only for {F32,F64,F16,Int32}; widen the narrow
    // integer / Bool types to Int32 (exact) so they're supported.
    {
        const DType dt = input_raw.dtype();
        if (dt == DType::Int8 || dt == DType::UInt8 || dt == DType::Bool) {
            return argmax_kernel(input_raw.to(DType::Int32), dim, keepdim, queue);
        }
        if (dt == DType::Int16 || dt == DType::UInt16) {
            // A *device* 16-bit integer cast crashes the Intel oneAPI runtime
            // ("submit a bug report to Intel"); the cast must happen on the host.
            // Device<->host memcpy and host casts of 16-bit ints are both safe,
            // so widen on CPU (lossless), then run the Int32 argmax ON DEVICE.
            Tensor as_i32 = input_raw.to(Device::cpu()).to(DType::Int32).to(input_raw.device());
            return argmax_kernel(as_i32, dim, keepdim, queue);
        }
    }
    // Ensure contiguous input: the shape-derived offset math below assumes row-major
    // contiguous layout, so a non-contiguous view must be materialized first.
    Tensor input = input_raw.is_contiguous() ? input_raw : contiguous_kernel(input_raw, queue);
    const auto& shape = input.shape();

    // For full reduction (dim=-1)
    if (dim == -1) {
        int64_t total_size = 1;
        for (auto s : shape) {
            total_size *= s;
        }

        // argmax has no identity element; PyTorch raises on a full reduction of an
        // empty tensor. The two-phase device reduction would otherwise write an
        // uninitialized index into out_ptr[0].
        if (total_size == 0) {
            throw std::runtime_error("argmax: cannot reduce empty tensor");
        }

        // Full reduction without keepdim yields a scalar (shape {}),
        // matching the CPU backend and PyTorch conventions.
        std::vector<int64_t> out_shape = keepdim ? std::vector<int64_t>(shape.size(), 1) : std::vector<int64_t>{};
        Tensor output(out_shape, DType::Int64, input.device());
        int64_t* out_ptr = get_data_ptr<int64_t>(output);

        if (input.dtype() == DType::Float32) {
            sycl_arg_reduce_full<float, ArgmaxFullPhase1Float32, ArgmaxFullPhase2Float32, true>(
                get_data_ptr<const float>(input), out_ptr, total_size, queue);
        }
        else if (input.dtype() == DType::Float64) {
            sycl_arg_reduce_full<double, ArgmaxFullPhase1Float64, ArgmaxFullPhase2Float64, true>(
                get_data_ptr<const double>(input), out_ptr, total_size, queue);
        }
        else if (input.dtype() == DType::Float16) {
            sycl_arg_reduce_full<float, ArgmaxFullPhase1Float16, ArgmaxFullPhase2Float16, true, sycl::half>(
                get_data_ptr<const sycl::half>(input), out_ptr, total_size, queue);
        }
        else if (input.dtype() == DType::Int32) {
            sycl_arg_reduce_full<int32_t, ArgmaxFullPhase1Int32, ArgmaxFullPhase2Int32, true>(
                get_data_ptr<const int32_t>(input), out_ptr, total_size, queue);
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
auto argmin_kernel(const Tensor& input_raw, int64_t dim, bool keepdim, sycl::queue& queue) -> Tensor {
    // See argmax_kernel: widen narrow-int / bool to Int32 (exact, index-
    // invariant). 16-bit casts must be done on the host (device 16-bit cast
    // crashes the Intel oneAPI runtime); the argmin itself stays on device.
    {
        const DType dt = input_raw.dtype();
        if (dt == DType::Int8 || dt == DType::UInt8 || dt == DType::Bool) {
            return argmin_kernel(input_raw.to(DType::Int32), dim, keepdim, queue);
        }
        if (dt == DType::Int16 || dt == DType::UInt16) {
            Tensor as_i32 = input_raw.to(Device::cpu()).to(DType::Int32).to(input_raw.device());
            return argmin_kernel(as_i32, dim, keepdim, queue);
        }
    }
    // Ensure contiguous input: the shape-derived offset math below assumes row-major
    // contiguous layout, so a non-contiguous view must be materialized first.
    Tensor input = input_raw.is_contiguous() ? input_raw : contiguous_kernel(input_raw, queue);
    const auto& shape = input.shape();

    // For full reduction (dim=-1) — two-phase device-side reduction
    if (dim == -1) {
        int64_t total_size = 1;
        for (auto s : shape) {
            total_size *= s;
        }

        // argmin has no identity element; PyTorch raises on a full reduction of an
        // empty tensor. The two-phase device reduction would otherwise write an
        // uninitialized index into out_ptr[0].
        if (total_size == 0) {
            throw std::runtime_error("argmin: cannot reduce empty tensor");
        }

        // Full reduction without keepdim yields a scalar (shape {}),
        // matching the CPU backend and PyTorch conventions.
        std::vector<int64_t> out_shape = keepdim ? std::vector<int64_t>(shape.size(), 1) : std::vector<int64_t>{};
        Tensor output(out_shape, DType::Int64, input.device());
        int64_t* out_ptr = get_data_ptr<int64_t>(output);

        if (input.dtype() == DType::Float32) {
            sycl_arg_reduce_full<float, ArgminFullPhase1Float32, ArgminFullPhase2Float32, false>(
                get_data_ptr<const float>(input), out_ptr, total_size, queue);
        }
        else if (input.dtype() == DType::Float64) {
            sycl_arg_reduce_full<double, ArgminFullPhase1Float64, ArgminFullPhase2Float64, false>(
                get_data_ptr<const double>(input), out_ptr, total_size, queue);
        }
        else if (input.dtype() == DType::Float16) {
            sycl_arg_reduce_full<float, ArgminFullPhase1Float16, ArgminFullPhase2Float16, false, sycl::half>(
                get_data_ptr<const sycl::half>(input), out_ptr, total_size, queue);
        }
        else if (input.dtype() == DType::Int32) {
            sycl_arg_reduce_full<int32_t, ArgminFullPhase1Int32, ArgminFullPhase2Int32, false>(
                get_data_ptr<const int32_t>(input), out_ptr, total_size, queue);
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

    int64_t inner_size = 1;
    for (int64_t i = dim + 1; i < ndim; ++i) inner_size *= shape[i];

#ifdef TENZOR_HAS_ONEDPL
    // Device-side sort for contiguous sort dimension (inner_size == 1)
    if (inner_size == 1) {
        int64_t outer_size = 1;
        for (int64_t i = 0; i < dim; ++i) outer_size *= shape[i];
        int64_t dim_size = shape[dim];
        auto policy = ::oneapi::dpl::execution::make_device_policy(queue);

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
                    ::oneapi::dpl::sort_by_key(policy, slice_vals, slice_vals + dim_size,
                                              slice_idx, std::greater<T>());
                } else {
                    ::oneapi::dpl::sort_by_key(policy, slice_vals, slice_vals + dim_size,
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
    // Fall through: transpose so sort dim is last, sort on device, transpose back
    {
        std::vector<int64_t> perm(ndim);
        std::iota(perm.begin(), perm.end(), 0);
        std::swap(perm[dim], perm[ndim - 1]);

        // Build inverse permutation
        std::vector<int64_t> inv_perm(ndim);
        for (int64_t i = 0; i < ndim; ++i) inv_perm[perm[i]] = i;

        Tensor transposed = input.permute(perm).contiguous();
        auto [sorted_t, indices_t] = sort_kernel(transposed, ndim - 1, descending, queue);

        // Permute back
        values = sorted_t.permute(inv_perm).contiguous();
        indices = indices_t.permute(inv_perm).contiguous();
    }

    return {values, indices};
#else
    // Without oneDPL, no device-side sort is available
    throw std::runtime_error("sort: oneDPL required for device-side sort");
#endif
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

    int64_t inner_size = 1;
    for (int64_t i = dim + 1; i < ndim; ++i) inner_size *= shape[i];

#ifdef TENZOR_HAS_ONEDPL
    // Device-side topk for contiguous sort dimension (inner_size == 1)
    // Strategy: sort each slice on device via oneDPL, then copy first k elements
    if (inner_size == 1) {
        int64_t outer_size = 1;
        for (int64_t i = 0; i < dim; ++i) outer_size *= shape[i];
        int64_t dim_size = shape[dim];
        auto policy = ::oneapi::dpl::execution::make_device_policy(queue);
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
                    ::oneapi::dpl::sort_by_key(policy, slice_vals, slice_vals + dim_size,
                                              slice_idx, std::greater<T>());
                } else {
                    ::oneapi::dpl::sort_by_key(policy, slice_vals, slice_vals + dim_size,
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
    // Fall through: transpose so sort dim is last, topk on device, transpose back
    {
        std::vector<int64_t> perm(ndim);
        std::iota(perm.begin(), perm.end(), 0);
        std::swap(perm[dim], perm[ndim - 1]);

        std::vector<int64_t> inv_perm(ndim);
        for (int64_t i = 0; i < ndim; ++i) inv_perm[perm[i]] = i;

        Tensor transposed = input.permute(perm).contiguous();
        auto [vals_t, idxs_t] = topk_kernel(transposed, k, ndim - 1, largest, sorted, queue);

        values = vals_t.permute(inv_perm).contiguous();
        indices = idxs_t.permute(inv_perm).contiguous();
    }

    return {values, indices};
#else
    // Without oneDPL, no device-side topk is available
    throw std::runtime_error("topk: oneDPL required for device-side topk");
#endif
}

// ============================================================================
// Unique kernel - find unique values
// ============================================================================
auto unique_kernel(const Tensor& input, bool sorted, bool return_inverse, bool return_counts,
                   sycl::queue& queue) -> std::tuple<Tensor, Tensor, Tensor> {
    int64_t numel = input.numel();

    // Upcast unsupported dtypes to wider types supported by the device paths
    if (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16) {
        auto f32 = cast_kernel(input, DType::Float32, queue);
        auto [u, inv, cnt] = unique_kernel(f32, sorted, return_inverse, return_counts, queue);
        return {cast_kernel(u, input.dtype(), queue), inv, cnt};
    }
    if (input.dtype() == DType::Int8 || input.dtype() == DType::UInt8 || input.dtype() == DType::Bool) {
        auto i32 = cast_kernel(input, DType::Int32, queue);
        auto [u, inv, cnt] = unique_kernel(i32, sorted, return_inverse, return_counts, queue);
        return {cast_kernel(u, input.dtype(), queue), inv, cnt};
    }

#ifdef TENZOR_HAS_ONEDPL
    // Device-side unique using oneDPL: sort + unique on device.
    // Used for both sorted=true and sorted=false — unique values have no canonical
    // order, so sorting is acceptable even when the user did not request it.
    if (numel > 0) {
        auto policy = ::oneapi::dpl::execution::make_device_policy(queue);

        auto device_unique_impl = [&](const auto* in_ptr) {
            using T = std::remove_const_t<std::remove_pointer_t<decltype(in_ptr)>>;

            // Allocate device buffer for sorted copy
            T* d_sorted = sycl::malloc_device<T>(numel, queue);
            queue.memcpy(d_sorted, in_ptr, numel * sizeof(T)).wait();

            // Sort on device
            ::oneapi::dpl::sort(policy, d_sorted, d_sorted + numel);

            // Find unique elements using oneDPL unique
            auto new_end = ::oneapi::dpl::unique(policy, d_sorted, d_sorted + numel);
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
                    inv_ptr[gid] = (lo < n_unique) ? lo : (n_unique - 1);
                }).wait();
            }

            // Compute counts if needed — device-side using atomic_ref
            Tensor out_counts({return_counts ? n_unique : 0}, DType::Int64, input.device());
            if (return_counts && n_unique > 0) {
                int64_t* cnt_ptr = get_data_ptr<int64_t>(out_counts);
                queue.memset(cnt_ptr, 0, n_unique * sizeof(int64_t)).wait();

                // Ensure we have inverse indices on device
                const int64_t* inv_ptr;
                int64_t* tmp_inv = nullptr;
                if (return_inverse) {
                    inv_ptr = get_data_ptr<const int64_t>(out_inverse);
                } else {
                    // Compute inverse on device for counting
                    tmp_inv = sycl::malloc_device<int64_t>(numel, queue);
                    const T* unique_ptr = get_data_ptr<const T>(out_vals);
                    queue.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> gid) {
                        T val = in_ptr[gid];
                        int64_t lo = 0, hi = n_unique;
                        while (lo < hi) {
                            int64_t mid = lo + (hi - lo) / 2;
                            if (unique_ptr[mid] < val) lo = mid + 1;
                            else hi = mid;
                        }
                        tmp_inv[gid] = (lo < n_unique) ? lo : (n_unique - 1);
                    }).wait();
                    inv_ptr = tmp_inv;
                }

                // Atomically increment counts on device using int32 atomics
                // (int64 atomic not universally supported; use two int32 halves or int32 counts)
                // Use int32 counts buffer, then widen to int64
                int32_t* d_counts32 = sycl::malloc_device<int32_t>(n_unique, queue);
                queue.memset(d_counts32, 0, n_unique * sizeof(int32_t)).wait();

                queue.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> gid) {
                    int64_t idx = inv_ptr[gid];
                    sycl::atomic_ref<int32_t, sycl::memory_order::relaxed,
                                     sycl::memory_scope::device,
                                     sycl::access::address_space::global_space> aref(d_counts32[idx]);
                    aref.fetch_add(1);
                }).wait();

                // Widen int32 counts to int64
                queue.parallel_for(sycl::range<1>(n_unique), [=](sycl::id<1> i) {
                    cnt_ptr[i] = static_cast<int64_t>(d_counts32[i]);
                }).wait();

                sycl::free(d_counts32, queue);
                if (tmp_inv) sycl::free(tmp_inv, queue);
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
        // Unsupported dtypes fall through to the device-side sorted-unique impl
        // below (or a loud "unsupported dtype" throw) — there is no host path.
    }
    // SYCL device-side unique (sorted path only; unsorted throws)
#endif

    // Device-side unique for sorted=true without oneDPL:
    // Sort on device -> mark boundaries -> prefix sum -> compact + inverse + counts
    if (sorted && numel > 0) {
        auto device_sorted_unique_impl = [&](const auto* in_ptr) {
            using T = std::remove_const_t<std::remove_pointer_t<decltype(in_ptr)>>;

            // Step 1: Sort input on device using existing sort_kernel (dim=0, ascending)
            Tensor flat_input = input.reshape({numel});
            auto [sorted_vals, sorted_indices] = sort_kernel(flat_input, /*dim=*/0, /*descending=*/false, queue);
            const T* d_sorted = get_data_ptr<const T>(sorted_vals);

            // Step 2: Mark boundaries — flags[i] = 1 if sorted[i] != sorted[i-1], flags[0] = 1
            int32_t* d_flags = sycl::malloc_device<int32_t>(numel, queue);
            queue.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> gid) {
                int64_t i = gid[0];
                d_flags[i] = (i == 0 || d_sorted[i] != d_sorted[i - 1]) ? 1 : 0;
            }).wait();

            // Step 3: Inclusive prefix sum on flags to get group IDs (1-based)
            // Use sequential single_task scan — flags array is size numel, but this
            // avoids any host roundtrip. For very large inputs, the oneDPL path above
            // should be preferred.
            int32_t* d_group_ids = sycl::malloc_device<int32_t>(numel, queue);
            queue.single_task([=]() {
                int32_t running = 0;
                for (int64_t i = 0; i < numel; ++i) {
                    running += d_flags[i];
                    d_group_ids[i] = running;
                }
            }).wait();

            // Step 4: Read n_unique from last element (single scalar D2H)
            int32_t n_unique_i32 = 0;
            queue.memcpy(&n_unique_i32, d_group_ids + numel - 1, sizeof(int32_t)).wait();
            int64_t n_unique = static_cast<int64_t>(n_unique_i32);

            // Step 5: Compact unique values — scatter first element of each group
            Tensor out_vals({n_unique}, input.dtype(), input.device());
            T* d_unique = get_data_ptr<T>(out_vals);
            queue.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> gid) {
                int64_t i = gid[0];
                if (d_flags[i]) {
                    // group_ids are 1-based, so output index is group_ids[i] - 1
                    d_unique[d_group_ids[i] - 1] = d_sorted[i];
                }
            }).wait();

            // Step 6: Build inverse indices — for each original element, binary search
            // in the unique values to find its group index
            Tensor out_inverse({numel}, DType::Int64, input.device());
            if (return_inverse) {
                int64_t* inv_ptr = get_data_ptr<int64_t>(out_inverse);
                const T* unique_ptr = get_data_ptr<const T>(out_vals);
                queue.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> gid) {
                    T val = in_ptr[gid];
                    int64_t lo = 0, hi = n_unique;
                    while (lo < hi) {
                        int64_t mid = lo + (hi - lo) / 2;
                        if (unique_ptr[mid] < val) {
                            lo = mid + 1;
                        } else {
                            hi = mid;
                        }
                    }
                    inv_ptr[gid] = (lo < n_unique) ? lo : (n_unique - 1);
                }).wait();
            }

            // Step 7: Count per-group using atomics (same pattern as oneDPL path)
            Tensor out_counts({return_counts ? n_unique : 0}, DType::Int64, input.device());
            if (return_counts && n_unique > 0) {
                int64_t* cnt_ptr = get_data_ptr<int64_t>(out_counts);

                // Use int32 atomics (widely supported), then widen
                int32_t* d_counts32 = sycl::malloc_device<int32_t>(n_unique, queue);
                queue.memset(d_counts32, 0, n_unique * sizeof(int32_t)).wait();

                // We can count directly from group_ids (0-based = group_ids[i] - 1)
                queue.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> gid) {
                    int32_t group = d_group_ids[gid] - 1;
                    sycl::atomic_ref<int32_t, sycl::memory_order::relaxed,
                                     sycl::memory_scope::device,
                                     sycl::access::address_space::global_space> aref(d_counts32[group]);
                    aref.fetch_add(1);
                }).wait();

                // Widen int32 counts to int64
                queue.parallel_for(sycl::range<1>(n_unique), [=](sycl::id<1> i) {
                    cnt_ptr[i] = static_cast<int64_t>(d_counts32[i]);
                }).wait();

                sycl::free(d_counts32, queue);
            }

            sycl::free(d_flags, queue);
            sycl::free(d_group_ids, queue);
            return std::make_tuple(out_vals, out_inverse, out_counts);
        };

        if (input.dtype() == DType::Float32) {
            return device_sorted_unique_impl(get_data_ptr<const float>(input));
        } else if (input.dtype() == DType::Int64) {
            return device_sorted_unique_impl(get_data_ptr<const int64_t>(input));
        } else if (input.dtype() == DType::Float64) {
            return device_sorted_unique_impl(get_data_ptr<const double>(input));
        } else if (input.dtype() == DType::Int32) {
            return device_sorted_unique_impl(get_data_ptr<const int32_t>(input));
        } else {
            throw std::runtime_error("unique: unsupported dtype (supported: Float32, Float64, Int32, Int64)");
        }
    }

    // Without oneDPL, device-side unique with insertion-order preservation
    // is not supported. Require oneDPL for this operation.
    if (!sorted) {
        throw std::runtime_error(
            "unique(sorted=false) requires oneDPL for device-side execution. "
            "Build with -DTENZOR_HAS_ONEDPL=ON or use unique(sorted=true).");
    }

    // sorted=true with numel==0 — return empty tensors
    return std::make_tuple(
        Tensor({0}, input.dtype(), input.device()),
        Tensor({numel}, DType::Int64, input.device()),
        Tensor({0}, DType::Int64, input.device()));
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
                m = nan_fmax(m, val);
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
                m = nan_fmax(m, val);
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
                m = nan_fmax(m, val);
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
                m = nan_fmax(m, val);
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

// ============================================================================
// Median kernel — sort-based per-slice reduction
// ============================================================================

auto median_kernel(const Tensor& input, int64_t dim, bool keepdim, sycl::queue& queue)
    -> std::vector<Tensor>
{
    const auto dtype = input.dtype();
    const auto& device = input.device();
    const auto& input_shape = input.shape();
    const int64_t ndim = static_cast<int64_t>(input_shape.size());

    if (input.numel() == 0) {
        throw std::runtime_error("median: cannot compute median of empty tensor");
    }

    // Handle full reduction
    int64_t normalized_dim = dim;
    if (normalized_dim == INT64_MIN) {
        Tensor flat = input.contiguous().reshape({input.numel()});
        auto result = median_kernel(flat, 0, false, queue);
        if (keepdim) {
            std::vector<int64_t> kshape(ndim, 1);
            result[0] = result[0].reshape(kshape);
            result[1] = result[1].reshape(kshape);
        }
        return result;
    }

    if (normalized_dim < 0) normalized_dim += ndim;

    Tensor in_cont = input.is_contiguous() ? input : contiguous_kernel(input, queue);

    std::vector<int64_t> shape_vec(input_shape.begin(), input_shape.end());
    auto output_shape = compute_reduction_shape(shape_vec, normalized_dim, keepdim);

    int64_t dim_size = input_shape[normalized_dim];
    int64_t mid = (dim_size - 1) / 2;

    int64_t outer_size = 1;
    for (int64_t d = 0; d < ndim; ++d) {
        if (d != normalized_dim) outer_size *= input_shape[d];
    }

    // Compute strides
    std::vector<int64_t> in_strides(ndim);
    {
        int64_t stride = 1;
        for (int64_t d = ndim - 1; d >= 0; --d) {
            in_strides[d] = stride;
            stride *= input_shape[d];
        }
    }

    Tensor values(output_shape, dtype, device);
    Tensor indices(output_shape, DType::Int64, device);

    // Handle FP16/BF16 by converting to FP32
    if (dtype == DType::Float16 || dtype == DType::BFloat16) {
        auto input_f32 = input.to(DType::Float32);
        auto result = median_kernel(input_f32, dim, keepdim, queue);
        result[0] = result[0].to(dtype);
        return result;
    }

    auto process = [&](auto* type_ptr) {
        using T = std::remove_pointer_t<decltype(type_ptr)>;
        const T* in_ptr = in_cont.data<T>();
        T* val_ptr = values.data<T>();
        int64_t* idx_ptr = indices.data<int64_t>();

        // Copy strides and shape to device
        auto* d_strides = sycl::malloc_device<int64_t>(ndim, queue);
        auto* d_shape = sycl::malloc_device<int64_t>(ndim, queue);
        queue.memcpy(d_strides, in_strides.data(), ndim * sizeof(int64_t));
        queue.memcpy(d_shape, shape_vec.data(), ndim * sizeof(int64_t));
        queue.wait_and_throw();

        auto event = queue.parallel_for(sycl::range<1>(outer_size), [=](sycl::id<1> id) {
            int64_t o = id[0];

            // Compute base offset for this slice
            int64_t tmp = o;
            int64_t base_offset = 0;
            for (int64_t d = ndim - 1; d >= 0; --d) {
                if (d == normalized_dim) continue;
                int64_t coord = tmp % d_shape[d];
                tmp /= d_shape[d];
                base_offset += coord * d_strides[d];
            }

            // Selection-based median: count rank of each element
            for (int64_t i = 0; i < dim_size; ++i) {
                T val_i = in_ptr[base_offset + i * d_strides[normalized_dim]];
                int64_t rank = 0;
                for (int64_t j = 0; j < dim_size; ++j) {
                    T val_j = in_ptr[base_offset + j * d_strides[normalized_dim]];
                    if (val_j < val_i || (val_j == val_i && j < i)) {
                        rank++;
                    }
                }
                if (rank == mid) {
                    val_ptr[o] = val_i;
                    idx_ptr[o] = i;
                    break;
                }
            }
        });
        event.wait();

        sycl::free(d_strides, queue);
        sycl::free(d_shape, queue);
    };

    switch (dtype) {
        case DType::Float32: process(static_cast<float*>(nullptr)); break;
        case DType::Float64: process(static_cast<double*>(nullptr)); break;
        case DType::Int32:   process(static_cast<int32_t*>(nullptr)); break;
        case DType::Int64:   process(static_cast<int64_t*>(nullptr)); break;
        default: throw std::runtime_error("median_kernel: unsupported dtype");
    }

    return {values, indices};
}

// ============================================================================
// Mode kernel — count-based per-slice reduction
// ============================================================================

auto mode_kernel(const Tensor& input, int64_t dim, bool keepdim, sycl::queue& queue)
    -> std::vector<Tensor>
{
    const auto dtype = input.dtype();
    const auto& device = input.device();
    const auto& input_shape = input.shape();
    const int64_t ndim = static_cast<int64_t>(input_shape.size());

    if (input.numel() == 0) {
        throw std::runtime_error("mode: cannot compute mode of empty tensor");
    }

    int64_t normalized_dim = dim;
    if (normalized_dim == INT64_MIN) {
        Tensor flat = input.contiguous().reshape({input.numel()});
        auto result = mode_kernel(flat, 0, false, queue);
        if (keepdim) {
            std::vector<int64_t> kshape(ndim, 1);
            result[0] = result[0].reshape(kshape);
            result[1] = result[1].reshape(kshape);
        }
        return result;
    }

    if (normalized_dim < 0) normalized_dim += ndim;

    Tensor in_cont = input.is_contiguous() ? input : contiguous_kernel(input, queue);

    std::vector<int64_t> shape_vec(input_shape.begin(), input_shape.end());
    auto output_shape = compute_reduction_shape(shape_vec, normalized_dim, keepdim);

    int64_t dim_size = input_shape[normalized_dim];
    int64_t outer_size = 1;
    for (int64_t d = 0; d < ndim; ++d) {
        if (d != normalized_dim) outer_size *= input_shape[d];
    }

    std::vector<int64_t> in_strides(ndim);
    {
        int64_t stride = 1;
        for (int64_t d = ndim - 1; d >= 0; --d) {
            in_strides[d] = stride;
            stride *= input_shape[d];
        }
    }

    Tensor values(output_shape, dtype, device);
    Tensor indices(output_shape, DType::Int64, device);

    if (dtype == DType::Float16 || dtype == DType::BFloat16) {
        auto input_f32 = input.to(DType::Float32);
        auto result = mode_kernel(input_f32, dim, keepdim, queue);
        result[0] = result[0].to(dtype);
        return result;
    }

    auto process = [&](auto* type_ptr) {
        using T = std::remove_pointer_t<decltype(type_ptr)>;
        const T* in_ptr = in_cont.data<T>();
        T* val_ptr = values.data<T>();
        int64_t* idx_ptr = indices.data<int64_t>();

        auto* d_strides = sycl::malloc_device<int64_t>(ndim, queue);
        auto* d_shape = sycl::malloc_device<int64_t>(ndim, queue);
        queue.memcpy(d_strides, in_strides.data(), ndim * sizeof(int64_t));
        queue.memcpy(d_shape, shape_vec.data(), ndim * sizeof(int64_t));
        queue.wait_and_throw();

        auto event = queue.parallel_for(sycl::range<1>(outer_size), [=](sycl::id<1> id) {
            int64_t o = id[0];

            int64_t tmp = o;
            int64_t base_offset = 0;
            for (int64_t d = ndim - 1; d >= 0; --d) {
                if (d == normalized_dim) continue;
                int64_t coord = tmp % d_shape[d];
                tmp /= d_shape[d];
                base_offset += coord * d_strides[d];
            }

            // For each element, count how many times its value appears
            T best_val = in_ptr[base_offset];
            int64_t best_idx = 0;
            int64_t best_count = 0;

            for (int64_t i = 0; i < dim_size; ++i) {
                T val_i = in_ptr[base_offset + i * d_strides[normalized_dim]];
                int64_t count = 0;
                for (int64_t j = 0; j < dim_size; ++j) {
                    if (in_ptr[base_offset + j * d_strides[normalized_dim]] == val_i) {
                        count++;
                    }
                }
                if (count > best_count || (count == best_count && i < best_idx)) {
                    best_count = count;
                    best_val = val_i;
                    best_idx = i;
                }
            }

            val_ptr[o] = best_val;
            idx_ptr[o] = best_idx;
        });
        event.wait();

        sycl::free(d_strides, queue);
        sycl::free(d_shape, queue);
    };

    switch (dtype) {
        case DType::Float32: process(static_cast<float*>(nullptr)); break;
        case DType::Float64: process(static_cast<double*>(nullptr)); break;
        case DType::Int32:   process(static_cast<int32_t*>(nullptr)); break;
        case DType::Int64:   process(static_cast<int64_t*>(nullptr)); break;
        default: throw std::runtime_error("mode_kernel: unsupported dtype");
    }

    return {values, indices};
}

// ============================================================================
// CountNonzero kernel — full and dimensional reduction
// ============================================================================
class CountNonzeroKernelFloat32;
class CountNonzeroKernelFloat64;
class CountNonzeroKernelFloat16;
class CountNonzeroKernelBFloat16;
class CountNonzeroKernelInt32;
class CountNonzeroKernelInt64;
class CountNonzeroDimFloat32;
class CountNonzeroDimFloat64;
class CountNonzeroDimFloat16;
class CountNonzeroDimBFloat16;
class CountNonzeroDimInt32;
class CountNonzeroDimInt64;

auto count_nonzero_kernel(const Tensor& input, int64_t dim, sycl::queue& queue) -> Tensor {
    Tensor in_cont = input.is_contiguous() ? input : contiguous_kernel(input, queue);
    auto shape = in_cont.shape();
    const int64_t ndim = static_cast<int64_t>(shape.size());
    const int64_t total = in_cont.numel();

    // Full reduction (dim < 0 or not specified)
    if (dim < 0) {
        auto count_buf = sycl::malloc_shared<int64_t>(1, queue);
        count_buf[0] = 0;

        auto launch_full = [&]<typename T, typename KernelName>() {
            const T* in_ptr = get_data_ptr<const T>(in_cont);
            queue.parallel_for<KernelName>(sycl::range<1>(total),
                sycl::reduction(count_buf, sycl::plus<int64_t>()),
                [=](sycl::id<1> idx, auto& cnt) {
                    if (in_ptr[idx] != static_cast<T>(0)) cnt.combine(int64_t(1));
                }).wait();
        };

        auto launch_full_bf16 = [&]() {
            const uint16_t* in_ptr = get_data_ptr<const uint16_t>(in_cont);
            queue.parallel_for<CountNonzeroKernelBFloat16>(sycl::range<1>(total),
                sycl::reduction(count_buf, sycl::plus<int64_t>()),
                [=](sycl::id<1> idx, auto& cnt) {
                    if (bf16_to_f32(in_ptr[idx]) != 0.0f) cnt.combine(int64_t(1));
                }).wait();
        };

        auto launch_full_fp16 = [&]() {
            const sycl::half* in_ptr = get_data_ptr<const sycl::half>(in_cont);
            queue.parallel_for<CountNonzeroKernelFloat16>(sycl::range<1>(total),
                sycl::reduction(count_buf, sycl::plus<int64_t>()),
                [=](sycl::id<1> idx, auto& cnt) {
                    if (static_cast<float>(in_ptr[idx]) != 0.0f) cnt.combine(int64_t(1));
                }).wait();
        };

        switch (in_cont.dtype()) {
            case DType::Float32: launch_full.template operator()<float, CountNonzeroKernelFloat32>(); break;
            case DType::Float64: launch_full.template operator()<double, CountNonzeroKernelFloat64>(); break;
            case DType::Float16: launch_full_fp16(); break;
            case DType::BFloat16: launch_full_bf16(); break;
            case DType::Int32: launch_full.template operator()<int32_t, CountNonzeroKernelInt32>(); break;
            case DType::Int64: launch_full.template operator()<int64_t, CountNonzeroKernelInt64>(); break;
            default: throw std::runtime_error("count_nonzero_kernel: unsupported dtype");
        }

        Tensor result({1}, DType::Int64, in_cont.device());
        int64_t* out_ptr = get_data_ptr<int64_t>(result);
        queue.single_task([=]() { out_ptr[0] = count_buf[0]; }).wait();
        sycl::free(count_buf, queue);
        return result;
    }

    // Dimensional reduction
    int64_t norm_dim = dim < 0 ? dim + ndim : dim;
    if (norm_dim < 0 || norm_dim >= ndim) {
        throw std::runtime_error("count_nonzero: dim out of range");
    }

    int64_t reduce_size = shape[norm_dim];
    int64_t outer = 1, inner = 1;
    for (int64_t d = 0; d < norm_dim; d++) outer *= shape[d];
    for (int64_t d = norm_dim + 1; d < ndim; d++) inner *= shape[d];
    int64_t out_n = outer * inner;

    std::vector<int64_t> out_shape;
    for (int64_t d = 0; d < ndim; d++) {
        if (d != norm_dim) out_shape.push_back(shape[d]);
    }
    if (out_shape.empty()) out_shape.push_back(1);

    Tensor result(out_shape, DType::Int64, in_cont.device());
    int64_t* out_ptr = get_data_ptr<int64_t>(result);

    auto launch_dim = [&]<typename T, typename KernelName>() {
        const T* in_ptr = get_data_ptr<const T>(in_cont);
        queue.parallel_for<KernelName>(sycl::range<1>(out_n),
            [=](sycl::id<1> id) {
                int64_t idx = id[0];
                int64_t o = idx / inner;
                int64_t i_inner = idx % inner;
                int64_t count = 0;
                for (int64_t r = 0; r < reduce_size; r++) {
                    int64_t src = (o * reduce_size + r) * inner + i_inner;
                    if (in_ptr[src] != static_cast<T>(0)) count++;
                }
                out_ptr[idx] = count;
            }).wait();
    };

    auto launch_dim_bf16 = [&]() {
        const uint16_t* in_ptr = get_data_ptr<const uint16_t>(in_cont);
        queue.parallel_for<CountNonzeroDimBFloat16>(sycl::range<1>(out_n),
            [=](sycl::id<1> id) {
                int64_t idx = id[0];
                int64_t o = idx / inner;
                int64_t i_inner = idx % inner;
                int64_t count = 0;
                for (int64_t r = 0; r < reduce_size; r++) {
                    int64_t src = (o * reduce_size + r) * inner + i_inner;
                    if (bf16_to_f32(in_ptr[src]) != 0.0f) count++;
                }
                out_ptr[idx] = count;
            }).wait();
    };

    auto launch_dim_fp16 = [&]() {
        const sycl::half* in_ptr = get_data_ptr<const sycl::half>(in_cont);
        queue.parallel_for<CountNonzeroDimFloat16>(sycl::range<1>(out_n),
            [=](sycl::id<1> id) {
                int64_t idx = id[0];
                int64_t o = idx / inner;
                int64_t i_inner = idx % inner;
                int64_t count = 0;
                for (int64_t r = 0; r < reduce_size; r++) {
                    int64_t src = (o * reduce_size + r) * inner + i_inner;
                    if (static_cast<float>(in_ptr[src]) != 0.0f) count++;
                }
                out_ptr[idx] = count;
            }).wait();
    };

    switch (in_cont.dtype()) {
        case DType::Float32: launch_dim.template operator()<float, CountNonzeroDimFloat32>(); break;
        case DType::Float64: launch_dim.template operator()<double, CountNonzeroDimFloat64>(); break;
        case DType::Float16: launch_dim_fp16(); break;
        case DType::BFloat16: launch_dim_bf16(); break;
        case DType::Int32: launch_dim.template operator()<int32_t, CountNonzeroDimInt32>(); break;
        case DType::Int64: launch_dim.template operator()<int64_t, CountNonzeroDimInt64>(); break;
        default: throw std::runtime_error("count_nonzero_kernel: unsupported dtype");
    }

    return result;
}

// ============================================================================
// Nansum kernel — sum skipping NaN, full and dimensional reduction
// ============================================================================
class NansumKernelFloat32;
class NansumKernelFloat64;
class NansumKernelFloat16;
class NansumKernelBFloat16;
class NansumDimFloat32;
class NansumDimFloat64;
class NansumDimFloat16;
class NansumDimBFloat16;

auto nansum_kernel(const Tensor& input, int64_t dim, bool keepdim, sycl::queue& queue) -> Tensor {
    // Upcast Float16/BFloat16 to Float32 for computation, then downcast
    if (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16) {
        auto f32 = cast_kernel(input, DType::Float32, queue);
        auto result = nansum_kernel(f32, dim, keepdim, queue);
        return cast_kernel(result, input.dtype(), queue);
    }

    Tensor in_cont = input.is_contiguous() ? input : contiguous_kernel(input, queue);
    auto shape = in_cont.shape();
    const int64_t ndim = static_cast<int64_t>(shape.size());
    const int64_t total = in_cont.numel();

    bool is_full = (dim < 0);

    if (is_full) {
        Tensor output({1}, in_cont.dtype(), in_cont.device());

        if (in_cont.dtype() == DType::Float32) {
            const float* in_ptr = get_data_ptr<const float>(in_cont);
            float* out_ptr = get_data_ptr<float>(output);
            auto acc_buf = sycl::malloc_shared<float>(1, queue);
            acc_buf[0] = 0.0f;
            queue.parallel_for<NansumKernelFloat32>(sycl::range<1>(total),
                sycl::reduction(acc_buf, sycl::plus<float>()),
                [=](sycl::id<1> idx, auto& sum) {
                    float v = in_ptr[idx];
                    if (!sycl::isnan(v)) sum.combine(v);
                }).wait();
            out_ptr[0] = acc_buf[0];
            sycl::free(acc_buf, queue);
        } else if (in_cont.dtype() == DType::Float64) {
            const double* in_ptr = get_data_ptr<const double>(in_cont);
            double* out_ptr = get_data_ptr<double>(output);
            auto acc_buf = sycl::malloc_shared<double>(1, queue);
            acc_buf[0] = 0.0;
            queue.parallel_for<NansumKernelFloat64>(sycl::range<1>(total),
                sycl::reduction(acc_buf, sycl::plus<double>()),
                [=](sycl::id<1> idx, auto& sum) {
                    double v = in_ptr[idx];
                    if (!sycl::isnan(v)) sum.combine(v);
                }).wait();
            out_ptr[0] = acc_buf[0];
            sycl::free(acc_buf, queue);
        } else {
            throw std::runtime_error("nansum_kernel: unsupported dtype (expected floating type)");
        }

        return output;
    }

    // Dimensional reduction
    int64_t norm_dim = dim < 0 ? dim + ndim : dim;
    if (norm_dim < 0 || norm_dim >= ndim) {
        throw std::runtime_error("nansum: dim out of range");
    }

    int64_t reduce_size = shape[norm_dim];
    int64_t outer = 1, inner = 1;
    for (int64_t d = 0; d < norm_dim; d++) outer *= shape[d];
    for (int64_t d = norm_dim + 1; d < ndim; d++) inner *= shape[d];

    auto out_shape = compute_reduction_shape(
        std::vector<int64_t>(shape.begin(), shape.end()), norm_dim, keepdim);
    Tensor output(out_shape, in_cont.dtype(), in_cont.device());
    int64_t out_n = outer * inner;

    if (in_cont.dtype() == DType::Float32) {
        const float* in_ptr = get_data_ptr<const float>(in_cont);
        float* out_ptr = get_data_ptr<float>(output);
        queue.parallel_for<NansumDimFloat32>(sycl::range<1>(out_n),
            [=](sycl::id<1> id) {
                int64_t idx = id[0];
                int64_t o = idx / inner;
                int64_t i_inner = idx % inner;
                float acc = 0.0f;
                for (int64_t r = 0; r < reduce_size; r++) {
                    int64_t src = (o * reduce_size + r) * inner + i_inner;
                    float v = in_ptr[src];
                    if (!sycl::isnan(v)) acc += v;
                }
                out_ptr[idx] = acc;
            }).wait();
    } else if (in_cont.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(in_cont);
        double* out_ptr = get_data_ptr<double>(output);
        queue.parallel_for<NansumDimFloat64>(sycl::range<1>(out_n),
            [=](sycl::id<1> id) {
                int64_t idx = id[0];
                int64_t o = idx / inner;
                int64_t i_inner = idx % inner;
                double acc = 0.0;
                for (int64_t r = 0; r < reduce_size; r++) {
                    int64_t src = (o * reduce_size + r) * inner + i_inner;
                    double v = in_ptr[src];
                    if (!sycl::isnan(v)) acc += v;
                }
                out_ptr[idx] = acc;
            }).wait();
    } else {
        throw std::runtime_error("nansum_kernel: unsupported dtype (expected floating type)");
    }

    return output;
}

// ============================================================================
// Nanmean kernel — nansum / count_non_nan
// ============================================================================
class NanmeanKernelFloat32Sum;
class NanmeanKernelFloat32Count;
class NanmeanKernelFloat64Sum;
class NanmeanKernelFloat64Count;
class NanmeanDimFloat32;
class NanmeanDimFloat64;

auto nanmean_kernel(const Tensor& input, int64_t dim, bool keepdim, sycl::queue& queue) -> Tensor {
    if (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16) {
        auto f32 = cast_kernel(input, DType::Float32, queue);
        auto result = nanmean_kernel(f32, dim, keepdim, queue);
        return cast_kernel(result, input.dtype(), queue);
    }

    Tensor in_cont = input.is_contiguous() ? input : contiguous_kernel(input, queue);
    auto shape = in_cont.shape();
    const int64_t ndim = static_cast<int64_t>(shape.size());
    const int64_t total = in_cont.numel();

    bool is_full = (dim < 0);

    if (is_full) {
        Tensor output({1}, in_cont.dtype(), in_cont.device());

        if (in_cont.dtype() == DType::Float32) {
            const float* in_ptr = get_data_ptr<const float>(in_cont);
            float* out_ptr = get_data_ptr<float>(output);

            auto sum_buf = sycl::malloc_shared<float>(1, queue);
            auto cnt_buf = sycl::malloc_shared<int64_t>(1, queue);
            sum_buf[0] = 0.0f;
            cnt_buf[0] = 0;

            queue.parallel_for<NanmeanKernelFloat32Sum>(sycl::range<1>(total),
                sycl::reduction(sum_buf, sycl::plus<float>()),
                [=](sycl::id<1> idx, auto& sum) {
                    float v = in_ptr[idx];
                    if (!sycl::isnan(v)) sum.combine(v);
                }).wait();

            queue.parallel_for<NanmeanKernelFloat32Count>(sycl::range<1>(total),
                sycl::reduction(cnt_buf, sycl::plus<int64_t>()),
                [=](sycl::id<1> idx, auto& cnt) {
                    float v = in_ptr[idx];
                    if (!sycl::isnan(v)) cnt.combine(int64_t(1));
                }).wait();

            out_ptr[0] = cnt_buf[0] > 0 ? sum_buf[0] / static_cast<float>(cnt_buf[0])
                                        : std::numeric_limits<float>::quiet_NaN();
            sycl::free(sum_buf, queue);
            sycl::free(cnt_buf, queue);
        } else if (in_cont.dtype() == DType::Float64) {
            const double* in_ptr = get_data_ptr<const double>(in_cont);
            double* out_ptr = get_data_ptr<double>(output);

            auto sum_buf = sycl::malloc_shared<double>(1, queue);
            auto cnt_buf = sycl::malloc_shared<int64_t>(1, queue);
            sum_buf[0] = 0.0;
            cnt_buf[0] = 0;

            queue.parallel_for<NanmeanKernelFloat64Sum>(sycl::range<1>(total),
                sycl::reduction(sum_buf, sycl::plus<double>()),
                [=](sycl::id<1> idx, auto& sum) {
                    double v = in_ptr[idx];
                    if (!sycl::isnan(v)) sum.combine(v);
                }).wait();

            queue.parallel_for<NanmeanKernelFloat64Count>(sycl::range<1>(total),
                sycl::reduction(cnt_buf, sycl::plus<int64_t>()),
                [=](sycl::id<1> idx, auto& cnt) {
                    double v = in_ptr[idx];
                    if (!sycl::isnan(v)) cnt.combine(int64_t(1));
                }).wait();

            out_ptr[0] = cnt_buf[0] > 0 ? sum_buf[0] / static_cast<double>(cnt_buf[0])
                                        : std::numeric_limits<double>::quiet_NaN();
            sycl::free(sum_buf, queue);
            sycl::free(cnt_buf, queue);
        } else {
            throw std::runtime_error("nanmean_kernel: unsupported dtype (expected floating type)");
        }

        return output;
    }

    // Dimensional reduction
    int64_t norm_dim = dim < 0 ? dim + ndim : dim;
    if (norm_dim < 0 || norm_dim >= ndim) {
        throw std::runtime_error("nanmean: dim out of range");
    }

    int64_t reduce_size = shape[norm_dim];
    int64_t outer = 1, inner = 1;
    for (int64_t d = 0; d < norm_dim; d++) outer *= shape[d];
    for (int64_t d = norm_dim + 1; d < ndim; d++) inner *= shape[d];

    auto out_shape = compute_reduction_shape(
        std::vector<int64_t>(shape.begin(), shape.end()), norm_dim, keepdim);
    Tensor output(out_shape, in_cont.dtype(), in_cont.device());
    int64_t out_n = outer * inner;

    if (in_cont.dtype() == DType::Float32) {
        const float* in_ptr = get_data_ptr<const float>(in_cont);
        float* out_ptr = get_data_ptr<float>(output);
        queue.parallel_for<NanmeanDimFloat32>(sycl::range<1>(out_n),
            [=](sycl::id<1> id) {
                int64_t idx = id[0];
                int64_t o = idx / inner;
                int64_t i_inner = idx % inner;
                float acc = 0.0f;
                int64_t count = 0;
                for (int64_t r = 0; r < reduce_size; r++) {
                    int64_t src = (o * reduce_size + r) * inner + i_inner;
                    float v = in_ptr[src];
                    if (!sycl::isnan(v)) { acc += v; count++; }
                }
                out_ptr[idx] = count > 0 ? acc / static_cast<float>(count)
                                         : std::numeric_limits<float>::quiet_NaN();
            }).wait();
    } else if (in_cont.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(in_cont);
        double* out_ptr = get_data_ptr<double>(output);
        queue.parallel_for<NanmeanDimFloat64>(sycl::range<1>(out_n),
            [=](sycl::id<1> id) {
                int64_t idx = id[0];
                int64_t o = idx / inner;
                int64_t i_inner = idx % inner;
                double acc = 0.0;
                int64_t count = 0;
                for (int64_t r = 0; r < reduce_size; r++) {
                    int64_t src = (o * reduce_size + r) * inner + i_inner;
                    double v = in_ptr[src];
                    if (!sycl::isnan(v)) { acc += v; count++; }
                }
                out_ptr[idx] = count > 0 ? acc / static_cast<double>(count)
                                         : std::numeric_limits<double>::quiet_NaN();
            }).wait();
    } else {
        throw std::runtime_error("nanmean_kernel: unsupported dtype (expected floating type)");
    }

    return output;
}

// ============================================================================
// Aminmax kernel — simultaneous min+max, returns {min, max} tensors
// ============================================================================
class AminmaxKernelFloat32;
class AminmaxKernelFloat64;
class AminmaxDimFloat32;
class AminmaxDimFloat64;
class AminmaxDimFloat16;
class AminmaxDimBFloat16;

auto aminmax_kernel(const Tensor& input, int64_t dim, bool keepdim, sycl::queue& queue)
    -> std::vector<Tensor> {
    // Upcast Float16/BFloat16 for computation
    if (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16) {
        auto f32 = cast_kernel(input, DType::Float32, queue);
        auto results = aminmax_kernel(f32, dim, keepdim, queue);
        return {cast_kernel(results[0], input.dtype(), queue),
                cast_kernel(results[1], input.dtype(), queue)};
    }

    Tensor in_cont = input.is_contiguous() ? input : contiguous_kernel(input, queue);
    auto shape = in_cont.shape();
    const int64_t ndim = static_cast<int64_t>(shape.size());
    const int64_t total = in_cont.numel();

    bool is_full = (dim < 0);

    if (is_full) {
        Tensor min_out({1}, in_cont.dtype(), in_cont.device());
        Tensor max_out({1}, in_cont.dtype(), in_cont.device());

        // NaN must propagate (PyTorch / CPU contract). The built-in
        // sycl::minimum/maximum reduction operators drop NaNs, so use a
        // single_task sequential pass with nan_fmin/nan_fmax, matching the
        // full-reduction path of max_kernel/min_kernel in this file.
        if (in_cont.dtype() == DType::Float32) {
            const float* in_ptr = get_data_ptr<const float>(in_cont);
            float* min_ptr = get_data_ptr<float>(min_out);
            float* max_ptr = get_data_ptr<float>(max_out);

            queue.single_task<AminmaxKernelFloat32>([=]() {
                float mn = std::numeric_limits<float>::infinity();
                float mx = -std::numeric_limits<float>::infinity();
                for (int64_t i = 0; i < total; ++i) {
                    float v = in_ptr[i];
                    mn = nan_fmin(mn, v);
                    mx = nan_fmax(mx, v);
                }
                min_ptr[0] = mn;
                max_ptr[0] = mx;
            }).wait();
        } else if (in_cont.dtype() == DType::Float64) {
            const double* in_ptr = get_data_ptr<const double>(in_cont);
            double* min_ptr = get_data_ptr<double>(min_out);
            double* max_ptr = get_data_ptr<double>(max_out);

            queue.single_task<AminmaxKernelFloat64>([=]() {
                double mn = std::numeric_limits<double>::infinity();
                double mx = -std::numeric_limits<double>::infinity();
                for (int64_t i = 0; i < total; ++i) {
                    double v = in_ptr[i];
                    mn = nan_fmin(mn, v);
                    mx = nan_fmax(mx, v);
                }
                min_ptr[0] = mn;
                max_ptr[0] = mx;
            }).wait();
        } else {
            throw std::runtime_error("aminmax_kernel: unsupported dtype (expected floating type)");
        }

        return {min_out, max_out};
    }

    // Dimensional reduction
    int64_t norm_dim = dim < 0 ? dim + ndim : dim;
    if (norm_dim < 0 || norm_dim >= ndim) {
        throw std::runtime_error("aminmax: dim out of range");
    }

    int64_t reduce_size = shape[norm_dim];
    int64_t outer = 1, inner = 1;
    for (int64_t d = 0; d < norm_dim; d++) outer *= shape[d];
    for (int64_t d = norm_dim + 1; d < ndim; d++) inner *= shape[d];

    auto out_shape = compute_reduction_shape(
        std::vector<int64_t>(shape.begin(), shape.end()), norm_dim, keepdim);
    Tensor min_out(out_shape, in_cont.dtype(), in_cont.device());
    Tensor max_out(out_shape, in_cont.dtype(), in_cont.device());
    int64_t out_n = outer * inner;

    if (in_cont.dtype() == DType::Float32) {
        const float* in_ptr = get_data_ptr<const float>(in_cont);
        float* min_ptr = get_data_ptr<float>(min_out);
        float* max_ptr = get_data_ptr<float>(max_out);
        queue.parallel_for<AminmaxDimFloat32>(sycl::range<1>(out_n),
            [=](sycl::id<1> id) {
                int64_t idx = id[0];
                int64_t o = idx / inner;
                int64_t i_inner = idx % inner;
                int64_t base = (o * reduce_size) * inner + i_inner;
                float mn = in_ptr[base], mx = in_ptr[base];
                for (int64_t r = 1; r < reduce_size; r++) {
                    float v = in_ptr[base + r * inner];
                    mn = nan_fmin(mn, v);
                    mx = nan_fmax(mx, v);
                }
                min_ptr[idx] = mn;
                max_ptr[idx] = mx;
            }).wait();
    } else if (in_cont.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(in_cont);
        double* min_ptr = get_data_ptr<double>(min_out);
        double* max_ptr = get_data_ptr<double>(max_out);
        queue.parallel_for<AminmaxDimFloat64>(sycl::range<1>(out_n),
            [=](sycl::id<1> id) {
                int64_t idx = id[0];
                int64_t o = idx / inner;
                int64_t i_inner = idx % inner;
                int64_t base = (o * reduce_size) * inner + i_inner;
                double mn = in_ptr[base], mx = in_ptr[base];
                for (int64_t r = 1; r < reduce_size; r++) {
                    double v = in_ptr[base + r * inner];
                    mn = nan_fmin(mn, v);
                    mx = nan_fmax(mx, v);
                }
                min_ptr[idx] = mn;
                max_ptr[idx] = mx;
            }).wait();
    } else {
        throw std::runtime_error("aminmax_kernel: unsupported dtype (expected floating type)");
    }

    return {min_out, max_out};
}

// ============================================================================
// NanVar kernel — variance ignoring NaN values
// Uses two-pass: first compute nanmean, then compute sum of squared deviations
// ============================================================================
class NanvarFullF32Sum;
class NanvarFullF32Count;
class NanvarFullF32Dev;
class NanvarFullF64Sum;
class NanvarFullF64Count;
class NanvarFullF64Dev;
class NanvarDimF32;
class NanvarDimF64;

auto nanvar_kernel(const Tensor& input, int64_t dim, bool keepdim, int64_t correction,
                   sycl::queue& queue) -> Tensor {
    if (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16) {
        auto f32 = cast_kernel(input, DType::Float32, queue);
        auto result = nanvar_kernel(f32, dim, keepdim, correction, queue);
        return cast_kernel(result, input.dtype(), queue);
    }

    Tensor in_cont = input.is_contiguous() ? input : contiguous_kernel(input, queue);
    auto shape = in_cont.shape();
    const int64_t ndim = static_cast<int64_t>(shape.size());
    const int64_t total = in_cont.numel();

    bool is_full = (dim < 0);

    if (is_full) {
        Tensor output({1}, in_cont.dtype(), in_cont.device());

        if (in_cont.dtype() == DType::Float32) {
            const float* in_ptr = get_data_ptr<const float>(in_cont);
            float* out_ptr = get_data_ptr<float>(output);

            // Pass 1: compute nanmean
            auto sum_buf = sycl::malloc_shared<float>(1, queue);
            auto cnt_buf = sycl::malloc_shared<int64_t>(1, queue);
            sum_buf[0] = 0.0f;
            cnt_buf[0] = 0;

            queue.parallel_for<NanvarFullF32Sum>(sycl::range<1>(total),
                sycl::reduction(sum_buf, sycl::plus<float>()),
                [=](sycl::id<1> idx, auto& sum) {
                    float v = in_ptr[idx];
                    if (!sycl::isnan(v)) sum.combine(v);
                }).wait();

            queue.parallel_for<NanvarFullF32Count>(sycl::range<1>(total),
                sycl::reduction(cnt_buf, sycl::plus<int64_t>()),
                [=](sycl::id<1> idx, auto& cnt) {
                    float v = in_ptr[idx];
                    if (!sycl::isnan(v)) cnt.combine(int64_t(1));
                }).wait();

            float mean_val = cnt_buf[0] > 0 ? sum_buf[0] / static_cast<float>(cnt_buf[0]) : 0.0f;
            int64_t count = cnt_buf[0];

            // Pass 2: sum of squared deviations
            auto dev_buf = sycl::malloc_shared<float>(1, queue);
            dev_buf[0] = 0.0f;

            queue.parallel_for<NanvarFullF32Dev>(sycl::range<1>(total),
                sycl::reduction(dev_buf, sycl::plus<float>()),
                [=](sycl::id<1> idx, auto& acc) {
                    float v = in_ptr[idx];
                    if (!sycl::isnan(v)) {
                        float d = v - mean_val;
                        acc.combine(d * d);
                    }
                }).wait();

            int64_t denom = count - correction;
            out_ptr[0] = denom > 0 ? dev_buf[0] / static_cast<float>(denom)
                                   : std::numeric_limits<float>::quiet_NaN();

            sycl::free(sum_buf, queue);
            sycl::free(cnt_buf, queue);
            sycl::free(dev_buf, queue);
        } else if (in_cont.dtype() == DType::Float64) {
            const double* in_ptr = get_data_ptr<const double>(in_cont);
            double* out_ptr = get_data_ptr<double>(output);

            auto sum_buf = sycl::malloc_shared<double>(1, queue);
            auto cnt_buf = sycl::malloc_shared<int64_t>(1, queue);
            sum_buf[0] = 0.0;
            cnt_buf[0] = 0;

            queue.parallel_for<NanvarFullF64Sum>(sycl::range<1>(total),
                sycl::reduction(sum_buf, sycl::plus<double>()),
                [=](sycl::id<1> idx, auto& sum) {
                    double v = in_ptr[idx];
                    if (!sycl::isnan(v)) sum.combine(v);
                }).wait();

            queue.parallel_for<NanvarFullF64Count>(sycl::range<1>(total),
                sycl::reduction(cnt_buf, sycl::plus<int64_t>()),
                [=](sycl::id<1> idx, auto& cnt) {
                    double v = in_ptr[idx];
                    if (!sycl::isnan(v)) cnt.combine(int64_t(1));
                }).wait();

            double mean_val = cnt_buf[0] > 0 ? sum_buf[0] / static_cast<double>(cnt_buf[0]) : 0.0;
            int64_t count = cnt_buf[0];

            auto dev_buf = sycl::malloc_shared<double>(1, queue);
            dev_buf[0] = 0.0;

            queue.parallel_for<NanvarFullF64Dev>(sycl::range<1>(total),
                sycl::reduction(dev_buf, sycl::plus<double>()),
                [=](sycl::id<1> idx, auto& acc) {
                    double v = in_ptr[idx];
                    if (!sycl::isnan(v)) {
                        double d = v - mean_val;
                        acc.combine(d * d);
                    }
                }).wait();

            int64_t denom = count - correction;
            out_ptr[0] = denom > 0 ? dev_buf[0] / static_cast<double>(denom)
                                   : std::numeric_limits<double>::quiet_NaN();

            sycl::free(sum_buf, queue);
            sycl::free(cnt_buf, queue);
            sycl::free(dev_buf, queue);
        } else {
            throw std::runtime_error("nanvar_kernel: unsupported dtype (expected floating type)");
        }

        return output;
    }

    // Dimensional reduction
    int64_t norm_dim = dim < 0 ? dim + ndim : dim;
    if (norm_dim < 0 || norm_dim >= ndim) {
        throw std::runtime_error("nanvar: dim out of range");
    }

    int64_t reduce_size = shape[norm_dim];
    int64_t outer = 1, inner = 1;
    for (int64_t d = 0; d < norm_dim; d++) outer *= shape[d];
    for (int64_t d = norm_dim + 1; d < ndim; d++) inner *= shape[d];

    auto out_shape = compute_reduction_shape(
        std::vector<int64_t>(shape.begin(), shape.end()), norm_dim, keepdim);
    Tensor output(out_shape, in_cont.dtype(), in_cont.device());
    int64_t out_n = outer * inner;

    if (in_cont.dtype() == DType::Float32) {
        const float* in_ptr = get_data_ptr<const float>(in_cont);
        float* out_ptr = get_data_ptr<float>(output);
        int64_t corr = correction;
        queue.parallel_for<NanvarDimF32>(sycl::range<1>(out_n),
            [=](sycl::id<1> id) {
                int64_t idx = id[0];
                int64_t o = idx / inner;
                int64_t i_inner = idx % inner;
                // Pass 1: mean
                float sum = 0.0f;
                int64_t count = 0;
                for (int64_t r = 0; r < reduce_size; r++) {
                    int64_t src = (o * reduce_size + r) * inner + i_inner;
                    float v = in_ptr[src];
                    if (!sycl::isnan(v)) { sum += v; count++; }
                }
                float mean = count > 0 ? sum / static_cast<float>(count) : 0.0f;
                // Pass 2: squared deviations
                float dev_sum = 0.0f;
                for (int64_t r = 0; r < reduce_size; r++) {
                    int64_t src = (o * reduce_size + r) * inner + i_inner;
                    float v = in_ptr[src];
                    if (!sycl::isnan(v)) {
                        float d = v - mean;
                        dev_sum += d * d;
                    }
                }
                int64_t denom = count - corr;
                out_ptr[idx] = denom > 0 ? dev_sum / static_cast<float>(denom)
                                         : std::numeric_limits<float>::quiet_NaN();
            }).wait();
    } else if (in_cont.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(in_cont);
        double* out_ptr = get_data_ptr<double>(output);
        int64_t corr = correction;
        queue.parallel_for<NanvarDimF64>(sycl::range<1>(out_n),
            [=](sycl::id<1> id) {
                int64_t idx = id[0];
                int64_t o = idx / inner;
                int64_t i_inner = idx % inner;
                double sum = 0.0;
                int64_t count = 0;
                for (int64_t r = 0; r < reduce_size; r++) {
                    int64_t src = (o * reduce_size + r) * inner + i_inner;
                    double v = in_ptr[src];
                    if (!sycl::isnan(v)) { sum += v; count++; }
                }
                double mean = count > 0 ? sum / static_cast<double>(count) : 0.0;
                double dev_sum = 0.0;
                for (int64_t r = 0; r < reduce_size; r++) {
                    int64_t src = (o * reduce_size + r) * inner + i_inner;
                    double v = in_ptr[src];
                    if (!sycl::isnan(v)) {
                        double d = v - mean;
                        dev_sum += d * d;
                    }
                }
                int64_t denom = count - corr;
                out_ptr[idx] = denom > 0 ? dev_sum / static_cast<double>(denom)
                                         : std::numeric_limits<double>::quiet_NaN();
            }).wait();
    } else {
        throw std::runtime_error("nanvar_kernel: unsupported dtype (expected floating type)");
    }

    return output;
}

// ============================================================================
// NanStd kernel — standard deviation ignoring NaN (sqrt of nanvar)
// ============================================================================
auto nanstd_kernel(const Tensor& input, int64_t dim, bool keepdim, int64_t correction,
                   sycl::queue& queue) -> Tensor {
    Tensor var = nanvar_kernel(input, dim, keepdim, correction, queue);

    // Take sqrt of the variance
    int64_t numel = var.numel();
    Tensor result(std::vector<int64_t>(var.shape().begin(), var.shape().end()),
                  var.dtype(), var.device());

    if (var.dtype() == DType::Float32) {
        const float* v_ptr = get_data_ptr<const float>(var);
        float* out_ptr = get_data_ptr<float>(result);
        queue.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::sqrt(v_ptr[idx]);
        }).wait();
    } else if (var.dtype() == DType::Float64) {
        const double* v_ptr = get_data_ptr<const double>(var);
        double* out_ptr = get_data_ptr<double>(result);
        queue.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::sqrt(v_ptr[idx]);
        }).wait();
    } else if (var.dtype() == DType::Float16) {
        const sycl::half* v_ptr = get_data_ptr<const sycl::half>(var);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(result);
        queue.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::half(sycl::sqrt(static_cast<float>(v_ptr[idx])));
        }).wait();
    } else if (var.dtype() == DType::BFloat16) {
        const uint16_t* v_ptr = get_data_ptr<const uint16_t>(var);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(result);
        queue.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = f32_to_bf16(sycl::sqrt(bf16_to_f32(v_ptr[idx])));
        }).wait();
    } else {
        throw std::runtime_error("nanstd_kernel: unsupported dtype");
    }

    return result;
}

} // namespace oneapi
} // namespace tenzor
