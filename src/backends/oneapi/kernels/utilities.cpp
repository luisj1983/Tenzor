#include "tenzor/core/tensor.hpp"
#include "oneapi_kernel_utils.hpp"
#include <sycl/sycl.hpp>
#include <stdexcept>
#include <algorithm>
#include <cmath>

namespace tenzor {
namespace oneapi {

// Templated SYCL kernel-name tags for integer clamp/sign (PyTorch integer
// support; matches the CPU/CUDA/ROCm integer paths for cross-backend parity).
template <typename T> class OneapiClampIntKernel;
template <typename T> class OneapiSignIntKernel;

// SYCL Kernel name classes
struct CatKernelFloat32 {};
struct CatKernelFloat64 {};
struct CatKernelFloat16 {};
struct CatKernelInt32 {};
struct CatKernelInt64 {};
struct CatKernelBFloat16 {};
struct CatKernelBool {};
struct ClampKernelFloat32 {};
struct ClampKernelFloat64 {};
struct ClampKernelFloat16 {};
struct ClampKernelBFloat16 {};
struct ClampKernelInt32 {};
struct SignKernelFloat32 {};
struct SignKernelFloat64 {};
struct SignKernelFloat16 {};
struct SignKernelBFloat16 {};
struct SignKernelInt32 {};



/**
 * @brief Concatenate tensors along a specified dimension
 *
 * @param tensors Span of input tensors to concatenate
 * @param dim Dimension along which to concatenate
 * @param queue SYCL queue for kernel execution
 * @return Concatenated output tensor
 *
 * The operation concatenates multiple tensors along the specified dimension.
 * All input tensors must have the same shape except in the concatenation dimension.
 *
 * Example:
 *   A = [[1, 2], [3, 4]]  shape: (2, 2)
 *   B = [[5, 6]]          shape: (1, 2)
 *   cat([A, B], dim=0) = [[1, 2], [3, 4], [5, 6]]  shape: (3, 2)
 */
auto cat_kernel(std::span<const Tensor> tensors, int64_t dim, sycl::queue& queue) -> Tensor {
    // Validate inputs
    if (tensors.empty()) {
        throw std::invalid_argument("cat_kernel requires at least one input tensor");
    }

    if (tensors.size() == 1) {
        // Single tensor, just clone it (materialize any view into contiguous storage)
        Tensor contig = tensors[0].contiguous();
        const void* in_ptr = contig.data_ptr();
        const size_t bytes = contig.numel() * contig.dtype_size();

        Tensor output(std::vector<int64_t>(contig.shape().begin(), contig.shape().end()),
                      contig.dtype(), contig.device());
        void* out_ptr = const_cast<void*>(output.data_ptr());
        queue.memcpy(out_ptr, in_ptr, bytes).wait();
        return output;
    }

    // Get reference shape and validate all tensors
    auto ref_shape_span = tensors[0].shape();
    std::vector<int64_t> ref_shape(ref_shape_span.begin(), ref_shape_span.end());
    const size_t ndim = ref_shape.size();
    const DType dtype = tensors[0].dtype();
    const Device device = tensors[0].device();

    // Normalize dimension
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= static_cast<int64_t>(ndim)) {
        throw std::invalid_argument("cat_kernel: invalid dimension " + std::to_string(dim));
    }

    // Validate all tensors have compatible shapes and same dtype
    for (size_t i = 1; i < tensors.size(); ++i) {
        auto shape_span = tensors[i].shape();
        if (shape_span.size() != ndim) {
            throw std::invalid_argument("cat_kernel: all tensors must have same number of dimensions");
        }
        if (tensors[i].dtype() != dtype) {
            throw std::invalid_argument("cat_kernel: all tensors must have same dtype");
        }

        // Check all dimensions except concat dimension match
        for (size_t d = 0; d < ndim; ++d) {
            if (d != static_cast<size_t>(dim) && shape_span[d] != ref_shape[d]) {
                throw std::invalid_argument("cat_kernel: tensor shapes must match except in concat dimension");
            }
        }
    }

    // Compute output shape
    std::vector<int64_t> out_shape = ref_shape;
    out_shape[dim] = 0;
    for (const auto& tensor : tensors) {
        out_shape[dim] += tensor.shape()[dim];
    }

    // Create output tensor
    Tensor output(out_shape, dtype, device);

    // Calculate strides for efficient indexing
    auto out_strides = calculate_strides(out_shape);

    // For each input tensor, copy its data to the appropriate offset in output.
    // The linear `src_idx` computation below assumes the source tensor is
    // contiguous — call `.contiguous()` to materialize any view (e.g. the
    // stride-0 result of `expand()` that ReplicationPad uses) into a real
    // contiguous buffer before reading with linear indexing.
    int64_t offset_in_concat_dim = 0;

    for (const auto& tensor_in : tensors) {
        Tensor tensor = tensor_in.contiguous();
        auto tensor_shape = tensor.shape();
        const int64_t tensor_size_in_dim = tensor_shape[dim];

        // Calculate how many elements to copy
        // Elements per "slice" in the concatenation dimension
        int64_t elements_per_slice = 1;
        for (size_t d = dim + 1; d < ndim; ++d) {
            elements_per_slice *= tensor_shape[d];
        }

        // Number of slices to copy
        int64_t num_slices = 1;
        for (size_t d = 0; d < static_cast<size_t>(dim); ++d) {
            num_slices *= tensor_shape[d];
        }

        const int64_t tensor_numel = tensor.numel();

        // Dispatch based on dtype
        if (dtype == DType::Float32) {
            const float* src_ptr = get_data_ptr<const float>(tensor);
            float* dst_ptr = get_data_ptr<float>(output);

            // Calculate stride sizes
            const int64_t src_slice_stride = elements_per_slice * tensor_size_in_dim;
            const int64_t dst_slice_stride = elements_per_slice * out_shape[dim];

            queue.parallel_for<CatKernelFloat32>(
                sycl::range<1>(num_slices * tensor_size_in_dim * elements_per_slice),
                [=](sycl::id<1> idx) {
                    // Decompose index into slice, concat_idx, and element
                    const int64_t flat_idx = idx[0];
                    const int64_t slice_idx = flat_idx / src_slice_stride;
                    const int64_t remainder = flat_idx % src_slice_stride;
                    const int64_t concat_idx = remainder / elements_per_slice;
                    const int64_t elem_idx = remainder % elements_per_slice;

                    // Source index
                    const int64_t src_idx = slice_idx * src_slice_stride + concat_idx * elements_per_slice + elem_idx;

                    // Destination index
                    const int64_t dst_idx = slice_idx * dst_slice_stride +
                                           (offset_in_concat_dim + concat_idx) * elements_per_slice +
                                           elem_idx;

                    dst_ptr[dst_idx] = src_ptr[src_idx];
                }
            );
        }
        else if (dtype == DType::Float64) {
            const double* src_ptr = get_data_ptr<const double>(tensor);
            double* dst_ptr = get_data_ptr<double>(output);

            const int64_t src_slice_stride = elements_per_slice * tensor_size_in_dim;
            const int64_t dst_slice_stride = elements_per_slice * out_shape[dim];

            queue.parallel_for<CatKernelFloat64>(
                sycl::range<1>(num_slices * tensor_size_in_dim * elements_per_slice),
                [=](sycl::id<1> idx) {
                    const int64_t flat_idx = idx[0];
                    const int64_t slice_idx = flat_idx / src_slice_stride;
                    const int64_t remainder = flat_idx % src_slice_stride;
                    const int64_t concat_idx = remainder / elements_per_slice;
                    const int64_t elem_idx = remainder % elements_per_slice;

                    const int64_t src_idx = slice_idx * src_slice_stride + concat_idx * elements_per_slice + elem_idx;
                    const int64_t dst_idx = slice_idx * dst_slice_stride +
                                           (offset_in_concat_dim + concat_idx) * elements_per_slice +
                                           elem_idx;

                    dst_ptr[dst_idx] = src_ptr[src_idx];
                }
            );
        }
        else if (dtype == DType::Float16) {
            const sycl::half* src_ptr = get_data_ptr<const sycl::half>(tensor);
            sycl::half* dst_ptr = get_data_ptr<sycl::half>(output);

            const int64_t src_slice_stride = elements_per_slice * tensor_size_in_dim;
            const int64_t dst_slice_stride = elements_per_slice * out_shape[dim];

            queue.parallel_for<CatKernelFloat16>(
                sycl::range<1>(num_slices * tensor_size_in_dim * elements_per_slice),
                [=](sycl::id<1> idx) {
                    const int64_t flat_idx = idx[0];
                    const int64_t slice_idx = flat_idx / src_slice_stride;
                    const int64_t remainder = flat_idx % src_slice_stride;
                    const int64_t concat_idx = remainder / elements_per_slice;
                    const int64_t elem_idx = remainder % elements_per_slice;

                    const int64_t src_idx = slice_idx * src_slice_stride + concat_idx * elements_per_slice + elem_idx;
                    const int64_t dst_idx = slice_idx * dst_slice_stride +
                                           (offset_in_concat_dim + concat_idx) * elements_per_slice +
                                           elem_idx;

                    dst_ptr[dst_idx] = src_ptr[src_idx];
                }
            );
        }
        else if (dtype == DType::BFloat16) {
            // BFloat16 stored as uint16_t — pure copy, no conversion needed
            const uint16_t* src_ptr = get_data_ptr<const uint16_t>(tensor);
            uint16_t* dst_ptr = get_data_ptr<uint16_t>(output);

            const int64_t src_slice_stride = elements_per_slice * tensor_size_in_dim;
            const int64_t dst_slice_stride = elements_per_slice * out_shape[dim];

            queue.parallel_for<CatKernelBFloat16>(
                sycl::range<1>(num_slices * tensor_size_in_dim * elements_per_slice),
                [=](sycl::id<1> idx) {
                    const int64_t flat_idx = idx[0];
                    const int64_t slice_idx = flat_idx / src_slice_stride;
                    const int64_t remainder = flat_idx % src_slice_stride;
                    const int64_t concat_idx = remainder / elements_per_slice;
                    const int64_t elem_idx = remainder % elements_per_slice;

                    const int64_t src_idx = slice_idx * src_slice_stride + concat_idx * elements_per_slice + elem_idx;
                    const int64_t dst_idx = slice_idx * dst_slice_stride +
                                           (offset_in_concat_dim + concat_idx) * elements_per_slice +
                                           elem_idx;

                    dst_ptr[dst_idx] = src_ptr[src_idx];
                }
            );
        }
        else if (dtype == DType::Int32) {
            const int32_t* src_ptr = get_data_ptr<const int32_t>(tensor);
            int32_t* dst_ptr = get_data_ptr<int32_t>(output);

            const int64_t src_slice_stride = elements_per_slice * tensor_size_in_dim;
            const int64_t dst_slice_stride = elements_per_slice * out_shape[dim];

            queue.parallel_for<CatKernelInt32>(
                sycl::range<1>(num_slices * tensor_size_in_dim * elements_per_slice),
                [=](sycl::id<1> idx) {
                    const int64_t flat_idx = idx[0];
                    const int64_t slice_idx = flat_idx / src_slice_stride;
                    const int64_t remainder = flat_idx % src_slice_stride;
                    const int64_t concat_idx = remainder / elements_per_slice;
                    const int64_t elem_idx = remainder % elements_per_slice;

                    const int64_t src_idx = slice_idx * src_slice_stride + concat_idx * elements_per_slice + elem_idx;
                    const int64_t dst_idx = slice_idx * dst_slice_stride +
                                           (offset_in_concat_dim + concat_idx) * elements_per_slice +
                                           elem_idx;

                    dst_ptr[dst_idx] = src_ptr[src_idx];
                }
            );
        }
        else if (dtype == DType::Int64) {
            const int64_t* src_ptr = get_data_ptr<const int64_t>(tensor);
            int64_t* dst_ptr = get_data_ptr<int64_t>(output);

            const int64_t src_slice_stride = elements_per_slice * tensor_size_in_dim;
            const int64_t dst_slice_stride = elements_per_slice * out_shape[dim];

            queue.parallel_for<CatKernelInt64>(
                sycl::range<1>(num_slices * tensor_size_in_dim * elements_per_slice),
                [=](sycl::id<1> idx) {
                    const int64_t flat_idx = idx[0];
                    const int64_t slice_idx = flat_idx / src_slice_stride;
                    const int64_t remainder = flat_idx % src_slice_stride;
                    const int64_t concat_idx = remainder / elements_per_slice;
                    const int64_t elem_idx = remainder % elements_per_slice;

                    const int64_t src_idx = slice_idx * src_slice_stride + concat_idx * elements_per_slice + elem_idx;
                    const int64_t dst_idx = slice_idx * dst_slice_stride +
                                           (offset_in_concat_dim + concat_idx) * elements_per_slice +
                                           elem_idx;

                    dst_ptr[dst_idx] = src_ptr[src_idx];
                }
            );
        }
        else if (dtype == DType::Bool) {
            const bool* src_ptr = get_data_ptr<const bool>(tensor);
            bool* dst_ptr = get_data_ptr<bool>(output);

            const int64_t src_slice_stride = elements_per_slice * tensor_size_in_dim;
            const int64_t dst_slice_stride = elements_per_slice * out_shape[dim];

            queue.parallel_for<CatKernelBool>(
                sycl::range<1>(num_slices * tensor_size_in_dim * elements_per_slice),
                [=](sycl::id<1> idx) {
                    const int64_t flat_idx = idx[0];
                    const int64_t slice_idx = flat_idx / src_slice_stride;
                    const int64_t remainder = flat_idx % src_slice_stride;
                    const int64_t concat_idx = remainder / elements_per_slice;
                    const int64_t elem_idx = remainder % elements_per_slice;

                    const int64_t src_idx = slice_idx * src_slice_stride + concat_idx * elements_per_slice + elem_idx;
                    const int64_t dst_idx = slice_idx * dst_slice_stride +
                                           (offset_in_concat_dim + concat_idx) * elements_per_slice +
                                           elem_idx;

                    dst_ptr[dst_idx] = src_ptr[src_idx];
                }
            );
        }
        else if (dtype == DType::Complex64 || dtype == DType::Complex128 ||
                 dtype == DType::UInt16 || dtype == DType::UInt32 || dtype == DType::UInt64 ||
                 dtype == DType::FP8_E4M3 || dtype == DType::FP8_E5M2 ||
                 dtype == DType::FP8_E4M3FNUZ || dtype == DType::FP8_E5M2FNUZ) {
            // Generic fixed-width element copy (pure data movement). Each logical
            // element is dtype_size() bytes; reinterpret to the matching POD width.
            const int64_t src_slice_stride = elements_per_slice * tensor_size_in_dim;
            const int64_t dst_slice_stride = elements_per_slice * out_shape[dim];
            const int64_t total = num_slices * tensor_size_in_dim * elements_per_slice;
            const int64_t off = offset_in_concat_dim;
            const int64_t eps = elements_per_slice;
            auto gen = [&]<typename U>(int64_t lanes) {
                const U* src_ptr = get_data_ptr<const U>(tensor);
                U* dst_ptr = get_data_ptr<U>(output);
                queue.parallel_for(sycl::range<1>(total), [=](sycl::id<1> idx) {
                    const int64_t flat_idx = idx[0];
                    const int64_t slice_idx = flat_idx / src_slice_stride;
                    const int64_t remainder = flat_idx % src_slice_stride;
                    const int64_t concat_idx = remainder / eps;
                    const int64_t elem_idx = remainder % eps;
                    const int64_t src_idx = slice_idx * src_slice_stride + concat_idx * eps + elem_idx;
                    const int64_t dst_idx = slice_idx * dst_slice_stride + (off + concat_idx) * eps + elem_idx;
                    for (int64_t l = 0; l < lanes; ++l)
                        dst_ptr[dst_idx * lanes + l] = src_ptr[src_idx * lanes + l];
                });
            };
            switch (dtype_size(dtype)) {
                case 1:  gen.template operator()<uint8_t>(1);  break;
                case 2:  gen.template operator()<uint16_t>(1); break;
                case 4:  gen.template operator()<uint32_t>(1); break;
                case 8:  gen.template operator()<uint64_t>(1); break;
                case 16: gen.template operator()<uint64_t>(2); break;
                default: throw std::runtime_error("cat_kernel: unsupported dtype");
            }
        }
        else {
            throw std::runtime_error("cat_kernel: unsupported dtype");
        }

        offset_in_concat_dim += tensor_size_in_dim;
    }

    return output;
}

/**
 * @brief Clamp tensor values to a specified range [min_val, max_val]
 *
 * @param input Input tensor
 * @param min_val Minimum value (values below this are set to min_val)
 * @param max_val Maximum value (values above this are set to max_val)
 * @param queue SYCL queue for kernel execution
 * @return Output tensor with clamped values
 *
 * Each element x in the input is transformed to:
 *   clamp(x, min, max) = min if x < min
 *                      = max if x > max
 *                      = x   otherwise
 *
 * Example:
 *   input = [-2.0, -1.0, 0.0, 1.0, 2.0]
 *   clamp(input, -1.0, 1.0) = [-1.0, -1.0, 0.0, 1.0, 1.0]
 */
auto clamp_kernel(const Tensor& input, double min_val, double max_val, sycl::queue& queue) -> Tensor {
    const float min_val_f = static_cast<float>(min_val);
    const float max_val_f = static_cast<float>(max_val);
    // Validate min <= max
    if (min_val > max_val) {
        throw std::invalid_argument("clamp_kernel: min_val must be <= max_val");
    }

    // Create output tensor
    Tensor output(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                  input.dtype(), input.device());

    const int64_t numel = input.numel();

    // Dispatch based on dtype
    if (input.dtype() == DType::Float32) {
        const float* in_ptr = get_data_ptr<const float>(input);
        float* out_ptr = get_data_ptr<float>(output);

        queue.parallel_for<ClampKernelFloat32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            const float val = in_ptr[idx];
            out_ptr[idx] = sycl::fmin(sycl::fmax(val, min_val_f), max_val_f);
        }).wait();
    }
    else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        double* out_ptr = get_data_ptr<double>(output);

        const double min_d = static_cast<double>(min_val);
        const double max_d = static_cast<double>(max_val);

        queue.parallel_for<ClampKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            const double val = in_ptr[idx];
            out_ptr[idx] = sycl::fmin(sycl::fmax(val, min_d), max_d);
        }).wait();
    }
    else if (input.dtype() == DType::Float16) {
        const sycl::half* in_ptr = get_data_ptr<const sycl::half>(input);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);

        queue.parallel_for<ClampKernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            const float val = static_cast<float>(in_ptr[idx]);
            out_ptr[idx] = sycl::half(sycl::fmin(sycl::fmax(val, min_val_f), max_val_f));
        }).wait();
    }
    else if (input.dtype() == DType::BFloat16) {
        const uint16_t* in_ptr = get_data_ptr<const uint16_t>(input);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);

        queue.parallel_for<ClampKernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            const float val = bf16_to_f32(in_ptr[idx]);
            out_ptr[idx] = f32_to_bf16(sycl::fmin(sycl::fmax(val, min_val_f), max_val_f));
        }).wait();
    }
    else if (input.dtype() == DType::Int32) {
        const int32_t* in_ptr = get_data_ptr<const int32_t>(input);
        int32_t* out_ptr = get_data_ptr<int32_t>(output);

        const int32_t min_i = static_cast<int32_t>(min_val);
        const int32_t max_i = static_cast<int32_t>(max_val);

        queue.parallel_for<ClampKernelInt32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            const int32_t val = in_ptr[idx];
            out_ptr[idx] = sycl::min(sycl::max(val, min_i), max_i);
        }).wait();
    }
    else {
        // Remaining integer dtypes (Int32 handled above). Matches CPU/CUDA/ROCm.
        auto run = [&]<typename T>() {
            const T* in_ptr = get_data_ptr<const T>(input);
            T* out_ptr = get_data_ptr<T>(output);
            const T lo = static_cast<T>(min_val), hi = static_cast<T>(max_val);
            queue.parallel_for<OneapiClampIntKernel<T>>(sycl::range<1>(numel),
                [=](sycl::id<1> idx) {
                    const T v = in_ptr[idx];
                    out_ptr[idx] = sycl::min(sycl::max(v, lo), hi);
                }).wait();
        };
        switch (input.dtype()) {
            case DType::Int8:   run.template operator()<int8_t>();   break;
            case DType::Int16:  run.template operator()<int16_t>();  break;
            case DType::Int64:  run.template operator()<int64_t>();  break;
            case DType::UInt8:  run.template operator()<uint8_t>();  break;
            case DType::UInt16: run.template operator()<uint16_t>(); break;
            case DType::UInt32: run.template operator()<uint32_t>(); break;
            case DType::UInt64: run.template operator()<uint64_t>(); break;
            default:
                throw std::runtime_error("clamp_kernel: unsupported dtype");
        }
    }

    return output;
}

/**
 * @brief Compute the sign of tensor elements
 *
 * @param input Input tensor
 * @param queue SYCL queue for kernel execution
 * @return Output tensor with sign values
 *
 * Each element x in the input is transformed to:
 *   sign(x) = -1 if x < 0
 *           =  0 if x == 0
 *           = +1 if x > 0
 *
 * Note: For floating-point, handles -0.0 correctly (returns 0.0)
 *
 * Example:
 *   input = [-2.5, -0.0, 0.0, 3.7]
 *   sign(input) = [-1.0, 0.0, 0.0, 1.0]
 */
auto sign_kernel(const Tensor& input, sycl::queue& queue) -> Tensor {
    // Create output tensor with same shape and dtype
    Tensor output(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                  input.dtype(), input.device());

    const int64_t numel = input.numel();

    // Dispatch based on dtype
    if (input.dtype() == DType::Float32) {
        const float* in_ptr = get_data_ptr<const float>(input);
        float* out_ptr = get_data_ptr<float>(output);

        queue.parallel_for<SignKernelFloat32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            const float val = in_ptr[idx];
            // Use copysign for IEEE 754 correctness, handles -0.0 properly
            if (val == 0.0f) {
                out_ptr[idx] = 0.0f;
            } else if (val > 0.0f) {
                out_ptr[idx] = 1.0f;
            } else {
                out_ptr[idx] = -1.0f;
            }
        }).wait();
    }
    else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        double* out_ptr = get_data_ptr<double>(output);

        queue.parallel_for<SignKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            const double val = in_ptr[idx];
            if (val == 0.0) {
                out_ptr[idx] = 0.0;
            } else if (val > 0.0) {
                out_ptr[idx] = 1.0;
            } else {
                out_ptr[idx] = -1.0;
            }
        }).wait();
    }
    else if (input.dtype() == DType::Float16) {
        const sycl::half* in_ptr = get_data_ptr<const sycl::half>(input);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);

        queue.parallel_for<SignKernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            const float val = static_cast<float>(in_ptr[idx]);
            if (val == 0.0f) {
                out_ptr[idx] = sycl::half(0.0f);
            } else if (val > 0.0f) {
                out_ptr[idx] = sycl::half(1.0f);
            } else {
                out_ptr[idx] = sycl::half(-1.0f);
            }
        }).wait();
    }
    else if (input.dtype() == DType::BFloat16) {
        const uint16_t* in_ptr = get_data_ptr<const uint16_t>(input);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);

        const uint16_t bf16_zero = f32_to_bf16(0.0f);
        const uint16_t bf16_pos_one = f32_to_bf16(1.0f);
        const uint16_t bf16_neg_one = f32_to_bf16(-1.0f);

        queue.parallel_for<SignKernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            const float val = bf16_to_f32(in_ptr[idx]);
            if (val == 0.0f) {
                out_ptr[idx] = bf16_zero;
            } else if (val > 0.0f) {
                out_ptr[idx] = bf16_pos_one;
            } else {
                out_ptr[idx] = bf16_neg_one;
            }
        }).wait();
    }
    else if (input.dtype() == DType::Int32) {
        const int32_t* in_ptr = get_data_ptr<const int32_t>(input);
        int32_t* out_ptr = get_data_ptr<int32_t>(output);

        queue.parallel_for<SignKernelInt32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            const int32_t val = in_ptr[idx];
            if (val == 0) {
                out_ptr[idx] = 0;
            } else if (val > 0) {
                out_ptr[idx] = 1;
            } else {
                out_ptr[idx] = -1;
            }
        }).wait();
    }
    else {
        // Remaining integer dtypes (Int32 handled above): -1/0/1 signed, 0/1 unsigned.
        auto run = [&]<typename T>() {
            const T* in_ptr = get_data_ptr<const T>(input);
            T* out_ptr = get_data_ptr<T>(output);
            const bool is_signed = (T(-1) < T(0));
            queue.parallel_for<OneapiSignIntKernel<T>>(sycl::range<1>(numel),
                [=](sycl::id<1> idx) {
                    const T v = in_ptr[idx];
                    const int neg = (is_signed && v < T(0)) ? 1 : 0;
                    const int pos = (v > T(0)) ? 1 : 0;
                    out_ptr[idx] = static_cast<T>(pos - neg);
                }).wait();
        };
        switch (input.dtype()) {
            case DType::Int8:   run.template operator()<int8_t>();   break;
            case DType::Int16:  run.template operator()<int16_t>();  break;
            case DType::Int64:  run.template operator()<int64_t>();  break;
            case DType::UInt8:  run.template operator()<uint8_t>();  break;
            case DType::UInt16: run.template operator()<uint16_t>(); break;
            case DType::UInt32: run.template operator()<uint32_t>(); break;
            case DType::UInt64: run.template operator()<uint64_t>(); break;
            default:
                throw std::runtime_error("sign_kernel: unsupported dtype");
        }
    }

    return output;
}

} // namespace oneapi
} // namespace tenzor
