#include "tenzor/core/tensor.hpp"
#include "tenzor/backend/backend.hpp"
#include <sycl/sycl.hpp>
#include <limits>
#include <stdexcept>
#include <utility>

#ifdef TENZOR_HAS_ONEDNN
#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_sycl.hpp>
#endif

namespace tenzor {
namespace oneapi {

// Kernel class declarations for SYCL
class MaxPool2dKernelFloat32;
class MaxPool2dKernelFloat64;
class MaxPool2dWithIndicesKernelFloat32;
class MaxPool2dWithIndicesKernelFloat64;
class MaxPool2dWithIndicesKernelFloat16;
class AvgPool2dKernelFloat32;
class AvgPool2dKernelFloat64;
class AvgPool2dKernelFloat16;
class AdaptiveAvgPool2dKernelFloat32;
class AdaptiveAvgPool2dKernelFloat64;
class AdaptiveAvgPool2dKernelFloat16;
class AdaptiveMaxPool2dKernelFloat32;
class AdaptiveMaxPool2dKernelFloat64;
class AdaptiveMaxPool2dKernelFloat16;
class AdaptiveAvgPool2dBackwardKernelFloat32;
class AdaptiveAvgPool2dBackwardKernelFloat64;
class AdaptiveAvgPool2dBackwardKernelFloat16;
class AvgPool2dBackwardKernelFloat32;
class AvgPool2dBackwardKernelFloat64;
class AvgPool2dBackwardKernelFloat16;
class MaxPool2dBackwardKernelFloat32;
class MaxPool2dBackwardKernelFloat64;
class MaxPool2dBackwardWithIndicesKernelFloat32;
class MaxPool2dBackwardWithIndicesKernelFloat64;
class MaxPool2dBackwardWithIndicesKernelFloat16;
class MaxPool2dWithIndicesKernelBFloat16;
class AvgPool2dKernelBFloat16;
class AdaptiveAvgPool2dKernelBFloat16;
class AdaptiveMaxPool2dKernelBFloat16;
class AdaptiveAvgPool2dBackwardKernelBFloat16;
class AvgPool2dBackwardKernelBFloat16;
class MaxPool2dBackwardKernelBFloat16;
class MaxPool2dBackwardWithIndicesKernelBFloat16;

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

#ifdef TENZOR_HAS_ONEDNN

// MaxPool2d forward with indices using oneDNN - returns both output and indices for backward pass
auto maxpool2d_forward_with_indices(const Tensor& input, int64_t kernel_size, int64_t stride,
                                    int64_t padding, int64_t dilation, sycl::queue& queue) -> std::pair<Tensor, Tensor> {
    using namespace dnnl;

    auto shape = input.shape();
    if (shape.size() != 4) {
        throw std::invalid_argument("MaxPool2d requires 4D input (N, C, H, W)");
    }

    const int64_t N = shape[0];
    const int64_t C = shape[1];
    const int64_t H_in = shape[2];
    const int64_t W_in = shape[3];

    const int64_t H_out = (H_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    const int64_t W_out = (W_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    Tensor output({N, C, H_out, W_out}, input.dtype(), input.device());
    // Store indices as same dtype as input (following CPU path convention)
    Tensor indices({N, C, H_out, W_out}, input.dtype(), input.device());

    // Create oneDNN engine and stream
    auto dnnl_engine = sycl_interop::make_engine(queue.get_device(), queue.get_context());
    auto dnnl_stream = sycl_interop::make_stream(dnnl_engine, queue);

    // Memory descriptors
    memory::dims src_dims = {N, C, H_in, W_in};
    memory::dims dst_dims = {N, C, H_out, W_out};
    memory::dims kernel_dims = {kernel_size, kernel_size};
    memory::dims strides_dims = {stride, stride};
    memory::dims padding_dims = {padding, padding};
    memory::dims dilation_dims = {dilation - 1, dilation - 1};

    auto src_md = memory::desc(src_dims, memory::data_type::f32, memory::format_tag::nchw);
    auto dst_md = memory::desc(dst_dims, memory::data_type::f32, memory::format_tag::nchw);

    // Create pooling descriptor - use forward_training to get workspace (indices)
    auto pool_desc = pooling_forward::desc(
        prop_kind::forward_training,
        algorithm::pooling_max,
        src_md, dst_md,
        strides_dims, kernel_dims,
        dilation_dims,
        padding_dims, padding_dims
    );

    auto pool_pd = pooling_forward::primitive_desc(pool_desc, dnnl_engine);

    // Wrap tensors
    auto src_mem = sycl_interop::make_memory(pool_pd.src_desc(), dnnl_engine,
                                              sycl_interop::memory_kind::usm,
                                              const_cast<void*>(input.data_ptr()));

    auto dst_mem = sycl_interop::make_memory(pool_pd.dst_desc(), dnnl_engine,
                                              sycl_interop::memory_kind::usm,
                                              const_cast<void*>(output.data_ptr()));

    // Create workspace for indices (oneDNN stores max indices in workspace)
    auto workspace_mem = memory(pool_pd.workspace_desc(), dnnl_engine);

    // Execute
    auto pool_prim = pooling_forward(pool_pd);
    pool_prim.execute(dnnl_stream, {
        {DNNL_ARG_SRC, src_mem},
        {DNNL_ARG_DST, dst_mem},
        {DNNL_ARG_WORKSPACE, workspace_mem}
    });

    dnnl_stream.wait();

    // oneDNN workspace format is internal, so we need to compute indices manually
    // for compatibility with our backward pass that expects linear indices
    if (input.dtype() == DType::Float32) {
        const float* in_ptr = get_data_ptr<const float>(input);
        float* idx_ptr = get_data_ptr<float>(indices);

        const int64_t total_size = N * C * H_out * W_out;
        queue.parallel_for(sycl::range<1>(total_size), [=](sycl::id<1> flat_idx) {
            int64_t temp = flat_idx;
            const int64_t w_out = temp % W_out;
            temp /= W_out;
            const int64_t h_out = temp % H_out;
            temp /= H_out;
            const int64_t c = temp % C;
            const int64_t n = temp / C;

            float max_val = -3.4028235e+38f;
            int64_t max_idx = 0;

            for (int64_t kh = 0; kh < kernel_size; ++kh) {
                for (int64_t kw = 0; kw < kernel_size; ++kw) {
                    int64_t h_in = h_out * stride - padding + kh * dilation;
                    int64_t w_in = w_out * stride - padding + kw * dilation;

                    if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                        int64_t input_idx = ((n * C + c) * H_in + h_in) * W_in + w_in;
                        float val = in_ptr[input_idx];
                        if (val > max_val) {
                            max_val = val;
                            max_idx = input_idx;
                        }
                    }
                }
            }

            idx_ptr[((n * C + c) * H_out + h_out) * W_out + w_out] = static_cast<float>(max_idx);
        }).wait();
    }

    return {output, indices};
}

// MaxPool2d forward using oneDNN - returns only output
auto maxpool2d_forward(const Tensor& input, int64_t kernel_size, int64_t stride,
                       int64_t padding, int64_t dilation, sycl::queue& queue) -> Tensor {
    auto [output, indices] = maxpool2d_forward_with_indices(input, kernel_size, stride, padding, dilation, queue);
    return output;
}

// AvgPool2d forward using oneDNN
auto avgpool2d_forward(const Tensor& input, int64_t kernel_size, int64_t stride,
                       int64_t padding, bool count_include_pad, sycl::queue& queue) -> Tensor {
    using namespace dnnl;

    auto shape = input.shape();
    if (shape.size() != 4) {
        throw std::invalid_argument("AvgPool2d requires 4D input (N, C, H, W)");
    }

    const int64_t N = shape[0];
    const int64_t C = shape[1];
    const int64_t H_in = shape[2];
    const int64_t W_in = shape[3];

    const int64_t H_out = (H_in + 2 * padding - kernel_size) / stride + 1;
    const int64_t W_out = (W_in + 2 * padding - kernel_size) / stride + 1;

    Tensor output({N, C, H_out, W_out}, input.dtype(), input.device());

    // Create oneDNN engine and stream
    auto dnnl_engine = sycl_interop::make_engine(queue.get_device(), queue.get_context());
    auto dnnl_stream = sycl_interop::make_stream(dnnl_engine, queue);

    // Memory descriptors
    memory::dims src_dims = {N, C, H_in, W_in};
    memory::dims dst_dims = {N, C, H_out, W_out};
    memory::dims kernel_dims = {kernel_size, kernel_size};
    memory::dims strides_dims = {stride, stride};
    memory::dims padding_dims = {padding, padding};

    auto src_md = memory::desc(src_dims, memory::data_type::f32, memory::format_tag::nchw);
    auto dst_md = memory::desc(dst_dims, memory::data_type::f32, memory::format_tag::nchw);

    // Choose algorithm based on count_include_pad
    auto algo = count_include_pad ?
        algorithm::pooling_avg_include_padding :
        algorithm::pooling_avg_exclude_padding;

    // Create pooling descriptor
    auto pool_desc = pooling_forward::desc(
        prop_kind::forward_inference,
        algo,
        src_md, dst_md,
        strides_dims, kernel_dims,
        padding_dims, padding_dims
    );

    auto pool_pd = pooling_forward::primitive_desc(pool_desc, dnnl_engine);

    // Wrap tensors
    auto src_mem = sycl_interop::make_memory(pool_pd.src_desc(), dnnl_engine,
                                              sycl_interop::memory_kind::usm,
                                              const_cast<void*>(input.data_ptr()));

    auto dst_mem = sycl_interop::make_memory(pool_pd.dst_desc(), dnnl_engine,
                                              sycl_interop::memory_kind::usm,
                                              const_cast<void*>(output.data_ptr()));

    // Execute
    auto pool_prim = pooling_forward(pool_pd);
    pool_prim.execute(dnnl_stream, {
        {DNNL_ARG_SRC, src_mem},
        {DNNL_ARG_DST, dst_mem}
    });

    dnnl_stream.wait();

    return output;
}

#else // !TENZOR_HAS_ONEDNN - Pure SYCL implementation

// MaxPool2d forward with indices (pure SYCL) - returns both output and indices for backward pass
auto maxpool2d_forward_with_indices(const Tensor& input, int64_t kernel_size, int64_t stride,
                                    int64_t padding, int64_t dilation, sycl::queue& queue) -> std::pair<Tensor, Tensor> {
    auto shape = input.shape();
    if (shape.size() != 4) {
        throw std::invalid_argument("MaxPool2d requires 4D input (N, C, H, W)");
    }

    const int64_t N = shape[0];
    const int64_t C = shape[1];
    const int64_t H_in = shape[2];
    const int64_t W_in = shape[3];

    const int64_t H_out = (H_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    const int64_t W_out = (W_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    Tensor output({N, C, H_out, W_out}, input.dtype(), input.device());
    // Store indices as same dtype as input (for simplicity, following CPU path convention)
    Tensor indices({N, C, H_out, W_out}, input.dtype(), input.device());

    if (input.dtype() == DType::Float32) {
        const float* in_ptr = get_data_ptr<const float>(input);
        float* out_ptr = get_data_ptr<float>(output);
        float* idx_ptr = get_data_ptr<float>(indices);

        const int64_t total_size = N * C * H_out * W_out;
        queue.parallel_for<MaxPool2dWithIndicesKernelFloat32>(sycl::range<1>(total_size), [=](sycl::id<1> flat_idx) {
            int64_t temp = flat_idx;
            const int64_t w_out = temp % W_out;
            temp /= W_out;
            const int64_t h_out = temp % H_out;
            temp /= H_out;
            const int64_t c = temp % C;
            const int64_t n = temp / C;

            float max_val = -3.4028235e+38f;
            int64_t max_idx = 0;

            for (int64_t kh = 0; kh < kernel_size; ++kh) {
                for (int64_t kw = 0; kw < kernel_size; ++kw) {
                    int64_t h_in = h_out * stride - padding + kh * dilation;
                    int64_t w_in = w_out * stride - padding + kw * dilation;

                    if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                        int64_t input_idx = ((n * C + c) * H_in + h_in) * W_in + w_in;
                        float val = in_ptr[input_idx];
                        if (val > max_val) {
                            max_val = val;
                            max_idx = input_idx;
                        }
                    }
                }
            }

            int64_t out_idx = ((n * C + c) * H_out + h_out) * W_out + w_out;
            out_ptr[out_idx] = max_val;
            idx_ptr[out_idx] = static_cast<float>(max_idx);
        }).wait();
    }
    else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        double* out_ptr = get_data_ptr<double>(output);
        double* idx_ptr = get_data_ptr<double>(indices);

        const int64_t total_size = N * C * H_out * W_out;
        queue.parallel_for<MaxPool2dWithIndicesKernelFloat64>(sycl::range<1>(total_size), [=](sycl::id<1> flat_idx) {
            int64_t temp = flat_idx;
            const int64_t w_out = temp % W_out;
            temp /= W_out;
            const int64_t h_out = temp % H_out;
            temp /= H_out;
            const int64_t c = temp % C;
            const int64_t n = temp / C;

            double max_val = -1.7976931348623157e+308;
            int64_t max_idx = 0;

            for (int64_t kh = 0; kh < kernel_size; ++kh) {
                for (int64_t kw = 0; kw < kernel_size; ++kw) {
                    int64_t h_in = h_out * stride - padding + kh * dilation;
                    int64_t w_in = w_out * stride - padding + kw * dilation;

                    if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                        int64_t input_idx = ((n * C + c) * H_in + h_in) * W_in + w_in;
                        double val = in_ptr[input_idx];
                        if (val > max_val) {
                            max_val = val;
                            max_idx = input_idx;
                        }
                    }
                }
            }

            int64_t out_idx = ((n * C + c) * H_out + h_out) * W_out + w_out;
            out_ptr[out_idx] = max_val;
            idx_ptr[out_idx] = static_cast<double>(max_idx);
        }).wait();
    }
    else if (input.dtype() == DType::Float16) {
        const sycl::half* in_ptr = get_data_ptr<const sycl::half>(input);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);
        sycl::half* idx_ptr = get_data_ptr<sycl::half>(indices);

        const int64_t total_size = N * C * H_out * W_out;
        queue.parallel_for<MaxPool2dWithIndicesKernelFloat16>(sycl::range<1>(total_size), [=](sycl::id<1> flat_idx) {
            int64_t temp = flat_idx;
            const int64_t w_out = temp % W_out;
            temp /= W_out;
            const int64_t h_out = temp % H_out;
            temp /= H_out;
            const int64_t c = temp % C;
            const int64_t n = temp / C;

            // Use float for max computation for numerical stability
            float max_val = -3.4028235e+38f;
            int64_t max_idx = 0;

            for (int64_t kh = 0; kh < kernel_size; ++kh) {
                for (int64_t kw = 0; kw < kernel_size; ++kw) {
                    int64_t h_in = h_out * stride - padding + kh * dilation;
                    int64_t w_in = w_out * stride - padding + kw * dilation;

                    if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                        int64_t input_idx = ((n * C + c) * H_in + h_in) * W_in + w_in;
                        float val = static_cast<float>(in_ptr[input_idx]);
                        if (val > max_val) {
                            max_val = val;
                            max_idx = input_idx;
                        }
                    }
                }
            }

            int64_t out_idx = ((n * C + c) * H_out + h_out) * W_out + w_out;
            out_ptr[out_idx] = sycl::half(max_val);
            idx_ptr[out_idx] = sycl::half(static_cast<float>(max_idx));
        }).wait();
    }
    else if (input.dtype() == DType::BFloat16) {
        const uint16_t* in_ptr = get_data_ptr<const uint16_t>(input);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);
        uint16_t* idx_ptr = get_data_ptr<uint16_t>(indices);

        const int64_t total_size = N * C * H_out * W_out;
        queue.parallel_for<MaxPool2dWithIndicesKernelBFloat16>(sycl::range<1>(total_size), [=](sycl::id<1> flat_idx) {
            int64_t temp = flat_idx;
            const int64_t w_out = temp % W_out;
            temp /= W_out;
            const int64_t h_out = temp % H_out;
            temp /= H_out;
            const int64_t c = temp % C;
            const int64_t n = temp / C;

            // Use float for max computation for numerical stability
            float max_val = -3.4028235e+38f;
            int64_t max_idx = 0;

            for (int64_t kh = 0; kh < kernel_size; ++kh) {
                for (int64_t kw = 0; kw < kernel_size; ++kw) {
                    int64_t h_in = h_out * stride - padding + kh * dilation;
                    int64_t w_in = w_out * stride - padding + kw * dilation;

                    if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                        int64_t input_idx = ((n * C + c) * H_in + h_in) * W_in + w_in;
                        float val = bf16_to_f32(in_ptr[input_idx]);
                        if (val > max_val) {
                            max_val = val;
                            max_idx = input_idx;
                        }
                    }
                }
            }

            int64_t out_idx = ((n * C + c) * H_out + h_out) * W_out + w_out;
            out_ptr[out_idx] = f32_to_bf16(max_val);
            idx_ptr[out_idx] = f32_to_bf16(static_cast<float>(max_idx));
        }).wait();
    }
    else {
        throw std::runtime_error("Unsupported dtype for maxpool2d_forward_with_indices");
    }

    return {output, indices};
}

// MaxPool2d forward (pure SYCL) - returns only output (for operations that don't need indices)
auto maxpool2d_forward(const Tensor& input, int64_t kernel_size, int64_t stride,
                       int64_t padding, int64_t dilation, sycl::queue& queue) -> Tensor {
    auto [output, indices] = maxpool2d_forward_with_indices(input, kernel_size, stride, padding, dilation, queue);
    return output;
}

// AvgPool2d forward (pure SYCL)
auto avgpool2d_forward(const Tensor& input, int64_t kernel_size, int64_t stride,
                       int64_t padding, bool count_include_pad, sycl::queue& queue) -> Tensor {
    auto shape = input.shape();
    if (shape.size() != 4) {
        throw std::invalid_argument("AvgPool2d requires 4D input (N, C, H, W)");
    }

    const int64_t N = shape[0];
    const int64_t C = shape[1];
    const int64_t H_in = shape[2];
    const int64_t W_in = shape[3];

    const int64_t H_out = (H_in + 2 * padding - kernel_size) / stride + 1;
    const int64_t W_out = (W_in + 2 * padding - kernel_size) / stride + 1;

    Tensor output({N, C, H_out, W_out}, input.dtype(), input.device());

    if (input.dtype() == DType::Float32) {
        const float* in_ptr = get_data_ptr<const float>(input);
        float* out_ptr = get_data_ptr<float>(output);

        const int64_t total_size = N * C * H_out * W_out;
        queue.parallel_for<AvgPool2dKernelFloat32>(sycl::range<1>(total_size), [=](sycl::id<1> flat_idx) {
            int64_t temp = flat_idx;
            const int64_t w_out = temp % W_out;
            temp /= W_out;
            const int64_t h_out = temp % H_out;
            temp /= H_out;
            const int64_t c = temp % C;
            const int64_t n = temp / C;

            float sum = 0.0f;
            int64_t count = 0;

            for (int64_t kh = 0; kh < kernel_size; ++kh) {
                for (int64_t kw = 0; kw < kernel_size; ++kw) {
                    int64_t h_in = h_out * stride - padding + kh;
                    int64_t w_in = w_out * stride - padding + kw;

                    if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                        sum += in_ptr[((n * C + c) * H_in + h_in) * W_in + w_in];
                        count++;
                    } else if (count_include_pad) {
                        count++;
                    }
                }
            }

            out_ptr[((n * C + c) * H_out + h_out) * W_out + w_out] =
                count > 0 ? sum / static_cast<float>(count) : 0.0f;
        }).wait();
    }
    else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        double* out_ptr = get_data_ptr<double>(output);

        const int64_t total_size = N * C * H_out * W_out;
        queue.parallel_for<AvgPool2dKernelFloat64>(sycl::range<1>(total_size), [=](sycl::id<1> flat_idx) {
            int64_t temp = flat_idx;
            const int64_t w_out = temp % W_out;
            temp /= W_out;
            const int64_t h_out = temp % H_out;
            temp /= H_out;
            const int64_t c = temp % C;
            const int64_t n = temp / C;

            double sum = 0.0;
            int64_t count = 0;

            for (int64_t kh = 0; kh < kernel_size; ++kh) {
                for (int64_t kw = 0; kw < kernel_size; ++kw) {
                    int64_t h_in = h_out * stride - padding + kh;
                    int64_t w_in = w_out * stride - padding + kw;

                    if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                        sum += in_ptr[((n * C + c) * H_in + h_in) * W_in + w_in];
                        count++;
                    } else if (count_include_pad) {
                        count++;
                    }
                }
            }

            out_ptr[((n * C + c) * H_out + h_out) * W_out + w_out] =
                count > 0 ? sum / static_cast<double>(count) : 0.0;
        }).wait();
    }
    else if (input.dtype() == DType::Float16) {
        const sycl::half* in_ptr = get_data_ptr<const sycl::half>(input);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);

        const int64_t total_size = N * C * H_out * W_out;
        queue.parallel_for<AvgPool2dKernelFloat16>(sycl::range<1>(total_size), [=](sycl::id<1> flat_idx) {
            int64_t temp = flat_idx;
            const int64_t w_out = temp % W_out;
            temp /= W_out;
            const int64_t h_out = temp % H_out;
            temp /= H_out;
            const int64_t c = temp % C;
            const int64_t n = temp / C;

            // Use float for accumulation for numerical stability
            float sum = 0.0f;
            int64_t count = 0;

            for (int64_t kh = 0; kh < kernel_size; ++kh) {
                for (int64_t kw = 0; kw < kernel_size; ++kw) {
                    int64_t h_in = h_out * stride - padding + kh;
                    int64_t w_in = w_out * stride - padding + kw;

                    if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                        sum += static_cast<float>(in_ptr[((n * C + c) * H_in + h_in) * W_in + w_in]);
                        count++;
                    } else if (count_include_pad) {
                        count++;
                    }
                }
            }

            out_ptr[((n * C + c) * H_out + h_out) * W_out + w_out] =
                sycl::half(count > 0 ? sum / static_cast<float>(count) : 0.0f);
        }).wait();
    }
    else if (input.dtype() == DType::BFloat16) {
        const uint16_t* in_ptr = get_data_ptr<const uint16_t>(input);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);

        const int64_t total_size = N * C * H_out * W_out;
        queue.parallel_for<AvgPool2dKernelBFloat16>(sycl::range<1>(total_size), [=](sycl::id<1> flat_idx) {
            int64_t temp = flat_idx;
            const int64_t w_out = temp % W_out;
            temp /= W_out;
            const int64_t h_out = temp % H_out;
            temp /= H_out;
            const int64_t c = temp % C;
            const int64_t n = temp / C;

            // Use float for accumulation for numerical stability
            float sum = 0.0f;
            int64_t count = 0;

            for (int64_t kh = 0; kh < kernel_size; ++kh) {
                for (int64_t kw = 0; kw < kernel_size; ++kw) {
                    int64_t h_in = h_out * stride - padding + kh;
                    int64_t w_in = w_out * stride - padding + kw;

                    if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                        sum += bf16_to_f32(in_ptr[((n * C + c) * H_in + h_in) * W_in + w_in]);
                        count++;
                    } else if (count_include_pad) {
                        count++;
                    }
                }
            }

            out_ptr[((n * C + c) * H_out + h_out) * W_out + w_out] =
                f32_to_bf16(count > 0 ? sum / static_cast<float>(count) : 0.0f);
        }).wait();
    }
    else {
        throw std::runtime_error("Unsupported dtype for avgpool2d_forward");
    }

    return output;
}

#endif // TENZOR_HAS_ONEDNN

// AdaptiveAvgPool2d - always pure SYCL
auto adaptive_avgpool2d_forward(const Tensor& input, int64_t output_h, int64_t output_w,
                                 sycl::queue& queue) -> Tensor {
    auto shape = input.shape();
    if (shape.size() != 4) {
        throw std::invalid_argument("AdaptiveAvgPool2d requires 4D input (N, C, H, W)");
    }

    const int64_t N = shape[0];
    const int64_t C = shape[1];
    const int64_t H_in = shape[2];
    const int64_t W_in = shape[3];

    Tensor output({N, C, output_h, output_w}, input.dtype(), input.device());

    if (input.dtype() == DType::Float32) {
        const float* in_ptr = get_data_ptr<const float>(input);
        float* out_ptr = get_data_ptr<float>(output);

        const int64_t total_size = N * C * output_h * output_w;
        queue.parallel_for<AdaptiveAvgPool2dKernelFloat32>(sycl::range<1>(total_size), [=](sycl::id<1> flat_idx) {
            int64_t temp = flat_idx;
            const int64_t w_out = temp % output_w;
            temp /= output_w;
            const int64_t h_out = temp % output_h;
            temp /= output_h;
            const int64_t c = temp % C;
            const int64_t n = temp / C;

            // Calculate input region
            const int64_t h_start = (h_out * H_in) / output_h;
            const int64_t h_end = ((h_out + 1) * H_in) / output_h;
            const int64_t w_start = (w_out * W_in) / output_w;
            const int64_t w_end = ((w_out + 1) * W_in) / output_w;

            float sum = 0.0f;
            int64_t count = 0;

            for (int64_t h = h_start; h < h_end; ++h) {
                for (int64_t w = w_start; w < w_end; ++w) {
                    sum += in_ptr[((n * C + c) * H_in + h) * W_in + w];
                    count++;
                }
            }

            out_ptr[((n * C + c) * output_h + h_out) * output_w + w_out] =
                count > 0 ? sum / static_cast<float>(count) : 0.0f;
        }).wait();
    }
    else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        double* out_ptr = get_data_ptr<double>(output);

        const int64_t total_size = N * C * output_h * output_w;
        queue.parallel_for<AdaptiveAvgPool2dKernelFloat64>(sycl::range<1>(total_size), [=](sycl::id<1> flat_idx) {
            int64_t temp = flat_idx;
            const int64_t w_out = temp % output_w;
            temp /= output_w;
            const int64_t h_out = temp % output_h;
            temp /= output_h;
            const int64_t c = temp % C;
            const int64_t n = temp / C;

            const int64_t h_start = (h_out * H_in) / output_h;
            const int64_t h_end = ((h_out + 1) * H_in) / output_h;
            const int64_t w_start = (w_out * W_in) / output_w;
            const int64_t w_end = ((w_out + 1) * W_in) / output_w;

            double sum = 0.0;
            int64_t count = 0;

            for (int64_t h = h_start; h < h_end; ++h) {
                for (int64_t w = w_start; w < w_end; ++w) {
                    sum += in_ptr[((n * C + c) * H_in + h) * W_in + w];
                    count++;
                }
            }

            out_ptr[((n * C + c) * output_h + h_out) * output_w + w_out] =
                count > 0 ? sum / static_cast<double>(count) : 0.0;
        }).wait();
    }
    else if (input.dtype() == DType::Float16) {
        const sycl::half* in_ptr = get_data_ptr<const sycl::half>(input);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);

        const int64_t total_size = N * C * output_h * output_w;
        queue.parallel_for<AdaptiveAvgPool2dKernelFloat16>(sycl::range<1>(total_size), [=](sycl::id<1> flat_idx) {
            int64_t temp = flat_idx;
            const int64_t w_out = temp % output_w;
            temp /= output_w;
            const int64_t h_out = temp % output_h;
            temp /= output_h;
            const int64_t c = temp % C;
            const int64_t n = temp / C;

            // Calculate input region
            const int64_t h_start = (h_out * H_in) / output_h;
            const int64_t h_end = ((h_out + 1) * H_in) / output_h;
            const int64_t w_start = (w_out * W_in) / output_w;
            const int64_t w_end = ((w_out + 1) * W_in) / output_w;

            // Use float accumulation for numerical stability
            float sum = 0.0f;
            int64_t count = 0;

            for (int64_t h = h_start; h < h_end; ++h) {
                for (int64_t w = w_start; w < w_end; ++w) {
                    sum += static_cast<float>(in_ptr[((n * C + c) * H_in + h) * W_in + w]);
                    count++;
                }
            }

            out_ptr[((n * C + c) * output_h + h_out) * output_w + w_out] =
                sycl::half(count > 0 ? sum / static_cast<float>(count) : 0.0f);
        }).wait();
    }
    else if (input.dtype() == DType::BFloat16) {
        const uint16_t* in_ptr = get_data_ptr<const uint16_t>(input);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);

        const int64_t total_size = N * C * output_h * output_w;
        queue.parallel_for<AdaptiveAvgPool2dKernelBFloat16>(sycl::range<1>(total_size), [=](sycl::id<1> flat_idx) {
            int64_t temp = flat_idx;
            const int64_t w_out = temp % output_w;
            temp /= output_w;
            const int64_t h_out = temp % output_h;
            temp /= output_h;
            const int64_t c = temp % C;
            const int64_t n = temp / C;

            // Calculate input region
            const int64_t h_start = (h_out * H_in) / output_h;
            const int64_t h_end = ((h_out + 1) * H_in) / output_h;
            const int64_t w_start = (w_out * W_in) / output_w;
            const int64_t w_end = ((w_out + 1) * W_in) / output_w;

            // Use float accumulation for numerical stability
            float sum = 0.0f;
            int64_t count = 0;

            for (int64_t h = h_start; h < h_end; ++h) {
                for (int64_t w = w_start; w < w_end; ++w) {
                    sum += bf16_to_f32(in_ptr[((n * C + c) * H_in + h) * W_in + w]);
                    count++;
                }
            }

            out_ptr[((n * C + c) * output_h + h_out) * output_w + w_out] =
                f32_to_bf16(count > 0 ? sum / static_cast<float>(count) : 0.0f);
        }).wait();
    }
    else {
        throw std::runtime_error("Unsupported dtype for adaptive_avgpool2d_forward");
    }

    return output;
}

// AdaptiveAvgPool2d backward - always pure SYCL
auto adaptive_avgpool2d_backward(const Tensor& grad_output, int64_t H_in, int64_t W_in,
                                  sycl::queue& queue) -> Tensor {
    auto shape = grad_output.shape();
    if (shape.size() != 4) {
        throw std::invalid_argument("AdaptiveAvgPool2d backward requires 4D grad_output (N, C, H, W)");
    }

    const int64_t N = shape[0];
    const int64_t C = shape[1];
    const int64_t H_out = shape[2];
    const int64_t W_out = shape[3];

    Tensor grad_input({N, C, H_in, W_in}, grad_output.dtype(), grad_output.device());

    // Initialize grad_input to zeros
    const size_t bytes = grad_input.numel() * grad_input.dtype_size();
    queue.memset(const_cast<void*>(grad_input.data_ptr()), 0, bytes).wait();

    if (grad_output.dtype() == DType::Float32) {
        const float* grad_out_ptr = get_data_ptr<const float>(grad_output);
        float* grad_in_ptr = get_data_ptr<float>(grad_input);

        const int64_t total_size = N * C * H_out * W_out;
        queue.parallel_for<AdaptiveAvgPool2dBackwardKernelFloat32>(sycl::range<1>(total_size), [=](sycl::id<1> flat_idx) {
            int64_t temp = flat_idx;
            const int64_t w_out = temp % W_out;
            temp /= W_out;
            const int64_t h_out = temp % H_out;
            temp /= H_out;
            const int64_t c = temp % C;
            const int64_t n = temp / C;

            // Calculate input region
            const int64_t h_start = (h_out * H_in) / H_out;
            const int64_t h_end = ((h_out + 1) * H_in) / H_out;
            const int64_t w_start = (w_out * W_in) / W_out;
            const int64_t w_end = ((w_out + 1) * W_in) / W_out;

            const int64_t count = (h_end - h_start) * (w_end - w_start);
            const float grad_val = grad_out_ptr[flat_idx] / static_cast<float>(count);

            // Distribute gradient to input positions using atomic add
            for (int64_t h = h_start; h < h_end; ++h) {
                for (int64_t w = w_start; w < w_end; ++w) {
                    const int64_t input_idx = ((n * C + c) * H_in + h) * W_in + w;
                    sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                     sycl::access::address_space::global_space> atomic_val(grad_in_ptr[input_idx]);
                    atomic_val.fetch_add(grad_val);
                }
            }
        }).wait();
    }
    else if (grad_output.dtype() == DType::Float64) {
        const double* grad_out_ptr = get_data_ptr<const double>(grad_output);
        double* grad_in_ptr = get_data_ptr<double>(grad_input);

        const int64_t total_size = N * C * H_out * W_out;
        queue.parallel_for<AdaptiveAvgPool2dBackwardKernelFloat64>(sycl::range<1>(total_size), [=](sycl::id<1> flat_idx) {
            int64_t temp = flat_idx;
            const int64_t w_out = temp % W_out;
            temp /= W_out;
            const int64_t h_out = temp % H_out;
            temp /= H_out;
            const int64_t c = temp % C;
            const int64_t n = temp / C;

            const int64_t h_start = (h_out * H_in) / H_out;
            const int64_t h_end = ((h_out + 1) * H_in) / H_out;
            const int64_t w_start = (w_out * W_in) / W_out;
            const int64_t w_end = ((w_out + 1) * W_in) / W_out;

            const int64_t count = (h_end - h_start) * (w_end - w_start);
            const double grad_val = grad_out_ptr[flat_idx] / static_cast<double>(count);

            for (int64_t h = h_start; h < h_end; ++h) {
                for (int64_t w = w_start; w < w_end; ++w) {
                    const int64_t input_idx = ((n * C + c) * H_in + h) * W_in + w;
                    sycl::atomic_ref<double, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                     sycl::access::address_space::global_space> atomic_val(grad_in_ptr[input_idx]);
                    atomic_val.fetch_add(grad_val);
                }
            }
        }).wait();
    }
    else if (grad_output.dtype() == DType::Float16) {
        // Float16 backward uses float accumulation for numerical stability
        // We need to first accumulate in float, then convert
        const sycl::half* grad_out_ptr = get_data_ptr<const sycl::half>(grad_output);
        sycl::half* grad_in_ptr = get_data_ptr<sycl::half>(grad_input);

        // Use a temporary float buffer for atomic accumulation
        Tensor grad_input_f32({N, C, H_in, W_in}, DType::Float32, grad_output.device());
        queue.memset(const_cast<void*>(grad_input_f32.data_ptr()), 0, grad_input_f32.numel() * sizeof(float)).wait();
        float* grad_in_f32_ptr = get_data_ptr<float>(grad_input_f32);

        const int64_t total_size = N * C * H_out * W_out;
        queue.parallel_for<AdaptiveAvgPool2dBackwardKernelFloat16>(sycl::range<1>(total_size), [=](sycl::id<1> flat_idx) {
            int64_t temp = flat_idx;
            const int64_t w_out = temp % W_out;
            temp /= W_out;
            const int64_t h_out = temp % H_out;
            temp /= H_out;
            const int64_t c = temp % C;
            const int64_t n = temp / C;

            const int64_t h_start = (h_out * H_in) / H_out;
            const int64_t h_end = ((h_out + 1) * H_in) / H_out;
            const int64_t w_start = (w_out * W_in) / W_out;
            const int64_t w_end = ((w_out + 1) * W_in) / W_out;

            const int64_t count = (h_end - h_start) * (w_end - w_start);
            const float grad_val = static_cast<float>(grad_out_ptr[flat_idx]) / static_cast<float>(count);

            for (int64_t h = h_start; h < h_end; ++h) {
                for (int64_t w = w_start; w < w_end; ++w) {
                    const int64_t input_idx = ((n * C + c) * H_in + h) * W_in + w;
                    sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                     sycl::access::address_space::global_space> atomic_val(grad_in_f32_ptr[input_idx]);
                    atomic_val.fetch_add(grad_val);
                }
            }
        }).wait();

        // Convert float result back to half
        const int64_t total_input = N * C * H_in * W_in;
        queue.parallel_for(sycl::range<1>(total_input), [=](sycl::id<1> idx) {
            grad_in_ptr[idx] = sycl::half(grad_in_f32_ptr[idx]);
        }).wait();
    }
    else if (grad_output.dtype() == DType::BFloat16) {
        // BFloat16 backward uses float accumulation for numerical stability
        const uint16_t* grad_out_ptr = get_data_ptr<const uint16_t>(grad_output);
        uint16_t* grad_in_ptr = get_data_ptr<uint16_t>(grad_input);

        // Use a temporary float buffer for atomic accumulation
        Tensor grad_input_f32({N, C, H_in, W_in}, DType::Float32, grad_output.device());
        queue.memset(const_cast<void*>(grad_input_f32.data_ptr()), 0, grad_input_f32.numel() * sizeof(float)).wait();
        float* grad_in_f32_ptr = get_data_ptr<float>(grad_input_f32);

        const int64_t total_size = N * C * H_out * W_out;
        queue.parallel_for<AdaptiveAvgPool2dBackwardKernelBFloat16>(sycl::range<1>(total_size), [=](sycl::id<1> flat_idx) {
            int64_t temp = flat_idx;
            const int64_t w_out = temp % W_out;
            temp /= W_out;
            const int64_t h_out = temp % H_out;
            temp /= H_out;
            const int64_t c = temp % C;
            const int64_t n = temp / C;

            const int64_t h_start = (h_out * H_in) / H_out;
            const int64_t h_end = ((h_out + 1) * H_in) / H_out;
            const int64_t w_start = (w_out * W_in) / W_out;
            const int64_t w_end = ((w_out + 1) * W_in) / W_out;

            const int64_t count = (h_end - h_start) * (w_end - w_start);
            const float grad_val = bf16_to_f32(grad_out_ptr[flat_idx]) / static_cast<float>(count);

            for (int64_t h = h_start; h < h_end; ++h) {
                for (int64_t w = w_start; w < w_end; ++w) {
                    const int64_t input_idx = ((n * C + c) * H_in + h) * W_in + w;
                    sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                     sycl::access::address_space::global_space> atomic_val(grad_in_f32_ptr[input_idx]);
                    atomic_val.fetch_add(grad_val);
                }
            }
        }).wait();

        // Convert float result back to BFloat16
        const int64_t total_input = N * C * H_in * W_in;
        queue.parallel_for(sycl::range<1>(total_input), [=](sycl::id<1> idx) {
            grad_in_ptr[idx] = f32_to_bf16(grad_in_f32_ptr[idx]);
        }).wait();
    }
    else {
        throw std::runtime_error("Unsupported dtype for adaptive_avgpool2d_backward");
    }

    return grad_input;
}

// AdaptiveMaxPool2d - always pure SYCL
auto adaptive_maxpool2d_forward(const Tensor& input, int64_t output_h, int64_t output_w,
                                 sycl::queue& queue) -> Tensor {
    auto shape = input.shape();
    if (shape.size() != 4) {
        throw std::invalid_argument("AdaptiveMaxPool2d requires 4D input (N, C, H, W)");
    }

    const int64_t N = shape[0];
    const int64_t C = shape[1];
    const int64_t H_in = shape[2];
    const int64_t W_in = shape[3];

    Tensor output({N, C, output_h, output_w}, input.dtype(), input.device());

    if (input.dtype() == DType::Float32) {
        const float* in_ptr = get_data_ptr<const float>(input);
        float* out_ptr = get_data_ptr<float>(output);

        const int64_t total_size = N * C * output_h * output_w;
        queue.parallel_for<AdaptiveMaxPool2dKernelFloat32>(sycl::range<1>(total_size), [=](sycl::id<1> flat_idx) {
            int64_t temp = flat_idx;
            const int64_t w_out = temp % output_w;
            temp /= output_w;
            const int64_t h_out = temp % output_h;
            temp /= output_h;
            const int64_t c = temp % C;
            const int64_t n = temp / C;

            // Calculate input region
            const int64_t h_start = (h_out * H_in) / output_h;
            const int64_t h_end = ((h_out + 1) * H_in) / output_h;
            const int64_t w_start = (w_out * W_in) / output_w;
            const int64_t w_end = ((w_out + 1) * W_in) / output_w;

            float max_val = -3.4028235e+38f;

            for (int64_t h = h_start; h < h_end; ++h) {
                for (int64_t w = w_start; w < w_end; ++w) {
                    float val = in_ptr[((n * C + c) * H_in + h) * W_in + w];
                    max_val = sycl::fmax(max_val, val);
                }
            }

            out_ptr[((n * C + c) * output_h + h_out) * output_w + w_out] = max_val;
        }).wait();
    }
    else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        double* out_ptr = get_data_ptr<double>(output);

        const int64_t total_size = N * C * output_h * output_w;
        queue.parallel_for<AdaptiveMaxPool2dKernelFloat64>(sycl::range<1>(total_size), [=](sycl::id<1> flat_idx) {
            int64_t temp = flat_idx;
            const int64_t w_out = temp % output_w;
            temp /= output_w;
            const int64_t h_out = temp % output_h;
            temp /= output_h;
            const int64_t c = temp % C;
            const int64_t n = temp / C;

            const int64_t h_start = (h_out * H_in) / output_h;
            const int64_t h_end = ((h_out + 1) * H_in) / output_h;
            const int64_t w_start = (w_out * W_in) / output_w;
            const int64_t w_end = ((w_out + 1) * W_in) / output_w;

            double max_val = -1.7976931348623157e+308;

            for (int64_t h = h_start; h < h_end; ++h) {
                for (int64_t w = w_start; w < w_end; ++w) {
                    double val = in_ptr[((n * C + c) * H_in + h) * W_in + w];
                    max_val = sycl::fmax(max_val, val);
                }
            }

            out_ptr[((n * C + c) * output_h + h_out) * output_w + w_out] = max_val;
        }).wait();
    }
    else if (input.dtype() == DType::Float16) {
        const sycl::half* in_ptr = get_data_ptr<const sycl::half>(input);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);

        const int64_t total_size = N * C * output_h * output_w;
        queue.parallel_for<AdaptiveMaxPool2dKernelFloat16>(sycl::range<1>(total_size), [=](sycl::id<1> flat_idx) {
            int64_t temp = flat_idx;
            const int64_t w_out = temp % output_w;
            temp /= output_w;
            const int64_t h_out = temp % output_h;
            temp /= output_h;
            const int64_t c = temp % C;
            const int64_t n = temp / C;

            // Calculate input region
            const int64_t h_start = (h_out * H_in) / output_h;
            const int64_t h_end = ((h_out + 1) * H_in) / output_h;
            const int64_t w_start = (w_out * W_in) / output_w;
            const int64_t w_end = ((w_out + 1) * W_in) / output_w;

            // Use float for max computation
            float max_val = -3.4028235e+38f;

            for (int64_t h = h_start; h < h_end; ++h) {
                for (int64_t w = w_start; w < w_end; ++w) {
                    float val = static_cast<float>(in_ptr[((n * C + c) * H_in + h) * W_in + w]);
                    max_val = sycl::fmax(max_val, val);
                }
            }

            out_ptr[((n * C + c) * output_h + h_out) * output_w + w_out] = sycl::half(max_val);
        }).wait();
    }
    else if (input.dtype() == DType::BFloat16) {
        const uint16_t* in_ptr = get_data_ptr<const uint16_t>(input);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);

        const int64_t total_size = N * C * output_h * output_w;
        queue.parallel_for<AdaptiveMaxPool2dKernelBFloat16>(sycl::range<1>(total_size), [=](sycl::id<1> flat_idx) {
            int64_t temp = flat_idx;
            const int64_t w_out = temp % output_w;
            temp /= output_w;
            const int64_t h_out = temp % output_h;
            temp /= output_h;
            const int64_t c = temp % C;
            const int64_t n = temp / C;

            // Calculate input region
            const int64_t h_start = (h_out * H_in) / output_h;
            const int64_t h_end = ((h_out + 1) * H_in) / output_h;
            const int64_t w_start = (w_out * W_in) / output_w;
            const int64_t w_end = ((w_out + 1) * W_in) / output_w;

            // Use float for max computation
            float max_val = -3.4028235e+38f;

            for (int64_t h = h_start; h < h_end; ++h) {
                for (int64_t w = w_start; w < w_end; ++w) {
                    float val = bf16_to_f32(in_ptr[((n * C + c) * H_in + h) * W_in + w]);
                    max_val = sycl::fmax(max_val, val);
                }
            }

            out_ptr[((n * C + c) * output_h + h_out) * output_w + w_out] = f32_to_bf16(max_val);
        }).wait();
    }
    else {
        throw std::runtime_error("Unsupported dtype for adaptive_maxpool2d_forward");
    }

    return output;
}

/**
 * @brief Average pooling 2D operation wrapper.
 *
 * Applies 2D average pooling over an input signal.
 * OpAttributes wrapper for avg_pool2d operation.
 *
 * @param input Input tensor (4D: batch, channels, height, width)
 * @param attrs Operation attributes containing kernel_size, stride, padding
 * @param queue SYCL queue for execution
 * @return Tensor Pooled output tensor
 */
auto avg_pool2d_kernel(const Tensor& input, const OpAttributes& attrs, sycl::queue& queue) -> Tensor {
    if (!attrs.contains("kernel_size")) {
        throw std::invalid_argument("avg_pool2d: 'kernel_size' attribute is required");
    }

    int64_t kernel_size = std::stoll(attrs.at("kernel_size"));
    int64_t stride = attrs.contains("stride") ? std::stoll(attrs.at("stride")) : kernel_size;
    int64_t padding = attrs.contains("padding") ? std::stoll(attrs.at("padding")) : 0;
    bool count_include_pad = attrs.contains("count_include_pad") && attrs.at("count_include_pad") == "1";

    return avgpool2d_forward(input, kernel_size, stride, padding, count_include_pad, queue);
}

/**
 * @brief Max pooling 2D operation wrapper.
 *
 * Applies 2D max pooling over an input signal.
 * OpAttributes wrapper for max_pool2d operation.
 * Returns both output and indices tensors for backward pass support.
 *
 * @param input Input tensor (4D: batch, channels, height, width)
 * @param attrs Operation attributes containing kernel_size, stride, padding
 * @param queue SYCL queue for execution
 * @return std::pair<Tensor, Tensor> Pooled output tensor and indices tensor
 */
auto max_pool2d_kernel(const Tensor& input, const OpAttributes& attrs, sycl::queue& queue) -> std::pair<Tensor, Tensor> {
    if (!attrs.contains("kernel_size")) {
        throw std::invalid_argument("max_pool2d: 'kernel_size' attribute is required");
    }

    int64_t kernel_size = std::stoll(attrs.at("kernel_size"));
    int64_t stride = attrs.contains("stride") ? std::stoll(attrs.at("stride")) : kernel_size;
    int64_t padding = attrs.contains("padding") ? std::stoll(attrs.at("padding")) : 0;
    int64_t dilation = attrs.contains("dilation") ? std::stoll(attrs.at("dilation")) : 1;

    return maxpool2d_forward_with_indices(input, kernel_size, stride, padding, dilation, queue);
}

/**
 * @brief Adaptive average pooling 2D operation wrapper.
 *
 * Applies 2D adaptive average pooling over an input signal.
 * OpAttributes wrapper for adaptive_avg_pool2d operation.
 *
 * @param input Input tensor (4D: batch, channels, height, width)
 * @param attrs Operation attributes containing output_size
 * @param queue SYCL queue for execution
 * @return Tensor Pooled output tensor
 */
auto adaptive_avg_pool2d_kernel(const Tensor& input, const OpAttributes& attrs, sycl::queue& queue) -> Tensor {
    int64_t output_h = 1, output_w = 1;

    // Support both formats: "output_size" (H,W string) and "output_h"/"output_w" (separate integers)
    if (attrs.contains("output_size")) {
        // Parse output_size (format: "H,W")
        std::string output_size_str = attrs.at("output_size");
        size_t comma_pos = output_size_str.find(',');
        if (comma_pos == std::string::npos) {
            throw std::invalid_argument("adaptive_avg_pool2d: output_size must be in format 'H,W'");
        }
        output_h = std::stoll(output_size_str.substr(0, comma_pos));
        output_w = std::stoll(output_size_str.substr(comma_pos + 1));
    } else if (attrs.contains("output_h") && attrs.contains("output_w")) {
        output_h = std::stoll(attrs.at("output_h"));
        output_w = std::stoll(attrs.at("output_w"));
    } else if (attrs.contains("output_h")) {
        // Square output if only output_h is provided
        output_h = std::stoll(attrs.at("output_h"));
        output_w = output_h;
    } else {
        throw std::invalid_argument("adaptive_avg_pool2d: 'output_size' or 'output_h'/'output_w' attributes are required");
    }

    return adaptive_avgpool2d_forward(input, output_h, output_w, queue);
}

/**
 * @brief Adaptive max pooling 2D operation wrapper.
 *
 * Applies 2D adaptive max pooling over an input signal.
 * OpAttributes wrapper for adaptive_max_pool2d operation.
 *
 * @param input Input tensor (4D: batch, channels, height, width)
 * @param attrs Operation attributes containing output_size
 * @param queue SYCL queue for execution
 * @return Tensor Pooled output tensor
 */
auto adaptive_max_pool2d_kernel(const Tensor& input, const OpAttributes& attrs, sycl::queue& queue) -> Tensor {
    int64_t output_h = 1, output_w = 1;

    // Support both formats: "output_size" (H,W string) and "output_h"/"output_w" (separate integers)
    if (attrs.contains("output_size")) {
        // Parse output_size (format: "H,W")
        std::string output_size_str = attrs.at("output_size");
        size_t comma_pos = output_size_str.find(',');
        if (comma_pos == std::string::npos) {
            throw std::invalid_argument("adaptive_max_pool2d: output_size must be in format 'H,W'");
        }
        output_h = std::stoll(output_size_str.substr(0, comma_pos));
        output_w = std::stoll(output_size_str.substr(comma_pos + 1));
    } else if (attrs.contains("output_h") && attrs.contains("output_w")) {
        output_h = std::stoll(attrs.at("output_h"));
        output_w = std::stoll(attrs.at("output_w"));
    } else if (attrs.contains("output_h")) {
        // Square output if only output_h is provided
        output_h = std::stoll(attrs.at("output_h"));
        output_w = output_h;
    } else {
        throw std::invalid_argument("adaptive_max_pool2d: 'output_size' or 'output_h'/'output_w' attributes are required");
    }

    return adaptive_maxpool2d_forward(input, output_h, output_w, queue);
}

/**
 * @brief Average pooling 2D backward operation.
 *
 * Computes gradients for average pooling by distributing the gradient
 * evenly across the pooling window for each output position.
 *
 * @param grad_output Gradient from next layer (4D: batch, channels, height_out, width_out)
 * @param input Original input tensor (used for shape information)
 * @param attrs Operation attributes containing kernel_size, stride, padding
 * @param queue SYCL queue for execution
 * @return Tensor Gradient with respect to input
 */
auto avg_pool2d_backward_kernel(const Tensor& grad_output, const Tensor& input,
                                const OpAttributes& attrs, sycl::queue& queue) -> Tensor {
    if (!attrs.contains("kernel_size")) {
        throw std::invalid_argument("avg_pool2d_backward: 'kernel_size' attribute is required");
    }

    auto grad_shape = grad_output.shape();
    auto input_shape = input.shape();

    if (grad_shape.size() != 4 || input_shape.size() != 4) {
        throw std::invalid_argument("avg_pool2d_backward requires 4D inputs (N, C, H, W)");
    }

    int64_t kernel_size = std::stoll(attrs.at("kernel_size"));
    int64_t stride = attrs.contains("stride") ? std::stoll(attrs.at("stride")) : kernel_size;
    int64_t padding = attrs.contains("padding") ? std::stoll(attrs.at("padding")) : 0;
    bool count_include_pad = attrs.contains("count_include_pad") && attrs.at("count_include_pad") == "1";

    const int64_t N = input_shape[0];
    const int64_t C = input_shape[1];
    const int64_t H_in = input_shape[2];
    const int64_t W_in = input_shape[3];

    const int64_t H_out = grad_shape[2];
    const int64_t W_out = grad_shape[3];

    // Create gradient input tensor (same shape as original input)
    Tensor grad_input({N, C, H_in, W_in}, input.dtype(), input.device());

    if (input.dtype() == DType::Float32) {
        const float* grad_out_ptr = get_data_ptr<const float>(grad_output);
        float* grad_in_ptr = get_data_ptr<float>(grad_input);

        // Initialize grad_input to zero
        const int64_t input_size = N * C * H_in * W_in;
        queue.fill(grad_in_ptr, 0.0f, input_size).wait();

        // For each output position, distribute gradient to input positions
        const int64_t total_output_size = N * C * H_out * W_out;
        queue.parallel_for<AvgPool2dBackwardKernelFloat32>(sycl::range<1>(total_output_size),
            [=](sycl::id<1> flat_idx) {
                int64_t temp = flat_idx;
                const int64_t w_out = temp % W_out;
                temp /= W_out;
                const int64_t h_out = temp % H_out;
                temp /= H_out;
                const int64_t c = temp % C;
                const int64_t n = temp / C;

                const float grad_val = grad_out_ptr[((n * C + c) * H_out + h_out) * W_out + w_out];

                // Count valid positions in pooling window
                int64_t count = 0;
                for (int64_t kh = 0; kh < kernel_size; ++kh) {
                    for (int64_t kw = 0; kw < kernel_size; ++kw) {
                        int64_t h_in = h_out * stride - padding + kh;
                        int64_t w_in = w_out * stride - padding + kw;

                        if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                            count++;
                        } else if (count_include_pad) {
                            count++;
                        }
                    }
                }

                // Distribute gradient evenly
                const float grad_per_input = count > 0 ? grad_val / static_cast<float>(count) : 0.0f;

                for (int64_t kh = 0; kh < kernel_size; ++kh) {
                    for (int64_t kw = 0; kw < kernel_size; ++kw) {
                        int64_t h_in = h_out * stride - padding + kh;
                        int64_t w_in = w_out * stride - padding + kw;

                        if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                            int64_t input_idx = ((n * C + c) * H_in + h_in) * W_in + w_in;
                            sycl::atomic_ref<float, sycl::memory_order::relaxed,
                                           sycl::memory_scope::device> atomic_grad(grad_in_ptr[input_idx]);
                            atomic_grad += grad_per_input;
                        }
                    }
                }
        }).wait();
    }
    else if (input.dtype() == DType::Float64) {
        const double* grad_out_ptr = get_data_ptr<const double>(grad_output);
        double* grad_in_ptr = get_data_ptr<double>(grad_input);

        const int64_t input_size = N * C * H_in * W_in;
        queue.fill(grad_in_ptr, 0.0, input_size).wait();

        const int64_t total_output_size = N * C * H_out * W_out;
        queue.parallel_for<AvgPool2dBackwardKernelFloat64>(sycl::range<1>(total_output_size),
            [=](sycl::id<1> flat_idx) {
                int64_t temp = flat_idx;
                const int64_t w_out = temp % W_out;
                temp /= W_out;
                const int64_t h_out = temp % H_out;
                temp /= H_out;
                const int64_t c = temp % C;
                const int64_t n = temp / C;

                const double grad_val = grad_out_ptr[((n * C + c) * H_out + h_out) * W_out + w_out];

                int64_t count = 0;
                for (int64_t kh = 0; kh < kernel_size; ++kh) {
                    for (int64_t kw = 0; kw < kernel_size; ++kw) {
                        int64_t h_in = h_out * stride - padding + kh;
                        int64_t w_in = w_out * stride - padding + kw;

                        if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                            count++;
                        } else if (count_include_pad) {
                            count++;
                        }
                    }
                }

                const double grad_per_input = count > 0 ? grad_val / static_cast<double>(count) : 0.0;

                for (int64_t kh = 0; kh < kernel_size; ++kh) {
                    for (int64_t kw = 0; kw < kernel_size; ++kw) {
                        int64_t h_in = h_out * stride - padding + kh;
                        int64_t w_in = w_out * stride - padding + kw;

                        if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                            int64_t input_idx = ((n * C + c) * H_in + h_in) * W_in + w_in;
                            sycl::atomic_ref<double, sycl::memory_order::relaxed,
                                           sycl::memory_scope::device> atomic_grad(grad_in_ptr[input_idx]);
                            atomic_grad += grad_per_input;
                        }
                    }
                }
        }).wait();
    }
    else if (input.dtype() == DType::Float16) {
        const sycl::half* grad_out_ptr = get_data_ptr<const sycl::half>(grad_output);
        sycl::half* grad_in_ptr = get_data_ptr<sycl::half>(grad_input);

        // Use float32 intermediate buffer since atomic_ref<sycl::half> is not widely supported
        const int64_t input_size = N * C * H_in * W_in;
        float* acc_ptr = sycl::malloc_device<float>(input_size, queue);
        queue.fill(acc_ptr, 0.0f, input_size).wait();

        const int64_t total_output_size = N * C * H_out * W_out;
        queue.parallel_for<AvgPool2dBackwardKernelFloat16>(sycl::range<1>(total_output_size),
            [=](sycl::id<1> flat_idx) {
                int64_t temp = flat_idx;
                const int64_t w_out = temp % W_out;
                temp /= W_out;
                const int64_t h_out = temp % H_out;
                temp /= H_out;
                const int64_t c = temp % C;
                const int64_t n = temp / C;

                const float grad_val = static_cast<float>(grad_out_ptr[((n * C + c) * H_out + h_out) * W_out + w_out]);

                int64_t count = 0;
                for (int64_t kh = 0; kh < kernel_size; ++kh) {
                    for (int64_t kw = 0; kw < kernel_size; ++kw) {
                        int64_t h_in = h_out * stride - padding + kh;
                        int64_t w_in = w_out * stride - padding + kw;

                        if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                            count++;
                        } else if (count_include_pad) {
                            count++;
                        }
                    }
                }

                const float grad_per_input = count > 0 ? grad_val / static_cast<float>(count) : 0.0f;

                for (int64_t kh = 0; kh < kernel_size; ++kh) {
                    for (int64_t kw = 0; kw < kernel_size; ++kw) {
                        int64_t h_in = h_out * stride - padding + kh;
                        int64_t w_in = w_out * stride - padding + kw;

                        if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                            int64_t input_idx = ((n * C + c) * H_in + h_in) * W_in + w_in;
                            sycl::atomic_ref<float, sycl::memory_order::relaxed,
                                           sycl::memory_scope::device> atomic_grad(acc_ptr[input_idx]);
                            atomic_grad += grad_per_input;
                        }
                    }
                }
        }).wait();

        // Convert float32 accumulator back to Float16
        queue.parallel_for(sycl::range<1>(input_size), [=](sycl::id<1> idx) {
            grad_in_ptr[idx] = sycl::half(acc_ptr[idx]);
        }).wait();

        sycl::free(acc_ptr, queue);
    }
    else if (input.dtype() == DType::BFloat16) {
        const uint16_t* grad_out_ptr = get_data_ptr<const uint16_t>(grad_output);
        uint16_t* grad_in_ptr = get_data_ptr<uint16_t>(grad_input);

        // Use float32 intermediate buffer since atomic_ref<uint16_t> is not suitable for BFloat16
        const int64_t input_size = N * C * H_in * W_in;
        float* acc_ptr = sycl::malloc_device<float>(input_size, queue);
        queue.fill(acc_ptr, 0.0f, input_size).wait();

        const int64_t total_output_size = N * C * H_out * W_out;
        queue.parallel_for<AvgPool2dBackwardKernelBFloat16>(sycl::range<1>(total_output_size),
            [=](sycl::id<1> flat_idx) {
                int64_t temp = flat_idx;
                const int64_t w_out = temp % W_out;
                temp /= W_out;
                const int64_t h_out = temp % H_out;
                temp /= H_out;
                const int64_t c = temp % C;
                const int64_t n = temp / C;

                const float grad_val = bf16_to_f32(grad_out_ptr[((n * C + c) * H_out + h_out) * W_out + w_out]);

                int64_t count = 0;
                for (int64_t kh = 0; kh < kernel_size; ++kh) {
                    for (int64_t kw = 0; kw < kernel_size; ++kw) {
                        int64_t h_in = h_out * stride - padding + kh;
                        int64_t w_in = w_out * stride - padding + kw;

                        if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                            count++;
                        } else if (count_include_pad) {
                            count++;
                        }
                    }
                }

                const float grad_per_input = count > 0 ? grad_val / static_cast<float>(count) : 0.0f;

                for (int64_t kh = 0; kh < kernel_size; ++kh) {
                    for (int64_t kw = 0; kw < kernel_size; ++kw) {
                        int64_t h_in = h_out * stride - padding + kh;
                        int64_t w_in = w_out * stride - padding + kw;

                        if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                            int64_t input_idx = ((n * C + c) * H_in + h_in) * W_in + w_in;
                            sycl::atomic_ref<float, sycl::memory_order::relaxed,
                                           sycl::memory_scope::device> atomic_grad(acc_ptr[input_idx]);
                            atomic_grad += grad_per_input;
                        }
                    }
                }
        }).wait();

        // Convert float32 accumulator back to BFloat16
        queue.parallel_for(sycl::range<1>(input_size), [=](sycl::id<1> idx) {
            grad_in_ptr[idx] = f32_to_bf16(acc_ptr[idx]);
        }).wait();

        sycl::free(acc_ptr, queue);
    }
    else {
        throw std::runtime_error("Unsupported dtype for avg_pool2d_backward");
    }

    return grad_input;
}

/**
 * @brief Max pooling 2D backward operation.
 *
 * Computes gradients for max pooling by routing gradient only to the
 * maximum element in each pooling window. Max positions are recomputed
 * from the original input.
 *
 * @param grad_output Gradient from next layer (4D: batch, channels, height_out, width_out)
 * @param input Original input tensor (used to find max positions)
 * @param attrs Operation attributes containing kernel_size, stride, padding, dilation
 * @param queue SYCL queue for execution
 * @return Tensor Gradient with respect to input
 */
auto max_pool2d_backward_kernel(const Tensor& grad_output, const Tensor& input,
                                const OpAttributes& attrs, sycl::queue& queue) -> Tensor {
    if (!attrs.contains("kernel_size")) {
        throw std::invalid_argument("max_pool2d_backward: 'kernel_size' attribute is required");
    }

    auto grad_shape = grad_output.shape();
    auto input_shape = input.shape();

    if (grad_shape.size() != 4 || input_shape.size() != 4) {
        throw std::invalid_argument("max_pool2d_backward requires 4D inputs (N, C, H, W)");
    }

    int64_t kernel_size = std::stoll(attrs.at("kernel_size"));
    int64_t stride = attrs.contains("stride") ? std::stoll(attrs.at("stride")) : kernel_size;
    int64_t padding = attrs.contains("padding") ? std::stoll(attrs.at("padding")) : 0;
    int64_t dilation = attrs.contains("dilation") ? std::stoll(attrs.at("dilation")) : 1;

    const int64_t N = input_shape[0];
    const int64_t C = input_shape[1];
    const int64_t H_in = input_shape[2];
    const int64_t W_in = input_shape[3];

    const int64_t H_out = grad_shape[2];
    const int64_t W_out = grad_shape[3];

    // Create gradient input tensor (same shape as original input)
    Tensor grad_input({N, C, H_in, W_in}, input.dtype(), input.device());

    if (input.dtype() == DType::Float32) {
        const float* in_ptr = get_data_ptr<const float>(input);
        const float* grad_out_ptr = get_data_ptr<const float>(grad_output);
        float* grad_in_ptr = get_data_ptr<float>(grad_input);

        // Initialize grad_input to zero
        const int64_t input_size = N * C * H_in * W_in;
        queue.fill(grad_in_ptr, 0.0f, input_size).wait();

        // For each output position, find max and route gradient
        const int64_t total_output_size = N * C * H_out * W_out;
        queue.parallel_for<MaxPool2dBackwardKernelFloat32>(sycl::range<1>(total_output_size),
            [=](sycl::id<1> flat_idx) {
                int64_t temp = flat_idx;
                const int64_t w_out = temp % W_out;
                temp /= W_out;
                const int64_t h_out = temp % H_out;
                temp /= H_out;
                const int64_t c = temp % C;
                const int64_t n = temp / C;

                const float grad_val = grad_out_ptr[((n * C + c) * H_out + h_out) * W_out + w_out];

                // Find max position in pooling window
                float max_val = -3.4028235e+38f;
                int64_t max_h = -1;
                int64_t max_w = -1;

                for (int64_t kh = 0; kh < kernel_size; ++kh) {
                    for (int64_t kw = 0; kw < kernel_size; ++kw) {
                        int64_t h_in = h_out * stride - padding + kh * dilation;
                        int64_t w_in = w_out * stride - padding + kw * dilation;

                        if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                            float val = in_ptr[((n * C + c) * H_in + h_in) * W_in + w_in];
                            if (val > max_val) {
                                max_val = val;
                                max_h = h_in;
                                max_w = w_in;
                            }
                        }
                    }
                }

                // Route gradient to max position
                if (max_h >= 0 && max_w >= 0) {
                    int64_t input_idx = ((n * C + c) * H_in + max_h) * W_in + max_w;
                    sycl::atomic_ref<float, sycl::memory_order::relaxed,
                                   sycl::memory_scope::device> atomic_grad(grad_in_ptr[input_idx]);
                    atomic_grad += grad_val;
                }
        }).wait();
    }
    else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        const double* grad_out_ptr = get_data_ptr<const double>(grad_output);
        double* grad_in_ptr = get_data_ptr<double>(grad_input);

        const int64_t input_size = N * C * H_in * W_in;
        queue.fill(grad_in_ptr, 0.0, input_size).wait();

        const int64_t total_output_size = N * C * H_out * W_out;
        queue.parallel_for<MaxPool2dBackwardKernelFloat64>(sycl::range<1>(total_output_size),
            [=](sycl::id<1> flat_idx) {
                int64_t temp = flat_idx;
                const int64_t w_out = temp % W_out;
                temp /= W_out;
                const int64_t h_out = temp % H_out;
                temp /= H_out;
                const int64_t c = temp % C;
                const int64_t n = temp / C;

                const double grad_val = grad_out_ptr[((n * C + c) * H_out + h_out) * W_out + w_out];

                double max_val = -1.7976931348623157e+308;
                int64_t max_h = -1;
                int64_t max_w = -1;

                for (int64_t kh = 0; kh < kernel_size; ++kh) {
                    for (int64_t kw = 0; kw < kernel_size; ++kw) {
                        int64_t h_in = h_out * stride - padding + kh * dilation;
                        int64_t w_in = w_out * stride - padding + kw * dilation;

                        if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                            double val = in_ptr[((n * C + c) * H_in + h_in) * W_in + w_in];
                            if (val > max_val) {
                                max_val = val;
                                max_h = h_in;
                                max_w = w_in;
                            }
                        }
                    }
                }

                if (max_h >= 0 && max_w >= 0) {
                    int64_t input_idx = ((n * C + c) * H_in + max_h) * W_in + max_w;
                    sycl::atomic_ref<double, sycl::memory_order::relaxed,
                                   sycl::memory_scope::device> atomic_grad(grad_in_ptr[input_idx]);
                    atomic_grad += grad_val;
                }
        }).wait();
    }
    else if (input.dtype() == DType::BFloat16) {
        const uint16_t* in_ptr = get_data_ptr<const uint16_t>(input);
        const uint16_t* grad_out_ptr = get_data_ptr<const uint16_t>(grad_output);
        uint16_t* grad_in_ptr = get_data_ptr<uint16_t>(grad_input);

        // Use float32 intermediate buffer for atomic accumulation
        const int64_t input_size = N * C * H_in * W_in;
        float* acc_ptr = sycl::malloc_device<float>(input_size, queue);
        queue.fill(acc_ptr, 0.0f, input_size).wait();

        const int64_t total_output_size = N * C * H_out * W_out;
        queue.parallel_for<MaxPool2dBackwardKernelBFloat16>(sycl::range<1>(total_output_size),
            [=](sycl::id<1> flat_idx) {
                int64_t temp = flat_idx;
                const int64_t w_out = temp % W_out;
                temp /= W_out;
                const int64_t h_out = temp % H_out;
                temp /= H_out;
                const int64_t c = temp % C;
                const int64_t n = temp / C;

                const float grad_val = bf16_to_f32(grad_out_ptr[((n * C + c) * H_out + h_out) * W_out + w_out]);

                // Find max position in pooling window
                float max_val = -3.4028235e+38f;
                int64_t max_h = -1;
                int64_t max_w = -1;

                for (int64_t kh = 0; kh < kernel_size; ++kh) {
                    for (int64_t kw = 0; kw < kernel_size; ++kw) {
                        int64_t h_in = h_out * stride - padding + kh * dilation;
                        int64_t w_in = w_out * stride - padding + kw * dilation;

                        if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                            float val = bf16_to_f32(in_ptr[((n * C + c) * H_in + h_in) * W_in + w_in]);
                            if (val > max_val) {
                                max_val = val;
                                max_h = h_in;
                                max_w = w_in;
                            }
                        }
                    }
                }

                // Route gradient to max position
                if (max_h >= 0 && max_w >= 0) {
                    int64_t input_idx = ((n * C + c) * H_in + max_h) * W_in + max_w;
                    sycl::atomic_ref<float, sycl::memory_order::relaxed,
                                   sycl::memory_scope::device> atomic_grad(acc_ptr[input_idx]);
                    atomic_grad += grad_val;
                }
        }).wait();

        // Convert float32 accumulator back to BFloat16
        queue.parallel_for(sycl::range<1>(input_size), [=](sycl::id<1> idx) {
            grad_in_ptr[idx] = f32_to_bf16(acc_ptr[idx]);
        }).wait();

        sycl::free(acc_ptr, queue);
    }
    else {
        throw std::runtime_error("Unsupported dtype for max_pool2d_backward");
    }

    return grad_input;
}

/**
 * @brief Adaptive average pooling 2D backward operation wrapper.
 *
 * Computes gradients for adaptive average pooling.
 * OpAttributes wrapper for adaptive_avg_pool2d_backward operation.
 *
 * @param grad_output Gradient from next layer (4D: batch, channels, height_out, width_out)
 * @param input Original input tensor (used for shape information)
 * @param attrs Operation attributes (optional, H_in and W_in can be derived from input)
 * @param queue SYCL queue for execution
 * @return Tensor Gradient with respect to input
 */
auto adaptive_avg_pool2d_backward_kernel(const Tensor& grad_output, const Tensor& input,
                                          const OpAttributes& attrs, sycl::queue& queue) -> Tensor {
    auto input_shape = input.shape();
    if (input_shape.size() != 4) {
        throw std::invalid_argument("adaptive_avg_pool2d_backward requires 4D input (N, C, H, W)");
    }

    int64_t H_in = input_shape[2];
    int64_t W_in = input_shape[3];

    // Allow override from attrs if provided
    if (attrs.contains("input_h")) {
        H_in = std::stoll(attrs.at("input_h"));
    }
    if (attrs.contains("input_w")) {
        W_in = std::stoll(attrs.at("input_w"));
    }

    return adaptive_avgpool2d_backward(grad_output, H_in, W_in, queue);
}

/**
 * @brief Max pooling 2D backward operation using stored indices.
 *
 * Computes gradients for max pooling by routing gradient to the
 * max element positions stored in the indices tensor.
 *
 * @param grad_output Gradient from next layer (4D: batch, channels, height_out, width_out)
 * @param indices Indices tensor from forward pass (same shape as grad_output)
 * @param H_in Original input height
 * @param W_in Original input width
 * @param queue SYCL queue for execution
 * @return Tensor Gradient with respect to input
 */
auto max_pool2d_backward_with_indices(const Tensor& grad_output, const Tensor& indices,
                                       int64_t H_in, int64_t W_in, sycl::queue& queue) -> Tensor {
    auto grad_shape = grad_output.shape();
    if (grad_shape.size() != 4) {
        throw std::invalid_argument("max_pool2d_backward_with_indices requires 4D grad_output (N, C, H, W)");
    }

    const int64_t N = grad_shape[0];
    const int64_t C = grad_shape[1];
    const int64_t H_out = grad_shape[2];
    const int64_t W_out = grad_shape[3];

    // Create gradient input tensor
    Tensor grad_input({N, C, H_in, W_in}, grad_output.dtype(), grad_output.device());

    const int64_t input_size = N * C * H_in * W_in;
    const int64_t output_size = N * C * H_out * W_out;

    if (grad_output.dtype() == DType::Float32) {
        const float* grad_out_ptr = get_data_ptr<const float>(grad_output);
        const float* idx_ptr = get_data_ptr<const float>(indices);  // Indices stored as same dtype
        float* grad_in_ptr = get_data_ptr<float>(grad_input);

        // Initialize grad_input to zero
        queue.fill(grad_in_ptr, 0.0f, input_size).wait();

        // For each output position, route gradient to the stored index
        queue.parallel_for<MaxPool2dBackwardWithIndicesKernelFloat32>(sycl::range<1>(output_size),
            [=](sycl::id<1> flat_idx) {
                int64_t temp = flat_idx;
                const int64_t w_out = temp % W_out;
                temp /= W_out;
                const int64_t h_out = temp % H_out;
                temp /= H_out;
                const int64_t c = temp % C;
                const int64_t n = temp / C;

                const float grad_val = grad_out_ptr[flat_idx];
                const int64_t max_idx = static_cast<int64_t>(idx_ptr[flat_idx]);

                // Convert flat index to spatial index
                const int64_t h_in = max_idx / W_in;
                const int64_t w_in = max_idx % W_in;

                if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                    const int64_t input_idx = ((n * C + c) * H_in + h_in) * W_in + w_in;
                    sycl::atomic_ref<float, sycl::memory_order::relaxed,
                                   sycl::memory_scope::device,
                                   sycl::access::address_space::global_space> atomic_grad(grad_in_ptr[input_idx]);
                    atomic_grad.fetch_add(grad_val);
                }
        }).wait();
    }
    else if (grad_output.dtype() == DType::Float64) {
        const double* grad_out_ptr = get_data_ptr<const double>(grad_output);
        const double* idx_ptr = get_data_ptr<const double>(indices);
        double* grad_in_ptr = get_data_ptr<double>(grad_input);

        queue.fill(grad_in_ptr, 0.0, input_size).wait();

        queue.parallel_for<MaxPool2dBackwardWithIndicesKernelFloat64>(sycl::range<1>(output_size),
            [=](sycl::id<1> flat_idx) {
                int64_t temp = flat_idx;
                const int64_t w_out = temp % W_out;
                temp /= W_out;
                const int64_t h_out = temp % H_out;
                temp /= H_out;
                const int64_t c = temp % C;
                const int64_t n = temp / C;

                const double grad_val = grad_out_ptr[flat_idx];
                const int64_t max_idx = static_cast<int64_t>(idx_ptr[flat_idx]);

                const int64_t h_in = max_idx / W_in;
                const int64_t w_in = max_idx % W_in;

                if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                    const int64_t input_idx = ((n * C + c) * H_in + h_in) * W_in + w_in;
                    sycl::atomic_ref<double, sycl::memory_order::relaxed,
                                   sycl::memory_scope::device,
                                   sycl::access::address_space::global_space> atomic_grad(grad_in_ptr[input_idx]);
                    atomic_grad.fetch_add(grad_val);
                }
        }).wait();
    }
    else if (grad_output.dtype() == DType::Float16) {
        const sycl::half* grad_out_ptr = get_data_ptr<const sycl::half>(grad_output);
        const sycl::half* idx_ptr = get_data_ptr<const sycl::half>(indices);
        sycl::half* grad_in_ptr = get_data_ptr<sycl::half>(grad_input);

        // Use float accumulation buffer for Float16
        Tensor grad_input_float({N, C, H_in, W_in}, DType::Float32, grad_output.device());
        float* grad_in_float_ptr = get_data_ptr<float>(grad_input_float);

        queue.fill(grad_in_float_ptr, 0.0f, input_size).wait();

        queue.parallel_for<MaxPool2dBackwardWithIndicesKernelFloat16>(sycl::range<1>(output_size),
            [=](sycl::id<1> flat_idx) {
                int64_t temp = flat_idx;
                const int64_t w_out = temp % W_out;
                temp /= W_out;
                const int64_t h_out = temp % H_out;
                temp /= H_out;
                const int64_t c = temp % C;
                const int64_t n = temp / C;

                const float grad_val = static_cast<float>(grad_out_ptr[flat_idx]);
                const int64_t max_idx = static_cast<int64_t>(static_cast<float>(idx_ptr[flat_idx]));

                const int64_t h_in = max_idx / W_in;
                const int64_t w_in = max_idx % W_in;

                if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                    const int64_t input_idx = ((n * C + c) * H_in + h_in) * W_in + w_in;
                    sycl::atomic_ref<float, sycl::memory_order::relaxed,
                                   sycl::memory_scope::device,
                                   sycl::access::address_space::global_space> atomic_grad(grad_in_float_ptr[input_idx]);
                    atomic_grad.fetch_add(grad_val);
                }
        }).wait();

        // Convert back to Float16
        queue.parallel_for(sycl::range<1>(input_size), [=](sycl::id<1> i) {
            grad_in_ptr[i] = sycl::half(grad_in_float_ptr[i]);
        }).wait();
    }
    else if (grad_output.dtype() == DType::BFloat16) {
        const uint16_t* grad_out_ptr = get_data_ptr<const uint16_t>(grad_output);
        const uint16_t* idx_ptr = get_data_ptr<const uint16_t>(indices);
        uint16_t* grad_in_ptr = get_data_ptr<uint16_t>(grad_input);

        // Use float accumulation buffer for BFloat16
        Tensor grad_input_float({N, C, H_in, W_in}, DType::Float32, grad_output.device());
        float* grad_in_float_ptr = get_data_ptr<float>(grad_input_float);

        queue.fill(grad_in_float_ptr, 0.0f, input_size).wait();

        queue.parallel_for<MaxPool2dBackwardWithIndicesKernelBFloat16>(sycl::range<1>(output_size),
            [=](sycl::id<1> flat_idx) {
                int64_t temp = flat_idx;
                const int64_t w_out = temp % W_out;
                temp /= W_out;
                const int64_t h_out = temp % H_out;
                temp /= H_out;
                const int64_t c = temp % C;
                const int64_t n = temp / C;

                const float grad_val = bf16_to_f32(grad_out_ptr[flat_idx]);
                const int64_t max_idx = static_cast<int64_t>(bf16_to_f32(idx_ptr[flat_idx]));

                const int64_t h_in = max_idx / W_in;
                const int64_t w_in = max_idx % W_in;

                if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                    const int64_t input_idx = ((n * C + c) * H_in + h_in) * W_in + w_in;
                    sycl::atomic_ref<float, sycl::memory_order::relaxed,
                                   sycl::memory_scope::device,
                                   sycl::access::address_space::global_space> atomic_grad(grad_in_float_ptr[input_idx]);
                    atomic_grad.fetch_add(grad_val);
                }
        }).wait();

        // Convert back to BFloat16
        queue.parallel_for(sycl::range<1>(input_size), [=](sycl::id<1> i) {
            grad_in_ptr[i] = f32_to_bf16(grad_in_float_ptr[i]);
        }).wait();
    }
    else {
        throw std::runtime_error("Unsupported dtype for max_pool2d_backward_with_indices");
    }

    return grad_input;
}

} // namespace oneapi
} // namespace tenzor
