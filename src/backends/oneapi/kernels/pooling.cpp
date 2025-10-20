#include "tenzor/core/tensor.hpp"
#include <CL/sycl.hpp>
#include <limits>
#include <stdexcept>

#ifdef TENZOR_HAS_ONEDNN
#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_sycl.hpp>
#endif

namespace tenzor {
namespace oneapi {

// Helper function to get typed pointer from tensor
template<typename T>
inline auto get_data_ptr(const Tensor& t) -> T* {
    return static_cast<T*>(const_cast<void*>(t.data_ptr()));
}

#ifdef TENZOR_HAS_ONEDNN

// MaxPool2d forward using oneDNN
auto maxpool2d_forward(const Tensor& input, int64_t kernel_size, int64_t stride,
                       int64_t padding, int64_t dilation, sycl::queue& queue) -> Tensor {
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

    // Create pooling descriptor
    auto pool_desc = pooling_forward::desc(
        prop_kind::forward_inference,
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

    // Execute
    auto pool_prim = pooling_forward(pool_pd);
    pool_prim.execute(dnnl_stream, {
        {DNNL_ARG_SRC, src_mem},
        {DNNL_ARG_DST, dst_mem}
    });

    dnnl_stream.wait();

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

// MaxPool2d forward (pure SYCL)
auto maxpool2d_forward(const Tensor& input, int64_t kernel_size, int64_t stride,
                       int64_t padding, int64_t dilation, sycl::queue& queue) -> Tensor {
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

    if (input.dtype() == DType::Float32) {
        const float* in_ptr = get_data_ptr<const float>(input);
        float* out_ptr = get_data_ptr<float>(output);

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

            for (int64_t kh = 0; kh < kernel_size; ++kh) {
                for (int64_t kw = 0; kw < kernel_size; ++kw) {
                    int64_t h_in = h_out * stride - padding + kh * dilation;
                    int64_t w_in = w_out * stride - padding + kw * dilation;

                    if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                        float val = in_ptr[((n * C + c) * H_in + h_in) * W_in + w_in];
                        max_val = sycl::fmax(max_val, val);
                    }
                }
            }

            out_ptr[((n * C + c) * H_out + h_out) * W_out + w_out] = max_val;
        }).wait();
    }
    else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        double* out_ptr = get_data_ptr<double>(output);

        const int64_t total_size = N * C * H_out * W_out;
        queue.parallel_for(sycl::range<1>(total_size), [=](sycl::id<1> flat_idx) {
            int64_t temp = flat_idx;
            const int64_t w_out = temp % W_out;
            temp /= W_out;
            const int64_t h_out = temp % H_out;
            temp /= H_out;
            const int64_t c = temp % C;
            const int64_t n = temp / C;

            double max_val = -1.7976931348623157e+308;

            for (int64_t kh = 0; kh < kernel_size; ++kh) {
                for (int64_t kw = 0; kw < kernel_size; ++kw) {
                    int64_t h_in = h_out * stride - padding + kh * dilation;
                    int64_t w_in = w_out * stride - padding + kw * dilation;

                    if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                        double val = in_ptr[((n * C + c) * H_in + h_in) * W_in + w_in];
                        max_val = sycl::fmax(max_val, val);
                    }
                }
            }

            out_ptr[((n * C + c) * H_out + h_out) * W_out + w_out] = max_val;
        }).wait();
    }
    else {
        throw std::runtime_error("Unsupported dtype for maxpool2d_forward");
    }

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
        queue.parallel_for(sycl::range<1>(total_size), [=](sycl::id<1> flat_idx) {
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
        queue.parallel_for(sycl::range<1>(total_size), [=](sycl::id<1> flat_idx) {
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
        queue.parallel_for(sycl::range<1>(total_size), [=](sycl::id<1> flat_idx) {
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
        queue.parallel_for(sycl::range<1>(total_size), [=](sycl::id<1> flat_idx) {
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
    else {
        throw std::runtime_error("Unsupported dtype for adaptive_avgpool2d_forward");
    }

    return output;
}

} // namespace oneapi
} // namespace tenzor
