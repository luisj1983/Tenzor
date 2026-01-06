/**
 * @file pooling.cpp
 * @brief CPU pooling kernel implementations
 */

#include "tenzor/core/tensor.hpp"
#include "tenzor/ops/creation.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <omp.h>

// Intel oneDNN for optimized pooling operations
#ifdef TENZOR_USE_ONEDNN
#include <dnnl.hpp>
#endif

namespace tenzor {
namespace cpu {

// ============================================================================
// oneDNN Pooling Helpers
// ============================================================================
#ifdef TENZOR_USE_ONEDNN
static thread_local dnnl::engine g_pooling_engine(dnnl::engine::kind::cpu, 0);
static thread_local dnnl::stream g_pooling_stream(g_pooling_engine);

// Threshold for using oneDNN (elements in output)
constexpr size_t ONEDNN_POOLING_THRESHOLD = 4096;

// oneDNN max pooling forward
static bool onednn_maxpool2d_forward(
    const float* input, float* output, int64_t* indices,
    int64_t N, int64_t C, int64_t H, int64_t W,
    int64_t H_out, int64_t W_out,
    int64_t kernel_size, int64_t stride, int64_t padding) {

    // oneDNN doesn't provide indices, so we fall back for cases that need them
    if (indices != nullptr) {
        return false;
    }

    size_t output_size = N * C * H_out * W_out;
    if (output_size < ONEDNN_POOLING_THRESHOLD) {
        return false;
    }

    try {
        auto& engine = g_pooling_engine;
        auto& stream = g_pooling_stream;

        // Memory descriptors for NCHW format
        dnnl::memory::dims src_dims = {N, C, H, W};
        dnnl::memory::dims dst_dims = {N, C, H_out, W_out};
        dnnl::memory::dims kernel_dims = {kernel_size, kernel_size};
        dnnl::memory::dims stride_dims = {stride, stride};
        dnnl::memory::dims dilation_dims = {0, 0};  // No dilation for pooling
        dnnl::memory::dims padding_l = {padding, padding};
        dnnl::memory::dims padding_r = {padding, padding};

        auto src_md = dnnl::memory::desc(src_dims, dnnl::memory::data_type::f32, dnnl::memory::format_tag::nchw);
        auto dst_md = dnnl::memory::desc(dst_dims, dnnl::memory::data_type::f32, dnnl::memory::format_tag::nchw);

        // Create pooling primitive descriptor
        // Args: engine, prop_kind, algorithm, src_md, dst_md, strides, kernel, dilation, padding_l, padding_r
        auto pool_pd = dnnl::pooling_forward::primitive_desc(
            engine,
            dnnl::prop_kind::forward_inference,
            dnnl::algorithm::pooling_max,
            src_md, dst_md,
            stride_dims, kernel_dims,
            dilation_dims, padding_l, padding_r);

        // Create memory objects
        auto src_mem = dnnl::memory(src_md, engine, const_cast<float*>(input));
        auto dst_mem = dnnl::memory(dst_md, engine, output);

        // Create and execute pooling primitive
        auto pool_prim = dnnl::pooling_forward(pool_pd);
        pool_prim.execute(stream, {
            {DNNL_ARG_SRC, src_mem},
            {DNNL_ARG_DST, dst_mem}
        });
        stream.wait();

        return true;
    } catch (const dnnl::error&) {
        return false;
    }
}

// oneDNN average pooling forward
static bool onednn_avgpool2d_forward(
    const float* input, float* output,
    int64_t N, int64_t C, int64_t H, int64_t W,
    int64_t H_out, int64_t W_out,
    int64_t kernel_size, int64_t stride, int64_t padding) {

    size_t output_size = N * C * H_out * W_out;
    if (output_size < ONEDNN_POOLING_THRESHOLD) {
        return false;
    }

    try {
        auto& engine = g_pooling_engine;
        auto& stream = g_pooling_stream;

        dnnl::memory::dims src_dims = {N, C, H, W};
        dnnl::memory::dims dst_dims = {N, C, H_out, W_out};
        dnnl::memory::dims kernel_dims = {kernel_size, kernel_size};
        dnnl::memory::dims stride_dims = {stride, stride};
        dnnl::memory::dims dilation_dims = {0, 0};  // No dilation for pooling
        dnnl::memory::dims padding_l = {padding, padding};
        dnnl::memory::dims padding_r = {padding, padding};

        auto src_md = dnnl::memory::desc(src_dims, dnnl::memory::data_type::f32, dnnl::memory::format_tag::nchw);
        auto dst_md = dnnl::memory::desc(dst_dims, dnnl::memory::data_type::f32, dnnl::memory::format_tag::nchw);

        // Use exclude_padding for accurate averaging at edges
        // Args: engine, prop_kind, algorithm, src_md, dst_md, strides, kernel, dilation, padding_l, padding_r
        auto pool_pd = dnnl::pooling_forward::primitive_desc(
            engine,
            dnnl::prop_kind::forward_inference,
            dnnl::algorithm::pooling_avg_exclude_padding,
            src_md, dst_md,
            stride_dims, kernel_dims,
            dilation_dims, padding_l, padding_r);

        auto src_mem = dnnl::memory(src_md, engine, const_cast<float*>(input));
        auto dst_mem = dnnl::memory(dst_md, engine, output);

        auto pool_prim = dnnl::pooling_forward(pool_pd);
        pool_prim.execute(stream, {
            {DNNL_ARG_SRC, src_mem},
            {DNNL_ARG_DST, dst_mem}
        });
        stream.wait();

        return true;
    } catch (const dnnl::error&) {
        return false;
    }
}
#endif // TENZOR_USE_ONEDNN

auto maxpool2d_forward_kernel(const Tensor& input, int64_t kernel_size,
                               int64_t stride, int64_t padding)
    -> std::pair<Tensor, Tensor> {
    auto shape = input.shape();
    int64_t N = shape[0];
    int64_t C = shape[1];
    int64_t H = shape[2];
    int64_t W = shape[3];

    int64_t H_out = (H + 2 * padding - kernel_size) / stride + 1;
    int64_t W_out = (W + 2 * padding - kernel_size) / stride + 1;

    auto output = Tensor::empty_uninitialized({N, C, H_out, W_out}, input.dtype(), input.device());
    auto indices = Tensor::empty_uninitialized({N, C, H_out, W_out}, DType::Int64, input.device());

    const float* in_data = input.data<float>();
    float* out_data = output.data<float>();
    int64_t* idx_data = indices.data<int64_t>();

    #pragma omp parallel for collapse(4)
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            for (int64_t oh = 0; oh < H_out; ++oh) {
                for (int64_t ow = 0; ow < W_out; ++ow) {
                    int64_t h_start = oh * stride - padding;
                    int64_t w_start = ow * stride - padding;

                    float max_val = -std::numeric_limits<float>::infinity();
                    int64_t max_idx = 0;

                    for (int64_t kh = 0; kh < kernel_size; ++kh) {
                        for (int64_t kw = 0; kw < kernel_size; ++kw) {
                            int64_t h = h_start + kh;
                            int64_t w = w_start + kw;

                            if (h >= 0 && h < H && w >= 0 && w < W) {
                                int64_t in_idx = ((n * C + c) * H + h) * W + w;
                                if (in_data[in_idx] > max_val) {
                                    max_val = in_data[in_idx];
                                    max_idx = h * W + w;
                                }
                            }
                        }
                    }

                    int64_t out_idx = ((n * C + c) * H_out + oh) * W_out + ow;
                    out_data[out_idx] = max_val;
                    idx_data[out_idx] = max_idx;
                }
            }
        }
    }

    return {output, indices};
}

auto maxpool2d_backward_kernel(const Tensor& grad_output, const Tensor& indices,
                                const std::vector<int64_t>& input_shape) -> Tensor {
    int64_t N = input_shape[0];
    int64_t C = input_shape[1];
    int64_t H = input_shape[2];
    int64_t W = input_shape[3];

    auto grad_shape = grad_output.shape();
    int64_t H_out = grad_shape[2];
    int64_t W_out = grad_shape[3];

    auto grad_input = zeros(input_shape, grad_output.dtype(), grad_output.device());

    const float* grad_out_data = grad_output.data<float>();
    const int64_t* idx_data = indices.data<int64_t>();
    float* grad_in_data = grad_input.data<float>();

    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            for (int64_t oh = 0; oh < H_out; ++oh) {
                for (int64_t ow = 0; ow < W_out; ++ow) {
                    int64_t out_idx = ((n * C + c) * H_out + oh) * W_out + ow;
                    int64_t max_idx = idx_data[out_idx];
                    int64_t h = max_idx / W;
                    int64_t w = max_idx % W;
                    int64_t in_idx = ((n * C + c) * H + h) * W + w;
                    grad_in_data[in_idx] += grad_out_data[out_idx];
                }
            }
        }
    }

    return grad_input;
}

auto avgpool2d_forward_kernel(const Tensor& input, int64_t kernel_size,
                               int64_t stride, int64_t padding) -> Tensor {
    auto shape = input.shape();
    int64_t N = shape[0];
    int64_t C = shape[1];
    int64_t H = shape[2];
    int64_t W = shape[3];

    int64_t H_out = (H + 2 * padding - kernel_size) / stride + 1;
    int64_t W_out = (W + 2 * padding - kernel_size) / stride + 1;

    auto output = Tensor::empty_uninitialized({N, C, H_out, W_out}, input.dtype(), input.device());

    const float* in_data = input.data<float>();
    float* out_data = output.data<float>();

#ifdef TENZOR_USE_ONEDNN
    // Try oneDNN for Float32 tensors (significantly faster for large tensors)
    if (input.dtype() == DType::Float32) {
        if (onednn_avgpool2d_forward(in_data, out_data, N, C, H, W, H_out, W_out, kernel_size, stride, padding)) {
            return output;
        }
    }
#endif

    // Fall back to OpenMP implementation
    #pragma omp parallel for collapse(4)
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            for (int64_t oh = 0; oh < H_out; ++oh) {
                for (int64_t ow = 0; ow < W_out; ++ow) {
                    int64_t h_start = oh * stride - padding;
                    int64_t w_start = ow * stride - padding;

                    float sum = 0.0f;
                    int64_t count = 0;

                    for (int64_t kh = 0; kh < kernel_size; ++kh) {
                        for (int64_t kw = 0; kw < kernel_size; ++kw) {
                            int64_t h = h_start + kh;
                            int64_t w = w_start + kw;

                            if (h >= 0 && h < H && w >= 0 && w < W) {
                                sum += in_data[((n * C + c) * H + h) * W + w];
                                count++;
                            }
                        }
                    }

                    out_data[((n * C + c) * H_out + oh) * W_out + ow] = sum / count;
                }
            }
        }
    }

    return output;
}

auto avgpool2d_backward_kernel(const Tensor& grad_output,
                                const std::vector<int64_t>& input_shape,
                                int64_t kernel_size, int64_t stride,
                                int64_t padding) -> Tensor {
    int64_t N = input_shape[0];
    int64_t C = input_shape[1];
    int64_t H = input_shape[2];
    int64_t W = input_shape[3];

    auto grad_shape = grad_output.shape();
    int64_t H_out = grad_shape[2];
    int64_t W_out = grad_shape[3];

    auto grad_input = zeros(input_shape, grad_output.dtype(), grad_output.device());

    const float* grad_out_data = grad_output.data<float>();
    float* grad_in_data = grad_input.data<float>();

    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            for (int64_t oh = 0; oh < H_out; ++oh) {
                for (int64_t ow = 0; ow < W_out; ++ow) {
                    int64_t h_start = oh * stride - padding;
                    int64_t w_start = ow * stride - padding;

                    int64_t count = 0;
                    for (int64_t kh = 0; kh < kernel_size; ++kh) {
                        for (int64_t kw = 0; kw < kernel_size; ++kw) {
                            int64_t h = h_start + kh;
                            int64_t w = w_start + kw;
                            if (h >= 0 && h < H && w >= 0 && w < W) count++;
                        }
                    }

                    float grad_val = grad_out_data[((n * C + c) * H_out + oh) * W_out + ow] / count;

                    for (int64_t kh = 0; kh < kernel_size; ++kh) {
                        for (int64_t kw = 0; kw < kernel_size; ++kw) {
                            int64_t h = h_start + kh;
                            int64_t w = w_start + kw;
                            if (h >= 0 && h < H && w >= 0 && w < W) {
                                grad_in_data[((n * C + c) * H + h) * W + w] += grad_val;
                            }
                        }
                    }
                }
            }
        }
    }

    return grad_input;
}

auto adaptive_avgpool2d_kernel(const Tensor& input, int64_t output_h,
                                int64_t output_w) -> Tensor {
    auto shape = input.shape();
    int64_t N = shape[0];
    int64_t C = shape[1];
    int64_t H = shape[2];
    int64_t W = shape[3];

    auto output = Tensor::empty_uninitialized({N, C, output_h, output_w}, input.dtype(), input.device());

    const float* in_data = input.data<float>();
    float* out_data = output.data<float>();

    #pragma omp parallel for collapse(4)
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            for (int64_t oh = 0; oh < output_h; ++oh) {
                for (int64_t ow = 0; ow < output_w; ++ow) {
                    int64_t h_start = (oh * H) / output_h;
                    int64_t h_end = ((oh + 1) * H) / output_h;
                    int64_t w_start = (ow * W) / output_w;
                    int64_t w_end = ((ow + 1) * W) / output_w;

                    float sum = 0.0f;
                    int64_t count = 0;

                    for (int64_t h = h_start; h < h_end; ++h) {
                        for (int64_t w = w_start; w < w_end; ++w) {
                            sum += in_data[((n * C + c) * H + h) * W + w];
                            count++;
                        }
                    }

                    out_data[((n * C + c) * output_h + oh) * output_w + ow] = sum / count;
                }
            }
        }
    }

    return output;
}

auto adaptive_avgpool2d_backward_kernel(const Tensor& grad_output,
                                         const std::vector<int64_t>& input_shape) -> Tensor {
    int64_t N = input_shape[0];
    int64_t C = input_shape[1];
    int64_t H = input_shape[2];
    int64_t W = input_shape[3];

    auto grad_shape = grad_output.shape();
    int64_t output_h = grad_shape[2];
    int64_t output_w = grad_shape[3];

    auto grad_input = zeros(input_shape, grad_output.dtype(), grad_output.device());

    const float* grad_out_data = grad_output.data<float>();
    float* grad_in_data = grad_input.data<float>();

    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            for (int64_t oh = 0; oh < output_h; ++oh) {
                for (int64_t ow = 0; ow < output_w; ++ow) {
                    int64_t h_start = (oh * H) / output_h;
                    int64_t h_end = ((oh + 1) * H) / output_h;
                    int64_t w_start = (ow * W) / output_w;
                    int64_t w_end = ((ow + 1) * W) / output_w;

                    int64_t count = (h_end - h_start) * (w_end - w_start);
                    float grad_val = grad_out_data[((n * C + c) * output_h + oh) * output_w + ow] / count;

                    for (int64_t h = h_start; h < h_end; ++h) {
                        for (int64_t w = w_start; w < w_end; ++w) {
                            grad_in_data[((n * C + c) * H + h) * W + w] += grad_val;
                        }
                    }
                }
            }
        }
    }

    return grad_input;
}

auto adaptive_maxpool2d_kernel(const Tensor& input, int64_t output_h,
                                int64_t output_w) -> std::pair<Tensor, Tensor> {
    auto shape = input.shape();
    int64_t N = shape[0];
    int64_t C = shape[1];
    int64_t H = shape[2];
    int64_t W = shape[3];

    auto output = Tensor::empty_uninitialized({N, C, output_h, output_w}, input.dtype(), input.device());
    auto indices = Tensor::empty_uninitialized({N, C, output_h, output_w}, DType::Int64, input.device());

    const float* in_data = input.data<float>();
    float* out_data = output.data<float>();
    int64_t* idx_data = indices.data<int64_t>();

    #pragma omp parallel for collapse(4)
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            for (int64_t oh = 0; oh < output_h; ++oh) {
                for (int64_t ow = 0; ow < output_w; ++ow) {
                    int64_t h_start = (oh * H) / output_h;
                    int64_t h_end = ((oh + 1) * H) / output_h;
                    int64_t w_start = (ow * W) / output_w;
                    int64_t w_end = ((ow + 1) * W) / output_w;

                    float max_val = -std::numeric_limits<float>::infinity();
                    int64_t max_idx = 0;

                    for (int64_t h = h_start; h < h_end; ++h) {
                        for (int64_t w = w_start; w < w_end; ++w) {
                            int64_t in_idx = ((n * C + c) * H + h) * W + w;
                            if (in_data[in_idx] > max_val) {
                                max_val = in_data[in_idx];
                                max_idx = h * W + w;
                            }
                        }
                    }

                    int64_t out_idx = ((n * C + c) * output_h + oh) * output_w + ow;
                    out_data[out_idx] = max_val;
                    idx_data[out_idx] = max_idx;
                }
            }
        }
    }

    return {output, indices};
}

auto adaptive_maxpool2d_backward_kernel(const Tensor& grad_output, const Tensor& indices,
                                         const std::vector<int64_t>& input_shape) -> Tensor {
    int64_t N = input_shape[0];
    int64_t C = input_shape[1];
    int64_t H = input_shape[2];
    int64_t W = input_shape[3];

    auto grad_shape = grad_output.shape();
    int64_t output_h = grad_shape[2];
    int64_t output_w = grad_shape[3];

    auto grad_input = zeros(input_shape, grad_output.dtype(), grad_output.device());

    const float* grad_out_data = grad_output.data<float>();
    const int64_t* idx_data = indices.data<int64_t>();
    float* grad_in_data = grad_input.data<float>();

    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            for (int64_t oh = 0; oh < output_h; ++oh) {
                for (int64_t ow = 0; ow < output_w; ++ow) {
                    int64_t out_idx = ((n * C + c) * output_h + oh) * output_w + ow;
                    int64_t max_idx = idx_data[out_idx];
                    int64_t h = max_idx / W;
                    int64_t w = max_idx % W;
                    int64_t in_idx = ((n * C + c) * H + h) * W + w;
                    grad_in_data[in_idx] += grad_out_data[out_idx];
                }
            }
        }
    }

    return grad_input;
}

} // namespace cpu
} // namespace tenzor
