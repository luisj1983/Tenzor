/**
 * @file pooling.cpp
 * @brief CPU pooling kernel implementations
 */

#include "tenzor/core/tensor.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/backends/cpu/simd.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <omp.h>

// Intel oneDNN for optimized pooling operations
#ifdef TENZOR_USE_ONEDNN
#include <dnnl.hpp>
#include "onednn_cache.hpp"
#include <list>
#include <unordered_map>
#endif

namespace tenzor {
namespace cpu {

// AVX-512 forward declarations (defined in pooling_avx512.cpp)
namespace avx512 {
void avgpool2d_forward_f32(const float*, float*, int64_t, int64_t, int64_t, int64_t,
                           int64_t, int64_t, int64_t, int64_t, int64_t);
void adaptive_avgpool2d_forward_f32(const float*, float*, int64_t, int64_t, int64_t, int64_t,
                                     int64_t, int64_t);
} // namespace avx512

// ============================================================================
// oneDNN Pooling Helpers with Primitive Caching
// ============================================================================
#ifdef TENZOR_USE_ONEDNN
// Use shared lazy-init accessors from onednn_cache.hpp to avoid static
// thread_local initialization issues in dlopen'd libraries.

// Threshold for using oneDNN (elements in output)
constexpr size_t ONEDNN_POOLING_THRESHOLD = 4096;

// --------------------------------------------------------------------------
// Pooling Primitive Caching (eliminates ~1-5ms primitive creation overhead)
// --------------------------------------------------------------------------
struct PoolingCacheKey {
    dnnl::algorithm algo;
    int64_t N, C, H, W;
    int64_t H_out, W_out;
    int64_t kernel_size, stride, padding;

    bool operator==(const PoolingCacheKey& other) const {
        return algo == other.algo && N == other.N && C == other.C &&
               H == other.H && W == other.W &&
               H_out == other.H_out && W_out == other.W_out &&
               kernel_size == other.kernel_size &&
               stride == other.stride && padding == other.padding;
    }
};

struct PoolingCacheKeyHash {
    size_t operator()(const PoolingCacheKey& k) const {
        size_t h = std::hash<int>{}(static_cast<int>(k.algo));
        auto hash_combine = [&h](int64_t v) {
            h ^= std::hash<int64_t>{}(v) + 0x9e3779b9 + (h << 6) + (h >> 2);
        };
        hash_combine(k.N);
        hash_combine(k.C);
        hash_combine(k.H);
        hash_combine(k.W);
        hash_combine(k.H_out);
        hash_combine(k.W_out);
        hash_combine(k.kernel_size);
        hash_combine(k.stride);
        hash_combine(k.padding);
        return h;
    }
};

struct PoolingCachedPrimitive {
    dnnl::pooling_forward prim;
    dnnl::memory::desc src_md, dst_md;
};

static constexpr size_t POOLING_CACHE_SIZE = 32;

using PoolingPrimitiveCache = OneDNNPrimitiveCache<PoolingCacheKey, PoolingCachedPrimitive, PoolingCacheKeyHash, POOLING_CACHE_SIZE>;

static thread_local PoolingPrimitiveCache g_pooling_cache;

// oneDNN average pooling forward with caching
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
        auto& engine = get_onednn_engine();
        auto& stream = get_onednn_stream();

        // Create cache key
        PoolingCacheKey cache_key{dnnl::algorithm::pooling_avg_exclude_padding,
                                   N, C, H, W, H_out, W_out,
                                   kernel_size, stride, padding};

        auto cached = g_pooling_cache.get(cache_key);

        if (!cached) {
            cached = std::make_shared<PoolingCachedPrimitive>();

            dnnl::memory::dims src_dims = {N, C, H, W};
            dnnl::memory::dims dst_dims = {N, C, H_out, W_out};
            dnnl::memory::dims kernel_dims = {kernel_size, kernel_size};
            dnnl::memory::dims stride_dims = {stride, stride};
            dnnl::memory::dims dilation_dims = {0, 0};
            dnnl::memory::dims padding_l = {padding, padding};
            dnnl::memory::dims padding_r = {padding, padding};

            cached->src_md = dnnl::memory::desc(src_dims, dnnl::memory::data_type::f32, dnnl::memory::format_tag::nchw);
            cached->dst_md = dnnl::memory::desc(dst_dims, dnnl::memory::data_type::f32, dnnl::memory::format_tag::nchw);

            auto pool_pd = dnnl::pooling_forward::primitive_desc(
                engine,
                dnnl::prop_kind::forward_inference,
                dnnl::algorithm::pooling_avg_exclude_padding,
                cached->src_md, cached->dst_md,
                stride_dims, kernel_dims,
                dilation_dims, padding_l, padding_r);
            cached->prim = dnnl::pooling_forward(pool_pd);

            g_pooling_cache.put(cache_key, cached);
        }

        auto src_mem = dnnl::memory(cached->src_md, engine, const_cast<float*>(input));
        auto dst_mem = dnnl::memory(cached->dst_md, engine, output);

        cached->prim.execute(stream, {
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

template<typename T>
void maxpool2d_forward_impl(const T* in_data, T* out_data, int64_t* idx_data,
                             int64_t N, int64_t C, int64_t H, int64_t W,
                             int64_t H_out, int64_t W_out,
                             int64_t kernel_size, int64_t stride, int64_t padding, int64_t dilation) {
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
                            int64_t h = h_start + kh * dilation;
                            int64_t w = w_start + kw * dilation;

                            if (h >= 0 && h < H && w >= 0 && w < W) {
                                int64_t in_idx = ((n * C + c) * H + h) * W + w;
                                float val = static_cast<float>(in_data[in_idx]);
                                if (val > max_val) {
                                    max_val = val;
                                    max_idx = h * W + w;
                                }
                            }
                        }
                    }

                    int64_t out_idx = ((n * C + c) * H_out + oh) * W_out + ow;
                    out_data[out_idx] = static_cast<T>(max_val);
                    idx_data[out_idx] = max_idx;
                }
            }
        }
    }
}

auto maxpool2d_forward_kernel(const Tensor& input, int64_t kernel_size,
                               int64_t stride, int64_t padding, int64_t dilation)
    -> std::pair<Tensor, Tensor> {
    auto shape = input.shape();
    int64_t N = shape[0];
    int64_t C = shape[1];
    int64_t H = shape[2];
    int64_t W = shape[3];

    int64_t H_out = (H + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int64_t W_out = (W + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    auto output = Tensor::empty_uninitialized({N, C, H_out, W_out}, input.dtype(), input.device());
    auto indices = Tensor::empty_uninitialized({N, C, H_out, W_out}, DType::Int64, input.device());
    int64_t* idx_data = indices.data<int64_t>();

    // Note: oneDNN maxpool forward is defined but not dispatched here because
    // oneDNN cannot produce indices, which are always required for backward pass.
    // Computing output via oneDNN + indices via template impl would be no faster
    // than computing both together in the template impl.

    if (input.dtype() == DType::Float32) {
        maxpool2d_forward_impl<float>(input.data<float>(), output.data<float>(), idx_data,
                                      N, C, H, W, H_out, W_out, kernel_size, stride, padding, dilation);
    } else if (input.dtype() == DType::Float64) {
        maxpool2d_forward_impl<double>(input.data<double>(), output.data<double>(), idx_data,
                                       N, C, H, W, H_out, W_out, kernel_size, stride, padding, dilation);
    } else if (input.dtype() == DType::Float16) {
        maxpool2d_forward_impl<Float16>(input.data<Float16>(), output.data<Float16>(), idx_data,
                                        N, C, H, W, H_out, W_out, kernel_size, stride, padding, dilation);
    } else if (input.dtype() == DType::BFloat16) {
        maxpool2d_forward_impl<BFloat16>(input.data<BFloat16>(), output.data<BFloat16>(), idx_data,
                                         N, C, H, W, H_out, W_out, kernel_size, stride, padding, dilation);
    } else {
        throw std::runtime_error("Unsupported dtype for maxpool2d_forward");
    }

    return {output, indices};
}

template<typename T>
void maxpool2d_backward_impl(const T* grad_out_data, const int64_t* idx_data, T* grad_in_data,
                              int64_t N, int64_t C, int64_t H, int64_t W,
                              int64_t H_out, int64_t W_out) {
    #pragma omp parallel for collapse(2) if(N * C > 4)
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            for (int64_t oh = 0; oh < H_out; ++oh) {
                for (int64_t ow = 0; ow < W_out; ++ow) {
                    int64_t out_idx = ((n * C + c) * H_out + oh) * W_out + ow;
                    int64_t max_idx = idx_data[out_idx];
                    int64_t h = max_idx / W;
                    int64_t w = max_idx % W;
                    int64_t in_idx = ((n * C + c) * H + h) * W + w;
                    // For half types, accumulate via float cast
                    float val = static_cast<float>(grad_in_data[in_idx]) + static_cast<float>(grad_out_data[out_idx]);
                    grad_in_data[in_idx] = static_cast<T>(val);
                }
            }
        }
    }
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
    const int64_t* idx_data = indices.data<int64_t>();

    if (grad_output.dtype() == DType::Float32) {
        maxpool2d_backward_impl<float>(grad_output.data<float>(), idx_data, grad_input.data<float>(),
                                       N, C, H, W, H_out, W_out);
    } else if (grad_output.dtype() == DType::Float64) {
        maxpool2d_backward_impl<double>(grad_output.data<double>(), idx_data, grad_input.data<double>(),
                                        N, C, H, W, H_out, W_out);
    } else if (grad_output.dtype() == DType::Float16) {
        maxpool2d_backward_impl<Float16>(grad_output.data<Float16>(), idx_data, grad_input.data<Float16>(),
                                         N, C, H, W, H_out, W_out);
    } else if (grad_output.dtype() == DType::BFloat16) {
        maxpool2d_backward_impl<BFloat16>(grad_output.data<BFloat16>(), idx_data, grad_input.data<BFloat16>(),
                                          N, C, H, W, H_out, W_out);
    } else {
        throw std::runtime_error("Unsupported dtype for maxpool2d_backward");
    }

    return grad_input;
}

template<typename T>
void avgpool2d_forward_impl(const T* in_data, T* out_data,
                             int64_t N, int64_t C, int64_t H, int64_t W,
                             int64_t H_out, int64_t W_out,
                             int64_t kernel_size, int64_t stride, int64_t padding) {
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
                                sum += static_cast<float>(in_data[((n * C + c) * H + h) * W + w]);
                                count++;
                            }
                        }
                    }

                    out_data[((n * C + c) * H_out + oh) * W_out + ow] = static_cast<T>(sum / count);
                }
            }
        }
    }
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

#ifdef TENZOR_USE_ONEDNN
    if (input.dtype() == DType::Float32) {
        if (onednn_avgpool2d_forward(input.data<float>(), output.data<float>(),
                                     N, C, H, W, H_out, W_out, kernel_size, stride, padding)) {
            return output;
        }
    }
#endif

    if (input.dtype() == DType::Float32) {
        if (CPUInfo::get().has_avx512()) {
            avx512::avgpool2d_forward_f32(input.data<float>(), output.data<float>(),
                                          N, C, H, W, H_out, W_out, kernel_size, stride, padding);
        } else {
            avgpool2d_forward_impl<float>(input.data<float>(), output.data<float>(),
                                          N, C, H, W, H_out, W_out, kernel_size, stride, padding);
        }
    } else if (input.dtype() == DType::Float64) {
        avgpool2d_forward_impl<double>(input.data<double>(), output.data<double>(),
                                       N, C, H, W, H_out, W_out, kernel_size, stride, padding);
    } else if (input.dtype() == DType::Float16) {
        avgpool2d_forward_impl<Float16>(input.data<Float16>(), output.data<Float16>(),
                                        N, C, H, W, H_out, W_out, kernel_size, stride, padding);
    } else if (input.dtype() == DType::BFloat16) {
        avgpool2d_forward_impl<BFloat16>(input.data<BFloat16>(), output.data<BFloat16>(),
                                         N, C, H, W, H_out, W_out, kernel_size, stride, padding);
    } else {
        throw std::runtime_error("Unsupported dtype for avgpool2d_forward");
    }

    return output;
}

template<typename T>
void avgpool2d_backward_impl(const T* grad_out_data, T* grad_in_data,
                              int64_t N, int64_t C, int64_t H, int64_t W,
                              int64_t H_out, int64_t W_out,
                              int64_t kernel_size, int64_t stride, int64_t padding) {
    #pragma omp parallel for collapse(2) if(N * C > 4)
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

                    float grad_val = static_cast<float>(grad_out_data[((n * C + c) * H_out + oh) * W_out + ow]) / count;

                    for (int64_t kh = 0; kh < kernel_size; ++kh) {
                        for (int64_t kw = 0; kw < kernel_size; ++kw) {
                            int64_t h = h_start + kh;
                            int64_t w = w_start + kw;
                            if (h >= 0 && h < H && w >= 0 && w < W) {
                                int64_t idx = ((n * C + c) * H + h) * W + w;
                                float val = static_cast<float>(grad_in_data[idx]) + grad_val;
                                grad_in_data[idx] = static_cast<T>(val);
                            }
                        }
                    }
                }
            }
        }
    }
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

    if (grad_output.dtype() == DType::Float32) {
        avgpool2d_backward_impl<float>(grad_output.data<float>(), grad_input.data<float>(),
                                       N, C, H, W, H_out, W_out, kernel_size, stride, padding);
    } else if (grad_output.dtype() == DType::Float64) {
        avgpool2d_backward_impl<double>(grad_output.data<double>(), grad_input.data<double>(),
                                        N, C, H, W, H_out, W_out, kernel_size, stride, padding);
    } else if (grad_output.dtype() == DType::Float16) {
        avgpool2d_backward_impl<Float16>(grad_output.data<Float16>(), grad_input.data<Float16>(),
                                         N, C, H, W, H_out, W_out, kernel_size, stride, padding);
    } else if (grad_output.dtype() == DType::BFloat16) {
        avgpool2d_backward_impl<BFloat16>(grad_output.data<BFloat16>(), grad_input.data<BFloat16>(),
                                          N, C, H, W, H_out, W_out, kernel_size, stride, padding);
    } else {
        throw std::runtime_error("Unsupported dtype for avgpool2d_backward");
    }

    return grad_input;
}

template<typename T>
void adaptive_avgpool2d_impl(const T* in_data, T* out_data,
                              int64_t N, int64_t C, int64_t H, int64_t W,
                              int64_t output_h, int64_t output_w) {
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
                            sum += static_cast<float>(in_data[((n * C + c) * H + h) * W + w]);
                            count++;
                        }
                    }

                    out_data[((n * C + c) * output_h + oh) * output_w + ow] = static_cast<T>(sum / count);
                }
            }
        }
    }
}

auto adaptive_avgpool2d_kernel(const Tensor& input, int64_t output_h,
                                int64_t output_w) -> Tensor {
    auto shape = input.shape();
    int64_t N = shape[0];
    int64_t C = shape[1];
    int64_t H = shape[2];
    int64_t W = shape[3];

    auto output = Tensor::empty_uninitialized({N, C, output_h, output_w}, input.dtype(), input.device());

    if (input.dtype() == DType::Float32) {
        if (CPUInfo::get().has_avx512()) {
            avx512::adaptive_avgpool2d_forward_f32(input.data<float>(), output.data<float>(),
                                                    N, C, H, W, output_h, output_w);
        } else {
            adaptive_avgpool2d_impl<float>(input.data<float>(), output.data<float>(),
                                           N, C, H, W, output_h, output_w);
        }
    } else if (input.dtype() == DType::Float64) {
        adaptive_avgpool2d_impl<double>(input.data<double>(), output.data<double>(), N, C, H, W, output_h, output_w);
    } else if (input.dtype() == DType::Float16) {
        adaptive_avgpool2d_impl<Float16>(input.data<Float16>(), output.data<Float16>(), N, C, H, W, output_h, output_w);
    } else if (input.dtype() == DType::BFloat16) {
        adaptive_avgpool2d_impl<BFloat16>(input.data<BFloat16>(), output.data<BFloat16>(), N, C, H, W, output_h, output_w);
    } else {
        throw std::runtime_error("Unsupported dtype for adaptive_avgpool2d");
    }

    return output;
}

template<typename T>
void adaptive_avgpool2d_backward_impl(const T* grad_out_data, T* grad_in_data,
                                       int64_t N, int64_t C, int64_t H, int64_t W,
                                       int64_t output_h, int64_t output_w) {
    #pragma omp parallel for collapse(2) if(N * C > 4)
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            for (int64_t oh = 0; oh < output_h; ++oh) {
                for (int64_t ow = 0; ow < output_w; ++ow) {
                    int64_t h_start = (oh * H) / output_h;
                    int64_t h_end = ((oh + 1) * H) / output_h;
                    int64_t w_start = (ow * W) / output_w;
                    int64_t w_end = ((ow + 1) * W) / output_w;

                    int64_t count = (h_end - h_start) * (w_end - w_start);
                    float grad_val = static_cast<float>(grad_out_data[((n * C + c) * output_h + oh) * output_w + ow]) / count;

                    for (int64_t h = h_start; h < h_end; ++h) {
                        for (int64_t w = w_start; w < w_end; ++w) {
                            int64_t idx = ((n * C + c) * H + h) * W + w;
                            float val = static_cast<float>(grad_in_data[idx]) + grad_val;
                            grad_in_data[idx] = static_cast<T>(val);
                        }
                    }
                }
            }
        }
    }
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

    if (grad_output.dtype() == DType::Float32) {
        adaptive_avgpool2d_backward_impl<float>(grad_output.data<float>(), grad_input.data<float>(),
                                                N, C, H, W, output_h, output_w);
    } else if (grad_output.dtype() == DType::Float64) {
        adaptive_avgpool2d_backward_impl<double>(grad_output.data<double>(), grad_input.data<double>(),
                                                 N, C, H, W, output_h, output_w);
    } else if (grad_output.dtype() == DType::Float16) {
        adaptive_avgpool2d_backward_impl<Float16>(grad_output.data<Float16>(), grad_input.data<Float16>(),
                                                  N, C, H, W, output_h, output_w);
    } else if (grad_output.dtype() == DType::BFloat16) {
        adaptive_avgpool2d_backward_impl<BFloat16>(grad_output.data<BFloat16>(), grad_input.data<BFloat16>(),
                                                   N, C, H, W, output_h, output_w);
    } else {
        throw std::runtime_error("Unsupported dtype for adaptive_avgpool2d_backward");
    }

    return grad_input;
}

template<typename T>
void adaptive_maxpool2d_impl(const T* in_data, T* out_data, int64_t* idx_data,
                              int64_t N, int64_t C, int64_t H, int64_t W,
                              int64_t output_h, int64_t output_w) {
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
                            float val = static_cast<float>(in_data[in_idx]);
                            if (val > max_val) {
                                max_val = val;
                                max_idx = h * W + w;
                            }
                        }
                    }

                    int64_t out_idx = ((n * C + c) * output_h + oh) * output_w + ow;
                    out_data[out_idx] = static_cast<T>(max_val);
                    idx_data[out_idx] = max_idx;
                }
            }
        }
    }
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
    int64_t* idx_data = indices.data<int64_t>();

    if (input.dtype() == DType::Float32) {
        adaptive_maxpool2d_impl<float>(input.data<float>(), output.data<float>(), idx_data,
                                       N, C, H, W, output_h, output_w);
    } else if (input.dtype() == DType::Float64) {
        adaptive_maxpool2d_impl<double>(input.data<double>(), output.data<double>(), idx_data,
                                        N, C, H, W, output_h, output_w);
    } else if (input.dtype() == DType::Float16) {
        adaptive_maxpool2d_impl<Float16>(input.data<Float16>(), output.data<Float16>(), idx_data,
                                         N, C, H, W, output_h, output_w);
    } else if (input.dtype() == DType::BFloat16) {
        adaptive_maxpool2d_impl<BFloat16>(input.data<BFloat16>(), output.data<BFloat16>(), idx_data,
                                          N, C, H, W, output_h, output_w);
    } else {
        throw std::runtime_error("Unsupported dtype for adaptive_maxpool2d");
    }

    return {output, indices};
}

template<typename T>
void adaptive_maxpool2d_backward_impl(const T* grad_out_data, const int64_t* idx_data, T* grad_in_data,
                                       int64_t N, int64_t C, int64_t H, int64_t W,
                                       int64_t output_h, int64_t output_w) {
    #pragma omp parallel for collapse(2) if(N * C > 4)
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            for (int64_t oh = 0; oh < output_h; ++oh) {
                for (int64_t ow = 0; ow < output_w; ++ow) {
                    int64_t out_idx = ((n * C + c) * output_h + oh) * output_w + ow;
                    int64_t max_idx = idx_data[out_idx];
                    int64_t h = max_idx / W;
                    int64_t w = max_idx % W;
                    int64_t in_idx = ((n * C + c) * H + h) * W + w;
                    float val = static_cast<float>(grad_in_data[in_idx]) + static_cast<float>(grad_out_data[out_idx]);
                    grad_in_data[in_idx] = static_cast<T>(val);
                }
            }
        }
    }
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
    const int64_t* idx_data = indices.data<int64_t>();

    if (grad_output.dtype() == DType::Float32) {
        adaptive_maxpool2d_backward_impl<float>(grad_output.data<float>(), idx_data, grad_input.data<float>(),
                                                N, C, H, W, output_h, output_w);
    } else if (grad_output.dtype() == DType::Float64) {
        adaptive_maxpool2d_backward_impl<double>(grad_output.data<double>(), idx_data, grad_input.data<double>(),
                                                 N, C, H, W, output_h, output_w);
    } else if (grad_output.dtype() == DType::Float16) {
        adaptive_maxpool2d_backward_impl<Float16>(grad_output.data<Float16>(), idx_data, grad_input.data<Float16>(),
                                                  N, C, H, W, output_h, output_w);
    } else if (grad_output.dtype() == DType::BFloat16) {
        adaptive_maxpool2d_backward_impl<BFloat16>(grad_output.data<BFloat16>(), idx_data, grad_input.data<BFloat16>(),
                                                   N, C, H, W, output_h, output_w);
    } else {
        throw std::runtime_error("Unsupported dtype for adaptive_maxpool2d_backward");
    }

    return grad_input;
}

// ============================================================================
// 1D Pooling Kernels
// ============================================================================

template<typename T>
void maxpool1d_forward_impl(const T* in_data, T* out_data, int64_t* idx_data,
                             int64_t N, int64_t C, int64_t L,
                             int64_t L_out,
                             int64_t kernel_size, int64_t stride, int64_t padding, int64_t dilation) {
    #pragma omp parallel for collapse(3)
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            for (int64_t ol = 0; ol < L_out; ++ol) {
                int64_t l_start = ol * stride - padding;

                float max_val = -std::numeric_limits<float>::infinity();
                int64_t max_idx = 0;

                for (int64_t k = 0; k < kernel_size; ++k) {
                    int64_t l = l_start + k * dilation;
                    if (l >= 0 && l < L) {
                        int64_t in_idx = (n * C + c) * L + l;
                        float val = static_cast<float>(in_data[in_idx]);
                        if (val > max_val) {
                            max_val = val;
                            max_idx = l;
                        }
                    }
                }

                int64_t out_idx = (n * C + c) * L_out + ol;
                out_data[out_idx] = static_cast<T>(max_val);
                idx_data[out_idx] = max_idx;
            }
        }
    }
}

auto maxpool1d_forward_kernel(const Tensor& input, int64_t kernel_size,
                               int64_t stride, int64_t padding, int64_t dilation)
    -> std::pair<Tensor, Tensor> {
    auto shape = input.shape();
    int64_t N = shape[0];
    int64_t C = shape[1];
    int64_t L = shape[2];

    int64_t L_out = (L + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    auto output = Tensor::empty_uninitialized({N, C, L_out}, input.dtype(), input.device());
    auto indices = Tensor::empty_uninitialized({N, C, L_out}, DType::Int64, input.device());
    int64_t* idx_data = indices.data<int64_t>();

    if (input.dtype() == DType::Float32) {
        maxpool1d_forward_impl<float>(input.data<float>(), output.data<float>(), idx_data,
                                      N, C, L, L_out, kernel_size, stride, padding, dilation);
    } else if (input.dtype() == DType::Float64) {
        maxpool1d_forward_impl<double>(input.data<double>(), output.data<double>(), idx_data,
                                       N, C, L, L_out, kernel_size, stride, padding, dilation);
    } else if (input.dtype() == DType::Float16) {
        maxpool1d_forward_impl<Float16>(input.data<Float16>(), output.data<Float16>(), idx_data,
                                        N, C, L, L_out, kernel_size, stride, padding, dilation);
    } else if (input.dtype() == DType::BFloat16) {
        maxpool1d_forward_impl<BFloat16>(input.data<BFloat16>(), output.data<BFloat16>(), idx_data,
                                         N, C, L, L_out, kernel_size, stride, padding, dilation);
    } else {
        throw std::runtime_error("Unsupported dtype for maxpool1d_forward");
    }

    return {output, indices};
}

template<typename T>
void maxpool1d_backward_impl(const T* grad_out_data, const int64_t* idx_data, T* grad_in_data,
                              int64_t N, int64_t C, int64_t L,
                              int64_t L_out) {
    #pragma omp parallel for collapse(2) if(N * C > 4)
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            for (int64_t ol = 0; ol < L_out; ++ol) {
                int64_t out_idx = (n * C + c) * L_out + ol;
                int64_t max_idx = idx_data[out_idx];
                int64_t in_idx = (n * C + c) * L + max_idx;
                float val = static_cast<float>(grad_in_data[in_idx]) + static_cast<float>(grad_out_data[out_idx]);
                grad_in_data[in_idx] = static_cast<T>(val);
            }
        }
    }
}

auto maxpool1d_backward_kernel(const Tensor& grad_output, const Tensor& indices,
                                const std::vector<int64_t>& input_shape) -> Tensor {
    int64_t N = input_shape[0];
    int64_t C = input_shape[1];
    int64_t L = input_shape[2];

    auto grad_shape = grad_output.shape();
    int64_t L_out = grad_shape[2];

    auto grad_input = zeros(input_shape, grad_output.dtype(), grad_output.device());
    const int64_t* idx_data = indices.data<int64_t>();

    if (grad_output.dtype() == DType::Float32) {
        maxpool1d_backward_impl<float>(grad_output.data<float>(), idx_data, grad_input.data<float>(),
                                       N, C, L, L_out);
    } else if (grad_output.dtype() == DType::Float64) {
        maxpool1d_backward_impl<double>(grad_output.data<double>(), idx_data, grad_input.data<double>(),
                                        N, C, L, L_out);
    } else if (grad_output.dtype() == DType::Float16) {
        maxpool1d_backward_impl<Float16>(grad_output.data<Float16>(), idx_data, grad_input.data<Float16>(),
                                         N, C, L, L_out);
    } else if (grad_output.dtype() == DType::BFloat16) {
        maxpool1d_backward_impl<BFloat16>(grad_output.data<BFloat16>(), idx_data, grad_input.data<BFloat16>(),
                                          N, C, L, L_out);
    } else {
        throw std::runtime_error("Unsupported dtype for maxpool1d_backward");
    }

    return grad_input;
}

template<typename T>
void avgpool1d_forward_impl(const T* in_data, T* out_data,
                             int64_t N, int64_t C, int64_t L,
                             int64_t L_out,
                             int64_t kernel_size, int64_t stride, int64_t padding) {
    #pragma omp parallel for collapse(3)
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            for (int64_t ol = 0; ol < L_out; ++ol) {
                int64_t l_start = ol * stride - padding;

                float sum = 0.0f;
                int64_t count = 0;

                for (int64_t k = 0; k < kernel_size; ++k) {
                    int64_t l = l_start + k;
                    if (l >= 0 && l < L) {
                        sum += static_cast<float>(in_data[(n * C + c) * L + l]);
                        count++;
                    }
                }

                out_data[(n * C + c) * L_out + ol] = static_cast<T>(sum / count);
            }
        }
    }
}

auto avgpool1d_forward_kernel(const Tensor& input, int64_t kernel_size,
                               int64_t stride, int64_t padding) -> Tensor {
    auto shape = input.shape();
    int64_t N = shape[0];
    int64_t C = shape[1];
    int64_t L = shape[2];

    int64_t L_out = (L + 2 * padding - kernel_size) / stride + 1;

    auto output = Tensor::empty_uninitialized({N, C, L_out}, input.dtype(), input.device());

    if (input.dtype() == DType::Float32) {
        avgpool1d_forward_impl<float>(input.data<float>(), output.data<float>(),
                                      N, C, L, L_out, kernel_size, stride, padding);
    } else if (input.dtype() == DType::Float64) {
        avgpool1d_forward_impl<double>(input.data<double>(), output.data<double>(),
                                       N, C, L, L_out, kernel_size, stride, padding);
    } else if (input.dtype() == DType::Float16) {
        avgpool1d_forward_impl<Float16>(input.data<Float16>(), output.data<Float16>(),
                                        N, C, L, L_out, kernel_size, stride, padding);
    } else if (input.dtype() == DType::BFloat16) {
        avgpool1d_forward_impl<BFloat16>(input.data<BFloat16>(), output.data<BFloat16>(),
                                         N, C, L, L_out, kernel_size, stride, padding);
    } else {
        throw std::runtime_error("Unsupported dtype for avgpool1d_forward");
    }

    return output;
}

template<typename T>
void avgpool1d_backward_impl(const T* grad_out_data, T* grad_in_data,
                              int64_t N, int64_t C, int64_t L,
                              int64_t L_out,
                              int64_t kernel_size, int64_t stride, int64_t padding) {
    #pragma omp parallel for collapse(2) if(N * C > 4)
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            for (int64_t ol = 0; ol < L_out; ++ol) {
                int64_t l_start = ol * stride - padding;

                int64_t count = 0;
                for (int64_t k = 0; k < kernel_size; ++k) {
                    int64_t l = l_start + k;
                    if (l >= 0 && l < L) count++;
                }

                float grad_val = static_cast<float>(grad_out_data[(n * C + c) * L_out + ol]) / count;

                for (int64_t k = 0; k < kernel_size; ++k) {
                    int64_t l = l_start + k;
                    if (l >= 0 && l < L) {
                        int64_t idx = (n * C + c) * L + l;
                        float val = static_cast<float>(grad_in_data[idx]) + grad_val;
                        grad_in_data[idx] = static_cast<T>(val);
                    }
                }
            }
        }
    }
}

auto avgpool1d_backward_kernel(const Tensor& grad_output,
                                const std::vector<int64_t>& input_shape,
                                int64_t kernel_size, int64_t stride,
                                int64_t padding) -> Tensor {
    int64_t N = input_shape[0];
    int64_t C = input_shape[1];
    int64_t L = input_shape[2];

    auto grad_shape = grad_output.shape();
    int64_t L_out = grad_shape[2];

    auto grad_input = zeros(input_shape, grad_output.dtype(), grad_output.device());

    if (grad_output.dtype() == DType::Float32) {
        avgpool1d_backward_impl<float>(grad_output.data<float>(), grad_input.data<float>(),
                                       N, C, L, L_out, kernel_size, stride, padding);
    } else if (grad_output.dtype() == DType::Float64) {
        avgpool1d_backward_impl<double>(grad_output.data<double>(), grad_input.data<double>(),
                                        N, C, L, L_out, kernel_size, stride, padding);
    } else if (grad_output.dtype() == DType::Float16) {
        avgpool1d_backward_impl<Float16>(grad_output.data<Float16>(), grad_input.data<Float16>(),
                                         N, C, L, L_out, kernel_size, stride, padding);
    } else if (grad_output.dtype() == DType::BFloat16) {
        avgpool1d_backward_impl<BFloat16>(grad_output.data<BFloat16>(), grad_input.data<BFloat16>(),
                                          N, C, L, L_out, kernel_size, stride, padding);
    } else {
        throw std::runtime_error("Unsupported dtype for avgpool1d_backward");
    }

    return grad_input;
}

// ============================================================================
// Adaptive 1D Pooling Kernels
// ============================================================================

template<typename T>
void adaptive_avgpool1d_impl(const T* in_data, T* out_data,
                              int64_t N, int64_t C, int64_t L,
                              int64_t L_out) {
    #pragma omp parallel for collapse(3)
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            for (int64_t ol = 0; ol < L_out; ++ol) {
                int64_t l_start = (ol * L) / L_out;
                int64_t l_end = ((ol + 1) * L) / L_out;

                float sum = 0.0f;
                int64_t count = l_end - l_start;

                for (int64_t l = l_start; l < l_end; ++l) {
                    sum += static_cast<float>(in_data[(n * C + c) * L + l]);
                }

                out_data[(n * C + c) * L_out + ol] = static_cast<T>(sum / count);
            }
        }
    }
}

auto adaptive_avgpool1d_kernel(const Tensor& input, int64_t output_size) -> Tensor {
    auto shape = input.shape();
    int64_t N = shape[0];
    int64_t C = shape[1];
    int64_t L = shape[2];

    auto output = Tensor::empty_uninitialized({N, C, output_size}, input.dtype(), input.device());

    if (input.dtype() == DType::Float32) {
        adaptive_avgpool1d_impl<float>(input.data<float>(), output.data<float>(), N, C, L, output_size);
    } else if (input.dtype() == DType::Float64) {
        adaptive_avgpool1d_impl<double>(input.data<double>(), output.data<double>(), N, C, L, output_size);
    } else if (input.dtype() == DType::Float16) {
        adaptive_avgpool1d_impl<Float16>(input.data<Float16>(), output.data<Float16>(), N, C, L, output_size);
    } else if (input.dtype() == DType::BFloat16) {
        adaptive_avgpool1d_impl<BFloat16>(input.data<BFloat16>(), output.data<BFloat16>(), N, C, L, output_size);
    } else {
        throw std::runtime_error("Unsupported dtype for adaptive_avgpool1d");
    }

    return output;
}

template<typename T>
void adaptive_avgpool1d_backward_impl(const T* grad_out_data, T* grad_in_data,
                                       int64_t N, int64_t C, int64_t L,
                                       int64_t L_out) {
    #pragma omp parallel for collapse(2) if(N * C > 4)
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            for (int64_t ol = 0; ol < L_out; ++ol) {
                int64_t l_start = (ol * L) / L_out;
                int64_t l_end = ((ol + 1) * L) / L_out;

                int64_t count = l_end - l_start;
                float grad_val = static_cast<float>(grad_out_data[(n * C + c) * L_out + ol]) / count;

                for (int64_t l = l_start; l < l_end; ++l) {
                    int64_t idx = (n * C + c) * L + l;
                    float val = static_cast<float>(grad_in_data[idx]) + grad_val;
                    grad_in_data[idx] = static_cast<T>(val);
                }
            }
        }
    }
}

auto adaptive_avgpool1d_backward_kernel(const Tensor& grad_output,
                                         const std::vector<int64_t>& input_shape) -> Tensor {
    int64_t N = input_shape[0];
    int64_t C = input_shape[1];
    int64_t L = input_shape[2];

    auto grad_shape = grad_output.shape();
    int64_t L_out = grad_shape[2];

    auto grad_input = zeros(input_shape, grad_output.dtype(), grad_output.device());

    if (grad_output.dtype() == DType::Float32) {
        adaptive_avgpool1d_backward_impl<float>(grad_output.data<float>(), grad_input.data<float>(),
                                                N, C, L, L_out);
    } else if (grad_output.dtype() == DType::Float64) {
        adaptive_avgpool1d_backward_impl<double>(grad_output.data<double>(), grad_input.data<double>(),
                                                 N, C, L, L_out);
    } else if (grad_output.dtype() == DType::Float16) {
        adaptive_avgpool1d_backward_impl<Float16>(grad_output.data<Float16>(), grad_input.data<Float16>(),
                                                  N, C, L, L_out);
    } else if (grad_output.dtype() == DType::BFloat16) {
        adaptive_avgpool1d_backward_impl<BFloat16>(grad_output.data<BFloat16>(), grad_input.data<BFloat16>(),
                                                   N, C, L, L_out);
    } else {
        throw std::runtime_error("Unsupported dtype for adaptive_avgpool1d_backward");
    }

    return grad_input;
}

template<typename T>
void adaptive_maxpool1d_impl(const T* in_data, T* out_data, int64_t* idx_data,
                              int64_t N, int64_t C, int64_t L,
                              int64_t L_out) {
    #pragma omp parallel for collapse(3)
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            for (int64_t ol = 0; ol < L_out; ++ol) {
                int64_t l_start = (ol * L) / L_out;
                int64_t l_end = ((ol + 1) * L) / L_out;

                float max_val = -std::numeric_limits<float>::infinity();
                int64_t max_idx = 0;

                for (int64_t l = l_start; l < l_end; ++l) {
                    int64_t in_idx = (n * C + c) * L + l;
                    float val = static_cast<float>(in_data[in_idx]);
                    if (val > max_val) {
                        max_val = val;
                        max_idx = l;
                    }
                }

                int64_t out_idx = (n * C + c) * L_out + ol;
                out_data[out_idx] = static_cast<T>(max_val);
                idx_data[out_idx] = max_idx;
            }
        }
    }
}

auto adaptive_maxpool1d_kernel(const Tensor& input, int64_t output_size)
    -> std::pair<Tensor, Tensor> {
    auto shape = input.shape();
    int64_t N = shape[0];
    int64_t C = shape[1];
    int64_t L = shape[2];

    auto output = Tensor::empty_uninitialized({N, C, output_size}, input.dtype(), input.device());
    auto indices = Tensor::empty_uninitialized({N, C, output_size}, DType::Int64, input.device());
    int64_t* idx_data = indices.data<int64_t>();

    if (input.dtype() == DType::Float32) {
        adaptive_maxpool1d_impl<float>(input.data<float>(), output.data<float>(), idx_data,
                                       N, C, L, output_size);
    } else if (input.dtype() == DType::Float64) {
        adaptive_maxpool1d_impl<double>(input.data<double>(), output.data<double>(), idx_data,
                                        N, C, L, output_size);
    } else if (input.dtype() == DType::Float16) {
        adaptive_maxpool1d_impl<Float16>(input.data<Float16>(), output.data<Float16>(), idx_data,
                                         N, C, L, output_size);
    } else if (input.dtype() == DType::BFloat16) {
        adaptive_maxpool1d_impl<BFloat16>(input.data<BFloat16>(), output.data<BFloat16>(), idx_data,
                                          N, C, L, output_size);
    } else {
        throw std::runtime_error("Unsupported dtype for adaptive_maxpool1d");
    }

    return {output, indices};
}

template<typename T>
void adaptive_maxpool1d_backward_impl(const T* grad_out_data, const int64_t* idx_data, T* grad_in_data,
                                       int64_t N, int64_t C, int64_t L,
                                       int64_t L_out) {
    #pragma omp parallel for collapse(2) if(N * C > 4)
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            for (int64_t ol = 0; ol < L_out; ++ol) {
                int64_t out_idx = (n * C + c) * L_out + ol;
                int64_t max_idx = idx_data[out_idx];
                int64_t in_idx = (n * C + c) * L + max_idx;
                float val = static_cast<float>(grad_in_data[in_idx]) + static_cast<float>(grad_out_data[out_idx]);
                grad_in_data[in_idx] = static_cast<T>(val);
            }
        }
    }
}

auto adaptive_maxpool1d_backward_kernel(const Tensor& grad_output, const Tensor& indices,
                                         const std::vector<int64_t>& input_shape) -> Tensor {
    int64_t N = input_shape[0];
    int64_t C = input_shape[1];
    int64_t L = input_shape[2];

    auto grad_shape = grad_output.shape();
    int64_t L_out = grad_shape[2];

    auto grad_input = zeros(input_shape, grad_output.dtype(), grad_output.device());
    const int64_t* idx_data = indices.data<int64_t>();

    if (grad_output.dtype() == DType::Float32) {
        adaptive_maxpool1d_backward_impl<float>(grad_output.data<float>(), idx_data, grad_input.data<float>(),
                                                N, C, L, L_out);
    } else if (grad_output.dtype() == DType::Float64) {
        adaptive_maxpool1d_backward_impl<double>(grad_output.data<double>(), idx_data, grad_input.data<double>(),
                                                 N, C, L, L_out);
    } else if (grad_output.dtype() == DType::Float16) {
        adaptive_maxpool1d_backward_impl<Float16>(grad_output.data<Float16>(), idx_data, grad_input.data<Float16>(),
                                                  N, C, L, L_out);
    } else if (grad_output.dtype() == DType::BFloat16) {
        adaptive_maxpool1d_backward_impl<BFloat16>(grad_output.data<BFloat16>(), idx_data, grad_input.data<BFloat16>(),
                                                   N, C, L, L_out);
    } else {
        throw std::runtime_error("Unsupported dtype for adaptive_maxpool1d_backward");
    }

    return grad_input;
}

// ============================================================================
// 3D Pooling Kernels
// ============================================================================

template<typename T>
void maxpool3d_forward_impl(const T* in_data, T* out_data, int64_t* idx_data,
                             int64_t N, int64_t C,
                             int64_t D, int64_t H, int64_t W,
                             int64_t D_out, int64_t H_out, int64_t W_out,
                             int64_t kernel_size, int64_t stride, int64_t padding) {
    #pragma omp parallel for collapse(2) if(N * C > 4)
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            for (int64_t od = 0; od < D_out; ++od) {
                for (int64_t oh = 0; oh < H_out; ++oh) {
                    for (int64_t ow = 0; ow < W_out; ++ow) {
                        int64_t d_start = od * stride - padding;
                        int64_t h_start = oh * stride - padding;
                        int64_t w_start = ow * stride - padding;

                        float max_val = -std::numeric_limits<float>::infinity();
                        int64_t max_idx = 0;

                        for (int64_t kd = 0; kd < kernel_size; ++kd) {
                            for (int64_t kh = 0; kh < kernel_size; ++kh) {
                                for (int64_t kw = 0; kw < kernel_size; ++kw) {
                                    int64_t d = d_start + kd;
                                    int64_t h = h_start + kh;
                                    int64_t w = w_start + kw;
                                    if (d >= 0 && d < D && h >= 0 && h < H && w >= 0 && w < W) {
                                        int64_t in_idx = ((n * C + c) * D + d) * H * W + h * W + w;
                                        float val = static_cast<float>(in_data[in_idx]);
                                        if (val > max_val) {
                                            max_val = val;
                                            max_idx = d * H * W + h * W + w;
                                        }
                                    }
                                }
                            }
                        }

                        int64_t out_idx = ((n * C + c) * D_out + od) * H_out * W_out + oh * W_out + ow;
                        out_data[out_idx] = static_cast<T>(max_val);
                        idx_data[out_idx] = max_idx;
                    }
                }
            }
        }
    }
}

auto maxpool3d_forward_kernel(const Tensor& input, int64_t kernel_size,
                               int64_t stride, int64_t padding)
    -> std::pair<Tensor, Tensor> {
    auto shape = input.shape();
    int64_t N = shape[0];
    int64_t C = shape[1];
    int64_t D = shape[2];
    int64_t H = shape[3];
    int64_t W = shape[4];

    int64_t D_out = (D + 2 * padding - kernel_size) / stride + 1;
    int64_t H_out = (H + 2 * padding - kernel_size) / stride + 1;
    int64_t W_out = (W + 2 * padding - kernel_size) / stride + 1;

    auto output = Tensor::empty_uninitialized({N, C, D_out, H_out, W_out}, input.dtype(), input.device());
    auto indices = Tensor::empty_uninitialized({N, C, D_out, H_out, W_out}, DType::Int64, input.device());
    int64_t* idx_data = indices.data<int64_t>();

    if (input.dtype() == DType::Float32) {
        maxpool3d_forward_impl<float>(input.data<float>(), output.data<float>(), idx_data,
                                      N, C, D, H, W, D_out, H_out, W_out, kernel_size, stride, padding);
    } else if (input.dtype() == DType::Float64) {
        maxpool3d_forward_impl<double>(input.data<double>(), output.data<double>(), idx_data,
                                       N, C, D, H, W, D_out, H_out, W_out, kernel_size, stride, padding);
    } else if (input.dtype() == DType::Float16) {
        maxpool3d_forward_impl<Float16>(input.data<Float16>(), output.data<Float16>(), idx_data,
                                        N, C, D, H, W, D_out, H_out, W_out, kernel_size, stride, padding);
    } else if (input.dtype() == DType::BFloat16) {
        maxpool3d_forward_impl<BFloat16>(input.data<BFloat16>(), output.data<BFloat16>(), idx_data,
                                         N, C, D, H, W, D_out, H_out, W_out, kernel_size, stride, padding);
    } else {
        throw std::runtime_error("Unsupported dtype for maxpool3d_forward");
    }

    return {output, indices};
}

template<typename T>
void maxpool3d_backward_impl(const T* grad_out_data, const int64_t* idx_data, T* grad_in_data,
                              int64_t N, int64_t C,
                              int64_t D, int64_t H, int64_t W,
                              int64_t D_out, int64_t H_out, int64_t W_out) {
    #pragma omp parallel for collapse(2) if(N * C > 4)
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            for (int64_t od = 0; od < D_out; ++od) {
                for (int64_t oh = 0; oh < H_out; ++oh) {
                    for (int64_t ow = 0; ow < W_out; ++ow) {
                        int64_t out_idx = ((n * C + c) * D_out + od) * H_out * W_out + oh * W_out + ow;
                        int64_t max_idx = idx_data[out_idx];
                        // max_idx = d * H * W + h * W + w (spatial index within single channel)
                        int64_t in_idx = (n * C + c) * D * H * W + max_idx;
                        float val = static_cast<float>(grad_in_data[in_idx]) + static_cast<float>(grad_out_data[out_idx]);
                        grad_in_data[in_idx] = static_cast<T>(val);
                    }
                }
            }
        }
    }
}

auto maxpool3d_backward_kernel(const Tensor& grad_output, const Tensor& indices,
                                const std::vector<int64_t>& input_shape) -> Tensor {
    int64_t N = input_shape[0];
    int64_t C = input_shape[1];
    int64_t D = input_shape[2];
    int64_t H = input_shape[3];
    int64_t W = input_shape[4];

    auto grad_shape = grad_output.shape();
    int64_t D_out = grad_shape[2];
    int64_t H_out = grad_shape[3];
    int64_t W_out = grad_shape[4];

    auto grad_input = zeros(input_shape, grad_output.dtype(), grad_output.device());
    const int64_t* idx_data = indices.data<int64_t>();

    if (grad_output.dtype() == DType::Float32) {
        maxpool3d_backward_impl<float>(grad_output.data<float>(), idx_data, grad_input.data<float>(),
                                       N, C, D, H, W, D_out, H_out, W_out);
    } else if (grad_output.dtype() == DType::Float64) {
        maxpool3d_backward_impl<double>(grad_output.data<double>(), idx_data, grad_input.data<double>(),
                                        N, C, D, H, W, D_out, H_out, W_out);
    } else if (grad_output.dtype() == DType::Float16) {
        maxpool3d_backward_impl<Float16>(grad_output.data<Float16>(), idx_data, grad_input.data<Float16>(),
                                         N, C, D, H, W, D_out, H_out, W_out);
    } else if (grad_output.dtype() == DType::BFloat16) {
        maxpool3d_backward_impl<BFloat16>(grad_output.data<BFloat16>(), idx_data, grad_input.data<BFloat16>(),
                                          N, C, D, H, W, D_out, H_out, W_out);
    } else {
        throw std::runtime_error("Unsupported dtype for maxpool3d_backward");
    }

    return grad_input;
}

template<typename T>
void avgpool3d_forward_impl(const T* in_data, T* out_data,
                             int64_t N, int64_t C,
                             int64_t D, int64_t H, int64_t W,
                             int64_t D_out, int64_t H_out, int64_t W_out,
                             int64_t kernel_size, int64_t stride, int64_t padding) {
    #pragma omp parallel for collapse(2) if(N * C > 4)
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            for (int64_t od = 0; od < D_out; ++od) {
                for (int64_t oh = 0; oh < H_out; ++oh) {
                    for (int64_t ow = 0; ow < W_out; ++ow) {
                        int64_t d_start = od * stride - padding;
                        int64_t h_start = oh * stride - padding;
                        int64_t w_start = ow * stride - padding;

                        float sum = 0.0f;
                        int64_t count = 0;

                        for (int64_t kd = 0; kd < kernel_size; ++kd) {
                            for (int64_t kh = 0; kh < kernel_size; ++kh) {
                                for (int64_t kw = 0; kw < kernel_size; ++kw) {
                                    int64_t d = d_start + kd;
                                    int64_t h = h_start + kh;
                                    int64_t w = w_start + kw;
                                    if (d >= 0 && d < D && h >= 0 && h < H && w >= 0 && w < W) {
                                        sum += static_cast<float>(in_data[((n * C + c) * D + d) * H * W + h * W + w]);
                                        count++;
                                    }
                                }
                            }
                        }

                        int64_t out_idx = ((n * C + c) * D_out + od) * H_out * W_out + oh * W_out + ow;
                        out_data[out_idx] = static_cast<T>(count > 0 ? sum / count : 0.0f);
                    }
                }
            }
        }
    }
}

auto avgpool3d_forward_kernel(const Tensor& input, int64_t kernel_size,
                               int64_t stride, int64_t padding) -> Tensor {
    auto shape = input.shape();
    int64_t N = shape[0];
    int64_t C = shape[1];
    int64_t D = shape[2];
    int64_t H = shape[3];
    int64_t W = shape[4];

    int64_t D_out = (D + 2 * padding - kernel_size) / stride + 1;
    int64_t H_out = (H + 2 * padding - kernel_size) / stride + 1;
    int64_t W_out = (W + 2 * padding - kernel_size) / stride + 1;

    auto output = Tensor::empty_uninitialized({N, C, D_out, H_out, W_out}, input.dtype(), input.device());

    if (input.dtype() == DType::Float32) {
        avgpool3d_forward_impl<float>(input.data<float>(), output.data<float>(),
                                      N, C, D, H, W, D_out, H_out, W_out, kernel_size, stride, padding);
    } else if (input.dtype() == DType::Float64) {
        avgpool3d_forward_impl<double>(input.data<double>(), output.data<double>(),
                                       N, C, D, H, W, D_out, H_out, W_out, kernel_size, stride, padding);
    } else if (input.dtype() == DType::Float16) {
        avgpool3d_forward_impl<Float16>(input.data<Float16>(), output.data<Float16>(),
                                        N, C, D, H, W, D_out, H_out, W_out, kernel_size, stride, padding);
    } else if (input.dtype() == DType::BFloat16) {
        avgpool3d_forward_impl<BFloat16>(input.data<BFloat16>(), output.data<BFloat16>(),
                                         N, C, D, H, W, D_out, H_out, W_out, kernel_size, stride, padding);
    } else {
        throw std::runtime_error("Unsupported dtype for avgpool3d_forward");
    }

    return output;
}

template<typename T>
void avgpool3d_backward_impl(const T* grad_out_data, T* grad_in_data,
                              int64_t N, int64_t C,
                              int64_t D, int64_t H, int64_t W,
                              int64_t D_out, int64_t H_out, int64_t W_out,
                              int64_t kernel_size, int64_t stride, int64_t padding) {
    #pragma omp parallel for collapse(2) if(N * C > 4)
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            for (int64_t od = 0; od < D_out; ++od) {
                for (int64_t oh = 0; oh < H_out; ++oh) {
                    for (int64_t ow = 0; ow < W_out; ++ow) {
                        int64_t d_start = od * stride - padding;
                        int64_t h_start = oh * stride - padding;
                        int64_t w_start = ow * stride - padding;

                        int64_t count = 0;
                        for (int64_t kd = 0; kd < kernel_size; ++kd) {
                            for (int64_t kh = 0; kh < kernel_size; ++kh) {
                                for (int64_t kw = 0; kw < kernel_size; ++kw) {
                                    int64_t d = d_start + kd;
                                    int64_t h = h_start + kh;
                                    int64_t w = w_start + kw;
                                    if (d >= 0 && d < D && h >= 0 && h < H && w >= 0 && w < W) count++;
                                }
                            }
                        }

                        int64_t out_idx = ((n * C + c) * D_out + od) * H_out * W_out + oh * W_out + ow;
                        float grad_val = static_cast<float>(grad_out_data[out_idx]) / count;

                        for (int64_t kd = 0; kd < kernel_size; ++kd) {
                            for (int64_t kh = 0; kh < kernel_size; ++kh) {
                                for (int64_t kw = 0; kw < kernel_size; ++kw) {
                                    int64_t d = d_start + kd;
                                    int64_t h = h_start + kh;
                                    int64_t w = w_start + kw;
                                    if (d >= 0 && d < D && h >= 0 && h < H && w >= 0 && w < W) {
                                        int64_t idx = ((n * C + c) * D + d) * H * W + h * W + w;
                                        float val = static_cast<float>(grad_in_data[idx]) + grad_val;
                                        grad_in_data[idx] = static_cast<T>(val);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

auto avgpool3d_backward_kernel(const Tensor& grad_output,
                                const std::vector<int64_t>& input_shape,
                                int64_t kernel_size, int64_t stride,
                                int64_t padding) -> Tensor {
    int64_t N = input_shape[0];
    int64_t C = input_shape[1];
    int64_t D = input_shape[2];
    int64_t H = input_shape[3];
    int64_t W = input_shape[4];

    auto grad_shape = grad_output.shape();
    int64_t D_out = grad_shape[2];
    int64_t H_out = grad_shape[3];
    int64_t W_out = grad_shape[4];

    auto grad_input = zeros(input_shape, grad_output.dtype(), grad_output.device());

    if (grad_output.dtype() == DType::Float32) {
        avgpool3d_backward_impl<float>(grad_output.data<float>(), grad_input.data<float>(),
                                       N, C, D, H, W, D_out, H_out, W_out, kernel_size, stride, padding);
    } else if (grad_output.dtype() == DType::Float64) {
        avgpool3d_backward_impl<double>(grad_output.data<double>(), grad_input.data<double>(),
                                        N, C, D, H, W, D_out, H_out, W_out, kernel_size, stride, padding);
    } else if (grad_output.dtype() == DType::Float16) {
        avgpool3d_backward_impl<Float16>(grad_output.data<Float16>(), grad_input.data<Float16>(),
                                         N, C, D, H, W, D_out, H_out, W_out, kernel_size, stride, padding);
    } else if (grad_output.dtype() == DType::BFloat16) {
        avgpool3d_backward_impl<BFloat16>(grad_output.data<BFloat16>(), grad_input.data<BFloat16>(),
                                          N, C, D, H, W, D_out, H_out, W_out, kernel_size, stride, padding);
    } else {
        throw std::runtime_error("Unsupported dtype for avgpool3d_backward");
    }

    return grad_input;
}

// ============================================================================
// Adaptive 3D Pooling Kernels
// ============================================================================

template<typename T>
void adaptive_maxpool3d_impl(const T* in_data, T* out_data, int64_t* idx_data,
                              int64_t N, int64_t C,
                              int64_t D, int64_t H, int64_t W,
                              int64_t D_out, int64_t H_out, int64_t W_out) {
    #pragma omp parallel for collapse(2) if(N * C > 4)
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            for (int64_t od = 0; od < D_out; ++od) {
                int64_t d_start = (od * D) / D_out;
                int64_t d_end = ((od + 1) * D) / D_out;
                for (int64_t oh = 0; oh < H_out; ++oh) {
                    int64_t h_start = (oh * H) / H_out;
                    int64_t h_end = ((oh + 1) * H) / H_out;
                    for (int64_t ow = 0; ow < W_out; ++ow) {
                        int64_t w_start = (ow * W) / W_out;
                        int64_t w_end = ((ow + 1) * W) / W_out;

                        float max_val = -std::numeric_limits<float>::infinity();
                        int64_t max_idx = 0;

                        for (int64_t di = d_start; di < d_end; ++di) {
                            for (int64_t hi = h_start; hi < h_end; ++hi) {
                                for (int64_t wi = w_start; wi < w_end; ++wi) {
                                    int64_t in_idx = ((n * C + c) * D + di) * H * W + hi * W + wi;
                                    float val = static_cast<float>(in_data[in_idx]);
                                    if (val > max_val) {
                                        max_val = val;
                                        max_idx = di * H * W + hi * W + wi;
                                    }
                                }
                            }
                        }

                        int64_t out_idx = ((n * C + c) * D_out + od) * H_out * W_out + oh * W_out + ow;
                        out_data[out_idx] = static_cast<T>(max_val);
                        idx_data[out_idx] = max_idx;
                    }
                }
            }
        }
    }
}

auto adaptive_maxpool3d_kernel(const Tensor& input,
                                int64_t output_d, int64_t output_h, int64_t output_w)
    -> std::pair<Tensor, Tensor> {
    auto shape = input.shape();
    int64_t N = shape[0];
    int64_t C = shape[1];
    int64_t D = shape[2];
    int64_t H = shape[3];
    int64_t W = shape[4];

    auto output = Tensor::empty_uninitialized({N, C, output_d, output_h, output_w}, input.dtype(), input.device());
    auto indices = Tensor::empty_uninitialized({N, C, output_d, output_h, output_w}, DType::Int64, input.device());
    int64_t* idx_data = indices.data<int64_t>();

    if (input.dtype() == DType::Float32) {
        adaptive_maxpool3d_impl<float>(input.data<float>(), output.data<float>(), idx_data,
                                       N, C, D, H, W, output_d, output_h, output_w);
    } else if (input.dtype() == DType::Float64) {
        adaptive_maxpool3d_impl<double>(input.data<double>(), output.data<double>(), idx_data,
                                        N, C, D, H, W, output_d, output_h, output_w);
    } else if (input.dtype() == DType::Float16) {
        adaptive_maxpool3d_impl<Float16>(input.data<Float16>(), output.data<Float16>(), idx_data,
                                         N, C, D, H, W, output_d, output_h, output_w);
    } else if (input.dtype() == DType::BFloat16) {
        adaptive_maxpool3d_impl<BFloat16>(input.data<BFloat16>(), output.data<BFloat16>(), idx_data,
                                          N, C, D, H, W, output_d, output_h, output_w);
    } else {
        throw std::runtime_error("Unsupported dtype for adaptive_maxpool3d");
    }

    return {output, indices};
}

template<typename T>
void adaptive_maxpool3d_backward_impl(const T* grad_out_data, const int64_t* idx_data, T* grad_in_data,
                                       int64_t N, int64_t C,
                                       int64_t D, int64_t H, int64_t W,
                                       int64_t D_out, int64_t H_out, int64_t W_out) {
    #pragma omp parallel for collapse(2) if(N * C > 4)
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            for (int64_t od = 0; od < D_out; ++od) {
                for (int64_t oh = 0; oh < H_out; ++oh) {
                    for (int64_t ow = 0; ow < W_out; ++ow) {
                        int64_t out_idx = ((n * C + c) * D_out + od) * H_out * W_out + oh * W_out + ow;
                        int64_t max_idx = idx_data[out_idx];
                        int64_t in_idx = (n * C + c) * D * H * W + max_idx;
                        float val = static_cast<float>(grad_in_data[in_idx]) + static_cast<float>(grad_out_data[out_idx]);
                        grad_in_data[in_idx] = static_cast<T>(val);
                    }
                }
            }
        }
    }
}

auto adaptive_maxpool3d_backward_kernel(const Tensor& grad_output, const Tensor& indices,
                                         const std::vector<int64_t>& input_shape) -> Tensor {
    int64_t N = input_shape[0];
    int64_t C = input_shape[1];
    int64_t D = input_shape[2];
    int64_t H = input_shape[3];
    int64_t W = input_shape[4];

    auto grad_shape = grad_output.shape();
    int64_t D_out = grad_shape[2];
    int64_t H_out = grad_shape[3];
    int64_t W_out = grad_shape[4];

    auto grad_input = zeros(input_shape, grad_output.dtype(), grad_output.device());
    const int64_t* idx_data = indices.data<int64_t>();

    if (grad_output.dtype() == DType::Float32) {
        adaptive_maxpool3d_backward_impl<float>(grad_output.data<float>(), idx_data, grad_input.data<float>(),
                                                N, C, D, H, W, D_out, H_out, W_out);
    } else if (grad_output.dtype() == DType::Float64) {
        adaptive_maxpool3d_backward_impl<double>(grad_output.data<double>(), idx_data, grad_input.data<double>(),
                                                 N, C, D, H, W, D_out, H_out, W_out);
    } else if (grad_output.dtype() == DType::Float16) {
        adaptive_maxpool3d_backward_impl<Float16>(grad_output.data<Float16>(), idx_data, grad_input.data<Float16>(),
                                                  N, C, D, H, W, D_out, H_out, W_out);
    } else if (grad_output.dtype() == DType::BFloat16) {
        adaptive_maxpool3d_backward_impl<BFloat16>(grad_output.data<BFloat16>(), idx_data, grad_input.data<BFloat16>(),
                                                   N, C, D, H, W, D_out, H_out, W_out);
    } else {
        throw std::runtime_error("Unsupported dtype for adaptive_maxpool3d_backward");
    }

    return grad_input;
}

template<typename T>
void adaptive_avgpool3d_impl(const T* in_data, T* out_data,
                              int64_t N, int64_t C,
                              int64_t D, int64_t H, int64_t W,
                              int64_t D_out, int64_t H_out, int64_t W_out) {
    #pragma omp parallel for collapse(2) if(N * C > 4)
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            for (int64_t od = 0; od < D_out; ++od) {
                int64_t d_start = (od * D) / D_out;
                int64_t d_end = ((od + 1) * D) / D_out;
                for (int64_t oh = 0; oh < H_out; ++oh) {
                    int64_t h_start = (oh * H) / H_out;
                    int64_t h_end = ((oh + 1) * H) / H_out;
                    for (int64_t ow = 0; ow < W_out; ++ow) {
                        int64_t w_start = (ow * W) / W_out;
                        int64_t w_end = ((ow + 1) * W) / W_out;

                        float sum = 0.0f;
                        int64_t count = (d_end - d_start) * (h_end - h_start) * (w_end - w_start);

                        for (int64_t di = d_start; di < d_end; ++di) {
                            for (int64_t hi = h_start; hi < h_end; ++hi) {
                                for (int64_t wi = w_start; wi < w_end; ++wi) {
                                    sum += static_cast<float>(in_data[((n * C + c) * D + di) * H * W + hi * W + wi]);
                                }
                            }
                        }

                        int64_t out_idx = ((n * C + c) * D_out + od) * H_out * W_out + oh * W_out + ow;
                        out_data[out_idx] = static_cast<T>(count > 0 ? sum / count : 0.0f);
                    }
                }
            }
        }
    }
}

auto adaptive_avgpool3d_kernel(const Tensor& input,
                                int64_t output_d, int64_t output_h, int64_t output_w) -> Tensor {
    auto shape = input.shape();
    int64_t N = shape[0];
    int64_t C = shape[1];
    int64_t D = shape[2];
    int64_t H = shape[3];
    int64_t W = shape[4];

    auto output = Tensor::empty_uninitialized({N, C, output_d, output_h, output_w}, input.dtype(), input.device());

    if (input.dtype() == DType::Float32) {
        adaptive_avgpool3d_impl<float>(input.data<float>(), output.data<float>(),
                                       N, C, D, H, W, output_d, output_h, output_w);
    } else if (input.dtype() == DType::Float64) {
        adaptive_avgpool3d_impl<double>(input.data<double>(), output.data<double>(),
                                        N, C, D, H, W, output_d, output_h, output_w);
    } else if (input.dtype() == DType::Float16) {
        adaptive_avgpool3d_impl<Float16>(input.data<Float16>(), output.data<Float16>(),
                                         N, C, D, H, W, output_d, output_h, output_w);
    } else if (input.dtype() == DType::BFloat16) {
        adaptive_avgpool3d_impl<BFloat16>(input.data<BFloat16>(), output.data<BFloat16>(),
                                          N, C, D, H, W, output_d, output_h, output_w);
    } else {
        throw std::runtime_error("Unsupported dtype for adaptive_avgpool3d");
    }

    return output;
}

template<typename T>
void adaptive_avgpool3d_backward_impl(const T* grad_out_data, T* grad_in_data,
                                       int64_t N, int64_t C,
                                       int64_t D, int64_t H, int64_t W,
                                       int64_t D_out, int64_t H_out, int64_t W_out) {
    #pragma omp parallel for collapse(2) if(N * C > 4)
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            for (int64_t od = 0; od < D_out; ++od) {
                int64_t d_start = (od * D) / D_out;
                int64_t d_end = ((od + 1) * D) / D_out;
                for (int64_t oh = 0; oh < H_out; ++oh) {
                    int64_t h_start = (oh * H) / H_out;
                    int64_t h_end = ((oh + 1) * H) / H_out;
                    for (int64_t ow = 0; ow < W_out; ++ow) {
                        int64_t w_start = (ow * W) / W_out;
                        int64_t w_end = ((ow + 1) * W) / W_out;

                        int64_t count = (d_end - d_start) * (h_end - h_start) * (w_end - w_start);
                        int64_t out_idx = ((n * C + c) * D_out + od) * H_out * W_out + oh * W_out + ow;
                        float grad_val = static_cast<float>(grad_out_data[out_idx]) / count;

                        for (int64_t di = d_start; di < d_end; ++di) {
                            for (int64_t hi = h_start; hi < h_end; ++hi) {
                                for (int64_t wi = w_start; wi < w_end; ++wi) {
                                    int64_t idx = ((n * C + c) * D + di) * H * W + hi * W + wi;
                                    float val = static_cast<float>(grad_in_data[idx]) + grad_val;
                                    grad_in_data[idx] = static_cast<T>(val);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

auto adaptive_avgpool3d_backward_kernel(const Tensor& grad_output,
                                         const std::vector<int64_t>& input_shape) -> Tensor {
    int64_t N = input_shape[0];
    int64_t C = input_shape[1];
    int64_t D = input_shape[2];
    int64_t H = input_shape[3];
    int64_t W = input_shape[4];

    auto grad_shape = grad_output.shape();
    int64_t D_out = grad_shape[2];
    int64_t H_out = grad_shape[3];
    int64_t W_out = grad_shape[4];

    auto grad_input = zeros(input_shape, grad_output.dtype(), grad_output.device());

    if (grad_output.dtype() == DType::Float32) {
        adaptive_avgpool3d_backward_impl<float>(grad_output.data<float>(), grad_input.data<float>(),
                                                N, C, D, H, W, D_out, H_out, W_out);
    } else if (grad_output.dtype() == DType::Float64) {
        adaptive_avgpool3d_backward_impl<double>(grad_output.data<double>(), grad_input.data<double>(),
                                                 N, C, D, H, W, D_out, H_out, W_out);
    } else if (grad_output.dtype() == DType::Float16) {
        adaptive_avgpool3d_backward_impl<Float16>(grad_output.data<Float16>(), grad_input.data<Float16>(),
                                                  N, C, D, H, W, D_out, H_out, W_out);
    } else if (grad_output.dtype() == DType::BFloat16) {
        adaptive_avgpool3d_backward_impl<BFloat16>(grad_output.data<BFloat16>(), grad_input.data<BFloat16>(),
                                                   N, C, D, H, W, D_out, H_out, W_out);
    } else {
        throw std::runtime_error("Unsupported dtype for adaptive_avgpool3d_backward");
    }

    return grad_input;
}

// ============================================================================
// Phase 9: Fractional Max Pool 2D
// ============================================================================

template<typename T>
void fractional_maxpool2d_impl(const T* in_data, T* out_data, int64_t* idx_data,
                               int64_t N, int64_t C, int64_t H, int64_t W,
                               int64_t out_h, int64_t out_w,
                               const float* samples) {
    // samples layout: [N, C, 2] — fractional values in (0,1) for h and w axes
    #pragma omp parallel for collapse(2)
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            // Generate pool region boundaries from random samples
            // Using the disjoint-subsequence method from the paper:
            // "Fractional Max-Pooling" by Ben Graham (arXiv:1412.6071)
            // Per-(n,c) random offset from samples
            float sample_h = samples ? samples[(n * C + c) * 2 + 0] : 0.5f;
            float sample_w = samples ? samples[(n * C + c) * 2 + 1] : 0.5f;

            for (int64_t oh = 0; oh < out_h; ++oh) {
                for (int64_t ow = 0; ow < out_w; ++ow) {
                    // Compute pool region using pseudo-random sequence
                    int64_t h_start = static_cast<int64_t>(std::floor(
                        (oh + sample_h) * (static_cast<float>(H) / out_h) - sample_h));
                    int64_t h_end = static_cast<int64_t>(std::floor(
                        (oh + 1 + sample_h) * (static_cast<float>(H) / out_h) - sample_h));
                    int64_t w_start = static_cast<int64_t>(std::floor(
                        (ow + sample_w) * (static_cast<float>(W) / out_w) - sample_w));
                    int64_t w_end = static_cast<int64_t>(std::floor(
                        (ow + 1 + sample_w) * (static_cast<float>(W) / out_w) - sample_w));

                    h_start = std::max(h_start, int64_t{0});
                    h_end = std::min(h_end, H);
                    w_start = std::max(w_start, int64_t{0});
                    w_end = std::min(w_end, W);
                    if (h_end <= h_start) h_end = h_start + 1;
                    if (w_end <= w_start) w_end = w_start + 1;
                    h_end = std::min(h_end, H);
                    w_end = std::min(w_end, W);

                    float max_val = -std::numeric_limits<float>::infinity();
                    int64_t max_idx = h_start * W + w_start;

                    for (int64_t h = h_start; h < h_end; ++h) {
                        for (int64_t w = w_start; w < w_end; ++w) {
                            int64_t in_idx = ((n * C + c) * H + h) * W + w;
                            float val = static_cast<float>(in_data[in_idx]);
                            if (val > max_val) {
                                max_val = val;
                                max_idx = h * W + w;
                            }
                        }
                    }

                    int64_t out_idx = ((n * C + c) * out_h + oh) * out_w + ow;
                    out_data[out_idx] = static_cast<T>(max_val);
                    idx_data[out_idx] = max_idx;
                }
            }
        }
    }
}

auto fractional_maxpool2d_forward_kernel(const Tensor& input,
                                         int64_t out_h, int64_t out_w,
                                         const Tensor* random_samples)
    -> std::pair<Tensor, Tensor> {
    auto shape = input.shape();
    int64_t N = shape[0];
    int64_t C = shape[1];
    int64_t H = shape[2];
    int64_t W = shape[3];

    auto output = Tensor::empty_uninitialized({N, C, out_h, out_w}, input.dtype(), input.device());
    auto indices = Tensor::empty_uninitialized({N, C, out_h, out_w}, DType::Int64, input.device());
    int64_t* idx_data = indices.data<int64_t>();

    const float* samples_ptr = nullptr;
    if (random_samples && random_samples->numel() > 0) {
        samples_ptr = random_samples->data<float>();
    }

    if (input.dtype() == DType::Float32) {
        fractional_maxpool2d_impl<float>(input.data<float>(), output.data<float>(), idx_data,
                                          N, C, H, W, out_h, out_w, samples_ptr);
    } else if (input.dtype() == DType::Float64) {
        fractional_maxpool2d_impl<double>(input.data<double>(), output.data<double>(), idx_data,
                                           N, C, H, W, out_h, out_w, samples_ptr);
    } else if (input.dtype() == DType::Float16) {
        fractional_maxpool2d_impl<Float16>(input.data<Float16>(), output.data<Float16>(), idx_data,
                                            N, C, H, W, out_h, out_w, samples_ptr);
    } else if (input.dtype() == DType::BFloat16) {
        fractional_maxpool2d_impl<BFloat16>(input.data<BFloat16>(), output.data<BFloat16>(), idx_data,
                                             N, C, H, W, out_h, out_w, samples_ptr);
    } else {
        throw std::runtime_error("Unsupported dtype for fractional_maxpool2d_forward");
    }

    return {output, indices};
}

template<typename T>
void fractional_maxpool2d_backward_impl(const T* grad_out_data, const int64_t* idx_data,
                                        T* grad_in_data,
                                        int64_t N, int64_t C, int64_t H, int64_t W,
                                        int64_t out_h, int64_t out_w) {
    int64_t in_spatial = H * W;
    int64_t out_spatial = out_h * out_w;

    // Zero the gradient buffer
    std::fill_n(grad_in_data, N * C * in_spatial, T{});

    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            int64_t base_in = (n * C + c) * in_spatial;
            int64_t base_out = (n * C + c) * out_spatial;
            for (int64_t i = 0; i < out_spatial; ++i) {
                int64_t max_idx = idx_data[base_out + i];
                float acc = static_cast<float>(grad_in_data[base_in + max_idx]);
                acc += static_cast<float>(grad_out_data[base_out + i]);
                grad_in_data[base_in + max_idx] = static_cast<T>(acc);
            }
        }
    }
}

auto fractional_maxpool2d_backward_kernel(const Tensor& grad_output, const Tensor& indices,
                                          const std::vector<int64_t>& input_shape) -> Tensor {
    auto grad_shape = grad_output.shape();
    int64_t N = input_shape[0];
    int64_t C = input_shape[1];
    int64_t H = input_shape[2];
    int64_t W = input_shape[3];
    int64_t out_h = grad_shape[2];
    int64_t out_w = grad_shape[3];

    auto grad_input = zeros(input_shape, grad_output.dtype(), grad_output.device());
    const int64_t* idx_data = indices.data<int64_t>();

    if (grad_output.dtype() == DType::Float32) {
        fractional_maxpool2d_backward_impl<float>(grad_output.data<float>(), idx_data,
            grad_input.data<float>(), N, C, H, W, out_h, out_w);
    } else if (grad_output.dtype() == DType::Float64) {
        fractional_maxpool2d_backward_impl<double>(grad_output.data<double>(), idx_data,
            grad_input.data<double>(), N, C, H, W, out_h, out_w);
    } else if (grad_output.dtype() == DType::Float16) {
        fractional_maxpool2d_backward_impl<Float16>(grad_output.data<Float16>(), idx_data,
            grad_input.data<Float16>(), N, C, H, W, out_h, out_w);
    } else if (grad_output.dtype() == DType::BFloat16) {
        fractional_maxpool2d_backward_impl<BFloat16>(grad_output.data<BFloat16>(), idx_data,
            grad_input.data<BFloat16>(), N, C, H, W, out_h, out_w);
    } else {
        throw std::runtime_error("Unsupported dtype for fractional_maxpool2d_backward");
    }

    return grad_input;
}

// ============================================================================
// Phase 9: Fractional Max Pool 3D
// ============================================================================

template<typename T>
void fractional_maxpool3d_impl(const T* in_data, T* out_data, int64_t* idx_data,
                               int64_t N, int64_t C, int64_t D, int64_t H, int64_t W,
                               int64_t out_d, int64_t out_h, int64_t out_w,
                               const float* samples) {
    #pragma omp parallel for collapse(2)
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            float sample_d = samples ? samples[(n * C + c) * 3 + 0] : 0.5f;
            float sample_h = samples ? samples[(n * C + c) * 3 + 1] : 0.5f;
            float sample_w = samples ? samples[(n * C + c) * 3 + 2] : 0.5f;

            for (int64_t od = 0; od < out_d; ++od) {
                for (int64_t oh = 0; oh < out_h; ++oh) {
                    for (int64_t ow = 0; ow < out_w; ++ow) {
                        int64_t d_start = static_cast<int64_t>(std::floor(
                            (od + sample_d) * (static_cast<float>(D) / out_d) - sample_d));
                        int64_t d_end = static_cast<int64_t>(std::floor(
                            (od + 1 + sample_d) * (static_cast<float>(D) / out_d) - sample_d));
                        int64_t h_start = static_cast<int64_t>(std::floor(
                            (oh + sample_h) * (static_cast<float>(H) / out_h) - sample_h));
                        int64_t h_end = static_cast<int64_t>(std::floor(
                            (oh + 1 + sample_h) * (static_cast<float>(H) / out_h) - sample_h));
                        int64_t w_start = static_cast<int64_t>(std::floor(
                            (ow + sample_w) * (static_cast<float>(W) / out_w) - sample_w));
                        int64_t w_end = static_cast<int64_t>(std::floor(
                            (ow + 1 + sample_w) * (static_cast<float>(W) / out_w) - sample_w));

                        d_start = std::max(d_start, int64_t{0}); d_end = std::min(d_end, D);
                        h_start = std::max(h_start, int64_t{0}); h_end = std::min(h_end, H);
                        w_start = std::max(w_start, int64_t{0}); w_end = std::min(w_end, W);
                        if (d_end <= d_start) d_end = d_start + 1;
                        if (h_end <= h_start) h_end = h_start + 1;
                        if (w_end <= w_start) w_end = w_start + 1;
                        d_end = std::min(d_end, D);
                        h_end = std::min(h_end, H);
                        w_end = std::min(w_end, W);

                        float max_val = -std::numeric_limits<float>::infinity();
                        int64_t max_idx = d_start * H * W + h_start * W + w_start;

                        for (int64_t d = d_start; d < d_end; ++d) {
                            for (int64_t h = h_start; h < h_end; ++h) {
                                for (int64_t w = w_start; w < w_end; ++w) {
                                    int64_t in_idx = (((n * C + c) * D + d) * H + h) * W + w;
                                    float val = static_cast<float>(in_data[in_idx]);
                                    if (val > max_val) {
                                        max_val = val;
                                        max_idx = (d * H + h) * W + w;
                                    }
                                }
                            }
                        }

                        int64_t out_idx = (((n * C + c) * out_d + od) * out_h + oh) * out_w + ow;
                        out_data[out_idx] = static_cast<T>(max_val);
                        idx_data[out_idx] = max_idx;
                    }
                }
            }
        }
    }
}

auto fractional_maxpool3d_forward_kernel(const Tensor& input,
                                         int64_t out_d, int64_t out_h, int64_t out_w,
                                         const Tensor* random_samples)
    -> std::pair<Tensor, Tensor> {
    auto shape = input.shape();
    int64_t N = shape[0], C = shape[1], D = shape[2], H = shape[3], W = shape[4];

    auto output = Tensor::empty_uninitialized({N, C, out_d, out_h, out_w}, input.dtype(), input.device());
    auto indices = Tensor::empty_uninitialized({N, C, out_d, out_h, out_w}, DType::Int64, input.device());
    int64_t* idx_data = indices.data<int64_t>();

    const float* samples_ptr = nullptr;
    if (random_samples && random_samples->numel() > 0) {
        samples_ptr = random_samples->data<float>();
    }

    if (input.dtype() == DType::Float32) {
        fractional_maxpool3d_impl<float>(input.data<float>(), output.data<float>(), idx_data,
                                          N, C, D, H, W, out_d, out_h, out_w, samples_ptr);
    } else if (input.dtype() == DType::Float64) {
        fractional_maxpool3d_impl<double>(input.data<double>(), output.data<double>(), idx_data,
                                           N, C, D, H, W, out_d, out_h, out_w, samples_ptr);
    } else if (input.dtype() == DType::Float16) {
        fractional_maxpool3d_impl<Float16>(input.data<Float16>(), output.data<Float16>(), idx_data,
                                            N, C, D, H, W, out_d, out_h, out_w, samples_ptr);
    } else if (input.dtype() == DType::BFloat16) {
        fractional_maxpool3d_impl<BFloat16>(input.data<BFloat16>(), output.data<BFloat16>(), idx_data,
                                             N, C, D, H, W, out_d, out_h, out_w, samples_ptr);
    } else {
        throw std::runtime_error("Unsupported dtype for fractional_maxpool3d_forward");
    }

    return {output, indices};
}

template<typename T>
void fractional_maxpool3d_backward_impl(const T* grad_out_data, const int64_t* idx_data,
                                        T* grad_in_data,
                                        int64_t N, int64_t C,
                                        int64_t D, int64_t H, int64_t W,
                                        int64_t out_d, int64_t out_h, int64_t out_w) {
    int64_t in_spatial = D * H * W;
    int64_t out_spatial = out_d * out_h * out_w;

    std::memset(grad_in_data, 0, N * C * in_spatial * sizeof(T));

    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            int64_t base_in = (n * C + c) * in_spatial;
            int64_t base_out = (n * C + c) * out_spatial;
            for (int64_t i = 0; i < out_spatial; ++i) {
                int64_t max_idx = idx_data[base_out + i];
                float acc = static_cast<float>(grad_in_data[base_in + max_idx]);
                acc += static_cast<float>(grad_out_data[base_out + i]);
                grad_in_data[base_in + max_idx] = static_cast<T>(acc);
            }
        }
    }
}

auto fractional_maxpool3d_backward_kernel(const Tensor& grad_output, const Tensor& indices,
                                          const std::vector<int64_t>& input_shape) -> Tensor {
    auto grad_shape = grad_output.shape();
    int64_t N = input_shape[0], C = input_shape[1];
    int64_t D = input_shape[2], H = input_shape[3], W = input_shape[4];
    int64_t out_d = grad_shape[2], out_h = grad_shape[3], out_w = grad_shape[4];

    auto grad_input = zeros(input_shape, grad_output.dtype(), grad_output.device());
    const int64_t* idx_data = indices.data<int64_t>();

    if (grad_output.dtype() == DType::Float32) {
        fractional_maxpool3d_backward_impl<float>(grad_output.data<float>(), idx_data,
            grad_input.data<float>(), N, C, D, H, W, out_d, out_h, out_w);
    } else if (grad_output.dtype() == DType::Float64) {
        fractional_maxpool3d_backward_impl<double>(grad_output.data<double>(), idx_data,
            grad_input.data<double>(), N, C, D, H, W, out_d, out_h, out_w);
    } else if (grad_output.dtype() == DType::Float16) {
        fractional_maxpool3d_backward_impl<Float16>(grad_output.data<Float16>(), idx_data,
            grad_input.data<Float16>(), N, C, D, H, W, out_d, out_h, out_w);
    } else if (grad_output.dtype() == DType::BFloat16) {
        fractional_maxpool3d_backward_impl<BFloat16>(grad_output.data<BFloat16>(), idx_data,
            grad_input.data<BFloat16>(), N, C, D, H, W, out_d, out_h, out_w);
    } else {
        throw std::runtime_error("Unsupported dtype for fractional_maxpool3d_backward");
    }

    return grad_input;
}

// ============================================================================
// Phase 9: Max Unpool 2D
// ============================================================================

template<typename T>
void max_unpool2d_impl(const T* in_data, const int64_t* idx_data, T* out_data,
                       int64_t N, int64_t C,
                       int64_t in_h, int64_t in_w,
                       int64_t out_h, int64_t out_w) {
    int64_t out_spatial = out_h * out_w;
    int64_t in_spatial = in_h * in_w;

    // Zero output
    std::fill_n(out_data, N * C * out_spatial, T{});

    #pragma omp parallel for collapse(2)
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            int64_t base_in = (n * C + c) * in_spatial;
            int64_t base_out = (n * C + c) * out_spatial;
            for (int64_t i = 0; i < in_spatial; ++i) {
                int64_t idx = idx_data[base_in + i];
                if (idx >= 0 && idx < out_spatial) {
                    out_data[base_out + idx] = in_data[base_in + i];
                }
            }
        }
    }
}

auto max_unpool2d_forward_kernel(const Tensor& input, const Tensor& indices,
                                 int64_t out_h, int64_t out_w) -> Tensor {
    auto shape = input.shape();
    int64_t N = shape[0], C = shape[1], in_h = shape[2], in_w = shape[3];

    auto output = Tensor::empty_uninitialized({N, C, out_h, out_w}, input.dtype(), input.device());
    const int64_t* idx_data = indices.data<int64_t>();

    if (input.dtype() == DType::Float32) {
        max_unpool2d_impl<float>(input.data<float>(), idx_data, output.data<float>(),
                                  N, C, in_h, in_w, out_h, out_w);
    } else if (input.dtype() == DType::Float64) {
        max_unpool2d_impl<double>(input.data<double>(), idx_data, output.data<double>(),
                                   N, C, in_h, in_w, out_h, out_w);
    } else if (input.dtype() == DType::Float16) {
        max_unpool2d_impl<Float16>(input.data<Float16>(), idx_data, output.data<Float16>(),
                                    N, C, in_h, in_w, out_h, out_w);
    } else if (input.dtype() == DType::BFloat16) {
        max_unpool2d_impl<BFloat16>(input.data<BFloat16>(), idx_data, output.data<BFloat16>(),
                                     N, C, in_h, in_w, out_h, out_w);
    } else {
        throw std::runtime_error("Unsupported dtype for max_unpool2d_forward");
    }

    return output;
}

template<typename T>
void max_unpool2d_backward_impl(const T* grad_out_data, const int64_t* idx_data,
                                T* grad_in_data,
                                int64_t N, int64_t C,
                                int64_t in_h, int64_t in_w,
                                int64_t out_h, int64_t out_w) {
    int64_t in_spatial = in_h * in_w;
    int64_t out_spatial = out_h * out_w;

    #pragma omp parallel for collapse(2)
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            int64_t base_in = (n * C + c) * in_spatial;
            int64_t base_out = (n * C + c) * out_spatial;
            for (int64_t i = 0; i < in_spatial; ++i) {
                int64_t idx = idx_data[base_in + i];
                if (idx >= 0 && idx < out_spatial) {
                    grad_in_data[base_in + i] = grad_out_data[base_out + idx];
                } else {
                    grad_in_data[base_in + i] = T{};
                }
            }
        }
    }
}

auto max_unpool2d_backward_kernel(const Tensor& grad_output, const Tensor& indices,
                                  const std::vector<int64_t>& input_shape) -> Tensor {
    auto grad_shape = grad_output.shape();
    int64_t N = input_shape[0], C = input_shape[1];
    int64_t in_h = input_shape[2], in_w = input_shape[3];
    int64_t out_h = grad_shape[2], out_w = grad_shape[3];

    auto grad_input = Tensor::empty_uninitialized(input_shape, grad_output.dtype(), grad_output.device());
    const int64_t* idx_data = indices.data<int64_t>();

    if (grad_output.dtype() == DType::Float32) {
        max_unpool2d_backward_impl<float>(grad_output.data<float>(), idx_data,
            grad_input.data<float>(), N, C, in_h, in_w, out_h, out_w);
    } else if (grad_output.dtype() == DType::Float64) {
        max_unpool2d_backward_impl<double>(grad_output.data<double>(), idx_data,
            grad_input.data<double>(), N, C, in_h, in_w, out_h, out_w);
    } else if (grad_output.dtype() == DType::Float16) {
        max_unpool2d_backward_impl<Float16>(grad_output.data<Float16>(), idx_data,
            grad_input.data<Float16>(), N, C, in_h, in_w, out_h, out_w);
    } else if (grad_output.dtype() == DType::BFloat16) {
        max_unpool2d_backward_impl<BFloat16>(grad_output.data<BFloat16>(), idx_data,
            grad_input.data<BFloat16>(), N, C, in_h, in_w, out_h, out_w);
    } else {
        throw std::runtime_error("Unsupported dtype for max_unpool2d_backward");
    }

    return grad_input;
}

// ============================================================================
// Phase 9: Max Unpool 3D
// ============================================================================

template<typename T>
void max_unpool3d_impl(const T* in_data, const int64_t* idx_data, T* out_data,
                       int64_t N, int64_t C,
                       int64_t in_d, int64_t in_h, int64_t in_w,
                       int64_t out_d, int64_t out_h, int64_t out_w) {
    int64_t out_spatial = out_d * out_h * out_w;
    int64_t in_spatial = in_d * in_h * in_w;

    std::memset(out_data, 0, N * C * out_spatial * sizeof(T));

    #pragma omp parallel for collapse(2)
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            int64_t base_in = (n * C + c) * in_spatial;
            int64_t base_out = (n * C + c) * out_spatial;
            for (int64_t i = 0; i < in_spatial; ++i) {
                int64_t idx = idx_data[base_in + i];
                if (idx >= 0 && idx < out_spatial) {
                    out_data[base_out + idx] = in_data[base_in + i];
                }
            }
        }
    }
}

auto max_unpool3d_forward_kernel(const Tensor& input, const Tensor& indices,
                                 int64_t out_d, int64_t out_h, int64_t out_w) -> Tensor {
    auto shape = input.shape();
    int64_t N = shape[0], C = shape[1], in_d = shape[2], in_h = shape[3], in_w = shape[4];

    auto output = Tensor::empty_uninitialized({N, C, out_d, out_h, out_w}, input.dtype(), input.device());
    const int64_t* idx_data = indices.data<int64_t>();

    if (input.dtype() == DType::Float32) {
        max_unpool3d_impl<float>(input.data<float>(), idx_data, output.data<float>(),
                                  N, C, in_d, in_h, in_w, out_d, out_h, out_w);
    } else if (input.dtype() == DType::Float64) {
        max_unpool3d_impl<double>(input.data<double>(), idx_data, output.data<double>(),
                                   N, C, in_d, in_h, in_w, out_d, out_h, out_w);
    } else if (input.dtype() == DType::Float16) {
        max_unpool3d_impl<Float16>(input.data<Float16>(), idx_data, output.data<Float16>(),
                                    N, C, in_d, in_h, in_w, out_d, out_h, out_w);
    } else if (input.dtype() == DType::BFloat16) {
        max_unpool3d_impl<BFloat16>(input.data<BFloat16>(), idx_data, output.data<BFloat16>(),
                                     N, C, in_d, in_h, in_w, out_d, out_h, out_w);
    } else {
        throw std::runtime_error("Unsupported dtype for max_unpool3d_forward");
    }

    return output;
}

template<typename T>
void max_unpool3d_backward_impl(const T* grad_out_data, const int64_t* idx_data,
                                T* grad_in_data,
                                int64_t N, int64_t C,
                                int64_t in_d, int64_t in_h, int64_t in_w,
                                int64_t out_d, int64_t out_h, int64_t out_w) {
    int64_t in_spatial = in_d * in_h * in_w;
    int64_t out_spatial = out_d * out_h * out_w;

    #pragma omp parallel for collapse(2)
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            int64_t base_in = (n * C + c) * in_spatial;
            int64_t base_out = (n * C + c) * out_spatial;
            for (int64_t i = 0; i < in_spatial; ++i) {
                int64_t idx = idx_data[base_in + i];
                if (idx >= 0 && idx < out_spatial) {
                    grad_in_data[base_in + i] = grad_out_data[base_out + idx];
                } else {
                    grad_in_data[base_in + i] = T{};
                }
            }
        }
    }
}

auto max_unpool3d_backward_kernel(const Tensor& grad_output, const Tensor& indices,
                                  const std::vector<int64_t>& input_shape) -> Tensor {
    auto grad_shape = grad_output.shape();
    int64_t N = input_shape[0], C = input_shape[1];
    int64_t in_d = input_shape[2], in_h = input_shape[3], in_w = input_shape[4];
    int64_t out_d = grad_shape[2], out_h = grad_shape[3], out_w = grad_shape[4];

    auto grad_input = Tensor::empty_uninitialized(input_shape, grad_output.dtype(), grad_output.device());
    const int64_t* idx_data = indices.data<int64_t>();

    if (grad_output.dtype() == DType::Float32) {
        max_unpool3d_backward_impl<float>(grad_output.data<float>(), idx_data,
            grad_input.data<float>(), N, C, in_d, in_h, in_w, out_d, out_h, out_w);
    } else if (grad_output.dtype() == DType::Float64) {
        max_unpool3d_backward_impl<double>(grad_output.data<double>(), idx_data,
            grad_input.data<double>(), N, C, in_d, in_h, in_w, out_d, out_h, out_w);
    } else if (grad_output.dtype() == DType::Float16) {
        max_unpool3d_backward_impl<Float16>(grad_output.data<Float16>(), idx_data,
            grad_input.data<Float16>(), N, C, in_d, in_h, in_w, out_d, out_h, out_w);
    } else if (grad_output.dtype() == DType::BFloat16) {
        max_unpool3d_backward_impl<BFloat16>(grad_output.data<BFloat16>(), idx_data,
            grad_input.data<BFloat16>(), N, C, in_d, in_h, in_w, out_d, out_h, out_w);
    } else {
        throw std::runtime_error("Unsupported dtype for max_unpool3d_backward");
    }

    return grad_input;
}

} // namespace cpu
} // namespace tenzor
