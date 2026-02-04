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
#include <list>
#include <unordered_map>
#endif

namespace tenzor {
namespace cpu {

// ============================================================================
// oneDNN Pooling Helpers with Primitive Caching
// ============================================================================
#ifdef TENZOR_USE_ONEDNN
static thread_local dnnl::engine g_pooling_engine(dnnl::engine::kind::cpu, 0);
static thread_local dnnl::stream g_pooling_stream(g_pooling_engine);

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

class PoolingPrimitiveCache {
public:
    std::shared_ptr<PoolingCachedPrimitive> get(const PoolingCacheKey& key) {
        auto it = cache_.find(key);
        if (it != cache_.end()) {
            lru_list_.remove(key);
            lru_list_.push_front(key);
            return it->second;
        }
        return nullptr;
    }

    void put(const PoolingCacheKey& key, std::shared_ptr<PoolingCachedPrimitive> value) {
        if (cache_.size() >= POOLING_CACHE_SIZE) {
            auto evict_key = lru_list_.back();
            lru_list_.pop_back();
            cache_.erase(evict_key);
        }
        cache_[key] = value;
        lru_list_.push_front(key);
    }

private:
    std::unordered_map<PoolingCacheKey, std::shared_ptr<PoolingCachedPrimitive>, PoolingCacheKeyHash> cache_;
    std::list<PoolingCacheKey> lru_list_;
};

static thread_local PoolingPrimitiveCache g_pooling_cache;

// oneDNN max pooling forward with caching
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

        // Create cache key
        PoolingCacheKey cache_key{dnnl::algorithm::pooling_max,
                                   N, C, H, W, H_out, W_out,
                                   kernel_size, stride, padding};

        auto cached = g_pooling_cache.get(cache_key);

        if (!cached) {
            // Cache miss - create new primitive and cache it
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
                dnnl::algorithm::pooling_max,
                cached->src_md, cached->dst_md,
                stride_dims, kernel_dims,
                dilation_dims, padding_l, padding_r);
            cached->prim = dnnl::pooling_forward(pool_pd);

            g_pooling_cache.put(cache_key, cached);
        }

        // Create memory objects with user data (fast - just wraps pointers)
        auto src_mem = dnnl::memory(cached->src_md, engine, const_cast<float*>(input));
        auto dst_mem = dnnl::memory(cached->dst_md, engine, output);

        // Execute cached primitive
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
        auto& engine = g_pooling_engine;
        auto& stream = g_pooling_stream;

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
        avgpool2d_forward_impl<float>(input.data<float>(), output.data<float>(),
                                      N, C, H, W, H_out, W_out, kernel_size, stride, padding);
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
        adaptive_avgpool2d_impl<float>(input.data<float>(), output.data<float>(), N, C, H, W, output_h, output_w);
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

} // namespace cpu
} // namespace tenzor
