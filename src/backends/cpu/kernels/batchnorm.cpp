#include "tenzor/core/tensor.hpp"
#include "tenzor/core/dtype.hpp"
#include <cmath>
#include <vector>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

// Intel oneDNN for optimized batch normalization (2-3x faster)
#ifdef TENZOR_USE_ONEDNN
#include <dnnl.hpp>
#endif

namespace tenzor {
namespace cpu {

// ============================================================================
// Float16 Arithmetic Helper Functions
// ============================================================================
// These inline helpers allow Float16 to work with template code that uses
// arithmetic operators. Operations are performed in Float32 precision.

inline Float16 operator+(const Float16& a, const Float16& b) {
    return Float16(static_cast<float>(a) + static_cast<float>(b));
}

inline Float16 operator-(const Float16& a, const Float16& b) {
    return Float16(static_cast<float>(a) - static_cast<float>(b));
}

inline Float16 operator*(const Float16& a, const Float16& b) {
    return Float16(static_cast<float>(a) * static_cast<float>(b));
}

inline Float16 operator/(const Float16& a, const Float16& b) {
    return Float16(static_cast<float>(a) / static_cast<float>(b));
}

inline Float16& operator+=(Float16& a, const Float16& b) {
    a = Float16(static_cast<float>(a) + static_cast<float>(b));
    return a;
}

inline Float16& operator-=(Float16& a, const Float16& b) {
    a = Float16(static_cast<float>(a) - static_cast<float>(b));
    return a;
}

inline Float16& operator*=(Float16& a, const Float16& b) {
    a = Float16(static_cast<float>(a) * static_cast<float>(b));
    return a;
}

inline Float16& operator/=(Float16& a, const Float16& b) {
    a = Float16(static_cast<float>(a) / static_cast<float>(b));
    return a;
}

// Math helper templates for Float16 support
template<typename T>
inline T safe_sqrt(const T& x) {
    return std::sqrt(x);
}

template<>
inline Float16 safe_sqrt<Float16>(const Float16& x) {
    return Float16(safe_sqrt(static_cast<float>(x)));
}

// ============================================================================
// BatchNorm2d Mean/Variance Computation
// ============================================================================

// Compute per-channel mean and variance
// Input: [N, C, H, W] - NCHW format
// Output: mean[C], variance[C]
template<typename T>
void batchnorm_mean_var_impl(const T* input,
                             T* mean,
                             T* variance,
                             int64_t N,
                             int64_t C,
                             int64_t H,
                             int64_t W) {
    int64_t spatial_size = H * W;
    int64_t total_elements = N * spatial_size;

    // Check for division by zero
    if (total_elements == 0) {
        throw std::runtime_error("BatchNorm2d: Cannot compute mean/variance for empty tensor (total_elements = 0)");
    }

    // Compute mean and variance for each channel
    #pragma omp parallel for if(C > 1)
    for (int64_t c = 0; c < C; c++) {
        // Compute mean using Kahan summation for numerical stability
        T sum = T(0.0f);
        T compensation = T(0.0f);

        for (int64_t n = 0; n < N; n++) {
            for (int64_t h = 0; h < H; h++) {
                for (int64_t w = 0; w < W; w++) {
                    int64_t idx = ((n * C + c) * H + h) * W + w;
                    T value = input[idx];
                    T y = value - compensation;
                    T t = sum + y;
                    compensation = (t - sum) - y;
                    sum = t;
                }
            }
        }
        mean[c] = sum / T(static_cast<float>(total_elements));

        // Compute variance using Kahan summation
        T sum_sq_diff = T(0.0f);
        T var_compensation = T(0.0f);
        T channel_mean = mean[c];

        for (int64_t n = 0; n < N; n++) {
            for (int64_t h = 0; h < H; h++) {
                for (int64_t w = 0; w < W; w++) {
                    int64_t idx = ((n * C + c) * H + h) * W + w;
                    T diff = input[idx] - channel_mean;
                    T sq_diff = diff * diff;
                    T y = sq_diff - var_compensation;
                    T t = sum_sq_diff + y;
                    var_compensation = (t - sum_sq_diff) - y;
                    sum_sq_diff = t;
                }
            }
        }
        variance[c] = sum_sq_diff / T(static_cast<float>(total_elements));
    }
}

auto batchnorm2d_mean_var_kernel(const Tensor& input) -> std::vector<Tensor> {
    auto shape = input.shape();
    if (shape.size() != 4) {
        throw std::invalid_argument("batchnorm2d_mean_var expects 4D input (NCHW)");
    }

    int64_t N = shape[0];
    int64_t C = shape[1];
    int64_t H = shape[2];
    int64_t W = shape[3];

    // Allocate output tensors
    Tensor mean({C}, input.dtype(), input.device());
    Tensor variance({C}, input.dtype(), input.device());

    if (input.dtype() == DType::Float32) {
        batchnorm_mean_var_impl<float>(
            input.data<float>(),
            mean.data<float>(),
            variance.data<float>(),
            N, C, H, W
        );
    } else if (input.dtype() == DType::Float64) {
        batchnorm_mean_var_impl<double>(
            input.data<double>(),
            mean.data<double>(),
            variance.data<double>(),
            N, C, H, W
        );
    } else if (input.dtype() == DType::Float16) {
        batchnorm_mean_var_impl<Float16>(
            input.data<Float16>(),
            mean.data<Float16>(),
            variance.data<Float16>(),
            N, C, H, W
        );
    } else {
        throw std::runtime_error("BatchNorm2d only supports Float32, Float64, and Float16 dtypes");
    }

    return {mean, variance};
}

// ============================================================================
// BatchNorm2d Normalization Kernel
// ============================================================================

// Normalize: (x - mean) / sqrt(variance + epsilon)
template<typename T>
void batchnorm_forward_impl(const T* input,
                            T* output,
                            const T* mean,
                            const T* variance,
                            T epsilon,
                            int64_t N,
                            int64_t C,
                            int64_t H,
                            int64_t W) {
    int64_t spatial_size = H * W;
    int64_t total_size = N * C * spatial_size;

    #pragma omp parallel for if(total_size > 10000)
    for (int64_t idx = 0; idx < total_size; idx++) {
        // Decode NCHW index
        int64_t w = idx % W;
        int64_t h = (idx / W) % H;
        int64_t c = (idx / (W * H)) % C;

        T channel_mean = mean[c];
        T channel_var = variance[c];
        T invstd = T(1.0f) / safe_sqrt(channel_var + epsilon);

        output[idx] = (input[idx] - channel_mean) * invstd;
    }
}

auto batchnorm2d_forward_kernel(const Tensor& input,
                               const Tensor& mean,
                               const Tensor& variance,
                               float epsilon) -> Tensor {
    auto shape = input.shape();
    std::vector<int64_t> shape_vec(shape.begin(), shape.end());
    Tensor output(shape_vec, input.dtype(), input.device());

    int64_t N = shape[0];
    int64_t C = shape[1];
    int64_t H = shape[2];
    int64_t W = shape[3];

    if (input.dtype() == DType::Float32) {
        batchnorm_forward_impl<float>(
            input.data<float>(),
            output.data<float>(),
            mean.data<float>(),
            variance.data<float>(),
            epsilon,
            N, C, H, W
        );
    } else if (input.dtype() == DType::Float64) {
        batchnorm_forward_impl<double>(
            input.data<double>(),
            output.data<double>(),
            mean.data<double>(),
            variance.data<double>(),
            static_cast<double>(epsilon),
            N, C, H, W
        );
    } else if (input.dtype() == DType::Float16) {
        batchnorm_forward_impl<Float16>(
            input.data<Float16>(),
            output.data<Float16>(),
            mean.data<Float16>(),
            variance.data<Float16>(),
            Float16(static_cast<float>(epsilon)),
            N, C, H, W
        );
    } else {
        throw std::runtime_error("BatchNorm2d only supports Float32, Float64, and Float16 dtypes");
    }

    return output;
}

// ============================================================================
// BatchNorm2d Affine Transform Kernel
// ============================================================================

#ifdef TENZOR_USE_ONEDNN
// oneDNN-accelerated BatchNorm2d Forward with Affine Transform (Float32 only)
// Provides 2-3x speedup over scalar implementation
static bool batchnorm2d_forward_affine_onednn(
    const Tensor& input,
    Tensor& output,
    const Tensor& mean,
    const Tensor& variance,
    const Tensor& gamma,
    const Tensor& beta,
    float epsilon
) {
    // oneDNN only supports Float32 for now
    if (input.dtype() != DType::Float32) {
        return false;
    }

    try {
        auto shape = input.shape();
        int64_t N = shape[0];
        int64_t C = shape[1];
        int64_t H = shape[2];
        int64_t W = shape[3];

        // Create oneDNN engine and stream
        dnnl::engine engine(dnnl::engine::kind::cpu, 0);
        dnnl::stream stream(engine);

        // Memory descriptors
        dnnl::memory::dims src_dims = {N, C, H, W};
        auto src_md = dnnl::memory::desc(src_dims, dnnl::memory::data_type::f32,
                                          dnnl::memory::format_tag::nchw);

        // Create batch normalization primitive descriptor for inference
        // Use global stats (pre-computed mean/variance)
        // oneDNN requires both src and dst memory descriptors
        auto bn_pd = dnnl::batch_normalization_forward::primitive_desc(
            engine,
            dnnl::prop_kind::forward_inference,
            src_md,  // src_desc
            src_md,  // dst_desc (same as src for in-place capable op)
            epsilon,
            dnnl::normalization_flags::use_global_stats |
            dnnl::normalization_flags::use_scale |
            dnnl::normalization_flags::use_shift
        );

        // Create memory objects
        auto src_mem = dnnl::memory(src_md, engine, const_cast<float*>(input.data<float>()));
        auto dst_mem = dnnl::memory(src_md, engine, output.data<float>());

        // oneDNN expects scale and shift in specific format
        dnnl::memory::dims sc_dims = {C};
        auto sc_md = dnnl::memory::desc(sc_dims, dnnl::memory::data_type::f32,
                                         dnnl::memory::format_tag::a);

        auto scale_mem = dnnl::memory(sc_md, engine, const_cast<float*>(gamma.data<float>()));
        auto shift_mem = dnnl::memory(sc_md, engine, const_cast<float*>(beta.data<float>()));
        auto mean_mem = dnnl::memory(sc_md, engine, const_cast<float*>(mean.data<float>()));
        auto var_mem = dnnl::memory(sc_md, engine, const_cast<float*>(variance.data<float>()));

        // Create and execute batch normalization primitive
        auto bn_prim = dnnl::batch_normalization_forward(bn_pd);

        bn_prim.execute(stream, {
            {DNNL_ARG_SRC, src_mem},
            {DNNL_ARG_DST, dst_mem},
            {DNNL_ARG_SCALE, scale_mem},
            {DNNL_ARG_SHIFT, shift_mem},
            {DNNL_ARG_MEAN, mean_mem},
            {DNNL_ARG_VARIANCE, var_mem}
        });

        stream.wait();
        return true;

    } catch (const dnnl::error& e) {
        // oneDNN error, fall back to scalar implementation
        return false;
    }
}
#endif

// Combined normalization + affine: y = gamma * normalized + beta
template<typename T>
void batchnorm_forward_affine_impl(const T* input,
                                   T* output,
                                   const T* mean,
                                   const T* variance,
                                   const T* gamma,
                                   const T* beta,
                                   T epsilon,
                                   int64_t N,
                                   int64_t C,
                                   int64_t H,
                                   int64_t W) {
    int64_t spatial_size = H * W;
    int64_t total_size = N * C * spatial_size;

    #pragma omp parallel for if(total_size > 10000)
    for (int64_t idx = 0; idx < total_size; idx++) {
        // Decode NCHW index
        int64_t c = (idx / (H * W)) % C;

        T channel_mean = mean[c];
        T channel_var = variance[c];
        T invstd = T(1.0f) / safe_sqrt(channel_var + epsilon);

        T normalized = (input[idx] - channel_mean) * invstd;
        output[idx] = gamma[c] * normalized + beta[c];
    }
}

auto batchnorm2d_forward_affine_kernel(const Tensor& input,
                                       const Tensor& mean,
                                       const Tensor& variance,
                                       const Tensor& gamma,
                                       const Tensor& beta,
                                       float epsilon) -> Tensor {
    auto shape = input.shape();
    std::vector<int64_t> shape_vec(shape.begin(), shape.end());
    Tensor output(shape_vec, input.dtype(), input.device());

    int64_t N = shape[0];
    int64_t C = shape[1];
    int64_t H = shape[2];
    int64_t W = shape[3];

#ifdef TENZOR_USE_ONEDNN
    // Try oneDNN-accelerated batch normalization first (2-3x faster for Float32)
    if (batchnorm2d_forward_affine_onednn(input, output, mean, variance, gamma, beta, epsilon)) {
        return output;
    }
    // Fall through to scalar implementation if oneDNN not applicable
#endif

    if (input.dtype() == DType::Float32) {
        batchnorm_forward_affine_impl<float>(
            input.data<float>(),
            output.data<float>(),
            mean.data<float>(),
            variance.data<float>(),
            gamma.data<float>(),
            beta.data<float>(),
            epsilon,
            N, C, H, W
        );
    } else if (input.dtype() == DType::Float64) {
        batchnorm_forward_affine_impl<double>(
            input.data<double>(),
            output.data<double>(),
            mean.data<double>(),
            variance.data<double>(),
            gamma.data<double>(),
            beta.data<double>(),
            static_cast<double>(epsilon),
            N, C, H, W
        );
    } else if (input.dtype() == DType::Float16) {
        batchnorm_forward_affine_impl<Float16>(
            input.data<Float16>(),
            output.data<Float16>(),
            mean.data<Float16>(),
            variance.data<Float16>(),
            gamma.data<Float16>(),
            beta.data<Float16>(),
            Float16(static_cast<float>(epsilon)),
            N, C, H, W
        );
    } else {
        throw std::runtime_error("BatchNorm2d only supports Float32, Float64, and Float16 dtypes");
    }

    return output;
}

// ============================================================================
// BatchNorm2d Running Statistics Update Kernel
// ============================================================================

// Update running statistics: running = (1 - momentum) * running + momentum * batch
template<typename T>
void batchnorm_update_running_stats_impl(T* running_mean,
                                         T* running_var,
                                         const T* batch_mean,
                                         const T* batch_var,
                                         T momentum,
                                         int64_t C) {
    #pragma omp parallel for if(C > 100)
    for (int64_t c = 0; c < C; c++) {
        running_mean[c] = (T(1.0f) - momentum) * running_mean[c] + momentum * batch_mean[c];
        running_var[c] = (T(1.0f) - momentum) * running_var[c] + momentum * batch_var[c];
    }
}

auto batchnorm2d_update_running_stats_kernel(Tensor& running_mean,
                                             Tensor& running_var,
                                             const Tensor& batch_mean,
                                             const Tensor& batch_var,
                                             float momentum) -> void {
    int64_t C = batch_mean.shape()[0];

    if (running_mean.dtype() == DType::Float32) {
        batchnorm_update_running_stats_impl<float>(
            running_mean.data<float>(),
            running_var.data<float>(),
            batch_mean.data<float>(),
            batch_var.data<float>(),
            momentum,
            C
        );
    } else if (running_mean.dtype() == DType::Float64) {
        batchnorm_update_running_stats_impl<double>(
            running_mean.data<double>(),
            running_var.data<double>(),
            batch_mean.data<double>(),
            batch_var.data<double>(),
            static_cast<double>(momentum),
            C
        );
    } else if (running_mean.dtype() == DType::Float16) {
        batchnorm_update_running_stats_impl<Float16>(
            running_mean.data<Float16>(),
            running_var.data<Float16>(),
            batch_mean.data<Float16>(),
            batch_var.data<Float16>(),
            Float16(static_cast<float>(momentum)),
            C
        );
    } else {
        throw std::runtime_error("BatchNorm2d only supports Float32, Float64, and Float16 dtypes");
    }
}

// ============================================================================
// BatchNorm2d Backward Kernels
// ============================================================================

// Compute gradients w.r.t input, gamma, and beta
template<typename T>
void batchnorm_backward_impl(const T* grad_output,
                            const T* input,
                            T* grad_input,
                            T* grad_gamma,
                            T* grad_beta,
                            const T* mean,
                            const T* variance,
                            const T* gamma,
                            T epsilon,
                            int64_t N,
                            int64_t C,
                            int64_t H,
                            int64_t W) {
    int64_t spatial_size = H * W;
    int64_t total_elements = N * spatial_size;

    // Check for division by zero
    if (total_elements == 0) {
        throw std::runtime_error("BatchNorm2d backward: Cannot compute gradients for empty tensor (total_elements = 0)");
    }

    // Compute grad_gamma and grad_beta for each channel
    #pragma omp parallel for if(C > 1)
    for (int64_t c = 0; c < C; c++) {
        T channel_mean = mean[c];
        T channel_var = variance[c];
        T invstd = T(1.0f) / safe_sqrt(channel_var + epsilon);

        // Compute grad_gamma = sum(grad_output * normalized)
        // Compute grad_beta = sum(grad_output)
        T sum_grad_gamma = T(0.0f);
        T sum_grad_beta = T(0.0f);

        for (int64_t n = 0; n < N; n++) {
            for (int64_t h = 0; h < H; h++) {
                for (int64_t w = 0; w < W; w++) {
                    int64_t idx = ((n * C + c) * H + h) * W + w;
                    T grad_out = grad_output[idx];
                    T normalized = (input[idx] - channel_mean) * invstd;

                    sum_grad_gamma += grad_out * normalized;
                    sum_grad_beta += grad_out;
                }
            }
        }

        grad_gamma[c] = sum_grad_gamma;
        grad_beta[c] = sum_grad_beta;
    }

    // Compute grad_input
    // Efficient formulation: grad_input = gamma * invstd * (grad_output - mean(grad_output) - normalized * mean(grad_output * normalized))
    #pragma omp parallel for if(C > 1)
    for (int64_t c = 0; c < C; c++) {
        T channel_mean = mean[c];
        T channel_var = variance[c];
        T invstd = T(1.0f) / safe_sqrt(channel_var + epsilon);
        T channel_gamma = gamma[c];

        // Compute auxiliary statistics
        T sum_grad = T(0.0f);
        T sum_grad_norm = T(0.0f);

        for (int64_t n = 0; n < N; n++) {
            for (int64_t h = 0; h < H; h++) {
                for (int64_t w = 0; w < W; w++) {
                    int64_t idx = ((n * C + c) * H + h) * W + w;
                    T grad_out = grad_output[idx];
                    T normalized = (input[idx] - channel_mean) * invstd;

                    sum_grad += grad_out;
                    sum_grad_norm += grad_out * normalized;
                }
            }
        }

        T mean_grad = sum_grad / T(static_cast<float>(total_elements));
        T mean_grad_norm = sum_grad_norm / T(static_cast<float>(total_elements));

        // Compute gradient w.r.t input
        for (int64_t n = 0; n < N; n++) {
            for (int64_t h = 0; h < H; h++) {
                for (int64_t w = 0; w < W; w++) {
                    int64_t idx = ((n * C + c) * H + h) * W + w;
                    T grad_out = grad_output[idx];
                    T normalized = (input[idx] - channel_mean) * invstd;

                    // Efficient backward formulation
                    T grad_normalized = grad_out - mean_grad - normalized * mean_grad_norm;
                    grad_input[idx] = channel_gamma * invstd * grad_normalized;
                }
            }
        }
    }
}

auto batchnorm2d_backward_kernel(const Tensor& grad_output,
                                 const Tensor& input,
                                 const Tensor& mean,
                                 const Tensor& variance,
                                 const Tensor& gamma,
                                 float epsilon) -> std::vector<Tensor> {
    auto shape = input.shape();
    int64_t N = shape[0];
    int64_t C = shape[1];
    int64_t H = shape[2];
    int64_t W = shape[3];

    // Allocate output gradients
    std::vector<int64_t> shape_vec(shape.begin(), shape.end());
    Tensor grad_input(shape_vec, input.dtype(), input.device());
    Tensor grad_gamma({C}, input.dtype(), input.device());
    Tensor grad_beta({C}, input.dtype(), input.device());

    if (input.dtype() == DType::Float32) {
        batchnorm_backward_impl<float>(
            grad_output.data<float>(),
            input.data<float>(),
            grad_input.data<float>(),
            grad_gamma.data<float>(),
            grad_beta.data<float>(),
            mean.data<float>(),
            variance.data<float>(),
            gamma.data<float>(),
            epsilon,
            N, C, H, W
        );
    } else if (input.dtype() == DType::Float64) {
        batchnorm_backward_impl<double>(
            grad_output.data<double>(),
            input.data<double>(),
            grad_input.data<double>(),
            grad_gamma.data<double>(),
            grad_beta.data<double>(),
            mean.data<double>(),
            variance.data<double>(),
            gamma.data<double>(),
            static_cast<double>(epsilon),
            N, C, H, W
        );
    } else if (input.dtype() == DType::Float16) {
        batchnorm_backward_impl<Float16>(
            grad_output.data<Float16>(),
            input.data<Float16>(),
            grad_input.data<Float16>(),
            grad_gamma.data<Float16>(),
            grad_beta.data<Float16>(),
            mean.data<Float16>(),
            variance.data<Float16>(),
            gamma.data<Float16>(),
            Float16(static_cast<float>(epsilon)),
            N, C, H, W
        );
    } else {
        throw std::runtime_error("BatchNorm2d only supports Float32, Float64, and Float16 dtypes");
    }

    return {grad_input, grad_gamma, grad_beta};
}

} // namespace cpu
} // namespace tenzor
