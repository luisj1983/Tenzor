#include "tenzor/core/tensor.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/reduction.hpp"
#include "tenzor/utils/log.hpp"
#include "tenzor/utils/widen_narrow.hpp"
#include "buffer_pool.hpp"
#include "fused_conv_bn_relu.hpp"
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <algorithm>
#include <limits>
#include <iostream>

#if defined(__x86_64__) || defined(_M_X64)
    #include <immintrin.h>
    #if defined(__AVX512F__)
        #define TENZOR_ATTN_AVX512
    #endif
    #if defined(__AVX2__)
        #define TENZOR_ATTN_AVX2
    #endif
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

// Import shared Float16/BFloat16 operator overloads
#include "half_operators.hpp"

// SIMD fast math (vectorized tanh, GELU, etc.)
#include "simd_fast_math.hpp"
#include "tenzor/backend/omp_thresholds.hpp"

namespace tenzor {
namespace cpu {

// Forward declaration for conv2d_forward_kernel (used by fused conv+activation kernels)
auto conv2d_forward_kernel(const Tensor& input, const Tensor& weight, const Tensor* bias,
                           int64_t stride, int64_t padding, int64_t dilation, int64_t groups) -> Tensor;
// Per-axis overload (audit F.11) — defined in conv2d.cpp.
auto conv2d_forward_kernel(const Tensor& input, const Tensor& weight, const Tensor* bias,
                           int64_t stride_h, int64_t stride_w,
                           int64_t pad_h, int64_t pad_w,
                           int64_t dil_h, int64_t dil_w, int64_t groups) -> Tensor;

/**
 * @brief Fused linear + ReLU kernel (CPU implementation)
 *
 * Combines matrix multiplication, bias addition, and ReLU activation.
 * Formula: max(0, input @ weight.T + bias)
 */
auto fused_linear_relu_kernel(
    const Tensor& input,
    const Tensor& weight,
    const Tensor* bias
) -> Tensor {
    // Widen Float16/BFloat16 inputs to Float32 so matmul + bias-add + ReLU
    // can run on a single contiguous Float32 buffer; Float32/Float64 paths
    // are pass-through (bit-identical to the prior implementation).
    return ::tenzor::utils::widen_narrow_compute(input,
        [&](const Tensor& in_wide) -> Tensor {
            Tensor w_wide = ::tenzor::utils::is_half_precision(weight.dtype())
                                ? weight.to(DType::Float32)
                                : weight;
            // F045: the bias is read below with data<float>()/data<double>()
            // according to the COMPUTE dtype (in_wide.dtype()). Materialize the
            // bias in that exact dtype whenever it differs, so a Float64 compute
            // path never reinterprets a Float32 bias buffer as doubles (OOB read
            // of ~4*out_features bytes) — and so the half path is still widened.
            Tensor b_wide_storage;
            const Tensor* b_ptr = bias;
            if (bias != nullptr && bias->dtype() != in_wide.dtype()) {
                b_wide_storage = bias->to(in_wide.dtype());
                b_ptr = &b_wide_storage;
            }

            // Flatten input to 2D if needed
            auto input_shape = in_wide.shape();
            int64_t batch_size = 1;
            for (size_t i = 0; i < input_shape.size() - 1; ++i) {
                batch_size *= input_shape[i];
            }
            int64_t in_features = input_shape[input_shape.size() - 1];
            int64_t out_features = w_wide.shape()[0];

            // Reshape input to (batch_size, in_features)
            Tensor input_2d = in_wide.reshape({batch_size, in_features});

            // Transpose weight from (out_features, in_features) to (in_features, out_features)
            Tensor weight_t = w_wide.transpose(0, 1);

            // Matrix multiplication
            Tensor output = matmul(input_2d, weight_t);

            const size_t total_elems =
                static_cast<size_t>(batch_size) * static_cast<size_t>(out_features);

            if (output.dtype() == DType::Float32) {
                float* out_data = output.data<float>();
                const float* bias_data = b_ptr ? b_ptr->data<float>() : nullptr;

                #pragma omp parallel for if(static_cast<int64_t>(total_elems) > ::tenzor::OmpThresholds::simple())
                for (size_t i = 0; i < static_cast<size_t>(batch_size); ++i) {
                    for (size_t j = 0; j < static_cast<size_t>(out_features); ++j) {
                        size_t idx = i * out_features + j;
                        float val = out_data[idx];
                        if (bias_data) {
                            val += bias_data[j];
                        }
                        out_data[idx] = std::max(0.0f, val);
                    }
                }
            } else if (output.dtype() == DType::Float64) {
                double* out_data = output.data<double>();
                const double* bias_data = b_ptr ? b_ptr->data<double>() : nullptr;

                #pragma omp parallel for if(static_cast<int64_t>(total_elems) > ::tenzor::OmpThresholds::simple())
                for (size_t i = 0; i < static_cast<size_t>(batch_size); ++i) {
                    for (size_t j = 0; j < static_cast<size_t>(out_features); ++j) {
                        size_t idx = i * out_features + j;
                        double val = out_data[idx];
                        if (bias_data) {
                            val += bias_data[j];
                        }
                        out_data[idx] = std::max(0.0, val);
                    }
                }
            }

            // Reshape back to original batch dimensions
            std::vector<int64_t> output_shape(input_shape.begin(), input_shape.end() - 1);
            output_shape.push_back(out_features);
            return output.reshape(output_shape);
        });
}

/**
 * @brief Fused conv2d + ReLU kernel (CPU implementation)
 *
 * Delegates to conv2d_forward_kernel (im2col+GEMM) then applies ReLU in-place.
 * Supports dilation and groups for parity with CUDA backend.
 */
// Per-axis overload — preferred entry point for anisotropic stride/pad/dilation
// (audit F.11).  Delegates to the per-axis conv2d_forward_kernel overload.
auto fused_conv2d_relu_kernel(
    const Tensor& input,
    const Tensor& weight,
    const Tensor* bias,
    int64_t stride_h,
    int64_t stride_w,
    int64_t pad_h,
    int64_t pad_w,
    int64_t dil_h,
    int64_t dil_w,
    int64_t groups
) -> Tensor {
    // Widen Float16/BFloat16 inputs to Float32 so the activation step
    // does not silently no-op. Float32/Float64 paths are pass-through and
    // remain bit-identical (see tenzor::utils::widen_narrow_compute).
    return ::tenzor::utils::widen_narrow_compute(input,
        [&](const Tensor& in_wide) -> Tensor {
            Tensor w_wide = ::tenzor::utils::is_half_precision(weight.dtype())
                                ? weight.to(DType::Float32)
                                : weight;
            Tensor b_wide_storage;
            const Tensor* b_ptr = bias;
            if (bias != nullptr && ::tenzor::utils::is_half_precision(bias->dtype())) {
                b_wide_storage = bias->to(DType::Float32);
                b_ptr = &b_wide_storage;
            }
            Tensor result = conv2d_forward_kernel(in_wide, w_wide, b_ptr,
                                                  stride_h, stride_w, pad_h, pad_w, dil_h, dil_w,
                                                  groups);
            int64_t n = result.numel();
            if (result.dtype() == DType::Float32) {
                float* data = result.data<float>();
                #pragma omp parallel for if(static_cast<int64_t>(n) > ::tenzor::OmpThresholds::simple())
                for (int64_t i = 0; i < n; ++i) {
                    data[i] = std::max(0.0f, data[i]);
                }
            } else if (result.dtype() == DType::Float64) {
                double* data = result.data<double>();
                #pragma omp parallel for if(static_cast<int64_t>(n) > ::tenzor::OmpThresholds::simple())
                for (int64_t i = 0; i < n; ++i) {
                    data[i] = std::max(0.0, data[i]);
                }
            }
            return result;
        });
}

auto fused_conv2d_relu_kernel(
    const Tensor& input,
    const Tensor& weight,
    const Tensor* bias,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    int64_t groups
) -> Tensor {
    // Widen Float16/BFloat16 inputs to Float32 so the activation step
    // does not silently no-op. Float32/Float64 paths are pass-through.
    return ::tenzor::utils::widen_narrow_compute(input,
        [&](const Tensor& in_wide) -> Tensor {
            Tensor w_wide = ::tenzor::utils::is_half_precision(weight.dtype())
                                ? weight.to(DType::Float32)
                                : weight;
            Tensor b_wide_storage;
            const Tensor* b_ptr = bias;
            if (bias != nullptr && ::tenzor::utils::is_half_precision(bias->dtype())) {
                b_wide_storage = bias->to(DType::Float32);
                b_ptr = &b_wide_storage;
            }
            Tensor result = conv2d_forward_kernel(in_wide, w_wide, b_ptr,
                                                  stride, padding, dilation, groups);
            int64_t n = result.numel();
            if (result.dtype() == DType::Float32) {
                float* data = result.data<float>();
                #pragma omp parallel for if(static_cast<int64_t>(n) > ::tenzor::OmpThresholds::simple())
                for (int64_t i = 0; i < n; ++i) {
                    data[i] = std::max(0.0f, data[i]);
                }
            } else if (result.dtype() == DType::Float64) {
                double* data = result.data<double>();
                #pragma omp parallel for if(static_cast<int64_t>(n) > ::tenzor::OmpThresholds::simple())
                for (int64_t i = 0; i < n; ++i) {
                    data[i] = std::max(0.0, data[i]);
                }
            }
            return result;
        });
}

/**
 * @brief Fused batchnorm + ReLU kernel (CPU implementation)
 */
auto fused_batchnorm_relu_kernel(
    const Tensor& input,
    const Tensor& running_mean,
    const Tensor& running_var,
    const Tensor& weight,
    const Tensor& bias,
    float eps
) -> Tensor {
    // Contiguity guard: the kernel below indexes `input.data<T>()` with
    // shape-derived offsets (n*C*S + c*S + s), which assumes a contiguous
    // [N,C,...] layout. A non-contiguous input silently produces wrong output
    // (see fused_attention_kernel / fused_layer_norm_backward_kernel).
    if (!input.is_contiguous()) {
        return fused_batchnorm_relu_kernel(input.contiguous(), running_mean,
                                           running_var, weight, bias, eps);
    }

    // F047: running_mean/running_var/weight/bias are indexed linearly as
    // data<T>()[c]; a strided 1-D param view would read the wrong per-channel
    // element. Contiguify them like the input above.
    const Tensor running_mean_c =
        running_mean.is_contiguous() ? running_mean : running_mean.contiguous();
    const Tensor running_var_c =
        running_var.is_contiguous() ? running_var : running_var.contiguous();
    const Tensor weight_c = weight.is_contiguous() ? weight : weight.contiguous();
    const Tensor bias_c = bias.is_contiguous() ? bias : bias.contiguous();

    int64_t num_features = input.shape()[1];
    std::vector<int64_t> shape_vec(input.shape().begin(), input.shape().end());
    Tensor output = zeros(shape_vec, input.dtype(), input.device());

    int64_t batch_size = input.shape()[0];
    int64_t spatial_size = 1;
    for (size_t i = 2; i < input.shape().size(); ++i) {
        spatial_size *= input.shape()[i];
    }

    if (input.dtype() == DType::Float32) {
        const float* in_data = input.data<float>();
        const float* mean_data = running_mean_c.data<float>();
        const float* var_data = running_var_c.data<float>();
        const float* gamma_data = weight_c.data<float>();
        const float* beta_data = bias_c.data<float>();
        float* out_data = output.data<float>();

        // Fused batchnorm + ReLU
        #pragma omp parallel for collapse(2) if(batch_size * num_features > 64)
        for (int64_t n = 0; n < batch_size; ++n) {
            for (int64_t c = 0; c < num_features; ++c) {
                float mean = mean_data[c];
                float var = var_data[c];
                float gamma = gamma_data[c];
                float beta = beta_data[c];
                float inv_std = 1.0f / std::sqrt(var + eps);

                for (int64_t s = 0; s < spatial_size; ++s) {
                    size_t idx = n * num_features * spatial_size + c * spatial_size + s;
                    float normalized = (in_data[idx] - mean) * inv_std;
                    float scaled = normalized * gamma + beta;
                    out_data[idx] = std::max(0.0f, scaled);
                }
            }
        }
    } else if (input.dtype() == DType::Float64) {
        const double* in_data = input.data<double>();
        const double* mean_data = running_mean_c.data<double>();
        const double* var_data = running_var_c.data<double>();
        const double* gamma_data = weight_c.data<double>();
        const double* beta_data = bias_c.data<double>();
        double* out_data = output.data<double>();

        // Fused batchnorm + ReLU
        #pragma omp parallel for collapse(2) if(batch_size * num_features > 64)
        for (int64_t n = 0; n < batch_size; ++n) {
            for (int64_t c = 0; c < num_features; ++c) {
                double mean = mean_data[c];
                double var = var_data[c];
                double gamma = gamma_data[c];
                double beta = beta_data[c];
                double inv_std = 1.0 / std::sqrt(var + static_cast<double>(eps));

                for (int64_t s = 0; s < spatial_size; ++s) {
                    size_t idx = n * num_features * spatial_size + c * spatial_size + s;
                    double normalized = (in_data[idx] - mean) * inv_std;
                    double scaled = normalized * gamma + beta;
                    out_data[idx] = std::max(0.0, scaled);
                }
            }
        }
    } else if (input.dtype() == DType::Float16) {
        const Float16* in_data = input.data<Float16>();
        const Float16* mean_data = running_mean_c.data<Float16>();
        const Float16* var_data = running_var_c.data<Float16>();
        const Float16* gamma_data = weight_c.data<Float16>();
        const Float16* beta_data = bias_c.data<Float16>();
        Float16* out_data = output.data<Float16>();

        // Fused batchnorm + ReLU (compute in float for numerical stability)
        #pragma omp parallel for collapse(2) if(batch_size * num_features > 64)
        for (int64_t n = 0; n < batch_size; ++n) {
            for (int64_t c = 0; c < num_features; ++c) {
                float mean = static_cast<float>(mean_data[c]);
                float var = static_cast<float>(var_data[c]);
                float gamma = static_cast<float>(gamma_data[c]);
                float beta = static_cast<float>(beta_data[c]);
                float inv_std = 1.0f / std::sqrt(var + eps);

                for (int64_t s = 0; s < spatial_size; ++s) {
                    size_t idx = n * num_features * spatial_size + c * spatial_size + s;
                    float in_val = static_cast<float>(in_data[idx]);
                    float normalized = (in_val - mean) * inv_std;
                    float scaled = normalized * gamma + beta;
                    out_data[idx] = Float16(std::max(0.0f, scaled));
                }
            }
        }
    } else if (input.dtype() == DType::BFloat16) {
        const BFloat16* in_data = input.data<BFloat16>();
        const BFloat16* mean_data = running_mean_c.data<BFloat16>();
        const BFloat16* var_data = running_var_c.data<BFloat16>();
        const BFloat16* gamma_data = weight_c.data<BFloat16>();
        const BFloat16* beta_data = bias_c.data<BFloat16>();
        BFloat16* out_data = output.data<BFloat16>();

        #pragma omp parallel for collapse(2) if(batch_size * num_features > 64)
        for (int64_t n = 0; n < batch_size; ++n) {
            for (int64_t c = 0; c < num_features; ++c) {
                float mean = static_cast<float>(mean_data[c]);
                float var = static_cast<float>(var_data[c]);
                float gamma = static_cast<float>(gamma_data[c]);
                float beta = static_cast<float>(beta_data[c]);
                float inv_std = 1.0f / std::sqrt(var + eps);

                for (int64_t s = 0; s < spatial_size; ++s) {
                    size_t idx = n * num_features * spatial_size + c * spatial_size + s;
                    float in_val = static_cast<float>(in_data[idx]);
                    float normalized = (in_val - mean) * inv_std;
                    float scaled = normalized * gamma + beta;
                    out_data[idx] = BFloat16(std::max(0.0f, scaled));
                }
            }
        }
    } else {
        throw std::runtime_error("fused_batchnorm_relu: Unsupported dtype");
    }

    return output;
}

/**
 * @brief Fused softmax + cross entropy kernel (CPU implementation)
 *
 * Uses log-sum-exp trick for numerical stability.
 * Returns {loss} or {loss, grad_logits} depending on compute_grad.
 */
auto fused_softmax_cross_entropy_kernel(
    const Tensor& logits,
    const Tensor& targets,
    bool compute_grad,
    const std::string& reduction
) -> std::vector<Tensor> {
    // S13 rank-generalisation + dtype-preservation.
    //
    // Rank > 2: callers in seq2seq / classification-over-time hand us
    // logits with shape (D1, ..., Dk, C) and targets with shape
    // (D1, ..., Dk). Flatten the leading dims into a (B = prod(Di), C)
    // problem, do the existing 2-D compute, then reshape grad_logits
    // back to the original shape. Loss is reduced to scalar/per-sample
    // by the existing reduction logic; for "none" reduction it returns
    // a 1-D (B,) vector that we reshape back to (D1, ..., Dk).
    //
    // Dtype:
    //   Float32/Float64: native, output dtype = input dtype (unchanged).
    //   Float16/BFloat16: widen to Float32 internally for numeric safety
    //     (matmul/exp/log range), but narrow grad_logits back to the
    //     original half-precision dtype on return. Loss stays Float32 —
    //     it's a scalar (or small per-sample vector) where extra
    //     precision is desirable and downstream consumers handle the
    //     dtype mismatch (gradient flow doesn't transit through loss).
    const auto& logits_shape_v = logits.shape();
    const int64_t ndim_logits = static_cast<int64_t>(logits_shape_v.size());
    if (ndim_logits < 2) {
        throw std::runtime_error(
            "fused_softmax_cross_entropy: logits must have rank >= 2 "
            "(found rank " + std::to_string(ndim_logits) + ")");
    }

    // Rank > 2: flatten leading dims into batch.
    if (ndim_logits > 2) {
        int64_t leading = 1;
        for (int64_t i = 0; i < ndim_logits - 1; ++i) {
            leading *= logits_shape_v[i];
        }
        int64_t C = logits_shape_v[ndim_logits - 1];
        Tensor logits_2d = logits.reshape({leading, C});
        Tensor targets_1d = targets.reshape({leading});
        auto inner = fused_softmax_cross_entropy_kernel(
            logits_2d, targets_1d, compute_grad, reduction);

        // Reshape outputs back to leading-dim layout.
        std::vector<int64_t> leading_shape(
            logits_shape_v.begin(), logits_shape_v.end() - 1);
        std::vector<Tensor> out;
        out.reserve(inner.size());
        // loss: scalar for mean/sum (passes through); per-sample (B,) for
        // "none" — reshape back to leading dims.
        if (reduction == "none") {
            out.push_back(inner[0].reshape(leading_shape));
        } else {
            out.push_back(inner[0]);
        }
        if (compute_grad && inner.size() > 1) {
            // grad_logits has shape (B, C); reshape back to (D1, ..., Dk, C).
            std::vector<int64_t> grad_shape = leading_shape;
            grad_shape.push_back(C);
            out.push_back(inner[1].reshape(grad_shape));
        }
        return out;
    }

    int64_t batch_size = logits.shape()[0];
    int64_t num_classes = logits.shape()[1];

    // For half-precision inputs we widen to Float32, run the Float32 path,
    // and narrow grad_logits back to the original dtype. The loss stays
    // Float32 by intent (see comment above).
    const DType orig_dtype = logits.dtype();
    if (orig_dtype == DType::Float16 || orig_dtype == DType::BFloat16) {
        Tensor logits_f32({batch_size, num_classes}, DType::Float32, logits.device());
        float* lf32 = logits_f32.data<float>();
        if (orig_dtype == DType::Float16) {
            const Float16* src = logits.data<Float16>();
            for (int64_t i = 0; i < batch_size * num_classes; ++i)
                lf32[i] = static_cast<float>(src[i]);
        } else {
            const BFloat16* src = logits.data<BFloat16>();
            for (int64_t i = 0; i < batch_size * num_classes; ++i)
                lf32[i] = static_cast<float>(src[i]);
        }
        auto result = fused_softmax_cross_entropy_kernel(
            logits_f32, targets, compute_grad, reduction);
        // Narrow grad_logits back to the original half-precision dtype.
        if (compute_grad && result.size() > 1) {
            result[1] = result[1].to(orig_dtype);
        }
        return result;
    }

    Tensor losses = zeros({batch_size}, logits.dtype(), logits.device());
    Tensor grad_logits;
    if (compute_grad) {
        std::vector<int64_t> logits_shape(logits.shape().begin(), logits.shape().end());
        grad_logits = zeros(logits_shape, logits.dtype(), logits.device());
    }

    // Validate target ranges in a cheap serial pre-pass BEFORE the compute
    // loops. Doing this here (rather than via a NaN-loss sentinel checked after
    // the loop) keeps out-of-range targets distinct from NaN losses that arise
    // from legitimately NaN/Inf logits, so the two failure modes are not
    // conflated in the error message.
    {
        const int64_t* targets_data = targets.data<int64_t>();
        for (int64_t i = 0; i < batch_size; ++i) {
            const int64_t target = targets_data[i];
            if (target < 0 || target >= num_classes) {
                throw std::runtime_error(
                    "fused_softmax_cross_entropy: target index " + std::to_string(target) +
                    " at row " + std::to_string(i) + " out of range [0, " +
                    std::to_string(num_classes) + ")");
            }
        }
    }

    if (logits.dtype() == DType::Float32) {
        const float* logits_data = logits.data<float>();
        const int64_t* targets_data = targets.data<int64_t>();
        float* losses_data = losses.data<float>();
        float* grad_data = compute_grad ? grad_logits.data<float>() : nullptr;

        #pragma omp parallel for if(batch_size > 64)
        for (int64_t i = 0; i < batch_size; ++i) {
            const float* row = logits_data + i * num_classes;

            // Find max for numerical stability
            float max_logit = row[0];
            for (int64_t j = 1; j < num_classes; ++j) {
                max_logit = std::max(max_logit, row[j]);
            }

            // Compute log_sum_exp and softmax probabilities
            float sum_exp = 0.0f;
            for (int64_t j = 0; j < num_classes; ++j) {
                sum_exp += std::exp(row[j] - max_logit);
            }
            float log_sum_exp = std::log(sum_exp) + max_logit;

            // Compute loss for target class (target range validated above).
            int64_t target = targets_data[i];
            losses_data[i] = log_sum_exp - row[target];

            // Compute gradient: softmax(logits) - one_hot(target)
            if (compute_grad) {
                float* grad_row = grad_data + i * num_classes;
                float inv_sum_exp = 1.0f / sum_exp;
                for (int64_t j = 0; j < num_classes; ++j) {
                    grad_row[j] = std::exp(row[j] - max_logit) * inv_sum_exp;
                }
                grad_row[target] -= 1.0f;
                // Scale gradient based on reduction mode
                if (reduction == "mean") {
                    float scale = 1.0f / static_cast<float>(batch_size);
                    for (int64_t j = 0; j < num_classes; ++j) {
                        grad_row[j] *= scale;
                    }
                }
                // "sum": scale=1.0 (no scaling needed)
                // "none": per-sample gradients (no scaling needed)
            }
        }
    } else if (logits.dtype() == DType::Float64) {
        const double* logits_data = logits.data<double>();
        const int64_t* targets_data = targets.data<int64_t>();
        double* losses_data = losses.data<double>();
        double* grad_data = compute_grad ? grad_logits.data<double>() : nullptr;

        #pragma omp parallel for if(batch_size > 64)
        for (int64_t i = 0; i < batch_size; ++i) {
            const double* row = logits_data + i * num_classes;

            double max_logit = row[0];
            for (int64_t j = 1; j < num_classes; ++j) {
                max_logit = std::max(max_logit, row[j]);
            }

            double sum_exp = 0.0;
            for (int64_t j = 0; j < num_classes; ++j) {
                sum_exp += std::exp(row[j] - max_logit);
            }
            double log_sum_exp = std::log(sum_exp) + max_logit;

            int64_t target = targets_data[i];  // range validated above
            losses_data[i] = log_sum_exp - row[target];

            if (compute_grad) {
                double* grad_row = grad_data + i * num_classes;
                double inv_sum_exp = 1.0 / sum_exp;
                for (int64_t j = 0; j < num_classes; ++j) {
                    grad_row[j] = std::exp(row[j] - max_logit) * inv_sum_exp;
                }
                grad_row[target] -= 1.0;
                if (reduction == "mean") {
                    double scale = 1.0 / static_cast<double>(batch_size);
                    for (int64_t j = 0; j < num_classes; ++j) {
                        grad_row[j] *= scale;
                    }
                }
            }
        }
    } else {
        throw std::runtime_error("fused_softmax_cross_entropy: unsupported dtype");
    }

    // Apply reduction to per-sample losses
    Tensor loss;
    if (reduction == "mean") {
        loss = tenzor::mean(losses);
    } else if (reduction == "sum") {
        loss = tenzor::sum(losses);
    } else {
        // "none" — return per-sample losses
        loss = losses;
    }

    if (compute_grad) {
        return {loss, grad_logits};
    }
    return {loss};
}

/**
 * @brief Fused add + ReLU kernel (CPU implementation)
 */
auto fused_add_relu_kernel(const Tensor& a, const Tensor& b) -> Tensor {
    Tensor result = add(a, b);

    if (result.dtype() == DType::Float32) {
        float* data = result.data<float>();
        size_t n = static_cast<size_t>(result.numel());
#if defined(TENZOR_ATTN_AVX2)
        size_t vec_end = n - (n % 8);
        __m256 zero_vec = _mm256_setzero_ps();
        #pragma omp parallel for if(static_cast<int64_t>(n) > ::tenzor::OmpThresholds::simple())
        for (size_t i = 0; i < vec_end; i += 8) {
            __m256 v = _mm256_loadu_ps(data + i);
            _mm256_storeu_ps(data + i, _mm256_max_ps(zero_vec, v));
        }
        for (size_t i = vec_end; i < n; ++i) {
            data[i] = std::max(0.0f, data[i]);
        }
#else
        #pragma omp parallel for if(static_cast<int64_t>(n) > ::tenzor::OmpThresholds::simple())
        for (size_t i = 0; i < n; ++i) {
            data[i] = std::max(0.0f, data[i]);
        }
#endif
    } else if (result.dtype() == DType::Float64) {
        double* data = result.data<double>();
        size_t n = static_cast<size_t>(result.numel());
        #pragma omp parallel for if(static_cast<int64_t>(n) > ::tenzor::OmpThresholds::simple())
        for (size_t i = 0; i < n; ++i) {
            data[i] = std::max(0.0, data[i]);
        }
    } else if (result.dtype() == DType::Float16) {
        Tensor result_f32 = result.to(DType::Float32);
        float* data = result_f32.data<float>();
        size_t n = static_cast<size_t>(result.numel());
        #pragma omp parallel for if(static_cast<int64_t>(n) > ::tenzor::OmpThresholds::simple())
        for (size_t i = 0; i < n; ++i) {
            data[i] = std::max(0.0f, data[i]);
        }
        result = result_f32.to(DType::Float16);
    } else if (result.dtype() == DType::BFloat16) {
        Tensor result_f32 = result.to(DType::Float32);
        float* data = result_f32.data<float>();
        size_t n = static_cast<size_t>(result.numel());
        #pragma omp parallel for if(static_cast<int64_t>(n) > ::tenzor::OmpThresholds::simple())
        for (size_t i = 0; i < n; ++i) {
            data[i] = std::max(0.0f, data[i]);
        }
        result = result_f32.to(DType::BFloat16);
    } else {
        throw std::runtime_error("fused_add_relu: unsupported dtype");
    }

    return result;
}

/**
 * @brief Fused GELU kernel (CPU implementation)
 *
 * GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
 */
auto fused_gelu_kernel(const Tensor& input) -> Tensor {
    // Contiguity guard: the kernel iterates `in_data[i]` linearly over numel,
    // which assumes contiguous storage; a strided view would read the wrong
    // elements (see fused_attention_kernel / fused_layer_norm_backward_kernel).
    if (!input.is_contiguous()) {
        return fused_gelu_kernel(input.contiguous());
    }

    std::vector<int64_t> shape_vec(input.shape().begin(), input.shape().end());
    Tensor result = zeros(shape_vec, input.dtype(), input.device());

    // Exact GELU: 0.5 * x * (1 + erf(x / sqrt(2))) — matches PyTorch default and
    // the canonical gelu_kernel, so the fused inference path agrees with the
    // (autograd) training path.
    constexpr float  INV_SQRT2_F = 0.70710678f;
    constexpr double INV_SQRT2_D = 0.7071067811865475244;

    if (input.dtype() == DType::Float32) {
        const float* in_data = input.data<float>();
        float* out_data = result.data<float>();
        size_t n = static_cast<size_t>(input.numel());

        #pragma omp parallel for if(static_cast<int64_t>(n) > ::tenzor::OmpThresholds::simple())
        for (size_t i = 0; i < n; ++i) {
            float x = in_data[i];
            out_data[i] = 0.5f * x * (1.0f + std::erf(x * INV_SQRT2_F));
        }
    } else if (input.dtype() == DType::Float64) {
        const double* in_data = input.data<double>();
        double* out_data = result.data<double>();
        size_t n = static_cast<size_t>(input.numel());

        #pragma omp parallel for if(static_cast<int64_t>(n) > ::tenzor::OmpThresholds::simple())
        for (size_t i = 0; i < n; ++i) {
            double x = in_data[i];
            out_data[i] = 0.5 * x * (1.0 + std::erf(x * INV_SQRT2_D));
        }
    } else if (input.dtype() == DType::Float16) {
        const Float16* in_data = input.data<Float16>();
        Float16* out_data = result.data<Float16>();
        size_t n = static_cast<size_t>(input.numel());

        #pragma omp parallel for if(static_cast<int64_t>(n) > ::tenzor::OmpThresholds::simple())
        for (size_t i = 0; i < n; ++i) {
            float x = static_cast<float>(in_data[i]);
            out_data[i] = Float16(0.5f * x * (1.0f + std::erf(x * INV_SQRT2_F)));
        }
    } else if (input.dtype() == DType::BFloat16) {
        const BFloat16* in_data = input.data<BFloat16>();
        BFloat16* out_data = result.data<BFloat16>();
        size_t n = static_cast<size_t>(input.numel());

        #pragma omp parallel for if(static_cast<int64_t>(n) > ::tenzor::OmpThresholds::simple())
        for (size_t i = 0; i < n; ++i) {
            float x = static_cast<float>(in_data[i]);
            out_data[i] = BFloat16(0.5f * x * (1.0f + std::erf(x * INV_SQRT2_F)));
        }
    } else {
        throw std::runtime_error("fused_gelu: Only Float32/Float64/Float16/BFloat16 supported");
    }

    return result;
}

/**
 * @brief Fused layer norm kernel (CPU implementation)
 *
 * Single-pass computation of mean, variance, and normalization.
 * Returns {output, mean, inv_std} to match CUDA backend for backward pass support.
 */
// Double-precision reductions for LayerNorm stats (widen f32 lanes to f64).
// Mirrors the hardened accumulation in nn_kernels.cpp's layer_norm path so the
// fused kernel does not catastrophically cancel for large-magnitude rows.
static double fused_ln_sum_f64(const float* p, int64_t n) {
    double total = 0.0;
    int64_t i = 0;
#ifdef TENZOR_ATTN_AVX2
    __m256d acc = _mm256_setzero_pd();
    for (; i + 4 <= n; i += 4) {
        acc = _mm256_add_pd(acc, _mm256_cvtps_pd(_mm_loadu_ps(p + i)));
    }
    __m128d hi = _mm256_extractf128_pd(acc, 1);
    __m128d lo = _mm256_castpd256_pd128(acc);
    __m128d s = _mm_add_pd(lo, hi);
    s = _mm_hadd_pd(s, s);
    total = _mm_cvtsd_f64(s);
#endif
    for (; i < n; ++i) total += static_cast<double>(p[i]);
    return total;
}

static double fused_ln_sumsq_f64(const float* p, int64_t n, double mean) {
    double total = 0.0;
    int64_t i = 0;
#ifdef TENZOR_ATTN_AVX2
    const __m256d vmean = _mm256_set1_pd(mean);
    __m256d acc = _mm256_setzero_pd();
    for (; i + 4 <= n; i += 4) {
        __m256d d = _mm256_sub_pd(_mm256_cvtps_pd(_mm_loadu_ps(p + i)), vmean);
        acc = _mm256_fmadd_pd(d, d, acc);
    }
    __m128d hi = _mm256_extractf128_pd(acc, 1);
    __m128d lo = _mm256_castpd256_pd128(acc);
    __m128d s = _mm_add_pd(lo, hi);
    s = _mm_hadd_pd(s, s);
    total = _mm_cvtsd_f64(s);
#endif
    for (; i < n; ++i) {
        const double d = static_cast<double>(p[i]) - mean;
        total += d * d;
    }
    return total;
}

auto fused_layer_norm_kernel(
    const Tensor& input,
    const std::vector<int64_t>& normalized_shape,
    const Tensor& weight,
    const Tensor& bias,
    float eps
) -> std::tuple<Tensor, Tensor, Tensor> {
    // Contiguity guard: the kernel indexes `input.data<T>()` with row-major
    // batch*norm_size offsets, which assumes contiguous storage; a strided
    // view silently produces wrong output (see fused_layer_norm_backward_kernel).
    if (!input.is_contiguous()) {
        return fused_layer_norm_kernel(input.contiguous(), normalized_shape,
                                       weight, bias, eps);
    }

    // Calculate batch size and normalization size
    int64_t norm_size = 1;
    for (auto dim : normalized_shape) {
        norm_size *= dim;
    }

    // F069: norm_size == 0 (normalized_shape containing 0) would make the
    // division below an integer divide-by-zero (SIGFPE). Reject before any
    // work / OpenMP region.
    if (norm_size <= 0) {
        throw std::invalid_argument(
            "fused_layer_norm: normalized_shape must have positive size "
            "(norm_size > 0)");
    }

    // F046: weight/bias are indexed linearly (weight_data[i]/bias_data[i],
    // including AVX2 _mm256_loadu_ps); a strided 1-D param view would read the
    // wrong scale/shift elements. Contiguify them like the input above.
    const Tensor weight_c = weight.is_contiguous() ? weight : weight.contiguous();
    const Tensor bias_c = bias.is_contiguous() ? bias : bias.contiguous();

    int64_t batch_size = input.numel() / norm_size;

    std::vector<int64_t> shape_vec(input.shape().begin(), input.shape().end());
    Tensor result = zeros(shape_vec, input.dtype(), input.device());

    // Per-row mean and inv_std for the backward pass.  These must NOT be stored
    // in a half dtype: the backward consumes them and half-precision stats
    // degrade the gradient.  Use full Float64 for Float64 input (the fused
    // backward reads them directly) and Float32 otherwise.  Stats are always
    // accumulated in double to avoid catastrophic cancellation.
    const DType stats_dtype =
        (input.dtype() == DType::Float64) ? DType::Float64 : DType::Float32;
    Tensor mean_out({batch_size}, stats_dtype, input.device());
    Tensor inv_std_out({batch_size}, stats_dtype, input.device());

    if (input.dtype() == DType::Float32) {
        const float* in_data = input.data<float>();
        const float* weight_data = weight_c.data<float>();
        const float* bias_data = bias_c.data<float>();
        float* out_data = result.data<float>();
        float* mean_data = mean_out.data<float>();
        float* inv_std_data = inv_std_out.data<float>();

        #pragma omp parallel for if(batch_size > 64)
        for (int64_t b = 0; b < batch_size; ++b) {
            const float* batch_in = in_data + b * norm_size;
            float* batch_out = out_data + b * norm_size;

            const double mean_d =
                fused_ln_sum_f64(batch_in, norm_size) / static_cast<double>(norm_size);
            const double var_d =
                fused_ln_sumsq_f64(batch_in, norm_size, mean_d) / static_cast<double>(norm_size);
            const double inv_std_d = 1.0 / std::sqrt(var_d + static_cast<double>(eps));

            const float mean = static_cast<float>(mean_d);
            const float inv_std = static_cast<float>(inv_std_d);
            mean_data[b] = mean;
            inv_std_data[b] = inv_std;

            // Normalize with AVX2
#ifdef TENZOR_ATTN_AVX2
            {
                __m256 vmean = _mm256_set1_ps(mean);
                __m256 vinv = _mm256_set1_ps(inv_std);
                int64_t i = 0;
                for (; i + 8 <= norm_size; i += 8) {
                    __m256 v = _mm256_loadu_ps(batch_in + i);
                    __m256 w = _mm256_loadu_ps(weight_data + i);
                    __m256 bi = _mm256_loadu_ps(bias_data + i);
                    __m256 normed = _mm256_mul_ps(_mm256_sub_ps(v, vmean), vinv);
                    __m256 out = _mm256_fmadd_ps(normed, w, bi);
                    _mm256_storeu_ps(batch_out + i, out);
                }
                for (; i < norm_size; ++i) {
                    float normalized = (batch_in[i] - mean) * inv_std;
                    batch_out[i] = normalized * weight_data[i] + bias_data[i];
                }
            }
#else
            for (int64_t i = 0; i < norm_size; ++i) {
                float normalized = (batch_in[i] - mean) * inv_std;
                batch_out[i] = normalized * weight_data[i] + bias_data[i];
            }
#endif
        }
    } else if (input.dtype() == DType::Float64) {
        const double* in_data = input.data<double>();
        const double* weight_data = weight_c.data<double>();
        const double* bias_data = bias_c.data<double>();
        double* out_data = result.data<double>();
        double* mean_data = mean_out.data<double>();
        double* inv_std_data = inv_std_out.data<double>();

        #pragma omp parallel for if(batch_size > 64)
        for (int64_t b = 0; b < batch_size; ++b) {
            const double* batch_in = in_data + b * norm_size;
            double* batch_out = out_data + b * norm_size;

            double mean = 0.0;
            for (int64_t i = 0; i < norm_size; ++i) {
                mean += batch_in[i];
            }
            mean /= static_cast<double>(norm_size);

            double variance = 0.0;
            for (int64_t i = 0; i < norm_size; ++i) {
                double diff = batch_in[i] - mean;
                variance += diff * diff;
            }
            variance /= static_cast<double>(norm_size);

            double inv_std = 1.0 / std::sqrt(variance + static_cast<double>(eps));
            mean_data[b] = mean;
            inv_std_data[b] = inv_std;

            for (int64_t i = 0; i < norm_size; ++i) {
                double normalized = (batch_in[i] - mean) * inv_std;
                batch_out[i] = normalized * weight_data[i] + bias_data[i];
            }
        }
    } else if (input.dtype() == DType::Float16) {
        const Float16* in_data = input.data<Float16>();
        const Float16* weight_data = weight_c.data<Float16>();
        const Float16* bias_data = bias_c.data<Float16>();
        Float16* out_data = result.data<Float16>();
        float* mean_data = mean_out.data<float>();
        float* inv_std_data = inv_std_out.data<float>();

        #pragma omp parallel for if(batch_size > 64)
        for (int64_t b = 0; b < batch_size; ++b) {
            const Float16* batch_in = in_data + b * norm_size;
            Float16* batch_out = out_data + b * norm_size;

            double mean = 0.0;
            for (int64_t i = 0; i < norm_size; ++i) {
                mean += static_cast<double>(static_cast<float>(batch_in[i]));
            }
            mean /= static_cast<double>(norm_size);

            double variance = 0.0;
            for (int64_t i = 0; i < norm_size; ++i) {
                double diff = static_cast<double>(static_cast<float>(batch_in[i])) - mean;
                variance += diff * diff;
            }
            variance /= static_cast<double>(norm_size);

            double inv_std = 1.0 / std::sqrt(variance + static_cast<double>(eps));
            mean_data[b] = static_cast<float>(mean);
            inv_std_data[b] = static_cast<float>(inv_std);

            for (int64_t i = 0; i < norm_size; ++i) {
                double normalized = (static_cast<double>(static_cast<float>(batch_in[i])) - mean) * inv_std;
                double scaled = normalized * static_cast<double>(static_cast<float>(weight_data[i]))
                              + static_cast<double>(static_cast<float>(bias_data[i]));
                batch_out[i] = Float16(static_cast<float>(scaled));
            }
        }
    } else if (input.dtype() == DType::BFloat16) {
        const BFloat16* in_data = input.data<BFloat16>();
        const BFloat16* weight_data = weight_c.data<BFloat16>();
        const BFloat16* bias_data = bias_c.data<BFloat16>();
        BFloat16* out_data = result.data<BFloat16>();
        float* mean_data = mean_out.data<float>();
        float* inv_std_data = inv_std_out.data<float>();

        #pragma omp parallel for if(batch_size > 64)
        for (int64_t b = 0; b < batch_size; ++b) {
            const BFloat16* batch_in = in_data + b * norm_size;
            BFloat16* batch_out = out_data + b * norm_size;

            double mean = 0.0;
            for (int64_t i = 0; i < norm_size; ++i) {
                mean += static_cast<double>(static_cast<float>(batch_in[i]));
            }
            mean /= static_cast<double>(norm_size);

            double variance = 0.0;
            for (int64_t i = 0; i < norm_size; ++i) {
                double diff = static_cast<double>(static_cast<float>(batch_in[i])) - mean;
                variance += diff * diff;
            }
            variance /= static_cast<double>(norm_size);

            double inv_std = 1.0 / std::sqrt(variance + static_cast<double>(eps));
            mean_data[b] = static_cast<float>(mean);
            inv_std_data[b] = static_cast<float>(inv_std);

            for (int64_t i = 0; i < norm_size; ++i) {
                double normalized = (static_cast<double>(static_cast<float>(batch_in[i])) - mean) * inv_std;
                double scaled = normalized * static_cast<double>(static_cast<float>(weight_data[i]))
                              + static_cast<double>(static_cast<float>(bias_data[i]));
                batch_out[i] = BFloat16(static_cast<float>(scaled));
            }
        }
    } else {
        throw std::runtime_error("fused_layer_norm: Only Float32/Float64/Float16/BFloat16 supported");
    }

    return {result, mean_out, inv_std_out};
}

// =========================================================================
// RMSNorm Operations
// =========================================================================

auto fused_rms_norm_kernel(const Tensor& input, const Tensor& weight, float eps)
    -> std::tuple<Tensor, Tensor> {
    // Contiguity guard: the kernel indexes `input.data<T>()` with row-major
    // batch*norm_size offsets, which assumes contiguous storage; a strided
    // view silently produces wrong output (see fused_layer_norm_backward_kernel).
    if (!input.is_contiguous()) {
        return fused_rms_norm_kernel(input.contiguous(), weight, eps);
    }

    // RMSNorm: output = input * weight / sqrt(mean(input^2) + eps)
    const auto& shape = input.shape();

    // Normalized shape is the last dimension (weight.shape)
    const int64_t norm_size = weight.numel();
    // F070: norm_size == 0 (empty weight tensor) would make the division below
    // an integer divide-by-zero (SIGFPE). Reject before any work / OpenMP.
    if (norm_size <= 0) {
        throw std::invalid_argument(
            "fused_rms_norm: weight must be non-empty (norm_size > 0)");
    }
    int64_t batch_size = input.numel() / norm_size;

    Tensor output(std::vector<int64_t>(shape.begin(), shape.end()),
                  input.dtype(), input.device());
    // rrms stores reciprocal RMS for each batch element. Store it at Float32 for
    // half inputs (and Float64 for Float64) so the backward reconstructs the
    // gradient from the full-precision statistic rather than a half-rounded one.
    DType rrms_dtype = (input.dtype() == DType::Float64) ? DType::Float64 : DType::Float32;
    Tensor rrms({batch_size}, rrms_dtype, input.device());

    if (input.dtype() == DType::Float32) {
        const float* in_data = input.data<float>();
        const float* w_data = weight.data<float>();
        float* out_data = output.data<float>();
        float* rrms_data = rrms.data<float>();

        #pragma omp parallel for if(batch_size > 64)
        for (int64_t b = 0; b < batch_size; ++b) {
            const float* x = in_data + b * norm_size;
            float* y = out_data + b * norm_size;

            // Sum of squares accumulated in double (widen f32 lanes) to avoid
            // mantissa loss in the RMS denominator for long hidden dims.
            const double sum_sq = fused_ln_sumsq_f64(x, norm_size, 0.0);
            const double mean_sq = sum_sq / static_cast<double>(norm_size);
            const float inv_rms =
                static_cast<float>(1.0 / std::sqrt(mean_sq + static_cast<double>(eps)));
            rrms_data[b] = inv_rms;

            // Apply normalization and weight with AVX2
#ifdef TENZOR_ATTN_AVX2
            {
                __m256 vinv = _mm256_set1_ps(inv_rms);
                int64_t i = 0;
                for (; i + 8 <= norm_size; i += 8) {
                    __m256 vx = _mm256_loadu_ps(x + i);
                    __m256 vw = _mm256_loadu_ps(w_data + i);
                    __m256 out = _mm256_mul_ps(_mm256_mul_ps(vx, vinv), vw);
                    _mm256_storeu_ps(y + i, out);
                }
                for (; i < norm_size; ++i) {
                    y[i] = x[i] * inv_rms * w_data[i];
                }
            }
#else
            for (int64_t i = 0; i < norm_size; ++i) {
                y[i] = x[i] * inv_rms * w_data[i];
            }
#endif
        }
    } else if (input.dtype() == DType::Float64) {
        const double* in_data = input.data<double>();
        const double* w_data = weight.data<double>();
        double* out_data = output.data<double>();
        double* rrms_data = rrms.data<double>();

        #pragma omp parallel for if(batch_size > 64)
        for (int64_t b = 0; b < batch_size; ++b) {
            const double* x = in_data + b * norm_size;
            double* y = out_data + b * norm_size;

            double sum_sq = 0.0;
            for (int64_t i = 0; i < norm_size; ++i) {
                sum_sq += x[i] * x[i];
            }
            double mean_sq = sum_sq / static_cast<double>(norm_size);
            double inv_rms = 1.0 / std::sqrt(mean_sq + static_cast<double>(eps));
            rrms_data[b] = inv_rms;

            for (int64_t i = 0; i < norm_size; ++i) {
                y[i] = x[i] * inv_rms * w_data[i];
            }
        }
    } else if (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16) {
        DType orig = input.dtype();
        Tensor in_f32 = input.to(DType::Float32);
        Tensor w_f32 = weight.to(DType::Float32);
        auto [out_f32, rrms_f32] = fused_rms_norm_kernel(in_f32, w_f32, eps);
        output = out_f32.to(orig);
        // Keep rrms at Float32 (do not narrow to the half input dtype) so the
        // backward consumes the full-precision reciprocal-RMS statistic.
        rrms = rrms_f32;
    } else {
        throw std::runtime_error("fused_rms_norm: unsupported dtype");
    }

    return {output, rrms};
}

auto rms_norm_backward_kernel(const Tensor& grad_output, const Tensor& input,
                               const Tensor& weight, const Tensor& rrms)
    -> std::tuple<Tensor, Tensor> {
    const int64_t norm_size = weight.numel();
    int64_t batch_size = input.numel() / norm_size;

    Tensor grad_input(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                      input.dtype(), input.device());
    Tensor grad_weight({norm_size}, input.dtype(), input.device());

    if (input.dtype() == DType::Float32) {
        const float* go = grad_output.data<float>();
        const float* x = input.data<float>();
        const float* w = weight.data<float>();
        const float* r = rrms.data<float>();
        float* gi = grad_input.data<float>();
        float* gw = grad_weight.data<float>();

        std::memset(gw, 0, norm_size * sizeof(float));

        // F035: match the forward's double accumulation. Accumulate the
        // per-feature grad_weight sum (across the batch) and the per-row dot
        // reduction in double; narrow to Float32 only on the final store.
        std::vector<double> gw_acc(static_cast<size_t>(norm_size), 0.0);

        for (int64_t b = 0; b < batch_size; ++b) {
            const float* go_b = go + b * norm_size;
            const float* x_b = x + b * norm_size;
            float* gi_b = gi + b * norm_size;
            const double inv_rms = static_cast<double>(r[b]);

            // grad_weight accumulation (double)
            for (int64_t i = 0; i < norm_size; ++i) {
                gw_acc[i] += static_cast<double>(go_b[i]) *
                             static_cast<double>(x_b[i]) * inv_rms;
            }

            // grad_input: d/dx of (x * rrms * w) with rrms depending on x
            double dot = 0.0;
            for (int64_t i = 0; i < norm_size; ++i) {
                dot += static_cast<double>(go_b[i]) *
                       static_cast<double>(w[i]) * static_cast<double>(x_b[i]);
            }
            const double coeff =
                dot * inv_rms * inv_rms / static_cast<double>(norm_size);
            for (int64_t i = 0; i < norm_size; ++i) {
                gi_b[i] = static_cast<float>(
                    (static_cast<double>(go_b[i]) * static_cast<double>(w[i]) -
                     static_cast<double>(x_b[i]) * coeff) * inv_rms);
            }
        }

        for (int64_t i = 0; i < norm_size; ++i) {
            gw[i] = static_cast<float>(gw_acc[i]);
        }
    } else if (input.dtype() == DType::Float64) {
        const double* go = grad_output.data<double>();
        const double* x = input.data<double>();
        const double* w = weight.data<double>();
        const double* r = rrms.data<double>();
        double* gi = grad_input.data<double>();
        double* gw = grad_weight.data<double>();

        std::memset(gw, 0, norm_size * sizeof(double));

        for (int64_t b = 0; b < batch_size; ++b) {
            const double* go_b = go + b * norm_size;
            const double* x_b = x + b * norm_size;
            double* gi_b = gi + b * norm_size;
            double inv_rms = r[b];

            for (int64_t i = 0; i < norm_size; ++i) {
                gw[i] += go_b[i] * x_b[i] * inv_rms;
            }

            double dot = 0.0;
            for (int64_t i = 0; i < norm_size; ++i) {
                dot += go_b[i] * w[i] * x_b[i];
            }
            double coeff = dot * inv_rms * inv_rms / static_cast<double>(norm_size);
            for (int64_t i = 0; i < norm_size; ++i) {
                gi_b[i] = (go_b[i] * w[i] - x_b[i] * coeff) * inv_rms;
            }
        }
    } else if (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16) {
        DType orig = input.dtype();
        Tensor go_f32 = grad_output.to(DType::Float32);
        Tensor in_f32 = input.to(DType::Float32);
        Tensor w_f32 = weight.to(DType::Float32);
        Tensor r_f32 = rrms.to(DType::Float32);
        auto [gi_f32, gw_f32] = rms_norm_backward_kernel(go_f32, in_f32, w_f32, r_f32);
        grad_input = gi_f32.to(orig);
        grad_weight = gw_f32.to(orig);
    } else {
        throw std::runtime_error("rms_norm_backward: unsupported dtype");
    }

    return {grad_input, grad_weight};
}

// =========================================================================
// Fused Attention with Online Softmax, SIMD, and Causal Masking
// =========================================================================
//
// Implements a tiled attention kernel inspired by Flash Attention:
//   output = softmax(Q @ K^T * scale [+ causal_mask]) @ V
//
// Key optimizations over the previous naive implementation:
//   1. Online softmax: processes K/V in tiles, never materializes the full
//      (seq_q x seq_k) score matrix. Memory is O(seq_q * TILE_K) instead
//      of O(seq_q * seq_k). This is critical for long sequences.
//   2. AVX2/AVX-512 SIMD for dot products and weighted accumulation.
//   3. Causal masking applied during score computation (no separate pass).
//   4. Float16/Float64 support via convert-compute-convert.
//

namespace {

// Tile size for K/V dimension. Chosen to keep the tile score buffer
// (seq_q * TILE_K floats) comfortably in L2 while giving enough work
// per tile to amortize loop overhead.
constexpr int64_t ATTN_TILE_K = 64;

// ---- SIMD helpers (local to this TU) ------------------------------------

#ifdef TENZOR_ATTN_AVX2

__attribute__((target("avx2,fma")))
static inline float attn_hsum_avx2(__m256 v) {
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 s  = _mm_add_ps(hi, lo);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s);
}

__attribute__((target("avx2,fma")))
static inline float attn_hmax_avx2(__m256 v) {
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 m  = _mm_max_ps(hi, lo);
    m = _mm_max_ps(m, _mm_shuffle_ps(m, m, _MM_SHUFFLE(2, 3, 0, 1)));
    m = _mm_max_ps(m, _mm_shuffle_ps(m, m, _MM_SHUFFLE(1, 0, 3, 2)));
    return _mm_cvtss_f32(m);
}

// SIMD dot product of two float vectors of length `len`.
__attribute__((target("avx2,fma")))
static inline float attn_dot_avx2(const float* a, const float* b, int64_t len) {
    __m256 vsum = _mm256_setzero_ps();
    int64_t d = 0;
    for (; d + 8 <= len; d += 8) {
        __m256 va = _mm256_loadu_ps(a + d);
        __m256 vb = _mm256_loadu_ps(b + d);
        vsum = _mm256_fmadd_ps(va, vb, vsum);
    }
    float dot = attn_hsum_avx2(vsum);
    for (; d < len; ++d) {
        dot += a[d] * b[d];
    }
    return dot;
}

// Weighted accumulation: out[d] += weight * v[d] for d in [0, len).
__attribute__((target("avx2,fma")))
static inline void attn_axpy_avx2(float* out, float weight, const float* v, int64_t len) {
    __m256 vw = _mm256_set1_ps(weight);
    int64_t d = 0;
    for (; d + 8 <= len; d += 8) {
        __m256 vo = _mm256_loadu_ps(out + d);
        __m256 vv = _mm256_loadu_ps(v + d);
        vo = _mm256_fmadd_ps(vw, vv, vo);
        _mm256_storeu_ps(out + d, vo);
    }
    for (; d < len; ++d) {
        out[d] += weight * v[d];
    }
}

// Scale a float vector: out[d] *= s for d in [0, len).
__attribute__((target("avx2,fma")))
static inline void attn_scale_avx2(float* out, float s, int64_t len) {
    __m256 vs = _mm256_set1_ps(s);
    int64_t d = 0;
    for (; d + 8 <= len; d += 8) {
        __m256 v = _mm256_loadu_ps(out + d);
        v = _mm256_mul_ps(v, vs);
        _mm256_storeu_ps(out + d, v);
    }
    for (; d < len; ++d) {
        out[d] *= s;
    }
}

#endif // TENZOR_ATTN_AVX2

#ifdef TENZOR_ATTN_AVX512

__attribute__((target("avx512f")))
static inline float attn_hsum_avx512(__m512 v) {
    return _mm512_reduce_add_ps(v);
}

__attribute__((target("avx512f")))
static inline float attn_hmax_avx512(__m512 v) {
    return _mm512_reduce_max_ps(v);
}

__attribute__((target("avx512f")))
static inline float attn_dot_avx512(const float* a, const float* b, int64_t len) {
    __m512 vsum = _mm512_setzero_ps();
    int64_t d = 0;
    for (; d + 16 <= len; d += 16) {
        __m512 va = _mm512_loadu_ps(a + d);
        __m512 vb = _mm512_loadu_ps(b + d);
        vsum = _mm512_fmadd_ps(va, vb, vsum);
    }
    float dot = attn_hsum_avx512(vsum);
    // Handle remainder with AVX2 or scalar
    for (; d < len; ++d) {
        dot += a[d] * b[d];
    }
    return dot;
}

__attribute__((target("avx512f")))
static inline void attn_axpy_avx512(float* out, float weight, const float* v, int64_t len) {
    __m512 vw = _mm512_set1_ps(weight);
    int64_t d = 0;
    for (; d + 16 <= len; d += 16) {
        __m512 vo = _mm512_loadu_ps(out + d);
        __m512 vv = _mm512_loadu_ps(v + d);
        vo = _mm512_fmadd_ps(vw, vv, vo);
        _mm512_storeu_ps(out + d, vo);
    }
    for (; d < len; ++d) {
        out[d] += weight * v[d];
    }
}

__attribute__((target("avx512f")))
static inline void attn_scale_avx512(float* out, float s, int64_t len) {
    __m512 vs = _mm512_set1_ps(s);
    int64_t d = 0;
    for (; d + 16 <= len; d += 16) {
        __m512 v = _mm512_loadu_ps(out + d);
        v = _mm512_mul_ps(v, vs);
        _mm512_storeu_ps(out + d, v);
    }
    for (; d < len; ++d) {
        out[d] *= s;
    }
}

#endif // TENZOR_ATTN_AVX512

// ---- Dispatch wrappers --------------------------------------------------

static inline float attn_dot(const float* a, const float* b, int64_t len) {
#ifdef TENZOR_ATTN_AVX512
    return attn_dot_avx512(a, b, len);
#elif defined(TENZOR_ATTN_AVX2)
    return attn_dot_avx2(a, b, len);
#else
    float dot = 0.0f;
    for (int64_t d = 0; d < len; ++d) dot += a[d] * b[d];
    return dot;
#endif
}

static inline void attn_axpy(float* out, float w, const float* v, int64_t len) {
#ifdef TENZOR_ATTN_AVX512
    attn_axpy_avx512(out, w, v, len);
#elif defined(TENZOR_ATTN_AVX2)
    attn_axpy_avx2(out, w, v, len);
#else
    for (int64_t d = 0; d < len; ++d) out[d] += w * v[d];
#endif
}

static inline void attn_scale(float* out, float s, int64_t len) {
#ifdef TENZOR_ATTN_AVX512
    attn_scale_avx512(out, s, len);
#elif defined(TENZOR_ATTN_AVX2)
    attn_scale_avx2(out, s, len);
#else
    for (int64_t d = 0; d < len; ++d) out[d] *= s;
#endif
}

// ---- Online-softmax tiled attention for one (batch, head) ---------------
//
// For each query row i, we iterate over K/V in tiles of ATTN_TILE_K columns.
// Per query row we maintain:
//   m_i  = running max of scores seen so far
//   l_i  = running sum of exp(scores - m_i)
//   o_i  = running weighted sum (unnormalized output)
//
// When processing a new tile [j0, j1):
//   1. Compute score[j] = q_i . k_j * scale  for j in [j0, j1)
//      Apply causal mask: if j > i, score[j] = -inf
//   2. m_new = max(m_i, max(score[j0..j1)))
//   3. Rescale previous accumulator:
//        correction = exp(m_i - m_new)
//        o_i *= correction
//        l_i *= correction
//   4. For each j in [j0, j1):
//        p_j  = exp(score[j] - m_new)
//        l_i += p_j
//        o_i += p_j * v_j
//   5. m_i = m_new
// After all tiles: o_i /= l_i
//
static void attention_online_f32(
    const float* __restrict__ q_data,
    const float* __restrict__ k_data,
    const float* __restrict__ v_data,
    float* __restrict__ out_data,
    int64_t batch_heads,
    int64_t seq_q,
    int64_t seq_k,
    int64_t head_dim,
    float scale,
    bool causal,
    int64_t H_q = 1,                  // Q heads per batch (equals batch_heads/B)
    int64_t q_heads_per_kv_head = 1   // 1 = MHA, > 1 = GQA, == H_q = MQA
) {
    // Per docs/internals/attention-contract.md GQA section: when H_kv < H_q,
    // K/V have fewer heads and the kernel broadcasts them along the head dim
    // via index math (kv_h = h_q / q_heads_per_kv_head). H_q == 1 (default)
    // matches the legacy MHA path where K and V are indexed identically to Q.
    const int64_t q_stride = seq_q * head_dim;
    const int64_t k_stride = seq_k * head_dim;
    const int64_t H_kv = H_q / q_heads_per_kv_head;
    const bool is_gqa = (q_heads_per_kv_head > 1);

    #pragma omp parallel for schedule(dynamic) if(batch_heads > 1)
    for (int64_t bh = 0; bh < batch_heads; ++bh) {
        // Compute the matching KV head index. For MHA (q_heads_per_kv_head==1)
        // this is simply bh; for GQA we map H_q query heads onto H_kv KV heads.
        int64_t bh_kv;
        if (is_gqa) {
            int64_t b = bh / H_q;
            int64_t h_q = bh % H_q;
            int64_t h_kv = h_q / q_heads_per_kv_head;
            bh_kv = b * H_kv + h_kv;
        } else {
            bh_kv = bh;
        }
        const float* q = q_data + bh * q_stride;
        const float* k = k_data + bh_kv * k_stride;
        const float* v = v_data + bh_kv * k_stride;  // V stride == K stride
        float* o = out_data + bh * q_stride;

        // Per-thread tile score buffer, kept small for L1 residency.
        alignas(64) float tile_scores[ATTN_TILE_K];

        for (int64_t i = 0; i < seq_q; ++i) {
            const float* qi = q + i * head_dim;
            float* oi = o + i * head_dim;

            // Running online-softmax state for query row i.
            float m_i = -std::numeric_limits<float>::infinity();
            float l_i = 0.0f;
            std::memset(oi, 0, head_dim * sizeof(float));

            // Upper bound for j when causal: only attend to positions <= i.
            const int64_t j_limit = causal ? std::min(i + 1, seq_k) : seq_k;

            for (int64_t j0 = 0; j0 < j_limit; j0 += ATTN_TILE_K) {
                const int64_t j1 = std::min(j0 + ATTN_TILE_K, j_limit);
                const int64_t tile_len = j1 - j0;

                // --- Step 1: Compute scores for this tile ---
                float tile_max = -std::numeric_limits<float>::infinity();
                for (int64_t t = 0; t < tile_len; ++t) {
                    tile_scores[t] = attn_dot(qi, k + (j0 + t) * head_dim, head_dim) * scale;
                    tile_max = std::max(tile_max, tile_scores[t]);
                }

                // --- Step 2: Compute new running max ---
                const float m_new = std::max(m_i, tile_max);

                // --- Step 3: Rescale previous accumulator ---
                if (l_i > 0.0f) {
                    const float correction = std::exp(m_i - m_new);
                    attn_scale(oi, correction, head_dim);
                    l_i *= correction;
                }

                // --- Step 4: Accumulate this tile ---
                float tile_sum = 0.0f;
                for (int64_t t = 0; t < tile_len; ++t) {
                    const float p = std::exp(tile_scores[t] - m_new);
                    tile_sum += p;
                    attn_axpy(oi, p, v + (j0 + t) * head_dim, head_dim);
                }
                l_i += tile_sum;

                // --- Step 5: Update running max ---
                m_i = m_new;
            }

            // --- Final normalization ---
            if (l_i > 0.0f) {
                attn_scale(oi, 1.0f / l_i, head_dim);
            }
        }
    }
}

// S13 fix — dtype-preservation. Native Float64 online-softmax attention.
//
// The Float32 path above relies on AVX2/AVX-512 SIMD intrinsics in
// attn_dot/attn_scale/attn_axpy. The Float64 path here is a scalar
// implementation (no Float64 SIMD intrinsics provided), but it keeps
// every accumulator at double precision throughout — no silent
// Float32 downcast like the previous warn-and-cast behaviour. The
// scalar loop is small enough to be vectorised by the compiler on
// most targets; correctness over peak throughput is the right
// tradeoff for the Float64 path.
static void attention_online_f64(
    const double* __restrict__ q_data,
    const double* __restrict__ k_data,
    const double* __restrict__ v_data,
    double* __restrict__ out_data,
    int64_t batch_heads,
    int64_t seq_q,
    int64_t seq_k,
    int64_t head_dim,
    double scale,
    bool causal,
    int64_t H_q = 1,
    int64_t q_heads_per_kv_head = 1
) {
    const int64_t q_stride = seq_q * head_dim;
    const int64_t k_stride = seq_k * head_dim;
    const int64_t H_kv = H_q / q_heads_per_kv_head;
    const bool is_gqa = (q_heads_per_kv_head > 1);

    #pragma omp parallel for schedule(dynamic) if(batch_heads > 1)
    for (int64_t bh = 0; bh < batch_heads; ++bh) {
        int64_t bh_kv;
        if (is_gqa) {
            int64_t b = bh / H_q;
            int64_t h_q = bh % H_q;
            int64_t h_kv = h_q / q_heads_per_kv_head;
            bh_kv = b * H_kv + h_kv;
        } else {
            bh_kv = bh;
        }
        const double* q = q_data + bh * q_stride;
        const double* k = k_data + bh_kv * k_stride;
        const double* v = v_data + bh_kv * k_stride;
        double* o = out_data + bh * q_stride;

        alignas(64) double tile_scores[ATTN_TILE_K];

        for (int64_t i = 0; i < seq_q; ++i) {
            const double* qi = q + i * head_dim;
            double* oi = o + i * head_dim;

            double m_i = -std::numeric_limits<double>::infinity();
            double l_i = 0.0;
            std::memset(oi, 0, head_dim * sizeof(double));

            const int64_t j_limit = causal ? std::min(i + 1, seq_k) : seq_k;

            for (int64_t j0 = 0; j0 < j_limit; j0 += ATTN_TILE_K) {
                const int64_t j1 = std::min(j0 + ATTN_TILE_K, j_limit);
                const int64_t tile_len = j1 - j0;

                double tile_max = -std::numeric_limits<double>::infinity();
                for (int64_t t = 0; t < tile_len; ++t) {
                    const double* k_row = k + (j0 + t) * head_dim;
                    double dot = 0.0;
                    for (int64_t d = 0; d < head_dim; ++d) {
                        dot += qi[d] * k_row[d];
                    }
                    tile_scores[t] = dot * scale;
                    tile_max = std::max(tile_max, tile_scores[t]);
                }

                const double m_new = std::max(m_i, tile_max);

                if (l_i > 0.0) {
                    const double correction = std::exp(m_i - m_new);
                    for (int64_t d = 0; d < head_dim; ++d) {
                        oi[d] *= correction;
                    }
                    l_i *= correction;
                }

                double tile_sum = 0.0;
                for (int64_t t = 0; t < tile_len; ++t) {
                    const double p = std::exp(tile_scores[t] - m_new);
                    tile_sum += p;
                    const double* v_row = v + (j0 + t) * head_dim;
                    for (int64_t d = 0; d < head_dim; ++d) {
                        oi[d] += p * v_row[d];
                    }
                }
                l_i += tile_sum;

                m_i = m_new;
            }

            if (l_i > 0.0) {
                const double inv = 1.0 / l_i;
                for (int64_t d = 0; d < head_dim; ++d) {
                    oi[d] *= inv;
                }
            }
        }
    }
}

} // anonymous namespace

auto fused_attention_kernel(const Tensor& Q, const Tensor& K, const Tensor& V,
                            float scale, bool causal) -> Tensor {
    // Q, K, V: (batch_heads, seq_len, head_dim) for 3D (no GQA — H_q == H_kv)
    //          (batch, num_heads, seq_len, head_dim) for 4D (GQA when K/V have
    //          fewer heads than Q; per attention-contract.md kv_h = h_q*H_kv/H_q).
    //
    // Callers commonly hand us Q/K/V as the output of `permute({0,2,1,3})` on
    // freshly reshaped projections — strided views, not contiguous tensors.
    // attention_online_f32 below reads via raw pointer arithmetic assuming
    // contiguous [B*H, L, D] / [B, H, L, D] layout, so non-contiguous inputs
    // silently produce wrong outputs. The cuDNN-SDPA / FlashAttention sibling
    // paths in MultiheadAttention already contiguise upstream; mirror that so
    // the CPU fused path agrees with the GPU paths.
    if (!Q.is_contiguous() || !K.is_contiguous() || !V.is_contiguous()) {
        return fused_attention_kernel(
            Q.is_contiguous() ? Q : Q.contiguous(),
            K.is_contiguous() ? K : K.contiguous(),
            V.is_contiguous() ? V : V.contiguous(),
            scale, causal);
    }

    const auto& q_shape = Q.shape();
    const int64_t ndim = Q.ndim();

    int64_t batch_heads, seq_len_q, head_dim, seq_len_k;
    int64_t H_q = 1;
    int64_t q_heads_per_kv_head = 1;

    if (ndim == 3) {
        batch_heads = q_shape[0];
        seq_len_q = q_shape[1];
        head_dim = q_shape[2];
        seq_len_k = K.shape()[1];
        if (K.shape()[0] != batch_heads) {
            throw std::runtime_error(
                "fused_attention: 3D layout requires Q.shape[0] == K.shape[0]; "
                "GQA must be expressed as 4D (B, H, S, D) so the kernel can "
                "reason about the H_q vs H_kv ratio.");
        }
    } else if (ndim == 4) {
        H_q = q_shape[1];
        int64_t H_kv = K.shape()[1];
        if (H_kv == 0 || H_q % H_kv != 0) {
            throw std::runtime_error(
                "fused_attention: H_q (" + std::to_string(H_q) + ") must be a "
                "positive multiple of H_kv (" + std::to_string(H_kv) +
                ") for GQA; got non-divisible heads.");
        }
        q_heads_per_kv_head = H_q / H_kv;
        batch_heads = q_shape[0] * H_q;
        seq_len_q = q_shape[2];
        head_dim = q_shape[3];
        seq_len_k = K.shape()[2];
    } else {
        throw std::runtime_error("fused_attention: Q must be 3D or 4D");
    }

    // Validate V's shape against K (the online-softmax loops read V with the
    // SAME stride/extents as K: v + bh_kv*k_stride and v + (j0+t)*head_dim for
    // seq_len_k rows). If V's seq_len or head_dim is smaller than K's, those
    // reads run past V's buffer. Likewise the Q·K dot product assumes K's
    // head_dim equals Q's head_dim. Validate both before launching the kernels.
    if (K.ndim() != ndim || V.ndim() != ndim) {
        throw std::runtime_error(
            "fused_attention: Q, K, V must all have the same rank (" +
            std::to_string(ndim) + ").");
    }
    {
        auto k_shape = K.shape();
        auto v_shape = V.shape();
        bool v_matches_k =
            (v_shape.size() == k_shape.size()) &&
            std::equal(k_shape.begin(), k_shape.end(), v_shape.begin());
        if (!v_matches_k) {
            throw std::runtime_error(
                "fused_attention: V.shape must equal K.shape (same heads, "
                "seq_len_k and head_dim); V is read with K's stride and "
                "extents.");
        }
    }
    const int64_t k_head_dim = K.shape()[ndim - 1];
    if (k_head_dim != head_dim) {
        throw std::runtime_error(
            "fused_attention: K head_dim (" + std::to_string(k_head_dim) +
            ") must equal Q head_dim (" + std::to_string(head_dim) +
            ") for the Q·K dot product.");
    }

    Tensor output(std::vector<int64_t>(q_shape.begin(), q_shape.end()),
                  Q.dtype(), Q.device());

    if (Q.dtype() == DType::Float32) {
        attention_online_f32(
            Q.data<float>(), K.data<float>(), V.data<float>(),
            output.data<float>(),
            batch_heads, seq_len_q, seq_len_k, head_dim,
            scale, causal, H_q, q_heads_per_kv_head
        );
    } else if (Q.dtype() == DType::Float64) {
        // S13 fix — native Float64 path. Previously this branch widened to
        // Float32, computed, and narrowed back with a one-shot warning;
        // that silently destroyed the user's requested precision. Now we
        // run the online softmax in double throughout via
        // attention_online_f64 (scalar, but precise).
        attention_online_f64(
            Q.data<double>(), K.data<double>(), V.data<double>(),
            output.data<double>(),
            batch_heads, seq_len_q, seq_len_k, head_dim,
            static_cast<double>(scale), causal, H_q, q_heads_per_kv_head
        );
    } else if (Q.dtype() == DType::Float16) {
        // Float16: convert to Float32, compute, convert back.
        Tensor q32 = Q.to(DType::Float32);
        Tensor k32 = K.to(DType::Float32);
        Tensor v32 = V.to(DType::Float32);
        Tensor out32(std::vector<int64_t>(q_shape.begin(), q_shape.end()),
                     DType::Float32, Q.device());

        attention_online_f32(
            q32.data<float>(), k32.data<float>(), v32.data<float>(),
            out32.data<float>(),
            batch_heads, seq_len_q, seq_len_k, head_dim,
            scale, causal, H_q, q_heads_per_kv_head
        );
        output = out32.to(DType::Float16);
    } else if (Q.dtype() == DType::BFloat16) {
        // BFloat16: convert to Float32, compute, convert back.
        Tensor q32 = Q.to(DType::Float32);
        Tensor k32 = K.to(DType::Float32);
        Tensor v32 = V.to(DType::Float32);
        Tensor out32(std::vector<int64_t>(q_shape.begin(), q_shape.end()),
                     DType::Float32, Q.device());

        attention_online_f32(
            q32.data<float>(), k32.data<float>(), v32.data<float>(),
            out32.data<float>(),
            batch_heads, seq_len_q, seq_len_k, head_dim,
            scale, causal, H_q, q_heads_per_kv_head
        );
        output = out32.to(DType::BFloat16);
    } else {
        throw std::runtime_error("fused_attention: unsupported dtype (expected Float32, Float64, Float16, or BFloat16)");
    }

    return output;
}

// =========================================================================
// Fused Conv2d + Activation Variants
// =========================================================================

// Helper: apply activation in-place
namespace {

template<typename T>
auto apply_sigmoid_inplace(T* data, int64_t n) -> void {
    for (int64_t i = 0; i < n; ++i) {
        data[i] = T(1) / (T(1) + std::exp(-data[i]));
    }
}

template<typename T>
auto apply_tanh_inplace(T* data, int64_t n) -> void {
    for (int64_t i = 0; i < n; ++i) {
        data[i] = std::tanh(data[i]);
    }
}

template<typename T>
auto apply_swish_inplace(T* data, int64_t n) -> void {
    for (int64_t i = 0; i < n; ++i) {
        T sigmoid = T(1) / (T(1) + std::exp(-data[i]));
        data[i] = data[i] * sigmoid;
    }
}

} // anonymous namespace

// Per-axis overloads (audit F.11) — delegate to per-axis conv2d_forward_kernel.
auto fused_conv2d_sigmoid_kernel(const Tensor& input, const Tensor& weight, const Tensor* bias,
                                  int64_t stride_h, int64_t stride_w,
                                  int64_t pad_h, int64_t pad_w,
                                  int64_t dil_h, int64_t dil_w, int64_t groups) -> Tensor {
    // Widen Float16/BFloat16 inputs to Float32 so the activation step
    // does not silently no-op. Float32/Float64 paths are pass-through.
    return ::tenzor::utils::widen_narrow_compute(input,
        [&](const Tensor& in_wide) -> Tensor {
            Tensor w_wide = ::tenzor::utils::is_half_precision(weight.dtype())
                                ? weight.to(DType::Float32)
                                : weight;
            Tensor b_wide_storage;
            const Tensor* b_ptr = bias;
            if (bias != nullptr && ::tenzor::utils::is_half_precision(bias->dtype())) {
                b_wide_storage = bias->to(DType::Float32);
                b_ptr = &b_wide_storage;
            }
            Tensor result = conv2d_forward_kernel(in_wide, w_wide, b_ptr,
                                                  stride_h, stride_w, pad_h, pad_w, dil_h, dil_w, groups);
            int64_t n = result.numel();
            if (result.dtype() == DType::Float32) {
                apply_sigmoid_inplace(result.data<float>(), n);
            } else if (result.dtype() == DType::Float64) {
                apply_sigmoid_inplace(result.data<double>(), n);
            }
            return result;
        });
}

auto fused_conv2d_sigmoid_kernel(const Tensor& input, const Tensor& weight, const Tensor* bias,
                                  int64_t stride, int64_t padding, int64_t dilation, int64_t groups) -> Tensor {
    return fused_conv2d_sigmoid_kernel(input, weight, bias,
                                       stride, stride, padding, padding, dilation, dilation, groups);
}

auto fused_conv2d_tanh_kernel(const Tensor& input, const Tensor& weight, const Tensor* bias,
                               int64_t stride_h, int64_t stride_w,
                               int64_t pad_h, int64_t pad_w,
                               int64_t dil_h, int64_t dil_w, int64_t groups) -> Tensor {
    // Widen Float16/BFloat16 inputs to Float32 so the activation step
    // does not silently no-op. Float32/Float64 paths are pass-through.
    return ::tenzor::utils::widen_narrow_compute(input,
        [&](const Tensor& in_wide) -> Tensor {
            Tensor w_wide = ::tenzor::utils::is_half_precision(weight.dtype())
                                ? weight.to(DType::Float32)
                                : weight;
            Tensor b_wide_storage;
            const Tensor* b_ptr = bias;
            if (bias != nullptr && ::tenzor::utils::is_half_precision(bias->dtype())) {
                b_wide_storage = bias->to(DType::Float32);
                b_ptr = &b_wide_storage;
            }
            Tensor result = conv2d_forward_kernel(in_wide, w_wide, b_ptr,
                                                  stride_h, stride_w, pad_h, pad_w, dil_h, dil_w, groups);
            int64_t n = result.numel();
            if (result.dtype() == DType::Float32) {
                apply_tanh_inplace(result.data<float>(), n);
            } else if (result.dtype() == DType::Float64) {
                apply_tanh_inplace(result.data<double>(), n);
            }
            return result;
        });
}

auto fused_conv2d_tanh_kernel(const Tensor& input, const Tensor& weight, const Tensor* bias,
                               int64_t stride, int64_t padding, int64_t dilation, int64_t groups) -> Tensor {
    return fused_conv2d_tanh_kernel(input, weight, bias,
                                    stride, stride, padding, padding, dilation, dilation, groups);
}

auto fused_conv2d_swish_kernel(const Tensor& input, const Tensor& weight, const Tensor* bias,
                                int64_t stride_h, int64_t stride_w,
                                int64_t pad_h, int64_t pad_w,
                                int64_t dil_h, int64_t dil_w, int64_t groups) -> Tensor {
    // Widen Float16/BFloat16 inputs to Float32 so the activation step
    // does not silently no-op. Float32/Float64 paths are pass-through.
    return ::tenzor::utils::widen_narrow_compute(input,
        [&](const Tensor& in_wide) -> Tensor {
            Tensor w_wide = ::tenzor::utils::is_half_precision(weight.dtype())
                                ? weight.to(DType::Float32)
                                : weight;
            Tensor b_wide_storage;
            const Tensor* b_ptr = bias;
            if (bias != nullptr && ::tenzor::utils::is_half_precision(bias->dtype())) {
                b_wide_storage = bias->to(DType::Float32);
                b_ptr = &b_wide_storage;
            }
            Tensor result = conv2d_forward_kernel(in_wide, w_wide, b_ptr,
                                                  stride_h, stride_w, pad_h, pad_w, dil_h, dil_w, groups);
            int64_t n = result.numel();
            if (result.dtype() == DType::Float32) {
                apply_swish_inplace(result.data<float>(), n);
            } else if (result.dtype() == DType::Float64) {
                apply_swish_inplace(result.data<double>(), n);
            }
            return result;
        });
}

auto fused_conv2d_swish_kernel(const Tensor& input, const Tensor& weight, const Tensor* bias,
                                int64_t stride, int64_t padding, int64_t dilation, int64_t groups) -> Tensor {
    return fused_conv2d_swish_kernel(input, weight, bias,
                                     stride, stride, padding, padding, dilation, dilation, groups);
}

// =========================================================================
// BatchNorm2d Fused Training
// =========================================================================

// Forward declarations
auto batchnorm2d_mean_var_kernel(const Tensor& input) -> std::vector<Tensor>;
auto batchnorm2d_forward_affine_kernel(const Tensor& input, const Tensor& mean,
                                        const Tensor& variance, const Tensor& gamma,
                                        const Tensor& beta, float epsilon) -> Tensor;
auto batchnorm2d_update_running_stats_kernel(Tensor& running_mean, Tensor& running_var,
                                              const Tensor& batch_mean, const Tensor& batch_var,
                                              float momentum) -> void;

auto batchnorm2d_fused_training_kernel(const Tensor& input, Tensor& running_mean, Tensor& running_var,
                                        const Tensor& gamma, const Tensor& beta,
                                        float momentum, float epsilon) -> std::vector<Tensor> {
    // Compute batch mean and variance
    auto mean_var = batchnorm2d_mean_var_kernel(input);
    Tensor batch_mean = mean_var[0];
    Tensor batch_var = mean_var[1];

    // Forward with affine (uses the BIASED batch variance)
    Tensor output = batchnorm2d_forward_affine_kernel(input, batch_mean, batch_var, gamma, beta, epsilon);

    // Update running stats. The RUNNING variance uses the UNBIASED
    // (Bessel-corrected) estimate var * count/(count-1) to match PyTorch / cuDNN
    // and the nn-layer's non-fused path; normalization above keeps the biased var.
    auto in_shape = input.shape();
    int64_t bn_count = in_shape[0] * in_shape[2] * in_shape[3];
    Tensor running_batch_var = batch_var;
    if (bn_count >= 2) {
        running_batch_var = tenzor::mul(
            batch_var, static_cast<double>(bn_count) / static_cast<double>(bn_count - 1));
    }
    batchnorm2d_update_running_stats_kernel(running_mean, running_var, batch_mean, running_batch_var, momentum);

    // Return: output, running_mean, running_var, saved_mean, saved_inv_var
    // Compute saved_inv_var for backward pass
    int64_t C = batch_var.numel();
    Tensor saved_inv_var({C}, batch_var.dtype(), batch_var.device());
    if (batch_var.dtype() == DType::Float32) {
        const float* var_data = batch_var.data<float>();
        float* inv_var_data = saved_inv_var.data<float>();
        for (int64_t i = 0; i < C; ++i) {
            inv_var_data[i] = 1.0f / std::sqrt(var_data[i] + epsilon);
        }
    } else if (batch_var.dtype() == DType::Float64) {
        const double* var_data = batch_var.data<double>();
        double* inv_var_data = saved_inv_var.data<double>();
        for (int64_t i = 0; i < C; ++i) {
            inv_var_data[i] = 1.0 / std::sqrt(var_data[i] + static_cast<double>(epsilon));
        }
    } else if (batch_var.dtype() == DType::Float16) {
        const Float16* var_data = batch_var.data<Float16>();
        Float16* inv_var_data = saved_inv_var.data<Float16>();
        for (int64_t i = 0; i < C; ++i) {
            float v = static_cast<float>(var_data[i]);
            inv_var_data[i] = Float16(1.0f / std::sqrt(v + epsilon));
        }
    } else if (batch_var.dtype() == DType::BFloat16) {
        const BFloat16* var_data = batch_var.data<BFloat16>();
        BFloat16* inv_var_data = saved_inv_var.data<BFloat16>();
        for (int64_t i = 0; i < C; ++i) {
            float v = static_cast<float>(var_data[i]);
            inv_var_data[i] = BFloat16(1.0f / std::sqrt(v + epsilon));
        }
    }

    return {output, running_mean, running_var, batch_mean, saved_inv_var};
}

// =========================================================================
// Fused Optimizer Steps
// =========================================================================

auto fused_sgd_step_kernel(Tensor& param, const Tensor& grad, Tensor* momentum_buffer,
                           double lr, double momentum, double weight_decay,
                           double dampening, bool nesterov) -> void {
    const int64_t n = param.numel();

    if (param.dtype() == DType::Float32) {
        float* p = param.data<float>();
        const float* g = grad.data<float>();

        #pragma omp parallel for if(static_cast<int64_t>(n) > ::tenzor::OmpThresholds::simple())
        for (int64_t i = 0; i < n; ++i) {
            float grad_val = g[i];
            if (weight_decay != 0.0f) {
                grad_val += weight_decay * p[i];
            }

            if (momentum != 0.0f && momentum_buffer) {
                float* buf = momentum_buffer->data<float>();
                buf[i] = momentum * buf[i] + (1.0f - dampening) * grad_val;
                if (nesterov) {
                    grad_val = grad_val + momentum * buf[i];
                } else {
                    grad_val = buf[i];
                }
            }

            p[i] -= lr * grad_val;
        }
    } else if (param.dtype() == DType::Float64) {
        double* p = param.data<double>();
        const double* g = grad.data<double>();

        #pragma omp parallel for if(static_cast<int64_t>(n) > ::tenzor::OmpThresholds::simple())
        for (int64_t i = 0; i < n; ++i) {
            double grad_val = g[i];
            if (weight_decay != 0.0) {
                grad_val += static_cast<double>(weight_decay) * p[i];
            }
            if (momentum != 0.0f && momentum_buffer) {
                double* buf = momentum_buffer->data<double>();
                buf[i] = static_cast<double>(momentum) * buf[i] + (1.0 - static_cast<double>(dampening)) * grad_val;
                if (nesterov) {
                    grad_val = grad_val + static_cast<double>(momentum) * buf[i];
                } else {
                    grad_val = buf[i];
                }
            }
            p[i] -= static_cast<double>(lr) * grad_val;
        }
    } else {
        // audit-8 GG.2: F16/BF16 must be widened by the caller before reaching
        // this kernel — silently no-op'ing was a P0 training-quality regression.
        throw std::runtime_error(
            "fused_sgd_step_kernel (CPU): unsupported dtype " +
            std::string(dtype_name(param.dtype())) +
            " — widen at caller via param.to(Float32) -> step -> cast back");
    }
}

auto fused_adam_step_kernel(Tensor& param, const Tensor& grad,
                           Tensor& exp_avg, Tensor& exp_avg_sq,
                           double lr, double beta1, double beta2,
                           double eps, double weight_decay,
                           int64_t step, bool decoupled_weight_decay,
                           Tensor* max_exp_avg_sq, bool amsgrad) -> void {
    const int64_t n = param.numel();

    // Bias correction
    double bias_correction1 = 1.0 - std::pow(beta1, static_cast<double>(step));
    double bias_correction2 = 1.0 - std::pow(beta2, static_cast<double>(step));
    double step_size = lr / bias_correction1;

    if (param.dtype() == DType::Float32) {
        float* p = param.data<float>();
        const float* g = grad.data<float>();
        float* m = exp_avg.data<float>();
        float* v = exp_avg_sq.data<float>();
        float* v_max = (amsgrad && max_exp_avg_sq) ? max_exp_avg_sq->data<float>() : nullptr;

        #pragma omp parallel for if(static_cast<int64_t>(n) > ::tenzor::OmpThresholds::simple())
        for (int64_t i = 0; i < n; ++i) {
            float grad_val = g[i];

            if (decoupled_weight_decay && weight_decay != 0.0) {
                p[i] *= static_cast<float>(1.0 - lr * weight_decay);
            } else if (weight_decay != 0.0) {
                grad_val += static_cast<float>(weight_decay) * p[i];
            }

            // Update biased first and second moment estimates
            m[i] = static_cast<float>(beta1) * m[i] + static_cast<float>(1.0 - beta1) * grad_val;
            v[i] = static_cast<float>(beta2) * v[i] + static_cast<float>(1.0 - beta2) * grad_val * grad_val;

            float denom;
            if (amsgrad && v_max) {
                v_max[i] = std::max(v_max[i], v[i]);
                denom = std::sqrt(v_max[i] / static_cast<float>(bias_correction2)) + static_cast<float>(eps);
            } else {
                denom = std::sqrt(v[i] / static_cast<float>(bias_correction2)) + static_cast<float>(eps);
            }

            p[i] -= static_cast<float>(step_size) * m[i] / denom;
        }
    } else if (param.dtype() == DType::Float64) {
        double* p = param.data<double>();
        const double* g = grad.data<double>();
        double* m = exp_avg.data<double>();
        double* v = exp_avg_sq.data<double>();
        double* v_max = (amsgrad && max_exp_avg_sq) ? max_exp_avg_sq->data<double>() : nullptr;

        #pragma omp parallel for if(static_cast<int64_t>(n) > ::tenzor::OmpThresholds::simple())
        for (int64_t i = 0; i < n; ++i) {
            double grad_val = g[i];
            if (decoupled_weight_decay && weight_decay != 0.0) {
                p[i] *= 1.0 - lr * weight_decay;
            } else if (weight_decay != 0.0) {
                grad_val += weight_decay * p[i];
            }
            m[i] = beta1 * m[i] + (1.0 - beta1) * grad_val;
            v[i] = beta2 * v[i] + (1.0 - beta2) * grad_val * grad_val;

            double denom;
            if (amsgrad && v_max) {
                v_max[i] = std::max(v_max[i], v[i]);
                denom = std::sqrt(v_max[i] / bias_correction2) + eps;
            } else {
                denom = std::sqrt(v[i] / bias_correction2) + eps;
            }
            p[i] -= step_size * m[i] / denom;
        }
    } else {
        // audit-8 GG.2: F16/BF16 must be widened by caller; silent no-op was a P0.
        throw std::runtime_error(
            "fused_adam_step_kernel (CPU): unsupported dtype " +
            std::string(dtype_name(param.dtype())) +
            " — widen at caller via param.to(Float32) -> step -> cast back");
    }
}

auto fused_adam_atan2_step_kernel(Tensor& param, const Tensor& grad,
                                  Tensor& exp_avg, Tensor& exp_avg_sq,
                                  Tensor* max_exp_avg_sq,
                                  double lr, double beta1, double beta2,
                                  double eps, double weight_decay,
                                  int64_t step, bool amsgrad) -> void {
    const int64_t n = param.numel();

    double bias_correction1 = 1.0 - std::pow(beta1, static_cast<double>(step));
    double bias_correction2 = 1.0 - std::pow(beta2, static_cast<double>(step));
    double step_size = lr / bias_correction1;

    if (param.dtype() == DType::Float32) {
        float* p = param.data<float>();
        const float* g = grad.data<float>();
        float* m = exp_avg.data<float>();
        float* v = exp_avg_sq.data<float>();
        float* v_max = (amsgrad && max_exp_avg_sq) ? max_exp_avg_sq->data<float>() : nullptr;

        #pragma omp parallel for if(static_cast<int64_t>(n) > ::tenzor::OmpThresholds::simple())
        for (int64_t i = 0; i < n; ++i) {
            float grad_val = g[i];
            if (weight_decay != 0.0f) {
                p[i] *= 1.0f - lr * weight_decay;
            }
            m[i] = beta1 * m[i] + (1.0f - beta1) * grad_val;
            v[i] = beta2 * v[i] + (1.0f - beta2) * grad_val * grad_val;

            float v_hat;
            if (amsgrad && v_max) {
                v_max[i] = std::max(v_max[i], v[i]);
                v_hat = v_max[i] / bias_correction2;
            } else {
                v_hat = v[i] / bias_correction2;
            }

            // Use atan2 for numerically stable update
            float update = std::atan2(m[i], std::sqrt(v_hat) + eps);
            p[i] -= step_size * update;
        }
    } else if (param.dtype() == DType::Float64) {
        double* p = param.data<double>();
        const double* g = grad.data<double>();
        double* m = exp_avg.data<double>();
        double* v = exp_avg_sq.data<double>();
        double* v_max = (amsgrad && max_exp_avg_sq) ? max_exp_avg_sq->data<double>() : nullptr;

        double d_bias_correction2 = 1.0 - std::pow(static_cast<double>(beta2), static_cast<double>(step));
        double d_step_size = static_cast<double>(lr) / (1.0 - std::pow(static_cast<double>(beta1), static_cast<double>(step)));

        #pragma omp parallel for if(static_cast<int64_t>(n) > ::tenzor::OmpThresholds::simple())
        for (int64_t i = 0; i < n; ++i) {
            double grad_val = g[i];
            if (weight_decay != 0.0f) {
                p[i] *= 1.0 - static_cast<double>(lr) * static_cast<double>(weight_decay);
            }
            m[i] = static_cast<double>(beta1) * m[i] + (1.0 - static_cast<double>(beta1)) * grad_val;
            v[i] = static_cast<double>(beta2) * v[i] + (1.0 - static_cast<double>(beta2)) * grad_val * grad_val;

            double v_hat;
            if (amsgrad && v_max) {
                v_max[i] = std::max(v_max[i], v[i]);
                v_hat = v_max[i] / d_bias_correction2;
            } else {
                v_hat = v[i] / d_bias_correction2;
            }

            double update = std::atan2(m[i], std::sqrt(v_hat) + static_cast<double>(eps));
            p[i] -= d_step_size * update;
        }
    } else {
        // audit-8 GG.2: F16/BF16 must be widened by caller; silent no-op was a P0.
        throw std::runtime_error(
            "fused_adam_atan2_step_kernel (CPU): unsupported dtype " +
            std::string(dtype_name(param.dtype())) +
            " — widen at caller via param.to(Float32) -> step -> cast back");
    }
}

auto fused_rmsprop_step_kernel(Tensor& param, const Tensor& grad,
                                Tensor& square_avg, Tensor* grad_avg,
                                Tensor* momentum_buffer,
                                double lr, double alpha, double eps,
                                double weight_decay, double momentum,
                                bool centered) -> void {
    const int64_t n = param.numel();

    if (param.dtype() == DType::Float32) {
        float* p = param.data<float>();
        const float* g = grad.data<float>();
        float* sq = square_avg.data<float>();
        float* ga = (centered && grad_avg) ? grad_avg->data<float>() : nullptr;
        float* buf = (momentum > 0.0f && momentum_buffer) ? momentum_buffer->data<float>() : nullptr;

        #pragma omp parallel for if(static_cast<int64_t>(n) > ::tenzor::OmpThresholds::simple())
        for (int64_t i = 0; i < n; ++i) {
            float grad_val = g[i];
            if (weight_decay != 0.0f) {
                grad_val += weight_decay * p[i];
            }

            sq[i] = alpha * sq[i] + (1.0f - alpha) * grad_val * grad_val;

            float avg;
            if (centered && ga) {
                ga[i] = alpha * ga[i] + (1.0f - alpha) * grad_val;
                // F040: the variance estimate sq - ga^2 can go slightly negative
                // under float rounding -> sqrt(NaN). Clamp to >=0. eps is added
                // OUTSIDE the sqrt to match PyTorch RMSprop (avg.sqrt().add_(eps))
                // and the non-centered path above.
                avg = std::sqrt(std::max(0.0f, sq[i] - ga[i] * ga[i])) +
                      static_cast<float>(eps);
            } else {
                avg = std::sqrt(sq[i]) + eps;
            }

            if (buf) {
                buf[i] = momentum * buf[i] + grad_val / avg;
                p[i] -= lr * buf[i];
            } else {
                p[i] -= lr * grad_val / avg;
            }
        }
    } else if (param.dtype() == DType::Float64) {
        double* p = param.data<double>();
        const double* g = grad.data<double>();
        double* sq = square_avg.data<double>();
        double* ga = (centered && grad_avg) ? grad_avg->data<double>() : nullptr;
        double* buf = (momentum > 0.0f && momentum_buffer) ? momentum_buffer->data<double>() : nullptr;

        #pragma omp parallel for if(static_cast<int64_t>(n) > ::tenzor::OmpThresholds::simple())
        for (int64_t i = 0; i < n; ++i) {
            double grad_val = g[i];
            if (weight_decay != 0.0f) {
                grad_val += static_cast<double>(weight_decay) * p[i];
            }

            sq[i] = static_cast<double>(alpha) * sq[i] + (1.0 - static_cast<double>(alpha)) * grad_val * grad_val;

            double avg;
            if (centered && ga) {
                ga[i] = alpha * ga[i] + (1.0 - alpha) * grad_val;
                // F040: clamp variance estimate to >=0; eps OUTSIDE the sqrt to
                // match PyTorch RMSprop and the non-centered path.
                avg = std::sqrt(std::max(0.0, sq[i] - ga[i] * ga[i])) + eps;
            } else {
                avg = std::sqrt(sq[i]) + eps;
            }

            if (buf) {
                buf[i] = static_cast<double>(momentum) * buf[i] + grad_val / avg;
                p[i] -= static_cast<double>(lr) * buf[i];
            } else {
                p[i] -= static_cast<double>(lr) * grad_val / avg;
            }
        }
    } else {
        // audit-8 GG.2: F16/BF16 must be widened by caller; silent no-op was a P0.
        throw std::runtime_error(
            "fused_rmsprop_step_kernel (CPU): unsupported dtype " +
            std::string(dtype_name(param.dtype())) +
            " — widen at caller via param.to(Float32) -> step -> cast back");
    }
}

auto fused_adadelta_step_kernel(Tensor& param, const Tensor& grad,
                                 Tensor& square_avg, Tensor& acc_delta,
                                 double rho, double eps, double lr,
                                 double weight_decay) -> void {
    const int64_t n = param.numel();

    if (param.dtype() == DType::Float32) {
        float* p = param.data<float>();
        const float* g = grad.data<float>();
        float* sq = square_avg.data<float>();
        float* ad = acc_delta.data<float>();

        #pragma omp parallel for if(static_cast<int64_t>(n) > ::tenzor::OmpThresholds::simple())
        for (int64_t i = 0; i < n; ++i) {
            float grad_val = g[i];
            if (weight_decay != 0.0f) {
                grad_val += weight_decay * p[i];
            }

            sq[i] = rho * sq[i] + (1.0f - rho) * grad_val * grad_val;
            float delta = std::sqrt(ad[i] + eps) / std::sqrt(sq[i] + eps) * grad_val;
            ad[i] = rho * ad[i] + (1.0f - rho) * delta * delta;
            p[i] -= lr * delta;
        }
    } else if (param.dtype() == DType::Float64) {
        double* p = param.data<double>();
        const double* g = grad.data<double>();
        double* sq = square_avg.data<double>();
        double* ad = acc_delta.data<double>();

        #pragma omp parallel for if(static_cast<int64_t>(n) > ::tenzor::OmpThresholds::simple())
        for (int64_t i = 0; i < n; ++i) {
            double grad_val = g[i];
            if (weight_decay != 0.0f) {
                grad_val += static_cast<double>(weight_decay) * p[i];
            }

            sq[i] = static_cast<double>(rho) * sq[i] + (1.0 - static_cast<double>(rho)) * grad_val * grad_val;
            double delta = std::sqrt(ad[i] + static_cast<double>(eps)) / std::sqrt(sq[i] + static_cast<double>(eps)) * grad_val;
            ad[i] = static_cast<double>(rho) * ad[i] + (1.0 - static_cast<double>(rho)) * delta * delta;
            p[i] -= static_cast<double>(lr) * delta;
        }
    } else {
        // audit-8 GG.2: F16/BF16 must be widened by caller; silent no-op was a P0.
        throw std::runtime_error(
            "fused_adadelta_step_kernel (CPU): unsupported dtype " +
            std::string(dtype_name(param.dtype())) +
            " — widen at caller via param.to(Float32) -> step -> cast back");
    }
}

auto fused_adagrad_step_kernel(Tensor& param, const Tensor& grad,
                                Tensor& sum_sq, double lr, double lr_decay,
                                double eps, double weight_decay,
                                int64_t step) -> void {
    const int64_t n = param.numel();
    double clr = lr / (1.0 + static_cast<double>(step - 1) * lr_decay);

    if (param.dtype() == DType::Float32) {
        float* p = param.data<float>();
        const float* g = grad.data<float>();
        float* ss = sum_sq.data<float>();

        #pragma omp parallel for if(static_cast<int64_t>(n) > ::tenzor::OmpThresholds::simple())
        for (int64_t i = 0; i < n; ++i) {
            float grad_val = g[i];
            if (weight_decay != 0.0f) {
                grad_val += weight_decay * p[i];
            }
            ss[i] += grad_val * grad_val;
            p[i] -= clr * grad_val / (std::sqrt(ss[i]) + eps);
        }
    } else if (param.dtype() == DType::Float64) {
        double* p = param.data<double>();
        const double* g = grad.data<double>();
        double* ss = sum_sq.data<double>();
        double clr_d = static_cast<double>(lr) / (1.0 + static_cast<double>(step - 1) * static_cast<double>(lr_decay));

        #pragma omp parallel for if(static_cast<int64_t>(n) > ::tenzor::OmpThresholds::simple())
        for (int64_t i = 0; i < n; ++i) {
            double grad_val = g[i];
            if (weight_decay != 0.0f) {
                grad_val += static_cast<double>(weight_decay) * p[i];
            }
            ss[i] += grad_val * grad_val;
            p[i] -= clr_d * grad_val / (std::sqrt(ss[i]) + static_cast<double>(eps));
        }
    } else {
        // audit-8 GG.2: F16/BF16 must be widened by caller; silent no-op was a P0.
        throw std::runtime_error(
            "fused_adagrad_step_kernel (CPU): unsupported dtype " +
            std::string(dtype_name(param.dtype())) +
            " — widen at caller via param.to(Float32) -> step -> cast back");
    }
}

// =========================================================================
// FusedConv2dBnReLU
// =========================================================================

auto fused_conv2d_bn_relu_kernel(
    const Tensor& input, const Tensor& weight, const Tensor& conv_bias,
    const Tensor& bn_gamma, const Tensor& bn_beta,
    const Tensor& bn_running_mean, const Tensor& bn_running_var,
    int64_t stride_h, int64_t stride_w,
    int64_t padding_h, int64_t padding_w,
    int64_t dilation_h, int64_t dilation_w,
    float bn_momentum, float bn_eps, bool training) -> Tensor {

    // Widen-narrow for non-Float32 inputs (audit M3 / feedback_float16_widen_narrow).
    // The underlying SIMD GEMM + conv code is float-typed; we cast to Float32 at
    // the boundary and back to the input dtype on return so all four dtypes
    // (Float32 / Float64 / Float16 / BFloat16) get parity coverage.
    DType orig_dtype = input.dtype();
    if (orig_dtype != DType::Float32) {
        Tensor input_f32 = input.dtype() == DType::Float32 ? input : input.to(DType::Float32);
        Tensor weight_f32 = weight.dtype() == DType::Float32 ? weight : weight.to(DType::Float32);
        Tensor conv_bias_f32 = (conv_bias.numel() == 0 || conv_bias.dtype() == DType::Float32)
                                   ? conv_bias
                                   : conv_bias.to(DType::Float32);
        Tensor bn_gamma_f32 = bn_gamma.dtype() == DType::Float32 ? bn_gamma : bn_gamma.to(DType::Float32);
        Tensor bn_beta_f32  = bn_beta.dtype()  == DType::Float32 ? bn_beta  : bn_beta.to(DType::Float32);
        Tensor bn_rm_f32    = bn_running_mean.dtype() == DType::Float32 ? bn_running_mean : bn_running_mean.to(DType::Float32);
        Tensor bn_rv_f32    = bn_running_var.dtype()  == DType::Float32 ? bn_running_var  : bn_running_var.to(DType::Float32);
        Tensor out_f32 = fused_conv2d_bn_relu_kernel(
            input_f32, weight_f32, conv_bias_f32,
            bn_gamma_f32, bn_beta_f32, bn_rm_f32, bn_rv_f32,
            stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w,
            bn_momentum, bn_eps, training);
        return out_f32.to(orig_dtype);
    }

    // F036: the raw-pointer conv routines below walk input/weight as densely
    // packed NCHW/OIHW buffers. A non-contiguous input (permute/channels-last)
    // or a non-contiguous training-path weight would be read with wrong strides
    // (or OOB for a view whose logical extent exceeds its storage). Contiguify
    // both here, mirroring the inference path's contiguous-weight handling.
    const Tensor input_c = input.is_contiguous() ? input : input.contiguous();
    const Tensor weight_c = weight.is_contiguous() ? weight : weight.contiguous();

    auto in_shape = input_c.shape();
    auto w_shape = weight_c.shape();
    int64_t batch = in_shape[0];
    int64_t in_channels = in_shape[1];
    int64_t height = in_shape[2];
    int64_t width = in_shape[3];
    int64_t out_channels = w_shape[0];
    int64_t kernel_h = w_shape[2];
    int64_t kernel_w = w_shape[3];
    int64_t out_h = (height + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    int64_t out_w = (width  + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;

    Tensor output({batch, out_channels, out_h, out_w}, DType::Float32, input.device());

    if (training) {
        // Training path: compute batch stats inline
        Tensor rm = bn_running_mean;  // Will be updated in-place
        Tensor rv = bn_running_var;
        fused::conv_bn_relu_training(
            input_c.data<float>(), weight_c.data<float>(),
            conv_bias.numel() > 0 ? conv_bias.data<float>() : nullptr,
            bn_gamma.data<float>(), bn_beta.data<float>(),
            rm.data<float>(), rv.data<float>(),
            output.data<float>(),
            batch, in_channels, height, width,
            out_channels, kernel_h, kernel_w,
            stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w,
            out_h, out_w, bn_momentum, bn_eps);
    } else {
        // Inference path: fold BN into conv weights.
        //
        // fold_bn_params MUTATES the weight and bias buffers it is handed.
        // Tensor::contiguous() can return a view sharing storage with the
        // input — folding into that view would silently corrupt the caller's
        // weight tensor across subsequent calls (audit item A.1).  Materialise
        // a private buffer per call.
        Tensor weight_folded({out_channels, in_channels, kernel_h, kernel_w},
                             DType::Float32, input.device());
        {
            Tensor weight_src = weight_c;
            std::memcpy(weight_folded.data<float>(), weight_src.data<float>(),
                        static_cast<size_t>(out_channels) * in_channels *
                            kernel_h * kernel_w * sizeof(float));
        }
        Tensor bias_folded({out_channels}, DType::Float32, input.device());
        if (conv_bias.numel() > 0) {
            Tensor bias_src = conv_bias.contiguous();
            std::memcpy(bias_folded.data<float>(), bias_src.data<float>(),
                        static_cast<size_t>(out_channels) * sizeof(float));
        } else {
            std::memset(bias_folded.data<float>(), 0,
                        static_cast<size_t>(out_channels) * sizeof(float));
        }

        fused::fold_bn_params(
            weight_folded.data<float>(), bias_folded.data<float>(),
            bn_gamma.data<float>(), bn_beta.data<float>(),
            bn_running_mean.data<float>(), bn_running_var.data<float>(),
            out_channels, in_channels, kernel_h, kernel_w, bn_eps);

        fused::conv_bn_relu_folded(
            input_c.data<float>(), weight_folded.data<float>(), bias_folded.data<float>(),
            output.data<float>(),
            batch, in_channels, height, width,
            out_channels, kernel_h, kernel_w,
            stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w,
            out_h, out_w);
    }

    return output;
}

// =========================================================================
// FusedLayerNormBackward
// =========================================================================

// ---------------------------------------------------------------------------
// Implementation: templatized on T for loads/stores in the input precision.
// All reductions (ds, db, and the per-feature grad_weight/grad_bias partials)
// are accumulated in double regardless of T, mirroring the forward kernel's
// double-accumulation strategy (fused_ln_sum_f64 / fused_ln_sumsq_f64). This
// avoids reintroducing the float-accumulator cancellation that the forward was
// hardened against, keeping the forward/backward precision symmetric on long
// normalized dims. Called for Float32 and Float64.
// ---------------------------------------------------------------------------
template <typename T>
static void fused_layer_norm_backward_impl(
    const T* in_data,
    const T* go_data,
    const T* mean_data,
    const T* inv_std_data,
    const T* w_data,
    T*       gi_data,
    T*       gw_data,
    T*       gb_data,
    int64_t  batch_size,
    int64_t  norm_size)
{
    #pragma omp parallel if(batch_size > 16)
    {
        std::vector<double> local_gw(static_cast<size_t>(norm_size), 0.0);
        std::vector<double> local_gb(static_cast<size_t>(norm_size), 0.0);

        #pragma omp for
        for (int64_t b = 0; b < batch_size; ++b) {
            const T* in_b = in_data + b * norm_size;
            const T* go_b = go_data + b * norm_size;
            T*       gi_b = gi_data + b * norm_size;
            double m    = static_cast<double>(mean_data[b]);
            double rstd = static_cast<double>(inv_std_data[b]);

            // dot products needed to form grad_input (double accumulation)
            double ds = 0.0;  // sum(grad_output * weight * normalized)
            double db = 0.0;  // sum(grad_output * weight)
            for (int64_t j = 0; j < norm_size; ++j) {
                double normalized = (static_cast<double>(in_b[j]) - m) * rstd;
                double go_w = static_cast<double>(go_b[j]) *
                              static_cast<double>(w_data[j]);
                ds += go_w * normalized;
                db += go_w;
                local_gw[static_cast<size_t>(j)] +=
                    static_cast<double>(go_b[j]) * normalized;
                local_gb[static_cast<size_t>(j)] += static_cast<double>(go_b[j]);
            }

            double inv_n = 1.0 / static_cast<double>(norm_size);
            for (int64_t j = 0; j < norm_size; ++j) {
                double normalized = (static_cast<double>(in_b[j]) - m) * rstd;
                double gi = rstd * static_cast<double>(w_data[j]) *
                            (static_cast<double>(go_b[j]) -
                             inv_n * (db + normalized * ds));
                gi_b[j] = static_cast<T>(gi);
            }
        }

        #pragma omp critical
        {
            for (int64_t j = 0; j < norm_size; ++j) {
                gw_data[j] += static_cast<T>(local_gw[static_cast<size_t>(j)]);
                gb_data[j] += static_cast<T>(local_gb[static_cast<size_t>(j)]);
            }
        }
    }
}

auto fused_layer_norm_backward_kernel(
    const Tensor& grad_output, const Tensor& input,
    const std::vector<int64_t>& normalized_shape,
    const Tensor& mean, const Tensor& inv_std,
    const Tensor& weight) -> std::vector<Tensor> {

    Tensor input_cont = input.is_contiguous() ? input : input.contiguous();
    Tensor grad_cont  = grad_output.is_contiguous() ? grad_output : grad_output.contiguous();

    int64_t norm_size = 1;
    for (auto s : normalized_shape) {
        norm_size *= s;
    }
    int64_t batch_size = input_cont.numel() / norm_size;

    auto in_shape = input_cont.shape();
    DType dtype   = input_cont.dtype();

    Tensor grad_input = Tensor::empty_uninitialized(
        std::vector<int64_t>(in_shape.begin(), in_shape.end()),
        dtype, input_cont.device());

    // Float16 / BFloat16: widen-narrow shim — recurse with Float32 tensors,
    // then cast the output back to the original dtype.
    if (dtype == DType::Float16 || dtype == DType::BFloat16) {
        auto result = fused_layer_norm_backward_kernel(
            grad_output.to(DType::Float32), input.to(DType::Float32),
            normalized_shape,
            mean.to(DType::Float32), inv_std.to(DType::Float32),
            weight.to(DType::Float32));
        result[0] = result[0].to(dtype);  // grad_input
        result[1] = result[1].to(dtype);  // grad_weight
        result[2] = result[2].to(dtype);  // grad_bias
        return result;
    }

    // Allocate grad_weight and grad_bias with the input's dtype and
    // zero-initialise them.  This is the primary fix for audit P0 #2:
    // previously these were unconditionally DType::Float32.
    Tensor grad_weight(std::vector<int64_t>{norm_size}, dtype, input_cont.device());
    Tensor grad_bias  (std::vector<int64_t>{norm_size}, dtype, input_cont.device());

    if (dtype == DType::Float32) {
        std::memset(grad_weight.data<float>(), 0, static_cast<size_t>(norm_size) * sizeof(float));
        std::memset(grad_bias.data<float>(),   0, static_cast<size_t>(norm_size) * sizeof(float));

        Tensor mean_cont     = mean.is_contiguous()    ? mean    : mean.contiguous();
        Tensor inv_std_cont  = inv_std.is_contiguous() ? inv_std : inv_std.contiguous();
        Tensor weight_cont   = weight.is_contiguous()  ? weight  : weight.contiguous();

        fused_layer_norm_backward_impl<float>(
            input_cont.data<float>(),
            grad_cont.data<float>(),
            mean_cont.data<float>(),
            inv_std_cont.data<float>(),
            weight_cont.data<float>(),
            grad_input.data<float>(),
            grad_weight.data<float>(),
            grad_bias.data<float>(),
            batch_size, norm_size);

    } else if (dtype == DType::Float64) {
        std::memset(grad_weight.data<double>(), 0, static_cast<size_t>(norm_size) * sizeof(double));
        std::memset(grad_bias.data<double>(),   0, static_cast<size_t>(norm_size) * sizeof(double));

        // The saved mean and inv_std may have been stored as Float32 by the
        // CPU forward path (see normalization.cpp, "CPU backward converts to
        // Float32 anyway" comment).  Widen them to Float64 before dispatch so
        // that data<double>() is safe to call.
        Tensor mean_f64     = (mean.dtype()    == DType::Float64) ? mean    : mean.to(DType::Float64);
        Tensor inv_std_f64  = (inv_std.dtype() == DType::Float64) ? inv_std : inv_std.to(DType::Float64);
        Tensor weight_f64   = (weight.dtype()  == DType::Float64) ? weight  : weight.to(DType::Float64);

        Tensor mean_cont    = mean_f64.is_contiguous()    ? mean_f64    : mean_f64.contiguous();
        Tensor istd_cont    = inv_std_f64.is_contiguous() ? inv_std_f64 : inv_std_f64.contiguous();
        Tensor weight_cont  = weight_f64.is_contiguous()  ? weight_f64  : weight_f64.contiguous();

        fused_layer_norm_backward_impl<double>(
            input_cont.data<double>(),
            grad_cont.data<double>(),
            mean_cont.data<double>(),
            istd_cont.data<double>(),
            weight_cont.data<double>(),
            grad_input.data<double>(),
            grad_weight.data<double>(),
            grad_bias.data<double>(),
            batch_size, norm_size);

    } else {
        throw std::runtime_error(
            "fused_layer_norm_backward: unsupported dtype " +
            std::string(dtype_name(dtype)));
    }

    return std::vector<Tensor>{grad_input, grad_weight, grad_bias};
}

} // namespace cpu
} // namespace tenzor
