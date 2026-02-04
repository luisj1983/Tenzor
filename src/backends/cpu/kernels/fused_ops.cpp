#include "tenzor/core/tensor.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/reduction.hpp"
#include <cmath>
#include <stdexcept>
#include <algorithm>

namespace tenzor {
namespace cpu {

// Forward declaration for conv2d_forward_kernel (used by fused conv+activation kernels)
auto conv2d_forward_kernel(const Tensor& input, const Tensor& weight, const Tensor* bias,
                           int64_t stride, int64_t padding, int64_t dilation, int64_t groups) -> Tensor;

// Float16 Arithmetic Helper Functions for LayerNorm
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
    // Flatten input to 2D if needed
    auto input_shape = input.shape();
    int64_t batch_size = 1;
    for (size_t i = 0; i < input_shape.size() - 1; ++i) {
        batch_size *= input_shape[i];
    }
    int64_t in_features = input_shape[input_shape.size() - 1];
    int64_t out_features = weight.shape()[0];

    // Perform matmul: (batch_size, in_features) @ (out_features, in_features).T
    // = (batch_size, in_features) @ (in_features, out_features)
    // = (batch_size, out_features)

    // Reshape input to (batch_size, in_features)
    Tensor input_2d = input.reshape({batch_size, in_features});

    // Transpose weight from (out_features, in_features) to (in_features, out_features)
    Tensor weight_t = weight.transpose(0, 1);

    // Matrix multiplication
    Tensor output = matmul(input_2d, weight_t);

    // Add bias if provided and apply ReLU in single pass
    if (input.dtype() == DType::Float32) {
        float* out_data = output.data<float>();
        const float* bias_data = bias ? bias->data<float>() : nullptr;
        size_t n = static_cast<size_t>(output.numel());

        for (size_t i = 0; i < static_cast<size_t>(batch_size); ++i) {
            for (size_t j = 0; j < static_cast<size_t>(out_features); ++j) {
                size_t idx = i * out_features + j;
                float val = out_data[idx];
                if (bias_data) {
                    val += bias_data[j];
                }
                // ReLU
                out_data[idx] = std::max(0.0f, val);
            }
        }
    } else if (input.dtype() == DType::Float64) {
        double* out_data = output.data<double>();
        const double* bias_data = bias ? bias->data<double>() : nullptr;

        for (size_t i = 0; i < static_cast<size_t>(batch_size); ++i) {
            for (size_t j = 0; j < static_cast<size_t>(out_features); ++j) {
                size_t idx = i * out_features + j;
                double val = out_data[idx];
                if (bias_data) {
                    val += bias_data[j];
                }
                // ReLU
                out_data[idx] = std::max(0.0, val);
            }
        }
    } else if (input.dtype() == DType::Float16) {
        // Float16: convert to Float32 for computation, convert back
        Tensor output_f32 = output.to(DType::Float32);
        Tensor bias_f32 = bias ? bias->to(DType::Float32) : Tensor();

        float* out_data = output_f32.data<float>();
        const float* bias_data = bias ? bias_f32.data<float>() : nullptr;

        for (size_t i = 0; i < static_cast<size_t>(batch_size); ++i) {
            for (size_t j = 0; j < static_cast<size_t>(out_features); ++j) {
                size_t idx = i * out_features + j;
                float val = out_data[idx];
                if (bias_data) {
                    val += bias_data[j];
                }
                // ReLU
                out_data[idx] = std::max(0.0f, val);
            }
        }

        output = output_f32.to(DType::Float16);
    }

    // Reshape back to original batch dimensions
    std::vector<int64_t> output_shape(input_shape.begin(), input_shape.end() - 1);
    output_shape.push_back(out_features);
    return output.reshape(output_shape);
}

/**
 * @brief Fused conv2d + ReLU kernel (CPU implementation)
 *
 * Delegates to conv2d_forward_kernel (im2col+GEMM) then applies ReLU in-place.
 * Supports dilation and groups for parity with CUDA backend.
 */
auto fused_conv2d_relu_kernel(
    const Tensor& input,
    const Tensor& weight,
    const Tensor* bias,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    int64_t groups
) -> Tensor {
    Tensor result = conv2d_forward_kernel(input, weight, bias, stride, padding, dilation, groups);
    int64_t n = result.numel();
    if (result.dtype() == DType::Float32) {
        float* data = result.data<float>();
        #pragma omp parallel for if(n > 65536)
        for (int64_t i = 0; i < n; ++i) {
            data[i] = std::max(0.0f, data[i]);
        }
    } else if (result.dtype() == DType::Float64) {
        double* data = result.data<double>();
        #pragma omp parallel for if(n > 65536)
        for (int64_t i = 0; i < n; ++i) {
            data[i] = std::max(0.0, data[i]);
        }
    }
    return result;
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
        const float* mean_data = running_mean.data<float>();
        const float* var_data = running_var.data<float>();
        const float* gamma_data = weight.data<float>();
        const float* beta_data = bias.data<float>();
        float* out_data = output.data<float>();

        // Fused batchnorm + ReLU
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
                    // ReLU
                    out_data[idx] = std::max(0.0f, scaled);
                }
            }
        }
    } else if (input.dtype() == DType::Float64) {
        const double* in_data = input.data<double>();
        const double* mean_data = running_mean.data<double>();
        const double* var_data = running_var.data<double>();
        const double* gamma_data = weight.data<double>();
        const double* beta_data = bias.data<double>();
        double* out_data = output.data<double>();

        // Fused batchnorm + ReLU
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
                    // ReLU
                    out_data[idx] = std::max(0.0, scaled);
                }
            }
        }
    } else if (input.dtype() == DType::Float16) {
        const Float16* in_data = input.data<Float16>();
        const Float16* mean_data = running_mean.data<Float16>();
        const Float16* var_data = running_var.data<Float16>();
        const Float16* gamma_data = weight.data<Float16>();
        const Float16* beta_data = bias.data<Float16>();
        Float16* out_data = output.data<Float16>();

        // Fused batchnorm + ReLU (compute in float for numerical stability)
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
                    // ReLU
                    out_data[idx] = Float16(std::max(0.0f, scaled));
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
    bool compute_grad
) -> std::vector<Tensor> {
    int64_t batch_size = logits.shape()[0];
    int64_t num_classes = logits.shape()[1];

    Tensor losses = zeros({batch_size}, logits.dtype(), logits.device());
    Tensor grad_logits;
    if (compute_grad) {
        std::vector<int64_t> logits_shape(logits.shape().begin(), logits.shape().end());
        grad_logits = zeros(logits_shape, logits.dtype(), logits.device());
    }

    if (logits.dtype() == DType::Float32) {
        const float* logits_data = logits.data<float>();
        const int64_t* targets_data = targets.data<int64_t>();
        float* losses_data = losses.data<float>();
        float* grad_data = compute_grad ? grad_logits.data<float>() : nullptr;

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

            // Compute loss for target class
            int64_t target = targets_data[i];
            if (target < 0 || target >= num_classes) {
                throw std::runtime_error(
                    "fused_softmax_cross_entropy: target index out of range"
                );
            }
            losses_data[i] = log_sum_exp - row[target];

            // Compute gradient: softmax(logits) - one_hot(target)
            if (compute_grad) {
                float* grad_row = grad_data + i * num_classes;
                float inv_sum_exp = 1.0f / sum_exp;
                for (int64_t j = 0; j < num_classes; ++j) {
                    grad_row[j] = std::exp(row[j] - max_logit) * inv_sum_exp;
                }
                grad_row[target] -= 1.0f;
                // Normalize by batch size for mean reduction
                float scale = 1.0f / static_cast<float>(batch_size);
                for (int64_t j = 0; j < num_classes; ++j) {
                    grad_row[j] *= scale;
                }
            }
        }
    } else {
        throw std::runtime_error("fused_softmax_cross_entropy: Only Float32 supported");
    }

    // Apply mean reduction to loss (matching CUDA behavior)
    Tensor loss = tenzor::mean(losses);

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
        for (size_t i = 0; i < n; ++i) {
            data[i] = std::max(0.0f, data[i]);
        }
    } else if (result.dtype() == DType::Float64) {
        double* data = result.data<double>();
        size_t n = static_cast<size_t>(result.numel());
        for (size_t i = 0; i < n; ++i) {
            data[i] = std::max(0.0, data[i]);
        }
    } else if (result.dtype() == DType::Float16) {
        // Float16: convert to Float32 for computation, convert back
        Tensor result_f32 = result.to(DType::Float32);
        float* data = result_f32.data<float>();
        size_t n = static_cast<size_t>(result.numel());
        for (size_t i = 0; i < n; ++i) {
            data[i] = std::max(0.0f, data[i]);
        }
        result = result_f32.to(DType::Float16);
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
    std::vector<int64_t> shape_vec(input.shape().begin(), input.shape().end());
    Tensor result = zeros(shape_vec, input.dtype(), input.device());

    constexpr float sqrt_2_over_pi = 0.7978845608f;  // sqrt(2/pi)
    constexpr float coeff = 0.044715f;

    if (input.dtype() == DType::Float32) {
        const float* in_data = input.data<float>();
        float* out_data = result.data<float>();
        size_t n = static_cast<size_t>(input.numel());

        for (size_t i = 0; i < n; ++i) {
            float x = in_data[i];
            float x_cubed = x * x * x;
            float inner = sqrt_2_over_pi * (x + coeff * x_cubed);
            float tanh_val = std::tanh(inner);
            out_data[i] = 0.5f * x * (1.0f + tanh_val);
        }
    } else if (input.dtype() == DType::Float64) {
        const double* in_data = input.data<double>();
        double* out_data = result.data<double>();
        size_t n = static_cast<size_t>(input.numel());

        for (size_t i = 0; i < n; ++i) {
            double x = in_data[i];
            double x_cubed = x * x * x;
            double inner = sqrt_2_over_pi * (x + coeff * x_cubed);
            double tanh_val = std::tanh(inner);
            out_data[i] = 0.5 * x * (1.0 + tanh_val);
        }
    } else if (input.dtype() == DType::Float16) {
        const Float16* in_data = input.data<Float16>();
        Float16* out_data = result.data<Float16>();
        size_t n = static_cast<size_t>(input.numel());

        for (size_t i = 0; i < n; ++i) {
            float x = static_cast<float>(in_data[i]);
            float x_cubed = x * x * x;
            float inner = sqrt_2_over_pi * (x + coeff * x_cubed);
            float tanh_val = std::tanh(inner);
            out_data[i] = Float16(0.5f * x * (1.0f + tanh_val));
        }
    } else {
        throw std::runtime_error("fused_gelu: Only Float32/Float64/Float16 supported");
    }

    return result;
}

/**
 * @brief Fused layer norm kernel (CPU implementation)
 *
 * Single-pass computation of mean, variance, and normalization.
 * Returns {output, mean, inv_std} to match CUDA backend for backward pass support.
 */
auto fused_layer_norm_kernel(
    const Tensor& input,
    const std::vector<int64_t>& normalized_shape,
    const Tensor& weight,
    const Tensor& bias,
    float eps
) -> std::tuple<Tensor, Tensor, Tensor> {
    // Calculate batch size and normalization size
    int64_t norm_size = 1;
    for (auto dim : normalized_shape) {
        norm_size *= dim;
    }

    int64_t batch_size = input.numel() / norm_size;

    std::vector<int64_t> shape_vec(input.shape().begin(), input.shape().end());
    Tensor result = zeros(shape_vec, input.dtype(), input.device());
    // Store per-batch-element mean and inv_std for backward pass
    Tensor mean_out({batch_size}, input.dtype(), input.device());
    Tensor inv_std_out({batch_size}, input.dtype(), input.device());

    if (input.dtype() == DType::Float32) {
        const float* in_data = input.data<float>();
        const float* weight_data = weight.data<float>();
        const float* bias_data = bias.data<float>();
        float* out_data = result.data<float>();
        float* mean_data = mean_out.data<float>();
        float* inv_std_data = inv_std_out.data<float>();

        for (int64_t b = 0; b < batch_size; ++b) {
            const float* batch_in = in_data + b * norm_size;
            float* batch_out = out_data + b * norm_size;

            float mean = 0.0f;
            for (int64_t i = 0; i < norm_size; ++i) {
                mean += batch_in[i];
            }
            mean /= norm_size;

            float variance = 0.0f;
            for (int64_t i = 0; i < norm_size; ++i) {
                float diff = batch_in[i] - mean;
                variance += diff * diff;
            }
            variance /= norm_size;

            float inv_std = 1.0f / std::sqrt(variance + eps);
            mean_data[b] = mean;
            inv_std_data[b] = inv_std;

            for (int64_t i = 0; i < norm_size; ++i) {
                float normalized = (batch_in[i] - mean) * inv_std;
                batch_out[i] = normalized * weight_data[i] + bias_data[i];
            }
        }
    } else if (input.dtype() == DType::Float64) {
        const double* in_data = input.data<double>();
        const double* weight_data = weight.data<double>();
        const double* bias_data = bias.data<double>();
        double* out_data = result.data<double>();
        double* mean_data = mean_out.data<double>();
        double* inv_std_data = inv_std_out.data<double>();

        for (int64_t b = 0; b < batch_size; ++b) {
            const double* batch_in = in_data + b * norm_size;
            double* batch_out = out_data + b * norm_size;

            double mean = 0.0;
            for (int64_t i = 0; i < norm_size; ++i) {
                mean += batch_in[i];
            }
            mean /= norm_size;

            double variance = 0.0;
            for (int64_t i = 0; i < norm_size; ++i) {
                double diff = batch_in[i] - mean;
                variance += diff * diff;
            }
            variance /= norm_size;

            double inv_std = 1.0 / std::sqrt(variance + eps);
            mean_data[b] = mean;
            inv_std_data[b] = inv_std;

            for (int64_t i = 0; i < norm_size; ++i) {
                double normalized = (batch_in[i] - mean) * inv_std;
                batch_out[i] = normalized * weight_data[i] + bias_data[i];
            }
        }
    } else if (input.dtype() == DType::Float16) {
        const Float16* in_data = input.data<Float16>();
        const Float16* weight_data = weight.data<Float16>();
        const Float16* bias_data = bias.data<Float16>();
        Float16* out_data = result.data<Float16>();
        Float16* mean_data = mean_out.data<Float16>();
        Float16* inv_std_data = inv_std_out.data<Float16>();

        for (int64_t b = 0; b < batch_size; ++b) {
            const Float16* batch_in = in_data + b * norm_size;
            Float16* batch_out = out_data + b * norm_size;

            float mean = 0.0f;
            for (int64_t i = 0; i < norm_size; ++i) {
                mean += static_cast<float>(batch_in[i]);
            }
            mean /= static_cast<float>(norm_size);

            float variance = 0.0f;
            for (int64_t i = 0; i < norm_size; ++i) {
                float diff = static_cast<float>(batch_in[i]) - mean;
                variance += diff * diff;
            }
            variance /= static_cast<float>(norm_size);

            float inv_std = 1.0f / std::sqrt(variance + eps);
            mean_data[b] = Float16(mean);
            inv_std_data[b] = Float16(inv_std);

            for (int64_t i = 0; i < norm_size; ++i) {
                float normalized = (static_cast<float>(batch_in[i]) - mean) * inv_std;
                batch_out[i] = Float16(normalized * static_cast<float>(weight_data[i]) + static_cast<float>(bias_data[i]));
            }
        }
    } else {
        throw std::runtime_error("fused_layer_norm: Only Float32/Float64/Float16 supported");
    }

    return {result, mean_out, inv_std_out};
}

// =========================================================================
// RMSNorm Operations
// =========================================================================

auto fused_rms_norm_kernel(const Tensor& input, const Tensor& weight, float eps)
    -> std::tuple<Tensor, Tensor> {
    // RMSNorm: output = input * weight / sqrt(mean(input^2) + eps)
    const auto& shape = input.shape();
    const int64_t ndim = input.ndim();

    // Normalized shape is the last dimension (weight.shape)
    const int64_t norm_size = weight.numel();
    int64_t batch_size = input.numel() / norm_size;

    Tensor output(std::vector<int64_t>(shape.begin(), shape.end()),
                  input.dtype(), input.device());
    // rrms stores reciprocal RMS for each batch element
    Tensor rrms({batch_size}, input.dtype(), input.device());

    if (input.dtype() == DType::Float32) {
        const float* in_data = input.data<float>();
        const float* w_data = weight.data<float>();
        float* out_data = output.data<float>();
        float* rrms_data = rrms.data<float>();

        #pragma omp parallel for if(batch_size > 64)
        for (int64_t b = 0; b < batch_size; ++b) {
            const float* x = in_data + b * norm_size;
            float* y = out_data + b * norm_size;

            // Compute mean(x^2)
            float sum_sq = 0.0f;
            for (int64_t i = 0; i < norm_size; ++i) {
                sum_sq += x[i] * x[i];
            }
            float mean_sq = sum_sq / static_cast<float>(norm_size);
            float inv_rms = 1.0f / std::sqrt(mean_sq + eps);
            rrms_data[b] = inv_rms;

            // Apply normalization and weight
            for (int64_t i = 0; i < norm_size; ++i) {
                y[i] = x[i] * inv_rms * w_data[i];
            }
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
    } else {
        throw std::runtime_error("fused_rms_norm: only Float32/Float64 supported");
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

        for (int64_t b = 0; b < batch_size; ++b) {
            const float* go_b = go + b * norm_size;
            const float* x_b = x + b * norm_size;
            float* gi_b = gi + b * norm_size;
            float inv_rms = r[b];

            // grad_weight accumulation
            for (int64_t i = 0; i < norm_size; ++i) {
                gw[i] += go_b[i] * x_b[i] * inv_rms;
            }

            // grad_input: d/dx of (x * rrms * w) with rrms depending on x
            float dot = 0.0f;
            for (int64_t i = 0; i < norm_size; ++i) {
                dot += go_b[i] * w[i] * x_b[i];
            }
            float coeff = dot * inv_rms * inv_rms / static_cast<float>(norm_size);
            for (int64_t i = 0; i < norm_size; ++i) {
                gi_b[i] = (go_b[i] * w[i] - x_b[i] * coeff) * inv_rms;
            }
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
    } else {
        throw std::runtime_error("rms_norm_backward: only Float32/Float64 supported");
    }

    return {grad_input, grad_weight};
}

// =========================================================================
// Fused Attention (Q*K^T*V with scaling)
// =========================================================================

auto fused_attention_kernel(const Tensor& Q, const Tensor& K, const Tensor& V,
                            float scale) -> Tensor {
    // Q, K, V: (batch_heads, seq_len, head_dim) for 3D
    //          (batch, num_heads, seq_len, head_dim) for 4D
    const auto& q_shape = Q.shape();
    const int64_t ndim = Q.ndim();

    int64_t batch_heads, seq_len_q, head_dim, seq_len_k;

    if (ndim == 3) {
        batch_heads = q_shape[0];
        seq_len_q = q_shape[1];
        head_dim = q_shape[2];
        seq_len_k = K.shape()[1];
    } else if (ndim == 4) {
        batch_heads = q_shape[0] * q_shape[1];
        seq_len_q = q_shape[2];
        head_dim = q_shape[3];
        seq_len_k = K.shape()[2];
    } else {
        throw std::runtime_error("fused_attention: Q must be 3D or 4D");
    }

    Tensor output(std::vector<int64_t>(q_shape.begin(), q_shape.end()),
                  Q.dtype(), Q.device());

    if (Q.dtype() == DType::Float32) {
        const float* q_data = Q.data<float>();
        const float* k_data = K.data<float>();
        const float* v_data = V.data<float>();
        float* out_data = output.data<float>();

        int64_t q_stride = seq_len_q * head_dim;
        int64_t k_stride = seq_len_k * head_dim;
        int64_t v_stride = seq_len_k * head_dim;

        #pragma omp parallel for if(batch_heads > 4)
        for (int64_t bh = 0; bh < batch_heads; ++bh) {
            const float* q = q_data + bh * q_stride;
            const float* k = k_data + bh * k_stride;
            const float* v = v_data + bh * v_stride;
            float* o = out_data + bh * q_stride;

            // Compute attention scores: Q * K^T * scale
            std::vector<float> scores(seq_len_q * seq_len_k);
            for (int64_t i = 0; i < seq_len_q; ++i) {
                for (int64_t j = 0; j < seq_len_k; ++j) {
                    float dot = 0.0f;
                    for (int64_t d = 0; d < head_dim; ++d) {
                        dot += q[i * head_dim + d] * k[j * head_dim + d];
                    }
                    scores[i * seq_len_k + j] = dot * scale;
                }
            }

            // Softmax along last dimension
            for (int64_t i = 0; i < seq_len_q; ++i) {
                float max_val = scores[i * seq_len_k];
                for (int64_t j = 1; j < seq_len_k; ++j) {
                    max_val = std::max(max_val, scores[i * seq_len_k + j]);
                }
                float sum = 0.0f;
                for (int64_t j = 0; j < seq_len_k; ++j) {
                    scores[i * seq_len_k + j] = std::exp(scores[i * seq_len_k + j] - max_val);
                    sum += scores[i * seq_len_k + j];
                }
                float inv_sum = 1.0f / sum;
                for (int64_t j = 0; j < seq_len_k; ++j) {
                    scores[i * seq_len_k + j] *= inv_sum;
                }
            }

            // Compute output: scores * V
            for (int64_t i = 0; i < seq_len_q; ++i) {
                for (int64_t d = 0; d < head_dim; ++d) {
                    float val = 0.0f;
                    for (int64_t j = 0; j < seq_len_k; ++j) {
                        val += scores[i * seq_len_k + j] * v[j * head_dim + d];
                    }
                    o[i * head_dim + d] = val;
                }
            }
        }
    } else {
        throw std::runtime_error("fused_attention: only Float32 supported");
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

auto fused_conv2d_sigmoid_kernel(const Tensor& input, const Tensor& weight, const Tensor* bias,
                                  int64_t stride, int64_t padding, int64_t dilation, int64_t groups) -> Tensor {
    Tensor result = conv2d_forward_kernel(input, weight, bias, stride, padding, dilation, groups);
    int64_t n = result.numel();
    if (result.dtype() == DType::Float32) {
        apply_sigmoid_inplace(result.data<float>(), n);
    } else if (result.dtype() == DType::Float64) {
        apply_sigmoid_inplace(result.data<double>(), n);
    }
    return result;
}

auto fused_conv2d_tanh_kernel(const Tensor& input, const Tensor& weight, const Tensor* bias,
                               int64_t stride, int64_t padding, int64_t dilation, int64_t groups) -> Tensor {
    Tensor result = conv2d_forward_kernel(input, weight, bias, stride, padding, dilation, groups);
    int64_t n = result.numel();
    if (result.dtype() == DType::Float32) {
        apply_tanh_inplace(result.data<float>(), n);
    } else if (result.dtype() == DType::Float64) {
        apply_tanh_inplace(result.data<double>(), n);
    }
    return result;
}

auto fused_conv2d_swish_kernel(const Tensor& input, const Tensor& weight, const Tensor* bias,
                                int64_t stride, int64_t padding, int64_t dilation, int64_t groups) -> Tensor {
    Tensor result = conv2d_forward_kernel(input, weight, bias, stride, padding, dilation, groups);
    int64_t n = result.numel();
    if (result.dtype() == DType::Float32) {
        apply_swish_inplace(result.data<float>(), n);
    } else if (result.dtype() == DType::Float64) {
        apply_swish_inplace(result.data<double>(), n);
    }
    return result;
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

    // Forward with affine
    Tensor output = batchnorm2d_forward_affine_kernel(input, batch_mean, batch_var, gamma, beta, epsilon);

    // Update running stats
    batchnorm2d_update_running_stats_kernel(running_mean, running_var, batch_mean, batch_var, momentum);

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
    }

    return {output, running_mean, running_var, batch_mean, saved_inv_var};
}

// =========================================================================
// Fused Optimizer Steps
// =========================================================================

auto fused_sgd_step_kernel(Tensor& param, const Tensor& grad, Tensor* momentum_buffer,
                           float lr, float momentum, float weight_decay,
                           float dampening, bool nesterov) -> void {
    const int64_t n = param.numel();

    if (param.dtype() == DType::Float32) {
        float* p = param.data<float>();
        const float* g = grad.data<float>();

        #pragma omp parallel for if(n > 65536)
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

        #pragma omp parallel for if(n > 65536)
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

        #pragma omp parallel for if(n > 32768)
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

        #pragma omp parallel for if(n > 32768)
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
    }
}

auto fused_adam_atan2_step_kernel(Tensor& param, const Tensor& grad,
                                  Tensor& exp_avg, Tensor& exp_avg_sq,
                                  Tensor* max_exp_avg_sq,
                                  float lr, float beta1, float beta2,
                                  float eps, float weight_decay,
                                  int64_t step, bool amsgrad) -> void {
    const int64_t n = param.numel();

    float bias_correction1 = 1.0f - std::pow(beta1, static_cast<float>(step));
    float bias_correction2 = 1.0f - std::pow(beta2, static_cast<float>(step));
    float step_size = lr / bias_correction1;

    if (param.dtype() == DType::Float32) {
        float* p = param.data<float>();
        const float* g = grad.data<float>();
        float* m = exp_avg.data<float>();
        float* v = exp_avg_sq.data<float>();
        float* v_max = (amsgrad && max_exp_avg_sq) ? max_exp_avg_sq->data<float>() : nullptr;

        #pragma omp parallel for if(n > 32768)
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
    }
}

auto fused_rmsprop_step_kernel(Tensor& param, const Tensor& grad,
                                Tensor& square_avg, Tensor* grad_avg,
                                Tensor* momentum_buffer,
                                float lr, float alpha, float eps,
                                float weight_decay, float momentum,
                                bool centered) -> void {
    const int64_t n = param.numel();

    if (param.dtype() == DType::Float32) {
        float* p = param.data<float>();
        const float* g = grad.data<float>();
        float* sq = square_avg.data<float>();
        float* ga = (centered && grad_avg) ? grad_avg->data<float>() : nullptr;
        float* buf = (momentum > 0.0f && momentum_buffer) ? momentum_buffer->data<float>() : nullptr;

        #pragma omp parallel for if(n > 65536)
        for (int64_t i = 0; i < n; ++i) {
            float grad_val = g[i];
            if (weight_decay != 0.0f) {
                grad_val += weight_decay * p[i];
            }

            sq[i] = alpha * sq[i] + (1.0f - alpha) * grad_val * grad_val;

            float avg;
            if (centered && ga) {
                ga[i] = alpha * ga[i] + (1.0f - alpha) * grad_val;
                avg = std::sqrt(sq[i] - ga[i] * ga[i] + eps);
            } else {
                avg = std::sqrt(sq[i] + eps);
            }

            if (buf) {
                buf[i] = momentum * buf[i] + grad_val / avg;
                p[i] -= lr * buf[i];
            } else {
                p[i] -= lr * grad_val / avg;
            }
        }
    }
}

auto fused_adadelta_step_kernel(Tensor& param, const Tensor& grad,
                                 Tensor& square_avg, Tensor& acc_delta,
                                 float rho, float eps, float lr,
                                 float weight_decay) -> void {
    const int64_t n = param.numel();

    if (param.dtype() == DType::Float32) {
        float* p = param.data<float>();
        const float* g = grad.data<float>();
        float* sq = square_avg.data<float>();
        float* ad = acc_delta.data<float>();

        #pragma omp parallel for if(n > 65536)
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
    }
}

auto fused_adagrad_step_kernel(Tensor& param, const Tensor& grad,
                                Tensor& sum_sq, float lr, float lr_decay,
                                float eps, float weight_decay,
                                int64_t step) -> void {
    const int64_t n = param.numel();
    float clr = lr / (1.0f + static_cast<float>(step - 1) * lr_decay);

    if (param.dtype() == DType::Float32) {
        float* p = param.data<float>();
        const float* g = grad.data<float>();
        float* ss = sum_sq.data<float>();

        #pragma omp parallel for if(n > 65536)
        for (int64_t i = 0; i < n; ++i) {
            float grad_val = g[i];
            if (weight_decay != 0.0f) {
                grad_val += weight_decay * p[i];
            }
            ss[i] += grad_val * grad_val;
            p[i] -= clr * grad_val / (std::sqrt(ss[i]) + eps);
        }
    }
}

} // namespace cpu
} // namespace tenzor
