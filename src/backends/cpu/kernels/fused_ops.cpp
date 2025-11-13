#include "tenzor/core/tensor.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/reduction.hpp"
#include <cmath>
#include <stdexcept>
#include <algorithm>

namespace tenzor {
namespace cpu {

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
    }

    // Reshape back to original batch dimensions
    std::vector<int64_t> output_shape(input_shape.begin(), input_shape.end() - 1);
    output_shape.push_back(out_features);
    return output.reshape(output_shape);
}

/**
 * @brief Fused conv2d + ReLU kernel (CPU implementation)
 *
 * Optimized implementation that applies ReLU during convolution output
 * to reduce memory bandwidth requirements.
 */
auto fused_conv2d_relu_kernel(
    const Tensor& input,
    const Tensor& weight,
    const Tensor* bias,
    int64_t stride,
    int64_t padding
) -> Tensor {
    // Extract dimensions
    auto input_shape = input.shape();
    auto weight_shape = weight.shape();

    int64_t batch = input_shape[0];
    int64_t in_channels = input_shape[1];
    int64_t in_h = input_shape[2];
    int64_t in_w = input_shape[3];

    int64_t out_channels = weight_shape[0];
    int64_t kernel_h = weight_shape[2];
    int64_t kernel_w = weight_shape[3];

    // Calculate output dimensions
    int64_t out_h = (in_h + 2 * padding - kernel_h) / stride + 1;
    int64_t out_w = (in_w + 2 * padding - kernel_w) / stride + 1;

    // Create output tensor
    std::vector<int64_t> output_shape = {batch, out_channels, out_h, out_w};
    Tensor output = zeros(output_shape, input.dtype(), input.device());

    if (input.dtype() == DType::Float32) {
        const float* in_data = input.data<float>();
        const float* w_data = weight.data<float>();
        const float* b_data = bias ? bias->data<float>() : nullptr;
        float* out_data = output.data<float>();

        // Naive convolution with fused ReLU
        // Parallelize over batch and output channels
        #pragma omp parallel for collapse(2)
        for (int64_t n = 0; n < batch; ++n) {
            for (int64_t oc = 0; oc < out_channels; ++oc) {
                // Process each output spatial position
                for (int64_t oh = 0; oh < out_h; ++oh) {
                    for (int64_t ow = 0; ow < out_w; ++ow) {
                        float sum = 0.0f;

                        // Convolution computation
                        for (int64_t ic = 0; ic < in_channels; ++ic) {
                            for (int64_t kh = 0; kh < kernel_h; ++kh) {
                                for (int64_t kw = 0; kw < kernel_w; ++kw) {
                                    int64_t ih = oh * stride - padding + kh;
                                    int64_t iw = ow * stride - padding + kw;

                                    // Check bounds
                                    if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                                        int64_t in_idx = n * (in_channels * in_h * in_w) +
                                                        ic * (in_h * in_w) +
                                                        ih * in_w + iw;
                                        int64_t w_idx = oc * (in_channels * kernel_h * kernel_w) +
                                                       ic * (kernel_h * kernel_w) +
                                                       kh * kernel_w + kw;
                                        sum += in_data[in_idx] * w_data[w_idx];
                                    }
                                }
                            }
                        }

                        // Add bias if present
                        if (b_data) {
                            sum += b_data[oc];
                        }

                        // Apply ReLU immediately (fused operation)
                        int64_t out_idx = n * (out_channels * out_h * out_w) +
                                         oc * (out_h * out_w) +
                                         oh * out_w + ow;
                        out_data[out_idx] = std::max(0.0f, sum);
                    }
                }
            }
        }
    } else if (input.dtype() == DType::Float64) {
        const double* in_data = input.data<double>();
        const double* w_data = weight.data<double>();
        const double* b_data = bias ? bias->data<double>() : nullptr;
        double* out_data = output.data<double>();

        #pragma omp parallel for collapse(2)
        for (int64_t n = 0; n < batch; ++n) {
            for (int64_t oc = 0; oc < out_channels; ++oc) {
                for (int64_t oh = 0; oh < out_h; ++oh) {
                    for (int64_t ow = 0; ow < out_w; ++ow) {
                        double sum = 0.0;

                        for (int64_t ic = 0; ic < in_channels; ++ic) {
                            for (int64_t kh = 0; kh < kernel_h; ++kh) {
                                for (int64_t kw = 0; kw < kernel_w; ++kw) {
                                    int64_t ih = oh * stride - padding + kh;
                                    int64_t iw = ow * stride - padding + kw;

                                    if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                                        int64_t in_idx = n * (in_channels * in_h * in_w) +
                                                        ic * (in_h * in_w) +
                                                        ih * in_w + iw;
                                        int64_t w_idx = oc * (in_channels * kernel_h * kernel_w) +
                                                       ic * (kernel_h * kernel_w) +
                                                       kh * kernel_w + kw;
                                        sum += in_data[in_idx] * w_data[w_idx];
                                    }
                                }
                            }
                        }

                        if (b_data) {
                            sum += b_data[oc];
                        }

                        int64_t out_idx = n * (out_channels * out_h * out_w) +
                                         oc * (out_h * out_w) +
                                         oh * out_w + ow;
                        out_data[out_idx] = std::max(0.0, sum);
                    }
                }
            }
        }
    } else {
        throw std::runtime_error("fused_conv2d_relu: Unsupported dtype");
    }

    return output;
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

    if (input.dtype() == DType::Float32) {
        const float* in_data = input.data<float>();
        const float* mean_data = running_mean.data<float>();
        const float* var_data = running_var.data<float>();
        const float* gamma_data = weight.data<float>();
        const float* beta_data = bias.data<float>();
        float* out_data = output.data<float>();

        int64_t batch_size = input.shape()[0];
        int64_t spatial_size = 1;
        for (size_t i = 2; i < input.shape().size(); ++i) {
            spatial_size *= input.shape()[i];
        }

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
    } else {
        throw std::runtime_error("fused_batchnorm_relu: Only Float32 supported");
    }

    return output;
}

/**
 * @brief Fused softmax + cross entropy kernel (CPU implementation)
 *
 * Uses log-sum-exp trick for numerical stability.
 */
auto fused_softmax_cross_entropy_kernel(
    const Tensor& logits,
    const Tensor& targets,
    const std::string& reduction
) -> Tensor {
    int64_t batch_size = logits.shape()[0];
    int64_t num_classes = logits.shape()[1];

    Tensor losses = zeros({batch_size}, logits.dtype(), logits.device());

    if (logits.dtype() == DType::Float32) {
        const float* logits_data = logits.data<float>();
        const int64_t* targets_data = targets.data<int64_t>();
        float* losses_data = losses.data<float>();

        for (int64_t i = 0; i < batch_size; ++i) {
            const float* row = logits_data + i * num_classes;

            // Find max for numerical stability
            float max_logit = row[0];
            for (int64_t j = 1; j < num_classes; ++j) {
                max_logit = std::max(max_logit, row[j]);
            }

            // Compute log_sum_exp
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
        }
    } else {
        throw std::runtime_error("fused_softmax_cross_entropy: Only Float32 supported");
    }

    // Apply reduction
    if (reduction == "mean") {
        return tenzor::mean(losses);
    } else if (reduction == "sum") {
        return tenzor::sum(losses);
    } else {
        return losses;
    }
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
    } else {
        throw std::runtime_error("fused_add_relu: Only Float32/Float64 supported");
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
    } else {
        throw std::runtime_error("fused_gelu: Only Float32/Float64 supported");
    }

    return result;
}

/**
 * @brief Fused layer norm kernel (CPU implementation)
 *
 * Single-pass computation of mean, variance, and normalization.
 */
auto fused_layer_norm_kernel(
    const Tensor& input,
    const std::vector<int64_t>& normalized_shape,
    const Tensor& weight,
    const Tensor& bias,
    float eps
) -> Tensor {
    // Calculate batch size and normalization size
    int64_t norm_size = 1;
    for (auto dim : normalized_shape) {
        norm_size *= dim;
    }

    int64_t batch_size = input.numel() / norm_size;

    std::vector<int64_t> shape_vec(input.shape().begin(), input.shape().end());
    Tensor result = zeros(shape_vec, input.dtype(), input.device());

    if (input.dtype() == DType::Float32) {
        const float* in_data = input.data<float>();
        const float* weight_data = weight.data<float>();
        const float* bias_data = bias.data<float>();
        float* out_data = result.data<float>();

        for (int64_t b = 0; b < batch_size; ++b) {
            const float* batch_in = in_data + b * norm_size;
            float* batch_out = out_data + b * norm_size;

            // Single-pass mean and variance computation (Welford's algorithm)
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

            // Normalize and scale
            float inv_std = 1.0f / std::sqrt(variance + eps);
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

        for (int64_t b = 0; b < batch_size; ++b) {
            const Float16* batch_in = in_data + b * norm_size;
            Float16* batch_out = out_data + b * norm_size;

            // Single-pass mean and variance computation (use float accumulation for stability)
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

            // Normalize and scale
            float inv_std = 1.0f / std::sqrt(variance + eps);
            for (int64_t i = 0; i < norm_size; ++i) {
                float normalized = (static_cast<float>(batch_in[i]) - mean) * inv_std;
                batch_out[i] = Float16(normalized * static_cast<float>(weight_data[i]) + static_cast<float>(bias_data[i]));
            }
        }
    } else {
        throw std::runtime_error("fused_layer_norm: Only Float32/Float64/Float16 supported");
    }

    return result;
}

} // namespace cpu
} // namespace tenzor
