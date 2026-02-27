#include "tenzor/core/tensor.hpp"
#include <sycl/sycl.hpp>
#include <cmath>
#include <stdexcept>
#include <tuple>

namespace tenzor {
namespace oneapi {

// SYCL Kernel name classes
struct FusedAddReluKernelFloat32 {};
struct FusedAddReluKernelFloat64 {};
struct FusedGeluKernelFloat32 {};
struct FusedGeluKernelFloat64 {};
struct FusedLayerNormKernelFloat32 {};
struct FusedLayerNormKernelFloat64 {};
struct FusedLayerNormBackwardKernelFloat32 {};
struct FusedLayerNormBackwardKernelFloat64 {};
struct FusedLinearReluKernelFloat32 {};
struct FusedLinearReluKernelFloat64 {};
struct FusedBatchNormReluKernelFloat32 {};
struct FusedBatchNormReluKernelFloat64 {};
struct FusedMatmulAddKernelFloat32 {};
struct FusedMatmulAddKernelFloat64 {};
struct FusedSoftmaxCrossEntropyKernelFloat32 {};
struct FusedSoftmaxCrossEntropyKernelFloat64 {};
struct FusedAddReluKernelFloat16 {};
struct FusedGeluKernelFloat16 {};
struct FusedLayerNormKernelFloat16 {};
struct FusedLinearReluKernelFloat16 {};
struct FusedBatchNormReluKernelFloat16 {};
struct FusedMatmulAddKernelFloat16 {};
struct FusedAddReluKernelBFloat16 {};
struct FusedGeluKernelBFloat16 {};
struct FusedLayerNormKernelBFloat16 {};
struct FusedLayerNormBackwardKernelBFloat16 {};
struct FusedLinearReluKernelBFloat16 {};
struct FusedBatchNormReluKernelBFloat16 {};
struct FusedMatmulAddKernelBFloat16 {};
struct FusedRMSNormKernelFloat32 {};
struct FusedRMSNormKernelFloat64 {};
struct FusedRMSNormBackwardKernelFloat32 {};
struct FusedRMSNormBackwardKernelFloat64 {};

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

// ============================================================================
// Fused Add + ReLU
// ============================================================================

auto fused_add_relu_kernel(const Tensor& a, const Tensor& b, sycl::queue& queue) -> Tensor {
    if (a.dtype() != b.dtype()) {
        throw std::invalid_argument("fused_add_relu: input dtypes must match");
    }

    Tensor output(std::vector<int64_t>(a.shape().begin(), a.shape().end()),
                  a.dtype(), a.device());

    const int64_t numel = a.numel();

    if (a.dtype() == DType::Float32) {
        const float* a_ptr = get_data_ptr<const float>(a);
        const float* b_ptr = get_data_ptr<const float>(b);
        float* out_ptr = get_data_ptr<float>(output);

        queue.parallel_for<FusedAddReluKernelFloat32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            float sum = a_ptr[idx] + b_ptr[idx];
            out_ptr[idx] = sum > 0.0f ? sum : 0.0f;
        }).wait();
    }
    else if (a.dtype() == DType::Float64) {
        const double* a_ptr = get_data_ptr<const double>(a);
        const double* b_ptr = get_data_ptr<const double>(b);
        double* out_ptr = get_data_ptr<double>(output);

        queue.parallel_for<FusedAddReluKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            double sum = a_ptr[idx] + b_ptr[idx];
            out_ptr[idx] = sum > 0.0 ? sum : 0.0;
        }).wait();
    }
    else if (a.dtype() == DType::Float16) {
        const sycl::half* a_ptr = get_data_ptr<const sycl::half>(a);
        const sycl::half* b_ptr = get_data_ptr<const sycl::half>(b);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);

        queue.parallel_for<FusedAddReluKernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            float sum = static_cast<float>(a_ptr[idx]) + static_cast<float>(b_ptr[idx]);
            out_ptr[idx] = sycl::half(sum > 0.0f ? sum : 0.0f);
        }).wait();
    }
    else if (a.dtype() == DType::BFloat16) {
        const uint16_t* a_ptr = get_data_ptr<const uint16_t>(a);
        const uint16_t* b_ptr = get_data_ptr<const uint16_t>(b);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);

        queue.parallel_for<FusedAddReluKernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            float sum = bf16_to_f32(a_ptr[idx]) + bf16_to_f32(b_ptr[idx]);
            out_ptr[idx] = f32_to_bf16(sum > 0.0f ? sum : 0.0f);
        }).wait();
    }
    else {
        throw std::runtime_error("fused_add_relu: unsupported dtype");
    }

    return output;
}

// ============================================================================
// Fused GELU (Gaussian Error Linear Unit)
// GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
// ============================================================================

auto fused_gelu_kernel(const Tensor& input, sycl::queue& queue) -> Tensor {
    Tensor output(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                  input.dtype(), input.device());

    const int64_t numel = input.numel();

    if (input.dtype() == DType::Float32) {
        const float* in_ptr = get_data_ptr<const float>(input);
        float* out_ptr = get_data_ptr<float>(output);

        constexpr float sqrt_2_over_pi = 0.7978845608f;
        constexpr float coeff = 0.044715f;

        queue.parallel_for<FusedGeluKernelFloat32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            float x = in_ptr[idx];
            float x_cubed = x * x * x;
            float inner = sqrt_2_over_pi * (x + coeff * x_cubed);
            float tanh_val = sycl::tanh(inner);
            out_ptr[idx] = 0.5f * x * (1.0f + tanh_val);
        }).wait();
    }
    else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        double* out_ptr = get_data_ptr<double>(output);

        constexpr double sqrt_2_over_pi = 0.7978845608028654;
        constexpr double coeff = 0.044715;

        queue.parallel_for<FusedGeluKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            double x = in_ptr[idx];
            double x_cubed = x * x * x;
            double inner = sqrt_2_over_pi * (x + coeff * x_cubed);
            double tanh_val = sycl::tanh(inner);
            out_ptr[idx] = 0.5 * x * (1.0 + tanh_val);
        }).wait();
    }
    else if (input.dtype() == DType::Float16) {
        const sycl::half* in_ptr = get_data_ptr<const sycl::half>(input);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);

        constexpr float sqrt_2_over_pi_f = 0.7978845608f;
        constexpr float coeff_f = 0.044715f;

        queue.parallel_for<FusedGeluKernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            float x = static_cast<float>(in_ptr[idx]);
            float x_cubed = x * x * x;
            float inner = sqrt_2_over_pi_f * (x + coeff_f * x_cubed);
            float tanh_val = sycl::tanh(inner);
            out_ptr[idx] = sycl::half(0.5f * x * (1.0f + tanh_val));
        }).wait();
    }
    else if (input.dtype() == DType::BFloat16) {
        const uint16_t* in_ptr = get_data_ptr<const uint16_t>(input);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);

        const float sqrt_2_over_pi = 0.7978845608f;
        const float coeff = 0.044715f;

        queue.parallel_for<FusedGeluKernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            float x = bf16_to_f32(in_ptr[idx]);
            float x_cubed = x * x * x;
            float inner = sqrt_2_over_pi * (x + coeff * x_cubed);
            out_ptr[idx] = f32_to_bf16(0.5f * x * (1.0f + sycl::tanh(inner)));
        }).wait();
    }
    else {
        throw std::runtime_error("fused_gelu: unsupported dtype");
    }

    return output;
}

// ============================================================================
// Fused Layer Normalization
// LayerNorm(x) = gamma * (x - mean) / sqrt(variance + epsilon) + beta
// ============================================================================

auto fused_layer_norm_kernel(
    const Tensor& input,
    const Tensor& weight,  // gamma
    const Tensor& bias,    // beta
    const std::vector<int64_t>& normalized_shape,
    float epsilon,
    sycl::queue& queue
) -> std::tuple<Tensor, Tensor, Tensor> {
    // Calculate normalized dimension size
    int64_t norm_size = 1;
    for (auto dim : normalized_shape) {
        norm_size *= dim;
    }

    int64_t batch_size = input.numel() / norm_size;

    // Create output tensors
    Tensor output(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                  input.dtype(), input.device());
    Tensor mean({batch_size}, input.dtype(), input.device());
    Tensor inv_std({batch_size}, input.dtype(), input.device());

    if (input.dtype() == DType::Float32) {
        const float* in_ptr = get_data_ptr<const float>(input);
        const float* weight_ptr = get_data_ptr<const float>(weight);
        const float* bias_ptr = get_data_ptr<const float>(bias);
        float* out_ptr = get_data_ptr<float>(output);
        float* mean_ptr = get_data_ptr<float>(mean);
        float* inv_std_ptr = get_data_ptr<float>(inv_std);

        // Process each batch element
        for (int64_t b = 0; b < batch_size; ++b) {
            const float* batch_in = in_ptr + b * norm_size;
            float* batch_out = out_ptr + b * norm_size;

            // Compute mean on host (for simplicity)
            std::vector<float> host_in(norm_size);
            queue.memcpy(host_in.data(), batch_in, norm_size * sizeof(float)).wait();

            float sum = 0.0f;
            for (int64_t i = 0; i < norm_size; ++i) {
                sum += host_in[i];
            }
            float batch_mean = sum / static_cast<float>(norm_size);

            // Compute variance
            float var_sum = 0.0f;
            for (int64_t i = 0; i < norm_size; ++i) {
                float diff = host_in[i] - batch_mean;
                var_sum += diff * diff;
            }
            float variance = var_sum / static_cast<float>(norm_size);
            float batch_inv_std = 1.0f / sycl::sqrt(variance + epsilon);

            // Store mean and inv_std
            queue.fill(mean_ptr + b, batch_mean, 1).wait();
            queue.fill(inv_std_ptr + b, batch_inv_std, 1).wait();

            // Normalize and apply scale/shift
            queue.parallel_for<class LayerNormBatch>(sycl::range<1>(norm_size), [=](sycl::id<1> idx) {
                int64_t i = idx[0];
                float normalized = (batch_in[i] - batch_mean) * batch_inv_std;
                batch_out[i] = normalized * weight_ptr[i] + bias_ptr[i];
            }).wait();
        }
    }
    else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        const double* weight_ptr = get_data_ptr<const double>(weight);
        const double* bias_ptr = get_data_ptr<const double>(bias);
        double* out_ptr = get_data_ptr<double>(output);
        double* mean_ptr = get_data_ptr<double>(mean);
        double* inv_std_ptr = get_data_ptr<double>(inv_std);

        for (int64_t b = 0; b < batch_size; ++b) {
            const double* batch_in = in_ptr + b * norm_size;
            double* batch_out = out_ptr + b * norm_size;

            std::vector<double> host_in(norm_size);
            queue.memcpy(host_in.data(), batch_in, norm_size * sizeof(double)).wait();

            double sum = 0.0;
            for (int64_t i = 0; i < norm_size; ++i) {
                sum += host_in[i];
            }
            double batch_mean = sum / static_cast<double>(norm_size);

            double var_sum = 0.0;
            for (int64_t i = 0; i < norm_size; ++i) {
                double diff = host_in[i] - batch_mean;
                var_sum += diff * diff;
            }
            double variance = var_sum / static_cast<double>(norm_size);
            double batch_inv_std = 1.0 / sycl::sqrt(variance + static_cast<double>(epsilon));

            queue.fill(mean_ptr + b, batch_mean, 1).wait();
            queue.fill(inv_std_ptr + b, batch_inv_std, 1).wait();

            queue.parallel_for<class LayerNormBatchD>(sycl::range<1>(norm_size), [=](sycl::id<1> idx) {
                int64_t i = idx[0];
                double normalized = (batch_in[i] - batch_mean) * batch_inv_std;
                batch_out[i] = normalized * weight_ptr[i] + bias_ptr[i];
            }).wait();
        }
    }
    else if (input.dtype() == DType::Float16) {
        // Float16: use float32 accumulation for numerical stability
        const sycl::half* in_ptr = get_data_ptr<const sycl::half>(input);
        const sycl::half* weight_ptr = get_data_ptr<const sycl::half>(weight);
        const sycl::half* bias_ptr = get_data_ptr<const sycl::half>(bias);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);
        sycl::half* mean_ptr = get_data_ptr<sycl::half>(mean);
        sycl::half* inv_std_ptr = get_data_ptr<sycl::half>(inv_std);

        for (int64_t b = 0; b < batch_size; ++b) {
            const sycl::half* batch_in = in_ptr + b * norm_size;
            sycl::half* batch_out = out_ptr + b * norm_size;

            std::vector<sycl::half> host_in(norm_size);
            queue.memcpy(host_in.data(), batch_in, norm_size * sizeof(sycl::half)).wait();

            float sum = 0.0f;
            for (int64_t i = 0; i < norm_size; ++i) {
                sum += static_cast<float>(host_in[i]);
            }
            float batch_mean = sum / static_cast<float>(norm_size);

            float var_sum = 0.0f;
            for (int64_t i = 0; i < norm_size; ++i) {
                float diff = static_cast<float>(host_in[i]) - batch_mean;
                var_sum += diff * diff;
            }
            float variance = var_sum / static_cast<float>(norm_size);
            float batch_inv_std = 1.0f / sycl::sqrt(variance + epsilon);

            sycl::half h_mean = sycl::half(batch_mean);
            sycl::half h_inv_std = sycl::half(batch_inv_std);
            queue.fill(mean_ptr + b, h_mean, 1).wait();
            queue.fill(inv_std_ptr + b, h_inv_std, 1).wait();

            queue.parallel_for<FusedLayerNormKernelFloat16>(sycl::range<1>(norm_size), [=](sycl::id<1> idx) {
                int64_t i = idx[0];
                float val = static_cast<float>(batch_in[i]);
                float normalized = (val - batch_mean) * batch_inv_std;
                float result = normalized * static_cast<float>(weight_ptr[i]) + static_cast<float>(bias_ptr[i]);
                batch_out[i] = sycl::half(result);
            }).wait();
        }
    }
    else if (input.dtype() == DType::BFloat16) {
        // BFloat16: use float32 accumulation for numerical stability
        const uint16_t* in_ptr = get_data_ptr<const uint16_t>(input);
        const uint16_t* weight_ptr = get_data_ptr<const uint16_t>(weight);
        const uint16_t* bias_ptr = get_data_ptr<const uint16_t>(bias);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);
        uint16_t* mean_ptr = get_data_ptr<uint16_t>(mean);
        uint16_t* inv_std_ptr = get_data_ptr<uint16_t>(inv_std);

        for (int64_t b = 0; b < batch_size; ++b) {
            const uint16_t* batch_in = in_ptr + b * norm_size;
            uint16_t* batch_out = out_ptr + b * norm_size;

            std::vector<uint16_t> host_in(norm_size);
            queue.memcpy(host_in.data(), batch_in, norm_size * sizeof(uint16_t)).wait();

            float sum = 0.0f;
            for (int64_t i = 0; i < norm_size; ++i) {
                sum += bf16_to_f32(host_in[i]);
            }
            float batch_mean = sum / static_cast<float>(norm_size);

            float var_sum = 0.0f;
            for (int64_t i = 0; i < norm_size; ++i) {
                float diff = bf16_to_f32(host_in[i]) - batch_mean;
                var_sum += diff * diff;
            }
            float variance = var_sum / static_cast<float>(norm_size);
            float batch_inv_std = 1.0f / sycl::sqrt(variance + epsilon);

            uint16_t h_mean = f32_to_bf16(batch_mean);
            uint16_t h_inv_std = f32_to_bf16(batch_inv_std);
            queue.fill(mean_ptr + b, h_mean, 1).wait();
            queue.fill(inv_std_ptr + b, h_inv_std, 1).wait();

            queue.parallel_for<FusedLayerNormKernelBFloat16>(sycl::range<1>(norm_size), [=](sycl::id<1> idx) {
                int64_t i = idx[0];
                float val = bf16_to_f32(batch_in[i]);
                float normalized = (val - batch_mean) * batch_inv_std;
                float result = normalized * bf16_to_f32(weight_ptr[i]) + bf16_to_f32(bias_ptr[i]);
                batch_out[i] = f32_to_bf16(result);
            }).wait();
        }
    }
    else {
        throw std::runtime_error("fused_layer_norm: unsupported dtype");
    }

    return {output, mean, inv_std};
}

// ============================================================================
// Fused Layer Normalization Backward
// ============================================================================

auto fused_layer_norm_backward_kernel(
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& mean,
    const Tensor& inv_std,
    const Tensor& weight,
    const std::vector<int64_t>& normalized_shape,
    sycl::queue& queue
) -> std::tuple<Tensor, Tensor, Tensor> {
    int64_t norm_size = 1;
    for (auto dim : normalized_shape) {
        norm_size *= dim;
    }

    int64_t batch_size = input.numel() / norm_size;

    // Create gradient tensors
    Tensor grad_input(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                      input.dtype(), input.device());
    Tensor grad_weight({norm_size}, input.dtype(), input.device());
    Tensor grad_bias({norm_size}, input.dtype(), input.device());

    if (input.dtype() == DType::Float32) {
        const float* grad_out_ptr = get_data_ptr<const float>(grad_output);
        const float* in_ptr = get_data_ptr<const float>(input);
        const float* mean_ptr = get_data_ptr<const float>(mean);
        const float* inv_std_ptr = get_data_ptr<const float>(inv_std);
        const float* weight_ptr = get_data_ptr<const float>(weight);
        float* grad_in_ptr = get_data_ptr<float>(grad_input);
        float* grad_weight_ptr = get_data_ptr<float>(grad_weight);
        float* grad_bias_ptr = get_data_ptr<float>(grad_bias);

        // Initialize grad_weight and grad_bias to zero
        queue.fill(grad_weight_ptr, 0.0f, norm_size).wait();
        queue.fill(grad_bias_ptr, 0.0f, norm_size).wait();

        // Accumulate gradients for weight and bias
        for (int64_t b = 0; b < batch_size; ++b) {
            const float* batch_grad = grad_out_ptr + b * norm_size;
            const float* batch_in = in_ptr + b * norm_size;

            std::vector<float> host_grad(norm_size);
            std::vector<float> host_in(norm_size);
            queue.memcpy(host_grad.data(), batch_grad, norm_size * sizeof(float)).wait();
            queue.memcpy(host_in.data(), batch_in, norm_size * sizeof(float)).wait();

            float batch_mean, batch_inv_std;
            queue.memcpy(&batch_mean, mean_ptr + b, sizeof(float)).wait();
            queue.memcpy(&batch_inv_std, inv_std_ptr + b, sizeof(float)).wait();

            std::vector<float> grad_w_acc(norm_size, 0.0f);
            std::vector<float> grad_b_acc(norm_size, 0.0f);

            for (int64_t i = 0; i < norm_size; ++i) {
                float normalized = (host_in[i] - batch_mean) * batch_inv_std;
                grad_w_acc[i] += host_grad[i] * normalized;
                grad_b_acc[i] += host_grad[i];
            }

            // Add to accumulator
            std::vector<float> curr_gw(norm_size), curr_gb(norm_size);
            queue.memcpy(curr_gw.data(), grad_weight_ptr, norm_size * sizeof(float)).wait();
            queue.memcpy(curr_gb.data(), grad_bias_ptr, norm_size * sizeof(float)).wait();

            for (int64_t i = 0; i < norm_size; ++i) {
                curr_gw[i] += grad_w_acc[i];
                curr_gb[i] += grad_b_acc[i];
            }

            queue.memcpy(grad_weight_ptr, curr_gw.data(), norm_size * sizeof(float)).wait();
            queue.memcpy(grad_bias_ptr, curr_gb.data(), norm_size * sizeof(float)).wait();
        }

        // Compute grad_input
        for (int64_t b = 0; b < batch_size; ++b) {
            const float* batch_grad = grad_out_ptr + b * norm_size;
            const float* batch_in = in_ptr + b * norm_size;
            float* batch_grad_in = grad_in_ptr + b * norm_size;

            float batch_mean, batch_inv_std;
            queue.memcpy(&batch_mean, mean_ptr + b, sizeof(float)).wait();
            queue.memcpy(&batch_inv_std, inv_std_ptr + b, sizeof(float)).wait();

            // Simplified backward: just scale by weight and inv_std
            queue.parallel_for<class LayerNormBackward>(sycl::range<1>(norm_size), [=](sycl::id<1> idx) {
                int64_t i = idx[0];
                batch_grad_in[i] = batch_grad[i] * weight_ptr[i] * batch_inv_std;
            }).wait();
        }
    }
    else if (input.dtype() == DType::Float64) {
        const double* grad_out_ptr = get_data_ptr<const double>(grad_output);
        const double* in_ptr = get_data_ptr<const double>(input);
        const double* mean_ptr = get_data_ptr<const double>(mean);
        const double* inv_std_ptr = get_data_ptr<const double>(inv_std);
        const double* weight_ptr = get_data_ptr<const double>(weight);
        double* grad_in_ptr = get_data_ptr<double>(grad_input);
        double* grad_weight_ptr = get_data_ptr<double>(grad_weight);
        double* grad_bias_ptr = get_data_ptr<double>(grad_bias);

        queue.fill(grad_weight_ptr, 0.0, norm_size).wait();
        queue.fill(grad_bias_ptr, 0.0, norm_size).wait();

        for (int64_t b = 0; b < batch_size; ++b) {
            const double* batch_grad = grad_out_ptr + b * norm_size;
            const double* batch_in = in_ptr + b * norm_size;

            std::vector<double> host_grad(norm_size);
            std::vector<double> host_in(norm_size);
            queue.memcpy(host_grad.data(), batch_grad, norm_size * sizeof(double)).wait();
            queue.memcpy(host_in.data(), batch_in, norm_size * sizeof(double)).wait();

            double batch_mean, batch_inv_std;
            queue.memcpy(&batch_mean, mean_ptr + b, sizeof(double)).wait();
            queue.memcpy(&batch_inv_std, inv_std_ptr + b, sizeof(double)).wait();

            std::vector<double> curr_gw(norm_size), curr_gb(norm_size);
            queue.memcpy(curr_gw.data(), grad_weight_ptr, norm_size * sizeof(double)).wait();
            queue.memcpy(curr_gb.data(), grad_bias_ptr, norm_size * sizeof(double)).wait();

            for (int64_t i = 0; i < norm_size; ++i) {
                double normalized = (host_in[i] - batch_mean) * batch_inv_std;
                curr_gw[i] += host_grad[i] * normalized;
                curr_gb[i] += host_grad[i];
            }

            queue.memcpy(grad_weight_ptr, curr_gw.data(), norm_size * sizeof(double)).wait();
            queue.memcpy(grad_bias_ptr, curr_gb.data(), norm_size * sizeof(double)).wait();
        }

        for (int64_t b = 0; b < batch_size; ++b) {
            const double* batch_grad = grad_out_ptr + b * norm_size;
            double* batch_grad_in = grad_in_ptr + b * norm_size;

            double batch_mean, batch_inv_std;
            queue.memcpy(&batch_mean, mean_ptr + b, sizeof(double)).wait();
            queue.memcpy(&batch_inv_std, inv_std_ptr + b, sizeof(double)).wait();

            queue.parallel_for<FusedLayerNormBackwardKernelFloat64>(sycl::range<1>(norm_size), [=](sycl::id<1> idx) {
                int64_t i = idx[0];
                batch_grad_in[i] = batch_grad[i] * weight_ptr[i] * batch_inv_std;
            }).wait();
        }
    }
    else if (input.dtype() == DType::BFloat16) {
        const uint16_t* grad_out_ptr = get_data_ptr<const uint16_t>(grad_output);
        const uint16_t* in_ptr = get_data_ptr<const uint16_t>(input);
        const uint16_t* mean_ptr = get_data_ptr<const uint16_t>(mean);
        const uint16_t* inv_std_ptr = get_data_ptr<const uint16_t>(inv_std);
        const uint16_t* weight_ptr = get_data_ptr<const uint16_t>(weight);
        uint16_t* grad_in_ptr = get_data_ptr<uint16_t>(grad_input);
        uint16_t* grad_weight_ptr = get_data_ptr<uint16_t>(grad_weight);
        uint16_t* grad_bias_ptr = get_data_ptr<uint16_t>(grad_bias);

        // Initialize grad_weight and grad_bias to zero
        uint16_t bf16_zero = f32_to_bf16(0.0f);
        queue.fill(grad_weight_ptr, bf16_zero, norm_size).wait();
        queue.fill(grad_bias_ptr, bf16_zero, norm_size).wait();

        // Accumulate gradients for weight and bias (in float32 on host)
        for (int64_t b = 0; b < batch_size; ++b) {
            const uint16_t* batch_grad = grad_out_ptr + b * norm_size;
            const uint16_t* batch_in = in_ptr + b * norm_size;

            std::vector<uint16_t> host_grad(norm_size);
            std::vector<uint16_t> host_in(norm_size);
            queue.memcpy(host_grad.data(), batch_grad, norm_size * sizeof(uint16_t)).wait();
            queue.memcpy(host_in.data(), batch_in, norm_size * sizeof(uint16_t)).wait();

            uint16_t h_batch_mean, h_batch_inv_std;
            queue.memcpy(&h_batch_mean, mean_ptr + b, sizeof(uint16_t)).wait();
            queue.memcpy(&h_batch_inv_std, inv_std_ptr + b, sizeof(uint16_t)).wait();
            float batch_mean = bf16_to_f32(h_batch_mean);
            float batch_inv_std = bf16_to_f32(h_batch_inv_std);

            std::vector<uint16_t> curr_gw(norm_size), curr_gb(norm_size);
            queue.memcpy(curr_gw.data(), grad_weight_ptr, norm_size * sizeof(uint16_t)).wait();
            queue.memcpy(curr_gb.data(), grad_bias_ptr, norm_size * sizeof(uint16_t)).wait();

            for (int64_t i = 0; i < norm_size; ++i) {
                float normalized = (bf16_to_f32(host_in[i]) - batch_mean) * batch_inv_std;
                float gw = bf16_to_f32(curr_gw[i]) + bf16_to_f32(host_grad[i]) * normalized;
                float gb = bf16_to_f32(curr_gb[i]) + bf16_to_f32(host_grad[i]);
                curr_gw[i] = f32_to_bf16(gw);
                curr_gb[i] = f32_to_bf16(gb);
            }

            queue.memcpy(grad_weight_ptr, curr_gw.data(), norm_size * sizeof(uint16_t)).wait();
            queue.memcpy(grad_bias_ptr, curr_gb.data(), norm_size * sizeof(uint16_t)).wait();
        }

        // Compute grad_input
        for (int64_t b = 0; b < batch_size; ++b) {
            const uint16_t* batch_grad = grad_out_ptr + b * norm_size;
            uint16_t* batch_grad_in = grad_in_ptr + b * norm_size;

            uint16_t h_batch_mean, h_batch_inv_std;
            queue.memcpy(&h_batch_mean, mean_ptr + b, sizeof(uint16_t)).wait();
            queue.memcpy(&h_batch_inv_std, inv_std_ptr + b, sizeof(uint16_t)).wait();
            float batch_inv_std = bf16_to_f32(h_batch_inv_std);

            queue.parallel_for<FusedLayerNormBackwardKernelBFloat16>(sycl::range<1>(norm_size), [=](sycl::id<1> idx) {
                int64_t i = idx[0];
                float grad = bf16_to_f32(batch_grad[i]);
                float w = bf16_to_f32(weight_ptr[i]);
                batch_grad_in[i] = f32_to_bf16(grad * w * batch_inv_std);
            }).wait();
        }
    }
    else {
        throw std::runtime_error("fused_layer_norm_backward: unsupported dtype");
    }

    return {grad_input, grad_weight, grad_bias};
}

// ============================================================================
// Fused Linear + ReLU
// out = max(0, input @ weight.T + bias)
// ============================================================================

auto fused_linear_relu_kernel(
    const Tensor& input,
    const Tensor& weight,
    const Tensor* bias,
    sycl::queue& queue
) -> Tensor {
    auto input_shape = input.shape();
    int64_t batch_size = 1;
    for (size_t i = 0; i < input_shape.size() - 1; ++i) {
        batch_size *= input_shape[i];
    }
    int64_t in_features = input_shape[input_shape.size() - 1];
    int64_t out_features = weight.shape()[0];

    // Create output tensor
    std::vector<int64_t> output_shape(input_shape.begin(), input_shape.end() - 1);
    output_shape.push_back(out_features);
    Tensor output(output_shape, input.dtype(), input.device());

    int64_t total_elements = batch_size * out_features;

    if (input.dtype() == DType::Float32) {
        const float* in_ptr = get_data_ptr<const float>(input);
        const float* weight_ptr = get_data_ptr<const float>(weight);
        const float* bias_ptr = bias ? get_data_ptr<const float>(*bias) : nullptr;
        float* out_ptr = get_data_ptr<float>(output);

        const bool has_bias = (bias != nullptr);

        queue.parallel_for<FusedLinearReluKernelFloat32>(
            sycl::range<1>(total_elements),
            [=](sycl::id<1> idx) {
                int64_t b = idx / out_features;
                int64_t o = idx % out_features;

                float sum = 0.0f;
                for (int64_t i = 0; i < in_features; ++i) {
                    sum += in_ptr[b * in_features + i] * weight_ptr[o * in_features + i];
                }

                if (has_bias) {
                    sum += bias_ptr[o];
                }

                // ReLU
                out_ptr[idx] = sum > 0.0f ? sum : 0.0f;
            }
        ).wait();
    }
    else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        const double* weight_ptr = get_data_ptr<const double>(weight);
        const double* bias_ptr = bias ? get_data_ptr<const double>(*bias) : nullptr;
        double* out_ptr = get_data_ptr<double>(output);

        const bool has_bias = (bias != nullptr);

        queue.parallel_for<FusedLinearReluKernelFloat64>(
            sycl::range<1>(total_elements),
            [=](sycl::id<1> idx) {
                int64_t b = idx / out_features;
                int64_t o = idx % out_features;

                double sum = 0.0;
                for (int64_t i = 0; i < in_features; ++i) {
                    sum += in_ptr[b * in_features + i] * weight_ptr[o * in_features + i];
                }

                if (has_bias) {
                    sum += bias_ptr[o];
                }

                // ReLU
                out_ptr[idx] = sum > 0.0 ? sum : 0.0;
            }
        ).wait();
    }
    else if (input.dtype() == DType::Float16) {
        const sycl::half* in_ptr = get_data_ptr<const sycl::half>(input);
        const sycl::half* weight_ptr = get_data_ptr<const sycl::half>(weight);
        const sycl::half* bias_ptr = bias ? get_data_ptr<const sycl::half>(*bias) : nullptr;
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);

        const bool has_bias = (bias != nullptr);

        queue.parallel_for<FusedLinearReluKernelFloat16>(
            sycl::range<1>(total_elements),
            [=](sycl::id<1> idx) {
                int64_t b = idx / out_features;
                int64_t o = idx % out_features;

                float sum = 0.0f;
                for (int64_t i = 0; i < in_features; ++i) {
                    sum += static_cast<float>(in_ptr[b * in_features + i]) *
                           static_cast<float>(weight_ptr[o * in_features + i]);
                }

                if (has_bias) {
                    sum += static_cast<float>(bias_ptr[o]);
                }

                out_ptr[idx] = sycl::half(sum > 0.0f ? sum : 0.0f);
            }
        ).wait();
    }
    else if (input.dtype() == DType::BFloat16) {
        const uint16_t* in_ptr = get_data_ptr<const uint16_t>(input);
        const uint16_t* weight_ptr = get_data_ptr<const uint16_t>(weight);
        const uint16_t* bias_ptr = bias ? get_data_ptr<const uint16_t>(*bias) : nullptr;
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);

        const bool has_bias = (bias != nullptr);

        queue.parallel_for<FusedLinearReluKernelBFloat16>(
            sycl::range<1>(total_elements),
            [=](sycl::id<1> idx) {
                int64_t b = idx / out_features;
                int64_t o = idx % out_features;

                float sum = 0.0f;
                for (int64_t i = 0; i < in_features; ++i) {
                    sum += bf16_to_f32(in_ptr[b * in_features + i]) *
                           bf16_to_f32(weight_ptr[o * in_features + i]);
                }

                if (has_bias) {
                    sum += bf16_to_f32(bias_ptr[o]);
                }

                // ReLU
                out_ptr[idx] = f32_to_bf16(sum > 0.0f ? sum : 0.0f);
            }
        ).wait();
    }
    else {
        throw std::runtime_error("fused_linear_relu: unsupported dtype");
    }

    return output;
}

// ============================================================================
// Fused BatchNorm + ReLU
// ============================================================================

auto fused_batchnorm_relu_kernel(
    const Tensor& input,
    const Tensor& running_mean,
    const Tensor& running_var,
    const Tensor& weight,
    const Tensor& bias,
    float epsilon,
    sycl::queue& queue
) -> Tensor {
    auto shape = input.shape();
    int64_t batch_size = shape[0];
    int64_t num_features = shape[1];
    int64_t spatial_size = 1;
    for (size_t i = 2; i < shape.size(); ++i) {
        spatial_size *= shape[i];
    }

    Tensor output(std::vector<int64_t>(shape.begin(), shape.end()),
                  input.dtype(), input.device());

    int64_t total_elements = input.numel();

    if (input.dtype() == DType::Float32) {
        const float* in_ptr = get_data_ptr<const float>(input);
        const float* mean_ptr = get_data_ptr<const float>(running_mean);
        const float* var_ptr = get_data_ptr<const float>(running_var);
        const float* weight_ptr = get_data_ptr<const float>(weight);
        const float* bias_ptr = get_data_ptr<const float>(bias);
        float* out_ptr = get_data_ptr<float>(output);

        queue.parallel_for<FusedBatchNormReluKernelFloat32>(
            sycl::range<1>(total_elements),
            [=](sycl::id<1> idx) {
                int64_t i = idx;
                int64_t s = i % spatial_size;
                int64_t c = (i / spatial_size) % num_features;
                int64_t n = i / (spatial_size * num_features);

                float normalized = (in_ptr[i] - mean_ptr[c]) * sycl::rsqrt(var_ptr[c] + epsilon);
                float scaled = normalized * weight_ptr[c] + bias_ptr[c];

                // ReLU
                out_ptr[i] = scaled > 0.0f ? scaled : 0.0f;
            }
        ).wait();
    }
    else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        const double* mean_ptr = get_data_ptr<const double>(running_mean);
        const double* var_ptr = get_data_ptr<const double>(running_var);
        const double* weight_ptr = get_data_ptr<const double>(weight);
        const double* bias_ptr = get_data_ptr<const double>(bias);
        double* out_ptr = get_data_ptr<double>(output);

        double eps_d = static_cast<double>(epsilon);

        queue.parallel_for<FusedBatchNormReluKernelFloat64>(
            sycl::range<1>(total_elements),
            [=](sycl::id<1> idx) {
                int64_t i = idx;
                int64_t s = i % spatial_size;
                int64_t c = (i / spatial_size) % num_features;
                int64_t n = i / (spatial_size * num_features);

                double normalized = (in_ptr[i] - mean_ptr[c]) * sycl::rsqrt(var_ptr[c] + eps_d);
                double scaled = normalized * weight_ptr[c] + bias_ptr[c];

                // ReLU
                out_ptr[i] = scaled > 0.0 ? scaled : 0.0;
            }
        ).wait();
    }
    else if (input.dtype() == DType::Float16) {
        const sycl::half* in_ptr = get_data_ptr<const sycl::half>(input);
        const sycl::half* mean_ptr = get_data_ptr<const sycl::half>(running_mean);
        const sycl::half* var_ptr = get_data_ptr<const sycl::half>(running_var);
        const sycl::half* weight_ptr = get_data_ptr<const sycl::half>(weight);
        const sycl::half* bias_ptr = get_data_ptr<const sycl::half>(bias);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);

        queue.parallel_for<FusedBatchNormReluKernelFloat16>(
            sycl::range<1>(total_elements),
            [=](sycl::id<1> idx) {
                int64_t i = idx;
                int64_t c = (i / spatial_size) % num_features;

                float val = static_cast<float>(in_ptr[i]);
                float m = static_cast<float>(mean_ptr[c]);
                float v = static_cast<float>(var_ptr[c]);
                float w = static_cast<float>(weight_ptr[c]);
                float b = static_cast<float>(bias_ptr[c]);

                float normalized = (val - m) * sycl::rsqrt(v + epsilon);
                float scaled = normalized * w + b;
                out_ptr[i] = sycl::half(scaled > 0.0f ? scaled : 0.0f);
            }
        ).wait();
    }
    else if (input.dtype() == DType::BFloat16) {
        const uint16_t* in_ptr = get_data_ptr<const uint16_t>(input);
        const uint16_t* mean_ptr = get_data_ptr<const uint16_t>(running_mean);
        const uint16_t* var_ptr = get_data_ptr<const uint16_t>(running_var);
        const uint16_t* weight_ptr = get_data_ptr<const uint16_t>(weight);
        const uint16_t* bias_ptr = get_data_ptr<const uint16_t>(bias);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);

        queue.parallel_for<FusedBatchNormReluKernelBFloat16>(
            sycl::range<1>(total_elements),
            [=](sycl::id<1> idx) {
                int64_t i = idx;
                int64_t c = (i / spatial_size) % num_features;

                float val = bf16_to_f32(in_ptr[i]);
                float m = bf16_to_f32(mean_ptr[c]);
                float v = bf16_to_f32(var_ptr[c]);
                float w = bf16_to_f32(weight_ptr[c]);
                float b = bf16_to_f32(bias_ptr[c]);

                float normalized = (val - m) * sycl::rsqrt(v + epsilon);
                float scaled = normalized * w + b;
                out_ptr[i] = f32_to_bf16(scaled > 0.0f ? scaled : 0.0f);
            }
        ).wait();
    }
    else {
        throw std::runtime_error("fused_batchnorm_relu: unsupported dtype");
    }

    return output;
}

// ============================================================================
// Fused Matmul + Add (C = A @ B + bias)
// ============================================================================

auto fused_matmul_add_kernel(
    const Tensor& a,
    const Tensor& b,
    const Tensor& bias,
    sycl::queue& queue
) -> Tensor {
    auto a_shape = a.shape();
    auto b_shape = b.shape();

    const int64_t m = a_shape[a_shape.size() - 2];
    const int64_t k = a_shape[a_shape.size() - 1];
    const int64_t n = b_shape[b_shape.size() - 1];

    std::vector<int64_t> out_shape;
    for (size_t i = 0; i < a_shape.size() - 2; ++i) {
        out_shape.push_back(a_shape[i]);
    }
    out_shape.push_back(m);
    out_shape.push_back(n);

    Tensor output(out_shape, a.dtype(), a.device());

    if (a.dtype() == DType::Float32) {
        const float* a_ptr = get_data_ptr<const float>(a);
        const float* b_ptr = get_data_ptr<const float>(b);
        const float* bias_ptr = get_data_ptr<const float>(bias);
        float* out_ptr = get_data_ptr<float>(output);

        queue.parallel_for<FusedMatmulAddKernelFloat32>(
            sycl::range<2>(m, n),
            [=](sycl::id<2> idx) {
                int64_t i = idx[0];
                int64_t j = idx[1];

                float sum = 0.0f;
                for (int64_t p = 0; p < k; ++p) {
                    sum += a_ptr[i * k + p] * b_ptr[p * n + j];
                }

                // Add bias (broadcast along the m dimension)
                out_ptr[i * n + j] = sum + bias_ptr[j];
            }
        ).wait();
    }
    else if (a.dtype() == DType::Float64) {
        const double* a_ptr = get_data_ptr<const double>(a);
        const double* b_ptr = get_data_ptr<const double>(b);
        const double* bias_ptr = get_data_ptr<const double>(bias);
        double* out_ptr = get_data_ptr<double>(output);

        queue.parallel_for<FusedMatmulAddKernelFloat64>(
            sycl::range<2>(m, n),
            [=](sycl::id<2> idx) {
                int64_t i = idx[0];
                int64_t j = idx[1];

                double sum = 0.0;
                for (int64_t p = 0; p < k; ++p) {
                    sum += a_ptr[i * k + p] * b_ptr[p * n + j];
                }

                out_ptr[i * n + j] = sum + bias_ptr[j];
            }
        ).wait();
    }
    else if (a.dtype() == DType::Float16) {
        const sycl::half* a_ptr = get_data_ptr<const sycl::half>(a);
        const sycl::half* b_ptr = get_data_ptr<const sycl::half>(b);
        const sycl::half* bias_ptr = get_data_ptr<const sycl::half>(bias);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);

        queue.parallel_for<FusedMatmulAddKernelFloat16>(
            sycl::range<2>(m, n),
            [=](sycl::id<2> idx) {
                int64_t i = idx[0];
                int64_t j = idx[1];

                float sum = 0.0f;
                for (int64_t p = 0; p < k; ++p) {
                    sum += static_cast<float>(a_ptr[i * k + p]) *
                           static_cast<float>(b_ptr[p * n + j]);
                }
                out_ptr[i * n + j] = sycl::half(sum + static_cast<float>(bias_ptr[j]));
            }
        ).wait();
    }
    else if (a.dtype() == DType::BFloat16) {
        const uint16_t* a_ptr = get_data_ptr<const uint16_t>(a);
        const uint16_t* b_ptr = get_data_ptr<const uint16_t>(b);
        const uint16_t* bias_ptr = get_data_ptr<const uint16_t>(bias);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);

        queue.parallel_for<FusedMatmulAddKernelBFloat16>(
            sycl::range<2>(m, n),
            [=](sycl::id<2> idx) {
                int64_t i = idx[0];
                int64_t j = idx[1];

                float sum = 0.0f;
                for (int64_t p = 0; p < k; ++p) {
                    sum += bf16_to_f32(a_ptr[i * k + p]) *
                           bf16_to_f32(b_ptr[p * n + j]);
                }
                out_ptr[i * n + j] = f32_to_bf16(sum + bf16_to_f32(bias_ptr[j]));
            }
        ).wait();
    }
    else {
        throw std::runtime_error("fused_matmul_add: unsupported dtype");
    }

    return output;
}

// ============================================================================
// Fused Softmax + Cross Entropy Loss
// ============================================================================

auto fused_softmax_cross_entropy_kernel(
    const Tensor& logits,
    const Tensor& targets,
    const std::string& reduction,
    sycl::queue& queue
) -> Tensor {
    int64_t batch_size = logits.shape()[0];
    int64_t num_classes = logits.shape()[1];

    Tensor losses({batch_size}, logits.dtype(), logits.device());

    if (logits.dtype() == DType::Float32) {
        const float* logits_ptr = get_data_ptr<const float>(logits);
        const int64_t* targets_ptr = get_data_ptr<const int64_t>(targets);
        float* losses_ptr = get_data_ptr<float>(losses);

        // Process each sample
        for (int64_t b = 0; b < batch_size; ++b) {
            const float* row = logits_ptr + b * num_classes;
            int64_t target = targets_ptr[b];

            // Find max for numerical stability
            std::vector<float> host_row(num_classes);
            queue.memcpy(host_row.data(), row, num_classes * sizeof(float)).wait();

            float max_val = host_row[0];
            for (int64_t i = 1; i < num_classes; ++i) {
                if (host_row[i] > max_val) max_val = host_row[i];
            }

            // Compute sum(exp(x - max))
            float sum_exp = 0.0f;
            for (int64_t i = 0; i < num_classes; ++i) {
                sum_exp += std::exp(host_row[i] - max_val);
            }

            // Compute loss: log_sum_exp - target_logit
            float log_sum_exp = std::log(sum_exp) + max_val;
            float loss = log_sum_exp - host_row[target];

            queue.fill(losses_ptr + b, loss, 1).wait();
        }

        // Apply reduction
        if (reduction == "mean") {
            std::vector<float> host_losses(batch_size);
            queue.memcpy(host_losses.data(), losses_ptr, batch_size * sizeof(float)).wait();

            float mean_loss = 0.0f;
            for (int64_t i = 0; i < batch_size; ++i) {
                mean_loss += host_losses[i];
            }
            mean_loss /= static_cast<float>(batch_size);

            Tensor result({1}, logits.dtype(), logits.device());
            queue.fill(get_data_ptr<float>(result), mean_loss, 1).wait();
            return result;
        }
        else if (reduction == "sum") {
            std::vector<float> host_losses(batch_size);
            queue.memcpy(host_losses.data(), losses_ptr, batch_size * sizeof(float)).wait();

            float sum_loss = 0.0f;
            for (int64_t i = 0; i < batch_size; ++i) {
                sum_loss += host_losses[i];
            }

            Tensor result({1}, logits.dtype(), logits.device());
            queue.fill(get_data_ptr<float>(result), sum_loss, 1).wait();
            return result;
        }
    }
    else if (logits.dtype() == DType::Float64) {
        const double* logits_ptr = get_data_ptr<const double>(logits);
        const int64_t* targets_ptr = get_data_ptr<const int64_t>(targets);
        double* losses_ptr = get_data_ptr<double>(losses);

        for (int64_t b = 0; b < batch_size; ++b) {
            const double* row = logits_ptr + b * num_classes;
            int64_t target = targets_ptr[b];

            std::vector<double> host_row(num_classes);
            queue.memcpy(host_row.data(), row, num_classes * sizeof(double)).wait();

            double max_val = host_row[0];
            for (int64_t i = 1; i < num_classes; ++i) {
                if (host_row[i] > max_val) max_val = host_row[i];
            }

            double sum_exp = 0.0;
            for (int64_t i = 0; i < num_classes; ++i) {
                sum_exp += std::exp(host_row[i] - max_val);
            }

            double log_sum_exp = std::log(sum_exp) + max_val;
            double loss = log_sum_exp - host_row[target];
            queue.fill(losses_ptr + b, loss, 1).wait();
        }

        if (reduction == "mean") {
            std::vector<double> host_losses(batch_size);
            queue.memcpy(host_losses.data(), losses_ptr, batch_size * sizeof(double)).wait();

            double mean_loss = 0.0;
            for (int64_t i = 0; i < batch_size; ++i) {
                mean_loss += host_losses[i];
            }
            mean_loss /= static_cast<double>(batch_size);

            Tensor result({1}, logits.dtype(), logits.device());
            queue.fill(get_data_ptr<double>(result), mean_loss, 1).wait();
            return result;
        }
        else if (reduction == "sum") {
            std::vector<double> host_losses(batch_size);
            queue.memcpy(host_losses.data(), losses_ptr, batch_size * sizeof(double)).wait();

            double sum_loss = 0.0;
            for (int64_t i = 0; i < batch_size; ++i) {
                sum_loss += host_losses[i];
            }

            Tensor result({1}, logits.dtype(), logits.device());
            queue.fill(get_data_ptr<double>(result), sum_loss, 1).wait();
            return result;
        }
    }
    else if (logits.dtype() == DType::Float16) {
        // Float16: compute in float32 for numerical stability
        const sycl::half* logits_ptr = get_data_ptr<const sycl::half>(logits);
        const int64_t* targets_ptr = get_data_ptr<const int64_t>(targets);
        sycl::half* losses_ptr = get_data_ptr<sycl::half>(losses);

        for (int64_t b = 0; b < batch_size; ++b) {
            const sycl::half* row = logits_ptr + b * num_classes;
            int64_t target = targets_ptr[b];

            std::vector<sycl::half> host_row(num_classes);
            queue.memcpy(host_row.data(), row, num_classes * sizeof(sycl::half)).wait();

            float max_val = static_cast<float>(host_row[0]);
            for (int64_t i = 1; i < num_classes; ++i) {
                float v = static_cast<float>(host_row[i]);
                if (v > max_val) max_val = v;
            }

            float sum_exp = 0.0f;
            for (int64_t i = 0; i < num_classes; ++i) {
                sum_exp += std::exp(static_cast<float>(host_row[i]) - max_val);
            }

            float log_sum_exp = std::log(sum_exp) + max_val;
            float loss = log_sum_exp - static_cast<float>(host_row[target]);
            sycl::half h_loss = sycl::half(loss);
            queue.fill(losses_ptr + b, h_loss, 1).wait();
        }

        if (reduction == "mean") {
            std::vector<sycl::half> host_losses(batch_size);
            queue.memcpy(host_losses.data(), losses_ptr, batch_size * sizeof(sycl::half)).wait();

            float mean_loss = 0.0f;
            for (int64_t i = 0; i < batch_size; ++i) {
                mean_loss += static_cast<float>(host_losses[i]);
            }
            mean_loss /= static_cast<float>(batch_size);

            Tensor result({1}, logits.dtype(), logits.device());
            sycl::half h_mean = sycl::half(mean_loss);
            queue.fill(get_data_ptr<sycl::half>(result), h_mean, 1).wait();
            return result;
        }
        else if (reduction == "sum") {
            std::vector<sycl::half> host_losses(batch_size);
            queue.memcpy(host_losses.data(), losses_ptr, batch_size * sizeof(sycl::half)).wait();

            float sum_loss = 0.0f;
            for (int64_t i = 0; i < batch_size; ++i) {
                sum_loss += static_cast<float>(host_losses[i]);
            }

            Tensor result({1}, logits.dtype(), logits.device());
            sycl::half h_sum = sycl::half(sum_loss);
            queue.fill(get_data_ptr<sycl::half>(result), h_sum, 1).wait();
            return result;
        }
    }
    else {
        throw std::runtime_error("fused_softmax_cross_entropy: unsupported dtype");
    }

    return losses;
}

// ============================================================================
// Fused RMSNorm Forward
// RMSNorm: output = x * weight / sqrt(mean(x^2) + eps)
// Returns: (output, rrms) where rrms = 1/sqrt(mean(x^2) + eps)
// ============================================================================

auto fused_rms_norm_kernel(const Tensor& input, const Tensor& weight, float eps,
                            sycl::queue& queue) -> std::tuple<Tensor, Tensor> {
    auto shape = input.shape();
    int64_t norm_size = shape.back();
    int64_t batch_size = input.numel() / norm_size;

    Tensor output(std::vector<int64_t>(shape.begin(), shape.end()),
                  input.dtype(), input.device());
    Tensor rrms({batch_size}, input.dtype(), input.device());

    if (input.dtype() == DType::Float32) {
        const float* in_ptr = get_data_ptr<const float>(input);
        const float* w_ptr = get_data_ptr<const float>(weight);
        float* out_ptr = get_data_ptr<float>(output);
        float* rrms_ptr = get_data_ptr<float>(rrms);

        queue.parallel_for<FusedRMSNormKernelFloat32>(
            sycl::range<1>(batch_size),
            [=](sycl::id<1> idx) {
                int64_t b = idx[0];
                const float* row = in_ptr + b * norm_size;

                // Compute sum of squares
                float ss = 0.0f;
                for (int64_t i = 0; i < norm_size; ++i) {
                    ss += row[i] * row[i];
                }
                ss /= static_cast<float>(norm_size);

                // Compute reciprocal RMS
                float rr = 1.0f / sycl::sqrt(ss + eps);
                rrms_ptr[b] = rr;

                // Apply normalization with weight
                float* out_row = out_ptr + b * norm_size;
                for (int64_t i = 0; i < norm_size; ++i) {
                    out_row[i] = row[i] * rr * w_ptr[i];
                }
            }
        ).wait();
    }
    else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        const double* w_ptr = get_data_ptr<const double>(weight);
        double* out_ptr = get_data_ptr<double>(output);
        double* rrms_ptr = get_data_ptr<double>(rrms);

        queue.parallel_for<FusedRMSNormKernelFloat64>(
            sycl::range<1>(batch_size),
            [=](sycl::id<1> idx) {
                int64_t b = idx[0];
                const double* row = in_ptr + b * norm_size;

                double ss = 0.0;
                for (int64_t i = 0; i < norm_size; ++i) {
                    ss += row[i] * row[i];
                }
                ss /= static_cast<double>(norm_size);

                double rr = 1.0 / sycl::sqrt(ss + static_cast<double>(eps));
                rrms_ptr[b] = rr;

                double* out_row = out_ptr + b * norm_size;
                for (int64_t i = 0; i < norm_size; ++i) {
                    out_row[i] = row[i] * rr * w_ptr[i];
                }
            }
        ).wait();
    }
    else {
        throw std::runtime_error("fused_rms_norm: unsupported dtype (only Float32/Float64)");
    }

    return {output, rrms};
}

// ============================================================================
// RMSNorm Backward
// Inputs: grad_output, input, weight, rrms
// Returns: (grad_input, grad_weight)
// ============================================================================

auto rms_norm_backward_kernel(const Tensor& grad_output, const Tensor& input,
                               const Tensor& weight, const Tensor& rrms,
                               sycl::queue& queue) -> std::tuple<Tensor, Tensor> {
    auto shape = input.shape();
    int64_t norm_size = shape.back();
    int64_t batch_size = input.numel() / norm_size;

    Tensor grad_input(std::vector<int64_t>(shape.begin(), shape.end()),
                      input.dtype(), input.device());
    Tensor grad_weight({norm_size}, input.dtype(), input.device());

    // Zero-initialize grad_weight
    queue.memset(const_cast<void*>(grad_weight.data_ptr()), 0,
                 norm_size * grad_weight.dtype_size()).wait();

    if (input.dtype() == DType::Float32) {
        const float* go_ptr = get_data_ptr<const float>(grad_output);
        const float* in_ptr = get_data_ptr<const float>(input);
        const float* w_ptr = get_data_ptr<const float>(weight);
        const float* rrms_ptr = get_data_ptr<const float>(rrms);
        float* gi_ptr = get_data_ptr<float>(grad_input);
        float* gw_ptr = get_data_ptr<float>(grad_weight);

        // Compute grad_input per batch element
        queue.parallel_for<FusedRMSNormBackwardKernelFloat32>(
            sycl::range<1>(batch_size),
            [=](sycl::id<1> idx) {
                int64_t b = idx[0];
                const float* go_row = go_ptr + b * norm_size;
                const float* in_row = in_ptr + b * norm_size;
                float* gi_row = gi_ptr + b * norm_size;
                float rr = rrms_ptr[b];

                // Compute dot product: sum(grad_output * weight * input)
                float dot = 0.0f;
                for (int64_t i = 0; i < norm_size; ++i) {
                    dot += go_row[i] * w_ptr[i] * in_row[i];
                }
                dot *= rr * rr / static_cast<float>(norm_size);

                // grad_input = rrms * (grad_output * weight - input * dot)
                for (int64_t i = 0; i < norm_size; ++i) {
                    gi_row[i] = rr * (go_row[i] * w_ptr[i] - in_row[i] * dot);
                }
            }
        ).wait();

        // Compute grad_weight (accumulate across batch on host)
        std::vector<float> go_host(batch_size * norm_size);
        std::vector<float> in_host(batch_size * norm_size);
        std::vector<float> rrms_host(batch_size);
        queue.memcpy(go_host.data(), go_ptr, batch_size * norm_size * sizeof(float)).wait();
        queue.memcpy(in_host.data(), in_ptr, batch_size * norm_size * sizeof(float)).wait();
        queue.memcpy(rrms_host.data(), rrms_ptr, batch_size * sizeof(float)).wait();

        std::vector<float> gw_host(norm_size, 0.0f);
        for (int64_t b = 0; b < batch_size; ++b) {
            float rr = rrms_host[b];
            for (int64_t i = 0; i < norm_size; ++i) {
                gw_host[i] += go_host[b * norm_size + i] * in_host[b * norm_size + i] * rr;
            }
        }
        queue.memcpy(gw_ptr, gw_host.data(), norm_size * sizeof(float)).wait();
    }
    else if (input.dtype() == DType::Float64) {
        const double* go_ptr = get_data_ptr<const double>(grad_output);
        const double* in_ptr = get_data_ptr<const double>(input);
        const double* w_ptr = get_data_ptr<const double>(weight);
        const double* rrms_ptr = get_data_ptr<const double>(rrms);
        double* gi_ptr = get_data_ptr<double>(grad_input);
        double* gw_ptr = get_data_ptr<double>(grad_weight);

        queue.parallel_for<FusedRMSNormBackwardKernelFloat64>(
            sycl::range<1>(batch_size),
            [=](sycl::id<1> idx) {
                int64_t b = idx[0];
                const double* go_row = go_ptr + b * norm_size;
                const double* in_row = in_ptr + b * norm_size;
                double* gi_row = gi_ptr + b * norm_size;
                double rr = rrms_ptr[b];

                double dot = 0.0;
                for (int64_t i = 0; i < norm_size; ++i) {
                    dot += go_row[i] * w_ptr[i] * in_row[i];
                }
                dot *= rr * rr / static_cast<double>(norm_size);

                for (int64_t i = 0; i < norm_size; ++i) {
                    gi_row[i] = rr * (go_row[i] * w_ptr[i] - in_row[i] * dot);
                }
            }
        ).wait();

        std::vector<double> go_host(batch_size * norm_size);
        std::vector<double> in_host(batch_size * norm_size);
        std::vector<double> rrms_host(batch_size);
        queue.memcpy(go_host.data(), go_ptr, batch_size * norm_size * sizeof(double)).wait();
        queue.memcpy(in_host.data(), in_ptr, batch_size * norm_size * sizeof(double)).wait();
        queue.memcpy(rrms_host.data(), rrms_ptr, batch_size * sizeof(double)).wait();

        std::vector<double> gw_host(norm_size, 0.0);
        for (int64_t b = 0; b < batch_size; ++b) {
            double rr = rrms_host[b];
            for (int64_t i = 0; i < norm_size; ++i) {
                gw_host[i] += go_host[b * norm_size + i] * in_host[b * norm_size + i] * rr;
            }
        }
        queue.memcpy(gw_ptr, gw_host.data(), norm_size * sizeof(double)).wait();
    }
    else {
        throw std::runtime_error("rms_norm_backward: unsupported dtype (only Float32/Float64)");
    }

    return {grad_input, grad_weight};
}

} // namespace oneapi
} // namespace tenzor
