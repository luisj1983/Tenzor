#include "tenzor/core/tensor.hpp"
#include <sycl/sycl.hpp>
#include <cmath>
#include <limits>
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
struct FusedRMSNormKernelFloat16 {};
struct FusedRMSNormKernelBFloat16 {};
struct FusedRMSNormBackwardKernelFloat32 {};
struct FusedRMSNormBackwardKernelFloat64 {};
struct FusedRMSNormBackwardKernelFloat16 {};
struct FusedRMSNormBackwardKernelBFloat16 {};
struct FusedSoftmaxCrossEntropyKernelBFloat16 {};
struct FusedLayerNormBackwardKernelFloat16 {};
struct FlashAttentionKernelFloat32 {};
struct FlashAttentionKernelFloat64 {};
struct FlashAttentionKernelFloat16 {};
struct FlashAttentionKernelBFloat16 {};

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
        });
    }
    else if (a.dtype() == DType::Float64) {
        const double* a_ptr = get_data_ptr<const double>(a);
        const double* b_ptr = get_data_ptr<const double>(b);
        double* out_ptr = get_data_ptr<double>(output);

        queue.parallel_for<FusedAddReluKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            double sum = a_ptr[idx] + b_ptr[idx];
            out_ptr[idx] = sum > 0.0 ? sum : 0.0;
        });
    }
    else if (a.dtype() == DType::Float16) {
        const sycl::half* a_ptr = get_data_ptr<const sycl::half>(a);
        const sycl::half* b_ptr = get_data_ptr<const sycl::half>(b);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);

        queue.parallel_for<FusedAddReluKernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            float sum = static_cast<float>(a_ptr[idx]) + static_cast<float>(b_ptr[idx]);
            out_ptr[idx] = sycl::half(sum > 0.0f ? sum : 0.0f);
        });
    }
    else if (a.dtype() == DType::BFloat16) {
        const uint16_t* a_ptr = get_data_ptr<const uint16_t>(a);
        const uint16_t* b_ptr = get_data_ptr<const uint16_t>(b);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);

        queue.parallel_for<FusedAddReluKernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            float sum = bf16_to_f32(a_ptr[idx]) + bf16_to_f32(b_ptr[idx]);
            out_ptr[idx] = f32_to_bf16(sum > 0.0f ? sum : 0.0f);
        });
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
        });
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
        });
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
        });
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
        });
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

        int64_t wg_size = std::min(norm_size, static_cast<int64_t>(256));
        // Round up to power of 2 for reduction
        int64_t wg_pow2 = 1;
        while (wg_pow2 < wg_size) wg_pow2 *= 2;
        wg_pow2 = std::min(wg_pow2, static_cast<int64_t>(256));

        queue.submit([&](sycl::handler& cgh) {
            sycl::local_accessor<float, 1> local_sum(sycl::range<1>(wg_pow2), cgh);
            cgh.parallel_for<class LayerNormFwdF32>(
                sycl::nd_range<1>(batch_size * wg_pow2, wg_pow2),
                [=](sycl::nd_item<1> item) {
                int64_t b = item.get_group(0);
                int64_t lid = item.get_local_id(0);
                int64_t lsize = item.get_local_range(0);
                const float* batch_in = in_ptr + b * norm_size;
                float* batch_out = out_ptr + b * norm_size;

                // Step 1: Compute mean
                float thread_sum = 0.0f;
                for (int64_t i = lid; i < norm_size; i += lsize) {
                    thread_sum += batch_in[i];
                }
                local_sum[lid] = thread_sum;
                item.barrier(sycl::access::fence_space::local_space);

                for (int64_t stride = lsize / 2; stride > 0; stride >>= 1) {
                    if (lid < stride) {
                        local_sum[lid] += local_sum[lid + stride];
                    }
                    item.barrier(sycl::access::fence_space::local_space);
                }
                float batch_mean = local_sum[0] / static_cast<float>(norm_size);
                item.barrier(sycl::access::fence_space::local_space);

                // Step 2: Compute variance
                float thread_var = 0.0f;
                for (int64_t i = lid; i < norm_size; i += lsize) {
                    float diff = batch_in[i] - batch_mean;
                    thread_var += diff * diff;
                }
                local_sum[lid] = thread_var;
                item.barrier(sycl::access::fence_space::local_space);

                for (int64_t stride = lsize / 2; stride > 0; stride >>= 1) {
                    if (lid < stride) {
                        local_sum[lid] += local_sum[lid + stride];
                    }
                    item.barrier(sycl::access::fence_space::local_space);
                }
                float variance = local_sum[0] / static_cast<float>(norm_size);
                float batch_inv_std = 1.0f / sycl::sqrt(variance + epsilon);

                // Store stats
                if (lid == 0) {
                    mean_ptr[b] = batch_mean;
                    inv_std_ptr[b] = batch_inv_std;
                }
                item.barrier(sycl::access::fence_space::local_space);

                // Step 3: Normalize
                for (int64_t i = lid; i < norm_size; i += lsize) {
                    float normalized = (batch_in[i] - batch_mean) * batch_inv_std;
                    batch_out[i] = normalized * weight_ptr[i] + bias_ptr[i];
                }
            });
        });
    }
    else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        const double* weight_ptr = get_data_ptr<const double>(weight);
        const double* bias_ptr = get_data_ptr<const double>(bias);
        double* out_ptr = get_data_ptr<double>(output);
        double* mean_ptr = get_data_ptr<double>(mean);
        double* inv_std_ptr = get_data_ptr<double>(inv_std);

        int64_t wg_size = std::min(norm_size, static_cast<int64_t>(256));
        int64_t wg_pow2 = 1;
        while (wg_pow2 < wg_size) wg_pow2 *= 2;
        wg_pow2 = std::min(wg_pow2, static_cast<int64_t>(256));

        queue.submit([&](sycl::handler& cgh) {
            sycl::local_accessor<double, 1> local_sum(sycl::range<1>(wg_pow2), cgh);
            cgh.parallel_for<class LayerNormFwdF64>(
                sycl::nd_range<1>(batch_size * wg_pow2, wg_pow2),
                [=](sycl::nd_item<1> item) {
                int64_t b = item.get_group(0);
                int64_t lid = item.get_local_id(0);
                int64_t lsize = item.get_local_range(0);
                const double* batch_in = in_ptr + b * norm_size;
                double* batch_out = out_ptr + b * norm_size;

                // Step 1: Compute mean
                double thread_sum = 0.0;
                for (int64_t i = lid; i < norm_size; i += lsize) {
                    thread_sum += batch_in[i];
                }
                local_sum[lid] = thread_sum;
                item.barrier(sycl::access::fence_space::local_space);

                for (int64_t stride = lsize / 2; stride > 0; stride >>= 1) {
                    if (lid < stride) {
                        local_sum[lid] += local_sum[lid + stride];
                    }
                    item.barrier(sycl::access::fence_space::local_space);
                }
                double batch_mean = local_sum[0] / static_cast<double>(norm_size);
                item.barrier(sycl::access::fence_space::local_space);

                // Step 2: Compute variance
                double thread_var = 0.0;
                for (int64_t i = lid; i < norm_size; i += lsize) {
                    double diff = batch_in[i] - batch_mean;
                    thread_var += diff * diff;
                }
                local_sum[lid] = thread_var;
                item.barrier(sycl::access::fence_space::local_space);

                for (int64_t stride = lsize / 2; stride > 0; stride >>= 1) {
                    if (lid < stride) {
                        local_sum[lid] += local_sum[lid + stride];
                    }
                    item.barrier(sycl::access::fence_space::local_space);
                }
                double variance = local_sum[0] / static_cast<double>(norm_size);
                double batch_inv_std = 1.0 / sycl::sqrt(variance + static_cast<double>(epsilon));

                if (lid == 0) {
                    mean_ptr[b] = batch_mean;
                    inv_std_ptr[b] = batch_inv_std;
                }
                item.barrier(sycl::access::fence_space::local_space);

                // Step 3: Normalize
                for (int64_t i = lid; i < norm_size; i += lsize) {
                    double normalized = (batch_in[i] - batch_mean) * batch_inv_std;
                    batch_out[i] = normalized * weight_ptr[i] + bias_ptr[i];
                }
            });
        });
    }
    else if (input.dtype() == DType::Float16) {
        // Float16: use float32 accumulation for numerical stability
        const sycl::half* in_ptr = get_data_ptr<const sycl::half>(input);
        const sycl::half* weight_ptr = get_data_ptr<const sycl::half>(weight);
        const sycl::half* bias_ptr = get_data_ptr<const sycl::half>(bias);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);
        sycl::half* mean_ptr = get_data_ptr<sycl::half>(mean);
        sycl::half* inv_std_ptr = get_data_ptr<sycl::half>(inv_std);

        int64_t wg_size = std::min(norm_size, static_cast<int64_t>(256));
        int64_t wg_pow2 = 1;
        while (wg_pow2 < wg_size) wg_pow2 *= 2;
        wg_pow2 = std::min(wg_pow2, static_cast<int64_t>(256));

        queue.submit([&](sycl::handler& cgh) {
            sycl::local_accessor<float, 1> local_sum(sycl::range<1>(wg_pow2), cgh);
            cgh.parallel_for<class LayerNormFwdF16>(
                sycl::nd_range<1>(batch_size * wg_pow2, wg_pow2),
                [=](sycl::nd_item<1> item) {
                int64_t b = item.get_group(0);
                int64_t lid = item.get_local_id(0);
                int64_t lsize = item.get_local_range(0);
                const sycl::half* batch_in = in_ptr + b * norm_size;
                sycl::half* batch_out = out_ptr + b * norm_size;

                // Step 1: Compute mean (float32 accumulation)
                float thread_sum = 0.0f;
                for (int64_t i = lid; i < norm_size; i += lsize) {
                    thread_sum += static_cast<float>(batch_in[i]);
                }
                local_sum[lid] = thread_sum;
                item.barrier(sycl::access::fence_space::local_space);

                for (int64_t stride = lsize / 2; stride > 0; stride >>= 1) {
                    if (lid < stride) {
                        local_sum[lid] += local_sum[lid + stride];
                    }
                    item.barrier(sycl::access::fence_space::local_space);
                }
                float batch_mean = local_sum[0] / static_cast<float>(norm_size);
                item.barrier(sycl::access::fence_space::local_space);

                // Step 2: Compute variance (float32 accumulation)
                float thread_var = 0.0f;
                for (int64_t i = lid; i < norm_size; i += lsize) {
                    float diff = static_cast<float>(batch_in[i]) - batch_mean;
                    thread_var += diff * diff;
                }
                local_sum[lid] = thread_var;
                item.barrier(sycl::access::fence_space::local_space);

                for (int64_t stride = lsize / 2; stride > 0; stride >>= 1) {
                    if (lid < stride) {
                        local_sum[lid] += local_sum[lid + stride];
                    }
                    item.barrier(sycl::access::fence_space::local_space);
                }
                float variance = local_sum[0] / static_cast<float>(norm_size);
                float batch_inv_std = 1.0f / sycl::sqrt(variance + epsilon);

                if (lid == 0) {
                    mean_ptr[b] = sycl::half(batch_mean);
                    inv_std_ptr[b] = sycl::half(batch_inv_std);
                }
                item.barrier(sycl::access::fence_space::local_space);

                // Step 3: Normalize
                for (int64_t i = lid; i < norm_size; i += lsize) {
                    float val = static_cast<float>(batch_in[i]);
                    float normalized = (val - batch_mean) * batch_inv_std;
                    float result = normalized * static_cast<float>(weight_ptr[i]) + static_cast<float>(bias_ptr[i]);
                    batch_out[i] = sycl::half(result);
                }
            });
        });
    }
    else if (input.dtype() == DType::BFloat16) {
        // BFloat16: use float32 accumulation for numerical stability
        const uint16_t* in_ptr = get_data_ptr<const uint16_t>(input);
        const uint16_t* weight_ptr = get_data_ptr<const uint16_t>(weight);
        const uint16_t* bias_ptr = get_data_ptr<const uint16_t>(bias);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);
        uint16_t* mean_ptr = get_data_ptr<uint16_t>(mean);
        uint16_t* inv_std_ptr = get_data_ptr<uint16_t>(inv_std);

        int64_t wg_size = std::min(norm_size, static_cast<int64_t>(256));
        int64_t wg_pow2 = 1;
        while (wg_pow2 < wg_size) wg_pow2 *= 2;
        wg_pow2 = std::min(wg_pow2, static_cast<int64_t>(256));

        queue.submit([&](sycl::handler& cgh) {
            sycl::local_accessor<float, 1> local_sum(sycl::range<1>(wg_pow2), cgh);
            cgh.parallel_for<class LayerNormFwdBF16>(
                sycl::nd_range<1>(batch_size * wg_pow2, wg_pow2),
                [=](sycl::nd_item<1> item) {
                int64_t b = item.get_group(0);
                int64_t lid = item.get_local_id(0);
                int64_t lsize = item.get_local_range(0);
                const uint16_t* batch_in = in_ptr + b * norm_size;
                uint16_t* batch_out = out_ptr + b * norm_size;

                // Step 1: Compute mean (float32 accumulation)
                float thread_sum = 0.0f;
                for (int64_t i = lid; i < norm_size; i += lsize) {
                    thread_sum += bf16_to_f32(batch_in[i]);
                }
                local_sum[lid] = thread_sum;
                item.barrier(sycl::access::fence_space::local_space);

                for (int64_t stride = lsize / 2; stride > 0; stride >>= 1) {
                    if (lid < stride) {
                        local_sum[lid] += local_sum[lid + stride];
                    }
                    item.barrier(sycl::access::fence_space::local_space);
                }
                float batch_mean = local_sum[0] / static_cast<float>(norm_size);
                item.barrier(sycl::access::fence_space::local_space);

                // Step 2: Compute variance (float32 accumulation)
                float thread_var = 0.0f;
                for (int64_t i = lid; i < norm_size; i += lsize) {
                    float diff = bf16_to_f32(batch_in[i]) - batch_mean;
                    thread_var += diff * diff;
                }
                local_sum[lid] = thread_var;
                item.barrier(sycl::access::fence_space::local_space);

                for (int64_t stride = lsize / 2; stride > 0; stride >>= 1) {
                    if (lid < stride) {
                        local_sum[lid] += local_sum[lid + stride];
                    }
                    item.barrier(sycl::access::fence_space::local_space);
                }
                float variance = local_sum[0] / static_cast<float>(norm_size);
                float batch_inv_std = 1.0f / sycl::sqrt(variance + epsilon);

                if (lid == 0) {
                    mean_ptr[b] = f32_to_bf16(batch_mean);
                    inv_std_ptr[b] = f32_to_bf16(batch_inv_std);
                }
                item.barrier(sycl::access::fence_space::local_space);

                // Step 3: Normalize
                for (int64_t i = lid; i < norm_size; i += lsize) {
                    float val = bf16_to_f32(batch_in[i]);
                    float normalized = (val - batch_mean) * batch_inv_std;
                    float result = normalized * bf16_to_f32(weight_ptr[i]) + bf16_to_f32(bias_ptr[i]);
                    batch_out[i] = f32_to_bf16(result);
                }
            });
        });
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
        queue.fill(grad_weight_ptr, 0.0f, norm_size);
        queue.fill(grad_bias_ptr, 0.0f, norm_size);

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

            queue.memcpy(grad_weight_ptr, curr_gw.data(), norm_size * sizeof(float));
            queue.memcpy(grad_bias_ptr, curr_gb.data(), norm_size * sizeof(float));
        }

        // Full layer norm backward: compute grad_input with mean/variance corrections.
        // Copy weight to host (constant across batch)
        std::vector<float> host_w_f32(norm_size);
        queue.memcpy(host_w_f32.data(), weight_ptr, norm_size * sizeof(float)).wait();

        for (int64_t b = 0; b < batch_size; ++b) {
            const float* batch_grad = grad_out_ptr + b * norm_size;
            const float* batch_in = in_ptr + b * norm_size;
            float* batch_grad_in = grad_in_ptr + b * norm_size;

            float batch_mean, batch_inv_std;
            queue.memcpy(&batch_mean, mean_ptr + b, sizeof(float)).wait();
            queue.memcpy(&batch_inv_std, inv_std_ptr + b, sizeof(float)).wait();

            std::vector<float> host_grad_f32(norm_size), host_in_f32(norm_size);
            queue.memcpy(host_grad_f32.data(), batch_grad, norm_size * sizeof(float)).wait();
            queue.memcpy(host_in_f32.data(), batch_in, norm_size * sizeof(float)).wait();

            float ds = 0.0f, db = 0.0f;
            for (int64_t i = 0; i < norm_size; ++i) {
                float normalized = (host_in_f32[i] - batch_mean) * batch_inv_std;
                float go_w = host_grad_f32[i] * host_w_f32[i];
                ds += go_w * normalized;
                db += go_w;
            }

            float inv_n = 1.0f / static_cast<float>(norm_size);
            queue.parallel_for<class LayerNormBackward>(sycl::range<1>(norm_size), [=](sycl::id<1> idx) {
                int64_t i = idx[0];
                float normalized = (batch_in[i] - batch_mean) * batch_inv_std;
                batch_grad_in[i] = batch_inv_std * weight_ptr[i] *
                    (batch_grad[i] - inv_n * (db + normalized * ds));
            });
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

        // Complete host-based backward for Float64
        queue.wait();  // Ensure all prior work is complete

        // Copy all needed data to host
        std::vector<double> h_go(batch_size * norm_size), h_in(batch_size * norm_size);
        std::vector<double> h_mean(batch_size), h_inv_std(batch_size);
        std::vector<double> h_weight(norm_size);
        queue.memcpy(h_go.data(), grad_out_ptr, batch_size * norm_size * sizeof(double)).wait();
        queue.memcpy(h_in.data(), in_ptr, batch_size * norm_size * sizeof(double)).wait();
        queue.memcpy(h_mean.data(), mean_ptr, batch_size * sizeof(double)).wait();
        queue.memcpy(h_inv_std.data(), inv_std_ptr, batch_size * sizeof(double)).wait();
        queue.memcpy(h_weight.data(), weight_ptr, norm_size * sizeof(double)).wait();

        // Compute on host
        std::vector<double> h_gi(batch_size * norm_size);
        std::vector<double> h_gw(norm_size, 0.0), h_gb(norm_size, 0.0);

        for (int64_t b = 0; b < batch_size; ++b) {
            const double* go_b = h_go.data() + b * norm_size;
            const double* in_b = h_in.data() + b * norm_size;
            double* gi_b = h_gi.data() + b * norm_size;
            double m = h_mean[b];
            double rstd = h_inv_std[b];

            // Dot products for the full gradient formula
            double ds = 0.0, db_val = 0.0;
            for (int64_t j = 0; j < norm_size; ++j) {
                double normalized = (in_b[j] - m) * rstd;
                double go_w = go_b[j] * h_weight[j];
                ds += go_w * normalized;
                db_val += go_w;
                h_gw[j] += go_b[j] * normalized;
                h_gb[j] += go_b[j];
            }

            double inv_n = 1.0 / static_cast<double>(norm_size);
            for (int64_t j = 0; j < norm_size; ++j) {
                double normalized = (in_b[j] - m) * rstd;
                gi_b[j] = rstd * h_weight[j] * (go_b[j] - inv_n * (db_val + normalized * ds));
            }
        }

        // Copy results back to device
        queue.memcpy(grad_in_ptr, h_gi.data(), batch_size * norm_size * sizeof(double)).wait();
        queue.memcpy(grad_weight_ptr, h_gw.data(), norm_size * sizeof(double)).wait();
        queue.memcpy(grad_bias_ptr, h_gb.data(), norm_size * sizeof(double)).wait();
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
        queue.fill(grad_weight_ptr, bf16_zero, norm_size);
        queue.fill(grad_bias_ptr, bf16_zero, norm_size);

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

            queue.memcpy(grad_weight_ptr, curr_gw.data(), norm_size * sizeof(uint16_t));
            queue.memcpy(grad_bias_ptr, curr_gb.data(), norm_size * sizeof(uint16_t));
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
            });
        }
    }
    else if (input.dtype() == DType::Float16) {
        const sycl::half* grad_out_ptr = get_data_ptr<const sycl::half>(grad_output);
        const sycl::half* in_ptr = get_data_ptr<const sycl::half>(input);
        const sycl::half* mean_ptr = get_data_ptr<const sycl::half>(mean);
        const sycl::half* inv_std_ptr = get_data_ptr<const sycl::half>(inv_std);
        const sycl::half* weight_ptr = get_data_ptr<const sycl::half>(weight);
        sycl::half* grad_in_ptr = get_data_ptr<sycl::half>(grad_input);
        sycl::half* grad_weight_ptr = get_data_ptr<sycl::half>(grad_weight);
        sycl::half* grad_bias_ptr = get_data_ptr<sycl::half>(grad_bias);

        // Initialize grad_weight and grad_bias to zero
        sycl::half h_zero = sycl::half(0.0f);
        queue.fill(grad_weight_ptr, h_zero, norm_size);
        queue.fill(grad_bias_ptr, h_zero, norm_size);

        // Accumulate gradients for weight and bias (in float32 on host)
        for (int64_t b = 0; b < batch_size; ++b) {
            const sycl::half* batch_grad = grad_out_ptr + b * norm_size;
            const sycl::half* batch_in = in_ptr + b * norm_size;

            std::vector<sycl::half> host_grad(norm_size);
            std::vector<sycl::half> host_in(norm_size);
            queue.memcpy(host_grad.data(), batch_grad, norm_size * sizeof(sycl::half)).wait();
            queue.memcpy(host_in.data(), batch_in, norm_size * sizeof(sycl::half)).wait();

            sycl::half h_batch_mean, h_batch_inv_std;
            queue.memcpy(&h_batch_mean, mean_ptr + b, sizeof(sycl::half)).wait();
            queue.memcpy(&h_batch_inv_std, inv_std_ptr + b, sizeof(sycl::half)).wait();
            float batch_mean = static_cast<float>(h_batch_mean);
            float batch_inv_std = static_cast<float>(h_batch_inv_std);

            std::vector<sycl::half> curr_gw(norm_size), curr_gb(norm_size);
            queue.memcpy(curr_gw.data(), grad_weight_ptr, norm_size * sizeof(sycl::half)).wait();
            queue.memcpy(curr_gb.data(), grad_bias_ptr, norm_size * sizeof(sycl::half)).wait();

            for (int64_t i = 0; i < norm_size; ++i) {
                float normalized = (static_cast<float>(host_in[i]) - batch_mean) * batch_inv_std;
                float gw = static_cast<float>(curr_gw[i]) + static_cast<float>(host_grad[i]) * normalized;
                float gb = static_cast<float>(curr_gb[i]) + static_cast<float>(host_grad[i]);
                curr_gw[i] = sycl::half(gw);
                curr_gb[i] = sycl::half(gb);
            }

            queue.memcpy(grad_weight_ptr, curr_gw.data(), norm_size * sizeof(sycl::half));
            queue.memcpy(grad_bias_ptr, curr_gb.data(), norm_size * sizeof(sycl::half));
        }

        // Full layer norm backward: compute grad_input with mean/variance corrections.
        // Copy weight to host (constant across batch)
        std::vector<sycl::half> h_weight(norm_size);
        queue.memcpy(h_weight.data(), weight_ptr, norm_size * sizeof(sycl::half)).wait();

        for (int64_t b = 0; b < batch_size; ++b) {
            const sycl::half* batch_grad = grad_out_ptr + b * norm_size;
            const sycl::half* batch_in = in_ptr + b * norm_size;
            sycl::half* batch_grad_in = grad_in_ptr + b * norm_size;

            sycl::half h_batch_mean, h_batch_inv_std;
            queue.memcpy(&h_batch_mean, mean_ptr + b, sizeof(sycl::half)).wait();
            queue.memcpy(&h_batch_inv_std, inv_std_ptr + b, sizeof(sycl::half)).wait();
            float batch_mean = static_cast<float>(h_batch_mean);
            float batch_inv_std = static_cast<float>(h_batch_inv_std);

            std::vector<sycl::half> h_grad(norm_size), h_in(norm_size);
            queue.memcpy(h_grad.data(), batch_grad, norm_size * sizeof(sycl::half)).wait();
            queue.memcpy(h_in.data(), batch_in, norm_size * sizeof(sycl::half)).wait();

            float ds = 0.0f, db_val = 0.0f;
            for (int64_t i = 0; i < norm_size; ++i) {
                float normalized = (static_cast<float>(h_in[i]) - batch_mean) * batch_inv_std;
                float go_w = static_cast<float>(h_grad[i]) * static_cast<float>(h_weight[i]);
                ds += go_w * normalized;
                db_val += go_w;
            }

            float inv_n = 1.0f / static_cast<float>(norm_size);
            queue.parallel_for<FusedLayerNormBackwardKernelFloat16>(sycl::range<1>(norm_size), [=](sycl::id<1> idx) {
                int64_t i = idx[0];
                float in_val = static_cast<float>(batch_in[i]);
                float normalized = (in_val - batch_mean) * batch_inv_std;
                float grad = static_cast<float>(batch_grad[i]);
                float w = static_cast<float>(weight_ptr[i]);
                batch_grad_in[i] = sycl::half(batch_inv_std * w *
                    (grad - inv_n * (db_val + normalized * ds)));
            });
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
        );
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
        );
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
        );
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
        );
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
        );
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
        );
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
        );
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
        );
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
        );
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
        );
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
        );
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
        );
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

            queue.fill(losses_ptr + b, loss, 1);
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
            queue.fill(get_data_ptr<float>(result), mean_loss, 1);
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
            queue.fill(get_data_ptr<float>(result), sum_loss, 1);
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
            queue.fill(losses_ptr + b, loss, 1);
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
            queue.fill(get_data_ptr<double>(result), mean_loss, 1);
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
            queue.fill(get_data_ptr<double>(result), sum_loss, 1);
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
            queue.fill(losses_ptr + b, h_loss, 1);
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
            queue.fill(get_data_ptr<sycl::half>(result), h_mean, 1);
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
            queue.fill(get_data_ptr<sycl::half>(result), h_sum, 1);
            return result;
        }
    }
    else if (logits.dtype() == DType::BFloat16) {
        // BFloat16: compute in float32 for numerical stability
        const uint16_t* logits_ptr = get_data_ptr<const uint16_t>(logits);
        const int64_t* targets_ptr = get_data_ptr<const int64_t>(targets);
        uint16_t* losses_ptr = get_data_ptr<uint16_t>(losses);

        for (int64_t b = 0; b < batch_size; ++b) {
            const uint16_t* row = logits_ptr + b * num_classes;
            int64_t target = targets_ptr[b];

            std::vector<uint16_t> host_row(num_classes);
            queue.memcpy(host_row.data(), row, num_classes * sizeof(uint16_t)).wait();

            float max_val = bf16_to_f32(host_row[0]);
            for (int64_t i = 1; i < num_classes; ++i) {
                float v = bf16_to_f32(host_row[i]);
                if (v > max_val) max_val = v;
            }

            float sum_exp = 0.0f;
            for (int64_t i = 0; i < num_classes; ++i) {
                sum_exp += std::exp(bf16_to_f32(host_row[i]) - max_val);
            }

            float log_sum_exp = std::log(sum_exp) + max_val;
            float loss = log_sum_exp - bf16_to_f32(host_row[target]);
            uint16_t bf_loss = f32_to_bf16(loss);
            queue.fill(losses_ptr + b, bf_loss, 1);
        }

        if (reduction == "mean") {
            std::vector<uint16_t> host_losses(batch_size);
            queue.memcpy(host_losses.data(), losses_ptr, batch_size * sizeof(uint16_t)).wait();

            float mean_loss = 0.0f;
            for (int64_t i = 0; i < batch_size; ++i) {
                mean_loss += bf16_to_f32(host_losses[i]);
            }
            mean_loss /= static_cast<float>(batch_size);

            Tensor result({1}, logits.dtype(), logits.device());
            uint16_t bf_mean = f32_to_bf16(mean_loss);
            queue.fill(get_data_ptr<uint16_t>(result), bf_mean, 1);
            return result;
        }
        else if (reduction == "sum") {
            std::vector<uint16_t> host_losses(batch_size);
            queue.memcpy(host_losses.data(), losses_ptr, batch_size * sizeof(uint16_t)).wait();

            float sum_loss = 0.0f;
            for (int64_t i = 0; i < batch_size; ++i) {
                sum_loss += bf16_to_f32(host_losses[i]);
            }

            Tensor result({1}, logits.dtype(), logits.device());
            uint16_t bf_sum = f32_to_bf16(sum_loss);
            queue.fill(get_data_ptr<uint16_t>(result), bf_sum, 1);
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
        );
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
        );
    }
    else if (input.dtype() == DType::Float16) {
        const sycl::half* in_ptr = get_data_ptr<const sycl::half>(input);
        const sycl::half* w_ptr = get_data_ptr<const sycl::half>(weight);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);
        sycl::half* rrms_ptr = get_data_ptr<sycl::half>(rrms);

        queue.parallel_for<FusedRMSNormKernelFloat16>(
            sycl::range<1>(batch_size),
            [=](sycl::id<1> idx) {
                int64_t b = idx[0];
                const sycl::half* row = in_ptr + b * norm_size;

                float ss = 0.0f;
                for (int64_t i = 0; i < norm_size; ++i) {
                    float v = static_cast<float>(row[i]);
                    ss += v * v;
                }
                ss /= static_cast<float>(norm_size);

                float rr = 1.0f / sycl::sqrt(ss + eps);
                rrms_ptr[b] = sycl::half(rr);

                sycl::half* out_row = out_ptr + b * norm_size;
                for (int64_t i = 0; i < norm_size; ++i) {
                    out_row[i] = sycl::half(static_cast<float>(row[i]) * rr * static_cast<float>(w_ptr[i]));
                }
            }
        );
    }
    else if (input.dtype() == DType::BFloat16) {
        const uint16_t* in_ptr = get_data_ptr<const uint16_t>(input);
        const uint16_t* w_ptr = get_data_ptr<const uint16_t>(weight);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);
        uint16_t* rrms_ptr = get_data_ptr<uint16_t>(rrms);

        queue.parallel_for<FusedRMSNormKernelBFloat16>(
            sycl::range<1>(batch_size),
            [=](sycl::id<1> idx) {
                int64_t b = idx[0];
                const uint16_t* row = in_ptr + b * norm_size;

                float ss = 0.0f;
                for (int64_t i = 0; i < norm_size; ++i) {
                    float v = bf16_to_f32(row[i]);
                    ss += v * v;
                }
                ss /= static_cast<float>(norm_size);

                float rr = 1.0f / sycl::sqrt(ss + eps);
                rrms_ptr[b] = f32_to_bf16(rr);

                uint16_t* out_row = out_ptr + b * norm_size;
                for (int64_t i = 0; i < norm_size; ++i) {
                    out_row[i] = f32_to_bf16(bf16_to_f32(row[i]) * rr * bf16_to_f32(w_ptr[i]));
                }
            }
        );
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
                 norm_size * grad_weight.dtype_size());

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
        );

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
        );

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
    else if (input.dtype() == DType::Float16) {
        const sycl::half* go_ptr = get_data_ptr<const sycl::half>(grad_output);
        const sycl::half* in_ptr = get_data_ptr<const sycl::half>(input);
        const sycl::half* w_ptr = get_data_ptr<const sycl::half>(weight);
        const sycl::half* rrms_ptr = get_data_ptr<const sycl::half>(rrms);
        sycl::half* gi_ptr = get_data_ptr<sycl::half>(grad_input);
        sycl::half* gw_ptr = get_data_ptr<sycl::half>(grad_weight);

        // Compute grad_input per batch element
        queue.parallel_for<FusedRMSNormBackwardKernelFloat16>(
            sycl::range<1>(batch_size),
            [=](sycl::id<1> idx) {
                int64_t b = idx[0];
                const sycl::half* go_row = go_ptr + b * norm_size;
                const sycl::half* in_row = in_ptr + b * norm_size;
                sycl::half* gi_row = gi_ptr + b * norm_size;
                float rr = static_cast<float>(rrms_ptr[b]);

                float dot = 0.0f;
                for (int64_t i = 0; i < norm_size; ++i) {
                    dot += static_cast<float>(go_row[i]) * static_cast<float>(w_ptr[i]) * static_cast<float>(in_row[i]);
                }
                dot *= rr * rr / static_cast<float>(norm_size);

                for (int64_t i = 0; i < norm_size; ++i) {
                    gi_row[i] = sycl::half(rr * (static_cast<float>(go_row[i]) * static_cast<float>(w_ptr[i]) - static_cast<float>(in_row[i]) * dot));
                }
            }
        );

        // Compute grad_weight (accumulate across batch on host)
        std::vector<sycl::half> go_host(batch_size * norm_size);
        std::vector<sycl::half> in_host(batch_size * norm_size);
        std::vector<sycl::half> rrms_host(batch_size);
        queue.memcpy(go_host.data(), go_ptr, batch_size * norm_size * sizeof(sycl::half)).wait();
        queue.memcpy(in_host.data(), in_ptr, batch_size * norm_size * sizeof(sycl::half)).wait();
        queue.memcpy(rrms_host.data(), rrms_ptr, batch_size * sizeof(sycl::half)).wait();

        std::vector<float> gw_host(norm_size, 0.0f);
        for (int64_t b = 0; b < batch_size; ++b) {
            float rr = static_cast<float>(rrms_host[b]);
            for (int64_t i = 0; i < norm_size; ++i) {
                gw_host[i] += static_cast<float>(go_host[b * norm_size + i]) * static_cast<float>(in_host[b * norm_size + i]) * rr;
            }
        }
        std::vector<sycl::half> gw_half(norm_size);
        for (int64_t i = 0; i < norm_size; ++i) gw_half[i] = sycl::half(gw_host[i]);
        queue.memcpy(gw_ptr, gw_half.data(), norm_size * sizeof(sycl::half)).wait();
    }
    else if (input.dtype() == DType::BFloat16) {
        const uint16_t* go_ptr = get_data_ptr<const uint16_t>(grad_output);
        const uint16_t* in_ptr = get_data_ptr<const uint16_t>(input);
        const uint16_t* w_ptr = get_data_ptr<const uint16_t>(weight);
        const uint16_t* rrms_ptr = get_data_ptr<const uint16_t>(rrms);
        uint16_t* gi_ptr = get_data_ptr<uint16_t>(grad_input);
        uint16_t* gw_ptr = get_data_ptr<uint16_t>(grad_weight);

        queue.parallel_for<FusedRMSNormBackwardKernelBFloat16>(
            sycl::range<1>(batch_size),
            [=](sycl::id<1> idx) {
                int64_t b = idx[0];
                const uint16_t* go_row = go_ptr + b * norm_size;
                const uint16_t* in_row = in_ptr + b * norm_size;
                uint16_t* gi_row = gi_ptr + b * norm_size;
                float rr = bf16_to_f32(rrms_ptr[b]);

                float dot = 0.0f;
                for (int64_t i = 0; i < norm_size; ++i) {
                    dot += bf16_to_f32(go_row[i]) * bf16_to_f32(w_ptr[i]) * bf16_to_f32(in_row[i]);
                }
                dot *= rr * rr / static_cast<float>(norm_size);

                for (int64_t i = 0; i < norm_size; ++i) {
                    gi_row[i] = f32_to_bf16(rr * (bf16_to_f32(go_row[i]) * bf16_to_f32(w_ptr[i]) - bf16_to_f32(in_row[i]) * dot));
                }
            }
        );

        // Compute grad_weight (accumulate across batch on host)
        std::vector<uint16_t> go_host(batch_size * norm_size);
        std::vector<uint16_t> in_host(batch_size * norm_size);
        std::vector<uint16_t> rrms_host(batch_size);
        queue.memcpy(go_host.data(), go_ptr, batch_size * norm_size * sizeof(uint16_t)).wait();
        queue.memcpy(in_host.data(), in_ptr, batch_size * norm_size * sizeof(uint16_t)).wait();
        queue.memcpy(rrms_host.data(), rrms_ptr, batch_size * sizeof(uint16_t)).wait();

        std::vector<float> gw_host(norm_size, 0.0f);
        for (int64_t b = 0; b < batch_size; ++b) {
            float rr = bf16_to_f32(rrms_host[b]);
            for (int64_t i = 0; i < norm_size; ++i) {
                gw_host[i] += bf16_to_f32(go_host[b * norm_size + i]) * bf16_to_f32(in_host[b * norm_size + i]) * rr;
            }
        }
        std::vector<uint16_t> gw_bf16(norm_size);
        for (int64_t i = 0; i < norm_size; ++i) gw_bf16[i] = f32_to_bf16(gw_host[i]);
        queue.memcpy(gw_ptr, gw_bf16.data(), norm_size * sizeof(uint16_t)).wait();
    }
    else {
        throw std::runtime_error("rms_norm_backward: unsupported dtype (only Float32/Float64)");
    }

    return {grad_input, grad_weight};
}

// ============================================================================
// Fused Adam Optimizer Step
// ============================================================================

// SYCL Kernel name classes for fused optimizer kernels
struct FusedAdamStepKernelFloat32 {};
struct FusedAdamStepKernelFloat64 {};
struct FusedSGDStepKernelFloat32 {};
struct FusedSGDStepKernelFloat64 {};
struct FusedRMSPropStepKernelFloat32 {};
struct FusedRMSPropStepKernelFloat64 {};
struct FusedAdadeltaStepKernelFloat32 {};
struct FusedAdadeltaStepKernelFloat64 {};
struct FusedAdagradStepKernelFloat32 {};
struct FusedAdagradStepKernelFloat64 {};

auto fused_adam_step_kernel(
    Tensor& param,
    const Tensor& grad,
    Tensor& exp_avg,
    Tensor& exp_avg_sq,
    double lr,
    double beta1,
    double beta2,
    double eps,
    double weight_decay,
    int64_t step,
    bool decoupled_weight_decay,
    sycl::queue& queue,
    Tensor* max_exp_avg_sq,
    bool amsgrad
) -> void {
    const int64_t numel = param.numel();
    if (numel == 0) return;

    // Compute bias corrections in double precision for accuracy
    double bias_correction1 = 1.0 - std::pow(beta1, static_cast<double>(step));
    double bias_correction2 = 1.0 - std::pow(beta2, static_cast<double>(step));

    if (param.dtype() == DType::Float32) {
        float* param_ptr = get_data_ptr<float>(param);
        const float* grad_ptr = get_data_ptr<const float>(grad);
        float* m_ptr = get_data_ptr<float>(exp_avg);
        float* v_ptr = get_data_ptr<float>(exp_avg_sq);
        float* max_v_ptr = (amsgrad && max_exp_avg_sq) ? get_data_ptr<float>(*max_exp_avg_sq) : nullptr;

        float f_lr = static_cast<float>(lr);
        float f_beta1 = static_cast<float>(beta1);
        float f_beta2 = static_cast<float>(beta2);
        float f_eps = static_cast<float>(eps);
        float f_wd = static_cast<float>(weight_decay);
        float f_bc1 = static_cast<float>(bias_correction1);
        float f_bc2 = static_cast<float>(bias_correction2);
        bool f_decoupled = decoupled_weight_decay;
        bool f_amsgrad = amsgrad;

        queue.parallel_for<FusedAdamStepKernelFloat32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            float g = grad_ptr[idx];
            float p = param_ptr[idx];
            float m = m_ptr[idx];
            float v = v_ptr[idx];

            // Apply weight decay
            if (f_wd != 0.0f) {
                if (f_decoupled) {
                    p = p * (1.0f - f_lr * f_wd);
                } else {
                    g = g + f_wd * p;
                }
            }

            // Update biased first moment estimate
            m = f_beta1 * m + (1.0f - f_beta1) * g;
            // Update biased second raw moment estimate
            v = f_beta2 * v + (1.0f - f_beta2) * g * g;

            m_ptr[idx] = m;
            v_ptr[idx] = v;

            // Bias-corrected estimates
            float m_hat = m / f_bc1;
            float v_hat = v / f_bc2;

            // AMSGrad
            if (f_amsgrad && max_v_ptr) {
                float max_v = max_v_ptr[idx];
                if (v_hat > max_v) max_v = v_hat;
                max_v_ptr[idx] = max_v;
                v_hat = max_v;
            }

            // Update parameters
            param_ptr[idx] = p - f_lr * m_hat / (sycl::sqrt(v_hat) + f_eps);
        });
    } else if (param.dtype() == DType::Float64) {
        double* param_ptr = get_data_ptr<double>(param);
        const double* grad_ptr = get_data_ptr<const double>(grad);
        double* m_ptr = get_data_ptr<double>(exp_avg);
        double* v_ptr = get_data_ptr<double>(exp_avg_sq);
        double* max_v_ptr = (amsgrad && max_exp_avg_sq) ? get_data_ptr<double>(*max_exp_avg_sq) : nullptr;

        double d_bc1 = bias_correction1;
        double d_bc2 = bias_correction2;
        bool d_decoupled = decoupled_weight_decay;
        bool d_amsgrad = amsgrad;

        queue.parallel_for<FusedAdamStepKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            double g = grad_ptr[idx];
            double p = param_ptr[idx];
            double m = m_ptr[idx];
            double v = v_ptr[idx];

            if (weight_decay != 0.0) {
                if (d_decoupled) {
                    p = p * (1.0 - lr * weight_decay);
                } else {
                    g = g + weight_decay * p;
                }
            }

            m = beta1 * m + (1.0 - beta1) * g;
            v = beta2 * v + (1.0 - beta2) * g * g;

            m_ptr[idx] = m;
            v_ptr[idx] = v;

            double m_hat = m / d_bc1;
            double v_hat = v / d_bc2;

            if (d_amsgrad && max_v_ptr) {
                double max_v = max_v_ptr[idx];
                if (v_hat > max_v) max_v = v_hat;
                max_v_ptr[idx] = max_v;
                v_hat = max_v;
            }

            param_ptr[idx] = p - lr * m_hat / (sycl::sqrt(v_hat) + eps);
        });
    } else {
        throw std::runtime_error("fused_adam_step_kernel: Only Float32 and Float64 supported");
    }
}

// ============================================================================
// Fused SGD with Momentum
// ============================================================================

auto fused_sgd_step_kernel(
    Tensor& param,
    const Tensor& grad,
    Tensor* momentum_buffer,
    float lr,
    float momentum,
    float weight_decay,
    float dampening,
    bool nesterov,
    sycl::queue& queue
) -> void {
    const int64_t numel = param.numel();
    if (numel == 0) return;

    bool has_momentum_buffer = (momentum_buffer != nullptr && momentum > 0.0f);

    if (param.dtype() == DType::Float32) {
        float* param_ptr = get_data_ptr<float>(param);
        const float* grad_ptr = get_data_ptr<const float>(grad);
        float* mom_ptr = has_momentum_buffer ? get_data_ptr<float>(*momentum_buffer) : nullptr;

        queue.parallel_for<FusedSGDStepKernelFloat32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            float g = grad_ptr[idx];
            float p = param_ptr[idx];

            // Apply weight decay
            if (weight_decay > 0.0f) {
                g = g + weight_decay * p;
            }

            if (has_momentum_buffer && mom_ptr) {
                float v = mom_ptr[idx];
                v = momentum * v + (1.0f - dampening) * g;
                mom_ptr[idx] = v;

                if (nesterov) {
                    g = g + momentum * v;
                } else {
                    g = v;
                }
            }

            param_ptr[idx] = p - lr * g;
        });
    } else if (param.dtype() == DType::Float64) {
        double* param_ptr = get_data_ptr<double>(param);
        const double* grad_ptr = get_data_ptr<const double>(grad);
        double* mom_ptr = has_momentum_buffer ? get_data_ptr<double>(*momentum_buffer) : nullptr;

        double d_lr = static_cast<double>(lr);
        double d_momentum = static_cast<double>(momentum);
        double d_weight_decay = static_cast<double>(weight_decay);
        double d_dampening = static_cast<double>(dampening);

        queue.parallel_for<FusedSGDStepKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            double g = grad_ptr[idx];
            double p = param_ptr[idx];

            if (d_weight_decay > 0.0) {
                g = g + d_weight_decay * p;
            }

            if (has_momentum_buffer && mom_ptr) {
                double v = mom_ptr[idx];
                v = d_momentum * v + (1.0 - d_dampening) * g;
                mom_ptr[idx] = v;

                if (nesterov) {
                    g = g + d_momentum * v;
                } else {
                    g = v;
                }
            }

            param_ptr[idx] = p - d_lr * g;
        });
    } else {
        throw std::runtime_error("fused_sgd_step_kernel: Only Float32 and Float64 supported");
    }
}

// ============================================================================
// Fused RMSProp Optimizer Step
// ============================================================================

auto fused_rmsprop_step_kernel(
    Tensor& param,
    const Tensor& grad,
    Tensor& square_avg,
    Tensor* grad_avg,
    Tensor* momentum_buffer,
    float lr,
    float alpha,
    float eps,
    float weight_decay,
    float momentum,
    bool centered,
    sycl::queue& queue
) -> void {
    const int64_t numel = param.numel();
    if (numel == 0) return;

    if (param.dtype() == DType::Float32) {
        float* param_ptr = get_data_ptr<float>(param);
        const float* grad_ptr = get_data_ptr<const float>(grad);
        float* sq_ptr = get_data_ptr<float>(square_avg);
        float* ga_ptr = (centered && grad_avg) ? get_data_ptr<float>(*grad_avg) : nullptr;
        float* mom_ptr = (momentum > 0.0f && momentum_buffer) ? get_data_ptr<float>(*momentum_buffer) : nullptr;

        queue.parallel_for<FusedRMSPropStepKernelFloat32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            float g = grad_ptr[idx];

            if (weight_decay != 0.0f) {
                g = g + weight_decay * param_ptr[idx];
            }

            // Update square average: v = alpha * v + (1 - alpha) * g^2
            float sq = sq_ptr[idx];
            sq = alpha * sq + (1.0f - alpha) * g * g;
            sq_ptr[idx] = sq;

            float avg;
            if (centered && ga_ptr) {
                float ga = ga_ptr[idx];
                ga = alpha * ga + (1.0f - alpha) * g;
                ga_ptr[idx] = ga;
                avg = sycl::sqrt(sq - ga * ga + eps);
            } else {
                avg = sycl::sqrt(sq + eps);
            }

            if (momentum > 0.0f && mom_ptr) {
                float buf = mom_ptr[idx];
                buf = momentum * buf + g / avg;
                mom_ptr[idx] = buf;
                param_ptr[idx] = param_ptr[idx] - lr * buf;
            } else {
                param_ptr[idx] = param_ptr[idx] - lr * g / avg;
            }
        });
    } else if (param.dtype() == DType::Float64) {
        double* param_ptr = get_data_ptr<double>(param);
        const double* grad_ptr = get_data_ptr<const double>(grad);
        double* sq_ptr = get_data_ptr<double>(square_avg);
        double* ga_ptr = (centered && grad_avg) ? get_data_ptr<double>(*grad_avg) : nullptr;
        double* mom_ptr = (momentum > 0.0f && momentum_buffer) ? get_data_ptr<double>(*momentum_buffer) : nullptr;

        double d_lr = static_cast<double>(lr);
        double d_alpha = static_cast<double>(alpha);
        double d_eps = static_cast<double>(eps);
        double d_wd = static_cast<double>(weight_decay);
        double d_momentum = static_cast<double>(momentum);

        queue.parallel_for<FusedRMSPropStepKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            double g = grad_ptr[idx];

            if (d_wd != 0.0) {
                g = g + d_wd * param_ptr[idx];
            }

            double sq = sq_ptr[idx];
            sq = d_alpha * sq + (1.0 - d_alpha) * g * g;
            sq_ptr[idx] = sq;

            double avg;
            if (centered && ga_ptr) {
                double ga = ga_ptr[idx];
                ga = d_alpha * ga + (1.0 - d_alpha) * g;
                ga_ptr[idx] = ga;
                avg = sycl::sqrt(sq - ga * ga + d_eps);
            } else {
                avg = sycl::sqrt(sq + d_eps);
            }

            if (d_momentum > 0.0 && mom_ptr) {
                double buf = mom_ptr[idx];
                buf = d_momentum * buf + g / avg;
                mom_ptr[idx] = buf;
                param_ptr[idx] = param_ptr[idx] - d_lr * buf;
            } else {
                param_ptr[idx] = param_ptr[idx] - d_lr * g / avg;
            }
        });
    } else {
        throw std::runtime_error("fused_rmsprop_step_kernel: Only Float32 and Float64 supported");
    }
}

// ============================================================================
// Fused Adadelta Optimizer Step
// ============================================================================

auto fused_adadelta_step_kernel(
    Tensor& param,
    const Tensor& grad,
    Tensor& square_avg,
    Tensor& acc_delta,
    float rho,
    float eps,
    float lr,
    float weight_decay,
    sycl::queue& queue
) -> void {
    const int64_t numel = param.numel();
    if (numel == 0) return;

    if (param.dtype() == DType::Float32) {
        float* param_ptr = get_data_ptr<float>(param);
        const float* grad_ptr = get_data_ptr<const float>(grad);
        float* sq_ptr = get_data_ptr<float>(square_avg);
        float* ad_ptr = get_data_ptr<float>(acc_delta);

        queue.parallel_for<FusedAdadeltaStepKernelFloat32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            float g = grad_ptr[idx];
            if (weight_decay != 0.0f) {
                g = g + weight_decay * param_ptr[idx];
            }

            // v = rho * v + (1 - rho) * g^2
            float sq = sq_ptr[idx];
            sq = rho * sq + (1.0f - rho) * g * g;
            sq_ptr[idx] = sq;

            // delta = sqrt(acc_delta + eps) / sqrt(sq + eps) * g
            float std_val = sycl::sqrt(sq + eps);
            float delta = sycl::sqrt(ad_ptr[idx] + eps) / std_val * g;

            // acc_delta = rho * acc_delta + (1 - rho) * delta^2
            ad_ptr[idx] = rho * ad_ptr[idx] + (1.0f - rho) * delta * delta;

            param_ptr[idx] = param_ptr[idx] - lr * delta;
        });
    } else if (param.dtype() == DType::Float64) {
        double* param_ptr = get_data_ptr<double>(param);
        const double* grad_ptr = get_data_ptr<const double>(grad);
        double* sq_ptr = get_data_ptr<double>(square_avg);
        double* ad_ptr = get_data_ptr<double>(acc_delta);

        double d_rho = static_cast<double>(rho);
        double d_eps = static_cast<double>(eps);
        double d_lr = static_cast<double>(lr);
        double d_wd = static_cast<double>(weight_decay);

        queue.parallel_for<FusedAdadeltaStepKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            double g = grad_ptr[idx];
            if (d_wd != 0.0) {
                g = g + d_wd * param_ptr[idx];
            }

            double sq = sq_ptr[idx];
            sq = d_rho * sq + (1.0 - d_rho) * g * g;
            sq_ptr[idx] = sq;

            double std_val = sycl::sqrt(sq + d_eps);
            double delta = sycl::sqrt(ad_ptr[idx] + d_eps) / std_val * g;

            ad_ptr[idx] = d_rho * ad_ptr[idx] + (1.0 - d_rho) * delta * delta;

            param_ptr[idx] = param_ptr[idx] - d_lr * delta;
        });
    } else {
        throw std::runtime_error("fused_adadelta_step_kernel: Only Float32 and Float64 supported");
    }
}

// ============================================================================
// Fused Adagrad Optimizer Step
// ============================================================================

auto fused_adagrad_step_kernel(
    Tensor& param,
    const Tensor& grad,
    Tensor& sum_sq,
    float lr,
    float lr_decay,
    float eps,
    float weight_decay,
    int64_t step,
    sycl::queue& queue
) -> void {
    const int64_t numel = param.numel();
    if (numel == 0) return;

    if (param.dtype() == DType::Float32) {
        float* param_ptr = get_data_ptr<float>(param);
        const float* grad_ptr = get_data_ptr<const float>(grad);
        float* sq_ptr = get_data_ptr<float>(sum_sq);
        float clr = lr / (1.0f + static_cast<float>(step - 1) * lr_decay);

        queue.parallel_for<FusedAdagradStepKernelFloat32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            float g = grad_ptr[idx];
            if (weight_decay != 0.0f) {
                g = g + weight_decay * param_ptr[idx];
            }

            // sum_sq += g^2
            float sq = sq_ptr[idx] + g * g;
            sq_ptr[idx] = sq;

            // param -= clr * g / (sqrt(sum_sq) + eps)
            param_ptr[idx] = param_ptr[idx] - clr * g / (sycl::sqrt(sq) + eps);
        });
    } else if (param.dtype() == DType::Float64) {
        double* param_ptr = get_data_ptr<double>(param);
        const double* grad_ptr = get_data_ptr<const double>(grad);
        double* sq_ptr = get_data_ptr<double>(sum_sq);
        double d_lr = static_cast<double>(lr);
        double d_lr_decay = static_cast<double>(lr_decay);
        double d_eps = static_cast<double>(eps);
        double d_wd = static_cast<double>(weight_decay);
        double clr = d_lr / (1.0 + static_cast<double>(step - 1) * d_lr_decay);

        queue.parallel_for<FusedAdagradStepKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            double g = grad_ptr[idx];
            if (d_wd != 0.0) {
                g = g + d_wd * param_ptr[idx];
            }

            double sq = sq_ptr[idx] + g * g;
            sq_ptr[idx] = sq;

            param_ptr[idx] = param_ptr[idx] - clr * g / (sycl::sqrt(sq) + d_eps);
        });
    } else {
        throw std::runtime_error("fused_adagrad_step_kernel: Only Float32 and Float64 supported");
    }
}

// ============================================================================
// Flash Attention (memory-efficient tiled attention with online softmax)
// ============================================================================

// Templated implementation: ComputeT is the accumulation type (always float for half types),
// DataT is the storage type, KernelName is the SYCL kernel tag.
// IsBFloat16 enables the bf16 conversion path.
template<typename DataT, typename ComputeT, typename KernelName, bool IsBFloat16 = false>
auto flash_attention_impl(
    const Tensor& Q,    // [batch_heads, seq_len_q, head_dim]
    const Tensor& K,    // [batch_heads, seq_len_k, head_dim]
    const Tensor& V,    // [batch_heads, seq_len_k, head_dim]
    const Tensor* mask,  // optional [batch_heads, seq_len_q, seq_len_k] or broadcastable
    ComputeT scale,
    bool is_causal,
    sycl::queue& queue
) -> Tensor {
    auto q_shape = Q.shape();
    auto k_shape = K.shape();

    const int64_t batch_heads = q_shape[0];
    const int64_t seq_len_q   = q_shape[1];
    const int64_t head_dim    = q_shape[2];
    const int64_t seq_len_k   = k_shape[1];

    Tensor output(std::vector<int64_t>{batch_heads, seq_len_q, head_dim},
                  Q.dtype(), Q.device());

    // Tile size for K/V blocks
    constexpr int Bc = 32;
    // Padding stride to avoid bank conflicts in local memory
    const int K_STRIDE = static_cast<int>(head_dim) + 4;
    const int BLOCK_SIZE = 128;

    // Local memory: K_tile[Bc][K_STRIDE] + V_tile[Bc][K_STRIDE] + scores[Bc]
    const size_t local_mem_size = static_cast<size_t>(
        2 * Bc * K_STRIDE + Bc) * sizeof(ComputeT);

    const DataT* q_ptr = get_data_ptr<const DataT>(Q);
    const DataT* k_ptr = get_data_ptr<const DataT>(K);
    const DataT* v_ptr = get_data_ptr<const DataT>(V);
    DataT* o_ptr = get_data_ptr<DataT>(output);

    // 2D grid: (batch_heads, seq_len_q) — one work-group per query row
    queue.submit([&](sycl::handler& cgh) {
        sycl::local_accessor<ComputeT, 1> local_mem(local_mem_size / sizeof(ComputeT), cgh);

        const int hd = static_cast<int>(head_dim);
        const int slq = static_cast<int>(seq_len_q);
        const int slk = static_cast<int>(seq_len_k);
        const int ks = K_STRIDE;
        const bool causal = is_causal;
        const ComputeT sc = scale;

        cgh.parallel_for<KernelName>(
            sycl::nd_range<2>(
                sycl::range<2>(static_cast<size_t>(batch_heads),
                               static_cast<size_t>(seq_len_q) * BLOCK_SIZE),
                sycl::range<2>(1, BLOCK_SIZE)
            ),
            [=](sycl::nd_item<2> item) {
                const int batch_head = static_cast<int>(item.get_global_id(0));
                const int query_idx  = static_cast<int>(item.get_global_id(1)) / BLOCK_SIZE;
                const int tid        = static_cast<int>(item.get_local_id(1));

                if (query_idx >= slq) return;

                // Local memory pointers
                ComputeT* lmem = local_mem.get_pointer();
                ComputeT* K_tile = lmem;                     // [Bc][ks]
                ComputeT* V_tile = lmem + Bc * ks;           // [Bc][ks]
                ComputeT* scores  = lmem + 2 * Bc * ks;      // [Bc]

                // Base pointers for this batch_head
                const DataT* Q_row  = q_ptr + batch_head * slq * hd + query_idx * hd;
                const DataT* K_base = k_ptr + batch_head * slk * hd;
                const DataT* V_base = v_ptr + batch_head * slk * hd;
                DataT* O_row        = o_ptr + batch_head * slq * hd + query_idx * hd;

                // Each thread handles multiple output dimensions
                // Max elements per thread: ceil(head_dim / BLOCK_SIZE)
                constexpr int MAX_D_PER_THREAD = 8;  // Supports head_dim up to 1024
                ComputeT o_local[MAX_D_PER_THREAD];
                for (int i = 0; i < MAX_D_PER_THREAD; ++i) o_local[i] = ComputeT(0);

                // Online softmax state
                ComputeT m_prev = -std::numeric_limits<ComputeT>::infinity();
                ComputeT l_prev = ComputeT(0);

                const int num_kv_blocks = (slk + Bc - 1) / Bc;

                for (int kv_block = 0; kv_block < num_kv_blocks; ++kv_block) {
                    const int k_start = kv_block * Bc;

                    // For causal masking: skip blocks entirely past the causal boundary
                    if (causal && k_start > query_idx) break;

                    const int actual_Bc = sycl::min(Bc, slk - k_start);

                    // Load K/V tile cooperatively into local memory
                    for (int idx = tid; idx < actual_Bc * hd; idx += BLOCK_SIZE) {
                        int row = idx / hd;
                        int col = idx % hd;
                        DataT kval = K_base[(k_start + row) * hd + col];
                        DataT vval = V_base[(k_start + row) * hd + col];
                        if constexpr (IsBFloat16) {
                            K_tile[row * ks + col] = bf16_to_f32(kval);
                            V_tile[row * ks + col] = bf16_to_f32(vval);
                        } else {
                            K_tile[row * ks + col] = static_cast<ComputeT>(kval);
                            V_tile[row * ks + col] = static_cast<ComputeT>(vval);
                        }
                    }
                    sycl::group_barrier(item.get_group());

                    // Step 1: Compute Q·K^T scores and find block max
                    ComputeT local_max = -std::numeric_limits<ComputeT>::infinity();

                    for (int j = tid; j < actual_Bc; j += BLOCK_SIZE) {
                        ComputeT score = ComputeT(0);
                        for (int d = 0; d < hd; ++d) {
                            ComputeT q_val;
                            if constexpr (IsBFloat16) {
                                q_val = bf16_to_f32(Q_row[d]);
                            } else {
                                q_val = static_cast<ComputeT>(Q_row[d]);
                            }
                            score += q_val * sc * K_tile[j * ks + d];
                        }
                        // Apply causal mask
                        if (causal && (k_start + j) > query_idx) {
                            score = -std::numeric_limits<ComputeT>::infinity();
                        }
                        scores[j] = score;
                        local_max = sycl::fmax(local_max, score);
                    }

                    // Reduce max across work-group
                    ComputeT block_max = sycl::reduce_over_group(
                        item.get_group(), local_max,
                        sycl::maximum<ComputeT>());

                    // Step 2: Compute exp(score - max) and sum
                    ComputeT local_sum = ComputeT(0);
                    for (int j = tid; j < actual_Bc; j += BLOCK_SIZE) {
                        ComputeT exp_score = sycl::exp(scores[j] - block_max);
                        scores[j] = exp_score;
                        local_sum += exp_score;
                    }
                    sycl::group_barrier(item.get_group());

                    // Reduce sum across work-group
                    ComputeT block_sum = sycl::reduce_over_group(
                        item.get_group(), local_sum,
                        sycl::plus<ComputeT>());

                    // Step 3: Online softmax rescaling
                    ComputeT m_new = sycl::fmax(m_prev, block_max);
                    ComputeT exp_prev = sycl::exp(m_prev - m_new);
                    ComputeT exp_curr = sycl::exp(block_max - m_new);
                    ComputeT l_new = exp_prev * l_prev + exp_curr * block_sum;

                    // Step 4: Rescale previous output and accumulate P @ V
                    for (int i = 0; i < MAX_D_PER_THREAD; ++i) {
                        int d = tid + i * BLOCK_SIZE;
                        if (d < hd) {
                            // Rescale previous accumulator
                            o_local[i] *= exp_prev;

                            // Add new contribution: sum_j P[j] * V[j, d]
                            ComputeT pv_sum = ComputeT(0);
                            for (int j = 0; j < actual_Bc; ++j) {
                                pv_sum += scores[j] * V_tile[j * ks + d];
                            }
                            o_local[i] += exp_curr * pv_sum;
                        }
                    }

                    m_prev = m_new;
                    l_prev = l_new;

                    sycl::group_barrier(item.get_group());
                }

                // Final normalization and write output
                ComputeT l_inv = (l_prev > ComputeT(0))
                    ? (ComputeT(1) / l_prev) : ComputeT(0);

                for (int i = 0; i < MAX_D_PER_THREAD; ++i) {
                    int d = tid + i * BLOCK_SIZE;
                    if (d < hd) {
                        if constexpr (IsBFloat16) {
                            O_row[d] = f32_to_bf16(o_local[i] * l_inv);
                        } else {
                            O_row[d] = static_cast<DataT>(o_local[i] * l_inv);
                        }
                    }
                }
            }
        );
    });

    return output;
}

auto flash_attention_kernel(
    const Tensor& Q,
    const Tensor& K,
    const Tensor& V,
    const Tensor* mask,
    float scale,
    bool is_causal,
    sycl::queue& queue
) -> Tensor {
    if (Q.shape().size() != 3 || K.shape().size() != 3 || V.shape().size() != 3) {
        throw std::invalid_argument(
            "flash_attention_kernel: Q, K, V must be 3D [batch_heads, seq_len, head_dim]");
    }
    if (Q.dtype() != K.dtype() || Q.dtype() != V.dtype()) {
        throw std::invalid_argument("flash_attention_kernel: Q, K, V must have the same dtype");
    }

    if (Q.dtype() == DType::Float32) {
        return flash_attention_impl<float, float, FlashAttentionKernelFloat32>(
            Q, K, V, mask, scale, is_causal, queue);
    } else if (Q.dtype() == DType::Float64) {
        return flash_attention_impl<double, double, FlashAttentionKernelFloat64>(
            Q, K, V, mask, scale, is_causal, queue);
    } else if (Q.dtype() == DType::Float16) {
        return flash_attention_impl<sycl::half, float, FlashAttentionKernelFloat16>(
            Q, K, V, mask, scale, is_causal, queue);
    } else if (Q.dtype() == DType::BFloat16) {
        return flash_attention_impl<uint16_t, float, FlashAttentionKernelBFloat16, true>(
            Q, K, V, mask, scale, is_causal, queue);
    } else {
        throw std::runtime_error("flash_attention_kernel: unsupported dtype");
    }
}

} // namespace oneapi
} // namespace tenzor
