#include "tenzor/core/tensor.hpp"
#include "oneapi_kernel_utils.hpp"
#include "tenzor/core/shape.hpp"            // F16: broadcast_shapes
#include "tenzor/ops/transform.hpp"         // F16: broadcast_to
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
struct LayerNormBwdGradWB_F32 {};
struct LayerNormBwdGradWB_F64 {};
struct LayerNormBwdGradIn_F32 {};
struct LayerNormBwdGradIn_F64 {};
struct LayerNormBwdGradWB_F16 {};
struct LayerNormBwdGradIn_F16 {};
struct LayerNormBwdGradWB_BF16 {};
struct LayerNormBwdGradIn_BF16 {};
struct FlashAttentionKernelFloat32 {};
struct FlashAttentionKernelFloat64 {};
struct FlashAttentionKernelFloat16 {};
struct FlashAttentionKernelBFloat16 {};



// ============================================================================
// Fused Add + ReLU
// ============================================================================

auto fused_add_relu_kernel(const Tensor& a_orig, const Tensor& b_orig, sycl::queue& queue) -> Tensor {
    // Contiguify: the kernel reads a[i]+b[i] flat, so views with differing
    // physical layouts would be paired incorrectly (matches the CPU kernel).
    const Tensor a = a_orig.is_contiguous() ? a_orig : a_orig.contiguous();
    const Tensor b = b_orig.is_contiguous() ? b_orig : b_orig.contiguous();
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

auto fused_gelu_kernel(const Tensor& input_orig, sycl::queue& queue) -> Tensor {
    // Contiguify: the kernel reads/writes input flat (matches the CPU kernel).
    const Tensor input = input_orig.is_contiguous() ? input_orig : input_orig.contiguous();
    Tensor output(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                  input.dtype(), input.device());

    const int64_t numel = input.numel();

    if (input.dtype() == DType::Float32) {
        const float* in_ptr = get_data_ptr<const float>(input);
        float* out_ptr = get_data_ptr<float>(output);

        constexpr float inv_sqrt2 = 0.70710678f;

        queue.parallel_for<FusedGeluKernelFloat32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            float x = in_ptr[idx];
            out_ptr[idx] = 0.5f * x * (1.0f + sycl::erf(x * inv_sqrt2));
        });
    }
    else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        double* out_ptr = get_data_ptr<double>(output);

        constexpr double inv_sqrt2 = 0.70710678118654752;

        queue.parallel_for<FusedGeluKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            double x = in_ptr[idx];
            out_ptr[idx] = 0.5 * x * (1.0 + sycl::erf(x * inv_sqrt2));
        });
    }
    else if (input.dtype() == DType::Float16) {
        const sycl::half* in_ptr = get_data_ptr<const sycl::half>(input);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);

        constexpr float inv_sqrt2 = 0.70710678f;

        queue.parallel_for<FusedGeluKernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            float x = static_cast<float>(in_ptr[idx]);
            out_ptr[idx] = sycl::half(0.5f * x * (1.0f + sycl::erf(x * inv_sqrt2)));
        });
    }
    else if (input.dtype() == DType::BFloat16) {
        const uint16_t* in_ptr = get_data_ptr<const uint16_t>(input);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);

        const float inv_sqrt2 = 0.70710678f;

        queue.parallel_for<FusedGeluKernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            float x = bf16_to_f32(in_ptr[idx]);
            out_ptr[idx] = f32_to_bf16(0.5f * x * (1.0f + sycl::erf(x * inv_sqrt2)));
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
    const Tensor& input_orig,
    const Tensor& weight,  // gamma
    const Tensor& bias,    // beta
    const std::vector<int64_t>& normalized_shape,
    float epsilon,
    sycl::queue& queue
) -> std::tuple<Tensor, Tensor, Tensor> {
    // Contiguify: the kernel indexes input flat (in_ptr + b*norm_size), so a
    // non-contiguous view would read the wrong storage and corrupt the saved
    // mean/inv_std. Mirrors the CPU kernel's guard.
    const Tensor input = input_orig.is_contiguous() ? input_orig : input_orig.contiguous();
    // Calculate normalized dimension size
    int64_t norm_size = 1;
    for (auto dim : normalized_shape) {
        norm_size *= dim;
    }

    int64_t batch_size = input.numel() / norm_size;

    // Per docs/internals/attention-contract.md: mean / inv_std must be Float32
    // for FP16/BF16 inputs (rstd dynamic range exceeds FP16 max=65504 when
    // var ~ 1e-11). Audit M11 OneAPI — was storing stats as input.dtype().
    Tensor output(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                  input.dtype(), input.device());
    DType stats_dtype = (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16)
                            ? DType::Float32
                            : input.dtype();
    Tensor mean({batch_size}, stats_dtype, input.device());
    Tensor inv_std({batch_size}, stats_dtype, input.device());

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
        // Float16: use float32 accumulation for numerical stability.
        // mean/inv_std tensors are Float32 (audit M11 — was Float16, overflow risk).
        const sycl::half* in_ptr = get_data_ptr<const sycl::half>(input);
        const sycl::half* weight_ptr = get_data_ptr<const sycl::half>(weight);
        const sycl::half* bias_ptr = get_data_ptr<const sycl::half>(bias);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);
        float* mean_ptr = get_data_ptr<float>(mean);
        float* inv_std_ptr = get_data_ptr<float>(inv_std);

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
                    // Stats stored as Float32 per attention-contract.md
                    mean_ptr[b] = batch_mean;
                    inv_std_ptr[b] = batch_inv_std;
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
        // BFloat16: use float32 accumulation. mean/inv_std are Float32 per
        // attention-contract.md (audit M11 — was BF16 with overflow risk).
        const uint16_t* in_ptr = get_data_ptr<const uint16_t>(input);
        const uint16_t* weight_ptr = get_data_ptr<const uint16_t>(weight);
        const uint16_t* bias_ptr = get_data_ptr<const uint16_t>(bias);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);
        float* mean_ptr = get_data_ptr<float>(mean);
        float* inv_std_ptr = get_data_ptr<float>(inv_std);

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
                    // Stats stored as Float32 per attention-contract.md
                    mean_ptr[b] = batch_mean;
                    inv_std_ptr[b] = batch_inv_std;
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

    // Helper: round up to next power of 2, capped at 256
    auto round_up_pow2 = [](int64_t n) -> int64_t {
        int64_t p = 1;
        while (p < n) p *= 2;
        return std::min(p, static_cast<int64_t>(256));
    };

    // audit-2026-05-03 — Compute parameter generalises the previously
    // hardcoded `float` accumulator. For Float64 input we compute in
    // double end-to-end (was casting through float, dropping ~30 mantissa
    // bits and breaking gradcheck).
    auto dispatch_backward = [&]<typename Compute, typename GradWBKernel, typename GradInKernel>(
        auto load_go, auto load_in, auto load_mean, auto load_inv_std, auto load_weight,
        auto store_gw, auto store_gb, auto store_gi,
        GradWBKernel /*tag*/, GradInKernel /*tag2*/) {

        int64_t wg_wb = round_up_pow2(batch_size);
        queue.submit([&](sycl::handler& cgh) {
            sycl::local_accessor<Compute, 1> local_gw(sycl::range<1>(wg_wb), cgh);
            sycl::local_accessor<Compute, 1> local_gb(sycl::range<1>(wg_wb), cgh);
            cgh.parallel_for<GradWBKernel>(
                sycl::nd_range<1>(norm_size * wg_wb, wg_wb),
                [=](sycl::nd_item<1> item) {
                    int64_t f = item.get_group(0);
                    int64_t lid = item.get_local_id(0);
                    int64_t lsize = item.get_local_range(0);

                    Compute acc_gw = Compute(0), acc_gb = Compute(0);
                    for (int64_t b = lid; b < batch_size; b += lsize) {
                        Compute go = load_go(b * norm_size + f);
                        Compute inp = load_in(b * norm_size + f);
                        Compute m = load_mean(b);
                        Compute rstd = load_inv_std(b);
                        Compute normalized = (inp - m) * rstd;
                        acc_gw += go * normalized;
                        acc_gb += go;
                    }

                    local_gw[lid] = acc_gw;
                    local_gb[lid] = acc_gb;
                    item.barrier(sycl::access::fence_space::local_space);

                    for (int64_t stride = lsize / 2; stride > 0; stride >>= 1) {
                        if (lid < stride) {
                            local_gw[lid] += local_gw[lid + stride];
                            local_gb[lid] += local_gb[lid + stride];
                        }
                        item.barrier(sycl::access::fence_space::local_space);
                    }

                    if (lid == 0) {
                        store_gw(f, local_gw[0]);
                        store_gb(f, local_gb[0]);
                    }
                });
        });

        int64_t wg_in = round_up_pow2(norm_size);
        queue.submit([&](sycl::handler& cgh) {
            sycl::local_accessor<Compute, 1> local_ds(sycl::range<1>(wg_in), cgh);
            sycl::local_accessor<Compute, 1> local_db(sycl::range<1>(wg_in), cgh);
            cgh.parallel_for<GradInKernel>(
                sycl::nd_range<1>(batch_size * wg_in, wg_in),
                [=](sycl::nd_item<1> item) {
                    int64_t b = item.get_group(0);
                    int64_t lid = item.get_local_id(0);
                    int64_t lsize = item.get_local_range(0);
                    Compute m = load_mean(b);
                    Compute rstd = load_inv_std(b);

                    Compute thread_ds = Compute(0), thread_db = Compute(0);
                    for (int64_t i = lid; i < norm_size; i += lsize) {
                        Compute normalized = (Compute(load_in(b * norm_size + i)) - m) * rstd;
                        Compute go_w = Compute(load_go(b * norm_size + i)) * Compute(load_weight(i));
                        thread_ds += go_w * normalized;
                        thread_db += go_w;
                    }

                    local_ds[lid] = thread_ds;
                    local_db[lid] = thread_db;
                    item.barrier(sycl::access::fence_space::local_space);

                    for (int64_t stride = lsize / 2; stride > 0; stride >>= 1) {
                        if (lid < stride) {
                            local_ds[lid] += local_ds[lid + stride];
                            local_db[lid] += local_db[lid + stride];
                        }
                        item.barrier(sycl::access::fence_space::local_space);
                    }

                    Compute ds = local_ds[0];
                    Compute db = local_db[0];
                    Compute inv_n = Compute(1) / Compute(norm_size);
                    item.barrier(sycl::access::fence_space::local_space);

                    for (int64_t i = lid; i < norm_size; i += lsize) {
                        Compute normalized = (Compute(load_in(b * norm_size + i)) - m) * rstd;
                        Compute go = load_go(b * norm_size + i);
                        Compute w = load_weight(i);
                        // db = Σ(go·w) and ds = Σ(go·w·x_hat) are ALREADY weighted, so
                        // the weight must apply only to the go term — multiplying the
                        // whole expression by w double-weights the mean-correction and
                        // gives a wrong grad_input. Matches the CPU reference.
                        Compute gi = rstd * (w * go - inv_n * (db + normalized * ds));
                        store_gi(b * norm_size + i, gi);
                    }
                });
        });

        queue.wait_and_throw();
    };

    if (input.dtype() == DType::Float32) {
        const float* grad_out_ptr = get_data_ptr<const float>(grad_output);
        const float* in_ptr = get_data_ptr<const float>(input);
        const float* mean_ptr = get_data_ptr<const float>(mean);
        const float* inv_std_ptr = get_data_ptr<const float>(inv_std);
        const float* weight_ptr = get_data_ptr<const float>(weight);
        float* grad_in_ptr = get_data_ptr<float>(grad_input);
        float* grad_weight_ptr = get_data_ptr<float>(grad_weight);
        float* grad_bias_ptr = get_data_ptr<float>(grad_bias);

        dispatch_backward.template operator()<float>(
            [=](int64_t i) { return grad_out_ptr[i]; },
            [=](int64_t i) { return in_ptr[i]; },
            [=](int64_t i) { return mean_ptr[i]; },
            [=](int64_t i) { return inv_std_ptr[i]; },
            [=](int64_t i) { return weight_ptr[i]; },
            [=](int64_t i, float v) { grad_weight_ptr[i] = v; },
            [=](int64_t i, float v) { grad_bias_ptr[i] = v; },
            [=](int64_t i, float v) { grad_in_ptr[i] = v; },
            LayerNormBwdGradWB_F32{}, LayerNormBwdGradIn_F32{});
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

        // audit-2026-05-03 — compute end-to-end in double (was casting
        // through float).
        dispatch_backward.template operator()<double>(
            [=](int64_t i) { return grad_out_ptr[i]; },
            [=](int64_t i) { return in_ptr[i]; },
            [=](int64_t i) { return mean_ptr[i]; },
            [=](int64_t i) { return inv_std_ptr[i]; },
            [=](int64_t i) { return weight_ptr[i]; },
            [=](int64_t i, double v) { grad_weight_ptr[i] = v; },
            [=](int64_t i, double v) { grad_bias_ptr[i] = v; },
            [=](int64_t i, double v) { grad_in_ptr[i] = v; },
            LayerNormBwdGradWB_F64{}, LayerNormBwdGradIn_F64{});
    }
    else if (input.dtype() == DType::BFloat16) {
        const uint16_t* grad_out_ptr = get_data_ptr<const uint16_t>(grad_output);
        const uint16_t* in_ptr = get_data_ptr<const uint16_t>(input);
        // Forward stores mean/inv_std as Float32 for BFloat16 inputs (rstd dynamic
        // range exceeds BF16 max), so read them through const float* here.
        const float* mean_ptr = get_data_ptr<const float>(mean);
        const float* inv_std_ptr = get_data_ptr<const float>(inv_std);
        const uint16_t* weight_ptr = get_data_ptr<const uint16_t>(weight);
        uint16_t* grad_in_ptr = get_data_ptr<uint16_t>(grad_input);
        uint16_t* grad_weight_ptr = get_data_ptr<uint16_t>(grad_weight);
        uint16_t* grad_bias_ptr = get_data_ptr<uint16_t>(grad_bias);

        dispatch_backward.template operator()<float>(
            [=](int64_t i) { return bf16_to_f32(grad_out_ptr[i]); },
            [=](int64_t i) { return bf16_to_f32(in_ptr[i]); },
            [=](int64_t i) { return mean_ptr[i]; },
            [=](int64_t i) { return inv_std_ptr[i]; },
            [=](int64_t i) { return bf16_to_f32(weight_ptr[i]); },
            [=](int64_t i, float v) { grad_weight_ptr[i] = f32_to_bf16(v); },
            [=](int64_t i, float v) { grad_bias_ptr[i] = f32_to_bf16(v); },
            [=](int64_t i, float v) { grad_in_ptr[i] = f32_to_bf16(v); },
            LayerNormBwdGradWB_BF16{}, LayerNormBwdGradIn_BF16{});
    }
    else if (input.dtype() == DType::Float16) {
        const sycl::half* grad_out_ptr = get_data_ptr<const sycl::half>(grad_output);
        const sycl::half* in_ptr = get_data_ptr<const sycl::half>(input);
        // Forward stores mean/inv_std as Float32 for Float16 inputs (rstd dynamic
        // range exceeds FP16 max), so read them through const float* here.
        const float* mean_ptr = get_data_ptr<const float>(mean);
        const float* inv_std_ptr = get_data_ptr<const float>(inv_std);
        const sycl::half* weight_ptr = get_data_ptr<const sycl::half>(weight);
        sycl::half* grad_in_ptr = get_data_ptr<sycl::half>(grad_input);
        sycl::half* grad_weight_ptr = get_data_ptr<sycl::half>(grad_weight);
        sycl::half* grad_bias_ptr = get_data_ptr<sycl::half>(grad_bias);

        dispatch_backward.template operator()<float>(
            [=](int64_t i) { return static_cast<float>(grad_out_ptr[i]); },
            [=](int64_t i) { return static_cast<float>(in_ptr[i]); },
            [=](int64_t i) { return mean_ptr[i]; },
            [=](int64_t i) { return inv_std_ptr[i]; },
            [=](int64_t i) { return static_cast<float>(weight_ptr[i]); },
            [=](int64_t i, float v) { grad_weight_ptr[i] = sycl::half(v); },
            [=](int64_t i, float v) { grad_bias_ptr[i] = sycl::half(v); },
            [=](int64_t i, float v) { grad_in_ptr[i] = sycl::half(v); },
            LayerNormBwdGradWB_F16{}, LayerNormBwdGradIn_F16{});
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
    const Tensor& input_orig,
    const Tensor& running_mean,
    const Tensor& running_var,
    const Tensor& weight,
    const Tensor& bias,
    float epsilon,
    sycl::queue& queue
) -> Tensor {
    // Contiguify: the kernel indexes input flat as NCHW, so a channels-last /
    // permuted view would map elements to the wrong channel (matches CPU).
    const Tensor input = input_orig.is_contiguous() ? input_orig : input_orig.contiguous();
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

    // Count of batched matrices (product of all leading dims of A). For rank-2
    // inputs this is 1. The kernels below launch range<3>(batch, m, n) and
    // offset each operand by its per-batch matrix stride so the entire output
    // buffer (batch*m*n) is filled — previously only the first matrix was
    // written, leaving the rest of the buffer uninitialised for rank>2 inputs.
    int64_t batch = 1;
    for (size_t i = 0; i + 2 < a_shape.size(); ++i) {
        batch *= a_shape[i];
    }
    // B may be a single shared matrix (rank-2) broadcast across the batch, or
    // batched with its own leading dims. Detect a batched B by rank.
    const bool b_batched = (b_shape.size() > 2);
    const int64_t a_batch_stride = m * k;
    const int64_t b_batch_stride = b_batched ? (k * n) : 0;
    const int64_t out_batch_stride = m * n;

    Tensor output(out_shape, a.dtype(), a.device());

    if (a.dtype() == DType::Float32) {
        const float* a_ptr = get_data_ptr<const float>(a);
        const float* b_ptr = get_data_ptr<const float>(b);
        const float* bias_ptr = get_data_ptr<const float>(bias);
        float* out_ptr = get_data_ptr<float>(output);

        queue.parallel_for<FusedMatmulAddKernelFloat32>(
            sycl::range<3>(batch, m, n),
            [=](sycl::id<3> idx) {
                int64_t bt = idx[0];
                int64_t i = idx[1];
                int64_t j = idx[2];

                const float* a_mat = a_ptr + bt * a_batch_stride;
                const float* b_mat = b_ptr + bt * b_batch_stride;

                float sum = 0.0f;
                for (int64_t p = 0; p < k; ++p) {
                    sum += a_mat[i * k + p] * b_mat[p * n + j];
                }

                // Add bias (broadcast along the m and batch dimensions)
                out_ptr[bt * out_batch_stride + i * n + j] = sum + bias_ptr[j];
            }
        );
    }
    else if (a.dtype() == DType::Float64) {
        const double* a_ptr = get_data_ptr<const double>(a);
        const double* b_ptr = get_data_ptr<const double>(b);
        const double* bias_ptr = get_data_ptr<const double>(bias);
        double* out_ptr = get_data_ptr<double>(output);

        queue.parallel_for<FusedMatmulAddKernelFloat64>(
            sycl::range<3>(batch, m, n),
            [=](sycl::id<3> idx) {
                int64_t bt = idx[0];
                int64_t i = idx[1];
                int64_t j = idx[2];

                const double* a_mat = a_ptr + bt * a_batch_stride;
                const double* b_mat = b_ptr + bt * b_batch_stride;

                double sum = 0.0;
                for (int64_t p = 0; p < k; ++p) {
                    sum += a_mat[i * k + p] * b_mat[p * n + j];
                }

                out_ptr[bt * out_batch_stride + i * n + j] = sum + bias_ptr[j];
            }
        );
    }
    else if (a.dtype() == DType::Float16) {
        const sycl::half* a_ptr = get_data_ptr<const sycl::half>(a);
        const sycl::half* b_ptr = get_data_ptr<const sycl::half>(b);
        const sycl::half* bias_ptr = get_data_ptr<const sycl::half>(bias);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);

        queue.parallel_for<FusedMatmulAddKernelFloat16>(
            sycl::range<3>(batch, m, n),
            [=](sycl::id<3> idx) {
                int64_t bt = idx[0];
                int64_t i = idx[1];
                int64_t j = idx[2];

                const sycl::half* a_mat = a_ptr + bt * a_batch_stride;
                const sycl::half* b_mat = b_ptr + bt * b_batch_stride;

                float sum = 0.0f;
                for (int64_t p = 0; p < k; ++p) {
                    sum += static_cast<float>(a_mat[i * k + p]) *
                           static_cast<float>(b_mat[p * n + j]);
                }
                out_ptr[bt * out_batch_stride + i * n + j] =
                    sycl::half(sum + static_cast<float>(bias_ptr[j]));
            }
        );
    }
    else if (a.dtype() == DType::BFloat16) {
        const uint16_t* a_ptr = get_data_ptr<const uint16_t>(a);
        const uint16_t* b_ptr = get_data_ptr<const uint16_t>(b);
        const uint16_t* bias_ptr = get_data_ptr<const uint16_t>(bias);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);

        queue.parallel_for<FusedMatmulAddKernelBFloat16>(
            sycl::range<3>(batch, m, n),
            [=](sycl::id<3> idx) {
                int64_t bt = idx[0];
                int64_t i = idx[1];
                int64_t j = idx[2];

                const uint16_t* a_mat = a_ptr + bt * a_batch_stride;
                const uint16_t* b_mat = b_ptr + bt * b_batch_stride;

                float sum = 0.0f;
                for (int64_t p = 0; p < k; ++p) {
                    sum += bf16_to_f32(a_mat[i * k + p]) *
                           bf16_to_f32(b_mat[p * n + j]);
                }
                out_ptr[bt * out_batch_stride + i * n + j] =
                    f32_to_bf16(sum + bf16_to_f32(bias_ptr[j]));
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
    // Classes are the LAST dim; flatten all leading dims into the batch so this
    // works for rank-2 (N,C) and rank>2 (D1,...,Dk,C) seq2seq logits alike.
    // (Was shape[0]/shape[1], which mis-read rank-3 as (N, T) and ignored C.)
    int64_t num_classes = logits.shape().back();
    int64_t batch_size = (num_classes > 0) ? logits.numel() / num_classes : 0;

    Tensor losses({batch_size}, logits.dtype(), logits.device());

    // Device-side softmax cross-entropy: one work-item per batch element computes
    // max, sum_exp, and loss entirely on device (no host roundtrip).
    auto ce_impl = [&]<typename T>(T /*tag*/) {
        using AccT = std::conditional_t<std::is_same_v<T, double>, double, float>;
        const T* logits_ptr = get_data_ptr<const T>(logits);
        const int64_t* targets_ptr = get_data_ptr<const int64_t>(targets);
        T* losses_ptr = get_data_ptr<T>(losses);
        const int64_t nc = num_classes;

        queue.parallel_for(sycl::range<1>(batch_size), [=](sycl::id<1> idx) {
            int64_t b = static_cast<int64_t>(idx[0]);
            const T* row = logits_ptr + b * nc;
            int64_t target = targets_ptr[b];

            // Per docs/internals/attention-contract.md: backends must bounds-
            // check targets to prevent OOB device-memory access (audit C10
            // OneAPI). Out-of-range targets get NaN loss matching the CPU
            // contract; without this, row[target] reads adjacent rows or
            // beyond-buffer memory and silently corrupts neighboring data.
            if (target < 0 || target >= nc) {
                // SYCL device-side NaN via the standard math intrinsic.
                losses_ptr[b] = static_cast<T>(sycl::nan(0u));
                return;
            }

            // Find max for numerical stability
            AccT max_val = static_cast<AccT>(row[0]);
            for (int64_t i = 1; i < nc; ++i) {
                AccT v = static_cast<AccT>(row[i]);
                if (v > max_val) max_val = v;
            }

            // Compute sum(exp(x - max))
            AccT sum_exp = AccT(0);
            for (int64_t i = 0; i < nc; ++i) {
                sum_exp += sycl::exp(static_cast<AccT>(row[i]) - max_val);
            }

            // loss = log_sum_exp - target_logit
            AccT log_sum_exp = sycl::log(sum_exp) + max_val;
            AccT loss = log_sum_exp - static_cast<AccT>(row[target]);
            losses_ptr[b] = static_cast<T>(loss);
        }).wait();
    };

    if (logits.dtype() == DType::Float32) {
        ce_impl(float{});
    } else if (logits.dtype() == DType::Float64) {
        ce_impl(double{});
    } else if (logits.dtype() == DType::Float16) {
        // Float16: compute in float32 via cast to float pointers isn't possible,
        // so use a dedicated kernel that reads half and accumulates in float
        const sycl::half* logits_ptr = get_data_ptr<const sycl::half>(logits);
        const int64_t* targets_ptr = get_data_ptr<const int64_t>(targets);
        sycl::half* losses_ptr = get_data_ptr<sycl::half>(losses);
        const int64_t nc = num_classes;

        queue.parallel_for(sycl::range<1>(batch_size), [=](sycl::id<1> idx) {
            int64_t b = static_cast<int64_t>(idx[0]);
            const sycl::half* row = logits_ptr + b * nc;
            int64_t target = targets_ptr[b];
            if (target < 0 || target >= nc) {
                losses_ptr[b] = sycl::half(sycl::nan(0u));
                return;
            }

            float max_val = static_cast<float>(row[0]);
            for (int64_t i = 1; i < nc; ++i) {
                float v = static_cast<float>(row[i]);
                if (v > max_val) max_val = v;
            }
            float sum_exp = 0.0f;
            for (int64_t i = 0; i < nc; ++i) {
                sum_exp += sycl::exp(static_cast<float>(row[i]) - max_val);
            }
            float loss = sycl::log(sum_exp) + max_val - static_cast<float>(row[target]);
            losses_ptr[b] = sycl::half(loss);
        }).wait();
    } else if (logits.dtype() == DType::BFloat16) {
        const uint16_t* logits_ptr = get_data_ptr<const uint16_t>(logits);
        const int64_t* targets_ptr = get_data_ptr<const int64_t>(targets);
        uint16_t* losses_ptr = get_data_ptr<uint16_t>(losses);
        const int64_t nc = num_classes;

        queue.parallel_for(sycl::range<1>(batch_size), [=](sycl::id<1> idx) {
            int64_t b = static_cast<int64_t>(idx[0]);
            const uint16_t* row = logits_ptr + b * nc;
            int64_t target = targets_ptr[b];
            if (target < 0 || target >= nc) {
                losses_ptr[b] = f32_to_bf16(sycl::nan(0u));
                return;
            }

            float max_val = bf16_to_f32(row[0]);
            for (int64_t i = 1; i < nc; ++i) {
                float v = bf16_to_f32(row[i]);
                if (v > max_val) max_val = v;
            }
            float sum_exp = 0.0f;
            for (int64_t i = 0; i < nc; ++i) {
                sum_exp += sycl::exp(bf16_to_f32(row[i]) - max_val);
            }
            float loss = sycl::log(sum_exp) + max_val - bf16_to_f32(row[target]);
            losses_ptr[b] = f32_to_bf16(loss);
        }).wait();
    } else {
        throw std::runtime_error("fused_softmax_cross_entropy: unsupported dtype");
    }

    // Apply reduction on device using parallel sycl::reduction (was previously
    // a serial loop in a single GPU thread — bad for non-trivial batch sizes).
    if (reduction == "mean" || reduction == "sum") {
        Tensor result({1}, logits.dtype(), logits.device());
        const int64_t bs = batch_size;

        if (logits.dtype() == DType::Float32) {
            float* result_ptr = get_data_ptr<float>(result);
            const float* losses_ptr = get_data_ptr<const float>(losses);
            const float scale = (reduction == "mean") ? 1.0f / static_cast<float>(bs) : 1.0f;
            queue.memset(result_ptr, 0, sizeof(float));
            queue.parallel_for(sycl::range<1>(bs), sycl::reduction(result_ptr, sycl::plus<float>()),
                [=](sycl::id<1> i, auto& sum) {
                    sum += losses_ptr[static_cast<int64_t>(i)];
                });
            queue.single_task([=]() { result_ptr[0] *= scale; }).wait();
        } else if (logits.dtype() == DType::Float64) {
            double* result_ptr = get_data_ptr<double>(result);
            const double* losses_ptr = get_data_ptr<const double>(losses);
            const double scale = (reduction == "mean") ? 1.0 / static_cast<double>(bs) : 1.0;
            queue.memset(result_ptr, 0, sizeof(double));
            queue.parallel_for(sycl::range<1>(bs), sycl::reduction(result_ptr, sycl::plus<double>()),
                [=](sycl::id<1> i, auto& sum) {
                    sum += losses_ptr[static_cast<int64_t>(i)];
                });
            queue.single_task([=]() { result_ptr[0] *= scale; }).wait();
        } else if (logits.dtype() == DType::Float16) {
            // Accumulate in float32 for precision, cast back to Float16 at the end.
            sycl::half* result_ptr = get_data_ptr<sycl::half>(result);
            const sycl::half* losses_ptr = get_data_ptr<const sycl::half>(losses);
            const float scale = (reduction == "mean") ? 1.0f / static_cast<float>(bs) : 1.0f;
            float* scratch = sycl::malloc_device<float>(1, queue);
            queue.memset(scratch, 0, sizeof(float));
            queue.parallel_for(sycl::range<1>(bs), sycl::reduction(scratch, sycl::plus<float>()),
                [=](sycl::id<1> i, auto& sum) {
                    sum += static_cast<float>(losses_ptr[static_cast<int64_t>(i)]);
                });
            queue.single_task([=]() {
                result_ptr[0] = sycl::half(scratch[0] * scale);
            }).wait();
            sycl::free(scratch, queue);
        } else if (logits.dtype() == DType::BFloat16) {
            uint16_t* result_ptr = get_data_ptr<uint16_t>(result);
            const uint16_t* losses_ptr = get_data_ptr<const uint16_t>(losses);
            const float scale = (reduction == "mean") ? 1.0f / static_cast<float>(bs) : 1.0f;
            float* scratch = sycl::malloc_device<float>(1, queue);
            queue.memset(scratch, 0, sizeof(float));
            queue.parallel_for(sycl::range<1>(bs), sycl::reduction(scratch, sycl::plus<float>()),
                [=](sycl::id<1> i, auto& sum) {
                    sum += bf16_to_f32(losses_ptr[static_cast<int64_t>(i)]);
                });
            queue.single_task([=]() {
                result_ptr[0] = f32_to_bf16(scratch[0] * scale);
            }).wait();
            sycl::free(scratch, queue);
        }
        return result;
    }

    // "none" reduction: per-sample loss. Reshape the flat [batch] buffer back to
    // the logits' leading dims (e.g. (N,T) for rank-3 logits (N,T,C)) so the
    // op preserves rank, matching the CPU contract.
    if (logits.ndim() > 2) {
        std::vector<int64_t> lead(logits.shape().begin(), logits.shape().end() - 1);
        return losses.reshape(lead);
    }
    return losses;
}

// Gradient of softmax-cross-entropy w.r.t. logits (assuming upstream grad 1 on
// the reduced scalar): grad[b,c] = (softmax(logits)[b,c] - [c==target_b]) * s,
// where s = 1/N for "mean" and 1 for "sum"/"none". Preserves the logits shape
// (incl. rank>2) and dtype; computes in Float32 internally. targets are Int64
// (matching the loss kernel's contract).
auto fused_softmax_cross_entropy_grad_kernel(
    const Tensor& logits,
    const Tensor& targets,
    const std::string& reduction,
    sycl::queue& queue
) -> Tensor {
    const int64_t num_classes = logits.shape().back();
    const int64_t batch_size = (num_classes > 0) ? logits.numel() / num_classes : 0;
    std::vector<int64_t> shape(logits.shape().begin(), logits.shape().end());

    Tensor grad(shape, logits.dtype(), logits.device());
    if (batch_size == 0) return grad;

    const bool is_f32 = (logits.dtype() == DType::Float32);
    Tensor lp = is_f32 ? logits.contiguous() : logits.to(DType::Float32);
    Tensor gp = is_f32 ? grad : Tensor(shape, DType::Float32, logits.device());

    const float* in_ptr  = get_data_ptr<const float>(lp);
    float*       out_ptr = get_data_ptr<float>(gp);
    auto targets_i64 = (targets.dtype() == DType::Int64) ? targets : targets.to(DType::Int64);
    const int64_t* tgt_ptr = get_data_ptr<const int64_t>(targets_i64);

    const int64_t nc = num_classes;
    const float scale = (reduction == "mean") ? 1.0f / static_cast<float>(batch_size) : 1.0f;

    queue.parallel_for(sycl::range<1>(batch_size), [=](sycl::id<1> idx) {
        int64_t b = static_cast<int64_t>(idx[0]);
        const float* row = in_ptr + b * nc;
        int64_t t = tgt_ptr[b];
        float mx = row[0];
        for (int64_t i = 1; i < nc; ++i) mx = sycl::fmax(mx, row[i]);
        float se = 0.0f;
        for (int64_t i = 0; i < nc; ++i) se += sycl::exp(row[i] - mx);
        for (int64_t c = 0; c < nc; ++c) {
            float sm = sycl::exp(row[c] - mx) / se;
            out_ptr[b * nc + c] = (sm - ((c == t) ? 1.0f : 0.0f)) * scale;
        }
    }).wait();

    return is_f32 ? gp : gp.to(logits.dtype());
}

// ============================================================================
// Fused RMSNorm Forward
// RMSNorm: output = x * weight / sqrt(mean(x^2) + eps)
// Returns: (output, rrms) where rrms = 1/sqrt(mean(x^2) + eps)
// ============================================================================

auto fused_rms_norm_kernel(const Tensor& input_orig, const Tensor& weight, float eps,
                            sycl::queue& queue) -> std::tuple<Tensor, Tensor> {
    // Contiguify: the kernel indexes input flat (in_ptr + b*norm_size), so a
    // non-contiguous residual view would read the wrong storage and corrupt the
    // saved rrms. Mirrors the CPU kernel's guard.
    const Tensor input = input_orig.is_contiguous() ? input_orig : input_orig.contiguous();
    auto shape = input.shape();
    int64_t norm_size = shape.back();
    int64_t batch_size = input.numel() / norm_size;

    Tensor output(std::vector<int64_t>(shape.begin(), shape.end()),
                  input.dtype(), input.device());
    // Store rrms (1/sqrt(var)) at >= Float32 precision: in Float16/BFloat16 the
    // reciprocal-sqrt can overflow to Inf, producing NaN gradients. Half inputs
    // use Float32; Float32/Float64 keep their own dtype.
    DType rrms_dtype = (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16)
                           ? DType::Float32 : input.dtype();
    Tensor rrms({batch_size}, rrms_dtype, input.device());

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
        float* rrms_ptr = get_data_ptr<float>(rrms);  // rrms is Float32 for half inputs

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
                rrms_ptr[b] = rr;  // Float32 rrms (no half overflow)

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
        float* rrms_ptr = get_data_ptr<float>(rrms);  // rrms is Float32 for half inputs

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
                rrms_ptr[b] = rr;  // Float32 rrms (no bf16 overflow)

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

        // Compute grad_weight on device: each work-item accumulates one feature over the batch
        queue.parallel_for(sycl::range<1>(norm_size), [=](sycl::id<1> idx) {
            int64_t i = static_cast<int64_t>(idx[0]);
            float sum = 0.0f;
            for (int64_t b = 0; b < batch_size; ++b) {
                sum += go_ptr[b * norm_size + i] * in_ptr[b * norm_size + i] * rrms_ptr[b];
            }
            gw_ptr[i] = sum;
        }).wait();
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

        // Compute grad_weight on device: each work-item accumulates one feature over the batch
        queue.parallel_for(sycl::range<1>(norm_size), [=](sycl::id<1> idx) {
            int64_t i = static_cast<int64_t>(idx[0]);
            double sum = 0.0;
            for (int64_t b = 0; b < batch_size; ++b) {
                sum += go_ptr[b * norm_size + i] * in_ptr[b * norm_size + i] * rrms_ptr[b];
            }
            gw_ptr[i] = sum;
        }).wait();
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

        // Compute grad_weight on device: each work-item accumulates one feature over the batch
        queue.parallel_for(sycl::range<1>(norm_size), [=](sycl::id<1> idx) {
            int64_t i = static_cast<int64_t>(idx[0]);
            float sum = 0.0f;
            for (int64_t b = 0; b < batch_size; ++b) {
                sum += static_cast<float>(go_ptr[b * norm_size + i]) *
                       static_cast<float>(in_ptr[b * norm_size + i]) *
                       static_cast<float>(rrms_ptr[b]);
            }
            gw_ptr[i] = sycl::half(sum);
        }).wait();
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

        // Compute grad_weight on device: each work-item accumulates one feature over the batch
        queue.parallel_for(sycl::range<1>(norm_size), [=](sycl::id<1> idx) {
            int64_t i = static_cast<int64_t>(idx[0]);
            float sum = 0.0f;
            for (int64_t b = 0; b < batch_size; ++b) {
                sum += bf16_to_f32(go_ptr[b * norm_size + i]) *
                       bf16_to_f32(in_ptr[b * norm_size + i]) *
                       bf16_to_f32(rrms_ptr[b]);
            }
            gw_ptr[i] = f32_to_bf16(sum);
        }).wait();
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

            // Bias-corrected estimates. AMSGrad tracks the running maximum over
            // the RAW second moment and applies bias correction AFTER, matching
            // the CPU reference / PyTorch (maxing the bias-corrected v_hat is
            // wrong because bias_correction2 grows toward 1 across steps).
            float m_hat = m / f_bc1;
            float v_hat;
            if (f_amsgrad && max_v_ptr) {
                float max_v = max_v_ptr[idx];
                if (v > max_v) max_v = v;
                max_v_ptr[idx] = max_v;
                v_hat = max_v / f_bc2;
            } else {
                v_hat = v / f_bc2;
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
            // AMSGrad: max over RAW v, bias-correct after (matches CPU / PyTorch).
            double v_hat;
            if (d_amsgrad && max_v_ptr) {
                double max_v = max_v_ptr[idx];
                if (v > max_v) max_v = v;
                max_v_ptr[idx] = max_v;
                v_hat = max_v / d_bc2;
            } else {
                v_hat = v / d_bc2;
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
//
// Phase 8.2: now also writes the per-query log-sum-exp into L_out so the fused
// backward kernel can recover P = exp(S - L) without rerunning the softmax pass.
// L_out is shape [batch_heads, seq_len_q] in Float32.
template<typename DataT, typename ComputeT, typename KernelName, bool IsBFloat16 = false>
auto flash_attention_impl(
    const Tensor& Q,    // [batch_heads, seq_len_q, head_dim]
    const Tensor& K,    // [batch_heads, seq_len_k, head_dim]
    const Tensor& V,    // [batch_heads, seq_len_k, head_dim]
    const Tensor* mask,  // optional [batch_heads, seq_len_q, seq_len_k] or broadcastable
    ComputeT scale,
    bool is_causal,
    sycl::queue& queue,
    Tensor* L_out = nullptr  // optional [batch_heads, seq_len_q] Float32
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

    // The per-thread output accumulator o_local[MAX_D_PER_THREAD] together with
    // BLOCK_SIZE threads covers output dims d in [0, MAX_D_PER_THREAD*BLOCK_SIZE).
    // Reject head_dim beyond that bound up front: otherwise O_row[d] for the
    // excess dims would never be written (uninitialised output / silently
    // dropped contributions) instead of producing a clear error. Mirrors the
    // backward kernel's explicit head_dim validation.
    constexpr int MAX_D_PER_THREAD = 8;  // must match the kernel's accumulator
    const int MAX_HEAD_DIM = MAX_D_PER_THREAD * BLOCK_SIZE;
    if (head_dim > MAX_HEAD_DIM) {
        throw std::invalid_argument(
            "FlashAttention OneAPI: head_dim must be <= "
            + std::to_string(MAX_HEAD_DIM) + ". Got "
            + std::to_string(head_dim));
    }

    // Local memory: K_tile[Bc][K_STRIDE] + V_tile[Bc][K_STRIDE] + scores[Bc]
    const size_t local_mem_size = static_cast<size_t>(
        2 * Bc * K_STRIDE + Bc) * sizeof(ComputeT);

    // Preflight the device's shared-local-memory capacity. local_mem_size grows
    // linearly with head_dim (K_STRIDE = head_dim + 4); for large head_dim it
    // can exceed the device SLM limit (e.g. 64KB on common Intel iGPU/Arc),
    // which would otherwise fail at kernel submission with an opaque SYCL
    // exception. Check here and throw a clear, attributed diagnostic instead.
    const size_t device_local_mem =
        queue.get_device().get_info<sycl::info::device::local_mem_size>();
    if (local_mem_size > device_local_mem) {
        throw std::runtime_error(
            "FlashAttention OneAPI: required shared local memory ("
            + std::to_string(local_mem_size)
            + " bytes for head_dim=" + std::to_string(head_dim)
            + ") exceeds device local_mem_size ("
            + std::to_string(device_local_mem)
            + " bytes). Reduce head_dim or use a device with more local memory.");
    }

    const DataT* q_ptr = get_data_ptr<const DataT>(Q);
    const DataT* k_ptr = get_data_ptr<const DataT>(K);
    const DataT* v_ptr = get_data_ptr<const DataT>(V);
    DataT* o_ptr = get_data_ptr<DataT>(output);

    // Phase 8.2: optional logsumexp output for the fused backward kernel.
    float* l_ptr = nullptr;
    if (L_out != nullptr && L_out->is_valid() && L_out->numel() > 0) {
        l_ptr = get_data_ptr<float>(*L_out);
    }

    // Audit F16: accept any mask shape broadcast-compatible with
    // [batch_heads, seq_q, seq_k]. The kernel itself reads from a 3D
    // [batch_heads, seq_q, seq_k] strided view, so when the caller supplies
    // a smaller mask (e.g. an attention mask [seq_q, seq_k] shared across
    // heads, or [1, 1, seq_k] for KV-padding-only) we broadcast it to the
    // full 3D shape host-side via the existing `broadcast_to` op. This
    // matches the CPU FlashAttention contract — previously, anything other
    // than the exact 3D shape threw on this code path.
    const float* mask_ptr = nullptr;
    int64_t mask_b_stride = 0, mask_q_stride = 0;
    Tensor broadcasted_mask;  // owns the (optional) materialised broadcast.
    if (mask != nullptr && mask->is_valid() && mask->numel() > 0) {
        if (mask->dtype() != DType::Float32) {
            throw std::invalid_argument(
                "FlashAttention OneAPI: mask must be Float32 (cast at host).");
        }
        std::vector<int64_t> target_shape = {batch_heads, seq_len_q, seq_len_k};
        std::vector<int64_t> mask_shape_vec(mask->shape().begin(), mask->shape().end());
        const Tensor* effective_mask = mask;
        if (mask_shape_vec != target_shape) {
            // Verify broadcast compatibility before materialising.
            try {
                auto bs = tenzor::broadcast_shapes(mask->shape(), std::span<const int64_t>(target_shape));
                if (bs != target_shape) {
                    throw std::invalid_argument(
                        "FlashAttention OneAPI: mask shape is not broadcast-compatible with "
                        "[batch_heads, seq_q, seq_k].");
                }
            } catch (const std::exception& e) {
                throw std::invalid_argument(
                    std::string("FlashAttention OneAPI: mask broadcast error: ") + e.what());
            }
            broadcasted_mask = tenzor::broadcast_to(*mask, target_shape).contiguous();
            effective_mask = &broadcasted_mask;
        }
        mask_ptr = get_data_ptr<const float>(*effective_mask);
        mask_b_stride = seq_len_q * seq_len_k;
        mask_q_stride = seq_len_k;
    }

    // 2D grid: (batch_heads, seq_len_q) — one work-group per query row
    queue.submit([&](sycl::handler& cgh) {
        sycl::local_accessor<ComputeT, 1> local_mem(local_mem_size / sizeof(ComputeT), cgh);

        const int hd = static_cast<int>(head_dim);
        const int slq = static_cast<int>(seq_len_q);
        const int slk = static_cast<int>(seq_len_k);
        const int ks = K_STRIDE;
        const bool causal = is_causal;
        const ComputeT sc = scale;
        const float* m_ptr = mask_ptr;
        const int64_t m_bstr = mask_b_stride;
        const int64_t m_qstr = mask_q_stride;
        float* lse_ptr = l_ptr;  // Phase 8.2: capture for kernel

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
                ComputeT* lmem = local_mem.template get_multi_ptr<sycl::access::decorated::no>().get();
                ComputeT* K_tile = lmem;                     // [Bc][ks]
                ComputeT* V_tile = lmem + Bc * ks;           // [Bc][ks]
                ComputeT* scores  = lmem + 2 * Bc * ks;      // [Bc]

                // Base pointers for this batch_head. Offsets are computed in
                // int64_t so that batch_heads*seq_len*head_dim products beyond
                // 2^31 (large-memory GPUs, >8GB Q/K/V buffers) do not wrap to a
                // negative int and index out of bounds.
                const int64_t q_off = static_cast<int64_t>(batch_head) * slq * hd
                                    + static_cast<int64_t>(query_idx) * hd;
                const int64_t kv_off = static_cast<int64_t>(batch_head) * slk * hd;
                const DataT* Q_row  = q_ptr + q_off;
                const DataT* K_base = k_ptr + kv_off;
                const DataT* V_base = v_ptr + kv_off;
                DataT* O_row        = o_ptr + q_off;

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
                        int kv_pos = k_start + j;
                        if (causal && kv_pos > query_idx) {
                            score = -std::numeric_limits<ComputeT>::infinity();
                        }
                        // Apply additive attn_mask (audit C4 OneAPI fix —
                        // was silently dropped). Mask is Float32; broadcast
                        // by promoting through ComputeT.
                        if (m_ptr != nullptr) {
                            int64_t off = static_cast<int64_t>(batch_head) * m_bstr
                                        + static_cast<int64_t>(query_idx) * m_qstr
                                        + static_cast<int64_t>(kv_pos);
                            score += static_cast<ComputeT>(m_ptr[off]);
                        }
                        scores[j] = score;
                        local_max = sycl::fmax(local_max, score);
                    }

                    // Reduce max across work-group. reduce_over_group is a
                    // collective with implicit barrier, so it pairs the
                    // scores[j] writes above with the reads below (audit
                    // High #7 OneAPI flagged a missing barrier here — adding
                    // an explicit one before the read for clarity).
                    ComputeT block_max = sycl::reduce_over_group(
                        item.get_group(), local_max,
                        sycl::maximum<ComputeT>());
                    sycl::group_barrier(item.get_group());

                    // If the entire KV tile is masked (every score -inf), block_max
                    // is -inf; skip the tile so exp(-inf - (-inf)) = NaN does not
                    // contaminate the running softmax (mirrors the FP64 kernel).
                    if (block_max == -std::numeric_limits<ComputeT>::infinity()) {
                        sycl::group_barrier(item.get_group());
                        continue;
                    }

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

                // Phase 8.2: emit LSE per (batch_head, query_idx) for the
                // fused backward kernel. LSE = log(sum_kv exp(scaled_score)) =
                // log(l_prev) + m_prev. Single-thread write (tid 0 only).
                if (lse_ptr != nullptr && tid == 0) {
                    float lse_val;
                    if (l_prev > ComputeT(0)) {
                        lse_val = static_cast<float>(sycl::log(l_prev) + m_prev);
                    } else {
                        // No valid attention positions (e.g. fully-masked row);
                        // emit -inf so backward's exp(s - lse) zeros out cleanly.
                        lse_val = -std::numeric_limits<float>::infinity();
                    }
                    lse_ptr[batch_head * slq + query_idx] = lse_val;
                }
            }
        );
    });

    return output;
}

// Phase 8.2: forward variant that also writes LSE per query row. LSE is the
// log-sum-exp of (scaled, masked) Q·K^T scores along the K axis — exactly the
// quantity the fused backward kernel needs to recompute attention weights as
// P = exp(S - LSE) without rerunning the softmax pass over all KV positions.
auto flash_attention_kernel_with_lse(
    const Tensor& Q,
    const Tensor& K,
    const Tensor& V,
    const Tensor* mask,
    float scale,
    bool is_causal,
    sycl::queue& queue,
    Tensor* L_out  // [batch_heads, seq_len_q] Float32
) -> Tensor {
    if (Q.shape().size() != 3 || K.shape().size() != 3 || V.shape().size() != 3) {
        throw std::invalid_argument(
            "flash_attention_kernel: Q, K, V must be 3D [batch_heads, seq_len, head_dim]");
    }
    if (Q.dtype() != K.dtype() || Q.dtype() != V.dtype()) {
        throw std::invalid_argument("flash_attention_kernel: Q, K, V must have the same dtype");
    }

    // Allocate LSE if requested but not yet allocated.
    if (L_out != nullptr && (!L_out->is_valid() || L_out->numel() == 0)) {
        const int64_t bh = Q.shape()[0];
        const int64_t sq = Q.shape()[1];
        *L_out = Tensor(std::vector<int64_t>{bh, sq}, DType::Float32, Q.device());
    }

    if (Q.dtype() == DType::Float32) {
        return flash_attention_impl<float, float, FlashAttentionKernelFloat32>(
            Q, K, V, mask, scale, is_causal, queue, L_out);
    } else if (Q.dtype() == DType::Float64) {
        return flash_attention_impl<double, double, FlashAttentionKernelFloat64>(
            Q, K, V, mask, scale, is_causal, queue, L_out);
    } else if (Q.dtype() == DType::Float16) {
        return flash_attention_impl<sycl::half, float, FlashAttentionKernelFloat16>(
            Q, K, V, mask, scale, is_causal, queue, L_out);
    } else if (Q.dtype() == DType::BFloat16) {
        return flash_attention_impl<uint16_t, float, FlashAttentionKernelBFloat16, true>(
            Q, K, V, mask, scale, is_causal, queue, L_out);
    } else {
        throw std::runtime_error("flash_attention_kernel: unsupported dtype");
    }
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
    Tensor lse_unused;
    return flash_attention_kernel_with_lse(Q, K, V, mask, scale, is_causal, queue, &lse_unused);
}

// ============================================================================
// Phase 8.2: Flash Attention Backward — fused tile-based SYCL kernel (Float32)
// ============================================================================

// Kernel name tag (one per (head_dim, dtype) combination would be ideal, but
// SYCL handles distinct lambdas per nd_range submission via the lambda type).
struct FlashAttentionBackwardKernelF32 {};

namespace fa_bwd {

// Tile sizes match the CUDA kernel for parity (Br=Bc=32, BLOCK_SIZE=128 to
// fit OneAPI's smaller default workgroup limits).
constexpr int Br_const = 32;
constexpr int Bc_const = 32;
constexpr int BLOCK_SIZE_const = 128;

// Per-thread accumulator capacity: ceil(Bc * HEAD_DIM / BLOCK_SIZE).
// HEAD_DIM <= 128 supported → 32*128/128 = 32 elements per thread.
constexpr int MAX_ELEMS_PER_THREAD_const = (Bc_const * 128 + BLOCK_SIZE_const - 1) / BLOCK_SIZE_const;

}  // namespace fa_bwd

/**
 * Fused FlashAttention backward (Float32).
 *
 * Algorithm mirrors src/backends/cuda/kernels/fused_ops.cu :: flash_attention_backward_kernel.
 * One workgroup per (batch_head, kv_tile). Iterates over Q tiles. Local memory
 * holds K/V tiles, Q/dO tiles, S-tile (Br × Bc), per-row LSE and D = rowsum(dO ⊙ O).
 * dK/dV accumulated in private memory then written; dQ via atomic_ref<float>.
 *
 * Working memory per workgroup: 2*Bc*HEAD_DIM + 2*Br*HEAD_DIM + Br*Bc + 2*Br floats.
 * For HEAD_DIM=128, Br=Bc=32: 2*32*128 + 2*32*128 + 32*32 + 64 = ~17.4K floats = 70K bytes.
 * Fits in 64KB SLM on most Intel GPUs (Iris Xe, Arc).
 */
auto flash_attention_backward_oneapi_f32(
    const Tensor& dO,    // [batch_heads, seq_len, head_dim]
    const Tensor& Q,
    const Tensor& K,
    const Tensor& V,
    const Tensor& O,
    const Tensor& L,     // [batch_heads, seq_len] logsumexp
    float scale,
    bool causal,
    sycl::queue& queue
) -> std::tuple<Tensor, Tensor, Tensor> {
    const int64_t batch_heads = Q.shape()[0];
    const int64_t seq_len = Q.shape()[1];
    const int64_t head_dim = Q.shape()[2];

    if (head_dim != 32 && head_dim != 64 && head_dim != 128) {
        throw std::runtime_error(
            "flash_attention_backward_oneapi_f32: head_dim must be 32, 64, or 128. Got "
            + std::to_string(head_dim));
    }

    using namespace fa_bwd;
    const int Br = Br_const;
    const int Bc = Bc_const;
    const int BLOCK_SIZE = BLOCK_SIZE_const;

    Tensor dQ(std::vector<int64_t>{batch_heads, seq_len, head_dim},
              DType::Float32, Q.device());
    Tensor dK(std::vector<int64_t>{batch_heads, seq_len, head_dim},
              DType::Float32, K.device());
    Tensor dV(std::vector<int64_t>{batch_heads, seq_len, head_dim},
              DType::Float32, V.device());

    // Zero-init dQ (atomicAdd target). dK/dV are fully written once per kv_tile.
    queue.memset(dQ.data_ptr(), 0, batch_heads * seq_len * head_dim * sizeof(float)).wait();

    const int num_kv_tiles = static_cast<int>((seq_len + Bc - 1) / Bc);
    const int hd = static_cast<int>(head_dim);
    const int sl = static_cast<int>(seq_len);

    // Local memory layout (floats):
    //   K_tile [Bc][hd] | V_tile [Bc][hd] | Q_tile [Br][hd] | dO_tile [Br][hd]
    //   | S_tile [Br][Bc] | l_tile [Br] | D_tile [Br]
    const size_t local_floats =
        static_cast<size_t>(2 * Bc * hd + 2 * Br * hd + Br * Bc + 2 * Br);

    const float* q_ptr  = get_data_ptr<const float>(Q);
    const float* k_ptr  = get_data_ptr<const float>(K);
    const float* v_ptr  = get_data_ptr<const float>(V);
    const float* o_ptr  = get_data_ptr<const float>(O);
    const float* do_ptr = get_data_ptr<const float>(dO);
    const float* l_ptr  = get_data_ptr<const float>(L);
    float* dq_ptr = get_data_ptr<float>(dQ);
    float* dk_ptr = get_data_ptr<float>(dK);
    float* dv_ptr = get_data_ptr<float>(dV);

    queue.submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float, 1> lmem(local_floats, cgh);

        const int hd_k = hd;
        const int sl_k = sl;
        const float sc_k = scale;
        const bool causal_k = causal;
        const int num_q_tiles = (sl + Br - 1) / Br;

        cgh.parallel_for<FlashAttentionBackwardKernelF32>(
            sycl::nd_range<2>(
                sycl::range<2>(static_cast<size_t>(batch_heads),
                               static_cast<size_t>(num_kv_tiles) * BLOCK_SIZE),
                sycl::range<2>(1, BLOCK_SIZE)),
            [=](sycl::nd_item<2> item) {
                const int batch_head = static_cast<int>(item.get_global_id(0));
                const int kv_tile_idx = static_cast<int>(item.get_global_id(1)) / BLOCK_SIZE;
                const int tid = static_cast<int>(item.get_local_id(1));

                const int kv_start = kv_tile_idx * Bc;
                if (kv_start >= sl_k) return;
                const int actual_Bc = sycl::min(Bc, sl_k - kv_start);

                // Per-batch_head base offset in int64_t: batch_head*sl_k*hd_k can
                // exceed 2^31 for large batch_heads*seq_len even with bounded
                // head_dim, and int*int*int arithmetic would wrap negative and
                // index out of bounds.
                const int64_t bh_off = static_cast<int64_t>(batch_head)
                                     * sl_k * hd_k;
                const float* Q_base  = q_ptr  + bh_off;
                const float* K_base  = k_ptr  + bh_off;
                const float* V_base  = v_ptr  + bh_off;
                const float* O_base  = o_ptr  + bh_off;
                const float* dO_base = do_ptr + bh_off;
                const float* L_base  = l_ptr  + static_cast<int64_t>(batch_head) * sl_k;
                float* dQ_base = dq_ptr + bh_off;
                float* dK_base = dk_ptr + bh_off;
                float* dV_base = dv_ptr + bh_off;

                float* slm = lmem.template get_multi_ptr<sycl::access::decorated::no>().get();
                float* K_tile  = slm;
                float* V_tile  = K_tile  + Bc * hd_k;
                float* Q_tile  = V_tile  + Bc * hd_k;
                float* dO_tile = Q_tile  + Br * hd_k;
                float* S_tile  = dO_tile + Br * hd_k;
                float* l_tile  = S_tile  + Br * Bc;
                float* D_tile  = l_tile  + Br;

                // Load K_j, V_j tiles
                for (int i = tid; i < actual_Bc * hd_k; i += BLOCK_SIZE) {
                    int row = i / hd_k;
                    int col = i % hd_k;
                    K_tile[row * hd_k + col] = K_base[(kv_start + row) * hd_k + col];
                    V_tile[row * hd_k + col] = V_base[(kv_start + row) * hd_k + col];
                }
                for (int i = tid + actual_Bc * hd_k; i < Bc * hd_k; i += BLOCK_SIZE) {
                    K_tile[i] = 0.0f;
                    V_tile[i] = 0.0f;
                }
                sycl::group_barrier(item.get_group());

                // Per-thread dK/dV accumulators
                float dk_acc[MAX_ELEMS_PER_THREAD_const];
                float dv_acc[MAX_ELEMS_PER_THREAD_const];
                for (int e = 0; e < MAX_ELEMS_PER_THREAD_const; ++e) {
                    dk_acc[e] = 0.0f;
                    dv_acc[e] = 0.0f;
                }

                for (int q_tile_idx = 0; q_tile_idx < num_q_tiles; ++q_tile_idx) {
                    const int q_start = q_tile_idx * Br;
                    if (q_start >= sl_k) break;
                    const int actual_Br = sycl::min(Br, sl_k - q_start);

                    // Causal early-exit: skip if all Q rows precede all K cols
                    if (causal_k && (q_start + actual_Br - 1) < kv_start) continue;

                    // Load Q_i, dO_i tiles
                    for (int i = tid; i < actual_Br * hd_k; i += BLOCK_SIZE) {
                        int row = i / hd_k;
                        int col = i % hd_k;
                        Q_tile[row * hd_k + col]  = Q_base[(q_start + row) * hd_k + col];
                        dO_tile[row * hd_k + col] = dO_base[(q_start + row) * hd_k + col];
                    }
                    for (int i = tid + actual_Br * hd_k; i < Br * hd_k; i += BLOCK_SIZE) {
                        Q_tile[i] = 0.0f;
                        dO_tile[i] = 0.0f;
                    }

                    // Load LSE and compute D_i = sum_d (dO_i[d] * O_i[d])
                    for (int row = tid; row < actual_Br; row += BLOCK_SIZE) {
                        l_tile[row] = L_base[q_start + row];
                        float dsum = 0.0f;
                        for (int d = 0; d < hd_k; ++d) {
                            dsum += dO_base[(q_start + row) * hd_k + d]
                                  * O_base[(q_start + row) * hd_k + d];
                        }
                        D_tile[row] = dsum;
                    }
                    for (int row = tid + actual_Br; row < Br; row += BLOCK_SIZE) {
                        l_tile[row] = -std::numeric_limits<float>::infinity();
                        D_tile[row] = 0.0f;
                    }
                    sycl::group_barrier(item.get_group());

                    // S_ij = Q_i · K_j^T * scale
                    for (int idx = tid; idx < actual_Br * actual_Bc; idx += BLOCK_SIZE) {
                        int i = idx / actual_Bc;
                        int j = idx % actual_Bc;
                        float dot = 0.0f;
                        for (int d = 0; d < hd_k; ++d) {
                            dot += Q_tile[i * hd_k + d] * K_tile[j * hd_k + d];
                        }
                        S_tile[i * Bc + j] = dot * sc_k;
                    }
                    // Out-of-bounds entries -> -inf so exp() gives 0
                    for (int idx = tid; idx < Br * Bc; idx += BLOCK_SIZE) {
                        int i = idx / Bc;
                        int j = idx % Bc;
                        if (i >= actual_Br || j >= actual_Bc) {
                            S_tile[idx] = -std::numeric_limits<float>::infinity();
                        }
                    }
                    sycl::group_barrier(item.get_group());

                    // P_ij = exp(S_ij - L_i) (with causal mask -> 0)
                    for (int idx = tid; idx < Br * Bc; idx += BLOCK_SIZE) {
                        int i = idx / Bc;
                        int j = idx % Bc;
                        float p = 0.0f;
                        if (i < actual_Br && j < actual_Bc) {
                            if (causal_k && (q_start + i) < (kv_start + j)) {
                                p = 0.0f;
                            } else {
                                p = sycl::exp(S_tile[i * Bc + j] - l_tile[i]);
                            }
                        }
                        S_tile[i * Bc + j] = p;  // reuse S_tile as P_ij
                    }
                    sycl::group_barrier(item.get_group());

                    // dV_j += P_ij^T · dO_i  → per-thread (j, d) accumulators
                    {
                        int e = 0;
                        for (int idx = tid; idx < Bc * hd_k; idx += BLOCK_SIZE, ++e) {
                            int j = idx / hd_k;
                            int d = idx % hd_k;
                            if (j < actual_Bc) {
                                float sum = 0.0f;
                                for (int i = 0; i < actual_Br; ++i) {
                                    sum += S_tile[i * Bc + j] * dO_tile[i * hd_k + d];
                                }
                                dv_acc[e] += sum;
                            }
                        }
                    }
                    sycl::group_barrier(item.get_group());

                    // dS_ij = P_ij * (dP_ij - D_i) where dP_ij = dO_i · V_j
                    for (int idx = tid; idx < actual_Br * actual_Bc; idx += BLOCK_SIZE) {
                        int i = idx / actual_Bc;
                        int j = idx % actual_Bc;
                        float dp = 0.0f;
                        for (int d = 0; d < hd_k; ++d) {
                            dp += dO_tile[i * hd_k + d] * V_tile[j * hd_k + d];
                        }
                        float p_ij = S_tile[i * Bc + j];
                        S_tile[i * Bc + j] = p_ij * (dp - D_tile[i]);
                    }
                    sycl::group_barrier(item.get_group());

                    // dK_j += dS_ij^T · Q_i * scale
                    {
                        int e = 0;
                        for (int idx = tid; idx < Bc * hd_k; idx += BLOCK_SIZE, ++e) {
                            int j = idx / hd_k;
                            int d = idx % hd_k;
                            if (j < actual_Bc) {
                                float sum = 0.0f;
                                for (int i = 0; i < actual_Br; ++i) {
                                    sum += S_tile[i * Bc + j] * Q_tile[i * hd_k + d];
                                }
                                dk_acc[e] += sum * sc_k;
                            }
                        }
                    }

                    // dQ_i += dS_ij · K_j * scale (atomic, multiple kv_tiles contribute)
                    for (int idx = tid; idx < actual_Br * hd_k; idx += BLOCK_SIZE) {
                        int i = idx / hd_k;
                        int d = idx % hd_k;
                        float sum = 0.0f;
                        for (int j = 0; j < actual_Bc; ++j) {
                            sum += S_tile[i * Bc + j] * K_tile[j * hd_k + d];
                        }
                        sycl::atomic_ref<float, sycl::memory_order::relaxed,
                                         sycl::memory_scope::device,
                                         sycl::access::address_space::global_space>
                            atomic_dq(dQ_base[(q_start + i) * hd_k + d]);
                        atomic_dq.fetch_add(sum * sc_k);
                    }
                    sycl::group_barrier(item.get_group());
                }

                // Write accumulated dK_j, dV_j to global memory
                {
                    int e = 0;
                    for (int idx = tid; idx < Bc * hd_k; idx += BLOCK_SIZE, ++e) {
                        int row = idx / hd_k;
                        int col = idx % hd_k;
                        if (row < actual_Bc) {
                            dK_base[(kv_start + row) * hd_k + col] = dk_acc[e];
                            dV_base[(kv_start + row) * hd_k + col] = dv_acc[e];
                        }
                    }
                }
            });
    }).wait();

    return {dQ, dK, dV};
}

} // namespace oneapi
} // namespace tenzor
