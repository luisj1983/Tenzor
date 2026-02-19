/**
 * @file cudnn_batchnorm.cu
 * @brief cuDNN optimized batch normalization operations
 *
 * Provides 2-5x speedup over custom CUDA kernels through:
 * - Highly optimized cuDNN implementation
 * - Better memory access patterns
 * - Hardware-specific optimizations
 * - Fused operations for training and inference
 */

#ifdef TENZOR_HAS_CUDNN

#include "tenzor/backend/cudnn_wrapper.hpp"
#include "tenzor/core/tensor.hpp"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <vector>

namespace tenzor {
namespace cuda {

// Direct FP32->FP16 copy kernel — avoids intermediate tensor allocation
__global__ void f32_to_half_kernel(const float* src, __half* dst, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < n; i += stride) {
        dst[i] = __float2half(src[i]);
    }
}

// Direct FP32->BF16 copy kernel — avoids intermediate tensor allocation
__global__ void f32_to_bfloat16_kernel(const float* src, __nv_bfloat16* dst, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < n; i += stride) {
        dst[i] = __float2bfloat16(src[i]);
    }
}

// ============================================================================
// cuDNN Batch Normalization Inference (Optimized)
// ============================================================================

/**
 * @brief cuDNN-optimized BatchNorm inference forward pass
 *
 * Uses cudnnBatchNormalizationForwardInference for maximum performance.
 * This is a fused operation that applies: y = gamma * (x - mean) / sqrt(var + eps) + beta
 *
 * @param input Input tensor (N, C, H, W) in NCHW format
 * @param running_mean Running mean (C,)
 * @param running_var Running variance (C,)
 * @param gamma Scale parameter (C,)
 * @param beta Bias parameter (C,)
 * @param epsilon Epsilon for numerical stability
 * @param stream CUDA stream
 * @return Output tensor (N, C, H, W)
 */
auto cudnn_batchnorm2d_forward_inference(
    const Tensor& input,
    const Tensor& running_mean,
    const Tensor& running_var,
    const Tensor& gamma,
    const Tensor& beta,
    float epsilon,
    cudaStream_t stream
) -> Tensor {
    // Get cuDNN handle and set stream
    cudnnHandle_t handle = CuDNNHandle::get();
    if (stream) {
        CuDNNHandle::set_stream(stream);
    }

    // Extract dimensions (NCHW format)
    auto shape = input.shape();
    int N = static_cast<int>(shape[0]);
    int C = static_cast<int>(shape[1]);
    int H = static_cast<int>(shape[2]);
    int W = static_cast<int>(shape[3]);

    // Determine cuDNN data type
    cudnnDataType_t cudnn_dtype;
    switch (input.dtype()) {
        case DType::Float32: cudnn_dtype = CUDNN_DATA_FLOAT; break;
        case DType::Float64: cudnn_dtype = CUDNN_DATA_DOUBLE; break;
        case DType::Float16: cudnn_dtype = CUDNN_DATA_HALF; break;
        case DType::BFloat16: cudnn_dtype = CUDNN_DATA_BFLOAT16; break;
        default:
            throw std::runtime_error("cuDNN BatchNorm: unsupported dtype");
    }

    // Create tensor descriptors
    TensorDescriptor input_desc, output_desc;
    input_desc.set(cudnn_dtype, N, C, H, W);
    output_desc.set(cudnn_dtype, N, C, H, W);

    // Create BN parameter descriptor (derived from input descriptor)
    cudnnTensorDescriptor_t bn_desc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&bn_desc));
    CUDNN_CHECK(cudnnDeriveBNTensorDescriptor(
        bn_desc,
        input_desc.get(),
        CUDNN_BATCHNORM_SPATIAL  // Per-channel normalization (standard for conv layers)
    ));

    // Create output tensor
    Tensor output(std::vector<int64_t>(shape.begin(), shape.end()), input.dtype(), input.device());

    // Ensure tensors are contiguous
    Tensor input_c = input.is_contiguous() ? input : input.contiguous();
    Tensor mean_c = running_mean.is_contiguous() ? running_mean : running_mean.contiguous();
    Tensor var_c = running_var.is_contiguous() ? running_var : running_var.contiguous();
    Tensor gamma_c = gamma.is_contiguous() ? gamma : gamma.contiguous();
    Tensor beta_c = beta.is_contiguous() ? beta : beta.contiguous();

    if (input.dtype() == DType::Float32) {
        const float alpha = 1.0f;
        const float zero = 0.0f;

        CUDNN_CHECK(cudnnBatchNormalizationForwardInference(
            handle,
            CUDNN_BATCHNORM_SPATIAL,
            &alpha,
            &zero,
            input_desc.get(),
            input_c.data<float>(),
            output_desc.get(),
            output.data<float>(),
            bn_desc,
            gamma_c.data<float>(),
            beta_c.data<float>(),
            mean_c.data<float>(),
            var_c.data<float>(),
            epsilon
        ));
    } else if (input.dtype() == DType::Float64) {
        const double alpha = 1.0;
        const double zero = 0.0;

        CUDNN_CHECK(cudnnBatchNormalizationForwardInference(
            handle,
            CUDNN_BATCHNORM_SPATIAL,
            &alpha,
            &zero,
            input_desc.get(),
            input_c.data<double>(),
            output_desc.get(),
            output.data<double>(),
            bn_desc,
            gamma_c.data<double>(),
            beta_c.data<double>(),
            mean_c.data<double>(),
            var_c.data<double>(),
            epsilon
        ));
    } else if (input.dtype() == DType::Float16) {
        // cuDNN mandates FP32 for bnScale, bnBias, estimatedMean, estimatedVariance
        // when the input tensor is FP16 (see cudnnBatchNormalizationForwardInference docs).
        // The conversion cost is negligible since these are small per-channel tensors
        // (C elements), not full activation tensors.
        const float alpha = 1.0f;
        const float zero = 0.0f;

        // Convert parameters to Float32 as required by cuDNN
        Tensor gamma_f32 = gamma_c.to(DType::Float32);
        Tensor beta_f32 = beta_c.to(DType::Float32);
        Tensor mean_f32 = mean_c.to(DType::Float32);
        Tensor var_f32 = var_c.to(DType::Float32);

        CUDNN_CHECK(cudnnBatchNormalizationForwardInference(
            handle,
            CUDNN_BATCHNORM_SPATIAL,
            &alpha,
            &zero,
            input_desc.get(),
            input_c.data_ptr(),
            output_desc.get(),
            output.data_ptr(),
            bn_desc,
            gamma_f32.data<float>(),
            beta_f32.data<float>(),
            mean_f32.data<float>(),
            var_f32.data<float>(),
            epsilon
        ));
    } else if (input.dtype() == DType::BFloat16) {
        // cuDNN mandates FP32 for bnScale, bnBias, estimatedMean, estimatedVariance
        // when the input tensor is BF16 (same as FP16 case).
        const float alpha = 1.0f;
        const float zero = 0.0f;

        // Convert parameters to Float32 as required by cuDNN
        Tensor gamma_f32 = gamma_c.to(DType::Float32);
        Tensor beta_f32 = beta_c.to(DType::Float32);
        Tensor mean_f32 = mean_c.to(DType::Float32);
        Tensor var_f32 = var_c.to(DType::Float32);

        CUDNN_CHECK(cudnnBatchNormalizationForwardInference(
            handle,
            CUDNN_BATCHNORM_SPATIAL,
            &alpha,
            &zero,
            input_desc.get(),
            input_c.data_ptr(),
            output_desc.get(),
            output.data_ptr(),
            bn_desc,
            gamma_f32.data<float>(),
            beta_f32.data<float>(),
            mean_f32.data<float>(),
            var_f32.data<float>(),
            epsilon
        ));
    }

    // Cleanup
    cudnnDestroyTensorDescriptor(bn_desc);

    return output;
}

// ============================================================================
// cuDNN Batch Normalization Training Forward
// ============================================================================

/**
 * @brief cuDNN-optimized BatchNorm training forward pass
 *
 * Computes batch statistics, updates running statistics, and normalizes.
 * Returns saved mean and inv_variance for backward pass.
 *
 * @param input Input tensor (N, C, H, W)
 * @param running_mean Running mean to update (C,)
 * @param running_var Running variance to update (C,)
 * @param gamma Scale parameter (C,)
 * @param beta Bias parameter (C,)
 * @param momentum Momentum for running statistics update
 * @param epsilon Epsilon for numerical stability
 * @param stream CUDA stream
 * @return Tuple of (output, saved_mean, saved_inv_variance)
 */
auto cudnn_batchnorm2d_forward_training(
    const Tensor& input,
    Tensor& running_mean,
    Tensor& running_var,
    const Tensor& gamma,
    const Tensor& beta,
    float momentum,
    float epsilon,
    cudaStream_t stream
) -> std::tuple<Tensor, Tensor, Tensor> {
    cudnnHandle_t handle = CuDNNHandle::get();
    if (stream) {
        CuDNNHandle::set_stream(stream);
    }

    auto shape = input.shape();
    int N = static_cast<int>(shape[0]);
    int C = static_cast<int>(shape[1]);
    int H = static_cast<int>(shape[2]);
    int W = static_cast<int>(shape[3]);

    cudnnDataType_t cudnn_dtype;
    switch (input.dtype()) {
        case DType::Float32: cudnn_dtype = CUDNN_DATA_FLOAT; break;
        case DType::Float64: cudnn_dtype = CUDNN_DATA_DOUBLE; break;
        case DType::Float16: cudnn_dtype = CUDNN_DATA_HALF; break;
        case DType::BFloat16: cudnn_dtype = CUDNN_DATA_BFLOAT16; break;
        default:
            throw std::runtime_error("cuDNN BatchNorm: unsupported dtype");
    }

    TensorDescriptor input_desc, output_desc;
    input_desc.set(cudnn_dtype, N, C, H, W);
    output_desc.set(cudnn_dtype, N, C, H, W);

    cudnnTensorDescriptor_t bn_desc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&bn_desc));
    CUDNN_CHECK(cudnnDeriveBNTensorDescriptor(
        bn_desc,
        input_desc.get(),
        CUDNN_BATCHNORM_SPATIAL
    ));

    // Create output tensor (same dtype as input)
    Tensor output(std::vector<int64_t>(shape.begin(), shape.end()), input.dtype(), input.device());

    // For Float16/BFloat16 input, cuDNN requires saved stats to be Float32
    DType stats_dtype = (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16) ? DType::Float32 : input.dtype();
    Tensor saved_mean({C}, stats_dtype, input.device());
    Tensor saved_inv_var({C}, stats_dtype, input.device());

    Tensor input_c = input.is_contiguous() ? input : input.contiguous();
    Tensor gamma_c = gamma.is_contiguous() ? gamma : gamma.contiguous();
    Tensor beta_c = beta.is_contiguous() ? beta : beta.contiguous();

    if (input.dtype() == DType::Float32) {
        const float alpha = 1.0f;
        const float zero = 0.0f;

        CUDNN_CHECK(cudnnBatchNormalizationForwardTraining(
            handle,
            CUDNN_BATCHNORM_SPATIAL,
            &alpha,
            &zero,
            input_desc.get(),
            input_c.data<float>(),
            output_desc.get(),
            output.data<float>(),
            bn_desc,
            gamma_c.data<float>(),
            beta_c.data<float>(),
            momentum,  // exponentialAverageFactor
            running_mean.data<float>(),
            running_var.data<float>(),
            epsilon,
            saved_mean.data<float>(),
            saved_inv_var.data<float>()
        ));
    } else if (input.dtype() == DType::Float64) {
        const double alpha = 1.0;
        const double zero = 0.0;

        CUDNN_CHECK(cudnnBatchNormalizationForwardTraining(
            handle,
            CUDNN_BATCHNORM_SPATIAL,
            &alpha,
            &zero,
            input_desc.get(),
            input_c.data<double>(),
            output_desc.get(),
            output.data<double>(),
            bn_desc,
            gamma_c.data<double>(),
            beta_c.data<double>(),
            static_cast<double>(momentum),
            running_mean.data<double>(),
            running_var.data<double>(),
            static_cast<double>(epsilon),
            saved_mean.data<double>(),
            saved_inv_var.data<double>()
        ));
    } else if (input.dtype() == DType::Float16) {
        // For FP16 input, cuDNN requires FP32 for all BN parameters
        const float alpha = 1.0f;
        const float zero = 0.0f;

        // Convert parameters to Float32 as required by cuDNN
        Tensor gamma_f32 = gamma_c.to(DType::Float32);
        Tensor beta_f32 = beta_c.to(DType::Float32);

        // Convert running stats to Float32, update, then copy back
        Tensor running_mean_f32 = running_mean.to(DType::Float32);
        Tensor running_var_f32 = running_var.to(DType::Float32);

        CUDNN_CHECK(cudnnBatchNormalizationForwardTraining(
            handle,
            CUDNN_BATCHNORM_SPATIAL,
            &alpha,
            &zero,
            input_desc.get(),
            input_c.data_ptr(),
            output_desc.get(),
            output.data_ptr(),
            bn_desc,
            gamma_f32.data<float>(),
            beta_f32.data<float>(),
            momentum,
            running_mean_f32.data<float>(),
            running_var_f32.data<float>(),
            epsilon,
            saved_mean.data<float>(),
            saved_inv_var.data<float>()
        ));

        // Copy updated running stats directly from FP32 to FP16 — no intermediate tensors
        int64_t stat_n = running_mean.numel();
        int block = 256;
        int grid = (stat_n + block - 1) / block;
        f32_to_half_kernel<<<grid, block, 0, stream>>>(
            running_mean_f32.data<float>(),
            reinterpret_cast<__half*>(running_mean.data_ptr()),
            stat_n);
        f32_to_half_kernel<<<grid, block, 0, stream>>>(
            running_var_f32.data<float>(),
            reinterpret_cast<__half*>(running_var.data_ptr()),
            stat_n);
    } else if (input.dtype() == DType::BFloat16) {
        // For BF16 input, cuDNN requires FP32 for all BN parameters
        const float alpha = 1.0f;
        const float zero = 0.0f;

        // Convert parameters to Float32 as required by cuDNN
        Tensor gamma_f32 = gamma_c.to(DType::Float32);
        Tensor beta_f32 = beta_c.to(DType::Float32);

        // Convert running stats to Float32, update, then copy back
        Tensor running_mean_f32 = running_mean.to(DType::Float32);
        Tensor running_var_f32 = running_var.to(DType::Float32);

        CUDNN_CHECK(cudnnBatchNormalizationForwardTraining(
            handle,
            CUDNN_BATCHNORM_SPATIAL,
            &alpha,
            &zero,
            input_desc.get(),
            input_c.data_ptr(),
            output_desc.get(),
            output.data_ptr(),
            bn_desc,
            gamma_f32.data<float>(),
            beta_f32.data<float>(),
            momentum,
            running_mean_f32.data<float>(),
            running_var_f32.data<float>(),
            epsilon,
            saved_mean.data<float>(),
            saved_inv_var.data<float>()
        ));

        // Copy updated running stats directly from FP32 to BF16 — no intermediate tensors
        int64_t stat_n = running_mean.numel();
        int block = 256;
        int grid = (stat_n + block - 1) / block;
        f32_to_bfloat16_kernel<<<grid, block, 0, stream>>>(
            running_mean_f32.data<float>(),
            reinterpret_cast<__nv_bfloat16*>(running_mean.data_ptr()),
            stat_n);
        f32_to_bfloat16_kernel<<<grid, block, 0, stream>>>(
            running_var_f32.data<float>(),
            reinterpret_cast<__nv_bfloat16*>(running_var.data_ptr()),
            stat_n);
    }

    cudnnDestroyTensorDescriptor(bn_desc);

    return {output, saved_mean, saved_inv_var};
}

// ============================================================================
// cuDNN Batch Normalization Backward
// ============================================================================

/**
 * @brief cuDNN-optimized BatchNorm backward pass
 *
 * Computes gradients with respect to input, gamma, and beta.
 *
 * @param grad_output Gradient from next layer (N, C, H, W)
 * @param input Original input (N, C, H, W)
 * @param gamma Scale parameter (C,)
 * @param saved_mean Saved mean from forward (C,)
 * @param saved_inv_var Saved inverse variance from forward (C,)
 * @param epsilon Epsilon used in forward
 * @param stream CUDA stream
 * @return Tuple of (grad_input, grad_gamma, grad_beta)
 */
auto cudnn_batchnorm2d_backward(
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& gamma,
    const Tensor& saved_mean,
    const Tensor& saved_inv_var,
    float epsilon,
    cudaStream_t stream
) -> std::tuple<Tensor, Tensor, Tensor> {
    cudnnHandle_t handle = CuDNNHandle::get();
    if (stream) {
        CuDNNHandle::set_stream(stream);
    }

    auto shape = input.shape();
    int N = static_cast<int>(shape[0]);
    int C = static_cast<int>(shape[1]);
    int H = static_cast<int>(shape[2]);
    int W = static_cast<int>(shape[3]);

    cudnnDataType_t cudnn_dtype;
    switch (input.dtype()) {
        case DType::Float32: cudnn_dtype = CUDNN_DATA_FLOAT; break;
        case DType::Float64: cudnn_dtype = CUDNN_DATA_DOUBLE; break;
        case DType::Float16: cudnn_dtype = CUDNN_DATA_HALF; break;
        case DType::BFloat16: cudnn_dtype = CUDNN_DATA_BFLOAT16; break;
        default:
            throw std::runtime_error("cuDNN BatchNorm backward: unsupported dtype");
    }

    TensorDescriptor input_desc, grad_desc;
    input_desc.set(cudnn_dtype, N, C, H, W);
    grad_desc.set(cudnn_dtype, N, C, H, W);

    cudnnTensorDescriptor_t bn_desc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&bn_desc));
    CUDNN_CHECK(cudnnDeriveBNTensorDescriptor(
        bn_desc,
        input_desc.get(),
        CUDNN_BATCHNORM_SPATIAL
    ));

    // Create gradient tensors (same dtype as input to match original parameters)
    Tensor grad_input(std::vector<int64_t>(shape.begin(), shape.end()), input.dtype(), input.device());
    Tensor grad_gamma({C}, input.dtype(), input.device());
    Tensor grad_beta({C}, input.dtype(), input.device());

    Tensor grad_out_c = grad_output.is_contiguous() ? grad_output : grad_output.contiguous();
    Tensor input_c = input.is_contiguous() ? input : input.contiguous();
    Tensor gamma_c = gamma.is_contiguous() ? gamma : gamma.contiguous();
    Tensor mean_c = saved_mean.is_contiguous() ? saved_mean : saved_mean.contiguous();
    Tensor inv_var_c = saved_inv_var.is_contiguous() ? saved_inv_var : saved_inv_var.contiguous();

    if (input.dtype() == DType::Float32) {
        const float alpha = 1.0f;
        const float zero = 0.0f;

        CUDNN_CHECK(cudnnBatchNormalizationBackward(
            handle,
            CUDNN_BATCHNORM_SPATIAL,
            &alpha,  // alphaDataDiff
            &zero,   // betaDataDiff
            &alpha,  // alphaParamDiff
            &zero,   // betaParamDiff
            input_desc.get(),
            input_c.data<float>(),
            grad_desc.get(),
            grad_out_c.data<float>(),
            input_desc.get(),
            grad_input.data<float>(),
            bn_desc,
            gamma_c.data<float>(),
            grad_gamma.data<float>(),
            grad_beta.data<float>(),
            epsilon,
            mean_c.data<float>(),
            inv_var_c.data<float>()
        ));
    } else if (input.dtype() == DType::Float64) {
        const double alpha = 1.0;
        const double zero = 0.0;

        CUDNN_CHECK(cudnnBatchNormalizationBackward(
            handle,
            CUDNN_BATCHNORM_SPATIAL,
            &alpha,
            &zero,
            &alpha,
            &zero,
            input_desc.get(),
            input_c.data<double>(),
            grad_desc.get(),
            grad_out_c.data<double>(),
            input_desc.get(),
            grad_input.data<double>(),
            bn_desc,
            gamma_c.data<double>(),
            grad_gamma.data<double>(),
            grad_beta.data<double>(),
            static_cast<double>(epsilon),
            mean_c.data<double>(),
            inv_var_c.data<double>()
        ));
    } else if (input.dtype() == DType::Float16) {
        // For FP16 input, cuDNN requires FP32 for gamma, saved_mean, saved_inv_var
        const float alpha = 1.0f;
        const float zero = 0.0f;

        // Convert parameters to Float32 as required by cuDNN
        Tensor gamma_f32 = gamma_c.to(DType::Float32);
        Tensor mean_f32 = mean_c.to(DType::Float32);
        Tensor inv_var_f32 = inv_var_c.to(DType::Float32);

        // Temporary Float32 tensors for gradients
        Tensor grad_gamma_f32({C}, DType::Float32, input.device());
        Tensor grad_beta_f32({C}, DType::Float32, input.device());

        CUDNN_CHECK(cudnnBatchNormalizationBackward(
            handle,
            CUDNN_BATCHNORM_SPATIAL,
            &alpha,
            &zero,
            &alpha,
            &zero,
            input_desc.get(),
            input_c.data_ptr(),
            grad_desc.get(),
            grad_out_c.data_ptr(),
            input_desc.get(),
            grad_input.data_ptr(),
            bn_desc,
            gamma_f32.data<float>(),
            grad_gamma_f32.data<float>(),
            grad_beta_f32.data<float>(),
            epsilon,
            mean_f32.data<float>(),
            inv_var_f32.data<float>()
        ));

        // Convert gradients back to Float16 to match original parameter dtypes
        grad_gamma = grad_gamma_f32.to(DType::Float16);
        grad_beta = grad_beta_f32.to(DType::Float16);
    } else if (input.dtype() == DType::BFloat16) {
        // For BF16 input, cuDNN requires FP32 for gamma, saved_mean, saved_inv_var
        const float alpha = 1.0f;
        const float zero = 0.0f;

        // Convert parameters to Float32 as required by cuDNN
        Tensor gamma_f32 = gamma_c.to(DType::Float32);
        Tensor mean_f32 = mean_c.to(DType::Float32);
        Tensor inv_var_f32 = inv_var_c.to(DType::Float32);

        // Temporary Float32 tensors for gradients
        Tensor grad_gamma_f32({C}, DType::Float32, input.device());
        Tensor grad_beta_f32({C}, DType::Float32, input.device());

        CUDNN_CHECK(cudnnBatchNormalizationBackward(
            handle,
            CUDNN_BATCHNORM_SPATIAL,
            &alpha,
            &zero,
            &alpha,
            &zero,
            input_desc.get(),
            input_c.data_ptr(),
            grad_desc.get(),
            grad_out_c.data_ptr(),
            input_desc.get(),
            grad_input.data_ptr(),
            bn_desc,
            gamma_f32.data<float>(),
            grad_gamma_f32.data<float>(),
            grad_beta_f32.data<float>(),
            epsilon,
            mean_f32.data<float>(),
            inv_var_f32.data<float>()
        ));

        // Convert gradients back to BFloat16 to match original parameter dtypes
        grad_gamma = grad_gamma_f32.to(DType::BFloat16);
        grad_beta = grad_beta_f32.to(DType::BFloat16);
    }

    cudnnDestroyTensorDescriptor(bn_desc);

    return {grad_input, grad_gamma, grad_beta};
}

// ============================================================================
// Wrapper functions matching existing kernel signatures
// ============================================================================

/**
 * @brief Wrapper that matches batchnorm2d_forward_affine signature
 *
 * This function adapts the cuDNN implementation to match the existing
 * kernel registry interface.
 */
auto cudnn_batchnorm2d_forward_affine_wrapper(
    const Tensor& input,
    const Tensor& mean,
    const Tensor& variance,
    const Tensor& gamma,
    const Tensor& beta,
    float epsilon,
    cudaStream_t stream
) -> Tensor {
    return cudnn_batchnorm2d_forward_inference(
        input, mean, variance, gamma, beta, epsilon, stream
    );
}

} // namespace cuda
} // namespace tenzor

#endif // TENZOR_HAS_CUDNN
