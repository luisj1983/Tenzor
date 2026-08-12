#ifdef TENZOR_HAS_CUDNN

#include <array>
#include "tenzor/backend/cudnn_wrapper.hpp"
#include "tenzor/backend/cuda_caching_allocator.hpp"
#include "tenzor/backend/cuda_config.hpp"
#include "tenzor/core/tensor.hpp"
#include "tenzor/core/shape.hpp"
#include "cuda_error.hpp"
#include "kernels/cuda_launch_utils.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <algorithm>
#include <functional>
#include <mutex>
#include <unordered_map>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace tenzor {
namespace cuda {

// Forward declarations for activation kernels (defined in kernels/activations.cu)
auto sigmoid_kernel(const Tensor& input, cudaStream_t stream) -> Tensor;
auto tanh_kernel(const Tensor& input, cudaStream_t stream) -> Tensor;
auto swish_kernel(const Tensor& input, cudaStream_t stream) -> Tensor;

// ============================================================================
// FP16 Saturating Clamp Kernel
// ============================================================================

// Clamps FP16 values to max finite range, replacing ±Inf with ±65504.
// This provides saturating Float16 behavior for cuDNN outputs, matching CPU
// Float16 operator overloads which naturally clamp per-element.
__global__ void fp16_saturate_kernel(__half* data, int64_t n) {
    constexpr float kHalfMax = 65504.0f;
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += blockDim.x * gridDim.x) {
        float val = __half2float(data[idx]);
        if (val > kHalfMax || val < -kHalfMax) {
            data[idx] = __float2half(fminf(fmaxf(val, -kHalfMax), kHalfMax));
        }
    }
}

// Launch FP16 saturation on a tensor
static void fp16_saturate(Float16* data, int64_t n, cudaStream_t stream) {
    if (n <= 0) return;
    auto [grid, block] = optimal_launch_config(fp16_saturate_kernel, n);
    fp16_saturate_kernel<<<grid, block, 0, stream>>>(reinterpret_cast<__half*>(data), n);
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// NCHW <-> NHWC Format Conversion Kernels
// ============================================================================

/**
 * @brief Convert NCHW to NHWC format kernel
 *
 * Input layout:  [N][C][H][W] - contiguous in W, then H, then C, then N
 * Output layout: [N][H][W][C] - contiguous in C, then W, then H, then N
 *
 * NHWC enables better Tensor Core utilization because channels are contiguous,
 * matching the expected layout for implicit GEMM convolution algorithms.
 */
template<typename T>
__global__ void nchw_to_nhwc_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    int64_t batch,
    int64_t channels,
    int64_t height,
    int64_t width
) {
    const int64_t hw = height * width;
    const int64_t chw = channels * hw;
    const int64_t hwc = hw * channels;

    // Grid-stride loop for full coverage
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    const int64_t total = batch * chw;

    for (; idx < total; idx += stride) {
        // Decode NCHW index
        int64_t n = idx / chw;
        int64_t rem = idx % chw;
        int64_t c = rem / hw;
        rem = rem % hw;
        int64_t h = rem / width;
        int64_t w = rem % width;

        // Compute NHWC output index
        int64_t out_idx = n * hwc + h * width * channels + w * channels + c;
        output[out_idx] = input[idx];
    }
}

/**
 * @brief Convert NHWC to NCHW format kernel
 *
 * Input layout:  [N][H][W][C] - contiguous in C, then W, then H, then N
 * Output layout: [N][C][H][W] - contiguous in W, then H, then C, then N
 */
template<typename T>
__global__ void nhwc_to_nchw_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    int64_t batch,
    int64_t channels,
    int64_t height,
    int64_t width
) {
    const int64_t hw = height * width;
    const int64_t chw = channels * hw;
    const int64_t hwc = hw * channels;

    // Grid-stride loop for full coverage
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    const int64_t total = batch * hwc;

    for (; idx < total; idx += stride) {
        // Decode NHWC index
        int64_t n = idx / hwc;
        int64_t rem = idx % hwc;
        int64_t h = rem / (width * channels);
        rem = rem % (width * channels);
        int64_t w = rem / channels;
        int64_t c = rem % channels;

        // Compute NCHW output index
        int64_t out_idx = n * chw + c * hw + h * width + w;
        output[out_idx] = input[idx];
    }
}

/**
 * @brief Convert filter from NCHW to NHWC format kernel
 *
 * Filter layout change: [K][C][kH][kW] -> [K][kH][kW][C]
 */
template<typename T>
__global__ void filter_nchw_to_nhwc_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    int64_t out_channels,
    int64_t in_channels,
    int64_t kernel_h,
    int64_t kernel_w
) {
    const int64_t khw = kernel_h * kernel_w;
    const int64_t ckhw = in_channels * khw;
    const int64_t khwc = khw * in_channels;

    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    const int64_t total = out_channels * ckhw;

    for (; idx < total; idx += stride) {
        // Decode NCHW filter index [K][C][kH][kW]
        int64_t k = idx / ckhw;
        int64_t rem = idx % ckhw;
        int64_t c = rem / khw;
        rem = rem % khw;
        int64_t kh = rem / kernel_w;
        int64_t kw = rem % kernel_w;

        // Compute NHWC output index [K][kH][kW][C]
        int64_t out_idx = k * khwc + kh * kernel_w * in_channels + kw * in_channels + c;
        output[out_idx] = input[idx];
    }
}

/**
 * @brief Host function to convert NCHW tensor to NHWC
 */
auto nchw_to_nhwc(const Tensor& input, cudaStream_t stream) -> Tensor {
    auto shape = input.shape();
    if (shape.size() != 4) {
        throw std::runtime_error("nchw_to_nhwc: expected 4D tensor");
    }

    int64_t batch = shape[0];
    int64_t channels = shape[1];
    int64_t height = shape[2];
    int64_t width = shape[3];

    // Output has same logical shape but different memory layout (NHWC)
    // We store as [N, H, W, C] contiguous
    std::vector<int64_t> out_shape = {batch, height, width, channels};
    Tensor output(out_shape, input.dtype(), input.device());

    const int64_t total = batch * channels * height * width;

    if (input.dtype() == DType::Float32) {
        auto [grid_size, block_size] = optimal_launch_config(nchw_to_nhwc_kernel<float>, total);
        nchw_to_nhwc_kernel<float><<<grid_size, block_size, 0, stream>>>(
            input.data<float>(), output.data<float>(),
            batch, channels, height, width
        );
    } else if (input.dtype() == DType::Float16) {
        auto [grid_size, block_size] = optimal_launch_config(nchw_to_nhwc_kernel<Float16>, total);
        nchw_to_nhwc_kernel<Float16><<<grid_size, block_size, 0, stream>>>(
            input.data<Float16>(), output.data<Float16>(),
            batch, channels, height, width
        );
    } else if (input.dtype() == DType::Float64) {
        auto [grid_size, block_size] = optimal_launch_config(nchw_to_nhwc_kernel<double>, total);
        nchw_to_nhwc_kernel<double><<<grid_size, block_size, 0, stream>>>(
            input.data<double>(), output.data<double>(),
            batch, channels, height, width
        );
    } else if (input.dtype() == DType::BFloat16) {
        auto [grid_size, block_size] = optimal_launch_config(nchw_to_nhwc_kernel<BFloat16>, total);
        nchw_to_nhwc_kernel<BFloat16><<<grid_size, block_size, 0, stream>>>(
            input.data<BFloat16>(), output.data<BFloat16>(),
            batch, channels, height, width
        );
    } else {
        throw std::runtime_error("nchw_to_nhwc: unsupported dtype");
    }
    CUDA_CHECK(cudaGetLastError());

    return output;
}

/**
 * @brief Host function to convert NHWC tensor to NCHW
 */
auto nhwc_to_nchw(const Tensor& input, int64_t channels, cudaStream_t stream) -> Tensor {
    auto shape = input.shape();
    if (shape.size() != 4) {
        throw std::runtime_error("nhwc_to_nchw: expected 4D tensor");
    }

    int64_t batch = shape[0];
    int64_t height = shape[1];
    int64_t width = shape[2];
    // channels passed as parameter to verify

    // Output in NCHW format
    std::vector<int64_t> out_shape = {batch, channels, height, width};
    Tensor output(out_shape, input.dtype(), input.device());

    const int64_t total = batch * channels * height * width;

    if (input.dtype() == DType::Float32) {
        auto [grid_size, block_size] = optimal_launch_config(nhwc_to_nchw_kernel<float>, total);
        nhwc_to_nchw_kernel<float><<<grid_size, block_size, 0, stream>>>(
            input.data<float>(), output.data<float>(),
            batch, channels, height, width
        );
    } else if (input.dtype() == DType::Float16) {
        auto [grid_size, block_size] = optimal_launch_config(nhwc_to_nchw_kernel<Float16>, total);
        nhwc_to_nchw_kernel<Float16><<<grid_size, block_size, 0, stream>>>(
            input.data<Float16>(), output.data<Float16>(),
            batch, channels, height, width
        );
    } else if (input.dtype() == DType::Float64) {
        auto [grid_size, block_size] = optimal_launch_config(nhwc_to_nchw_kernel<double>, total);
        nhwc_to_nchw_kernel<double><<<grid_size, block_size, 0, stream>>>(
            input.data<double>(), output.data<double>(),
            batch, channels, height, width
        );
    } else if (input.dtype() == DType::BFloat16) {
        auto [grid_size, block_size] = optimal_launch_config(nhwc_to_nchw_kernel<BFloat16>, total);
        nhwc_to_nchw_kernel<BFloat16><<<grid_size, block_size, 0, stream>>>(
            input.data<BFloat16>(), output.data<BFloat16>(),
            batch, channels, height, width
        );
    } else {
        throw std::runtime_error("nhwc_to_nchw: unsupported dtype");
    }
    CUDA_CHECK(cudaGetLastError());

    return output;
}

/**
 * @brief Convert filter from NCHW to NHWC format
 */
auto filter_nchw_to_nhwc(const Tensor& weight, cudaStream_t stream) -> Tensor {
    auto shape = weight.shape();
    if (shape.size() != 4) {
        throw std::runtime_error("filter_nchw_to_nhwc: expected 4D tensor");
    }

    int64_t out_channels = shape[0];
    int64_t in_channels = shape[1];
    int64_t kernel_h = shape[2];
    int64_t kernel_w = shape[3];

    // Output as [K, kH, kW, C]
    std::vector<int64_t> out_shape = {out_channels, kernel_h, kernel_w, in_channels};
    Tensor output(out_shape, weight.dtype(), weight.device());

    const int64_t total = out_channels * in_channels * kernel_h * kernel_w;

    if (weight.dtype() == DType::Float32) {
        auto [grid_size, block_size] = optimal_launch_config(filter_nchw_to_nhwc_kernel<float>, total);
        filter_nchw_to_nhwc_kernel<float><<<grid_size, block_size, 0, stream>>>(
            weight.data<float>(), output.data<float>(),
            out_channels, in_channels, kernel_h, kernel_w
        );
    } else if (weight.dtype() == DType::Float16) {
        auto [grid_size, block_size] = optimal_launch_config(filter_nchw_to_nhwc_kernel<Float16>, total);
        filter_nchw_to_nhwc_kernel<Float16><<<grid_size, block_size, 0, stream>>>(
            weight.data<Float16>(), output.data<Float16>(),
            out_channels, in_channels, kernel_h, kernel_w
        );
    } else if (weight.dtype() == DType::Float64) {
        auto [grid_size, block_size] = optimal_launch_config(filter_nchw_to_nhwc_kernel<double>, total);
        filter_nchw_to_nhwc_kernel<double><<<grid_size, block_size, 0, stream>>>(
            weight.data<double>(), output.data<double>(),
            out_channels, in_channels, kernel_h, kernel_w
        );
    } else if (weight.dtype() == DType::BFloat16) {
        auto [grid_size, block_size] = optimal_launch_config(filter_nchw_to_nhwc_kernel<BFloat16>, total);
        filter_nchw_to_nhwc_kernel<BFloat16><<<grid_size, block_size, 0, stream>>>(
            weight.data<BFloat16>(), output.data<BFloat16>(),
            out_channels, in_channels, kernel_h, kernel_w
        );
    } else {
        throw std::runtime_error("filter_nchw_to_nhwc: unsupported dtype");
    }
    CUDA_CHECK(cudaGetLastError());

    return output;
}

/**
 * @brief Convert tensor to specified memory format (persistent format support)
 *
 * This is the key function for PyTorch-compatible memory format support.
 * Unlike nchw_to_nhwc/nhwc_to_nchw which change the logical shape, this function:
 * - Keeps the logical shape unchanged [N, C, H, W]
 * - Reorders data to match the target format's physical layout
 * - Sets strides to reflect the new memory layout
 *
 * For ChannelsLast: logical [N,C,H,W] with strides [H*W*C, 1, W*C, C]
 * For Contiguous:   logical [N,C,H,W] with strides [C*H*W, H*W, W, 1]
 */
auto to_memory_format_kernel(const Tensor& input, MemoryFormat format, void* stream_ptr) -> Tensor {
    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);

    // Get input shape
    auto shape = input.shape();

    // Only 4D tensors support ChannelsLast
    if (shape.size() != 4) {
        if (format == MemoryFormat::ChannelsLast) {
            return input;  // No-op for non-4D
        }
        // For Contiguous, just return a contiguous copy
        return input.contiguous();
    }

    int64_t N = shape[0];
    int64_t C = shape[1];
    int64_t H = shape[2];
    int64_t W = shape[3];

    // Create output tensor with target format's strides
    // We need to manually set the strides after creation
    Tensor output = Tensor::empty_uninitialized(
        std::vector<int64_t>{N, C, H, W},
        input.dtype(),
        input.device()
    );

    // Calculate target strides based on format
    std::vector<int64_t> target_strides;
    if (format == MemoryFormat::ChannelsLast) {
        // NHWC physical layout: strides = [H*W*C, 1, W*C, C]
        target_strides = {H * W * C, 1, W * C, C};
    } else {
        // NCHW (Contiguous): strides = [C*H*W, H*W, W, 1]
        target_strides = {C * H * W, H * W, W, 1};
    }

    // Set the output tensor's strides
    output.mutable_strides() = target_strides;

    // Determine current format from input strides
    auto input_strides = input.strides();
    // Require ALL four strides to describe a fully PACKED channels-last layout
    // {H*W*C, 1, W*C, C}. Checking only strides[1]==1 && strides[3]==C misclassifies
    // a spatially-sliced channels-last view (which keeps those two but has gaps in
    // strides[0]/strides[2]) as packed NHWC, then the fast-path memcpy / stride-
    // ignoring kernel copies the wrong bytes.
    bool input_is_nhwc = (input_strides.size() == 4 &&
                          input_strides[0] == static_cast<int64_t>(H) * W * C &&
                          input_strides[1] == 1 &&
                          input_strides[2] == static_cast<int64_t>(W) * C &&
                          input_strides[3] == C);
    bool input_is_nchw = !input_is_nhwc && input.is_contiguous();

    // Early return if already in target format — just copy data
    if ((format == MemoryFormat::ChannelsLast && input_is_nhwc) ||
        (format != MemoryFormat::ChannelsLast && input_is_nchw)) {
        const size_t size_bytes = static_cast<size_t>(N * C * H * W) * dtype_size(input.dtype());
        CUDA_CHECK(cudaMemcpyAsync(output.data_ptr(), input.data_ptr(),
                                    size_bytes, cudaMemcpyDeviceToDevice, stream));
        return output;
    }

    // The packed-layout conversion kernels below decode a linear index against
    // the *implied* NCHW/NHWC packing and read input[idx] directly, ignoring
    // strides. They are only correct when the source genuinely has that packed
    // layout. If the input is neither NHWC-strided nor fully contiguous (e.g. a
    // permute()/transpose view), both flags above are false and feeding such a
    // view to nhwc_to_nchw_kernel emits garbage (stride-from-shape bug). Coerce
    // the input to a packed layout first.
    if (!input_is_nhwc && !input_is_nchw) {
        if (format != MemoryFormat::ChannelsLast) {
            // Target Contiguous (NCHW): a contiguous materialization of the
            // logical [N,C,H,W] tensor already *is* the desired output.
            return input.contiguous();
        }
        // Target ChannelsLast: materialize a contiguous NCHW copy so the
        // NCHW -> NHWC kernel reads a valid packed source, then fall through.
        Tensor contig = input.contiguous();
        const int64_t total = N * C * H * W;
        if (contig.dtype() == DType::Float32) {
            auto [grid_size, block_size] = optimal_launch_config(nchw_to_nhwc_kernel<float>, total);
            nchw_to_nhwc_kernel<float><<<grid_size, block_size, 0, stream>>>(
                contig.data<float>(), output.data<float>(), N, C, H, W);
        } else if (contig.dtype() == DType::Float16) {
            auto [grid_size, block_size] = optimal_launch_config(nchw_to_nhwc_kernel<Float16>, total);
            nchw_to_nhwc_kernel<Float16><<<grid_size, block_size, 0, stream>>>(
                contig.data<Float16>(), output.data<Float16>(), N, C, H, W);
        } else if (contig.dtype() == DType::Float64) {
            auto [grid_size, block_size] = optimal_launch_config(nchw_to_nhwc_kernel<double>, total);
            nchw_to_nhwc_kernel<double><<<grid_size, block_size, 0, stream>>>(
                contig.data<double>(), output.data<double>(), N, C, H, W);
        } else if (contig.dtype() == DType::BFloat16) {
            auto [grid_size, block_size] = optimal_launch_config(nchw_to_nhwc_kernel<BFloat16>, total);
            nchw_to_nhwc_kernel<BFloat16><<<grid_size, block_size, 0, stream>>>(
                contig.data<BFloat16>(), output.data<BFloat16>(), N, C, H, W);
        } else {
            throw std::runtime_error("to_memory_format_kernel: unsupported dtype for ChannelsLast");
        }
        CUDA_CHECK(cudaGetLastError());
        return output;
    }

    const int64_t total = N * C * H * W;

    // Choose conversion direction based on actual input layout
    if (format == MemoryFormat::ChannelsLast) {
        // NCHW -> NHWC (but keeping logical shape [N,C,H,W])
        if (input.dtype() == DType::Float32) {
            auto [grid_size, block_size] = optimal_launch_config(nchw_to_nhwc_kernel<float>, total);
            nchw_to_nhwc_kernel<float><<<grid_size, block_size, 0, stream>>>(
                input.data<float>(), output.data<float>(),
                N, C, H, W
            );
        } else if (input.dtype() == DType::Float16) {
            auto [grid_size, block_size] = optimal_launch_config(nchw_to_nhwc_kernel<Float16>, total);
            nchw_to_nhwc_kernel<Float16><<<grid_size, block_size, 0, stream>>>(
                input.data<Float16>(), output.data<Float16>(),
                N, C, H, W
            );
        } else if (input.dtype() == DType::Float64) {
            auto [grid_size, block_size] = optimal_launch_config(nchw_to_nhwc_kernel<double>, total);
            nchw_to_nhwc_kernel<double><<<grid_size, block_size, 0, stream>>>(
                input.data<double>(), output.data<double>(),
                N, C, H, W
            );
        } else if (input.dtype() == DType::BFloat16) {
            auto [grid_size, block_size] = optimal_launch_config(nchw_to_nhwc_kernel<BFloat16>, total);
            nchw_to_nhwc_kernel<BFloat16><<<grid_size, block_size, 0, stream>>>(
                input.data<BFloat16>(), output.data<BFloat16>(),
                N, C, H, W
            );
        } else {
            throw std::runtime_error("to_memory_format_kernel: unsupported dtype for ChannelsLast");
        }
    } else {
        // NHWC -> NCHW (but keeping logical shape [N,C,H,W])
        if (input.dtype() == DType::Float32) {
            auto [grid_size, block_size] = optimal_launch_config(nhwc_to_nchw_kernel<float>, total);
            nhwc_to_nchw_kernel<float><<<grid_size, block_size, 0, stream>>>(
                input.data<float>(), output.data<float>(),
                N, C, H, W
            );
        } else if (input.dtype() == DType::Float16) {
            auto [grid_size, block_size] = optimal_launch_config(nhwc_to_nchw_kernel<Float16>, total);
            nhwc_to_nchw_kernel<Float16><<<grid_size, block_size, 0, stream>>>(
                input.data<Float16>(), output.data<Float16>(),
                N, C, H, W
            );
        } else if (input.dtype() == DType::Float64) {
            auto [grid_size, block_size] = optimal_launch_config(nhwc_to_nchw_kernel<double>, total);
            nhwc_to_nchw_kernel<double><<<grid_size, block_size, 0, stream>>>(
                input.data<double>(), output.data<double>(),
                N, C, H, W
            );
        } else if (input.dtype() == DType::BFloat16) {
            auto [grid_size, block_size] = optimal_launch_config(nhwc_to_nchw_kernel<BFloat16>, total);
            nhwc_to_nchw_kernel<BFloat16><<<grid_size, block_size, 0, stream>>>(
                input.data<BFloat16>(), output.data<BFloat16>(),
                N, C, H, W
            );
        } else {
            throw std::runtime_error("to_memory_format_kernel: unsupported dtype for Contiguous");
        }
    }
    CUDA_CHECK(cudaGetLastError());

    return output;
}

// ============================================================================
// cuDNN Conv2d Forward Implementation
// ============================================================================

// Per-axis primary impl (audit E1). The scalar overload below delegates here.
auto cudnn_conv2d_forward(
    const Tensor& input,
    const Tensor& weight,
    const Tensor* bias,
    int64_t stride_h, int64_t stride_w,
    int64_t pad_h, int64_t pad_w,
    int64_t dil_h, int64_t dil_w,
    int64_t groups,
    cudaStream_t stream
) -> Tensor {
    // Extract dimensions
    auto input_shape = input.shape();
    auto weight_shape = weight.shape();

    int64_t batch = input_shape[0];
    int64_t in_channels = input_shape[1];
    int64_t height = input_shape[2];
    int64_t width = input_shape[3];

    int64_t out_channels = weight_shape[0];
    int64_t kernel_h = weight_shape[2];
    int64_t kernel_w = weight_shape[3];

    // Calculate output dimensions (per-axis)
    int64_t out_h = (height + 2 * pad_h - dil_h * (kernel_h - 1) - 1) / stride_h + 1;
    int64_t out_w = (width  + 2 * pad_w - dil_w * (kernel_w - 1) - 1) / stride_w + 1;

    // Create output tensor
    std::vector<int64_t> output_shape = {batch, out_channels, out_h, out_w};
    Tensor output(output_shape, input.dtype(), input.device());

    // cuDNN descriptors below are built with packed NCHW/NCHW-filter strides
    // derived from shape, so a non-contiguous (channels-last / transposed /
    // sliced) input or weight would be read with the wrong strides. Materialize
    // contiguous copies and read the data pointers from those.
    const Tensor& input_c = input.is_contiguous() ? input : input.contiguous();
    const Tensor& weight_c = weight.is_contiguous() ? weight : weight.contiguous();

    // Make the tensor's device current so the (device-keyed) cuDNN handle and
    // workspace are fetched/allocated on the GPU the op runs on. Restored on exit.
    CudaDeviceGuard dev_guard(input.device().index);
    // Use singleton cuDNN handle (much faster than creating new one each time)
    cudnnHandle_t handle = CuDNNHandle::get();
    CuDNNHandle::set_stream(stream);

    // Convert dtype
    cudnnDataType_t cudnn_dtype;
    switch (input.dtype()) {
        case DType::Float32: cudnn_dtype = CUDNN_DATA_FLOAT; break;
        case DType::Float64: cudnn_dtype = CUDNN_DATA_DOUBLE; break;
        case DType::Float16: cudnn_dtype = CUDNN_DATA_HALF; break;
        case DType::BFloat16: cudnn_dtype = CUDNN_DATA_BFLOAT16; break;
        default:
            throw std::runtime_error("cuDNN Conv2d: unsupported dtype");
    }

    // Create descriptors
    TensorDescriptor input_desc, output_desc;
    FilterDescriptor filter_desc;
    ConvolutionDescriptor conv_desc;

    input_desc.set(cudnn_dtype, batch, in_channels, height, width);
    output_desc.set(cudnn_dtype, batch, out_channels, out_h, out_w);
    filter_desc.set(cudnn_dtype, out_channels, in_channels / groups, kernel_h, kernel_w);
    conv_desc.set(pad_h, pad_w, stride_h, stride_w, dil_h, dil_w, cudnn_dtype);

    // Enable TF32 tensor cores for FP32 convolution when explicitly opted in
    // via allow_tf32() (TENZOR_ENABLE_TF32=1 / set_allow_tf32(true)). PyTorch
    // defaults torch.backends.cudnn.allow_tf32 to True; tenzor deliberately
    // defaults allow_tf32() to False (F-108) for CPU<->CUDA numerical parity,
    // so out of the box this leaves FP32 conv at DEFAULT_MATH (no tensor
    // cores, ~10-30% slower than PyTorch's default, worst on Winograd-friendly
    // k3s1 shapes) in exchange for exact FP32 matching CPU. Gated on the same
    // flag as the Winograd exclusion in the algo search below. FindEx below
    // then times the TF32 algorithms when the flag is on.
    if (cudnn_dtype == CUDNN_DATA_FLOAT && ::tenzor::cuda::matmul::allow_tf32()) {
        CUDNN_CHECK(cudnnSetConvolutionMathType(
            conv_desc.get(), CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION));
    }

    if (groups > 1) {
        conv_desc.set_group_count(groups);
    }

    // Create cache key for algorithm lookup (NCHW format) — per-axis (E1).
    Conv2dCacheKey cache_key{
        batch, in_channels, height, width,
        out_channels, kernel_h, kernel_w,
        stride_h, stride_w, pad_h, pad_w, dil_h, dil_w, groups,
        cudnn_dtype, TensorFormat::NCHW,
        /*prefer_precise_f32=*/(cudnn_dtype == CUDNN_DATA_FLOAT) &&
                                !::tenzor::cuda::matmul::allow_tf32()
    };

    cudnnConvolutionFwdAlgo_t algo;
    size_t workspace_size;

    // Try to get cached algorithm
    CachedFwdAlgo cached;
    if (Conv2dAlgoCache::instance().get_fwd(cache_key, cached)) {
        // Cache hit - use cached algorithm
        algo = cached.algo;
        workspace_size = cached.workspace_size;
    } else {
        // Cache miss - use cudnnFindConvolutionForwardAlgorithmEx for true auto-tuning
        // This actually runs the algorithms and measures performance (like PyTorch)
        constexpr int kMaxAlgos = 8;

        // Use dynamic workspace size based on available GPU memory
        // This ensures optimal algorithm selection even for large convolutions
        const size_t kMaxWorkspaceSize = CuDNNWorkspace::max_workspace_size();

        // Use persistent workspace for algorithm search (avoids malloc/free overhead)
        void* search_workspace = CuDNNWorkspace::get(kMaxWorkspaceSize);

        int returned_algo_count = 0;
        cudnnConvolutionFwdAlgoPerf_t perf_results[kMaxAlgos];

        // Actually run and time each algorithm
        // Use data_ptr() for correct dtype handling (cuDNN uses void* internally)
        cudnnStatus_t find_status = cudnnFindConvolutionForwardAlgorithmEx(
            handle,
            input_desc.get(),
            input_c.data_ptr(),
            filter_desc.get(),
            weight_c.data_ptr(),
            conv_desc.get(),
            output_desc.get(),
            output.data_ptr(),
            kMaxAlgos,
            &returned_algo_count,
            perf_results,
            search_workspace,
            kMaxWorkspaceSize
        );

        if (find_status != CUDNN_STATUS_SUCCESS || returned_algo_count == 0) {
            // Fallback to heuristic if FindEx fails
            cudnnConvolutionFwdAlgoPerf_t heuristic_result;
            int heuristic_count = 0;
            CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm_v7(
                handle, input_desc.get(), filter_desc.get(), conv_desc.get(),
                output_desc.get(), 1, &heuristic_count, &heuristic_result
            ));
            if (heuristic_count <= 0) { throw std::runtime_error("cuDNN Conv2d forward: no convolution algorithm available for this descriptor"); }
            algo = heuristic_result.algo;
            CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(
                handle, input_desc.get(), filter_desc.get(), conv_desc.get(),
                output_desc.get(), algo, &workspace_size
            ));
        } else {
            // Find the fastest successful algorithm
            // For half-precision types, prefer GEMM-based algorithms which have
            // better FP16/BF16 support and use tensor cores effectively
            algo = perf_results[0].algo;
            workspace_size = perf_results[0].memory;
            float best_time = std::numeric_limits<float>::max();

            bool is_half_precision = (cudnn_dtype == CUDNN_DATA_HALF ||
                                       cudnn_dtype == CUDNN_DATA_BFLOAT16);
            // When the user has explicitly disabled TF32 (and the input is
            // Float32) they're asking for the most accurate math available.
            // Winograd-based convolutions transform inputs/filters into a
            // different domain before doing the multiply-add chain — that
            // accumulator order differs from the implicit-GEMM ("im2col +
            // sgemm") order used by CPU oneDNN, and the resulting per-
            // element abs diff in the 1e-5 range exceeds tight Float32
            // parity tolerances. Skip the Winograd variants in that mode
            // so cuDNN falls back to an implicit-GEMM-family algorithm
            // whose accumulator order matches what every other backend
            // does.
            bool prefer_precise_f32 = (cudnn_dtype == CUDNN_DATA_FLOAT) &&
                                       !::tenzor::cuda::matmul::allow_tf32();

            for (int i = 0; i < returned_algo_count; ++i) {
                if (perf_results[i].status != CUDNN_STATUS_SUCCESS ||
                    perf_results[i].memory > kMaxWorkspaceSize) {
                    continue;
                }

                // For half precision, skip FFT algorithms which don't support FP16/BF16
                if (is_half_precision) {
                    cudnnConvolutionFwdAlgo_t candidate = perf_results[i].algo;
                    if (candidate == CUDNN_CONVOLUTION_FWD_ALGO_FFT ||
                        candidate == CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING) {
                        continue;
                    }
                }

                if (prefer_precise_f32) {
                    cudnnConvolutionFwdAlgo_t candidate = perf_results[i].algo;
                    if (candidate == CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD ||
                        candidate == CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED) {
                        continue;
                    }
                }

                if (perf_results[i].time < best_time) {
                    best_time = perf_results[i].time;
                    algo = perf_results[i].algo;
                    workspace_size = perf_results[i].memory;
                }
            }
            // If we eliminated everything (only Winograd was viable for
            // this shape), fall back to IMPLICIT_PRECOMP_GEMM which is
            // available for any Conv2d shape and has the canonical
            // accumulator order.
            if (prefer_precise_f32 && best_time == std::numeric_limits<float>::max()) {
                algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
                CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(
                    handle, input_desc.get(), filter_desc.get(), conv_desc.get(),
                    output_desc.get(), algo, &workspace_size));
            }
        }

        // Cache the result
        Conv2dAlgoCache::instance().set_fwd(cache_key, {algo, workspace_size});
    }

    // Use persistent workspace buffer (avoids malloc/free per call)
    void* workspace = CuDNNWorkspace::get(workspace_size);

    // Set alpha and beta for the operation: output = alpha * conv(input) + beta * output
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Perform convolution
    if (input.dtype() == DType::Float32) {
        CUDNN_CHECK(cudnnConvolutionForward(
            handle,
            &alpha,
            input_desc.get(),
            input_c.data<float>(),
            filter_desc.get(),
            weight_c.data<float>(),
            conv_desc.get(),
            algo,
            workspace,
            workspace_size,
            &beta,
            output_desc.get(),
            output.data<float>()
        ));
    } else if (input.dtype() == DType::Float64) {
        const double alpha_d = 1.0;
        const double beta_d = 0.0;
        CUDNN_CHECK(cudnnConvolutionForward(
            handle,
            &alpha_d,
            input_desc.get(),
            input_c.data<double>(),
            filter_desc.get(),
            weight_c.data<double>(),
            conv_desc.get(),
            algo,
            workspace,
            workspace_size,
            &beta_d,
            output_desc.get(),
            output.data<double>()
        ));
    } else if (input.dtype() == DType::Float16) {
        // Try the selected algorithm; if it fails, fall back to IMPLICIT_PRECOMP_GEMM
        // which is the most reliable algorithm for half precision types
        cudnnStatus_t status = cudnnConvolutionForward(
            handle,
            &alpha,
            input_desc.get(),
            input_c.data<Float16>(),
            filter_desc.get(),
            weight_c.data<Float16>(),
            conv_desc.get(),
            algo,
            workspace,
            workspace_size,
            &beta,
            output_desc.get(),
            output.data<Float16>()
        );
        if (status != CUDNN_STATUS_SUCCESS) {
            // Fallback to IMPLICIT_PRECOMP_GEMM for better FP16 compatibility
            algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
            CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(
                handle, input_desc.get(), filter_desc.get(), conv_desc.get(),
                output_desc.get(), algo, &workspace_size
            ));
            workspace = CuDNNWorkspace::get(workspace_size);
            CUDNN_CHECK(cudnnConvolutionForward(
                handle,
                &alpha,
                input_desc.get(),
                input_c.data<Float16>(),
                filter_desc.get(),
                weight_c.data<Float16>(),
                conv_desc.get(),
                algo,
                workspace,
                workspace_size,
                &beta,
                output_desc.get(),
                output.data<Float16>()
            ));
            // Update the cache with the working algorithm
            Conv2dAlgoCache::instance().set_fwd(cache_key, {algo, workspace_size});
        }
    } else if (input.dtype() == DType::BFloat16) {
        // BFloat16 uses float alpha/beta like Float16
        cudnnStatus_t status = cudnnConvolutionForward(
            handle,
            &alpha,
            input_desc.get(),
            input_c.data<BFloat16>(),
            filter_desc.get(),
            weight_c.data<BFloat16>(),
            conv_desc.get(),
            algo,
            workspace,
            workspace_size,
            &beta,
            output_desc.get(),
            output.data<BFloat16>()
        );
        if (status != CUDNN_STATUS_SUCCESS) {
            // Fallback to IMPLICIT_PRECOMP_GEMM for better BF16 compatibility
            algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
            CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(
                handle, input_desc.get(), filter_desc.get(), conv_desc.get(),
                output_desc.get(), algo, &workspace_size
            ));
            workspace = CuDNNWorkspace::get(workspace_size);
            CUDNN_CHECK(cudnnConvolutionForward(
                handle,
                &alpha,
                input_desc.get(),
                input_c.data<BFloat16>(),
                filter_desc.get(),
                weight_c.data<BFloat16>(),
                conv_desc.get(),
                algo,
                workspace,
                workspace_size,
                &beta,
                output_desc.get(),
                output.data<BFloat16>()
            ));
            // Update the cache with the working algorithm
            Conv2dAlgoCache::instance().set_fwd(cache_key, {algo, workspace_size});
        }
    }

    // Add bias if present
    if (bias != nullptr) {
        TensorDescriptor bias_desc;
        bias_desc.set(cudnn_dtype, 1, out_channels, 1, 1);

        const float alpha_bias = 1.0f;
        const float beta_bias = 1.0f;

        if (input.dtype() == DType::Float32) {
            CUDNN_CHECK(cudnnAddTensor(
                handle,
                &alpha_bias,
                bias_desc.get(),
                bias->data<float>(),
                &beta_bias,
                output_desc.get(),
                output.data<float>()
            ));
        } else if (input.dtype() == DType::Float64) {
            const double alpha_bias_d = 1.0;
            const double beta_bias_d = 1.0;
            CUDNN_CHECK(cudnnAddTensor(
                handle,
                &alpha_bias_d,
                bias_desc.get(),
                bias->data<double>(),
                &beta_bias_d,
                output_desc.get(),
                output.data<double>()
            ));
        } else if (input.dtype() == DType::Float16) {
            CUDNN_CHECK(cudnnAddTensor(
                handle,
                &alpha_bias,
                bias_desc.get(),
                bias->data<Float16>(),
                &beta_bias,
                output_desc.get(),
                output.data<Float16>()
            ));
        } else if (input.dtype() == DType::BFloat16) {
            CUDNN_CHECK(cudnnAddTensor(
                handle,
                &alpha_bias,
                bias_desc.get(),
                bias->data<BFloat16>(),
                &beta_bias,
                output_desc.get(),
                output.data<BFloat16>()
            ));
        }
    }

    // No cleanup needed - workspace is persistent

    // Saturate FP16 output: clamp any ±Inf to ±65504 (max finite Float16 value).
    // cuDNN with FP32 compute type can produce values exceeding Float16 range,
    // which IEEE 754 converts to Inf. Saturating conversion matches CPU behavior.
    if (input.dtype() == DType::Float16) {
        fp16_saturate(output.data<Float16>(), output.numel(), stream);
    }

    return output;
}

// Scalar-form back-compat overload — delegates to the per-axis impl (E1).
auto cudnn_conv2d_forward(
    const Tensor& input,
    const Tensor& weight,
    const Tensor* bias,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    int64_t groups,
    cudaStream_t stream
) -> Tensor {
    return cudnn_conv2d_forward(input, weight, bias,
                                 stride, stride,
                                 padding, padding,
                                 dilation, dilation,
                                 groups, stream);
}

// ============================================================================
// Fused Conv2d + Bias + Activation using cudnnConvolutionBiasActivationForward
// ============================================================================

auto cudnn_fused_conv2d_activation_forward(
    const Tensor& input,
    const Tensor& weight,
    const Tensor* bias,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    int64_t groups,
    cudnnActivationMode_t activation_mode,
    double activation_coeff,
    cudaStream_t stream
) -> Tensor {
    return cudnn_fused_conv2d_activation_forward(
        input, weight, bias,
        stride, stride,
        padding, padding,
        dilation, dilation,
        groups, activation_mode, activation_coeff, stream);
}

auto cudnn_fused_conv2d_activation_forward(
    const Tensor& input_in,
    const Tensor& weight_in,
    const Tensor* bias,
    int64_t stride_h, int64_t stride_w,
    int64_t pad_h, int64_t pad_w,
    int64_t dil_h, int64_t dil_w,
    int64_t groups,
    cudnnActivationMode_t activation_mode,
    double activation_coeff,
    cudaStream_t stream
) -> Tensor {
    // cuDNN reads input/weight via data_ptr() with descriptors built from shape,
    // assuming contiguous NCHW/OIHW; a non-contiguous (sliced/permuted) input or
    // weight would be misread. Materialize contiguous copies first.
    Tensor input = input_in.is_contiguous() ? input_in : input_in.contiguous();
    Tensor weight = weight_in.is_contiguous() ? weight_in : weight_in.contiguous();
    auto input_shape = input.shape();
    auto weight_shape = weight.shape();

    int64_t batch = input_shape[0];
    int64_t in_channels = input_shape[1];
    int64_t height = input_shape[2];
    int64_t width = input_shape[3];

    int64_t out_channels = weight_shape[0];
    int64_t kernel_h = weight_shape[2];
    int64_t kernel_w = weight_shape[3];

    int64_t out_h = (height + 2 * pad_h - dil_h * (kernel_h - 1) - 1) / stride_h + 1;
    int64_t out_w = (width + 2 * pad_w - dil_w * (kernel_w - 1) - 1) / stride_w + 1;

    std::vector<int64_t> output_shape = {batch, out_channels, out_h, out_w};
    Tensor output(output_shape, input.dtype(), input.device());

    CudaDeviceGuard dev_guard(input.device().index);
    cudnnHandle_t handle = CuDNNHandle::get();
    CuDNNHandle::set_stream(stream);

    cudnnDataType_t cudnn_dtype;
    switch (input.dtype()) {
        case DType::Float32: cudnn_dtype = CUDNN_DATA_FLOAT; break;
        case DType::Float64: cudnn_dtype = CUDNN_DATA_DOUBLE; break;
        case DType::Float16: cudnn_dtype = CUDNN_DATA_HALF; break;
        case DType::BFloat16: cudnn_dtype = CUDNN_DATA_BFLOAT16; break;
        default:
            throw std::runtime_error("cuDNN FusedConv2dActivation: unsupported dtype");
    }

    TensorDescriptor input_desc, output_desc;
    FilterDescriptor filter_desc;
    ConvolutionDescriptor conv_desc;
    ActivationDescriptor act_desc;

    input_desc.set(cudnn_dtype, batch, in_channels, height, width);
    output_desc.set(cudnn_dtype, batch, out_channels, out_h, out_w);
    filter_desc.set(cudnn_dtype, out_channels, in_channels / groups, kernel_h, kernel_w);
    conv_desc.set(pad_h, pad_w, stride_h, stride_w, dil_h, dil_w, cudnn_dtype);
    act_desc.set(activation_mode, activation_coeff);

    if (groups > 1) {
        conv_desc.set_group_count(groups);
    }

    // Create bias descriptor (1 x C x 1 x 1)
    TensorDescriptor bias_desc;
    bias_desc.set(cudnn_dtype, 1, out_channels, 1, 1);

    // Allocate a zero bias if none provided (cudnnConvolutionBiasActivationForward requires bias)
    Tensor zero_bias;
    Tensor bias_cont;
    const void* bias_ptr;
    if (bias != nullptr) {
        // The bias descriptor is packed [1,C,1,1]; a non-contiguous bias view
        // would be read with the wrong strides. Materialize a contiguous copy.
        bias_cont = bias->is_contiguous() ? *bias : bias->contiguous();
        bias_ptr = bias_cont.data_ptr();
    } else {
        zero_bias = Tensor({out_channels}, input.dtype(), input.device());
        cudaMemsetAsync(zero_bias.data_ptr(), 0, out_channels * dtype_size(input.dtype()), stream);
        bias_ptr = zero_bias.data_ptr();
    }

    // Find algorithm (using heuristic for simplicity since this is a fused path)
    cudnnConvolutionFwdAlgo_t algo;
    size_t workspace_size;

    cudnnConvolutionFwdAlgoPerf_t heuristic_result;
    int heuristic_count = 0;
    CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm_v7(
        handle, input_desc.get(), filter_desc.get(), conv_desc.get(),
        output_desc.get(), 1, &heuristic_count, &heuristic_result
    ));
            if (heuristic_count <= 0) { throw std::runtime_error("cuDNN Conv2d forward: no convolution algorithm available for this descriptor"); }
    algo = heuristic_result.algo;
    CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(
        handle, input_desc.get(), filter_desc.get(), conv_desc.get(),
        output_desc.get(), algo, &workspace_size
    ));

    void* workspace = CuDNNWorkspace::get(workspace_size);

    const float alpha1 = 1.0f;
    const float alpha2 = 0.0f;

    if (input.dtype() == DType::Float32 || input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16) {
        CUDNN_CHECK(cudnnConvolutionBiasActivationForward(
            handle,
            &alpha1,
            input_desc.get(), input.data_ptr(),
            filter_desc.get(), weight.data_ptr(),
            conv_desc.get(), algo,
            workspace, workspace_size,
            &alpha2,
            output_desc.get(), output.data_ptr(),
            bias_desc.get(), bias_ptr,
            act_desc.get(),
            output_desc.get(), output.data_ptr()
        ));
    } else if (input.dtype() == DType::Float64) {
        const double alpha1_d = 1.0;
        const double alpha2_d = 0.0;
        CUDNN_CHECK(cudnnConvolutionBiasActivationForward(
            handle,
            &alpha1_d,
            input_desc.get(), input.data_ptr(),
            filter_desc.get(), weight.data_ptr(),
            conv_desc.get(), algo,
            workspace, workspace_size,
            &alpha2_d,
            output_desc.get(), output.data_ptr(),
            bias_desc.get(), bias_ptr,
            act_desc.get(),
            output_desc.get(), output.data_ptr()
        ));
    }

    if (input.dtype() == DType::Float16) {
        fp16_saturate(output.data<Float16>(), output.numel(), stream);
    }

    return output;
}

auto cudnn_fused_conv2d_relu_forward(
    const Tensor& input,
    const Tensor& weight,
    const Tensor* bias,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    int64_t groups,
    cudaStream_t stream
) -> Tensor {
    return cudnn_fused_conv2d_activation_forward(
        input, weight, bias, stride, padding, dilation, groups,
        CUDNN_ACTIVATION_RELU, 0.0, stream);
}

auto cudnn_fused_conv2d_relu_forward(
    const Tensor& input,
    const Tensor& weight,
    const Tensor* bias,
    int64_t stride_h, int64_t stride_w,
    int64_t pad_h, int64_t pad_w,
    int64_t dil_h, int64_t dil_w,
    int64_t groups,
    cudaStream_t stream
) -> Tensor {
    return cudnn_fused_conv2d_activation_forward(
        input, weight, bias,
        stride_h, stride_w, pad_h, pad_w, dil_h, dil_w, groups,
        CUDNN_ACTIVATION_RELU, 0.0, stream);
}

auto cudnn_fused_conv2d_sigmoid_forward(
    const Tensor& input,
    const Tensor& weight,
    const Tensor* bias,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    int64_t groups,
    cudaStream_t stream
) -> Tensor {
    // cudnnConvolutionBiasActivationForward does not support SIGMOID on most
    // GPU/cuDNN combos.  Compose conv2d + sigmoid instead.
    Tensor result = cudnn_conv2d_forward(input, weight, bias, stride, padding,
                                          dilation, groups, stream);
    return sigmoid_kernel(result, stream);
}

auto cudnn_fused_conv2d_sigmoid_forward(
    const Tensor& input,
    const Tensor& weight,
    const Tensor* bias,
    int64_t stride_h, int64_t stride_w,
    int64_t pad_h, int64_t pad_w,
    int64_t dil_h, int64_t dil_w,
    int64_t groups,
    cudaStream_t stream
) -> Tensor {
    Tensor result = cudnn_conv2d_forward(input, weight, bias,
                                          stride_h, stride_w, pad_h, pad_w, dil_h, dil_w,
                                          groups, stream);
    return sigmoid_kernel(result, stream);
}

auto cudnn_fused_conv2d_tanh_forward(
    const Tensor& input,
    const Tensor& weight,
    const Tensor* bias,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    int64_t groups,
    cudaStream_t stream
) -> Tensor {
    // cudnnConvolutionBiasActivationForward does not support TANH on most
    // GPU/cuDNN combos.  Compose conv2d + tanh instead.
    Tensor result = cudnn_conv2d_forward(input, weight, bias, stride, padding,
                                          dilation, groups, stream);
    return tanh_kernel(result, stream);
}

auto cudnn_fused_conv2d_tanh_forward(
    const Tensor& input,
    const Tensor& weight,
    const Tensor* bias,
    int64_t stride_h, int64_t stride_w,
    int64_t pad_h, int64_t pad_w,
    int64_t dil_h, int64_t dil_w,
    int64_t groups,
    cudaStream_t stream
) -> Tensor {
    Tensor result = cudnn_conv2d_forward(input, weight, bias,
                                          stride_h, stride_w, pad_h, pad_w, dil_h, dil_w,
                                          groups, stream);
    return tanh_kernel(result, stream);
}

auto cudnn_fused_conv2d_swish_forward(
    const Tensor& input,
    const Tensor& weight,
    const Tensor* bias,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    int64_t groups,
    cudaStream_t stream
) -> Tensor {
    // cudnnConvolutionBiasActivationForward does not support SWISH on most
    // GPU/cuDNN combos.  Compose conv2d + swish instead.
    Tensor result = cudnn_conv2d_forward(input, weight, bias, stride, padding,
                                          dilation, groups, stream);
    return swish_kernel(result, stream);
}

auto cudnn_fused_conv2d_swish_forward(
    const Tensor& input,
    const Tensor& weight,
    const Tensor* bias,
    int64_t stride_h, int64_t stride_w,
    int64_t pad_h, int64_t pad_w,
    int64_t dil_h, int64_t dil_w,
    int64_t groups,
    cudaStream_t stream
) -> Tensor {
    Tensor result = cudnn_conv2d_forward(input, weight, bias,
                                          stride_h, stride_w, pad_h, pad_w, dil_h, dil_w,
                                          groups, stream);
    return swish_kernel(result, stream);
}

// ============================================================================
// cuDNN Conv2d Backward Implementation
// ============================================================================

// Per-axis primary impl (audit E1). Scalar delegate below.
auto cudnn_conv2d_backward(
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& weight,
    int64_t stride_h, int64_t stride_w,
    int64_t pad_h, int64_t pad_w,
    int64_t dil_h, int64_t dil_w,
    int64_t groups,
    bool compute_grad_input,
    bool compute_grad_weight,
    bool compute_grad_bias,
    cudaStream_t stream
) -> std::tuple<Tensor, Tensor, Tensor> {
    // Extract dimensions
    auto input_shape = input.shape();
    auto weight_shape = weight.shape();
    auto grad_shape = grad_output.shape();

    int64_t batch = input_shape[0];
    int64_t in_channels = input_shape[1];
    int64_t height = input_shape[2];
    int64_t width = input_shape[3];

    int64_t out_channels = weight_shape[0];
    int64_t kernel_h = weight_shape[2];
    int64_t kernel_w = weight_shape[3];

    int64_t out_h = grad_shape[2];
    int64_t out_w = grad_shape[3];

    // Initialize gradients
    Tensor grad_input({batch, in_channels, height, width}, input.dtype(), input.device());
    Tensor grad_weight({out_channels, in_channels / groups, kernel_h, kernel_w}, weight.dtype(), weight.device());
    Tensor grad_bias({out_channels}, weight.dtype(), weight.device());

    // cuDNN reads input/weight/grad_output with packed NCHW strides derived from
    // shape; a non-contiguous view would feed wrong strides. Materialize
    // contiguous copies for the data pointers.
    const Tensor& input_c = input.is_contiguous() ? input : input.contiguous();
    const Tensor& weight_c = weight.is_contiguous() ? weight : weight.contiguous();
    const Tensor& grad_output_c = grad_output.is_contiguous() ? grad_output : grad_output.contiguous();

    CudaDeviceGuard dev_guard(input.device().index);
    // Use singleton cuDNN handle
    cudnnHandle_t handle = CuDNNHandle::get();
    CuDNNHandle::set_stream(stream);

    // Convert dtype
    cudnnDataType_t cudnn_dtype;
    switch (input.dtype()) {
        case DType::Float32: cudnn_dtype = CUDNN_DATA_FLOAT; break;
        case DType::Float64: cudnn_dtype = CUDNN_DATA_DOUBLE; break;
        case DType::Float16: cudnn_dtype = CUDNN_DATA_HALF; break;
        case DType::BFloat16: cudnn_dtype = CUDNN_DATA_BFLOAT16; break;
        default:
            throw std::runtime_error("cuDNN Conv2d backward: unsupported dtype");
    }

    // Create descriptors
    TensorDescriptor input_desc, grad_output_desc;
    FilterDescriptor filter_desc;
    ConvolutionDescriptor conv_desc;

    input_desc.set(cudnn_dtype, batch, in_channels, height, width);
    grad_output_desc.set(cudnn_dtype, batch, out_channels, out_h, out_w);
    filter_desc.set(cudnn_dtype, out_channels, in_channels / groups, kernel_h, kernel_w);
    conv_desc.set(pad_h, pad_w, stride_h, stride_w, dil_h, dil_w, cudnn_dtype);

    if (groups > 1) {
        conv_desc.set_group_count(groups);
    }

    // Create cache key for algorithm lookup (NCHW format) — per-axis (E1).
    Conv2dCacheKey cache_key{
        batch, in_channels, height, width,
        out_channels, kernel_h, kernel_w,
        stride_h, stride_w, pad_h, pad_w, dil_h, dil_w, groups,
        cudnn_dtype, TensorFormat::NCHW,
        /*prefer_precise_f32=*/(cudnn_dtype == CUDNN_DATA_FLOAT) &&
                                !::tenzor::cuda::matmul::allow_tf32()
    };

    // Same precise-FP32 source of truth as the forward path and Conv3d backward:
    // an FP32 input with TF32 disabled wants Winograd excluded from the bwd-data
    // and bwd-filter algorithm selection so the backward accumulator order matches
    // the implicit-GEMM order the other backends use and stays inside the tight
    // Float32 parity tolerance.
    const bool prefer_precise_f32 =
        (cudnn_dtype == CUDNN_DATA_FLOAT) &&
        !::tenzor::cuda::matmul::allow_tf32();

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Compute gradient w.r.t. input
    if (compute_grad_input) {
        cudnnConvolutionBwdDataAlgo_t algo;
        size_t workspace_size;

        // Try to get cached algorithm
        CachedBwdDataAlgo cached;
        if (Conv2dAlgoCache::instance().get_bwd_data(cache_key, cached)) {
            algo = cached.algo;
            workspace_size = cached.workspace_size;
        } else {
            // Cache miss - query multiple algorithms and find the fastest
            constexpr int kMaxAlgos = 8;
            const size_t kMaxWorkspaceSize = CuDNNWorkspace::max_workspace_size();
            int returned_algo_count = 0;
            cudnnConvolutionBwdDataAlgoPerf_t perf_results[kMaxAlgos];

            CUDNN_CHECK(cudnnGetConvolutionBackwardDataAlgorithm_v7(
                handle,
                filter_desc.get(),
                grad_output_desc.get(),
                conv_desc.get(),
                input_desc.get(),
                kMaxAlgos,
                &returned_algo_count,
                perf_results
            ));

            if (returned_algo_count <= 0) { throw std::runtime_error("cuDNN Conv2d backward-data: no algorithm available for this descriptor"); }
            algo = perf_results[0].algo;
            workspace_size = 0;
            float best_time = std::numeric_limits<float>::max();

            for (int i = 0; i < returned_algo_count; ++i) {
                if (perf_results[i].status == CUDNN_STATUS_SUCCESS &&
                    perf_results[i].memory <= kMaxWorkspaceSize) {
                    if (prefer_precise_f32) {
                        cudnnConvolutionBwdDataAlgo_t candidate = perf_results[i].algo;
                        if (candidate == CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD ||
                            candidate == CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED) {
                            continue;
                        }
                    }
                    size_t ws_size = 0;
                    cudnnStatus_t ws_status = cudnnGetConvolutionBackwardDataWorkspaceSize(
                        handle,
                        filter_desc.get(),
                        grad_output_desc.get(),
                        conv_desc.get(),
                        input_desc.get(),
                        perf_results[i].algo,
                        &ws_size
                    );
                    if (ws_status == CUDNN_STATUS_SUCCESS && ws_size <= kMaxWorkspaceSize) {
                        if (perf_results[i].time < best_time) {
                            best_time = perf_results[i].time;
                            algo = perf_results[i].algo;
                            workspace_size = ws_size;
                        }
                    }
                }
            }

            if (best_time == std::numeric_limits<float>::max()) {
                // Nothing viable survived. For precise FP32, fall back to the
                // deterministic ALGO_1 (implicit-GEMM-style, non-Winograd) rather
                // than perf_results[0], which could be a Winograd variant.
                algo = prefer_precise_f32 ? CUDNN_CONVOLUTION_BWD_DATA_ALGO_1
                                          : perf_results[0].algo;
                CUDNN_CHECK(cudnnGetConvolutionBackwardDataWorkspaceSize(
                    handle, filter_desc.get(), grad_output_desc.get(),
                    conv_desc.get(), input_desc.get(), algo, &workspace_size
                ));
            }

            Conv2dAlgoCache::instance().set_bwd_data(cache_key, {algo, workspace_size});
        }

        void* workspace = CuDNNWorkspace::get(workspace_size);

        if (input.dtype() == DType::Float32) {
            CUDNN_CHECK(cudnnConvolutionBackwardData(
                handle,
                &alpha,
                filter_desc.get(),
                weight_c.data<float>(),
                grad_output_desc.get(),
                grad_output_c.data<float>(),
                conv_desc.get(),
                algo,
                workspace,
                workspace_size,
                &beta,
                input_desc.get(),
                grad_input.data<float>()
            ));
        } else if (input.dtype() == DType::Float64) {
            const double alpha_d = 1.0;
            const double beta_d = 0.0;
            CUDNN_CHECK(cudnnConvolutionBackwardData(
                handle,
                &alpha_d,
                filter_desc.get(),
                weight_c.data<double>(),
                grad_output_desc.get(),
                grad_output_c.data<double>(),
                conv_desc.get(),
                algo,
                workspace,
                workspace_size,
                &beta_d,
                input_desc.get(),
                grad_input.data<double>()
            ));
        } else if (input.dtype() == DType::Float16) {
            CUDNN_CHECK(cudnnConvolutionBackwardData(
                handle,
                &alpha,
                filter_desc.get(),
                weight_c.data<Float16>(),
                grad_output_desc.get(),
                grad_output_c.data<Float16>(),
                conv_desc.get(),
                algo,
                workspace,
                workspace_size,
                &beta,
                input_desc.get(),
                grad_input.data<Float16>()
            ));
            // Saturate FP16 output: clamp ±Inf to ±65504 to prevent NaN propagation
            fp16_saturate(grad_input.data<Float16>(), grad_input.numel(), stream);
        } else if (input.dtype() == DType::BFloat16) {
            CUDNN_CHECK(cudnnConvolutionBackwardData(
                handle,
                &alpha,
                filter_desc.get(),
                weight_c.data<BFloat16>(),
                grad_output_desc.get(),
                grad_output_c.data<BFloat16>(),
                conv_desc.get(),
                algo,
                workspace,
                workspace_size,
                &beta,
                input_desc.get(),
                grad_input.data<BFloat16>()
            ));
        }
    }

    // Compute gradient w.r.t. weight
    if (compute_grad_weight) {
        cudnnConvolutionBwdFilterAlgo_t algo;
        size_t workspace_size;

        // Try to get cached algorithm
        CachedBwdFilterAlgo cached;
        if (Conv2dAlgoCache::instance().get_bwd_filter(cache_key, cached)) {
            algo = cached.algo;
            workspace_size = cached.workspace_size;
        } else {
            // Cache miss - query multiple algorithms and find the fastest
            constexpr int kMaxAlgos = 8;
            const size_t kMaxWorkspaceSize = CuDNNWorkspace::max_workspace_size();
            int returned_algo_count = 0;
            cudnnConvolutionBwdFilterAlgoPerf_t perf_results[kMaxAlgos];

            CUDNN_CHECK(cudnnGetConvolutionBackwardFilterAlgorithm_v7(
                handle,
                input_desc.get(),
                grad_output_desc.get(),
                conv_desc.get(),
                filter_desc.get(),
                kMaxAlgos,
                &returned_algo_count,
                perf_results
            ));

            if (returned_algo_count <= 0) { throw std::runtime_error("cuDNN Conv2d backward-filter: no algorithm available for this descriptor"); }
            algo = perf_results[0].algo;
            workspace_size = 0;
            float best_time = std::numeric_limits<float>::max();

            for (int i = 0; i < returned_algo_count; ++i) {
                if (perf_results[i].status == CUDNN_STATUS_SUCCESS &&
                    perf_results[i].memory <= kMaxWorkspaceSize) {
                    if (prefer_precise_f32) {
                        cudnnConvolutionBwdFilterAlgo_t candidate = perf_results[i].algo;
                        if (candidate == CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD ||
                            candidate == CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED) {
                            continue;
                        }
                    }
                    size_t ws_size = 0;
                    cudnnStatus_t ws_status = cudnnGetConvolutionBackwardFilterWorkspaceSize(
                        handle,
                        input_desc.get(),
                        grad_output_desc.get(),
                        conv_desc.get(),
                        filter_desc.get(),
                        perf_results[i].algo,
                        &ws_size
                    );
                    if (ws_status == CUDNN_STATUS_SUCCESS && ws_size <= kMaxWorkspaceSize) {
                        if (perf_results[i].time < best_time) {
                            best_time = perf_results[i].time;
                            algo = perf_results[i].algo;
                            workspace_size = ws_size;
                        }
                    }
                }
            }

            if (best_time == std::numeric_limits<float>::max()) {
                // Nothing viable survived. For precise FP32, fall back to the
                // deterministic ALGO_1 (implicit-GEMM-style, non-Winograd) rather
                // than perf_results[0], which could be a Winograd variant.
                algo = prefer_precise_f32 ? CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1
                                          : perf_results[0].algo;
                CUDNN_CHECK(cudnnGetConvolutionBackwardFilterWorkspaceSize(
                    handle, input_desc.get(), grad_output_desc.get(),
                    conv_desc.get(), filter_desc.get(), algo, &workspace_size
                ));
            }

            Conv2dAlgoCache::instance().set_bwd_filter(cache_key, {algo, workspace_size});
        }

        void* workspace = CuDNNWorkspace::get(workspace_size);

        if (input.dtype() == DType::Float32) {
            CUDNN_CHECK(cudnnConvolutionBackwardFilter(
                handle,
                &alpha,
                input_desc.get(),
                input_c.data<float>(),
                grad_output_desc.get(),
                grad_output_c.data<float>(),
                conv_desc.get(),
                algo,
                workspace,
                workspace_size,
                &beta,
                filter_desc.get(),
                grad_weight.data<float>()
            ));
        } else if (input.dtype() == DType::Float64) {
            const double alpha_d = 1.0;
            const double beta_d = 0.0;
            CUDNN_CHECK(cudnnConvolutionBackwardFilter(
                handle,
                &alpha_d,
                input_desc.get(),
                input_c.data<double>(),
                grad_output_desc.get(),
                grad_output_c.data<double>(),
                conv_desc.get(),
                algo,
                workspace,
                workspace_size,
                &beta_d,
                filter_desc.get(),
                grad_weight.data<double>()
            ));
        } else if (input.dtype() == DType::Float16) {
            CUDNN_CHECK(cudnnConvolutionBackwardFilter(
                handle,
                &alpha,
                input_desc.get(),
                input_c.data<Float16>(),
                grad_output_desc.get(),
                grad_output_c.data<Float16>(),
                conv_desc.get(),
                algo,
                workspace,
                workspace_size,
                &beta,
                filter_desc.get(),
                grad_weight.data<Float16>()
            ));
            // Saturate FP16 output: clamp ±Inf to ±65504 to prevent NaN propagation
            fp16_saturate(grad_weight.data<Float16>(), grad_weight.numel(), stream);
        } else if (input.dtype() == DType::BFloat16) {
            CUDNN_CHECK(cudnnConvolutionBackwardFilter(
                handle,
                &alpha,
                input_desc.get(),
                input_c.data<BFloat16>(),
                grad_output_desc.get(),
                grad_output_c.data<BFloat16>(),
                conv_desc.get(),
                algo,
                workspace,
                workspace_size,
                &beta,
                filter_desc.get(),
                grad_weight.data<BFloat16>()
            ));
        }
    }

    // Compute gradient w.r.t. bias
    if (compute_grad_bias) {
        TensorDescriptor bias_desc;
        bias_desc.set(cudnn_dtype, 1, out_channels, 1, 1);

        if (input.dtype() == DType::Float32) {
            CUDNN_CHECK(cudnnConvolutionBackwardBias(
                handle,
                &alpha,
                grad_output_desc.get(),
                grad_output_c.data<float>(),
                &beta,
                bias_desc.get(),
                grad_bias.data<float>()
            ));
        } else if (input.dtype() == DType::Float64) {
            const double alpha_d = 1.0;
            const double beta_d = 0.0;
            CUDNN_CHECK(cudnnConvolutionBackwardBias(
                handle,
                &alpha_d,
                grad_output_desc.get(),
                grad_output_c.data<double>(),
                &beta_d,
                bias_desc.get(),
                grad_bias.data<double>()
            ));
        } else if (input.dtype() == DType::Float16) {
            CUDNN_CHECK(cudnnConvolutionBackwardBias(
                handle,
                &alpha,
                grad_output_desc.get(),
                grad_output_c.data<Float16>(),
                &beta,
                bias_desc.get(),
                grad_bias.data<Float16>()
            ));
        } else if (input.dtype() == DType::BFloat16) {
            CUDNN_CHECK(cudnnConvolutionBackwardBias(
                handle,
                &alpha,
                grad_output_desc.get(),
                grad_output_c.data<BFloat16>(),
                &beta,
                bias_desc.get(),
                grad_bias.data<BFloat16>()
            ));
        }
    }

    return std::make_tuple(grad_input, grad_weight, grad_bias);
}

// Scalar-form back-compat overload (E1).
auto cudnn_conv2d_backward(
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& weight,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    int64_t groups,
    bool compute_grad_input,
    bool compute_grad_weight,
    bool compute_grad_bias,
    cudaStream_t stream
) -> std::tuple<Tensor, Tensor, Tensor> {
    return cudnn_conv2d_backward(grad_output, input, weight,
                                  stride, stride,
                                  padding, padding,
                                  dilation, dilation,
                                  groups,
                                  compute_grad_input, compute_grad_weight, compute_grad_bias,
                                  stream);
}

// ============================================================================
// NHWC-Optimized Conv2d Forward Implementation
// ============================================================================

// ============================================================================
// Weight Conversion Cache for NHWC Conv2D
// ============================================================================
// Caches NCHW->NHWC converted weights to avoid conversion on every forward pass.
// Uses weak reference semantics - when the original NCHW weight is deallocated,
// the cache entry becomes invalid and will be replaced on next access.
namespace {

struct WeightCacheEntry {
    const void* original_data_ptr = nullptr;  // Pointer to original NCHW weight data
    std::vector<int64_t> shape;               // Shape of the weight
    uint64_t version = 0;                     // Version counter to detect ABA reuse
    Tensor nhwc_weight;                       // Cached NHWC-converted weight

    WeightCacheEntry() = default;
    WeightCacheEntry(const void* ptr, const std::vector<int64_t>& s, uint64_t ver, const Tensor& t)
        : original_data_ptr(ptr), shape(s), version(ver), nhwc_weight(t) {}
};

class NHWCWeightCache {
public:
    static NHWCWeightCache& instance() {
        static NHWCWeightCache cache;
        return cache;
    }

    // Try to get cached NHWC weight. Returns true if found and valid.
    bool get(const Tensor& weight, Tensor& nhwc_out) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = cache_.find(weight.data_ptr());
        if (it != cache_.end()) {
            // Verify shape AND version match to prevent ABA reuse after dealloc
            auto weight_shape = weight.shape();
            std::vector<int64_t> weight_shape_vec(weight_shape.begin(), weight_shape.end());
            if (it->second.shape == weight_shape_vec &&
                it->second.version == weight.version()) {
                nhwc_out = it->second.nhwc_weight;
                return true;
            }
        }
        return false;
    }

    // Store converted NHWC weight in cache
    void set(const Tensor& weight, const Tensor& nhwc_weight) {
        std::lock_guard<std::mutex> lock(mutex_);
        // Limit cache size to prevent unbounded growth
        if (cache_.size() >= kMaxCacheSize) {
            cache_.clear();  // Simple eviction: clear all when full
        }
        auto weight_shape = weight.shape();
        std::vector<int64_t> weight_shape_vec(weight_shape.begin(), weight_shape.end());
        cache_[weight.data_ptr()] = WeightCacheEntry(
            weight.data_ptr(),
            weight_shape_vec,
            weight.version(),
            nhwc_weight
        );
    }

private:
    NHWCWeightCache() = default;
    static constexpr size_t kMaxCacheSize = 128;
    std::unordered_map<const void*, WeightCacheEntry> cache_;
    std::mutex mutex_;
};

} // anonymous namespace

/**
 * @brief NHWC-optimized Conv2D forward using cuDNN
 *
 * This function provides significant performance improvements by using NHWC
 * tensor format internally, which enables:
 * - Better Tensor Core utilization on modern NVIDIA GPUs (Ampere, Ada, Blackwell)
 * - Improved memory coalescing for channel-wise operations
 * - Access to faster implicit GEMM algorithms in cuDNN
 *
 * The function handles NCHW<->NHWC conversion internally so the external API
 * remains unchanged. For very small convolutions where conversion overhead
 * exceeds the compute benefit, consider using the NCHW version.
 *
 * Weight tensors are cached after NHWC conversion to avoid repeated conversions.
 *
 * Typical speedup: 1.3-2.0x on RTX 30xx/40xx/50xx series GPUs
 */
auto cudnn_conv2d_forward_nhwc(
    const Tensor& input,
    const Tensor& weight,
    const Tensor* bias,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    int64_t groups,
    void* stream_ptr
) -> Tensor {
    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);

    // Extract dimensions - input has logical shape [N, C, H, W]
    auto input_shape = input.shape();
    auto weight_shape = weight.shape();

    int64_t batch = input_shape[0];
    int64_t in_channels = input_shape[1];
    int64_t height = input_shape[2];
    int64_t width = input_shape[3];

    int64_t out_channels = weight_shape[0];
    int64_t kernel_h = weight_shape[2];
    int64_t kernel_w = weight_shape[3];

    // Calculate output dimensions
    int64_t out_h = (height + 2 * padding - dilation * (kernel_h - 1) - 1) / stride + 1;
    int64_t out_w = (width + 2 * padding - dilation * (kernel_w - 1) - 1) / stride + 1;

    // Check if input is already in NHWC format (via strides)
    bool input_is_nhwc = (input.memory_format() == MemoryFormat::ChannelsLast);

    // If input is already NHWC, use it directly; otherwise convert
    Tensor input_nhwc;
    if (input_is_nhwc) {
        // Input already has NHWC physical layout - use directly
        input_nhwc = input;
    } else {
        // Convert NCHW to NHWC
        input_nhwc = nchw_to_nhwc(input, stream);
    }

    // Get or convert weight to NHWC format
    // Use cache to avoid repeated conversions for the same weight tensor
    Tensor weight_nhwc;
    bool weight_is_nhwc = (weight.memory_format() == MemoryFormat::ChannelsLast);
    if (weight_is_nhwc) {
        // Weight is already in NHWC format - use directly
        weight_nhwc = weight;
    } else if (NHWCWeightCache::instance().get(weight, weight_nhwc)) {
        // Cache hit - use cached NHWC weight (no conversion needed)
    } else {
        // Cache miss - convert and cache
        weight_nhwc = filter_nchw_to_nhwc(weight, stream);
        NHWCWeightCache::instance().set(weight, weight_nhwc);
    }

    // Create output tensor with NCHW logical shape but NHWC physical layout
    // First allocate with physical shape [N, H, W, C], then set logical shape and NHWC strides
    std::vector<int64_t> output_physical_shape = {batch, out_h, out_w, out_channels};
    Tensor output_nhwc(output_physical_shape, input.dtype(), input.device());
    // Immediately set to logical NCHW shape with NHWC strides
    output_nhwc.mutable_shape() = {batch, out_channels, out_h, out_w};
    output_nhwc.mutable_strides() = {out_h * out_w * out_channels, 1, out_w * out_channels, out_channels};

    CudaDeviceGuard dev_guard(input.device().index);
    // Use singleton cuDNN handle
    cudnnHandle_t handle = CuDNNHandle::get();
    CuDNNHandle::set_stream(stream);

    // Convert dtype
    cudnnDataType_t cudnn_dtype;
    switch (input.dtype()) {
        case DType::Float32: cudnn_dtype = CUDNN_DATA_FLOAT; break;
        case DType::Float64: cudnn_dtype = CUDNN_DATA_DOUBLE; break;
        case DType::Float16: cudnn_dtype = CUDNN_DATA_HALF; break;
        case DType::BFloat16: cudnn_dtype = CUDNN_DATA_BFLOAT16; break;
        default:
            throw std::runtime_error("cuDNN Conv2d NHWC: unsupported dtype");
    }

    // Create NHWC descriptors
    TensorDescriptor input_desc, output_desc;
    FilterDescriptor filter_desc;
    ConvolutionDescriptor conv_desc;

    // Use NHWC format for all descriptors
    input_desc.set_nhwc(cudnn_dtype, batch, in_channels, height, width);
    output_desc.set_nhwc(cudnn_dtype, batch, out_channels, out_h, out_w);
    filter_desc.set_nhwc(cudnn_dtype, out_channels, in_channels / groups, kernel_h, kernel_w);
    conv_desc.set(padding, padding, stride, stride, dilation, dilation, cudnn_dtype);

    if (groups > 1) {
        conv_desc.set_group_count(groups);
    }

    // Create cache key for NHWC algorithm lookup (still scalar API — E1
    // extends the cache key struct to per-axis fields, so we replicate
    // each scalar across both axes here).
    Conv2dCacheKey cache_key{
        batch, in_channels, height, width,
        out_channels, kernel_h, kernel_w,
        stride, stride, padding, padding, dilation, dilation, groups,
        cudnn_dtype, TensorFormat::NHWC,
        /*prefer_precise_f32=*/(cudnn_dtype == CUDNN_DATA_FLOAT) &&
                                !::tenzor::cuda::matmul::allow_tf32()
    };

    cudnnConvolutionFwdAlgo_t algo;
    size_t workspace_size;

    // Try to get cached algorithm
    CachedFwdAlgo cached;
    if (Conv2dAlgoCache::instance().get_fwd(cache_key, cached)) {
        algo = cached.algo;
        workspace_size = cached.workspace_size;
    } else {
        // Cache miss - use cudnnFindConvolutionForwardAlgorithmEx for auto-tuning
        constexpr int kMaxAlgos = 8;
        const size_t kMaxWorkspaceSize = CuDNNWorkspace::max_workspace_size();
        void* search_workspace = CuDNNWorkspace::get(kMaxWorkspaceSize);

        int returned_algo_count = 0;
        cudnnConvolutionFwdAlgoPerf_t perf_results[kMaxAlgos];

        cudnnStatus_t find_status = cudnnFindConvolutionForwardAlgorithmEx(
            handle,
            input_desc.get(),
            input_nhwc.data_ptr(),
            filter_desc.get(),
            weight_nhwc.data_ptr(),
            conv_desc.get(),
            output_desc.get(),
            output_nhwc.data_ptr(),
            kMaxAlgos,
            &returned_algo_count,
            perf_results,
            search_workspace,
            kMaxWorkspaceSize
        );

        if (find_status != CUDNN_STATUS_SUCCESS || returned_algo_count == 0) {
            // Fallback to heuristic
            cudnnConvolutionFwdAlgoPerf_t heuristic_result;
            int heuristic_count = 0;
            CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm_v7(
                handle, input_desc.get(), filter_desc.get(), conv_desc.get(),
                output_desc.get(), 1, &heuristic_count, &heuristic_result
            ));
            if (heuristic_count <= 0) { throw std::runtime_error("cuDNN Conv2d forward: no convolution algorithm available for this descriptor"); }
            algo = heuristic_result.algo;
            CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(
                handle, input_desc.get(), filter_desc.get(), conv_desc.get(),
                output_desc.get(), algo, &workspace_size
            ));
        } else {
            // Find the fastest successful algorithm
            algo = perf_results[0].algo;
            workspace_size = perf_results[0].memory;
            float best_time = std::numeric_limits<float>::max();

            for (int i = 0; i < returned_algo_count; ++i) {
                if (perf_results[i].status == CUDNN_STATUS_SUCCESS &&
                    perf_results[i].memory <= kMaxWorkspaceSize &&
                    perf_results[i].time < best_time) {
                    best_time = perf_results[i].time;
                    algo = perf_results[i].algo;
                    workspace_size = perf_results[i].memory;
                }
            }
        }

        // Cache the result
        Conv2dAlgoCache::instance().set_fwd(cache_key, {algo, workspace_size});
    }

    // Get workspace
    void* workspace = CuDNNWorkspace::get(workspace_size);

    // Perform convolution in NHWC format
    const float alpha = 1.0f;
    const float beta = 0.0f;

    if (input.dtype() == DType::Float32) {
        CUDNN_CHECK(cudnnConvolutionForward(
            handle,
            &alpha,
            input_desc.get(),
            input_nhwc.data<float>(),
            filter_desc.get(),
            weight_nhwc.data<float>(),
            conv_desc.get(),
            algo,
            workspace,
            workspace_size,
            &beta,
            output_desc.get(),
            output_nhwc.data<float>()
        ));
    } else if (input.dtype() == DType::Float64) {
        const double alpha_d = 1.0;
        const double beta_d = 0.0;
        CUDNN_CHECK(cudnnConvolutionForward(
            handle,
            &alpha_d,
            input_desc.get(),
            input_nhwc.data<double>(),
            filter_desc.get(),
            weight_nhwc.data<double>(),
            conv_desc.get(),
            algo,
            workspace,
            workspace_size,
            &beta_d,
            output_desc.get(),
            output_nhwc.data<double>()
        ));
    } else if (input.dtype() == DType::Float16) {
        CUDNN_CHECK(cudnnConvolutionForward(
            handle,
            &alpha,
            input_desc.get(),
            input_nhwc.data<Float16>(),
            filter_desc.get(),
            weight_nhwc.data<Float16>(),
            conv_desc.get(),
            algo,
            workspace,
            workspace_size,
            &beta,
            output_desc.get(),
            output_nhwc.data<Float16>()
        ));
    } else if (input.dtype() == DType::BFloat16) {
        CUDNN_CHECK(cudnnConvolutionForward(
            handle,
            &alpha,
            input_desc.get(),
            input_nhwc.data<BFloat16>(),
            filter_desc.get(),
            weight_nhwc.data<BFloat16>(),
            conv_desc.get(),
            algo,
            workspace,
            workspace_size,
            &beta,
            output_desc.get(),
            output_nhwc.data<BFloat16>()
        ));
    }

    // Add bias if present (in NHWC format, bias is added to channel dimension)
    if (bias != nullptr) {
        TensorDescriptor bias_desc;
        bias_desc.set_nhwc(cudnn_dtype, 1, out_channels, 1, 1);

        const float alpha_bias = 1.0f;
        const float beta_bias = 1.0f;

        if (input.dtype() == DType::Float32) {
            CUDNN_CHECK(cudnnAddTensor(
                handle,
                &alpha_bias,
                bias_desc.get(),
                bias->data<float>(),
                &beta_bias,
                output_desc.get(),
                output_nhwc.data<float>()
            ));
        } else if (input.dtype() == DType::Float64) {
            const double alpha_bias_d = 1.0;
            const double beta_bias_d = 1.0;
            CUDNN_CHECK(cudnnAddTensor(
                handle,
                &alpha_bias_d,
                bias_desc.get(),
                bias->data<double>(),
                &beta_bias_d,
                output_desc.get(),
                output_nhwc.data<double>()
            ));
        } else if (input.dtype() == DType::Float16) {
            CUDNN_CHECK(cudnnAddTensor(
                handle,
                &alpha_bias,
                bias_desc.get(),
                bias->data<Float16>(),
                &beta_bias,
                output_desc.get(),
                output_nhwc.data<Float16>()
            ));
        } else if (input.dtype() == DType::BFloat16) {
            CUDNN_CHECK(cudnnAddTensor(
                handle,
                &alpha_bias,
                bias_desc.get(),
                bias->data<BFloat16>(),
                &beta_bias,
                output_desc.get(),
                output_nhwc.data<BFloat16>()
            ));
        }
    }

    // Saturate FP16 output: clamp any ±Inf to ±65504 (max finite Float16 value)
    if (input.dtype() == DType::Float16) {
        fp16_saturate(output_nhwc.data<Float16>(), output_nhwc.numel(), stream);
    }

    // If input was already NHWC, return output directly (it already has NHWC strides)
    if (input_is_nhwc) {
        return output_nhwc;
    }

    // For NCHW input, convert output from NHWC strides to contiguous NCHW
    return output_nhwc.contiguous();
}

// ============================================================================
// NHWC-Optimized Conv2d Backward Implementation
// ============================================================================

auto cudnn_conv2d_backward_nhwc(
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& weight,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    int64_t groups,
    bool compute_grad_input,
    bool compute_grad_weight,
    bool compute_grad_bias,
    void* stream_ptr
) -> std::tuple<Tensor, Tensor, Tensor> {
    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);

    // Extract dimensions
    auto input_shape = input.shape();
    auto weight_shape = weight.shape();
    auto grad_shape = grad_output.shape();

    int64_t batch = input_shape[0];
    int64_t in_channels = input_shape[1];
    int64_t height = input_shape[2];
    int64_t width = input_shape[3];

    int64_t out_channels = weight_shape[0];
    int64_t kernel_h = weight_shape[2];
    int64_t kernel_w = weight_shape[3];

    int64_t out_h = grad_shape[2];
    int64_t out_w = grad_shape[3];

    // Initialize gradients in NCHW format
    Tensor grad_input({batch, in_channels, height, width}, input.dtype(), input.device());
    Tensor grad_weight({out_channels, in_channels / groups, kernel_h, kernel_w}, weight.dtype(), weight.device());
    Tensor grad_bias({out_channels}, weight.dtype(), weight.device());

    // Convert to NHWC format
    Tensor input_nhwc = nchw_to_nhwc(input, stream);
    Tensor weight_nhwc = filter_nchw_to_nhwc(weight, stream);
    Tensor grad_output_nhwc = nchw_to_nhwc(grad_output, stream);

    CudaDeviceGuard dev_guard(input.device().index);
    // Use singleton cuDNN handle
    cudnnHandle_t handle = CuDNNHandle::get();
    CuDNNHandle::set_stream(stream);

    // Convert dtype
    cudnnDataType_t cudnn_dtype;
    switch (input.dtype()) {
        case DType::Float32: cudnn_dtype = CUDNN_DATA_FLOAT; break;
        case DType::Float64: cudnn_dtype = CUDNN_DATA_DOUBLE; break;
        case DType::Float16: cudnn_dtype = CUDNN_DATA_HALF; break;
        case DType::BFloat16: cudnn_dtype = CUDNN_DATA_BFLOAT16; break;
        default:
            throw std::runtime_error("cuDNN Conv2d backward NHWC: unsupported dtype");
    }

    // Create NHWC descriptors
    TensorDescriptor input_desc, grad_output_desc;
    FilterDescriptor filter_desc;
    ConvolutionDescriptor conv_desc;

    input_desc.set_nhwc(cudnn_dtype, batch, in_channels, height, width);
    grad_output_desc.set_nhwc(cudnn_dtype, batch, out_channels, out_h, out_w);
    filter_desc.set_nhwc(cudnn_dtype, out_channels, in_channels / groups, kernel_h, kernel_w);
    conv_desc.set(padding, padding, stride, stride, dilation, dilation, cudnn_dtype);

    if (groups > 1) {
        conv_desc.set_group_count(groups);
    }

    // Create cache key for NHWC algorithm lookup (scalar API → replicated).
    Conv2dCacheKey cache_key{
        batch, in_channels, height, width,
        out_channels, kernel_h, kernel_w,
        stride, stride, padding, padding, dilation, dilation, groups,
        cudnn_dtype, TensorFormat::NHWC,
        /*prefer_precise_f32=*/(cudnn_dtype == CUDNN_DATA_FLOAT) &&
                                !::tenzor::cuda::matmul::allow_tf32()
    };

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Compute gradient w.r.t. input
    if (compute_grad_input) {
        // Create NHWC output for grad_input
        std::vector<int64_t> grad_input_nhwc_shape = {batch, height, width, in_channels};
        Tensor grad_input_nhwc(grad_input_nhwc_shape, input.dtype(), input.device());

        TensorDescriptor grad_input_desc;
        grad_input_desc.set_nhwc(cudnn_dtype, batch, in_channels, height, width);

        cudnnConvolutionBwdDataAlgo_t algo;
        size_t workspace_size;

        CachedBwdDataAlgo cached;
        if (Conv2dAlgoCache::instance().get_bwd_data(cache_key, cached)) {
            algo = cached.algo;
            workspace_size = cached.workspace_size;
        } else {
            constexpr int kMaxAlgos = 8;
            const size_t kMaxWorkspaceSize = CuDNNWorkspace::max_workspace_size();
            int returned_algo_count = 0;
            cudnnConvolutionBwdDataAlgoPerf_t perf_results[kMaxAlgos];

            CUDNN_CHECK(cudnnGetConvolutionBackwardDataAlgorithm_v7(
                handle, filter_desc.get(), grad_output_desc.get(),
                conv_desc.get(), grad_input_desc.get(),
                kMaxAlgos, &returned_algo_count, perf_results
            ));

            if (returned_algo_count <= 0) { throw std::runtime_error("cuDNN Conv2d backward-data: no algorithm available for this descriptor"); }
            algo = perf_results[0].algo;
            workspace_size = 0;
            float best_time = std::numeric_limits<float>::max();

            for (int i = 0; i < returned_algo_count; ++i) {
                if (perf_results[i].status == CUDNN_STATUS_SUCCESS &&
                    perf_results[i].memory <= kMaxWorkspaceSize) {
                    size_t ws_size = 0;
                    cudnnStatus_t ws_status = cudnnGetConvolutionBackwardDataWorkspaceSize(
                        handle, filter_desc.get(), grad_output_desc.get(),
                        conv_desc.get(), grad_input_desc.get(),
                        perf_results[i].algo, &ws_size
                    );
                    if (ws_status == CUDNN_STATUS_SUCCESS && ws_size <= kMaxWorkspaceSize) {
                        if (perf_results[i].time < best_time) {
                            best_time = perf_results[i].time;
                            algo = perf_results[i].algo;
                            workspace_size = ws_size;
                        }
                    }
                }
            }

            Conv2dAlgoCache::instance().set_bwd_data(cache_key, {algo, workspace_size});
        }

        void* workspace = CuDNNWorkspace::get(workspace_size);

        if (input.dtype() == DType::Float32) {
            CUDNN_CHECK(cudnnConvolutionBackwardData(
                handle, &alpha, filter_desc.get(), weight_nhwc.data<float>(),
                grad_output_desc.get(), grad_output_nhwc.data<float>(),
                conv_desc.get(), algo, workspace, workspace_size,
                &beta, grad_input_desc.get(), grad_input_nhwc.data<float>()
            ));
        } else if (input.dtype() == DType::Float64) {
            const double alpha_d = 1.0, beta_d = 0.0;
            CUDNN_CHECK(cudnnConvolutionBackwardData(
                handle, &alpha_d, filter_desc.get(), weight_nhwc.data<double>(),
                grad_output_desc.get(), grad_output_nhwc.data<double>(),
                conv_desc.get(), algo, workspace, workspace_size,
                &beta_d, grad_input_desc.get(), grad_input_nhwc.data<double>()
            ));
        } else if (input.dtype() == DType::Float16) {
            CUDNN_CHECK(cudnnConvolutionBackwardData(
                handle, &alpha, filter_desc.get(), weight_nhwc.data<Float16>(),
                grad_output_desc.get(), grad_output_nhwc.data<Float16>(),
                conv_desc.get(), algo, workspace, workspace_size,
                &beta, grad_input_desc.get(), grad_input_nhwc.data<Float16>()
            ));
        } else if (input.dtype() == DType::BFloat16) {
            CUDNN_CHECK(cudnnConvolutionBackwardData(
                handle, &alpha, filter_desc.get(), weight_nhwc.data<BFloat16>(),
                grad_output_desc.get(), grad_output_nhwc.data<BFloat16>(),
                conv_desc.get(), algo, workspace, workspace_size,
                &beta, grad_input_desc.get(), grad_input_nhwc.data<BFloat16>()
            ));
        } else {
            throw std::runtime_error("cuDNN Conv2d backward NHWC (grad_input): unsupported dtype");
        }

        // Convert grad_input back to NCHW
        grad_input = nhwc_to_nchw(grad_input_nhwc, in_channels, stream);
    }

    // Compute gradient w.r.t. weight
    if (compute_grad_weight) {
        // Create NHWC output for grad_weight
        std::vector<int64_t> grad_weight_nhwc_shape = {out_channels, kernel_h, kernel_w, in_channels / groups};
        Tensor grad_weight_nhwc(grad_weight_nhwc_shape, weight.dtype(), weight.device());

        FilterDescriptor grad_filter_desc;
        grad_filter_desc.set_nhwc(cudnn_dtype, out_channels, in_channels / groups, kernel_h, kernel_w);

        cudnnConvolutionBwdFilterAlgo_t algo;
        size_t workspace_size;

        CachedBwdFilterAlgo cached;
        if (Conv2dAlgoCache::instance().get_bwd_filter(cache_key, cached)) {
            algo = cached.algo;
            workspace_size = cached.workspace_size;
        } else {
            constexpr int kMaxAlgos = 8;
            const size_t kMaxWorkspaceSize = CuDNNWorkspace::max_workspace_size();
            int returned_algo_count = 0;
            cudnnConvolutionBwdFilterAlgoPerf_t perf_results[kMaxAlgos];

            CUDNN_CHECK(cudnnGetConvolutionBackwardFilterAlgorithm_v7(
                handle, input_desc.get(), grad_output_desc.get(),
                conv_desc.get(), grad_filter_desc.get(),
                kMaxAlgos, &returned_algo_count, perf_results
            ));

            if (returned_algo_count <= 0) { throw std::runtime_error("cuDNN Conv2d backward-filter: no algorithm available for this descriptor"); }
            algo = perf_results[0].algo;
            workspace_size = 0;
            float best_time = std::numeric_limits<float>::max();

            for (int i = 0; i < returned_algo_count; ++i) {
                if (perf_results[i].status == CUDNN_STATUS_SUCCESS &&
                    perf_results[i].memory <= kMaxWorkspaceSize) {
                    size_t ws_size = 0;
                    cudnnStatus_t ws_status = cudnnGetConvolutionBackwardFilterWorkspaceSize(
                        handle, input_desc.get(), grad_output_desc.get(),
                        conv_desc.get(), grad_filter_desc.get(),
                        perf_results[i].algo, &ws_size
                    );
                    if (ws_status == CUDNN_STATUS_SUCCESS && ws_size <= kMaxWorkspaceSize) {
                        if (perf_results[i].time < best_time) {
                            best_time = perf_results[i].time;
                            algo = perf_results[i].algo;
                            workspace_size = ws_size;
                        }
                    }
                }
            }

            Conv2dAlgoCache::instance().set_bwd_filter(cache_key, {algo, workspace_size});
        }

        void* workspace = CuDNNWorkspace::get(workspace_size);

        if (input.dtype() == DType::Float32) {
            CUDNN_CHECK(cudnnConvolutionBackwardFilter(
                handle, &alpha, input_desc.get(), input_nhwc.data<float>(),
                grad_output_desc.get(), grad_output_nhwc.data<float>(),
                conv_desc.get(), algo, workspace, workspace_size,
                &beta, grad_filter_desc.get(), grad_weight_nhwc.data<float>()
            ));
        } else if (input.dtype() == DType::Float64) {
            const double alpha_d = 1.0, beta_d = 0.0;
            CUDNN_CHECK(cudnnConvolutionBackwardFilter(
                handle, &alpha_d, input_desc.get(), input_nhwc.data<double>(),
                grad_output_desc.get(), grad_output_nhwc.data<double>(),
                conv_desc.get(), algo, workspace, workspace_size,
                &beta_d, grad_filter_desc.get(), grad_weight_nhwc.data<double>()
            ));
        } else if (input.dtype() == DType::Float16) {
            CUDNN_CHECK(cudnnConvolutionBackwardFilter(
                handle, &alpha, input_desc.get(), input_nhwc.data<Float16>(),
                grad_output_desc.get(), grad_output_nhwc.data<Float16>(),
                conv_desc.get(), algo, workspace, workspace_size,
                &beta, grad_filter_desc.get(), grad_weight_nhwc.data<Float16>()
            ));
        } else if (input.dtype() == DType::BFloat16) {
            CUDNN_CHECK(cudnnConvolutionBackwardFilter(
                handle, &alpha, input_desc.get(), input_nhwc.data<BFloat16>(),
                grad_output_desc.get(), grad_output_nhwc.data<BFloat16>(),
                conv_desc.get(), algo, workspace, workspace_size,
                &beta, grad_filter_desc.get(), grad_weight_nhwc.data<BFloat16>()
            ));
        } else {
            throw std::runtime_error("cuDNN Conv2d backward NHWC (grad_weight): unsupported dtype");
        }

        // Convert grad_weight back to NCHW format
        // Need filter_nhwc_to_nchw function (reverse of filter_nchw_to_nhwc)
        // For now, use a simple kernel call
        auto gw_shape = grad_weight_nhwc.shape();
        int64_t gw_k = gw_shape[0], gw_kh = gw_shape[1], gw_kw = gw_shape[2], gw_c = gw_shape[3];
        const int64_t total = gw_k * gw_c * gw_kh * gw_kw;

        // Reuse existing kernel with swapped interpretation
        if (weight.dtype() == DType::Float32) {
            // NHWC to NCHW for filter: [K, kH, kW, C] -> [K, C, kH, kW]
            // This is same as nhwc_to_nchw but with different dims
            auto [grid_size, block_size] = optimal_launch_config(nhwc_to_nchw_kernel<float>, total);
            nhwc_to_nchw_kernel<float><<<grid_size, block_size, 0, stream>>>(
                grad_weight_nhwc.data<float>(), grad_weight.data<float>(),
                gw_k, gw_c, gw_kh, gw_kw
            );
        } else if (weight.dtype() == DType::Float16) {
            auto [grid_size, block_size] = optimal_launch_config(nhwc_to_nchw_kernel<Float16>, total);
            nhwc_to_nchw_kernel<Float16><<<grid_size, block_size, 0, stream>>>(
                grad_weight_nhwc.data<Float16>(), grad_weight.data<Float16>(),
                gw_k, gw_c, gw_kh, gw_kw
            );
        } else if (weight.dtype() == DType::Float64) {
            auto [grid_size, block_size] = optimal_launch_config(nhwc_to_nchw_kernel<double>, total);
            nhwc_to_nchw_kernel<double><<<grid_size, block_size, 0, stream>>>(
                grad_weight_nhwc.data<double>(), grad_weight.data<double>(),
                gw_k, gw_c, gw_kh, gw_kw
            );
        } else if (weight.dtype() == DType::BFloat16) {
            auto [grid_size, block_size] = optimal_launch_config(nhwc_to_nchw_kernel<BFloat16>, total);
            nhwc_to_nchw_kernel<BFloat16><<<grid_size, block_size, 0, stream>>>(
                grad_weight_nhwc.data<BFloat16>(), grad_weight.data<BFloat16>(),
                gw_k, gw_c, gw_kh, gw_kw
            );
        } else {
            throw std::runtime_error("cuDNN Conv2d backward NHWC (grad_weight NCHW convert): unsupported dtype");
        }
        CUDA_CHECK(cudaGetLastError());
    }

    // Compute gradient w.r.t. bias
    if (compute_grad_bias) {
        TensorDescriptor bias_desc;
        bias_desc.set_nhwc(cudnn_dtype, 1, out_channels, 1, 1);

        if (input.dtype() == DType::Float32) {
            CUDNN_CHECK(cudnnConvolutionBackwardBias(
                handle, &alpha, grad_output_desc.get(), grad_output_nhwc.data<float>(),
                &beta, bias_desc.get(), grad_bias.data<float>()
            ));
        } else if (input.dtype() == DType::Float64) {
            const double alpha_d = 1.0, beta_d = 0.0;
            CUDNN_CHECK(cudnnConvolutionBackwardBias(
                handle, &alpha_d, grad_output_desc.get(), grad_output_nhwc.data<double>(),
                &beta_d, bias_desc.get(), grad_bias.data<double>()
            ));
        } else if (input.dtype() == DType::Float16) {
            CUDNN_CHECK(cudnnConvolutionBackwardBias(
                handle, &alpha, grad_output_desc.get(), grad_output_nhwc.data<Float16>(),
                &beta, bias_desc.get(), grad_bias.data<Float16>()
            ));
        } else if (input.dtype() == DType::BFloat16) {
            CUDNN_CHECK(cudnnConvolutionBackwardBias(
                handle, &alpha, grad_output_desc.get(), grad_output_nhwc.data<BFloat16>(),
                &beta, bias_desc.get(), grad_bias.data<BFloat16>()
            ));
        } else {
            throw std::runtime_error("cuDNN Conv2d backward NHWC (grad_bias): unsupported dtype");
        }
    }

    return std::make_tuple(grad_input, grad_weight, grad_bias);
}


// ============================================================================
// cuDNN MaxPool2d argmax index kernel
// ============================================================================
//
// cudnnPoolingForward produces only the pooled values, not the argmax indices
// that return_indices=True / MaxUnpool2d require. This kernel recomputes the
// flattened H*W argmax for each output element using the SAME deterministic
// first-occurrence tie-break and index convention (max_idx = h * W + w) as the
// native cuda::maxpool2d_forward_impl and the CPU reference, so the returned
// indices are valid and cross-backend consistent. cuDNN MaxPool here uses
// dilation = 1 (no dilation is set on the pooling descriptor).
template<typename T>
__global__ void cudnn_maxpool2d_argmax_kernel(
    const T* __restrict__ input,
    int64_t* __restrict__ indices,
    int64_t N, int64_t C, int64_t H, int64_t W,
    int64_t H_out, int64_t W_out,
    int64_t kernel_h, int64_t kernel_w,
    int64_t stride_h, int64_t stride_w,
    int64_t pad_h, int64_t pad_w
) {
    const int64_t total = N * C * H_out * W_out;

    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {

        int64_t ow = idx % W_out;
        int64_t oh = (idx / W_out) % H_out;
        int64_t c  = (idx / (W_out * H_out)) % C;
        int64_t n  = idx / (W_out * H_out * C);

        int64_t h_start = oh * stride_h - pad_h;
        int64_t w_start = ow * stride_w - pad_w;

        float max_val = -std::numeric_limits<float>::infinity();
        int64_t max_idx = 0;

        for (int64_t kh = 0; kh < kernel_h; ++kh) {
            for (int64_t kw = 0; kw < kernel_w; ++kw) {
                int64_t h = h_start + kh;
                int64_t w = w_start + kw;
                if (h >= 0 && h < H && w >= 0 && w < W) {
                    int64_t in_idx = ((n * C + c) * H + h) * W + w;
                    float val;
                    if constexpr (std::is_same_v<T, __half>) {
                        val = __half2float(input[in_idx]);
                    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
                        val = __bfloat162float(input[in_idx]);
                    } else {
                        val = static_cast<float>(input[in_idx]);
                    }
                    if (val > max_val) {
                        max_val = val;
                        max_idx = h * W + w;
                    }
                }
            }
        }
        indices[idx] = max_idx;
    }
}

// Float64 variant keeps double comparison precision.
__global__ void cudnn_maxpool2d_argmax_kernel_f64(
    const double* __restrict__ input,
    int64_t* __restrict__ indices,
    int64_t N, int64_t C, int64_t H, int64_t W,
    int64_t H_out, int64_t W_out,
    int64_t kernel_h, int64_t kernel_w,
    int64_t stride_h, int64_t stride_w,
    int64_t pad_h, int64_t pad_w
) {
    const int64_t total = N * C * H_out * W_out;

    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {

        int64_t ow = idx % W_out;
        int64_t oh = (idx / W_out) % H_out;
        int64_t c  = (idx / (W_out * H_out)) % C;
        int64_t n  = idx / (W_out * H_out * C);

        int64_t h_start = oh * stride_h - pad_h;
        int64_t w_start = ow * stride_w - pad_w;

        double max_val = -std::numeric_limits<double>::infinity();
        int64_t max_idx = 0;

        for (int64_t kh = 0; kh < kernel_h; ++kh) {
            for (int64_t kw = 0; kw < kernel_w; ++kw) {
                int64_t h = h_start + kh;
                int64_t w = w_start + kw;
                if (h >= 0 && h < H && w >= 0 && w < W) {
                    int64_t in_idx = ((n * C + c) * H + h) * W + w;
                    double val = input[in_idx];
                    if (val > max_val) {
                        max_val = val;
                        max_idx = h * W + w;
                    }
                }
            }
        }
        indices[idx] = max_idx;
    }
}

// ============================================================================
// cuDNN MaxPool2d Forward Implementation
// ============================================================================

auto cudnn_maxpool2d_forward(
    const Tensor& input,
    int64_t kernel_size,
    int64_t stride,
    int64_t padding,
    cudaStream_t stream
) -> std::pair<Tensor, Tensor> {
    return cudnn_maxpool2d_forward(
        input,
        kernel_size, kernel_size,
        stride, stride,
        padding, padding,
        stream);
}

auto cudnn_maxpool2d_forward(
    const Tensor& input,
    int64_t kernel_h, int64_t kernel_w,
    int64_t stride_h, int64_t stride_w,
    int64_t pad_h, int64_t pad_w,
    cudaStream_t stream
) -> std::pair<Tensor, Tensor> {
    // The cuDNN descriptors below are built with packed NCHW strides derived
    // from shape, and the argmax recompute kernel indexes input flat as
    // ((n*C+c)*H+h)*W+w — both assume contiguity. Materialize a contiguous
    // copy so a non-contiguous (channels-last / sliced / permuted) view is not
    // read with the wrong strides.
    const Tensor& input_c = input.is_contiguous() ? input : input.contiguous();
    auto shape = input_c.shape();
    int64_t batch = shape[0];
    int64_t channels = shape[1];
    int64_t height = shape[2];
    int64_t width = shape[3];

    // Calculate output dimensions
    int64_t out_h = (height + 2 * pad_h - kernel_h) / stride_h + 1;
    int64_t out_w = (width + 2 * pad_w - kernel_w) / stride_w + 1;

    // Create output tensor
    Tensor output({batch, channels, out_h, out_w}, input_c.dtype(), input_c.device());
    // cuDNN doesn't return indices directly, we'll compute them separately if needed
    Tensor indices({batch, channels, out_h, out_w}, DType::Int64, input_c.device());

    CudaDeviceGuard dev_guard(input_c.device().index);
    cudnnHandle_t handle = CuDNNHandle::get();
    CuDNNHandle::set_stream(stream);

    // Setup descriptors
    cudnnDataType_t cudnn_dtype;
    switch (input_c.dtype()) {
        case DType::Float32: cudnn_dtype = CUDNN_DATA_FLOAT; break;
        case DType::Float64: cudnn_dtype = CUDNN_DATA_DOUBLE; break;
        case DType::Float16: cudnn_dtype = CUDNN_DATA_HALF; break;
        case DType::BFloat16: cudnn_dtype = CUDNN_DATA_BFLOAT16; break;
        default:
            throw std::runtime_error("cuDNN MaxPool2d: unsupported dtype");
    }

    TensorDescriptor input_desc, output_desc;
    PoolingDescriptor pool_desc;

    input_desc.set(cudnn_dtype, batch, channels, height, width);
    output_desc.set(cudnn_dtype, batch, channels, out_h, out_w);
    pool_desc.set_maxpool(kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w);

    // cuDNN requires alpha/beta type to match tensor dtype
    if (input_c.dtype() == DType::Float64) {
        double alpha = 1.0, beta = 0.0;
        CUDNN_CHECK(cudnnPoolingForward(
            handle,
            pool_desc.get(),
            &alpha,
            input_desc.get(),
            input_c.data_ptr(),
            &beta,
            output_desc.get(),
            output.data_ptr()
        ));
    } else {
        float alpha = 1.0f, beta = 0.0f;
        CUDNN_CHECK(cudnnPoolingForward(
            handle,
            pool_desc.get(),
            &alpha,
            input_desc.get(),
            input_c.data_ptr(),
            &beta,
            output_desc.get(),
            output.data_ptr()
        ));
    }

    // cudnnPoolingForward does not emit argmax indices. Recompute them so that
    // return_indices=True / MaxUnpool2d on the cuDNN CUDA path get valid,
    // cross-backend-consistent flattened H*W indices (same convention and
    // first-occurrence tie-break as the native cuda::maxpool2d_forward_impl).
    {
        const int64_t total = batch * channels * out_h * out_w;
        switch (input_c.dtype()) {
            case DType::Float32: {
                auto [grid, block] = optimal_launch_config(cudnn_maxpool2d_argmax_kernel<float>, total);
                cudnn_maxpool2d_argmax_kernel<float><<<grid, block, 0, stream>>>(
                    input_c.data<float>(), indices.data<int64_t>(),
                    batch, channels, height, width, out_h, out_w,
                    kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w);
                break;
            }
            case DType::Float64: {
                auto [grid, block] = optimal_launch_config(cudnn_maxpool2d_argmax_kernel_f64, total);
                cudnn_maxpool2d_argmax_kernel_f64<<<grid, block, 0, stream>>>(
                    input_c.data<double>(), indices.data<int64_t>(),
                    batch, channels, height, width, out_h, out_w,
                    kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w);
                break;
            }
            case DType::Float16: {
                auto [grid, block] = optimal_launch_config(cudnn_maxpool2d_argmax_kernel<__half>, total);
                cudnn_maxpool2d_argmax_kernel<__half><<<grid, block, 0, stream>>>(
                    reinterpret_cast<const __half*>(input_c.data<Float16>()), indices.data<int64_t>(),
                    batch, channels, height, width, out_h, out_w,
                    kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w);
                break;
            }
            case DType::BFloat16: {
                auto [grid, block] = optimal_launch_config(cudnn_maxpool2d_argmax_kernel<__nv_bfloat16>, total);
                cudnn_maxpool2d_argmax_kernel<__nv_bfloat16><<<grid, block, 0, stream>>>(
                    reinterpret_cast<const __nv_bfloat16*>(input_c.data<BFloat16>()), indices.data<int64_t>(),
                    batch, channels, height, width, out_h, out_w,
                    kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w);
                break;
            }
            default:
                throw std::runtime_error("cuDNN MaxPool2d argmax: unsupported dtype");
        }
        CUDA_CHECK(cudaGetLastError());
    }

    return {output, indices};
}

// ============================================================================
// cuDNN MaxPool2d Backward Implementation
// ============================================================================

auto cudnn_maxpool2d_backward(
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& output,
    int64_t kernel_size,
    int64_t stride,
    int64_t padding,
    cudaStream_t stream
) -> Tensor {
    return cudnn_maxpool2d_backward(
        grad_output, input, output,
        kernel_size, kernel_size,
        stride, stride,
        padding, padding,
        stream);
}

auto cudnn_maxpool2d_backward(
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& output,
    int64_t kernel_h, int64_t kernel_w,
    int64_t stride_h, int64_t stride_w,
    int64_t pad_h, int64_t pad_w,
    cudaStream_t stream
) -> Tensor {
    // Descriptors below assume packed NCHW; materialize contiguous copies so a
    // non-contiguous input/output/grad_output view is not read with wrong strides.
    const Tensor& input_c = input.is_contiguous() ? input : input.contiguous();
    const Tensor& output_c = output.is_contiguous() ? output : output.contiguous();
    const Tensor& grad_output_c = grad_output.is_contiguous() ? grad_output : grad_output.contiguous();

    auto in_shape = input_c.shape();
    int64_t batch = in_shape[0];
    int64_t channels = in_shape[1];
    int64_t height = in_shape[2];
    int64_t width = in_shape[3];

    auto out_shape = output_c.shape();
    int64_t out_h = out_shape[2];
    int64_t out_w = out_shape[3];

    Tensor grad_input({batch, channels, height, width}, input_c.dtype(), input_c.device());

    CudaDeviceGuard dev_guard(input_c.device().index);
    cudnnHandle_t handle = CuDNNHandle::get();
    CuDNNHandle::set_stream(stream);

    cudnnDataType_t cudnn_dtype;
    switch (input_c.dtype()) {
        case DType::Float32: cudnn_dtype = CUDNN_DATA_FLOAT; break;
        case DType::Float64: cudnn_dtype = CUDNN_DATA_DOUBLE; break;
        case DType::Float16: cudnn_dtype = CUDNN_DATA_HALF; break;
        case DType::BFloat16: cudnn_dtype = CUDNN_DATA_BFLOAT16; break;
        default:
            throw std::runtime_error("cuDNN MaxPool2d backward: unsupported dtype");
    }

    TensorDescriptor input_desc, output_desc;
    PoolingDescriptor pool_desc;

    input_desc.set(cudnn_dtype, batch, channels, height, width);
    output_desc.set(cudnn_dtype, batch, channels, out_h, out_w);
    pool_desc.set_maxpool(kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w);

    // cuDNN requires alpha/beta type to match tensor dtype
    if (input_c.dtype() == DType::Float64) {
        double alpha = 1.0, beta = 0.0;
        CUDNN_CHECK(cudnnPoolingBackward(
            handle,
            pool_desc.get(),
            &alpha,
            output_desc.get(),
            output_c.data_ptr(),
            output_desc.get(),
            grad_output_c.data_ptr(),
            input_desc.get(),
            input_c.data_ptr(),
            &beta,
            input_desc.get(),
            grad_input.data_ptr()
        ));
    } else {
        float alpha = 1.0f, beta = 0.0f;
        CUDNN_CHECK(cudnnPoolingBackward(
            handle,
            pool_desc.get(),
            &alpha,
            output_desc.get(),
            output_c.data_ptr(),
            output_desc.get(),
            grad_output_c.data_ptr(),
            input_desc.get(),
            input_c.data_ptr(),
            &beta,
            input_desc.get(),
            grad_input.data_ptr()
        ));
    }

    return grad_input;
}

// ============================================================================
// cuDNN AvgPool2d Forward Implementation
// ============================================================================

auto cudnn_avgpool2d_forward(
    const Tensor& input,
    int64_t kernel_size,
    int64_t stride,
    int64_t padding,
    bool count_include_pad,
    cudaStream_t stream
) -> Tensor {
    return cudnn_avgpool2d_forward(
        input,
        kernel_size, kernel_size,
        stride, stride,
        padding, padding,
        count_include_pad,
        stream);
}

auto cudnn_avgpool2d_forward(
    const Tensor& input,
    int64_t kernel_h, int64_t kernel_w,
    int64_t stride_h, int64_t stride_w,
    int64_t pad_h, int64_t pad_w,
    bool count_include_pad,
    cudaStream_t stream
) -> Tensor {
    // Descriptor below assumes packed NCHW; materialize a contiguous copy.
    const Tensor& input_c = input.is_contiguous() ? input : input.contiguous();
    auto shape = input_c.shape();
    int64_t batch = shape[0];
    int64_t channels = shape[1];
    int64_t height = shape[2];
    int64_t width = shape[3];

    int64_t out_h = (height + 2 * pad_h - kernel_h) / stride_h + 1;
    int64_t out_w = (width + 2 * pad_w - kernel_w) / stride_w + 1;

    Tensor output({batch, channels, out_h, out_w}, input_c.dtype(), input_c.device());

    CudaDeviceGuard dev_guard(input_c.device().index);
    cudnnHandle_t handle = CuDNNHandle::get();
    CuDNNHandle::set_stream(stream);

    cudnnDataType_t cudnn_dtype;
    switch (input_c.dtype()) {
        case DType::Float32: cudnn_dtype = CUDNN_DATA_FLOAT; break;
        case DType::Float64: cudnn_dtype = CUDNN_DATA_DOUBLE; break;
        case DType::Float16: cudnn_dtype = CUDNN_DATA_HALF; break;
        case DType::BFloat16: cudnn_dtype = CUDNN_DATA_BFLOAT16; break;
        default:
            throw std::runtime_error("cuDNN AvgPool2d: unsupported dtype");
    }

    TensorDescriptor input_desc, output_desc;
    PoolingDescriptor pool_desc;

    input_desc.set(cudnn_dtype, batch, channels, height, width);
    output_desc.set(cudnn_dtype, batch, channels, out_h, out_w);
    pool_desc.set_avgpool(kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w, count_include_pad);

    // cuDNN requires alpha/beta type to match tensor dtype
    if (input_c.dtype() == DType::Float64) {
        double alpha = 1.0, beta = 0.0;
        CUDNN_CHECK(cudnnPoolingForward(
            handle,
            pool_desc.get(),
            &alpha,
            input_desc.get(),
            input_c.data_ptr(),
            &beta,
            output_desc.get(),
            output.data_ptr()
        ));
    } else {
        float alpha = 1.0f, beta = 0.0f;
        CUDNN_CHECK(cudnnPoolingForward(
            handle,
            pool_desc.get(),
            &alpha,
            input_desc.get(),
            input_c.data_ptr(),
            &beta,
            output_desc.get(),
            output.data_ptr()
        ));
    }

    return output;
}

// ============================================================================
// cuDNN AvgPool2d Backward Implementation
// ============================================================================

auto cudnn_avgpool2d_backward(
    const Tensor& grad_output,
    const Tensor& input,
    int64_t kernel_size,
    int64_t stride,
    int64_t padding,
    bool count_include_pad,
    cudaStream_t stream
) -> Tensor {
    return cudnn_avgpool2d_backward(
        grad_output, input,
        kernel_size, kernel_size,
        stride, stride,
        padding, padding,
        count_include_pad,
        stream);
}

auto cudnn_avgpool2d_backward(
    const Tensor& grad_output,
    const Tensor& input,
    int64_t kernel_h, int64_t kernel_w,
    int64_t stride_h, int64_t stride_w,
    int64_t pad_h, int64_t pad_w,
    bool count_include_pad,
    cudaStream_t stream
) -> Tensor {
    // Descriptors below assume packed NCHW; materialize contiguous copies.
    const Tensor& input_c = input.is_contiguous() ? input : input.contiguous();
    const Tensor& grad_output_c = grad_output.is_contiguous() ? grad_output : grad_output.contiguous();

    auto in_shape = input_c.shape();
    int64_t batch = in_shape[0];
    int64_t channels = in_shape[1];
    int64_t height = in_shape[2];
    int64_t width = in_shape[3];

    auto out_shape = grad_output_c.shape();
    int64_t out_h = out_shape[2];
    int64_t out_w = out_shape[3];

    Tensor grad_input({batch, channels, height, width}, input_c.dtype(), input_c.device());

    CudaDeviceGuard dev_guard(input_c.device().index);
    cudnnHandle_t handle = CuDNNHandle::get();
    CuDNNHandle::set_stream(stream);

    cudnnDataType_t cudnn_dtype;
    switch (input_c.dtype()) {
        case DType::Float32: cudnn_dtype = CUDNN_DATA_FLOAT; break;
        case DType::Float64: cudnn_dtype = CUDNN_DATA_DOUBLE; break;
        case DType::Float16: cudnn_dtype = CUDNN_DATA_HALF; break;
        case DType::BFloat16: cudnn_dtype = CUDNN_DATA_BFLOAT16; break;
        default:
            throw std::runtime_error("cuDNN AvgPool2d backward: unsupported dtype");
    }

    TensorDescriptor input_desc, output_desc;
    PoolingDescriptor pool_desc;

    input_desc.set(cudnn_dtype, batch, channels, height, width);
    output_desc.set(cudnn_dtype, batch, channels, out_h, out_w);
    pool_desc.set_avgpool(kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w, count_include_pad);

    // AvgPool backward doesn't need original output, but cuDNN API requires all params
    Tensor dummy_output({batch, channels, out_h, out_w}, input_c.dtype(), input_c.device());

    // cuDNN requires alpha/beta type to match tensor dtype
    if (input_c.dtype() == DType::Float64) {
        double alpha = 1.0, beta = 0.0;
        CUDNN_CHECK(cudnnPoolingBackward(
            handle,
            pool_desc.get(),
            &alpha,
            output_desc.get(),
            dummy_output.data_ptr(),
            output_desc.get(),
            grad_output_c.data_ptr(),
            input_desc.get(),
            input_c.data_ptr(),
            &beta,
            input_desc.get(),
            grad_input.data_ptr()
        ));
    } else {
        float alpha = 1.0f, beta = 0.0f;
        CUDNN_CHECK(cudnnPoolingBackward(
            handle,
            pool_desc.get(),
            &alpha,
            output_desc.get(),
            dummy_output.data_ptr(),
            output_desc.get(),
            grad_output_c.data_ptr(),
            input_desc.get(),
            input_c.data_ptr(),
            &beta,
            input_desc.get(),
            grad_input.data_ptr()
        ));
    }

    return grad_input;
}

// ============================================================================
// cuDNN Softmax Forward Implementation
// ============================================================================

auto cudnn_softmax_forward(
    const Tensor& input,
    int64_t dim,
    cudaStream_t stream
) -> Tensor {
    // cuDNN softmax works best with 4D tensors in NCHW format
    // We reshape the input to [N, C, 1, 1] where softmax is over C dimension
    auto shape = input.shape();
    int64_t ndim = shape.size();

    // Normalize dim
    if (dim < 0) dim += ndim;

    // cuDNN reads the buffer assuming a contiguous [outer, dim, inner] layout
    // (set on the descriptor below). A non-contiguous input (e.g. a transposed
    // view) would be read with the wrong strides — the stride-from-shape audit
    // bug. Materialise a contiguous copy so the descriptor matches the data.
    const Tensor input_c = input.contiguous();
    CudaDeviceGuard dev_guard(input.device().index);

    // Calculate sizes before and after the softmax dimension
    int64_t outer_size = 1;
    for (int64_t i = 0; i < dim; ++i) outer_size *= shape[i];

    int64_t inner_size = 1;
    for (int64_t i = dim + 1; i < ndim; ++i) inner_size *= shape[i];

    int64_t dim_size = shape[dim];

    Tensor output = Tensor(std::vector<int64_t>(shape.begin(), shape.end()), input_c.dtype(), input_c.device());

    cudnnHandle_t handle = CuDNNHandle::get();
    CuDNNHandle::set_stream(stream);

    cudnnDataType_t cudnn_dtype;
    switch (input_c.dtype()) {
        case DType::Float32: cudnn_dtype = CUDNN_DATA_FLOAT; break;
        case DType::Float64: cudnn_dtype = CUDNN_DATA_DOUBLE; break;
        case DType::Float16: cudnn_dtype = CUDNN_DATA_HALF; break;
        case DType::BFloat16: cudnn_dtype = CUDNN_DATA_BFLOAT16; break;
        default:
            throw std::runtime_error("cuDNN Softmax: unsupported dtype");
    }

    TensorDescriptor input_desc, output_desc;
    // Reshape as [outer_size, dim_size, inner_size, 1] for cuDNN
    input_desc.set(cudnn_dtype, outer_size, dim_size, inner_size, 1);
    output_desc.set(cudnn_dtype, outer_size, dim_size, inner_size, 1);

    // cuDNN requires alpha/beta type to match tensor dtype
    if (input_c.dtype() == DType::Float64) {
        double alpha = 1.0, beta = 0.0;
        CUDNN_CHECK(cudnnSoftmaxForward(
            handle,
            CUDNN_SOFTMAX_ACCURATE,
            CUDNN_SOFTMAX_MODE_CHANNEL,
            &alpha,
            input_desc.get(),
            input_c.data_ptr(),
            &beta,
            output_desc.get(),
            output.data_ptr()
        ));
    } else {
        float alpha = 1.0f, beta = 0.0f;
        CUDNN_CHECK(cudnnSoftmaxForward(
            handle,
            CUDNN_SOFTMAX_ACCURATE,
            CUDNN_SOFTMAX_MODE_CHANNEL,
            &alpha,
            input_desc.get(),
            input_c.data_ptr(),
            &beta,
            output_desc.get(),
            output.data_ptr()
        ));
    }

    return output;
}

// ============================================================================
// cuDNN Softmax Backward Implementation
// ============================================================================

auto cudnn_softmax_backward(
    const Tensor& grad_output,
    const Tensor& output,
    int64_t dim,
    cudaStream_t stream
) -> Tensor {
    auto shape = output.shape();
    int64_t ndim = shape.size();

    if (dim < 0) dim += ndim;

    // cuDNN reads both buffers with a contiguous [outer, dim, inner] layout;
    // non-contiguous inputs would be mis-read (stride-from-shape audit bug).
    const Tensor output_c = output.contiguous();
    const Tensor grad_output_c = grad_output.contiguous();
    CudaDeviceGuard dev_guard(output.device().index);

    int64_t outer_size = 1;
    for (int64_t i = 0; i < dim; ++i) outer_size *= shape[i];

    int64_t inner_size = 1;
    for (int64_t i = dim + 1; i < ndim; ++i) inner_size *= shape[i];

    int64_t dim_size = shape[dim];

    Tensor grad_input = Tensor(std::vector<int64_t>(shape.begin(), shape.end()), output_c.dtype(), output_c.device());

    cudnnHandle_t handle = CuDNNHandle::get();
    CuDNNHandle::set_stream(stream);

    cudnnDataType_t cudnn_dtype;
    switch (output_c.dtype()) {
        case DType::Float32: cudnn_dtype = CUDNN_DATA_FLOAT; break;
        case DType::Float64: cudnn_dtype = CUDNN_DATA_DOUBLE; break;
        case DType::Float16: cudnn_dtype = CUDNN_DATA_HALF; break;
        case DType::BFloat16: cudnn_dtype = CUDNN_DATA_BFLOAT16; break;
        default:
            throw std::runtime_error("cuDNN Softmax backward: unsupported dtype");
    }

    TensorDescriptor output_desc, grad_desc;
    output_desc.set(cudnn_dtype, outer_size, dim_size, inner_size, 1);
    grad_desc.set(cudnn_dtype, outer_size, dim_size, inner_size, 1);

    // cuDNN requires alpha/beta type to match tensor dtype
    if (output_c.dtype() == DType::Float64) {
        double alpha = 1.0, beta = 0.0;
        CUDNN_CHECK(cudnnSoftmaxBackward(
            handle,
            CUDNN_SOFTMAX_ACCURATE,
            CUDNN_SOFTMAX_MODE_CHANNEL,
            &alpha,
            output_desc.get(),
            output_c.data_ptr(),
            grad_desc.get(),
            grad_output_c.data_ptr(),
            &beta,
            grad_desc.get(),
            grad_input.data_ptr()
        ));
    } else {
        float alpha = 1.0f, beta = 0.0f;
        CUDNN_CHECK(cudnnSoftmaxBackward(
            handle,
            CUDNN_SOFTMAX_ACCURATE,
            CUDNN_SOFTMAX_MODE_CHANNEL,
            &alpha,
            output_desc.get(),
            output_c.data_ptr(),
            grad_desc.get(),
            grad_output_c.data_ptr(),
            &beta,
            grad_desc.get(),
            grad_input.data_ptr()
        ));
    }

    return grad_input;
}

// ============================================================================
// cuDNN Log-Softmax Forward Implementation
// ============================================================================

auto cudnn_log_softmax_forward(
    const Tensor& input,
    int64_t dim,
    cudaStream_t stream
) -> Tensor {
    auto shape = input.shape();
    int64_t ndim = shape.size();

    if (dim < 0) dim += ndim;

    // Contiguous copy so cuDNN's [outer, dim, inner] descriptor matches the
    // buffer layout (stride-from-shape audit bug for transposed views).
    const Tensor input_c = input.contiguous();
    CudaDeviceGuard dev_guard(input.device().index);

    int64_t outer_size = 1;
    for (int64_t i = 0; i < dim; ++i) outer_size *= shape[i];

    int64_t inner_size = 1;
    for (int64_t i = dim + 1; i < ndim; ++i) inner_size *= shape[i];

    int64_t dim_size = shape[dim];

    Tensor output = Tensor(std::vector<int64_t>(shape.begin(), shape.end()), input_c.dtype(), input_c.device());

    cudnnHandle_t handle = CuDNNHandle::get();
    CuDNNHandle::set_stream(stream);

    cudnnDataType_t cudnn_dtype;
    switch (input_c.dtype()) {
        case DType::Float32: cudnn_dtype = CUDNN_DATA_FLOAT; break;
        case DType::Float64: cudnn_dtype = CUDNN_DATA_DOUBLE; break;
        case DType::Float16: cudnn_dtype = CUDNN_DATA_HALF; break;
        case DType::BFloat16: cudnn_dtype = CUDNN_DATA_BFLOAT16; break;
        default:
            throw std::runtime_error("cuDNN LogSoftmax: unsupported dtype");
    }

    TensorDescriptor input_desc, output_desc;
    input_desc.set(cudnn_dtype, outer_size, dim_size, inner_size, 1);
    output_desc.set(cudnn_dtype, outer_size, dim_size, inner_size, 1);

    // cuDNN requires alpha/beta type to match tensor dtype
    if (input_c.dtype() == DType::Float64) {
        double alpha = 1.0, beta = 0.0;
        CUDNN_CHECK(cudnnSoftmaxForward(
            handle,
            CUDNN_SOFTMAX_LOG,
            CUDNN_SOFTMAX_MODE_CHANNEL,
            &alpha,
            input_desc.get(),
            input_c.data_ptr(),
            &beta,
            output_desc.get(),
            output.data_ptr()
        ));
    } else {
        float alpha = 1.0f, beta = 0.0f;
        CUDNN_CHECK(cudnnSoftmaxForward(
            handle,
            CUDNN_SOFTMAX_LOG,
            CUDNN_SOFTMAX_MODE_CHANNEL,
            &alpha,
            input_desc.get(),
            input_c.data_ptr(),
            &beta,
            output_desc.get(),
            output.data_ptr()
        ));
    }

    return output;
}

// ============================================================================
// cuDNN Log-Softmax Backward Implementation
// ============================================================================

auto cudnn_log_softmax_backward(
    const Tensor& grad_output,
    const Tensor& output,
    int64_t dim,
    cudaStream_t stream
) -> Tensor {
    auto shape = output.shape();
    int64_t ndim = shape.size();

    if (dim < 0) dim += ndim;

    int64_t outer_size = 1;
    for (int64_t i = 0; i < dim; ++i) outer_size *= shape[i];

    int64_t inner_size = 1;
    for (int64_t i = dim + 1; i < ndim; ++i) inner_size *= shape[i];

    int64_t dim_size = shape[dim];

    // Contiguous copies so cuDNN's [outer, dim, inner] descriptor matches the
    // buffer layout (stride-from-shape audit bug for transposed views).
    const Tensor output_c = output.contiguous();
    const Tensor grad_output_c = grad_output.contiguous();
    CudaDeviceGuard dev_guard(output.device().index);

    Tensor grad_input = Tensor(std::vector<int64_t>(shape.begin(), shape.end()), output_c.dtype(), output_c.device());

    cudnnHandle_t handle = CuDNNHandle::get();
    CuDNNHandle::set_stream(stream);

    cudnnDataType_t cudnn_dtype;
    switch (output_c.dtype()) {
        case DType::Float32: cudnn_dtype = CUDNN_DATA_FLOAT; break;
        case DType::Float64: cudnn_dtype = CUDNN_DATA_DOUBLE; break;
        case DType::Float16: cudnn_dtype = CUDNN_DATA_HALF; break;
        case DType::BFloat16: cudnn_dtype = CUDNN_DATA_BFLOAT16; break;
        default:
            throw std::runtime_error("cuDNN LogSoftmax backward: unsupported dtype");
    }

    TensorDescriptor output_desc, grad_desc;
    output_desc.set(cudnn_dtype, outer_size, dim_size, inner_size, 1);
    grad_desc.set(cudnn_dtype, outer_size, dim_size, inner_size, 1);

    // cuDNN requires alpha/beta type to match tensor dtype
    if (output_c.dtype() == DType::Float64) {
        double alpha = 1.0, beta = 0.0;
        CUDNN_CHECK(cudnnSoftmaxBackward(
            handle,
            CUDNN_SOFTMAX_LOG,
            CUDNN_SOFTMAX_MODE_CHANNEL,
            &alpha,
            output_desc.get(),
            output_c.data_ptr(),
            grad_desc.get(),
            grad_output_c.data_ptr(),
            &beta,
            grad_desc.get(),
            grad_input.data_ptr()
        ));
    } else {
        float alpha = 1.0f, beta = 0.0f;
        CUDNN_CHECK(cudnnSoftmaxBackward(
            handle,
            CUDNN_SOFTMAX_LOG,
            CUDNN_SOFTMAX_MODE_CHANNEL,
            &alpha,
            output_desc.get(),
            output_c.data_ptr(),
            grad_desc.get(),
            grad_output_c.data_ptr(),
            &beta,
            grad_desc.get(),
            grad_input.data_ptr()
        ));
    }

    return grad_input;
}

// ============================================================================
// Optimized LayerNorm using cuDNN Normalization API
// ============================================================================

/**
 * @brief Warp-level sum reduction using shuffle intrinsics
 *
 * Much faster than shared memory reduction for small warps.
 * Uses __shfl_xor_sync for butterfly reduction pattern.
 */
__device__ __forceinline__ float warpReduceSum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

/**
 * @brief Block-level sum reduction using warp shuffles and shared memory
 *
 * First reduces within warps using shuffles, then across warps using shared memory.
 * This is 2-5x faster than pure shared memory reduction.
 */
template<int BLOCK_SIZE>
__device__ __forceinline__ float blockReduceSum(float val, float* shared) {
    const int lane = threadIdx.x % 32;
    const int wid = threadIdx.x / 32;

    // Warp-level reduction
    val = warpReduceSum(val);

    // Write reduced warp values to shared memory
    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();

    // Final reduction across warps (only first warp participates)
    constexpr int numWarps = BLOCK_SIZE / 32;
    val = (threadIdx.x < numWarps) ? shared[threadIdx.x] : 0.0f;
    if (wid == 0) {
        val = warpReduceSum(val);
    }

    return val;
}

// Forward-declared here (defined later in this file, alongside the Float64
// LayerNorm kernels that introduced it) so optimized_layer_norm_kernel below
// can also accumulate its Float32 mean/variance reduction in double --
// matches the non-cuDNN-build fused_layer_norm_kernel (fused_ops.cu), whose
// `Acc = double` accumulator this build-flag-dependent path was silently
// missing.
template<int BLOCK_SIZE>
__device__ __forceinline__ double blockReduceSumDouble(double val, double* shared);

/**
 * @brief Optimized LayerNorm forward kernel with warp shuffles and vectorized loads
 *
 * Key optimizations:
 * 1. Warp shuffle reductions (5x faster than shared memory)
 * 2. Vectorized float4 loads (4x memory bandwidth)
 * 3. Welford online algorithm (single pass, numerically stable)
 * 4. Fused mean/variance computation
 *
 * Performance: ~2x faster than naive implementation
 */
/**
 * @brief Welford's online (mean, M2) accumulator kept in Float32 -- native
 * throughput, no double-precision math. Unlike a plain sum/sum-of-squares
 * accumulation, this tracks deviations from a running mean rather than
 * squares of the raw values, so it stays numerically stable even when the
 * input mean is large relative to its variance (e.g. mean=1e6, std=1.0,
 * where sum-of-squares would need ~12 significant digits to resolve the
 * variance signal and Float32 only has ~7). Combining two partial
 * accumulators uses Chan's parallel-merge formula.
 */
struct WelfordAcc {
    float count;
    float mean;
    float m2;
};

__device__ __forceinline__ void welfordUpdate(WelfordAcc& acc, float x) {
    acc.count += 1.0f;
    float delta = x - acc.mean;
    acc.mean += delta / acc.count;
    float delta2 = x - acc.mean;
    acc.m2 += delta * delta2;
}

__device__ __forceinline__ WelfordAcc welfordCombine(WelfordAcc a, WelfordAcc b) {
    if (a.count == 0.0f) return b;
    if (b.count == 0.0f) return a;
    float count = a.count + b.count;
    float delta = b.mean - a.mean;
    float mean = a.mean + delta * b.count / count;
    float m2 = a.m2 + b.m2 + delta * delta * a.count * b.count / count;
    return WelfordAcc{count, mean, m2};
}

/**
 * @brief Block-level Welford reduction using warp shuffles + shared memory.
 *
 * Mirrors blockReduceSum's warp-then-shared-memory structure, but combines
 * (count, mean, M2) triples instead of scalars. Result is only valid on
 * thread 0 of the block.
 */
template<int BLOCK_SIZE>
__device__ __forceinline__ WelfordAcc blockReduceWelford(WelfordAcc val, WelfordAcc* shared) {
    const int lane = threadIdx.x % 32;
    const int wid = threadIdx.x / 32;

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        WelfordAcc other;
        other.count = __shfl_down_sync(0xffffffff, val.count, offset);
        other.mean = __shfl_down_sync(0xffffffff, val.mean, offset);
        other.m2 = __shfl_down_sync(0xffffffff, val.m2, offset);
        val = welfordCombine(val, other);
    }

    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();

    constexpr int numWarps = BLOCK_SIZE / 32;
    WelfordAcc result{0.0f, 0.0f, 0.0f};
    if (threadIdx.x == 0) {
        result = shared[0];
        #pragma unroll
        for (int i = 1; i < numWarps; ++i) {
            result = welfordCombine(result, shared[i]);
        }
    }
    return result;
}

template<int BLOCK_SIZE>
__global__ void optimized_layer_norm_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    float* __restrict__ mean_out,
    float* __restrict__ inv_std_out,
    int64_t batch_size,
    int64_t norm_size,
    float eps
) {
    const int64_t b = blockIdx.x;
    if (b >= batch_size) return;

    const float* batch_in = input + b * norm_size;
    float* batch_out = output + b * norm_size;

    // Single-pass Welford reduction, entirely in Float32 (native throughput,
    // no double-precision math). Deviation-based (never squares the raw
    // input), so it stays stable even for large-mean/small-variance rows --
    // see WelfordAcc's doc comment and LayerNormVarianceStability's
    // LargeMeanProducesValidVariance test (mean=1e6, std=1.0), which a plain
    // sum/sum-of-squares accumulation cannot pass at Float32 precision no
    // matter how carefully the summation itself is done.
    __shared__ WelfordAcc shared[BLOCK_SIZE / 32];

    // Use vectorized loads only when each per-row base is 16-byte aligned.
    // cudaMalloc guarantees 256-byte base alignment, but per-row bases are
    // offset by b*norm_size floats; that offset is a multiple of 16 bytes
    // (and float4 access is well-defined) only when norm_size % 4 == 0.
    // Otherwise vec_norm_size is forced to 0 so every element is processed
    // by the scalar remainder loops (matching the FP16/BF16/FP64 kernels).
    const int vec_size = 4;
    const int64_t vec_norm_size = (norm_size % vec_size == 0) ? (norm_size / vec_size) : 0;
    const int64_t remainder_start = vec_norm_size * vec_size;

    const float4* batch_in_vec = reinterpret_cast<const float4*>(batch_in);
    WelfordAcc local{0.0f, 0.0f, 0.0f};
    for (int64_t i = threadIdx.x; i < vec_norm_size; i += blockDim.x) {
        float4 v = batch_in_vec[i];
        welfordUpdate(local, v.x);
        welfordUpdate(local, v.y);
        welfordUpdate(local, v.z);
        welfordUpdate(local, v.w);
    }
    for (int64_t i = remainder_start + threadIdx.x; i < norm_size; i += blockDim.x) {
        welfordUpdate(local, batch_in[i]);
    }

    WelfordAcc block_result = blockReduceWelford<BLOCK_SIZE>(local, shared);

    __shared__ float mean_shared;
    __shared__ float inv_std_shared;
    if (threadIdx.x == 0) {
        float variance = block_result.m2 / static_cast<float>(norm_size);
        mean_shared = block_result.mean;
        inv_std_shared = 1.0f / sqrtf(variance + eps);
        mean_out[b] = mean_shared;
        inv_std_out[b] = inv_std_shared;
    }
    __syncthreads();
    float mean = mean_shared;
    float inv_std = inv_std_shared;

    // ===== Second pass: Normalize and apply affine transform =====
    // Vectorized writes
    float4* batch_out_vec = reinterpret_cast<float4*>(batch_out);
    const float4* weight_vec = reinterpret_cast<const float4*>(weight);
    const float4* bias_vec = reinterpret_cast<const float4*>(bias);

    for (int64_t i = threadIdx.x; i < vec_norm_size; i += blockDim.x) {
        float4 v = batch_in_vec[i];
        float4 w = weight_vec[i];
        float4 bb = bias_vec[i];

        float4 result;
        result.x = ((v.x - mean) * inv_std) * w.x + bb.x;
        result.y = ((v.y - mean) * inv_std) * w.y + bb.y;
        result.z = ((v.z - mean) * inv_std) * w.z + bb.z;
        result.w = ((v.w - mean) * inv_std) * w.w + bb.w;

        batch_out_vec[i] = result;
    }

    // Handle remainder
    for (int64_t i = remainder_start + threadIdx.x; i < norm_size; i += blockDim.x) {
        float normalized = (batch_in[i] - mean) * inv_std;
        batch_out[i] = normalized * weight[i] + bias[i];
    }
}

/**
 * @brief Optimized LayerNorm backward kernel with warp shuffles
 *
 * Computes:
 * - grad_input = inv_std * (grad_out * weight - mean(grad_out * weight)
 *                          - normalized * mean(grad_out * weight * normalized))
 * - grad_weight = sum(grad_out * normalized) over batch
 * - grad_bias = sum(grad_out) over batch
 */
template<int BLOCK_SIZE>
__global__ void optimized_layer_norm_backward_kernel(
    const float* __restrict__ grad_output,
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ mean,
    const float* __restrict__ inv_std,
    float* __restrict__ grad_input,
    float* __restrict__ grad_weight,
    float* __restrict__ grad_bias,
    int64_t batch_size,
    int64_t norm_size
) {
    const int64_t b = blockIdx.x;
    if (b >= batch_size) return;

    const float* batch_grad_out = grad_output + b * norm_size;
    const float* batch_in = input + b * norm_size;
    float* batch_grad_in = grad_input + b * norm_size;

    const float batch_mean = mean[b];
    const float batch_inv_std = inv_std[b];

    __shared__ float shared[BLOCK_SIZE / 32];

    // Compute two sums needed for gradient:
    // sum1 = sum(grad_out * weight)
    // sum2 = sum(grad_out * weight * normalized)
    float sum1 = 0.0f;
    float sum2 = 0.0f;

    for (int64_t i = threadIdx.x; i < norm_size; i += blockDim.x) {
        float grad_w = batch_grad_out[i] * weight[i];
        float normalized = (batch_in[i] - batch_mean) * batch_inv_std;
        sum1 += grad_w;
        sum2 += grad_w * normalized;
    }

    sum1 = blockReduceSum<BLOCK_SIZE>(sum1, shared);
    __syncthreads();
    sum2 = blockReduceSum<BLOCK_SIZE>(sum2, shared);

    // Broadcast sums
    if (threadIdx.x == 0) {
        shared[0] = sum1 / static_cast<float>(norm_size);
        shared[1] = sum2 / static_cast<float>(norm_size);
    }
    __syncthreads();
    const float mean_grad_w = shared[0];
    const float mean_grad_w_norm = shared[1];

    // Compute input gradients
    for (int64_t i = threadIdx.x; i < norm_size; i += blockDim.x) {
        float normalized = (batch_in[i] - batch_mean) * batch_inv_std;
        float grad_w = batch_grad_out[i] * weight[i];

        // grad_input = inv_std * (grad_w - mean_grad_w - normalized * mean_grad_w_norm)
        batch_grad_in[i] = batch_inv_std * (grad_w - mean_grad_w - normalized * mean_grad_w_norm);

        // grad_weight[i] / grad_bias[i] accumulate across batch instances (one
        // block per instance), so each thread atomic-adds ITS OWN feature i.
        // (The previous warp shfl_down reduced across lanes — which hold DIFFERENT
        // feature indices — and only lane 0 stored, so features handled by other
        // lanes got zero gradient, and for norm_size < 32 the shfl was undefined.)
        atomicAdd(&grad_weight[i], batch_grad_out[i] * normalized);
        atomicAdd(&grad_bias[i], batch_grad_out[i]);
    }
}

/**
 * @brief Mixed-precision LayerNorm forward kernel
 *
 * Reads InputT, accumulates in float, writes OutputT for the normalized
 * output but StatsT (always float — see M4) for mean/inv_std: rstd's
 * dynamic range exceeds FP16 max=65504 when var ~ 1e-11, so narrowing the
 * saved stats to the input's half-precision dtype (like the output) can
 * saturate to Inf and poison backward with NaN. Mirrors the non-cuDNN
 * fallback fused_layer_norm_cuda, which already keeps mean/inv_std at
 * Float32 for F16/BF16 input.
 * Eliminates full-tensor dtype conversion passes for FP16 inputs.
 */
template<int BLOCK_SIZE, typename InputT, typename OutputT, typename StatsT>
__global__ void layer_norm_mixed_kernel(
    const InputT* __restrict__ input,
    const InputT* __restrict__ weight,
    const InputT* __restrict__ bias,
    OutputT* __restrict__ output,
    StatsT* __restrict__ mean_out,
    StatsT* __restrict__ inv_std_out,
    int64_t batch_size,
    int64_t norm_size,
    float eps
) {
    const int64_t b = blockIdx.x;
    if (b >= batch_size) return;

    const InputT* batch_in = input + b * norm_size;
    OutputT* batch_out = output + b * norm_size;

    __shared__ float shared[BLOCK_SIZE / 32];

    // 7th-audit Fix #1: numerically stable two-pass variance.
    //
    // Pre-fix this kernel used `var = E[X^2] - E[X]^2` (computed as
    // `sum_sq / N - mean*mean`), which catastrophically cancels in Float32
    // when |mean|^2 ≈ E[X^2]. The classic trigger is FP16/BF16 LayerNorm on
    // a tensor with a large mean (pretrained embeddings, mean ~ 1e3+),
    // where the cancellation produces variance ≈ 0 → rsqrtf(eps) → output
    // overflow. Sibling site to the CPU forward fix (commit 2ee72b5b) and
    // the CPU backward fix shipped in the 5th-audit (A1 — normalization.cpp).
    //
    // Two-pass cost is one extra global-memory read per element per row
    // (we re-stream the input to accumulate squared deviations against
    // the freshly-reduced mean). Eliminates the catastrophic-cancellation
    // entirely.

    // ---- Pass 1: per-row mean ------------------------------------------
    float sum = 0.0f;
    for (int64_t i = threadIdx.x; i < norm_size; i += blockDim.x) {
        sum += static_cast<float>(batch_in[i]);
    }
    sum = blockReduceSum<BLOCK_SIZE>(sum, shared);
    __syncthreads();

    float mean, inv_std;
    if (threadIdx.x == 0) {
        mean = sum / static_cast<float>(norm_size);
        shared[0] = mean;  // publish to whole block
    }
    __syncthreads();
    mean = shared[0];

    // ---- Pass 2: per-row sum of squared deviations against the mean ----
    float sum_sq_dev = 0.0f;
    for (int64_t i = threadIdx.x; i < norm_size; i += blockDim.x) {
        const float d = static_cast<float>(batch_in[i]) - mean;
        sum_sq_dev += d * d;
    }
    sum_sq_dev = blockReduceSum<BLOCK_SIZE>(sum_sq_dev, shared);

    if (threadIdx.x == 0) {
        const float variance = sum_sq_dev / static_cast<float>(norm_size);
        inv_std = rsqrtf(variance + eps);
        mean_out[b] = static_cast<StatsT>(mean);
        inv_std_out[b] = static_cast<StatsT>(inv_std);
        shared[1] = inv_std;
    }
    __syncthreads();
    inv_std = shared[1];

    // Second pass: normalize and apply affine transform
    for (int64_t i = threadIdx.x; i < norm_size; i += blockDim.x) {
        float val = static_cast<float>(batch_in[i]);
        float normalized = (val - mean) * inv_std;
        float w = static_cast<float>(weight[i]);
        float bb = static_cast<float>(bias[i]);
        batch_out[i] = static_cast<OutputT>(normalized * w + bb);
    }
}

/**
 * @brief Mixed-precision LayerNorm backward kernel
 *
 * mean/inv_std are StatsT (always float — see M4 / the forward kernel's
 * doc comment): the forward saves them at Float32 regardless of InputT, so
 * the backward must read them back at that same width, not InputT's.
 */
template<int BLOCK_SIZE, typename InputT, typename OutputT, typename StatsT>
__global__ void layer_norm_backward_mixed_kernel(
    const InputT* __restrict__ grad_output,
    const InputT* __restrict__ input,
    const InputT* __restrict__ weight,
    const StatsT* __restrict__ mean,
    const StatsT* __restrict__ inv_std,
    OutputT* __restrict__ grad_input,
    float* __restrict__ grad_weight,
    float* __restrict__ grad_bias,
    int64_t batch_size,
    int64_t norm_size
) {
    const int64_t b = blockIdx.x;
    if (b >= batch_size) return;

    const InputT* batch_grad_out = grad_output + b * norm_size;
    const InputT* batch_in = input + b * norm_size;
    OutputT* batch_grad_in = grad_input + b * norm_size;

    const float batch_mean = static_cast<float>(mean[b]);
    const float batch_inv_std = static_cast<float>(inv_std[b]);

    __shared__ float shared[BLOCK_SIZE / 32];

    float sum1 = 0.0f;
    float sum2 = 0.0f;

    for (int64_t i = threadIdx.x; i < norm_size; i += blockDim.x) {
        float grad_w = static_cast<float>(batch_grad_out[i]) * static_cast<float>(weight[i]);
        float normalized = (static_cast<float>(batch_in[i]) - batch_mean) * batch_inv_std;
        sum1 += grad_w;
        sum2 += grad_w * normalized;
    }

    sum1 = blockReduceSum<BLOCK_SIZE>(sum1, shared);
    __syncthreads();
    sum2 = blockReduceSum<BLOCK_SIZE>(sum2, shared);

    if (threadIdx.x == 0) {
        shared[0] = sum1 / static_cast<float>(norm_size);
        shared[1] = sum2 / static_cast<float>(norm_size);
    }
    __syncthreads();
    const float mean_grad_w = shared[0];
    const float mean_grad_w_norm = shared[1];

    for (int64_t i = threadIdx.x; i < norm_size; i += blockDim.x) {
        float normalized = (static_cast<float>(batch_in[i]) - batch_mean) * batch_inv_std;
        float grad_w = static_cast<float>(batch_grad_out[i]) * static_cast<float>(weight[i]);

        batch_grad_in[i] = static_cast<OutputT>(
            batch_inv_std * (grad_w - mean_grad_w - normalized * mean_grad_w_norm));

        // grad_weight[i] / grad_bias[i] accumulate across batch instances (one
        // block per instance), so each thread atomic-adds ITS OWN feature i.
        // (The previous warp shfl_down reduced across lanes — which hold DIFFERENT
        // feature indices — and only lane 0 stored, so features handled by other
        // lanes got zero gradient, and for norm_size < 32 the shfl was undefined.)
        atomicAdd(&grad_weight[i], static_cast<float>(batch_grad_out[i]) * normalized);
        atomicAdd(&grad_bias[i], static_cast<float>(batch_grad_out[i]));
    }
}

/**
 * @brief Double-precision warp reduce sum
 */
__device__ __forceinline__ double warpReduceSumDouble(double val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

template<int BLOCK_SIZE>
__device__ __forceinline__ double blockReduceSumDouble(double val, double* shared) {
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    val = warpReduceSumDouble(val);

    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();

    constexpr int numWarps = BLOCK_SIZE / 32;
    val = (threadIdx.x < numWarps) ? shared[threadIdx.x] : 0.0;
    if (wid == 0) {
        val = warpReduceSumDouble(val);
    }

    return val;
}

/**
 * @brief Double-precision LayerNorm forward kernel
 */
template<int BLOCK_SIZE>
__global__ void layer_norm_fp64_kernel(
    const double* __restrict__ input,
    const double* __restrict__ weight,
    const double* __restrict__ bias,
    double* __restrict__ output,
    double* __restrict__ mean_out,
    double* __restrict__ inv_std_out,
    int64_t batch_size,
    int64_t norm_size,
    double eps
) {
    const int64_t b = blockIdx.x;
    if (b >= batch_size) return;

    const double* batch_in = input + b * norm_size;
    double* batch_out = output + b * norm_size;

    __shared__ double shared[BLOCK_SIZE / 32];

    // Pass 1: compute mean.
    double sum = 0.0;
    for (int64_t i = threadIdx.x; i < norm_size; i += blockDim.x) {
        sum += batch_in[i];
    }
    sum = blockReduceSumDouble<BLOCK_SIZE>(sum, shared);

    __shared__ double mean_shared;
    if (threadIdx.x == 0) {
        mean_shared = sum / static_cast<double>(norm_size);
    }
    __syncthreads();
    const double mean = mean_shared;

    // Pass 2: compute variance via the numerically stable Σ(x - mean)²
    // form (Welford-style). The previous one-pass `E[x²] - E[x]²` formula
    // suffers catastrophic cancellation whenever the mean's magnitude is
    // comparable to the standard deviation — for Float64 inputs around
    // N(0,1) this lost ~1e-7 of relative precision and broke parity with
    // backends that use the stable form (CPU oneDNN, ROCm).
    double var_sum = 0.0;
    for (int64_t i = threadIdx.x; i < norm_size; i += blockDim.x) {
        double diff = batch_in[i] - mean;
        var_sum += diff * diff;
    }
    var_sum = blockReduceSumDouble<BLOCK_SIZE>(var_sum, shared);

    __shared__ double inv_std_shared;
    if (threadIdx.x == 0) {
        double variance = var_sum / static_cast<double>(norm_size);
        inv_std_shared = rsqrt(variance + eps);
        mean_out[b] = mean;
        inv_std_out[b] = inv_std_shared;
    }
    __syncthreads();
    const double inv_std = inv_std_shared;

    for (int64_t i = threadIdx.x; i < norm_size; i += blockDim.x) {
        double normalized = (batch_in[i] - mean) * inv_std;
        batch_out[i] = normalized * weight[i] + bias[i];
    }
}

/**
 * @brief Double-precision LayerNorm backward kernel
 */
template<int BLOCK_SIZE>
__global__ void layer_norm_backward_fp64_kernel(
    const double* __restrict__ grad_output,
    const double* __restrict__ input,
    const double* __restrict__ weight,
    const double* __restrict__ mean,
    const double* __restrict__ inv_std,
    double* __restrict__ grad_input,
    double* __restrict__ grad_weight,
    double* __restrict__ grad_bias,
    int64_t batch_size,
    int64_t norm_size
) {
    const int64_t b = blockIdx.x;
    if (b >= batch_size) return;

    const double* batch_grad_out = grad_output + b * norm_size;
    const double* batch_in = input + b * norm_size;
    double* batch_grad_in = grad_input + b * norm_size;

    const double batch_mean = mean[b];
    const double batch_inv_std = inv_std[b];

    __shared__ double shared[BLOCK_SIZE / 32];

    double sum1 = 0.0;
    double sum2 = 0.0;

    for (int64_t i = threadIdx.x; i < norm_size; i += blockDim.x) {
        double grad_w = batch_grad_out[i] * weight[i];
        double normalized = (batch_in[i] - batch_mean) * batch_inv_std;
        sum1 += grad_w;
        sum2 += grad_w * normalized;
    }

    sum1 = blockReduceSumDouble<BLOCK_SIZE>(sum1, shared);
    __syncthreads();
    sum2 = blockReduceSumDouble<BLOCK_SIZE>(sum2, shared);

    if (threadIdx.x == 0) {
        shared[0] = sum1 / static_cast<double>(norm_size);
        shared[1] = sum2 / static_cast<double>(norm_size);
    }
    __syncthreads();
    const double mean_grad_w = shared[0];
    const double mean_grad_w_norm = shared[1];

    for (int64_t i = threadIdx.x; i < norm_size; i += blockDim.x) {
        double normalized = (batch_in[i] - batch_mean) * batch_inv_std;
        double grad_w = batch_grad_out[i] * weight[i];

        batch_grad_in[i] = batch_inv_std * (grad_w - mean_grad_w - normalized * mean_grad_w_norm);

        // grad_weight[i] / grad_bias[i] accumulate across batch instances (one
        // block per instance), so each thread atomic-adds ITS OWN feature i.
        // (The previous warp shfl_down reduced across lanes — which hold DIFFERENT
        // feature indices — and only lane 0 stored, so features handled by other
        // lanes got zero gradient, and for norm_size < 32 the shfl was undefined.)
        atomicAdd(&grad_weight[i], batch_grad_out[i] * normalized);
        atomicAdd(&grad_bias[i], batch_grad_out[i]);
    }
}

/**
 * @brief Host wrapper for optimized LayerNorm forward
 *
 * Uses warp shuffle reductions and vectorized memory access for
 * significant performance improvement over naive implementation.
 */
auto cudnn_layer_norm_forward(
    const Tensor& input,
    const std::vector<int64_t>& normalized_shape,
    const Tensor& weight,
    const Tensor& bias,
    float eps,
    cudaStream_t stream
) -> std::tuple<Tensor, Tensor, Tensor> {
    // Calculate norm_size from normalized_shape
    int64_t norm_size = 1;
    for (auto dim : normalized_shape) {
        norm_size *= dim;
    }

    int64_t batch_size = input.numel() / norm_size;

    // These kernels launch one block per normalized row (block index =
    // blockIdx.x) with no grid-stride loop, so the grid dim must fit in the
    // int gridDim.x. Reject row counts that would overflow int rather than
    // silently wrapping to a negative/garbage launch dim and leaving the
    // tail of the output unnormalized.
    if (batch_size > static_cast<int64_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error(
            "cudnn_layer_norm_forward: normalized row count exceeds INT_MAX");
    }

    // Create output tensors
    auto shape = input.shape();
    Tensor output(std::vector<int64_t>(shape.begin(), shape.end()), input.dtype(), input.device());
    // M4: mean/inv_std are always saved at Float32, even for F16/BF16 input —
    // narrowing them to the input's half-precision dtype risks rstd
    // overflowing FP16's max (65504) when variance is tiny, saturating to
    // Inf and poisoning backward with NaN. See layer_norm_mixed_kernel's doc
    // comment; matches the non-cuDNN fused_layer_norm_cuda fallback.
    const bool narrow_stats =
        (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16);
    const DType stats_dtype = narrow_stats ? DType::Float32 : input.dtype();
    Tensor mean_tensor({batch_size}, stats_dtype, input.device());
    Tensor inv_std_tensor({batch_size}, stats_dtype, input.device());

    // Ensure tensors are contiguous
    Tensor input_c = input.is_contiguous() ? input : input.contiguous();
    Tensor weight_c = weight.is_contiguous() ? weight : weight.contiguous();
    Tensor bias_c = bias.is_contiguous() ? bias : bias.contiguous();

    CudaDeviceGuard dev_guard(input.device().index);
    cudnnHandle_t handle = CuDNNHandle::get();
    if (stream) {
        CuDNNHandle::set_stream(stream);
    }

    if (input_c.dtype() == DType::Float32) {
        // Choose optimal block size based on norm_size
        constexpr int BLOCK_SIZE = 256;
        int blocks = static_cast<int>(batch_size);

        optimized_layer_norm_kernel<BLOCK_SIZE><<<blocks, BLOCK_SIZE, 0, stream>>>(
            input_c.data<float>(),
            weight_c.data<float>(),
            bias_c.data<float>(),
            output.data<float>(),
            mean_tensor.data<float>(),
            inv_std_tensor.data<float>(),
            batch_size,
            norm_size,
            eps
        );
    } else if (input_c.dtype() == DType::Float64) {
        // Float64: compute directly in double precision — no conversion needed
        constexpr int BLOCK_SIZE = 256;
        int blocks = static_cast<int>(batch_size);

        layer_norm_fp64_kernel<BLOCK_SIZE><<<blocks, BLOCK_SIZE, 0, stream>>>(
            input_c.data<double>(),
            weight_c.data<double>(),
            bias_c.data<double>(),
            output.data<double>(),
            mean_tensor.data<double>(),
            inv_std_tensor.data<double>(),
            batch_size,
            norm_size,
            static_cast<double>(eps)
        );
    } else if (input_c.dtype() == DType::Float16) {
        // Float16: read __half, accumulate in float, write __half — no full-tensor conversion
        constexpr int BLOCK_SIZE = 256;
        int blocks = static_cast<int>(batch_size);

        layer_norm_mixed_kernel<BLOCK_SIZE, __half, __half, float><<<blocks, BLOCK_SIZE, 0, stream>>>(
            reinterpret_cast<const __half*>(input_c.data_ptr()),
            reinterpret_cast<const __half*>(weight_c.data_ptr()),
            reinterpret_cast<const __half*>(bias_c.data_ptr()),
            reinterpret_cast<__half*>(output.data_ptr()),
            mean_tensor.data<float>(),
            inv_std_tensor.data<float>(),
            batch_size,
            norm_size,
            eps
        );
    } else if (input_c.dtype() == DType::BFloat16) {
        // BFloat16: read __nv_bfloat16, accumulate in float, write __nv_bfloat16
        constexpr int BLOCK_SIZE = 256;
        int blocks = static_cast<int>(batch_size);

        layer_norm_mixed_kernel<BLOCK_SIZE, __nv_bfloat16, __nv_bfloat16, float><<<blocks, BLOCK_SIZE, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(input_c.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(weight_c.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(bias_c.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
            mean_tensor.data<float>(),
            inv_std_tensor.data<float>(),
            batch_size,
            norm_size,
            eps
        );
    } else {
        throw std::runtime_error("cudnn_layer_norm_forward: unsupported dtype");
    }
    CUDA_CHECK(cudaGetLastError());

    return {output, mean_tensor, inv_std_tensor};
}

/**
 * @brief Host wrapper for optimized LayerNorm backward
 */
auto cudnn_layer_norm_backward(
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& weight,
    const Tensor& mean,
    const Tensor& inv_std,
    const std::vector<int64_t>& normalized_shape,
    cudaStream_t stream
) -> std::tuple<Tensor, Tensor, Tensor> {
    int64_t norm_size = 1;
    for (auto dim : normalized_shape) {
        norm_size *= dim;
    }

    int64_t batch_size = input.numel() / norm_size;

    // One block per normalized row, no grid-stride loop — see the matching
    // guard in cudnn_layer_norm_forward.
    if (batch_size > static_cast<int64_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error(
            "cudnn_layer_norm_backward: normalized row count exceeds INT_MAX");
    }

    // Create gradient tensors
    auto shape = input.shape();
    Tensor grad_input(std::vector<int64_t>(shape.begin(), shape.end()), input.dtype(), input.device());
    Tensor grad_weight({norm_size}, weight.dtype(), weight.device());
    Tensor grad_bias({norm_size}, weight.dtype(), weight.device());

    // Zero initialize gradient accumulation tensors
    size_t elem_size = dtype_size(weight.dtype());
    cudaMemsetAsync(grad_weight.data_ptr(), 0, grad_weight.numel() * elem_size, stream);
    cudaMemsetAsync(grad_bias.data_ptr(), 0, grad_bias.numel() * elem_size, stream);

    // Ensure tensors are contiguous
    Tensor grad_out_c = grad_output.is_contiguous() ? grad_output : grad_output.contiguous();
    Tensor input_c = input.is_contiguous() ? input : input.contiguous();
    Tensor weight_c = weight.is_contiguous() ? weight : weight.contiguous();
    Tensor mean_c = mean.is_contiguous() ? mean : mean.contiguous();
    Tensor inv_std_c = inv_std.is_contiguous() ? inv_std : inv_std.contiguous();

    if (input_c.dtype() == DType::Float32) {
        constexpr int BLOCK_SIZE = 256;
        int blocks = static_cast<int>(batch_size);

        optimized_layer_norm_backward_kernel<BLOCK_SIZE><<<blocks, BLOCK_SIZE, 0, stream>>>(
            grad_out_c.data<float>(),
            input_c.data<float>(),
            weight_c.data<float>(),
            mean_c.data<float>(),
            inv_std_c.data<float>(),
            grad_input.data<float>(),
            grad_weight.data<float>(),
            grad_bias.data<float>(),
            batch_size,
            norm_size
        );
    } else if (input_c.dtype() == DType::Float64) {
        // Float64: compute directly in double precision
        constexpr int BLOCK_SIZE = 256;
        int blocks = static_cast<int>(batch_size);

        Tensor grad_weight_f64({norm_size}, DType::Float64, weight.device());
        Tensor grad_bias_f64({norm_size}, DType::Float64, weight.device());
        cudaMemsetAsync(grad_weight_f64.data_ptr(), 0, grad_weight_f64.numel() * sizeof(double), stream);
        cudaMemsetAsync(grad_bias_f64.data_ptr(), 0, grad_bias_f64.numel() * sizeof(double), stream);

        layer_norm_backward_fp64_kernel<BLOCK_SIZE><<<blocks, BLOCK_SIZE, 0, stream>>>(
            grad_out_c.data<double>(),
            input_c.data<double>(),
            weight_c.data<double>(),
            mean_c.data<double>(),
            inv_std_c.data<double>(),
            grad_input.data<double>(),
            grad_weight_f64.data<double>(),
            grad_bias_f64.data<double>(),
            batch_size,
            norm_size
        );

        grad_weight = grad_weight_f64;
        grad_bias = grad_bias_f64;
    } else if (input_c.dtype() == DType::Float16) {
        // Float16: read __half, accumulate grad_weight/grad_bias in float, write __half grad_input
        constexpr int BLOCK_SIZE = 256;
        int blocks = static_cast<int>(batch_size);

        // grad_weight and grad_bias accumulate atomically in float for precision
        Tensor grad_weight_f32({norm_size}, DType::Float32, weight.device());
        Tensor grad_bias_f32({norm_size}, DType::Float32, weight.device());
        cudaMemsetAsync(grad_weight_f32.data_ptr(), 0, grad_weight_f32.numel() * sizeof(float), stream);
        cudaMemsetAsync(grad_bias_f32.data_ptr(), 0, grad_bias_f32.numel() * sizeof(float), stream);

        // M4: mean/inv_std were saved at Float32 by the forward (see
        // layer_norm_mixed_kernel's doc comment) regardless of input's
        // dtype — read them back at that same width, not as __half.
        layer_norm_backward_mixed_kernel<BLOCK_SIZE, __half, __half, float><<<blocks, BLOCK_SIZE, 0, stream>>>(
            reinterpret_cast<const __half*>(grad_out_c.data_ptr()),
            reinterpret_cast<const __half*>(input_c.data_ptr()),
            reinterpret_cast<const __half*>(weight_c.data_ptr()),
            mean_c.data<float>(),
            inv_std_c.data<float>(),
            reinterpret_cast<__half*>(grad_input.data_ptr()),
            grad_weight_f32.data<float>(),
            grad_bias_f32.data<float>(),
            batch_size,
            norm_size
        );

        grad_weight = grad_weight_f32.to(DType::Float16);
        grad_bias = grad_bias_f32.to(DType::Float16);
    } else if (input_c.dtype() == DType::BFloat16) {
        // BFloat16: read __nv_bfloat16, accumulate grad_weight/grad_bias in float, write __nv_bfloat16 grad_input
        constexpr int BLOCK_SIZE = 256;
        int blocks = static_cast<int>(batch_size);

        // grad_weight and grad_bias accumulate atomically in float for precision
        Tensor grad_weight_f32({norm_size}, DType::Float32, weight.device());
        Tensor grad_bias_f32({norm_size}, DType::Float32, weight.device());
        cudaMemsetAsync(grad_weight_f32.data_ptr(), 0, grad_weight_f32.numel() * sizeof(float), stream);
        cudaMemsetAsync(grad_bias_f32.data_ptr(), 0, grad_bias_f32.numel() * sizeof(float), stream);

        // M4: mean/inv_std were saved at Float32 by the forward — see the
        // __half branch above.
        layer_norm_backward_mixed_kernel<BLOCK_SIZE, __nv_bfloat16, __nv_bfloat16, float><<<blocks, BLOCK_SIZE, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(grad_out_c.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(input_c.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(weight_c.data_ptr()),
            mean_c.data<float>(),
            inv_std_c.data<float>(),
            reinterpret_cast<__nv_bfloat16*>(grad_input.data_ptr()),
            grad_weight_f32.data<float>(),
            grad_bias_f32.data<float>(),
            batch_size,
            norm_size
        );

        grad_weight = grad_weight_f32.to(DType::BFloat16);
        grad_bias = grad_bias_f32.to(DType::BFloat16);
    } else {
        throw std::runtime_error("cudnn_layer_norm_backward: unsupported dtype");
    }
    CUDA_CHECK(cudaGetLastError());

    return {grad_input, grad_weight, grad_bias};
}

// ============================================================================
// cuDNN Conv3d Forward Implementation
// ============================================================================

static cudnnDataType_t to_cudnn_dtype(DType dtype) {
    switch (dtype) {
        case DType::Float32: return CUDNN_DATA_FLOAT;
        case DType::Float64: return CUDNN_DATA_DOUBLE;
        case DType::Float16: return CUDNN_DATA_HALF;
        case DType::BFloat16: return CUDNN_DATA_BFLOAT16;
        default:
            throw std::runtime_error("cuDNN Conv3d: unsupported dtype");
    }
}

// Range-check an int64 tensor dimension before narrowing to int for the cuDNN
// Nd descriptor API (which takes int* arrays). The 4D TensorDescriptor path
// routes every dim through TensorDescriptor::checked_dim(); the Nd path used
// bare (int) casts that would silently truncate a > INT_MAX extent into a
// wrong/negative descriptor. Mirror that guard here.
static int checked_nd_dim(int64_t v, const char* name) {
    if (v < 0 || v > static_cast<int64_t>(std::numeric_limits<int>::max())) {
        throw std::overflow_error(
            std::string("cuDNN Nd conv: dimension '") + name + "' = " +
            std::to_string(v) + " does not fit in int (cuDNN Nd descriptor limit)");
    }
    return static_cast<int>(v);
}

// Helper: dispatch cudnnConvolutionForward for the correct dtype
static void dispatch_conv_forward(
    cudnnHandle_t handle,
    const cudnnTensorDescriptor_t& input_desc, const void* input_ptr,
    const cudnnFilterDescriptor_t& filter_desc, const void* weight_ptr,
    const cudnnConvolutionDescriptor_t& conv_desc,
    cudnnConvolutionFwdAlgo_t algo, void* workspace, size_t workspace_size,
    const cudnnTensorDescriptor_t& output_desc, void* output_ptr,
    DType dtype
) {
    if (dtype == DType::Float64) {
        const double alpha = 1.0, beta = 0.0;
        CUDNN_CHECK(cudnnConvolutionForward(
            handle, &alpha, input_desc, input_ptr, filter_desc, weight_ptr,
            conv_desc, algo, workspace, workspace_size, &beta, output_desc, output_ptr));
    } else {
        const float alpha = 1.0f, beta = 0.0f;
        CUDNN_CHECK(cudnnConvolutionForward(
            handle, &alpha, input_desc, input_ptr, filter_desc, weight_ptr,
            conv_desc, algo, workspace, workspace_size, &beta, output_desc, output_ptr));
    }
}

// Helper: dispatch cudnnConvolutionBackwardData for the correct dtype
static void dispatch_conv_bwd_data(
    cudnnHandle_t handle,
    const cudnnFilterDescriptor_t& filter_desc, const void* weight_ptr,
    const cudnnTensorDescriptor_t& grad_output_desc, const void* grad_output_ptr,
    const cudnnConvolutionDescriptor_t& conv_desc,
    cudnnConvolutionBwdDataAlgo_t algo, void* workspace, size_t workspace_size,
    const cudnnTensorDescriptor_t& input_desc, void* grad_input_ptr,
    DType dtype
) {
    if (dtype == DType::Float64) {
        const double alpha = 1.0, beta = 0.0;
        CUDNN_CHECK(cudnnConvolutionBackwardData(
            handle, &alpha, filter_desc, weight_ptr, grad_output_desc, grad_output_ptr,
            conv_desc, algo, workspace, workspace_size, &beta, input_desc, grad_input_ptr));
    } else {
        const float alpha = 1.0f, beta = 0.0f;
        CUDNN_CHECK(cudnnConvolutionBackwardData(
            handle, &alpha, filter_desc, weight_ptr, grad_output_desc, grad_output_ptr,
            conv_desc, algo, workspace, workspace_size, &beta, input_desc, grad_input_ptr));
    }
}

// Helper: dispatch cudnnConvolutionBackwardFilter for the correct dtype
static void dispatch_conv_bwd_filter(
    cudnnHandle_t handle,
    const cudnnTensorDescriptor_t& input_desc, const void* input_ptr,
    const cudnnTensorDescriptor_t& grad_output_desc, const void* grad_output_ptr,
    const cudnnConvolutionDescriptor_t& conv_desc,
    cudnnConvolutionBwdFilterAlgo_t algo, void* workspace, size_t workspace_size,
    const cudnnFilterDescriptor_t& filter_desc, void* grad_weight_ptr,
    DType dtype
) {
    if (dtype == DType::Float64) {
        const double alpha = 1.0, beta = 0.0;
        CUDNN_CHECK(cudnnConvolutionBackwardFilter(
            handle, &alpha, input_desc, input_ptr, grad_output_desc, grad_output_ptr,
            conv_desc, algo, workspace, workspace_size, &beta, filter_desc, grad_weight_ptr));
    } else {
        const float alpha = 1.0f, beta = 0.0f;
        CUDNN_CHECK(cudnnConvolutionBackwardFilter(
            handle, &alpha, input_desc, input_ptr, grad_output_desc, grad_output_ptr,
            conv_desc, algo, workspace, workspace_size, &beta, filter_desc, grad_weight_ptr));
    }
}

// Helper: dispatch cudnnConvolutionBackwardBias for the correct dtype
static void dispatch_conv_bwd_bias(
    cudnnHandle_t handle,
    const cudnnTensorDescriptor_t& grad_output_desc, const void* grad_output_ptr,
    const cudnnTensorDescriptor_t& bias_desc, void* grad_bias_ptr,
    DType dtype
) {
    if (dtype == DType::Float64) {
        const double alpha = 1.0, beta = 0.0;
        CUDNN_CHECK(cudnnConvolutionBackwardBias(
            handle, &alpha, grad_output_desc, grad_output_ptr,
            &beta, bias_desc, grad_bias_ptr));
    } else {
        const float alpha = 1.0f, beta = 0.0f;
        CUDNN_CHECK(cudnnConvolutionBackwardBias(
            handle, &alpha, grad_output_desc, grad_output_ptr,
            &beta, bias_desc, grad_bias_ptr));
    }
}

// Helper: dispatch cudnnAddTensor for bias addition
static void dispatch_add_tensor(
    cudnnHandle_t handle,
    const cudnnTensorDescriptor_t& bias_desc, const void* bias_ptr,
    const cudnnTensorDescriptor_t& output_desc, void* output_ptr,
    DType dtype
) {
    if (dtype == DType::Float64) {
        const double alpha = 1.0, beta = 1.0;
        CUDNN_CHECK(cudnnAddTensor(handle, &alpha, bias_desc, bias_ptr, &beta, output_desc, output_ptr));
    } else {
        const float alpha = 1.0f, beta = 1.0f;
        CUDNN_CHECK(cudnnAddTensor(handle, &alpha, bias_desc, bias_ptr, &beta, output_desc, output_ptr));
    }
}

// RAII wrapper for cudnnTensorDescriptor_t (Nd)
struct TensorDescriptorNd {
    cudnnTensorDescriptor_t desc = nullptr;
    TensorDescriptorNd() { CUDNN_CHECK(cudnnCreateTensorDescriptor(&desc)); }
    ~TensorDescriptorNd() { if (desc) cudnnDestroyTensorDescriptor(desc); }
    TensorDescriptorNd(const TensorDescriptorNd&) = delete;
    TensorDescriptorNd& operator=(const TensorDescriptorNd&) = delete;

    void set(cudnnDataType_t dtype, const std::vector<int>& dims) {
        int nbDims = static_cast<int>(dims.size());
        // Compute strides for contiguous layout
        std::vector<int> strides(nbDims);
        strides[nbDims - 1] = 1;
        for (int i = nbDims - 2; i >= 0; --i) {
            strides[i] = strides[i + 1] * dims[i + 1];
        }
        CUDNN_CHECK(cudnnSetTensorNdDescriptor(desc, dtype, nbDims, dims.data(), strides.data()));
    }
};

// RAII wrapper for cudnnFilterDescriptor_t (Nd)
struct FilterDescriptorNd {
    cudnnFilterDescriptor_t desc = nullptr;
    FilterDescriptorNd() { CUDNN_CHECK(cudnnCreateFilterDescriptor(&desc)); }
    ~FilterDescriptorNd() { if (desc) cudnnDestroyFilterDescriptor(desc); }
    FilterDescriptorNd(const FilterDescriptorNd&) = delete;
    FilterDescriptorNd& operator=(const FilterDescriptorNd&) = delete;

    void set(cudnnDataType_t dtype, const std::vector<int>& dims) {
        int nbDims = static_cast<int>(dims.size());
        CUDNN_CHECK(cudnnSetFilterNdDescriptor(desc, dtype, CUDNN_TENSOR_NCHW, nbDims, dims.data()));
    }
};

// RAII wrapper for cudnnConvolutionDescriptor_t (Nd)
struct ConvolutionDescriptorNd {
    cudnnConvolutionDescriptor_t desc = nullptr;
    ConvolutionDescriptorNd() { CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&desc)); }
    ~ConvolutionDescriptorNd() { if (desc) cudnnDestroyConvolutionDescriptor(desc); }
    ConvolutionDescriptorNd(const ConvolutionDescriptorNd&) = delete;
    ConvolutionDescriptorNd& operator=(const ConvolutionDescriptorNd&) = delete;

    void set(int spatial_dims, const int* padding, const int* stride, const int* dilation,
             cudnnConvolutionMode_t mode, cudnnDataType_t compute_type) {
        CUDNN_CHECK(cudnnSetConvolutionNdDescriptor(
            desc, spatial_dims, padding, stride, dilation, mode, compute_type));
    }

    void set_group_count(int groups) {
        CUDNN_CHECK(cudnnSetConvolutionGroupCount(desc, groups));
    }
};

// audit L.6: RAII wrapper that owns BOTH the cudnnConvolutionDescriptor_t and
// the std::array<int, 3> storage for padding/stride/dilation. The raw int*
// pointers handed to cudnnSetConvolutionNdDescriptor() previously aliased
// lambda-local stack arrays declared at the call site (pad_arr/str_arr/dil_arr).
// A refactor that moved descriptor setup into a helper without forwarding
// those arrays would silently leave the descriptor pointing at freed stack
// memory, with no compile-time diagnostic. Wrapping the arrays here ties
// their lifetime to the descriptor itself: as long as the CudnnConvNdDesc
// object is alive, the cuDNN descriptor remains valid.
//
// The constructor narrows int64_t -> int with range validation so a
// pathological input (negative dilation, INT_MAX-sized padding) throws a
// clean std::runtime_error instead of producing a silently truncated
// descriptor that cuDNN then misinterprets.
struct CudnnConvNdDesc {
    cudnnConvolutionDescriptor_t desc = nullptr;
    std::array<int, 3> padding{};
    std::array<int, 3> stride{};
    std::array<int, 3> dilation{};

    CudnnConvNdDesc(const std::array<int64_t, 3>& pad_in,
                    const std::array<int64_t, 3>& str_in,
                    const std::array<int64_t, 3>& dil_in,
                    cudnnConvolutionMode_t mode,
                    cudnnDataType_t compute_type) {
        auto validate_narrow = [](int64_t v, const char* field, int axis) -> int {
            if (v < 0) {
                throw std::runtime_error(
                    std::string("CudnnConvNdDesc: negative ") + field
                    + " at axis " + std::to_string(axis)
                    + " (value=" + std::to_string(v) + ")");
            }
            if (v > static_cast<int64_t>(std::numeric_limits<int>::max())) {
                throw std::runtime_error(
                    std::string("CudnnConvNdDesc: ") + field
                    + " at axis " + std::to_string(axis)
                    + " exceeds INT_MAX (value=" + std::to_string(v) + ")");
            }
            return static_cast<int>(v);
        };
        // Stride and dilation must be >= 1 — cuDNN treats 0 as undefined
        // behaviour and silently produces garbage output strides.
        auto validate_positive = [&](int64_t v, const char* field, int axis) -> int {
            int narrowed = validate_narrow(v, field, axis);
            if (narrowed < 1) {
                throw std::runtime_error(
                    std::string("CudnnConvNdDesc: ") + field
                    + " at axis " + std::to_string(axis)
                    + " must be >= 1 (value=" + std::to_string(v) + ")");
            }
            return narrowed;
        };
        for (int i = 0; i < 3; ++i) {
            padding[i]  = validate_narrow(pad_in[i],  "padding",  i);
            stride[i]   = validate_positive(str_in[i], "stride",  i);
            dilation[i] = validate_positive(dil_in[i], "dilation", i);
        }
        CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&desc));
        CUDNN_CHECK(cudnnSetConvolutionNdDescriptor(
            desc, /*arrayLength=*/3,
            padding.data(), stride.data(), dilation.data(),
            mode, compute_type));
    }

    ~CudnnConvNdDesc() {
        if (desc) cudnnDestroyConvolutionDescriptor(desc);
    }

    CudnnConvNdDesc(const CudnnConvNdDesc&) = delete;
    CudnnConvNdDesc& operator=(const CudnnConvNdDesc&) = delete;

    // Accessor — the returned descriptor's int* arrays are guaranteed valid
    // for the lifetime of this CudnnConvNdDesc instance.
    cudnnConvolutionDescriptor_t handle() const { return desc; }

    void set_group_count(int groups) {
        CUDNN_CHECK(cudnnSetConvolutionGroupCount(desc, groups));
    }
};

// ============================================================================
// Conv3d forward algorithm cache
//
// Mirrors Conv2dAlgoCache (include/tenzor/backend/cudnn_wrapper.hpp) but keyed
// on the full 5D (N,C,D,H,W) problem with per-axis D/H/W stride, padding and
// dilation. Without this cache cudnn_conv3d_forward called
// cudnnGetConvolutionForwardAlgorithm_v7 fresh on every call and picked the
// lowest heuristic-estimated-time algo. Heuristic ties reorder between runs,
// so for FP32 a TF32/Winograd algo could be chosen on some runs and diverge
// past the parity tolerance — the source of the intermittent
// NNConvParity.Conv3d_Basic/cuda failure. A shape+dtype-keyed cache makes the
// same problem always resolve to the same algorithm.
//
// As in Conv2dCacheKey, prefer_precise_f32 is part of the key: a Winograd algo
// cached when TF32 was allowed must not be reused after TF32 is disabled, and
// vice versa.
namespace {

struct Conv3dCacheKey {
    int64_t batch;
    int64_t in_channels;
    int64_t depth;
    int64_t height;
    int64_t width;
    int64_t out_channels;
    int64_t kernel_d;
    int64_t kernel_h;
    int64_t kernel_w;
    int64_t stride_d;
    int64_t stride_h;
    int64_t stride_w;
    int64_t pad_d;
    int64_t pad_h;
    int64_t pad_w;
    int64_t dil_d;
    int64_t dil_h;
    int64_t dil_w;
    int64_t groups;
    cudnnDataType_t dtype;
    bool prefer_precise_f32 = false;

    bool operator==(const Conv3dCacheKey& o) const {
        return batch == o.batch && in_channels == o.in_channels &&
               depth == o.depth && height == o.height && width == o.width &&
               out_channels == o.out_channels &&
               kernel_d == o.kernel_d && kernel_h == o.kernel_h && kernel_w == o.kernel_w &&
               stride_d == o.stride_d && stride_h == o.stride_h && stride_w == o.stride_w &&
               pad_d == o.pad_d && pad_h == o.pad_h && pad_w == o.pad_w &&
               dil_d == o.dil_d && dil_h == o.dil_h && dil_w == o.dil_w &&
               groups == o.groups && dtype == o.dtype &&
               prefer_precise_f32 == o.prefer_precise_f32;
    }
};

struct Conv3dCacheKeyHash {
    size_t operator()(const Conv3dCacheKey& k) const {
        size_t h = 0;
        auto hash_combine = [&h](auto val) {
            h ^= std::hash<decltype(val)>{}(val) + 0x9e3779b9 + (h << 6) + (h >> 2);
        };
        hash_combine(k.batch);
        hash_combine(k.in_channels);
        hash_combine(k.depth);
        hash_combine(k.height);
        hash_combine(k.width);
        hash_combine(k.out_channels);
        hash_combine(k.kernel_d);
        hash_combine(k.kernel_h);
        hash_combine(k.kernel_w);
        hash_combine(k.stride_d);
        hash_combine(k.stride_h);
        hash_combine(k.stride_w);
        hash_combine(k.pad_d);
        hash_combine(k.pad_h);
        hash_combine(k.pad_w);
        hash_combine(k.dil_d);
        hash_combine(k.dil_h);
        hash_combine(k.dil_w);
        hash_combine(k.groups);
        hash_combine(static_cast<int>(k.dtype));
        hash_combine(static_cast<int>(k.prefer_precise_f32));
        return h;
    }
};

// Thread-safe forward-algorithm cache. Reuses CachedFwdAlgo from the header.
class Conv3dAlgoCache {
public:
    static Conv3dAlgoCache& instance() {
        static Conv3dAlgoCache cache;
        return cache;
    }

    bool get_fwd(const Conv3dCacheKey& key, CachedFwdAlgo& result) {
        std::lock_guard<std::mutex> lock(fwd_mutex_);
        auto it = fwd_cache_.find(key);
        if (it != fwd_cache_.end()) {
            result = it->second;
            return true;
        }
        return false;
    }

    void set_fwd(const Conv3dCacheKey& key, const CachedFwdAlgo& algo) {
        std::lock_guard<std::mutex> lock(fwd_mutex_);
        fwd_cache_[key] = algo;
    }

private:
    std::mutex fwd_mutex_;
    std::unordered_map<Conv3dCacheKey, CachedFwdAlgo, Conv3dCacheKeyHash> fwd_cache_;
};

} // anonymous namespace

auto cudnn_conv3d_forward(
    const Tensor& input,
    const Tensor& weight,
    const Tensor* bias,
    std::array<int64_t, 3> stride,    // {sD, sH, sW}
    std::array<int64_t, 3> padding,   // {pD, pH, pW}
    std::array<int64_t, 3> dilation,  // {dD, dH, dW}
    int64_t groups,
    cudaStream_t stream
) -> Tensor {
    auto input_shape = input.shape();
    auto weight_shape = weight.shape();

    // 5D: [N, C, D, H, W]
    int64_t batch = input_shape[0];
    int64_t in_channels = input_shape[1];
    int64_t depth = input_shape[2];
    int64_t height = input_shape[3];
    int64_t width = input_shape[4];

    int64_t out_channels = weight_shape[0];
    int64_t kernel_d = weight_shape[2];
    int64_t kernel_h = weight_shape[3];
    int64_t kernel_w = weight_shape[4];

    int64_t out_d = (depth  + 2 * padding[0] - dilation[0] * (kernel_d - 1) - 1) / stride[0] + 1;
    int64_t out_h = (height + 2 * padding[1] - dilation[1] * (kernel_h - 1) - 1) / stride[1] + 1;
    int64_t out_w = (width  + 2 * padding[2] - dilation[2] * (kernel_w - 1) - 1) / stride[2] + 1;

    std::vector<int64_t> output_shape = {batch, out_channels, out_d, out_h, out_w};
    Tensor output(output_shape, input.dtype(), input.device());

    // TensorDescriptorNd builds packed NCDHW strides from shape; a non-contiguous
    // 5D input/weight would be read with the wrong strides. Materialize
    // contiguous copies for the data pointers.
    const Tensor& input_c = input.is_contiguous() ? input : input.contiguous();
    const Tensor& weight_c = weight.is_contiguous() ? weight : weight.contiguous();

    CudaDeviceGuard dev_guard(input.device().index);
    cudnnHandle_t handle = CuDNNHandle::get();
    CuDNNHandle::set_stream(stream);

    cudnnDataType_t cudnn_dtype = to_cudnn_dtype(input.dtype());

    // Compute type: FP32 for half-precision inputs
    cudnnDataType_t compute_type = cudnn_dtype;
    if (cudnn_dtype == CUDNN_DATA_HALF || cudnn_dtype == CUDNN_DATA_BFLOAT16) {
        compute_type = CUDNN_DATA_FLOAT;
    }

    // Set up Nd descriptors (5D for Conv3d)
    TensorDescriptorNd input_desc, output_desc;
    FilterDescriptorNd filter_desc;

    std::vector<int> input_dims = {checked_nd_dim(batch, "N"), checked_nd_dim(in_channels, "C"), checked_nd_dim(depth, "D"), checked_nd_dim(height, "H"), checked_nd_dim(width, "W")};
    std::vector<int> output_dims = {checked_nd_dim(batch, "N"), checked_nd_dim(out_channels, "C_out"), checked_nd_dim(out_d, "D_out"), checked_nd_dim(out_h, "H_out"), checked_nd_dim(out_w, "W_out")};
    std::vector<int> filter_dims = {checked_nd_dim(out_channels, "out_channels"), checked_nd_dim(in_channels / groups, "in_per_group"), checked_nd_dim(kernel_d, "kD"), checked_nd_dim(kernel_h, "kH"), checked_nd_dim(kernel_w, "kW")};

    input_desc.set(cudnn_dtype, input_dims);
    output_desc.set(cudnn_dtype, output_dims);
    filter_desc.set(cudnn_dtype, filter_dims);

    // audit L.6: descriptor + int arrays bundled in CudnnConvNdDesc — the
    // raw int* pointers passed to cudnnSetConvolutionNdDescriptor() live
    // inside the object instead of on the caller's stack.
    CudnnConvNdDesc conv_desc(padding, stride, dilation,
                              CUDNN_CROSS_CORRELATION, compute_type);

    if (groups > 1) {
        conv_desc.set_group_count(static_cast<int>(groups));
    }

    #ifdef TENZOR_HAS_TENSOR_CORES
    if (cudnn_dtype == CUDNN_DATA_HALF || cudnn_dtype == CUDNN_DATA_BFLOAT16) {
        CUDNN_CHECK(cudnnSetConvolutionMathType(conv_desc.handle(), CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION));
    } else if (cudnn_dtype == CUDNN_DATA_FLOAT) {
        // cuDNN 8+ silently uses TF32 on Ampere+ for Float32 convolutions
        // when the math type is left at CUDNN_DEFAULT_MATH. That breaks
        // Float32 parity tests against backends that compute in true FP32
        // (CPU oneDNN, Vulkan, ROCm). Honour the project-wide
        // allow_tf32 flag (and the TENZOR_DISABLE_TF32 env var that drives
        // it) by forcing CUDNN_FMA_MATH (full FP32, no TF32 conversion)
        // when the user has disabled TF32. When TF32 is allowed we leave
        // the math type at CUDNN_DEFAULT_MATH so cuDNN can opportunistically
        // pick TF32 for speed.
        if (!::tenzor::cuda::matmul::allow_tf32()) {
            CUDNN_CHECK(cudnnSetConvolutionMathType(conv_desc.handle(), CUDNN_FMA_MATH));
        }
    }
    #endif

    // Algorithm selection. Mirrors cudnn_conv2d_forward: shape+dtype-keyed
    // cache for deterministic algo reuse, plus a Winograd exclusion for
    // precise FP32 (TF32 disabled) so the chosen accumulator order matches
    // the implicit-GEMM order used by the other backends and stays inside
    // the tight Float32 parity tolerance.
    const size_t kMaxWorkspaceSize = CuDNNWorkspace::max_workspace_size();

    // Same precise-FP32 source of truth as the math-type block above and as
    // Conv2d: an FP32 input with TF32 disabled wants Winograd excluded.
    const bool prefer_precise_f32 =
        (cudnn_dtype == CUDNN_DATA_FLOAT) &&
        !::tenzor::cuda::matmul::allow_tf32();

    Conv3dCacheKey cache_key{
        batch, in_channels, depth, height, width,
        out_channels, kernel_d, kernel_h, kernel_w,
        stride[0], stride[1], stride[2],
        padding[0], padding[1], padding[2],
        dilation[0], dilation[1], dilation[2],
        groups, cudnn_dtype, prefer_precise_f32
    };

    cudnnConvolutionFwdAlgo_t algo;
    size_t workspace_size = 0;

    CachedFwdAlgo cached;
    if (Conv3dAlgoCache::instance().get_fwd(cache_key, cached)) {
        // Cache hit — same problem always resolves to the same algorithm.
        algo = cached.algo;
        workspace_size = cached.workspace_size;
    } else {
        constexpr int kMaxAlgos = 8;
        int returned_algo_count = 0;
        cudnnConvolutionFwdAlgoPerf_t perf_results[kMaxAlgos];

        CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm_v7(
            handle, input_desc.desc, filter_desc.desc, conv_desc.handle(),
            output_desc.desc, kMaxAlgos, &returned_algo_count, perf_results));

        // cuDNN's contract permits returning 0 algorithms for an unsupported
        // descriptor; perf_results[0] would then be uninitialized stack →
        // garbage algo enum. Guard before dereferencing.
        if (returned_algo_count <= 0) {
            throw std::runtime_error(
                "cuDNN Conv3d forward: no convolution algorithm available for this descriptor");
        }

        algo = perf_results[0].algo;

        // Pick fastest algorithm that fits in workspace, excluding Winograd
        // variants when precise FP32 is requested.
        float best_time = std::numeric_limits<float>::max();
        for (int i = 0; i < returned_algo_count; ++i) {
            if (perf_results[i].status != CUDNN_STATUS_SUCCESS) continue;

            if (prefer_precise_f32) {
                cudnnConvolutionFwdAlgo_t candidate = perf_results[i].algo;
                if (candidate == CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD ||
                    candidate == CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED) {
                    continue;
                }
            }

            size_t ws_size = 0;
            cudnnStatus_t ws_status = cudnnGetConvolutionForwardWorkspaceSize(
                handle, input_desc.desc, filter_desc.desc, conv_desc.handle(),
                output_desc.desc, perf_results[i].algo, &ws_size);
            if (ws_status == CUDNN_STATUS_SUCCESS && ws_size <= kMaxWorkspaceSize) {
                if (perf_results[i].time < best_time) {
                    best_time = perf_results[i].time;
                    algo = perf_results[i].algo;
                    workspace_size = ws_size;
                }
            }
        }
        if (best_time == std::numeric_limits<float>::max()) {
            // Nothing viable survived. For precise FP32, fall back to the
            // deterministic implicit-precomp-GEMM (matches Conv2d) rather
            // than whatever heuristic ordering left in perf_results[0],
            // which could be a Winograd variant.
            if (prefer_precise_f32) {
                algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
            }
            CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(
                handle, input_desc.desc, filter_desc.desc, conv_desc.handle(),
                output_desc.desc, algo, &workspace_size));
        }

        Conv3dAlgoCache::instance().set_fwd(cache_key, {algo, workspace_size});
    }

    void* workspace = CuDNNWorkspace::get(workspace_size);

    dispatch_conv_forward(
        handle, input_desc.desc, input_c.data_ptr(), filter_desc.desc, weight_c.data_ptr(),
        conv_desc.handle(), algo, workspace, workspace_size,
        output_desc.desc, output.data_ptr(), input.dtype());

    // Add bias if present: bias shape is [out_channels], described as [1, out_channels, 1, 1, 1]
    if (bias != nullptr) {
        TensorDescriptorNd bias_desc;
        std::vector<int> bias_dims = {1, (int)out_channels, 1, 1, 1};
        bias_desc.set(cudnn_dtype, bias_dims);
        dispatch_add_tensor(handle, bias_desc.desc, bias->data_ptr(), output_desc.desc, output.data_ptr(), input.dtype());
    }

    // Saturate FP16 output
    if (input.dtype() == DType::Float16) {
        fp16_saturate(output.data<Float16>(), output.numel(), stream);
    }

    return output;
}

// ============================================================================
// cuDNN Conv3d Backward Implementation
// ============================================================================

auto cudnn_conv3d_backward(
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& weight,
    std::array<int64_t, 3> stride,
    std::array<int64_t, 3> padding,
    std::array<int64_t, 3> dilation,
    int64_t groups,
    bool compute_grad_input,
    bool compute_grad_weight,
    bool compute_grad_bias,
    cudaStream_t stream
) -> std::tuple<Tensor, Tensor, Tensor> {
    auto input_shape = input.shape();
    auto weight_shape = weight.shape();
    auto grad_shape = grad_output.shape();

    int64_t batch = input_shape[0];
    int64_t in_channels = input_shape[1];
    int64_t depth = input_shape[2];
    int64_t height = input_shape[3];
    int64_t width = input_shape[4];

    int64_t out_channels = weight_shape[0];
    int64_t kernel_d = weight_shape[2];
    int64_t kernel_h = weight_shape[3];
    int64_t kernel_w = weight_shape[4];

    int64_t out_d = grad_shape[2];
    int64_t out_h = grad_shape[3];
    int64_t out_w = grad_shape[4];

    Tensor grad_input({batch, in_channels, depth, height, width}, input.dtype(), input.device());
    Tensor grad_weight({out_channels, in_channels / groups, kernel_d, kernel_h, kernel_w}, weight.dtype(), weight.device());
    Tensor grad_bias({out_channels}, weight.dtype(), weight.device());

    // TensorDescriptorNd builds packed NCDHW strides from shape; contiguify the
    // read-side tensors so non-contiguous views are not read with wrong strides.
    const Tensor& input_c = input.is_contiguous() ? input : input.contiguous();
    const Tensor& weight_c = weight.is_contiguous() ? weight : weight.contiguous();
    const Tensor& grad_output_c = grad_output.is_contiguous() ? grad_output : grad_output.contiguous();

    CudaDeviceGuard dev_guard(input.device().index);
    cudnnHandle_t handle = CuDNNHandle::get();
    CuDNNHandle::set_stream(stream);

    cudnnDataType_t cudnn_dtype = to_cudnn_dtype(input.dtype());
    cudnnDataType_t compute_type = cudnn_dtype;
    if (cudnn_dtype == CUDNN_DATA_HALF || cudnn_dtype == CUDNN_DATA_BFLOAT16) {
        compute_type = CUDNN_DATA_FLOAT;
    }

    TensorDescriptorNd input_desc, grad_output_desc;
    FilterDescriptorNd filter_desc;

    std::vector<int> input_dims = {checked_nd_dim(batch, "N"), checked_nd_dim(in_channels, "C"), checked_nd_dim(depth, "D"), checked_nd_dim(height, "H"), checked_nd_dim(width, "W")};
    std::vector<int> grad_output_dims = {checked_nd_dim(batch, "N"), checked_nd_dim(out_channels, "C_out"), checked_nd_dim(out_d, "D_out"), checked_nd_dim(out_h, "H_out"), checked_nd_dim(out_w, "W_out")};
    std::vector<int> filter_dims = {checked_nd_dim(out_channels, "out_channels"), checked_nd_dim(in_channels / groups, "in_per_group"), checked_nd_dim(kernel_d, "kD"), checked_nd_dim(kernel_h, "kH"), checked_nd_dim(kernel_w, "kW")};

    input_desc.set(cudnn_dtype, input_dims);
    grad_output_desc.set(cudnn_dtype, grad_output_dims);
    filter_desc.set(cudnn_dtype, filter_dims);

    // audit L.6: descriptor + int arrays bundled in CudnnConvNdDesc.
    CudnnConvNdDesc conv_desc(padding, stride, dilation,
                              CUDNN_CROSS_CORRELATION, compute_type);

    if (groups > 1) {
        conv_desc.set_group_count(static_cast<int>(groups));
    }

    #ifdef TENZOR_HAS_TENSOR_CORES
    if (cudnn_dtype == CUDNN_DATA_HALF || cudnn_dtype == CUDNN_DATA_BFLOAT16) {
        CUDNN_CHECK(cudnnSetConvolutionMathType(conv_desc.handle(), CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION));
    } else if (cudnn_dtype == CUDNN_DATA_FLOAT) {
        // cuDNN 8+ silently uses TF32 on Ampere+ for Float32 convolutions
        // when the math type is left at CUDNN_DEFAULT_MATH. That breaks
        // Float32 parity tests against backends that compute in true FP32
        // (CPU oneDNN, Vulkan, ROCm). Honour the project-wide
        // allow_tf32 flag (and the TENZOR_DISABLE_TF32 env var that drives
        // it) by forcing CUDNN_FMA_MATH (full FP32, no TF32 conversion)
        // when the user has disabled TF32. When TF32 is allowed we leave
        // the math type at CUDNN_DEFAULT_MATH so cuDNN can opportunistically
        // pick TF32 for speed.
        if (!::tenzor::cuda::matmul::allow_tf32()) {
            CUDNN_CHECK(cudnnSetConvolutionMathType(conv_desc.handle(), CUDNN_FMA_MATH));
        }
    }
    #endif

    const size_t kMaxWorkspaceSize = CuDNNWorkspace::max_workspace_size();

    // Same precise-FP32 source of truth as the math-type block above and as
    // cudnn_conv3d_forward: an FP32 input with TF32 disabled wants Winograd
    // excluded from the bwd-data and bwd-filter algorithm selection too.
    // CUDNN_FMA_MATH (set above) only suppresses TF32 conversion, not the
    // algorithmic transform, so without this exclusion cuDNN could still pick
    // a Winograd backward algorithm whose accumulator order diverges past the
    // tight Float32 parity tolerance even though the forward stays inside it.
    const bool prefer_precise_f32 =
        (cudnn_dtype == CUDNN_DATA_FLOAT) &&
        !::tenzor::cuda::matmul::allow_tf32();

    // Gradient w.r.t. input
    if (compute_grad_input) {
        constexpr int kMaxAlgos = 8;
        int returned_algo_count = 0;
        cudnnConvolutionBwdDataAlgoPerf_t perf_results[kMaxAlgos];

        CUDNN_CHECK(cudnnGetConvolutionBackwardDataAlgorithm_v7(
            handle, filter_desc.desc, grad_output_desc.desc, conv_desc.handle(),
            input_desc.desc, kMaxAlgos, &returned_algo_count, perf_results));

        if (returned_algo_count <= 0) {
            throw std::runtime_error(
                "cuDNN Conv3d backward-data: no algorithm available for this descriptor");
        }
        cudnnConvolutionBwdDataAlgo_t algo = perf_results[0].algo;
        size_t workspace_size = 0;
        float best_time = std::numeric_limits<float>::max();

        for (int i = 0; i < returned_algo_count; ++i) {
            if (perf_results[i].status != CUDNN_STATUS_SUCCESS) continue;

            if (prefer_precise_f32) {
                cudnnConvolutionBwdDataAlgo_t candidate = perf_results[i].algo;
                if (candidate == CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD ||
                    candidate == CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED) {
                    continue;
                }
            }

            size_t ws_size = 0;
            cudnnStatus_t ws_status = cudnnGetConvolutionBackwardDataWorkspaceSize(
                handle, filter_desc.desc, grad_output_desc.desc, conv_desc.handle(),
                input_desc.desc, perf_results[i].algo, &ws_size);
            if (ws_status == CUDNN_STATUS_SUCCESS && ws_size <= kMaxWorkspaceSize) {
                if (perf_results[i].time < best_time) {
                    best_time = perf_results[i].time;
                    algo = perf_results[i].algo;
                    workspace_size = ws_size;
                }
            }
        }
        if (best_time == std::numeric_limits<float>::max()) {
            // Nothing viable survived. For precise FP32, fall back to the
            // deterministic ALGO_1 (implicit-GEMM-style, non-Winograd) rather
            // than whatever heuristic ordering left in perf_results[0], which
            // could be a Winograd variant. Mirrors the forward fallback.
            if (prefer_precise_f32) {
                algo = CUDNN_CONVOLUTION_BWD_DATA_ALGO_1;
            }
            CUDNN_CHECK(cudnnGetConvolutionBackwardDataWorkspaceSize(
                handle, filter_desc.desc, grad_output_desc.desc, conv_desc.handle(),
                input_desc.desc, algo, &workspace_size));
        }

        void* workspace = CuDNNWorkspace::get(workspace_size);
        dispatch_conv_bwd_data(
            handle, filter_desc.desc, weight_c.data_ptr(), grad_output_desc.desc, grad_output_c.data_ptr(),
            conv_desc.handle(), algo, workspace, workspace_size,
            input_desc.desc, grad_input.data_ptr(), input.dtype());
    }

    // Gradient w.r.t. weight
    if (compute_grad_weight) {
        constexpr int kMaxAlgos = 8;
        int returned_algo_count = 0;
        cudnnConvolutionBwdFilterAlgoPerf_t perf_results[kMaxAlgos];

        CUDNN_CHECK(cudnnGetConvolutionBackwardFilterAlgorithm_v7(
            handle, input_desc.desc, grad_output_desc.desc, conv_desc.handle(),
            filter_desc.desc, kMaxAlgos, &returned_algo_count, perf_results));

        if (returned_algo_count <= 0) {
            throw std::runtime_error(
                "cuDNN Conv3d backward-filter: no algorithm available for this descriptor");
        }
        cudnnConvolutionBwdFilterAlgo_t algo = perf_results[0].algo;
        size_t workspace_size = 0;
        float best_time = std::numeric_limits<float>::max();

        for (int i = 0; i < returned_algo_count; ++i) {
            if (perf_results[i].status != CUDNN_STATUS_SUCCESS) continue;

            if (prefer_precise_f32) {
                cudnnConvolutionBwdFilterAlgo_t candidate = perf_results[i].algo;
                if (candidate == CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD ||
                    candidate == CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED) {
                    continue;
                }
            }

            size_t ws_size = 0;
            cudnnStatus_t ws_status = cudnnGetConvolutionBackwardFilterWorkspaceSize(
                handle, input_desc.desc, grad_output_desc.desc, conv_desc.handle(),
                filter_desc.desc, perf_results[i].algo, &ws_size);
            if (ws_status == CUDNN_STATUS_SUCCESS && ws_size <= kMaxWorkspaceSize) {
                if (perf_results[i].time < best_time) {
                    best_time = perf_results[i].time;
                    algo = perf_results[i].algo;
                    workspace_size = ws_size;
                }
            }
        }
        if (best_time == std::numeric_limits<float>::max()) {
            // Nothing viable survived. For precise FP32, fall back to the
            // deterministic ALGO_1 (implicit-GEMM-style, non-Winograd) rather
            // than whatever heuristic ordering left in perf_results[0], which
            // could be a Winograd variant. Mirrors the forward fallback.
            if (prefer_precise_f32) {
                algo = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1;
            }
            CUDNN_CHECK(cudnnGetConvolutionBackwardFilterWorkspaceSize(
                handle, input_desc.desc, grad_output_desc.desc, conv_desc.handle(),
                filter_desc.desc, algo, &workspace_size));
        }

        void* workspace = CuDNNWorkspace::get(workspace_size);
        dispatch_conv_bwd_filter(
            handle, input_desc.desc, input_c.data_ptr(), grad_output_desc.desc, grad_output_c.data_ptr(),
            conv_desc.handle(), algo, workspace, workspace_size,
            filter_desc.desc, grad_weight.data_ptr(), input.dtype());
    }

    // Gradient w.r.t. bias
    if (compute_grad_bias) {
        TensorDescriptorNd bias_desc;
        std::vector<int> bias_dims = {1, (int)out_channels, 1, 1, 1};
        bias_desc.set(cudnn_dtype, bias_dims);
        dispatch_conv_bwd_bias(
            handle, grad_output_desc.desc, grad_output_c.data_ptr(),
            bias_desc.desc, grad_bias.data_ptr(), input.dtype());
    }

    return std::make_tuple(grad_input, grad_weight, grad_bias);
}

// ============================================================================
// cuDNN ConvTranspose3d Forward Implementation
// Transposed convolution forward is mathematically cudnnConvolutionBackwardData
// ============================================================================

auto cudnn_conv_transpose3d_forward(
    const Tensor& input,
    const Tensor& weight,
    const Tensor* bias,
    std::array<int64_t, 3> stride,
    std::array<int64_t, 3> padding,
    std::array<int64_t, 3> output_padding,
    std::array<int64_t, 3> dilation,
    int64_t groups,
    cudaStream_t stream
) -> Tensor {
    auto input_shape = input.shape();
    auto weight_shape = weight.shape();

    // Input: [N, C_in, D_in, H_in, W_in]
    // Weight for ConvTranspose: [C_in, C_out/groups, kD, kH, kW]
    int64_t batch = input_shape[0];
    int64_t in_channels = input_shape[1];
    int64_t d_in = input_shape[2];
    int64_t h_in = input_shape[3];
    int64_t w_in = input_shape[4];

    int64_t out_channels = weight_shape[1] * groups;
    int64_t kernel_d = weight_shape[2];
    int64_t kernel_h = weight_shape[3];
    int64_t kernel_w = weight_shape[4];

    // Output dimensions for transposed convolution (full, including
    // output_padding).
    int64_t d_out = (d_in - 1) * stride[0] - 2 * padding[0] + dilation[0] * (kernel_d - 1) + output_padding[0] + 1;
    int64_t h_out = (h_in - 1) * stride[1] - 2 * padding[1] + dilation[1] * (kernel_h - 1) + output_padding[1] + 1;
    int64_t w_out = (w_in - 1) * stride[2] - 2 * padding[2] + dilation[2] * (kernel_w - 1) + output_padding[2] + 1;

    // output_padding support (mirrors the CPU conv_transpose3d path):
    //
    // cuDNN's ConvolutionBackwardData expresses a transposed conv whose output
    // extent is exactly the base size (output_padding == 0):
    //     base = (in-1)*stride - 2*padding + dilation*(k-1) + 1
    // output_padding only ENLARGES the output by `op` rows/cols on the HIGH
    // side of each spatial axis. Crucially, the transposed-conv scatter never
    // writes those extra positions: the largest written index along an axis is
    //     (in-1)*stride - padding + (k-1)*dilation  ==  base - 1
    // which is < base <= every output_padding index. So the output_padding
    // region is provably all-zero (this is exactly PyTorch's semantics, and
    // matches our CPU kernel which sizes the output with op and leaves the
    // extra border at the memset-zero value).
    //
    // Therefore the correct, cuDNN-only (NO CPU fallback) implementation is:
    //   1. run cuDNN BackwardData at the BASE extent into a contiguous tensor,
    //   2. if any output_padding != 0, embed that base result into the leading
    //      sub-volume of a zero-initialized full-size output (device-to-device).
    const int64_t d_base = d_out - output_padding[0];
    const int64_t h_base = h_out - output_padding[1];
    const int64_t w_base = w_out - output_padding[2];
    const bool has_output_padding =
        output_padding[0] != 0 || output_padding[1] != 0 || output_padding[2] != 0;

    // cuDNN always writes the BASE-sized output; `output` is what we return.
    std::vector<int64_t> base_shape = {batch, out_channels, d_base, h_base, w_base};
    std::vector<int64_t> full_shape = {batch, out_channels, d_out, h_out, w_out};
    Tensor base_output(base_shape, input.dtype(), input.device());
    // For the descriptor / cuDNN write target below, "output" names the base
    // tensor; we only construct the padded full tensor at the very end.
    Tensor& output = base_output;
    // The descriptor must describe the BASE extent that cuDNN actually computes.
    int64_t d_out_desc = d_base, h_out_desc = h_base, w_out_desc = w_base;

    // TensorDescriptorNd builds packed NCDHW strides from shape; contiguify reads.
    const Tensor& input_c = input.is_contiguous() ? input : input.contiguous();
    const Tensor& weight_c = weight.is_contiguous() ? weight : weight.contiguous();

    CudaDeviceGuard dev_guard(input.device().index);
    cudnnHandle_t handle = CuDNNHandle::get();
    CuDNNHandle::set_stream(stream);

    cudnnDataType_t cudnn_dtype = to_cudnn_dtype(input.dtype());
    cudnnDataType_t compute_type = cudnn_dtype;
    if (cudnn_dtype == CUDNN_DATA_HALF || cudnn_dtype == CUDNN_DATA_BFLOAT16) {
        compute_type = CUDNN_DATA_FLOAT;
    }

    // For transposed conv, the "input" to cudnnConvolutionBackwardData is our grad_output (input),
    // and the "output" (grad_input) is our actual output.
    // The filter descriptor uses the same weight layout.
    TensorDescriptorNd input_desc, output_desc;
    FilterDescriptorNd filter_desc;

    // input_desc describes our input (which is the "grad_output" in cuDNN backward data terms)
    std::vector<int> input_dims = {checked_nd_dim(batch, "N"), checked_nd_dim(in_channels, "C"), checked_nd_dim(d_in, "D"), checked_nd_dim(h_in, "H"), checked_nd_dim(w_in, "W")};
    // output_desc describes our output (which is the "grad_input" in cuDNN backward data terms)
    std::vector<int> output_dims = {checked_nd_dim(batch, "N"), checked_nd_dim(out_channels, "C_out"), checked_nd_dim(d_out_desc, "D_out"), checked_nd_dim(h_out_desc, "H_out"), checked_nd_dim(w_out_desc, "W_out")};
    // Filter: [C_in, C_out/groups, kD, kH, kW]
    std::vector<int> filter_dims = {checked_nd_dim(in_channels, "C_in"), checked_nd_dim(out_channels / groups, "out_per_group"), checked_nd_dim(kernel_d, "kD"), checked_nd_dim(kernel_h, "kH"), checked_nd_dim(kernel_w, "kW")};

    input_desc.set(cudnn_dtype, input_dims);
    output_desc.set(cudnn_dtype, output_dims);
    filter_desc.set(cudnn_dtype, filter_dims);

    // audit L.6: descriptor + int arrays bundled in CudnnConvNdDesc.
    CudnnConvNdDesc conv_desc(padding, stride, dilation,
                              CUDNN_CROSS_CORRELATION, compute_type);

    if (groups > 1) {
        conv_desc.set_group_count(static_cast<int>(groups));
    }

    #ifdef TENZOR_HAS_TENSOR_CORES
    if (cudnn_dtype == CUDNN_DATA_HALF || cudnn_dtype == CUDNN_DATA_BFLOAT16) {
        CUDNN_CHECK(cudnnSetConvolutionMathType(conv_desc.handle(), CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION));
    } else if (cudnn_dtype == CUDNN_DATA_FLOAT) {
        // cuDNN 8+ silently uses TF32 on Ampere+ for Float32 convolutions
        // when the math type is left at CUDNN_DEFAULT_MATH. That breaks
        // Float32 parity tests against backends that compute in true FP32
        // (CPU oneDNN, Vulkan, ROCm). Honour the project-wide
        // allow_tf32 flag (and the TENZOR_DISABLE_TF32 env var that drives
        // it) by forcing CUDNN_FMA_MATH (full FP32, no TF32 conversion)
        // when the user has disabled TF32. When TF32 is allowed we leave
        // the math type at CUDNN_DEFAULT_MATH so cuDNN can opportunistically
        // pick TF32 for speed.
        if (!::tenzor::cuda::matmul::allow_tf32()) {
            CUDNN_CHECK(cudnnSetConvolutionMathType(conv_desc.handle(), CUDNN_FMA_MATH));
        }
    }
    #endif

    // Use cudnnConvolutionBackwardData for transposed convolution forward
    constexpr int kMaxAlgos = 8;
    int returned_algo_count = 0;
    cudnnConvolutionBwdDataAlgoPerf_t perf_results[kMaxAlgos];

    CUDNN_CHECK(cudnnGetConvolutionBackwardDataAlgorithm_v7(
        handle, filter_desc.desc, input_desc.desc, conv_desc.handle(),
        output_desc.desc, kMaxAlgos, &returned_algo_count, perf_results));

    if (returned_algo_count <= 0) {
        throw std::runtime_error(
            "cuDNN ConvTranspose3d forward: no algorithm available for this descriptor");
    }
    const size_t kMaxWorkspaceSize = CuDNNWorkspace::max_workspace_size();
    cudnnConvolutionBwdDataAlgo_t algo = perf_results[0].algo;
    size_t workspace_size = 0;
    float best_time = std::numeric_limits<float>::max();

    for (int i = 0; i < returned_algo_count; ++i) {
        if (perf_results[i].status != CUDNN_STATUS_SUCCESS) continue;
        size_t ws_size = 0;
        cudnnStatus_t ws_status = cudnnGetConvolutionBackwardDataWorkspaceSize(
            handle, filter_desc.desc, input_desc.desc, conv_desc.handle(),
            output_desc.desc, perf_results[i].algo, &ws_size);
        if (ws_status == CUDNN_STATUS_SUCCESS && ws_size <= kMaxWorkspaceSize) {
            if (perf_results[i].time < best_time) {
                best_time = perf_results[i].time;
                algo = perf_results[i].algo;
                workspace_size = ws_size;
            }
        }
    }
    if (best_time == std::numeric_limits<float>::max()) {
        CUDNN_CHECK(cudnnGetConvolutionBackwardDataWorkspaceSize(
            handle, filter_desc.desc, input_desc.desc, conv_desc.handle(),
            output_desc.desc, algo, &workspace_size));
    }

    void* workspace = CuDNNWorkspace::get(workspace_size);

    // BackwardData: filter * input -> output (transposed conv forward)
    dispatch_conv_bwd_data(
        handle, filter_desc.desc, weight_c.data_ptr(), input_desc.desc, input_c.data_ptr(),
        conv_desc.handle(), algo, workspace, workspace_size,
        output_desc.desc, output.data_ptr(), input.dtype());

    // Add bias
    if (bias != nullptr) {
        TensorDescriptorNd bias_desc;
        std::vector<int> bias_dims = {1, (int)out_channels, 1, 1, 1};
        bias_desc.set(cudnn_dtype, bias_dims);
        dispatch_add_tensor(handle, bias_desc.desc, bias->data_ptr(), output_desc.desc, output.data_ptr(), input.dtype());
    }

    if (input.dtype() == DType::Float16) {
        fp16_saturate(output.data<Float16>(), output.numel(), stream);
    }

    if (!has_output_padding) {
        return base_output;
    }

    // Embed the base result into the leading sub-volume of a zero-initialized
    // full-size output. The output_padding border stays zero (proven above).
    // This is a pure device-to-device copy — no host bounce, no CPU fallback.
    Tensor full_output(full_shape, input.dtype(), input.device());
    const size_t elem = input.dtype_size();
    // Zero the whole full output first so the output_padding border is 0.
    CUDA_CHECK(cudaMemsetAsync(full_output.data_ptr(), 0,
                                      static_cast<size_t>(full_output.numel()) * elem,
                                      stream));

    // Copy each (n, c, d) plane [h_base x w_base] from the contiguous base
    // tensor into the matching corner of the strided full tensor. cudaMemcpy2D
    // handles the H/W pitch difference (w_base -> w_out); we loop the leading
    // N*C*d_base planes (d only runs over d_base, so the op-D border is skipped
    // and stays zero).
    const auto* src_base = static_cast<const uint8_t*>(base_output.data_ptr());
    auto* dst_base = static_cast<uint8_t*>(full_output.data_ptr());
    const int64_t nc = batch * out_channels;
    const size_t src_plane = static_cast<size_t>(h_base) * w_base * elem;       // contiguous base D-plane
    const size_t dst_plane = static_cast<size_t>(h_out) * w_out * elem;          // strided full D-plane
    const size_t src_pitch = static_cast<size_t>(w_base) * elem;
    const size_t dst_pitch = static_cast<size_t>(w_out) * elem;
    const size_t row_bytes = static_cast<size_t>(w_base) * elem;
    for (int64_t p = 0; p < nc; ++p) {
        // p indexes a (n,c); within it, d runs 0..d_base-1 (skipping op-D rows).
        const uint8_t* src_nc = src_base + static_cast<size_t>(p) * d_base * src_plane;
        // In the full tensor, the (n,c) block has d_out planes; leading d_base
        // are written, the rest stay zero.
        uint8_t* dst_nc = dst_base + static_cast<size_t>(p) * d_out * dst_plane;
        for (int64_t d = 0; d < d_base; ++d) {
            CUDA_CHECK(cudaMemcpy2DAsync(
                dst_nc + static_cast<size_t>(d) * dst_plane, dst_pitch,
                src_nc + static_cast<size_t>(d) * src_plane, src_pitch,
                row_bytes, static_cast<size_t>(h_base),
                cudaMemcpyDeviceToDevice, stream));
        }
    }

    return full_output;
}

// ============================================================================
// cuDNN ConvTranspose3d Backward Implementation
// ============================================================================

auto cudnn_conv_transpose3d_backward(
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& weight,
    std::array<int64_t, 3> stride,
    std::array<int64_t, 3> padding,
    std::array<int64_t, 3> output_padding,
    std::array<int64_t, 3> dilation,
    int64_t groups,
    bool compute_grad_input,
    bool compute_grad_weight,
    bool compute_grad_bias,
    cudaStream_t stream
) -> std::tuple<Tensor, Tensor, Tensor> {
    auto input_shape = input.shape();
    auto weight_shape = weight.shape();
    auto grad_shape = grad_output.shape();

    int64_t batch = input_shape[0];
    int64_t in_channels = input_shape[1];
    int64_t d_in = input_shape[2];
    int64_t h_in = input_shape[3];
    int64_t w_in = input_shape[4];

    int64_t out_channels = weight_shape[1] * groups;
    int64_t kernel_d = weight_shape[2];
    int64_t kernel_h = weight_shape[3];
    int64_t kernel_w = weight_shape[4];

    int64_t d_out = grad_shape[2];
    int64_t h_out = grad_shape[3];
    int64_t w_out = grad_shape[4];

    Tensor grad_input({batch, in_channels, d_in, h_in, w_in}, input.dtype(), input.device());
    Tensor grad_weight({in_channels, out_channels / groups, kernel_d, kernel_h, kernel_w}, weight.dtype(), weight.device());
    Tensor grad_bias({out_channels}, weight.dtype(), weight.device());

    // TensorDescriptorNd builds packed NCDHW strides from shape; contiguify reads.
    const Tensor& input_c = input.is_contiguous() ? input : input.contiguous();
    const Tensor& weight_c = weight.is_contiguous() ? weight : weight.contiguous();
    const Tensor& grad_output_c = grad_output.is_contiguous() ? grad_output : grad_output.contiguous();

    CudaDeviceGuard dev_guard(input.device().index);
    cudnnHandle_t handle = CuDNNHandle::get();
    CuDNNHandle::set_stream(stream);

    cudnnDataType_t cudnn_dtype = to_cudnn_dtype(input.dtype());
    cudnnDataType_t compute_type = cudnn_dtype;
    if (cudnn_dtype == CUDNN_DATA_HALF || cudnn_dtype == CUDNN_DATA_BFLOAT16) {
        compute_type = CUDNN_DATA_FLOAT;
    }

    // Descriptors: same layout as forward
    TensorDescriptorNd input_desc, grad_output_desc;
    FilterDescriptorNd filter_desc;

    std::vector<int> input_dims = {checked_nd_dim(batch, "N"), checked_nd_dim(in_channels, "C"), checked_nd_dim(d_in, "D"), checked_nd_dim(h_in, "H"), checked_nd_dim(w_in, "W")};
    std::vector<int> grad_output_dims = {checked_nd_dim(batch, "N"), checked_nd_dim(out_channels, "C_out"), checked_nd_dim(d_out, "D_out"), checked_nd_dim(h_out, "H_out"), checked_nd_dim(w_out, "W_out")};
    std::vector<int> filter_dims = {checked_nd_dim(in_channels, "C_in"), checked_nd_dim(out_channels / groups, "out_per_group"), checked_nd_dim(kernel_d, "kD"), checked_nd_dim(kernel_h, "kH"), checked_nd_dim(kernel_w, "kW")};

    input_desc.set(cudnn_dtype, input_dims);
    grad_output_desc.set(cudnn_dtype, grad_output_dims);
    filter_desc.set(cudnn_dtype, filter_dims);

    // audit L.6: descriptor + int arrays bundled in CudnnConvNdDesc.
    CudnnConvNdDesc conv_desc(padding, stride, dilation,
                              CUDNN_CROSS_CORRELATION, compute_type);

    if (groups > 1) {
        conv_desc.set_group_count(static_cast<int>(groups));
    }

    #ifdef TENZOR_HAS_TENSOR_CORES
    if (cudnn_dtype == CUDNN_DATA_HALF || cudnn_dtype == CUDNN_DATA_BFLOAT16) {
        CUDNN_CHECK(cudnnSetConvolutionMathType(conv_desc.handle(), CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION));
    } else if (cudnn_dtype == CUDNN_DATA_FLOAT) {
        // cuDNN 8+ silently uses TF32 on Ampere+ for Float32 convolutions
        // when the math type is left at CUDNN_DEFAULT_MATH. That breaks
        // Float32 parity tests against backends that compute in true FP32
        // (CPU oneDNN, Vulkan, ROCm). Honour the project-wide
        // allow_tf32 flag (and the TENZOR_DISABLE_TF32 env var that drives
        // it) by forcing CUDNN_FMA_MATH (full FP32, no TF32 conversion)
        // when the user has disabled TF32. When TF32 is allowed we leave
        // the math type at CUDNN_DEFAULT_MATH so cuDNN can opportunistically
        // pick TF32 for speed.
        if (!::tenzor::cuda::matmul::allow_tf32()) {
            CUDNN_CHECK(cudnnSetConvolutionMathType(conv_desc.handle(), CUDNN_FMA_MATH));
        }
    }
    #endif

    const size_t kMaxWorkspaceSize = CuDNNWorkspace::max_workspace_size();

    // Gradient w.r.t. input: use cudnnConvolutionForward
    // (transposed conv backward-input IS regular conv forward)
    if (compute_grad_input) {
        constexpr int kMaxAlgos = 8;
        int returned_algo_count = 0;
        cudnnConvolutionFwdAlgoPerf_t perf_results[kMaxAlgos];

        CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm_v7(
            handle, grad_output_desc.desc, filter_desc.desc, conv_desc.handle(),
            input_desc.desc, kMaxAlgos, &returned_algo_count, perf_results));

        if (returned_algo_count <= 0) {
            throw std::runtime_error(
                "cuDNN ConvTranspose3d backward-input: no algorithm available for this descriptor");
        }
        cudnnConvolutionFwdAlgo_t algo = perf_results[0].algo;
        size_t workspace_size = 0;
        float best_time = std::numeric_limits<float>::max();

        for (int i = 0; i < returned_algo_count; ++i) {
            if (perf_results[i].status != CUDNN_STATUS_SUCCESS) continue;
            size_t ws_size = 0;
            cudnnStatus_t ws_status = cudnnGetConvolutionForwardWorkspaceSize(
                handle, grad_output_desc.desc, filter_desc.desc, conv_desc.handle(),
                input_desc.desc, perf_results[i].algo, &ws_size);
            if (ws_status == CUDNN_STATUS_SUCCESS && ws_size <= kMaxWorkspaceSize) {
                if (perf_results[i].time < best_time) {
                    best_time = perf_results[i].time;
                    algo = perf_results[i].algo;
                    workspace_size = ws_size;
                }
            }
        }
        if (best_time == std::numeric_limits<float>::max()) {
            CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(
                handle, grad_output_desc.desc, filter_desc.desc, conv_desc.handle(),
                input_desc.desc, algo, &workspace_size));
        }

        void* workspace = CuDNNWorkspace::get(workspace_size);
        dispatch_conv_forward(
            handle, grad_output_desc.desc, grad_output_c.data_ptr(), filter_desc.desc, weight_c.data_ptr(),
            conv_desc.handle(), algo, workspace, workspace_size,
            input_desc.desc, grad_input.data_ptr(), input.dtype());
    }

    // Gradient w.r.t. weight: use cudnnConvolutionBackwardFilter
    // but with swapped input/grad_output roles
    if (compute_grad_weight) {
        constexpr int kMaxAlgos = 8;
        int returned_algo_count = 0;
        cudnnConvolutionBwdFilterAlgoPerf_t perf_results[kMaxAlgos];

        CUDNN_CHECK(cudnnGetConvolutionBackwardFilterAlgorithm_v7(
            handle, grad_output_desc.desc, input_desc.desc, conv_desc.handle(),
            filter_desc.desc, kMaxAlgos, &returned_algo_count, perf_results));

        if (returned_algo_count <= 0) {
            throw std::runtime_error(
                "cuDNN ConvTranspose3d backward-filter: no algorithm available for this descriptor");
        }
        cudnnConvolutionBwdFilterAlgo_t algo = perf_results[0].algo;
        size_t workspace_size = 0;
        float best_time = std::numeric_limits<float>::max();

        for (int i = 0; i < returned_algo_count; ++i) {
            if (perf_results[i].status != CUDNN_STATUS_SUCCESS) continue;
            size_t ws_size = 0;
            cudnnStatus_t ws_status = cudnnGetConvolutionBackwardFilterWorkspaceSize(
                handle, grad_output_desc.desc, input_desc.desc, conv_desc.handle(),
                filter_desc.desc, perf_results[i].algo, &ws_size);
            if (ws_status == CUDNN_STATUS_SUCCESS && ws_size <= kMaxWorkspaceSize) {
                if (perf_results[i].time < best_time) {
                    best_time = perf_results[i].time;
                    algo = perf_results[i].algo;
                    workspace_size = ws_size;
                }
            }
        }
        if (best_time == std::numeric_limits<float>::max()) {
            CUDNN_CHECK(cudnnGetConvolutionBackwardFilterWorkspaceSize(
                handle, grad_output_desc.desc, input_desc.desc, conv_desc.handle(),
                filter_desc.desc, algo, &workspace_size));
        }

        void* workspace = CuDNNWorkspace::get(workspace_size);
        dispatch_conv_bwd_filter(
            handle, grad_output_desc.desc, grad_output_c.data_ptr(), input_desc.desc, input_c.data_ptr(),
            conv_desc.handle(), algo, workspace, workspace_size,
            filter_desc.desc, grad_weight.data_ptr(), input.dtype());
    }

    // Gradient w.r.t. bias: sum grad_output over N,D,H,W dimensions
    if (compute_grad_bias) {
        TensorDescriptorNd bias_desc;
        std::vector<int> bias_dims = {1, (int)out_channels, 1, 1, 1};
        bias_desc.set(cudnn_dtype, bias_dims);
        dispatch_conv_bwd_bias(
            handle, grad_output_desc.desc, grad_output_c.data_ptr(),
            bias_desc.desc, grad_bias.data_ptr(), input.dtype());
    }

    return std::make_tuple(grad_input, grad_weight, grad_bias);
}

// ABI-safe wrappers that avoid returning std::tuple across nvcc/g++ boundary.
// The tuple return type can cause parameter misalignment between .cu (nvcc) and
// .cpp (g++) translation units on some toolchain combinations.

Tensor cudnn_conv_transpose3d_backward_input(
    const Tensor& grad_output, const Tensor& input, const Tensor& weight,
    std::array<int64_t, 3> stride, std::array<int64_t, 3> padding,
    std::array<int64_t, 3> output_padding, std::array<int64_t, 3> dilation,
    int64_t groups, cudaStream_t stream)
{
    auto result = cudnn_conv_transpose3d_backward(
        grad_output, input, weight, stride, padding, output_padding,
        dilation, groups, true, false, false, stream);
    return std::get<0>(result);
}

Tensor cudnn_conv_transpose3d_backward_weight(
    const Tensor& grad_output, const Tensor& input, const Tensor& weight,
    std::array<int64_t, 3> stride, std::array<int64_t, 3> padding,
    std::array<int64_t, 3> output_padding, std::array<int64_t, 3> dilation,
    int64_t groups, cudaStream_t stream)
{
    auto result = cudnn_conv_transpose3d_backward(
        grad_output, input, weight, stride, padding, output_padding,
        dilation, groups, false, true, false, stream);
    return std::get<1>(result);
}

Tensor cudnn_conv_transpose3d_backward_bias(
    const Tensor& grad_output, const Tensor& input, const Tensor& weight,
    std::array<int64_t, 3> stride, std::array<int64_t, 3> padding,
    std::array<int64_t, 3> output_padding, std::array<int64_t, 3> dilation,
    int64_t groups, cudaStream_t stream)
{
    auto result = cudnn_conv_transpose3d_backward(
        grad_output, input, weight, stride, padding, output_padding,
        dilation, groups, false, false, true, stream);
    return std::get<2>(result);
}

} // namespace cuda
} // namespace tenzor

#endif // TENZOR_HAS_CUDNN
