#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include "tenzor/core/tensor.hpp"
#include "tenzor/backend/loader_fwd.hpp"
#include <stdexcept>
#include <cstdint>

namespace tenzor {
namespace rocm {

// Error checking macro
#define HIP_CHECK(call) \
    do { \
        hipError_t error = call; \
        if (error != hipSuccess) { \
            throw std::runtime_error( \
                std::string("HIP error at ") + __FILE__ + ":" + \
                std::to_string(__LINE__) + " - " + hipGetErrorString(error) \
            ); \
        } \
    } while(0)

// ==============================================================================
// Quantized Linear HIP Kernel (INT8 inputs, INT32 accumulation)
// ==============================================================================

__global__ void quantized_linear_kernel_hip(
    const int8_t* __restrict__ input,
    const int8_t* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int64_t batch_size,
    int64_t in_features,
    int64_t out_features,
    float combined_scale,
    int32_t input_zp,
    int32_t weight_zp,
    // Per-channel (F045): when weight_scales != nullptr each output channel o
    // uses weight_scales[o] and (weight_zps ? weight_zps[o] : 0); the dequant
    // scale is input_over_output * weight_scales[o]. nullptr => per-tensor.
    const float* __restrict__ weight_scales,
    const int32_t* __restrict__ weight_zps,
    float input_over_output
) {
    // 2D grid: blockIdx.y = batch, blockIdx.x * blockDim.x + threadIdx.x = output feature
    int64_t b = blockIdx.y;
    int64_t o = blockIdx.x * blockDim.x + threadIdx.x;

    if (b >= batch_size || o >= out_features) return;

    // Per-channel weight scale / zero-point (each output channel o).
    const int32_t w_zp = weight_scales ? (weight_zps ? weight_zps[o] : 0) : weight_zp;
    const float out_scale = weight_scales ? (input_over_output * weight_scales[o])
                                          : combined_scale;

    const int8_t* input_row = input + b * in_features;
    const int8_t* weight_row = weight + o * in_features;

    // int64 accumulation to match the CUDA kernel (quantized_linear.cu): each
    // int8*int8 product fits in int32, but the row sum and the
    // input_zp*weight_zp*in_features correction term can exceed INT32_MAX for
    // wide layers (in_features >~ 131072), so accumulate in int64 to avoid
    // silent wraparound.
    int64_t acc = 0;
    int64_t sum_x = 0;
    int64_t sum_w = 0;

    // Vectorized loading: process 16 int8 values at a time via int4 (16 bytes).
    // The int4 (16-byte) dereference requires the per-row base pointer to be
    // 16-byte aligned. Each row base is offset by `* in_features` int8 bytes,
    // so when in_features is not a multiple of 16 the per-row bases drift out of
    // alignment, making the int4 load undefined behavior on HIP. Only take the
    // vectorized fast path when in_features is a multiple of VEC_SIZE; otherwise
    // fall back entirely to the scalar loop.
    constexpr int VEC_SIZE = 16;
    int64_t start = 0;

    if ((in_features % VEC_SIZE) == 0) {
        int64_t vec_steps = in_features / VEC_SIZE;
        for (int64_t v = 0; v < vec_steps; ++v) {
            int4 input_vec = reinterpret_cast<const int4*>(input_row)[v];
            int4 weight_vec = reinterpret_cast<const int4*>(weight_row)[v];

            const int8_t* input_bytes = reinterpret_cast<const int8_t*>(&input_vec);
            const int8_t* weight_bytes = reinterpret_cast<const int8_t*>(&weight_vec);

            #pragma unroll
            for (int i = 0; i < VEC_SIZE; ++i) {
                acc += static_cast<int64_t>(input_bytes[i]) * static_cast<int64_t>(weight_bytes[i]);
                sum_x += static_cast<int64_t>(input_bytes[i]);
                sum_w += static_cast<int64_t>(weight_bytes[i]);
            }
        }
        start = vec_steps * VEC_SIZE;
    }

    // Remainder / scalar fallback elements
    for (int64_t i = start; i < in_features; ++i) {
        acc += static_cast<int64_t>(input_row[i]) * static_cast<int64_t>(weight_row[i]);
        sum_x += static_cast<int64_t>(input_row[i]);
        sum_w += static_cast<int64_t>(weight_row[i]);
    }

    // Zero point correction (int64 to avoid wraparound, matching CUDA):
    // Full expansion: sum((x_i - x_zp) * (w_j - w_zp))
    //   = sum(x_i * w_j) - x_zp * sum(w_j) - w_zp * sum(x_i) + x_zp * w_zp * K
    acc = acc - static_cast<int64_t>(input_zp) * sum_w
          - static_cast<int64_t>(w_zp) * sum_x
          + static_cast<int64_t>(input_zp) * static_cast<int64_t>(w_zp) * in_features;

    // Dequantize to float and add bias
    float result = static_cast<float>(acc) * out_scale;
    if (bias != nullptr) {
        result += bias[o];
    }

    output[b * out_features + o] = result;
}

/**
 * @brief Host wrapper for quantized linear (INT8 → INT32 accumulation → Float32 output).
 */
auto quantized_linear_hip(
    const Tensor& input,
    const Tensor& weight,
    const Tensor* bias,
    float input_scale,
    int32_t input_zero_point,
    float weight_scale,
    int32_t weight_zero_point,
    float output_scale,
    int32_t output_zero_point,
    hipStream_t stream,
    // Per-channel (F045): [out_features] weight scales / zero-points; nullptr => per-tensor.
    const Tensor* weight_scales,
    const Tensor* weight_zps
) -> Tensor {
    // The kernel flat-indexes input/weight from shape-derived row offsets
    // (input_row = input + b*in_features, weight_row = weight + o*in_features),
    // which is only valid for contiguous storage. Materialize contiguous copies
    // for non-contiguous views (transposed/sliced) to avoid reading at wrong
    // offsets, matching the roi_align/vision sibling kernels.
    Tensor input_c = input.is_contiguous() ? input : input.contiguous();
    Tensor weight_c = weight.is_contiguous() ? weight : weight.contiguous();

    auto input_shape = input_c.shape();
    auto weight_shape = weight_c.shape();
    int64_t batch_size = input_shape[0];
    int64_t in_features = input_shape[1];
    int64_t out_features = weight_shape[0];

    Tensor output({batch_size, out_features}, DType::Float32, input_c.device());

    // Output dequant scale = input_scale * weight_scale / output_scale, matching
    // the CPU (quantized_linear.cpp:78), CUDA (quantized_linear.cu:136) and
    // OneAPI (quantization.cpp:170) reference backends. The earlier code dropped
    // the `/ output_scale`, so for any dispatch with output_scale != 1.0 the
    // ROCm result diverged from every other backend by a factor of output_scale.
    // Guard against a zero output_scale like OneAPI does (safe_output_scale).
    const float safe_output_scale = (output_scale != 0.0f) ? output_scale : 1.0f;
    (void)output_zero_point;
    float combined_scale = input_scale * weight_scale / safe_output_scale;
    float input_over_output = input_scale / safe_output_scale;

    // Per-channel weight scale/zero-point (optional). Materialize contiguous so
    // the kernel can index by output channel.
    Tensor wscale_c, wzp_c;
    const float* wscale_ptr = nullptr;
    const int32_t* wzp_ptr = nullptr;
    if (weight_scales != nullptr && weight_scales->numel() > 1) {
        wscale_c = weight_scales->is_contiguous() ? *weight_scales : weight_scales->contiguous();
        wscale_ptr = wscale_c.data<float>();
        if (weight_zps != nullptr && weight_zps->numel() > 0) {
            wzp_c = weight_zps->is_contiguous() ? *weight_zps : weight_zps->contiguous();
            wzp_ptr = wzp_c.data<int32_t>();
        }
    }

    const int THREADS = 256;
    dim3 blocks((out_features + THREADS - 1) / THREADS, batch_size);
    dim3 threads(THREADS);

    hipLaunchKernelGGL(quantized_linear_kernel_hip,
        blocks, threads, 0, stream,
        input_c.data<int8_t>(),
        weight_c.data<int8_t>(),
        bias ? bias->data<float>() : nullptr,
        output.data<float>(),
        batch_size, in_features, out_features,
        combined_scale, input_zero_point, weight_zero_point,
        wscale_ptr, wzp_ptr, input_over_output);

    HIP_CHECK(hipGetLastError());
    return output;
}

// ==============================================================================
// INT4 (QInt4x2) quantized linear (F031 parity fix). Weights are packed two
// signed int4 per byte ([out_features, in_features/2]); the input is int8.
// Symmetric only (no weight zero-point), matching the CPU/CUDA int4 kernels
// (src/backends/cuda/kernels/quantization/quantized_linear.cu). Before this,
// ROCm had no dedicated int4 path: the registry unconditionally routed
// QInt4x2 weights into quantized_linear_hip above, which reads
// `in_features` int8 bytes per output row via weight_c.data<int8_t>() — for
// a QInt4x2 tensor the real backing buffer is only `in_features/2` bytes per
// row, so that was a heap OOB read of up to 2x the allocated buffer.
// ==============================================================================

__device__ __forceinline__ void unpack_int4_hip(uint8_t packed, int& low, int& high) {
    int lo = packed & 0x0F;
    if (lo & 0x08) lo |= ~0x0F;      // sign-extend bit 3
    int hi = (packed >> 4) & 0x0F;
    if (hi & 0x08) hi |= ~0x0F;
    low = lo;
    high = hi;
}

__global__ void quantized_linear_int4_kernel_hip(
    const int8_t* __restrict__ input,
    const uint8_t* __restrict__ weight_packed,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int64_t batch_size, int64_t in_features, int64_t out_features,
    float combined_scale,
    const float* __restrict__ weight_scales,  // per-channel (nullptr => per-tensor)
    float input_over_output,
    int32_t input_zp
) {
    int64_t b = blockIdx.y;
    int64_t o = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch_size || o >= out_features) return;

    int64_t packed_features = in_features / 2;
    const int8_t* input_row = input + b * in_features;
    const uint8_t* weight_row = weight_packed + o * packed_features;

    int64_t acc = 0;
    // sum_w: needed for the asymmetric activation zero-point correction below.
    // INT4 weights are symmetric (weight_zp==0 always), so unlike the INT8
    // kernel's full 4-term correction, only the -input_zp*sum_w cross term is
    // needed (mirrors CPU's fused_qlinear_dequant col_sum_w correction and the
    // CUDA int4 kernel).
    int64_t sum_w = 0;
    for (int64_t p = 0; p < packed_features; ++p) {
        int lo, hi;
        unpack_int4_hip(weight_row[p], lo, hi);
        acc += static_cast<int64_t>(input_row[p * 2]) * lo;
        acc += static_cast<int64_t>(input_row[p * 2 + 1]) * hi;
        sum_w += lo + hi;
    }

    int64_t corrected = acc;
    if (input_zp != 0) {
        corrected -= static_cast<int64_t>(input_zp) * sum_w;
    }

    float scale = weight_scales ? (input_over_output * weight_scales[o]) : combined_scale;
    float result = static_cast<float>(corrected) * scale;
    if (bias != nullptr) result += bias[o];
    output[b * out_features + o] = result;
}

/**
 * @brief Host wrapper for INT4 (QInt4x2 packed) quantized linear.
 */
auto quantized_linear_int4_hip(
    const Tensor& input,
    const Tensor& weight,
    const Tensor* bias,
    float input_scale,
    int32_t input_zero_point,
    float weight_scale,
    float output_scale,
    hipStream_t stream,
    const Tensor* weight_scales  // per-channel [out_features]; nullptr => per-tensor
) -> Tensor {
    Tensor input_c = input.is_contiguous() ? input : input.contiguous();
    Tensor weight_c = weight.is_contiguous() ? weight : weight.contiguous();

    auto input_shape = input_c.shape();
    auto weight_shape = weight_c.shape();
    int64_t batch_size = input_shape[0];
    int64_t in_features = input_shape[1];
    int64_t out_features = weight_shape[0];

    if (in_features % 2 != 0) {
        throw std::invalid_argument("quantized_linear_int4_hip: in_features must be even");
    }
    // F074 parity: match the CPU/CUDA int4 kernels' output_scale contract
    // (throws on output_scale <= 0) instead of silently substituting 1.0.
    if (output_scale <= 0.0f) {
        throw std::invalid_argument("quantized_linear_int4_hip: output_scale must be > 0");
    }

    Tensor output({batch_size, out_features}, DType::Float32, input_c.device());

    float combined_scale = input_scale * weight_scale / output_scale;
    float input_over_output = input_scale / output_scale;

    Tensor wscale_c;
    const float* wscale_ptr = nullptr;
    if (weight_scales != nullptr && weight_scales->numel() > 1) {
        wscale_c = weight_scales->is_contiguous() ? *weight_scales : weight_scales->contiguous();
        wscale_ptr = wscale_c.data<float>();
    }

    const int THREADS = 256;
    dim3 blocks((out_features + THREADS - 1) / THREADS, batch_size);
    dim3 threads(THREADS);

    hipLaunchKernelGGL(quantized_linear_int4_kernel_hip,
        blocks, threads, 0, stream,
        input_c.data<int8_t>(),
        reinterpret_cast<const uint8_t*>(weight_c.data_ptr()),
        bias ? bias->data<float>() : nullptr,
        output.data<float>(),
        batch_size, in_features, out_features,
        combined_scale, wscale_ptr, input_over_output, input_zero_point);

    HIP_CHECK(hipGetLastError());
    return output;
}

// ==============================================================================
// Quantized Conv2d — im2col + rocBLAS GEMM (INT8 → INT32 accumulation)
// ==============================================================================

// Helper macros (scoped to this translation unit)
#define ROCBLAS_CHECK(call) do { \
    rocblas_status status = call; \
    if (status != rocblas_status_success) { \
        throw std::runtime_error( \
            std::string("rocBLAS error at ") + __FILE__ + ":" + \
            std::to_string(__LINE__) + " - status " + std::to_string(status) \
        ); \
    } \
} while(0)

// Thread-local cached rocBLAS handle. rocblas_create_handle is expensive
// (device workspace alloc, property queries) and meant to be reused, so the
// hot quantized-conv path keeps one handle alive per thread and only sets the
// stream per call instead of create/destroy on every invocation.
namespace {
struct CachedQuantHandle {
    rocblas_handle handle = nullptr;
    ~CachedQuantHandle() {
        // Guard teardown: destroying after the backend library has unloaded
        // calls into freed code (mirrors the rocSPARSE/rocSOLVER pools).
        if (handle && is_backend_registry_alive()) {
            rocblas_destroy_handle(handle);
            handle = nullptr;
        }
    }
};
inline rocblas_handle get_cached_quant_handle(hipStream_t stream) {
    thread_local CachedQuantHandle cached;
    if (cached.handle == nullptr) {
        ROCBLAS_CHECK(rocblas_create_handle(&cached.handle));
    }
    ROCBLAS_CHECK(rocblas_set_stream(cached.handle, stream));
    return cached.handle;
}
}  // namespace

/**
 * @brief im2col kernel for Int8 input: unfolds input patches into a column matrix.
 *
 * Converts NCHW input into a 2D matrix where each row is one convolution patch.
 * Output shape: (batch * out_h * out_w, in_channels * kernel_h * kernel_w)
 * Padded positions are filled with the input zero point (not 0) so that the
 * subsequent GEMM correctly accounts for zero-point subtraction.
 */
__global__ void im2col_int8_kernel(
    const int8_t* __restrict__ input,
    int8_t* __restrict__ col_buffer,
    int64_t batch,
    int64_t in_channels,          // total input channels (full tensor stride)
    int64_t in_channels_per_group,// channels unfolded for this group
    int64_t channel_offset,       // first input channel of this group
    int64_t h_in,
    int64_t w_in,
    int64_t kernel_h,             // per-axis kernel size (F044: rectangular)
    int64_t kernel_w,
    int64_t stride_h,             // per-axis stride/padding/dilation (F044: asymmetric)
    int64_t stride_w,
    int64_t pad_h,
    int64_t pad_w,
    int64_t dil_h,
    int64_t dil_w,
    int64_t h_out,
    int64_t w_out,
    int8_t input_zp
) {
    int64_t total = batch * h_out * w_out * in_channels_per_group * kernel_h * kernel_w;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    // Decode flat index to (b, oh, ow, ic_local, kh, kw)
    int64_t temp = idx;
    int64_t kw = temp % kernel_w; temp /= kernel_w;
    int64_t kh = temp % kernel_h; temp /= kernel_h;
    int64_t ic_local = temp % in_channels_per_group; temp /= in_channels_per_group;
    int64_t ow = temp % w_out; temp /= w_out;
    int64_t oh = temp % h_out; temp /= h_out;
    int64_t b = temp;

    int64_t ih = oh * stride_h - pad_h + kh * dil_h;
    int64_t iw = ow * stride_w - pad_w + kw * dil_w;

    // Output index in column matrix (rows are per-group spatial patches)
    // Row: b * out_h * out_w + oh * out_w + ow
    // Col: ic_local * kernel_h * kernel_w + kh * kernel_w + kw
    int64_t col_cols = in_channels_per_group * kernel_h * kernel_w;
    int64_t out_row = b * h_out * w_out + oh * w_out + ow;
    int64_t out_col = ic_local * kernel_h * kernel_w + kh * kernel_w + kw;

    int8_t value = input_zp;  // Use zero point for out-of-bounds (padding)
    if (ih >= 0 && ih < h_in && iw >= 0 && iw < w_in) {
        // Map to the full input tensor: actual channel = channel_offset + ic_local
        int64_t ic = channel_offset + ic_local;
        value = input[((b * in_channels + ic) * h_in + ih) * w_in + iw];
    }
    col_buffer[out_row * col_cols + out_col] = value;
}

/**
 * @brief Dequantize Int32 GEMM output to Float32 and add bias.
 *
 * After the Int8 GEMM with Int32 accumulation, this kernel applies zero-point
 * correction, dequantization scaling, and bias addition.
 *
 * The GEMM computes: C[m][n] = sum_k (A[m][k] * B[k][n])
 * where A is im2col(input) with padding filled by input_zp, and B = weight^T.
 * So C already contains: sum_k (input_k * weight_k) with input_zp used for padding.
 *
 * The true quantized result needs:
 *   sum_k ((input_k - input_zp) * (weight_k - weight_zp))
 *   = C - input_zp * col_sum_w - weight_zp * row_sum_x + input_zp * weight_zp * K
 *
 * Since we set padding to input_zp in im2col, the GEMM result already correctly
 * handles the padding. We just need the standard zero-point correction.
 *
 * For simplicity and to avoid a separate reduction pass, we apply a per-output-channel
 * correction: precompute weight_col_sums on host and pass them in. This avoids the
 * triple-nested loop of the old kernel.
 */
// Per-row Int8 sum (rows × K) accumulated in Int32. Builds the zero-point
// correction terms (sum over each weight filter, and sum over each im2col row)
// entirely on-device — no host round-trip, no CPU fallback.
__global__ void quant_row_sum_kernel(
    const int8_t* __restrict__ mat,
    int32_t* __restrict__ row_sums,
    int64_t rows,
    int64_t K
) {
    int64_t r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    const int8_t* row_ptr = mat + r * K;
    int32_t s = 0;
    for (int64_t k = 0; k < K; ++k) {
        s += static_cast<int32_t>(row_ptr[k]);
    }
    row_sums[r] = s;
}

__global__ void dequantize_bias_kernel(
    const int32_t* __restrict__ gemm_output,
    float* __restrict__ output,
    const float* __restrict__ bias,
    const int32_t* __restrict__ weight_col_sums,  // [out_channels_per_group] sum_k weight[oc_local][k]
    const int32_t* __restrict__ row_sums,         // [M] sum_k im2col_row[m][k]
    int64_t total,                  // M * out_channels_per_group
    int64_t out_channels,           // full output channel count (NCHW stride)
    int64_t out_channels_per_group, // output channels produced by this group
    int64_t oc_base,                // first global output channel of this group
    int64_t spatial_size,           // h_out * w_out
    int64_t k_inner,                // in_channels_per_group * kernel_h * kernel_w
    int32_t input_zp,
    int32_t weight_zp,
    float combined_scale,
    // Per-channel (F045): [out_channels] weight scales / zero-points; nullptr => per-tensor.
    // Per-channel dequant scale = input_scale_pc * weight_scales[oc].
    const float* __restrict__ weight_scales,
    const int32_t* __restrict__ weight_zps,
    float input_scale_pc
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    // GEMM output layout for this group: (batch * h_out * w_out, out_channels_per_group)
    int64_t row = idx / out_channels_per_group;  // spatial position within batch
    int64_t oc_local = idx % out_channels_per_group;
    int64_t oc = oc_base + oc_local;             // global output channel

    // Per-channel weight zero-point / scale (each global output channel oc).
    const int32_t w_zp = weight_scales ? (weight_zps ? weight_zps[oc] : 0) : weight_zp;
    const float out_scale = weight_scales ? (input_scale_pc * weight_scales[oc])
                                          : combined_scale;

    // Zero-point correction (mirrors quantized_linear_kernel_hip):
    //   sum_k (A - input_zp)(B - weight_zp)
    //     = C - input_zp*sum_k B[oc] - weight_zp*sum_k A[row] + input_zp*weight_zp*K
    // A = im2col(input) (padding filled with input_zp), B = weight[oc].
    int64_t acc = static_cast<int64_t>(gemm_output[idx])
                  - static_cast<int64_t>(input_zp) * weight_col_sums[oc_local]
                  - static_cast<int64_t>(w_zp) * row_sums[row]
                  + static_cast<int64_t>(input_zp) * static_cast<int64_t>(w_zp) * k_inner;

    float result = static_cast<float>(acc) * out_scale;
    if (bias != nullptr) {
        result += bias[oc];
    }

    // Reorder from GEMM layout (batch*h_out*w_out, out_channels_per_group) to NCHW
    int64_t batch_idx = row / spatial_size;
    int64_t spatial_idx = row % spatial_size;
    int64_t nchw_idx = (batch_idx * out_channels + oc) * spatial_size + spatial_idx;

    output[nchw_idx] = result;
}

/**
 * @brief Host wrapper for quantized conv2d using im2col + rocBLAS GEMM.
 *
 * Strategy:
 * 1. im2col: unfold input patches into column matrix (Int8), padding = input_zp
 * 2. rocblas_gemm_ex: Int8 x Int8 → Int32 accumulation (hardware-accelerated on MI-series GPUs)
 * 3. Dequantize kernel: Int32 → Float32 with zero-point correction and bias
 *
 * This replaces the previous naive triple-nested-loop kernel with a BLAS-based
 * approach that leverages AMD's hardware Int8 dot product instructions.
 */
auto quantized_conv2d_hip(
    const Tensor& input,
    const Tensor& weight,
    const Tensor* bias,
    int64_t stride_h,
    int64_t stride_w,
    int64_t pad_h,
    int64_t pad_w,
    int64_t dil_h,
    int64_t dil_w,
    int64_t groups,
    float input_scale,
    int32_t input_zero_point,
    float weight_scale,
    int32_t weight_zero_point,
    float output_scale,
    int32_t output_zero_point,
    hipStream_t stream,
    // Per-channel (F045): [out_channels] weight scales / zero-points; nullptr => per-tensor.
    const Tensor* weight_scales,
    const Tensor* weight_zps
) -> Tensor {
    // im2col and the GEMM/dequant kernels flat-index input/weight/bias from
    // shape-derived offsets, which is only valid for contiguous storage.
    // Materialize contiguous copies for non-contiguous views, matching the
    // roi_align/vision sibling kernels.
    Tensor input_c = input.is_contiguous() ? input : input.contiguous();
    Tensor weight_c = weight.is_contiguous() ? weight : weight.contiguous();
    Tensor bias_c;
    const Tensor* bias_ptr = bias;
    if (bias != nullptr) {
        bias_c = bias->is_contiguous() ? *bias : bias->contiguous();
        bias_ptr = &bias_c;
    }

    auto input_shape = input_c.shape();
    int64_t batch = input_shape[0];
    int64_t in_channels = input_shape[1];
    int64_t h_in = input_shape[2];
    int64_t w_in = input_shape[3];

    auto weight_shape = weight_c.shape();
    int64_t out_channels = weight_shape[0];
    // Per-axis kernel size (F044: rectangular kernels).
    int64_t kernel_h = weight_shape[2];
    int64_t kernel_w = weight_shape[3];

    if (groups < 1)
        throw std::invalid_argument("quantized_conv2d: groups must be >= 1");
    if (in_channels % groups != 0)
        throw std::invalid_argument("quantized_conv2d: in_channels must be divisible by groups");
    if (out_channels % groups != 0)
        throw std::invalid_argument("quantized_conv2d: out_channels must be divisible by groups");

    int64_t h_out = (h_in + 2 * pad_h - dil_h * (kernel_h - 1) - 1) / stride_h + 1;
    int64_t w_out = (w_in + 2 * pad_w - dil_w * (kernel_w - 1) - 1) / stride_w + 1;

    Tensor output({batch, out_channels, h_out, w_out}, DType::Float32, input_c.device());

    // Dequantized Float32 output: scale = input_scale * weight_scale (NO division
    // by output_scale). This op produces a dequantized result, not a requantized
    // int8 tensor, so output_scale/output_zero_point are unused here — matching the
    // CPU (quantized_conv2d.cpp) and CUDA (quantized_conv2d.cu) reference backends.
    (void)output_scale;
    (void)output_zero_point;
    float combined_scale = input_scale * weight_scale;

    // Per-channel weight scale/zero-point (optional, F045). Materialize contiguous
    // device tensors so the dequantize kernel can index by global output channel.
    Tensor wscale_c, wzp_c;
    const float* wscale_ptr = nullptr;
    const int32_t* wzp_ptr = nullptr;
    if (weight_scales != nullptr && weight_scales->numel() > 1) {
        wscale_c = weight_scales->is_contiguous() ? *weight_scales : weight_scales->contiguous();
        wscale_ptr = wscale_c.data<float>();
        if (weight_zps != nullptr && weight_zps->numel() > 0) {
            wzp_c = weight_zps->is_contiguous() ? *weight_zps : weight_zps->contiguous();
            wzp_ptr = wzp_c.data<int32_t>();
        }
    }

    const int64_t in_channels_per_group = in_channels / groups;
    const int64_t out_channels_per_group = out_channels / groups;

    // Per-group dimensions for im2col and GEMM. Weight is stored as
    // (out_channels, in_channels_per_group, kh, kw) — i.e. each filter already
    // only spans its own group's input channels.
    int64_t M = batch * h_out * w_out;                                   // patches per group
    int64_t K = in_channels_per_group * kernel_h * kernel_w;             // patch size (per group)
    int64_t N = out_channels_per_group;                                  // filters per group

    const int THREADS = 256;
    int64_t spatial_size = h_out * w_out;

    // Buffers reused across groups (shape is identical per group).
    int8_t* col_buffer = nullptr;
    int32_t* weight_col_sums = nullptr;
    int32_t* row_sums = nullptr;
    int32_t* gemm_output = nullptr;
    HIP_CHECK(hipMalloc(&col_buffer, M * K * sizeof(int8_t)));
    HIP_CHECK(hipMalloc(&weight_col_sums, N * sizeof(int32_t)));
    HIP_CHECK(hipMalloc(&row_sums, M * sizeof(int32_t)));
    HIP_CHECK(hipMalloc(&gemm_output, M * N * sizeof(int32_t)));

    auto free_all = [&]() {
        if (col_buffer)      (void)hipFree(col_buffer);
        if (weight_col_sums) (void)hipFree(weight_col_sums);
        if (row_sums)        (void)hipFree(row_sums);
        if (gemm_output)     (void)hipFree(gemm_output);
    };

    try {
        rocblas_handle handle = get_cached_quant_handle(stream);
        int32_t alpha_i32 = 1;
        int32_t beta_i32 = 0;

        for (int64_t g = 0; g < groups; ++g) {
            int64_t ic_base = g * in_channels_per_group;
            int64_t oc_base = g * out_channels_per_group;

            // Step 1: im2col for this group's input channels.
            int64_t im2col_total =
                batch * h_out * w_out * in_channels_per_group * kernel_h * kernel_w;
            int im2col_blocks = static_cast<int>((im2col_total + THREADS - 1) / THREADS);

            hipLaunchKernelGGL(im2col_int8_kernel,
                dim3(im2col_blocks), dim3(THREADS), 0, stream,
                input_c.data<int8_t>(),
                col_buffer,
                batch, in_channels, in_channels_per_group, ic_base, h_in, w_in,
                kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, dil_h, dil_w,
                h_out, w_out,
                static_cast<int8_t>(input_zero_point));
            HIP_CHECK(hipGetLastError());

            // Step 1b: zero-point correction sums for this group.
            //   weight_col_sums[oc_local] = sum_k weight[oc_base+oc_local][k]
            //   row_sums[m]               = sum_k col_buffer[m][k] (padding = input_zp)
            const int8_t* weight_group = weight_c.data<int8_t>() + oc_base * K;

            hipLaunchKernelGGL(quant_row_sum_kernel,
                dim3(static_cast<int>((N + THREADS - 1) / THREADS)), dim3(THREADS), 0, stream,
                weight_group, weight_col_sums, N, K);
            HIP_CHECK(hipGetLastError());

            hipLaunchKernelGGL(quant_row_sum_kernel,
                dim3(static_cast<int>((M + THREADS - 1) / THREADS)), dim3(THREADS), 0, stream,
                col_buffer, row_sums, M, K);
            HIP_CHECK(hipGetLastError());

            // Step 2: GEMM via rocblas_gemm_ex — Int8 × Int8 → Int32
            //   C(M, N) = A(M, K) * B^T(K, N)  → rocBLAS: op(B)=T, op(A)=N
            ROCBLAS_CHECK(rocblas_gemm_ex(
                handle,
                rocblas_operation_transpose,
                rocblas_operation_none,
                N, M, K,
                &alpha_i32,
                weight_group,
                rocblas_datatype_i8_r,
                K,
                col_buffer,
                rocblas_datatype_i8_r,
                K,
                &beta_i32,
                gemm_output,
                rocblas_datatype_i32_r,
                N,
                gemm_output,
                rocblas_datatype_i32_r,
                N,
                rocblas_datatype_i32_r,
                rocblas_gemm_algo_standard,
                0, 0
            ));

            // The GEMM reads col_buffer and writes gemm_output asynchronously on
            // `stream`; the dequantize kernel below runs on the same stream, so it
            // is correctly ordered after the GEMM without an explicit sync. But the
            // NEXT group reuses col_buffer/row_sums/weight_col_sums/gemm_output, so
            // sync at the end of each iteration before they are overwritten.

            // Step 3: Dequantize Int32 → Float32 and add bias for this group.
            int64_t total_output = M * N;
            int dequant_blocks = static_cast<int>((total_output + THREADS - 1) / THREADS);

            hipLaunchKernelGGL(dequantize_bias_kernel,
                dim3(dequant_blocks), dim3(THREADS), 0, stream,
                gemm_output,
                output.data<float>(),
                bias_ptr ? bias_ptr->data<float>() : nullptr,
                weight_col_sums,
                row_sums,
                total_output,
                out_channels,
                out_channels_per_group,
                oc_base,
                spatial_size,
                K,
                input_zero_point,
                weight_zero_point,
                combined_scale,
                wscale_ptr,
                wzp_ptr,
                input_scale);
            HIP_CHECK(hipGetLastError());

            // Sync before the next group reuses the shared buffers.
            HIP_CHECK(hipStreamSynchronize(stream));
        }
    } catch (...) {
        free_all();
        throw;
    }

    free_all();
    return output;
}

} // namespace rocm
} // namespace tenzor
