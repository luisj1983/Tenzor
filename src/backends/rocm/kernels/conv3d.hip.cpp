#include "tenzor/core/tensor.hpp"
#include "tenzor/core/dtype.hpp"
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <rocblas/rocblas.h>
#include <stdexcept>
#include <vector>
#include <type_traits>
#include "../rocm_error.hpp"
#include "../hip_buffer.hpp"
#include "rocm_nan_helpers.hip.h"

namespace tenzor {
namespace rocm {

// Non-owning handle wrapper backed by a thread-local cached rocBLAS handle.
// rocblas_create_handle allocates device workspace and probes device
// properties, so creating/destroying it per forward/backward call (as this
// previously did) is pure per-layer/per-step overhead. Mirrors conv2d's
// get_cached_conv_handle: create once per thread, reuse, only rebind the
// stream. The cached handle is intentionally never destroyed (process-lifetime
// thread_local), matching the conv2d cache.
class RocBLASHandleGuardConv3d {
public:
    RocBLASHandleGuardConv3d() {
        struct CachedHandle {
            rocblas_handle h = nullptr;
            CachedHandle() { ROCBLAS_CHECK(rocblas_create_handle(&h)); }
            // No destructor: leak deliberately at process exit to avoid teardown
            // ordering issues with the rocBLAS/HIP runtime.
        };
        static thread_local CachedHandle cached;
        handle_ = cached.h;
    }
    RocBLASHandleGuardConv3d(const RocBLASHandleGuardConv3d&) = delete;
    RocBLASHandleGuardConv3d& operator=(const RocBLASHandleGuardConv3d&) = delete;
    rocblas_handle get() const { return handle_; }
    void set_stream(hipStream_t stream) {
        ROCBLAS_CHECK(rocblas_set_stream(handle_, stream));
    }
private:
    rocblas_handle handle_ = nullptr;
};

#define HIP_KERNEL_LOOP_CONV3D(i, n) \
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; \
         i < (n); \
         i += blockDim.x * gridDim.x)

inline void compute_launch_config_conv3d(int64_t n, dim3& grid, dim3& block) {
    const int block_size = 256;
    block = dim3(block_size, 1, 1);
    grid = dim3(static_cast<unsigned int>((n + block_size - 1) / block_size), 1, 1);
}

// Post-GEMM transpose: the rocBLAS call in the Conv3d forward produces
// a result laid out as (batch*D_out*H_out*W_out, out_channels_per_group)
// row-major = NDHWC — the output tensor is NCDHW, so we reshuffle.
// Matches the 2-D conv_nhwc_to_nchw_kernel in conv2d.hip.cpp.
template<typename T>
__global__ void conv3d_ndhwc_to_ncdhw_kernel(
    const T* __restrict__ ndhwc,
    T* __restrict__ ncdhw_out,
    int64_t batch, int64_t D_out, int64_t H_out, int64_t W_out,
    int64_t out_channels, int64_t channels_per_group, int64_t channel_offset
) {
    int64_t total_spatial = batch * D_out * H_out * W_out;
    int64_t total = total_spatial * channels_per_group;
    HIP_KERNEL_LOOP_CONV3D(idx, total) {
        int64_t c = idx % channels_per_group;
        int64_t spatial_idx = idx / channels_per_group;
        int64_t b = spatial_idx / (D_out * H_out * W_out);
        int64_t dhw = spatial_idx % (D_out * H_out * W_out);
        int64_t d = dhw / (H_out * W_out);
        int64_t hw = dhw % (H_out * W_out);
        int64_t h = hw / W_out;
        int64_t w = hw % W_out;
        int64_t global_c = channel_offset + c;
        int64_t ncdhw_idx = (((b * out_channels + global_c) * D_out + d) * H_out + h) * W_out + w;
        ncdhw_out[ncdhw_idx] = ndhwc[idx];
    }
}

// Inverse of conv3d_ndhwc_to_ncdhw_kernel. The Conv3d backward GEMMs feed
// grad_output to rocBLAS in a layout that, decoded, is NDHWC
// (batch*D_out*H_out*W_out, out_channels_per_group) row-major. But the
// grad_output tensor's real memory is NCDHW. This kernel gathers the
// per-group NCDHW grad_output into a contiguous NDHWC temp buffer so the
// backward consumes exactly the layout the forward produced.
template<typename T>
__global__ void conv3d_ncdhw_to_ndhwc_kernel(
    const T* __restrict__ ncdhw,
    T* __restrict__ ndhwc_out,
    int64_t batch, int64_t D_out, int64_t H_out, int64_t W_out,
    int64_t out_channels, int64_t channels_per_group, int64_t channel_offset
) {
    int64_t total_spatial = batch * D_out * H_out * W_out;
    int64_t total = total_spatial * channels_per_group;
    HIP_KERNEL_LOOP_CONV3D(idx, total) {
        int64_t c = idx % channels_per_group;
        int64_t spatial_idx = idx / channels_per_group;
        int64_t b = spatial_idx / (D_out * H_out * W_out);
        int64_t dhw = spatial_idx % (D_out * H_out * W_out);
        int64_t d = dhw / (H_out * W_out);
        int64_t hw = dhw % (H_out * W_out);
        int64_t h = hw / W_out;
        int64_t w = hw % W_out;
        int64_t global_c = channel_offset + c;
        int64_t ncdhw_idx = (((b * out_channels + global_c) * D_out + d) * H_out + h) * W_out + w;
        ndhwc_out[idx] = ncdhw[ncdhw_idx];
    }
}

// ============================================================================
// im3col HIP Kernel: Convert 5D input (N,C,D,H,W) to 2D for Conv3d GEMM
// ============================================================================
template<typename T>
__global__ void im3col_kernel_nchw(
    const T* __restrict__ input,
    T* __restrict__ output,
    int64_t batch,
    int64_t in_channels_total,   // full C_in for the per-batch stride (channels = per-group)
    int64_t channels,
    int64_t depth, int64_t height, int64_t width,
    int64_t kD, int64_t kH, int64_t kW,
    int64_t stride_d, int64_t stride_h, int64_t stride_w,
    int64_t pad_d, int64_t pad_h, int64_t pad_w,
    int64_t dil_d, int64_t dil_h, int64_t dil_w,
    int64_t out_d, int64_t out_h, int64_t out_w
) {
    int64_t col_cols = channels * kD * kH * kW;
    int64_t total_elements = batch * out_d * out_h * out_w * col_cols;

    HIP_KERNEL_LOOP_CONV3D(idx, total_elements) {
        int64_t tmp = idx;
        int64_t kw = tmp % kW; tmp /= kW;
        int64_t kh = tmp % kH; tmp /= kH;
        int64_t kd = tmp % kD; tmp /= kD;
        int64_t c  = tmp % channels; tmp /= channels;
        int64_t ow = tmp % out_w; tmp /= out_w;
        int64_t oh = tmp % out_h; tmp /= out_h;
        int64_t od = tmp % out_d; tmp /= out_d;
        int64_t b  = tmp;

        int64_t id = od * stride_d - pad_d + kd * dil_d;
        int64_t ih = oh * stride_h - pad_h + kh * dil_h;
        int64_t iw = ow * stride_w - pad_w + kw * dil_w;

        int64_t out_row = b * out_d * out_h * out_w + od * out_h * out_w + oh * out_w + ow;
        int64_t out_col = c * kD * kH * kW + kd * kH * kW + kh * kW + kw;
        int64_t out_idx = out_row * col_cols + out_col;

        T value = (id >= 0 && id < depth && ih >= 0 && ih < height && iw >= 0 && iw < width)
                  ? input[b * (in_channels_total * depth * height * width) +
                          c * (depth * height * width) +
                          id * (height * width) + ih * width + iw]
                  : T(0);
        output[out_idx] = value;
    }
}

// ============================================================================
// col3im HIP Kernel: Reverse of im3col (output-centric, no atomics)
// ============================================================================
template<typename T>
__global__ void col3im_kernel_nchw(
    const T* __restrict__ col,
    T* __restrict__ output,
    int64_t batch,
    int64_t in_channels_total,   // full C_in for the per-batch stride (channels = per-group)
    int64_t channels,
    int64_t depth, int64_t height, int64_t width,
    int64_t kD, int64_t kH, int64_t kW,
    int64_t stride_d, int64_t stride_h, int64_t stride_w,
    int64_t pad_d, int64_t pad_h, int64_t pad_w,
    int64_t dil_d, int64_t dil_h, int64_t dil_w,
    int64_t out_d, int64_t out_h, int64_t out_w
) {
    int64_t col_cols = channels * kD * kH * kW;
    int64_t total_output = batch * channels * depth * height * width;

    HIP_KERNEL_LOOP_CONV3D(output_idx, total_output) {
        int64_t tmp = output_idx;
        int64_t iw = tmp % width; tmp /= width;
        int64_t ih = tmp % height; tmp /= height;
        int64_t id = tmp % depth; tmp /= depth;
        int64_t c  = tmp % channels; tmp /= channels;
        int64_t b  = tmp;

        // Float accumulator for half precision; native otherwise (double stays double).
        using Acc = std::conditional_t<std::is_same_v<T, __half>, float, T>;
        Acc sum = Acc(0);

        for (int64_t kd = 0; kd < kD; ++kd) {
            for (int64_t kh = 0; kh < kH; ++kh) {
                for (int64_t kw_iter = 0; kw_iter < kW; ++kw_iter) {
                    int64_t id_shifted = id + pad_d - kd * dil_d;
                    int64_t ih_shifted = ih + pad_h - kh * dil_h;
                    int64_t iw_shifted = iw + pad_w - kw_iter * dil_w;

                    if (id_shifted % stride_d == 0 && ih_shifted % stride_h == 0 && iw_shifted % stride_w == 0) {
                        int64_t od = id_shifted / stride_d;
                        int64_t oh = ih_shifted / stride_h;
                        int64_t ow = iw_shifted / stride_w;

                        if (od >= 0 && od < out_d && oh >= 0 && oh < out_h && ow >= 0 && ow < out_w) {
                            int64_t col_row = b * out_d * out_h * out_w + od * out_h * out_w + oh * out_w + ow;
                            int64_t col_col = c * kD * kH * kW + kd * kH * kW + kh * kW + kw_iter;
                            T cv = col[col_row * col_cols + col_col];
                            if constexpr (std::is_same_v<T, __half>) {
                                sum += tenzor::rocm::safe_h2f(cv);
                            } else {
                                sum += cv;
                            }
                        }
                    }
                }
            }
        }

        // The batch loop decoded output_idx with the per-group `channels` stride,
        // but grad_input is laid out with the full C_in stride; correct the batch
        // term so batch>0 with groups>1 writes the right channel block.
        int64_t out_write = output_idx + b * (in_channels_total - channels) * depth * height * width;
        if constexpr (std::is_same_v<T, __half>) {
            output[out_write] = tenzor::rocm::safe_f2h(sum);
        } else {
            output[out_write] = static_cast<T>(sum);
        }
    }
}

// ============================================================================
// Bias addition kernels for 5D tensors (N, C, D, H, W)
// ============================================================================
__global__ void add_bias_3d_kernel(
    float* __restrict__ output,
    const float* __restrict__ bias,
    int64_t channels,
    int64_t spatial_size,
    int64_t n
) {
    HIP_KERNEL_LOOP_CONV3D(idx, n) {
        int64_t c = (idx / spatial_size) % channels;
        output[idx] += bias[c];
    }
}

__global__ void add_bias_3d_kernel_double(
    double* __restrict__ output,
    const double* __restrict__ bias,
    int64_t channels,
    int64_t spatial_size,
    int64_t n
) {
    HIP_KERNEL_LOOP_CONV3D(idx, n) {
        int64_t c = (idx / spatial_size) % channels;
        output[idx] += bias[c];
    }
}

__global__ void add_bias_3d_kernel_half(
    __half* __restrict__ output,
    const __half* __restrict__ bias,
    int64_t channels,
    int64_t spatial_size,
    int64_t n
) {
    HIP_KERNEL_LOOP_CONV3D(idx, n) {
        int64_t c = (idx / spatial_size) % channels;
        output[idx] = __hadd(output[idx], bias[c]);
    }
}

// ============================================================================
// Bias gradient kernel for 5D tensors (wavefront-optimized reduction)
// ============================================================================
template<typename T>
__global__ void sum_bias_grad_3d_kernel(
    const T* __restrict__ grad_output,
    T* __restrict__ grad_bias,
    int64_t batch,
    int64_t channels,
    int64_t spatial_size
) {
    // Accumulate in an accumulator type wide enough for T: double for Float64,
    // float otherwise. Hardcoding float would truncate every Float64 grad_output
    // value to single precision, degrading Float64 bias gradients to ~7 digits.
    using Acc = std::conditional_t<std::is_same_v<T, double>, double, float>;
    __shared__ Acc shared_data[256];

    int64_t c = blockIdx.x;
    if (c < channels) {
        int64_t tid = threadIdx.x;
        int64_t block_size = blockDim.x;

        Acc local_sum = Acc(0);
        for (int64_t idx = tid; idx < batch * spatial_size; idx += block_size) {
            int64_t b = idx / spatial_size;
            int64_t s = idx % spatial_size;
            int64_t grad_idx = b * (channels * spatial_size) + c * spatial_size + s;
            local_sum += static_cast<Acc>(grad_output[grad_idx]);
        }

        shared_data[tid] = local_sum;
        __syncthreads();

        for (int64_t s = block_size / 2; s > 0; s >>= 1) {
            if (tid < s) {
                shared_data[tid] += shared_data[tid + s];
            }
            __syncthreads();
        }

        if (tid == 0) {
            grad_bias[c] = static_cast<T>(shared_data[0]);
        }
    }
}

// ============================================================================
// Helper: calculate output size
// ============================================================================
static inline int64_t calc_out_size_3d(int64_t in, int64_t kernel, int64_t stride,
                                        int64_t padding, int64_t dilation) {
    return (in + 2 * padding - dilation * (kernel - 1) - 1) / stride + 1;
}

// ============================================================================
// Conv3d Forward: im3col + rocBLAS GEMM
// ============================================================================
auto conv3d_forward_hip(
    const Tensor& input,         // (N, C_in, D, H, W)
    const Tensor& weight,        // (C_out, C_in/groups, kD, kH, kW)
    const Tensor& bias,          // (C_out) or empty
    const std::vector<int64_t>& stride,
    const std::vector<int64_t>& padding,
    const std::vector<int64_t>& dilation,
    int64_t groups,
    hipStream_t stream
) -> Tensor {
    // BFloat16/Float16: upcast to Float32, compute, convert back. The deeper
    // rocblas_hgemm branches accumulate the GEMM reduction (K = C_in_per_group *
    // kD*kH*kW, often >1000) entirely in FP16, which overflows past 65504 and
    // loses 2-3 decimal digits vs the CPU/CUDA FP32-accumulate reference, so we
    // never reach them for half-precision input.
    if (input.dtype() == DType::BFloat16 || input.dtype() == DType::Float16) {
        const DType orig = input.dtype();
        auto input_f32 = input.to(DType::Float32);
        auto weight_f32 = weight.to(DType::Float32);
        auto bias_f32 = bias.numel() > 0 ? bias.to(DType::Float32) : bias;
        auto result = conv3d_forward_hip(input_f32, weight_f32, bias_f32,
                                          stride, padding, dilation, groups, stream);
        return result.to(orig);
    }

    auto input_shape = input.shape();
    auto weight_shape = weight.shape();
    const DType dtype = input.dtype();

    if (input_shape.size() != 5) throw std::invalid_argument("Conv3d: input must be 5D (N,C,D,H,W)");
    if (weight_shape.size() != 5) throw std::invalid_argument("Conv3d: weight must be 5D");

    const int64_t N = input_shape[0];
    const int64_t C_in = input_shape[1];
    const int64_t D_in = input_shape[2];
    const int64_t H_in = input_shape[3];
    const int64_t W_in = input_shape[4];

    const int64_t C_out = weight_shape[0];
    const int64_t C_in_per_group = weight_shape[1];
    const int64_t kD = weight_shape[2];
    const int64_t kH = weight_shape[3];
    const int64_t kW = weight_shape[4];

    const int64_t stride_d = stride.size() > 0 ? stride[0] : 1;
    const int64_t stride_h = stride.size() > 1 ? stride[1] : stride_d;
    const int64_t stride_w = stride.size() > 2 ? stride[2] : stride_h;

    const int64_t pad_d = padding.size() > 0 ? padding[0] : 0;
    const int64_t pad_h = padding.size() > 1 ? padding[1] : pad_d;
    const int64_t pad_w = padding.size() > 2 ? padding[2] : pad_h;

    const int64_t dil_d = dilation.size() > 0 ? dilation[0] : 1;
    const int64_t dil_h = dilation.size() > 1 ? dilation[1] : dil_d;
    const int64_t dil_w = dilation.size() > 2 ? dilation[2] : dil_h;

    const int64_t D_out = calc_out_size_3d(D_in, kD, stride_d, pad_d, dil_d);
    const int64_t H_out = calc_out_size_3d(H_in, kH, stride_h, pad_h, dil_h);
    const int64_t W_out = calc_out_size_3d(W_in, kW, stride_w, pad_w, dil_w);

    Tensor output({N, C_out, D_out, H_out, W_out}, dtype, input.device());

    size_t elem_size = (dtype == DType::Float64) ? sizeof(double) :
                       (dtype == DType::Float16) ? sizeof(__half) : sizeof(float);
    HIP_CHECK(hipMemsetAsync(output.data_ptr(), 0, output.numel() * elem_size, stream));

    RocBLASHandleGuardConv3d rocblas_guard;
    rocblas_handle handle = rocblas_guard.get();
    rocblas_guard.set_stream(stream);

    const int64_t C_out_per_group = C_out / groups;
    const int64_t col_rows = N * D_out * H_out * W_out;
    const int64_t col_cols = C_in_per_group * kD * kH * kW;

    for (int64_t g = 0; g < groups; ++g) {
        int64_t in_start = g * C_in_per_group;
        int64_t out_start = g * C_out_per_group;

        dim3 grid, block;
        int64_t total_elements = N * D_out * H_out * W_out * C_in_per_group * kD * kH * kW;
        compute_launch_config_conv3d(total_elements, grid, block);

        int64_t M = col_rows;
        int64_t K = col_cols;
        int64_t N_gemm = C_out_per_group;

        if (dtype == DType::Float32) {
            HipBuffer col_buf(col_rows * col_cols * sizeof(float));
            float* col_buffer = col_buf.as<float>();

            const float* input_ptr = input.data<float>() + in_start * D_in * H_in * W_in;
            im3col_kernel_nchw<<<grid, block, 0, stream>>>(
                input_ptr, col_buffer, N, C_in, C_in_per_group,
                D_in, H_in, W_in, kD, kH, kW,
                stride_d, stride_h, stride_w,
                pad_d, pad_h, pad_w,
                dil_d, dil_h, dil_w,
                D_out, H_out, W_out
            );
            HIP_CHECK(hipGetLastError());

            float alpha = 1.0f, beta = 0.0f;
            const float* weight_ptr = weight.data<float>() + out_start * C_in_per_group * kD * kH * kW;

            // rocBLAS's output layout (N_gemm, M) col-major is equivalent
            // to (M, N_gemm) row-major, i.e. NDHWC-ordered. The Conv3d
            // output tensor is NCDHW — write to a temp buffer and
            // post-transpose into NCDHW. Writing straight into the NCDHW
            // pointer (the previous implementation) silently corrupted
            // the layout, making outputs differ from CPU by up to ~7.
            HipBuffer gemm_out_buf(M * N_gemm * sizeof(float));
            float* gemm_out = gemm_out_buf.as<float>();

            ROCBLAS_CHECK(rocblas_sgemm(
                handle,
                rocblas_operation_transpose, rocblas_operation_none,
                N_gemm, M, K,
                &alpha,
                weight_ptr, K,
                col_buffer, K,
                &beta,
                gemm_out, N_gemm
            ));

            dim3 t_grid, t_block;
            compute_launch_config_conv3d(M * N_gemm, t_grid, t_block);
            conv3d_ndhwc_to_ncdhw_kernel<float><<<t_grid, t_block, 0, stream>>>(
                gemm_out, output.data<float>(),
                N, D_out, H_out, W_out, C_out, C_out_per_group, out_start
            );
            HIP_CHECK(hipGetLastError());

        } else if (dtype == DType::Float64) {
            HipBuffer col_buf(col_rows * col_cols * sizeof(double));
            double* col_buffer = col_buf.as<double>();

            const double* input_ptr = input.data<double>() + in_start * D_in * H_in * W_in;
            im3col_kernel_nchw<<<grid, block, 0, stream>>>(
                input_ptr, col_buffer, N, C_in, C_in_per_group,
                D_in, H_in, W_in, kD, kH, kW,
                stride_d, stride_h, stride_w,
                pad_d, pad_h, pad_w,
                dil_d, dil_h, dil_w,
                D_out, H_out, W_out
            );
            HIP_CHECK(hipGetLastError());

            double alpha = 1.0, beta = 0.0;
            const double* weight_ptr = weight.data<double>() + out_start * C_in_per_group * kD * kH * kW;

            HipBuffer gemm_out_buf(M * N_gemm * sizeof(double));
            double* gemm_out = gemm_out_buf.as<double>();

            ROCBLAS_CHECK(rocblas_dgemm(
                handle,
                rocblas_operation_transpose, rocblas_operation_none,
                N_gemm, M, K,
                &alpha,
                weight_ptr, K,
                col_buffer, K,
                &beta,
                gemm_out, N_gemm
            ));

            dim3 t_grid, t_block;
            compute_launch_config_conv3d(M * N_gemm, t_grid, t_block);
            conv3d_ndhwc_to_ncdhw_kernel<double><<<t_grid, t_block, 0, stream>>>(
                gemm_out, output.data<double>(),
                N, D_out, H_out, W_out, C_out, C_out_per_group, out_start
            );
            HIP_CHECK(hipGetLastError());

        } else if (dtype == DType::Float16) {
            static_assert(sizeof(Float16) == sizeof(__half) && alignof(Float16) == alignof(__half),
                          "Float16 and __half must have identical size and alignment");
            HipBuffer col_buf(col_rows * col_cols * sizeof(__half));
            __half* col_buffer = col_buf.as<__half>();

            const __half* input_ptr = reinterpret_cast<const __half*>(input.data<Float16>()) + in_start * D_in * H_in * W_in;
            im3col_kernel_nchw<<<grid, block, 0, stream>>>(
                input_ptr, col_buffer, N, C_in, C_in_per_group,
                D_in, H_in, W_in, kD, kH, kW,
                stride_d, stride_h, stride_w,
                pad_d, pad_h, pad_w,
                dil_d, dil_h, dil_w,
                D_out, H_out, W_out
            );
            HIP_CHECK(hipGetLastError());

            rocblas_half alpha_h{static_cast<uint16_t>(0x3C00)};  // 1.0 in FP16
            rocblas_half beta_h{static_cast<uint16_t>(0x0000)};   // 0.0 in FP16

            const rocblas_half* weight_ptr = reinterpret_cast<const rocblas_half*>(weight.data<Float16>()) + out_start * C_in_per_group * kD * kH * kW;

            HipBuffer gemm_out_buf(M * N_gemm * sizeof(__half));
            __half* gemm_out = gemm_out_buf.as<__half>();

            ROCBLAS_CHECK(rocblas_hgemm(
                handle,
                rocblas_operation_transpose, rocblas_operation_none,
                N_gemm, M, K,
                &alpha_h,
                weight_ptr, K,
                reinterpret_cast<const rocblas_half*>(col_buffer), K,
                &beta_h,
                reinterpret_cast<rocblas_half*>(gemm_out), N_gemm
            ));

            dim3 t_grid, t_block;
            compute_launch_config_conv3d(M * N_gemm, t_grid, t_block);
            conv3d_ndhwc_to_ncdhw_kernel<__half><<<t_grid, t_block, 0, stream>>>(
                gemm_out,
                reinterpret_cast<__half*>(output.data<Float16>()),
                N, D_out, H_out, W_out, C_out, C_out_per_group, out_start
            );
            HIP_CHECK(hipGetLastError());

        } else {
            throw std::runtime_error("Conv3d: unsupported dtype (only Float32, Float64, Float16)");
        }
    }

    // Add bias
    bool has_bias = bias.numel() > 0;
    if (has_bias) {
        int64_t spatial_size = D_out * H_out * W_out;
        int64_t total = N * C_out * spatial_size;
        dim3 grid, block;
        compute_launch_config_conv3d(total, grid, block);

        if (dtype == DType::Float32) {
            add_bias_3d_kernel<<<grid, block, 0, stream>>>(
                output.data<float>(), bias.data<float>(), C_out, spatial_size, total);
        } else if (dtype == DType::Float64) {
            add_bias_3d_kernel_double<<<grid, block, 0, stream>>>(
                output.data<double>(), bias.data<double>(), C_out, spatial_size, total);
        } else if (dtype == DType::Float16) {
            add_bias_3d_kernel_half<<<grid, block, 0, stream>>>(
                reinterpret_cast<__half*>(output.data<Float16>()),
                reinterpret_cast<const __half*>(bias.data<Float16>()),
                C_out, spatial_size, total);
        }
        HIP_CHECK(hipGetLastError());
    }

    return output;
}

// ============================================================================
// Conv3d Backward Input: col3im + GEMM
// ============================================================================
auto conv3d_backward_input_hip(
    const Tensor& grad_output,
    const Tensor& weight,
    const std::vector<int64_t>& input_shape,
    const std::vector<int64_t>& stride,
    const std::vector<int64_t>& padding,
    const std::vector<int64_t>& dilation,
    int64_t groups,
    hipStream_t stream
) -> Tensor {
    // BFloat16/Float16: upcast to Float32, compute, convert back (the deeper
    // rocblas_hgemm branch accumulates in FP16; see conv3d_forward_hip comment).
    if (grad_output.dtype() == DType::BFloat16 || grad_output.dtype() == DType::Float16) {
        const DType orig = grad_output.dtype();
        auto grad_output_f32 = grad_output.to(DType::Float32);
        auto weight_f32 = weight.to(DType::Float32);
        auto result = conv3d_backward_input_hip(grad_output_f32, weight_f32, input_shape,
                                                 stride, padding, dilation, groups, stream);
        return result.to(orig);
    }

    auto weight_shape = weight.shape();
    auto grad_shape = grad_output.shape();
    const DType dtype = grad_output.dtype();

    const int64_t N = input_shape[0];
    const int64_t C_in = input_shape[1];
    const int64_t D_in = input_shape[2];
    const int64_t H_in = input_shape[3];
    const int64_t W_in = input_shape[4];

    const int64_t C_out = weight_shape[0];
    const int64_t C_in_per_group = weight_shape[1];
    const int64_t kD = weight_shape[2];
    const int64_t kH = weight_shape[3];
    const int64_t kW = weight_shape[4];

    const int64_t C_out_per_group = C_out / groups;

    const int64_t D_out = grad_shape[2];
    const int64_t H_out = grad_shape[3];
    const int64_t W_out = grad_shape[4];

    const int64_t stride_d = stride.size() > 0 ? stride[0] : 1;
    const int64_t stride_h = stride.size() > 1 ? stride[1] : stride_d;
    const int64_t stride_w = stride.size() > 2 ? stride[2] : stride_h;

    const int64_t pad_d = padding.size() > 0 ? padding[0] : 0;
    const int64_t pad_h = padding.size() > 1 ? padding[1] : pad_d;
    const int64_t pad_w = padding.size() > 2 ? padding[2] : pad_h;

    const int64_t dil_d = dilation.size() > 0 ? dilation[0] : 1;
    const int64_t dil_h = dilation.size() > 1 ? dilation[1] : dil_d;
    const int64_t dil_w = dilation.size() > 2 ? dilation[2] : dil_h;

    Tensor grad_input(input_shape, dtype, grad_output.device());

    const int64_t col_rows = N * D_out * H_out * W_out;
    const int64_t col_cols = C_in_per_group * kD * kH * kW;

    RocBLASHandleGuardConv3d rocblas_guard;
    rocblas_handle handle = rocblas_guard.get();
    rocblas_guard.set_stream(stream);

    for (int64_t g = 0; g < groups; ++g) {
        int64_t in_start = g * C_in_per_group;
        int64_t out_start = g * C_out_per_group;

        // GEMM: grad_col = grad_output @ weight (transpose)
        // grad_output group: (col_rows, C_out_per_group) row-major
        // weight group: (C_out_per_group, col_cols) row-major
        // grad_col: (col_rows, col_cols) row-major
        int64_t M_gemm = col_rows;
        int64_t K_gemm = C_out_per_group;
        int64_t N_col = col_cols;

        if (dtype == DType::Float32) {
            HipBuffer grad_col_buf(col_rows * col_cols * sizeof(float));
            float* grad_col = grad_col_buf.as<float>();

            float alpha = 1.0f, beta = 0.0f;
            // rocBLAS reads grad_out as op(B) column-major [K×M] with ld=K, i.e.
            // row-major [col_row][oc] = NDHWC. Permute NCDHW grad_output first.
            HipBuffer grad_out_ndhwc_buf(col_rows * C_out_per_group * sizeof(float));
            float* grad_out_ndhwc = grad_out_ndhwc_buf.as<float>();
            {
                dim3 p_grid, p_block;
                compute_launch_config_conv3d(col_rows * C_out_per_group, p_grid, p_block);
                conv3d_ncdhw_to_ndhwc_kernel<float><<<p_grid, p_block, 0, stream>>>(
                    grad_output.data<float>(), grad_out_ndhwc,
                    N, D_out, H_out, W_out, C_out, C_out_per_group, out_start
                );
                HIP_CHECK(hipGetLastError());
            }
            const float* grad_out_ptr = grad_out_ndhwc;
            const float* weight_ptr = weight.data<float>() + out_start * C_in_per_group * kD * kH * kW;

            // rocBLAS col-major: C = A * B^T => row-major: C = B * A^T
            ROCBLAS_CHECK(rocblas_sgemm(
                handle,
                rocblas_operation_none, rocblas_operation_none,
                N_col, M_gemm, K_gemm,
                &alpha,
                weight_ptr, N_col,
                grad_out_ptr, K_gemm,
                &beta,
                grad_col, N_col
            ));

            // col3im to scatter grad_col back to grad_input
            int64_t total_output = N * C_in_per_group * D_in * H_in * W_in;
            dim3 grid, block;
            compute_launch_config_conv3d(total_output, grid, block);

            float* grad_input_ptr = grad_input.data<float>() + in_start * D_in * H_in * W_in;
            col3im_kernel_nchw<<<grid, block, 0, stream>>>(
                grad_col, grad_input_ptr,
                N, C_in, C_in_per_group,
                D_in, H_in, W_in, kD, kH, kW,
                stride_d, stride_h, stride_w,
                pad_d, pad_h, pad_w,
                dil_d, dil_h, dil_w,
                D_out, H_out, W_out
            );
            HIP_CHECK(hipGetLastError());

        } else if (dtype == DType::Float64) {
            HipBuffer grad_col_buf(col_rows * col_cols * sizeof(double));
            double* grad_col = grad_col_buf.as<double>();

            double alpha = 1.0, beta = 0.0;
            // rocBLAS reads grad_out as op(B) column-major [K×M] with ld=K, i.e.
            // row-major [col_row][oc] = NDHWC. Permute NCDHW grad_output first.
            HipBuffer grad_out_ndhwc_buf(col_rows * C_out_per_group * sizeof(double));
            double* grad_out_ndhwc = grad_out_ndhwc_buf.as<double>();
            {
                dim3 p_grid, p_block;
                compute_launch_config_conv3d(col_rows * C_out_per_group, p_grid, p_block);
                conv3d_ncdhw_to_ndhwc_kernel<double><<<p_grid, p_block, 0, stream>>>(
                    grad_output.data<double>(), grad_out_ndhwc,
                    N, D_out, H_out, W_out, C_out, C_out_per_group, out_start
                );
                HIP_CHECK(hipGetLastError());
            }
            const double* grad_out_ptr = grad_out_ndhwc;
            const double* weight_ptr = weight.data<double>() + out_start * C_in_per_group * kD * kH * kW;

            ROCBLAS_CHECK(rocblas_dgemm(
                handle,
                rocblas_operation_none, rocblas_operation_none,
                N_col, M_gemm, K_gemm,
                &alpha,
                weight_ptr, N_col,
                grad_out_ptr, K_gemm,
                &beta,
                grad_col, N_col
            ));

            int64_t total_output = N * C_in_per_group * D_in * H_in * W_in;
            dim3 grid, block;
            compute_launch_config_conv3d(total_output, grid, block);

            double* grad_input_ptr = grad_input.data<double>() + in_start * D_in * H_in * W_in;
            col3im_kernel_nchw<<<grid, block, 0, stream>>>(
                grad_col, grad_input_ptr,
                N, C_in, C_in_per_group,
                D_in, H_in, W_in, kD, kH, kW,
                stride_d, stride_h, stride_w,
                pad_d, pad_h, pad_w,
                dil_d, dil_h, dil_w,
                D_out, H_out, W_out
            );
            HIP_CHECK(hipGetLastError());

        } else if (dtype == DType::Float16) {
            HipBuffer grad_col_buf(col_rows * col_cols * sizeof(rocblas_half));
            rocblas_half* grad_col = grad_col_buf.as<rocblas_half>();

            rocblas_half alpha_h{static_cast<uint16_t>(0x3C00)};
            rocblas_half beta_h{static_cast<uint16_t>(0x0000)};

            // rocBLAS reads grad_out as op(B) column-major [K×M] with ld=K, i.e.
            // row-major [col_row][oc] = NDHWC. Permute NCDHW grad_output first.
            HipBuffer grad_out_ndhwc_buf(col_rows * C_out_per_group * sizeof(rocblas_half));
            rocblas_half* grad_out_ndhwc = grad_out_ndhwc_buf.as<rocblas_half>();
            {
                dim3 p_grid, p_block;
                compute_launch_config_conv3d(col_rows * C_out_per_group, p_grid, p_block);
                conv3d_ncdhw_to_ndhwc_kernel<__half><<<p_grid, p_block, 0, stream>>>(
                    reinterpret_cast<const __half*>(grad_output.data<Float16>()),
                    reinterpret_cast<__half*>(grad_out_ndhwc),
                    N, D_out, H_out, W_out, C_out, C_out_per_group, out_start
                );
                HIP_CHECK(hipGetLastError());
            }
            const rocblas_half* grad_out_ptr = grad_out_ndhwc;
            const rocblas_half* weight_ptr = reinterpret_cast<const rocblas_half*>(weight.data<Float16>()) + out_start * C_in_per_group * kD * kH * kW;

            ROCBLAS_CHECK(rocblas_hgemm(
                handle,
                rocblas_operation_none, rocblas_operation_none,
                N_col, M_gemm, K_gemm,
                &alpha_h,
                weight_ptr, N_col,
                grad_out_ptr, K_gemm,
                &beta_h,
                grad_col, N_col
            ));

            int64_t total_output = N * C_in_per_group * D_in * H_in * W_in;
            dim3 grid, block;
            compute_launch_config_conv3d(total_output, grid, block);

            __half* grad_input_ptr = reinterpret_cast<__half*>(grad_input.data<Float16>()) + in_start * D_in * H_in * W_in;
            col3im_kernel_nchw<<<grid, block, 0, stream>>>(
                reinterpret_cast<const __half*>(grad_col), grad_input_ptr,
                N, C_in, C_in_per_group,
                D_in, H_in, W_in, kD, kH, kW,
                stride_d, stride_h, stride_w,
                pad_d, pad_h, pad_w,
                dil_d, dil_h, dil_w,
                D_out, H_out, W_out
            );
            HIP_CHECK(hipGetLastError());
        } else {
            throw std::runtime_error("Conv3d backward input: unsupported dtype");
        }
    }

    return grad_input;
}

// ============================================================================
// Conv3d Backward Weight: im3col(input) * grad_output^T
// ============================================================================
auto conv3d_backward_weight_hip(
    const Tensor& grad_output,
    const Tensor& input,
    const std::vector<int64_t>& weight_shape,
    const std::vector<int64_t>& stride,
    const std::vector<int64_t>& padding,
    const std::vector<int64_t>& dilation,
    int64_t groups,
    hipStream_t stream
) -> Tensor {
    // BFloat16/Float16: upcast to Float32, compute, convert back (the deeper
    // rocblas_hgemm branch accumulates in FP16; see conv3d_forward_hip comment).
    if (input.dtype() == DType::BFloat16 || input.dtype() == DType::Float16) {
        const DType orig = input.dtype();
        auto grad_output_f32 = grad_output.to(DType::Float32);
        auto input_f32 = input.to(DType::Float32);
        auto result = conv3d_backward_weight_hip(grad_output_f32, input_f32, weight_shape,
                                                  stride, padding, dilation, groups, stream);
        return result.to(orig);
    }

    auto input_shape = input.shape();
    auto grad_shape = grad_output.shape();
    const DType dtype = input.dtype();

    const int64_t N = input_shape[0];
    const int64_t C_in = input_shape[1];
    const int64_t D_in = input_shape[2];
    const int64_t H_in = input_shape[3];
    const int64_t W_in = input_shape[4];

    const int64_t C_out = weight_shape[0];
    const int64_t C_in_per_group = weight_shape[1];
    const int64_t kD = weight_shape[2];
    const int64_t kH = weight_shape[3];
    const int64_t kW = weight_shape[4];

    const int64_t C_out_per_group = C_out / groups;

    const int64_t D_out = grad_shape[2];
    const int64_t H_out = grad_shape[3];
    const int64_t W_out = grad_shape[4];

    const int64_t stride_d = stride.size() > 0 ? stride[0] : 1;
    const int64_t stride_h = stride.size() > 1 ? stride[1] : stride_d;
    const int64_t stride_w = stride.size() > 2 ? stride[2] : stride_h;

    const int64_t pad_d = padding.size() > 0 ? padding[0] : 0;
    const int64_t pad_h = padding.size() > 1 ? padding[1] : pad_d;
    const int64_t pad_w = padding.size() > 2 ? padding[2] : pad_h;

    const int64_t dil_d = dilation.size() > 0 ? dilation[0] : 1;
    const int64_t dil_h = dilation.size() > 1 ? dilation[1] : dil_d;
    const int64_t dil_w = dilation.size() > 2 ? dilation[2] : dil_h;

    Tensor grad_weight(weight_shape, dtype, input.device());
    size_t elem_size = (dtype == DType::Float64) ? sizeof(double) :
                       (dtype == DType::Float16) ? sizeof(__half) : sizeof(float);
    HIP_CHECK(hipMemsetAsync(grad_weight.data_ptr(), 0, grad_weight.numel() * elem_size, stream));

    RocBLASHandleGuardConv3d rocblas_guard;
    rocblas_handle handle = rocblas_guard.get();
    rocblas_guard.set_stream(stream);

    const int64_t col_rows = N * D_out * H_out * W_out;
    const int64_t col_cols = C_in_per_group * kD * kH * kW;

    for (int64_t g = 0; g < groups; ++g) {
        int64_t in_start = g * C_in_per_group;
        int64_t out_start = g * C_out_per_group;

        dim3 grid, block;
        int64_t total_elements = N * D_out * H_out * W_out * C_in_per_group * kD * kH * kW;
        compute_launch_config_conv3d(total_elements, grid, block);

        // GEMM: grad_weight = grad_output^T @ im3col(input)
        // grad_output group: (col_rows, C_out_per_group) row-major
        // input_col:         (col_rows, col_cols)        row-major
        // grad_weight group: (C_out_per_group, col_cols) row-major
        int64_t M_gemm = C_out_per_group;
        int64_t K_gemm = col_rows;
        int64_t N_weight = col_cols;

        if (dtype == DType::Float32) {
            HipBuffer input_col_buf(col_rows * col_cols * sizeof(float));
            float* input_col = input_col_buf.as<float>();

            const float* input_ptr = input.data<float>() + in_start * D_in * H_in * W_in;
            im3col_kernel_nchw<<<grid, block, 0, stream>>>(
                input_ptr, input_col, N, C_in, C_in_per_group,
                D_in, H_in, W_in, kD, kH, kW,
                stride_d, stride_h, stride_w,
                pad_d, pad_h, pad_w,
                dil_d, dil_h, dil_w,
                D_out, H_out, W_out
            );
            HIP_CHECK(hipGetLastError());

            float alpha = 1.0f, beta = 0.0f;
            // With op(B)=transpose, rocBLAS reads grad_out as stored [M×K]
            // col-major ld=M => row-major [col_row][oc] = NDHWC. Permute the
            // NCDHW grad_output into a contiguous NDHWC temp first.
            HipBuffer grad_out_ndhwc_buf(col_rows * C_out_per_group * sizeof(float));
            float* grad_out_ndhwc = grad_out_ndhwc_buf.as<float>();
            {
                dim3 p_grid, p_block;
                compute_launch_config_conv3d(col_rows * C_out_per_group, p_grid, p_block);
                conv3d_ncdhw_to_ndhwc_kernel<float><<<p_grid, p_block, 0, stream>>>(
                    grad_output.data<float>(), grad_out_ndhwc,
                    N, D_out, H_out, W_out, C_out, C_out_per_group, out_start
                );
                HIP_CHECK(hipGetLastError());
            }
            const float* grad_out_ptr = grad_out_ndhwc;
            float* grad_weight_ptr = grad_weight.data<float>() + out_start * C_in_per_group * kD * kH * kW;

            // Row-major: C = A^T * B
            // rocBLAS col-major: C = B^T * A (with transposed args)
            ROCBLAS_CHECK(rocblas_sgemm(
                handle,
                rocblas_operation_none, rocblas_operation_transpose,
                N_weight, M_gemm, K_gemm,
                &alpha,
                input_col, N_weight,
                grad_out_ptr, M_gemm,
                &beta,
                grad_weight_ptr, N_weight
            ));


        } else if (dtype == DType::Float64) {
            HipBuffer input_col_buf(col_rows * col_cols * sizeof(double));
            double* input_col = input_col_buf.as<double>();

            const double* input_ptr = input.data<double>() + in_start * D_in * H_in * W_in;
            im3col_kernel_nchw<<<grid, block, 0, stream>>>(
                input_ptr, input_col, N, C_in, C_in_per_group,
                D_in, H_in, W_in, kD, kH, kW,
                stride_d, stride_h, stride_w,
                pad_d, pad_h, pad_w,
                dil_d, dil_h, dil_w,
                D_out, H_out, W_out
            );
            HIP_CHECK(hipGetLastError());

            double alpha = 1.0, beta = 0.0;
            // With op(B)=transpose, rocBLAS reads grad_out as stored [M×K]
            // col-major ld=M => row-major [col_row][oc] = NDHWC. Permute the
            // NCDHW grad_output into a contiguous NDHWC temp first.
            HipBuffer grad_out_ndhwc_buf(col_rows * C_out_per_group * sizeof(double));
            double* grad_out_ndhwc = grad_out_ndhwc_buf.as<double>();
            {
                dim3 p_grid, p_block;
                compute_launch_config_conv3d(col_rows * C_out_per_group, p_grid, p_block);
                conv3d_ncdhw_to_ndhwc_kernel<double><<<p_grid, p_block, 0, stream>>>(
                    grad_output.data<double>(), grad_out_ndhwc,
                    N, D_out, H_out, W_out, C_out, C_out_per_group, out_start
                );
                HIP_CHECK(hipGetLastError());
            }
            const double* grad_out_ptr = grad_out_ndhwc;
            double* grad_weight_ptr = grad_weight.data<double>() + out_start * C_in_per_group * kD * kH * kW;

            ROCBLAS_CHECK(rocblas_dgemm(
                handle,
                rocblas_operation_none, rocblas_operation_transpose,
                N_weight, M_gemm, K_gemm,
                &alpha,
                input_col, N_weight,
                grad_out_ptr, M_gemm,
                &beta,
                grad_weight_ptr, N_weight
            ));


        } else if (dtype == DType::Float16) {
            HipBuffer input_col_buf(col_rows * col_cols * sizeof(rocblas_half));
            rocblas_half* input_col = input_col_buf.as<rocblas_half>();

            const __half* input_ptr = reinterpret_cast<const __half*>(input.data<Float16>()) + in_start * D_in * H_in * W_in;
            im3col_kernel_nchw<<<grid, block, 0, stream>>>(
                input_ptr, reinterpret_cast<__half*>(input_col), N, C_in, C_in_per_group,
                D_in, H_in, W_in, kD, kH, kW,
                stride_d, stride_h, stride_w,
                pad_d, pad_h, pad_w,
                dil_d, dil_h, dil_w,
                D_out, H_out, W_out
            );
            HIP_CHECK(hipGetLastError());

            rocblas_half alpha_h{static_cast<uint16_t>(0x3C00)};
            rocblas_half beta_h{static_cast<uint16_t>(0x0000)};

            // With op(B)=transpose, rocBLAS reads grad_out as stored [M×K]
            // col-major ld=M => row-major [col_row][oc] = NDHWC. Permute the
            // NCDHW grad_output into a contiguous NDHWC temp first.
            HipBuffer grad_out_ndhwc_buf(col_rows * C_out_per_group * sizeof(rocblas_half));
            rocblas_half* grad_out_ndhwc = grad_out_ndhwc_buf.as<rocblas_half>();
            {
                dim3 p_grid, p_block;
                compute_launch_config_conv3d(col_rows * C_out_per_group, p_grid, p_block);
                conv3d_ncdhw_to_ndhwc_kernel<__half><<<p_grid, p_block, 0, stream>>>(
                    reinterpret_cast<const __half*>(grad_output.data<Float16>()),
                    reinterpret_cast<__half*>(grad_out_ndhwc),
                    N, D_out, H_out, W_out, C_out, C_out_per_group, out_start
                );
                HIP_CHECK(hipGetLastError());
            }
            const rocblas_half* grad_out_ptr = grad_out_ndhwc;
            rocblas_half* grad_weight_ptr = reinterpret_cast<rocblas_half*>(grad_weight.data<Float16>()) + out_start * C_in_per_group * kD * kH * kW;

            ROCBLAS_CHECK(rocblas_hgemm(
                handle,
                rocblas_operation_none, rocblas_operation_transpose,
                N_weight, M_gemm, K_gemm,
                &alpha_h,
                input_col, N_weight,
                grad_out_ptr, M_gemm,
                &beta_h,
                grad_weight_ptr, N_weight
            ));

        } else {
            throw std::runtime_error("Conv3d backward weight: unsupported dtype");
        }
    }

    return grad_weight;
}

// ============================================================================
// Conv3d Backward Bias: sum over (N, D_out, H_out, W_out) dims
// ============================================================================
auto conv3d_backward_bias_hip(
    const Tensor& grad_output,
    hipStream_t stream
) -> Tensor {
    // BFloat16: upcast to Float32, compute, convert back
    if (grad_output.dtype() == DType::BFloat16) {
        auto grad_output_f32 = grad_output.to(DType::Float32);
        auto result = conv3d_backward_bias_hip(grad_output_f32, stream);
        return result.to(DType::BFloat16);
    }

    auto grad_shape = grad_output.shape();
    const int64_t N = grad_shape[0];
    const int64_t C_out = grad_shape[1];
    const int64_t D_out = grad_shape[2];
    const int64_t H_out = grad_shape[3];
    const int64_t W_out = grad_shape[4];
    const int64_t spatial_size = D_out * H_out * W_out;

    const DType dtype = grad_output.dtype();
    Tensor grad_bias({C_out}, dtype, grad_output.device());

    size_t elem_size = (dtype == DType::Float64) ? sizeof(double) :
                       (dtype == DType::Float16) ? sizeof(__half) : sizeof(float);
    HIP_CHECK(hipMemsetAsync(grad_bias.data_ptr(), 0, C_out * elem_size, stream));

    if (dtype == DType::Float32) {
        sum_bias_grad_3d_kernel<<<static_cast<unsigned int>(C_out), 256, 0, stream>>>(
            grad_output.data<float>(), grad_bias.data<float>(),
            N, C_out, spatial_size);
    } else if (dtype == DType::Float64) {
        sum_bias_grad_3d_kernel<<<static_cast<unsigned int>(C_out), 256, 0, stream>>>(
            grad_output.data<double>(), grad_bias.data<double>(),
            N, C_out, spatial_size);
    } else if (dtype == DType::Float16) {
        sum_bias_grad_3d_kernel<<<static_cast<unsigned int>(C_out), 256, 0, stream>>>(
            reinterpret_cast<const __half*>(grad_output.data<Float16>()),
            reinterpret_cast<__half*>(grad_bias.data<Float16>()),
            N, C_out, spatial_size);
    } else {
        throw std::runtime_error("Conv3d backward bias: unsupported dtype");
    }
    HIP_CHECK(hipGetLastError());

    return grad_bias;
}

// ============================================================================
// ConvTranspose3d Forward: Gather approach (each output gathers from input)
// ============================================================================
template<typename T>
__global__ void conv_transpose3d_forward_kernel(
    const T* input,
    const T* weight,
    const T* bias,
    T* output,
    int64_t batch,
    int64_t in_channels, int64_t in_d, int64_t in_h, int64_t in_w,
    int64_t out_channels, int64_t out_d, int64_t out_h, int64_t out_w,
    int64_t kD, int64_t kH, int64_t kW,
    int64_t stride_d, int64_t stride_h, int64_t stride_w,
    int64_t pad_d, int64_t pad_h, int64_t pad_w,
    int64_t out_pad_d, int64_t out_pad_h, int64_t out_pad_w,
    int64_t dil_d, int64_t dil_h, int64_t dil_w,
    int64_t groups
) {
    int64_t total_elements = batch * out_channels * out_d * out_h * out_w;

    // Grouped transposed conv: weight is [in_channels, out_channels/groups, kD,kH,kW].
    // Each output channel only sees its own group's input channels. groups==1
    // reproduces the dense behaviour (in_cpg==in_channels, out_cpg==out_channels).
    const int64_t out_cpg = out_channels / groups;
    const int64_t in_cpg  = in_channels / groups;

    HIP_KERNEL_LOOP_CONV3D(idx, total_elements) {
        int64_t ow = idx % out_w;
        int64_t oh = (idx / out_w) % out_h;
        int64_t od = (idx / (out_w * out_h)) % out_d;
        int64_t oc = (idx / (out_w * out_h * out_d)) % out_channels;
        int64_t n = idx / (out_w * out_h * out_d * out_channels);

        int64_t g = oc / out_cpg;          // group of this output channel
        int64_t oc_in_g = oc % out_cpg;    // its index within the group

        // Float accumulator for half precision; native otherwise (double stays double).
        using Acc = std::conditional_t<std::is_same_v<T, __half>, float, T>;
        Acc sum;
        if constexpr (std::is_same_v<T, __half>) {
            sum = bias ? tenzor::rocm::safe_h2f(bias[oc]) : Acc(0);
        } else {
            sum = bias ? bias[oc] : Acc(0);
        }

        for (int64_t ic_local = 0; ic_local < in_cpg; ++ic_local) {
            int64_t ic = g * in_cpg + ic_local;
            for (int64_t kd = 0; kd < kD; ++kd) {
                for (int64_t kh = 0; kh < kH; ++kh) {
                    for (int64_t kw = 0; kw < kW; ++kw) {
                        // Dilated transposed-conv gather: the kernel tap (kd,kh,kw)
                        // is spaced by the dilation factor.
                        int64_t d_offset = od + pad_d - kd * dil_d;
                        int64_t h_offset = oh + pad_h - kh * dil_h;
                        int64_t w_offset = ow + pad_w - kw * dil_w;

                        if (d_offset % stride_d != 0 || h_offset % stride_h != 0 || w_offset % stride_w != 0) continue;

                        int64_t id = d_offset / stride_d;
                        int64_t ih = h_offset / stride_h;
                        int64_t iw = w_offset / stride_w;

                        if (id >= 0 && id < in_d && ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                            int64_t input_idx = n * (in_channels * in_d * in_h * in_w) +
                                               ic * (in_d * in_h * in_w) + id * (in_h * in_w) + ih * in_w + iw;
                            // Weight: [in_channels, out_channels/groups, kD, kH, kW]
                            int64_t weight_idx = ic * (out_cpg * kD * kH * kW) +
                                                oc_in_g * (kD * kH * kW) + kd * (kH * kW) + kh * kW + kw;
                            if constexpr (std::is_same_v<T, __half>) {
                                sum += tenzor::rocm::safe_h2f(input[input_idx]) *
                                       tenzor::rocm::safe_h2f(weight[weight_idx]);
                            } else {
                                sum += input[input_idx] * weight[weight_idx];
                            }
                        }
                    }
                }
            }
        }

        if constexpr (std::is_same_v<T, __half>) {
            output[idx] = tenzor::rocm::safe_f2h(sum);
        } else {
            output[idx] = static_cast<T>(sum);
        }
    }
}

auto conv_transpose3d_forward_hip(
    const Tensor& input,         // (N, C_in, D_in, H_in, W_in)
    const Tensor& weight,        // (C_in, C_out, kD, kH, kW)
    const Tensor& bias,          // (C_out) or empty
    const std::vector<int64_t>& stride,
    const std::vector<int64_t>& padding,
    const std::vector<int64_t>& output_padding,
    const std::vector<int64_t>& dilation,
    int64_t groups,
    hipStream_t stream
) -> Tensor {
    // BFloat16: upcast to Float32, compute, convert back
    if (input.dtype() == DType::BFloat16) {
        auto input_f32 = input.to(DType::Float32);
        auto weight_f32 = weight.to(DType::Float32);
        auto bias_f32 = bias.numel() > 0 ? bias.to(DType::Float32) : bias;
        auto result = conv_transpose3d_forward_hip(input_f32, weight_f32, bias_f32,
                                                    stride, padding, output_padding,
                                                    dilation, groups, stream);
        return result.to(DType::BFloat16);
    }

    auto input_shape = input.shape();
    auto weight_shape = weight.shape();

    const int64_t N = input_shape[0];
    const int64_t C_in = input_shape[1];
    const int64_t D_in = input_shape[2];
    const int64_t H_in = input_shape[3];
    const int64_t W_in = input_shape[4];

    // ConvTranspose weight layout is [in_channels, out_channels/groups, kD,kH,kW],
    // so the true output-channel count is weight_shape[1] * groups. Using
    // weight_shape[1] directly produced out_channels/groups channels and ignored
    // grouping (mirrors the conv2d transpose fix at conv2d.hip.cpp:1862).
    const int64_t C_out = weight_shape[1] * groups;
    const int64_t kD = weight_shape[2];
    const int64_t kH = weight_shape[3];
    const int64_t kW = weight_shape[4];

    const int64_t stride_d = stride.size() > 0 ? stride[0] : 1;
    const int64_t stride_h = stride.size() > 1 ? stride[1] : stride_d;
    const int64_t stride_w = stride.size() > 2 ? stride[2] : stride_h;

    const int64_t pad_d = padding.size() > 0 ? padding[0] : 0;
    const int64_t pad_h = padding.size() > 1 ? padding[1] : pad_d;
    const int64_t pad_w = padding.size() > 2 ? padding[2] : pad_h;

    const int64_t out_pad_d = output_padding.size() > 0 ? output_padding[0] : 0;
    const int64_t out_pad_h = output_padding.size() > 1 ? output_padding[1] : out_pad_d;
    const int64_t out_pad_w = output_padding.size() > 2 ? output_padding[2] : out_pad_h;

    const int64_t dil_d = dilation.size() > 0 ? dilation[0] : 1;
    const int64_t dil_h = dilation.size() > 1 ? dilation[1] : dil_d;
    const int64_t dil_w = dilation.size() > 2 ? dilation[2] : dil_h;

    const int64_t D_out = (D_in - 1) * stride_d - 2 * pad_d + dil_d * (kD - 1) + out_pad_d + 1;
    const int64_t H_out = (H_in - 1) * stride_h - 2 * pad_h + dil_h * (kH - 1) + out_pad_h + 1;
    const int64_t W_out = (W_in - 1) * stride_w - 2 * pad_w + dil_w * (kW - 1) + out_pad_w + 1;

    Tensor output({N, C_out, D_out, H_out, W_out}, input.dtype(), input.device());

    int64_t total_elements = output.numel();
    int threads = 256;
    int blocks = static_cast<int>((total_elements + threads - 1) / threads);
    // Empty output: a zero-grid launch is rejected by HIP; return as-is.
    if (blocks == 0) return output;

    bool has_bias = bias.numel() > 0;

    if (input.dtype() == DType::Float32) {
        hipLaunchKernelGGL(conv_transpose3d_forward_kernel<float>,
            dim3(blocks), dim3(threads), 0, stream,
            input.data<float>(), weight.data<float>(),
            has_bias ? bias.data<float>() : nullptr,
            output.data<float>(),
            N, C_in, D_in, H_in, W_in, C_out, D_out, H_out, W_out,
            kD, kH, kW, stride_d, stride_h, stride_w,
            pad_d, pad_h, pad_w, out_pad_d, out_pad_h, out_pad_w,
            dil_d, dil_h, dil_w, groups);
    } else if (input.dtype() == DType::Float64) {
        hipLaunchKernelGGL(conv_transpose3d_forward_kernel<double>,
            dim3(blocks), dim3(threads), 0, stream,
            input.data<double>(), weight.data<double>(),
            has_bias ? bias.data<double>() : nullptr,
            output.data<double>(),
            N, C_in, D_in, H_in, W_in, C_out, D_out, H_out, W_out,
            kD, kH, kW, stride_d, stride_h, stride_w,
            pad_d, pad_h, pad_w, out_pad_d, out_pad_h, out_pad_w,
            dil_d, dil_h, dil_w, groups);
    } else if (input.dtype() == DType::Float16) {
        hipLaunchKernelGGL(conv_transpose3d_forward_kernel<__half>,
            dim3(blocks), dim3(threads), 0, stream,
            reinterpret_cast<const __half*>(input.data<Float16>()),
            reinterpret_cast<const __half*>(weight.data<Float16>()),
            has_bias ? reinterpret_cast<const __half*>(bias.data<Float16>()) : nullptr,
            reinterpret_cast<__half*>(output.data<Float16>()),
            N, C_in, D_in, H_in, W_in, C_out, D_out, H_out, W_out,
            kD, kH, kW, stride_d, stride_h, stride_w,
            pad_d, pad_h, pad_w, out_pad_d, out_pad_h, out_pad_w,
            dil_d, dil_h, dil_w, groups);
    } else {
        throw std::runtime_error("ConvTranspose3d forward: unsupported dtype");
    }

    HIP_CHECK(hipGetLastError());
    return output;
}

// ============================================================================
// ConvTranspose3d backward-input: dedicated kernel.
//
// Mirrors CPU conv_transpose3d_backward_input_impl. For transposed conv the
// forward relation is  od = id*stride - pad + kd*dil  (note: + kd*dil, the
// inverse-stride relation, NOT conv3d-forward's id = od*stride - pad + kd*dil),
// so backward-input gathers grad_output at that od/oh/ow for each grad_input
// element. Weight layout is the transpose weight (C_in, C_out/groups, kD,kH,kW)
// — channel roles are NOT the same as a conv3d-forward weight.
//
//   grad_input[b][g*icpg+icl][id,ih,iw] =
//       sum_{ocl,kd,kh,kw} weight[(g*icpg+icl)][ocl][kd,kh,kw]
//                          * grad_output[b][g*ocpg+ocl][od,oh,ow]
// over taps where (od,oh,ow) lands in-bounds.
// ============================================================================
template<typename T>
__global__ void conv_transpose3d_backward_input_kernel(
    const T* __restrict__ grad_output,   // (N, C_out, D_out, H_out, W_out)
    const T* __restrict__ weight,        // (C_in, C_out/groups, kD, kH, kW)
    T* __restrict__ grad_input,          // (N, C_in, D_in, H_in, W_in)
    int64_t batch,
    int64_t in_channels, int64_t in_d, int64_t in_h, int64_t in_w,
    int64_t out_channels, int64_t out_d, int64_t out_h, int64_t out_w,
    int64_t kD, int64_t kH, int64_t kW,
    int64_t stride_d, int64_t stride_h, int64_t stride_w,
    int64_t pad_d, int64_t pad_h, int64_t pad_w,
    int64_t dil_d, int64_t dil_h, int64_t dil_w,
    int64_t groups
) {
    int64_t in_channels_per_group = in_channels / groups;
    int64_t out_channels_per_group = out_channels / groups;
    int64_t total = batch * in_channels * in_d * in_h * in_w;

    HIP_KERNEL_LOOP_CONV3D(idx, total) {
        int64_t iw = idx % in_w;
        int64_t ih = (idx / in_w) % in_h;
        int64_t id = (idx / (in_w * in_h)) % in_d;
        int64_t ic = (idx / (in_w * in_h * in_d)) % in_channels;
        int64_t n  = idx / (in_w * in_h * in_d * in_channels);

        int64_t g = ic / in_channels_per_group;
        // weight first-dim index is the global input channel (ic); second dim is
        // the in-group output channel.
        const T* w_ic = weight + ic * (out_channels_per_group * kD * kH * kW);

        // Float accumulator for half precision; native otherwise (double stays double).
        using Acc = std::conditional_t<std::is_same_v<T, __half>, float, T>;
        Acc acc = Acc(0);

        for (int64_t ocl = 0; ocl < out_channels_per_group; ++ocl) {
            int64_t oc = g * out_channels_per_group + ocl;
            const T* go_oc = grad_output
                + ((n * out_channels + oc) * out_d) * (out_h * out_w);
            const T* w_oc = w_ic + ocl * (kD * kH * kW);
            for (int64_t kd = 0; kd < kD; ++kd) {
                int64_t od = id * stride_d - pad_d + kd * dil_d;
                if (od < 0 || od >= out_d) continue;
                for (int64_t kh = 0; kh < kH; ++kh) {
                    int64_t oh = ih * stride_h - pad_h + kh * dil_h;
                    if (oh < 0 || oh >= out_h) continue;
                    for (int64_t kw = 0; kw < kW; ++kw) {
                        int64_t ow = iw * stride_w - pad_w + kw * dil_w;
                        if (ow < 0 || ow >= out_w) continue;
                        T gv = go_oc[(od * out_h + oh) * out_w + ow];
                        T wv = w_oc[(kd * kH + kh) * kW + kw];
                        if constexpr (std::is_same_v<T, __half>) {
                            acc += tenzor::rocm::safe_h2f(gv) * tenzor::rocm::safe_h2f(wv);
                        } else {
                            acc += gv * wv;
                        }
                    }
                }
            }
        }

        if constexpr (std::is_same_v<T, __half>) {
            grad_input[idx] = tenzor::rocm::safe_f2h(acc);
        } else {
            grad_input[idx] = static_cast<T>(acc);
        }
    }
}

// ConvTranspose3d backward input
auto conv_transpose3d_backward_input_hip(
    const Tensor& grad_output,
    const Tensor& weight,        // (C_in, C_out/groups, kD, kH, kW) - transposed weight
    const std::vector<int64_t>& input_shape,
    const std::vector<int64_t>& stride,
    const std::vector<int64_t>& padding,
    const std::vector<int64_t>& dilation,
    int64_t groups,
    hipStream_t stream
) -> Tensor {
    // BFloat16: upcast to Float32, compute, convert back.
    if (grad_output.dtype() == DType::BFloat16) {
        auto go_f32 = grad_output.to(DType::Float32);
        auto w_f32 = weight.to(DType::Float32);
        auto result = conv_transpose3d_backward_input_hip(
            go_f32, w_f32, input_shape, stride, padding, dilation, groups, stream);
        return result.to(DType::BFloat16);
    }

    auto go_shape = grad_output.shape();
    auto w_shape = weight.shape();

    const int64_t N = go_shape[0];
    const int64_t C_out = go_shape[1];
    const int64_t D_out = go_shape[2], H_out = go_shape[3], W_out = go_shape[4];

    const int64_t C_in = input_shape[1];
    const int64_t D_in = input_shape[2], H_in = input_shape[3], W_in = input_shape[4];

    const int64_t kD = w_shape[2], kH = w_shape[3], kW = w_shape[4];

    const int64_t stride_d = stride.size() > 0 ? stride[0] : 1;
    const int64_t stride_h = stride.size() > 1 ? stride[1] : stride_d;
    const int64_t stride_w = stride.size() > 2 ? stride[2] : stride_h;

    const int64_t pad_d = padding.size() > 0 ? padding[0] : 0;
    const int64_t pad_h = padding.size() > 1 ? padding[1] : pad_d;
    const int64_t pad_w = padding.size() > 2 ? padding[2] : pad_h;

    const int64_t dil_d = dilation.size() > 0 ? dilation[0] : 1;
    const int64_t dil_h = dilation.size() > 1 ? dilation[1] : dil_d;
    const int64_t dil_w = dilation.size() > 2 ? dilation[2] : dil_h;

    Tensor grad_input(std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                      grad_output.dtype(), grad_output.device());

    int64_t total = grad_input.numel();
    // Empty grad: a zero-grid launch is rejected by HIP; return as-is.
    if (total == 0) return grad_input;
    dim3 grid, block;
    compute_launch_config_conv3d(total, grid, block);

    if (grad_output.dtype() == DType::Float32) {
        hipLaunchKernelGGL(conv_transpose3d_backward_input_kernel<float>,
            grid, block, 0, stream,
            grad_output.data<float>(), weight.data<float>(), grad_input.data<float>(),
            N, C_in, D_in, H_in, W_in, C_out, D_out, H_out, W_out,
            kD, kH, kW, stride_d, stride_h, stride_w,
            pad_d, pad_h, pad_w, dil_d, dil_h, dil_w, groups);
    } else if (grad_output.dtype() == DType::Float64) {
        hipLaunchKernelGGL(conv_transpose3d_backward_input_kernel<double>,
            grid, block, 0, stream,
            grad_output.data<double>(), weight.data<double>(), grad_input.data<double>(),
            N, C_in, D_in, H_in, W_in, C_out, D_out, H_out, W_out,
            kD, kH, kW, stride_d, stride_h, stride_w,
            pad_d, pad_h, pad_w, dil_d, dil_h, dil_w, groups);
    } else if (grad_output.dtype() == DType::Float16) {
        hipLaunchKernelGGL(conv_transpose3d_backward_input_kernel<__half>,
            grid, block, 0, stream,
            reinterpret_cast<const __half*>(grad_output.data<Float16>()),
            reinterpret_cast<const __half*>(weight.data<Float16>()),
            reinterpret_cast<__half*>(grad_input.data<Float16>()),
            N, C_in, D_in, H_in, W_in, C_out, D_out, H_out, W_out,
            kD, kH, kW, stride_d, stride_h, stride_w,
            pad_d, pad_h, pad_w, dil_d, dil_h, dil_w, groups);
    } else {
        throw std::runtime_error("ConvTranspose3d backward input: unsupported dtype");
    }

    HIP_CHECK(hipGetLastError());
    return grad_input;
}

// ============================================================================
// ConvTranspose3d backward-weight: dedicated kernel.
//
// Mirrors CPU conv_transpose3d_backward_weight_impl. grad_weight has the
// transpose layout (C_in, C_out/groups, kD, kH, kW). Using the same inverse
// stride relation od = id*stride - pad + kd*dil:
//
//   grad_weight[g*icpg+icl][ocl][kd,kh,kw] =
//       sum_{b,id,ih,iw} input[b][g*icpg+icl][id,ih,iw]
//                        * grad_output[b][g*ocpg+ocl][od,oh,ow]
// One thread per grad_weight element.
// ============================================================================
template<typename T>
__global__ void conv_transpose3d_backward_weight_kernel(
    const T* __restrict__ grad_output,   // (N, C_out, D_out, H_out, W_out)
    const T* __restrict__ input,         // (N, C_in, D_in, H_in, W_in)
    T* __restrict__ grad_weight,         // (C_in, C_out/groups, kD, kH, kW)
    int64_t batch,
    int64_t in_channels, int64_t in_d, int64_t in_h, int64_t in_w,
    int64_t out_channels, int64_t out_d, int64_t out_h, int64_t out_w,
    int64_t kD, int64_t kH, int64_t kW,
    int64_t stride_d, int64_t stride_h, int64_t stride_w,
    int64_t pad_d, int64_t pad_h, int64_t pad_w,
    int64_t dil_d, int64_t dil_h, int64_t dil_w,
    int64_t groups
) {
    int64_t in_channels_per_group = in_channels / groups;
    int64_t out_channels_per_group = out_channels / groups;
    int64_t total = in_channels * out_channels_per_group * kD * kH * kW;

    HIP_KERNEL_LOOP_CONV3D(idx, total) {
        int64_t kw  = idx % kW;
        int64_t kh  = (idx / kW) % kH;
        int64_t kd  = (idx / (kW * kH)) % kD;
        int64_t ocl = (idx / (kW * kH * kD)) % out_channels_per_group;
        int64_t ic  = idx / (kW * kH * kD * out_channels_per_group);  // global input channel

        int64_t g = ic / in_channels_per_group;
        int64_t oc = g * out_channels_per_group + ocl;

        using Acc = std::conditional_t<std::is_same_v<T, __half>, float, T>;
        Acc acc = Acc(0);

        for (int64_t b = 0; b < batch; ++b) {
            const T* in_ic = input + ((b * in_channels + ic) * in_d) * (in_h * in_w);
            const T* go_oc = grad_output + ((b * out_channels + oc) * out_d) * (out_h * out_w);
            for (int64_t id = 0; id < in_d; ++id) {
                int64_t od = id * stride_d - pad_d + kd * dil_d;
                if (od < 0 || od >= out_d) continue;
                for (int64_t ih = 0; ih < in_h; ++ih) {
                    int64_t oh = ih * stride_h - pad_h + kh * dil_h;
                    if (oh < 0 || oh >= out_h) continue;
                    for (int64_t iw = 0; iw < in_w; ++iw) {
                        int64_t ow = iw * stride_w - pad_w + kw * dil_w;
                        if (ow < 0 || ow >= out_w) continue;
                        T inv = in_ic[(id * in_h + ih) * in_w + iw];
                        T gv  = go_oc[(od * out_h + oh) * out_w + ow];
                        if constexpr (std::is_same_v<T, __half>) {
                            acc += tenzor::rocm::safe_h2f(inv) * tenzor::rocm::safe_h2f(gv);
                        } else {
                            acc += inv * gv;
                        }
                    }
                }
            }
        }

        if constexpr (std::is_same_v<T, __half>) {
            grad_weight[idx] = tenzor::rocm::safe_f2h(acc);
        } else {
            grad_weight[idx] = static_cast<T>(acc);
        }
    }
}

// ConvTranspose3d backward weight
auto conv_transpose3d_backward_weight_hip(
    const Tensor& grad_output,
    const Tensor& input,
    const std::vector<int64_t>& weight_shape,  // (C_in, C_out/groups, kD, kH, kW)
    const std::vector<int64_t>& stride,
    const std::vector<int64_t>& padding,
    const std::vector<int64_t>& dilation,
    int64_t groups,
    hipStream_t stream
) -> Tensor {
    // BFloat16: upcast to Float32, compute, convert back.
    if (grad_output.dtype() == DType::BFloat16) {
        auto go_f32 = grad_output.to(DType::Float32);
        auto in_f32 = input.to(DType::Float32);
        auto result = conv_transpose3d_backward_weight_hip(
            go_f32, in_f32, weight_shape, stride, padding, dilation, groups, stream);
        return result.to(DType::BFloat16);
    }

    auto go_shape = grad_output.shape();
    auto in_shape = input.shape();

    const int64_t N = go_shape[0];
    const int64_t C_out = go_shape[1];
    const int64_t D_out = go_shape[2], H_out = go_shape[3], W_out = go_shape[4];

    const int64_t C_in = in_shape[1];
    const int64_t D_in = in_shape[2], H_in = in_shape[3], W_in = in_shape[4];

    const int64_t kD = weight_shape[2], kH = weight_shape[3], kW = weight_shape[4];

    const int64_t stride_d = stride.size() > 0 ? stride[0] : 1;
    const int64_t stride_h = stride.size() > 1 ? stride[1] : stride_d;
    const int64_t stride_w = stride.size() > 2 ? stride[2] : stride_h;

    const int64_t pad_d = padding.size() > 0 ? padding[0] : 0;
    const int64_t pad_h = padding.size() > 1 ? padding[1] : pad_d;
    const int64_t pad_w = padding.size() > 2 ? padding[2] : pad_h;

    const int64_t dil_d = dilation.size() > 0 ? dilation[0] : 1;
    const int64_t dil_h = dilation.size() > 1 ? dilation[1] : dil_d;
    const int64_t dil_w = dilation.size() > 2 ? dilation[2] : dil_h;

    Tensor grad_weight(std::vector<int64_t>(weight_shape.begin(), weight_shape.end()),
                       grad_output.dtype(), grad_output.device());

    int64_t total = grad_weight.numel();
    // Empty grad: a zero-grid launch is rejected by HIP; return as-is.
    if (total == 0) return grad_weight;
    dim3 grid, block;
    compute_launch_config_conv3d(total, grid, block);

    if (grad_output.dtype() == DType::Float32) {
        hipLaunchKernelGGL(conv_transpose3d_backward_weight_kernel<float>,
            grid, block, 0, stream,
            grad_output.data<float>(), input.data<float>(), grad_weight.data<float>(),
            N, C_in, D_in, H_in, W_in, C_out, D_out, H_out, W_out,
            kD, kH, kW, stride_d, stride_h, stride_w,
            pad_d, pad_h, pad_w, dil_d, dil_h, dil_w, groups);
    } else if (grad_output.dtype() == DType::Float64) {
        hipLaunchKernelGGL(conv_transpose3d_backward_weight_kernel<double>,
            grid, block, 0, stream,
            grad_output.data<double>(), input.data<double>(), grad_weight.data<double>(),
            N, C_in, D_in, H_in, W_in, C_out, D_out, H_out, W_out,
            kD, kH, kW, stride_d, stride_h, stride_w,
            pad_d, pad_h, pad_w, dil_d, dil_h, dil_w, groups);
    } else if (grad_output.dtype() == DType::Float16) {
        hipLaunchKernelGGL(conv_transpose3d_backward_weight_kernel<__half>,
            grid, block, 0, stream,
            reinterpret_cast<const __half*>(grad_output.data<Float16>()),
            reinterpret_cast<const __half*>(input.data<Float16>()),
            reinterpret_cast<__half*>(grad_weight.data<Float16>()),
            N, C_in, D_in, H_in, W_in, C_out, D_out, H_out, W_out,
            kD, kH, kW, stride_d, stride_h, stride_w,
            pad_d, pad_h, pad_w, dil_d, dil_h, dil_w, groups);
    } else {
        throw std::runtime_error("ConvTranspose3d backward weight: unsupported dtype");
    }

    HIP_CHECK(hipGetLastError());
    return grad_weight;
}

// ConvTranspose3d backward bias = same as Conv3d backward bias
auto conv_transpose3d_backward_bias_hip(
    const Tensor& grad_output,
    hipStream_t stream
) -> Tensor {
    return conv3d_backward_bias_hip(grad_output, stream);
}

} // namespace rocm
} // namespace tenzor
