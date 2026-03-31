#include "tenzor/core/tensor.hpp"
#include "tenzor/core/dtype.hpp"
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <rocblas/rocblas.h>
#include <stdexcept>
#include <vector>
#include "../rocm_error.hpp"
#include "../hip_buffer.hpp"

namespace tenzor {
namespace rocm {

// RAII wrapper for rocBLAS handle
class RocBLASHandleGuardConv3d {
public:
    RocBLASHandleGuardConv3d() {
        ROCBLAS_CHECK(rocblas_create_handle(&handle_));
    }
    ~RocBLASHandleGuardConv3d() {
        if (handle_) rocblas_destroy_handle(handle_);
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

// ============================================================================
// im3col HIP Kernel: Convert 5D input (N,C,D,H,W) to 2D for Conv3d GEMM
// ============================================================================
template<typename T>
__global__ void im3col_kernel_nchw(
    const T* __restrict__ input,
    T* __restrict__ output,
    int64_t batch,
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
                  ? input[b * (channels * depth * height * width) +
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

        T sum = T(0);

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
                            sum += col[col_row * col_cols + col_col];
                        }
                    }
                }
            }
        }

        output[output_idx] = sum;
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
    __shared__ float shared_data[256];

    int64_t c = blockIdx.x;
    if (c < channels) {
        int64_t tid = threadIdx.x;
        int64_t block_size = blockDim.x;

        float local_sum = 0.0f;
        for (int64_t idx = tid; idx < batch * spatial_size; idx += block_size) {
            int64_t b = idx / spatial_size;
            int64_t s = idx % spatial_size;
            int64_t grad_idx = b * (channels * spatial_size) + c * spatial_size + s;
            local_sum += static_cast<float>(grad_output[grad_idx]);
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
    // BFloat16: upcast to Float32, compute, convert back
    if (input.dtype() == DType::BFloat16) {
        auto input_f32 = input.to(DType::Float32);
        auto weight_f32 = weight.to(DType::Float32);
        auto bias_f32 = bias.numel() > 0 ? bias.to(DType::Float32) : bias;
        auto result = conv3d_forward_hip(input_f32, weight_f32, bias_f32,
                                          stride, padding, dilation, groups, stream);
        return result.to(DType::BFloat16);
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
                input_ptr, col_buffer, N, C_in_per_group,
                D_in, H_in, W_in, kD, kH, kW,
                stride_d, stride_h, stride_w,
                pad_d, pad_h, pad_w,
                dil_d, dil_h, dil_w,
                D_out, H_out, W_out
            );
            HIP_CHECK(hipGetLastError());

            float alpha = 1.0f, beta = 0.0f;
            const float* weight_ptr = weight.data<float>() + out_start * C_in_per_group * kD * kH * kW;
            float* output_ptr = output.data<float>() + out_start * D_out * H_out * W_out;

            // Row-major: output(M,N_gemm) = col(M,K) @ weight(N_gemm,K)^T
            // rocBLAS uses column-major, so we compute the transpose:
            //   C^T(N_gemm,M) = weight(N_gemm,K) @ col(M,K)^T
            // rocBLAS params: op_a=Transpose, op_b=None,
            //   m=N_gemm, n=M, k=K, A=weight(lda=K), B=col(ldb=K), C=output(ldc=N_gemm)
            ROCBLAS_CHECK(rocblas_sgemm(
                handle,
                rocblas_operation_transpose, rocblas_operation_none,
                N_gemm, M, K,
                &alpha,
                weight_ptr, K,
                col_buffer, K,
                &beta,
                output_ptr, N_gemm
            ));

        } else if (dtype == DType::Float64) {
            HipBuffer col_buf(col_rows * col_cols * sizeof(double));
            double* col_buffer = col_buf.as<double>();

            const double* input_ptr = input.data<double>() + in_start * D_in * H_in * W_in;
            im3col_kernel_nchw<<<grid, block, 0, stream>>>(
                input_ptr, col_buffer, N, C_in_per_group,
                D_in, H_in, W_in, kD, kH, kW,
                stride_d, stride_h, stride_w,
                pad_d, pad_h, pad_w,
                dil_d, dil_h, dil_w,
                D_out, H_out, W_out
            );
            HIP_CHECK(hipGetLastError());

            double alpha = 1.0, beta = 0.0;
            const double* weight_ptr = weight.data<double>() + out_start * C_in_per_group * kD * kH * kW;
            double* output_ptr = output.data<double>() + out_start * D_out * H_out * W_out;

            ROCBLAS_CHECK(rocblas_dgemm(
                handle,
                rocblas_operation_transpose, rocblas_operation_none,
                N_gemm, M, K,
                &alpha,
                weight_ptr, K,
                col_buffer, K,
                &beta,
                output_ptr, N_gemm
            ));

        } else if (dtype == DType::Float16) {
            static_assert(sizeof(Float16) == sizeof(__half) && alignof(Float16) == alignof(__half),
                          "Float16 and __half must have identical size and alignment");
            HipBuffer col_buf(col_rows * col_cols * sizeof(__half));
            __half* col_buffer = col_buf.as<__half>();

            const __half* input_ptr = reinterpret_cast<const __half*>(input.data<Float16>()) + in_start * D_in * H_in * W_in;
            im3col_kernel_nchw<<<grid, block, 0, stream>>>(
                input_ptr, col_buffer, N, C_in_per_group,
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
            rocblas_half* output_ptr = reinterpret_cast<rocblas_half*>(output.data<Float16>()) + out_start * D_out * H_out * W_out;

            ROCBLAS_CHECK(rocblas_hgemm(
                handle,
                rocblas_operation_transpose, rocblas_operation_none,
                N_gemm, M, K,
                &alpha_h,
                weight_ptr, K,
                reinterpret_cast<const rocblas_half*>(col_buffer), K,
                &beta_h,
                output_ptr, N_gemm
            ));

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
    // BFloat16: upcast to Float32, compute, convert back
    if (grad_output.dtype() == DType::BFloat16) {
        auto grad_output_f32 = grad_output.to(DType::Float32);
        auto weight_f32 = weight.to(DType::Float32);
        auto result = conv3d_backward_input_hip(grad_output_f32, weight_f32, input_shape,
                                                 stride, padding, dilation, groups, stream);
        return result.to(DType::BFloat16);
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
            const float* grad_out_ptr = grad_output.data<float>() + out_start * D_out * H_out * W_out;
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
                N, C_in_per_group,
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
            const double* grad_out_ptr = grad_output.data<double>() + out_start * D_out * H_out * W_out;
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
                N, C_in_per_group,
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

            const rocblas_half* grad_out_ptr = reinterpret_cast<const rocblas_half*>(grad_output.data<Float16>()) + out_start * D_out * H_out * W_out;
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
                N, C_in_per_group,
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
    // BFloat16: upcast to Float32, compute, convert back
    if (input.dtype() == DType::BFloat16) {
        auto grad_output_f32 = grad_output.to(DType::Float32);
        auto input_f32 = input.to(DType::Float32);
        auto result = conv3d_backward_weight_hip(grad_output_f32, input_f32, weight_shape,
                                                  stride, padding, dilation, groups, stream);
        return result.to(DType::BFloat16);
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
                input_ptr, input_col, N, C_in_per_group,
                D_in, H_in, W_in, kD, kH, kW,
                stride_d, stride_h, stride_w,
                pad_d, pad_h, pad_w,
                dil_d, dil_h, dil_w,
                D_out, H_out, W_out
            );
            HIP_CHECK(hipGetLastError());

            float alpha = 1.0f, beta = 0.0f;
            const float* grad_out_ptr = grad_output.data<float>() + out_start * D_out * H_out * W_out;
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
                input_ptr, input_col, N, C_in_per_group,
                D_in, H_in, W_in, kD, kH, kW,
                stride_d, stride_h, stride_w,
                pad_d, pad_h, pad_w,
                dil_d, dil_h, dil_w,
                D_out, H_out, W_out
            );
            HIP_CHECK(hipGetLastError());

            double alpha = 1.0, beta = 0.0;
            const double* grad_out_ptr = grad_output.data<double>() + out_start * D_out * H_out * W_out;
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
                input_ptr, reinterpret_cast<__half*>(input_col), N, C_in_per_group,
                D_in, H_in, W_in, kD, kH, kW,
                stride_d, stride_h, stride_w,
                pad_d, pad_h, pad_w,
                dil_d, dil_h, dil_w,
                D_out, H_out, W_out
            );
            HIP_CHECK(hipGetLastError());

            rocblas_half alpha_h{static_cast<uint16_t>(0x3C00)};
            rocblas_half beta_h{static_cast<uint16_t>(0x0000)};

            const rocblas_half* grad_out_ptr = reinterpret_cast<const rocblas_half*>(grad_output.data<Float16>()) + out_start * D_out * H_out * W_out;
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
    int64_t out_pad_d, int64_t out_pad_h, int64_t out_pad_w
) {
    int64_t total_elements = batch * out_channels * out_d * out_h * out_w;

    HIP_KERNEL_LOOP_CONV3D(idx, total_elements) {
        int64_t ow = idx % out_w;
        int64_t oh = (idx / out_w) % out_h;
        int64_t od = (idx / (out_w * out_h)) % out_d;
        int64_t oc = (idx / (out_w * out_h * out_d)) % out_channels;
        int64_t n = idx / (out_w * out_h * out_d * out_channels);

        T sum = bias ? bias[oc] : T(0);

        for (int64_t ic = 0; ic < in_channels; ++ic) {
            for (int64_t kd = 0; kd < kD; ++kd) {
                for (int64_t kh = 0; kh < kH; ++kh) {
                    for (int64_t kw = 0; kw < kW; ++kw) {
                        int64_t d_offset = od + pad_d - kd;
                        int64_t h_offset = oh + pad_h - kh;
                        int64_t w_offset = ow + pad_w - kw;

                        if (d_offset % stride_d != 0 || h_offset % stride_h != 0 || w_offset % stride_w != 0) continue;

                        int64_t id = d_offset / stride_d;
                        int64_t ih = h_offset / stride_h;
                        int64_t iw = w_offset / stride_w;

                        if (id >= 0 && id < in_d && ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                            int64_t input_idx = n * (in_channels * in_d * in_h * in_w) +
                                               ic * (in_d * in_h * in_w) + id * (in_h * in_w) + ih * in_w + iw;
                            // Weight: [in_channels, out_channels, kD, kH, kW]
                            int64_t weight_idx = ic * (out_channels * kD * kH * kW) +
                                                oc * (kD * kH * kW) + kd * (kH * kW) + kh * kW + kw;
                            sum += input[input_idx] * weight[weight_idx];
                        }
                    }
                }
            }
        }

        output[idx] = sum;
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

    const int64_t C_out = weight_shape[1];
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

    bool has_bias = bias.numel() > 0;

    if (input.dtype() == DType::Float32) {
        hipLaunchKernelGGL(conv_transpose3d_forward_kernel<float>,
            dim3(blocks), dim3(threads), 0, stream,
            input.data<float>(), weight.data<float>(),
            has_bias ? bias.data<float>() : nullptr,
            output.data<float>(),
            N, C_in, D_in, H_in, W_in, C_out, D_out, H_out, W_out,
            kD, kH, kW, stride_d, stride_h, stride_w,
            pad_d, pad_h, pad_w, out_pad_d, out_pad_h, out_pad_w);
    } else if (input.dtype() == DType::Float64) {
        hipLaunchKernelGGL(conv_transpose3d_forward_kernel<double>,
            dim3(blocks), dim3(threads), 0, stream,
            input.data<double>(), weight.data<double>(),
            has_bias ? bias.data<double>() : nullptr,
            output.data<double>(),
            N, C_in, D_in, H_in, W_in, C_out, D_out, H_out, W_out,
            kD, kH, kW, stride_d, stride_h, stride_w,
            pad_d, pad_h, pad_w, out_pad_d, out_pad_h, out_pad_w);
    } else if (input.dtype() == DType::Float16) {
        hipLaunchKernelGGL(conv_transpose3d_forward_kernel<__half>,
            dim3(blocks), dim3(threads), 0, stream,
            reinterpret_cast<const __half*>(input.data<Float16>()),
            reinterpret_cast<const __half*>(weight.data<Float16>()),
            has_bias ? reinterpret_cast<const __half*>(bias.data<Float16>()) : nullptr,
            reinterpret_cast<__half*>(output.data<Float16>()),
            N, C_in, D_in, H_in, W_in, C_out, D_out, H_out, W_out,
            kD, kH, kW, stride_d, stride_h, stride_w,
            pad_d, pad_h, pad_w, out_pad_d, out_pad_h, out_pad_w);
    } else {
        throw std::runtime_error("ConvTranspose3d forward: unsupported dtype");
    }

    HIP_CHECK(hipGetLastError());
    return output;
}

// ConvTranspose3d backward input: this is a regular conv3d forward
auto conv_transpose3d_backward_input_hip(
    const Tensor& grad_output,
    const Tensor& weight,        // (C_in, C_out, kD, kH, kW) - transposed weight
    const std::vector<int64_t>& input_shape,
    const std::vector<int64_t>& stride,
    const std::vector<int64_t>& padding,
    const std::vector<int64_t>& dilation,
    int64_t groups,
    hipStream_t stream
) -> Tensor {
    // ConvTranspose3d backward w.r.t. input is a regular Conv3d forward
    // with weight transposed (C_out and C_in swapped)
    // We need to permute weight from (C_in, C_out, kD, kH, kW) to (C_out, C_in, kD, kH, kW)
    // But since we're calling conv3d_forward with the weight as-is and the shapes work out,
    // we use the forward conv with appropriate shapes
    auto weight_shape = weight.shape();
    const int64_t C_in_orig = weight_shape[0];
    const int64_t C_out_orig = weight_shape[1];

    // For backward of ConvTranspose3d, grad_input = conv3d(grad_output, weight_transposed)
    // where weight_transposed has shape (C_in, C_out/groups, kD, kH, kW) -> (C_in, C_out/groups, kD, kH, kW)
    // The weight is already in (C_in, C_out, kD, kH, kW) format for transpose conv
    // For the backward, we need conv3d with weight (C_in, C_out/groups, kD, kH, kW)
    // treating C_in as the output channels of the forward conv

    // Use conv3d_forward_hip with swapped channels interpretation
    Tensor empty_bias;
    return conv3d_forward_hip(grad_output, weight, empty_bias, stride, padding, dilation, groups, stream);
}

// ConvTranspose3d backward weight
auto conv_transpose3d_backward_weight_hip(
    const Tensor& grad_output,
    const Tensor& input,
    const std::vector<int64_t>& weight_shape,
    const std::vector<int64_t>& stride,
    const std::vector<int64_t>& padding,
    const std::vector<int64_t>& dilation,
    int64_t groups,
    hipStream_t stream
) -> Tensor {
    // For ConvTranspose3d, backward weight is similar but with swapped roles
    // weight_shape: (C_in, C_out, kD, kH, kW)
    // grad_weight = input^T @ im3col(grad_output)
    return conv3d_backward_weight_hip(grad_output, input, weight_shape, stride, padding, dilation, groups, stream);
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
