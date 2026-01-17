#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "tenzor/core/tensor.hpp"
#include "tenzor/core/dtype.hpp"
#include <stdexcept>
#include <vector>
#include <limits>

namespace tenzor {
namespace rocm {

// HIP Error checking macro
#define HIP_CHECK(call) \
    do { \
        hipError_t err = call; \
        if (err != hipSuccess) { \
            throw std::runtime_error( \
                std::string("HIP error at ") + __FILE__ + ":" + \
                std::to_string(__LINE__) + " - " + hipGetErrorString(err) \
            ); \
        } \
    } while(0)

// Grid-stride loop for HIP kernels
#define HIP_KERNEL_LOOP(i, n) \
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; \
         i < (n); \
         i += blockDim.x * gridDim.x)

// ==============================================================================
// MaxPool2D Forward
// ==============================================================================

template<typename T>
__global__ void maxpool2d_forward_kernel(
    const T* input,
    T* output,
    int64_t* indices,
    int64_t batch_size,
    int64_t channels,
    int64_t input_h,
    int64_t input_w,
    int64_t output_h,
    int64_t output_w,
    int64_t kernel_h,
    int64_t kernel_w,
    int64_t stride_h,
    int64_t stride_w,
    int64_t pad_h,
    int64_t pad_w,
    bool return_indices
) {
    int64_t total_elements = batch_size * channels * output_h * output_w;

    HIP_KERNEL_LOOP(idx, total_elements) {
        int64_t ow = idx % output_w;
        int64_t oh = (idx / output_w) % output_h;
        int64_t c = (idx / (output_w * output_h)) % channels;
        int64_t n = idx / (output_w * output_h * channels);

        int64_t h_start = oh * stride_h - pad_h;
        int64_t w_start = ow * stride_w - pad_w;
        int64_t h_end = min(h_start + kernel_h, input_h);
        int64_t w_end = min(w_start + kernel_w, input_w);
        h_start = max(h_start, (int64_t)0);
        w_start = max(w_start, (int64_t)0);

        T max_val = -INFINITY;
        int64_t max_idx = -1;

        for (int64_t h = h_start; h < h_end; ++h) {
            for (int64_t w = w_start; w < w_end; ++w) {
                int64_t input_idx = ((n * channels + c) * input_h + h) * input_w + w;
                T val = input[input_idx];
                if (val > max_val) {
                    max_val = val;
                    max_idx = input_idx;
                }
            }
        }

        output[idx] = max_val;
        if (return_indices && indices != nullptr) {
            indices[idx] = max_idx;
        }
    }
}

// Float16 MaxPool2D Forward
__global__ void maxpool2d_forward_kernel_fp16(
    const __half* input,
    __half* output,
    int64_t* indices,
    int64_t batch_size,
    int64_t channels,
    int64_t input_h,
    int64_t input_w,
    int64_t output_h,
    int64_t output_w,
    int64_t kernel_h,
    int64_t kernel_w,
    int64_t stride_h,
    int64_t stride_w,
    int64_t pad_h,
    int64_t pad_w,
    bool return_indices
) {
    int64_t total_elements = batch_size * channels * output_h * output_w;

    HIP_KERNEL_LOOP(idx, total_elements) {
        int64_t ow = idx % output_w;
        int64_t oh = (idx / output_w) % output_h;
        int64_t c = (idx / (output_w * output_h)) % channels;
        int64_t n = idx / (output_w * output_h * channels);

        int64_t h_start = oh * stride_h - pad_h;
        int64_t w_start = ow * stride_w - pad_w;
        int64_t h_end = min(h_start + kernel_h, input_h);
        int64_t w_end = min(w_start + kernel_w, input_w);
        h_start = max(h_start, (int64_t)0);
        w_start = max(w_start, (int64_t)0);

        float max_val = -INFINITY;
        int64_t max_idx = -1;

        for (int64_t h = h_start; h < h_end; ++h) {
            for (int64_t w = w_start; w < w_end; ++w) {
                int64_t input_idx = ((n * channels + c) * input_h + h) * input_w + w;
                float val = __half2float(input[input_idx]);
                if (val > max_val) {
                    max_val = val;
                    max_idx = input_idx;
                }
            }
        }

        output[idx] = __float2half(max_val);
        if (return_indices && indices != nullptr) {
            indices[idx] = max_idx;
        }
    }
}

auto maxpool2d_forward_hip(
    const Tensor& input,
    int64_t kernel_h, int64_t kernel_w,
    int64_t stride_h, int64_t stride_w,
    int64_t pad_h, int64_t pad_w,
    bool return_indices
) -> std::pair<Tensor, Tensor> {

    auto input_shape = input.shape();
    int64_t batch_size = input_shape[0];
    int64_t channels = input_shape[1];
    int64_t input_h = input_shape[2];
    int64_t input_w = input_shape[3];

    int64_t output_h = (input_h + 2 * pad_h - kernel_h) / stride_h + 1;
    int64_t output_w = (input_w + 2 * pad_w - kernel_w) / stride_w + 1;

    std::vector<int64_t> output_shape = {batch_size, channels, output_h, output_w};
    Tensor output = Tensor(output_shape, input.dtype(), input.device());
    Tensor indices;

    if (return_indices) {
        indices = Tensor(output_shape, DType::Int64, input.device());
    }

    int64_t total_elements = batch_size * channels * output_h * output_w;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;

    if (input.dtype() == DType::Float32) {
        hipLaunchKernelGGL(maxpool2d_forward_kernel<float>,
            dim3(blocks), dim3(threads), 0, 0,
            input.data<float>(),
            output.data<float>(),
            return_indices ? indices.data<int64_t>() : nullptr,
            batch_size, channels, input_h, input_w,
            output_h, output_w, kernel_h, kernel_w,
            stride_h, stride_w, pad_h, pad_w, return_indices
        );
    } else if (input.dtype() == DType::Float64) {
        hipLaunchKernelGGL(maxpool2d_forward_kernel<double>,
            dim3(blocks), dim3(threads), 0, 0,
            input.data<double>(),
            output.data<double>(),
            return_indices ? indices.data<int64_t>() : nullptr,
            batch_size, channels, input_h, input_w,
            output_h, output_w, kernel_h, kernel_w,
            stride_h, stride_w, pad_h, pad_w, return_indices
        );
    } else if (input.dtype() == DType::Float16) {
        hipLaunchKernelGGL(maxpool2d_forward_kernel_fp16,
            dim3(blocks), dim3(threads), 0, 0,
            reinterpret_cast<const __half*>(input.data<Float16>()),
            reinterpret_cast<__half*>(output.data<Float16>()),
            return_indices ? indices.data<int64_t>() : nullptr,
            batch_size, channels, input_h, input_w,
            output_h, output_w, kernel_h, kernel_w,
            stride_h, stride_w, pad_h, pad_w, return_indices
        );
    } else {
        throw std::runtime_error("maxpool2d_forward_hip: Only Float32, Float64, and Float16 supported");
    }

    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    return {output, indices};
}

// ==============================================================================
// MaxPool2D Backward
// ==============================================================================

template<typename T>
__global__ void maxpool2d_backward_kernel(
    const T* grad_output,
    const int64_t* indices,
    T* grad_input,
    int64_t total_elements
) {
    HIP_KERNEL_LOOP(idx, total_elements) {
        int64_t input_idx = indices[idx];
        atomicAdd(&grad_input[input_idx], grad_output[idx]);
    }
}

// Float16 maxpool2d backward (accumulate in float)
__global__ void maxpool2d_backward_kernel_fp16(
    const __half* grad_output,
    const int64_t* indices,
    float* grad_input_f32,
    int64_t total_elements
) {
    HIP_KERNEL_LOOP(idx, total_elements) {
        int64_t input_idx = indices[idx];
        atomicAdd(&grad_input_f32[input_idx], __half2float(grad_output[idx]));
    }
}

// Convert float to half kernel
__global__ void convert_f32_to_f16_pool(const float* src, __half* dst, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        dst[idx] = __float2half(src[idx]);
    }
}

auto maxpool2d_backward_hip(
    const Tensor& grad_output,
    const Tensor& indices,
    const std::vector<int64_t>& input_shape
) -> Tensor {

    Tensor grad_input = Tensor(input_shape, grad_output.dtype(), grad_output.device());

    int64_t total_elements = grad_output.numel();
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;

    if (grad_output.dtype() == DType::Float32) {
        hipLaunchKernelGGL(maxpool2d_backward_kernel<float>,
            dim3(blocks), dim3(threads), 0, 0,
            grad_output.data<float>(),
            indices.data<int64_t>(),
            grad_input.data<float>(),
            total_elements
        );
    } else if (grad_output.dtype() == DType::Float64) {
        hipLaunchKernelGGL(maxpool2d_backward_kernel<double>,
            dim3(blocks), dim3(threads), 0, 0,
            grad_output.data<double>(),
            indices.data<int64_t>(),
            grad_input.data<double>(),
            total_elements
        );
    } else if (grad_output.dtype() == DType::Float16) {
        // Accumulate in float, then convert back
        int64_t input_numel = 1;
        for (auto s : input_shape) input_numel *= s;
        Tensor grad_input_f32 = Tensor(input_shape, DType::Float32, grad_output.device());
        HIP_CHECK(hipMemset(grad_input_f32.data<float>(), 0, input_numel * sizeof(float)));

        hipLaunchKernelGGL(maxpool2d_backward_kernel_fp16,
            dim3(blocks), dim3(threads), 0, 0,
            reinterpret_cast<const __half*>(grad_output.data<Float16>()),
            indices.data<int64_t>(),
            grad_input_f32.data<float>(),
            total_elements
        );

        int convert_blocks = (input_numel + threads - 1) / threads;
        hipLaunchKernelGGL(convert_f32_to_f16_pool,
            dim3(convert_blocks), dim3(threads), 0, 0,
            grad_input_f32.data<float>(),
            reinterpret_cast<__half*>(grad_input.data<Float16>()),
            input_numel
        );
    } else {
        throw std::runtime_error("maxpool2d_backward_hip: Only Float32, Float64, and Float16 supported");
    }

    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    return grad_input;
}

// ==============================================================================
// AvgPool2D Forward
// ==============================================================================

template<typename T>
__global__ void avgpool2d_forward_kernel(
    const T* input,
    T* output,
    int64_t batch_size,
    int64_t channels,
    int64_t input_h,
    int64_t input_w,
    int64_t output_h,
    int64_t output_w,
    int64_t kernel_h,
    int64_t kernel_w,
    int64_t stride_h,
    int64_t stride_w,
    int64_t pad_h,
    int64_t pad_w,
    bool count_include_pad
) {
    int64_t total_elements = batch_size * channels * output_h * output_w;

    HIP_KERNEL_LOOP(idx, total_elements) {
        int64_t ow = idx % output_w;
        int64_t oh = (idx / output_w) % output_h;
        int64_t c = (idx / (output_w * output_h)) % channels;
        int64_t n = idx / (output_w * output_h * channels);

        int64_t h_start = oh * stride_h - pad_h;
        int64_t w_start = ow * stride_w - pad_w;
        int64_t h_end = min(h_start + kernel_h, input_h);
        int64_t w_end = min(w_start + kernel_w, input_w);
        h_start = max(h_start, (int64_t)0);
        w_start = max(w_start, (int64_t)0);

        T sum = 0;
        int64_t count = 0;

        for (int64_t h = h_start; h < h_end; ++h) {
            for (int64_t w = w_start; w < w_end; ++w) {
                int64_t input_idx = ((n * channels + c) * input_h + h) * input_w + w;
                sum += input[input_idx];
                count++;
            }
        }

        if (count_include_pad) {
            count = kernel_h * kernel_w;
        }

        output[idx] = sum / static_cast<T>(count);
    }
}

// Float16 AvgPool2D Forward
__global__ void avgpool2d_forward_kernel_fp16(
    const __half* input,
    __half* output,
    int64_t batch_size,
    int64_t channels,
    int64_t input_h,
    int64_t input_w,
    int64_t output_h,
    int64_t output_w,
    int64_t kernel_h,
    int64_t kernel_w,
    int64_t stride_h,
    int64_t stride_w,
    int64_t pad_h,
    int64_t pad_w,
    bool count_include_pad
) {
    int64_t total_elements = batch_size * channels * output_h * output_w;

    HIP_KERNEL_LOOP(idx, total_elements) {
        int64_t ow = idx % output_w;
        int64_t oh = (idx / output_w) % output_h;
        int64_t c = (idx / (output_w * output_h)) % channels;
        int64_t n = idx / (output_w * output_h * channels);

        int64_t h_start = oh * stride_h - pad_h;
        int64_t w_start = ow * stride_w - pad_w;
        int64_t h_end = min(h_start + kernel_h, input_h);
        int64_t w_end = min(w_start + kernel_w, input_w);
        h_start = max(h_start, (int64_t)0);
        w_start = max(w_start, (int64_t)0);

        float sum = 0.0f;
        int64_t count = 0;

        for (int64_t h = h_start; h < h_end; ++h) {
            for (int64_t w = w_start; w < w_end; ++w) {
                int64_t input_idx = ((n * channels + c) * input_h + h) * input_w + w;
                sum += __half2float(input[input_idx]);
                count++;
            }
        }

        if (count_include_pad) {
            count = kernel_h * kernel_w;
        }

        output[idx] = __float2half(sum / static_cast<float>(count));
    }
}

auto avgpool2d_forward_hip(
    const Tensor& input,
    int64_t kernel_h, int64_t kernel_w,
    int64_t stride_h, int64_t stride_w,
    int64_t pad_h, int64_t pad_w,
    bool count_include_pad
) -> Tensor {

    auto input_shape = input.shape();
    int64_t batch_size = input_shape[0];
    int64_t channels = input_shape[1];
    int64_t input_h = input_shape[2];
    int64_t input_w = input_shape[3];

    int64_t output_h = (input_h + 2 * pad_h - kernel_h) / stride_h + 1;
    int64_t output_w = (input_w + 2 * pad_w - kernel_w) / stride_w + 1;

    std::vector<int64_t> output_shape = {batch_size, channels, output_h, output_w};
    Tensor output = Tensor(output_shape, input.dtype(), input.device());

    int64_t total_elements = batch_size * channels * output_h * output_w;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;

    if (input.dtype() == DType::Float32) {
        hipLaunchKernelGGL(avgpool2d_forward_kernel<float>,
            dim3(blocks), dim3(threads), 0, 0,
            input.data<float>(),
            output.data<float>(),
            batch_size, channels, input_h, input_w,
            output_h, output_w, kernel_h, kernel_w,
            stride_h, stride_w, pad_h, pad_w, count_include_pad
        );
    } else if (input.dtype() == DType::Float64) {
        hipLaunchKernelGGL(avgpool2d_forward_kernel<double>,
            dim3(blocks), dim3(threads), 0, 0,
            input.data<double>(),
            output.data<double>(),
            batch_size, channels, input_h, input_w,
            output_h, output_w, kernel_h, kernel_w,
            stride_h, stride_w, pad_h, pad_w, count_include_pad
        );
    } else if (input.dtype() == DType::Float16) {
        hipLaunchKernelGGL(avgpool2d_forward_kernel_fp16,
            dim3(blocks), dim3(threads), 0, 0,
            reinterpret_cast<const __half*>(input.data<Float16>()),
            reinterpret_cast<__half*>(output.data<Float16>()),
            batch_size, channels, input_h, input_w,
            output_h, output_w, kernel_h, kernel_w,
            stride_h, stride_w, pad_h, pad_w, count_include_pad
        );
    } else {
        throw std::runtime_error("avgpool2d_forward_hip: Only Float32, Float64, and Float16 supported");
    }

    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    return output;
}

// ==============================================================================
// AvgPool2D Backward
// ==============================================================================

template<typename T>
__global__ void avgpool2d_backward_kernel(
    const T* grad_output,
    T* grad_input,
    int64_t batch_size,
    int64_t channels,
    int64_t input_h,
    int64_t input_w,
    int64_t output_h,
    int64_t output_w,
    int64_t kernel_h,
    int64_t kernel_w,
    int64_t stride_h,
    int64_t stride_w,
    int64_t pad_h,
    int64_t pad_w,
    bool count_include_pad
) {
    int64_t total_output_elements = batch_size * channels * output_h * output_w;

    HIP_KERNEL_LOOP(idx, total_output_elements) {
        int64_t ow = idx % output_w;
        int64_t oh = (idx / output_w) % output_h;
        int64_t c = (idx / (output_w * output_h)) % channels;
        int64_t n = idx / (output_w * output_h * channels);

        int64_t h_start = oh * stride_h - pad_h;
        int64_t w_start = ow * stride_w - pad_w;
        int64_t h_end = min(h_start + kernel_h, input_h);
        int64_t w_end = min(w_start + kernel_w, input_w);
        h_start = max(h_start, (int64_t)0);
        w_start = max(w_start, (int64_t)0);

        int64_t count = (h_end - h_start) * (w_end - w_start);
        if (count_include_pad) {
            count = kernel_h * kernel_w;
        }

        T grad_val = grad_output[idx] / static_cast<T>(count);

        for (int64_t h = h_start; h < h_end; ++h) {
            for (int64_t w = w_start; w < w_end; ++w) {
                int64_t input_idx = ((n * channels + c) * input_h + h) * input_w + w;
                atomicAdd(&grad_input[input_idx], grad_val);
            }
        }
    }
}

// Float16 AvgPool2D Backward (accumulate in float)
__global__ void avgpool2d_backward_kernel_fp16(
    const __half* grad_output,
    float* grad_input_f32,
    int64_t batch_size,
    int64_t channels,
    int64_t input_h,
    int64_t input_w,
    int64_t output_h,
    int64_t output_w,
    int64_t kernel_h,
    int64_t kernel_w,
    int64_t stride_h,
    int64_t stride_w,
    int64_t pad_h,
    int64_t pad_w,
    bool count_include_pad
) {
    int64_t total_output_elements = batch_size * channels * output_h * output_w;

    HIP_KERNEL_LOOP(idx, total_output_elements) {
        int64_t ow = idx % output_w;
        int64_t oh = (idx / output_w) % output_h;
        int64_t c = (idx / (output_w * output_h)) % channels;
        int64_t n = idx / (output_w * output_h * channels);

        int64_t h_start = oh * stride_h - pad_h;
        int64_t w_start = ow * stride_w - pad_w;
        int64_t h_end = min(h_start + kernel_h, input_h);
        int64_t w_end = min(w_start + kernel_w, input_w);
        h_start = max(h_start, (int64_t)0);
        w_start = max(w_start, (int64_t)0);

        int64_t count = (h_end - h_start) * (w_end - w_start);
        if (count_include_pad) {
            count = kernel_h * kernel_w;
        }

        float grad_val = __half2float(grad_output[idx]) / static_cast<float>(count);

        for (int64_t h = h_start; h < h_end; ++h) {
            for (int64_t w = w_start; w < w_end; ++w) {
                int64_t input_idx = ((n * channels + c) * input_h + h) * input_w + w;
                atomicAdd(&grad_input_f32[input_idx], grad_val);
            }
        }
    }
}

auto avgpool2d_backward_hip(
    const Tensor& grad_output,
    const std::vector<int64_t>& input_shape,
    int64_t kernel_h, int64_t kernel_w,
    int64_t stride_h, int64_t stride_w,
    int64_t pad_h, int64_t pad_w,
    bool count_include_pad
) -> Tensor {

    Tensor grad_input = Tensor(input_shape, grad_output.dtype(), grad_output.device());

    auto output_shape = grad_output.shape();
    int64_t batch_size = output_shape[0];
    int64_t channels = output_shape[1];
    int64_t output_h = output_shape[2];
    int64_t output_w = output_shape[3];
    int64_t input_h = input_shape[2];
    int64_t input_w = input_shape[3];

    int64_t total_elements = grad_output.numel();
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;

    if (grad_output.dtype() == DType::Float32) {
        hipLaunchKernelGGL(avgpool2d_backward_kernel<float>,
            dim3(blocks), dim3(threads), 0, 0,
            grad_output.data<float>(),
            grad_input.data<float>(),
            batch_size, channels, input_h, input_w,
            output_h, output_w, kernel_h, kernel_w,
            stride_h, stride_w, pad_h, pad_w, count_include_pad
        );
    } else if (grad_output.dtype() == DType::Float64) {
        hipLaunchKernelGGL(avgpool2d_backward_kernel<double>,
            dim3(blocks), dim3(threads), 0, 0,
            grad_output.data<double>(),
            grad_input.data<double>(),
            batch_size, channels, input_h, input_w,
            output_h, output_w, kernel_h, kernel_w,
            stride_h, stride_w, pad_h, pad_w, count_include_pad
        );
    } else if (grad_output.dtype() == DType::Float16) {
        // Accumulate in float, then convert back
        int64_t input_numel = 1;
        for (auto s : input_shape) input_numel *= s;
        Tensor grad_input_f32 = Tensor(input_shape, DType::Float32, grad_output.device());
        HIP_CHECK(hipMemset(grad_input_f32.data<float>(), 0, input_numel * sizeof(float)));

        hipLaunchKernelGGL(avgpool2d_backward_kernel_fp16,
            dim3(blocks), dim3(threads), 0, 0,
            reinterpret_cast<const __half*>(grad_output.data<Float16>()),
            grad_input_f32.data<float>(),
            batch_size, channels, input_h, input_w,
            output_h, output_w, kernel_h, kernel_w,
            stride_h, stride_w, pad_h, pad_w, count_include_pad
        );

        int convert_blocks = (input_numel + threads - 1) / threads;
        hipLaunchKernelGGL(convert_f32_to_f16_pool,
            dim3(convert_blocks), dim3(threads), 0, 0,
            grad_input_f32.data<float>(),
            reinterpret_cast<__half*>(grad_input.data<Float16>()),
            input_numel
        );
    } else {
        throw std::runtime_error("avgpool2d_backward_hip: Only Float32, Float64, and Float16 supported");
    }

    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    return grad_input;
}

// ==============================================================================
// Adaptive AvgPool2D
// ==============================================================================

template<typename T>
__global__ void adaptive_avgpool2d_kernel(
    const T* input,
    T* output,
    int64_t batch_size,
    int64_t channels,
    int64_t input_h,
    int64_t input_w,
    int64_t output_h,
    int64_t output_w
) {
    int64_t total_elements = batch_size * channels * output_h * output_w;

    HIP_KERNEL_LOOP(idx, total_elements) {
        int64_t ow = idx % output_w;
        int64_t oh = (idx / output_w) % output_h;
        int64_t c = (idx / (output_w * output_h)) % channels;
        int64_t n = idx / (output_w * output_h * channels);

        int64_t h_start = (oh * input_h) / output_h;
        int64_t h_end = ((oh + 1) * input_h) / output_h;
        int64_t w_start = (ow * input_w) / output_w;
        int64_t w_end = ((ow + 1) * input_w) / output_w;

        T sum = 0;
        int64_t count = 0;

        for (int64_t h = h_start; h < h_end; ++h) {
            for (int64_t w = w_start; w < w_end; ++w) {
                int64_t input_idx = ((n * channels + c) * input_h + h) * input_w + w;
                sum += input[input_idx];
                count++;
            }
        }

        output[idx] = sum / static_cast<T>(count);
    }
}

auto adaptive_avgpool2d_hip(
    const Tensor& input,
    int64_t output_h,
    int64_t output_w
) -> Tensor {

    auto input_shape = input.shape();
    int64_t batch_size = input_shape[0];
    int64_t channels = input_shape[1];
    int64_t input_h = input_shape[2];
    int64_t input_w = input_shape[3];

    std::vector<int64_t> output_shape = {batch_size, channels, output_h, output_w};
    Tensor output = Tensor(output_shape, input.dtype(), input.device());

    int64_t total_elements = batch_size * channels * output_h * output_w;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;

    if (input.dtype() == DType::Float32) {
        hipLaunchKernelGGL(adaptive_avgpool2d_kernel<float>,
            dim3(blocks), dim3(threads), 0, 0,
            input.data<float>(),
            output.data<float>(),
            batch_size, channels, input_h, input_w,
            output_h, output_w
        );
    } else if (input.dtype() == DType::Float64) {
        hipLaunchKernelGGL(adaptive_avgpool2d_kernel<double>,
            dim3(blocks), dim3(threads), 0, 0,
            input.data<double>(),
            output.data<double>(),
            batch_size, channels, input_h, input_w,
            output_h, output_w
        );
    } else if (input.dtype() == DType::Float16) {
        hipLaunchKernelGGL(adaptive_avgpool2d_kernel<__half>,
            dim3(blocks), dim3(threads), 0, 0,
            reinterpret_cast<const __half*>(input.data<Float16>()),
            reinterpret_cast<__half*>(output.data<Float16>()),
            batch_size, channels, input_h, input_w,
            output_h, output_w
        );
    } else {
        throw std::runtime_error("adaptive_avgpool2d_hip: unsupported dtype");
    }

    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    return output;
}

// ==============================================================================
// Adaptive MaxPool2D
// ==============================================================================

template<typename T>
__global__ void adaptive_maxpool2d_kernel(
    const T* input,
    T* output,
    int64_t* indices,
    int64_t batch_size,
    int64_t channels,
    int64_t input_h,
    int64_t input_w,
    int64_t output_h,
    int64_t output_w,
    bool return_indices
) {
    int64_t total_elements = batch_size * channels * output_h * output_w;

    HIP_KERNEL_LOOP(idx, total_elements) {
        int64_t ow = idx % output_w;
        int64_t oh = (idx / output_w) % output_h;
        int64_t c = (idx / (output_w * output_h)) % channels;
        int64_t n = idx / (output_w * output_h * channels);

        int64_t h_start = (oh * input_h) / output_h;
        int64_t h_end = ((oh + 1) * input_h) / output_h;
        int64_t w_start = (ow * input_w) / output_w;
        int64_t w_end = ((ow + 1) * input_w) / output_w;

        T max_val = -INFINITY;
        int64_t max_idx = -1;

        for (int64_t h = h_start; h < h_end; ++h) {
            for (int64_t w = w_start; w < w_end; ++w) {
                int64_t input_idx = ((n * channels + c) * input_h + h) * input_w + w;
                T val = input[input_idx];
                if (val > max_val) {
                    max_val = val;
                    max_idx = input_idx;
                }
            }
        }

        output[idx] = max_val;
        if (return_indices && indices != nullptr) {
            indices[idx] = max_idx;
        }
    }
}

auto adaptive_maxpool2d_hip(
    const Tensor& input,
    int64_t output_h,
    int64_t output_w,
    bool return_indices
) -> std::pair<Tensor, Tensor> {

    auto input_shape = input.shape();
    int64_t batch_size = input_shape[0];
    int64_t channels = input_shape[1];
    int64_t input_h = input_shape[2];
    int64_t input_w = input_shape[3];

    std::vector<int64_t> output_shape = {batch_size, channels, output_h, output_w};
    Tensor output = Tensor(output_shape, input.dtype(), input.device());
    Tensor indices;

    if (return_indices) {
        indices = Tensor(output_shape, DType::Int64, input.device());
    }

    int64_t total_elements = batch_size * channels * output_h * output_w;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;

    if (input.dtype() == DType::Float32) {
        hipLaunchKernelGGL(adaptive_maxpool2d_kernel<float>,
            dim3(blocks), dim3(threads), 0, 0,
            input.data<float>(),
            output.data<float>(),
            return_indices ? indices.data<int64_t>() : nullptr,
            batch_size, channels, input_h, input_w,
            output_h, output_w, return_indices
        );
    } else if (input.dtype() == DType::Float64) {
        hipLaunchKernelGGL(adaptive_maxpool2d_kernel<double>,
            dim3(blocks), dim3(threads), 0, 0,
            input.data<double>(),
            output.data<double>(),
            return_indices ? indices.data<int64_t>() : nullptr,
            batch_size, channels, input_h, input_w,
            output_h, output_w, return_indices
        );
    } else {
        throw std::runtime_error("adaptive_maxpool2d_hip: Only Float32 and Float64 supported");
    }

    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    return {output, indices};
}

// ==============================================================================
// Adaptive Average Pooling 2D Backward
// ==============================================================================

template<typename T>
__global__ void adaptive_avgpool2d_backward_kernel(
    const T* grad_output,
    T* grad_input,
    int64_t N,
    int64_t C,
    int64_t in_H,
    int64_t in_W,
    int64_t out_H,
    int64_t out_W
) {
    int64_t total = N * C * in_H * in_W;

    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
         idx += blockDim.x * gridDim.x) {

        int64_t iw = idx % in_W;
        int64_t ih = (idx / in_W) % in_H;
        int64_t c = (idx / (in_W * in_H)) % C;
        int64_t n = idx / (in_W * in_H * C);

        T sum = T(0);

        // Find all output positions that this input contributes to
        for (int64_t oh = 0; oh < out_H; ++oh) {
            int64_t start_h = (ih * out_H) / in_H;
            int64_t end_h = ((ih + 1) * out_H + in_H - 1) / in_H;

            if (oh < start_h || oh >= end_h) continue;

            int64_t pool_start_h = (oh * in_H) / out_H;
            int64_t pool_end_h = ((oh + 1) * in_H + out_H - 1) / out_H;
            if (ih < pool_start_h || ih >= pool_end_h) continue;

            for (int64_t ow = 0; ow < out_W; ++ow) {
                int64_t start_w = (iw * out_W) / in_W;
                int64_t end_w = ((iw + 1) * out_W + in_W - 1) / in_W;

                if (ow < start_w || ow >= end_w) continue;

                int64_t pool_start_w = (ow * in_W) / out_W;
                int64_t pool_end_w = ((ow + 1) * in_W + out_W - 1) / out_W;
                if (iw < pool_start_w || iw >= pool_end_w) continue;

                T pool_size = T((pool_end_h - pool_start_h) * (pool_end_w - pool_start_w));
                int64_t grad_idx = n * (C * out_H * out_W) + c * (out_H * out_W) + oh * out_W + ow;
                sum += grad_output[grad_idx] / pool_size;
            }
        }

        grad_input[idx] = sum;
    }
}

auto adaptive_avgpool2d_backward_hip(
    const Tensor& grad_output,
    const Tensor& input,
    hipStream_t stream
) -> Tensor {
    auto input_shape = input.shape();
    auto grad_shape = grad_output.shape();

    int64_t N = input_shape[0];
    int64_t C = input_shape[1];
    int64_t in_H = input_shape[2];
    int64_t in_W = input_shape[3];
    int64_t out_H = grad_shape[2];
    int64_t out_W = grad_shape[3];

    Tensor grad_input(std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                      input.dtype(), input.device());

    int64_t total = grad_input.numel();
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    if (input.dtype() == DType::Float32) {
        hipLaunchKernelGGL(adaptive_avgpool2d_backward_kernel<float>,
            dim3(blocks), dim3(threads), 0, stream,
            grad_output.data<float>(),
            grad_input.data<float>(),
            N, C, in_H, in_W, out_H, out_W);
    } else if (input.dtype() == DType::Float64) {
        hipLaunchKernelGGL(adaptive_avgpool2d_backward_kernel<double>,
            dim3(blocks), dim3(threads), 0, stream,
            grad_output.data<double>(),
            grad_input.data<double>(),
            N, C, in_H, in_W, out_H, out_W);
    } else if (input.dtype() == DType::Float16) {
        hipLaunchKernelGGL(adaptive_avgpool2d_backward_kernel<__half>,
            dim3(blocks), dim3(threads), 0, stream,
            reinterpret_cast<const __half*>(grad_output.data<Float16>()),
            reinterpret_cast<__half*>(grad_input.data<Float16>()),
            N, C, in_H, in_W, out_H, out_W);
    } else {
        throw std::runtime_error("adaptive_avgpool2d_backward: unsupported dtype");
    }

    HIP_CHECK(hipGetLastError());
    return grad_input;
}

// ==============================================================================
// Adaptive Max Pooling 2D Backward
// ==============================================================================

template<typename T>
__global__ void adaptive_maxpool2d_backward_kernel(
    const T* grad_output,
    const int64_t* indices,
    T* grad_input,
    int64_t N,
    int64_t C,
    int64_t in_H,
    int64_t in_W,
    int64_t out_H,
    int64_t out_W
) {
    int64_t total = N * C * out_H * out_W;

    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
         idx += blockDim.x * gridDim.x) {

        int64_t ow = idx % out_W;
        int64_t oh = (idx / out_W) % out_H;
        int64_t c = (idx / (out_W * out_H)) % C;
        int64_t n = idx / (out_W * out_H * C);

        int64_t max_idx = indices[idx];
        int64_t grad_input_idx = n * (C * in_H * in_W) + c * (in_H * in_W) + max_idx;
        atomicAdd(&grad_input[grad_input_idx], grad_output[idx]);
    }
}

auto adaptive_maxpool2d_backward_hip(
    const Tensor& grad_output,
    const Tensor& indices,
    const Tensor& input,
    hipStream_t stream
) -> Tensor {
    auto input_shape = input.shape();
    auto grad_shape = grad_output.shape();

    int64_t N = input_shape[0];
    int64_t C = input_shape[1];
    int64_t in_H = input_shape[2];
    int64_t in_W = input_shape[3];
    int64_t out_H = grad_shape[2];
    int64_t out_W = grad_shape[3];

    Tensor grad_input(std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                      input.dtype(), input.device());
    HIP_CHECK(hipMemsetAsync(grad_input.data<uint8_t>(), 0,
        grad_input.numel() * dtype_size(input.dtype()), stream));

    int64_t total = grad_output.numel();
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    if (input.dtype() == DType::Float32) {
        hipLaunchKernelGGL(adaptive_maxpool2d_backward_kernel<float>,
            dim3(blocks), dim3(threads), 0, stream,
            grad_output.data<float>(),
            indices.data<int64_t>(),
            grad_input.data<float>(),
            N, C, in_H, in_W, out_H, out_W);
    } else if (input.dtype() == DType::Float64) {
        hipLaunchKernelGGL(adaptive_maxpool2d_backward_kernel<double>,
            dim3(blocks), dim3(threads), 0, stream,
            grad_output.data<double>(),
            indices.data<int64_t>(),
            grad_input.data<double>(),
            N, C, in_H, in_W, out_H, out_W);
    } else {
        throw std::runtime_error("adaptive_maxpool2d_backward: unsupported dtype");
    }

    HIP_CHECK(hipGetLastError());
    return grad_input;
}

// ==============================================================================
// Adaptive Average Pooling 2D (with stream parameter)
// ==============================================================================

auto adaptive_avgpool2d_forward(
    const Tensor& input,
    int64_t output_h,
    int64_t output_w,
    hipStream_t stream
) -> Tensor {
    auto shape = input.shape();
    int64_t N = shape[0];
    int64_t C = shape[1];
    int64_t H_in = shape[2];
    int64_t W_in = shape[3];

    Tensor output({N, C, output_h, output_w}, input.dtype(), input.device());

    int64_t total = N * C * output_h * output_w;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    if (input.dtype() == DType::Float32) {
        hipLaunchKernelGGL(adaptive_avgpool2d_kernel<float>,
            dim3(blocks), dim3(threads), 0, stream,
            input.data<float>(), output.data<float>(),
            N, C, H_in, W_in, output_h, output_w);
    } else if (input.dtype() == DType::Float64) {
        hipLaunchKernelGGL(adaptive_avgpool2d_kernel<double>,
            dim3(blocks), dim3(threads), 0, stream,
            input.data<double>(), output.data<double>(),
            N, C, H_in, W_in, output_h, output_w);
    } else if (input.dtype() == DType::Float16) {
        hipLaunchKernelGGL(adaptive_avgpool2d_kernel<__half>,
            dim3(blocks), dim3(threads), 0, stream,
            reinterpret_cast<const __half*>(input.data<Float16>()),
            reinterpret_cast<__half*>(output.data<Float16>()),
            N, C, H_in, W_in, output_h, output_w);
    } else {
        throw std::runtime_error("adaptive_avgpool2d_forward: unsupported dtype");
    }

    HIP_CHECK(hipGetLastError());
    return output;
}

auto adaptive_avgpool2d_backward(
    const Tensor& grad_output,
    int64_t H_in,
    int64_t W_in,
    hipStream_t stream
) -> Tensor {
    auto shape = grad_output.shape();
    int64_t N = shape[0];
    int64_t C = shape[1];
    int64_t H_out = shape[2];
    int64_t W_out = shape[3];

    Tensor grad_input({N, C, H_in, W_in}, grad_output.dtype(), grad_output.device());

    // The kernel iterates over input elements, so use input size for launch config
    int64_t total = N * C * H_in * W_in;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    if (grad_output.dtype() == DType::Float32) {
        hipLaunchKernelGGL(adaptive_avgpool2d_backward_kernel<float>,
            dim3(blocks), dim3(threads), 0, stream,
            grad_output.data<float>(), grad_input.data<float>(),
            N, C, H_in, W_in, H_out, W_out);
    } else if (grad_output.dtype() == DType::Float64) {
        hipLaunchKernelGGL(adaptive_avgpool2d_backward_kernel<double>,
            dim3(blocks), dim3(threads), 0, stream,
            grad_output.data<double>(), grad_input.data<double>(),
            N, C, H_in, W_in, H_out, W_out);
    } else if (grad_output.dtype() == DType::Float16) {
        hipLaunchKernelGGL(adaptive_avgpool2d_backward_kernel<__half>,
            dim3(blocks), dim3(threads), 0, stream,
            reinterpret_cast<const __half*>(grad_output.data<Float16>()),
            reinterpret_cast<__half*>(grad_input.data<Float16>()),
            N, C, H_in, W_in, H_out, W_out);
    } else {
        throw std::runtime_error("adaptive_avgpool2d_backward: unsupported dtype");
    }

    HIP_CHECK(hipGetLastError());
    return grad_input;
}

} // namespace rocm
} // namespace tenzor
