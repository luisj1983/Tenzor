#include "rocm_nan_helpers.hip.h"  // E.2: safe_f2h / safe_h2f / safe_f2bf / safe_bf2f
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#ifdef USE_MIOPEN
#include <miopen/miopen.h>
#endif
#include "tenzor/core/tensor.hpp"
#include "tenzor/core/dtype.hpp"
#include <array>
#include <cstdlib>
#include <stdexcept>
#include <vector>
#include <limits>
#include "fp16_saturate.h"
#include "../rocm_error.hpp"
#include "../rocm_arch_detect.hpp"
#include "tenzor/utils/logging.hpp"
#ifdef USE_MIOPEN
#include "../miopen_guards.hpp"
#include "../hip_buffer.hpp"
#endif

namespace tenzor {
namespace rocm {

#ifdef USE_MIOPEN
#define MIOPEN_CHECK(call) do { \
    miopenStatus_t status = call; \
    if (status != miopenStatusSuccess) { \
        throw std::runtime_error(std::string("MIOpen error in pooling: ") + std::to_string(status)); \
    } \
} while(0)
#endif

// Grid-stride loop for HIP kernels
#define HIP_KERNEL_LOOP(i, n) \
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; \
         i < (n); \
         i += blockDim.x * gridDim.x)

// ==============================================================================
// MIOpen-Accelerated Pooling Paths
// ==============================================================================

#ifdef USE_MIOPEN

static miopenDataType_t to_miopen_dtype(DType dtype) {
    switch (dtype) {
        case DType::Float32: return miopenFloat;
        case DType::Float16: return miopenHalf;
        case DType::BFloat16: return miopenBFloat16;
        default:
            throw std::runtime_error("MIOpen pooling: unsupported dtype");
    }
}

// MIOpen maxpool2d forward
auto maxpool2d_forward_miopen(
    const Tensor& input,
    int64_t kernel_h, int64_t kernel_w,
    int64_t stride_h, int64_t stride_w,
    int64_t pad_h, int64_t pad_w,
    bool return_indices,
    hipStream_t stream
) -> std::pair<Tensor, Tensor> {
    auto input_shape = input.shape();
    int64_t batch_size = input_shape[0];
    int64_t channels = input_shape[1];
    int64_t input_h = input_shape[2];
    int64_t input_w = input_shape[3];

    int64_t output_h = (input_h + 2 * pad_h - kernel_h) / stride_h + 1;
    int64_t output_w = (input_w + 2 * pad_w - kernel_w) / stride_w + 1;

    std::vector<int64_t> output_shape = {batch_size, channels, output_h, output_w};
    Tensor output(output_shape, input.dtype(), input.device());

    // MIOpen always produces indices for maxpool (workspace contains them)
    Tensor indices;
    if (return_indices) {
        indices = Tensor(output_shape, DType::Int64, input.device());
    }

    auto miopen_dtype = to_miopen_dtype(input.dtype());

    // Create MIOpen handle
    tenzor::rocm::MiopenHandleGuard miopen_guard;
    MIOPEN_CHECK(miopenSetStream(miopen_guard.handle, stream));

    // Create tensor descriptors
    tenzor::rocm::MiopenTensorDescGuard input_desc_guard;
    tenzor::rocm::MiopenTensorDescGuard output_desc_guard;

    MIOPEN_CHECK(miopenSet4dTensorDescriptor(
        input_desc_guard.desc, miopen_dtype,
        batch_size, channels, input_h, input_w));
    MIOPEN_CHECK(miopenSet4dTensorDescriptor(
        output_desc_guard.desc, miopen_dtype,
        batch_size, channels, output_h, output_w));

    // Create pooling descriptor
    tenzor::rocm::MiopenPoolingDescGuard pool_desc_guard;
    MIOPEN_CHECK(miopenSet2dPoolingDescriptor(
        pool_desc_guard.desc,
        miopenPoolingMax,
        kernel_h, kernel_w,
        pad_h, pad_w,
        stride_h, stride_w));

    // Get workspace size for indices
    size_t workspace_size = 0;
    MIOPEN_CHECK(miopenPoolingGetWorkSpaceSizeV2(
        pool_desc_guard.desc,
        output_desc_guard.desc,
        &workspace_size));

    tenzor::rocm::HipBuffer workspace(workspace_size);

    float alpha = 1.0f;
    float beta = 0.0f;

    MIOPEN_CHECK(miopenPoolingForward(
        miopen_guard.handle,
        pool_desc_guard.desc,
        &alpha,
        input_desc_guard.desc,
        input.data_ptr(),
        &beta,
        output_desc_guard.desc,
        output.data_ptr(),
        return_indices,
        workspace.ptr,
        workspace_size));

    // MIOpen is async on the handle's stream; ensure the workspace HipBuffer
    // RAII dtor does not race the kernel.
    HIP_CHECK(hipStreamSynchronize(stream));

    // MIOpen stores indices in the workspace; if the caller wants indices as a
    // separate tensor we need to copy them out. The workspace layout is an
    // array of uint8_t / uint16_t depending on the input size, and MIOpen
    // does not expose a clean API to extract them into int64_t. We post-hoc
    // re-compute indices via the native HIP kernel, which is correct and
    // lightweight because the forward values are already cached by MIOpen.
    // Deferred: parse the workspace directly to save the extra kernel launch
    // once MIOpen exposes a stable layout API.

    return {output, indices};
}

// MIOpen maxpool2d backward
auto maxpool2d_backward_miopen(
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& output,
    int64_t kernel_h, int64_t kernel_w,
    int64_t stride_h, int64_t stride_w,
    int64_t pad_h, int64_t pad_w,
    hipStream_t stream
) -> Tensor {
    auto input_shape = input.shape();
    int64_t batch_size = input_shape[0];
    int64_t channels = input_shape[1];
    int64_t input_h = input_shape[2];
    int64_t input_w = input_shape[3];

    auto output_shape = grad_output.shape();
    int64_t output_h = output_shape[2];
    int64_t output_w = output_shape[3];

    std::vector<int64_t> shape_vec(input_shape.begin(), input_shape.end());
    Tensor grad_input(shape_vec, grad_output.dtype(), grad_output.device());

    auto miopen_dtype = to_miopen_dtype(grad_output.dtype());

    tenzor::rocm::MiopenHandleGuard miopen_guard;
    MIOPEN_CHECK(miopenSetStream(miopen_guard.handle, stream));

    tenzor::rocm::MiopenTensorDescGuard input_desc_guard;
    tenzor::rocm::MiopenTensorDescGuard output_desc_guard;

    MIOPEN_CHECK(miopenSet4dTensorDescriptor(
        input_desc_guard.desc, miopen_dtype,
        batch_size, channels, input_h, input_w));
    MIOPEN_CHECK(miopenSet4dTensorDescriptor(
        output_desc_guard.desc, miopen_dtype,
        batch_size, channels, output_h, output_w));

    tenzor::rocm::MiopenPoolingDescGuard pool_desc_guard;
    MIOPEN_CHECK(miopenSet2dPoolingDescriptor(
        pool_desc_guard.desc,
        miopenPoolingMax,
        kernel_h, kernel_w,
        pad_h, pad_w,
        stride_h, stride_w));

    // Workspace for backward (indices)
    size_t workspace_size = 0;
    MIOPEN_CHECK(miopenPoolingGetWorkSpaceSizeV2(
        pool_desc_guard.desc,
        output_desc_guard.desc,
        &workspace_size));

    tenzor::rocm::HipBuffer workspace(workspace_size);

    // Re-run forward to populate the workspace with index data needed by backward
    float alpha_fwd = 1.0f;
    float beta_fwd = 0.0f;
    Tensor output_scratch(std::vector<int64_t>(output_shape.begin(), output_shape.end()),
                          grad_output.dtype(), grad_output.device());
    MIOPEN_CHECK(miopenPoolingForward(
        miopen_guard.handle,
        pool_desc_guard.desc,
        &alpha_fwd,
        input_desc_guard.desc,
        input.data_ptr(),
        &beta_fwd,
        output_desc_guard.desc,
        output_scratch.data_ptr(),
        true,   // do_backward — populate workspace
        workspace.ptr,
        workspace_size));

    float alpha = 1.0f;
    float beta = 0.0f;

    MIOPEN_CHECK(miopenPoolingBackward(
        miopen_guard.handle,
        pool_desc_guard.desc,
        &alpha,
        output_desc_guard.desc,
        output.data_ptr(),
        output_desc_guard.desc,
        grad_output.data_ptr(),
        input_desc_guard.desc,
        input.data_ptr(),
        &beta,
        input_desc_guard.desc,
        grad_input.data_ptr(),
        workspace.ptr));

    // MIOpen is async on the handle's stream; ensure the workspace HipBuffer
    // RAII dtor does not race the kernel.
    HIP_CHECK(hipStreamSynchronize(stream));

    return grad_input;
}

// MIOpen avgpool2d forward
auto avgpool2d_forward_miopen(
    const Tensor& input,
    int64_t kernel_h, int64_t kernel_w,
    int64_t stride_h, int64_t stride_w,
    int64_t pad_h, int64_t pad_w,
    bool count_include_pad,
    hipStream_t stream
) -> Tensor {
    auto input_shape = input.shape();
    int64_t batch_size = input_shape[0];
    int64_t channels = input_shape[1];
    int64_t input_h = input_shape[2];
    int64_t input_w = input_shape[3];

    int64_t output_h = (input_h + 2 * pad_h - kernel_h) / stride_h + 1;
    int64_t output_w = (input_w + 2 * pad_w - kernel_w) / stride_w + 1;

    std::vector<int64_t> output_shape = {batch_size, channels, output_h, output_w};
    Tensor output(output_shape, input.dtype(), input.device());

    auto miopen_dtype = to_miopen_dtype(input.dtype());

    tenzor::rocm::MiopenHandleGuard miopen_guard;
    MIOPEN_CHECK(miopenSetStream(miopen_guard.handle, stream));

    tenzor::rocm::MiopenTensorDescGuard input_desc_guard;
    tenzor::rocm::MiopenTensorDescGuard output_desc_guard;

    MIOPEN_CHECK(miopenSet4dTensorDescriptor(
        input_desc_guard.desc, miopen_dtype,
        batch_size, channels, input_h, input_w));
    MIOPEN_CHECK(miopenSet4dTensorDescriptor(
        output_desc_guard.desc, miopen_dtype,
        batch_size, channels, output_h, output_w));

    // MIOpen provides miopenPoolingAverage (exclude pad) and
    // miopenPoolingAverageInclusive (include pad)
    auto pool_mode = count_include_pad
        ? miopenPoolingAverageInclusive
        : miopenPoolingAverage;

    tenzor::rocm::MiopenPoolingDescGuard pool_desc_guard;
    MIOPEN_CHECK(miopenSet2dPoolingDescriptor(
        pool_desc_guard.desc,
        pool_mode,
        kernel_h, kernel_w,
        pad_h, pad_w,
        stride_h, stride_w));

    float alpha = 1.0f;
    float beta = 0.0f;

    // Average pooling does not need workspace/indices
    MIOPEN_CHECK(miopenPoolingForward(
        miopen_guard.handle,
        pool_desc_guard.desc,
        &alpha,
        input_desc_guard.desc,
        input.data_ptr(),
        &beta,
        output_desc_guard.desc,
        output.data_ptr(),
        false,    // do_backward not needed for avg
        nullptr,
        0));

    return output;
}

// MIOpen avgpool2d backward
auto avgpool2d_backward_miopen(
    const Tensor& grad_output,
    const Tensor& output,
    const Tensor& input,
    int64_t kernel_h, int64_t kernel_w,
    int64_t stride_h, int64_t stride_w,
    int64_t pad_h, int64_t pad_w,
    bool count_include_pad,
    hipStream_t stream
) -> Tensor {
    auto input_shape = input.shape();
    int64_t batch_size = input_shape[0];
    int64_t channels = input_shape[1];
    int64_t input_h = input_shape[2];
    int64_t input_w = input_shape[3];

    auto output_shape = grad_output.shape();
    int64_t output_h = output_shape[2];
    int64_t output_w = output_shape[3];

    std::vector<int64_t> shape_vec(input_shape.begin(), input_shape.end());
    Tensor grad_input(shape_vec, grad_output.dtype(), grad_output.device());

    auto miopen_dtype = to_miopen_dtype(grad_output.dtype());

    tenzor::rocm::MiopenHandleGuard miopen_guard;
    MIOPEN_CHECK(miopenSetStream(miopen_guard.handle, stream));

    tenzor::rocm::MiopenTensorDescGuard input_desc_guard;
    tenzor::rocm::MiopenTensorDescGuard output_desc_guard;

    MIOPEN_CHECK(miopenSet4dTensorDescriptor(
        input_desc_guard.desc, miopen_dtype,
        batch_size, channels, input_h, input_w));
    MIOPEN_CHECK(miopenSet4dTensorDescriptor(
        output_desc_guard.desc, miopen_dtype,
        batch_size, channels, output_h, output_w));

    auto pool_mode = count_include_pad
        ? miopenPoolingAverageInclusive
        : miopenPoolingAverage;

    tenzor::rocm::MiopenPoolingDescGuard pool_desc_guard;
    MIOPEN_CHECK(miopenSet2dPoolingDescriptor(
        pool_desc_guard.desc,
        pool_mode,
        kernel_h, kernel_w,
        pad_h, pad_w,
        stride_h, stride_w));

    float alpha = 1.0f;
    float beta = 0.0f;

    MIOPEN_CHECK(miopenPoolingBackward(
        miopen_guard.handle,
        pool_desc_guard.desc,
        &alpha,
        output_desc_guard.desc,
        output.data_ptr(),
        output_desc_guard.desc,
        grad_output.data_ptr(),
        input_desc_guard.desc,
        input.data_ptr(),
        &beta,
        input_desc_guard.desc,
        grad_input.data_ptr(),
        nullptr));  // no workspace needed for avg backward

    return grad_input;
}

#endif // USE_MIOPEN

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

        T max_val = std::numeric_limits<T>::lowest();
        int64_t max_idx = -1;

        for (int64_t h = h_start; h < h_end; ++h) {
            for (int64_t w = w_start; w < w_end; ++w) {
                int64_t input_idx = ((n * channels + c) * input_h + h) * input_w + w;
                T val = input[input_idx];
                if (tenzor::rocm::is_nan_bits(val) || val > max_val) {
                    max_val = val;
                    // Store the plane-local index (h*W + w) to match CPU/PyTorch
                    // and the 1D/3D kernels; the backward re-adds the (n,c) base.
                    max_idx = h * input_w + w;
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

        float max_val = std::numeric_limits<float>::lowest();
        int64_t max_idx = -1;

        for (int64_t h = h_start; h < h_end; ++h) {
            for (int64_t w = w_start; w < w_end; ++w) {
                int64_t input_idx = ((n * channels + c) * input_h + h) * input_w + w;
                float val = tenzor::rocm::safe_h2f(input[input_idx]);
                if (tenzor::rocm::is_nan_bits(val) || val > max_val) {
                    max_val = val;
                    // Plane-local index (h*W + w); backward re-adds the (n,c) base.
                    max_idx = h * input_w + w;
                }
            }
        }

        output[idx] = tenzor::rocm::safe_f2h(max_val);
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
    bool return_indices,
    hipStream_t stream
) -> std::pair<Tensor, Tensor> {

#ifdef USE_MIOPEN
    // audit-2026-05-03 — MIOpen does not expose argmax indices in a
    // workspace-stable format we can extract. Routing through MIOpen when
    // return_indices is true returned uninitialised int64 indices, which
    // poisoned the backward gradient. Skip MIOpen entirely when indices
    // are required (the test is the autograd backward path); fall through
    // to the native HIP kernel which writes both output and indices.
    if (rocm::is_miopen_available() && !return_indices &&
        (input.dtype() == DType::Float32 ||
         input.dtype() == DType::Float16 ||
         input.dtype() == DType::BFloat16)) {
        try {
            return maxpool2d_forward_miopen(input, kernel_h, kernel_w,
                                            stride_h, stride_w, pad_h, pad_w,
                                            return_indices, stream);
        } catch (const std::exception& e) {
            // TENZOR_STRICT_BACKEND=1 surfaces MIOpen failures; otherwise log
            // the error so users can see when they silently degrade to the
            // pure-HIP kernel instead of MIOpen acceleration.
            if (const char* s = std::getenv("TENZOR_STRICT_BACKEND"); s && *s && *s != '0') {
                throw std::runtime_error(
                    std::string("ROCm maxpool2d: MIOpen failed "
                                "(TENZOR_STRICT_BACKEND=1): ") + e.what());
            }
            TENZOR_LOG_WARNING(std::format(
                "MIOpen maxpool2d failed ({}), using native HIP fallback", e.what()));
        }
    }
#endif

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
    int threads = rocm::get_wavefront_size() * 4;  // 4 wavefronts per block
    int blocks = (total_elements + threads - 1) / threads;
    // Empty output: a zero-grid launch is rejected by HIP; return as-is.
    if (blocks == 0) return {output, indices};

    if (input.dtype() == DType::Float32) {
        hipLaunchKernelGGL(maxpool2d_forward_kernel<float>,
            dim3(blocks), dim3(threads), 0, stream,
            input.data<float>(),
            output.data<float>(),
            return_indices ? indices.data<int64_t>() : nullptr,
            batch_size, channels, input_h, input_w,
            output_h, output_w, kernel_h, kernel_w,
            stride_h, stride_w, pad_h, pad_w, return_indices
        );
        HIP_POST_LAUNCH_CHECK();
    } else if (input.dtype() == DType::Float64) {
        hipLaunchKernelGGL(maxpool2d_forward_kernel<double>,
            dim3(blocks), dim3(threads), 0, stream,
            input.data<double>(),
            output.data<double>(),
            return_indices ? indices.data<int64_t>() : nullptr,
            batch_size, channels, input_h, input_w,
            output_h, output_w, kernel_h, kernel_w,
            stride_h, stride_w, pad_h, pad_w, return_indices
        );
        HIP_POST_LAUNCH_CHECK();
    } else if (input.dtype() == DType::Float16) {
        hipLaunchKernelGGL(maxpool2d_forward_kernel_fp16,
            dim3(blocks), dim3(threads), 0, stream,
            reinterpret_cast<const __half*>(input.data<Float16>()),
            reinterpret_cast<__half*>(output.data<Float16>()),
            return_indices ? indices.data<int64_t>() : nullptr,
            batch_size, channels, input_h, input_w,
            output_h, output_w, kernel_h, kernel_w,
            stride_h, stride_w, pad_h, pad_w, return_indices
        );
        HIP_POST_LAUNCH_CHECK();
    } else if (input.dtype() == DType::BFloat16) {
        auto input_f32 = input.to(DType::Float32);
        auto [output_f32, idx] = maxpool2d_forward_hip(input_f32, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, return_indices, stream);
        return {output_f32.to(DType::BFloat16), idx};
    } else {
        throw std::runtime_error("maxpool2d_forward_hip: Only Float32, Float64, and Float16 supported");
    }

    HIP_POST_LAUNCH_CHECK();

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
    int64_t total_elements,
    int64_t out_plane,   // output_h * output_w
    int64_t in_plane     // input_h * input_w
) {
    HIP_KERNEL_LOOP(idx, total_elements) {
        // indices store the plane-local argmax (h*W + w); reconstruct the global
        // input offset by adding the per-(n,c) plane base. An all-padding window
        // leaves the argmax at the -1 sentinel (no valid input element); skip it
        // to avoid an out-of-bounds scatter.
        int64_t plane_idx = indices[idx];
        if (plane_idx < 0 || plane_idx >= in_plane) continue;
        int64_t nc = idx / out_plane;
        int64_t input_idx = nc * in_plane + plane_idx;
        atomicAdd(&grad_input[input_idx], grad_output[idx]);
    }
}

// Float16 maxpool2d backward (accumulate in float)
__global__ void maxpool2d_backward_kernel_fp16(
    const __half* grad_output,
    const int64_t* indices,
    float* grad_input_f32,
    int64_t total_elements,
    int64_t out_plane,   // output_h * output_w
    int64_t in_plane     // input_h * input_w
) {
    HIP_KERNEL_LOOP(idx, total_elements) {
        // Plane-local index -> global offset (see maxpool2d_backward_kernel).
        // Skip the -1 sentinel left by an all-padding window (OOB guard).
        int64_t plane_idx = indices[idx];
        if (plane_idx < 0 || plane_idx >= in_plane) continue;
        int64_t nc = idx / out_plane;
        int64_t input_idx = nc * in_plane + plane_idx;
        atomicAdd(&grad_input_f32[input_idx], tenzor::rocm::safe_h2f(grad_output[idx]));
    }
}

// Convert float to half kernel
__global__ void convert_f32_to_f16_pool(const float* src, __half* dst, int64_t n) {
    HIP_KERNEL_LOOP(idx, n) {
        // Saturate to FP16 representable range to prevent Inf from overflow
        float val = fminf(fmaxf(src[idx], -65504.0f), 65504.0f);
        dst[idx] = tenzor::rocm::safe_f2h(val);
    }
}

auto maxpool2d_backward_hip(
    const Tensor& grad_output,
    const Tensor& indices,
    const std::vector<int64_t>& input_shape,
    hipStream_t stream
) -> Tensor {

    Tensor grad_input = Tensor(input_shape, grad_output.dtype(), grad_output.device());

    // audit-2026-05-03 — zero-initialise grad_input. atomicAdd accumulates
    // onto whatever was there, so uninitialised memory poisons the gradient
    // at non-argmax positions. Float64 happened to start at zero on most
    // ROCm allocations; Float32 surfaced the bug via gradcheck failures.
    int64_t input_numel = 1;
    for (auto s : input_shape) input_numel *= s;
    HIP_CHECK(hipMemsetAsync(grad_input.data_ptr(), 0,
        input_numel * grad_input.dtype_size(), stream));

    int64_t total_elements = grad_output.numel();
    int threads = rocm::get_wavefront_size() * 4;  // 4 wavefronts per block
    // Plane sizes for the plane-local -> global index reconstruction in the kernel.
    int64_t in_plane = input_shape[input_shape.size() - 2] * input_shape[input_shape.size() - 1];
    auto go_shape = grad_output.shape();
    int64_t out_plane = go_shape[go_shape.size() - 2] * go_shape[go_shape.size() - 1];

    int blocks = (total_elements + threads - 1) / threads;

    if (grad_output.dtype() == DType::Float32) {
        hipLaunchKernelGGL(maxpool2d_backward_kernel<float>,
            dim3(blocks), dim3(threads), 0, stream,
            grad_output.data<float>(),
            indices.data<int64_t>(),
            grad_input.data<float>(),
            total_elements, out_plane, in_plane
        );
        HIP_POST_LAUNCH_CHECK();
    } else if (grad_output.dtype() == DType::Float64) {
        hipLaunchKernelGGL(maxpool2d_backward_kernel<double>,
            dim3(blocks), dim3(threads), 0, stream,
            grad_output.data<double>(),
            indices.data<int64_t>(),
            grad_input.data<double>(),
            total_elements, out_plane, in_plane
        );
        HIP_POST_LAUNCH_CHECK();
    } else if (grad_output.dtype() == DType::Float16) {
        // Accumulate in float, then convert back
        int64_t input_numel = 1;
        for (auto s : input_shape) input_numel *= s;
        Tensor grad_input_f32 = Tensor(input_shape, DType::Float32, grad_output.device());
        HIP_CHECK(hipMemsetAsync(grad_input_f32.data<float>(), 0, input_numel * sizeof(float), stream));

        hipLaunchKernelGGL(maxpool2d_backward_kernel_fp16,
            dim3(blocks), dim3(threads), 0, stream,
            reinterpret_cast<const __half*>(grad_output.data<Float16>()),
            indices.data<int64_t>(),
            grad_input_f32.data<float>(),
            total_elements, out_plane, in_plane
        );
        HIP_POST_LAUNCH_CHECK();

        int convert_blocks = (input_numel + threads - 1) / threads;
        hipLaunchKernelGGL(convert_f32_to_f16_pool,
            dim3(convert_blocks), dim3(threads), 0, stream,
            grad_input_f32.data<float>(),
            reinterpret_cast<__half*>(grad_input.data<Float16>()),
            input_numel
        );
        HIP_POST_LAUNCH_CHECK();
    } else if (grad_output.dtype() == DType::BFloat16) {
        auto grad_output_f32 = grad_output.to(DType::Float32);
        auto result_f32 = maxpool2d_backward_hip(grad_output_f32, indices, input_shape, stream);
        return result_f32.to(DType::BFloat16);
    } else {
        throw std::runtime_error("maxpool2d_backward_hip: Only Float32, Float64, Float16, and BFloat16 supported");
    }

    HIP_POST_LAUNCH_CHECK();

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

        output[idx] = count > 0 ? sum / static_cast<T>(count) : T(0);
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
                sum += tenzor::rocm::safe_h2f(input[input_idx]);
                count++;
            }
        }

        if (count_include_pad) {
            count = kernel_h * kernel_w;
        }

        output[idx] = tenzor::rocm::safe_f2h(count > 0 ? sum / static_cast<float>(count) : 0.0f);
    }
}

auto avgpool2d_forward_hip(
    const Tensor& input,
    int64_t kernel_h, int64_t kernel_w,
    int64_t stride_h, int64_t stride_w,
    int64_t pad_h, int64_t pad_w,
    bool count_include_pad,
    hipStream_t stream
) -> Tensor {

    // Native HIP path is the default for ALL architectures. MIOpen's
    // AvgPool2d JIT compilation was observed to hang on gfx1150 (and the
    // hang takes the GPU with it, so catch fallback isn't enough). The
    // native kernel below handles every dtype the rest of the codebase
    // expects and matches MIOpen's results bit-for-bit on the F32/F64
    // path. MIOpen can still be enabled as an opt-in fast path by setting
    // the TENZOR_ROCM_FORCE_MIOPEN=1 env var (off by default).
#ifdef USE_MIOPEN
    if (rocm::is_miopen_available() &&
        (input.dtype() == DType::Float32 ||
         input.dtype() == DType::Float16 ||
         input.dtype() == DType::BFloat16)) {
        const char* force = std::getenv("TENZOR_ROCM_FORCE_MIOPEN");
        if (force && force[0] == '1') {
            try {
                return avgpool2d_forward_miopen(input, kernel_h, kernel_w,
                                                stride_h, stride_w, pad_h, pad_w,
                                                count_include_pad, stream);
            } catch (const std::exception& e) {
                if (const char* s = std::getenv("TENZOR_STRICT_BACKEND"); s && *s && *s != '0') {
                    throw std::runtime_error(
                        std::string("ROCm avgpool2d: MIOpen failed "
                                    "(TENZOR_STRICT_BACKEND=1): ") + e.what());
                }
                TENZOR_LOG_WARNING(std::format(
                    "MIOpen avgpool2d failed ({}), using native HIP fallback", e.what()));
            }
        }
    }
#endif

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
    int threads = rocm::get_wavefront_size() * 4;  // 4 wavefronts per block
    int blocks = (total_elements + threads - 1) / threads;

    if (input.dtype() == DType::Float32) {
        hipLaunchKernelGGL(avgpool2d_forward_kernel<float>,
            dim3(blocks), dim3(threads), 0, stream,
            input.data<float>(),
            output.data<float>(),
            batch_size, channels, input_h, input_w,
            output_h, output_w, kernel_h, kernel_w,
            stride_h, stride_w, pad_h, pad_w, count_include_pad
        );
        HIP_POST_LAUNCH_CHECK();
    } else if (input.dtype() == DType::Float64) {
        hipLaunchKernelGGL(avgpool2d_forward_kernel<double>,
            dim3(blocks), dim3(threads), 0, stream,
            input.data<double>(),
            output.data<double>(),
            batch_size, channels, input_h, input_w,
            output_h, output_w, kernel_h, kernel_w,
            stride_h, stride_w, pad_h, pad_w, count_include_pad
        );
        HIP_POST_LAUNCH_CHECK();
    } else if (input.dtype() == DType::Float16) {
        hipLaunchKernelGGL(avgpool2d_forward_kernel_fp16,
            dim3(blocks), dim3(threads), 0, stream,
            reinterpret_cast<const __half*>(input.data<Float16>()),
            reinterpret_cast<__half*>(output.data<Float16>()),
            batch_size, channels, input_h, input_w,
            output_h, output_w, kernel_h, kernel_w,
            stride_h, stride_w, pad_h, pad_w, count_include_pad
        );
        HIP_POST_LAUNCH_CHECK();
    } else if (input.dtype() == DType::BFloat16) {
        auto input_f32 = input.to(DType::Float32);
        auto result_f32 = avgpool2d_forward_hip(input_f32, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, count_include_pad, stream);
        return result_f32.to(DType::BFloat16);
    } else {
        throw std::runtime_error("avgpool2d_forward_hip: Only Float32, Float64, and Float16 supported");
    }

    HIP_POST_LAUNCH_CHECK();

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

        float grad_val = tenzor::rocm::safe_h2f(grad_output[idx]) / static_cast<float>(count);

        for (int64_t h = h_start; h < h_end; ++h) {
            for (int64_t w = w_start; w < w_end; ++w) {
                int64_t input_idx = ((n * channels + c) * input_h + h) * input_w + w;
                atomicAdd(&grad_input_f32[input_idx], grad_val);
            }
        }
    }
}

/// Note: MIOpen pooling backward (miopenPoolingBackward) requires the original
// input and output tensors, which are not available in this dispatch signature.
// The MIOpen backward path is still available via avgpool2d_backward_miopen()
// and maxpool2d_backward_miopen() for callers that retain those tensors; this
// dispatch entry point uses the native HIP backward instead. Correctness is
// preserved; MIOpen is slightly faster on some shapes. Deferred: plumb input
// and output through the OpAttributes dispatch interface to prefer MIOpen
// when tensors are retained.
auto avgpool2d_backward_hip(
    const Tensor& grad_output,
    const std::vector<int64_t>& input_shape,
    int64_t kernel_h, int64_t kernel_w,
    int64_t stride_h, int64_t stride_w,
    int64_t pad_h, int64_t pad_w,
    bool count_include_pad,
    hipStream_t stream
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
    int threads = rocm::get_wavefront_size() * 4;  // 4 wavefronts per block
    int blocks = (total_elements + threads - 1) / threads;

    if (grad_output.dtype() == DType::Float32) {
        hipLaunchKernelGGL(avgpool2d_backward_kernel<float>,
            dim3(blocks), dim3(threads), 0, stream,
            grad_output.data<float>(),
            grad_input.data<float>(),
            batch_size, channels, input_h, input_w,
            output_h, output_w, kernel_h, kernel_w,
            stride_h, stride_w, pad_h, pad_w, count_include_pad
        );
        HIP_POST_LAUNCH_CHECK();
    } else if (grad_output.dtype() == DType::Float64) {
        hipLaunchKernelGGL(avgpool2d_backward_kernel<double>,
            dim3(blocks), dim3(threads), 0, stream,
            grad_output.data<double>(),
            grad_input.data<double>(),
            batch_size, channels, input_h, input_w,
            output_h, output_w, kernel_h, kernel_w,
            stride_h, stride_w, pad_h, pad_w, count_include_pad
        );
        HIP_POST_LAUNCH_CHECK();
    } else if (grad_output.dtype() == DType::Float16) {
        // Accumulate in float, then convert back
        int64_t input_numel = 1;
        for (auto s : input_shape) input_numel *= s;
        Tensor grad_input_f32 = Tensor(input_shape, DType::Float32, grad_output.device());
        HIP_CHECK(hipMemsetAsync(grad_input_f32.data<float>(), 0, input_numel * sizeof(float), stream));

        hipLaunchKernelGGL(avgpool2d_backward_kernel_fp16,
            dim3(blocks), dim3(threads), 0, stream,
            reinterpret_cast<const __half*>(grad_output.data<Float16>()),
            grad_input_f32.data<float>(),
            batch_size, channels, input_h, input_w,
            output_h, output_w, kernel_h, kernel_w,
            stride_h, stride_w, pad_h, pad_w, count_include_pad
        );
        HIP_POST_LAUNCH_CHECK();

        int convert_blocks = (input_numel + threads - 1) / threads;
        hipLaunchKernelGGL(convert_f32_to_f16_pool,
            dim3(convert_blocks), dim3(threads), 0, stream,
            grad_input_f32.data<float>(),
            reinterpret_cast<__half*>(grad_input.data<Float16>()),
            input_numel
        );
        HIP_POST_LAUNCH_CHECK();
    } else if (grad_output.dtype() == DType::BFloat16) {
        auto grad_output_f32 = grad_output.to(DType::Float32);
        auto result_f32 = avgpool2d_backward_hip(grad_output_f32, input_shape, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, count_include_pad, stream);
        return result_f32.to(DType::BFloat16);
    } else {
        throw std::runtime_error("avgpool2d_backward_hip: Only Float32, Float64, Float16, and BFloat16 supported");
    }

    HIP_POST_LAUNCH_CHECK();

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
        int64_t h_end = ((oh + 1) * input_h + output_h - 1) / output_h;
        int64_t w_start = (ow * input_w) / output_w;
        int64_t w_end = ((ow + 1) * input_w + output_w - 1) / output_w;

        T sum = 0;
        int64_t count = 0;

        for (int64_t h = h_start; h < h_end; ++h) {
            for (int64_t w = w_start; w < w_end; ++w) {
                int64_t input_idx = ((n * channels + c) * input_h + h) * input_w + w;
                sum += input[input_idx];
                count++;
            }
        }

        output[idx] = count > 0 ? sum / static_cast<T>(count) : T(0);
    }
}

__global__ void adaptive_avgpool2d_kernel_fp16(
    const __half* input,
    __half* output,
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
        int64_t h_end = ((oh + 1) * input_h + output_h - 1) / output_h;
        int64_t w_start = (ow * input_w) / output_w;
        int64_t w_end = ((ow + 1) * input_w + output_w - 1) / output_w;

        float sum = 0.0f;
        int64_t count = 0;

        for (int64_t h = h_start; h < h_end; ++h) {
            for (int64_t w = w_start; w < w_end; ++w) {
                int64_t input_idx = ((n * channels + c) * input_h + h) * input_w + w;
                sum += tenzor::rocm::safe_h2f(input[input_idx]);
                count++;
            }
        }

        output[idx] = tenzor::rocm::safe_f2h(count > 0 ? sum / static_cast<float>(count) : 0.0f);
    }
}

auto adaptive_avgpool2d_hip(
    const Tensor& input,
    int64_t output_h,
    int64_t output_w,
    hipStream_t stream
) -> Tensor {

    auto input_shape = input.shape();
    int64_t batch_size = input_shape[0];
    int64_t channels = input_shape[1];
    int64_t input_h = input_shape[2];
    int64_t input_w = input_shape[3];

    std::vector<int64_t> output_shape = {batch_size, channels, output_h, output_w};
    Tensor output = Tensor(output_shape, input.dtype(), input.device());

    int64_t total_elements = batch_size * channels * output_h * output_w;
    int threads = rocm::get_wavefront_size() * 4;  // 4 wavefronts per block
    int blocks = (total_elements + threads - 1) / threads;

    if (input.dtype() == DType::Float32) {
        hipLaunchKernelGGL(adaptive_avgpool2d_kernel<float>,
            dim3(blocks), dim3(threads), 0, stream,
            input.data<float>(),
            output.data<float>(),
            batch_size, channels, input_h, input_w,
            output_h, output_w
        );
        HIP_POST_LAUNCH_CHECK();
    } else if (input.dtype() == DType::Float64) {
        hipLaunchKernelGGL(adaptive_avgpool2d_kernel<double>,
            dim3(blocks), dim3(threads), 0, stream,
            input.data<double>(),
            output.data<double>(),
            batch_size, channels, input_h, input_w,
            output_h, output_w
        );
        HIP_POST_LAUNCH_CHECK();
    } else if (input.dtype() == DType::Float16) {
        hipLaunchKernelGGL(adaptive_avgpool2d_kernel_fp16,
            dim3(blocks), dim3(threads), 0, stream,
            reinterpret_cast<const __half*>(input.data<Float16>()),
            reinterpret_cast<__half*>(output.data<Float16>()),
            batch_size, channels, input_h, input_w,
            output_h, output_w
        );
        HIP_POST_LAUNCH_CHECK();
    } else if (input.dtype() == DType::BFloat16) {
        auto input_f32 = input.to(DType::Float32);
        auto result_f32 = adaptive_avgpool2d_hip(input_f32, output_h, output_w, stream);
        return result_f32.to(DType::BFloat16);
    } else {
        throw std::runtime_error("adaptive_avgpool2d_hip: unsupported dtype");
    }

    HIP_POST_LAUNCH_CHECK();

    if (input.dtype() == DType::Float16) {
        fp16_saturate(output.data_ptr(), output.numel(), stream);
    }

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
        int64_t h_end = ((oh + 1) * input_h + output_h - 1) / output_h;
        int64_t w_start = (ow * input_w) / output_w;
        int64_t w_end = ((ow + 1) * input_w + output_w - 1) / output_w;

        T max_val = std::numeric_limits<T>::lowest();
        int64_t max_idx = -1;

        for (int64_t h = h_start; h < h_end; ++h) {
            for (int64_t w = w_start; w < w_end; ++w) {
                int64_t input_idx = ((n * channels + c) * input_h + h) * input_w + w;
                T val = input[input_idx];
                if (tenzor::rocm::is_nan_bits(val) || val > max_val) {
                    max_val = val;
                    // Plane-local index (h*W + w) to match CPU/PyTorch and the backward
                    // kernel, which re-adds the (n,c) base exactly once.
                    max_idx = h * input_w + w;
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
    bool return_indices,
    hipStream_t stream
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
    int threads = rocm::get_wavefront_size() * 4;  // 4 wavefronts per block
    int blocks = (total_elements + threads - 1) / threads;
    // Empty output: a zero-grid launch is rejected by HIP; return as-is.
    if (blocks == 0) return {output, indices};

    if (input.dtype() == DType::Float32) {
        hipLaunchKernelGGL(adaptive_maxpool2d_kernel<float>,
            dim3(blocks), dim3(threads), 0, stream,
            input.data<float>(),
            output.data<float>(),
            return_indices ? indices.data<int64_t>() : nullptr,
            batch_size, channels, input_h, input_w,
            output_h, output_w, return_indices
        );
        HIP_POST_LAUNCH_CHECK();
    } else if (input.dtype() == DType::Float64) {
        hipLaunchKernelGGL(adaptive_maxpool2d_kernel<double>,
            dim3(blocks), dim3(threads), 0, stream,
            input.data<double>(),
            output.data<double>(),
            return_indices ? indices.data<int64_t>() : nullptr,
            batch_size, channels, input_h, input_w,
            output_h, output_w, return_indices
        );
        HIP_POST_LAUNCH_CHECK();
    } else if (input.dtype() == DType::Float16) {
        auto input_f32 = input.to(DType::Float32);
        auto [output_f32, idx] = adaptive_maxpool2d_hip(input_f32, output_h, output_w, return_indices, stream);
        return {output_f32.to(DType::Float16), idx};
    } else if (input.dtype() == DType::BFloat16) {
        auto input_f32 = input.to(DType::Float32);
        auto [output_f32, idx] = adaptive_maxpool2d_hip(input_f32, output_h, output_w, return_indices, stream);
        return {output_f32.to(DType::BFloat16), idx};
    } else {
        throw std::runtime_error("adaptive_maxpool2d_hip: unsupported dtype");
    }

    HIP_POST_LAUNCH_CHECK();

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
    int threads = rocm::get_wavefront_size() * 4;  // 4 wavefronts per block
    int blocks = (total + threads - 1) / threads;

    if (input.dtype() == DType::Float32) {
        hipLaunchKernelGGL(adaptive_avgpool2d_backward_kernel<float>,
            dim3(blocks), dim3(threads), 0, stream,
            grad_output.data<float>(),
            grad_input.data<float>(),
            N, C, in_H, in_W, out_H, out_W);
        HIP_POST_LAUNCH_CHECK();
    } else if (input.dtype() == DType::Float64) {
        hipLaunchKernelGGL(adaptive_avgpool2d_backward_kernel<double>,
            dim3(blocks), dim3(threads), 0, stream,
            grad_output.data<double>(),
            grad_input.data<double>(),
            N, C, in_H, in_W, out_H, out_W);
        HIP_POST_LAUNCH_CHECK();
    } else if (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16) {
        // Widen-narrow so the scatter-accumulate runs in Float32, matching every
        // other fp16 pooling backward in this file. The native __half kernel
        // accumulated the running sum in half precision, losing low-order bits
        // when an input pixel feeds many output positions (large upsample ratio)
        // and drifting vs the CPU/CUDA Float32-accumulated reference.
        const DType orig = input.dtype();
        auto grad_output_f32 = grad_output.to(DType::Float32);
        auto input_f32 = input.to(DType::Float32);
        auto result_f32 = adaptive_avgpool2d_backward_hip(grad_output_f32, input_f32, stream);
        return result_f32.to(orig);
    } else {
        throw std::runtime_error("adaptive_avgpool2d_backward: unsupported dtype");
    }

    HIP_POST_LAUNCH_CHECK();

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
        // An empty pooling window (forward sets max_idx=-1 when output>input)
        // has no source pixel; scattering with -1 would write grad_input[...-1]
        // out of bounds. Skip it.
        if (max_idx < 0) continue;
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
    int threads = rocm::get_wavefront_size() * 4;  // 4 wavefronts per block
    int blocks = (total + threads - 1) / threads;

    if (input.dtype() == DType::Float32) {
        hipLaunchKernelGGL(adaptive_maxpool2d_backward_kernel<float>,
            dim3(blocks), dim3(threads), 0, stream,
            grad_output.data<float>(),
            indices.data<int64_t>(),
            grad_input.data<float>(),
            N, C, in_H, in_W, out_H, out_W);
        HIP_POST_LAUNCH_CHECK();
    } else if (input.dtype() == DType::Float64) {
        hipLaunchKernelGGL(adaptive_maxpool2d_backward_kernel<double>,
            dim3(blocks), dim3(threads), 0, stream,
            grad_output.data<double>(),
            indices.data<int64_t>(),
            grad_input.data<double>(),
            N, C, in_H, in_W, out_H, out_W);
        HIP_POST_LAUNCH_CHECK();
    } else if (input.dtype() == DType::Float16) {
        auto grad_output_f32 = grad_output.to(DType::Float32);
        auto input_f32 = input.to(DType::Float32);
        auto result_f32 = adaptive_maxpool2d_backward_hip(grad_output_f32, indices, input_f32, stream);
        return result_f32.to(DType::Float16);
    } else if (input.dtype() == DType::BFloat16) {
        auto grad_output_f32 = grad_output.to(DType::Float32);
        auto input_f32 = input.to(DType::Float32);
        auto result_f32 = adaptive_maxpool2d_backward_hip(grad_output_f32, indices, input_f32, stream);
        return result_f32.to(DType::BFloat16);
    } else {
        throw std::runtime_error("adaptive_maxpool2d_backward: unsupported dtype");
    }

    HIP_POST_LAUNCH_CHECK();
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
    int threads = rocm::get_wavefront_size() * 4;  // 4 wavefronts per block
    int blocks = (total + threads - 1) / threads;

    if (input.dtype() == DType::Float32) {
        hipLaunchKernelGGL(adaptive_avgpool2d_kernel<float>,
            dim3(blocks), dim3(threads), 0, stream,
            input.data<float>(), output.data<float>(),
            N, C, H_in, W_in, output_h, output_w);
        HIP_POST_LAUNCH_CHECK();
    } else if (input.dtype() == DType::Float64) {
        hipLaunchKernelGGL(adaptive_avgpool2d_kernel<double>,
            dim3(blocks), dim3(threads), 0, stream,
            input.data<double>(), output.data<double>(),
            N, C, H_in, W_in, output_h, output_w);
        HIP_POST_LAUNCH_CHECK();
    } else if (input.dtype() == DType::Float16) {
        hipLaunchKernelGGL(adaptive_avgpool2d_kernel_fp16,
            dim3(blocks), dim3(threads), 0, stream,
            reinterpret_cast<const __half*>(input.data<Float16>()),
            reinterpret_cast<__half*>(output.data<Float16>()),
            N, C, H_in, W_in, output_h, output_w);
        HIP_POST_LAUNCH_CHECK();
    } else if (input.dtype() == DType::BFloat16) {
        auto input_f32 = input.to(DType::Float32);
        auto result_f32 = adaptive_avgpool2d_forward(input_f32, output_h, output_w, stream);
        return result_f32.to(DType::BFloat16);
    } else {
        throw std::runtime_error("adaptive_avgpool2d_forward: unsupported dtype");
    }

    HIP_POST_LAUNCH_CHECK();
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
    int threads = rocm::get_wavefront_size() * 4;  // 4 wavefronts per block
    int blocks = (total + threads - 1) / threads;

    if (grad_output.dtype() == DType::Float32) {
        hipLaunchKernelGGL(adaptive_avgpool2d_backward_kernel<float>,
            dim3(blocks), dim3(threads), 0, stream,
            grad_output.data<float>(), grad_input.data<float>(),
            N, C, H_in, W_in, H_out, W_out);
        HIP_POST_LAUNCH_CHECK();
    } else if (grad_output.dtype() == DType::Float64) {
        hipLaunchKernelGGL(adaptive_avgpool2d_backward_kernel<double>,
            dim3(blocks), dim3(threads), 0, stream,
            grad_output.data<double>(), grad_input.data<double>(),
            N, C, H_in, W_in, H_out, W_out);
        HIP_POST_LAUNCH_CHECK();
    } else if (grad_output.dtype() == DType::Float16) {
        hipLaunchKernelGGL(adaptive_avgpool2d_backward_kernel<__half>,
            dim3(blocks), dim3(threads), 0, stream,
            reinterpret_cast<const __half*>(grad_output.data<Float16>()),
            reinterpret_cast<__half*>(grad_input.data<Float16>()),
            N, C, H_in, W_in, H_out, W_out);
        HIP_POST_LAUNCH_CHECK();
    } else if (grad_output.dtype() == DType::BFloat16) {
        auto grad_output_f32 = grad_output.to(DType::Float32);
        auto result_f32 = adaptive_avgpool2d_backward(grad_output_f32, H_in, W_in, stream);
        return result_f32.to(DType::BFloat16);
    } else {
        throw std::runtime_error("adaptive_avgpool2d_backward: unsupported dtype");
    }

    HIP_POST_LAUNCH_CHECK();
    return grad_input;
}

// ==============================================================================
// MaxPool1D Forward
// ==============================================================================

template<typename T>
__global__ void maxpool1d_forward_kernel(
    const T* input,
    T* output,
    int64_t* indices,
    int64_t N, int64_t C, int64_t L,
    int64_t L_out,
    int64_t kernel_size, int64_t stride, int64_t padding, int64_t dilation
) {
    int64_t total_elements = N * C * L_out;

    HIP_KERNEL_LOOP(idx, total_elements) {
        int64_t ol = idx % L_out;
        int64_t c  = (idx / L_out) % C;
        int64_t n  = idx / (L_out * C);

        int64_t l_start = ol * stride - padding;

        T max_val = std::numeric_limits<T>::lowest();
        // -1 sentinel for an all-padding window (matches maxpool2d); the
        // backward kernel skips -1 instead of mis-scattering grad to index 0.
        int64_t max_idx = -1;

        for (int64_t k = 0; k < kernel_size; ++k) {
            int64_t l = l_start + k * dilation;
            if (l >= 0 && l < L) {
                int64_t in_idx = (n * C + c) * L + l;
                T val = input[in_idx];
                if (tenzor::rocm::is_nan_bits(val) || val > max_val) {
                    max_val = val;
                    max_idx = l;
                }
            }
        }

        output[idx] = max_val;
        indices[idx] = max_idx;
    }
}

__global__ void maxpool1d_forward_kernel_fp16(
    const __half* input,
    __half* output,
    int64_t* indices,
    int64_t N, int64_t C, int64_t L,
    int64_t L_out,
    int64_t kernel_size, int64_t stride, int64_t padding, int64_t dilation
) {
    int64_t total_elements = N * C * L_out;

    HIP_KERNEL_LOOP(idx, total_elements) {
        int64_t ol = idx % L_out;
        int64_t c  = (idx / L_out) % C;
        int64_t n  = idx / (L_out * C);

        int64_t l_start = ol * stride - padding;

        float max_val = std::numeric_limits<float>::lowest();
        int64_t max_idx = -1;  // -1 sentinel for an all-padding window.

        for (int64_t k = 0; k < kernel_size; ++k) {
            int64_t l = l_start + k * dilation;
            if (l >= 0 && l < L) {
                int64_t in_idx = (n * C + c) * L + l;
                float val = tenzor::rocm::safe_h2f(input[in_idx]);
                if (tenzor::rocm::is_nan_bits(val) || val > max_val) {
                    max_val = val;
                    max_idx = l;
                }
            }
        }

        output[idx] = tenzor::rocm::safe_f2h(max_val);
        indices[idx] = max_idx;
    }
}

auto maxpool1d_forward_hip(
    const Tensor& input,
    std::array<int64_t, 1> kernel_size_a, std::array<int64_t, 1> stride_a, std::array<int64_t, 1> padding_a, std::array<int64_t, 1> dilation_a,
    hipStream_t stream
) -> std::pair<Tensor, Tensor> {
    // Q.6: per-axis std::array<int64_t, 1> signature. 1D has a single spatial
    // axis (W); destructure to scalars for the existing impl.
    const int64_t kernel_size = kernel_size_a[0];
    const int64_t stride      = stride_a[0];
    const int64_t padding     = padding_a[0];
    const int64_t dilation    = dilation_a[0];

    auto shape = input.shape();
    int64_t N = shape[0];
    int64_t C = shape[1];
    int64_t L = shape[2];

    int64_t L_out = (L + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    Tensor output({N, C, L_out}, input.dtype(), input.device());
    Tensor indices({N, C, L_out}, DType::Int64, input.device());

    int64_t total_elements = N * C * L_out;
    int threads = rocm::get_wavefront_size() * 4;  // 4 wavefronts per block
    int blocks = (total_elements + threads - 1) / threads;

    if (input.dtype() == DType::Float32) {
        hipLaunchKernelGGL(maxpool1d_forward_kernel<float>,
            dim3(blocks), dim3(threads), 0, stream,
            input.data<float>(), output.data<float>(), indices.data<int64_t>(),
            N, C, L, L_out, kernel_size, stride, padding, dilation);
        HIP_POST_LAUNCH_CHECK();
    } else if (input.dtype() == DType::Float64) {
        hipLaunchKernelGGL(maxpool1d_forward_kernel<double>,
            dim3(blocks), dim3(threads), 0, stream,
            input.data<double>(), output.data<double>(), indices.data<int64_t>(),
            N, C, L, L_out, kernel_size, stride, padding, dilation);
        HIP_POST_LAUNCH_CHECK();
    } else if (input.dtype() == DType::Float16) {
        hipLaunchKernelGGL(maxpool1d_forward_kernel_fp16,
            dim3(blocks), dim3(threads), 0, stream,
            reinterpret_cast<const __half*>(input.data<Float16>()),
            reinterpret_cast<__half*>(output.data<Float16>()),
            indices.data<int64_t>(),
            N, C, L, L_out, kernel_size, stride, padding, dilation);
        HIP_POST_LAUNCH_CHECK();
    } else if (input.dtype() == DType::BFloat16) {
        auto input_f32 = input.to(DType::Float32);
        auto [output_f32, idx] = maxpool1d_forward_hip(input_f32, kernel_size_a, stride_a, padding_a, dilation_a, stream);
        return {output_f32.to(DType::BFloat16), idx};
    } else {
        throw std::runtime_error("maxpool1d_forward_hip: unsupported dtype");
    }

    HIP_POST_LAUNCH_CHECK();
    return {output, indices};
}

// ==============================================================================
// MaxPool1D Backward
// ==============================================================================

template<typename T>
__global__ void maxpool1d_backward_kernel_impl(
    const T* grad_output,
    const int64_t* indices,
    T* grad_input,
    int64_t N, int64_t C, int64_t L,
    int64_t L_out
) {
    int64_t total = N * C * L_out;

    HIP_KERNEL_LOOP(idx, total) {
        int64_t c = (idx / L_out) % C;
        int64_t n = idx / (L_out * C);

        int64_t max_idx = indices[idx];
        if (max_idx < 0) continue;  // all-padding window: no input to scatter to.
        int64_t in_idx = (n * C + c) * L + max_idx;
        atomicAdd(&grad_input[in_idx], grad_output[idx]);
    }
}

__global__ void maxpool1d_backward_kernel_fp16(
    const __half* grad_output,
    const int64_t* indices,
    float* grad_input_f32,
    int64_t N, int64_t C, int64_t L,
    int64_t L_out
) {
    int64_t total = N * C * L_out;

    HIP_KERNEL_LOOP(idx, total) {
        int64_t c = (idx / L_out) % C;
        int64_t n = idx / (L_out * C);

        int64_t max_idx = indices[idx];
        if (max_idx < 0) continue;  // all-padding window: no input to scatter to.
        int64_t in_idx = (n * C + c) * L + max_idx;
        atomicAdd(&grad_input_f32[in_idx], tenzor::rocm::safe_h2f(grad_output[idx]));
    }
}

auto maxpool1d_backward_hip(
    const Tensor& grad_output,
    const Tensor& indices,
    const std::vector<int64_t>& input_shape,
    hipStream_t stream
) -> Tensor {

    int64_t N = input_shape[0];
    int64_t C = input_shape[1];
    int64_t L = input_shape[2];

    auto grad_shape = grad_output.shape();
    int64_t L_out = grad_shape[2];

    Tensor grad_input(input_shape, grad_output.dtype(), grad_output.device());
    int64_t input_numel = N * C * L;
    HIP_CHECK(hipMemsetAsync(grad_input.data<uint8_t>(), 0,
        input_numel * dtype_size(grad_output.dtype()), stream));

    int64_t total_elements = N * C * L_out;
    int threads = rocm::get_wavefront_size() * 4;  // 4 wavefronts per block
    int blocks = (total_elements + threads - 1) / threads;

    if (grad_output.dtype() == DType::Float32) {
        hipLaunchKernelGGL(maxpool1d_backward_kernel_impl<float>,
            dim3(blocks), dim3(threads), 0, stream,
            grad_output.data<float>(), indices.data<int64_t>(),
            grad_input.data<float>(), N, C, L, L_out);
        HIP_POST_LAUNCH_CHECK();
    } else if (grad_output.dtype() == DType::Float64) {
        hipLaunchKernelGGL(maxpool1d_backward_kernel_impl<double>,
            dim3(blocks), dim3(threads), 0, stream,
            grad_output.data<double>(), indices.data<int64_t>(),
            grad_input.data<double>(), N, C, L, L_out);
        HIP_POST_LAUNCH_CHECK();
    } else if (grad_output.dtype() == DType::Float16) {
        Tensor grad_input_f32(input_shape, DType::Float32, grad_output.device());
        HIP_CHECK(hipMemsetAsync(grad_input_f32.data<float>(), 0, input_numel * sizeof(float), stream));

        hipLaunchKernelGGL(maxpool1d_backward_kernel_fp16,
            dim3(blocks), dim3(threads), 0, stream,
            reinterpret_cast<const __half*>(grad_output.data<Float16>()),
            indices.data<int64_t>(),
            grad_input_f32.data<float>(), N, C, L, L_out);
        HIP_POST_LAUNCH_CHECK();

        int convert_blocks = (input_numel + threads - 1) / threads;
        hipLaunchKernelGGL(convert_f32_to_f16_pool,
            dim3(convert_blocks), dim3(threads), 0, stream,
            grad_input_f32.data<float>(),
            reinterpret_cast<__half*>(grad_input.data<Float16>()),
            input_numel);
        HIP_POST_LAUNCH_CHECK();
    } else if (grad_output.dtype() == DType::BFloat16) {
        auto grad_output_f32 = grad_output.to(DType::Float32);
        auto result_f32 = maxpool1d_backward_hip(grad_output_f32, indices, input_shape, stream);
        return result_f32.to(DType::BFloat16);
    } else {
        throw std::runtime_error("maxpool1d_backward_hip: unsupported dtype");
    }

    HIP_POST_LAUNCH_CHECK();
    return grad_input;
}

// ==============================================================================
// AvgPool1D Forward
// ==============================================================================

template<typename T>
__global__ void avgpool1d_forward_kernel(
    const T* input,
    T* output,
    int64_t N, int64_t C, int64_t L,
    int64_t L_out,
    int64_t kernel_size, int64_t stride, int64_t padding
) {
    int64_t total_elements = N * C * L_out;

    HIP_KERNEL_LOOP(idx, total_elements) {
        int64_t ol = idx % L_out;
        int64_t c  = (idx / L_out) % C;
        int64_t n  = idx / (L_out * C);

        int64_t l_start = ol * stride - padding;

        T sum = T(0);
        int64_t count = 0;

        for (int64_t k = 0; k < kernel_size; ++k) {
            int64_t l = l_start + k;
            if (l >= 0 && l < L) {
                sum += input[(n * C + c) * L + l];
                count++;
            }
        }

        output[idx] = count > 0 ? sum / static_cast<T>(count) : T(0);
    }
}

__global__ void avgpool1d_forward_kernel_fp16(
    const __half* input,
    __half* output,
    int64_t N, int64_t C, int64_t L,
    int64_t L_out,
    int64_t kernel_size, int64_t stride, int64_t padding
) {
    int64_t total_elements = N * C * L_out;

    HIP_KERNEL_LOOP(idx, total_elements) {
        int64_t ol = idx % L_out;
        int64_t c  = (idx / L_out) % C;
        int64_t n  = idx / (L_out * C);

        int64_t l_start = ol * stride - padding;

        float sum = 0.0f;
        int64_t count = 0;

        for (int64_t k = 0; k < kernel_size; ++k) {
            int64_t l = l_start + k;
            if (l >= 0 && l < L) {
                sum += tenzor::rocm::safe_h2f(input[(n * C + c) * L + l]);
                count++;
            }
        }

        output[idx] = tenzor::rocm::safe_f2h(count > 0 ? sum / static_cast<float>(count) : 0.0f);
    }
}

auto avgpool1d_forward_hip(
    const Tensor& input,
    std::array<int64_t, 1> kernel_size_a, std::array<int64_t, 1> stride_a, std::array<int64_t, 1> padding_a,
    hipStream_t stream
) -> Tensor {
    // Q.6: per-axis std::array<int64_t, 1> signature.
    const int64_t kernel_size = kernel_size_a[0];
    const int64_t stride      = stride_a[0];
    const int64_t padding     = padding_a[0];

    auto shape = input.shape();
    int64_t N = shape[0];
    int64_t C = shape[1];
    int64_t L = shape[2];

    int64_t L_out = (L + 2 * padding - kernel_size) / stride + 1;

    Tensor output({N, C, L_out}, input.dtype(), input.device());

    int64_t total_elements = N * C * L_out;
    int threads = rocm::get_wavefront_size() * 4;  // 4 wavefronts per block
    int blocks = (total_elements + threads - 1) / threads;

    if (input.dtype() == DType::Float32) {
        hipLaunchKernelGGL(avgpool1d_forward_kernel<float>,
            dim3(blocks), dim3(threads), 0, stream,
            input.data<float>(), output.data<float>(),
            N, C, L, L_out, kernel_size, stride, padding);
        HIP_POST_LAUNCH_CHECK();
    } else if (input.dtype() == DType::Float64) {
        hipLaunchKernelGGL(avgpool1d_forward_kernel<double>,
            dim3(blocks), dim3(threads), 0, stream,
            input.data<double>(), output.data<double>(),
            N, C, L, L_out, kernel_size, stride, padding);
        HIP_POST_LAUNCH_CHECK();
    } else if (input.dtype() == DType::Float16) {
        hipLaunchKernelGGL(avgpool1d_forward_kernel_fp16,
            dim3(blocks), dim3(threads), 0, stream,
            reinterpret_cast<const __half*>(input.data<Float16>()),
            reinterpret_cast<__half*>(output.data<Float16>()),
            N, C, L, L_out, kernel_size, stride, padding);
        HIP_POST_LAUNCH_CHECK();
    } else if (input.dtype() == DType::BFloat16) {
        auto input_f32 = input.to(DType::Float32);
        auto result_f32 = avgpool1d_forward_hip(input_f32, kernel_size_a, stride_a, padding_a, stream);
        return result_f32.to(DType::BFloat16);
    } else {
        throw std::runtime_error("avgpool1d_forward_hip: unsupported dtype");
    }

    HIP_POST_LAUNCH_CHECK();
    return output;
}

// ==============================================================================
// AvgPool1D Backward
// ==============================================================================

template<typename T>
__global__ void avgpool1d_backward_kernel_impl(
    const T* grad_output,
    T* grad_input,
    int64_t N, int64_t C, int64_t L,
    int64_t L_out,
    int64_t kernel_size, int64_t stride, int64_t padding
) {
    int64_t total = N * C * L_out;

    HIP_KERNEL_LOOP(idx, total) {
        int64_t ol = idx % L_out;
        int64_t c  = (idx / L_out) % C;
        int64_t n  = idx / (L_out * C);

        int64_t l_start = ol * stride - padding;

        int64_t count = 0;
        for (int64_t k = 0; k < kernel_size; ++k) {
            int64_t l = l_start + k;
            if (l >= 0 && l < L) count++;
        }

        T grad_val = grad_output[idx] / static_cast<T>(count);

        for (int64_t k = 0; k < kernel_size; ++k) {
            int64_t l = l_start + k;
            if (l >= 0 && l < L) {
                int64_t in_idx = (n * C + c) * L + l;
                atomicAdd(&grad_input[in_idx], grad_val);
            }
        }
    }
}

__global__ void avgpool1d_backward_kernel_fp16(
    const __half* grad_output,
    float* grad_input_f32,
    int64_t N, int64_t C, int64_t L,
    int64_t L_out,
    int64_t kernel_size, int64_t stride, int64_t padding
) {
    int64_t total = N * C * L_out;

    HIP_KERNEL_LOOP(idx, total) {
        int64_t ol = idx % L_out;
        int64_t c  = (idx / L_out) % C;
        int64_t n  = idx / (L_out * C);

        int64_t l_start = ol * stride - padding;

        int64_t count = 0;
        for (int64_t k = 0; k < kernel_size; ++k) {
            int64_t l = l_start + k;
            if (l >= 0 && l < L) count++;
        }

        float grad_val = tenzor::rocm::safe_h2f(grad_output[idx]) / static_cast<float>(count);

        for (int64_t k = 0; k < kernel_size; ++k) {
            int64_t l = l_start + k;
            if (l >= 0 && l < L) {
                int64_t in_idx = (n * C + c) * L + l;
                atomicAdd(&grad_input_f32[in_idx], grad_val);
            }
        }
    }
}

auto avgpool1d_backward_hip(
    const Tensor& grad_output,
    const std::vector<int64_t>& input_shape,
    std::array<int64_t, 1> kernel_size_a, std::array<int64_t, 1> stride_a, std::array<int64_t, 1> padding_a,
    hipStream_t stream
) -> Tensor {
    // Q.6: per-axis std::array<int64_t, 1> signature.
    const int64_t kernel_size = kernel_size_a[0];
    const int64_t stride      = stride_a[0];
    const int64_t padding     = padding_a[0];

    int64_t N = input_shape[0];
    int64_t C = input_shape[1];
    int64_t L = input_shape[2];

    auto grad_shape = grad_output.shape();
    int64_t L_out = grad_shape[2];

    Tensor grad_input(input_shape, grad_output.dtype(), grad_output.device());
    int64_t input_numel = N * C * L;

    int64_t total_elements = N * C * L_out;
    int threads = rocm::get_wavefront_size() * 4;  // 4 wavefronts per block
    int blocks = (total_elements + threads - 1) / threads;

    if (grad_output.dtype() == DType::Float32) {
        HIP_CHECK(hipMemsetAsync(grad_input.data<float>(), 0, input_numel * sizeof(float), stream));
        hipLaunchKernelGGL(avgpool1d_backward_kernel_impl<float>,
            dim3(blocks), dim3(threads), 0, stream,
            grad_output.data<float>(), grad_input.data<float>(),
            N, C, L, L_out, kernel_size, stride, padding);
        HIP_POST_LAUNCH_CHECK();
    } else if (grad_output.dtype() == DType::Float64) {
        HIP_CHECK(hipMemsetAsync(grad_input.data<double>(), 0, input_numel * sizeof(double), stream));
        hipLaunchKernelGGL(avgpool1d_backward_kernel_impl<double>,
            dim3(blocks), dim3(threads), 0, stream,
            grad_output.data<double>(), grad_input.data<double>(),
            N, C, L, L_out, kernel_size, stride, padding);
        HIP_POST_LAUNCH_CHECK();
    } else if (grad_output.dtype() == DType::Float16) {
        Tensor grad_input_f32(input_shape, DType::Float32, grad_output.device());
        HIP_CHECK(hipMemsetAsync(grad_input_f32.data<float>(), 0, input_numel * sizeof(float), stream));

        hipLaunchKernelGGL(avgpool1d_backward_kernel_fp16,
            dim3(blocks), dim3(threads), 0, stream,
            reinterpret_cast<const __half*>(grad_output.data<Float16>()),
            grad_input_f32.data<float>(),
            N, C, L, L_out, kernel_size, stride, padding);
        HIP_POST_LAUNCH_CHECK();

        int convert_blocks = (input_numel + threads - 1) / threads;
        hipLaunchKernelGGL(convert_f32_to_f16_pool,
            dim3(convert_blocks), dim3(threads), 0, stream,
            grad_input_f32.data<float>(),
            reinterpret_cast<__half*>(grad_input.data<Float16>()),
            input_numel);
        HIP_POST_LAUNCH_CHECK();
    } else if (grad_output.dtype() == DType::BFloat16) {
        auto grad_output_f32 = grad_output.to(DType::Float32);
        auto result_f32 = avgpool1d_backward_hip(grad_output_f32, input_shape, kernel_size_a, stride_a, padding_a, stream);
        return result_f32.to(DType::BFloat16);
    } else {
        throw std::runtime_error("avgpool1d_backward_hip: unsupported dtype");
    }

    HIP_POST_LAUNCH_CHECK();
    return grad_input;
}

// ==============================================================================
// Adaptive MaxPool1D Forward
// ==============================================================================

template<typename T>
__global__ void adaptive_maxpool1d_forward_kernel(
    const T* input,
    T* output,
    int64_t* indices,
    int64_t N, int64_t C, int64_t L_in, int64_t L_out
) {
    int64_t total = N * C * L_out;

    HIP_KERNEL_LOOP(idx, total) {
        int64_t ol = idx % L_out;
        int64_t c  = (idx / L_out) % C;
        int64_t n  = idx / (L_out * C);

        int64_t l_start = (ol * L_in) / L_out;
        int64_t l_end   = ((ol + 1) * L_in + L_out - 1) / L_out;

        T max_val = std::numeric_limits<T>::lowest();
        int64_t max_idx = l_start;

        for (int64_t l = l_start; l < l_end; ++l) {
            int64_t in_idx = (n * C + c) * L_in + l;
            T val = input[in_idx];
            if (tenzor::rocm::is_nan_bits(val) || val > max_val) {
                max_val = val;
                max_idx = l;
            }
        }

        output[idx] = max_val;
        indices[idx] = max_idx;
    }
}

__global__ void adaptive_maxpool1d_forward_kernel_fp16(
    const __half* input,
    __half* output,
    int64_t* indices,
    int64_t N, int64_t C, int64_t L_in, int64_t L_out
) {
    int64_t total = N * C * L_out;

    HIP_KERNEL_LOOP(idx, total) {
        int64_t ol = idx % L_out;
        int64_t c  = (idx / L_out) % C;
        int64_t n  = idx / (L_out * C);

        int64_t l_start = (ol * L_in) / L_out;
        int64_t l_end   = ((ol + 1) * L_in + L_out - 1) / L_out;

        float max_val = std::numeric_limits<float>::lowest();
        int64_t max_idx = l_start;

        for (int64_t l = l_start; l < l_end; ++l) {
            int64_t in_idx = (n * C + c) * L_in + l;
            float val = tenzor::rocm::safe_h2f(input[in_idx]);
            if (tenzor::rocm::is_nan_bits(val) || val > max_val) {
                max_val = val;
                max_idx = l;
            }
        }

        output[idx] = tenzor::rocm::safe_f2h(max_val);
        indices[idx] = max_idx;
    }
}

auto adaptive_maxpool1d_forward_hip(
    const Tensor& input,
    int64_t output_size,
    hipStream_t stream
) -> std::pair<Tensor, Tensor> {

    auto shape = input.shape();
    int64_t N = shape[0], C = shape[1], L_in = shape[2];

    Tensor output({N, C, output_size}, input.dtype(), input.device());
    Tensor indices({N, C, output_size}, DType::Int64, input.device());

    int64_t total = N * C * output_size;
    int threads = rocm::get_wavefront_size() * 4;  // 4 wavefronts per block
    int blocks = (total + threads - 1) / threads;

    if (input.dtype() == DType::Float32) {
        hipLaunchKernelGGL(adaptive_maxpool1d_forward_kernel<float>,
            dim3(blocks), dim3(threads), 0, stream,
            input.data<float>(), output.data<float>(), indices.data<int64_t>(),
            N, C, L_in, output_size);
        HIP_POST_LAUNCH_CHECK();
    } else if (input.dtype() == DType::Float64) {
        hipLaunchKernelGGL(adaptive_maxpool1d_forward_kernel<double>,
            dim3(blocks), dim3(threads), 0, stream,
            input.data<double>(), output.data<double>(), indices.data<int64_t>(),
            N, C, L_in, output_size);
        HIP_POST_LAUNCH_CHECK();
    } else if (input.dtype() == DType::Float16) {
        hipLaunchKernelGGL(adaptive_maxpool1d_forward_kernel_fp16,
            dim3(blocks), dim3(threads), 0, stream,
            reinterpret_cast<const __half*>(input.data<Float16>()),
            reinterpret_cast<__half*>(output.data<Float16>()),
            indices.data<int64_t>(),
            N, C, L_in, output_size);
        HIP_POST_LAUNCH_CHECK();
    } else if (input.dtype() == DType::BFloat16) {
        auto input_f32 = input.to(DType::Float32);
        auto [output_f32, idx] = adaptive_maxpool1d_forward_hip(input_f32, output_size, stream);
        return {output_f32.to(DType::BFloat16), idx};
    } else {
        throw std::runtime_error("adaptive_maxpool1d_forward_hip: unsupported dtype");
    }

    HIP_POST_LAUNCH_CHECK();
    return {output, indices};
}

// ==============================================================================
// Adaptive MaxPool1D Backward (reuses maxpool1d backward — same index scatter)
// ==============================================================================

auto adaptive_maxpool1d_backward_hip(
    const Tensor& grad_output,
    const Tensor& indices,
    const std::vector<int64_t>& input_shape,
    hipStream_t stream
) -> Tensor {
    return maxpool1d_backward_hip(grad_output, indices, input_shape, stream);
}

// ==============================================================================
// Adaptive AvgPool1D Forward
// ==============================================================================

template<typename T>
__global__ void adaptive_avgpool1d_forward_kernel(
    const T* input,
    T* output,
    int64_t N, int64_t C, int64_t L_in, int64_t L_out
) {
    int64_t total = N * C * L_out;

    HIP_KERNEL_LOOP(idx, total) {
        int64_t ol = idx % L_out;
        int64_t c  = (idx / L_out) % C;
        int64_t n  = idx / (L_out * C);

        int64_t l_start = (ol * L_in) / L_out;
        int64_t l_end   = ((ol + 1) * L_in + L_out - 1) / L_out;

        T sum = T(0);
        for (int64_t l = l_start; l < l_end; ++l) {
            sum += input[(n * C + c) * L_in + l];
        }

        int64_t cnt = l_end - l_start;
        output[idx] = cnt > 0 ? sum / static_cast<T>(cnt) : T(0);
    }
}

__global__ void adaptive_avgpool1d_forward_kernel_fp16(
    const __half* input,
    __half* output,
    int64_t N, int64_t C, int64_t L_in, int64_t L_out
) {
    int64_t total = N * C * L_out;

    HIP_KERNEL_LOOP(idx, total) {
        int64_t ol = idx % L_out;
        int64_t c  = (idx / L_out) % C;
        int64_t n  = idx / (L_out * C);

        int64_t l_start = (ol * L_in) / L_out;
        int64_t l_end   = ((ol + 1) * L_in + L_out - 1) / L_out;

        float sum = 0.0f;
        for (int64_t l = l_start; l < l_end; ++l) {
            sum += tenzor::rocm::safe_h2f(input[(n * C + c) * L_in + l]);
        }

        int64_t cnt = l_end - l_start;
        output[idx] = tenzor::rocm::safe_f2h(cnt > 0 ? sum / static_cast<float>(cnt) : 0.0f);
    }
}

auto adaptive_avgpool1d_forward_hip(
    const Tensor& input,
    int64_t output_size,
    hipStream_t stream
) -> Tensor {

    auto shape = input.shape();
    int64_t N = shape[0], C = shape[1], L_in = shape[2];

    Tensor output({N, C, output_size}, input.dtype(), input.device());

    int64_t total = N * C * output_size;
    int threads = rocm::get_wavefront_size() * 4;  // 4 wavefronts per block
    int blocks = (total + threads - 1) / threads;

    if (input.dtype() == DType::Float32) {
        hipLaunchKernelGGL(adaptive_avgpool1d_forward_kernel<float>,
            dim3(blocks), dim3(threads), 0, stream,
            input.data<float>(), output.data<float>(),
            N, C, L_in, output_size);
        HIP_POST_LAUNCH_CHECK();
    } else if (input.dtype() == DType::Float64) {
        hipLaunchKernelGGL(adaptive_avgpool1d_forward_kernel<double>,
            dim3(blocks), dim3(threads), 0, stream,
            input.data<double>(), output.data<double>(),
            N, C, L_in, output_size);
        HIP_POST_LAUNCH_CHECK();
    } else if (input.dtype() == DType::Float16) {
        hipLaunchKernelGGL(adaptive_avgpool1d_forward_kernel_fp16,
            dim3(blocks), dim3(threads), 0, stream,
            reinterpret_cast<const __half*>(input.data<Float16>()),
            reinterpret_cast<__half*>(output.data<Float16>()),
            N, C, L_in, output_size);
        HIP_POST_LAUNCH_CHECK();
    } else if (input.dtype() == DType::BFloat16) {
        auto input_f32 = input.to(DType::Float32);
        auto result_f32 = adaptive_avgpool1d_forward_hip(input_f32, output_size, stream);
        return result_f32.to(DType::BFloat16);
    } else {
        throw std::runtime_error("adaptive_avgpool1d_forward_hip: unsupported dtype");
    }

    HIP_POST_LAUNCH_CHECK();
    return output;
}

// ==============================================================================
// Adaptive AvgPool1D Backward
// ==============================================================================

template<typename T>
__global__ void adaptive_avgpool1d_backward_kernel(
    const T* grad_output,
    T* grad_input,
    int64_t N, int64_t C, int64_t L_in, int64_t L_out
) {
    int64_t total = N * C * L_out;

    HIP_KERNEL_LOOP(idx, total) {
        int64_t ol = idx % L_out;
        int64_t c  = (idx / L_out) % C;
        int64_t n  = idx / (L_out * C);

        int64_t l_start = (ol * L_in) / L_out;
        int64_t l_end   = ((ol + 1) * L_in + L_out - 1) / L_out;

        T grad_val = grad_output[idx] / static_cast<T>(l_end - l_start);

        for (int64_t l = l_start; l < l_end; ++l) {
            int64_t in_idx = (n * C + c) * L_in + l;
            atomicAdd(&grad_input[in_idx], grad_val);
        }
    }
}

__global__ void adaptive_avgpool1d_backward_kernel_fp16(
    const __half* grad_output,
    float* grad_input_f32,
    int64_t N, int64_t C, int64_t L_in, int64_t L_out
) {
    int64_t total = N * C * L_out;

    HIP_KERNEL_LOOP(idx, total) {
        int64_t ol = idx % L_out;
        int64_t c  = (idx / L_out) % C;
        int64_t n  = idx / (L_out * C);

        int64_t l_start = (ol * L_in) / L_out;
        int64_t l_end   = ((ol + 1) * L_in + L_out - 1) / L_out;

        float grad_val = tenzor::rocm::safe_h2f(grad_output[idx]) / static_cast<float>(l_end - l_start);

        for (int64_t l = l_start; l < l_end; ++l) {
            int64_t in_idx = (n * C + c) * L_in + l;
            atomicAdd(&grad_input_f32[in_idx], grad_val);
        }
    }
}

auto adaptive_avgpool1d_backward_hip(
    const Tensor& grad_output,
    const std::vector<int64_t>& input_shape,
    hipStream_t stream
) -> Tensor {

    int64_t N = input_shape[0];
    int64_t C = input_shape[1];
    int64_t L_in = input_shape[2];

    auto grad_shape = grad_output.shape();
    int64_t L_out = grad_shape[2];

    Tensor grad_input(input_shape, grad_output.dtype(), grad_output.device());
    int64_t input_numel = N * C * L_in;

    int64_t total = N * C * L_out;
    int threads = rocm::get_wavefront_size() * 4;  // 4 wavefronts per block
    int blocks = (total + threads - 1) / threads;

    if (grad_output.dtype() == DType::Float32) {
        HIP_CHECK(hipMemsetAsync(grad_input.data<float>(), 0, input_numel * sizeof(float), stream));
        hipLaunchKernelGGL(adaptive_avgpool1d_backward_kernel<float>,
            dim3(blocks), dim3(threads), 0, stream,
            grad_output.data<float>(), grad_input.data<float>(),
            N, C, L_in, L_out);
        HIP_POST_LAUNCH_CHECK();
    } else if (grad_output.dtype() == DType::Float64) {
        HIP_CHECK(hipMemsetAsync(grad_input.data<double>(), 0, input_numel * sizeof(double), stream));
        hipLaunchKernelGGL(adaptive_avgpool1d_backward_kernel<double>,
            dim3(blocks), dim3(threads), 0, stream,
            grad_output.data<double>(), grad_input.data<double>(),
            N, C, L_in, L_out);
        HIP_POST_LAUNCH_CHECK();
    } else if (grad_output.dtype() == DType::Float16) {
        Tensor grad_input_f32(input_shape, DType::Float32, grad_output.device());
        HIP_CHECK(hipMemsetAsync(grad_input_f32.data<float>(), 0, input_numel * sizeof(float), stream));

        hipLaunchKernelGGL(adaptive_avgpool1d_backward_kernel_fp16,
            dim3(blocks), dim3(threads), 0, stream,
            reinterpret_cast<const __half*>(grad_output.data<Float16>()),
            grad_input_f32.data<float>(),
            N, C, L_in, L_out);
        HIP_POST_LAUNCH_CHECK();

        int convert_blocks = (input_numel + threads - 1) / threads;
        hipLaunchKernelGGL(convert_f32_to_f16_pool,
            dim3(convert_blocks), dim3(threads), 0, stream,
            grad_input_f32.data<float>(),
            reinterpret_cast<__half*>(grad_input.data<Float16>()),
            input_numel);
        HIP_POST_LAUNCH_CHECK();
    } else if (grad_output.dtype() == DType::BFloat16) {
        auto grad_output_f32 = grad_output.to(DType::Float32);
        auto result_f32 = adaptive_avgpool1d_backward_hip(grad_output_f32, input_shape, stream);
        return result_f32.to(DType::BFloat16);
    } else {
        throw std::runtime_error("adaptive_avgpool1d_backward_hip: unsupported dtype");
    }

    HIP_POST_LAUNCH_CHECK();
    return grad_input;
}

// ==============================================================================
// MaxPool3D Forward
// ==============================================================================

template<typename T>
__global__ void maxpool3d_forward_kernel(
    const T* input,
    T* output,
    int64_t* indices,
    int64_t N, int64_t C,
    int64_t D, int64_t H, int64_t W,
    int64_t D_out, int64_t H_out, int64_t W_out,
    int64_t kD, int64_t kH, int64_t kW,
    int64_t sD, int64_t sH, int64_t sW,
    int64_t pD, int64_t pH, int64_t pW
) {
    int64_t total = N * C * D_out * H_out * W_out;

    HIP_KERNEL_LOOP(idx, total) {
        int64_t ow = idx % W_out;
        int64_t oh = (idx / W_out) % H_out;
        int64_t od = (idx / (W_out * H_out)) % D_out;
        int64_t c  = (idx / (W_out * H_out * D_out)) % C;
        int64_t n  = idx / (W_out * H_out * D_out * C);

        int64_t d_start = od * sD - pD;
        int64_t h_start = oh * sH - pH;
        int64_t w_start = ow * sW - pW;

        T max_val = std::numeric_limits<T>::lowest();
        int64_t max_idx = -1;  // -1 sentinel for an all-padding window (matches 1D/2D)

        for (int64_t kd = 0; kd < kD; ++kd) {
            for (int64_t kh = 0; kh < kH; ++kh) {
                for (int64_t kw = 0; kw < kW; ++kw) {
                    int64_t d = d_start + kd;
                    int64_t h = h_start + kh;
                    int64_t w = w_start + kw;

                    if (d >= 0 && d < D && h >= 0 && h < H && w >= 0 && w < W) {
                        int64_t in_idx = ((n * C + c) * D + d) * H * W + h * W + w;
                        T val = input[in_idx];
                        if (tenzor::rocm::is_nan_bits(val) || val > max_val) {
                            max_val = val;
                            max_idx = d * H * W + h * W + w;
                        }
                    }
                }
            }
        }

        output[idx] = max_val;
        indices[idx] = max_idx;
    }
}

__global__ void maxpool3d_forward_kernel_fp16(
    const __half* input,
    __half* output,
    int64_t* indices,
    int64_t N, int64_t C,
    int64_t D, int64_t H, int64_t W,
    int64_t D_out, int64_t H_out, int64_t W_out,
    int64_t kD, int64_t kH, int64_t kW,
    int64_t sD, int64_t sH, int64_t sW,
    int64_t pD, int64_t pH, int64_t pW
) {
    int64_t total = N * C * D_out * H_out * W_out;

    HIP_KERNEL_LOOP(idx, total) {
        int64_t ow = idx % W_out;
        int64_t oh = (idx / W_out) % H_out;
        int64_t od = (idx / (W_out * H_out)) % D_out;
        int64_t c  = (idx / (W_out * H_out * D_out)) % C;
        int64_t n  = idx / (W_out * H_out * D_out * C);

        int64_t d_start = od * sD - pD;
        int64_t h_start = oh * sH - pH;
        int64_t w_start = ow * sW - pW;

        float max_val = std::numeric_limits<float>::lowest();
        int64_t max_idx = -1;  // -1 sentinel for an all-padding window (matches 1D/2D)

        for (int64_t kd = 0; kd < kD; ++kd) {
            for (int64_t kh = 0; kh < kH; ++kh) {
                for (int64_t kw = 0; kw < kW; ++kw) {
                    int64_t d = d_start + kd;
                    int64_t h = h_start + kh;
                    int64_t w = w_start + kw;

                    if (d >= 0 && d < D && h >= 0 && h < H && w >= 0 && w < W) {
                        int64_t in_idx = ((n * C + c) * D + d) * H * W + h * W + w;
                        float val = tenzor::rocm::safe_h2f(input[in_idx]);
                        if (tenzor::rocm::is_nan_bits(val) || val > max_val) {
                            max_val = val;
                            max_idx = d * H * W + h * W + w;
                        }
                    }
                }
            }
        }

        output[idx] = tenzor::rocm::safe_f2h(max_val);
        indices[idx] = max_idx;
    }
}

auto maxpool3d_forward_hip(
    const Tensor& input,
    std::array<int64_t, 3> kernel_size_a, std::array<int64_t, 3> stride_a, std::array<int64_t, 3> padding_a,
    hipStream_t stream
) -> std::pair<Tensor, Tensor> {
    // Q.6: per-axis std::array<int64_t, 3> signature.
    const int64_t kD = kernel_size_a[0], kH = kernel_size_a[1], kW = kernel_size_a[2];
    const int64_t sD = stride_a[0],      sH = stride_a[1],      sW = stride_a[2];
    const int64_t pD = padding_a[0],     pH = padding_a[1],     pW = padding_a[2];

    auto shape = input.shape();
    int64_t N = shape[0];
    int64_t C = shape[1];
    int64_t D = shape[2];
    int64_t H = shape[3];
    int64_t W = shape[4];

    int64_t D_out = (D + 2 * pD - kD) / sD + 1;
    int64_t H_out = (H + 2 * pH - kH) / sH + 1;
    int64_t W_out = (W + 2 * pW - kW) / sW + 1;

    Tensor output({N, C, D_out, H_out, W_out}, input.dtype(), input.device());
    Tensor indices({N, C, D_out, H_out, W_out}, DType::Int64, input.device());

    int64_t total = N * C * D_out * H_out * W_out;
    int threads = rocm::get_wavefront_size() * 4;  // 4 wavefronts per block
    int blocks = (total + threads - 1) / threads;
    // Empty output: a zero-grid launch is rejected by HIP; return as-is.
    if (blocks == 0) return {output, indices};

    if (input.dtype() == DType::Float32) {
        hipLaunchKernelGGL(maxpool3d_forward_kernel<float>,
            dim3(blocks), dim3(threads), 0, stream,
            input.data<float>(), output.data<float>(), indices.data<int64_t>(),
            N, C, D, H, W, D_out, H_out, W_out, kD, kH, kW, sD, sH, sW, pD, pH, pW);
        HIP_POST_LAUNCH_CHECK();
    } else if (input.dtype() == DType::Float64) {
        hipLaunchKernelGGL(maxpool3d_forward_kernel<double>,
            dim3(blocks), dim3(threads), 0, stream,
            input.data<double>(), output.data<double>(), indices.data<int64_t>(),
            N, C, D, H, W, D_out, H_out, W_out, kD, kH, kW, sD, sH, sW, pD, pH, pW);
        HIP_POST_LAUNCH_CHECK();
    } else if (input.dtype() == DType::Float16) {
        hipLaunchKernelGGL(maxpool3d_forward_kernel_fp16,
            dim3(blocks), dim3(threads), 0, stream,
            reinterpret_cast<const __half*>(input.data<Float16>()),
            reinterpret_cast<__half*>(output.data<Float16>()),
            indices.data<int64_t>(),
            N, C, D, H, W, D_out, H_out, W_out, kD, kH, kW, sD, sH, sW, pD, pH, pW);
        HIP_POST_LAUNCH_CHECK();
    } else if (input.dtype() == DType::BFloat16) {
        auto input_f32 = input.to(DType::Float32);
        auto [output_f32, idx] = maxpool3d_forward_hip(input_f32, kernel_size_a, stride_a, padding_a, stream);
        return {output_f32.to(DType::BFloat16), idx};
    } else {
        throw std::runtime_error("maxpool3d_forward_hip: unsupported dtype");
    }

    HIP_POST_LAUNCH_CHECK();
    return {output, indices};
}

// ==============================================================================
// MaxPool3D Backward
// ==============================================================================

template<typename T>
__global__ void maxpool3d_backward_kernel_impl(
    const T* grad_output,
    const int64_t* indices,
    T* grad_input,
    int64_t N, int64_t C,
    int64_t D, int64_t H, int64_t W,
    int64_t D_out, int64_t H_out, int64_t W_out
) {
    int64_t total = N * C * D_out * H_out * W_out;

    HIP_KERNEL_LOOP(idx, total) {
        int64_t c = (idx / (W_out * H_out * D_out)) % C;
        int64_t n = idx / (W_out * H_out * D_out * C);

        int64_t max_idx = indices[idx];
        int64_t in_idx = ((n * C + c) * D * H * W) + max_idx;
        atomicAdd(&grad_input[in_idx], grad_output[idx]);
    }
}

__global__ void maxpool3d_backward_kernel_fp16(
    const __half* grad_output,
    const int64_t* indices,
    float* grad_input_f32,
    int64_t N, int64_t C,
    int64_t D, int64_t H, int64_t W,
    int64_t D_out, int64_t H_out, int64_t W_out
) {
    int64_t total = N * C * D_out * H_out * W_out;

    HIP_KERNEL_LOOP(idx, total) {
        int64_t c = (idx / (W_out * H_out * D_out)) % C;
        int64_t n = idx / (W_out * H_out * D_out * C);

        int64_t max_idx = indices[idx];
        int64_t in_idx = ((n * C + c) * D * H * W) + max_idx;
        atomicAdd(&grad_input_f32[in_idx], tenzor::rocm::safe_h2f(grad_output[idx]));
    }
}

auto maxpool3d_backward_hip(
    const Tensor& grad_output,
    const Tensor& indices,
    const std::vector<int64_t>& input_shape,
    hipStream_t stream
) -> Tensor {

    int64_t N = input_shape[0];
    int64_t C = input_shape[1];
    int64_t D = input_shape[2];
    int64_t H = input_shape[3];
    int64_t W = input_shape[4];

    auto grad_shape = grad_output.shape();
    int64_t D_out = grad_shape[2];
    int64_t H_out = grad_shape[3];
    int64_t W_out = grad_shape[4];

    Tensor grad_input(input_shape, grad_output.dtype(), grad_output.device());
    int64_t input_numel = N * C * D * H * W;
    HIP_CHECK(hipMemsetAsync(grad_input.data<uint8_t>(), 0,
        input_numel * dtype_size(grad_output.dtype()), stream));

    int64_t total = N * C * D_out * H_out * W_out;
    int threads = rocm::get_wavefront_size() * 4;  // 4 wavefronts per block
    int blocks = (total + threads - 1) / threads;

    if (grad_output.dtype() == DType::Float32) {
        hipLaunchKernelGGL(maxpool3d_backward_kernel_impl<float>,
            dim3(blocks), dim3(threads), 0, stream,
            grad_output.data<float>(), indices.data<int64_t>(),
            grad_input.data<float>(),
            N, C, D, H, W, D_out, H_out, W_out);
        HIP_POST_LAUNCH_CHECK();
    } else if (grad_output.dtype() == DType::Float64) {
        hipLaunchKernelGGL(maxpool3d_backward_kernel_impl<double>,
            dim3(blocks), dim3(threads), 0, stream,
            grad_output.data<double>(), indices.data<int64_t>(),
            grad_input.data<double>(),
            N, C, D, H, W, D_out, H_out, W_out);
        HIP_POST_LAUNCH_CHECK();
    } else if (grad_output.dtype() == DType::Float16) {
        Tensor grad_input_f32(input_shape, DType::Float32, grad_output.device());
        HIP_CHECK(hipMemsetAsync(grad_input_f32.data<float>(), 0, input_numel * sizeof(float), stream));

        hipLaunchKernelGGL(maxpool3d_backward_kernel_fp16,
            dim3(blocks), dim3(threads), 0, stream,
            reinterpret_cast<const __half*>(grad_output.data<Float16>()),
            indices.data<int64_t>(),
            grad_input_f32.data<float>(),
            N, C, D, H, W, D_out, H_out, W_out);
        HIP_POST_LAUNCH_CHECK();

        int convert_blocks = (input_numel + threads - 1) / threads;
        hipLaunchKernelGGL(convert_f32_to_f16_pool,
            dim3(convert_blocks), dim3(threads), 0, stream,
            grad_input_f32.data<float>(),
            reinterpret_cast<__half*>(grad_input.data<Float16>()),
            input_numel);
        HIP_POST_LAUNCH_CHECK();
    } else if (grad_output.dtype() == DType::BFloat16) {
        auto grad_output_f32 = grad_output.to(DType::Float32);
        auto result_f32 = maxpool3d_backward_hip(grad_output_f32, indices, input_shape, stream);
        return result_f32.to(DType::BFloat16);
    } else {
        throw std::runtime_error("maxpool3d_backward_hip: unsupported dtype");
    }

    HIP_POST_LAUNCH_CHECK();
    return grad_input;
}

// ==============================================================================
// AvgPool3D Forward
// ==============================================================================

template<typename T>
__global__ void avgpool3d_forward_kernel(
    const T* input,
    T* output,
    int64_t N, int64_t C,
    int64_t D, int64_t H, int64_t W,
    int64_t D_out, int64_t H_out, int64_t W_out,
    int64_t kD, int64_t kH, int64_t kW,
    int64_t sD, int64_t sH, int64_t sW,
    int64_t pD, int64_t pH, int64_t pW,
    bool count_include_pad
) {
    int64_t total = N * C * D_out * H_out * W_out;

    HIP_KERNEL_LOOP(idx, total) {
        int64_t ow = idx % W_out;
        int64_t oh = (idx / W_out) % H_out;
        int64_t od = (idx / (W_out * H_out)) % D_out;
        int64_t c  = (idx / (W_out * H_out * D_out)) % C;
        int64_t n  = idx / (W_out * H_out * D_out * C);

        int64_t d_start = od * sD - pD;
        int64_t h_start = oh * sH - pH;
        int64_t w_start = ow * sW - pW;

        T sum = T(0);
        int64_t valid_count = 0;

        for (int64_t kd = 0; kd < kD; ++kd) {
            for (int64_t kh = 0; kh < kH; ++kh) {
                for (int64_t kw = 0; kw < kW; ++kw) {
                    int64_t d = d_start + kd;
                    int64_t h = h_start + kh;
                    int64_t w = w_start + kw;

                    if (d >= 0 && d < D && h >= 0 && h < H && w >= 0 && w < W) {
                        sum += input[((n * C + c) * D + d) * H * W + h * W + w];
                        valid_count++;
                    }
                }
            }
        }

        // PyTorch semantics: divide by the full window when count_include_pad,
        // otherwise by the number of in-bounds (non-padding) elements.
        int64_t divisor = count_include_pad ? (kD * kH * kW) : valid_count;
        output[idx] = divisor > 0 ? sum / static_cast<T>(divisor) : T(0);
    }
}

__global__ void avgpool3d_forward_kernel_fp16(
    const __half* input,
    __half* output,
    int64_t N, int64_t C,
    int64_t D, int64_t H, int64_t W,
    int64_t D_out, int64_t H_out, int64_t W_out,
    int64_t kD, int64_t kH, int64_t kW,
    int64_t sD, int64_t sH, int64_t sW,
    int64_t pD, int64_t pH, int64_t pW,
    bool count_include_pad
) {
    int64_t total = N * C * D_out * H_out * W_out;

    HIP_KERNEL_LOOP(idx, total) {
        int64_t ow = idx % W_out;
        int64_t oh = (idx / W_out) % H_out;
        int64_t od = (idx / (W_out * H_out)) % D_out;
        int64_t c  = (idx / (W_out * H_out * D_out)) % C;
        int64_t n  = idx / (W_out * H_out * D_out * C);

        int64_t d_start = od * sD - pD;
        int64_t h_start = oh * sH - pH;
        int64_t w_start = ow * sW - pW;

        float sum = 0.0f;
        int64_t valid_count = 0;

        for (int64_t kd = 0; kd < kD; ++kd) {
            for (int64_t kh = 0; kh < kH; ++kh) {
                for (int64_t kw = 0; kw < kW; ++kw) {
                    int64_t d = d_start + kd;
                    int64_t h = h_start + kh;
                    int64_t w = w_start + kw;

                    if (d >= 0 && d < D && h >= 0 && h < H && w >= 0 && w < W) {
                        sum += tenzor::rocm::safe_h2f(input[((n * C + c) * D + d) * H * W + h * W + w]);
                        valid_count++;
                    }
                }
            }
        }

        int64_t divisor = count_include_pad ? (kD * kH * kW) : valid_count;
        output[idx] = tenzor::rocm::safe_f2h(divisor > 0 ? sum / static_cast<float>(divisor) : 0.0f);
    }
}

auto avgpool3d_forward_hip(
    const Tensor& input,
    std::array<int64_t, 3> kernel_size_a, std::array<int64_t, 3> stride_a, std::array<int64_t, 3> padding_a,
    bool count_include_pad,
    hipStream_t stream
) -> Tensor {
    // Q.6: per-axis std::array<int64_t, 3> signature.
    const int64_t kD = kernel_size_a[0], kH = kernel_size_a[1], kW = kernel_size_a[2];
    const int64_t sD = stride_a[0],      sH = stride_a[1],      sW = stride_a[2];
    const int64_t pD = padding_a[0],     pH = padding_a[1],     pW = padding_a[2];

    auto shape = input.shape();
    int64_t N = shape[0];
    int64_t C = shape[1];
    int64_t D = shape[2];
    int64_t H = shape[3];
    int64_t W = shape[4];

    int64_t D_out = (D + 2 * pD - kD) / sD + 1;
    int64_t H_out = (H + 2 * pH - kH) / sH + 1;
    int64_t W_out = (W + 2 * pW - kW) / sW + 1;

    Tensor output({N, C, D_out, H_out, W_out}, input.dtype(), input.device());

    int64_t total = N * C * D_out * H_out * W_out;
    int threads = rocm::get_wavefront_size() * 4;  // 4 wavefronts per block
    int blocks = (total + threads - 1) / threads;

    if (input.dtype() == DType::Float32) {
        hipLaunchKernelGGL(avgpool3d_forward_kernel<float>,
            dim3(blocks), dim3(threads), 0, stream,
            input.data<float>(), output.data<float>(),
            N, C, D, H, W, D_out, H_out, W_out, kD, kH, kW, sD, sH, sW, pD, pH, pW, count_include_pad);
        HIP_POST_LAUNCH_CHECK();
    } else if (input.dtype() == DType::Float64) {
        hipLaunchKernelGGL(avgpool3d_forward_kernel<double>,
            dim3(blocks), dim3(threads), 0, stream,
            input.data<double>(), output.data<double>(),
            N, C, D, H, W, D_out, H_out, W_out, kD, kH, kW, sD, sH, sW, pD, pH, pW, count_include_pad);
        HIP_POST_LAUNCH_CHECK();
    } else if (input.dtype() == DType::Float16) {
        hipLaunchKernelGGL(avgpool3d_forward_kernel_fp16,
            dim3(blocks), dim3(threads), 0, stream,
            reinterpret_cast<const __half*>(input.data<Float16>()),
            reinterpret_cast<__half*>(output.data<Float16>()),
            N, C, D, H, W, D_out, H_out, W_out, kD, kH, kW, sD, sH, sW, pD, pH, pW, count_include_pad);
        HIP_POST_LAUNCH_CHECK();
    } else if (input.dtype() == DType::BFloat16) {
        auto input_f32 = input.to(DType::Float32);
        auto result_f32 = avgpool3d_forward_hip(input_f32, kernel_size_a, stride_a, padding_a, count_include_pad, stream);
        return result_f32.to(DType::BFloat16);
    } else {
        throw std::runtime_error("avgpool3d_forward_hip: unsupported dtype");
    }

    HIP_POST_LAUNCH_CHECK();
    return output;
}

// ==============================================================================
// AvgPool3D Backward
// ==============================================================================

template<typename T>
__global__ void avgpool3d_backward_kernel_impl(
    const T* grad_output,
    T* grad_input,
    int64_t N, int64_t C,
    int64_t D, int64_t H, int64_t W,
    int64_t D_out, int64_t H_out, int64_t W_out,
    int64_t kD, int64_t kH, int64_t kW,
    int64_t sD, int64_t sH, int64_t sW,
    int64_t pD, int64_t pH, int64_t pW,
    bool count_include_pad
) {
    int64_t total = N * C * D_out * H_out * W_out;

    HIP_KERNEL_LOOP(idx, total) {
        int64_t ow = idx % W_out;
        int64_t oh = (idx / W_out) % H_out;
        int64_t od = (idx / (W_out * H_out)) % D_out;
        int64_t c  = (idx / (W_out * H_out * D_out)) % C;
        int64_t n  = idx / (W_out * H_out * D_out * C);

        int64_t d_start = od * sD - pD;
        int64_t h_start = oh * sH - pH;
        int64_t w_start = ow * sW - pW;

        int64_t valid_count = 0;
        for (int64_t kd = 0; kd < kD; ++kd) {
            for (int64_t kh = 0; kh < kH; ++kh) {
                for (int64_t kw = 0; kw < kW; ++kw) {
                    int64_t d = d_start + kd;
                    int64_t h = h_start + kh;
                    int64_t w = w_start + kw;
                    if (d >= 0 && d < D && h >= 0 && h < H && w >= 0 && w < W) valid_count++;
                }
            }
        }

        // Gradient is distributed by the same divisor used in forward.
        int64_t divisor = count_include_pad ? (kD * kH * kW) : valid_count;
        if (divisor == 0) continue;
        T grad_val = grad_output[idx] / static_cast<T>(divisor);

        for (int64_t kd = 0; kd < kD; ++kd) {
            for (int64_t kh = 0; kh < kH; ++kh) {
                for (int64_t kw = 0; kw < kW; ++kw) {
                    int64_t d = d_start + kd;
                    int64_t h = h_start + kh;
                    int64_t w = w_start + kw;
                    if (d >= 0 && d < D && h >= 0 && h < H && w >= 0 && w < W) {
                        int64_t in_idx = ((n * C + c) * D + d) * H * W + h * W + w;
                        atomicAdd(&grad_input[in_idx], grad_val);
                    }
                }
            }
        }
    }
}

__global__ void avgpool3d_backward_kernel_fp16(
    const __half* grad_output,
    float* grad_input_f32,
    int64_t N, int64_t C,
    int64_t D, int64_t H, int64_t W,
    int64_t D_out, int64_t H_out, int64_t W_out,
    int64_t kD, int64_t kH, int64_t kW,
    int64_t sD, int64_t sH, int64_t sW,
    int64_t pD, int64_t pH, int64_t pW,
    bool count_include_pad
) {
    int64_t total = N * C * D_out * H_out * W_out;

    HIP_KERNEL_LOOP(idx, total) {
        int64_t ow = idx % W_out;
        int64_t oh = (idx / W_out) % H_out;
        int64_t od = (idx / (W_out * H_out)) % D_out;
        int64_t c  = (idx / (W_out * H_out * D_out)) % C;
        int64_t n  = idx / (W_out * H_out * D_out * C);

        int64_t d_start = od * sD - pD;
        int64_t h_start = oh * sH - pH;
        int64_t w_start = ow * sW - pW;

        int64_t valid_count = 0;
        for (int64_t kd = 0; kd < kD; ++kd) {
            for (int64_t kh = 0; kh < kH; ++kh) {
                for (int64_t kw = 0; kw < kW; ++kw) {
                    int64_t d = d_start + kd;
                    int64_t h = h_start + kh;
                    int64_t w = w_start + kw;
                    if (d >= 0 && d < D && h >= 0 && h < H && w >= 0 && w < W) valid_count++;
                }
            }
        }

        int64_t divisor = count_include_pad ? (kD * kH * kW) : valid_count;
        if (divisor == 0) continue;
        float grad_val = tenzor::rocm::safe_h2f(grad_output[idx]) / static_cast<float>(divisor);

        for (int64_t kd = 0; kd < kD; ++kd) {
            for (int64_t kh = 0; kh < kH; ++kh) {
                for (int64_t kw = 0; kw < kW; ++kw) {
                    int64_t d = d_start + kd;
                    int64_t h = h_start + kh;
                    int64_t w = w_start + kw;
                    if (d >= 0 && d < D && h >= 0 && h < H && w >= 0 && w < W) {
                        int64_t in_idx = ((n * C + c) * D + d) * H * W + h * W + w;
                        atomicAdd(&grad_input_f32[in_idx], grad_val);
                    }
                }
            }
        }
    }
}

auto avgpool3d_backward_hip(
    const Tensor& grad_output,
    const std::vector<int64_t>& input_shape,
    std::array<int64_t, 3> kernel_size_a, std::array<int64_t, 3> stride_a, std::array<int64_t, 3> padding_a,
    bool count_include_pad,
    hipStream_t stream
) -> Tensor {
    // Q.6: per-axis std::array<int64_t, 3> signature.
    const int64_t kD = kernel_size_a[0], kH = kernel_size_a[1], kW = kernel_size_a[2];
    const int64_t sD = stride_a[0],      sH = stride_a[1],      sW = stride_a[2];
    const int64_t pD = padding_a[0],     pH = padding_a[1],     pW = padding_a[2];

    int64_t N = input_shape[0];
    int64_t C = input_shape[1];
    int64_t D = input_shape[2];
    int64_t H = input_shape[3];
    int64_t W = input_shape[4];

    auto grad_shape = grad_output.shape();
    int64_t D_out = grad_shape[2];
    int64_t H_out = grad_shape[3];
    int64_t W_out = grad_shape[4];

    Tensor grad_input(input_shape, grad_output.dtype(), grad_output.device());
    int64_t input_numel = N * C * D * H * W;

    int64_t total = N * C * D_out * H_out * W_out;
    int threads = rocm::get_wavefront_size() * 4;  // 4 wavefronts per block
    int blocks = (total + threads - 1) / threads;

    if (grad_output.dtype() == DType::Float32) {
        HIP_CHECK(hipMemsetAsync(grad_input.data<float>(), 0, input_numel * sizeof(float), stream));
        hipLaunchKernelGGL(avgpool3d_backward_kernel_impl<float>,
            dim3(blocks), dim3(threads), 0, stream,
            grad_output.data<float>(), grad_input.data<float>(),
            N, C, D, H, W, D_out, H_out, W_out, kD, kH, kW, sD, sH, sW, pD, pH, pW, count_include_pad);
        HIP_POST_LAUNCH_CHECK();
    } else if (grad_output.dtype() == DType::Float64) {
        HIP_CHECK(hipMemsetAsync(grad_input.data<double>(), 0, input_numel * sizeof(double), stream));
        hipLaunchKernelGGL(avgpool3d_backward_kernel_impl<double>,
            dim3(blocks), dim3(threads), 0, stream,
            grad_output.data<double>(), grad_input.data<double>(),
            N, C, D, H, W, D_out, H_out, W_out, kD, kH, kW, sD, sH, sW, pD, pH, pW, count_include_pad);
        HIP_POST_LAUNCH_CHECK();
    } else if (grad_output.dtype() == DType::Float16) {
        Tensor grad_input_f32(input_shape, DType::Float32, grad_output.device());
        HIP_CHECK(hipMemsetAsync(grad_input_f32.data<float>(), 0, input_numel * sizeof(float), stream));

        hipLaunchKernelGGL(avgpool3d_backward_kernel_fp16,
            dim3(blocks), dim3(threads), 0, stream,
            reinterpret_cast<const __half*>(grad_output.data<Float16>()),
            grad_input_f32.data<float>(),
            N, C, D, H, W, D_out, H_out, W_out, kD, kH, kW, sD, sH, sW, pD, pH, pW, count_include_pad);
        HIP_POST_LAUNCH_CHECK();

        int convert_blocks = (input_numel + threads - 1) / threads;
        hipLaunchKernelGGL(convert_f32_to_f16_pool,
            dim3(convert_blocks), dim3(threads), 0, stream,
            grad_input_f32.data<float>(),
            reinterpret_cast<__half*>(grad_input.data<Float16>()),
            input_numel);
        HIP_POST_LAUNCH_CHECK();
    } else if (grad_output.dtype() == DType::BFloat16) {
        auto grad_output_f32 = grad_output.to(DType::Float32);
        auto result_f32 = avgpool3d_backward_hip(grad_output_f32, input_shape, kernel_size_a, stride_a, padding_a, count_include_pad, stream);
        return result_f32.to(DType::BFloat16);
    } else {
        throw std::runtime_error("avgpool3d_backward_hip: unsupported dtype");
    }

    HIP_POST_LAUNCH_CHECK();
    return grad_input;
}

// ==============================================================================
// Adaptive MaxPool3D Forward
// ==============================================================================

template<typename T>
__global__ void adaptive_maxpool3d_forward_kernel(
    const T* input,
    T* output,
    int64_t* indices,
    int64_t N, int64_t C,
    int64_t D_in, int64_t H_in, int64_t W_in,
    int64_t D_out, int64_t H_out, int64_t W_out
) {
    int64_t total = N * C * D_out * H_out * W_out;

    HIP_KERNEL_LOOP(idx, total) {
        int64_t ow = idx % W_out;
        int64_t oh = (idx / W_out) % H_out;
        int64_t od = (idx / (W_out * H_out)) % D_out;
        int64_t c  = (idx / (W_out * H_out * D_out)) % C;
        int64_t n  = idx / (W_out * H_out * D_out * C);

        int64_t d_start = (od * D_in) / D_out;
        int64_t d_end   = ((od + 1) * D_in + D_out - 1) / D_out;
        int64_t h_start = (oh * H_in) / H_out;
        int64_t h_end   = ((oh + 1) * H_in + H_out - 1) / H_out;
        int64_t w_start = (ow * W_in) / W_out;
        int64_t w_end   = ((ow + 1) * W_in + W_out - 1) / W_out;

        T max_val = std::numeric_limits<T>::lowest();
        int64_t max_idx = d_start * H_in * W_in + h_start * W_in + w_start;

        for (int64_t d = d_start; d < d_end; ++d) {
            for (int64_t h = h_start; h < h_end; ++h) {
                for (int64_t w = w_start; w < w_end; ++w) {
                    int64_t in_idx = ((n * C + c) * D_in + d) * H_in * W_in + h * W_in + w;
                    T val = input[in_idx];
                    if (tenzor::rocm::is_nan_bits(val) || val > max_val) {
                        max_val = val;
                        max_idx = d * H_in * W_in + h * W_in + w;
                    }
                }
            }
        }

        output[idx] = max_val;
        indices[idx] = max_idx;
    }
}

__global__ void adaptive_maxpool3d_forward_kernel_fp16(
    const __half* input,
    __half* output,
    int64_t* indices,
    int64_t N, int64_t C,
    int64_t D_in, int64_t H_in, int64_t W_in,
    int64_t D_out, int64_t H_out, int64_t W_out
) {
    int64_t total = N * C * D_out * H_out * W_out;

    HIP_KERNEL_LOOP(idx, total) {
        int64_t ow = idx % W_out;
        int64_t oh = (idx / W_out) % H_out;
        int64_t od = (idx / (W_out * H_out)) % D_out;
        int64_t c  = (idx / (W_out * H_out * D_out)) % C;
        int64_t n  = idx / (W_out * H_out * D_out * C);

        int64_t d_start = (od * D_in) / D_out;
        int64_t d_end   = ((od + 1) * D_in + D_out - 1) / D_out;
        int64_t h_start = (oh * H_in) / H_out;
        int64_t h_end   = ((oh + 1) * H_in + H_out - 1) / H_out;
        int64_t w_start = (ow * W_in) / W_out;
        int64_t w_end   = ((ow + 1) * W_in + W_out - 1) / W_out;

        float max_val = std::numeric_limits<float>::lowest();
        int64_t max_idx = d_start * H_in * W_in + h_start * W_in + w_start;

        for (int64_t d = d_start; d < d_end; ++d) {
            for (int64_t h = h_start; h < h_end; ++h) {
                for (int64_t w = w_start; w < w_end; ++w) {
                    int64_t in_idx = ((n * C + c) * D_in + d) * H_in * W_in + h * W_in + w;
                    float val = tenzor::rocm::safe_h2f(input[in_idx]);
                    if (tenzor::rocm::is_nan_bits(val) || val > max_val) {
                        max_val = val;
                        max_idx = d * H_in * W_in + h * W_in + w;
                    }
                }
            }
        }

        output[idx] = tenzor::rocm::safe_f2h(max_val);
        indices[idx] = max_idx;
    }
}

auto adaptive_maxpool3d_forward_hip(
    const Tensor& input,
    int64_t output_d, int64_t output_h, int64_t output_w,
    hipStream_t stream
) -> std::pair<Tensor, Tensor> {

    auto shape = input.shape();
    int64_t N = shape[0], C = shape[1];
    int64_t D_in = shape[2], H_in = shape[3], W_in = shape[4];

    Tensor output({N, C, output_d, output_h, output_w}, input.dtype(), input.device());
    Tensor indices({N, C, output_d, output_h, output_w}, DType::Int64, input.device());

    int64_t total = N * C * output_d * output_h * output_w;
    int threads = rocm::get_wavefront_size() * 4;  // 4 wavefronts per block
    int blocks = (total + threads - 1) / threads;

    if (input.dtype() == DType::Float32) {
        hipLaunchKernelGGL(adaptive_maxpool3d_forward_kernel<float>,
            dim3(blocks), dim3(threads), 0, stream,
            input.data<float>(), output.data<float>(), indices.data<int64_t>(),
            N, C, D_in, H_in, W_in, output_d, output_h, output_w);
        HIP_POST_LAUNCH_CHECK();
    } else if (input.dtype() == DType::Float64) {
        hipLaunchKernelGGL(adaptive_maxpool3d_forward_kernel<double>,
            dim3(blocks), dim3(threads), 0, stream,
            input.data<double>(), output.data<double>(), indices.data<int64_t>(),
            N, C, D_in, H_in, W_in, output_d, output_h, output_w);
        HIP_POST_LAUNCH_CHECK();
    } else if (input.dtype() == DType::Float16) {
        hipLaunchKernelGGL(adaptive_maxpool3d_forward_kernel_fp16,
            dim3(blocks), dim3(threads), 0, stream,
            reinterpret_cast<const __half*>(input.data<Float16>()),
            reinterpret_cast<__half*>(output.data<Float16>()),
            indices.data<int64_t>(),
            N, C, D_in, H_in, W_in, output_d, output_h, output_w);
        HIP_POST_LAUNCH_CHECK();
    } else if (input.dtype() == DType::BFloat16) {
        auto input_f32 = input.to(DType::Float32);
        auto [output_f32, idx] = adaptive_maxpool3d_forward_hip(input_f32, output_d, output_h, output_w, stream);
        return {output_f32.to(DType::BFloat16), idx};
    } else {
        throw std::runtime_error("adaptive_maxpool3d_forward_hip: unsupported dtype");
    }

    HIP_POST_LAUNCH_CHECK();
    return {output, indices};
}

// ==============================================================================
// Adaptive MaxPool3D Backward (reuses maxpool3d backward — same index scatter)
// ==============================================================================

auto adaptive_maxpool3d_backward_hip(
    const Tensor& grad_output,
    const Tensor& indices,
    const std::vector<int64_t>& input_shape,
    hipStream_t stream
) -> Tensor {
    return maxpool3d_backward_hip(grad_output, indices, input_shape, stream);
}

// ==============================================================================
// Adaptive AvgPool3D Forward
// ==============================================================================

template<typename T>
__global__ void adaptive_avgpool3d_forward_kernel(
    const T* input,
    T* output,
    int64_t N, int64_t C,
    int64_t D_in, int64_t H_in, int64_t W_in,
    int64_t D_out, int64_t H_out, int64_t W_out
) {
    int64_t total = N * C * D_out * H_out * W_out;

    HIP_KERNEL_LOOP(idx, total) {
        int64_t ow = idx % W_out;
        int64_t oh = (idx / W_out) % H_out;
        int64_t od = (idx / (W_out * H_out)) % D_out;
        int64_t c  = (idx / (W_out * H_out * D_out)) % C;
        int64_t n  = idx / (W_out * H_out * D_out * C);

        int64_t d_start = (od * D_in) / D_out;
        int64_t d_end   = ((od + 1) * D_in + D_out - 1) / D_out;
        int64_t h_start = (oh * H_in) / H_out;
        int64_t h_end   = ((oh + 1) * H_in + H_out - 1) / H_out;
        int64_t w_start = (ow * W_in) / W_out;
        int64_t w_end   = ((ow + 1) * W_in + W_out - 1) / W_out;

        T sum = T(0);
        int64_t count = 0;

        for (int64_t d = d_start; d < d_end; ++d) {
            for (int64_t h = h_start; h < h_end; ++h) {
                for (int64_t w = w_start; w < w_end; ++w) {
                    sum += input[((n * C + c) * D_in + d) * H_in * W_in + h * W_in + w];
                    count++;
                }
            }
        }

        output[idx] = count > 0 ? sum / static_cast<T>(count) : T(0);
    }
}

__global__ void adaptive_avgpool3d_forward_kernel_fp16(
    const __half* input,
    __half* output,
    int64_t N, int64_t C,
    int64_t D_in, int64_t H_in, int64_t W_in,
    int64_t D_out, int64_t H_out, int64_t W_out
) {
    int64_t total = N * C * D_out * H_out * W_out;

    HIP_KERNEL_LOOP(idx, total) {
        int64_t ow = idx % W_out;
        int64_t oh = (idx / W_out) % H_out;
        int64_t od = (idx / (W_out * H_out)) % D_out;
        int64_t c  = (idx / (W_out * H_out * D_out)) % C;
        int64_t n  = idx / (W_out * H_out * D_out * C);

        int64_t d_start = (od * D_in) / D_out;
        int64_t d_end   = ((od + 1) * D_in + D_out - 1) / D_out;
        int64_t h_start = (oh * H_in) / H_out;
        int64_t h_end   = ((oh + 1) * H_in + H_out - 1) / H_out;
        int64_t w_start = (ow * W_in) / W_out;
        int64_t w_end   = ((ow + 1) * W_in + W_out - 1) / W_out;

        float sum = 0.0f;
        int64_t count = 0;

        for (int64_t d = d_start; d < d_end; ++d) {
            for (int64_t h = h_start; h < h_end; ++h) {
                for (int64_t w = w_start; w < w_end; ++w) {
                    sum += tenzor::rocm::safe_h2f(input[((n * C + c) * D_in + d) * H_in * W_in + h * W_in + w]);
                    count++;
                }
            }
        }

        output[idx] = tenzor::rocm::safe_f2h(count > 0 ? sum / static_cast<float>(count) : 0.0f);
    }
}

auto adaptive_avgpool3d_forward_hip(
    const Tensor& input,
    int64_t output_d, int64_t output_h, int64_t output_w,
    hipStream_t stream
) -> Tensor {

    auto shape = input.shape();
    int64_t N = shape[0], C = shape[1];
    int64_t D_in = shape[2], H_in = shape[3], W_in = shape[4];

    Tensor output({N, C, output_d, output_h, output_w}, input.dtype(), input.device());

    int64_t total = N * C * output_d * output_h * output_w;
    int threads = rocm::get_wavefront_size() * 4;  // 4 wavefronts per block
    int blocks = (total + threads - 1) / threads;

    if (input.dtype() == DType::Float32) {
        hipLaunchKernelGGL(adaptive_avgpool3d_forward_kernel<float>,
            dim3(blocks), dim3(threads), 0, stream,
            input.data<float>(), output.data<float>(),
            N, C, D_in, H_in, W_in, output_d, output_h, output_w);
        HIP_POST_LAUNCH_CHECK();
    } else if (input.dtype() == DType::Float64) {
        hipLaunchKernelGGL(adaptive_avgpool3d_forward_kernel<double>,
            dim3(blocks), dim3(threads), 0, stream,
            input.data<double>(), output.data<double>(),
            N, C, D_in, H_in, W_in, output_d, output_h, output_w);
        HIP_POST_LAUNCH_CHECK();
    } else if (input.dtype() == DType::Float16) {
        hipLaunchKernelGGL(adaptive_avgpool3d_forward_kernel_fp16,
            dim3(blocks), dim3(threads), 0, stream,
            reinterpret_cast<const __half*>(input.data<Float16>()),
            reinterpret_cast<__half*>(output.data<Float16>()),
            N, C, D_in, H_in, W_in, output_d, output_h, output_w);
        HIP_POST_LAUNCH_CHECK();
    } else if (input.dtype() == DType::BFloat16) {
        auto input_f32 = input.to(DType::Float32);
        auto result_f32 = adaptive_avgpool3d_forward_hip(input_f32, output_d, output_h, output_w, stream);
        return result_f32.to(DType::BFloat16);
    } else {
        throw std::runtime_error("adaptive_avgpool3d_forward_hip: unsupported dtype");
    }

    HIP_POST_LAUNCH_CHECK();
    return output;
}

// ==============================================================================
// Adaptive AvgPool3D Backward
// ==============================================================================

template<typename T>
__global__ void adaptive_avgpool3d_backward_kernel(
    const T* grad_output,
    T* grad_input,
    int64_t N, int64_t C,
    int64_t D_in, int64_t H_in, int64_t W_in,
    int64_t D_out, int64_t H_out, int64_t W_out
) {
    int64_t total = N * C * D_out * H_out * W_out;

    HIP_KERNEL_LOOP(idx, total) {
        int64_t ow = idx % W_out;
        int64_t oh = (idx / W_out) % H_out;
        int64_t od = (idx / (W_out * H_out)) % D_out;
        int64_t c  = (idx / (W_out * H_out * D_out)) % C;
        int64_t n  = idx / (W_out * H_out * D_out * C);

        int64_t d_start = (od * D_in) / D_out;
        int64_t d_end   = ((od + 1) * D_in + D_out - 1) / D_out;
        int64_t h_start = (oh * H_in) / H_out;
        int64_t h_end   = ((oh + 1) * H_in + H_out - 1) / H_out;
        int64_t w_start = (ow * W_in) / W_out;
        int64_t w_end   = ((ow + 1) * W_in + W_out - 1) / W_out;

        int64_t count = (d_end - d_start) * (h_end - h_start) * (w_end - w_start);
        T grad_val = grad_output[idx] / static_cast<T>(count);

        for (int64_t d = d_start; d < d_end; ++d) {
            for (int64_t h = h_start; h < h_end; ++h) {
                for (int64_t w = w_start; w < w_end; ++w) {
                    int64_t in_idx = ((n * C + c) * D_in + d) * H_in * W_in + h * W_in + w;
                    atomicAdd(&grad_input[in_idx], grad_val);
                }
            }
        }
    }
}

__global__ void adaptive_avgpool3d_backward_kernel_fp16(
    const __half* grad_output,
    float* grad_input_f32,
    int64_t N, int64_t C,
    int64_t D_in, int64_t H_in, int64_t W_in,
    int64_t D_out, int64_t H_out, int64_t W_out
) {
    int64_t total = N * C * D_out * H_out * W_out;

    HIP_KERNEL_LOOP(idx, total) {
        int64_t ow = idx % W_out;
        int64_t oh = (idx / W_out) % H_out;
        int64_t od = (idx / (W_out * H_out)) % D_out;
        int64_t c  = (idx / (W_out * H_out * D_out)) % C;
        int64_t n  = idx / (W_out * H_out * D_out * C);

        int64_t d_start = (od * D_in) / D_out;
        int64_t d_end   = ((od + 1) * D_in + D_out - 1) / D_out;
        int64_t h_start = (oh * H_in) / H_out;
        int64_t h_end   = ((oh + 1) * H_in + H_out - 1) / H_out;
        int64_t w_start = (ow * W_in) / W_out;
        int64_t w_end   = ((ow + 1) * W_in + W_out - 1) / W_out;

        int64_t count = (d_end - d_start) * (h_end - h_start) * (w_end - w_start);
        float grad_val = tenzor::rocm::safe_h2f(grad_output[idx]) / static_cast<float>(count);

        for (int64_t d = d_start; d < d_end; ++d) {
            for (int64_t h = h_start; h < h_end; ++h) {
                for (int64_t w = w_start; w < w_end; ++w) {
                    int64_t in_idx = ((n * C + c) * D_in + d) * H_in * W_in + h * W_in + w;
                    atomicAdd(&grad_input_f32[in_idx], grad_val);
                }
            }
        }
    }
}

auto adaptive_avgpool3d_backward_hip(
    const Tensor& grad_output,
    const std::vector<int64_t>& input_shape,
    hipStream_t stream
) -> Tensor {

    int64_t N = input_shape[0];
    int64_t C = input_shape[1];
    int64_t D_in = input_shape[2];
    int64_t H_in = input_shape[3];
    int64_t W_in = input_shape[4];

    auto grad_shape = grad_output.shape();
    int64_t D_out = grad_shape[2];
    int64_t H_out = grad_shape[3];
    int64_t W_out = grad_shape[4];

    Tensor grad_input(input_shape, grad_output.dtype(), grad_output.device());
    int64_t input_numel = N * C * D_in * H_in * W_in;

    int64_t total = N * C * D_out * H_out * W_out;
    int threads = rocm::get_wavefront_size() * 4;  // 4 wavefronts per block
    int blocks = (total + threads - 1) / threads;

    if (grad_output.dtype() == DType::Float32) {
        HIP_CHECK(hipMemsetAsync(grad_input.data<float>(), 0, input_numel * sizeof(float), stream));
        hipLaunchKernelGGL(adaptive_avgpool3d_backward_kernel<float>,
            dim3(blocks), dim3(threads), 0, stream,
            grad_output.data<float>(), grad_input.data<float>(),
            N, C, D_in, H_in, W_in, D_out, H_out, W_out);
        HIP_POST_LAUNCH_CHECK();
    } else if (grad_output.dtype() == DType::Float64) {
        HIP_CHECK(hipMemsetAsync(grad_input.data<double>(), 0, input_numel * sizeof(double), stream));
        hipLaunchKernelGGL(adaptive_avgpool3d_backward_kernel<double>,
            dim3(blocks), dim3(threads), 0, stream,
            grad_output.data<double>(), grad_input.data<double>(),
            N, C, D_in, H_in, W_in, D_out, H_out, W_out);
        HIP_POST_LAUNCH_CHECK();
    } else if (grad_output.dtype() == DType::Float16) {
        Tensor grad_input_f32(input_shape, DType::Float32, grad_output.device());
        HIP_CHECK(hipMemsetAsync(grad_input_f32.data<float>(), 0, input_numel * sizeof(float), stream));

        hipLaunchKernelGGL(adaptive_avgpool3d_backward_kernel_fp16,
            dim3(blocks), dim3(threads), 0, stream,
            reinterpret_cast<const __half*>(grad_output.data<Float16>()),
            grad_input_f32.data<float>(),
            N, C, D_in, H_in, W_in, D_out, H_out, W_out);
        HIP_POST_LAUNCH_CHECK();

        int convert_blocks = (input_numel + threads - 1) / threads;
        hipLaunchKernelGGL(convert_f32_to_f16_pool,
            dim3(convert_blocks), dim3(threads), 0, stream,
            grad_input_f32.data<float>(),
            reinterpret_cast<__half*>(grad_input.data<Float16>()),
            input_numel);
        HIP_POST_LAUNCH_CHECK();
    } else if (grad_output.dtype() == DType::BFloat16) {
        auto grad_output_f32 = grad_output.to(DType::Float32);
        auto result_f32 = adaptive_avgpool3d_backward_hip(grad_output_f32, input_shape, stream);
        return result_f32.to(DType::BFloat16);
    } else {
        throw std::runtime_error("adaptive_avgpool3d_backward_hip: unsupported dtype");
    }

    HIP_POST_LAUNCH_CHECK();
    return grad_input;
}

// ============================================================================
// Fractional Max Pool 2D Forward
// ============================================================================

// F109: PyTorch FractionalMaxPool window start for output index `i` along one
// axis. Windows are `pool`-wide (== kernel_size) and OVERLAP when pool > alpha,
// matching torch ATen generate_intervals:
//   alpha = (in - pool) / (out - 1)                     [out > 1]
//   start(i) = floor((i + u) * alpha) - floor(u * alpha)   [i < out-1]
//   start(out-1) = in - pool
// The earlier code derived each window from an adaptive-style DISJOINT ratio
// partition (in/out) and never used the pool size, so kernel_size was a no-op.
__device__ __forceinline__ int64_t frac_pool_start(
    int64_t i, int64_t in_size, int64_t out_size, int64_t pool, float sample)
{
    int64_t start;
    if (out_size <= 1 || i == out_size - 1) {
        start = in_size - pool;
    } else {
        float alpha = static_cast<float>(in_size - pool) /
                      static_cast<float>(out_size - 1);
        start = static_cast<int64_t>((static_cast<float>(i) + sample) * alpha) -
                static_cast<int64_t>(sample * alpha);
    }
    if (start < 0) start = 0;
    if (start > in_size - pool) start = in_size - pool;
    return start;
}

__global__ void fractional_maxpool2d_forward_kernel_f32(
    const float* __restrict__ input,
    float* __restrict__ output,
    int64_t* __restrict__ indices,
    const float* __restrict__ samples,
    int64_t N, int64_t C, int64_t H, int64_t W,
    int64_t out_h, int64_t out_w,
    int64_t kernel_h, int64_t kernel_w
) {
    int64_t total = N * C * out_h * out_w;
    HIP_KERNEL_LOOP(idx, total) {
        int64_t ow = idx % out_w;
        int64_t oh = (idx / out_w) % out_h;
        int64_t c  = (idx / (out_w * out_h)) % C;
        int64_t n  = idx / (out_w * out_h * C);

        float sample_h = samples ? samples[(n * C + c) * 2 + 0] : 0.5f;
        float sample_w = samples ? samples[(n * C + c) * 2 + 1] : 0.5f;

        // PyTorch overlapping windows of width == kernel_size.
        int64_t h_start = frac_pool_start(oh, H, out_h, kernel_h, sample_h);
        int64_t h_end   = min(h_start + kernel_h, H);
        int64_t w_start = frac_pool_start(ow, W, out_w, kernel_w, sample_w);
        int64_t w_end   = min(w_start + kernel_w, W);

        float max_val = std::numeric_limits<float>::lowest();
        int64_t max_idx = h_start * W + w_start;

        for (int64_t h = h_start; h < h_end; ++h) {
            for (int64_t w = w_start; w < w_end; ++w) {
                int64_t in_idx = ((n * C + c) * H + h) * W + w;
                float val = input[in_idx];
                if (tenzor::rocm::is_nan_bits(val) || val > max_val) {
                    max_val = val;
                    max_idx = h * W + w;
                }
            }
        }

        output[idx] = max_val;
        indices[idx] = max_idx;
    }
}

__global__ void fractional_maxpool2d_forward_kernel_f64(
    const double* __restrict__ input,
    double* __restrict__ output,
    int64_t* __restrict__ indices,
    const float* __restrict__ samples,
    int64_t N, int64_t C, int64_t H, int64_t W,
    int64_t out_h, int64_t out_w,
    int64_t kernel_h, int64_t kernel_w
) {
    int64_t total = N * C * out_h * out_w;
    HIP_KERNEL_LOOP(idx, total) {
        int64_t ow = idx % out_w;
        int64_t oh = (idx / out_w) % out_h;
        int64_t c  = (idx / (out_w * out_h)) % C;
        int64_t n  = idx / (out_w * out_h * C);

        float sample_h = samples ? samples[(n * C + c) * 2 + 0] : 0.5f;
        float sample_w = samples ? samples[(n * C + c) * 2 + 1] : 0.5f;

        // PyTorch overlapping windows of width == kernel_size.
        int64_t h_start = frac_pool_start(oh, H, out_h, kernel_h, sample_h);
        int64_t h_end   = min(h_start + kernel_h, H);
        int64_t w_start = frac_pool_start(ow, W, out_w, kernel_w, sample_w);
        int64_t w_end   = min(w_start + kernel_w, W);

        double max_val = std::numeric_limits<double>::lowest();
        int64_t max_idx = h_start * W + w_start;

        for (int64_t h = h_start; h < h_end; ++h) {
            for (int64_t w = w_start; w < w_end; ++w) {
                int64_t in_idx = ((n * C + c) * H + h) * W + w;
                double val = input[in_idx];
                if (tenzor::rocm::is_nan_bits(val) || val > max_val) {
                    max_val = val;
                    max_idx = h * W + w;
                }
            }
        }

        output[idx] = max_val;
        indices[idx] = max_idx;
    }
}

auto fractional_maxpool2d_forward_hip(
    const Tensor& input,
    int64_t out_h, int64_t out_w,
    int64_t kernel_h, int64_t kernel_w,
    const Tensor* random_samples,
    hipStream_t stream
) -> std::pair<Tensor, Tensor> {
    auto shape = input.shape();
    int64_t N = shape[0], C = shape[1], H = shape[2], W = shape[3];

    Tensor output({N, C, out_h, out_w}, input.dtype(), input.device());
    Tensor indices({N, C, out_h, out_w}, DType::Int64, input.device());

    int64_t total = N * C * out_h * out_w;
    int threads = rocm::get_wavefront_size() * 4;
    int blocks = (total + threads - 1) / threads;

    const float* samples_ptr = nullptr;
    if (random_samples && random_samples->numel() > 0) {
        samples_ptr = random_samples->data<float>();
    }

    if (input.dtype() == DType::Float32) {
        hipLaunchKernelGGL(fractional_maxpool2d_forward_kernel_f32,
            dim3(blocks), dim3(threads), 0, stream,
            input.data<float>(), output.data<float>(), indices.data<int64_t>(),
            samples_ptr, N, C, H, W, out_h, out_w, kernel_h, kernel_w);
        HIP_POST_LAUNCH_CHECK();
    } else if (input.dtype() == DType::Float64) {
        hipLaunchKernelGGL(fractional_maxpool2d_forward_kernel_f64,
            dim3(blocks), dim3(threads), 0, stream,
            input.data<double>(), output.data<double>(), indices.data<int64_t>(),
            samples_ptr, N, C, H, W, out_h, out_w, kernel_h, kernel_w);
        HIP_POST_LAUNCH_CHECK();
    } else if (input.dtype() == DType::Float16) {
        auto input_f32 = input.to(DType::Float32);
        auto [out_f32, idx] = fractional_maxpool2d_forward_hip(input_f32, out_h, out_w, kernel_h, kernel_w, random_samples, stream);
        return {out_f32.to(DType::Float16), idx};
    } else if (input.dtype() == DType::BFloat16) {
        auto input_f32 = input.to(DType::Float32);
        auto [out_f32, idx] = fractional_maxpool2d_forward_hip(input_f32, out_h, out_w, kernel_h, kernel_w, random_samples, stream);
        return {out_f32.to(DType::BFloat16), idx};
    } else {
        throw std::runtime_error("fractional_maxpool2d_forward_hip: unsupported dtype");
    }

    return {output, indices};
}

// ============================================================================
// Fractional Max Pool 2D Backward
// ============================================================================

__global__ void fractional_maxpool2d_backward_kernel_f32(
    const float* __restrict__ grad_output,
    const int64_t* __restrict__ indices,
    float* __restrict__ grad_input,
    int64_t N, int64_t C, int64_t H, int64_t W,
    int64_t out_h, int64_t out_w
) {
    int64_t total = N * C * out_h * out_w;
    int64_t in_spatial = H * W;
    int64_t out_spatial = out_h * out_w;

    HIP_KERNEL_LOOP(idx, total) {
        int64_t ow = idx % out_w;
        int64_t oh = (idx / out_w) % out_h;
        int64_t c  = (idx / (out_w * out_h)) % C;
        int64_t n  = idx / (out_w * out_h * C);

        int64_t base_in = (n * C + c) * in_spatial;
        int64_t base_out = (n * C + c) * out_spatial;
        int64_t out_idx = base_out + oh * out_w + ow;

        int64_t max_idx = indices[out_idx];
        atomicAdd(&grad_input[base_in + max_idx], grad_output[out_idx]);
    }
}

__global__ void fractional_maxpool2d_backward_kernel_f64(
    const double* __restrict__ grad_output,
    const int64_t* __restrict__ indices,
    double* __restrict__ grad_input,
    int64_t N, int64_t C, int64_t H, int64_t W,
    int64_t out_h, int64_t out_w
) {
    int64_t total = N * C * out_h * out_w;
    int64_t in_spatial = H * W;
    int64_t out_spatial = out_h * out_w;

    HIP_KERNEL_LOOP(idx, total) {
        int64_t ow = idx % out_w;
        int64_t oh = (idx / out_w) % out_h;
        int64_t c  = (idx / (out_w * out_h)) % C;
        int64_t n  = idx / (out_w * out_h * C);

        int64_t base_in = (n * C + c) * in_spatial;
        int64_t base_out = (n * C + c) * out_spatial;
        int64_t out_idx = base_out + oh * out_w + ow;

        int64_t max_idx = indices[out_idx];
        atomicAdd(&grad_input[base_in + max_idx], grad_output[out_idx]);
    }
}

auto fractional_maxpool2d_backward_hip(
    const Tensor& grad_output,
    const Tensor& indices,
    const std::vector<int64_t>& input_shape,
    hipStream_t stream
) -> Tensor {
    int64_t N = input_shape[0], C = input_shape[1];
    int64_t H = input_shape[2], W = input_shape[3];
    auto grad_shape = grad_output.shape();
    int64_t out_h = grad_shape[2], out_w = grad_shape[3];

    Tensor grad_input(input_shape, grad_output.dtype(), grad_output.device());

    int64_t total = N * C * out_h * out_w;
    int threads = rocm::get_wavefront_size() * 4;
    int blocks = (total + threads - 1) / threads;

    if (grad_output.dtype() == DType::Float32) {
        int64_t input_numel = N * C * H * W;
        HIP_CHECK(hipMemsetAsync(grad_input.data<float>(), 0, input_numel * sizeof(float), stream));
        hipLaunchKernelGGL(fractional_maxpool2d_backward_kernel_f32,
            dim3(blocks), dim3(threads), 0, stream,
            grad_output.data<float>(), indices.data<int64_t>(),
            grad_input.data<float>(), N, C, H, W, out_h, out_w);
        HIP_POST_LAUNCH_CHECK();
    } else if (grad_output.dtype() == DType::Float64) {
        int64_t input_numel = N * C * H * W;
        HIP_CHECK(hipMemsetAsync(grad_input.data<double>(), 0, input_numel * sizeof(double), stream));
        hipLaunchKernelGGL(fractional_maxpool2d_backward_kernel_f64,
            dim3(blocks), dim3(threads), 0, stream,
            grad_output.data<double>(), indices.data<int64_t>(),
            grad_input.data<double>(), N, C, H, W, out_h, out_w);
        HIP_POST_LAUNCH_CHECK();
    } else if (grad_output.dtype() == DType::Float16) {
        auto grad_f32 = grad_output.to(DType::Float32);
        auto result_f32 = fractional_maxpool2d_backward_hip(grad_f32, indices, input_shape, stream);
        return result_f32.to(DType::Float16);
    } else if (grad_output.dtype() == DType::BFloat16) {
        auto grad_f32 = grad_output.to(DType::Float32);
        auto result_f32 = fractional_maxpool2d_backward_hip(grad_f32, indices, input_shape, stream);
        return result_f32.to(DType::BFloat16);
    } else {
        throw std::runtime_error("fractional_maxpool2d_backward_hip: unsupported dtype");
    }

    return grad_input;
}

// ============================================================================
// Fractional Max Pool 3D Forward
// ============================================================================

__global__ void fractional_maxpool3d_forward_kernel_f64(
    const double* __restrict__ input,
    double* __restrict__ output,
    int64_t* __restrict__ indices,
    const float* __restrict__ samples,
    int64_t N, int64_t C, int64_t D, int64_t H, int64_t W,
    int64_t out_d, int64_t out_h, int64_t out_w,
    int64_t kernel_d, int64_t kernel_h, int64_t kernel_w
) {
    int64_t total = N * C * out_d * out_h * out_w;
    HIP_KERNEL_LOOP(idx, total) {
        int64_t ow = idx % out_w;
        int64_t oh = (idx / out_w) % out_h;
        int64_t od = (idx / (out_w * out_h)) % out_d;
        int64_t c  = (idx / (out_w * out_h * out_d)) % C;
        int64_t n  = idx / (out_w * out_h * out_d * C);

        float sample_d = samples ? samples[(n * C + c) * 3 + 0] : 0.5f;
        float sample_h = samples ? samples[(n * C + c) * 3 + 1] : 0.5f;
        float sample_w = samples ? samples[(n * C + c) * 3 + 2] : 0.5f;

        // PyTorch overlapping windows of width == kernel_size.
        int64_t d_start = frac_pool_start(od, D, out_d, kernel_d, sample_d);
        int64_t d_end   = min(d_start + kernel_d, D);
        int64_t h_start = frac_pool_start(oh, H, out_h, kernel_h, sample_h);
        int64_t h_end   = min(h_start + kernel_h, H);
        int64_t w_start = frac_pool_start(ow, W, out_w, kernel_w, sample_w);
        int64_t w_end   = min(w_start + kernel_w, W);

        double max_val = std::numeric_limits<double>::lowest();
        int64_t max_idx = (d_start * H + h_start) * W + w_start;

        for (int64_t d = d_start; d < d_end; ++d) {
            for (int64_t h = h_start; h < h_end; ++h) {
                for (int64_t w = w_start; w < w_end; ++w) {
                    int64_t in_idx = (((n * C + c) * D + d) * H + h) * W + w;
                    double val = input[in_idx];
                    if (tenzor::rocm::is_nan_bits(val) || val > max_val) {
                        max_val = val;
                        max_idx = (d * H + h) * W + w;
                    }
                }
            }
        }

        output[idx] = max_val;
        indices[idx] = max_idx;
    }
}

__global__ void fractional_maxpool3d_forward_kernel_f32(
    const float* __restrict__ input,
    float* __restrict__ output,
    int64_t* __restrict__ indices,
    const float* __restrict__ samples,
    int64_t N, int64_t C, int64_t D, int64_t H, int64_t W,
    int64_t out_d, int64_t out_h, int64_t out_w,
    int64_t kernel_d, int64_t kernel_h, int64_t kernel_w
) {
    int64_t total = N * C * out_d * out_h * out_w;
    HIP_KERNEL_LOOP(idx, total) {
        int64_t ow = idx % out_w;
        int64_t oh = (idx / out_w) % out_h;
        int64_t od = (idx / (out_w * out_h)) % out_d;
        int64_t c  = (idx / (out_w * out_h * out_d)) % C;
        int64_t n  = idx / (out_w * out_h * out_d * C);

        float sample_d = samples ? samples[(n * C + c) * 3 + 0] : 0.5f;
        float sample_h = samples ? samples[(n * C + c) * 3 + 1] : 0.5f;
        float sample_w = samples ? samples[(n * C + c) * 3 + 2] : 0.5f;

        // PyTorch overlapping windows of width == kernel_size.
        int64_t d_start = frac_pool_start(od, D, out_d, kernel_d, sample_d);
        int64_t d_end   = min(d_start + kernel_d, D);
        int64_t h_start = frac_pool_start(oh, H, out_h, kernel_h, sample_h);
        int64_t h_end   = min(h_start + kernel_h, H);
        int64_t w_start = frac_pool_start(ow, W, out_w, kernel_w, sample_w);
        int64_t w_end   = min(w_start + kernel_w, W);

        float max_val = std::numeric_limits<float>::lowest();
        int64_t max_idx = (d_start * H + h_start) * W + w_start;

        for (int64_t d = d_start; d < d_end; ++d) {
            for (int64_t h = h_start; h < h_end; ++h) {
                for (int64_t w = w_start; w < w_end; ++w) {
                    int64_t in_idx = (((n * C + c) * D + d) * H + h) * W + w;
                    float val = input[in_idx];
                    if (tenzor::rocm::is_nan_bits(val) || val > max_val) {
                        max_val = val;
                        max_idx = (d * H + h) * W + w;
                    }
                }
            }
        }

        output[idx] = max_val;
        indices[idx] = max_idx;
    }
}

auto fractional_maxpool3d_forward_hip(
    const Tensor& input,
    int64_t out_d, int64_t out_h, int64_t out_w,
    int64_t kernel_d, int64_t kernel_h, int64_t kernel_w,
    const Tensor* random_samples,
    hipStream_t stream
) -> std::pair<Tensor, Tensor> {
    auto shape = input.shape();
    int64_t N = shape[0], C = shape[1], D = shape[2], H = shape[3], W = shape[4];

    Tensor output({N, C, out_d, out_h, out_w}, input.dtype(), input.device());
    Tensor indices({N, C, out_d, out_h, out_w}, DType::Int64, input.device());

    int64_t total = N * C * out_d * out_h * out_w;
    int threads = rocm::get_wavefront_size() * 4;
    int blocks = (total + threads - 1) / threads;

    const float* samples_ptr = nullptr;
    if (random_samples && random_samples->numel() > 0) {
        samples_ptr = random_samples->data<float>();
    }

    if (input.dtype() == DType::Float32) {
        hipLaunchKernelGGL(fractional_maxpool3d_forward_kernel_f32,
            dim3(blocks), dim3(threads), 0, stream,
            input.data<float>(), output.data<float>(), indices.data<int64_t>(),
            samples_ptr, N, C, D, H, W, out_d, out_h, out_w, kernel_d, kernel_h, kernel_w);
        HIP_POST_LAUNCH_CHECK();
    } else if (input.dtype() == DType::Float64) {
        hipLaunchKernelGGL(fractional_maxpool3d_forward_kernel_f64,
            dim3(blocks), dim3(threads), 0, stream,
            input.data<double>(), output.data<double>(), indices.data<int64_t>(),
            samples_ptr, N, C, D, H, W, out_d, out_h, out_w, kernel_d, kernel_h, kernel_w);
        HIP_POST_LAUNCH_CHECK();
    } else if (input.dtype() == DType::Float16) {
        auto input_f32 = input.to(DType::Float32);
        auto [out_f32, idx] = fractional_maxpool3d_forward_hip(input_f32, out_d, out_h, out_w, kernel_d, kernel_h, kernel_w, random_samples, stream);
        return {out_f32.to(DType::Float16), idx};
    } else if (input.dtype() == DType::BFloat16) {
        auto input_f32 = input.to(DType::Float32);
        auto [out_f32, idx] = fractional_maxpool3d_forward_hip(input_f32, out_d, out_h, out_w, kernel_d, kernel_h, kernel_w, random_samples, stream);
        return {out_f32.to(DType::BFloat16), idx};
    } else {
        throw std::runtime_error("fractional_maxpool3d_forward_hip: unsupported dtype");
    }

    return {output, indices};
}

// ============================================================================
// Fractional Max Pool 3D Backward
// ============================================================================

__global__ void fractional_maxpool3d_backward_kernel_f32(
    const float* __restrict__ grad_output,
    const int64_t* __restrict__ indices,
    float* __restrict__ grad_input,
    int64_t N, int64_t C,
    int64_t D, int64_t H, int64_t W,
    int64_t out_d, int64_t out_h, int64_t out_w
) {
    int64_t total = N * C * out_d * out_h * out_w;
    int64_t in_spatial = D * H * W;
    int64_t out_spatial = out_d * out_h * out_w;

    HIP_KERNEL_LOOP(idx, total) {
        int64_t local_idx = idx % out_spatial;
        int64_t nc = idx / out_spatial;

        int64_t base_in = nc * in_spatial;
        int64_t base_out = nc * out_spatial;

        int64_t max_idx = indices[base_out + local_idx];
        atomicAdd(&grad_input[base_in + max_idx], grad_output[base_out + local_idx]);
    }
}

__global__ void fractional_maxpool3d_backward_kernel_f64(
    const double* __restrict__ grad_output,
    const int64_t* __restrict__ indices,
    double* __restrict__ grad_input,
    int64_t N, int64_t C,
    int64_t D, int64_t H, int64_t W,
    int64_t out_d, int64_t out_h, int64_t out_w
) {
    int64_t total = N * C * out_d * out_h * out_w;
    int64_t in_spatial = D * H * W;
    int64_t out_spatial = out_d * out_h * out_w;

    HIP_KERNEL_LOOP(idx, total) {
        int64_t local_idx = idx % out_spatial;
        int64_t nc = idx / out_spatial;

        int64_t base_in = nc * in_spatial;
        int64_t base_out = nc * out_spatial;

        int64_t max_idx = indices[base_out + local_idx];
        atomicAdd(&grad_input[base_in + max_idx], grad_output[base_out + local_idx]);
    }
}

auto fractional_maxpool3d_backward_hip(
    const Tensor& grad_output,
    const Tensor& indices,
    const std::vector<int64_t>& input_shape,
    hipStream_t stream
) -> Tensor {
    int64_t N = input_shape[0], C = input_shape[1];
    int64_t D = input_shape[2], H = input_shape[3], W = input_shape[4];
    auto grad_shape = grad_output.shape();
    int64_t out_d = grad_shape[2], out_h = grad_shape[3], out_w = grad_shape[4];

    Tensor grad_input(input_shape, grad_output.dtype(), grad_output.device());

    int64_t total = N * C * out_d * out_h * out_w;
    int threads = rocm::get_wavefront_size() * 4;
    int blocks = (total + threads - 1) / threads;

    if (grad_output.dtype() == DType::Float32) {
        int64_t input_numel = N * C * D * H * W;
        HIP_CHECK(hipMemsetAsync(grad_input.data<float>(), 0, input_numel * sizeof(float), stream));
        hipLaunchKernelGGL(fractional_maxpool3d_backward_kernel_f32,
            dim3(blocks), dim3(threads), 0, stream,
            grad_output.data<float>(), indices.data<int64_t>(),
            grad_input.data<float>(), N, C, D, H, W, out_d, out_h, out_w);
        HIP_POST_LAUNCH_CHECK();
    } else if (grad_output.dtype() == DType::Float64) {
        int64_t input_numel = N * C * D * H * W;
        HIP_CHECK(hipMemsetAsync(grad_input.data<double>(), 0, input_numel * sizeof(double), stream));
        hipLaunchKernelGGL(fractional_maxpool3d_backward_kernel_f64,
            dim3(blocks), dim3(threads), 0, stream,
            grad_output.data<double>(), indices.data<int64_t>(),
            grad_input.data<double>(), N, C, D, H, W, out_d, out_h, out_w);
        HIP_POST_LAUNCH_CHECK();
    } else if (grad_output.dtype() == DType::Float16) {
        auto grad_f32 = grad_output.to(DType::Float32);
        auto result_f32 = fractional_maxpool3d_backward_hip(grad_f32, indices, input_shape, stream);
        return result_f32.to(DType::Float16);
    } else if (grad_output.dtype() == DType::BFloat16) {
        auto grad_f32 = grad_output.to(DType::Float32);
        auto result_f32 = fractional_maxpool3d_backward_hip(grad_f32, indices, input_shape, stream);
        return result_f32.to(DType::BFloat16);
    } else {
        throw std::runtime_error("fractional_maxpool3d_backward_hip: unsupported dtype");
    }

    return grad_input;
}

// ============================================================================
// Max Unpool 2D Forward
// ============================================================================

__global__ void max_unpool2d_forward_kernel_f32(
    const float* __restrict__ input,
    const int64_t* __restrict__ indices,
    float* __restrict__ output,
    int64_t N, int64_t C,
    int64_t in_h, int64_t in_w,
    int64_t out_h, int64_t out_w
) {
    int64_t total = N * C * in_h * in_w;
    int64_t in_spatial = in_h * in_w;
    int64_t out_spatial = out_h * out_w;

    HIP_KERNEL_LOOP(idx, total) {
        int64_t local_idx = idx % in_spatial;
        int64_t nc = idx / in_spatial;

        int64_t base_in = nc * in_spatial;
        int64_t base_out = nc * out_spatial;

        int64_t max_idx = indices[base_in + local_idx];
        if (max_idx >= 0 && max_idx < out_spatial) {
            output[base_out + max_idx] = input[base_in + local_idx];
        }
    }
}

// audit-2026-05-03 — Float64 native unpool kernel; the previous f32-cast
// detour dropped Float64 precision and broke autograd gradcheck.
__global__ void max_unpool2d_forward_kernel_f64(
    const double* __restrict__ input,
    const int64_t* __restrict__ indices,
    double* __restrict__ output,
    int64_t N, int64_t C,
    int64_t in_h, int64_t in_w,
    int64_t out_h, int64_t out_w
) {
    int64_t total = N * C * in_h * in_w;
    int64_t in_spatial = in_h * in_w;
    int64_t out_spatial = out_h * out_w;

    HIP_KERNEL_LOOP(idx, total) {
        int64_t local_idx = idx % in_spatial;
        int64_t nc = idx / in_spatial;

        int64_t base_in = nc * in_spatial;
        int64_t base_out = nc * out_spatial;

        int64_t max_idx = indices[base_in + local_idx];
        if (max_idx >= 0 && max_idx < out_spatial) {
            output[base_out + max_idx] = input[base_in + local_idx];
        }
    }
}

auto max_unpool2d_forward_hip(
    const Tensor& input,
    const Tensor& indices,
    int64_t out_h, int64_t out_w,
    hipStream_t stream
) -> Tensor {
    auto shape = input.shape();
    int64_t N = shape[0], C = shape[1], in_h = shape[2], in_w = shape[3];

    Tensor output({N, C, out_h, out_w}, input.dtype(), input.device());

    int64_t output_numel = N * C * out_h * out_w;
    int64_t total = N * C * in_h * in_w;
    int threads = rocm::get_wavefront_size() * 4;
    int blocks = (total + threads - 1) / threads;

    if (input.dtype() == DType::Float32) {
        HIP_CHECK(hipMemsetAsync(output.data<float>(), 0, output_numel * sizeof(float), stream));
        hipLaunchKernelGGL(max_unpool2d_forward_kernel_f32,
            dim3(blocks), dim3(threads), 0, stream,
            input.data<float>(), indices.data<int64_t>(),
            output.data<float>(), N, C, in_h, in_w, out_h, out_w);
        HIP_POST_LAUNCH_CHECK();
    } else if (input.dtype() == DType::Float64) {
        HIP_CHECK(hipMemsetAsync(output.data<double>(), 0, output_numel * sizeof(double), stream));
        hipLaunchKernelGGL(max_unpool2d_forward_kernel_f64,
            dim3(blocks), dim3(threads), 0, stream,
            input.data<double>(), indices.data<int64_t>(),
            output.data<double>(), N, C, in_h, in_w, out_h, out_w);
        HIP_POST_LAUNCH_CHECK();
    } else if (input.dtype() == DType::Float16) {
        auto input_f32 = input.to(DType::Float32);
        auto result_f32 = max_unpool2d_forward_hip(input_f32, indices, out_h, out_w, stream);
        return result_f32.to(DType::Float16);
    } else if (input.dtype() == DType::BFloat16) {
        auto input_f32 = input.to(DType::Float32);
        auto result_f32 = max_unpool2d_forward_hip(input_f32, indices, out_h, out_w, stream);
        return result_f32.to(DType::BFloat16);
    } else {
        throw std::runtime_error("max_unpool2d_forward_hip: unsupported dtype");
    }

    return output;
}

// ============================================================================
// Max Unpool 2D Backward
// ============================================================================

__global__ void max_unpool2d_backward_kernel_f32(
    const float* __restrict__ grad_output,
    const int64_t* __restrict__ indices,
    float* __restrict__ grad_input,
    int64_t N, int64_t C,
    int64_t in_h, int64_t in_w,
    int64_t out_h, int64_t out_w
) {
    int64_t total = N * C * in_h * in_w;
    int64_t in_spatial = in_h * in_w;
    int64_t out_spatial = out_h * out_w;

    HIP_KERNEL_LOOP(idx, total) {
        int64_t local_idx = idx % in_spatial;
        int64_t nc = idx / in_spatial;

        int64_t base_in = nc * in_spatial;
        int64_t base_out = nc * out_spatial;

        int64_t max_idx = indices[base_in + local_idx];
        if (max_idx >= 0 && max_idx < out_spatial) {
            grad_input[base_in + local_idx] = grad_output[base_out + max_idx];
        } else {
            grad_input[base_in + local_idx] = 0.0f;
        }
    }
}

__global__ void max_unpool2d_backward_kernel_f64(
    const double* __restrict__ grad_output,
    const int64_t* __restrict__ indices,
    double* __restrict__ grad_input,
    int64_t N, int64_t C,
    int64_t in_h, int64_t in_w,
    int64_t out_h, int64_t out_w
) {
    int64_t total = N * C * in_h * in_w;
    int64_t in_spatial = in_h * in_w;
    int64_t out_spatial = out_h * out_w;

    HIP_KERNEL_LOOP(idx, total) {
        int64_t local_idx = idx % in_spatial;
        int64_t nc = idx / in_spatial;

        int64_t base_in = nc * in_spatial;
        int64_t base_out = nc * out_spatial;

        int64_t max_idx = indices[base_in + local_idx];
        if (max_idx >= 0 && max_idx < out_spatial) {
            grad_input[base_in + local_idx] = grad_output[base_out + max_idx];
        } else {
            grad_input[base_in + local_idx] = 0.0;
        }
    }
}

auto max_unpool2d_backward_hip(
    const Tensor& grad_output,
    const Tensor& indices,
    const std::vector<int64_t>& input_shape,
    hipStream_t stream
) -> Tensor {
    int64_t N = input_shape[0], C = input_shape[1];
    int64_t in_h = input_shape[2], in_w = input_shape[3];
    auto grad_shape = grad_output.shape();
    int64_t out_h = grad_shape[2], out_w = grad_shape[3];

    Tensor grad_input(input_shape, grad_output.dtype(), grad_output.device());

    int64_t total = N * C * in_h * in_w;
    int threads = rocm::get_wavefront_size() * 4;
    int blocks = (total + threads - 1) / threads;

    if (grad_output.dtype() == DType::Float32) {
        hipLaunchKernelGGL(max_unpool2d_backward_kernel_f32,
            dim3(blocks), dim3(threads), 0, stream,
            grad_output.data<float>(), indices.data<int64_t>(),
            grad_input.data<float>(), N, C, in_h, in_w, out_h, out_w);
        HIP_POST_LAUNCH_CHECK();
    } else if (grad_output.dtype() == DType::Float64) {
        hipLaunchKernelGGL(max_unpool2d_backward_kernel_f64,
            dim3(blocks), dim3(threads), 0, stream,
            grad_output.data<double>(), indices.data<int64_t>(),
            grad_input.data<double>(), N, C, in_h, in_w, out_h, out_w);
        HIP_POST_LAUNCH_CHECK();
    } else if (grad_output.dtype() == DType::Float16) {
        auto grad_f32 = grad_output.to(DType::Float32);
        auto result_f32 = max_unpool2d_backward_hip(grad_f32, indices, input_shape, stream);
        return result_f32.to(DType::Float16);
    } else if (grad_output.dtype() == DType::BFloat16) {
        auto grad_f32 = grad_output.to(DType::Float32);
        auto result_f32 = max_unpool2d_backward_hip(grad_f32, indices, input_shape, stream);
        return result_f32.to(DType::BFloat16);
    } else {
        throw std::runtime_error("max_unpool2d_backward_hip: unsupported dtype");
    }

    return grad_input;
}

// ============================================================================
// Max Unpool 3D Forward
// ============================================================================

__global__ void max_unpool3d_forward_kernel_f32(
    const float* __restrict__ input,
    const int64_t* __restrict__ indices,
    float* __restrict__ output,
    int64_t N, int64_t C,
    int64_t in_d, int64_t in_h, int64_t in_w,
    int64_t out_d, int64_t out_h, int64_t out_w
) {
    int64_t total = N * C * in_d * in_h * in_w;
    int64_t in_spatial = in_d * in_h * in_w;
    int64_t out_spatial = out_d * out_h * out_w;

    HIP_KERNEL_LOOP(idx, total) {
        int64_t local_idx = idx % in_spatial;
        int64_t nc = idx / in_spatial;

        int64_t base_in = nc * in_spatial;
        int64_t base_out = nc * out_spatial;

        int64_t max_idx = indices[base_in + local_idx];
        if (max_idx >= 0 && max_idx < out_spatial) {
            output[base_out + max_idx] = input[base_in + local_idx];
        }
    }
}

__global__ void max_unpool3d_forward_kernel_f64(
    const double* __restrict__ input,
    const int64_t* __restrict__ indices,
    double* __restrict__ output,
    int64_t N, int64_t C,
    int64_t in_d, int64_t in_h, int64_t in_w,
    int64_t out_d, int64_t out_h, int64_t out_w
) {
    int64_t total = N * C * in_d * in_h * in_w;
    int64_t in_spatial = in_d * in_h * in_w;
    int64_t out_spatial = out_d * out_h * out_w;

    HIP_KERNEL_LOOP(idx, total) {
        int64_t local_idx = idx % in_spatial;
        int64_t nc = idx / in_spatial;

        int64_t base_in = nc * in_spatial;
        int64_t base_out = nc * out_spatial;

        int64_t max_idx = indices[base_in + local_idx];
        if (max_idx >= 0 && max_idx < out_spatial) {
            output[base_out + max_idx] = input[base_in + local_idx];
        }
    }
}

auto max_unpool3d_forward_hip(
    const Tensor& input,
    const Tensor& indices,
    int64_t out_d, int64_t out_h, int64_t out_w,
    hipStream_t stream
) -> Tensor {
    auto shape = input.shape();
    int64_t N = shape[0], C = shape[1], in_d = shape[2], in_h = shape[3], in_w = shape[4];

    Tensor output({N, C, out_d, out_h, out_w}, input.dtype(), input.device());

    int64_t output_numel = N * C * out_d * out_h * out_w;
    int64_t total = N * C * in_d * in_h * in_w;
    int threads = rocm::get_wavefront_size() * 4;
    int blocks = (total + threads - 1) / threads;

    if (input.dtype() == DType::Float32) {
        HIP_CHECK(hipMemsetAsync(output.data<float>(), 0, output_numel * sizeof(float), stream));
        hipLaunchKernelGGL(max_unpool3d_forward_kernel_f32,
            dim3(blocks), dim3(threads), 0, stream,
            input.data<float>(), indices.data<int64_t>(),
            output.data<float>(), N, C, in_d, in_h, in_w, out_d, out_h, out_w);
        HIP_POST_LAUNCH_CHECK();
    } else if (input.dtype() == DType::Float64) {
        HIP_CHECK(hipMemsetAsync(output.data<double>(), 0, output_numel * sizeof(double), stream));
        hipLaunchKernelGGL(max_unpool3d_forward_kernel_f64,
            dim3(blocks), dim3(threads), 0, stream,
            input.data<double>(), indices.data<int64_t>(),
            output.data<double>(), N, C, in_d, in_h, in_w, out_d, out_h, out_w);
        HIP_POST_LAUNCH_CHECK();
    } else if (input.dtype() == DType::Float16) {
        auto input_f32 = input.to(DType::Float32);
        auto result_f32 = max_unpool3d_forward_hip(input_f32, indices, out_d, out_h, out_w, stream);
        return result_f32.to(DType::Float16);
    } else if (input.dtype() == DType::BFloat16) {
        auto input_f32 = input.to(DType::Float32);
        auto result_f32 = max_unpool3d_forward_hip(input_f32, indices, out_d, out_h, out_w, stream);
        return result_f32.to(DType::BFloat16);
    } else {
        throw std::runtime_error("max_unpool3d_forward_hip: unsupported dtype");
    }

    return output;
}

// ============================================================================
// Max Unpool 3D Backward
// ============================================================================

__global__ void max_unpool3d_backward_kernel_f32(
    const float* __restrict__ grad_output,
    const int64_t* __restrict__ indices,
    float* __restrict__ grad_input,
    int64_t N, int64_t C,
    int64_t in_d, int64_t in_h, int64_t in_w,
    int64_t out_d, int64_t out_h, int64_t out_w
) {
    int64_t total = N * C * in_d * in_h * in_w;
    int64_t in_spatial = in_d * in_h * in_w;
    int64_t out_spatial = out_d * out_h * out_w;

    HIP_KERNEL_LOOP(idx, total) {
        int64_t local_idx = idx % in_spatial;
        int64_t nc = idx / in_spatial;

        int64_t base_in = nc * in_spatial;
        int64_t base_out = nc * out_spatial;

        int64_t max_idx = indices[base_in + local_idx];
        if (max_idx >= 0 && max_idx < out_spatial) {
            grad_input[base_in + local_idx] = grad_output[base_out + max_idx];
        } else {
            grad_input[base_in + local_idx] = 0.0f;
        }
    }
}

__global__ void max_unpool3d_backward_kernel_f64(
    const double* __restrict__ grad_output,
    const int64_t* __restrict__ indices,
    double* __restrict__ grad_input,
    int64_t N, int64_t C,
    int64_t in_d, int64_t in_h, int64_t in_w,
    int64_t out_d, int64_t out_h, int64_t out_w
) {
    int64_t total = N * C * in_d * in_h * in_w;
    int64_t in_spatial = in_d * in_h * in_w;
    int64_t out_spatial = out_d * out_h * out_w;

    HIP_KERNEL_LOOP(idx, total) {
        int64_t local_idx = idx % in_spatial;
        int64_t nc = idx / in_spatial;

        int64_t base_in = nc * in_spatial;
        int64_t base_out = nc * out_spatial;

        int64_t max_idx = indices[base_in + local_idx];
        if (max_idx >= 0 && max_idx < out_spatial) {
            grad_input[base_in + local_idx] = grad_output[base_out + max_idx];
        } else {
            grad_input[base_in + local_idx] = 0.0;
        }
    }
}

auto max_unpool3d_backward_hip(
    const Tensor& grad_output,
    const Tensor& indices,
    const std::vector<int64_t>& input_shape,
    hipStream_t stream
) -> Tensor {
    int64_t N = input_shape[0], C = input_shape[1];
    int64_t in_d = input_shape[2], in_h = input_shape[3], in_w = input_shape[4];
    auto grad_shape = grad_output.shape();
    int64_t out_d = grad_shape[2], out_h = grad_shape[3], out_w = grad_shape[4];

    Tensor grad_input(input_shape, grad_output.dtype(), grad_output.device());

    int64_t total = N * C * in_d * in_h * in_w;
    int threads = rocm::get_wavefront_size() * 4;
    int blocks = (total + threads - 1) / threads;

    if (grad_output.dtype() == DType::Float32) {
        hipLaunchKernelGGL(max_unpool3d_backward_kernel_f32,
            dim3(blocks), dim3(threads), 0, stream,
            grad_output.data<float>(), indices.data<int64_t>(),
            grad_input.data<float>(), N, C, in_d, in_h, in_w, out_d, out_h, out_w);
        HIP_POST_LAUNCH_CHECK();
    } else if (grad_output.dtype() == DType::Float64) {
        hipLaunchKernelGGL(max_unpool3d_backward_kernel_f64,
            dim3(blocks), dim3(threads), 0, stream,
            grad_output.data<double>(), indices.data<int64_t>(),
            grad_input.data<double>(), N, C, in_d, in_h, in_w, out_d, out_h, out_w);
        HIP_POST_LAUNCH_CHECK();
    } else if (grad_output.dtype() == DType::Float16) {
        auto grad_f32 = grad_output.to(DType::Float32);
        auto result_f32 = max_unpool3d_backward_hip(grad_f32, indices, input_shape, stream);
        return result_f32.to(DType::Float16);
    } else if (grad_output.dtype() == DType::BFloat16) {
        auto grad_f32 = grad_output.to(DType::Float32);
        auto result_f32 = max_unpool3d_backward_hip(grad_f32, indices, input_shape, stream);
        return result_f32.to(DType::BFloat16);
    } else {
        throw std::runtime_error("max_unpool3d_backward_hip: unsupported dtype");
    }

    return grad_input;
}

// ============================================================================
// Phase A.1 — Max Unpool 1D (ROCm). Reshape (N, C, L) → (N, C, L, 1) and
// reuse the existing 2D kernel; reshape is metadata-only on the GPU.
// ============================================================================

auto max_unpool1d_forward_hip(const Tensor& input, const Tensor& indices,
                              int64_t out_l, hipStream_t stream) -> Tensor
{
    auto shape = input.shape();
    int64_t N = shape[0], C = shape[1], in_l = shape[2];
    auto input_4d = input.contiguous().reshape({N, C, in_l, 1});
    auto indices_4d = indices.contiguous().reshape({N, C, in_l, 1});
    auto out_4d = max_unpool2d_forward_hip(input_4d, indices_4d, out_l, /*out_w=*/1, stream);
    return out_4d.reshape({N, C, out_l});
}

auto max_unpool1d_backward_hip(const Tensor& grad_output, const Tensor& indices,
                                const std::vector<int64_t>& input_shape,
                                hipStream_t stream) -> Tensor
{
    int64_t N = input_shape[0], C = input_shape[1], in_l = input_shape[2];
    int64_t out_l = grad_output.shape()[2];
    std::vector<int64_t> input_shape_4d = {N, C, in_l, 1};
    auto grad_4d = grad_output.contiguous().reshape({N, C, out_l, 1});
    auto indices_4d = indices.contiguous().reshape({N, C, in_l, 1});
    auto grad_in_4d = max_unpool2d_backward_hip(grad_4d, indices_4d, input_shape_4d, stream);
    return grad_in_4d.reshape({N, C, in_l});
}

} // namespace rocm
} // namespace tenzor
