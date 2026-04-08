/**
 * @file stft.hip.cpp
 * @brief HIP/ROCm port of STFT and ISTFT.
 *
 * Native ROCm STFT/ISTFT implementation. Forward: frame+window kernel +
 * batched rFFT (rocFFT) + transpose + reshape. Inverse: transpose/reshape +
 * batched irFFT + output-centric overlap-add + normalize. Added to build in
 * Phase 7 (post-Phase 4.3 fix for the earlier shape bug: the previous version
 * called rocm_fft_kernel for both onesided and non-onesided branches, but
 * onesided must use rocm_rfft_kernel which returns n/2+1 freq bins).
 *
 * Mirrors src/backends/cuda/kernels/advanced.cu (stft_cuda_kernel and
 * istft_cuda_kernel) — uses the existing rocm fft kernels for the actual
 * batched FFT and overlap-add via custom kernels.
 */

#include "tenzor/core/tensor.hpp"
#include "tenzor/core/dtype.hpp"
#include "tenzor/ops/transform.hpp"
#include "tenzor/ops/indexing.hpp"
#include "tenzor/ops/math.hpp"
#include <hip/hip_runtime.h>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>
#include <optional>

namespace tenzor {
namespace rocm {

#ifndef HIP_CHECK
#define HIP_CHECK(call) do { \
    hipError_t err = call; \
    if (err != hipSuccess) { \
        throw std::runtime_error(std::string("HIP error: ") + hipGetErrorString(err)); \
    } \
} while(0)
#endif

// Forward decls for the FFT kernels (defined in fft.hip.cpp)
auto rocm_rfft_kernel(const Tensor& input, int64_t dim, int64_t n,
                      const std::string& norm, hipStream_t stream) -> Tensor;
auto rocm_fft_kernel(const Tensor& input, int64_t dim, int64_t n,
                     const std::string& norm, hipStream_t stream) -> Tensor;
auto rocm_irfft_kernel(const Tensor& input, int64_t dim, int64_t n,
                       const std::string& norm, hipStream_t stream) -> Tensor;
auto rocm_ifft_kernel(const Tensor& input, int64_t dim, int64_t n,
                      const std::string& norm, hipStream_t stream) -> Tensor;

__global__ void stft_frame_window_kernel_hip(
    const float* __restrict__ signal,
    const float* __restrict__ window,
    float* __restrict__ framed,
    int64_t batch_size, int64_t signal_length, int64_t num_frames,
    int64_t n_fft, int64_t hop_length, int64_t pad, int64_t total
) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int64_t i = idx % n_fft;
    int64_t f = (idx / n_fft) % num_frames;
    int64_t b = idx / (n_fft * num_frames);

    int64_t sig_idx = f * hop_length + i - pad;

    float val;
    if (signal_length <= 1) {
        val = (signal_length == 1) ? signal[b * signal_length] : 0.0f;
    } else if (sig_idx < 0) {
        int64_t r = -sig_idx;
        if (r >= signal_length) r = signal_length - 1;
        val = signal[b * signal_length + r];
    } else if (sig_idx >= signal_length) {
        int64_t r = 2 * signal_length - 2 - sig_idx;
        if (r < 0) r = 0;
        val = signal[b * signal_length + r];
    } else {
        val = signal[b * signal_length + sig_idx];
    }
    framed[idx] = val * window[i];
}

auto stft_kernel(const Tensor& input, int64_t n_fft,
                 int64_t hop_length, int64_t win_length,
                 const Tensor& window, bool center,
                 bool normalized, bool onesided,
                 hipStream_t stream) -> Tensor {
    if (n_fft <= 0) throw std::runtime_error("stft: n_fft must be > 0");
    if (hop_length <= 0) hop_length = n_fft / 4;
    if (win_length <= 0) win_length = n_fft;
    if (win_length > n_fft) throw std::runtime_error("stft: win_length must be <= n_fft");

    auto input_f32 = (input.dtype() != DType::Float32) ? input.to(DType::Float32) : input.contiguous();

    auto in_shape = input_f32.shape();
    int64_t ndim = static_cast<int64_t>(in_shape.size());
    if (ndim < 1) throw std::runtime_error("stft: input must have ≥ 1 dim");

    int64_t signal_length = in_shape[ndim - 1];
    int64_t batch_size = 1;
    for (int64_t d = 0; d < ndim - 1; ++d) batch_size *= in_shape[d];

    int64_t pad = center ? (n_fft / 2) : 0;
    int64_t padded_length = signal_length + 2 * pad;
    int64_t num_frames = (padded_length - n_fft) / hop_length + 1;
    if (num_frames <= 0) throw std::runtime_error("stft: signal too short");

    int64_t freq_bins = onesided ? (n_fft / 2 + 1) : n_fft;

    Tensor window_dev({n_fft}, DType::Float32, input.device());
    HIP_CHECK(hipMemsetAsync(window_dev.data_ptr(), 0, n_fft * sizeof(float), stream));
    int64_t win_offset = (n_fft - win_length) / 2;
    if (window.is_valid() && window.numel() > 0) {
        Tensor win_f32 = (window.dtype() != DType::Float32) ? window.to(DType::Float32)
                                                              : window.contiguous();
        HIP_CHECK(hipMemcpyAsync(static_cast<float*>(window_dev.data_ptr()) + win_offset,
                                  win_f32.data<float>(),
                                  win_length * sizeof(float),
                                  hipMemcpyDeviceToDevice, stream));
    } else {
        std::vector<float> host_ones(win_length, 1.0f);
        HIP_CHECK(hipMemcpyAsync(static_cast<float*>(window_dev.data_ptr()) + win_offset,
                                  host_ones.data(),
                                  win_length * sizeof(float),
                                  hipMemcpyHostToDevice, stream));
    }

    Tensor framed({batch_size, num_frames, n_fft}, DType::Float32, input.device());
    int64_t total = batch_size * num_frames * n_fft;
    int threads = 256;
    int blocks = static_cast<int>((total + threads - 1) / threads);
    hipLaunchKernelGGL(stft_frame_window_kernel_hip,
        dim3(blocks), dim3(threads), 0, stream,
        input_f32.data<float>(), window_dev.data<float>(), framed.data<float>(),
        batch_size, signal_length, num_frames, n_fft, hop_length, pad, total);
    HIP_CHECK(hipGetLastError());

    Tensor fft_out;
    if (onesided) {
        // Real-to-complex batched FFT along the last dim → (B, num_frames, n_fft/2+1)
        fft_out = rocm_rfft_kernel(framed, /*dim=*/2, n_fft, normalized ? "ortho" : "backward", stream);
    } else {
        // Full complex FFT → (B, num_frames, n_fft). Input is real, promoted by rocm_fft_kernel.
        fft_out = rocm_fft_kernel(framed, /*dim=*/2, n_fft, normalized ? "ortho" : "backward", stream);
    }

    Tensor transposed = tenzor::transpose(fft_out, -1, -2);

    std::vector<int64_t> out_shape;
    for (int64_t d = 0; d < ndim - 1; ++d) out_shape.push_back(in_shape[d]);
    out_shape.push_back(freq_bins);
    out_shape.push_back(num_frames);
    return tenzor::reshape(transposed, out_shape);
}

__global__ void istft_overlap_add_kernel_hip(
    const float* __restrict__ time_frames,
    const float* __restrict__ window,
    float* __restrict__ output,
    float* __restrict__ window_sum,
    int64_t batch_size, int64_t num_frames, int64_t n_fft,
    int64_t hop_length, int64_t expected_length
) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = batch_size * num_frames * n_fft;
    if (idx >= total) return;

    int64_t i = idx % n_fft;
    int64_t f = (idx / n_fft) % num_frames;
    int64_t b = idx / (n_fft * num_frames);

    int64_t out_pos = f * hop_length + i;
    if (out_pos < 0 || out_pos >= expected_length) return;

    float w = window[i];
    float val = time_frames[((b * num_frames) + f) * n_fft + i] * w;
    float w2 = w * w;

    atomicAdd(&output[b * expected_length + out_pos], val);
    atomicAdd(&window_sum[b * expected_length + out_pos], w2);
}

__global__ void istft_normalize_kernel_hip(
    float* __restrict__ output,
    const float* __restrict__ window_sum,
    int64_t total
) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    float ws = window_sum[idx];
    if (ws > 1e-11f) output[idx] /= ws;
}

auto istft_kernel(const Tensor& input, int64_t n_fft,
                  int64_t hop_length, int64_t win_length,
                  const Tensor& window, bool center,
                  bool /*normalized*/, bool onesided,
                  int64_t length, hipStream_t stream) -> Tensor {
    if (n_fft <= 0) throw std::runtime_error("istft: n_fft must be > 0");
    if (hop_length <= 0) hop_length = n_fft / 4;
    if (win_length <= 0) win_length = n_fft;

    auto in_shape = input.shape();
    int64_t ndim = static_cast<int64_t>(in_shape.size());
    if (ndim < 2) throw std::runtime_error("istft: input must have ≥ 2 dims");

    int64_t freq_bins = in_shape[ndim - 2];
    int64_t num_frames = in_shape[ndim - 1];
    int64_t batch_size = 1;
    for (int64_t d = 0; d < ndim - 2; ++d) batch_size *= in_shape[d];

    int64_t expected_length = n_fft + (num_frames - 1) * hop_length;

    Tensor input_c64 = (input.dtype() != DType::Complex64) ? input.to(DType::Complex64)
                                                            : input.contiguous();
    Tensor reshaped = tenzor::reshape(input_c64, {batch_size, freq_bins, num_frames});
    Tensor transposed = tenzor::transpose(reshaped, -1, -2);
    Tensor transposed_contig = transposed.contiguous();

    Tensor time_frames;
    if (onesided) {
        time_frames = rocm_irfft_kernel(transposed_contig, /*dim=*/2, n_fft, "backward", stream);
    } else {
        time_frames = rocm_ifft_kernel(transposed_contig, /*dim=*/2, n_fft, "backward", stream);
        time_frames = tenzor::real(time_frames);
    }

    Tensor window_dev({n_fft}, DType::Float32, input.device());
    HIP_CHECK(hipMemsetAsync(window_dev.data_ptr(), 0, n_fft * sizeof(float), stream));
    int64_t win_offset = (n_fft - win_length) / 2;
    if (window.is_valid() && window.numel() > 0) {
        Tensor win_f32 = (window.dtype() != DType::Float32) ? window.to(DType::Float32)
                                                              : window.contiguous();
        HIP_CHECK(hipMemcpyAsync(static_cast<float*>(window_dev.data_ptr()) + win_offset,
                                  win_f32.data<float>(),
                                  win_length * sizeof(float),
                                  hipMemcpyDeviceToDevice, stream));
    } else {
        std::vector<float> host_ones(win_length, 1.0f);
        HIP_CHECK(hipMemcpyAsync(static_cast<float*>(window_dev.data_ptr()) + win_offset,
                                  host_ones.data(),
                                  win_length * sizeof(float),
                                  hipMemcpyHostToDevice, stream));
    }

    Tensor output_buf({batch_size, expected_length}, DType::Float32, input.device());
    Tensor wsum_buf({batch_size, expected_length}, DType::Float32, input.device());
    HIP_CHECK(hipMemsetAsync(output_buf.data_ptr(), 0, batch_size * expected_length * sizeof(float), stream));
    HIP_CHECK(hipMemsetAsync(wsum_buf.data_ptr(),  0, batch_size * expected_length * sizeof(float), stream));

    {
        int64_t total = batch_size * num_frames * n_fft;
        int threads = 256;
        int blocks = static_cast<int>((total + threads - 1) / threads);
        hipLaunchKernelGGL(istft_overlap_add_kernel_hip,
            dim3(blocks), dim3(threads), 0, stream,
            time_frames.data<float>(), window_dev.data<float>(),
            output_buf.data<float>(), wsum_buf.data<float>(),
            batch_size, num_frames, n_fft, hop_length, expected_length);
        HIP_CHECK(hipGetLastError());
    }

    {
        int64_t total = batch_size * expected_length;
        int threads = 256;
        int blocks = static_cast<int>((total + threads - 1) / threads);
        hipLaunchKernelGGL(istft_normalize_kernel_hip,
            dim3(blocks), dim3(threads), 0, stream,
            output_buf.data<float>(), wsum_buf.data<float>(), total);
        HIP_CHECK(hipGetLastError());
    }

    int64_t pad = center ? (n_fft / 2) : 0;
    int64_t out_length = expected_length - 2 * pad;
    if (length >= 0) out_length = length;

    Tensor result;
    if (pad == 0 && length < 0) {
        result = output_buf;
    } else {
        Tensor sliced = tenzor::slice(output_buf, 1, pad, pad + out_length);
        result = sliced.contiguous();
    }

    std::vector<int64_t> out_shape;
    for (int64_t d = 0; d < ndim - 2; ++d) out_shape.push_back(in_shape[d]);
    out_shape.push_back(out_length);
    return tenzor::reshape(result, out_shape);
}

}  // namespace rocm
}  // namespace tenzor
