/**
 * @file stft.cpp
 * @brief OneAPI/SYCL implementation of STFT and ISTFT.
 *
 * Mirrors src/backends/cuda/kernels/advanced.cu (stft_cuda_kernel /
 * istft_cuda_kernel). Replaces the previous CPU-roundtrip fallback.
 */

#include "tenzor/core/tensor.hpp"
#include "tenzor/core/dtype.hpp"
#include "tenzor/ops/transform.hpp"
#include "tenzor/ops/indexing.hpp"
#include "tenzor/ops/math.hpp"
#include "oneapi_kernel_utils.hpp"
#include <sycl/sycl.hpp>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace tenzor {
namespace oneapi {

namespace {

struct STFTFrameWindowKernel {};
struct ISTFTOverlapAddKernel {};
struct ISTFTNormalizeKernel {};

}  // namespace

// Forward decls (defined in fft.cpp)
auto rfft_kernel(const Tensor& input, int64_t dim, int64_t n,
                 const std::string& norm, sycl::queue& queue) -> Tensor;
auto fft_kernel(const Tensor& input, int64_t dim, int64_t n,
                const std::string& norm, sycl::queue& queue) -> Tensor;
auto irfft_kernel(const Tensor& input, int64_t dim, int64_t n,
                  const std::string& norm, sycl::queue& queue) -> Tensor;
auto ifft_kernel(const Tensor& input, int64_t dim, int64_t n,
                 const std::string& norm, sycl::queue& queue) -> Tensor;

auto stft_kernel(const Tensor& input, int64_t n_fft,
                 int64_t hop_length, int64_t win_length,
                 const Tensor& window, bool center,
                 bool normalized, bool onesided,
                 sycl::queue& queue) -> Tensor {
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

    // Build window on-device
    Tensor window_dev(std::vector<int64_t>{n_fft}, DType::Float32, input.device());
    queue.memset(window_dev.data_ptr(), 0, n_fft * sizeof(float)).wait();
    int64_t win_offset = (n_fft - win_length) / 2;
    if (window.is_valid() && window.numel() > 0) {
        Tensor win_f32 = (window.dtype() != DType::Float32) ? window.to(DType::Float32)
                                                              : window.contiguous();
        queue.memcpy(get_data_ptr<float>(window_dev) + win_offset,
                     get_data_ptr<const float>(win_f32),
                     win_length * sizeof(float)).wait();
    } else {
        std::vector<float> host_ones(win_length, 1.0f);
        queue.memcpy(get_data_ptr<float>(window_dev) + win_offset,
                     host_ones.data(),
                     win_length * sizeof(float)).wait();
    }

    // Frame+window kernel: produces (batch_size, num_frames, n_fft)
    Tensor framed(std::vector<int64_t>{batch_size, num_frames, n_fft},
                  DType::Float32, input.device());
    int64_t total = batch_size * num_frames * n_fft;
    if (total > 0) {
        const float* sig_ptr = get_data_ptr<const float>(input_f32);
        const float* win_ptr = get_data_ptr<const float>(window_dev);
        float* out_ptr = get_data_ptr<float>(framed);
        const int64_t sl = signal_length;
        const int64_t nf = n_fft;
        const int64_t nfr = num_frames;
        const int64_t hop = hop_length;
        const int64_t pd = pad;

        queue.parallel_for<STFTFrameWindowKernel>(sycl::range<1>(total),
            [=](sycl::id<1> idx_) {
                int64_t idx = static_cast<int64_t>(idx_);
                int64_t i = idx % nf;
                int64_t f = (idx / nf) % nfr;
                int64_t b = idx / (nf * nfr);
                int64_t sig_idx = f * hop + i - pd;

                float val;
                if (sl <= 1) {
                    val = (sl == 1) ? sig_ptr[b * sl] : 0.0f;
                } else if (sig_idx < 0) {
                    int64_t r = -sig_idx;
                    if (r >= sl) r = sl - 1;
                    val = sig_ptr[b * sl + r];
                } else if (sig_idx >= sl) {
                    int64_t r = 2 * sl - 2 - sig_idx;
                    if (r < 0) r = 0;
                    val = sig_ptr[b * sl + r];
                } else {
                    val = sig_ptr[b * sl + sig_idx];
                }
                out_ptr[idx] = val * win_ptr[i];
            });
        queue.wait_and_throw();
    }

    // OneAPI rfft/fft returns Float32 with a trailing pair dim of 2:
    // shape (B, num_frames, freq_bins, 2). We re-wrap this as Complex64 with
    // shape (B, num_frames, freq_bins) since the byte layout is identical.
    Tensor fft_out_f32;
    if (onesided) {
        fft_out_f32 = rfft_kernel(framed, /*dim=*/2, n_fft,
                                  normalized ? "ortho" : "backward", queue);
    } else {
        fft_out_f32 = fft_kernel(framed, /*dim=*/2, n_fft,
                                 normalized ? "ortho" : "backward", queue);
    }

    // Build a Complex64 tensor of shape (B, num_frames, freq_bins) and copy bytes
    Tensor fft_out(std::vector<int64_t>{batch_size, num_frames, freq_bins},
                   DType::Complex64, input.device());
    queue.memcpy(get_data_ptr<float>(fft_out),
                 get_data_ptr<const float>(fft_out_f32),
                 batch_size * num_frames * freq_bins * 2 * sizeof(float)).wait();

    // Transpose last two dims: (B, freq_bins, num_frames) Complex64
    Tensor transposed = tenzor::transpose(fft_out, -1, -2);

    std::vector<int64_t> out_shape;
    for (int64_t d = 0; d < ndim - 1; ++d) out_shape.push_back(in_shape[d]);
    out_shape.push_back(freq_bins);
    out_shape.push_back(num_frames);
    return tenzor::reshape(transposed, out_shape);
}

auto istft_kernel(const Tensor& input, int64_t n_fft,
                  int64_t hop_length, int64_t win_length,
                  const Tensor& window, bool center,
                  bool /*normalized*/, bool onesided,
                  int64_t length, sycl::queue& queue) -> Tensor {
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
    // transposed_contig: (B, num_frames, freq_bins) Complex64

    // irfft / ifft now require Complex64/Complex128 input directly; the old
    // Float32-with-trailing-2 layout contract was dropped. Pass the complex
    // tensor straight through.
    Tensor time_frames;
    if (onesided) {
        time_frames = irfft_kernel(transposed_contig, /*dim=*/2, n_fft, "backward", queue);
    } else {
        Tensor ifft_out = ifft_kernel(transposed_contig, /*dim=*/2, n_fft, "backward", queue);
        // ifft returns Complex64; take the real part.
        time_frames = tenzor::real(ifft_out);
    }

    // Build window
    Tensor window_dev(std::vector<int64_t>{n_fft}, DType::Float32, input.device());
    queue.memset(window_dev.data_ptr(), 0, n_fft * sizeof(float)).wait();
    int64_t win_offset = (n_fft - win_length) / 2;
    if (window.is_valid() && window.numel() > 0) {
        Tensor win_f32 = (window.dtype() != DType::Float32) ? window.to(DType::Float32)
                                                              : window.contiguous();
        queue.memcpy(get_data_ptr<float>(window_dev) + win_offset,
                     get_data_ptr<const float>(win_f32),
                     win_length * sizeof(float)).wait();
    } else {
        std::vector<float> host_ones(win_length, 1.0f);
        queue.memcpy(get_data_ptr<float>(window_dev) + win_offset,
                     host_ones.data(),
                     win_length * sizeof(float)).wait();
    }

    Tensor output_buf(std::vector<int64_t>{batch_size, expected_length},
                      DType::Float32, input.device());
    Tensor wsum_buf(std::vector<int64_t>{batch_size, expected_length},
                    DType::Float32, input.device());
    queue.memset(output_buf.data_ptr(), 0,
                 batch_size * expected_length * sizeof(float)).wait();
    queue.memset(wsum_buf.data_ptr(), 0,
                 batch_size * expected_length * sizeof(float)).wait();

    // Overlap-add via atomic_ref
    {
        const float* tf_ptr = get_data_ptr<const float>(time_frames);
        const float* w_ptr = get_data_ptr<const float>(window_dev);
        float* out_ptr = get_data_ptr<float>(output_buf);
        float* ws_ptr = get_data_ptr<float>(wsum_buf);

        const int64_t bs = batch_size;
        const int64_t nfr = num_frames;
        const int64_t nf = n_fft;
        const int64_t hop = hop_length;
        const int64_t el = expected_length;
        const int64_t total = bs * nfr * nf;

        queue.parallel_for<ISTFTOverlapAddKernel>(sycl::range<1>(total),
            [=](sycl::id<1> idx_) {
                int64_t idx = static_cast<int64_t>(idx_);
                int64_t i = idx % nf;
                int64_t f = (idx / nf) % nfr;
                int64_t b = idx / (nf * nfr);
                int64_t out_pos = f * hop + i;
                if (out_pos < 0 || out_pos >= el) return;

                float w = w_ptr[i];
                float val = tf_ptr[((b * nfr) + f) * nf + i] * w;
                float w2 = w * w;

                sycl::atomic_ref<float, sycl::memory_order::relaxed,
                                  sycl::memory_scope::device,
                                  sycl::access::address_space::global_space>
                    out_atom(out_ptr[b * el + out_pos]);
                sycl::atomic_ref<float, sycl::memory_order::relaxed,
                                  sycl::memory_scope::device,
                                  sycl::access::address_space::global_space>
                    ws_atom(ws_ptr[b * el + out_pos]);
                out_atom.fetch_add(val);
                ws_atom.fetch_add(w2);
            });
        queue.wait_and_throw();
    }

    // Normalize
    {
        float* out_ptr = get_data_ptr<float>(output_buf);
        const float* ws_ptr = get_data_ptr<const float>(wsum_buf);
        const int64_t total = batch_size * expected_length;

        queue.parallel_for<ISTFTNormalizeKernel>(sycl::range<1>(total),
            [=](sycl::id<1> idx_) {
                int64_t idx = static_cast<int64_t>(idx_);
                float ws = ws_ptr[idx];
                if (ws > 1e-11f) out_ptr[idx] /= ws;
            });
        queue.wait_and_throw();
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

}  // namespace oneapi
}  // namespace tenzor
