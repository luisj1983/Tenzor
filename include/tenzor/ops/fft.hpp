/**
 * @file fft.hpp
 * @brief Fast Fourier Transform operations
 *
 * Provides 1D, 2D, and N-D FFT operations with forward and inverse transforms.
 * Supports both complex-to-complex and real-to-complex (and inverse) variants.
 */

#pragma once

#include "../core/tensor.hpp"
#include <optional>
#include <string>
#include <vector>

namespace tenzor {
namespace fft {

/**
 * @brief 1-D complex-to-complex FFT.
 *
 * @param input Complex input tensor
 * @param n Signal length (default: input.shape[dim])
 * @param dim Dimension to transform (default: -1)
 * @param norm Normalization mode: "backward" (default), "forward", "ortho"
 * @return Complex output tensor
 */
auto fft(const Tensor& input,
         std::optional<int64_t> n = std::nullopt,
         int64_t dim = -1,
         const std::string& norm = "backward") -> Tensor;

/**
 * @brief 1-D inverse complex-to-complex FFT.
 */
auto ifft(const Tensor& input,
          std::optional<int64_t> n = std::nullopt,
          int64_t dim = -1,
          const std::string& norm = "backward") -> Tensor;

/**
 * @brief 1-D real-to-complex FFT.
 *
 * Input must be real. Output has shape [..., n/2+1] along the transform dimension.
 */
auto rfft(const Tensor& input,
          std::optional<int64_t> n = std::nullopt,
          int64_t dim = -1,
          const std::string& norm = "backward") -> Tensor;

/**
 * @brief 1-D complex-to-real inverse FFT.
 *
 * Input is the half-spectrum from rfft. Output is real with length n.
 * @param n Output signal length (required for unambiguous reconstruction)
 */
auto irfft(const Tensor& input,
           std::optional<int64_t> n = std::nullopt,
           int64_t dim = -1,
           const std::string& norm = "backward") -> Tensor;

/**
 * @brief 2-D complex-to-complex FFT.
 */
auto fft2(const Tensor& input,
          std::optional<std::vector<int64_t>> s = std::nullopt,
          std::vector<int64_t> dim = {-2, -1},
          const std::string& norm = "backward") -> Tensor;

/**
 * @brief 2-D inverse complex-to-complex FFT.
 */
auto ifft2(const Tensor& input,
           std::optional<std::vector<int64_t>> s = std::nullopt,
           std::vector<int64_t> dim = {-2, -1},
           const std::string& norm = "backward") -> Tensor;

/**
 * @brief 2-D real-to-complex FFT.
 *
 * Applies rfft along the last dimension, then fft along the second-to-last.
 * Output has shape [..., s[-2], s[-1]/2+1] along the transform dimensions.
 */
auto rfft2(const Tensor& input,
           std::optional<std::vector<int64_t>> s = std::nullopt,
           std::vector<int64_t> dim = {-2, -1},
           const std::string& norm = "backward") -> Tensor;

/**
 * @brief 2-D complex-to-real inverse FFT.
 *
 * Inverse of rfft2. Input is the half-spectrum. Output is real.
 */
auto irfft2(const Tensor& input,
            std::optional<std::vector<int64_t>> s = std::nullopt,
            std::vector<int64_t> dim = {-2, -1},
            const std::string& norm = "backward") -> Tensor;

/**
 * @brief N-D real-to-complex FFT.
 *
 * Applies rfft along the last dimension, then fft along the remaining dimensions.
 */
auto rfftn(const Tensor& input,
           std::optional<std::vector<int64_t>> s = std::nullopt,
           std::optional<std::vector<int64_t>> dim = std::nullopt,
           const std::string& norm = "backward") -> Tensor;

/**
 * @brief N-D complex-to-real inverse FFT.
 *
 * Inverse of rfftn. Input is the half-spectrum. Output is real.
 */
auto irfftn(const Tensor& input,
            std::optional<std::vector<int64_t>> s = std::nullopt,
            std::optional<std::vector<int64_t>> dim = std::nullopt,
            const std::string& norm = "backward") -> Tensor;

/**
 * @brief N-D complex-to-complex FFT.
 */
auto fftn(const Tensor& input,
          std::optional<std::vector<int64_t>> s = std::nullopt,
          std::optional<std::vector<int64_t>> dim = std::nullopt,
          const std::string& norm = "backward") -> Tensor;

/**
 * @brief N-D inverse complex-to-complex FFT.
 */
auto ifftn(const Tensor& input,
           std::optional<std::vector<int64_t>> s = std::nullopt,
           std::optional<std::vector<int64_t>> dim = std::nullopt,
           const std::string& norm = "backward") -> Tensor;

/**
 * @brief Short-time Fourier Transform (STFT).
 *
 * Computes the STFT of a signal using a sliding window.
 *
 * @param input Input signal tensor (..., signal_length)
 * @param n_fft FFT window size
 * @param hop_length Hop between windows (default: n_fft / 4)
 * @param win_length Window length (default: n_fft)
 * @param window Optional window tensor of length win_length
 * @param center Whether to pad input on both sides (default: true)
 * @param normalized Whether to normalize the STFT (default: false)
 * @param onesided Whether to return half of the FFT (default: true for real input)
 * @return Complex tensor (..., n_fft/2+1, num_frames) if onesided, else (..., n_fft, num_frames)
 */
auto stft(const Tensor& input,
          int64_t n_fft,
          int64_t hop_length = -1,
          int64_t win_length = -1,
          const Tensor& window = Tensor{},
          bool center = true,
          bool normalized = false,
          bool onesided = true) -> Tensor;

/**
 * @brief Inverse Short-time Fourier Transform (ISTFT).
 *
 * Reconstructs signal from STFT output via overlap-add.
 *
 * @param input Complex STFT tensor (..., freq_bins, num_frames)
 * @param n_fft FFT window size
 * @param hop_length Hop between windows (default: n_fft / 4)
 * @param win_length Window length (default: n_fft)
 * @param window Optional window tensor
 * @param center Whether input was center-padded (default: true)
 * @param normalized Whether STFT was normalized (default: false)
 * @param onesided Whether input is onesided (default: true)
 * @param length Optional output signal length
 * @return Real signal tensor
 */
auto istft(const Tensor& input,
           int64_t n_fft,
           int64_t hop_length = -1,
           int64_t win_length = -1,
           const Tensor& window = Tensor{},
           bool center = true,
           bool normalized = false,
           bool onesided = true,
           std::optional<int64_t> length = std::nullopt) -> Tensor;

/**
 * @brief Griffin-Lim phase reconstruction algorithm.
 *
 * Reconstructs a time-domain signal from a magnitude spectrogram |S| by
 * iteratively estimating the phase via STFT/ISTFT round-trips.
 *
 * The implementation is the canonical Fast Griffin-Lim algorithm
 * (Perraudin et al., 2013):
 *   - Initialize phase uniformly in [-pi, pi) (zeros if n_iter==0).
 *   - Maintain a complex spectrogram target S_k = |S| * exp(i * phase).
 *   - At each iteration: x_hat = ISTFT(S_k); S_hat = STFT(x_hat);
 *     t_n = S_hat - momentum * t_{n-1}; S_k = |S| * exp(i * angle(t_n)).
 *   - momentum = 0 recovers the original Griffin-Lim (1984).
 *   - momentum in (0, 1) accelerates convergence.
 *
 * The magnitude tensor may be Float32 or Float64. Internally everything is
 * widened to Float32 / Complex64 (STFT's working dtype) and narrowed back on
 * return.
 *
 * @param magnitude Magnitude spectrogram (real, non-negative).
 *                  Shape: (..., freq_bins, num_frames).
 * @param n_fft FFT window size.
 * @param hop_length Hop between windows (default: n_fft / 4 when < 0).
 * @param win_length Window length (default: n_fft when < 0).
 * @param window Optional window tensor.
 * @param n_iter Number of iterations (default 32). n_iter==0 returns the
 *               zero-phase ISTFT of |S|.
 * @param momentum Fast Griffin-Lim momentum in [0, 1) (default 0.99).
 *                 0.0 recovers original Griffin-Lim.
 * @return Reconstructed time-domain signal in the dtype of `magnitude`.
 */
auto griffin_lim(const Tensor& magnitude,
                int64_t n_fft,
                int64_t hop_length = -1,
                int64_t win_length = -1,
                const Tensor& window = Tensor{},
                int64_t n_iter = 32,
                double momentum = 0.99) -> Tensor;

/**
 * @brief Shift the zero-frequency component to the center of the spectrum.
 *
 * Applies `roll` along each of `dims` (default: all dims) by `size // 2`.
 * Matches numpy.fft.fftshift / torch.fft.fftshift.
 */
auto fftshift(const Tensor& input, std::vector<int64_t> dims = {}) -> Tensor;

/**
 * @brief Inverse of fftshift — undoes the circular shift.
 *
 * For an even-length axis this is identical to fftshift. For odd-length
 * axes it differs by one, which is why a separate function exists.
 */
auto ifftshift(const Tensor& input, std::vector<int64_t> dims = {}) -> Tensor;

/**
 * @brief Hermitian FFT: real output from Hermitian-symmetric complex input.
 *
 * `hfft(x)` = `irfft(conj(x))`. Matches torch.fft.hfft.
 */
auto hfft(const Tensor& input,
          std::optional<int64_t> n = std::nullopt,
          int64_t dim = -1,
          const std::string& norm = "backward") -> Tensor;

/**
 * @brief Inverse Hermitian FFT: Hermitian-symmetric output from real input.
 *
 * `ihfft(x)` = `conj(rfft(x))`. Matches torch.fft.ihfft.
 */
auto ihfft(const Tensor& input,
           std::optional<int64_t> n = std::nullopt,
           int64_t dim = -1,
           const std::string& norm = "backward") -> Tensor;

/// Discrete Fourier Transform sample frequencies
auto fftfreq(int64_t n, double d = 1.0, DType dtype = DType::Float32,
             Device device = Device::cpu()) -> Tensor;

/// Real FFT sample frequencies (only non-negative frequencies)
auto rfftfreq(int64_t n, double d = 1.0, DType dtype = DType::Float32,
              Device device = Device::cpu()) -> Tensor;

/**
 * @brief Discrete Cosine Transform (DCT).
 *
 * Computes DCT of the given type along the specified dimension.
 * Supports types 1-4 (type 2 is the most common, used in JPEG/MPEG).
 *
 * Implemented via RFFT composition, so it works on all backends that
 * support RFFT. Backends may also register a native OpId::DCT kernel.
 *
 * @param input Real input tensor
 * @param type DCT type (1, 2, 3, or 4). Default: 2
 * @param n Signal length (default: input.shape[dim])
 * @param dim Dimension to transform (default: -1)
 * @param norm Normalization mode: "backward" (default), "forward", "ortho"
 * @return Real output tensor
 */
auto dct(const Tensor& input, int type = 2,
         std::optional<int64_t> n = std::nullopt, int64_t dim = -1,
         const std::string& norm = "backward") -> Tensor;

/**
 * @brief Inverse Discrete Cosine Transform (IDCT).
 *
 * Computes the inverse DCT. For type 2, this is DCT-III (and vice versa).
 *
 * @param input Real input tensor
 * @param type DCT type (1, 2, 3, or 4). Default: 2
 * @param n Signal length (default: input.shape[dim])
 * @param dim Dimension to transform (default: -1)
 * @param norm Normalization mode: "backward" (default), "forward", "ortho"
 * @return Real output tensor
 */
auto idct(const Tensor& input, int type = 2,
          std::optional<int64_t> n = std::nullopt, int64_t dim = -1,
          const std::string& norm = "backward") -> Tensor;

/**
 * @brief Apply a mel-frequency filterbank to a spectrogram.
 *
 * Converts a linear-frequency spectrogram into a mel-frequency spectrogram
 * using triangular mel-spaced filter banks (HTK formula).
 *
 * @param spectrogram Input tensor of shape (..., n_freqs, time_frames)
 * @param n_mels Number of mel bands (default: 128)
 * @param f_min Minimum frequency in Hz (default: 0.0)
 * @param f_max Maximum frequency in Hz (default: 0.0 means sample_rate/2)
 * @param sample_rate Audio sample rate in Hz (default: 16000)
 * @return Tensor of shape (..., n_mels, time_frames)
 */
auto mel_scale(const Tensor& spectrogram, int64_t n_mels = 128,
               double f_min = 0.0, double f_max = 0.0,
               int64_t sample_rate = 16000) -> Tensor;

/**
 * @brief Compute Mel-Frequency Cepstral Coefficients (MFCC).
 *
 * Computes MFCCs from a raw waveform by computing a power spectrogram,
 * applying mel-scale filterbank, taking log, and applying DCT.
 *
 * @param waveform Input waveform tensor of shape (..., signal_length)
 * @param sample_rate Audio sample rate in Hz (default: 16000)
 * @param n_mfcc Number of MFCC coefficients to return (default: 40)
 * @param n_mels Number of mel bands (default: 128)
 * @param n_fft FFT window size (default: 400)
 * @param hop_length Hop length for STFT (default: 160)
 * @param f_min Minimum frequency in Hz (default: 0.0)
 * @param f_max Maximum frequency in Hz (default: 0.0 means sample_rate/2)
 * @return Tensor of shape (..., n_mfcc, time_frames)
 */
auto mfcc(const Tensor& waveform, int64_t sample_rate = 16000,
          int64_t n_mfcc = 40, int64_t n_mels = 128,
          int64_t n_fft = 400, int64_t hop_length = 160,
          double f_min = 0.0, double f_max = 0.0) -> Tensor;

} // namespace fft
} // namespace tenzor
