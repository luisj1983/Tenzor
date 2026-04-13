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
 * Reconstructs a time-domain signal from a magnitude spectrogram
 * by iteratively estimating the phase using STFT/ISTFT round-trips.
 *
 * @param magnitude Magnitude spectrogram (from |STFT(signal)|)
 * @param n_fft FFT window size
 * @param hop_length Hop between windows (default: n_fft / 4)
 * @param win_length Window length (default: n_fft)
 * @param window Optional window tensor
 * @param n_iter Number of iterations (default: 32)
 * @param momentum Momentum for phase update (default: 0.99)
 * @return Reconstructed time-domain signal
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

} // namespace fft
} // namespace tenzor
