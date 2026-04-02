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

} // namespace fft
} // namespace tenzor
