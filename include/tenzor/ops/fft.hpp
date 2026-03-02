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

} // namespace fft
} // namespace tenzor
