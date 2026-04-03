/**
 * @file windows.hpp
 * @brief Window functions for spectral analysis (Hann, Hamming, Blackman, etc.)
 */

#pragma once

#include "../core/tensor.hpp"
#include <cmath>

namespace tenzor {

/**
 * @brief Create a Hann (Hanning) window.
 * @param size Window length
 * @param periodic If true, returns periodic window (for spectral analysis)
 * @param dtype Output data type (default: Float32)
 * @param device Output device (default: CPU)
 */
auto hann_window(int64_t size, bool periodic = true,
                 DType dtype = DType::Float32,
                 Device device = Device::cpu()) -> Tensor;

/**
 * @brief Create a Hamming window.
 * @param size Window length
 * @param periodic If true, returns periodic window
 * @param alpha Hamming coefficient alpha (default: 0.54)
 * @param beta Hamming coefficient beta (default: 0.46)
 */
auto hamming_window(int64_t size, bool periodic = true,
                    double alpha = 0.54, double beta = 0.46,
                    DType dtype = DType::Float32,
                    Device device = Device::cpu()) -> Tensor;

/**
 * @brief Create a Blackman window.
 */
auto blackman_window(int64_t size, bool periodic = true,
                     DType dtype = DType::Float32,
                     Device device = Device::cpu()) -> Tensor;

/**
 * @brief Create a Bartlett (triangular) window.
 */
auto bartlett_window(int64_t size, bool periodic = true,
                     DType dtype = DType::Float32,
                     Device device = Device::cpu()) -> Tensor;

/**
 * @brief Create a Kaiser window.
 * @param beta Shape parameter (higher = narrower main lobe)
 */
auto kaiser_window(int64_t size, bool periodic = true,
                   double beta = 12.0,
                   DType dtype = DType::Float32,
                   Device device = Device::cpu()) -> Tensor;

} // namespace tenzor
