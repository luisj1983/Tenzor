/**
 * @file creation.hpp
 * @brief Tensor creation and initialization operations
 *
 * Provides functions for creating new tensors with various initialization
 * patterns including zeros, ones, random values, ranges, and data copying.
 */

#pragma once

#include <vector>
#include <optional>
#include <cstring>
#include <type_traits>
#include "../core/tensor.hpp"
#include "../core/dtype.hpp"
#include "../core/device.hpp"

namespace tenzor {

/**
 * @defgroup tensor_creation Tensor Creation Operations
 * @brief Functions for creating and initializing tensors
 * @{
 */

/**
 * @brief Create tensor filled with zeros.
 *
 * @param shape Tensor dimensions
 * @param dtype Element data type (default: Float32)
 * @param device Target device (default: CPU)
 * @return New tensor with all elements set to 0
 *
 * @code
 * auto t = zeros({3, 4}, DType::Float32, Device::cpu());
 * @endcode
 */
auto zeros(std::vector<int64_t> shape,
          DType dtype = DType::Float32,
          Device device = Device::cpu()) -> Tensor;

/**
 * @brief Create tensor filled with ones.
 *
 * @param shape Tensor dimensions
 * @param dtype Element data type (default: Float32)
 * @param device Target device (default: CPU)
 * @return New tensor with all elements set to 1
 *
 * @code
 * auto t = ones({2, 3}, DType::Float64, Device::cuda(0));
 * @endcode
 */
auto ones(std::vector<int64_t> shape,
         DType dtype = DType::Float32,
         Device device = Device::cpu()) -> Tensor;

/**
 * @brief Create tensor filled with specific value.
 *
 * @param shape Tensor dimensions
 * @param value Fill value
 * @param dtype Element data type (default: Float32)
 * @param device Target device (default: CPU)
 * @return New tensor filled with value
 */
auto full(std::vector<int64_t> shape,
         float value,
         DType dtype = DType::Float32,
         Device device = Device::cpu()) -> Tensor;

/**
 * @brief Create uninitialized tensor.
 *
 * Allocates memory without initialization for performance.
 *
 * @param shape Tensor dimensions
 * @param dtype Element data type (default: Float32)
 * @param device Target device (default: CPU)
 * @return New tensor with uninitialized data
 * @warning Contents are undefined until written
 */
auto empty(std::vector<int64_t> shape,
          DType dtype = DType::Float32,
          Device device = Device::cpu()) -> Tensor;

/**
 * @brief Create tensor with random uniform values [0, 1).
 *
 * @param shape Tensor dimensions
 * @param dtype Element data type (default: Float32)
 * @param device Target device (default: CPU)
 * @return New tensor with uniform random values
 */
auto rand(std::vector<int64_t> shape,
         DType dtype = DType::Float32,
         Device device = Device::cpu()) -> Tensor;

/**
 * @brief Create tensor with random normal values N(0, 1).
 *
 * @param shape Tensor dimensions
 * @param dtype Element data type (default: Float32)
 * @param device Target device (default: CPU)
 * @return New tensor with standard normal random values
 */
auto randn(std::vector<int64_t> shape,
          DType dtype = DType::Float32,
          Device device = Device::cpu()) -> Tensor;

/**
 * @brief Create 1D tensor with evenly spaced values.
 *
 * Creates sequence [start, start+step, start+2*step, ..., end).
 *
 * @param start Start value (inclusive)
 * @param end End value (exclusive)
 * @param step Step size (default: 1.0)
 * @param dtype Element data type (default: Float32)
 * @param device Target device (default: CPU)
 * @return 1D tensor with range values
 *
 * @code
 * auto t = arange(0.0f, 10.0f, 2.0f);  // [0, 2, 4, 6, 8]
 * @endcode
 */
auto arange(float start, float end, float step = 1.0f,
           DType dtype = DType::Float32,
           Device device = Device::cpu()) -> Tensor;

/**
 * @brief Create 1D tensor with linearly spaced values.
 *
 * Creates sequence with exact number of steps.
 *
 * @param start Start value (inclusive)
 * @param end End value (inclusive)
 * @param steps Number of values to generate
 * @param dtype Element data type (default: Float32)
 * @param device Target device (default: CPU)
 * @return 1D tensor with linearly spaced values
 *
 * @code
 * auto t = linspace(0.0f, 1.0f, 5);  // [0.0, 0.25, 0.5, 0.75, 1.0]
 * @endcode
 */
auto linspace(float start, float end, int64_t steps,
             DType dtype = DType::Float32,
             Device device = Device::cpu()) -> Tensor;

/**
 * @brief Create identity matrix.
 *
 * @param n Number of rows
 * @param m Number of columns (default: same as n)
 * @param dtype Element data type (default: Float32)
 * @param device Target device (default: CPU)
 * @return 2D tensor with ones on diagonal, zeros elsewhere
 *
 * @code
 * auto I = eye(3);  // 3x3 identity matrix
 * @endcode
 */
auto eye(int64_t n, std::optional<int64_t> m = std::nullopt,
        DType dtype = DType::Float32,
        Device device = Device::cpu()) -> Tensor;

/**
 * @brief Create tensor from raw data array.
 *
 * Copies data from C-style array into new tensor.
 *
 * @tparam T Element type (must match one of the supported DTypes)
 * @param data Pointer to source data
 * @param shape Tensor dimensions
 * @param device Target device (default: CPU)
 * @return New tensor with copied data
 *
 * @code
 * float data[] = {1.0f, 2.0f, 3.0f, 4.0f};
 * auto t = from_data(data, {2, 2});
 * @endcode
 */
template<typename T>
auto from_data(const T* data, std::vector<int64_t> shape,
              Device device = Device::cpu()) -> Tensor {
    // Determine dtype from T
    DType dtype;
    if constexpr (std::is_same_v<T, float>) {
        dtype = DType::Float32;
    } else if constexpr (std::is_same_v<T, double>) {
        dtype = DType::Float64;
    } else if constexpr (std::is_same_v<T, int32_t>) {
        dtype = DType::Int32;
    } else if constexpr (std::is_same_v<T, int64_t>) {
        dtype = DType::Int64;
    } else if constexpr (std::is_same_v<T, uint8_t>) {
        dtype = DType::UInt8;
    } else if constexpr (std::is_same_v<T, bool>) {
        dtype = DType::Bool;
    } else {
        static_assert(std::is_same_v<T, float>, "Unsupported type for from_data");
    }

    // Create empty tensor
    auto tensor = empty(shape, dtype, device);

    // Copy data
    if (tensor.impl() && tensor.impl()->storage) {
        size_t bytes = tensor.numel() * sizeof(T);
        std::memcpy(tensor.impl()->storage->data(), data, bytes);
    }

    return tensor;
}

/**
 * @brief Create zero tensor with same shape/dtype/device as input.
 * @param tensor Template tensor
 * @return Zero tensor matching input properties
 */
auto zeros_like(const Tensor& tensor) -> Tensor;

/**
 * @brief Create one tensor with same shape/dtype/device as input.
 * @param tensor Template tensor
 * @return One tensor matching input properties
 */
auto ones_like(const Tensor& tensor) -> Tensor;

/**
 * @brief Create uniform random tensor with same shape/dtype/device as input.
 * @param tensor Template tensor
 * @return Random tensor matching input properties
 */
auto rand_like(const Tensor& tensor) -> Tensor;

/**
 * @brief Create normal random tensor with same shape/dtype/device as input.
 * @param tensor Template tensor
 * @return Random tensor matching input properties
 */
auto randn_like(const Tensor& tensor) -> Tensor;

/** @} */ // end of tensor_creation group

} // namespace tenzor
