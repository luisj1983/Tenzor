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
#include "../backend/loader_fwd.hpp"
#include "../backend/backend.hpp"

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
 * @brief Create tensor filled with specific value (double precision).
 *
 * @param shape Tensor dimensions
 * @param value Fill value (double precision)
 * @param dtype Element data type
 * @param device Target device
 * @return New tensor filled with value
 */
auto full(std::vector<int64_t> shape,
         double value,
         DType dtype,
         Device device) -> Tensor;

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
 * @brief Create random permutation of integers.
 *
 * Returns a tensor containing a random permutation of integers from 0 to n-1.
 *
 * @param n Upper bound (exclusive)
 * @param device Target device (default: CPU)
 * @return Tensor of shape {n} with permuted indices
 */
auto randperm(int64_t n, Device device = Device::cpu()) -> Tensor;

/**
 * @brief Create tensor with random integers.
 *
 * Returns a tensor filled with random integers from uniform distribution [low, high).
 *
 * @param low Lower bound (inclusive)
 * @param high Upper bound (exclusive)
 * @param shape Tensor dimensions
 * @param dtype Element data type (default: Int64)
 * @param device Target device (default: CPU)
 * @return Tensor with random integers
 *
 * @code
 * auto t = randint(0, 10, {3, 4});  // 3x4 tensor with ints in [0, 10)
 * @endcode
 */
auto randint(int64_t low, int64_t high, std::vector<int64_t> shape,
            DType dtype = DType::Int64,
            Device device = Device::cpu()) -> Tensor;

/**
 * @brief Draw samples from a multinomial distribution.
 *
 * @param input Tensor of probabilities (unnormalized weights)
 * @param num_samples Number of samples to draw
 * @param replacement Whether to sample with replacement (default: false)
 * @return Int64 tensor of sampled indices
 */
auto multinomial(const Tensor& input, int64_t num_samples, bool replacement = false) -> Tensor;

/**
 * @brief Sample from a Bernoulli distribution.
 *
 * Each element is independently sampled from Bernoulli(p) where p comes
 * from the input tensor.
 *
 * @param probs Tensor of probabilities in [0, 1]
 * @return Tensor with same shape, values 0.0 or 1.0
 */
auto bernoulli(const Tensor& probs) -> Tensor;

/**
 * @brief Sample from a normal (Gaussian) distribution with specified mean and std.
 *
 * @param mean Mean of the distribution (tensor or scalar broadcast)
 * @param std Standard deviation of the distribution (tensor or scalar broadcast)
 * @return Tensor with same shape as mean/std, values sampled from N(mean, std^2)
 */
auto normal(const Tensor& mean, const Tensor& std) -> Tensor;

/**
 * @brief Sample from a Poisson distribution with the given rates.
 *
 * @param rates Tensor of rate parameters (lambda > 0)
 * @return Int64 tensor with same shape, values sampled from Poisson(rates)
 */
auto poisson(const Tensor& rates) -> Tensor;

/**
 * @brief Sample from an exponential distribution with the given rate.
 *
 * @param rate Tensor of rate parameters (lambda > 0)
 * @return Tensor with same shape, values sampled from Exponential(rate)
 */
auto exponential(const Tensor& rate) -> Tensor;

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
auto arange(double start, double end, double step = 1.0,
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
 * @brief Create a 1-D tensor of logarithmically spaced values.
 *
 * Generates `steps` values logarithmically spaced between base^start and base^end.
 * Equivalent to pow(base, linspace(start, end, steps)).
 *
 * @param start Exponent of the start value
 * @param end Exponent of the end value
 * @param steps Number of values to generate
 * @param base Base of the logarithm (default: 10.0)
 * @param dtype Element data type (default: Float32)
 * @param device Target device (default: CPU)
 * @return 1-D tensor of size `steps`
 */
auto logspace(float start, float end, int64_t steps, double base = 10.0,
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
    if (tensor.impl() && tensor.storage()) {
        size_t bytes = tensor.numel() * sizeof(T);

        // For CPU device, use memcpy
        if (device.type == Device::Type::CPU) {
            std::memcpy(tensor.storage()->data(), data, bytes);
        } else {
            // For GPU devices, use backend's copy method
            auto* backend = tenzor::try_get_backend(device.type);
            if (backend) {
                backend->copy(tensor.storage()->data(),
                             const_cast<T*>(data),
                             bytes,
                             tenzor::CopyKind::HostToDevice);
            } else {
                throw std::runtime_error("Backend not available for device type");
            }
        }
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

/// Create tensor filled with fill_value, same shape/dtype/device as input
auto full_like(const Tensor& tensor, double fill_value) -> Tensor;

/// Create uninitialized tensor with same shape/dtype/device as input
auto empty_like(const Tensor& tensor) -> Tensor;

/// Create random integer tensor with same shape/device as input
auto randint_like(const Tensor& tensor, int64_t low, int64_t high) -> Tensor;

/**
 * @brief Set the seed for random number generation.
 * @param seed Seed value for reproducible random tensors
 *
 * Sets the seed for rand() and randn() operations to enable reproducible
 * random tensor generation across multiple runs.
 */
void manual_seed(unsigned int seed);

/**
 * @brief Check if a manual seed has been set and retrieve it
 *
 * Returns the seed value set by manual_seed(), or generates a time-based
 * seed if manual_seed() has not been called. Backend implementations
 * should use this to respect the global seed setting.
 */
uint64_t get_global_seed();

/**
 * @brief Generate coordinate grids from 1-D coordinate tensors.
 *
 * @param tensors List of 1-D tensors
 * @param indexing "ij" for matrix indexing (default), "xy" for Cartesian
 * @return Vector of N-D coordinate tensors
 */
auto meshgrid(const std::vector<Tensor>& tensors, const std::string& indexing = "ij") -> std::vector<Tensor>;

/**
 * @brief Generate lower-triangular index pairs.
 *
 * Like PyTorch's torch.tril_indices. Returns a (2, N) Int64 tensor where
 * N is the number of lower-triangular elements (including the diagonal
 * shifted by offset).
 *
 * @param row Number of rows
 * @param col Number of columns
 * @param offset Diagonal offset (0 = main diagonal, positive = above, negative = below)
 * @param dtype Element data type (default: Int64)
 * @param device Target device (default: CPU)
 * @return (2, N) tensor of (row_indices, col_indices)
 */
auto tril_indices(int64_t row, int64_t col, int64_t offset = 0,
                  DType dtype = DType::Int64,
                  Device device = Device::cpu()) -> Tensor;

/**
 * @brief Generate upper-triangular index pairs.
 *
 * Like PyTorch's torch.triu_indices. Returns a (2, N) Int64 tensor where
 * N is the number of upper-triangular elements (including the diagonal
 * shifted by offset).
 *
 * @param row Number of rows
 * @param col Number of columns
 * @param offset Diagonal offset (0 = main diagonal, positive = above, negative = below)
 * @param dtype Element data type (default: Int64)
 * @param device Target device (default: CPU)
 * @return (2, N) tensor of (row_indices, col_indices)
 */
auto triu_indices(int64_t row, int64_t col, int64_t offset = 0,
                  DType dtype = DType::Int64,
                  Device device = Device::cpu()) -> Tensor;

/**
 * @brief Create a complex tensor from separate real and imaginary parts.
 *
 * Interleaves real and imaginary components into Complex64 storage.
 * Both inputs must have the same shape and a float dtype.
 *
 * @param real Tensor of real parts
 * @param imag Tensor of imaginary parts
 * @return Complex64 tensor with same shape as inputs
 */
auto complex(const Tensor& real, const Tensor& imag) -> Tensor;

/** @} */ // end of tensor_creation group

} // namespace tenzor

namespace tenzor {
namespace ops {
// Convenience namespace alias for common operations
using tenzor::zeros;
using tenzor::zeros_like;
using tenzor::ones;
using tenzor::ones_like;
using tenzor::full;
using tenzor::arange;
using tenzor::linspace;
using tenzor::eye;
using tenzor::rand;
using tenzor::randn;
using tenzor::rand_like;
using tenzor::randn_like;
using tenzor::randperm;
using tenzor::randint;
using tenzor::tril_indices;
using tenzor::triu_indices;
} // namespace ops
} // namespace tenzor
