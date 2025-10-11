/**
 * @file reduction.hpp
 * @brief Tensor reduction operations
 *
 * Provides operations that reduce tensors along specified dimensions,
 * including sum, mean, min, max, and statistical operations.
 */

#pragma once

#include <optional>
#include <vector>
#include "../core/tensor.hpp"

namespace tenzor {

/**
 * @defgroup tensor_reduction Reduction Operations
 * @brief Operations that reduce tensor dimensions
 * @{
 */

/**
 * @brief Sum of tensor elements.
 *
 * @param input Input tensor
 * @param dim Dimension to reduce (nullopt = reduce all)
 * @param keepdim Keep reduced dimension as size 1
 * @return Sum tensor
 *
 * @code
 * auto t = Tensor({3, 4}, DType::Float32, Device::cpu());
 * auto total = sum(t);            // Scalar result
 * auto col_sums = sum(t, 0, true); // Shape: {1, 4}
 * @endcode
 */
auto sum(const Tensor& input,
        std::optional<int64_t> dim = std::nullopt,
        bool keepdim = false) -> Tensor;

/** @brief Mean (average) of tensor elements. */
auto mean(const Tensor& input,
         std::optional<int64_t> dim = std::nullopt,
         bool keepdim = false) -> Tensor;

/** @brief Maximum value of tensor elements. */
auto max(const Tensor& input,
        std::optional<int64_t> dim = std::nullopt,
        bool keepdim = false) -> Tensor;

/** @brief Minimum value of tensor elements. */
auto min(const Tensor& input,
        std::optional<int64_t> dim = std::nullopt,
        bool keepdim = false) -> Tensor;

/** @brief Indices of maximum values. */
auto argmax(const Tensor& input,
           std::optional<int64_t> dim = std::nullopt,
           bool keepdim = false) -> Tensor;

/** @brief Indices of minimum values. */
auto argmin(const Tensor& input,
           std::optional<int64_t> dim = std::nullopt,
           bool keepdim = false) -> Tensor;

/** @brief Product of tensor elements. */
auto prod(const Tensor& input,
         std::optional<int64_t> dim = std::nullopt,
         bool keepdim = false) -> Tensor;

/**
 * @brief Standard deviation of tensor elements.
 * @param unbiased Use unbiased estimator (N-1 denominator)
 */
auto std(const Tensor& input,
        std::optional<int64_t> dim = std::nullopt,
        bool keepdim = false,
        bool unbiased = true) -> Tensor;

/**
 * @brief Variance of tensor elements.
 * @param unbiased Use unbiased estimator (N-1 denominator)
 */
auto var(const Tensor& input,
        std::optional<int64_t> dim = std::nullopt,
        bool keepdim = false,
        bool unbiased = true) -> Tensor;

/**
 * @brief p-norm of tensor.
 * @param p Norm order (1=L1, 2=L2, inf=max)
 */
auto norm(const Tensor& input,
         float p = 2.0f,
         std::optional<int64_t> dim = std::nullopt,
         bool keepdim = false) -> Tensor;

/** @} */ // end of tensor_reduction group

} // namespace tenzor
