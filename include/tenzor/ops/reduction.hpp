/**
 * @file reduction.hpp
 * @brief Tensor reduction operations
 *
 * Provides operations that reduce tensors along specified dimensions,
 * including sum, mean, min, max, and statistical operations.
 */

#pragma once

#include <optional>
#include <utility>
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

/**
 * @brief Sort tensor and return indices.
 * @param input Input tensor
 * @param dim Dimension to sort along (default: last dimension)
 * @param descending Sort in descending order (default: false)
 * @return Tensor of indices that sort the input
 */
auto argsort(const Tensor& input,
            int64_t dim = -1,
            bool descending = false) -> Tensor;

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

/**
 * @brief Numerically stable log-sum-exp reduction.
 *
 * Computes log(sum(exp(input), dim)) in a numerically stable way by
 * subtracting the maximum value before exponentiation.
 *
 * @param input Input tensor
 * @param dim Dimension to reduce
 * @param keepdim Keep reduced dimension as size 1
 * @return log(sum(exp(input), dim))
 *
 * @code
 * auto t = Tensor({3, 4}, DType::Float32, Device::cpu());
 * auto lse = logsumexp(t, 1);       // Shape: {3}
 * auto lse_kd = logsumexp(t, 1, true); // Shape: {3, 1}
 * @endcode
 */
auto logsumexp(const Tensor& input,
              int64_t dim,
              bool keepdim = false) -> Tensor;

/** @brief True if any element is nonzero. Returns Bool tensor. */
auto any(const Tensor& input,
        std::optional<int64_t> dim = std::nullopt,
        bool keepdim = false) -> Tensor;

/** @brief True if all elements are nonzero. Returns Bool tensor. */
auto all(const Tensor& input,
        std::optional<int64_t> dim = std::nullopt,
        bool keepdim = false) -> Tensor;

/** @brief Check if tensor contains any inf or nan values. Returns Bool scalar tensor. */
auto has_inf_nan(const Tensor& input) -> Tensor;

/**
 * @brief Compute histogram of a tensor.
 *
 * @param input Input tensor (flattened for histogram computation)
 * @param bins Number of equal-width bins
 * @param min Minimum value of the range (inclusive). If min == max, uses data range.
 * @param max Maximum value of the range (exclusive). If min == max, uses data range.
 * @return Pair of (histogram counts as Int64 tensor, bin edges as Float tensor)
 */
auto histogram(const Tensor& input, int64_t bins = 10, double min = 0.0, double max = 0.0)
    -> std::pair<Tensor, Tensor>;

/**
 * @brief Compute multi-dimensional histogram.
 *
 * @param input Input tensor of shape (N, D) where N is the number of samples
 *              and D is the number of dimensions
 * @param bins Number of bins per dimension (length D)
 * @param ranges Optional per-dimension (min, max) ranges. If nullopt, auto-detects from data.
 * @param density If true, normalize the histogram so that the integral over the
 *                bins equals 1 (counts / (total_count * bin_volume))
 * @return Pair of (counts tensor of shape bins[0] x bins[1] x ... x bins[D-1],
 *         vector of D edge tensors)
 */
auto histogramdd(const Tensor& input, std::vector<int64_t> bins,
                 std::optional<std::vector<std::pair<double,double>>> ranges = std::nullopt,
                 bool density = false) -> std::pair<Tensor, std::vector<Tensor>>;

/** @brief Count nonzero elements along a dimension or globally. */
auto count_nonzero(const Tensor& input,
                   std::optional<int64_t> dim = std::nullopt) -> Tensor;

/** @brief Sum of tensor elements, treating NaN as zero. */
auto nansum(const Tensor& input,
            std::optional<int64_t> dim = std::nullopt,
            bool keepdim = false) -> Tensor;

/** @brief Mean of tensor elements, ignoring NaN values. */
auto nanmean(const Tensor& input,
             std::optional<int64_t> dim = std::nullopt,
             bool keepdim = false) -> Tensor;

/** @brief Variance of tensor elements, ignoring NaN values. */
auto nanvar(const Tensor& input,
            std::optional<int64_t> dim = std::nullopt,
            bool keepdim = false,
            int64_t correction = 1) -> Tensor;

/** @brief Standard deviation of tensor elements, ignoring NaN values. */
auto nanstd(const Tensor& input,
            std::optional<int64_t> dim = std::nullopt,
            bool keepdim = false,
            int64_t correction = 1) -> Tensor;

/** @brief Simultaneous min and max in a single pass. Returns (min_values, max_values). */
auto aminmax(const Tensor& input,
             std::optional<int64_t> dim = std::nullopt,
             bool keepdim = false) -> std::pair<Tensor, Tensor>;

// =========================================================================
// Fused reductions (compositions, no new OpIds)
// =========================================================================

/// Fused std + mean in a single pass (returns std, mean)
auto std_mean(const Tensor& input, std::optional<int64_t> dim = std::nullopt,
              bool keepdim = false, bool unbiased = true) -> std::pair<Tensor, Tensor>;

/// Fused var + mean in a single pass (returns var, mean)
auto var_mean(const Tensor& input, std::optional<int64_t> dim = std::nullopt,
              bool keepdim = false, bool unbiased = true) -> std::pair<Tensor, Tensor>;

/// p-norm of (a - b)
auto dist(const Tensor& a, const Tensor& b, float p = 2.0f) -> Tensor;

// =========================================================================
// New reduction operations for PyTorch parity
// =========================================================================

/// Cumulative maximum along dim (returns values, indices)
auto cummax(const Tensor& input, int64_t dim) -> std::pair<Tensor, Tensor>;

/// Cumulative minimum along dim (returns values, indices)
auto cummin(const Tensor& input, int64_t dim) -> std::pair<Tensor, Tensor>;

/// Test if elements are in a set of test values
auto isin(const Tensor& elements, const Tensor& test_elements) -> Tensor;

/// k-th smallest value along dim (returns value, index)
auto kthvalue(const Tensor& input, int64_t k, int64_t dim = -1, bool keepdim = false) -> std::pair<Tensor, Tensor>;

/// Element-wise max ignoring NaN (returns non-NaN if one input is NaN)
auto fmax(const Tensor& a, const Tensor& b) -> Tensor;

/// Element-wise min ignoring NaN (returns non-NaN if one input is NaN)
auto fmin(const Tensor& a, const Tensor& b) -> Tensor;

/// Quantile along dim
auto quantile(const Tensor& input, double q, std::optional<int64_t> dim = std::nullopt,
              bool keepdim = false) -> Tensor;

/// NaN-ignoring quantile along dim
auto nanquantile(const Tensor& input, double q, std::optional<int64_t> dim = std::nullopt,
                 bool keepdim = false) -> Tensor;

/// NaN-ignoring median along dim
auto nanmedian(const Tensor& input, std::optional<int64_t> dim = std::nullopt) -> Tensor;

/// Fixed-bin histogram
auto histc(const Tensor& input, int64_t bins = 100, double min_val = 0, double max_val = 0) -> Tensor;

/// Deduplicate consecutive equal elements
auto unique_consecutive(const Tensor& input, bool return_inverse = false,
                        bool return_counts = false, std::optional<int64_t> dim = std::nullopt)
    -> std::tuple<Tensor, Tensor, Tensor>;

/**
 * @brief Covariance matrix of the given variables.
 *
 * If input is a 2D matrix of shape (N, M), treats each row as a variable
 * and each column as an observation. Returns an (N, N) covariance matrix.
 * For 1D input, returns the variance as a scalar tensor.
 *
 * @param input Input tensor (1D or 2D)
 * @param correction Degrees of freedom correction (default: 1 for Bessel's)
 * @return Covariance matrix
 */
auto cov(const Tensor& input, int64_t correction = 1) -> Tensor;

/**
 * @brief Pearson correlation coefficient matrix.
 *
 * Equivalent to normalizing the covariance matrix by the product of
 * standard deviations. Returns values in [-1, 1].
 *
 * @param input Input tensor (1D or 2D)
 * @return Correlation coefficient matrix
 */
auto corrcoef(const Tensor& input) -> Tensor;

// =========================================================================
// Numerical integration and gradient operations
// =========================================================================

/**
 * @brief Trapezoidal numerical integration with non-uniform spacing.
 *
 * @param y Values tensor
 * @param x Coordinates tensor (same shape as y along dim)
 * @param dim Dimension to integrate along (default: -1)
 * @return Tensor with integrated dimension removed
 */
auto trapezoid(const Tensor& y, const Tensor& x, int64_t dim = -1) -> Tensor;

/**
 * @brief Trapezoidal numerical integration with uniform spacing.
 *
 * @param y Values tensor
 * @param dx Uniform spacing (default: 1.0)
 * @param dim Dimension to integrate along (default: -1)
 * @return Tensor with integrated dimension removed
 */
auto trapezoid(const Tensor& y, double dx = 1.0, int64_t dim = -1) -> Tensor;

/**
 * @brief Cumulative trapezoidal integration with non-uniform spacing.
 *
 * @param y Values tensor
 * @param x Coordinates tensor
 * @param dim Dimension to integrate along (default: -1)
 * @return Tensor with dim size reduced by 1
 */
auto cumulative_trapezoid(const Tensor& y, const Tensor& x, int64_t dim = -1) -> Tensor;

/**
 * @brief Cumulative trapezoidal integration with uniform spacing.
 *
 * @param y Values tensor
 * @param dx Uniform spacing (default: 1.0)
 * @param dim Dimension to integrate along (default: -1)
 * @return Tensor with dim size reduced by 1
 */
auto cumulative_trapezoid(const Tensor& y, double dx = 1.0, int64_t dim = -1) -> Tensor;

/**
 * @brief NumPy-style numerical gradient using finite differences.
 *
 * Central differences in the interior, one-sided at boundaries.
 *
 * @param input Input tensor
 * @param dim Dimension along which to compute gradient (default: -1)
 * @param spacing Uniform spacing between samples (default: 1.0)
 * @return Gradient tensor (same shape as input)
 */
auto gradient(const Tensor& input, int64_t dim = -1, double spacing = 1.0) -> Tensor;

/**
 * @brief Reduce tensor elements over segments defined by offsets.
 *
 * Segment i contains elements from offsets[i] to offsets[i+1] along the
 * given axis.  The supported reduction modes are "sum", "mean", "max",
 * "min", and "prod".
 *
 * @param data   Input tensor
 * @param offsets 1-D Int64 tensor of length (num_segments + 1)
 * @param reduce Reduction mode: "sum" | "mean" | "max" | "min" | "prod"
 * @param axis   Axis along which segments are defined (default: 0)
 * @return Tensor with the segment dimension replaced by num_segments
 *
 * @code
 * // data = [1, 2, 3, 4, 5], offsets = [0, 2, 5]
 * // segment 0 = [1,2], segment 1 = [3,4,5]
 * auto out = segment_reduce(data, offsets, "sum"); // [3, 12]
 * @endcode
 */
auto segment_reduce(const Tensor& data, const Tensor& offsets,
                    const std::string& reduce = "sum",
                    int64_t axis = 0) -> Tensor;

/** @} */ // end of tensor_reduction group

} // namespace tenzor

namespace tenzor {
namespace ops {
// Convenience namespace alias for common operations
using tenzor::sum;
using tenzor::mean;
using tenzor::max;
using tenzor::min;
using tenzor::argmax;
using tenzor::argmin;
using tenzor::argsort;
using tenzor::prod;
} // namespace ops
} // namespace tenzor
