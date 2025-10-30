/**
 * @file gradient_utils.hpp
 * @brief Utility functions for gradient operations in ZeRO Stage 2
 *
 * Provides functions for gradient flattening, bucketing, and memory management
 * needed by ZeRO Stage 2 gradient partitioning.
 */

#pragma once

#include "../../core/tensor.hpp"
#include "../../autograd/variable.hpp"
#include <vector>
#include <cstddef>
#include <memory>

namespace tenzor {
namespace optim {

/**
 * @brief Configuration for gradient bucketing
 */
struct BucketConfig {
    size_t max_bucket_size_mb{25};      ///< Maximum bucket size in megabytes
    size_t min_bucket_size_mb{1};       ///< Minimum bucket size in megabytes
    bool align_buckets{true};           ///< Align bucket boundaries to parameter boundaries
    bool sort_by_size{false};           ///< Sort parameters by size before bucketing
};

/**
 * @brief Information about a gradient bucket
 */
struct BucketInfo {
    size_t start_idx{0};                ///< Start index in parameter list
    size_t end_idx{0};                  ///< End index (exclusive) in parameter list
    size_t num_elements{0};             ///< Total number of elements in bucket
    size_t size_bytes{0};               ///< Total size in bytes
    DType dtype{DType::Float32};        ///< Data type of tensors in bucket
    Device device{Device::cpu()};       ///< Device where bucket resides
};

/**
 * @brief Memory usage estimates for gradients
 */
struct GradientMemoryStats {
    size_t total_parameters{0};         ///< Total number of parameters
    size_t total_elements{0};           ///< Total number of elements across all tensors
    size_t total_bytes{0};              ///< Total memory in bytes
    size_t bytes_per_rank{0};           ///< Memory per rank (partitioned)
    size_t num_buckets{0};              ///< Number of communication buckets
    size_t max_bucket_bytes{0};         ///< Size of largest bucket
    size_t min_bucket_bytes{0};         ///< Size of smallest bucket
    double fragmentation_ratio{0.0};    ///< Memory fragmentation (wasted space ratio)
};

/**
 * @brief Flatten a vector of tensors into a single contiguous tensor
 *
 * Combines multiple tensors with potentially different shapes into a single
 * 1D tensor by concatenating their data in memory order.
 *
 * All input tensors must:
 * - Have the same dtype
 * - Be on the same device
 * - Be contiguous in memory (call .contiguous() if needed)
 *
 * @param tensors Vector of tensors to flatten
 * @return Single 1D tensor containing all data from input tensors
 * @throws std::invalid_argument if tensors have different dtypes or devices
 * @throws std::invalid_argument if input is empty
 *
 * @code
 * std::vector<Tensor> grads = {
 *     Tensor({10, 20}, DType::Float32, Device::cpu()),
 *     Tensor({5, 5}, DType::Float32, Device::cpu())
 * };
 * Tensor flat = flatten_tensors(grads);  // Shape: {225} (200 + 25)
 * @endcode
 */
auto flatten_tensors(const std::vector<Tensor>& tensors) -> Tensor;

/**
 * @brief Flatten gradients from a vector of Variables
 *
 * Convenience overload that extracts and flattens gradients from Variables.
 * Skips Variables with null gradients.
 *
 * @param variables Vector of Variables with gradients
 * @return Single 1D tensor containing all gradient data
 * @throws std::invalid_argument if no Variables have gradients
 *
 * @code
 * std::vector<std::shared_ptr<Variable>> params = model.parameters();
 * Tensor flat_grad = flatten_tensors(params);
 * @endcode
 */
auto flatten_tensors(const std::vector<std::shared_ptr<Variable>>& variables) -> Tensor;

/**
 * @brief Unflatten a contiguous tensor back into a vector of tensors
 *
 * Splits a flattened 1D tensor back into multiple tensors with specified shapes.
 * This is the inverse operation of flatten_tensors().
 *
 * The sum of elements in shapes must equal the number of elements in the input tensor.
 *
 * @param flat_tensor Flattened 1D tensor
 * @param shapes Vector of shapes for output tensors
 * @return Vector of tensors with specified shapes
 * @throws std::invalid_argument if shapes don't match flat_tensor size
 * @throws std::invalid_argument if flat_tensor is not 1D
 *
 * @code
 * Tensor flat = Tensor({225}, DType::Float32, Device::cpu());
 * std::vector<std::vector<int64_t>> shapes = {{10, 20}, {5, 5}};
 * auto tensors = unflatten_into(flat, shapes);  // 2 tensors: {10,20} and {5,5}
 * @endcode
 */
auto unflatten_into(const Tensor& flat_tensor,
                   const std::vector<std::vector<int64_t>>& shapes) -> std::vector<Tensor>;

/**
 * @brief Unflatten a tensor and copy into existing tensor views
 *
 * Splits a flattened tensor and copies data into provided output tensors.
 * Output tensors must already be allocated with correct shapes and dtypes.
 *
 * This is more efficient than unflatten_into() when output tensors are already allocated,
 * as it avoids creating new tensors.
 *
 * @param flat_tensor Flattened 1D tensor
 * @param output_tensors Pre-allocated output tensors to copy into
 * @throws std::invalid_argument if output tensors don't match flat_tensor size
 * @throws std::invalid_argument if dtypes don't match
 *
 * @code
 * Tensor flat = flatten_tensors(gradients);
 * // ... communicate flat tensor ...
 * unflatten_into(flat, gradients);  // Copy back into original gradients
 * @endcode
 */
auto unflatten_into(const Tensor& flat_tensor,
                   std::vector<Tensor>& output_tensors) -> void;

/**
 * @brief Compute optimal bucket sizes for gradient communication
 *
 * Divides parameters into buckets for efficient collective communication operations.
 * Tries to balance:
 * - Bucket size (should be large enough to amortize communication overhead)
 * - Number of buckets (more buckets = more parallelism)
 * - Alignment (avoid splitting parameters across buckets if possible)
 *
 * Algorithm:
 * 1. Calculate total gradient memory
 * 2. Determine number of buckets based on max_bucket_size
 * 3. Assign parameters to buckets greedily
 * 4. Ensure all parameters have the same dtype per bucket
 *
 * @param tensors Vector of tensors to bucket (typically parameters)
 * @param config Bucketing configuration
 * @return Vector of bucket information structures
 * @throws std::invalid_argument if input is empty
 *
 * @code
 * BucketConfig config;
 * config.max_bucket_size_mb = 25;  // 25MB buckets
 * auto buckets = compute_bucket_sizes(model.parameters(), config);
 * std::cout << "Created " << buckets.size() << " buckets\n";
 * @endcode
 */
auto compute_bucket_sizes(const std::vector<Tensor>& tensors,
                         const BucketConfig& config = BucketConfig{}) -> std::vector<BucketInfo>;

/**
 * @brief Compute bucket sizes for Variables (convenience overload)
 *
 * @param variables Vector of Variables (typically model parameters)
 * @param config Bucketing configuration
 * @return Vector of bucket information structures
 */
auto compute_bucket_sizes(const std::vector<std::shared_ptr<Variable>>& variables,
                         const BucketConfig& config = BucketConfig{}) -> std::vector<BucketInfo>;

/**
 * @brief Estimate memory usage for gradients
 *
 * Analyzes gradient memory requirements for a set of parameters.
 * Useful for:
 * - Planning memory allocation
 * - Determining optimal bucket sizes
 * - Estimating distributed memory savings
 *
 * @param tensors Vector of tensors (typically parameters)
 * @param world_size Number of distributed ranks (for partitioning estimates)
 * @param config Bucketing configuration (for bucket analysis)
 * @return Memory statistics structure
 *
 * @code
 * auto stats = estimate_gradient_memory(model.parameters(), 4);
 * std::cout << "Total gradient memory: " << stats.total_bytes / (1024*1024) << " MB\n";
 * std::cout << "Per-rank memory: " << stats.bytes_per_rank / (1024*1024) << " MB\n";
 * std::cout << "Memory savings: " << (1.0 - 1.0/world_size) * 100 << "%\n";
 * @endcode
 */
auto estimate_gradient_memory(const std::vector<Tensor>& tensors,
                              int world_size = 1,
                              const BucketConfig& config = BucketConfig{}) -> GradientMemoryStats;

/**
 * @brief Estimate memory usage for Variables (convenience overload)
 *
 * @param variables Vector of Variables (typically model parameters)
 * @param world_size Number of distributed ranks
 * @param config Bucketing configuration
 * @return Memory statistics structure
 */
auto estimate_gradient_memory(const std::vector<std::shared_ptr<Variable>>& variables,
                              int world_size = 1,
                              const BucketConfig& config = BucketConfig{}) -> GradientMemoryStats;

/**
 * @brief Check if all tensors in a vector are contiguous
 *
 * @param tensors Vector of tensors to check
 * @return true if all tensors are contiguous
 */
auto all_contiguous(const std::vector<Tensor>& tensors) -> bool;

/**
 * @brief Check if all tensors have the same dtype
 *
 * @param tensors Vector of tensors to check
 * @return true if all tensors have the same dtype
 */
auto same_dtype(const std::vector<Tensor>& tensors) -> bool;

/**
 * @brief Check if all tensors are on the same device
 *
 * @param tensors Vector of tensors to check
 * @return true if all tensors are on the same device
 */
auto same_device(const std::vector<Tensor>& tensors) -> bool;

/**
 * @brief Get total number of elements across all tensors
 *
 * @param tensors Vector of tensors
 * @return Sum of numel() for all tensors
 */
auto total_numel(const std::vector<Tensor>& tensors) -> size_t;

/**
 * @brief Get total memory in bytes for all tensors
 *
 * @param tensors Vector of tensors
 * @return Total memory in bytes
 */
auto total_bytes(const std::vector<Tensor>& tensors) -> size_t;

/**
 * @brief Extract shapes from a vector of tensors
 *
 * @param tensors Vector of tensors
 * @return Vector of shapes
 */
auto extract_shapes(const std::vector<Tensor>& tensors) -> std::vector<std::vector<int64_t>>;

/**
 * @brief Make all tensors in a vector contiguous
 *
 * Calls .contiguous() on each tensor and returns the results.
 * Original tensors are not modified.
 *
 * @param tensors Vector of tensors
 * @return Vector of contiguous tensors
 */
auto make_contiguous(const std::vector<Tensor>& tensors) -> std::vector<Tensor>;

} // namespace optim
} // namespace tenzor
