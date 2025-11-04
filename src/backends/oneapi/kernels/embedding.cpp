#include "tenzor/core/tensor.hpp"
#include <sycl/sycl.hpp>
#include <stdexcept>
#include <cmath>

namespace tenzor {
namespace oneapi {

// Kernel class declarations for SYCL
class EmbeddingLookupKernel;
class EmbeddingBackwardKernel;

// Helper function to get typed pointer from tensor
template<typename T>
inline auto get_data_ptr(const Tensor& t) -> T* {
    return static_cast<T*>(const_cast<void*>(t.data_ptr()));
}

/**
 * @brief Embedding lookup kernel for OneAPI/SYCL
 *
 * Given indices tensor and weight matrix, return corresponding embeddings.
 * Supports batch processing and multi-dimensional indices.
 *
 * @param indices Tensor of shape [batch, seq_len] or any shape containing int64 indices
 * @param weights Embedding weight matrix of shape [vocab_size, embedding_dim]
 * @param padding_idx Special index to fill with zeros (use -1 to disable)
 * @param queue SYCL queue for execution
 * @return Output embeddings of shape indices.shape() + [embedding_dim]
 */
auto embedding_lookup_kernel(const Tensor& indices, const Tensor& weights,
                             int64_t padding_idx, sycl::queue& queue) -> Tensor {
    // Get shapes
    auto indices_shape_span = indices.shape();
    auto weights_shape_span = weights.shape();

    std::vector<int64_t> indices_shape(indices_shape_span.begin(), indices_shape_span.end());
    std::vector<int64_t> weights_shape(weights_shape_span.begin(), weights_shape_span.end());

    int64_t num_indices = indices.numel();
    int64_t vocab_size = weights_shape[0];
    int64_t embedding_dim = weights_shape[1];

    // Output shape: indices.shape() + [embedding_dim]
    std::vector<int64_t> output_shape = indices_shape;
    output_shape.push_back(embedding_dim);

    // Create output tensor
    Tensor output(output_shape, weights.dtype(), weights.device());

    // Get device pointers
    const int64_t* indices_ptr = get_data_ptr<const int64_t>(indices);
    const float* weights_ptr = get_data_ptr<const float>(weights);
    float* output_ptr = get_data_ptr<float>(output);

    // Launch kernel: one work item per (index, embedding_dim) pair
    queue.parallel_for<EmbeddingLookupKernel>(
        sycl::range<2>(num_indices, embedding_dim),
        [=](sycl::id<2> idx) {
            int64_t index_idx = idx[0];
            int64_t emb_dim_idx = idx[1];

            // Get the vocabulary index
            int64_t vocab_idx = indices_ptr[index_idx];

            // Handle padding index
            if (vocab_idx == padding_idx) {
                output_ptr[index_idx * embedding_dim + emb_dim_idx] = 0.0f;
            } else {
                // Bounds checking
                if (vocab_idx < 0 || vocab_idx >= vocab_size) {
                    // Can't throw in device code, just clamp
                    vocab_idx = 0;
                }

                // Perform lookup
                output_ptr[index_idx * embedding_dim + emb_dim_idx] =
                    weights_ptr[vocab_idx * embedding_dim + emb_dim_idx];
            }
        }
    ).wait();

    return output;
}

/**
 * @brief Embedding backward kernel for OneAPI/SYCL
 *
 * Computes gradient w.r.t. embedding weights given gradient w.r.t. output.
 *
 * @param grad_output Gradient of shape indices.shape() + [embedding_dim]
 * @param indices Original indices used in forward pass
 * @param vocab_size Number of embeddings in vocabulary
 * @param embedding_dim Dimension of each embedding
 * @param queue SYCL queue for execution
 * @return Gradient w.r.t. weights of shape [vocab_size, embedding_dim]
 */
auto embedding_backward_kernel(const Tensor& grad_output, const Tensor& indices,
                               int64_t vocab_size, int64_t embedding_dim,
                               sycl::queue& queue) -> Tensor {
    // Create gradient tensor initialized to zeros
    Tensor grad_weight({vocab_size, embedding_dim}, grad_output.dtype(), grad_output.device());

    // Initialize to zero
    float* grad_weight_ptr = get_data_ptr<float>(grad_weight);
    int64_t total_weight_elements = vocab_size * embedding_dim;

    queue.parallel_for<class ZeroInitKernel>(
        sycl::range<1>(total_weight_elements),
        [=](sycl::id<1> idx) {
            grad_weight_ptr[idx] = 0.0f;
        }
    ).wait();

    // Get pointers
    const int64_t* indices_ptr = get_data_ptr<const int64_t>(indices);
    const float* grad_output_ptr = get_data_ptr<const float>(grad_output);

    int64_t num_indices = indices.numel();

    // Accumulate gradients
    // Use atomic operations for race-free accumulation
    queue.parallel_for<EmbeddingBackwardKernel>(
        sycl::range<2>(num_indices, embedding_dim),
        [=](sycl::id<2> idx) {
            int64_t index_idx = idx[0];
            int64_t emb_dim_idx = idx[1];

            int64_t vocab_idx = indices_ptr[index_idx];

            // Bounds check
            if (vocab_idx >= 0 && vocab_idx < vocab_size) {
                // Atomic add to handle multiple indices pointing to same embedding
                float grad_val = grad_output_ptr[index_idx * embedding_dim + emb_dim_idx];

                // Use atomic_ref properly with a reference
                sycl::atomic_ref<float,
                                sycl::memory_order::relaxed,
                                sycl::memory_scope::device,
                                sycl::access::address_space::global_space>
                    atomic_grad(grad_weight_ptr[vocab_idx * embedding_dim + emb_dim_idx]);

                atomic_grad.fetch_add(grad_val);
            }
        }
    ).wait();

    return grad_weight;
}

/**
 * @brief EmbeddingBag forward kernel - aggregate embeddings
 *
 * Computes sum, mean, or max aggregation of embeddings for bags of indices.
 *
 * @param embeddings Pre-computed embeddings from embedding_lookup
 * @param offsets Starting index of each bag in the indices tensor
 * @param mode Aggregation mode: "sum", "mean", or "max"
 * @param include_last_offset Whether offsets includes final boundary
 * @param queue SYCL queue for execution
 * @return Aggregated embeddings of shape [num_bags, embedding_dim]
 */
auto embedding_bag_forward_kernel(const Tensor& embeddings, const Tensor& offsets,
                                  const std::string& mode, bool include_last_offset,
                                  sycl::queue& queue) -> Tensor {
    auto embeddings_shape = embeddings.shape();
    int64_t total_elements = embeddings_shape[0];
    int64_t embedding_dim = embeddings_shape[1];

    int64_t num_bags = offsets.numel();

    // Create output tensor
    Tensor output({num_bags, embedding_dim}, embeddings.dtype(), embeddings.device());

    // Get pointers
    const float* embeddings_ptr = get_data_ptr<const float>(embeddings);
    const int64_t* offsets_ptr = get_data_ptr<const int64_t>(offsets);
    float* output_ptr = get_data_ptr<float>(output);

    // Convert mode string to enum for device copyability
    // 0 = sum, 1 = mean, 2 = max
    int mode_enum = 1; // default mean
    if (mode == "sum") mode_enum = 0;
    else if (mode == "mean") mode_enum = 1;
    else if (mode == "max") mode_enum = 2;

    // Process each bag in parallel
    queue.parallel_for<class EmbeddingBagKernel>(
        sycl::range<2>(num_bags, embedding_dim),
        [=](sycl::id<2> idx) {
            int64_t bag_idx = idx[0];
            int64_t emb_dim_idx = idx[1];

            int64_t start_idx = offsets_ptr[bag_idx];
            int64_t end_idx;

            if (bag_idx + 1 < num_bags) {
                end_idx = offsets_ptr[bag_idx + 1];
            } else {
                end_idx = total_elements;
            }

            int64_t bag_size = end_idx - start_idx;

            if (bag_size <= 0) {
                output_ptr[bag_idx * embedding_dim + emb_dim_idx] = 0.0f;
                return;
            }

            float result = 0.0f;
            bool first = true;

            for (int64_t i = start_idx; i < end_idx; ++i) {
                float val = embeddings_ptr[i * embedding_dim + emb_dim_idx];

                if (mode_enum == 0 || mode_enum == 1) { // sum or mean
                    result += val;
                } else if (mode_enum == 2) { // max
                    if (first || val > result) {
                        result = val;
                        first = false;
                    }
                }
            }

            // Apply mean normalization if needed
            if (mode_enum == 1 && bag_size > 0) { // mean
                result /= bag_size;
            }

            output_ptr[bag_idx * embedding_dim + emb_dim_idx] = result;
        }
    ).wait();

    return output;
}

/**
 * @brief Renormalize embeddings that exceed max_norm
 *
 * For each unique index in indices, if the corresponding embedding's norm
 * exceeds max_norm, scale it down to have norm equal to max_norm.
 *
 * @param weights Embedding weight matrix [vocab_size, embedding_dim]
 * @param indices Indices to check and renormalize
 * @param max_norm Maximum allowed norm
 * @param norm_type Type of norm (1.0 for L1, 2.0 for L2)
 * @param queue SYCL queue for execution
 */
auto embedding_renorm_kernel(Tensor& weights, const Tensor& indices,
                             double max_norm, double norm_type,
                             sycl::queue& queue) -> void {
    // This is a simplified version - for efficiency, should deduplicate indices first
    // For now, we'll process on CPU to handle the complexity

    auto weights_cpu = weights.to(Device::cpu());
    auto indices_cpu = indices.to(Device::cpu());

    auto weights_shape = weights_cpu.shape();
    int64_t embedding_dim = weights_shape[1];

    float* weights_ptr = get_data_ptr<float>(weights_cpu);
    const int64_t* indices_ptr = get_data_ptr<const int64_t>(indices_cpu);
    int64_t num_indices = indices.numel();

    // Track which indices we've already processed
    std::vector<bool> processed(weights_shape[0], false);

    for (int64_t i = 0; i < num_indices; ++i) {
        int64_t vocab_idx = indices_ptr[i];

        if (vocab_idx < 0 || vocab_idx >= weights_shape[0] || processed[vocab_idx]) {
            continue;
        }

        processed[vocab_idx] = true;

        // Compute norm
        double norm = 0.0;
        for (int64_t j = 0; j < embedding_dim; ++j) {
            double val = weights_ptr[vocab_idx * embedding_dim + j];
            if (norm_type == 2.0) {
                norm += val * val;
            } else {
                norm += std::pow(std::abs(val), norm_type);
            }
        }

        if (norm_type == 2.0) {
            norm = std::sqrt(norm);
        } else {
            norm = std::pow(norm, 1.0 / norm_type);
        }

        // Renormalize if exceeds max_norm
        if (norm > max_norm) {
            double scale = max_norm / (norm + 1e-8);
            for (int64_t j = 0; j < embedding_dim; ++j) {
                weights_ptr[vocab_idx * embedding_dim + j] *= scale;
            }
        }
    }

    // Copy back to device
    float* device_weights_ptr = get_data_ptr<float>(weights);
    const float* host_weights_ptr = get_data_ptr<const float>(weights_cpu);
    queue.memcpy(device_weights_ptr, host_weights_ptr,
                 weights.numel() * sizeof(float)).wait();
}

} // namespace oneapi
} // namespace tenzor
