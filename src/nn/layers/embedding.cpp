/**
 * @file embedding.cpp
 * @brief Implementation of embedding layers
 */

#include "tenzor/nn/layers/embedding.hpp"
#include "tenzor/core/tensor.hpp"
#include "tenzor/ops/creation.hpp"
#include <stdexcept>
#include <cmath>

namespace tenzor {
namespace nn {

// ============================================================================
// Embedding Implementation
// ============================================================================

Embedding::Embedding(int64_t num_embeddings, int64_t embedding_dim,
                     int64_t padding_idx, double max_norm,
                     double norm_type, bool scale_grad_by_freq,
                     bool sparse)
    : num_embeddings_(num_embeddings),
      embedding_dim_(embedding_dim),
      padding_idx_(padding_idx),
      max_norm_(max_norm),
      norm_type_(norm_type),
      scale_grad_by_freq_(scale_grad_by_freq),
      sparse_(sparse) {

    if (num_embeddings <= 0) {
        throw std::invalid_argument("num_embeddings must be positive");
    }
    if (embedding_dim <= 0) {
        throw std::invalid_argument("embedding_dim must be positive");
    }
    if (padding_idx >= num_embeddings || padding_idx < -1) {
        throw std::invalid_argument("padding_idx must be in range [-1, num_embeddings)");
    }

    // Initialize embedding weight matrix
    initialize_weights();

    // Register as parameter
    register_parameter("weight", weight_);
}

auto Embedding::initialize_weights() -> void {
    // Initialize with Normal(0, 1)
    weight_ = Variable(randn({num_embeddings_, embedding_dim_}), true);

    // Set padding_idx embedding to zeros if specified
    if (padding_idx_ >= 0) {
        auto weight_ptr = weight_.tensor().data<float>();

        // Zero out the row corresponding to padding_idx
        for (int64_t j = 0; j < embedding_dim_; ++j) {
            weight_ptr[padding_idx_ * embedding_dim_ + j] = 0.0f;
        }
    }
}

auto Embedding::forward(const Variable& input) -> Variable {
    // Input shape: any (e.g., [batch, seq_len])
    // Output shape: input.shape() + [embedding_dim]

    const auto& input_tensor = input.tensor();
    auto input_shape = input_tensor.shape();
    auto input_ptr = input_tensor.data<int64_t>();

    // Calculate total number of indices
    int64_t num_indices = 1;
    for (auto dim : input_shape) {
        num_indices *= dim;
    }

    // Validate indices
    for (int64_t i = 0; i < num_indices; ++i) {
        auto idx = input_ptr[i];
        if (idx < 0 || idx >= num_embeddings_) {
            throw std::out_of_range("Index out of range: " + std::to_string(idx));
        }
    }

    // Apply max_norm if specified
    if (max_norm_ > 0.0) {
        renorm_embeddings(input_tensor);
    }

    // Create output shape: input_shape + [embedding_dim]
    std::vector<int64_t> output_shape(input_shape.begin(), input_shape.end());
    output_shape.push_back(embedding_dim_);

    // Perform lookup
    auto output = zeros(output_shape);
    auto output_ptr = output.data<float>();
    auto weight_ptr = weight_.tensor().data<float>();

    for (int64_t i = 0; i < num_indices; ++i) {
        auto idx = input_ptr[i];

        // Copy embedding vector
        for (int64_t j = 0; j < embedding_dim_; ++j) {
            output_ptr[i * embedding_dim_ + j] = weight_ptr[idx * embedding_dim_ + j];
        }
    }

    // Create variable with gradient tracking
    auto result = Variable(output, input.requires_grad() || weight_.requires_grad());

    // TODO: Implement backward pass with padding_idx masking and scale_grad_by_freq

    return result;
}

auto Embedding::renorm_embeddings(const Tensor& indices) -> void {
    // Renormalize embeddings that exceed max_norm
    auto weight_ptr = weight_.tensor().data<float>();
    auto indices_ptr = indices.data<int64_t>();

    int64_t num_indices = indices.numel();

    for (int64_t i = 0; i < num_indices; ++i) {
        auto idx = indices_ptr[i];

        // Skip if this is padding_idx
        if (idx == padding_idx_) {
            continue;
        }

        // Compute norm of embedding vector
        double norm = 0.0;
        for (int64_t j = 0; j < embedding_dim_; ++j) {
            double val = weight_ptr[idx * embedding_dim_ + j];
            if (norm_type_ == 2.0) {
                norm += val * val;
            } else {
                norm += std::pow(std::abs(val), norm_type_);
            }
        }

        if (norm_type_ == 2.0) {
            norm = std::sqrt(norm);
        } else {
            norm = std::pow(norm, 1.0 / norm_type_);
        }

        // Renormalize if exceeds max_norm
        if (norm > max_norm_) {
            double scale = max_norm_ / (norm + 1e-8);
            for (int64_t j = 0; j < embedding_dim_; ++j) {
                weight_ptr[idx * embedding_dim_ + j] *= scale;
            }
        }
    }
}

auto Embedding::weight() -> Variable& {
    return weight_;
}

auto Embedding::weight() const -> const Variable& {
    return weight_;
}

// ============================================================================
// EmbeddingBag Implementation
// ============================================================================

EmbeddingBag::EmbeddingBag(int64_t num_embeddings, int64_t embedding_dim,
                           double max_norm, double norm_type,
                           bool scale_grad_by_freq, const std::string& mode,
                           bool sparse, bool include_last_offset)
    : mode_(mode),
      include_last_offset_(include_last_offset) {

    // Validate mode
    if (mode != "sum" && mode != "mean" && mode != "max") {
        throw std::invalid_argument("mode must be 'sum', 'mean', or 'max'");
    }

    // Create underlying embedding layer
    embedding_ = std::make_shared<Embedding>(
        num_embeddings, embedding_dim, -1, max_norm, norm_type, scale_grad_by_freq, sparse
    );

    // Register as submodule
    register_module("embedding", embedding_);
}

auto EmbeddingBag::forward(const Variable& input, const Variable& offsets) -> Variable {
    // Get embeddings for all indices
    auto embeddings = embedding_->forward(input);

    // Check if offsets is empty/uninitialized
    bool offsets_empty = false;
    try {
        offsets_empty = (offsets.tensor().numel() == 0);
    } catch (...) {
        // Variable not initialized, treat as empty
        offsets_empty = true;
    }

    // If no offsets provided, treat entire input as single bag
    if (offsets_empty) {
        return aggregate_embeddings(embeddings, Variable{});
    }

    // Otherwise, aggregate based on offsets
    return aggregate_embeddings(embeddings, offsets);
}

auto EmbeddingBag::forward(const Variable& input) -> Variable {
    // Default forward: treat entire input as single bag
    return forward(input, Variable{});
}

auto EmbeddingBag::aggregate_embeddings(const Variable& embeddings, const Variable& offsets) -> Variable {
    const auto& emb_tensor = embeddings.tensor();
    auto emb_ptr = emb_tensor.data<float>();
    auto emb_shape = emb_tensor.shape();

    int64_t total_elements = emb_shape[0];
    int64_t embedding_dim = emb_shape[1];

    // If no offsets, aggregate all embeddings into single vector
    if (offsets.tensor().numel() == 0) {
        auto output = zeros({1, embedding_dim});
        auto output_ptr = output.data<float>();

        if (mode_ == "sum" || mode_ == "mean") {
            for (int64_t i = 0; i < total_elements; ++i) {
                for (int64_t j = 0; j < embedding_dim; ++j) {
                    output_ptr[j] += emb_ptr[i * embedding_dim + j];
                }
            }

            if (mode_ == "mean" && total_elements > 0) {
                for (int64_t j = 0; j < embedding_dim; ++j) {
                    output_ptr[j] /= total_elements;
                }
            }
        } else if (mode_ == "max") {
            // Initialize with first embedding
            for (int64_t j = 0; j < embedding_dim; ++j) {
                output_ptr[j] = emb_ptr[j];
            }

            // Find max across all embeddings
            for (int64_t i = 1; i < total_elements; ++i) {
                for (int64_t j = 0; j < embedding_dim; ++j) {
                    float val = emb_ptr[i * embedding_dim + j];
                    if (val > output_ptr[j]) {
                        output_ptr[j] = val;
                    }
                }
            }
        }

        return Variable(output, embeddings.requires_grad());
    }

    // With offsets: aggregate each bag separately
    const auto& offsets_tensor = offsets.tensor();
    auto offsets_ptr = offsets_tensor.data<int64_t>();
    int64_t num_bags = offsets_tensor.numel();

    auto output = zeros({num_bags, embedding_dim});
    auto output_ptr = output.data<float>();

    for (int64_t bag = 0; bag < num_bags; ++bag) {
        int64_t start_idx = offsets_ptr[bag];
        int64_t end_idx;

        if (bag + 1 < num_bags) {
            // Not the last bag: use next offset
            end_idx = offsets_ptr[bag + 1];
        } else if (include_last_offset_ && bag + 1 < offsets_tensor.numel()) {
            // Last bag with include_last_offset and valid index
            end_idx = offsets_ptr[bag + 1];
        } else {
            // Last bag without include_last_offset or no more offsets
            end_idx = total_elements;
        }

        int64_t bag_size = end_idx - start_idx;

        if (bag_size <= 0) {
            continue;
        }

        if (mode_ == "sum" || mode_ == "mean") {
            for (int64_t i = start_idx; i < end_idx; ++i) {
                for (int64_t j = 0; j < embedding_dim; ++j) {
                    output_ptr[bag * embedding_dim + j] += emb_ptr[i * embedding_dim + j];
                }
            }

            if (mode_ == "mean") {
                for (int64_t j = 0; j < embedding_dim; ++j) {
                    output_ptr[bag * embedding_dim + j] /= bag_size;
                }
            }
        } else if (mode_ == "max") {
            // Initialize with first embedding in bag
            for (int64_t j = 0; j < embedding_dim; ++j) {
                output_ptr[bag * embedding_dim + j] = emb_ptr[start_idx * embedding_dim + j];
            }

            // Find max within bag
            for (int64_t i = start_idx + 1; i < end_idx; ++i) {
                for (int64_t j = 0; j < embedding_dim; ++j) {
                    float val = emb_ptr[i * embedding_dim + j];
                    if (val > output_ptr[bag * embedding_dim + j]) {
                        output_ptr[bag * embedding_dim + j] = val;
                    }
                }
            }
        }
    }

    return Variable(output, embeddings.requires_grad());
}

auto EmbeddingBag::weight() -> Variable& {
    return embedding_->weight();
}

auto EmbeddingBag::weight() const -> const Variable& {
    return embedding_->weight();
}

} // namespace nn
} // namespace tenzor
