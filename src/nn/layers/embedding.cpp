/**
 * @file embedding.cpp
 * @brief Implementation of embedding layers
 */

#include "tenzor/nn/layers/embedding.hpp"
#include "tenzor/core/tensor.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/autograd/function.hpp"
#include <stdexcept>
#include <cmath>
#include <iostream>

namespace tenzor {
namespace nn {

// ============================================================================
// EmbeddingBackward - Gradient function for embedding lookup
// ============================================================================

class EmbeddingBackward : public Function {
public:
    EmbeddingBackward(Tensor indices, int64_t num_embeddings, int64_t embedding_dim)
        : indices_(std::move(indices)),
          num_embeddings_(num_embeddings),
          embedding_dim_(embedding_dim) {}

    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override {
        // inputs[0] = weight matrix [num_embeddings, embedding_dim]
        // Returns embeddings looked up by indices

        if (inputs.empty()) {
            throw std::runtime_error("EmbeddingBackward: No inputs provided");
        }

        const auto& weight = inputs[0];
        const auto& weight_tensor = weight.tensor();
        auto input_ptr = indices_.data<int64_t>();

        // Calculate output shape: indices.shape() + [embedding_dim]
        auto indices_shape = indices_.shape();
        std::vector<int64_t> output_shape(indices_shape.begin(), indices_shape.end());
        output_shape.push_back(embedding_dim_);

        // Use weight's dtype for output
        DType weight_dtype = weight_tensor.dtype();
        auto output = zeros(output_shape, weight_dtype);

        int64_t num_indices = indices_.numel();

        // Perform lookup using weight's dtype
        if (weight_dtype == DType::Float32) {
            auto weight_ptr = weight_tensor.data<float>();
            auto output_ptr = output.data<float>();
            for (int64_t i = 0; i < num_indices; ++i) {
                auto idx = input_ptr[i];
                for (int64_t j = 0; j < embedding_dim_; ++j) {
                    output_ptr[i * embedding_dim_ + j] = weight_ptr[idx * embedding_dim_ + j];
                }
            }
        } else if (weight_dtype == DType::Float64) {
            auto weight_ptr = weight_tensor.data<double>();
            auto output_ptr = output.data<double>();
            for (int64_t i = 0; i < num_indices; ++i) {
                auto idx = input_ptr[i];
                for (int64_t j = 0; j < embedding_dim_; ++j) {
                    output_ptr[i * embedding_dim_ + j] = weight_ptr[idx * embedding_dim_ + j];
                }
            }
        } else if (weight_dtype == DType::Float16) {
            std::cerr << "[EMBED_FWD_F16] Starting Float16 forward, num_indices=" << num_indices
                      << ", embedding_dim=" << embedding_dim_ << std::endl;

            // Convert to Float32 for computation
            auto weight_f32 = weight_tensor.to(DType::Float32);
            std::cerr << "[EMBED_FWD_F16] Converted weight to Float32" << std::endl;

            auto output_f32 = zeros(output_shape, DType::Float32);
            std::cerr << "[EMBED_FWD_F16] Created output_f32 with shape [";
            for (size_t i = 0; i < output_shape.size(); ++i) {
                std::cerr << output_shape[i] << (i < output_shape.size()-1 ? "," : "");
            }
            std::cerr << "]" << std::endl;

            auto weight_ptr = weight_f32.data<float>();
            auto output_ptr = output_f32.data<float>();
            for (int64_t i = 0; i < num_indices; ++i) {
                auto idx = input_ptr[i];
                for (int64_t j = 0; j < embedding_dim_; ++j) {
                    output_ptr[i * embedding_dim_ + j] = weight_ptr[idx * embedding_dim_ + j];
                }
            }
            std::cerr << "[EMBED_FWD_F16] Completed lookup loop" << std::endl;

            // Convert back to Float16
            output = output_f32.to(DType::Float16);
            std::cerr << "[EMBED_FWD_F16] Converted output back to Float16" << std::endl;
        } else {
            throw std::runtime_error("EmbeddingBackward: Unsupported weight dtype");
        }

        return {Variable(output, weight.requires_grad())};
    }

    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override {
        // Gradient w.r.t. weight matrix
        // grad_output shape: indices.shape() + [embedding_dim]
        // weight gradient shape: [num_embeddings, embedding_dim]

        if (grad_outputs.empty()) {
            throw std::runtime_error("EmbeddingBackward: No gradient outputs");
        }

        const auto& grad_output = grad_outputs[0];

        // Save original device and transfer to CPU for computation
        Device original_device = grad_output.device();
        Tensor grad_output_cpu = (original_device == Device::cpu()) ?
                                  grad_output : grad_output.to(Device::cpu());

        auto input_ptr = indices_.data<int64_t>();
        int64_t num_indices = indices_.numel();

        // Use grad_output's dtype for gradient
        DType grad_dtype = grad_output.dtype();
        auto grad_weight = zeros({num_embeddings_, embedding_dim_}, grad_dtype);

        // Accumulate gradients for each embedding
        if (grad_dtype == DType::Float32) {
            auto grad_output_ptr = grad_output_cpu.data<float>();
            auto grad_weight_ptr = grad_weight.data<float>();
            for (int64_t i = 0; i < num_indices; ++i) {
                auto idx = input_ptr[i];
                for (int64_t j = 0; j < embedding_dim_; ++j) {
                    grad_weight_ptr[idx * embedding_dim_ + j] += grad_output_ptr[i * embedding_dim_ + j];
                }
            }
        } else if (grad_dtype == DType::Float64) {
            auto grad_output_ptr = grad_output_cpu.data<double>();
            auto grad_weight_ptr = grad_weight.data<double>();
            for (int64_t i = 0; i < num_indices; ++i) {
                auto idx = input_ptr[i];
                for (int64_t j = 0; j < embedding_dim_; ++j) {
                    grad_weight_ptr[idx * embedding_dim_ + j] += grad_output_ptr[i * embedding_dim_ + j];
                }
            }
        } else if (grad_dtype == DType::Float16) {
            std::cerr << "[EMBED_BWD_F16] Starting Float16 backward, num_indices=" << num_indices
                      << ", num_embeddings=" << num_embeddings_
                      << ", embedding_dim=" << embedding_dim_ << std::endl;

            // Convert to Float32 for computation
            auto grad_output_f32 = grad_output_cpu.to(DType::Float32);
            std::cerr << "[EMBED_BWD_F16] Converted grad_output to Float32" << std::endl;

            auto grad_weight_f32 = zeros({num_embeddings_, embedding_dim_}, DType::Float32);
            std::cerr << "[EMBED_BWD_F16] Created grad_weight_f32 with shape ["
                      << num_embeddings_ << "," << embedding_dim_ << "]" << std::endl;

            auto grad_weight_ptr = grad_weight_f32.data<float>();
            auto grad_output_ptr = grad_output_f32.data<float>();
            for (int64_t i = 0; i < num_indices; ++i) {
                auto idx = input_ptr[i];
                for (int64_t j = 0; j < embedding_dim_; ++j) {
                    grad_weight_ptr[idx * embedding_dim_ + j] += grad_output_ptr[i * embedding_dim_ + j];
                }
            }
            std::cerr << "[EMBED_BWD_F16] Completed gradient accumulation loop" << std::endl;

            // Convert back to Float16
            grad_weight = grad_weight_f32.to(DType::Float16);
            std::cerr << "[EMBED_BWD_F16] Converted grad_weight back to Float16" << std::endl;
        } else {
            throw std::runtime_error("EmbeddingBackward: Unsupported gradient dtype");
        }

        // Transfer gradient back to original device if needed
        if (original_device != Device::cpu()) {
            grad_weight = grad_weight.to(original_device);
        }

        return {grad_weight};
    }

private:
    Tensor indices_;
    int64_t num_embeddings_;
    int64_t embedding_dim_;
};

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
    auto target_device = input_tensor.device();

    // For device tensors (OneAPI, CUDA, Vulkan), transfer to CPU for lookup
    // This is a simple implementation; optimized version would use backend dispatch
    bool is_device_tensor = (target_device.type != Device::Type::CPU);

    Tensor input_cpu = is_device_tensor ? input_tensor.to(Device::cpu()) : input_tensor;
    auto input_ptr = input_cpu.data<int64_t>();

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
        renorm_embeddings(input_cpu);
    }

    // Ensure weights are on CPU for lookup
    auto weight_cpu = weight_.tensor().device().type == Device::Type::CPU ?
                      weight_.tensor() : weight_.tensor().to(Device::cpu());

    // Check if gradient is needed
    if (!weight_.requires_grad() || !is_grad_enabled()) {
        // No gradient needed, just compute
        std::vector<int64_t> output_shape(input_shape.begin(), input_shape.end());
        output_shape.push_back(embedding_dim_);

        // Use weight's dtype for output (not Float32 hardcoded)
        DType weight_dtype = weight_cpu.dtype();
        Tensor output(output_shape, weight_dtype, weight_cpu.device());

        // Perform lookup using weight's dtype
        if (weight_dtype == DType::Float32) {
            auto output_ptr = output.data<float>();
            auto weight_ptr = weight_cpu.data<float>();
            for (int64_t i = 0; i < num_indices; ++i) {
                auto idx = input_ptr[i];
                for (int64_t j = 0; j < embedding_dim_; ++j) {
                    output_ptr[i * embedding_dim_ + j] = weight_ptr[idx * embedding_dim_ + j];
                }
            }
        } else if (weight_dtype == DType::Float64) {
            auto output_ptr = output.data<double>();
            auto weight_ptr = weight_cpu.data<double>();
            for (int64_t i = 0; i < num_indices; ++i) {
                auto idx = input_ptr[i];
                for (int64_t j = 0; j < embedding_dim_; ++j) {
                    output_ptr[i * embedding_dim_ + j] = weight_ptr[idx * embedding_dim_ + j];
                }
            }
        } else if (weight_dtype == DType::Float16) {
            // Convert to Float32 for computation
            auto weight_f32 = weight_cpu.to(DType::Float32);
            auto output_f32 = zeros(output_shape, DType::Float32);

            auto weight_ptr = weight_f32.data<float>();
            auto output_ptr = output_f32.data<float>();
            for (int64_t i = 0; i < num_indices; ++i) {
                auto idx = input_ptr[i];
                for (int64_t j = 0; j < embedding_dim_; ++j) {
                    output_ptr[i * embedding_dim_ + j] = weight_ptr[idx * embedding_dim_ + j];
                }
            }

            // Convert back to Float16
            output = output_f32.to(DType::Float16);
        } else {
            throw std::runtime_error("Embedding: Unsupported weight dtype");
        }

        // Transfer back to target device if needed
        if (is_device_tensor) {
            output = output.to(target_device);
        }

        return Variable(output, false);
    }

    // Use EmbeddingBackward function to preserve gradient graph
    // Note: For device tensors, this uses CPU tensors internally
    auto grad_fn = std::make_shared<EmbeddingBackward>(input_cpu, num_embeddings_, embedding_dim_);

    // Create temporary weight variable on CPU if needed
    Variable weight_for_lookup = weight_.tensor().device().type == Device::Type::CPU ?
                                 weight_ : Variable(weight_cpu, weight_.requires_grad());

    // Perform forward pass
    auto outputs = grad_fn->forward({weight_for_lookup});

    if (outputs.empty()) {
        throw std::runtime_error("EmbeddingBackward returned no outputs");
    }

    auto& result = outputs[0];

    // Transfer result back to target device if needed
    if (is_device_tensor) {
        Tensor result_device = result.tensor().to(target_device);
        result = Variable(result_device, result.requires_grad());
    }

    // Set up backward graph
    std::vector<std::shared_ptr<Function>> next_funcs;
    next_funcs.push_back(weight_.grad_fn());  // nullptr if weight is leaf

    grad_fn->set_next_functions(next_funcs);
    grad_fn->set_input_variables({weight_});

    result.set_grad_fn(grad_fn);

    return result;
}

auto Embedding::renorm_embeddings(const Tensor& indices) -> void {
    // Renormalize embeddings that exceed max_norm
    // Transfer to CPU for pointer-based computation
    Device weight_device = weight_.tensor().device();
    Device indices_device = indices.device();

    Tensor weight_cpu = (weight_device == Device::cpu()) ? weight_.tensor() : weight_.tensor().to(Device::cpu());
    Tensor indices_cpu = (indices_device == Device::cpu()) ? indices : indices.to(Device::cpu());

    auto indices_ptr = indices_cpu.data<int64_t>();
    int64_t num_indices = indices_cpu.numel();

    DType weight_dtype = weight_cpu.dtype();

    if (weight_dtype == DType::Float32) {
        auto weight_ptr = weight_cpu.data<float>();
        for (int64_t i = 0; i < num_indices; ++i) {
            auto idx = indices_ptr[i];
            if (idx == padding_idx_) continue;

            // Compute norm
            double norm = 0.0;
            for (int64_t j = 0; j < embedding_dim_; ++j) {
                double val = weight_ptr[idx * embedding_dim_ + j];
                if (norm_type_ == 2.0) {
                    norm += val * val;
                } else {
                    norm += std::pow(std::abs(val), norm_type_);
                }
            }
            norm = (norm_type_ == 2.0) ? std::sqrt(norm) : std::pow(norm, 1.0 / norm_type_);

            // Renormalize if needed
            if (norm > max_norm_) {
                double scale = max_norm_ / (norm + 1e-8);
                for (int64_t j = 0; j < embedding_dim_; ++j) {
                    weight_ptr[idx * embedding_dim_ + j] *= scale;
                }
            }
        }
    } else if (weight_dtype == DType::Float64) {
        auto weight_ptr = weight_cpu.data<double>();
        for (int64_t i = 0; i < num_indices; ++i) {
            auto idx = indices_ptr[i];
            if (idx == padding_idx_) continue;

            // Compute norm
            double norm = 0.0;
            for (int64_t j = 0; j < embedding_dim_; ++j) {
                double val = weight_ptr[idx * embedding_dim_ + j];
                if (norm_type_ == 2.0) {
                    norm += val * val;
                } else {
                    norm += std::pow(std::abs(val), norm_type_);
                }
            }
            norm = (norm_type_ == 2.0) ? std::sqrt(norm) : std::pow(norm, 1.0 / norm_type_);

            // Renormalize if needed
            if (norm > max_norm_) {
                double scale = max_norm_ / (norm + 1e-8);
                for (int64_t j = 0; j < embedding_dim_; ++j) {
                    weight_ptr[idx * embedding_dim_ + j] *= scale;
                }
            }
        }
    } else {
        throw std::runtime_error("Embedding::renorm_embeddings: Unsupported weight dtype");
    }

    // Transfer modified weights back to original device if needed
    if (weight_device != Device::cpu()) {
        Tensor weight_device_tensor = weight_cpu.to(weight_device);
        weight_ = Variable(weight_device_tensor, weight_.requires_grad());
        if (weight_.grad_fn()) {
            weight_.set_grad_fn(weight_.grad_fn());
        }
    } else if (weight_cpu.data_ptr() != weight_.tensor().data_ptr()) {
        // Even on CPU, update if we created a copy
        weight_ = Variable(weight_cpu, weight_.requires_grad());
        if (weight_.grad_fn()) {
            weight_.set_grad_fn(weight_.grad_fn());
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
        offsets_empty = (!offsets.is_initialized() || offsets.tensor().numel() == 0);
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
    auto emb_shape = emb_tensor.shape();

    int64_t total_elements = emb_shape[0];
    int64_t embedding_dim = emb_shape[1];

    // Save original device and transfer to CPU for pointer-based computation
    Device original_device = emb_tensor.device();
    Tensor emb_cpu = (original_device == Device::cpu()) ? emb_tensor : emb_tensor.to(Device::cpu());
    auto emb_ptr = emb_cpu.data<float>();

    // If no offsets, aggregate all embeddings into single vector
    if (!offsets.is_initialized() || offsets.tensor().numel() == 0) {
        auto output = zeros({1, embedding_dim}, DType::Float32, Device::cpu());
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

        // Transfer output back to original device if needed
        if (original_device != Device::cpu()) {
            output = output.to(original_device);
        }

        return Variable(output, embeddings.requires_grad());
    }

    // With offsets: aggregate each bag separately
    const auto& offsets_tensor = offsets.tensor();
    Tensor offsets_cpu = (original_device == Device::cpu()) ? offsets_tensor : offsets_tensor.to(Device::cpu());
    auto offsets_ptr = offsets_cpu.data<int64_t>();
    int64_t num_bags = offsets_cpu.numel();

    auto output = zeros({num_bags, embedding_dim}, DType::Float32, Device::cpu());
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

    // Transfer output back to original device if needed
    if (original_device != Device::cpu()) {
        output = output.to(original_device);
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
