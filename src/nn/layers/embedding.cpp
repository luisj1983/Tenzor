/**
 * @file embedding.cpp
 * @brief Implementation of embedding layers
 */

#include "tenzor/nn/layers/embedding.hpp"
#include "tenzor/core/tensor.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/autograd/function.hpp"
#include "tenzor/backend/fast_dispatch.hpp"
#include "tenzor/backend/op_attributes.hpp"
#include "tenzor/ops/op_id.hpp"
#include <stdexcept>
#include <cmath>
#include <cstring>
#include <iostream>

namespace tenzor {
namespace nn {

// ============================================================================
// EmbeddingBackward - Gradient function for embedding lookup
// ============================================================================

class EmbeddingBackward : public Function {
public:
    EmbeddingBackward(Tensor indices, int64_t num_embeddings, int64_t embedding_dim,
                      int64_t padding_idx = -1)
        : indices_(std::move(indices)),
          num_embeddings_(num_embeddings),
          embedding_dim_(embedding_dim),
          padding_idx_(padding_idx) {}

    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override {
        // inputs[0] = weight matrix [num_embeddings, embedding_dim]
        // Returns embeddings looked up by indices

        if (inputs.empty()) {
            throw std::runtime_error("EmbeddingBackward: No inputs provided");
        }

        const auto& weight = inputs[0];
        const auto& weight_tensor = weight.tensor();

        // GPU fast path: dispatch to backend kernel
        if (weight_tensor.device().type != Device::Type::CPU) {
            Tensor indices_dev = (indices_.device() == weight_tensor.device())
                               ? indices_ : indices_.to(weight_tensor.device());

            std::vector<Tensor> inputs_vec = {weight_tensor, indices_dev};
            auto results = dispatch<OpId::Embedding>(inputs_vec, {});
            return {Variable(results[0], weight.requires_grad())};
        }

        // CPU path: pointer-based lookup
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
            // Convert to Float32 for computation
            auto weight_f32 = weight_tensor.to(DType::Float32);
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

        // GPU fast path: dispatch to backend kernel (CUDA, Vulkan, ROCm, etc.)
        if (grad_output.device().type != Device::Type::CPU) {
            // Ensure indices are on the same device
            Tensor indices_dev = (indices_.device() == grad_output.device())
                               ? indices_ : indices_.to(grad_output.device());

            NewOpAttributes attrs;
            attrs.set(AttrKey::NumEmbeddings, num_embeddings_);
            std::vector<Tensor> inputs_vec = {grad_output, indices_dev};
            auto results = dispatch<OpId::EmbeddingBackward>(inputs_vec, attrs);
            return results;
        }

        // CPU path: pointer-based gradient accumulation
        auto input_ptr = indices_.data<int64_t>();
        int64_t num_indices = indices_.numel();

        // Use grad_output's dtype for gradient
        DType grad_dtype = grad_output.dtype();
        auto grad_weight = zeros({num_embeddings_, embedding_dim_}, grad_dtype);

        // Accumulate gradients for each embedding
        if (grad_dtype == DType::Float32) {
            auto grad_output_ptr = grad_output.data<float>();
            auto grad_weight_ptr = grad_weight.data<float>();
            for (int64_t i = 0; i < num_indices; ++i) {
                auto idx = input_ptr[i];
                for (int64_t j = 0; j < embedding_dim_; ++j) {
                    grad_weight_ptr[idx * embedding_dim_ + j] += grad_output_ptr[i * embedding_dim_ + j];
                }
            }
        } else if (grad_dtype == DType::Float64) {
            auto grad_output_ptr = grad_output.data<double>();
            auto grad_weight_ptr = grad_weight.data<double>();
            for (int64_t i = 0; i < num_indices; ++i) {
                auto idx = input_ptr[i];
                for (int64_t j = 0; j < embedding_dim_; ++j) {
                    grad_weight_ptr[idx * embedding_dim_ + j] += grad_output_ptr[i * embedding_dim_ + j];
                }
            }
        } else if (grad_dtype == DType::Float16) {
            // Convert to Float32 for computation
            auto grad_output_f32 = grad_output.to(DType::Float32);
            auto grad_weight_f32 = zeros({num_embeddings_, embedding_dim_}, DType::Float32);

            auto grad_weight_ptr = grad_weight_f32.data<float>();
            auto grad_output_ptr = grad_output_f32.data<float>();
            for (int64_t i = 0; i < num_indices; ++i) {
                auto idx = input_ptr[i];
                for (int64_t j = 0; j < embedding_dim_; ++j) {
                    grad_weight_ptr[idx * embedding_dim_ + j] += grad_output_ptr[i * embedding_dim_ + j];
                }
            }

            // Convert back to Float16
            grad_weight = grad_weight_f32.to(DType::Float16);
        } else {
            throw std::runtime_error("EmbeddingBackward: Unsupported gradient dtype");
        }

        // Zero out padding_idx row so padding embeddings receive no gradient
        if (padding_idx_ >= 0 && padding_idx_ < num_embeddings_) {
            if (grad_dtype == DType::Float32) {
                auto* ptr = grad_weight.data<float>();
                for (int64_t j = 0; j < embedding_dim_; ++j) {
                    ptr[padding_idx_ * embedding_dim_ + j] = 0.0f;
                }
            } else if (grad_dtype == DType::Float64) {
                auto* ptr = grad_weight.data<double>();
                for (int64_t j = 0; j < embedding_dim_; ++j) {
                    ptr[padding_idx_ * embedding_dim_ + j] = 0.0;
                }
            }
            // Float16 grad_weight was converted from Float32 where padding row
            // was already accumulated — zero it after conversion
            else if (grad_dtype == DType::Float16) {
                auto grad_f32 = grad_weight.to(DType::Float32);
                auto* ptr = grad_f32.data<float>();
                for (int64_t j = 0; j < embedding_dim_; ++j) {
                    ptr[padding_idx_ * embedding_dim_ + j] = 0.0f;
                }
                grad_weight = grad_f32.to(DType::Float16);
            }
        }

        return {grad_weight};
    }

    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override {
        // Embedding backward uses scatter/index operations -- delegate to tensor backward and wrap results
        std::vector<Tensor> tensor_grads;
        for (auto& v : grad_outputs) tensor_grads.push_back(v.tensor());
        auto results = backward(std::move(tensor_grads));
        std::vector<Variable> var_results;
        for (auto& t : results) var_results.emplace_back(t, false);
        return var_results;
    }

private:
    Tensor indices_;
    int64_t num_embeddings_;
    int64_t embedding_dim_;
    int64_t padding_idx_;
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
    if (scale_grad_by_freq) {
        throw std::runtime_error("Embedding: scale_grad_by_freq not implemented");
    }
    if (sparse) {
        throw std::runtime_error("Embedding: sparse not implemented");
    }

    // Initialize embedding weight matrix
    initialize_weights();

    // Register as parameter
    register_parameter("weight", weight_);
}

auto Embedding::initialize_weights() -> void {
    // Initialize with Normal(0, 1)
    weight_ = Variable(randn({num_embeddings_, embedding_dim_}), true);

    // Set padding_idx embedding to zeros if specified (dtype-aware)
    if (padding_idx_ >= 0) {
        DType weight_dtype = weight_.tensor().dtype();
        int64_t row_offset = padding_idx_ * embedding_dim_;

        if (weight_dtype == DType::Float32) {
            auto* ptr = weight_.tensor().data<float>();
            for (int64_t j = 0; j < embedding_dim_; ++j) {
                ptr[row_offset + j] = 0.0f;
            }
        } else if (weight_dtype == DType::Float64) {
            auto* ptr = weight_.tensor().data<double>();
            for (int64_t j = 0; j < embedding_dim_; ++j) {
                ptr[row_offset + j] = 0.0;
            }
        } else if (weight_dtype == DType::Float16) {
            auto* ptr = weight_.tensor().data<Float16>();
            for (int64_t j = 0; j < embedding_dim_; ++j) {
                ptr[row_offset + j] = Float16(0.0f);
            }
        } else if (weight_dtype == DType::BFloat16) {
            auto* ptr = weight_.tensor().data<BFloat16>();
            for (int64_t j = 0; j < embedding_dim_; ++j) {
                ptr[row_offset + j] = BFloat16(0.0f);
            }
        } else {
            // Fallback: use memset for integer and other types
            std::memset(
                static_cast<char*>(weight_.tensor().data_ptr()) + row_offset * weight_.tensor().dtype_size(),
                0,
                embedding_dim_ * weight_.tensor().dtype_size()
            );
        }
    }
}

auto Embedding::forward_impl(const Variable& input) -> Variable {
    // Input shape: any (e.g., [batch, seq_len])
    // Output shape: input.shape() + [embedding_dim]

    const auto& input_tensor = input.tensor();
    auto input_shape = input_tensor.shape();
    auto target_device = input_tensor.device();
    bool is_device_tensor = (target_device.type != Device::Type::CPU);

    // Get weight from parameters_ map to respect offload hooks
    Tensor weight_tensor = parameters_["weight"]->tensor();

    // GPU fast path: dispatch embedding lookup to backend kernel
    if (is_device_tensor) {
        // Ensure weight is on same device as input
        Tensor weight_dev = (weight_tensor.device() == target_device)
                           ? weight_tensor : weight_tensor.to(target_device);
        Tensor indices_dev = input_tensor;

        // Debug mode: validate indices on CPU before GPU dispatch to catch
        // out-of-bounds errors early with clear error messages
#ifndef NDEBUG
        {
            Tensor indices_cpu = input_tensor.to(Device::cpu());
            auto idx_ptr = indices_cpu.data<int64_t>();
            int64_t num_idx = indices_cpu.numel();
            for (int64_t i = 0; i < num_idx; ++i) {
                if (idx_ptr[i] < 0 || idx_ptr[i] >= num_embeddings_) {
                    throw std::out_of_range(
                        "Embedding index out of range: " + std::to_string(idx_ptr[i]) +
                        " not in [0, " + std::to_string(num_embeddings_) + ")");
                }
            }
        }
#endif

        // Apply max_norm if specified (requires CPU for now)
        if (max_norm_ > 0.0) {
            Tensor input_cpu = input_tensor.to(Device::cpu());
            renorm_embeddings(input_cpu);
            weight_dev = parameters_["weight"]->tensor();
            if (weight_dev.device() != target_device) {
                weight_dev = weight_dev.to(target_device);
            }
        }

        std::vector<Tensor> inputs_vec = {weight_dev, indices_dev};
        auto results = dispatch<OpId::Embedding>(inputs_vec, {});
        Tensor output = results[0];

        if (!parameters_["weight"]->requires_grad() || !is_grad_enabled()) {
            return Variable(output, false);
        }

        // Set up autograd for GPU path
        Tensor indices_for_grad = input_tensor;
        auto grad_fn = std::make_shared<EmbeddingBackward>(indices_for_grad, num_embeddings_, embedding_dim_, padding_idx_);

        auto result = Variable(output, true);

        std::vector<std::shared_ptr<Function>> next_funcs;
        next_funcs.push_back(parameters_["weight"]->grad_fn());

        grad_fn->set_next_functions(next_funcs);
        grad_fn->set_input_variables({*parameters_["weight"]});

        result.set_grad_fn(grad_fn);
        return result;
    }

    // CPU path: pointer-based lookup
    Tensor input_cpu = input_tensor;
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
    auto weight_cpu = weight_tensor.device().type == Device::Type::CPU ?
                      weight_tensor : weight_tensor.to(Device::cpu());

    // Check if gradient is needed
    if (!parameters_["weight"]->requires_grad() || !is_grad_enabled()) {
        // No gradient needed, just compute
        std::vector<int64_t> output_shape(input_shape.begin(), input_shape.end());
        output_shape.push_back(embedding_dim_);

        DType weight_dtype = weight_cpu.dtype();
        Tensor output(output_shape, weight_dtype, weight_cpu.device());

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
        } else if (weight_dtype == DType::Float16 || weight_dtype == DType::BFloat16) {
            // Convert to Float32 for computation, convert back
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

            output = output_f32.to(weight_dtype);
        } else {
            throw std::runtime_error("Embedding: Unsupported weight dtype: " +
                                     std::to_string(static_cast<int>(weight_dtype)));
        }

        return Variable(output, false);
    }

    // Use EmbeddingBackward function to preserve gradient graph
    auto grad_fn = std::make_shared<EmbeddingBackward>(input_cpu, num_embeddings_, embedding_dim_, padding_idx_);

    // Perform forward pass (CPU path)
    auto outputs = grad_fn->forward({*parameters_["weight"]});

    if (outputs.empty()) {
        throw std::runtime_error("EmbeddingBackward returned no outputs");
    }

    auto& result = outputs[0];

    // Set up backward graph
    std::vector<std::shared_ptr<Function>> next_funcs;
    next_funcs.push_back(parameters_["weight"]->grad_fn());

    grad_fn->set_next_functions(next_funcs);
    grad_fn->set_input_variables({*parameters_["weight"]});

    result.set_grad_fn(grad_fn);

    return result;
}

auto Embedding::renorm_embeddings(const Tensor& indices) -> void {
    // Renormalize embeddings that exceed max_norm
    // Get weight from parameters_ map to respect offload hooks
    Tensor weight_tensor = parameters_["weight"]->tensor();

    // Transfer to CPU for pointer-based computation
    Device weight_device = weight_tensor.device();
    Device indices_device = indices.device();

    Tensor weight_cpu = (weight_device == Device::cpu()) ? weight_tensor : weight_tensor.to(Device::cpu());
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
    // Update parameters_ map and member variable
    if (weight_device != Device::cpu()) {
        Tensor weight_device_tensor = weight_cpu.to(weight_device);
        auto new_var = std::make_shared<Variable>(weight_device_tensor, parameters_["weight"]->requires_grad());
        parameters_["weight"] = new_var;
        weight_ = *new_var;
    } else if (weight_cpu.data_ptr() != weight_tensor.data_ptr()) {
        // Even on CPU, update if we created a copy
        auto new_var = std::make_shared<Variable>(weight_cpu, parameters_["weight"]->requires_grad());
        parameters_["weight"] = new_var;
        weight_ = *new_var;
    }
}

auto Embedding::weight() -> Variable& {
    // Return from parameters_ to respect offload hooks
    return *parameters_["weight"];
}

auto Embedding::weight() const -> const Variable& {
    // Return from parameters_ to respect offload hooks
    return *parameters_.at("weight");
}

auto Embedding::from_pretrained(const Tensor& embeddings, bool freeze,
                                int64_t padding_idx) -> std::shared_ptr<Embedding> {
    if (embeddings.ndim() != 2) {
        throw std::runtime_error(
            "Embedding.from_pretrained: expected 2D tensor, got " +
            std::to_string(embeddings.ndim()) + "D");
    }
    int64_t num_embeddings = embeddings.shape()[0];
    int64_t embedding_dim = embeddings.shape()[1];

    auto emb = std::make_shared<Embedding>(num_embeddings, embedding_dim, padding_idx);

    // Copy pretrained weights into the embedding
    Tensor weight_copy = embeddings.clone();
    bool requires_grad = !freeze;
    auto weight_var = std::make_shared<Variable>(weight_copy, requires_grad);
    emb->parameters_["weight"] = weight_var;
    emb->weight_ = *weight_var;

    return emb;
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

auto EmbeddingBag::forward_impl(const Variable& input) -> Variable {
    // Default forward: treat entire input as single bag
    return forward(input, Variable{});
}

auto EmbeddingBag::aggregate_embeddings(const Variable& embeddings, const Variable& offsets) -> Variable {
    const auto& emb_tensor = embeddings.tensor();
    auto emb_shape = emb_tensor.shape();

    int64_t total_elements = emb_shape[0];
    int64_t embedding_dim = emb_shape[1];

    // Save original device and dtype, then transfer to CPU Float32 for computation
    Device original_device = emb_tensor.device();
    DType original_dtype = emb_tensor.dtype();
    Tensor emb_cpu = (original_device == Device::cpu()) ? emb_tensor : emb_tensor.to(Device::cpu());
    // Upcast to Float32 for aggregation if needed (avoids precision loss with Float16/BFloat16)
    if (emb_cpu.dtype() != DType::Float32) {
        emb_cpu = emb_cpu.to(DType::Float32);
    }
    auto emb_ptr = emb_cpu.data<float>();

    // If no offsets, aggregate all embeddings into single vector
    if (!offsets.is_initialized() || offsets.tensor().numel() == 0) {
        auto output = zeros({1, embedding_dim}, DType::Float32, Device::cpu());
        auto output_ptr = output.data<float>();

        if (mode_ == "sum" || mode_ == "mean") {
            std::vector<float> compensation(embedding_dim, 0.0f);
            for (int64_t i = 0; i < total_elements; ++i) {
                for (int64_t j = 0; j < embedding_dim; ++j) {
                    float y = emb_ptr[i * embedding_dim + j] - compensation[j];
                    float t = output_ptr[j] + y;
                    compensation[j] = (t - output_ptr[j]) - y;
                    output_ptr[j] = t;
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

        // Convert back to original dtype and transfer to original device
        if (original_dtype != DType::Float32) {
            output = output.to(original_dtype);
        }
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
            std::vector<float> compensation(embedding_dim, 0.0f);
            for (int64_t i = start_idx; i < end_idx; ++i) {
                for (int64_t j = 0; j < embedding_dim; ++j) {
                    float y = emb_ptr[i * embedding_dim + j] - compensation[j];
                    float t = output_ptr[bag * embedding_dim + j] + y;
                    compensation[j] = (t - output_ptr[bag * embedding_dim + j]) - y;
                    output_ptr[bag * embedding_dim + j] = t;
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

    // Convert back to original dtype and transfer to original device
    if (original_dtype != DType::Float32) {
        output = output.to(original_dtype);
    }
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
