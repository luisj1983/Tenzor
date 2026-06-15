/**
 * @file embedding.cpp
 * @brief Implementation of embedding layers
 */

#include "tenzor/nn/layers/embedding.hpp"
#include "tenzor/core/tensor.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/transform.hpp"
#include "tenzor/ops/reduction.hpp"
#include "tenzor/ops/indexing.hpp"
#include "tenzor/autograd/function.hpp"
#include "tenzor/backend/fast_dispatch.hpp"
#include "tenzor/backend/op_attributes.hpp"
#include "tenzor/jit/tracer.hpp"

namespace {

/// Record an `Embedding` op into the active JIT trace, if any. The CPU
/// fast path of Embedding::forward_impl rewrites memory directly
/// instead of dispatching through `OpId::Embedding`, so the dispatch
/// interceptor never sees it. Calling this from inside the CPU forward
/// gives the tracer the (weight, indices) -> output edge it needs to
/// reproduce the lookup in lowered MLIR.
inline auto jit_record_embedding(const ::tenzor::Tensor& weight,
                                 const ::tenzor::Tensor& indices,
                                 const ::tenzor::Tensor& output) -> void {
    auto& tracer = ::tenzor::jit::Tracer::get_instance();
    if (!tracer.is_tracing()) return;
    auto w_id = tracer.register_tensor(weight);
    auto i_id = tracer.register_tensor(indices);
    auto o_id = tracer.register_new_tensor(output);
    ::tenzor::jit::TracedOp op(::tenzor::jit::OpType::Embedding,
                               {w_id, i_id}, {o_id});
    tracer.record_op(std::move(op));
}

}  // namespace
#include "tenzor/ops/op_id.hpp"
#include "tenzor/sparse/sparse_tensor.hpp"
#include "tenzor/sparse/sparse_ops.hpp"
#include <array>
#include <stdexcept>
#include <cmath>
#include <cstring>
#include <iostream>
#include <thread>

namespace tenzor {
namespace nn {

// ============================================================================
// EmbeddingBackward - Gradient function for embedding lookup
// ============================================================================

class EmbeddingBackward : public Function {
public:
    EmbeddingBackward(Tensor indices, int64_t num_embeddings, int64_t embedding_dim,
                      int64_t padding_idx = -1, bool scale_grad_by_freq = false,
                      bool sparse = false)
        : indices_(std::move(indices)),
          num_embeddings_(num_embeddings),
          embedding_dim_(embedding_dim),
          padding_idx_(padding_idx),
          scale_grad_by_freq_(scale_grad_by_freq),
          sparse_(sparse) {}

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

        for (int64_t i = 0; i < num_indices; ++i) {
            auto idx = input_ptr[i];
            if (idx < 0 || idx >= num_embeddings_) {
                throw std::out_of_range("Embedding index " + std::to_string(idx) +
                                        " out of range [0, " + std::to_string(num_embeddings_) + ")");
            }
        }

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
        } else if (weight_dtype == DType::Float16 || weight_dtype == DType::BFloat16) {
            // Widen to Float32 for the gather, then narrow back to the original
            // half-precision dtype.
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

            output = output_f32.to(weight_dtype);
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

            // R.19: GPU path previously returned only the dense gradient and
            // never populated weight.sparse_grad, so sparse-aware optimisers
            // (SparseAdam, etc.) saw an empty sparse buffer and silently fell
            // back to dense updates. Mirror the CPU branch below: build a COO
            // SparseTensor from indices_ + the dense grad rows and accumulate
            // it onto the weight Variable.
            if (sparse_ && !input_variables_.empty()) {
                // Indices come back to CPU for the SparseTensor; matches the
                // CPU branch's COO production. The optimiser side decides
                // how/when to move them back to device.
                Tensor indices_cpu = (indices_.device() == Device::cpu())
                                       ? indices_ : indices_.to(Device::cpu());
                int64_t num_indices = indices_cpu.numel();

                auto idx_tensor = zeros({1, num_indices}, DType::Int64);
                auto* idx_ptr = idx_tensor.data<int64_t>();
                auto* src_ptr = indices_cpu.data<int64_t>();
                for (int64_t i = 0; i < num_indices; ++i) {
                    idx_ptr[i] = src_ptr[i];
                }

                Tensor grad_values = grad_output.to(Device::cpu())
                                                .reshape({num_indices, embedding_dim_});

                auto sparse_grad = SparseTensor::sparse_coo(
                    idx_tensor, grad_values, {num_embeddings_, embedding_dim_});

                auto& weight_var = input_variables_[0];
                weight_var.accumulate_sparse_grad(sparse_grad);
            }

            return results;
        }

        // CPU path
        auto input_ptr = indices_.data<int64_t>();
        int64_t num_indices = indices_.numel();

        for (int64_t i = 0; i < num_indices; ++i) {
            auto idx = input_ptr[i];
            if (idx < 0 || idx >= num_embeddings_) {
                throw std::out_of_range("Embedding index " + std::to_string(idx) +
                                        " out of range [0, " + std::to_string(num_embeddings_) + ")");
            }
        }

        // Sparse gradient path: build COO sparse tensor.
        // The sparse gradient is stored directly on the weight Variable via
        // set_sparse_grad() for sparse-aware optimizers (e.g., SparseAdam).
        // A dense version is also returned for compatibility with the dense engine.
        if (sparse_) {
            // indices shape: [1, num_indices] (row=embedding_idx)
            // values shape: [num_indices, embedding_dim]
            auto idx_tensor = zeros({1, num_indices}, DType::Int64);
            auto* idx_ptr = idx_tensor.data<int64_t>();
            for (int64_t i = 0; i < num_indices; ++i) {
                idx_ptr[i] = input_ptr[i];
            }
            // Reshape grad_output to [num_indices, embedding_dim]
            auto grad_values = grad_output.reshape({num_indices, embedding_dim_});
            auto sparse_grad = SparseTensor::sparse_coo(
                idx_tensor, grad_values, {num_embeddings_, embedding_dim_});

            // Accumulate sparse gradient on the weight variable for sparse-aware optimizers.
            // When the same embedding is used in multiple forward paths, backward is
            // called once per path — we must accumulate, not overwrite, to get the
            // correct total gradient.
            // accumulate_sparse_grad() acquires grad_mutex_ internally for thread safety.
            auto& weight_var = input_variables_[0];
            weight_var.accumulate_sparse_grad(sparse_grad);

            // Convert to dense for the standard backward engine
            auto grad_weight = sparse_grad.to_dense();

            // Apply scale_grad_by_freq if needed
            if (scale_grad_by_freq_) {
                auto freq = zeros({num_embeddings_}, DType::Float32);
                auto freq_ptr = freq.data<float>();
                for (int64_t i = 0; i < num_indices; ++i) {
                    freq_ptr[input_ptr[i]] += 1.0f;
                }
                DType grad_dtype = grad_output.dtype();
                if (grad_dtype == DType::Float32) {
                    auto* gw_ptr = grad_weight.data<float>();
                    for (int64_t r = 0; r < num_embeddings_; ++r) {
                        if (freq_ptr[r] > 0.0f) {
                            float inv = 1.0f / freq_ptr[r];
                            for (int64_t j = 0; j < embedding_dim_; ++j)
                                gw_ptr[r * embedding_dim_ + j] *= inv;
                        }
                    }
                } else if (grad_dtype == DType::Float64) {
                    auto* gw_ptr = grad_weight.data<double>();
                    for (int64_t r = 0; r < num_embeddings_; ++r) {
                        if (freq_ptr[r] > 0.0f) {
                            double inv = 1.0 / static_cast<double>(freq_ptr[r]);
                            for (int64_t j = 0; j < embedding_dim_; ++j)
                                gw_ptr[r * embedding_dim_ + j] *= inv;
                        }
                    }
                }
            }

            // Zero padding row
            if (padding_idx_ >= 0 && padding_idx_ < num_embeddings_) {
                DType grad_dtype = grad_output.dtype();
                if (grad_dtype == DType::Float32) {
                    auto* ptr = grad_weight.data<float>();
                    for (int64_t j = 0; j < embedding_dim_; ++j)
                        ptr[padding_idx_ * embedding_dim_ + j] = 0.0f;
                } else if (grad_dtype == DType::Float64) {
                    auto* ptr = grad_weight.data<double>();
                    for (int64_t j = 0; j < embedding_dim_; ++j)
                        ptr[padding_idx_ * embedding_dim_ + j] = 0.0;
                }
            }
            return {grad_weight};
        }

        // Dense gradient accumulation path

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
        } else if (grad_dtype == DType::Float16 || grad_dtype == DType::BFloat16) {
            // Widen to Float32 for accumulation, then narrow back to the
            // original half-precision dtype.
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

            grad_weight = grad_weight_f32.to(grad_dtype);
        } else {
            throw std::runtime_error("EmbeddingBackward: Unsupported gradient dtype");
        }

        // Scale gradients by inverse frequency of each index
        if (scale_grad_by_freq_) {
            // Count frequency of each index
            auto freq = zeros({num_embeddings_}, DType::Float32);
            auto freq_ptr = freq.data<float>();
            for (int64_t i = 0; i < num_indices; ++i) {
                freq_ptr[input_ptr[i]] += 1.0f;
            }
            // Divide each row of grad_weight by its frequency (skip zero-freq rows)
            if (grad_dtype == DType::Float32) {
                auto* gw_ptr = grad_weight.data<float>();
                for (int64_t r = 0; r < num_embeddings_; ++r) {
                    if (freq_ptr[r] > 0.0f) {
                        float inv_freq = 1.0f / freq_ptr[r];
                        for (int64_t j = 0; j < embedding_dim_; ++j) {
                            gw_ptr[r * embedding_dim_ + j] *= inv_freq;
                        }
                    }
                }
            } else if (grad_dtype == DType::Float64) {
                auto* gw_ptr = grad_weight.data<double>();
                for (int64_t r = 0; r < num_embeddings_; ++r) {
                    if (freq_ptr[r] > 0.0f) {
                        double inv_freq = 1.0 / static_cast<double>(freq_ptr[r]);
                        for (int64_t j = 0; j < embedding_dim_; ++j) {
                            gw_ptr[r * embedding_dim_ + j] *= inv_freq;
                        }
                    }
                }
            } else if (grad_dtype == DType::Float16) {
                // grad_weight was computed in Float32 and converted — reconvert for scaling
                auto gw_f32 = grad_weight.to(DType::Float32);
                auto* gw_ptr = gw_f32.data<float>();
                for (int64_t r = 0; r < num_embeddings_; ++r) {
                    if (freq_ptr[r] > 0.0f) {
                        float inv_freq = 1.0f / freq_ptr[r];
                        for (int64_t j = 0; j < embedding_dim_; ++j) {
                            gw_ptr[r * embedding_dim_ + j] *= inv_freq;
                        }
                    }
                }
                grad_weight = gw_f32.to(DType::Float16);
            }
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
        // Embedding backward is index-based scatter (non-differentiable indices),
        // so higher-order derivative through the scatter is zero.
        // Delegate to tensor-level backward and wrap with requires_grad=true
        // so the gradient of the gradient output is tracked.
        std::vector<Tensor> tensor_grads;
        tensor_grads.reserve(grad_outputs.size());
        for (auto& v : grad_outputs) tensor_grads.push_back(v.tensor());
        auto results = backward(std::move(tensor_grads));
        std::vector<Variable> var_results;
        var_results.reserve(results.size());
        // Input gradient (no grad needed - indices are non-differentiable)
        // Weight gradient (tracked for higher-order)
        for (size_t i = 0; i < results.size(); ++i) {
            var_results.emplace_back(results[i], i > 0 && grad_outputs[0].requires_grad());
        }
        return var_results;
    }

    // P4.2d: Embedding is a scatter by non-differentiable indices —
    // the second derivative through it is structurally zero.
    auto supports_higher_order() const -> bool override { return true; }
    auto is_higher_order_stub() const -> bool override { return true; }

private:
    Tensor indices_;
    int64_t num_embeddings_;
    int64_t embedding_dim_;
    int64_t padding_idx_;
    bool scale_grad_by_freq_;
    bool sparse_;
};

// ============================================================================
// EmbeddingBagBackward - Gradient function for EmbeddingBag aggregation
// ============================================================================

class EmbeddingBagBackward : public Function {
public:
    EmbeddingBagBackward(Tensor bag_indices, Tensor offsets, bool has_offsets,
                         std::string mode,
                         int64_t total_elements, int64_t embedding_dim,
                         bool include_last_offset)
        : bag_indices_(std::move(bag_indices)),
          offsets_(std::move(offsets)),
          has_offsets_(has_offsets),
          mode_(std::move(mode)),
          total_elements_(total_elements),
          embedding_dim_(embedding_dim),
          include_last_offset_(include_last_offset) {
        // Stash the original vocabulary indices so the OpId::EmbeddingBagBackward
        // dispatch path (which now takes indices, not the looked-up rows) can
        // recover them. The current backward() loop returns grad-w.r.t-embeddings
        // and doesn't use bag_indices_ directly — Embedding's own backward
        // scatters that grad to the weight matrix using its own saved indices.
        // save_for_backward keeps version-counter book-keeping consistent.
        if (bag_indices_.is_valid()) {
            save_for_backward({bag_indices_});
        }
    }

    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override {
        // inputs[0] = embeddings [total_elements, embedding_dim]
        if (inputs.empty()) {
            throw std::runtime_error("EmbeddingBagBackward: No inputs provided");
        }

        const auto& emb_tensor = inputs[0].tensor();

        // Save original device and dtype for backward
        Device original_device = emb_tensor.device();
        DType original_dtype = emb_tensor.dtype();

        // Determine bag boundaries (needed for backward regardless of path)
        int64_t num_bags;
        Tensor offsets_cpu;
        const int64_t* offsets_ptr = nullptr;

        if (has_offsets_) {
            offsets_cpu = (offsets_.device() == Device::cpu()) ? offsets_ : offsets_.to(Device::cpu());
            offsets_ptr = offsets_cpu.data<int64_t>();
            num_bags = offsets_cpu.numel();
        } else {
            num_bags = 1;
        }

        std::vector<int64_t> bag_starts(num_bags);
        std::vector<int64_t> bag_ends(num_bags);
        for (int64_t bag = 0; bag < num_bags; ++bag) {
            if (has_offsets_) {
                bag_starts[bag] = offsets_ptr[bag];
                if (bag + 1 < num_bags) {
                    bag_ends[bag] = offsets_ptr[bag + 1];
                } else if (include_last_offset_ && bag + 1 < offsets_.numel()) {
                    bag_ends[bag] = offsets_ptr[bag + 1];
                } else {
                    bag_ends[bag] = total_elements_;
                }
            } else {
                bag_starts[bag] = 0;
                bag_ends[bag] = total_elements_;
            }
        }

        // Save bag boundaries for backward
        bag_starts_ = std::move(bag_starts);
        bag_ends_ = std::move(bag_ends);
        num_bags_ = num_bags;
        original_device_ = original_device;
        original_dtype_ = original_dtype;

        // Dispatch to backend kernel if available (avoids CPU roundtrip on GPU).
        // Forward returns {output, max_indices}: for mode="max", max_indices holds
        // the per-(bag, feature) GLOBAL argmax element index. Backward uses it to
        // route the gradient exactly via an on-device scatter — no CPU recompute,
        // works on every device.
        if (is_op_supported(OpId::EmbeddingBagForward, original_device.type)) {
            OpAttributes bag_attrs;
            bag_attrs.set(AttrKey::Mode, mode_);
            bag_attrs.set(AttrKey::EmbeddingDim, embedding_dim_);
            bag_attrs.set(AttrKey::IncludeLastOffset, include_last_offset_);

            Tensor offsets_dev;
            if (has_offsets_) {
                offsets_dev = (offsets_.device() == original_device) ? offsets_ : offsets_.to(original_device);
            } else {
                offsets_dev = zeros({1}, DType::Int64, original_device);
            }

            std::array<Tensor, 2> bag_inputs = {emb_tensor, offsets_dev};
            auto result = dispatch<OpId::EmbeddingBagForward>(bag_inputs, bag_attrs);
            if (result.empty()) {
                throw std::runtime_error("EmbeddingBagForward returned no outputs");
            }
            if (mode_ == "max" && result.size() > 1) {
                max_arg_ = result[1];  // [num_bags, embedding_dim] Int64 global argmax
            }
            return {Variable(result[0], inputs[0].requires_grad())};
        }

        // No GPU kernel registered: refuse to silently demote to CPU. The
        // user asked for an EmbeddingBag forward on a specific device; if
        // that device has no kernel, the right behaviour is a clear error,
        // not a hidden D2H + Float32 host loop + H2D round-trip.
        if (original_device != Device::cpu()) {
            throw std::runtime_error(
                "EmbeddingBag forward: no EmbeddingBagForward kernel registered for device " +
                original_device.to_string() +
                ". Either register a backend kernel for this device or call "
                ".cpu() on the embedding tensor explicitly.");
        }

        // CPU compute path (input was already on CPU).
        Tensor emb_cpu = (original_device == Device::cpu()) ? emb_tensor : emb_tensor.to(Device::cpu());
        if (emb_cpu.dtype() != DType::Float32) {
            emb_cpu = emb_cpu.to(DType::Float32);
        }
        auto emb_ptr = emb_cpu.data<float>();

        auto output = zeros({num_bags_, embedding_dim_}, DType::Float32, Device::cpu());
        auto output_ptr = output.data<float>();

        for (int64_t bag = 0; bag < num_bags_; ++bag) {
            int64_t start_idx = bag_starts_[bag];
            int64_t end_idx = bag_ends_[bag];
            int64_t bag_size = end_idx - start_idx;

            if (bag_size <= 0) continue;

            if (mode_ == "sum" || mode_ == "mean") {
                std::vector<float> compensation(embedding_dim_, 0.0f);
                for (int64_t i = start_idx; i < end_idx; ++i) {
                    for (int64_t j = 0; j < embedding_dim_; ++j) {
                        float y = emb_ptr[i * embedding_dim_ + j] - compensation[j];
                        float t = output_ptr[bag * embedding_dim_ + j] + y;
                        compensation[j] = (t - output_ptr[bag * embedding_dim_ + j]) - y;
                        output_ptr[bag * embedding_dim_ + j] = t;
                    }
                }
                if (mode_ == "mean") {
                    for (int64_t j = 0; j < embedding_dim_; ++j) {
                        output_ptr[bag * embedding_dim_ + j] /= bag_size;
                    }
                }
            } else if (mode_ == "max") {
                for (int64_t j = 0; j < embedding_dim_; ++j) {
                    output_ptr[bag * embedding_dim_ + j] = emb_ptr[start_idx * embedding_dim_ + j];
                }
                for (int64_t i = start_idx + 1; i < end_idx; ++i) {
                    for (int64_t j = 0; j < embedding_dim_; ++j) {
                        float val = emb_ptr[i * embedding_dim_ + j];
                        if (val > output_ptr[bag * embedding_dim_ + j]) {
                            output_ptr[bag * embedding_dim_ + j] = val;
                        }
                    }
                }
            }
        }

        // Convert back to original dtype and device
        if (original_dtype != DType::Float32) {
            output = output.to(original_dtype);
        }
        if (original_device != Device::cpu()) {
            output = output.to(original_device);
        }

        return {Variable(output, inputs[0].requires_grad())};
    }

    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override {
        // grad_output: [num_bags, embedding_dim]
        // grad_emb:    [total_elements, embedding_dim]  (grad w.r.t. gathered rows)
        //
        // Computed entirely with device ops so the gradient stays on the
        // embedding's original device (no CPU host loop / round-trip):
        //   sum:  grad_emb[i]   = grad_output[bag(i)]
        //   mean: grad_emb[i]   = grad_output[bag(i)] / bag_size(bag(i))
        //   max:  grad_emb[i,j] = grad_output[bag,j] iff element i is the argmax
        //         for (bag, j); routed via the forward's saved max_indices.
        if (grad_outputs.empty()) {
            throw std::runtime_error("EmbeddingBagBackward: No gradient outputs");
        }

        Tensor go = grad_outputs[0];
        if (go.device() != original_device_) {
            go = go.to(original_device_);
        }
        const DType compute_dtype = go.dtype();

        Tensor grad_emb;

        if (mode_ == "max") {
            if (!max_arg_.is_valid() || max_arg_.numel() == 0) {
                throw std::runtime_error(
                    "EmbeddingBag(mode=\"max\") backward: forward did not provide "
                    "argmax indices.");
            }
            Tensor argmax = max_arg_;
            if (argmax.device() != original_device_) {
                argmax = argmax.to(original_device_);
            }
            // Empty bags carry argmax = -1. Route those (and only those) to a
            // throwaway row at index total_elements_, then slice it off, so a
            // possibly-nonzero empty-bag grad never corrupts a real element.
            Tensor neg_mask = lt(argmax, full_like(argmax, 0.0));        // bool
            Tensor dummy = full_like(argmax, static_cast<double>(total_elements_));
            Tensor safe_idx = where(neg_mask, dummy, argmax);            // Int64
            Tensor base = zeros({total_elements_ + 1, embedding_dim_},
                                compute_dtype, original_device_);
            Tensor scattered = scatter(base, /*dim=*/0, safe_idx, go);
            grad_emb = narrow(scattered, /*dim=*/0, /*start=*/0, /*length=*/total_elements_);
        } else {
            // Build, on host, the per-element bag id (uncovered elements map to a
            // dummy bag whose grad row is zero). This is O(total_elements) index
            // metadata, NOT gradient compute — the math below runs on-device.
            std::vector<int64_t> bag_id(static_cast<size_t>(total_elements_),
                                        num_bags_);  // default: dummy bag
            for (int64_t bag = 0; bag < num_bags_; ++bag) {
                for (int64_t i = bag_starts_[bag]; i < bag_ends_[bag]; ++i) {
                    if (i >= 0 && i < total_elements_) bag_id[static_cast<size_t>(i)] = bag;
                }
            }
            Tensor bag_ids_cpu({total_elements_}, DType::Int64, Device::cpu());
            std::memcpy(bag_ids_cpu.data<int64_t>(), bag_id.data(),
                        static_cast<size_t>(total_elements_) * sizeof(int64_t));
            Tensor bag_ids = (original_device_ == Device::cpu())
                ? bag_ids_cpu : bag_ids_cpu.to(original_device_);

            Tensor go_rows = go;
            if (mode_ == "mean") {
                // Per-bag scale 1/bag_size (0 for empty bags), broadcast over dim.
                Tensor inv_cpu({num_bags_, 1}, DType::Float32, Device::cpu());
                float* inv_ptr = inv_cpu.data<float>();
                for (int64_t bag = 0; bag < num_bags_; ++bag) {
                    int64_t sz = bag_ends_[bag] - bag_starts_[bag];
                    inv_ptr[bag] = (sz > 0) ? 1.0f / static_cast<float>(sz) : 0.0f;
                }
                Tensor inv_col = inv_cpu;
                if (compute_dtype != DType::Float32) inv_col = inv_col.to(compute_dtype);
                if (original_device_ != Device::cpu()) inv_col = inv_col.to(original_device_);
                go_rows = mul(go, inv_col);
            }
            // Append a trailing zero row so the dummy bag id yields a zero grad.
            Tensor zero_row = zeros({1, embedding_dim_}, compute_dtype, original_device_);
            Tensor go_ext = cat({go_rows, zero_row}, /*dim=*/0);  // [num_bags+1, dim]
            grad_emb = index_select(go_ext, /*dim=*/0, bag_ids);
        }

        if (grad_emb.dtype() != original_dtype_) {
            grad_emb = grad_emb.to(original_dtype_);
        }
        return {grad_emb};
    }

    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override {
        std::vector<Tensor> tensor_grads;
        for (auto& v : grad_outputs) tensor_grads.push_back(v.tensor());
        auto results = backward(std::move(tensor_grads));
        std::vector<Variable> var_results;
        for (auto& t : results) var_results.emplace_back(t, false);
        return var_results;
    }

    // P4.2d: EmbeddingBag is scatter+reduce on non-differentiable
    // indices — second derivative is zero.
    auto supports_higher_order() const -> bool override { return true; }
    auto is_higher_order_stub() const -> bool override { return true; }

private:
    Tensor bag_indices_;   ///< Original vocabulary indices (Int64) — saved so
                           ///< that the OpId::EmbeddingBagBackward dispatch
                           ///< path can recover them (the kernel signature
                           ///< now takes indices, not the looked-up rows).
    Tensor offsets_;
    bool has_offsets_;
    std::string mode_;
    int64_t total_elements_;
    int64_t embedding_dim_;
    bool include_last_offset_;

    // Computed during forward, used during backward
    std::vector<int64_t> bag_starts_;
    std::vector<int64_t> bag_ends_;
    int64_t num_bags_ = 0;
    Device original_device_{Device::cpu()};
    DType original_dtype_ = DType::Float32;
    Tensor max_arg_;  ///< [num_bags, embedding_dim] Int64 per-feature global argmax
                      ///< element index from forward; drives exact on-device
                      ///< max-mode backward (empty unless mode_ == "max").
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
    // scale_grad_by_freq and sparse are handled in EmbeddingBackward::backward()

    // Initialize embedding weight matrix
    initialize_weights();

    // Register as parameter
    register_parameter("weight", weight_);
}

auto Embedding::initialize_weights() -> void {
    // Initialize with Normal(0, 1)
    weight_ = Variable(randn({num_embeddings_, embedding_dim_}), true);

    // Padding embeddings are always zeroed: both at init and after each backward pass.
    // The gradient for padding_idx is explicitly zeroed in EmbeddingBackward.
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

        // Out-of-bounds indices are validated device-side inside the embedding
        // kernel (it flags OOB and the host throws std::out_of_range after a
        // single 1-int readback). The previous host-side validation copied ALL
        // indices to the CPU and looped over them before dispatch — a full D2H +
        // serial scan that serialized the GPU path on every forward.

        // Apply max_norm if specified. renorm_embeddings now runs entirely
        // through device-aware op composition (index_select + per-row norm
        // + minimum + index_copy), so we re-fetch the weight after it
        // mutates the parameter.
        if (max_norm_ > 0.0) {
            renorm_embeddings(input_tensor);
            Tensor renormed = parameters_["weight"]->tensor();
            weight_dev = (renormed.device() == target_device)
                       ? renormed : renormed.to(target_device);
        }

        std::vector<Tensor> inputs_vec = {weight_dev, indices_dev};
        auto results = dispatch<OpId::Embedding>(inputs_vec, {});
        Tensor output = results[0];

        if (!parameters_["weight"]->requires_grad() || !is_grad_enabled()) {
            return Variable(output, false);
        }

        // Set up autograd for GPU path
        Tensor indices_for_grad = input_tensor;
        auto grad_fn = std::make_shared<EmbeddingBackward>(indices_for_grad, num_embeddings_, embedding_dim_, padding_idx_, scale_grad_by_freq_, sparse_);

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
        if (weight_dtype != DType::Float32 && weight_dtype != DType::Float64 &&
            weight_dtype != DType::Float16 && weight_dtype != DType::BFloat16) {
            throw std::runtime_error("Embedding: Unsupported weight dtype: " +
                                     std::to_string(static_cast<int>(weight_dtype)));
        }

        // A gather copies whole embedding rows verbatim, so it is dtype-agnostic:
        // memcpy each contiguous row by its byte span. This replaces the previous
        // single-threaded, element-by-element scalar copy (and the pointless
        // Float16/BFloat16 widen→narrow round-trip, which never changed values).
        // The lookup is memory-bound, so parallelize across the gathered rows
        // using the full logical-core pool — matching PyTorch, which threads this
        // gather. The old serial loop was ~16x slower than PyTorch.
        Tensor weight_contig = weight_cpu.is_contiguous() ? weight_cpu : weight_cpu.contiguous();
        // Every byte of `output` is overwritten by the gather below, so skip the
        // zero-initialization a plain Tensor ctor would do (a wasted ~output-sized
        // memset on the memory-bound critical path).
        Tensor output = Tensor::empty_uninitialized(output_shape, weight_dtype,
                                                    weight_contig.device());

        const int64_t row_bytes =
            embedding_dim_ * static_cast<int64_t>(weight_contig.dtype_size());
        const char* weight_bytes = static_cast<const char*>(weight_contig.data_ptr());
        char* output_bytes = static_cast<char*>(output.data_ptr());

        static const int kLogicalCores =
            std::max(1u, std::thread::hardware_concurrency());
        const bool parallel =
            num_indices > 1 && num_indices * embedding_dim_ > 65536;
        const int nthreads = parallel
            ? std::max(1, std::min(kLogicalCores, static_cast<int>(num_indices)))
            : 1;
        #pragma omp parallel for if(parallel) num_threads(nthreads) schedule(static)
        for (int64_t i = 0; i < num_indices; ++i) {
            const int64_t idx = input_ptr[i];
            std::memcpy(output_bytes + i * row_bytes,
                        weight_bytes + idx * row_bytes,
                        static_cast<size_t>(row_bytes));
        }

        jit_record_embedding(weight_cpu, input_tensor, output);
        return Variable(output, false);
    }

    // Use EmbeddingBackward function to preserve gradient graph
    auto grad_fn = std::make_shared<EmbeddingBackward>(input_cpu, num_embeddings_, embedding_dim_, padding_idx_, scale_grad_by_freq_, sparse_);

    // Perform forward pass (CPU path)
    auto outputs = grad_fn->forward({*parameters_["weight"]});

    if (outputs.empty()) {
        throw std::runtime_error("EmbeddingBackward returned no outputs");
    }

    auto& result = outputs[0];

    jit_record_embedding(parameters_["weight"]->tensor(), input_tensor,
                         result.tensor());

    // Set up backward graph
    std::vector<std::shared_ptr<Function>> next_funcs;
    next_funcs.push_back(parameters_["weight"]->grad_fn());

    grad_fn->set_next_functions(next_funcs);
    grad_fn->set_input_variables({*parameters_["weight"]});

    result.set_grad_fn(grad_fn);

    return result;
}

auto Embedding::renorm_embeddings(const Tensor& indices) -> void {
    // Renormalise rows of the weight matrix that are referenced by `indices`
    // and exceed max_norm under the L_p norm. Implementation is pure op
    // composition over registered per-backend primitives — no host pointer
    // loops, no D2H/H2D round-trip, no device branch:
    //
    //   rows  = index_select(weight, 0, idx)
    //   norm  = (sum(|rows|^p, dim=-1, keepdim))^(1/p)
    //   scale = min(1, max_norm / (norm + eps))    (per-row, broadcast)
    //   scale = where(idx == padding_idx, 1, scale)
    //   weight = index_copy(weight, 0, idx, rows * scale)
    //
    // For p = 2 we use sqrt(sum x²) instead of pow(2)/pow(0.5) so backends
    // without a Float16/BFloat16 pow path (per project memory) still hit a
    // mature kernel.
    Tensor weight = parameters_["weight"]->tensor();
    const Device dev = weight.device();
    const DType wt_dtype = weight.dtype();

    if (wt_dtype != DType::Float32 && wt_dtype != DType::Float64) {
        throw std::runtime_error(
            "Embedding::renorm_embeddings: unsupported weight dtype " +
            std::string(tenzor::dtype_name(wt_dtype)) +
            " (only Float32/Float64 supported)");
    }

    // Flatten indices and ensure they're Int64 on the weight's device.
    Tensor flat_idx = tenzor::reshape(indices, {indices.numel()}).contiguous();
    if (flat_idx.dtype() != DType::Int64) flat_idx = flat_idx.to(DType::Int64);
    if (flat_idx.device().type != dev.type ||
        flat_idx.device().index != dev.index) {
        flat_idx = flat_idx.to(dev);
    }
    const int64_t num_idx = flat_idx.numel();
    if (num_idx == 0) return;

    // Gather referenced rows: (num_idx, embedding_dim).
    Tensor rows = tenzor::index_select(weight, /*dim=*/0, flat_idx);

    // Per-row L_p norm with keepdim → (num_idx, 1).
    Tensor norm;
    if (norm_type_ == 2.0) {
        Tensor sq    = tenzor::mul(rows, rows);
        Tensor sumsq = tenzor::sum(sq, /*dim=*/-1, /*keepdim=*/true);
        norm         = tenzor::sqrt(sumsq);
    } else {
        Tensor abs_rows = tenzor::abs(rows);
        Tensor pow_rows = tenzor::pow(abs_rows, static_cast<float>(norm_type_));
        Tensor sumpow   = tenzor::sum(pow_rows, /*dim=*/-1, /*keepdim=*/true);
        norm            = tenzor::pow(sumpow, static_cast<float>(1.0 / norm_type_));
    }

    // scale = min(1, max_norm / (norm + eps)).
    //
    // HH.12: 1e-8 underflows to 0 in Float16/BFloat16, so the divide-by-zero
    // branch produces inf/nan whenever a row has zero norm. Build the eps /
    // ratio in Float32 when wt_dtype is half-precision and cast back to
    // wt_dtype just before the in-place scale, so the actual weight tensor
    // remains in its declared dtype.
    const bool eps_needs_upcast = (wt_dtype == DType::Float16 ||
                                   wt_dtype == DType::BFloat16);
    const DType ratio_dtype = eps_needs_upcast ? DType::Float32 : wt_dtype;
    Tensor norm_for_ratio = eps_needs_upcast ? norm.to(DType::Float32) : norm;
    Tensor eps   = tenzor::full({}, 1e-8, ratio_dtype, dev);
    Tensor mn    = tenzor::full({}, max_norm_, ratio_dtype, dev);
    Tensor one   = tenzor::full({}, 1.0, ratio_dtype, dev);
    Tensor norms_eps = tenzor::add(norm_for_ratio, eps);
    Tensor ratio = tenzor::div(mn, norms_eps);
    Tensor scale = tenzor::minimum(ratio, one);              // (num_idx, 1)
    if (eps_needs_upcast) {
        scale = scale.to(wt_dtype);
        one   = one.to(wt_dtype);
    }

    // Override scale = 1 for any entry that points at the padding row.
    if (padding_idx_ >= 0) {
        Tensor pad_t = tenzor::full({}, static_cast<double>(padding_idx_),
                                    DType::Int64, dev);
        Tensor pad_mask    = tenzor::eq(flat_idx, pad_t);                  // (num_idx,)
        Tensor pad_mask_2d = tenzor::reshape(pad_mask, {num_idx, 1});      // (num_idx, 1)
        scale = tenzor::where(pad_mask_2d, one, scale);
    }

    // Apply per-row scale and scatter rows back into the weight matrix.
    // index_copy preserves rows that aren't referenced by flat_idx; for
    // duplicate indices, the last write wins — but every duplicate writes
    // the *same* rescaled row (scale was computed from the unmodified row),
    // so the result is correct.
    Tensor new_rows  = tenzor::mul(rows, scale);
    Tensor new_weight = tenzor::index_copy(weight, /*dim=*/0, flat_idx, new_rows);

    auto new_var = std::make_shared<Variable>(
        new_weight, parameters_["weight"]->requires_grad());
    parameters_["weight"] = new_var;
    weight_ = *new_var;
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

    // If no offsets provided, treat entire input as single bag.
    // Always pass the original `input` (the integer vocabulary indices)
    // through to aggregate_embeddings so EmbeddingBagBackward can stash it
    // for the OpId::EmbeddingBagBackward kernel path.
    if (offsets_empty) {
        return aggregate_embeddings(embeddings, input, Variable{});
    }

    // Otherwise, aggregate based on offsets
    return aggregate_embeddings(embeddings, input, offsets);
}

auto EmbeddingBag::forward_impl(const Variable& input) -> Variable {
    // Default forward: treat entire input as single bag
    return forward(input, Variable{});
}

auto EmbeddingBag::aggregate_embeddings(const Variable& embeddings,
                                        const Variable& indices,
                                        const Variable& offsets) -> Variable {
    const auto& emb_tensor = embeddings.tensor();
    auto emb_shape = emb_tensor.shape();

    int64_t total_elements = emb_shape[0];
    int64_t embedding_dim = emb_shape[1];

    // Determine the offsets tensor (empty Tensor if no offsets provided)
    bool has_offsets = offsets.is_initialized() && offsets.tensor().numel() > 0;
    Tensor offsets_tensor = has_offsets ? offsets.tensor() : Tensor();

    // Pull the raw indices tensor (Int64) if supplied; the grad_fn saves it
    // for use by the OpId::EmbeddingBagBackward dispatch path (kernel now
    // scatters into rows selected by indices, not by flat position).
    Tensor indices_tensor;
    if (indices.is_initialized()) {
        try {
            indices_tensor = indices.tensor();
        } catch (...) {
            // Variable lazy/uninitialized — leave empty.
        }
    }

    // If gradient tracking is needed, use EmbeddingBagBackward to preserve the graph
    if (embeddings.requires_grad() && is_grad_enabled()) {
        auto grad_fn = std::make_shared<EmbeddingBagBackward>(
            indices_tensor, offsets_tensor, has_offsets, mode_,
            total_elements, embedding_dim, include_last_offset_);

        // Perform forward pass through the grad_fn
        auto outputs = grad_fn->forward({embeddings});
        if (outputs.empty()) {
            throw std::runtime_error("EmbeddingBagBackward returned no outputs");
        }

        auto& result = outputs[0];

        // Wire up the backward graph: chain to embeddings' grad_fn
        std::vector<std::shared_ptr<Function>> next_funcs;
        next_funcs.push_back(embeddings.grad_fn());
        grad_fn->set_next_functions(next_funcs);
        grad_fn->set_input_variables({embeddings});

        result.set_grad_fn(grad_fn);
        return result;
    }

    // No gradient needed — compute aggregation directly (no autograd overhead)
    Device original_device = emb_tensor.device();
    DType original_dtype = emb_tensor.dtype();

    // GPU path: dispatch to on-device EmbeddingBag kernel (avoids D2H/H2D transfers)
    if (original_device != Device::cpu()) {
        OpAttributes bag_attrs;
        bag_attrs.set(AttrKey::Mode, mode_);
        bag_attrs.set(AttrKey::EmbeddingDim, embedding_dim);
        bag_attrs.set(AttrKey::IncludeLastOffset, include_last_offset_);

        Tensor offsets_gpu;
        if (has_offsets) {
            offsets_gpu = offsets_tensor;
        } else {
            // No offsets: single bag starting at index 0
            offsets_gpu = zeros({1}, DType::Int64, original_device);
        }

        std::array<Tensor, 2> bag_inputs = {emb_tensor, offsets_gpu};
        auto result = dispatch_single<OpId::EmbeddingBagForward>(
            bag_inputs, bag_attrs);
        return Variable(result, false);
    }

    // CPU path: compute aggregation directly. The branch above either
    // returned or threw via dispatch_single, so we know we're on CPU here —
    // no silent .to(cpu) coercion.
    Tensor emb_cpu = emb_tensor;
    // Upcast to Float32 for aggregation if needed (avoids precision loss with Float16/BFloat16)
    if (emb_cpu.dtype() != DType::Float32) {
        emb_cpu = emb_cpu.to(DType::Float32);
    }
    auto emb_ptr = emb_cpu.data<float>();

    // If no offsets, aggregate all embeddings into single vector
    if (!has_offsets) {
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

        return Variable(output, false);
    }

    // With offsets: aggregate each bag separately. (Already on CPU here — see
    // comment above; the GPU path returned earlier.)
    Tensor offsets_cpu = offsets_tensor;
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

    return Variable(output, false);
}

auto EmbeddingBag::weight() -> Variable& {
    return embedding_->weight();
}

auto EmbeddingBag::weight() const -> const Variable& {
    return embedding_->weight();
}

// Factory exposed via embedding.hpp for use by F::embedding in functional.cpp.
namespace internal {
auto make_embedding_backward(::tenzor::Tensor indices, int64_t num_embeddings,
                             int64_t embedding_dim, int64_t padding_idx,
                             bool scale_grad_by_freq, bool sparse)
    -> std::shared_ptr<::tenzor::Function> {
    return std::make_shared<EmbeddingBackward>(std::move(indices),
                                                num_embeddings, embedding_dim,
                                                padding_idx, scale_grad_by_freq,
                                                sparse);
}
}  // namespace internal

} // namespace nn
} // namespace tenzor
