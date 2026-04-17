#include "tenzor/nn/layers/sparse_embedding.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/op_id.hpp"
#include "tenzor/backend/fast_dispatch.hpp"
#include "tenzor/backend/op_attributes.hpp"
#include "tenzor/autograd/function.hpp"
#include <array>
#include <cstring>
#include <vector>

namespace tenzor::nn {

// Backward function that produces sparse gradients (COO format)
class SparseEmbeddingBackward : public Function {
public:
    SparseEmbeddingBackward(int64_t num_embeddings, int64_t embedding_dim,
                            Tensor indices, int64_t padding_idx)
        : num_embeddings_(num_embeddings), embedding_dim_(embedding_dim),
          indices_(std::move(indices)), padding_idx_(padding_idx) {}

    auto forward([[maybe_unused]] std::vector<Variable> inputs) -> std::vector<Variable> override {
        throw std::runtime_error("SparseEmbeddingBackward::forward should not be called");
    }

    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override {
        const auto& grad = grad_outputs[0];  // [num_indices, embedding_dim]

        // Use existing EmbeddingBackward kernel which only accumulates into accessed rows
        OpAttributes attrs;
        attrs.set(AttrKey::NumEmbeddings, num_embeddings_);
        std::array<Tensor, 2> inputs = {grad, indices_};
        auto result = dispatch_single<OpId::EmbeddingBackward>(inputs, attrs);

        // Zero out padding_idx gradient if applicable
        if (padding_idx_ >= 0) {
            auto result_data = result.data_ptr();
            auto elem_size = result.element_size();
            std::memset(
                static_cast<char*>(result_data) + padding_idx_ * embedding_dim_ * elem_size,
                0, embedding_dim_ * elem_size);
        }

        return {result};
    }

    // Embedding is an index-scatter — linear in the gradient — so the
    // second derivative is structurally zero.
    TENZOR_HIGHER_ORDER_STRUCTURAL_ZERO_STUB()

private:
    int64_t num_embeddings_;
    int64_t embedding_dim_;
    Tensor indices_;
    int64_t padding_idx_;
};

SparseEmbedding::SparseEmbedding(int64_t num_embeddings, int64_t embedding_dim,
                                   int64_t padding_idx)
    : num_embeddings_(num_embeddings), embedding_dim_(embedding_dim),
      padding_idx_(padding_idx) {

    if (num_embeddings <= 0 || embedding_dim <= 0) {
        throw std::runtime_error("SparseEmbedding: dimensions must be positive");
    }

    // Initialize with normal distribution (matches PyTorch default)
    Variable weight_var(randn({num_embeddings, embedding_dim}), true);

    // Zero out padding index
    if (padding_idx >= 0 && padding_idx < num_embeddings) {
        auto w = weight_var.tensor();
        auto* data = w.data<float>();
        std::memset(data + padding_idx * embedding_dim, 0,
                    embedding_dim * sizeof(float));
    }

    register_parameter("weight", std::move(weight_var));
}

auto SparseEmbedding::forward_impl(const Variable& input) -> Variable {
    // Forward is identical to regular Embedding — lookup rows from weight table
    auto& weight_var = *parameters_.at("weight");
    const auto& indices = input.tensor();

    // Use standard Embedding dispatch
    std::array<Tensor, 2> inputs = {weight_var.tensor(), indices};
    auto result = dispatch_single<OpId::Embedding>(inputs);

    Variable output(result, input.requires_grad() || weight_var.requires_grad());

    // Wire up sparse backward
    if (output.requires_grad() && is_grad_enabled()) {
        auto grad_fn = std::make_shared<SparseEmbeddingBackward>(
            num_embeddings_, embedding_dim_, indices, padding_idx_);

        std::vector<std::shared_ptr<Function>> next_funcs;
        if (weight_var.grad_fn()) {
            next_funcs.push_back(weight_var.grad_fn());
        }
        grad_fn->set_next_functions(next_funcs);
        grad_fn->set_input_variables({weight_var});
        output.set_grad_fn(grad_fn);
    }

    return output;
}

} // namespace tenzor::nn
