/**
 * @file tensor_parallel.cpp
 * @brief Implementation of tensor parallelism layers
 *
 * Implements Megatron-LM style tensor parallelism for linear layers and
 * multi-head attention. Column-parallel splits output dimension, row-parallel
 * splits input dimension, and parallel attention distributes heads.
 */

#include "tenzor/distributed/tensor_parallel.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/transform.hpp"
#include "tenzor/autograd/ops.hpp"
#include "tenzor/core/dtype.hpp"
#include <cmath>
#include <stdexcept>
#include <cstring>

namespace tenzor::distributed {

// Namespace alias for autograd operations (matches linear.cpp pattern)
namespace autograd = tenzor;

// ============================================================================
// ColumnParallelLinear Implementation
// ============================================================================

ColumnParallelLinear::ColumnParallelLinear(
    int64_t in_features,
    int64_t out_features,
    ProcessGroup& pg,
    bool bias,
    bool gather_output
) : in_features_(in_features),
    out_features_(out_features),
    pg_(pg),
    has_bias_(bias),
    gather_output_(gather_output) {

    int ws = pg_.world_size();

    if (out_features % ws != 0) {
        throw std::invalid_argument(
            "ColumnParallelLinear: out_features (" + std::to_string(out_features) +
            ") must be divisible by world_size (" + std::to_string(ws) + ")"
        );
    }

    local_out_features_ = out_features / ws;

    // Initialize weight: [local_out_features, in_features]
    // Kaiming uniform initialization
    float bound = std::sqrt(1.0f / static_cast<float>(in_features));
    Variable weight(rand({local_out_features_, in_features}) * (2.0f * bound) - bound, true);
    register_parameter("weight", std::move(weight));

    if (bias) {
        Variable bias_var(rand({local_out_features_}) * (2.0f * bound) - bound, true);
        register_parameter("bias", std::move(bias_var));
    }
}

auto ColumnParallelLinear::forward_impl(const Variable& input) -> Variable {
    // Local matmul: input @ weight.T
    // input: (*, in_features), weight: (local_out_features, in_features)
    auto weight = get_parameter("weight");
    auto weight_t = autograd::permute(*weight, {1, 0});
    auto output = autograd::matmul(input, weight_t);

    // Add local bias
    if (has_bias_) {
        auto bias = get_parameter("bias");
        output = output + *bias;
    }

    if (gather_output_) {
        // All-gather the partial outputs from all ranks
        // Each rank has (*, local_out_features), result is (*, out_features)
        int ws = pg_.world_size();
        Tensor output_tensor = output.tensor();

        std::vector<Tensor> gathered(ws);
        for (int i = 0; i < ws; ++i) {
            gathered[i] = tenzor::zeros(std::vector<int64_t>(output_tensor.shape().begin(), output_tensor.shape().end()),
                                        output_tensor.dtype(), output_tensor.device());
        }

        pg_.all_gather(output_tensor, gathered);

        // Concatenate along the last dimension
        std::vector<Variable> gathered_vars;
        gathered_vars.reserve(ws);
        for (int i = 0; i < ws; ++i) {
            gathered_vars.emplace_back(gathered[i], false);
        }

        output = autograd::cat(gathered_vars, -1);
    }

    return output;
}

auto ColumnParallelLinear::extra_repr() const -> std::string {
    return "in_features=" + std::to_string(in_features_) +
           ", out_features=" + std::to_string(out_features_) +
           ", local_out_features=" + std::to_string(local_out_features_) +
           ", bias=" + (has_bias_ ? "true" : "false") +
           ", gather_output=" + (gather_output_ ? "true" : "false");
}

// ============================================================================
// RowParallelLinear Implementation
// ============================================================================

RowParallelLinear::RowParallelLinear(
    int64_t in_features,
    int64_t out_features,
    ProcessGroup& pg,
    bool bias,
    bool input_is_parallel
) : in_features_(in_features),
    out_features_(out_features),
    pg_(pg),
    has_bias_(bias),
    input_is_parallel_(input_is_parallel) {

    int ws = pg_.world_size();

    if (in_features % ws != 0) {
        throw std::invalid_argument(
            "RowParallelLinear: in_features (" + std::to_string(in_features) +
            ") must be divisible by world_size (" + std::to_string(ws) + ")"
        );
    }

    local_in_features_ = in_features / ws;

    // Initialize weight: [out_features, local_in_features]
    float bound = std::sqrt(1.0f / static_cast<float>(in_features));
    Variable weight(rand({out_features, local_in_features_}) * (2.0f * bound) - bound, true);
    register_parameter("weight", std::move(weight));

    // Only rank 0 holds the bias to avoid double-counting after all-reduce
    if (bias && pg_.rank() == 0) {
        Variable bias_var(rand({out_features}) * (2.0f * bound) - bound, true);
        register_parameter("bias", std::move(bias_var));
    }
}

auto RowParallelLinear::forward_impl(const Variable& input) -> Variable {
    Variable local_input = input;

    // If input is not already parallel (not split across ranks), split it
    if (!input_is_parallel_) {
        int rank = pg_.rank();
        // Split input along last dimension
        // input: (*, in_features) -> (*, local_in_features)
        int64_t start = rank * local_in_features_;
        // Use narrow on the last dimension
        auto input_shape = input.tensor().shape();
        int64_t last_dim = static_cast<int64_t>(input_shape.size()) - 1;
        local_input = autograd::narrow(input, last_dim, start, local_in_features_);
    }

    // Local matmul: local_input @ weight.T
    // local_input: (*, local_in_features), weight: (out_features, local_in_features)
    auto weight = get_parameter("weight");
    auto weight_t = autograd::permute(*weight, {1, 0});
    auto output = autograd::matmul(local_input, weight_t);

    // All-reduce to sum partial results across ranks
    Tensor output_tensor = output.tensor();
    pg_.all_reduce(output_tensor, ReduceOp::SUM);

    // Wrap back as Variable (gradient tracking through all-reduce is approximate;
    // for full autograd support, a custom Function would be needed)
    output = Variable(output_tensor, output.requires_grad());

    // Add bias (only rank 0 has it, but after all-reduce all ranks need it)
    if (has_bias_ && pg_.rank() == 0) {
        auto bias = get_parameter("bias");
        output = output + *bias;
    }

    // Broadcast the output from rank 0 so all ranks have identical output
    // (rank 0 added bias, others didn't)
    if (has_bias_) {
        Tensor out_t = output.tensor();
        pg_.broadcast(out_t, /*src_rank=*/0);
        output = Variable(out_t, output.requires_grad());
    }

    return output;
}

auto RowParallelLinear::extra_repr() const -> std::string {
    return "in_features=" + std::to_string(in_features_) +
           ", out_features=" + std::to_string(out_features_) +
           ", local_in_features=" + std::to_string(local_in_features_) +
           ", bias=" + (has_bias_ ? "true" : "false") +
           ", input_is_parallel=" + (input_is_parallel_ ? "true" : "false");
}

// ============================================================================
// ParallelAttention Implementation
// ============================================================================

ParallelAttention::ParallelAttention(
    int64_t embed_dim,
    int64_t num_heads,
    ProcessGroup& pg,
    float dropout
) : embed_dim_(embed_dim),
    num_heads_(num_heads),
    dropout_(dropout),
    pg_(pg) {

    int ws = pg_.world_size();

    if (num_heads % ws != 0) {
        throw std::invalid_argument(
            "ParallelAttention: num_heads (" + std::to_string(num_heads) +
            ") must be divisible by world_size (" + std::to_string(ws) + ")"
        );
    }

    if (embed_dim % num_heads != 0) {
        throw std::invalid_argument(
            "ParallelAttention: embed_dim (" + std::to_string(embed_dim) +
            ") must be divisible by num_heads (" + std::to_string(num_heads) + ")"
        );
    }

    local_num_heads_ = num_heads / ws;
    head_dim_ = embed_dim / num_heads;

    // Q/K/V projections: column-parallel (split output = head dim * local_num_heads)
    // gather_output = false because we compute attention locally
    q_proj_ = std::make_shared<ColumnParallelLinear>(
        embed_dim, num_heads * head_dim_, pg, /*bias=*/true, /*gather_output=*/false);
    k_proj_ = std::make_shared<ColumnParallelLinear>(
        embed_dim, num_heads * head_dim_, pg, /*bias=*/true, /*gather_output=*/false);
    v_proj_ = std::make_shared<ColumnParallelLinear>(
        embed_dim, num_heads * head_dim_, pg, /*bias=*/true, /*gather_output=*/false);

    // Output projection: row-parallel (split input = local_num_heads * head_dim)
    out_proj_ = std::make_shared<RowParallelLinear>(
        num_heads * head_dim_, embed_dim, pg, /*bias=*/true, /*input_is_parallel=*/true);

    register_module("q_proj", q_proj_);
    register_module("k_proj", k_proj_);
    register_module("v_proj", v_proj_);
    register_module("out_proj", out_proj_);
}

auto ParallelAttention::forward_impl(const Variable& input) -> Variable {
    // input: (batch, seq_len, embed_dim)
    auto input_shape = input.tensor().shape();
    if (input_shape.size() != 3) {
        throw std::runtime_error(
            "ParallelAttention: expected 3D input (batch, seq_len, embed_dim), got " +
            std::to_string(input_shape.size()) + "D"
        );
    }

    int64_t batch = input_shape[0];
    int64_t seq_len = input_shape[1];

    // Q/K/V projections (column-parallel, no gather)
    // Each produces (batch, seq_len, local_num_heads * head_dim)
    auto q = q_proj_->forward(input);
    auto k = k_proj_->forward(input);
    auto v = v_proj_->forward(input);

    // Reshape to (batch, seq_len, local_num_heads, head_dim)
    q = autograd::reshape(q, {batch, seq_len, local_num_heads_, head_dim_});
    k = autograd::reshape(k, {batch, seq_len, local_num_heads_, head_dim_});
    v = autograd::reshape(v, {batch, seq_len, local_num_heads_, head_dim_});

    // Transpose to (batch, local_num_heads, seq_len, head_dim)
    q = autograd::permute(q, {0, 2, 1, 3});
    k = autograd::permute(k, {0, 2, 1, 3});
    v = autograd::permute(v, {0, 2, 1, 3});

    // Scaled dot-product attention (local computation, no communication)
    // Reshape to 3D for bmm: (batch * heads, seq_len, head_dim)
    int64_t num_heads = local_num_heads_;
    auto q_3d = autograd::reshape(q, {batch * num_heads, seq_len, head_dim_});
    auto k_3d = autograd::reshape(k, {batch * num_heads, seq_len, head_dim_});
    auto v_3d = autograd::reshape(v, {batch * num_heads, seq_len, head_dim_});

    // scores = Q @ K^T / sqrt(head_dim)
    auto k_t = autograd::permute(k_3d, {0, 2, 1});  // (batch*heads, head_dim, seq_len)
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim_));

    auto scores = autograd::matmul(q_3d, k_t);  // (batch*heads, seq_len, seq_len)
    scores = scores * scale;

    // Softmax along last dimension
    auto attn_weights = autograd::softmax(scores, -1);

    // Attention output: weights @ V
    auto attn_output = autograd::matmul(attn_weights, v_3d);  // (batch*heads, seq_len, head_dim)

    // Reshape back to 4D: (batch, heads, seq_len, head_dim)
    attn_output = autograd::reshape(attn_output, {batch, num_heads, seq_len, head_dim_});

    // Transpose back: (batch, seq_len, local_num_heads, head_dim)
    attn_output = autograd::permute(attn_output, {0, 2, 1, 3});

    // Reshape to (batch, seq_len, local_num_heads * head_dim)
    attn_output = autograd::reshape(attn_output, {batch, seq_len, local_num_heads_ * head_dim_});

    // Output projection (row-parallel with all-reduce)
    auto output = out_proj_->forward(attn_output);

    return output;
}

auto ParallelAttention::extra_repr() const -> std::string {
    return "embed_dim=" + std::to_string(embed_dim_) +
           ", num_heads=" + std::to_string(num_heads_) +
           ", local_num_heads=" + std::to_string(local_num_heads_) +
           ", head_dim=" + std::to_string(head_dim_) +
           ", dropout=" + std::to_string(dropout_);
}

} // namespace tenzor::distributed
