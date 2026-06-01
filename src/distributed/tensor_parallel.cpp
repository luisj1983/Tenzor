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
#include "tenzor/autograd/function.hpp"
#include "tenzor/autograd/variable.hpp"
#include "tenzor/core/dtype.hpp"
#include <cmath>
#include <memory>
#include <stdexcept>
#include <cstring>

namespace tenzor::distributed {

// Namespace alias for autograd operations (matches linear.cpp pattern)
namespace autograd = tenzor;

// ============================================================================
// Megatron-LM tensor-parallel collective autograd operators (f / g / gather /
// scatter). These keep the autograd graph intact across collective ops so that
// gradients reach the local sharded weights. Forward/backward semantics follow
// Megatron-LM exactly:
//   - g  (TPReduce):  forward all-reduce(SUM),  backward identity.
//   - f  (TPCopy):    forward identity,         backward all-reduce(SUM).
//   - gather:         forward all-gather + cat, backward narrow to local slice.
//   - scatter:        forward narrow to slice,  backward all-gather + cat.
// g and f are duals: re-reducing the gradient in g's backward (as a plain
// differentiable all-reduce would) overcounts by world_size, so g's backward
// must be identity.
// ============================================================================
namespace {

auto normalize_dim(int64_t dim, int64_t rank) -> int64_t {
    return dim < 0 ? dim + rank : dim;
}

// Wire a single-input/single-output collective Function into the graph,
// mirroring tenzor::distributed_all_reduce's manual graph construction.
auto make_collective_node(const Variable& input, Tensor out,
                          std::shared_ptr<Function> grad_fn) -> Variable {
    Variable result(std::move(out), input.requires_grad());
    if (input.requires_grad() && tenzor::is_grad_enabled()) {
        std::vector<std::shared_ptr<Function>> next_funcs;
        next_funcs.push_back(input.grad_fn());  // nullptr if leaf
        grad_fn->set_next_functions(std::move(next_funcs));
        std::vector<Variable> input_vars;
        input_vars.push_back(input);
        grad_fn->set_input_variables(std::move(input_vars));
        result.set_grad_fn(grad_fn);
    }
    return result;
}

// g operator: backward is identity (gradient passes straight through).
class TPReduceBackward : public Function {
public:
    auto forward(std::vector<Variable>) -> std::vector<Variable> override {
        throw std::runtime_error(
            "TPReduceBackward::forward should not be called directly");
    }
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override {
        return {grad_outputs.at(0)};
    }
    auto name() const -> std::string override { return "TPReduceBackward"; }
};

// f operator: backward all-reduces the incoming gradient across ranks.
class TPCopyBackward : public Function {
public:
    explicit TPCopyBackward(ProcessGroup* pg) : pg_(pg) {}
    auto forward(std::vector<Variable>) -> std::vector<Variable> override {
        throw std::runtime_error(
            "TPCopyBackward::forward should not be called directly");
    }
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override {
        Tensor g = grad_outputs.at(0).clone();
        pg_->all_reduce(g, ReduceOp::SUM);
        return {g};
    }
    auto name() const -> std::string override { return "TPCopyBackward"; }
private:
    ProcessGroup* pg_;
};

// gather: backward narrows the full gradient to this rank's slice.
class TPGatherBackward : public Function {
public:
    TPGatherBackward(int rank, int64_t dim, int64_t local_len)
        : rank_(rank), dim_(dim), local_len_(local_len) {}
    auto forward(std::vector<Variable>) -> std::vector<Variable> override {
        throw std::runtime_error(
            "TPGatherBackward::forward should not be called directly");
    }
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override {
        return {grad_outputs.at(0).narrow(dim_, rank_ * local_len_, local_len_).clone()};
    }
    auto name() const -> std::string override { return "TPGatherBackward"; }
private:
    int rank_;
    int64_t dim_;
    int64_t local_len_;
};

// scatter: backward all-gathers the per-rank gradient slices into the full
// (replicated) gradient.
class TPScatterBackward : public Function {
public:
    TPScatterBackward(ProcessGroup* pg, int64_t dim) : pg_(pg), dim_(dim) {}
    auto forward(std::vector<Variable>) -> std::vector<Variable> override {
        throw std::runtime_error(
            "TPScatterBackward::forward should not be called directly");
    }
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override {
        const Tensor& g_local = grad_outputs.at(0);
        int ws = pg_->world_size();
        std::vector<Tensor> parts(ws);
        std::vector<int64_t> shp(g_local.shape().begin(), g_local.shape().end());
        for (int i = 0; i < ws; ++i) {
            parts[i] = tenzor::zeros(shp, g_local.dtype(), g_local.device());
        }
        pg_->all_gather(g_local, parts);
        return {tenzor::cat(parts, dim_)};
    }
    auto name() const -> std::string override { return "TPScatterBackward"; }
private:
    ProcessGroup* pg_;
    int64_t dim_;
};

// g: all-reduce(SUM) forward, identity backward.
auto tp_reduce(const Variable& input, ProcessGroup& pg) -> Variable {
    Tensor out = input.tensor().clone();
    pg.all_reduce(out, ReduceOp::SUM);
    return make_collective_node(input, std::move(out),
                                std::make_shared<TPReduceBackward>());
}

// f: identity forward, all-reduce(SUM) backward.
auto tp_copy(const Variable& input, ProcessGroup& pg) -> Variable {
    Tensor out = input.tensor().clone();
    return make_collective_node(input, std::move(out),
                                std::make_shared<TPCopyBackward>(&pg));
}

// gather: all-gather + cat forward, narrow backward.
auto tp_gather(const Variable& input, ProcessGroup& pg, int64_t dim) -> Variable {
    int ws = pg.world_size();
    const Tensor& local = input.tensor();
    std::vector<int64_t> shp(local.shape().begin(), local.shape().end());
    int64_t pos_dim = normalize_dim(dim, static_cast<int64_t>(shp.size()));
    int64_t local_len = shp.at(pos_dim);
    std::vector<Tensor> parts(ws);
    for (int i = 0; i < ws; ++i) {
        parts[i] = tenzor::zeros(shp, local.dtype(), local.device());
    }
    pg.all_gather(local, parts);
    Tensor out = tenzor::cat(parts, pos_dim);
    return make_collective_node(
        input, std::move(out),
        std::make_shared<TPGatherBackward>(pg.rank(), pos_dim, local_len));
}

// scatter: narrow-to-local forward, all-gather backward.
auto tp_scatter(const Variable& input, ProcessGroup& pg, int64_t dim,
                int64_t start, int64_t length) -> Variable {
    const Tensor& full = input.tensor();
    int64_t pos_dim = normalize_dim(dim, static_cast<int64_t>(full.shape().size()));
    Tensor out = full.narrow(pos_dim, start, length).clone();
    return make_collective_node(
        input, std::move(out),
        std::make_shared<TPScatterBackward>(&pg, pos_dim));
}

} // namespace

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
    // f operator: identity forward, all-reduce backward. Makes the replicated
    // input's gradient correct (summed across ranks) while staying graph-connected.
    Variable input_parallel = (pg_.world_size() > 1) ? tp_copy(input, pg_) : input;

    // Local matmul: input @ weight.T
    // input: (*, in_features), weight: (local_out_features, in_features)
    auto weight = get_parameter("weight");
    auto weight_t = autograd::permute(*weight, {1, 0});
    auto output = autograd::matmul(input_parallel, weight_t);

    // Add local bias
    if (has_bias_) {
        auto bias = get_parameter("bias");
        output = output + *bias;
    }

    if (gather_output_ && pg_.world_size() > 1) {
        // gather operator: all-gather the partial outputs and concatenate along
        // the last dimension. Backward narrows the gradient to this rank's slice.
        // Each rank has (*, local_out_features), result is (*, out_features).
        output = tp_gather(output, pg_, -1);
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

    // Bias is replicated on every rank and added once AFTER the all-reduce of
    // partial products, so there is no double-counting. Replicating (rather than
    // keeping it only on rank 0 + broadcasting) keeps the add autograd-connected
    // and identical on all ranks, and its gradient is identical on every rank.
    if (bias) {
        Variable bias_var(rand({out_features}) * (2.0f * bound) - bound, true);
        register_parameter("bias", std::move(bias_var));
    }
}

auto RowParallelLinear::forward_impl(const Variable& input) -> Variable {
    Variable local_input = input;

    // If input is not already parallel (replicated full input), scatter it to
    // this rank's column slice. scatter's backward all-gathers the per-rank
    // gradient slices into the full (replicated) input gradient.
    if (!input_is_parallel_ && pg_.world_size() > 1) {
        int rank = pg_.rank();
        int64_t start = rank * local_in_features_;
        int64_t last_dim = static_cast<int64_t>(input.tensor().shape().size()) - 1;
        local_input = tp_scatter(input, pg_, last_dim, start, local_in_features_);
    }

    // Local matmul: local_input @ weight.T
    // local_input: (*, local_in_features), weight: (out_features, local_in_features)
    auto weight = get_parameter("weight");
    auto weight_t = autograd::permute(*weight, {1, 0});
    auto output = autograd::matmul(local_input, weight_t);

    // g operator: all-reduce(SUM) the partial products across ranks. Backward is
    // identity, so the gradient flows straight to the local weight (and input).
    if (pg_.world_size() > 1) {
        output = tp_reduce(output, pg_);
    }

    // Replicated bias added once on every rank after the all-reduce (identical
    // result on all ranks; graph-connected so the bias gradient flows).
    if (has_bias_) {
        auto bias = get_parameter("bias");
        output = output + *bias;
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
