#include "tenzor/nn/layers/sparse_linear.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/transform.hpp"
#include "tenzor/sparse/sparse_ops.hpp"
#include "tenzor/sparse/sparse_tensor.hpp"
#include "tenzor/autograd/ops.hpp"
#include "tenzor/autograd/function.hpp"
#include <cmath>
#include <random>
#include <utility>

namespace tenzor::nn {

// ============================================================================
// SparseLinearBackward — joint backward for (values, input_t) through
// `result_t = S @ input_t` where S is CSR with FIXED pattern and the
// values vector is a trainable Parameter.
//
// Forward (computed outside the Function, by SparseLinear::forward_impl):
//     result_t = sparse::spmm(S_csr, input_t)
//
// Backward:
//     grad_input_t = S^T @ grad_result_t                     (dense, (K, N))
//     grad_values[p] = sum_j grad_result_t[row(p), j] * input_t[col(p), j]
//                    = (grad_result_t @ input_t^T) restricted to S's pattern
//                    = sddmm(S_pattern, grad_result_t, input_t).values()
//
// `input_variables_` order MUST match the gradient slot order:
//   [0] values Variable   (the trainable Parameter)
//   [1] input_t Variable  (intermediate after permute)
// ============================================================================
class SparseLinearBackward : public Function {
public:
    /// Set the sparse pattern (forward S) and its transpose (used in backward).
    void set_sparse(SparseTensor sparse, SparseTensor sparse_t) {
        sparse_.emplace(std::move(sparse));
        sparse_transposed_.emplace(std::move(sparse_t));
    }

    auto forward(std::vector<Variable> /*inputs*/) -> std::vector<Variable> override {
        throw std::runtime_error(
            "SparseLinearBackward::forward should not be called directly");
    }

    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override {
        // grad_outputs[0] = grad w.r.t. result_t, shape (M, N).
        require_saved_tensors(1);
        const auto& input_t = saved_tensors()[0];  // (K, N) dense
        const auto& grad_result_t = grad_outputs[0];

        std::vector<Tensor> result(2);  // [0]=grad_values, [1]=grad_input_t

        // grad_values via SDDMM: only need to fill in the (row, col) entries
        // of S's pattern, exactly matching the values vector layout.
        if (sparse_.has_value()) {
            // SDDMM is CPU-only Float32/Float64 in this codebase.  Widen
            // Float16/BFloat16 inputs and the sparse pattern's values dtype
            // up to Float32 for the compute, then narrow the resulting
            // grad-values vector back to the original dtype so the Parameter
            // (whose Tensor is the original dtype) accumulates correctly.
            const DType orig = grad_result_t.dtype();
            const bool widen =
                (orig == DType::Float16 || orig == DType::BFloat16);

            Tensor gr = widen ? grad_result_t.to(DType::Float32) : grad_result_t;
            Tensor in = widen ? input_t.to(DType::Float32) : input_t;

            SparseTensor pat = sparse_.value();
            if (widen) {
                // Rebuild the CSR with Float32 values just so sddmm's dtype
                // checks pass — we only consume the OUTPUT values vector.
                pat = SparseTensor::sparse_csr(
                    pat.crow_indices(), pat.col_indices(),
                    pat.values().to(DType::Float32), pat.shape());
            }

            SparseTensor grad_values_sparse = sparse::sddmm(pat, gr, in);
            // The SDDMM result is a CSR matrix with S's pattern; its values()
            // is the vector of d L / d values[p] in CSR order, which is the
            // same order in which the SparseLinear stores its values
            // Parameter.
            Tensor grad_values = grad_values_sparse.values();
            if (widen) {
                grad_values = grad_values.to(orig);
            }
            result[0] = std::move(grad_values);
        }

        // grad_input_t = S^T @ grad_result_t (standard sparse-dense matmul).
        if (sparse_transposed_.has_value()) {
            result[1] = sparse::spmm(sparse_transposed_.value(), grad_result_t);
        }
        return result;
    }

    auto backward_with_variables(std::vector<Variable> grad_outputs)
        -> std::vector<Variable> override {
        auto result_tensors = backward({grad_outputs[0].tensor()});
        std::vector<Variable> results;
        results.reserve(result_tensors.size());
        for (auto& t : result_tensors) {
            results.emplace_back(t, grad_outputs[0].requires_grad());
        }
        return results;
    }

    auto supports_higher_order() const -> bool override { return true; }
    auto name() const -> std::string override { return "SparseLinearBackward"; }

private:
    std::optional<SparseTensor> sparse_;             ///< S in CSR (fixed pattern)
    std::optional<SparseTensor> sparse_transposed_;  ///< S^T in CSR, for grad_input_t
};

namespace {

/// Build the autograd-tracked spmm specialisation used by SparseLinear:
/// computes `result_t = S @ input_t` while routing gradients back to BOTH
/// `values_var` (the trainable values Parameter) and `input_t` (the intermediate
/// after permute).  S itself is constructed from the values_var's tensor + the
/// fixed CSR pattern carried by `pattern`.
auto sparse_linear_spmm(const Variable& values_var,
                        const SparseTensor& sparse,
                        const Variable& input_t) -> Variable {
    // Compute forward at the Tensor level (avoids constructing the sparse op
    // dispatch graph — S's pattern is constant so the standard sparse::spmm
    // suffices).
    auto result_tensor = sparse::spmm(sparse, input_t.tensor());

    const bool wants_grad =
        is_grad_enabled() &&
        (values_var.requires_grad() || input_t.requires_grad());

    if (!wants_grad) {
        return Variable(std::move(result_tensor), false);
    }

    auto grad_fn = std::make_shared<SparseLinearBackward>();
    grad_fn->set_sparse(sparse, sparse.transpose());

    // Save input_t for backward; values_var is encoded via the sparse pattern.
    grad_fn->save_for_backward({input_t.tensor()});

    // Order MUST match SparseLinearBackward::backward return order.
    grad_fn->set_next_functions({values_var.grad_fn(), input_t.grad_fn()});
    grad_fn->set_input_variables({values_var, input_t});

    Variable output(std::move(result_tensor), true);
    output.set_grad_fn(grad_fn);
    return output;
}

}  // namespace

SparseLinear::SparseLinear(int64_t in_features, int64_t out_features,
                           double density, bool bias)
    : in_features_(in_features), out_features_(out_features),
      density_(density), has_bias_(bias) {

    if (in_features <= 0 || out_features <= 0) {
        throw std::runtime_error("SparseLinear: features must be positive");
    }
    if (density <= 0.0 || density > 1.0) {
        throw std::runtime_error("SparseLinear: density must be in (0, 1]");
    }

    reset_parameters();
}

SparseLinear::SparseLinear(const SparseTensor& sparse_weight, bool bias)
    : has_bias_(bias), sparse_weight_(sparse_weight) {

    auto shape = sparse_weight.shape();
    if (shape.size() != 2) {
        throw std::runtime_error("SparseLinear: weight must be 2D");
    }
    out_features_ = shape[0];
    in_features_ = shape[1];

    int64_t total = out_features_ * in_features_;
    density_ = (total > 0) ? static_cast<double>(sparse_weight.nnz()) / total : 0.0;

    // Make sure we hold the CSR layout (the values-Parameter scheme assumes CSR).
    if (sparse_weight_->layout() != SparseLayout::CSR) {
        sparse_weight_ = sparse_weight_->coalesce().to_csr();
    }

    // Register the CSR values vector as a trainable Parameter so optimisers
    // (SGD/Adam/Adagrad/...) update it.  The pattern (crow, col) stays fixed.
    Variable values_var(sparse_weight_->values(), /*requires_grad=*/true);
    register_parameter("sparse_weight.values", std::move(values_var));

    if (bias) {
        float bound = 1.0f / std::sqrt(static_cast<float>(in_features_));
        Variable bias_var(rand({out_features_}) * (2.0f * bound) - bound, true);
        register_parameter("bias", std::move(bias_var));
    }
}

auto SparseLinear::sync_sparse_weight_values() -> void {
    // Rebuild sparse_weight_ so its values() reflects the latest tensor stored
    // in the "sparse_weight.values" Parameter.  Optimisers update parameters
    // by REPLACING Variable::tensor() with a freshly-allocated buffer
    // (e.g. SGD does `p = p - lr*grad`), so the SparseTensor's by-value
    // `values_` member becomes stale after each optimiser step.  We rebuild
    // here using the fixed CSR pattern (crow_indices, col_indices, shape)
    // and the current values Tensor.  Cheap — no data is copied; the new
    // SparseTensor's `values_` shares storage with the Parameter's Tensor.
    auto it = parameters_.find("sparse_weight.values");
    if (it == parameters_.end() || !sparse_weight_.has_value()) {
        return;
    }
    const auto& cur_values = it->second->tensor();
    // Fast path: storage already matches AND device already consistent.
    if (cur_values.data_ptr() == sparse_weight_->values().data_ptr() &&
        sparse_weight_->crow_indices().device().type == cur_values.device().type) {
        return;
    }
    // module.to(device) moves the registered "sparse_weight.values" Parameter
    // but NOT the SparseTensor member's CSR index tensors. Rebuilding the CSR
    // with the moved (e.g. OneAPI) values but stale CPU crow/col indices yields
    // a mixed-device SparseTensor → spmm throws "tensors must be on the same
    // device". Move the indices onto the values' device so the pattern follows
    // the parameter across .to() (no-op when already co-located).
    auto crow = sparse_weight_->crow_indices();
    auto col  = sparse_weight_->col_indices();
    if (crow.device().type != cur_values.device().type) crow = crow.to(cur_values.device());
    if (col.device().type  != cur_values.device().type) col  = col.to(cur_values.device());
    sparse_weight_ = SparseTensor::sparse_csr(
        crow, col, cur_values, sparse_weight_->shape());
}

auto SparseLinear::forward_impl(const Variable& input) -> Variable {
    // input: [batch, in_features]
    // sparse_weight_: [out_features, in_features] in CSR
    // output = spmm(sparse_weight, input^T)^T = [batch, out_features]
    //
    // Route through the Variable-aware permute / sparse_linear_spmm so the
    // autograd graph sees (input -> permute -> spmm -> permute) AND so the
    // gradient flows back to both `input` and the values-Parameter.

    // (1) Make sure the SparseTensor's values() reflects the latest Parameter
    // tensor.  Optimisers replace Variable::tensor() each step, so a no-op
    // refresh is required before reading values out.
    sync_sparse_weight_values();

    // (2) Resolve the values Parameter as a Variable for the autograd hookup.
    auto values_it = parameters_.find("sparse_weight.values");
    if (values_it == parameters_.end()) {
        throw std::runtime_error(
            "SparseLinear: 'sparse_weight.values' Parameter not registered");
    }
    const Variable& values_var = *values_it->second;

    auto input_t = tenzor::permute(input, {1, 0});
    auto result_t = sparse_linear_spmm(values_var, sparse_weight_.value(), input_t);
    auto output = tenzor::permute(result_t, {1, 0});

    if (has_bias_) {
        auto bias_it = parameters_.find("bias");
        if (bias_it != parameters_.end()) {
            output = output + *bias_it->second;
        }
    }

    return output;
}

auto SparseLinear::reset_parameters() -> void {
    // Create sparse weight with Kaiming initialization
    float bound = std::sqrt(1.0f / static_cast<float>(in_features_));
    int64_t total = out_features_ * in_features_;
    int64_t nnz = static_cast<int64_t>(density_ * total);
    if (nnz < 1) nnz = 1;

    // Generate random non-zero positions
    std::mt19937 gen(42);
    std::uniform_int_distribution<int64_t> row_dist(0, out_features_ - 1);
    std::uniform_int_distribution<int64_t> col_dist(0, in_features_ - 1);
    std::uniform_real_distribution<float> val_dist(-bound, bound);

    std::vector<int64_t> rows(nnz), cols(nnz);
    std::vector<float> vals(nnz);

    for (int64_t i = 0; i < nnz; ++i) {
        rows[i] = row_dist(gen);
        cols[i] = col_dist(gen);
        vals[i] = val_dist(gen);
    }

    // Build COO indices tensor: shape [2, nnz]
    auto indices_data = zeros({2, nnz}, DType::Int64, Device::cpu());
    auto* idx_ptr = indices_data.data<int64_t>();
    for (int64_t i = 0; i < nnz; ++i) {
        idx_ptr[i] = rows[i];
        idx_ptr[nnz + i] = cols[i];
    }

    auto values_data = zeros({nnz}, DType::Float32, Device::cpu());
    auto* val_ptr = values_data.data<float>();
    for (int64_t i = 0; i < nnz; ++i) {
        val_ptr[i] = vals[i];
    }

    sparse_weight_ = SparseTensor::sparse_coo(
        indices_data, values_data, {out_features_, in_features_});
    sparse_weight_ = sparse_weight_->coalesce().to_csr();

    // After coalesce + to_csr the values may have been reordered/deduplicated;
    // register the *final* CSR values vector as the trainable Parameter so its
    // ordering matches the CSR pattern used by sparse::spmm / sparse::sddmm.
    Variable values_var(sparse_weight_->values(), /*requires_grad=*/true);
    register_parameter("sparse_weight.values", std::move(values_var));

    if (has_bias_) {
        float bias_bound = 1.0f / std::sqrt(static_cast<float>(in_features_));
        Variable bias_var(rand({out_features_}) * (2.0f * bias_bound) - bias_bound, true);
        register_parameter("bias", std::move(bias_var));
    }
}

} // namespace tenzor::nn
