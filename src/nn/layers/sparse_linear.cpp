#include "tenzor/nn/layers/sparse_linear.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/sparse/sparse_ops.hpp"
#include "tenzor/autograd/ops.hpp"
#include <cmath>
#include <random>

namespace tenzor::nn {

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

    if (bias) {
        float bound = 1.0f / std::sqrt(static_cast<float>(in_features_));
        Variable bias_var(rand({out_features_}) * (2.0f * bound) - bound, true);
        register_parameter("bias", std::move(bias_var));
    }
}

auto SparseLinear::forward_impl(const Variable& input) -> Variable {
    // input: [batch, in_features]
    // sparse_weight_: [out_features, in_features] in CSR
    // output = spmm(sparse_weight, input^T)^T = [batch, out_features]

    const auto& inp = input.tensor();

    // SpMM: (out_features x in_features) * (in_features x batch) = (out_features x batch)
    auto input_t = inp.permute({1, 0});
    auto result_t = sparse::spmm(sparse_weight_.value(), input_t);
    // Transpose back: (out_features x batch) -> (batch x out_features)
    auto result = result_t.permute({1, 0});

    Variable output(result, input.requires_grad());

    // Add bias
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

    if (has_bias_) {
        float bias_bound = 1.0f / std::sqrt(static_cast<float>(in_features_));
        Variable bias_var(rand({out_features_}) * (2.0f * bias_bound) - bias_bound, true);
        register_parameter("bias", std::move(bias_var));
    }
}

} // namespace tenzor::nn
