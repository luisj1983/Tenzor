#include "tenzor/nn/optim/sparse_adam.hpp"
#include "tenzor/nn/optim/master_weights.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/indexing.hpp"
#include "tenzor/ops/reduction.hpp"
#include <cmath>

namespace tenzor::optim {

// R.16 master-weights helpers come from include/tenzor/nn/optim/master_weights.hpp.

SparseAdam::SparseAdam(std::vector<std::shared_ptr<Variable>> params,
                       double lr, double beta1, double beta2, double eps)
    : Optimizer(std::move(params)), lr_(lr), beta1_(beta1), beta2_(beta2), eps_(eps) {
    initialize_buffers();
}

SparseAdam::SparseAdam(std::vector<optim::ParamGroup> groups,
                       double default_lr, double default_beta1,
                       double default_beta2, double default_eps)
    : Optimizer(std::move(groups)),
      lr_(default_lr),
      beta1_(default_beta1),
      beta2_(default_beta2),
      eps_(default_eps) {
    initialize_buffers();
}

auto SparseAdam::step_impl() -> void {
    step_count_++;

    // Audit D.4: per-parameter hyperparameters resolve from the
    // active ParamGroup (when one was set up) or fall through to
    // the optimiser-wide defaults stored on this SparseAdam instance.
    // SparseAdam has no weight_decay (PyTorch parity), so the
    // ParamGroup's non-optional weight_decay field is intentionally
    // ignored here.
    struct SparseAdamHP {
        double lr;
        double beta1;
        double beta2;
        double eps;
    };

    auto resolve = [&](size_t i) -> SparseAdamHP {
        SparseAdamHP hp{lr_, beta1_, beta2_, eps_};
        if (const auto* g = find_group_for_param(i)) {
            hp.lr    = g->lr;
            hp.beta1 = ParamGroup::or_else(g->beta1, beta1_);
            hp.beta2 = ParamGroup::or_else(g->beta2, beta2_);
            hp.eps   = ParamGroup::or_else(g->eps,   eps_);
        }
        return hp;
    };

    for (size_t i = 0; i < parameters_.size(); ++i) {
        auto& param_ptr = parameters_[i];
        if (!param_ptr) continue;
        auto& param = *param_ptr;

        // R.20: prefer the sparse gradient when the producing op (Embedding
        // with sparse=true, etc.) populated it. The dense `.grad()` slot is
        // an always-on convenience that mirrors the sparse rows to a dense
        // tensor for engine compatibility; reading from it forces the
        // optimiser through the dense-rows scan path even when the producer
        // already gave us the explicit nonzero indices+values.
        if (!param.has_sparse_grad() && !param.has_grad()) continue;

        const SparseAdamHP hp = resolve(i);

        // R.16: scalars and the rolling state buffers live in state_dt
        // (F32 for half-precision params); grad and param rows are upcast
        // before the math and downcast on assignment.
        const DType param_dt = param.tensor().dtype();
        const DType state_dt = optim_state_dtype(param_dt);
        const bool needs_upcast = (state_dt != param_dt);
        auto scalar = [&](double value) -> Tensor {
            return full({1}, value, state_dt, param.tensor().device());
        };

        // Bias correction factors (same for sparse and dense paths)
        double bias_correction1 = 1.0 - std::pow(hp.beta1, step_count_);
        double bias_correction2 = 1.0 - std::pow(hp.beta2, step_count_);
        double step_size = hp.lr / bias_correction1;

        // Fast path: producer supplied an explicit sparse gradient. Use its
        // COO indices directly instead of scanning the dense buffer to find
        // nonzero rows. Falls through to the dense-rows path on shape
        // mismatch (e.g. sparse_dim != 1).
        if (param.has_sparse_grad()) {
            const auto& sg_opt = param.sparse_grad();
            if (sg_opt.has_value()) {
                const auto& sg = sg_opt.value();
                if (sg.sparse_dim() == 1 && sg.nnz() > 0) {
                    auto row_indices = sg.indices().reshape({sg.nnz()});
                    Tensor grad_rows = needs_upcast ? sg.values().to(state_dt) : sg.values();

                    auto m_rows = index_select(exp_avg_[i], 0, row_indices);
                    auto v_rows = index_select(exp_avg_sq_[i], 0, row_indices);

                    m_rows = m_rows * scalar(hp.beta1) + grad_rows * scalar(1.0 - hp.beta1);
                    v_rows = v_rows * scalar(hp.beta2) +
                             grad_rows * grad_rows * scalar(1.0 - hp.beta2);

                    std::vector<int64_t> idx_shape = {row_indices.shape()[0]};
                    for (int64_t d = 1; d < m_rows.ndim(); ++d) idx_shape.push_back(1);
                    auto idx_expanded = row_indices.reshape(idx_shape);
                    auto shape_span = m_rows.shape();
                    std::vector<int64_t> expand_shape(shape_span.begin(), shape_span.end());
                    auto idx_broadcast = idx_expanded.expand(expand_shape);

                    exp_avg_[i] = scatter(exp_avg_[i], 0, idx_broadcast, m_rows);
                    exp_avg_sq_[i] = scatter(exp_avg_sq_[i], 0, idx_broadcast, v_rows);

                    auto denom = sqrt(v_rows) * scalar(1.0 / std::sqrt(bias_correction2))
                                + scalar(hp.eps);
                    auto update = div(m_rows, denom) * scalar(step_size);

                    Tensor param_rows = needs_upcast
                        ? index_select(param.tensor(), 0, row_indices).to(state_dt)
                        : index_select(param.tensor(), 0, row_indices);
                    param_rows = param_rows - update;
                    Tensor param_rows_out = needs_upcast ? param_rows.to(param_dt) : param_rows;
                    param.tensor() = scatter(param.tensor(), 0, idx_broadcast, param_rows_out);

                    continue;
                }
            }
        }

        if (!param.has_grad()) continue;
        const Tensor& grad = param.grad().value();

        // For 2D+ parameters (e.g., embedding tables), use the sparse path:
        // only update rows that have non-zero gradients.
        // For 1D parameters (biases), fall through to the dense path.
        if (grad.ndim() >= 2) {
            // Flatten trailing dims and check which rows have any non-zero gradient
            // any() along all dims except dim 0 gives a boolean per row
            // Reshape grad to (num_rows, -1) for the reduction
            auto grad_2d = grad.reshape({grad.shape()[0], -1});

            // any(grad_2d != 0, dim=1) => boolean mask per row
            auto row_nonzero = any(grad_2d, /*dim=*/1, /*keepdim=*/false);

            // Get indices of non-zero rows
            auto nz_indices = nonzero(row_nonzero);  // shape: (num_nonzero_rows, 1)

            // If no rows have gradients, skip
            if (nz_indices.numel() == 0) continue;

            // Flatten to 1D index tensor for index_select
            auto row_indices = nz_indices.reshape({-1});

            // Extract corresponding rows from gradient and moment buffers
            // R.16: upcast grad rows to state_dt for half-precision params.
            Tensor grad_rows = needs_upcast
                ? index_select(grad, 0, row_indices).to(state_dt)
                : index_select(grad, 0, row_indices);
            auto m_rows = index_select(exp_avg_[i], 0, row_indices);
            auto v_rows = index_select(exp_avg_sq_[i], 0, row_indices);

            // Update moments for selected rows only
            m_rows = m_rows * scalar(hp.beta1) + grad_rows * scalar(1.0 - hp.beta1);
            v_rows = v_rows * scalar(hp.beta2) + grad_rows * grad_rows * scalar(1.0 - hp.beta2);

            // Write updated moments back using scatter
            // scatter(input, dim, index, src): places src values into input at index positions along dim
            // We need to expand row_indices to match the shape of m_rows for scatter
            // Build index shape with trailing 1s for N-dimensional parameters
            std::vector<int64_t> idx_shape = {row_indices.shape()[0]};
            for (int64_t d = 1; d < m_rows.ndim(); ++d) idx_shape.push_back(1);
            auto idx_expanded = row_indices.reshape(idx_shape);
            // Broadcast index to match moment row shape
            auto shape_span = m_rows.shape();
            std::vector<int64_t> expand_shape(shape_span.begin(), shape_span.end());
            auto idx_broadcast = idx_expanded.expand(expand_shape);

            exp_avg_[i] = scatter(exp_avg_[i], 0, idx_broadcast, m_rows);
            exp_avg_sq_[i] = scatter(exp_avg_sq_[i], 0, idx_broadcast, v_rows);

            // Compute bias-corrected update for affected rows
            auto denom = sqrt(v_rows) * scalar(1.0 / std::sqrt(bias_correction2))
                        + scalar(hp.eps);
            auto update = div(m_rows, denom) * scalar(step_size);

            // Update only the affected parameter rows
            Tensor param_rows = needs_upcast
                ? index_select(param.tensor(), 0, row_indices).to(state_dt)
                : index_select(param.tensor(), 0, row_indices);
            param_rows = param_rows - update;
            Tensor param_rows_out = needs_upcast ? param_rows.to(param_dt) : param_rows;
            param.tensor() = scatter(param.tensor(), 0, idx_broadcast, param_rows_out);
        } else {
            // Dense fallback: standard Adam for 1D parameters (biases, etc.)
            // R.16: half-precision dense fallback also runs in state_dt.
            Tensor grad_copy = needs_upcast ? grad.to(state_dt) : grad.clone();

            exp_avg_[i] = exp_avg_[i] * scalar(hp.beta1) +
                         grad_copy * scalar(1.0 - hp.beta1);
            exp_avg_sq_[i] = exp_avg_sq_[i] * scalar(hp.beta2) +
                            grad_copy * grad_copy * scalar(1.0 - hp.beta2);

            auto denom = sqrt(exp_avg_sq_[i]) * scalar(1.0 / std::sqrt(bias_correction2))
                        + scalar(hp.eps);
            Tensor param_hi = needs_upcast ? param.tensor().to(state_dt) : param.tensor();
            Tensor updated = param_hi - div(exp_avg_[i], denom) * scalar(step_size);
            param.tensor() = needs_upcast ? updated.to(param_dt) : updated;
        }
    }
}

auto SparseAdam::initialize_buffers() -> void {
    exp_avg_.clear();
    exp_avg_sq_.clear();
    for (auto& param : parameters_) {
        if (param) {
            // R.16: half-precision params get Float32 state buffers.
            exp_avg_.push_back(make_optim_state(param->tensor()));
            exp_avg_sq_.push_back(make_optim_state(param->tensor()));
        }
    }
}

// Audit K.1: extend exp_avg_ / exp_avg_sq_ for parameters appended via
// add_param_group.
auto SparseAdam::on_parameters_appended_(size_t old_count, size_t new_count) -> void {
    exp_avg_.reserve(new_count);
    exp_avg_sq_.reserve(new_count);
    for (size_t i = old_count; i < new_count; ++i) {
        const auto& param = parameters_[i];
        if (param) {
            // R.16: see SparseAdam::initialize_buffers.
            exp_avg_.push_back(make_optim_state(param->tensor()));
            exp_avg_sq_.push_back(make_optim_state(param->tensor()));
        } else {
            exp_avg_.push_back(Tensor{});
            exp_avg_sq_.push_back(Tensor{});
        }
    }
}

auto SparseAdam::set_lr(double lr) -> void { lr_ = lr; }
auto SparseAdam::get_lr() const -> double { return lr_; }

auto SparseAdam::state_dict() const -> std::unordered_map<std::string, Tensor> {
    std::unordered_map<std::string, Tensor> state;

    state["step_count"] = Tensor({1}, DType::Int64, Device::cpu());
    state["step_count"].data<int64_t>()[0] = step_count_;

    state["lr"] = Tensor({1}, DType::Float64, Device::cpu());
    state["lr"].data<double>()[0] = lr_;

    state["beta1"] = Tensor({1}, DType::Float64, Device::cpu());
    state["beta1"].data<double>()[0] = beta1_;

    state["beta2"] = Tensor({1}, DType::Float64, Device::cpu());
    state["beta2"].data<double>()[0] = beta2_;

    state["eps"] = Tensor({1}, DType::Float64, Device::cpu());
    state["eps"].data<double>()[0] = eps_;

    for (size_t i = 0; i < exp_avg_.size(); ++i) {
        state["exp_avg_" + std::to_string(i)] = exp_avg_[i].clone();
        state["exp_avg_sq_" + std::to_string(i)] = exp_avg_sq_[i].clone();
    }

    return state;
}

auto SparseAdam::load_state_dict(const std::unordered_map<std::string, Tensor>& state) -> void {
    if (state.count("step_count")) {
        step_count_ = state.at("step_count").data<int64_t>()[0];
    }

    if (state.count("lr")) {
        lr_ = state.at("lr").data<double>()[0];
    }

    if (state.count("beta1")) {
        beta1_ = state.at("beta1").data<double>()[0];
    }

    if (state.count("beta2")) {
        beta2_ = state.at("beta2").data<double>()[0];
    }

    if (state.count("eps")) {
        eps_ = state.at("eps").data<double>()[0];
    }

    // Validate momentum buffer counts match current parameter count
    size_t saved_count = 0;
    for (const auto& [key, _] : state) {
        if (key.starts_with("exp_avg_") && !key.starts_with("exp_avg_sq_")) {
            ++saved_count;
        }
    }
    if (saved_count > 0 && saved_count != exp_avg_.size()) {
        throw std::runtime_error(
            "SparseAdam::load_state_dict: momentum buffer count mismatch - "
            "saved " + std::to_string(saved_count) + " but have " +
            std::to_string(exp_avg_.size()) + " parameters");
    }

    // V.27: cast to R.16 master-weights dtype on load.
    for (size_t i = 0; i < exp_avg_.size(); ++i) {
        std::string ea_key = "exp_avg_" + std::to_string(i);
        std::string eas_key = "exp_avg_sq_" + std::to_string(i);
        const DType state_dt = (i < parameters_.size() && parameters_[i])
            ? optim_state_dtype(parameters_[i]->tensor().dtype())
            : DType::Float32;
        if (state.count(ea_key)) exp_avg_[i] = state.at(ea_key).to(state_dt);
        if (state.count(eas_key)) exp_avg_sq_[i] = state.at(eas_key).to(state_dt);
    }
}

} // namespace tenzor::optim
