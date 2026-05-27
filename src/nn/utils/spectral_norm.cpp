/**
 * @file spectral_norm.cpp
 * @brief Spectral normalization implementation using power iteration
 *
 * Miyato et al. (2018) constrain the spectral norm of a weight matrix to
 * stabilise GAN training:
 *
 *     W_normalised = W / sigma(W)
 *
 * PP.2 fix: `weight_orig` is the trainable parameter (registered on the
 * module). The pre-hook re-runs power iteration (raw-Tensor, NOT
 * differentiable — only the W/sigma divide participates in autograd) and
 * writes an autograd-derived Variable into the layer's `weight` slot, so
 * gradients flow into `weight_orig` on backward(). Power-iteration vectors
 * `u`, `v` are registered as buffers (non-trainable state).
 */

#include "tenzor/nn/utils/spectral_norm.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/reduction.hpp"
#include "tenzor/ops/transform.hpp"
#include "tenzor/ops/indexing.hpp"
#include "tenzor/autograd/ops.hpp"
#include "tenzor/utils/safe_math.hpp"

#include <algorithm>

namespace {
/// Floor the caller-supplied eps_ at the dtype's representable epsilon so
/// the where-mask doesn't compare/replace with a value that has already
/// underflowed to zero in F16/BF16 (QQ.11).
inline double floored_eps(double eps_in, ::tenzor::DType dtype) {
    return std::max(eps_in, ::tenzor::detail::dtype_epsilon(dtype));
}
} // namespace

namespace tenzor::nn::utils {

SpectralNorm::SpectralNorm(std::shared_ptr<Module> module,
                           std::string name,
                           int64_t n_power_iterations,
                           double eps)
    : module_(std::move(module))
    , param_name_(std::move(name))
    , n_power_iterations_(n_power_iterations)
    , eps_(eps) {}

auto SpectralNorm::apply(std::shared_ptr<Module> module,
                         const std::string& name,
                         int64_t n_power_iterations,
                         double eps) -> std::shared_ptr<SpectralNorm> {
    auto sn = std::shared_ptr<SpectralNorm>(
        new SpectralNorm(module, name, n_power_iterations, eps));

    // Locate the parameter we are wrapping.
    std::shared_ptr<Variable> orig_slot;
    {
        auto params = module->named_parameters();
        for (auto& [pname, pvar] : params) {
            if (pname == name || pname.ends_with("." + name)) {
                orig_slot = pvar;
                break;
            }
        }
    }
    if (!orig_slot) {
        throw std::runtime_error(
            "SpectralNorm: parameter '" + name + "' not found in module");
    }

    Tensor weight = orig_slot->tensor();
    auto wshape = weight.shape();
    sn->original_shape_.assign(wshape.begin(), wshape.end());

    int64_t h = wshape[0];
    int64_t w_dim = 1;
    for (size_t i = 1; i < wshape.size(); ++i) {
        w_dim *= wshape[i];
    }

    // Initialise u, v with random unit vectors (raw Tensor, no autograd).
    sn->u_ = randn({h}, weight.dtype(), weight.device());
    sn->v_ = randn({w_dim}, weight.dtype(), weight.device());

    auto unit_normalise = [](Tensor& vec) {
        Tensor n = norm(vec);
        Tensor zero_t = zeros({}, n.dtype(), n.device());
        Tensor mask   = ::tenzor::gt(n, zero_t);
        Tensor one_t  = ::tenzor::full({}, 1.0, n.dtype(), n.device());
        Tensor safe   = ::tenzor::where(mask, n, one_t);
        vec = div(vec, safe);
    };
    unit_normalise(sn->u_);
    unit_normalise(sn->v_);

    sn->sigma_ = ones({1}, weight.dtype(), weight.device());

    // Run a few warm-up power iterations for a better initial sigma.
    Tensor weight_2d = reshape(weight, {h, w_dim});
    for (int64_t i = 0; i < std::max(n_power_iterations, int64_t(3)); ++i) {
        sn->power_iteration(weight_2d);
    }

    // weight_orig is the trainable leaf — the optimiser steps on this.
    sn->weight_orig_ = std::make_shared<Variable>(weight.clone(),
                                                  /*requires_grad=*/true);
    module->register_parameter_shared(name + "_orig", sn->weight_orig_);

    // u, v are non-trainable buffers (state for power iteration).
    sn->buffer_u_ = std::make_shared<Variable>(sn->u_, /*requires_grad=*/false);
    sn->buffer_v_ = std::make_shared<Variable>(sn->v_, /*requires_grad=*/false);
    module->register_buffer_shared(name + "_u", sn->buffer_u_);
    module->register_buffer_shared(name + "_v", sn->buffer_v_);

    // The layer's `weight` slot stays in parameters_ (Linear etc. read it
    // by name), but it becomes derived: the pre-hook overwrites its
    // contents with an autograd Variable whose grad_fn traces back to
    // weight_orig_.
    {
        sn->param_ = std::make_shared<Variable>(weight.clone(),
                                                /*requires_grad=*/false);
        module->unregister_parameter(name);
        module->register_parameter_shared(name, sn->param_);
    }

    // Pre-hook: run power iteration on the current `weight_orig` (raw
    // Tensor — power iteration itself is not differentiable), then write
    // `weight_orig / sigma` (autograd-tracked) into the slot.
    std::weak_ptr<SpectralNorm> weak_sn = sn;
    sn->hook_id_ = module->register_forward_pre_hook(
        [weak_sn](Module* /*mod*/, const Variable& /*input*/) {
            auto sn_ptr = weak_sn.lock();
            if (!sn_ptr || !sn_ptr->weight_orig_ || !sn_ptr->param_) return;

            Tensor w_orig_t = sn_ptr->weight_orig_->tensor();
            auto shape = w_orig_t.shape();
            int64_t h = shape[0];
            int64_t w_dim = 1;
            for (size_t i = 1; i < shape.size(); ++i) {
                w_dim *= shape[i];
            }
            Tensor weight_2d = reshape(w_orig_t, {h, w_dim});
            sn_ptr->power_iteration(weight_2d);

            // Compose W / sigma as an autograd op on weight_orig_, so
            // backward flows into the trainable leaf.
            *sn_ptr->param_ = sn_ptr->compute_weight_variable();
        });

    // Prime the slot with the autograd-tracked initial value.
    *sn->param_ = sn->compute_weight_variable();

    return sn;
}

auto SpectralNorm::power_iteration(const Tensor& weight_2d) -> void {
    // weight_2d is (h, w)
    Tensor wt = transpose(weight_2d, 0, 1);  // (w, h)

    for (int64_t i = 0; i < n_power_iterations_; ++i) {
        // v_new = W^T @ u, reshape from (w,1) -> (w,)
        Tensor u_col = reshape(u_, {static_cast<int64_t>(u_.shape()[0]), 1});
        Tensor v_new = reshape(matmul(wt, u_col),
                               {static_cast<int64_t>(v_.shape()[0])});

        // QQ.11: keep eps as-is (caller supplied), but clamp on-device with
        // a where(mask, ...) idiom so a zero norm becomes 1.0 (preserving
        // the previous direction) instead of a NaN/Inf.
        auto v_norm_t = norm(v_new);
        Tensor v_eps_t = ::tenzor::full({}, floored_eps(eps_, v_norm_t.dtype()),
                                        v_norm_t.dtype(), v_norm_t.device());
        Tensor v_mask  = ::tenzor::gt(v_norm_t, v_eps_t);
        Tensor v_one_t = ::tenzor::full({}, 1.0, v_norm_t.dtype(), v_norm_t.device());
        Tensor v_safe  = ::tenzor::where(v_mask, v_norm_t, v_one_t);
        Tensor v_div   = ::tenzor::where(v_mask, v_new,    v_);
        v_ = div(v_div, v_safe);

        Tensor v_col = reshape(v_, {static_cast<int64_t>(v_.shape()[0]), 1});
        Tensor u_new = reshape(matmul(weight_2d, v_col),
                               {static_cast<int64_t>(u_.shape()[0])});

        auto u_norm_t = norm(u_new);
        Tensor u_eps_t = ::tenzor::full({}, floored_eps(eps_, u_norm_t.dtype()),
                                        u_norm_t.dtype(), u_norm_t.device());
        Tensor u_mask  = ::tenzor::gt(u_norm_t, u_eps_t);
        Tensor u_one_t = ::tenzor::full({}, 1.0, u_norm_t.dtype(), u_norm_t.device());
        Tensor u_safe  = ::tenzor::where(u_mask, u_norm_t, u_one_t);
        Tensor u_div   = ::tenzor::where(u_mask, u_new,    u_);
        u_ = div(u_div, u_safe);
    }

    // sigma = u^T W v
    Tensor u_row = reshape(u_, {1, static_cast<int64_t>(u_.shape()[0])});
    Tensor v_col = reshape(v_, {static_cast<int64_t>(v_.shape()[0]), 1});
    Tensor wv = matmul(weight_2d, v_col);
    sigma_ = reshape(matmul(u_row, wv), {1});

    // Keep the buffer Variables in sync so state_dict() / serialisation
    // sees the up-to-date values.
    if (buffer_u_) *buffer_u_ = Variable(u_, /*requires_grad=*/false);
    if (buffer_v_) *buffer_v_ = Variable(v_, /*requires_grad=*/false);
}

// Helper used by the pre-hook and remove(): build W_normalised as an
// autograd Variable rooted at weight_orig_, so backward flows into the
// trainable leaf.
auto SpectralNorm::compute_weight_variable() -> Variable {
    if (!weight_orig_) {
        throw std::runtime_error(
            "SpectralNorm::compute_weight_variable: weight_orig_ is null. "
            "Call SpectralNorm::apply() to install the reparameterisation.");
    }

    // Build a non-differentiable safe-sigma scalar (the power iteration
    // is intentionally not part of the autograd graph — only the divide
    // is). Use a fresh constant Tensor so backward doesn't try to
    // differentiate sigma_.
    Tensor eps_t = ::tenzor::full({}, floored_eps(eps_, sigma_.dtype()),
                                   sigma_.dtype(), sigma_.device());
    Tensor mask  = ::tenzor::gt(sigma_, eps_t);
    Tensor safe  = ::tenzor::where(mask, sigma_, eps_t);
    if (safe.device() != weight_orig_->tensor().device()) safe = safe.to(weight_orig_->tensor().device());
    if (safe.dtype()  != weight_orig_->tensor().dtype())  safe = safe.to(weight_orig_->tensor().dtype());

    Variable safe_var(safe, /*requires_grad=*/false);
    return (*weight_orig_) / safe_var;
}

auto SpectralNorm::compute_weight(const Tensor& /*weight*/) -> Tensor {
    return compute_weight_variable().tensor();
}

auto SpectralNorm::remove() -> void {
    if (module_) {
        // Materialise the current normalised weight as a regular trainable
        // leaf so the layer keeps training without the reparameterisation.
        Tensor final_t = compute_weight_variable().tensor().clone();
        auto leaf = std::make_shared<Variable>(std::move(final_t),
                                               /*requires_grad=*/true);

        module_->unregister_parameter(param_name_);
        module_->register_parameter_shared(param_name_, leaf);

        try { module_->unregister_parameter(param_name_ + "_orig"); } catch (...) {}
        try { module_->unregister_buffer(param_name_ + "_u"); } catch (...) {}
        try { module_->unregister_buffer(param_name_ + "_v"); } catch (...) {}

        module_->remove_hook(hook_id_);
        param_.reset();
        weight_orig_.reset();
        buffer_u_.reset();
        buffer_v_.reset();
        module_.reset();
    }
}

} // namespace tenzor::nn::utils
