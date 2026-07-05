/**
 * @file weight_norm.cpp
 * @brief Weight normalization implementation (autograd-aware reparameterisation)
 *
 * Salimans & Kingma (2016) decompose a weight `w` into magnitude `g` and
 * direction `v`:
 *
 *     w = g * (v / ||v||)
 *
 * PP.1 fix: `g` and `v` are registered as actual module parameters so the
 * optimiser updates them. The pre-hook recomputes `w` on every forward via
 * autograd ops on those parameters, so the resulting `weight` slot carries a
 * grad_fn back to `g` and `v` — gradients then flow into the trainable
 * leaves on backward(), instead of dead-ending on a Variable that the next
 * forward will overwrite.
 */

#include "tenzor/nn/utils/weight_norm.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/reduction.hpp"
#include "tenzor/ops/transform.hpp"
#include "tenzor/autograd/ops.hpp"
#include "tenzor/nn/utils/variable_cast.hpp"
#include "tenzor/utils/safe_math.hpp"

namespace tenzor::nn::utils {

namespace {

// Compute ||v|| along all dimensions except `dim` (keepdim=true) using
// autograd ops on a Variable. The returned Variable carries a grad_fn back
// to the supplied Variable so backward() propagates into the leaf.
auto v_norm_autograd(const Variable& v_var, int64_t dim) -> Variable {
    auto vshape = v_var.tensor().shape();
    std::vector<int64_t> norm_dims;
    norm_dims.reserve(vshape.size());
    for (int64_t d = 0; d < static_cast<int64_t>(vshape.size()); ++d) {
        if (d != dim) norm_dims.push_back(d);
    }

    // F064: the squared magnitude of a complex vector is v*conj(v) = |v|^2
    // (real), not v*v (which squares the complex value). Use the conjugated
    // product for complex weights so the direction norm is correct; leave the
    // real path unchanged.
    Variable sq = v_var.tensor().is_complex()
                      ? v_var * ::tenzor::conj(v_var)
                      : v_var * v_var;
    if (norm_dims.empty()) {
        return ::tenzor::sqrt(sq);
    }
    Variable acc = sq;
    for (int64_t i = static_cast<int64_t>(norm_dims.size()) - 1; i >= 0; --i) {
        acc = ::tenzor::sum(acc, norm_dims[i], /*keepdim=*/true);
    }
    return ::tenzor::sqrt(acc);
}

// Same reduction logic, raw-Tensor variant (used at apply() time when
// initialising `g` from the original weight).
auto v_norm_tensor(const Tensor& v, int64_t dim) -> Tensor {
    auto vshape = v.shape();
    std::vector<int64_t> norm_dims;
    norm_dims.reserve(vshape.size());
    for (int64_t d = 0; d < static_cast<int64_t>(vshape.size()); ++d) {
        if (d != dim) norm_dims.push_back(d);
    }
    if (norm_dims.empty()) {
        return ::tenzor::norm(v);
    }
    // F064: |v|^2 = v*conj(v) for complex weights (v*v squares the complex
    // value instead). Real path unchanged.
    Tensor sq = v.is_complex() ? mul(v, ::tenzor::conj(v)) : mul(v, v);
    Tensor acc = sq;
    for (int64_t i = static_cast<int64_t>(norm_dims.size()) - 1; i >= 0; --i) {
        acc = sum(acc, norm_dims[i], /*keepdim=*/true);
    }
    return ::tenzor::sqrt(acc);
}

} // namespace

WeightNorm::WeightNorm(std::shared_ptr<Module> module,
                       std::string name,
                       int64_t dim)
    : module_(std::move(module))
    , param_name_(std::move(name))
    , dim_(dim) {}

auto WeightNorm::apply(std::shared_ptr<Module> module,
                       const std::string& name,
                       int64_t dim) -> std::shared_ptr<WeightNorm> {
    auto wn = std::shared_ptr<WeightNorm>(new WeightNorm(module, name, dim));

    // Locate the target parameter on the module (or any submodule).
    std::shared_ptr<Variable> orig_param;
    {
        auto params = module->named_parameters();
        for (auto& [pname, pvar] : params) {
            if (pname == name || pname.ends_with("." + name)) {
                orig_param = pvar;
                break;
            }
        }
    }
    if (!orig_param) {
        throw std::runtime_error("WeightNorm: parameter '" + name + "' not found in module");
    }

    Tensor weight = orig_param->tensor();

    // Normalize a negative dim (e.g. -1, a valid PyTorch usage) against the
    // weight's rank, then validate. Without this the reduction-axis builder
    // (`d != dim`) treats every real axis as a non-`dim` axis and reduces over
    // ALL dimensions, producing a scalar ‖v‖ and a wrong reparameterization.
    const int64_t ndim = static_cast<int64_t>(weight.shape().size());
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) {
        throw std::runtime_error("WeightNorm: dim " + std::to_string(dim) +
            " out of range for weight of rank " + std::to_string(ndim));
    }
    wn->dim_ = dim;

    // Initialise g = ||w|| (keepdim, non-`dim` axes) and v = w.
    // Mirror the BF16 widen-narrow trick used in compute_weight_variable
    // so this also works for backends that lack a BF16 sqrt.
    Tensor g_tensor;
    if (weight.dtype() == DType::BFloat16 || weight.dtype() == DType::Float16) {
        const DType orig = weight.dtype();
        Tensor w32 = weight.to(DType::Float32);
        g_tensor = v_norm_tensor(w32, dim).to(orig);
    } else {
        g_tensor = v_norm_tensor(weight, dim);
    }
    Tensor v_tensor = weight.clone();

    // g and v are the trainable leaves the optimiser will step on.
    wn->weight_g_ = std::make_shared<Variable>(std::move(g_tensor), /*requires_grad=*/true);
    wn->weight_v_ = std::make_shared<Variable>(std::move(v_tensor), /*requires_grad=*/true);

    // Register g and v under the layer's own parameter map. The exact
    // shared_ptrs are kept alive both by `wn` and by `module->parameters_`,
    // so gradients accumulated during backward are visible via both
    // handles.
    module->register_parameter_shared(name + "_g", wn->weight_g_);
    module->register_parameter_shared(name + "_v", wn->weight_v_);

    // The original `weight` slot stays in parameters_ because layer code
    // (Linear, Conv2d, …) reads it by name. But it must no longer be a
    // trainable leaf: replace it with a non-grad placeholder, then on the
    // first forward the pre-hook overwrites the slot's contents with an
    // autograd-derived Variable that carries grad_fn back to g/v.
    {
        wn->weight_slot_ = std::make_shared<Variable>(weight.clone(),
                                                      /*requires_grad=*/false);
        module->unregister_parameter(name);
        module->register_parameter_shared(name, wn->weight_slot_);
    }

    // Forward pre-hook: recompute w from g, v via autograd and write into
    // the slot. The slot's shared_ptr identity is preserved, so any layer
    // that cached a reference to parameters_.at(name) keeps observing the
    // refreshed Variable.
    std::weak_ptr<WeightNorm> weak_wn = wn;
    wn->hook_id_ = module->register_forward_pre_hook(
        [weak_wn](Module* /*mod*/, const Variable& /*input*/) {
            auto wn_ptr = weak_wn.lock();
            if (!wn_ptr || !wn_ptr->weight_slot_) return;
            *wn_ptr->weight_slot_ = wn_ptr->compute_weight_variable();
        });

    // Prime the slot with the autograd-tracked value so a forward fired
    // before any explicit hook trigger still uses the correct weight.
    *wn->weight_slot_ = wn->compute_weight_variable();

    return wn;
}

auto WeightNorm::compute_weight_variable() -> Variable {
    if (!weight_g_ || !weight_v_) {
        throw std::runtime_error(
            "WeightNorm::compute_weight_variable: weight_g_ or weight_v_ is null. "
            "Call apply() on the parent Module to register the "
            "weight_norm reparameterization before invoking compute_weight.");
    }

    // Float16 / BFloat16 widen-narrow: sqrt / div / mul lack a BFloat16
    // dispatch on several backends (see user feedback: F16 widen-narrow
    // pattern). Compute the reparameterisation in Float32 when the dtype
    // is BF16 (or any future "narrow" type) and cast back so the resulting
    // weight matches the original dtype.
    DType orig_dtype = weight_v_->tensor().dtype();
    bool widen = (orig_dtype == DType::BFloat16 || orig_dtype == DType::Float16);

    Variable g_eff = widen ? tenzor::nn::variable_cast(*weight_g_, DType::Float32) : *weight_g_;
    Variable v_eff = widen ? tenzor::nn::variable_cast(*weight_v_, DType::Float32) : *weight_v_;

    Variable v_norm_var = v_norm_autograd(v_eff, dim_);

    // QQ.11: build eps via dtype_epsilon — 1e-12f underflows F16 to zero
    // and reintroduces the divide-by-zero footgun.
    double eps_d = ::tenzor::detail::dtype_epsilon(v_norm_var.tensor().dtype());
    auto vshape = v_norm_var.tensor().shape();
    std::vector<int64_t> vshape_v(vshape.begin(), vshape.end());
    Tensor eps_t = full(vshape_v, eps_d,
                        v_norm_var.tensor().dtype(),
                        v_norm_var.tensor().device());
    Variable eps_var(eps_t, /*requires_grad=*/false);
    Variable v_norm_safe = v_norm_var + eps_var;

    // w = g * (v / (||v|| + eps))
    Variable w = g_eff * (v_eff / v_norm_safe);
    if (widen) {
        w = tenzor::nn::variable_cast(w, orig_dtype);
    }
    return w;
}

auto WeightNorm::compute_weight() -> Tensor {
    return compute_weight_variable().tensor();
}

auto WeightNorm::remove() -> void {
    if (module_) {
        // Materialise the current weight as a regular trainable leaf, so
        // the layer keeps training without the reparameterisation.
        Tensor final_t = compute_weight_variable().tensor().clone();
        auto leaf = std::make_shared<Variable>(std::move(final_t),
                                               /*requires_grad=*/true);

        module_->unregister_parameter(param_name_);
        module_->register_parameter_shared(param_name_, leaf);

        try { module_->unregister_parameter(param_name_ + "_g"); } catch (...) {}
        try { module_->unregister_parameter(param_name_ + "_v"); } catch (...) {}

        module_->remove_hook(hook_id_);
        weight_slot_.reset();
        module_.reset();
    }
}

} // namespace tenzor::nn::utils
