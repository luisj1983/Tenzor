/**
 * @file weight_norm.cpp
 * @brief Weight normalization implementation
 */

#include "tenzor/nn/utils/weight_norm.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/reduction.hpp"
#include "tenzor/ops/transform.hpp"

namespace tenzor::nn::utils {

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

    // Find the target parameter
    std::shared_ptr<Variable> orig_param;
    auto params = module->named_parameters();
    for (auto& [pname, pvar] : params) {
        if (pname == name || pname.ends_with("." + name)) {
            orig_param = pvar;
            break;
        }
    }
    if (!orig_param) {
        throw std::runtime_error("WeightNorm: parameter '" + name + "' not found in module");
    }

    Tensor weight = orig_param->tensor();

    // Compute initial g (magnitude) and v (direction)
    // g = ||weight|| along all dims except `dim`
    // For a 2D weight [out, in] with dim=0: g has shape [out, 1]
    auto wshape = weight.shape();
    std::vector<int64_t> norm_dims;
    for (int64_t d = 0; d < static_cast<int64_t>(wshape.size()); ++d) {
        if (d != dim) norm_dims.push_back(d);
    }

    // Compute per-output-channel norm
    Tensor g_tensor;
    if (norm_dims.empty()) {
        g_tensor = norm(weight);
    } else {
        // Compute norm over all non-dim dimensions
        Tensor squared = mul(weight, weight);
        Tensor sum_sq = squared;
        // Reduce over non-dim dimensions (from highest to lowest to preserve indices)
        for (int64_t i = static_cast<int64_t>(norm_dims.size()) - 1; i >= 0; --i) {
            sum_sq = sum(sum_sq, norm_dims[i], true);
        }
        g_tensor = sqrt(sum_sq);
    }

    // v = weight (direction, will be normalized in compute_weight)
    Tensor v_tensor = weight.clone();

    // Store g and v as internal state (not registered as module parameters
    // since register_parameter is protected; they're updated via the hook)
    wn->weight_g_ = std::make_shared<Variable>(g_tensor, orig_param->requires_grad());
    wn->weight_v_ = std::make_shared<Variable>(v_tensor, orig_param->requires_grad());

    // Register forward pre-hook
    std::weak_ptr<WeightNorm> weak_wn = wn;
    wn->hook_id_ = module->register_forward_pre_hook(
        [weak_wn, name](Module* mod, const Variable& /*input*/) {
            auto wn_ptr = weak_wn.lock();
            if (!wn_ptr) return;

            Tensor normalized_weight = wn_ptr->compute_weight();
            // Update the original weight parameter
            auto params = mod->named_parameters();
            for (auto& [pname, pvar] : params) {
                if (pname == name || pname.ends_with("." + name)) {
                    *pvar = Variable(normalized_weight, pvar->requires_grad());
                    break;
                }
            }
        });

    return wn;
}

auto WeightNorm::compute_weight() -> Tensor {
    if (!weight_g_ || !weight_v_) return Tensor{};

    Tensor g = weight_g_->tensor();
    Tensor v = weight_v_->tensor();

    // Compute ||v|| along non-dim dimensions
    auto vshape = v.shape();
    std::vector<int64_t> norm_dims;
    for (int64_t d = 0; d < static_cast<int64_t>(vshape.size()); ++d) {
        if (d != dim_) norm_dims.push_back(d);
    }

    Tensor v_norm;
    if (norm_dims.empty()) {
        v_norm = norm(v);
    } else {
        Tensor squared = mul(v, v);
        Tensor sum_sq = squared;
        for (int64_t i = static_cast<int64_t>(norm_dims.size()) - 1; i >= 0; --i) {
            sum_sq = sum(sum_sq, norm_dims[i], true);
        }
        v_norm = sqrt(sum_sq);
    }

    // Add epsilon for numerical stability
    float eps = 1e-12f;
    auto eps_tensor = full(std::vector<int64_t>(v_norm.shape().begin(), v_norm.shape().end()),
                           eps, v_norm.dtype(), v_norm.device());
    v_norm = add(v_norm, eps_tensor);

    // w = g * (v / ||v||)
    return mul(g, div(v, v_norm));
}

auto WeightNorm::remove() -> void {
    if (module_) {
        // Compute final weight and set as the parameter
        Tensor final_weight = compute_weight();
        auto params = module_->named_parameters();
        for (auto& [pname, pvar] : params) {
            if (pname == param_name_ || pname.ends_with("." + param_name_)) {
                *pvar = Variable(final_weight, pvar->requires_grad());
                break;
            }
        }

        module_->remove_hook(hook_id_);
        module_.reset();
    }
}

} // namespace tenzor::nn::utils
