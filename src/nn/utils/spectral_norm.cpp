/**
 * @file spectral_norm.cpp
 * @brief Spectral normalization implementation using power iteration
 */

#include "tenzor/nn/utils/spectral_norm.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/reduction.hpp"
#include "tenzor/ops/transform.hpp"
#include "tenzor/ops/indexing.hpp"

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
    // Create instance (private constructor, so use new + shared_ptr)
    auto sn = std::shared_ptr<SpectralNorm>(
        new SpectralNorm(module, name, n_power_iterations, eps));

    // Get the weight parameter via named_parameters (public API)
    auto params = module->named_parameters();
    for (auto& [pname, pvar] : params) {
        if (pname == name || pname.ends_with("." + name)) {
            sn->param_ = pvar;
            break;
        }
    }

    if (!sn->param_) {
        throw std::runtime_error(
            "SpectralNorm: parameter '" + name + "' not found in module");
    }

    Tensor weight = sn->param_->tensor();

    // Store original shape (shape() returns span, convert to vector)
    auto wshape = weight.shape();
    sn->original_shape_.assign(wshape.begin(), wshape.end());

    // Reshape weight to 2D: (out_features, in_features_product)
    auto shape = weight.shape();
    int64_t h = shape[0];
    int64_t w = 1;
    for (size_t i = 1; i < shape.size(); ++i) {
        w *= shape[i];
    }

    // Initialize u and v with random normal vectors
    sn->u_ = randn({h}, weight.dtype(), weight.device());
    sn->v_ = randn({w}, weight.dtype(), weight.device());

    // Normalize initial vectors on-device — substitute 1.0 for the divisor
    // when the norm is zero (avoids NaN from 0/0). The previous host-scalar
    // read forced a per-call CPU roundtrip on every spectral-norm setup.
    {
        Tensor u_norm = norm(sn->u_);
        Tensor zero_t = zeros({}, u_norm.dtype(), u_norm.device());
        Tensor mask = ::tenzor::gt(u_norm, zero_t);
        Tensor one_t = ::tenzor::full({}, 1.0, u_norm.dtype(), u_norm.device());
        Tensor safe = ::tenzor::where(mask, u_norm, one_t);
        sn->u_ = div(sn->u_, safe);
    }
    {
        Tensor v_norm = norm(sn->v_);
        Tensor zero_t = zeros({}, v_norm.dtype(), v_norm.device());
        Tensor mask = ::tenzor::gt(v_norm, zero_t);
        Tensor one_t = ::tenzor::full({}, 1.0, v_norm.dtype(), v_norm.device());
        Tensor safe = ::tenzor::where(mask, v_norm, one_t);
        sn->v_ = div(sn->v_, safe);
    }

    // Initial sigma
    sn->sigma_ = ones({1}, weight.dtype(), weight.device());

    // Run a few initial power iterations for better convergence
    Tensor weight_2d = reshape(weight, {h, w});
    for (int64_t i = 0; i < std::max(n_power_iterations, int64_t(3)); ++i) {
        sn->power_iteration(weight_2d);
    }

    // Register forward pre-hook to normalize weight before each forward pass
    // Capture sn by weak_ptr to avoid preventing removal
    std::weak_ptr<SpectralNorm> weak_sn = sn;
    sn->hook_id_ = module->register_forward_pre_hook(
        [weak_sn](Module* /*mod*/, const Variable& /*input*/) {
            auto sn_ptr = weak_sn.lock();
            if (!sn_ptr || !sn_ptr->param_) return;

            // Use the cached parameter pointer (avoids protected member access)
            Tensor weight = sn_ptr->param_->tensor();

            // Run power iteration on 2D view
            auto shape = weight.shape();
            int64_t h = shape[0];
            int64_t w = 1;
            for (size_t i = 1; i < shape.size(); ++i) {
                w *= shape[i];
            }
            Tensor weight_2d = reshape(weight, {h, w});
            sn_ptr->power_iteration(weight_2d);

            // Compute normalized weight and replace parameter
            Tensor normalized = sn_ptr->compute_weight(weight);
            *sn_ptr->param_ = Variable(normalized, sn_ptr->param_->requires_grad());
        });

    return sn;
}

auto SpectralNorm::power_iteration(const Tensor& weight_2d) -> void {
    // weight_2d is (h, w)
    // v = W^T u / ||W^T u||
    // u = W v / ||W v||

    Tensor wt = transpose(weight_2d, 0, 1);  // (w, h)

    for (int64_t i = 0; i < n_power_iterations_; ++i) {
        // v_new = W^T @ u
        // Reshape u to (h, 1), matmul with wt (w, h) won't work directly
        // Use: v = (W^T) @ u  where W^T is (w, h) and u is (h,)
        // We need to reshape for matmul: (w, h) @ (h, 1) -> (w, 1) -> (w,)
        Tensor u_col = reshape(u_, {static_cast<int64_t>(u_.shape()[0]), 1});
        Tensor v_new = reshape(matmul(wt, u_col), {static_cast<int64_t>(v_.shape()[0])});

        // On-device clamp: divide by max(norm, eps) — avoids any host
        // roundtrip in the per-forward power iteration loop.
        auto v_norm_t = norm(v_new);
        Tensor v_eps_t = ::tenzor::full({}, eps_, v_norm_t.dtype(), v_norm_t.device());
        Tensor v_mask = ::tenzor::gt(v_norm_t, v_eps_t);
        Tensor v_safe = ::tenzor::where(v_mask, v_norm_t, ::tenzor::full({}, 1.0, v_norm_t.dtype(), v_norm_t.device()));
        Tensor v_div  = ::tenzor::where(v_mask, v_new,    v_);
        v_ = div(v_div, v_safe);

        // u_new = W @ v
        Tensor v_col = reshape(v_, {static_cast<int64_t>(v_.shape()[0]), 1});
        Tensor u_new = reshape(matmul(weight_2d, v_col), {static_cast<int64_t>(u_.shape()[0])});

        auto u_norm_t = norm(u_new);
        Tensor u_eps_t = ::tenzor::full({}, eps_, u_norm_t.dtype(), u_norm_t.device());
        Tensor u_mask = ::tenzor::gt(u_norm_t, u_eps_t);
        Tensor u_safe = ::tenzor::where(u_mask, u_norm_t, ::tenzor::full({}, 1.0, u_norm_t.dtype(), u_norm_t.device()));
        Tensor u_div  = ::tenzor::where(u_mask, u_new,    u_);
        u_ = div(u_div, u_safe);
    }

    // Compute sigma = u^T W v
    Tensor u_row = reshape(u_, {1, static_cast<int64_t>(u_.shape()[0])});  // (1, h)
    Tensor v_col = reshape(v_, {static_cast<int64_t>(v_.shape()[0]), 1});  // (w, 1)
    Tensor wv = matmul(weight_2d, v_col);  // (h, 1)
    sigma_ = reshape(matmul(u_row, wv), {1});  // scalar
}

auto SpectralNorm::compute_weight(const Tensor& weight) -> Tensor {
    // W_normalized = W / max(sigma, eps) — keeps the divide on-device.
    Tensor eps_t = ::tenzor::full({}, eps_, sigma_.dtype(), sigma_.device());
    Tensor mask = ::tenzor::gt(sigma_, eps_t);
    Tensor safe = ::tenzor::where(mask, sigma_, eps_t);
    if (safe.device() != weight.device()) safe = safe.to(weight.device());
    if (safe.dtype()  != weight.dtype())  safe = safe.to(weight.dtype());
    return div(weight, safe);
}

auto SpectralNorm::remove() -> void {
    if (module_) {
        module_->remove_hook(hook_id_);
        module_.reset();
    }
}

} // namespace tenzor::nn::utils
