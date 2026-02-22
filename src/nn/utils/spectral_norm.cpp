/**
 * @file spectral_norm.cpp
 * @brief Spectral normalization implementation using power iteration
 */

#include "tenzor/nn/utils/spectral_norm.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/reduction.hpp"
#include "tenzor/ops/transform.hpp"

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

    // Normalize initial vectors
    auto u_norm = norm(sn->u_);
    float u_norm_val = u_norm.data<float>()[0];
    if (u_norm_val > 0) {
        sn->u_ = div(sn->u_, u_norm_val);
    }

    auto v_norm = norm(sn->v_);
    float v_norm_val = v_norm.data<float>()[0];
    if (v_norm_val > 0) {
        sn->v_ = div(sn->v_, v_norm_val);
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

        auto v_norm_t = norm(v_new);
        float v_norm_val = v_norm_t.data<float>()[0];
        if (v_norm_val > eps_) {
            v_ = div(v_new, v_norm_val);
        }

        // u_new = W @ v
        Tensor v_col = reshape(v_, {static_cast<int64_t>(v_.shape()[0]), 1});
        Tensor u_new = reshape(matmul(weight_2d, v_col), {static_cast<int64_t>(u_.shape()[0])});

        auto u_norm_t = norm(u_new);
        float u_norm_val = u_norm_t.data<float>()[0];
        if (u_norm_val > eps_) {
            u_ = div(u_new, u_norm_val);
        }
    }

    // Compute sigma = u^T W v
    Tensor u_row = reshape(u_, {1, static_cast<int64_t>(u_.shape()[0])});  // (1, h)
    Tensor v_col = reshape(v_, {static_cast<int64_t>(v_.shape()[0]), 1});  // (w, 1)
    Tensor wv = matmul(weight_2d, v_col);  // (h, 1)
    sigma_ = reshape(matmul(u_row, wv), {1});  // scalar
}

auto SpectralNorm::compute_weight(const Tensor& weight) -> Tensor {
    // W_normalized = W / sigma(W)
    float sigma_val = sigma_.data<float>()[0];
    if (sigma_val < eps_) {
        sigma_val = static_cast<float>(eps_);
    }
    return div(weight, sigma_val);
}

auto SpectralNorm::remove() -> void {
    if (module_) {
        module_->remove_hook(hook_id_);
        module_.reset();
    }
}

} // namespace tenzor::nn::utils
