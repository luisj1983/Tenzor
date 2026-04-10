/**
 * @file clip_grad.cpp
 * @brief Gradient clipping implementations
 */

#include "tenzor/nn/utils/clip_grad.hpp"
#include "tenzor/ops/creation.hpp"

namespace tenzor::nn::utils {

auto clip_grad_norm_(std::vector<std::shared_ptr<Variable>> parameters,
                     double max_norm,
                     double norm_type) -> double {
    // Collect all gradient tensors
    std::vector<Tensor> grads;
    for (auto& p : parameters) {
        if (p && p->has_grad()) {
            grads.push_back(p->grad().value());
        }
    }

    if (grads.empty()) {
        return 0.0;
    }

    double total_norm = 0.0;

    // Helper to read a scalar norm tensor (may be on GPU, may be F16/BF16)
    auto read_scalar = [](const Tensor& t) -> double {
        Tensor cpu_f32 = t.to(DType::Float32).to(Device::cpu());
        return static_cast<double>(cpu_f32.data<float>()[0]);
    };

    if (std::isinf(norm_type)) {
        // Max norm: find the maximum absolute value across all gradients
        for (auto& g : grads) {
            auto g_abs = tenzor::abs(g);
            auto g_max = tenzor::norm(g_abs, std::numeric_limits<float>::infinity());
            total_norm = std::max(total_norm, read_scalar(g_max));
        }
    } else {
        // p-norm: compute (sum(|g|^p))^(1/p) across all parameters
        for (auto& g : grads) {
            auto g_norm = tenzor::norm(g, static_cast<float>(norm_type));
            double val = read_scalar(g_norm);
            total_norm += std::pow(val, norm_type);
        }
        total_norm = std::pow(total_norm, 1.0 / norm_type);
    }

    double clip_coef = max_norm / (total_norm + 1e-6);

    if (clip_coef < 1.0) {
        for (auto& p : parameters) {
            if (p && p->has_grad()) {
                auto scaled = tenzor::mul(p->grad().value(),
                                          static_cast<float>(clip_coef));
                p->set_grad(scaled);
            }
        }
    }

    return total_norm;
}

void clip_grad_value_(std::vector<std::shared_ptr<Variable>> parameters,
                      double clip_value) {
    for (auto& p : parameters) {
        if (p && p->has_grad()) {
            auto clamped = tenzor::clamp(p->grad().value(),
                                         static_cast<float>(-clip_value),
                                         static_cast<float>(clip_value));
            p->set_grad(clamped);
        }
    }
}

} // namespace tenzor::nn::utils
