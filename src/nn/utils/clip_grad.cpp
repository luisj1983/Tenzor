/**
 * @file clip_grad.cpp
 * @brief Gradient clipping implementations
 */

#include "tenzor/nn/utils/clip_grad.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/reduction.hpp"

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

    // Accumulate all per-parameter norms (or norm^p) into a single Float32
    // scalar tensor on each grad's device, then read it once at the end.
    // Previously this loop did N host-roundtrips per training step (one per
    // parameter); the new shape does at most 1 read regardless of N.
    Tensor accum;  // scalar on the first grad's device, Float32

    auto promote = [](const Tensor& t) -> Tensor {
        return (t.dtype() == DType::Float32) ? t : t.to(DType::Float32);
    };

    if (std::isinf(norm_type)) {
        for (auto& g : grads) {
            auto g_max = tenzor::norm(tenzor::abs(g),
                                      std::numeric_limits<float>::infinity());
            Tensor m_f32 = promote(g_max);
            if (!accum.impl()) {
                accum = m_f32;
            } else {
                if (m_f32.device() != accum.device()) m_f32 = m_f32.to(accum.device());
                accum = tenzor::maximum(accum, m_f32);
            }
        }
    } else {
        for (auto& g : grads) {
            auto g_norm = tenzor::norm(g, static_cast<float>(norm_type));
            Tensor n_f32 = promote(g_norm);
            Tensor contribution = ::tenzor::pow(n_f32, static_cast<float>(norm_type));
            if (!accum.impl()) {
                accum = contribution;
            } else {
                if (contribution.device() != accum.device()) {
                    contribution = contribution.to(accum.device());
                }
                accum = ::tenzor::add(accum, contribution);
            }
        }
        accum = ::tenzor::pow(accum, static_cast<float>(1.0 / norm_type));
    }

    // Single host readback for the final scalar — needed both to return
    // the value to the caller and to decide whether to scale.
    Tensor accum_cpu = accum.to(Device::cpu()).to(DType::Float32);
    double total_norm = static_cast<double>(accum_cpu.data<float>()[0]);

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
