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

    // Accumulate all per-parameter norms (or norm^p) into a single scalar
    // tensor on each grad's device, then read it once at the end. Previously
    // this loop did N host-roundtrips per training step (one per parameter);
    // the new shape does at most 1 read regardless of N.
    //
    // Pick the accumulation dtype by scanning the grads first: if any grad is
    // Float64, accumulate in Float64 (and read back as double) so the norm of
    // double-precision grads is not silently truncated to Float32; otherwise
    // Float32 is sufficient and cheaper.
    DType accum_dtype = DType::Float32;
    for (auto& g : grads) {
        if (g.dtype() == DType::Float64) {
            accum_dtype = DType::Float64;
            break;
        }
    }

    Tensor accum;  // scalar on the first grad's device, accum_dtype

    auto promote = [accum_dtype](const Tensor& t) -> Tensor {
        return (t.dtype() == accum_dtype) ? t : t.to(accum_dtype);
    };

    // Widen Float16/BFloat16 grads to Float32 BEFORE reducing: norm(g, p) for
    // p==2 sums squares, and an F16 sum-of-squares overflows to Inf for grads
    // with magnitude > ~255 (F16 max 65504). The reduction must therefore run
    // in at least Float32, not the grad's native half precision (F062). (F64
    // grads are left as F64 so their norm is not truncated.)
    auto widen_grad = [](const Tensor& g) -> Tensor {
        return (g.dtype() == DType::Float16 || g.dtype() == DType::BFloat16)
            ? g.to(DType::Float32)
            : g;
    };

    if (std::isinf(norm_type)) {
        for (auto& g : grads) {
            auto g_max = tenzor::norm(tenzor::abs(widen_grad(g)),
                                      std::numeric_limits<float>::infinity());
            Tensor m_acc = promote(g_max);
            if (!accum.impl()) {
                accum = m_acc;
            } else {
                if (m_acc.device() != accum.device()) m_acc = m_acc.to(accum.device());
                accum = tenzor::maximum(accum, m_acc);
            }
        }
    } else {
        for (auto& g : grads) {
            auto g_norm = tenzor::norm(widen_grad(g), static_cast<float>(norm_type));
            Tensor n_acc = promote(g_norm);
            Tensor contribution = ::tenzor::pow(n_acc, static_cast<float>(norm_type));
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
    Tensor accum_cpu = accum.to(Device::cpu()).to(accum_dtype);
    double total_norm = (accum_dtype == DType::Float64)
        ? accum_cpu.data<double>()[0]
        : static_cast<double>(accum_cpu.data<float>()[0]);

    double clip_coef = max_norm / (total_norm + 1e-6);

    if (clip_coef < 1.0) {
        for (auto& p : parameters) {
            if (p && p->has_grad()) {
                // mul() takes a double scalar; pass clip_coef directly so
                // Float64 grads aren't scaled by a float-truncated coefficient.
                auto scaled = tenzor::mul(p->grad().value(), clip_coef);
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
            // clamp() takes double bounds; pass ±clip_value directly so
            // Float64 grads aren't clamped to float-truncated bounds.
            auto clamped = tenzor::clamp(p->grad().value(),
                                         -clip_value,
                                         clip_value);
            p->set_grad(clamped);
        }
    }
}

} // namespace tenzor::nn::utils
