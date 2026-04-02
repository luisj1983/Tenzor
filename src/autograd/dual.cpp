#include "tenzor/autograd/dual.hpp"
#include "tenzor/ops/creation.hpp"

namespace tenzor {

// Thread-local dual-mode flag
static thread_local bool g_dual_mode = false;

DualTensor::DualTensor(Tensor primal, Tensor tangent)
    : primal_(std::move(primal)), tangent_(std::move(tangent)) {}

DualTensor::DualTensor(Tensor primal)
    : primal_(std::move(primal)), tangent_(zeros_like(primal_)) {}

auto is_dual_mode() -> bool {
    return g_dual_mode;
}

void set_dual_mode(bool enabled) {
    g_dual_mode = enabled;
}

DualModeGuard::DualModeGuard()
    : prev_state_(g_dual_mode) {
    g_dual_mode = true;
}

DualModeGuard::~DualModeGuard() {
    g_dual_mode = prev_state_;
}

} // namespace tenzor
