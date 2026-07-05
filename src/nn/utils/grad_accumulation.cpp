#include "tenzor/nn/utils/grad_accumulation.hpp"
#include <stdexcept>

namespace tenzor::nn::utils {

namespace {

// Scale every accumulated gradient by 1/divisor in-place before the optimizer
// step. Micro-batch backward() calls SUM their gradients into param->grad();
// PyTorch's standard convention treats the effective batch as the mean over the
// accumulated micro-batches, so we divide the summed gradients by the number of
// micro-batches actually accumulated. Without this the effective gradient (and
// hence the effective learning rate) is `divisor`x too large (F066).
auto scale_accumulated_grads(optim::Optimizer& optimizer, int64_t divisor) -> void {
    if (divisor <= 1) {
        return;  // nothing to scale for a single micro-batch
    }
    const double scale = static_cast<double>(divisor);
    for (const auto& param : optimizer.parameters()) {
        if (param && param->has_grad() && param->grad().has_value()) {
            param->set_grad(*param->grad() / scale);
        }
    }
}

}  // namespace

GradientAccumulator::GradientAccumulator(optim::Optimizer& optimizer, int64_t accumulation_steps)
    : optimizer_(optimizer)
    , accumulation_steps_(accumulation_steps) {
    if (accumulation_steps <= 0) {
        throw std::invalid_argument(
            "GradientAccumulator: accumulation_steps must be positive, got " +
            std::to_string(accumulation_steps));
    }
}

auto GradientAccumulator::step() -> bool {
    ++current_step_;
    if (current_step_ >= accumulation_steps_) {
        // Mean convention: divide the summed grads by the number of accumulated
        // micro-batches so the effective learning rate is correct (F066).
        scale_accumulated_grads(optimizer_, accumulation_steps_);
        optimizer_.step();
        optimizer_.zero_grad();
        current_step_ = 0;
        return true;
    }
    return false;
}

auto GradientAccumulator::flush() -> bool {
    if (current_step_ > 0) {
        // Partial flush: only `current_step_` micro-batches were accumulated, so
        // scale by that count (not accumulation_steps_) to keep the mean correct.
        scale_accumulated_grads(optimizer_, current_step_);
        optimizer_.step();
        optimizer_.zero_grad();
        current_step_ = 0;
        return true;
    }
    return false;
}

auto GradientAccumulator::should_sync() const -> bool {
    return current_step_ == accumulation_steps_ - 1;
}

auto GradientAccumulator::accumulation_steps() const -> int64_t {
    return accumulation_steps_;
}

auto GradientAccumulator::current_step() const -> int64_t {
    return current_step_;
}

auto GradientAccumulator::reset() -> void {
    current_step_ = 0;
    optimizer_.zero_grad();
}

} // namespace tenzor::nn::utils
