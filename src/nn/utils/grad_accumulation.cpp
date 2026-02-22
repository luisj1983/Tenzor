#include "tenzor/nn/utils/grad_accumulation.hpp"
#include <stdexcept>

namespace tenzor::nn::utils {

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
        optimizer_.step();
        optimizer_.zero_grad();
        current_step_ = 0;
        return true;
    }
    return false;
}

auto GradientAccumulator::flush() -> bool {
    if (current_step_ > 0) {
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
