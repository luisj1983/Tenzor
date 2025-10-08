#include "tenzor/nn/loss/losses.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/reduction.hpp"

namespace tenzor::nn {

MSELoss::MSELoss(Reduction reduction) : reduction_(reduction) {}

auto MSELoss::forward(const Variable& input, const Variable& target) -> Variable {
    auto diff = input - target;
    auto squared = diff * diff;

    switch (reduction_) {
        case Reduction::None:
            return squared;
        case Reduction::Mean:
            return Variable(tenzor::mean(squared.tensor()), true);
        case Reduction::Sum:
            return Variable(tenzor::sum(squared.tensor()), true);
    }
    return squared;
}

CrossEntropyLoss::CrossEntropyLoss(Reduction reduction) : reduction_(reduction) {}

auto CrossEntropyLoss::forward(const Variable& input, const Tensor& target) -> Variable {
    // TODO: Implement cross entropy loss
    return input;
}

// Functional implementations
auto mse_loss(const Variable& input, const Variable& target,
             Reduction reduction) -> Variable {
    MSELoss loss(reduction);
    return loss.forward(input, target);
}

} // namespace tenzor::nn
