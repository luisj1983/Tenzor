#include "tenzor/nn/layers/dropout.hpp"

namespace tenzor::nn {

Dropout::Dropout(double p) : p_(p) {}

auto Dropout::forward(const Variable& input) -> Variable {
    if (!is_training()) {
        return input;
    }
    // TODO: Implement dropout with random mask
    return input;
}

Dropout2d::Dropout2d(double p) : p_(p) {}

auto Dropout2d::forward(const Variable& input) -> Variable {
    if (!is_training()) {
        return input;
    }
    // TODO: Implement 2D dropout
    return input;
}

} // namespace tenzor::nn
