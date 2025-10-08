#pragma once

#include "../module.hpp"

namespace tenzor {
namespace nn {

// Dropout layer
class Dropout : public Module {
public:
    explicit Dropout(double p = 0.5);

    auto forward(const Variable& input) -> Variable override;

private:
    double p_;
};

// Dropout2d layer (spatial dropout)
class Dropout2d : public Module {
public:
    explicit Dropout2d(double p = 0.5);

    auto forward(const Variable& input) -> Variable override;

private:
    double p_;
};

// Alpha Dropout (for SELU networks)
class AlphaDropout : public Module {
public:
    explicit AlphaDropout(double p = 0.5);

    auto forward(const Variable& input) -> Variable override;

private:
    double p_;
};

} // namespace nn
} // namespace tenzor
