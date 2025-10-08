#pragma once

#include "../module.hpp"

namespace tenzor {
namespace nn {

// ReLU activation
class ReLU : public Module {
public:
    ReLU() = default;
    auto forward(const Variable& input) -> Variable override;
};

// Leaky ReLU activation
class LeakyReLU : public Module {
public:
    explicit LeakyReLU(double negative_slope = 0.01);
    auto forward(const Variable& input) -> Variable override;

private:
    double negative_slope_;
};

// Sigmoid activation
class Sigmoid : public Module {
public:
    Sigmoid() = default;
    auto forward(const Variable& input) -> Variable override;
};

// Tanh activation
class Tanh : public Module {
public:
    Tanh() = default;
    auto forward(const Variable& input) -> Variable override;
};

// GELU activation
class GELU : public Module {
public:
    GELU() = default;
    auto forward(const Variable& input) -> Variable override;
};

// Softmax activation
class Softmax : public Module {
public:
    explicit Softmax(int64_t dim = -1);
    auto forward(const Variable& input) -> Variable override;

private:
    int64_t dim_;
};

// LogSoftmax activation
class LogSoftmax : public Module {
public:
    explicit LogSoftmax(int64_t dim = -1);
    auto forward(const Variable& input) -> Variable override;

private:
    int64_t dim_;
};

// ELU activation
class ELU : public Module {
public:
    explicit ELU(double alpha = 1.0);
    auto forward(const Variable& input) -> Variable override;

private:
    double alpha_;
};

// SELU activation
class SELU : public Module {
public:
    SELU() = default;
    auto forward(const Variable& input) -> Variable override;
};

// Swish (SiLU) activation
class Swish : public Module {
public:
    Swish() = default;
    auto forward(const Variable& input) -> Variable override;
};

// Mish activation
class Mish : public Module {
public:
    Mish() = default;
    auto forward(const Variable& input) -> Variable override;
};

// Functional activation functions (stateless)
auto relu(const Variable& input) -> Variable;
auto leaky_relu(const Variable& input, double negative_slope = 0.01) -> Variable;
auto sigmoid(const Variable& input) -> Variable;
auto tanh(const Variable& input) -> Variable;
auto gelu(const Variable& input) -> Variable;
auto softmax(const Variable& input, int64_t dim = -1) -> Variable;
auto log_softmax(const Variable& input, int64_t dim = -1) -> Variable;
auto elu(const Variable& input, double alpha = 1.0) -> Variable;
auto selu(const Variable& input) -> Variable;
auto swish(const Variable& input) -> Variable;
auto mish(const Variable& input) -> Variable;

} // namespace nn
} // namespace tenzor
