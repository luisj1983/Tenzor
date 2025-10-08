#pragma once

#include "../module.hpp"

namespace tenzor {
namespace nn {

// Reduction mode for losses
enum class Reduction {
    None,
    Mean,
    Sum
};

// Mean Squared Error Loss
class MSELoss {
public:
    explicit MSELoss(Reduction reduction = Reduction::Mean);

    auto forward(const Variable& input, const Variable& target) -> Variable;
    auto operator()(const Variable& input, const Variable& target) -> Variable {
        return forward(input, target);
    }

private:
    Reduction reduction_;
};

// Cross Entropy Loss
class CrossEntropyLoss {
public:
    explicit CrossEntropyLoss(Reduction reduction = Reduction::Mean);

    auto forward(const Variable& input, const Tensor& target) -> Variable;
    auto operator()(const Variable& input, const Tensor& target) -> Variable {
        return forward(input, target);
    }

private:
    Reduction reduction_;
};

// Binary Cross Entropy Loss
class BCELoss {
public:
    explicit BCELoss(Reduction reduction = Reduction::Mean);

    auto forward(const Variable& input, const Variable& target) -> Variable;
    auto operator()(const Variable& input, const Variable& target) -> Variable {
        return forward(input, target);
    }

private:
    Reduction reduction_;
};

// Binary Cross Entropy with Logits Loss
class BCEWithLogitsLoss {
public:
    explicit BCEWithLogitsLoss(Reduction reduction = Reduction::Mean);

    auto forward(const Variable& input, const Variable& target) -> Variable;
    auto operator()(const Variable& input, const Variable& target) -> Variable {
        return forward(input, target);
    }

private:
    Reduction reduction_;
};

// Negative Log Likelihood Loss
class NLLLoss {
public:
    explicit NLLLoss(Reduction reduction = Reduction::Mean);

    auto forward(const Variable& input, const Tensor& target) -> Variable;
    auto operator()(const Variable& input, const Tensor& target) -> Variable {
        return forward(input, target);
    }

private:
    Reduction reduction_;
};

// L1 Loss (Mean Absolute Error)
class L1Loss {
public:
    explicit L1Loss(Reduction reduction = Reduction::Mean);

    auto forward(const Variable& input, const Variable& target) -> Variable;
    auto operator()(const Variable& input, const Variable& target) -> Variable {
        return forward(input, target);
    }

private:
    Reduction reduction_;
};

// Smooth L1 Loss (Huber Loss)
class SmoothL1Loss {
public:
    explicit SmoothL1Loss(Reduction reduction = Reduction::Mean, double beta = 1.0);

    auto forward(const Variable& input, const Variable& target) -> Variable;
    auto operator()(const Variable& input, const Variable& target) -> Variable {
        return forward(input, target);
    }

private:
    Reduction reduction_;
    double beta_;
};

// Functional loss functions
auto mse_loss(const Variable& input, const Variable& target,
             Reduction reduction = Reduction::Mean) -> Variable;

auto cross_entropy(const Variable& input, const Tensor& target,
                  Reduction reduction = Reduction::Mean) -> Variable;

auto bce_loss(const Variable& input, const Variable& target,
             Reduction reduction = Reduction::Mean) -> Variable;

auto nll_loss(const Variable& input, const Tensor& target,
             Reduction reduction = Reduction::Mean) -> Variable;

} // namespace nn
} // namespace tenzor
