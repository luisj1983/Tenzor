#pragma once

#include "optimizer.hpp"

namespace tenzor {
namespace optim {

// Adam optimizer
class Adam : public Optimizer {
public:
    Adam(std::vector<Variable*> params,
         double lr = 1e-3,
         double beta1 = 0.9,
         double beta2 = 0.999,
         double eps = 1e-8,
         double weight_decay = 0.0,
         bool amsgrad = false);

    auto step() -> void override;

    // Learning rate management
    auto set_lr(double lr) -> void;
    auto get_lr() const -> double;

private:
    double lr_;
    double beta1_;
    double beta2_;
    double eps_;
    double weight_decay_;
    bool amsgrad_;

    int64_t step_count_{0};
    std::vector<Tensor> exp_avg_;       // First moment estimates
    std::vector<Tensor> exp_avg_sq_;    // Second moment estimates
    std::vector<Tensor> max_exp_avg_sq_; // Max of second moment (for AMSGrad)

    auto initialize_buffers() -> void;
};

// AdamW optimizer (Adam with decoupled weight decay)
class AdamW : public Optimizer {
public:
    AdamW(std::vector<Variable*> params,
          double lr = 1e-3,
          double beta1 = 0.9,
          double beta2 = 0.999,
          double eps = 1e-8,
          double weight_decay = 0.01,
          bool amsgrad = false);

    auto step() -> void override;

    auto set_lr(double lr) -> void;
    auto get_lr() const -> double;

private:
    double lr_;
    double beta1_;
    double beta2_;
    double eps_;
    double weight_decay_;
    bool amsgrad_;

    int64_t step_count_{0};
    std::vector<Tensor> exp_avg_;
    std::vector<Tensor> exp_avg_sq_;
    std::vector<Tensor> max_exp_avg_sq_;

    auto initialize_buffers() -> void;
};

} // namespace optim
} // namespace tenzor
