/**
 * @file lamb.hpp
 * @brief LAMB (Layer-wise Adaptive Moments) optimizer for large-batch training
 */

#pragma once

#include "optimizer.hpp"

namespace tenzor {
namespace optim {

/**
 * @brief LAMB optimizer for large-batch training
 *
 * Extends Adam with per-layer trust ratio scaling:
 *   trust_ratio = ||param|| / ||adam_update||
 *   param -= lr * trust_ratio * adam_update
 *
 * Critical for training BERT, ViT, and other models with large batch sizes.
 * Weight decay is applied in decoupled fashion (like AdamW).
 */
class LAMB : public Optimizer {
public:
    LAMB(std::vector<std::shared_ptr<Variable>> params,
         double lr = 1e-3,
         double beta1 = 0.9,
         double beta2 = 0.999,
         double eps = 1e-6,
         double weight_decay = 0.01);

    auto step_impl() -> void override;
    auto set_lr(double lr) -> void override;
    auto get_lr() const -> double override;
    auto state_dict() const -> std::unordered_map<std::string, Tensor> override;
    auto load_state_dict(const std::unordered_map<std::string, Tensor>& state) -> void override;

private:
    double lr_;
    double beta1_;
    double beta2_;
    double eps_;
    double weight_decay_;

    int64_t step_count_{0};
    std::vector<Tensor> exp_avg_;
    std::vector<Tensor> exp_avg_sq_;

    auto initialize_buffers() -> void;
};

} // namespace optim
} // namespace tenzor
