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
         double weight_decay = 0.01,
         double min_norm = 0.0,
         double max_norm = 10.0);

    explicit LAMB(std::vector<optim::ParamGroup> groups,
                  double default_lr = 1e-3,
                  double default_beta1 = 0.9,
                  double default_beta2 = 0.999,
                  double default_eps = 1e-6,
                  double default_weight_decay = 0.01,
                  double default_min_norm = 0.0,
                  double default_max_norm = 10.0);

    auto step_impl() -> void override;
    auto set_lr(double lr) -> void override;
    auto get_lr() const -> double override;
    auto state_dict() const -> std::unordered_map<std::string, Tensor> override;
    auto load_state_dict(const std::unordered_map<std::string, Tensor>& state) -> void override;

    // QQ.12: trust-ratio clamp setters for runtime tuning.  Defaults map
    // to PyTorch NVlamb's [0, 10] range.
    auto set_min_norm(double v) -> void { min_norm_ = v; }
    auto set_max_norm(double v) -> void { max_norm_ = v; }
    auto get_min_norm() const -> double { return min_norm_; }
    auto get_max_norm() const -> double { return max_norm_; }

protected:
    // Audit K.1: extend exp_avg_ / exp_avg_sq_ when add_param_group
    // appends new parameters mid-training.
    auto on_parameters_appended_(size_t old_count, size_t new_count) -> void override;

private:
    double lr_;
    double beta1_;
    double beta2_;
    double eps_;
    double weight_decay_;
    // QQ.12: trust-ratio clamp range.  Without these, a near-zero
    // update_norm makes trust_ratio explode and the parameter step
    // diverges.  Defaults match PyTorch NVlamb.
    double min_norm_{0.0};
    double max_norm_{10.0};

    int64_t step_count_{0};
    std::vector<Tensor> exp_avg_;
    std::vector<Tensor> exp_avg_sq_;

    auto initialize_buffers() -> void;
};

} // namespace optim
} // namespace tenzor
