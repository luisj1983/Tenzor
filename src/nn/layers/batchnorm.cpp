#include "tenzor/nn/layers/batchnorm.hpp"
#include "tenzor/ops/creation.hpp"

namespace tenzor::nn {

BatchNorm2d::BatchNorm2d(int64_t num_features, double eps, double momentum,
                        bool affine, bool track_running_stats)
    : num_features_(num_features), eps_(eps), momentum_(momentum),
      affine_(affine), track_running_stats_(track_running_stats) {

    if (affine) {
        weight_ = Variable(ones({num_features}), true);
        bias_ = Variable(zeros({num_features}), true);
        register_parameter("weight", weight_);
        register_parameter("bias", bias_);
    } else {
        weight_ = Variable(ones({num_features}), false);
        bias_ = Variable(zeros({num_features}), false);
    }

    if (track_running_stats) {
        running_mean_ = Variable(zeros({num_features}), false);
        running_var_ = Variable(ones({num_features}), false);
        register_buffer("running_mean", running_mean_);
        register_buffer("running_var", running_var_);
    }

    reset_parameters();
}

auto BatchNorm2d::forward(const Variable& input) -> Variable {
    // TODO: Implement batch normalization
    return input;
}

auto BatchNorm2d::reset_parameters() -> void {
    // Already initialized
}

} // namespace tenzor::nn
