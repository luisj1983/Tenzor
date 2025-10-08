#include "tenzor/nn/layers/linear.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"

namespace tenzor::nn {

Linear::Linear(int64_t in_features, int64_t out_features, bool bias)
    : in_features_(in_features), out_features_(out_features) {

    // Initialize weight
    weight_ = Variable(randn({out_features, in_features}), true);
    register_parameter("weight", weight_);

    // Initialize bias
    if (bias) {
        bias_ = Variable(zeros({out_features}), true);
        register_parameter("bias", *bias_);
    }

    reset_parameters();
}

auto Linear::forward(const Variable& input) -> Variable {
    // output = input @ weight.T + bias
    auto output = Variable(matmul(input.tensor(), weight_.tensor().transpose(-2, -1)), true);

    if (bias_) {
        output = output + *bias_;
    }

    return output;
}

auto Linear::reset_parameters() -> void {
    // TODO: Implement proper Kaiming initialization
}

} // namespace tenzor::nn
