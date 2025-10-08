#include "tenzor/nn/layers/conv.hpp"
#include "tenzor/ops/creation.hpp"

namespace tenzor::nn {

Conv2d::Conv2d(int64_t in_channels, int64_t out_channels, int64_t kernel_size,
              int64_t stride, int64_t padding, int64_t dilation,
              int64_t groups, bool bias)
    : in_channels_(in_channels), out_channels_(out_channels),
      kernel_size_(kernel_size), stride_(stride),
      padding_(padding), dilation_(dilation), groups_(groups) {

    // Initialize weight
    weight_ = Variable(
        randn({out_channels, in_channels / groups, kernel_size, kernel_size}),
        true
    );
    register_parameter("weight", weight_);

    // Initialize bias
    if (bias) {
        bias_ = Variable(zeros({out_channels}), true);
        register_parameter("bias", *bias_);
    }

    reset_parameters();
}

auto Conv2d::forward(const Variable& input) -> Variable {
    // TODO: Implement convolution operation
    return input;
}

auto Conv2d::reset_parameters() -> void {
    // TODO: Implement proper initialization
}

} // namespace tenzor::nn
