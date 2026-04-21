#include "tenzor/nn/layers/lazy_conv.hpp"
#include <stdexcept>

namespace tenzor::nn {

// =============================================================================
// LazyConv1d
// =============================================================================

LazyConv1d::LazyConv1d(int64_t out_channels, int64_t kernel_size,
                       int64_t stride, int64_t padding,
                       int64_t dilation, int64_t groups, bool bias)
    : out_channels_(out_channels), kernel_size_(kernel_size),
      stride_(stride), padding_(padding),
      dilation_(dilation), groups_(groups), has_bias_(bias) {

    if (out_channels <= 0) {
        throw std::runtime_error("LazyConv1d: out_channels must be positive, got " +
            std::to_string(out_channels));
    }
}

auto LazyConv1d::materialize(int64_t in_channels, Device device, DType dtype) -> void {
    if (in_channels <= 0) {
        throw std::runtime_error("LazyConv1d: inferred in_channels must be positive, got " +
            std::to_string(in_channels));
    }

    in_channels_ = in_channels;
    conv_ = std::make_shared<Conv1d>(in_channels, out_channels_, kernel_size_,
                                      stride_, padding_, dilation_, groups_, has_bias_);
    register_module("conv", conv_);

    // Conv1d is constructed with Float32 Float32-on-CPU weights by default;
    // match the first input's device *and* dtype so the first forward pass
    // does not throw a dtype-mismatch when operating on e.g. Float64 inputs.
    // convert_model() runs before the lazy module has any children to walk,
    // so this is the only opportunity to sync parameter dtype to the input.
    if (dtype != DType::Float32) {
        conv_->to(dtype);
    }
    if (device.type != Device::Type::CPU) {
        conv_->to(device);
    }
}

auto LazyConv1d::forward_impl(const Variable& input) -> Variable {
    auto input_shape = input.shape();

    if (input_shape.size() < 3) {
        throw std::runtime_error("LazyConv1d: input must have at least 3 dimensions "
            "(N, C_in, L), got " + std::to_string(input_shape.size()));
    }

    if (!conv_) {
        auto in_channels = input_shape[1];
        if (in_channels <= 0) {
            throw std::runtime_error("LazyConv1d: input channel dimension must be positive, got " +
                std::to_string(in_channels));
        }
        materialize(in_channels, input.tensor().device(), input.tensor().dtype());
    } else if (input_shape[1] != in_channels_) {
        throw std::runtime_error(
            "LazyConv1d: input channels (" + std::to_string(input_shape[1]) +
            ") don't match materialized in_channels (" + std::to_string(in_channels_) + ")");
    }

    return conv_->forward(input);
}

auto LazyConv1d::parameters() -> std::vector<std::shared_ptr<Variable>> {
    if (!conv_) {
        return {};
    }
    return Module::parameters();
}

auto LazyConv1d::own_parameters() -> std::vector<std::shared_ptr<Variable>> {
    if (!conv_) {
        return {};
    }
    return Module::own_parameters();
}

auto LazyConv1d::named_parameters() -> std::vector<std::pair<std::string, std::shared_ptr<Variable>>> {
    if (!conv_) {
        return {};
    }
    return Module::named_parameters();
}

// =============================================================================
// LazyConv2d
// =============================================================================

LazyConv2d::LazyConv2d(int64_t out_channels, int64_t kernel_size,
                       int64_t stride, int64_t padding,
                       int64_t dilation, int64_t groups, bool bias)
    : out_channels_(out_channels), kernel_size_(kernel_size),
      stride_(stride), padding_(padding),
      dilation_(dilation), groups_(groups), has_bias_(bias) {

    if (out_channels <= 0) {
        throw std::runtime_error("LazyConv2d: out_channels must be positive, got " +
            std::to_string(out_channels));
    }
}

auto LazyConv2d::materialize(int64_t in_channels, Device device, DType dtype) -> void {
    if (in_channels <= 0) {
        throw std::runtime_error("LazyConv2d: inferred in_channels must be positive, got " +
            std::to_string(in_channels));
    }

    in_channels_ = in_channels;
    conv_ = std::make_shared<Conv2d>(in_channels, out_channels_, kernel_size_,
                                      stride_, padding_, dilation_, groups_, has_bias_);
    register_module("conv", conv_);

    if (dtype != DType::Float32) {
        conv_->to(dtype);
    }
    if (device.type != Device::Type::CPU) {
        conv_->to(device);
    }
}

auto LazyConv2d::forward_impl(const Variable& input) -> Variable {
    auto input_shape = input.shape();

    if (input_shape.size() < 4) {
        throw std::runtime_error("LazyConv2d: input must have at least 4 dimensions "
            "(N, C_in, H, W), got " + std::to_string(input_shape.size()));
    }

    if (!conv_) {
        auto in_channels = input_shape[1];
        if (in_channels <= 0) {
            throw std::runtime_error("LazyConv2d: input channel dimension must be positive, got " +
                std::to_string(in_channels));
        }
        materialize(in_channels, input.tensor().device(), input.tensor().dtype());
    } else if (input_shape[1] != in_channels_) {
        throw std::runtime_error(
            "LazyConv2d: input channels (" + std::to_string(input_shape[1]) +
            ") don't match materialized in_channels (" + std::to_string(in_channels_) + ")");
    }

    return conv_->forward(input);
}

auto LazyConv2d::parameters() -> std::vector<std::shared_ptr<Variable>> {
    if (!conv_) {
        return {};
    }
    return Module::parameters();
}

auto LazyConv2d::own_parameters() -> std::vector<std::shared_ptr<Variable>> {
    if (!conv_) {
        return {};
    }
    return Module::own_parameters();
}

auto LazyConv2d::named_parameters() -> std::vector<std::pair<std::string, std::shared_ptr<Variable>>> {
    if (!conv_) {
        return {};
    }
    return Module::named_parameters();
}

// =============================================================================
// LazyConv3d
// =============================================================================

LazyConv3d::LazyConv3d(int64_t out_channels, int64_t kernel_size,
                       int64_t stride, int64_t padding,
                       int64_t dilation, int64_t groups, bool bias)
    : out_channels_(out_channels), kernel_size_(kernel_size),
      stride_(stride), padding_(padding),
      dilation_(dilation), groups_(groups), has_bias_(bias) {

    if (out_channels <= 0) {
        throw std::runtime_error("LazyConv3d: out_channels must be positive, got " +
            std::to_string(out_channels));
    }
}

auto LazyConv3d::materialize(int64_t in_channels, Device device, DType dtype) -> void {
    if (in_channels <= 0) {
        throw std::runtime_error("LazyConv3d: inferred in_channels must be positive, got " +
            std::to_string(in_channels));
    }

    in_channels_ = in_channels;
    conv_ = std::make_shared<Conv3d>(in_channels, out_channels_, kernel_size_,
                                      stride_, padding_, dilation_, groups_, has_bias_);
    register_module("conv", conv_);

    if (dtype != DType::Float32) {
        conv_->to(dtype);
    }
    if (device.type != Device::Type::CPU) {
        conv_->to(device);
    }
}

auto LazyConv3d::forward_impl(const Variable& input) -> Variable {
    auto input_shape = input.shape();

    if (input_shape.size() < 5) {
        throw std::runtime_error("LazyConv3d: input must have at least 5 dimensions "
            "(N, C_in, D, H, W), got " + std::to_string(input_shape.size()));
    }

    if (!conv_) {
        auto in_channels = input_shape[1];
        if (in_channels <= 0) {
            throw std::runtime_error("LazyConv3d: input channel dimension must be positive, got " +
                std::to_string(in_channels));
        }
        materialize(in_channels, input.tensor().device(), input.tensor().dtype());
    } else if (input_shape[1] != in_channels_) {
        throw std::runtime_error(
            "LazyConv3d: input channels (" + std::to_string(input_shape[1]) +
            ") don't match materialized in_channels (" + std::to_string(in_channels_) + ")");
    }

    return conv_->forward(input);
}

auto LazyConv3d::parameters() -> std::vector<std::shared_ptr<Variable>> {
    if (!conv_) {
        return {};
    }
    return Module::parameters();
}

auto LazyConv3d::own_parameters() -> std::vector<std::shared_ptr<Variable>> {
    if (!conv_) {
        return {};
    }
    return Module::own_parameters();
}

auto LazyConv3d::named_parameters() -> std::vector<std::pair<std::string, std::shared_ptr<Variable>>> {
    if (!conv_) {
        return {};
    }
    return Module::named_parameters();
}

} // namespace tenzor::nn
