#include "tenzor/nn/layers/padding.hpp"
#include "tenzor/autograd/function.hpp"
#include "tenzor/autograd/ops.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/transform.hpp"
#include "tenzor/ops/indexing.hpp"
#include <stdexcept>

namespace tenzor::nn {

// ============================================================================
// Pad backward: gradient of constant/zero padding is just slicing out the
// original region. Reflection/replication backward accumulate gradients
// from the padded positions back to their source positions.
// For simplicity and correctness, all padding backward uses the same
// approach: slice out the interior gradient (for constant/zero), or
// accumulate reflected/replicated gradients.
// ============================================================================

namespace {

/// Helper: pad a single dimension using constant fill.
/// Operates on Variables for autograd support.
auto pad_dim_constant(const Variable& input, int64_t dim, int64_t pad_before,
                      int64_t pad_after, double value) -> Variable {
    if (pad_before == 0 && pad_after == 0) return input;

    auto shape = input.tensor().shape();
    auto dtype = input.tensor().dtype();
    auto device = input.tensor().device();

    std::vector<Variable> parts;

    if (pad_before > 0) {
        auto pad_shape = std::vector<int64_t>(shape.begin(), shape.end());
        pad_shape[dim] = pad_before;
        auto pad_tensor = full(std::move(pad_shape), static_cast<float>(value), dtype, device);
        parts.push_back(Variable(pad_tensor, false));
    }

    parts.push_back(input);

    if (pad_after > 0) {
        auto pad_shape = std::vector<int64_t>(shape.begin(), shape.end());
        pad_shape[dim] = pad_after;
        auto pad_tensor = full(std::move(pad_shape), static_cast<float>(value), dtype, device);
        parts.push_back(Variable(pad_tensor, false));
    }

    return cat(parts, dim);
}

/// Helper: pad a single dimension using reflection.
auto pad_dim_reflect(const Variable& input, int64_t dim, int64_t pad_before,
                     int64_t pad_after) -> Variable {
    if (pad_before == 0 && pad_after == 0) return input;

    auto shape = input.tensor().shape();
    int64_t dim_size = shape[dim];

    if (pad_before >= dim_size || pad_after >= dim_size) {
        throw std::invalid_argument(
            "Reflection padding size must be less than the corresponding "
            "input dimension, but got padding (" + std::to_string(pad_before) +
            ", " + std::to_string(pad_after) + ") for dimension of size " +
            std::to_string(dim_size));
    }

    std::vector<Variable> parts;

    if (pad_before > 0) {
        // Reflect: take elements [1, pad_before] and flip them
        auto reflected = slice(input, dim, 1, pad_before + 1);
        reflected = flip(reflected, {dim});
        parts.push_back(reflected);
    }

    parts.push_back(input);

    if (pad_after > 0) {
        // Reflect: take elements [dim_size - pad_after - 1, dim_size - 1) and flip
        auto reflected = slice(input, dim, dim_size - pad_after - 1, dim_size - 1);
        reflected = flip(reflected, {dim});
        parts.push_back(reflected);
    }

    return cat(parts, dim);
}

/// Helper: pad a single dimension using replication (edge values).
auto pad_dim_replicate(const Variable& input, int64_t dim, int64_t pad_before,
                       int64_t pad_after) -> Variable {
    if (pad_before == 0 && pad_after == 0) return input;

    auto shape = input.tensor().shape();

    std::vector<Variable> parts;

    if (pad_before > 0) {
        // Replicate: repeat the first element pad_before times
        auto edge = slice(input, dim, 0, 1);
        std::vector<int64_t> rep_shape(shape.size(), 1);
        rep_shape[dim] = pad_before;
        // expand broadcasts to the desired shape
        auto edge_shape = std::vector<int64_t>(shape.begin(), shape.end());
        edge_shape[dim] = pad_before;
        auto expanded = Variable(expand(edge.tensor(), edge_shape), input.requires_grad());
        // Need contiguous copy for cat
        auto contiguous_expanded = Variable(contiguous(expanded.tensor()), input.requires_grad());
        parts.push_back(contiguous_expanded);
    }

    parts.push_back(input);

    if (pad_after > 0) {
        int64_t dim_size = shape[dim];
        auto edge = slice(input, dim, dim_size - 1, dim_size);
        auto edge_shape = std::vector<int64_t>(shape.begin(), shape.end());
        edge_shape[dim] = pad_after;
        auto expanded = Variable(expand(edge.tensor(), edge_shape), input.requires_grad());
        auto contiguous_expanded = Variable(contiguous(expanded.tensor()), input.requires_grad());
        parts.push_back(contiguous_expanded);
    }

    return cat(parts, dim);
}

} // anonymous namespace

// ============================================================================
// ConstantPad1d
// ============================================================================

ConstantPad1d::ConstantPad1d(int64_t padding_left, int64_t padding_right, double value)
    : padding_left_(padding_left), padding_right_(padding_right), value_(value) {}

ConstantPad1d::ConstantPad1d(int64_t padding, double value)
    : padding_left_(padding), padding_right_(padding), value_(value) {}

auto ConstantPad1d::forward_impl(const Variable& input) -> Variable {
    if (input.tensor().shape().size() < 2) {
        throw std::invalid_argument("ConstantPad1d: input must have at least 2 dimensions");
    }
    int64_t last_dim = static_cast<int64_t>(input.tensor().shape().size()) - 1;
    return pad_dim_constant(input, last_dim, padding_left_, padding_right_, value_);
}

// ============================================================================
// ConstantPad2d
// ============================================================================

ConstantPad2d::ConstantPad2d(int64_t padding_left, int64_t padding_right,
                             int64_t padding_top, int64_t padding_bottom, double value)
    : padding_left_(padding_left), padding_right_(padding_right),
      padding_top_(padding_top), padding_bottom_(padding_bottom), value_(value) {}

ConstantPad2d::ConstantPad2d(int64_t padding, double value)
    : padding_left_(padding), padding_right_(padding),
      padding_top_(padding), padding_bottom_(padding), value_(value) {}

auto ConstantPad2d::forward_impl(const Variable& input) -> Variable {
    if (input.tensor().shape().size() < 3) {
        throw std::invalid_argument("ConstantPad2d: input must have at least 3 dimensions");
    }
    auto ndim = static_cast<int64_t>(input.tensor().shape().size());
    // Pad W dimension (last), then H dimension (second-to-last)
    auto result = pad_dim_constant(input, ndim - 1, padding_left_, padding_right_, value_);
    result = pad_dim_constant(result, ndim - 2, padding_top_, padding_bottom_, value_);
    return result;
}

// ============================================================================
// ConstantPad3d
// ============================================================================

ConstantPad3d::ConstantPad3d(std::vector<int64_t> padding, double value)
    : padding_(std::move(padding)), value_(value) {
    if (padding_.size() != 6) {
        throw std::invalid_argument(
            "ConstantPad3d: padding must have 6 elements "
            "(left, right, top, bottom, front, back), got " +
            std::to_string(padding_.size()));
    }
}

ConstantPad3d::ConstantPad3d(int64_t padding, double value)
    : padding_({padding, padding, padding, padding, padding, padding}), value_(value) {}

auto ConstantPad3d::forward_impl(const Variable& input) -> Variable {
    if (input.tensor().shape().size() < 4) {
        throw std::invalid_argument("ConstantPad3d: input must have at least 4 dimensions");
    }
    auto ndim = static_cast<int64_t>(input.tensor().shape().size());
    // Pad W (last), then H, then D
    auto result = pad_dim_constant(input, ndim - 1, padding_[0], padding_[1], value_);
    result = pad_dim_constant(result, ndim - 2, padding_[2], padding_[3], value_);
    result = pad_dim_constant(result, ndim - 3, padding_[4], padding_[5], value_);
    return result;
}

// ============================================================================
// ReflectionPad1d
// ============================================================================

ReflectionPad1d::ReflectionPad1d(int64_t padding_left, int64_t padding_right)
    : padding_left_(padding_left), padding_right_(padding_right) {}

ReflectionPad1d::ReflectionPad1d(int64_t padding)
    : padding_left_(padding), padding_right_(padding) {}

auto ReflectionPad1d::forward_impl(const Variable& input) -> Variable {
    if (input.tensor().shape().size() < 2) {
        throw std::invalid_argument("ReflectionPad1d: input must have at least 2 dimensions");
    }
    int64_t last_dim = static_cast<int64_t>(input.tensor().shape().size()) - 1;
    return pad_dim_reflect(input, last_dim, padding_left_, padding_right_);
}

// ============================================================================
// ReflectionPad2d
// ============================================================================

ReflectionPad2d::ReflectionPad2d(int64_t padding_left, int64_t padding_right,
                                 int64_t padding_top, int64_t padding_bottom)
    : padding_left_(padding_left), padding_right_(padding_right),
      padding_top_(padding_top), padding_bottom_(padding_bottom) {}

ReflectionPad2d::ReflectionPad2d(int64_t padding)
    : padding_left_(padding), padding_right_(padding),
      padding_top_(padding), padding_bottom_(padding) {}

auto ReflectionPad2d::forward_impl(const Variable& input) -> Variable {
    if (input.tensor().shape().size() < 3) {
        throw std::invalid_argument("ReflectionPad2d: input must have at least 3 dimensions");
    }
    auto ndim = static_cast<int64_t>(input.tensor().shape().size());
    auto result = pad_dim_reflect(input, ndim - 1, padding_left_, padding_right_);
    result = pad_dim_reflect(result, ndim - 2, padding_top_, padding_bottom_);
    return result;
}

// ============================================================================
// ReplicationPad1d
// ============================================================================

ReplicationPad1d::ReplicationPad1d(int64_t padding_left, int64_t padding_right)
    : padding_left_(padding_left), padding_right_(padding_right) {}

ReplicationPad1d::ReplicationPad1d(int64_t padding)
    : padding_left_(padding), padding_right_(padding) {}

auto ReplicationPad1d::forward_impl(const Variable& input) -> Variable {
    if (input.tensor().shape().size() < 2) {
        throw std::invalid_argument("ReplicationPad1d: input must have at least 2 dimensions");
    }
    int64_t last_dim = static_cast<int64_t>(input.tensor().shape().size()) - 1;
    return pad_dim_replicate(input, last_dim, padding_left_, padding_right_);
}

// ============================================================================
// ReplicationPad2d
// ============================================================================

ReplicationPad2d::ReplicationPad2d(int64_t padding_left, int64_t padding_right,
                                   int64_t padding_top, int64_t padding_bottom)
    : padding_left_(padding_left), padding_right_(padding_right),
      padding_top_(padding_top), padding_bottom_(padding_bottom) {}

ReplicationPad2d::ReplicationPad2d(int64_t padding)
    : padding_left_(padding), padding_right_(padding),
      padding_top_(padding), padding_bottom_(padding) {}

auto ReplicationPad2d::forward_impl(const Variable& input) -> Variable {
    if (input.tensor().shape().size() < 3) {
        throw std::invalid_argument("ReplicationPad2d: input must have at least 3 dimensions");
    }
    auto ndim = static_cast<int64_t>(input.tensor().shape().size());
    auto result = pad_dim_replicate(input, ndim - 1, padding_left_, padding_right_);
    result = pad_dim_replicate(result, ndim - 2, padding_top_, padding_bottom_);
    return result;
}

// ============================================================================
// ReplicationPad3d
// ============================================================================

ReplicationPad3d::ReplicationPad3d(std::vector<int64_t> padding)
    : padding_(std::move(padding)) {
    if (padding_.size() != 6) {
        throw std::invalid_argument(
            "ReplicationPad3d: padding must have 6 elements "
            "(left, right, top, bottom, front, back), got " +
            std::to_string(padding_.size()));
    }
}

ReplicationPad3d::ReplicationPad3d(int64_t padding)
    : padding_({padding, padding, padding, padding, padding, padding}) {}

auto ReplicationPad3d::forward_impl(const Variable& input) -> Variable {
    if (input.tensor().shape().size() < 4) {
        throw std::invalid_argument("ReplicationPad3d: input must have at least 4 dimensions");
    }
    auto ndim = static_cast<int64_t>(input.tensor().shape().size());
    auto result = pad_dim_replicate(input, ndim - 1, padding_[0], padding_[1]);
    result = pad_dim_replicate(result, ndim - 2, padding_[2], padding_[3]);
    result = pad_dim_replicate(result, ndim - 3, padding_[4], padding_[5]);
    return result;
}

// ============================================================================
// ZeroPad2d
// ============================================================================

ZeroPad2d::ZeroPad2d(int64_t padding_left, int64_t padding_right,
                     int64_t padding_top, int64_t padding_bottom)
    : impl_(padding_left, padding_right, padding_top, padding_bottom, 0.0) {}

ZeroPad2d::ZeroPad2d(int64_t padding)
    : impl_(padding, 0.0) {}

auto ZeroPad2d::forward_impl(const Variable& input) -> Variable {
    return impl_.forward_impl(input);
}

} // namespace tenzor::nn
