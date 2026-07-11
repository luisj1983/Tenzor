#include "tenzor/nn/layers/flatten.hpp"
#include "tenzor/autograd/function.hpp"
#include "tenzor/autograd/ops.hpp"
#include <stdexcept>

namespace tenzor::nn {

// Flatten backward function
class FlattenBackward : public Function {
public:
    FlattenBackward(std::vector<int64_t> input_shape)
        : input_shape_(std::move(input_shape)) {}

    auto forward([[maybe_unused]] std::vector<Variable> inputs) -> std::vector<Variable> override {
        throw std::runtime_error("FlattenBackward::forward should not be called");
    }

    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override {
        if (grad_outputs.size() != 1) {
            throw std::invalid_argument("FlattenBackward expects 1 gradient output");
        }

        // Reshape gradient back to original input shape
        auto grad_input = grad_outputs[0].reshape(input_shape_);

        std::vector<Tensor> result;
        result.push_back(grad_input);
        return result;
    }

    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override {
        return {reshape(grad_outputs[0], input_shape_)};
    }

private:
    std::vector<int64_t> input_shape_;
};

Flatten::Flatten(int64_t start_dim, int64_t end_dim)
    : start_dim_(start_dim), end_dim_(end_dim) {}

auto Flatten::forward_impl(const Variable& input) -> Variable {
    auto shape = input.tensor().shape();
    auto ndim = static_cast<int64_t>(shape.size());

    // Normalize negative dimensions
    int64_t start = start_dim_ < 0 ? ndim + start_dim_ : start_dim_;
    int64_t end = end_dim_ < 0 ? ndim + end_dim_ : end_dim_;

    if (start < 0 || start >= ndim) {
        throw std::invalid_argument("start_dim out of range");
    }
    if (end < 0 || end >= ndim) {
        throw std::invalid_argument("end_dim out of range");
    }
    if (start > end) {
        throw std::invalid_argument("start_dim must be <= end_dim");
    }

    // Compute new shape
    std::vector<int64_t> new_shape;

    // Keep dimensions before start_dim
    for (int64_t i = 0; i < start; ++i) {
        new_shape.push_back(shape[i]);
    }

    // Flatten dimensions from start_dim to end_dim (inclusive)
    int64_t flattened_size = 1;
    for (int64_t i = start; i <= end; ++i) {
        flattened_size *= shape[i];
    }
    new_shape.push_back(flattened_size);

    // Keep dimensions after end_dim
    for (int64_t i = end + 1; i < ndim; ++i) {
        new_shape.push_back(shape[i]);
    }

    // Reshape tensor
    auto output_tensor = input.tensor().reshape(new_shape);

    // Create output variable
    Variable output(output_tensor, input.requires_grad());

    // Set up autograd if input requires grad
    if (input.requires_grad()) {
        // Save original input shape for backward
        std::vector<int64_t> input_shape_vec(shape.begin(), shape.end());

        // Create backward function
        auto flatten_fn = std::make_shared<FlattenBackward>(input_shape_vec);

        // Track input variable for gradient accumulation
        std::vector<Variable> input_vars;
        input_vars.push_back(input);
        flatten_fn->set_input_variables(input_vars);

        // Set up backward graph - link to input's grad_fn if it exists
        std::vector<std::shared_ptr<Function>> next_funcs;
        if (input.grad_fn()) {
            next_funcs.push_back(input.grad_fn());
        }
        flatten_fn->set_next_functions(next_funcs);

        // Set gradient function on output
        output.set_grad_fn(flatten_fn);
    }

    return output;
}

// Unflatten backward function
class UnflattenBackward : public Function {
public:
    UnflattenBackward(std::vector<int64_t> input_shape)
        : input_shape_(std::move(input_shape)) {}

    auto forward([[maybe_unused]] std::vector<Variable> inputs) -> std::vector<Variable> override {
        throw std::runtime_error("UnflattenBackward::forward should not be called");
    }

    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override {
        // Reshape gradient back to the flattened shape
        return {grad_outputs[0].reshape(input_shape_)};
    }

    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override {
        return {reshape(grad_outputs[0], input_shape_)};
    }

private:
    std::vector<int64_t> input_shape_;
};

Unflatten::Unflatten(int64_t dim, std::vector<int64_t> sizes)
    : dim_(dim), sizes_(std::move(sizes)) {}

auto Unflatten::forward_impl(const Variable& input) -> Variable {
    auto shape = input.tensor().shape();
    auto ndim = static_cast<int64_t>(shape.size());

    int64_t dim = dim_ < 0 ? ndim + dim_ : dim_;
    if (dim < 0 || dim >= ndim) {
        throw std::invalid_argument("Unflatten: dim out of range");
    }

    // Verify the product of sizes matches the dimension being unflattened
    int64_t product = 1;
    for (auto s : sizes_) product *= s;
    if (product != shape[dim]) {
        throw std::invalid_argument("Unflatten: product of sizes (" +
            std::to_string(product) + ") must match dim size (" +
            std::to_string(shape[dim]) + ")");
    }

    // Build new shape: replace dim with sizes_
    std::vector<int64_t> new_shape;
    for (int64_t i = 0; i < dim; ++i) new_shape.push_back(shape[i]);
    for (auto s : sizes_) new_shape.push_back(s);
    for (int64_t i = dim + 1; i < ndim; ++i) new_shape.push_back(shape[i]);

    auto output_tensor = input.tensor().reshape(new_shape);
    Variable output(output_tensor, input.requires_grad());

    if (input.requires_grad()) {
        std::vector<int64_t> input_shape_vec(shape.begin(), shape.end());
        auto unflatten_fn = std::make_shared<UnflattenBackward>(input_shape_vec);
        std::vector<Variable> input_vars{input};
        unflatten_fn->set_input_variables(input_vars);
        std::vector<std::shared_ptr<Function>> next_funcs;
        if (input.grad_fn()) next_funcs.push_back(input.grad_fn());
        unflatten_fn->set_next_functions(next_funcs);
        output.set_grad_fn(unflatten_fn);
    }

    return output;
}

// ============================================================================
// PixelShuffle implementation
// ============================================================================

class PixelShuffleBackward : public Function {
public:
    PixelShuffleBackward(int64_t downscale_factor, std::vector<int64_t> input_shape)
        : downscale_factor_(downscale_factor), input_shape_(std::move(input_shape)) {}

    auto forward(std::vector<Variable> /*inputs*/) -> std::vector<Variable> override {
        throw std::runtime_error("PixelShuffleBackward::forward should not be called");
    }

    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override {
        // Inverse of pixel_shuffle is pixel_unshuffle (reshape + permute)
        auto grad = grad_outputs[0];
        auto shape = grad.shape();
        auto ndim = static_cast<int64_t>(shape.size());
        if (ndim < 3) throw std::runtime_error("PixelShuffleBackward: need at least 3D");

        int64_t r = downscale_factor_;
        int64_t C = shape[ndim - 3];
        int64_t H_out = shape[ndim - 2];
        int64_t W_out = shape[ndim - 1];

        // Reverse: reshape (*, C, H*r, W*r) -> (*, C, H, r, W, r) -> permute -> (*, C*r*r, H, W)
        std::vector<int64_t> new_shape;
        for (int64_t i = 0; i < ndim - 3; ++i) new_shape.push_back(shape[i]);
        new_shape.push_back(C);
        new_shape.push_back(H_out / r);
        new_shape.push_back(r);
        new_shape.push_back(W_out / r);
        new_shape.push_back(r);

        auto reshaped = grad.reshape(new_shape);
        auto rndim = static_cast<int64_t>(reshaped.shape().size());

        // permute: move r dims next to C: (*, C, H, r, W, r) -> (*, C, r, r, H, W)
        std::vector<int64_t> perm;
        for (int64_t i = 0; i < rndim - 5; ++i) perm.push_back(i);
        perm.push_back(rndim - 5);  // C
        perm.push_back(rndim - 3);  // r (from H)
        perm.push_back(rndim - 1);  // r (from W)
        perm.push_back(rndim - 4);  // H
        perm.push_back(rndim - 2);  // W

        auto permuted = reshaped.permute(perm).contiguous();
        auto result = permuted.reshape(input_shape_);
        return {result};
    }

    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override {
        auto grad = grad_outputs[0];
        auto shape = grad.tensor().shape();
        auto ndim = static_cast<int64_t>(shape.size());
        if (ndim < 3) throw std::runtime_error("PixelShuffleBackward: need at least 3D");

        int64_t r = downscale_factor_;
        int64_t C = shape[ndim - 3];
        int64_t H_out = shape[ndim - 2];
        int64_t W_out = shape[ndim - 1];

        // Reverse: reshape (*, C, H*r, W*r) -> (*, C, H, r, W, r) -> permute -> (*, C*r*r, H, W)
        std::vector<int64_t> new_shape;
        for (int64_t i = 0; i < ndim - 3; ++i) new_shape.push_back(shape[i]);
        new_shape.push_back(C);
        new_shape.push_back(H_out / r);
        new_shape.push_back(r);
        new_shape.push_back(W_out / r);
        new_shape.push_back(r);

        auto reshaped = reshape(grad, new_shape);
        auto rndim = static_cast<int64_t>(reshaped.tensor().shape().size());

        // permute: move r dims next to C: (*, C, H, r, W, r) -> (*, C, r, r, H, W)
        std::vector<int64_t> perm;
        for (int64_t i = 0; i < rndim - 5; ++i) perm.push_back(i);
        perm.push_back(rndim - 5);  // C
        perm.push_back(rndim - 3);  // r (from H)
        perm.push_back(rndim - 1);  // r (from W)
        perm.push_back(rndim - 4);  // H
        perm.push_back(rndim - 2);  // W

        auto permuted = permute(reshaped, perm);
        return {reshape(permuted, input_shape_)};
    }

private:
    int64_t downscale_factor_;
    std::vector<int64_t> input_shape_;
};

PixelShuffle::PixelShuffle(int64_t upscale_factor)
    : upscale_factor_(upscale_factor) {
    if (upscale_factor < 1) {
        throw std::invalid_argument("PixelShuffle: upscale_factor must be >= 1");
    }
}

auto PixelShuffle::forward_impl(const Variable& input) -> Variable {
    auto shape = input.tensor().shape();
    auto ndim = static_cast<int64_t>(shape.size());
    if (ndim < 3) {
        throw std::invalid_argument("PixelShuffle: input must have at least 3 dimensions");
    }

    int64_t r = upscale_factor_;
    int64_t C_in = shape[ndim - 3];
    int64_t H = shape[ndim - 2];
    int64_t W = shape[ndim - 1];

    if (C_in % (r * r) != 0) {
        throw std::invalid_argument("PixelShuffle: channels (" + std::to_string(C_in) +
            ") must be divisible by upscale_factor^2 (" + std::to_string(r * r) + ")");
    }

    int64_t C_out = C_in / (r * r);

    // Reshape: (*, C*r*r, H, W) -> (*, C, r, r, H, W)
    std::vector<int64_t> reshaped_shape;
    for (int64_t i = 0; i < ndim - 3; ++i) reshaped_shape.push_back(shape[i]);
    reshaped_shape.push_back(C_out);
    reshaped_shape.push_back(r);
    reshaped_shape.push_back(r);
    reshaped_shape.push_back(H);
    reshaped_shape.push_back(W);

    auto reshaped = input.tensor().reshape(reshaped_shape);
    auto rndim = static_cast<int64_t>(reshaped.shape().size());

    // Permute: (*, C, r, r, H, W) -> (*, C, H, r, W, r)
    std::vector<int64_t> perm;
    for (int64_t i = 0; i < rndim - 5; ++i) perm.push_back(i);
    perm.push_back(rndim - 5);  // C
    perm.push_back(rndim - 2);  // H
    perm.push_back(rndim - 4);  // r
    perm.push_back(rndim - 1);  // W
    perm.push_back(rndim - 3);  // r

    // tenzor::permute (Variable-level) records a real OpType::Permute node
    // with the JIT tracer when tracing is active. A raw Tensor::permute() is
    // pure metadata with zero dispatch() calls, so it was invisible to the
    // tracer; the subsequent .contiguous() (which DOES dispatch
    // OpId::Contiguous) then aliased its output to that untracked permuted
    // tensor, so a traced PixelShuffle call froze the whole
    // permute+contiguous+reshape tail at its trace-time value (JIT-R045).
    auto permuted = tenzor::permute(Variable(reshaped, false), perm).tensor().contiguous();

    // Reshape: (*, C, H, r, W, r) -> (*, C, H*r, W*r)
    std::vector<int64_t> output_shape;
    for (int64_t i = 0; i < ndim - 3; ++i) output_shape.push_back(shape[i]);
    output_shape.push_back(C_out);
    output_shape.push_back(H * r);
    output_shape.push_back(W * r);

    auto output_tensor = permuted.reshape(output_shape);
    Variable output(output_tensor, input.requires_grad());

    if (input.requires_grad()) {
        std::vector<int64_t> input_shape_vec(shape.begin(), shape.end());
        auto ps_fn = std::make_shared<PixelShuffleBackward>(r, input_shape_vec);
        std::vector<Variable> input_vars{input};
        ps_fn->set_input_variables(input_vars);
        std::vector<std::shared_ptr<Function>> next_funcs;
        if (input.grad_fn()) next_funcs.push_back(input.grad_fn());
        ps_fn->set_next_functions(next_funcs);
        output.set_grad_fn(ps_fn);
    }

    return output;
}

// ============================================================================
// PixelUnshuffle implementation
// ============================================================================

class PixelUnshuffleBackward : public Function {
public:
    PixelUnshuffleBackward(int64_t upscale_factor, std::vector<int64_t> input_shape)
        : upscale_factor_(upscale_factor), input_shape_(std::move(input_shape)) {}

    auto forward(std::vector<Variable> /*inputs*/) -> std::vector<Variable> override {
        throw std::runtime_error("PixelUnshuffleBackward::forward should not be called");
    }

    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override {
        // Inverse of pixel_unshuffle is pixel_shuffle
        auto grad = grad_outputs[0];
        auto shape = grad.shape();
        auto ndim = static_cast<int64_t>(shape.size());

        int64_t r = upscale_factor_;
        int64_t C_in = shape[ndim - 3];
        int64_t H = shape[ndim - 2];
        int64_t W = shape[ndim - 1];
        int64_t C_out = C_in / (r * r);

        // pixel_shuffle: reshape + permute + reshape
        std::vector<int64_t> reshaped_shape;
        for (int64_t i = 0; i < ndim - 3; ++i) reshaped_shape.push_back(shape[i]);
        reshaped_shape.push_back(C_out);
        reshaped_shape.push_back(r);
        reshaped_shape.push_back(r);
        reshaped_shape.push_back(H);
        reshaped_shape.push_back(W);

        auto reshaped = grad.reshape(reshaped_shape);
        auto rndim = static_cast<int64_t>(reshaped.shape().size());

        std::vector<int64_t> perm;
        for (int64_t i = 0; i < rndim - 5; ++i) perm.push_back(i);
        perm.push_back(rndim - 5);  // C
        perm.push_back(rndim - 2);  // H
        perm.push_back(rndim - 4);  // r
        perm.push_back(rndim - 1);  // W
        perm.push_back(rndim - 3);  // r

        auto permuted = reshaped.permute(perm).contiguous();
        auto result = permuted.reshape(input_shape_);
        return {result};
    }

    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override {
        auto grad = grad_outputs[0];
        auto shape = grad.tensor().shape();
        auto ndim = static_cast<int64_t>(shape.size());

        int64_t r = upscale_factor_;
        int64_t C_in = shape[ndim - 3];
        int64_t H = shape[ndim - 2];
        int64_t W = shape[ndim - 1];
        int64_t C_out = C_in / (r * r);

        // pixel_shuffle: reshape + permute + reshape
        std::vector<int64_t> reshaped_shape;
        for (int64_t i = 0; i < ndim - 3; ++i) reshaped_shape.push_back(shape[i]);
        reshaped_shape.push_back(C_out);
        reshaped_shape.push_back(r);
        reshaped_shape.push_back(r);
        reshaped_shape.push_back(H);
        reshaped_shape.push_back(W);

        auto reshaped = reshape(grad, reshaped_shape);
        auto rndim = static_cast<int64_t>(reshaped.tensor().shape().size());

        std::vector<int64_t> perm;
        for (int64_t i = 0; i < rndim - 5; ++i) perm.push_back(i);
        perm.push_back(rndim - 5);  // C
        perm.push_back(rndim - 2);  // H
        perm.push_back(rndim - 4);  // r
        perm.push_back(rndim - 1);  // W
        perm.push_back(rndim - 3);  // r

        auto permuted = permute(reshaped, perm);
        return {reshape(permuted, input_shape_)};
    }

private:
    int64_t upscale_factor_;
    std::vector<int64_t> input_shape_;
};

PixelUnshuffle::PixelUnshuffle(int64_t downscale_factor)
    : downscale_factor_(downscale_factor) {
    if (downscale_factor < 1) {
        throw std::invalid_argument("PixelUnshuffle: downscale_factor must be >= 1");
    }
}

auto PixelUnshuffle::forward_impl(const Variable& input) -> Variable {
    auto shape = input.tensor().shape();
    auto ndim = static_cast<int64_t>(shape.size());
    if (ndim < 3) {
        throw std::invalid_argument("PixelUnshuffle: input must have at least 3 dimensions");
    }

    int64_t r = downscale_factor_;
    int64_t C = shape[ndim - 3];
    int64_t H = shape[ndim - 2];
    int64_t W = shape[ndim - 1];

    if (H % r != 0 || W % r != 0) {
        throw std::invalid_argument("PixelUnshuffle: spatial dims must be divisible by downscale_factor");
    }

    // Reshape: (*, C, H, W) -> (*, C, H/r, r, W/r, r)
    std::vector<int64_t> reshaped_shape;
    for (int64_t i = 0; i < ndim - 3; ++i) reshaped_shape.push_back(shape[i]);
    reshaped_shape.push_back(C);
    reshaped_shape.push_back(H / r);
    reshaped_shape.push_back(r);
    reshaped_shape.push_back(W / r);
    reshaped_shape.push_back(r);

    auto reshaped = input.tensor().reshape(reshaped_shape);
    auto rndim = static_cast<int64_t>(reshaped.shape().size());

    // Permute: (*, C, H/r, r, W/r, r) -> (*, C, r, r, H/r, W/r)
    std::vector<int64_t> perm;
    for (int64_t i = 0; i < rndim - 5; ++i) perm.push_back(i);
    perm.push_back(rndim - 5);  // C
    perm.push_back(rndim - 3);  // r (from H)
    perm.push_back(rndim - 1);  // r (from W)
    perm.push_back(rndim - 4);  // H/r
    perm.push_back(rndim - 2);  // W/r

    // See PixelShuffle::forward_impl above for why tenzor::permute
    // (Variable-level, tracer-visible) replaces the raw Tensor::permute()
    // here (JIT-R045).
    auto permuted = tenzor::permute(Variable(reshaped, false), perm).tensor().contiguous();

    // Reshape: (*, C, r, r, H/r, W/r) -> (*, C*r*r, H/r, W/r)
    std::vector<int64_t> output_shape;
    for (int64_t i = 0; i < ndim - 3; ++i) output_shape.push_back(shape[i]);
    output_shape.push_back(C * r * r);
    output_shape.push_back(H / r);
    output_shape.push_back(W / r);

    auto output_tensor = permuted.reshape(output_shape);
    Variable output(output_tensor, input.requires_grad());

    if (input.requires_grad()) {
        std::vector<int64_t> input_shape_vec(shape.begin(), shape.end());
        auto pus_fn = std::make_shared<PixelUnshuffleBackward>(r, input_shape_vec);
        std::vector<Variable> input_vars{input};
        pus_fn->set_input_variables(input_vars);
        std::vector<std::shared_ptr<Function>> next_funcs;
        if (input.grad_fn()) next_funcs.push_back(input.grad_fn());
        pus_fn->set_next_functions(next_funcs);
        output.set_grad_fn(pus_fn);
    }

    return output;
}

// ============================================================================
// ChannelShuffle implementation
// ============================================================================

class ChannelShuffleBackward : public Function {
public:
    ChannelShuffleBackward(int64_t groups, std::vector<int64_t> input_shape)
        : groups_(groups), input_shape_(std::move(input_shape)) {}

    auto forward(std::vector<Variable> /*inputs*/) -> std::vector<Variable> override {
        throw std::runtime_error("ChannelShuffleBackward::forward should not be called");
    }

    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override {
        // Inverse of channel_shuffle(groups) is channel_shuffle(C/groups)
        // but it's simpler to just do the inverse permutation:
        // (B, C/G, G, H, W) -> permute(0,2,1,3,4) -> (B, G, C/G, H, W) -> reshape
        auto grad = grad_outputs[0];
        auto shape = grad.shape();
        auto ndim = static_cast<int64_t>(shape.size());
        if (ndim < 3) throw std::runtime_error("ChannelShuffleBackward: need at least 3D");

        int64_t channels = shape[ndim - 3];
        int64_t cpg = channels / groups_;

        // Reshape: (*, C, H, W) -> (*, C/G, G, H, W)
        std::vector<int64_t> reshaped;
        for (int64_t i = 0; i < ndim - 3; ++i) reshaped.push_back(shape[i]);
        reshaped.push_back(cpg);
        reshaped.push_back(groups_);
        for (int64_t i = ndim - 2; i < ndim; ++i) reshaped.push_back(shape[i]);

        auto r = grad.reshape(reshaped);
        auto rndim = static_cast<int64_t>(r.shape().size());

        // Permute: swap the group dims back
        std::vector<int64_t> perm;
        for (int64_t i = 0; i < rndim - 4; ++i) perm.push_back(i);
        perm.push_back(rndim - 3);  // G
        perm.push_back(rndim - 4);  // C/G
        perm.push_back(rndim - 2);  // H
        perm.push_back(rndim - 1);  // W

        auto permuted = r.permute(perm).contiguous();
        return {permuted.reshape(input_shape_)};
    }

    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override {
        auto grad = grad_outputs[0];
        auto shape = grad.tensor().shape();
        auto ndim = static_cast<int64_t>(shape.size());

        int64_t channels = shape[ndim - 3];
        int64_t cpg = channels / groups_;

        std::vector<int64_t> reshaped;
        for (int64_t i = 0; i < ndim - 3; ++i) reshaped.push_back(shape[i]);
        reshaped.push_back(cpg);
        reshaped.push_back(groups_);
        for (int64_t i = ndim - 2; i < ndim; ++i) reshaped.push_back(shape[i]);

        auto r = reshape(grad, reshaped);
        auto rndim = static_cast<int64_t>(r.tensor().shape().size());

        std::vector<int64_t> perm;
        for (int64_t i = 0; i < rndim - 4; ++i) perm.push_back(i);
        perm.push_back(rndim - 3);
        perm.push_back(rndim - 4);
        perm.push_back(rndim - 2);
        perm.push_back(rndim - 1);

        auto permuted = permute(r, perm);
        return {reshape(permuted, input_shape_)};
    }

private:
    int64_t groups_;
    std::vector<int64_t> input_shape_;
};

ChannelShuffle::ChannelShuffle(int64_t groups)
    : groups_(groups) {
    if (groups < 1) {
        throw std::invalid_argument("ChannelShuffle: groups must be >= 1");
    }
}

auto ChannelShuffle::forward_impl(const Variable& input) -> Variable {
    auto shape = input.tensor().shape();
    auto ndim = static_cast<int64_t>(shape.size());
    if (ndim < 3) {
        throw std::invalid_argument("ChannelShuffle: input must have at least 3 dimensions (*, C, H, W)");
    }

    int64_t channels = shape[ndim - 3];
    if (channels % groups_ != 0) {
        throw std::invalid_argument("ChannelShuffle: channels (" + std::to_string(channels) +
            ") must be divisible by groups (" + std::to_string(groups_) + ")");
    }

    int64_t cpg = channels / groups_;

    // Reshape: (*, C, H, W) -> (*, G, C/G, H, W)
    std::vector<int64_t> reshaped_shape;
    for (int64_t i = 0; i < ndim - 3; ++i) reshaped_shape.push_back(shape[i]);
    reshaped_shape.push_back(groups_);
    reshaped_shape.push_back(cpg);
    for (int64_t i = ndim - 2; i < ndim; ++i) reshaped_shape.push_back(shape[i]);

    auto reshaped = input.tensor().reshape(reshaped_shape);
    auto rndim = static_cast<int64_t>(reshaped.shape().size());

    // Permute: (*, G, C/G, H, W) -> (*, C/G, G, H, W)
    std::vector<int64_t> perm;
    for (int64_t i = 0; i < rndim - 4; ++i) perm.push_back(i);
    perm.push_back(rndim - 3);  // C/G
    perm.push_back(rndim - 4);  // G
    perm.push_back(rndim - 2);  // H
    perm.push_back(rndim - 1);  // W

    // See PixelShuffle::forward_impl above for why tenzor::permute
    // (Variable-level, tracer-visible) replaces the raw Tensor::permute()
    // here (JIT-R045).
    auto permuted = tenzor::permute(Variable(reshaped, false), perm).tensor().contiguous();

    // Reshape back: (*, C/G, G, H, W) -> (*, C, H, W)
    std::vector<int64_t> original_shape(shape.begin(), shape.end());
    auto output_tensor = permuted.reshape(original_shape);
    Variable output(output_tensor, input.requires_grad());

    if (input.requires_grad()) {
        std::vector<int64_t> input_shape_vec(shape.begin(), shape.end());
        auto cs_fn = std::make_shared<ChannelShuffleBackward>(groups_, input_shape_vec);
        std::vector<Variable> input_vars{input};
        cs_fn->set_input_variables(input_vars);
        std::vector<std::shared_ptr<Function>> next_funcs;
        if (input.grad_fn()) next_funcs.push_back(input.grad_fn());
        cs_fn->set_next_functions(next_funcs);
        output.set_grad_fn(cs_fn);
    }

    return output;
}

} // namespace tenzor::nn
