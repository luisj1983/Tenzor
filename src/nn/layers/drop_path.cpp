#include "tenzor/nn/layers/drop_path.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/autograd/function.hpp"
#include <stdexcept>

namespace tenzor::nn {

// DropPath autograd function
class DropPathBackward : public Function {
public:
    DropPathBackward(Tensor mask, double scale) : mask_(std::move(mask)), scale_(scale) {}

    auto forward([[maybe_unused]] std::vector<Variable> inputs) -> std::vector<Variable> override {
        throw std::runtime_error("DropPathBackward::forward should not be called");
    }

    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override {
        if (grad_outputs.size() != 1) {
            throw std::invalid_argument("DropPathBackward expects 1 gradient output");
        }

        // Gradient: grad_input = grad_output * mask * scale
        // mask is (batch, 1, 1, ..., 1) and broadcasts to input shape. scale is
        // a constant, so multiply by a scalar ({1}-shaped) tensor that
        // broadcasts, instead of materialising a full grad-sized buffer.
        auto grad_masked = mul(grad_outputs[0], mask_);
        auto scale_tensor = full({1}, static_cast<float>(scale_),
                                 grad_outputs[0].dtype(), grad_outputs[0].device());
        auto grad_input = mul(grad_masked, scale_tensor);

        return {grad_input};
    }

    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override {
        if (grad_outputs.size() != 1) {
            throw std::invalid_argument("DropPathBackward expects 1 gradient output");
        }

        // Gradient: grad_input = grad_output * mask * scale
        // mask is a constant (no grad tracking needed), broadcasts automatically
        auto& grad = grad_outputs[0];
        Variable mask_var(mask_, false);
        auto grad_masked = grad * mask_var;
        auto grad_input = grad_masked * scale_;
        return {grad_input};
    }

private:
    Tensor mask_;
    double scale_;
};

DropPath::DropPath(double p) : p_(p) {
    if (p < 0.0 || p >= 1.0) {
        throw std::invalid_argument("DropPath probability must be in [0, 1)");
    }
}

auto DropPath::forward_impl(const Variable& input) -> Variable {
    // During inference or p==0, return input unchanged
    if (!is_training() || p_ == 0.0) {
        return input;
    }

    auto shape_span = input.tensor().shape();
    if (shape_span.empty()) {
        return input;
    }

    int64_t batch_size = shape_span[0];

    // Create mask shape: (batch, 1, 1, ..., 1) — one value per sample
    std::vector<int64_t> mask_shape(shape_span.size(), 1);
    mask_shape[0] = batch_size;

    // Generate Bernoulli mask on target device using tensor ops. Draw the
    // uniforms and run the threshold compare in Float32, then cast the 0/1 mask
    // to the input dtype. Generating randoms directly in a half dtype quantizes
    // both the uniform draw and the compare, biasing the realised keep
    // probability away from (1 - p); Dropout does exactly this for the same
    // reason.
    Device dev = input.tensor().device();
    DType dt = input.tensor().dtype();
    // Draw the uniforms on CPU, then copy to the input's device. On-device RNG
    // (rand(..., dev)) yields a backend-divergent mask, so a seeded run would
    // produce different masks on CPU vs CUDA vs ROCm etc. Generating on CPU and
    // transferring (like nn::init) makes the mask reproducible across backends
    // (F095). The inverted-dropout 1/(1-p) scaling and eval no-op above are
    // unchanged.
    auto random_tensor = rand(mask_shape, DType::Float32, Device::cpu());
    if (dev.type != Device::Type::CPU) {
        random_tensor = random_tensor.to(dev);
    }
    auto threshold = full(mask_shape, static_cast<float>(p_), DType::Float32, dev);
    auto mask_f32 = gt(random_tensor, threshold).to(DType::Float32);
    auto mask_data = (dt == DType::Float32) ? mask_f32 : mask_f32.to(dt);

    // Apply inverted drop path: output = input * mask / (1 - p)
    // mask broadcasts from (batch, 1, ..., 1) to input shape
    double scale = 1.0 / (1.0 - p_);
    auto scale_tensor = full({1}, static_cast<float>(scale), dt, dev);
    auto output_tensor = mul(mul(input.tensor(), mask_data), scale_tensor);

    Variable output(output_tensor, input.requires_grad());

    // Set up autograd
    if (input.requires_grad()) {
        auto drop_path_fn = std::make_shared<DropPathBackward>(mask_data, scale);

        std::vector<Variable> input_vars = {input};
        drop_path_fn->set_input_variables(input_vars);

        std::vector<std::shared_ptr<Function>> next_funcs;
        if (input.grad_fn()) {
            next_funcs.push_back(input.grad_fn());
        }
        drop_path_fn->set_next_functions(next_funcs);

        output.set_grad_fn(drop_path_fn);
    }

    return output;
}

} // namespace tenzor::nn
