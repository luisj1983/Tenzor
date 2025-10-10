#include "tenzor/nn/layers/linear.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/reduction.hpp"
#include "tenzor/autograd/function.hpp"
#include <cmath>

namespace tenzor::nn {

// LinearBackward autograd function
class LinearBackward : public Function {
public:
    LinearBackward(bool has_bias, std::vector<Tensor> tensors_to_save)
        : has_bias_(has_bias) {
        saved_tensors_ = std::move(tensors_to_save);
    }

    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override {
        throw std::runtime_error("LinearBackward::forward should not be called");
    }

    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override {
        auto& grad_output = grad_outputs[0];
        auto saved = saved_tensors();
        auto& input = saved[0];
        auto& weight = saved[1];

        // grad_output: [batch, out_features]
        // input: [batch, in_features]
        // weight: [out_features, in_features]

        // Ensure tensors are contiguous for matmul
        // For GPU tensors, use .to() which handles non-contiguous transfers
        auto make_contiguous = [](const Tensor& t) -> Tensor {
            if (t.device().type == Device::Type::CUDA) {
                return t.to(t.device());  // .to() makes GPU tensors contiguous
            } else {
                return t.contiguous();  // CPU can use .contiguous() directly
            }
        };

        auto grad_output_contig = make_contiguous(grad_output);
        auto input_contig = make_contiguous(input);
        auto weight_contig = make_contiguous(weight);

        // Gradient with respect to input: grad_output @ weight
        auto grad_input = matmul(grad_output_contig, weight_contig);

        // Gradient with respect to weight: grad_output.T @ input
        auto grad_weight = matmul(make_contiguous(grad_output_contig.transpose(-2, -1)), input_contig);

        if (has_bias_) {
            // Gradient with respect to bias: sum(grad_output, dim=0)
            auto grad_bias = sum(grad_output_contig, 0, false);
            return std::vector<Tensor>{grad_input, grad_weight, grad_bias};
        } else {
            return std::vector<Tensor>{grad_input, grad_weight};
        }
    }

private:
    bool has_bias_;
};

Linear::Linear(int64_t in_features, int64_t out_features, bool bias)
    : in_features_(in_features), out_features_(out_features) {

    // Initialize weight with Xavier/Glorot initialization
    float std = std::sqrt(2.0f / (in_features + out_features));
    weight_ = Variable(randn({out_features, in_features}) * std, true);
    register_parameter("weight", weight_);

    // Initialize bias
    if (bias) {
        bias_ = Variable(zeros({out_features}), true);
        register_parameter("bias", *bias_);
    }
}

auto Linear::forward(const Variable& input) -> Variable {
    // input: [batch, in_features]
    // weight: [out_features, in_features]
    // output: [batch, out_features]

    // CRITICAL: Access weight from parameters_ map, not cached member variable
    // This ensures we get the correct device after Module::to() is called
    auto& weight = parameters_["weight"];

    // CRITICAL FIX: Use .to(device) instead of .contiguous() to handle non-contiguous GPU tensors
    // transpose() creates a non-contiguous view, and .to(same_device) makes it contiguous
    // Use input's device to ensure weight is on the same device as input
    auto weight_t = weight.tensor().transpose(-2, -1).to(input.tensor().device());
    auto output_tensor = matmul(input.tensor(), weight_t);

    // Access bias from parameters_ map if it exists
    bool has_bias = parameters_.find("bias") != parameters_.end();
    if (has_bias) {
        auto& bias = parameters_["bias"];

        // Use native broadcasting (now works on both CPU and CUDA!)
        // bias shape: [out_features], output shape: [batch, out_features]
        // Broadcasting will automatically expand bias from [out_features] to [batch, out_features]
        output_tensor = output_tensor + bias.tensor();
    }

    // Set up autograd if needed
    bool requires_grad = input.requires_grad() || weight.requires_grad();
    if (has_bias && parameters_["bias"].requires_grad()) {
        requires_grad = true;
    }

    if (requires_grad) {
        // Create result variable from output
        auto result = Variable(output_tensor, true);

        // Prepare tensors to save for backward
        std::vector<Tensor> tensors_to_save = {
            input.tensor(),
            weight.tensor()
        };

        // Create backward function with saved tensors
        auto grad_fn = std::make_shared<LinearBackward>(
            has_bias, std::move(tensors_to_save)
        );

        result.set_grad_fn(grad_fn);

        // Track input variables for gradient accumulation
        std::vector<Variable*> input_vars;
        if (input.requires_grad()) {
            input_vars.push_back(const_cast<Variable*>(&input));
        }

        auto weight_it = parameters_.find("weight");
        if (weight_it != parameters_.end() && weight_it->second.requires_grad()) {
            input_vars.push_back(&(weight_it->second));
        }

        if (has_bias) {
            auto bias_it = parameters_.find("bias");
            if (bias_it != parameters_.end() && bias_it->second.requires_grad()) {
                input_vars.push_back(&(bias_it->second));
            }
        }

        grad_fn->set_input_variables(input_vars);

        // CRITICAL FIX: Connect to input's grad_fn to continue the backward chain
        std::vector<std::shared_ptr<Function>> next_funcs;
        if (input.grad_fn()) {
            next_funcs.push_back(input.grad_fn());
        }
        grad_fn->set_next_functions(next_funcs);

        return result;
    } else {
        return Variable(output_tensor, false);
    }
}

auto Linear::reset_parameters() -> void {
    // Xavier/Glorot initialization already done in constructor
}

} // namespace tenzor::nn
