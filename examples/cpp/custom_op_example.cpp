/**
 * @file custom_op_example.cpp
 * @brief Demonstrates defining a custom differentiable op via autograd::Function.
 *
 * A custom op is created by subclassing `tenzor::Function`, implementing
 * `forward()` and `backward()`, and wiring the result into the autograd
 * graph. This example implements the logistic sigmoid from scratch —
 * forward:  y = 1 / (1 + e^-x)
 * backward: dy/dx = y * (1 - y)
 * — runs it on a small CPU tensor, back-propagates from loss = sum(y), and
 * checks the autograd gradient against the closed-form sigmoid derivative.
 */

#include <tenzor/tenzor.hpp>

#include <cmath>
#include <iostream>
#include <vector>

namespace {

/// A user-defined differentiable op: the logistic sigmoid, built from raw
/// tensor ops with a hand-written backward.
///
/// The contract (see include/tenzor/autograd/function.hpp):
///   - forward() computes the output and calls save_for_backward() with any
///     tensor backward() will need (here, the output y itself).
///   - backward() receives the gradient w.r.t. the output and returns the
///     gradient w.r.t. each input, using the saved tensors.
class MySigmoid : public tenzor::Function {
public:
    auto forward(std::vector<tenzor::Variable> inputs) -> std::vector<tenzor::Variable> override {
        const auto& x = inputs[0].tensor();
        // y = 1 / (1 + exp(-x)). Scalar/tensor operators broadcast the scalar.
        tenzor::Tensor y = 1.0 / (1.0 + tenzor::exp(0.0 - x));
        // dy/dx = y * (1 - y), so save y for backward.
        save_for_backward({y});
        return {tenzor::Variable(y, /*requires_grad=*/false)};
    }

    auto backward(std::vector<tenzor::Tensor> grad_outputs) -> std::vector<tenzor::Tensor> override {
        const auto& grad_y = grad_outputs[0];
        const auto& y = saved_tensors()[0];
        // grad_x = grad_y * y * (1 - y)
        tenzor::Tensor grad_x = grad_y * y * (1.0 - y);
        return {grad_x};
    }

    auto name() const -> std::string override { return "MySigmoid"; }
};

/// Apply the custom op to a Variable, wiring the result into the autograd
/// graph so backward() will be invoked during Variable::backward().
/// This mirrors the framework's own apply path (see python/bindings.cpp
/// `apply_custom_function`): run forward, then connect next_functions /
/// input_variables / grad_fn when any input requires grad.
auto my_sigmoid(tenzor::Variable x) -> tenzor::Variable {
    auto fn = std::make_shared<MySigmoid>();
    tenzor::Variable y = fn->forward({x})[0];

    if (tenzor::is_grad_enabled() && x.requires_grad()) {
        fn->set_input_variables({x});
        fn->set_next_functions({x.grad_fn()});
        y.set_requires_grad(true);
        y.set_grad_fn(fn);
    }
    return y;
}

/// Print a 1-D CPU Float32 tensor as [v0, v1, ...].
auto format_tensor(const tenzor::Tensor& t) -> std::string {
    const float* p = t.data<float>();
    std::string s = "[";
    for (size_t i = 0; i < t.numel(); ++i) {
        if (i != 0) {
            s += ", ";
        }
        s += std::to_string(p[i]);
    }
    s += "]";
    return s;
}

} // namespace

int main() {
    using namespace tenzor;

    initialize();  // register CPU (and any built) backends

    std::cout << "Tenzor Custom Autograd Function Example\n";
    std::cout << "========================================\n\n";

    // Leaf input: x = [-2, -1, 0, 1, 2], tracked for autograd.
    std::vector<float> x_vals{-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
    Tensor x_tensor = from_data(x_vals.data(),
                                {static_cast<int64_t>(x_vals.size())},
                                Device::cpu());
    Variable x(x_tensor, /*requires_grad=*/true);

    std::cout << "x        = " << format_tensor(x_tensor) << "\n";

    // Forward through the custom op.
    Variable y = my_sigmoid(x);
    std::cout << "y=sigmoid(x) = " << format_tensor(y.tensor()) << "\n";

    // Backward from loss = sum(y). dL/dy = 1, so seed with an all-ones tensor.
    Tensor grad_seed = from_data(std::vector<float>(x_vals.size(), 1.0f).data(),
                                 {static_cast<int64_t>(x_vals.size())},
                                 Device::cpu());
    y.backward(grad_seed);

    // Autograd input gradient: dL/dx = y * (1 - y).
    const auto& x_grad_opt = x.grad();
    if (!x_grad_opt) {
        std::cerr << "ERROR: no gradient accumulated on x\n";
        return 1;
    }
    std::cout << "dL/dx (autograd) = " << format_tensor(*x_grad_opt) << "\n";

    // Verify against the closed-form derivative: sigmoid'(x) = s*(1-s).
    std::cout << "\nVerifying dL/dx == sigmoid(x) * (1 - sigmoid(x)):\n";
    bool ok = true;
    const float* y_ptr = y.tensor().data<float>();
    const float* g_ptr = x_grad_opt->data<float>();
    for (size_t i = 0; i < x_vals.size(); ++i) {
        const float s = 1.0f / (1.0f + std::exp(-x_vals[i]));
        const float expected = s * (1.0f - s);
        const float got = g_ptr[i];
        const float err = std::fabs(got - expected);
        const bool elem_ok = err <= 1e-5f * (1.0f + std::fabs(expected));
        ok = ok && elem_ok;
        std::cout << "  x=" << x_vals[i]
                  << "  y=" << y_ptr[i]
                  << "  grad=" << got
                  << "  expected=" << expected
                  << (elem_ok ? "  OK" : "  MISMATCH") << "\n";
    }

    std::cout << "\nCustom op " << (ok ? "PASSED" : "FAILED") << "\n";
    return ok ? 0 : 1;
}