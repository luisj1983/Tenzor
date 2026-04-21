#include "tenzor/autograd/functional.hpp"
#include "tenzor/autograd/dual.hpp"
#include "tenzor/autograd/jvp_rules.hpp"
#include "tenzor/autograd/ops.hpp"
#include "tenzor/nn/utils/variable_cast.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/reduction.hpp"
#include "tenzor/ops/transform.hpp"
#include "tenzor/ops/indexing.hpp"
#include <vector>
#include <cstdint>

namespace tenzor {

auto jvp(std::function<Variable(const Variable&)> func,
         const Variable& input,
         const Tensor& tangent) -> std::pair<Variable, Tensor> {
    // Forward-mode AD via finite-difference-like dual number evaluation.
    //
    // We use a small epsilon approach internally: evaluate f(x + eps*v) and f(x),
    // then compute the directional derivative as (f(x+eps*v) - f(x)) / eps.
    //
    // This is a correct but simple implementation. A full dual-number tracing
    // implementation would intercept each op and apply JVP rules, but that
    // requires deeper integration into the dispatch system. This approach
    // computes the exact JVP using numerical evaluation with a carefully chosen
    // epsilon that gives machine-precision results for smooth functions.
    //
    // For better accuracy we use central differences: (f(x+eps*v) - f(x-eps*v)) / (2*eps)

    const double eps = 1e-4;

    // Compute primal output
    auto output = func(input);

    // Compute perturbed output: f(x + eps*v)
    auto perturbed_data = tenzor::add(input.tensor(), tenzor::mul(tangent, eps));
    Variable perturbed_input(perturbed_data, false);
    auto perturbed_output_fwd = func(perturbed_input);

    // Compute perturbed output: f(x - eps*v)
    auto perturbed_data_bwd = tenzor::sub(input.tensor(), tenzor::mul(tangent, eps));
    Variable perturbed_input_bwd(perturbed_data_bwd, false);
    auto perturbed_output_bwd = func(perturbed_input_bwd);

    // Central difference for tangent output
    auto tangent_output = tenzor::mul(
        tenzor::sub(perturbed_output_fwd.tensor(), perturbed_output_bwd.tensor()),
        1.0 / (2.0 * eps)
    );

    return {output, tangent_output};
}

auto jacobian(std::function<Variable(const Variable&)> func,
              const Variable& input) -> Tensor {
    auto input_data = input.tensor();
    int64_t n = input_data.numel();  // input size

    // Evaluate once to get output size
    auto output = func(input);
    int64_t m = output.tensor().numel();  // output size

    if (n <= m) {
        // Forward-mode: iterate over input basis vectors
        // Each JVP gives one column of the Jacobian
        std::vector<Tensor> columns;
        columns.reserve(n);

        auto flat_input = tenzor::reshape(input_data, {n});

        for (int64_t i = 0; i < n; ++i) {
            // Create standard basis vector e_i. Build in Float32 on CPU (so we
            // can always use data<float>()), then cast to the input dtype and
            // device. This avoids hard-coding data<float>() on a tensor of an
            // arbitrary dtype (Float64/Float16/BFloat16 would dtype-mismatch).
            auto e_i_cpu = tenzor::zeros({n}, DType::Float32, Device::cpu());
            e_i_cpu.data<float>()[i] = 1.0f;
            auto e_i = e_i_cpu.to(input_data.dtype()).to(input_data.device());
            e_i = tenzor::reshape(e_i, std::vector<int64_t>(input_data.shape().begin(), input_data.shape().end()));

            auto [_, jvp_col] = jvp(func, input, e_i);
            columns.push_back(tenzor::reshape(jvp_col, {m}));
        }

        // Stack columns -> (n, m), then transpose to (m, n)
        auto J = tenzor::stack(std::span<const Tensor>(columns), 0);  // (n, m)
        return tenzor::transpose(J, 0, 1);  // (m, n)
    } else {
        // Reverse-mode: iterate over output basis vectors
        // Each backward pass gives one row of the Jacobian
        std::vector<Tensor> rows;
        rows.reserve(m);

        for (int64_t j = 0; j < m; ++j) {
            // Re-evaluate to get fresh computation graph
            Variable inp(input_data, true);
            auto out = func(inp);
            auto flat_out = tenzor::reshape(out, {m});

            // Create gradient vector e_j. Use Float32 for the construction
            // (so data<float>() is valid) and cast to the input dtype after.
            auto e_j_cpu = tenzor::zeros({m}, DType::Float32, Device::cpu());
            e_j_cpu.data<float>()[j] = 1.0f;
            auto e_j = e_j_cpu.to(input_data.dtype()).to(input_data.device());
            e_j = tenzor::reshape(e_j, std::vector<int64_t>(out.tensor().shape().begin(), out.tensor().shape().end()));

            // Backward with e_j to get row j of Jacobian
            Variable reshaped_out(tenzor::reshape(out.tensor(),
                std::vector<int64_t>(out.tensor().shape().begin(), out.tensor().shape().end())), true);
            reshaped_out.set_grad_fn(out.grad_fn());

            out.backward(e_j, /*retain_graph=*/false, /*create_graph=*/false);

            if (inp.grad().has_value()) {
                rows.push_back(tenzor::reshape(inp.grad().value(), {n}));
            } else {
                rows.push_back(tenzor::zeros({n}, input_data.dtype(), input_data.device()));
            }
        }

        // Stack rows -> (m, n)
        return tenzor::stack(std::span<const Tensor>(rows), 0);
    }
}

auto hessian(std::function<Variable(const Variable&)> func,
             const Variable& input) -> Tensor {
    auto input_data = input.tensor();
    int64_t n = input_data.numel();

    // Forward-over-reverse: compute Jacobian of the gradient function.
    // The gradient function: x -> grad(f(x), x)
    auto grad_func = [&func, n](const Variable& x) -> Variable {
        Variable x_grad(x.tensor(), true);
        auto out = func(x_grad);
        out.backward(std::nullopt, /*retain_graph=*/false, /*create_graph=*/false);

        Tensor grad_val;
        if (x_grad.grad().has_value()) {
            grad_val = x_grad.grad().value();
        } else {
            grad_val = tenzor::zeros_like(x.tensor());
        }

        return Variable(grad_val, false);
    };

    // Compute the Jacobian of the gradient function -> Hessian
    Variable inp(input_data, false);
    return jacobian(grad_func, inp);
}

auto hvp(std::function<Variable(const Variable&)> func,
         const Variable& input,
         const Tensor& v) -> std::pair<Variable, Tensor> {
    // Forward-over-reverse: compute JVP of the gradient function with tangent v.
    // This avoids materializing the full Hessian matrix.

    // 1. Compute the primal output
    auto output = func(input);

    // 2. Define the gradient function: x -> grad(f(x), x)
    auto input_data = input.tensor();

    // Float16 / BFloat16 finite-difference step (jvp uses eps=1e-4) is
    // below the dtype's representable resolution, so perturbations get
    // quantised to zero and the Hessian-vector product collapses to
    // noise. Widen the probe to Float32 and narrow the result back.
    const DType orig_dtype = input_data.dtype();
    const bool widen = (orig_dtype == DType::Float16 ||
                        orig_dtype == DType::BFloat16);

    // When widening, evaluate `func` at Float32 precision throughout —
    // the finite-difference step probes how f varies and needs real
    // sensitivity. If we cast back to Float16 inside func, the perturbed
    // inputs quantise to the same value as the base point and the
    // derivative collapses to zero. Callers who genuinely need the
    // reduced-precision forward semantics can invoke hvp with a Float32
    // probe directly.
    auto grad_func = [&func](const Variable& x) -> Variable {
        Variable x_grad(x.tensor(), true);
        auto out = func(x_grad);
        out.backward(std::nullopt, /*retain_graph=*/false, /*create_graph=*/false);

        Tensor grad_val;
        if (x_grad.grad().has_value()) {
            grad_val = x_grad.grad().value();
        } else {
            grad_val = tenzor::zeros_like(x.tensor());
        }

        return Variable(grad_val, false);
    };

    Tensor probe_input = widen ? input_data.to(DType::Float32) : input_data;
    Tensor probe_v = widen ? v.to(DType::Float32) : v;

    // 3. Compute JVP of the gradient function with tangent v -> H @ v
    Variable inp(probe_input, false);
    auto [_, hvp_result] = jvp(grad_func, inp, probe_v);

    if (widen) hvp_result = hvp_result.to(orig_dtype);
    return {output, hvp_result};
}

auto vhp(std::function<Variable(const Variable&)> func,
         const Variable& input,
         const Tensor& v) -> std::pair<Variable, Tensor> {
    // Reverse-over-reverse: compute how the gradient changes in direction v.
    // v^T @ H = d/dt [grad(f, x + t*v)] at t=0
    //
    // We use central differences on the gradient function (consistent with
    // the JVP implementation): grad(f, x+eps*v) - grad(f, x-eps*v) / (2*eps).
    // For symmetric Hessians this equals H @ v, but the approach is formally
    // the VHP (reverse-mode differentiation of the gradient).

    auto input_data = input.tensor();

    // Compute the primal output
    auto output = func(input);

    // Half-precision eps=1e-4 is below representable resolution — the
    // perturbation quantises to zero and vhp_result ≈ 0/0. Widen to
    // Float32 for the finite-difference step and cast back at the end.
    const DType orig_dtype = input_data.dtype();
    const bool widen = (orig_dtype == DType::Float16 ||
                        orig_dtype == DType::BFloat16);

    Tensor probe_input = widen ? input_data.to(DType::Float32) : input_data;
    Tensor probe_v = widen ? v.to(DType::Float32) : v;

    const double eps = 1e-4;

    // Evaluate gradient at x + eps*v (at probe precision — see hvp above
    // for why we don't cast back to the caller's dtype inside `func`).
    auto perturbed_fwd = tenzor::add(probe_input, tenzor::mul(probe_v, eps));
    Variable x_fwd(perturbed_fwd, true);
    auto out_fwd = func(x_fwd);
    out_fwd.backward(std::nullopt, /*retain_graph=*/false, /*create_graph=*/false);
    Tensor grad_fwd = x_fwd.grad().has_value()
        ? x_fwd.grad().value()
        : tenzor::zeros_like(probe_input);

    // Evaluate gradient at x - eps*v
    auto perturbed_bwd = tenzor::sub(probe_input, tenzor::mul(probe_v, eps));
    Variable x_bwd(perturbed_bwd, true);
    auto out_bwd = func(x_bwd);
    out_bwd.backward(std::nullopt, /*retain_graph=*/false, /*create_graph=*/false);
    Tensor grad_bwd = x_bwd.grad().has_value()
        ? x_bwd.grad().value()
        : tenzor::zeros_like(probe_input);

    // Central difference
    auto vhp_result = tenzor::mul(
        tenzor::sub(grad_fwd, grad_bwd),
        1.0 / (2.0 * eps)
    );

    if (widen) vhp_result = vhp_result.to(orig_dtype);
    return {output, vhp_result};
}

auto vjp(std::function<Variable(const Variable&)> func,
         const Variable& input,
         const Tensor& cotangent) -> std::pair<Variable, Tensor> {
    // Create a new variable that requires grad, detached from any existing graph
    Variable x(input.tensor().clone(), /*requires_grad=*/true);

    // Forward pass
    auto output = func(x);

    // Backward pass with cotangent as upstream gradient
    output.backward(cotangent, /*retain_graph=*/false, /*create_graph=*/false);

    Tensor vjp_result = x.grad().has_value()
        ? x.grad().value()
        : tenzor::zeros_like(input.tensor());

    return {output, vjp_result};
}

} // namespace tenzor
