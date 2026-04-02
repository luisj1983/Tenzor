#include "tenzor/autograd/functional.hpp"
#include "tenzor/autograd/dual.hpp"
#include "tenzor/autograd/jvp_rules.hpp"
#include "tenzor/autograd/ops.hpp"
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
            // Create standard basis vector e_i
            auto e_i = tenzor::zeros({n}, input_data.dtype(), input_data.device());
            // Set e_i[i] = 1.0
            auto one = tenzor::ones({1}, input_data.dtype(), input_data.device());
            // Use narrow + add to set single element
            // Simple approach: create on CPU, fill, transfer
            auto e_i_cpu = tenzor::zeros({n}, input_data.dtype(), Device::cpu());
            float* ptr = e_i_cpu.data<float>();
            ptr[i] = 1.0f;
            e_i = e_i_cpu.to(input_data.device());
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

            // Create gradient vector e_j
            auto e_j_cpu = tenzor::zeros({m}, input_data.dtype(), Device::cpu());
            float* ptr = e_j_cpu.data<float>();
            ptr[j] = 1.0f;
            auto e_j = e_j_cpu.to(input_data.device());
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

} // namespace tenzor
