/**
 * @file gradcheck.cpp
 * @brief Implementation of numerical gradient checking
 */

#include "tenzor/autograd/gradcheck.hpp"
#include "tenzor/autograd/variable.hpp"
#include "tenzor/autograd/ops.hpp"
#include "tenzor/autograd/functional.hpp"
#include "tenzor/core/tensor.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/reduction.hpp"
#include <cmath>
#include <complex>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <algorithm>
#include <chrono>

namespace tenzor {

namespace {

/**
 * @brief Create a zero tensor with same properties as input.
 *
 * Uses backend-agnostic zeros() operation instead of zero_() to avoid
 * CUDA pointer dereference issues. The zeros() function has native
 * implementations for all backends (CPU, CUDA, OneAPI, Vulkan, ROCm).
 */
auto zeros_like_tensor(const Tensor& tensor) -> Tensor {
    return zeros(std::vector<int64_t>(tensor.shape().begin(), tensor.shape().end()),
                 tensor.dtype(),
                 tensor.device());
}

/**
 * @brief Extract scalar value from a tensor (supports 0-d and 1-element tensors).
 */
auto extract_scalar(const Tensor& tensor) -> double {
    // Handle 0-dimensional tensors
    if (tensor.ndim() == 0 || tensor.numel() == 1) {
        if (tensor.dtype() == DType::Float32) {
            return static_cast<double>(tensor.item<float>());
        } else if (tensor.dtype() == DType::Float64) {
            return tensor.item<double>();
        } else if (tensor.dtype() == DType::Int32) {
            return static_cast<double>(tensor.item<int32_t>());
        } else if (tensor.dtype() == DType::Int64) {
            return static_cast<double>(tensor.item<int64_t>());
        }
    }
    throw std::runtime_error("Cannot extract scalar from non-scalar tensor");
}

/**
 * @brief Extract a real scalar from a (possibly complex) scalar tensor.
 *
 * For real dtypes this is identical to extract_scalar. For complex dtypes it
 * mirrors the first-order gradcheck contraction L = Σ Re(y) + Σ Im(y): a
 * scalar complex output y is reduced to Re(y) + Im(y). This lets the
 * second-order finite-difference path in gradgradcheck_detailed evaluate
 * complex-valued scalar functions instead of throwing in extract_scalar.
 */
auto extract_scalar_real_contraction(const Tensor& tensor) -> double {
    if (tensor.ndim() == 0 || tensor.numel() == 1) {
        if (tensor.dtype() == DType::Complex64) {
            auto v = tensor.item<std::complex<float>>();
            return static_cast<double>(v.real()) + static_cast<double>(v.imag());
        } else if (tensor.dtype() == DType::Complex128) {
            auto v = tensor.item<std::complex<double>>();
            return v.real() + v.imag();
        }
    }
    return extract_scalar(tensor);
}

} // anonymous namespace

auto numerical_gradient(
    std::function<Variable(const Variable&)> func,
    const Variable& input,
    double eps
) -> Tensor {
    // Adjust epsilon based on dtype to avoid precision loss
    // Float32 needs larger epsilon due to ~7 digits of precision
    // Float64 can use smaller epsilon with ~15 digits of precision
    if (input.dtype() == DType::Float32 && eps < 5e-4) {
        eps = 5e-4;  // Minimum safe epsilon for Float32 (avoids catastrophic cancellation)
    }

    // Create gradient tensor with same shape as input
    auto num_grad = zeros_like_tensor(input.tensor());

    // Get input data pointer and properties
    int64_t total_elements = input.tensor().numel();

    // Transfer num_grad to CPU once for efficient pointer-based writes
    Device original_grad_device = num_grad.device();
    Tensor num_grad_cpu = (original_grad_device == Device::cpu()) ? num_grad : num_grad.to(Device::cpu());

    // We need to perturb each element individually
    // Create a copy of input for perturbation
    Tensor input_copy = input.tensor().clone();

    // For each element in the input
    for (int64_t i = 0; i < total_elements; ++i) {
        // Clone input tensor and transfer to CPU for modification
        Device original_device = input.tensor().device();
        Tensor x_plus_cpu = input_copy.clone();
        Tensor x_minus_cpu = input_copy.clone();

        // Ensure we're on CPU for pointer-based modification
        if (original_device != Device::cpu()) {
            x_plus_cpu = x_plus_cpu.to(Device::cpu());
            x_minus_cpu = x_minus_cpu.to(Device::cpu());
        }

        // Access data based on dtype and perturb element i
        if (input.dtype() == DType::Float32) {
            float* data_plus = x_plus_cpu.data<float>();
            float* data_minus = x_minus_cpu.data<float>();

            // Perturb element i
            data_plus[i] += static_cast<float>(eps);
            data_minus[i] -= static_cast<float>(eps);

        } else if (input.dtype() == DType::Float64) {
            double* data_plus = x_plus_cpu.data<double>();
            double* data_minus = x_minus_cpu.data<double>();

            data_plus[i] += eps;
            data_minus[i] -= eps;

        } else {
            throw std::runtime_error("gradcheck only supports Float32 and Float64 dtypes");
        }

        // Create variables with perturbed tensors, transferring back to original device if needed
        Variable x_plus, x_minus;
        if (original_device != Device::cpu()) {
            x_plus = Variable(x_plus_cpu.to(original_device), false);
            x_minus = Variable(x_minus_cpu.to(original_device), false);
        } else {
            x_plus = Variable(x_plus_cpu, false);
            x_minus = Variable(x_minus_cpu, false);
        }

        // Compute function outputs
        Variable f_plus = func(x_plus);
        Variable f_minus = func(x_minus);

        // Extract scalar values (function should return scalar or we sum it)
        double val_plus, val_minus;

        // Gate the scalar fast-path on a non-complex dtype: extract_scalar only
        // handles real dtypes and throws on Complex64/Complex128. A single-
        // element complex output must take the summation branch below, which
        // reduces complex values as Re(z)+Im(z) (matching the (1+1i) backward
        // seed used by the analytical path).
        if (f_plus.tensor().numel() == 1 && !f_plus.tensor().is_complex()) {
            val_plus = extract_scalar(f_plus.tensor());
            val_minus = extract_scalar(f_minus.tensor());
        } else {
            // If output is not scalar, sum it to get scalar
            // This allows checking functions with multiple outputs
            Variable sum_plus = f_plus;
            Variable sum_minus = f_minus;

            // Sum all elements
            Tensor sum_plus_tensor = f_plus.tensor();
            Tensor sum_minus_tensor = f_minus.tensor();

            // Manual sum since we might not have sum() implemented yet
            double total_plus = 0.0, total_minus = 0.0;
            int64_t n = sum_plus_tensor.numel();

            // Transfer to CPU and materialise a contiguous copy so that
            // data<T>() + n correctly enumerates logical elements. Views
            // returned by ops like slice() share storage with the parent
            // tensor and have non-trivial strides; without contiguous() we
            // would iterate over the parent's memory instead of the view.
            Device sum_device = sum_plus_tensor.device();
            Tensor sum_plus_cpu = (sum_device == Device::cpu()) ? sum_plus_tensor : sum_plus_tensor.to(Device::cpu());
            Tensor sum_minus_cpu = (sum_device == Device::cpu()) ? sum_minus_tensor : sum_minus_tensor.to(Device::cpu());
            if (!sum_plus_cpu.is_contiguous())  sum_plus_cpu  = sum_plus_cpu.contiguous();
            if (!sum_minus_cpu.is_contiguous()) sum_minus_cpu = sum_minus_cpu.contiguous();

            // Reduced-precision outputs (e.g. a func that casts a Float32 input
            // down to Float16/BFloat16) have no dedicated summation branch
            // below. Widen the CPU copy to Float32 first (widen-narrow pattern)
            // so total_plus/total_minus are actually populated; otherwise both
            // stay 0 and the numerical gradient is silently zero for every
            // element. The output dtype is otherwise unconstrained (the input
            // guard only restricts the INPUT to Float32/Float64).
            DType out_dtype = f_plus.dtype();
            if (out_dtype == DType::Float16 || out_dtype == DType::BFloat16) {
                sum_plus_cpu  = sum_plus_cpu.to(DType::Float32);
                sum_minus_cpu = sum_minus_cpu.to(DType::Float32);
                out_dtype = DType::Float32;
            }

            if (out_dtype == DType::Float32) {
                const float* data_p = sum_plus_cpu.data<float>();
                const float* data_m = sum_minus_cpu.data<float>();
                for (int64_t j = 0; j < n; ++j) {
                    total_plus += data_p[j];
                    total_minus += data_m[j];
                }
            } else if (out_dtype == DType::Float64) {
                const double* data_p = sum_plus_cpu.data<double>();
                const double* data_m = sum_minus_cpu.data<double>();
                for (int64_t j = 0; j < n; ++j) {
                    total_plus += data_p[j];
                    total_minus += data_m[j];
                }
            } else if (out_dtype == DType::Complex64) {
                // For a real-valued input x producing complex y = f(x),
                // gradcheck reduces y to a real scalar by summing both
                // real and imaginary parts. Equivalent to evaluating
                // ∂(Σ Re(y) + Σ Im(y)) / ∂x so the finite-difference
                // and analytical branches see the same scalar contraction.
                const auto* data_p = sum_plus_cpu.data<std::complex<float>>();
                const auto* data_m = sum_minus_cpu.data<std::complex<float>>();
                for (int64_t j = 0; j < n; ++j) {
                    total_plus += static_cast<double>(data_p[j].real())
                                + static_cast<double>(data_p[j].imag());
                    total_minus += static_cast<double>(data_m[j].real())
                                 + static_cast<double>(data_m[j].imag());
                }
            } else if (out_dtype == DType::Complex128) {
                const auto* data_p = sum_plus_cpu.data<std::complex<double>>();
                const auto* data_m = sum_minus_cpu.data<std::complex<double>>();
                for (int64_t j = 0; j < n; ++j) {
                    total_plus += data_p[j].real() + data_p[j].imag();
                    total_minus += data_m[j].real() + data_m[j].imag();
                }
            }

            val_plus = total_plus;
            val_minus = total_minus;
        }

        // Compute numerical gradient using central differences
        double numerical_grad_i = (val_plus - val_minus) / (2.0 * eps);

        // Store in gradient tensor (using pre-created CPU tensor)
        if (num_grad.dtype() == DType::Float32) {
            num_grad_cpu.data<float>()[i] = static_cast<float>(numerical_grad_i);
        } else if (num_grad.dtype() == DType::Float64) {
            num_grad_cpu.data<double>()[i] = numerical_grad_i;
        }
    }

    // Transfer gradient tensor back to original device if needed
    if (original_grad_device != Device::cpu()) {
        num_grad = num_grad_cpu.to(original_grad_device);
    } else {
        num_grad = num_grad_cpu;
    }

    return num_grad;
}

auto compare_gradients(
    const Tensor& numerical,
    const Tensor& analytical,
    double atol,
    double rtol
) -> GradCheckResult {
    GradCheckResult result;

    // Check shapes match
    auto num_shape = numerical.shape();
    auto ana_shape = analytical.shape();

    if (num_shape.size() != ana_shape.size()) {
        result.passed = false;
        result.error_message = "Gradient shapes don't match";
        return result;
    }

    for (size_t i = 0; i < num_shape.size(); ++i) {
        if (num_shape[i] != ana_shape[i]) {
            result.passed = false;
            result.error_message = "Gradient shapes don't match";
            return result;
        }
    }

    // Compare element-wise
    int64_t total = numerical.numel();
    result.total_elements = total;
    result.max_abs_error = 0.0;
    result.max_rel_error = 0.0;
    result.failing_elements = 0;

    const bool is_float32 = numerical.dtype() == DType::Float32;

    // Transfer to CPU for pointer-based computation
    Device num_device = numerical.device();
    Tensor numerical_cpu = (num_device == Device::cpu()) ? numerical : numerical.to(Device::cpu());
    Tensor analytical_cpu = (num_device == Device::cpu()) ? analytical : analytical.to(Device::cpu());

    for (int64_t i = 0; i < total; ++i) {
        double num_val, ana_val;

        if (is_float32) {
            num_val = static_cast<double>(numerical_cpu.data<float>()[i]);
            ana_val = static_cast<double>(analytical_cpu.data<float>()[i]);
        } else {
            num_val = numerical_cpu.data<double>()[i];
            ana_val = analytical_cpu.data<double>()[i];
        }

        // Compute errors
        double abs_error = std::abs(num_val - ana_val);
        double rel_error = 0.0;

        if (std::abs(ana_val) > 1e-12) {
            rel_error = abs_error / std::abs(ana_val);
        }

        // Update max errors
        result.max_abs_error = std::max(result.max_abs_error, abs_error);
        result.max_rel_error = std::max(result.max_rel_error, rel_error);

        // Check tolerance
        double threshold = atol + rtol * std::abs(ana_val);
        bool element_passed = abs_error <= threshold;

        if (!element_passed) {
            result.failing_elements++;

            // Store first few failing indices
            if (result.fail_indices.size() < 10) {
                result.fail_indices.push_back(i);
            }
        }

        // Store first few values for debugging
        if (i < 10) {
            result.numerical_grad.push_back(num_val);
            result.analytical_grad.push_back(ana_val);
        }
    }

    // Overall pass/fail
    result.passed = (result.failing_elements == 0);

    // Generate error message if failed
    if (!result.passed) {
        std::ostringstream oss;
        oss << std::scientific << std::setprecision(4);
        oss << "Gradient check failed!\n";
        oss << "  Max absolute error: " << result.max_abs_error << "\n";
        oss << "  Max relative error: " << result.max_rel_error << "\n";
        oss << "  Failing elements: " << result.failing_elements
            << " / " << result.total_elements << "\n";
        oss << "  Tolerances: atol=" << atol << ", rtol=" << rtol << "\n";

        if (!result.fail_indices.empty()) {
            oss << "  First failing indices: ";
            for (size_t i = 0; i < std::min(size_t(5), result.fail_indices.size()); ++i) {
                if (i > 0) oss << ", ";
                oss << result.fail_indices[i];
            }
            oss << "\n";
        }

        result.error_message = oss.str();
    } else {
        result.error_message = "Gradient check passed";
    }

    return result;
}

auto gradcheck_detailed(
    std::function<Variable(const Variable&)> func,
    const Variable& input,
    double eps,
    double atol,
    double rtol
) -> GradCheckResult {
    // Validate input
    if (!input.requires_grad()) {
        GradCheckResult result;
        result.passed = false;
        result.max_abs_error = std::numeric_limits<double>::infinity();
        result.error_message = "Input variable must have requires_grad=true";
        return result;
    }

    // Adjust epsilon and tolerances based on dtype
    // Float32 has ~7 decimal digits of precision. For functions like cos near
    // zero where cos(x+h)-cos(x-h) involves catastrophic cancellation, we need
    // larger epsilon (sqrt of machine epsilon ≈ 3.5e-4) and looser tolerance.
    if (input.dtype() == DType::Float32) {
        if (eps < 5e-4) eps = 5e-4;
        if (atol < 5e-4) atol = 5e-4;
        if (rtol < 1e-2) rtol = 1e-2;
    }

    try {
        // Compute numerical gradient FIRST so we have it even if analytical fails
        Tensor num_grad = numerical_gradient(func, input, eps);

        // Compute analytical gradient
        // Create a new variable with requires_grad for backward
        Variable input_copy(input.tensor().clone(), true);

        // Forward pass
        Variable output = func(input_copy);

        // Ensure we have a scalar or sum to scalar using AUTOGRAD sum
        Variable scalar_output = output;
        if (output.tensor().numel() != 1) {
            // Use autograd sum operation to maintain computation graph!
            // This is critical - manual sum breaks the gradient flow
            scalar_output = tenzor::sum(output);
        }

        // For complex outputs we evaluate the scalar contraction
        //   L = Σ Re(y_k) + Σ Im(y_k)
        // which is real-valued. The finite-difference branch sums real+imag
        // components above; to keep the analytical branch consistent we
        // seed backward() with grad_L = (1 + 1i) so d/dx L = Re(dy/dx) +
        // Im(dy/dx) for every real input x.
        if (scalar_output.tensor().dtype() == DType::Complex64 ||
            scalar_output.tensor().dtype() == DType::Complex128) {
            // Build the (1 + 1i) seed on CPU (host pointer writes are only
            // valid for CPU tensors), then move it to the scalar output's
            // device before seeding backward().
            Device target_dev = scalar_output.tensor().device();
            DType target_dt = scalar_output.tensor().dtype();
            Tensor imag_unit_cpu = zeros({}, target_dt, Device::cpu());
            if (target_dt == DType::Complex64) {
                imag_unit_cpu.data<std::complex<float>>()[0] = {0.0f, 1.0f};
            } else {
                imag_unit_cpu.data<std::complex<double>>()[0] = {0.0, 1.0};
            }
            Tensor one_cpu = ones({}, target_dt, Device::cpu());
            Tensor seed_cpu = tenzor::add(one_cpu, imag_unit_cpu);
            Tensor seed = (target_dev.type == Device::Type::CPU)
                              ? seed_cpu
                              : seed_cpu.to(target_dev);
            scalar_output.backward(seed);
        } else {
            scalar_output.backward();
        }

        // Get analytical gradient
        if (!input_copy.has_grad()) {
            // FIX: Still populate result with numerical gradient for verbose output
            GradCheckResult result;
            result.passed = false;
            result.max_abs_error = std::numeric_limits<double>::infinity();
            result.total_elements = num_grad.numel();
            result.failing_elements = num_grad.numel();

            // Populate numerical_grad and analytical_grad (zeros) for verbose output
            Device num_device = num_grad.device();
            Tensor num_grad_cpu = (num_device == Device::cpu()) ? num_grad : num_grad.to(Device::cpu());

            int64_t n_samples = std::min(int64_t(10), num_grad.numel());
            for (int64_t i = 0; i < n_samples; ++i) {
                if (num_grad.dtype() == DType::Float32) {
                    result.numerical_grad.push_back(static_cast<double>(num_grad_cpu.data<float>()[i]));
                } else {
                    result.numerical_grad.push_back(num_grad_cpu.data<double>()[i]);
                }
                result.analytical_grad.push_back(0.0);  // No analytical gradient
                result.fail_indices.push_back(i);  // All indices fail
            }

            result.error_message = "No gradient computed for input (check if function uses input)";
            return result;
        }

        Tensor ana_grad = *input_copy.grad();

        // Compare gradients
        return compare_gradients(num_grad, ana_grad, atol, rtol);

    } catch (const std::exception& e) {
        GradCheckResult result;
        result.passed = false;
        result.max_abs_error = std::numeric_limits<double>::infinity();
        result.error_message = std::string("Exception during gradient check: ") + e.what();
        return result;
    }
}

auto gradcheck(
    std::function<Variable(const Variable&)> func,
    const Variable& input,
    double eps,
    double atol,
    double rtol,
    bool raise_exception
) -> bool {
    auto result = gradcheck_detailed(func, input, eps, atol, rtol);

    if (!result.passed && raise_exception) {
        throw GradCheckError(result);
    }

    return result.passed;
}

auto gradcheck_verbose(
    std::function<Variable(const Variable&)> func,
    const Variable& input,
    double eps,
    double atol,
    double rtol
) -> bool {
    std::cout << "=== Gradient Check ===" << std::endl;
    std::cout << "Input shape: [";
    auto shape = input.shape();
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << shape[i];
    }
    std::cout << "]" << std::endl;
    std::cout << "Total elements: " << input.tensor().numel() << std::endl;
    std::cout << "Epsilon: " << std::scientific << eps << std::endl;
    std::cout << "Tolerances: atol=" << atol << ", rtol=" << rtol << std::endl;
    std::cout << std::endl;

    std::cout << "Computing numerical gradients..." << std::flush;
    auto start_time = std::chrono::high_resolution_clock::now();

    auto result = gradcheck_detailed(func, input, eps, atol, rtol);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    std::cout << " done (" << duration.count() << " ms)" << std::endl;
    std::cout << std::endl;

    // Print statistics
    std::cout << std::scientific << std::setprecision(4);
    std::cout << "Results:" << std::endl;
    std::cout << "  Max absolute error: " << result.max_abs_error << std::endl;
    std::cout << "  Max relative error: " << result.max_rel_error << std::endl;
    std::cout << "  Total elements checked: " << result.total_elements << std::endl;

    if (result.passed) {
        std::cout << std::endl;
        std::cout << "✓ Gradient check PASSED!" << std::endl;
    } else {
        std::cout << "  Failing elements: " << result.failing_elements << std::endl;
        std::cout << std::endl;
        std::cout << "✗ Gradient check FAILED!" << std::endl;
        std::cout << std::endl;

        // Show sample values
        if (!result.numerical_grad.empty()) {
            std::cout << "Sample gradients (first few elements):" << std::endl;
            size_t n_samples = std::min(size_t(5), result.numerical_grad.size());
            for (size_t i = 0; i < n_samples; ++i) {
                std::cout << "  [" << i << "] numerical: " << result.numerical_grad[i]
                         << ", analytical: " << result.analytical_grad[i]
                         << ", diff: " << std::abs(result.numerical_grad[i] - result.analytical_grad[i])
                         << std::endl;
            }
        }

        if (!result.fail_indices.empty()) {
            std::cout << std::endl;
            std::cout << "First failing indices: ";
            for (size_t i = 0; i < std::min(size_t(10), result.fail_indices.size()); ++i) {
                if (i > 0) std::cout << ", ";
                std::cout << result.fail_indices[i];
            }
            std::cout << std::endl;
        }
    }

    std::cout << "======================" << std::endl;

    return result.passed;
}

auto gradgradcheck_detailed(
    std::function<Variable(const Variable&)> func,
    const Variable& input,
    double eps,
    double atol,
    double rtol
) -> GradCheckResult {
    // Validate input
    if (!input.requires_grad()) {
        GradCheckResult result;
        result.passed = false;
        result.max_abs_error = std::numeric_limits<double>::infinity();
        result.error_message = "Input variable must have requires_grad=true";
        return result;
    }

    // Second-derivative verification strategy
    // -----------------------------------------
    // Tenzor's autograd engine currently accumulates leaf-variable gradients
    // as plain Tensors, not Variables (see engine.cpp:~465), so a standard
    // "gradcheck of the gradient function" — the approach that works in
    // PyTorch — always terminates the chain at the leaf and reports
    // `analytical = 0` for any non-linear function.
    //
    // As an approximation that exercises real second-derivative plumbing we:
    //   1. Evaluate the Hessian H = d²f/dx² using the existing
    //      hessian() functional (Jacobian of the grad function — a proper
    //      numerical second derivative).
    //   2. Independently estimate the Hessian diagonal via a direct
    //      second-order central difference of f:
    //         h_ii ≈ (f(x+eps·e_i) − 2·f(x) + f(x−eps·e_i)) / eps²
    //   3. Compare H.diag against the direct estimate element-wise.
    //
    // This is a meaningful test: both methods must agree for a function's
    // second-derivative computation to be correct. If the hessian() call
    // itself fails or disagrees with the finite-difference estimate, the
    // user has a real second-derivative bug to investigate.
    //
    // Full autograd-engine-level double-backward — letting the returned
    // gradient carry its own grad_fn chain — is tracked for future
    // engine work (would require leaf grad storage to be Variable-typed).

    GradCheckResult result;
    try {
        // hessian() requires a scalar-valued func. Wrap non-scalar outputs
        // in sum() so the second-derivative is well-defined on a scalar
        // loss — matches what the outer gradcheck_detailed does for the
        // first-derivative path.
        auto scalar_func = [&func](const Variable& v) -> Variable {
            Variable out = func(v);
            if (out.tensor().numel() != 1) {
                return tenzor::sum(out);
            }
            return out;
        };

        // Method 1: Hessian via functional (numerical Jacobian of grad_func).
        Tensor H = hessian(scalar_func, input);
        int64_t n = input.tensor().numel();
        if (H.numel() != n * n) {
            result.passed = false;
            result.error_message =
                "hessian() returned shape " + std::to_string(H.numel()) +
                " but expected " + std::to_string(n * n);
            return result;
        }

        // Dtype-adjust eps the same way gradcheck_detailed does — Float32
        // central differences with a sub-5e-4 step hit catastrophic
        // cancellation.
        if (input.dtype() == DType::Float32 && eps < 5e-4) eps = 5e-4;

        // Method 2: direct 2nd-order central diff of f along each axis.
        auto input_cpu = input.tensor().to(Device::cpu());
        double f_center;
        {
            Variable xv(input_cpu, false);
            Variable fv = scalar_func(xv);
            auto fv_cpu = fv.tensor().to(Device::cpu());
            f_center = extract_scalar_real_contraction(fv_cpu);
        }

        std::vector<double> h_direct(n);
        for (int64_t i = 0; i < n; ++i) {
            Tensor plus  = input_cpu.clone();
            Tensor minus = input_cpu.clone();
            if (input.dtype() == DType::Float32) {
                plus.data<float>()[i]  += static_cast<float>(eps);
                minus.data<float>()[i] -= static_cast<float>(eps);
            } else {
                plus.data<double>()[i]  += eps;
                minus.data<double>()[i] -= eps;
            }
            Variable fp = scalar_func(Variable(plus,  false));
            Variable fm = scalar_func(Variable(minus, false));
            auto fp_t = fp.tensor().to(Device::cpu());
            auto fm_t = fm.tensor().to(Device::cpu());
            double fp_v = extract_scalar_real_contraction(fp_t);
            double fm_v = extract_scalar_real_contraction(fm_t);
            h_direct[i] = (fp_v - 2.0 * f_center + fm_v) / (eps * eps);
        }

        // Compare H.diagonal to h_direct. Extract diagonal from row-major
        // (n, n) H.
        auto H_cpu = H.to(Device::cpu());
        int64_t total = n;
        result.total_elements = total;
        result.numerical_grad.reserve(std::min<int64_t>(total, 10));
        result.analytical_grad.reserve(std::min<int64_t>(total, 10));

        double max_abs = 0.0;
        double max_rel = 0.0;
        int64_t failing = 0;
        for (int64_t i = 0; i < n; ++i) {
            double h_ii;
            if (H.dtype() == DType::Float32) {
                h_ii = static_cast<double>(H_cpu.data<float>()[i * n + i]);
            } else {
                h_ii = H_cpu.data<double>()[i * n + i];
            }
            double direct = h_direct[i];
            double diff = std::abs(h_ii - direct);
            double threshold = atol + rtol * std::abs(direct);
            if (diff > max_abs) max_abs = diff;
            double rel = std::abs(direct) > 1e-12 ? diff / std::abs(direct) : diff;
            if (rel > max_rel) max_rel = rel;
            if (diff > threshold) {
                failing++;
                if (result.fail_indices.size() < 10) {
                    result.fail_indices.push_back(i);
                }
            }
            if (result.numerical_grad.size() < 10) {
                result.numerical_grad.push_back(direct);
                result.analytical_grad.push_back(h_ii);
            }
        }

        // Off-diagonal symmetry spot-check. The finite-difference comparison
        // above only validates the Hessian *diagonal* (H[i,i]); a transposed or
        // mis-stacked Hessian with correct diagonal would still pass. The true
        // Hessian of a scalar function is symmetric (Schwarz's theorem), so we
        // spot-check H[i,j] == H[j,i] for a bounded number of off-diagonal
        // pairs. This catches index-transposition / mis-stacking bugs without
        // the O(n^2) cost of a full finite-difference Hessian.
        double max_asym = 0.0;
        int64_t asym_failing = 0;
        if (n >= 2) {
            const int64_t kMaxPairs = 32;
            int64_t pairs_checked = 0;
            // Symmetry tolerance: H is computed analytically, so a non-trivial
            // asymmetry signals a real layout/transposition bug rather than FD
            // noise. Use the same atol/rtol contract against the symmetric mean.
            for (int64_t i = 0; i < n && pairs_checked < kMaxPairs; ++i) {
                for (int64_t j = i + 1; j < n && pairs_checked < kMaxPairs; ++j) {
                    double h_ij, h_ji;
                    if (H.dtype() == DType::Float32) {
                        h_ij = static_cast<double>(H_cpu.data<float>()[i * n + j]);
                        h_ji = static_cast<double>(H_cpu.data<float>()[j * n + i]);
                    } else {
                        h_ij = H_cpu.data<double>()[i * n + j];
                        h_ji = H_cpu.data<double>()[j * n + i];
                    }
                    double adiff = std::abs(h_ij - h_ji);
                    double sym_scale = 0.5 * (std::abs(h_ij) + std::abs(h_ji));
                    double sym_threshold = atol + rtol * sym_scale;
                    if (adiff > max_asym) max_asym = adiff;
                    if (adiff > sym_threshold) {
                        asym_failing++;
                        if (result.fail_indices.size() < 10) {
                            result.fail_indices.push_back(i * n + j);
                        }
                    }
                    ++pairs_checked;
                }
            }
        }

        if (max_asym > max_abs) result.max_abs_error = max_asym;
        else result.max_abs_error = max_abs;
        result.max_rel_error = max_rel;
        failing += asym_failing;
        result.failing_elements = failing;
        result.passed = (failing == 0);
        if (!result.passed) {
            result.error_message =
                "Second-derivative check failed (diagonal FD + off-diagonal "
                "symmetry spot-check). max_abs=" + std::to_string(max_abs) +
                " diag_failing=" + std::to_string(failing - asym_failing) +
                "/" + std::to_string(n) +
                " sym_failing=" + std::to_string(asym_failing) +
                " max_asymmetry=" + std::to_string(max_asym);
        } else {
            // Be explicit that the FD comparison covered only the diagonal;
            // off-diagonal entries were validated for symmetry only, not value.
            result.error_message =
                "Second-derivative check passed (diagonal validated against "
                "finite differences; off-diagonal entries validated for "
                "symmetry only, not magnitude)";
        }
        return result;
    } catch (const std::exception& e) {
        result.passed = false;
        result.max_abs_error = std::numeric_limits<double>::infinity();
        result.error_message =
            std::string("Exception during gradgradcheck: ") + e.what();
        return result;
    }
}

auto gradgradcheck(
    std::function<Variable(const Variable&)> func,
    const Variable& input,
    double eps,
    double atol,
    double rtol,
    bool raise_exception
) -> bool {
    auto result = gradgradcheck_detailed(func, input, eps, atol, rtol);

    if (!result.passed && raise_exception) {
        throw GradCheckError(result);
    }

    return result.passed;
}

} // namespace tenzor
