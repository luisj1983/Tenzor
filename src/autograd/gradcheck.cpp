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
        } else if (tensor.dtype() == DType::Float16 || tensor.dtype() == DType::BFloat16) {
            // Reduced-precision scalar outputs have no native item<T>() path
            // here. Widen to Float32 first (widen-narrow pattern, mirroring the
            // summation branch in numerical_gradient) so a function returning a
            // single Float16/BFloat16 element produces a numerical gradient
            // instead of throwing.
            return static_cast<double>(tensor.to(DType::Float32).item<float>());
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
    // Central finite-difference gradient. The perturbation is applied at the
    // input's NATIVE dtype (so any tensors `func` captures at that dtype are
    // not dtype-mismatched against the probe), but the perturbed values are
    // computed and the difference is reduced in DOUBLE precision to minimise
    // the catastrophic cancellation in (f(x+eps) − f(x−eps)).
    //
    // Float16 / BFloat16 inputs were previously rejected outright. They are now
    // supported: the perturbation is staged in Float64 and cast to the native
    // half dtype on write (with a dtype-appropriate eps floor that is actually
    // representable), so reduced-precision backward paths can be gradchecked.
    const DType input_dtype = input.dtype();
    const bool is_f32   = (input_dtype == DType::Float32);
    const bool is_f64   = (input_dtype == DType::Float64);
    const bool is_f16   = (input_dtype == DType::Float16);
    const bool is_bf16  = (input_dtype == DType::BFloat16);
    if (!is_f32 && !is_f64 && !is_f16 && !is_bf16) {
        throw std::runtime_error(
            "gradcheck numerical_gradient supports real floating dtypes "
            "(Float64/Float32/Float16/BFloat16)");
    }

    // Dtype-appropriate minimum step so the perturbation is representable and
    // does not vanish to catastrophic cancellation:
    //   Float64: tiny step is fine.
    //   Float32: ~sqrt(eps_machine) ≈ 5e-4.
    //   Float16/BFloat16: only ~3 decimal digits, so a much larger step.
    if (is_f32 && eps < 5e-4) eps = 5e-4;
    if ((is_f16 || is_bf16) && eps < 1e-2) eps = 1e-2;
    if (eps <= 0.0) eps = 1e-6;

    const Device original_device = input.tensor().device();

    // Float64 staging copy on CPU for exact pointer-level perturbation; the
    // probe handed to `func` is cast back to the input's native dtype so its
    // captured operands match.
    Tensor input_f64 = input.tensor();
    if (input_f64.dtype() != DType::Float64) input_f64 = input_f64.to(DType::Float64);
    if (input_f64.device() != Device::cpu()) input_f64 = input_f64.to(Device::cpu());
    if (!input_f64.is_contiguous()) input_f64 = input_f64.contiguous();

    const int64_t total_elements = input_f64.numel();

    Tensor grad_f64 = zeros(
        std::vector<int64_t>(input_f64.shape().begin(), input_f64.shape().end()),
        DType::Float64, Device::cpu());
    double* grad_data = grad_f64.data<double>();

    // The probe handed to `func` is at the input's NATIVE dtype: `func` may
    // capture other operands at that dtype (logaddexp(a, ·), cosine_similarity
    // (a, ·), …) and reject a widened probe, and some ops internally fix their
    // working precision (STFT/ISTFT use Complex64 regardless of the input
    // dtype) so a Float64 probe would not actually improve — and would even
    // mislead — the finite-difference reference. The perturbation is still
    // STAGED and the difference REDUCED in Float64 to cut cancellation; only
    // the value seen by `func` is cast back to the native dtype.
    auto make_probe = [&](const Tensor& staged_cpu) -> Variable {
        Tensor t = (staged_cpu.dtype() != input_dtype) ? staged_cpu.to(input_dtype)
                                                        : staged_cpu;
        if (original_device != Device::cpu()) t = t.to(original_device);
        return Variable(t, false);
    };

    auto reduce_to_scalar = [](const Variable& f) -> double {
        if (f.tensor().numel() == 1 && !f.tensor().is_complex()) {
            return extract_scalar(f.tensor());
        }
        Tensor t = f.tensor();
        if (t.device() != Device::cpu()) t = t.to(Device::cpu());
        if (!t.is_contiguous()) t = t.contiguous();
        const int64_t n = t.numel();
        double total = 0.0;
        if (t.is_complex()) {
            // Real-input -> complex-output: reduce as Re(z)+Im(z), matching the
            // (1+1i) backward seed used by the analytical complex path.
            if (t.dtype() == DType::Complex64) {
                const auto* d = t.data<std::complex<float>>();
                for (int64_t j = 0; j < n; ++j)
                    total += static_cast<double>(d[j].real()) +
                             static_cast<double>(d[j].imag());
            } else {
                const auto* d = t.data<std::complex<double>>();
                for (int64_t j = 0; j < n; ++j) total += d[j].real() + d[j].imag();
            }
            return total;
        }
        // Reduced-precision outputs widen to Float64 for an accurate sum.
        if (t.dtype() != DType::Float64) t = t.to(DType::Float64);
        const double* d = t.data<double>();
        for (int64_t j = 0; j < n; ++j) total += d[j];
        return total;
    };

    for (int64_t i = 0; i < total_elements; ++i) {
        Tensor x_plus  = input_f64.clone();
        Tensor x_minus = input_f64.clone();
        x_plus.data<double>()[i]  += eps;
        x_minus.data<double>()[i] -= eps;

        double val_plus  = reduce_to_scalar(func(make_probe(x_plus)));
        double val_minus = reduce_to_scalar(func(make_probe(x_minus)));
        grad_data[i] = (val_plus - val_minus) / (2.0 * eps);
    }

    // Cast the Float64 shadow back to the input's native dtype/device so the
    // result aligns with the analytical gradient in the comparison — EXCEPT for
    // half dtypes (Float16/BFloat16), where narrowing the Float64 FD reference
    // would truncate ~3 mantissa bits and corrupt the numerical reference.
    // compare_gradients widens both sides to Float64 whenever they are not both
    // Float32, so a Float64 numerical vs half analytical still compares cleanly.
    Tensor result = (is_f64 || is_f16 || is_bf16) ? grad_f64 : grad_f64.to(input_dtype);
    if (original_device != Device::cpu()) result = result.to(original_device);
    return result;
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

    const bool both_float32 = numerical.dtype() == DType::Float32 &&
                              analytical.dtype() == DType::Float32;

    // Transfer to CPU for pointer-based computation
    Device num_device = numerical.device();
    // .contiguous(): data<T>()[i] walks raw storage order, but the analytical
    // gradient from backward() can be a non-contiguous (e.g. transposed) view,
    // which would otherwise be compared in the wrong element order.
    Tensor numerical_cpu =
        ((num_device == Device::cpu()) ? numerical : numerical.to(Device::cpu())).contiguous();
    Tensor analytical_cpu =
        ((num_device == Device::cpu()) ? analytical : analytical.to(Device::cpu())).contiguous();

    // Float16/BFloat16 gradients reach here (numerical_gradient widens then casts
    // back to the input dtype); data<double>() throws on them. Read the all-Float32
    // case directly, otherwise widen both to Float64 so the loop reads uniformly.
    if (!both_float32) {
        if (numerical_cpu.dtype() != DType::Float64) numerical_cpu = numerical_cpu.to(DType::Float64);
        if (analytical_cpu.dtype() != DType::Float64) analytical_cpu = analytical_cpu.to(DType::Float64);
    }

    for (int64_t i = 0; i < total; ++i) {
        double num_val, ana_val;

        if (both_float32) {
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

        // Check tolerance. Scale the relative band by max(|analytical|,
        // |numerical|) rather than |analytical| alone: when the analytical
        // gradient is (wrongly) near zero but the numerical gradient is large,
        // |ana_val|≈0 would collapse the band to atol and hide the error.
        // PyTorch's gradcheck uses the same max() formulation.
        double threshold = atol + rtol * std::max(std::abs(ana_val),
                                                  std::abs(num_val));
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

    // Float32 finite-difference floors. The numerical gradient is reduced in
    // Float64, but a Float32 forward still has ~7-digit outputs, so the central
    // difference carries ~5e-4 ABSOLUTE noise at small-gradient elements and
    // needs a representable step. We therefore keep the eps step floor and the
    // atol (absolute) floor — these are FD-numerical necessities.
    //
    // We DELIBERATELY do NOT touch rtol: the previous code also forced
    // `rtol -> 1e-2`, which let a Float32 analytical backward that was wrong by
    // up to 1% (relative) still pass — the exact bug-masking anti-pattern. The
    // relative tolerance is the lever that hides real backward errors, so it is
    // honoured exactly as the caller passed it; only the absolute FD floors are
    // applied.
    if (input.dtype() == DType::Float32) {
        if (eps < 5e-4) eps = 5e-4;
        if (atol < 5e-4) atol = 5e-4;
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
    // `hessian()` computes each column as a Float64 central-difference
    // Hessian-vector product (accurate to ~1e-7), so gradgradcheck validates
    // that Hessian against an independent finite-difference reference computed
    // here with a different stencil:
    //   1. H = hessian(f)                         (analytic, exact)
    //   2. Diagonal FD reference:
    //         h_ii ≈ (f(x+eps·e_i) − 2·f(x) + f(x−eps·e_i)) / eps²
    //      compared element-wise against H[i,i].
    //   3. Off-diagonal FD reference for small n (n ≤ kFullOffDiagN):
    //         h_ij ≈ (f(x+eps·e_i+eps·e_j) − f(x+eps·e_i−eps·e_j)
    //                − f(x−eps·e_i+eps·e_j) + f(x−eps·e_i−eps·e_j)) / (4 eps²)
    //      compared element-wise against H[i,j] (MAGNITUDE, not just symmetry).
    //      For larger n the O(n²) FD cost is avoided and we fall back to a
    //      bounded symmetry spot-check (H[i,j] == H[j,i]).
    //
    // Both the diagonal and (for small n) the full off-diagonal magnitudes of
    // the analytic Hessian must agree with finite differences, so a
    // backward-of-backward bug that produces wrong-but-symmetric off-diagonal
    // entries is now caught for small inputs.

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

        // Dtype-appropriate eps floors (mirroring numerical_gradient /
        // gradcheck_detailed): a too-small step makes the second-order central
        // difference's eps² denominator catastrophically cancellation-prone for
        // reduced-precision inputs. Float32 ≈ sqrt(eps_machine) ≈ 5e-4; half
        // precision needs ≈ 1e-2.
        const DType gg_dtype = input.tensor().dtype();
        if (gg_dtype == DType::Float32 && eps < 5e-4) eps = 5e-4;
        if ((gg_dtype == DType::Float16 || gg_dtype == DType::BFloat16) && eps < 1e-2) eps = 1e-2;
        if (eps <= 0.0) eps = 1e-4;

        // The FD reference is evaluated in Float64 (perturb a Float64 copy of
        // the input, run scalar_func at Float64) so the second-order central
        // difference — whose eps² denominator is acutely cancellation-prone —
        // is accurate regardless of the input's native dtype. This mirrors the
        // first-order numerical_gradient widening and lets us compare against
        // the analytic Hessian at the caller's tolerances without inflation.
        const Device orig_device = input.tensor().device();
        Tensor input_f64 = input.tensor();
        if (input_f64.dtype() != DType::Float64) input_f64 = input_f64.to(DType::Float64);
        if (input_f64.device() != Device::cpu()) input_f64 = input_f64.to(Device::cpu());
        if (!input_f64.is_contiguous()) input_f64 = input_f64.contiguous();

        // The probe handed to `func` must be at the input's NATIVE dtype: `func`
        // may capture other operands at that dtype (and reject a widened Float64
        // probe), and some ops fix their internal working precision. The
        // perturbation is still STAGED in Float64; only the value seen by `func`
        // is cast back to native dtype. Mirrors numerical_gradient's make_probe.
        auto eval_f64 = [&](const Tensor& probe_cpu) -> double {
            Tensor probe = (probe_cpu.dtype() != gg_dtype)
                               ? probe_cpu.to(gg_dtype) : probe_cpu;
            Variable v(orig_device != Device::cpu() ? probe.to(orig_device)
                                                    : probe, false);
            Variable fv = scalar_func(v);
            Tensor t = fv.tensor();
            if (t.device() != Device::cpu()) t = t.to(Device::cpu());
            return extract_scalar_real_contraction(t);
        };

        double f_center = eval_f64(input_f64);

        // Method 2a: diagonal 2nd-order central difference.
        std::vector<double> h_direct(n);
        for (int64_t i = 0; i < n; ++i) {
            Tensor plus  = input_f64.clone();
            Tensor minus = input_f64.clone();
            plus.data<double>()[i]  += eps;
            minus.data<double>()[i] -= eps;
            double fp_v = eval_f64(plus);
            double fm_v = eval_f64(minus);
            h_direct[i] = (fp_v - 2.0 * f_center + fm_v) / (eps * eps);
        }

        // Compare H.diagonal to h_direct. Extract diagonal from row-major
        // (n, n) H.
        auto H_cpu = H.to(Device::cpu());
        // F014: hessian() narrows H back to the INPUT dtype, so for a
        // Float16/BFloat16 input H is half. The reads below take the
        // `data<double>()` branch for any non-Float32 dtype, which throws on a
        // half tensor and is reported as "Exception during gradgradcheck" — a
        // false failure. Widen a half H to Float64 up front so every non-Float32
        // read is a valid double read (mirrors compare_gradients' handling).
        if (H_cpu.dtype() == DType::Float16 || H_cpu.dtype() == DType::BFloat16) {
            H_cpu = H_cpu.to(DType::Float64);
        }
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

        // Off-diagonal validation.
        //   * Small n: full mixed-partial FD reference, comparing the
        //     MAGNITUDE of every H[i,j] (i<j) against finite differences. This
        //     catches a backward-of-backward bug that produces a wrong but
        //     symmetric off-diagonal entry — the gap the old symmetry-only
        //     spot-check left open.
        //   * Large n: O(n²) FD is too costly, so fall back to a bounded
        //     symmetry spot-check (H[i,j] == H[j,i]).
        constexpr int64_t kFullOffDiagN = 16;
        double max_offdiag_err = 0.0;
        double max_asym = 0.0;
        int64_t offdiag_failing = 0;
        const bool full_offdiag = (n >= 2 && n <= kFullOffDiagN);

        if (full_offdiag) {
            for (int64_t i = 0; i < n; ++i) {
                for (int64_t j = i + 1; j < n; ++j) {
                    // Mixed second partial via the 4-point central stencil.
                    Tensor pp = input_f64.clone();  // +e_i +e_j
                    Tensor pm = input_f64.clone();  // +e_i -e_j
                    Tensor mp = input_f64.clone();  // -e_i +e_j
                    Tensor mm = input_f64.clone();  // -e_i -e_j
                    pp.data<double>()[i] += eps; pp.data<double>()[j] += eps;
                    pm.data<double>()[i] += eps; pm.data<double>()[j] -= eps;
                    mp.data<double>()[i] -= eps; mp.data<double>()[j] += eps;
                    mm.data<double>()[i] -= eps; mm.data<double>()[j] -= eps;
                    double h_ij_fd = (eval_f64(pp) - eval_f64(pm)
                                      - eval_f64(mp) + eval_f64(mm))
                                     / (4.0 * eps * eps);

                    double h_ij = (H.dtype() == DType::Float32)
                        ? static_cast<double>(H_cpu.data<float>()[i * n + j])
                        : H_cpu.data<double>()[i * n + j];

                    double diff = std::abs(h_ij - h_ij_fd);
                    double threshold = atol + rtol * std::max(std::abs(h_ij),
                                                              std::abs(h_ij_fd));
                    if (diff > max_offdiag_err) max_offdiag_err = diff;
                    if (diff > threshold) {
                        offdiag_failing++;
                        if (result.fail_indices.size() < 10) {
                            result.fail_indices.push_back(i * n + j);
                        }
                    }
                }
            }
        } else if (n >= 2) {
            const int64_t kMaxPairs = 32;
            int64_t pairs_checked = 0;
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
                        offdiag_failing++;
                        if (result.fail_indices.size() < 10) {
                            result.fail_indices.push_back(i * n + j);
                        }
                    }
                    ++pairs_checked;
                }
            }
        }

        result.max_abs_error =
            std::max({max_abs, max_asym, max_offdiag_err});
        result.max_rel_error = max_rel;
        failing += offdiag_failing;
        result.failing_elements = failing;
        result.passed = (failing == 0);
        if (!result.passed) {
            result.error_message =
                std::string("Second-derivative check failed (") +
                (full_offdiag ? "diagonal + full off-diagonal FD"
                              : "diagonal FD + off-diagonal symmetry spot-check") +
                "). max_abs=" + std::to_string(max_abs) +
                " diag_failing=" + std::to_string(failing - offdiag_failing) +
                "/" + std::to_string(n) +
                " offdiag_failing=" + std::to_string(offdiag_failing) +
                " max_offdiag_err=" + std::to_string(max_offdiag_err) +
                " max_asymmetry=" + std::to_string(max_asym);
        } else {
            result.error_message = full_offdiag
                ? "Second-derivative check passed (diagonal AND full "
                  "off-diagonal magnitudes validated against finite differences)"
                : "Second-derivative check passed (diagonal validated against "
                  "finite differences; off-diagonal entries validated for "
                  "symmetry only, not magnitude — n too large for full FD)";
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
