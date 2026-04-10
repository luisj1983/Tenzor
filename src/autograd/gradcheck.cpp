/**
 * @file gradcheck.cpp
 * @brief Implementation of numerical gradient checking
 */

#include "tenzor/autograd/gradcheck.hpp"
#include "tenzor/autograd/variable.hpp"
#include "tenzor/autograd/ops.hpp"
#include "tenzor/core/tensor.hpp"
#include "tenzor/ops/creation.hpp"
#include <cmath>
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

        if (f_plus.tensor().numel() == 1) {
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

            // Transfer to CPU for pointer-based computation
            Device sum_device = sum_plus_tensor.device();
            Tensor sum_plus_cpu = (sum_device == Device::cpu()) ? sum_plus_tensor : sum_plus_tensor.to(Device::cpu());
            Tensor sum_minus_cpu = (sum_device == Device::cpu()) ? sum_minus_tensor : sum_minus_tensor.to(Device::cpu());

            if (f_plus.dtype() == DType::Float32) {
                const float* data_p = sum_plus_cpu.data<float>();
                const float* data_m = sum_minus_cpu.data<float>();
                for (int64_t j = 0; j < n; ++j) {
                    total_plus += data_p[j];
                    total_minus += data_m[j];
                }
            } else if (f_plus.dtype() == DType::Float64) {
                const double* data_p = sum_plus_cpu.data<double>();
                const double* data_m = sum_minus_cpu.data<double>();
                for (int64_t j = 0; j < n; ++j) {
                    total_plus += data_p[j];
                    total_minus += data_m[j];
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

        // Backward pass
        scalar_output.backward();

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

} // namespace tenzor
