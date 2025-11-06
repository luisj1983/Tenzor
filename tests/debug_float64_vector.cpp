#include <tenzor/tenzor.hpp>
#include <tenzor/autograd/gradcheck.hpp>
#include <iostream>
#include <iomanip>

using namespace tenzor;

int main() {
    tenzor::initialize();

    auto device = Device::cpu();

    // Test exactly like the failing test
    auto f = [](const Variable& x) -> Variable {
        return x * x + x * 2.0;
    };

    Tensor data = zeros({4}, DType::Float64, device);
    auto data_cpu = data.to(Device::cpu());
    double* ptr = data_cpu.data<double>();
    for (int i = 0; i < 4; ++i) {
        ptr[i] = static_cast<double>(i + 1) * 0.1;
    }
    data = data_cpu.to(device);

    std::cout << "Input x = [";
    for (int i = 0; i < 4; ++i) {
        std::cout << ptr[i];
        if (i < 3) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    Variable x(data, true);

    // Call the function
    Variable y = f(x);

    std::cout << "\nOutput y = ";
    auto y_cpu = y.tensor().to(Device::cpu());
    const double* y_ptr = y_cpu.data<double>();
    std::cout << "[";
    for (int i = 0; i < 4; ++i) {
        std::cout << y_ptr[i];
        if (i < 3) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    // For gradcheck, we need to sum the output
    std::cout << "\nCalling backward (for gradient check, would sum y first)..." << std::endl;

    // Sum y first (like gradcheck does)
    double sum_y = 0.0;
    for (int i = 0; i < 4; ++i) {
        sum_y += y_ptr[i];
    }
    std::cout << "sum(y) = " << sum_y << std::endl;

    // Now backward on a scalar sum
    Tensor sum_tensor({}, DType::Float64, device);
    sum_tensor.data<double>()[0] = sum_y;
    Variable sum_var(sum_tensor, true);

    // Actually, let's just use backward on y directly and check gradients
    y.backward();

    if (x.has_grad()) {
        std::cout << "\nAnalytical gradient (element-wise backward): " << std::endl;
        auto grad_cpu = x.grad().value().to(Device::cpu());
        const double* grad_ptr = grad_cpu.data<double>();
        std::cout << "[";
        for (int i = 0; i < 4; ++i) {
            std::cout << std::setprecision(15) << grad_ptr[i];
            if (i < 3) std::cout << ", ";
        }
        std::cout << "]" << std::endl;

        std::cout << "\nExpected gradient (2*x + 2): " << std::endl;
        std::cout << "[";
        for (int i = 0; i < 4; ++i) {
            double expected = 2.0 * ptr[i] + 2.0;
            std::cout << expected;
            if (i < 3) std::cout << ", ";
        }
        std::cout << "]" << std::endl;

        std::cout << "\nGradient errors: " << std::endl;
        std::cout << "[";
        for (int i = 0; i < 4; ++i) {
            double expected = 2.0 * ptr[i] + 2.0;
            double error = std::abs(grad_ptr[i] - expected);
            std::cout << error;
            if (i < 3) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    } else {
        std::cout << "ERROR: No gradient computed!" << std::endl;
    }

    // Now test with gradcheck
    std::cout << "\n========================================" << std::endl;
    std::cout << "Running gradcheck_detailed..." << std::endl;
    auto result = gradcheck_detailed(f, x, 1e-6, 1e-5, 1e-3);

    std::cout << "Gradcheck passed: " << (result.passed ? "YES" : "NO") << std::endl;
    std::cout << "Max absolute error: " << result.max_abs_error << std::endl;
    std::cout << "Max relative error: " << result.max_rel_error << std::endl;
    if (!result.error_message.empty()) {
        std::cout << "Error message: " << result.error_message << std::endl;
    }

    return 0;
}
