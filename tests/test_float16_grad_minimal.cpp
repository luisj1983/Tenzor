#include <iostream>
#include <tenzor/tenzor.hpp>

using namespace tenzor;

int main() {
    tenzor::initialize();

    // Simple test: x -> linear -> mean -> backward
    auto device = Device::cpu();
    auto dtype = DType::Float16;

    // Create input with gradient enabled
    Variable x(Tensor({2, 3}, dtype, device), true);

    // Fill with some values
    {
        auto* data = x.tensor().data<Float16>();
        for (int i = 0; i < 6; i++) {
            data[i] = Float16(1.0f + i * 0.1f);
        }
    }

    // Create a simple weight matrix
    Variable w(Tensor({4, 3}, dtype, device), true);
    {
        auto* data = w.tensor().data<Float16>();
        for (int i = 0; i < 12; i++) {
            data[i] = Float16(0.5f);
        }
    }

    std::cout << "Input x:" << std::endl;
    auto x_cpu = x.tensor().to(Device::cpu()).to(DType::Float32);
    auto* x_data = x_cpu.data<float>();
    for (int i = 0; i < 6; i++) {
        std::cout << x_data[i] << " ";
    }
    std::cout << std::endl;

    // y = x @ w.T
    auto w_t = autograd::permute(w, {1, 0});
    auto y = autograd::matmul(x, w_t);

    std::cout << "Output y shape: [";
    for (auto d : y.shape()) {
        std::cout << d << " ";
    }
    std::cout << "]" << std::endl;

    // loss = mean(y)
    auto loss = autograd::mean(y);

    std::cout << "Loss: " << static_cast<float>(loss.tensor().data<Float16>()[0]) << std::endl;

    // Backward
    loss.backward();

    std::cout << "Checking gradients..." << std::endl;

    // Check x gradient
    if (x.grad().has_value()) {
        std::cout << "x.grad() exists" << std::endl;
        auto x_grad_cpu = x.grad()->to(Device::cpu()).to(DType::Float32);
        auto* x_grad_data = x_grad_cpu.data<float>();
        std::cout << "x.grad(): ";
        for (int i = 0; i < 6; i++) {
            std::cout << x_grad_data[i] << " ";
        }
        std::cout << std::endl;

        bool has_nonzero = false;
        for (int i = 0; i < 6; i++) {
            if (std::abs(x_grad_data[i]) > 1e-6f) {
                has_nonzero = true;
                break;
            }
        }
        std::cout << "Has non-zero gradient: " << (has_nonzero ? "YES" : "NO") << std::endl;
    } else {
        std::cout << "x.grad() is NONE!" << std::endl;
    }

    // Check w gradient
    if (w.grad().has_value()) {
        std::cout << "w.grad() exists" << std::endl;
        auto w_grad_cpu = w.grad()->to(Device::cpu()).to(DType::Float32);
        auto* w_grad_data = w_grad_cpu.data<float>();
        std::cout << "w.grad() first 4: ";
        for (int i = 0; i < std::min(4, 12); i++) {
            std::cout << w_grad_data[i] << " ";
        }
        std::cout << std::endl;
    } else {
        std::cout << "w.grad() is NONE!" << std::endl;
    }

    return 0;
}
