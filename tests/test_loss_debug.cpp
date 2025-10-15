#include <tenzor/tenzor.hpp>
#include <tenzor/nn/loss/losses.hpp>
#include <iostream>

using namespace tenzor;
using namespace tenzor::nn;

int main() {
    tenzor::initialize();

    std::cout << "Test 1: Simple multiplication with scalar Variable" << std::endl;
    try {
        auto x = Variable(full({2, 3}, 2.0f, DType::Float32), true);  // requires_grad=true
        auto scalar_tensor = full({2, 3}, 0.5f, DType::Float32);
        auto scalar_var = Variable(scalar_tensor, false);  // requires_grad=false

        auto result = x * scalar_var;
        std::cout << "Forward passed" << std::endl;

        result.backward(ones({2, 3}, DType::Float32));
        std::cout << "Backward passed" << std::endl;

        if (x.grad().has_value()) {
            std::cout << "Gradient computed successfully" << std::endl;
        } else {
            std::cout << "ERROR: No gradient!" << std::endl;
        }
    } catch (const std::exception& e) {
        std::cout << "ERROR: " << e.what() << std::endl;
    }

    std::cout << "\nTest 2: KLDivLoss minimal (without clamp)" << std::endl;
    try {
        auto input = Variable(full({2, 3}, -1.0f, DType::Float32), true);
        auto target = Variable(full({2, 3}, 0.5f, DType::Float32), false);

        // Minimal KLDiv computation WITHOUT clamp
        auto log_target = log(target);
        auto diff = log_target - input;
        auto loss_unreduced = target * diff;
        auto loss = mean(loss_unreduced);

        std::cout << "Forward passed, loss = " << loss.tensor().item<float>() << std::endl;

        loss.backward();
        std::cout << "Backward passed" << std::endl;

        if (input.grad().has_value()) {
            std::cout << "Gradient computed successfully" << std::endl;
        } else {
            std::cout << "ERROR: No gradient!" << std::endl;
        }
    } catch (const std::exception& e) {
        std::cout << "ERROR: " << e.what() << std::endl;
    }

    std::cout << "\nTest 2b: KLDivLoss minimal (WITH clamp)" << std::endl;
    try {
        auto input = Variable(full({2, 3}, -1.0f, DType::Float32), true);
        auto target = Variable(full({2, 3}, 0.5f, DType::Float32), false);

        // Minimal KLDiv computation WITH clamp - matching KLDivLoss exactly
        auto target_clamped = clamp(target, 1e-7f, 1.0f);
        auto log_target = log(target_clamped);
        auto diff = log_target - input;
        auto loss_unreduced = target * diff;
        auto loss = mean(loss_unreduced);

        std::cout << "Forward passed, loss = " << loss.tensor().item<float>() << std::endl;

        loss.backward();
        std::cout << "Backward passed" << std::endl;

        if (input.grad().has_value()) {
            std::cout << "Gradient computed successfully" << std::endl;
        } else {
            std::cout << "ERROR: No gradient!" << std::endl;
        }
    } catch (const std::exception& e) {
        std::cout << "ERROR: " << e.what() << std::endl;
    }

    std::cout << "\nTest 3: Actual KLDivLoss class" << std::endl;
    try {
        auto input = Variable(full({2, 3}, -1.0f, DType::Float32), true);
        auto target = Variable(full({2, 3}, 0.5f, DType::Float32), false);

        auto criterion = KLDivLoss("mean");
        auto loss = criterion(input, target);

        std::cout << "Forward passed, loss = " << loss.tensor().item<float>() << std::endl;

        loss.backward();
        std::cout << "Backward passed" << std::endl;

        if (input.grad().has_value()) {
            std::cout << "Gradient computed successfully" << std::endl;
        } else {
            std::cout << "ERROR: No gradient!" << std::endl;
        }
    } catch (const std::exception& e) {
        std::cout << "ERROR: " << e.what() << std::endl;
    }

    return 0;
}
