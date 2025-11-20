#include <iostream>
#include <gtest/gtest.h>
#include "tenzor/tenzor.hpp"
#include "../include/tenzor/models/swin_transformer.hpp"

using namespace tenzor;
using namespace tenzor::models;

void debug_tensor(const std::string& name, const Tensor& t, bool check_nonzero = true) {
    auto t_cpu = t.to(Device::cpu()).to(DType::Float32);
    auto* data = t_cpu.data<float>();

    std::cout << name << " shape: [";
    for (size_t i = 0; i < t.shape().size(); i++) {
        if (i > 0) std::cout << ", ";
        std::cout << t.shape()[i];
    }
    std::cout << "], dtype=" << static_cast<int>(t.dtype());

    if (check_nonzero && t_cpu.numel() > 0) {
        float sum = 0.0f;
        float max_abs = 0.0f;
        int nonzero_count = 0;
        for (int i = 0; i < std::min(1000, static_cast<int>(t_cpu.numel())); i++) {
            sum += std::abs(data[i]);
            max_abs = std::max(max_abs, std::abs(data[i]));
            if (std::abs(data[i]) > 1e-10f) nonzero_count++;
        }
        std::cout << ", avg_abs=" << (sum / std::min(1000, static_cast<int>(t_cpu.numel())))
                  << ", max_abs=" << max_abs
                  << ", nonzero_count=" << nonzero_count << "/" << std::min(1000, static_cast<int>(t_cpu.numel()));
    }
    std::cout << "\n";
}

void debug_variable(const std::string& name, const Variable& v) {
    std::cout << "\n=== " << name << " ===\n";
    debug_tensor("  Tensor", v.tensor());
    if (v.has_grad()) {
        debug_tensor("  Gradient", v.grad().value());
    } else {
        std::cout << "  Gradient: None\n";
    }
}

TEST(MinimalSwinDebug, CompareFloat32VsFloat16) {
    std::cout << "\n========================================\n";
    std::cout << "MINIMAL SWIN TRANSFORMER DEBUG TEST\n";
    std::cout << "========================================\n";

    initialize();
    auto device = Device::cpu();
    const int img_size = 224;

    // Test Float32
    {
        std::cout << "\n\n######## FLOAT32 TEST ########\n";
        auto model = swin_tiny(10, img_size, false);
        model->to(DType::Float32);
        model->train();

        Variable input(Tensor({1, 3, img_size, img_size}, DType::Float32, device), true);
        debug_variable("Input", input);

        std::cout << "\n--- Forward Pass ---\n";
        Variable output = model->forward(input);
        debug_variable("Output", output);

        std::cout << "\n--- Computing Loss ---\n";
        Variable loss = tenzor::mean(output);
        debug_variable("Loss", loss);

        std::cout << "\n--- Backward Pass ---\n";
        loss.backward();

        debug_variable("Input After Backward", input);

        // Check parameters
        auto params = model->parameters();
        std::cout << "\n--- Parameters (first 3) ---\n";
        for (size_t i = 0; i < std::min(size_t(3), params.size()); i++) {
            debug_variable("Param " + std::to_string(i), *params[i]);
        }
    }

    // Test Float16
    {
        std::cout << "\n\n######## FLOAT16 TEST ########\n";
        auto model = swin_tiny(10, img_size, false);
        model->to(DType::Float16);
        model->train();

        Variable input(Tensor({1, 3, img_size, img_size}, DType::Float16, device), true);
        debug_variable("Input", input);

        std::cout << "\n--- Forward Pass ---\n";
        Variable output = model->forward(input);
        debug_variable("Output", output);

        std::cout << "\n--- Computing Loss ---\n";
        Variable loss = tenzor::mean(output);
        debug_variable("Loss", loss);

        std::cout << "\n--- Backward Pass ---\n";
        loss.backward();

        debug_variable("Input After Backward", input);

        // Check parameters
        auto params = model->parameters();
        std::cout << "\n--- Parameters (first 3) ---\n";
        for (size_t i = 0; i < std::min(size_t(3), params.size()); i++) {
            debug_variable("Param " + std::to_string(i), *params[i]);
        }
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
