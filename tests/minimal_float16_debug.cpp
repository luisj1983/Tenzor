#include <iostream>
#include <gtest/gtest.h>
#include "tenzor/tenzor.hpp"

using namespace tenzor;
using namespace tenzor::nn;

TEST(MinimalFloat16Debug, ZeroInputWithRandomWeights) {
    std::cout << "\n=== Testing Float16 with zero input and random weights ===\n";

    auto device = Device::cpu();

    // Test Float32 first (baseline)
    {
        std::cout << "\n--- Float32 (baseline) ---\n";
        auto linear = std::make_shared<Linear>(4, 3);
        linear->to(DType::Float32);

        // Zero-initialized input (like in the test)
        Variable input_f32(Tensor({2, 4}, DType::Float32, device), true);

        std::cout << "Input (first 4 values): ";
        auto* input_data = input_f32.tensor().data<float>();
        for (int i = 0; i < 4; i++) {
            std::cout << input_data[i] << " ";
        }
        std::cout << "\n";

        // Check weight values
        auto params = linear->parameters();
        std::cout << "Number of parameters: " << params.size() << "\n";
        if (!params.empty()) {
            auto weight = params[0]->tensor();
            std::cout << "Weight dtype: " << static_cast<int>(weight.dtype()) << "\n";
            std::cout << "Weight shape: [";
            for (size_t i = 0; i < weight.shape().size(); i++) {
                if (i > 0) std::cout << ", ";
                std::cout << weight.shape()[i];
            }
            std::cout << "]\n";
            std::cout << "Weight values (first 4): ";
            auto* weight_data = weight.data<float>();
            for (int i = 0; i < std::min(4, static_cast<int>(weight.numel())); i++) {
                std::cout << weight_data[i] << " ";
            }
            std::cout << "\n";
        }

        auto output = linear->forward(input_f32);
        std::cout << "Output shape: [";
        for (size_t i = 0; i < output.shape().size(); i++) {
            if (i > 0) std::cout << ", ";
            std::cout << output.shape()[i];
        }
        std::cout << "]\n";
        std::cout << "Output values (first 3): ";
        auto* output_data = output.tensor().data<float>();
        for (int i = 0; i < std::min(3, static_cast<int>(output.tensor().numel())); i++) {
            std::cout << output_data[i] << " ";
        }
        std::cout << "\n";

        // Check if output has non-zero values
        bool has_nonzero = false;
        for (int i = 0; i < output.tensor().numel(); i++) {
            if (std::abs(output_data[i]) > 1e-6f) {
                has_nonzero = true;
                break;
            }
        }
        std::cout << "Has non-zero output: " << (has_nonzero ? "YES" : "NO") << "\n";
        EXPECT_TRUE(has_nonzero) << "Float32 should produce non-zero output with random weights";
    }

    // Now test Float16
    {
        std::cout << "\n--- Float16 (testing) ---\n";
        auto linear = std::make_shared<Linear>(4, 3);
        linear->to(DType::Float16);

        // Zero-initialized input (like in the test)
        Variable input_f16(Tensor({2, 4}, DType::Float16, device), true);

        std::cout << "Input (first 4 values): ";
        auto input_cpu = input_f16.tensor().to(DType::Float32);
        auto* input_data = input_cpu.data<float>();
        for (int i = 0; i < 4; i++) {
            std::cout << input_data[i] << " ";
        }
        std::cout << "\n";

        // Check weight values
        auto params = linear->parameters();
        std::cout << "Number of parameters: " << params.size() << "\n";
        if (!params.empty()) {
            auto weight = params[0]->tensor();
            std::cout << "Weight dtype: " << static_cast<int>(weight.dtype()) << "\n";
            std::cout << "Weight shape: [";
            for (size_t i = 0; i < weight.shape().size(); i++) {
                if (i > 0) std::cout << ", ";
                std::cout << weight.shape()[i];
            }
            std::cout << "]\n";

            auto weight_f32 = weight.to(DType::Float32);
            auto* weight_data = weight_f32.data<float>();
            std::cout << "Weight values (first 4): ";
            for (int i = 0; i < std::min(4, static_cast<int>(weight_f32.numel())); i++) {
                std::cout << weight_data[i] << " ";
            }
            std::cout << "\n";

            // Check if weights are all zero
            bool weights_all_zero = true;
            for (int i = 0; i < weight_f32.numel(); i++) {
                if (std::abs(weight_data[i]) > 1e-6f) {
                    weights_all_zero = false;
                    break;
                }
            }
            std::cout << "Weights all zero: " << (weights_all_zero ? "YES (BUG!)" : "NO (good)") << "\n";
        }

        auto output = linear->forward(input_f16);
        std::cout << "Output shape: [";
        for (size_t i = 0; i < output.shape().size(); i++) {
            if (i > 0) std::cout << ", ";
            std::cout << output.shape()[i];
        }
        std::cout << "]\n";

        auto output_f32 = output.tensor().to(DType::Float32);
        auto* output_data = output_f32.data<float>();
        std::cout << "Output values (first 3): ";
        for (int i = 0; i < std::min(3, static_cast<int>(output_f32.numel())); i++) {
            std::cout << output_data[i] << " ";
        }
        std::cout << "\n";

        // Check if output has non-zero values
        bool has_nonzero = false;
        for (int i = 0; i < output_f32.numel(); i++) {
            if (std::abs(output_data[i]) > 1e-6f) {
                has_nonzero = true;
                break;
            }
        }
        std::cout << "Has non-zero output: " << (has_nonzero ? "YES" : "NO (BUG!)") << "\n";
        EXPECT_TRUE(has_nonzero) << "Float16 should produce non-zero output with random weights";
    }
}

int main(int argc, char** argv) {
    initialize();
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
