#include <iostream>
#include <gtest/gtest.h>
#include "tenzor/tenzor.hpp"

using namespace tenzor;
using namespace tenzor::nn;

TEST(MinimalFloat16Conv2d, ZeroInputWithRandomWeights) {
    std::cout << "\n=== Testing Conv2d Float16 vs Float32 with zero input ===\n";

    initialize();
    auto device = Device::cpu();

    // Test Float32 first (baseline)
    {
        std::cout << "\n--- Float32 (baseline) ---\n";

        // Create Conv2d: 3 input channels -> 96 output channels, 4x4 kernel, stride=4
        auto conv = std::make_shared<Conv2d>(3, 96, 4, 4);  // in, out, kernel, stride
        conv->to(DType::Float32);

        // Zero-initialized input (like in the test): [1, 3, 224, 224]
        Variable input_f32(Tensor({1, 3, 224, 224}, DType::Float32, device), true);

        std::cout << "Input (first 4 values): ";
        auto* input_data = input_f32.tensor().data<float>();
        for (int i = 0; i < 4; i++) {
            std::cout << input_data[i] << " ";
        }
        std::cout << "\n";

        // Check weight values
        auto params = conv->parameters();
        std::cout << "Number of parameters: " << params.size() << "\n";
        if (!params.empty()) {
            auto weight = params[0]->tensor();
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

        auto output = conv->forward(input_f32);
        std::cout << "Output shape: [";
        for (size_t i = 0; i < output.shape().size(); i++) {
            if (i > 0) std::cout << ", ";
            std::cout << output.shape()[i];
        }
        std::cout << "]\n";

        std::cout << "Output values (first 10): ";
        auto* output_data = output.tensor().data<float>();
        for (int i = 0; i < std::min(10, static_cast<int>(output.tensor().numel())); i++) {
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
    }

    // Now test Float16
    {
        std::cout << "\n--- Float16 (testing) ---\n";

        // Create Conv2d with same configuration
        auto conv = std::make_shared<Conv2d>(3, 96, 4, 4);
        conv->to(DType::Float16);

        // Zero-initialized input: [1, 3, 224, 224]
        Variable input_f16(Tensor({1, 3, 224, 224}, DType::Float16, device), true);

        std::cout << "Input (first 4 values): ";
        auto input_cpu = input_f16.tensor().to(DType::Float32);
        auto* input_data = input_cpu.data<float>();
        for (int i = 0; i < 4; i++) {
            std::cout << input_data[i] << " ";
        }
        std::cout << "\n";

        // Check weight values
        auto params = conv->parameters();
        std::cout << "Number of parameters: " << params.size() << "\n";
        if (!params.empty()) {
            auto weight = params[0]->tensor();
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

        auto output = conv->forward(input_f16);
        std::cout << "Output shape: [";
        for (size_t i = 0; i < output.shape().size(); i++) {
            if (i > 0) std::cout << ", ";
            std::cout << output.shape()[i];
        }
        std::cout << "]\n";

        auto output_f32 = output.tensor().to(DType::Float32);
        auto* output_data = output_f32.data<float>();
        std::cout << "Output values (first 10): ";
        for (int i = 0; i < std::min(10, static_cast<int>(output_f32.numel())); i++) {
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
        std::cout << "Has non-zero output: " << (has_nonzero ? "YES" : "NO (this is EXPECTED with zero input)") << "\n";
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
