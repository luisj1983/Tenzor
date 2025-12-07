#include <gtest/gtest.h>
#include "../backend_test_fixture.hpp"
#include <iostream>
#include <cmath>
#include <tenzor/tenzor.hpp>

using namespace tenzor;
using namespace tenzor::testing;

// Utility to check for NaN in tensor
static bool has_nan_debug(const Tensor& t) {
    auto cpu = t.device().type == Device::Type::CPU ? t : t.to(Device::cpu());
    if (t.dtype() == DType::Float32) {
        const float* data = cpu.data<float>();
        for (int64_t i = 0; i < cpu.numel(); ++i) {
            if (std::isnan(data[i])) return true;
        }
    }
    return false;
}

class DebugMLPTest : public BackendTest {};

TEST_P(DebugMLPTest, DirectMatmulComparison) {
    // Compare direct matmul between backends
    std::cout << "Device: " << GetParam() << std::endl;

    // Create simple matrices on CPU
    // A: 4x3, B: 3x2, C = A @ B: 4x2
    auto A_cpu = ones({4, 3}, DType::Float32, Device::cpu());
    auto B_cpu = ones({3, 2}, DType::Float32, Device::cpu());

    // Set specific values for verification
    // A = [[1,2,3], [4,5,6], [7,8,9], [10,11,12]]
    // B = [[1,2], [3,4], [5,6]]
    // Expected C = [[22, 28], [49, 64], [76, 100], [103, 136]]
    {
        float* a_data = const_cast<float*>(A_cpu.data<float>());
        float* b_data = const_cast<float*>(B_cpu.data<float>());
        for (int i = 0; i < 12; ++i) a_data[i] = static_cast<float>(i + 1);
        for (int i = 0; i < 6; ++i) b_data[i] = static_cast<float>(i + 1);
    }

    // CPU matmul
    auto C_cpu = matmul(A_cpu, B_cpu);
    std::cout << "CPU matmul result:" << std::endl;
    {
        const float* c_data = C_cpu.data<float>();
        for (int i = 0; i < 4; ++i) {
            std::cout << "  [" << c_data[i*2] << ", " << c_data[i*2+1] << "]" << std::endl;
        }
    }

    // Device matmul
    auto A_dev = A_cpu.to(device);
    auto B_dev = B_cpu.to(device);
    auto C_dev = matmul(A_dev, B_dev);
    auto C_dev_cpu = C_dev.to(Device::cpu());

    std::cout << "Device matmul result:" << std::endl;
    {
        const float* c_data = C_dev_cpu.data<float>();
        for (int i = 0; i < 4; ++i) {
            std::cout << "  [" << c_data[i*2] << ", " << c_data[i*2+1] << "]" << std::endl;
        }
    }

    // Verify correctness
    const float* cpu_data = C_cpu.data<float>();
    const float* dev_data = C_dev_cpu.data<float>();
    for (int i = 0; i < 8; ++i) {
        EXPECT_NEAR(cpu_data[i], dev_data[i], 1e-4f) << "Mismatch at index " << i;
    }
}

TEST_P(DebugMLPTest, VariableMatmulWithPermute) {
    // Test Variable matmul with permuted weight - exact Linear layer simulation
    std::cout << "Device: " << GetParam() << std::endl;

    // Simulate Linear layer: input (4, 3) @ weight.T (3, 2) = output (4, 2)
    // weight: (2, 3), weight.T: (3, 2)
    auto input_tensor = ones({4, 3}, DType::Float32, device);
    auto weight_tensor = ones({2, 3}, DType::Float32, device);

    // Set specific values (same as MatmulWithTranspose test)
    {
        auto in_cpu = input_tensor.to(Device::cpu());
        auto w_cpu = weight_tensor.to(Device::cpu());
        float* in_data = const_cast<float*>(in_cpu.data<float>());
        float* w_data = const_cast<float*>(w_cpu.data<float>());
        for (int i = 0; i < 12; ++i) in_data[i] = static_cast<float>(i + 1);
        for (int i = 0; i < 6; ++i) w_data[i] = static_cast<float>(i + 1);
        input_tensor = in_cpu.to(device);
        weight_tensor = w_cpu.to(device);
    }

    // Create Variables (weight requires grad, like in Linear)
    auto input_var = Variable(input_tensor, false);
    auto weight_var = Variable(weight_tensor, true);

    // Permute weight (like Linear layer does)
    auto weight_t_var = permute(weight_var, {1, 0});

    std::cout << "weight_t is_contiguous: " << weight_t_var.tensor().is_contiguous() << std::endl;

    // Matmul
    auto output_var = matmul(input_var, weight_t_var);
    auto output_cpu = output_var.tensor().to(Device::cpu());

    std::cout << "Variable matmul result:" << std::endl;
    {
        const float* c_data = output_cpu.data<float>();
        for (int i = 0; i < 4; ++i) {
            std::cout << "  [" << c_data[i*2] << ", " << c_data[i*2+1] << "]" << std::endl;
        }
    }

    // Expected: [[14, 32], [32, 77], [50, 122], [68, 167]]
    const float* data = output_cpu.data<float>();
    EXPECT_NEAR(data[0], 14.0f, 1e-4f);
    EXPECT_NEAR(data[1], 32.0f, 1e-4f);
    EXPECT_NEAR(data[2], 32.0f, 1e-4f);
    EXPECT_NEAR(data[3], 77.0f, 1e-4f);
    EXPECT_NEAR(data[4], 50.0f, 1e-4f);
    EXPECT_NEAR(data[5], 122.0f, 1e-4f);
    EXPECT_NEAR(data[6], 68.0f, 1e-4f);
    EXPECT_NEAR(data[7], 167.0f, 1e-4f);
}

TEST_P(DebugMLPTest, LinearLayerSimulation) {
    // Exactly simulate what Linear layer does
    std::cout << "Device: " << GetParam() << std::endl;

    // Create Linear layer on CPU first (like actual test)
    // Linear(3, 2) = 3 inputs, 2 outputs
    auto fc = std::make_shared<nn::Linear>(3, 2, false);  // No bias for simplicity

    // Get weight before device transfer
    {
        auto w = fc->weight()->tensor();
        const float* wdata = w.data<float>();
        std::cout << "Weight on CPU (shape " << w.shape()[0] << "x" << w.shape()[1] << "):" << std::endl;
        for (int64_t i = 0; i < w.shape()[0]; ++i) {
            std::cout << "  [";
            for (int64_t j = 0; j < w.shape()[1]; ++j) {
                if (j > 0) std::cout << ", ";
                std::cout << wdata[i * w.shape()[1] + j];
            }
            std::cout << "]" << std::endl;
        }
    }

    // Move to device
    fc->to(device);

    // Get weight after device transfer
    {
        auto w = fc->weight()->tensor().to(Device::cpu());
        const float* wdata = w.data<float>();
        std::cout << "Weight after to(device):" << std::endl;
        for (int64_t i = 0; i < w.shape()[0]; ++i) {
            std::cout << "  [";
            for (int64_t j = 0; j < w.shape()[1]; ++j) {
                if (j > 0) std::cout << ", ";
                std::cout << wdata[i * w.shape()[1] + j];
            }
            std::cout << "]" << std::endl;
        }
    }

    // Create input: 2x3 (batch_size=2, in_features=3)
    auto input_cpu = ones({2, 3}, DType::Float32, Device::cpu());
    {
        float* in_data = const_cast<float*>(input_cpu.data<float>());
        // input = [[1,2,3], [4,5,6]]
        for (int i = 0; i < 6; ++i) in_data[i] = static_cast<float>(i + 1);
    }
    std::cout << "Input:" << std::endl;
    {
        const float* in_data = input_cpu.data<float>();
        for (int i = 0; i < 2; ++i) {
            std::cout << "  [" << in_data[i*3] << ", " << in_data[i*3+1] << ", " << in_data[i*3+2] << "]" << std::endl;
        }
    }

    // Forward on device (layer already on device from to())
    auto input_dev = input_cpu.to(device);
    auto input_var_dev = Variable(input_dev, false);
    auto output_dev = fc->forward(input_var_dev);
    std::cout << "Device forward result:" << std::endl;
    {
        auto out = output_dev.tensor().to(Device::cpu());
        const float* out_data = out.data<float>();
        for (int i = 0; i < 2; ++i) {
            std::cout << "  [" << out_data[i*2] << ", " << out_data[i*2+1] << "]" << std::endl;
        }
    }

    // Create a reference: manually compute expected output
    // output = input @ weight.T
    // weight = [[w00, w01, w02], [w10, w11, w12]]
    // weight.T = [[w00, w10], [w01, w11], [w02, w12]]
    // output[0] = [1,2,3] @ weight.T = [1*w00+2*w01+3*w02, 1*w10+2*w11+3*w12]
    // output[1] = [4,5,6] @ weight.T = [4*w00+5*w01+6*w02, 4*w10+5*w11+6*w12]
    auto w_cpu = fc->weight()->tensor().to(Device::cpu());
    const float* w = w_cpu.data<float>();
    float expected[4];
    expected[0] = 1*w[0] + 2*w[1] + 3*w[2];  // row 0, col 0 of output
    expected[1] = 1*w[3] + 2*w[4] + 3*w[5];  // row 0, col 1 of output
    expected[2] = 4*w[0] + 5*w[1] + 6*w[2];  // row 1, col 0 of output
    expected[3] = 4*w[3] + 5*w[4] + 6*w[5];  // row 1, col 1 of output

    std::cout << "Expected (manually computed):" << std::endl;
    std::cout << "  [" << expected[0] << ", " << expected[1] << "]" << std::endl;
    std::cout << "  [" << expected[2] << ", " << expected[3] << "]" << std::endl;

    // Verify
    auto out_cpu = output_dev.tensor().to(Device::cpu());
    const float* out_data = out_cpu.data<float>();
    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(out_data[i], expected[i], 1e-3f) << "Mismatch at index " << i;
    }
}

TEST_P(DebugMLPTest, MatmulWithTranspose) {
    // Test matmul with transposed matrix - simulates Linear layer operation
    std::cout << "Device: " << GetParam() << std::endl;

    // Create matrices on CPU
    // Input: 4x3, Weight: 2x3 (like Linear with in=3, out=2)
    // Linear does: input @ weight.T = 4x3 @ 3x2 = 4x2
    auto input_cpu = ones({4, 3}, DType::Float32, Device::cpu());
    auto weight_cpu = ones({2, 3}, DType::Float32, Device::cpu());

    // Set specific values
    // input = [[1,2,3], [4,5,6], [7,8,9], [10,11,12]]
    // weight = [[1,2,3], [4,5,6]]
    // weight.T = [[1,4], [2,5], [3,6]]
    // Expected: input @ weight.T = [[14, 32], [32, 77], [50, 122], [68, 167]]
    {
        float* in_data = const_cast<float*>(input_cpu.data<float>());
        float* w_data = const_cast<float*>(weight_cpu.data<float>());
        for (int i = 0; i < 12; ++i) in_data[i] = static_cast<float>(i + 1);
        for (int i = 0; i < 6; ++i) w_data[i] = static_cast<float>(i + 1);
    }

    // CPU: input @ weight.T
    auto weight_t_cpu = weight_cpu.permute({1, 0});  // 3x2
    std::cout << "weight_t is_contiguous: " << weight_t_cpu.is_contiguous() << std::endl;

    auto C_cpu = matmul(input_cpu, weight_t_cpu);
    std::cout << "CPU matmul(input, weight.T) result:" << std::endl;
    {
        const float* c_data = C_cpu.data<float>();
        for (int i = 0; i < 4; ++i) {
            std::cout << "  [" << c_data[i*2] << ", " << c_data[i*2+1] << "]" << std::endl;
        }
    }

    // Device: same operation
    auto input_dev = input_cpu.to(device);
    auto weight_dev = weight_cpu.to(device);
    auto weight_t_dev = weight_dev.permute({1, 0});  // Creates non-contiguous view
    std::cout << "Device weight_t is_contiguous: " << weight_t_dev.is_contiguous() << std::endl;

    auto C_dev = matmul(input_dev, weight_t_dev);
    auto C_dev_cpu = C_dev.to(Device::cpu());

    std::cout << "Device matmul(input, weight.T) result:" << std::endl;
    {
        const float* c_data = C_dev_cpu.data<float>();
        for (int i = 0; i < 4; ++i) {
            std::cout << "  [" << c_data[i*2] << ", " << c_data[i*2+1] << "]" << std::endl;
        }
    }

    // Verify correctness
    const float* cpu_data = C_cpu.data<float>();
    const float* dev_data = C_dev_cpu.data<float>();
    for (int i = 0; i < 8; ++i) {
        EXPECT_NEAR(cpu_data[i], dev_data[i], 1e-4f) << "Mismatch at index " << i;
    }
}

TEST_P(DebugMLPTest, LinearForwardComparison) {
    // Compare Linear forward pass between backends
    std::cout << "Device: " << GetParam() << std::endl;

    // Create a Linear layer on CPU first
    auto fc_cpu = std::make_shared<nn::Linear>(50, 30);

    // Create fixed input data on CPU
    auto input_cpu = ones({8, 50}, DType::Float32, Device::cpu());

    // Forward on CPU
    auto input_var_cpu = Variable(input_cpu, false);
    auto output_cpu = fc_cpu->forward(input_var_cpu);

    // Get CPU output stats
    float cpu_max, cpu_mean;
    {
        auto out = output_cpu.tensor();
        const float* data = out.data<float>();
        float max_v = 0, sum = 0;
        for (int64_t i = 0; i < out.numel(); ++i) {
            if (std::abs(data[i]) > max_v) max_v = std::abs(data[i]);
            sum += data[i];
        }
        cpu_max = max_v;
        cpu_mean = sum / out.numel();
        std::cout << "CPU output: max=" << max_v << ", mean=" << cpu_mean << std::endl;
    }

    // Now move to target device and forward again
    fc_cpu->to(device);

    // Check weight after moving
    {
        auto w = fc_cpu->weight()->tensor().to(Device::cpu());
        const float* wdata = w.data<float>();
        float max_w = 0, sum = 0;
        for (int64_t i = 0; i < w.numel(); ++i) {
            if (std::abs(wdata[i]) > max_w) max_w = std::abs(wdata[i]);
            sum += wdata[i];
        }
        std::cout << "Weight after to(device): max=" << max_w << ", mean=" << sum/w.numel()
                  << ", is_contiguous=" << fc_cpu->weight()->tensor().is_contiguous() << std::endl;
    }

    auto input_dev = input_cpu.to(device);
    auto input_var_dev = Variable(input_dev, false);
    auto output_dev = fc_cpu->forward(input_var_dev);

    // Get device output stats
    float dev_max, dev_mean;
    {
        auto out = output_dev.tensor().to(Device::cpu());
        const float* data = out.data<float>();
        float max_v = 0, sum = 0;
        for (int64_t i = 0; i < out.numel(); ++i) {
            if (std::abs(data[i]) > max_v) max_v = std::abs(data[i]);
            sum += data[i];
        }
        dev_max = max_v;
        dev_mean = sum / out.numel();
        std::cout << "Device output: max=" << max_v << ", mean=" << dev_mean << std::endl;
    }

    // Verify outputs are close (within 10% relative error)
    EXPECT_NEAR(dev_max, cpu_max, cpu_max * 0.1f) << "Max values differ significantly";
    EXPECT_NEAR(dev_mean, cpu_mean, std::abs(cpu_mean) * 0.1f + 0.01f) << "Mean values differ significantly";
}

TEST_P(DebugMLPTest, CPUForwardThenDeviceForward) {
    // This test checks for the bug: forward on CPU first, move to device, forward on device
    // Use EXACT same dimensions as SimpleMLP_Training: 50->30 and 8x50 input
    std::cout << "Device: " << GetParam() << std::endl;

    // Create layer on CPU with dimensions matching SimpleMLP's fc1
    auto fc = std::make_shared<nn::Linear>(50, 30);

    // Fixed input (same size as SimpleMLP_Training uses, but fixed values)
    auto input_cpu = ones({8, 50}, DType::Float32, Device::cpu());

    // Forward on CPU FIRST
    auto input_var_cpu = Variable(input_cpu, false);
    auto output_cpu1 = fc->forward(input_var_cpu);
    float cpu_max = 0;
    {
        auto out = output_cpu1.tensor();
        const float* data = out.data<float>();
        for (int64_t i = 0; i < out.numel(); ++i) {
            if (std::abs(data[i]) > cpu_max) cpu_max = std::abs(data[i]);
        }
    }
    std::cout << "CPU forward output max: " << cpu_max << std::endl;

    // Move layer to device
    fc->to(device);

    // Forward on DEVICE
    auto input_dev = input_cpu.to(device);
    auto input_var_dev = Variable(input_dev, false);
    auto output_dev = fc->forward(input_var_dev);
    float dev_max = 0;
    {
        auto out = output_dev.tensor().to(Device::cpu());
        const float* data = out.data<float>();
        for (int64_t i = 0; i < out.numel(); ++i) {
            if (std::abs(data[i]) > dev_max) dev_max = std::abs(data[i]);
        }
    }
    std::cout << "Device forward output max: " << dev_max << std::endl;

    // The max values should be similar (not orders of magnitude different)
    EXPECT_NEAR(dev_max, cpu_max, cpu_max * 0.1f + 0.01f) << "Output max values differ too much";
}

TEST_P(DebugMLPTest, TraceNaN) {
    std::cout << "Device: " << GetParam() << std::endl;

    // Simple MLP: 50 -> 30 -> 10
    auto fc1 = std::make_shared<nn::Linear>(50, 30);
    auto fc2 = std::make_shared<nn::Linear>(30, 10);
    fc1->to(device);
    fc2->to(device);

    std::vector<std::shared_ptr<Variable>> params;
    for (auto& p : fc1->parameters()) params.push_back(p);
    for (auto& p : fc2->parameters()) params.push_back(p);

    auto optimizer = optim::SGD(params, 0.01);

    // Check initial weights before any training
    std::cout << "\n=== Initial weights ===" << std::endl;
    {
        auto w1 = fc1->weight()->tensor().to(Device::cpu());
        const float* wdata = w1.data<float>();
        float max_w = 0, sum = 0;
        for (int64_t i = 0; i < w1.numel(); ++i) {
            if (std::abs(wdata[i]) > max_w) max_w = std::abs(wdata[i]);
            sum += wdata[i];
        }
        std::cout << "fc1.weight init max: " << max_w << ", mean: " << sum / w1.numel() << std::endl;
    }
    {
        auto w2 = fc2->weight()->tensor().to(Device::cpu());
        const float* wdata = w2.data<float>();
        float max_w = 0, sum = 0;
        for (int64_t i = 0; i < w2.numel(); ++i) {
            if (std::abs(wdata[i]) > max_w) max_w = std::abs(wdata[i]);
            sum += wdata[i];
        }
        std::cout << "fc2.weight init max: " << max_w << ", mean: " << sum / w2.numel() << std::endl;
    }

    for (int epoch = 0; epoch < 3; ++epoch) {
        std::cout << "\n=== Epoch " << epoch << " ===" << std::endl;

        auto input = Variable(randn({8, 50}, DType::Float32, device), true);
        auto target = Variable(randn({8, 10}, DType::Float32, device));

        // Check input range
        {
            auto inp_cpu = input.tensor().to(Device::cpu());
            const float* inp_data = inp_cpu.data<float>();
            float max_inp = 0;
            for (int64_t i = 0; i < inp_cpu.numel(); ++i) {
                if (std::abs(inp_data[i]) > max_inp) max_inp = std::abs(inp_data[i]);
            }
            std::cout << "Input max val: " << max_inp << std::endl;
        }

        std::cout << "Input has NaN: " << has_nan_debug(input.tensor()) << std::endl;

        optimizer.zero_grad();

        // Forward fc1
        auto h1 = fc1->forward(input);
        std::cout << "After fc1 has NaN: " << has_nan_debug(h1.tensor()) << std::endl;
        {
            auto h1_cpu = h1.tensor().to(Device::cpu());
            const float* h1_data = h1_cpu.data<float>();
            float max_h1 = 0;
            for (int64_t i = 0; i < h1_cpu.numel(); ++i) {
                if (std::abs(h1_data[i]) > max_h1) max_h1 = std::abs(h1_data[i]);
            }
            std::cout << "h1 max val: " << max_h1 << std::endl;
        }

        // ReLU
        auto h2 = nn::relu(h1);
        std::cout << "After relu has NaN: " << has_nan_debug(h2.tensor()) << std::endl;
        {
            auto h2_cpu = h2.tensor().to(Device::cpu());
            const float* h2_data = h2_cpu.data<float>();
            float max_h2 = 0;
            for (int64_t i = 0; i < h2_cpu.numel(); ++i) {
                if (std::abs(h2_data[i]) > max_h2) max_h2 = std::abs(h2_data[i]);
            }
            std::cout << "h2 max val: " << max_h2 << std::endl;
        }

        // Forward fc2
        auto output = fc2->forward(h2);
        std::cout << "After fc2 has NaN: " << has_nan_debug(output.tensor()) << std::endl;
        {
            auto out_cpu = output.tensor().to(Device::cpu());
            const float* out_data = out_cpu.data<float>();
            float max_out = 0;
            for (int64_t i = 0; i < out_cpu.numel(); ++i) {
                if (std::abs(out_data[i]) > max_out) max_out = std::abs(out_data[i]);
            }
            std::cout << "output max val: " << max_out << std::endl;
        }

        // MSE Loss
        auto loss = nn::mse_loss(output, target);
        auto loss_cpu = loss.tensor().to(Device::cpu());
        const float* loss_ptr = loss_cpu.data<float>();
        float loss_val = loss_ptr[0];
        std::cout << "Loss: " << loss_val << " (NaN: " << std::isnan(loss_val) << ")" << std::endl;

        if (std::isnan(loss_val)) {
            FAIL() << "NaN detected in loss at epoch " << epoch << " before backward";
        }

        // Backward
        std::cout << "Starting backward..." << std::endl;
        loss.backward();
        std::cout << "Backward complete." << std::endl;

        // Check gradients
        if (fc1->weight()->has_grad()) {
            std::cout << "fc1.weight.grad has NaN: " << has_nan_debug(fc1->weight()->grad().value()) << std::endl;
        }
        if (fc2->weight()->has_grad()) {
            std::cout << "fc2.weight.grad has NaN: " << has_nan_debug(fc2->weight()->grad().value()) << std::endl;
        }

        // Optimizer step
        std::cout << "Optimizer step..." << std::endl;
        optimizer.step();

        // Check weights after step
        std::cout << "fc1.weight after step has NaN: " << has_nan_debug(fc1->weight()->tensor()) << std::endl;
        std::cout << "fc2.weight after step has NaN: " << has_nan_debug(fc2->weight()->tensor()) << std::endl;

        // Print max gradient magnitude
        if (fc1->weight()->has_grad()) {
            auto g = fc1->weight()->grad().value().to(Device::cpu());
            const float* gdata = g.data<float>();
            float max_grad = 0;
            for (int64_t i = 0; i < g.numel(); ++i) {
                if (std::abs(gdata[i]) > max_grad) max_grad = std::abs(gdata[i]);
            }
            std::cout << "fc1.weight max grad: " << max_grad << std::endl;
        }
        if (fc2->weight()->has_grad()) {
            auto g = fc2->weight()->grad().value().to(Device::cpu());
            const float* gdata = g.data<float>();
            float max_grad = 0;
            for (int64_t i = 0; i < g.numel(); ++i) {
                if (std::abs(gdata[i]) > max_grad) max_grad = std::abs(gdata[i]);
            }
            std::cout << "fc2.weight max grad: " << max_grad << std::endl;
        }

        // Print max weight magnitude
        {
            auto w = fc1->weight()->tensor().to(Device::cpu());
            const float* wdata = w.data<float>();
            float max_w = 0;
            for (int64_t i = 0; i < w.numel(); ++i) {
                if (std::abs(wdata[i]) > max_w) max_w = std::abs(wdata[i]);
            }
            std::cout << "fc1.weight max val: " << max_w << std::endl;
        }
        {
            auto w = fc2->weight()->tensor().to(Device::cpu());
            const float* wdata = w.data<float>();
            float max_w = 0;
            for (int64_t i = 0; i < w.numel(); ++i) {
                if (std::abs(wdata[i]) > max_w) max_w = std::abs(wdata[i]);
            }
            std::cout << "fc2.weight max val: " << max_w << std::endl;
        }
    }

    SUCCEED();
}

TEST_P(DebugMLPTest, RawMatmulDimensionScan) {
    // Scan different dimensions to find where the bug appears
    std::cout << "Device: " << GetParam() << std::endl;

    // Test the minimal failing case: 2x17 @ 17x3
    // Verify contiguous data element by element
    int m = 2, k = 17, n = 3;
    std::cout << "Testing " << m << "x" << k << " @ " << k << "x" << n << std::endl;

    // Create simple input on CPU
    auto input_cpu = ones({m, k}, DType::Float32, Device::cpu());
    auto weight_cpu = ones({n, k}, DType::Float32, Device::cpu());

    // Set identifiable values
    {
        float* in_data = const_cast<float*>(input_cpu.data<float>());
        float* w_data = const_cast<float*>(weight_cpu.data<float>());
        for (int64_t i = 0; i < m * k; ++i) in_data[i] = static_cast<float>(i + 1);  // 1, 2, 3, ..., 34
        for (int64_t i = 0; i < n * k; ++i) w_data[i] = 1.0f;  // all ones
    }

    // CPU: input @ weight.T
    // Each row of output should be sum of that row of input (since weight is all ones)
    auto weight_t_cpu = weight_cpu.permute({1, 0});  // 17x3
    auto output_cpu = matmul(input_cpu, weight_t_cpu);  // 2x3

    std::cout << "CPU output (2x3):" << std::endl;
    {
        const float* data = output_cpu.data<float>();
        for (int i = 0; i < m; ++i) {
            std::cout << "  [";
            for (int j = 0; j < n; ++j) {
                if (j > 0) std::cout << ", ";
                std::cout << data[i * n + j];
            }
            std::cout << "]" << std::endl;
        }
    }
    // Expected: row 0 = 1+2+...+17 = 153, row 1 = 18+19+...+34 = 442

    // Device
    auto input_dev = input_cpu.to(device);
    auto weight_dev = weight_cpu.to(device);
    auto weight_t_dev = weight_dev.permute({1, 0});

    // Check if contiguous makes the data correct
    auto weight_t_dev_cont = weight_t_dev.contiguous();
    auto weight_t_dev_cont_cpu = weight_t_dev_cont.to(Device::cpu());

    std::cout << "Weight_t shape: " << weight_t_dev.shape()[0] << "x" << weight_t_dev.shape()[1] << std::endl;
    std::cout << "Weight_t strides: " << weight_t_dev.strides()[0] << ", " << weight_t_dev.strides()[1] << std::endl;
    std::cout << "Weight_t_cont shape: " << weight_t_dev_cont.shape()[0] << "x" << weight_t_dev_cont.shape()[1] << std::endl;

    std::cout << "Device weight_t contiguous (17x3) ALL data:" << std::endl;
    {
        const float* data = weight_t_dev_cont_cpu.data<float>();
        for (int i = 0; i < k; ++i) {
            std::cout << "  row " << i << ": [";
            for (int j = 0; j < n; ++j) {
                if (j > 0) std::cout << ", ";
                std::cout << data[i * n + j];
            }
            std::cout << "]" << std::endl;
        }
    }

    // Compare with CPU contiguous weight_t
    auto weight_t_cpu_cont = weight_t_cpu.contiguous();
    std::cout << "CPU weight_t contiguous (17x3) ALL data:" << std::endl;
    {
        const float* data = weight_t_cpu_cont.data<float>();
        for (int i = 0; i < k; ++i) {
            std::cout << "  row " << i << ": [";
            for (int j = 0; j < n; ++j) {
                if (j > 0) std::cout << ", ";
                std::cout << data[i * n + j];
            }
            std::cout << "]" << std::endl;
        }
    }

    auto output_dev = matmul(input_dev, weight_t_dev);
    auto output_dev_cpu = output_dev.to(Device::cpu());

    std::cout << "Device output (2x3):" << std::endl;
    {
        const float* data = output_dev_cpu.data<float>();
        for (int i = 0; i < m; ++i) {
            std::cout << "  [";
            for (int j = 0; j < n; ++j) {
                if (j > 0) std::cout << ", ";
                std::cout << data[i * n + j];
            }
            std::cout << "]" << std::endl;
        }
    }

    // Verify
    const float* cpu_data = output_cpu.data<float>();
    const float* dev_data = output_dev_cpu.data<float>();
    for (int i = 0; i < m * n; ++i) {
        EXPECT_NEAR(cpu_data[i], dev_data[i], 1e-3f) << "Mismatch at index " << i;
    }
}

TEST_P(DebugMLPTest, RawMatmulWithPermute50x30) {
    // Test raw matmul with permute at SimpleMLP dimensions (50->30)
    std::cout << "Device: " << GetParam() << std::endl;

    // input: 8x50, weight: 30x50, weight.T: 50x30, output: 8x30
    auto input_cpu = randn({8, 50}, DType::Float32, Device::cpu());
    auto weight_cpu = randn({30, 50}, DType::Float32, Device::cpu());

    // CPU forward: input @ weight.T
    auto weight_t_cpu = weight_cpu.permute({1, 0});  // 50x30
    auto output_cpu = matmul(input_cpu, weight_t_cpu);  // 8x30

    float cpu_max = 0, cpu_sum = 0;
    {
        const float* data = output_cpu.data<float>();
        for (int64_t i = 0; i < output_cpu.numel(); ++i) {
            if (std::abs(data[i]) > cpu_max) cpu_max = std::abs(data[i]);
            cpu_sum += data[i];
        }
    }
    std::cout << "CPU matmul: max=" << cpu_max << ", sum=" << cpu_sum << std::endl;

    // Device forward: same operation
    auto input_dev = input_cpu.to(device);
    auto weight_dev = weight_cpu.to(device);
    auto weight_t_dev = weight_dev.permute({1, 0});  // 50x30

    std::cout << "weight_t_dev is_contiguous: " << weight_t_dev.is_contiguous() << std::endl;
    std::cout << "weight_t_dev strides: [" << weight_t_dev.strides()[0] << ", " << weight_t_dev.strides()[1] << "]" << std::endl;

    // First, check if contiguous() works correctly
    auto weight_t_dev_cont = weight_t_dev.contiguous();
    auto weight_t_dev_cont_cpu = weight_t_dev_cont.to(Device::cpu());
    std::cout << "weight_t_dev_cont is_contiguous: " << weight_t_dev_cont_cpu.is_contiguous() << std::endl;

    // Compare contiguous versions
    auto weight_t_cpu_cont = weight_t_cpu.contiguous();
    float cpu_w_sum = 0, dev_w_sum = 0;
    {
        const float* cpu_data = weight_t_cpu_cont.data<float>();
        const float* dev_data = weight_t_dev_cont_cpu.data<float>();
        for (int64_t i = 0; i < weight_t_cpu_cont.numel(); ++i) {
            cpu_w_sum += cpu_data[i];
            dev_w_sum += dev_data[i];
        }
    }
    std::cout << "CPU weight_t contiguous sum: " << cpu_w_sum << std::endl;
    std::cout << "Device weight_t contiguous sum: " << dev_w_sum << std::endl;

    // Check first few elements
    std::cout << "CPU weight_t_cont first 5 elements: ";
    {
        const float* data = weight_t_cpu_cont.data<float>();
        for (int i = 0; i < 5; ++i) std::cout << data[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "Dev weight_t_cont first 5 elements: ";
    {
        const float* data = weight_t_dev_cont_cpu.data<float>();
        for (int i = 0; i < 5; ++i) std::cout << data[i] << " ";
    }
    std::cout << std::endl;

    // IMPORTANT: Test with manually made contiguous tensor
    std::cout << "\n=== Testing matmul with manually contiguous weight_t ===" << std::endl;
    auto output_cont_dev = matmul(input_dev, weight_t_dev_cont);  // Should use contiguous version
    auto output_cont_dev_cpu = output_cont_dev.to(Device::cpu());

    float dev_cont_max = 0, dev_cont_sum = 0;
    {
        const float* data = output_cont_dev_cpu.data<float>();
        for (int64_t i = 0; i < output_cont_dev_cpu.numel(); ++i) {
            if (std::abs(data[i]) > dev_cont_max) dev_cont_max = std::abs(data[i]);
            dev_cont_sum += data[i];
        }
    }
    std::cout << "Device matmul (with manually contiguous B): max=" << dev_cont_max << ", sum=" << dev_cont_sum << std::endl;

    auto output_dev = matmul(input_dev, weight_t_dev);  // 8x30
    auto output_dev_cpu = output_dev.to(Device::cpu());

    float dev_max = 0, dev_sum = 0;
    {
        const float* data = output_dev_cpu.data<float>();
        for (int64_t i = 0; i < output_dev_cpu.numel(); ++i) {
            if (std::abs(data[i]) > dev_max) dev_max = std::abs(data[i]);
            dev_sum += data[i];
        }
    }
    std::cout << "Device matmul (non-contiguous B): max=" << dev_max << ", sum=" << dev_sum << std::endl;

    float ratio = dev_max / cpu_max;
    std::cout << "Ratio (device/cpu) non-contig: " << ratio << std::endl;
    float ratio_cont = dev_cont_max / cpu_max;
    std::cout << "Ratio (device/cpu) manual contig: " << ratio_cont << std::endl;

    EXPECT_NEAR(dev_cont_max, cpu_max, cpu_max * 0.1f + 0.01f) << "Max values differ (manual contiguous)";
    EXPECT_NEAR(dev_max, cpu_max, cpu_max * 0.1f + 0.01f) << "Max values differ (non-contiguous)";
}

TEST_P(DebugMLPTest, CompareSingleLinearForward) {
    // Compare a single Linear forward pass with IDENTICAL weights and input
    std::cout << "Device: " << GetParam() << std::endl;

    // Create layer on CPU
    auto fc = std::make_shared<nn::Linear>(50, 30, false);  // No bias

    // Create fixed input on CPU (using randn-like distribution but fixed)
    auto input_cpu = randn({8, 50}, DType::Float32, Device::cpu());

    // Save input values for comparison
    float input_sum = 0;
    {
        const float* data = input_cpu.data<float>();
        for (int64_t i = 0; i < input_cpu.numel(); ++i) {
            input_sum += data[i];
        }
    }
    std::cout << "Input sum: " << input_sum << std::endl;

    // Forward on CPU
    auto input_var_cpu = Variable(input_cpu, false);
    auto output_cpu_var = fc->forward(input_var_cpu);
    auto output_cpu = output_cpu_var.tensor();

    float cpu_max = 0, cpu_sum = 0;
    {
        const float* data = output_cpu.data<float>();
        for (int64_t i = 0; i < output_cpu.numel(); ++i) {
            if (std::abs(data[i]) > cpu_max) cpu_max = std::abs(data[i]);
            cpu_sum += data[i];
        }
    }
    std::cout << "CPU forward: max=" << cpu_max << ", sum=" << cpu_sum << std::endl;

    // Get weight before moving to device
    auto weight_before = fc->weight()->tensor().contiguous();
    float weight_sum_before = 0;
    {
        const float* data = weight_before.data<float>();
        for (int64_t i = 0; i < weight_before.numel(); ++i) {
            weight_sum_before += data[i];
        }
    }
    std::cout << "Weight sum before to(device): " << weight_sum_before << std::endl;

    // Move layer to device
    fc->to(device);

    // Check weight after transfer
    auto weight_after = fc->weight()->tensor().to(Device::cpu());
    float weight_sum_after = 0;
    {
        const float* data = weight_after.data<float>();
        for (int64_t i = 0; i < weight_after.numel(); ++i) {
            weight_sum_after += data[i];
        }
    }
    std::cout << "Weight sum after to(device): " << weight_sum_after << std::endl;

    EXPECT_NEAR(weight_sum_before, weight_sum_after, 0.001f) << "Weight changed during to(device)!";

    // Move input to device (same values)
    auto input_dev = input_cpu.to(device);

    // Check input after transfer
    float input_sum_dev = 0;
    {
        auto inp_back = input_dev.to(Device::cpu());
        const float* data = inp_back.data<float>();
        for (int64_t i = 0; i < inp_back.numel(); ++i) {
            input_sum_dev += data[i];
        }
    }
    std::cout << "Input sum after to(device): " << input_sum_dev << std::endl;
    EXPECT_NEAR(input_sum, input_sum_dev, 0.001f) << "Input changed during to(device)!";

    // Forward on device
    auto input_var_dev = Variable(input_dev, false);
    auto output_dev_var = fc->forward(input_var_dev);
    auto output_dev_cpu = output_dev_var.tensor().to(Device::cpu());

    float dev_max = 0, dev_sum = 0;
    {
        const float* data = output_dev_cpu.data<float>();
        for (int64_t i = 0; i < output_dev_cpu.numel(); ++i) {
            if (std::abs(data[i]) > dev_max) dev_max = std::abs(data[i]);
            dev_sum += data[i];
        }
    }
    std::cout << "Device forward: max=" << dev_max << ", sum=" << dev_sum << std::endl;

    // Compare - they should be nearly identical
    float ratio = dev_max / cpu_max;
    std::cout << "Ratio (device/cpu): " << ratio << std::endl;

    EXPECT_NEAR(dev_max, cpu_max, cpu_max * 0.1f + 0.01f) << "Max values differ significantly";
    EXPECT_NEAR(dev_sum, cpu_sum, std::abs(cpu_sum) * 0.1f + 0.01f) << "Sum values differ significantly";
}

INSTANTIATE_BACKEND_TESTS(DebugMLPTest);
