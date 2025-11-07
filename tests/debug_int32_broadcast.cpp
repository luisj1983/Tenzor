#include <iostream>
#include <cstdint>
#include <tenzor/tenzor.hpp>

using namespace tenzor;

void print_int32_data(const std::string& label, const Tensor& t) {
    auto cpu = t.to(Device::cpu());
    auto data = cpu.data<int32_t>();
    std::cout << label << ": shape " << t.shape()[0];
    if (t.ndim() > 1) std::cout << "x" << t.shape()[1];
    std::cout << " = [";
    for (int64_t i = 0; i < cpu.numel(); i++) {
        if (i > 0) std::cout << ", ";
        std::cout << data[i];
    }
    std::cout << "]" << std::endl;
}

void print_float32_data(const std::string& label, const Tensor& t) {
    auto cpu = t.to(Device::cpu());
    auto data = cpu.data<float>();
    std::cout << label << ": shape " << t.shape()[0];
    if (t.ndim() > 1) std::cout << "x" << t.shape()[1];
    std::cout << " = [";
    for (int64_t i = 0; i < cpu.numel(); i++) {
        if (i > 0) std::cout << ", ";
        std::cout << data[i];
    }
    std::cout << "]" << std::endl;
}

int main() {
    try {
        auto device = Device::vulkan(0);
        std::cout << "=== Int32 Broadcast Debug Test ===" << std::endl;
        std::cout << "Device: " << device.to_string() << std::endl << std::endl;

        // Test 1: Verify Int32 data transfer works
        std::cout << "Test 1: Int32 Data Transfer" << std::endl;
        std::cout << "----------------------------" << std::endl;
        auto t1 = ones({3}, DType::Int32, device);
        print_int32_data("ones({3}, Int32) on Vulkan", t1);

        auto t1_cpu = t1.to(Device::cpu());
        auto t1_data = t1_cpu.data<int32_t>();
        t1_data[0] = 10; t1_data[1] = 20; t1_data[2] = 30;
        auto t1_gpu = t1_cpu.to(device);
        print_int32_data("Modified and transferred back", t1_gpu);
        std::cout << std::endl;

        // Test 2: Same-shape Int32 add (NO broadcasting)
        std::cout << "Test 2: Same-Shape Int32 Add (no broadcasting)" << std::endl;
        std::cout << "-----------------------------------------------" << std::endl;
        auto a2 = ones({3}, DType::Int32, device);
        auto b2 = ones({3}, DType::Int32, device);

        auto a2_cpu = a2.to(Device::cpu());
        auto b2_cpu = b2.to(Device::cpu());
        auto a2_data = a2_cpu.data<int32_t>();
        auto b2_data = b2_cpu.data<int32_t>();

        a2_data[0] = 1; a2_data[1] = 2; a2_data[2] = 3;
        b2_data[0] = 10; b2_data[1] = 20; b2_data[2] = 30;

        a2 = a2_cpu.to(device);
        b2 = b2_cpu.to(device);

        print_int32_data("a (same-shape)", a2);
        print_int32_data("b (same-shape)", b2);

        auto c2 = add(a2, b2);
        print_int32_data("a + b (same-shape, Int32)", c2);
        std::cout << "Expected: [11, 22, 33]" << std::endl;
        std::cout << std::endl;

        // Test 3: Broadcast Int32 add (1D broadcast)
        std::cout << "Test 3: Broadcast Int32 Add (scalar to vector)" << std::endl;
        std::cout << "-----------------------------------------------" << std::endl;
        auto a3 = ones({3}, DType::Int32, device);
        auto b3 = ones({1}, DType::Int32, device);

        auto a3_cpu = a3.to(Device::cpu());
        auto b3_cpu = b3.to(Device::cpu());
        auto a3_data = a3_cpu.data<int32_t>();
        auto b3_data = b3_cpu.data<int32_t>();

        a3_data[0] = 1; a3_data[1] = 2; a3_data[2] = 3;
        b3_data[0] = 10;

        a3 = a3_cpu.to(device);
        b3 = b3_cpu.to(device);

        print_int32_data("a", a3);
        print_int32_data("b (broadcast)", b3);

        auto c3 = add(a3, b3);
        print_int32_data("a + b (broadcast, Int32)", c3);
        std::cout << "Expected: [11, 12, 13]" << std::endl;
        std::cout << std::endl;

        // Test 4: 2D Broadcast Int32 add (THE FAILING CASE)
        std::cout << "Test 4: 2D Broadcast Int32 Add (row to matrix)" << std::endl;
        std::cout << "-----------------------------------------------" << std::endl;
        auto a4 = ones({2, 3}, DType::Int32, device);
        auto b4 = ones({1, 3}, DType::Int32, device);

        auto a4_cpu = a4.to(Device::cpu());
        auto b4_cpu = b4.to(Device::cpu());
        auto a4_data = a4_cpu.data<int32_t>();
        auto b4_data = b4_cpu.data<int32_t>();

        for (int i = 0; i < 6; i++) {
            a4_data[i] = i + 1;
        }
        b4_data[0] = 10; b4_data[1] = 20; b4_data[2] = 30;

        a4 = a4_cpu.to(device);
        b4 = b4_cpu.to(device);

        print_int32_data("a (2x3)", a4);
        print_int32_data("b (1x3, broadcast)", b4);

        auto c4 = add(a4, b4);
        print_int32_data("a + b (2D broadcast, Int32)", c4);
        std::cout << "Expected: [11, 22, 33, 14, 25, 36]" << std::endl;
        std::cout << std::endl;

        // Test 5: Same test with Float32 for comparison
        std::cout << "Test 5: 2D Broadcast Float32 Add (for comparison)" << std::endl;
        std::cout << "--------------------------------------------------" << std::endl;
        auto a5 = ones({2, 3}, DType::Float32, device);
        auto b5 = ones({1, 3}, DType::Float32, device);

        auto a5_cpu = a5.to(Device::cpu());
        auto b5_cpu = b5.to(Device::cpu());
        auto a5_data = a5_cpu.data<float>();
        auto b5_data = b5_cpu.data<float>();

        for (int i = 0; i < 6; i++) {
            a5_data[i] = static_cast<float>(i + 1);
        }
        b5_data[0] = 10.0f; b5_data[1] = 20.0f; b5_data[2] = 30.0f;

        a5 = a5_cpu.to(device);
        b5 = b5_cpu.to(device);

        print_float32_data("a (2x3, Float32)", a5);
        print_float32_data("b (1x3, Float32, broadcast)", b5);

        auto c5 = add(a5, b5);
        print_float32_data("a + b (2D broadcast, Float32)", c5);
        std::cout << "Expected: [11, 22, 33, 14, 25, 36]" << std::endl;
        std::cout << std::endl;

        // Summary
        std::cout << "=== Test Summary ===" << std::endl;
        auto verify_result = [](const Tensor& result, const std::vector<int32_t>& expected, const std::string& test_name) {
            auto cpu = result.to(Device::cpu());
            auto data = cpu.data<int32_t>();
            bool pass = true;
            for (size_t i = 0; i < expected.size(); i++) {
                if (data[i] != expected[i]) {
                    pass = false;
                    break;
                }
            }
            std::cout << test_name << ": " << (pass ? "PASS" : "FAIL") << std::endl;
            return pass;
        };

        verify_result(c2, {11, 22, 33}, "Test 2 (same-shape Int32)");
        verify_result(c3, {11, 12, 13}, "Test 3 (1D broadcast Int32)");
        verify_result(c4, {11, 22, 33, 14, 25, 36}, "Test 4 (2D broadcast Int32)");

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
