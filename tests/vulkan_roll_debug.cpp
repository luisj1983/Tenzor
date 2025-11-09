#include "tenzor/tenzor.hpp"
#include <iostream>
#include <vector>

using namespace tenzor;

void print_tensor(const std::string& name, const Tensor& t) {
    auto cpu_t = t.cpu();
    std::cout << name << ": [";
    for (int64_t i = 0; i < cpu_t.numel(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << cpu_t.data<float>()[i];
    }
    std::cout << "] (contiguous: " << t.is_contiguous() << ")\n";
}

int main() {
    std::cout << "=== Vulkan Roll Debug Test ===\n\n";

    // Initialize library to load backends
    initialize();

    // Create test tensor [0, 1, 2, 3, 4]
    std::vector<float> data = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f};
    auto cpu_tensor = full({5}, 0.0f, DType::Float32, Device::cpu());
    for (int i = 0; i < 5; ++i) {
        cpu_tensor.data<float>()[i] = data[i];
    }

    auto vulkan_tensor = cpu_tensor.to(Device::vulkan(0));
    std::cout << "Original tensor:\n";
    print_tensor("  input", vulkan_tensor);

    std::cout << "\n--- Testing slice operation ---\n";
    // Test slice: should get [3, 4]
    auto slice1 = vulkan_tensor.slice(0, 3, 5);
    print_tensor("  slice(0, 3, 5)", slice1);

    // Test slice: should get [0, 1, 2]
    auto slice2 = vulkan_tensor.slice(0, 0, 3);
    print_tensor("  slice(0, 0, 3)", slice2);

    std::cout << "\n--- Testing contiguous on slices ---\n";
    auto slice1_cont = slice1.contiguous();
    print_tensor("  slice1.contiguous()", slice1_cont);

    auto slice2_cont = slice2.contiguous();
    print_tensor("  slice2.contiguous()", slice2_cont);

    std::cout << "\n--- Testing cat with non-contiguous slices ---\n";
    std::vector<Tensor> parts = {slice1, slice2};
    auto cat_result = cat(parts, 0);
    print_tensor("  cat([slice1, slice2], 0)", cat_result);
    std::cout << "  Expected: [3, 4, 0, 1, 2]\n";

    std::cout << "\n--- Testing cat with contiguous slices ---\n";
    std::vector<Tensor> cont_parts = {slice1_cont, slice2_cont};
    auto cat_cont_result = cat(cont_parts, 0);
    print_tensor("  cat([slice1_cont, slice2_cont], 0)", cat_cont_result);
    std::cout << "  Expected: [3, 4, 0, 1, 2]\n";

    std::cout << "\n--- Testing full roll operation ---\n";
    auto rolled = roll(vulkan_tensor, 2, 0);
    print_tensor("  roll(input, 2, 0)", rolled);
    std::cout << "  Expected: [3, 4, 0, 1, 2]\n";

    // Verify
    auto rolled_cpu = rolled.cpu();
    std::vector<float> expected = {3.0f, 4.0f, 0.0f, 1.0f, 2.0f};
    bool success = true;
    for (int i = 0; i < 5; ++i) {
        if (std::abs(rolled_cpu.data<float>()[i] - expected[i]) > 1e-5f) {
            success = false;
            break;
        }
    }

    std::cout << "\n=== Result: " << (success ? "PASS" : "FAIL") << " ===\n";
    return success ? 0 : 1;
}
