/**
 * @file test_bmm_fix.cpp
 * @brief Minimal test to verify bmm fix for autograd
 */

#include <iostream>
#include <cmath>
#include "tenzor/tenzor.hpp"

using namespace tenzor;

bool test_basic_bmm() {
    std::cout << "Test 1: Basic BMM forward pass..." << std::endl;

    // Create simple 3D tensors
    Tensor a = ones({2, 3, 4}, DType::Float32, Device::cpu());
    Tensor b = ones({2, 4, 5}, DType::Float32, Device::cpu());

    Tensor result = bmm(a, b);

    // Check shape
    if (result.shape().size() != 3 ||
        result.shape()[0] != 2 ||
        result.shape()[1] != 3 ||
        result.shape()[2] != 5) {
        std::cout << "  FAILED: Incorrect shape" << std::endl;
        return false;
    }

    // Check values (ones @ ones with middle dim=4 should give all 4s)
    auto* data = result.data<float>();
    for (int64_t i = 0; i < result.numel(); ++i) {
        if (std::abs(data[i] - 4.0f) > 1e-5f) {
            std::cout << "  FAILED: Incorrect value at index " << i << ": " << data[i] << std::endl;
            return false;
        }
    }

    std::cout << "  PASSED" << std::endl;
    return true;
}

bool test_bmm_with_autograd() {
    std::cout << "Test 2: BMM with autograd..." << std::endl;

    set_grad_enabled(true);

    // Create variables
    auto a_tensor = ones({1, 2, 3}, DType::Float32, Device::cpu());
    auto b_tensor = ones({1, 3, 2}, DType::Float32, Device::cpu());

    Variable a(a_tensor, true);
    Variable b(b_tensor, true);

    // Forward pass
    auto c = bmm(a, b);

    // Check that gradient function is set
    if (c.grad_fn() == nullptr) {
        std::cout << "  FAILED: grad_fn not set" << std::endl;
        return false;
    }

    // Check shape
    if (c.shape().size() != 3 || c.shape()[0] != 1 || c.shape()[1] != 2 || c.shape()[2] != 2) {
        std::cout << "  FAILED: Incorrect output shape" << std::endl;
        return false;
    }

    // Sum for scalar loss
    auto loss = sum(c);

    // Backward pass - this is where the bug would occur
    try {
        loss.backward();
    } catch (const std::exception& e) {
        std::cout << "  FAILED: Exception during backward: " << e.what() << std::endl;
        return false;
    }

    // Check that gradients are computed
    if (!a.has_grad() || !b.has_grad()) {
        std::cout << "  FAILED: Gradients not computed" << std::endl;
        return false;
    }

    // Verify gradient shapes
    if (a.grad()->shape().size() != 3 ||
        a.grad()->shape()[0] != 1 ||
        a.grad()->shape()[1] != 2 ||
        a.grad()->shape()[2] != 3) {
        std::cout << "  FAILED: Incorrect gradient shape for a" << std::endl;
        return false;
    }

    // Verify gradient values (should all be 2.0)
    auto* grad_a_data = a.grad()->data<float>();
    for (int i = 0; i < 6; ++i) {
        if (std::abs(grad_a_data[i] - 2.0f) > 1e-5f) {
            std::cout << "  FAILED: Incorrect gradient value at index " << i << ": " << grad_a_data[i] << std::endl;
            return false;
        }
    }

    std::cout << "  PASSED" << std::endl;
    return true;
}

bool test_bmm_with_permute() {
    std::cout << "Test 3: BMM with permuted tensors (the original bug)..." << std::endl;

    set_grad_enabled(true);

    // Create variables
    auto a_tensor = ones({2, 3, 4}, DType::Float32, Device::cpu());
    auto b_tensor = ones({2, 4, 5}, DType::Float32, Device::cpu());

    Variable a(a_tensor, true);
    Variable b(b_tensor, true);

    // Forward pass
    auto c = bmm(a, b);
    auto loss = sum(c);

    // Backward pass will call bmm on permuted tensors internally
    // This is where the original "matmul requires 2D tensors" error would occur
    try {
        loss.backward();
    } catch (const std::exception& e) {
        std::cout << "  FAILED: Exception during backward: " << e.what() << std::endl;
        return false;
    }

    // Check that gradients exist and have correct shapes
    if (!a.has_grad() || !b.has_grad()) {
        std::cout << "  FAILED: Gradients not computed" << std::endl;
        return false;
    }

    if (a.grad()->shape()[0] != 2 || a.grad()->shape()[1] != 3 || a.grad()->shape()[2] != 4) {
        std::cout << "  FAILED: Incorrect gradient shape for a" << std::endl;
        return false;
    }

    if (b.grad()->shape()[0] != 2 || b.grad()->shape()[1] != 4 || b.grad()->shape()[2] != 5) {
        std::cout << "  FAILED: Incorrect gradient shape for b" << std::endl;
        return false;
    }

    std::cout << "  PASSED" << std::endl;
    return true;
}

int main() {
    // Initialize Tenzor
    initialize();

    std::cout << "\n=== Testing BMM Fix ===\n" << std::endl;

    bool all_passed = true;

    all_passed &= test_basic_bmm();
    all_passed &= test_bmm_with_autograd();
    all_passed &= test_bmm_with_permute();

    std::cout << "\n=== Summary ===" << std::endl;
    if (all_passed) {
        std::cout << "All tests PASSED!" << std::endl;
        return 0;
    } else {
        std::cout << "Some tests FAILED!" << std::endl;
        return 1;
    }
}
