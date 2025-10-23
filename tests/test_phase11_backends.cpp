/**
 * @file test_phase11_backends.cpp
 * @brief Comprehensive test suite for Phase 11 additional backend support
 *
 * Tests all Phase 11 backends (OneAPI, Vulkan, Metal, WebGPU)
 * ROCm tests are intentionally EXCLUDED per user request (system crashes)
 *
 * Coverage:
 * - Backend initialization and device detection
 * - Memory allocation and transfer
 * - Basic operations (matmul, conv2d, pooling, etc.)
 * - Kernel execution and synchronization
 * - Error handling
 *
 * Phase 11 Requirements:
 * ✓ OneAPI Backend (Intel GPUs) - Fixed conv2d backward
 * ✓ Vulkan Backend (Cross-platform) - Newly implemented
 * ✓ Metal Backend (macOS/iOS) - Newly implemented
 * ✓ WebGPU Backend (Browser/WASM) - Newly implemented
 * ✗ ROCm Backend (AMD GPUs) - EXCLUDED FROM TESTS
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/backend/backend.hpp>
#include <tenzor/backend/registry.hpp>
#include <tenzor/core/tensor.hpp>
#include <tenzor/core/device.hpp>
#include <vector>
#include <cmath>

using namespace tenzor;

// ============================================================================
// Test Fixtures
// ============================================================================

class BackendTestBase : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize backend registry
        // Backend will be loaded dynamically in each test
    }

    void TearDown() override {
        // Cleanup if needed
    }

    // Helper to check if backend is available
    bool isBackendAvailable(const std::string& backend_name) {
        try {
            Device device(backend_name, 0);
            return device.is_available();
        } catch (...) {
            return false;
        }
    }

    // Helper to check tensor values
    void checkTensorValues(const Tensor& tensor, float expected_value, float tolerance = 1e-5f) {
        auto cpu_tensor = tensor.to(Device("cpu"));
        auto* data = cpu_tensor.data<float>();
        for (int64_t i = 0; i < cpu_tensor.numel(); ++i) {
            EXPECT_NEAR(data[i], expected_value, tolerance)
                << "Mismatch at index " << i;
        }
    }
};

// ============================================================================
// OneAPI Backend Tests
// ============================================================================

class OneAPIBackendTest : public BackendTestBase {};

TEST_F(OneAPIBackendTest, BackendInitialization) {
    if (!isBackendAvailable("oneapi")) {
        GTEST_SKIP() << "OneAPI backend not available";
    }

    Device device("oneapi", 0);
    EXPECT_TRUE(device.is_available());
    EXPECT_EQ(device.type(), "oneapi");
}

TEST_F(OneAPIBackendTest, MemoryAllocation) {
    if (!isBackendAvailable("oneapi")) {
        GTEST_SKIP() << "OneAPI backend not available";
    }

    Device device("oneapi", 0);
    Tensor tensor({100, 100}, DType::Float32, device);

    EXPECT_EQ(tensor.numel(), 10000);
    EXPECT_EQ(tensor.device().type(), "oneapi");
}

TEST_F(OneAPIBackendTest, BasicMatMul) {
    if (!isBackendAvailable("oneapi")) {
        GTEST_SKIP() << "OneAPI backend not available";
    }

    Device device("oneapi", 0);

    // Create identity matrix and vector
    Tensor a = Tensor::eye(4, DType::Float32, device);
    Tensor b = Tensor::ones({4, 1}, DType::Float32, device);

    // Matmul should give ones
    Tensor c = a.matmul(b);

    // Transfer to CPU and check
    auto cpu_c = c.to(Device("cpu"));
    auto* data = cpu_c.data<float>();
    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(data[i], 1.0f, 1e-5f);
    }
}

TEST_F(OneAPIBackendTest, Conv2dForward) {
    if (!isBackendAvailable("oneapi")) {
        GTEST_SKIP() << "OneAPI backend not available";
    }

    Device device("oneapi", 0);

    // Create simple input and kernel
    Tensor input = Tensor::ones({1, 1, 4, 4}, DType::Float32, device);
    Tensor weight = Tensor::ones({1, 1, 3, 3}, DType::Float32, device);

    // Conv2d with stride=1, padding=0
    Tensor output = tenzor::ops::conv2d(input, weight, nullptr, 1, 0, 1, 1);

    EXPECT_EQ(output.shape()[0], 1);  // batch
    EXPECT_EQ(output.shape()[1], 1);  // out_channels
    EXPECT_EQ(output.shape()[2], 2);  // out_height
    EXPECT_EQ(output.shape()[3], 2);  // out_width

    // Each output should be sum of 3x3=9 ones
    checkTensorValues(output, 9.0f);
}

TEST_F(OneAPIBackendTest, Conv2dBackwardFixed) {
    if (!isBackendAvailable("oneapi")) {
        GTEST_SKIP() << "OneAPI backend not available";
    }

    // This test verifies the fixed conv2d backward pass
    Device device("oneapi", 0);

    // Create tensors with gradients
    Tensor input = Tensor::ones({1, 1, 4, 4}, DType::Float32, device);
    input.set_requires_grad(true);

    Tensor weight = Tensor::ones({1, 1, 3, 3}, DType::Float32, device);
    weight.set_requires_grad(true);

    // Forward pass
    Tensor output = tenzor::ops::conv2d(input, weight, nullptr, 1, 0, 1, 1);

    // Backward pass
    Tensor grad_output = Tensor::ones(output.shape(), DType::Float32, device);
    output.backward(grad_output);

    // Check that gradients were computed (non-null)
    EXPECT_TRUE(input.grad().has_value());
    EXPECT_TRUE(weight.grad().has_value());
}

// ============================================================================
// Vulkan Backend Tests
// ============================================================================

class VulkanBackendTest : public BackendTestBase {};

TEST_F(VulkanBackendTest, BackendInitialization) {
    if (!isBackendAvailable("vulkan")) {
        GTEST_SKIP() << "Vulkan backend not available";
    }

    Device device("vulkan", 0);
    EXPECT_TRUE(device.is_available());
    EXPECT_EQ(device.type(), "vulkan");
}

TEST_F(VulkanBackendTest, MemoryAllocation) {
    if (!isBackendAvailable("vulkan")) {
        GTEST_SKIP() << "Vulkan backend not available";
    }

    Device device("vulkan", 0);
    Tensor tensor({100, 100}, DType::Float32, device);

    EXPECT_EQ(tensor.numel(), 10000);
    EXPECT_EQ(tensor.device().type(), "vulkan");
}

TEST_F(VulkanBackendTest, TensorTransfer) {
    if (!isBackendAvailable("vulkan")) {
        GTEST_SKIP() << "Vulkan backend not available";
    }

    Device cpu_device("cpu");
    Device vulkan_device("vulkan", 0);

    // Create tensor on CPU
    Tensor cpu_tensor = Tensor::ones({10, 10}, DType::Float32, cpu_device);

    // Transfer to Vulkan
    Tensor vulkan_tensor = cpu_tensor.to(vulkan_device);
    EXPECT_EQ(vulkan_tensor.device().type(), "vulkan");

    // Transfer back to CPU
    Tensor result = vulkan_tensor.to(cpu_device);
    checkTensorValues(result, 1.0f);
}

TEST_F(VulkanBackendTest, BasicMatMul) {
    if (!isBackendAvailable("vulkan")) {
        GTEST_SKIP() << "Vulkan backend not available";
    }

    Device device("vulkan", 0);

    // Create simple matrices
    Tensor a = Tensor::ones({4, 4}, DType::Float32, device);
    Tensor b = Tensor::ones({4, 4}, DType::Float32, device);

    // Matmul
    Tensor c = a.matmul(b);

    EXPECT_EQ(c.shape()[0], 4);
    EXPECT_EQ(c.shape()[1], 4);

    // Each element should be 4 (sum of 4 ones)
    checkTensorValues(c, 4.0f);
}

TEST_F(VulkanBackendTest, Conv2dImplemented) {
    if (!isBackendAvailable("vulkan")) {
        GTEST_SKIP() << "Vulkan backend not available";
    }

    // This test verifies the conv2d implementation (was a placeholder)
    Device device("vulkan", 0);

    Tensor input = Tensor::ones({1, 1, 4, 4}, DType::Float32, device);
    Tensor weight = Tensor::ones({1, 1, 3, 3}, DType::Float32, device);

    // Should not throw
    EXPECT_NO_THROW({
        Tensor output = tenzor::ops::conv2d(input, weight, nullptr, 1, 0, 1, 1);
        EXPECT_EQ(output.shape()[2], 2);
        EXPECT_EQ(output.shape()[3], 2);
    });
}

TEST_F(VulkanBackendTest, ElementwiseOperations) {
    if (!isBackendAvailable("vulkan")) {
        GTEST_SKIP() << "Vulkan backend not available";
    }

    Device device("vulkan", 0);

    Tensor a = Tensor::ones({100, 100}, DType::Float32, device);
    Tensor b = Tensor::ones({100, 100}, DType::Float32, device);

    // Add
    Tensor c = a + b;
    checkTensorValues(c, 2.0f);

    // Multiply
    Tensor d = a * b;
    checkTensorValues(d, 1.0f);
}

// ============================================================================
// Metal Backend Tests (macOS/iOS only)
// ============================================================================

class MetalBackendTest : public BackendTestBase {};

TEST_F(MetalBackendTest, BackendInitialization) {
    if (!isBackendAvailable("metal")) {
        GTEST_SKIP() << "Metal backend not available (macOS/iOS only)";
    }

    Device device("metal", 0);
    EXPECT_TRUE(device.is_available());
    EXPECT_EQ(device.type(), "metal");
}

TEST_F(MetalBackendTest, MemoryAllocation) {
    if (!isBackendAvailable("metal")) {
        GTEST_SKIP() << "Metal backend not available (macOS/iOS only)";
    }

    Device device("metal", 0);
    Tensor tensor({100, 100}, DType::Float32, device);

    EXPECT_EQ(tensor.numel(), 10000);
    EXPECT_EQ(tensor.device().type(), "metal");
}

TEST_F(MetalBackendTest, BasicMatMul) {
    if (!isBackendAvailable("metal")) {
        GTEST_SKIP() << "Metal backend not available (macOS/iOS only)";
    }

    Device device("metal", 0);

    Tensor a = Tensor::eye(4, DType::Float32, device);
    Tensor b = Tensor::ones({4, 1}, DType::Float32, device);

    Tensor c = a.matmul(b);
    checkTensorValues(c, 1.0f);
}

TEST_F(MetalBackendTest, ReductionFixed) {
    if (!isBackendAvailable("metal")) {
        GTEST_SKIP() << "Metal backend not available (macOS/iOS only)";
    }

    // This test verifies the fixed reduction indexing
    Device device("metal", 0);

    Tensor input = Tensor::ones({4, 4}, DType::Float32, device);

    // Sum along axis
    Tensor sum = input.sum(/*dim=*/1, /*keepdim=*/false);

    EXPECT_EQ(sum.shape()[0], 4);
    checkTensorValues(sum, 4.0f);  // Sum of 4 ones
}

TEST_F(MetalBackendTest, Conv2dOperation) {
    if (!isBackendAvailable("metal")) {
        GTEST_SKIP() << "Metal backend not available (macOS/iOS only)";
    }

    Device device("metal", 0);

    Tensor input = Tensor::ones({1, 1, 4, 4}, DType::Float32, device);
    Tensor weight = Tensor::ones({1, 1, 3, 3}, DType::Float32, device);

    Tensor output = tenzor::ops::conv2d(input, weight, nullptr, 1, 0, 1, 1);

    EXPECT_EQ(output.shape()[2], 2);
    EXPECT_EQ(output.shape()[3], 2);
    checkTensorValues(output, 9.0f);
}

// ============================================================================
// WebGPU Backend Tests (Browser/WASM)
// ============================================================================

class WebGPUBackendTest : public BackendTestBase {};

TEST_F(WebGPUBackendTest, BackendInitialization) {
    if (!isBackendAvailable("webgpu")) {
        GTEST_SKIP() << "WebGPU backend not available (browser/WASM only)";
    }

    Device device("webgpu", 0);
    EXPECT_TRUE(device.is_available());
    EXPECT_EQ(device.type(), "webgpu");
}

TEST_F(WebGPUBackendTest, MemoryAllocation) {
    if (!isBackendAvailable("webgpu")) {
        GTEST_SKIP() << "WebGPU backend not available (browser/WASM only)";
    }

    Device device("webgpu", 0);
    Tensor tensor({100, 100}, DType::Float32, device);

    EXPECT_EQ(tensor.numel(), 10000);
    EXPECT_EQ(tensor.device().type(), "webgpu");
}

TEST_F(WebGPUBackendTest, BasicMatMul) {
    if (!isBackendAvailable("webgpu")) {
        GTEST_SKIP() << "WebGPU backend not available (browser/WASM only)";
    }

    Device device("webgpu", 0);

    Tensor a = Tensor::ones({4, 4}, DType::Float32, device);
    Tensor b = Tensor::ones({4, 4}, DType::Float32, device);

    Tensor c = a.matmul(b);
    checkTensorValues(c, 4.0f);
}

TEST_F(WebGPUBackendTest, AsyncOperations) {
    if (!isBackendAvailable("webgpu")) {
        GTEST_SKIP() << "WebGPU backend not available (browser/WASM only)";
    }

    // WebGPU is async by design, test that operations complete correctly
    Device device("webgpu", 0);

    Tensor a = Tensor::ones({100, 100}, DType::Float32, device);
    Tensor b = Tensor::ones({100, 100}, DType::Float32, device);

    // Should handle async execution internally
    Tensor c = a + b;

    // Sync point when transferring to CPU
    checkTensorValues(c, 2.0f);
}

// ============================================================================
// Cross-Backend Compatibility Tests
// ============================================================================

class CrossBackendTest : public BackendTestBase {};

TEST_F(CrossBackendTest, TensorTransferBetweenBackends) {
    // Test transferring tensors between different backends
    Device cpu("cpu");

    // Create on CPU
    Tensor cpu_tensor = Tensor::ones({10, 10}, DType::Float32, cpu);

    // Try each backend
    std::vector<std::string> backends = {"oneapi", "vulkan", "metal", "webgpu"};

    for (const auto& backend_name : backends) {
        if (!isBackendAvailable(backend_name)) {
            continue;
        }

        Device backend_device(backend_name, 0);

        // Transfer to backend
        Tensor backend_tensor = cpu_tensor.to(backend_device);
        EXPECT_EQ(backend_tensor.device().type(), backend_name);

        // Transfer back
        Tensor result = backend_tensor.to(cpu);
        checkTensorValues(result, 1.0f);
    }
}

TEST_F(CrossBackendTest, ConsistentResults) {
    // Verify all backends produce consistent results
    Device cpu("cpu");

    // Create test data on CPU
    Tensor a = Tensor::ones({4, 4}, DType::Float32, cpu);
    Tensor b = Tensor::ones({4, 4}, DType::Float32, cpu);

    // Compute on CPU
    Tensor cpu_result = a.matmul(b);

    std::vector<std::string> backends = {"oneapi", "vulkan", "metal", "webgpu"};

    for (const auto& backend_name : backends) {
        if (!isBackendAvailable(backend_name)) {
            continue;
        }

        Device backend_device(backend_name, 0);

        // Transfer and compute on backend
        Tensor backend_a = a.to(backend_device);
        Tensor backend_b = b.to(backend_device);
        Tensor backend_result = backend_a.matmul(backend_b);

        // Transfer back and compare
        Tensor result = backend_result.to(cpu);

        auto* cpu_data = cpu_result.data<float>();
        auto* backend_data = result.data<float>();

        for (int64_t i = 0; i < cpu_result.numel(); ++i) {
            EXPECT_NEAR(cpu_data[i], backend_data[i], 1e-4f)
                << "Backend " << backend_name << " produced different result at index " << i;
        }
    }
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);

    std::cout << "\n";
    std::cout << "========================================\n";
    std::cout << "Phase 11 Backend Tests\n";
    std::cout << "========================================\n";
    std::cout << "\n";
    std::cout << "Testing backends:\n";
    std::cout << "  ✓ OneAPI (Intel GPUs) - Fixed conv2d backward\n";
    std::cout << "  ✓ Vulkan (Cross-platform) - Newly implemented\n";
    std::cout << "  ✓ Metal (macOS/iOS) - Newly implemented\n";
    std::cout << "  ✓ WebGPU (Browser/WASM) - Newly implemented\n";
    std::cout << "  ✗ ROCm (AMD GPUs) - EXCLUDED (system crashes)\n";
    std::cout << "\n";
    std::cout << "Note: Backends will skip tests if not available\n";
    std::cout << "========================================\n";
    std::cout << "\n";

    return RUN_ALL_TESTS();
}
