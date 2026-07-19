#include <gtest/gtest.h>
#include "tenzor/tenzor.hpp"

using namespace tenzor;

// Helper to check if backend is available
bool isBackendAvailable(Device::Type backend_type, int32_t index = 0) {
    try {
        Device device{backend_type, index};
        auto t = zeros({2, 2}, DType::Float32, device);
        return true;
    } catch (...) {
        return false;
    }
}

// Test that slice operations work correctly on all backends
TEST(SliceBackendParityTest, SliceSubtraction) {
    // Create a simple 2D tensor [N, 4] like boxes
    auto boxes_cpu = zeros({10, 4}, DType::Float32, Device::cpu());
    auto* data = boxes_cpu.data<float>();

    // Fill with test data: each row is [x1, y1, x2, y2]
    for (int i = 0; i < 10; i++) {
        data[i * 4 + 0] = static_cast<float>(i);      // x1
        data[i * 4 + 1] = static_cast<float>(i + 1);  // y1
        data[i * 4 + 2] = static_cast<float>(i + 10); // x2
        data[i * 4 + 3] = static_cast<float>(i + 11); // y2
    }

    // Test slice-based computation on CPU
    auto widths_cpu = boxes_cpu.slice(1, 2, 3) - boxes_cpu.slice(1, 0, 1);   // x2 - x1
    auto heights_cpu = boxes_cpu.slice(1, 3, 4) - boxes_cpu.slice(1, 1, 2);  // y2 - y1
    auto areas_cpu = (widths_cpu * heights_cpu).squeeze(1);

    // Expected: all areas should be 10 * 10 = 100
    ASSERT_EQ(areas_cpu.shape()[0], 10);
    for (int i = 0; i < 10; i++) {
        float area = areas_cpu.data<float>()[i];
        EXPECT_NEAR(area, 100.0f, 1e-5f) << "CPU area mismatch at index " << i;
    }

    // Test on CUDA if available
    if (isBackendAvailable(Device::Type::CUDA)) {
        auto boxes_cuda = boxes_cpu.to(Device::cuda(0));
        auto widths_cuda = boxes_cuda.slice(1, 2, 3) - boxes_cuda.slice(1, 0, 1);
        auto heights_cuda = boxes_cuda.slice(1, 3, 4) - boxes_cuda.slice(1, 1, 2);
        auto areas_cuda = (widths_cuda * heights_cuda).squeeze(1);
        auto areas_cuda_cpu = areas_cuda.to(Device::cpu());

        for (int i = 0; i < 10; i++) {
            float area = areas_cuda_cpu.data<float>()[i];
            EXPECT_NEAR(area, 100.0f, 1e-5f) << "CUDA area mismatch at index " << i;
        }
    }

    // Test on Vulkan if available
    if (isBackendAvailable(Device::Type::Vulkan)) {
        auto boxes_vulkan = boxes_cpu.to(Device::vulkan(0));
        auto widths_vulkan = boxes_vulkan.slice(1, 2, 3) - boxes_vulkan.slice(1, 0, 1);
        auto heights_vulkan = boxes_vulkan.slice(1, 3, 4) - boxes_vulkan.slice(1, 1, 2);
        auto areas_vulkan = (widths_vulkan * heights_vulkan).squeeze(1);
        auto areas_vulkan_cpu = areas_vulkan.to(Device::cpu());

        for (int i = 0; i < 10; i++) {
            float area = areas_vulkan_cpu.data<float>()[i];
            EXPECT_NEAR(area, 100.0f, 1e-5f) << "Vulkan area mismatch at index " << i;
        }
    }

    // Test on ROCm if available
    if (isBackendAvailable(Device::Type::ROCm)) {
        auto boxes_rocm = boxes_cpu.to(Device::rocm(0));
        auto widths_rocm = boxes_rocm.slice(1, 2, 3) - boxes_rocm.slice(1, 0, 1);
        auto heights_rocm = boxes_rocm.slice(1, 3, 4) - boxes_rocm.slice(1, 1, 2);
        auto areas_rocm = (widths_rocm * heights_rocm).squeeze(1);
        auto areas_rocm_cpu = areas_rocm.to(Device::cpu());

        for (int i = 0; i < 10; i++) {
            float area = areas_rocm_cpu.data<float>()[i];
            EXPECT_NEAR(area, 100.0f, 1e-5f) << "ROCm area mismatch at index " << i;
        }
    }

    // Test on OneAPI if available
    if (isBackendAvailable(Device::Type::OneAPI)) {
        auto boxes_oneapi = boxes_cpu.to(Device::oneapi(0));
        auto widths_oneapi = boxes_oneapi.slice(1, 2, 3) - boxes_oneapi.slice(1, 0, 1);
        auto heights_oneapi = boxes_oneapi.slice(1, 3, 4) - boxes_oneapi.slice(1, 1, 2);
        auto areas_oneapi = (widths_oneapi * heights_oneapi).squeeze(1);
        auto areas_oneapi_cpu = areas_oneapi.to(Device::cpu());

        for (int i = 0; i < 10; i++) {
            float area = areas_oneapi_cpu.data<float>()[i];
            EXPECT_NEAR(area, 100.0f, 1e-5f) << "OneAPI area mismatch at index " << i;
        }
    }
}

// Test exact computation used in box_area
TEST(SliceBackendParityTest, BoxAreaComputation) {
    auto boxes_cpu = zeros({5, 4}, DType::Float32, Device::cpu());
    auto* data = boxes_cpu.data<float>();

    // boxes: [x1, y1, x2, y2]
    // Box 0: [0, 0, 10, 10] -> area = 100
    // Box 1: [0, 0, 20, 10] -> area = 200
    // Box 2: [0, 0, 10, 20] -> area = 200
    // Box 3: [5, 5, 15, 15] -> area = 100
    // Box 4: [0, 0, 1, 1] -> area = 1

    float test_boxes[][4] = {
        {0, 0, 10, 10},
        {0, 0, 20, 10},
        {0, 0, 10, 20},
        {5, 5, 15, 15},
        {0, 0, 1, 1}
    };

    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 4; j++) {
            data[i * 4 + j] = test_boxes[i][j];
        }
    }

    float expected_areas[] = {100.0f, 200.0f, 200.0f, 100.0f, 1.0f};

    // CPU computation
    auto widths_cpu = boxes_cpu.slice(1, 2, 3) - boxes_cpu.slice(1, 0, 1);
    auto heights_cpu = boxes_cpu.slice(1, 3, 4) - boxes_cpu.slice(1, 1, 2);
    auto areas_cpu = (widths_cpu * heights_cpu).squeeze(1);

    for (int i = 0; i < 5; i++) {
        EXPECT_NEAR(areas_cpu.data<float>()[i], expected_areas[i], 1e-5f)
            << "CPU box area mismatch at index " << i;
    }

    // CUDA computation
    if (isBackendAvailable(Device::Type::CUDA)) {
        auto boxes_cuda = boxes_cpu.to(Device::cuda(0));
        auto widths_cuda = boxes_cuda.slice(1, 2, 3) - boxes_cuda.slice(1, 0, 1);
        auto heights_cuda = boxes_cuda.slice(1, 3, 4) - boxes_cuda.slice(1, 1, 2);
        auto areas_cuda = (widths_cuda * heights_cuda).squeeze(1);
        auto areas_cuda_cpu = areas_cuda.to(Device::cpu());

        for (int i = 0; i < 5; i++) {
            EXPECT_NEAR(areas_cuda_cpu.data<float>()[i], expected_areas[i], 1e-5f)
                << "CUDA box area mismatch at index " << i;
        }
    }

    // Vulkan computation
    if (isBackendAvailable(Device::Type::Vulkan)) {
        auto boxes_vulkan = boxes_cpu.to(Device::vulkan(0));
        auto widths_vulkan = boxes_vulkan.slice(1, 2, 3) - boxes_vulkan.slice(1, 0, 1);
        auto heights_vulkan = boxes_vulkan.slice(1, 3, 4) - boxes_vulkan.slice(1, 1, 2);
        auto areas_vulkan = (widths_vulkan * heights_vulkan).squeeze(1);
        auto areas_vulkan_cpu = areas_vulkan.to(Device::cpu());

        for (int i = 0; i < 5; i++) {
            EXPECT_NEAR(areas_vulkan_cpu.data<float>()[i], expected_areas[i], 1e-5f)
                << "Vulkan box area mismatch at index " << i;
        }
    }

    // ROCm computation
    if (isBackendAvailable(Device::Type::ROCm)) {
        auto boxes_rocm = boxes_cpu.to(Device::rocm(0));
        auto widths_rocm = boxes_rocm.slice(1, 2, 3) - boxes_rocm.slice(1, 0, 1);
        auto heights_rocm = boxes_rocm.slice(1, 3, 4) - boxes_rocm.slice(1, 1, 2);
        auto areas_rocm = (widths_rocm * heights_rocm).squeeze(1);
        auto areas_rocm_cpu = areas_rocm.to(Device::cpu());

        for (int i = 0; i < 5; i++) {
            EXPECT_NEAR(areas_rocm_cpu.data<float>()[i], expected_areas[i], 1e-5f)
                << "ROCm box area mismatch at index " << i;
        }
    }

    // OneAPI computation
    if (isBackendAvailable(Device::Type::OneAPI)) {
        auto boxes_oneapi = boxes_cpu.to(Device::oneapi(0));
        auto widths_oneapi = boxes_oneapi.slice(1, 2, 3) - boxes_oneapi.slice(1, 0, 1);
        auto heights_oneapi = boxes_oneapi.slice(1, 3, 4) - boxes_oneapi.slice(1, 1, 2);
        auto areas_oneapi = (widths_oneapi * heights_oneapi).squeeze(1);
        auto areas_oneapi_cpu = areas_oneapi.to(Device::cpu());

        for (int i = 0; i < 5; i++) {
            EXPECT_NEAR(areas_oneapi_cpu.data<float>()[i], expected_areas[i], 1e-5f)
                << "OneAPI box area mismatch at index " << i;
        }
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    if (!::testing::GTEST_FLAG(list_tests)) {
        tenzor::initialize();
    }
    return RUN_ALL_TESTS();
}
