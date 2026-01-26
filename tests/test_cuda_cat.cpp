#include "tenzor/core/tensor.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/transform.hpp"
#include "tenzor/tenzor.hpp"
#include <iostream>
#include <cassert>
#include <cmath>

using namespace tenzor;

void test_cuda_cat_1d() {
    std::cout << "Testing 1D CUDA concatenation..." << std::endl;

    // Create tensors on CUDA
    auto a = full({3}, 1.0f, DType::Float32, Device::cuda(0));
    auto b = full({2}, 2.0f, DType::Float32, Device::cuda(0));
    auto c = full({4}, 3.0f, DType::Float32, Device::cuda(0));

    // Concatenate
    std::vector<Tensor> tensors = {a, b, c};
    auto result = cat(std::span<const Tensor>(tensors), 0);

    // Check shape
    assert(result.shape()[0] == 9);

    // Copy to CPU and check values
    auto cpu_result = result.cpu();
    const float* data = cpu_result.data<float>();

    for (int i = 0; i < 3; ++i) {
        assert(std::abs(data[i] - 1.0f) < 1e-5);
    }
    for (int i = 3; i < 5; ++i) {
        assert(std::abs(data[i] - 2.0f) < 1e-5);
    }
    for (int i = 5; i < 9; ++i) {
        assert(std::abs(data[i] - 3.0f) < 1e-5);
    }

    std::cout << "✓ 1D concatenation passed" << std::endl;
}

void test_cuda_cat_2d() {
    std::cout << "Testing 2D CUDA concatenation..." << std::endl;

    // Create tensors on CUDA
    auto a = full({2, 3}, 1.0f, DType::Float32, Device::cuda(0));
    auto b = full({2, 2}, 2.0f, DType::Float32, Device::cuda(0));

    // Concatenate along dimension 1 (columns)
    std::vector<Tensor> tensors = {a, b};
    auto result = cat(std::span<const Tensor>(tensors), 1);

    // Check shape
    assert(result.shape()[0] == 2);
    assert(result.shape()[1] == 5);

    // Copy to CPU and check values
    auto cpu_result = result.cpu();
    const float* data = cpu_result.data<float>();

    // First row: [1, 1, 1, 2, 2]
    for (int i = 0; i < 3; ++i) {
        assert(std::abs(data[i] - 1.0f) < 1e-5);
    }
    for (int i = 3; i < 5; ++i) {
        assert(std::abs(data[i] - 2.0f) < 1e-5);
    }

    // Second row: [1, 1, 1, 2, 2]
    for (int i = 5; i < 8; ++i) {
        assert(std::abs(data[i] - 1.0f) < 1e-5);
    }
    for (int i = 8; i < 10; ++i) {
        assert(std::abs(data[i] - 2.0f) < 1e-5);
    }

    std::cout << "✓ 2D concatenation passed" << std::endl;
}

void test_cuda_cat_2d_dim0() {
    std::cout << "Testing 2D CUDA concatenation along dim 0..." << std::endl;

    // Create tensors on CUDA
    auto a = full({2, 3}, 1.0f, DType::Float32, Device::cuda(0));
    auto b = full({3, 3}, 2.0f, DType::Float32, Device::cuda(0));

    // Concatenate along dimension 0 (rows)
    std::vector<Tensor> tensors = {a, b};
    auto result = cat(std::span<const Tensor>(tensors), 0);

    // Check shape
    assert(result.shape()[0] == 5);
    assert(result.shape()[1] == 3);

    // Copy to CPU and check values
    auto cpu_result = result.cpu();
    const float* data = cpu_result.data<float>();

    // First 2 rows should be 1.0
    for (int i = 0; i < 6; ++i) {
        assert(std::abs(data[i] - 1.0f) < 1e-5);
    }

    // Last 3 rows should be 2.0
    for (int i = 6; i < 15; ++i) {
        assert(std::abs(data[i] - 2.0f) < 1e-5);
    }

    std::cout << "✓ 2D concatenation along dim 0 passed" << std::endl;
}

void test_cuda_cat_3d() {
    std::cout << "Testing 3D CUDA concatenation..." << std::endl;

    // Create tensors on CUDA
    auto a = full({2, 3, 4}, 1.0f, DType::Float32, Device::cuda(0));
    auto b = full({2, 3, 2}, 2.0f, DType::Float32, Device::cuda(0));

    // Concatenate along dimension 2 (depth)
    std::vector<Tensor> tensors = {a, b};
    auto result = cat(std::span<const Tensor>(tensors), 2);

    // Check shape
    assert(result.shape()[0] == 2);
    assert(result.shape()[1] == 3);
    assert(result.shape()[2] == 6);

    // Copy to CPU and verify some values
    auto cpu_result = result.cpu();
    const float* data = cpu_result.data<float>();

    // Check first slice
    for (int i = 0; i < 4; ++i) {
        assert(std::abs(data[i] - 1.0f) < 1e-5);
    }
    for (int i = 4; i < 6; ++i) {
        assert(std::abs(data[i] - 2.0f) < 1e-5);
    }

    std::cout << "✓ 3D concatenation passed" << std::endl;
}

void test_cuda_cat_many_tensors() {
    std::cout << "Testing concatenation of many tensors..." << std::endl;

    // Create 10 tensors
    std::vector<Tensor> tensors;
    for (int i = 0; i < 10; ++i) {
        tensors.push_back(full({5}, static_cast<float>(i), DType::Float32, Device::cuda(0)));
    }

    auto result = cat(std::span<const Tensor>(tensors), 0);

    // Check shape
    assert(result.shape()[0] == 50);

    // Copy to CPU and check values
    auto cpu_result = result.cpu();
    const float* data = cpu_result.data<float>();

    for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < 5; ++j) {
            int idx = i * 5 + j;
            assert(std::abs(data[idx] - static_cast<float>(i)) < 1e-5);
        }
    }

    std::cout << "✓ Many tensors concatenation passed" << std::endl;
}

int main() {
    try {
        // Initialize tenzor library first
        tenzor::initialize();

        // Check if CUDA is available - wrap in try-catch
        // Device initialization will throw if CUDA isn't available
        try {
            auto test_device = Device::cuda(0);
            (void)test_device; // Suppress unused variable warning
        } catch (const std::exception&) {
            std::cout << "CUDA not available, skipping tests" << std::endl;
            return 0;
        }

        std::cout << "Running CUDA concatenation tests..." << std::endl;

        test_cuda_cat_1d();
        test_cuda_cat_2d();
        test_cuda_cat_2d_dim0();
        test_cuda_cat_3d();
        test_cuda_cat_many_tensors();

        std::cout << "\nAll CUDA concatenation tests passed!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
}
