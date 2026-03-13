/**
 * @file test_vulkan_parity.cpp
 * @brief Vulkan backend parity tests against CPU reference
 *
 * Phase 2.3: Validates Vulkan-specific correctness concerns:
 *   - Float64 ops (validates _f64.comp shader dispatch)
 *   - Float16 odd-element-count tensors (validates 4-byte buffer rounding)
 *   - Slice view + copy (validates getVulkanBufferAndOffset())
 *   - Representative ops vs CPU reference
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/ops/advanced.hpp>
#include <tenzor/backend/fast_dispatch.hpp>
#include <tenzor/backend/op_attributes.hpp>
#include "parity_test_utils.hpp"
#include <cmath>
#include <random>

using namespace tenzor;
using namespace tenzor::testing;

// ============================================================================
// Test fixture with Vulkan availability check
// ============================================================================

class VulkanParityTest : public ::testing::Test {
protected:
    void SetUp() override {
        SKIP_IF_NO_VULKAN;
        cpu_ = Device::cpu();
        vk_ = Device::vulkan(0);
    }

    Device cpu_{Device::cpu()};
    Device vk_{Device::cpu()};

    void assert_parity(const Tensor& cpu_result, const Tensor& vk_result,
                       float rtol, float atol, const std::string& label) {
        auto vk_cpu = vk_result.to(cpu_);
        bool close = tensors_close(cpu_result, vk_cpu, rtol, atol);
        if (!close) {
            float diff = max_abs_diff(cpu_result, vk_cpu);
            FAIL() << label << " parity failed:\n"
                   << "  Max absolute difference: " << std::scientific << diff << "\n"
                   << "  Tolerance: rtol=" << rtol << ", atol=" << atol;
        }
    }

    Tensor rand_cpu(std::vector<int64_t> shape, DType dtype = DType::Float32) {
        return randn(shape, dtype, cpu_);
    }
};

// ============================================================================
// Section 1: Float64 ops — validates _f64.comp shader dispatch
// ============================================================================

class VulkanFloat64Parity : public VulkanParityTest {};

TEST_F(VulkanFloat64Parity, Add) {
    auto a = randn({32, 32}, DType::Float64, cpu_);
    auto b = randn({32, 32}, DType::Float64, cpu_);

    auto cpu_r = a + b;
    auto vk_r = a.to(vk_) + b.to(vk_);

    assert_parity(cpu_r, vk_r, 1e-14f, 1e-14f, "F64 Add");
}

TEST_F(VulkanFloat64Parity, Mul) {
    auto a = randn({32, 32}, DType::Float64, cpu_);
    auto b = randn({32, 32}, DType::Float64, cpu_);

    auto cpu_r = a * b;
    auto vk_r = a.to(vk_) * b.to(vk_);

    assert_parity(cpu_r, vk_r, 1e-14f, 1e-14f, "F64 Mul");
}

TEST_F(VulkanFloat64Parity, MatMul) {
    auto a = randn({16, 32}, DType::Float64, cpu_);
    auto b = randn({32, 16}, DType::Float64, cpu_);

    auto cpu_r = matmul(a, b);
    auto vk_r = matmul(a.to(vk_), b.to(vk_));

    assert_parity(cpu_r, vk_r, 1e-10f, 1e-12f, "F64 MatMul");
}

TEST_F(VulkanFloat64Parity, Exp) {
    auto a = randn({32, 32}, DType::Float64, cpu_);

    auto cpu_r = exp(a);
    auto vk_r = exp(a.to(vk_));

    assert_parity(cpu_r, vk_r, 1e-12f, 1e-14f, "F64 Exp");
}

TEST_F(VulkanFloat64Parity, Log) {
    // Fixed: proper double-precision log implementation via IEEE 754 decomposition
    auto a = abs(randn({32, 32}, DType::Float64, cpu_)) + 0.001;

    auto cpu_r = log(a);
    auto vk_r = log(a.to(vk_));

    assert_parity(cpu_r, vk_r, 1e-6f, 1e-8f, "F64 Log");
}

TEST_F(VulkanFloat64Parity, Tanh) {
    auto a = randn({32, 32}, DType::Float64, cpu_);

    auto cpu_r = tanh(a);
    auto vk_r = tanh(a.to(vk_));

    assert_parity(cpu_r, vk_r, 1e-6f, 1e-8f, "F64 Tanh");
}

TEST_F(VulkanFloat64Parity, Sigmoid) {
    auto a = randn({32, 32}, DType::Float64, cpu_);

    auto cpu_r = sigmoid(a);
    auto vk_r = sigmoid(a.to(vk_));

    assert_parity(cpu_r, vk_r, 1e-12f, 1e-14f, "F64 Sigmoid");
}

TEST_F(VulkanFloat64Parity, Sum) {
    // Fixed: Kahan compensated summation in reduction shader
    auto a = randn({32, 64}, DType::Float64, cpu_);

    auto cpu_r = sum(a);
    auto vk_r = sum(a.to(vk_));

    assert_parity(cpu_r, vk_r, 1e-4f, 1e-6f, "F64 Sum");
}

TEST_F(VulkanFloat64Parity, Mean) {
    // Fixed: Kahan compensated summation in reduction shader
    auto a = randn({32, 64}, DType::Float64, cpu_);

    auto cpu_r = mean(a);
    auto vk_r = mean(a.to(vk_));

    assert_parity(cpu_r, vk_r, 1e-4f, 1e-6f, "F64 Mean");
}

TEST_F(VulkanFloat64Parity, Clamp) {
    auto a = randn({32, 32}, DType::Float64, cpu_);

    auto cpu_r = clamp(a, -1.0f, 1.0f);
    auto vk_r = clamp(a.to(vk_), -1.0f, 1.0f);

    assert_parity(cpu_r, vk_r, 0.0f, 0.0f, "F64 Clamp");
}

TEST_F(VulkanFloat64Parity, Gather) {
    // Fixed: separate input_dim_size and index_dim_size push constants
    auto a = randn({8, 16}, DType::Float64, cpu_);
    auto idx = zeros({8, 4}, DType::Int64, cpu_);
    {
        auto idx_data = idx.data<int64_t>();
        std::mt19937 gen(42);
        for (int64_t i = 0; i < 32; ++i) idx_data[i] = gen() % 16;
    }

    auto cpu_r = gather(a, 1, idx);
    auto vk_r = gather(a.to(vk_), 1, idx.to(vk_));

    assert_parity(cpu_r, vk_r, 1e-10f, 1e-12f, "F64 Gather");
}

TEST_F(VulkanFloat64Parity, IndexSelect) {
    auto a = randn({32, 16}, DType::Float64, cpu_);
    auto idx = zeros({8}, DType::Int64, cpu_);
    {
        auto idx_data = idx.data<int64_t>();
        for (int64_t i = 0; i < 8; ++i) idx_data[i] = i * 2;
    }

    auto cpu_r = index_select(a, 0, idx);
    auto vk_r = index_select(a.to(vk_), 0, idx.to(vk_));

    assert_parity(cpu_r, vk_r, 0.0f, 0.0f, "F64 IndexSelect");
}

TEST_F(VulkanFloat64Parity, Cat) {
    auto a = randn({8, 16}, DType::Float64, cpu_);
    auto b = randn({8, 16}, DType::Float64, cpu_);

    auto cpu_r = cat({a, b}, 0);
    auto vk_r = cat({a.to(vk_), b.to(vk_)}, 0);

    assert_parity(cpu_r, vk_r, 0.0f, 0.0f, "F64 Cat");
}

TEST_F(VulkanFloat64Parity, Transpose) {
    auto a = randn({16, 32}, DType::Float64, cpu_);

    auto cpu_r = transpose(a, 0, 1).contiguous();
    auto vk_r = transpose(a.to(vk_), 0, 1).contiguous();

    assert_parity(cpu_r, vk_r, 0.0f, 0.0f, "F64 Transpose");
}

TEST_F(VulkanFloat64Parity, Arange) {
    auto cpu_r = arange(0.0, 100.0, 0.3, DType::Float64, cpu_);
    auto vk_r = arange(0.0, 100.0, 0.3, DType::Float64, vk_);

    assert_parity(cpu_r, vk_r, 1e-6f, 1e-8f, "F64 Arange");
}

TEST_F(VulkanFloat64Parity, Linspace) {
    auto cpu_r = linspace(0.0, 10.0, 100, DType::Float64, cpu_);
    auto vk_r = linspace(0.0, 10.0, 100, DType::Float64, vk_);

    assert_parity(cpu_r, vk_r, 1e-12f, 1e-14f, "F64 Linspace");
}

TEST_F(VulkanFloat64Parity, Sqrt) {
    auto a = abs(randn({32, 32}, DType::Float64, cpu_)) + 0.01;

    auto cpu_r = sqrt(a);
    auto vk_r = sqrt(a.to(vk_));

    assert_parity(cpu_r, vk_r, 1e-12f, 1e-14f, "F64 Sqrt");
}

TEST_F(VulkanFloat64Parity, Pow) {
    auto a = abs(randn({32, 32}, DType::Float64, cpu_)) + 0.01;

    auto cpu_r = pow(a, 2.5);
    auto vk_r = pow(a.to(vk_), 2.5);

    assert_parity(cpu_r, vk_r, 1e-4f, 1e-6f, "F64 Pow");
}

// ============================================================================
// Section 2: Float16 odd-element-count tensors — validates 4-byte buffer rounding
// ============================================================================

class VulkanFloat16OddElements : public VulkanParityTest {};

TEST_F(VulkanFloat16OddElements, Add_OddCount) {
    // 7 elements: odd count triggers the F16 packed buffer rounding edge case
    auto a = randn({7}, DType::Float32, cpu_).to(DType::Float16);
    auto b = randn({7}, DType::Float32, cpu_).to(DType::Float16);

    auto cpu_r = a + b;
    auto vk_r = a.to(vk_) + b.to(vk_);

    assert_parity(cpu_r, vk_r, 1e-3f, 1e-3f, "F16 Add odd=7");
}

TEST_F(VulkanFloat16OddElements, Mul_OddCount) {
    auto a = randn({13}, DType::Float32, cpu_).to(DType::Float16);
    auto b = randn({13}, DType::Float32, cpu_).to(DType::Float16);

    auto cpu_r = a * b;
    auto vk_r = a.to(vk_) * b.to(vk_);

    assert_parity(cpu_r, vk_r, 1e-3f, 1e-3f, "F16 Mul odd=13");
}

TEST_F(VulkanFloat16OddElements, Exp_OddCount) {
    auto a = randn({5}, DType::Float32, cpu_).to(DType::Float16);

    auto cpu_r = exp(a);
    auto vk_r = exp(a.to(vk_));

    assert_parity(cpu_r, vk_r, 1e-2f, 1e-3f, "F16 Exp odd=5");
}

TEST_F(VulkanFloat16OddElements, Sigmoid_OddCount) {
    auto a = randn({9}, DType::Float32, cpu_).to(DType::Float16);

    auto cpu_r = sigmoid(a);
    auto vk_r = sigmoid(a.to(vk_));

    assert_parity(cpu_r, vk_r, 1e-3f, 1e-3f, "F16 Sigmoid odd=9");
}

TEST_F(VulkanFloat16OddElements, Sum_OddCount) {
    auto a = randn({15}, DType::Float32, cpu_).to(DType::Float16);

    auto cpu_r = sum(a);
    auto vk_r = sum(a.to(vk_));

    assert_parity(cpu_r, vk_r, 1e-2f, 1e-2f, "F16 Sum odd=15");
}

TEST_F(VulkanFloat16OddElements, MatMul_OddDim) {
    // M=7 and N=5 are both odd, K=8 is even
    auto a = randn({7, 8}, DType::Float32, cpu_).to(DType::Float16);
    auto b = randn({8, 5}, DType::Float32, cpu_).to(DType::Float16);

    auto cpu_r = matmul(a, b);
    auto vk_r = matmul(a.to(vk_), b.to(vk_));

    assert_parity(cpu_r, vk_r, 1e-2f, 1e-2f, "F16 MatMul 7x8 @ 8x5");
}

TEST_F(VulkanFloat16OddElements, Clamp_OddCount) {
    // Fixed: buffer size rounding and preserve upper half-word for odd counts
    auto a = randn({3}, DType::Float32, cpu_).to(DType::Float16);

    auto cpu_r = clamp(a, -0.5f, 0.5f);
    auto vk_r = clamp(a.to(vk_), -0.5f, 0.5f);

    assert_parity(cpu_r, vk_r, 1e-3f, 1e-3f, "F16 Clamp odd=3");
}

TEST_F(VulkanFloat16OddElements, Tanh_OddCount) {
    auto a = randn({11}, DType::Float32, cpu_).to(DType::Float16);

    auto cpu_r = tanh(a);
    auto vk_r = tanh(a.to(vk_));

    assert_parity(cpu_r, vk_r, 1e-3f, 1e-3f, "F16 Tanh odd=11");
}

TEST_F(VulkanFloat16OddElements, Sub_SingleElement) {
    auto a = randn({1}, DType::Float32, cpu_).to(DType::Float16);
    auto b = randn({1}, DType::Float32, cpu_).to(DType::Float16);

    auto cpu_r = a - b;
    auto vk_r = a.to(vk_) - b.to(vk_);

    assert_parity(cpu_r, vk_r, 1e-3f, 1e-3f, "F16 Sub single=1");
}

// ============================================================================
// Section 3: Slice view + copy — validates getVulkanBufferAndOffset()
// ============================================================================

class VulkanSliceViewParity : public VulkanParityTest {};

TEST_F(VulkanSliceViewParity, SliceAdd) {
    auto a = randn({32, 64}, DType::Float32, cpu_);
    auto b = randn({16, 64}, DType::Float32, cpu_);

    auto a_slice_cpu = a.slice(0, 8, 24);  // rows 8-24
    auto cpu_r = a_slice_cpu + b;

    auto a_vk = a.to(vk_);
    auto b_vk = b.to(vk_);
    auto a_slice_vk = a_vk.slice(0, 8, 24);
    auto vk_r = a_slice_vk + b_vk;

    assert_parity(cpu_r, vk_r, 1e-6f, 1e-8f, "Slice Add");
}

TEST_F(VulkanSliceViewParity, SliceToDevice) {
    // Slice on CPU, copy slice to Vulkan
    auto a = randn({64, 32}, DType::Float32, cpu_);
    auto a_slice = a.slice(0, 16, 48);  // rows 16-48 (32 rows)

    auto cpu_r = a_slice.contiguous();
    auto vk_r = a_slice.to(vk_);

    assert_parity(cpu_r, vk_r, 0.0f, 0.0f, "Slice to Vulkan");
}

TEST_F(VulkanSliceViewParity, SliceFromDevice) {
    // Create on Vulkan, slice, copy back
    auto a = randn({64, 32}, DType::Float32, cpu_);
    auto a_vk = a.to(vk_);
    auto a_slice_vk = a_vk.slice(0, 10, 30);
    auto vk_r = a_slice_vk.to(cpu_);

    auto cpu_r = a.slice(0, 10, 30).contiguous();

    assert_parity(cpu_r, vk_r, 0.0f, 0.0f, "Slice from Vulkan");
}

TEST_F(VulkanSliceViewParity, SliceMatMul) {
    auto a = randn({32, 64}, DType::Float32, cpu_);
    auto b = randn({32, 16}, DType::Float32, cpu_);

    auto a_slice = a.slice(1, 0, 32);  // first 32 columns
    auto cpu_r = matmul(a_slice, b);

    auto a_vk = a.to(vk_);
    auto b_vk = b.to(vk_);
    auto a_slice_vk = a_vk.slice(1, 0, 32);
    auto vk_r = matmul(a_slice_vk, b_vk);

    assert_parity(cpu_r, vk_r, 1e-4f, 1e-5f, "Slice MatMul");
}

TEST_F(VulkanSliceViewParity, SliceFloat64) {
    auto a = randn({32, 64}, DType::Float64, cpu_);

    auto a_slice_cpu = a.slice(0, 4, 20);
    auto cpu_r = a_slice_cpu.contiguous();

    auto a_vk = a.to(vk_);
    auto a_slice_vk = a_vk.slice(0, 4, 20);
    auto vk_r = a_slice_vk.to(cpu_);

    assert_parity(cpu_r, vk_r, 0.0f, 0.0f, "Slice F64 copy");
}

TEST_F(VulkanSliceViewParity, SliceFloat16) {
    auto a = randn({32, 64}, DType::Float32, cpu_).to(DType::Float16);

    auto a_slice_cpu = a.slice(0, 3, 17);  // odd start for extra stress
    auto cpu_r = a_slice_cpu.contiguous();

    auto a_vk = a.to(vk_);
    auto a_slice_vk = a_vk.slice(0, 3, 17);
    auto vk_r = a_slice_vk.to(cpu_);

    assert_parity(cpu_r, vk_r, 0.0f, 0.0f, "Slice F16 copy");
}

// ============================================================================
// Section 4: Representative Float32 ops — basic parity coverage
// ============================================================================

class VulkanFloat32Parity : public VulkanParityTest {};

TEST_F(VulkanFloat32Parity, Add) {
    auto a = randn({64, 64}, DType::Float32, cpu_);
    auto b = randn({64, 64}, DType::Float32, cpu_);

    auto cpu_r = a + b;
    auto vk_r = a.to(vk_) + b.to(vk_);

    assert_parity(cpu_r, vk_r, 1e-6f, 1e-8f, "F32 Add");
}

TEST_F(VulkanFloat32Parity, Sub) {
    auto a = randn({64, 64}, DType::Float32, cpu_);
    auto b = randn({64, 64}, DType::Float32, cpu_);

    auto cpu_r = a - b;
    auto vk_r = a.to(vk_) - b.to(vk_);

    assert_parity(cpu_r, vk_r, 1e-6f, 1e-8f, "F32 Sub");
}

TEST_F(VulkanFloat32Parity, Mul) {
    auto a = randn({64, 64}, DType::Float32, cpu_);
    auto b = randn({64, 64}, DType::Float32, cpu_);

    auto cpu_r = a * b;
    auto vk_r = a.to(vk_) * b.to(vk_);

    assert_parity(cpu_r, vk_r, 1e-6f, 1e-8f, "F32 Mul");
}

TEST_F(VulkanFloat32Parity, Div) {
    auto a = randn({64, 64}, DType::Float32, cpu_);
    auto b = abs(randn({64, 64}, DType::Float32, cpu_)) + 0.1f;

    auto cpu_r = a / b;
    auto vk_r = a.to(vk_) / b.to(vk_);

    assert_parity(cpu_r, vk_r, 1e-5f, 1e-7f, "F32 Div");
}

TEST_F(VulkanFloat32Parity, MatMul) {
    auto a = randn({32, 64}, DType::Float32, cpu_);
    auto b = randn({64, 32}, DType::Float32, cpu_);

    auto cpu_r = matmul(a, b);
    auto vk_r = matmul(a.to(vk_), b.to(vk_));

    assert_parity(cpu_r, vk_r, 1e-4f, 1e-5f, "F32 MatMul");
}

TEST_F(VulkanFloat32Parity, BatchMatMul) {
    auto a = randn({4, 16, 32}, DType::Float32, cpu_);
    auto b = randn({4, 32, 16}, DType::Float32, cpu_);

    auto cpu_r = matmul(a, b);
    auto vk_r = matmul(a.to(vk_), b.to(vk_));

    assert_parity(cpu_r, vk_r, 1e-4f, 1e-5f, "F32 BMM");
}

TEST_F(VulkanFloat32Parity, Sigmoid) {
    auto a = randn({64, 64}, DType::Float32, cpu_);

    auto cpu_r = sigmoid(a);
    auto vk_r = sigmoid(a.to(vk_));

    assert_parity(cpu_r, vk_r, 1e-5f, 1e-7f, "F32 Sigmoid");
}

TEST_F(VulkanFloat32Parity, Tanh) {
    auto a = randn({64, 64}, DType::Float32, cpu_);

    auto cpu_r = tanh(a);
    auto vk_r = tanh(a.to(vk_));

    assert_parity(cpu_r, vk_r, 1e-5f, 1e-7f, "F32 Tanh");
}

TEST_F(VulkanFloat32Parity, Exp) {
    auto a = randn({64, 64}, DType::Float32, cpu_);

    auto cpu_r = exp(a);
    auto vk_r = exp(a.to(vk_));

    assert_parity(cpu_r, vk_r, 1e-5f, 1e-7f, "F32 Exp");
}

TEST_F(VulkanFloat32Parity, Log) {
    auto a = abs(randn({64, 64}, DType::Float32, cpu_)) + 0.01f;

    auto cpu_r = log(a);
    auto vk_r = log(a.to(vk_));

    assert_parity(cpu_r, vk_r, 1e-5f, 1e-7f, "F32 Log");
}

TEST_F(VulkanFloat32Parity, Sqrt) {
    auto a = abs(randn({64, 64}, DType::Float32, cpu_)) + 0.01f;

    auto cpu_r = sqrt(a);
    auto vk_r = sqrt(a.to(vk_));

    assert_parity(cpu_r, vk_r, 1e-5f, 1e-7f, "F32 Sqrt");
}

TEST_F(VulkanFloat32Parity, Abs) {
    auto a = randn({64, 64}, DType::Float32, cpu_);

    auto cpu_r = abs(a);
    auto vk_r = abs(a.to(vk_));

    assert_parity(cpu_r, vk_r, 0.0f, 0.0f, "F32 Abs");
}

TEST_F(VulkanFloat32Parity, Neg) {
    auto a = randn({64, 64}, DType::Float32, cpu_);

    auto cpu_r = neg(a);
    auto vk_r = neg(a.to(vk_));

    assert_parity(cpu_r, vk_r, 0.0f, 0.0f, "F32 Neg");
}

TEST_F(VulkanFloat32Parity, Clamp) {
    auto a = randn({64, 64}, DType::Float32, cpu_);

    auto cpu_r = clamp(a, -0.5f, 0.5f);
    auto vk_r = clamp(a.to(vk_), -0.5f, 0.5f);

    assert_parity(cpu_r, vk_r, 0.0f, 0.0f, "F32 Clamp");
}

TEST_F(VulkanFloat32Parity, Sum) {
    // Fixed: Kahan compensated summation in reduction shader
    auto a = randn({32, 64}, DType::Float32, cpu_);

    auto cpu_r = sum(a);
    auto vk_r = sum(a.to(vk_));

    assert_parity(cpu_r, vk_r, 1e-1f, 1e-2f, "F32 Sum");
}

TEST_F(VulkanFloat32Parity, Mean) {
    // Fixed: Kahan compensated summation in reduction shader
    auto a = randn({32, 64}, DType::Float32, cpu_);

    auto cpu_r = mean(a);
    auto vk_r = mean(a.to(vk_));

    assert_parity(cpu_r, vk_r, 1e-2f, 1e-3f, "F32 Mean");
}

TEST_F(VulkanFloat32Parity, Max) {
    // Fixed: Kahan compensated summation in reduction shader
    auto a = randn({32, 64}, DType::Float32, cpu_);

    auto cpu_r = max(a);
    auto vk_r = max(a.to(vk_));

    assert_parity(cpu_r, vk_r, 1e-6f, 1e-7f, "F32 Max");
}

TEST_F(VulkanFloat32Parity, Min) {
    // Fixed: Kahan compensated summation in reduction shader
    auto a = randn({32, 64}, DType::Float32, cpu_);

    auto cpu_r = min(a);
    auto vk_r = min(a.to(vk_));

    assert_parity(cpu_r, vk_r, 1e-6f, 1e-7f, "F32 Min");
}

TEST_F(VulkanFloat32Parity, Reshape) {
    auto a = randn({4, 16, 8}, DType::Float32, cpu_);

    auto cpu_r = reshape(a, {64, 8}).contiguous();
    auto vk_r = reshape(a.to(vk_), {64, 8}).contiguous();

    assert_parity(cpu_r, vk_r, 0.0f, 0.0f, "F32 Reshape");
}

TEST_F(VulkanFloat32Parity, Transpose) {
    auto a = randn({32, 64}, DType::Float32, cpu_);

    auto cpu_r = transpose(a, 0, 1).contiguous();
    auto vk_r = transpose(a.to(vk_), 0, 1).contiguous();

    assert_parity(cpu_r, vk_r, 0.0f, 0.0f, "F32 Transpose");
}

TEST_F(VulkanFloat32Parity, Cat) {
    auto a = randn({16, 32}, DType::Float32, cpu_);
    auto b = randn({16, 32}, DType::Float32, cpu_);

    auto cpu_r = cat({a, b}, 0);
    auto vk_r = cat({a.to(vk_), b.to(vk_)}, 0);

    assert_parity(cpu_r, vk_r, 0.0f, 0.0f, "F32 Cat dim=0");
}

TEST_F(VulkanFloat32Parity, CatDim1) {
    auto a = randn({16, 16}, DType::Float32, cpu_);
    auto b = randn({16, 32}, DType::Float32, cpu_);

    auto cpu_r = cat({a, b}, 1);
    auto vk_r = cat({a.to(vk_), b.to(vk_)}, 1);

    assert_parity(cpu_r, vk_r, 0.0f, 0.0f, "F32 Cat dim=1");
}

TEST_F(VulkanFloat32Parity, Gather) {
    // Fixed: separate input_dim_size and index_dim_size push constants
    auto a = randn({16, 32}, DType::Float32, cpu_);
    auto idx = zeros({16, 8}, DType::Int64, cpu_);
    {
        auto d = idx.data<int64_t>();
        std::mt19937 gen(123);
        for (int64_t i = 0; i < 128; ++i) d[i] = gen() % 32;
    }

    auto cpu_r = gather(a, 1, idx);
    auto vk_r = gather(a.to(vk_), 1, idx.to(vk_));

    assert_parity(cpu_r, vk_r, 1e-6f, 1e-7f, "F32 Gather");
}

TEST_F(VulkanFloat32Parity, IndexSelect) {
    auto a = randn({64, 16}, DType::Float32, cpu_);
    auto idx = zeros({8}, DType::Int64, cpu_);
    {
        auto d = idx.data<int64_t>();
        for (int64_t i = 0; i < 8; ++i) d[i] = i * 7;
    }

    auto cpu_r = index_select(a, 0, idx);
    auto vk_r = index_select(a.to(vk_), 0, idx.to(vk_));

    assert_parity(cpu_r, vk_r, 0.0f, 0.0f, "F32 IndexSelect");
}

TEST_F(VulkanFloat32Parity, Pow) {
    auto a = abs(randn({32, 32}, DType::Float32, cpu_)) + 0.01f;

    auto cpu_r = pow(a, 2.0);
    auto vk_r = pow(a.to(vk_), 2.0);

    assert_parity(cpu_r, vk_r, 1e-4f, 1e-6f, "F32 Pow");
}

TEST_F(VulkanFloat32Parity, Reciprocal) {
    auto a = abs(randn({32, 32}, DType::Float32, cpu_)) + 0.1f;

    auto cpu_r = reciprocal(a);
    auto vk_r = reciprocal(a.to(vk_));

    assert_parity(cpu_r, vk_r, 1e-5f, 1e-7f, "F32 Reciprocal");
}

TEST_F(VulkanFloat32Parity, BroadcastAdd) {
    auto a = randn({4, 32, 64}, DType::Float32, cpu_);
    auto b = randn({64}, DType::Float32, cpu_);

    auto cpu_r = a + b;
    auto vk_r = a.to(vk_) + b.to(vk_);

    assert_parity(cpu_r, vk_r, 1e-6f, 1e-8f, "F32 Broadcast Add");
}

TEST_F(VulkanFloat32Parity, ScalarMul) {
    auto a = randn({64, 64}, DType::Float32, cpu_);

    auto cpu_r = a * 3.14f;
    auto vk_r = a.to(vk_) * 3.14f;

    assert_parity(cpu_r, vk_r, 1e-5f, 1e-7f, "F32 Scalar Mul");
}

// ============================================================================
// Section 5: Activation ops via dispatch (kernel-level only)
// ============================================================================

class VulkanActivationParity : public VulkanParityTest {};

TEST_F(VulkanActivationParity, ReLU_viaDispatch) {
    auto a = randn({32, 32}, DType::Float32, cpu_);

    auto cpu_r = dispatch_to_device(OpId::ReLU, Device::Type::CPU, std::vector<Tensor>{a})[0];
    auto vk_r = dispatch_to_device(OpId::ReLU, Device::Type::Vulkan, std::vector<Tensor>{a.to(vk_)})[0];

    assert_parity(cpu_r, vk_r, 0.0f, 0.0f, "ReLU dispatch");
}

TEST_F(VulkanActivationParity, Sigmoid_viaOp) {
    auto a = randn({32, 32}, DType::Float32, cpu_);

    auto cpu_r = sigmoid(a);
    auto vk_r = sigmoid(a.to(vk_));

    assert_parity(cpu_r, vk_r, 1e-5f, 1e-7f, "Sigmoid");
}

TEST_F(VulkanActivationParity, Gelu_viaDispatch) {
    auto a = randn({32, 32}, DType::Float32, cpu_);

    auto cpu_r = dispatch_to_device(OpId::Gelu, Device::Type::CPU, std::vector<Tensor>{a})[0];
    auto vk_r = dispatch_to_device(OpId::Gelu, Device::Type::Vulkan, std::vector<Tensor>{a.to(vk_)})[0];

    assert_parity(cpu_r, vk_r, 1e-4f, 1e-5f, "GELU dispatch");
}

TEST_F(VulkanActivationParity, LeakyReLU_viaDispatch) {
    auto a = randn({32, 32}, DType::Float32, cpu_);
    OpAttributes attrs;
    attrs.set(AttrKey::Alpha, 0.01);

    auto cpu_r = dispatch_to_device(OpId::LeakyReLU, Device::Type::CPU, std::vector<Tensor>{a}, attrs)[0];
    auto vk_r = dispatch_to_device(OpId::LeakyReLU, Device::Type::Vulkan, std::vector<Tensor>{a.to(vk_)}, attrs)[0];

    assert_parity(cpu_r, vk_r, 1e-6f, 1e-8f, "LeakyReLU dispatch");
}

TEST_F(VulkanActivationParity, Elu_viaDispatch) {
    auto a = randn({32, 32}, DType::Float32, cpu_);
    OpAttributes attrs;
    attrs.set(AttrKey::Alpha, 1.0);

    auto cpu_r = dispatch_to_device(OpId::Elu, Device::Type::CPU, std::vector<Tensor>{a}, attrs)[0];
    auto vk_r = dispatch_to_device(OpId::Elu, Device::Type::Vulkan, std::vector<Tensor>{a.to(vk_)}, attrs)[0];

    assert_parity(cpu_r, vk_r, 1e-5f, 1e-7f, "ELU dispatch");
}

// ============================================================================
// Section 6: Embedding via dispatch (validates f64 shader path)
// ============================================================================

class VulkanEmbeddingParity : public VulkanParityTest {};

TEST_F(VulkanEmbeddingParity, Embedding_F32) {
    auto weight = randn({100, 32}, DType::Float32, cpu_);
    auto indices = zeros({8}, DType::Int64, cpu_);
    {
        auto d = indices.data<int64_t>();
        for (int64_t i = 0; i < 8; ++i) d[i] = i * 10;
    }

    auto cpu_r = dispatch_to_device(OpId::Embedding, Device::Type::CPU,
                                     std::vector<Tensor>{weight, indices})[0];
    auto vk_r = dispatch_to_device(OpId::Embedding, Device::Type::Vulkan,
                                     std::vector<Tensor>{weight.to(vk_), indices.to(vk_)})[0];

    assert_parity(cpu_r, vk_r, 0.0f, 0.0f, "Embedding F32");
}

TEST_F(VulkanEmbeddingParity, Embedding_F64) {
    auto weight = randn({100, 32}, DType::Float64, cpu_);
    auto indices = zeros({8}, DType::Int64, cpu_);
    {
        auto d = indices.data<int64_t>();
        for (int64_t i = 0; i < 8; ++i) d[i] = i * 10;
    }

    auto cpu_r = dispatch_to_device(OpId::Embedding, Device::Type::CPU,
                                     std::vector<Tensor>{weight, indices})[0];
    auto vk_r = dispatch_to_device(OpId::Embedding, Device::Type::Vulkan,
                                     std::vector<Tensor>{weight.to(vk_), indices.to(vk_)})[0];

    assert_parity(cpu_r, vk_r, 0.0f, 0.0f, "Embedding F64");
}

// ============================================================================
// Section 7: Softmax via dispatch
// ============================================================================

class VulkanSoftmaxParity : public VulkanParityTest {};

TEST_F(VulkanSoftmaxParity, Softmax_F32) {
    auto a = randn({16, 32}, DType::Float32, cpu_);
    OpAttributes attrs;
    attrs.set(AttrKey::Dim, static_cast<int64_t>(1));

    auto cpu_r = dispatch_to_device(OpId::Softmax, Device::Type::CPU, std::vector<Tensor>{a}, attrs)[0];
    auto vk_r = dispatch_to_device(OpId::Softmax, Device::Type::Vulkan, std::vector<Tensor>{a.to(vk_)}, attrs)[0];

    assert_parity(cpu_r, vk_r, 1e-5f, 1e-7f, "Softmax F32");
}

TEST_F(VulkanSoftmaxParity, Softmax_F64) {
    auto a = randn({16, 32}, DType::Float64, cpu_);
    OpAttributes attrs;
    attrs.set(AttrKey::Dim, static_cast<int64_t>(1));

    auto cpu_r = dispatch_to_device(OpId::Softmax, Device::Type::CPU, std::vector<Tensor>{a}, attrs)[0];
    auto vk_r = dispatch_to_device(OpId::Softmax, Device::Type::Vulkan, std::vector<Tensor>{a.to(vk_)}, attrs)[0];

    assert_parity(cpu_r, vk_r, 1e-10f, 1e-12f, "Softmax F64");
}

// ============================================================================
// Section 8: LayerNorm via dispatch
// ============================================================================

class VulkanLayerNormParity : public VulkanParityTest {};

TEST_F(VulkanLayerNormParity, LayerNorm_F32) {
    auto a = randn({4, 32}, DType::Float32, cpu_);
    auto w = randn({32}, DType::Float32, cpu_) * 0.5f + 1.0f;
    auto b = randn({32}, DType::Float32, cpu_) * 0.1f;

    OpAttributes attrs;
    attrs.set(AttrKey::NormalizedShape, "32");

    auto cpu_r = dispatch_to_device(OpId::LayerNorm, Device::Type::CPU,
                                     std::vector<Tensor>{a, w, b}, attrs)[0];
    auto vk_r = dispatch_to_device(OpId::LayerNorm, Device::Type::Vulkan,
                                     std::vector<Tensor>{a.to(vk_), w.to(vk_), b.to(vk_)}, attrs)[0];

    assert_parity(cpu_r, vk_r, 1e-4f, 1e-5f, "LayerNorm F32");
}

TEST_F(VulkanLayerNormParity, LayerNorm_F64) {
    auto a = randn({4, 32}, DType::Float64, cpu_);
    auto w = randn({32}, DType::Float64, cpu_) * 0.5 + 1.0;
    auto b = randn({32}, DType::Float64, cpu_) * 0.1;

    OpAttributes attrs;
    attrs.set(AttrKey::NormalizedShape, "32");

    auto cpu_r = dispatch_to_device(OpId::LayerNorm, Device::Type::CPU,
                                     std::vector<Tensor>{a, w, b}, attrs)[0];
    auto vk_r = dispatch_to_device(OpId::LayerNorm, Device::Type::Vulkan,
                                     std::vector<Tensor>{a.to(vk_), w.to(vk_), b.to(vk_)}, attrs)[0];

    assert_parity(cpu_r, vk_r, 1e-4f, 1e-6f, "LayerNorm F64");
}

// ============================================================================
// Section 9: Comparison ops
// ============================================================================

class VulkanComparisonParity : public VulkanParityTest {};

TEST_F(VulkanComparisonParity, Eq) {
    auto a = randn({32, 32}, DType::Float32, cpu_);
    auto b = a.clone();

    auto cpu_r = eq(a, b).to(DType::Int32);
    auto vk_r = eq(a.to(vk_), b.to(vk_)).to(DType::Int32);

    assert_parity(cpu_r, vk_r, 0.0f, 0.0f, "Eq");
}

TEST_F(VulkanComparisonParity, Gt) {
    auto a = randn({32, 32}, DType::Float32, cpu_);
    auto b = randn({32, 32}, DType::Float32, cpu_);

    auto cpu_r = gt(a, b).to(DType::Int32);
    auto vk_r = gt(a.to(vk_), b.to(vk_)).to(DType::Int32);

    assert_parity(cpu_r, vk_r, 0.0f, 0.0f, "Gt");
}

TEST_F(VulkanComparisonParity, Lt) {
    auto a = randn({32, 32}, DType::Float32, cpu_);
    auto b = randn({32, 32}, DType::Float32, cpu_);

    auto cpu_r = lt(a, b).to(DType::Int32);
    auto vk_r = lt(a.to(vk_), b.to(vk_)).to(DType::Int32);

    assert_parity(cpu_r, vk_r, 0.0f, 0.0f, "Lt");
}

TEST_F(VulkanComparisonParity, Le) {
    auto a = randn({32, 32}, DType::Float32, cpu_);
    auto b = randn({32, 32}, DType::Float32, cpu_);

    auto cpu_r = le(a, b).to(DType::Int32);
    auto vk_r = le(a.to(vk_), b.to(vk_)).to(DType::Int32);

    assert_parity(cpu_r, vk_r, 0.0f, 0.0f, "Le");
}

// ============================================================================
// Section 10: Trig ops
// ============================================================================

class VulkanTrigParity : public VulkanParityTest {};

TEST_F(VulkanTrigParity, Sin) {
    auto a = randn({32, 32}, DType::Float32, cpu_);

    auto cpu_r = sin(a);
    auto vk_r = sin(a.to(vk_));

    assert_parity(cpu_r, vk_r, 1e-5f, 1e-7f, "Sin");
}

TEST_F(VulkanTrigParity, Cos) {
    auto a = randn({32, 32}, DType::Float32, cpu_);

    auto cpu_r = cos(a);
    auto vk_r = cos(a.to(vk_));

    assert_parity(cpu_r, vk_r, 1e-5f, 1e-7f, "Cos");
}

TEST_F(VulkanTrigParity, Asin) {
    auto a = randn({32, 32}, DType::Float32, cpu_) * 0.9f;  // Keep in [-0.9, 0.9]

    auto cpu_r = asin(a);
    auto vk_r = asin(a.to(vk_));

    assert_parity(cpu_r, vk_r, 1e-5f, 1e-7f, "Asin");
}

TEST_F(VulkanTrigParity, Acos) {
    auto a = randn({32, 32}, DType::Float32, cpu_) * 0.9f;

    auto cpu_r = acos(a);
    auto vk_r = acos(a.to(vk_));

    assert_parity(cpu_r, vk_r, 1e-5f, 1e-7f, "Acos");
}

TEST_F(VulkanTrigParity, Atan) {
    auto a = randn({32, 32}, DType::Float32, cpu_);

    auto cpu_r = atan(a);
    auto vk_r = atan(a.to(vk_));

    assert_parity(cpu_r, vk_r, 1e-5f, 1e-7f, "Atan");
}

TEST_F(VulkanTrigParity, Sinh) {
    auto a = randn({32, 32}, DType::Float32, cpu_) * 2.0f;

    auto cpu_r = sinh(a);
    auto vk_r = sinh(a.to(vk_));

    assert_parity(cpu_r, vk_r, 1e-5f, 1e-7f, "Sinh");
}

TEST_F(VulkanTrigParity, Cosh) {
    auto a = randn({32, 32}, DType::Float32, cpu_) * 2.0f;

    auto cpu_r = cosh(a);
    auto vk_r = cosh(a.to(vk_));

    assert_parity(cpu_r, vk_r, 1e-5f, 1e-7f, "Cosh");
}

// ============================================================================
// Section 11: Reduction ops
// ============================================================================

class VulkanReductionParity : public VulkanParityTest {};

TEST_F(VulkanReductionParity, SumDim0) {
    auto a = randn({32, 64}, DType::Float32, cpu_);

    auto cpu_r = sum(a, 0);
    auto vk_r = sum(a.to(vk_), 0);

    assert_parity(cpu_r, vk_r, 1e-4f, 1e-5f, "Sum dim0");
}

TEST_F(VulkanReductionParity, SumDim1) {
    auto a = randn({32, 64}, DType::Float32, cpu_);

    auto cpu_r = sum(a, 1);
    auto vk_r = sum(a.to(vk_), 1);

    assert_parity(cpu_r, vk_r, 1e-4f, 1e-5f, "Sum dim1");
}

TEST_F(VulkanReductionParity, ArgMax) {
    auto a = randn({16, 32}, DType::Float32, cpu_);

    auto cpu_r = argmax(a, 1);
    auto vk_r = argmax(a.to(vk_), 1);

    assert_parity(cpu_r, vk_r, 0.0f, 0.0f, "ArgMax dim1");
}

TEST_F(VulkanReductionParity, ArgMin) {
    auto a = randn({16, 32}, DType::Float32, cpu_);

    auto cpu_r = argmin(a, 1);
    auto vk_r = argmin(a.to(vk_), 1);

    assert_parity(cpu_r, vk_r, 0.0f, 0.0f, "ArgMin dim1");
}

TEST_F(VulkanReductionParity, Prod) {
    auto a = abs(randn({8, 8}, DType::Float32, cpu_)) * 0.05f + 0.95f;

    auto cpu_r = prod(a);
    auto vk_r = prod(a.to(vk_));

    assert_parity(cpu_r, vk_r, 1e-2f, 1e-4f, "Prod global");
}

TEST_F(VulkanReductionParity, Var) {
    auto a = randn({32, 64}, DType::Float32, cpu_);

    auto cpu_r = var(a);
    auto vk_r = var(a.to(vk_));

    assert_parity(cpu_r, vk_r, 1e-4f, 1e-5f, "Var global");
}

TEST_F(VulkanReductionParity, Std) {
    auto a = randn({32, 64}, DType::Float32, cpu_);

    auto cpu_r = tenzor::std(a);
    auto vk_r = tenzor::std(a.to(vk_));

    assert_parity(cpu_r, vk_r, 1e-4f, 1e-5f, "Std global");
}

TEST_F(VulkanReductionParity, Norm) {
    auto a = randn({32, 32}, DType::Float32, cpu_);

    auto cpu_r = norm(a);
    auto vk_r = norm(a.to(vk_));

    assert_parity(cpu_r, vk_r, 1e-4f, 1e-5f, "Norm global");
}

// ============================================================================
// Section 12: More activations via dispatch
// ============================================================================

class VulkanMoreActivationsParity : public VulkanParityTest {};

TEST_F(VulkanMoreActivationsParity, Swish) {
    auto a = randn({32, 32}, DType::Float32, cpu_);

    auto cpu_r = dispatch_to_device(OpId::Swish, Device::Type::CPU, std::vector<Tensor>{a})[0];
    auto vk_r = dispatch_to_device(OpId::Swish, Device::Type::Vulkan, std::vector<Tensor>{a.to(vk_)})[0];

    assert_parity(cpu_r, vk_r, 1e-5f, 1e-7f, "Swish dispatch");
}

TEST_F(VulkanMoreActivationsParity, Selu) {
    auto a = randn({32, 32}, DType::Float32, cpu_);

    auto cpu_r = dispatch_to_device(OpId::Selu, Device::Type::CPU, std::vector<Tensor>{a})[0];
    auto vk_r = dispatch_to_device(OpId::Selu, Device::Type::Vulkan, std::vector<Tensor>{a.to(vk_)})[0];

    assert_parity(cpu_r, vk_r, 1e-5f, 1e-7f, "SELU dispatch");
}

TEST_F(VulkanMoreActivationsParity, Mish) {
    auto a = randn({32, 32}, DType::Float32, cpu_);

    auto cpu_r = dispatch_to_device(OpId::Mish, Device::Type::CPU, std::vector<Tensor>{a})[0];
    auto vk_r = dispatch_to_device(OpId::Mish, Device::Type::Vulkan, std::vector<Tensor>{a.to(vk_)})[0];

    assert_parity(cpu_r, vk_r, 1e-4f, 1e-5f, "Mish dispatch");
}

TEST_F(VulkanMoreActivationsParity, Softplus) {
    auto a = randn({32, 32}, DType::Float32, cpu_);

    auto cpu_r = dispatch_to_device(OpId::Softplus, Device::Type::CPU, std::vector<Tensor>{a})[0];
    auto vk_r = dispatch_to_device(OpId::Softplus, Device::Type::Vulkan, std::vector<Tensor>{a.to(vk_)})[0];

    assert_parity(cpu_r, vk_r, 1e-5f, 1e-7f, "Softplus dispatch");
}

TEST_F(VulkanMoreActivationsParity, LogSoftmax) {
    auto a = randn({16, 32}, DType::Float32, cpu_);
    OpAttributes attrs;
    attrs.set(AttrKey::Dim, static_cast<int64_t>(1));

    auto cpu_r = dispatch_to_device(OpId::LogSoftmax, Device::Type::CPU, std::vector<Tensor>{a}, attrs)[0];
    auto vk_r = dispatch_to_device(OpId::LogSoftmax, Device::Type::Vulkan, std::vector<Tensor>{a.to(vk_)}, attrs)[0];

    assert_parity(cpu_r, vk_r, 1e-4f, 1e-5f, "LogSoftmax dispatch");
}

// ============================================================================
// Section 13: Transform ops
// ============================================================================

class VulkanTransformParity : public VulkanParityTest {};

TEST_F(VulkanTransformParity, Permute) {
    auto a = randn({4, 8, 16}, DType::Float32, cpu_);

    auto cpu_r = permute(a, {2, 0, 1}).contiguous();
    auto vk_r = permute(a.to(vk_), {2, 0, 1}).contiguous();

    assert_parity(cpu_r, vk_r, 0.0f, 0.0f, "Permute");
}

TEST_F(VulkanTransformParity, Squeeze) {
    auto a = randn({4, 1, 16}, DType::Float32, cpu_);

    auto cpu_r = squeeze(a, 1);
    auto vk_r = squeeze(a.to(vk_), 1);

    assert_parity(cpu_r, vk_r, 0.0f, 0.0f, "Squeeze");
}

TEST_F(VulkanTransformParity, Unsqueeze) {
    auto a = randn({4, 16}, DType::Float32, cpu_);

    auto cpu_r = unsqueeze(a, 1);
    auto vk_r = unsqueeze(a.to(vk_), 1);

    assert_parity(cpu_r, vk_r, 0.0f, 0.0f, "Unsqueeze");
}

TEST_F(VulkanTransformParity, Expand) {
    auto a = randn({1, 16}, DType::Float32, cpu_);

    auto cpu_r = expand(a, {8, 16}).contiguous();
    auto vk_r = expand(a.to(vk_), {8, 16}).contiguous();

    assert_parity(cpu_r, vk_r, 0.0f, 0.0f, "Expand");
}

TEST_F(VulkanTransformParity, Flatten) {
    auto a = randn({4, 8, 16}, DType::Float32, cpu_);

    auto cpu_r = flatten(a, 1, 2);
    auto vk_r = flatten(a.to(vk_), 1, 2);

    assert_parity(cpu_r, vk_r, 0.0f, 0.0f, "Flatten");
}

TEST_F(VulkanTransformParity, Split) {
    auto a = randn({16, 32}, DType::Float32, cpu_);

    auto cpu_parts = split(a, 4, 0);
    auto vk_parts = split(a.to(vk_), 4, 0);

    ASSERT_EQ(cpu_parts.size(), vk_parts.size());
    for (size_t i = 0; i < cpu_parts.size(); ++i) {
        assert_parity(cpu_parts[i], vk_parts[i], 0.0f, 0.0f,
                      "Split part " + std::to_string(i));
    }
}

TEST_F(VulkanTransformParity, Chunk) {
    auto a = randn({16, 32}, DType::Float32, cpu_);

    auto cpu_parts = chunk(a, 4, 0);
    auto vk_parts = chunk(a.to(vk_), 4, 0);

    ASSERT_EQ(cpu_parts.size(), vk_parts.size());
    for (size_t i = 0; i < cpu_parts.size(); ++i) {
        assert_parity(cpu_parts[i], vk_parts[i], 0.0f, 0.0f,
                      "Chunk part " + std::to_string(i));
    }
}

TEST_F(VulkanTransformParity, Repeat) {
    auto a = randn({4, 8}, DType::Float32, cpu_);

    auto cpu_r = repeat(a, {2, 3});
    auto vk_r = repeat(a.to(vk_), {2, 3});

    assert_parity(cpu_r, vk_r, 0.0f, 0.0f, "Repeat");
}

// ============================================================================
// Section 14: Indexing ops (scatter, where, masked_fill)
// ============================================================================

class VulkanIndexingParity : public VulkanParityTest {};

TEST_F(VulkanIndexingParity, Scatter) {
    auto a = zeros({8, 16}, DType::Float32, cpu_);
    auto src = randn({8, 4}, DType::Float32, cpu_);
    auto idx = zeros({8, 4}, DType::Int64, cpu_);
    {
        auto d = idx.data<int64_t>();
        for (int i = 0; i < 8; ++i)
            for (int j = 0; j < 4; ++j)
                d[i * 4 + j] = j * 4;
    }

    auto cpu_r = scatter(a, 1, idx, src);
    auto vk_r = scatter(a.to(vk_), 1, idx.to(vk_), src.to(vk_));

    assert_parity(cpu_r, vk_r, 0.0f, 0.0f, "Scatter");
}

TEST_F(VulkanIndexingParity, Where) {
    auto a = randn({32, 32}, DType::Float32, cpu_);
    auto b = randn({32, 32}, DType::Float32, cpu_);
    auto cond = gt(a, b);

    auto cpu_r = where(cond, a, b);
    auto vk_r = where(cond.to(vk_), a.to(vk_), b.to(vk_));

    assert_parity(cpu_r, vk_r, 0.0f, 0.0f, "Where");
}

TEST_F(VulkanIndexingParity, MaskedFill) {
    auto a = randn({32, 32}, DType::Float32, cpu_);
    auto mask = gt(a, zeros({32, 32}, DType::Float32, cpu_));

    auto cpu_r = masked_fill(a, mask, -999.0f);
    auto vk_r = masked_fill(a.to(vk_), mask.to(vk_), -999.0f);

    assert_parity(cpu_r, vk_r, 0.0f, 0.0f, "MaskedFill");
}

TEST_F(VulkanIndexingParity, Nonzero) {
    auto a = randn({8, 8}, DType::Float32, cpu_);
    auto mask = gt(a, zeros({8, 8}, DType::Float32, cpu_));
    auto input = mask.to(DType::Float32) * a;

    auto cpu_r = nonzero(input);
    auto vk_r = nonzero(input.to(vk_));

    ASSERT_EQ(cpu_r.shape()[1], vk_r.shape()[1]);
    assert_parity(cpu_r, vk_r, 0.0f, 0.0f, "Nonzero");
}

// ============================================================================
// Section 15: Sort/TopK
// ============================================================================

class VulkanSortParity : public VulkanParityTest {};

TEST_F(VulkanSortParity, Sort) {
    auto a = randn({16, 32}, DType::Float32, cpu_);

    auto [cpu_vals, cpu_idx] = sort(a, 1);
    auto [vk_vals, vk_idx] = sort(a.to(vk_), 1);

    assert_parity(cpu_vals, vk_vals, 1e-6f, 1e-8f, "Sort values");
}

TEST_F(VulkanSortParity, TopK) {
    auto a = randn({8, 32}, DType::Float32, cpu_);

    auto [cpu_vals, cpu_idx] = topk(a, 5, 1);
    auto [vk_vals, vk_idx] = topk(a.to(vk_), 5, 1);

    assert_parity(cpu_vals, vk_vals, 1e-6f, 1e-8f, "TopK values");
}

// ============================================================================
// Section 16: Creation ops
// ============================================================================

class VulkanCreationParity : public VulkanParityTest {};

TEST_F(VulkanCreationParity, Zeros) {
    auto cpu_r = zeros({32, 32}, DType::Float32, cpu_);
    auto vk_r = zeros({32, 32}, DType::Float32, vk_);

    assert_parity(cpu_r, vk_r, 0.0f, 0.0f, "Zeros");
}

TEST_F(VulkanCreationParity, Ones) {
    auto cpu_r = ones({32, 32}, DType::Float32, cpu_);
    auto vk_r = ones({32, 32}, DType::Float32, vk_);

    assert_parity(cpu_r, vk_r, 0.0f, 0.0f, "Ones");
}

TEST_F(VulkanCreationParity, Full) {
    auto cpu_r = full({32, 32}, 3.14f, DType::Float32, cpu_);
    auto vk_r = full({32, 32}, 3.14f, DType::Float32, vk_);

    assert_parity(cpu_r, vk_r, 0.0f, 0.0f, "Full");
}

TEST_F(VulkanCreationParity, Eye) {
    auto cpu_r = eye(16, std::nullopt, DType::Float32, cpu_);
    auto vk_r = eye(16, std::nullopt, DType::Float32, vk_);

    assert_parity(cpu_r, vk_r, 0.0f, 0.0f, "Eye");
}

TEST_F(VulkanCreationParity, Arange) {
    auto cpu_r = arange(0.0f, 100.0f, 1.0f, DType::Float32, cpu_);
    auto vk_r = arange(0.0f, 100.0f, 1.0f, DType::Float32, vk_);

    assert_parity(cpu_r, vk_r, 1e-6f, 1e-8f, "Arange");
}

TEST_F(VulkanCreationParity, Linspace) {
    auto cpu_r = linspace(0.0f, 10.0f, 50, DType::Float32, cpu_);
    auto vk_r = linspace(0.0f, 10.0f, 50, DType::Float32, vk_);

    assert_parity(cpu_r, vk_r, 1e-6f, 1e-8f, "Linspace");
}

TEST_F(VulkanCreationParity, Clone) {
    auto a = randn({32, 32}, DType::Float32, cpu_);

    auto cpu_r = a.clone();
    auto vk_r = a.to(vk_).clone();

    assert_parity(cpu_r, vk_r, 0.0f, 0.0f, "Clone");
}

// ============================================================================
// Section 17: Cast
// ============================================================================

class VulkanCastParity : public VulkanParityTest {};

TEST_F(VulkanCastParity, F32toF64) {
    auto a = randn({32, 32}, DType::Float32, cpu_);

    auto cpu_r = a.to(DType::Float64);
    auto vk_r = a.to(vk_).to(DType::Float64);

    assert_parity(cpu_r, vk_r, 0.0f, 0.0f, "Cast F32->F64");
}

TEST_F(VulkanCastParity, F64toF32) {
    auto a = randn({32, 32}, DType::Float64, cpu_);

    auto cpu_r = a.to(DType::Float32);
    auto vk_r = a.to(vk_).to(DType::Float32);

    assert_parity(cpu_r, vk_r, 1e-6f, 1e-8f, "Cast F64->F32");
}

// ============================================================================
// Section 18: Broadcasting
// ============================================================================

class VulkanBroadcastParity : public VulkanParityTest {};

TEST_F(VulkanBroadcastParity, Mul_ColBroadcast) {
    auto a = randn({32, 64}, DType::Float32, cpu_);
    auto b = randn({32, 1}, DType::Float32, cpu_);

    auto cpu_r = a * b;
    auto vk_r = a.to(vk_) * b.to(vk_);

    assert_parity(cpu_r, vk_r, 1e-6f, 1e-8f, "Broadcast mul col");
}

TEST_F(VulkanBroadcastParity, Sub_3D) {
    auto a = randn({4, 1, 32}, DType::Float32, cpu_);
    auto b = randn({1, 16, 32}, DType::Float32, cpu_);

    auto cpu_r = a - b;
    auto vk_r = a.to(vk_) - b.to(vk_);

    assert_parity(cpu_r, vk_r, 1e-6f, 1e-8f, "Broadcast sub 3D");
}

// ============================================================================
// Section 19: Complex expressions
// ============================================================================

class VulkanComplexExprParity : public VulkanParityTest {};

TEST_F(VulkanComplexExprParity, AttentionPattern) {
    auto q = randn({4, 8, 32}, DType::Float32, cpu_);
    auto k = randn({4, 32, 8}, DType::Float32, cpu_);

    OpAttributes sm_attrs;
    sm_attrs.set(AttrKey::Dim, static_cast<int64_t>(2));

    auto cpu_scores = bmm(q, k) * static_cast<float>(1.0 / std::sqrt(32.0));
    auto vk_scores = bmm(q.to(vk_), k.to(vk_)) * static_cast<float>(1.0 / std::sqrt(32.0));

    auto cpu_r = dispatch_to_device(OpId::Softmax, Device::Type::CPU,
                                     std::vector<Tensor>{cpu_scores}, sm_attrs)[0];
    auto vk_r = dispatch_to_device(OpId::Softmax, Device::Type::Vulkan,
                                     std::vector<Tensor>{vk_scores}, sm_attrs)[0];

    assert_parity(cpu_r, vk_r, 1e-4f, 1e-5f, "Attention pattern");
}

TEST_F(VulkanComplexExprParity, ResidualBlock) {
    auto x = randn({4, 32}, DType::Float32, cpu_);
    auto w = randn({32, 32}, DType::Float32, cpu_);

    auto cpu_lin = matmul(x, transpose(w, 0, 1));
    auto vk_lin = matmul(x.to(vk_), transpose(w.to(vk_), 0, 1));

    auto cpu_act = dispatch_to_device(OpId::ReLU, Device::Type::CPU,
                                       std::vector<Tensor>{cpu_lin})[0];
    auto vk_act = dispatch_to_device(OpId::ReLU, Device::Type::Vulkan,
                                       std::vector<Tensor>{vk_lin})[0];

    auto cpu_r = x + cpu_act;
    auto vk_r = x.to(vk_) + vk_act;

    assert_parity(cpu_r, vk_r, 1e-4f, 1e-5f, "Residual block");
}

TEST_F(VulkanComplexExprParity, NormGELUChain) {
    auto a = randn({4, 32}, DType::Float32, cpu_);
    auto w = abs(randn({32}, DType::Float32, cpu_)) + 0.5f;
    auto b = randn({32}, DType::Float32, cpu_) * 0.1f;

    OpAttributes ln_attrs;
    ln_attrs.set(AttrKey::NormalizedShape, "32");

    auto cpu_ln = dispatch_to_device(OpId::LayerNorm, Device::Type::CPU,
                                      std::vector<Tensor>{a, w, b}, ln_attrs)[0];
    auto vk_ln = dispatch_to_device(OpId::LayerNorm, Device::Type::Vulkan,
                                      std::vector<Tensor>{a.to(vk_), w.to(vk_), b.to(vk_)}, ln_attrs)[0];

    auto cpu_gelu = dispatch_to_device(OpId::Gelu, Device::Type::CPU,
                                        std::vector<Tensor>{cpu_ln})[0];
    auto vk_gelu = dispatch_to_device(OpId::Gelu, Device::Type::Vulkan,
                                        std::vector<Tensor>{vk_ln})[0];

    auto cpu_r = sum(cpu_gelu);
    auto vk_r = sum(vk_gelu);

    assert_parity(cpu_r, vk_r, 1e-3f, 1e-4f, "Norm+GELU+Sum chain");
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);

    try {
        tenzor::initialize();
    } catch (const std::exception& e) {
        std::cerr << "Failed to initialize Tenzor: " << e.what() << std::endl;
        return 1;
    }

    int result = RUN_ALL_TESTS();

    try {
        tenzor::finalize();
    } catch (...) {
        // Ignore cleanup errors
    }

    return result;
}
