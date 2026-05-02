/**
 * @file test_histogramdd.cpp
 * @brief Multi-backend tests for the histogramdd reduction.
 *
 * Migrated from CPU-only `::testing::Test` to BackendTest. Native
 * histogramdd kernels are registered on every backend (CUDA, ROCm,
 * OneAPI, Vulkan in addition to CPU); the previous CPU-only test
 * pattern only verified the CPU path. The migration uses BackendTest
 * since histogramdd's correctness is dtype-agnostic for input data
 * (Float32/Float64 covered explicitly via Float64Basic).
 *
 * Phase 7.5 of the test-coverage campaign — the audit's claim that
 * histogramdd needed GPU kernels was outdated; the native kernels
 * already exist (`cuda::histogramdd_kernel`, `rocm::histogramdd_kernel`,
 * etc.). Only the test file's CPU-only fixture needed migration.
 */

#include <gtest/gtest.h>
#include "../backend_test_fixture.hpp"
#include <tenzor/tenzor.hpp>
#include <tenzor/ops/reduction.hpp>
#include <tenzor/ops/creation.hpp>
#include <cmath>

using namespace tenzor;
using namespace tenzor::testing;

class HistogramddTest : public BackendTest {};

// Helper — create a populated input on CPU then move to the test device.
// We need direct write access to the tensor data which only works for CPU
// storage; the histogramdd op then runs on the target device.
namespace {
auto make_cpu_then_to(const std::vector<int64_t>& shape, DType dtype, Device target,
                     std::function<void(void*)> fill) -> Tensor {
    auto cpu = zeros(shape, dtype, Device::cpu());
    fill(const_cast<void*>(cpu.data_ptr()));
    return (target.type == Device::Type::CPU) ? cpu : cpu.to(target);
}
}  // namespace

// ============================================================================
// Basic 2D histogram with known data
// ============================================================================

TEST_P(HistogramddTest, Basic2DKnownData) {
    auto input = make_cpu_then_to({4, 2}, DType::Float32, device, [](void* p) {
        auto* d = static_cast<float*>(p);
        d[0] = 0.1f; d[1] = 0.1f;
        d[2] = 0.9f; d[3] = 0.9f;
        d[4] = 0.1f; d[5] = 0.9f;
        d[6] = 0.9f; d[7] = 0.1f;
    });

    auto [counts, edges] = histogramdd(input, {2, 2},
        std::vector<std::pair<double,double>>{{0.0, 1.0}, {0.0, 1.0}});

    EXPECT_EQ(counts.shape().size(), 2u);
    EXPECT_EQ(counts.shape()[0], 2);
    EXPECT_EQ(counts.shape()[1], 2);
    EXPECT_EQ(edges.size(), 2u);

    auto counts_cpu = counts.to(Device::cpu()).contiguous();
    auto* c = counts_cpu.data<int64_t>();
    int64_t total = c[0] + c[1] + c[2] + c[3];
    EXPECT_EQ(total, 4) << "on " << device.to_string();
}

// ============================================================================
// Auto-range detection
// ============================================================================

TEST_P(HistogramddTest, AutoRange) {
    auto input = make_cpu_then_to({6, 2}, DType::Float32, device, [](void* p) {
        auto* d = static_cast<float*>(p);
        d[0]  = 1.0f; d[1]  = 1.0f;
        d[2]  = 3.0f; d[3]  = 3.0f;
        d[4]  = 5.0f; d[5]  = 5.0f;
        d[6]  = 7.0f; d[7]  = 7.0f;
        d[8]  = 9.0f; d[9]  = 9.0f;
        d[10] = 2.0f; d[11] = 8.0f;
    });

    auto [counts, edges] = histogramdd(input, {5, 5});

    EXPECT_EQ(counts.shape().size(), 2u);
    EXPECT_EQ(counts.shape()[0], 5);
    EXPECT_EQ(counts.shape()[1], 5);

    auto counts_cpu = counts.to(Device::cpu()).contiguous();
    auto* c = counts_cpu.data<int64_t>();
    int64_t total = 0;
    for (int64_t i = 0; i < counts_cpu.numel(); ++i) total += c[i];
    EXPECT_EQ(total, 6) << "on " << device.to_string();
}

// ============================================================================
// Explicit ranges
// ============================================================================

TEST_P(HistogramddTest, ExplicitRanges) {
    auto input = make_cpu_then_to({3, 2}, DType::Float32, device, [](void* p) {
        auto* d = static_cast<float*>(p);
        d[0] = 0.5f; d[1] = 0.5f;
        d[2] = 1.5f; d[3] = 1.5f;
        d[4] = 2.5f; d[5] = 2.5f;
    });

    auto [counts, edges] = histogramdd(input, {3, 3},
        std::vector<std::pair<double,double>>{{0.0, 3.0}, {0.0, 3.0}});

    EXPECT_EQ(counts.shape()[0], 3);
    EXPECT_EQ(counts.shape()[1], 3);

    auto counts_cpu = counts.to(Device::cpu()).contiguous();
    auto* c = counts_cpu.data<int64_t>();
    EXPECT_EQ(c[0 * 3 + 0], 1);
    EXPECT_EQ(c[1 * 3 + 1], 1);
    EXPECT_EQ(c[2 * 3 + 2], 1);
}

// ============================================================================
// Density normalization
// ============================================================================

TEST_P(HistogramddTest, DensityNormalization) {
    auto input = make_cpu_then_to({4, 2}, DType::Float32, device, [](void* p) {
        auto* d = static_cast<float*>(p);
        d[0] = 0.25f; d[1] = 0.25f;
        d[2] = 0.75f; d[3] = 0.75f;
        d[4] = 0.25f; d[5] = 0.75f;
        d[6] = 0.75f; d[7] = 0.25f;
    });

    auto [counts, edges] = histogramdd(input, {2, 2},
        std::vector<std::pair<double,double>>{{0.0, 1.0}, {0.0, 1.0}},
        /*density=*/true);

    auto counts_cpu = counts.to(Device::cpu()).contiguous();
    auto* c = counts_cpu.data<float>();
    float integral = 0.0f;
    double bin_vol = 0.5 * 0.5;
    for (int64_t i = 0; i < counts_cpu.numel(); ++i) {
        integral += static_cast<float>(c[i] * bin_vol);
    }
    EXPECT_NEAR(integral, 1.0f, 1e-4f) << "on " << device.to_string();
}

// ============================================================================
// Edge cases
// ============================================================================

TEST_P(HistogramddTest, SingleSample) {
    auto input = make_cpu_then_to({1, 2}, DType::Float32, device, [](void* p) {
        auto* d = static_cast<float*>(p);
        d[0] = 0.5f; d[1] = 0.5f;
    });

    auto [counts, edges] = histogramdd(input, {3, 3},
        std::vector<std::pair<double,double>>{{0.0, 1.0}, {0.0, 1.0}});

    auto counts_cpu = counts.to(Device::cpu()).contiguous();
    auto* c = counts_cpu.data<int64_t>();
    int64_t total = 0;
    for (int64_t i = 0; i < counts_cpu.numel(); ++i) total += c[i];
    EXPECT_EQ(total, 1);
}

TEST_P(HistogramddTest, AllSameValue) {
    auto input = full({10, 2}, 0.5f, DType::Float32, device);

    auto [counts, edges] = histogramdd(input, {4, 4},
        std::vector<std::pair<double,double>>{{0.0, 1.0}, {0.0, 1.0}});

    auto counts_cpu = counts.to(Device::cpu()).contiguous();
    auto* c = counts_cpu.data<int64_t>();
    int64_t max_val = 0;
    for (int64_t i = 0; i < counts_cpu.numel(); ++i) {
        max_val = std::max(max_val, c[i]);
    }
    EXPECT_EQ(max_val, 10) << "on " << device.to_string();
}

// ============================================================================
// Multi-dtype: Float64
// ============================================================================

TEST_P(HistogramddTest, Float64Basic) {
    auto input = make_cpu_then_to({3, 2}, DType::Float64, device, [](void* p) {
        auto* d = static_cast<double*>(p);
        d[0] = 0.1; d[1] = 0.1;
        d[2] = 0.5; d[3] = 0.5;
        d[4] = 0.9; d[5] = 0.9;
    });

    auto [counts, edges] = histogramdd(input, {2, 2},
        std::vector<std::pair<double,double>>{{0.0, 1.0}, {0.0, 1.0}});

    EXPECT_EQ(counts.shape()[0], 2);
    EXPECT_EQ(counts.shape()[1], 2);

    auto counts_cpu = counts.to(Device::cpu()).contiguous();
    auto counts_f32 = counts_cpu.to(DType::Float32);
    auto* c = counts_f32.data<float>();
    float total = 0.0f;
    for (int64_t i = 0; i < counts_f32.numel(); ++i) total += c[i];
    EXPECT_NEAR(total, 3.0f, 1e-5f) << "on " << device.to_string();
}

INSTANTIATE_BACKEND_TESTS(HistogramddTest);
