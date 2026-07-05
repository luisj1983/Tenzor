/**
 * @file test_cdist.cpp
 * @brief Tests for pairwise distance computation — parameterized across
 *        every available backend. Previously CPU-only.
 */

#include <gtest/gtest.h>
#include "tenzor/tenzor.hpp"
#include "../backend_test_fixture.hpp"
#include <cmath>
#include <limits>

using namespace tenzor;

class CDistTest : public tenzor::testing::BackendTest {};

TEST_P(CDistTest, ND4DMatchesCPU) {
    // 4-D inputs with a matching multi-dim batch (2,3) — the old CUDA cdist threw
    // for ndim>3; CPU handles any ndim>=2.
    auto ha = tenzor::randn({2, 3, 4, 5}, DType::Float32, Device::cpu());
    auto hb = tenzor::randn({2, 3, 6, 5}, DType::Float32, Device::cpu());
    auto rc = tenzor::cdist(ha, hb, 2.0);                       // CPU reference
    auto rd = tenzor::cdist(ha.to(device), hb.to(device), 2.0).cpu();
    ASSERT_EQ(rc.numel(), rd.numel());
    EXPECT_EQ(rd.shape().size(), 4u);  // (2,3,4,6)
    const float* a = rc.data<float>();
    const float* b = rd.data<float>();
    for (int64_t i = 0; i < rc.numel(); ++i)
        EXPECT_NEAR(a[i], b[i], 1e-3f) << "elem " << i << " on " << device.to_string();
}

TEST_P(CDistTest, BatchOneBroadcastMatchesCPU) {
    // 3-D with one batch == 1 broadcasts to the other's batch (CPU behavior).
    auto ha = tenzor::randn({1, 4, 5}, DType::Float32, Device::cpu());
    auto hb = tenzor::randn({3, 6, 5}, DType::Float32, Device::cpu());
    auto rc = tenzor::cdist(ha, hb, 2.0);
    auto rd = tenzor::cdist(ha.to(device), hb.to(device), 2.0).cpu();
    ASSERT_EQ(rc.numel(), rd.numel());
    const float* a = rc.data<float>();
    const float* b = rd.data<float>();
    for (int64_t i = 0; i < rc.numel(); ++i)
        EXPECT_NEAR(a[i], b[i], 1e-3f) << "elem " << i << " on " << device.to_string();
}

TEST_P(CDistTest, PdistChebyshevInfinity) {
    // p=inf → Chebyshev (max abs coordinate diff). Points [[0,0],[3,4],[1,-2]]:
    //   (0,1)=max(3,4)=4 ; (0,2)=max(1,2)=2 ; (1,2)=max(2,6)=6.
    // The old CUDA generic path returned all-ones for p=inf.
    auto host = tenzor::empty({3, 2}, DType::Float32);
    const float vals[6] = {0, 0, 3, 4, 1, -2};
    for (int i = 0; i < 6; ++i) host.data<float>()[i] = vals[i];
    auto x = host.to(device);
    auto d = tenzor::pdist(x, std::numeric_limits<double>::infinity()).cpu();
    const float* p = d.data<float>();
    EXPECT_NEAR(p[0], 4.0f, 1e-5f) << "on " << device.to_string();
    EXPECT_NEAR(p[1], 2.0f, 1e-5f) << "on " << device.to_string();
    EXPECT_NEAR(p[2], 6.0f, 1e-5f) << "on " << device.to_string();
}

TEST_P(CDistTest, EuclideanDistance2D) {
    auto x1 = tenzor::zeros({2, 3}, DType::Float32, device);
    auto x2 = tenzor::ones({3, 3}, DType::Float32, device);

    auto result = tenzor::cdist(x1, x2, 2.0);

    EXPECT_EQ(result.ndim(), 2);
    EXPECT_EQ(result.shape()[0], 2);
    EXPECT_EQ(result.shape()[1], 3);

    auto r_cpu = result.to(Device::cpu()).to(DType::Float32).contiguous();
    EXPECT_NEAR(r_cpu.data<float>()[0], std::sqrt(3.0f), 1e-4);
}

TEST_P(CDistTest, ManhattanDistance) {
    // Previously skipped on non-CPU backends because every GPU kernel
    // ignored the `p` argument and returned L2 (≈1.732) instead of L1 (=3).
    // Fixed by adding p=1 / p=2 / generic-Lp branches to the CUDA/ROCm/
    // OneAPI/Vulkan cdist kernels and plumbing `p` through the Vulkan
    // dispatch. Now runs on every backend.
    auto x1 = tenzor::zeros({2, 3}, DType::Float32, device);
    auto x2 = tenzor::ones({2, 3}, DType::Float32, device);

    auto result = tenzor::cdist(x1, x2, 1.0);

    EXPECT_EQ(result.shape()[0], 2);
    EXPECT_EQ(result.shape()[1], 2);

    auto r_cpu = result.to(Device::cpu()).to(DType::Float32).contiguous();
    EXPECT_NEAR(r_cpu.data<float>()[0], 3.0f, 1e-4);
}

TEST_P(CDistTest, BatchedCDist) {
    auto x1 = tenzor::randn({4, 5, 3}, DType::Float32, device);
    auto x2 = tenzor::randn({4, 6, 3}, DType::Float32, device);

    auto result = tenzor::cdist(x1, x2, 2.0);

    EXPECT_EQ(result.ndim(), 3);
    EXPECT_EQ(result.shape()[0], 4);
    EXPECT_EQ(result.shape()[1], 5);
    EXPECT_EQ(result.shape()[2], 6);

    auto r_cpu = result.to(Device::cpu()).to(DType::Float32).contiguous();
    const float* data = r_cpu.data<float>();
    for (int64_t i = 0; i < r_cpu.numel(); i++) {
        EXPECT_GE(data[i], 0.0f);
    }
}

TEST_P(CDistTest, SelfDistance) {
    auto x = tenzor::randn({3, 2}, DType::Float32, device);
    auto result = tenzor::cdist(x, x, 2.0);

    auto r_cpu = result.to(Device::cpu()).to(DType::Float32).contiguous();
    const float* data = r_cpu.data<float>();
    for (int64_t i = 0; i < 3; i++) {
        EXPECT_NEAR(data[i * 3 + i], 0.0f, 1e-4);
    }
}

INSTANTIATE_BACKEND_TESTS(CDistTest);
