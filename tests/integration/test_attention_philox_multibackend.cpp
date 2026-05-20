/**
 * @file test_attention_philox_multibackend.cpp
 * @brief Phase 13b (audit-2026-05-03) — Cross-backend FlashAttention
 * Philox-replay invariants as parameterized tests.
 *
 * The TEST_F counterparts in test_attention_autograd.cpp run only on the
 * default device. This file lifts the seed-determinism check to a per-
 * backend TEST_P so each backend independently asserts:
 *   - Re-seeding the default generator produces identical output and
 *     identical gradients across two forward+backward passes.
 *
 * Cross-backend bit equality is covered by FlashAttentionPhiloxReplay_
 * CrossBackendMask in test_attention_autograd.cpp (a TEST_F that
 * iterates over available GPU backends inside the test body).
 */

#include "../backend_test_fixture.hpp"
#include "tenzor/tenzor.hpp"
#include "tenzor/autograd/ops.hpp"
#include "tenzor/autograd/variable.hpp"
#include "tenzor/autograd/engine.hpp"
#include "tenzor/core/generator.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/reduction.hpp"
#include <gtest/gtest.h>
#include <cmath>
#include "../grad_flow_helpers.hpp"

using ::tenzor::Variable;
using ::tenzor::Tensor;
using ::tenzor::testing::BackendTest;

namespace {

TEST_P(BackendTest, FlashAttentionPhiloxReplay_SeedDeterminism) {
    // BackendTest::SetUp() already honors TENZOR_REQUIRE_MULTI_BACKEND
    // and TENZOR_SKIP_BACKENDS via its own SetUp logic.
    //
    // Vulkan and OneAPI flash-attention kernels explicitly throw on
    // dropout > 0 — full Philox kernels are pending the per-backend
    // shader refit (audit C2 OneAPI / M8 Vulkan). Skip the determinism
    // test on those backends; the CPU/CUDA/ROCm Philox replay already
    // covers the contract. This is an environmental skip, not a
    // bug-tracking one.
    if (device.type == tenzor::Device::Type::Vulkan ||
        device.type == tenzor::Device::Type::OneAPI) {
        GTEST_SKIP() << "FlashAttention dropout not implemented on "
                     << device.to_string()
                     << " (kernel-level Philox refit pending)";
    }
    int64_t B = 1, H = 2, S = 4, D = 8;
    auto qt = tenzor::randn({B, H, S, D}, tenzor::DType::Float32, device) * 0.3f;
    auto kt = tenzor::randn({B, H, S, D}, tenzor::DType::Float32, device) * 0.3f;
    auto vt = tenzor::randn({B, H, S, D}, tenzor::DType::Float32, device) * 0.3f;
    float scale = 1.0f / std::sqrt(static_cast<float>(D));

    tenzor::default_generator(device).manual_seed(42);
    Variable Q1(qt, true), K1(kt, false), V1(vt, false);
    auto out1 = tenzor::flash_attention(Q1, K1, V1, scale,
        /*causal=*/false, /*dropout_p=*/0.5f, /*is_training=*/true);
    auto loss1 = tenzor::sum(out1);
    loss1.backward();

    tenzor::default_generator(device).manual_seed(42);
    Variable Q2(qt, true), K2(kt, false), V2(vt, false);
    auto out2 = tenzor::flash_attention(Q2, K2, V2, scale,
        /*causal=*/false, /*dropout_p=*/0.5f, /*is_training=*/true);
    auto loss2 = tenzor::sum(out2);
    loss2.backward();

    // Same seed → same forward output.
    auto o1 = out1.tensor().to(tenzor::Device::cpu()).contiguous();
    auto o2 = out2.tensor().to(tenzor::Device::cpu()).contiguous();
    auto* p1 = o1.data<float>();
    auto* p2 = o2.data<float>();
    for (int64_t i = 0; i < o1.numel(); ++i) {
        EXPECT_FLOAT_EQ(p1[i], p2[i])
            << "Forward output not deterministic at element " << i
            << " on " << device.to_string();
    }

    // Same seed → same gradient (the dropout-mask backward replay must
    // match the forward draw bit-for-bit).
    EXPECT_GRAD_FLOWS(Q1);
    EXPECT_GRAD_FLOWS(Q2);
    auto g1 = Q1.grad()->to(tenzor::Device::cpu()).contiguous();
    auto g2 = Q2.grad()->to(tenzor::Device::cpu()).contiguous();
    auto* gp1 = g1.data<float>();
    auto* gp2 = g2.data<float>();
    for (int64_t i = 0; i < g1.numel(); ++i) {
        EXPECT_FLOAT_EQ(gp1[i], gp2[i])
            << "Backward gradient not deterministic at element " << i
            << " on " << device.to_string();
    }
}

INSTANTIATE_BACKEND_TESTS(BackendTest);

} // namespace
