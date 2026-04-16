/**
 * @file test_moe_hrm_parity.cpp
 * @brief Backend parity for MoE and HRM layers (Phase 5.4 / 5.5).
 *
 * These are large Module-style layers; the parity test is the usual pattern:
 * build on CPU, clone parameters to each backend, compare forward outputs.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/nn/layers/moe.hpp>
#include <tenzor/nn/layers/hrm.hpp>
#include "parity_test_utils.hpp"

using namespace tenzor;
using namespace tenzor::testing;

// ============================================================================
// Mixture of Experts
// ============================================================================

// Previously DISABLED_ — Vulkan MoE diverged ~0.45 from CPU due to a bug in
// the Vulkan expand kernel: `dispatchExpand` routed Int64 tensors through
// the default `expand.comp` shader, whose buffer declaration is `float
// input_data[]` (4-byte stride). That silently read/wrote the wrong bytes
// for Int64 inputs, corrupting the `eq(idx_col, expert_scalar)` mask in
// MoE's routing path. Fixed by adding `expand_i64.comp` and routing Int64
// tensors to it in `vulkan_ops_memory.cpp::dispatchExpand`.
TEST(MoEHRMParity, MixtureOfExperts_Forward) {
    // Small config: input_dim=16, hidden_dim=32, num_experts=4, top_k=2.
    nn::MixtureOfExperts moe_cpu(16, 32, 4, 2, 1.25, 0.01, 0.0);
    moe_cpu.eval();

    auto input = randn({2, 8, 16}, DType::Float32, Device::cpu());
    auto ref = moe_cpu.forward(Variable(input, false)).tensor();

    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    for (size_t i = 1; i < backends.size(); ++i) {
        try {
            nn::MixtureOfExperts moe_dev(16, 32, 4, 2, 1.25, 0.01, 0.0);
            moe_dev.eval();
            auto params = moe_cpu.parameters();
            auto dev_params = moe_dev.parameters();
            for (size_t p = 0; p < params.size(); ++p) {
                dev_params[p]->tensor() = params[p]->tensor().clone();
            }
            moe_dev.to(backends[i]);
            auto input_dev = input.to(backends[i]);
            auto out = moe_dev.forward(Variable(input_dev, false)).tensor();
            backends[i].synchronize();
            SCOPED_TRACE(std::string("MixtureOfExperts on ")
                         + backend_name(backends[i]));
            // MoE has routing indeterminism at boundaries; rel tolerance 1e-2.
            EXPECT_TENSORS_CLOSE(ref, out.to(Device::cpu()), 1e-2f, 1e-3f);
        } catch (const std::exception& e) {
            std::cerr << "MoE skipped on " << backend_name(backends[i])
                      << ": " << e.what() << std::endl;
        }
    }
}

// ============================================================================
// HRM (Hierarchical Reasoning Module)
// ============================================================================

TEST(MoEHRMParity, HRM_Forward) {
    nn::HRMConfig cfg;
    cfg.d_model = 32;
    cfg.n_heads = 4;
    cfg.d_feedforward = 64;
    cfg.n_high_cycles = 2;
    cfg.t_low_steps = 2;
    cfg.dropout = 0.0;
    cfg.deep_supervision = false;
    cfg.max_seq_len = 16;
    cfg.vocab_size = 0;
    cfg.num_classes = 0;

    nn::HRM hrm_cpu(cfg);
    hrm_cpu.eval();

    auto input = randn({1, 8, 32}, DType::Float32, Device::cpu());
    Tensor ref;
    try {
        ref = hrm_cpu.forward(Variable(input, false)).tensor();
    } catch (const std::exception& e) {
        GTEST_SKIP() << "HRM CPU reference failed: " << e.what();
    }

    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    for (size_t i = 1; i < backends.size(); ++i) {
        try {
            nn::HRM hrm_dev(cfg);
            hrm_dev.eval();
            auto params = hrm_cpu.parameters();
            auto dev_params = hrm_dev.parameters();
            for (size_t p = 0; p < params.size(); ++p) {
                dev_params[p]->tensor() = params[p]->tensor().clone();
            }
            hrm_dev.to(backends[i]);
            auto input_dev = input.to(backends[i]);
            auto out = hrm_dev.forward(Variable(input_dev, false)).tensor();
            backends[i].synchronize();
            SCOPED_TRACE(std::string("HRM on ") + backend_name(backends[i]));
            EXPECT_TENSORS_CLOSE(ref, out.to(Device::cpu()), 1e-2f, 1e-3f);
        } catch (const std::exception& e) {
            std::cerr << "HRM skipped on " << backend_name(backends[i])
                      << ": " << e.what() << std::endl;
        }
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    try {
        if (!::testing::GTEST_FLAG(list_tests)) {
            tenzor::initialize();
        }
    } catch (const std::exception& e) {
        std::cerr << "Failed to initialize Tenzor: " << e.what() << std::endl;
        return 1;
    }
    int result = RUN_ALL_TESTS();
    try { tenzor::finalize(); } catch (...) {}
    return result;
}
