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
#include <unordered_map>
#include "../backend_test_fixture.hpp"
#include "parity_test_utils.hpp"

using namespace tenzor;
using namespace tenzor::testing;


class MoEHRMParity : public BackendTest {};
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
TEST_P(MoEHRMParity, MixtureOfExperts_Forward) {
    // Small config: input_dim=16, hidden_dim=32, num_experts=4, top_k=2.
    nn::MixtureOfExperts moe_cpu(16, 32, 4, 2, 1.25, 0.01, 0.0);
    moe_cpu.eval();

    auto input = randn({2, 8, 16}, DType::Float32, Device::cpu());
    auto ref = moe_cpu.forward(Variable(input, false)).tensor();

    auto backends = get_available_backends();
    REQUIRE_MULTI_BACKEND_OR_SKIP("moe/hrm parity");

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
            ADD_FAILURE() << "MoE failed on " << backend_name(backends[i])
                      << ": " << e.what() << std::endl;
        }
    }
}

// ============================================================================
// HRM (Hierarchical Reasoning Module)
// ============================================================================

TEST_P(MoEHRMParity, HRM_Forward) {
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
    // Audit: previously wrapped in try{...}catch(...){GTEST_SKIP("HRM CPU
    // reference failed")}. The CPU forward is the parity reference (the MoE
    // forward/backward tests in this file run it un-wrapped); a CPU reference
    // failure is a real bug, not a "feature unavailable" skip. Let it propagate.
    Tensor ref = hrm_cpu.forward(Variable(input, false)).tensor();

    auto backends = get_available_backends();
    REQUIRE_MULTI_BACKEND_OR_SKIP("moe/hrm parity");

    for (size_t i = 1; i < backends.size(); ++i) {
        try {
            nn::HRM hrm_dev(cfg);
            hrm_dev.eval();
            auto params = hrm_cpu.parameters();
            auto dev_params = hrm_dev.parameters();
            for (size_t p = 0; p < params.size(); ++p) {
                dev_params[p]->tensor() = params[p]->tensor().clone();
            }
            // HRM registers h_init_state / l_init_state as non-trainable
            // buffers (random truncated-normal init kept fixed during
            // training). They are NOT in parameters() and differ between
            // fresh HRM(cfg) constructions, so the parameter-clone loop
            // above doesn't sync them — copy buffers too, by name, so the
            // CPU reference and the dev instance share initial state.
            auto cpu_bufs = hrm_cpu.named_buffers();
            auto dev_bufs = hrm_dev.named_buffers();
            std::unordered_map<std::string, std::shared_ptr<Variable>> dev_buf_map;
            for (auto& [name, var] : dev_bufs) dev_buf_map[name] = var;
            for (auto& [name, var] : cpu_bufs) {
                auto it = dev_buf_map.find(name);
                if (it != dev_buf_map.end()) {
                    it->second->tensor() = var->tensor().clone();
                }
            }
            hrm_dev.to(backends[i]);
            auto input_dev = input.to(backends[i]);
            auto out = hrm_dev.forward(Variable(input_dev, false)).tensor();
            backends[i].synchronize();
            SCOPED_TRACE(std::string("HRM on ") + backend_name(backends[i]));
            EXPECT_TENSORS_CLOSE(ref, out.to(Device::cpu()), 1e-2f, 1e-3f);
        } catch (const std::exception& e) {
            ADD_FAILURE() << "HRM failed on " << backend_name(backends[i])
                      << ": " << e.what() << std::endl;
        }
    }
}

// ============================================================================
// D1: MoE backward parity. Complements the forward test above by running
// loss.backward() on each backend and comparing the gradient of `input`
// against the CPU reference. MoE's backward is the recorded raw-tensor-op
// autograd-break hazard — if a backend reverts to that pattern the grad
// silently zeros out.
// ============================================================================
TEST_P(MoEHRMParity, MixtureOfExperts_Backward) {
    nn::MixtureOfExperts moe_cpu(16, 32, 4, 2, 1.25, 0.01, 0.0);
    moe_cpu.eval();
    Variable input_cpu(randn({2, 8, 16}, DType::Float32, Device::cpu()), true);
    auto out_cpu = moe_cpu.forward(input_cpu);
    sum(out_cpu).backward();
    ASSERT_TRUE(input_cpu.has_grad()) << "CPU reference backward produced no grad";
    auto ref_grad = input_cpu.grad()->contiguous();

    auto backends = get_available_backends();
    REQUIRE_MULTI_BACKEND_OR_SKIP("moe backward parity");

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
            Variable input_dev(input_cpu.tensor().to(backends[i]), true);
            auto out_dev = moe_dev.forward(input_dev);
            sum(out_dev).backward();
            backends[i].synchronize();

            SCOPED_TRACE(std::string("MoE backward on ") + backend_name(backends[i]));
            ASSERT_TRUE(input_dev.has_grad())
                << "backward produced no grad on " << backend_name(backends[i])
                << " — autograd graph may be broken";
            auto dev_grad = input_dev.grad()->to(Device::cpu()).contiguous();
            // Same tolerance as the forward test — routing gates inject
            // small numerical differences that compound through backward.
            EXPECT_TENSORS_CLOSE(ref_grad, dev_grad, 1e-2f, 1e-3f);
        } catch (const std::exception& e) {
            ADD_FAILURE() << "MoE backward failed on " << backend_name(backends[i])
                          << ": " << e.what() << std::endl;
        }
    }
}

INSTANTIATE_BACKEND_TESTS(MoEHRMParity);


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
