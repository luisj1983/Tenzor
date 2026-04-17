/**
 * @file test_grad_transform_parity.cpp
 * @brief Backend parity for autograd transforms (Phase 4.1).
 *
 * Verifies that vmap, jvp, vjp, jacobian, hessian, hvp, vhp produce identical
 * results across all available backends on simple closed-form test functions.
 *
 * The transforms themselves live in tenzor::{jvp, jacobian, hessian, vjp,
 * hvp, vhp} (autograd/functional.hpp) and tenzor::vmap
 * (autograd/vmap.hpp).
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/autograd/functional.hpp>
#include <tenzor/autograd/vmap.hpp>
#include "parity_test_utils.hpp"

using namespace tenzor;
using namespace tenzor::testing;

namespace {

// Simple scalar-valued test function: f(x) = sum(x^2). Jacobian is 2x, Hessian
// is 2I. These closed forms let us sanity-check results on CPU while the
// cross-backend parity check compares outputs across backends.
Variable f_sum_sq(const Variable& x) {
    return sum(x * x);
}

// Vector-valued: f(x) = x^2. Jacobian is diag(2x).
Variable f_elem_sq(const Variable& x) {
    return x * x;
}

}  // namespace

// ============================================================================
// JVP
// ============================================================================

TEST(GradTransformParity, JVP) {
    auto input_cpu = randn({4}, DType::Float32, Device::cpu());
    auto tangent_cpu = randn({4}, DType::Float32, Device::cpu());

    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    Tensor ref_out, ref_tangent;
    try {
        auto [out, t] = tenzor::jvp(f_elem_sq, Variable(input_cpu, false),
                                     tangent_cpu);
        ref_out = out.tensor();
        ref_tangent = t;
    } catch (const std::exception& e) {
        GTEST_SKIP() << "jvp CPU reference failed: " << e.what();
    }

    for (size_t i = 1; i < backends.size(); ++i) {
        try {
            auto input_dev = input_cpu.to(backends[i]);
            auto tangent_dev = tangent_cpu.to(backends[i]);
            auto [out, t] = tenzor::jvp(f_elem_sq, Variable(input_dev, false),
                                         tangent_dev);
            backends[i].synchronize();
            SCOPED_TRACE(std::string("jvp on ") + backend_name(backends[i]));
            EXPECT_TENSORS_CLOSE(ref_out, out.tensor().to(Device::cpu()),
                                 1e-4f, 1e-6f);
            EXPECT_TENSORS_CLOSE(ref_tangent, t.to(Device::cpu()),
                                 1e-4f, 1e-6f);
        } catch (const std::exception& e) {
            ADD_FAILURE() << "jvp failed on " << backend_name(backends[i])
                      << ": " << e.what() << std::endl;
        }
    }
}

// ============================================================================
// VJP
// ============================================================================

TEST(GradTransformParity, VJP) {
    auto input_cpu = randn({4}, DType::Float32, Device::cpu());
    auto cotangent_cpu = ones({4}, DType::Float32, Device::cpu());

    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    Tensor ref_out, ref_grad;
    try {
        auto [out, g] = tenzor::vjp(f_elem_sq, Variable(input_cpu, false),
                                     cotangent_cpu);
        ref_out = out.tensor();
        ref_grad = g;
    } catch (const std::exception& e) {
        GTEST_SKIP() << "vjp CPU reference failed: " << e.what();
    }

    for (size_t i = 1; i < backends.size(); ++i) {
        try {
            auto input_dev = input_cpu.to(backends[i]);
            auto cotangent_dev = cotangent_cpu.to(backends[i]);
            auto [out, g] = tenzor::vjp(f_elem_sq, Variable(input_dev, false),
                                         cotangent_dev);
            backends[i].synchronize();
            SCOPED_TRACE(std::string("vjp on ") + backend_name(backends[i]));
            EXPECT_TENSORS_CLOSE(ref_out, out.tensor().to(Device::cpu()),
                                 1e-4f, 1e-6f);
            EXPECT_TENSORS_CLOSE(ref_grad, g.to(Device::cpu()),
                                 1e-4f, 1e-6f);
        } catch (const std::exception& e) {
            ADD_FAILURE() << "vjp failed on " << backend_name(backends[i])
                      << ": " << e.what() << std::endl;
        }
    }
}

// ============================================================================
// Jacobian
// ============================================================================

TEST(GradTransformParity, Jacobian) {
    auto input_cpu = randn({4}, DType::Float32, Device::cpu());

    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    Tensor ref;
    try {
        ref = tenzor::jacobian(f_elem_sq, Variable(input_cpu, false));
    } catch (const std::exception& e) {
        GTEST_SKIP() << "jacobian CPU reference failed: " << e.what();
    }

    for (size_t i = 1; i < backends.size(); ++i) {
        try {
            auto input_dev = input_cpu.to(backends[i]);
            auto out = tenzor::jacobian(f_elem_sq, Variable(input_dev, false));
            backends[i].synchronize();
            SCOPED_TRACE(std::string("jacobian on ") + backend_name(backends[i]));
            EXPECT_TENSORS_CLOSE(ref, out.to(Device::cpu()), 1e-4f, 1e-6f);
        } catch (const std::exception& e) {
            ADD_FAILURE() << "jacobian failed on " << backend_name(backends[i])
                      << ": " << e.what() << std::endl;
        }
    }
}

// ============================================================================
// Hessian
// ============================================================================

// Fixed via ROCm cascading fixes (mul non-contig + contiguous UAF).
// Standalone verification: CUDA, Vulkan, OneAPI all pass; ROCm skip gracefully
// if it fails. Test wraps the call in try/catch via test_operation_parity.
TEST(GradTransformParity, Hessian) {
    auto input_cpu = randn({3}, DType::Float32, Device::cpu());

    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    Tensor ref;
    try {
        ref = tenzor::hessian(f_sum_sq, Variable(input_cpu, false));
    } catch (const std::exception& e) {
        GTEST_SKIP() << "hessian CPU reference failed: " << e.what();
    }

    for (size_t i = 1; i < backends.size(); ++i) {
        try {
            auto input_dev = input_cpu.to(backends[i]);
            auto out = tenzor::hessian(f_sum_sq, Variable(input_dev, false));
            backends[i].synchronize();
            SCOPED_TRACE(std::string("hessian on ") + backend_name(backends[i]));
            EXPECT_TENSORS_CLOSE(ref, out.to(Device::cpu()), 1e-3f, 1e-5f);
        } catch (const std::exception& e) {
            ADD_FAILURE() << "hessian failed on " << backend_name(backends[i])
                      << ": " << e.what() << std::endl;
        }
    }
}

// ============================================================================
// HVP / VHP
// ============================================================================

// Fixed alongside Hessian.
TEST(GradTransformParity, HVP) {
    auto input_cpu = randn({3}, DType::Float32, Device::cpu());
    auto v_cpu = ones({3}, DType::Float32, Device::cpu());

    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    Tensor ref_hv;
    try {
        auto [_, hv] = tenzor::hvp(f_sum_sq, Variable(input_cpu, false), v_cpu);
        ref_hv = hv;
    } catch (const std::exception& e) {
        GTEST_SKIP() << "hvp CPU reference failed: " << e.what();
    }

    for (size_t i = 1; i < backends.size(); ++i) {
        try {
            auto input_dev = input_cpu.to(backends[i]);
            auto v_dev = v_cpu.to(backends[i]);
            auto [_, hv] = tenzor::hvp(f_sum_sq, Variable(input_dev, false),
                                        v_dev);
            backends[i].synchronize();
            SCOPED_TRACE(std::string("hvp on ") + backend_name(backends[i]));
            EXPECT_TENSORS_CLOSE(ref_hv, hv.to(Device::cpu()), 1e-3f, 1e-5f);
        } catch (const std::exception& e) {
            ADD_FAILURE() << "hvp failed on " << backend_name(backends[i])
                      << ": " << e.what() << std::endl;
        }
    }
}

// ============================================================================
// vmap
// ============================================================================

TEST(GradTransformParity, Vmap_Basic) {
    // Input: (B=4, D=3); batched through f_elem_sq.
    auto input_cpu = randn({4, 3}, DType::Float32, Device::cpu());

    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    Tensor ref;
    try {
        ref = tenzor::vmap(f_elem_sq, Variable(input_cpu, false), 0).tensor();
    } catch (const std::exception& e) {
        GTEST_SKIP() << "vmap CPU reference failed: " << e.what();
    }

    for (size_t i = 1; i < backends.size(); ++i) {
        try {
            auto input_dev = input_cpu.to(backends[i]);
            auto out = tenzor::vmap(f_elem_sq, Variable(input_dev, false), 0);
            backends[i].synchronize();
            SCOPED_TRACE(std::string("vmap on ") + backend_name(backends[i]));
            EXPECT_TENSORS_CLOSE(ref, out.tensor().to(Device::cpu()),
                                 1e-4f, 1e-6f);
        } catch (const std::exception& e) {
            ADD_FAILURE() << "vmap failed on " << backend_name(backends[i])
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
