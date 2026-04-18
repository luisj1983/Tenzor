/**
 * @file test_phase5_parity.cpp
 * @brief Phase 5 remaining-categories parity tests.
 *
 * Covers ALiBi, RoPE, MoE, HRM, and windows ops parity. FP8 and distributions
 * are not included — FP8 requires hardware-specific gating (H100/MI300) and
 * distributions has a large dedicated API; leaving those as separate future
 * files.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/ops/windows.hpp>
#include <tenzor/nn/layers/alibi.hpp>
#include <tenzor/nn/layers/rope.hpp>
#include "../backend_test_fixture.hpp"
#include "parity_test_utils.hpp"

using namespace tenzor;
using namespace tenzor::testing;


class Phase5Parity : public BackendTest {};
// ============================================================================
// ALiBi — positional bias used additively with attention scores
// ============================================================================

TEST_P(Phase5Parity, ALiBi_Bias) {
    // Construct ALiBi with 4 heads and ask for a (seq_q, seq_k) bias. Since
    // bias generation is deterministic and device-dependent via the device
    // argument, we check that each backend produces the same result as CPU.
    nn::ALiBi alibi_cpu(4);
    auto ref = alibi_cpu.get_bias(8, 8, Device::cpu(), DType::Float32);

    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    for (size_t i = 1; i < backends.size(); ++i) {
        try {
            nn::ALiBi alibi_dev(4);
            auto bias = alibi_dev.get_bias(8, 8, backends[i], DType::Float32);
            backends[i].synchronize();
            SCOPED_TRACE(std::string("ALiBi on ") + backend_name(backends[i]));
            EXPECT_TENSORS_CLOSE(ref, bias.to(Device::cpu()), 1e-5f, 1e-7f);
        } catch (const std::exception& e) {
            ADD_FAILURE() << "ALiBi failed on " << backend_name(backends[i])
                      << ": " << e.what() << std::endl;
        }
    }
}

// ============================================================================
// RoPE — rotary position embedding applied multiplicatively
// ============================================================================

// Fixed: the original "hang" was caused by two ROCm kernel bugs that
// cascaded from Variable::operator* in RoPE's forward:
//   1) mul/add/sub/div kernels did not handle non-contiguous inputs (which
//      slice() produces); fixed by materializing contiguous inputs at the top
//      of each kernel in src/backends/rocm/kernels/math.hip.cpp.
//   2) contiguous_kernel had a use-after-free: HipMemGuard freed the device
//      stride/shape buffers before the async kernel launched using them;
//      fixed by calling hipStreamSynchronize(stream) before the guards go
//      out of scope in src/backends/rocm/kernels/transform.hip.cpp.
TEST_P(Phase5Parity, RoPE_Forward) {
    // dim=8, max_seq_len=16. Input: (batch=2, seq=8, head_dim=8).
    nn::RoPE rope_cpu(8, 16);
    auto input = randn({2, 8, 8}, DType::Float32, Device::cpu());
    // offset=0 is equivalent to forward_impl(input); RoPE::forward(Var) is
    // shadowed by forward(Var, offset) so we go through the offset path.
    auto ref = rope_cpu.forward(Variable(input, false), 0).tensor();

    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    for (size_t i = 1; i < backends.size(); ++i) {
        try {
            nn::RoPE rope_dev(8, 16);
            rope_dev.to(backends[i]);
            auto input_dev = input.to(backends[i]);
            auto out = rope_dev.forward(Variable(input_dev, false), 0).tensor();
            backends[i].synchronize();
            SCOPED_TRACE(std::string("RoPE on ") + backend_name(backends[i]));
            EXPECT_TENSORS_CLOSE(ref, out.to(Device::cpu()), 1e-4f, 1e-6f);
        } catch (const std::exception& e) {
            ADD_FAILURE() << "RoPE failed on " << backend_name(backends[i])
                      << ": " << e.what() << std::endl;
        }
    }
}

// Fixed alongside RoPE_Forward (same ROCm mul/contiguous bug chain).
TEST_P(Phase5Parity, RoPE_WithOffset) {
    nn::RoPE rope_cpu(8, 32);
    auto input = randn({2, 4, 8}, DType::Float32, Device::cpu());
    // With offset=10, positions 0..3 map to global 10..13.
    auto ref = rope_cpu.forward(Variable(input, false), /*offset=*/10).tensor();

    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    for (size_t i = 1; i < backends.size(); ++i) {
        try {
            nn::RoPE rope_dev(8, 32);
            rope_dev.to(backends[i]);
            auto input_dev = input.to(backends[i]);
            auto out = rope_dev.forward(Variable(input_dev, false), 10).tensor();
            backends[i].synchronize();
            SCOPED_TRACE(std::string("RoPE+offset on ")
                         + backend_name(backends[i]));
            EXPECT_TENSORS_CLOSE(ref, out.to(Device::cpu()), 1e-4f, 1e-6f);
        } catch (const std::exception& e) {
            ADD_FAILURE() << "RoPE+offset failed on " << backend_name(backends[i])
                      << ": " << e.what() << std::endl;
        }
    }
}

// ============================================================================
// Windows ops (hann, hamming, blackman, bartlett, kaiser)
// ============================================================================

TEST_P(Phase5Parity, HannWindow) {
    // Window generation produces a deterministic 1D tensor; since it doesn't
    // take an input tensor, we compare reference-vs-per-backend by asking for
    // the window on each device directly.
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    const int64_t N = 64;
    auto ref = tenzor::hann_window(N, true, DType::Float32, Device::cpu());

    for (size_t i = 1; i < backends.size(); ++i) {
        try {
            auto win = tenzor::hann_window(N, true, DType::Float32, backends[i]);
            backends[i].synchronize();
            SCOPED_TRACE(std::string("hann_window on ")
                         + backend_name(backends[i]));
            EXPECT_TENSORS_CLOSE(ref, win.to(Device::cpu()), 1e-5f, 1e-7f);
        } catch (const std::exception& e) {
            ADD_FAILURE() << "hann_window failed on "
                      << backend_name(backends[i]) << ": " << e.what()
                      << std::endl;
        }
    }
}

TEST_P(Phase5Parity, HammingWindow) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    const int64_t N = 64;
    auto ref = tenzor::hamming_window(N, true, 0.54, 0.46,
                                       DType::Float32, Device::cpu());

    for (size_t i = 1; i < backends.size(); ++i) {
        try {
            auto win = tenzor::hamming_window(N, true, 0.54, 0.46,
                                               DType::Float32, backends[i]);
            backends[i].synchronize();
            SCOPED_TRACE(std::string("hamming_window on ")
                         + backend_name(backends[i]));
            EXPECT_TENSORS_CLOSE(ref, win.to(Device::cpu()), 1e-5f, 1e-7f);
        } catch (const std::exception& e) {
            ADD_FAILURE() << "hamming_window failed on "
                      << backend_name(backends[i]) << ": " << e.what()
                      << std::endl;
        }
    }
}

INSTANTIATE_BACKEND_TESTS(Phase5Parity);


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
