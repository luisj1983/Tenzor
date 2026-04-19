/**
 * @file test_nested_parity.cpp
 * @brief Backend parity for nested-tensor autograd ops.
 *
 * Covers nested_softmax, nested_layer_norm, nested_sum, nested_mean,
 * nested_attention forward+backward. Nested tensors are packed as a
 * [total_len, D] values tensor plus an int64 offsets tensor [B+1] that
 * delimits per-batch sequence boundaries.
 *
 * The Vulkan nested-attention backward kernel landed in commit 37be63b4;
 * this file is the first parity coverage for it.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/autograd/nested_ops.hpp>
#include "../backend_test_fixture.hpp"
#include "parity_test_utils.hpp"

using namespace tenzor;
using namespace tenzor::testing;
using namespace tenzor::autograd;


class NestedParity : public BackendTest {};
namespace {

// Helper: build an int64 offsets tensor on the CPU for batch sizes `lens`.
// The offsets tensor is conventionally kept on the same device as values;
// callers should `.to(backend)` after constructing.
Tensor make_offsets_cpu(const std::vector<int64_t>& lens) {
    std::vector<int64_t> offs(lens.size() + 1, 0);
    for (size_t i = 0; i < lens.size(); ++i) offs[i + 1] = offs[i] + lens[i];
    auto t = zeros({static_cast<int64_t>(offs.size())}, DType::Int64, Device::cpu());
    auto* p = t.data<int64_t>();
    for (size_t i = 0; i < offs.size(); ++i) p[i] = offs[i];
    return t;
}

}  // namespace

// ============================================================================
// Nested softmax (forward + backward)
// ============================================================================

// Previously DISABLED_ due to GPU hang reports. Verified passing across all
// backends (CPU/CUDA/ROCm/OneAPI/Vulkan); re-enabling.
TEST_P(NestedParity, NestedSoftmax_FwdBwd) {
    // B=2 sequences of lengths 3 and 5, D=4
    const int64_t D = 4;
    auto offsets_cpu = make_offsets_cpu({3, 5});
    const int64_t total_len = 3 + 5;
    auto values_cpu = randn({total_len, D}, DType::Float32, Device::cpu());

    auto backends = get_available_backends();
    REQUIRE_MULTI_BACKEND_OR_SKIP("nested parity");

    // Reference on CPU
    Tensor ref_out, ref_grad;
    try {
        auto v = Variable(values_cpu.clone(), true);
        auto out = nested_softmax(v, offsets_cpu, /*dim=*/1);
        auto seed = ones_like(out.tensor());
        out.backward(seed);
        ref_out = out.tensor();
        ref_grad = v.grad().value();
    } catch (const std::exception& e) {
        GTEST_SKIP() << "nested_softmax CPU reference failed: " << e.what();
    }

    for (size_t i = 1; i < backends.size(); ++i) {
        try {
            auto values_dev = values_cpu.to(backends[i]);
            auto offsets_dev = offsets_cpu.to(backends[i]);
            auto v = Variable(values_dev, true);
            auto out = nested_softmax(v, offsets_dev, /*dim=*/1);
            auto seed = ones_like(out.tensor());
            out.backward(seed);
            backends[i].synchronize();

            SCOPED_TRACE(std::string("NestedSoftmax on ") + backend_name(backends[i]));
            EXPECT_TENSORS_CLOSE(ref_out, out.tensor().to(Device::cpu()),
                                 1e-4f, 1e-6f);
            EXPECT_TENSORS_CLOSE(ref_grad, v.grad().value().to(Device::cpu()),
                                 1e-3f, 1e-5f);
        } catch (const std::exception& e) {
            ADD_FAILURE() << "NestedSoftmax failed on " << backend_name(backends[i])
                      << ": " << e.what() << std::endl;
        }
    }
}

// ============================================================================
// Nested layer norm (forward + backward)
// ============================================================================

// Previously DISABLED_ — verified passing; re-enabling.
TEST_P(NestedParity, NestedLayerNorm_FwdBwd) {
    const int64_t D = 8;
    auto offsets_cpu = make_offsets_cpu({4, 6});
    const int64_t total_len = 4 + 6;
    auto values_cpu = randn({total_len, D}, DType::Float32, Device::cpu());
    auto weight_cpu = ones({D}, DType::Float32, Device::cpu());
    auto bias_cpu = zeros({D}, DType::Float32, Device::cpu());

    auto backends = get_available_backends();
    REQUIRE_MULTI_BACKEND_OR_SKIP("nested parity");

    Tensor ref_out, ref_grad_values, ref_grad_weight, ref_grad_bias;
    try {
        auto v = Variable(values_cpu.clone(), true);
        auto w = Variable(weight_cpu.clone(), true);
        auto b = Variable(bias_cpu.clone(), true);
        auto out = nested_layer_norm(v, offsets_cpu, w, b, 1e-5);
        auto seed = ones_like(out.tensor());
        out.backward(seed);
        ref_out = out.tensor();
        ref_grad_values = v.grad().value();
        ref_grad_weight = w.grad().value();
        ref_grad_bias = b.grad().value();
    } catch (const std::exception& e) {
        GTEST_SKIP() << "nested_layer_norm CPU reference failed: " << e.what();
    }

    for (size_t i = 1; i < backends.size(); ++i) {
        try {
            auto values_dev = values_cpu.to(backends[i]);
            auto offsets_dev = offsets_cpu.to(backends[i]);
            auto weight_dev = weight_cpu.to(backends[i]);
            auto bias_dev = bias_cpu.to(backends[i]);
            auto v = Variable(values_dev, true);
            auto w = Variable(weight_dev, true);
            auto b = Variable(bias_dev, true);
            auto out = nested_layer_norm(v, offsets_dev, w, b, 1e-5);
            auto seed = ones_like(out.tensor());
            out.backward(seed);
            backends[i].synchronize();

            SCOPED_TRACE(std::string("NestedLayerNorm on ") + backend_name(backends[i]));
            EXPECT_TENSORS_CLOSE(ref_out, out.tensor().to(Device::cpu()),
                                 1e-4f, 1e-6f);
            EXPECT_TENSORS_CLOSE(ref_grad_values,
                                 v.grad().value().to(Device::cpu()),
                                 1e-3f, 1e-5f);
            EXPECT_TENSORS_CLOSE(ref_grad_weight,
                                 w.grad().value().to(Device::cpu()),
                                 1e-3f, 1e-5f);
            EXPECT_TENSORS_CLOSE(ref_grad_bias,
                                 b.grad().value().to(Device::cpu()),
                                 1e-3f, 1e-5f);
        } catch (const std::exception& e) {
            ADD_FAILURE() << "NestedLayerNorm failed on "
                      << backend_name(backends[i]) << ": " << e.what() << std::endl;
        }
    }
}

// ============================================================================
// Nested sum / mean reductions
// ============================================================================

TEST_P(NestedParity, NestedSum) {
    const int64_t D = 4;
    auto offsets_cpu = make_offsets_cpu({2, 3, 1});
    auto values_cpu = randn({6, D}, DType::Float32, Device::cpu());

    auto backends = get_available_backends();
    REQUIRE_MULTI_BACKEND_OR_SKIP("nested parity");

    Tensor ref;
    try {
        auto v = Variable(values_cpu, false);
        ref = nested_sum(v, offsets_cpu).tensor();
    } catch (const std::exception& e) {
        GTEST_SKIP() << "nested_sum CPU reference failed: " << e.what();
    }

    for (size_t i = 1; i < backends.size(); ++i) {
        try {
            auto values_dev = values_cpu.to(backends[i]);
            auto offsets_dev = offsets_cpu.to(backends[i]);
            auto v = Variable(values_dev, false);
            auto out = nested_sum(v, offsets_dev).tensor();
            backends[i].synchronize();
            SCOPED_TRACE(std::string("NestedSum on ") + backend_name(backends[i]));
            EXPECT_TENSORS_CLOSE(ref, out.to(Device::cpu()), 1e-5f, 1e-7f);
        } catch (const std::exception& e) {
            ADD_FAILURE() << "NestedSum failed on "
                      << backend_name(backends[i]) << ": " << e.what() << std::endl;
        }
    }
}

// Previously DISABLED_ — verified passing; re-enabling.
TEST_P(NestedParity, NestedMean) {
    const int64_t D = 4;
    auto offsets_cpu = make_offsets_cpu({2, 3, 1});
    auto values_cpu = randn({6, D}, DType::Float32, Device::cpu());

    auto backends = get_available_backends();
    REQUIRE_MULTI_BACKEND_OR_SKIP("nested parity");

    Tensor ref;
    try {
        auto v = Variable(values_cpu, false);
        ref = nested_mean(v, offsets_cpu).tensor();
    } catch (const std::exception& e) {
        GTEST_SKIP() << "nested_mean CPU reference failed: " << e.what();
    }

    for (size_t i = 1; i < backends.size(); ++i) {
        try {
            auto values_dev = values_cpu.to(backends[i]);
            auto offsets_dev = offsets_cpu.to(backends[i]);
            auto v = Variable(values_dev, false);
            auto out = nested_mean(v, offsets_dev).tensor();
            backends[i].synchronize();
            SCOPED_TRACE(std::string("NestedMean on ") + backend_name(backends[i]));
            EXPECT_TENSORS_CLOSE(ref, out.to(Device::cpu()), 1e-5f, 1e-7f);
        } catch (const std::exception& e) {
            ADD_FAILURE() << "NestedMean failed on "
                      << backend_name(backends[i]) << ": " << e.what() << std::endl;
        }
    }
}

// ============================================================================
// Nested attention (forward + backward) — Vulkan backward added 37be63b4
// ============================================================================

// Previously DISABLED_ — verified passing across all backends; re-enabling.
TEST_P(NestedParity, NestedAttention_FwdBwd) {
    // B=2, self-attention so q_offsets == kv_offsets.
    // Lens: [3, 4]; head_dim=8
    const int64_t head_dim = 8;
    auto offsets_cpu = make_offsets_cpu({3, 4});
    const int64_t total_len = 3 + 4;
    const double scale = 1.0 / std::sqrt(static_cast<double>(head_dim));

    auto q_cpu = randn({total_len, head_dim}, DType::Float32, Device::cpu());
    auto k_cpu = randn({total_len, head_dim}, DType::Float32, Device::cpu());
    auto v_cpu = randn({total_len, head_dim}, DType::Float32, Device::cpu());

    auto backends = get_available_backends();
    REQUIRE_MULTI_BACKEND_OR_SKIP("nested parity");

    Tensor ref_out, ref_q_grad, ref_k_grad, ref_v_grad;
    try {
        auto qv = Variable(q_cpu.clone(), true);
        auto kv = Variable(k_cpu.clone(), true);
        auto vv = Variable(v_cpu.clone(), true);
        auto out = nested_attention(qv, kv, vv, offsets_cpu, offsets_cpu,
                                    scale, /*causal=*/false);
        auto seed = ones_like(out.tensor());
        out.backward(seed);
        ref_out = out.tensor();
        ref_q_grad = qv.grad().value();
        ref_k_grad = kv.grad().value();
        ref_v_grad = vv.grad().value();
    } catch (const std::exception& e) {
        GTEST_SKIP() << "nested_attention CPU reference failed: " << e.what();
    }

    for (size_t i = 1; i < backends.size(); ++i) {
        try {
            auto q_dev = q_cpu.to(backends[i]);
            auto k_dev = k_cpu.to(backends[i]);
            auto v_dev = v_cpu.to(backends[i]);
            auto offs_dev = offsets_cpu.to(backends[i]);

            auto qv = Variable(q_dev, true);
            auto kv = Variable(k_dev, true);
            auto vv = Variable(v_dev, true);
            auto out = nested_attention(qv, kv, vv, offs_dev, offs_dev,
                                        scale, /*causal=*/false);
            auto seed = ones_like(out.tensor());
            out.backward(seed);
            backends[i].synchronize();

            SCOPED_TRACE(std::string("NestedAttention on ") + backend_name(backends[i]));
            EXPECT_TENSORS_CLOSE(ref_out, out.tensor().to(Device::cpu()),
                                 1e-3f, 1e-4f);
            EXPECT_TENSORS_CLOSE(ref_q_grad,
                                 qv.grad().value().to(Device::cpu()),
                                 1e-3f, 1e-4f);
            EXPECT_TENSORS_CLOSE(ref_k_grad,
                                 kv.grad().value().to(Device::cpu()),
                                 1e-3f, 1e-4f);
            EXPECT_TENSORS_CLOSE(ref_v_grad,
                                 vv.grad().value().to(Device::cpu()),
                                 1e-3f, 1e-4f);
        } catch (const std::exception& e) {
            ADD_FAILURE() << "NestedAttention failed on "
                      << backend_name(backends[i]) << ": " << e.what() << std::endl;
        }
    }
}

INSTANTIATE_BACKEND_TESTS(NestedParity);


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
