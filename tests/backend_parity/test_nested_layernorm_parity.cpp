/**
 * @file test_nested_layernorm_parity.cpp
 * @brief Backend parity for nested_layer_norm with a NON-power-of-two inner dim.
 *
 * The existing test_nested_parity.cpp exercises nested_layer_norm only with
 * power-of-two inner dimensions (D=8). The per-row mean/variance reduction in
 * the nested LN kernels is implemented as a tree reduction; a tree reduction
 * that assumes the reduction width is a power of two (e.g. halving the active
 * lane count and never handling the leftover odd lane) silently drops the tail
 * elements of every row whenever D is not a power of two. The ROCm nested LN
 * tree-reduction fix made that reduction handle arbitrary widths; this file
 * guards it by running with several non-power-of-two inner dimensions and
 * comparing every available GPU backend's forward AND backward to the CPU
 * reference.
 *
 * Layout matches test_nested_parity.cpp: nested tensors are a [total_len, D]
 * values tensor plus an int64 offsets tensor [B+1] delimiting per-batch rows.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/autograd/nested_ops.hpp>
#include "../backend_test_fixture.hpp"
#include "parity_test_utils.hpp"

using namespace tenzor;
using namespace tenzor::testing;
using namespace tenzor::autograd;

class NestedLayerNormParity : public BackendTest {};

namespace {

// Build an int64 offsets tensor on the CPU for batch sizes `lens`.
Tensor make_offsets_cpu(const std::vector<int64_t>& lens) {
    std::vector<int64_t> offs(lens.size() + 1, 0);
    for (size_t i = 0; i < lens.size(); ++i) offs[i + 1] = offs[i] + lens[i];
    auto t = zeros({static_cast<int64_t>(offs.size())}, DType::Int64, Device::cpu());
    auto* p = t.data<int64_t>();
    for (size_t i = 0; i < offs.size(); ++i) p[i] = offs[i];
    return t;
}

// Run nested_layer_norm forward + backward for a given inner dim D, comparing
// each available backend to the CPU reference. D is deliberately chosen to be
// a non-power-of-two so the per-row reduction's tail handling is exercised.
void run_layernorm_parity(int64_t D, const std::vector<int64_t>& lens) {
    SCOPED_TRACE(::testing::Message() << "nested_layer_norm D=" << D);

    auto offsets_cpu = make_offsets_cpu(lens);
    int64_t total_len = 0;
    for (auto l : lens) total_len += l;

    // Non-trivial weight/bias so the affine transform contributes to grads and
    // weight/bias gradients are non-zero (a reduction that drops tail channels
    // would corrupt the per-channel weight/bias grads as well as the output).
    auto values_cpu = randn({total_len, D}, DType::Float32, Device::cpu());
    auto weight_cpu = randn({D}, DType::Float32, Device::cpu());
    auto bias_cpu   = randn({D}, DType::Float32, Device::cpu());

    auto backends = get_available_backends();
    REQUIRE_MULTI_BACKEND_OR_SKIP("nested layernorm parity");

    // CPU is ground truth — a throw here is a real bug, so let it propagate.
    Tensor ref_out, ref_grad_values, ref_grad_weight, ref_grad_bias;
    {
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
            ADD_FAILURE() << "NestedLayerNorm (D=" << D << ") failed on "
                          << backend_name(backends[i]) << ": " << e.what();
        }
    }
}

}  // namespace

// Prime (non-power-of-two) inner dim: the canonical regression case. A
// power-of-two-only tree reduction drops channels 8..12 of every row.
TEST_P(NestedLayerNormParity, NonPowerOfTwoInnerDim_13) {
    run_layernorm_parity(/*D=*/13, /*lens=*/{4, 6, 3});
}

// Odd, larger-than-typical-warp width to exercise multi-step tree reductions
// where the leftover lane appears at more than one reduction level.
TEST_P(NestedLayerNormParity, NonPowerOfTwoInnerDim_37) {
    run_layernorm_parity(/*D=*/37, /*lens=*/{5, 2, 7});
}

// Just-above-a-power-of-two (33 = 32 + 1): the single tail element past the
// 32-wide reduction boundary must still be folded in.
TEST_P(NestedLayerNormParity, NonPowerOfTwoInnerDim_33) {
    run_layernorm_parity(/*D=*/33, /*lens=*/{3, 8});
}

INSTANTIATE_BACKEND_TESTS(NestedLayerNormParity);

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
