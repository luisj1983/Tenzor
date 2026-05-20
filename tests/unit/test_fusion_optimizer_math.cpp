/**
 * @file test_fusion_optimizer_math.cpp
 * @brief Numerical-correctness tests for FusionOptimizer::execute_fused_op
 *
 * Audit item A.1 (Tenzor 1.0 pre-release remediation): the previously
 * shipped execute_fused_op composed three fused operations with the wrong
 * math:
 *
 *  - "conv_bn_relu" wrapped fused_conv2d_relu+fused_batchnorm_relu, which
 *    inserts an extra ReLU between Conv and BatchNorm (the standard
 *    Conv→BN→ReLU contract).
 *  - "elementwise_chain" applied ReLU via fused_add_relu(result, zeros_like)
 *    which materialised a zero tensor on every call ("fake fusion").
 *  - "attention" multiplied the mask with the softmax output instead of
 *    adding it to the scores before softmax — leaks probability mass on
 *    masked positions.
 *
 * These tests pin the correct mathematical contract by comparing the
 * fused execution against an autograd-graph reference computation.
 */

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "tenzor/autograd/variable.hpp"
#include "tenzor/backend/fast_dispatch.hpp"
#include "tenzor/backend/op_attributes.hpp"
#include "tenzor/core/dtype.hpp"
#include "tenzor/core/device.hpp"
#include "tenzor/core/tensor.hpp"
#include "tenzor/nn/functional.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/fused_ops.hpp"
#include "tenzor/ops/fusion_optimizer.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/op_id.hpp"
#include "tenzor/tenzor.hpp"

using namespace tenzor;
using namespace tenzor::ops;

namespace {

class FusionOptimizerMathTest : public ::testing::Test {
protected:
    void SetUp() override { tenzor::initialize(); }

    /// Compare two CPU/Float32 tensors elementwise within atol.
    static void expect_close(const Tensor& a,
                             const Tensor& b,
                             double atol,
                             const char* label) {
        ASSERT_EQ(a.shape().size(), b.shape().size()) << label;
        for (std::size_t i = 0; i < a.shape().size(); ++i) {
            ASSERT_EQ(a.shape()[i], b.shape()[i])
                << label << ": shape mismatch at dim " << i;
        }
        ASSERT_EQ(a.dtype(), DType::Float32) << label;
        ASSERT_EQ(b.dtype(), DType::Float32) << label;

        const auto* pa = a.data<float>();
        const auto* pb = b.data<float>();
        const std::int64_t n = a.numel();
        for (std::int64_t i = 0; i < n; ++i) {
            ASSERT_NEAR(pa[i], pb[i], atol)
                << label << ": mismatch at flat index " << i
                << " (fused=" << pa[i] << ", ref=" << pb[i] << ")";
        }
    }
};

// ---------------------------------------------------------------------------
// A.1 — conv_bn_relu must compute ReLU(BN(Conv(x,w,b))), not
//       ReLU(BN(ReLU(Conv(x,w,b)))).
//
// We construct inputs guaranteed to make some conv outputs negative
// (uniform negative weight) and verify the fused path matches the
// autograd-graph reference Conv→BN→ReLU.
// ---------------------------------------------------------------------------
TEST_F(FusionOptimizerMathTest, ConvBnReluMatchesAutogradReference) {
    const int64_t N = 1, C_in = 2, H = 4, W = 4;
    const int64_t C_out = 3, kH = 3, kW = 3;
    const int64_t stride = 1;
    const int64_t padding = 1;
    const float eps = 1e-5f;

    // Positive input, negative weight ⇒ conv output strictly negative
    // ⇒ inner ReLU bug clamps everything to zero ⇒ obvious divergence
    // from the correct Conv→BN→ReLU path.
    auto input = full({N, C_in, H, W}, 1.0f, DType::Float32, Device::cpu());
    auto weight = full({C_out, C_in, kH, kW}, -0.1f, DType::Float32, Device::cpu());
    auto bias = full({C_out}, 0.0f, DType::Float32, Device::cpu());

    // BN running stats / affine params chosen so that the BN output spans
    // both signs (final ReLU then becomes meaningful).
    auto bn_mean = full({C_out}, -1.0f, DType::Float32, Device::cpu());
    auto bn_var = full({C_out}, 1.0f, DType::Float32, Device::cpu());
    auto bn_gamma = full({C_out}, 1.0f, DType::Float32, Device::cpu());
    auto bn_beta = full({C_out}, 0.0f, DType::Float32, Device::cpu());

    // ----- Reference: composed Conv → BN-eval → ReLU via raw dispatch -----
    // We compare against the same backend kernels the fused path will dispatch
    // to (Conv2D + BatchNorm2dForwardAffine + ReLU), so any divergence is the
    // execute_fused_op composition bug — not an autograd-layer disagreement.
    Tensor reference;
    {
        std::vector<Tensor> conv_inputs = {input, weight, bias};
        OpAttributes conv_attrs;
        conv_attrs.set(AttrKey::Stride, static_cast<int64_t>(stride));
        conv_attrs.set(AttrKey::Padding, static_cast<int64_t>(padding));
        conv_attrs.set(AttrKey::StrideH, static_cast<int64_t>(stride));
        conv_attrs.set(AttrKey::StrideW, static_cast<int64_t>(stride));
        conv_attrs.set(AttrKey::PaddingH, static_cast<int64_t>(padding));
        conv_attrs.set(AttrKey::PaddingW, static_cast<int64_t>(padding));
        conv_attrs.set(AttrKey::HasBias, true);
        Tensor conv_out = dispatch(OpId::Conv2dForward, conv_inputs, conv_attrs)[0];

        std::vector<Tensor> bn_inputs = {conv_out, bn_mean, bn_var, bn_gamma, bn_beta};
        OpAttributes bn_attrs;
        bn_attrs.set(AttrKey::Eps, static_cast<double>(eps));
        Tensor bn_out = dispatch(OpId::BatchNorm2dForwardAffine, bn_inputs, bn_attrs)[0];

        std::vector<Tensor> relu_inputs = {bn_out};
        reference = dispatch(OpId::ReLU, relu_inputs)[0];
    }

    // ----- Fused: execute_fused_op("conv_bn_relu") -----
    // Canonical input order (matches the CPU dispatch kernel):
    //   [input, weight, conv_bias, bn_gamma, bn_beta, bn_running_mean, bn_running_var]
    std::vector<Tensor> fused_inputs = {
        input, weight, bias, bn_gamma, bn_beta, bn_mean, bn_var};
    std::unordered_map<std::string, std::string> attrs = {
        {"stride", std::to_string(stride)},
        {"padding", std::to_string(padding)},
        {"eps", std::to_string(eps)},
    };
    FusedOp fused_op("conv_bn_relu", {});
    auto fused_outputs = execute_fused_op(fused_op, fused_inputs, attrs);

    ASSERT_EQ(fused_outputs.size(), 1u);
    expect_close(fused_outputs[0], reference, 1e-4, "conv_bn_relu");
}

// ---------------------------------------------------------------------------
// A.1 — elementwise_chain with op_type=0 must compute ReLU((a+b)*c)
//       directly, not via fused_add_relu(result, zeros_like(result)).
// ---------------------------------------------------------------------------
TEST_F(FusionOptimizerMathTest, ElementwiseChainAddMulReluMatchesReference) {
    auto a = full({2, 3}, 1.0f, DType::Float32, Device::cpu());
    auto b = full({2, 3}, -2.0f, DType::Float32, Device::cpu());
    auto c = full({2, 3}, 1.5f, DType::Float32, Device::cpu());
    // (a + b) * c = (1 + (-2)) * 1.5 = -1.5  ⇒  ReLU = 0
    // Verifies that ReLU is genuinely applied (negative input → 0).

    Variable v_a(a, false), v_b(b, false), v_c(c, false);
    auto expected_var = nn::relu(v_a + v_b) * v_c;
    // Use mathematically equivalent form ReLU((a+b)*c) — for the case
    // a+b=-1, c=1.5, we want ReLU(-1.5)=0.
    Variable add_v = v_a + v_b;
    Variable mul_v = add_v * v_c;
    auto ref_var = nn::relu(mul_v);
    Tensor reference = ref_var.tensor();

    std::vector<Tensor> fused_inputs = {a, b, c};
    std::unordered_map<std::string, std::string> attrs = {{"op_type", "0"}};
    FusedOp fused_op("elementwise_chain", {});
    auto fused_outputs = execute_fused_op(fused_op, fused_inputs, attrs);

    ASSERT_EQ(fused_outputs.size(), 1u);
    expect_close(fused_outputs[0], reference, 1e-6, "elementwise_chain(op_type=0)");

    // op_type=1 path: (a * b) + c, then ReLU
    Variable mul1 = v_a * v_b;
    Variable add1 = mul1 + v_c;
    auto ref1 = nn::relu(add1).tensor();
    std::unordered_map<std::string, std::string> attrs1 = {{"op_type", "1"}};
    auto out1 = execute_fused_op(fused_op, fused_inputs, attrs1);
    ASSERT_EQ(out1.size(), 1u);
    expect_close(out1[0], ref1, 1e-6, "elementwise_chain(op_type=1)");
}

// ---------------------------------------------------------------------------
// A.1 — attention must compute softmax(scores + mask) @ V, where the mask
//       is additive (-inf at masked positions, 0 at kept positions).  The
//       previous code applied softmax first then multiplied the mask,
//       leaking probability mass.
// ---------------------------------------------------------------------------
TEST_F(FusionOptimizerMathTest, AttentionMaskAddedBeforeSoftmax) {
    const int64_t B = 1, Sq = 1, Sk = 4, D = 2;
    const float scale = 1.0f / std::sqrt(static_cast<float>(D));

    // Q, K, V are 3-D: [B, S, D]
    auto Q = full({B, Sq, D}, 1.0f, DType::Float32, Device::cpu());
    auto K = full({B, Sk, D}, 1.0f, DType::Float32, Device::cpu());
    auto V = full({B, Sk, D}, 0.0f, DType::Float32, Device::cpu());
    // Distinguish the V rows so the test detects mass leakage.
    {
        float* vp = V.data<float>();
        for (int64_t s = 0; s < Sk; ++s) {
            for (int64_t d = 0; d < D; ++d) {
                vp[s * D + d] = static_cast<float>(s + 1); // 1,2,3,4 (broadcast across D)
            }
        }
    }

    // Additive mask: 0 at kept positions, -inf at masked positions.
    // Mask out positions 2 and 3 entirely.
    auto mask = zeros({B, Sq, Sk}, DType::Float32, Device::cpu());
    {
        float* mp = mask.data<float>();
        const float neg_inf = -std::numeric_limits<float>::infinity();
        mp[2] = neg_inf;
        mp[3] = neg_inf;
    }

    // Reference: softmax(Q @ K^T * scale + mask) @ V
    // With Q=1, K=1, scale = 1/sqrt(2): scores = 2*scale = sqrt(2) at every
    // position before mask.  After mask: positions 0,1 -> sqrt(2);
    // 2,3 -> -inf. Softmax gives [0.5, 0.5, 0, 0].
    // Output: 0.5*1 + 0.5*2 + 0*3 + 0*4 = 1.5 at every D channel.
    const float expected_value = 1.5f;
    auto reference = full({B, Sq, D}, expected_value, DType::Float32, Device::cpu());

    std::vector<Tensor> fused_inputs = {Q, K, V, mask};
    std::unordered_map<std::string, std::string> attrs = {
        {"scale", std::to_string(scale)},
    };
    FusedOp fused_op("attention", {});
    auto fused_outputs = execute_fused_op(fused_op, fused_inputs, attrs);

    ASSERT_EQ(fused_outputs.size(), 1u);
    expect_close(fused_outputs[0], reference, 1e-5, "attention(masked)");
}

}  // namespace
