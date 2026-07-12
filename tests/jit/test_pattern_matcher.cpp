/**
 * @file test_pattern_matcher.cpp
 * @brief Direct unit coverage for PatternMatcher::find_all, focused on the
 *        LayerNorm/RMSNorm gamma/beta affine-absorption shape guard
 *        (JIT-R111).
 *
 * R1-05: match_layer_norm/match_rms_norm used to absorb a trailing Add/Mul
 * into the fused match purely by op-type + data-edge connectivity, with no
 * check that the operand becoming gamma/beta was actually a genuine
 * per-channel [norm_size] tensor. A full-tensor operand (a residual add
 * after a bias-less LayerNorm, or a gating multiply after RMSNorm) would be
 * silently absorbed and later indexed as gamma[i]/beta[i] for i in
 * [0, norm_size) by extended_codegen's generated kernel -- reading only the
 * operand's first norm_size elements and reusing that slice for every outer
 * instance. These tests build the decomposed op sequence directly via the
 * Graph API (bypassing the tracer, mirroring test_fusion_passes.cpp's
 * pattern) and assert PatternMatcher::find_all: (a) still fuses a genuine
 * per-channel affine operand (no regression), and (b) does NOT absorb a
 * full-tensor operand into the match (the fix).
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/jit/graph.hpp>
#include <tenzor/jit/pattern_matcher.hpp>

using namespace tenzor;
using namespace tenzor::jit;

namespace {
struct InitTenzorOnce : public ::testing::Environment {
    void SetUp() override { tenzor::initialize(); }
};
::testing::Environment* const _g_init =
    ::testing::AddGlobalTestEnvironment(new InitTenzorOnce);

// Shape [rows, C]; RMS/variance reduces the last axis (C = norm_size).
constexpr int64_t kRows = 2;
constexpr int64_t kC = 8;

// Builds x -> Pow(2) -> Mean(dim=-1) -> Add(eps) -> Sqrt -> Div(x, sqrt) ->
// Mul(divresult, affine_operand), i.e. a decomposed RMSNorm whose optional
// trailing gamma Mul's non-chain operand is `affine_shape` (defaults to a
// genuine per-channel [C] tensor; pass {kRows, kC} to simulate a full-tensor
// gate/residual instead).
auto build_decomposed_rms_norm(Graph& g, const std::vector<int64_t>& affine_shape)
    -> void {
    auto dev = Device::cpu();
    auto x = g.create_value("x", {kRows, kC}, DType::Float32, dev);

    auto sq_out = g.create_value("sq", {kRows, kC}, DType::Float32, dev);
    auto n0 = g.create_node(OpType::Pow, "sq_pow");
    n0->add_input(x);
    n0->set_attr("exponent", 2.0);
    sq_out->set_node(n0);
    n0->add_output(sq_out);
    g.add_node(n0);

    auto mean_out = g.create_value("mean", {kRows, 1}, DType::Float32, dev);
    auto n1 = g.create_node(OpType::Mean, "mean");
    n1->add_input(sq_out);
    n1->set_int_attr("dim", -1);
    n1->set_bool_attr("keepdim", true);
    mean_out->set_node(n1);
    n1->add_output(mean_out);
    g.add_node(n1);

    auto eps = g.create_value("eps", {}, DType::Float32, dev);
    auto eps_out = g.create_value("mean_eps", {kRows, 1}, DType::Float32, dev);
    auto n2 = g.create_node(OpType::Add, "add_eps");
    n2->add_input(mean_out);
    n2->add_input(eps);
    eps_out->set_node(n2);
    n2->add_output(eps_out);
    g.add_node(n2);

    auto sqrt_out = g.create_value("rms", {kRows, 1}, DType::Float32, dev);
    auto n3 = g.create_node(OpType::Sqrt, "sqrt");
    n3->add_input(eps_out);
    sqrt_out->set_node(n3);
    n3->add_output(sqrt_out);
    g.add_node(n3);

    auto norm_out = g.create_value("normalized", {kRows, kC}, DType::Float32, dev);
    auto n4 = g.create_node(OpType::Div, "normalize");
    n4->add_input(x);
    n4->add_input(sqrt_out);
    norm_out->set_node(n4);
    n4->add_output(norm_out);
    g.add_node(n4);

    auto affine = g.create_value("affine_operand", affine_shape, DType::Float32, dev);
    auto out = g.create_value("out", {kRows, kC}, DType::Float32, dev);
    auto n5 = g.create_node(OpType::Mul, "affine_mul");
    n5->add_input(norm_out);
    n5->add_input(affine);
    out->set_node(n5);
    n5->add_output(out);
    g.add_node(n5);

    g.set_inputs({x});
    g.set_outputs({out});
}

}  // namespace

TEST(PatternMatcherAffineShapeGuard, RmsNormAbsorbsGenuinePerChannelGamma) {
    Graph g;
    build_decomposed_rms_norm(g, /*affine_shape=*/{kC});  // [8], matches norm_size

    PatternMatcher matcher;
    auto matches = matcher.find_all(g);
    ASSERT_EQ(matches.size(), 1u) << "expected exactly one RMSNorm match";
    EXPECT_EQ(matches[0].kind, FusionKind::RMSNorm);
    EXPECT_EQ(matches[0].nodes.size(), 6u)
        << "genuine per-channel gamma must be absorbed into the match "
           "(Pow, Mean, Add, Sqrt, Div, Mul == 6 nodes)";
}

// R1-05/JIT-R111 regression: a full-tensor operand (same shape as the
// normalized input, e.g. a gating tensor or residual) must NOT be absorbed
// as gamma -- extended_codegen would otherwise index it as gamma[i] for
// i in [0, norm_size), silently truncating and broadcasting it.
TEST(PatternMatcherAffineShapeGuard, RmsNormRejectsFullTensorAsGamma) {
    Graph g;
    build_decomposed_rms_norm(g, /*affine_shape=*/{kRows, kC});  // [2,8] != [8]

    PatternMatcher matcher;
    auto matches = matcher.find_all(g);
    ASSERT_EQ(matches.size(), 1u)
        << "the RMS core (Pow/Mean/Add/Sqrt/Div) must still match";
    EXPECT_EQ(matches[0].kind, FusionKind::RMSNorm);
    EXPECT_EQ(matches[0].nodes.size(), 5u)
        << "a full-tensor operand must NOT be absorbed as gamma -- match "
           "should stop at Div (5 nodes), leaving the Mul unfused for the "
           "correct eager path";
    for (const auto& n : matches[0].nodes) {
        EXPECT_NE(n->op_type(), OpType::Mul)
            << "the affine Mul must not be part of the fused match";
    }
}

// findings.txt JIT-R117 regression: a same-NUMEL-different-SHAPE operand
// (e.g. [norm_size, 1] -- total element count equals norm_size, but that
// count sits in a LEADING axis, not the trailing one) must NOT be absorbed
// as gamma either. This is the exact gap the old `onumel == norm_size`-only
// check missed: it happens to equal norm_size in total count while being a
// legitimate per-ROW (not per-channel) broadcast operand in eager semantics
// (e.g. a per-token gate/keepdim=True reduction result), which
// extended_codegen would silently misread as gamma[i] and broadcast
// per-channel instead of per-row.
TEST(PatternMatcherAffineShapeGuard, RmsNormRejectsSameNumelDifferentShapeAsGamma) {
    Graph g;
    build_decomposed_rms_norm(g, /*affine_shape=*/{kC, 1});  // numel==kC but back()==1, not kC

    PatternMatcher matcher;
    auto matches = matcher.find_all(g);
    ASSERT_EQ(matches.size(), 1u)
        << "the RMS core (Pow/Mean/Add/Sqrt/Div) must still match";
    EXPECT_EQ(matches[0].kind, FusionKind::RMSNorm);
    EXPECT_EQ(matches[0].nodes.size(), 5u)
        << "a [norm_size, 1]-shaped operand (same numel as norm_size, but "
           "wrong axis) must NOT be absorbed as gamma -- match should stop "
           "at Div (5 nodes), leaving the Mul unfused for the correct eager "
           "path";
    for (const auto& n : matches[0].nodes) {
        EXPECT_NE(n->op_type(), OpType::Mul)
            << "the affine Mul must not be part of the fused match";
    }
}

// A scalar (numel==1) operand is legitimate (mirrors how eps is a scalar);
// must still be absorbed, not rejected as "not per-channel".
TEST(PatternMatcherAffineShapeGuard, RmsNormAbsorbsScalarGamma) {
    Graph g;
    build_decomposed_rms_norm(g, /*affine_shape=*/{1});  // scalar-like

    PatternMatcher matcher;
    auto matches = matcher.find_all(g);
    ASSERT_EQ(matches.size(), 1u);
    EXPECT_EQ(matches[0].nodes.size(), 6u)
        << "a scalar affine operand (numel==1) must still be absorbed";
}
