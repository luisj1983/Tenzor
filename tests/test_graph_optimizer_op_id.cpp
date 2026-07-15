/**
 * @file test_graph_optimizer_op_id.cpp
 * @brief Audit A.2: `Function::op_id()` virtual returns the canonical
 *        forward OpId for opted-in subclasses. Pattern matchers can now
 *        identify Functions via the OpId enum rather than RTTI string
 *        substring — see `GraphOptimizer::is_operation_type(node, OpId)`
 *        (private member, exercised indirectly through the public fuse_*
 *        passes; the public-facing API regression is in
 *        `test_function_op_id.cpp`).
 *
 * This test pins the contract that Function::op_id() returns the
 * expected canonical enum value for every Backward class the graph
 * optimiser's pattern matchers care about (the inputs to its fuse_*
 * passes).
 */

#include <gtest/gtest.h>

#include "tenzor/autograd/function.hpp"
#include "tenzor/autograd/graph_optimizer.hpp"
#include "tenzor/ops/op_id.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/tenzor.hpp"

using namespace tenzor;

namespace {

class GraphOptimizerOpIdTest : public ::testing::Test {
protected:
    void SetUp() override { tenzor::initialize(); }
};

// The optimizer must ACTUALLY rewrite the live grad_fn chain (not just detect),
// and the rewrite must preserve gradients exactly. An inverse transpose pair is
// the identity, so eliminating it must leave x.grad byte-identical.
TEST_F(GraphOptimizerOpIdTest, EliminatesInverseTransposePairPreservingGradients) {
    auto run = [](bool optimize, size_t* eliminated) -> Tensor {
        auto x = Variable(ones({3, 4}, DType::Float32, Device::cpu()), true);
        auto c2 = Variable(full({3, 4}, 2.0f, DType::Float32, Device::cpu()), false);
        auto h = x * c2;                    // non-leaf intermediate (so the pair's
                                            // input is not a leaf -> safe to elide)
        auto t1 = h.transpose(0, 1);        // {4,3}
        auto t2 = t1.transpose(0, 1);       // {3,4} == h  (identity pair)
        auto c = Variable(full({3, 4}, 3.0f, DType::Float32, Device::cpu()), false);
        auto z = t2 * c;                    // MulBackward is the single consumer of t2's TransposeBackward
        auto loss = sum(z);
        if (optimize) {
            GraphOptimizer opt;
            auto stats = opt.optimize_variable(loss);
            if (eliminated) *eliminated = stats.transpose_pairs_eliminated;
        }
        loss.backward();
        return x.grad().value().to(Device::cpu()).contiguous();
    };

    Tensor g_baseline = run(/*optimize=*/false, nullptr);
    size_t eliminated = 0;
    Tensor g_optimized = run(/*optimize=*/true, &eliminated);

    // The optimizer genuinely rewrote the executable chain.
    EXPECT_GE(eliminated, 1u);

    // ...and gradients are byte-identical (d/dx of sum(3*x) is 3 everywhere).
    const float* a = g_baseline.data<float>();
    const float* b = g_optimized.data<float>();
    for (int i = 0; i < 12; ++i) {
        EXPECT_FLOAT_EQ(a[i], 6.0f);
        EXPECT_FLOAT_EQ(b[i], a[i]);
    }
}

// M1: the transpose-pair splice used to compute "single consumer" via a walk
// starting ONLY at the optimized root, so it couldn't see a sibling Variable
// sharing the same transpose-pair sub-expression through retain_graph=true —
// splicing one root's graph silently corrupted the other root's graph too
// (both reference the SAME shared_ptr<Function> nodes). This constructs
// exactly that: y = transpose(transpose(h)) feeds TWO independent consumers
// (a and b); optimizing only a's graph must not touch the pair T1/T2, since
// T1 has TWO parents globally (a's MulBackward and b's MulBackward) even
// though only one of them (a's) is reachable by walking from loss_a alone.
//
// Verified via revert: with the old walk-local "consumers" count (which
// can't see b's edge), transpose_pairs_eliminated == 1 for this exact
// construction — the splice DOES fire. The fix's parent_count() check
// (globally accurate, maintained by Function::set_next_functions()) brings
// that to 0. Note the splice itself mutates only the SPLICING PARENT's own
// next_functions() storage (a's MulBackward), never T1/T2's — b's MulBackward
// keeps its own independent edge to T1 regardless, so this construction does
// not additionally demonstrate corrupted VALUES on `b`'s side; the direct,
// unambiguous signal is transpose_pairs_eliminated itself, asserted below.
// Both a's and b's gradients are still checked as a value-level sanity check.
TEST_F(GraphOptimizerOpIdTest, DoesNotSpliceTransposePairSharedByAnotherVariable) {
    auto x = Variable(ones({3, 4}, DType::Float32, Device::cpu()), true);
    auto c0 = Variable(full({3, 4}, 1.0f, DType::Float32, Device::cpu()), false);
    auto h = x * c0;  // non-leaf intermediate (matches
                      // EliminatesInverseTransposePairPreservingGradients
                      // above — the pair's input must not be a leaf, or
                      // accumulates_locally() blocks the splice entirely
                      // regardless of the sharing concern this test targets).
    auto y = h.transpose(0, 1).transpose(0, 1);  // identity pair, shape {3,4}

    auto c1 = Variable(full({3, 4}, 2.0f, DType::Float32, Device::cpu()), false);
    auto c2 = Variable(full({3, 4}, 5.0f, DType::Float32, Device::cpu()), false);
    auto a = y * c1;  // consumer #1 of y's transpose-pair chain
    auto b = y * c2;  // consumer #2 of the SAME transpose-pair chain

    auto loss_a = sum(a);
    auto loss_b = sum(b);

    GraphOptimizer opt;
    auto opt_stats = opt.optimize_variable(loss_a);
    EXPECT_EQ(opt_stats.transpose_pairs_eliminated, 0u)
        << "optimizer spliced a transpose pair that b's independent graph "
           "also references — parent_count() should have blocked this";

    loss_b.backward(std::nullopt, /*retain_graph=*/true);
    ASSERT_TRUE(x.has_grad());
    auto xg = x.grad().value().to(Device::cpu()).contiguous();
    for (int i = 0; i < 12; ++i) {
        EXPECT_FLOAT_EQ(xg.data<float>()[i], 5.0f) << "b's gradient wrong at i=" << i;
    }

    x.zero_grad();
    loss_a.backward();
    auto xg2 = x.grad().value().to(Device::cpu()).contiguous();
    for (int i = 0; i < 12; ++i) {
        EXPECT_FLOAT_EQ(xg2.data<float>()[i], 2.0f) << "a's gradient wrong at i=" << i;
    }
}

TEST_F(GraphOptimizerOpIdTest, FusePatternInputsReportTheirOpId) {
    // The classes consumed by graph_optimizer fuse_* passes.
    EXPECT_EQ(AddBackward{}.op_id(), OpId::Add);
    EXPECT_EQ(MatMulBackward{}.op_id(), OpId::MatMul);
    EXPECT_EQ(LinearBackward{}.op_id(), OpId::Linear);
    EXPECT_EQ(GeluBackward{}.op_id(), OpId::Gelu);
    EXPECT_EQ(TransposeBackward(0, 1).op_id(), OpId::Transpose);
    EXPECT_EQ(ReshapeBackward({1}).op_id(), OpId::Reshape);
}

TEST_F(GraphOptimizerOpIdTest, OpIdComparisonStrongerThanRTTISubstring) {
    // The RTTI-substring matcher's classic failure mode is the
    // "MatMulBackward" substring accidentally matching e.g.
    // "FlashAttentionMatMulBackward". OpId comparisons are immune to that:
    // each Function returns its own enum value.
    EXPECT_NE(AddBackward{}.op_id(), MatMulBackward{}.op_id());
    EXPECT_NE(AddBackward{}.op_id(), SubBackward{}.op_id());
    EXPECT_NE(MulBackward{}.op_id(), DivBackward{}.op_id());
}

}  // namespace
