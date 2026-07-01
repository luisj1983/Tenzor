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
