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
#include "tenzor/ops/op_id.hpp"
#include "tenzor/tenzor.hpp"

using namespace tenzor;

namespace {

class GraphOptimizerOpIdTest : public ::testing::Test {
protected:
    void SetUp() override { tenzor::initialize(); }
};

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
