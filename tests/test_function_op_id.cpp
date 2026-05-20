/**
 * @file test_function_op_id.cpp
 * @brief Audit A.2: `Function::op_id()` virtual returns the forward
 *        OpId for opted-in subclasses, `OpId::Unknown` otherwise.
 *
 * The graph optimiser fusion-pattern matchers and vmap rule registry
 * use this virtual to identify a Function by its forward op instead of
 * RTTI/string substring matching.  This regression pins the contract:
 *
 *   - Opted-in subclasses (Add/Sub/Mul/Div/MatMul, Histc/Bincount/
 *     SearchSorted) return their canonical OpId.
 *   - A Function subclass that doesn't override returns `Unknown` so
 *     pattern matchers treat it as "do not match" rather than
 *     mis-matching against an unrelated default.
 */

#include <gtest/gtest.h>

#include "tenzor/autograd/function.hpp"
#include "tenzor/ops/op_id.hpp"
#include "tenzor/tenzor.hpp"

using namespace tenzor;

namespace {

class FunctionOpIdTest : public ::testing::Test {
protected:
    void SetUp() override { tenzor::initialize(); }
};

TEST_F(FunctionOpIdTest, ArithmeticBackwardsReturnForwardOpId) {
    EXPECT_EQ(AddBackward{}.op_id(), OpId::Add);
    EXPECT_EQ(SubBackward{}.op_id(), OpId::Sub);
    EXPECT_EQ(MulBackward{}.op_id(), OpId::Mul);
    EXPECT_EQ(DivBackward{}.op_id(), OpId::Div);
    EXPECT_EQ(MatMulBackward{}.op_id(), OpId::MatMul);
}

TEST_F(FunctionOpIdTest, NonDifferentiableBackwardsReturnForwardOpId) {
    EXPECT_EQ(HistcBackward{}.op_id(), OpId::Histc);
    EXPECT_EQ(BincountBackward{}.op_id(), OpId::Bincount);
    EXPECT_EQ(SearchSortedBackward{}.op_id(), OpId::SearchSorted);
}

namespace {
// Test scaffold: a Function subclass that does NOT override op_id().
class UnoptInFunction : public Function {
public:
    auto forward(std::vector<Variable>) -> std::vector<Variable> override {
        return {};
    }
    auto backward(std::vector<Tensor>) -> std::vector<Tensor> override {
        return {};
    }
    auto name() const -> std::string override { return "UnoptInFunction"; }
};
}  // namespace

TEST_F(FunctionOpIdTest, UnoptedInSubclassReturnsUnknown) {
    UnoptInFunction f;
    EXPECT_EQ(f.op_id(), OpId::Unknown);
    // OpId::Unknown is strictly less than OP_COUNT so pattern matchers
    // can range-check, and OpId::Unknown != any real OpId.
    EXPECT_NE(f.op_id(), OpId::Add);
    EXPECT_NE(f.op_id(), OpId::MatMul);
}

}  // namespace
