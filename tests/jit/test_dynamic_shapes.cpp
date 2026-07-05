/**
 * @file test_dynamic_shapes.cpp
 * @brief Tests for JIT dynamic shape support via SymbolicShapeEnvironment
 */

#include <gtest/gtest.h>
#include <memory>
#include <tenzor/jit/symbolic_shape.hpp>
#include <tenzor/jit/graph.hpp>
#include <tenzor/jit/tracer.hpp>
#include <tenzor/jit/symbolic_shape_inference.hpp>
#include <tenzor/core/dtype.hpp>
#include <tenzor/core/device.hpp>

using namespace tenzor::jit;

namespace {
std::shared_ptr<Value> mk_val(std::string id, std::vector<int64_t> shape) {
    return std::make_shared<Value>(std::move(id), std::move(shape),
                                   tenzor::DType::Float32, tenzor::Device::cpu());
}
}  // namespace

// JIT-F021: a permutation whose length differs from the input rank must yield
// no confident inference (empty), not a wrong-rank shape.
TEST(SymbolicShapeInferenceFixes, PermuteWrongLengthReturnsEmpty) {
    auto x = mk_val("x", {2, 3, 4});
    auto node = std::make_shared<Node>(OpType::Permute);
    node->add_input(x);
    node->add_output(mk_val("y", {}));
    node->set_vec_attr("dims", {0, 2});  // length 2 != rank 3
    SymbolicShapeInference infer;
    EXPECT_TRUE(infer.infer(node.get()).empty());
    node->set_vec_attr("dims", {0, 2, 1});  // valid full-rank permutation
    auto out = infer.infer(node.get());
    ASSERT_EQ(out.size(), 1u);
    EXPECT_EQ(out[0].rank(), 3u);
}

// JIT-F020: a dimensioned Max produces values + indices; both outputs must
// receive the reduced symbolic shape.
TEST(SymbolicShapeInferenceFixes, MaxWithDimEmitsTwoOutputShapes) {
    auto x = mk_val("x", {2, 3, 4});
    auto node = std::make_shared<Node>(OpType::Max);
    node->add_input(x);
    node->add_output(mk_val("vals", {}));
    node->add_output(mk_val("idx", {}));
    node->set_int_attr("dim", 1);
    node->set_bool_attr("keepdim", false);
    SymbolicShapeInference infer;
    auto out = infer.infer(node.get());
    ASSERT_EQ(out.size(), 2u);
    EXPECT_EQ(out[0].rank(), 2u);
    EXPECT_EQ(out[1].rank(), 2u);
}

// JIT-F022: a genuinely fixed reshape (no -1, trailing product differs from the
// input's) on a dynamic input must resolve to the concrete target, not a
// symbolized leading dim.
TEST(SymbolicShapeInferenceFixes, FixedReshapeOnDynamicInputStaysConcrete) {
    auto x = mk_val("x", {0, 10});
    x->set_symbolic_shape(SymbolicShape(
        {SymbolicDim::symbolic("B"), SymbolicDim::concrete(10)}));
    auto node = std::make_shared<Node>(OpType::Reshape);
    node->add_input(x);
    node->add_output(mk_val("y", {}));
    node->set_vec_attr("shape", {5, 4});  // fixed; trailing 4 != input trailing 10
    SymbolicShapeInference infer;
    auto out = infer.infer(node.get());
    ASSERT_EQ(out.size(), 1u);
    ASSERT_EQ(out[0].rank(), 2u);
    ASSERT_TRUE(out[0][0].is_concrete());
    EXPECT_EQ(out[0][0].value(), 5);
    ASSERT_TRUE(out[0][1].is_concrete());
    EXPECT_EQ(out[0][1].value(), 4);
}

// Companion: a batch-preserving reshape (trailing product matches) still
// symbolizes the leading dim.
TEST(SymbolicShapeInferenceFixes, BatchReshapeOnDynamicInputSymbolizesLeading) {
    auto x = mk_val("x", {0, 2, 5});
    x->set_symbolic_shape(SymbolicShape({SymbolicDim::symbolic("B"),
                                         SymbolicDim::concrete(2),
                                         SymbolicDim::concrete(5)}));
    auto node = std::make_shared<Node>(OpType::Reshape);
    node->add_input(x);
    node->add_output(mk_val("y", {}));
    node->set_vec_attr("shape", {0, 10});  // placeholder leading, trailing 10 == 2*5
    // Use a concrete leading in the target to mimic x.reshape(x.size(0), 10):
    node->set_vec_attr("shape", {32, 10});
    SymbolicShapeInference infer;
    auto out = infer.infer(node.get());
    ASSERT_EQ(out.size(), 1u);
    ASSERT_EQ(out[0].rank(), 2u);
    EXPECT_TRUE(out[0][0].is_symbolic());   // leading dim tracks batch
    ASSERT_TRUE(out[0][1].is_concrete());
    EXPECT_EQ(out[0][1].value(), 10);
}

class DynamicShapesTest : public ::testing::Test {};

TEST_F(DynamicShapesTest, SymbolicDimConcrete) {
    auto dim = SymbolicDim::concrete(64);
    EXPECT_TRUE(dim.is_concrete());
    EXPECT_EQ(dim.value(), 64);
}

TEST_F(DynamicShapesTest, SymbolicDimSymbolic) {
    auto dim = SymbolicDim::symbolic("batch");
    EXPECT_FALSE(dim.is_concrete());
    EXPECT_EQ(dim.name(), "batch");
}

TEST_F(DynamicShapesTest, EnvironmentBindAndResolve) {
    SymbolicShapeEnvironment env;
    env.bind("batch", 32);
    env.bind("seq_len", 128);

    EXPECT_TRUE(env.is_bound("batch"));
    EXPECT_TRUE(env.is_bound("seq_len"));
    EXPECT_FALSE(env.is_bound("hidden"));

    EXPECT_EQ(env.get("batch"), 32);
    EXPECT_EQ(env.get("seq_len"), 128);
}

TEST_F(DynamicShapesTest, EnvironmentResolveConcretePassthrough) {
    SymbolicShapeEnvironment env;
    auto dim = SymbolicDim::concrete(64);
    EXPECT_EQ(env.resolve(dim), 64);
}

TEST_F(DynamicShapesTest, EnvironmentResolveSymbolic) {
    SymbolicShapeEnvironment env;
    env.bind("batch", 32);

    auto dim = SymbolicDim::symbolic("batch");
    EXPECT_EQ(env.resolve(dim), 32);
}

TEST_F(DynamicShapesTest, EnvironmentResolveUnboundThrows) {
    SymbolicShapeEnvironment env;
    auto dim = SymbolicDim::symbolic("unbound");
    EXPECT_THROW(env.resolve(dim), std::runtime_error);
}

TEST_F(DynamicShapesTest, EnvironmentResolveShape) {
    SymbolicShapeEnvironment env;
    env.bind("batch", 16);

    SymbolicShape shape = {
        SymbolicDim::symbolic("batch"),
        SymbolicDim::concrete(3),
        SymbolicDim::concrete(224),
        SymbolicDim::concrete(224)
    };

    auto concrete = env.resolve(shape);
    EXPECT_EQ(concrete.size(), 4);
    EXPECT_EQ(concrete[0], 16);
    EXPECT_EQ(concrete[1], 3);
    EXPECT_EQ(concrete[2], 224);
    EXPECT_EQ(concrete[3], 224);
}

TEST_F(DynamicShapesTest, EnvironmentCanResolve) {
    SymbolicShapeEnvironment env;
    env.bind("batch", 32);

    SymbolicShape resolvable = {
        SymbolicDim::symbolic("batch"),
        SymbolicDim::concrete(64)
    };
    EXPECT_TRUE(env.can_resolve(resolvable));

    SymbolicShape unresolvable = {
        SymbolicDim::symbolic("batch"),
        SymbolicDim::symbolic("missing")
    };
    EXPECT_FALSE(env.can_resolve(unresolvable));
}

TEST_F(DynamicShapesTest, EnvironmentUnbind) {
    SymbolicShapeEnvironment env;
    env.bind("batch", 32);
    EXPECT_TRUE(env.is_bound("batch"));

    env.unbind("batch");
    EXPECT_FALSE(env.is_bound("batch"));
}

TEST_F(DynamicShapesTest, EnvironmentClear) {
    SymbolicShapeEnvironment env;
    env.bind("a", 1);
    env.bind("b", 2);
    EXPECT_EQ(env.size(), 2);

    env.clear();
    EXPECT_EQ(env.size(), 0);
}

TEST_F(DynamicShapesTest, RebindOverwrites) {
    SymbolicShapeEnvironment env;
    env.bind("batch", 32);
    EXPECT_EQ(env.get("batch"), 32);

    env.bind("batch", 64);
    EXPECT_EQ(env.get("batch"), 64);
}

// ============================================================================
// SymbolicExpr AST tests
// ============================================================================

TEST_F(DynamicShapesTest, ExprCreation) {
    auto batch = SymbolicDim::symbolic("batch");
    auto result = batch + SymbolicDim::concrete(32);
    EXPECT_FALSE(result.is_concrete());
    EXPECT_TRUE(result.is_symbolic());
    EXPECT_TRUE(result.is_expr());
    EXPECT_FALSE(result.is_named_symbol());
}

TEST_F(DynamicShapesTest, ExprToString) {
    auto batch = SymbolicDim::symbolic("batch");
    auto result = batch + SymbolicDim::concrete(32);
    EXPECT_EQ(result.to_string(), "(batch + 32)");

    auto nested = (batch + SymbolicDim::concrete(4)) - SymbolicDim::concrete(3);
    EXPECT_EQ(nested.to_string(), "((batch + 4) - 3)");
}

TEST_F(DynamicShapesTest, ExprEquality) {
    auto batch = SymbolicDim::symbolic("batch");
    auto a = batch + SymbolicDim::concrete(32);
    auto b = batch + SymbolicDim::concrete(32);
    auto c = batch + SymbolicDim::concrete(64);
    EXPECT_EQ(a, b);
    EXPECT_NE(a, c);
}

TEST_F(DynamicShapesTest, ConstantFolding) {
    auto a = SymbolicDim::concrete(3);
    auto b = SymbolicDim::concrete(4);
    auto result = a + b;
    EXPECT_TRUE(result.is_concrete());
    EXPECT_EQ(result.value(), 7);

    auto product = a * b;
    EXPECT_TRUE(product.is_concrete());
    EXPECT_EQ(product.value(), 12);
}

TEST_F(DynamicShapesTest, IdentityAdd) {
    auto batch = SymbolicDim::symbolic("batch");
    auto result = batch + SymbolicDim::concrete(0);
    // Should simplify to just 'batch', not an expression
    EXPECT_TRUE(result.is_named_symbol());
    EXPECT_EQ(result.name(), "batch");

    auto result2 = SymbolicDim::concrete(0) + batch;
    EXPECT_TRUE(result2.is_named_symbol());
    EXPECT_EQ(result2.name(), "batch");
}

TEST_F(DynamicShapesTest, IdentityMul) {
    auto batch = SymbolicDim::symbolic("batch");
    auto result = batch * SymbolicDim::concrete(1);
    EXPECT_TRUE(result.is_named_symbol());
    EXPECT_EQ(result.name(), "batch");

    auto result2 = SymbolicDim::concrete(1) * batch;
    EXPECT_TRUE(result2.is_named_symbol());
}

TEST_F(DynamicShapesTest, ZeroMul) {
    auto batch = SymbolicDim::symbolic("batch");
    auto result = batch * SymbolicDim::concrete(0);
    EXPECT_TRUE(result.is_concrete());
    EXPECT_EQ(result.value(), 0);
}

TEST_F(DynamicShapesTest, ExprResolve) {
    SymbolicShapeEnvironment env;
    env.bind("batch", 32);

    auto batch = SymbolicDim::symbolic("batch");
    auto expr = batch + SymbolicDim::concrete(10);
    EXPECT_EQ(env.resolve(expr), 42);
}

TEST_F(DynamicShapesTest, NestedExprResolve) {
    SymbolicShapeEnvironment env;
    env.bind("H", 224);

    // Simulate conv2d output: (H + 2*pad - kernel) / stride + 1
    // With pad=1, kernel=3, stride=1: (224 + 2 - 3) / 1 + 1 = 224
    auto H = SymbolicDim::symbolic("H");
    auto pad2 = SymbolicDim::concrete(2);
    auto kernel = SymbolicDim::concrete(3);
    auto stride = SymbolicDim::concrete(1);
    auto one = SymbolicDim::concrete(1);

    auto H_out = (H + pad2 - kernel) / stride + one;
    EXPECT_TRUE(H_out.is_expr());
    EXPECT_EQ(env.resolve(H_out), 224);

    // Different H value
    env.bind("H", 112);
    EXPECT_EQ(env.resolve(H_out), 112);
}

TEST_F(DynamicShapesTest, CanResolveExpr) {
    SymbolicShapeEnvironment env;
    env.bind("batch", 32);

    auto batch = SymbolicDim::symbolic("batch");
    auto expr = batch + SymbolicDim::concrete(10);
    EXPECT_TRUE(env.can_resolve_dim(expr));

    // Expression with unbound symbol
    auto seq = SymbolicDim::symbolic("seq_len");
    auto expr2 = batch * seq;
    EXPECT_FALSE(env.can_resolve_dim(expr2));

    // Bind the missing symbol
    env.bind("seq_len", 128);
    EXPECT_TRUE(env.can_resolve_dim(expr2));
    EXPECT_EQ(env.resolve(expr2), 32 * 128);
}

TEST_F(DynamicShapesTest, ExprNameThrows) {
    auto batch = SymbolicDim::symbolic("batch");
    auto expr = batch + SymbolicDim::concrete(32);
    EXPECT_THROW((void)expr.name(), std::runtime_error);
}

TEST_F(DynamicShapesTest, BackwardCompatStringConstruction) {
    // Old-style construction should still work identically
    SymbolicDim dim(std::string("batch"));
    EXPECT_TRUE(dim.is_symbolic());
    EXPECT_TRUE(dim.is_named_symbol());
    EXPECT_FALSE(dim.is_expr());
    EXPECT_EQ(dim.name(), "batch");
    EXPECT_EQ(dim.to_string(), "batch");
}
