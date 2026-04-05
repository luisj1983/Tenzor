/**
 * @file test_dynamic_shapes.cpp
 * @brief Tests for JIT dynamic shape support via SymbolicShapeEnvironment
 */

#include <gtest/gtest.h>
#include <tenzor/jit/symbolic_shape.hpp>

using namespace tenzor::jit;

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
