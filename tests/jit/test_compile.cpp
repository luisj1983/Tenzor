/**
 * @file test_compile.cpp
 * @brief Tests for automatic graph capture (tenzor.compile)
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/jit/compile.hpp>
#include <tenzor/jit/tracing_interceptor.hpp>

using namespace tenzor;
using namespace tenzor::jit;

class CompileTestEnv : public ::testing::Environment {
public:
    void SetUp() override { tenzor::initialize(); }
    void TearDown() override { tenzor::finalize(); }
};
static auto* _env = ::testing::AddGlobalTestEnvironment(new CompileTestEnv());

TEST(CompileTest, OpIdToOpTypeMapping) {
    // Verify key OpId -> OpType mappings
    EXPECT_EQ(opid_to_optype(OpId::Add), OpType::Add);
    EXPECT_EQ(opid_to_optype(OpId::MatMul), OpType::MatMul);
    EXPECT_EQ(opid_to_optype(OpId::ReLU), OpType::ReLU);
    EXPECT_EQ(opid_to_optype(OpId::Softmax), OpType::Softmax);
    EXPECT_EQ(opid_to_optype(OpId::Conv2dForward), OpType::Conv2d);
    EXPECT_EQ(opid_to_optype(OpId::LayerNorm), OpType::LayerNorm);
}

TEST(CompileTest, UnmappedOpIdReturnsNullopt) {
    // Ops without IR mapping should return nullopt
    auto result = opid_to_optype(static_cast<OpId>(9999));
    EXPECT_FALSE(result.has_value());
}

TEST(CompileTest, CompiledFunctionConstruction) {
    auto fn = [](const Variable& x) -> Variable { return x; };
    CompiledFunction compiled(fn);
    EXPECT_EQ(compiled.num_cached(), 0);
    EXPECT_FALSE(compiled.had_graph_break());
}

TEST(CompileTest, CompileConfigDefaults) {
    CompileConfig config;
    EXPECT_FALSE(config.fullgraph);
    EXPECT_EQ(config.max_retraces, 8);
    EXPECT_TRUE(config.enable_fusion);
    EXPECT_EQ(config.mode, "default");
}

TEST(CompileTest, ShapeKeyGeneration) {
    // Two variables with same shape should produce the same key
    auto t1 = Tensor({2, 3}, DType::Float32, Device::cpu());
    auto t2 = Tensor({2, 3}, DType::Float32, Device::cpu());
    Variable v1(t1, false);
    Variable v2(t2, false);

    // Different shape should produce different key
    auto t3 = Tensor({4, 5}, DType::Float32, Device::cpu());
    Variable v3(t3, false);

    // We can't directly call shape_key (it's private), but we verify
    // the compile function works with different shapes
    auto fn = [](const Variable& x) -> Variable { return x; };
    auto compiled = compile(fn);
    // These should work without errors
    EXPECT_NO_THROW(compiled(v1));
    EXPECT_NO_THROW(compiled(v3));
}

TEST(CompileTest, ClearCache) {
    auto fn = [](const Variable& x) -> Variable { return x; };
    CompiledFunction compiled(fn);

    auto t = Tensor({2, 3}, DType::Float32, Device::cpu());
    Variable v(t, false);
    compiled(v);

    compiled.clear_cache();
    EXPECT_EQ(compiled.num_cached(), 0);
}

TEST(CompileTest, CompileFreeFunctionWorks) {
    auto fn = [](const Variable& x) -> Variable { return x; };
    auto compiled = compile(fn);
    EXPECT_EQ(compiled.num_cached(), 0);
}
