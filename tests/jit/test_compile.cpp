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

// JIT-F027: @tz.jit / tz.compile must install the in-place op hook so an
// in-place mutation inside the compiled function is recorded into the graph, not
// silently dropped. Pre-fix, the compiled replay returned the PRE-mutation value.
TEST(CompileTest, InplaceMutationCapturedViaCompile) {
    Tensor one = ones({4}, DType::Float32, Device::cpu());
    auto fn = [one](const Variable& x) -> Variable {
        Tensor y = x.tensor() + one;   // y = x + 1
        tenzor::add_(y, one);          // in-place: y = x + 2
        Tensor z = y + one;            // reads the MUTATED y -> x + 3
        return Variable(z, false);
    };
    auto compiled = compile(fn);
    Tensor x = full({4}, 2.0f, DType::Float32, Device::cpu());
    // First call traces (and runs fn eagerly); subsequent calls replay the
    // compiled graph, which must reproduce x + 3 == 5 (not the dropped-mutation
    // value x + 2 == 4).
    (void)compiled(Variable(x, false));
    Variable out = compiled(Variable(x, false));
    Tensor replay = out.tensor().to(Device::cpu());
    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(replay.data<float>()[i], 5.0f, 1e-5f)
            << "in-place add_ was dropped from the compiled graph (got the "
               "pre-mutation value)";
    }
}

// JIT-F033: a compiled function must EXECUTE correctly at multiple concrete
// shapes (recompiling as needed), not just have its symbolic-dim algebra
// asserted. A reduction makes the output shape shape-dependent, so replaying a
// graph baked at the wrong shape would be caught.
TEST(CompileTest, ExecutesCorrectlyAcrossBatchSizes) {
    auto fn = [](const Variable& x) -> Variable {
        return tenzor::sum(x, /*dim=*/1);  // reduce dim 1 -> shape [B]
    };
    auto compiled = compile(fn);
    for (int64_t B : {2, 5, 8}) {
        Tensor xt = full({B, 4}, 1.5f, DType::Float32, Device::cpu());
        Variable x(xt, false);
        Tensor eager = fn(x).tensor().to(Device::cpu());
        (void)compiled(x);                                   // trace/compile shape
        Tensor jit = compiled(x).tensor().to(Device::cpu());  // replay
        ASSERT_EQ(jit.numel(), B) << "B=" << B;
        for (int64_t i = 0; i < B; ++i) {
            EXPECT_NEAR(jit.data<float>()[i], eager.data<float>()[i], 1e-4f)
                << "B=" << B << " i=" << i;  // each = 4 * 1.5 = 6.0
        }
    }
}

// JIT-F054: fullgraph=True must error on a graph break (here a data-dependent
// .item() read), not silently fall back to eager.
TEST(CompileTest, FullgraphThrowsOnItemGraphBreak) {
    CompileConfig cfg;
    cfg.fullgraph = true;
    auto fn = [](const Variable& x) -> Variable {
        // .item() is a data-dependent read -> graph break.
        float s = tenzor::sum(x).tensor().to(Device::cpu()).item<float>();
        return x * s;
    };
    auto compiled = CompiledFunction(CompiledFunction::FnType(fn), cfg);
    Tensor xt = full({4}, 2.0f, DType::Float32, Device::cpu());
    EXPECT_THROW(compiled(Variable(xt, false)), std::exception);
}
