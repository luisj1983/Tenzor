/**
 * @file test_codegen.cpp
 * @brief Tests for runtime GPU kernel generation (codegen)
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/jit/codegen.hpp>

using namespace tenzor;
using namespace tenzor::jit;

class CodegenTestEnv : public ::testing::Environment {
public:
    void SetUp() override { tenzor::initialize(); }
};
static ::testing::Environment* const env =
    ::testing::AddGlobalTestEnvironment(new CodegenTestEnv);

// =========================================================================
// FusionGroup construction tests
// =========================================================================

TEST(Codegen, BuildFusionGroup) {
    auto group = build_fusion({
        {ElemOp::Relu, 0, -1, 0.0},
        {ElemOp::MulScalar, -1, -1, 2.0},
    }, 1, DType::Float32);

    EXPECT_EQ(group.num_inputs, 1);
    EXPECT_EQ(group.steps.size(), 2u);
    EXPECT_FALSE(group.signature.empty());
}

TEST(Codegen, SignatureUniqueness) {
    auto g1 = build_fusion({
        {ElemOp::Relu, 0, -1, 0.0},
        {ElemOp::Sigmoid, -1, -1, 0.0},
    }, 1, DType::Float32);

    auto g2 = build_fusion({
        {ElemOp::Sigmoid, 0, -1, 0.0},
        {ElemOp::Relu, -1, -1, 0.0},
    }, 1, DType::Float32);

    EXPECT_NE(g1.signature, g2.signature);
}

// =========================================================================
// Code generation tests
// =========================================================================

TEST(Codegen, GenerateSimpleKernel) {
    auto group = build_fusion({
        {ElemOp::Relu, 0, -1, 0.0},
    }, 1, DType::Float32);

    auto source = KernelCodegen::generate(group);

    // Should contain key elements
    EXPECT_NE(source.find("__global__"), std::string::npos);
    // relu is emitted as a NaN-propagating `x < 0 ? 0 : x` select. It used to
    // use fmax, but fmax(NaN, 0) returns 0 and diverged from the CPU clamp_min
    // fallback (which propagates NaN); the select matches CPU/eager semantics.
    EXPECT_NE(source.find("? 0.0f :"), std::string::npos);
    EXPECT_NE(source.find("blockIdx"), std::string::npos);
    EXPECT_NE(source.find("threadIdx"), std::string::npos);
}

TEST(Codegen, GenerateFusedKernel) {
    auto group = build_fusion({
        {ElemOp::Mul, 0, 1, 0.0},        // tmp = a * b
        {ElemOp::AddScalar, -1, -1, 1.0}, // tmp = tmp + 1
        {ElemOp::Sigmoid, -1, -1, 0.0},   // out = sigmoid(tmp)
    }, 2, DType::Float32);

    auto source = KernelCodegen::generate(group);

    // Should have 2 input pointers
    EXPECT_NE(source.find("inp0"), std::string::npos);
    EXPECT_NE(source.find("inp1"), std::string::npos);
    // Should contain sigmoid formula
    EXPECT_NE(source.find("exp"), std::string::npos);
}

TEST(Codegen, GenerateAllActivations) {
    // Test that all activation codegen compiles to valid source
    for (auto op : {ElemOp::Relu, ElemOp::LeakyRelu, ElemOp::Elu,
                    ElemOp::Selu, ElemOp::Gelu, ElemOp::Mish,
                    ElemOp::Softplus, ElemOp::Sigmoid, ElemOp::Tanh}) {
        auto group = build_fusion({{op, 0, -1, 0.1}}, 1, DType::Float32);
        auto source = KernelCodegen::generate(group);
        EXPECT_NE(source.find("__global__"), std::string::npos)
            << "Failed for op " << static_cast<int>(op);
    }
}

// =========================================================================
// Execution tests (CPU fallback or GPU)
// =========================================================================

TEST(Codegen, ExecuteFused_ReluSigmoid) {
    auto group = build_fusion({
        {ElemOp::Relu, 0, -1, 0.0},
        {ElemOp::Sigmoid, -1, -1, 0.0},
    }, 1, DType::Float32);

    // Create input with mix of positive and negative values
    auto input = tenzor::randn({8}, DType::Float32, Device::cpu());
    auto result = execute_fused(group, {input});

    // Verify: sigmoid(relu(x)) == sigmoid(max(0,x))
    auto expected = tenzor::sigmoid(tenzor::clamp_min(input, 0.0f));
    auto r = result.to(Device::cpu()).contiguous();
    auto e = expected.to(Device::cpu()).contiguous();
    auto* rd = r.data<float>();
    auto* ed = e.data<float>();
    for (int i = 0; i < 8; ++i) {
        EXPECT_NEAR(rd[i], ed[i], 1e-5) << "Mismatch at " << i;
    }
}

TEST(Codegen, ExecuteFused_MulAdd) {
    auto group = build_fusion({
        {ElemOp::Mul, 0, 1, 0.0},         // a * b
        {ElemOp::AddScalar, -1, -1, 3.0},  // + 3.0
    }, 2, DType::Float32);

    auto a = tenzor::ones({4}, DType::Float32, Device::cpu());
    auto b = tenzor::mul(tenzor::ones({4}, DType::Float32, Device::cpu()), 2.0);
    auto result = execute_fused(group, {a, b});

    // Expected: 1*2 + 3 = 5
    auto r = result.to(Device::cpu()).contiguous();
    auto* rd = r.data<float>();
    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(rd[i], 5.0f, 1e-5);
    }
}

TEST(Codegen, ExecuteFused_ChainedMath) {
    auto group = build_fusion({
        {ElemOp::Exp, 0, -1, 0.0},        // exp(x)
        {ElemOp::Log, -1, -1, 0.0},       // log(exp(x)) = x
    }, 1, DType::Float32);

    auto input = tenzor::mul(tenzor::randn({16}, DType::Float32, Device::cpu()), 0.5f);
    auto result = execute_fused(group, {input});

    // exp(log(x)) ≈ x
    auto r = result.to(Device::cpu()).contiguous();
    auto inp = input.to(Device::cpu()).contiguous();
    auto* rd = r.data<float>();
    auto* id = inp.data<float>();
    for (int i = 0; i < 16; ++i) {
        EXPECT_NEAR(rd[i], id[i], 1e-4) << "Round-trip failed at " << i;
    }
}

// =========================================================================
// Cache tests
// =========================================================================

TEST(Codegen, KernelCacheHit) {
    auto& cache = KernelCache::instance();
    size_t compilations_before = cache.num_compilations();

    auto group = build_fusion({
        {ElemOp::Neg, 0, -1, 0.0},
        {ElemOp::Abs, -1, -1, 0.0},
    }, 1, DType::Float32);

    // First call compiles
    auto input = tenzor::ones({4}, DType::Float32, Device::cpu());
    execute_fused(group, {input});

    // Second call should hit cache (no new compilation)
    size_t compilations_after_first = cache.num_compilations();
    execute_fused(group, {input});
    size_t compilations_after_second = cache.num_compilations();

    // On CPU fallback, no compilations happen, but cache is still tested
    // On GPU, second call should NOT compile again
    EXPECT_LE(compilations_after_second, compilations_after_first + 0);
}
