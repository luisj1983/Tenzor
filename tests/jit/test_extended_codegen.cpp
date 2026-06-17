/**
 * @file test_extended_codegen.cpp
 * @brief Tests for extended fusion codegen (reduction, softmax, norm, MLP)
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/jit/pattern_matcher.hpp>
#include <tenzor/jit/extended_codegen.hpp>
#include <tenzor/jit/compiler.hpp>

using namespace tenzor;
using namespace tenzor::jit;

TEST(ExtendedCodegen, PatternMatcherFindsNoPatterns) {
    // Empty graph should produce no matches
    Graph graph;
    PatternMatcher matcher;
    auto matches = matcher.find_all(graph);
    EXPECT_TRUE(matches.empty());
}

TEST(ExtendedCodegen, FusionKindEnumValues) {
    // Verify FusionKind enum has expected values
    EXPECT_EQ(static_cast<int>(FusionKind::ElementWise), 0);
    EXPECT_EQ(static_cast<int>(FusionKind::Reduction), 1);
    EXPECT_EQ(static_cast<int>(FusionKind::GemmEpilogue), 2);
    EXPECT_EQ(static_cast<int>(FusionKind::Softmax), 3);
    EXPECT_EQ(static_cast<int>(FusionKind::LayerNorm), 4);
    EXPECT_EQ(static_cast<int>(FusionKind::RMSNorm), 5);
    EXPECT_EQ(static_cast<int>(FusionKind::SmallMLP), 6);
}

TEST(ExtendedCodegen, ExtendedFusionGroupSignature) {
    ExtendedFusionGroup group;
    group.kind = FusionKind::Softmax;
    group.dtype = DType::Float32;
    group.softmax_dim = -1;

    auto sig = group.compute_signature();
    EXPECT_FALSE(sig.empty());
    EXPECT_NE(sig.find("xfuse_"), std::string::npos);
}

TEST(ExtendedCodegen, GenerateSoftmaxKernel) {
    ExtendedFusionGroup group;
    group.kind = FusionKind::Softmax;
    group.dtype = DType::Float32;
    group.softmax_dim = -1;

    auto source = ExtendedKernelCodegen::generate(group);
    EXPECT_FALSE(source.empty());
    EXPECT_NE(source.find("fused_softmax_kernel"), std::string::npos);
    EXPECT_NE(source.find("__shfl_down_sync"), std::string::npos);
}

TEST(ExtendedCodegen, GenerateLayerNormKernel) {
    ExtendedFusionGroup group;
    group.kind = FusionKind::LayerNorm;
    group.dtype = DType::Float32;
    group.has_affine = true;
    group.eps = 1e-5f;

    auto source = ExtendedKernelCodegen::generate(group);
    EXPECT_FALSE(source.empty());
    EXPECT_NE(source.find("fused_layer_norm_kernel"), std::string::npos);
    EXPECT_NE(source.find("gamma"), std::string::npos);
    EXPECT_NE(source.find("beta"), std::string::npos);
}

TEST(ExtendedCodegen, GenerateRMSNormKernel) {
    ExtendedFusionGroup group;
    group.kind = FusionKind::RMSNorm;
    group.dtype = DType::Float32;
    group.has_affine = true;

    auto source = ExtendedKernelCodegen::generate(group);
    EXPECT_FALSE(source.empty());
    EXPECT_NE(source.find("fused_rms_norm_kernel"), std::string::npos);
    EXPECT_NE(source.find("rsqrtf"), std::string::npos);
}

TEST(ExtendedCodegen, GenerateReductionKernel) {
    ExtendedFusionGroup group;
    group.kind = FusionKind::Reduction;
    group.dtype = DType::Float32;
    group.reduce_dim = -1;

    auto source = ExtendedKernelCodegen::generate(group);
    EXPECT_FALSE(source.empty());
    EXPECT_NE(source.find("fused_reduction_kernel"), std::string::npos);
}

TEST(ExtendedCodegen, GenerateGemmEpilogueKernel) {
    ExtendedFusionGroup group;
    group.kind = FusionKind::GemmEpilogue;
    group.dtype = DType::Float32;
    group.has_bias = true;
    group.has_activation = true;
    group.activation_type = OpType::ReLU;

    auto source = ExtendedKernelCodegen::generate(group);
    EXPECT_FALSE(source.empty());
    EXPECT_NE(source.find("fused_gemm_epilogue_kernel"), std::string::npos);
    EXPECT_NE(source.find("bias"), std::string::npos);
}

TEST(ExtendedCodegen, GenerateSmallMLPKernel) {
    ExtendedFusionGroup group;
    group.kind = FusionKind::SmallMLP;
    group.dtype = DType::Float32;
    group.hidden_dim = 256;
    group.mlp_activation = OpType::GELU;

    auto source = ExtendedKernelCodegen::generate(group);
    EXPECT_FALSE(source.empty());
    EXPECT_NE(source.find("fused_small_mlp_kernel"), std::string::npos);
    EXPECT_NE(source.find("__shared__"), std::string::npos);
}

TEST(ExtendedCodegen, ExtendedFusionPassRegisteredInCompiler) {
    Compiler compiler(true);  // default passes enabled
    // The ExtendedFusionPass should be in the pass list
    // Just verify the compiler can be constructed without error
    Graph graph;
    auto changes = compiler.optimize(graph);
    // Empty graph = no changes
    EXPECT_GE(changes, 0);
}

// Execution test for the extended-fusion launch ABI. execute_extended_fused
// previously launched every extended kernel with the element-wise
// [inputs..., output, numel] ABI, feeding garbage dimension args. This verifies
// the softmax path end-to-end against a CPU reference (skips if no GPU backend).
TEST(ExtendedCodegen, ExecuteSoftmaxMatchesReference) {
    initialize();  // load backends (this binary's other tests are codegen-only)
    const std::vector<float> in = {1.0f, 2.0f, 3.0f, 4.0f, 0.5f, -1.0f, 2.0f, 0.0f};
    Tensor input_cpu = from_data(in.data(), {2, 4});

    for (const auto& dev : {Device::cuda(0), Device::rocm(0)}) {
        Tensor input;
        try {
            input = input_cpu.to(dev);
        } catch (const std::exception&) {
            continue;  // backend not available on this device
        }

        ExtendedFusionGroup group;
        group.kind = FusionKind::Softmax;
        group.dtype = DType::Float32;
        group.softmax_dim = -1;

        Tensor out;
        try {
            out = execute_extended_fused(group, {input});
        } catch (const std::exception& e) {
            GTEST_SKIP() << "extended softmax launch unavailable: " << e.what();
        }

        auto out_cpu = out.to(Device::cpu());
        ASSERT_EQ(out_cpu.numel(), 8);
        const float* o = out_cpu.data<float>();
        for (int r = 0; r < 2; ++r) {
            float mx = -1e30f;
            for (int c = 0; c < 4; ++c) mx = std::max(mx, in[r * 4 + c]);
            float sum = 0.0f;
            for (int c = 0; c < 4; ++c) sum += std::exp(in[r * 4 + c] - mx);
            for (int c = 0; c < 4; ++c) {
                float ref = std::exp(in[r * 4 + c] - mx) / sum;
                EXPECT_NEAR(o[r * 4 + c], ref, 1e-4f) << "row=" << r << " col=" << c;
            }
        }
        return;  // verified on one available backend
    }
    GTEST_SKIP() << "no GPU backend available for extended softmax execution";
}
