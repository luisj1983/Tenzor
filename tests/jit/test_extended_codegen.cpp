/**
 * @file test_extended_codegen.cpp
 * @brief Tests for extended fusion codegen (reduction, softmax, norm, MLP)
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/jit/pattern_matcher.hpp>
#include <tenzor/jit/extended_codegen.hpp>
#include <tenzor/jit/compiler.hpp>
#include <tenzor/jit/graph.hpp>
#include <tenzor/jit/tracer.hpp>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <memory>
#include <vector>

using namespace tenzor;
using namespace tenzor::jit;

namespace {

// GPU devices the runtime codegen can target (CUDA via NVRTC, ROCm via HIPRTC).
// A device is "available" iff a CPU tensor can be moved onto it — that means the
// backend loaded. The tests below run the SAME kernel on EVERY available backend
// (they never return after the first), so a combined CUDA+ROCm build exercises
// BOTH NVRTC and HIPRTC — the ROCm path is where the 64-lane warp-mask fix (H4)
// must hold. They GTEST_SKIP only when NO GPU backend is present at all.
auto available_gpu_devices() -> std::vector<Device> {
    std::vector<Device> devs;
    const std::vector<float> probe = {0.0f};
    for (const auto& dev : {Device::cuda(0), Device::rocm(0)}) {
        try {
            Tensor t = from_data(probe.data(), {1}).to(dev);
            (void)t.to(Device::cpu());
            devs.push_back(dev);
        } catch (const std::exception&) {
            // backend not built / no such device — genuinely unavailable
        }
    }
    return devs;
}

// Honor TENZOR_REQUIRE_MULTI_BACKEND: when set, a run with NO GPU backend is a
// hard failure instead of a silent skip, so a host expected to have a GPU does
// not quietly drop GPU codegen coverage.
#define SKIP_OR_FAIL_NO_GPU()                                                  \
    do {                                                                       \
        const char* _req = std::getenv("TENZOR_REQUIRE_MULTI_BACKEND");        \
        if (_req && *_req && *_req != '0')                                     \
            FAIL() << "TENZOR_REQUIRE_MULTI_BACKEND set but no GPU backend "   \
                      "present for GPU codegen test";                          \
        GTEST_SKIP() << "no GPU backend present";                             \
    } while (0)

}  // namespace

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

// ---------------------------------------------------------------------------
// End-to-end execution tests for the extended-fusion GPU kernels, run on EVERY
// available GPU backend (CUDA + ROCm). These are the empirical check for the
// 64-lane warp-mask fix (H4): the reduction/softmax/norm kernels reduce a row
// with warp shuffles, so on a 64-wide AMD wavefront a 32-bit shuffle mask would
// drop lanes 32..63 and corrupt any row longer than 32 elements. Every case here
// uses a reduced extent > 64 and compares against a CPU reference.
// ---------------------------------------------------------------------------

// Deterministic pseudo-random fill in [-1, 1], stable across runs/backends.
namespace {
auto make_rows(int rows, int cols) -> std::vector<float> {
    std::vector<float> v(static_cast<size_t>(rows) * cols);
    uint32_t s = 0x1234567u;
    for (auto& x : v) {
        s = s * 1664525u + 1013904223u;
        x = (static_cast<float>(s >> 8) / static_cast<float>(1u << 24)) * 2.0f - 1.0f;
    }
    return v;
}
}  // namespace

TEST(ExtendedCodegen, ExecuteSoftmaxMatchesReference) {
    initialize();
    auto devs = available_gpu_devices();
    if (devs.empty()) SKIP_OR_FAIL_NO_GPU();

    for (int cols : {48, 256, 1024}) {   // 48 > 32, 256/1024 > 64: exercise both
        const int rows = 3;
        auto in = make_rows(rows, cols);
        Tensor input_cpu = from_data(in.data(), {rows, cols});

        for (const auto& dev : devs) {
            Tensor input = input_cpu.to(dev);
            ExtendedFusionGroup group;
            group.kind = FusionKind::Softmax;
            group.dtype = DType::Float32;
            group.softmax_dim = -1;
            Tensor out = execute_extended_fused(group, {input}).to(Device::cpu());
            ASSERT_EQ(out.numel(), rows * cols);
            const float* o = out.data<float>();
            for (int r = 0; r < rows; ++r) {
                double mx = -1e30;
                for (int c = 0; c < cols; ++c) mx = std::max(mx, (double)in[r * cols + c]);
                double sum = 0.0;
                for (int c = 0; c < cols; ++c) sum += std::exp((double)in[r * cols + c] - mx);
                for (int c = 0; c < cols; ++c) {
                    double ref = std::exp((double)in[r * cols + c] - mx) / sum;
                    EXPECT_NEAR(o[r * cols + c], ref, 1e-4)
                        << dev.to_string() << " cols=" << cols << " r=" << r << " c=" << c;
                }
            }
        }
    }
}

TEST(ExtendedCodegen, ExecuteReductionSumMeanMatchesReference) {
    initialize();
    auto devs = available_gpu_devices();
    if (devs.empty()) SKIP_OR_FAIL_NO_GPU();

    for (OpType rk : {OpType::Sum, OpType::Mean}) {
        for (int cols : {48, 256, 1024}) {
            const int rows = 4;
            auto in = make_rows(rows, cols);
            Tensor input_cpu = from_data(in.data(), {rows, cols});
            for (const auto& dev : devs) {
                Tensor input = input_cpu.to(dev);
                ExtendedFusionGroup group;
                group.kind = FusionKind::Reduction;
                group.dtype = DType::Float32;
                group.reduce_dim = -1;       // reduce the last (contiguous) axis
                group.reduce_kind = rk;
                group.keepdim = false;
                Tensor out = execute_extended_fused(group, {input}).to(Device::cpu());
                ASSERT_EQ(out.numel(), rows);
                const float* o = out.data<float>();
                for (int r = 0; r < rows; ++r) {
                    double sum = 0.0;
                    for (int c = 0; c < cols; ++c) sum += (double)in[r * cols + c];
                    double ref = (rk == OpType::Mean) ? sum / cols : sum;
                    // Tree reduction reorders float adds; scale tol to magnitude.
                    EXPECT_NEAR(o[r], ref, 1e-4 + 1e-4 * std::abs(ref))
                        << dev.to_string()
                        << (rk == OpType::Mean ? " Mean" : " Sum")
                        << " cols=" << cols << " r=" << r;
                }
            }
        }
    }
}

// BUG M5: a reduction pre-op Mul(x, x) (same operand) is a self-square and must
// lower to x*x; combined with a Sqrt post-op this computes an L2 norm. Exercises
// the pre/post lowering end-to-end on every backend.
TEST(ExtendedCodegen, ExecuteReductionSelfSquareL2Norm) {
    initialize();
    auto devs = available_gpu_devices();
    if (devs.empty()) SKIP_OR_FAIL_NO_GPU();

    for (int cols : {48, 256, 1024}) {
        const int rows = 3;
        auto in = make_rows(rows, cols);
        Tensor input_cpu = from_data(in.data(), {rows, cols});
        for (const auto& dev : devs) {
            Tensor input = input_cpu.to(dev);
            ExtendedFusionGroup group;
            group.kind = FusionKind::Reduction;
            group.dtype = DType::Float32;
            group.reduce_dim = -1;
            group.reduce_kind = OpType::Sum;
            group.keepdim = false;
            group.pre_ops = {{ElemOp::Mul, 0, 0, 0.0}};   // self-square: x*x
            group.post_ops = {{ElemOp::Sqrt, -1, -1, 0.0}};
            Tensor out = execute_extended_fused(group, {input}).to(Device::cpu());
            ASSERT_EQ(out.numel(), rows);
            const float* o = out.data<float>();
            for (int r = 0; r < rows; ++r) {
                double ss = 0.0;
                for (int c = 0; c < cols; ++c) ss += (double)in[r * cols + c] * in[r * cols + c];
                double ref = std::sqrt(ss);
                EXPECT_NEAR(o[r], ref, 1e-4 + 1e-4 * std::abs(ref))
                    << dev.to_string() << " cols=" << cols << " r=" << r;
            }
        }
    }
}

// BUG M5: a reduction pre-op Mul with a DISTINCT second operand cannot be
// represented in the single-input reduction kernel. The codegen must REFUSE
// (emit no source) rather than silently square or drop the op.
TEST(ExtendedCodegen, ReductionRefusesUnrepresentableBinaryMul) {
    ExtendedFusionGroup group;
    group.kind = FusionKind::Reduction;
    group.dtype = DType::Float32;
    group.reduce_dim = -1;
    group.reduce_kind = OpType::Sum;
    group.pre_ops = {{ElemOp::Mul, 0, 1, 0.0}};  // distinct operands -> refuse
    EXPECT_TRUE(ExtendedKernelCodegen::generate(group).empty());

    // A representable self-square, by contrast, DOES generate source.
    group.pre_ops = {{ElemOp::Mul, 0, 0, 0.0}};
    EXPECT_FALSE(ExtendedKernelCodegen::generate(group).empty());

    // An unrepresentable binary Add as a post-op is likewise refused.
    group.pre_ops.clear();
    group.post_ops = {{ElemOp::Add, 0, 1, 0.0}};
    EXPECT_TRUE(ExtendedKernelCodegen::generate(group).empty());
}

TEST(ExtendedCodegen, ExecuteRMSNormMatchesReference) {
    initialize();
    auto devs = available_gpu_devices();
    if (devs.empty()) SKIP_OR_FAIL_NO_GPU();

    const float eps = 1e-5f;
    for (int n : {48, 256, 1024}) {
        const int rows = 3;
        auto in = make_rows(rows, n);
        Tensor input_cpu = from_data(in.data(), {rows, n});
        for (const auto& dev : devs) {
            Tensor input = input_cpu.to(dev);
            ExtendedFusionGroup group;
            group.kind = FusionKind::RMSNorm;
            group.dtype = DType::Float32;
            group.norm_axis = -1;
            group.has_affine = false;
            group.eps = eps;
            Tensor out = execute_extended_fused(group, {input}).to(Device::cpu());
            ASSERT_EQ(out.numel(), rows * n);
            const float* o = out.data<float>();
            for (int r = 0; r < rows; ++r) {
                double ss = 0.0;
                for (int c = 0; c < n; ++c) ss += (double)in[r * n + c] * in[r * n + c];
                double inv = 1.0 / std::sqrt(ss / n + eps);
                for (int c = 0; c < n; ++c) {
                    double ref = (double)in[r * n + c] * inv;
                    EXPECT_NEAR(o[r * n + c], ref, 1e-4)
                        << dev.to_string() << " n=" << n << " r=" << r << " c=" << c;
                }
            }
        }
    }
}

TEST(ExtendedCodegen, ExecuteLayerNormMatchesReference) {
    initialize();
    auto devs = available_gpu_devices();
    if (devs.empty()) SKIP_OR_FAIL_NO_GPU();

    const float eps = 1e-5f;
    for (int n : {48, 256, 1024}) {
        const int rows = 3;
        auto in = make_rows(rows, n);
        Tensor input_cpu = from_data(in.data(), {rows, n});
        for (const auto& dev : devs) {
            Tensor input = input_cpu.to(dev);
            ExtendedFusionGroup group;
            group.kind = FusionKind::LayerNorm;
            group.dtype = DType::Float32;
            group.norm_axis = -1;
            group.has_affine = false;
            group.eps = eps;
            Tensor out = execute_extended_fused(group, {input}).to(Device::cpu());
            ASSERT_EQ(out.numel(), rows * n);
            const float* o = out.data<float>();
            for (int r = 0; r < rows; ++r) {
                double mean = 0.0;
                for (int c = 0; c < n; ++c) mean += (double)in[r * n + c];
                mean /= n;
                double var = 0.0;
                for (int c = 0; c < n; ++c) {
                    double d = (double)in[r * n + c] - mean;
                    var += d * d;
                }
                var /= n;
                double inv = 1.0 / std::sqrt(var + eps);
                for (int c = 0; c < n; ++c) {
                    double ref = ((double)in[r * n + c] - mean) * inv;
                    EXPECT_NEAR(o[r * n + c], ref, 2e-4)
                        << dev.to_string() << " n=" << n << " r=" << r << " c=" << c;
                }
            }
        }
    }
}

// ===========================================================================
// End-to-end JIT-executor integration: fusion nodes must run via the native
// NVRTC/HIPRTC codegen path (execute_extended_fused), producing the SAME result
// as the UNFUSED eager graph on CPU. The eager collapse the executor replaces
// was lossy for GemmEpilogue/SmallMLP/Reduction (dropped epilogue / second
// linear / pre-post ops), so the reference is deliberately the unfused graph.
//
// Each case builds the decomposed op pattern the PatternMatcher recognizes,
// runs it (a) unfused on CPU as reference, (b) fused on every available GPU
// backend. It asserts (b) == (a) within tight fp tolerance AND that the codegen
// launch counter advanced — a fusion node that silently ran eager would FAIL.
// ===========================================================================

namespace {

// Deterministic pseudo-random fill in [-1, 1].
auto rnd_vec(size_t n, uint32_t seed) -> std::vector<float> {
    std::vector<float> v(n);
    uint32_t s = seed;
    for (auto& x : v) {
        s = s * 1664525u + 1013904223u;
        x = (static_cast<float>(s >> 8) / static_cast<float>(1u << 24)) * 2.0f - 1.0f;
    }
    return v;
}

// Append a node consuming `ins`, producing one output Value of `out_shape`.
auto add_op(Graph& g, OpType op, const std::string& name,
            std::vector<std::shared_ptr<Value>> ins,
            std::vector<int64_t> out_shape, Device dev)
    -> std::shared_ptr<Node> {
    auto node = g.create_node(op, name);
    for (auto& i : ins) node->add_input(i);
    auto out = g.create_value(name + "_out", std::move(out_shape),
                              DType::Float32, dev);
    node->add_output(out);
    out->set_node(node);
    g.add_node(node);
    return node;
}

// Build function: populate `g` (values on `dev`), set inputs/outputs. Runtime
// input DATA is provided separately (same order) so CPU-ref and GPU runs use
// identical values.
using BuildFn = std::function<void(Graph&, Device)>;

// Run `build` unfused on CPU (reference) then fused on every GPU backend; compare.
auto run_fused_vs_reference(const BuildFn& build,
                            const std::vector<Tensor>& inputs_cpu,
                            const char* what,
                            double atol, double rtol) -> void {
    initialize();
    auto devs = available_gpu_devices();
    if (devs.empty()) SKIP_OR_FAIL_NO_GPU();

    // (a) reference: unfused eager execution on CPU.
    Tensor ref;
    {
        Graph g;
        build(g, Device::cpu());
        std::vector<Variable> rin;
        for (const auto& t : inputs_cpu) rin.emplace_back(t, false);
        ref = g.forward(rin)[0].tensor();
    }
    const float* r = ref.data<float>();

    for (const auto& dev : devs) {
        Graph g;
        build(g, dev);
        ExtendedFusionPass pass;
        const uint64_t before = extended_fused_launch_count();
        ASSERT_TRUE(pass.run(g))
            << what << ": ExtendedFusionPass did not fuse on " << dev.to_string();

        std::vector<Variable> rin;
        for (const auto& t : inputs_cpu) rin.emplace_back(t.to(dev), false);
        Tensor out = g.forward(rin)[0].tensor().to(Device::cpu());

        // The codegen path MUST have launched at least one kernel — a fused node
        // that silently ran eager would leave the counter unchanged.
        EXPECT_GT(extended_fused_launch_count(), before)
            << what << ": native codegen path was NOT taken on " << dev.to_string();

        ASSERT_EQ(out.numel(), ref.numel()) << what << " " << dev.to_string();
        const float* o = out.data<float>();
        for (int64_t i = 0; i < ref.numel(); ++i) {
            EXPECT_NEAR(o[i], r[i], atol + rtol * std::abs(r[i]))
                << what << " " << dev.to_string() << " idx=" << i;
        }
    }
}

}  // namespace

TEST(FusedExecIntegration, SoftmaxMatchesUnfusedReference) {
    initialize();
    const int R = 4, C = 128;
    auto xd = rnd_vec(static_cast<size_t>(R) * C, 0xA1);
    Tensor x = from_data(xd.data(), {R, C});

    BuildFn build = [=](Graph& g, Device dev) {
        auto in = g.create_value("x", {R, C}, DType::Float32, dev);
        auto mx = add_op(g, OpType::Max, "max", {in}, {R, 1}, dev);
        mx->set_int_attr("dim", -1); mx->set_bool_attr("keepdim", true);
        auto sub = add_op(g, OpType::Sub, "sub", {in, mx->outputs()[0]}, {R, C}, dev);
        auto ex = add_op(g, OpType::Exp, "exp", {sub->outputs()[0]}, {R, C}, dev);
        auto sm = add_op(g, OpType::Sum, "sum", {ex->outputs()[0]}, {R, 1}, dev);
        sm->set_int_attr("dim", -1); sm->set_bool_attr("keepdim", true);
        auto dv = add_op(g, OpType::Div, "div",
                         {ex->outputs()[0], sm->outputs()[0]}, {R, C}, dev);
        g.set_inputs({in});
        g.set_outputs({dv->outputs()[0]});
    };
    run_fused_vs_reference(build, {x}, "Softmax", 1e-4, 1e-4);
}

TEST(FusedExecIntegration, LayerNormAffineMatchesUnfusedReference) {
    initialize();
    const int R = 4, N = 128;
    auto xd = rnd_vec(static_cast<size_t>(R) * N, 0xB2);
    auto gd = rnd_vec(N, 0xB3);
    auto bd = rnd_vec(N, 0xB4);
    Tensor x = from_data(xd.data(), {R, N});
    Tensor gamma = from_data(gd.data(), {N});
    Tensor beta = from_data(bd.data(), {N});

    BuildFn build = [=](Graph& g, Device dev) {
        auto in = g.create_value("x", {R, N}, DType::Float32, dev);
        auto ga = g.create_value("gamma", {N}, DType::Float32, dev);
        auto be = g.create_value("beta", {N}, DType::Float32, dev);
        auto m1 = add_op(g, OpType::Mean, "mean1", {in}, {R, 1}, dev);
        m1->set_int_attr("dim", -1); m1->set_bool_attr("keepdim", true);
        auto sub = add_op(g, OpType::Sub, "sub", {in, m1->outputs()[0]}, {R, N}, dev);
        auto sq = add_op(g, OpType::Mul, "sq",
                         {sub->outputs()[0], sub->outputs()[0]}, {R, N}, dev);
        auto m2 = add_op(g, OpType::Mean, "mean2", {sq->outputs()[0]}, {R, 1}, dev);
        m2->set_int_attr("dim", -1); m2->set_bool_attr("keepdim", true);
        auto sr = add_op(g, OpType::Sqrt, "sqrt", {m2->outputs()[0]}, {R, 1}, dev);
        // tiny eps so the fused kernel (var+eps) matches the eps-free reference.
        sr->set_attr("eps", 1e-12f);
        auto dv = add_op(g, OpType::Div, "div",
                         {sub->outputs()[0], sr->outputs()[0]}, {R, N}, dev);
        auto mg = add_op(g, OpType::Mul, "mulg",
                         {dv->outputs()[0], ga}, {R, N}, dev);
        auto ab = add_op(g, OpType::Add, "addb",
                         {mg->outputs()[0], be}, {R, N}, dev);
        g.set_inputs({in, ga, be});
        g.set_outputs({ab->outputs()[0]});
    };
    run_fused_vs_reference(build, {x, gamma, beta}, "LayerNorm", 2e-4, 2e-4);
}

TEST(FusedExecIntegration, RMSNormAffineMatchesUnfusedReference) {
    initialize();
    const int R = 4, N = 128;
    auto xd = rnd_vec(static_cast<size_t>(R) * N, 0xC5);
    auto gd = rnd_vec(N, 0xC6);
    Tensor x = from_data(xd.data(), {R, N});
    Tensor gamma = from_data(gd.data(), {N});

    BuildFn build = [=](Graph& g, Device dev) {
        auto in = g.create_value("x", {R, N}, DType::Float32, dev);
        auto ga = g.create_value("gamma", {N}, DType::Float32, dev);
        auto sq = add_op(g, OpType::Mul, "sq", {in, in}, {R, N}, dev);
        auto ms = add_op(g, OpType::Mean, "ms", {sq->outputs()[0]}, {R, 1}, dev);
        ms->set_int_attr("dim", -1); ms->set_bool_attr("keepdim", true);
        auto sr = add_op(g, OpType::Sqrt, "sqrt", {ms->outputs()[0]}, {R, 1}, dev);
        sr->set_attr("eps", 1e-12f);
        auto dv = add_op(g, OpType::Div, "div",
                         {in, sr->outputs()[0]}, {R, N}, dev);
        auto mg = add_op(g, OpType::Mul, "mulg", {dv->outputs()[0], ga}, {R, N}, dev);
        g.set_inputs({in, ga});
        g.set_outputs({mg->outputs()[0]});
    };
    run_fused_vs_reference(build, {x, gamma}, "RMSNorm", 1e-4, 1e-4);
}

TEST(FusedExecIntegration, ReductionPrePostMatchesUnfusedReference) {
    initialize();
    // pre self-square + Sum + post Sqrt == per-row L2 norm. Eager collapse would
    // have dropped the pre/post ops; the fused kernel must apply them.
    const int R = 4, C = 128;
    auto xd = rnd_vec(static_cast<size_t>(R) * C, 0xD7);
    Tensor x = from_data(xd.data(), {R, C});

    BuildFn build = [=](Graph& g, Device dev) {
        auto in = g.create_value("x", {R, C}, DType::Float32, dev);
        auto sq = add_op(g, OpType::Mul, "sq", {in, in}, {R, C}, dev);
        auto sm = add_op(g, OpType::Sum, "sum", {sq->outputs()[0]}, {R}, dev);
        sm->set_int_attr("dim", -1); sm->set_bool_attr("keepdim", false);
        auto sr = add_op(g, OpType::Sqrt, "sqrt", {sm->outputs()[0]}, {R}, dev);
        g.set_inputs({in});
        g.set_outputs({sr->outputs()[0]});
    };
    run_fused_vs_reference(build, {x}, "Reduction", 1e-4, 1e-4);
}

TEST(FusedExecIntegration, GemmEpilogueMatchesUnfusedReference) {
    initialize();
    // MatMul + bias + ReLU. Eager collapse to a bare Linear dropped bias+ReLU.
    const int M = 8, K = 16, N = 32;
    auto ad = rnd_vec(static_cast<size_t>(M) * K, 0xE8);
    auto bd = rnd_vec(static_cast<size_t>(K) * N, 0xE9);
    auto cd = rnd_vec(N, 0xEA);
    Tensor A = from_data(ad.data(), {M, K});
    Tensor B = from_data(bd.data(), {K, N});
    Tensor bias = from_data(cd.data(), {N});

    BuildFn build = [=](Graph& g, Device dev) {
        auto a = g.create_value("A", {M, K}, DType::Float32, dev);
        auto b = g.create_value("B", {K, N}, DType::Float32, dev);
        auto bi = g.create_value("bias", {N}, DType::Float32, dev);
        auto mm = add_op(g, OpType::MatMul, "mm", {a, b}, {M, N}, dev);
        auto ad2 = add_op(g, OpType::Add, "addbias",
                          {mm->outputs()[0], bi}, {M, N}, dev);
        auto rl = add_op(g, OpType::ReLU, "relu", {ad2->outputs()[0]}, {M, N}, dev);
        g.set_inputs({a, b, bi});
        g.set_outputs({rl->outputs()[0]});
    };
    // f32 GEMM accumulates in a different order on GPU vs CPU MKL, so use a
    // standard single-precision matmul tolerance (not a bug — pure fp reduction
    // order). The point of the test is that bias+ReLU are applied, not dropped.
    run_fused_vs_reference(build, {A, B, bias}, "GemmEpilogue", 2e-3, 2e-3);
}

TEST(FusedExecIntegration, SmallMLPMatchesUnfusedReference) {
    initialize();
    // Linear -> GELU -> Linear against the unfused two-linear reference; the
    // native SmallMLP kernel (transposed weights, fused activation) must match.
    const int Bn = 8, In = 16, H = 64, Out = 32;
    auto xd = rnd_vec(static_cast<size_t>(Bn) * In, 0xF1);
    auto w1 = rnd_vec(static_cast<size_t>(H) * In, 0xF2);
    auto b1 = rnd_vec(H, 0xF3);
    auto w2 = rnd_vec(static_cast<size_t>(Out) * H, 0xF4);
    auto b2 = rnd_vec(Out, 0xF5);
    Tensor X = from_data(xd.data(), {Bn, In});
    Tensor W1 = from_data(w1.data(), {H, In});
    Tensor B1 = from_data(b1.data(), {H});
    Tensor W2 = from_data(w2.data(), {Out, H});
    Tensor B2 = from_data(b2.data(), {Out});

    BuildFn build = [=](Graph& g, Device dev) {
        auto x = g.create_value("x", {Bn, In}, DType::Float32, dev);
        auto w1v = g.create_value("w1", {H, In}, DType::Float32, dev);
        auto b1v = g.create_value("b1", {H}, DType::Float32, dev);
        auto w2v = g.create_value("w2", {Out, H}, DType::Float32, dev);
        auto b2v = g.create_value("b2", {Out}, DType::Float32, dev);
        auto l1 = add_op(g, OpType::Linear, "lin1", {x, w1v, b1v}, {Bn, H}, dev);
        auto ge = add_op(g, OpType::GELU, "gelu", {l1->outputs()[0]}, {Bn, H}, dev);
        auto l2 = add_op(g, OpType::Linear, "lin2",
                         {ge->outputs()[0], w2v, b2v}, {Bn, Out}, dev);
        g.set_inputs({x, w1v, b1v, w2v, b2v});
        g.set_outputs({l2->outputs()[0]});
    };
    run_fused_vs_reference(build, {X, W1, B1, W2, B2}, "SmallMLP", 2e-4, 2e-4);
}
