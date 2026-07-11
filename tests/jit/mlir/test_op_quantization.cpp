// JIT-R081 — MLIR lowering for Quantize/Dequantize/QuantizedLinear/
// QuantizedLinearStatic/QuantizedConv2d.
//
// Coverage:
//   - QuantizedLinearStatic: reached via the NORMAL CompiledFunction trace
//     path (nn::quantization::QuantizedLinear::forward_impl records this
//     node directly when calibrated). Full numeric IREE round-trip vs eager.
//   - QuantizedLinear / QuantizedConv2d (dynamic): only reachable via the
//     low-level Graph/QuantizationPass API (never through CompiledFunction —
//     QuantizationPass is opt-in). Built manually via TracingGuard +
//     QuantizationPass, lowered via GraphToMLIR, compiled+invoked directly
//     via compile_mlir/IreeInvoker, and compared against
//     nn::quantization::quantized_linear_dynamic/quantized_conv2d_dynamic
//     (the same eager reference test_jit_quantization.cpp's interpreter-path
//     tests use).
//   - Quantize / Dequantize (standalone QuantStub/DeQuantStub ops): as of
//     this fix, NO producer anywhere in the codebase emits these as a graph
//     node (verified: zero references outside the enum / shape-inference /
//     interpreter-throw sites) — the handler exists and is verified here via
//     a hand-built Graph/Node (bypassing the tracer, since there is no
//     traced producer to exercise), establishing the "scale"/"zero_point"
//     attr convention for whenever a producer is added.

#include "tenzor/autograd/variable.hpp"
#include "tenzor/jit/compile.hpp"
#include "tenzor/jit/compiler.hpp"
#include "tenzor/jit/graph.hpp"
#include "tenzor/jit/mlir/iree_compile.hpp"
#include "tenzor/jit/mlir/iree_runtime.hpp"
#include "tenzor/jit/mlir/lowering.hpp"
#include "tenzor/jit/tracer.hpp"
#include "tenzor/nn/layers/conv.hpp"
#include "tenzor/nn/layers/linear.hpp"
#include "tenzor/nn/quantization/quantized_layers.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/reduction.hpp"
#include "tenzor/tenzor.hpp"

#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <filesystem>
#include <random>
#include <string>

#include "mlir_target_util.hpp"

namespace tzj = ::tenzor::jit;
namespace tzm = ::tenzor::jit::mlir_jit;
namespace fs = std::filesystem;

namespace {

auto ensure_core_init() -> void {
    static const bool inited = []() {
        ::tenzor::initialize();
        return true;
    }();
    (void)inited;
}

auto make_tmp_dir(const std::string& tag) -> fs::path {
    auto now = std::chrono::system_clock::now().time_since_epoch().count();
    std::mt19937_64 rng(static_cast<uint64_t>(now));
    fs::path dir = fs::temp_directory_path() /
                   ("tenzor_quant_lower_" + tag + "_" + std::to_string(rng()));
    fs::create_directories(dir);
    return dir;
}

// Relative L2 error, matching test_jit_quantization.cpp's rel_l2 helper.
auto rel_l2(const ::tenzor::Tensor& a, const ::tenzor::Tensor& b) -> float {
    auto ac = a.to(::tenzor::DType::Float32).to(::tenzor::Device::cpu());
    auto bc = b.to(::tenzor::DType::Float32).to(::tenzor::Device::cpu());
    auto diff = ::tenzor::sub(ac, bc);
    float num = std::sqrt(::tenzor::sum(::tenzor::mul(diff, diff)).item<float>());
    float den = std::sqrt(::tenzor::sum(::tenzor::mul(ac, ac)).item<float>()) + 1e-8f;
    return num / den;
}

// Compile a Graph directly (bypassing CompiledFunction, which never applies
// QuantizationPass) and invoke it via the same low-level compile_mlir +
// IreeInvoker path test_iree_invoke.cpp exercises.
auto lower_compile_invoke(const tzj::Graph& g, const std::vector<::tenzor::Tensor>& inputs,
                          const std::string& tag) -> std::vector<::tenzor::Tensor> {
    tzm::GraphToMLIR lowerer;
    const std::string mlir_text = lowerer.lower(g);

    const fs::path tmp = make_tmp_dir(tag);
    tzm::CompileOptions opts;
    opts.target = "llvm-cpu";
    opts.cache_dir = tmp;
    auto artifact = tzm::compile_mlir(mlir_text, opts);
    auto invoker = tzm::IreeInvoker::load(artifact);
    auto outs = invoker->invoke(inputs);
    fs::remove_all(tmp);
    return outs;
}

}  // namespace

// ─── QuantizedLinearStatic: full numeric IREE round-trip via CompiledFunction ──

TEST(OpQuantization, QuantizedLinearStaticEndToEnd) {
    namespace mt = ::tenzor::testing::mlir;
    ensure_core_init();
    mt::ensure_core_init();

    const int64_t B = 4, IN = 16, OUT = 8;
    ::tenzor::nn::Linear fp_linear(IN, OUT, /*bias=*/true);

    auto qconfig = ::tenzor::nn::quantization::DefaultQConfigs::default_qconfig();
    auto q_layer = ::tenzor::nn::quantization::QuantizedLinear::from_float(fp_linear, qconfig);

    // Calibrate the activation qparams so forward_impl records
    // OpType::QuantizedLinearStatic instead of throwing (JIT-R066's
    // "dynamic activation quantization cannot be traced" guard).
    auto input = ::tenzor::randn({B, IN}, ::tenzor::DType::Float32, ::tenzor::Device::cpu());
    auto in_qparams = ::tenzor::nn::quantization::quantize_per_tensor_symmetric(input).params();
    q_layer->set_activation_qparams(in_qparams);

    ::tenzor::jit::CompiledFunction::FnType fn =
        [&q_layer](const ::tenzor::Variable& x) -> ::tenzor::Variable {
            return q_layer->forward(x);
        };

    const auto eager = fn(::tenzor::Variable(input, false)).tensor();

    ::tenzor::jit::CompileConfig cfg;
    cfg.backend = "mlir";
    cfg.target = "llvm-cpu";
    auto compiled = ::tenzor::jit::CompiledFunction(fn, cfg);

    mt::reset_jit_stats();
    auto jitted = compiled(::tenzor::Variable(input, false));
    mt::assert_jit_used("quantized_linear_static", "llvm-cpu");

    auto eager_cpu = eager.to(::tenzor::Device::cpu());
    auto jit_cpu = jitted.tensor().to(::tenzor::Device::cpu());
    ASSERT_EQ(eager_cpu.numel(), jit_cpu.numel());
    EXPECT_LT(rel_l2(eager_cpu, jit_cpu), 1e-3f)
        << "QuantizedLinearStatic MLIR lowering diverged from eager/interpreter";
}

// ─── QuantizedLinear / QuantizedConv2d (dynamic): manual Graph + QuantizationPass ──

TEST(OpQuantization, QuantizedLinearDynamicEndToEnd) {
    ensure_core_init();

    const int64_t B = 4, IN = 16, OUT = 8;
    ::tenzor::nn::Linear fc(IN, OUT, /*bias=*/true);
    auto input = ::tenzor::randn({B, IN}, ::tenzor::DType::Float32, ::tenzor::Device::cpu());

    // Eager reference: the SAME helper the JIT interpreter uses (matches
    // test_jit_quantization.cpp's QuantizedLinearCrossBackend convention).
    auto q_eager = ::tenzor::nn::quantization::quantized_linear_dynamic(
        input, fc.weight()->tensor(), std::optional<::tenzor::Tensor>(fc.bias()->tensor()));

    std::shared_ptr<tzj::Graph> graph;
    {
        tzj::TracingGuard guard;
        ::tenzor::Variable x(input, /*requires_grad=*/false);
        ::tenzor::Variable y = fc.forward(x);
        graph = guard.get_graph({x}, {y});
    }
    ASSERT_TRUE(graph != nullptr);

    tzj::Compiler compiler(/*enable_default_passes=*/false);
    compiler.add_pass(std::make_unique<tzj::QuantizationPass>());
    compiler.optimize(*graph);

    auto outs = lower_compile_invoke(*graph, {input}, "qlinear_dynamic");
    ASSERT_EQ(outs.size(), 1u);

    EXPECT_LT(rel_l2(q_eager, outs[0]), 1e-3f)
        << "QuantizedLinear (dynamic) MLIR lowering diverged from "
           "quantized_linear_dynamic eager reference";
}

TEST(OpQuantization, QuantizedConv2dDynamicEndToEnd) {
    ensure_core_init();

    const int64_t N = 2, C = 4, H = 8, W = 8, OUTC = 6, K = 3;
    ::tenzor::nn::Conv2d conv(C, OUTC, K, /*stride=*/1, /*padding=*/1);
    auto input = ::tenzor::randn({N, C, H, W}, ::tenzor::DType::Float32, ::tenzor::Device::cpu());

    auto q_eager = ::tenzor::nn::quantization::quantized_conv2d_dynamic(
        input, conv.get_parameter("weight")->tensor(),
        std::optional<::tenzor::Tensor>(conv.get_parameter("bias")->tensor()),
        /*stride=*/1, /*padding=*/1, /*dilation=*/1, /*groups=*/1);

    std::shared_ptr<tzj::Graph> graph;
    {
        tzj::TracingGuard guard;
        ::tenzor::Variable x(input, /*requires_grad=*/false);
        ::tenzor::Variable y = conv.forward(x);
        graph = guard.get_graph({x}, {y});
    }
    ASSERT_TRUE(graph != nullptr);

    tzj::Compiler compiler(/*enable_default_passes=*/false);
    compiler.add_pass(std::make_unique<tzj::QuantizationPass>());
    compiler.optimize(*graph);

    auto outs = lower_compile_invoke(*graph, {input}, "qconv2d_dynamic");
    ASSERT_EQ(outs.size(), 1u);

    EXPECT_LT(rel_l2(q_eager, outs[0]), 1e-3f)
        << "QuantizedConv2d (dynamic) MLIR lowering diverged from "
           "quantized_conv2d_dynamic eager reference";
}

// ─── Standalone Quantize / Dequantize: hand-built Graph (no current producer) ──

TEST(OpQuantization, DequantizeHandBuiltGraph) {
    ensure_core_init();

    tzj::Graph g;
    auto x = g.create_value("x", {4}, ::tenzor::DType::Int8, ::tenzor::Device::cpu());
    g.set_inputs({x});

    auto node = g.create_node(tzj::OpType::Dequantize, "dequant");
    node->add_input(x);
    node->set_attr("scale", 0.5);
    node->set_int_attr("zero_point", 2);
    auto y = g.create_value("y", {4}, ::tenzor::DType::Float32, ::tenzor::Device::cpu());
    node->add_output(y);
    g.add_node(node);
    g.set_outputs({y});

    ::tenzor::Tensor x_t({4}, ::tenzor::DType::Int8, ::tenzor::Device::cpu());
    {
        auto* p = x_t.data<int8_t>();
        p[0] = 2; p[1] = 4; p[2] = -3; p[3] = 127;
    }

    auto outs = lower_compile_invoke(g, {x_t}, "dequantize");
    ASSERT_EQ(outs.size(), 1u);
    ASSERT_EQ(outs[0].dtype(), ::tenzor::DType::Float32);
    const float* p = outs[0].data<const float>();
    // dequantized = (quantized - zero_point) * scale
    EXPECT_NEAR(p[0], (2 - 2) * 0.5f, 1e-5f);
    EXPECT_NEAR(p[1], (4 - 2) * 0.5f, 1e-5f);
    EXPECT_NEAR(p[2], (-3 - 2) * 0.5f, 1e-5f);
    EXPECT_NEAR(p[3], (127 - 2) * 0.5f, 1e-5f);
}

TEST(OpQuantization, QuantizeHandBuiltGraph) {
    ensure_core_init();

    tzj::Graph g;
    auto x = g.create_value("x", {4}, ::tenzor::DType::Float32, ::tenzor::Device::cpu());
    g.set_inputs({x});

    auto node = g.create_node(tzj::OpType::Quantize, "quant");
    node->add_input(x);
    node->set_attr("scale", 0.5);
    node->set_int_attr("zero_point", 2);
    auto y = g.create_value("y", {4}, ::tenzor::DType::Int8, ::tenzor::Device::cpu());
    node->add_output(y);
    g.add_node(node);
    g.set_outputs({y});

    ::tenzor::Tensor x_t({4}, ::tenzor::DType::Float32, ::tenzor::Device::cpu());
    {
        auto* p = x_t.data<float>();
        p[0] = 0.0f; p[1] = 1.0f; p[2] = -1.5f; p[3] = 62.0f;
    }

    auto outs = lower_compile_invoke(g, {x_t}, "quantize");
    ASSERT_EQ(outs.size(), 1u);
    ASSERT_EQ(outs[0].dtype(), ::tenzor::DType::Int8);
    const int8_t* p = outs[0].data<const int8_t>();
    // quantized = clamp(round_nearest_even(value/scale) + zero_point, -128, 127)
    EXPECT_EQ(p[0], 2);   // round(0/0.5)+2 = 2
    EXPECT_EQ(p[1], 4);   // round(1/0.5)+2 = 4
    EXPECT_EQ(p[2], -1);  // round(-1.5/0.5)+2 = -3+2 = -1
    EXPECT_EQ(p[3], 126); // round(62/0.5)+2 = 124+2 = 126
}

TEST(OpQuantization, QuantizeDequantizeRoundTrip) {
    ensure_core_init();

    tzj::Graph g;
    auto x = g.create_value("x", {6}, ::tenzor::DType::Float32, ::tenzor::Device::cpu());
    g.set_inputs({x});

    auto qnode = g.create_node(tzj::OpType::Quantize, "quant");
    qnode->add_input(x);
    qnode->set_attr("scale", 0.1);
    qnode->set_int_attr("zero_point", 0);
    auto q = g.create_value("q", {6}, ::tenzor::DType::Int8, ::tenzor::Device::cpu());
    qnode->add_output(q);
    g.add_node(qnode);

    auto dqnode = g.create_node(tzj::OpType::Dequantize, "dequant");
    dqnode->add_input(q);
    dqnode->set_attr("scale", 0.1);
    dqnode->set_int_attr("zero_point", 0);
    auto y = g.create_value("y", {6}, ::tenzor::DType::Float32, ::tenzor::Device::cpu());
    dqnode->add_output(y);
    g.add_node(dqnode);

    g.set_outputs({y});

    ::tenzor::Tensor x_t({6}, ::tenzor::DType::Float32, ::tenzor::Device::cpu());
    {
        auto* p = x_t.data<float>();
        p[0] = 0.3f; p[1] = -0.7f; p[2] = 1.25f; p[3] = -5.0f; p[4] = 12.0f; p[5] = 0.0f;
    }

    auto outs = lower_compile_invoke(g, {x_t}, "roundtrip");
    ASSERT_EQ(outs.size(), 1u);
    const float* in_p = x_t.data<const float>();
    const float* out_p = outs[0].data<const float>();
    for (int i = 0; i < 6; ++i) {
        // Round-trip error must stay within one quantization step (scale=0.1).
        EXPECT_NEAR(in_p[i], out_p[i], 0.1f + 1e-5f) << "i=" << i;
    }
}
