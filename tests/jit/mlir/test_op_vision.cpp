// Phase 13 / Group C.8 — Vision ops via the MLIR backend.
//
// Covered ops:
//   Conv2d              → stablehlo.convolution (NCHW, OIHW)
//   MaxPool2d           → stablehlo.reduce_window {maximum}
//   AvgPool2d           → stablehlo.reduce_window {add} / window_area
//   AdaptiveAvgPool2d   → reduce_window with computed kernel/stride
//   Dropout             → identity (inference)
//   ResidualAdd         → stablehlo.add (covered by elementwise tests)
//   Padding             → stablehlo.pad
//   Interpolate         → reshape + broadcast_in_dim + reshape (nearest,
//                         integer upsample only in MVP-1)

#include "tenzor/jit/graph.hpp"
#include "tenzor/jit/mlir/lowering.hpp"
#include "tenzor/jit/mlir/iree_compile.hpp"
#include "tenzor/jit/mlir/iree_paths.hpp"
#include "tenzor/jit/tracer.hpp"
#include "tenzor/tenzor.hpp"

#include <gtest/gtest.h>

#include "mlir_target_util.hpp"

#include <filesystem>
#include <fstream>
#include <string>

namespace tzj = ::tenzor::jit;
namespace tzm = ::tenzor::jit::mlir_jit;

namespace {

auto ensure_core_init() -> void {
    static const bool inited = []() {
        ::tenzor::initialize();
        return true;
    }();
    (void)inited;
}

void assert_iree_compile_accepts(const std::string& mlir_text) {
    auto tmp = std::filesystem::temp_directory_path() /
               ("tenzor_op_vision_" + std::to_string(::getpid()) + "_" +
                std::to_string(reinterpret_cast<std::uintptr_t>(&mlir_text)));
    std::filesystem::create_directories(tmp);
    auto in_path  = tmp / "mod.mlir";
    auto out_path = tmp / "mod.vmfb";
    { std::ofstream of(in_path); of << mlir_text; }
    const std::string& iree_compile =
        ::tenzor::jit::mlir_jit::resolve_iree_compile();
    std::string cmd = iree_compile +
        " --iree-hal-target-backends=llvm-cpu \"" + in_path.string() +
        "\" -o \"" + out_path.string() + "\" 2>&1";
    const int rc = std::system(cmd.c_str());
    EXPECT_EQ(rc, 0) << "iree-compile rejected emitted MLIR:\n" << mlir_text;
    std::filesystem::remove_all(tmp);
}

}  // namespace

TEST(OpVision, Conv2dEmitsAndParses) {
    ensure_core_init();
    // x: (1, 3, 8, 8); w: (4, 3, 3, 3); out: (1, 4, 6, 6) with stride=1,
    // no padding.
    tzj::Graph g;
    auto x = g.create_value("x", {1, 3, 8, 8}, ::tenzor::DType::Float32,
                            ::tenzor::Device::cpu());
    auto w = g.create_value("w", {4, 3, 3, 3}, ::tenzor::DType::Float32,
                            ::tenzor::Device::cpu());
    g.set_inputs({x, w});
    auto node = g.create_node(tzj::OpType::Conv2d);
    node->add_input(x);
    node->add_input(w);
    auto z = g.create_value("z", {1, 4, 6, 6}, ::tenzor::DType::Float32,
                            ::tenzor::Device::cpu());
    node->add_output(z);
    node->set_int_attr("stride_h", 1);
    node->set_int_attr("stride_w", 1);
    node->set_int_attr("padding_h", 0);
    node->set_int_attr("padding_w", 0);
    node->set_int_attr("dilation_h", 1);
    node->set_int_attr("dilation_w", 1);
    node->set_int_attr("groups", 1);
    g.add_node(node);
    g.set_outputs({z});
    tzm::GraphToMLIR lowerer;
    auto mlir = lowerer.lower(g);
    EXPECT_NE(mlir.find("stablehlo.convolution"), std::string::npos) << mlir;
    assert_iree_compile_accepts(mlir);
}

TEST(OpVision, MaxPool2dEmitsAndParses) {
    ensure_core_init();
    // x: (1, 3, 8, 8); k=2, s=2 → (1, 3, 4, 4)
    tzj::Graph g;
    auto x = g.create_value("x", {1, 3, 8, 8}, ::tenzor::DType::Float32,
                            ::tenzor::Device::cpu());
    g.set_inputs({x});
    auto node = g.create_node(tzj::OpType::MaxPool2d);
    node->add_input(x);
    auto z = g.create_value("z", {1, 3, 4, 4}, ::tenzor::DType::Float32,
                            ::tenzor::Device::cpu());
    node->add_output(z);
    node->set_int_attr("kernel_h", 2);
    node->set_int_attr("kernel_w", 2);
    node->set_int_attr("stride_h", 2);
    node->set_int_attr("stride_w", 2);
    g.add_node(node);
    g.set_outputs({z});
    tzm::GraphToMLIR lowerer;
    auto mlir = lowerer.lower(g);
    EXPECT_NE(mlir.find("stablehlo.reduce_window"), std::string::npos) << mlir;
    EXPECT_NE(mlir.find("stablehlo.maximum"), std::string::npos) << mlir;
    assert_iree_compile_accepts(mlir);
}

TEST(OpVision, AvgPool2dEmitsAndParses) {
    ensure_core_init();
    tzj::Graph g;
    auto x = g.create_value("x", {1, 3, 8, 8}, ::tenzor::DType::Float32,
                            ::tenzor::Device::cpu());
    g.set_inputs({x});
    auto node = g.create_node(tzj::OpType::AvgPool2d);
    node->add_input(x);
    auto z = g.create_value("z", {1, 3, 4, 4}, ::tenzor::DType::Float32,
                            ::tenzor::Device::cpu());
    node->add_output(z);
    node->set_int_attr("kernel_h", 2);
    node->set_int_attr("kernel_w", 2);
    node->set_int_attr("stride_h", 2);
    node->set_int_attr("stride_w", 2);
    g.add_node(node);
    g.set_outputs({z});
    tzm::GraphToMLIR lowerer;
    auto mlir = lowerer.lower(g);
    EXPECT_NE(mlir.find("stablehlo.reduce_window"), std::string::npos) << mlir;
    EXPECT_NE(mlir.find("stablehlo.divide"), std::string::npos) << mlir;
    assert_iree_compile_accepts(mlir);
}

TEST(OpVision, AdaptiveAvgPool2dEmitsAndParses) {
    ensure_core_init();
    // (1, 3, 6, 6) → (1, 3, 2, 2)
    tzj::Graph g;
    auto x = g.create_value("x", {1, 3, 6, 6}, ::tenzor::DType::Float32,
                            ::tenzor::Device::cpu());
    g.set_inputs({x});
    auto node = g.create_node(tzj::OpType::AdaptiveAvgPool2d);
    node->add_input(x);
    auto z = g.create_value("z", {1, 3, 2, 2}, ::tenzor::DType::Float32,
                            ::tenzor::Device::cpu());
    node->add_output(z);
    g.add_node(node);
    g.set_outputs({z});
    tzm::GraphToMLIR lowerer;
    auto mlir = lowerer.lower(g);
    EXPECT_NE(mlir.find("stablehlo.reduce_window"), std::string::npos) << mlir;
    assert_iree_compile_accepts(mlir);
}

TEST(OpVision, DropoutIsIdentity) {
    ensure_core_init();
    tzj::Graph g;
    auto x = g.create_value("x", {4}, ::tenzor::DType::Float32,
                            ::tenzor::Device::cpu());
    g.set_inputs({x});
    auto node = g.create_node(tzj::OpType::Dropout);
    node->add_input(x);
    auto z = g.create_value("z", {4}, ::tenzor::DType::Float32,
                            ::tenzor::Device::cpu());
    node->add_output(z);
    g.add_node(node);
    g.set_outputs({z});
    tzm::GraphToMLIR lowerer;
    auto mlir = lowerer.lower(g);
    // Inference dropout is an identity — no op inserted, just return %arg0.
    EXPECT_NE(mlir.find("return %arg0"), std::string::npos) << mlir;
    assert_iree_compile_accepts(mlir);
}

// JIT-R013 regression: handle_dropout must refuse to lower a
// training=true Dropout node (silent identity would drop the
// regularization mask/scale with no error), mirroring handle_batch_norm2d's
// established training-mode guard.
TEST(OpVision, DropoutTrainingModeRefusesToLower) {
    ensure_core_init();
    tzj::Graph g;
    auto x = g.create_value("x", {4}, ::tenzor::DType::Float32,
                            ::tenzor::Device::cpu());
    g.set_inputs({x});
    auto node = g.create_node(tzj::OpType::Dropout);
    node->add_input(x);
    node->set_bool_attr("training", true);
    auto z = g.create_value("z", {4}, ::tenzor::DType::Float32,
                            ::tenzor::Device::cpu());
    node->add_output(z);
    g.add_node(node);
    g.set_outputs({z});
    tzm::GraphToMLIR lowerer;
    EXPECT_THROW(lowerer.lower(g), std::runtime_error);
}

TEST(OpVision, PaddingEmitsAndParses) {
    ensure_core_init();
    // (4,) → (8,) with pad_left=2, pad_right=2.
    tzj::Graph g;
    auto x = g.create_value("x", {4}, ::tenzor::DType::Float32,
                            ::tenzor::Device::cpu());
    g.set_inputs({x});
    auto node = g.create_node(tzj::OpType::Padding);
    node->add_input(x);
    auto z = g.create_value("z", {8}, ::tenzor::DType::Float32,
                            ::tenzor::Device::cpu());
    node->add_output(z);
    node->set_vec_attr("padding", {2, 2});
    g.add_node(node);
    g.set_outputs({z});
    tzm::GraphToMLIR lowerer;
    auto mlir = lowerer.lower(g);
    EXPECT_NE(mlir.find("stablehlo.pad"), std::string::npos) << mlir;
    assert_iree_compile_accepts(mlir);
}

TEST(OpVision, InterpolateUpsample2x) {
    ensure_core_init();
    // (1, 3, 4, 4) → (1, 3, 8, 8) nearest, factor 2.
    tzj::Graph g;
    auto x = g.create_value("x", {1, 3, 4, 4}, ::tenzor::DType::Float32,
                            ::tenzor::Device::cpu());
    g.set_inputs({x});
    auto node = g.create_node(tzj::OpType::Interpolate);
    node->add_input(x);
    auto z = g.create_value("z", {1, 3, 8, 8}, ::tenzor::DType::Float32,
                            ::tenzor::Device::cpu());
    node->add_output(z);
    g.add_node(node);
    g.set_outputs({z});
    tzm::GraphToMLIR lowerer;
    auto mlir = lowerer.lower(g);
    EXPECT_NE(mlir.find("stablehlo.broadcast_in_dim"), std::string::npos)
        << mlir;
    EXPECT_NE(mlir.find("stablehlo.reshape"), std::string::npos) << mlir;
    assert_iree_compile_accepts(mlir);
}

// =====================================================================
// End-to-end @tz.jit tests for vision-family OpType handlers.
// MaxPool2d / AvgPool2d / AdaptiveAvgPool2d / Padding / Interpolate /
// Conv2d each reach the trace via a different surface: pooling layers
// post a TracedOp directly from src/nn/layers/pooling.cpp (Group E),
// Conv2d/Padding/Interpolate ride OpId dispatch and get mapped by the
// tracer interceptor.
// =====================================================================

#include "tenzor/autograd/variable.hpp"
#include "tenzor/jit/compile.hpp"
#include "tenzor/nn/functional.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/reduction.hpp"

namespace tzv_e2e {
namespace F = ::tenzor::nn::functional;

void run_jit_vs_eager(::tenzor::jit::CompiledFunction::FnType fn,
                      const ::tenzor::Variable& x,
                      float tol = 1e-4F) {
    namespace mt = ::tenzor::testing::mlir;
    auto eager = fn(x);
    auto eager_cpu = eager.tensor().to(::tenzor::Device::cpu())
                          .to(::tenzor::DType::Float32);
    // Fan out over every available IREE target so the conv/pool lowerings — incl.
    // the conv precision_config (JIT-F012) — are validated on GPU too (JIT-F028).
    const auto targets = mt::available_iree_targets();
    ASSERT_FALSE(targets.empty()) << "no IREE target available";
    for (const auto& target : targets) {
        ::tenzor::jit::CompileConfig cfg;
        cfg.backend = "mlir";
        cfg.target  = target;
        auto compiled = ::tenzor::jit::CompiledFunction(fn, cfg);
        mt::reset_jit_stats();
        auto jit = compiled(x);
        mt::assert_jit_used("vision", target);
        auto jit_cpu = jit.tensor().to(::tenzor::Device::cpu())
                          .to(::tenzor::DType::Float32);
        auto diff = ::tenzor::max(::tenzor::abs(eager_cpu - jit_cpu))
                        .template item<float>();
        EXPECT_LT(diff, tol) << "target=" << target << " diff=" << diff;
    }
}

}  // namespace tzv_e2e

TEST(OpVision, MaxPool2dEndToEnd) {
    ensure_core_init();
    auto fn = [](const ::tenzor::Variable& x) -> ::tenzor::Variable {
        return tzv_e2e::F::max_pool2d(x, {2, 2}, {2, 2}, {0, 0});
    };
    auto x_t = ::tenzor::full({1, 1, 4, 4}, 0.0F, ::tenzor::DType::Float32);
    auto* xd = x_t.data<float>();
    for (int i = 0; i < 16; ++i) xd[i] = static_cast<float>(i);
    ::tenzor::Variable x(x_t, false);
    tzv_e2e::run_jit_vs_eager(fn, x);
}

TEST(OpVision, AvgPool2dEndToEnd) {
    ensure_core_init();
    auto fn = [](const ::tenzor::Variable& x) -> ::tenzor::Variable {
        return tzv_e2e::F::avg_pool2d(x, {2, 2}, {2, 2}, {0, 0});
    };
    auto x_t = ::tenzor::full({1, 1, 4, 4}, 0.0F, ::tenzor::DType::Float32);
    auto* xd = x_t.data<float>();
    for (int i = 0; i < 16; ++i) xd[i] = static_cast<float>(i);
    ::tenzor::Variable x(x_t, false);
    tzv_e2e::run_jit_vs_eager(fn, x);
}

TEST(OpVision, AdaptiveAvgPool2dEndToEnd) {
    ensure_core_init();
    auto fn = [](const ::tenzor::Variable& x) -> ::tenzor::Variable {
        return tzv_e2e::F::adaptive_avg_pool2d(x, {1, 1});
    };
    auto x_t = ::tenzor::full({1, 2, 4, 4}, 0.0F, ::tenzor::DType::Float32);
    auto* xd = x_t.data<float>();
    for (int i = 0; i < 32; ++i) xd[i] = static_cast<float>(i);
    ::tenzor::Variable x(x_t, false);
    tzv_e2e::run_jit_vs_eager(fn, x);
}

TEST(OpVision, PaddingEndToEnd) {
    ensure_core_init();
    // F::pad constant mode is implemented as a sequence of Cat ops
    // (each leg dispatches OpId::Cat). The JIT therefore exercises
    // OpType::Cat repeatedly — no dedicated OpType::Padding node is
    // emitted, but the lowering of the cat-decomposition is what
    // real models will hit when they pad inputs.
    auto fn = [](const ::tenzor::Variable& x) -> ::tenzor::Variable {
        return tzv_e2e::F::pad(x, /*pad=*/{1, 1, 1, 1},
                               /*mode=*/"constant", /*value=*/0.0);
    };
    auto x_t = ::tenzor::full({1, 1, 2, 2}, 1.0F, ::tenzor::DType::Float32);
    ::tenzor::Variable x(x_t, false);
    tzv_e2e::run_jit_vs_eager(fn, x);
}

TEST(OpVision, InterpolateEndToEnd) {
    ensure_core_init();
    // 2x upsample of a 2x2 -> 4x4 feature map. The lowering implements
    // integer-ratio nearest as reshape + broadcast + reshape. Note
    // F::interpolate's "nearest" mode dispatches OpId::Interpolate
    // which the tracer now maps to OpType::Interpolate.
    auto fn = [](const ::tenzor::Variable& x) -> ::tenzor::Variable {
        return tzv_e2e::F::interpolate(x, std::pair<int64_t, int64_t>{4, 4},
                                       "nearest", /*align_corners=*/false);
    };
    auto x_t = ::tenzor::full({1, 1, 2, 2}, 0.0F, ::tenzor::DType::Float32);
    auto* xd = x_t.data<float>();
    xd[0] = 1.0F; xd[1] = 2.0F; xd[2] = 3.0F; xd[3] = 4.0F;
    ::tenzor::Variable x(x_t, false);
    tzv_e2e::run_jit_vs_eager(fn, x);
}

TEST(OpVision, Conv2dEndToEnd) {
    ensure_core_init();
    // 3x3 identity-ish kernel on a [1,1,4,4] feature map, stride 1, no pad.
    auto fn = [](const ::tenzor::Variable& x) -> ::tenzor::Variable {
        // Constant weight Variable (no grad) inside the lambda — full()
        // dispatches OpId::Full which the tracer leaves as a frozen
        // graph constant. That's the right shape for an inference-style
        // JIT lower where weights are baked in.
        auto w_t = ::tenzor::full({1, 1, 3, 3}, 1.0F, ::tenzor::DType::Float32);
        ::tenzor::Variable w(w_t, false);
        return tzv_e2e::F::conv2d(x, w, std::nullopt,
                                  /*stride=*/{1, 1}, /*padding=*/{0, 0});
    };
    auto x_t = ::tenzor::full({1, 1, 4, 4}, 0.0F, ::tenzor::DType::Float32);
    auto* xd = x_t.data<float>();
    for (int i = 0; i < 16; ++i) xd[i] = static_cast<float>(i);
    ::tenzor::Variable x(x_t, false);
    // Conv2d through IREE's compile pipeline can pick up ULP-scale drift
    // compared to oneDNN; widen tolerance slightly.
    tzv_e2e::run_jit_vs_eager(fn, x, 5e-3F);
}

TEST(OpVision, Conv2dOutput1x1ImcolPath) {
    ensure_core_init();
    // input 2x2, kernel 3x3, stride 2, pad 1 -> output 1x1. This strided conv
    // (non-exact division collapsing the output to size 1) is the geometry
    // IREE's stablehlo.convolution -> linalg conversion mis-lowers; the handler
    // must route it through the im2col (reshape + dot_general) path. Verify the
    // JIT output matches eager.
    // Fixed weight captured OUTSIDE the lambda so eager and JIT see the SAME
    // weight (randn inside the lambda would draw a fresh weight per call).
    ::tenzor::Variable w(::tenzor::randn({8, 4, 3, 3}, ::tenzor::DType::Float32),
                         false);
    auto fn = [w](const ::tenzor::Variable& x) -> ::tenzor::Variable {
        return tzv_e2e::F::conv2d(x, w, std::nullopt,
                                  /*stride=*/{2, 2}, /*padding=*/{1, 1});
    };
    auto x_t = ::tenzor::randn({1, 4, 2, 2}, ::tenzor::DType::Float32);
    ::tenzor::Variable x(x_t, false);
    tzv_e2e::run_jit_vs_eager(fn, x, 5e-3F);
}

TEST(OpVision, Conv2dPartialCollapsePadSlicePath) {
    ensure_core_init();
    // input 2x8, kernel 3x3, stride 2, pad 1 -> output 1x4: H collapses to 1
    // with an unused padded tail (the degenerate case) while W does not. The
    // handler must trim the tail and emit an exact-division convolution.
    ::tenzor::Variable w(::tenzor::randn({8, 4, 3, 3}, ::tenzor::DType::Float32),
                         false);
    auto fn = [w](const ::tenzor::Variable& x) -> ::tenzor::Variable {
        return tzv_e2e::F::conv2d(x, w, std::nullopt,
                                  /*stride=*/{2, 2}, /*padding=*/{1, 1});
    };
    auto x_t = ::tenzor::randn({1, 4, 2, 8}, ::tenzor::DType::Float32);
    ::tenzor::Variable x(x_t, false);
    tzv_e2e::run_jit_vs_eager(fn, x, 5e-3F);
}

// Regression (JIT review Fix #4): handle_max_pool2d emitted a 32-bit f32 -inf
// init literal (0xFF800000) for every float dtype except Float64, so a Float16 /
// BFloat16 max-pool produced `stablehlo.constant dense<0xFF800000> : tensor<f16>`
// — a hex width mismatch that iree-compile rejects on ALL targets. These must now
// compile (width-correct 0xFC00 / 0xFF80, matching handle_max) and match eager.
TEST(OpVision, MaxPool2dFloat16CompilesAndMatches) {
    ensure_core_init();
    auto fn = [](const ::tenzor::Variable& x) -> ::tenzor::Variable {
        return tzv_e2e::F::max_pool2d(x, {2, 2}, {2, 2}, {0, 0});
    };
    auto x32 = ::tenzor::full({1, 1, 4, 4}, 0.0F, ::tenzor::DType::Float32);
    auto* xd = x32.data<float>();
    for (int i = 0; i < 16; ++i) xd[i] = static_cast<float>(i);
    ::tenzor::Variable x(x32.to(::tenzor::DType::Float16), false);
    tzv_e2e::run_jit_vs_eager(fn, x, 1e-2F);
}

TEST(OpVision, MaxPool2dBFloat16CompilesAndMatches) {
    ensure_core_init();
    auto fn = [](const ::tenzor::Variable& x) -> ::tenzor::Variable {
        return tzv_e2e::F::max_pool2d(x, {2, 2}, {2, 2}, {0, 0});
    };
    auto x32 = ::tenzor::full({1, 1, 4, 4}, 0.0F, ::tenzor::DType::Float32);
    auto* xd = x32.data<float>();
    for (int i = 0; i < 16; ++i) xd[i] = static_cast<float>(i);
    ::tenzor::Variable x(x32.to(::tenzor::DType::BFloat16), false);
    tzv_e2e::run_jit_vs_eager(fn, x, 1e-2F);
}

// JIT-F035/F012: F16 Conv2d through the JIT across every available IREE target
// (llvm-cpu / cuda / vulkan-spirv). The eager reference and inputs are on CPU
// (safe), and the conv lowering must widen F16->F32 to match eager.
TEST(OpVision, Conv2dF16EndToEnd) {
    ensure_core_init();
    auto w = ::tenzor::randn({4, 3, 3, 3}, ::tenzor::DType::Float32)
                 .to(::tenzor::DType::Float16);
    ::tenzor::Variable wv(w, false);
    auto fn = [wv](const ::tenzor::Variable& x) -> ::tenzor::Variable {
        return tzv_e2e::F::conv2d(x, wv, std::nullopt, {1, 1}, {1, 1});
    };
    auto x_t = ::tenzor::randn({1, 3, 8, 8}, ::tenzor::DType::Float32)
                   .to(::tenzor::DType::Float16);
    ::tenzor::Variable x(x_t, /*requires_grad=*/false);
    tzv_e2e::run_jit_vs_eager(fn, x, /*tol=*/6e-2F);
}
