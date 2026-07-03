// Phase 13 / Group C.7 — Norm ops via the MLIR backend.
//
// Covered ops:
//   LayerNorm   → mean/var/scale/shift decomposition over stablehlo
//                 primitives (no fused stablehlo op)
//   BatchNorm2d → stablehlo.batch_norm_inference (inference path)
//
// Tenzor's nn::LayerNorm / nn::BatchNorm2d dispatch through fused-op
// IDs that aren't in the OpType mapping, so a trace doesn't surface
// OpType::LayerNorm / BatchNorm2d. These tests therefore build graphs
// by hand and validate the emitted MLIR via iree-compile.

#include "tenzor/jit/graph.hpp"
#include "tenzor/jit/mlir/lowering.hpp"
#include "tenzor/jit/mlir/iree_compile.hpp"
#include "tenzor/jit/mlir/iree_paths.hpp"
#include "tenzor/jit/tracer.hpp"
#include "tenzor/tenzor.hpp"

#include <gtest/gtest.h>

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
               ("tenzor_op_norms_" + std::to_string(::getpid()) + "_" +
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

TEST(OpNorms, LayerNormEmitsAndParses) {
    ensure_core_init();
    // x: (4, 8), normalize last 1 dim. weight & bias: (8,)
    tzj::Graph g;
    auto x = g.create_value("x", {4, 8}, ::tenzor::DType::Float32,
                            ::tenzor::Device::cpu());
    auto w = g.create_value("w", {8}, ::tenzor::DType::Float32,
                            ::tenzor::Device::cpu());
    auto b = g.create_value("b", {8}, ::tenzor::DType::Float32,
                            ::tenzor::Device::cpu());
    g.set_inputs({x, w, b});
    auto node = g.create_node(tzj::OpType::LayerNorm);
    node->add_input(x);
    node->add_input(w);
    node->add_input(b);
    auto z = g.create_value("z", {4, 8}, ::tenzor::DType::Float32,
                            ::tenzor::Device::cpu());
    node->add_output(z);
    node->set_vec_attr("normalized_shape", {8});
    node->set_attr("eps", 1e-5f);
    g.add_node(node);
    g.set_outputs({z});
    tzm::GraphToMLIR lowerer;
    auto mlir = lowerer.lower(g);
    EXPECT_NE(mlir.find("stablehlo.reduce"), std::string::npos) << mlir;
    EXPECT_NE(mlir.find("stablehlo.sqrt"),   std::string::npos) << mlir;
    EXPECT_NE(mlir.find("stablehlo.divide"), std::string::npos) << mlir;
    assert_iree_compile_accepts(mlir);
}

TEST(OpNorms, BatchNorm2dEmitsAndParses) {
    ensure_core_init();
    // x: (N=2, C=3, H=4, W=4); weight/bias/running_mean/running_var: (3,)
    tzj::Graph g;
    auto x  = g.create_value("x",  {2, 3, 4, 4}, ::tenzor::DType::Float32,
                             ::tenzor::Device::cpu());
    auto w  = g.create_value("w",  {3}, ::tenzor::DType::Float32,
                             ::tenzor::Device::cpu());
    auto b  = g.create_value("b",  {3}, ::tenzor::DType::Float32,
                             ::tenzor::Device::cpu());
    auto rm = g.create_value("rm", {3}, ::tenzor::DType::Float32,
                             ::tenzor::Device::cpu());
    auto rv = g.create_value("rv", {3}, ::tenzor::DType::Float32,
                             ::tenzor::Device::cpu());
    g.set_inputs({x, w, b, rm, rv});
    auto node = g.create_node(tzj::OpType::BatchNorm2d);
    node->add_input(x);
    node->add_input(w);
    node->add_input(b);
    node->add_input(rm);
    node->add_input(rv);
    auto z = g.create_value("z", {2, 3, 4, 4}, ::tenzor::DType::Float32,
                            ::tenzor::Device::cpu());
    node->add_output(z);
    node->set_attr("eps", 1e-5f);
    g.add_node(node);
    g.set_outputs({z});
    tzm::GraphToMLIR lowerer;
    auto mlir = lowerer.lower(g);
    EXPECT_NE(mlir.find("stablehlo.batch_norm_inference"),
              std::string::npos) << mlir;
    assert_iree_compile_accepts(mlir);
}

// =====================================================================
// End-to-end @tz.jit tests for BatchNorm2d + LayerNorm. Both ops
// dispatch through OpId on every device (the tracer's OpId -> OpType
// mapping in src/jit/tracing_interceptor.cpp handles them) so they
// don't need a side-channel `tracer.record_op` like the pooling layers.
// =====================================================================

#include "tenzor/autograd/variable.hpp"
#include "tenzor/jit/compile.hpp"
#include "tenzor/nn/functional.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/reduction.hpp"

namespace tzn_e2e {
namespace F = ::tenzor::nn::functional;

void run_jit_vs_eager(::tenzor::jit::CompiledFunction::FnType fn,
                      const ::tenzor::Variable& x,
                      float tol = 5e-4F) {
    ::tenzor::jit::CompileConfig cfg;
    cfg.backend = "mlir";
    cfg.target  = "llvm-cpu";
    auto compiled = ::tenzor::jit::CompiledFunction(fn, cfg);
    auto eager = fn(x);
    ::tenzor::jit::mlir_jit::reset_cache_stats();
    auto jit   = compiled(x);
    EXPECT_GE(::tenzor::jit::mlir_jit::cache_stats().misses, 1u)
        << "op did not run through IREE (silent eager fallback; llvm-cpu)";
    auto eager_cpu = eager.tensor().to(::tenzor::Device::cpu());
    auto jit_cpu   = jit.tensor().to(::tenzor::Device::cpu());
    auto diff = ::tenzor::max(::tenzor::abs(eager_cpu - jit_cpu))
                    .template item<float>();
    EXPECT_LT(diff, tol);
}

}  // namespace tzn_e2e

TEST(OpNorms, BatchNorm2dEndToEnd) {
    ensure_core_init();
    // Inference-mode BatchNorm: weights/bias and running stats are
    // constant tensors baked in at trace time (they dispatch
    // OpId::Full inside the lambda which the tracer drops as a
    // graph break and freezes as constants).
    auto fn = [](const ::tenzor::Variable& x) -> ::tenzor::Variable {
        const int64_t C = 2;
        auto rm  = ::tenzor::full({C}, 0.0F, ::tenzor::DType::Float32);
        auto rv  = ::tenzor::full({C}, 1.0F, ::tenzor::DType::Float32);
        auto w_t = ::tenzor::full({C}, 1.0F, ::tenzor::DType::Float32);
        auto b_t = ::tenzor::full({C}, 0.0F, ::tenzor::DType::Float32);
        ::tenzor::Variable w(w_t, false), b(b_t, false);
        return tzn_e2e::F::batch_norm(x, rm, rv, w, b,
                                      /*training=*/false,
                                      /*momentum=*/0.1,
                                      /*eps=*/1e-5);
    };
    auto x_t = ::tenzor::full({1, 2, 3, 3}, 0.0F, ::tenzor::DType::Float32);
    auto* xd = x_t.data<float>();
    for (int i = 0; i < 18; ++i) xd[i] = static_cast<float>(i) * 0.1F;
    ::tenzor::Variable x(x_t, false);
    tzn_e2e::run_jit_vs_eager(fn, x);
}

TEST(OpNorms, LayerNormEndToEnd) {
    ensure_core_init();
    auto fn = [](const ::tenzor::Variable& x) -> ::tenzor::Variable {
        std::vector<int64_t> normalized_shape = {4};
        auto w_t = ::tenzor::full({4}, 1.0F, ::tenzor::DType::Float32);
        auto b_t = ::tenzor::full({4}, 0.0F, ::tenzor::DType::Float32);
        ::tenzor::Variable w(w_t, false), b(b_t, false);
        return tzn_e2e::F::layer_norm(x, normalized_shape, w, b,
                                      /*eps=*/1e-5);
    };
    auto x_t = ::tenzor::full({2, 4}, 0.0F, ::tenzor::DType::Float32);
    auto* xd = x_t.data<float>();
    for (int i = 0; i < 8; ++i) xd[i] = static_cast<float>(i) * 0.3F + 0.1F;
    ::tenzor::Variable x(x_t, false);
    tzn_e2e::run_jit_vs_eager(fn, x);
}
