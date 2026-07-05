// Phase 13 / Group C.6 — Cast / Index ops via the MLIR backend.
//
// Covered ops:
//   Cast        → stablehlo.convert
//   Embedding   → stablehlo.gather
//   IndexSelect → stablehlo.gather
//
// Cast on CPU rewrites in Tensor::to(dtype) without dispatching, so we
// test it via a manually-built graph round-tripped through iree-compile.
// Embedding and IndexSelect do dispatch on CPU, so we can run end-to-end
// equivalence checks against eager.

#include "tenzor/autograd/ops.hpp"
#include "tenzor/autograd/variable.hpp"
#include "tenzor/backend/loader.hpp"
#include "tenzor/jit/compile.hpp"
#include "tenzor/jit/graph.hpp"
#include "tenzor/jit/mlir/iree_compile.hpp"
#include "tenzor/jit/mlir/lowering.hpp"
#include "tenzor/jit/mlir/iree_paths.hpp"
#include "tenzor/jit/tracer.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/reduction.hpp"
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
               ("tenzor_op_castindex_" + std::to_string(::getpid()) + "_" +
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

TEST(OpCastIndex, CastEmitsAndParses) {
    ensure_core_init();
    tzj::Graph g;
    auto x = g.create_value("x", {4, 8}, ::tenzor::DType::Float32,
                            ::tenzor::Device::cpu());
    g.set_inputs({x});
    auto node = g.create_node(tzj::OpType::Cast);
    node->add_input(x);
    auto z = g.create_value("z", {4, 8}, ::tenzor::DType::Float64,
                            ::tenzor::Device::cpu());
    node->add_output(z);
    g.add_node(node);
    g.set_outputs({z});
    tzm::GraphToMLIR lowerer;
    auto mlir = lowerer.lower(g);
    EXPECT_NE(mlir.find("stablehlo.convert"), std::string::npos) << mlir;
    assert_iree_compile_accepts(mlir);
}

TEST(OpCastIndex, EmbeddingEmitsAndParses) {
    ensure_core_init();
    // weight: (V=10, E=4), indices: (3,), out: (3, 4)
    tzj::Graph g;
    auto w = g.create_value("w", {10, 4}, ::tenzor::DType::Float32,
                            ::tenzor::Device::cpu());
    auto i = g.create_value("i", {3}, ::tenzor::DType::Int64,
                            ::tenzor::Device::cpu());
    g.set_inputs({w, i});
    auto node = g.create_node(tzj::OpType::Embedding);
    node->add_input(w);
    node->add_input(i);
    auto z = g.create_value("z", {3, 4}, ::tenzor::DType::Float32,
                            ::tenzor::Device::cpu());
    node->add_output(z);
    g.add_node(node);
    g.set_outputs({z});
    tzm::GraphToMLIR lowerer;
    auto mlir = lowerer.lower(g);
    EXPECT_NE(mlir.find("stablehlo.gather"), std::string::npos) << mlir;
    assert_iree_compile_accepts(mlir);
}

TEST(OpCastIndex, IndexSelectEmitsAndParses) {
    ensure_core_init();
    // x: (5, 4), index: (3,), dim=0, out: (3, 4)
    tzj::Graph g;
    auto x = g.create_value("x", {5, 4}, ::tenzor::DType::Float32,
                            ::tenzor::Device::cpu());
    auto i = g.create_value("i", {3}, ::tenzor::DType::Int64,
                            ::tenzor::Device::cpu());
    g.set_inputs({x, i});
    auto node = g.create_node(tzj::OpType::IndexSelect);
    node->add_input(x);
    node->add_input(i);
    auto z = g.create_value("z", {3, 4}, ::tenzor::DType::Float32,
                            ::tenzor::Device::cpu());
    node->add_output(z);
    node->set_int_attr("dim", 0);
    g.add_node(node);
    g.set_outputs({z});
    tzm::GraphToMLIR lowerer;
    auto mlir = lowerer.lower(g);
    EXPECT_NE(mlir.find("stablehlo.gather"), std::string::npos) << mlir;
    assert_iree_compile_accepts(mlir);
}

// =====================================================================
// End-to-end tests for OpType::IndexSelect via @tz.jit. `tenzor::
// index_select` dispatches OpId::IndexSelect on all devices so the
// tracer's OpId -> OpType mapping is sufficient — no nn-layer side-
// channel record needed.
// =====================================================================

#include "tenzor/autograd/ops.hpp"
#include "tenzor/jit/compile.hpp"
#include "tenzor/ops/indexing.hpp"

TEST(OpCastIndex, IndexSelectEndToEnd) {
    ensure_core_init();
    auto fn = [](const ::tenzor::Variable& x) -> ::tenzor::Variable {
        // Pick rows [1, 3, 2] from x. Use Int64 indices (the dtype the
        // CPU index_select kernel and StableHLO gather both expect).
        auto idx_t = ::tenzor::full({3}, 1.0F, ::tenzor::DType::Int64);
        auto idx_data = idx_t.data<int64_t>();
        idx_data[0] = 1; idx_data[1] = 3; idx_data[2] = 2;
        return ::tenzor::index_select(x, /*dim=*/0, idx_t);
    };

    // Build a [5,4] input where each row has distinct values so an
    // index permutation is observable, then JIT and compare to eager.
    auto x_t = ::tenzor::full({5, 4}, 0.0F, ::tenzor::DType::Float32);
    auto* xd = x_t.data<float>();
    for (int r = 0; r < 5; ++r)
        for (int c = 0; c < 4; ++c) xd[r * 4 + c] = static_cast<float>(r) + 0.1F * static_cast<float>(c);

    ::tenzor::Variable x(x_t, /*requires_grad=*/false);

    auto eager = fn(x);
    auto eager_cpu = eager.tensor().to(::tenzor::Device::cpu());
    // Fan out over every available IREE target (JIT-F028).
    namespace mt = ::tenzor::testing::mlir;
    const auto targets = mt::available_iree_targets();
    ASSERT_FALSE(targets.empty()) << "no IREE target available";
    for (const auto& target : targets) {
        ::tenzor::jit::CompileConfig cfg;
        cfg.backend = "mlir";
        cfg.target  = target;
        auto compiled = ::tenzor::jit::CompiledFunction(fn, cfg);
        mt::reset_jit_stats();
        auto jit = compiled(x);
        mt::assert_jit_used("index_select", target);
        auto jit_cpu = jit.tensor().to(::tenzor::Device::cpu());
        auto diff = ::tenzor::max(::tenzor::abs(
            eager_cpu.to(::tenzor::DType::Float32) -
            jit_cpu.to(::tenzor::DType::Float32))).template item<float>();
        EXPECT_LT(diff, 1e-5F) << "target=" << target << " diff=" << diff;
    }
}
