// Phase 13 / Task A.8 — IREE Runtime invoker tests.

#include "tenzor/jit/mlir/iree_compile.hpp"
#include "tenzor/jit/mlir/iree_runtime.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/tenzor.hpp"

#include <gtest/gtest.h>

#include <chrono>
#include <filesystem>
#include <random>
#include <string>

namespace tj = ::tenzor::jit::mlir_jit;
namespace fs = std::filesystem;

namespace {

auto ensure_core_init() -> void {
    static const bool inited = []() {
        ::tenzor::initialize();
        return true;
    }();
    (void)inited;
}

auto make_tmp_dir() -> fs::path {
    auto now = std::chrono::system_clock::now().time_since_epoch().count();
    std::mt19937_64 rng(static_cast<uint64_t>(now));
    fs::path dir =
        fs::temp_directory_path() /
        ("tenzor_jit_invoke_test_" + std::to_string(rng()));
    fs::create_directories(dir);
    return dir;
}

constexpr const char* kIdentityModule = R"MLIR(module {
  func.func @main(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    return %arg0 : tensor<4xf32>
  }
}
)MLIR";

constexpr const char* kAddModule = R"MLIR(module {
  func.func @main(%a: tensor<3xf32>, %b: tensor<3xf32>) -> tensor<3xf32> {
    %r = stablehlo.add %a, %b : tensor<3xf32>
    return %r : tensor<3xf32>
  }
}
)MLIR";

}  // namespace

TEST(IreeInvoke, Identity_Float32) {
    ensure_core_init();
    const fs::path tmp = make_tmp_dir();
    tj::CompileOptions opts;
    opts.target = "llvm-cpu";
    opts.cache_dir = tmp;

    auto artifact = tj::compile_mlir(kIdentityModule, opts);
    auto invoker  = tj::IreeInvoker::load(artifact);

    auto x = ::tenzor::full({4}, 3.14F);  // Float32
    auto outs = invoker->invoke({x});
    ASSERT_EQ(outs.size(), 1U);
    ASSERT_EQ(outs[0].dtype(), ::tenzor::DType::Float32);
    ASSERT_EQ(outs[0].numel(), 4);
    const float* p = outs[0].data<float>();
    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(p[i], 3.14F, 1e-5F) << "i=" << i;
    }

    fs::remove_all(tmp);
}

TEST(IreeInvoke, Add_Float32) {
    ensure_core_init();
    const fs::path tmp = make_tmp_dir();
    tj::CompileOptions opts;
    opts.target = "llvm-cpu";
    opts.cache_dir = tmp;

    auto artifact = tj::compile_mlir(kAddModule, opts);
    auto invoker  = tj::IreeInvoker::load(artifact);

    auto a = ::tenzor::full({3}, 1.5F);
    auto b = ::tenzor::full({3}, 0.25F);
    auto outs = invoker->invoke({a, b});
    ASSERT_EQ(outs.size(), 1U);
    ASSERT_EQ(outs[0].numel(), 3);
    const float* p = outs[0].data<float>();
    for (int i = 0; i < 3; ++i) {
        EXPECT_NEAR(p[i], 1.75F, 1e-5F);
    }

    fs::remove_all(tmp);
}
