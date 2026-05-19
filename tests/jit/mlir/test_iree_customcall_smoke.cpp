// Phase 13 / Task A.9 — Custom-call smoke test.
//
// Verifies that an MLIR module which invokes one of the 4 MVP-1 tenzor_*
// custom_calls cannot yet be compiled to a vmfb without a real resolver,
// and that the failure message identifies the unresolved op. This is the
// same signal the runtime IREE_STATUS_UNIMPLEMENTED callback in
// iree_customcalls.cpp would emit if the runtime invocation path had a
// session to register against.

#include "tenzor/jit/mlir/iree_compile.hpp"
#include "tenzor/jit/mlir/iree_customcalls.hpp"

#include <gtest/gtest.h>

#include <chrono>
#include <filesystem>
#include <random>
#include <string>

namespace tj = ::tenzor::jit::mlir_jit;
namespace fs = std::filesystem;

namespace {

auto make_tmp_dir() -> fs::path {
    auto now = std::chrono::system_clock::now().time_since_epoch().count();
    std::mt19937_64 rng(static_cast<uint64_t>(now));
    fs::path dir =
        fs::temp_directory_path() /
        ("tenzor_jit_custom_call_test_" + std::to_string(rng()));
    fs::create_directories(dir);
    return dir;
}

constexpr const char* kFlashAttentionModule = R"MLIR(module {
  func.func @main(%q: tensor<2x4x8xf32>, %k: tensor<2x4x8xf32>, %v: tensor<2x4x8xf32>) -> tensor<2x4x8xf32> {
    %o = stablehlo.custom_call @tenzor_flash_attention(%q, %k, %v) {backend_config = "causal=1"} : (tensor<2x4x8xf32>, tensor<2x4x8xf32>, tensor<2x4x8xf32>) -> tensor<2x4x8xf32>
    return %o : tensor<2x4x8xf32>
  }
}
)MLIR";

}  // namespace

TEST(IreeCustomCallSmoke, FlashAttention_RejectedWithoutResolver) {
    const fs::path tmp = make_tmp_dir();
    tj::CompileOptions opts;
    opts.target = "llvm-cpu";
    opts.cache_dir = tmp;

    try {
        (void)tj::compile_mlir(kFlashAttentionModule, opts);
        FAIL() << "Expected JitCompileError — tenzor_flash_attention has no "
                  "resolver yet";
    } catch (const tj::JitCompileError& e) {
        const std::string msg = e.what();
        // The IREE compiler error message names the unresolved op.
        EXPECT_NE(msg.find("tenzor_flash_attention"), std::string::npos)
            << "expected the diagnostic to mention tenzor_flash_attention, "
               "got:\n" << msg;
        EXPECT_NE(msg.find("stablehlo.custom_call"), std::string::npos)
            << "expected the diagnostic to mention stablehlo.custom_call, "
               "got:\n" << msg;
    }

    fs::remove_all(tmp);
}

TEST(IreeCustomCallSmoke, PlaceholderMessages_AllPresent) {
    using namespace ::tenzor::jit::mlir_jit::placeholder_messages;
    const std::string fa = flash_attention();
    const std::string gqa_msg = gqa();
    const std::string rope = rope_apply();
    const std::string rms = rms_norm();

    // Each must mention Group D as the resolution path, matching the
    // structure spec'd for tenzor_*_callback in iree_customcalls.cpp.
    EXPECT_NE(fa.find("Group D"), std::string::npos);
    EXPECT_NE(gqa_msg.find("Group D"), std::string::npos);
    EXPECT_NE(rope.find("Group D"), std::string::npos);
    EXPECT_NE(rms.find("Group D"), std::string::npos);

    EXPECT_NE(fa.find("flash_attention"), std::string::npos);
    EXPECT_NE(gqa_msg.find("gqa"), std::string::npos);
    EXPECT_NE(rope.find("rope_apply"), std::string::npos);
    EXPECT_NE(rms.find("rms_norm"), std::string::npos);
}
