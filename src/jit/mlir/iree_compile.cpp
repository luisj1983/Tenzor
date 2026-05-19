// Phase 13 / Task A.7 — IREE Compiler embedding API integration.

#include "tenzor/jit/mlir/iree_compile.hpp"

#include <iree/compiler/embedding_api.h>

#include <openssl/sha.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

namespace tenzor::jit::mlir_jit {

namespace fs = std::filesystem;

namespace {

// One-time global init/shutdown of the IREE compiler API. The embedding API
// requires ireeCompilerGlobalInitialize before any other call; shutdown is
// registered to run at exit so multiple compile_mlir calls share state.
auto ensure_global_init() -> void {
    static std::once_flag flag;
    std::call_once(flag, []() {
        ireeCompilerGlobalInitialize();
        std::atexit([]() { ireeCompilerGlobalShutdown(); });
    });
}

auto hex_encode(const unsigned char* bytes, std::size_t n) -> std::string {
    std::ostringstream os;
    os << std::hex << std::setfill('0');
    for (std::size_t i = 0; i < n; ++i) {
        os << std::setw(2) << static_cast<unsigned>(bytes[i]);
    }
    return os.str();
}

auto sha256_hex(const std::string& data) -> std::string {
    unsigned char digest[SHA256_DIGEST_LENGTH];
    SHA256(reinterpret_cast<const unsigned char*>(data.data()), data.size(),
           digest);
    return hex_encode(digest, SHA256_DIGEST_LENGTH);
}

auto default_cache_dir() -> fs::path {
    if (const char* xdg = std::getenv("XDG_CACHE_HOME"); xdg && *xdg) {
        return fs::path(xdg) / "tenzor" / "jit_mlir";
    }
    if (const char* home = std::getenv("HOME"); home && *home) {
        return fs::path(home) / ".cache" / "tenzor" / "jit_mlir";
    }
    return fs::temp_directory_path() / "tenzor" / "jit_mlir";
}

/// RAII wrapper for the IREE compiler error type so `throw` doesn't leak.
struct CompilerError {
    iree_compiler_error_t* ptr = nullptr;
    ~CompilerError() {
        if (ptr) {
            ireeCompilerErrorDestroy(ptr);
        }
    }
    auto message() const -> std::string {
        if (!ptr) {
            return {};
        }
        const char* m = ireeCompilerErrorGetMessage(ptr);
        return m ? std::string(m) : std::string{};
    }
};

}  // namespace

auto iree_compiler_version() -> std::string {
    ensure_global_init();
    const char* rev = ireeCompilerGetRevision();
    std::string rev_s = rev ? std::string(rev) : std::string{};
    // Some IREE builds leave the revision empty. Fold in the embedding API
    // version so the cache key still rotates between IREE upgrades, and so
    // that callers can rely on a non-empty identifier.
    std::ostringstream os;
    os << "iree-api-v" << ireeCompilerGetAPIVersion();
    if (!rev_s.empty()) {
        os << '+' << rev_s;
    }
    return os.str();
}

auto compute_cache_key(const std::string& mlir_text,
                       const std::string& target) -> std::string {
    std::ostringstream os;
    os << "ver:" << iree_compiler_version() << '|'
       << "tgt:" << target << '|'
       << "src:" << mlir_text;
    return sha256_hex(os.str());
}

auto compile_mlir(const std::string& mlir_text,
                  const CompileOptions& opts) -> CompiledArtifact {
    ensure_global_init();

    if (opts.target.empty()) {
        throw JitCompileError(
            "compile_mlir: CompileOptions::target is empty", {});
    }

    const fs::path cache_dir =
        opts.cache_dir.empty() ? default_cache_dir() : opts.cache_dir;
    fs::create_directories(cache_dir);

    const std::string key = compute_cache_key(mlir_text, opts.target);
    const fs::path vmfb_path = cache_dir / (key + ".vmfb");
    const fs::path mlir_path = cache_dir / (key + ".mlir");

    // Always dump the source for offline debug, even on cache hit, so that the
    // path returned in JitCompileError exists.
    {
        std::ofstream f(mlir_path, std::ios::binary | std::ios::trunc);
        f.write(mlir_text.data(),
                static_cast<std::streamsize>(mlir_text.size()));
    }

    if (fs::exists(vmfb_path) && fs::file_size(vmfb_path) > 0) {
        return CompiledArtifact{vmfb_path, opts.target};
    }

    iree_compiler_session_t* session = ireeCompilerSessionCreate();
    if (session == nullptr) {
        throw JitCompileError(
            "ireeCompilerSessionCreate returned nullptr", mlir_path);
    }
    struct SessionGuard {
        iree_compiler_session_t* s;
        ~SessionGuard() { ireeCompilerSessionDestroy(s); }
    } session_guard{session};

    // Configure target via the standard flag. Tenzor exposes a small enum
    // (target field) and we translate it 1:1 to IREE's --iree-hal-target-backends
    // string. Additional --iree-input-type=stablehlo signals the input dialect.
    const std::string target_flag =
        "--iree-hal-target-backends=" + opts.target;
    const std::string input_flag = "--iree-input-type=stablehlo";
    const char* flags[] = {target_flag.c_str(), input_flag.c_str()};
    if (auto* set_err = ireeCompilerSessionSetFlags(session, /*argc=*/2, flags);
        set_err != nullptr) {
        CompilerError err{set_err};
        throw JitCompileError(
            "ireeCompilerSessionSetFlags failed: " + err.message(), mlir_path);
    }

    iree_compiler_source_t* source = nullptr;
    if (auto* src_err = ireeCompilerSourceWrapBuffer(
            session, /*bufferName=*/"tenzor_jit_input.mlir",
            mlir_text.c_str(), mlir_text.size(),
            /*isNullTerminated=*/false, &source);
        src_err != nullptr) {
        CompilerError err{src_err};
        throw JitCompileError(
            "ireeCompilerSourceWrapBuffer failed: " + err.message(),
            mlir_path);
    }
    struct SourceGuard {
        iree_compiler_source_t* s;
        ~SourceGuard() { ireeCompilerSourceDestroy(s); }
    } source_guard{source};

    iree_compiler_invocation_t* inv = ireeCompilerInvocationCreate(session);
    if (inv == nullptr) {
        throw JitCompileError(
            "ireeCompilerInvocationCreate returned nullptr", mlir_path);
    }
    struct InvGuard {
        iree_compiler_invocation_t* i;
        ~InvGuard() { ireeCompilerInvocationDestroy(i); }
    } inv_guard{inv};

    // Capture diagnostics into a buffer so failures surface as JitCompileError
    // messages rather than going to stderr silently.
    std::string diag_buffer;
    struct DiagCtx {
        std::string* out;
    } diag_ctx{&diag_buffer};
    ireeCompilerInvocationEnableCallbackDiagnostics(
        inv, /*flags=*/0,
        [](enum iree_compiler_diagnostic_severity_t /*severity*/,
           const char* message, size_t message_size, void* user_data) {
            auto* ctx = static_cast<DiagCtx*>(user_data);
            ctx->out->append(message, message_size);
            ctx->out->push_back('\n');
        },
        &diag_ctx);

    if (!ireeCompilerInvocationParseSource(inv, source)) {
        throw JitCompileError(
            "ireeCompilerInvocationParseSource failed:\n" + diag_buffer,
            mlir_path);
    }

    if (!ireeCompilerInvocationPipeline(inv, IREE_COMPILER_PIPELINE_STD)) {
        throw JitCompileError(
            "ireeCompilerInvocationPipeline(STD) failed:\n" + diag_buffer,
            mlir_path);
    }

    iree_compiler_output_t* output = nullptr;
    const std::string vmfb_path_str = vmfb_path.string();
    if (auto* out_err = ireeCompilerOutputOpenFile(vmfb_path_str.c_str(),
                                                   &output);
        out_err != nullptr) {
        CompilerError err{out_err};
        throw JitCompileError(
            "ireeCompilerOutputOpenFile(" + vmfb_path_str +
                ") failed: " + err.message(),
            mlir_path);
    }
    struct OutputGuard {
        iree_compiler_output_t* o;
        ~OutputGuard() { ireeCompilerOutputDestroy(o); }
    } output_guard{output};

    if (auto* vmb_err =
            ireeCompilerInvocationOutputVMBytecode(inv, output);
        vmb_err != nullptr) {
        CompilerError err{vmb_err};
        throw JitCompileError(
            "ireeCompilerInvocationOutputVMBytecode failed: " + err.message() +
                "\nDiagnostics:\n" + diag_buffer,
            mlir_path);
    }
    ireeCompilerOutputKeep(output);

    return CompiledArtifact{vmfb_path, opts.target};
}

}  // namespace tenzor::jit::mlir_jit
