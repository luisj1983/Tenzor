// Phase 13 / Task A.7 — IREE Compiler embedding API integration.
//
// The primary path uses IREE's in-process embedding API (fast, no fork).
// For HAL targets the linked libIREECompiler.so was not built with (commonly
// `cuda` / `rocm` on lite distributions), compile_mlir() transparently falls
// through to a subprocess invocation of `iree-compile` discovered via the
// iree_paths.hpp chain — typically the pip-installed `iree-base-compiler`
// wheel which always ships every HAL target.

#include "tenzor/jit/mlir/iree_compile.hpp"
#include "tenzor/jit/mlir/iree_paths.hpp"

#include <iree/compiler/embedding_api.h>

#include <openssl/sha.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include <fcntl.h>
#include <sys/wait.h>
#include <unistd.h>

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

// ============================================================================
// Embedding-API target probe.
// ============================================================================

/// Enumerate the HAL target backends the *linked* libIREECompiler supports
/// via the embedding API itself. This is distinct from
/// iree_compile_supported_targets() (which probes the binary on disk) — the
/// in-process shared library may have been built with a smaller subset.
auto embedding_api_supported_targets() -> const std::set<std::string>& {
    static const std::set<std::string> cached = []() {
        ensure_global_init();
        std::set<std::string> out;
        struct Ctx {
            std::set<std::string>* out;
        } ctx{&out};
        ireeCompilerEnumerateRegisteredHALTargetBackends(
            [](const char* backend, void* user_data) {
                auto* c = static_cast<Ctx*>(user_data);
                if (backend != nullptr) {
                    c->out->insert(backend);
                }
            },
            &ctx);
        return out;
    }();
    return cached;
}

// ============================================================================
// Subprocess fallback: iree-compile <mlir> --output-format=vm-bytecode
// ============================================================================

/// Spawn an iree-compile subprocess with the given args, write `mlir_text`
/// to its stdin, and read the compiled vmfb bytes from its stdout. Returns
/// the captured stderr in `stderr_out`.
auto run_iree_compile_subprocess(const std::string& binary,
                                 const std::vector<std::string>& args,
                                 const std::string& mlir_text,
                                 std::string& stderr_out,
                                 std::string& vmfb_out) -> int {
    int in_pipe[2];   // parent -> child stdin
    int out_pipe[2];  // child stdout -> parent
    int err_pipe[2];  // child stderr -> parent
    if (pipe(in_pipe) != 0 || pipe(out_pipe) != 0 || pipe(err_pipe) != 0) {
        throw JitCompileError(
            std::string("pipe() failed: ") + std::strerror(errno), {});
    }

    pid_t pid = fork();
    if (pid < 0) {
        for (int fd : {in_pipe[0], in_pipe[1], out_pipe[0], out_pipe[1],
                       err_pipe[0], err_pipe[1]}) {
            close(fd);
        }
        throw JitCompileError(
            std::string("fork() failed: ") + std::strerror(errno), {});
    }

    if (pid == 0) {
        // Child: wire up stdio and exec.
        dup2(in_pipe[0],  STDIN_FILENO);
        dup2(out_pipe[1], STDOUT_FILENO);
        dup2(err_pipe[1], STDERR_FILENO);
        close(in_pipe[0]);  close(in_pipe[1]);
        close(out_pipe[0]); close(out_pipe[1]);
        close(err_pipe[0]); close(err_pipe[1]);
        std::vector<char*> argv;
        argv.reserve(args.size() + 2);
        argv.push_back(const_cast<char*>(binary.c_str()));
        for (const auto& a : args) {
            argv.push_back(const_cast<char*>(a.c_str()));
        }
        argv.push_back(nullptr);
        execv(binary.c_str(), argv.data());
        // execv failed.
        const std::string msg =
            std::string("execv failed: ") + std::strerror(errno) + "\n";
        (void)!write(STDERR_FILENO, msg.data(), msg.size());
        _exit(127);
    }

    // Parent.
    close(in_pipe[0]);
    close(out_pipe[1]);
    close(err_pipe[1]);

    // Write the MLIR text to the child stdin. Doing the write before reading
    // stdout/stderr could deadlock if the child's stdout buffer fills before
    // it finishes consuming stdin; for typical Phase-1A graphs the MLIR is a
    // few KB so a single write fits comfortably, but for safety we still
    // handle short writes.
    {
        std::size_t total = 0;
        while (total < mlir_text.size()) {
            const ssize_t n = write(in_pipe[1], mlir_text.data() + total,
                                    mlir_text.size() - total);
            if (n < 0) {
                if (errno == EINTR) continue;
                break;
            }
            total += static_cast<std::size_t>(n);
        }
    }
    close(in_pipe[1]);

    auto drain = [](int fd) {
        std::string s;
        char buf[8192];
        while (true) {
            const ssize_t r = read(fd, buf, sizeof(buf));
            if (r > 0) {
                s.append(buf, static_cast<std::size_t>(r));
            } else if (r == 0) {
                break;
            } else {
                if (errno == EINTR) continue;
                break;
            }
        }
        return s;
    };

    vmfb_out   = drain(out_pipe[0]);
    stderr_out = drain(err_pipe[0]);
    close(out_pipe[0]);
    close(err_pipe[0]);

    int status = 0;
    while (waitpid(pid, &status, 0) < 0) {
        if (errno != EINTR) {
            throw JitCompileError(
                std::string("waitpid() failed: ") + std::strerror(errno), {});
        }
    }
    if (WIFEXITED(status))   return WEXITSTATUS(status);
    if (WIFSIGNALED(status)) return 128 + WTERMSIG(status);
    return -1;
}

/// Compile via the iree-compile subprocess for targets the linked
/// libIREECompiler.so does not register. Writes the vmfb to `vmfb_path`.
auto compile_via_subprocess(const std::string& mlir_text,
                            const std::string& target,
                            const fs::path& vmfb_path,
                            const fs::path& mlir_path) -> void {
    const std::string& bin = ::tenzor::jit::mlir_jit::resolve_iree_compile();
    std::vector<std::string> args{
        "--iree-hal-target-backends=" + target,
        "--iree-input-type=stablehlo",
        "--output-format=vm-bytecode",
        "-o", vmfb_path.string(),
        "-",  // read MLIR from stdin
    };
    std::string stderr_text, vmfb_unused;
    const int rc = run_iree_compile_subprocess(bin, args, mlir_text,
                                               stderr_text, vmfb_unused);
    if (rc != 0) {
        throw JitCompileError(
            "iree-compile subprocess (target=" + target + ") exit=" +
                std::to_string(rc) + " from " + bin + "\nstderr:\n" +
                stderr_text,
            mlir_path);
    }
    if (!fs::exists(vmfb_path) || fs::file_size(vmfb_path) == 0) {
        throw JitCompileError(
            "iree-compile subprocess (target=" + target +
                ") reported success but produced no vmfb at " +
                vmfb_path.string() + "\nstderr:\n" + stderr_text,
            mlir_path);
    }
}

}  // namespace

// ============================================================================
// Cache stats (Group F.5)
// ============================================================================

namespace {

// Process-wide counters. Wrapped in a struct to keep all the atomics
// together and to make reset trivially correct.
struct GlobalCacheCounters {
    std::atomic<std::uint64_t> hits{0};
    std::atomic<std::uint64_t> misses{0};
    std::atomic<std::uint64_t> retraces{0};
    std::atomic<std::uint64_t> evictions{0};
    // Total compile time is stored as integer nanoseconds for lock-free
    // accumulation; converted to milliseconds in the snapshot.
    std::atomic<std::uint64_t> total_compile_ns{0};
};

auto counters() -> GlobalCacheCounters& {
    static GlobalCacheCounters c;
    return c;
}

}  // namespace

auto cache_stats() -> CacheStats {
    const auto& c = counters();
    CacheStats s;
    s.hits             = c.hits.load(std::memory_order_relaxed);
    s.misses           = c.misses.load(std::memory_order_relaxed);
    s.retraces         = c.retraces.load(std::memory_order_relaxed);
    s.evictions        = c.evictions.load(std::memory_order_relaxed);
    s.total_compile_ms =
        static_cast<double>(c.total_compile_ns.load(std::memory_order_relaxed))
        / 1.0e6;
    return s;
}

auto reset_cache_stats() -> void {
    auto& c = counters();
    c.hits.store(0, std::memory_order_relaxed);
    c.misses.store(0, std::memory_order_relaxed);
    c.retraces.store(0, std::memory_order_relaxed);
    c.evictions.store(0, std::memory_order_relaxed);
    c.total_compile_ns.store(0, std::memory_order_relaxed);
}

namespace internal {
auto record_cache_hit() -> void {
    counters().hits.fetch_add(1, std::memory_order_relaxed);
}
auto record_cache_miss() -> void {
    counters().misses.fetch_add(1, std::memory_order_relaxed);
}
auto record_retrace() -> void {
    counters().retraces.fetch_add(1, std::memory_order_relaxed);
}
auto record_eviction() -> void {
    counters().evictions.fetch_add(1, std::memory_order_relaxed);
}
auto record_compile_ms(double ms) -> void {
    auto ns = static_cast<std::uint64_t>(ms * 1.0e6);
    counters().total_compile_ns.fetch_add(ns, std::memory_order_relaxed);
}
}  // namespace internal

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

    // Cache miss — measure the actual compile time end-to-end so that
    // cache_stats().total_compile_ms reflects only real iree-compile work.
    const auto compile_t0 = std::chrono::steady_clock::now();

    // Path selection: if the linked libIREECompiler registers `opts.target`,
    // use the in-process embedding API (fast, no fork). Otherwise fall
    // through to the iree-compile subprocess, which is the full-GPU pip
    // wheel on this host. Both paths produce the same vmfb format.
    const auto& embedded = embedding_api_supported_targets();
    if (embedded.count(opts.target) == 0) {
        if (!iree_compile_supports(opts.target)) {
            std::ostringstream available;
            available << "[";
            bool first = true;
            for (const auto& t : iree_compile_supported_targets()) {
                if (!first) available << ", ";
                available << t;
                first = false;
            }
            available << "]";
            throw JitCompileError(
                "compile_mlir: target '" + opts.target +
                    "' is not supported by either the linked libIREECompiler "
                    "or the discovered iree-compile binary " +
                    resolve_iree_compile() +
                    ". Subprocess-available targets=" + available.str(),
                mlir_path);
        }
        compile_via_subprocess(mlir_text, opts.target, vmfb_path, mlir_path);
        const auto compile_t1 = std::chrono::steady_clock::now();
        const double compile_ms =
            std::chrono::duration<double, std::milli>(compile_t1 - compile_t0)
                .count();
        internal::record_compile_ms(compile_ms);
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

    const auto compile_t1 = std::chrono::steady_clock::now();
    const double compile_ms =
        std::chrono::duration<double, std::milli>(compile_t1 - compile_t0)
            .count();
    internal::record_compile_ms(compile_ms);

    return CompiledArtifact{vmfb_path, opts.target};
}

}  // namespace tenzor::jit::mlir_jit
