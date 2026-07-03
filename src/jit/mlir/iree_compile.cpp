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

#include <cerrno>

#include <fcntl.h>
#include <poll.h>
#include <sys/stat.h>
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

/// Resolve the default cache directory. Sets `is_shared_temp` when neither
/// XDG_CACHE_HOME nor HOME is available and we fall back to the world-shared
/// temp directory — that path needs ownership/permission hardening before any
/// pre-existing artifact is trusted, since another local user could otherwise
/// pre-create the deterministically-named .vmfb.
auto default_cache_dir(bool& is_shared_temp) -> fs::path {
    is_shared_temp = false;
    if (const char* xdg = std::getenv("XDG_CACHE_HOME"); xdg && *xdg) {
        return fs::path(xdg) / "tenzor" / "jit_mlir";
    }
    if (const char* home = std::getenv("HOME"); home && *home) {
        return fs::path(home) / ".cache" / "tenzor" / "jit_mlir";
    }
    is_shared_temp = true;
    // Per-user subdirectory so distinct users do not share a directory in the
    // world-writable temp root.
    const std::string user_dir = "tenzor-" + std::to_string(::getuid());
    return fs::temp_directory_path() / user_dir / "jit_mlir";
}

/// Create (if needed) the cache directory and, on the shared-temp fallback,
/// lock it down to owner-only (0700) and verify the current process owns it.
/// Throws if an existing directory on the shared path is owned by another user
/// (a sign of squatting).
auto secure_cache_dir(const fs::path& dir, bool is_shared_temp) -> void {
    if (!is_shared_temp) {
        fs::create_directories(dir);
        return;
    }

    // Create the per-user root first so we can chmod it to 0700 before any
    // nested directory is exposed.
    const fs::path user_root = dir.parent_path();
    if (::mkdir(user_root.c_str(), 0700) != 0 && errno != EEXIST) {
        throw JitCompileError(
            "compile_mlir: cannot create per-user cache dir " +
                user_root.string() + ": " + std::strerror(errno),
            {});
    }

    struct stat st {};
    if (::stat(user_root.c_str(), &st) != 0) {
        throw JitCompileError(
            "compile_mlir: cannot stat per-user cache dir " +
                user_root.string() + ": " + std::strerror(errno),
            {});
    }
    if (st.st_uid != ::getuid()) {
        throw JitCompileError(
            "compile_mlir: per-user cache dir " + user_root.string() +
                " is owned by another user (uid " + std::to_string(st.st_uid) +
                "); refusing to trust shared-temp cache",
            {});
    }
    // Enforce owner-only permissions even if the directory pre-existed.
    ::chmod(user_root.c_str(), 0700);

    if (::mkdir(dir.c_str(), 0700) != 0 && errno != EEXIST) {
        throw JitCompileError(
            "compile_mlir: cannot create cache dir " + dir.string() + ": " +
                std::strerror(errno),
            {});
    }
    ::chmod(dir.c_str(), 0700);
}

/// On the shared-temp fallback, verify a pre-existing cached artifact is owned
/// by the current user before it is trusted/executed.
auto cache_file_is_trustworthy(const fs::path& file, bool is_shared_temp)
    -> bool {
    if (!is_shared_temp) {
        return true;
    }
    struct stat st {};
    if (::stat(file.c_str(), &st) != 0) {
        return false;
    }
    return st.st_uid == ::getuid();
}

/// Verify that a .vmfb on disk carries the expected container magic. IREE
/// emits vmfb artifacts as ZIP containers, whose first four bytes are the
/// local-file-header signature "PK\x03\x04". A crash mid-write or a race
/// between two processes on the same cache key can persist a truncated but
/// non-empty file; the previous "size > 0" gate would then accept it as a
/// cache hit and fail later at module-load with an opaque error. Checking the
/// magic lets compile_mlir treat such a file as a miss and recompile.
auto vmfb_has_valid_magic(const fs::path& file) -> bool {
    std::ifstream f(file, std::ios::binary);
    if (!f) {
        return false;
    }
    char magic[4] = {0, 0, 0, 0};
    f.read(magic, sizeof(magic));
    if (f.gcount() != static_cast<std::streamsize>(sizeof(magic))) {
        return false;
    }
    return magic[0] == 'P' && magic[1] == 'K' &&
           magic[2] == '\x03' && magic[3] == '\x04';
}

/// Compile into a unique temp file in `dir` then atomically rename it into
/// `final_path`. The writer is invoked with the temp path; on success the
/// rename publishes a fully-written artifact in one step, so a reader never
/// observes a partial file and concurrent writers cannot corrupt the cache
/// entry. On any failure the temp file is removed and the exception
/// propagates. Returns the temp path actually used (already renamed away).
template <typename Writer>
auto write_vmfb_atomically(const fs::path& final_path,
                           const fs::path& dir,
                           Writer&& writer) -> void {
    const std::string tmpl =
        (dir / (final_path.filename().string() + ".tmpXXXXXX")).string();
    std::vector<char> tmp_buf(tmpl.begin(), tmpl.end());
    tmp_buf.push_back('\0');
    int fd = ::mkstemp(tmp_buf.data());
    if (fd < 0) {
        throw JitCompileError(
            std::string("mkstemp for vmfb output failed: ") +
                std::strerror(errno),
            final_path);
    }
    // The writer reopens/owns the file by path; close our descriptor so it
    // isn't left dangling. The file already exists with the secure mkstemp
    // permissions (0600).
    ::close(fd);
    const fs::path tmp_path(tmp_buf.data());

    try {
        writer(tmp_path);
    } catch (...) {
        std::error_code ec;
        fs::remove(tmp_path, ec);
        throw;
    }

    if (!fs::exists(tmp_path) || fs::file_size(tmp_path) == 0 ||
        !vmfb_has_valid_magic(tmp_path)) {
        std::error_code ec;
        fs::remove(tmp_path, ec);
        throw JitCompileError(
            "compile produced an empty or malformed vmfb at " +
                tmp_path.string(),
            final_path);
    }

    std::error_code ec;
    fs::rename(tmp_path, final_path, ec);
    if (ec) {
        fs::remove(tmp_path, ec);
        throw JitCompileError(
            "atomic rename of vmfb into place failed: " + ec.message(),
            final_path);
    }
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
// Target-name normalization (H6).
// ============================================================================

/// Canonicalize a user-facing target name to the exact string IREE's
/// `--iree-hal-target-backends` flag expects. The Python docs (jit.py /
/// jit.pyi) advertise `target="vulkan"`, but IREE requires `vulkan-spirv`;
/// passed verbatim, iree-compile rejects `vulkan`. Accepting the documented
/// aliases here — at the single compile boundary every path funnels through —
/// makes the documented names actually work while leaving canonical names and
/// any target the local dist supports untouched (validation against the
/// dist's real target set still happens below in compile_mlir()).
auto normalize_iree_target(const std::string& t) -> std::string {
    if (t == "vulkan") return "vulkan-spirv";
    if (t == "hip")    return "rocm";
    return t;
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
    // Open each pipe separately and close any previously-opened ones before
    // throwing, so a later pipe() failure does not leak the earlier fds.
    if (pipe(in_pipe) != 0) {
        throw JitCompileError(std::string("pipe() failed: ") + std::strerror(errno), {});
    }
    if (pipe(out_pipe) != 0) {
        close(in_pipe[0]); close(in_pipe[1]);
        throw JitCompileError(std::string("pipe() failed: ") + std::strerror(errno), {});
    }
    if (pipe(err_pipe) != 0) {
        close(in_pipe[0]); close(in_pipe[1]);
        close(out_pipe[0]); close(out_pipe[1]);
        throw JitCompileError(std::string("pipe() failed: ") + std::strerror(errno), {});
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
        // Scrub LD_LIBRARY_PATH before exec. The parent process loads ROCm
        // (/opt/rocm/lib) and OneAPI (/opt/intel/oneapi/.../lib) shared
        // libraries to register its own backends. If we inherit that
        // LD_LIBRARY_PATH, the child iree-compile picks up MLIR-shaped
        // libraries from /opt/intel/oneapi at dlopen time that don't
        // include the StableHLO dialect — producing a silent
        // "Dialect 'stablehlo' not found" failure on any non-trivial IR.
        // iree-compile bundles its own MLIR in the pip wheel's
        // _mlir_libs/ via rpath, so it doesn't need LD_LIBRARY_PATH to
        // resolve its own dependencies. Unsetting is the correct fix.
        unsetenv("LD_LIBRARY_PATH");
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

    // Drain stdout and stderr concurrently with poll(). Draining stdout to
    // EOF before touching stderr would deadlock: with `-o <file>` the child's
    // stdout is empty and only reaches EOF at process exit, while iree-compile
    // can emit more diagnostics on stderr than the ~64KB pipe buffer holds.
    // Once the stderr pipe fills the child blocks on write(STDERR) forever,
    // never exits, so stdout never reaches EOF. Interleaving the reads lets us
    // keep both pipes drained so the child can always make progress.
    vmfb_out.clear();
    stderr_out.clear();
    {
        struct pollfd fds[2];
        fds[0].fd = out_pipe[0];
        fds[1].fd = err_pipe[0];
        fds[0].events = fds[1].events = POLLIN;
        std::string* const targets[2] = {&vmfb_out, &stderr_out};
        int open_fds = 2;
        char buf[8192];
        while (open_fds > 0) {
            for (int i = 0; i < 2; ++i) {
                fds[i].revents = 0;
                if (fds[i].fd < 0) {
                    fds[i].events = 0;  // ignored by poll() once fd is -1
                }
            }
            const int pr = poll(fds, 2, -1);
            if (pr < 0) {
                if (errno == EINTR) continue;
                break;  // unexpected poll failure; stop draining
            }
            for (int i = 0; i < 2; ++i) {
                if (fds[i].fd < 0) continue;
                // POLLHUP/POLLERR may arrive with or without POLLIN; attempt a
                // read whenever the kernel reports any activity so we capture
                // bytes buffered alongside a hangup.
                if (fds[i].revents & (POLLIN | POLLHUP | POLLERR)) {
                    const ssize_t r = read(fds[i].fd, buf, sizeof(buf));
                    if (r > 0) {
                        targets[i]->append(buf, static_cast<std::size_t>(r));
                    } else if (r == 0) {
                        // EOF: stop polling this descriptor.
                        fds[i].fd = -1;
                        --open_fds;
                    } else if (errno != EINTR && errno != EAGAIN) {
                        fds[i].fd = -1;
                        --open_fds;
                    }
                }
            }
        }
    }
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
                            const std::string& rocm_arch,
                            const std::string& cuda_arch,
                            const fs::path& vmfb_path,
                            const fs::path& mlir_path) -> void {
    const std::string& bin = ::tenzor::jit::mlir_jit::resolve_iree_compile();
    std::vector<std::string> args{
        "--iree-hal-target-backends=" + target,
        // --iree-input-type=auto explicitly enables StableHLO dialect
        // detection. Older IREE accepted "stablehlo"; IREE 3.11+ replaced
        // it with "auto" (analyze-and-pick). "auto" works on both. Without
        // it, complex stablehlo IR fails with "Dialect 'stablehlo' not
        // found for custom op 'stablehlo.constant'".
        "--iree-input-type=auto",
        "--output-format=vm-bytecode",
    };
    // ROCm requires an explicit chip — `--iree-rocm-target=<chip>`. The arch
    // is derived from the actual ROCm device by the caller (M1); compile_mlir
    // guarantees `rocm_arch` is non-empty for a rocm target (falling back to
    // the build constant only when device detection fails).
    if (target == "rocm") {
        args.push_back("--iree-rocm-target=" + rocm_arch);
    }
    // CUDA also needs an explicit target (see compile_mlir); without it
    // iree-compile rejects the cuda backend with "missing GPU target".
    if (target == "cuda" && !cuda_arch.empty()) {
        args.push_back("--iree-cuda-target=" + cuda_arch);
    }
    // Write MLIR text to a temp file rather than piping via stdin. The
    // stdin pipe path was hitting auto-input-type detection failures on
    // larger modules (>32 KB) in IREE 3.11+ — the parser couldn't decide
    // the dialect from a partial pipe read. Passing a file path lets
    // iree-compile mmap the whole thing and sniff dialects reliably.
    // Audit item I.15: use the platform-specific temp dir instead of a
    // hard-coded `/tmp/` prefix.  std::filesystem::temp_directory_path()
    // honours $TMPDIR on POSIX and %TEMP%/%TMP% on Windows, so sandboxed
    // and Windows builds work without patching.
    std::string mlir_tmp_str =
        (std::filesystem::temp_directory_path() / "tenzor_mlir_XXXXXX.mlir").string();
    // mkstemps mutates the template buffer in place; copy into a writable
    // char array of known size (max-path is fine for our use).
    std::vector<char> mlir_tmp_template(mlir_tmp_str.begin(), mlir_tmp_str.end());
    mlir_tmp_template.push_back('\0');
    int mlir_fd = mkstemps(mlir_tmp_template.data(), /*suffixlen=*/5);
    if (mlir_fd < 0) {
        throw JitCompileError(
            std::string("mkstemps for MLIR input failed: ") + std::strerror(errno),
            mlir_path);
    }
    {
        std::size_t total = 0;
        while (total < mlir_text.size()) {
            const ssize_t n = ::write(mlir_fd, mlir_text.data() + total,
                                      mlir_text.size() - total);
            if (n < 0) {
                if (errno == EINTR) continue;
                ::close(mlir_fd);
                ::unlink(mlir_tmp_template.data());
                throw JitCompileError(
                    std::string("write to MLIR temp failed: ") + std::strerror(errno),
                    mlir_path);
            }
            total += static_cast<std::size_t>(n);
        }
    }
    ::close(mlir_fd);
    args.push_back("-o");
    args.push_back(vmfb_path.string());
    args.push_back(std::string(mlir_tmp_template.data()));  // path; iree-compile mmaps it
    std::string stderr_text, vmfb_unused;
    // Pass empty stdin since we're using a file path.
    const int rc = run_iree_compile_subprocess(bin, args, std::string{},
                                               stderr_text, vmfb_unused);
    ::unlink(mlir_tmp_template.data());
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

    // H6: canonicalize documented aliases (e.g. "vulkan" → "vulkan-spirv")
    // before the target string reaches --iree-hal-target-backends / the cache
    // key / the runtime device map, so every downstream user sees the exact
    // name IREE expects.
    const std::string target = normalize_iree_target(opts.target);

    // M1: the ROCm ISA to compile for. The caller (mlir_invoke) fills
    // opts.rocm_arch from the actual device; only if that detection failed do
    // we fall back to the build-time constant / default so a rocm compile is
    // never left without an explicit chip.
    std::string rocm_arch = opts.rocm_arch;
    if (target == "rocm" && rocm_arch.empty()) {
#ifdef TENZOR_DEFAULT_ROCM_TARGET
        rocm_arch = TENZOR_DEFAULT_ROCM_TARGET;
#else
        rocm_arch = "gfx1150";
#endif
    }

    // CUDA, like ROCm, requires an explicit target — current iree-compile
    // rejects a cuda backend with no `--iree-cuda-target` ("missing GPU target
    // in #hal.executable.target"). The old code assumed a "sensible default"
    // and passed nothing, so every cuda compile failed and silently fell back
    // to eager (cuda JIT never ran). Resolve an arch: TENZOR_CUDA_TARGET env
    // override, else the build-time default, else sm_80 (Ampere baseline —
    // PTX forward-compat JITs onto newer GPUs, verified on Blackwell).
    std::string cuda_arch;
    if (target == "cuda") {
        if (const char* e = std::getenv("TENZOR_CUDA_TARGET"); e != nullptr && *e != '\0') {
            cuda_arch = e;
        } else {
#ifdef TENZOR_DEFAULT_CUDA_TARGET
            cuda_arch = TENZOR_DEFAULT_CUDA_TARGET;
#else
            cuda_arch = "sm_80";
#endif
        }
    }

    bool is_shared_temp = false;
    fs::path cache_dir;
    if (opts.cache_dir.empty()) {
        cache_dir = default_cache_dir(is_shared_temp);
    } else {
        // Caller-supplied directory is assumed to be a private location.
        cache_dir = opts.cache_dir;
    }
    secure_cache_dir(cache_dir, is_shared_temp);

    // Fold the ROCm arch into the cache key: two GPUs with different ISAs
    // (e.g. gfx1150 vs gfx90a) produce different bytecode from the same MLIR,
    // and must not alias to one .vmfb. Non-rocm targets keep the bare target.
    const std::string key_target =
        (target == "rocm" && !rocm_arch.empty()) ? (target + ":" + rocm_arch)
      : (target == "cuda" && !cuda_arch.empty()) ? (target + ":" + cuda_arch)
                                                 : target;
    const std::string key = compute_cache_key(mlir_text, key_target);
    const fs::path vmfb_path = cache_dir / (key + ".vmfb");
    const fs::path mlir_path = cache_dir / (key + ".mlir");

    // Always dump the source for offline debug, even on cache hit, so that the
    // path returned in JitCompileError exists.
    {
        std::ofstream f(mlir_path, std::ios::binary | std::ios::trunc);
        f.write(mlir_text.data(),
                static_cast<std::streamsize>(mlir_text.size()));
    }

    if (fs::exists(vmfb_path) && fs::file_size(vmfb_path) > 0 &&
        cache_file_is_trustworthy(vmfb_path, is_shared_temp) &&
        vmfb_has_valid_magic(vmfb_path)) {
        return CompiledArtifact{vmfb_path, target};
    }

    // Cache miss — measure the actual compile time end-to-end so that
    // cache_stats().total_compile_ms reflects only real iree-compile work.
    const auto compile_t0 = std::chrono::steady_clock::now();

    // Path selection: if the linked libIREECompiler registers `target`,
    // use the in-process embedding API (fast, no fork). Otherwise fall
    // through to the iree-compile subprocess, which is the full-GPU pip
    // wheel on this host. Both paths produce the same vmfb format.
    const auto& embedded = embedding_api_supported_targets();
    if (embedded.count(target) == 0) {
        if (!iree_compile_supports(target)) {
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
                "compile_mlir: target '" + target +
                    "' is not supported by either the linked libIREECompiler "
                    "or the discovered iree-compile binary " +
                    resolve_iree_compile() +
                    ". Supported target names=[llvm-cpu, cuda, rocm, "
                    "vulkan-spirv (alias: vulkan)]. "
                    "Subprocess-available targets=" + available.str(),
                mlir_path);
        }
        write_vmfb_atomically(vmfb_path, cache_dir,
                              [&](const fs::path& tmp_out) {
            compile_via_subprocess(mlir_text, target, rocm_arch, cuda_arch,
                                   tmp_out, mlir_path);
        });
        const auto compile_t1 = std::chrono::steady_clock::now();
        const double compile_ms =
            std::chrono::duration<double, std::milli>(compile_t1 - compile_t0)
                .count();
        internal::record_compile_ms(compile_ms);
        return CompiledArtifact{vmfb_path, target};
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
    // string. StableHLO is the default input form on IREE 3.11+ (auto-detected).
    const std::string target_flag =
        "--iree-hal-target-backends=" + target;
    // IREE 3.11+ dropped the explicit "stablehlo" input-type alias in
    // favour of "auto" (analyze-and-pick). "auto" is the right value on
    // both old (3.0–3.10, where it was already accepted alongside
    // "stablehlo") and new (3.11+, where it's the standard).
    // We CAN'T leave this empty — ireeCompilerSessionSetFlags treats an
    // empty string as a positional arg and rejects it.
    const std::string input_flag = "--iree-input-type=auto";
    // ROCm requires an explicit chip — `--iree-rocm-target=<chip>`. CUDA
    // has a sensible default in iree-compile; Vulkan/CPU are
    // chip-independent.
    std::string rocm_target_flag;
    std::string cuda_target_flag;
    std::vector<const char*> flags = {target_flag.c_str(), input_flag.c_str()};
    if (target == "rocm") {
        // Device-derived arch (M1); rocm_arch is guaranteed non-empty above.
        rocm_target_flag = std::string("--iree-rocm-target=") + rocm_arch;
        flags.push_back(rocm_target_flag.c_str());
    }
    // CUDA needs an explicit target too — the old "sensible default" comment was
    // stale (current iree-compile rejects a cuda backend without it). cuda_arch
    // is resolved (env / build default / sm_80) above.
    if (target == "cuda" && !cuda_arch.empty()) {
        cuda_target_flag = std::string("--iree-cuda-target=") + cuda_arch;
        flags.push_back(cuda_target_flag.c_str());
    }
    if (auto* set_err = ireeCompilerSessionSetFlags(
            session, /*argc=*/static_cast<int>(flags.size()), flags.data());
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

    // Write to a unique temp file in the cache dir, then atomically rename it
    // into the final vmfb path only after a successful, verified write. This
    // prevents a crash mid-write or two racing processes from persisting a
    // truncated-but-nonzero artifact that a later run would accept as a cache
    // hit (and then fail at module-load).
    const std::string tmpl =
        (cache_dir / (vmfb_path.filename().string() + ".tmpXXXXXX")).string();
    std::vector<char> tmp_buf(tmpl.begin(), tmpl.end());
    tmp_buf.push_back('\0');
    int tmp_fd = ::mkstemp(tmp_buf.data());
    if (tmp_fd < 0) {
        throw JitCompileError(
            std::string("mkstemp for vmfb output failed: ") +
                std::strerror(errno),
            mlir_path);
    }
    ::close(tmp_fd);
    const fs::path tmp_vmfb(tmp_buf.data());
    // Remove the temp file on any error path below.
    struct TmpFileGuard {
        fs::path path;
        bool armed = true;
        ~TmpFileGuard() {
            if (armed) {
                std::error_code ec;
                fs::remove(path, ec);
            }
        }
    } tmp_guard{tmp_vmfb};

    iree_compiler_output_t* output = nullptr;
    const std::string tmp_vmfb_str = tmp_vmfb.string();
    if (auto* out_err = ireeCompilerOutputOpenFile(tmp_vmfb_str.c_str(),
                                                   &output);
        out_err != nullptr) {
        CompilerError err{out_err};
        throw JitCompileError(
            "ireeCompilerOutputOpenFile(" + tmp_vmfb_str +
                ") failed: " + err.message(),
            mlir_path);
    }
    {
        struct OutputGuard {
            iree_compiler_output_t* o;
            ~OutputGuard() { ireeCompilerOutputDestroy(o); }
        } output_guard{output};

        if (auto* vmb_err =
                ireeCompilerInvocationOutputVMBytecode(inv, output);
            vmb_err != nullptr) {
            CompilerError err{vmb_err};
            throw JitCompileError(
                "ireeCompilerInvocationOutputVMBytecode failed: " +
                    err.message() + "\nDiagnostics:\n" + diag_buffer,
                mlir_path);
        }
        ireeCompilerOutputKeep(output);
        // output_guard destructor here flushes/closes the temp file before we
        // verify and rename it.
    }

    if (!fs::exists(tmp_vmfb) || fs::file_size(tmp_vmfb) == 0 ||
        !vmfb_has_valid_magic(tmp_vmfb)) {
        throw JitCompileError(
            "in-process compile produced an empty or malformed vmfb at " +
                tmp_vmfb_str,
            mlir_path);
    }
    {
        std::error_code ec;
        fs::rename(tmp_vmfb, vmfb_path, ec);
        if (ec) {
            throw JitCompileError(
                "atomic rename of vmfb into place failed: " + ec.message(),
                mlir_path);
        }
    }
    tmp_guard.armed = false;  // renamed away; nothing to clean up.

    const auto compile_t1 = std::chrono::steady_clock::now();
    const double compile_ms =
        std::chrono::duration<double, std::milli>(compile_t1 - compile_t0)
            .count();
    internal::record_compile_ms(compile_ms);

    return CompiledArtifact{vmfb_path, target};
}

}  // namespace tenzor::jit::mlir_jit
