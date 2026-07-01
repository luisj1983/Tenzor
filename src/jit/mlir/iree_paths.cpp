// Phase 13 — IREE binary discovery + capability probing implementation.

#include "tenzor/jit/mlir/iree_paths.hpp"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>
#include <fcntl.h>
#include <sys/wait.h>
#include <unistd.h>

namespace tenzor::jit::mlir_jit {

namespace {

namespace fs = std::filesystem;

/// Test whether `path` is a real, executable file. access(X_OK) alone is not
/// enough: POSIX X_OK on a directory tests searchability, not runnability, so a
/// directory whose basename matches the requested binary would otherwise be
/// accepted and later fail with a confusing popen/shell error. Require a
/// regular file (following symlinks) in addition to the execute bit.
auto is_executable(const std::string& path) -> bool {
    if (path.empty()) return false;
    if (access(path.c_str(), X_OK) != 0) return false;
    std::error_code ec;
    return fs::is_regular_file(fs::status(path, ec)) && !ec;
}

/// Search $PATH for `name`. Returns absolute path or empty string.
auto find_on_path(const std::string& name) -> std::string {
    const char* path_env = std::getenv("PATH");
    if (path_env == nullptr) return {};
    std::string path(path_env);
    std::size_t pos = 0;
    while (pos < path.size()) {
        const std::size_t next = path.find(':', pos);
        const std::size_t end  = next == std::string::npos ? path.size() : next;
        const std::string dir  = path.substr(pos, end - pos);
        if (!dir.empty()) {
            const std::string cand = dir + "/" + name;
            if (is_executable(cand)) {
                return cand;
            }
        }
        if (next == std::string::npos) break;
        pos = next + 1;
    }
    return {};
}

/// The discovery chain used by resolve_iree_compile / resolve_iree_run_module.
/// `env_var` is the user-facing override (e.g. TENZOR_IREE_COMPILE) and
/// `binary` is the executable basename (e.g. "iree-compile").
auto resolve_iree_binary(const char* env_var,
                         const std::string& binary) -> std::string {
    // 1. Explicit env var override.
    if (const char* env = std::getenv(env_var); env != nullptr && *env != '\0') {
        if (is_executable(env)) {
            return env;
        }
        throw std::runtime_error(std::string("$") + env_var + " is set to '" +
                                 env + "' but it is not an executable file.");
    }

    // 2. Project-local pip venv at .venv-iree/, the full-GPU iree-compile
    //    that the project's CMake configure tells users to install. Lives
    //    inside the source tree (gitignored). Preferred over the
    //    CMake-found dist when both are present because the dist often
    //    ships without cuda/rocm backends.
    {
        const std::string cand = std::string(TENZOR_SOURCE_DIR) + "/.venv-iree/bin/" + binary;
        if (is_executable(cand)) {
            return cand;
        }
    }

    // 4. CMake-found binaries. Each binary has its own preprocessor symbol
    // set by src/jit/mlir/CMakeLists.txt; if the requested binary has no
    // direct symbol we fall back to deriving it from iree-compile's parent
    // directory (CMake-found iree-compile + sibling iree-run-module).
    if (binary == "iree-compile") {
#ifdef TENZOR_IREE_COMPILE_PATH
        const std::string cmake_path{TENZOR_IREE_COMPILE_PATH};
        if (is_executable(cmake_path)) {
            return cmake_path;
        }
#endif
    } else if (binary == "iree-run-module") {
#ifdef TENZOR_IREE_RUN_MODULE_PATH
        const std::string cmake_path{TENZOR_IREE_RUN_MODULE_PATH};
        if (is_executable(cmake_path)) {
            return cmake_path;
        }
#endif
#ifdef TENZOR_IREE_COMPILE_PATH
        const auto sibling =
            fs::path(TENZOR_IREE_COMPILE_PATH).parent_path() / binary;
        if (is_executable(sibling.string())) {
            return sibling.string();
        }
#endif
    }

    // 5. $PATH (last resort, mainly for ad-hoc Docker / CI images).
    if (auto p = find_on_path(binary); !p.empty()) {
        return p;
    }

    throw std::runtime_error(
        binary +
        " not found. Set $TENZOR_IREE_COMPILE / $TENZOR_IREE_RUN_MODULE "
        "explicitly, or install IREE (`pip install iree-base-compiler "
        "iree-base-runtime`).");
}

/// Parse one whitespace-separated token per line from `text`, stripping a
/// trailing colon, and return all entries.
auto parse_token_list(const std::string& text) -> std::set<std::string> {
    std::set<std::string> out;
    std::string current;
    for (char c : text) {
        if (c == '\n' || c == '\r') {
            // Find first non-whitespace.
            std::size_t s = 0;
            while (s < current.size() &&
                   (current[s] == ' ' || current[s] == '\t')) {
                ++s;
            }
            if (s >= current.size() || current[s] == '#') {
                current.clear();
                continue;
            }
            std::size_t e = s;
            while (e < current.size() && current[e] != ' ' &&
                   current[e] != '\t' && current[e] != ':') {
                ++e;
            }
            const std::string tok = current.substr(s, e - s);
            if (!tok.empty() && tok != "Registered" && tok != "Available" &&
                tok.find("===") == std::string::npos) {
                out.insert(tok);
            }
            current.clear();
        } else {
            current.push_back(c);
        }
    }
    return out;
}

}  // namespace

auto exec_capture(const std::string& exe,
                  const std::vector<std::string>& args,
                  bool capture_stderr) -> std::string {
    // Build argv in the parent (post-fork allocation in a multithreaded
    // process is unsafe); the child only calls async-signal-safe functions.
    std::vector<char*> argv;
    argv.reserve(args.size() + 2);
    argv.push_back(const_cast<char*>(exe.c_str()));
    for (const auto& a : args) argv.push_back(const_cast<char*>(a.c_str()));
    argv.push_back(nullptr);

    int fds[2];
    if (::pipe(fds) != 0) return {};

    const pid_t pid = ::fork();
    if (pid < 0) {
        ::close(fds[0]);
        ::close(fds[1]);
        return {};
    }

    if (pid == 0) {
        // Child: wire stdout (and optionally stderr) to the pipe write end.
        ::dup2(fds[1], STDOUT_FILENO);
        if (capture_stderr) {
            ::dup2(fds[1], STDERR_FILENO);
        } else {
            const int devnull = ::open("/dev/null", O_WRONLY);
            if (devnull >= 0) {
                ::dup2(devnull, STDERR_FILENO);
                ::close(devnull);
            }
        }
        ::close(fds[0]);
        ::close(fds[1]);
        // execvp: uses `exe` directly when it contains a slash, else searches
        // $PATH — matching the previous shell behaviour without a shell.
        ::execvp(exe.c_str(), argv.data());
        _exit(127);  // exec failed; skip atexit handlers.
    }

    // Parent: drain the pipe, then reap the child.
    ::close(fds[1]);
    std::string out;
    char buf[4096];
    ssize_t n;
    while ((n = ::read(fds[0], buf, sizeof(buf))) > 0) {
        out.append(buf, static_cast<std::size_t>(n));
    }
    ::close(fds[0]);
    int status = 0;
    ::waitpid(pid, &status, 0);
    return out;
}

auto resolve_iree_compile() -> const std::string& {
    static const std::string cached =
        resolve_iree_binary("TENZOR_IREE_COMPILE", "iree-compile");
    return cached;
}

auto resolve_iree_run_module() -> const std::string& {
    static const std::string cached =
        resolve_iree_binary("TENZOR_IREE_RUN_MODULE", "iree-run-module");
    return cached;
}

auto iree_compile_supported_targets() -> const std::set<std::string>& {
    static const std::set<std::string> cached = []() {
        const std::string& bin = resolve_iree_compile();
        const std::string out =
            exec_capture(bin, {"--iree-hal-list-target-backends"},
                         /*capture_stderr=*/false);
        return parse_token_list(out);
    }();
    return cached;
}

auto iree_runtime_supported_drivers() -> const std::set<std::string>& {
    static const std::set<std::string> cached = []() {
        const std::string& bin = resolve_iree_run_module();
        const std::string out =
            exec_capture(bin, {"--list_drivers"}, /*capture_stderr=*/false);
        return parse_token_list(out);
    }();
    return cached;
}

auto iree_compile_supports(const std::string& target) -> bool {
    const auto& set = iree_compile_supported_targets();
    return set.count(target) > 0;
}

auto driver_for_target(const std::string& target) -> std::string {
    if (target == "llvm-cpu")     return "local-task";
    if (target == "cuda")         return "cuda";
    if (target == "rocm")         return "hip";
    if (target == "vulkan-spirv") return "vulkan";
    if (target == "vulkan")       return "vulkan";
    if (target == "metal-spirv")  return "metal";
    if (target == "vmvx")         return "local-sync";
    if (target == "vmvx-inline")  return "local-sync";
    throw std::invalid_argument(
        "driver_for_target: unknown IREE target backend '" + target + "'");
}

}  // namespace tenzor::jit::mlir_jit
