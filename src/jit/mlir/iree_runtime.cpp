// Phase 13 / Task A.8 — IREE Runtime invocation wrapper.

#include "tenzor/jit/mlir/iree_runtime.hpp"

#include "tenzor/core/tensor.hpp"
#include "tenzor/jit/mlir/iree_paths.hpp"
#include "tenzor/ops/creation.hpp"

#include <array>
#include <cctype>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <sys/wait.h>
#include <unistd.h>
#include <fcntl.h>

namespace tenzor::jit::mlir_jit {

namespace fs = std::filesystem;

namespace {

/// Map an IREE HAL target (as set in CompileOptions::target) to the runtime
/// driver name accepted by `iree-run-module --device=...`.
auto device_for_target(const std::string& target) -> std::string {
    if (target == "llvm-cpu") {
        return "local-sync";
    }
    if (target == "cuda") {
        return "cuda";
    }
    if (target == "rocm") {
        return "hip";
    }
    if (target == "vulkan-spirv" || target == "vulkan") {
        return "vulkan";
    }
    throw JitInvokeError("Unsupported IREE target for runtime invoke: " +
                         target);
}

auto dtype_to_iree_element(::tenzor::DType d) -> std::string {
    using ::tenzor::DType;
    switch (d) {
        case DType::Float32: return "f32";
        case DType::Float64: return "f64";
        case DType::Int32:   return "i32";
        case DType::Int64:   return "i64";
        default: break;
    }
    throw std::invalid_argument(
        "iree-run-module marshaling: dtype not yet supported (only Float32, "
        "Float64, Int32, Int64). value=" +
        std::to_string(static_cast<int>(d)));
}

auto iree_element_to_dtype(const std::string& s) -> ::tenzor::DType {
    using ::tenzor::DType;
    if (s == "f32") return DType::Float32;
    if (s == "f64") return DType::Float64;
    if (s == "i32") return DType::Int32;
    if (s == "i64") return DType::Int64;
    throw JitInvokeError("Unsupported IREE element type in output: " + s);
}

/// Render `--input=DxDx...xT=v0 v1 ...`.
auto render_input_flag(const ::tenzor::Tensor& t) -> std::string {
    const ::tenzor::Tensor cpu = t.cpu().contiguous();
    std::ostringstream os;
    os << "--input=";
    for (auto d : cpu.shape()) {
        os << d << 'x';
    }
    const std::string et = dtype_to_iree_element(cpu.dtype());
    os << et << '=';
    const int64_t n = cpu.numel();
    os.setf(std::ios::fmtflags(0), std::ios::floatfield);
    os << std::setprecision(17);
    if (cpu.dtype() == ::tenzor::DType::Float32) {
        const float* p = cpu.data<float>();
        for (int64_t i = 0; i < n; ++i) {
            if (i != 0) os << ' ';
            os << p[i];
        }
    } else if (cpu.dtype() == ::tenzor::DType::Float64) {
        const double* p = cpu.data<double>();
        for (int64_t i = 0; i < n; ++i) {
            if (i != 0) os << ' ';
            os << p[i];
        }
    } else if (cpu.dtype() == ::tenzor::DType::Int32) {
        const int32_t* p = cpu.data<int32_t>();
        for (int64_t i = 0; i < n; ++i) {
            if (i != 0) os << ' ';
            os << p[i];
        }
    } else if (cpu.dtype() == ::tenzor::DType::Int64) {
        const int64_t* p = cpu.data<int64_t>();
        for (int64_t i = 0; i < n; ++i) {
            if (i != 0) os << ' ';
            os << p[i];
        }
    } else {
        (void)dtype_to_iree_element(cpu.dtype());  // throws
    }
    return os.str();
}

struct SubprocessResult {
    int exit_code;
    std::string stdout_text;
    std::string stderr_text;
};

/// Run `argv` as a child process with no stdin, captured stdout+stderr.
auto run_subprocess(const std::vector<std::string>& argv) -> SubprocessResult {
    int out_pipe[2];
    int err_pipe[2];
    if (pipe(out_pipe) != 0 || pipe(err_pipe) != 0) {
        throw JitInvokeError("pipe() failed: " +
                             std::string(std::strerror(errno)));
    }

    pid_t pid = fork();
    if (pid < 0) {
        close(out_pipe[0]); close(out_pipe[1]);
        close(err_pipe[0]); close(err_pipe[1]);
        throw JitInvokeError("fork() failed: " +
                             std::string(std::strerror(errno)));
    }

    if (pid == 0) {
        // Child
        dup2(out_pipe[1], STDOUT_FILENO);
        dup2(err_pipe[1], STDERR_FILENO);
        close(out_pipe[0]); close(out_pipe[1]);
        close(err_pipe[0]); close(err_pipe[1]);

        std::vector<char*> c_argv;
        c_argv.reserve(argv.size() + 1);
        for (const auto& a : argv) {
            c_argv.push_back(const_cast<char*>(a.c_str()));
        }
        c_argv.push_back(nullptr);
        execv(c_argv[0], c_argv.data());
        // If we reach here, execv failed.
        const std::string msg =
            "execv failed: " + std::string(std::strerror(errno)) + "\n";
        (void)!write(STDERR_FILENO, msg.data(), msg.size());
        _exit(127);
    }

    // Parent
    close(out_pipe[1]);
    close(err_pipe[1]);

    auto drain = [](int fd) {
        std::string s;
        char buf[4096];
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

    SubprocessResult res;
    res.stdout_text = drain(out_pipe[0]);
    res.stderr_text = drain(err_pipe[0]);
    close(out_pipe[0]);
    close(err_pipe[0]);

    int status = 0;
    while (waitpid(pid, &status, 0) < 0) {
        if (errno != EINTR) {
            throw JitInvokeError("waitpid() failed: " +
                                 std::string(std::strerror(errno)));
        }
    }
    if (WIFEXITED(status)) {
        res.exit_code = WEXITSTATUS(status);
    } else if (WIFSIGNALED(status)) {
        res.exit_code = 128 + WTERMSIG(status);
    } else {
        res.exit_code = -1;
    }
    return res;
}

/// Parse one `result[N]: hal.buffer_view` block from stdout into a Tensor.
/// The line after the header has format `DxDxDxT=v0 v1 v2 ...`.
auto parse_output_line(const std::string& shape_and_data)
    -> ::tenzor::Tensor {
    const std::size_t eq_pos = shape_and_data.find('=');
    if (eq_pos == std::string::npos) {
        throw JitInvokeError("malformed iree-run-module output (no '='): " +
                             shape_and_data);
    }
    const std::string header = shape_and_data.substr(0, eq_pos);
    std::string values = shape_and_data.substr(eq_pos + 1);

    // header looks like "4xf32" or "2x3xf64" or "f32" (rank 0).
    std::vector<int64_t> shape;
    std::string element;
    std::size_t pos = 0;
    while (pos < header.size()) {
        if (std::isdigit(static_cast<unsigned char>(header[pos]))) {
            int64_t v = 0;
            while (pos < header.size() &&
                   std::isdigit(static_cast<unsigned char>(header[pos]))) {
                v = v * 10 + (header[pos] - '0');
                ++pos;
            }
            shape.push_back(v);
            if (pos < header.size() && header[pos] == 'x') {
                ++pos;
            }
        } else {
            element = header.substr(pos);
            break;
        }
    }
    if (element.empty()) {
        throw JitInvokeError("malformed iree-run-module output header: " +
                             header);
    }
    const ::tenzor::DType dt = iree_element_to_dtype(element);

    int64_t numel = 1;
    for (auto d : shape) numel *= d;
    if (shape.empty()) numel = 1;

    // For multi-dim outputs `iree-run-module` formats values with inner
    // brackets — e.g. `4x1xf32=[13][2][11][0]` or `2x2xf32=[[1 2][3 4]]`.
    // Normalize: replace any non-numeric punctuation that confuses
    // stream reads with a single space, leaving only whitespace, digits,
    // signs, decimal points and exponent letters.
    for (auto& c : values) {
        if (c == '[' || c == ']' || c == ',') c = ' ';
    }

    ::tenzor::Tensor out = ::tenzor::full(shape, 0.0, dt);
    std::istringstream is(values);
    if (dt == ::tenzor::DType::Float32) {
        float* p = out.data<float>();
        for (int64_t i = 0; i < numel; ++i) is >> p[i];
    } else if (dt == ::tenzor::DType::Float64) {
        double* p = out.data<double>();
        for (int64_t i = 0; i < numel; ++i) is >> p[i];
    } else if (dt == ::tenzor::DType::Int32) {
        int32_t* p = out.data<int32_t>();
        for (int64_t i = 0; i < numel; ++i) is >> p[i];
    } else if (dt == ::tenzor::DType::Int64) {
        int64_t* p = out.data<int64_t>();
        for (int64_t i = 0; i < numel; ++i) is >> p[i];
    }
    return out;
}

/// Extract every `DxDx...=v0 v1 ...` line that follows a `result[N]:
/// hal.buffer_view` header from the iree-run-module stdout.
auto parse_outputs(const std::string& stdout_text)
    -> std::vector<::tenzor::Tensor> {
    std::vector<::tenzor::Tensor> outs;
    std::istringstream is(stdout_text);
    std::string line;
    bool expect_value = false;
    while (std::getline(is, line)) {
        if (expect_value) {
            outs.push_back(parse_output_line(line));
            expect_value = false;
            continue;
        }
        if (line.find("result[") == 0 &&
            line.find("hal.buffer_view") != std::string::npos) {
            expect_value = true;
        }
    }
    return outs;
}

}  // namespace

auto IreeInvoker::load(const CompiledArtifact& artifact)
    -> std::unique_ptr<IreeInvoker> {
    if (!fs::exists(artifact.vmfb_path)) {
        throw JitInvokeError("vmfb not found: " + artifact.vmfb_path.string());
    }
    auto inv = std::unique_ptr<IreeInvoker>(new IreeInvoker());
    inv->vmfb_path_       = artifact.vmfb_path.string();
    inv->target_          = artifact.target;
    inv->device_          = device_for_target(artifact.target);
    inv->iree_run_module_ = ::tenzor::jit::mlir_jit::resolve_iree_run_module();
    return inv;
}

auto IreeInvoker::invoke(const std::vector<::tenzor::Tensor>& inputs)
    -> std::vector<::tenzor::Tensor> {
    std::vector<std::string> argv;
    argv.reserve(inputs.size() + 5);
    argv.push_back(iree_run_module_);
    argv.push_back("--device=" + device_);
    argv.push_back("--module=" + vmfb_path_);
    argv.push_back("--function=main");
    // Bump element cap so larger tensors print in full.
    argv.push_back("--output_max_element_count=1048576");
    for (const auto& t : inputs) {
        argv.push_back(render_input_flag(t));
    }
    SubprocessResult res = run_subprocess(argv);
    if (res.exit_code != 0) {
        throw JitInvokeError(
            "iree-run-module exit=" + std::to_string(res.exit_code) +
            "\nstderr:\n" + res.stderr_text +
            "\nstdout:\n" + res.stdout_text);
    }
    auto outs = parse_outputs(res.stdout_text);
    if (outs.empty()) {
        throw JitInvokeError(
            "iree-run-module produced no parseable outputs.\nstdout:\n" +
            res.stdout_text);
    }
    return outs;
}

IreeInvoker::~IreeInvoker() = default;

}  // namespace tenzor::jit::mlir_jit
