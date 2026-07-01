// Phase 13 / Task A.8 + X.2 — IREE Runtime invocation wrapper.
//
// Implements both modes of IreeInvoker:
//
//   - Mode::InProcess  (default): drives the IREE C runtime in-process via
//                                 iree_runtime_instance + session + call.
//                                 Registers the Tenzor VM native module
//                                 (`tenzor_plugin`) before loading the
//                                 bytecode so the 4 dialect-op callbacks
//                                 resolve to the existing tenzor kernels.
//   - Mode::Subprocess          : invokes iree-run-module on the cached .vmfb.
//                                 Used when the linked runtime lacks the HAL
//                                 driver for a target.

#include "tenzor/jit/mlir/iree_runtime.hpp"

#include "_iree_marshal.hpp"
#include "tenzor/core/tensor.hpp"
#include "tenzor/jit/mlir/iree_paths.hpp"
#include "tenzor/ops/creation.hpp"

#include <iree/runtime/api.h>
#include <iree/vm/api.h>

#include <array>
#include <cctype>
#include <limits>
#include <cerrno>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <sys/wait.h>
#include <unistd.h>
#include <fcntl.h>
#include <poll.h>

namespace tenzor::jit::mlir_jit {

namespace fs = std::filesystem;

// Declared in iree_customcalls.cpp: creates the in-process VM native module
// "tenzor_plugin" providing the 4 dialect-op callbacks.
auto create_tenzor_plugin_module(iree_vm_instance_t* instance,
                                 iree_allocator_t allocator,
                                 iree_hal_device_t* device,
                                 iree_vm_module_t** out_module)
    -> iree_status_t;

namespace {

/// Map an IREE HAL target (as set in CompileOptions::target) to the runtime
/// driver name accepted by `iree-run-module --device=...` and
/// `iree_runtime_instance_try_create_default_device`.
auto device_for_target(const std::string& target) -> std::string {
    if (target == "llvm-cpu") {
        return "local-task";
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
        case DType::Float32:    return "f32";
        case DType::Float64:    return "f64";
        case DType::Float16:    return "f16";
        case DType::BFloat16:   return "bf16";
        case DType::Int8:       return "i8";
        case DType::Int16:      return "i16";
        case DType::Int32:      return "i32";
        case DType::Int64:      return "i64";
        case DType::UInt8:      return "ui8";
        case DType::UInt16:     return "ui16";
        case DType::UInt32:     return "ui32";
        case DType::UInt64:     return "ui64";
        case DType::Bool:       return "i1";
        case DType::Complex64:  return "complex<f32>";
        case DType::Complex128: return "complex<f64>";
        default: break;
    }
    throw std::invalid_argument(
        "iree-run-module marshaling: dtype not supported. value=" +
        std::to_string(static_cast<int>(d)));
}

auto iree_element_to_dtype(const std::string& s) -> ::tenzor::DType {
    using ::tenzor::DType;
    if (s == "f32")           return DType::Float32;
    if (s == "f64")           return DType::Float64;
    if (s == "f16")           return DType::Float16;
    if (s == "bf16")          return DType::BFloat16;
    if (s == "i8")            return DType::Int8;
    if (s == "i16")           return DType::Int16;
    if (s == "i32")           return DType::Int32;
    if (s == "i64")           return DType::Int64;
    if (s == "ui8")           return DType::UInt8;
    if (s == "ui16")          return DType::UInt16;
    if (s == "ui32")          return DType::UInt32;
    if (s == "ui64")          return DType::UInt64;
    if (s == "i1")            return DType::Bool;
    if (s == "complex<f32>")  return DType::Complex64;
    if (s == "complex<f64>")  return DType::Complex128;
    throw JitInvokeError("Unsupported IREE element type in output: " + s);
}

/// Render `--input=DxDx...xT=v0 v1 ...` for the subprocess path.
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
    } else if (cpu.dtype() == ::tenzor::DType::Int8) {
        const int8_t* p = cpu.data<int8_t>();
        for (int64_t i = 0; i < n; ++i) {
            if (i != 0) os << ' ';
            os << static_cast<int>(p[i]);
        }
    } else if (cpu.dtype() == ::tenzor::DType::Int16) {
        const int16_t* p = cpu.data<int16_t>();
        for (int64_t i = 0; i < n; ++i) {
            if (i != 0) os << ' ';
            os << p[i];
        }
    } else if (cpu.dtype() == ::tenzor::DType::UInt8) {
        const uint8_t* p = cpu.data<uint8_t>();
        for (int64_t i = 0; i < n; ++i) {
            if (i != 0) os << ' ';
            os << static_cast<unsigned>(p[i]);
        }
    } else if (cpu.dtype() == ::tenzor::DType::UInt16) {
        const uint16_t* p = cpu.data<uint16_t>();
        for (int64_t i = 0; i < n; ++i) {
            if (i != 0) os << ' ';
            os << p[i];
        }
    } else if (cpu.dtype() == ::tenzor::DType::UInt32) {
        const uint32_t* p = cpu.data<uint32_t>();
        for (int64_t i = 0; i < n; ++i) {
            if (i != 0) os << ' ';
            os << p[i];
        }
    } else if (cpu.dtype() == ::tenzor::DType::UInt64) {
        const uint64_t* p = cpu.data<uint64_t>();
        for (int64_t i = 0; i < n; ++i) {
            if (i != 0) os << ' ';
            os << p[i];
        }
    } else if (cpu.dtype() == ::tenzor::DType::Bool) {
        const bool* p = cpu.data<bool>();
        for (int64_t i = 0; i < n; ++i) {
            if (i != 0) os << ' ';
            os << (p[i] ? 1 : 0);
        }
    } else if (cpu.dtype() == ::tenzor::DType::Float16) {
        // IREE accepts f16 input values as ASCII floats; upcast each element via
        // the Float16 → float conversion operator.
        const ::tenzor::Float16* p = cpu.data<::tenzor::Float16>();
        for (int64_t i = 0; i < n; ++i) {
            if (i != 0) os << ' ';
            os << static_cast<float>(p[i]);
        }
    } else if (cpu.dtype() == ::tenzor::DType::BFloat16) {
        const ::tenzor::BFloat16* p = cpu.data<::tenzor::BFloat16>();
        for (int64_t i = 0; i < n; ++i) {
            if (i != 0) os << ' ';
            os << static_cast<float>(p[i]);
        }
    } else if (cpu.dtype() == ::tenzor::DType::Complex64) {
        // IREE's iree-run-module accepts complex values as "(re,im)" pairs.
        const std::complex<float>* p = cpu.data<std::complex<float>>();
        for (int64_t i = 0; i < n; ++i) {
            if (i != 0) os << ' ';
            os << '(' << p[i].real() << ',' << p[i].imag() << ')';
        }
    } else if (cpu.dtype() == ::tenzor::DType::Complex128) {
        const std::complex<double>* p = cpu.data<std::complex<double>>();
        for (int64_t i = 0; i < n; ++i) {
            if (i != 0) os << ' ';
            os << '(' << p[i].real() << ',' << p[i].imag() << ')';
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
    if (pipe(out_pipe) != 0) {
        throw JitInvokeError("pipe() failed: " + std::string(std::strerror(errno)));
    }
    if (pipe(err_pipe) != 0) {
        // Close the first pipe's fds before throwing so they don't leak.
        close(out_pipe[0]); close(out_pipe[1]);
        throw JitInvokeError("pipe() failed: " + std::string(std::strerror(errno)));
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
        const std::string msg =
            "execv failed: " + std::string(std::strerror(errno)) + "\n";
        (void)!write(STDERR_FILENO, msg.data(), msg.size());
        _exit(127);
    }

    // Parent
    close(out_pipe[1]);
    close(err_pipe[1]);

    SubprocessResult res;

    // Drain stdout and stderr concurrently. A sequential drain (read stdout to
    // EOF, then read stderr) can deadlock: the child can block writing to the
    // stderr pipe once its ~64KB kernel buffer fills while it is still emitting
    // stdout, but the parent would be blocked reading stdout (EOF only arrives
    // on child exit, which never happens because the child is blocked). With
    // --output_max_element_count=2147483647 stdout can be very large, so this is
    // reachable. poll() over both fds, reading whichever is ready, prevents
    // either pipe from filling.
    //
    // Use non-blocking reads so that a ready fd is fully drained without ever
    // blocking on a partially-filled buffer.
    auto set_nonblocking = [](int fd) {
        const int flags = fcntl(fd, F_GETFL, 0);
        if (flags >= 0) {
            (void)fcntl(fd, F_SETFL, flags | O_NONBLOCK);
        }
    };
    set_nonblocking(out_pipe[0]);
    set_nonblocking(err_pipe[0]);

    std::array<struct pollfd, 2> pfds{};
    pfds[0].fd = out_pipe[0];
    pfds[0].events = POLLIN;
    pfds[1].fd = err_pipe[0];
    pfds[1].events = POLLIN;

    std::array<std::string*, 2> sinks{&res.stdout_text, &res.stderr_text};

    int open_count = 2;
    char buf[4096];
    while (open_count > 0) {
        int npfds = 0;
        std::array<int, 2> idx_map{-1, -1};
        for (int i = 0; i < 2; ++i) {
            if (pfds[i].fd >= 0) {
                idx_map[npfds] = i;
                ++npfds;
            }
        }
        // Compact poll array to the still-open fds.
        std::array<struct pollfd, 2> active{};
        for (int j = 0; j < npfds; ++j) {
            active[j] = pfds[idx_map[j]];
        }

        const int pr = poll(active.data(), static_cast<nfds_t>(npfds), -1);
        if (pr < 0) {
            if (errno == EINTR) continue;
            break;
        }

        for (int j = 0; j < npfds; ++j) {
            const int i = idx_map[j];
            const short revents = active[j].revents;
            if (revents == 0) continue;

            if (revents & (POLLIN | POLLHUP | POLLERR)) {
                bool closed = false;
                while (true) {
                    const ssize_t r = read(pfds[i].fd, buf, sizeof(buf));
                    if (r > 0) {
                        sinks[i]->append(buf, static_cast<std::size_t>(r));
                    } else if (r == 0) {
                        closed = true;  // EOF
                        break;
                    } else {
                        if (errno == EINTR) continue;
                        if (errno == EAGAIN || errno == EWOULDBLOCK) break;
                        closed = true;  // unrecoverable error
                        break;
                    }
                }
                if (closed) {
                    close(pfds[i].fd);
                    pfds[i].fd = -1;
                    --open_count;
                }
            }
        }
    }

    for (int i = 0; i < 2; ++i) {
        if (pfds[i].fd >= 0) {
            close(pfds[i].fd);
            pfds[i].fd = -1;
        }
    }

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
auto parse_output_line(const std::string& shape_and_data)
    -> ::tenzor::Tensor {
    const std::size_t eq_pos = shape_and_data.find('=');
    if (eq_pos == std::string::npos) {
        throw JitInvokeError("malformed iree-run-module output (no '='): " +
                             shape_and_data);
    }
    const std::string header = shape_and_data.substr(0, eq_pos);
    std::string values = shape_and_data.substr(eq_pos + 1);

    std::vector<int64_t> shape;
    std::string element;
    std::size_t pos = 0;
    while (pos < header.size()) {
        if (std::isdigit(static_cast<unsigned char>(header[pos]))) {
            int64_t v = 0;
            while (pos < header.size() &&
                   std::isdigit(static_cast<unsigned char>(header[pos]))) {
                int digit = header[pos] - '0';
                // Overflow-guarded accumulation: a corrupted/huge header must
                // not silently wrap to a negative/garbage dimension.
                if (v > (std::numeric_limits<int64_t>::max() - digit) / 10) {
                    throw JitInvokeError(
                        "iree-run-module output header: dimension overflows int64: " +
                        header);
                }
                v = v * 10 + digit;
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
    for (auto d : shape) {
        // Each dim is already non-negative (parsed from digits) and overflow-
        // bounded above; guard the running product too so numel cannot wrap.
        if (d != 0 && numel > std::numeric_limits<int64_t>::max() / d) {
            throw JitInvokeError(
                "iree-run-module output header: shape product overflows int64: " +
                header);
        }
        numel *= d;
    }
    if (shape.empty()) numel = 1;

    // Strip iree-run-module formatting punctuation. For complex values we
    // additionally turn the "(re,im)" wrappers into whitespace-separated pairs
    // so the same stream-based reader works.
    const bool is_complex =
        (dt == ::tenzor::DType::Complex64 || dt == ::tenzor::DType::Complex128);
    for (auto& c : values) {
        if (c == '[' || c == ']') c = ' ';
        if (c == ',' && !is_complex) c = ' ';
    }
    if (is_complex) {
        for (auto& c : values) {
            if (c == '(' || c == ')' || c == ',') c = ' ';
        }
    }

    ::tenzor::Tensor out = ::tenzor::full(shape, 0.0, dt);
    std::istringstream is(values);
    if (dt == ::tenzor::DType::Float32) {
        float* p = out.data<float>();
        for (int64_t i = 0; i < numel; ++i) is >> p[i];
    } else if (dt == ::tenzor::DType::Float64) {
        double* p = out.data<double>();
        for (int64_t i = 0; i < numel; ++i) is >> p[i];
    } else if (dt == ::tenzor::DType::Float16) {
        ::tenzor::Float16* p = out.data<::tenzor::Float16>();
        for (int64_t i = 0; i < numel; ++i) {
            float v = 0.0f;
            is >> v;
            p[i] = ::tenzor::Float16(v);
        }
    } else if (dt == ::tenzor::DType::BFloat16) {
        ::tenzor::BFloat16* p = out.data<::tenzor::BFloat16>();
        for (int64_t i = 0; i < numel; ++i) {
            float v = 0.0f;
            is >> v;
            p[i] = ::tenzor::BFloat16(v);
        }
    } else if (dt == ::tenzor::DType::Int8) {
        int8_t* p = out.data<int8_t>();
        for (int64_t i = 0; i < numel; ++i) {
            int v = 0;
            is >> v;
            p[i] = static_cast<int8_t>(v);
        }
    } else if (dt == ::tenzor::DType::Int16) {
        int16_t* p = out.data<int16_t>();
        for (int64_t i = 0; i < numel; ++i) is >> p[i];
    } else if (dt == ::tenzor::DType::Int32) {
        int32_t* p = out.data<int32_t>();
        for (int64_t i = 0; i < numel; ++i) is >> p[i];
    } else if (dt == ::tenzor::DType::Int64) {
        int64_t* p = out.data<int64_t>();
        for (int64_t i = 0; i < numel; ++i) is >> p[i];
    } else if (dt == ::tenzor::DType::UInt8) {
        uint8_t* p = out.data<uint8_t>();
        for (int64_t i = 0; i < numel; ++i) {
            unsigned v = 0;
            is >> v;
            p[i] = static_cast<uint8_t>(v);
        }
    } else if (dt == ::tenzor::DType::UInt16) {
        uint16_t* p = out.data<uint16_t>();
        for (int64_t i = 0; i < numel; ++i) is >> p[i];
    } else if (dt == ::tenzor::DType::UInt32) {
        uint32_t* p = out.data<uint32_t>();
        for (int64_t i = 0; i < numel; ++i) is >> p[i];
    } else if (dt == ::tenzor::DType::UInt64) {
        uint64_t* p = out.data<uint64_t>();
        for (int64_t i = 0; i < numel; ++i) is >> p[i];
    } else if (dt == ::tenzor::DType::Bool) {
        bool* p = out.data<bool>();
        for (int64_t i = 0; i < numel; ++i) {
            int v = 0;
            is >> v;
            p[i] = (v != 0);
        }
    } else if (dt == ::tenzor::DType::Complex64) {
        std::complex<float>* p = out.data<std::complex<float>>();
        for (int64_t i = 0; i < numel; ++i) {
            float re = 0.0f, im = 0.0f;
            is >> re >> im;
            p[i] = std::complex<float>(re, im);
        }
    } else if (dt == ::tenzor::DType::Complex128) {
        std::complex<double>* p = out.data<std::complex<double>>();
        for (int64_t i = 0; i < numel; ++i) {
            double re = 0.0, im = 0.0;
            is >> re >> im;
            p[i] = std::complex<double>(re, im);
        }
    }

    // Short-read guard: iree-run-module elides large outputs (the middle is
    // replaced with an ellipsis when the element count exceeds
    // --output_max_element_count). The stream's first extraction past the
    // non-numeric sentinel sets failbit, after which every remaining `is >> p[i]`
    // is a no-op leaving the buffer tail at the zero-init value. numel comes from
    // the header shape, not the count actually printed, so without this check a
    // truncated output is silently returned as a correctly-shaped tensor with a
    // zero tail. A complete parse leaves the stream non-failed (eofbit is fine).
    if (numel > 0 && is.fail()) {
        throw JitInvokeError(
            "iree-run-module output for a tensor of " + std::to_string(numel) +
            " elements was truncated (short read while parsing values) — raise "
            "--output_max_element_count to cover the output size");
    }
    return out;
}

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

/// Convert an iree_status_t error into a string and free the status object.
auto status_to_string(iree_status_t status) -> std::string {
    if (iree_status_is_ok(status)) return "ok";
    char* buf = nullptr;
    iree_host_size_t len = 0;
    iree_allocator_t alloc = iree_allocator_system();
    if (!iree_status_to_string(status, &alloc, &buf, &len) || !buf) {
        iree_status_ignore(status);
        return "(failed to format status)";
    }
    std::string out(buf, len);
    iree_allocator_free(alloc, buf);
    iree_status_ignore(status);
    return out;
}

#define TENZOR_IREE_CHECK(expr, what)                                          \
    do {                                                                       \
        iree_status_t _s = (expr);                                             \
        if (!iree_status_is_ok(_s)) {                                          \
            throw JitInvokeError(std::string(what) + ": " +                    \
                                 status_to_string(_s));                        \
        }                                                                      \
    } while (0)

}  // namespace

auto IreeInvoker::load(const CompiledArtifact& artifact, Mode mode)
    -> std::unique_ptr<IreeInvoker> {
    if (!fs::exists(artifact.vmfb_path)) {
        throw JitInvokeError("vmfb not found: " + artifact.vmfb_path.string());
    }
    auto inv = std::unique_ptr<IreeInvoker>(new IreeInvoker());
    inv->mode_      = mode;
    inv->vmfb_path_ = artifact.vmfb_path.string();
    inv->target_    = artifact.target;
    inv->device_    = device_for_target(artifact.target);

    if (mode == Mode::Subprocess) {
        inv->iree_run_module_ =
            ::tenzor::jit::mlir_jit::resolve_iree_run_module();
        return inv;
    }

    // ─── InProcess setup ─────────────────────────────────────────────────
    // 1. Create the runtime instance with all available HAL drivers.
    iree_runtime_instance_options_t opts;
    iree_runtime_instance_options_initialize(&opts);
    iree_runtime_instance_options_use_all_available_drivers(&opts);
    iree_runtime_instance_t* instance = nullptr;
    TENZOR_IREE_CHECK(
        iree_runtime_instance_create(&opts, iree_allocator_system(),
                                     &instance),
        "iree_runtime_instance_create");
    inv->instance_ = instance;

    // 2. Acquire the HAL device for the requested driver.
    iree_hal_device_t* device = nullptr;
    iree_string_view_t driver_sv =
        iree_make_string_view(inv->device_.data(), inv->device_.size());
    TENZOR_IREE_CHECK(
        iree_runtime_instance_try_create_default_device(instance, driver_sv,
                                                         &device),
        std::string("iree_runtime_instance_try_create_default_device(") +
            inv->device_ + ")");
    inv->device_handle_ = device;

    // 3. Create the session bound to the device.
    iree_runtime_session_options_t session_opts;
    iree_runtime_session_options_initialize(&session_opts);
    iree_runtime_session_t* session = nullptr;
    TENZOR_IREE_CHECK(
        iree_runtime_session_create_with_device(
            instance, &session_opts, device,
            iree_runtime_instance_host_allocator(instance), &session),
        "iree_runtime_session_create_with_device");
    inv->session_ = session;

    // 4. Append the Tenzor plugin VM module so the upcoming bytecode load
    //    can resolve `tenzor_plugin.<op>` imports.
    iree_vm_module_t* plugin_module = nullptr;
    TENZOR_IREE_CHECK(
        create_tenzor_plugin_module(
            iree_runtime_instance_vm_instance(instance),
            iree_runtime_session_host_allocator(session), device,
            &plugin_module),
        "create_tenzor_plugin_module");
    inv->plugin_module_ = plugin_module;
    TENZOR_IREE_CHECK(
        iree_runtime_session_append_module(session, plugin_module),
        "iree_runtime_session_append_module(tenzor_plugin)");

    // 5. Load the compiled bytecode module — its `tenzor_plugin.<op>`
    //    imports now resolve against the plugin module just appended.
    TENZOR_IREE_CHECK(
        iree_runtime_session_append_bytecode_module_from_file(
            session, inv->vmfb_path_.c_str()),
        "iree_runtime_session_append_bytecode_module_from_file");

    return inv;
}

namespace {

auto invoke_subprocess(IreeInvoker& self,
                       const std::vector<::tenzor::Tensor>& inputs,
                       const std::string& vmfb_path,
                       const std::string& device,
                       const std::string& iree_run_module)
    -> std::vector<::tenzor::Tensor> {
    (void)self;
    std::vector<std::string> argv;
    argv.reserve(inputs.size() + 5);
    argv.push_back(iree_run_module);
    argv.push_back("--device=" + (device == "local-task" ? "local-sync"
                                                         : device));
    argv.push_back("--module=" + vmfb_path);
    argv.push_back("--function=main");
    // 1M was far too small (a single 512x2048 activation already hits the cap),
    // causing iree-run-module to elide the middle of larger outputs and the
    // parser to silently zero-pad the tail. Raise it well past any realistic
    // output size so full tensors are always printed; the short-read guard in
    // parse_output_line still catches any unexpected truncation.
    argv.push_back("--output_max_element_count=2147483647");
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

}  // namespace

auto IreeInvoker::invoke(const std::vector<::tenzor::Tensor>& inputs)
    -> std::vector<::tenzor::Tensor> {
    if (mode_ == Mode::Subprocess) {
        return invoke_subprocess(*this, inputs, vmfb_path_, device_,
                                 iree_run_module_);
    }

    // ─── InProcess invoke ────────────────────────────────────────────────
    auto* session  = static_cast<iree_runtime_session_t*>(session_);
    auto* device   = static_cast<iree_hal_device_t*>(device_handle_);
    auto* allocator = iree_runtime_session_device_allocator(session);

    iree_runtime_call_t call;
    TENZOR_IREE_CHECK(
        iree_runtime_call_initialize_by_name(
            session, iree_make_cstring_view("module.main"), &call),
        "iree_runtime_call_initialize_by_name(module.main)");

    // Push each input as a buffer_view ref.
    for (const auto& t : inputs) {
        iree_hal_buffer_view_t* bv = nullptr;
        TENZOR_IREE_CHECK(
            marshal::tensor_to_buffer_view(device, allocator, t, &bv),
            "marshal::tensor_to_buffer_view");
        iree_status_t s =
            iree_runtime_call_inputs_push_back_buffer_view(&call, bv);
        iree_hal_buffer_view_release(bv);
        if (!iree_status_is_ok(s)) {
            iree_runtime_call_deinitialize(&call);
            throw JitInvokeError(
                "iree_runtime_call_inputs_push_back_buffer_view: " +
                status_to_string(s));
        }
    }

    // Run.
    iree_status_t invoke_s = iree_runtime_call_invoke(&call, 0);
    if (!iree_status_is_ok(invoke_s)) {
        iree_runtime_call_deinitialize(&call);
        throw JitInvokeError("iree_runtime_call_invoke: " +
                             status_to_string(invoke_s));
    }

    // Collect outputs.
    std::vector<::tenzor::Tensor> out_tensors;
    while (true) {
        iree_hal_buffer_view_t* bv = nullptr;
        iree_status_t pop_s =
            iree_runtime_call_outputs_pop_front_buffer_view(&call, &bv);
        if (iree_status_is_out_of_range(pop_s)) {
            iree_status_ignore(pop_s);
            break;
        }
        if (!iree_status_is_ok(pop_s)) {
            iree_runtime_call_deinitialize(&call);
            throw JitInvokeError(
                "iree_runtime_call_outputs_pop_front_buffer_view: " +
                status_to_string(pop_s));
        }
        try {
            out_tensors.push_back(marshal::buffer_view_to_tensor(bv));
        } catch (...) {
            iree_hal_buffer_view_release(bv);
            iree_runtime_call_deinitialize(&call);
            throw;
        }
        iree_hal_buffer_view_release(bv);
    }

    iree_runtime_call_deinitialize(&call);

    if (out_tensors.empty()) {
        throw JitInvokeError(
            "in-process IREE invoke produced no outputs from @main");
    }
    return out_tensors;
}

IreeInvoker::~IreeInvoker() {
    // Release in reverse order of creation. nullptr is safe: each release
    // helper short-circuits on a null pointer.
    if (plugin_module_) {
        iree_vm_module_release(static_cast<iree_vm_module_t*>(plugin_module_));
        plugin_module_ = nullptr;
    }
    if (session_) {
        iree_runtime_session_release(
            static_cast<iree_runtime_session_t*>(session_));
        session_ = nullptr;
    }
    if (device_handle_) {
        iree_hal_device_release(
            static_cast<iree_hal_device_t*>(device_handle_));
        device_handle_ = nullptr;
    }
    if (instance_) {
        iree_runtime_instance_release(
            static_cast<iree_runtime_instance_t*>(instance_));
        instance_ = nullptr;
    }
}

namespace {

// Prepend `dir` to LD_LIBRARY_PATH (or set it if unset). Called before
// the first IREE HIP HAL device probe so dlopen("libamdhip64.so") finds
// a working copy on hosts where /opt/rocm is corrupt. setenv() during
// process startup is safe; LD_LIBRARY_PATH is only consulted by
// dlopen, which the IREE HIP HAL driver does lazily on first use.
auto prepend_to_ld_library_path(const std::string& dir) -> void {
    const char* current = std::getenv("LD_LIBRARY_PATH");
    std::string merged = dir;
    if (current && *current) {
        merged += ':';
        merged += current;
    }
    ::setenv("LD_LIBRARY_PATH", merged.c_str(), /*overwrite=*/1);
}

}  // namespace

auto iree_can_initialize_default_device(const std::string& driver_name)
    -> bool {
    // Per-driver cache: dlopen + device-create costs ~30ms on a warm
    // laptop and tests may probe the same driver dozens of times.
    static std::mutex cache_mu;
    static std::unordered_map<std::string, bool> cache;
    {
        std::lock_guard<std::mutex> g(cache_mu);
        auto it = cache.find(driver_name);
        if (it != cache.end()) return it->second;
    }

    // For the HIP driver: prepend the compile-time-discovered ROCm runtime
    // library directory to LD_LIBRARY_PATH so dlopen("libamdhip64.so")
    // resolves to a working library on hosts where /opt/rocm is corrupt.
    // No-op if TENZOR_ROCM_RUNTIME_LIB_DIR wasn't defined at build time.
    // Must be set before the iree-run-module subprocess fork+exec below
    // and before the in-process HAL driver attempts dlopen.
#ifdef TENZOR_ROCM_RUNTIME_LIB_DIR
    if (driver_name == "hip") {
        prepend_to_ld_library_path(TENZOR_ROCM_RUNTIME_LIB_DIR);
    }
#endif

    // First try the in-process path (cheap when the driver is linked in).
    iree_runtime_instance_options_t opts;
    iree_runtime_instance_options_initialize(&opts);
    iree_runtime_instance_options_use_all_available_drivers(&opts);
    iree_runtime_instance_t* instance = nullptr;
    iree_status_t s = iree_runtime_instance_create(
        &opts, iree_allocator_system(), &instance);
    bool inproc_ok = false;
    if (iree_status_is_ok(s)) {
        iree_hal_device_t* device = nullptr;
        iree_string_view_t driver_sv =
            iree_make_string_view(driver_name.data(), driver_name.size());
        iree_status_t ds = iree_runtime_instance_try_create_default_device(
            instance, driver_sv, &device);
        inproc_ok = iree_status_is_ok(ds);
        if (!inproc_ok) {
            iree_status_ignore(ds);
        }
        if (device) {
            iree_hal_device_release(device);
        }
        iree_runtime_instance_release(instance);
    } else {
        iree_status_ignore(s);
    }

    if (inproc_ok) {
        std::lock_guard<std::mutex> g(cache_mu);
        cache[driver_name] = true;
        return true;
    }

    // The linked-in IREE runtime distributions Tenzor uses commonly omit
    // cuda/hip drivers (they're not in the IREE_HAVE_HAL_*_DRIVER_MODULE
    // defines list at build time). For those targets, IreeInvoker
    // automatically falls back to Mode::Subprocess via iree-run-module —
    // which is a separate binary built with all drivers. To match that
    // gating we probe the subprocess path here too: run `iree-run-module
    // --list_devices=<driver>` and treat any non-empty device line as
    // success.
    //
    // Probe via a short-lived subprocess (exec_capture); the binary returns
    // within a few hundred ms even on a cold ROCm load.
    std::string run_module;
    try {
        run_module = resolve_iree_run_module();
    } catch (...) {
        std::lock_guard<std::mutex> g(cache_mu);
        cache[driver_name] = false;
        return false;
    }
    // Probe directly via fork/exec (no shell): `iree-run-module
    // --list_devices=<driver>`. run_module may come from an env override or an
    // install dir with odd characters; passing argv verbatim removes any
    // shell-metacharacter / injection concern.
    const std::string out =
        exec_capture(run_module, {"--list_devices=" + driver_name},
                     /*capture_stderr=*/false);
    bool subproc_ok = false;
    // Any line containing "://" denotes a device URI; an empty list implies no
    // devices, "FLAGS ERROR" implies the driver couldn't load its underlying
    // vendor library. A spawn/exec failure yields empty output → not ok.
    if (out.find("://") != std::string::npos &&
        out.find("FLAGS ERROR") == std::string::npos) {
        subproc_ok = true;
    }

    std::lock_guard<std::mutex> g(cache_mu);
    cache[driver_name] = subproc_ok;
    return subproc_ok;
}

}  // namespace tenzor::jit::mlir_jit
