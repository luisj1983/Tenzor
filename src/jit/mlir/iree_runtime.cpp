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
#include "tenzor/utils/log.hpp"  // TENZOR_LOG_WARN (F041 skew warning)
#include "tenzor/ops/creation.hpp"

#include <iree/runtime/api.h>
#include <iree/hal/api.h>
#include <iree/vm/api.h>

#include <array>
#include <cctype>
#include <limits>
#include <cerrno>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <filesystem>
#include <fstream>
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
    // Delegate to the single source of truth in iree_paths so the runtime-side
    // driver mapping stays in lockstep with the compile-side one. Previously this
    // rejected metal-spirv / vmvx / vmvx-inline (which compile_mlir / driver_for_
    // target accept), so those targets compiled a valid .vmfb and then died at
    // IreeInvoker::load — an asymmetry between compile and runtime (JIT-F047).
    try {
        return driver_for_target(target);
    } catch (const std::exception& e) {
        throw JitInvokeError("Unsupported IREE target for runtime invoke: " +
                             target);
    }
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

    // Allocate raw storage of the correct dtype (matching _iree_marshal.hpp)
    // rather than ::tenzor::full, whose float/double scalar overload mis-handles
    // integer/complex dtypes; the parse loop below overwrites every element
    // (JIT-016).
    ::tenzor::Tensor out(shape, dt, ::tenzor::Device::cpu());
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

// File-scope (not function-local) so cleanup_shared_iree_state() -- a plain
// atexit callback, which cannot capture -- can reach them. See
// shared_iree_runtime_instance()/shared_iree_hal_device()'s doc comments for
// why these are shared/cached in the first place.
std::mutex g_shared_iree_device_mu;
std::unordered_map<std::string, iree_hal_device_t*> g_shared_iree_devices;
iree_runtime_instance_t* g_shared_iree_instance = nullptr;

// Releases the shared device cache and instance, in that order (devices
// before the instance/driver-registry that created them -- matching IREE's
// own hierarchical ownership). Registered via std::atexit() the first time
// shared_iree_runtime_instance() runs (see there), so this runs during
// normal process termination's atexit-handler phase, which the C/C++
// standard guarantees happens BEFORE the dynamic loader unloads shared
// libraries (__cxa_finalize / _dl_fini).
//
// This is required, not optional cleanliness: leaving these process-
// lifetime objects unreleased (the original, simpler version of this fix)
// leaves any background thread a device's HAL driver started for it --
// e.g. IREE's Vulkan HAL driver starts a "completion watcher" thread per
// device to poll GPU fence completions -- still running at process exit.
// That thread keeps calling into its driver's shared library (here,
// NVIDIA's libnvidia-glcore.so via libvulkan) with no coordination with the
// dynamic loader's unload sequence, so once the loader starts unmapping
// that library the thread's next call into it segfaults (observed
// consistently as a SIGSEGV inside iree_hal_vulkan_completion_watcher_
// thread_main, or a libc++abi "pure virtual function called" abort
// depending on exactly which teardown step the race lands in). Explicitly
// releasing here lets each device's HAL driver properly stop/join that
// thread as part of its own iree_hal_device_release(), before the loader
// ever begins unloading anything.
auto cleanup_shared_iree_state() -> void {
    std::lock_guard<std::mutex> guard(g_shared_iree_device_mu);
    for (auto& [uri, device] : g_shared_iree_devices) {
        if (device) iree_hal_device_release(device);
    }
    g_shared_iree_devices.clear();
    if (g_shared_iree_instance) {
        iree_runtime_instance_release(g_shared_iree_instance);
        g_shared_iree_instance = nullptr;
    }
}

// Returns a single, process-lifetime iree_runtime_instance_t, created lazily
// on first use and released only at process exit (via cleanup_shared_iree_
// state(), registered with std::atexit() below on first creation).
//
// TSAN-confirmed data race (JIT SIGSEGV investigation): IreeInvoker::load()
// used to create a BRAND NEW iree_runtime_instance_t (via
// iree_runtime_instance_create) on every call -- every JIT cache miss, i.e.
// every distinct traced shape/dtype/device. Each instance registers/owns its
// own HAL driver registry, and creating a device from a driver that uses
// IREE's shared task-queue infrastructure (confirmed on both "local-task"
// and "hip" HAL devices, not CPU-driver-specific) spins up background
// worker thread(s) whose teardown (triggered by iree_hal_device_release,
// itself triggered by the PREVIOUS invoker's destructor) is not fully
// synchronous: the worker thread can still be writing internal queue-
// bookkeeping state (iree_hal_task_queue_process_drain /
// _drain_recording) after release() returns. The very next IreeInvoker::load
// call -- the normal pattern immediately after a cache-miss retrace, with no
// gap -- creates a NEW instance/device whose allocations
// (iree_hal_task_queue_initialize, or invoke()'s buffer-marshalling malloc)
// can land on the exact memory address the still-finishing old worker
// thread is writing into: a genuine, if intermittent, heap-corruption bug
// (observed as a SIGSEGV inside malloc's own consistency checks, not just a
// benign TSAN report).
//
// iree_runtime_instance_create's own doc comment already prescribes the fix:
// "Instances should be shared with as many sessions in an application as is
// reasonable to ensure that resources are tracked properly and threads are
// managed correctly." Sharing one instance across every IreeInvoker in the
// process means driver registries/executors are created once and torn down
// only at process exit (never mid-process, so this specific create-right-
// after-destroy race can no longer occur), matching IREE's own intended
// usage pattern rather than Tenzor's previous per-call-fresh-instance one.
auto shared_iree_runtime_instance() -> iree_runtime_instance_t* {
    static iree_runtime_instance_t* const instance = [] {
        iree_runtime_instance_options_t opts;
        iree_runtime_instance_options_initialize(&opts);
        iree_runtime_instance_options_use_all_available_drivers(&opts);
        iree_runtime_instance_t* inst = nullptr;
        TENZOR_IREE_CHECK(
            iree_runtime_instance_create(&opts, iree_allocator_system(),
                                         &inst),
            "iree_runtime_instance_create (shared)");
        g_shared_iree_instance = inst;
        std::atexit(cleanup_shared_iree_state);
        return inst;
    }();
    return instance;
}

// Returns a single, process-lifetime iree_hal_device_t for the given
// driver+ordinal ("device_uri", e.g. "local-sync", "hip://0"), created
// lazily on first use per key and released only at process exit (see
// cleanup_shared_iree_state()).
//
// Sharing the runtime instance alone (shared_iree_runtime_instance()) did
// NOT eliminate the task-queue teardown race described there: TSAN kept
// reporting the identical race afterward, confirming the racy worker
// thread(s) belong to the iree_hal_device_t, not the instance/driver
// registry. IREE creates a fresh executor/thread-pool per device for
// drivers backed by its task-queue infrastructure (confirmed on both
// "local-sync" and "hip" HAL devices -- not CPU-driver-specific), so
// creating a fresh device per IreeInvoker::load() call recreates that
// churn at the device level instead. Caching devices by URI, exactly
// mirroring the instance-sharing rationale (and IREE's own guidance that
// instances/devices should be shared "to ensure resources are tracked
// properly and threads are managed correctly"), means a given driver+
// ordinal's device (and its worker threads, if any) is created once and
// torn down only at process exit.
auto shared_iree_hal_device(iree_runtime_instance_t* instance,
                            const std::string& driver,
                            const std::string& device_uri,
                            int device_index) -> iree_hal_device_t* {
    std::lock_guard<std::mutex> guard(g_shared_iree_device_mu);
    auto& cache = g_shared_iree_devices;
    auto it = cache.find(device_uri);
    if (it != cache.end()) return it->second;

    iree_hal_device_t* device = nullptr;
    if (device_index > 0 && driver != "local-task" && driver != "local-sync") {
        iree_hal_driver_registry_t* registry =
            iree_runtime_instance_driver_registry(instance);
        iree_string_view_t uri_sv =
            iree_make_string_view(device_uri.data(), device_uri.size());
        TENZOR_IREE_CHECK(
            iree_hal_create_device(registry, uri_sv,
                                   iree_runtime_instance_host_allocator(instance),
                                   &device),
            std::string("iree_hal_create_device(") + device_uri + ")");
    } else {
        iree_string_view_t driver_sv =
            iree_make_string_view(driver.data(), driver.size());
        TENZOR_IREE_CHECK(
            iree_runtime_instance_try_create_default_device(instance, driver_sv,
                                                             &device),
            std::string("iree_runtime_instance_try_create_default_device(") +
                driver + ")");
    }
    cache.emplace(device_uri, device);
    return device;
}

}  // namespace

auto IreeInvoker::load(const CompiledArtifact& artifact, Mode mode,
                       int device_index)
    -> std::unique_ptr<IreeInvoker> {
    if (!fs::exists(artifact.vmfb_path)) {
        throw JitInvokeError("vmfb not found: " + artifact.vmfb_path.string());
    }
    auto inv = std::unique_ptr<IreeInvoker>(new IreeInvoker());
    inv->mode_         = mode;
    inv->vmfb_path_    = artifact.vmfb_path.string();
    inv->target_       = artifact.target;
    inv->device_       = device_for_target(artifact.target);
    inv->device_index_ = device_index < 0 ? 0 : device_index;
    // M3: bake the requested ordinal into the device URI so both paths run on
    // the HAL device matching the input's Device::index, not the driver
    // default (GPU 0).
    inv->device_uri_   = hal_device_uri(inv->device_, inv->device_index_);

    if (mode == Mode::Subprocess) {
        inv->iree_run_module_ =
            ::tenzor::jit::mlir_jit::resolve_iree_run_module();
        return inv;
    }

    // ─── InProcess setup ─────────────────────────────────────────────────
    // 1. Acquire the shared, process-lifetime runtime instance (see
    //    shared_iree_runtime_instance()'s doc comment for why this must not
    //    be a fresh instance per invoker).
    iree_runtime_instance_t* instance = shared_iree_runtime_instance();
    iree_runtime_instance_retain(instance);
    inv->instance_ = instance;

    // 2. Acquire the shared, process-lifetime HAL device for this driver+
    //    ordinal (see shared_iree_hal_device()'s doc comment: caching by URI
    //    exactly as done for the instance above -- device-level, not just
    //    instance-level, sharing is what actually eliminates the task-queue
    //    teardown race). For ordinal 0 (the common single-GPU / CPU case)
    //    this uses the driver's default device, preserving the long-tested
    //    path. For a non-zero ordinal (M3) it selects the specific device by
    //    URI ("cuda://1", "hip://1", "vulkan://1") via the instance's driver
    //    registry so a cuda:1 model actually runs on GPU 1.
    iree_hal_device_t* device = shared_iree_hal_device(
        instance, inv->device_, inv->device_uri_, inv->device_index_);
    iree_hal_device_retain(device);
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

// Map a NumPy dtype descr (e.g. "<f4") to a Tenzor DType. All supported hosts
// are little-endian; a big-endian ('>') multi-byte descr would need byte-
// swapping, which iree-run-module never emits on these platforms — reject it.
auto npy_descr_to_dtype(const std::string& descr) -> ::tenzor::DType {
    using ::tenzor::DType;
    std::string t = descr;
    if (!t.empty() && (t[0] == '<' || t[0] == '|' || t[0] == '=')) {
        t = t.substr(1);
    } else if (!t.empty() && t[0] == '>') {
        throw JitInvokeError(
            "iree-run-module: big-endian .npy output '" + descr +
            "' unsupported on this little-endian host");
    }
    if (t == "f2")  return DType::Float16;
    if (t == "f4")  return DType::Float32;
    if (t == "f8")  return DType::Float64;
    if (t == "i1")  return DType::Int8;
    if (t == "i2")  return DType::Int16;
    if (t == "i4")  return DType::Int32;
    if (t == "i8")  return DType::Int64;
    if (t == "u1")  return DType::UInt8;
    if (t == "u2")  return DType::UInt16;
    if (t == "u4")  return DType::UInt32;
    if (t == "u8")  return DType::UInt64;
    if (t == "b1")  return DType::Bool;
    if (t == "c8")  return DType::Complex64;
    if (t == "c16") return DType::Complex128;
    throw JitInvokeError("iree-run-module: unsupported .npy descr '" + descr + "'");
}

// Read a NumPy .npy file (little-endian, C order) into a CPU Tensor. Used for
// bit-exact subprocess output: iree-run-module's stdout printing rounds floats
// to ~6 significant figures, silently losing precision for f32/f64 relative to
// the bit-exact InProcess buffer-view path.
auto read_npy_output(const std::string& path) -> ::tenzor::Tensor {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        throw JitInvokeError("iree-run-module: cannot open output .npy: " + path);
    }
    char magic[6] = {};
    f.read(magic, 6);
    static const char kMagic[6] = {'\x93', 'N', 'U', 'M', 'P', 'Y'};
    if (!f || std::memcmp(magic, kMagic, 6) != 0) {
        throw JitInvokeError("iree-run-module: bad .npy magic in " + path);
    }
    unsigned char vmaj = 0, vmin = 0;
    f.read(reinterpret_cast<char*>(&vmaj), 1);
    f.read(reinterpret_cast<char*>(&vmin), 1);
    (void)vmin;
    uint32_t hlen = 0;
    if (vmaj >= 2) {
        unsigned char b[4] = {};
        f.read(reinterpret_cast<char*>(b), 4);
        hlen = uint32_t(b[0]) | (uint32_t(b[1]) << 8) |
               (uint32_t(b[2]) << 16) | (uint32_t(b[3]) << 24);
    } else {
        unsigned char b[2] = {};
        f.read(reinterpret_cast<char*>(b), 2);
        hlen = uint32_t(b[0]) | (uint32_t(b[1]) << 8);
    }
    std::string header(hlen, '\0');
    f.read(header.data(), static_cast<std::streamsize>(hlen));
    if (!f) {
        throw JitInvokeError("iree-run-module: truncated .npy header in " + path);
    }
    if (header.find("'fortran_order': True") != std::string::npos) {
        throw JitInvokeError("iree-run-module: fortran-order .npy unsupported: " + path);
    }

    auto extract_quoted = [&](const char* key) -> std::string {
        auto k = header.find(key);
        if (k == std::string::npos) return {};
        auto q1 = header.find('\'', k + std::strlen(key));
        if (q1 == std::string::npos) return {};
        auto q2 = header.find('\'', q1 + 1);
        if (q2 == std::string::npos) return {};
        return header.substr(q1 + 1, q2 - q1 - 1);
    };
    const ::tenzor::DType dt = npy_descr_to_dtype(extract_quoted("'descr':"));

    std::vector<int64_t> shape;
    {
        auto s = header.find("'shape':");
        auto lp = (s == std::string::npos) ? std::string::npos : header.find('(', s);
        auto rp = (lp == std::string::npos) ? std::string::npos : header.find(')', lp);
        if (lp == std::string::npos || rp == std::string::npos) {
            throw JitInvokeError("iree-run-module: bad .npy shape header in " + path);
        }
        std::istringstream ss(header.substr(lp + 1, rp - lp - 1));
        std::string tok;
        while (std::getline(ss, tok, ',')) {
            const size_t a = tok.find_first_not_of(" \t");
            if (a == std::string::npos) continue;
            const size_t b = tok.find_last_not_of(" \t");
            shape.push_back(std::stoll(tok.substr(a, b - a + 1)));
        }
    }

    int64_t numel = 1;
    for (auto dv : shape) numel *= dv;
    ::tenzor::Tensor out(shape, dt, ::tenzor::Device::cpu());
    const std::streamsize nbytes =
        static_cast<std::streamsize>(numel) *
        static_cast<std::streamsize>(::tenzor::dtype_size(dt));
    f.read(reinterpret_cast<char*>(out.data_ptr()), nbytes);
    if (f.gcount() != nbytes) {
        throw JitInvokeError("iree-run-module: short .npy data read in " + path);
    }
    return out;
}

// Inverse of npy_descr_to_dtype. Returns "" for dtypes with no standard NumPy
// descr (e.g. BFloat16), which then use the inline ASCII input path.
auto dtype_to_npy_descr(::tenzor::DType d) -> std::string {
    using ::tenzor::DType;
    switch (d) {
        case DType::Float16:    return "<f2";
        case DType::Float32:    return "<f4";
        case DType::Float64:    return "<f8";
        case DType::Int8:       return "<i1";
        case DType::Int16:      return "<i2";
        case DType::Int32:      return "<i4";
        case DType::Int64:      return "<i8";
        case DType::UInt8:      return "<u1";
        case DType::UInt16:     return "<u2";
        case DType::UInt32:     return "<u4";
        case DType::UInt64:     return "<u8";
        case DType::Bool:       return "|b1";
        case DType::Complex64:  return "<c8";
        case DType::Complex128: return "<c16";
        default:                return {};
    }
}

// Write a CPU, contiguous tensor as a little-endian C-order NumPy .npy (v1.0).
// The exact binary buffer is written, so all bits survive — unlike the ASCII
// --input= rendering (NaN payload/sign, Inf; JIT-F045) — and there is no
// command-line length limit, which fixes the "Argument list too long" execv
// failure that silently degraded large-tensor GPU subprocess runs to eager.
auto write_npy(const ::tenzor::Tensor& t, const std::string& path) -> bool {
    const std::string descr = dtype_to_npy_descr(t.dtype());
    if (descr.empty()) return false;
    const ::tenzor::Tensor cpu = t.cpu().contiguous();
    const auto shp = cpu.shape();
    std::string dims;
    for (size_t i = 0; i < shp.size(); ++i) {
        if (i) dims += ", ";
        dims += std::to_string(shp[i]);
    }
    if (shp.size() == 1) dims += ",";
    std::string header = "{'descr': '" + descr +
        "', 'fortran_order': False, 'shape': (" + dims + "), }";
    // Pad the header so (10-byte preamble + header + '\n') is a 64-byte multiple.
    const size_t base = 10 + header.size() + 1;
    header.append((64 - (base % 64)) % 64, ' ');
    header += '\n';
    const auto hlen = static_cast<uint16_t>(header.size());
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    static const char kMagic[6] = {'\x93', 'N', 'U', 'M', 'P', 'Y'};
    f.write(kMagic, 6);
    const char ver[2] = {1, 0};
    f.write(ver, 2);
    const char lb[2] = {static_cast<char>(hlen & 0xff),
                        static_cast<char>((hlen >> 8) & 0xff)};
    f.write(lb, 2);
    f.write(header.data(), static_cast<std::streamsize>(header.size()));
    const int64_t nbytes =
        cpu.numel() * static_cast<int64_t>(::tenzor::dtype_size(cpu.dtype()));
    f.write(reinterpret_cast<const char*>(cpu.data_ptr()),
            static_cast<std::streamsize>(nbytes));
    return static_cast<bool>(f);
}

// Write a tensor's raw contiguous bytes (no header) for the
// `--input=SHAPExTYPE=@FILE` form, where the shape/dtype travel in the flag
// text itself and FILE supplies only the value bytes. Distinct from
// write_npy's `--input=@FILE.npy` form (whole-file numpy header, shape/dtype
// inferred from the file).
auto write_raw_binary(const ::tenzor::Tensor& t, const std::string& path) -> bool {
    const ::tenzor::Tensor cpu = t.cpu().contiguous();
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    const int64_t nbytes =
        cpu.numel() * static_cast<int64_t>(::tenzor::dtype_size(cpu.dtype()));
    f.write(reinterpret_cast<const char*>(cpu.data_ptr()),
            static_cast<std::streamsize>(nbytes));
    return static_cast<bool>(f);
}

// Build one --input flag. A large operand of an NPY-representable dtype is
// written to a temp .npy and passed as --input=@path (exact bits, no ARG_MAX
// limit). A large operand whose dtype has no NPY encoding in
// iree-run-module's numpy_io (BFloat16 -- see invoke_subprocess's output-side
// comment on the same limitation) instead uses the raw-binary
// `--input=SHAPExTYPE=@path` form, which carries shape/dtype in the flag
// text and only the value bytes in the file -- still exact bits, still no
// ARG_MAX growth with tensor size. Only genuinely small operands (or a dtype
// with neither encoding available) fall through to the inline ASCII form.
// Temp paths are pushed onto `temp_files` for the caller to clean up after
// the subprocess returns.
auto build_input_arg(const ::tenzor::Tensor& t,
                     std::vector<std::string>& temp_files) -> std::string {
    const std::string descr = dtype_to_npy_descr(t.dtype());
    if (t.numel() > 4096 && !descr.empty()) {
        auto tmpl = (std::filesystem::temp_directory_path() /
                     "tenzor_iree_in_XXXXXX.npy").string();
        std::vector<char> buf(tmpl.begin(), tmpl.end());
        buf.push_back('\0');
        const int fd = ::mkstemps(buf.data(), 4);
        if (fd >= 0) {
            ::close(fd);
            std::string p(buf.data());
            if (write_npy(t, p)) {
                temp_files.push_back(p);
                return "--input=@" + p;
            }
            std::remove(p.c_str());
        }
    }
    if (t.numel() > 4096 && t.dtype() == ::tenzor::DType::BFloat16) {
        auto tmpl = (std::filesystem::temp_directory_path() /
                     "tenzor_iree_in_XXXXXX.bin").string();
        std::vector<char> buf(tmpl.begin(), tmpl.end());
        buf.push_back('\0');
        const int fd = ::mkstemps(buf.data(), 4);
        if (fd >= 0) {
            ::close(fd);
            std::string p(buf.data());
            if (write_raw_binary(t, p)) {
                temp_files.push_back(p);
                std::ostringstream os;
                os << "--input=";
                for (auto d : t.shape()) os << d << 'x';
                os << dtype_to_iree_element(t.dtype()) << "=@" << p;
                return os.str();
            }
            std::remove(p.c_str());
        }
    }
    return render_input_flag(t);
}

auto invoke_subprocess(IreeInvoker& self,
                       const std::vector<::tenzor::Tensor>& inputs,
                       const std::string& vmfb_path,
                       const std::string& device,
                       const std::string& iree_run_module,
                       int expected_outputs)
    -> std::vector<::tenzor::Tensor> {
    (void)self;
    // F041: warn once if iree-compile (which produced the .vmfb) and
    // iree-run-module (which loads it) resolve from DIFFERENT directories. A
    // partial install — e.g. a .venv-iree shipping only the compiler wheel while
    // run-module comes from $PATH — can pair a compiler and runtime from
    // different IREE builds whose bytecode formats are incompatible, surfacing as
    // an opaque module-load failure that reproduces only on that host. Cheap
    // parent-dir comparison; no subprocess.
    static std::once_flag skew_warned;
    std::call_once(skew_warned, [&]() {
        try {
            const std::string& compile_bin =
                ::tenzor::jit::mlir_jit::resolve_iree_compile();
            if (!compile_bin.empty() && !iree_run_module.empty()) {
                auto cdir = std::filesystem::path(compile_bin).parent_path();
                auto rdir = std::filesystem::path(iree_run_module).parent_path();
                if (cdir != rdir) {
                    TENZOR_LOG_WARN(
                        "IREE JIT: iree-compile (" + cdir.string() +
                        ") and iree-run-module (" + rdir.string() +
                        ") resolve from different directories; if they come from "
                        "different IREE builds a vmfb compiled by one may fail to "
                        "load in the other. Install both from the same IREE "
                        "distribution to avoid opaque module-load errors.");
                }
            }
        } catch (...) {
        }
    });
    auto base_argv = [&]() {
        std::vector<std::string> argv;
        argv.reserve(inputs.size() + 6);
        argv.push_back(iree_run_module);
        // driver_for_target("llvm-cpu") now always resolves to "local-sync"
        // (see its doc comment: IREE's multithreaded "local-task" driver has
        // a genuine, TSAN-confirmed data race in its own worker-thread
        // teardown that a fresh IreeInvoker created shortly after a previous
        // one's release can lose to -- observed as an intermittent heap-
        // corruption SIGSEGV). This ternary is therefore currently a no-op
        // for llvm-cpu specifically; kept as a defensive downgrade in case a
        // future target ever maps to "local-task" here, since a short-lived
        // `iree-run-module` subprocess (one call and exit) would pay full
        // thread-pool spin-up/teardown cost with no amortization even if
        // local-task were otherwise safe to use. Numeric agreement between
        // InProcess and Subprocess is verified by
        // JitRocmArch.InProcessSubprocessParity_ReductionOnLlvmCpu.
        argv.push_back("--device=" + (device == "local-task" ? "local-sync"
                                                             : device));
        argv.push_back("--module=" + vmfb_path);
        argv.push_back("--function=main");
        return argv;
    };

    // Bit-exact output via a temp NumPy .npy (iree-run-module --output=@path):
    // stdout printing rounds floats to ~6 significant figures, silently losing
    // precision for f32/f64 relative to the InProcess buffer-view path. @main is
    // compiled to a single output (enforced by mlir_invoke_impl), so one file
    // suffices. bf16 has no numpy encoding in iree-run-module's numpy_io — for
    // it we fall back to stdout parsing, whose ~6 sig figs already exceed bf16's
    // ~3-digit mantissa precision, so no bits are lost.
    std::string npy_path;
    {
        // The temp path MUST end in ".npy": iree-run-module's `--output=@path`
        // selects the output serialization from the file EXTENSION. Without
        // ".npy" it writes a raw little-endian buffer dump, which read_npy_output
        // then rejects as "bad .npy magic" — so the run silently fell back to
        // eager (this path is the only route for targets whose in-process HAL
        // isn't linked, e.g. ROCm, meaning ROCm JIT never actually executed).
        // mkstemps() keeps the fixed ".npy" suffix (4 chars) while randomizing
        // the XXXXXX template.
        std::string tmpl = (std::filesystem::temp_directory_path() /
                            "tenzor_iree_out_XXXXXX.npy").string();
        std::vector<char> buf(tmpl.begin(), tmpl.end());
        buf.push_back('\0');
        int fd = ::mkstemps(buf.data(), 4);
        if (fd >= 0) {
            ::close(fd);
            npy_path.assign(buf.data());
        }
    }
    struct FileGuard {
        std::string p;
        ~FileGuard() { if (!p.empty()) std::remove(p.c_str()); }
    } guard{npy_path};
    // Temp .npy input files written by build_input_arg for large operands; cleaned
    // up when this guard leaves scope (after both subprocess attempts below).
    std::vector<std::string> input_temps;
    struct InputTempGuard {
        std::vector<std::string>& v;
        ~InputTempGuard() {
            for (const auto& p : v) std::remove(p.c_str());
        }
    } input_guard{input_temps};

    if (!npy_path.empty()) {
        // F003: write ONE --output=@file per result so a multi-output @main
        // returns ALL results on the subprocess path (the in-process path already
        // does). npy_path is the first; extra outputs get their own temp files,
        // cleaned up with the inputs at scope exit.
        std::vector<std::string> out_paths;
        out_paths.push_back(npy_path);
        for (int j = 1; j < expected_outputs; ++j) {
            auto tmpl = (std::filesystem::temp_directory_path() /
                         "tenzor_iree_out_XXXXXX.npy").string();
            std::vector<char> b(tmpl.begin(), tmpl.end());
            b.push_back('\0');
            const int fd = ::mkstemps(b.data(), 4);
            if (fd >= 0) {
                ::close(fd);
                out_paths.emplace_back(b.data());
                input_temps.push_back(out_paths.back());  // reuse the temp guard
            }
        }
        std::vector<std::string> argv = base_argv();
        for (const auto& op : out_paths) {
            argv.push_back("--output=@" + op);
        }
        for (const auto& t : inputs) {
            argv.push_back(build_input_arg(t, input_temps));
        }
        SubprocessResult res = run_subprocess(argv);
        if (res.exit_code == 0) {
            std::vector<::tenzor::Tensor> outs;
            outs.reserve(out_paths.size());
            for (const auto& op : out_paths) {
                outs.push_back(read_npy_output(op));
            }
            return outs;
        }
        // Only the known numpy-encoding gap (bf16) is retried via stdout; any
        // other non-zero exit is a genuine failure.
        if (res.stderr_text.find("unsupported data encoding") ==
            std::string::npos) {
            throw JitInvokeError(
                "iree-run-module exit=" + std::to_string(res.exit_code) +
                "\nstderr:\n" + res.stderr_text +
                "\nstdout:\n" + res.stdout_text);
        }
    }

    // Fallback: stdout ASCII parse (bf16, or if no temp file could be created).
    // --output_max_element_count is raised well past any realistic output size
    // so full tensors print (the short-read guard in parse_output_line catches
    // any unexpected truncation).
    std::vector<std::string> argv = base_argv();
    argv.push_back("--output_max_element_count=2147483647");
    for (const auto& t : inputs) {
        argv.push_back(build_input_arg(t, input_temps));
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
    // R1-06: this ASCII-stdout fallback (only reached because at least one
    // output's dtype has no numpy encoding in iree-run-module -- currently
    // only BFloat16 -- or a temp file couldn't be created) rounds every
    // value to ~6 significant figures. That's lossless for BFloat16
    // (~3-digit mantissa) and for integer/bool dtypes (printed exactly), but
    // would SILENTLY discard real precision for Float32 (~7 digits needed),
    // Float64 (~15-17 digits), or Complex64/128 (wrapping those). With a
    // single output this can't currently happen in production (@main is
    // enforced to have exactly one output, and that one output is whatever
    // triggered this fallback), but guard multi-output explicitly rather
    // than silently return a precision-degraded tensor if that invariant is
    // ever loosened.
    if (outs.size() > 1) {
        for (std::size_t i = 0; i < outs.size(); ++i) {
            const auto dt = outs[i].dtype();
            if (dt == ::tenzor::DType::Float32 || dt == ::tenzor::DType::Float64 ||
                dt == ::tenzor::DType::Complex64 || dt == ::tenzor::DType::Complex128) {
                throw JitInvokeError(
                    "iree-run-module: multi-output subprocess invocation fell "
                    "back to lossy ~6-significant-figure ASCII parsing (likely "
                    "because a co-produced BFloat16 output has no numpy "
                    "encoding in iree-run-module), which would silently "
                    "discard real precision for output[" + std::to_string(i) +
                    "] (dtype requires full precision). Refusing to return a "
                    "silently precision-degraded result.");
            }
        }
    }
    return outs;
}

}  // namespace

auto IreeInvoker::invoke(const std::vector<::tenzor::Tensor>& inputs)
    -> std::vector<::tenzor::Tensor> {
    // Serialize: the IREE runtime session is not concurrency-safe and this
    // invoker is shared across callers via the compile cache.
    std::lock_guard<std::mutex> lock(invoke_mutex_);
    if (mode_ == Mode::Subprocess) {
        // Ordinal 0: pass the bare driver name (long-tested default-device
        // path). Non-zero ordinal (M3): pass the full "driver://<ordinal>"
        // URI so iree-run-module selects the requested GPU.
        const std::string& dev =
            (device_index_ > 0) ? device_uri_ : device_;
        return invoke_subprocess(*this, inputs, vmfb_path_, dev,
                                 iree_run_module_, expected_outputs_);
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

    // RAII: deinitialize the call on EVERY exit path, including a C++ exception
    // thrown by marshalling (marshal::tensor_to_buffer_view -> dtype_to_iree
    // throws std::invalid_argument for an unsupported dtype). The manual
    // deinitialize calls in the status-error paths below are removed in favor of
    // this guard so the call state + any pushed buffer_views are never leaked.
    struct CallGuard {
        iree_runtime_call_t* c;
        ~CallGuard() { iree_runtime_call_deinitialize(c); }
    } call_guard{&call};

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
            throw JitInvokeError(
                "iree_runtime_call_inputs_push_back_buffer_view: " +
                status_to_string(s));
        }
    }

    // Run.
    iree_status_t invoke_s = iree_runtime_call_invoke(&call, 0);
    if (!iree_status_is_ok(invoke_s)) {
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
            throw JitInvokeError(
                "iree_runtime_call_outputs_pop_front_buffer_view: " +
                status_to_string(pop_s));
        }
        try {
            out_tensors.push_back(marshal::buffer_view_to_tensor(bv));
        } catch (...) {
            iree_hal_buffer_view_release(bv);
            throw;
        }
        iree_hal_buffer_view_release(bv);
    }

    // call is deinitialized by call_guard on return.
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

auto hal_device_uri(const std::string& driver, int device_index)
    -> std::string {
    // The CPU HAL drivers have no device ordinal — a URI path would be
    // rejected. GPU drivers select a specific device via `driver://<ordinal>`.
    if (driver == "local-task" || driver == "local-sync") {
        return driver;
    }
    return driver + "://" + std::to_string(device_index < 0 ? 0 : device_index);
}

namespace {

/// Locate a ROCm device-arch enumerator (`amdgpu-arch`, else
/// `rocm_agent_enumerator`). Returns an absolute path or "" if none is found.
auto find_rocm_arch_tool() -> std::string {
    auto usable = [](const std::string& p) {
        return !p.empty() && access(p.c_str(), X_OK) == 0;
    };
    std::vector<std::string> roots;
#ifdef TENZOR_ROCM_RUNTIME_LIB_DIR
    {
        const fs::path lib{TENZOR_ROCM_RUNTIME_LIB_DIR};  // e.g. /opt/rocm/lib
        roots.push_back(lib.parent_path().string());       // e.g. /opt/rocm
        roots.push_back(lib.string());
    }
#endif
    if (const char* r = std::getenv("ROCM_PATH"); r != nullptr && *r != '\0') {
        roots.push_back(r);
    }
    roots.push_back("/opt/rocm");

    // amdgpu-arch prints exactly one gfx name per GPU, in HSA/KFD device
    // order (i.e. honoring ROCR_VISIBLE_DEVICES/GPU_DEVICE_ORDINAL, but not
    // HIP_VISIBLE_DEVICES). detect_rocm_gfx_arch() remaps the caller's HIP
    // ordinal into this order before indexing.
    for (const auto& root : roots) {
        for (const char* sub : {"/llvm/bin/amdgpu-arch",
                                "/lib/llvm/bin/amdgpu-arch",
                                "/bin/amdgpu-arch"}) {
            const std::string cand = root + sub;
            if (usable(cand)) return cand;
        }
    }
    // rocm_agent_enumerator additionally lists the host CPU agent ("gfx000"),
    // filtered out below.
    for (const auto& root : roots) {
        const std::string cand = root + "/bin/rocm_agent_enumerator";
        if (usable(cand)) return cand;
    }
    return {};
}

}  // namespace

auto remap_hip_visible_device_index(int hip_index,
                                     const char* hip_visible_devices) -> int {
    if (hip_index < 0 || hip_visible_devices == nullptr ||
        *hip_visible_devices == '\0') {
        return hip_index;
    }
    std::vector<int> visible;
    std::istringstream vs(hip_visible_devices);
    std::string tok;
    while (std::getline(vs, tok, ',')) {
        const auto b = tok.find_first_not_of(" \t\r\n");
        if (b == std::string::npos) continue;
        const auto e = tok.find_last_not_of(" \t\r\n");
        const std::string trimmed = tok.substr(b, e - b + 1);
        try {
            std::size_t pos = 0;
            const int parsed = std::stoi(trimmed, &pos);
            if (pos != trimmed.size() || parsed < 0) {
                // HIP_VISIBLE_DEVICES parsing stops at the first invalid
                // entry (matching HIP runtime behavior); anything already
                // parsed before it remains valid.
                break;
            }
            visible.push_back(parsed);
        } catch (...) {
            break;
        }
    }
    if (hip_index >= static_cast<int>(visible.size())) {
        return hip_index;
    }
    return visible[static_cast<std::size_t>(hip_index)];
}

auto detect_rocm_gfx_arch(int device_index) -> std::string {
    if (device_index < 0) return {};
    static std::mutex mu;
    static std::unordered_map<int, std::string> cache;
    {
        std::lock_guard<std::mutex> g(mu);
        auto it = cache.find(device_index);
        if (it != cache.end()) return it->second;
    }

    std::string result;
    try {
        const std::string tool = find_rocm_arch_tool();
        if (!tool.empty()) {
            const std::string out =
                exec_capture(tool, {}, /*capture_stderr=*/false);
            std::vector<std::string> archs;
            std::istringstream is(out);
            std::string line;
            while (std::getline(is, line)) {
                const auto b = line.find_first_not_of(" \t\r\n");
                if (b == std::string::npos) continue;
                const auto e = line.find_last_not_of(" \t\r\n");
                std::string tok = line.substr(b, e - b + 1);
                // Keep gfx ISA names; drop the CPU agent that
                // rocm_agent_enumerator emits as "gfx000".
                if (tok.rfind("gfx", 0) == 0 && tok != "gfx000") {
                    archs.push_back(std::move(tok));
                }
            }
            const int physical_index = remap_hip_visible_device_index(
                device_index, std::getenv("HIP_VISIBLE_DEVICES"));
            if (physical_index >= 0 &&
                physical_index < static_cast<int>(archs.size())) {
                result = archs[static_cast<std::size_t>(physical_index)];
            }
        }
    } catch (...) {
        // Best-effort: on any failure leave `result` empty so the caller
        // falls back to the build-time default arch.
        result.clear();
    }

    std::lock_guard<std::mutex> g(mu);
    cache[device_index] = result;
    return result;
}

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
    //
    // Acquire (and cache) the SAME shared, process-lifetime instance/device
    // that a real IreeInvoker::load() would use for this driver at ordinal
    // 0, rather than creating a throwaway instance+device and immediately
    // releasing them. A probe-only device is created and torn down just as
    // fast as a real one -- including spinning up whatever background
    // thread(s) the driver's device constructor starts (e.g. Vulkan's HAL
    // driver starts a completion-watcher thread per device) -- and
    // immediately releasing it hits the exact same task-queue/completion-
    // watcher teardown race documented on shared_iree_hal_device(), just
    // with a probe-scoped device instead of an invoker-scoped one (observed
    // as a "pure virtual function called" abort / SIGSEGV inside the
    // Vulkan driver's completion-watcher thread at process exit, once the
    // invoker-level race above was fixed and the process ran far enough to
    // reach this probe's teardown). Sharing here both avoids a second,
    // separate instance of the bug class and means a probe that succeeds is
    // never wasted work -- the cached device is reused by the first real
    // invocation for this driver.
    bool inproc_ok = false;
    try {
        iree_runtime_instance_t* instance = shared_iree_runtime_instance();
        iree_hal_device_t* device = shared_iree_hal_device(
            instance, driver_name, hal_device_uri(driver_name, 0), 0);
        inproc_ok = device != nullptr;
    } catch (...) {
        inproc_ok = false;
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
