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

    for (auto& c : values) {
        if (c == '[' || c == ']' || c == ',') c = ' ';
    }

    ::tenzor::Tensor out;
    if (dt == ::tenzor::DType::Float32) {
        out = ::tenzor::full(shape, 0.0f, dt);
    } else if (dt == ::tenzor::DType::Float64) {
        out = ::tenzor::full(shape, 0.0, dt);
    } else if (dt == ::tenzor::DType::Int32) {
        out = ::tenzor::full(shape, 0.0f, dt);
    } else if (dt == ::tenzor::DType::Int64) {
        out = ::tenzor::full(shape, 0.0f, dt);
    }
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

}  // namespace tenzor::jit::mlir_jit
