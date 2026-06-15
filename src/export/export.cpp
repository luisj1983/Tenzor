/**
 * @file export.cpp
 * @brief Implementation of AOT export for Tenzor models
 *
 * Uses the JIT Tracer to capture a module's forward pass as an IR graph,
 * bundles it with the module's state dict, and provides binary
 * serialization via the existing JIT GraphWriter/GraphReader.
 */

#include "../../include/tenzor/export/export.hpp"
#include "../../include/tenzor/jit/tracer.hpp"
#include "../../include/tenzor/jit/graph.hpp"
#include "../../include/tenzor/jit/serialization.hpp"
#include "../../include/tenzor/autograd/variable.hpp"
#include "../../include/tenzor/backend/loader.hpp"
#include "../../include/tenzor/backend/backend.hpp"
#include <fstream>
#include <limits>
#include <stdexcept>

namespace tenzor {
namespace export_ {

// ============================================================================
// File format constants
// ============================================================================

/// Magic number for Tenzor Exported Program files: "TZEP"
static constexpr uint32_t TZEP_MAGIC   = 0x545A4550;
static constexpr uint32_t TZEP_VERSION = 1;

// ============================================================================
// ExportedProgram::Impl
// ============================================================================

struct ExportedProgram::Impl {
    std::shared_ptr<jit::Graph> graph;
    std::unordered_map<std::string, Tensor> state;
    size_t n_inputs{0};
    size_t n_outputs{0};
};

// ============================================================================
// Binary I/O helpers (local to this TU)
// ============================================================================

namespace {

auto write_uint32(std::ofstream& f, uint32_t v) -> void {
    f.write(reinterpret_cast<const char*>(&v), sizeof(v));
}

auto write_uint64(std::ofstream& f, uint64_t v) -> void {
    f.write(reinterpret_cast<const char*>(&v), sizeof(v));
}

auto write_int64(std::ofstream& f, int64_t v) -> void {
    f.write(reinterpret_cast<const char*>(&v), sizeof(v));
}

auto write_string(std::ofstream& f, const std::string& s) -> void {
    write_uint64(f, s.size());
    f.write(s.data(), static_cast<std::streamsize>(s.size()));
}

auto write_tensor(std::ofstream& f, const Tensor& t) -> void {
    // shape
    const auto& shape = t.shape();
    write_uint64(f, shape.size());
    for (auto dim : shape) write_int64(f, dim);
    // dtype + device (the *original* device the tensor lived on at save
    // time, which load() restores by default).
    write_uint32(f, static_cast<uint32_t>(t.dtype()));
    write_uint32(f, static_cast<uint32_t>(t.device().type));
    write_int64(f, t.device().index);
    // Audit D.2: previously this wrote raw bytes from `t.data_ptr()`
    // regardless of device or stride layout. That was both unsafe (the
    // raw pointer for a non-CPU tensor isn't host-accessible) and
    // non-portable (the receiver couldn't reconstruct a non-contiguous
    // tensor from a packed byte stream that didn't include strides).
    //
    // Now we *always* serialise the host-contiguous form: move the tensor
    // to CPU, materialise contiguous, then write its bytes. The receiver
    // gets a fully-packed buffer and reconstructs the tensor on either
    // the saved device (default) or `map_location` (load argument).
    Tensor host = t.device().type == Device::Type::CPU
                      ? t.contiguous()
                      : t.to(Device::cpu()).contiguous();
    size_t bytes = host.numel() * host.dtype_size();
    write_uint64(f, bytes);
    f.write(reinterpret_cast<const char*>(host.data_ptr()),
            static_cast<std::streamsize>(bytes));
}

auto read_uint32(std::ifstream& f) -> uint32_t {
    uint32_t v = 0;
    f.read(reinterpret_cast<char*>(&v), sizeof(v));
    if (!f || f.gcount() != static_cast<std::streamsize>(sizeof(v))) {
        throw std::runtime_error("TZEP: truncated file (read_uint32)");
    }
    return v;
}

auto read_uint64(std::ifstream& f) -> uint64_t {
    uint64_t v = 0;
    f.read(reinterpret_cast<char*>(&v), sizeof(v));
    if (!f || f.gcount() != static_cast<std::streamsize>(sizeof(v))) {
        throw std::runtime_error("TZEP: truncated file (read_uint64)");
    }
    return v;
}

auto read_int64(std::ifstream& f) -> int64_t {
    int64_t v = 0;
    f.read(reinterpret_cast<char*>(&v), sizeof(v));
    if (!f || f.gcount() != static_cast<std::streamsize>(sizeof(v))) {
        throw std::runtime_error("TZEP: truncated file (read_int64)");
    }
    return v;
}

auto read_string(std::ifstream& f) -> std::string {
    uint64_t len = read_uint64(f);
    // Bound against the remaining file length so a crafted huge len can't
    // trigger an enormous allocation before the read fails.
    std::streampos cur = f.tellg();
    f.seekg(0, std::ios::end);
    std::streampos end = f.tellg();
    f.seekg(cur);
    if (cur < 0 || end < 0 || len > static_cast<uint64_t>(end - cur)) {
        throw std::runtime_error("TZEP read_string: declared size exceeds remaining file");
    }
    std::string s(len, '\0');
    f.read(s.data(), static_cast<std::streamsize>(len));
    if (!f) {
        throw std::runtime_error("TZEP read_string: truncated string");
    }
    return s;
}

auto read_tensor(std::ifstream& f,
                 const std::optional<Device>& map_location) -> Tensor {
    uint64_t ndim = read_uint64(f);
    // Cap ndim so a hostile/corrupt file can't request a giant shape vector.
    if (ndim > 4096) {
        throw std::runtime_error("TZEP read_tensor: implausible tensor rank");
    }
    std::vector<int64_t> shape(ndim);
    for (uint64_t i = 0; i < ndim; ++i) {
        shape[i] = read_int64(f);
        if (shape[i] < 0) {
            throw std::runtime_error("TZEP read_tensor: negative dimension");
        }
    }

    // Validate the raw dtype/device-type integers against the known
    // enumerators BEFORE they flow into dtype_size()/the Tensor ctor. An
    // unrecognized DType makes dtype_size() return 0 (so an empty-shape tensor
    // would slip past the byte-count guard with a bogus dtype), and an
    // out-of-range Device::Type is undefined behaviour when later dispatched.
    uint32_t raw_dtype = read_uint32(f);
    // QInt4x2 is the last enumerator in DType (uint8_t-backed, dense 0..N-1).
    if (raw_dtype > static_cast<uint32_t>(DType::QInt4x2)) {
        throw std::runtime_error("TZEP read_tensor: unknown dtype value " +
                                 std::to_string(raw_dtype));
    }
    DType dtype = static_cast<DType>(raw_dtype);

    uint32_t raw_dev = read_uint32(f);
    if (raw_dev >= static_cast<uint32_t>(Device::Type::COUNT)) {
        throw std::runtime_error("TZEP read_tensor: unknown device type " +
                                 std::to_string(raw_dev));
    }
    auto dev_type = static_cast<Device::Type>(raw_dev);
    int64_t dev_index = read_int64(f);
    Device saved_device(dev_type, dev_index);

    // Read the SEPARATE byte count and validate it BEFORE allocating the
    // tensor. A crafted shape (e.g. [1<<40, 1<<40]) passes the ndim/non-negative
    // checks above but would trigger a massive allocation in the Tensor ctor; so
    // we must (1) compute the expected element/byte count with overflow checks,
    // (2) verify it matches the declared byte count, and (3) bound it against
    // the bytes actually remaining in the file — all before constructing the
    // host tensor.
    uint64_t bytes = read_uint64(f);

    uint64_t numel = 1;
    for (uint64_t i = 0; i < ndim; ++i) {
        if (shape[i] != 0 &&
            numel > std::numeric_limits<uint64_t>::max() /
                        static_cast<uint64_t>(shape[i])) {
            throw std::runtime_error("TZEP read_tensor: element count overflow");
        }
        numel *= static_cast<uint64_t>(shape[i]);
    }
    uint64_t elem_size = static_cast<uint64_t>(dtype_size(dtype));
    if (elem_size != 0 && numel > std::numeric_limits<uint64_t>::max() / elem_size) {
        throw std::runtime_error("TZEP read_tensor: byte size overflow");
    }
    uint64_t expected = numel * elem_size;
    if (bytes != expected) {
        throw std::runtime_error(
            "TZEP read_tensor: byte count (" + std::to_string(bytes) +
            ") does not match tensor size (" + std::to_string(expected) + ")");
    }
    // Bound the declared byte count against the bytes remaining in the file so a
    // hostile size can't force a huge allocation before the read fails.
    std::streampos cur = f.tellg();
    f.seekg(0, std::ios::end);
    std::streampos end = f.tellg();
    f.seekg(cur);
    if (cur < 0 || end < 0 || bytes > static_cast<uint64_t>(end - cur)) {
        throw std::runtime_error(
            "TZEP read_tensor: declared byte count exceeds remaining file");
    }

    // Audit D.2: read the host-contiguous bytes into a CPU tensor first,
    // then move to either `map_location` (caller override) or the saved
    // device. The save side guarantees the bytes are CPU-contiguous, so
    // the read is a pure host-side memcpy that doesn't require the saved
    // device to be available on this machine.
    Tensor host_tensor(shape, dtype, Device::cpu());
    f.read(reinterpret_cast<char*>(host_tensor.data_ptr()),
           static_cast<std::streamsize>(bytes));
    if (!f) {
        throw std::runtime_error("TZEP read_tensor: truncated tensor data");
    }

    Device target = map_location.value_or(saved_device);
    if (target.type == Device::Type::CPU) {
        return host_tensor;
    }
    // The documented contract: if map_location is absent and the saved device
    // is not available on this build/machine, throw a std::runtime_error with a
    // clear message (rather than letting whatever .to(target) raises propagate).
    if (!map_location.has_value()) {
        Backend* backend = backend_registry().get_backend(target.type);
        if (backend == nullptr || !backend->is_available()) {
            throw std::runtime_error(
                "ExportedProgram::load: saved device is not available on this "
                "build; pass map_location to relocate the program to an "
                "available device (e.g. Device::cpu())");
        }
    }
    return host_tensor.to(target);
}

}  // anonymous namespace

// ============================================================================
// ExportedProgram public API
// ============================================================================

auto ExportedProgram::save(const std::string& path) const -> void {
    if (!impl_) {
        throw std::runtime_error("ExportedProgram::save: program is empty");
    }

    // 1. Write the JIT graph to a temporary path using the existing
    //    GraphWriter so we don't reimplement the graph format.
    std::string graph_tmp = path + ".graph.tmp";
    {
        jit::GraphWriter gw(graph_tmp);
        gw.write(*impl_->graph);
    }

    // Read the serialized graph bytes back.
    std::ifstream graph_in(graph_tmp, std::ios::binary | std::ios::ate);
    if (!graph_in.is_open()) {
        throw std::runtime_error("ExportedProgram::save: failed to read temp graph file");
    }
    auto graph_size = static_cast<size_t>(graph_in.tellg());
    graph_in.seekg(0);
    std::vector<char> graph_bytes(graph_size);
    graph_in.read(graph_bytes.data(), static_cast<std::streamsize>(graph_size));
    graph_in.close();
    std::remove(graph_tmp.c_str());

    // 2. Write the TZEP file.
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("ExportedProgram::save: failed to open " + path);
    }

    write_uint32(file, TZEP_MAGIC);
    write_uint32(file, TZEP_VERSION);
    write_uint64(file, impl_->n_inputs);
    write_uint64(file, impl_->n_outputs);

    // State dict
    write_uint64(file, impl_->state.size());
    for (const auto& [name, tensor] : impl_->state) {
        write_string(file, name);
        write_tensor(file, tensor);
    }

    // Graph blob
    write_uint64(file, graph_size);
    file.write(graph_bytes.data(), static_cast<std::streamsize>(graph_size));
}

auto ExportedProgram::load(const std::string& path,
                           std::optional<Device> map_location) -> ExportedProgram {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("ExportedProgram::load: failed to open " + path);
    }

    uint32_t magic = read_uint32(file);
    if (magic != TZEP_MAGIC) {
        throw std::runtime_error("ExportedProgram::load: invalid magic number "
                                 "(expected TZEP format)");
    }

    uint32_t version = read_uint32(file);
    if (version != TZEP_VERSION) {
        throw std::runtime_error(
            "ExportedProgram::load: unsupported version " +
            std::to_string(version) +
            " (this build supports TZEP version " +
            std::to_string(TZEP_VERSION) +
            "). Re-export the program with the current Tenzor build.");
    }

    auto impl = std::make_shared<Impl>();
    impl->n_inputs  = read_uint64(file);
    impl->n_outputs = read_uint64(file);

    // State dict (audit D.2: map_location overrides the saved device on
    // every state tensor — torch.load(..., map_location=...) parity).
    uint64_t num_entries = read_uint64(file);
    for (uint64_t i = 0; i < num_entries; ++i) {
        std::string name = read_string(file);
        Tensor tensor = read_tensor(file, map_location);
        impl->state[std::move(name)] = std::move(tensor);
    }

    // Graph blob -> write to temp file so GraphReader can load it.
    uint64_t graph_size = read_uint64(file);
    // Bound graph_size against the bytes remaining so a crafted huge size can't
    // force an enormous allocation before any data is read (mirrors read_string).
    {
        std::streampos cur = file.tellg();
        file.seekg(0, std::ios::end);
        std::streampos end = file.tellg();
        file.seekg(cur);
        if (cur < 0 || end < 0 ||
            graph_size > static_cast<uint64_t>(end - cur)) {
            throw std::runtime_error(
                "ExportedProgram::load: declared graph size exceeds remaining file");
        }
    }
    std::vector<char> graph_bytes(graph_size);
    file.read(graph_bytes.data(), static_cast<std::streamsize>(graph_size));
    if (file.gcount() != static_cast<std::streamsize>(graph_size)) {
        throw std::runtime_error("ExportedProgram::load: truncated graph blob");
    }

    std::string graph_tmp = path + ".graph.tmp";
    {
        std::ofstream graph_out(graph_tmp, std::ios::binary);
        if (!graph_out.is_open()) {
            throw std::runtime_error("ExportedProgram::load: failed to write temp graph file");
        }
        graph_out.write(graph_bytes.data(), static_cast<std::streamsize>(graph_size));
    }

    jit::GraphReader gr(graph_tmp);
    impl->graph = gr.read();
    std::remove(graph_tmp.c_str());

    ExportedProgram ep;
    ep.impl_ = std::move(impl);
    return ep;
}

auto ExportedProgram::run(const std::vector<Tensor>& inputs) const -> std::vector<Tensor> {
    if (!impl_ || !impl_->graph) {
        throw std::runtime_error("ExportedProgram::run: program is empty");
    }

    // Wrap raw Tensors as Variables (no grad needed for inference).
    std::vector<Variable> var_inputs;
    var_inputs.reserve(inputs.size());
    for (const auto& t : inputs) {
        var_inputs.emplace_back(t, /*requires_grad=*/false);
    }

    // Execute the graph.
    auto var_outputs = impl_->graph->forward(var_inputs);

    // Unwrap Variables back to Tensors.
    std::vector<Tensor> outputs;
    outputs.reserve(var_outputs.size());
    for (auto& v : var_outputs) {
        outputs.push_back(v.tensor());
    }
    return outputs;
}

auto ExportedProgram::state_dict() const -> const std::unordered_map<std::string, Tensor>& {
    if (!impl_) {
        static const std::unordered_map<std::string, Tensor> empty;
        return empty;
    }
    return impl_->state;
}

auto ExportedProgram::num_inputs() const -> size_t {
    return impl_ ? impl_->n_inputs : 0;
}

auto ExportedProgram::num_outputs() const -> size_t {
    return impl_ ? impl_->n_outputs : 0;
}

// ============================================================================
// export_model
// ============================================================================

auto export_model(nn::Module& module,
                  const std::vector<Tensor>& example_inputs,
                  const ExportOptions& opts) -> ExportedProgram {
    if (example_inputs.empty()) {
        throw std::runtime_error("export_model: at least one example input is required");
    }

    // Put the module in eval mode for tracing.
    module.eval();

    // Configure the tracer.
    auto& tracer = jit::Tracer::get_instance();
    tracer.set_strict_mode(opts.strict);

    // Trace using the first input (matches the Module::forward(Variable) API).
    // For models that only consume a single input (the common case), we trace
    // with that. For multi-input models, we trace a lambda.
    std::shared_ptr<jit::Graph> graph;

    if (example_inputs.size() != 1) {
        // Module::forward takes a single Variable, so tracing can only consume
        // one input. The old code silently traced only example_inputs[0] yet
        // recorded n_inputs = example_inputs.size(), producing a program that
        // advertised N inputs but mis-handled all but the first. Fail loudly
        // instead of exporting a broken program.
        throw std::runtime_error(
            "export_model: multi-input export is not supported (got " +
            std::to_string(example_inputs.size()) +
            " inputs); the traced Module::forward consumes a single input");
    }
    {
        Variable input_var(example_inputs[0], /*requires_grad=*/false);
        jit::TracingGuard guard;
        Variable output = module.forward(input_var);
        graph = guard.get_graph({input_var}, {output});
    }

    if (!graph) {
        throw std::runtime_error("export_model: tracing produced no graph");
    }

    // Check for graph breaks in strict mode.
    if (opts.strict && tracer.graph_break_count() > 0) {
        throw std::runtime_error(
            "export_model: strict mode detected " +
            std::to_string(tracer.graph_break_count()) +
            " graph break(s) during tracing. Use ExportOptions{.strict=false} "
            "to allow non-strict export, or refactor the model to avoid "
            "data-dependent control flow.");
    }

    // Capture the module's state dict.
    auto state = module.state_dict();

    // Build the ExportedProgram.
    auto impl = std::make_shared<ExportedProgram::Impl>();
    impl->graph     = std::move(graph);
    impl->state     = std::move(state);
    impl->n_inputs  = example_inputs.size();
    impl->n_outputs = impl->graph->outputs().size();

    ExportedProgram ep;
    ep.impl_ = std::move(impl);
    return ep;
}

}  // namespace export_
}  // namespace tenzor
