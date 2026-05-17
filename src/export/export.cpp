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
#include <fstream>
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
    // dtype + device
    write_uint32(f, static_cast<uint32_t>(t.dtype()));
    write_uint32(f, static_cast<uint32_t>(t.device().type));
    write_int64(f, t.device().index);
    // raw data
    size_t bytes = t.numel() * t.dtype_size();
    write_uint64(f, bytes);
    f.write(reinterpret_cast<const char*>(t.data_ptr()), static_cast<std::streamsize>(bytes));
}

auto read_uint32(std::ifstream& f) -> uint32_t {
    uint32_t v;
    f.read(reinterpret_cast<char*>(&v), sizeof(v));
    return v;
}

auto read_uint64(std::ifstream& f) -> uint64_t {
    uint64_t v;
    f.read(reinterpret_cast<char*>(&v), sizeof(v));
    return v;
}

auto read_int64(std::ifstream& f) -> int64_t {
    int64_t v;
    f.read(reinterpret_cast<char*>(&v), sizeof(v));
    return v;
}

auto read_string(std::ifstream& f) -> std::string {
    uint64_t len = read_uint64(f);
    std::string s(len, '\0');
    f.read(s.data(), static_cast<std::streamsize>(len));
    return s;
}

auto read_tensor(std::ifstream& f) -> Tensor {
    uint64_t ndim = read_uint64(f);
    std::vector<int64_t> shape(ndim);
    for (uint64_t i = 0; i < ndim; ++i) shape[i] = read_int64(f);

    DType dtype = static_cast<DType>(read_uint32(f));
    auto dev_type = static_cast<Device::Type>(read_uint32(f));
    int64_t dev_index = read_int64(f);
    Device device(dev_type, dev_index);

    Tensor tensor(shape, dtype, device);
    uint64_t bytes = read_uint64(f);
    f.read(reinterpret_cast<char*>(tensor.data_ptr()), static_cast<std::streamsize>(bytes));
    return tensor;
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

auto ExportedProgram::load(const std::string& path) -> ExportedProgram {
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

    // State dict
    uint64_t num_entries = read_uint64(file);
    for (uint64_t i = 0; i < num_entries; ++i) {
        std::string name = read_string(file);
        Tensor tensor = read_tensor(file);
        impl->state[std::move(name)] = std::move(tensor);
    }

    // Graph blob -> write to temp file so GraphReader can load it.
    uint64_t graph_size = read_uint64(file);
    std::vector<char> graph_bytes(graph_size);
    file.read(graph_bytes.data(), static_cast<std::streamsize>(graph_size));

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

    if (example_inputs.size() == 1) {
        Variable input_var(example_inputs[0], /*requires_grad=*/false);
        jit::TracingGuard guard;
        Variable output = module.forward(input_var);
        graph = guard.get_graph({input_var}, {output});
    } else {
        // Wrap all inputs as Variables, trace through a lambda that calls
        // forward with the first input (Module::forward takes one Variable).
        // This is a limitation of the current Module API; future versions
        // may support multi-input forward.
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
