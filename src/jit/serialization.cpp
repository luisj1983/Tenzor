/**
 * @file serialization.cpp
 * @brief Implementation of graph serialization and deserialization
 */

#include "../../include/tenzor/jit/serialization.hpp"
#include "../../include/tenzor/jit/tracer.hpp"
#include <stdexcept>
#include <sstream>
#include <iomanip>

namespace tenzor {
namespace jit {

// ============================================================================
// GraphWriter Implementation
// ============================================================================

GraphWriter::GraphWriter(const std::string& path) : file_(path, std::ios::binary) {
    if (!file_.is_open()) {
        throw std::runtime_error("Failed to open file for writing: " + path);
    }
}

GraphWriter::~GraphWriter() {
    if (file_.is_open()) {
        file_.close();
    }
}

auto GraphWriter::write(const Graph& graph) -> void {
    write_header();
    write_metadata(graph);
    write_values(graph);
    write_nodes(graph);
    write_tensors(graph);
    // v2 additions: graph input/output ID lists, captured constants.
    write_io_lists(graph);
    write_constants(graph);
}

auto GraphWriter::write_header() -> void {
    write_uint32(MAGIC_NUMBER);
    write_uint32(SERIALIZATION_VERSION);
}

auto GraphWriter::write_metadata(const Graph& graph) -> void {
    write_uint64(graph.num_nodes());
    write_uint64(graph.num_values());
    write_uint64(graph.inputs().size());
    write_uint64(graph.outputs().size());
}

auto GraphWriter::write_values(const Graph& graph) -> void {
    // Collect every Value reachable via (a) node outputs, (b) graph
    // inputs (which have no producing node), (c) graph outputs
    // (in case something exotic landed there without a producer), and
    // (d) *node inputs* — captured module parameters (e.g. Linear
    // weight / bias) are referenced as node inputs but have no producing
    // node and are not members of graph.inputs(). Without including
    // them here, read_values() never creates Value records for those
    // IDs, and then read_nodes() silently drops the parameter inputs
    // because graph.get_value() returns null — leaving the loaded
    // Linear node with only 1 input instead of 3, which makes forward()
    // throw "Output value not computed" downstream.
    // Deduplicated by Value ID — the same Value can be referenced
    // from multiple places.
    std::unordered_map<std::string, std::shared_ptr<Value>> seen;
    for (const auto& input : graph.inputs())  seen.emplace(input->id(),  input);
    for (const auto& output : graph.outputs()) seen.emplace(output->id(), output);
    for (const auto& node : graph.nodes()) {
        for (const auto& output : node->outputs()) {
            seen.emplace(output->id(), output);
        }
        for (const auto& input : node->inputs()) {
            seen.emplace(input->id(), input);
        }
    }

    std::vector<std::shared_ptr<Value>> all_values;
    all_values.reserve(seen.size());
    for (auto& [id, v] : seen) all_values.push_back(v);

    write_uint64(all_values.size());
    for (const auto& value : all_values) {
        write_string(value->id());
        write_int64_vector(value->shape());
        write_uint32(static_cast<uint32_t>(value->dtype()));
        write_uint32(static_cast<uint32_t>(value->device().type));
        write_int64(value->device().index);
    }
}

// v2: write the lists of graph input and output IDs so the reader
// can call set_inputs()/set_outputs() with the right Values. Without
// this the loaded graph silently had empty inputs/outputs, which
// broke every round-trip test.
auto GraphWriter::write_io_lists(const Graph& graph) -> void {
    write_uint64(graph.inputs().size());
    for (const auto& v : graph.inputs()) write_string(v->id());
    write_uint64(graph.outputs().size());
    for (const auto& v : graph.outputs()) write_string(v->id());
}

// v2: serialize the captured-parameter constants map so traced
// modules survive a save/load round trip.
auto GraphWriter::write_constants(const Graph& graph) -> void {
    const auto& constants = graph.constants();
    write_uint64(constants.size());
    for (const auto& [id, tensor] : constants) {
        write_string(id);
        write_tensor(tensor);
    }
}

auto GraphWriter::write_nodes(const Graph& graph) -> void {
    for (const auto& node : graph.nodes()) {
        // Write node type and name
        write_uint32(static_cast<uint32_t>(node->op_type()));
        write_string(node->name());

        // Write inputs
        write_uint64(node->inputs().size());
        for (const auto& input : node->inputs()) {
            write_string(input->id());
        }

        // Write outputs
        write_uint64(node->outputs().size());
        for (const auto& output : node->outputs()) {
            write_string(output->id());
        }

        // Write attributes
        auto [float_attrs, int_attrs, vec_attrs, bool_attrs, tensor_attrs] =
            const_cast<Node*>(node.get())->get_all_attrs();

        write_uint64(float_attrs.size());
        for (const auto& [name, val] : float_attrs) {
            write_string(name);
            write_float(val);
        }

        write_uint64(int_attrs.size());
        for (const auto& [name, val] : int_attrs) {
            write_string(name);
            write_int64(val);
        }

        write_uint64(vec_attrs.size());
        for (const auto& [name, val] : vec_attrs) {
            write_string(name);
            write_int64_vector(val);
        }

        write_uint64(bool_attrs.size());
        for (const auto& [name, val] : bool_attrs) {
            write_string(name);
            write_bool(val);
        }

        write_uint64(tensor_attrs.size());
        for (const auto& [name, val] : tensor_attrs) {
            write_string(name);
            write_tensor(val);
        }
    }
}

auto GraphWriter::write_tensors([[maybe_unused]] const Graph& graph) -> void {
    // Tensors are written inline with nodes
    // This section is reserved for future extensions
}

auto GraphWriter::write_uint32(uint32_t val) -> void {
    file_.write(reinterpret_cast<const char*>(&val), sizeof(val));
}

auto GraphWriter::write_uint64(uint64_t val) -> void {
    file_.write(reinterpret_cast<const char*>(&val), sizeof(val));
}

auto GraphWriter::write_int64(int64_t val) -> void {
    file_.write(reinterpret_cast<const char*>(&val), sizeof(val));
}

auto GraphWriter::write_float(float val) -> void {
    file_.write(reinterpret_cast<const char*>(&val), sizeof(val));
}

auto GraphWriter::write_double(double val) -> void {
    file_.write(reinterpret_cast<const char*>(&val), sizeof(val));
}

auto GraphWriter::write_bool(bool val) -> void {
    uint8_t byte = val ? 1 : 0;
    file_.write(reinterpret_cast<const char*>(&byte), sizeof(byte));
}

auto GraphWriter::write_string(const std::string& str) -> void {
    write_uint64(str.size());
    file_.write(str.data(), str.size());
}

auto GraphWriter::write_tensor(const Tensor& tensor) -> void {
    // Write shape
    write_int64_vector(std::vector<int64_t>(tensor.shape().begin(), tensor.shape().end()));

    // Write dtype
    write_uint32(static_cast<uint32_t>(tensor.dtype()));

    // Write device
    write_uint32(static_cast<uint32_t>(tensor.device().type));
    write_int64(tensor.device().index);

    // Write data. fstream::write expects a host pointer — if the tensor
    // lives on a GPU backend, data_ptr() returns a device pointer which
    // would SEGV on host read. Also materialise a contiguous CPU copy so
    // the serialized byte order matches what the reader expects.
    Tensor host = tensor.device().type == Device::Type::CPU
                      ? tensor.contiguous()
                      : tensor.to(Device::cpu()).contiguous();
    size_t data_size = host.numel() * host.dtype_size();
    write_uint64(data_size);
    file_.write(reinterpret_cast<const char*>(host.data_ptr()), data_size);
}

auto GraphWriter::write_int64_vector(const std::vector<int64_t>& vec) -> void {
    write_uint64(vec.size());
    for (int64_t val : vec) {
        write_int64(val);
    }
}

// ============================================================================
// GraphReader Implementation
// ============================================================================

GraphReader::GraphReader(const std::string& path) : file_(path, std::ios::binary) {
    if (!file_.is_open()) {
        throw std::runtime_error("Failed to open file for reading: " + path);
    }
}

GraphReader::~GraphReader() {
    if (file_.is_open()) {
        file_.close();
    }
}

auto GraphReader::read() -> std::shared_ptr<Graph> {
    read_header();

    auto graph = std::make_shared<Graph>();

    read_metadata(*graph);
    read_values(*graph);
    read_nodes(*graph);
    read_tensors(*graph);
    // v2 additions: graph input/output ID lists + constants map.
    read_io_lists(*graph);
    read_constants(*graph);

    return graph;
}

auto GraphReader::read_header() -> void {
    uint32_t magic = read_uint32();
    if (magic != MAGIC_NUMBER) {
        throw std::runtime_error("Invalid file format: magic number mismatch");
    }

    version_ = read_uint32();
    if (version_ != SERIALIZATION_VERSION) {
        throw std::runtime_error("Unsupported serialization version: " +
                                std::to_string(version_));
    }
}

auto GraphReader::read_metadata(Graph& graph) -> void {
    meta_num_nodes_ = read_uint64();
    (void)read_uint64();  // num_values — consumed by read_values() directly
    meta_num_inputs_ = read_uint64();
    meta_num_outputs_ = read_uint64();
    (void)graph;
}

auto GraphReader::read_values(Graph& graph) -> void {
    uint64_t num_values = read_uint64();

    for (uint64_t i = 0; i < num_values; ++i) {
        std::string id = read_string();
        std::vector<int64_t> shape = read_int64_vector();
        DType dtype = static_cast<DType>(read_uint32());
        auto dev_type = static_cast<Device::Type>(read_uint32());
        int64_t dev_index = read_int64();

        Device device(dev_type, dev_index);
        graph.create_value(id, std::move(shape), dtype, device);
    }
}

auto GraphReader::read_nodes(Graph& graph) -> void {
    // Use the metadata-declared node count rather than a peek+try/catch
    // loop. The old loop could not coexist with any trailing section
    // (v2 adds I/O lists and constants) — it would either stop short
    // or swallow unrelated bytes as "bad node" exceptions.
    for (uint64_t n = 0; n < meta_num_nodes_; ++n) {
        OpType op_type = static_cast<OpType>(read_uint32());
        std::string name = read_string();

        auto node = graph.create_node(op_type, name);

        // Read inputs.
        // Phase P0 / JIT correctness fix: if an input/output value ID isn't
        // in the values table (truncated or corrupted file), refuse to load
        // rather than silently dropping the missing input. The previous
        // silent-skip path produced a node with fewer inputs than the file
        // declared, causing forward() to crash later with a misleading
        // "value not computed" error instead of a clear deserialization
        // failure here.
        uint64_t num_inputs = read_uint64();
        for (uint64_t i = 0; i < num_inputs; ++i) {
            std::string input_id = read_string();
            auto value = graph.get_value(input_id);
            if (!value) {
                throw std::runtime_error(
                    "GraphReader: malformed graph — node '" + name +
                    "' references input value '" + input_id +
                    "' that does not exist in the values section");
            }
            node->add_input(value);
        }

        // Read outputs — same rule.
        uint64_t num_outputs = read_uint64();
        for (uint64_t i = 0; i < num_outputs; ++i) {
            std::string output_id = read_string();
            auto value = graph.get_value(output_id);
            if (!value) {
                throw std::runtime_error(
                    "GraphReader: malformed graph — node '" + name +
                    "' produces output value '" + output_id +
                    "' that does not exist in the values section");
            }
            value->set_node(node);
            node->add_output(value);
        }

        // Read attributes
        uint64_t num_float_attrs = read_uint64();
        for (uint64_t i = 0; i < num_float_attrs; ++i) {
            std::string attr_name = read_string();
            float val = read_float();
            node->set_attr(attr_name, val);
        }

        uint64_t num_int_attrs = read_uint64();
        for (uint64_t i = 0; i < num_int_attrs; ++i) {
            std::string attr_name = read_string();
            int64_t val = read_int64();
            node->set_int_attr(attr_name, val);
        }

        uint64_t num_vec_attrs = read_uint64();
        for (uint64_t i = 0; i < num_vec_attrs; ++i) {
            std::string attr_name = read_string();
            std::vector<int64_t> val = read_int64_vector();
            node->set_vec_attr(attr_name, std::move(val));
        }

        uint64_t num_bool_attrs = read_uint64();
        for (uint64_t i = 0; i < num_bool_attrs; ++i) {
            std::string attr_name = read_string();
            bool val = read_bool();
            node->set_bool_attr(attr_name, val);
        }

        uint64_t num_tensor_attrs = read_uint64();
        for (uint64_t i = 0; i < num_tensor_attrs; ++i) {
            std::string attr_name = read_string();
            Tensor val = read_tensor();
            node->set_tensor_attr(attr_name, std::move(val));
        }

        graph.add_node(node);
    }
}

auto GraphReader::read_tensors(Graph& graph) -> void {
    // Reserved for future use
    (void)graph;
}

// v2: read the graph input/output ID lists written by write_io_lists.
// We resolve each ID to a Value via graph.get_value() (populated by
// read_values above) and wire them into the graph via set_inputs /
// set_outputs — otherwise Graph::forward() has no idea which Values
// are inputs vs. intermediates.
auto GraphReader::read_io_lists(Graph& graph) -> void {
    uint64_t num_inputs = read_uint64();
    std::vector<std::shared_ptr<Value>> ins;
    ins.reserve(num_inputs);
    for (uint64_t i = 0; i < num_inputs; ++i) {
        std::string id = read_string();
        auto v = graph.get_value(id);
        if (v) ins.push_back(v);
    }
    graph.set_inputs(std::move(ins));

    uint64_t num_outputs = read_uint64();
    std::vector<std::shared_ptr<Value>> outs;
    outs.reserve(num_outputs);
    for (uint64_t i = 0; i < num_outputs; ++i) {
        std::string id = read_string();
        auto v = graph.get_value(id);
        if (v) outs.push_back(v);
    }
    graph.set_outputs(std::move(outs));
}

// v2: read the captured-parameter constants map.
auto GraphReader::read_constants(Graph& graph) -> void {
    uint64_t num_constants = read_uint64();
    for (uint64_t i = 0; i < num_constants; ++i) {
        std::string id = read_string();
        Tensor tensor = read_tensor();
        graph.set_constant(id, tensor);
    }
}

auto GraphReader::read_uint32() -> uint32_t {
    uint32_t val;
    file_.read(reinterpret_cast<char*>(&val), sizeof(val));
    return val;
}

auto GraphReader::read_uint64() -> uint64_t {
    uint64_t val;
    file_.read(reinterpret_cast<char*>(&val), sizeof(val));
    return val;
}

auto GraphReader::read_int64() -> int64_t {
    int64_t val;
    file_.read(reinterpret_cast<char*>(&val), sizeof(val));
    return val;
}

auto GraphReader::read_float() -> float {
    float val;
    file_.read(reinterpret_cast<char*>(&val), sizeof(val));
    return val;
}

auto GraphReader::read_double() -> double {
    double val;
    file_.read(reinterpret_cast<char*>(&val), sizeof(val));
    return val;
}

auto GraphReader::read_bool() -> bool {
    uint8_t byte;
    file_.read(reinterpret_cast<char*>(&byte), sizeof(byte));
    return byte != 0;
}

auto GraphReader::read_string() -> std::string {
    uint64_t size = read_uint64();
    std::string str(size, '\0');
    file_.read(&str[0], size);
    return str;
}

auto GraphReader::read_tensor() -> Tensor {
    std::vector<int64_t> shape = read_int64_vector();
    DType dtype = static_cast<DType>(read_uint32());
    auto dev_type = static_cast<Device::Type>(read_uint32());
    int64_t dev_index = read_int64();
    Device original_device(dev_type, dev_index);

    // Mirror write_tensor: bytes were serialized from a CPU copy. Read into
    // a CPU tensor first (fstream::read needs a host pointer), then migrate
    // to the recorded device if needed.
    Tensor tensor(shape, dtype, Device::cpu());

    uint64_t data_size = read_uint64();
    file_.read(reinterpret_cast<char*>(tensor.data_ptr()), data_size);

    if (original_device.type != Device::Type::CPU) {
        tensor = tensor.to(original_device);
    }
    return tensor;
}

auto GraphReader::read_int64_vector() -> std::vector<int64_t> {
    uint64_t size = read_uint64();
    std::vector<int64_t> vec(size);
    for (uint64_t i = 0; i < size; ++i) {
        vec[i] = read_int64();
    }
    return vec;
}

// ============================================================================
// Convenience Functions
// ============================================================================

auto save_graph(const Graph& graph, const std::string& path) -> void {
    GraphWriter writer(path);
    writer.write(graph);
}

auto load_graph(const std::string& path) -> std::shared_ptr<Graph> {
    GraphReader reader(path);
    return reader.read();
}

auto export_graph_text(const Graph& graph, const std::string& path) -> void {
    std::ofstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for writing: " + path);
    }

    file << graph.to_string();
}

// Note: import_graph_text() was removed as it's not needed.
// Text export is one-way for debugging/visualization only.
// Use load_graph() for proper binary serialization.

auto get_graph_stats(const Graph& graph) -> std::string {
    std::ostringstream oss;
    oss << "Graph Statistics:\n";
    oss << "  Nodes: " << graph.num_nodes() << "\n";
    oss << "  Values: " << graph.num_values() << "\n";
    oss << "  Inputs: " << graph.inputs().size() << "\n";
    oss << "  Outputs: " << graph.outputs().size() << "\n\n";

    // Count operations by type
    std::unordered_map<OpType, int> op_counts;
    for (const auto& node : graph.nodes()) {
        op_counts[node->op_type()]++;
    }

    oss << "  Operation counts:\n";
    for (const auto& [op_type, count] : op_counts) {
        oss << "    " << op_type_to_string(op_type) << ": " << count << "\n";
    }

    return oss.str();
}

auto verify_graph(const Graph& graph) -> std::vector<std::string> {
    std::vector<std::string> errors;

    // Check for dangling values
    for (const auto& node : graph.nodes()) {
        for (const auto& input : node->inputs()) {
            if (!input) {
                errors.push_back("Node " + node->name() + " has null input");
            }
        }

        for (const auto& output : node->outputs()) {
            if (!output) {
                errors.push_back("Node " + node->name() + " has null output");
            }
        }
    }

    // Check outputs are computed
    for (const auto& output : graph.outputs()) {
        if (!output) {
            errors.push_back("Graph has null output");
        }
    }

    return errors;
}

auto export_graph_dot(const Graph& graph, const std::string& path) -> void {
    std::ofstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for writing: " + path);
    }

    file << "digraph G {\n";
    file << "  rankdir=TB;\n";
    file << "  node [shape=box];\n\n";

    // Write nodes
    for (const auto& node : graph.nodes()) {
        file << "  \"" << node->name() << "\" [label=\""
             << node->name() << "\\n" << op_type_to_string(node->op_type())
             << "\"];\n";

        // Write edges
        for (const auto& input : node->inputs()) {
            auto producer = input->node();
            if (producer) {
                file << "  \"" << producer->name() << "\" -> \""
                     << node->name() << "\" [label=\"" << input->id() << "\"];\n";
            }
        }
    }

    file << "}\n";
}

} // namespace jit
} // namespace tenzor
