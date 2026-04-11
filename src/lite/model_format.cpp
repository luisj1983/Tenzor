/**
 * @file model_format.cpp
 * @brief TZLITE binary model reader/writer implementation
 *
 * Layout:
 *   [TZLiteHeader] [NodeTable...] [WeightData...]
 *
 * Each node in the NodeTable is serialized as:
 *   uint16_t  op_type           (LiteOpType)
 *   uint16_t  num_inputs
 *   int16_t[] input_ids         (num_inputs entries)
 *   uint16_t  num_outputs
 *   int16_t[] output_ids        (num_outputs entries)
 *   float[4]  attrs.f
 *   int64_t[4] attrs.i
 *
 * All fields are little-endian and packed. The reader advances a cursor
 * through the buffer, validating bounds at each step.
 */

#include "tenzor/lite/model_format.hpp"

#include <cstring>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace tenzor::lite {

namespace {

// Bounds-checked read of a POD value from buffer at offset. Advances offset.
template <typename T>
auto read_pod(const uint8_t* buffer, size_t size, size_t& offset, T& out) -> void {
    if (offset + sizeof(T) > size) {
        throw std::runtime_error("TZLiteReader: unexpected end of node table");
    }
    std::memcpy(&out, buffer + offset, sizeof(T));
    offset += sizeof(T);
}

template <typename T>
auto write_pod(std::ostream& file, const T& value) -> void {
    file.write(reinterpret_cast<const char*>(&value), sizeof(T));
    if (!file) {
        throw std::runtime_error("TZLiteWriter: write failed");
    }
}

} // anonymous namespace

auto TZLiteReader::load(const std::string& path) -> std::unique_ptr<LiteGraph> {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        throw std::runtime_error("TZLiteReader: failed to open file: " + path);
    }
    auto size = static_cast<size_t>(file.tellg());
    file.seekg(0, std::ios::beg);

    std::vector<uint8_t> buffer(size);
    if (!file.read(reinterpret_cast<char*>(buffer.data()), static_cast<std::streamsize>(size))) {
        throw std::runtime_error("TZLiteReader: failed to read file: " + path);
    }

    return load(buffer.data(), size);
}

auto TZLiteReader::load(const void* data, size_t size) -> std::unique_ptr<LiteGraph> {
    if (data == nullptr || size < sizeof(TZLiteHeader)) {
        throw std::runtime_error("TZLiteReader: invalid input data");
    }

    const auto* buffer = static_cast<const uint8_t*>(data);
    const auto* header = reinterpret_cast<const TZLiteHeader*>(buffer);
    if (header->magic != TZLITE_MAGIC) {
        throw std::runtime_error("TZLiteReader: bad magic — not a TZLITE file");
    }
    if (header->version != TZLITE_VERSION) {
        throw std::runtime_error("TZLiteReader: unsupported TZLITE version " +
                                  std::to_string(header->version));
    }

    auto graph = std::make_unique<LiteGraph>();

    // Parse the node table starting immediately after the header.
    size_t offset = sizeof(TZLiteHeader);
    for (uint32_t n = 0; n < header->num_nodes; ++n) {
        LiteNode node{};

        uint16_t op_raw = 0;
        read_pod(buffer, size, offset, op_raw);
        node.op = static_cast<LiteOpType>(op_raw);

        uint16_t num_inputs = 0;
        read_pod(buffer, size, offset, num_inputs);
        node.input_ids.resize(num_inputs);
        for (uint16_t i = 0; i < num_inputs; ++i) {
            read_pod(buffer, size, offset, node.input_ids[i]);
        }

        uint16_t num_outputs = 0;
        read_pod(buffer, size, offset, num_outputs);
        node.output_ids.resize(num_outputs);
        for (uint16_t i = 0; i < num_outputs; ++i) {
            read_pod(buffer, size, offset, node.output_ids[i]);
        }

        for (int i = 0; i < 4; ++i) read_pod(buffer, size, offset, node.attrs.f[i]);
        for (int i = 0; i < 4; ++i) read_pod(buffer, size, offset, node.attrs.i[i]);

        graph->add_node(std::move(node));
    }

    return graph;
}

auto TZLiteWriter::save(const LiteGraph& graph, const std::string& path) -> void {
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("TZLiteWriter: failed to open file: " + path);
    }

    // Compute the size of the serialized node table so we can set
    // weight_data_offset to point past it.
    size_t node_table_bytes = 0;
    for (const auto& node : graph.nodes()) {
        node_table_bytes += sizeof(uint16_t);                               // op
        node_table_bytes += sizeof(uint16_t);                               // num_inputs
        node_table_bytes += node.input_ids.size() * sizeof(int16_t);
        node_table_bytes += sizeof(uint16_t);                               // num_outputs
        node_table_bytes += node.output_ids.size() * sizeof(int16_t);
        node_table_bytes += 4 * sizeof(float);
        node_table_bytes += 4 * sizeof(int64_t);
    }

    TZLiteHeader header{};
    header.magic = TZLITE_MAGIC;
    header.version = TZLITE_VERSION;
    header.num_nodes = static_cast<uint32_t>(graph.num_nodes());
    header.num_weights = 0;
    header.weight_data_offset = sizeof(TZLiteHeader) + node_table_bytes;

    write_pod(file, header);

    // Serialize each node in the format documented above.
    for (const auto& node : graph.nodes()) {
        write_pod<uint16_t>(file, static_cast<uint16_t>(node.op));

        auto num_inputs = static_cast<uint16_t>(node.input_ids.size());
        write_pod(file, num_inputs);
        for (int16_t id : node.input_ids) {
            write_pod(file, id);
        }

        auto num_outputs = static_cast<uint16_t>(node.output_ids.size());
        write_pod(file, num_outputs);
        for (int16_t id : node.output_ids) {
            write_pod(file, id);
        }

        for (int i = 0; i < 4; ++i) write_pod(file, node.attrs.f[i]);
        for (int i = 0; i < 4; ++i) write_pod(file, node.attrs.i[i]);
    }
}

}  // namespace tenzor::lite
