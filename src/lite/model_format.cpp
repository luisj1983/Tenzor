/**
 * @file model_format.cpp
 * @brief Minimal TZLITE binary model reader/writer implementation
 *
 * This is a minimum-viable implementation that supports the header-only
 * (zero-node) case used by smoke tests. Full node table parsing is left
 * to a future expansion of the lite runtime.
 */

#include "tenzor/lite/model_format.hpp"

#include <fstream>
#include <stdexcept>
#include <vector>

namespace tenzor::lite {

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

    const auto* header = static_cast<const TZLiteHeader*>(data);
    if (header->magic != TZLITE_MAGIC) {
        throw std::runtime_error("TZLiteReader: bad magic — not a TZLITE file");
    }
    if (header->version != TZLITE_VERSION) {
        throw std::runtime_error("TZLiteReader: unsupported TZLITE version " +
                                  std::to_string(header->version));
    }

    auto graph = std::make_unique<LiteGraph>();
    // For the minimal-viable case with num_nodes==0 we are done.
    // Full node table parsing will be added when the runtime needs it.
    if (header->num_nodes > 0) {
        // TODO: parse node table from buffer.
        // For now, we accept the file but leave the graph empty.
    }
    return graph;
}

auto TZLiteWriter::save(const LiteGraph& graph, const std::string& path) -> void {
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("TZLiteWriter: failed to open file: " + path);
    }

    TZLiteHeader header{};
    header.magic = TZLITE_MAGIC;
    header.version = TZLITE_VERSION;
    header.num_nodes = static_cast<uint32_t>(graph.num_nodes());
    header.num_weights = 0;
    header.weight_data_offset = sizeof(TZLiteHeader);

    file.write(reinterpret_cast<const char*>(&header), sizeof(header));
    if (!file) {
        throw std::runtime_error("TZLiteWriter: failed to write header to: " + path);
    }
}

}  // namespace tenzor::lite
