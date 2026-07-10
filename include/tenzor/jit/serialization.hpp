/**
 * @file serialization.hpp
 * @brief Graph serialization and deserialization
 *
 * Provides binary serialization format for saving and loading compiled
 * JIT graphs. The format is designed to be:
 * - Compact (binary encoding)
 * - Fast to load (minimal parsing)
 * - Extensible (version tagging)
 * - Cross-platform compatible
 *
 * File format structure:
 * - Magic number (4 bytes): "TZJT"
 * - Version (4 bytes)
 * - Metadata section (variable)
 * - Values section (variable)
 * - Nodes section (variable)
 * - Tensors section (variable)
 */

#pragma once

#include <memory>
#include <string>
#include <vector>
#include <fstream>
#include <cstdint>
#include "graph.hpp"

namespace tenzor {
namespace jit {

/**
 * @brief Serialization format version.
 *
 * Increment when making incompatible changes to format.
 *
 * v2 (2026-04-11): write graph input/output ID lists, include values
 *                  that are graph inputs (not only node outputs), and
 *                  serialize the captured-parameter constants map.
 *                  v1 silently dropped all three which made loaded
 *                  graphs lose their inputs/outputs and fail to
 *                  execute traced modules.
 * v3 (2026-07-02): serialize control-flow subgraphs. Each Node now emits
 *                  its then/else/body Graphs (presence flag + full nested
 *                  Graph, recursively). v1/v2 dropped them, so every loaded
 *                  If/Loop node had null branches and produced no outputs.
 *                  Also makes constants device-neutral: tensor/value device
 *                  fields are written as CPU (the byte domain), and the
 *                  executor moves each constant onto the runtime-input
 *                  device at execution time, so a graph traced on one
 *                  backend can replay on any other.
 */
constexpr uint32_t SERIALIZATION_VERSION = 5;  // v5: scalar attrs stored as f64
                                               // (JIT-F057; v4 = user KV metadata)

/**
 * @brief Magic number for Tenzor JIT files.
 */
constexpr uint32_t MAGIC_NUMBER = 0x544A5A54;  // "TZJT" in ASCII

/**
 * @brief Binary writer for graph serialization.
 *
 * Writes graph data to a binary stream in a compact, efficient format.
 * Handles endianness conversion for cross-platform compatibility.
 */
class GraphWriter {
public:
    /**
     * @brief Construct writer for file.
     *
     * @param path Output file path
     * @throws std::runtime_error if file cannot be opened
     */
    explicit GraphWriter(const std::string& path);

    /**
     * @brief Destructor (closes file).
     */
    ~GraphWriter();

    /**
     * @brief Write graph to file.
     *
     * @param graph Graph to serialize
     */
    auto write(const Graph& graph) -> void;

private:
    std::ofstream file_;

    /**
     * @brief Write magic number and version header.
     */
    auto write_header() -> void;

    /**
     * @brief Write a full graph body (metadata, values, nodes, I/O lists,
     *        constants) WITHOUT the file header.
     *
     * Factored out of write() so control-flow subgraphs (If then/else,
     * Loop body) can be serialized recursively by write_subgraph().
     */
    auto write_graph(const Graph& graph) -> void;

    /**
     * @brief Write an optional nested subgraph (presence flag + body).
     *
     * A null subgraph is a single `false` byte; a present subgraph is a
     * `true` byte followed by a full write_graph() body.
     */
    auto write_subgraph(const std::shared_ptr<Graph>& sub) -> void;

    /**
     * @brief Write metadata section.
     *
     * @param graph Graph to serialize
     */
    auto write_metadata(const Graph& graph) -> void;

    /**
     * @brief Write all values.
     *
     * @param graph Graph to serialize
     */
    auto write_values(const Graph& graph) -> void;

    /**
     * @brief Write all nodes.
     *
     * @param graph Graph to serialize
     */
    auto write_nodes(const Graph& graph) -> void;

    /**
     * @brief Write tensor data section.
     *
     * @param graph Graph to serialize
     */
    auto write_tensors(const Graph& graph) -> void;

    /**
     * @brief Write graph input/output ID lists (v2+).
     */
    auto write_io_lists(const Graph& graph) -> void;

    /**
     * @brief Write captured parameter constants map (v2+).
     */
    auto write_constants(const Graph& graph) -> void;

    // Primitive write methods
    auto write_uint32(uint32_t val) -> void;
    auto write_uint64(uint64_t val) -> void;
    auto write_int64(int64_t val) -> void;
    auto write_float(float val) -> void;
    auto write_double(double val) -> void;
    auto write_bool(bool val) -> void;
    auto write_string(const std::string& str) -> void;
    auto write_tensor(const Tensor& tensor) -> void;

    /**
     * @brief Write vector of int64.
     *
     * @param vec Vector to write
     */
    auto write_int64_vector(const std::vector<int64_t>& vec) -> void;
};

/**
 * @brief Binary reader for graph deserialization.
 *
 * Reads graph data from a binary stream, reconstructing the full
 * IR graph with all nodes, values, and tensor constants.
 */
class GraphReader {
public:
    /**
     * @brief Construct reader for file.
     *
     * @param path Input file path
     * @throws std::runtime_error if file cannot be opened or is invalid
     */
    explicit GraphReader(const std::string& path);

    /**
     * @brief Destructor (closes file).
     */
    ~GraphReader();

    /**
     * @brief Read graph from file.
     *
     * @return Reconstructed graph
     * @throws std::runtime_error if file is corrupted or version mismatch
     */
    auto read() -> std::shared_ptr<Graph>;

private:
    std::ifstream file_;
    uint32_t version_;

    /**
     * @brief Read a full graph body (metadata, values, nodes, I/O lists,
     *        constants) WITHOUT the file header.
     *
     * Factored out of read() so control-flow subgraphs can be reconstructed
     * recursively by read_subgraph(). Recursion-safe: read_nodes() captures
     * its node count into a local before any nested read clobbers the shared
     * meta_num_nodes_ member.
     */
    auto read_graph() -> std::shared_ptr<Graph>;

    /**
     * @brief Read an optional nested subgraph (presence flag + body).
     *
     * @return The reconstructed subgraph, or nullptr if the presence flag
     *         was false.
     */
    auto read_subgraph() -> std::shared_ptr<Graph>;
    // Count from the metadata section — saved so read_nodes() knows
    // exactly how many nodes to read (the old code used a peek+try/catch
    // loop that conflicted with any trailing section past the nodes).
    // The metadata section also carries num_inputs/num_outputs counts, but
    // those are parsed-and-discarded (see read_metadata()): the
    // authoritative input/output ID lists are read fresh by
    // read_io_lists(), so there's no need to stash the counts here.
    uint64_t meta_num_nodes_{0};

    /**
     * @brief Read and validate header.
     *
     * @throws std::runtime_error if magic number is wrong
     */
    auto read_header() -> void;

    /**
     * @brief Read metadata section.
     *
     * @param graph Graph to populate
     */
    auto read_metadata(Graph& graph) -> void;

    /**
     * @brief Read all values.
     *
     * @param graph Graph to populate
     */
    auto read_values(Graph& graph) -> void;

    /**
     * @brief Read all nodes.
     *
     * @param graph Graph to populate
     */
    auto read_nodes(Graph& graph) -> void;

    /**
     * @brief Read tensor data section.
     *
     * @param graph Graph to populate
     */
    auto read_tensors(Graph& graph) -> void;

    /**
     * @brief Read graph input/output ID lists (v2+) and wire them.
     */
    auto read_io_lists(Graph& graph) -> void;

    /**
     * @brief Read captured parameter constants map (v2+).
     */
    auto read_constants(Graph& graph) -> void;

    // Primitive read methods
    auto read_uint32() -> uint32_t;
    auto read_uint64() -> uint64_t;
    auto read_int64() -> int64_t;
    auto read_float() -> float;
    auto read_double() -> double;
    auto read_bool() -> bool;
    auto read_string() -> std::string;
    auto read_tensor() -> Tensor;

    /**
     * @brief Read vector of int64.
     *
     * @return Loaded vector
     */
    auto read_int64_vector() -> std::vector<int64_t>;

    /**
     * @brief Remaining unread bytes from the current get position.
     *
     * Used to bound untrusted declared element/record counts before
     * allocating or reserving, guarding against OOM-DoS on crafted files.
     * Returns 0 if the stream position is unavailable.
     */
    auto remaining_bytes() -> uint64_t;
};

/**
 * @brief Save graph to file.
 *
 * Convenience function for serializing a graph.
 *
 * @param graph Graph to save
 * @param path Output file path
 *
 * @code
 * auto graph = trace(model, input);
 * save_graph(*graph, "model.pt");
 * @endcode
 */
auto save_graph(const Graph& graph, const std::string& path) -> void;

/**
 * @brief Load graph from file.
 *
 * Convenience function for deserializing a graph.
 *
 * @param path Input file path
 * @return Loaded graph
 * @throws std::runtime_error if file is invalid
 *
 * @code
 * auto graph = load_graph("model.pt");
 * Variable output = graph->forward({input});
 * @endcode
 */
auto load_graph(const std::string& path) -> std::shared_ptr<Graph>;

/**
 * @brief Text-based graph export for debugging.
 *
 * Exports graph in human-readable format (similar to ONNX text format).
 * Not optimized for size or loading speed.
 *
 * @param graph Graph to export
 * @param path Output text file path
 *
 * @code
 * export_graph_text(*graph, "model.txt");
 * // Can be viewed/edited with text editor
 * @endcode
 */
auto export_graph_text(const Graph& graph, const std::string& path) -> void;

// Note: import_graph_text() removed - text export is one-way for debugging only.
// Use load_graph() for proper binary deserialization.

/**
 * @brief Print graph statistics.
 *
 * Outputs diagnostic information about graph structure:
 * - Number of nodes by type
 * - Parameter count and size
 * - Memory usage estimate
 * - Optimization opportunities
 *
 * @param graph Graph to analyze
 * @return Statistics string
 *
 * @code
 * auto stats = get_graph_stats(*graph);
 * std::cout << stats << std::endl;
 * @endcode
 */
auto get_graph_stats(const Graph& graph) -> std::string;

/**
 * @brief Verify graph integrity.
 *
 * Checks for common errors:
 * - Dangling value references
 * - Type mismatches
 * - Invalid node connections
 * - Missing required attributes
 *
 * @param graph Graph to verify
 * @return Vector of error messages (empty if valid)
 *
 * @code
 * auto errors = verify_graph(*graph);
 * if (!errors.empty()) {
 *     for (const auto& err : errors) {
 *         std::cerr << "Error: " << err << std::endl;
 *     }
 * }
 * @endcode
 */
auto verify_graph(const Graph& graph) -> std::vector<std::string>;

/**
 * @brief Graph visualization helper.
 *
 * Exports graph in DOT format for visualization with Graphviz.
 *
 * @param graph Graph to visualize
 * @param path Output .dot file path
 *
 * @code
 * export_graph_dot(*graph, "model.dot");
 * // Then run: dot -Tpng model.dot -o model.png
 * @endcode
 */
auto export_graph_dot(const Graph& graph, const std::string& path) -> void;

} // namespace jit
} // namespace tenzor
