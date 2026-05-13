/**
 * @file model_format.hpp
 * @brief TZLITE binary model format for the Lite runtime.
 *
 * Layout (v1, all extensions additive within the 24-byte header envelope):
 *
 *   [TZLiteHeader (24 bytes)]
 *   [Node table]
 *   [TLV section table]              (optional; pointed at by weight_data_offset)
 *     [u32 section_count]
 *     [foreach section: u32 tag, u64 size, u8[size] payload]
 *
 * Sections recognised in v1:
 *   'TVAL'  Tensor value table — per-tensor_id metadata
 *           (source, dtype, ndim, shape, plus source_index/nbytes for weights).
 *   'WGTS'  Concatenated raw weight bytes addressed by TVAL offsets.
 *           Weight Tensors are loaded as non-owning views over this region.
 *   'IOSP'  Graph-level input/output tensor_id ordering.
 *   'META'  UTF-8 key/value metadata.
 *
 * Backward compatibility: a v1 file with `num_weights == 0` and
 * `weight_data_offset == sizeof(TZLiteHeader) + node_table_bytes` (i.e. no
 * TLV section) loads as before. Old TZLITE files in the wild remain readable.
 */

#pragma once

#include "lite_graph.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "../core/dtype.hpp"
#include "../core/tensor.hpp"

namespace tenzor::lite {

constexpr uint32_t TZLITE_MAGIC   = 0x544C5A54;  // "TZLT"
constexpr uint32_t TZLITE_VERSION = 1;

// FourCC tags for TLV sections (little-endian byte order in the file).
constexpr uint32_t TZLITE_TAG_TVAL = 0x4C415654;  // 'T''V''A''L' read little-endian
constexpr uint32_t TZLITE_TAG_WGTS = 0x53544757;  // 'W''G''T''S'
constexpr uint32_t TZLITE_TAG_IOSP = 0x50534F49;  // 'I''O''S''P'
constexpr uint32_t TZLITE_TAG_META = 0x4154454D;  // 'M''E''T''A'

struct TZLiteHeader {
    uint32_t magic;
    uint32_t version;
    uint32_t num_nodes;
    uint32_t num_weights;          ///< number of weight tensors in WGTS
    uint64_t weight_data_offset;   ///< file offset to TLV section table (or EOF if none)
};
static_assert(sizeof(TZLiteHeader) == 24,
              "TZLiteHeader byte layout pinned by test_model_format.cpp");

/** How a tensor_id's storage is sourced when LiteGraph::execute() runs. */
enum class TensorSource : uint8_t {
    Input        = 0,   ///< supplied by the caller's forward() argument
    Weight       = 1,   ///< view into the WGTS section
    Intermediate = 2,   ///< produced by a node at runtime
};

/** TVAL entry describing one tensor_id's expected shape, dtype, and source.
 *
 * The runtime uses this table both to pre-populate the tensor pool with
 * weights and to validate that caller-supplied inputs have the right dtype
 * and shape.
 */
struct TensorValue {
    int16_t tensor_id{-1};
    TensorSource source{TensorSource::Intermediate};
    DType dtype{DType::Float32};
    std::vector<int64_t> shape;
    /** For Weight: byte offset into the WGTS payload. Otherwise unused. */
    uint64_t weight_offset{0};
    /** For Weight: byte length within WGTS. Otherwise unused. */
    uint64_t weight_nbytes{0};
    /** For Input: position in forward()'s inputs vector. Otherwise unused. */
    uint32_t input_index{0};
};

/** Loaded model artefacts — what TZLiteReader::load returns. */
struct LoadedModel {
    std::unique_ptr<LiteGraph> graph;
    std::vector<TensorValue> tensor_values;       ///< sorted by tensor_id ascending
    std::vector<uint8_t> weight_blob;             ///< raw bytes (empty if no WGTS)
    std::vector<int16_t> input_ids;
    std::vector<int16_t> output_ids;
    std::unordered_map<std::string, std::string> metadata;
};

// ============================================================================
// Reader — load a TZLITE model into a LiteGraph + ancillary data.
// ============================================================================

class TZLiteReader {
public:
    /** Load from a file path. Convenience for `load(buffer)`. */
    static auto load(const std::string& path) -> std::unique_ptr<LiteGraph>;

    /** Load from an in-memory buffer (e.g. Android asset, embedded resource). */
    static auto load(const void* data, size_t size) -> std::unique_ptr<LiteGraph>;

    /** Load the full model artefact bundle — includes weights, I/O, metadata. */
    static auto load_full(const std::string& path) -> LoadedModel;
    static auto load_full(const void* data, size_t size) -> LoadedModel;
};

// ============================================================================
// Writer — serialise a LiteGraph (+ optional weights / I/O spec / metadata).
// ============================================================================

/** Arguments grouped so the writer's signature stays stable as new optional
 *  payloads are added (memory plan section is reserved for Phase 5).
 */
struct WriteOptions {
    /** Map of tensor_id -> Tensor for weights. Tensors must be CPU-contiguous;
     *  the writer will call `.contiguous().to(cpu())` if needed.
     */
    std::unordered_map<int16_t, Tensor> weights;

    /** tensor_ids supplied by the caller's forward() argument, in order. */
    std::vector<int16_t> input_ids;

    /** tensor_ids returned by forward(), in order. */
    std::vector<int16_t> output_ids;

    /** Optional input shape/dtype declarations. Validated against caller
     *  inputs at runtime if present. Keyed by tensor_id.
     */
    std::unordered_map<int16_t, std::pair<DType, std::vector<int64_t>>>
        input_specs;

    /** Free-form metadata. */
    std::unordered_map<std::string, std::string> metadata;
};

class TZLiteWriter {
public:
    /** Save graph-only (no weights, no I/O spec). Preserves the original
     *  byte-exact v1 layout — used by existing format-pinning tests. */
    static auto save(const LiteGraph& graph, const std::string& path) -> void;

    /** Save graph + weights + I/O spec + metadata in TLV-extended form. */
    static auto save(const LiteGraph& graph,
                     const std::string& path,
                     const WriteOptions& opts) -> void;
};

}  // namespace tenzor::lite
