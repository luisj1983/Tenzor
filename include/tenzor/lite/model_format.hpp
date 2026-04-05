/**
 * @file model_format.hpp
 * @brief TZLITE binary model format for the lite runtime
 *
 * The TZLITE format is a compact, mmap-friendly binary format designed for
 * fast cold-start loading on mobile/embedded devices. Layout:
 *
 *   [TZLiteHeader] [NodeTable...] [WeightData...]
 *
 * Weight data is 64-byte aligned for SIMD access. The reader supports both
 * file-backed and memory-backed loading (for Android asset bundles, etc.).
 */

#pragma once

#include "lite_graph.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace tenzor::lite {

constexpr uint32_t TZLITE_MAGIC   = 0x544C5A54;  // "TZLT"
constexpr uint32_t TZLITE_VERSION = 1;

struct TZLiteHeader {
    uint32_t magic;
    uint32_t version;
    uint32_t num_nodes;
    uint32_t num_weights;
    uint64_t weight_data_offset;
};

// ============================================================================
// Reader — load a TZLITE model into a LiteGraph
// ============================================================================

class TZLiteReader {
public:
    /** Load from a file path. */
    static auto load(const std::string& path) -> std::unique_ptr<LiteGraph>;

    /** Load from an in-memory buffer (e.g. Android asset, embedded resource). */
    static auto load(const void* data, size_t size) -> std::unique_ptr<LiteGraph>;
};

// ============================================================================
// Writer — serialize a LiteGraph to TZLITE format
// ============================================================================

class TZLiteWriter {
public:
    /** Save the graph and its weights to a file. */
    static auto save(const LiteGraph& graph, const std::string& path) -> void;
};

}  // namespace tenzor::lite
