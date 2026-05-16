/**
 * @file model_format.cpp
 * @brief TZLITE reader/writer — Phase 2 TLV-section implementation.
 *
 * The serialised graph layout is unchanged from v1 (header + node table);
 * Phase 2 adds an optional TLV section table after the node table, whose
 * starting offset is recorded in `TZLiteHeader::weight_data_offset`. When no
 * extensions are present the writer falls back to the exact pre-Phase-2 byte
 * sequence — this keeps the magic/version/sizeof pin in
 * tests/lite/test_model_format.cpp green.
 *
 * Wire format quick reference (all little-endian, packed):
 *
 *   TZLiteHeader (24 bytes, see model_format.hpp)
 *
 *   Per node:
 *     u16 op_id
 *     u16 num_inputs;  i16[num_inputs]   input_ids
 *     u16 num_outputs; i16[num_outputs]  output_ids
 *     f32[4] attrs.f
 *     i64[4] attrs.i
 *
 *   TLV section table (optional, starting at weight_data_offset):
 *     u32 section_count
 *     repeated section_count times:
 *       u32 tag         (FourCC, see TZLITE_TAG_*)
 *       u64 payload_size
 *       u8[payload_size] payload
 *
 *   TVAL payload:
 *     u32 entry_count
 *     repeated entry_count times:
 *       i16 tensor_id
 *       u8  source       (TensorSource)
 *       u8  ndim
 *       u16 dtype
 *       u32 input_index
 *       u64 weight_offset
 *       u64 weight_nbytes
 *       i64[ndim] shape
 *
 *   WGTS payload:
 *     u8[N] raw weight bytes, addressed by weight_offset/weight_nbytes in TVAL.
 *     No internal alignment is enforced — Phase 5 layering can add 64-byte
 *     alignment if mmap-zero-copy is added.
 *
 *   IOSP payload:
 *     u32 num_inputs;  i16[num_inputs]  input_tensor_ids
 *     u32 num_outputs; i16[num_outputs] output_tensor_ids
 *
 *   META payload:
 *     u32 entry_count
 *     repeated entry_count times:
 *       u32 key_len;   u8[key_len]   utf-8 key
 *       u32 value_len; u8[value_len] utf-8 value
 */

#include "tenzor/lite/model_format.hpp"
#include "tenzor/lite/memory_planner.hpp"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <ios>
#include <iterator>
#include <stdexcept>

namespace tenzor::lite {

namespace {

// ---------------------------------------------------------------------------
// Buffer cursor primitives — bounds-checked read/write.
// ---------------------------------------------------------------------------

template <typename T>
auto read_pod(const uint8_t* buffer, size_t size, size_t& offset, T& out) -> void {
    if (offset + sizeof(T) > size) {
        throw std::runtime_error("TZLiteReader: unexpected end of buffer");
    }
    std::memcpy(&out, buffer + offset, sizeof(T));
    offset += sizeof(T);
}

auto read_bytes(const uint8_t* buffer, size_t size, size_t& offset,
                void* dst, size_t n) -> void {
    if (offset + n > size) {
        throw std::runtime_error("TZLiteReader: unexpected end of buffer");
    }
    std::memcpy(dst, buffer + offset, n);
    offset += n;
}

template <typename T>
auto append_pod(std::vector<uint8_t>& out, const T& value) -> void {
    const auto* p = reinterpret_cast<const uint8_t*>(&value);
    out.insert(out.end(), p, p + sizeof(T));
}

auto append_bytes(std::vector<uint8_t>& out, const void* src, size_t n) -> void {
    const auto* p = static_cast<const uint8_t*>(src);
    out.insert(out.end(), p, p + n);
}

template <typename T>
auto write_pod(std::ostream& file, const T& value) -> void {
    file.write(reinterpret_cast<const char*>(&value), sizeof(T));
    if (!file) {
        throw std::runtime_error("TZLiteWriter: write failed");
    }
}

// ---------------------------------------------------------------------------
// Node table — shared between v1 graph-only writer and v2 TLV writer.
// ---------------------------------------------------------------------------

auto serialise_node_table(const LiteGraph& graph, uint32_t version) -> std::vector<uint8_t> {
    std::vector<uint8_t> out;
    out.reserve(graph.num_nodes() * 64);
    for (const auto& node : graph.nodes()) {
        append_pod<uint16_t>(out, static_cast<uint16_t>(node.op));

        auto num_inputs = static_cast<uint16_t>(node.input_ids.size());
        append_pod(out, num_inputs);
        for (int16_t id : node.input_ids) append_pod(out, id);

        auto num_outputs = static_cast<uint16_t>(node.output_ids.size());
        append_pod(out, num_outputs);
        for (int16_t id : node.output_ids) append_pod(out, id);

        for (int i = 0; i < 4; ++i) append_pod(out, node.attrs.f[i]);
        for (int i = 0; i < 4; ++i) append_pod(out, node.attrs.i[i]);

        // Wave Inf-E5 (deferred → landed): v3+ writes variable-length extras
        // appended to each node record. v2 (and v1 implicit) files have no
        // extras suffix — readers detect this from the file header version
        // and skip the parsing entirely.
        if (version >= 3) {
            auto ec_i = static_cast<uint32_t>(node.attrs.extra_i.size());
            append_pod(out, ec_i);
            for (int64_t v : node.attrs.extra_i) append_pod(out, v);
            auto ec_f = static_cast<uint32_t>(node.attrs.extra_f.size());
            append_pod(out, ec_f);
            for (float v : node.attrs.extra_f) append_pod(out, v);
        }
    }
    return out;
}

auto deserialise_node_table(const uint8_t* buffer, size_t size,
                            size_t& offset, uint32_t num_nodes,
                            uint32_t version)
    -> std::unique_ptr<LiteGraph> {
    auto graph = std::make_unique<LiteGraph>();
    for (uint32_t n = 0; n < num_nodes; ++n) {
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

        // Wave Inf-E5: v3+ has variable-length extras per node.
        if (version >= 3) {
            uint32_t ec_i = 0;
            read_pod(buffer, size, offset, ec_i);
            node.attrs.extra_i.resize(ec_i);
            for (uint32_t k = 0; k < ec_i; ++k) {
                read_pod(buffer, size, offset, node.attrs.extra_i[k]);
            }
            uint32_t ec_f = 0;
            read_pod(buffer, size, offset, ec_f);
            node.attrs.extra_f.resize(ec_f);
            for (uint32_t k = 0; k < ec_f; ++k) {
                read_pod(buffer, size, offset, node.attrs.extra_f[k]);
            }
        }

        graph->add_node(std::move(node));
    }
    return graph;
}

// ---------------------------------------------------------------------------
// TLV section payload builders.
// ---------------------------------------------------------------------------

auto build_tval_payload(const std::vector<TensorValue>& tvs) -> std::vector<uint8_t> {
    std::vector<uint8_t> out;
    out.reserve(tvs.size() * 32);
    auto count = static_cast<uint32_t>(tvs.size());
    append_pod(out, count);
    for (const auto& tv : tvs) {
        append_pod(out, tv.tensor_id);
        append_pod<uint8_t>(out, static_cast<uint8_t>(tv.source));
        append_pod<uint8_t>(out, static_cast<uint8_t>(tv.shape.size()));
        append_pod<uint16_t>(out, static_cast<uint16_t>(tv.dtype));
        append_pod(out, tv.input_index);
        append_pod(out, tv.weight_offset);
        append_pod(out, tv.weight_nbytes);
        for (int64_t s : tv.shape) append_pod(out, s);
    }
    return out;
}

auto parse_tval_payload(const uint8_t* buffer, size_t size)
    -> std::vector<TensorValue> {
    std::vector<TensorValue> out;
    size_t offset = 0;
    uint32_t count = 0;
    read_pod(buffer, size, offset, count);
    out.reserve(count);
    for (uint32_t i = 0; i < count; ++i) {
        TensorValue tv;
        read_pod(buffer, size, offset, tv.tensor_id);
        uint8_t src = 0;
        read_pod(buffer, size, offset, src);
        tv.source = static_cast<TensorSource>(src);
        uint8_t ndim = 0;
        read_pod(buffer, size, offset, ndim);
        uint16_t dt = 0;
        read_pod(buffer, size, offset, dt);
        tv.dtype = static_cast<DType>(dt);
        read_pod(buffer, size, offset, tv.input_index);
        read_pod(buffer, size, offset, tv.weight_offset);
        read_pod(buffer, size, offset, tv.weight_nbytes);
        tv.shape.resize(ndim);
        for (uint8_t d = 0; d < ndim; ++d) {
            read_pod(buffer, size, offset, tv.shape[d]);
        }
        out.push_back(std::move(tv));
    }
    return out;
}

auto build_iosp_payload(const std::vector<int16_t>& inputs,
                        const std::vector<int16_t>& outputs)
    -> std::vector<uint8_t> {
    std::vector<uint8_t> out;
    auto ni = static_cast<uint32_t>(inputs.size());
    append_pod(out, ni);
    for (int16_t id : inputs) append_pod(out, id);
    auto no = static_cast<uint32_t>(outputs.size());
    append_pod(out, no);
    for (int16_t id : outputs) append_pod(out, id);
    return out;
}

auto parse_iosp_payload(const uint8_t* buffer, size_t size)
    -> std::pair<std::vector<int16_t>, std::vector<int16_t>> {
    std::pair<std::vector<int16_t>, std::vector<int16_t>> out;
    size_t offset = 0;
    uint32_t ni = 0;
    read_pod(buffer, size, offset, ni);
    out.first.resize(ni);
    for (uint32_t i = 0; i < ni; ++i) read_pod(buffer, size, offset, out.first[i]);
    uint32_t no = 0;
    read_pod(buffer, size, offset, no);
    out.second.resize(no);
    for (uint32_t i = 0; i < no; ++i) read_pod(buffer, size, offset, out.second[i]);
    return out;
}

auto build_meta_payload(
    const std::unordered_map<std::string, std::string>& m)
    -> std::vector<uint8_t> {
    std::vector<uint8_t> out;
    auto count = static_cast<uint32_t>(m.size());
    append_pod(out, count);
    for (const auto& [k, v] : m) {
        auto kl = static_cast<uint32_t>(k.size());
        append_pod(out, kl);
        append_bytes(out, k.data(), kl);
        auto vl = static_cast<uint32_t>(v.size());
        append_pod(out, vl);
        append_bytes(out, v.data(), vl);
    }
    return out;
}

auto parse_meta_payload(const uint8_t* buffer, size_t size)
    -> std::unordered_map<std::string, std::string> {
    std::unordered_map<std::string, std::string> out;
    size_t offset = 0;
    uint32_t count = 0;
    read_pod(buffer, size, offset, count);
    for (uint32_t i = 0; i < count; ++i) {
        uint32_t kl = 0;
        read_pod(buffer, size, offset, kl);
        std::string k(kl, '\0');
        read_bytes(buffer, size, offset, k.data(), kl);
        uint32_t vl = 0;
        read_pod(buffer, size, offset, vl);
        std::string v(vl, '\0');
        read_bytes(buffer, size, offset, v.data(), vl);
        out.emplace(std::move(k), std::move(v));
    }
    return out;
}

// ---------------------------------------------------------------------------
// Inf-E1: MMPL section build/parse.
// ---------------------------------------------------------------------------

auto build_mmpl_payload(const MmplPlan& plan) -> std::vector<uint8_t> {
    std::vector<uint8_t> out;
    out.reserve(16 + plan.pool_sizes.size() * 8 + plan.placements.size() * 16);
    append_pod<uint8_t>(out, static_cast<uint8_t>(plan.alignment));
    append_pod<uint32_t>(out, static_cast<uint32_t>(plan.pool_sizes.size()));
    for (uint64_t s : plan.pool_sizes) append_pod(out, s);
    append_pod<uint32_t>(out, static_cast<uint32_t>(plan.placements.size()));
    for (const auto& p : plan.placements) {
        append_pod(out, p.tensor_id);
        append_pod<uint8_t>(out, p.pool_index);
        append_pod(out, p.offset);
    }
    return out;
}

auto parse_mmpl_payload(const uint8_t* buffer, size_t size) -> MmplPlan {
    MmplPlan plan;
    size_t offset = 0;
    uint8_t alignment = 0;
    read_pod(buffer, size, offset, alignment);
    plan.alignment = alignment ? alignment : 64;
    uint32_t num_pools = 0;
    read_pod(buffer, size, offset, num_pools);
    plan.pool_sizes.resize(num_pools);
    for (uint32_t i = 0; i < num_pools; ++i) {
        read_pod(buffer, size, offset, plan.pool_sizes[i]);
    }
    uint32_t num_placements = 0;
    read_pod(buffer, size, offset, num_placements);
    plan.placements.resize(num_placements);
    for (uint32_t i = 0; i < num_placements; ++i) {
        auto& p = plan.placements[i];
        read_pod(buffer, size, offset, p.tensor_id);
        read_pod(buffer, size, offset, p.pool_index);
        read_pod(buffer, size, offset, p.offset);
    }
    return plan;
}

// ---------------------------------------------------------------------------
// File I/O helpers.
// ---------------------------------------------------------------------------

auto read_file_bytes(const std::string& path) -> std::vector<uint8_t> {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        throw std::runtime_error("TZLiteReader: failed to open file: " + path);
    }
    auto size = static_cast<size_t>(file.tellg());
    file.seekg(0, std::ios::beg);
    std::vector<uint8_t> buffer(size);
    if (size > 0 &&
        !file.read(reinterpret_cast<char*>(buffer.data()),
                   static_cast<std::streamsize>(size))) {
        throw std::runtime_error("TZLiteReader: failed to read file: " + path);
    }
    return buffer;
}

}  // anonymous namespace

// ============================================================================
// TZLiteReader
// ============================================================================

auto TZLiteReader::load(const std::string& path) -> std::unique_ptr<LiteGraph> {
    auto buffer = read_file_bytes(path);
    return load(buffer.data(), buffer.size());
}

auto TZLiteReader::load(const void* data, size_t size) -> std::unique_ptr<LiteGraph> {
    return std::move(load_full(data, size).graph);
}

auto TZLiteReader::load_full(const std::string& path) -> LoadedModel {
    auto buffer = read_file_bytes(path);
    return load_full(buffer.data(), buffer.size());
}

auto TZLiteReader::load_full(const void* data, size_t size) -> LoadedModel {
    if (data == nullptr || size < sizeof(TZLiteHeader)) {
        throw std::runtime_error("TZLiteReader: invalid input data");
    }

    const auto* buffer = static_cast<const uint8_t*>(data);

    TZLiteHeader header{};
    std::memcpy(&header, buffer, sizeof(TZLiteHeader));
    if (header.magic != TZLITE_MAGIC) {
        throw std::runtime_error("TZLiteReader: bad magic — not a TZLITE file");
    }
    // Inf-E7: accept v1 (legacy) and v2 (with optional MMPL section). v1
    // files simply have no MMPL section to parse — the back-compat path
    // in the reader handles that transparently.
    if (header.version < TZLITE_VERSION_MIN_SUPPORTED ||
        header.version > TZLITE_VERSION) {
        throw std::runtime_error("TZLiteReader: unsupported TZLITE version " +
                                  std::to_string(header.version) +
                                  " (this build supports versions " +
                                  std::to_string(TZLITE_VERSION_MIN_SUPPORTED) +
                                  ".." + std::to_string(TZLITE_VERSION) + ")");
    }

    LoadedModel result;
    size_t offset = sizeof(TZLiteHeader);
    result.graph = deserialise_node_table(buffer, size, offset, header.num_nodes, header.version);

    // TLV section table: only parsed when weight_data_offset points at a
    // valid in-file offset past the node table.
    const size_t tlv_start = static_cast<size_t>(header.weight_data_offset);
    if (tlv_start < size) {
        // Reposition cursor — `weight_data_offset` is authoritative, the
        // node-table length we just computed is sanity-only.
        offset = tlv_start;
        uint32_t section_count = 0;
        read_pod(buffer, size, offset, section_count);
        for (uint32_t s = 0; s < section_count; ++s) {
            uint32_t tag = 0;
            uint64_t section_size = 0;
            read_pod(buffer, size, offset, tag);
            read_pod(buffer, size, offset, section_size);
            if (offset + section_size > size) {
                throw std::runtime_error(
                    "TZLiteReader: TLV section overruns end of buffer");
            }
            const uint8_t* payload = buffer + offset;
            switch (tag) {
                case TZLITE_TAG_TVAL:
                    result.tensor_values =
                        parse_tval_payload(payload, section_size);
                    break;
                case TZLITE_TAG_WGTS:
                    result.weight_blob.assign(payload, payload + section_size);
                    break;
                case TZLITE_TAG_IOSP: {
                    auto io = parse_iosp_payload(payload, section_size);
                    result.input_ids  = std::move(io.first);
                    result.output_ids = std::move(io.second);
                    break;
                }
                case TZLITE_TAG_META:
                    result.metadata = parse_meta_payload(payload, section_size);
                    break;
                case TZLITE_TAG_MMPL:
                    result.memory_plan = std::make_unique<MmplPlan>(
                        parse_mmpl_payload(payload, section_size));
                    break;
                default:
                    // Unknown tag — forward compatibility: skip silently. A
                    // future version may add sections this build doesn't know
                    // about; ignoring them lets the file still load.
                    break;
            }
            offset += section_size;
        }

        // Hand the I/O ids over to the graph so execute() knows the
        // calling convention without the runtime having to copy them again.
        if (!result.input_ids.empty()) {
            result.graph->set_input_ids(result.input_ids);
        }
        if (!result.output_ids.empty()) {
            result.graph->set_output_ids(result.output_ids);
        }
    }
    return result;
}

// ============================================================================
// TZLiteWriter
// ============================================================================

namespace {

// Compute a default node-table-aware weight_data_offset for the legacy
// graph-only save() path. Layout in that case has no TLV section, so the
// offset points at end-of-file.
auto compute_node_table_size(const LiteGraph& graph) -> size_t {
    size_t bytes = 0;
    for (const auto& node : graph.nodes()) {
        bytes += sizeof(uint16_t);                              // op
        bytes += sizeof(uint16_t);                              // num_inputs
        bytes += node.input_ids.size() * sizeof(int16_t);
        bytes += sizeof(uint16_t);                              // num_outputs
        bytes += node.output_ids.size() * sizeof(int16_t);
        bytes += 4 * sizeof(float);
        bytes += 4 * sizeof(int64_t);
        // Wave Inf-E5: v3+ extras suffix per node.
        if (TZLITE_VERSION >= 3) {
            bytes += sizeof(uint32_t);
            bytes += node.attrs.extra_i.size() * sizeof(int64_t);
            bytes += sizeof(uint32_t);
            bytes += node.attrs.extra_f.size() * sizeof(float);
        }
    }
    return bytes;
}

}  // namespace

auto TZLiteWriter::save(const LiteGraph& graph, const std::string& path) -> void {
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("TZLiteWriter: failed to open file: " + path);
    }

    TZLiteHeader header{};
    header.magic              = TZLITE_MAGIC;
    header.version            = TZLITE_VERSION;
    header.num_nodes          = static_cast<uint32_t>(graph.num_nodes());
    header.num_weights        = 0;
    header.weight_data_offset =
        sizeof(TZLiteHeader) + compute_node_table_size(graph);
    write_pod(file, header);

    auto node_bytes = serialise_node_table(graph, TZLITE_VERSION);
    if (!node_bytes.empty()) {
        file.write(reinterpret_cast<const char*>(node_bytes.data()),
                   static_cast<std::streamsize>(node_bytes.size()));
        if (!file) {
            throw std::runtime_error("TZLiteWriter: write failed");
        }
    }
    // L14 fix: the comment previously claimed this path was "byte-exactly
    // equal to pre-Phase-2 writers" — that hasn't been true since v3
    // added LiteAttributes::extra_i/extra_f suffix per node. The save
    // still writes a complete current-version file; just without the
    // optional TLV (MMPL/WGTS/IOSP/META) sections. Format-pinning tests
    // updated to match.
}

auto TZLiteWriter::save(const LiteGraph& graph,
                        const std::string& path,
                        const WriteOptions& opts) -> void {
    // Build the WGTS payload first so weight_offset/weight_nbytes are known
    // when we construct TVAL entries.
    std::vector<uint8_t> wgts_payload;
    std::unordered_map<int16_t, std::pair<uint64_t, uint64_t>> weight_locs;
    // Iterate weights in a deterministic order (sorted by tensor_id) for
    // stable byte output across runs.
    std::vector<int16_t> weight_ids;
    weight_ids.reserve(opts.weights.size());
    for (const auto& [tid, _] : opts.weights) weight_ids.push_back(tid);
    std::sort(weight_ids.begin(), weight_ids.end());

    for (int16_t tid : weight_ids) {
        Tensor t = opts.weights.at(tid);
        if (t.device().type != Device::Type::CPU) t = t.to(Device::cpu());
        if (!t.is_contiguous()) t = t.contiguous();
        const auto nbytes = static_cast<uint64_t>(t.numel() * t.dtype_size());
        const auto offset = static_cast<uint64_t>(wgts_payload.size());
        if (nbytes > 0) {
            append_bytes(wgts_payload, t.data_ptr(), nbytes);
        }
        weight_locs.emplace(tid, std::pair{offset, nbytes});
    }

    // Build TVAL entries. We cover every tensor_id mentioned by inputs,
    // outputs, weights, or graph nodes — readers use this to know how to
    // populate the tensor pool.
    std::vector<TensorValue> tvs;
    auto add_or_update = [&](int16_t tid, TensorSource src) -> TensorValue& {
        for (auto& tv : tvs) {
            if (tv.tensor_id == tid) {
                tv.source = src;
                return tv;
            }
        }
        TensorValue n;
        n.tensor_id = tid;
        n.source = src;
        tvs.push_back(std::move(n));
        return tvs.back();
    };

    // Inputs.
    for (size_t i = 0; i < opts.input_ids.size(); ++i) {
        auto& tv = add_or_update(opts.input_ids[i], TensorSource::Input);
        tv.input_index = static_cast<uint32_t>(i);
        if (auto it = opts.input_specs.find(tv.tensor_id);
            it != opts.input_specs.end()) {
            tv.dtype = it->second.first;
            tv.shape = it->second.second;
        }
    }
    // Weights.
    for (int16_t tid : weight_ids) {
        auto& tv = add_or_update(tid, TensorSource::Weight);
        const Tensor& src = opts.weights.at(tid);
        tv.dtype = src.dtype();
        tv.shape.clear();
        tv.shape.reserve(static_cast<size_t>(src.ndim()));
        for (int64_t d = 0; d < src.ndim(); ++d) tv.shape.push_back(src.size(d));
        const auto& loc = weight_locs.at(tid);
        tv.weight_offset = loc.first;
        tv.weight_nbytes = loc.second;
    }
    // Outputs and node intermediates: leave as Intermediate so the runtime
    // produces them via dispatch. The TVAL entry's shape is left empty for
    // dynamically-shaped intermediates (Phase 3 exporter fills it where
    // shape inference is available).
    for (int16_t id : opts.output_ids) {
        (void)add_or_update(id, TensorSource::Intermediate);
    }
    for (const auto& node : graph.nodes()) {
        for (int16_t id : node.output_ids) {
            (void)add_or_update(id, TensorSource::Intermediate);
        }
    }
    // Stable order by tensor_id for deterministic output.
    std::sort(tvs.begin(), tvs.end(),
              [](const TensorValue& a, const TensorValue& b) {
                  return a.tensor_id < b.tensor_id;
              });

    // Assemble section payloads.
    auto tval_payload = build_tval_payload(tvs);
    auto iosp_payload = build_iosp_payload(opts.input_ids, opts.output_ids);
    auto meta_payload = build_meta_payload(opts.metadata);

    // Serialise the node table once so we can compute weight_data_offset.
    auto node_bytes = serialise_node_table(graph, TZLITE_VERSION);
    const uint64_t tlv_offset =
        sizeof(TZLiteHeader) + static_cast<uint64_t>(node_bytes.size());

    // Header.
    TZLiteHeader header{};
    header.magic              = TZLITE_MAGIC;
    header.version            = TZLITE_VERSION;
    header.num_nodes          = static_cast<uint32_t>(graph.num_nodes());
    header.num_weights        = static_cast<uint32_t>(weight_ids.size());
    header.weight_data_offset = tlv_offset;

    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("TZLiteWriter: failed to open file: " + path);
    }
    write_pod(file, header);
    if (!node_bytes.empty()) {
        file.write(reinterpret_cast<const char*>(node_bytes.data()),
                   static_cast<std::streamsize>(node_bytes.size()));
    }

    // TLV section table. Sections are emitted in fixed order so byte output
    // is deterministic; readers locate by tag, not position.
    uint32_t section_count = 1;  // TVAL is always present
    if (!wgts_payload.empty()) ++section_count;
    if (!opts.input_ids.empty() || !opts.output_ids.empty()) ++section_count;
    if (!opts.metadata.empty()) ++section_count;
    if (opts.memory_plan) ++section_count;  // Inf-E1: MMPL section
    write_pod(file, section_count);

    auto write_section = [&](uint32_t tag, const std::vector<uint8_t>& payload) {
        write_pod(file, tag);
        write_pod<uint64_t>(file, static_cast<uint64_t>(payload.size()));
        if (!payload.empty()) {
            file.write(reinterpret_cast<const char*>(payload.data()),
                       static_cast<std::streamsize>(payload.size()));
        }
    };
    write_section(TZLITE_TAG_TVAL, tval_payload);
    if (!wgts_payload.empty()) write_section(TZLITE_TAG_WGTS, wgts_payload);
    if (!opts.input_ids.empty() || !opts.output_ids.empty()) {
        write_section(TZLITE_TAG_IOSP, iosp_payload);
    }
    if (!opts.metadata.empty()) write_section(TZLITE_TAG_META, meta_payload);
    if (opts.memory_plan) {
        auto mmpl_payload = build_mmpl_payload(*opts.memory_plan);
        write_section(TZLITE_TAG_MMPL, mmpl_payload);
    }

    if (!file) {
        throw std::runtime_error("TZLiteWriter: write failed");
    }
}

}  // namespace tenzor::lite
