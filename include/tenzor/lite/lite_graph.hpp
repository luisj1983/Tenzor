/**
 * @file lite_graph.hpp
 * @brief Static execution graph for the lite inference runtime
 *
 * LiteGraph represents a topologically-sorted sequence of operations ready for
 * inference. Nodes reference tensors by integer ID into a flat tensor pool
 * managed by the runtime at execute() time. Each node's op is an `OpId` from
 * the main Tenzor dispatch table — Lite does not maintain a parallel kernel
 * registry; it routes through `tenzor::dispatch<OpId>(...)`.
 */

#pragma once

#include "runtime.hpp"
#include "../core/tensor.hpp"
#include "../ops/op_id.hpp"

#include <cstdint>
#include <unordered_map>
#include <vector>
#include <vector>

namespace tenzor::lite {

// ============================================================================
// LiteOpType — alias for the project-wide OpId enum
// ============================================================================
//
// The wire format serialises the op as a `uint16_t` matching this enum's
// underlying type, which is identical to `OpId`'s. The Lite runtime accepts
// any OpId registered on the chosen backend's dispatch table; the only
// per-OpId code in src/lite/ is the positional-attribute mapping in
// dispatch_bridge.cpp.
using LiteOpType = OpId;

// ============================================================================
// LiteAttributes — small-buffer attribute storage
// ============================================================================
//
// Wire layout is positional: each (OpId, slot) pair has a documented meaning
// defined in src/lite/dispatch_bridge.cpp (e.g. for OpId::Conv2dForward,
// attrs.i[0]..[3] are Stride, Padding, Dilation, Groups). The bridge layer
// translates these into the typed `OpAttributes` (AttrKey -> AttrValue) map
// the existing kernels consume.
struct LiteAttributes {
    float f[4]{};
    int64_t i[4]{};
    // Wave Inf-E5 (deferred → landed): variable-length extras for ops that
    // need more than 4 floats / 4 ints of positional state. RNN/LSTM/GRU
    // use it for (hidden_size, num_layers, bidirectional, batch_first,
    // dropout_at_layer_k...). MultiheadAttention uses it for
    // (num_heads, head_dim, embed_dim, causal, kv_seq_len, ...).
    //
    // Encoding is purely positional; each consumer in dispatch_bridge.cpp
    // documents the slot meaning inline next to the OpId switch arm.
    //
    // The serialiser at src/lite/model_format.cpp writes this as
    //   u32 extra_i_count; i64[extra_i_count]
    //   u32 extra_f_count; f32[extra_f_count]
    // appended to each node's record. v1 files (no extras) round-trip
    // unchanged because both counts default to 0.
    std::vector<int64_t> extra_i;
    std::vector<float>   extra_f;
};

// ============================================================================
// LiteNode — one operation in the graph
// ============================================================================
struct LiteNode {
    LiteOpType op{};
    std::vector<int16_t> input_ids;   ///< tensor_ids consumed
    std::vector<int16_t> output_ids;  ///< tensor_ids produced
    LiteAttributes attrs;
};

// ============================================================================
// LiteGraph — the full execution plan
// ============================================================================
//
// Tensor-id semantics:
//   - Inputs to the graph: tensor_ids listed by set_input_ids(). Position i
//     in that list maps to forward()'s inputs[i].
//   - Weights / constants: tensor_ids that appear as node inputs but are
//     never node outputs and are not graph inputs. Phase 1 has no weights;
//     Phase 2 populates them from the SafeTensors blob.
//   - Intermediates: tensor_ids produced by nodes and consumed by later nodes.
//   - Outputs: tensor_ids listed by set_output_ids(). Returned by execute()
//     in that order.
class LiteGraph {
public:
    /** Run the graph with the given external inputs and optional constants.
     *
     * `constants` pre-populates tensor_ids that are not graph inputs and not
     * produced by nodes — typically model weights loaded from a `.tzlite`
     * file's WGTS section. Tensors in `constants` may be non-owning views
     * (e.g. into an mmap'd buffer); the runtime never mutates them.
     *
     * Returns one LiteTensor per output_id in the order set by
     * set_output_ids(). Each returned LiteTensor owns its data (allocated
     * fresh per call). Phase 5 introduces zero-copy arena-backed outputs.
     */
    auto execute(const std::vector<LiteTensor>& inputs,
                 const std::unordered_map<int16_t, Tensor>& constants = {}) const
        -> std::vector<LiteTensor>;

    /** Append a node to the plan. */
    auto add_node(LiteNode node) -> void;

    /** Node count. */
    auto num_nodes() const -> size_t;

    /** Read-only node list (for serialisation). */
    auto nodes() const -> const std::vector<LiteNode>& { return nodes_; }

    /** Declare which tensor_ids correspond to the caller's inputs. */
    auto set_input_ids(std::vector<int16_t> ids) -> void { input_ids_ = std::move(ids); }
    /** Declare which tensor_ids to return from execute(). */
    auto set_output_ids(std::vector<int16_t> ids) -> void { output_ids_ = std::move(ids); }

    auto input_ids() const -> const std::vector<int16_t>& { return input_ids_; }
    auto output_ids() const -> const std::vector<int16_t>& { return output_ids_; }

    /** Highest tensor_id referenced anywhere in the graph (-1 if empty). */
    auto max_tensor_id() const -> int;

private:
    std::vector<LiteNode> nodes_;
    std::vector<int16_t> input_ids_;
    std::vector<int16_t> output_ids_;
};

}  // namespace tenzor::lite
