/**
 * @file dispatch_bridge.hpp
 * @brief Translate a LiteNode (OpId + positional LiteAttributes) into a call
 *        into the main Tenzor dispatch table.
 *
 * Internal to src/lite/. Not part of the public Lite ABI.
 *
 * The wire format stores attributes positionally (4 floats + 4 int64s per
 * node) for compactness. The main dispatch table consumes a typed
 * `OpAttributes` map keyed by `AttrKey`. This bridge owns the per-OpId
 * mapping that says "for OpId::Conv2dForward, attrs.i[0] is StrideH, etc.".
 *
 * Audit I.3: docstring previously said "stub for Softmax" — Softmax /
 * LogSoftmax now read `attrs.i[0]` as the dim attribute (see
 * dispatch_bridge.cpp). Coverage continues to grow as later phases need
 * it; an OpId without an entry in the mapping table is invoked with
 * empty attributes — fine for nullary-attr ops, an error caught at
 * kernel-level for ops that require attrs.
 */

#pragma once

#include "tenzor/lite/lite_graph.hpp"
#include "tenzor/core/tensor.hpp"

#include <span>
#include <vector>

namespace tenzor::lite {

/** Invoke the registered kernel for `op` on `inputs` with positional `attrs`.
 *
 * Throws std::invalid_argument if the OpId isn't registered on CPU. Phase 5
 * generalises this to a caller-supplied device.
 */
auto run_op(LiteOpType op,
            std::span<const Tensor> inputs,
            const LiteAttributes& attrs) -> std::vector<Tensor>;

}  // namespace tenzor::lite
