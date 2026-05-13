/**
 * @file lite_graph.cpp
 * @brief LiteGraph::execute — Phase 1 graph execution over the main dispatch
 *        table.
 */

#include "tenzor/lite/lite_graph.hpp"
#include "tenzor/lite/tensor_bridge.hpp"

#include "dispatch_bridge.hpp"

#include <optional>
#include <stdexcept>
#include <string>

namespace tenzor::lite {

auto LiteGraph::add_node(LiteNode node) -> void {
    nodes_.push_back(std::move(node));
}

auto LiteGraph::num_nodes() const -> size_t {
    return nodes_.size();
}

auto LiteGraph::max_tensor_id() const -> int {
    int hi = -1;
    auto consider = [&](int16_t id) { if (id > hi) hi = id; };
    for (const auto& n : nodes_) {
        for (auto id : n.input_ids)  consider(id);
        for (auto id : n.output_ids) consider(id);
    }
    for (auto id : input_ids_)  consider(id);
    for (auto id : output_ids_) consider(id);
    return hi;
}

auto LiteGraph::execute(
    const std::vector<LiteTensor>& inputs,
    const std::unordered_map<int16_t, Tensor>& constants) const
    -> std::vector<LiteTensor> {
    // Phase 1 invariant: callers must declare the same number of input_ids
    // as they supply inputs to forward(). Phase 2 will additionally cover
    // weight tensors by tensor_id from the SafeTensors blob.
    if (inputs.size() != input_ids_.size()) {
        throw std::invalid_argument(
            "LiteGraph::execute: input count (" + std::to_string(inputs.size()) +
            ") does not match declared input_ids (" +
            std::to_string(input_ids_.size()) + ")");
    }

    const int max_id = max_tensor_id();
    std::vector<std::optional<Tensor>> pool(static_cast<size_t>(max_id + 1));

    // Populate inputs as non-owning views over the caller's LiteTensors.
    for (size_t i = 0; i < inputs.size(); ++i) {
        const auto tid = static_cast<size_t>(input_ids_[i]);
        if (tid >= pool.size()) {
            throw std::out_of_range("LiteGraph::execute: input_id out of range");
        }
        pool[tid] = view_as_tensor(inputs[i]);
    }

    // Pre-populate constant tensors (weights from WGTS, baked-in literals).
    for (const auto& [tid, tensor] : constants) {
        const auto idx = static_cast<size_t>(tid);
        if (idx >= pool.size()) {
            throw std::out_of_range(
                "LiteGraph::execute: constant tensor_id " + std::to_string(tid) +
                " is outside the graph's tensor range");
        }
        pool[idx] = tensor;
    }

    // Execute nodes in order. Phase 1 assumes the graph is already
    // topologically sorted at construction (true for graphs built by hand or
    // emitted by the JIT tracer in Phase 3).
    std::vector<Tensor> arg_buf;
    for (const auto& node : nodes_) {
        arg_buf.clear();
        arg_buf.reserve(node.input_ids.size());
        for (auto in_id : node.input_ids) {
            const auto tid = static_cast<size_t>(in_id);
            if (tid >= pool.size() || !pool[tid].has_value()) {
                throw std::runtime_error(
                    "LiteGraph::execute: node references undefined tensor_id " +
                    std::to_string(in_id));
            }
            arg_buf.push_back(*pool[tid]);
        }

        auto outs = run_op(node.op, std::span<const Tensor>{arg_buf}, node.attrs);

        if (outs.size() != node.output_ids.size()) {
            throw std::runtime_error(
                "LiteGraph::execute: kernel produced " +
                std::to_string(outs.size()) + " outputs but node declares " +
                std::to_string(node.output_ids.size()));
        }
        for (size_t j = 0; j < outs.size(); ++j) {
            const auto tid = static_cast<size_t>(node.output_ids[j]);
            if (tid >= pool.size()) {
                throw std::out_of_range(
                    "LiteGraph::execute: output_id out of range");
            }
            pool[tid] = std::move(outs[j]);
        }
    }

    // Gather outputs as owning LiteTensors. Phase 2 makes this zero-copy
    // via the arena memory plan.
    std::vector<LiteTensor> result;
    result.reserve(output_ids_.size());
    for (auto out_id : output_ids_) {
        const auto tid = static_cast<size_t>(out_id);
        if (tid >= pool.size() || !pool[tid].has_value()) {
            throw std::runtime_error(
                "LiteGraph::execute: output_id " + std::to_string(out_id) +
                " was never produced");
        }
        result.push_back(to_lite_tensor(*pool[tid]));
    }
    return result;
}

}  // namespace tenzor::lite
