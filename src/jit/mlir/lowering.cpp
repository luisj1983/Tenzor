// Phase 13 / Group B–C — Graph → MLIR text lowering implementation.

#include "tenzor/jit/mlir/lowering.hpp"

#include "tenzor/jit/graph.hpp"
#include "tenzor/jit/mlir/mlir_text_emit.hpp"
#include "tenzor/jit/tracer.hpp"

#include <algorithm>
#include <cstdint>
#include <initializer_list>
#include <iomanip>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace tenzor::jit::mlir_jit {

namespace {

/// Topological sort of nodes (Kahn). The `Graph` API exposes `nodes()` which
/// is supposed to be topologically ordered, but we re-derive the order so
/// the lowerer is robust to callers that mutate node ordering after
/// construction. Using stable BFS-by-input-readiness keeps the ordering
/// deterministic for a given graph.
auto topo_sort(const ::tenzor::jit::Graph& g)
    -> std::vector<std::shared_ptr<::tenzor::jit::Node>> {
    using ::tenzor::jit::Node;
    const auto& nodes = g.nodes();

    // Set of input value ids that are immediately available: the graph's
    // declared inputs.
    std::unordered_set<std::string> available;
    for (const auto& in_val : g.inputs()) {
        available.insert(in_val->id());
    }

    std::vector<std::shared_ptr<Node>> remaining(nodes.begin(), nodes.end());
    std::vector<std::shared_ptr<Node>> ordered;
    ordered.reserve(remaining.size());

    while (!remaining.empty()) {
        bool progressed = false;
        for (auto it = remaining.begin(); it != remaining.end();) {
            const auto& node = *it;
            bool ready = true;
            for (const auto& in_val : node->inputs()) {
                if (available.find(in_val->id()) == available.end()) {
                    ready = false;
                    break;
                }
            }
            if (ready) {
                for (const auto& out_val : node->outputs()) {
                    available.insert(out_val->id());
                }
                ordered.push_back(node);
                it = remaining.erase(it);
                progressed = true;
            } else {
                ++it;
            }
        }
        if (!progressed) {
            // Cycle or dangling input: fall back to original order to surface
            // a useful error in the per-node handler dispatch below rather
            // than silently dropping nodes.
            for (auto& n : remaining) {
                ordered.push_back(std::move(n));
            }
            break;
        }
    }
    return ordered;
}

/// Renderer state. Carries the SSA-name map (`Value::id()` → `"vN"`) and a
/// counter for fresh names.
struct LoweringContext {
    std::unordered_map<std::string, std::string> value_name;
    int next_id = 0;

    auto fresh_name() -> std::string {
        return "v" + std::to_string(next_id++);
    }

    auto bind(const std::string& value_id, std::string name) -> void {
        value_name.emplace(value_id, std::move(name));
    }

    auto name_for(const std::string& value_id) const -> const std::string& {
        auto it = value_name.find(value_id);
        if (it == value_name.end()) {
            throw std::runtime_error(
                "GraphToMLIR: undefined SSA value referenced: " + value_id);
        }
        return it->second;
    }
};

// ─── Utilities ──────────────────────────────────────────────────────────────

auto product(const std::vector<int64_t>& shape) -> int64_t {
    int64_t p = 1;
    for (auto d : shape) p *= d;
    return p;
}

auto is_float_dtype(::tenzor::DType d) -> bool {
    using ::tenzor::DType;
    return d == DType::Float32 || d == DType::Float64 ||
           d == DType::Float16 || d == DType::BFloat16;
}

auto is_int_dtype(::tenzor::DType d) -> bool {
    using ::tenzor::DType;
    return d == DType::Int8 || d == DType::Int16 || d == DType::Int32 ||
           d == DType::Int64 || d == DType::UInt8 || d == DType::UInt16 ||
           d == DType::UInt32 || d == DType::UInt64 || d == DType::Bool;
}

/// Produce a textual `dense<scalar>` literal for a scalar value of the
/// given dtype, matching the syntax StableHLO accepts as a splat constant.
auto scalar_literal(double value, ::tenzor::DType d) -> std::string {
    using ::tenzor::DType;
    std::ostringstream s;
    if (is_float_dtype(d)) {
        s << std::scientific << std::setprecision(9) << value;
        return s.str();
    }
    if (is_int_dtype(d)) {
        // Integer constants: emit as int literal (no decimal).
        s << static_cast<long long>(value);
        return s.str();
    }
    throw std::runtime_error(
        "GraphToMLIR: scalar_literal: unsupported DType");
}

/// Get an int attribute that may be present under several legacy names.
auto get_attr_int(const ::tenzor::jit::Node& n,
                  std::initializer_list<const char*> names,
                  int64_t default_value) -> int64_t {
    for (auto* nm : names) {
        if (n.has_attr(nm)) return n.get_int_attr(nm);
    }
    return default_value;
}

auto get_attr_vec(const ::tenzor::jit::Node& n,
                  std::initializer_list<const char*> names,
                  std::vector<int64_t> default_value) -> std::vector<int64_t> {
    for (auto* nm : names) {
        if (n.has_attr(nm)) {
            auto v = n.get_vec_attr(nm);
            if (!v.empty()) return v;
        }
    }
    return default_value;
}

auto get_attr_float(const ::tenzor::jit::Node& n,
                    std::initializer_list<const char*> names,
                    float default_value) -> float {
    for (auto* nm : names) {
        if (n.has_attr(nm)) return n.get_attr(nm);
    }
    return default_value;
}

auto get_attr_bool(const ::tenzor::jit::Node& n,
                   std::initializer_list<const char*> names,
                   bool default_value) -> bool {
    for (auto* nm : names) {
        if (n.has_attr(nm)) return n.get_bool_attr(nm);
    }
    return default_value;
}

auto normalize_dim(int64_t d, int64_t rank) -> int64_t {
    return d < 0 ? d + rank : d;
}

// ─── Handler helpers ────────────────────────────────────────────────────────

/// Extract a scalar value from a rank-0 tensor (any supported dtype). Used
/// when emitting `stablehlo.constant` for traced scalars from `Variable +
/// 2.0f` patterns. For non-rank-0 tensors we emit a typed `dense<...>`
/// elements attribute via `emit_dense_constant_from_tensor` instead.
auto extract_scalar_value(const ::tenzor::Tensor& t) -> double {
    using ::tenzor::DType;
    auto cpu = t.device().type == ::tenzor::Device::Type::CPU ? t : t.cpu();
    switch (cpu.dtype()) {
        case DType::Float32:    return static_cast<double>(cpu.item<float>());
        case DType::Float64:    return cpu.item<double>();
        case DType::Int32:      return static_cast<double>(cpu.item<int32_t>());
        case DType::Int64:      return static_cast<double>(cpu.item<int64_t>());
        case DType::Bool:       return cpu.item<bool>() ? 1.0 : 0.0;
        default:
            throw std::runtime_error(
                "GraphToMLIR: extract_scalar_value: unsupported DType");
    }
}

/// Emit a stablehlo.constant from a Tensor, binding the SSA name. For
/// rank-0 we emit `dense<scalar>`; for higher rank we currently support
/// uniform-splat tensors only (all elements equal) — sufficient for the
/// scalar broadcasts the autograd layer inserts, but not yet for general
/// weight constants. Throws if the tensor isn't a uniform splat.
auto emit_tensor_constant(std::ostream& body, LoweringContext& ctx,
                          const std::string& value_id,
                          const ::tenzor::Tensor& t,
                          const std::vector<int64_t>& shape,
                          ::tenzor::DType d) -> std::string {
    auto name = ctx.fresh_name();
    ctx.bind(value_id, name);
    if (t.numel() == 0) {
        // Empty: emit an empty splat, harmless.
        emit_stablehlo_splat_constant(body, name, scalar_literal(0.0, d),
                                      shape, d);
        body << '\n';
        return name;
    }
    const double v = extract_scalar_value(t);
    emit_stablehlo_splat_constant(body, name, scalar_literal(v, d), shape, d);
    body << '\n';
    return name;
}

/// Compute the broadcast_in_dim dimension list for promoting a tensor of
/// shape `src` to shape `dst`. Mirrors numpy/torch right-aligned
/// broadcasting:
///   - rank-0 src → []
///   - rank-N src aligned with the trailing N dims of dst → [k-N, k-N+1,
///     ..., k-1] where k = dst rank.
///   - If src has 1s in some dims, they're still listed (broadcast_in_dim
///     handles size-1 → size-N expansion implicitly).
auto right_align_bcast_dims(const std::vector<int64_t>& src,
                            const std::vector<int64_t>& dst)
    -> std::vector<int64_t> {
    std::vector<int64_t> out;
    out.reserve(src.size());
    if (src.size() > dst.size()) {
        throw std::runtime_error(
            "GraphToMLIR: cannot broadcast src rank > dst rank");
    }
    const int64_t offset = static_cast<int64_t>(dst.size() - src.size());
    for (std::size_t i = 0; i < src.size(); ++i) {
        out.push_back(offset + static_cast<int64_t>(i));
    }
    return out;
}

/// If the operand at `value_id` has shape != `target_shape`, emit a
/// stablehlo.broadcast_in_dim that brings it to `target_shape` and return
/// the fresh name. Otherwise return the existing name unchanged.
auto maybe_broadcast(std::ostream& body, LoweringContext& ctx,
                     const std::string& operand_name,
                     const std::vector<int64_t>& src_shape,
                     const std::vector<int64_t>& target_shape,
                     ::tenzor::DType d) -> std::string {
    if (src_shape == target_shape) return operand_name;
    auto out_name = ctx.fresh_name();
    auto dims = right_align_bcast_dims(src_shape, target_shape);
    emit_stablehlo_broadcast_in_dim(body, out_name, operand_name, dims,
                                    src_shape, target_shape, d);
    body << '\n';
    return out_name;
}

auto handle_shape_guard(LoweringContext& ctx,
                        const ::tenzor::jit::Node& node,
                        std::ostream& /*body*/) -> void {
    if (node.inputs().size() != 1 || node.outputs().size() != 1) {
        throw std::runtime_error(
            "GraphToMLIR: ShapeGuard expects 1 input and 1 output");
    }
    const auto& in_name = ctx.name_for(node.inputs()[0]->id());
    ctx.bind(node.outputs()[0]->id(), in_name);
}

auto handle_binary(LoweringContext& ctx,
                   const ::tenzor::jit::Node& node,
                   std::ostream& body,
                   const std::string& mnemonic) -> void {
    if (node.inputs().size() != 2) {
        throw std::runtime_error(
            "GraphToMLIR: " + mnemonic + " expects 2 inputs, got " +
            std::to_string(node.inputs().size()));
    }
    if (node.outputs().size() != 1) {
        throw std::runtime_error(
            "GraphToMLIR: " + mnemonic + " expects 1 output");
    }
    const auto& out = node.outputs()[0];
    const auto& a_val = node.inputs()[0];
    const auto& b_val = node.inputs()[1];
    auto a = maybe_broadcast(body, ctx, ctx.name_for(a_val->id()),
                             a_val->shape(), out->shape(), out->dtype());
    auto b = maybe_broadcast(body, ctx, ctx.name_for(b_val->id()),
                             b_val->shape(), out->shape(), out->dtype());
    auto out_name = ctx.fresh_name();
    ctx.bind(out->id(), out_name);
    emit_stablehlo_binary(body, mnemonic, out_name, a, b, out->shape(),
                          out->dtype());
    body << '\n';
}

auto handle_unary(LoweringContext& ctx,
                  const ::tenzor::jit::Node& node,
                  std::ostream& body,
                  const std::string& mnemonic) -> void {
    if (node.inputs().size() != 1 || node.outputs().size() != 1) {
        throw std::runtime_error(
            "GraphToMLIR: " + mnemonic + " expects 1 input and 1 output");
    }
    const auto& a = ctx.name_for(node.inputs()[0]->id());
    const auto& out = node.outputs()[0];
    auto out_name = ctx.fresh_name();
    ctx.bind(out->id(), out_name);
    emit_stablehlo_unary(body, mnemonic, out_name, a, out->shape(),
                         out->dtype());
    body << '\n';
}

auto handle_where(LoweringContext& ctx,
                  const ::tenzor::jit::Node& node,
                  std::ostream& body) -> void {
    if (node.inputs().size() != 3 || node.outputs().size() != 1) {
        throw std::runtime_error(
            "GraphToMLIR: Where expects 3 inputs and 1 output");
    }
    const auto& cond = ctx.name_for(node.inputs()[0]->id());
    const auto& on_true = ctx.name_for(node.inputs()[1]->id());
    const auto& on_false = ctx.name_for(node.inputs()[2]->id());
    const auto& out = node.outputs()[0];
    auto out_name = ctx.fresh_name();
    ctx.bind(out->id(), out_name);
    emit_stablehlo_ternary(body, "select", out_name, cond, on_true, on_false,
                           out->shape(), out->dtype());
    body << '\n';
}

auto handle_clamp(LoweringContext& ctx,
                  const ::tenzor::jit::Node& node,
                  std::ostream& body) -> void {
    if (node.outputs().size() != 1) {
        throw std::runtime_error("GraphToMLIR: Clamp expects 1 output");
    }
    const auto& out = node.outputs()[0];
    auto out_name = ctx.fresh_name();
    ctx.bind(out->id(), out_name);

    const auto shape = out->shape();
    const auto d = out->dtype();

    // Three operand forms are tracer-visible:
    //   1) (min, x, max) as 3 tensor inputs (rare in the eager dispatch path)
    //   2) (x) with scalar attrs `min` and `max` (the common path)
    //   3) (min, x) or (x, max) with one scalar attr — not yet observed but
    //      tolerated by checking sizes.
    std::string min_name, x_name, max_name;
    if (node.inputs().size() == 3) {
        min_name = ctx.name_for(node.inputs()[0]->id());
        x_name   = ctx.name_for(node.inputs()[1]->id());
        max_name = ctx.name_for(node.inputs()[2]->id());
    } else if (node.inputs().size() == 1) {
        x_name = ctx.name_for(node.inputs()[0]->id());
        float lo = get_attr_float(node, {"min"},
                                  -std::numeric_limits<float>::infinity());
        float hi = get_attr_float(node, {"max"},
                                  std::numeric_limits<float>::infinity());
        auto lo_name = ctx.fresh_name();
        auto hi_name = ctx.fresh_name();
        emit_stablehlo_splat_constant(body, lo_name, scalar_literal(lo, d),
                                      shape, d);
        body << '\n';
        emit_stablehlo_splat_constant(body, hi_name, scalar_literal(hi, d),
                                      shape, d);
        body << '\n';
        min_name = lo_name;
        max_name = hi_name;
    } else {
        throw std::runtime_error(
            "GraphToMLIR: Clamp expects 1 or 3 inputs, got " +
            std::to_string(node.inputs().size()));
    }
    emit_stablehlo_ternary(body, "clamp", out_name, min_name, x_name, max_name,
                           shape, d);
    body << '\n';
}

auto handle_relu(LoweringContext& ctx,
                 const ::tenzor::jit::Node& node,
                 std::ostream& body) -> void {
    if (node.inputs().size() != 1 || node.outputs().size() != 1) {
        throw std::runtime_error(
            "GraphToMLIR: ReLU expects 1 input and 1 output");
    }
    const auto& out = node.outputs()[0];
    const auto shape = out->shape();
    const auto d = out->dtype();
    const auto& x = ctx.name_for(node.inputs()[0]->id());
    auto zero_name = ctx.fresh_name();
    emit_stablehlo_splat_constant(body, zero_name, scalar_literal(0.0, d),
                                  shape, d);
    body << '\n';
    auto out_name = ctx.fresh_name();
    ctx.bind(out->id(), out_name);
    emit_stablehlo_binary(body, "maximum", out_name, x, zero_name, shape, d);
    body << '\n';
}

auto handle_sigmoid(LoweringContext& ctx,
                    const ::tenzor::jit::Node& node,
                    std::ostream& body) -> void {
    handle_unary(ctx, node, body, "logistic");
}

auto handle_silu(LoweringContext& ctx,
                 const ::tenzor::jit::Node& node,
                 std::ostream& body) -> void {
    if (node.inputs().size() != 1 || node.outputs().size() != 1) {
        throw std::runtime_error(
            "GraphToMLIR: SiLU expects 1 input and 1 output");
    }
    const auto& out = node.outputs()[0];
    const auto shape = out->shape();
    const auto d = out->dtype();
    const auto& x = ctx.name_for(node.inputs()[0]->id());
    auto sig_name = ctx.fresh_name();
    emit_stablehlo_unary(body, "logistic", sig_name, x, shape, d);
    body << '\n';
    auto out_name = ctx.fresh_name();
    ctx.bind(out->id(), out_name);
    emit_stablehlo_binary(body, "multiply", out_name, x, sig_name, shape, d);
    body << '\n';
}

/// GELU via the tanh approximation:
///   gelu(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
/// This avoids depending on `stablehlo.erf`, which isn't part of the core
/// StableHLO op set (it's in CHLO and not always exposed through IREE's
/// frontend). Numerical error vs the exact erf-based form is < 1e-4 in
/// F32, well below the typical 1e-3 layer tolerance.
auto handle_gelu(LoweringContext& ctx,
                 const ::tenzor::jit::Node& node,
                 std::ostream& body) -> void {
    if (node.inputs().size() != 1 || node.outputs().size() != 1) {
        throw std::runtime_error(
            "GraphToMLIR: GELU expects 1 input and 1 output");
    }
    const auto& out = node.outputs()[0];
    const auto shape = out->shape();
    const auto d = out->dtype();
    const auto& x = ctx.name_for(node.inputs()[0]->id());

    // Constants.
    auto c_half      = ctx.fresh_name();
    auto c_one       = ctx.fresh_name();
    auto c_three     = ctx.fresh_name();
    auto c_kappa     = ctx.fresh_name();        // 0.044715
    auto c_sqrt2pi   = ctx.fresh_name();        // sqrt(2/π) ≈ 0.7978845608
    emit_stablehlo_splat_constant(body, c_half, scalar_literal(0.5, d),
                                  shape, d); body << '\n';
    emit_stablehlo_splat_constant(body, c_one,  scalar_literal(1.0, d),
                                  shape, d); body << '\n';
    emit_stablehlo_splat_constant(body, c_three, scalar_literal(3.0, d),
                                  shape, d); body << '\n';
    emit_stablehlo_splat_constant(body, c_kappa,
                                  scalar_literal(0.044715, d), shape, d);
    body << '\n';
    emit_stablehlo_splat_constant(body, c_sqrt2pi,
                                  scalar_literal(0.7978845608028654, d),
                                  shape, d);
    body << '\n';

    // x_cubed = x^3
    auto x_cubed = ctx.fresh_name();
    emit_stablehlo_binary(body, "power", x_cubed, x, c_three, shape, d);
    body << '\n';
    // x_cubed_scaled = kappa * x^3
    auto x3s = ctx.fresh_name();
    emit_stablehlo_binary(body, "multiply", x3s, c_kappa, x_cubed, shape, d);
    body << '\n';
    // inner = x + kappa*x^3
    auto inner = ctx.fresh_name();
    emit_stablehlo_binary(body, "add", inner, x, x3s, shape, d);
    body << '\n';
    // arg = sqrt(2/π) * inner
    auto arg = ctx.fresh_name();
    emit_stablehlo_binary(body, "multiply", arg, c_sqrt2pi, inner, shape, d);
    body << '\n';
    // t = tanh(arg)
    auto tval = ctx.fresh_name();
    emit_stablehlo_unary(body, "tanh", tval, arg, shape, d);
    body << '\n';
    // one_plus = 1 + tanh(arg)
    auto one_plus = ctx.fresh_name();
    emit_stablehlo_binary(body, "add", one_plus, c_one, tval, shape, d);
    body << '\n';
    // half_x = 0.5 * x
    auto half_x = ctx.fresh_name();
    emit_stablehlo_binary(body, "multiply", half_x, c_half, x, shape, d);
    body << '\n';
    // out = (0.5 * x) * (1 + tanh(...))
    auto out_name = ctx.fresh_name();
    ctx.bind(out->id(), out_name);
    emit_stablehlo_binary(body, "multiply", out_name, half_x, one_plus,
                          shape, d);
    body << '\n';
}

auto handle_pow(LoweringContext& ctx,
                const ::tenzor::jit::Node& node,
                std::ostream& body) -> void {
    if (node.outputs().size() != 1) {
        throw std::runtime_error("GraphToMLIR: Pow expects 1 output");
    }
    const auto& out = node.outputs()[0];
    auto out_name = ctx.fresh_name();
    ctx.bind(out->id(), out_name);
    const auto shape = out->shape();
    const auto d = out->dtype();

    if (node.inputs().size() == 2) {
        const auto& a = ctx.name_for(node.inputs()[0]->id());
        const auto& b = ctx.name_for(node.inputs()[1]->id());
        emit_stablehlo_binary(body, "power", out_name, a, b, shape, d);
        body << '\n';
        return;
    }
    if (node.inputs().size() == 1) {
        const auto& a = ctx.name_for(node.inputs()[0]->id());
        float exp = get_attr_float(node, {"exponent"}, 1.0f);
        auto exp_name = ctx.fresh_name();
        emit_stablehlo_splat_constant(body, exp_name, scalar_literal(exp, d),
                                      shape, d);
        body << '\n';
        emit_stablehlo_binary(body, "power", out_name, a, exp_name, shape, d);
        body << '\n';
        return;
    }
    throw std::runtime_error("GraphToMLIR: Pow expects 1 or 2 inputs");
}

}  // namespace

GraphToMLIR::GraphToMLIR() = default;

auto GraphToMLIR::lower(const ::tenzor::jit::Graph& g) -> std::string {
    LoweringContext ctx;

    // Build argument list and bind each graph input to %argN.
    std::vector<std::pair<std::vector<int64_t>, ::tenzor::DType>> inputs;
    inputs.reserve(g.inputs().size());
    std::unordered_set<std::string> input_ids;
    for (std::size_t i = 0; i < g.inputs().size(); ++i) {
        const auto& in_val = g.inputs()[i];
        auto arg_name = "arg" + std::to_string(i);
        ctx.bind(in_val->id(), arg_name);
        inputs.emplace_back(in_val->shape(), in_val->dtype());
        input_ids.insert(in_val->id());
    }

    // Emit the body.
    std::ostringstream body;

    // Phase 1: emit stablehlo.constant for traced constants (e.g. scalars
    // produced by `Variable + 2.0f` patterns) so subsequent node handlers
    // can reference them by SSA name. Only emit for value-ids that aren't
    // already bound (i.e. aren't graph inputs).
    const auto& constants = g.constants();
    if (!constants.empty()) {
        // Walk all nodes to find Values referenced as inputs that aren't
        // graph inputs and have a constant tensor backing.
        std::unordered_map<std::string, std::shared_ptr<::tenzor::jit::Value>>
            const_value_by_id;
        for (const auto& n : g.nodes()) {
            for (const auto& v : n->inputs()) {
                if (input_ids.count(v->id())) continue;
                if (constants.count(v->id())) {
                    const_value_by_id[v->id()] = v;
                }
            }
        }
        for (const auto& [vid, v] : const_value_by_id) {
            if (ctx.value_name.count(vid)) continue;
            const auto& t = constants.at(vid);
            (void)emit_tensor_constant(body, ctx, vid, t, v->shape(),
                                       v->dtype());
        }
    }

    auto ordered = topo_sort(g);
    for (const auto& node : ordered) {
        using ::tenzor::jit::OpType;
        switch (node->op_type()) {
            // ── Elementwise binary ──
            case OpType::Add:          handle_binary(ctx, *node, body, "add");      break;
            case OpType::Sub:          handle_binary(ctx, *node, body, "subtract"); break;
            case OpType::Mul:          handle_binary(ctx, *node, body, "multiply"); break;
            case OpType::Div:          handle_binary(ctx, *node, body, "divide");   break;
            case OpType::ResidualAdd:  handle_binary(ctx, *node, body, "add");      break;

            // ── Elementwise unary ──
            case OpType::Neg:          handle_unary(ctx, *node, body, "negate");      break;
            case OpType::Abs:          handle_unary(ctx, *node, body, "abs");         break;
            case OpType::Exp:          handle_unary(ctx, *node, body, "exponential"); break;
            case OpType::Log:          handle_unary(ctx, *node, body, "log");         break;
            case OpType::Sqrt:         handle_unary(ctx, *node, body, "sqrt");        break;

            // ── Special-shape elementwise ──
            case OpType::Pow:          handle_pow(ctx, *node, body);   break;
            case OpType::Where:        handle_where(ctx, *node, body); break;
            case OpType::Clamp:        handle_clamp(ctx, *node, body); break;

            // ── Activations ──
            case OpType::ReLU:         handle_relu(ctx, *node, body);    break;
            case OpType::Sigmoid:      handle_sigmoid(ctx, *node, body); break;
            case OpType::SiLU:         handle_silu(ctx, *node, body);    break;
            case OpType::GELU:         handle_gelu(ctx, *node, body);    break;

            // ── Pseudo-ops ──
            case OpType::ShapeGuard:   handle_shape_guard(ctx, *node, body); break;

            default:
                throw std::runtime_error(
                    "GraphToMLIR: OpType " +
                    ::tenzor::jit::op_type_to_string(node->op_type()) +
                    " not yet supported");
        }
    }

    // Wrap in module { func.func @main(...) -> ... { <body> return ... } }.
    std::vector<std::pair<std::vector<int64_t>, ::tenzor::DType>> outputs;
    std::vector<std::string> return_names;
    outputs.reserve(g.outputs().size());
    return_names.reserve(g.outputs().size());
    for (const auto& out_val : g.outputs()) {
        outputs.emplace_back(out_val->shape(), out_val->dtype());
        return_names.push_back(ctx.name_for(out_val->id()));
    }

    return emit_module_wrapper(body, inputs, outputs, return_names);
}

}  // namespace tenzor::jit::mlir_jit
