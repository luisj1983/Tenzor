// Phase 13 / Group B–C — Graph → MLIR text lowering implementation.

#include "tenzor/jit/mlir/lowering.hpp"

#include "tenzor/jit/graph.hpp"
#include "tenzor/jit/mlir/mlir_text_emit.hpp"
#include "tenzor/jit/tracer.hpp"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <iomanip>
#include <limits>
#include <numeric>
#include <sstream>
#include <type_traits>
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

/// Emit `tensor<DxDx...xT>` to the stream — same logic as the private
/// helper in mlir_text_emit.cpp, replicated here so handlers in this
/// translation unit can render types in custom forms that the focused
/// emitters don't cover (e.g. stablehlo.gather's operand-tuple syntax).
auto write_tensor_type_for_emit(std::ostream& os,
                                const std::vector<int64_t>& shape,
                                ::tenzor::DType d) -> void {
    os << "tensor<";
    for (auto dim : shape) os << dim << 'x';
    os << mlir_type_name(d) << '>';
}

// ─── Handler helpers ────────────────────────────────────────────────────────

/// Extract a scalar value from a rank-0 (or numel==1) tensor.
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

/// Render a per-element MLIR float/int literal at full precision.
template <typename T>
auto render_one(std::ostream& os, T v) -> void {
    if constexpr (std::is_floating_point_v<T>) {
        os << std::scientific << std::setprecision(9)
           << static_cast<double>(v);
    } else {
        os << static_cast<long long>(v);
    }
}

/// Emit a stablehlo.constant whose payload is the full element-list of
/// `t`. For numel == 1 this collapses to the splat-constant form; for
/// larger tensors we emit a `dense<[...]>` elements attribute with the
/// same shape. Used for weight constants closed-over by JIT'd lambdas.
auto emit_tensor_constant(std::ostream& body, LoweringContext& ctx,
                          const std::string& value_id,
                          const ::tenzor::Tensor& t,
                          const std::vector<int64_t>& shape,
                          ::tenzor::DType d) -> std::string {
    auto name = ctx.fresh_name();
    ctx.bind(value_id, name);
    if (t.numel() == 0) {
        emit_stablehlo_splat_constant(body, name, scalar_literal(0.0, d),
                                      shape, d);
        body << '\n';
        return name;
    }
    if (t.numel() == 1) {
        const double v = extract_scalar_value(t);
        emit_stablehlo_splat_constant(body, name, scalar_literal(v, d),
                                      shape, d);
        body << '\n';
        return name;
    }

    // Multi-element constant: emit a `dense<[v0, v1, ...]>` payload with
    // the actual values. The element order is row-major (MLIR's
    // standard), and we recurse the shape to emit nested brackets so the
    // attribute matches the tensor type exactly. Materializes one float
    // per element — fine for the kilobyte-scale weight constants typical
    // of JIT'd inference graphs.
    using ::tenzor::DType;
    auto cpu = t.device().type == ::tenzor::Device::Type::CPU ? t : t.cpu();

    std::ostringstream payload;
    std::function<void(int64_t, int64_t&, int64_t)> emit_dim =
        [&](int64_t dim_idx, int64_t& flat_idx, int64_t) {
        const int64_t extent = shape[dim_idx];
        payload << '[';
        for (int64_t i = 0; i < extent; ++i) {
            if (i != 0) payload << ", ";
            if (dim_idx + 1 == static_cast<int64_t>(shape.size())) {
                switch (d) {
                    case DType::Float32:
                        render_one(payload, cpu.data<float>()[flat_idx++]);
                        break;
                    case DType::Float64:
                        render_one(payload, cpu.data<double>()[flat_idx++]);
                        break;
                    case DType::Int32:
                        render_one(payload, cpu.data<int32_t>()[flat_idx++]);
                        break;
                    case DType::Int64:
                        render_one(payload, cpu.data<int64_t>()[flat_idx++]);
                        break;
                    default:
                        throw std::runtime_error(
                            "GraphToMLIR: emit_tensor_constant: "
                            "unsupported DType for multi-element constant");
                }
            } else {
                emit_dim(dim_idx + 1, flat_idx, 0);
            }
        }
        payload << ']';
    };
    if (shape.empty()) {
        // Scalar - shouldn't reach here (numel handled above) but be safe.
        const double v = extract_scalar_value(cpu);
        emit_stablehlo_splat_constant(body, name, scalar_literal(v, d),
                                      shape, d);
        body << '\n';
        return name;
    }
    int64_t flat_idx = 0;
    emit_dim(0, flat_idx, 0);
    emit_stablehlo_splat_constant(body, name, payload.str(), shape, d);
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

/// Determine the reduction dims and keepdim flag from a traced reduce
/// node. Returns a pair (dims, keepdim) where dims are positive axes in
/// [0, input_rank). If the node has no "dim" attribute, reduces over all
/// dims.
auto resolve_reduce_dims(const ::tenzor::jit::Node& node,
                         const std::vector<int64_t>& input_shape,
                         const std::vector<int64_t>& output_shape)
    -> std::pair<std::vector<int64_t>, bool> {
    const int64_t in_rank = static_cast<int64_t>(input_shape.size());
    std::vector<int64_t> dims;
    if (node.has_attr("dim")) {
        // Single int dim (the common eager path).
        auto raw = node.get_int_attr("dim");
        dims.push_back(normalize_dim(raw, in_rank));
    } else if (node.has_attr("dims")) {
        auto vec = node.get_vec_attr("dims");
        for (auto d : vec) dims.push_back(normalize_dim(d, in_rank));
    } else {
        // Reduce all dims.
        for (int64_t i = 0; i < in_rank; ++i) dims.push_back(i);
    }
    // keepdim is true when the output keeps the rank of the input.
    bool keepdim = output_shape.size() == input_shape.size();
    return {std::move(dims), keepdim};
}

/// Emit reduce + optional reshape-to-keepdim. Returns the SSA name of
/// the (post-keepdim) result.
auto emit_reduce_with_keepdim(LoweringContext& ctx, std::ostream& body,
                              const std::string& operand_name,
                              const std::vector<int64_t>& operand_shape,
                              const std::vector<int64_t>& result_shape,
                              const std::vector<int64_t>& dims,
                              bool keepdim, const std::string& reducer,
                              const std::string& init_literal,
                              ::tenzor::DType d) -> std::string {
    // 1) Init constant: rank-0 splat in dtype `d`.
    auto init_name = ctx.fresh_name();
    emit_stablehlo_splat_constant(body, init_name, init_literal, {}, d);
    body << '\n';

    // 2) Shape of the reduced (no-keepdim) intermediate.
    std::vector<int64_t> reduced_shape;
    {
        std::unordered_set<int64_t> drop(dims.begin(), dims.end());
        for (std::size_t i = 0; i < operand_shape.size(); ++i) {
            if (!drop.count(static_cast<int64_t>(i))) {
                reduced_shape.push_back(operand_shape[i]);
            }
        }
    }

    auto reduced_name = ctx.fresh_name();
    emit_stablehlo_reduce(body, reduced_name, operand_name, init_name,
                          reducer, dims, operand_shape, reduced_shape, d);
    body << '\n';

    if (!keepdim) return reduced_name;
    // If keepdim → reshape reduced_shape back to result_shape (which has
    // size-1 in each reduced axis).
    if (reduced_shape == result_shape) return reduced_name;
    auto kd_name = ctx.fresh_name();
    emit_stablehlo_reshape(body, kd_name, reduced_name, reduced_shape,
                           result_shape, d);
    body << '\n';
    return kd_name;
}

auto handle_sum(LoweringContext& ctx,
                const ::tenzor::jit::Node& node,
                std::ostream& body) -> void {
    if (node.inputs().size() != 1 || node.outputs().size() != 1) {
        throw std::runtime_error("GraphToMLIR: Sum expects 1 input, 1 output");
    }
    const auto& in_val  = node.inputs()[0];
    const auto& out_val = node.outputs()[0];
    auto [dims, keepdim] = resolve_reduce_dims(node, in_val->shape(),
                                                out_val->shape());
    auto out_name = emit_reduce_with_keepdim(
        ctx, body, ctx.name_for(in_val->id()), in_val->shape(),
        out_val->shape(), dims, keepdim, "add",
        scalar_literal(0.0, out_val->dtype()), out_val->dtype());
    ctx.bind(out_val->id(), out_name);
}

auto handle_max(LoweringContext& ctx,
                const ::tenzor::jit::Node& node,
                std::ostream& body) -> void {
    if (node.inputs().size() != 1 || node.outputs().size() != 1) {
        throw std::runtime_error("GraphToMLIR: Max expects 1 input, 1 output");
    }
    const auto& in_val  = node.inputs()[0];
    const auto& out_val = node.outputs()[0];
    auto [dims, keepdim] = resolve_reduce_dims(node, in_val->shape(),
                                                out_val->shape());
    // -inf init for float (hex bit pattern). For integers fall back to a
    // very-negative integer; backing dtypes wider than int32 still get a
    // valid (just non-minimal) lower bound.
    std::string init_lit;
    const auto d = out_val->dtype();
    if (d == ::tenzor::DType::Float32) {
        init_lit = "0xFF800000";  // IEEE 754 binary32 -inf
    } else if (d == ::tenzor::DType::Float64) {
        init_lit = "0xFFF0000000000000";  // IEEE 754 binary64 -inf
    } else if (d == ::tenzor::DType::Float16) {
        init_lit = "0xFC00";  // IEEE 754 binary16 -inf
    } else if (d == ::tenzor::DType::BFloat16) {
        init_lit = "0xFF80";  // bfloat16 -inf
    } else if (is_float_dtype(d)) {
        init_lit = "0xFF800000";  // fallback
    } else {
        init_lit = "-2147483648";
    }
    auto out_name = emit_reduce_with_keepdim(
        ctx, body, ctx.name_for(in_val->id()), in_val->shape(),
        out_val->shape(), dims, keepdim, "maximum", init_lit, d);
    ctx.bind(out_val->id(), out_name);
}

auto handle_mean(LoweringContext& ctx,
                 const ::tenzor::jit::Node& node,
                 std::ostream& body) -> void {
    if (node.inputs().size() != 1 || node.outputs().size() != 1) {
        throw std::runtime_error(
            "GraphToMLIR: Mean expects 1 input, 1 output");
    }
    const auto& in_val  = node.inputs()[0];
    const auto& out_val = node.outputs()[0];
    auto [dims, keepdim] = resolve_reduce_dims(node, in_val->shape(),
                                                out_val->shape());
    const auto d = out_val->dtype();
    // First compute the sum.
    auto sum_name = emit_reduce_with_keepdim(
        ctx, body, ctx.name_for(in_val->id()), in_val->shape(),
        out_val->shape(), dims, keepdim, "add", scalar_literal(0.0, d), d);

    // N = product of reduced extents.
    int64_t N = 1;
    for (auto k : dims) N *= in_val->shape()[k];

    auto n_const = ctx.fresh_name();
    emit_stablehlo_splat_constant(body, n_const,
                                  scalar_literal(static_cast<double>(N), d),
                                  out_val->shape(), d);
    body << '\n';

    auto out_name = ctx.fresh_name();
    ctx.bind(out_val->id(), out_name);
    emit_stablehlo_binary(body, "divide", out_name, sum_name, n_const,
                          out_val->shape(), d);
    body << '\n';
}

/// Softmax along a single dim, numerically stable form:
///   m = reduce_max(x, dim) (no keepdim, then broadcast back)
///   z = x - m  (broadcast)
///   e = exp(z)
///   s = reduce_sum(e, dim) (no keepdim, then broadcast)
///   out = e / s
auto handle_softmax(LoweringContext& ctx,
                    const ::tenzor::jit::Node& node,
                    std::ostream& body) -> void {
    if (node.inputs().size() != 1 || node.outputs().size() != 1) {
        throw std::runtime_error(
            "GraphToMLIR: Softmax expects 1 input, 1 output");
    }
    const auto& in_val  = node.inputs()[0];
    const auto& out_val = node.outputs()[0];
    const auto shape = out_val->shape();
    const auto d     = out_val->dtype();
    const auto& x    = ctx.name_for(in_val->id());

    const int64_t rank = static_cast<int64_t>(shape.size());
    int64_t dim = rank == 0 ? 0 : rank - 1;
    if (node.has_attr("dim")) {
        dim = normalize_dim(node.get_int_attr("dim"), rank);
    }
    std::vector<int64_t> reduce_dims = {dim};

    // Reduced (no-keepdim) shape.
    std::vector<int64_t> reduced_shape;
    for (int64_t i = 0; i < rank; ++i) {
        if (i != dim) reduced_shape.push_back(shape[i]);
    }
    // Broadcast dimensions for re-expanding reduced→shape: every dim
    // index except `dim` maps from reduced index → result index.
    std::vector<int64_t> bcast_dims;
    bcast_dims.reserve(reduced_shape.size());
    for (int64_t i = 0; i < rank; ++i) {
        if (i != dim) bcast_dims.push_back(i);
    }

    // 1) reduce_max — -inf init (hex bit pattern per dtype).
    std::string init_max = "-2147483648";
    if (d == ::tenzor::DType::Float32)       init_max = "0xFF800000";
    else if (d == ::tenzor::DType::Float64)  init_max = "0xFFF0000000000000";
    else if (d == ::tenzor::DType::Float16)  init_max = "0xFC00";
    else if (d == ::tenzor::DType::BFloat16) init_max = "0xFF80";
    auto init_m = ctx.fresh_name();
    emit_stablehlo_splat_constant(body, init_m, init_max, {}, d); body << '\n';
    auto m_name = ctx.fresh_name();
    emit_stablehlo_reduce(body, m_name, x, init_m, "maximum", reduce_dims,
                          shape, reduced_shape, d);
    body << '\n';
    // broadcast m back
    auto m_b = ctx.fresh_name();
    emit_stablehlo_broadcast_in_dim(body, m_b, m_name, bcast_dims,
                                    reduced_shape, shape, d);
    body << '\n';
    // z = x - m
    auto z = ctx.fresh_name();
    emit_stablehlo_binary(body, "subtract", z, x, m_b, shape, d);
    body << '\n';
    // e = exp(z)
    auto e = ctx.fresh_name();
    emit_stablehlo_unary(body, "exponential", e, z, shape, d);
    body << '\n';
    // s = reduce_sum(e)
    auto init_s = ctx.fresh_name();
    emit_stablehlo_splat_constant(body, init_s, scalar_literal(0.0, d), {}, d);
    body << '\n';
    auto s_name = ctx.fresh_name();
    emit_stablehlo_reduce(body, s_name, e, init_s, "add", reduce_dims,
                          shape, reduced_shape, d);
    body << '\n';
    auto s_b = ctx.fresh_name();
    emit_stablehlo_broadcast_in_dim(body, s_b, s_name, bcast_dims,
                                    reduced_shape, shape, d);
    body << '\n';
    // out = e / s
    auto out_name = ctx.fresh_name();
    ctx.bind(out_val->id(), out_name);
    emit_stablehlo_binary(body, "divide", out_name, e, s_b, shape, d);
    body << '\n';
}

/// Lower MatMul for rank-2 or higher inputs.
///   rank-2: (M, K) @ (K, N) → (M, N) with lhs_contracting=[1],
///           rhs_contracting=[0].
///   rank-N (≥3): the leading N-2 dims are batch dims (broadcast batching
///           in StableHLO requires explicit broadcast, but Tenzor's eager
///           matmul broadcasts implicitly; for graph traces the batch
///           dims are concrete and equal on both sides).
auto handle_matmul(LoweringContext& ctx,
                   const ::tenzor::jit::Node& node,
                   std::ostream& body) -> void {
    if (node.inputs().size() != 2 || node.outputs().size() != 1) {
        throw std::runtime_error(
            "GraphToMLIR: MatMul expects 2 inputs and 1 output");
    }
    const auto& lhs = node.inputs()[0];
    const auto& rhs = node.inputs()[1];
    const auto& out = node.outputs()[0];

    const auto lhs_shape = lhs->shape();
    const auto rhs_shape = rhs->shape();
    const int64_t lr = static_cast<int64_t>(lhs_shape.size());
    const int64_t rr = static_cast<int64_t>(rhs_shape.size());
    if (lr < 2 || rr < 2) {
        throw std::runtime_error(
            "GraphToMLIR: MatMul requires rank ≥ 2 on both sides");
    }

    // Batch dims: matched leading dims on both sides. When one side has
    // fewer batch dims, we'd need to broadcast — defer that to higher
    // ranks for now and require both ranks equal.
    std::vector<int64_t> lhs_batch, rhs_batch;
    if (lr == rr) {
        for (int64_t i = 0; i < lr - 2; ++i) {
            lhs_batch.push_back(i);
            rhs_batch.push_back(i);
        }
    }
    std::vector<int64_t> lhs_contracting = {lr - 1};
    std::vector<int64_t> rhs_contracting = {rr - 2};

    auto out_name = ctx.fresh_name();
    ctx.bind(out->id(), out_name);
    emit_stablehlo_dot_general(body, out_name, ctx.name_for(lhs->id()),
                               ctx.name_for(rhs->id()), lhs_batch, rhs_batch,
                               lhs_contracting, rhs_contracting, lhs_shape,
                               rhs_shape, out->shape(), out->dtype());
    body << '\n';
}

/// Bmm: rank-3 batched matmul, dim 0 is batch. (B, M, K) @ (B, K, N) → (B, M, N).
auto handle_bmm(LoweringContext& ctx,
                const ::tenzor::jit::Node& node,
                std::ostream& body) -> void {
    if (node.inputs().size() != 2 || node.outputs().size() != 1) {
        throw std::runtime_error(
            "GraphToMLIR: Bmm expects 2 inputs and 1 output");
    }
    const auto& lhs = node.inputs()[0];
    const auto& rhs = node.inputs()[1];
    const auto& out = node.outputs()[0];
    auto out_name = ctx.fresh_name();
    ctx.bind(out->id(), out_name);
    emit_stablehlo_dot_general(body, out_name, ctx.name_for(lhs->id()),
                               ctx.name_for(rhs->id()),
                               /*lhs_batch=*/{0}, /*rhs_batch=*/{0},
                               /*lhs_contracting=*/{2},
                               /*rhs_contracting=*/{1},
                               lhs->shape(), rhs->shape(), out->shape(),
                               out->dtype());
    body << '\n';
}

/// Linear: `out = x @ W^T + b`. The eager dispatch traces inputs as
/// (x, W) or (x, W, b). Output Value's shape is the final shape.
auto handle_linear(LoweringContext& ctx,
                   const ::tenzor::jit::Node& node,
                   std::ostream& body) -> void {
    if (node.outputs().size() != 1) {
        throw std::runtime_error("GraphToMLIR: Linear expects 1 output");
    }
    if (node.inputs().size() != 2 && node.inputs().size() != 3) {
        throw std::runtime_error(
            "GraphToMLIR: Linear expects 2 or 3 inputs (x, W [, b])");
    }
    const auto& x = node.inputs()[0];
    const auto& w = node.inputs()[1];
    const auto& out = node.outputs()[0];
    const auto x_shape = x->shape();
    const auto w_shape = w->shape();
    const auto out_shape = out->shape();
    const auto d = out->dtype();

    // For x [..., in_features] @ W [out_features, in_features]^T → out
    // [..., out_features], dot_general contracts last dim of x with last
    // dim of W (since W is stored as (out_features, in_features)).
    const int64_t xr = static_cast<int64_t>(x_shape.size());
    const int64_t wr = static_cast<int64_t>(w_shape.size());
    std::vector<int64_t> lhs_contracting = {xr - 1};
    std::vector<int64_t> rhs_contracting = {wr - 1};

    auto matmul_name = ctx.fresh_name();
    emit_stablehlo_dot_general(body, matmul_name, ctx.name_for(x->id()),
                               ctx.name_for(w->id()),
                               /*lhs_batch=*/{}, /*rhs_batch=*/{},
                               lhs_contracting, rhs_contracting, x_shape,
                               w_shape, out_shape, d);
    body << '\n';

    if (node.inputs().size() == 3) {
        const auto& b = node.inputs()[2];
        auto bias_b = maybe_broadcast(body, ctx, ctx.name_for(b->id()),
                                      b->shape(), out_shape, d);
        auto out_name = ctx.fresh_name();
        ctx.bind(out->id(), out_name);
        emit_stablehlo_binary(body, "add", out_name, matmul_name, bias_b,
                              out_shape, d);
        body << '\n';
    } else {
        ctx.bind(out->id(), matmul_name);
    }
}

// ── Shape ops ───────────────────────────────────────────────────────────────

auto handle_reshape(LoweringContext& ctx,
                    const ::tenzor::jit::Node& node,
                    std::ostream& body) -> void {
    if (node.inputs().size() != 1 || node.outputs().size() != 1) {
        throw std::runtime_error(
            "GraphToMLIR: Reshape expects 1 input, 1 output");
    }
    const auto& in_val  = node.inputs()[0];
    const auto& out_val = node.outputs()[0];
    auto out_name = ctx.fresh_name();
    ctx.bind(out_val->id(), out_name);
    emit_stablehlo_reshape(body, out_name, ctx.name_for(in_val->id()),
                           in_val->shape(), out_val->shape(),
                           out_val->dtype());
    body << '\n';
}

/// Compute a permutation `perm` such that `dst_shape[i] = src_shape[perm[i]]`.
/// When dims have unique extents this is unambiguous. For duplicates we
/// fall back to "dims" attribute (Permute) or "dim0/dim1" attributes
/// (Transpose). Callers can pass the explicit perm to skip inference.
auto infer_permutation(const std::vector<int64_t>& src_shape,
                       const std::vector<int64_t>& dst_shape)
    -> std::vector<int64_t> {
    if (src_shape.size() != dst_shape.size()) {
        throw std::runtime_error(
            "GraphToMLIR: permutation must preserve rank");
    }
    const int64_t r = static_cast<int64_t>(src_shape.size());
    std::vector<int64_t> perm(r);
    std::vector<bool> used(r, false);
    for (int64_t i = 0; i < r; ++i) {
        bool found = false;
        for (int64_t j = 0; j < r; ++j) {
            if (!used[j] && src_shape[j] == dst_shape[i]) {
                perm[i] = j;
                used[j] = true;
                found = true;
                break;
            }
        }
        if (!found) {
            throw std::runtime_error(
                "GraphToMLIR: cannot infer permutation; duplicate dims "
                "without explicit perm attr");
        }
    }
    return perm;
}

auto handle_permute(LoweringContext& ctx,
                    const ::tenzor::jit::Node& node,
                    std::ostream& body) -> void {
    if (node.inputs().size() != 1 || node.outputs().size() != 1) {
        throw std::runtime_error(
            "GraphToMLIR: Permute expects 1 input, 1 output");
    }
    const auto& in_val  = node.inputs()[0];
    const auto& out_val = node.outputs()[0];
    std::vector<int64_t> perm;
    if (node.has_attr("dims")) {
        perm = node.get_vec_attr("dims");
        const auto rank = static_cast<int64_t>(in_val->shape().size());
        for (auto& p : perm) p = normalize_dim(p, rank);
    } else {
        perm = infer_permutation(in_val->shape(), out_val->shape());
    }
    auto out_name = ctx.fresh_name();
    ctx.bind(out_val->id(), out_name);
    emit_stablehlo_transpose(body, out_name, ctx.name_for(in_val->id()),
                             perm, in_val->shape(), out_val->shape(),
                             out_val->dtype());
    body << '\n';
}

auto handle_transpose(LoweringContext& ctx,
                      const ::tenzor::jit::Node& node,
                      std::ostream& body) -> void {
    if (node.inputs().size() != 1 || node.outputs().size() != 1) {
        throw std::runtime_error(
            "GraphToMLIR: Transpose expects 1 input, 1 output");
    }
    const auto& in_val  = node.inputs()[0];
    const auto& out_val = node.outputs()[0];
    const auto rank = static_cast<int64_t>(in_val->shape().size());
    // Default 2D transpose: swap last two dims. If we have explicit dim0/
    // dim1 attrs (uncommon in traces but legal for manual graphs), use
    // them.
    int64_t d0 = rank >= 2 ? rank - 2 : 0;
    int64_t d1 = rank >= 2 ? rank - 1 : 0;
    if (node.has_attr("dim0")) d0 = normalize_dim(node.get_int_attr("dim0"), rank);
    if (node.has_attr("dim1")) d1 = normalize_dim(node.get_int_attr("dim1"), rank);
    std::vector<int64_t> perm(rank);
    for (int64_t i = 0; i < rank; ++i) perm[i] = i;
    std::swap(perm[d0], perm[d1]);

    auto out_name = ctx.fresh_name();
    ctx.bind(out_val->id(), out_name);
    emit_stablehlo_transpose(body, out_name, ctx.name_for(in_val->id()),
                             perm, in_val->shape(), out_val->shape(),
                             out_val->dtype());
    body << '\n';
}

auto handle_slice(LoweringContext& ctx,
                  const ::tenzor::jit::Node& node,
                  std::ostream& body) -> void {
    if (node.inputs().size() != 1 || node.outputs().size() != 1) {
        throw std::runtime_error(
            "GraphToMLIR: Slice expects 1 input, 1 output");
    }
    const auto& in_val  = node.inputs()[0];
    const auto& out_val = node.outputs()[0];
    const auto in_shape  = in_val->shape();
    const auto out_shape = out_val->shape();
    const int64_t rank = static_cast<int64_t>(in_shape.size());

    std::vector<int64_t> starts(rank, 0), limits = in_shape, strides(rank, 1);
    if (node.has_attr("starts") && node.has_attr("ends")) {
        auto sv = node.get_vec_attr("starts");
        auto ev = node.get_vec_attr("ends");
        if (static_cast<int64_t>(sv.size()) != rank ||
            static_cast<int64_t>(ev.size()) != rank) {
            throw std::runtime_error(
                "GraphToMLIR: Slice starts/ends must have rank entries");
        }
        for (int64_t i = 0; i < rank; ++i) {
            starts[i]  = sv[i];
            limits[i]  = ev[i];
        }
    } else if (node.has_attr("dim")) {
        // Per-dim slice form: dim + start + end (+ optional step).
        const int64_t dim = normalize_dim(node.get_int_attr("dim"), rank);
        starts[dim] = get_attr_int(node, {"start"}, 0);
        limits[dim] = get_attr_int(node, {"end"},   in_shape[dim]);
        strides[dim] = get_attr_int(node, {"step"}, 1);
    } else {
        // Fall back to inferring from output shape — start=0, stride=1.
        for (int64_t i = 0; i < rank; ++i) limits[i] = out_shape[i];
    }

    auto out_name = ctx.fresh_name();
    ctx.bind(out_val->id(), out_name);
    emit_stablehlo_slice(body, out_name, ctx.name_for(in_val->id()), starts,
                         limits, strides, in_shape, out_shape,
                         out_val->dtype());
    body << '\n';
}

auto handle_cat(LoweringContext& ctx,
                const ::tenzor::jit::Node& node,
                std::ostream& body) -> void {
    if (node.outputs().size() != 1) {
        throw std::runtime_error("GraphToMLIR: Cat expects 1 output");
    }
    const auto& out_val = node.outputs()[0];
    const auto rank = static_cast<int64_t>(out_val->shape().size());
    const int64_t dim = node.has_attr("dim")
                            ? normalize_dim(node.get_int_attr("dim"), rank)
                            : 0;
    std::vector<std::string> names;
    std::vector<std::vector<int64_t>> shapes;
    names.reserve(node.inputs().size());
    shapes.reserve(node.inputs().size());
    for (const auto& v : node.inputs()) {
        names.push_back(ctx.name_for(v->id()));
        shapes.push_back(v->shape());
    }
    auto out_name = ctx.fresh_name();
    ctx.bind(out_val->id(), out_name);
    emit_stablehlo_concatenate(body, out_name, names, shapes, dim,
                               out_val->shape(), out_val->dtype());
    body << '\n';
}

/// Stack: insert a new dim of size 1 in each input, then concatenate.
auto handle_stack(LoweringContext& ctx,
                  const ::tenzor::jit::Node& node,
                  std::ostream& body) -> void {
    if (node.outputs().size() != 1) {
        throw std::runtime_error("GraphToMLIR: Stack expects 1 output");
    }
    const auto& out_val = node.outputs()[0];
    const auto rank = static_cast<int64_t>(out_val->shape().size());
    const int64_t dim = node.has_attr("dim")
                            ? normalize_dim(node.get_int_attr("dim"), rank)
                            : 0;
    const auto d = out_val->dtype();

    std::vector<std::string> reshaped_names;
    std::vector<std::vector<int64_t>> reshaped_shapes;
    reshaped_names.reserve(node.inputs().size());
    reshaped_shapes.reserve(node.inputs().size());
    for (const auto& v : node.inputs()) {
        auto in_shape = v->shape();
        auto new_shape = in_shape;
        new_shape.insert(new_shape.begin() + dim, 1);
        auto r_name = ctx.fresh_name();
        emit_stablehlo_reshape(body, r_name, ctx.name_for(v->id()), in_shape,
                               new_shape, d);
        body << '\n';
        reshaped_names.push_back(r_name);
        reshaped_shapes.push_back(new_shape);
    }
    auto out_name = ctx.fresh_name();
    ctx.bind(out_val->id(), out_name);
    emit_stablehlo_concatenate(body, out_name, reshaped_names,
                               reshaped_shapes, dim, out_val->shape(), d);
    body << '\n';
}

auto handle_squeeze(LoweringContext& ctx,
                    const ::tenzor::jit::Node& node,
                    std::ostream& body) -> void {
    // Squeeze is shape-only; reshape from in_shape → out_shape suffices.
    handle_reshape(ctx, node, body);
}

auto handle_unsqueeze(LoweringContext& ctx,
                      const ::tenzor::jit::Node& node,
                      std::ostream& body) -> void {
    handle_reshape(ctx, node, body);
}

auto handle_flatten(LoweringContext& ctx,
                    const ::tenzor::jit::Node& node,
                    std::ostream& body) -> void {
    handle_reshape(ctx, node, body);
}

auto handle_broadcast(LoweringContext& ctx,
                      const ::tenzor::jit::Node& node,
                      std::ostream& body) -> void {
    if (node.inputs().size() != 1 || node.outputs().size() != 1) {
        throw std::runtime_error(
            "GraphToMLIR: Broadcast expects 1 input, 1 output");
    }
    const auto& in_val  = node.inputs()[0];
    const auto& out_val = node.outputs()[0];
    auto dims = right_align_bcast_dims(in_val->shape(), out_val->shape());
    auto out_name = ctx.fresh_name();
    ctx.bind(out_val->id(), out_name);
    emit_stablehlo_broadcast_in_dim(body, out_name, ctx.name_for(in_val->id()),
                                    dims, in_val->shape(), out_val->shape(),
                                    out_val->dtype());
    body << '\n';
}

// ── Vision ──────────────────────────────────────────────────────────────────

/// Conv2d: NCHW input, OIHW weight, optional bias.
auto handle_conv2d(LoweringContext& ctx,
                   const ::tenzor::jit::Node& node,
                   std::ostream& body) -> void {
    if (node.outputs().empty() || node.inputs().size() < 2) {
        throw std::runtime_error(
            "GraphToMLIR: Conv2d expects (x, w [, b]) inputs and 1 output");
    }
    const auto& x = node.inputs()[0];
    const auto& w = node.inputs()[1];
    const auto& out = node.outputs()[0];
    const auto d = out->dtype();

    // Attribute reads (defaults match torch's nn.Conv2d).
    auto get_pair = [&](const char* vec_key, const char* h_key,
                        const char* w_key, int64_t default_v) {
        if (node.has_attr(vec_key)) {
            auto v = node.get_vec_attr(vec_key);
            if (v.size() == 2) return std::pair{v[0], v[1]};
            if (v.size() == 1) return std::pair{v[0], v[0]};
        }
        int64_t h = node.has_attr(h_key)
                        ? node.get_int_attr(h_key)
                        : (node.has_attr("stride")
                               ? node.get_int_attr("stride") : default_v);
        int64_t wv = node.has_attr(w_key)
                         ? node.get_int_attr(w_key)
                         : (node.has_attr("stride")
                                ? node.get_int_attr("stride") : default_v);
        return std::pair{h, wv};
    };
    auto [stride_h, stride_w]   = get_pair("stride",   "stride_h",
                                           "stride_w", 1);
    auto [pad_h,    pad_w]      = get_pair("padding",  "padding_h",
                                           "padding_w", 0);
    auto [dilation_h, dilation_w] = get_pair("dilation", "dilation_h",
                                             "dilation_w", 1);
    const int64_t groups = node.has_attr("groups")
                               ? node.get_int_attr("groups") : 1;

    auto out_name = ctx.fresh_name();
    auto conv_name = node.inputs().size() >= 3 ? ctx.fresh_name()
                                               : out_name;

    body << '%' << conv_name << " = stablehlo.convolution(%"
         << ctx.name_for(x->id()) << ", %" << ctx.name_for(w->id()) << ")\n"
         << "    dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n"
         << "    window = {stride = [" << stride_h << ", " << stride_w
         << "], pad = [[" << pad_h << ", " << pad_h << "], ["
         << pad_w << ", " << pad_w << "]], rhs_dilate = ["
         << dilation_h << ", " << dilation_w << "]} "
         << "{batch_group_count = 1 : i64, feature_group_count = "
         << groups << " : i64} : (";
    write_tensor_type_for_emit(body, x->shape(), d);
    body << ", ";
    write_tensor_type_for_emit(body, w->shape(), d);
    body << ") -> ";
    write_tensor_type_for_emit(body, out->shape(), d);
    body << '\n';

    if (node.inputs().size() >= 3) {
        const auto& b_val = node.inputs()[2];
        // Bias: (C_out,) broadcast over (N, C_out, H_out, W_out) — dim 1.
        auto bias_b = ctx.fresh_name();
        emit_stablehlo_broadcast_in_dim(body, bias_b,
                                        ctx.name_for(b_val->id()),
                                        /*bcast_dims=*/{1}, b_val->shape(),
                                        out->shape(), d);
        body << '\n';
        emit_stablehlo_binary(body, "add", out_name, conv_name, bias_b,
                              out->shape(), d);
        body << '\n';
    }
    ctx.bind(out->id(), out_name);
}

/// Emit a stablehlo.reduce_window pooling op (MaxPool / AvgPool sum).
auto emit_reduce_window(std::ostream& body, LoweringContext& ctx,
                        const std::string& result, const std::string& operand,
                        const std::string& init_name,
                        const std::string& reducer,
                        const std::vector<int64_t>& window,
                        const std::vector<int64_t>& strides,
                        const std::vector<int64_t>& padding_low,
                        const std::vector<int64_t>& padding_high,
                        const std::vector<int64_t>& operand_shape,
                        const std::vector<int64_t>& result_shape,
                        ::tenzor::DType d) -> void {
    body << '%' << result
         << " = \"stablehlo.reduce_window\"(%" << operand << ", %"
         << init_name << ") <{window_dimensions = array<i64: ";
    for (std::size_t i = 0; i < window.size(); ++i) {
        if (i != 0) body << ", ";
        body << window[i];
    }
    body << ">, window_strides = array<i64: ";
    for (std::size_t i = 0; i < strides.size(); ++i) {
        if (i != 0) body << ", ";
        body << strides[i];
    }
    body << ">, padding = dense<[[";
    for (std::size_t i = 0; i < padding_low.size(); ++i) {
        if (i != 0) body << "], [";
        body << padding_low[i] << ", " << padding_high[i];
    }
    body << "]]> : tensor<" << padding_low.size() << "x2xi64>}> ({"
         << "\n  ^bb0(%a: tensor<" << mlir_type_name(d) << ">, %b: tensor<"
         << mlir_type_name(d) << ">):"
         << "\n    %r = stablehlo." << reducer << " %a, %b : tensor<"
         << mlir_type_name(d) << ">"
         << "\n    stablehlo.return %r : tensor<" << mlir_type_name(d) << ">"
         << "\n  }) : (";
    write_tensor_type_for_emit(body, operand_shape, d);
    body << ", ";
    write_tensor_type_for_emit(body, {}, d);
    body << ") -> ";
    write_tensor_type_for_emit(body, result_shape, d);
}

/// Build pool attributes (window, strides, padding) for 2D pooling on
/// NCHW: window/stride/padding apply only to dims 2,3.
auto pool_window_for_2d(const ::tenzor::jit::Node& node,
                        const std::vector<int64_t>& x_shape)
    -> std::tuple<std::vector<int64_t>, std::vector<int64_t>,
                  std::vector<int64_t>, std::vector<int64_t>> {
    auto get_hw = [&](const char* vec_key, const char* h_key,
                      const char* w_key, int64_t fallback) {
        if (node.has_attr(vec_key)) {
            auto v = node.get_vec_attr(vec_key);
            if (v.size() == 2) return std::pair{v[0], v[1]};
            if (v.size() == 1) return std::pair{v[0], v[0]};
        }
        int64_t h = node.has_attr(h_key) ? node.get_int_attr(h_key)
                                         : fallback;
        int64_t w = node.has_attr(w_key) ? node.get_int_attr(w_key)
                                         : fallback;
        return std::pair{h, w};
    };
    auto [kh, kw] = get_hw("kernel_size", "kernel_h", "kernel_w", 1);
    auto [sh, sw] = get_hw("stride",      "stride_h", "stride_w",
                           /*fallback to kernel*/ kh);
    auto [ph, pw] = get_hw("padding",     "padding_h", "padding_w", 0);
    std::vector<int64_t> window  = {1, 1, kh, kw};
    std::vector<int64_t> strides = {1, 1, sh, sw};
    std::vector<int64_t> pad_lo  = {0, 0, ph, pw};
    std::vector<int64_t> pad_hi  = {0, 0, ph, pw};
    (void)x_shape;
    return {window, strides, pad_lo, pad_hi};
}

auto handle_max_pool2d(LoweringContext& ctx,
                       const ::tenzor::jit::Node& node,
                       std::ostream& body) -> void {
    if (node.inputs().size() != 1 || node.outputs().empty()) {
        throw std::runtime_error(
            "GraphToMLIR: MaxPool2d expects 1 input and 1+ outputs");
    }
    const auto& x   = node.inputs()[0];
    const auto& out = node.outputs()[0];
    const auto d = out->dtype();
    auto [win, str, plo, phi] = pool_window_for_2d(node, x->shape());

    std::string init_max = "0xFF800000";
    if (d == ::tenzor::DType::Float64) init_max = "0xFFF0000000000000";
    auto init_name = ctx.fresh_name();
    emit_stablehlo_splat_constant(body, init_name, init_max, {}, d);
    body << '\n';

    auto out_name = ctx.fresh_name();
    ctx.bind(out->id(), out_name);
    emit_reduce_window(body, ctx, out_name, ctx.name_for(x->id()), init_name,
                       "maximum", win, str, plo, phi, x->shape(),
                       out->shape(), d);
    body << '\n';
}

auto handle_avg_pool2d(LoweringContext& ctx,
                       const ::tenzor::jit::Node& node,
                       std::ostream& body) -> void {
    if (node.inputs().size() != 1 || node.outputs().empty()) {
        throw std::runtime_error(
            "GraphToMLIR: AvgPool2d expects 1 input and 1+ outputs");
    }
    const auto& x   = node.inputs()[0];
    const auto& out = node.outputs()[0];
    const auto d = out->dtype();
    auto [win, str, plo, phi] = pool_window_for_2d(node, x->shape());

    auto init_name = ctx.fresh_name();
    emit_stablehlo_splat_constant(body, init_name, scalar_literal(0.0, d), {},
                                  d);
    body << '\n';

    auto sum_name = ctx.fresh_name();
    emit_reduce_window(body, ctx, sum_name, ctx.name_for(x->id()), init_name,
                       "add", win, str, plo, phi, x->shape(), out->shape(),
                       d);
    body << '\n';

    // Divide by window area kH * kW.
    const int64_t area = win[2] * win[3];
    auto area_const = ctx.fresh_name();
    emit_stablehlo_splat_constant(
        body, area_const, scalar_literal(static_cast<double>(area), d),
        out->shape(), d);
    body << '\n';
    auto out_name = ctx.fresh_name();
    ctx.bind(out->id(), out_name);
    emit_stablehlo_binary(body, "divide", out_name, sum_name, area_const,
                          out->shape(), d);
    body << '\n';
}

/// AdaptiveAvgPool2d: compute kernel/stride to bring (H_in, W_in) →
/// (H_out, W_out). Uses the standard formula stride = floor(H_in/H_out),
/// kernel = H_in - (H_out - 1) * stride.
auto handle_adaptive_avg_pool2d(LoweringContext& ctx,
                                const ::tenzor::jit::Node& node,
                                std::ostream& body) -> void {
    if (node.inputs().size() != 1 || node.outputs().empty()) {
        throw std::runtime_error(
            "GraphToMLIR: AdaptiveAvgPool2d expects 1 input, 1+ outputs");
    }
    const auto& x_val = node.inputs()[0];
    const auto& out_val = node.outputs()[0];
    const auto& xs = x_val->shape();
    const auto& os = out_val->shape();
    if (xs.size() != 4 || os.size() != 4) {
        throw std::runtime_error(
            "GraphToMLIR: AdaptiveAvgPool2d requires rank-4 input/output");
    }
    const auto d = out_val->dtype();
    const int64_t H_in = xs[2], W_in = xs[3];
    const int64_t H_out = os[2], W_out = os[3];
    const int64_t sh = H_out > 0 ? H_in / H_out : H_in;
    const int64_t sw = W_out > 0 ? W_in / W_out : W_in;
    const int64_t kh = H_in - (H_out - 1) * sh;
    const int64_t kw = W_in - (W_out - 1) * sw;

    std::vector<int64_t> win = {1, 1, kh, kw};
    std::vector<int64_t> str = {1, 1, sh, sw};
    std::vector<int64_t> plo = {0, 0, 0, 0};
    std::vector<int64_t> phi = {0, 0, 0, 0};

    auto init_name = ctx.fresh_name();
    emit_stablehlo_splat_constant(body, init_name, scalar_literal(0.0, d), {},
                                  d);
    body << '\n';
    auto sum_name = ctx.fresh_name();
    emit_reduce_window(body, ctx, sum_name, ctx.name_for(x_val->id()),
                       init_name, "add", win, str, plo, phi, xs, os, d);
    body << '\n';
    const int64_t area = kh * kw;
    auto area_const = ctx.fresh_name();
    emit_stablehlo_splat_constant(
        body, area_const, scalar_literal(static_cast<double>(area), d), os, d);
    body << '\n';
    auto out_name = ctx.fresh_name();
    ctx.bind(out_val->id(), out_name);
    emit_stablehlo_binary(body, "divide", out_name, sum_name, area_const, os,
                          d);
    body << '\n';
}

/// Dropout: in inference (training=false) it's the identity. In training
/// we'd need an RNG + threshold; for the JIT path we treat it as
/// inference-only since the JIT compile happens at evaluation time and
/// trained-mode dropout requires deterministic seeding to be useful.
auto handle_dropout(LoweringContext& ctx,
                    const ::tenzor::jit::Node& node,
                    std::ostream& /*body*/) -> void {
    if (node.inputs().empty() || node.outputs().empty()) {
        throw std::runtime_error(
            "GraphToMLIR: Dropout expects 1+ input, 1+ output");
    }
    // Identity: rebind output to input's SSA name.
    ctx.bind(node.outputs()[0]->id(),
             ctx.name_for(node.inputs()[0]->id()));
}

/// Padding: stablehlo.pad with given low/high (no interior padding).
auto handle_padding(LoweringContext& ctx,
                    const ::tenzor::jit::Node& node,
                    std::ostream& body) -> void {
    if (node.inputs().size() != 1 || node.outputs().empty()) {
        throw std::runtime_error(
            "GraphToMLIR: Padding expects 1 input, 1+ output");
    }
    const auto& x_val = node.inputs()[0];
    const auto& out_val = node.outputs()[0];
    const auto& xs = x_val->shape();
    const auto& os = out_val->shape();
    const auto d = out_val->dtype();
    if (xs.size() != os.size()) {
        throw std::runtime_error(
            "GraphToMLIR: Padding must preserve rank");
    }

    // Pull padding pairs from "padding" vec (pytorch order: last-dim
    // first, [w_lo, w_hi, h_lo, h_hi, ...]). Fall back to inferring
    // symmetric padding from shape difference.
    std::vector<int64_t> low(xs.size(), 0), high(xs.size(), 0);
    if (node.has_attr("padding")) {
        auto v = node.get_vec_attr("padding");
        // pytorch's F.pad uses last-dim-first pairs; ONNX/StableHLO use
        // first-dim-first single-pair-per-dim. Accept either; if length
        // is 2 * rank, assume first-dim-first.
        if (static_cast<int64_t>(v.size()) == 2 * static_cast<int64_t>(xs.size())) {
            for (std::size_t i = 0; i < xs.size(); ++i) {
                low[i]  = v[2 * i];
                high[i] = v[2 * i + 1];
            }
        } else if (v.size() % 2 == 0) {
            // pytorch order: pad innermost dims.
            const std::size_t pairs = v.size() / 2;
            for (std::size_t i = 0; i < pairs; ++i) {
                const std::size_t dim = xs.size() - 1 - i;
                low[dim]  = v[2 * i];
                high[dim] = v[2 * i + 1];
            }
        }
    } else {
        for (std::size_t i = 0; i < xs.size(); ++i) {
            const int64_t diff = os[i] - xs[i];
            low[i]  = diff / 2;
            high[i] = diff - low[i];
        }
    }
    std::vector<int64_t> interior(xs.size(), 0);

    auto padval = ctx.fresh_name();
    emit_stablehlo_splat_constant(body, padval, scalar_literal(0.0, d), {}, d);
    body << '\n';
    auto out_name = ctx.fresh_name();
    ctx.bind(out_val->id(), out_name);
    emit_stablehlo_pad(body, out_name, ctx.name_for(x_val->id()), padval,
                       low, high, interior, xs, os, d);
    body << '\n';
}

/// Interpolate: nearest-neighbor and bilinear upsampling. Without
/// dynamic-shape support we materialize index gather tensors at compile
/// time. For MVP-1 we implement nearest-neighbor by gather (an exact
/// reference impl) and leave bilinear as a more complex follow-up.
/// Falls back to a broadcast when the input already matches the output
/// shape (identity).
auto handle_interpolate(LoweringContext& ctx,
                        const ::tenzor::jit::Node& node,
                        std::ostream& body) -> void {
    if (node.inputs().size() != 1 || node.outputs().empty()) {
        throw std::runtime_error(
            "GraphToMLIR: Interpolate expects 1 input, 1+ output");
    }
    const auto& x_val   = node.inputs()[0];
    const auto& out_val = node.outputs()[0];
    const auto xs = x_val->shape();
    const auto os = out_val->shape();
    const auto d = out_val->dtype();

    if (xs == os) {
        // Identity passthrough — also handles rank mismatch errors via
        // the comparison being false.
        ctx.bind(out_val->id(), ctx.name_for(x_val->id()));
        return;
    }

    // Nearest-neighbor 2D over NCHW. We materialize a (H_out, W_out)
    // index pair via stablehlo.iota + arithmetic, then gather. For
    // simplicity in the text emitter we emit one gather over the H
    // axis then another over the W axis. For now, we implement a single
    // gather-by-iota along H; W is handled by repeating logic.
    //
    // This is a reference implementation that produces correct output
    // for integer-ratio upsampling. For mixed-ratio cases the IREE
    // compiler will still accept the IR; the result matches
    // torch's nearest-neighbor when the same rounding rule is used.
    //
    // To keep the IR small for MVP-1 we just emit a stablehlo.reshape
    // followed by stablehlo.broadcast_in_dim when the upscale factor is
    // an integer along each spatial dim. Otherwise we error.
    if (xs.size() != 4 || os.size() != 4) {
        throw std::runtime_error(
            "GraphToMLIR: Interpolate only implemented for rank-4 NCHW");
    }
    const int64_t H_in = xs[2], W_in = xs[3];
    const int64_t H_out = os[2], W_out = os[3];
    if (H_out % H_in != 0 || W_out % W_in != 0) {
        throw std::runtime_error(
            "GraphToMLIR: Interpolate non-integer upsample not supported "
            "in MVP-1");
    }
    const int64_t fh = H_out / H_in;
    const int64_t fw = W_out / W_in;

    // Reshape (N, C, H_in, W_in) -> (N, C, H_in, 1, W_in, 1)
    std::vector<int64_t> exp_shape =
        {xs[0], xs[1], xs[2], 1, xs[3], 1};
    auto exp_name = ctx.fresh_name();
    emit_stablehlo_reshape(body, exp_name, ctx.name_for(x_val->id()), xs,
                           exp_shape, d);
    body << '\n';
    // Broadcast to (N, C, H_in, fh, W_in, fw)
    std::vector<int64_t> bcast_shape =
        {xs[0], xs[1], xs[2], fh, xs[3], fw};
    auto bcast_name = ctx.fresh_name();
    std::vector<int64_t> bcast_dims = {0, 1, 2, 3, 4, 5};
    emit_stablehlo_broadcast_in_dim(body, bcast_name, exp_name, bcast_dims,
                                    exp_shape, bcast_shape, d);
    body << '\n';
    // Reshape back to (N, C, H_out, W_out)
    auto out_name = ctx.fresh_name();
    ctx.bind(out_val->id(), out_name);
    emit_stablehlo_reshape(body, out_name, bcast_name, bcast_shape, os, d);
    body << '\n';
}

// ── Norms ───────────────────────────────────────────────────────────────────

/// LayerNorm along the last `len(normalized_shape)` dims (StableHLO has
/// no direct layer-norm op so we decompose).
/// Inputs:
///   x       : tensor (rank ≥ rank of normalized_shape)
///   weight? : optional gamma — same shape as the trailing normalized
///             dims (rank-N where N = normalized_shape rank)
///   bias?   : optional beta — same shape as weight
auto handle_layer_norm(LoweringContext& ctx,
                       const ::tenzor::jit::Node& node,
                       std::ostream& body) -> void {
    if (node.outputs().empty()) {
        throw std::runtime_error("GraphToMLIR: LayerNorm expects 1+ outputs");
    }
    if (node.inputs().empty()) {
        throw std::runtime_error("GraphToMLIR: LayerNorm expects 1+ inputs");
    }
    const auto& x_val = node.inputs()[0];
    const auto& out_val = node.outputs()[0];
    const auto shape = x_val->shape();
    const auto d = out_val->dtype();
    const float eps = get_attr_float(node, {"eps"}, 1e-5f);

    // The normalized dims are the trailing dims given by attr
    // "normalized_shape" (or, if absent, the last dim only).
    std::vector<int64_t> norm_dims;
    if (node.has_attr("normalized_shape")) {
        auto ns = node.get_vec_attr("normalized_shape");
        // norm_dims = last len(ns) dims of x.
        const int64_t rank = static_cast<int64_t>(shape.size());
        const int64_t off = rank - static_cast<int64_t>(ns.size());
        for (int64_t i = off; i < rank; ++i) norm_dims.push_back(i);
    } else {
        norm_dims.push_back(static_cast<int64_t>(shape.size()) - 1);
    }
    // Shape after reducing across norm_dims (no-keepdim) and after re-
    // broadcasting back to x_shape.
    std::vector<int64_t> reduced_shape;
    {
        std::unordered_set<int64_t> drop(norm_dims.begin(), norm_dims.end());
        for (std::size_t i = 0; i < shape.size(); ++i) {
            if (!drop.count(static_cast<int64_t>(i))) {
                reduced_shape.push_back(shape[i]);
            }
        }
    }
    std::vector<int64_t> bcast_dims;
    for (int64_t i = 0; i < static_cast<int64_t>(shape.size()); ++i) {
        if (std::find(norm_dims.begin(), norm_dims.end(), i) ==
            norm_dims.end()) {
            bcast_dims.push_back(i);
        }
    }

    // N = product of normalized dim extents.
    int64_t N = 1;
    for (auto k : norm_dims) N *= shape[k];

    const auto& x = ctx.name_for(x_val->id());

    // sum_x = reduce_sum(x, norm_dims)
    auto init0 = ctx.fresh_name();
    emit_stablehlo_splat_constant(body, init0, scalar_literal(0.0, d), {}, d);
    body << '\n';
    auto sum_x = ctx.fresh_name();
    emit_stablehlo_reduce(body, sum_x, x, init0, "add", norm_dims, shape,
                          reduced_shape, d);
    body << '\n';
    // n_const = N
    auto n_const = ctx.fresh_name();
    emit_stablehlo_splat_constant(body, n_const,
                                  scalar_literal(static_cast<double>(N), d),
                                  reduced_shape, d);
    body << '\n';
    auto mean_r = ctx.fresh_name();
    emit_stablehlo_binary(body, "divide", mean_r, sum_x, n_const,
                          reduced_shape, d);
    body << '\n';
    auto mean_b = ctx.fresh_name();
    emit_stablehlo_broadcast_in_dim(body, mean_b, mean_r, bcast_dims,
                                    reduced_shape, shape, d);
    body << '\n';
    // centered = x - mean
    auto centered = ctx.fresh_name();
    emit_stablehlo_binary(body, "subtract", centered, x, mean_b, shape, d);
    body << '\n';
    // sq = centered * centered
    auto sq = ctx.fresh_name();
    emit_stablehlo_binary(body, "multiply", sq, centered, centered, shape, d);
    body << '\n';
    // sum_sq
    auto init1 = ctx.fresh_name();
    emit_stablehlo_splat_constant(body, init1, scalar_literal(0.0, d), {}, d);
    body << '\n';
    auto sum_sq = ctx.fresh_name();
    emit_stablehlo_reduce(body, sum_sq, sq, init1, "add", norm_dims, shape,
                          reduced_shape, d);
    body << '\n';
    auto var_r = ctx.fresh_name();
    emit_stablehlo_binary(body, "divide", var_r, sum_sq, n_const,
                          reduced_shape, d);
    body << '\n';
    // var + eps
    auto eps_c = ctx.fresh_name();
    emit_stablehlo_splat_constant(body, eps_c,
                                  scalar_literal(static_cast<double>(eps), d),
                                  reduced_shape, d);
    body << '\n';
    auto var_eps = ctx.fresh_name();
    emit_stablehlo_binary(body, "add", var_eps, var_r, eps_c, reduced_shape, d);
    body << '\n';
    auto std_r = ctx.fresh_name();
    emit_stablehlo_unary(body, "sqrt", std_r, var_eps, reduced_shape, d);
    body << '\n';
    auto std_b = ctx.fresh_name();
    emit_stablehlo_broadcast_in_dim(body, std_b, std_r, bcast_dims,
                                    reduced_shape, shape, d);
    body << '\n';
    auto x_hat = ctx.fresh_name();
    emit_stablehlo_binary(body, "divide", x_hat, centered, std_b, shape, d);
    body << '\n';

    // Apply weight/bias if present (inputs[1], inputs[2]).
    std::string final_name = x_hat;
    if (node.inputs().size() >= 2) {
        const auto& w_val = node.inputs()[1];
        auto w_b = maybe_broadcast(body, ctx, ctx.name_for(w_val->id()),
                                   w_val->shape(), shape, d);
        auto scaled = ctx.fresh_name();
        emit_stablehlo_binary(body, "multiply", scaled, x_hat, w_b, shape, d);
        body << '\n';
        final_name = scaled;
    }
    if (node.inputs().size() >= 3) {
        const auto& b_val = node.inputs()[2];
        auto b_b = maybe_broadcast(body, ctx, ctx.name_for(b_val->id()),
                                   b_val->shape(), shape, d);
        auto shifted = ctx.fresh_name();
        emit_stablehlo_binary(body, "add", shifted, final_name, b_b, shape, d);
        body << '\n';
        final_name = shifted;
    }

    // Bind the primary output (out_val->id()) to the final tensor.
    ctx.bind(out_val->id(), final_name);
    // If the node has additional outputs (mean, rstd from FusedLayerNorm
    // contract), we don't currently emit them — those output values
    // become orphaned but won't break the graph since they're not in
    // g.outputs(). If they ARE in g.outputs() the lowering will error
    // out when looking up their SSA name.
}

/// BatchNorm2d inference: y = (x - mean) / sqrt(var + eps) * weight + bias
/// Inputs: (x, weight, bias, running_mean, running_var)
auto handle_batch_norm2d(LoweringContext& ctx,
                         const ::tenzor::jit::Node& node,
                         std::ostream& body) -> void {
    if (node.inputs().size() < 5 || node.outputs().empty()) {
        throw std::runtime_error(
            "GraphToMLIR: BatchNorm2d expects 5 inputs and 1+ outputs");
    }
    const auto& x_val   = node.inputs()[0];
    const auto& w_val   = node.inputs()[1];
    const auto& b_val   = node.inputs()[2];
    const auto& m_val   = node.inputs()[3];
    const auto& v_val   = node.inputs()[4];
    const auto& out_val = node.outputs()[0];
    const auto shape = x_val->shape();
    const auto d = out_val->dtype();
    const float eps = get_attr_float(node, {"eps"}, 1e-5f);

    // Use the explicit stablehlo.batch_norm_inference op.
    auto out_name = ctx.fresh_name();
    ctx.bind(out_val->id(), out_name);
    body << '%' << out_name << " = \"stablehlo.batch_norm_inference\"(%"
         << ctx.name_for(x_val->id()) << ", %"
         << ctx.name_for(w_val->id()) << ", %"
         << ctx.name_for(b_val->id()) << ", %"
         << ctx.name_for(m_val->id()) << ", %"
         << ctx.name_for(v_val->id()) << ") <{"
         << "epsilon = " << std::scientific << std::setprecision(9) << eps
         << " : f32, feature_index = 1 : i64}> : (";
    write_tensor_type_for_emit(body, shape, d);
    body << ", ";
    write_tensor_type_for_emit(body, w_val->shape(), d);
    body << ", ";
    write_tensor_type_for_emit(body, b_val->shape(), d);
    body << ", ";
    write_tensor_type_for_emit(body, m_val->shape(), d);
    body << ", ";
    write_tensor_type_for_emit(body, v_val->shape(), d);
    body << ") -> ";
    write_tensor_type_for_emit(body, shape, d);
    body << '\n';
}

// ── Cast / Index ────────────────────────────────────────────────────────────

auto handle_cast(LoweringContext& ctx,
                 const ::tenzor::jit::Node& node,
                 std::ostream& body) -> void {
    if (node.inputs().size() != 1 || node.outputs().size() != 1) {
        throw std::runtime_error(
            "GraphToMLIR: Cast expects 1 input, 1 output");
    }
    const auto& in_val  = node.inputs()[0];
    const auto& out_val = node.outputs()[0];
    auto out_name = ctx.fresh_name();
    ctx.bind(out_val->id(), out_name);
    emit_stablehlo_convert(body, out_name, ctx.name_for(in_val->id()),
                           in_val->shape(), in_val->dtype(),
                           out_val->dtype());
    body << '\n';
}

/// Embedding: `out[i, ...] = weight[indices[i], ...]`.
///   weight: (V, E)        — operand 0
///   indices: (..., L)      — operand 1, integer
///   out:    (..., L, E)
/// Lower to stablehlo.gather. Per the StableHLO `gather` spec, with
/// `start_indices` = indices_with_one_extra_trailing_size_1_dim and
/// gather index_dim = -1, the result is gathered along the embedding's
/// first axis.
auto handle_embedding(LoweringContext& ctx,
                      const ::tenzor::jit::Node& node,
                      std::ostream& body) -> void {
    if (node.inputs().size() != 2 || node.outputs().size() != 1) {
        throw std::runtime_error(
            "GraphToMLIR: Embedding expects 2 inputs (weight, indices) and "
            "1 output");
    }
    const auto& w_val  = node.inputs()[0];
    const auto& i_val  = node.inputs()[1];
    const auto& out_val = node.outputs()[0];
    const auto w_shape = w_val->shape();
    const auto i_shape = i_val->shape();
    const auto out_shape = out_val->shape();
    const auto d = out_val->dtype();

    // Use stablehlo.gather. Operand: weight (V, E). Start indices:
    // indices reshaped to (i_shape..., 1) — adding a trailing size-1 dim
    // so the index vector index has length 1 (we gather along weight's
    // dim 0).
    auto i_reshaped_shape = i_shape;
    i_reshaped_shape.push_back(1);
    auto idx_name = ctx.fresh_name();
    emit_stablehlo_reshape(body, idx_name, ctx.name_for(i_val->id()),
                           i_shape, i_reshaped_shape, i_val->dtype());
    body << '\n';

    auto out_name = ctx.fresh_name();
    ctx.bind(out_val->id(), out_name);

    // Emit gather: offset_dims = [rank(indices)..rank(out)-1],
    // collapsed_slice_dims = [0], start_index_map = [0],
    // index_vector_dim = rank(indices), slice_sizes = [1, E].
    const int64_t ir = static_cast<int64_t>(i_shape.size());
    const int64_t er = static_cast<int64_t>(w_shape.size()) - 1;  // E dims
    body << '%' << out_name << " = \"stablehlo.gather\"(%"
         << ctx.name_for(w_val->id()) << ", %" << idx_name << ") <{"
         << "dimension_numbers = #stablehlo.gather<offset_dims = [";
    for (int64_t k = 0; k < er; ++k) {
        if (k != 0) body << ", ";
        body << (ir + k);
    }
    body << "], collapsed_slice_dims = [0], "
         << "start_index_map = [0], index_vector_dim = " << ir << ">, "
         << "slice_sizes = array<i64: 1";
    for (int64_t k = 1; k < static_cast<int64_t>(w_shape.size()); ++k) {
        body << ", " << w_shape[k];
    }
    body << ">, indices_are_sorted = false}> : (";
    write_tensor_type_for_emit(body, w_shape, w_val->dtype());
    body << ", ";
    write_tensor_type_for_emit(body, i_reshaped_shape, i_val->dtype());
    body << ") -> ";
    write_tensor_type_for_emit(body, out_shape, d);
    body << '\n';
}

auto handle_index_select(LoweringContext& ctx,
                         const ::tenzor::jit::Node& node,
                         std::ostream& body) -> void {
    if (node.inputs().size() != 2 || node.outputs().size() != 1) {
        throw std::runtime_error(
            "GraphToMLIR: IndexSelect expects 2 inputs (input, index) and "
            "1 output");
    }
    const auto& x_val   = node.inputs()[0];
    const auto& i_val   = node.inputs()[1];
    const auto& out_val = node.outputs()[0];
    const auto x_shape = x_val->shape();
    const auto i_shape = i_val->shape();
    const auto out_shape = out_val->shape();
    const auto d = out_val->dtype();
    const int64_t rank = static_cast<int64_t>(x_shape.size());
    const int64_t dim = node.has_attr("dim")
                            ? normalize_dim(node.get_int_attr("dim"), rank)
                            : 0;

    // index is 1-D of length L. Reshape to (L, 1) so the gather operand
    // form (start_indices, index_vector_dim) is well-formed.
    auto i_reshaped_shape = i_shape;
    i_reshaped_shape.push_back(1);
    auto idx_name = ctx.fresh_name();
    emit_stablehlo_reshape(body, idx_name, ctx.name_for(i_val->id()),
                           i_shape, i_reshaped_shape, i_val->dtype());
    body << '\n';

    auto out_name = ctx.fresh_name();
    ctx.bind(out_val->id(), out_name);

    // slice_sizes: same as x_shape but with dim `dim` set to 1.
    std::vector<int64_t> slice_sizes(x_shape);
    slice_sizes[dim] = 1;
    // offset_dims: every dim of out EXCEPT the gather index dims. With
    // index of rank 1 and result shape inserting the index dim at `dim`,
    // offset_dims = all dims of out except dim.
    std::vector<int64_t> offset_dims;
    for (int64_t i = 0; i < static_cast<int64_t>(out_shape.size()); ++i) {
        if (i != dim) offset_dims.push_back(i);
    }

    body << '%' << out_name << " = \"stablehlo.gather\"(%"
         << ctx.name_for(x_val->id()) << ", %" << idx_name << ") <{"
         << "dimension_numbers = #stablehlo.gather<offset_dims = [";
    for (std::size_t k = 0; k < offset_dims.size(); ++k) {
        if (k != 0) body << ", ";
        body << offset_dims[k];
    }
    body << "], collapsed_slice_dims = [" << dim << "], "
         << "start_index_map = [" << dim << "], index_vector_dim = "
         << static_cast<int64_t>(i_shape.size()) << ">, "
         << "slice_sizes = array<i64: ";
    for (std::size_t k = 0; k < slice_sizes.size(); ++k) {
        if (k != 0) body << ", ";
        body << slice_sizes[k];
    }
    body << ">, indices_are_sorted = false}> : (";
    write_tensor_type_for_emit(body, x_shape, d);
    body << ", ";
    write_tensor_type_for_emit(body, i_reshaped_shape, i_val->dtype());
    body << ") -> ";
    write_tensor_type_for_emit(body, out_shape, d);
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


// ── Tenzor dialect ops: custom_call lowering (Group D) ──────────────────────
//
// Each of the 4 dialect ops (FlashAttention, GQA, RoPE, RMSNorm) lowers to a
// single `stablehlo.custom_call @tenzor_<name>(...)` whose backend_config
// string carries the scalar attributes. The Tenzor IREE plugin in
// `src/jit/mlir/iree_customcalls.cpp` resolves these by name back to the
// existing OpId kernel. For deploy targets without the plugin, the
// expand-to-stablehlo path replaces them with pure StableHLO primitives
// (handled in handle_*_expand below, selected via GraphToMLIR::plugin_enabled).

/// FlashAttention dialect op. Inputs: (Q, K, V). Attrs: `causal` (bool,
/// default false), `scale` (float, default 0.0 meaning the kernel picks
/// 1/sqrt(d_head)). Output shape matches Q.
auto handle_flash_attention_custom_call(LoweringContext& ctx,
                                        const ::tenzor::jit::Node& node,
                                        std::ostream& body) -> void {
    if (node.inputs().size() != 3 || node.outputs().empty()) {
        throw std::runtime_error(
            "GraphToMLIR: FlashAttention expects 3 inputs (Q,K,V) and 1+ "
            "outputs");
    }
    const auto& q_val   = node.inputs()[0];
    const auto& k_val   = node.inputs()[1];
    const auto& v_val   = node.inputs()[2];
    const auto& out_val = node.outputs()[0];

    const bool  causal = get_attr_bool (node, {"causal"},      false);
    const float scale  = get_attr_float(node, {"scale"},       0.0f);

    std::ostringstream cfg;
    cfg << "causal=" << (causal ? "true" : "false")
        << ",scale=" << std::setprecision(9) << scale;

    auto out_name = ctx.fresh_name();
    ctx.bind(out_val->id(), out_name);
    emit_custom_call(body, "tenzor_flash_attention", out_name,
                     {ctx.name_for(q_val->id()),
                      ctx.name_for(k_val->id()),
                      ctx.name_for(v_val->id())},
                     {q_val->shape(), k_val->shape(), v_val->shape()},
                     {q_val->dtype(), k_val->dtype(), v_val->dtype()},
                     out_val->shape(), out_val->dtype(), cfg.str());
    body << '\n';
}


/// Grouped-Query Attention (GQA). Inputs: (Q, K, V) where K/V may have
/// fewer heads than Q (the kernel broadcasts KV-heads to Q-heads). Attrs:
/// `causal` (bool), `scale` (float). Output shape matches Q.
auto handle_gqa_custom_call(LoweringContext& ctx,
                            const ::tenzor::jit::Node& node,
                            std::ostream& body) -> void {
    if (node.inputs().size() != 3 || node.outputs().empty()) {
        throw std::runtime_error(
            "GraphToMLIR: GQA expects 3 inputs (Q,K,V) and 1+ outputs");
    }
    const auto& q_val   = node.inputs()[0];
    const auto& k_val   = node.inputs()[1];
    const auto& v_val   = node.inputs()[2];
    const auto& out_val = node.outputs()[0];

    const bool  causal = get_attr_bool (node, {"causal"}, false);
    const float scale  = get_attr_float(node, {"scale"},  0.0f);

    std::ostringstream cfg;
    cfg << "causal=" << (causal ? "true" : "false")
        << ",scale=" << std::setprecision(9) << scale;

    auto out_name = ctx.fresh_name();
    ctx.bind(out_val->id(), out_name);
    emit_custom_call(body, "tenzor_gqa", out_name,
                     {ctx.name_for(q_val->id()),
                      ctx.name_for(k_val->id()),
                      ctx.name_for(v_val->id())},
                     {q_val->shape(), k_val->shape(), v_val->shape()},
                     {q_val->dtype(), k_val->dtype(), v_val->dtype()},
                     out_val->shape(), out_val->dtype(), cfg.str());
    body << '\n';
}


/// RoPE (Rotary Position Embedding). Inputs: (x, cos_table, sin_table)
/// where the tables provide precomputed cos/sin per position. Attrs:
/// `offset` (int, default 0) — starting sequence offset for KV-cache
/// resumption. Output shape matches x.
auto handle_rope_apply_custom_call(LoweringContext& ctx,
                                   const ::tenzor::jit::Node& node,
                                   std::ostream& body) -> void {
    if (node.inputs().size() != 3 || node.outputs().empty()) {
        throw std::runtime_error(
            "GraphToMLIR: RoPE expects 3 inputs (x, cos, sin) and 1+ "
            "outputs");
    }
    const auto& x_val   = node.inputs()[0];
    const auto& cos_val = node.inputs()[1];
    const auto& sin_val = node.inputs()[2];
    const auto& out_val = node.outputs()[0];

    int64_t offset = 0;
    if (node.has_attr("offset")) offset = node.get_int_attr("offset");

    std::ostringstream cfg;
    cfg << "offset=" << offset;

    auto out_name = ctx.fresh_name();
    ctx.bind(out_val->id(), out_name);
    emit_custom_call(body, "tenzor_rope_apply", out_name,
                     {ctx.name_for(x_val->id()),
                      ctx.name_for(cos_val->id()),
                      ctx.name_for(sin_val->id())},
                     {x_val->shape(), cos_val->shape(), sin_val->shape()},
                     {x_val->dtype(), cos_val->dtype(), sin_val->dtype()},
                     out_val->shape(), out_val->dtype(), cfg.str());
    body << '\n';
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

            // ── Reductions ──
            case OpType::Sum:          handle_sum(ctx, *node, body);     break;
            case OpType::Mean:         handle_mean(ctx, *node, body);    break;
            case OpType::Max:          handle_max(ctx, *node, body);     break;
            case OpType::Softmax:      handle_softmax(ctx, *node, body); break;

            // ── Linalg ──
            case OpType::MatMul:       handle_matmul(ctx, *node, body); break;
            case OpType::Bmm:          handle_bmm(ctx, *node, body);    break;
            case OpType::Linear:       handle_linear(ctx, *node, body); break;

            // ── Shape ──
            case OpType::Reshape:      handle_reshape(ctx, *node, body);   break;
            case OpType::Permute:      handle_permute(ctx, *node, body);   break;
            case OpType::Transpose:    handle_transpose(ctx, *node, body); break;
            case OpType::Slice:        handle_slice(ctx, *node, body);     break;
            case OpType::Cat:          handle_cat(ctx, *node, body);       break;
            case OpType::Stack:        handle_stack(ctx, *node, body);     break;
            case OpType::Squeeze:      handle_squeeze(ctx, *node, body);   break;
            case OpType::Unsqueeze:    handle_unsqueeze(ctx, *node, body); break;
            case OpType::Flatten:      handle_flatten(ctx, *node, body);   break;
            case OpType::Broadcast:    handle_broadcast(ctx, *node, body); break;

            // ── Cast / Index ──
            case OpType::Cast:         handle_cast(ctx, *node, body);         break;
            case OpType::Embedding:    handle_embedding(ctx, *node, body);    break;
            case OpType::IndexSelect:  handle_index_select(ctx, *node, body); break;

            // ── Norms ──
            case OpType::LayerNorm:    handle_layer_norm(ctx, *node, body);   break;
            case OpType::BatchNorm2d:  handle_batch_norm2d(ctx, *node, body); break;

            // ── Vision ──
            case OpType::Conv2d:       handle_conv2d(ctx, *node, body);              break;
            case OpType::MaxPool2d:    handle_max_pool2d(ctx, *node, body);          break;
            case OpType::AvgPool2d:    handle_avg_pool2d(ctx, *node, body);          break;
            case OpType::AdaptiveAvgPool2d:
                                       handle_adaptive_avg_pool2d(ctx, *node, body); break;
            case OpType::Dropout:      handle_dropout(ctx, *node, body);             break;
            case OpType::Padding:      handle_padding(ctx, *node, body);             break;
            case OpType::Interpolate:  handle_interpolate(ctx, *node, body);         break;
            // ResidualAdd is already in the binary elementwise group.

            // ── Tenzor dialect ops (Group D) ──
            case OpType::FlashAttention:
                handle_flash_attention_custom_call(ctx, *node, body); break;
            case OpType::GQA:
                handle_gqa_custom_call(ctx, *node, body); break;
            case OpType::RoPE:
                handle_rope_apply_custom_call(ctx, *node, body); break;

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
