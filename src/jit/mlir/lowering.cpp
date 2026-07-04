// Phase 13 / Group B–C — Graph → MLIR text lowering implementation.

#include "tenzor/jit/mlir/lowering.hpp"

#include "tenzor/jit/graph.hpp"
#include "tenzor/jit/mlir/mlir_text_emit.hpp"
#include "tenzor/jit/tracer.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <complex>
#include <cstring>
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

    /// External `func.func private @<callee>(...) -> ...` declarations
    /// required by plugin-path `call @tenzor_plugin.<op>` sites. Emitted
    /// alongside @main in the module wrapper. Order is insertion order; a
    /// set guards against duplicate declarations within one lower call.
    std::vector<std::string> extern_decls;
    std::unordered_set<std::string> extern_decl_keys;

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

    /// Register a `func.func private @<callee>(types...) -> result_type`
    /// declaration. Idempotent — repeated calls with the same key are
    /// no-ops. The full declaration text (sans trailing newline) is stored.
    auto add_extern_decl(const std::string& key, std::string decl) -> void {
        if (extern_decl_keys.insert(key).second) {
            extern_decls.push_back(std::move(decl));
        }
    }
};;

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

/// Emit the true representable minimum (want_max=false) or maximum
/// (want_max=true) of an integer dtype as a decimal string. Used for
/// max-reduction init values and saturating non-finite clamp bounds, both of
/// which must stay inside the result dtype's range (a too-large/invalid splat
/// constant fails MLIR parsing, and 32-bit literals are non-minimal for Int64).
auto int_dtype_extreme_literal(::tenzor::DType d, bool want_max) -> std::string {
    using ::tenzor::DType;
    switch (d) {
        case DType::Int8:
            return std::to_string(want_max
                       ? std::numeric_limits<int8_t>::max()
                       : std::numeric_limits<int8_t>::min());
        case DType::Int16:
            return std::to_string(want_max
                       ? std::numeric_limits<int16_t>::max()
                       : std::numeric_limits<int16_t>::min());
        case DType::Int32:
            return std::to_string(want_max
                       ? std::numeric_limits<int32_t>::max()
                       : std::numeric_limits<int32_t>::min());
        case DType::Int64:
            return std::to_string(want_max
                       ? std::numeric_limits<int64_t>::max()
                       : std::numeric_limits<int64_t>::min());
        case DType::UInt8:
            return std::to_string(want_max
                       ? std::numeric_limits<uint8_t>::max() : 0u);
        case DType::UInt16:
            return std::to_string(want_max
                       ? std::numeric_limits<uint16_t>::max() : 0u);
        case DType::UInt32:
            return std::to_string(want_max
                       ? std::numeric_limits<uint32_t>::max() : 0u);
        case DType::UInt64:
            return want_max
                       ? std::to_string(std::numeric_limits<uint64_t>::max())
                       : std::string("0");
        case DType::Bool:
            return want_max ? "true" : "false";
        default:
            throw std::runtime_error(
                "GraphToMLIR: int_dtype_extreme_literal: non-integer DType");
    }
}

/// Produce a textual `dense<scalar>` literal for a scalar value of the
/// given dtype, matching the syntax StableHLO accepts as a splat constant.
// Emit a single floating-point element for an MLIR `dense<>` payload or splat
// constant. A finite value is written as a decimal literal with enough
// significant digits to round-trip the target element type. A non-finite value
// (±inf / NaN) MUST be written as a hex bit pattern whose WIDTH MATCHES the
// MLIR element type — MLIR's dense<> parser rejects textual `inf`/`nan`, and a
// hex pattern of the wrong width (e.g. an f32 `0x7F800000` under an `f16`
// element type) is rejected or mis-decoded. Centralizes what scalar_literal and
// render_one previously did inconsistently.
inline auto emit_float_literal(std::ostream& os, double value, ::tenzor::DType d)
    -> void {
    using ::tenzor::DType;
    if (std::isfinite(value)) {
        const int prec = (d == DType::Float64) ? 17
                       : (d == DType::Float32) ? 9
                       : 5;  // Float16 / BFloat16
        os << std::scientific << std::setprecision(prec) << value;
        return;
    }
    auto emit_hex = [&os](unsigned long long bits, int width) {
        os << "0x" << std::hex << std::uppercase << std::setfill('0')
           << std::setw(width) << bits
           << std::dec << std::nouppercase << std::setfill(' ');
    };
    switch (d) {
        case DType::Float16: {
            ::tenzor::Float16 h{static_cast<float>(value)};
            std::uint16_t bits;
            std::memcpy(&bits, &h, sizeof(bits));
            emit_hex(bits, 4);
            break;
        }
        case DType::BFloat16: {
            ::tenzor::BFloat16 h{static_cast<float>(value)};
            std::uint16_t bits;
            std::memcpy(&bits, &h, sizeof(bits));
            emit_hex(bits, 4);
            break;
        }
        case DType::Float32: {
            const float f = static_cast<float>(value);
            std::uint32_t bits;
            std::memcpy(&bits, &f, sizeof(bits));
            emit_hex(bits, 8);
            break;
        }
        default: {  // Float64 and any other float dtype
            std::uint64_t bits;
            std::memcpy(&bits, &value, sizeof(bits));
            emit_hex(bits, 16);
            break;
        }
    }
}

// The float element type backing each component of a complex dtype.
inline auto complex_component_dtype(::tenzor::DType d) -> ::tenzor::DType {
    return d == ::tenzor::DType::Complex64 ? ::tenzor::DType::Float32
                                           : ::tenzor::DType::Float64;
}

// Emit one complex element for an MLIR `dense<>` payload as an MLIR
// `(real, imag)` tuple. Each component goes through emit_float_literal so a
// non-finite real/imaginary part is written as a component-width hex pattern
// (f32 for complex<f32>, f64 for complex<f64>), consistent with the real path.
inline auto emit_complex_literal(std::ostream& os, double re, double im,
                                 ::tenzor::DType d) -> void {
    const ::tenzor::DType cd = complex_component_dtype(d);
    os << '(';
    emit_float_literal(os, re, cd);
    os << ',';
    emit_float_literal(os, im, cd);
    os << ')';
}

auto scalar_literal(double value, ::tenzor::DType d) -> std::string {
    using ::tenzor::DType;
    std::ostringstream s;
    if (is_float_dtype(d)) {
        // Finite values round-trip via a decimal literal; non-finite values are
        // emitted as a width-correct hex bit pattern (textual inf/nan is
        // rejected by MLIR's dense<> parser). See emit_float_literal.
        emit_float_literal(s, value, d);
        return s.str();
    }
    if (is_int_dtype(d)) {
        // Integer constants: emit as int literal (no decimal). Guard against
        // non-finite inputs (e.g. ±inf clamp defaults) before the cast, which
        // would otherwise be UB per [conv.fpint]; saturate to the dtype range.
        if (!std::isfinite(value)) {
            return int_dtype_extreme_literal(d, /*want_max=*/value > 0.0);
        }
        // Unsigned dtypes: cast via unsigned long long — a UInt64 value above
        // INT64_MAX cast to signed long long is UB / prints as negative.
        if (d == DType::UInt8 || d == DType::UInt16 ||
            d == DType::UInt32 || d == DType::UInt64) {
            s << static_cast<unsigned long long>(value);
        } else {
            s << static_cast<long long>(value);
        }
        return s.str();
    }
    if (d == DType::Bool) {
        // MLIR dense<> i1 splat constants use 'true'/'false'.
        return value != 0.0 ? "true" : "false";
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
        case DType::Float16:
            return static_cast<double>(
                static_cast<float>(cpu.item<::tenzor::Float16>()));
        case DType::BFloat16:
            return static_cast<double>(
                static_cast<float>(cpu.item<::tenzor::BFloat16>()));
        case DType::Int8:       return static_cast<double>(cpu.item<int8_t>());
        case DType::Int16:      return static_cast<double>(cpu.item<int16_t>());
        case DType::Int32:      return static_cast<double>(cpu.item<int32_t>());
        case DType::Int64:      return static_cast<double>(cpu.item<int64_t>());
        case DType::UInt8:      return static_cast<double>(cpu.item<uint8_t>());
        case DType::UInt16:     return static_cast<double>(cpu.item<uint16_t>());
        case DType::UInt32:     return static_cast<double>(cpu.item<uint32_t>());
        case DType::UInt64:     return static_cast<double>(cpu.item<uint64_t>());
        case DType::Bool:       return cpu.item<bool>() ? 1.0 : 0.0;
        default:
            throw std::runtime_error(
                "GraphToMLIR: extract_scalar_value: unsupported DType");
    }
}

/// Render a per-element MLIR float/int literal at full precision.
///
/// MLIR's dense<> elements-attribute parser doesn't accept `inf`/`-inf`/
/// `nan` keywords for float payloads — those have to be encoded as
/// hexadecimal bit patterns (`0xNNNNNNNN` for f32, `0xNNNNNNNNNNNNNNNN`
/// for f64). std::scientific emits the keywords, which silently produce
/// "expected integer or floating point literal" parse errors deep
/// inside iree-compile. Fall through to the hex form for non-finite
/// values; round-trip for finite ones unchanged.
template <typename T>
auto render_one(std::ostream& os, T v) -> void {
    if constexpr (std::is_floating_point_v<T>) {
        const double d = static_cast<double>(v);
        if (!std::isfinite(d)) {
            // Encode the bit pattern as 0x...  T's natural width is what
            // the surrounding type annotation requires, so use T's size.
            if constexpr (sizeof(T) == 4) {
                std::uint32_t bits;
                std::memcpy(&bits, &v, sizeof(bits));
                os << "0x" << std::hex << std::uppercase
                   << std::setfill('0') << std::setw(8) << bits
                   << std::dec << std::nouppercase << std::setfill(' ');
            } else {
                std::uint64_t bits;
                double dv = d;
                std::memcpy(&bits, &dv, sizeof(bits));
                os << "0x" << std::hex << std::uppercase
                   << std::setfill('0') << std::setw(16) << bits
                   << std::dec << std::nouppercase << std::setfill(' ');
            }
            return;
        }
        // Float64 needs 17 significant digits to round-trip; 9 (enough for
        // float) truncates double constants. Pick precision by element width.
        constexpr int prec = std::is_same_v<T, double> ? 17 : 9;
        os << std::scientific << std::setprecision(prec) << d;
    } else if constexpr (std::is_unsigned_v<T>) {
        // Unsigned integer payloads: print as unsigned so the full range
        // (including UInt64 values with the high bit set) renders as a
        // non-negative literal, which MLIR requires for unsigned element
        // types. A signed cast would emit a negative literal and fail
        // iree-compile verification.
        os << static_cast<unsigned long long>(v);
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

    // Complex constants need `(real, imag)` tuple payloads, which scalar_literal
    // / render_one cannot express — handle them here for the empty, splat, and
    // multi-element cases. IREE/StableHLO accepts complex<f32>/complex<f64>
    // dense constants and complex elementwise ops, so this lets complex JIT
    // graphs compile instead of throwing and falling back to eager.
    using ::tenzor::DType;
    if (d == DType::Complex64 || d == DType::Complex128) {
        auto ccpu = t.device().type == ::tenzor::Device::Type::CPU ? t : t.cpu();
        auto read_at = [&](int64_t i) -> std::pair<double, double> {
            if (d == DType::Complex64) {
                auto v = ccpu.data<std::complex<float>>()[i];
                return {static_cast<double>(v.real()),
                        static_cast<double>(v.imag())};
            }
            auto v = ccpu.data<std::complex<double>>()[i];
            return {v.real(), v.imag()};
        };
        if (t.numel() <= 1) {
            std::ostringstream one;
            if (t.numel() == 1) {
                auto [re, im] = read_at(0);
                emit_complex_literal(one, re, im, d);
            } else {
                emit_complex_literal(one, 0.0, 0.0, d);
            }
            emit_stablehlo_splat_constant(body, name, one.str(), shape, d);
            body << '\n';
            return name;
        }
        std::ostringstream cpayload;
        std::function<void(int64_t, int64_t&)> emit_cdim =
            [&](int64_t dim_idx, int64_t& flat_idx) {
            const int64_t extent = shape[dim_idx];
            cpayload << '[';
            for (int64_t i = 0; i < extent; ++i) {
                if (i != 0) cpayload << ", ";
                if (dim_idx + 1 == static_cast<int64_t>(shape.size())) {
                    auto [re, im] = read_at(flat_idx++);
                    emit_complex_literal(cpayload, re, im, d);
                } else {
                    emit_cdim(dim_idx + 1, flat_idx);
                }
            }
            cpayload << ']';
        };
        int64_t cflat = 0;
        emit_cdim(0, cflat);
        emit_stablehlo_splat_constant(body, name, cpayload.str(), shape, d);
        body << '\n';
        return name;
    }

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
                        emit_float_literal(payload,
                            static_cast<double>(cpu.data<float>()[flat_idx++]), d);
                        break;
                    case DType::Float64:
                        emit_float_literal(payload,
                            cpu.data<double>()[flat_idx++], d);
                        break;
                    case DType::Float16:
                        // Route through emit_float_literal with the ELEMENT
                        // dtype so a non-finite half emits a 16-bit (0x7C00
                        // -style) hex pattern; widening to float and calling
                        // render_one emitted a 32-bit pattern under an f16
                        // element type, which MLIR rejects.
                        emit_float_literal(payload, static_cast<double>(
                            static_cast<float>(
                                cpu.data<::tenzor::Float16>()[flat_idx++])), d);
                        break;
                    case DType::BFloat16:
                        emit_float_literal(payload, static_cast<double>(
                            static_cast<float>(
                                cpu.data<::tenzor::BFloat16>()[flat_idx++])), d);
                        break;
                    case DType::Int8:
                        render_one(payload, cpu.data<int8_t>()[flat_idx++]);
                        break;
                    case DType::Int16:
                        render_one(payload, cpu.data<int16_t>()[flat_idx++]);
                        break;
                    case DType::Int32:
                        render_one(payload, cpu.data<int32_t>()[flat_idx++]);
                        break;
                    case DType::Int64:
                        render_one(payload, cpu.data<int64_t>()[flat_idx++]);
                        break;
                    case DType::UInt8:
                        render_one(payload, cpu.data<uint8_t>()[flat_idx++]);
                        break;
                    case DType::UInt16:
                        render_one(payload, cpu.data<uint16_t>()[flat_idx++]);
                        break;
                    case DType::UInt32:
                        render_one(payload, cpu.data<uint32_t>()[flat_idx++]);
                        break;
                    case DType::UInt64:
                        render_one(payload, cpu.data<uint64_t>()[flat_idx++]);
                        break;
                    case DType::Bool: {
                        // MLIR `dense<>` Bool payloads use 'true'/'false'
                        // literals, not numeric 0/1. The underlying storage is
                        // one byte per element on the CPU side; cast to bool
                        // for correctness against any non-canonical 0/1.
                        const auto* bp = cpu.data<bool>();
                        payload << (bp[flat_idx++] ? "true" : "false");
                        break;
                    }
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
    // Convert each operand to the output element type before broadcasting (JIT-067):
    // emit_stablehlo_binary declares a single element type for both operands and the
    // result, so an operand whose dtype differs from out->dtype() would emit
    // ill-typed IR. Traced graphs already carry explicit Cast nodes (so this is a
    // no-op there); this guards hand-built graphs and any pass that rewrites a
    // binary node without inserting a Cast.
    auto convert_if = [&](const std::string& name,
                          const std::shared_ptr<::tenzor::jit::Value>& val) -> std::string {
        if (val->dtype() == out->dtype()) return name;
        auto c = ctx.fresh_name();
        emit_stablehlo_convert(body, c, name, val->shape(), val->dtype(), out->dtype());
        return c;
    };
    auto a = maybe_broadcast(body, ctx, convert_if(ctx.name_for(a_val->id()), a_val),
                             a_val->shape(), out->shape(), out->dtype());
    auto b = maybe_broadcast(body, ctx, convert_if(ctx.name_for(b_val->id()), b_val),
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
    const auto& cond_in = node.inputs()[0];
    const auto& on_true_in = node.inputs()[1];
    const auto& on_false_in = node.inputs()[2];
    const auto& out = node.outputs()[0];

    // stablehlo.select does NOT broadcast: all three operands must already
    // match the result shape. Tenzor's eager where() broadcasts the predicate
    // and both value operands, so bring each up to out->shape() with the same
    // broadcast_in_dim helper handle_binary uses before emitting the select.
    auto cond = maybe_broadcast(body, ctx, ctx.name_for(cond_in->id()),
                                cond_in->shape(), out->shape(), cond_in->dtype());
    auto on_true = maybe_broadcast(body, ctx, ctx.name_for(on_true_in->id()),
                                   on_true_in->shape(), out->shape(),
                                   on_true_in->dtype());
    auto on_false = maybe_broadcast(body, ctx, ctx.name_for(on_false_in->id()),
                                    on_false_in->shape(), out->shape(),
                                    on_false_in->dtype());
    auto out_name = ctx.fresh_name();
    ctx.bind(out->id(), out_name);

    // The predicate operand of stablehlo.select must be i1. The tracer
    // surfaces the cond as a Bool tensor (eager comparisons return
    // DType::Bool). Use the long-form select syntax so the predicate
    // type and value type can differ:
    //   %out = "stablehlo.select"(%pred, %a, %b)
    //       : (tensor<NxNxi1>, tensor<NxNxf32>, tensor<NxNxf32>)
    //       -> tensor<NxNxf32>
    // After the broadcasts above every operand carries out->shape().
    body << '%' << out_name
         << " = \"stablehlo.select\"(%" << cond << ", %" << on_true
         << ", %" << on_false << ") : (";
    write_tensor_type_for_emit(body, out->shape(), cond_in->dtype());
    body << ", ";
    write_tensor_type_for_emit(body, out->shape(), on_true_in->dtype());
    body << ", ";
    write_tensor_type_for_emit(body, out->shape(), on_false_in->dtype());
    body << ") -> ";
    write_tensor_type_for_emit(body, out->shape(), out->dtype());
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
        const bool has_lo = node.has_attr("min");
        const bool has_hi = node.has_attr("max");
        // For a missing bound, use the dtype's representable extreme rather than
        // a float infinity: for integer dtypes scalar_literal cannot represent
        // ±inf, and even for floats we want a valid in-dtype splat constant.
        std::string lo_lit;
        if (has_lo) {
            lo_lit = scalar_literal(node.get_attr("min"), d);
        } else if (is_int_dtype(d)) {
            lo_lit = int_dtype_extreme_literal(d, /*want_max=*/false);
        } else {
            // -inf as the dtype's IEEE bit pattern.
            switch (d) {
                case ::tenzor::DType::Float64: lo_lit = "0xFFF0000000000000"; break;
                case ::tenzor::DType::Float16: lo_lit = "0xFC00"; break;
                case ::tenzor::DType::BFloat16: lo_lit = "0xFF80"; break;
                default: lo_lit = "0xFF800000"; break;  // Float32
            }
        }
        std::string hi_lit;
        if (has_hi) {
            hi_lit = scalar_literal(node.get_attr("max"), d);
        } else if (is_int_dtype(d)) {
            hi_lit = int_dtype_extreme_literal(d, /*want_max=*/true);
        } else {
            // +inf as the dtype's IEEE bit pattern.
            switch (d) {
                case ::tenzor::DType::Float64: hi_lit = "0x7FF0000000000000"; break;
                case ::tenzor::DType::Float16: hi_lit = "0x7C00"; break;
                case ::tenzor::DType::BFloat16: hi_lit = "0x7F80"; break;
                default: hi_lit = "0x7F800000"; break;  // Float32
            }
        }
        auto lo_name = ctx.fresh_name();
        auto hi_name = ctx.fresh_name();
        emit_stablehlo_splat_constant(body, lo_name, lo_lit, shape, d);
        body << '\n';
        emit_stablehlo_splat_constant(body, hi_name, hi_lit, shape, d);
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

/// GELU (exact, erf-based): gelu(x) = 0.5 * x * (1 + erf(x / sqrt(2))).
/// Matches the eager kernel (approximate='none') and the NVRTC codegen path,
/// so the two JIT backends and eager all agree. Uses `chlo.erf` (verified to
/// lower through this IREE version); the earlier tanh approximation diverged
/// from eager by ~1e-4 and disagreed with the nvrtc path's exact-erf GELU.
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

    // Constants: 0.5, 1.0, 1/sqrt(2).
    auto c_half     = ctx.fresh_name();
    auto c_one      = ctx.fresh_name();
    auto c_invsqrt2 = ctx.fresh_name();         // 1/sqrt(2) ≈ 0.7071067811865476
    emit_stablehlo_splat_constant(body, c_half, scalar_literal(0.5, d),
                                  shape, d); body << '\n';
    emit_stablehlo_splat_constant(body, c_one,  scalar_literal(1.0, d),
                                  shape, d); body << '\n';
    emit_stablehlo_splat_constant(body, c_invsqrt2,
                                  scalar_literal(0.7071067811865476, d),
                                  shape, d);
    body << '\n';

    // scaled = x / sqrt(2)
    auto scaled = ctx.fresh_name();
    emit_stablehlo_binary(body, "multiply", scaled, x, c_invsqrt2, shape, d);
    body << '\n';
    // e = erf(x / sqrt(2))
    auto eval = ctx.fresh_name();
    emit_chlo_unary(body, "erf", eval, scaled, shape, d);
    body << '\n';
    // one_plus = 1 + erf(...)
    auto one_plus = ctx.fresh_name();
    emit_stablehlo_binary(body, "add", one_plus, c_one, eval, shape, d);
    body << '\n';
    // half_x = 0.5 * x
    auto half_x = ctx.fresh_name();
    emit_stablehlo_binary(body, "multiply", half_x, c_half, x, shape, d);
    body << '\n';
    // out = (0.5 * x) * (1 + erf(x / sqrt(2)))
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
    const auto d = out_val->dtype();
    // Accumulate F16/BF16 sums in F32 to match the eager kernel's F32
    // accumulator; reducing in half precision loses accuracy as the reduction
    // length grows.
    const bool widen = (d == ::tenzor::DType::Float16 ||
                        d == ::tenzor::DType::BFloat16);
    const auto cd = widen ? ::tenzor::DType::Float32 : d;
    std::string in_name = ctx.name_for(in_val->id());
    if (widen) {
        auto conv = ctx.fresh_name();
        emit_stablehlo_convert(body, conv, in_name, in_val->shape(), d, cd);
        body << '\n';
        in_name = conv;
    }
    auto reduced = emit_reduce_with_keepdim(
        ctx, body, in_name, in_val->shape(),
        out_val->shape(), dims, keepdim, "add",
        scalar_literal(0.0, cd), cd);
    std::string out_name = reduced;
    if (widen) {
        out_name = ctx.fresh_name();
        emit_stablehlo_convert(body, out_name, reduced, out_val->shape(), cd, d);
        body << '\n';
    }
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
    // -inf init for float (hex bit pattern). For integers use the true
    // per-dtype minimum (0 for unsigned), since a 32-bit literal is invalid
    // for unsigned result dtypes and non-minimal for Int64.
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
        init_lit = int_dtype_extreme_literal(d, /*want_max=*/false);
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
    // Accumulate AND divide in F32 for F16/BF16 (matches eager, which uses an
    // F32 accumulator and divides before narrowing); doing the division in half
    // precision would reintroduce the error the F32 accumulator avoids.
    const bool widen = (d == ::tenzor::DType::Float16 ||
                        d == ::tenzor::DType::BFloat16);
    const auto cd = widen ? ::tenzor::DType::Float32 : d;
    std::string in_name = ctx.name_for(in_val->id());
    if (widen) {
        auto conv = ctx.fresh_name();
        emit_stablehlo_convert(body, conv, in_name, in_val->shape(), d, cd);
        body << '\n';
        in_name = conv;
    }

    // First compute the sum (in compute dtype cd).
    auto sum_name = emit_reduce_with_keepdim(
        ctx, body, in_name, in_val->shape(),
        out_val->shape(), dims, keepdim, "add", scalar_literal(0.0, cd), cd);

    // N = product of reduced extents.
    int64_t N = 1;
    for (auto k : dims) N *= in_val->shape()[k];

    auto n_const = ctx.fresh_name();
    emit_stablehlo_splat_constant(body, n_const,
                                  scalar_literal(static_cast<double>(N), cd),
                                  out_val->shape(), cd);
    body << '\n';

    auto div_name = ctx.fresh_name();
    emit_stablehlo_binary(body, "divide", div_name, sum_name, n_const,
                          out_val->shape(), cd);
    body << '\n';

    std::string out_name = div_name;
    if (widen) {
        out_name = ctx.fresh_name();
        emit_stablehlo_convert(body, out_name, div_name, out_val->shape(), cd, d);
        body << '\n';
    }
    ctx.bind(out_val->id(), out_name);
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

    // Compute the softmax in a widened dtype for F16/BF16 so the exp-sum
    // reduction accumulates in F32 — matching the eager kernel (and the widened
    // handle_sum/handle_mean). Reducing in half precision loses accuracy as the
    // reduction length grows and diverges from eager. cd == d for F32/F64.
    const bool widen = (d == ::tenzor::DType::Float16 ||
                        d == ::tenzor::DType::BFloat16);
    const auto cd = widen ? ::tenzor::DType::Float32 : d;
    std::string xc = x;
    if (widen) {
        auto conv = ctx.fresh_name();
        emit_stablehlo_convert(body, conv, x, shape, d, cd);
        body << '\n';
        xc = conv;
    }

    // 1) reduce_max — -inf init (hex bit pattern per compute dtype).
    std::string init_max = "-2147483648";
    if (cd == ::tenzor::DType::Float32)       init_max = "0xFF800000";
    else if (cd == ::tenzor::DType::Float64)  init_max = "0xFFF0000000000000";
    else if (cd == ::tenzor::DType::Float16)  init_max = "0xFC00";
    else if (cd == ::tenzor::DType::BFloat16) init_max = "0xFF80";
    auto init_m = ctx.fresh_name();
    emit_stablehlo_splat_constant(body, init_m, init_max, {}, cd); body << '\n';
    auto m_name = ctx.fresh_name();
    emit_stablehlo_reduce(body, m_name, xc, init_m, "maximum", reduce_dims,
                          shape, reduced_shape, cd);
    body << '\n';
    // broadcast m back
    auto m_b = ctx.fresh_name();
    emit_stablehlo_broadcast_in_dim(body, m_b, m_name, bcast_dims,
                                    reduced_shape, shape, cd);
    body << '\n';
    // z = x - m
    auto z = ctx.fresh_name();
    emit_stablehlo_binary(body, "subtract", z, xc, m_b, shape, cd);
    body << '\n';
    // e = exp(z)
    auto e = ctx.fresh_name();
    emit_stablehlo_unary(body, "exponential", e, z, shape, cd);
    body << '\n';
    // s = reduce_sum(e)
    auto init_s = ctx.fresh_name();
    emit_stablehlo_splat_constant(body, init_s, scalar_literal(0.0, cd), {}, cd);
    body << '\n';
    auto s_name = ctx.fresh_name();
    emit_stablehlo_reduce(body, s_name, e, init_s, "add", reduce_dims,
                          shape, reduced_shape, cd);
    body << '\n';
    auto s_b = ctx.fresh_name();
    emit_stablehlo_broadcast_in_dim(body, s_b, s_name, bcast_dims,
                                    reduced_shape, shape, cd);
    body << '\n';
    // out = e / s (in compute dtype), then narrow back to storage dtype.
    auto div_out = ctx.fresh_name();
    emit_stablehlo_binary(body, "divide", div_out, e, s_b, shape, cd);
    body << '\n';
    if (widen) {
        auto out_name = ctx.fresh_name();
        ctx.bind(out_val->id(), out_name);
        emit_stablehlo_convert(body, out_name, div_out, shape, cd, d);
        body << '\n';
    } else {
        ctx.bind(out_val->id(), div_out);
    }
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

    // Batch dims: matched leading dims on both sides.  When the ranks
    // differ we right-align matmul (PyTorch / numpy semantics): the
    // smaller-rank operand is treated as having implicit leading 1s and
    // broadcast across the larger operand's batch dims (audit item A.8).
    //
    // Build the canonical batch shape from whichever operand has more
    // batch dims, then broadcast the other operand up to that rank.  Both
    // sides then share the same batch-dim list passed to dot_general.
    const int64_t max_rank = std::max(lr, rr);
    const int64_t batch_rank = max_rank - 2;

    // Canonical batch shape: take the corresponding dim from whichever
    // operand has a non-1 size; mismatches throw (caller should have
    // broadcast-aligned the graph already).
    std::vector<int64_t> batch_shape(batch_rank, 1);
    auto resolve = [&](const std::vector<int64_t>& shape, int64_t r) {
        const int64_t shape_batch = r - 2;
        for (int64_t i = 0; i < shape_batch; ++i) {
            const int64_t out_idx = batch_rank - shape_batch + i;
            const int64_t s = shape[i];
            if (batch_shape[out_idx] == 1) {
                batch_shape[out_idx] = s;
            } else if (s != 1 && s != batch_shape[out_idx]) {
                throw std::runtime_error(
                    "GraphToMLIR: MatMul batch shapes not broadcast-"
                    "compatible at dim " + std::to_string(i));
            }
        }
    };
    resolve(lhs_shape, lr);
    resolve(rhs_shape, rr);

    // Build target shapes for each side: (batch_shape..., M, K) for lhs,
    // (batch_shape..., K, N) for rhs.
    std::vector<int64_t> lhs_target = batch_shape;
    lhs_target.push_back(lhs_shape[lr - 2]);  // M
    lhs_target.push_back(lhs_shape[lr - 1]);  // K
    std::vector<int64_t> rhs_target = batch_shape;
    rhs_target.push_back(rhs_shape[rr - 2]);  // K
    rhs_target.push_back(rhs_shape[rr - 1]);  // N

    // Broadcast the operands up to the canonical rank.  maybe_broadcast
    // is a no-op when shapes already match.
    const std::string lhs_name = maybe_broadcast(
        body, ctx, ctx.name_for(lhs->id()),
        std::vector<int64_t>(lhs_shape.begin(), lhs_shape.end()),
        lhs_target, lhs->dtype());
    const std::string rhs_name = maybe_broadcast(
        body, ctx, ctx.name_for(rhs->id()),
        std::vector<int64_t>(rhs_shape.begin(), rhs_shape.end()),
        rhs_target, rhs->dtype());

    std::vector<int64_t> lhs_batch, rhs_batch;
    for (int64_t i = 0; i < batch_rank; ++i) {
        lhs_batch.push_back(i);
        rhs_batch.push_back(i);
    }
    std::vector<int64_t> lhs_contracting = {batch_rank + 1};   // K dim of lhs
    std::vector<int64_t> rhs_contracting = {batch_rank};       // K dim of rhs

    auto out_name = ctx.fresh_name();
    ctx.bind(out->id(), out_name);
    emit_stablehlo_dot_general(body, out_name, lhs_name, rhs_name,
                               lhs_batch, rhs_batch,
                               lhs_contracting, rhs_contracting,
                               lhs_target, rhs_target, out->shape(),
                               out->dtype());
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

    // Normalize a Python/torch-style index (which may be negative or refer
    // past the end) against the dimension extent, matching Tensor::slice in
    // src/core/tensor.cpp: a negative index is offset by `extent`, then the
    // result is clamped to [0, extent]. stablehlo.slice requires
    // non-negative, in-bounds start/limit indices, so the raw negative
    // values recorded in the trace (e.g. x[..., :-1] → end=-1) must be
    // resolved here before emission.
    auto normalize_index = [](int64_t idx, int64_t extent) -> int64_t {
        if (idx < 0) idx += extent;
        if (idx < 0) idx = 0;
        if (idx > extent) idx = extent;
        return idx;
    };

    std::vector<int64_t> starts(rank, 0), limits = in_shape, strides(rank, 1);
    if (node.has_attr("starts") && node.has_attr("ends")) {
        auto sv = node.get_vec_attr("starts");
        auto ev = node.get_vec_attr("ends");
        if (static_cast<int64_t>(sv.size()) != rank ||
            static_cast<int64_t>(ev.size()) != rank) {
            throw std::runtime_error(
                "GraphToMLIR: Slice starts/ends must have rank entries");
        }
        // Honor per-dim step/stride (JIT-013): a strided slice in the vector
        // form previously fell through with stride 1, returning un-strided
        // elements. stablehlo.slice requires positive strides.
        std::vector<int64_t> stv;
        if (node.has_attr("steps")) stv = node.get_vec_attr("steps");
        for (int64_t i = 0; i < rank; ++i) {
            starts[i]  = normalize_index(sv[i], in_shape[i]);
            limits[i]  = normalize_index(ev[i], in_shape[i]);
            if (i < static_cast<int64_t>(stv.size())) {
                if (stv[i] <= 0) {
                    throw std::runtime_error(
                        "GraphToMLIR: Slice requires a positive step "
                        "(stablehlo.slice does not support negative strides)");
                }
                strides[i] = stv[i];
            }
        }
    } else if (node.has_attr("dim")) {
        // Per-dim slice form: dim + start + end (+ optional step).
        const int64_t dim = normalize_dim(node.get_int_attr("dim"), rank);
        starts[dim] = normalize_index(get_attr_int(node, {"start"}, 0),
                                      in_shape[dim]);
        limits[dim] = normalize_index(get_attr_int(node, {"end"}, in_shape[dim]),
                                      in_shape[dim]);
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

    // IREE's stablehlo.convolution -> linalg conversion mis-lowers a degenerate
    // geometry: when an output spatial dim is 1 AND the stride skips part of the
    // padded input (the used extent is smaller than the padded extent), IREE
    // reduces over the whole padded input instead of the kernel window
    // ("operand #1 shape dim 4 vs 3"). Lower those cases without trusting IREE's
    // conv converter. NCHW input, OIHW weight, NCHW output.
    std::vector<int64_t> xs(x->shape().begin(),   x->shape().end());
    std::vector<int64_t> ws(w->shape().begin(),   w->shape().end());
    std::vector<int64_t> os(out->shape().begin(), out->shape().end());
    bool emitted = false;
    if (xs.size() == 4 && ws.size() == 4 && os.size() == 4 && groups == 1) {
        const int64_t N = xs[0], Cin = xs[1], Hin = xs[2], Win = xs[3];
        const int64_t Cout = ws[0], kh = ws[2], kw = ws[3];
        const int64_t Hout = os[2], Wout = os[3];
        const int64_t eff_kh = dilation_h * (kh - 1) + 1;
        const int64_t eff_kw = dilation_w * (kw - 1) + 1;
        const int64_t padded_h = Hin + 2 * pad_h, padded_w = Win + 2 * pad_w;
        const int64_t used_h = (Hout - 1) * stride_h + eff_kh;
        const int64_t used_w = (Wout - 1) * stride_w + eff_kw;
        const bool degenerate = (used_h < padded_h) || (used_w < padded_w);

        if (degenerate) {
            // Explicitly pad the input to [N, Cin, padded_h, padded_w].
            std::string cur = ctx.name_for(x->id());
            std::vector<int64_t> cur_shape = xs;
            if (pad_h > 0 || pad_w > 0) {
                auto pv = ctx.fresh_name();
                emit_stablehlo_splat_constant(body, pv, scalar_literal(0.0, d),
                                              {}, d);
                body << '\n';
                std::vector<int64_t> padded = {N, Cin, padded_h, padded_w};
                auto p = ctx.fresh_name();
                emit_stablehlo_pad(body, p, cur, pv,
                                   /*low=*/{0, 0, pad_h, pad_w},
                                   /*high=*/{0, 0, pad_h, pad_w},
                                   /*interior=*/{0, 0, 0, 0}, cur_shape, padded,
                                   d);
                body << '\n';
                cur = p;
                cur_shape = padded;
            }

            if (Hout == 1 && Wout == 1) {
                // Single receptive window -> im2col: slice the window, flatten,
                // and matmul against the flattened kernel. Pure reshape +
                // dot_general — no stablehlo.convolution at all.
                std::vector<int64_t> win = {N, Cin, kh, kw};
                auto wsl = ctx.fresh_name();
                emit_stablehlo_slice(body, wsl, cur,
                                     /*starts=*/{0, 0, 0, 0},
                                     /*limits=*/{N, Cin, eff_kh, eff_kw},
                                     /*strides=*/{1, 1, dilation_h, dilation_w},
                                     cur_shape, win, d);
                body << '\n';
                const int64_t K = Cin * kh * kw;
                auto win_flat = ctx.fresh_name();
                emit_stablehlo_reshape(body, win_flat, wsl, win, {N, K}, d);
                body << '\n';
                auto w_flat = ctx.fresh_name();
                emit_stablehlo_reshape(body, w_flat, ctx.name_for(w->id()), ws,
                                       {Cout, K}, d);
                body << '\n';
                auto mm = ctx.fresh_name();
                emit_stablehlo_dot_general(body, mm, win_flat, w_flat,
                                           /*lhs_batch=*/{}, /*rhs_batch=*/{},
                                           /*lhs_contracting=*/{1},
                                           /*rhs_contracting=*/{1},
                                           {N, K}, {Cout, K}, {N, Cout}, d);
                body << '\n';
                emit_stablehlo_reshape(body, conv_name, mm, {N, Cout}, os, d);
                body << '\n';
            } else {
                // Partial collapse (e.g. 1xN): trim the never-read padded tail so
                // the stride divides exactly, then emit an UNPADDED convolution —
                // the exact geometry IREE converts correctly.
                std::vector<int64_t> trimmed = {N, Cin, used_h, used_w};
                auto sl = ctx.fresh_name();
                emit_stablehlo_slice(body, sl, cur,
                                     /*starts=*/{0, 0, 0, 0},
                                     /*limits=*/{N, Cin, used_h, used_w},
                                     /*strides=*/{1, 1, 1, 1}, cur_shape, trimmed,
                                     d);
                body << '\n';
                body << '%' << conv_name << " = stablehlo.convolution(%" << sl
                     << ", %" << ctx.name_for(w->id()) << ")\n"
                     << "    dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, "
                        "1],\n"
                     << "    window = {stride = [" << stride_h << ", " << stride_w
                     << "], pad = [[0, 0], [0, 0]], rhs_dilate = [" << dilation_h
                     << ", " << dilation_w
                     << "]} {batch_group_count = 1 : i64, "
                        "feature_group_count = 1 : i64} : (";
                write_tensor_type_for_emit(body, trimmed, d);
                body << ", ";
                write_tensor_type_for_emit(body, ws, d);
                body << ") -> ";
                write_tensor_type_for_emit(body, os, d);
                body << '\n';
            }
            emitted = true;
        }
    }

    if (!emitted) {
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
    }

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
    // Read a (H, W) pair from either:
    //   - a vec_attr `vec_key` containing [H, W] (rectangular kernels)
    //   - a vec_attr `vec_key` containing [S]    (square kernels via
    //     copy_hw_pair in tracing_interceptor)
    //   - a scalar int_attr `vec_key` (the pooling-layer record path:
    //     src/nn/layers/pooling.cpp posts `kernel_size`/`stride`/
    //     `padding` as int_attrs because the layer's API is square-
    //     only). Previously we silently fell through to the default,
    //     dropping the real kernel size and emitting a 1x1 window.
    //   - per-axis int_attrs `h_key`/`w_key`
    // Per-axis fallback (fb_h, fb_w): the width fallback must be independent of
    // the height one so a rectangular pool with an unspecified stride defaults to
    // stride == kernel_size per axis (JIT-007). Previously a single `fallback`
    // made the width stride default to the kernel HEIGHT.
    auto get_hw = [&](const char* vec_key, const char* h_key,
                      const char* w_key, int64_t fb_h, int64_t fb_w) {
        if (node.has_vec_attr(vec_key)) {
            auto v = node.get_vec_attr(vec_key);
            if (v.size() == 2) return std::pair{v[0], v[1]};
            if (v.size() == 1) return std::pair{v[0], v[0]};
        }
        if (node.has_int_attr(vec_key)) {
            int64_t s = node.get_int_attr(vec_key);
            return std::pair{s, s};
        }
        int64_t h = node.has_int_attr(h_key) ? node.get_int_attr(h_key)
                                             : fb_h;
        int64_t w = node.has_int_attr(w_key) ? node.get_int_attr(w_key)
                                             : fb_w;
        return std::pair{h, w};
    };
    auto [kh, kw] = get_hw("kernel_size", "kernel_h", "kernel_w", 1, 1);
    auto [sh, sw] = get_hw("stride",      "stride_h", "stride_w",
                           /*fallback to kernel per axis*/ kh, kw);
    auto [ph, pw] = get_hw("padding",     "padding_h", "padding_w", 0, 0);
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

    // -inf init for the max reduce-window, as a width-correct hex bit pattern.
    // The literal must match the element bit-width or iree-compile rejects the
    // module (hex float out of range) — f16/bf16 need 16-bit patterns, not the
    // 32-bit f32 one. Mirrors handle_max's per-dtype init.
    std::string init_max;
    if (d == ::tenzor::DType::Float64) {
        init_max = "0xFFF0000000000000";  // IEEE 754 binary64 -inf
    } else if (d == ::tenzor::DType::Float16) {
        init_max = "0xFC00";  // IEEE 754 binary16 -inf
    } else if (d == ::tenzor::DType::BFloat16) {
        init_max = "0xFF80";  // bfloat16 -inf
    } else {
        init_max = "0xFF800000";  // IEEE 754 binary32 -inf (Float32 / fallback)
    }
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
    // Honor count_include_pad (default true, matching PyTorch/eager). When false,
    // border cells must divide by the count of REAL (non-padding) elements, not
    // the full window area (eager: pooling.cpp divisor = count_include_pad ?
    // kh*kw : valid_count).
    // Stored as an int attr (see tracing_interceptor copy_int); default true.
    const bool count_include_pad =
        node.has_int_attr("count_include_pad")
            ? node.get_int_attr("count_include_pad") != 0 : true;

    // Accumulate F16/BF16 in F32 to match the eager kernel's float accumulator
    // (pooling.cpp Compute=float); reduce-window in half precision loses
    // accuracy. cd == d for F32/F64; narrow back to d at the end.
    const bool widen = (d == ::tenzor::DType::Float16 ||
                        d == ::tenzor::DType::BFloat16);
    const auto cd = widen ? ::tenzor::DType::Float32 : d;
    std::string x_name = ctx.name_for(x->id());
    if (widen) {
        auto xc = ctx.fresh_name();
        emit_stablehlo_convert(body, xc, x_name, x->shape(), d, cd);
        body << '\n';
        x_name = xc;
    }

    auto init_name = ctx.fresh_name();
    emit_stablehlo_splat_constant(body, init_name, scalar_literal(0.0, cd), {},
                                  cd);
    body << '\n';

    auto sum_name = ctx.fresh_name();
    emit_reduce_window(body, ctx, sum_name, x_name, init_name,
                       "add", win, str, plo, phi, x->shape(), out->shape(),
                       cd);
    body << '\n';

    std::string result_name = ctx.fresh_name();
    if (count_include_pad) {
        // Divide by the full window area kH * kW. Padded cells contribute 0 to
        // the sum (via the 0.0 init) but are still counted in the divisor.
        const int64_t area = win[2] * win[3];
        auto area_const = ctx.fresh_name();
        emit_stablehlo_splat_constant(
            body, area_const, scalar_literal(static_cast<double>(area), cd),
            out->shape(), cd);
        body << '\n';
        emit_stablehlo_binary(body, "divide", result_name, sum_name, area_const,
                              out->shape(), cd);
        body << '\n';
    } else {
        // Per-position count of real (non-padding) elements: reduce_window("add")
        // over an all-ones tensor with the SAME window/stride/padding. Padded
        // positions are the 0.0 init and contribute 0, so each output holds the
        // number of in-bounds elements that fell in its window.
        auto ones_name = ctx.fresh_name();
        emit_stablehlo_splat_constant(body, ones_name, scalar_literal(1.0, cd),
                                      x->shape(), cd);
        body << '\n';
        auto cnt_init = ctx.fresh_name();
        emit_stablehlo_splat_constant(body, cnt_init, scalar_literal(0.0, cd),
                                      {}, cd);
        body << '\n';
        auto count_name = ctx.fresh_name();
        emit_reduce_window(body, ctx, count_name, ones_name, cnt_init, "add",
                           win, str, plo, phi, x->shape(), out->shape(), cd);
        body << '\n';
        emit_stablehlo_binary(body, "divide", result_name, sum_name, count_name,
                              out->shape(), cd);
        body << '\n';
    }

    if (widen) {
        auto narrowed = ctx.fresh_name();
        emit_stablehlo_convert(body, narrowed, result_name, out->shape(), cd, d);
        body << '\n';
        result_name = narrowed;
    }
    ctx.bind(out->id(), result_name);
}

/// AdaptiveAvgPool2d: compute kernel/stride to bring (H_in, W_in) →
/// (H_out, W_out). Uses a single uniform window stride = H_in/H_out,
/// kernel = H_in - (H_out - 1) * stride. This only matches PyTorch's
/// per-output-cell windows (start=floor(i*H_in/H_out),
/// end=ceil((i+1)*H_in/H_out)) when the input extent is an exact multiple of
/// the output extent; otherwise the cell sizes/divisors vary per position and
/// a uniform window silently diverges from eager. We therefore require exact
/// divisibility and throw a clear error for the non-divisible case.
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
    if (H_out <= 0 || W_out <= 0) {
        throw std::runtime_error(
            "GraphToMLIR: AdaptiveAvgPool2d requires positive output size");
    }
    if (H_in % H_out != 0 || W_in % W_out != 0) {
        throw std::runtime_error(
            "GraphToMLIR: AdaptiveAvgPool2d with non-divisible output size "
            "(H_in=" + std::to_string(H_in) + ", H_out=" +
            std::to_string(H_out) + ", W_in=" + std::to_string(W_in) +
            ", W_out=" + std::to_string(W_out) +
            ") is not supported by the JIT lowering; the uniform-window fast "
            "path only matches eager when H_in/W_in are exact multiples of "
            "H_out/W_out");
    }
    const int64_t sh = H_in / H_out;
    const int64_t sw = W_in / W_out;
    const int64_t kh = H_in - (H_out - 1) * sh;
    const int64_t kw = W_in - (W_out - 1) * sw;

    std::vector<int64_t> win = {1, 1, kh, kw};
    std::vector<int64_t> str = {1, 1, sh, sw};
    std::vector<int64_t> plo = {0, 0, 0, 0};
    std::vector<int64_t> phi = {0, 0, 0, 0};

    // Accumulate F16/BF16 in F32 to match the eager float accumulator. Adaptive
    // pooling has no padding, so count_include_pad is irrelevant here. cd == d
    // for F32/F64; narrow back to d at the end.
    const bool widen = (d == ::tenzor::DType::Float16 ||
                        d == ::tenzor::DType::BFloat16);
    const auto cd = widen ? ::tenzor::DType::Float32 : d;
    std::string x_name = ctx.name_for(x_val->id());
    if (widen) {
        auto xc = ctx.fresh_name();
        emit_stablehlo_convert(body, xc, x_name, xs, d, cd);
        body << '\n';
        x_name = xc;
    }

    auto init_name = ctx.fresh_name();
    emit_stablehlo_splat_constant(body, init_name, scalar_literal(0.0, cd), {},
                                  cd);
    body << '\n';
    auto sum_name = ctx.fresh_name();
    emit_reduce_window(body, ctx, sum_name, x_name,
                       init_name, "add", win, str, plo, phi, xs, os, cd);
    body << '\n';
    const int64_t area = kh * kw;
    auto area_const = ctx.fresh_name();
    emit_stablehlo_splat_constant(
        body, area_const, scalar_literal(static_cast<double>(area), cd), os, cd);
    body << '\n';
    std::string out_name = ctx.fresh_name();
    emit_stablehlo_binary(body, "divide", out_name, sum_name, area_const, os,
                          cd);
    body << '\n';
    if (widen) {
        auto narrowed = ctx.fresh_name();
        emit_stablehlo_convert(body, narrowed, out_name, os, cd, d);
        body << '\n';
        out_name = narrowed;
    }
    ctx.bind(out_val->id(), out_name);
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

    // This handler emits CONSTANT padding. Honor a "value" attr for the fill
    // (default 0). If a non-constant mode is requested (reflect/replicate), fail
    // loudly rather than silently emitting zeros. Those modes reach the graph
    // only via hand-built/ONNX graphs — the traced nn.*Pad layers decompose into
    // primitive ops (full/cat/reflection-index) that lower directly and never
    // produce this node — but we must not mis-lower them if they do appear.
    if (node.has_int_attr("mode") && node.get_int_attr("mode") != 0) {
        throw std::runtime_error(
            "GraphToMLIR: Padding handler only supports constant mode; mode=" +
            std::to_string(node.get_int_attr("mode")) +
            " (reflect/replicate) is not supported");
    }
    const double pad_value = node.has_float_attr("value")
                                 ? static_cast<double>(node.get_attr("value"))
                                 : 0.0;
    auto padval = ctx.fresh_name();
    emit_stablehlo_splat_constant(body, padval, scalar_literal(pad_value, d),
                                  {}, d);
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

    // This handler only implements nearest-neighbor upsampling. If the traced
    // op requested bilinear/bicubic/trilinear (mode != 0 nearest), fail loudly
    // rather than silently emitting nearest — which computes visibly wrong
    // (un-interpolated) pixels identically on every backend. The tracer records
    // AttrKey::Mode as int "mode" (0=nearest,1=bilinear,2=bicubic,3=trilinear).
    if (node.has_int_attr("mode") && node.get_int_attr("mode") != 0) {
        throw std::runtime_error(
            "GraphToMLIR: Interpolate only implements nearest-neighbor (mode=0);"
            " mode=" + std::to_string(node.get_int_attr("mode")) +
            " (bilinear/bicubic/trilinear) is not yet supported");
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

    // Compute the normalization (and affine) in a widened dtype for F16/BF16 so
    // the mean/variance reductions accumulate in F32 — matching the eager kernel
    // and the widened handle_sum/handle_mean/handle_softmax. Reducing in half
    // precision loses accuracy as the normalized length grows. cd == d for
    // F32/F64. Widened gamma/beta keep the affine in the same compute dtype;
    // the final result is narrowed back to the storage dtype d.
    const bool widen = (d == ::tenzor::DType::Float16 ||
                        d == ::tenzor::DType::BFloat16);
    const auto cd = widen ? ::tenzor::DType::Float32 : d;
    std::string x = ctx.name_for(x_val->id());
    if (widen) {
        auto xc = ctx.fresh_name();
        emit_stablehlo_convert(body, xc, x, shape, d, cd);
        body << '\n';
        x = xc;
    }

    // sum_x = reduce_sum(x, norm_dims)
    auto init0 = ctx.fresh_name();
    emit_stablehlo_splat_constant(body, init0, scalar_literal(0.0, cd), {}, cd);
    body << '\n';
    auto sum_x = ctx.fresh_name();
    emit_stablehlo_reduce(body, sum_x, x, init0, "add", norm_dims, shape,
                          reduced_shape, cd);
    body << '\n';
    // n_const = N
    auto n_const = ctx.fresh_name();
    emit_stablehlo_splat_constant(body, n_const,
                                  scalar_literal(static_cast<double>(N), cd),
                                  reduced_shape, cd);
    body << '\n';
    auto mean_r = ctx.fresh_name();
    emit_stablehlo_binary(body, "divide", mean_r, sum_x, n_const,
                          reduced_shape, cd);
    body << '\n';
    auto mean_b = ctx.fresh_name();
    emit_stablehlo_broadcast_in_dim(body, mean_b, mean_r, bcast_dims,
                                    reduced_shape, shape, cd);
    body << '\n';
    // centered = x - mean
    auto centered = ctx.fresh_name();
    emit_stablehlo_binary(body, "subtract", centered, x, mean_b, shape, cd);
    body << '\n';
    // sq = centered * centered
    auto sq = ctx.fresh_name();
    emit_stablehlo_binary(body, "multiply", sq, centered, centered, shape, cd);
    body << '\n';
    // sum_sq
    auto init1 = ctx.fresh_name();
    emit_stablehlo_splat_constant(body, init1, scalar_literal(0.0, cd), {}, cd);
    body << '\n';
    auto sum_sq = ctx.fresh_name();
    emit_stablehlo_reduce(body, sum_sq, sq, init1, "add", norm_dims, shape,
                          reduced_shape, cd);
    body << '\n';
    auto var_r = ctx.fresh_name();
    emit_stablehlo_binary(body, "divide", var_r, sum_sq, n_const,
                          reduced_shape, cd);
    body << '\n';
    // var + eps
    auto eps_c = ctx.fresh_name();
    emit_stablehlo_splat_constant(body, eps_c,
                                  scalar_literal(static_cast<double>(eps), cd),
                                  reduced_shape, cd);
    body << '\n';
    auto var_eps = ctx.fresh_name();
    emit_stablehlo_binary(body, "add", var_eps, var_r, eps_c, reduced_shape, cd);
    body << '\n';
    auto std_r = ctx.fresh_name();
    emit_stablehlo_unary(body, "sqrt", std_r, var_eps, reduced_shape, cd);
    body << '\n';
    auto std_b = ctx.fresh_name();
    emit_stablehlo_broadcast_in_dim(body, std_b, std_r, bcast_dims,
                                    reduced_shape, shape, cd);
    body << '\n';
    auto x_hat = ctx.fresh_name();
    emit_stablehlo_binary(body, "divide", x_hat, centered, std_b, shape, cd);
    body << '\n';

    // Widen a gamma/beta operand to the compute dtype (no-op when already cd).
    auto to_compute = [&](const std::shared_ptr<::tenzor::jit::Value>& val)
        -> std::string {
        std::string name = ctx.name_for(val->id());
        if (val->dtype() != cd) {
            auto c = ctx.fresh_name();
            emit_stablehlo_convert(body, c, name, val->shape(), val->dtype(), cd);
            body << '\n';
            name = c;
        }
        return name;
    };

    // Apply weight/bias if present (inputs[1], inputs[2]).
    std::string final_name = x_hat;
    if (node.inputs().size() >= 2) {
        const auto& w_val = node.inputs()[1];
        auto w_b = maybe_broadcast(body, ctx, to_compute(w_val),
                                   w_val->shape(), shape, cd);
        auto scaled = ctx.fresh_name();
        emit_stablehlo_binary(body, "multiply", scaled, x_hat, w_b, shape, cd);
        body << '\n';
        final_name = scaled;
    }
    if (node.inputs().size() >= 3) {
        const auto& b_val = node.inputs()[2];
        auto b_b = maybe_broadcast(body, ctx, to_compute(b_val),
                                   b_val->shape(), shape, cd);
        auto shifted = ctx.fresh_name();
        emit_stablehlo_binary(body, "add", shifted, final_name, b_b, shape, cd);
        body << '\n';
        final_name = shifted;
    }

    // Narrow the result back to the storage dtype (no-op when cd == d).
    if (widen) {
        auto out_name = ctx.fresh_name();
        emit_stablehlo_convert(body, out_name, final_name, shape, cd, d);
        body << '\n';
        final_name = out_name;
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

    // The MLIR/IREE path compiles inference graphs only (the grad/training
    // variant is replayed through the interpreter and is never lowered here), so
    // BatchNorm is always in eval mode and normalizes with the running
    // statistics. If a training-mode BatchNorm node ever reaches this lowering,
    // batch_norm_inference would silently use running stats instead of batch
    // stats (and skip the running-stat update) — a real divergence from eager.
    // Fail loudly instead of producing wrong numerics.
    if (node.get_bool_attr("training")) {
        throw std::runtime_error(
            "GraphToMLIR: training-mode BatchNorm2d cannot be lowered to "
            "batch_norm_inference; use the interpreter path (compile without "
            "MLIR) for training-through-JIT");
    }

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
    const auto shape = out->shape();
    const auto d = out->dtype();

    // Lower a small non-negative INTEGER exponent to repeated multiply.
    // stablehlo.power is implemented as exp(y*log(x)) on GPU/SPIR-V HAL targets,
    // which returns NaN for ANY negative base — diverging from eager (which
    // multiplies) and from the llvm-cpu target (libm pow). Repeated multiply is
    // exact and NaN-free for negative bases. A fractional/negative exponent
    // still uses stablehlo.power (a negative base to a fractional power is
    // genuinely NaN under eager too, so semantics match there).
    auto lower_int_pow = [&](const std::string& a, int n) {
        if (n == 0) {
            auto out_name = ctx.fresh_name();
            ctx.bind(out->id(), out_name);
            emit_stablehlo_splat_constant(body, out_name, scalar_literal(1.0f, d),
                                          shape, d);
            body << '\n';
            return;
        }
        std::string cur = a;
        for (int i = 1; i < n; ++i) {
            auto next = ctx.fresh_name();
            emit_stablehlo_binary(body, "multiply", next, cur, a, shape, d);
            body << '\n';
            cur = next;
        }
        if (n == 1) {
            // x^1 == x; realize with a multiply-by-one so the output gets its own
            // SSA value (1*x == x exactly under IEEE round-to-nearest).
            auto one_name = ctx.fresh_name();
            emit_stablehlo_splat_constant(body, one_name, scalar_literal(1.0f, d),
                                          shape, d);
            body << '\n';
            auto out_name = ctx.fresh_name();
            ctx.bind(out->id(), out_name);
            emit_stablehlo_binary(body, "multiply", out_name, a, one_name, shape, d);
            body << '\n';
            return;
        }
        ctx.bind(out->id(), cur);  // cur holds x^n for n >= 2
    };

    if (node.inputs().size() == 2) {
        const auto& a_val = node.inputs()[0];
        const auto& b_val = node.inputs()[1];
        // If the exponent is a compile-time Constant with a non-negative integer
        // scalar value, lower to repeated multiply (JIT-051): stablehlo.power is
        // exp(y*log(x)) on GPU/SPIR-V HAL targets and returns NaN for a negative
        // base, whereas eager and the nvrtc path give the correct real value for
        // an integer exponent. lower_int_pow (used by the 1-input path) is NaN-safe.
        if (auto exp_node = b_val->node();
            exp_node && exp_node->op_type() == ::tenzor::jit::OpType::Constant &&
            exp_node->has_attr("value")) {
            const auto& t = exp_node->get_tensor_attr("value");
            if (t.numel() == 1) {
                double ev = t.to(::tenzor::DType::Float64)
                             .to(::tenzor::Device::cpu()).item<double>();
                double r = std::round(ev);
                if (std::abs(ev - r) < 1e-6 && r >= 0.0 && r <= 64.0) {
                    auto a = maybe_broadcast(body, ctx, ctx.name_for(a_val->id()),
                                             a_val->shape(), shape, d);
                    lower_int_pow(a, static_cast<int>(r));  // binds out->id()
                    return;
                }
            }
        }
        // General case: broadcast both operands to the output shape (JIT-066) —
        // stablehlo.power requires equal operand shapes — then emit power.
        auto a = maybe_broadcast(body, ctx, ctx.name_for(a_val->id()),
                                 a_val->shape(), shape, d);
        auto b = maybe_broadcast(body, ctx, ctx.name_for(b_val->id()),
                                 b_val->shape(), shape, d);
        auto out_name = ctx.fresh_name();
        ctx.bind(out->id(), out_name);
        emit_stablehlo_binary(body, "power", out_name, a, b, shape, d);
        body << '\n';
        return;
    }
    if (node.inputs().size() == 1) {
        const auto& a = ctx.name_for(node.inputs()[0]->id());
        float exp = get_attr_float(node, {"exponent"}, 1.0f);
        float r = std::round(exp);
        if (std::abs(exp - r) < 1e-6f && r >= 0.0f && r <= 64.0f) {
            lower_int_pow(a, static_cast<int>(r));
            return;
        }
        auto exp_name = ctx.fresh_name();
        emit_stablehlo_splat_constant(body, exp_name, scalar_literal(exp, d),
                                      shape, d);
        body << '\n';
        auto out_name = ctx.fresh_name();
        ctx.bind(out->id(), out_name);
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

    const bool  causal = get_attr_bool (node, {"causal"}, false);
    const float scale_in = get_attr_float(node, {"scale"},  0.0f);
    // Bake the implicit-default 1/sqrt(D) into the constant we pass to the
    // plugin so the VM-side dispatcher receives a non-zero scale and skips
    // its own default-pick branch (which depends on Q's last dim).
    float scale = scale_in;
    if (scale == 0.0f) {
        const auto q_rank = q_val->shape().size();
        if (q_rank >= 1) {
            const auto D = q_val->shape()[q_rank - 1];
            scale = 1.0f / std::sqrt(static_cast<float>(D));
        }
    }
    int32_t scale_bits;
    std::memcpy(&scale_bits, &scale, sizeof(scale_bits));

    // Emit the i32 scalar args.
    auto scale_name  = ctx.fresh_name();
    auto causal_name = ctx.fresh_name();
    body << "    %" << scale_name  << " = arith.constant "
         << static_cast<int64_t>(scale_bits)  << " : i32\n";
    body << "    %" << causal_name << " = arith.constant "
         << (causal ? 1 : 0) << " : i32\n";

    // Register `func.func private @tenzor_plugin.flash_attention(...) -> ...`.
    auto type_str = [](const std::vector<int64_t>& shape,
                       ::tenzor::DType d) -> std::string {
        std::ostringstream s;
        s << "tensor<";
        for (auto dim : shape) s << dim << 'x';
        s << mlir_type_name(d) << '>';
        return s.str();
    };
    std::ostringstream decl;
    decl << "func.func private @tenzor_plugin.flash_attention("
         << type_str(q_val->shape(), q_val->dtype()) << ", "
         << type_str(k_val->shape(), k_val->dtype()) << ", "
         << type_str(v_val->shape(), v_val->dtype()) << ", i32, i32) -> "
         << type_str(out_val->shape(), out_val->dtype());
    ctx.add_extern_decl(decl.str(), decl.str());

    auto out_name = ctx.fresh_name();
    ctx.bind(out_val->id(), out_name);
    body << "    ";
    emit_plugin_call(body, "tenzor_plugin.flash_attention", out_name,
                     {ctx.name_for(q_val->id()),
                      ctx.name_for(k_val->id()),
                      ctx.name_for(v_val->id())},
                     {q_val->shape(), k_val->shape(), v_val->shape()},
                     {q_val->dtype(), k_val->dtype(), v_val->dtype()},
                     {{scale_name, "i32"}, {causal_name, "i32"}},
                     out_val->shape(), out_val->dtype());
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
    const float scale_in = get_attr_float(node, {"scale"},  0.0f);
    float scale = scale_in;
    if (scale == 0.0f) {
        const auto q_rank = q_val->shape().size();
        if (q_rank >= 1) {
            const auto D = q_val->shape()[q_rank - 1];
            scale = 1.0f / std::sqrt(static_cast<float>(D));
        }
    }
    int32_t scale_bits;
    std::memcpy(&scale_bits, &scale, sizeof(scale_bits));

    auto scale_name  = ctx.fresh_name();
    auto causal_name = ctx.fresh_name();
    body << "    %" << scale_name  << " = arith.constant "
         << static_cast<int64_t>(scale_bits)  << " : i32\n";
    body << "    %" << causal_name << " = arith.constant "
         << (causal ? 1 : 0) << " : i32\n";

    auto type_str = [](const std::vector<int64_t>& shape,
                       ::tenzor::DType d) -> std::string {
        std::ostringstream s;
        s << "tensor<";
        for (auto dim : shape) s << dim << 'x';
        s << mlir_type_name(d) << '>';
        return s.str();
    };
    std::ostringstream decl;
    decl << "func.func private @tenzor_plugin.gqa("
         << type_str(q_val->shape(), q_val->dtype()) << ", "
         << type_str(k_val->shape(), k_val->dtype()) << ", "
         << type_str(v_val->shape(), v_val->dtype()) << ", i32, i32) -> "
         << type_str(out_val->shape(), out_val->dtype());
    ctx.add_extern_decl(decl.str(), decl.str());

    auto out_name = ctx.fresh_name();
    ctx.bind(out_val->id(), out_name);
    body << "    ";
    emit_plugin_call(body, "tenzor_plugin.gqa", out_name,
                     {ctx.name_for(q_val->id()),
                      ctx.name_for(k_val->id()),
                      ctx.name_for(v_val->id())},
                     {q_val->shape(), k_val->shape(), v_val->shape()},
                     {q_val->dtype(), k_val->dtype(), v_val->dtype()},
                     {{scale_name, "i32"}, {causal_name, "i32"}},
                     out_val->shape(), out_val->dtype());
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

    // RoPE has no compile-time-known scalars on the calling path (offset
    // is baked into the cos/sin tables by the caller, per the dispatcher's
    // contract). No extra arith.constant emissions needed.

    auto type_str = [](const std::vector<int64_t>& shape,
                       ::tenzor::DType d) -> std::string {
        std::ostringstream s;
        s << "tensor<";
        for (auto dim : shape) s << dim << 'x';
        s << mlir_type_name(d) << '>';
        return s.str();
    };
    std::ostringstream decl;
    decl << "func.func private @tenzor_plugin.rope_apply("
         << type_str(x_val->shape(),   x_val->dtype())   << ", "
         << type_str(cos_val->shape(), cos_val->dtype()) << ", "
         << type_str(sin_val->shape(), sin_val->dtype()) << ") -> "
         << type_str(out_val->shape(), out_val->dtype());
    ctx.add_extern_decl(decl.str(), decl.str());

    auto out_name = ctx.fresh_name();
    ctx.bind(out_val->id(), out_name);
    body << "    ";
    emit_plugin_call(body, "tenzor_plugin.rope_apply", out_name,
                     {ctx.name_for(x_val->id()),
                      ctx.name_for(cos_val->id()),
                      ctx.name_for(sin_val->id())},
                     {x_val->shape(), cos_val->shape(), sin_val->shape()},
                     {x_val->dtype(), cos_val->dtype(), sin_val->dtype()},
                     {},  // no scalar args
                     out_val->shape(), out_val->dtype());
    body << '\n';
}


/// RMSNorm. Inputs: (x, weight). `weight` is optional in the eager API but
/// the lowering requires it as a second input value (use an all-ones
/// constant when the layer was constructed without weight). Attrs: `eps`
/// (float, default 1e-6).
auto handle_rms_norm_custom_call(LoweringContext& ctx,
                                 const ::tenzor::jit::Node& node,
                                 std::ostream& body) -> void {
    if (node.inputs().size() < 1 || node.outputs().empty()) {
        throw std::runtime_error(
            "GraphToMLIR: RMSNorm expects 1+ inputs and 1+ outputs");
    }
    const auto& x_val   = node.inputs()[0];
    const auto& out_val = node.outputs()[0];

    const float eps = get_attr_float(node, {"eps"}, 1e-6f);
    int32_t eps_bits;
    std::memcpy(&eps_bits, &eps, sizeof(eps_bits));

    auto eps_name = ctx.fresh_name();
    body << "    %" << eps_name << " = arith.constant "
         << static_cast<int64_t>(eps_bits) << " : i32\n";

    // Resolve / synthesize the weight tensor name. The dispatcher mirror in
    // iree_customcalls.cpp expects (x, weight); when no weight is provided,
    // synthesize a stablehlo.constant of ones with shape (D,) where D is
    // x's last dim. This matches the eager affine-free RMSNorm behavior.
    std::string weight_name;
    std::vector<int64_t> weight_shape;
    ::tenzor::DType weight_dtype;
    if (node.inputs().size() >= 2) {
        const auto& w_val = node.inputs()[1];
        weight_name  = ctx.name_for(w_val->id());
        weight_shape = w_val->shape();
        weight_dtype = w_val->dtype();
    } else {
        if (x_val->shape().empty()) {
            throw std::runtime_error(
                "GraphToMLIR: RMSNorm requires x to have rank >= 1");
        }
        const int64_t D = x_val->shape().back();
        weight_shape = {D};
        weight_dtype = x_val->dtype();
        weight_name  = ctx.fresh_name();
        // Emit a stablehlo.constant dense<1.0> for the synthesized weight.
        body << "    %" << weight_name
             << " = stablehlo.constant dense<1.0";
        if (weight_dtype == ::tenzor::DType::Float64) body << "00";
        body << "> : tensor<" << D << 'x' << mlir_type_name(weight_dtype)
             << ">\n";
    }

    auto type_str = [](const std::vector<int64_t>& shape,
                       ::tenzor::DType d) -> std::string {
        std::ostringstream s;
        s << "tensor<";
        for (auto dim : shape) s << dim << 'x';
        s << mlir_type_name(d) << '>';
        return s.str();
    };
    std::ostringstream decl;
    decl << "func.func private @tenzor_plugin.rms_norm("
         << type_str(x_val->shape(), x_val->dtype()) << ", "
         << type_str(weight_shape,   weight_dtype)   << ", i32) -> "
         << type_str(out_val->shape(), out_val->dtype());
    ctx.add_extern_decl(decl.str(), decl.str());

    auto out_name = ctx.fresh_name();
    ctx.bind(out_val->id(), out_name);
    body << "    ";
    emit_plugin_call(body, "tenzor_plugin.rms_norm", out_name,
                     {ctx.name_for(x_val->id()), weight_name},
                     {x_val->shape(), weight_shape},
                     {x_val->dtype(), weight_dtype},
                     {{eps_name, "i32"}},
                     out_val->shape(), out_val->dtype());
    body << '\n';
}


// ── Tenzor dialect ops: expand-to-stablehlo decomposition (Group D) ──────
//
// When `GraphToMLIR::plugin_enabled() == false`, the 4 dialect ops are
// decomposed to pure StableHLO primitives so the resulting MLIR module
// is consumable by stock iree-compile without the Tenzor plugin.

namespace expand {

/// Emit a `softmax(x, dim=last)` decomposition over a tensor of `shape`
/// and dtype `d`. Returns the SSA name of the result. Implements the
/// standard max-stabilized softmax (subtract max, exp, divide by sum).
auto emit_softmax_last_dim(LoweringContext& ctx, std::ostream& body,
                           const std::string& x,
                           const std::vector<int64_t>& shape,
                           ::tenzor::DType d) -> std::string {
    const int64_t rank = static_cast<int64_t>(shape.size());
    const int64_t dim  = rank - 1;
    std::vector<int64_t> reduce_dims = {dim};
    std::vector<int64_t> reduced_shape;
    std::vector<int64_t> bcast_dims;
    for (int64_t i = 0; i < rank; ++i) {
        if (i != dim) { reduced_shape.push_back(shape[i]); bcast_dims.push_back(i); }
    }

    // Compute the attention softmax in a widened dtype for F16/BF16 so the
    // exp-sum reduction accumulates in F32 (matches eager and handle_softmax).
    const bool widen = (d == ::tenzor::DType::Float16 ||
                        d == ::tenzor::DType::BFloat16);
    const auto cd = widen ? ::tenzor::DType::Float32 : d;
    std::string xc = x;
    if (widen) {
        auto conv = ctx.fresh_name();
        emit_stablehlo_convert(body, conv, x, shape, d, cd);
        body << '\n';
        xc = conv;
    }

    std::string init_max = "0xFF800000";
    if (cd == ::tenzor::DType::Float64)  init_max = "0xFFF0000000000000";
    else if (cd == ::tenzor::DType::Float16)  init_max = "0xFC00";
    else if (cd == ::tenzor::DType::BFloat16) init_max = "0xFF80";

    auto init_m = ctx.fresh_name();
    emit_stablehlo_splat_constant(body, init_m, init_max, {}, cd); body << '\n';
    auto m = ctx.fresh_name();
    emit_stablehlo_reduce(body, m, xc, init_m, "maximum", reduce_dims, shape,
                          reduced_shape, cd); body << '\n';
    auto m_b = ctx.fresh_name();
    emit_stablehlo_broadcast_in_dim(body, m_b, m, bcast_dims, reduced_shape,
                                    shape, cd); body << '\n';
    auto z = ctx.fresh_name();
    emit_stablehlo_binary(body, "subtract", z, xc, m_b, shape, cd); body << '\n';
    auto e = ctx.fresh_name();
    emit_stablehlo_unary(body, "exponential", e, z, shape, cd); body << '\n';
    auto init_s = ctx.fresh_name();
    emit_stablehlo_splat_constant(body, init_s, scalar_literal(0.0, cd), {}, cd);
    body << '\n';
    auto s = ctx.fresh_name();
    emit_stablehlo_reduce(body, s, e, init_s, "add", reduce_dims, shape,
                          reduced_shape, cd); body << '\n';
    auto s_b = ctx.fresh_name();
    emit_stablehlo_broadcast_in_dim(body, s_b, s, bcast_dims, reduced_shape,
                                    shape, cd); body << '\n';
    auto out = ctx.fresh_name();
    emit_stablehlo_binary(body, "divide", out, e, s_b, shape, cd); body << '\n';
    if (widen) {
        auto out_narrow = ctx.fresh_name();
        emit_stablehlo_convert(body, out_narrow, out, shape, cd, d);
        body << '\n';
        return out_narrow;
    }
    return out;
}

/// Build the upper-triangular -inf mask iota for causal attention, of
/// shape (Sq, Sk) where Sq=Sk=`S`, broadcast across batch+head dims.
/// `attn_shape` is the QK shape (B, H, Sq, Sk).
auto emit_causal_mask_and_add(LoweringContext& ctx, std::ostream& body,
                              const std::string& qk_scaled,
                              const std::vector<int64_t>& attn_shape,
                              ::tenzor::DType d) -> std::string {
    const int64_t rank = static_cast<int64_t>(attn_shape.size());
    const int64_t sq   = attn_shape[rank - 2];
    const int64_t sk   = attn_shape[rank - 1];

    auto dtype_suffix = [](::tenzor::DType dt) -> const char* {
        switch (dt) {
            case ::tenzor::DType::Float32:  return "f32";
            case ::tenzor::DType::Float64:  return "f64";
            case ::tenzor::DType::Float16:  return "f16";
            case ::tenzor::DType::BFloat16: return "bf16";
            default: return "f32";
        }
    };

    // iota_q in shape (sq, sk) along dim 0 -> q-row index.
    auto iq = ctx.fresh_name();
    body << '%' << iq
         << " = stablehlo.iota dim = 0 : tensor<" << sq << 'x' << sk << "xi32>\n";
    auto ik = ctx.fresh_name();
    body << '%' << ik
         << " = stablehlo.iota dim = 1 : tensor<" << sq << 'x' << sk << "xi32>\n";
    // For KV-cache / cross-length decode (Sq != Sk), query row q corresponds to
    // absolute position q + (Sk - Sq); key k must be masked when q + (Sk-Sq) < k.
    // Shift iq by the offset before the compare (offset 0 reduces to Sq==Sk).
    const int64_t causal_offset = sk - sq;
    std::string iq_cmp = iq;
    if (causal_offset != 0) {
        auto off_c = ctx.fresh_name();
        body << '%' << off_c << " = stablehlo.constant dense<" << causal_offset
             << "> : tensor<" << sq << 'x' << sk << "xi32>\n";
        auto iq_off = ctx.fresh_name();
        body << '%' << iq_off << " = stablehlo.add %" << iq << ", %" << off_c
             << " : tensor<" << sq << 'x' << sk << "xi32>\n";
        iq_cmp = iq_off;
    }
    // mask = (q + (Sk-Sq) < k)  → cells above the (shifted) diagonal become true.
    auto mask_bool = ctx.fresh_name();
    body << '%' << mask_bool
         << " = stablehlo.compare LT, %" << iq_cmp << ", %" << ik
         << " : (tensor<" << sq << 'x' << sk << "xi32>, tensor<"
         << sq << 'x' << sk << "xi32>) -> tensor<" << sq << 'x' << sk << "xi1>\n";

    // -inf splat in dtype d, shape (sq, sk).
    std::string ninf = "0xFF800000";
    if (d == ::tenzor::DType::Float64)       ninf = "0xFFF0000000000000";
    else if (d == ::tenzor::DType::Float16)  ninf = "0xFC00";
    else if (d == ::tenzor::DType::BFloat16) ninf = "0xFF80";
    auto ninf_c = ctx.fresh_name();
    emit_stablehlo_splat_constant(body, ninf_c, ninf, {sq, sk}, d); body << '\n';
    auto zero_c = ctx.fresh_name();
    emit_stablehlo_splat_constant(body, zero_c, scalar_literal(0.0, d), {sq, sk}, d);
    body << '\n';

    // mask_d = select(mask_bool, -inf, 0)
    auto mask_d = ctx.fresh_name();
    body << '%' << mask_d << " = stablehlo.select %" << mask_bool
         << ", %" << ninf_c << ", %" << zero_c
         << " : tensor<" << sq << 'x' << sk << "xi1>, tensor<"
         << sq << 'x' << sk << 'x' << dtype_suffix(d) << ">\n";

    // Broadcast mask_d from (sq, sk) to attn_shape across last two dims.
    std::vector<int64_t> bcast_dims = {rank - 2, rank - 1};
    auto mask_b = ctx.fresh_name();
    emit_stablehlo_broadcast_in_dim(body, mask_b, mask_d, bcast_dims,
                                    {sq, sk}, attn_shape, d); body << '\n';
    // attn_masked = qk_scaled + mask_b
    auto out = ctx.fresh_name();
    emit_stablehlo_binary(body, "add", out, qk_scaled, mask_b, attn_shape, d);
    body << '\n';
    return out;
}

}  // namespace expand

/// FlashAttention expand-to-stablehlo: softmax(Q @ K^T * scale [+ mask]) @ V.
/// Q/K/V shape: (B, H, S, D) with K/V possibly having H_kv < H — but for
/// FlashAttention (vs GQA) they must match. The mask is the upper-triangular
/// -inf mask when `causal=true`.
auto handle_flash_attention_expand(LoweringContext& ctx,
                                   const ::tenzor::jit::Node& node,
                                   std::ostream& body) -> void {
    if (node.inputs().size() != 3 || node.outputs().empty()) {
        throw std::runtime_error(
            "GraphToMLIR: FlashAttention (expand) expects 3 inputs and 1+ "
            "outputs");
    }
    const auto& q_val   = node.inputs()[0];
    const auto& k_val   = node.inputs()[1];
    const auto& v_val   = node.inputs()[2];
    const auto& out_val = node.outputs()[0];
    const auto q_shape  = q_val->shape();
    const auto k_shape  = k_val->shape();
    const auto v_shape  = v_val->shape();
    const auto d        = out_val->dtype();
    if (q_shape.size() != 4 || k_shape.size() != 4 || v_shape.size() != 4) {
        throw std::runtime_error(
            "GraphToMLIR: FlashAttention (expand) requires 4-D Q/K/V "
            "(B,H,S,D)");
    }
    const int64_t B    = q_shape[0];
    const int64_t H    = q_shape[1];
    const int64_t Sq   = q_shape[2];
    const int64_t D    = q_shape[3];
    const int64_t Sk   = k_shape[2];

    const bool causal = get_attr_bool (node, {"causal"}, false);
    float scale       = get_attr_float(node, {"scale"},  0.0f);
    if (scale == 0.0f) {
        scale = 1.0f / std::sqrt(static_cast<float>(D));
    }

    // Compute the scores, scale, mask and softmax in F32 for F16/BF16 (JIT-006).
    // Previously the QK dot_general result, the *scale and the causal mask were
    // all emitted in the storage dtype d, so F16/BF16 attention truncated the
    // scores to half precision BEFORE softmax — diverging from eager and from the
    // plugin path. Widen Q/K/V to F32, run the whole score pipeline in F32, and
    // narrow only the final attention output back to d (mirrors the norms).
    const bool widen = (d == ::tenzor::DType::Float16 ||
                        d == ::tenzor::DType::BFloat16);
    const auto cd = widen ? ::tenzor::DType::Float32 : d;
    auto q_c = ctx.name_for(q_val->id());
    auto k_c = ctx.name_for(k_val->id());
    auto v_c = ctx.name_for(v_val->id());
    if (widen) {
        auto qn = ctx.fresh_name();
        emit_stablehlo_convert(body, qn, q_c, q_shape, d, cd); body << '\n'; q_c = qn;
        auto kn = ctx.fresh_name();
        emit_stablehlo_convert(body, kn, k_c, k_shape, d, cd); body << '\n'; k_c = kn;
        auto vn = ctx.fresh_name();
        emit_stablehlo_convert(body, vn, v_c, v_shape, d, cd); body << '\n'; v_c = vn;
    }

    // QK = dot_general(Q, K) batch=(0,1) contract Q.dim=3 K.dim=3
    // -> shape (B, H, Sq, Sk)
    const std::vector<int64_t> qk_shape{B, H, Sq, Sk};
    auto qk = ctx.fresh_name();
    emit_stablehlo_dot_general(body, qk, q_c, k_c,
                               /*lhs_batch=*/{0, 1}, /*rhs_batch=*/{0, 1},
                               /*lhs_contracting=*/{3}, /*rhs_contracting=*/{3},
                               q_shape, k_shape, qk_shape, cd);
    body << '\n';

    // qk_scaled = qk * scale
    auto scale_c = ctx.fresh_name();
    emit_stablehlo_splat_constant(body, scale_c,
                                  scalar_literal(static_cast<double>(scale), cd),
                                  qk_shape, cd);
    body << '\n';
    auto qk_s = ctx.fresh_name();
    emit_stablehlo_binary(body, "multiply", qk_s, qk, scale_c, qk_shape, cd);
    body << '\n';

    std::string pre_softmax = qk_s;
    if (causal) {
        pre_softmax = expand::emit_causal_mask_and_add(ctx, body, qk_s, qk_shape, cd);
    }

    auto attn = expand::emit_softmax_last_dim(ctx, body, pre_softmax,
                                              qk_shape, cd);

    // out = dot_general(attn, V) batch=(0,1) contract attn.dim=3 V.dim=2
    // -> (B, H, Sq, D), computed in cd then narrowed back to d.
    auto out_cd = ctx.fresh_name();
    emit_stablehlo_dot_general(body, out_cd, attn, v_c,
                               {0, 1}, {0, 1},
                               {3}, {2},
                               qk_shape, v_shape, out_val->shape(), cd);
    body << '\n';
    if (widen) {
        auto out_name = ctx.fresh_name();
        ctx.bind(out_val->id(), out_name);
        emit_stablehlo_convert(body, out_name, out_cd, out_val->shape(), cd, d);
        body << '\n';
    } else {
        ctx.bind(out_val->id(), out_cd);
    }
}


/// GQA expand-to-stablehlo. K and V each have H_kv heads while Q has H_q,
/// with H_kv evenly dividing H_q (group size G = H_q / H_kv). Strategy:
/// broadcast K, V from (B, H_kv, S, D) to (B, H_q, S, D) by inserting a
/// size-G dim and re-flattening, then run the same FlashAttention expand
/// pipeline.
///
/// Broadcast recipe (StableHLO):
///   reshape K: (B, H_kv, S, D) -> (B, H_kv, 1, S, D)
///   broadcast_in_dim:           -> (B, H_kv, G, S, D)
///   reshape:                    -> (B, H_kv*G, S, D)
auto handle_gqa_expand(LoweringContext& ctx,
                       const ::tenzor::jit::Node& node,
                       std::ostream& body) -> void {
    if (node.inputs().size() != 3 || node.outputs().empty()) {
        throw std::runtime_error(
            "GraphToMLIR: GQA (expand) expects 3 inputs and 1+ outputs");
    }
    const auto& q_val   = node.inputs()[0];
    const auto& k_val   = node.inputs()[1];
    const auto& v_val   = node.inputs()[2];
    const auto& out_val = node.outputs()[0];
    const auto q_shape  = q_val->shape();
    const auto k_shape  = k_val->shape();
    const auto v_shape  = v_val->shape();
    const auto d        = out_val->dtype();
    if (q_shape.size() != 4 || k_shape.size() != 4 || v_shape.size() != 4) {
        throw std::runtime_error(
            "GraphToMLIR: GQA (expand) requires 4-D Q/K/V");
    }
    const int64_t B    = q_shape[0];
    const int64_t Hq   = q_shape[1];
    const int64_t Sq   = q_shape[2];
    const int64_t D    = q_shape[3];
    const int64_t Hkv  = k_shape[1];
    const int64_t Sk   = k_shape[2];
    if (Hkv == 0 || Hq % Hkv != 0) {
        throw std::runtime_error(
            "GraphToMLIR: GQA (expand) requires H_kv | H_q");
    }
    const int64_t G = Hq / Hkv;

    const bool causal = get_attr_bool (node, {"causal"}, false);
    float scale       = get_attr_float(node, {"scale"},  0.0f);
    if (scale == 0.0f) scale = 1.0f / std::sqrt(static_cast<float>(D));

    auto expand_kv = [&](const std::string& src_name,
                         const std::vector<int64_t>& src_shape) -> std::string {
        if (Hkv == Hq) return src_name;  // No-op for non-GQA
        const int64_t S_in = src_shape[2];
        // Step 1: reshape src to (B, Hkv, 1, S, D)
        const std::vector<int64_t> r1{B, Hkv, 1, S_in, D};
        auto reshaped = ctx.fresh_name();
        body << '%' << reshaped << " = stablehlo.reshape %" << src_name
             << " : (";
        write_tensor_type_for_emit(body, src_shape, d);
        body << ") -> ";
        write_tensor_type_for_emit(body, r1, d);
        body << '\n';
        // Step 2: broadcast_in_dim 5D->5D with G replicating dim 2.
        const std::vector<int64_t> r2{B, Hkv, G, S_in, D};
        auto bcasted = ctx.fresh_name();
        emit_stablehlo_broadcast_in_dim(body, bcasted, reshaped,
                                        /*bcast_dims=*/{0, 1, 2, 3, 4},
                                        r1, r2, d);
        body << '\n';
        // Step 3: reshape to (B, Hq, S, D)
        const std::vector<int64_t> r3{B, Hq, S_in, D};
        auto out = ctx.fresh_name();
        body << '%' << out << " = stablehlo.reshape %" << bcasted
             << " : (";
        write_tensor_type_for_emit(body, r2, d);
        body << ") -> ";
        write_tensor_type_for_emit(body, r3, d);
        body << '\n';
        return out;
    };

    auto k_expanded = expand_kv(ctx.name_for(k_val->id()), k_shape);
    auto v_expanded = expand_kv(ctx.name_for(v_val->id()), v_shape);

    const std::vector<int64_t> kv_full_shape{B, Hq, Sk, D};

    // Compute the scores/softmax in F32 for F16/BF16 (JIT-006) — see the
    // FlashAttention expand path. The KV expansion above is pure data movement
    // and stays in d; only the numeric score pipeline is widened.
    const bool widen = (d == ::tenzor::DType::Float16 ||
                        d == ::tenzor::DType::BFloat16);
    const auto cd = widen ? ::tenzor::DType::Float32 : d;
    auto q_c = ctx.name_for(q_val->id());
    auto k_c = k_expanded;
    auto v_c = v_expanded;
    if (widen) {
        auto qn = ctx.fresh_name();
        emit_stablehlo_convert(body, qn, q_c, q_shape, d, cd); body << '\n'; q_c = qn;
        auto kn = ctx.fresh_name();
        emit_stablehlo_convert(body, kn, k_c, kv_full_shape, d, cd); body << '\n'; k_c = kn;
        auto vn = ctx.fresh_name();
        emit_stablehlo_convert(body, vn, v_c, kv_full_shape, d, cd); body << '\n'; v_c = vn;
    }

    // QK = dot_general(Q, K_expanded), batch=(0,1), contract=(3,3) -> (B,Hq,Sq,Sk)
    const std::vector<int64_t> qk_shape{B, Hq, Sq, Sk};
    auto qk = ctx.fresh_name();
    emit_stablehlo_dot_general(body, qk, q_c, k_c,
                               {0, 1}, {0, 1}, {3}, {3},
                               q_shape, kv_full_shape, qk_shape, cd);
    body << '\n';

    auto scale_c = ctx.fresh_name();
    emit_stablehlo_splat_constant(body, scale_c,
                                  scalar_literal(static_cast<double>(scale), cd),
                                  qk_shape, cd);
    body << '\n';
    auto qk_s = ctx.fresh_name();
    emit_stablehlo_binary(body, "multiply", qk_s, qk, scale_c, qk_shape, cd);
    body << '\n';

    std::string pre_softmax = qk_s;
    if (causal) {
        pre_softmax = expand::emit_causal_mask_and_add(ctx, body, qk_s,
                                                       qk_shape, cd);
    }
    auto attn = expand::emit_softmax_last_dim(ctx, body, pre_softmax,
                                              qk_shape, cd);

    // out = dot_general(attn, V_expanded), batch=(0,1), contract=(3,2),
    // computed in cd then narrowed back to d (JIT-006).
    auto out_cd = ctx.fresh_name();
    emit_stablehlo_dot_general(body, out_cd, attn, v_c,
                               {0, 1}, {0, 1}, {3}, {2},
                               qk_shape, kv_full_shape, out_val->shape(), cd);
    body << '\n';
    if (widen) {
        auto out_name = ctx.fresh_name();
        ctx.bind(out_val->id(), out_name);
        emit_stablehlo_convert(body, out_name, out_cd, out_val->shape(), cd, d);
        body << '\n';
    } else {
        ctx.bind(out_val->id(), out_cd);
    }
}


/// RoPE expand-to-stablehlo. Inputs: (x, cos, sin) with x:(B,H,S,D),
/// cos/sin:(S,D). Output: x * cos_b + rotate_half(x) * sin_b where
/// rotate_half splits x along the last dim into x1,x2 (each D/2 wide),
/// concatenates [-x2, x1] along that dim.
///
/// The `offset` int attr is encoded in the table itself by the eager
/// API (precomputed sin/cos for the relevant positions), so the expand
/// path doesn't need to handle offset directly — the cos/sin operands
/// already cover the resumed range.
auto handle_rope_apply_expand(LoweringContext& ctx,
                              const ::tenzor::jit::Node& node,
                              std::ostream& body) -> void {
    if (node.inputs().size() != 3 || node.outputs().empty()) {
        throw std::runtime_error(
            "GraphToMLIR: RoPE (expand) expects 3 inputs (x,cos,sin)");
    }
    const auto& x_val   = node.inputs()[0];
    const auto& cos_val = node.inputs()[1];
    const auto& sin_val = node.inputs()[2];
    const auto& out_val = node.outputs()[0];
    const auto x_shape  = x_val->shape();
    const auto d        = out_val->dtype();
    const int64_t rank  = static_cast<int64_t>(x_shape.size());
    if (rank < 1) {
        throw std::runtime_error("GraphToMLIR: RoPE expand requires rank >= 1");
    }
    const int64_t D = x_shape[rank - 1];
    if (D % 2 != 0) {
        throw std::runtime_error(
            "GraphToMLIR: RoPE expand requires even last dim");
    }
    const int64_t H = D / 2;

    // Half-shape: same as x_shape but with last dim = H.
    std::vector<int64_t> half_shape = x_shape;
    half_shape.back() = H;

    auto dtype_suffix = [](::tenzor::DType dt) -> const char* {
        switch (dt) {
            case ::tenzor::DType::Float32:  return "f32";
            case ::tenzor::DType::Float64:  return "f64";
            case ::tenzor::DType::Float16:  return "f16";
            case ::tenzor::DType::BFloat16: return "bf16";
            default: return "f32";
        }
    };
    auto render_dims = [&](const std::vector<int64_t>& v) {
        std::ostringstream os;
        for (auto x : v) os << x << ',';
        std::string s = os.str();
        if (!s.empty()) s.pop_back();
        return s;
    };

    const auto& x = ctx.name_for(x_val->id());

    // Slice x into x1 = x[..., :H], x2 = x[..., H:].
    // stablehlo.slice uses start_indices/limit_indices/strides per dim.
    std::vector<int64_t> start_lo(rank, 0);
    std::vector<int64_t> start_hi(rank, 0);  start_hi.back() = H;
    std::vector<int64_t> limit_lo = x_shape; limit_lo.back() = H;
    std::vector<int64_t> limit_hi = x_shape;
    std::vector<int64_t> strides(rank, 1);

    auto x1 = ctx.fresh_name();
    body << '%' << x1 << " = \"stablehlo.slice\"(%" << x
         << ") <{start_indices = array<i64: " << render_dims(start_lo)
         << ">, limit_indices = array<i64: " << render_dims(limit_lo)
         << ">, strides = array<i64: " << render_dims(strides)
         << ">}> : (";
    write_tensor_type_for_emit(body, x_shape, d);
    body << ") -> ";
    write_tensor_type_for_emit(body, half_shape, d);
    body << '\n';

    auto x2 = ctx.fresh_name();
    body << '%' << x2 << " = \"stablehlo.slice\"(%" << x
         << ") <{start_indices = array<i64: " << render_dims(start_hi)
         << ">, limit_indices = array<i64: " << render_dims(limit_hi)
         << ">, strides = array<i64: " << render_dims(strides)
         << ">}> : (";
    write_tensor_type_for_emit(body, x_shape, d);
    body << ") -> ";
    write_tensor_type_for_emit(body, half_shape, d);
    body << '\n';

    // -x2
    auto neg_x2 = ctx.fresh_name();
    emit_stablehlo_unary(body, "negate", neg_x2, x2, half_shape, d);
    body << '\n';

    // rotated = concat([-x2, x1], dim=last).
    auto rotated = ctx.fresh_name();
    body << '%' << rotated << " = \"stablehlo.concatenate\"(%" << neg_x2
         << ", %" << x1 << ") <{dimension = " << (rank - 1) << " : i64}> : ("
         << "tensor<";
    for (auto v : half_shape) body << v << 'x';
    body << dtype_suffix(d) << ">, tensor<";
    for (auto v : half_shape) body << v << 'x';
    body << dtype_suffix(d) << ">) -> ";
    write_tensor_type_for_emit(body, x_shape, d);
    body << '\n';

    // Broadcast cos/sin from their actual shapes to x_shape. The table
    // shapes are typically (S, D) when x is (B, H, S, D); the broadcast
    // dims align them to the last two axes. We compute bcast_dims by
    // assuming the table dims correspond to the trailing N dims of x.
    auto broadcast_table = [&](const std::string& tab_name,
                               const std::vector<int64_t>& tab_shape)
        -> std::string {
        if (tab_shape == x_shape) return tab_name;
        std::vector<int64_t> bcast_dims;
        const int64_t n_tab = static_cast<int64_t>(tab_shape.size());
        const int64_t off   = rank - n_tab;
        for (int64_t i = 0; i < n_tab; ++i) bcast_dims.push_back(off + i);
        auto b = ctx.fresh_name();
        emit_stablehlo_broadcast_in_dim(body, b, tab_name, bcast_dims,
                                        tab_shape, x_shape, d);
        body << '\n';
        return b;
    };

    auto cos_b = broadcast_table(ctx.name_for(cos_val->id()),
                                 cos_val->shape());
    auto sin_b = broadcast_table(ctx.name_for(sin_val->id()),
                                 sin_val->shape());

    // x_mul_cos = x * cos
    auto x_mul_cos = ctx.fresh_name();
    emit_stablehlo_binary(body, "multiply", x_mul_cos, x, cos_b, x_shape, d);
    body << '\n';
    // rot_mul_sin = rotated * sin
    auto rot_mul_sin = ctx.fresh_name();
    emit_stablehlo_binary(body, "multiply", rot_mul_sin, rotated, sin_b,
                          x_shape, d);
    body << '\n';
    // out = x_mul_cos + rot_mul_sin
    auto out_name = ctx.fresh_name();
    ctx.bind(out_val->id(), out_name);
    emit_stablehlo_binary(body, "add", out_name, x_mul_cos, rot_mul_sin,
                          x_shape, d);
    body << '\n';
}


/// RMSNorm expand-to-stablehlo. Inputs: (x[, weight]). Output:
///   rms  = sqrt(mean(x^2, dim=-1) + eps)
///   xhat = x / broadcast(rms)
///   out  = xhat * broadcast(weight)  (if weight provided)
///
/// The "normalized_shape" attr is honored if present (multi-trailing-
/// dims), otherwise only the last dim is reduced.
auto handle_rms_norm_expand(LoweringContext& ctx,
                            const ::tenzor::jit::Node& node,
                            std::ostream& body) -> void {
    if (node.inputs().empty() || node.outputs().empty()) {
        throw std::runtime_error(
            "GraphToMLIR: RMSNorm (expand) expects 1+ inputs and 1+ outputs");
    }
    const auto& x_val   = node.inputs()[0];
    const auto& out_val = node.outputs()[0];
    const auto shape    = x_val->shape();
    const auto d        = out_val->dtype();
    const float eps     = get_attr_float(node, {"eps"}, 1e-6f);
    const int64_t rank  = static_cast<int64_t>(shape.size());

    // Resolve norm_dims. If `normalized_shape` is set, the trailing dims of
    // x whose extents match form the reduction axis set; otherwise the
    // last dim only.
    std::vector<int64_t> norm_dims;
    if (node.has_attr("normalized_shape")) {
        auto ns = node.get_vec_attr("normalized_shape");
        const int64_t off = rank - static_cast<int64_t>(ns.size());
        for (int64_t i = off; i < rank; ++i) norm_dims.push_back(i);
    } else {
        norm_dims.push_back(rank - 1);
    }
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
    for (int64_t i = 0; i < rank; ++i) {
        if (std::find(norm_dims.begin(), norm_dims.end(), i) ==
            norm_dims.end()) {
            bcast_dims.push_back(i);
        }
    }
    int64_t N = 1;
    for (auto k : norm_dims) N *= shape[k];

    // Accumulate the sum-of-squares / mean / rsqrt in F32 for F16/BF16 (cd),
    // matching the eager RMSNorm kernel (float sum_sq) and the widened
    // handle_layer_norm/handle_sum/handle_softmax. Reducing in half precision
    // loses accuracy as the normalized length N grows (every transformer block).
    // cd == d for F32/F64; the widened result is narrowed back to d at the end.
    const bool widen = (d == ::tenzor::DType::Float16 ||
                        d == ::tenzor::DType::BFloat16);
    const auto cd = widen ? ::tenzor::DType::Float32 : d;

    std::string x = ctx.name_for(x_val->id());
    if (widen) {
        auto xc = ctx.fresh_name();
        emit_stablehlo_convert(body, xc, x, shape, d, cd);
        body << '\n';
        x = xc;
    }

    // sq = x * x
    auto sq = ctx.fresh_name();
    emit_stablehlo_binary(body, "multiply", sq, x, x, shape, cd); body << '\n';
    // sum_sq = reduce_sum(sq, norm_dims)
    auto init0 = ctx.fresh_name();
    emit_stablehlo_splat_constant(body, init0, scalar_literal(0.0, cd), {}, cd);
    body << '\n';
    auto sum_sq = ctx.fresh_name();
    emit_stablehlo_reduce(body, sum_sq, sq, init0, "add", norm_dims, shape,
                          reduced_shape, cd); body << '\n';
    // mean = sum_sq / N
    auto n_c = ctx.fresh_name();
    emit_stablehlo_splat_constant(body, n_c,
                                  scalar_literal(static_cast<double>(N), cd),
                                  reduced_shape, cd);
    body << '\n';
    auto mean_sq = ctx.fresh_name();
    emit_stablehlo_binary(body, "divide", mean_sq, sum_sq, n_c, reduced_shape, cd);
    body << '\n';
    // var_eps = mean_sq + eps
    auto eps_c = ctx.fresh_name();
    emit_stablehlo_splat_constant(body, eps_c,
                                  scalar_literal(static_cast<double>(eps), cd),
                                  reduced_shape, cd);
    body << '\n';
    auto var_eps = ctx.fresh_name();
    emit_stablehlo_binary(body, "add", var_eps, mean_sq, eps_c, reduced_shape, cd);
    body << '\n';
    // rms = sqrt(var_eps)
    auto rms = ctx.fresh_name();
    emit_stablehlo_unary(body, "sqrt", rms, var_eps, reduced_shape, cd);
    body << '\n';
    auto rms_b = ctx.fresh_name();
    emit_stablehlo_broadcast_in_dim(body, rms_b, rms, bcast_dims,
                                    reduced_shape, shape, cd);
    body << '\n';
    // xhat = x / rms_b
    auto xhat = ctx.fresh_name();
    emit_stablehlo_binary(body, "divide", xhat, x, rms_b, shape, cd);
    body << '\n';

    std::string final_name = xhat;
    if (node.inputs().size() >= 2) {
        const auto& w_val = node.inputs()[1];
        std::string w_name = ctx.name_for(w_val->id());
        if (widen) {
            auto wc = ctx.fresh_name();
            emit_stablehlo_convert(body, wc, w_name, w_val->shape(), d, cd);
            body << '\n';
            w_name = wc;
        }
        auto w_b = maybe_broadcast(body, ctx, w_name, w_val->shape(), shape, cd);
        auto scaled = ctx.fresh_name();
        emit_stablehlo_binary(body, "multiply", scaled, xhat, w_b, shape, cd);
        body << '\n';
        final_name = scaled;
    }

    // Narrow the widened result back to the storage dtype d.
    if (widen) {
        auto narrowed = ctx.fresh_name();
        emit_stablehlo_convert(body, narrowed, final_name, shape, cd, d);
        body << '\n';
        final_name = narrowed;
    }

    // Bind the output value's SSA name to `final_name` — the chained
    // multiply/divide already produced the result tensor under that name,
    // no need to emit an extra copy.
    ctx.bind(out_val->id(), final_name);
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
        // Walk all nodes to find Values referenced as inputs that aren't graph
        // inputs and have a constant tensor backing. Emit in the deterministic
        // node/input walk order (deduped) — iterating an unordered_map here made
        // constant SSA names nondeterministic, which changed the emitted MLIR run
        // to run and defeated the content-hash compile cache.
        std::unordered_set<std::string> emitted_const_ids;
        for (const auto& n : g.nodes()) {
            for (const auto& v : n->inputs()) {
                if (input_ids.count(v->id())) continue;
                if (!constants.count(v->id())) continue;
                if (!emitted_const_ids.insert(v->id()).second) continue;  // already emitted
                if (ctx.value_name.count(v->id())) continue;
                const auto& t = constants.at(v->id());
                (void)emit_tensor_constant(body, ctx, v->id(), t, v->shape(),
                                           v->dtype());
            }
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
            case OpType::Sin:          handle_unary(ctx, *node, body, "sine");        break;
            case OpType::Cos:          handle_unary(ctx, *node, body, "cosine");      break;
            case OpType::Rsqrt:        handle_unary(ctx, *node, body, "rsqrt");       break;

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
            // SparseMatMul is a SparsePass retag of Linear (inputs [x, weight,
            // bias]) and is numerically identical (spmm(from_dense(W),xT)T + b =
            // x@WT + b). MLIR has no sparse kernel, so lower it as the equivalent
            // dense Linear instead of hitting the default: throw (JIT-025).
            case OpType::SparseMatMul: handle_linear(ctx, *node, body); break;

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
            // Plugin path: emit `stablehlo.custom_call @tenzor_<x>`.
            // Expand path (plugin_enabled_ == false): decompose to pure
            // StableHLO primitives.
            case OpType::FlashAttention:
                if (plugin_enabled_) {
                    handle_flash_attention_custom_call(ctx, *node, body);
                } else {
                    handle_flash_attention_expand(ctx, *node, body);
                }
                break;
            case OpType::GQA:
                if (plugin_enabled_) {
                    handle_gqa_custom_call(ctx, *node, body);
                } else {
                    handle_gqa_expand(ctx, *node, body);
                }
                break;
            case OpType::RoPE:
                if (plugin_enabled_) {
                    handle_rope_apply_custom_call(ctx, *node, body);
                } else {
                    handle_rope_apply_expand(ctx, *node, body);
                }
                break;
            case OpType::RMSNorm:
                if (plugin_enabled_) {
                    handle_rms_norm_custom_call(ctx, *node, body);
                } else {
                    handle_rms_norm_expand(ctx, *node, body);
                }
                break;

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

    return emit_module_wrapper(body, inputs, outputs, return_names,
                               ctx.extern_decls);
}

}  // namespace tenzor::jit::mlir_jit
