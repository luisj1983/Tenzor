// Phase 13 / Group B–C — Graph → MLIR text lowering implementation.

#include "tenzor/jit/mlir/lowering.hpp"

#include "tenzor/jit/graph.hpp"
#include "tenzor/jit/mlir/mlir_text_emit.hpp"
#include "tenzor/jit/tracer.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
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
    // JIT-R006: Bool must be checked BEFORE is_int_dtype(d) — is_int_dtype
    // deliberately includes DType::Bool (other call sites in this file rely
    // on that for integer-like buffer/type handling), which meant this
    // branch order previously routed every Bool value into the plain
    // integer branch below, emitting MLIR-invalid `0`/`1` instead of
    // `true`/`false` for a scalar i1 constant — the dedicated Bool branch
    // was permanently unreachable. Mirrors emit_tensor_constant's
    // already-correct multi-element Bool handling.
    if (d == DType::Bool) {
        // MLIR dense<> i1 splat constants use 'true'/'false'.
        return value != 0.0 ? "true" : "false";
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


// Like get_attr_float but returns the attribute at full double precision (node
// attrs are stored as double). Use this where a value emitted into an f64 graph
// must not be truncated through float (e.g. a fractional pow exponent).
auto get_attr_double(const ::tenzor::jit::Node& n,
                     std::initializer_list<const char*> names,
                     double default_value) -> double {
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
            // JIT-R022: the else branch below used to assume any non-4-byte T
            // was 8-byte double, memcpy'ing a DOUBLE PROMOTION of v (losing
            // T's own actual bits) with a 16-hex-digit width -- wrong for a
            // genuine 2-byte T (Float16/BFloat16). No current call site hits
            // this (F16/BF16 non-finite scalars route through
            // emit_float_literal instead), but a future reuse of render_one
            // for a half-precision T would silently reproduce this. Handle
            // sizeof(T)==2 explicitly, memcpy'ing v's own bits like the
            // sizeof(T)==4 case does, not a converted double's.
            if constexpr (sizeof(T) == 4) {
                std::uint32_t bits;
                std::memcpy(&bits, &v, sizeof(bits));
                os << "0x" << std::hex << std::uppercase
                   << std::setfill('0') << std::setw(8) << bits
                   << std::dec << std::nouppercase << std::setfill(' ');
            } else if constexpr (sizeof(T) == 2) {
                std::uint16_t bits;
                std::memcpy(&bits, &v, sizeof(bits));
                os << "0x" << std::hex << std::uppercase
                   << std::setfill('0') << std::setw(4) << bits
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
        // Materialize a CONTIGUOUS CPU copy: the payload walk below indexes the
        // storage by row-major flat index, which reads permuted/garbage values
        // for a non-contiguous captured constant (a transposed/sliced view closed
        // over by the JIT'd lambda) — JIT-F007. .contiguous() is a no-op when the
        // tensor is already contiguous.
        auto ccpu = (t.device().type == ::tenzor::Device::Type::CPU ? t : t.cpu())
                        .contiguous();
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
    // Materialize a CONTIGUOUS CPU copy before the row-major flat-index walk
    // below: a non-contiguous captured constant would otherwise be baked with
    // permuted/garbage element values (JIT-F007). .contiguous() is a no-op when
    // the tensor is already contiguous.
    // Materialize a CONTIGUOUS CPU copy before the row-major flat-index walk
    // below: a non-contiguous captured constant would otherwise be baked with
    // permuted/garbage element values (JIT-F007). .contiguous() is a no-op when
    // the tensor is already contiguous.
    auto cpu = (t.device().type == ::tenzor::Device::Type::CPU ? t : t.cpu())
                   .contiguous();

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
    const auto& in_val = node.inputs()[0];
    const auto& a = ctx.name_for(in_val->id());
    const auto& out = node.outputs()[0];
    const auto d = out->dtype();
    auto out_name = ctx.fresh_name();
    ctx.bind(out->id(), out_name);

    // Transcendental / approximated ops are computed by the eager kernels in F32
    // (widen -> compute -> narrow). Match that for F16/BF16 so the JIT does not
    // diverge from eager, and so f16 device-math differences across HAL targets
    // (llvm-cpu vs cuda vs rocm vs vulkan-spirv) do not surface. Exact ops
    // (negate/abs) are bit-identical in half precision and need no widening.
    static const std::unordered_set<std::string> kNeedsWiden = {
        "exponential", "log", "sqrt", "rsqrt", "sine", "cosine", "logistic"};
    const bool widen = (d == ::tenzor::DType::Float16 ||
                        d == ::tenzor::DType::BFloat16) &&
                       kNeedsWiden.count(mnemonic) > 0;

    if (!widen) {
        emit_stablehlo_unary(body, mnemonic, out_name, a, out->shape(), d);
        body << '\n';
        return;
    }

    const auto cd = ::tenzor::DType::Float32;
    auto xc = ctx.fresh_name();
    emit_stablehlo_convert(body, xc, a, in_val->shape(), d, cd);
    body << '\n';
    auto rc = ctx.fresh_name();
    emit_stablehlo_unary(body, mnemonic, rc, xc, out->shape(), cd);
    body << '\n';
    emit_stablehlo_convert(body, out_name, rc, out->shape(), cd, d);
    body << '\n';
}

// Comparison ops (Eq/Ne/Lt/Le/Gt/Ge) are traced (opid_to_optype maps them) so
// data-dependent control-flow predicates and things like
// tenzor::float_power's negative-base branch are real graph nodes instead of
// trace-time-frozen constants -- but had no MLIR lowering at all, so any
// traced graph reaching one always failed GraphToMLIR ("OpType ... not yet
// supported") and silently degraded the WHOLE compiled function to eager.
// Unlike handle_binary, the comparison itself must run at the OPERAND dtype
// (e.g. Float32), not out->dtype() (always Bool) -- only the result is Bool.
auto handle_compare(LoweringContext& ctx,
                    const ::tenzor::jit::Node& node,
                    std::ostream& body,
                    const char* direction) -> void {
    if (node.inputs().size() != 2) {
        throw std::runtime_error(
            std::string("GraphToMLIR: compare(") + direction + ") expects 2 inputs, got " +
            std::to_string(node.inputs().size()));
    }
    if (node.outputs().size() != 1) {
        throw std::runtime_error(
            std::string("GraphToMLIR: compare(") + direction + ") expects 1 output");
    }
    const auto& out = node.outputs()[0];
    const auto& a_val = node.inputs()[0];
    const auto& b_val = node.inputs()[1];
    // The eager comparison ops promote both operands to a common dtype before
    // dispatching, so a_val/b_val should already match; convert_if is a
    // defensive no-op in the expected case.
    const auto operand_dtype = a_val->dtype();
    auto convert_if = [&](const std::string& name,
                          const std::shared_ptr<::tenzor::jit::Value>& val) -> std::string {
        if (val->dtype() == operand_dtype) return name;
        auto c = ctx.fresh_name();
        emit_stablehlo_convert(body, c, name, val->shape(), val->dtype(), operand_dtype);
        return c;
    };
    auto a = maybe_broadcast(body, ctx, convert_if(ctx.name_for(a_val->id()), a_val),
                             a_val->shape(), out->shape(), operand_dtype);
    auto b = maybe_broadcast(body, ctx, convert_if(ctx.name_for(b_val->id()), b_val),
                             b_val->shape(), out->shape(), operand_dtype);
    auto out_name = ctx.fresh_name();
    ctx.bind(out->id(), out_name);
    body << '%' << out_name << " = stablehlo.compare " << direction << ", %" << a << ", %" << b
         << " : (";
    write_tensor_type_for_emit(body, out->shape(), operand_dtype);
    body << ", ";
    write_tensor_type_for_emit(body, out->shape(), operand_dtype);
    body << ") -> ";
    write_tensor_type_for_emit(body, out->shape(), ::tenzor::DType::Bool);
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

// Emit ReLU on a named SSA value (max(x, 0)) in dtype d, returning the result
// name (not bound). Shared by handle_relu and emit_fused_epilogue.
auto emit_relu_value(LoweringContext& ctx, std::ostream& body,
                     const std::string& x,
                     const std::vector<int64_t>& shape,
                     ::tenzor::DType d) -> std::string {
    auto zero_name = ctx.fresh_name();
    emit_stablehlo_splat_constant(body, zero_name, scalar_literal(0.0, d),
                                  shape, d);
    body << '\n';
    auto out_name = ctx.fresh_name();
    emit_stablehlo_binary(body, "maximum", out_name, x, zero_name, shape, d);
    body << '\n';
    return out_name;
}

// Emit exact erf-GELU on a named SSA value in dtype d, widening F16/BF16 to F32
// for the whole pipeline (matching eager and handle_gelu), returning the result
// name (not bound). Shared by handle_gelu and emit_fused_epilogue.
auto emit_gelu_value(LoweringContext& ctx, std::ostream& body,
                     const std::string& in_name,
                     const std::vector<int64_t>& shape,
                     ::tenzor::DType d) -> std::string {
    const bool widen = (d == ::tenzor::DType::Float16 ||
                        d == ::tenzor::DType::BFloat16);
    const auto cd = widen ? ::tenzor::DType::Float32 : d;
    std::string x = in_name;
    if (widen) {
        auto xc = ctx.fresh_name();
        emit_stablehlo_convert(body, xc, x, shape, d, cd);
        body << '\n';
        x = xc;
    }
    auto c_half     = ctx.fresh_name();
    auto c_one      = ctx.fresh_name();
    auto c_invsqrt2 = ctx.fresh_name();
    emit_stablehlo_splat_constant(body, c_half, scalar_literal(0.5, cd),
                                  shape, cd); body << '\n';
    emit_stablehlo_splat_constant(body, c_one,  scalar_literal(1.0, cd),
                                  shape, cd); body << '\n';
    emit_stablehlo_splat_constant(body, c_invsqrt2,
                                  scalar_literal(0.7071067811865476, cd),
                                  shape, cd); body << '\n';
    auto scaled = ctx.fresh_name();
    emit_stablehlo_binary(body, "multiply", scaled, x, c_invsqrt2, shape, cd);
    body << '\n';
    auto eval = ctx.fresh_name();
    emit_chlo_unary(body, "erf", eval, scaled, shape, cd);
    body << '\n';
    auto one_plus = ctx.fresh_name();
    emit_stablehlo_binary(body, "add", one_plus, c_one, eval, shape, cd);
    body << '\n';
    auto half_x = ctx.fresh_name();
    emit_stablehlo_binary(body, "multiply", half_x, c_half, x, shape, cd);
    body << '\n';
    auto prod = ctx.fresh_name();
    emit_stablehlo_binary(body, "multiply", prod, half_x, one_plus, shape, cd);
    body << '\n';
    if (widen) {
        auto out_name = ctx.fresh_name();
        emit_stablehlo_convert(body, out_name, prod, shape, cd, d);
        body << '\n';
        return out_name;
    }
    return prod;
}

// Apply the fused epilogue markers that optimize_for_inference bakes onto a
// producer node when it deletes the following ReLU / activation node
// (fused_relu; fused_activation with fused_activation_type 1=relu else gelu).
// This mirrors the interpreter's execute_node (graph.cpp) exactly. Without it
// the MLIR/IREE path silently drops the epilogue — Conv+ReLU, Linear+ReLU,
// LayerNorm+activation run with the activation missing on every IREE target,
// diverging from eager and the nvrtc backend (JIT-F013). `name` is the op result
// in storage dtype d. (MatMul's fused_bias is a 3rd input, added in handle_matmul
// before this call.)
auto emit_fused_epilogue(LoweringContext& ctx, const ::tenzor::jit::Node& node,
                         std::ostream& body, std::string name,
                         const std::vector<int64_t>& shape,
                         ::tenzor::DType d) -> std::string {
    if (node.get_bool_attr("fused_relu")) {
        name = emit_relu_value(ctx, body, name, shape, d);
    }
    if (node.get_bool_attr("fused_activation")) {
        const int64_t act = node.get_int_attr("fused_activation_type");
        name = (act == 1) ? emit_relu_value(ctx, body, name, shape, d)
                          : emit_gelu_value(ctx, body, name, shape, d);
    }
    return name;
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
    auto out_name = emit_relu_value(ctx, body, x, shape, d);
    ctx.bind(out->id(), out_name);
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
    // Match eager: compute the logistic (and the multiply) in F32 for F16/BF16,
    // then narrow. A half-precision logistic diverges from eager and across HAL
    // targets (f16 logistic lowering is target-specific).
    const bool widen = (d == ::tenzor::DType::Float16 ||
                        d == ::tenzor::DType::BFloat16);
    const auto cd = widen ? ::tenzor::DType::Float32 : d;
    std::string x = ctx.name_for(node.inputs()[0]->id());
    if (widen) {
        auto xc = ctx.fresh_name();
        emit_stablehlo_convert(body, xc, x, shape, d, cd);
        body << '\n';
        x = xc;
    }
    auto sig_name = ctx.fresh_name();
    emit_stablehlo_unary(body, "logistic", sig_name, x, shape, cd);
    body << '\n';
    auto out_name = ctx.fresh_name();
    ctx.bind(out->id(), out_name);
    if (widen) {
        auto prod = ctx.fresh_name();
        emit_stablehlo_binary(body, "multiply", prod, x, sig_name, shape, cd);
        body << '\n';
        emit_stablehlo_convert(body, out_name, prod, shape, cd, d);
        body << '\n';
    } else {
        emit_stablehlo_binary(body, "multiply", out_name, x, sig_name, shape, d);
        body << '\n';
    }
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
    auto out_name = emit_gelu_value(ctx, body, x, shape, d);
    ctx.bind(out->id(), out_name);
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

// Emit a dot_general that accumulates F16/BF16 operands in F32, matching the
// eager GEMM kernels (matmul_blocked_float16 and the BF16 path accumulate half
// products in float and narrow only on store). For F32/F64 this is a plain
// dot_general in the result dtype. IREE accumulates a dot_general in its RESULT
// element type, so passing f16/bf16 straight through would accumulate in half
// precision and diverge from eager by a relative error that grows with the
// contraction length K (JIT-F032). `result` is written but not bound — the
// caller supplies (and binds, if needed) the name.
auto emit_dot_general_widened(
    std::ostream& os, LoweringContext& ctx, const std::string& result,
    std::string a, std::string b,
    const std::vector<int64_t>& lhs_batch,
    const std::vector<int64_t>& rhs_batch,
    const std::vector<int64_t>& lhs_contracting,
    const std::vector<int64_t>& rhs_contracting,
    const std::vector<int64_t>& lhs_shape,
    const std::vector<int64_t>& rhs_shape,
    const std::vector<int64_t>& result_shape,
    ::tenzor::DType lhs_dt, ::tenzor::DType rhs_dt,
    ::tenzor::DType result_dt) -> void {
    const bool widen = (result_dt == ::tenzor::DType::Float16 ||
                        result_dt == ::tenzor::DType::BFloat16);
    if (!widen) {
        emit_stablehlo_dot_general(os, result, a, b, lhs_batch, rhs_batch,
                                   lhs_contracting, rhs_contracting, lhs_shape,
                                   rhs_shape, result_shape, result_dt);
        os << '\n';
        return;
    }
    const auto cd = ::tenzor::DType::Float32;
    if (lhs_dt != cd) {
        auto ac = ctx.fresh_name();
        emit_stablehlo_convert(os, ac, a, lhs_shape, lhs_dt, cd);
        os << '\n';
        a = ac;
    }
    if (rhs_dt != cd) {
        auto bc = ctx.fresh_name();
        emit_stablehlo_convert(os, bc, b, rhs_shape, rhs_dt, cd);
        os << '\n';
        b = bc;
    }
    auto acc = ctx.fresh_name();
    emit_stablehlo_dot_general(os, acc, a, b, lhs_batch, rhs_batch,
                               lhs_contracting, rhs_contracting, lhs_shape,
                               rhs_shape, result_shape, cd);
    os << '\n';
    emit_stablehlo_convert(os, result, acc, result_shape, cd, result_dt);
    os << '\n';
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
    if ((node.inputs().size() != 2 && node.inputs().size() != 3) ||
        node.outputs().size() != 1) {
        throw std::runtime_error(
            "GraphToMLIR: MatMul expects 2 or 3 inputs (a, b [, fused bias]) "
            "and 1 output");
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

    auto mm_name = ctx.fresh_name();
    emit_dot_general_widened(body, ctx, mm_name, lhs_name, rhs_name,
                             lhs_batch, rhs_batch,
                             lhs_contracting, rhs_contracting,
                             lhs_target, rhs_target, out->shape(),
                             lhs->dtype(), rhs->dtype(), out->dtype());
    // FuseMatMulAddPass folds a following bias-add into this node and adds the
    // bias as a 3rd input (interpreter: mm_out + input[2]). Emit it, then any
    // fused ReLU, so the epilogue is not dropped on the MLIR path (JIT-F013).
    std::string result = mm_name;
    if (node.inputs().size() == 3) {
        const auto& b = node.inputs()[2];
        auto bias_b = maybe_broadcast(body, ctx, ctx.name_for(b->id()),
                                      b->shape(), out->shape(), out->dtype());
        auto added = ctx.fresh_name();
        emit_stablehlo_binary(body, "add", added, result, bias_b, out->shape(),
                              out->dtype());
        body << '\n';
        result = added;
    }
    result = emit_fused_epilogue(ctx, node, body, result, out->shape(),
                                 out->dtype());
    ctx.bind(out->id(), result);
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
    emit_dot_general_widened(body, ctx, out_name, ctx.name_for(lhs->id()),
                             ctx.name_for(rhs->id()),
                             /*lhs_batch=*/{0}, /*rhs_batch=*/{0},
                             /*lhs_contracting=*/{2},
                             /*rhs_contracting=*/{1},
                             lhs->shape(), rhs->shape(), out->shape(),
                             lhs->dtype(), rhs->dtype(), out->dtype());
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
    emit_dot_general_widened(body, ctx, matmul_name, ctx.name_for(x->id()),
                             ctx.name_for(w->id()),
                             /*lhs_batch=*/{}, /*rhs_batch=*/{},
                             lhs_contracting, rhs_contracting, x_shape,
                             w_shape, out_shape, x->dtype(), w->dtype(), d);

    std::string result = matmul_name;
    if (node.inputs().size() == 3) {
        const auto& b = node.inputs()[2];
        auto bias_b = maybe_broadcast(body, ctx, ctx.name_for(b->id()),
                                      b->shape(), out_shape, d);
        auto added = ctx.fresh_name();
        emit_stablehlo_binary(body, "add", added, matmul_name, bias_b,
                              out_shape, d);
        body << '\n';
        result = added;
    }
    // Apply any fused ReLU folded into this Linear node (JIT-F013).
    result = emit_fused_epilogue(ctx, node, body, result, out_shape, d);
    ctx.bind(out->id(), result);
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
        const int64_t step = get_attr_int(node, {"step"}, 1);
        if (step <= 0) {
            // Mirror the vector-form guard (JIT-F008): stablehlo.slice has no
            // negative/zero stride, so a reversed/degenerate step must give the
            // same clean early diagnostic here instead of a late iree-compile
            // failure.
            throw std::runtime_error(
                "GraphToMLIR: Slice requires a positive step (got " +
                std::to_string(step) + "); negative-step slices are not "
                "representable as stablehlo.slice.");
        }
        strides[dim] = step;
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
    const auto out_dtype = out->dtype();
    // Compute the convolution in F32 for F16/BF16 (like the pooling/norm
    // handlers): a half-precision accumulate diverges from eager. `cd` is the
    // compute dtype; narrow back to out_dtype at the end when widened.
    const bool widen = (out_dtype == ::tenzor::DType::Float16 ||
                        out_dtype == ::tenzor::DType::BFloat16);
    const auto d = widen ? ::tenzor::DType::Float32 : out_dtype;

    // stablehlo.convolution needs an explicit HIGHEST precision_config or GPU HAL
    // targets (CUDA tensor cores / ROCm MFMA) silently use TF32/reduced-precision
    // accumulation, diverging from eager and from the llvm-cpu target. Every
    // dot_general in this file forces HIGHEST for the same reason.
    static constexpr const char* kConvPrecision =
        ", precision_config = [#stablehlo<precision HIGHEST>, "
        "#stablehlo<precision HIGHEST>]";

    // Widen the input/weight when needed; all emits below use `d` (== cd).
    std::string x_name = ctx.name_for(x->id());
    std::string w_name = ctx.name_for(w->id());
    if (widen) {
        auto xc = ctx.fresh_name();
        emit_stablehlo_convert(body, xc, x_name, x->shape(), out_dtype, d);
        body << '\n';
        x_name = xc;
        auto wc = ctx.fresh_name();
        emit_stablehlo_convert(body, wc, w_name, w->shape(), out_dtype, d);
        body << '\n';
        w_name = wc;
    }

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

    auto conv_name = ctx.fresh_name();

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
    if (xs.size() == 4 && ws.size() == 4 && os.size() == 4) {
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
            std::string cur = x_name;
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

            if (Hout == 1 && Wout == 1 && groups == 1) {
                // Single receptive window -> im2col: slice the window, flatten,
                // and matmul against the flattened kernel. Only for groups==1:
                // the flatten K = Cin*kh*kw assumes an ungrouped weight
                // [Cout, Cin, kh, kw]; a grouped weight is [Cout, Cin/groups,
                // kh, kw] and would mis-flatten, so grouped convs take the
                // convolution path below with feature_group_count=groups (F033).
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
                emit_stablehlo_reshape(body, w_flat, w_name, ws,
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
                     << ", %" << w_name << ")\n"
                     << "    dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, "
                        "1],\n"
                     << "    window = {stride = [" << stride_h << ", " << stride_w
                     << "], pad = [[0, 0], [0, 0]], rhs_dilate = [" << dilation_h
                     << ", " << dilation_w
                     << "]} {batch_group_count = 1 : i64, "
                        "feature_group_count = " << groups << " : i64"
                     << kConvPrecision << "} : (";
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
             << x_name << ", %" << w_name << ")\n"
             << "    dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],\n"
             << "    window = {stride = [" << stride_h << ", " << stride_w
             << "], pad = [[" << pad_h << ", " << pad_h << "], ["
             << pad_w << ", " << pad_w << "]], rhs_dilate = ["
             << dilation_h << ", " << dilation_w << "]} "
             << "{batch_group_count = 1 : i64, feature_group_count = "
             << groups << " : i64" << kConvPrecision << "} : (";
        write_tensor_type_for_emit(body, x->shape(), d);
        body << ", ";
        write_tensor_type_for_emit(body, w->shape(), d);
        body << ") -> ";
        write_tensor_type_for_emit(body, out->shape(), d);
        body << '\n';
    }

    // Bias add (in the compute dtype), then narrow to the storage dtype.
    std::string result = conv_name;
    if (node.inputs().size() >= 3) {
        const auto& b_val = node.inputs()[2];
        std::string b_name = ctx.name_for(b_val->id());
        if (widen) {
            auto bc = ctx.fresh_name();
            emit_stablehlo_convert(body, bc, b_name, b_val->shape(), out_dtype, d);
            body << '\n';
            b_name = bc;
        }
        // Bias: (C_out,) broadcast over (N, C_out, H_out, W_out) — dim 1.
        auto bias_b = ctx.fresh_name();
        emit_stablehlo_broadcast_in_dim(body, bias_b, b_name,
                                        /*bcast_dims=*/{1}, b_val->shape(),
                                        out->shape(), d);
        body << '\n';
        auto added = ctx.fresh_name();
        emit_stablehlo_binary(body, "add", added, conv_name, bias_b,
                              out->shape(), d);
        body << '\n';
        result = added;
    }

    // Apply any fused ReLU folded into this Conv node (JIT-F013). ReLU is exact,
    // so applying it in the compute dtype d before the narrow matches eager
    // (relu commutes with the monotonic narrowing).
    result = emit_fused_epilogue(ctx, node, body, result, out->shape(), d);

    if (widen) {
        auto out_name = ctx.fresh_name();
        ctx.bind(out->id(), out_name);
        emit_stablehlo_convert(body, out_name, result, out->shape(), d, out_dtype);
        body << '\n';
    } else {
        ctx.bind(out->id(), result);
    }
}

// ============================================================================
// Quantization (JIT-R081): Quantize/Dequantize/QuantizedLinear/
// QuantizedLinearStatic/QuantizedConv2d lowering. Formulas mirror
// include/tenzor/nn/quantization/quantize.hpp / quantize.cpp EXACTLY:
//   quantized   = clamp(round_nearest_even(value/scale) + zero_point, qmin, qmax)
//   dequantized = (quantized - zero_point) * scale
// Symmetric dynamic scale (matches compute_symmetric_scale, quantize.cpp):
//   scale = max(max(|value|), 1e-8) / 127   (zero_point = 0, INT8 range)
// ============================================================================

// Reduce max(|value|) over ALL dims to a rank-0 scalar SSA value.
auto emit_reduce_max_abs_scalar(LoweringContext& ctx, std::ostream& body,
                                const std::string& value_name,
                                const std::vector<int64_t>& shape,
                                ::tenzor::DType d) -> std::string {
    auto abs_name = ctx.fresh_name();
    emit_stablehlo_unary(body, "abs", abs_name, value_name, shape, d);
    body << '\n';
    std::vector<int64_t> all_dims;
    for (int64_t i = 0; i < static_cast<int64_t>(shape.size()); ++i) all_dims.push_back(i);
    std::string init_lit;
    if (d == ::tenzor::DType::Float64) init_lit = "0xFFF0000000000000";
    else if (d == ::tenzor::DType::Float16) init_lit = "0xFC00";
    else if (d == ::tenzor::DType::BFloat16) init_lit = "0xFF80";
    else init_lit = "0xFF800000";
    return emit_reduce_with_keepdim(ctx, body, abs_name, shape, /*result_shape=*/{},
                                    all_dims, /*keepdim=*/false, "maximum", init_lit, d);
}

// Symmetric dynamic INT8 fake-quantize (quantize then immediately
// dequantize), matching quantize_per_tensor_symmetric + dequantize_tensor
// exactly — the path nn::quantization::quantized_linear_dynamic /
// quantized_conv2d_dynamic use internally. Returns the SSA name of the
// round-tripped float tensor (same shape/dtype as the input).
auto emit_dynamic_symmetric_fake_quant(LoweringContext& ctx, std::ostream& body,
                                       const std::string& value_name,
                                       const std::vector<int64_t>& shape,
                                       ::tenzor::DType d) -> std::string {
    // JIT-R114: widen F16/BF16 to F32 for the divide/round_nearest_even/clamp/
    // multiply pipeline, matching eager's quantize_per_tensor_symmetric/
    // dequantize_tensor (src/nn/quantization/quantize.cpp), which always
    // compute this exact round-trip in Float32 regardless of the tensor's
    // storage dtype -- unlike every other numeric pipeline in this file
    // (softmax, layer/RMS/batch norm, conv2d, pooling, pow, attention), this
    // one ran entirely in F16/BF16 storage precision. round_nearest_even is a
    // discrete rounding-BOUNDARY decision (which int8 level a value snaps
    // to), not a smooth elementwise op, so F16-precision loss before
    // rounding can flip the resulting quantization level versus the F32
    // reference eager computes -- a more consequential divergence than
    // typical ULP-scale drift.
    const bool widen = (d == ::tenzor::DType::Float16 ||
                        d == ::tenzor::DType::BFloat16);
    const auto cd = widen ? ::tenzor::DType::Float32 : d;
    std::string v = value_name;
    if (widen) {
        auto vc = ctx.fresh_name();
        emit_stablehlo_convert(body, vc, value_name, shape, d, cd);
        body << '\n';
        v = vc;
    }

    auto abs_max = emit_reduce_max_abs_scalar(ctx, body, v, shape, cd);
    // scale = max(abs_max, 1e-8) / 127
    auto eps_name = ctx.fresh_name();
    emit_stablehlo_splat_constant(body, eps_name, scalar_literal(1e-8, cd), {}, cd);
    body << '\n';
    auto safe_max = ctx.fresh_name();
    emit_stablehlo_binary(body, "maximum", safe_max, abs_max, eps_name, {}, cd);
    body << '\n';
    auto range_name = ctx.fresh_name();
    emit_stablehlo_splat_constant(body, range_name, scalar_literal(127.0, cd), {}, cd);
    body << '\n';
    auto scale_scalar = ctx.fresh_name();
    emit_stablehlo_binary(body, "divide", scale_scalar, safe_max, range_name, {}, cd);
    body << '\n';
    auto scale_b = maybe_broadcast(body, ctx, scale_scalar, {}, shape, cd);

    auto ratio = ctx.fresh_name();
    emit_stablehlo_binary(body, "divide", ratio, v, scale_b, shape, cd);
    body << '\n';
    auto rounded = ctx.fresh_name();
    emit_stablehlo_unary(body, "round_nearest_even", rounded, ratio, shape, cd);
    body << '\n';
    auto lo_name = ctx.fresh_name();
    auto hi_name = ctx.fresh_name();
    emit_stablehlo_splat_constant(body, lo_name, scalar_literal(-128.0, cd), shape, cd);
    body << '\n';
    emit_stablehlo_splat_constant(body, hi_name, scalar_literal(127.0, cd), shape, cd);
    body << '\n';
    auto clamped = ctx.fresh_name();
    emit_stablehlo_ternary(body, "clamp", clamped, lo_name, rounded, hi_name, shape, cd);
    body << '\n';
    // Cast to Int8 and back: a true saturating round-trip through the same
    // storage dtype the eager kernel materializes, not just a float rounding
    // approximation.
    auto q_int8 = ctx.fresh_name();
    emit_stablehlo_convert(body, q_int8, clamped, shape, cd, ::tenzor::DType::Int8);
    body << '\n';
    auto dq_float = ctx.fresh_name();
    emit_stablehlo_convert(body, dq_float, q_int8, shape, ::tenzor::DType::Int8, cd);
    body << '\n';
    auto result_cd = ctx.fresh_name();
    emit_stablehlo_binary(body, "multiply", result_cd, dq_float, scale_b, shape, cd);
    body << '\n';
    if (widen) {
        auto result_d = ctx.fresh_name();
        emit_stablehlo_convert(body, result_d, result_cd, shape, cd, d);
        body << '\n';
        return result_d;
    }
    return result_cd;
}

// Static dequantize: (convert(value,out_dtype) - zero_point) * scale, with
// scale/zero_point known at lowering time (compile-time node attrs).
auto emit_static_dequant(LoweringContext& ctx, std::ostream& body,
                         const std::string& value_name,
                         const std::vector<int64_t>& shape,
                         ::tenzor::DType in_dtype, double scale, int64_t zero_point,
                         ::tenzor::DType out_dtype) -> std::string {
    // JIT-R114: see emit_dynamic_symmetric_fake_quant's identical rationale --
    // widen an F16/BF16 out_dtype to F32 for the subtract/multiply, matching
    // eager's dequantize_tensor (always Float32 scratch regardless of the
    // tensor's original dtype), then narrow back at the end.
    const bool widen = (out_dtype == ::tenzor::DType::Float16 ||
                        out_dtype == ::tenzor::DType::BFloat16);
    const auto cd = widen ? ::tenzor::DType::Float32 : out_dtype;
    std::string v = value_name;
    if (in_dtype != cd) {
        auto c = ctx.fresh_name();
        emit_stablehlo_convert(body, c, v, shape, in_dtype, cd);
        body << '\n';
        v = c;
    }
    std::string shifted = v;
    if (zero_point != 0) {
        auto zp_name = ctx.fresh_name();
        emit_stablehlo_splat_constant(
            body, zp_name, scalar_literal(static_cast<double>(zero_point), cd),
            shape, cd);
        body << '\n';
        auto sub_name = ctx.fresh_name();
        emit_stablehlo_binary(body, "subtract", sub_name, v, zp_name, shape, cd);
        body << '\n';
        shifted = sub_name;
    }
    auto scale_name = ctx.fresh_name();
    emit_stablehlo_splat_constant(body, scale_name, scalar_literal(scale, cd),
                                  shape, cd);
    body << '\n';
    auto result_cd = ctx.fresh_name();
    emit_stablehlo_binary(body, "multiply", result_cd, shifted, scale_name, shape, cd);
    body << '\n';
    if (widen) {
        auto result_out = ctx.fresh_name();
        emit_stablehlo_convert(body, result_out, result_cd, shape, cd, out_dtype);
        body << '\n';
        return result_out;
    }
    return result_cd;
}

// Static quantize: clamp(round_nearest_even(value/scale)+zero_point,
// qmin,qmax), cast to target_dtype.
auto emit_static_quantize(LoweringContext& ctx, std::ostream& body,
                          const std::string& value_name,
                          const std::vector<int64_t>& shape,
                          ::tenzor::DType in_dtype, double scale, int64_t zero_point,
                          ::tenzor::DType target_dtype) -> std::string {
    // JIT-R114: see emit_dynamic_symmetric_fake_quant's identical rationale --
    // widen an F16/BF16 in_dtype to F32 for the divide/round_nearest_even/
    // clamp pipeline, matching eager's quantize_tensor/quantize_per_tensor_
    // symmetric (always Float32 regardless of the tensor's original dtype).
    const bool widen = (in_dtype == ::tenzor::DType::Float16 ||
                        in_dtype == ::tenzor::DType::BFloat16);
    const auto cd = widen ? ::tenzor::DType::Float32 : in_dtype;
    std::string v = value_name;
    if (widen) {
        auto vc = ctx.fresh_name();
        emit_stablehlo_convert(body, vc, value_name, shape, in_dtype, cd);
        body << '\n';
        v = vc;
    }
    auto scale_name = ctx.fresh_name();
    emit_stablehlo_splat_constant(body, scale_name, scalar_literal(scale, cd),
                                  shape, cd);
    body << '\n';
    auto ratio = ctx.fresh_name();
    emit_stablehlo_binary(body, "divide", ratio, v, scale_name, shape, cd);
    body << '\n';
    auto rounded = ctx.fresh_name();
    emit_stablehlo_unary(body, "round_nearest_even", rounded, ratio, shape, cd);
    body << '\n';
    std::string with_zp = rounded;
    if (zero_point != 0) {
        auto zp_name = ctx.fresh_name();
        emit_stablehlo_splat_constant(
            body, zp_name, scalar_literal(static_cast<double>(zero_point), cd),
            shape, cd);
        body << '\n';
        auto added = ctx.fresh_name();
        emit_stablehlo_binary(body, "add", added, rounded, zp_name, shape, cd);
        body << '\n';
        with_zp = added;
    }
    int64_t qmin = 0, qmax = 0;
    switch (target_dtype) {
        case ::tenzor::DType::Int8:  qmin = -128; qmax = 127; break;
        case ::tenzor::DType::UInt8: qmin = 0;    qmax = 255; break;
        default:
            throw std::runtime_error(
                "GraphToMLIR: Quantize target dtype must be Int8/UInt8");
    }
    auto lo_name = ctx.fresh_name();
    auto hi_name = ctx.fresh_name();
    emit_stablehlo_splat_constant(body, lo_name, scalar_literal(static_cast<double>(qmin), cd),
                                  shape, cd);
    body << '\n';
    emit_stablehlo_splat_constant(body, hi_name, scalar_literal(static_cast<double>(qmax), cd),
                                  shape, cd);
    body << '\n';
    auto clamped = ctx.fresh_name();
    emit_stablehlo_ternary(body, "clamp", clamped, lo_name, with_zp, hi_name, shape, cd);
    body << '\n';
    auto result = ctx.fresh_name();
    emit_stablehlo_convert(body, result, clamped, shape, cd, target_dtype);
    body << '\n';
    return result;
}

// Standalone Quantize/Dequantize (QuantStub/DeQuantStub-style single-tensor
// ops). NOTE: as of this fix, no producer anywhere in the codebase emits
// OpType::Quantize/Dequantize as a graph node (verified: zero references
// outside the enum/shape-inference/interpreter-throw sites) — these two
// handlers make GraphToMLIR ABLE to lower such a node using the "scale"/
// "zero_point" node-attr convention below (matching QuantizedLinearStatic's
// attr naming), but the path is currently unreachable via any tracer/pass.
// Verified directly via a hand-built Graph/Node in the new lowering test
// (bypassing the tracer, since there is no traced producer to exercise).
auto handle_dequantize(LoweringContext& ctx, const ::tenzor::jit::Node& node,
                       std::ostream& body) -> void {
    if (node.inputs().size() != 1 || node.outputs().size() != 1) {
        throw std::runtime_error("GraphToMLIR: Dequantize expects 1 input, 1 output");
    }
    const auto& in_val = node.inputs()[0];
    const auto& out = node.outputs()[0];
    double scale = node.has_float_attr("scale") ? node.get_attr("scale") : 1.0;
    int64_t zero_point = node.has_int_attr("zero_point") ? node.get_int_attr("zero_point") : 0;
    auto result = emit_static_dequant(ctx, body, ctx.name_for(in_val->id()), out->shape(),
                                      in_val->dtype(), scale, zero_point, out->dtype());
    ctx.bind(out->id(), result);
}

auto handle_quantize(LoweringContext& ctx, const ::tenzor::jit::Node& node,
                     std::ostream& body) -> void {
    if (node.inputs().size() != 1 || node.outputs().size() != 1) {
        throw std::runtime_error("GraphToMLIR: Quantize expects 1 input, 1 output");
    }
    const auto& in_val = node.inputs()[0];
    const auto& out = node.outputs()[0];
    double scale = node.has_float_attr("scale") ? node.get_attr("scale") : 1.0;
    int64_t zero_point = node.has_int_attr("zero_point") ? node.get_int_attr("zero_point") : 0;
    auto result = emit_static_quantize(ctx, body, ctx.name_for(in_val->id()), in_val->shape(),
                                       in_val->dtype(), scale, zero_point, out->dtype());
    ctx.bind(out->id(), result);
}

// QuantizedLinear/QuantizedConv2d (dynamic, retagged IN PLACE from Linear/
// Conv2d by QuantizationPass — see compiler.cpp quantize_linear/
// quantize_conv2d): inputs/attrs are UNCHANGED from the original float
// Linear/Conv2d node (no int8 Values exist in the graph at all — the
// "quantization" is a runtime simulation, matching
// nn::quantization::quantized_linear_dynamic/quantized_conv2d_dynamic,
// which is why forward_quantized's real output is float, see
// quantized_linear.cpp's combined_scale contract). Fake-quantize x and
// weight symmetrically, then replay the ORIGINAL handle_linear/handle_conv2d
// on the round-tripped operands so fused-bias/epilogue handling stays
// exactly consistent with the plain float path. ctx.bind() uses
// map::emplace (insert-only, no overwrite) so a direct ctx.value_name[]
// write + restore is used to temporarily substitute the fake-quantized
// operand for the node's real x/weight inputs.
auto handle_quantized_linear_dynamic(LoweringContext& ctx,
                                     const ::tenzor::jit::Node& node,
                                     std::ostream& body) -> void {
    if (node.inputs().size() < 2) {
        throw std::runtime_error("GraphToMLIR: QuantizedLinear expects [x, weight(, bias)]");
    }
    const auto& x = node.inputs()[0];
    const auto& w = node.inputs()[1];
    auto fq_x = emit_dynamic_symmetric_fake_quant(ctx, body, ctx.name_for(x->id()),
                                                   x->shape(), x->dtype());
    auto fq_w = emit_dynamic_symmetric_fake_quant(ctx, body, ctx.name_for(w->id()),
                                                   w->shape(), w->dtype());
    auto saved_x = ctx.name_for(x->id());
    auto saved_w = ctx.name_for(w->id());
    ctx.value_name[x->id()] = fq_x;
    ctx.value_name[w->id()] = fq_w;
    handle_linear(ctx, node, body);
    ctx.value_name[x->id()] = saved_x;
    ctx.value_name[w->id()] = saved_w;
}

auto handle_quantized_conv2d_dynamic(LoweringContext& ctx,
                                     const ::tenzor::jit::Node& node,
                                     std::ostream& body) -> void {
    if (node.inputs().size() < 2) {
        throw std::runtime_error("GraphToMLIR: QuantizedConv2d expects [x, weight(, bias)]");
    }
    const auto& x = node.inputs()[0];
    const auto& w = node.inputs()[1];
    auto fq_x = emit_dynamic_symmetric_fake_quant(ctx, body, ctx.name_for(x->id()),
                                                   x->shape(), x->dtype());
    auto fq_w = emit_dynamic_symmetric_fake_quant(ctx, body, ctx.name_for(w->id()),
                                                   w->shape(), w->dtype());
    auto saved_x = ctx.name_for(x->id());
    auto saved_w = ctx.name_for(w->id());
    ctx.value_name[x->id()] = fq_x;
    ctx.value_name[w->id()] = fq_w;
    handle_conv2d(ctx, node, body);
    ctx.value_name[x->id()] = saved_x;
    ctx.value_name[w->id()] = saved_w;
}

// QuantizedLinearStatic (see nn::quantization::QuantizedLinear::forward_impl
// -> jit_record_quantized_linear_static / graph.cpp's execute_node case):
// inputs [input(float), weight(int8), bias (, weight_scale, weight_zp)].
// Attrs input_scale/weight_scale/output_scale/input_zero_point/
// weight_zero_point mirror the interpreter's case exactly. Output contract
// (quantized_linear_kernel, backends/cpu/kernels/quantization/
// quantized_linear.cpp):
//   output = (dequant(x) @ dequant(w)^T) / output_scale + bias
// (combined_scale = input_scale*weight_scale/output_scale; dividing the
// dequantized-domain dot product by output_scale is the exact float-domain
// equivalent since dequant(x)@dequant(w)^T = input_scale*weight_scale*
// raw_int_dot). Deliberately does NOT reuse handle_linear's fused-bias-add
// (which would add bias BEFORE the /output_scale, an incorrect order).
auto handle_quantized_linear_static(LoweringContext& ctx,
                                    const ::tenzor::jit::Node& node,
                                    std::ostream& body) -> void {
    if (node.inputs().size() < 3) {
        throw std::runtime_error(
            "GraphToMLIR: QuantizedLinearStatic expects [input, weight, bias(, wscale, wzp)]");
    }
    const auto& x = node.inputs()[0];
    const auto& w = node.inputs()[1];
    const auto& b = node.inputs()[2];
    const auto& out = node.outputs()[0];
    const auto d = out->dtype();

    double input_scale = node.has_float_attr("input_scale") ? node.get_attr("input_scale") : 1.0;
    double weight_scale = node.has_float_attr("weight_scale") ? node.get_attr("weight_scale") : 1.0;
    double output_scale = node.has_float_attr("output_scale") ? node.get_attr("output_scale") : 1.0;
    int64_t input_zp = node.has_int_attr("input_zero_point")
        ? node.get_int_attr("input_zero_point") : int64_t{0};
    int64_t weight_zp = node.has_int_attr("weight_zero_point")
        ? node.get_int_attr("weight_zero_point") : int64_t{0};

    // input[0] is the RAW FLOAT activation (re-quantized+dequantized here,
    // matching the interpreter's re-quantize-on-replay JIT-R058b fix) —
    // fake-quantize it with the static input_scale/input_zero_point.
    auto x_name = ctx.name_for(x->id());
    auto x_quant = emit_static_quantize(ctx, body, x_name, x->shape(), x->dtype(),
                                        input_scale, input_zp, ::tenzor::DType::Int8);
    auto x_dequant = emit_static_dequant(ctx, body, x_quant, x->shape(),
                                         ::tenzor::DType::Int8, input_scale, input_zp, d);

    // weight is ALREADY int8 at this node (a trace-time quantized constant).
    auto w_dequant = emit_static_dequant(ctx, body, ctx.name_for(w->id()), w->shape(),
                                         w->dtype(), weight_scale, weight_zp, d);

    const auto x_shape = x->shape();
    const auto w_shape = w->shape();
    const int64_t xr = static_cast<int64_t>(x_shape.size());
    const int64_t wr = static_cast<int64_t>(w_shape.size());
    std::vector<int64_t> out_shape = out->shape();

    auto mm_name = ctx.fresh_name();
    emit_dot_general_widened(body, ctx, mm_name, x_dequant, w_dequant,
                             /*lhs_batch=*/{}, /*rhs_batch=*/{},
                             /*lhs_contracting=*/{xr - 1}, /*rhs_contracting=*/{wr - 1},
                             x_shape, w_shape, out_shape, d, d, d);

    std::string scaled = mm_name;
    if (output_scale != 1.0) {
        auto os_name = ctx.fresh_name();
        emit_stablehlo_splat_constant(body, os_name, scalar_literal(output_scale, d),
                                      out_shape, d);
        body << '\n';
        auto div_name = ctx.fresh_name();
        emit_stablehlo_binary(body, "divide", div_name, mm_name, os_name, out_shape, d);
        body << '\n';
        scaled = div_name;
    }

    auto bias_b = maybe_broadcast(body, ctx, ctx.name_for(b->id()), b->shape(), out_shape, d);
    auto result = ctx.fresh_name();
    emit_stablehlo_binary(body, "add", result, scaled, bias_b, out_shape, d);
    body << '\n';
    ctx.bind(out->id(), result);
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
    // JIT-R013: sibling of the already-fixed handle_batch_norm2d guard
    // above and JIT-037 (the interpreter's grad-mode Dropout replay).
    // Unconditional identity here silently drops BOTH the regularization
    // mask AND the 1/(1-p) scaling for a training-mode Dropout node. The
    // only real producer of an OpType::Dropout node (nested_dropout,
    // nested_ops.cpp) already gates `if (!training || p == 0.0) return
    // input;` BEFORE ever dispatching, so every OpType::Dropout node that
    // can reach this lowering is training=true by construction — fail
    // loudly instead of producing wrong numerics, matching
    // handle_batch_norm2d's established pattern.
    if (node.get_bool_attr("training")) {
        throw std::runtime_error(
            "GraphToMLIR: Dropout(training=true) has no MLIR/IREE lowering "
            "(the fused kernel it needs is interpreter/backend-only); fall "
            "back to eager or the interpreter for this graph");
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
    const double eps = get_attr_double(node, {"eps"}, 1e-5);

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

    // Two dtypes are in play (mirrors handle_rms_norm):
    //  * cd — the elementwise/affine dtype. For F16/BF16 we widen to F32 and
    //         narrow the result at the end; else cd == d.
    //  * ad — the mean/variance ACCUMULATION dtype. The eager CPU LayerNorm
    //         kernel accumulates the two-pass mean and variance in DOUBLE
    //         UNCONDITIONALLY, for every T including Float16/BFloat16
    //         (layer_norm_scalar<T>'s `sum`/`var` are always `double`, fed via
    //         the widen-to-double `ln_to_double()` helper regardless of T --
    //         nn_kernels.cpp; F32 uses the SIMD double-accumulate path,
    //         layer_norm_simd_sum_f64/layer_norm_simd_sumsq_f64). This
    //         previously said "F16/BF16 accumulate in F32", which was true
    //         when first written but went stale once the Float32-accumulator-
    //         inside-template<T> class of bug (the same class fixed here for
    //         F32 as JIT-F053/this comment's own history) was ALSO fixed for
    //         the F16/BF16 scalar path -- leaving this MLIR lowering as the
    //         only place still narrowing F16/BF16 accumulation to F32,
    //         diverging from eager AND from extended_codegen's native-codegen
    //         path (which already accumulates unconditionally in double).
    //         Reduce mean/variance in `ad` (always F64), then narrow the
    //         float mean and the RECIPROCAL inv_std to cd and normalize in
    //         cd, matching eager (which centers the variance with the double
    //         mean, then narrows mean and 1/sqrt to float for the output
    //         normalize).
    const bool widen = (d == ::tenzor::DType::Float16 ||
                        d == ::tenzor::DType::BFloat16);
    const auto cd = widen ? ::tenzor::DType::Float32 : d;
    const auto ad = ::tenzor::DType::Float64;

    std::string x = ctx.name_for(x_val->id());
    if (widen) {
        auto xc = ctx.fresh_name();
        emit_stablehlo_convert(body, xc, x, shape, d, cd);
        body << '\n';
        x = xc;
    }
    // x in the accumulation dtype (an extra convert only when ad != cd, i.e. F32).
    std::string x_ad = x;
    if (ad != cd) {
        auto xa = ctx.fresh_name();
        emit_stablehlo_convert(body, xa, x, shape, cd, ad);
        body << '\n';
        // IREE's LLVMGPUVectorDistribute codegen (CUDA/ROCm) fails to
        // distribute a stablehlo.reduce that gets fused -- even transitively,
        // through elementwise ops -- with the stablehlo.convert that widened
        // to this Float64 accumulation dtype (confirmed via minimal repro:
        // reducing a *raw* f64 input compiles fine; reducing a value produced
        // by convert-to-f64 does not, on either the mean or variance pass
        // below). A barrier right after the widening convert blocks that
        // fusion and is a semantic no-op on every other target -- the same
        // fix pattern as JIT-R110's Vulkan dot_general barrier in
        // emit_stablehlo_dot_general.
        auto xab = ctx.fresh_name();
        emit_stablehlo_optimization_barrier(body, xab, xa, shape, ad);
        body << '\n';
        x_ad = xab;
    }

    // n_const = N (accumulation dtype), reused for the mean and variance divides.
    auto n_const = ctx.fresh_name();
    emit_stablehlo_splat_constant(body, n_const,
                                  scalar_literal(static_cast<double>(N), ad),
                                  reduced_shape, ad);
    body << '\n';

    // Pass 1: mean = reduce_sum(x_ad) / N   (in ad).
    auto init0 = ctx.fresh_name();
    emit_stablehlo_splat_constant(body, init0, scalar_literal(0.0, ad), {}, ad);
    body << '\n';
    auto sum_x = ctx.fresh_name();
    emit_stablehlo_reduce(body, sum_x, x_ad, init0, "add", norm_dims, shape,
                          reduced_shape, ad);
    body << '\n';
    auto mean_ad = ctx.fresh_name();
    emit_stablehlo_binary(body, "divide", mean_ad, sum_x, n_const,
                          reduced_shape, ad);
    body << '\n';
    auto mean_ad_b = ctx.fresh_name();
    emit_stablehlo_broadcast_in_dim(body, mean_ad_b, mean_ad, bcast_dims,
                                    reduced_shape, shape, ad);
    body << '\n';

    // Pass 2: var = reduce_sum((x_ad - mean_ad)^2) / N   (in ad, double-centered
    // exactly as eager's layer_norm_simd_sumsq_f64, which uses the double mean).
    auto centered_ad = ctx.fresh_name();
    emit_stablehlo_binary(body, "subtract", centered_ad, x_ad, mean_ad_b, shape,
                          ad);
    body << '\n';
    auto sq = ctx.fresh_name();
    emit_stablehlo_binary(body, "multiply", sq, centered_ad, centered_ad, shape,
                          ad);
    body << '\n';
    auto init1 = ctx.fresh_name();
    emit_stablehlo_splat_constant(body, init1, scalar_literal(0.0, ad), {}, ad);
    body << '\n';
    auto sum_sq = ctx.fresh_name();
    emit_stablehlo_reduce(body, sum_sq, sq, init1, "add", norm_dims, shape,
                          reduced_shape, ad);
    body << '\n';
    auto var_ad = ctx.fresh_name();
    emit_stablehlo_binary(body, "divide", var_ad, sum_sq, n_const,
                          reduced_shape, ad);
    body << '\n';

    // inv_std = 1 / sqrt(var + eps)   (in ad), matching eager's reciprocal.
    auto eps_c = ctx.fresh_name();
    emit_stablehlo_splat_constant(body, eps_c, scalar_literal(eps, ad),
                                  reduced_shape, ad);
    body << '\n';
    auto var_eps = ctx.fresh_name();
    emit_stablehlo_binary(body, "add", var_eps, var_ad, eps_c, reduced_shape, ad);
    body << '\n';
    auto std_r = ctx.fresh_name();
    emit_stablehlo_unary(body, "sqrt", std_r, var_eps, reduced_shape, ad);
    body << '\n';
    auto one_ad = ctx.fresh_name();
    emit_stablehlo_splat_constant(body, one_ad, scalar_literal(1.0, ad),
                                  reduced_shape, ad);
    body << '\n';
    auto inv_std = ctx.fresh_name();
    emit_stablehlo_binary(body, "divide", inv_std, one_ad, std_r, reduced_shape,
                          ad);
    body << '\n';

    // Narrow the float mean and the reciprocal inv_std to the elementwise dtype
    // cd (eager narrows both to float before the output normalize).
    std::string mean_cd = mean_ad;
    std::string inv_cd = inv_std;
    if (ad != cd) {
        auto m = ctx.fresh_name();
        emit_stablehlo_convert(body, m, mean_ad, reduced_shape, ad, cd);
        body << '\n';
        mean_cd = m;
        auto iv = ctx.fresh_name();
        emit_stablehlo_convert(body, iv, inv_std, reduced_shape, ad, cd);
        body << '\n';
        inv_cd = iv;
    }
    auto mean_b = ctx.fresh_name();
    emit_stablehlo_broadcast_in_dim(body, mean_b, mean_cd, bcast_dims,
                                    reduced_shape, shape, cd);
    body << '\n';
    auto inv_b = ctx.fresh_name();
    emit_stablehlo_broadcast_in_dim(body, inv_b, inv_cd, bcast_dims,
                                    reduced_shape, shape, cd);
    body << '\n';

    // x_hat = (x - mean) * inv_std   (elementwise dtype cd, float-centered like
    // eager's output pass).
    auto centered = ctx.fresh_name();
    emit_stablehlo_binary(body, "subtract", centered, x, mean_b, shape, cd);
    body << '\n';
    auto x_hat = ctx.fresh_name();
    emit_stablehlo_binary(body, "multiply", x_hat, centered, inv_b, shape, cd);
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

    // Apply any fused activation (ReLU/GELU) folded into this LayerNorm node by
    // FuseLayerNormActivationPass, mirroring the interpreter (JIT-F013).
    final_name = emit_fused_epilogue(ctx, node, body, final_name, shape, d);

    // Bind the primary output (out_val->id()) to the final tensor. Additional
    // outputs (mean, rstd from a FusedLayerNorm contract) are not emitted; they
    // become orphaned unless they appear in g.outputs(), in which case the
    // lowering errors when looking up their SSA name.
    ctx.bind(out_val->id(), final_name);
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
    const auto out_dtype = out_val->dtype();
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

    // Widen F16/BF16 to F32 for the normalize math (matching handle_layer_norm /
    // handle_rms_norm) then narrow. Running the (x-mean)*rsqrt(var+eps)*gamma+beta
    // pipeline in half precision diverges from an eager kernel that normalizes in
    // F32. epsilon is always an f32 attribute (StableHLO requires F32Attr).
    const bool widen = (out_dtype == ::tenzor::DType::Float16 ||
                        out_dtype == ::tenzor::DType::BFloat16);
    const auto d = widen ? ::tenzor::DType::Float32 : out_dtype;

    std::string xn = ctx.name_for(x_val->id());
    std::string wn = ctx.name_for(w_val->id());
    std::string bn = ctx.name_for(b_val->id());
    std::string mn = ctx.name_for(m_val->id());
    std::string vn = ctx.name_for(v_val->id());
    if (widen) {
        auto cvt = [&](std::string& nm,
                       const std::shared_ptr<::tenzor::jit::Value>& val) {
            auto c = ctx.fresh_name();
            emit_stablehlo_convert(body, c, nm, val->shape(), out_dtype, d);
            body << '\n';
            nm = c;
        };
        cvt(xn, x_val); cvt(wn, w_val); cvt(bn, b_val);
        cvt(mn, m_val); cvt(vn, v_val);
    }

    // Use the explicit stablehlo.batch_norm_inference op.
    auto bn_out = ctx.fresh_name();
    body << '%' << bn_out << " = \"stablehlo.batch_norm_inference\"(%"
         << xn << ", %" << wn << ", %" << bn << ", %" << mn << ", %" << vn
         << ") <{"
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

    if (widen) {
        auto out_name = ctx.fresh_name();
        ctx.bind(out_val->id(), out_name);
        emit_stablehlo_convert(body, out_name, bn_out, shape, d, out_dtype);
        body << '\n';
    } else {
        ctx.bind(out_val->id(), bn_out);
    }
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

    // For F16/BF16, compute in F32 and narrow once — matching eager
    // (Float16(pow(float(x), n))). Repeated half-precision multiply (int pow) and
    // half-precision stablehlo.power (fractional) otherwise accumulate rounding
    // that diverges from eager and across HAL targets.
    const bool widen = (d == ::tenzor::DType::Float16 ||
                        d == ::tenzor::DType::BFloat16);
    const auto cd = widen ? ::tenzor::DType::Float32 : d;

    // Bind out->id() to `result_cd` (a value in the compute dtype), narrowing to
    // d first when widening.
    auto bind_result = [&](const std::string& result_cd) {
        if (widen) {
            auto out_name = ctx.fresh_name();
            ctx.bind(out->id(), out_name);
            emit_stablehlo_convert(body, out_name, result_cd, shape, cd, d);
            body << '\n';
        } else {
            ctx.bind(out->id(), result_cd);
        }
    };

    // Lower a small non-negative INTEGER exponent to repeated multiply.
    // stablehlo.power is implemented as exp(y*log(x)) on GPU/SPIR-V HAL targets,
    // which returns NaN for ANY negative base — diverging from eager (which
    // multiplies) and from the llvm-cpu target (libm pow). Repeated multiply is
    // exact and NaN-free for negative bases. A fractional/negative exponent
    // still uses stablehlo.power (a negative base to a fractional power is
    // genuinely NaN under eager too, so semantics match there).
    auto lower_int_pow = [&](const std::string& a_in, int n) {
        std::string a = a_in;
        if (widen) {
            auto ac = ctx.fresh_name();
            emit_stablehlo_convert(body, ac, a, shape, d, cd);
            body << '\n';
            a = ac;
        }
        const int abs_n = n < 0 ? -n : n;
        std::string pos;  // x^|n| in the compute dtype
        if (abs_n == 0) {
            pos = ctx.fresh_name();
            emit_stablehlo_splat_constant(body, pos, scalar_literal(1.0, cd),
                                          shape, cd);
            body << '\n';
        } else {
            std::string cur = a;
            for (int i = 1; i < abs_n; ++i) {
                auto next = ctx.fresh_name();
                emit_stablehlo_binary(body, "multiply", next, cur, a, shape, cd);
                body << '\n';
                cur = next;
            }
            if (abs_n == 1) {
                // x^1 == x; realize with a multiply-by-one so the output gets its
                // own SSA value (1*x == x exactly under IEEE round-to-nearest).
                auto one_name = ctx.fresh_name();
                emit_stablehlo_splat_constant(body, one_name, scalar_literal(1.0, cd),
                                              shape, cd);
                body << '\n';
                pos = ctx.fresh_name();
                emit_stablehlo_binary(body, "multiply", pos, a, one_name, shape, cd);
                body << '\n';
            } else {
                pos = cur;  // cur holds x^|n| for |n| >= 2
            }
        }
        std::string result = pos;
        if (n < 0) {
            // x^n = 1 / x^|n| for a NEGATIVE integer exponent. Repeated-multiply +
            // reciprocal is NaN-free for negative bases, unlike stablehlo.power
            // (exp(y*log(x)) -> NaN on GPU/SPIR-V), and matches eager std::pow and
            // the llvm-cpu target (JIT-F061). x=0 gives 1/0 = inf, matching eager.
            auto one_name = ctx.fresh_name();
            emit_stablehlo_splat_constant(body, one_name, scalar_literal(1.0, cd),
                                          shape, cd);
            body << '\n';
            result = ctx.fresh_name();
            emit_stablehlo_binary(body, "divide", result, one_name, pos, shape, cd);
            body << '\n';
        }
        bind_result(result);
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
                if (std::abs(ev - r) < 1e-6 && std::abs(r) <= 64.0) {
                    auto a = maybe_broadcast(body, ctx, ctx.name_for(a_val->id()),
                                             a_val->shape(), shape, d);
                    lower_int_pow(a, static_cast<int>(r));  // binds out->id()
                    return;
                }
            }
        }
        // General case: broadcast both operands to the output shape (JIT-066) —
        // stablehlo.power requires equal operand shapes — then emit power. Widen
        // to F32 for F16/BF16 (fractional/tensor power via exp(y*log(x))).
        auto a = maybe_broadcast(body, ctx, ctx.name_for(a_val->id()),
                                 a_val->shape(), shape, d);
        auto b = maybe_broadcast(body, ctx, ctx.name_for(b_val->id()),
                                 b_val->shape(), shape, d);
        if (widen) {
            auto ac = ctx.fresh_name();
            emit_stablehlo_convert(body, ac, a, shape, d, cd); body << '\n';
            auto bc = ctx.fresh_name();
            emit_stablehlo_convert(body, bc, b, shape, d, cd); body << '\n';
            a = ac; b = bc;
        }
        // JIT-R011: stablehlo.power is exp(y*log(x)) on GPU/SPIR-V HAL targets,
        // which returns NaN for ANY negative base regardless of whether y
        // happens to be an integer at runtime -- diverging from eager/libm
        // pow, which gives the correctly-signed real result for a negative
        // base raised to an integer power (and NaN for a non-integer power,
        // matching real-analysis pow). The compile-time-constant-exponent
        // paths above (lower_int_pow) already handle this; this is the same
        // fix for a dynamically-computed (non-constant, 2-tensor-operand)
        // exponent tensor, decomposed as:
        //   magnitude = |x|^y = exp(y * log(|x|))
        //   result = magnitude,  negated when x<0 and y is an odd integer,
        //            forced to NaN when x<0 and y is not an integer.
        auto abs_x = ctx.fresh_name();
        emit_stablehlo_unary(body, "abs", abs_x, a, shape, cd); body << '\n';
        auto log_abs_x = ctx.fresh_name();
        emit_stablehlo_unary(body, "log", log_abs_x, abs_x, shape, cd); body << '\n';
        auto y_log = ctx.fresh_name();
        emit_stablehlo_binary(body, "multiply", y_log, b, log_abs_x, shape, cd); body << '\n';
        auto magnitude = ctx.fresh_name();
        emit_stablehlo_unary(body, "exponential", magnitude, y_log, shape, cd); body << '\n';

        auto emit_compare = [&](const char* direction, const std::string& lhs,
                                const std::string& rhs) {
            auto result = ctx.fresh_name();
            body << '%' << result << " = stablehlo.compare " << direction
                 << ", %" << lhs << ", %" << rhs << " : (";
            write_tensor_type_for_emit(body, shape, cd);
            body << ", ";
            write_tensor_type_for_emit(body, shape, cd);
            body << ") -> ";
            write_tensor_type_for_emit(body, shape, ::tenzor::DType::Bool);
            body << '\n';
            return result;
        };
        auto emit_select = [&](const std::string& pred, const std::string& on_true,
                               const std::string& on_false) {
            auto result = ctx.fresh_name();
            body << '%' << result << " = stablehlo.select %" << pred << ", %"
                 << on_true << ", %" << on_false << " : ";
            write_tensor_type_for_emit(body, shape, ::tenzor::DType::Bool);
            body << ", ";
            write_tensor_type_for_emit(body, shape, cd);
            body << '\n';
            return result;
        };

        // is_int = (y == round_nearest_even(y))
        auto y_round = ctx.fresh_name();
        emit_stablehlo_unary(body, "round_nearest_even", y_round, b, shape, cd); body << '\n';
        auto is_int = emit_compare("EQ", b, y_round);

        // is_odd = is_int AND round_nearest_even(y/2) != y/2 (a half-integer
        // remainder iff y_round is odd).
        auto half_c = ctx.fresh_name();
        emit_stablehlo_splat_constant(body, half_c, scalar_literal(0.5, cd), shape, cd);
        body << '\n';
        auto half_y = ctx.fresh_name();
        emit_stablehlo_binary(body, "multiply", half_y, y_round, half_c, shape, cd); body << '\n';
        auto half_y_round = ctx.fresh_name();
        emit_stablehlo_unary(body, "round_nearest_even", half_y_round, half_y, shape, cd);
        body << '\n';
        auto is_even = emit_compare("EQ", half_y, half_y_round);
        auto not_even = ctx.fresh_name();
        emit_stablehlo_unary(body, "not", not_even, is_even, shape, ::tenzor::DType::Bool);
        body << '\n';
        auto is_odd_int = ctx.fresh_name();
        emit_stablehlo_binary(body, "and", is_odd_int, is_int, not_even, shape,
                              ::tenzor::DType::Bool);
        body << '\n';

        // x_neg = (x < 0)
        auto zero_c = ctx.fresh_name();
        emit_stablehlo_splat_constant(body, zero_c, scalar_literal(0.0, cd), shape, cd);
        body << '\n';
        auto x_neg = emit_compare("LT", a, zero_c);

        // signed_result = select(x_neg AND is_odd_int, -magnitude, magnitude)
        auto apply_negate = ctx.fresh_name();
        emit_stablehlo_binary(body, "and", apply_negate, x_neg, is_odd_int, shape,
                              ::tenzor::DType::Bool);
        body << '\n';
        auto neg_magnitude = ctx.fresh_name();
        emit_stablehlo_unary(body, "negate", neg_magnitude, magnitude, shape, cd); body << '\n';
        auto signed_result = emit_select(apply_negate, neg_magnitude, magnitude);

        // final = select(x_neg AND NOT is_int, NaN, signed_result) -- a
        // negative base to a genuinely fractional power is NaN under
        // eager/libm pow too, matching real-analysis semantics.
        auto not_int = ctx.fresh_name();
        emit_stablehlo_unary(body, "not", not_int, is_int, shape, ::tenzor::DType::Bool);
        body << '\n';
        auto needs_nan = ctx.fresh_name();
        emit_stablehlo_binary(body, "and", needs_nan, x_neg, not_int, shape,
                              ::tenzor::DType::Bool);
        body << '\n';
        auto nan_c = ctx.fresh_name();
        emit_stablehlo_splat_constant(body, nan_c, scalar_literal(std::nan(""), cd), shape, cd);
        body << '\n';
        auto p = emit_select(needs_nan, nan_c, signed_result);
        bind_result(p);
        return;
    }
    if (node.inputs().size() == 1) {
        const auto& a_in = ctx.name_for(node.inputs()[0]->id());
        // Read the exponent at full double precision so an f64 graph's fractional
        // exponent (e.g. 0.1) is not truncated through float (JIT-F048).
        double exp = get_attr_double(node, {"exponent"}, 1.0);
        double r = std::round(exp);
        if (std::abs(exp - r) < 1e-6 && std::abs(r) <= 64.0) {
            lower_int_pow(a_in, static_cast<int>(r));
            return;
        }
        std::string a = a_in;
        if (widen) {
            auto ac = ctx.fresh_name();
            emit_stablehlo_convert(body, ac, a, shape, d, cd); body << '\n';
            a = ac;
        }
        auto exp_name = ctx.fresh_name();
        emit_stablehlo_splat_constant(body, exp_name, scalar_literal(exp, cd),
                                      shape, cd);
        body << '\n';
        auto p = ctx.fresh_name();
        emit_stablehlo_binary(body, "power", p, a, exp_name, shape, cd);
        body << '\n';
        bind_result(p);
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

    // JIT-R068: sibling of the interpreter path's OpType::FlashAttention
    // dropout_p guard (JIT-R054, graph.cpp execute_node) — this MLIR
    // lowering path emits NO dropout step at all, so a traced node with
    // dropout_p > 0 would silently compile to a dropout-free (wrong)
    // forward pass instead of failing loudly. Fail the LOWERING itself
    // (before any IREE compile/cache work) rather than let the interpreter
    // path's later guard be the only backstop — GraphToMLIR::lower() is a
    // separate, independently-reachable path.
    const float dropout_p = get_attr_float(node, {"dropout_p"}, 0.0f);
    if (dropout_p > 0.0f) {
        throw std::runtime_error(
            "GraphToMLIR: FlashAttention (expand) with dropout_p > 0 is not "
            "wired for MLIR/IREE lowering (the dropout mask is not "
            "captured); fall back to eager");
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

    // JIT-R068: see handle_flash_attention_expand's identical guard — this
    // GQA lowering emits no dropout step either.
    const float dropout_p = get_attr_float(node, {"dropout_p"}, 0.0f);
    if (dropout_p > 0.0f) {
        throw std::runtime_error(
            "GraphToMLIR: GQA (expand) with dropout_p > 0 is not wired for "
            "MLIR/IREE lowering (the dropout mask is not captured); fall "
            "back to eager");
    }

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

    // JIT-R007: widen F16/BF16 to F32 for the whole rotate-half pipeline,
    // matching eager's rope.cpp ("F121": computing x*cos +- x*sin in half
    // loses angle precision and makes cross-backend parity depend on each
    // backend's half-precision rounding) and every OTHER numeric handler in
    // this file (softmax/gelu/silu/pow/conv2d/pooling/layer_norm/
    // batch_norm/flash_attention_expand/gqa_expand all do this).
    const bool widen = (d == ::tenzor::DType::Float16 ||
                        d == ::tenzor::DType::BFloat16);
    const auto cd = widen ? ::tenzor::DType::Float32 : d;

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

    auto x = ctx.name_for(x_val->id());
    auto cos_name = ctx.name_for(cos_val->id());
    auto sin_name = ctx.name_for(sin_val->id());
    if (widen) {
        auto xc = ctx.fresh_name();
        emit_stablehlo_convert(body, xc, x, x_shape, d, cd);
        body << '\n';
        x = xc;
        auto cosc = ctx.fresh_name();
        emit_stablehlo_convert(body, cosc, cos_name, cos_val->shape(), d, cd);
        body << '\n';
        cos_name = cosc;
        auto sinc = ctx.fresh_name();
        emit_stablehlo_convert(body, sinc, sin_name, sin_val->shape(), d, cd);
        body << '\n';
        sin_name = sinc;
    }

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
    write_tensor_type_for_emit(body, x_shape, cd);
    body << ") -> ";
    write_tensor_type_for_emit(body, half_shape, cd);
    body << '\n';

    auto x2 = ctx.fresh_name();
    body << '%' << x2 << " = \"stablehlo.slice\"(%" << x
         << ") <{start_indices = array<i64: " << render_dims(start_hi)
         << ">, limit_indices = array<i64: " << render_dims(limit_hi)
         << ">, strides = array<i64: " << render_dims(strides)
         << ">}> : (";
    write_tensor_type_for_emit(body, x_shape, cd);
    body << ") -> ";
    write_tensor_type_for_emit(body, half_shape, cd);
    body << '\n';

    // -x2
    auto neg_x2 = ctx.fresh_name();
    emit_stablehlo_unary(body, "negate", neg_x2, x2, half_shape, cd);
    body << '\n';

    // rotated = concat([-x2, x1], dim=last).
    auto rotated = ctx.fresh_name();
    body << '%' << rotated << " = \"stablehlo.concatenate\"(%" << neg_x2
         << ", %" << x1 << ") <{dimension = " << (rank - 1) << " : i64}> : ("
         << "tensor<";
    for (auto v : half_shape) body << v << 'x';
    body << dtype_suffix(cd) << ">, tensor<";
    for (auto v : half_shape) body << v << 'x';
    body << dtype_suffix(cd) << ">) -> ";
    write_tensor_type_for_emit(body, x_shape, cd);
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
                                        tab_shape, x_shape, cd);
        body << '\n';
        return b;
    };

    auto cos_b = broadcast_table(cos_name, cos_val->shape());
    auto sin_b = broadcast_table(sin_name, sin_val->shape());

    // x_mul_cos = x * cos
    auto x_mul_cos = ctx.fresh_name();
    emit_stablehlo_binary(body, "multiply", x_mul_cos, x, cos_b, x_shape, cd);
    body << '\n';
    // rot_mul_sin = rotated * sin
    auto rot_mul_sin = ctx.fresh_name();
    emit_stablehlo_binary(body, "multiply", rot_mul_sin, rotated, sin_b,
                          x_shape, cd);
    body << '\n';
    // out = x_mul_cos + rot_mul_sin
    if (widen) {
        auto sum_cd = ctx.fresh_name();
        emit_stablehlo_binary(body, "add", sum_cd, x_mul_cos, rot_mul_sin,
                              x_shape, cd);
        body << '\n';
        auto out_name = ctx.fresh_name();
        ctx.bind(out_val->id(), out_name);
        emit_stablehlo_convert(body, out_name, sum_cd, x_shape, cd, d);
        body << '\n';
    } else {
        auto out_name = ctx.fresh_name();
        ctx.bind(out_val->id(), out_name);
        emit_stablehlo_binary(body, "add", out_name, x_mul_cos, rot_mul_sin,
                              x_shape, cd);
        body << '\n';
    }
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
    const double eps    = get_attr_double(node, {"eps"}, 1e-6);
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

    // Two dtypes are in play:
    //  * cd  — the normalize/elementwise dtype (x/rms and the affine scale). For
    //          F16/BF16 we widen to F32 and narrow at the end; else cd == d.
    //  * ad  — the sum-of-squares ACCUMULATION dtype. The eager RMSNorm kernel
    //          accumulates sum_sq (and computes mean/rsqrt) in DOUBLE
    //          UNCONDITIONALLY: fused_rms_norm_kernel widens an F16/BF16 input
    //          to Float32 and delegates to the Float32 branch, which itself
    //          accumulates sum_sq in double via fused_ln_sumsq_f64
    //          (fused_ops.cpp) -- so F16/BF16 accumulates in double just like
    //          F32/F64 do, not F32. This previously said "F16/BF16 accumulate
    //          in F32" (stale once that path was fixed to match F32's
    //          double-accumulate, the same JIT-F053 class this comment
    //          documents), leaving this MLIR lowering as the only path still
    //          narrowing F16/BF16 accumulation to F32 -- diverging from eager
    //          AND from extended_codegen's native-codegen path (already
    //          double, unconditionally). Reduce in `ad` (always F64), then
    //          narrow the resulting rms to cd (eager narrows inv_rms before
    //          applying it).
    const bool widen = (d == ::tenzor::DType::Float16 ||
                        d == ::tenzor::DType::BFloat16);
    const auto cd = widen ? ::tenzor::DType::Float32 : d;
    const auto ad = ::tenzor::DType::Float64;

    std::string x = ctx.name_for(x_val->id());
    if (widen) {
        auto xc = ctx.fresh_name();
        emit_stablehlo_convert(body, xc, x, shape, d, cd);
        body << '\n';
        x = xc;
    }
    // x in the accumulation dtype (only an extra convert when ad != cd, i.e. F32).
    std::string x_ad = x;
    if (ad != cd) {
        auto xa = ctx.fresh_name();
        emit_stablehlo_convert(body, xa, x, shape, cd, ad);
        body << '\n';
        // IREE's LLVMGPUVectorDistribute codegen (CUDA/ROCm) fails to
        // distribute a stablehlo.reduce that gets fused -- even transitively,
        // through elementwise ops -- with the stablehlo.convert that widened
        // to this Float64 accumulation dtype (confirmed via minimal repro:
        // reducing a *raw* f64 input compiles fine; reducing a value produced
        // by convert-to-f64 does not). A barrier right after the widening
        // convert blocks that fusion and is a semantic no-op on every other
        // target -- the same fix pattern as JIT-R110's Vulkan dot_general
        // barrier in emit_stablehlo_dot_general.
        auto xab = ctx.fresh_name();
        emit_stablehlo_optimization_barrier(body, xab, xa, shape, ad);
        body << '\n';
        x_ad = xab;
    }

    // sq = x * x  (in the accumulation dtype)
    auto sq = ctx.fresh_name();
    emit_stablehlo_binary(body, "multiply", sq, x_ad, x_ad, shape, ad); body << '\n';
    // sum_sq = reduce_sum(sq, norm_dims)
    auto init0 = ctx.fresh_name();
    emit_stablehlo_splat_constant(body, init0, scalar_literal(0.0, ad), {}, ad);
    body << '\n';
    auto sum_sq = ctx.fresh_name();
    emit_stablehlo_reduce(body, sum_sq, sq, init0, "add", norm_dims, shape,
                          reduced_shape, ad); body << '\n';
    // mean = sum_sq / N
    auto n_c = ctx.fresh_name();
    emit_stablehlo_splat_constant(body, n_c,
                                  scalar_literal(static_cast<double>(N), ad),
                                  reduced_shape, ad);
    body << '\n';
    auto mean_sq = ctx.fresh_name();
    emit_stablehlo_binary(body, "divide", mean_sq, sum_sq, n_c, reduced_shape, ad);
    body << '\n';
    // var_eps = mean_sq + eps
    auto eps_c = ctx.fresh_name();
    emit_stablehlo_splat_constant(body, eps_c,
                                  scalar_literal(static_cast<double>(eps), ad),
                                  reduced_shape, ad);
    body << '\n';
    auto var_eps = ctx.fresh_name();
    emit_stablehlo_binary(body, "add", var_eps, mean_sq, eps_c, reduced_shape, ad);
    body << '\n';
    // inv_rms = 1 / sqrt(var_eps), computed in the accumulation dtype, THEN
    // narrowed to cd and MULTIPLIED into x — matching the eager kernel, which
    // narrows the RECIPROCAL (float32(1/sqrt_double)) and multiplies (not narrow
    // rms then divide). This keeps the rounding point identical to eager (F058).
    auto rms = ctx.fresh_name();
    emit_stablehlo_unary(body, "sqrt", rms, var_eps, reduced_shape, ad);
    body << '\n';
    auto one_ad = ctx.fresh_name();
    emit_stablehlo_splat_constant(body, one_ad, scalar_literal(1.0, ad),
                                  reduced_shape, ad);
    body << '\n';
    auto inv_rms = ctx.fresh_name();
    emit_stablehlo_binary(body, "divide", inv_rms, one_ad, rms, reduced_shape, ad);
    body << '\n';
    if (ad != cd) {
        auto inv_cd = ctx.fresh_name();
        emit_stablehlo_convert(body, inv_cd, inv_rms, reduced_shape, ad, cd);
        body << '\n';
        inv_rms = inv_cd;
    }
    auto inv_b = ctx.fresh_name();
    emit_stablehlo_broadcast_in_dim(body, inv_b, inv_rms, bcast_dims,
                                    reduced_shape, shape, cd);
    body << '\n';
    // xhat = x * inv_rms_b
    auto xhat = ctx.fresh_name();
    emit_stablehlo_binary(body, "multiply", xhat, x, inv_b, shape, cd);
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
    leaf_args_.clear();

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

    // JIT-R109: closure-captured parameter/buffer leaves (Graph::param_leaves()/
    // buffer_leaves(), populated when the tracer had set_parameters()/
    // set_buffers() declared before tracing) must NOT be baked as constants --
    // that would freeze BatchNorm/InstanceNorm running stats and any other
    // captured parameter/buffer at their trace-time value forever, since the
    // compiled module is reused verbatim on every replay. Bind each as an
    // EXTRA @main argument beyond the declared graph inputs, in a
    // deterministic order (params first sorted by index, then buffers sorted
    // by index -- independent of unordered_map iteration order, so the
    // compile cache's content hash is stable). The caller (mlir_invoke_impl)
    // must pass each leaf's CURRENT value, in this exact order, appended
    // after the regular inputs on every invocation -- see leaf_args().
    {
        std::vector<std::pair<std::string, std::size_t>> param_ids(
            g.param_leaves().begin(), g.param_leaves().end());
        std::sort(param_ids.begin(), param_ids.end(),
                 [](const auto& a, const auto& b) { return a.second < b.second; });
        for (const auto& [id, param_index] : param_ids) {
            const auto& param = *g.parameters().at(param_index);
            auto arg_name = "arg" + std::to_string(inputs.size());
            ctx.bind(id, arg_name);
            const auto pshape = param.shape();
            inputs.emplace_back(std::vector<int64_t>(pshape.begin(), pshape.end()),
                                param.tensor().dtype());
            leaf_args_.push_back({/*is_buffer=*/false, param_index});
        }
        std::vector<std::pair<std::string, std::size_t>> buffer_ids(
            g.buffer_leaves().begin(), g.buffer_leaves().end());
        std::sort(buffer_ids.begin(), buffer_ids.end(),
                 [](const auto& a, const auto& b) { return a.second < b.second; });
        for (const auto& [id, buffer_index] : buffer_ids) {
            const auto& buf = *g.buffers().at(buffer_index);
            auto arg_name = "arg" + std::to_string(inputs.size());
            ctx.bind(id, arg_name);
            const auto bshape = buf.shape();
            inputs.emplace_back(std::vector<int64_t>(bshape.begin(), bshape.end()),
                                buf.tensor().dtype());
            leaf_args_.push_back({/*is_buffer=*/true, buffer_index});
        }
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
            // stablehlo.remainder's sign follows the dividend (truncated
            // toward zero), matching tenzor::fmod's C-style semantics exactly
            // (distinct from tenzor::remainder, which floors like Python %).
            case OpType::Fmod:         handle_binary(ctx, *node, body, "remainder"); break;
            case OpType::ResidualAdd:  handle_binary(ctx, *node, body, "add");      break;

            // ── Elementwise unary ──
            case OpType::Neg:          handle_unary(ctx, *node, body, "negate");      break;
            case OpType::Abs:          handle_unary(ctx, *node, body, "abs");         break;
            case OpType::Exp:          handle_unary(ctx, *node, body, "exponential"); break;
            case OpType::Log:          handle_unary(ctx, *node, body, "log");         break;
            case OpType::Sqrt:         handle_unary(ctx, *node, body, "sqrt");        break;
            case OpType::Round:        handle_unary(ctx, *node, body, "round_nearest_even"); break;
            case OpType::Sin:          handle_unary(ctx, *node, body, "sine");        break;
            case OpType::Cos:          handle_unary(ctx, *node, body, "cosine");      break;
            case OpType::Rsqrt:        handle_unary(ctx, *node, body, "rsqrt");       break;

            // ── Special-shape elementwise ──
            case OpType::Pow:          handle_pow(ctx, *node, body);   break;
            case OpType::Where:        handle_where(ctx, *node, body); break;
            case OpType::Eq:           handle_compare(ctx, *node, body, "EQ"); break;
            case OpType::Ne:           handle_compare(ctx, *node, body, "NE"); break;
            case OpType::Lt:           handle_compare(ctx, *node, body, "LT"); break;
            case OpType::Le:           handle_compare(ctx, *node, body, "LE"); break;
            case OpType::Gt:           handle_compare(ctx, *node, body, "GT"); break;
            case OpType::Ge:           handle_compare(ctx, *node, body, "GE"); break;
            // Inputs/outputs are already Bool (out->dtype() == in dtype), so
            // the generic dtype-converting handle_binary/handle_unary apply
            // directly -- no dtype mismatch between operand and output as
            // with comparisons above.
            case OpType::LogicalAnd:   handle_binary(ctx, *node, body, "and"); break;
            case OpType::LogicalOr:    handle_binary(ctx, *node, body, "or");  break;
            case OpType::LogicalNot:   handle_unary(ctx, *node, body, "not"); break;
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

            // ── Quantization (JIT-R081) ──
            case OpType::Quantize:             handle_quantize(ctx, *node, body); break;
            case OpType::Dequantize:           handle_dequantize(ctx, *node, body); break;
            case OpType::QuantizedLinear:      handle_quantized_linear_dynamic(ctx, *node, body); break;
            case OpType::QuantizedLinearStatic: handle_quantized_linear_static(ctx, *node, body); break;
            case OpType::QuantizedConv2d:      handle_quantized_conv2d_dynamic(ctx, *node, body); break;

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
