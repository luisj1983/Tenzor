/**
 * @file importer.cpp
 * @brief Implementation of ONNX model import functionality
 *
 * As of the P3 pass, this importer uses generated protobuf bindings from
 * proto/onnx.proto (via onnx.pb.h) instead of a hand-rolled wire-format
 * decoder. The hand-rolled version used non-canonical field numbers and
 * could not interoperate with onnxruntime / netron; this path targets the
 * canonical ONNX schema, so Tenzor-emitted and externally-produced models
 * are bidirectionally compatible.
 */

#include "../../include/tenzor/onnx/importer.hpp"
#include "../../include/tenzor/utils/error.hpp"
#include "../../include/tenzor/nn/layers/linear.hpp"
#include "../../include/tenzor/nn/layers/conv.hpp"
#include "../../include/tenzor/nn/layers/padding.hpp"
#include "../../include/tenzor/nn/module.hpp"
#include "../../include/tenzor/nn/layers/batchnorm.hpp"
#include "../../include/tenzor/nn/layers/normalization.hpp"
#include "../../include/tenzor/nn/layers/pooling.hpp"
#include "../../include/tenzor/nn/layers/flatten.hpp"
#include "../../include/tenzor/nn/layers/rnn.hpp"  // I6-followup: LSTM / GRU / RNN converters
#include "../../include/tenzor/nn/activations/activations.hpp"
#include "../../include/tenzor/ops/math.hpp"
#include "../../include/tenzor/ops/transform.hpp"
#include "../../include/tenzor/ops/creation.hpp"
#include "../../include/tenzor/ops/reduction.hpp"
#include "../../include/tenzor/ops/advanced.hpp"   // Audit I6: topk, einsum
#include "../../include/tenzor/nn/functional.hpp"  // Audit I1: nn::functional::pad
#include "../../include/tenzor/ops/indexing.hpp"
#include "../../include/tenzor/ops/vision.hpp"
#include "../../include/tenzor/ops/linalg.hpp"   // custom-domain linalg re-import
#include <array>
#include <cstring>
#include <limits>
#include <sstream>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <filesystem>

#ifdef TENZOR_HAS_ONNX_PROTOBUF
#include "onnx.pb.h"
#endif

namespace tenzor {
namespace onnx {

// Forward declaration: defined below at namespace scope but used by
// proto_to_ir_tensor (in the anonymous namespace) to size sub-32-bit
// int32_data initializers.
auto onnx_to_dtype(ONNXDataType onnx_dtype) -> DType;

// ============================================================================
// Protobuf → Internal IR conversion
// ============================================================================

namespace {

#ifdef TENZOR_HAS_ONNX_PROTOBUF

// Canonical ONNX AttributeType enum values (from proto/onnx.proto). Kept
// as plain integers here so the compile-time cost of including the full
// generated header in every translation unit is avoided.
constexpr int32_t kAttrTypeFloat  = 1;
constexpr int32_t kAttrTypeInt    = 2;
constexpr int32_t kAttrTypeString = 3;
constexpr int32_t kAttrTypeTensor = 4;
constexpr int32_t kAttrTypeFloats  = 6;
constexpr int32_t kAttrTypeInts    = 7;
constexpr int32_t kAttrTypeStrings = 8;  // STRINGS list (Wave Inf-C4)

// 6th-audit Fix #1: load the bytes for an EXTERNAL TensorProto from the
// sidecar file referenced by its `external_data` entries. Throws on missing
// keys, short reads, or path-traversal attempts (e.g. "../etc/passwd").
auto load_external_tensor_bytes(const tenzor_onnx::TensorProto& t,
                                 const std::string& base_dir) -> std::vector<uint8_t> {
    if (base_dir.empty()) {
        throw std::runtime_error(
            "ONNXImporter: tensor '" + t.name() + "' references external_data "
            "but the importer was given a byte buffer with no source-file path "
            "anchor. Use import_from_file() instead of import_from_bytes() for "
            "models that use external data.");
    }
    std::string location;
    int64_t offset = 0;
    int64_t length = -1;
    for (const auto& kv : t.external_data()) {
        if (kv.key() == "location")    location = kv.value();
        else if (kv.key() == "offset") {
            try { offset = std::stoll(kv.value()); }
            catch (const std::exception&) {
                throw std::runtime_error(
                    "ONNXImporter: tensor '" + t.name() +
                    "' has malformed external_data offset value '" + kv.value() + "'");
            }
        }
        else if (kv.key() == "length") {
            try { length = std::stoll(kv.value()); }
            catch (const std::exception&) {
                throw std::runtime_error(
                    "ONNXImporter: tensor '" + t.name() +
                    "' has malformed external_data length value '" + kv.value() + "'");
            }
        }
    }
    if (location.empty()) {
        throw std::runtime_error(
            "ONNXImporter: tensor '" + t.name() + "' has data_location=EXTERNAL "
            "but no `location` entry in external_data");
    }
    // 7th-audit Fix #2: harden path traversal. The sidecar must sit next
    // to the .onnx file or in a subdirectory below it. We reject:
    //
    //   (a) Any ".." component — classic parent-directory traversal.
    //   (b) Absolute paths (POSIX "/x" or Windows "C:\x") — `std::filesystem::
    //       path::operator/` REPLACES the base when the RHS is absolute,
    //       so a malicious "/etc/passwd" would silently bypass the base_dir
    //       anchor entirely.
    //   (c) Empty after the constructor's parsing (defensive).
    //
    // Why both (a) and (b): the previous-pass fix caught (a) but missed (b),
    // surfaced by the seventh-pass audit.
    if (location.find("..") != std::string::npos) {
        throw std::runtime_error(
            "ONNXImporter: external_data location '" + location +
            "' contains '..' (path-traversal attempt rejected)");
    }
    namespace fs = std::filesystem;
    fs::path loc_path(location);
    if (loc_path.is_absolute()) {
        throw std::runtime_error(
            "ONNXImporter: external_data location '" + location +
            "' is an absolute path (path-traversal attempt rejected). "
            "Sidecar paths must be relative to the .onnx file's directory.");
    }
    // Also reject locations beginning with '/' or '\' even on platforms
    // where std::filesystem might not classify them as "absolute"
    // (e.g. POSIX paths viewed on Windows). Defence in depth.
    if (!location.empty() && (location.front() == '/' || location.front() == '\\')) {
        throw std::runtime_error(
            "ONNXImporter: external_data location '" + location +
            "' begins with a root separator (rejected).");
    }
    fs::path full = fs::path(base_dir) / loc_path;
    std::ifstream sidecar(full, std::ios::binary);
    if (!sidecar.is_open()) {
        throw std::runtime_error(
            "ONNXImporter: failed to open external_data sidecar: " + full.string());
    }
    sidecar.seekg(0, std::ios::end);
    const int64_t file_size = sidecar.tellg();
    sidecar.seekg(offset, std::ios::beg);
    if (length < 0) {
        // Per ONNX spec, length is optional — when absent, read to EOF.
        length = file_size - offset;
    }
    // Check length against the remaining bytes WITHOUT computing offset+length,
    // which can overflow int64_t for attacker-controlled values and wrap to a
    // small number that passes the bound. At this point (short-circuit) offset
    // is in [0, file_size] and length >= 0, so file_size - offset >= 0.
    if (offset < 0 || offset > file_size || length < 0 ||
        length > file_size - offset) {
        throw std::runtime_error(
            "ONNXImporter: external_data offset/length out of range for " +
            full.string() + " (file=" + std::to_string(file_size) +
            ", offset=" + std::to_string(offset) +
            ", length=" + std::to_string(length) + ")");
    }
    std::vector<uint8_t> bytes(static_cast<size_t>(length));
    sidecar.read(reinterpret_cast<char*>(bytes.data()),
                 static_cast<std::streamsize>(length));
    if (!sidecar) {
        throw std::runtime_error(
            "ONNXImporter: short read on external_data sidecar " + full.string());
    }
    return bytes;
}

// 6th-audit Fix #1: thread-local propagation of the source-file directory for
// EXTERNAL initializers. Set by `ONNXImporter::parse_model` for the duration
// of the parse and cleared on exit (RAII). Used by `proto_to_ir_tensor` when
// it sees `data_location()==EXTERNAL`, plus by `proto_to_ir_attribute` when
// it recurses into an embedded tensor.
thread_local std::string g_onnx_external_data_dir;

auto proto_to_ir_tensor(const tenzor_onnx::TensorProto& t,
                         const std::string& external_data_dir = "") -> ONNXTensorData {
    ONNXTensorData tensor;
    tensor.name = t.name();
    tensor.dtype = static_cast<ONNXDataType>(t.data_type());
    tensor.shape.assign(t.dims().begin(), t.dims().end());

    // 6th-audit Fix #1: handle data_location=EXTERNAL initializers. Per the
    // ONNX spec, when this is set the tensor's raw bytes live in a sidecar
    // file referenced by `external_data`.
    if (t.data_location() == tenzor_onnx::TensorProto_DataLocation_EXTERNAL) {
        tensor.raw_data = load_external_tensor_bytes(t, external_data_dir);
        return tensor;
    }

    if (!t.raw_data().empty()) {
        const std::string& raw = t.raw_data();
        tensor.raw_data.assign(raw.begin(), raw.end());
    } else if (!t.float_data().empty()) {
        // Producer used typed data instead of raw_data. Pack into raw_data
        // for the existing ONNXTensorData::to_tensor path (little-endian).
        tensor.raw_data.resize(static_cast<size_t>(t.float_data_size()) * sizeof(float));
        std::memcpy(tensor.raw_data.data(), t.float_data().data(),
                    tensor.raw_data.size());
    } else if (!t.int32_data().empty()) {
        // Per the ONNX TensorProto spec, the int32_data field stores one logical
        // value per int32 not only for INT32, but also for the sub-32-bit dtypes
        // FLOAT16, BFLOAT16, INT8, INT16, UINT8, UINT16, BOOL and the FP8
        // variants (each packed into the low bits of an int32). When the target
        // element is narrower than 4 bytes we must narrow each value to its real
        // storage width, otherwise to_tensor() rejects the 4-bytes-per-element
        // blob as a size mismatch.
        const size_t count = static_cast<size_t>(t.int32_data_size());
        const size_t elem_size = dtype_size(onnx_to_dtype(tensor.dtype));
        tensor.raw_data.resize(count * elem_size);
        const auto& src = t.int32_data();
        if (elem_size > sizeof(uint32_t)) {
            // int32_data only encodes dtypes up to 4 bytes wide (int8/uint8/
            // int16/uint16/int32/bool/float16/bfloat16 per the ONNX spec). A
            // wider element size would make the byte-extraction shift below
            // (v >> 8*b, b >= 4) shift a uint32_t by >= 32 bits — undefined
            // behaviour. Reject the malformed combination loudly.
            throw std::runtime_error(
                "ONNX tensor '" + tensor.name +
                "': int32_data cannot encode a dtype wider than 4 bytes");
        }
        if (elem_size == sizeof(int32_t)) {
            std::memcpy(tensor.raw_data.data(), src.data(), tensor.raw_data.size());
        } else {
            // Little-endian: the low `elem_size` bytes of each int32 hold the
            // narrowed value (e.g. uint16 bit pattern for Float16/BFloat16/Int16,
            // uint8 for Int8/UInt8/Bool/FP8).
            for (size_t i = 0; i < count; ++i) {
                const auto v = static_cast<uint32_t>(src[static_cast<int>(i)]);
                for (size_t b = 0; b < elem_size; ++b) {
                    tensor.raw_data[i * elem_size + b] =
                        static_cast<uint8_t>((v >> (8 * b)) & 0xFFu);
                }
            }
        }
    } else if (!t.int64_data().empty()) {
        tensor.raw_data.resize(static_cast<size_t>(t.int64_data_size()) * sizeof(int64_t));
        std::memcpy(tensor.raw_data.data(), t.int64_data().data(),
                    tensor.raw_data.size());
    } else if (!t.double_data().empty()) {
        tensor.raw_data.resize(static_cast<size_t>(t.double_data_size()) * sizeof(double));
        std::memcpy(tensor.raw_data.data(), t.double_data().data(),
                    tensor.raw_data.size());
    }
    return tensor;
}

auto proto_to_ir_attribute(const tenzor_onnx::AttributeProto& a) -> ONNXAttribute {
    ONNXAttribute attr;
    attr.name = a.name();
    const int32_t type = static_cast<int32_t>(a.type());

    // Canonical ONNX marks the payload via `type`. We honour that, but also
    // fall back to whichever typed field is populated for producers that
    // leave `type` at UNDEFINED.
    if (type == kAttrTypeInt || (a.i() != 0 && type == 0)) {
        attr.i = static_cast<int64_t>(a.i());
    }
    if (type == kAttrTypeFloat || (a.f() != 0.0f && type == 0)) {
        attr.f = a.f();
    }
    if (type == kAttrTypeString || (!a.s().empty() && type == 0)) {
        attr.s = a.s();
    }
    if (type == kAttrTypeTensor && a.has_t()) {
        // 6th-audit Fix #1: attribute-tensors can also use external_data;
        // defer to the same dir-aware helper. `external_data_dir_seen` is a
        // thread-local-style propagation via a module-level static; safer
        // than threading the dir through every signature for a path that
        // rarely fires.
        extern thread_local std::string g_onnx_external_data_dir;
        attr.tensor = proto_to_ir_tensor(a.t(), g_onnx_external_data_dir);
    }
    if (type == kAttrTypeInts || (!a.ints().empty() && type == 0)) {
        std::vector<int64_t> ints(a.ints().begin(), a.ints().end());
        attr.ints = std::move(ints);
    }
    if (type == kAttrTypeFloats || (!a.floats().empty() && type == 0)) {
        std::vector<float> floats(a.floats().begin(), a.floats().end());
        attr.floats = std::move(floats);
    }
    // Wave Inf-C4: STRINGS list (used by RNN/LSTM/GRU `activations`).
    if (type == kAttrTypeStrings || (!a.strings().empty() && type == 0)) {
        std::vector<std::string> strs;
        strs.reserve(a.strings().size());
        for (const auto& s : a.strings()) strs.emplace_back(s);
        attr.strings = std::move(strs);
    }
    return attr;
}

auto proto_to_ir_node(const tenzor_onnx::NodeProto& n) -> ONNXImportNode {
    ONNXImportNode node;
    node.op_type = n.op_type();
    node.name    = n.name();
    node.inputs.assign(n.input().begin(), n.input().end());
    node.outputs.assign(n.output().begin(), n.output().end());
    for (const auto& a : n.attribute()) {
        auto attr = proto_to_ir_attribute(a);
        node.attributes[attr.name] = std::move(attr);
    }
    return node;
}

auto proto_to_ir_value_info(const tenzor_onnx::ValueInfoProto& vi) -> ONNXImportValueInfo {
    ONNXImportValueInfo info;
    info.name = vi.name();
    if (vi.has_type()) {
        const auto& type = vi.type();
        // Only tensor_type is supported (see comment on TypeProto in
        // proto/onnx.proto). Other branches silently fall through.
        if (type.has_tensor_type()) {
            const auto& tt = type.tensor_type();
            info.dtype = static_cast<ONNXDataType>(tt.elem_type());
            if (tt.has_shape()) {
                for (const auto& d : tt.shape().dim()) {
                    // Unknown / dynamic dimensions are signalled as -1.
                    info.shape.push_back(
                        d.dim_value() != 0 ? static_cast<int64_t>(d.dim_value()) : -1);
                }
            }
        }
    }
    return info;
}

auto proto_to_ir_graph(const tenzor_onnx::GraphProto& g) -> ONNXGraphData {
    ONNXGraphData graph;
    graph.name = g.name();
    for (const auto& n : g.node()) {
        graph.nodes.push_back(proto_to_ir_node(n));
    }
    for (const auto& init : g.initializer()) {
        // 6th-audit Fix #1: pass the active source-file directory so that
        // EXTERNAL initializers resolve their sidecar relative to the .onnx.
        auto tensor = proto_to_ir_tensor(init, g_onnx_external_data_dir);
        graph.initializers[tensor.name] = std::move(tensor);
    }
    for (const auto& i : g.input()) {
        graph.inputs.push_back(proto_to_ir_value_info(i));
    }
    for (const auto& o : g.output()) {
        graph.outputs.push_back(proto_to_ir_value_info(o));
    }
    for (const auto& vi : g.value_info()) {
        auto info = proto_to_ir_value_info(vi);
        graph.value_info[info.name] = std::move(info);
    }
    return graph;
}

#endif // TENZOR_HAS_ONNX_PROTOBUF

auto shape_to_string(const std::vector<int64_t>& shape) -> std::string {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << shape[i];
    }
    oss << "]";
    return oss.str();
}

} // anonymous namespace

// ============================================================================
// DType Conversion
// ============================================================================

auto onnx_to_dtype(ONNXDataType onnx_dtype) -> DType {
    switch (onnx_dtype) {
        case ONNXDataType::FLOAT: return DType::Float32;
        case ONNXDataType::DOUBLE: return DType::Float64;
        case ONNXDataType::FLOAT16: return DType::Float16;
        case ONNXDataType::BFLOAT16: return DType::BFloat16;
        case ONNXDataType::INT8: return DType::Int8;
        case ONNXDataType::INT16: return DType::Int16;
        case ONNXDataType::INT32: return DType::Int32;
        case ONNXDataType::INT64: return DType::Int64;
        case ONNXDataType::UINT8: return DType::UInt8;
        // Symmetric with dtype_to_onnx, which emits UINT16/UINT32/UINT64 for the
        // corresponding DTypes. Without these cases a model Tenzor itself
        // exported with an unsigned dtype could not be round-tripped back.
        case ONNXDataType::UINT16: return DType::UInt16;
        case ONNXDataType::UINT32: return DType::UInt32;
        case ONNXDataType::UINT64: return DType::UInt64;
        case ONNXDataType::BOOL: return DType::Bool;
        // ONNX opset 20+ Float8 variants. The IEEE FN encodings (E4M3FN /
        // E5M2) share Tenzor's FP8 bit layout, so the 8-bit pattern can be
        // carried verbatim. The FNUZ variants ("finite, unsigned zero") use a
        // DIFFERENT exponent bias and have no signed zero / no infinity, so the
        // same 8-bit pattern denotes a different numeric value. Aliasing them to
        // the IEEE FN dtypes would silently misinterpret every stored value, so
        // we reject them explicitly rather than import corrupted data.
        case ONNXDataType::FLOAT8E4M3FN:   return DType::FP8_E4M3;
        case ONNXDataType::FLOAT8E5M2:     return DType::FP8_E5M2;
        case ONNXDataType::FLOAT8E4M3FNUZ: return DType::FP8_E4M3FNUZ;
        case ONNXDataType::FLOAT8E5M2FNUZ: return DType::FP8_E5M2FNUZ;
        default:
            throw std::runtime_error("Unsupported ONNX data type for import");
    }
}

// ============================================================================
// ONNXTensorData Implementation
// ============================================================================

auto ONNXTensorData::to_tensor(Device device) const -> Tensor {
    auto tenzor_dtype = onnx_to_dtype(dtype);
    // Validate the shape (rejects negative dims / numel overflow) BEFORE
    // allocating the tensor from attacker-controlled proto dims.
    int64_t n = numel();
    // Validate the raw-data size against the declared shape BEFORE allocating
    // the tensor, so an attacker-declared huge shape carrying little/no data
    // cannot trigger a giant (multi-TB) allocation ahead of the size check.
    if (!raw_data.empty()) {
        size_t expected_bytes;
        if (__builtin_mul_overflow(static_cast<size_t>(n),
                                   dtype_size(tenzor_dtype), &expected_bytes)) {
            throw std::runtime_error(
                "ONNX tensor '" + name + "' byte size overflows size_t");
        }
        if (raw_data.size() != expected_bytes) {
            throw std::runtime_error(
                "ONNX tensor '" + name + "' has mismatched data size. Expected " +
                std::to_string(expected_bytes) + " bytes, got " + std::to_string(raw_data.size())
            );
        }
    } else if (n > 0) {
        // raw_data carries all initializer data (raw + consolidated typed
        // fields). An empty payload with a non-zero declared element count is a
        // malformed/hostile initializer: reject it BEFORE allocating, so a huge
        // attacker-declared shape with no data cannot trigger a giant
        // (multi-GB/TB) allocation.
        throw std::runtime_error(
            "ONNX tensor '" + name + "' declares " + std::to_string(n) +
            " elements but carries no data");
    }

    Tensor tensor(shape, tenzor_dtype, device);

    // Copy raw data
    if (!raw_data.empty()) {
        // For CPU tensors, direct memcpy
        if (device == Device::cpu()) {
            std::memcpy(tensor.data_ptr(), raw_data.data(), raw_data.size());
        } else {
            // For GPU tensors, create temporary CPU tensor and copy
            Tensor cpu_tensor(shape, tenzor_dtype, Device::cpu());
            std::memcpy(cpu_tensor.data_ptr(), raw_data.data(), raw_data.size());
            tensor = cpu_tensor.to(device);
        }
    }

    return tensor;
}

auto ONNXTensorData::numel() const -> int64_t {
    int64_t result = 1;
    for (int64_t dim : shape) {
        if (dim < 0) {
            throw std::runtime_error(
                "ONNX tensor '" + name + "' has a negative dimension");
        }
        if (__builtin_mul_overflow(result, dim, &result)) {
            throw std::runtime_error(
                "ONNX tensor '" + name + "' element count overflows int64");
        }
    }
    return result;
}

// ============================================================================
// ONNXAttribute Implementation
// ============================================================================

auto ONNXAttribute::get_int(int64_t default_val) const -> int64_t {
    return i.value_or(default_val);
}

auto ONNXAttribute::get_float(float default_val) const -> float {
    return f.value_or(default_val);
}

auto ONNXAttribute::get_string(const std::string& default_val) const -> std::string {
    return s.value_or(default_val);
}

auto ONNXAttribute::get_ints(const std::vector<int64_t>& default_val) const -> std::vector<int64_t> {
    return ints.value_or(default_val);
}

auto ONNXAttribute::get_floats(const std::vector<float>& default_val) const -> std::vector<float> {
    return floats.value_or(default_val);
}

auto ONNXAttribute::get_strings(const std::vector<std::string>& default_val) const
    -> std::vector<std::string> {
    return strings.value_or(default_val);
}

// ============================================================================
// ONNXImportNode Implementation
// ============================================================================

auto ONNXImportNode::get_attr(const std::string& name) const -> std::optional<ONNXAttribute> {
    auto it = attributes.find(name);
    if (it != attributes.end()) {
        return it->second;
    }
    return std::nullopt;
}

// ============================================================================
// ONNXImportContext Implementation
// ============================================================================

auto ONNXImportContext::register_value(const std::string& name, const Tensor& tensor) -> void {
    values_[name] = tensor;
}

auto ONNXImportContext::get_value(const std::string& name) const -> std::optional<Tensor> {
    auto it = values_.find(name);
    if (it != values_.end()) {
        return it->second;
    }
    return std::nullopt;
}

auto ONNXImportContext::has_value(const std::string& name) const -> bool {
    return values_.find(name) != values_.end();
}

auto ONNXImportContext::register_module(const std::string& name, std::shared_ptr<nn::Module> module) -> void {
    modules_[name] = module;
}

auto ONNXImportContext::get_modules() const -> const std::unordered_map<std::string, std::shared_ptr<nn::Module>>& {
    return modules_;
}

auto ONNXImportContext::set_device(Device device) -> void {
    device_ = device;
}

auto ONNXImportContext::get_device() const -> Device {
    return device_;
}

// ============================================================================
// ONNXImporter Implementation
// ============================================================================

ONNXImporter::ONNXImporter(bool verbose)
    : verbose_(verbose) {}

auto ONNXImporter::set_verbose(bool verbose) -> void {
    verbose_ = verbose;
}

auto ONNXImporter::set_device(Device device) -> void {
    device_ = device;
    context_.set_device(device);
}

auto ONNXImporter::import_from_file(const std::string& filepath) -> std::shared_ptr<nn::Module> {
    log("Loading ONNX model from file: " + filepath);

    // Read file
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open ONNX file: " + filepath);
    }

    file.seekg(0, std::ios::end);
    const std::streamoff file_size_off = file.tellg();
    if (file_size_off < 0) {
        throw std::runtime_error("Failed to determine size of ONNX file: " + filepath);
    }
    file.seekg(0, std::ios::beg);

    const size_t file_size = static_cast<size_t>(file_size_off);
    std::vector<uint8_t> bytes(file_size);
    file.read(reinterpret_cast<char*>(bytes.data()), file_size);
    file.close();

    log("Read " + std::to_string(file_size) + " bytes from file");

    // 6th-audit Fix #1: anchor external_data lookups at the source file's
    // directory. parse_model() picks this up via the thread-local in
    // importer.cpp.
    namespace fs = std::filesystem;
    external_data_dir_ = fs::path(filepath).parent_path().string();
    if (external_data_dir_.empty()) {
        external_data_dir_ = ".";  // file in current directory
    }
    // 7th-audit Fix #3: tell the immediate import_from_bytes call NOT to
    // clear our freshly-set anchor.
    called_from_file_path_anchor_set_ = true;
    return import_from_bytes(bytes);
}

auto ONNXImporter::import_from_bytes(const std::vector<uint8_t>& bytes) -> std::shared_ptr<nn::Module> {
    log("Parsing ONNX model");

    // 7th-audit Fix #3: clear any external_data_dir_ left over from a
    // prior `import_from_file` call. `import_from_bytes` has no source
    // file anchor, so EXTERNAL initializers in the buffer must surface
    // the documented "no anchor" error rather than silently resolving
    // against a stale directory. `import_from_file` sets the
    // single-shot flag BEFORE forwarding to us so we don't clear the
    // anchor it just established.
    if (called_from_file_path_anchor_set_) {
        called_from_file_path_anchor_set_ = false;
    } else {
        external_data_dir_.clear();
    }

    // Parse model
    model_data_ = parse_model(bytes);

    log("Validating ONNX model");
    validate_model(model_data_);

    log("Converting ONNX graph to Tenzor module");
    auto module = convert_graph(model_data_.graph);

    log("Model import completed successfully");
    return module;
}

auto ONNXImporter::get_model_data() const -> const ONNXModelData& {
    return model_data_;
}

auto ONNXImporter::parse_model(const std::vector<uint8_t>& bytes) -> ONNXModelData {
#ifdef TENZOR_HAS_ONNX_PROTOBUF
    // 6th-audit Fix #1: publish the source-file directory (set by
    // import_from_file) into the thread-local read by proto_to_ir_tensor /
    // proto_to_ir_attribute. RAII reset so we don't pollute the next parse
    // on the same thread, even on exception.
    struct ExternalDataDirScope {
        std::string previous;
        explicit ExternalDataDirScope(const std::string& set_to)
            : previous(g_onnx_external_data_dir) {
            g_onnx_external_data_dir = set_to;
        }
        ~ExternalDataDirScope() { g_onnx_external_data_dir = previous; }
    } _scope(external_data_dir_);

    tenzor_onnx::ModelProto proto;
    // ParseFromArray takes the byte count as `int`. Narrowing a >= 2 GB buffer
    // would wrap to a negative/truncated length: 2-4 GB yields a confusing
    // "not valid ONNX" failure and >4 GB silently parses only a truncated
    // prefix (dropping later initializers/nodes). Reject up front rather than
    // mis-parse a large but legitimate model.
    if (bytes.size() > static_cast<size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error(
            "ONNXImporter::parse_model: model too large for in-memory protobuf "
            "parse (exceeds INT_MAX bytes); use external_data for large weights");
    }
    if (!proto.ParseFromArray(bytes.data(), static_cast<int>(bytes.size()))) {
        throw std::runtime_error(
            "ONNXImporter::parse_model: failed to deserialize ModelProto — "
            "input is not a valid ONNX/protobuf message");
    }

    ONNXModelData model;
    model.ir_version     = static_cast<int64_t>(proto.ir_version());
    model.model_version  = static_cast<int64_t>(proto.model_version());
    model.producer_name  = proto.producer_name();
    model.doc_string     = proto.doc_string();

    // opset_version: pick the first (and usually only) ai.onnx opset entry.
    // Multi-domain opsets are collapsed to a single int, matching the
    // pre-migration behaviour.
    for (const auto& opset : proto.opset_import()) {
        if (opset.domain().empty() || opset.domain() == "ai.onnx") {
            model.opset_version = static_cast<int64_t>(opset.version());
            break;
        }
    }

    if (proto.has_graph()) {
        model.graph = proto_to_ir_graph(proto.graph());
    }
    return model;
#else
    (void)bytes;
    throw std::runtime_error(
        "ONNXImporter::parse_model: built without protobuf support. "
        "Reconfigure with libprotobuf-dev installed and rebuild tenzor_core.");
#endif
}

auto ONNXImporter::validate_model(const ONNXModelData& model) -> void {
    if (model.ir_version < 3) {
        throw std::runtime_error("Unsupported ONNX IR version: " + std::to_string(model.ir_version));
    }

    if (model.opset_version < 13) {
        log("Warning: ONNX opset version " + std::to_string(model.opset_version) +
            " is older than recommended version 13. Some operators may not be supported.");
    }

    if (model.graph.nodes.empty()) {
        throw std::runtime_error("ONNX model has no operations");
    }

    log("Model validation: IR version=" + std::to_string(model.ir_version) +
        ", opset=" + std::to_string(model.opset_version) +
        ", nodes=" + std::to_string(model.graph.nodes.size()) +
        ", initializers=" + std::to_string(model.graph.initializers.size()));
}

auto ONNXImporter::convert_graph(const ONNXGraphData& graph) -> std::shared_ptr<nn::Module> {
    // Retain the host initializer map so control/shape inputs can be read on the
    // host without uploading the constants to the compute device.
    initializers_ptr_ = &graph.initializers;

    // Load initializers (weights, biases)
    load_initializers(graph);

    // Seed graph inputs (runtime activations) with placeholder tensors so node
    // converters that resolve a data input by name during module construction
    // (e.g. BatchNormalization reading its data input) succeed. These are
    // placeholders (zeros, with dynamic/symbolic dims concretised to 1); the
    // built model is fed real inputs at forward() time.
    for (const auto& vi : graph.inputs) {
        if (initializers_ptr_ && initializers_ptr_->count(vi.name)) {
            continue;  // already a constant weight, not a runtime activation
        }
        std::vector<int64_t> ph_shape = vi.shape;
        for (auto& d : ph_shape) {
            if (d <= 0) d = 1;
        }
        context_.register_value(
            vi.name, zeros(ph_shape, onnx_to_dtype(vi.dtype), Device::cpu()));
    }

    // Create sequential container for model
    auto model = std::make_shared<nn::Sequential>();

    // Convert each node
    for (const auto& node : graph.nodes) {
        log("Converting node: " + node.op_type + " (" + node.name + ")");

        try {
            auto module = convert_node(node);
            if (module.has_value()) {
                model->add_module(module.value());
                context_.register_module(node.name, module.value());

                // Register a placeholder output value for this module node so
                // downstream module converters can resolve their data input by
                // name (e.g. Conv with auto_pad=SAME reads X's spatial shape,
                // BatchNormalization picks BatchNorm{1,2,3}d from the input
                // rank). Functional converters already call register_output();
                // module converters return an nn::Module and otherwise never
                // seed the value map, so a Conv->Relu->Conv(SAME) chain would
                // throw "Input tensor not found" and BN would mis-default to
                // rank 4. We materialise the shape by running the freshly-built
                // module on the (already-registered) placeholder input.
                if (!node.outputs.empty() && !node.outputs[0].empty() &&
                    !node.inputs.empty() &&
                    !context_.has_value(node.outputs[0])) {
                    if (auto data_in = context_.get_value(node.inputs[0]);
                        data_in.has_value()) {
                        try {
                            Variable ph_in(data_in.value(), false);
                            auto ph_out = module.value()->forward(ph_in);
                            context_.register_value(node.outputs[0],
                                                    ph_out.tensor());
                        } catch (const std::exception&) {
                            // Best-effort shape propagation: if the placeholder
                            // forward fails (e.g. unsupported placeholder dims),
                            // leave the value unregistered rather than aborting
                            // the import; downstream resolution falls back to
                            // its existing defaults.
                        }
                    }
                }
            }
        } catch (const std::exception& e) {
            throw std::runtime_error(
                "Failed to convert ONNX node '" + node.name +
                "' (op_type=" + node.op_type + "): " + e.what()
            );
        }
    }

    initializers_ptr_ = nullptr;
    return model;
}

auto ONNXImporter::get_host_input(const std::string& name) -> Tensor {
    // Prefer decoding the constant straight from the host proto bytes so it is
    // never uploaded to the compute device (and there is no device->host copy).
    if (initializers_ptr_) {
        auto it = initializers_ptr_->find(name);
        if (it != initializers_ptr_->end()) {
            return it->second.to_tensor(Device::cpu());
        }
    }
    // Non-initializer (computed) control input: its value only exists on the
    // import device, so a host copy is unavoidable to read it for graph
    // construction. This is rare — Slice/Pad/Resize control inputs are almost
    // always graph initializers.
    Tensor t = get_input(name);
    return (t.device().type == Device::Type::CPU) ? t : t.to(Device::cpu());
}

auto ONNXImporter::load_initializers(const ONNXGraphData& graph) -> void {
    log("Loading " + std::to_string(graph.initializers.size()) + " initializer tensors");

    for (const auto& [name, tensor_data] : graph.initializers) {
        auto tensor = tensor_data.to_tensor(device_);
        context_.register_value(name, tensor);
        std::vector<int64_t> shape_vec(tensor.shape().begin(), tensor.shape().end());
        log("  Loaded: " + name + " " + shape_to_string(shape_vec));
    }
}

auto ONNXImporter::convert_node(const ONNXImportNode& node) -> std::optional<std::shared_ptr<nn::Module>> {
    // Validate node arity BEFORE any converter indexes node.inputs/outputs. A
    // malformed (untrusted) model with too few inputs/outputs would otherwise
    // read past the vector (OOB). Default required minimum is (1 input, 1
    // output) — every operator this importer supports consumes at least one
    // input and produces at least one output. Operators with additional
    // *required* operands/results are listed explicitly; optional trailing
    // inputs are not counted (their converters guard on size()).
    {
        static const std::unordered_map<std::string, std::pair<size_t, size_t>> kMinArity = {
            {"Add", {2, 1}},  {"Sub", {2, 1}},   {"Mul", {2, 1}},     {"Div", {2, 1}},
            {"Pow", {2, 1}},  {"MatMul", {2, 1}},{"Gemm", {2, 1}},    {"Where", {3, 1}},
            {"Expand", {2, 1}},{"Tile", {2, 1}}, {"Gather", {2, 1}},  {"Reshape", {2, 1}},
            {"GatherElements", {2, 1}},              // data, indices

            {"Range", {3, 1}},{"LinalgSolve", {2, 1}}, {"TopK", {2, 2}},
            // Converters below subscript node.inputs past index 0 with the
            // unchecked vector operator[]; an under-supplied (untrusted) model
            // would otherwise read OOB. These minimums equal the number of
            // operands each converter actually dereferences.
            {"Slice", {3, 1}},                       // data, starts, ends
            {"Conv", {2, 1}},  {"ConvTranspose", {2, 1}},  // X, W
            {"BatchNormalization", {5, 1}},          // X, scale, bias, mean, var
            {"LayerNormalization", {3, 1}},          // X, scale, bias
            {"InstanceNormalization", {3, 1}},       // X, scale, bias
            {"GroupNormalization", {3, 1}},          // X, scale, bias
            {"LSTM", {3, 1}}, {"GRU", {3, 1}}, {"RNN", {3, 1}},  // X, W, R
            {"QuantizeLinear", {2, 1}},              // x, y_scale
            {"DequantizeLinear", {2, 1}},            // x, x_scale
        };
        auto it = kMinArity.find(node.op_type);
        const size_t min_in = (it != kMinArity.end()) ? it->second.first : 1;
        const size_t min_out = (it != kMinArity.end()) ? it->second.second : 1;
        if (node.inputs.size() < min_in || node.outputs.size() < min_out) {
            throw std::runtime_error(
                "ONNX import: operator '" + node.op_type + "' requires at least " +
                std::to_string(min_in) + " input(s) and " + std::to_string(min_out) +
                " output(s), but the node provides " + std::to_string(node.inputs.size()) +
                " input(s) and " + std::to_string(node.outputs.size()) + " output(s)");
        }
    }

    // Tensor operations (in-place, return nullopt as they modify context)
    if (node.op_type == "Add") {
        convert_add(node);
        return std::nullopt;
    } else if (node.op_type == "Sub") {
        convert_sub(node);
        return std::nullopt;
    } else if (node.op_type == "Mul") {
        convert_mul(node);
        return std::nullopt;
    } else if (node.op_type == "Div") {
        convert_div(node);
        return std::nullopt;
    } else if (node.op_type == "MatMul") {
        convert_matmul(node);
        return std::nullopt;
    } else if (node.op_type == "Reshape") {
        convert_reshape(node);
        return std::nullopt;
    } else if (node.op_type == "Transpose") {
        convert_transpose(node);
        return std::nullopt;
    } else if (node.op_type == "Concat") {
        convert_concat(node);
        return std::nullopt;
    } else if (node.op_type == "Split") {
        convert_split(node);
        return std::nullopt;
    } else if (node.op_type == "Flatten") {
        convert_flatten(node);
        return std::nullopt;
    } else if (node.op_type == "Squeeze") {
        convert_squeeze(node);
        return std::nullopt;
    } else if (node.op_type == "Unsqueeze") {
        convert_unsqueeze(node);
        return std::nullopt;
    } else if (node.op_type == "Slice") {
        convert_slice(node);
        return std::nullopt;
    } else if (node.op_type == "Pad") {
        convert_pad(node);
        return std::nullopt;
    } else if (node.op_type == "Gather") {
        convert_gather(node);
        return std::nullopt;
    } else if (node.op_type == "GatherElements") {
        convert_gather_elements(node);
        return std::nullopt;
    } else if (node.op_type == "Clip") {
        convert_clip(node);
        return std::nullopt;
    } else if (node.op_type == "Cast") {
        convert_cast(node);
        return std::nullopt;
    } else if (node.op_type == "Identity") {
        // Identity is a pure pass-through. The exporter only inserts one to
        // bridge a traced value name to the declared graph-output name, so it
        // contributes no structure to the imported model — skip it. (Module
        // converters track structure via the Sequential, not the value map, so
        // the Identity's input value is intentionally not resolved here.)
        return std::nullopt;
    } else if (node.op_type == "Dropout") {
        convert_dropout(node);
        return std::nullopt;
    } else if (node.op_type == "Resize") {
        convert_resize(node);
        return std::nullopt;
    } else if (node.op_type == "ReduceSum") {
        convert_reduce_sum(node);
        return std::nullopt;
    } else if (node.op_type == "ReduceMean") {
        convert_reduce_mean(node);
        return std::nullopt;
    } else if (node.op_type == "ReduceMax") {
        convert_reduce_max(node);
        return std::nullopt;
    } else if (node.op_type == "ReduceMin") {
        // Audit F.17: ReduceMin/Prod/L1/L2 honour the full axes array.
        convert_reduce_min(node);
        return std::nullopt;
    } else if (node.op_type == "ReduceProd") {
        convert_reduce_prod(node);
        return std::nullopt;
    } else if (node.op_type == "ReduceL1") {
        convert_reduce_l1(node);
        return std::nullopt;
    } else if (node.op_type == "ReduceL2") {
        convert_reduce_l2(node);
        return std::nullopt;
    } else if (node.op_type == "Shape") {
        convert_shape(node);
        return std::nullopt;
    } else if (node.op_type == "ConstantOfShape") {
        convert_constant_of_shape(node);
        return std::nullopt;
    } else if (node.op_type == "Where") {
        convert_where(node);
        return std::nullopt;
    } else if (node.op_type == "Expand") {
        convert_expand(node);
        return std::nullopt;
    } else if (node.op_type == "Pow") {
        convert_pow(node);
        return std::nullopt;
    } else if (node.op_type == "Sqrt") {
        convert_sqrt(node);
        return std::nullopt;
    } else if (node.op_type == "Neg") {
        convert_neg(node);
        return std::nullopt;
    } else if (node.op_type == "Exp") {
        convert_exp(node);
        return std::nullopt;
    } else if (node.op_type == "Log") {
        convert_log(node);
        return std::nullopt;
    }

    // Audit I6: additional ONNX op converters that route through existing
    // tensor ops. Each adds ONNX coverage without new backend kernels.
    else if (node.op_type == "ArgMax") {
        convert_argmax(node);
        return std::nullopt;
    } else if (node.op_type == "ArgMin") {
        convert_argmin(node);
        return std::nullopt;
    } else if (node.op_type == "TopK") {
        convert_topk(node);
        return std::nullopt;
    } else if (node.op_type == "Tile") {
        convert_tile(node);
        return std::nullopt;
    } else if (node.op_type == "Range") {
        convert_range(node);
        return std::nullopt;
    } else if (node.op_type == "NonZero") {
        convert_non_zero(node);
        return std::nullopt;
    } else if (node.op_type == "Round") {
        convert_round(node);
        return std::nullopt;
    } else if (node.op_type == "Einsum") {
        convert_einsum(node);
        return std::nullopt;
    } else if (node.op_type == "Trilu") {
        convert_trilu(node);
        return std::nullopt;
    }

    // Neural network layers (return module)
    else if (node.op_type == "Gemm") {
        return convert_gemm(node);
    } else if (node.op_type == "Conv") {
        return convert_conv(node);
    } else if (node.op_type == "ConvTranspose") {
        return convert_conv_transpose(node);
    } else if (node.op_type == "InstanceNormalization") {
        return convert_instance_normalization(node);
    } else if (node.op_type == "GroupNormalization") {
        return convert_group_normalization(node);
    } else if (node.op_type == "LSTM") {
        return convert_lstm(node);
    } else if (node.op_type == "GRU") {
        return convert_gru(node);
    } else if (node.op_type == "RNN") {
        return convert_rnn(node);
    } else if (node.op_type == "BatchNormalization") {
        return convert_batch_normalization(node);
    } else if (node.op_type == "LayerNormalization") {
        return convert_layer_normalization(node);
    }

    // Activation functions (return module so they join the Sequential model)
    else if (node.op_type == "Relu") {
        return convert_relu(node);
    } else if (node.op_type == "LeakyRelu") {
        return convert_leaky_relu(node);
    } else if (node.op_type == "Sigmoid") {
        return convert_sigmoid(node);
    } else if (node.op_type == "Tanh") {
        return convert_tanh(node);
    } else if (node.op_type == "Gelu") {
        return convert_gelu(node);
    } else if (node.op_type == "Softmax") {
        return convert_softmax(node);
    } else if (node.op_type == "LogSoftmax") {
        return convert_log_softmax(node);
    } else if (node.op_type == "Elu") {
        return convert_elu(node);
    } else if (node.op_type == "Selu") {
        return convert_selu(node);
    }

    // Pooling layers (return module)
    else if (node.op_type == "MaxPool") {
        return convert_maxpool(node);
    } else if (node.op_type == "AveragePool") {
        return convert_avgpool(node);
    } else if (node.op_type == "GlobalAveragePool") {
        return convert_global_avgpool(node);
    }

    // Quantization (QDQ)
    else if (node.op_type == "QuantizeLinear") {
        convert_quantize_linear(node);
        return std::nullopt;
    } else if (node.op_type == "DequantizeLinear") {
        convert_dequantize_linear(node);
        return std::nullopt;
    }

    // Tenzor custom-domain linear-algebra ops. Both export paths
    // (jit_op_type_to_onnx and the OpId visitor) now emit the same "Det" /
    // "Linalg*" op_type strings, so a model exported by either path can be
    // re-imported. Single-output ops are converted directly here; the
    // multi-output factorizations remain export-only (see error below).
    else if (node.op_type == "Det") {
        register_output(node.outputs[0], linalg::det(get_input(node.inputs[0])));
        return std::nullopt;
    } else if (node.op_type == "LinalgInv") {
        register_output(node.outputs[0], linalg::inv(get_input(node.inputs[0])));
        return std::nullopt;
    } else if (node.op_type == "LinalgSolve") {
        register_output(node.outputs[0],
                        linalg::solve(get_input(node.inputs[0]),
                                      get_input(node.inputs[1])));
        return std::nullopt;
    } else if (node.op_type == "LinalgCholesky") {
        bool upper = node.get_attr("upper").value_or(ONNXAttribute{}).get_int(0) != 0;
        register_output(node.outputs[0],
                        linalg::cholesky(get_input(node.inputs[0]), upper));
        return std::nullopt;
    } else if (node.op_type == "LinalgMatrixNorm") {
        register_output(node.outputs[0],
                        linalg::matrix_norm(get_input(node.inputs[0])));
        return std::nullopt;
    } else if (node.op_type == "LinalgSVD" || node.op_type == "LinalgQR" ||
               node.op_type == "LinalgEigh" || node.op_type == "LinalgEigvalsh" ||
               node.op_type == "LinalgSlogdet") {
        throw std::runtime_error(
            "ONNX import: tenzor custom-domain op '" + node.op_type +
            "' is export-only (multi-output factorization); re-importing it is "
            "not yet supported");
    }

    else {
        throw std::runtime_error("Unsupported ONNX operator: " + node.op_type);
    }
}

// ============================================================================
// Tensor Operations
// ============================================================================

auto ONNXImporter::convert_add(const ONNXImportNode& node) -> void {
    auto a = get_input(node.inputs[0]);
    auto b = get_input(node.inputs[1]);
    auto result = add(a, b);
    register_output(node.outputs[0], result);
}

auto ONNXImporter::convert_sub(const ONNXImportNode& node) -> void {
    auto a = get_input(node.inputs[0]);
    auto b = get_input(node.inputs[1]);
    auto result = sub(a, b);
    register_output(node.outputs[0], result);
}

auto ONNXImporter::convert_mul(const ONNXImportNode& node) -> void {
    auto a = get_input(node.inputs[0]);
    auto b = get_input(node.inputs[1]);
    auto result = mul(a, b);
    register_output(node.outputs[0], result);
}

auto ONNXImporter::convert_div(const ONNXImportNode& node) -> void {
    auto a = get_input(node.inputs[0]);
    auto b = get_input(node.inputs[1]);
    auto result = div(a, b);
    register_output(node.outputs[0], result);
}

auto ONNXImporter::convert_matmul(const ONNXImportNode& node) -> void {
    auto a = get_input(node.inputs[0]);
    auto b = get_input(node.inputs[1]);
    auto result = matmul(a, b);
    register_output(node.outputs[0], result);
}

auto ONNXImporter::convert_gemm(const ONNXImportNode& node) -> std::shared_ptr<nn::Module> {
    // ONNX Gemm: Y = alpha * A @ B^T + beta * C
    // Tenzor Linear: Y = X @ W^T + b
    //
    // Tenzor exports Linear(in_features, out_features) as Gemm with
    // alpha=beta=1, transB=1, weight stored as (out, in) on the wire (the
    // native Tenzor layout). ONNX Gemm with transB=1 takes B of shape
    // (N, K) and transposes it to (K, N) before the multiply; for our
    // case N=out, K=in, so X[M, in] @ B^T[in, out] → [M, out]. The wire
    // shape therefore matches Tenzor's (out, in) directly.
    //
    // For transB=0 the stored layout is (in, out) and we must transpose
    // to get Tenzor's (out, in) convention.

    auto weight = get_input(node.inputs[1]);
    std::optional<Tensor> bias;
    if (node.inputs.size() > 2) {
        bias = get_input(node.inputs[2]);
    }

    float alpha    = node.get_attr("alpha").value_or(ONNXAttribute{}).get_float(1.0f);
    float beta     = node.get_attr("beta").value_or(ONNXAttribute{}).get_float(1.0f);
    int64_t transB = node.get_attr("transB").value_or(ONNXAttribute{}).get_int(0);
    int64_t transA = node.get_attr("transA").value_or(ONNXAttribute{}).get_int(0);

    // ONNX Gemm computes Y = α*(A_eff @ B_eff) + β*C where A_eff = A^T when
    // transA=1. A is the runtime activation, so transA cannot be folded into
    // the Linear weights/bias the way transB and α/β can. Mapping a transA=1
    // Gemm onto nn::Linear (Y = X @ W^T + b) would silently use A untransposed
    // and produce wrong math. Reject rather than ignore.
    if (transA != 0) {
        throw std::runtime_error(
            "ONNX Gemm: transA=1 not representable through nn::Linear "
            "(the activation A would need a runtime transpose). Re-export "
            "the graph with transA=0 or insert an explicit Transpose.");
    }

    // Audit I4: real `alpha * (A @ B) + beta * C` support by folding the
    // scalars into the Linear weights/bias at import time. Previously this
    // threw for any α≠1 or β≠1; many real models (Gemm-after-GEMM fusion,
    // attention output projections in optimized ONNX exports) carry these.
    //
    // Tenzor's Linear computes `y = x @ W^T + b`. ONNX Gemm computes
    // `y = α * (A @ B_eff) + β * C` where B_eff = B^T when transB=1, else B.
    // Folding: setting W = α * B_eff^T and bias = β * C yields exactly
    // `y = α * (A @ B_eff) + β * C` under Linear's `x @ W^T + b`.

    const auto weight_shape = weight.shape();
    if (weight_shape.size() != 2) {
        throw std::runtime_error(
            "ONNX Gemm: weight must be 2D, got ndim=" +
            std::to_string(weight_shape.size()));
    }

    int64_t out_features;
    int64_t in_features;
    Tensor tenzor_weight;
    if (transB == 1) {
        out_features = weight_shape[0];
        in_features  = weight_shape[1];
        tenzor_weight = weight;
    } else {
        in_features  = weight_shape[0];
        out_features = weight_shape[1];
        tenzor_weight = weight.transpose(0, 1);
    }

    // Fold alpha into the weight.
    if (alpha != 1.0f) {
        tenzor_weight = tenzor_weight * static_cast<double>(alpha);
    }

    auto linear = std::make_shared<nn::Linear>(in_features, out_features, bias.has_value());
    linear->weight()->tensor() = tenzor_weight;
    if (bias.has_value()) {
        Tensor scaled_bias = bias.value();
        if (beta != 1.0f) {
            scaled_bias = scaled_bias * static_cast<double>(beta);
        }
        linear->bias()->tensor() = scaled_bias;
    }
    return linear;
}

auto ONNXImporter::convert_reshape(const ONNXImportNode& node) -> void {
    auto input = get_input(node.inputs[0]);
    // Read the shape control tensor on the host: get_input() would return it on
    // the import device, and data<int64_t>() below dereferences the raw storage
    // pointer (a device pointer for a GPU import) on the host -> crash.
    auto shape = get_host_input(node.inputs[1]); // Shape tensor

    // Extract shape values. Per the ONNX Reshape spec, a 0 in the target shape
    // means "copy the corresponding dimension from the input" (unless the
    // allowzero attribute is 1, in which case 0 is literal). The previous code
    // passed 0 through and produced an empty (0-sized) dimension.
    const bool allowzero =
        node.get_attr("allowzero").value_or(ONNXAttribute{}).get_int(0) != 0;
    auto in_shape = input.shape();
    std::vector<int64_t> new_shape;
    const int64_t* shape_data = shape.data<int64_t>();
    for (int64_t i = 0; i < shape.numel(); ++i) {
        int64_t d = shape_data[i];
        if (d == 0 && !allowzero) {
            if (i >= static_cast<int64_t>(in_shape.size())) {
                throw std::runtime_error(
                    "Reshape: target dim " + std::to_string(i) +
                    " is 0 (copy-input) but the input has no matching dimension");
            }
            d = in_shape[i];
        }
        new_shape.push_back(d);
    }

    auto result = input.reshape(new_shape);
    register_output(node.outputs[0], result);
}

auto ONNXImporter::convert_transpose(const ONNXImportNode& node) -> void {
    auto input = get_input(node.inputs[0]);
    auto perm_attr = node.get_attr("perm");

    if (!perm_attr.has_value() || !perm_attr->ints.has_value()) {
        throw std::runtime_error("Transpose node missing 'perm' attribute");
    }

    auto perm = perm_attr->ints.value();
    auto result = input.permute(perm);
    register_output(node.outputs[0], result);
}

auto ONNXImporter::convert_concat(const ONNXImportNode& node) -> void {
    std::vector<Tensor> inputs;
    for (const auto& input_name : node.inputs) {
        inputs.push_back(get_input(input_name));
    }

    auto axis_attr = node.get_attr("axis");
    int64_t axis = axis_attr.has_value() ? axis_attr->get_int(0) : 0;

    auto result = cat(inputs, axis);
    register_output(node.outputs[0], result);
}

auto ONNXImporter::convert_split(const ONNXImportNode& node) -> void {
    auto input = get_input(node.inputs[0]);
    auto axis_attr = node.get_attr("axis");
    int64_t axis = axis_attr.has_value() ? axis_attr->get_int(0) : 0;
    // Normalize and bounds-check axis before it indexes the shape span and the
    // slice() calls below. A negative or out-of-range axis from an untrusted
    // model otherwise reads past the shape vector.
    const int64_t split_rank = input.ndim();
    if (axis < 0) axis += split_rank;
    if (axis < 0 || axis >= split_rank) {
        throw std::runtime_error(
            "ONNX Split: axis " + std::to_string(axis) +
            " out of range for input rank " + std::to_string(split_rank));
    }

    auto split_attr = node.get_attr("split");
    std::vector<int64_t> split_sizes;
    if (split_attr.has_value() && split_attr->ints.has_value()) {
        // Opset-1..12: `split` is an attribute.
        split_sizes = split_attr->ints.value();
    } else if (node.inputs.size() > 1 && !node.inputs[1].empty()) {
        // Opset-13+: `split` migrated from attribute to a second (host) input.
        auto split_t = get_host_input(node.inputs[1]).to(DType::Int64);
        const int64_t* p = split_t.data<int64_t>();
        split_sizes.assign(p, p + split_t.numel());
    } else {
        // Equal splits. Per the ONNX Split spec, when the axis length is not
        // evenly divisible by num_outputs the remainder goes to the LAST chunk;
        // assigning the floor size to every output would silently drop the
        // trailing (dim - floor*num_outputs) elements.
        int64_t num_outputs = static_cast<int64_t>(node.outputs.size());
        if (num_outputs <= 0) {
            throw std::runtime_error("ONNX Split: node has no outputs");
        }
        int64_t dim = input.shape()[axis];
        // ONNX/torch equal split uses CEIL-sized leading chunks (the last chunk
        // takes the smaller remainder), not floor-with-remainder-to-last. e.g.
        // dim=7,n=3 -> [3,3,1], not [2,2,3].
        int64_t base = (dim + num_outputs - 1) / num_outputs;
        split_sizes.assign(num_outputs - 1, base);
        split_sizes.push_back(std::max<int64_t>(0, dim - base * (num_outputs - 1)));
    }

    // The number of split chunks must match the declared output count, else
    // the registration loop below would index node.outputs out of bounds with
    // an attacker-controlled 'split' attribute length.
    if (split_sizes.size() != node.outputs.size()) {
        throw std::runtime_error(
            "ONNX Split: 'split' length (" + std::to_string(split_sizes.size()) +
            ") does not match number of outputs (" +
            std::to_string(node.outputs.size()) + ")");
    }

    // Manually split the tensor using slice
    std::vector<Tensor> results;
    int64_t offset = 0;
    for (int64_t size : split_sizes) {
        results.push_back(input.slice(axis, offset, offset + size));
        offset += size;
    }

    for (size_t i = 0; i < results.size(); ++i) {
        register_output(node.outputs[i], results[i]);
    }
}

auto ONNXImporter::convert_flatten(const ONNXImportNode& node) -> void {
    auto input = get_input(node.inputs[0]);
    auto axis_attr = node.get_attr("axis");
    int64_t axis = axis_attr.has_value() ? axis_attr->get_int(1) : 1;

    auto result = input.flatten(axis);
    register_output(node.outputs[0], result);
}

// ============================================================================
// Neural Network Layers
// ============================================================================

// Load weight/bias tensors into a freshly-constructed Conv* module.
// Separated out since every Conv/ConvTranspose path does the same thing.
namespace {
auto load_conv_params(nn::Module& m, const Tensor& weight,
                      const std::optional<Tensor>& bias) -> void {
    auto params = m.named_parameters();
    for (auto& [name, param] : params) {
        if (name == "weight") {
            param->tensor() = weight;
        } else if (name == "bias" && bias.has_value()) {
            param->tensor() = bias.value();
        }
    }
}

// ONNX "pads" layout for an N-D spatial op is
// [begin_d0, begin_d1, ..., begin_dN-1, end_d0, end_d1, ..., end_dN-1].
// Returns true when every begin[i] == end[i] (symmetric).
auto pads_are_symmetric(const std::vector<int64_t>& pads, size_t ndim) -> bool {
    if (pads.size() != ndim * 2) return false;
    for (size_t i = 0; i < ndim; ++i) {
        if (pads[i] != pads[i + ndim]) return false;
    }
    return true;
}

// Validate that a per-axis attribute vector (strides/dilations/kernel_shape/
// output_padding) has EXACTLY the expected number of entries before any axis
// is indexed. ONNXAttribute::get_ints returns the attacker-supplied vector
// verbatim whenever the attribute is present (the size-correct default is used
// only when absent), so a crafted model declaring e.g. strides=[1] on a 2-D
// conv would otherwise drive an out-of-bounds std::vector::operator[] read.
inline void require_axis_count(const std::vector<int64_t>& v, size_t expected,
                               const char* op, const char* attr) {
    if (v.size() != expected) {
        throw std::runtime_error(std::string("ONNX ") + op + ": " + attr +
                                 " attribute has " + std::to_string(v.size()) +
                                 " entries, expected " + std::to_string(expected));
    }
}
}  // namespace

auto ONNXImporter::convert_conv(const ONNXImportNode& node) -> std::shared_ptr<nn::Module> {
    auto weight = get_input(node.inputs[1]);
    std::optional<Tensor> bias;
    if (node.inputs.size() > 2) {
        bias = get_input(node.inputs[2]);
    }

    // Weight shape: [out_channels, in_channels/groups, *kernel]
    auto weight_shape = weight.shape();
    size_t spatial_dims = weight_shape.size() - 2;
    if (spatial_dims < 1 || spatial_dims > 3) {
        throw std::runtime_error("ONNX Conv: unsupported weight rank " +
                                 std::to_string(weight_shape.size()));
    }

    // kernel_shape is optional in ONNX — when omitted it is inferred from the
    // weight tensor's spatial dimensions. Default to those instead of derefing
    // a possibly-empty optional.
    std::vector<int64_t> kernel_from_weight;
    for (size_t i = weight_shape.size() - spatial_dims; i < weight_shape.size(); ++i) {
        kernel_from_weight.push_back(weight_shape[i]);
    }
    auto kernel_shape = node.get_attr("kernel_shape")
                            .value_or(ONNXAttribute{})
                            .get_ints(kernel_from_weight);
    std::vector<int64_t> default_ones(spatial_dims, 1);
    std::vector<int64_t> default_pads(spatial_dims * 2, 0);
    auto strides = node.get_attr("strides").value_or(ONNXAttribute{}).get_ints(default_ones);
    auto pads = node.get_attr("pads").value_or(ONNXAttribute{}).get_ints(default_pads);
    auto dilations = node.get_attr("dilations").value_or(ONNXAttribute{}).get_ints(default_ones);
    int64_t groups = node.get_attr("group").value_or(ONNXAttribute{}).get_int(1);

    // Reject per-axis attribute vectors of the wrong length before any axis is
    // indexed below (kernel_shape[i]/strides[i]/dilations[i]). pads is checked
    // separately (it must be spatial_dims*2).
    require_axis_count(kernel_shape, spatial_dims, "Conv", "kernel_shape");
    require_axis_count(strides, spatial_dims, "Conv", "strides");
    require_axis_count(dilations, spatial_dims, "Conv", "dilations");

    // auto_pad: Wave Inf-C1.
    //   NOTSET                  → use `pads` as-is.
    //   VALID                   → zero padding.
    //   SAME_UPPER / SAME_LOWER → per ONNX spec: ceil(in_dim/stride) outputs;
    //     extra padding goes after (UPPER) or before (LOWER) when total pad
    //     is odd. Resolved at import time using the input tensor's spatial
    //     shape (registered upstream by topological order).
    auto auto_pad = node.get_attr("auto_pad").value_or(ONNXAttribute{}).get_string("NOTSET");
    if (auto_pad == "VALID") {
        pads.assign(spatial_dims * 2, 0);
    } else if (auto_pad == "SAME_UPPER" || auto_pad == "SAME_LOWER") {
        // Need input X's spatial dims.
        Tensor x_in = get_input(node.inputs[0]);
        auto x_shape = x_in.shape();
        if (x_shape.size() != spatial_dims + 2) {
            throw std::runtime_error("ONNX Conv: input has rank " +
                std::to_string(x_shape.size()) + ", expected " +
                std::to_string(spatial_dims + 2) + " (N, C, *spatial).");
        }
        bool same_upper = (auto_pad == "SAME_UPPER");
        pads.resize(spatial_dims * 2);
        for (size_t i = 0; i < spatial_dims; ++i) {
            int64_t in_dim  = x_shape[i + 2];
            int64_t k       = kernel_shape[i];
            int64_t s       = strides[i];
            int64_t d       = dilations[i];
            if (in_dim < 0) {
                throw std::runtime_error("ONNX Conv: auto_pad=" + auto_pad +
                    " requires concrete input spatial dim, got dynamic for dim " +
                    std::to_string(i + 2));
            }
            int64_t out_dim = (in_dim + s - 1) / s;  // ceil(in_dim / stride)
            int64_t total_pad = (out_dim - 1) * s + d * (k - 1) + 1 - in_dim;
            if (total_pad < 0) total_pad = 0;
            int64_t pad_lo, pad_hi;
            if (same_upper) {
                pad_lo = total_pad / 2;
                pad_hi = total_pad - pad_lo;
            } else {  // SAME_LOWER
                pad_hi = total_pad / 2;
                pad_lo = total_pad - pad_hi;
            }
            pads[i]                  = pad_lo;
            pads[i + spatial_dims]   = pad_hi;
        }
    } else if (auto_pad != "NOTSET") {
        throw std::runtime_error("ONNX Conv: auto_pad=" + auto_pad +
                                 " not recognized (expected NOTSET/VALID/SAME_UPPER/SAME_LOWER).");
    }
    if (pads.size() != spatial_dims * 2) {
        throw std::runtime_error("ONNX Conv: pads attribute has " +
                                 std::to_string(pads.size()) + " entries, expected " +
                                 std::to_string(spatial_dims * 2));
    }

    int64_t out_channels = weight_shape[0];
    int64_t in_channels = weight_shape[1] * groups;

    bool symmetric = pads_are_symmetric(pads, spatial_dims);

    // Build the Conv module, using zero padding when we need an explicit
    // ConstantPad prefix for asymmetric pads.
    std::shared_ptr<nn::Module> conv;
    std::shared_ptr<nn::Module> pre_pad;

    if (spatial_dims == 1) {
        int64_t conv_pad = symmetric ? pads[0] : 0;
        if (!symmetric) {
            pre_pad = std::make_shared<nn::ConstantPad1d>(pads[0], pads[1]);
        }
        conv = std::make_shared<nn::Conv1d>(
            in_channels, out_channels, kernel_shape[0],
            strides[0], conv_pad, dilations[0], groups, bias.has_value());
    } else if (spatial_dims == 2) {
        std::pair<int64_t, int64_t> kpair = {kernel_shape[0], kernel_shape[1]};
        std::pair<int64_t, int64_t> spair = {strides[0], strides[1]};
        std::pair<int64_t, int64_t> dpair = {dilations[0], dilations[1]};
        std::pair<int64_t, int64_t> ppair = symmetric
            ? std::pair<int64_t, int64_t>{pads[0], pads[1]}
            : std::pair<int64_t, int64_t>{0, 0};
        if (!symmetric) {
            // ONNX 2-D pads layout: [begin_h, begin_w, end_h, end_w]
            pre_pad = std::make_shared<nn::ConstantPad2d>(
                /*pad_left=*/pads[1], /*pad_right=*/pads[3],
                /*pad_top=*/pads[0], /*pad_bottom=*/pads[2]);
        }
        conv = std::make_shared<nn::Conv2d>(
            in_channels, out_channels, kpair, spair, ppair, dpair,
            groups, bias.has_value());
    } else {  // spatial_dims == 3
        // Audit I5: Conv3d supports per-axis kernel/stride/padding/dilation.
        // Any SYMMETRIC padding (begin==end on every axis) — including
        // symmetric-but-anisotropic values like [1,2,2,1,2,2] — flows through
        // Conv3d's per-axis padding ctor directly, mirroring the 2-D path.
        // Only truly asymmetric pads (begin!=end on some axis) need a
        // ConstantPad3d pre-pad with its extra allocation/copy per forward.
        int64_t conv_pD = symmetric ? pads[0] : 0;
        int64_t conv_pH = symmetric ? pads[1] : 0;
        int64_t conv_pW = symmetric ? pads[2] : 0;
        if (!symmetric) {
            pre_pad = std::make_shared<nn::ConstantPad3d>(std::vector<int64_t>{
                /*left=*/pads[2], /*right=*/pads[5],
                /*top=*/pads[1], /*bottom=*/pads[4],
                /*front=*/pads[0], /*back=*/pads[3]});
        }
        conv = std::make_shared<nn::Conv3d>(
            in_channels, out_channels,
            std::make_tuple(kernel_shape[0], kernel_shape[1], kernel_shape[2]),
            std::make_tuple(strides[0], strides[1], strides[2]),
            std::make_tuple(conv_pD, conv_pH, conv_pW),
            std::make_tuple(dilations[0], dilations[1], dilations[2]),
            groups, bias.has_value());
    }

    load_conv_params(*conv, weight, bias);

    if (pre_pad) {
        auto seq = std::make_shared<nn::Sequential>();
        seq->add_module(pre_pad);
        seq->add_module(conv);
        return seq;
    }
    return conv;
}

auto ONNXImporter::convert_conv_transpose(const ONNXImportNode& node)
    -> std::shared_ptr<nn::Module> {
    auto weight = get_input(node.inputs[1]);
    std::optional<Tensor> bias;
    if (node.inputs.size() > 2) {
        bias = get_input(node.inputs[2]);
    }

    // ConvTranspose weight layout: [in_channels, out_channels/groups, *kernel]
    auto weight_shape = weight.shape();
    size_t spatial_dims = weight_shape.size() - 2;
    if (spatial_dims < 1 || spatial_dims > 3) {
        throw std::runtime_error("ONNX ConvTranspose: unsupported weight rank " +
                                 std::to_string(weight_shape.size()));
    }

    // kernel_shape is optional in ONNX; when omitted, infer it from the
    // weight's trailing spatial dims ([in, out/groups, *kernel]).
    std::vector<int64_t> kernel_shape;
    if (auto ks = node.get_attr("kernel_shape")) {
        kernel_shape = ks->get_ints();
    } else {
        kernel_shape.assign(weight_shape.begin() + 2, weight_shape.end());
    }
    std::vector<int64_t> default_ones(spatial_dims, 1);
    std::vector<int64_t> default_zeros(spatial_dims, 0);
    std::vector<int64_t> default_pads(spatial_dims * 2, 0);
    auto strides = node.get_attr("strides").value_or(ONNXAttribute{}).get_ints(default_ones);
    auto pads = node.get_attr("pads").value_or(ONNXAttribute{}).get_ints(default_pads);
    auto dilations = node.get_attr("dilations").value_or(ONNXAttribute{}).get_ints(default_ones);
    auto output_padding = node.get_attr("output_padding")
                              .value_or(ONNXAttribute{}).get_ints(default_zeros);
    int64_t groups = node.get_attr("group").value_or(ONNXAttribute{}).get_int(1);

    // Reject per-axis attribute vectors of the wrong length before any axis is
    // indexed (kernel_shape[i]/strides[i]/dilations[i]/output_padding[i]).
    // kernel_shape from get_ints() has no size default, so it is the most
    // dangerous; validate all four against spatial_dims.
    require_axis_count(kernel_shape, spatial_dims, "ConvTranspose", "kernel_shape");
    require_axis_count(strides, spatial_dims, "ConvTranspose", "strides");
    require_axis_count(dilations, spatial_dims, "ConvTranspose", "dilations");
    require_axis_count(output_padding, spatial_dims, "ConvTranspose", "output_padding");

    // Wave Inf-C1: auto_pad — NOTSET/VALID/SAME_UPPER/SAME_LOWER all supported.
    // For ConvTranspose with SAME_*, the ONNX spec computes total_pad such that
    // out = in * stride; total_pad is then distributed UPPER (extra after) or
    // LOWER (extra before). Requires concrete input spatial dims.
    auto auto_pad = node.get_attr("auto_pad").value_or(ONNXAttribute{}).get_string("NOTSET");
    if (auto_pad == "VALID") {
        pads.assign(spatial_dims * 2, 0);
    } else if (auto_pad == "SAME_UPPER" || auto_pad == "SAME_LOWER") {
        Tensor x_in = get_input(node.inputs[0]);
        auto x_shape = x_in.shape();
        if (x_shape.size() != spatial_dims + 2) {
            throw std::runtime_error("ONNX ConvTranspose: input has rank " +
                std::to_string(x_shape.size()) + ", expected " +
                std::to_string(spatial_dims + 2) + " (N, C, *spatial).");
        }
        bool same_upper = (auto_pad == "SAME_UPPER");
        pads.resize(spatial_dims * 2);
        for (size_t i = 0; i < spatial_dims; ++i) {
            int64_t in_dim  = x_shape[i + 2];
            int64_t k       = kernel_shape[i];
            int64_t s       = strides[i];
            int64_t d       = dilations[i];
            int64_t op      = (i < output_padding.size()) ? output_padding[i] : 0;
            if (in_dim < 0) {
                throw std::runtime_error("ONNX ConvTranspose: auto_pad=" + auto_pad +
                    " requires concrete input spatial dim, got dynamic for dim " +
                    std::to_string(i + 2));
            }
            // For ConvTranspose (per ONNX spec):
            //   total_pad = stride*(in_dim-1) + output_padding + ((kernel-1)*dilation+1) - (in_dim*stride)
            //             = output_padding + (kernel-1)*dilation + 1 - stride
            // We pick UPPER vs LOWER for the asymmetric remainder.
            int64_t total_pad = op + (k - 1) * d + 1 - s;
            if (total_pad < 0) total_pad = 0;
            int64_t pad_lo, pad_hi;
            if (same_upper) {
                pad_lo = total_pad / 2;
                pad_hi = total_pad - pad_lo;
            } else {
                pad_hi = total_pad / 2;
                pad_lo = total_pad - pad_hi;
            }
            pads[i]                = pad_lo;
            pads[i + spatial_dims] = pad_hi;
        }
    } else if (auto_pad != "NOTSET") {
        throw std::runtime_error("ONNX ConvTranspose: auto_pad=" + auto_pad +
                                 " not recognized (expected NOTSET/VALID/SAME_UPPER/SAME_LOWER).");
    }

    if (!pads_are_symmetric(pads, spatial_dims)) {
        throw std::runtime_error(
            "ONNX ConvTranspose: asymmetric pads are not representable through "
            "Tenzor ConvTranspose. Re-export with symmetric padding.");
    }

    int64_t in_channels = weight_shape[0];
    int64_t out_channels = weight_shape[1] * groups;

    // Audit I5: ConvTranspose1d still scalar (1-D has no anisotropy). For
    // ConvTranspose2d and ConvTranspose3d, the old "anisotropic not supported"
    // guards are removed — the modules now accept per-axis std::pair / tuple
    // ctors. Output_padding's default-empty case is handled per axis.
    auto get_axis = [&](const std::vector<int64_t>& v, size_t i, int64_t def) {
        return i < v.size() ? v[i] : def;
    };

    std::shared_ptr<nn::Module> conv;
    if (spatial_dims == 1) {
        // ConvTranspose1d: only one spatial axis. Use index 0 throughout.
        // Audit F.17: dilation is now supported in ConvTranspose1d (passed
        // through to the underlying ConvTranspose2d kernel's W axis).
        int64_t k = kernel_shape[0], s = strides[0], d = dilations[0];
        int64_t p = pads[0];
        int64_t op = output_padding.empty() ? 0 : output_padding[0];
        conv = std::make_shared<nn::ConvTranspose1d>(
            in_channels, out_channels, k, s, p, op, groups, bias.has_value(), d);
    } else if (spatial_dims == 2) {
        conv = std::make_shared<nn::ConvTranspose2d>(
            in_channels, out_channels,
            std::make_pair(kernel_shape[0], kernel_shape[1]),
            std::make_pair(strides[0],      strides[1]),
            std::make_pair(pads[0],         pads[1]),
            std::make_pair(get_axis(output_padding, 0, 0),
                           get_axis(output_padding, 1, 0)),
            std::make_pair(dilations[0],    dilations[1]),
            groups, bias.has_value());
    } else {  // spatial_dims == 3
        conv = std::make_shared<nn::ConvTranspose3d>(
            in_channels, out_channels,
            std::make_tuple(kernel_shape[0], kernel_shape[1], kernel_shape[2]),
            std::make_tuple(strides[0],      strides[1],      strides[2]),
            std::make_tuple(pads[0],         pads[1],         pads[2]),
            std::make_tuple(get_axis(output_padding, 0, 0),
                           get_axis(output_padding, 1, 0),
                           get_axis(output_padding, 2, 0)),
            std::make_tuple(dilations[0],    dilations[1],    dilations[2]),
            groups, bias.has_value());
    }

    load_conv_params(*conv, weight, bias);
    return conv;
}

auto ONNXImporter::convert_layer_normalization(const ONNXImportNode& node) -> std::shared_ptr<nn::Module> {
    // ONNX LayerNormalization inputs: X, Scale, Bias (bias optional per spec;
    // we only support the bias-present form the exporter emits).
    auto scale = get_input(node.inputs[1]);
    auto bias  = get_input(node.inputs[2]);
    float eps  = node.get_attr("epsilon").value_or(ONNXAttribute{}).get_float(1e-5f);

    // Scale shape is the normalized_shape. Our LayerNorm constructor takes
    // that as a vector of ints.
    std::vector<int64_t> normalized_shape(scale.shape().begin(), scale.shape().end());
    auto ln = std::make_shared<nn::LayerNorm>(normalized_shape, static_cast<double>(eps),
                                              /*elementwise_affine=*/true);

    auto params = ln->named_parameters();
    for (auto& [name, param] : params) {
        if (name == "weight") param->tensor() = scale;
        else if (name == "bias") param->tensor() = bias;
    }
    return ln;
}

auto ONNXImporter::convert_batch_normalization(const ONNXImportNode& node) -> std::shared_ptr<nn::Module> {
    auto scale = get_input(node.inputs[1]);  // gamma
    auto bias = get_input(node.inputs[2]);   // beta
    auto mean = get_input(node.inputs[3]);   // running mean
    auto var = get_input(node.inputs[4]);    // running variance

    float eps = node.get_attr("epsilon").value_or(ONNXAttribute{}).get_float(1e-5f);

    if (scale.shape().empty()) {
        throw std::runtime_error(
            "BatchNormalization: scale (gamma) must be at least 1-D");
    }
    int64_t num_features = scale.shape()[0];

    // ONNX uses a single BatchNormalization op for all ranks; pick the Tenzor
    // layer from the data input's rank: rank 3 -> BatchNorm1d (N,C,L),
    // rank 4 -> BatchNorm2d (N,C,H,W), rank 5 -> BatchNorm3d (N,C,D,H,W).
    // Default to 2d (most common in CNNs) when the data input's rank is
    // unavailable — e.g. the graph input lacks a declared shape/value-info, in
    // which case we must NOT hard-fail on a missing activation tensor.
    size_t input_rank = 4;
    if (auto data_in = context_.get_value(node.inputs[0]); data_in.has_value()) {
        input_rank = data_in->shape().size();
    }
    std::shared_ptr<nn::Module> bn;
    if (input_rank == 3) {
        bn = std::make_shared<nn::BatchNorm1d>(num_features, static_cast<double>(eps));
    } else if (input_rank == 5) {
        bn = std::make_shared<nn::BatchNorm3d>(num_features, static_cast<double>(eps));
    } else {
        bn = std::make_shared<nn::BatchNorm2d>(num_features, static_cast<double>(eps));
    }

    // Load pretrained parameters from ONNX
    auto params = bn->named_parameters();
    for (auto& [name, param] : params) {
        if (name == "weight") {
            param->tensor() = scale;
        } else if (name == "bias") {
            param->tensor() = bias;
        }
    }

    // Load running statistics from ONNX buffers
    auto buffers = bn->named_buffers();
    for (auto& [name, buffer] : buffers) {
        if (name == "running_mean") {
            buffer->tensor() = mean;
        } else if (name == "running_var") {
            buffer->tensor() = var;
        }
    }

    return bn;
}

auto ONNXImporter::convert_instance_normalization(const ONNXImportNode& node) -> std::shared_ptr<nn::Module> {
    // I6-followup: ONNX InstanceNormalization inputs: X, scale, bias.
    // ONNX spec: per-channel affine, no running stats (training=False).
    // Tenzor's nn::InstanceNorm2d covers the (N, C, H, W) case which is by
    // far the most common (image / convolutional inputs).
    auto scale = get_input(node.inputs[1]);
    auto bias  = get_input(node.inputs[2]);
    float eps  = node.get_attr("epsilon").value_or(ONNXAttribute{}).get_float(1e-5f);

    int64_t num_channels = scale.shape()[0];
    auto in_layer = std::make_shared<nn::InstanceNorm2d>(num_channels,
                                                          static_cast<double>(eps),
                                                          /*affine=*/true);

    auto params = in_layer->named_parameters();
    for (auto& [name, param] : params) {
        if (name == "weight") param->tensor() = scale;
        else if (name == "bias") param->tensor() = bias;
    }
    return in_layer;
}

auto ONNXImporter::convert_group_normalization(const ONNXImportNode& node) -> std::shared_ptr<nn::Module> {
    // I6-followup: ONNX GroupNormalization (opset 18+) inputs: X, scale, bias.
    // Required attr: num_groups. Optional: epsilon.
    auto scale = get_input(node.inputs[1]);
    auto bias  = get_input(node.inputs[2]);
    auto ng_attr = node.get_attr("num_groups");
    if (!ng_attr) {
        throw std::runtime_error("ONNX GroupNormalization: missing required attribute `num_groups`.");
    }
    int64_t num_groups = ng_attr->get_int(1);
    float eps = node.get_attr("epsilon").value_or(ONNXAttribute{}).get_float(1e-5f);

    int64_t num_channels = scale.shape()[0];
    auto gn = std::make_shared<nn::GroupNorm>(num_groups, num_channels,
                                               static_cast<double>(eps),
                                               /*affine=*/true);

    auto params = gn->named_parameters();
    for (auto& [name, param] : params) {
        if (name == "weight") param->tensor() = scale;
        else if (name == "bias") param->tensor() = bias;
    }
    return gn;
}

namespace {
// Helper: validate the RNN-family ONNX op's W/R shapes match the
// single-layer / unidirectional case we currently support, and return the
// (num_directions, hidden_size, input_size) triple. Multi-layer + bidirectional
// require a per-layer weight split that's tracked as a separate followup —
// most exported ONNX RNNs are single-layer-unidirectional.
struct RnnDims {
    int64_t num_directions;
    int64_t hidden_size;
    int64_t input_size;
};
RnnDims onnx_rnn_dims(const Tensor& W, int64_t gates) {
    auto ws = W.shape();
    if (ws.size() != 3) {
        throw std::runtime_error(
            "ONNX RNN/LSTM/GRU: weight W must be 3D [num_directions, gates*hidden, input_size], got rank " +
            std::to_string(ws.size()));
    }
    int64_t num_directions = ws[0];
    int64_t hidden_size = ws[1] / gates;
    int64_t input_size = ws[2];
    if (hidden_size * gates != ws[1]) {
        throw std::runtime_error(
            "ONNX RNN/LSTM/GRU: weight W's dim 1 (" + std::to_string(ws[1]) +
            ") is not divisible by the expected gate count (" + std::to_string(gates) + ").");
    }
    return {num_directions, hidden_size, input_size};
}

// I6 weight remap helper: given an ONNX RNN-family weight tensor of shape
// [num_directions, gates*H, ...] sliced to the forward direction, reorder the
// gate slots according to `perm`. Returns a (gates*H, ...) 2D tensor in
// tenzor's gate order. `perm[k]` = ONNX slot index that supplies tenzor slot k.
//
//   LSTM:  ONNX [i, o, f, c] → tenzor [i, f, g, o]   perm = {0, 2, 3, 1}
//   GRU:   ONNX [z, r, h]    → tenzor [r, z, n]      perm = {1, 0, 2}
//   RNN:   single weight                              perm = {0}
//
// The remap composes via slice + cat; no kernel-level work needed.
Tensor reorder_rnn_gates(const Tensor& w_2d, int64_t hidden, const std::vector<int64_t>& perm) {
    if (perm.size() == 1 && perm[0] == 0) {
        return w_2d;  // no-op for vanilla RNN
    }
    std::vector<Tensor> chunks;
    chunks.reserve(perm.size());
    for (int64_t k = 0; k < static_cast<int64_t>(perm.size()); ++k) {
        int64_t src = perm[k];
        // Slice rows [src*hidden, (src+1)*hidden) along dim 0.
        chunks.push_back(::tenzor::slice(w_2d, /*dim=*/0,
                                          /*start=*/src * hidden,
                                          /*stop=*/(src + 1) * hidden));
    }
    return ::tenzor::cat(chunks, /*dim=*/0);
}

// Combine ONNX W-bias + R-bias (each [gates*H]) into a single bias [gates*H]
// in tenzor's gate order. PyTorch convention has separate bias_ih and bias_hh
// — tenzor's `weight_hh` is constructed without bias, so we collapse the two
// onto `weight_ih.bias`.
//
// For LSTM/vanilla-RNN gates the eval-time math is `act(X·W + Wb + H·R + Rb)`,
// so `Wb + Rb` is an exact equivalent and we simply sum.
//
// GRU is different. Tenzor's GRUCell implements `linear_before_reset=1`
// semantics: `n_t = tanh(n_i + r_t ⊙ (H·Rn))` where `n_i = X·Wn + bias_ih_n`.
// Under that mode the recurrent new-gate bias Rbn belongs INSIDE the reset
// multiply: `r_t ⊙ (H·Rn + Rbn)` — it is NOT equivalent to a pre-summed
// `Wbn + Rbn` on the input side. (Summing remains valid for the z and r
// gates, whose Rb is added before the nonlinearity, not gated by r_t.)
// Tenzor's weight_hh carries no bias, so there is no place to put Rbn inside
// the reset; pre-summing it would silently miscompute the new gate. We
// therefore sum only the valid gates and, for the GRU n-gate, keep only Wbn,
// rejecting a non-zero Rbn rather than producing a wrong result.
//
// `gru_n_gate_onnx_slot` selects the ONNX gate slot that is the GRU new/h
// gate (2 for ONNX [z,r,h]); pass -1 for LSTM/RNN to sum every gate.
Tensor combine_and_reorder_rnn_bias(const Tensor& bias_2H,
                                     int64_t hidden,
                                     int64_t gates,
                                     const std::vector<int64_t>& perm,
                                     int64_t gru_n_gate_onnx_slot = -1) {
    // bias_2H has length 2 * gates * hidden = concat(W_bias, R_bias).
    auto bshape = bias_2H.shape();
    if (bshape.size() != 1 || bshape[0] != 2 * gates * hidden) {
        throw std::runtime_error(
            "ONNX RNN bias: expected 1D length 2*gates*hidden, got rank " +
            std::to_string(bshape.size()) + " size " + std::to_string(bshape[0]));
    }
    Tensor w_bias = ::tenzor::slice(bias_2H, 0, 0,           gates * hidden);
    Tensor r_bias = ::tenzor::slice(bias_2H, 0, gates * hidden, 2 * gates * hidden);
    Tensor summed = ::tenzor::add(w_bias, r_bias);  // bias_ih + bias_hh

    if (gru_n_gate_onnx_slot >= 0) {
        // GRU under linear_before_reset=1: the new/h gate must not pre-sum Rbn.
        const int64_t s = gru_n_gate_onnx_slot;
        // Reject a non-zero recurrent new-gate bias we cannot represent.
        Tensor rbn = ::tenzor::slice(r_bias, 0, s * hidden, (s + 1) * hidden);
        double rbn_l1 =
            ::tenzor::sum(::tenzor::abs(rbn.to(DType::Float64))).item<double>();
        if (rbn_l1 != 0.0) {
            throw std::runtime_error(
                "ONNX GRU import: model has a non-zero recurrent new-gate bias "
                "(Rbh) with linear_before_reset=1, which Tenzor's GRUCell "
                "cannot represent (weight_hh has no bias; Rbh must be applied "
                "inside the reset multiply). Re-export with a zero new-gate Rbh "
                "or with linear_before_reset=0.");
        }
        // Overwrite the new-gate slot of `summed` with Wbn only (Rbn == 0 here,
        // so this matches; the explicit write documents the intent and stays
        // correct even if the zero-check tolerance is ever loosened).
        Tensor wbn = ::tenzor::slice(w_bias, 0, s * hidden, (s + 1) * hidden);
        std::vector<Tensor> gate_chunks;
        gate_chunks.reserve(gates);
        for (int64_t g = 0; g < gates; ++g) {
            if (g == s) {
                gate_chunks.push_back(wbn);
            } else {
                gate_chunks.push_back(
                    ::tenzor::slice(summed, 0, g * hidden, (g + 1) * hidden));
            }
        }
        summed = ::tenzor::cat(gate_chunks, 0);
    }

    if (perm.size() == 1 && perm[0] == 0) {
        return summed;  // vanilla RNN: no remap
    }
    std::vector<Tensor> chunks;
    chunks.reserve(perm.size());
    for (int64_t k = 0; k < static_cast<int64_t>(perm.size()); ++k) {
        int64_t src = perm[k];
        chunks.push_back(::tenzor::slice(summed, 0, src * hidden, (src + 1) * hidden));
    }
    return ::tenzor::cat(chunks, 0);
}

// Load ONNX W, R, optional B into a single-direction LSTM/GRU/RNN module's
// forward cell (and weight_hh). `cell_prefix` selects the parameter-name
// prefix (`"forward_cell_0."` for forward, `"backward_cell_0."` for reverse).
// Returns true if at least one parameter was matched and assigned.
bool load_rnn_direction_weights(nn::Module& module,
                                  const std::string& cell_prefix,
                                  const Tensor& W_dir,  // [gates*H, input]
                                  const Tensor& R_dir,  // [gates*H, hidden]
                                  const Tensor* B_dir,  // [2*gates*H] or null
                                  int64_t hidden,
                                  int64_t gates,
                                  const std::vector<int64_t>& perm,
                                  int64_t gru_n_gate_onnx_slot = -1) {
    Tensor w_ih_t = reorder_rnn_gates(W_dir, hidden, perm);
    Tensor w_hh_t = reorder_rnn_gates(R_dir, hidden, perm);
    Tensor bias_t;
    bool has_bias = (B_dir != nullptr && B_dir->is_valid() && B_dir->numel() > 0);
    if (has_bias) {
        bias_t = combine_and_reorder_rnn_bias(*B_dir, hidden, gates, perm,
                                              gru_n_gate_onnx_slot);
    }

    bool matched_any = false;
    auto params = module.named_parameters();
    for (auto& [name, param] : params) {
        if (name == cell_prefix + "weight_ih.weight") {
            param->tensor() = w_ih_t;  matched_any = true;
        } else if (name == cell_prefix + "weight_hh.weight") {
            param->tensor() = w_hh_t;  matched_any = true;
        } else if (has_bias && name == cell_prefix + "weight_ih.bias") {
            param->tensor() = bias_t;  matched_any = true;
        }
    }
    return matched_any;
}
}  // namespace

auto ONNXImporter::convert_lstm(const ONNXImportNode& node) -> std::shared_ptr<nn::Module> {
    // I6-followup: ONNX LSTM. Inputs (required prefix): X, W, R; optional:
    //   B, sequence_lens, initial_h, initial_c, P (peephole; not supported).
    auto W = get_input(node.inputs[1]);  // [num_directions, 4*hidden, input]
    auto R = get_input(node.inputs[2]);  // [num_directions, 4*hidden, hidden]
    auto dims = onnx_rnn_dims(W, /*gates=*/4);

    bool bidirectional = (dims.num_directions == 2);
    if (dims.num_directions != 1 && dims.num_directions != 2) {
        throw std::runtime_error("ONNX LSTM: num_directions must be 1 or 2, got " +
                                 std::to_string(dims.num_directions));
    }
    bool has_bias = node.inputs.size() > 3 && !node.inputs[3].empty();

    auto lstm = std::make_shared<nn::LSTM>(
        dims.input_size, dims.hidden_size,
        /*num_layers=*/1, /*bias=*/has_bias,
        /*batch_first=*/false,  // ONNX RNN ops use seq-first layout by default
        /*dropout=*/0.0,
        /*bidirectional=*/bidirectional,
        /*proj_size=*/0);

    // Weight loading with ONNX [i, o, f, c] → tenzor [i, f, g, o] gate remap.
    // The permutation `perm[k] = ONNX slot that supplies tenzor slot k`:
    //   tenzor slot 0 (i) ← ONNX slot 0 (i)
    //   tenzor slot 1 (f) ← ONNX slot 2 (f)
    //   tenzor slot 2 (g) ← ONNX slot 3 (c)   // "g" in tenzor == "c" in ONNX
    //   tenzor slot 3 (o) ← ONNX slot 1 (o)
    const std::vector<int64_t> lstm_perm = {0, 2, 3, 1};

    // Forward direction always present.
    Tensor W_fwd = ::tenzor::slice(W, 0, 0, 1);  // [1, 4*H, input]
    W_fwd = ::tenzor::reshape(W_fwd, std::vector<int64_t>{4 * dims.hidden_size, dims.input_size});
    Tensor R_fwd = ::tenzor::slice(R, 0, 0, 1);
    R_fwd = ::tenzor::reshape(R_fwd, std::vector<int64_t>{4 * dims.hidden_size, dims.hidden_size});

    Tensor B_fwd;  // empty if no bias
    Tensor B_full;
    if (has_bias) {
        B_full = get_input(node.inputs[3]);  // [num_directions, 8*hidden]
        B_fwd = ::tenzor::slice(B_full, 0, 0, 1);
        B_fwd = ::tenzor::reshape(B_fwd, std::vector<int64_t>{8 * dims.hidden_size});
    }
    load_rnn_direction_weights(*lstm, "forward_cell_0.", W_fwd, R_fwd,
                                has_bias ? &B_fwd : nullptr,
                                dims.hidden_size, /*gates=*/4, lstm_perm);

    if (bidirectional) {
        Tensor W_bwd = ::tenzor::slice(W, 0, 1, 2);
        W_bwd = ::tenzor::reshape(W_bwd, std::vector<int64_t>{4 * dims.hidden_size, dims.input_size});
        Tensor R_bwd = ::tenzor::slice(R, 0, 1, 2);
        R_bwd = ::tenzor::reshape(R_bwd, std::vector<int64_t>{4 * dims.hidden_size, dims.hidden_size});

        Tensor B_bwd;
        if (has_bias) {
            B_bwd = ::tenzor::slice(B_full, 0, 1, 2);
            B_bwd = ::tenzor::reshape(B_bwd, std::vector<int64_t>{8 * dims.hidden_size});
        }
        load_rnn_direction_weights(*lstm, "backward_cell_0.", W_bwd, R_bwd,
                                    has_bias ? &B_bwd : nullptr,
                                    dims.hidden_size, /*gates=*/4, lstm_perm);
    }

    return lstm;
}

auto ONNXImporter::convert_gru(const ONNXImportNode& node) -> std::shared_ptr<nn::Module> {
    // I6-followup: ONNX GRU. W has shape [num_directions, 3*hidden, input].
    auto W = get_input(node.inputs[1]);
    auto R = get_input(node.inputs[2]);
    auto dims = onnx_rnn_dims(W, /*gates=*/3);

    bool bidirectional = (dims.num_directions == 2);
    if (dims.num_directions != 1 && dims.num_directions != 2) {
        throw std::runtime_error("ONNX GRU: num_directions must be 1 or 2, got " +
                                 std::to_string(dims.num_directions));
    }
    bool has_bias = node.inputs.size() > 3 && !node.inputs[3].empty();

    // ONNX GRU `linear_before_reset` selects the new-gate ("n"/"h")
    // formulation. The ONNX default is 0:
    //   mode 0: n_t = tanh(Xt·Wn + Wbn + (r_t ⊙ Ht-1)·Rn + Rbn)
    //           (reset gate multiplies Ht-1 BEFORE the recurrent matmul; the
    //            recurrent new-gate bias Rbn is a normal post-matmul bias)
    //   mode 1: n_t = tanh(Xt·Wn + Wbn + r_t ⊙ (Ht-1·Rn + Rbn))
    //           (reset gate multiplies the recurrent matmul RESULT)
    // Both modes are now implemented by Tenzor's GRUCell, selected via the
    // linear_before_reset flag. (Default attr value 0 → mode 0 → flag false.)
    int64_t linear_before_reset =
        node.get_attr("linear_before_reset").value_or(ONNXAttribute{}).get_int(0);
    const bool lbr = (linear_before_reset != 0);

    auto gru = std::make_shared<nn::GRU>(
        dims.input_size, dims.hidden_size,
        /*num_layers=*/1, /*bias=*/has_bias,
        /*batch_first=*/false, /*dropout=*/0.0,
        /*bidirectional=*/bidirectional,
        /*linear_before_reset=*/lbr);

    // Gate-order remap: ONNX [z, r, h] → tenzor [r, z, n]. perm[k] = ONNX slot.
    //   tenzor slot 0 (r) ← ONNX slot 1 (r)
    //   tenzor slot 1 (z) ← ONNX slot 0 (z)
    //   tenzor slot 2 (n) ← ONNX slot 2 (h)
    const std::vector<int64_t> gru_perm = {1, 0, 2};

    // Bias handling differs by mode (the recurrent new-gate bias Rbn):
    //
    //   mode 1 (lbr == true): Rbn lives INSIDE the reset multiply,
    //     r_t ⊙ (Ht-1·Rn + Rbn). Tenzor's weight_hh carries no bias (all
    //     biases are collapsed onto weight_ih.bias), so there is nowhere to
    //     place Rbn inside the reset. We therefore keep only Wbn on the n-gate
    //     and reject a non-zero Rbn (combine_and_reorder_rnn_bias does this
    //     when passed the n-gate slot). Mark the ONNX n-gate slot (2 = "h").
    //
    //   mode 0 (lbr == false): Rbn is a normal post-matmul bias,
    //     ... + (r_t ⊙ Ht-1)·Rn + Rbn. Summing Wbn + Rbn onto weight_ih.bias is
    //     exactly equivalent here, because GRUCell (mode 0) adds the recurrent
    //     bias after the matmul and the input-side bias is folded into n_i:
    //       n_t = tanh(Xt·Wn + (Wbn+Rbn) + (r_t ⊙ Ht-1)·Rn)
    //     which matches the ONNX definition. So we sum every gate (slot = -1),
    //     identical to the LSTM/RNN bias handling.
    const int64_t gru_n_gate_slot = lbr ? 2 : -1;

    Tensor W_fwd = ::tenzor::slice(W, 0, 0, 1);
    W_fwd = ::tenzor::reshape(W_fwd, std::vector<int64_t>{3 * dims.hidden_size, dims.input_size});
    Tensor R_fwd = ::tenzor::slice(R, 0, 0, 1);
    R_fwd = ::tenzor::reshape(R_fwd, std::vector<int64_t>{3 * dims.hidden_size, dims.hidden_size});

    Tensor B_fwd;
    Tensor B_full;
    if (has_bias) {
        B_full = get_input(node.inputs[3]);  // [num_directions, 6*hidden]
        B_fwd = ::tenzor::slice(B_full, 0, 0, 1);
        B_fwd = ::tenzor::reshape(B_fwd, std::vector<int64_t>{6 * dims.hidden_size});
    }
    load_rnn_direction_weights(*gru, "forward_cell_0.", W_fwd, R_fwd,
                                has_bias ? &B_fwd : nullptr,
                                dims.hidden_size, /*gates=*/3, gru_perm,
                                gru_n_gate_slot);

    if (bidirectional) {
        Tensor W_bwd = ::tenzor::slice(W, 0, 1, 2);
        W_bwd = ::tenzor::reshape(W_bwd, std::vector<int64_t>{3 * dims.hidden_size, dims.input_size});
        Tensor R_bwd = ::tenzor::slice(R, 0, 1, 2);
        R_bwd = ::tenzor::reshape(R_bwd, std::vector<int64_t>{3 * dims.hidden_size, dims.hidden_size});

        Tensor B_bwd;
        if (has_bias) {
            B_bwd = ::tenzor::slice(B_full, 0, 1, 2);
            B_bwd = ::tenzor::reshape(B_bwd, std::vector<int64_t>{6 * dims.hidden_size});
        }
        load_rnn_direction_weights(*gru, "backward_cell_0.", W_bwd, R_bwd,
                                    has_bias ? &B_bwd : nullptr,
                                    dims.hidden_size, /*gates=*/3, gru_perm,
                                    gru_n_gate_slot);
    }

    return gru;
}

auto ONNXImporter::convert_rnn(const ONNXImportNode& node) -> std::shared_ptr<nn::Module> {
    // I6-followup: ONNX RNN (vanilla). W has shape [num_directions, hidden, input].
    auto W = get_input(node.inputs[1]);
    auto dims = onnx_rnn_dims(W, /*gates=*/1);

    bool bidirectional = (dims.num_directions == 2);
    if (dims.num_directions != 1 && dims.num_directions != 2) {
        throw std::runtime_error("ONNX RNN: num_directions must be 1 or 2, got " +
                                 std::to_string(dims.num_directions));
    }

    // ONNX RNN `activations` is a STRINGS list (one entry per direction). The
    // ONNX spec allows any of the supported activations per direction. Tenzor's
    // RNNCell now accepts the full ONNX activation alphabet, so we pass the
    // chosen activation through directly. For BIDIRECTIONAL RNNs with
    // heterogeneous activations (forward != backward) we build a small
    // BidirectionalRNNAdapter that wraps two single-direction nn::RNN modules.
    //
    // Aliases handled by `apply_rnn_activation` in src/nn/layers/rnn.cpp:
    //   Tanh, Relu, Sigmoid, LeakyRelu, Elu, HardSigmoid, HardTanh, Softsign,
    //   Affine, ScaledTanh.
    auto canonical_act = [](const std::string& raw) -> std::string {
        std::string a = raw;
        for (auto& c : a) c = static_cast<char>(std::tolower(c));
        // Map ONNX spelling to the canonical tenzor identifier.
        if (a == "leakyrelu")     return "leaky_relu";
        if (a == "hardsigmoid")   return "hardsigmoid";
        if (a == "hardtanh")      return "hardtanh";
        if (a == "scaledtanh")    return "scaledtanh";
        return a;
    };

    std::string nonlin = "tanh";
    std::string nonlin_fwd = "tanh";
    std::string nonlin_bwd = "tanh";
    bool heterogeneous = false;
    auto acts = node.get_attr("activations");
    if (acts) {
        std::vector<std::string> act_list;
        if (acts->strings.has_value()) {
            act_list = acts->strings.value();
        } else {
            act_list.push_back(acts->get_string("Tanh"));
        }
        if (!act_list.empty()) {
            nonlin_fwd = canonical_act(act_list[0]);
            nonlin_bwd = (act_list.size() > 1) ? canonical_act(act_list[1]) : nonlin_fwd;
        }
        nonlin = nonlin_fwd;
        heterogeneous = bidirectional && (nonlin_fwd != nonlin_bwd);
    }

    bool has_bias = node.inputs.size() > 3 && !node.inputs[3].empty();
    auto rnn = std::make_shared<nn::RNN>(
        dims.input_size, dims.hidden_size,
        /*num_layers=*/1, nonlin_fwd,
        /*bias=*/has_bias,
        /*batch_first=*/false, /*dropout=*/0.0,
        /*bidirectional=*/bidirectional,
        /*nonlinearity_bwd=*/heterogeneous ? nonlin_bwd : std::string{});
    (void)nonlin;  // retained for clarity; nonlin_fwd is the active value

    // Vanilla RNN: no gate remap (single weight per direction); perm = {0}.
    const std::vector<int64_t> rnn_perm = {0};
    auto R = get_input(node.inputs[2]);

    Tensor W_fwd = ::tenzor::slice(W, 0, 0, 1);
    W_fwd = ::tenzor::reshape(W_fwd, std::vector<int64_t>{dims.hidden_size, dims.input_size});
    Tensor R_fwd = ::tenzor::slice(R, 0, 0, 1);
    R_fwd = ::tenzor::reshape(R_fwd, std::vector<int64_t>{dims.hidden_size, dims.hidden_size});

    Tensor B_fwd;
    Tensor B_full;
    if (has_bias) {
        B_full = get_input(node.inputs[3]);  // [num_directions, 2*hidden]
        B_fwd = ::tenzor::slice(B_full, 0, 0, 1);
        B_fwd = ::tenzor::reshape(B_fwd, std::vector<int64_t>{2 * dims.hidden_size});
    }
    load_rnn_direction_weights(*rnn, "forward_cell_0.", W_fwd, R_fwd,
                                has_bias ? &B_fwd : nullptr,
                                dims.hidden_size, /*gates=*/1, rnn_perm);

    if (bidirectional) {
        Tensor W_bwd = ::tenzor::slice(W, 0, 1, 2);
        W_bwd = ::tenzor::reshape(W_bwd, std::vector<int64_t>{dims.hidden_size, dims.input_size});
        Tensor R_bwd = ::tenzor::slice(R, 0, 1, 2);
        R_bwd = ::tenzor::reshape(R_bwd, std::vector<int64_t>{dims.hidden_size, dims.hidden_size});

        Tensor B_bwd;
        if (has_bias) {
            B_bwd = ::tenzor::slice(B_full, 0, 1, 2);
            B_bwd = ::tenzor::reshape(B_bwd, std::vector<int64_t>{2 * dims.hidden_size});
        }
        load_rnn_direction_weights(*rnn, "backward_cell_0.", W_bwd, R_bwd,
                                    has_bias ? &B_bwd : nullptr,
                                    dims.hidden_size, /*gates=*/1, rnn_perm);
    }

    return rnn;
}

// ============================================================================
// Activation Functions
// ============================================================================

auto ONNXImporter::convert_relu(const ONNXImportNode&)
    -> std::shared_ptr<nn::Module> {
    return std::make_shared<nn::ReLU>();
}

auto ONNXImporter::convert_leaky_relu(const ONNXImportNode& node)
    -> std::shared_ptr<nn::Module> {
    float alpha = node.get_attr("alpha").value_or(ONNXAttribute{}).get_float(0.01f);
    return std::make_shared<nn::LeakyReLU>(static_cast<double>(alpha));
}

auto ONNXImporter::convert_sigmoid(const ONNXImportNode&)
    -> std::shared_ptr<nn::Module> {
    return std::make_shared<nn::Sigmoid>();
}

auto ONNXImporter::convert_tanh(const ONNXImportNode&)
    -> std::shared_ptr<nn::Module> {
    return std::make_shared<nn::Tanh>();
}

auto ONNXImporter::convert_gelu(const ONNXImportNode&)
    -> std::shared_ptr<nn::Module> {
    return std::make_shared<nn::GELU>();
}

auto ONNXImporter::convert_softmax(const ONNXImportNode& node)
    -> std::shared_ptr<nn::Module> {
    int64_t axis = node.get_attr("axis").value_or(ONNXAttribute{}).get_int(-1);
    return std::make_shared<nn::Softmax>(axis);
}

auto ONNXImporter::convert_log_softmax(const ONNXImportNode& node)
    -> std::shared_ptr<nn::Module> {
    int64_t axis = node.get_attr("axis").value_or(ONNXAttribute{}).get_int(-1);
    return std::make_shared<nn::LogSoftmax>(axis);
}

auto ONNXImporter::convert_elu(const ONNXImportNode& node)
    -> std::shared_ptr<nn::Module> {
    float alpha = node.get_attr("alpha").value_or(ONNXAttribute{}).get_float(1.0f);
    return std::make_shared<nn::ELU>(static_cast<double>(alpha));
}

auto ONNXImporter::convert_selu(const ONNXImportNode&)
    -> std::shared_ptr<nn::Module> {
    return std::make_shared<nn::SELU>();
}

// ============================================================================
// Pooling Layers
// ============================================================================

auto ONNXImporter::convert_maxpool(const ONNXImportNode& node) -> std::shared_ptr<nn::Module> {
    // kernel_shape is required; use value_or() + emptiness check rather than a
    // raw operator-> on the optional (which is UB if the attribute is absent,
    // e.g. a mis-routed Global* op).
    auto kernel_shape = node.get_attr("kernel_shape").value_or(ONNXAttribute{}).get_ints({});
    if (kernel_shape.empty()) {
        throw std::runtime_error("ONNX MaxPool: missing required 'kernel_shape' attribute");
    }

    if (kernel_shape.size() == 2) {
        // Per-axis kernel/stride/pad — anisotropic pooling must not be
        // collapsed to the H-axis values only.
        // ONNX-spec default for 'strides' is 1 along each spatial axis (NOT
        // kernel_shape; that is PyTorch MaxPool2d's default). Models that omit
        // strides expect a stride-1 sliding window.
        auto strides = node.get_attr("strides").value_or(ONNXAttribute{}).get_ints(
            std::vector<int64_t>(kernel_shape.size(), 1));
        auto pads = node.get_attr("pads").value_or(ONNXAttribute{}).get_ints({0, 0, 0, 0});
        bool ceil_mode = node.get_attr("ceil_mode").value_or(ONNXAttribute{}).get_int(0) != 0;

        // strides[0],strides[1] are indexed below; a present-but-short attribute
        // (e.g. strides=[1]) would otherwise read past the end.
        if (strides.size() != 2) {
            throw std::runtime_error("ONNX MaxPool2d: 'strides' must have 2 entries, got " +
                                     std::to_string(strides.size()));
        }
        if (pads.size() != 4) {
            throw std::runtime_error("ONNX MaxPool2d: 'pads' must have 4 entries "
                                     "[begin_h, begin_w, end_h, end_w]");
        }
        // ONNX pads layout: [begin_h, begin_w, end_h, end_w]. Pooling layers
        // only support symmetric (begin==end) per-axis padding.
        if (!pads_are_symmetric(pads, 2)) {
            throw std::runtime_error("ONNX MaxPool2d: asymmetric padding "
                                     "(begin != end on an axis) is not supported");
        }

        std::array<int64_t, 2> k = {kernel_shape[0], kernel_shape[1]};
        std::array<int64_t, 2> s = {strides[0], strides[1]};
        std::array<int64_t, 2> p = {pads[0], pads[1]};
        return std::make_shared<nn::MaxPool2d>(k, s, p, ceil_mode, /*return_indices=*/false);
    } else {
        throw std::runtime_error("Unsupported MaxPool dimension: " + std::to_string(kernel_shape.size()));
    }
}

auto ONNXImporter::convert_avgpool(const ONNXImportNode& node) -> std::shared_ptr<nn::Module> {
    auto kernel_shape = node.get_attr("kernel_shape").value_or(ONNXAttribute{}).get_ints({});
    if (kernel_shape.empty()) {
        throw std::runtime_error("ONNX AveragePool: missing required 'kernel_shape' attribute");
    }

    if (kernel_shape.size() == 2) {
        // ONNX-spec default for 'strides' is 1 along each spatial axis (NOT
        // kernel_shape; that is PyTorch AvgPool2d's default).
        auto strides = node.get_attr("strides").value_or(ONNXAttribute{}).get_ints(
            std::vector<int64_t>(kernel_shape.size(), 1));
        auto pads = node.get_attr("pads").value_or(ONNXAttribute{}).get_ints({0, 0, 0, 0});
        // ONNX default count_include_pad is 0 (exclude pad), unlike PyTorch's
        // AvgPool2d default of true.
        bool count_include_pad =
            node.get_attr("count_include_pad").value_or(ONNXAttribute{}).get_int(0) != 0;

        // strides[0],strides[1] are indexed below; a present-but-short attribute
        // (e.g. strides=[1]) would otherwise read past the end.
        if (strides.size() != 2) {
            throw std::runtime_error("ONNX AvgPool2d: 'strides' must have 2 entries, got " +
                                     std::to_string(strides.size()));
        }
        if (pads.size() != 4) {
            throw std::runtime_error("ONNX AvgPool2d: 'pads' must have 4 entries "
                                     "[begin_h, begin_w, end_h, end_w]");
        }
        if (!pads_are_symmetric(pads, 2)) {
            throw std::runtime_error("ONNX AvgPool2d: asymmetric padding "
                                     "(begin != end on an axis) is not supported");
        }

        std::array<int64_t, 2> k = {kernel_shape[0], kernel_shape[1]};
        std::array<int64_t, 2> s = {strides[0], strides[1]};
        std::array<int64_t, 2> p = {pads[0], pads[1]};
        return std::make_shared<nn::AvgPool2d>(k, s, p, count_include_pad);
    } else {
        throw std::runtime_error("Unsupported AvgPool dimension: " + std::to_string(kernel_shape.size()));
    }
}

auto ONNXImporter::convert_global_avgpool([[maybe_unused]] const ONNXImportNode& node) -> std::shared_ptr<nn::Module> {
    // GlobalAveragePool is AdaptiveAvgPool with output_size=(1, 1)
    auto pool = std::make_shared<nn::AdaptiveAvgPool2d>(1, 1);
    return pool;
}

// ============================================================================
// New Shape/Tensor Operations (Phase 6 expansion)
// ============================================================================

auto ONNXImporter::convert_squeeze(const ONNXImportNode& node) -> void {
    auto input = get_input(node.inputs[0]);
    auto axes_attr = node.get_attr("axes");

    if (axes_attr.has_value() && axes_attr->ints.has_value()) {
        // Squeeze specified axes (reverse order to keep indices valid)
        auto axes = axes_attr->ints.value();
        std::sort(axes.begin(), axes.end(), std::greater<>());
        for (int64_t axis : axes) {
            input = input.squeeze(axis);
        }
    } else if (node.inputs.size() > 1) {
        // ONNX opset 13+: axes as second input tensor
        // ONNX permits axes as int32 or int64; cast to Int64 before reading.
        auto axes_tensor = get_input(node.inputs[1]).to(DType::Int64);
        const int64_t* axes_data = axes_tensor.data<int64_t>();
        std::vector<int64_t> axes(axes_data, axes_data + axes_tensor.numel());
        std::sort(axes.begin(), axes.end(), std::greater<>());
        for (int64_t axis : axes) {
            input = input.squeeze(axis);
        }
    } else {
        input = input.squeeze();
    }
    register_output(node.outputs[0], input);
}

auto ONNXImporter::convert_unsqueeze(const ONNXImportNode& node) -> void {
    auto input = get_input(node.inputs[0]);

    std::vector<int64_t> axes;
    if (node.inputs.size() > 1) {
        // ONNX opset 13+: axes as second input tensor
        // ONNX permits axes as int32 or int64; cast to Int64 before reading.
        auto axes_tensor = get_input(node.inputs[1]).to(DType::Int64);
        const int64_t* axes_data = axes_tensor.data<int64_t>();
        axes.assign(axes_data, axes_data + axes_tensor.numel());
    } else {
        auto axes_attr = node.get_attr("axes");
        if (axes_attr.has_value() && axes_attr->ints.has_value()) {
            axes = axes_attr->ints.value();
        }
    }

    // Sort ascending so we insert dims in order
    std::sort(axes.begin(), axes.end());
    for (int64_t axis : axes) {
        input = input.unsqueeze(axis);
    }
    register_output(node.outputs[0], input);
}

auto ONNXImporter::convert_slice(const ONNXImportNode& node) -> void {
    auto input = get_input(node.inputs[0]);

    // ONNX Slice: inputs are [data, starts, ends, axes(optional), steps(optional)]
    // Control tensors (starts/ends/axes/steps) are read on the host below, so
    // force them to CPU first — they may be initializers placed on a GPU device
    // (dereferencing a device pointer on the host crashes).
    // ONNX Slice permits starts/ends/axes/steps as tensor(int32) OR
    // tensor(int64). Tensor::data<int64_t>() enforces the dtype and throws on
    // int32, so cast every control tensor to Int64 before reading.
    auto starts_t = get_host_input(node.inputs[1]).to(DType::Int64);
    auto ends_t = get_host_input(node.inputs[2]).to(DType::Int64);

    const int64_t* starts = starts_t.data<int64_t>();
    const int64_t* ends = ends_t.data<int64_t>();
    int64_t num_slices = starts_t.numel();

    std::vector<int64_t> axes_vec;
    std::vector<int64_t> steps_vec;

    if (node.inputs.size() > 3) {
        auto axes_t = get_host_input(node.inputs[3]).to(DType::Int64);
        const int64_t* axes_data = axes_t.data<int64_t>();
        axes_vec.assign(axes_data, axes_data + axes_t.numel());
    } else {
        for (int64_t i = 0; i < num_slices; ++i) axes_vec.push_back(i);
    }

    if (node.inputs.size() > 4) {
        auto steps_t = get_host_input(node.inputs[4]).to(DType::Int64);
        const int64_t* steps_data = steps_t.data<int64_t>();
        steps_vec.assign(steps_data, steps_data + steps_t.numel());
    } else {
        steps_vec.assign(num_slices, 1);
    }

    // Per the ONNX spec, starts/ends/axes/steps must all share a single length.
    // A malformed (untrusted) model with mismatched lengths would otherwise let
    // the loop below read ends[i] past the raw Int64 buffer and
    // axes_vec[i]/steps_vec[i] past their std::vectors (OOB). Enforce equality up
    // front so every per-index access is in bounds.
    if (ends_t.numel() != num_slices ||
        static_cast<int64_t>(axes_vec.size()) != num_slices ||
        static_cast<int64_t>(steps_vec.size()) != num_slices) {
        throw std::runtime_error(
            "ONNX Slice: starts/ends/axes/steps must all have the same length; got "
            "starts=" + std::to_string(num_slices) +
            ", ends=" + std::to_string(ends_t.numel()) +
            ", axes=" + std::to_string(axes_vec.size()) +
            ", steps=" + std::to_string(steps_vec.size()));
    }

    auto result = input;
    for (int64_t i = 0; i < num_slices; ++i) {
        int64_t dim = axes_vec[i];
        int64_t start = starts[i];
        int64_t end = ends[i];
        int64_t step = steps_vec[i];

        // Normalize and bounds-check the axis before it indexes the shape span.
        // An out-of-range axis from an untrusted model is otherwise an OOB read.
        int64_t norm_dim = dim < 0 ? dim + result.ndim() : dim;
        if (norm_dim < 0 || norm_dim >= result.ndim()) {
            throw std::runtime_error(
                "ONNX Slice: axis " + std::to_string(dim) +
                " out of range for input rank " + std::to_string(result.ndim()));
        }
        int64_t dim_size = result.shape()[norm_dim];
        if (start < 0) start += dim_size;
        if (end < 0) end += dim_size;
        // Clamp
        start = std::max(int64_t(0), std::min(start, dim_size));
        end = std::max(int64_t(0), std::min(end, dim_size));

        result = result.slice(norm_dim, start, end, step);
    }
    register_output(node.outputs[0], result);
}

auto ONNXImporter::convert_pad(const ONNXImportNode& node) -> void {
    // Audit I1: route through nn::functional::pad which supports all modes
    // ONNX cares about. Previous code hard-coded the constant-pad case and
    // silently ignored the ONNX `mode` attribute — every reflect/edge pad in
    // an imported model was being downgraded to constant zero-padding.
    auto input = get_input(node.inputs[0]);

    std::vector<int64_t> pads_vec;
    if (node.inputs.size() > 1) {
        auto pads_t = get_host_input(node.inputs[1]);
        const int64_t* pads_data = pads_t.data<int64_t>();
        pads_vec.assign(pads_data, pads_data + pads_t.numel());
    } else {
        auto pads_attr = node.get_attr("pads");
        if (pads_attr.has_value() && pads_attr->ints.has_value()) {
            pads_vec = pads_attr->ints.value();
        }
    }

    // Wave Inf-C3: pad value is taken from input #2 in Pad-11+ (tensor), or
    // from the legacy `value` attribute in Pad-2..10. Read both for compat.
    double value = 0.0;
    if (node.inputs.size() > 2) {
        // get_host_input: this scalar control tensor is dereferenced on the
        // host below, so it must be CPU-resident (a GPU import otherwise reads a
        // device pointer). Matches convert_reshape/expand/constant_of_shape.
        auto value_t = get_host_input(node.inputs[2]);
        if (value_t.numel() > 0) {
            // Read through Float64 (not Float32): the pad API takes a double,
            // so widening via Float64 is lossless for Float64 models and for
            // integer pad constants beyond float's exact-integer range (>2^24).
            value = *static_cast<const double*>(
                value_t.to(DType::Float64).data_ptr());
        }
    } else if (auto value_attr = node.get_attr("value")) {
        // Legacy Pad-2..10 attribute form.
        if (value_attr->f.has_value()) {
            value = static_cast<double>(value_attr->f.value());
        }
    }

    // Read ONNX `mode` attribute. Defaults to "constant" per ONNX spec.
    std::string onnx_mode = "constant";
    if (auto mode_attr = node.get_attr("mode")) {
        if (mode_attr->s.has_value() && !mode_attr->s->empty()) {
            onnx_mode = mode_attr->s.value();
        }
    }

    // Map ONNX mode strings to Tenzor's nn::functional::pad mode strings:
    //   "constant" → "constant"  (with `value`)
    //   "reflect"  → "reflect"
    //   "edge"     → "replicate"  (same op, different name across frameworks)
    //   "wrap"     → "circular"   (Wave Inf-C2: ONNX wrap-around padding
    //                              maps to circular/wrap-around).
    std::string tenzor_mode;
    if (onnx_mode == "constant")      tenzor_mode = "constant";
    else if (onnx_mode == "reflect")  tenzor_mode = "reflect";
    else if (onnx_mode == "edge")     tenzor_mode = "replicate";
    else if (onnx_mode == "wrap")     tenzor_mode = "circular";
    else {
        throw std::runtime_error("ONNX Pad: unknown mode '" + onnx_mode + "'");
    }

    // ONNX pads layout: [begin_d0, begin_d1, ..., begin_{n-1}, end_d0, end_d1, ..., end_{n-1}]
    // Tenzor pad layout (PyTorch-style): pairs in *reverse* dim order, each
    // pair is (begin, end). So for d in [n-1 .. 0]: append pads[d], pads[d+ndim].
    const int64_t ndim = input.ndim();
    if (static_cast<int64_t>(pads_vec.size()) != 2 * ndim) {
        throw std::runtime_error("ONNX Pad: pads vector size " +
            std::to_string(pads_vec.size()) + " does not match 2 * ndim = " +
            std::to_string(2 * ndim));
    }
    std::vector<int64_t> tenzor_pads;
    tenzor_pads.reserve(2 * ndim);
    for (int64_t d = ndim - 1; d >= 0; --d) {
        tenzor_pads.push_back(pads_vec[d]);             // begin of dim d
        tenzor_pads.push_back(pads_vec[d + ndim]);      // end of dim d
    }

    Variable padded = nn::functional::pad(
        Variable(input, false), tenzor_pads, tenzor_mode, value);
    register_output(node.outputs[0], padded.tensor());
}

auto ONNXImporter::convert_gather(const ONNXImportNode& node) -> void {
    // ONNX `Gather` is a slice-select along `axis` (output rank =
    // data.rank - 1 + indices.rank), NOT element-wise gather. That maps to
    // tenzor::index_select, not tenzor::gather (which is torch.gather /
    // ONNX GatherElements). The exporter mirrors this: it emits
    // IndexSelect/Embedding -> "Gather" and Gather(element-wise) ->
    // "GatherElements", so the importer must invert the same way.
    auto input = get_input(node.inputs[0]);
    auto indices = get_input(node.inputs[1]);
    int64_t axis = node.get_attr("axis").value_or(ONNXAttribute{}).get_int(0);

    auto result = tenzor::index_select(input, axis, indices);
    register_output(node.outputs[0], result);
}

auto ONNXImporter::convert_gather_elements(const ONNXImportNode& node) -> void {
    // ONNX `GatherElements` == torch.gather == tenzor::gather: element-wise
    // gather where indices has the same rank as data and the output takes the
    // shape of indices. This is the counterpart to convert_gather (slice-select)
    // and matches the exporter's Gather(element-wise) -> "GatherElements" map.
    auto input = get_input(node.inputs[0]);
    auto indices = get_input(node.inputs[1]);
    int64_t axis = node.get_attr("axis").value_or(ONNXAttribute{}).get_int(0);

    auto result = tenzor::gather(input, axis, indices);
    register_output(node.outputs[0], result);
}

auto ONNXImporter::convert_clip(const ONNXImportNode& node) -> void {
    auto input = get_input(node.inputs[0]);

    // Read bounds through Float64 (not Float32): tenzor::clamp takes double, so
    // widening via Float64 is lossless for Float64 inputs and for Int64 bounds
    // beyond float's exact-integer range (>2^24) that Float32 would round.
    double min_val = -std::numeric_limits<double>::infinity();
    double max_val = std::numeric_limits<double>::infinity();

    // Opset-11+ Clip carries min/max as INPUTS; opset-6..10 carried them as
    // float ATTRIBUTES. Read the inputs first, then fall back to the legacy
    // attributes so models from either opset (and our own round-trips) import
    // with their clamp bounds intact.
    if (node.inputs.size() > 1 && !node.inputs[1].empty()) {
        // Host-resident: dereferenced on the host below (see convert_pad).
        auto min_t = get_host_input(node.inputs[1]);
        if (min_t.numel() > 0) {
            min_val = *static_cast<const double*>(min_t.to(DType::Float64).data_ptr());
        }
    } else if (auto min_attr = node.get_attr("min");
               min_attr.has_value() && min_attr->f.has_value()) {
        min_val = static_cast<double>(min_attr->f.value());
    }
    if (node.inputs.size() > 2 && !node.inputs[2].empty()) {
        auto max_t = get_host_input(node.inputs[2]);
        if (max_t.numel() > 0) {
            max_val = *static_cast<const double*>(max_t.to(DType::Float64).data_ptr());
        }
    } else if (auto max_attr = node.get_attr("max");
               max_attr.has_value() && max_attr->f.has_value()) {
        max_val = static_cast<double>(max_attr->f.value());
    }

    auto result = tenzor::clamp(input, min_val, max_val);
    register_output(node.outputs[0], result);
}

auto ONNXImporter::convert_cast(const ONNXImportNode& node) -> void {
    auto input = get_input(node.inputs[0]);
    int64_t to_type = node.get_attr("to").value_or(ONNXAttribute{}).get_int(1);

    // ONNX tensor type enum to DType
    DType dtype;
    switch (to_type) {
        case 1:  dtype = DType::Float32; break;
        case 2:  dtype = DType::UInt8; break;
        case 3:  dtype = DType::Int8; break;
        case 4:  dtype = DType::UInt16; break;
        case 5:  dtype = DType::Int16; break;
        case 6:  dtype = DType::Int32; break;
        case 7:  dtype = DType::Int64; break;
        case 9:  dtype = DType::Bool; break;
        case 10: dtype = DType::Float16; break;
        case 11: dtype = DType::Float64; break;
        case 12: dtype = DType::UInt32; break;
        case 13: dtype = DType::UInt64; break;
        case 14: dtype = DType::Complex64; break;
        case 15: dtype = DType::Complex128; break;
        case 16: dtype = DType::BFloat16; break;
        case 17: dtype = DType::FP8_E4M3; break;       // ONNX FLOAT8E4M3FN
        case 18: dtype = DType::FP8_E4M3FNUZ; break;   // ONNX FLOAT8E4M3FNUZ
        case 19: dtype = DType::FP8_E5M2; break;       // ONNX FLOAT8E5M2
        case 20: dtype = DType::FP8_E5M2FNUZ; break;   // ONNX FLOAT8E5M2FNUZ
        default:
            // 8 = STRING (no Tenzor dtype) and any future/unknown id.
            throw std::runtime_error("Unsupported ONNX Cast target type: " + std::to_string(to_type));
    }

    register_output(node.outputs[0], input.to(dtype));
}

auto ONNXImporter::convert_dropout(const ONNXImportNode& node) -> void {
    // In inference mode, dropout is identity
    auto input = get_input(node.inputs[0]);
    register_output(node.outputs[0], input);
    // If there's a mask output, register an empty tensor
    if (node.outputs.size() > 1) {
        register_output(node.outputs[1], Tensor());
    }
}

auto ONNXImporter::convert_resize(const ONNXImportNode& node) -> void {
    auto input = get_input(node.inputs[0]);

    // ONNX Resize: inputs are [X, roi, scales, sizes]
    // Try sizes first (input[3]), then scales (input[2])
    std::vector<int64_t> output_size;

    if (node.inputs.size() > 3 && !node.inputs[3].empty()) {
        auto sizes_t = get_host_input(node.inputs[3]).to(DType::Int64);
        const int64_t* sizes_data = sizes_t.data<int64_t>();
        // sizes includes batch and channel dims — take only spatial
        int64_t ndim = sizes_t.numel();
        for (int64_t i = 2; i < ndim; ++i) {
            output_size.push_back(sizes_data[i]);
        }
    } else if (node.inputs.size() > 2 && !node.inputs[2].empty()) {
        auto scales_t = get_host_input(node.inputs[2]);
        // ONNX requires len(scales) == rank(input); without this check the
        // loop below reads input.shape()[i] past the end for an over-long,
        // attacker-controlled scales tensor (out-of-bounds read).
        if (scales_t.numel() != input.ndim()) {
            throw std::runtime_error(
                "ONNX Resize: scales length (" + std::to_string(scales_t.numel()) +
                ") must equal input rank (" + std::to_string(input.ndim()) + ")");
        }
        auto scales_f32 = scales_t.to(DType::Float32);
        const float* scales = scales_f32.data<float>();
        // scales includes batch and channel dims
        for (int64_t i = 2; i < scales_t.numel(); ++i) {
            output_size.push_back(static_cast<int64_t>(input.shape()[i] * scales[i]));
        }
    } else {
        throw std::runtime_error("ONNX Resize: no sizes or scales provided");
    }

    auto mode_attr = node.get_attr("mode");
    std::string mode = "nearest";
    if (mode_attr.has_value() && mode_attr->s.has_value()) {
        mode = mode_attr->s.value();
        if (mode == "linear") mode = "bilinear";
        if (mode == "cubic") mode = "bicubic";
    }

    // Honor coordinate_transformation_mode (default "half_pixel" per the ONNX
    // Resize spec). Tenzor's interpolate exposes a single `align_corners` knob:
    //   - "align_corners"           -> align_corners = true
    //   - "asymmetric" /
    //     "half_pixel" /
    //     "pytorch_half_pixel"      -> align_corners = false
    // The earlier code hard-coded false, so an "align_corners" Resize sampled
    // with the wrong coordinate mapping (visible as a sub-pixel shift). The
    // half-pixel variants all reduce to align_corners=false here; the finer
    // distinction between them (and the nearest-only `nearest_mode` rounding)
    // is not representable in the current interpolate API — see report.
    std::string coord_mode = node.get_attr("coordinate_transformation_mode")
                                 .value_or(ONNXAttribute{})
                                 .get_string("half_pixel");
    bool align_corners = (coord_mode == "align_corners");

    auto result = tenzor::ops::interpolate(input, output_size, mode, align_corners);
    register_output(node.outputs[0], result);
}

// Audit I2: extract the ONNX reduction `axes` attribute (or `axes` input
// tensor for opset-13+ where it migrated from attribute to second input)
// into the full vector, not just `axes[0]`. Returns an empty vector to mean
// "reduce all axes" (per ONNX spec: missing `axes` reduces over all dims).
static auto get_reduce_axes(const ONNXImportNode& node,
                            const std::function<Tensor(const std::string&)>& get_input)
    -> std::vector<int64_t> {
    auto axes_attr = node.get_attr("axes");
    if (axes_attr.has_value() && axes_attr->ints.has_value() && !axes_attr->ints->empty()) {
        return axes_attr->ints.value();  // full vector, not `at(0)`
    }
    if (node.inputs.size() > 1 && !node.inputs[1].empty()) {
        auto axes_t = get_input(node.inputs[1]);  // host-reader passed by caller
        if (axes_t.numel() > 0) {
            const int64_t* p = axes_t.data<int64_t>();
            return std::vector<int64_t>(p, p + axes_t.numel());
        }
    }
    return {};  // empty → reduce over all axes
}

// Opset-18+ reductions carry a `noop_with_empty_axes` attribute. When it is
// set (=1) AND no axes are supplied, the operator is the IDENTITY — it must NOT
// fall through to the legacy "reduce over all axes" behaviour (which is what
// noop_with_empty_axes=0 / pre-18 opsets mean). Returns true for the identity
// case so the caller can pass the input through untouched.
static auto reduce_is_noop(const ONNXImportNode& node,
                           const std::vector<int64_t>& axes) -> bool {
    return axes.empty() &&
           node.get_attr("noop_with_empty_axes")
               .value_or(ONNXAttribute{}).get_int(0) != 0;
}

// Audit I2: apply `single_dim_reduce` (sum/mean/max) over the listed axes.
// We don't have multi-dim Tensor reduction overloads, so we apply per-axis.
// Reductions are commutative under our operations, but the axis indices need
// to remain valid as the rank shrinks (when keepdims=false). Solution:
//   - Normalize negative axes against the input rank.
//   - Sort axes descending so that reducing axis N doesn't invalidate axis <N.
// With keepdims=true, the rank doesn't change so order doesn't matter — but
// we still sort descending for consistency.
template <typename Fn>
static auto apply_multi_axis_reduce(const Tensor& input,
                                    std::vector<int64_t> axes,
                                    bool keepdims, Fn single_dim_reduce) -> Tensor {
    const int64_t ndim = input.ndim();
    if (axes.empty()) {
        // ONNX semantics: missing axes → reduce over all axes.
        for (int64_t d = 0; d < ndim; ++d) axes.push_back(d);
    }
    // Normalize negative axes.
    for (auto& a : axes) {
        if (a < 0) a += ndim;
        if (a < 0 || a >= ndim) {
            throw std::runtime_error(
                "ONNX reduce: axis " + std::to_string(a) +
                " is out of range for a rank-" + std::to_string(ndim) + " input");
        }
    }
    std::sort(axes.begin(), axes.end(), std::greater<int64_t>());

    Tensor result = input;
    for (int64_t a : axes) {
        // When keepdims=false, applying reduction along `a` deletes that dim.
        // Since axes are sorted descending, smaller axes still refer to the
        // same logical dim after this step.
        result = single_dim_reduce(result, std::optional<int64_t>{a}, keepdims);
    }
    return result;
}

auto ONNXImporter::convert_reduce_sum(const ONNXImportNode& node) -> void {
    auto input = get_input(node.inputs[0]);
    bool keepdims = node.get_attr("keepdims").value_or(ONNXAttribute{}).get_int(1) != 0;
    auto axes = get_reduce_axes(node,
        [this](const std::string& s) { return this->get_host_input(s); });
    if (reduce_is_noop(node, axes)) {
        // noop_with_empty_axes=1 with no axes → identity (return input as-is).
        register_output(node.outputs[0], input);
        return;
    }
    auto result = apply_multi_axis_reduce(input, axes, keepdims,
        [](const Tensor& t, std::optional<int64_t> d, bool k) {
            return tenzor::sum(t, d, k);
        });
    register_output(node.outputs[0], result);
}

auto ONNXImporter::convert_reduce_mean(const ONNXImportNode& node) -> void {
    auto input = get_input(node.inputs[0]);
    bool keepdims = node.get_attr("keepdims").value_or(ONNXAttribute{}).get_int(1) != 0;
    auto axes = get_reduce_axes(node,
        [this](const std::string& s) { return this->get_host_input(s); });
    if (reduce_is_noop(node, axes)) {
        // noop_with_empty_axes=1 with no axes → identity (return input as-is).
        register_output(node.outputs[0], input);
        return;
    }
    auto result = apply_multi_axis_reduce(input, axes, keepdims,
        [](const Tensor& t, std::optional<int64_t> d, bool k) {
            return tenzor::mean(t, d, k);
        });
    register_output(node.outputs[0], result);
}

auto ONNXImporter::convert_reduce_max(const ONNXImportNode& node) -> void {
    auto input = get_input(node.inputs[0]);
    bool keepdims = node.get_attr("keepdims").value_or(ONNXAttribute{}).get_int(1) != 0;
    auto axes = get_reduce_axes(node,
        [this](const std::string& s) { return this->get_host_input(s); });
    if (reduce_is_noop(node, axes)) {
        // noop_with_empty_axes=1 with no axes → identity (return input as-is).
        register_output(node.outputs[0], input);
        return;
    }
    auto result = apply_multi_axis_reduce(input, axes, keepdims,
        [](const Tensor& t, std::optional<int64_t> d, bool k) {
            return tenzor::max(t, d, k);
        });
    register_output(node.outputs[0], result);
}

// Audit F.17: ReduceMin / Prod / L1 / L2.  Each routes through the
// existing multi-axis reduce helper so a full axes-array import is
// supported (the previous gap dropped every axis but axes[0]).

auto ONNXImporter::convert_reduce_min(const ONNXImportNode& node) -> void {
    auto input = get_input(node.inputs[0]);
    bool keepdims = node.get_attr("keepdims").value_or(ONNXAttribute{}).get_int(1) != 0;
    auto axes = get_reduce_axes(node,
        [this](const std::string& s) { return this->get_host_input(s); });
    if (reduce_is_noop(node, axes)) {
        // noop_with_empty_axes=1 with no axes → identity (return input as-is).
        register_output(node.outputs[0], input);
        return;
    }
    auto result = apply_multi_axis_reduce(input, axes, keepdims,
        [](const Tensor& t, std::optional<int64_t> d, bool k) {
            return tenzor::min(t, d, k);
        });
    register_output(node.outputs[0], result);
}

auto ONNXImporter::convert_reduce_prod(const ONNXImportNode& node) -> void {
    auto input = get_input(node.inputs[0]);
    bool keepdims = node.get_attr("keepdims").value_or(ONNXAttribute{}).get_int(1) != 0;
    auto axes = get_reduce_axes(node,
        [this](const std::string& s) { return this->get_host_input(s); });
    if (reduce_is_noop(node, axes)) {
        // noop_with_empty_axes=1 with no axes → identity (return input as-is).
        register_output(node.outputs[0], input);
        return;
    }
    auto result = apply_multi_axis_reduce(input, axes, keepdims,
        [](const Tensor& t, std::optional<int64_t> d, bool k) {
            return tenzor::prod(t, d, k);
        });
    register_output(node.outputs[0], result);
}

auto ONNXImporter::convert_reduce_l1(const ONNXImportNode& node) -> void {
    // ReduceL1: sum(|x|) over the given axes.
    auto input = get_input(node.inputs[0]);
    bool keepdims = node.get_attr("keepdims").value_or(ONNXAttribute{}).get_int(1) != 0;
    auto axes = get_reduce_axes(node,
        [this](const std::string& s) { return this->get_host_input(s); });
    if (reduce_is_noop(node, axes)) {
        // noop_with_empty_axes=1 with no axes → identity (return input as-is).
        register_output(node.outputs[0], input);
        return;
    }
    auto abs_in = tenzor::abs(input);
    auto result = apply_multi_axis_reduce(abs_in, axes, keepdims,
        [](const Tensor& t, std::optional<int64_t> d, bool k) {
            return tenzor::sum(t, d, k);
        });
    register_output(node.outputs[0], result);
}

auto ONNXImporter::convert_reduce_l2(const ONNXImportNode& node) -> void {
    // ReduceL2: sqrt(sum(x^2)) over the given axes.
    auto input = get_input(node.inputs[0]);
    bool keepdims = node.get_attr("keepdims").value_or(ONNXAttribute{}).get_int(1) != 0;
    auto axes = get_reduce_axes(node,
        [this](const std::string& s) { return this->get_host_input(s); });
    if (reduce_is_noop(node, axes)) {
        // noop_with_empty_axes=1 with no axes → identity (return input as-is).
        register_output(node.outputs[0], input);
        return;
    }
    auto sq = tenzor::mul(input, input);
    auto sum = apply_multi_axis_reduce(sq, axes, keepdims,
        [](const Tensor& t, std::optional<int64_t> d, bool k) {
            return tenzor::sum(t, d, k);
        });
    register_output(node.outputs[0], tenzor::sqrt(sum));
}


auto ONNXImporter::convert_shape(const ONNXImportNode& node) -> void {
    auto input = get_input(node.inputs[0]);
    auto shape = input.shape();

    // Create a 1D Int64 tensor containing the shape
    Tensor shape_tensor({static_cast<int64_t>(shape.size())}, DType::Int64, Device::cpu());
    int64_t* data = shape_tensor.data<int64_t>();
    for (size_t i = 0; i < shape.size(); ++i) {
        data[i] = shape[i];
    }

    register_output(node.outputs[0], shape_tensor);
}

auto ONNXImporter::convert_constant_of_shape(const ONNXImportNode& node) -> void {
    // Host-read the shape control tensor: get_input() returns it on the import
    // device, and data<int64_t>() dereferences a device pointer on the host
    // (crash) for a GPU import.
    auto shape_tensor = get_host_input(node.inputs[0]);
    const int64_t* shape_data = shape_tensor.data<int64_t>();
    std::vector<int64_t> shape(shape_data, shape_data + shape_tensor.numel());

    // The `value` attribute tensor's dtype determines the output dtype (ONNX
    // commonly uses Int64 ConstantOfShape to build shape/index/range/mask
    // constants). Defaulting to Float32 would silently mistype those, breaking
    // downstream gather/index/compare ops. Per ONNX spec the default value is a
    // single Float32 zero when the attribute is absent.
    double value = 0.0;
    DType out_dtype = DType::Float32;
    auto value_attr = node.get_attr("value");
    if (value_attr.has_value() && value_attr->tensor.has_value()) {
        auto val_tensor = value_attr->tensor->to_tensor();
        out_dtype = val_tensor.dtype();
        if (val_tensor.numel() > 0) {
            // Extract the scalar via Float64 (lossless for the int/float dtypes
            // ONNX uses here); tenzor::full re-narrows to out_dtype.
            value = *static_cast<const double*>(
                val_tensor.to(DType::Float64).data_ptr());
        }
    }

    auto result = tenzor::full(shape, value, out_dtype, Device::cpu());
    register_output(node.outputs[0], result);
}

auto ONNXImporter::convert_where(const ONNXImportNode& node) -> void {
    auto condition = get_input(node.inputs[0]);
    auto x = get_input(node.inputs[1]);
    auto y = get_input(node.inputs[2]);

    auto result = tenzor::where(condition, x, y);
    register_output(node.outputs[0], result);
}

auto ONNXImporter::convert_expand(const ONNXImportNode& node) -> void {
    auto input = get_input(node.inputs[0]);
    // Host-read the shape control tensor (see convert_reshape): data<int64_t>()
    // would otherwise dereference a device pointer on the host for a GPU import.
    auto shape_tensor = get_host_input(node.inputs[1]);
    const int64_t* shape_data = shape_tensor.data<int64_t>();
    std::vector<int64_t> shape(shape_data, shape_data + shape_tensor.numel());

    auto result = tenzor::expand(input, shape);
    register_output(node.outputs[0], result);
}

auto ONNXImporter::convert_pow(const ONNXImportNode& node) -> void {
    auto input = get_input(node.inputs[0]);
    auto exponent = get_input(node.inputs[1]);

    // Prefer the scalar pow op (tenzor::pow) whenever the exponent reduces to a
    // single value: it is a true element-wise power that handles zero/negative
    // bases per IEEE, unlike the exp(log(x)*y) decomposition which produces
    // NaN/-inf for any non-positive base (e.g. Pow([-2],[2]) -> NaN instead of 4).
    //
    // The common ONNX case is a constant scalar (or a constant tensor of one
    // repeated value broadcast across the bases); detect that on the host and
    // route through the scalar path.
    bool used_scalar = false;
    if (exponent.numel() >= 1) {
        // Read the exponent on the host so we can inspect its values without
        // dereferencing a device pointer. Use Float64 (not Float32) so a
        // non-float-exact exponent on a Float64 graph (e.g. 1.0/3.0, or an
        // integer > 2^24) keeps full precision: tenzor::pow already takes a
        // double, and inspecting/comparing in double avoids collapsing two
        // distinct Float64 exponents that happen to round to the same float.
        Tensor host_exp =
            (exponent.device().type == Device::Type::CPU)
                ? exponent.to(DType::Float64)
                : exponent.to(Device::cpu()).to(DType::Float64);
        const double* exp_data = static_cast<const double*>(host_exp.data_ptr());
        double first = exp_data[0];
        bool all_equal = true;
        for (int64_t i = 1; i < host_exp.numel(); ++i) {
            if (exp_data[i] != first) { all_equal = false; break; }
        }
        if (all_equal) {
            auto result = tenzor::pow(input, first);
            register_output(node.outputs[0], result);
            used_scalar = true;
        }
    }

    if (!used_scalar) {
        // Genuine element-wise tensor exponent. Use float_power (Float64
        // promotion) rather than the naive exp(log(x)*y): float_power avoids
        // log(0)=-inf by operating on |base|. This still cannot reproduce the
        // sign of a negative base raised to an odd integer power (no general
        // tensor-tensor pow op exists), but it no longer corrupts non-negative
        // bases and degrades gracefully.
        auto result = tenzor::float_power(input, exponent);
        register_output(node.outputs[0], result);
    }
}

auto ONNXImporter::convert_sqrt(const ONNXImportNode& node) -> void {
    auto input = get_input(node.inputs[0]);
    register_output(node.outputs[0], tenzor::sqrt(input));
}

auto ONNXImporter::convert_neg(const ONNXImportNode& node) -> void {
    auto input = get_input(node.inputs[0]);
    register_output(node.outputs[0], tenzor::neg(input));
}

auto ONNXImporter::convert_exp(const ONNXImportNode& node) -> void {
    auto input = get_input(node.inputs[0]);
    register_output(node.outputs[0], tenzor::exp(input));
}

auto ONNXImporter::convert_log(const ONNXImportNode& node) -> void {
    auto input = get_input(node.inputs[0]);
    register_output(node.outputs[0], tenzor::log(input));
}

// ============================================================================
// Audit I6: ONNX op converters wiring through existing tensor ops.
// ============================================================================

auto ONNXImporter::convert_argmax(const ONNXImportNode& node) -> void {
    auto input = get_input(node.inputs[0]);
    int64_t axis = node.get_attr("axis").value_or(ONNXAttribute{}).get_int(0);
    bool keepdims = node.get_attr("keepdims").value_or(ONNXAttribute{}).get_int(1) != 0;
    register_output(node.outputs[0], tenzor::argmax(input, std::optional<int64_t>{axis}, keepdims));
}

auto ONNXImporter::convert_argmin(const ONNXImportNode& node) -> void {
    auto input = get_input(node.inputs[0]);
    int64_t axis = node.get_attr("axis").value_or(ONNXAttribute{}).get_int(0);
    bool keepdims = node.get_attr("keepdims").value_or(ONNXAttribute{}).get_int(1) != 0;
    register_output(node.outputs[0], tenzor::argmin(input, std::optional<int64_t>{axis}, keepdims));
}

auto ONNXImporter::convert_topk(const ONNXImportNode& node) -> void {
    // ONNX TopK: inputs (X, K). Attrs: axis (default -1), largest (default 1), sorted (default 1).
    // Outputs: (Values, Indices).
    auto input = get_input(node.inputs[0]);
    int64_t k = 1;
    if (node.inputs.size() > 1) {
        // Host-resident + Int64: K is dereferenced on the host, and may be
        // provided as Int32 (data<int64_t>() would otherwise throw).
        auto k_t = get_host_input(node.inputs[1]).to(DType::Int64);
        if (k_t.numel() > 0) k = k_t.data<int64_t>()[0];
    } else {
        // Older opsets (pre-10) used `k` as an int attribute.
        auto k_attr = node.get_attr("k");
        if (k_attr.has_value()) k = k_attr->get_int(1);
    }
    int64_t axis    = node.get_attr("axis").value_or(ONNXAttribute{}).get_int(-1);
    bool   largest  = node.get_attr("largest").value_or(ONNXAttribute{}).get_int(1) != 0;
    bool   is_sorted= node.get_attr("sorted").value_or(ONNXAttribute{}).get_int(1) != 0;
    auto [values, indices] = tenzor::topk(input, k, axis, largest, is_sorted);
    register_output(node.outputs[0], values);
    if (node.outputs.size() > 1) register_output(node.outputs[1], indices);
}

auto ONNXImporter::convert_tile(const ONNXImportNode& node) -> void {
    // ONNX Tile: inputs (input, repeats). repeats is an Int64 1-D tensor.
    auto input   = get_input(node.inputs[0]);
    // Host-read the repeats control tensor (see convert_reshape): data<int64_t>()
    // would otherwise dereference a device pointer on the host for a GPU import.
    auto repeats = get_host_input(node.inputs[1]);
    const int64_t* rp = repeats.data<int64_t>();
    std::vector<int64_t> reps_vec(rp, rp + repeats.numel());
    register_output(node.outputs[0], tenzor::tile(input, reps_vec));
}

auto ONNXImporter::convert_range(const ONNXImportNode& node) -> void {
    // ONNX Range: inputs (start, limit, delta) — all scalars (0-D tensors).
    // Host-resident: these scalars are dereferenced on the host below; a GPU
    // import otherwise reads device pointers. (dtype/device reads farther down
    // do not dereference data, so get_input is fine there.)
    auto start_t = get_host_input(node.inputs[0]).to(DType::Float64);
    auto limit_t = get_host_input(node.inputs[1]).to(DType::Float64);
    auto delta_t = get_host_input(node.inputs[2]).to(DType::Float64);
    double start = start_t.data<double>()[0];
    double limit = limit_t.data<double>()[0];
    double delta = delta_t.data<double>()[0];
    // ONNX Range output dtype = input dtype; use start's dtype for the result.
    register_output(node.outputs[0],
        tenzor::arange(start, limit, delta, get_input(node.inputs[0]).dtype(),
                       get_input(node.inputs[0]).device()));
}

auto ONNXImporter::convert_non_zero(const ONNXImportNode& node) -> void {
    auto input = get_input(node.inputs[0]);
    register_output(node.outputs[0], tenzor::nonzero(input));
}

auto ONNXImporter::convert_round(const ONNXImportNode& node) -> void {
    auto input = get_input(node.inputs[0]);
    register_output(node.outputs[0], tenzor::round(input));
}

auto ONNXImporter::convert_einsum(const ONNXImportNode& node) -> void {
    // ONNX Einsum: variable number of inputs; equation in `equation` attr.
    std::string eq;
    auto eq_attr = node.get_attr("equation");
    if (eq_attr.has_value() && eq_attr->s.has_value()) eq = eq_attr->s.value();
    if (eq.empty()) throw std::runtime_error("ONNX Einsum: missing `equation` attribute");

    std::vector<Tensor> inputs;
    inputs.reserve(node.inputs.size());
    for (const auto& name : node.inputs) inputs.push_back(get_input(name));
    register_output(node.outputs[0], tenzor::einsum(eq, std::span<const Tensor>(inputs)));
}

auto ONNXImporter::convert_trilu(const ONNXImportNode& node) -> void {
    // ONNX Trilu: inputs (input, k=0). Attr `upper` (default 1).
    auto input = get_input(node.inputs[0]);
    int64_t k = 0;
    if (node.inputs.size() > 1 && !node.inputs[1].empty()) {
        auto k_t = get_input(node.inputs[1]);
        if (k_t.numel() > 0) k = k_t.data<int64_t>()[0];
    }
    bool upper = node.get_attr("upper").value_or(ONNXAttribute{}).get_int(1) != 0;
    register_output(node.outputs[0], upper ? tenzor::triu(input, k) : tenzor::tril(input, k));
}

// ============================================================================
// Helper Functions
// ============================================================================

auto ONNXImporter::get_input(const std::string& name) -> Tensor {
    auto value = context_.get_value(name);
    if (!value.has_value()) {
        throw std::runtime_error("Input tensor not found: " + name);
    }
    return value.value();
}

auto ONNXImporter::register_output(const std::string& name, const Tensor& tensor) -> void {
    context_.register_value(name, tensor);
}

auto ONNXImporter::log(const std::string& message) -> void {
    if (verbose_) {
        std::cout << "[ONNX Importer] " << message << std::endl;
    }
}

// ============================================================================
// Quantization (QDQ) Operations
// ============================================================================

// ONNX QuantizeLinear specifies round-half-to-even ("banker's rounding") when
// mapping the scaled value to the integer grid. tenzor::round delegates to
// std::round, which rounds halves AWAY from zero — biasing exactly-halfway
// values (e.g. 0.5 -> 1 instead of 0, 2.5 -> 3 instead of 2) and diverging from
// every ONNX reference runtime. Tenzor has no round-to-even op, so we implement
// it with device-agnostic tensor ops via floor + fractional analysis:
//   fl   = floor(v)            (toward -inf, so this is correct for negatives)
//   frac = v - fl              in [0, 1)
//   round up when frac > 0.5, OR (frac == 0.5 AND fl is odd) so the result
//   lands on the even neighbour. fl is odd  <=>  fmod(fl, 2) != 0 (works for
//   negative fl too, since fmod keeps the sign of the dividend).
static auto round_half_even(const Tensor& v) -> Tensor {
    Tensor fl = floor(v);
    Tensor frac = v - fl;
    Tensor half = full_like(v, 0.5);
    Tensor two = full_like(v, 2.0);
    Tensor zero = full_like(v, 0.0);
    Tensor up = gt(frac, half);                       // frac > 0.5
    Tensor is_half = eq(frac, half);                  // exactly halfway
    Tensor fl_odd = ne(fmod(fl, two), zero);          // floor is odd
    Tensor round_up = logical_or(up, logical_and(is_half, fl_odd));
    return fl + round_up.to(v.dtype());               // fl + {0, 1}
}

auto ONNXImporter::convert_quantize_linear(const ONNXImportNode& node) -> void {
    // Audit I3: dtype-aware quantization. The output dtype of ONNX
    // QuantizeLinear is the dtype of `y_zero_point` (or UInt8 [0,255], the
    // ONNX default output type, if zero_point is omitted). Previous code
    // hard-coded Int8 + [-128, 127] saturation,
    // which silently miscompiled UInt8 quantization (the most common case
    // for INT8 quantized image models) by clipping to negative range.
    //
    // Supported output dtypes per ONNX spec: UInt8, Int8, UInt16, Int16,
    // Int32 (later opsets), Float8E4M3/E5M2 (opset 20+). We handle the
    // common integer dtypes here; rarer dtypes fall through to a clear
    // error pointing at follow-up work.
    auto x = get_input(node.inputs[0]);
    auto y_scale = get_input(node.inputs[1]);

    // Per-channel quantization: scale/zero_point are 1-D along `axis` (ONNX
    // default axis = 1). Reshape them to broadcast along that axis rather than
    // the trailing dim (the previous code ignored `axis`, so per-channel QDQ on
    // any non-last axis was applied to the wrong dimension).
    auto qaxis_attr = node.get_attr("axis");
    int64_t qaxis = qaxis_attr.has_value() ? qaxis_attr->get_int(1) : 1;
    auto qbcast_shape = [&](const Tensor& t) {
        std::vector<int64_t> shp(static_cast<size_t>(x.ndim()), 1);
        int64_t ax = qaxis < 0 ? qaxis + x.ndim() : qaxis;
        if (ax >= 0 && ax < x.ndim() && t.ndim() == 1) shp[static_cast<size_t>(ax)] = t.shape()[0];
        return shp;
    };
    if (y_scale.ndim() == 1) {
        y_scale = y_scale.reshape(qbcast_shape(y_scale));
    }

    auto scaled = x / y_scale;
    auto rounded = round_half_even(scaled);  // ONNX: round-half-to-even

    // Determine output dtype + saturation range from zero_point dtype.
    // Per the ONNX QuantizeLinear spec, when y_zero_point is omitted the output
    // type T2 defaults to tensor(uint8) with range [0, 255] (NOT int8).
    DType out_dtype = DType::UInt8;         // ONNX default if zero_point absent
    double sat_lo = 0.0, sat_hi = 255.0;
    if (node.inputs.size() > 2 && !node.inputs[2].empty()) {
        auto y_zero_point = get_input(node.inputs[2]);
        out_dtype = y_zero_point.dtype();
        Tensor zp = (y_zero_point.ndim() == 1)
            ? y_zero_point.reshape(qbcast_shape(y_zero_point)) : y_zero_point;
        rounded = rounded + zp.to(rounded.dtype());

        switch (out_dtype) {
            case DType::UInt8:    sat_lo =       0.0; sat_hi =     255.0; break;
            case DType::Int8:     sat_lo =    -128.0; sat_hi =     127.0; break;
            case DType::UInt16:   sat_lo =       0.0; sat_hi =   65535.0; break;
            case DType::Int16:    sat_lo =  -32768.0; sat_hi =   32767.0; break;
            case DType::Int32:    sat_lo = -2147483648.0; sat_hi = 2147483647.0; break;
            // ONNX Float8 (opset 20+): E4M3FN finite range ≈ ±448, E5M2 ≈ ±57344.
            // Saturating-cast to the FP8 dtype after clamping the rounded value.
            case DType::FP8_E4M3: sat_lo =     -448.0; sat_hi =     448.0; break;
            case DType::FP8_E5M2: sat_lo =   -57344.0; sat_hi =   57344.0; break;
            case DType::FP8_E4M3FNUZ: sat_lo =   -240.0; sat_hi =     240.0; break;
            case DType::FP8_E5M2FNUZ: sat_lo = -57344.0; sat_hi =   57344.0; break;
            default:
                throw std::runtime_error(
                    std::string("ONNX QuantizeLinear: output dtype ") +
                    std::string(dtype_name(out_dtype)) +
                    " is not supported by the importer.");
        }
    }

    // Pass the saturation bounds as double (clamp's native parameter type) and
    // do NOT narrow them through float first: static_cast<float>(2147483647.0)
    // rounds up to 2147483648.0f (24-bit float mantissa), which would clamp the
    // Int32 max to a value that overflows INT32_MAX and wraps to INT32_MIN on
    // the subsequent .to(Int32) cast.
    auto result = clamp(rounded, sat_lo, sat_hi).to(out_dtype);
    register_output(node.outputs[0], result);
    log(std::string("Converted QuantizeLinear: ") + node.outputs[0] +
        " (output dtype " + std::string(dtype_name(out_dtype)) + ")");
}

auto ONNXImporter::convert_dequantize_linear(const ONNXImportNode& node) -> void {
    // DequantizeLinear: y = (x - x_zero_point) * x_scale
    // Inputs: x, x_scale, x_zero_point (optional)
    auto x = get_input(node.inputs[0]);
    auto x_scale = get_input(node.inputs[1]);

    // Per the ONNX DequantizeLinear spec the output dtype equals the dtype of
    // x_scale (Float16 / BFloat16 / Float32). The previous code hard-coded a
    // Float32 result, silently widening Float16/BFloat16-scaled models.
    const DType deq_out_dtype = x_scale.dtype();

    // Per-channel dequant: scale/zero_point are 1-D along `axis` (default 1);
    // reshape to broadcast along that axis instead of the trailing dim.
    auto dqaxis_attr = node.get_attr("axis");
    int64_t dqaxis = dqaxis_attr.has_value() ? dqaxis_attr->get_int(1) : 1;
    auto dqbcast_shape = [&](const Tensor& t) {
        std::vector<int64_t> shp(static_cast<size_t>(x.ndim()), 1);
        int64_t ax = dqaxis < 0 ? dqaxis + x.ndim() : dqaxis;
        if (ax >= 0 && ax < x.ndim() && t.ndim() == 1) shp[static_cast<size_t>(ax)] = t.shape()[0];
        return shp;
    };
    if (x_scale.ndim() == 1) {
        x_scale = x_scale.reshape(dqbcast_shape(x_scale));
    }

    // Convert quantized input to float for the arithmetic (subtract zero-point,
    // multiply by scale), then narrow the result back to x_scale's dtype so the
    // declared output type is preserved. Computing in Float32 keeps the
    // intermediate exact for integer inputs regardless of the scale dtype.
    auto x_float = x.to(DType::Float32);

    if (node.inputs.size() > 2 && !node.inputs[2].empty()) {
        auto x_zero_point = get_input(node.inputs[2]);
        Tensor zp = (x_zero_point.ndim() == 1)
            ? x_zero_point.reshape(dqbcast_shape(x_zero_point)) : x_zero_point;
        x_float = x_float - zp.to(DType::Float32);
    }

    auto result = (x_float * x_scale.to(DType::Float32)).to(deq_out_dtype);

    register_output(node.outputs[0], result);
    log("Converted DequantizeLinear: " + node.outputs[0]);
}

// ============================================================================
// High-level Import Function
// ============================================================================

auto import_onnx(const std::string& filepath, bool verbose) -> std::shared_ptr<nn::Module> {
    ONNXImporter importer(verbose);
    return importer.import_from_file(filepath);
}

} // namespace onnx
} // namespace tenzor
