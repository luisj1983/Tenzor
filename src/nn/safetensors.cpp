/**
 * @file safetensors.cpp
 * @brief SafeTensors format implementation
 */

#include "tenzor/nn/safetensors.hpp"
#include "tenzor/nn/serialize.hpp"
#include "tenzor/ops/creation.hpp"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <cstring>

namespace tenzor::nn {

// ============================================================================
// DType <-> SafeTensors string mapping
// ============================================================================

auto SafeTensorsSerializer::dtype_to_st_string(DType dtype) -> std::string {
    // Tenzor-specific extensions on top of the upstream SafeTensors dtype
    // strings: "C64" / "C128" for complex (audit item I.13), and the
    // already-vendored "F8_E4M3" / "F8_E5M2" for FP8.  Quantized types
    // (QInt8 / QUInt8 / QInt4x2) round-trip as their raw storage dtype;
    // the scale / zero_point need to ride alongside in metadata for full
    // round-trip — see the Q* handling below.
    switch (dtype) {
        case DType::Float32:    return "F32";
        case DType::Float64:    return "F64";
        case DType::Float16:    return "F16";
        case DType::BFloat16:   return "BF16";
        case DType::Int8:       return "I8";
        case DType::Int16:      return "I16";
        case DType::Int32:      return "I32";
        case DType::Int64:      return "I64";
        case DType::UInt8:      return "U8";
        case DType::UInt16:     return "U16";
        case DType::UInt32:     return "U32";
        case DType::UInt64:     return "U64";
        case DType::Bool:       return "BOOL";
        case DType::Complex64:  return "C64";   // Tenzor extension
        case DType::Complex128: return "C128";  // Tenzor extension
        case DType::FP8_E4M3:   return "F8_E4M3";
        case DType::FP8_E5M2:   return "F8_E5M2";
        case DType::QInt8:      return "QI8";   // Tenzor extension (raw int8 + qparams)
        case DType::QUInt8:     return "QU8";   // Tenzor extension
        case DType::QInt4x2:    return "QI4X2"; // Tenzor extension (4-bit packed)
    }
    return "F32";
}

auto SafeTensorsSerializer::st_string_to_dtype(const std::string& s) -> DType {
    if (s == "F32")      return DType::Float32;
    if (s == "F64")      return DType::Float64;
    if (s == "F16")      return DType::Float16;
    if (s == "BF16")     return DType::BFloat16;
    if (s == "I8")       return DType::Int8;
    if (s == "I16")      return DType::Int16;
    if (s == "I32")      return DType::Int32;
    if (s == "I64")      return DType::Int64;
    if (s == "U8")       return DType::UInt8;
    if (s == "U16")      return DType::UInt16;
    if (s == "U32")      return DType::UInt32;
    if (s == "U64")      return DType::UInt64;
    if (s == "BOOL")     return DType::Bool;
    if (s == "F8_E4M3")  return DType::FP8_E4M3;
    if (s == "F8_E5M2")  return DType::FP8_E5M2;
    // Audit item I.13 — Tenzor-specific dtype extensions.  Non-Tenzor
    // readers may not recognise these; that is the documented contract.
    if (s == "C64")      return DType::Complex64;
    if (s == "C128")     return DType::Complex128;
    if (s == "QI8")      return DType::QInt8;
    if (s == "QU8")      return DType::QUInt8;
    if (s == "QI4X2")    return DType::QInt4x2;
    throw std::runtime_error("SafeTensors: unknown dtype string '" + s + "'");
}

// ============================================================================
// JSON helpers (minimal, no external dependency)
// ============================================================================

auto SafeTensorsSerializer::escape_json_string(const std::string& s) -> std::string {
    std::string result;
    result.reserve(s.size() + 2);
    for (char c : s) {
        switch (c) {
            case '"':  result += "\\\""; break;
            case '\\': result += "\\\\"; break;
            case '\n': result += "\\n"; break;
            case '\r': result += "\\r"; break;
            case '\t': result += "\\t"; break;
            default:   result += c; break;
        }
    }
    return result;
}

auto SafeTensorsSerializer::build_header_json(
        const std::vector<std::pair<std::string, const Tensor*>>& ordered_tensors,
        const std::vector<size_t>& offsets) -> std::string {
    std::ostringstream ss;
    ss << "{";

    for (size_t i = 0; i < ordered_tensors.size(); ++i) {
        if (i > 0) ss << ",";

        const auto& [name, tensor] = ordered_tensors[i];
        ss << "\"" << escape_json_string(name) << "\":{";
        ss << "\"dtype\":\"" << dtype_to_st_string(tensor->dtype()) << "\",";

        // Shape
        ss << "\"shape\":[";
        auto shape = tensor->shape();
        for (size_t j = 0; j < shape.size(); ++j) {
            if (j > 0) ss << ",";
            ss << shape[j];
        }
        ss << "],";

        // Data offsets (relative to data section start)
        ss << "\"data_offsets\":[" << offsets[i] << "," << offsets[i + 1] << "]";
        ss << "}";
    }

    ss << "}";
    return ss.str();
}

// Minimal JSON parser for SafeTensors header
auto SafeTensorsSerializer::parse_header_json(const std::string& json)
        -> std::unordered_map<std::string, TensorMeta> {
    std::unordered_map<std::string, TensorMeta> result;

    size_t pos = 0;
    auto skip_ws = [&]() { while (pos < json.size() && std::isspace(json[pos])) ++pos; };
    auto expect = [&](char c) {
        skip_ws();
        if (pos >= json.size() || json[pos] != c)
            throw std::runtime_error("SafeTensors JSON parse error: expected '" +
                                      std::string(1, c) + "' at position " + std::to_string(pos));
        ++pos;
    };
    auto parse_string = [&]() -> std::string {
        skip_ws();
        expect('"');
        std::string s;
        while (pos < json.size() && json[pos] != '"') {
            if (json[pos] == '\\' && pos + 1 < json.size()) {
                ++pos;
                switch (json[pos]) {
                    case '"':  s += '"'; break;
                    case '\\': s += '\\'; break;
                    case 'n':  s += '\n'; break;
                    case 'r':  s += '\r'; break;
                    case 't':  s += '\t'; break;
                    default:   s += json[pos]; break;
                }
            } else {
                s += json[pos];
            }
            ++pos;
        }
        expect('"');
        return s;
    };
    auto parse_number = [&]() -> int64_t {
        skip_ws();
        size_t start = pos;
        if (pos < json.size() && json[pos] == '-') ++pos;
        while (pos < json.size() && std::isdigit(json[pos])) ++pos;
        return std::stoll(json.substr(start, pos - start));
    };

    skip_ws();
    expect('{');
    skip_ws();

    while (pos < json.size() && json[pos] != '}') {
        // Parse tensor name
        std::string tensor_name = parse_string();
        expect(':');

        // Skip __metadata__ entries
        skip_ws();
        if (pos < json.size() && json[pos] == '{') {
            // Check if this is a tensor entry or metadata
            size_t save_pos = pos;
            ++pos;
            skip_ws();
            std::string first_key = parse_string();
            pos = save_pos; // restore

            if (first_key == "__metadata__" || tensor_name == "__metadata__") {
                // Skip the entire value
                int depth = 0;
                while (pos < json.size()) {
                    if (json[pos] == '{') ++depth;
                    else if (json[pos] == '}') { --depth; if (depth == 0) { ++pos; break; } }
                    ++pos;
                }
                skip_ws();
                if (pos < json.size() && json[pos] == ',') ++pos;
                skip_ws();
                continue;
            }
        }

        TensorMeta meta;
        expect('{');

        // Parse tensor metadata fields
        while (pos < json.size() && json[pos] != '}') {
            std::string key = parse_string();
            expect(':');

            if (key == "dtype") {
                meta.dtype = st_string_to_dtype(parse_string());
            } else if (key == "shape") {
                expect('[');
                skip_ws();
                while (pos < json.size() && json[pos] != ']') {
                    meta.shape.push_back(parse_number());
                    skip_ws();
                    if (pos < json.size() && json[pos] == ',') ++pos;
                    skip_ws();
                }
                expect(']');
            } else if (key == "data_offsets") {
                expect('[');
                meta.data_start = static_cast<size_t>(parse_number());
                expect(',');
                meta.data_end = static_cast<size_t>(parse_number());
                expect(']');
            }

            skip_ws();
            if (pos < json.size() && json[pos] == ',') ++pos;
            skip_ws();
        }
        expect('}');

        result[tensor_name] = std::move(meta);

        skip_ws();
        if (pos < json.size() && json[pos] == ',') ++pos;
        skip_ws();
    }

    return result;
}

// ============================================================================
// Save
// ============================================================================

void SafeTensorsSerializer::save(
        const std::unordered_map<std::string, Tensor>& state_dict,
        const std::string& path) {
    // Sort tensors by name for deterministic output
    std::vector<std::pair<std::string, const Tensor*>> ordered;
    ordered.reserve(state_dict.size());
    for (const auto& [name, tensor] : state_dict) {
        ordered.emplace_back(name, &tensor);
    }
    std::sort(ordered.begin(), ordered.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });

    // Compute data offsets
    std::vector<size_t> offsets;
    offsets.reserve(ordered.size() + 1);
    size_t current_offset = 0;
    for (const auto& [name, tensor] : ordered) {
        offsets.push_back(current_offset);
        // Move GPU tensors to CPU for serialization
        Tensor cpu_tensor = tensor->device().type != Device::Type::CPU
                            ? tensor->to(Device::cpu()) : *tensor;
        size_t nbytes = cpu_tensor.numel() * dtype_size(cpu_tensor.dtype());
        current_offset += nbytes;
    }
    offsets.push_back(current_offset); // end offset for last tensor

    // Build JSON header
    std::string header_json = build_header_json(ordered, offsets);

    // Write file
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("SafeTensors: cannot open file for writing: " + path);
    }

    // Write header size (8 bytes, little-endian)
    uint64_t header_size = header_json.size();
    file.write(reinterpret_cast<const char*>(&header_size), 8);

    // Write JSON header
    file.write(header_json.data(), static_cast<std::streamsize>(header_json.size()));

    // Write tensor data
    for (const auto& [name, tensor] : ordered) {
        Tensor cpu_tensor = tensor->device().type != Device::Type::CPU
                            ? tensor->to(Device::cpu()) : *tensor;
        Tensor contiguous = cpu_tensor.is_contiguous() ? cpu_tensor : cpu_tensor.contiguous();
        size_t nbytes = contiguous.numel() * dtype_size(contiguous.dtype());
        file.write(static_cast<const char*>(contiguous.data_ptr()), static_cast<std::streamsize>(nbytes));
    }

    if (!file.good()) {
        throw std::runtime_error("SafeTensors: error writing file: " + path);
    }
}

// ============================================================================
// Load
// ============================================================================

auto SafeTensorsSerializer::load(const std::string& path)
        -> std::unordered_map<std::string, Tensor> {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        throw std::runtime_error("SafeTensors: cannot open file: " + path);
    }

    auto file_size = file.tellg();
    file.seekg(0);

    if (file_size < 8) {
        throw std::runtime_error("SafeTensors: file too small: " + path);
    }

    // Read header size
    uint64_t header_size = 0;
    file.read(reinterpret_cast<char*>(&header_size), 8);

    // Validate header size
    if (header_size > 100 * 1024 * 1024) { // 100MB max header
        throw std::runtime_error("SafeTensors: header too large (" +
                                  std::to_string(header_size) + " bytes)");
    }
    if (static_cast<uint64_t>(file_size) < 8 + header_size) {
        throw std::runtime_error("SafeTensors: file truncated (header claims " +
                                  std::to_string(header_size) + " bytes but file is only " +
                                  std::to_string(file_size) + " bytes)");
    }

    // Read JSON header
    std::string header_json(header_size, '\0');
    file.read(header_json.data(), static_cast<std::streamsize>(header_size));

    // Parse header
    auto tensor_metas = parse_header_json(header_json);

    // Data section starts at offset 8 + header_size
    size_t data_section_start = 8 + header_size;

    // Load tensors
    std::unordered_map<std::string, Tensor> result;
    result.reserve(tensor_metas.size());

    for (const auto& [name, meta] : tensor_metas) {
        // Validate offsets (all header fields are attacker-controlled).
        if (meta.data_end < meta.data_start) {
            throw std::runtime_error("SafeTensors: tensor '" + name +
                                      "' has data_end < data_start");
        }
        size_t data_size = meta.data_end - meta.data_start;
        // Both start and end must lie within the file's data section. Compute the
        // available span without overflow, then require data_end <= avail.
        size_t avail = (static_cast<size_t>(file_size) >= data_section_start)
                           ? static_cast<size_t>(file_size) - data_section_start
                           : 0;
        if (meta.data_end > avail) {
            throw std::runtime_error("SafeTensors: tensor '" + name +
                                      "' data extends past end of file");
        }

        // Validate size matches shape * dtype_size, with checked multiplication
        // and rejection of negative dims (overflow could otherwise make
        // expected_size collide with a crafted data_size).
        int64_t numel = 1;
        for (auto dim : meta.shape) {
            if (dim < 0) {
                throw std::runtime_error("SafeTensors: tensor '" + name +
                                          "' has a negative dimension");
            }
            if (__builtin_mul_overflow(numel, dim, &numel)) {
                throw std::runtime_error("SafeTensors: tensor '" + name +
                                          "' element count overflows int64");
            }
        }
        size_t expected_size;
        if (__builtin_mul_overflow(static_cast<size_t>(numel),
                                   dtype_size(meta.dtype), &expected_size)) {
            throw std::runtime_error("SafeTensors: tensor '" + name +
                                      "' byte size overflows size_t");
        }
        if (data_size != expected_size) {
            throw std::runtime_error("SafeTensors: tensor '" + name +
                                      "' data size mismatch (header: " +
                                      std::to_string(data_size) + ", expected: " +
                                      std::to_string(expected_size) + ")");
        }

        // Create tensor and read data
        std::vector<int64_t> shape_vec(meta.shape.begin(), meta.shape.end());
        Tensor tensor(shape_vec, meta.dtype, Device::cpu());

        file.seekg(static_cast<std::streamoff>(data_section_start + meta.data_start));
        file.read(static_cast<char*>(tensor.data_ptr()),
                  static_cast<std::streamsize>(data_size));

        result[name] = std::move(tensor);
    }

    return result;
}

// ============================================================================
// Validation
// ============================================================================

auto SafeTensorsSerializer::is_valid_file(const std::string& path) -> bool {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) return false;

    // Read header size
    uint64_t header_size = 0;
    file.read(reinterpret_cast<char*>(&header_size), 8);
    if (!file.good()) return false;

    // Sanity check: header shouldn't be impossibly large
    if (header_size > 100 * 1024 * 1024) return false;
    if (header_size < 2) return false; // At minimum "{}"

    // Read first byte of header — should be '{'
    char first_char = 0;
    file.read(&first_char, 1);
    return file.good() && first_char == '{';
}

// ============================================================================
// Format detection
// ============================================================================

auto detect_format(const std::string& path) -> SerializeFormat {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) return SerializeFormat::Unknown;

    // Read first 8 bytes
    char buf[8] = {};
    file.read(buf, 8);
    if (!file.good()) return SerializeFormat::Unknown;

    // Check for Tenzor magic number (first 4 bytes)
    uint32_t magic = 0;
    std::memcpy(&magic, buf, 4);
    if (magic == TENZOR_MAGIC) {
        return SerializeFormat::Tenzor;
    }

    // Check for SafeTensors: first 8 bytes are header_size (uint64 LE).
    // If it's a reasonable size and the next byte is '{', it's SafeTensors.
    uint64_t header_size = 0;
    std::memcpy(&header_size, buf, 8);
    if (header_size > 0 && header_size < 100 * 1024 * 1024) {
        char next = 0;
        file.read(&next, 1);
        if (file.good() && next == '{') {
            return SerializeFormat::SafeTensors;
        }
    }

    return SerializeFormat::Unknown;
}

} // namespace tenzor::nn
