#include "tenzor/nn/serialize.hpp"
#include "tenzor/core/device.hpp"
#include <algorithm>
#include <exception>
#include <stdexcept>
#include <cstring>
#include <cstdint>
#include <limits>
#include <bit>
#include <vector>

namespace tenzor::nn {

namespace {

// ============================================================================
// Endianness helpers
//
// The native Tenzor checkpoint format is little-endian on disk. On little-endian
// hosts (the common case) every conversion below is a no-op; on big-endian hosts
// scalar header fields and each tensor element component are byte-swapped so
// files round-trip identically regardless of host endianness.
// ============================================================================

constexpr bool kHostIsBigEndian = (std::endian::native == std::endian::big);

// Byte width of the scalar component to swap. Complex values are two real
// components stored back-to-back and are swapped per-half so each scalar stays
// little-endian.
inline size_t endian_component_width(DType dtype) {
    switch (dtype) {
        case DType::Complex64:  return 4;  // two float32
        case DType::Complex128: return 8;  // two float64
        default:                return dtype_size(dtype);
    }
}

// In-place host<->little-endian conversion of a raw tensor element buffer.
// Symmetric (its own inverse). No-op on LE hosts / 1-byte components.
inline void convert_tensor_endianness(void* data, size_t nbytes, DType dtype) {
    if constexpr (!kHostIsBigEndian) {
        (void)data; (void)nbytes; (void)dtype;
        return;
    } else {
        const size_t w = endian_component_width(dtype);
        if (w <= 1) return;
        auto* p = static_cast<uint8_t*>(data);
        for (size_t off = 0; off + w <= nbytes; off += w) {
            for (size_t i = 0; i < w / 2; ++i) {
                std::swap(p[off + i], p[off + w - 1 - i]);
            }
        }
    }
}

// Convert a trivially-copyable scalar between host order and little-endian
// (symmetric). No-op on LE hosts / 1-byte scalars.
template<typename T>
inline T to_le_scalar(T value) {
    if constexpr (kHostIsBigEndian && sizeof(T) > 1) {
        auto* p = reinterpret_cast<uint8_t*>(&value);
        for (size_t i = 0; i < sizeof(T) / 2; ++i) {
            std::swap(p[i], p[sizeof(T) - 1 - i]);
        }
    }
    return value;
}

// Returns the number of bytes remaining between the current get position and the
// end of the stream. Used to bound untrusted length/size fields read from a
// (potentially malicious) file so they cannot drive huge allocations or reads
// past the end of the file.
auto stream_remaining(std::ifstream& file) -> std::uintmax_t {
    const auto cur = file.tellg();
    if (cur < 0) {
        return 0;
    }
    file.seekg(0, std::ios::end);
    const auto endpos = file.tellg();
    file.seekg(cur, std::ios::beg);
    if (endpos < 0 || endpos < cur) {
        return 0;
    }
    return static_cast<std::uintmax_t>(endpos - cur);
}

} // namespace

// Save state dictionary to binary file
void Serializer::save(const std::unordered_map<std::string, Tensor>& state_dict,
                      const std::string& path) {
    std::ofstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open file for writing: " + path);
    }

    // Write magic number and version
    write_value(file, TENZOR_MAGIC);
    write_value(file, TENZOR_SERIALIZE_VERSION);

    // Write number of tensors
    write_value(file, static_cast<uint32_t>(state_dict.size()));

    // Write each tensor
    for (const auto& [name, tensor] : state_dict) {
        write_tensor(file, name, tensor);
    }

    // Detect write failures (e.g. disk full / I/O error) before reporting
    // success: an unchecked stream would leave a silently truncated checkpoint.
    if (!file) {
        throw std::runtime_error(
            "Serializer: failed to write checkpoint (I/O error or disk full): " + path);
    }

    file.close();
    if (file.fail()) {
        throw std::runtime_error(
            "Serializer: failed to flush/close checkpoint: " + path);
    }
}

// Load state dictionary from binary file
auto Serializer::load(const std::string& path) -> std::unordered_map<std::string, Tensor> {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open file for reading: " + path);
    }

    // Read and verify magic number
    uint32_t magic = read_value<uint32_t>(file);
    if (magic != TENZOR_MAGIC) {
        throw std::runtime_error("Invalid file format: magic number mismatch");
    }

    // Read and verify version
    uint32_t version = read_value<uint32_t>(file);
    if (version != TENZOR_SERIALIZE_VERSION) {
        throw std::runtime_error("Unsupported file format version: " + std::to_string(version));
    }

    // Read number of tensors
    uint32_t num_tensors = read_value<uint32_t>(file);

    // Read each tensor
    std::unordered_map<std::string, Tensor> state_dict;
    for (uint32_t i = 0; i < num_tensors; ++i) {
        auto [name, tensor] = read_tensor(file);
        state_dict[name] = std::move(tensor);
    }

    file.close();
    return state_dict;
}

// Check if file exists and is valid
auto Serializer::is_valid_file(const std::string& path) -> bool {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        return false;
    }

    // Read the magic + version. read_value throws on a short read, so a
    // truncated file (< 8 bytes) would otherwise propagate an exception out of
    // a predicate that is supposed to simply report validity. Catch it and
    // report the file as invalid instead.
    try {
        uint32_t magic = read_value<uint32_t>(file);
        if (magic != TENZOR_MAGIC) {
            return false;
        }

        uint32_t version = read_value<uint32_t>(file);
        if (version != TENZOR_SERIALIZE_VERSION) {
            return false;
        }
    } catch (const std::exception&) {
        return false;
    }

    return true;
}

// Write tensor to file
void Serializer::write_tensor(std::ofstream& file, const std::string& name, const Tensor& tensor) {
    // Move tensor to CPU if needed for serialization
    // Always use CPU tensor for serialization
    Tensor cpu_tensor = tensor.cpu();

    // Write name
    write_string(file, name);

    // Write dtype
    write_value(file, dtype_to_uint8(cpu_tensor.dtype()));

    // Write number of dimensions
    write_value(file, static_cast<uint32_t>(cpu_tensor.ndim()));

    // Write shape
    for (int64_t dim : cpu_tensor.shape()) {
        write_value(file, static_cast<int64_t>(dim));
    }

    // Write data. Every DType has a well-defined byte width via dtype_size(), so
    // the raw element buffer is written generically — this covers Complex64/128,
    // the FP8 variants, quantized (QInt8/QUInt8/QInt4x2) and the wide unsigned
    // integer types (UInt16/32/64) in addition to the base float/int dtypes.
    size_t num_bytes = cpu_tensor.numel() * dtype_size(cpu_tensor.dtype());
    const void* data_ptr = cpu_tensor.data_ptr();

    if constexpr (kHostIsBigEndian) {
        // Big-endian host: emit a little-endian copy without mutating the
        // tensor's storage.
        const auto* src = static_cast<const uint8_t*>(data_ptr);
        std::vector<uint8_t> le(src, src + num_bytes);
        convert_tensor_endianness(le.data(), num_bytes, cpu_tensor.dtype());
        file.write(reinterpret_cast<const char*>(le.data()),
                   static_cast<std::streamsize>(num_bytes));
    } else {
        file.write(static_cast<const char*>(data_ptr),
                   static_cast<std::streamsize>(num_bytes));
    }
}

// Read tensor from file
auto Serializer::read_tensor(std::ifstream& file) -> std::pair<std::string, Tensor> {
    // Read name
    std::string name = read_string(file);

    // Read dtype and validate it is a dtype we can actually deserialize before
    // it is used to size/allocate anything.
    DType dtype = uint8_to_dtype(read_value<uint8_t>(file));
    switch (dtype) {
        case DType::Float32: case DType::Float64:
        case DType::Int32:   case DType::Int64:
        case DType::UInt8:   case DType::Bool:
        case DType::Float16: case DType::BFloat16:
        case DType::Int8:    case DType::Int16:
        case DType::UInt16:  case DType::UInt32:  case DType::UInt64:
        case DType::Complex64: case DType::Complex128:
        case DType::FP8_E4M3:  case DType::FP8_E5M2:
        case DType::FP8_E4M3FNUZ: case DType::FP8_E5M2FNUZ:
        case DType::QInt8:   case DType::QUInt8:  case DType::QInt4x2:
            break;
        default:
            throw std::runtime_error("Serializer: unknown dtype in file: " +
                std::to_string(static_cast<int>(dtype)));
    }

    // Read number of dimensions. Each dimension consumes 8 bytes, so bound ndim
    // by the bytes remaining in the file to reject corrupt counts.
    uint32_t ndim = read_value<uint32_t>(file);
    if (static_cast<std::uintmax_t>(ndim) * sizeof(int64_t) > stream_remaining(file)) {
        throw std::runtime_error(
            "Serializer: shape dimension count exceeds remaining file size");
    }

    // Read shape with non-negativity and overflow checks on the element count.
    std::vector<int64_t> shape(ndim);
    int64_t numel = 1;
    for (uint32_t i = 0; i < ndim; ++i) {
        shape[i] = read_value<int64_t>(file);
        if (shape[i] < 0) {
            throw std::runtime_error("Serializer: negative dimension in shape");
        }
        if (shape[i] != 0 && numel > std::numeric_limits<int64_t>::max() / shape[i]) {
            throw std::runtime_error("Serializer: tensor element count overflow");
        }
        numel *= shape[i];
    }

    // Create tensor
    Tensor tensor(shape, dtype, Device::cpu());

    // Read data. The destination is sized from the (validated) shape; ensure the
    // file actually contains that many bytes before reading.
    size_t num_bytes = tensor.numel() * dtype_size(dtype);
    if (static_cast<std::uintmax_t>(num_bytes) > stream_remaining(file)) {
        throw std::runtime_error(
            "Serializer: tensor data exceeds remaining file size");
    }
    // The raw element buffer is read generically (dtype already validated above),
    // covering complex, FP8, quantized and wide-unsigned dtypes alongside the
    // base float/int types.
    void* data_ptr = tensor.data_ptr();

    file.read(static_cast<char*>(data_ptr), num_bytes);
    if (!file) {
        throw std::runtime_error(
            "Serializer: unexpected end of file while reading tensor data");
    }
    // On-disk bytes are little-endian; normalize to host order (no-op on LE).
    convert_tensor_endianness(data_ptr, num_bytes, dtype);

    return {name, tensor};
}

// Write primitive value in little-endian byte order (byte-swapped on BE hosts).
template<typename T>
void Serializer::write_value(std::ofstream& file, T value) {
    T le = to_le_scalar(value);
    file.write(reinterpret_cast<const char*>(&le), sizeof(T));
}

// Read primitive value stored little-endian (byte-swapped back on BE hosts).
template<typename T>
auto Serializer::read_value(std::ifstream& file) -> T {
    T value;
    file.read(reinterpret_cast<char*>(&value), sizeof(T));
    if (!file) {
        throw std::runtime_error("Unexpected end of file");
    }
    return to_le_scalar(value);
}

// Write string
void Serializer::write_string(std::ofstream& file, const std::string& str) {
    write_value(file, static_cast<uint32_t>(str.size()));
    file.write(str.data(), str.size());
}

// Read string
auto Serializer::read_string(std::ifstream& file) -> std::string {
    uint32_t size = read_value<uint32_t>(file);
    // Bound the allocation by the bytes actually left in the file: a corrupt or
    // malicious length must never trigger a huge allocation or an over-read.
    if (static_cast<std::uintmax_t>(size) > stream_remaining(file)) {
        throw std::runtime_error(
            "Serializer: string length exceeds remaining file size");
    }
    std::string str(size, '\0');
    file.read(str.data(), size);
    if (!file) {
        throw std::runtime_error(
            "Serializer: unexpected end of file while reading string");
    }
    return str;
}

// Convert DType to uint8_t
auto Serializer::dtype_to_uint8(DType dtype) -> uint8_t {
    return static_cast<uint8_t>(dtype);
}

// Convert uint8_t to DType
auto Serializer::uint8_to_dtype(uint8_t value) -> DType {
    return static_cast<DType>(value);
}

} // namespace tenzor::nn
