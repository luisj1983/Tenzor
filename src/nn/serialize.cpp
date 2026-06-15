#include "tenzor/nn/serialize.hpp"
#include "tenzor/core/device.hpp"
#include <algorithm>
#include <exception>
#include <stdexcept>
#include <cstring>
#include <cstdint>
#include <limits>

namespace tenzor::nn {

namespace {

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

    // Write data
    size_t num_bytes = cpu_tensor.numel() * dtype_size(cpu_tensor.dtype());
    const void* data_ptr = nullptr;

    // Get data pointer based on dtype
    switch (cpu_tensor.dtype()) {
        case DType::Float32:
            data_ptr = cpu_tensor.data<float>();
            break;
        case DType::Float64:
            data_ptr = cpu_tensor.data<double>();
            break;
        case DType::Int32:
            data_ptr = cpu_tensor.data<int32_t>();
            break;
        case DType::Int64:
            data_ptr = cpu_tensor.data<int64_t>();
            break;
        case DType::UInt8:
            data_ptr = cpu_tensor.data<uint8_t>();
            break;
        case DType::Bool:
            data_ptr = cpu_tensor.data<bool>();
            break;
        case DType::Float16:
            data_ptr = cpu_tensor.data<Float16>();
            break;
        case DType::BFloat16:
            data_ptr = cpu_tensor.data<BFloat16>();
            break;
        case DType::Int8:
            data_ptr = cpu_tensor.data<int8_t>();
            break;
        case DType::Int16:
            data_ptr = cpu_tensor.data<int16_t>();
            break;
        default:
            throw std::runtime_error("Unsupported dtype for serialization: " +
                std::string(dtype_name(cpu_tensor.dtype())));
    }

    file.write(static_cast<const char*>(data_ptr), num_bytes);
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
    void* data_ptr = nullptr;

    switch (dtype) {
        case DType::Float32:
            data_ptr = tensor.data<float>();
            break;
        case DType::Float64:
            data_ptr = tensor.data<double>();
            break;
        case DType::Int32:
            data_ptr = tensor.data<int32_t>();
            break;
        case DType::Int64:
            data_ptr = tensor.data<int64_t>();
            break;
        case DType::UInt8:
            data_ptr = tensor.data<uint8_t>();
            break;
        case DType::Bool:
            data_ptr = tensor.data<bool>();
            break;
        case DType::Float16:
            data_ptr = tensor.data<Float16>();
            break;
        case DType::BFloat16:
            data_ptr = tensor.data<BFloat16>();
            break;
        case DType::Int8:
            data_ptr = tensor.data<int8_t>();
            break;
        case DType::Int16:
            data_ptr = tensor.data<int16_t>();
            break;
        default:
            throw std::runtime_error("Unsupported dtype for deserialization: " +
                std::string(dtype_name(dtype)));
    }

    file.read(static_cast<char*>(data_ptr), num_bytes);
    if (!file) {
        throw std::runtime_error(
            "Serializer: unexpected end of file while reading tensor data");
    }

    return {name, tensor};
}

// Write primitive value with proper byte order
template<typename T>
void Serializer::write_value(std::ofstream& file, T value) {
    file.write(reinterpret_cast<const char*>(&value), sizeof(T));
}

// Read primitive value with proper byte order
template<typename T>
auto Serializer::read_value(std::ifstream& file) -> T {
    T value;
    file.read(reinterpret_cast<char*>(&value), sizeof(T));
    if (!file) {
        throw std::runtime_error("Unexpected end of file");
    }
    return value;
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
