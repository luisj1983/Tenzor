#include "tenzor/nn/serialize.hpp"
#include "tenzor/core/device.hpp"
#include <algorithm>
#include <stdexcept>
#include <cstring>

namespace tenzor::nn {

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

    file.close();
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

    // Check magic number
    uint32_t magic = read_value<uint32_t>(file);
    if (magic != TENZOR_MAGIC) {
        return false;
    }

    // Check version
    uint32_t version = read_value<uint32_t>(file);
    if (version != TENZOR_SERIALIZE_VERSION) {
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
        default:
            throw std::runtime_error("Unsupported dtype for serialization");
    }

    file.write(static_cast<const char*>(data_ptr), num_bytes);
}

// Read tensor from file
auto Serializer::read_tensor(std::ifstream& file) -> std::pair<std::string, Tensor> {
    // Read name
    std::string name = read_string(file);

    // Read dtype
    DType dtype = uint8_to_dtype(read_value<uint8_t>(file));

    // Read number of dimensions
    uint32_t ndim = read_value<uint32_t>(file);

    // Read shape
    std::vector<int64_t> shape(ndim);
    for (uint32_t i = 0; i < ndim; ++i) {
        shape[i] = read_value<int64_t>(file);
    }

    // Create tensor
    Tensor tensor(shape, dtype, Device::cpu());

    // Read data
    size_t num_bytes = tensor.numel() * dtype_size(dtype);
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
        default:
            throw std::runtime_error("Unsupported dtype for deserialization");
    }

    file.read(static_cast<char*>(data_ptr), num_bytes);

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
    std::string str(size, '\0');
    file.read(str.data(), size);
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
