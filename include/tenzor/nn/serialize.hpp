#pragma once

#include <string>
#include <unordered_map>
#include <fstream>
#include "../core/tensor.hpp"

namespace tenzor {
namespace nn {

// Magic number for file format validation
constexpr uint32_t TENZOR_MAGIC = 0x544E5A52; // "TNZR" in ASCII

// File format version (note: not to be confused with library version)
constexpr uint32_t TENZOR_SERIALIZE_VERSION = 1;

// Serialization utilities for saving and loading model state
class Serializer {
public:
    // Save state dictionary to file
    static void save(const std::unordered_map<std::string, Tensor>& state_dict,
                     const std::string& path);

    // Load state dictionary from file
    static auto load(const std::string& path) -> std::unordered_map<std::string, Tensor>;

    // Check if file exists and is valid
    static auto is_valid_file(const std::string& path) -> bool;

private:
    // Helper functions for tensor serialization
    static void write_tensor(std::ofstream& file, const std::string& name, const Tensor& tensor);
    static auto read_tensor(std::ifstream& file) -> std::pair<std::string, Tensor>;

    // Write/read primitives with endianness handling
    template<typename T>
    static void write_value(std::ofstream& file, T value);

    template<typename T>
    static auto read_value(std::ifstream& file) -> T;

    static void write_string(std::ofstream& file, const std::string& str);
    static auto read_string(std::ifstream& file) -> std::string;

    // Convert DType to/from uint8_t for serialization
    static auto dtype_to_uint8(DType dtype) -> uint8_t;
    static auto uint8_to_dtype(uint8_t value) -> DType;
};

} // namespace nn
} // namespace tenzor
