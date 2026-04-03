/**
 * @file safetensors.hpp
 * @brief SafeTensors format support for ecosystem interoperability
 *
 * Implements reading and writing of the SafeTensors format, enabling
 * interoperability with HuggingFace models and other frameworks.
 *
 * SafeTensors format:
 *   [8 bytes: header_size as little-endian uint64]
 *   [header_size bytes: JSON header]
 *   [remaining bytes: raw tensor data]
 *
 * JSON header maps tensor names to:
 *   {"dtype": "F32", "shape": [3, 4], "data_offsets": [start, end]}
 *
 * Data offsets are relative to the start of the data section.
 *
 * @see https://github.com/huggingface/safetensors
 */

#pragma once

#include <string>
#include <unordered_map>
#include <vector>
#include "../core/tensor.hpp"

namespace tenzor::nn {

/**
 * @brief SafeTensors format serializer/deserializer
 *
 * Provides static methods for reading and writing the SafeTensors format.
 * This format is designed to be safe (no code execution during loading),
 * fast (zero-copy possible), and simple.
 *
 * @code
 * // Save model in SafeTensors format
 * auto state = model.state_dict();
 * SafeTensorsSerializer::save(state, "model.safetensors");
 *
 * // Load model from SafeTensors
 * auto loaded = SafeTensorsSerializer::load("model.safetensors");
 * model.load_state_dict(loaded);
 * @endcode
 */
class SafeTensorsSerializer {
public:
    /**
     * @brief Save state dictionary in SafeTensors format
     *
     * @param state_dict Map of tensor names to tensors
     * @param path Output file path
     * @throws std::runtime_error if file cannot be written
     */
    static void save(const std::unordered_map<std::string, Tensor>& state_dict,
                     const std::string& path);

    /**
     * @brief Load state dictionary from SafeTensors file
     *
     * @param path Input file path
     * @return Map of tensor names to tensors (on CPU)
     * @throws std::runtime_error if file is invalid or corrupted
     */
    static auto load(const std::string& path) -> std::unordered_map<std::string, Tensor>;

    /**
     * @brief Check if file is a valid SafeTensors file
     *
     * Validates header size and basic JSON structure without loading data.
     *
     * @param path File path to check
     * @return true if file appears to be valid SafeTensors format
     */
    static auto is_valid_file(const std::string& path) -> bool;

private:
    // DType <-> SafeTensors dtype string mapping
    static auto dtype_to_st_string(DType dtype) -> std::string;
    static auto st_string_to_dtype(const std::string& s) -> DType;

    // Minimal JSON helpers (no external dependency)
    static auto build_header_json(
        const std::vector<std::pair<std::string, const Tensor*>>& ordered_tensors,
        const std::vector<size_t>& offsets) -> std::string;

    struct TensorMeta {
        DType dtype;
        std::vector<int64_t> shape;
        size_t data_start;
        size_t data_end;
    };

    static auto parse_header_json(const std::string& json)
        -> std::unordered_map<std::string, TensorMeta>;

    // JSON string escaping
    static auto escape_json_string(const std::string& s) -> std::string;
};

/**
 * @brief Detect serialization format from file contents
 */
enum class SerializeFormat {
    Tenzor,       ///< Native Tenzor format (magic 0x544E5A52)
    SafeTensors,  ///< HuggingFace SafeTensors format
    Unknown       ///< Unrecognized format
};

/**
 * @brief Auto-detect the serialization format of a file
 *
 * Reads the first few bytes to determine the format.
 *
 * @param path File path to check
 * @return Detected format
 */
auto detect_format(const std::string& path) -> SerializeFormat;

} // namespace tenzor::nn
