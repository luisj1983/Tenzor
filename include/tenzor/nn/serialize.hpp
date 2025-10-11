/**
 * @file serialize.hpp
 * @brief Model serialization and deserialization
 *
 * Provides utilities for saving and loading neural network model states,
 * enabling checkpoint persistence, model sharing, and training resumption.
 */

#pragma once

#include <string>
#include <unordered_map>
#include <fstream>
#include "../core/tensor.hpp"

namespace tenzor {
namespace nn {

/** @brief Magic number for file format validation ("TNZR" in ASCII) */
constexpr uint32_t TENZOR_MAGIC = 0x544E5A52;

/** @brief File format version for compatibility checking */
constexpr uint32_t TENZOR_SERIALIZE_VERSION = 1;

/**
 * @brief Serialization utilities for model state persistence
 *
 * Provides static methods for saving and loading model state dictionaries.
 * State dictionaries contain model parameters (weights, biases) and optimizer state.
 *
 * **File Format:**
 * - Header: Magic number + version
 * - Data: Tensor name-value pairs with metadata
 * - Endianness: Platform-independent (little-endian)
 *
 * **Use Cases:**
 * - Checkpointing during training
 * - Saving best model based on validation loss
 * - Model sharing and deployment
 * - Transfer learning (loading pretrained weights)
 *
 * @par Thread Safety
 * Static methods are not thread-safe for concurrent access to same file
 *
 * @code
 * // Save model
 * auto state = model.state_dict();
 * Serializer::save(state, "model.pth");
 *
 * // Load model
 * auto loaded_state = Serializer::load("model.pth");
 * model.load_state_dict(loaded_state);
 * @endcode
 *
 * @see Module::state_dict(), Optimizer::state_dict()
 */
class Serializer {
public:
    /**
     * @brief Save state dictionary to file
     *
     * Writes all tensors in state_dict to binary file with metadata.
     * File format includes magic number, version, and tensor data.
     *
     * @param state_dict Map of parameter names to tensors
     * @param path Output file path
     * @throws std::runtime_error if file cannot be opened or written
     */
    static void save(const std::unordered_map<std::string, Tensor>& state_dict,
                     const std::string& path);

    /**
     * @brief Load state dictionary from file
     *
     * Reads tensors from binary file, validating format and version.
     *
     * @param path Input file path
     * @return Map of parameter names to tensors
     * @throws std::runtime_error if file is invalid or incompatible
     */
    static auto load(const std::string& path) -> std::unordered_map<std::string, Tensor>;

    /**
     * @brief Check if file exists and has valid Tenzor format
     *
     * Verifies magic number and version without loading full data.
     *
     * @param path File path to validate
     * @return true if file is valid Tenzor checkpoint, false otherwise
     */
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
