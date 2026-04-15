/**
 * @file export.hpp
 * @brief AOT compilation and export for Tenzor models
 *
 * Provides torch.export-style ahead-of-time compilation. The entry point
 * export_model() traces a module with example inputs, captures the
 * computation graph and state dict, and produces an ExportedProgram that
 * can be serialized, loaded, and executed independently of the original
 * module definition.
 */

#pragma once

#include "../core/tensor.hpp"
#include "../nn/module.hpp"
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace tenzor {
namespace export_ {  // 'export' is a C++ reserved keyword

/**
 * @brief Options controlling the export process.
 */
struct ExportOptions {
    bool strict{true};                          ///< Strict mode (no dynamic control flow)
    bool preserve_module_call_signature{false};  ///< Preserve module call signature metadata
};

/**
 * @brief A self-contained, serializable representation of a traced model.
 *
 * An ExportedProgram bundles a JIT-traced computation graph together with
 * the module's state dict. It can be saved to disk in a compact binary
 * format and later loaded and executed without the original module class.
 *
 * @code
 * auto model = std::make_shared<MyNetwork>();
 * Tensor dummy({1, 784}, DType::Float32, Device::cpu());
 * auto ep = export_model(*model, {dummy});
 * ep.save("model.tzep");
 *
 * auto loaded = ExportedProgram::load("model.tzep");
 * auto outputs = loaded.run({real_input});
 * @endcode
 */
class ExportedProgram {
public:
    ExportedProgram() = default;

    /**
     * @brief Serialize the exported program to a binary file.
     *
     * Format: TZEP magic (0x545A4550) | version | graph (via JIT serializer)
     *         | num_state_dict_entries | [name_len, name, tensor]...
     *
     * @param path Output file path
     */
    auto save(const std::string& path) const -> void;

    /**
     * @brief Load an exported program from a binary file.
     *
     * @param path Input file path
     * @return Loaded ExportedProgram
     * @throws std::runtime_error on format errors
     */
    static auto load(const std::string& path) -> ExportedProgram;

    /**
     * @brief Execute the exported program with runtime inputs.
     *
     * @param inputs Input tensors matching the traced input shapes
     * @return Output tensors
     */
    auto run(const std::vector<Tensor>& inputs) const -> std::vector<Tensor>;

    /**
     * @brief Get the captured state dict.
     *
     * @return Map of parameter/buffer names to tensors
     */
    auto state_dict() const -> const std::unordered_map<std::string, Tensor>&;

    /**
     * @brief Get the number of expected inputs.
     */
    auto num_inputs() const -> size_t;

    /**
     * @brief Get the number of produced outputs.
     */
    auto num_outputs() const -> size_t;

private:
    friend auto export_model(nn::Module& module,
                             const std::vector<Tensor>& example_inputs,
                             const ExportOptions& opts) -> ExportedProgram;

    struct Impl;
    std::shared_ptr<Impl> impl_;
};

/**
 * @brief Export a module via JIT tracing with example inputs.
 *
 * Traces the module's forward pass using the JIT Tracer, captures the
 * resulting computation graph and the module's full state dict, and
 * returns an ExportedProgram.
 *
 * @param module Module to export (must implement forward_impl)
 * @param example_inputs Example input tensors for tracing
 * @param opts Export options (strict mode, etc.)
 * @return ExportedProgram containing the traced graph and state dict
 *
 * @code
 * auto model = std::make_shared<MyNetwork>();
 * model->eval();
 * Tensor dummy({1, 784}, DType::Float32, Device::cpu());
 * auto exported = export_model(*model, {dummy});
 * exported.save("model.tzep");
 * @endcode
 */
auto export_model(nn::Module& module,
                  const std::vector<Tensor>& example_inputs,
                  const ExportOptions& opts = {}) -> ExportedProgram;

}  // namespace export_
}  // namespace tenzor
