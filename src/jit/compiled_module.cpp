/**
 * @file compiled_module.cpp
 * @brief Implementation of CompiledModule and convenience functions
 *
 * Provides the high-level interface for traced JIT modules, including
 * tracing, execution, optimization, serialization, and metadata management.
 */

#include "../../include/tenzor/jit/compiler.hpp"
#include "../../include/tenzor/jit/tracer.hpp"
#include "../../include/tenzor/jit/serialization.hpp"
#include <stdexcept>

namespace tenzor {
namespace jit {

// ============================================================================
// CompiledModule Implementation
// ============================================================================

CompiledModule::CompiledModule(std::shared_ptr<Graph> graph)
    : graph_(std::move(graph)) {}

auto CompiledModule::trace(std::shared_ptr<nn::Module> module,
                            const Variable& example_input) -> std::shared_ptr<CompiledModule> {
    if (!module) {
        throw std::runtime_error("Cannot trace null module");
    }

    // Use the existing jit::trace function to build the graph
    auto graph = jit::trace(module, example_input);

    auto compiled = std::make_shared<CompiledModule>(graph);
    return compiled;
}

auto CompiledModule::trace(std::shared_ptr<nn::Module> module,
                            const Tensor& example_input) -> std::shared_ptr<CompiledModule> {
    return trace(module, Variable(example_input, false));
}

auto CompiledModule::forward(const Variable& input) -> Variable {
    if (!graph_) {
        throw std::runtime_error("CompiledModule has no graph");
    }

    auto results = graph_->forward({input});
    if (results.empty()) {
        throw std::runtime_error("CompiledModule produced no outputs");
    }
    return results[0];
}

auto CompiledModule::forward(const Tensor& input) -> Variable {
    return forward(Variable(input, false));
}

auto CompiledModule::forward(const std::vector<Variable>& inputs) -> std::vector<Variable> {
    if (!graph_) {
        throw std::runtime_error("CompiledModule has no graph");
    }
    return graph_->forward(inputs);
}

auto CompiledModule::optimize_for_inference() -> int {
    if (!graph_) {
        throw std::runtime_error("CompiledModule has no graph to optimize");
    }

    // Use Compiler directly so we can capture the memory plan
    Compiler compiler(true);
    int result = compiler.optimize(*graph_);
    memory_plan_ = compiler.memory_plan();
    return result;
}

auto CompiledModule::save(const std::string& path) const -> void {
    if (!graph_) {
        throw std::runtime_error("CompiledModule has no graph to save");
    }
    // Save the graph using existing serialization
    save_graph(*graph_, path);
}

auto CompiledModule::load(const std::string& path) -> std::shared_ptr<CompiledModule> {
    auto graph = load_graph(path);
    if (!graph) {
        throw std::runtime_error("Failed to load graph from: " + path);
    }
    return std::make_shared<CompiledModule>(graph);
}

auto CompiledModule::add_metadata(const std::string& key, const std::string& value) -> void {
    metadata_[key] = value;
}

auto CompiledModule::get_metadata(const std::string& key) const -> std::string {
    auto it = metadata_.find(key);
    return it != metadata_.end() ? it->second : "";
}

auto CompiledModule::has_metadata(const std::string& key) const -> bool {
    return metadata_.find(key) != metadata_.end();
}

auto CompiledModule::all_metadata() const -> const std::unordered_map<std::string, std::string>& {
    return metadata_;
}

// ============================================================================
// Free function convenience wrappers
// ============================================================================

auto optimize_for_inference(std::shared_ptr<CompiledModule> module) -> int {
    if (!module) {
        throw std::runtime_error("Cannot optimize null module");
    }
    return module->optimize_for_inference();
}

auto save(const std::shared_ptr<CompiledModule>& module, const std::string& path) -> void {
    if (!module) {
        throw std::runtime_error("Cannot save null module");
    }
    module->save(path);
}

auto load(const std::string& path) -> std::shared_ptr<CompiledModule> {
    return CompiledModule::load(path);
}

auto add_metadata(const std::shared_ptr<CompiledModule>& module,
                  const std::string& key, const std::string& value) -> void {
    if (!module) {
        throw std::runtime_error("Cannot add metadata to null module");
    }
    module->add_metadata(key, value);
}

auto get_metadata(const std::shared_ptr<CompiledModule>& module,
                  const std::string& key) -> std::string {
    if (!module) {
        throw std::runtime_error("Cannot get metadata from null module");
    }
    return module->get_metadata(key);
}

} // namespace jit
} // namespace tenzor
