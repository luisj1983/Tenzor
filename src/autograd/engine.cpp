#include "tenzor/autograd/engine.hpp"
#include "tenzor/ops/creation.hpp"

namespace tenzor {

auto BackwardEngine::execute(Variable& root, std::optional<Tensor> gradient) -> void {
    if (!root.requires_grad()) {
        return;
    }

    // Initialize gradient
    if (!gradient.has_value()) {
        gradient = ones_like(root.tensor());
    }

    root.grad() = *gradient;

    // TODO: Implement full backward pass with topological sort
}

auto BackwardEngine::topological_sort(std::shared_ptr<Function> root)
    -> std::vector<std::shared_ptr<Function>> {
    // TODO: Implement topological sort
    return {root};
}

auto BackwardEngine::clear_gradients() -> void {
    grad_accumulators_.clear();
}

auto BackwardEngine::accumulate_grad(Function* func, Tensor grad) -> void {
    grad_accumulators_[func].push_back(std::move(grad));
}

auto BackwardEngine::get_accumulated_grads(Function* func) -> std::vector<Tensor> {
    auto it = grad_accumulators_.find(func);
    if (it == grad_accumulators_.end()) {
        return {};
    }
    return it->second;
}

// Global engine
auto backward_engine() -> BackwardEngine& {
    static BackwardEngine engine;
    return engine;
}

} // namespace tenzor
