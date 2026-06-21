/**
 * @file function_registry.cpp
 * @brief Implementation of RPC function registry
 */

#include "tenzor/distributed/rpc/function_registry.hpp"

namespace tenzor {
namespace distributed {
namespace rpc {

auto FunctionRegistry::instance() -> FunctionRegistry& {
    static FunctionRegistry registry;
    return registry;
}

auto FunctionRegistry::register_function(const std::string& name, RpcFunction fn) -> void {
    std::lock_guard<std::mutex> lock(mutex_);
    functions_[name] = std::move(fn);
}

auto FunctionRegistry::get_function(const std::string& name) const -> std::optional<RpcFunction> {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = functions_.find(name);
    if (it == functions_.end()) {
        return std::nullopt;
    }
    // Return a copy held under the lock; the caller owns the callable across
    // the invocation even if a concurrent registration mutates the map.
    return it->second;
}

auto FunctionRegistry::has_function(const std::string& name) const -> bool {
    std::lock_guard<std::mutex> lock(mutex_);
    return functions_.count(name) > 0;
}

auto FunctionRegistry::unregister_function(const std::string& name) -> bool {
    std::lock_guard<std::mutex> lock(mutex_);
    return functions_.erase(name) > 0;
}

auto FunctionRegistry::clear() -> void {
    std::lock_guard<std::mutex> lock(mutex_);
    functions_.clear();
}

} // namespace rpc
} // namespace distributed
} // namespace tenzor
