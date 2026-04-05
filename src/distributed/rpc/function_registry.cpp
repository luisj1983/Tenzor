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

auto FunctionRegistry::get_function(const std::string& name) const -> const RpcFunction* {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = functions_.find(name);
    return it != functions_.end() ? &it->second : nullptr;
}

auto FunctionRegistry::has_function(const std::string& name) const -> bool {
    std::lock_guard<std::mutex> lock(mutex_);
    return functions_.count(name) > 0;
}

} // namespace rpc
} // namespace distributed
} // namespace tenzor
