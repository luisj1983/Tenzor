/**
 * @file function_registry.hpp
 * @brief Registry for RPC-callable functions
 */

#pragma once

#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include "types.hpp"

namespace tenzor {
namespace distributed {
namespace rpc {

/**
 * @brief Singleton registry for RPC functions.
 *
 * Functions must be registered before they can be called remotely.
 * Thread-safe for concurrent registration and lookup.
 */
class FunctionRegistry {
public:
    static auto instance() -> FunctionRegistry&;

    /**
     * @brief Register a function for remote execution.
     *
     * @param name Unique function name
     * @param fn Function implementation
     */
    auto register_function(const std::string& name, RpcFunction fn) -> void;

    /**
     * @brief Look up a registered function.
     *
     * Returns a *copy* of the registered function taken under the lock, so
     * the caller owns a stable callable for the duration of the invocation.
     * Returning a raw pointer into functions_ would dangle if a concurrent
     * register_function() rehashes the map or overwrites the entry.
     *
     * @param name Function name
     * @return The function, or std::nullopt if not found
     */
    auto get_function(const std::string& name) const -> std::optional<RpcFunction>;

    /**
     * @brief Check if a function is registered.
     */
    auto has_function(const std::string& name) const -> bool;

private:
    FunctionRegistry() = default;
    mutable std::mutex mutex_;
    std::unordered_map<std::string, RpcFunction> functions_;
};

/**
 * @brief Helper macro for static registration.
 */
#define TENZOR_REGISTER_RPC_FUNCTION(name, fn)                              \
    static auto _rpc_reg_##name = [] {                                      \
        ::tenzor::distributed::rpc::FunctionRegistry::instance()            \
            .register_function(#name, fn);                                  \
        return true;                                                        \
    }()

} // namespace rpc
} // namespace distributed
} // namespace tenzor
