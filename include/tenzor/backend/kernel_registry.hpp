/**
 * @file kernel_registry.hpp
 * @brief Macros and utilities for kernel registration
 *
 * Provides convenient macros for registering kernel implementations with
 * the dispatch table. Handles common patterns like unary/binary operations
 * and attribute parsing.
 */

#pragma once

#include <span>
#include <vector>
#include <string>
#include <stdexcept>
#include <charconv>
#include <climits>
#include "../core/tensor.hpp"
#include "../ops/op_id.hpp"
#include "dispatch_table.hpp"

namespace tenzor {

// ============================================================================
// Attribute Parsing Utilities
// ============================================================================

/**
 * @brief Parse attribute value with default fallback.
 *
 * @tparam T Target type
 * @param attrs Attribute map
 * @param key Attribute key
 * @param default_val Default value if key not found
 * @return Parsed value or default
 */
template<typename T>
T parse_attr(const OpAttributes& attrs, const std::string& key, T default_val);

// Specialization for int64_t
template<>
inline int64_t parse_attr<int64_t>(const OpAttributes& attrs, const std::string& key, int64_t default_val) {
    auto it = attrs.find(key);
    if (it == attrs.end()) return default_val;
    int64_t result;
    auto [ptr, ec] = std::from_chars(it->second.data(), it->second.data() + it->second.size(), result);
    return (ec == std::errc{}) ? result : default_val;
}

// Specialization for int
template<>
inline int parse_attr<int>(const OpAttributes& attrs, const std::string& key, int default_val) {
    return static_cast<int>(parse_attr<int64_t>(attrs, key, default_val));
}

// Specialization for size_t
template<>
inline size_t parse_attr<size_t>(const OpAttributes& attrs, const std::string& key, size_t default_val) {
    auto it = attrs.find(key);
    if (it == attrs.end()) return default_val;
    size_t result;
    auto [ptr, ec] = std::from_chars(it->second.data(), it->second.data() + it->second.size(), result);
    return (ec == std::errc{}) ? result : default_val;
}

// Specialization for double
template<>
inline double parse_attr<double>(const OpAttributes& attrs, const std::string& key, double default_val) {
    auto it = attrs.find(key);
    if (it == attrs.end()) return default_val;
    try {
        return std::stod(it->second);
    } catch (...) {
        return default_val;
    }
}

// Specialization for float
template<>
inline float parse_attr<float>(const OpAttributes& attrs, const std::string& key, float default_val) {
    return static_cast<float>(parse_attr<double>(attrs, key, default_val));
}

// Specialization for bool
template<>
inline bool parse_attr<bool>(const OpAttributes& attrs, const std::string& key, bool default_val) {
    auto it = attrs.find(key);
    if (it == attrs.end()) return default_val;
    return it->second == "true" || it->second == "1" || it->second == "yes";
}

// Specialization for string
template<>
inline std::string parse_attr<std::string>(const OpAttributes& attrs, const std::string& key, std::string default_val) {
    auto it = attrs.find(key);
    return (it != attrs.end()) ? it->second : default_val;
}

/**
 * @brief Parse paired attributes (e.g., stride_h/stride_w), falling back to a single value.
 *
 * Checks for "{key}_h" and "{key}_w" first. If both exist, returns them.
 * Otherwise falls back to single "{key}" value (used for both H and W).
 *
 * @param attrs Attribute map
 * @param key Base attribute name (e.g., "stride")
 * @param default_val Default value if neither key is found
 * @return Pair of (H, W) values
 */
inline std::pair<int64_t, int64_t> parse_pair_attr(const OpAttributes& attrs, const std::string& key, int64_t default_val = 1) {
    auto h_it = attrs.find(key + "_h");
    auto w_it = attrs.find(key + "_w");
    if (h_it != attrs.end() && w_it != attrs.end()) {
        return {parse_attr<int64_t>(attrs, key + "_h", default_val),
                parse_attr<int64_t>(attrs, key + "_w", default_val)};
    }
    int64_t val = parse_attr<int64_t>(attrs, key, default_val);
    return {val, val};
}

/**
 * @brief Parse comma-separated int64_t values.
 *
 * @param attrs Attribute map
 * @param key Attribute key
 * @return Vector of parsed values (empty if key not found)
 */
inline std::vector<int64_t> parse_int_list(const OpAttributes& attrs, const std::string& key) {
    std::vector<int64_t> result;
    auto it = attrs.find(key);
    if (it == attrs.end()) return result;

    const std::string& str = it->second;
    size_t start = 0;
    size_t end = str.find(',');

    while (start < str.size()) {
        if (end == std::string::npos) end = str.size();
        int64_t val;
        auto [ptr, ec] = std::from_chars(str.data() + start, str.data() + end, val);
        if (ec == std::errc{}) {
            result.push_back(val);
        }
        start = end + 1;
        end = str.find(',', start);
    }

    return result;
}

// ============================================================================
// Registration Macros
// ============================================================================

/**
 * @brief Register a kernel with custom wrapper.
 *
 * @param table BackendDispatchTable reference
 * @param op_id OpId enum value (without OpId:: prefix)
 * @param kernel_fn Kernel function or lambda
 */
#define TENZOR_REGISTER_KERNEL(table, op_id, kernel_fn) \
    (table).register_kernel(::tenzor::OpId::op_id, (kernel_fn))

/**
 * @brief Register a unary operation kernel.
 *
 * For kernels with signature: Tensor(const Tensor&)
 *
 * @param table BackendDispatchTable reference
 * @param op_id OpId enum value (without OpId:: prefix)
 * @param kernel_fn Unary kernel function
 */
#define TENZOR_REGISTER_UNARY_KERNEL(table, op_id, kernel_fn) \
    (table).register_kernel(::tenzor::OpId::op_id, \
        [](std::span<const ::tenzor::Tensor> inputs, const ::tenzor::OpAttributes&) \
            -> std::vector<::tenzor::Tensor> { \
            return {(kernel_fn)(inputs[0])}; \
        })

/**
 * @brief Register a binary operation kernel.
 *
 * For kernels with signature: Tensor(const Tensor&, const Tensor&)
 *
 * @param table BackendDispatchTable reference
 * @param op_id OpId enum value (without OpId:: prefix)
 * @param kernel_fn Binary kernel function
 */
#define TENZOR_REGISTER_BINARY_KERNEL(table, op_id, kernel_fn) \
    (table).register_kernel(::tenzor::OpId::op_id, \
        [](std::span<const ::tenzor::Tensor> inputs, const ::tenzor::OpAttributes&) \
            -> std::vector<::tenzor::Tensor> { \
            return {(kernel_fn)(inputs[0], inputs[1])}; \
        })

/**
 * @brief Register a unary operation as single-output kernel.
 * Avoids vector<Tensor> allocation overhead for ops returning one Tensor.
 */
#define TENZOR_REGISTER_UNARY_SINGLE_KERNEL(table, op_id, kernel_fn) \
    (table).register_single_output_kernel(::tenzor::OpId::op_id, \
        [](std::span<const ::tenzor::Tensor> inputs, const ::tenzor::OpAttributes&) \
            -> ::tenzor::Tensor { \
            return (kernel_fn)(inputs[0]); \
        })

/**
 * @brief Register a binary operation as single-output kernel.
 * Avoids vector<Tensor> allocation overhead for ops returning one Tensor.
 */
#define TENZOR_REGISTER_BINARY_SINGLE_KERNEL(table, op_id, kernel_fn) \
    (table).register_single_output_kernel(::tenzor::OpId::op_id, \
        [](std::span<const ::tenzor::Tensor> inputs, const ::tenzor::OpAttributes&) \
            -> ::tenzor::Tensor { \
            return (kernel_fn)(inputs[0], inputs[1]); \
        })

/**
 * @brief Register a ternary operation kernel.
 *
 * For kernels with signature: Tensor(const Tensor&, const Tensor&, const Tensor&)
 *
 * @param table BackendDispatchTable reference
 * @param op_id OpId enum value (without OpId:: prefix)
 * @param kernel_fn Ternary kernel function
 */
#define TENZOR_REGISTER_TERNARY_KERNEL(table, op_id, kernel_fn) \
    (table).register_kernel(::tenzor::OpId::op_id, \
        [](std::span<const ::tenzor::Tensor> inputs, const ::tenzor::OpAttributes&) \
            -> std::vector<::tenzor::Tensor> { \
            return {(kernel_fn)(inputs[0], inputs[1], inputs[2])}; \
        })

/**
 * @brief Register an inplace operation kernel.
 *
 * For kernels with signature: void(Tensor&, const Tensor&)
 * Returns the modified first input.
 *
 * @param table BackendDispatchTable reference
 * @param op_id OpId enum value (without OpId:: prefix)
 * @param kernel_fn Inplace kernel function
 */
#define TENZOR_REGISTER_INPLACE_KERNEL(table, op_id, kernel_fn) \
    (table).register_inplace_kernel(::tenzor::OpId::op_id, \
        [](::tenzor::Tensor& target, std::span<const ::tenzor::Tensor> others, \
           const ::tenzor::OpAttributes&) -> ::tenzor::Tensor& { \
            (kernel_fn)(target, others[0]); \
            return target; \
        })

/**
 * @brief Register a reduction operation kernel.
 *
 * For kernels with signature: Tensor(const Tensor&, int64_t dim, bool keepdim)
 * Parses "dim" and "keepdim" from attributes.
 *
 * @param table BackendDispatchTable reference
 * @param op_id OpId enum value (without OpId:: prefix)
 * @param kernel_fn Reduction kernel function
 */
#define TENZOR_REGISTER_REDUCTION_KERNEL(table, op_id, kernel_fn) \
    (table).register_kernel(::tenzor::OpId::op_id, \
        [](std::span<const ::tenzor::Tensor> inputs, const ::tenzor::OpAttributes& attrs) \
            -> std::vector<::tenzor::Tensor> { \
            /* Use LLONG_MIN as sentinel for "reduce all dimensions" (no dim specified) */ \
            int64_t dim = ::tenzor::parse_attr<int64_t>(attrs, "dim", LLONG_MIN); \
            bool keepdim = ::tenzor::parse_attr<bool>(attrs, "keepdim", false); \
            return {(kernel_fn)(inputs[0], dim, keepdim)}; \
        })

} // namespace tenzor
