/**
 * @file dimname.hpp
 * @brief Named dimension support for tensors (experimental)
 *
 * Provides an interned string type for naming tensor dimensions.
 * Dimension names enable name-based indexing, name-aware broadcasting,
 * and automatic name propagation through operations.
 *
 * Design principles:
 * - Zero overhead when tensors are unnamed (std::optional<DimnameList>)
 * - O(1) name comparison via pointer interning
 * - Thread-safe intern pool
 */

#pragma once

#include <string>
#include <string_view>
#include <vector>
#include <unordered_set>
#include <mutex>
#include <shared_mutex>
#include <optional>
#include <stdexcept>

namespace tenzor {

/**
 * @brief An interned dimension name.
 *
 * All identical name strings share a single allocation in a global pool,
 * making comparison O(1) (pointer equality) and copies trivially cheap.
 *
 * A wildcard Dimname (ptr_ == nullptr) represents an unnamed dimension,
 * analogous to PyTorch's `*` wildcard.
 */
class Dimname {
public:
    /// Create a wildcard (unnamed) dimension.
    Dimname() noexcept : ptr_(nullptr) {}

    /// Create a named dimension by interning the given string.
    explicit Dimname(std::string_view name) : ptr_(intern(name)) {}

    /// Create a wildcard sentinel.
    static auto wildcard() noexcept -> Dimname { return Dimname{}; }

    /// Check if this is a wildcard (unnamed dimension).
    [[nodiscard]] auto is_wildcard() const noexcept -> bool { return ptr_ == nullptr; }

    /// Get the name string. Returns empty string_view for wildcard.
    [[nodiscard]] auto name() const noexcept -> std::string_view {
        return ptr_ ? std::string_view(*ptr_) : std::string_view{};
    }

    /// Pointer-based equality (O(1) due to interning).
    auto operator==(const Dimname& other) const noexcept -> bool {
        return ptr_ == other.ptr_;
    }

    auto operator!=(const Dimname& other) const noexcept -> bool {
        return ptr_ != other.ptr_;
    }

    /// Matches if either is wildcard, or both have the same name.
    [[nodiscard]] auto matches(const Dimname& other) const noexcept -> bool {
        if (is_wildcard() || other.is_wildcard()) return true;
        return ptr_ == other.ptr_;
    }

private:
    const std::string* ptr_;

    /// Thread-safe global intern pool (defined in tensor.cpp to avoid ODR issues).
    static auto intern(std::string_view name) -> const std::string*;
};

/// A list of dimension names for a tensor.
using DimnameList = std::vector<Dimname>;

/**
 * @brief Validate that a DimnameList has no duplicate non-wildcard names.
 * @throws std::invalid_argument if duplicates found
 */
inline void validate_dimnames(const DimnameList& names) {
    for (size_t i = 0; i < names.size(); ++i) {
        if (names[i].is_wildcard()) continue;
        for (size_t j = i + 1; j < names.size(); ++j) {
            if (names[i] == names[j]) {
                throw std::invalid_argument(
                    "Duplicate dimension name: '" + std::string(names[i].name()) + "'");
            }
        }
    }
}

/**
 * @brief Find the index of a dimension by name.
 * @param names The dimension name list
 * @param name The name to find
 * @return Index of the dimension
 * @throws std::invalid_argument if not found
 */
inline auto find_dim_by_name(const DimnameList& names, std::string_view name) -> int64_t {
    Dimname target(name);
    for (size_t i = 0; i < names.size(); ++i) {
        if (names[i] == target) return static_cast<int64_t>(i);
    }
    throw std::invalid_argument(
        "Dimension name '" + std::string(name) + "' not found");
}

/**
 * @brief Propagate names through elementwise operations.
 *
 * For each dimension, if either input has a non-wildcard name, the output
 * inherits that name. If both have non-wildcard names, they must match.
 *
 * @throws std::invalid_argument if names conflict
 */


} // namespace tenzor
