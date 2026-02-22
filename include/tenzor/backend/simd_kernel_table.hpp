/**
 * @file simd_kernel_table.hpp
 * @brief Multi-ISA kernel registration and resolution table
 *
 * Provides a type-safe mechanism for registering multiple implementations of
 * the same kernel at different SIMD levels, and resolving to the best
 * available implementation at runtime based on detected CPU capabilities.
 *
 * Usage:
 * @code
 * // Register kernel variants at different ISA levels
 * SIMDKernelTable table;
 * table.register_variant("add_f32", SIMDLevel::None,    &add_scalar);
 * table.register_variant("add_f32", SIMDLevel::SSE42,   &add_sse42);
 * table.register_variant("add_f32", SIMDLevel::AVX2,    &add_avx2);
 * table.register_variant("add_f32", SIMDLevel::AVX512F, &add_avx512);
 *
 * // Resolve to best available kernel for this CPU
 * auto* kernel = table.resolve<void(*)(const float*, const float*, float*, size_t)>("add_f32");
 * kernel(a, b, out, n);
 *
 * // Or use the macro for compile-time registration
 * TENZOR_REGISTER_SIMD_KERNEL(table, "add_f32", SIMDLevel::AVX2, add_avx2);
 * @endcode
 *
 * The table stores kernels as type-erased void* pointers. The resolve<T>()
 * template casts back to the correct function pointer type. Callers are
 * responsible for ensuring type consistency between registration and resolution.
 */

#pragma once

#include "runtime_simd.hpp"
#include <algorithm>
#include <cassert>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace tenzor {
namespace backend {

/**
 * @brief Entry in the SIMD kernel table: a kernel pointer tagged with its
 *        minimum required SIMD level.
 */
struct SIMDKernelEntry {
    SIMDLevel min_level;  ///< Minimum SIMD level required to use this kernel
    void* kernel_ptr;     ///< Type-erased function pointer

    /// Order by SIMD level descending (highest first) for resolution
    auto operator>(const SIMDKernelEntry& other) const -> bool {
        return static_cast<uint8_t>(min_level) > static_cast<uint8_t>(other.min_level);
    }
};

/**
 * @brief Table for registering and resolving multi-ISA kernel variants.
 *
 * Each kernel is identified by a string name and can have multiple
 * implementations registered at different SIMDLevel values. Resolution
 * picks the implementation with the highest SIMDLevel that does not
 * exceed the CPU's detected level.
 *
 * Thread-safety: registration is protected by a mutex. Resolution after
 * finalization (finalize()) is lock-free. The typical usage pattern is:
 *   1. Register all variants (startup, single-threaded or with locking)
 *   2. Call finalize() to sort and prepare for fast lookup
 *   3. Resolve kernels (hot path, lock-free)
 *
 * If resolve() is called without finalize(), it will still work correctly
 * but will acquire a lock and sort on-demand.
 */
class SIMDKernelTable {
public:
    SIMDKernelTable() = default;

    // Non-copyable, movable
    SIMDKernelTable(const SIMDKernelTable&) = delete;
    auto operator=(const SIMDKernelTable&) -> SIMDKernelTable& = delete;
    SIMDKernelTable(SIMDKernelTable&&) = default;
    auto operator=(SIMDKernelTable&&) -> SIMDKernelTable& = default;

    /**
     * @brief Register a kernel variant for a named operation.
     *
     * Multiple variants can be registered for the same name at different
     * SIMD levels. If the same name+level pair is registered twice, the
     * later registration overwrites the earlier one.
     *
     * @param name   Kernel name (e.g., "add_f32", "matmul_f32")
     * @param level  Minimum SIMD level required to run this kernel
     * @param kernel Type-erased function pointer
     */
    auto register_variant(const std::string& name, SIMDLevel level, void* kernel) -> void {
        std::lock_guard<std::mutex> lock(mutex_);
        finalized_ = false;

        auto& entries = kernels_[name];

        // Check for existing entry at same level and overwrite
        for (auto& entry : entries) {
            if (entry.min_level == level) {
                entry.kernel_ptr = kernel;
                return;
            }
        }

        entries.push_back({level, kernel});
    }

    /**
     * @brief Register a kernel variant using a typed function pointer.
     *
     * Convenience overload that accepts any function pointer type and
     * casts to void* internally.
     *
     * @tparam Func Function pointer type
     * @param name   Kernel name
     * @param level  Minimum SIMD level
     * @param kernel Typed function pointer
     */
    template<typename Func>
    auto register_variant(const std::string& name, SIMDLevel level, Func kernel) -> void {
        register_variant(name, level, reinterpret_cast<void*>(kernel));
    }

    /**
     * @brief Sort all kernel entries by SIMD level for fast resolution.
     *
     * Call this after all registrations are complete and before the hot
     * path begins calling resolve(). Not strictly required (resolve will
     * sort on-demand) but avoids lock contention on the hot path.
     */
    auto finalize() -> void {
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto& [name, entries] : kernels_) {
            // Sort descending by SIMD level so we can iterate and pick first match
            std::sort(entries.begin(), entries.end(),
                [](const SIMDKernelEntry& a, const SIMDKernelEntry& b) {
                    return static_cast<uint8_t>(a.min_level) > static_cast<uint8_t>(b.min_level);
                });
        }
        finalized_ = true;
    }

    /**
     * @brief Resolve the best kernel for the given name.
     *
     * Returns the kernel registered at the highest SIMD level that does
     * not exceed the detected CPU capability. Returns nullptr if no
     * suitable kernel is registered.
     *
     * @param name Kernel name to look up
     * @return Type-erased function pointer, or nullptr if not found
     */
    auto resolve(const std::string& name) const -> void* {
        // Ensure sorted (idempotent if already finalized)
        if (!finalized_) {
            const_cast<SIMDKernelTable*>(this)->finalize();
        }

        auto it = kernels_.find(name);
        if (it == kernels_.end()) {
            return nullptr;
        }

        const auto& entries = it->second;

        // Entries are sorted descending by level. Pick first that the CPU supports.
        // has_simd_feature() handles both x86 and ARM families correctly.
        for (const auto& entry : entries) {
            if (has_simd_feature(entry.min_level)) {
                return entry.kernel_ptr;
            }
        }

        // If nothing matched (not even SIMDLevel::None), try to find a None-level fallback
        for (const auto& entry : entries) {
            if (entry.min_level == SIMDLevel::None) {
                return entry.kernel_ptr;
            }
        }

        return nullptr;
    }

    /**
     * @brief Resolve and cast to a typed function pointer.
     *
     * @tparam T Target function pointer type (e.g., void(*)(float*, size_t))
     * @param name Kernel name
     * @return Typed function pointer, or nullptr if not found
     */
    template<typename T>
    auto resolve(const std::string& name) const -> T {
        static_assert(std::is_pointer_v<T>, "T must be a pointer type");
        return reinterpret_cast<T>(resolve(name));
    }

    /**
     * @brief Check if a kernel with the given name has any registered variants.
     */
    auto has_kernel(const std::string& name) const -> bool {
        auto it = kernels_.find(name);
        return it != kernels_.end() && !it->second.empty();
    }

    /**
     * @brief Get the number of registered variants for a kernel name.
     */
    auto variant_count(const std::string& name) const -> size_t {
        auto it = kernels_.find(name);
        return (it != kernels_.end()) ? it->second.size() : 0;
    }

    /**
     * @brief Get the number of distinct kernel names registered.
     */
    auto kernel_count() const -> size_t {
        return kernels_.size();
    }

    /**
     * @brief Get all registered kernel names.
     */
    auto kernel_names() const -> std::vector<std::string> {
        std::vector<std::string> names;
        names.reserve(kernels_.size());
        for (const auto& [name, _] : kernels_) {
            names.push_back(name);
        }
        return names;
    }

    /**
     * @brief Get all entries for a kernel name (for debugging/introspection).
     *
     * Returns empty vector if name is not found.
     */
    auto get_entries(const std::string& name) const -> const std::vector<SIMDKernelEntry>& {
        static const std::vector<SIMDKernelEntry> empty;
        auto it = kernels_.find(name);
        return (it != kernels_.end()) ? it->second : empty;
    }

    /**
     * @brief Remove all registered kernels.
     */
    auto clear() -> void {
        std::lock_guard<std::mutex> lock(mutex_);
        kernels_.clear();
        finalized_ = false;
    }

private:
    mutable std::mutex mutex_;
    std::unordered_map<std::string, std::vector<SIMDKernelEntry>> kernels_;
    bool finalized_ = false;
};

/**
 * @brief Get the global SIMD kernel table singleton.
 *
 * This is the shared table used by the CPU backend for multi-ISA dispatch.
 * Backend kernel registration functions populate this table, and the dispatch
 * layer queries it.
 */
auto get_simd_kernel_table() -> SIMDKernelTable&;

// ============================================================================
// Registration macros
// ============================================================================

/**
 * @brief Register a kernel variant in a SIMD kernel table.
 *
 * @param table  SIMDKernelTable instance
 * @param name   Kernel name (string literal)
 * @param level  SIMDLevel enum value
 * @param func   Function pointer to register
 */
#define TENZOR_REGISTER_SIMD_KERNEL(table, name, level, func) \
    (table).register_variant((name), (level), reinterpret_cast<void*>(func))

/**
 * @brief Register a kernel variant in the global SIMD kernel table.
 *
 * Convenience macro that uses get_simd_kernel_table().
 */
#define TENZOR_REGISTER_GLOBAL_SIMD_KERNEL(name, level, func) \
    ::tenzor::backend::get_simd_kernel_table().register_variant( \
        (name), (level), reinterpret_cast<void*>(func))

} // namespace backend
} // namespace tenzor
