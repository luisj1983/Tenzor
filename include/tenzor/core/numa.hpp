/**
 * @file numa.hpp
 * @brief NUMA topology detection and memory allocation utilities
 *
 * Provides NUMA-aware memory allocation for improved locality on
 * multi-socket systems. Falls back gracefully on non-NUMA systems.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace tenzor {
namespace numa {

/**
 * @brief NUMA node information
 */
struct NodeInfo {
    int node_id;                    ///< NUMA node ID
    std::vector<int> cpu_ids;       ///< CPUs on this node
    size_t memory_total{0};         ///< Total memory in bytes (0 if unknown)
    size_t memory_free{0};          ///< Free memory in bytes (0 if unknown)
};

/**
 * @brief NUMA topology information
 */
struct Topology {
    int num_nodes{1};                       ///< Number of NUMA nodes
    std::vector<NodeInfo> nodes;            ///< Per-node info
    bool available{false};                  ///< True if NUMA is available
};

/**
 * @brief Get NUMA topology (cached after first call)
 */
auto get_topology() -> const Topology&;

/**
 * @brief Get the NUMA node of the calling thread
 * @return Node ID, or 0 if NUMA is not available
 */
auto get_current_node() -> int;

/**
 * @brief Allocate memory on a specific NUMA node
 *
 * Uses mbind/mmap or libnuma if available, falls back to aligned_alloc.
 *
 * @param bytes Number of bytes to allocate
 * @param node NUMA node ID (-1 for local node)
 * @param alignment Alignment requirement (default 64 for cache lines)
 * @return Pointer to allocated memory, or nullptr on failure
 */
auto allocate_on_node(size_t bytes, int node = -1, size_t alignment = 64) -> void*;

/**
 * @brief Free memory allocated with allocate_on_node
 */
void free_on_node(void* ptr, size_t bytes);

/**
 * @brief Check if NUMA is available on this system
 */
auto is_available() -> bool;

/**
 * @brief Get the NUMA node for a given CPU ID
 * @return Node ID, or 0 if unknown
 */
auto cpu_to_node(int cpu_id) -> int;

} // namespace numa
} // namespace tenzor
