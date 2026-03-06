/**
 * @file numa.cpp
 * @brief NUMA topology detection and memory allocation
 *
 * Uses Linux sysfs for topology detection and mbind/mmap for
 * NUMA-aware allocation. Falls back gracefully on non-NUMA systems.
 */

#include "tenzor/core/numa.hpp"

#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <algorithm>
#include <mutex>

#ifdef __linux__
#include <sched.h>
#include <unistd.h>
#include <sys/mman.h>
#include <dirent.h>
// Linux NUMA mbind support
#include <sys/syscall.h>
// MPOL constants
#ifndef MPOL_BIND
#define MPOL_BIND 2
#endif
#ifndef MPOL_MF_STRICT
#define MPOL_MF_STRICT (1 << 0)
#endif
#ifndef MPOL_MF_MOVE
#define MPOL_MF_MOVE (1 << 1)
#endif
#endif

namespace tenzor {
namespace numa {

namespace {

// Parse a CPU list string like "0-3,5,7-9" into a vector of CPU IDs
auto parse_cpu_list(const std::string& str) -> std::vector<int> {
    std::vector<int> cpus;
    std::istringstream iss(str);
    std::string token;

    while (std::getline(iss, token, ',')) {
        // Trim whitespace
        token.erase(std::remove_if(token.begin(), token.end(), ::isspace), token.end());
        if (token.empty()) continue;

        auto dash = token.find('-');
        if (dash != std::string::npos) {
            int start = std::stoi(token.substr(0, dash));
            int end = std::stoi(token.substr(dash + 1));
            for (int i = start; i <= end; ++i) {
                cpus.push_back(i);
            }
        } else {
            cpus.push_back(std::stoi(token));
        }
    }

    return cpus;
}

#ifdef __linux__

auto detect_topology_linux() -> Topology {
    Topology topo;

    // Count NUMA nodes by scanning /sys/devices/system/node/
    std::string node_base = "/sys/devices/system/node/";
    DIR* dir = opendir(node_base.c_str());
    if (!dir) {
        // No NUMA sysfs — single-node system
        topo.num_nodes = 1;
        topo.available = false;
        NodeInfo node;
        node.node_id = 0;
        // List all online CPUs
        std::ifstream online("/sys/devices/system/cpu/online");
        if (online.is_open()) {
            std::string line;
            std::getline(online, line);
            node.cpu_ids = parse_cpu_list(line);
        }
        topo.nodes.push_back(std::move(node));
        return topo;
    }

    std::vector<int> node_ids;
    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        std::string name = entry->d_name;
        if (name.substr(0, 4) == "node") {
            try {
                int id = std::stoi(name.substr(4));
                node_ids.push_back(id);
            } catch (...) {
                // Not a node directory
            }
        }
    }
    closedir(dir);

    std::sort(node_ids.begin(), node_ids.end());

    if (node_ids.empty()) {
        node_ids.push_back(0);
    }

    topo.num_nodes = static_cast<int>(node_ids.size());
    topo.available = (topo.num_nodes > 1);

    for (int nid : node_ids) {
        NodeInfo node;
        node.node_id = nid;

        // Read CPUs for this node
        std::string cpulist_path = node_base + "node" + std::to_string(nid) + "/cpulist";
        std::ifstream cpulist(cpulist_path);
        if (cpulist.is_open()) {
            std::string line;
            std::getline(cpulist, line);
            node.cpu_ids = parse_cpu_list(line);
        }

        // Read memory info
        std::string meminfo_path = node_base + "node" + std::to_string(nid) + "/meminfo";
        std::ifstream meminfo(meminfo_path);
        if (meminfo.is_open()) {
            std::string line;
            while (std::getline(meminfo, line)) {
                if (line.find("MemTotal") != std::string::npos) {
                    // Format: "Node X MemTotal: YYYY kB"
                    auto colon = line.find(':');
                    if (colon != std::string::npos) {
                        std::string val = line.substr(colon + 1);
                        // Remove "kB" suffix
                        auto kb_pos = val.find("kB");
                        if (kb_pos != std::string::npos) {
                            val = val.substr(0, kb_pos);
                        }
                        try {
                            node.memory_total = std::stoull(val) * 1024;
                        } catch (...) {}
                    }
                } else if (line.find("MemFree") != std::string::npos) {
                    auto colon = line.find(':');
                    if (colon != std::string::npos) {
                        std::string val = line.substr(colon + 1);
                        auto kb_pos = val.find("kB");
                        if (kb_pos != std::string::npos) {
                            val = val.substr(0, kb_pos);
                        }
                        try {
                            node.memory_free = std::stoull(val) * 1024;
                        } catch (...) {}
                    }
                }
            }
        }

        topo.nodes.push_back(std::move(node));
    }

    return topo;
}

#endif // __linux__

auto detect_topology() -> Topology {
#ifdef __linux__
    return detect_topology_linux();
#else
    // Non-Linux: report single-node topology
    Topology topo;
    topo.num_nodes = 1;
    topo.available = false;
    NodeInfo node;
    node.node_id = 0;
    topo.nodes.push_back(std::move(node));
    return topo;
#endif
}

// Lazy-initialized global topology
std::once_flag topology_init_flag;
Topology global_topology;

// Build CPU → node mapping for fast lookup
std::vector<int> cpu_node_map;  // cpu_node_map[cpu_id] = node_id

void init_topology() {
    global_topology = detect_topology();

    // Build reverse CPU → node map
    int max_cpu = 0;
    for (auto& node : global_topology.nodes) {
        for (int cpu : node.cpu_ids) {
            max_cpu = std::max(max_cpu, cpu);
        }
    }
    cpu_node_map.resize(max_cpu + 1, 0);
    for (auto& node : global_topology.nodes) {
        for (int cpu : node.cpu_ids) {
            cpu_node_map[cpu] = node.node_id;
        }
    }
}

} // anonymous namespace

auto get_topology() -> const Topology& {
    std::call_once(topology_init_flag, init_topology);
    return global_topology;
}

auto get_current_node() -> int {
#ifdef __linux__
    int cpu = sched_getcpu();
    if (cpu < 0) return 0;

    std::call_once(topology_init_flag, init_topology);

    if (cpu < static_cast<int>(cpu_node_map.size())) {
        return cpu_node_map[cpu];
    }
    return 0;
#else
    return 0;
#endif
}

auto allocate_on_node(size_t bytes, int node, size_t alignment) -> void* {
    if (bytes == 0) return nullptr;

    // Round up to alignment
    size_t aligned_bytes = (bytes + alignment - 1) & ~(alignment - 1);

#ifdef __linux__
    std::call_once(topology_init_flag, init_topology);

    if (node < 0) {
        node = get_current_node();
    }

    if (global_topology.available && node >= 0 && node < global_topology.num_nodes) {
        // Use mmap + mbind for NUMA-local allocation
        void* ptr = mmap(nullptr, aligned_bytes, PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (ptr == MAP_FAILED) {
            return nullptr;
        }

        // Build nodemask bitmask for mbind
        // nodemask is an array of unsigned long, one bit per node
        constexpr size_t bits_per_ulong = sizeof(unsigned long) * 8;
        size_t mask_size = (global_topology.num_nodes + bits_per_ulong - 1) / bits_per_ulong;
        std::vector<unsigned long> nodemask(mask_size, 0);
        nodemask[node / bits_per_ulong] |= (1UL << (node % bits_per_ulong));

        // mbind to bind the allocation to the specified node
        long ret = syscall(SYS_mbind, ptr, aligned_bytes, MPOL_BIND,
                          nodemask.data(), global_topology.num_nodes + 1, MPOL_MF_STRICT);
        if (ret != 0) {
            // mbind failed — the allocation still works, just not NUMA-bound
            // Fall through: mmap'd memory will use default policy
        }

        return ptr;
    }
#endif

    // Fallback: standard aligned allocation
    void* ptr = nullptr;
#ifdef _WIN32
    ptr = _aligned_malloc(aligned_bytes, alignment);
#else
    if (posix_memalign(&ptr, alignment, aligned_bytes) != 0) {
        return nullptr;
    }
#endif
    return ptr;
}

void free_on_node(void* ptr, size_t bytes) {
    if (!ptr) return;

#ifdef __linux__
    std::call_once(topology_init_flag, init_topology);

    if (global_topology.available && bytes > 0) {
        // Memory was allocated with mmap, free with munmap
        size_t aligned_bytes = (bytes + 63) & ~63UL;
        munmap(ptr, aligned_bytes);
        return;
    }
#endif

    // Fallback: standard free
#ifdef _WIN32
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

auto is_available() -> bool {
    std::call_once(topology_init_flag, init_topology);
    return global_topology.available;
}

auto cpu_to_node(int cpu_id) -> int {
    std::call_once(topology_init_flag, init_topology);

    if (cpu_id >= 0 && cpu_id < static_cast<int>(cpu_node_map.size())) {
        return cpu_node_map[cpu_id];
    }
    return 0;
}

} // namespace numa
} // namespace tenzor
