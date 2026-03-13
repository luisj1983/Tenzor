/**
 * @file op_coverage_report.cpp
 * @brief Backend operation coverage report tool
 *
 * Prints a table of which OpIds are registered on each backend.
 * Optionally compares against a baseline JSON and exits non-zero if coverage drops.
 *
 * Usage:
 *   ./op_coverage_report                    # Print coverage table
 *   ./op_coverage_report --json             # Output JSON
 *   ./op_coverage_report --check baseline.json  # Compare and fail on regression
 */

#include <tenzor/tenzor.hpp>
#include <tenzor/backend/dispatch_table.hpp>
#include <tenzor/ops/op_id.hpp>
#include <tenzor/core/device.hpp>

#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <set>
#include <string>
#include <vector>

using namespace tenzor;

struct BackendInfo {
    std::string name;
    Device::Type type;
};

static const std::vector<BackendInfo> backends = {
    {"CPU",    Device::Type::CPU},
    {"CUDA",   Device::Type::CUDA},
    {"ROCm",   Device::Type::ROCm},
    {"Vulkan", Device::Type::Vulkan},
    {"OneAPI", Device::Type::OneAPI},
};

static void print_table() {
    // Header
    std::printf("%-30s", "Operation");
    for (auto& b : backends) {
        std::printf(" %7s", b.name.c_str());
    }
    std::printf("\n");
    std::printf("%s\n", std::string(30 + backends.size() * 8, '-').c_str());

    // Collect ops per backend
    std::map<Device::Type, std::set<OpId>> supported;
    for (auto& b : backends) {
        auto ops = get_supported_ops(b.type);
        supported[b.type] = std::set<OpId>(ops.begin(), ops.end());
    }

    size_t total_ops = 0;
    std::map<Device::Type, size_t> counts;

    for (size_t i = 0; i < OP_COUNT; ++i) {
        auto op = static_cast<OpId>(i);
        if (op == OpId::OP_COUNT) break;

        // Check if any backend supports it
        bool any = false;
        for (auto& b : backends) {
            if (supported[b.type].count(op)) { any = true; break; }
        }
        if (!any) continue;

        total_ops++;
        auto name = op_id_to_name(op);
        std::printf("%-30.*s", static_cast<int>(name.size()), name.data());
        for (auto& b : backends) {
            bool has = supported[b.type].count(op) > 0;
            if (has) counts[b.type]++;
            std::printf(" %7s", has ? "  ✓" : "  -");
        }
        std::printf("\n");
    }

    // Summary
    std::printf("%s\n", std::string(30 + backends.size() * 8, '-').c_str());
    std::printf("%-30s", "TOTAL");
    for (auto& b : backends) {
        std::printf(" %4zu/%zu", counts[b.type], total_ops);
    }
    std::printf("\n");
}

static void print_json() {
    std::printf("{\n");
    for (size_t bi = 0; bi < backends.size(); ++bi) {
        auto& b = backends[bi];
        auto ops = get_supported_ops(b.type);
        std::printf("  \"%s\": {\n", b.name.c_str());
        std::printf("    \"count\": %zu,\n", ops.size());
        std::printf("    \"ops\": [");
        for (size_t i = 0; i < ops.size(); ++i) {
            auto name = op_id_to_name(ops[i]);
            std::printf("%s\"%.*s\"", i ? ", " : "",
                       static_cast<int>(name.size()), name.data());
        }
        std::printf("]\n");
        std::printf("  }%s\n", bi + 1 < backends.size() ? "," : "");
    }
    std::printf("}\n");
}

static int check_baseline(const char* baseline_path) {
    // Simple check: read baseline JSON, extract counts, compare
    std::ifstream f(baseline_path);
    if (!f) {
        std::fprintf(stderr, "Error: cannot open baseline %s\n", baseline_path);
        return 1;
    }

    // Parse counts from baseline (simple line-by-line search)
    std::map<std::string, size_t> baseline_counts;
    std::string line;
    std::string current_backend;
    while (std::getline(f, line)) {
        // Find backend name: "CPU": {
        for (auto& b : backends) {
            auto key = "\"" + b.name + "\"";
            if (line.find(key) != std::string::npos && line.find('{') != std::string::npos) {
                current_backend = b.name;
            }
        }
        // Find count: "count": 123
        auto pos = line.find("\"count\":");
        if (pos != std::string::npos && !current_backend.empty()) {
            auto num_start = line.find_first_of("0123456789", pos);
            if (num_start != std::string::npos) {
                baseline_counts[current_backend] = std::stoul(line.substr(num_start));
            }
        }
    }

    bool regression = false;
    for (auto& b : backends) {
        auto ops = get_supported_ops(b.type);
        size_t current = ops.size();
        auto it = baseline_counts.find(b.name);
        if (it == baseline_counts.end()) continue;
        size_t baseline = it->second;
        if (current < baseline) {
            std::fprintf(stderr, "REGRESSION: %s coverage dropped from %zu to %zu ops\n",
                        b.name.c_str(), baseline, current);
            regression = true;
        } else if (current > baseline) {
            std::fprintf(stderr, "IMPROVEMENT: %s coverage increased from %zu to %zu ops\n",
                        b.name.c_str(), baseline, current);
        }
    }

    return regression ? 1 : 0;
}

int main(int argc, char* argv[]) {
    tenzor::initialize();

    if (argc >= 2 && std::strcmp(argv[1], "--json") == 0) {
        print_json();
    } else if (argc >= 3 && std::strcmp(argv[1], "--check") == 0) {
        return check_baseline(argv[2]);
    } else {
        print_table();
    }

    tenzor::finalize();
    return 0;
}
