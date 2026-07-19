/**
 * @file test_registration_report.cpp
 * @brief Phase 4B: Informational test that generates a kernel coverage matrix.
 *
 * Iterates every valid OpId and checks each backend's dispatch table, then
 * prints a human-readable summary table to stdout.  Individual per-backend
 * coverage counts are recorded via RecordProperty so they appear in CTest XML
 * output as well.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/ops/op_id.hpp>
#include <tenzor/backend/dispatch_table.hpp>
#include "parity_test_utils.hpp"
#include "required_ops.hpp"
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

using namespace tenzor;
using namespace tenzor::testing;

namespace {

struct BackendInfo {
    const char* name;
    Device::Type type;
};

const std::vector<BackendInfo> kBackends = {
    {"CPU",    Device::Type::CPU},
    {"CUDA",   Device::Type::CUDA},
    {"ROCm",   Device::Type::ROCm},
    {"Vulkan", Device::Type::Vulkan},
    {"OneAPI", Device::Type::OneAPI},
    {"MPS",    Device::Type::MPS},
};

}  // namespace

TEST(RegistrationReport, CoverageMatrix) {
    tenzor::initialize();

    // Determine which backends are available. CPU goes through
    // is_backend_available() too (not an unconditional push_back), so
    // TENZOR_SKIP_BACKENDS=cpu is honored uniformly with every other
    // backend rather than being silently ignored for CPU alone.
    std::vector<BackendInfo> available;
    for (const auto& backend : kBackends) {
        if (is_backend_available(backend.type)) {
            available.push_back(backend);
        }
    }

    // Collect valid OpIds
    std::vector<OpId> valid_ops;
    for (uint16_t i = 0; i < static_cast<uint16_t>(OpId::OP_COUNT); ++i) {
        auto op = static_cast<OpId>(i);
        if (is_valid_op_id(op) && op_id_to_name(op) != "unknown") {
            valid_ops.push_back(op);
        }
    }

    int total_ops = static_cast<int>(valid_ops.size());
    std::map<std::string, int> backend_coverage;

    // Build the matrix: op name -> per-backend support
    // Also count per-backend totals
    for (const auto& be : available) {
        backend_coverage[be.name] = 0;
    }

    // ------------------------------------------------------------------
    // Print header
    // ------------------------------------------------------------------
    constexpr int name_width = 30;
    constexpr int id_width   = 5;
    constexpr int col_width  = 8;

    std::cout << "\n"
              << std::string(72, '=') << "\n"
              << "  Kernel Registration Coverage Matrix\n"
              << std::string(72, '=') << "\n\n";

    std::cout << std::left << std::setw(name_width) << "Operation"
              << std::right << std::setw(id_width) << "ID";
    for (const auto& be : available) {
        std::cout << std::setw(col_width) << be.name;
    }
    std::cout << "\n"
              << std::string(name_width + id_width + col_width * static_cast<int>(available.size()), '-')
              << "\n";

    // ------------------------------------------------------------------
    // Print each op row
    // ------------------------------------------------------------------
    int missing_anywhere = 0;
    for (auto op : valid_ops) {
        auto name = op_id_to_name(op);
        std::cout << std::left << std::setw(name_width) << name
                  << std::right << std::setw(id_width) << static_cast<uint16_t>(op);

        bool any_missing = false;
        for (const auto& be : available) {
            const auto& table = DispatchTableRegistry::get_table_const(be.type);
            bool has = table.has_kernel(op);
            std::cout << std::setw(col_width) << (has ? "yes" : "-");
            if (has) {
                backend_coverage[be.name]++;
            } else {
                any_missing = true;
            }
        }
        std::cout << "\n";
        if (any_missing) missing_anywhere++;
    }

    // ------------------------------------------------------------------
    // Summary
    // ------------------------------------------------------------------
    std::cout << "\n" << std::string(72, '=') << "\n"
              << "  Summary\n"
              << std::string(72, '=') << "\n\n";

    std::cout << "Total valid operations: " << total_ops << "\n\n";

    std::cout << std::left << std::setw(12) << "Backend"
              << std::right << std::setw(10) << "Registered"
              << std::setw(10) << "Missing"
              << std::setw(12) << "Coverage"
              << "\n"
              << std::string(44, '-') << "\n";

    for (const auto& be : available) {
        int reg = backend_coverage[be.name];
        int miss = total_ops - reg;
        double pct = total_ops > 0 ? (100.0 * reg / total_ops) : 0.0;
        std::cout << std::left << std::setw(12) << be.name
                  << std::right << std::setw(10) << reg
                  << std::setw(10) << miss
                  << std::setw(11) << std::fixed << std::setprecision(1) << pct << "%"
                  << "\n";
    }

    std::cout << "\nOperations missing on at least one available backend: "
              << missing_anywhere << "\n\n";

    // ------------------------------------------------------------------
    // RecordProperty for CTest XML / JUnit output
    // ------------------------------------------------------------------
    RecordProperty("TotalOps", total_ops);
    for (const auto& be : available) {
        RecordProperty(std::string(be.name) + "_Registered", backend_coverage[be.name]);
        RecordProperty(std::string(be.name) + "_Missing", total_ops - backend_coverage[be.name]);
    }

    // ------------------------------------------------------------------
    // Enforcement: the report must FAIL when a backend is missing kernels
    // it is required to have. The matrix above is informational; the gate
    // below is not. CPU is always available and must implement every op in
    // the required-op floor (get_required_ops()) — a CPU dispatch table
    // missing any required kernel is a hard build/registration regression.
    // Every other available backend must also cover the required floor
    // (mirroring KernelCompleteness), so a backend missing 100% of its
    // kernels can no longer slip through with a green report.
    // ------------------------------------------------------------------
    const auto required = get_required_ops();
    for (const auto& be : available) {
        const auto& table = DispatchTableRegistry::get_table_const(be.type);
        std::vector<std::string> missing_required;
        for (auto op : required) {
            if (!table.has_kernel(op)) {
                missing_required.emplace_back(std::string(op_id_to_name(op)));
            }
        }
        std::ostringstream missing_list;
        for (size_t i = 0; i < missing_required.size(); ++i) {
            if (i > 0) missing_list << ", ";
            missing_list << missing_required[i];
        }
        EXPECT_TRUE(missing_required.empty())
            << be.name << " backend is missing " << missing_required.size()
            << " of " << required.size() << " required kernels:\n  "
            << missing_list.str();
    }
}
