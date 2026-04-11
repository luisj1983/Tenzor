/**
 * @file dispatch_table.cpp
 * @brief Implementation of dispatch table infrastructure
 */

#include "tenzor/backend/dispatch_table.hpp"
#include "tenzor/ops/op_id.hpp"
#include "tenzor/utils/logging.hpp"

#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>

namespace tenzor {

// Static dispatch tables array - one per device type
std::array<BackendDispatchTable, DEVICE_TYPE_COUNT> DispatchTableRegistry::tables_{};

void BackendDispatchTable::throw_unsupported(OpId op) const {
    if (!ready.load(std::memory_order_acquire)) {
        throw std::runtime_error(
            std::string(device_type_to_string(device_type)) +
            " backend not available. Ensure tenzor::initialize() has been called "
            "and the backend library is installed."
        );
    }
    throw std::runtime_error(
        std::string("Operation '") + std::string(op_id_to_name(op)) +
        "' not supported on " + device_type_to_string(device_type) + " backend"
    );
}

auto DispatchTableRegistry::validate_coverage(bool strict) -> bool {
    // Walk every OpId and check whether any active (ready) backend has
    // a registered kernel for it. "Active" means the backend has been
    // loaded *and* marked ready — unloaded / probe-failed backends are
    // skipped so their missing coverage doesn't pollute the report.
    std::vector<OpId> uncovered;
    uncovered.reserve(16);

    for (size_t i = 0; i < OP_COUNT; ++i) {
        const auto op = static_cast<OpId>(i);
        // Skip enum gaps: op_id_to_name returns "unknown" for indices
        // that don't correspond to a real OpId. Those are padding in
        // the enum, not operations that need kernels.
        const auto name = op_id_to_name(op);
        if (name == "unknown") continue;

        bool covered = false;
        for (const auto& table : tables_) {
            if (!table.ready.load(std::memory_order_acquire)) continue;
            if (table.has_kernel(op)) {
                covered = true;
                break;
            }
        }
        if (!covered) {
            uncovered.push_back(op);
        }
    }

    if (uncovered.empty()) {
        return true;
    }

    // Build a single human-readable message so the user sees one report,
    // not one warning per op. (Log backends that collapse repeated lines
    // would otherwise hide the scale of the gap.)
    std::ostringstream oss;
    oss << "dispatch coverage: " << uncovered.size()
        << " OpId(s) have no kernel registered in any active backend: ";
    for (size_t i = 0; i < uncovered.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << op_id_to_name(uncovered[i]);
        if (i >= 19 && uncovered.size() > 20) {
            oss << " ... (+" << (uncovered.size() - 20) << " more)";
            break;
        }
    }
    const std::string msg = oss.str();

    // Strict mode: hard error. Triggered by the caller or by the env var.
    const char* env = std::getenv("TENZOR_DISPATCH_STRICT");
    const bool env_strict = (env != nullptr && env[0] != '\0' && env[0] != '0');
    if (strict || env_strict) {
        throw std::runtime_error(msg);
    }
    Logger::instance().warning(msg);
    return false;
}

} // namespace tenzor
