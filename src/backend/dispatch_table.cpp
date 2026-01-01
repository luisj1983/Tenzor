/**
 * @file dispatch_table.cpp
 * @brief Implementation of dispatch table infrastructure
 */

#include "tenzor/backend/dispatch_table.hpp"
#include "tenzor/ops/op_id.hpp"

namespace tenzor {

// Static dispatch tables array - one per device type
std::array<BackendDispatchTable, DEVICE_TYPE_COUNT> DispatchTableRegistry::tables_{};

void BackendDispatchTable::throw_unsupported(OpId op) const {
    throw std::runtime_error(
        std::string("Operation '") + std::string(op_id_to_name(op)) +
        "' not supported on " + device_type_to_string(device_type) + " backend"
    );
}

} // namespace tenzor
