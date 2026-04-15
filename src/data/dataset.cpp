#include "tenzor/data/dataset.hpp"

namespace tenzor {
namespace data {

static thread_local std::optional<WorkerInfo> g_worker_info;

auto get_worker_info() -> std::optional<WorkerInfo> {
    return g_worker_info;
}

auto set_worker_info(WorkerInfo info) -> void {
    g_worker_info = info;
}

auto clear_worker_info() -> void {
    g_worker_info.reset();
}

} // namespace data
} // namespace tenzor
