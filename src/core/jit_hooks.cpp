#include "tenzor/core/jit_hooks.hpp"

namespace tenzor::detail {

namespace {
// Thread-local so parallel traces on worker threads don't leak state
// into each other. The Tracer itself is thread-local (see
// Tracer::get_instance) so pairing the hook with the tracer 1:1 is
// correct.
thread_local GraphBreakHook tls_graph_break_hook;
} // namespace

void set_graph_break_hook(GraphBreakHook hook) {
    tls_graph_break_hook = std::move(hook);
}

void notify_graph_break(const std::string& reason) {
    if (tls_graph_break_hook) {
        tls_graph_break_hook(reason);
    }
}

} // namespace tenzor::detail
