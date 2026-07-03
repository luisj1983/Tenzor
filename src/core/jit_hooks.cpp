#include "tenzor/core/jit_hooks.hpp"

namespace tenzor::detail {

namespace {
// Thread-local so parallel traces on worker threads don't leak state
// into each other. The Tracer itself is thread-local (see
// Tracer::get_instance) so pairing the hook with the tracer 1:1 is
// correct.
thread_local GraphBreakHook tls_graph_break_hook;
thread_local InplaceOpHook tls_inplace_op_hook;
} // namespace

void set_graph_break_hook(GraphBreakHook hook) {
    tls_graph_break_hook = std::move(hook);
}

void notify_graph_break(const std::string& reason) {
    if (tls_graph_break_hook) {
        tls_graph_break_hook(reason);
    }
}

void set_inplace_op_hook(InplaceOpHook hook) {
    tls_inplace_op_hook = std::move(hook);
}

void notify_inplace_op(OpId op, Tensor& target, const Tensor* others,
                       std::size_t num_others, const OpAttributes& attrs,
                       const Tensor* pre_snapshot) {
    if (tls_inplace_op_hook) {
        tls_inplace_op_hook(op, target, others, num_others, attrs, pre_snapshot);
    }
}

bool inplace_op_hook_active() noexcept {
    return static_cast<bool>(tls_inplace_op_hook);
}

} // namespace tenzor::detail
