#include "tenzor/backend/dispatch.hpp"
#include "tenzor/backend/loader.hpp"
#include "tenzor/backend/dispatch_interceptor.hpp"

namespace tenzor {

// Single out-of-line definition of the dispatch-interceptor thread-local stack.
// Defining it here (one TU) guarantees a single thread_local instance shared by
// every push()/pop()/depth()/run() caller across all translation units; a
// header-inline definition could be duplicated, causing interceptors (e.g. the
// JIT tracer) to be invisible to dispatches made from other TUs.
auto DispatchInterceptorStack::stack_() -> std::vector<DispatchInterceptor>& {
    static thread_local std::vector<DispatchInterceptor> s;
    return s;
}

// Process-global interceptor registry (NOT thread_local): one instance shared
// by every thread so a globally-installed interceptor (e.g. cross-thread
// profiling) is consulted on all worker-thread dispatches. Defined here in a
// single TU to guarantee exactly one instance of each, mirroring stack_().
auto DispatchInterceptorStack::global_mutex_() -> std::shared_mutex& {
    static std::shared_mutex m;
    return m;
}

auto DispatchInterceptorStack::global_() -> std::vector<DispatchInterceptor>& {
    static std::vector<DispatchInterceptor> g;
    return g;
}

auto DispatchInterceptorStack::global_count_() -> std::atomic<std::size_t>& {
    static std::atomic<std::size_t> c{0};
    return c;
}

auto Dispatcher::get_backend(std::span<const Tensor> tensors) -> Backend* {
    if (tensors.empty()) {
        return nullptr;
    }

    auto device_type = tensors[0].device().type;
    return backend_registry().get_backend(device_type);
}

auto Dispatcher::check_device_compatibility(std::span<const Tensor> tensors) -> bool {
    if (tensors.empty()) {
        return true;
    }

    auto first_device = tensors[0].device();

    for (const auto& tensor : tensors) {
        if (tensor.device() != first_device) {
            return false;
        }
    }

    return true;
}

} // namespace tenzor
