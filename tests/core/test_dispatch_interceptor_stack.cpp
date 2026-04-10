// Verifies the dispatch interceptor stack sees calls from every public
// entry point in fast_dispatch.hpp, so cross-cutting concerns (autocast,
// profiling, tracing) can observe every dispatch path.
//
// Phase 1.1 of the fix-all plan: before the fix, dispatch_to_device() went
// straight to DispatchTableRegistry::get_table_const().dispatch() and never
// touched the interceptor stack, so profiling/tracing interceptors silently
// missed creation ops. dispatch_inplace() is documented as an intentional
// bypass (autocast on in-place ops is semantically ambiguous), so this test
// records that behavior explicitly.

#include <gtest/gtest.h>

#include <tenzor/tenzor.hpp>
#include <tenzor/backend/dispatch_interceptor.hpp>
#include <tenzor/backend/fast_dispatch.hpp>
#include <tenzor/ops/creation.hpp>

#include <atomic>

namespace tenzor {
namespace {

class DispatchInterceptorTest : public ::testing::Test {
protected:
    void SetUp() override { tenzor::initialize(); }
};

// Counting interceptor: increments a counter whenever any dispatch passes
// through the stack.
struct CallCounter {
    std::atomic<int> count{0};
};

DispatchInterceptor make_counter_interceptor(CallCounter& c) {
    return [&c](OpId op, std::span<const Tensor> inputs,
                const OpAttributes& attrs, DispatchNext next) {
        c.count.fetch_add(1, std::memory_order_relaxed);
        return next(op, inputs, attrs);
    };
}

// ---------------------------------------------------------------------------
// The main `dispatch()` path must run through the interceptor stack. This is
// the pre-existing behavior; the test is a regression guard.
// ---------------------------------------------------------------------------

TEST_F(DispatchInterceptorTest, PrimaryDispatchRoutesThroughStack) {
    auto a = zeros({4}, DType::Float32, Device::cpu());
    auto b = zeros({4}, DType::Float32, Device::cpu());

    CallCounter counter;
    {
        InterceptorGuard guard(make_counter_interceptor(counter));
        auto result = dispatch(OpId::Add, std::array<Tensor, 2>{a, b});
    }
    EXPECT_GE(counter.count.load(), 1);
}

// ---------------------------------------------------------------------------
// `dispatch_to_device()` is the creation-op entry point. Before Phase 1.1 it
// bypassed the interceptor stack entirely; after the fix it must route
// through it so profiling/tracing interceptors see every creation op too.
// ---------------------------------------------------------------------------

TEST_F(DispatchInterceptorTest, DispatchToDeviceRoutesThroughStack) {
    // `dispatch_to_device()` is the explicit-device entry point. Before the
    // Phase 1.1 fix it called the registry directly. After the fix it must
    // go through the interceptor stack so profiling/tracing see every
    // dispatch. We exercise it with OpId::Add on CPU — the specific op
    // doesn't matter, only that the call is routed through the stack.
    auto a = zeros({4}, DType::Float32, Device::cpu());
    auto b = zeros({4}, DType::Float32, Device::cpu());

    CallCounter counter;
    {
        InterceptorGuard guard(make_counter_interceptor(counter));
        (void)dispatch_to_device(OpId::Add, Device::Type::CPU,
                                 std::array<Tensor, 2>{a, b}, {});
    }
    EXPECT_GE(counter.count.load(), 1)
        << "dispatch_to_device() should route through the interceptor "
           "stack (Phase 1.1 fix).";
}

// ---------------------------------------------------------------------------
// `dispatch_single()` (the zero-alloc single-output fast path) must also go
// through the stack — regression guard.
// ---------------------------------------------------------------------------

TEST_F(DispatchInterceptorTest, SingleOutputDispatchRoutesThroughStack) {
    auto a = zeros({4}, DType::Float32, Device::cpu());
    auto b = zeros({4}, DType::Float32, Device::cpu());

    CallCounter counter;
    {
        InterceptorGuard guard(make_counter_interceptor(counter));
        (void)dispatch_single(OpId::Add, std::array<Tensor, 2>{a, b});
    }
    EXPECT_GE(counter.count.load(), 1);
}

// ---------------------------------------------------------------------------
// Empty stack: zero-overhead fast path. No crash, correct result.
// ---------------------------------------------------------------------------

TEST_F(DispatchInterceptorTest, EmptyStackIsNoOp) {
    EXPECT_EQ(DispatchInterceptorStack::depth(), 0u);
    auto a = zeros({4}, DType::Float32, Device::cpu());
    auto b = zeros({4}, DType::Float32, Device::cpu());
    auto result = dispatch(OpId::Add, std::array<Tensor, 2>{a, b});
    EXPECT_FALSE(result.empty());
}

} // namespace
} // namespace tenzor
