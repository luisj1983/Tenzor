/**
 * @file test_dispatch_interceptor_stack_multidtype.cpp
 * @brief Multi-backend + multi-dtype tests for dispatch interceptor stack
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/backend/dispatch_interceptor.hpp>
#include <tenzor/backend/fast_dispatch.hpp>
#include <tenzor/ops/creation.hpp>
#include "../multi_backend_dtype_fixture.hpp"

#include <atomic>

using namespace tenzor;
using namespace tenzor::testing;

class DispatchInterceptorMultiDTypeTest : public MultiBackendDTypeTest {};

namespace {

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

} // namespace

TEST_P(DispatchInterceptorMultiDTypeTest, PrimaryDispatchRoutesThroughStack) {
    auto a = zeros({4}, dtype(), device());
    auto b = zeros({4}, dtype(), device());

    CallCounter counter;
    {
        InterceptorGuard guard(make_counter_interceptor(counter));
        auto result = dispatch(OpId::Add, std::array<Tensor, 2>{a, b});
    }
    EXPECT_GE(counter.count.load(), 1);
}

TEST_P(DispatchInterceptorMultiDTypeTest, DispatchToDeviceRoutesThroughStack) {
    auto a = zeros({4}, dtype(), device());
    auto b = zeros({4}, dtype(), device());

    CallCounter counter;
    {
        InterceptorGuard guard(make_counter_interceptor(counter));
        (void)dispatch_to_device(OpId::Add, device().type,
                                 std::array<Tensor, 2>{a, b}, {});
    }
    EXPECT_GE(counter.count.load(), 1)
        << "dispatch_to_device() should route through the interceptor stack";
}

TEST_P(DispatchInterceptorMultiDTypeTest, SingleOutputDispatchRoutesThroughStack) {
    auto a = zeros({4}, dtype(), device());
    auto b = zeros({4}, dtype(), device());

    CallCounter counter;
    {
        InterceptorGuard guard(make_counter_interceptor(counter));
        (void)dispatch_single(OpId::Add, std::array<Tensor, 2>{a, b});
    }
    EXPECT_GE(counter.count.load(), 1);
}

TEST_P(DispatchInterceptorMultiDTypeTest, EmptyStackIsNoOp) {
    EXPECT_EQ(DispatchInterceptorStack::depth(), 0u);
    auto a = zeros({4}, dtype(), device());
    auto b = zeros({4}, dtype(), device());
    auto result = dispatch(OpId::Add, std::array<Tensor, 2>{a, b});
    EXPECT_FALSE(result.empty());
}

TEST_P(DispatchInterceptorMultiDTypeTest, ResultHasCorrectDTypeAndDevice) {
    auto a = zeros({4}, dtype(), device());
    auto b = zeros({4}, dtype(), device());

    CallCounter counter;
    std::vector<Tensor> result;
    {
        InterceptorGuard guard(make_counter_interceptor(counter));
        result = dispatch(OpId::Add, std::array<Tensor, 2>{a, b});
    }
    ASSERT_FALSE(result.empty());
    expectDType(result[0]);
    expectDevice(result[0]);
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(DispatchInterceptorMultiDTypeTest);
