/**
 * @file test_dispatch_interceptor_multidtype.cpp
 * @brief Multi-backend tests for the dispatch interceptor stack.
 */

#include <gtest/gtest.h>
#include "tenzor/tenzor.hpp"
#include "tenzor/core/tensor.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/backend/dispatch_interceptor.hpp"
#include "../backend_test_fixture.hpp"
#include <atomic>

using namespace tenzor;
using namespace tenzor::testing;

class DispatchInterceptorMultiBackendTest : public BackendTest {
protected:
    void SetUp() override {
        BackendTest::SetUp();
        // Ensure clean stack before each test
        while (DispatchInterceptorStack::depth() > 0) {
            DispatchInterceptorStack::pop();
        }
    }
};

TEST_P(DispatchInterceptorMultiBackendTest, EmptyStackPassthrough) {
    // With no interceptors, dispatch should work normally
    EXPECT_EQ(DispatchInterceptorStack::depth(), 0u);

    auto a = ones({4, 4}, DType::Float32, device);
    auto b = ones({4, 4}, DType::Float32, device);
    auto c = tenzor::add(a, b);

    EXPECT_EQ(c.dtype(), DType::Float32);
    auto c_cpu = c.to(Device::cpu());
    EXPECT_NEAR(c_cpu.data<float>()[0], 2.0f, 1e-6f);
}

TEST_P(DispatchInterceptorMultiBackendTest, IdentityInterceptor) {
    // An interceptor that just passes through to next
    InterceptorGuard guard([](OpId op, std::span<const Tensor> inputs,
                              const OpAttributes& attrs, DispatchNext next) {
        return next(op, inputs, attrs);
    });

    EXPECT_EQ(DispatchInterceptorStack::depth(), 1u);

    auto a = ones({4, 4}, DType::Float32, device);
    auto b = ones({4, 4}, DType::Float32, device);
    auto c = tenzor::add(a, b);

    auto c_cpu = c.to(Device::cpu());
    EXPECT_NEAR(c_cpu.data<float>()[0], 2.0f, 1e-6f);
}

TEST_P(DispatchInterceptorMultiBackendTest, CountingInterceptor) {
    // Interceptor that counts dispatch calls
    std::atomic<int> count{0};

    InterceptorGuard guard([&count](OpId op, std::span<const Tensor> inputs,
                                     const OpAttributes& attrs, DispatchNext next) {
        count.fetch_add(1, std::memory_order_relaxed);
        return next(op, inputs, attrs);
    });

    auto a = ones({4}, DType::Float32, device);
    auto b = ones({4}, DType::Float32, device);
    tenzor::add(a, b);
    tenzor::mul(a, b);

    EXPECT_GE(count.load(), 2) << "Interceptor should be called for each dispatch";
}

TEST_P(DispatchInterceptorMultiBackendTest, StackedInterceptors) {
    // Two stacked interceptors — both should run in order
    std::vector<int> order;

    InterceptorGuard guard1([&order](OpId op, std::span<const Tensor> inputs,
                                      const OpAttributes& attrs, DispatchNext next) {
        order.push_back(1);
        return next(op, inputs, attrs);
    });

    InterceptorGuard guard2([&order](OpId op, std::span<const Tensor> inputs,
                                      const OpAttributes& attrs, DispatchNext next) {
        order.push_back(2);
        return next(op, inputs, attrs);
    });

    EXPECT_EQ(DispatchInterceptorStack::depth(), 2u);

    auto a = ones({4}, DType::Float32, device);
    tenzor::add(a, a);

    // First pushed runs first (FIFO order)
    ASSERT_GE(order.size(), 2u);
    EXPECT_EQ(order[0], 1);
    EXPECT_EQ(order[1], 2);
}

TEST_P(DispatchInterceptorMultiBackendTest, RAIIGuardCleanup) {
    {
        InterceptorGuard guard([](OpId op, std::span<const Tensor> inputs,
                                  const OpAttributes& attrs, DispatchNext next) {
            return next(op, inputs, attrs);
        });
        EXPECT_EQ(DispatchInterceptorStack::depth(), 1u);
    }
    // After guard destruction, stack should be empty
    EXPECT_EQ(DispatchInterceptorStack::depth(), 0u);
}

TEST_P(DispatchInterceptorMultiBackendTest, SingleOutputDispatch) {
    std::atomic<int> count{0};

    InterceptorGuard guard([&count](OpId op, std::span<const Tensor> inputs,
                                     const OpAttributes& attrs, DispatchNext next) {
        count.fetch_add(1, std::memory_order_relaxed);
        return next(op, inputs, attrs);
    });

    // dispatch_single path also goes through interceptors
    auto a = ones({4, 4}, DType::Float32, device);
    auto b = ones({4, 4}, DType::Float32, device);
    auto c = tenzor::matmul(a, b);

    EXPECT_GE(count.load(), 1) << "Single-output dispatch should also hit interceptors";
    auto c_cpu = c.to(Device::cpu());
    EXPECT_NEAR(c_cpu.data<float>()[0], 4.0f, 1e-5f);  // 4x4 ones matmul -> 4.0
}

INSTANTIATE_BACKEND_TESTS(DispatchInterceptorMultiBackendTest);
